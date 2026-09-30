// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device callbacks and the rules that turn one guest access into the calls a device sees.
//!
//! Three steps, each with its own set of constraints. [`memory_access_size`] cuts a buffer access
//! into pieces no wider than the device accepts and no wider than the address alignment allows.
//! [`access_valid`] rejects a piece the device does not accept at all. The split in
//! `access_with_adjusted_size` then widens or narrows what is left to what the callback
//! implements. The rules are those of `memory_access_size()` in system/physmem.c and of
//! `memory_region_access_valid()` and `access_with_adjusted_size()` in system/memory.c, and the
//! addresses a device sees are the same, including the unaligned widened access QEMU makes.

use std::fmt;
use std::sync::{Arc, RwLock};

use crate::attrs::{AccessCtx, MemResult, MemTxAttrs, MemTxResult};

/// Byte order of a device's registers, `enum device_endian` with the native case resolved by
/// whoever creates the region.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Endian {
    /// `DEVICE_LITTLE_ENDIAN`.
    #[default]
    Little,
    /// `DEVICE_BIG_ENDIAN`.
    Big,
}

impl Endian {
    /// The byte order of the host.
    pub const HOST: Endian = if cfg!(target_endian = "big") { Endian::Big } else { Endian::Little };
}

/// The width of one device callback in bytes: 1, 2, 4 or 8.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AccessSize(u8);

impl AccessSize {
    /// One byte.
    pub const B1: AccessSize = AccessSize(1);
    /// Two bytes.
    pub const B2: AccessSize = AccessSize(2);
    /// Four bytes.
    pub const B4: AccessSize = AccessSize(4);
    /// Eight bytes.
    pub const B8: AccessSize = AccessSize(8);

    /// The size for `bytes`, if it is 1, 2, 4 or 8.
    pub const fn new(bytes: u32) -> Option<Self> {
        match bytes {
            1 | 2 | 4 | 8 => Some(AccessSize(bytes as u8)),
            _ => None,
        }
    }

    /// The width in bytes.
    pub const fn bytes(self) -> u32 {
        self.0 as u32
    }

    /// The width in bits.
    pub const fn bits(self) -> u32 {
        self.0 as u32 * 8
    }

    /// A mask of the low [`AccessSize::bits`] bits.
    pub const fn mask(self) -> u64 {
        mask_bytes(self.0 as u32)
    }
}

const fn mask_bytes(bytes: u32) -> u64 {
    if bytes >= 8 { u64::MAX } else { (1u64 << (bytes * 8)) - 1 }
}

/// One of the two constraint sets in `MemoryRegionOps`, `valid` or `impl`.
///
/// Zero sizes mean what they mean in QEMU. For `valid`, a zero `max` accepts every size, and for
/// `impl`, a zero `min` is 1 and a zero `max` is 4. The default is all zero, which is what a C
/// device that leaves the block out gets.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AccessConstraints {
    /// `min_access_size`.
    pub min: u32,
    /// `max_access_size`.
    pub max: u32,
    /// `unaligned`: accesses need not be aligned to their size.
    pub unaligned: bool,
}

impl AccessConstraints {
    /// Sizes from `min` to `max`, aligned.
    pub const fn any_size(min: u32, max: u32) -> Self {
        AccessConstraints { min, max, unaligned: false }
    }

    /// Exactly `size`, aligned.
    pub const fn exact(size: u32) -> Self {
        AccessConstraints { min: size, max: size, unaligned: false }
    }

    /// The same sizes, with unaligned accesses allowed.
    pub const fn allow_unaligned(self) -> Self {
        AccessConstraints { unaligned: true, ..self }
    }
}

/// The callbacks of an MMIO region, `MemoryRegionOps`.
///
/// `offset` is relative to the start of the region. A read's value is in the device's byte order
/// ([`MmioOps::endianness`]); the core swaps it for the requester.
pub trait MmioOps: Send + Sync {
    /// Reads `size` bytes at `offset`.
    fn read(&self, cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64>;

    /// Writes the low `size` bytes of `value` at `offset`.
    fn write(&self, cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()>;

    /// What the device accepts, `ops->valid`. Anything else is rejected with
    /// `MEMTX_DECODE_ERROR` before the device sees it.
    fn valid(&self) -> AccessConstraints {
        AccessConstraints::default()
    }

    /// What the callbacks implement, `ops->impl`. Other sizes are split or widened by the core.
    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    /// `ops->valid.accepts`: a last say on each access.
    fn accepts(&self, offset: u64, size: u32, is_write: bool, attrs: MemTxAttrs) -> bool {
        let _ = (offset, size, is_write, attrs);
        true
    }

    /// The byte order of the registers.
    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

/// `unassigned_mem_ops`: accepts nothing, so every access is a decode error that reads as 0.
/// Holes in an address space, reservations and writes to ROM all end up here.
#[derive(Debug, Default)]
pub(crate) struct Unassigned;

impl MmioOps for Unassigned {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, _v: u64) -> MemResult<()> {
        Ok(())
    }

    fn accepts(&self, _offset: u64, _size: u32, _is_write: bool, _attrs: MemTxAttrs) -> bool {
        false
    }
}

/// Where guest error messages go, see [`set_guest_error_log`].
pub type GuestErrorSink = Arc<dyn Fn(&str) + Send + Sync>;

static GUEST_ERRORS: RwLock<Option<GuestErrorSink>> = RwLock::new(None);

/// Routes the messages QEMU logs under `-d guest_errors` (`LOG_GUEST_ERROR`, which
/// `LOG_INVALID_MEM` is part of) to `sink`, or drops them if `sink` is `None`. The text of each
/// message is QEMU's, newline included.
pub fn set_guest_error_log(sink: Option<GuestErrorSink>) {
    *GUEST_ERRORS.write().unwrap_or_else(|p| p.into_inner()) = sink;
}

pub(crate) fn guest_error(msg: fmt::Arguments<'_>) {
    let sink = GUEST_ERRORS.read().unwrap_or_else(|p| p.into_inner()).clone();
    if let Some(sink) = sink {
        sink(&msg.to_string());
    }
}

fn rw(is_write: bool) -> &'static str {
    if is_write { "write" } else { "read" }
}

/// `memory_region_access_valid()`: whether the device accepts a `size` byte access at `offset`.
/// A rejection is logged as a guest error with QEMU's message.
pub fn access_valid(
    ops: &dyn MmioOps,
    name: &str,
    offset: u64,
    size: u32,
    is_write: bool,
    attrs: MemTxAttrs,
) -> bool {
    let valid = ops.valid();
    let what = rw(is_write);
    if !ops.accepts(offset, size, is_write, attrs) {
        guest_error(format_args!(
            "Invalid {what} at addr 0x{offset:X}, size {size}, region '{name}', reason: rejected\n"
        ));
        return false;
    }
    if !valid.unaligned && offset & u64::from(size.wrapping_sub(1)) != 0 {
        guest_error(format_args!(
            "Invalid {what} at addr 0x{offset:X}, size {size}, region '{name}', reason: unaligned\n"
        ));
        return false;
    }
    if valid.max == 0 {
        return true;
    }
    if size > valid.max || size < valid.min {
        guest_error(format_args!(
            "Invalid {what} at addr 0x{offset:X}, size {size}, region '{name}', reason: invalid \
             size (min:{} max:{})\n",
            valid.min, valid.max
        ));
        return false;
    }
    true
}

/// `memory_access_size()`: how many of the `len` bytes left at `offset` go to the device in one
/// call. The answer is at most `valid.max` (4 if that is zero), at most the alignment of `offset`
/// unless `impl.unaligned` is set, and a power of two.
pub fn memory_access_size(ops: &dyn MmioOps, len: u64, offset: u64) -> u32 {
    let mut max = u64::from(ops.valid().max);
    if max == 0 {
        max = 4;
    }
    if !ops.impl_constraints().unaligned {
        let align = offset & offset.wrapping_neg();
        if align != 0 && align < max {
            max = align;
        }
    }
    let l = len.min(max);
    if l == 0 {
        return 0;
    }
    // pow2floor
    (1u64 << (63 - l.leading_zeros())) as u32
}

fn impl_sizes(ops: &dyn MmioOps) -> (u32, u32) {
    let imp = ops.impl_constraints();
    let min = if imp.min == 0 { 1 } else { imp.min };
    let max = if imp.max == 0 { 4 } else { imp.max };
    (min.min(8), max.min(8))
}

fn shift_in(value: &mut u64, shift: i32, tmp: u64) {
    if shift >= 0 {
        *value |= tmp << shift;
    } else {
        *value |= tmp >> -shift;
    }
}

fn shift_out(value: u64, shift: i32) -> u64 {
    if shift >= 0 { value >> shift } else { value << -shift }
}

/// The piece offsets and shifts `access_with_adjusted_size()` uses.
fn pieces(size: u32, access: u32, endian: Endian) -> impl Iterator<Item = (u32, i32)> {
    (0..size).step_by(access as usize).map(move |i| {
        let shift = match endian {
            Endian::Big => (size as i32 - access as i32 - i as i32) * 8,
            Endian::Little => i as i32 * 8,
        };
        (i, shift)
    })
}

/// Reads through `access_with_adjusted_size()`. `size` must be 1, 2, 4 or 8, and the value comes
/// back in the device's byte order, masked to `size` bytes.
fn read_adjusted(
    ops: &dyn MmioOps,
    offset: u64,
    size: u32,
    attrs: MemTxAttrs,
) -> (u64, MemTxResult) {
    let (min, max) = impl_sizes(ops);
    let access = size.min(max).max(min);
    let access_size = AccessSize::new(access).unwrap_or(AccessSize::B4);
    let cx = AccessCtx::new(attrs);
    let mut value = 0;
    let mut r = MemTxResult::OK;
    for (i, shift) in pieces(size, access, ops.endianness()) {
        let (tmp, res) = match ops.read(&cx, offset.wrapping_add(u64::from(i)), access_size) {
            Ok(v) => (v, MemTxResult::OK),
            Err(e) => (0, e),
        };
        shift_in(&mut value, shift, tmp & mask_bytes(access));
        r |= res;
    }
    (value & mask_bytes(size), r)
}

fn write_adjusted(
    ops: &dyn MmioOps,
    offset: u64,
    size: u32,
    value: u64,
    attrs: MemTxAttrs,
) -> MemTxResult {
    let (min, max) = impl_sizes(ops);
    let access = size.min(max).max(min);
    let access_size = AccessSize::new(access).unwrap_or(AccessSize::B4);
    let cx = AccessCtx::new(attrs);
    let mut r = MemTxResult::OK;
    for (i, shift) in pieces(size, access, ops.endianness()) {
        let tmp = shift_out(value, shift) & mask_bytes(access);
        if let Err(e) = ops.write(&cx, offset.wrapping_add(u64::from(i)), access_size, tmp) {
            r |= e;
        }
    }
    r
}

/// `memory_region_dispatch_read()`: validates, then reads `size` bytes (1, 2, 4 or 8) at `offset`
/// and returns the value in `endian` byte order. A rejected read returns 0 with
/// `MEMTX_DECODE_ERROR`.
pub fn dispatch_read(
    ops: &dyn MmioOps,
    name: &str,
    offset: u64,
    size: u32,
    endian: Endian,
    attrs: MemTxAttrs,
) -> (u64, MemTxResult) {
    if !access_valid(ops, name, offset, size, false, attrs) {
        return (0, MemTxResult::DECODE_ERROR);
    }
    let (v, r) = read_adjusted(ops, offset, size, attrs);
    (swap_if(v, size, endian != ops.endianness()), r)
}

/// `memory_region_dispatch_write()`: validates, then writes the low `size` bytes of `value`, which
/// is in `endian` byte order, at `offset`. A rejected write is dropped with `MEMTX_DECODE_ERROR`.
pub fn dispatch_write(
    ops: &dyn MmioOps,
    name: &str,
    offset: u64,
    size: u32,
    value: u64,
    endian: Endian,
    attrs: MemTxAttrs,
) -> MemTxResult {
    if !access_valid(ops, name, offset, size, true, attrs) {
        return MemTxResult::DECODE_ERROR;
    }
    let value = swap_if(value & mask_bytes(size), size, endian != ops.endianness());
    write_adjusted(ops, offset, size, value, attrs)
}

fn swap_if(v: u64, size: u32, swap: bool) -> u64 {
    if !swap {
        return v;
    }
    match size {
        2 => u64::from((v as u16).swap_bytes()),
        4 => u64::from((v as u32).swap_bytes()),
        8 => v.swap_bytes(),
        _ => v,
    }
}

/// Encodes the low `size` bytes of `v` in `endian` order into `out`.
pub(crate) fn store_bytes(out: &mut [u8], v: u64, endian: Endian) {
    let n = out.len();
    match endian {
        Endian::Little => out.copy_from_slice(&v.to_le_bytes()[..n]),
        Endian::Big => out.copy_from_slice(&v.to_be_bytes()[8 - n..]),
    }
}

/// Decodes `bytes` in `endian` order.
pub(crate) fn load_bytes(bytes: &[u8], endian: Endian) -> u64 {
    let n = bytes.len();
    let mut b = [0u8; 8];
    match endian {
        Endian::Little => {
            b[..n].copy_from_slice(bytes);
            u64::from_le_bytes(b)
        }
        Endian::Big => {
            b[8 - n..].copy_from_slice(bytes);
            u64::from_be_bytes(b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// A device that records its calls and reads back the bytes of a counter pattern.
    struct Recorder {
        valid: AccessConstraints,
        imp: AccessConstraints,
        endian: Endian,
        log: Mutex<Vec<(char, u64, u32, u64)>>,
    }

    impl Recorder {
        fn new(valid: AccessConstraints, imp: AccessConstraints, endian: Endian) -> Self {
            Recorder { valid, imp, endian, log: Mutex::new(Vec::new()) }
        }
        fn take(&self) -> Vec<(char, u64, u32, u64)> {
            std::mem::take(&mut self.log.lock().unwrap())
        }
    }

    impl MmioOps for Recorder {
        fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
            // Byte k of the register file holds 0x10 + k, laid out in the device's order.
            let bytes: Vec<u8> =
                (0..size.bytes() as u64).map(|k| 0x10 + (offset + k) as u8).collect();
            let v = load_bytes(&bytes, self.endian);
            self.log.lock().unwrap().push(('r', offset, size.bytes(), v));
            Ok(v)
        }
        fn write(
            &self,
            _cx: &AccessCtx,
            offset: u64,
            size: AccessSize,
            value: u64,
        ) -> MemResult<()> {
            self.log.lock().unwrap().push(('w', offset, size.bytes(), value));
            Ok(())
        }
        fn valid(&self) -> AccessConstraints {
            self.valid
        }
        fn impl_constraints(&self) -> AccessConstraints {
            self.imp
        }
        fn endianness(&self) -> Endian {
            self.endian
        }
    }

    const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

    #[test]
    fn narrow_implementation_is_split_little_endian() {
        let d = Recorder::new(
            AccessConstraints::any_size(1, 8),
            AccessConstraints::any_size(1, 2),
            Endian::Little,
        );
        let (v, r) = dispatch_read(&d, "d", 4, 4, Endian::Little, U);
        assert!(r.is_ok());
        assert_eq!(v, 0x1716_1514);
        assert_eq!(d.take(), vec![('r', 4, 2, 0x1514), ('r', 6, 2, 0x1716)]);
        dispatch_write(&d, "d", 0, 4, 0xaabb_ccdd, Endian::Little, U);
        assert_eq!(d.take(), vec![('w', 0, 2, 0xccdd), ('w', 2, 2, 0xaabb)]);
    }

    #[test]
    fn narrow_implementation_is_split_big_endian() {
        let d = Recorder::new(
            AccessConstraints::any_size(1, 8),
            AccessConstraints::any_size(1, 2),
            Endian::Big,
        );
        let (v, _) = dispatch_read(&d, "d", 4, 4, Endian::Big, U);
        assert_eq!(v, 0x1415_1617);
        assert_eq!(d.take(), vec![('r', 4, 2, 0x1415), ('r', 6, 2, 0x1617)]);
        dispatch_write(&d, "d", 0, 4, 0xaabb_ccdd, Endian::Big, U);
        assert_eq!(d.take(), vec![('w', 0, 2, 0xaabb), ('w', 2, 2, 0xccdd)]);
    }

    #[test]
    fn wide_implementation_is_not_aligned_down() {
        let d = Recorder::new(
            AccessConstraints::any_size(1, 4).allow_unaligned(),
            AccessConstraints::exact(4),
            Endian::Little,
        );
        let (v, _) = dispatch_read(&d, "d", 3, 1, Endian::Little, U);
        assert_eq!(d.take(), vec![('r', 3, 4, 0x1615_1413)]);
        assert_eq!(v, 0x13);
        let d = Recorder::new(
            AccessConstraints::any_size(1, 4).allow_unaligned(),
            AccessConstraints::exact(4),
            Endian::Big,
        );
        let (v, _) = dispatch_read(&d, "d", 3, 1, Endian::Big, U);
        assert_eq!(d.take(), vec![('r', 3, 4, 0x1314_1516)]);
        assert_eq!(v, 0x13);
    }

    #[test]
    fn requester_byte_order_is_applied() {
        let d = Recorder::new(
            AccessConstraints::default(),
            AccessConstraints::default(),
            Endian::Little,
        );
        let (v, _) = dispatch_read(&d, "d", 0, 4, Endian::Big, U);
        assert_eq!(v, 0x1011_1213);
        dispatch_write(&d, "d", 0, 2, 0x1234, Endian::Big, U);
        assert_eq!(d.take().pop(), Some(('w', 0, 2, 0x3412)));
    }

    #[test]
    fn validation_follows_the_valid_block() {
        let d = Recorder::new(
            AccessConstraints::any_size(2, 4),
            AccessConstraints::default(),
            Endian::Little,
        );
        assert!(!access_valid(&d, "d", 0, 1, false, U));
        assert!(!access_valid(&d, "d", 0, 8, false, U));
        assert!(!access_valid(&d, "d", 2, 4, false, U));
        assert!(access_valid(&d, "d", 4, 4, false, U));
        let all = Recorder::new(
            AccessConstraints::default(),
            AccessConstraints::default(),
            Endian::Little,
        );
        assert!(access_valid(&all, "d", 8, 8, false, U));
        assert!(!access_valid(&all, "d", 1, 2, false, U));
        assert!(!access_valid(&Unassigned, "d", 0, 1, false, U));
        let (v, r) = dispatch_read(&d, "d", 0, 1, Endian::Little, U);
        assert_eq!((v, r), (0, MemTxResult::DECODE_ERROR));
        assert!(d.take().is_empty());
    }

    #[test]
    fn access_size_follows_alignment_and_valid_max() {
        let d = Recorder::new(
            AccessConstraints::any_size(1, 8),
            AccessConstraints::default(),
            Endian::Little,
        );
        assert_eq!(memory_access_size(&d, 8, 0), 8);
        assert_eq!(memory_access_size(&d, 8, 4), 4);
        assert_eq!(memory_access_size(&d, 8, 6), 2);
        assert_eq!(memory_access_size(&d, 7, 0), 4);
        assert_eq!(memory_access_size(&d, 3, 0), 2);
        let compat = Recorder::new(
            AccessConstraints::default(),
            AccessConstraints::default(),
            Endian::Little,
        );
        assert_eq!(memory_access_size(&compat, 8, 0), 4);
        let unaligned = Recorder::new(
            AccessConstraints::any_size(1, 8),
            AccessConstraints::default().allow_unaligned(),
            Endian::Little,
        );
        assert_eq!(memory_access_size(&unaligned, 8, 3), 8);
    }

    #[test]
    fn rejections_log_qemu_text() {
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let s = Arc::clone(&seen);
        set_guest_error_log(Some(Arc::new(move |m: &str| s.lock().unwrap().push(m.to_string()))));
        let d = Recorder::new(
            AccessConstraints::any_size(2, 4),
            AccessConstraints::default(),
            Endian::Little,
        );
        access_valid(&d, "uart", 0x10, 1, true, U);
        access_valid(&d, "uart", 0x11, 2, false, U);
        access_valid(&Unassigned, "pc.bios", 0x20, 4, true, U);
        set_guest_error_log(None);
        let seen = seen.lock().unwrap();
        assert!(seen.contains(
            &"Invalid write at addr 0x10, size 1, region 'uart', reason: invalid size (min:2 max:4)\n".to_string()
        ));
        assert!(seen.contains(
            &"Invalid read at addr 0x11, size 2, region 'uart', reason: unaligned\n".to_string()
        ));
        assert!(seen.contains(
            &"Invalid write at addr 0x20, size 4, region 'pc.bios', reason: rejected\n".to_string()
        ));
    }
}
