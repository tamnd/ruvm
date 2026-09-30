// SPDX-License-Identifier: MIT OR Apache-2.0

//! Address spaces and the accesses made through them.
//!
//! An address space publishes its current [`FlatView`] through an `ArcSwap`. A reader loads the
//! pointer without taking a lock, keeps the view alive for as long as it holds it, and never
//! waits for a writer; a topology change builds a whole new view and swaps it in. This is the
//! RCU publication of spec/05 with reference counts standing in for grace periods.
//!
//! The access paths follow `flatview_read()`, `flatview_write()` and the `address_space_ld*` and
//! `address_space_st*` families in system/physmem.c, which decide what a guest sees when an access
//! crosses regions, hits a hole or lands on ROM.

use std::fmt;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::access::{
    Endian, MmioOps, Unassigned, dispatch_read, dispatch_write, guest_error, load_bytes,
    memory_access_size, store_bytes,
};
use crate::attrs::{MemTxAttrs, MemTxResult};
use crate::flatview::{FlatRange, FlatView};
use crate::iommu::IommuAccessFlags;
use crate::region::{RegionId, RegionType, TargetKind};

/// How many IOMMUs one access may pass through before it is treated as unassigned.
const MAX_IOMMU_DEPTH: u32 = 16;

/// The name QEMU gives the region behind holes, `io_mem_unassigned`.
const UNASSIGNED_NAME: &str = "unassigned";

/// What one step of an access resolved to.
struct Resolved<'a> {
    /// The range, or `None` for a hole or a failed translation.
    range: Option<&'a FlatRange>,
    /// The offset into the range's region, or the address for a hole.
    offset: u64,
    /// How many bytes of the access this step may cover.
    len: u64,
}

/// A view of memory as seen by one kind of requester, `AddressSpace`.
///
/// Created with [`MemorySystem::address_space_init`](crate::MemorySystem::address_space_init).
/// All methods here are lock free and may be called from any thread, including while the
/// topology is being changed.
pub struct AddressSpace {
    name: String,
    root: RegionId,
    view: ArcSwap<FlatView>,
}

impl AddressSpace {
    pub(crate) fn new(name: &str, root: RegionId, view: Arc<FlatView>) -> Self {
        AddressSpace { name: name.to_string(), root, view: ArcSwap::new(view) }
    }

    /// The name given at creation.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The root region.
    pub fn root(&self) -> RegionId {
        self.root
    }

    /// The current view, `address_space_get_flatview()`. The view stays valid after later
    /// topology changes; it is simply no longer current.
    pub fn flatview(&self) -> Arc<FlatView> {
        self.view.load_full()
    }

    pub(crate) fn publish(&self, view: Arc<FlatView>) {
        self.view.store(view);
    }

    /// Finds what the access at `addr` of up to `len` bytes goes to and calls `f` with it.
    /// IOMMU regions are translated on the way, `address_space_translate_iommu()`.
    fn resolve<R>(
        &self,
        addr: u64,
        len: u64,
        is_write: bool,
        attrs: MemTxAttrs,
        depth: u32,
        f: impl FnOnce(Resolved<'_>) -> R,
    ) -> R {
        let view = self.view.load();
        let (range, left) = view.translate(addr);
        let Some(range) = range else {
            // A hole is served by a region covering the whole space, so the step is not cut at
            // the next range. The access size rules keep it small anyway.
            return f(Resolved { range: None, offset: addr, len });
        };
        let len = u128::from(len).min(left) as u64;
        let offset = range.offset_in_region + (addr - range.addr);
        let TargetKind::Iommu(ops) = &range.target.kind else {
            return f(Resolved { range: Some(range), offset, len });
        };
        let flag = IommuAccessFlags::for_access(is_write);
        let entry = ops.translate(offset, flag, ops.attrs_to_index(attrs));
        match entry.target_as {
            Some(target) if entry.perm.allows(flag) && depth < MAX_IOMMU_DEPTH => {
                let mask = entry.addr_mask;
                let next = (entry.translated_addr & !mask) | (offset & mask);
                let page_left = u128::from((next | mask) - next) + 1;
                let len = u128::from(len).min(page_left) as u64;
                target.resolve(next, len, is_write, attrs, depth + 1, f)
            }
            _ => f(Resolved { range: None, offset: addr, len }),
        }
    }

    /// Reads `buf.len()` bytes at `addr`, `address_space_read()`. Bytes the access could not
    /// read are left as the device or the hole returned them, usually zero, and the result
    /// collects every error seen along the way.
    pub fn read(&self, addr: u64, attrs: MemTxAttrs, buf: &mut [u8]) -> MemTxResult {
        let mut result = MemTxResult::OK;
        let mut done = 0;
        let mut addr = addr;
        while done < buf.len() {
            let rest = &mut buf[done..];
            let want = rest.len() as u64;
            let (l, r) =
                self.resolve(addr, want, false, attrs, 0, |res| read_step(res, attrs, rest));
            result |= r;
            done += l;
            addr = addr.wrapping_add(l as u64);
        }
        result
    }

    /// Writes `buf` at `addr`, `address_space_write()`. RAM written this way is marked dirty for
    /// every client logging it. Writes to ROM are dropped with `MEMTX_DECODE_ERROR`, except with
    /// the `debug` attribute, which writes through as a debugger would.
    pub fn write(&self, addr: u64, attrs: MemTxAttrs, buf: &[u8]) -> MemTxResult {
        let mut result = MemTxResult::OK;
        let mut done = 0;
        let mut addr = addr;
        while done < buf.len() {
            let rest = &buf[done..];
            let want = rest.len() as u64;
            let (l, r) =
                self.resolve(addr, want, true, attrs, 0, |res| write_step(res, attrs, rest));
            result |= r;
            done += l;
            addr = addr.wrapping_add(l as u64);
        }
        result
    }

    /// Loads a `size` byte value (1, 2, 4 or 8) at `addr` in `endian` byte order, the
    /// `address_space_ldl_le()` family. Unlike [`AddressSpace::read`] the access is never
    /// split: if it runs past the end of a RAM range it goes to the region's callbacks whole,
    /// and a device sees one access of the full size.
    pub fn load(
        &self,
        addr: u64,
        size: u32,
        endian: Endian,
        attrs: MemTxAttrs,
    ) -> (u64, MemTxResult) {
        self.resolve(addr, u64::from(size), false, attrs, 0, |res| {
            if let Some(range) = res.range {
                if res.len >= u64::from(size) && is_direct(range, false, attrs) {
                    let mut bytes = [0u8; 8];
                    let bytes = &mut bytes[..size as usize];
                    return match range.ram_block().map(|b| b.read(res.offset, bytes)) {
                        Some(Ok(())) => (load_bytes(bytes, endian), MemTxResult::OK),
                        _ => (0, MemTxResult::ERROR),
                    };
                }
            }
            let (ops, name) = slow_path(res.range);
            dispatch_read(ops, name, res.offset, size, endian, attrs)
        })
    }

    /// Stores the low `size` bytes of `value` (1, 2, 4 or 8) at `addr` in `endian` byte order,
    /// the `address_space_stl_le()` family. Splitting follows [`AddressSpace::load`].
    pub fn store(
        &self,
        addr: u64,
        size: u32,
        value: u64,
        endian: Endian,
        attrs: MemTxAttrs,
    ) -> MemTxResult {
        self.resolve(addr, u64::from(size), true, attrs, 0, |res| {
            if let Some(range) = res.range {
                if res.len >= u64::from(size) && is_direct(range, true, attrs) {
                    let mut bytes = [0u8; 8];
                    let bytes = &mut bytes[..size as usize];
                    store_bytes(bytes, value, endian);
                    return write_ram(range, res.offset, bytes);
                }
            }
            let (ops, name) = slow_path(res.range);
            dispatch_write(ops, name, res.offset, size, value, endian, attrs)
        })
    }

    /// `address_space_read()` of a little endian 32 bit value, for tests and simple callers.
    pub fn read_u32(&self, addr: u64, attrs: MemTxAttrs) -> (u32, MemTxResult) {
        let mut b = [0u8; 4];
        let r = self.read(addr, attrs, &mut b);
        (u32::from_le_bytes(b), r)
    }

    /// `address_space_write()` of a little endian 32 bit value.
    pub fn write_u32(&self, addr: u64, attrs: MemTxAttrs, value: u32) -> MemTxResult {
        self.write(addr, attrs, &value.to_le_bytes())
    }
}

impl fmt::Debug for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AddressSpace")
            .field("name", &self.name)
            .field("root", &self.root)
            .field("ranges", &self.view.load().len())
            .finish()
    }
}

/// `memory_access_is_direct()`: whether the access can copy bytes instead of calling a device.
fn is_direct(range: &FlatRange, is_write: bool, attrs: MemTxAttrs) -> bool {
    let writable = !is_write || attrs.debug();
    match range.target.ty {
        RegionType::Ram => writable || !range.region_readonly,
        RegionType::RomDevice => range.romd_mode && writable,
        _ => false,
    }
}

/// The callbacks and name used when an access is not direct. RAM and reservations have the
/// unassigned callbacks, as in QEMU, so a write to ROM is rejected with the ROM's name.
fn slow_path(range: Option<&FlatRange>) -> (&dyn MmioOps, &str) {
    static UNASSIGNED: Unassigned = Unassigned;
    match range {
        None => (&UNASSIGNED, UNASSIGNED_NAME),
        Some(r) => match &r.target.kind {
            TargetKind::Mmio(ops) | TargetKind::RomDevice(_, ops) => (ops.as_ref(), r.name()),
            _ => (&UNASSIGNED, r.name()),
        },
    }
}

/// `flatview_access_allowed()`: the `memory` attribute keeps an access away from devices.
fn access_allowed(range: Option<&FlatRange>, offset: u64, len: u64, attrs: MemTxAttrs) -> bool {
    if !attrs.memory() {
        return true;
    }
    if range.is_some_and(|r| r.target.ty == RegionType::Ram) {
        return true;
    }
    let name = range.map_or(UNASSIGNED_NAME, |r| r.name());
    guest_error(format_args!(
        "Invalid access to non-RAM device at addr 0x{offset:X}, size {len}, region '{name}'\n"
    ));
    false
}

fn write_ram(range: &FlatRange, offset: u64, bytes: &[u8]) -> MemTxResult {
    let Some(block) = range.ram_block() else { return MemTxResult::ERROR };
    match block.write(offset, bytes) {
        Ok(()) => {
            block.set_dirty(offset, bytes.len() as u64, range.dirty_log_mask);
            MemTxResult::OK
        }
        Err(_) => MemTxResult::ERROR,
    }
}

/// One step of `flatview_read_continue()`. Returns the bytes consumed and the result.
fn read_step(res: Resolved<'_>, attrs: MemTxAttrs, buf: &mut [u8]) -> (usize, MemTxResult) {
    let l = res.len.min(buf.len() as u64);
    if !access_allowed(res.range, res.offset, l, attrs) {
        return (l as usize, MemTxResult::ACCESS_ERROR);
    }
    if let Some(range) = res.range {
        if is_direct(range, false, attrs) {
            let out = &mut buf[..l as usize];
            return match range.ram_block().map(|b| b.read(res.offset, out)) {
                Some(Ok(())) => (out.len(), MemTxResult::OK),
                _ => (out.len(), MemTxResult::ERROR),
            };
        }
    }
    let (ops, name) = slow_path(res.range);
    let l = memory_access_size(ops, l, res.offset);
    let (v, r) = dispatch_read(ops, name, res.offset, l, Endian::HOST, attrs);
    store_bytes(&mut buf[..l as usize], v, Endian::HOST);
    (l as usize, r)
}

/// One step of `flatview_write_continue()`.
fn write_step(res: Resolved<'_>, attrs: MemTxAttrs, buf: &[u8]) -> (usize, MemTxResult) {
    let l = res.len.min(buf.len() as u64);
    if !access_allowed(res.range, res.offset, l, attrs) {
        return (l as usize, MemTxResult::ACCESS_ERROR);
    }
    if let Some(range) = res.range {
        if is_direct(range, true, attrs) {
            return (l as usize, write_ram(range, res.offset, &buf[..l as usize]));
        }
    }
    let (ops, name) = slow_path(res.range);
    let l = memory_access_size(ops, l, res.offset);
    let v = load_bytes(&buf[..l as usize], Endian::HOST);
    (l as usize, dispatch_write(ops, name, res.offset, l, v, Endian::HOST, attrs))
}
