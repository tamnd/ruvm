// SPDX-License-Identifier: MIT OR Apache-2.0

//! Transaction attributes and results, include/exec/memattrs.h.

use std::fmt;
use std::ops::{BitOr, BitOrAssign};

/// The attributes that travel with every memory transaction, `MemTxAttrs`.
///
/// The bits sit where QEMU's bitfields put them, so the raw value can be compared with what a QEMU
/// build produces: `secure` is bit 0, `space` bits 1 and 2, `user` bit 3, `memory` bit 4, `debug`
/// bit 5, `requester_id` bits 6 to 21, `pid` bits 22 to 29, `address_type` bit 30 and
/// `unspecified` bit 32.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MemTxAttrs(u64);

const SECURE: u32 = 0;
const SPACE: u32 = 1;
const USER: u32 = 3;
const MEMORY: u32 = 4;
const DEBUG: u32 = 5;
const REQUESTER_ID: u32 = 6;
const PID: u32 = 22;
const ADDRESS_TYPE: u32 = 30;
const UNSPECIFIED: u32 = 32;

impl MemTxAttrs {
    /// `MEMTXATTRS_UNSPECIFIED`: only the `unspecified` flag is set.
    pub const UNSPECIFIED: MemTxAttrs = MemTxAttrs(1 << UNSPECIFIED);

    /// The attributes with every field zero.
    pub const fn new() -> Self {
        MemTxAttrs(0)
    }

    /// The raw bits.
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Attributes from raw bits, as returned by [`MemTxAttrs::bits`].
    pub const fn from_bits(bits: u64) -> Self {
        MemTxAttrs(bits)
    }

    const fn field(self, shift: u32, width: u32) -> u64 {
        (self.0 >> shift) & ((1 << width) - 1)
    }

    const fn with_field(self, shift: u32, width: u32, value: u64) -> Self {
        let mask = ((1u64 << width) - 1) << shift;
        MemTxAttrs((self.0 & !mask) | ((value << shift) & mask))
    }

    /// The access comes from the secure world (Arm TrustZone).
    pub const fn secure(self) -> bool {
        self.field(SECURE, 1) != 0
    }

    /// Sets [`MemTxAttrs::secure`].
    pub const fn with_secure(self, v: bool) -> Self {
        self.with_field(SECURE, 1, v as u64)
    }

    /// The Arm security space, `ARMSecuritySpace`.
    pub const fn space(self) -> u8 {
        self.field(SPACE, 2) as u8
    }

    /// Sets [`MemTxAttrs::space`]. Only the low two bits are kept.
    pub const fn with_space(self, v: u8) -> Self {
        self.with_field(SPACE, 2, v as u64)
    }

    /// The access is unprivileged.
    pub const fn user(self) -> bool {
        self.field(USER, 1) != 0
    }

    /// Sets [`MemTxAttrs::user`].
    pub const fn with_user(self, v: bool) -> Self {
        self.with_field(USER, 1, v as u64)
    }

    /// The access must only reach normal memory, never a device.
    pub const fn memory(self) -> bool {
        self.field(MEMORY, 1) != 0
    }

    /// Sets [`MemTxAttrs::memory`].
    pub const fn with_memory(self, v: bool) -> Self {
        self.with_field(MEMORY, 1, v as u64)
    }

    /// A debugger or loader access. Debug writes go through to ROM.
    pub const fn debug(self) -> bool {
        self.field(DEBUG, 1) != 0
    }

    /// Sets [`MemTxAttrs::debug`].
    pub const fn with_debug(self, v: bool) -> Self {
        self.with_field(DEBUG, 1, v as u64)
    }

    /// The bus specific id of the requester, the PCI requester id on PCI.
    pub const fn requester_id(self) -> u16 {
        self.field(REQUESTER_ID, 16) as u16
    }

    /// Sets [`MemTxAttrs::requester_id`].
    pub const fn with_requester_id(self, v: u16) -> Self {
        self.with_field(REQUESTER_ID, 16, v as u64)
    }

    /// The PCI PASID.
    pub const fn pid(self) -> u8 {
        self.field(PID, 8) as u8
    }

    /// Sets [`MemTxAttrs::pid`].
    pub const fn with_pid(self, v: u8) -> Self {
        self.with_field(PID, 8, v as u64)
    }

    /// The PCI address type, set for translated addresses.
    pub const fn address_type(self) -> bool {
        self.field(ADDRESS_TYPE, 1) != 0
    }

    /// Sets [`MemTxAttrs::address_type`].
    pub const fn with_address_type(self, v: bool) -> Self {
        self.with_field(ADDRESS_TYPE, 1, v as u64)
    }

    /// The caller did not say. Devices that care treat this as the default for their bus.
    pub const fn unspecified(self) -> bool {
        self.field(UNSPECIFIED, 1) != 0
    }
}

impl fmt::Debug for MemTxAttrs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.unspecified() {
            return f.write_str("MemTxAttrs(unspecified)");
        }
        f.debug_struct("MemTxAttrs")
            .field("secure", &self.secure())
            .field("space", &self.space())
            .field("user", &self.user())
            .field("memory", &self.memory())
            .field("debug", &self.debug())
            .field("requester_id", &self.requester_id())
            .field("pid", &self.pid())
            .field("address_type", &self.address_type())
            .finish()
    }
}

/// The outcome of a transaction, a bit set like `MemTxResult`. Results of the pieces of a split
/// access are OR-ed together.
#[derive(Copy, Clone, Default, PartialEq, Eq, Hash, Debug)]
pub struct MemTxResult(u32);

impl MemTxResult {
    /// `MEMTX_OK`.
    pub const OK: MemTxResult = MemTxResult(0);
    /// `MEMTX_ERROR`: the device returned an error.
    pub const ERROR: MemTxResult = MemTxResult(1 << 0);
    /// `MEMTX_DECODE_ERROR`: nothing answered at that address, or the access was rejected.
    pub const DECODE_ERROR: MemTxResult = MemTxResult(1 << 1);
    /// `MEMTX_ACCESS_ERROR`: the access was denied.
    pub const ACCESS_ERROR: MemTxResult = MemTxResult(1 << 2);

    /// No error bit is set.
    pub const fn is_ok(self) -> bool {
        self.0 == 0
    }

    /// Every bit of `other` is set in `self`.
    pub const fn contains(self, other: MemTxResult) -> bool {
        self.0 & other.0 == other.0
    }

    /// The raw bits.
    pub const fn bits(self) -> u32 {
        self.0
    }
}

impl BitOr for MemTxResult {
    type Output = MemTxResult;

    fn bitor(self, rhs: MemTxResult) -> MemTxResult {
        MemTxResult(self.0 | rhs.0)
    }
}

impl BitOrAssign for MemTxResult {
    fn bitor_assign(&mut self, rhs: MemTxResult) {
        self.0 |= rhs.0;
    }
}

/// `MEMTX_OK`.
pub const MEMTX_OK: MemTxResult = MemTxResult::OK;
/// `MEMTX_ERROR`.
pub const MEMTX_ERROR: MemTxResult = MemTxResult::ERROR;
/// `MEMTX_DECODE_ERROR`.
pub const MEMTX_DECODE_ERROR: MemTxResult = MemTxResult::DECODE_ERROR;
/// `MEMTX_ACCESS_ERROR`.
pub const MEMTX_ACCESS_ERROR: MemTxResult = MemTxResult::ACCESS_ERROR;

/// What a device callback returns: a value, or the error bits to report.
pub type MemResult<T> = Result<T, MemTxResult>;

/// What a device callback is told about the access besides the address and size.
///
/// The lock domain guard from spec/03 joins this once device domains exist.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AccessCtx {
    /// The transaction attributes.
    pub attrs: MemTxAttrs,
}

impl AccessCtx {
    /// A context carrying `attrs`.
    pub const fn new(attrs: MemTxAttrs) -> Self {
        AccessCtx { attrs }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fields_sit_where_qemu_puts_them() {
        assert_eq!(MemTxAttrs::new().with_secure(true).bits(), 1);
        assert_eq!(MemTxAttrs::new().with_space(3).bits(), 0b110);
        assert_eq!(MemTxAttrs::new().with_debug(true).bits(), 1 << 5);
        assert_eq!(MemTxAttrs::new().with_requester_id(0xffff).bits(), 0xffff << 6);
        assert_eq!(MemTxAttrs::new().with_pid(0xff).bits(), 0xff << 22);
        assert_eq!(MemTxAttrs::new().with_address_type(true).bits(), 1 << 30);
        assert_eq!(MemTxAttrs::UNSPECIFIED.bits(), 1 << 32);
        let a = MemTxAttrs::new().with_requester_id(0x1234).with_user(true);
        assert_eq!(a.requester_id(), 0x1234);
        assert!(a.user() && !a.secure() && !a.unspecified());
    }

    #[test]
    fn results_accumulate() {
        let mut r = MEMTX_OK;
        assert!(r.is_ok());
        r |= MEMTX_DECODE_ERROR;
        r |= MEMTX_ERROR;
        assert!(r.contains(MEMTX_DECODE_ERROR) && r.contains(MEMTX_ERROR));
        assert!(!r.contains(MEMTX_ACCESS_ERROR));
    }
}
