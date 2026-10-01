// SPDX-License-Identifier: MIT OR Apache-2.0

//! Guest memory as seen by `qemu_ld` and `qemu_st`.
//!
//! The interpreter works out the byte order, sign extension and alignment from the [`MemOp`]
//! itself and only asks the memory for raw bytes, so an implementation is a plain byte store with
//! whatever translation and permission checks the machine needs.

use ruvm_jit_core::types::{MemOp, MemOpIdx};

/// Why a guest memory access failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaultKind {
    /// Nothing is mapped at the address.
    Unmapped,
    /// The access is not allowed.
    Protection,
    /// The address is not aligned as the [`MemOp`] requires.
    Unaligned,
}

/// A failed guest memory access. It leaves the translation block, as `cpu_loop_exit` does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemFault {
    /// The guest address of the access.
    pub addr: u64,
    /// The access was a store.
    pub write: bool,
    /// The memory operation and MMU index of the access.
    pub oi: MemOpIdx,
    /// What went wrong.
    pub kind: FaultKind,
}

/// The guest address space.
pub trait GuestMemory {
    /// Read `buf.len()` bytes starting at `addr`.
    fn read(&mut self, addr: u64, buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault>;

    /// Write `data` starting at `addr`.
    fn write(&mut self, addr: u64, data: &[u8], oi: MemOpIdx) -> Result<(), MemFault>;

    /// [`GuestMemory::read`] for an access made while running a block against `env`. A
    /// softmmu that has to walk guest page tables needs the CPU state to do it. The default
    /// ignores `env`.
    fn read_with_env(
        &mut self,
        env: &mut [u8],
        addr: u64,
        buf: &mut [u8],
        oi: MemOpIdx,
    ) -> Result<(), MemFault> {
        let _ = env;
        self.read(addr, buf, oi)
    }

    /// [`GuestMemory::write`] for an access made while running a block against `env`. The
    /// default ignores `env`.
    fn write_with_env(
        &mut self,
        env: &mut [u8],
        addr: u64,
        data: &[u8],
        oi: MemOpIdx,
    ) -> Result<(), MemFault> {
        let _ = env;
        self.write(addr, data, oi)
    }

    /// Called before the load and store of an atomic helper. A memory shared between threads
    /// takes a lock here so the pair is indivisible. The default does nothing.
    fn atomic_begin(&mut self) {}

    /// Called after an atomic helper is done with memory, whether it succeeded or not.
    fn atomic_end(&mut self) {}

    /// Called when an `insn_start` op runs, with its words. A runtime uses this to know which
    /// guest instruction a fault or helper call belongs to, as QEMU does by searching the
    /// block's unwind data with the host return address. The default does nothing.
    fn insn_start(&mut self, words: &[u64; ruvm_jit_core::types::INSN_START_WORDS]) {
        let _ = words;
    }

    /// The memory as [`Any`](std::any::Any), so helpers can reach the runtime behind it. The
    /// default is `None`.
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

/// A single flat region of guest memory starting at `base`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FlatMemory {
    /// The guest address of `bytes[0]`.
    pub base: u64,
    /// The contents.
    pub bytes: Vec<u8>,
}

impl FlatMemory {
    /// `size` zero bytes at guest address `base`.
    pub fn new(base: u64, size: usize) -> FlatMemory {
        FlatMemory { base, bytes: vec![0; size] }
    }

    fn range(&self, addr: u64, len: usize) -> Option<std::ops::Range<usize>> {
        let off = addr.checked_sub(self.base)?;
        let off = usize::try_from(off).ok()?;
        let end = off.checked_add(len)?;
        if end <= self.bytes.len() { Some(off..end) } else { None }
    }
}

impl GuestMemory for FlatMemory {
    fn read(&mut self, addr: u64, buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault> {
        match self.range(addr, buf.len()) {
            Some(r) => {
                buf.copy_from_slice(&self.bytes[r]);
                Ok(())
            }
            None => Err(MemFault { addr, write: false, oi, kind: FaultKind::Unmapped }),
        }
    }

    fn write(&mut self, addr: u64, data: &[u8], oi: MemOpIdx) -> Result<(), MemFault> {
        match self.range(addr, data.len()) {
            Some(r) => {
                self.bytes[r].copy_from_slice(data);
                Ok(())
            }
            None => Err(MemFault { addr, write: true, oi, kind: FaultKind::Unmapped }),
        }
    }
}

/// A memory with nothing mapped. Every access faults.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoMemory;

impl GuestMemory for NoMemory {
    fn read(&mut self, addr: u64, _buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault> {
        Err(MemFault { addr, write: false, oi, kind: FaultKind::Unmapped })
    }

    fn write(&mut self, addr: u64, _data: &[u8], oi: MemOpIdx) -> Result<(), MemFault> {
        Err(MemFault { addr, write: true, oi, kind: FaultKind::Unmapped })
    }
}

fn check_align(addr: u64, oi: MemOpIdx, write: bool) -> Result<(), MemFault> {
    let bits = oi.memop().alignment_bits();
    if bits != 0 && addr & ((1u64 << bits) - 1) != 0 {
        return Err(MemFault { addr, write, oi, kind: FaultKind::Unaligned });
    }
    Ok(())
}

/// Load the value a [`MemOpIdx`] describes: up to 16 bytes, byte swapped if the op is big
/// endian, and sign extended to 128 bits if the op is signed.
pub fn guest_load(mem: &mut dyn GuestMemory, addr: u64, oi: MemOpIdx) -> Result<u128, MemFault> {
    load_impl(mem, None, addr, oi)
}

/// [`guest_load`] through [`GuestMemory::read_with_env`].
pub fn guest_load_env(
    mem: &mut dyn GuestMemory,
    env: &mut [u8],
    addr: u64,
    oi: MemOpIdx,
) -> Result<u128, MemFault> {
    load_impl(mem, Some(env), addr, oi)
}

fn load_impl(
    mem: &mut dyn GuestMemory,
    env: Option<&mut [u8]>,
    addr: u64,
    oi: MemOpIdx,
) -> Result<u128, MemFault> {
    check_align(addr, oi, false)?;
    let mop = oi.memop();
    let n = mop.size_bytes() as usize;
    assert!(n <= 16, "guest access wider than 16 bytes");
    let mut buf = [0u8; 16];
    match env {
        Some(env) => mem.read_with_env(env, addr, &mut buf[..n], oi)?,
        None => mem.read(addr, &mut buf[..n], oi)?,
    }
    if mop.is_bswap() {
        buf[..n].reverse();
    }
    let v = u128::from_le_bytes(buf);
    if mop.is_signed() && n < 16 {
        let sh = 128 - 8 * n as u32;
        Ok((((v << sh) as i128) >> sh) as u128)
    } else {
        Ok(v)
    }
}

/// Store the low bytes of `val` as a [`MemOpIdx`] describes.
pub fn guest_store(
    mem: &mut dyn GuestMemory,
    addr: u64,
    val: u128,
    oi: MemOpIdx,
) -> Result<(), MemFault> {
    store_impl(mem, None, addr, val, oi)
}

/// [`guest_store`] through [`GuestMemory::write_with_env`].
pub fn guest_store_env(
    mem: &mut dyn GuestMemory,
    env: &mut [u8],
    addr: u64,
    val: u128,
    oi: MemOpIdx,
) -> Result<(), MemFault> {
    store_impl(mem, Some(env), addr, val, oi)
}

fn store_impl(
    mem: &mut dyn GuestMemory,
    env: Option<&mut [u8]>,
    addr: u64,
    val: u128,
    oi: MemOpIdx,
) -> Result<(), MemFault> {
    check_align(addr, oi, true)?;
    let mop = oi.memop();
    let n = mop.size_bytes() as usize;
    assert!(n <= 16, "guest access wider than 16 bytes");
    let mut buf = val.to_le_bytes();
    if mop.is_bswap() {
        buf[..n].reverse();
    }
    match env {
        Some(env) => mem.write_with_env(env, addr, &buf[..n], oi),
        None => mem.write(addr, &buf[..n], oi),
    }
}

/// The memop with only size and byte order kept, for accesses done by helpers.
pub(crate) fn plain(oi: MemOpIdx) -> MemOpIdx {
    let m = oi.memop().and(MemOp::SIZE.or(MemOp::BSWAP).or(MemOp::AMASK));
    MemOpIdx::new(m, oi.mmu_idx())
}
