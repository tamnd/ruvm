// SPDX-License-Identifier: GPL-2.0-or-later

//! The inline softmmu TLB lookup of `qemu_ld` and `qemu_st` ([`CompileOptions::tlb_page_bits`]):
//! accesses that hit go straight to host memory, and misses, flagged comparators, misaligned
//! and page crossing accesses, and accesses after the TLB is resized in the middle of a block go
//! to the guest memory, with the same results.
//!
//! Generated code only runs on an AArch64 host; elsewhere these tests only compile and link.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use ruvm_jit_aarch64::{CodeRegion, CodegenOptions, CompileOptions, CompiledTb, GenCodeError};
use ruvm_jit_core::ir::{FuncConfig, HelperType, Temp, TempI32, TempI64};
use ruvm_jit_core::types::{INSN_START_WORDS, MemOpIdx, call_flags};
use ruvm_jit_core::{Func, HelperInfo, MemOp, Type};
use ruvm_jit_interp::fast_tlb::TLB_INVALID_ENTRY;
use ruvm_jit_interp::{
    Exit, FastTlb, FaultKind, GuestMemory, HelperEnv, HelperRegistry, MemFault, TlbTables, Unwind,
};

const NATIVE: bool = cfg!(all(unix, target_arch = "aarch64"));
const ENV_SIZE: usize = 0x200;
const PAGE_BITS: u32 = 12;
const PAGE: u64 = 1 << PAGE_BITS;
/// The guest address of the first page of [`TlbMem`].
const GUEST: u64 = 0x4000_0000;
const PAGES: usize = 4;
/// TLB entries per MMU index.
const ENTRIES: usize = 16;
/// MMU indexes the TLB has tables for.
const MODES: usize = 2;
/// The comparator flag the softmmu uses for `TLB_NOTDIRTY`.
const NOTDIRTY: u64 = 1 << 7;

/// Guest memory of [`PAGES`] pages at [`GUEST`], backed by host bytes the TLB entries point
/// into, counting the accesses that reach it (the slow path).
struct TlbMem {
    host: Box<[AtomicU8]>,
    fast: Arc<FastTlb>,
    reads: usize,
    writes: usize,
    /// The first `insn_start` word last reported.
    insn: u64,
    /// [`TlbMem::insn`] at each slow path access.
    seen: Vec<u64>,
}

impl TlbMem {
    fn new(fast: Arc<FastTlb>) -> TlbMem {
        let host = (0..PAGES * PAGE as usize).map(|i| AtomicU8::new(i as u8)).collect();
        TlbMem { host, fast, reads: 0, writes: 0, insn: 0, seen: Vec::new() }
    }

    fn byte(&self, addr: u64, write: bool, oi: MemOpIdx) -> Result<&AtomicU8, MemFault> {
        let fault = MemFault { addr, write, oi, kind: FaultKind::Unmapped };
        let off = addr.checked_sub(GUEST).ok_or(fault)?;
        self.host.get(off as usize).ok_or(fault)
    }

    fn peek(&self, addr: u64, len: usize) -> u64 {
        let off = (addr - GUEST) as usize;
        let mut v = [0u8; 8];
        for (i, b) in v[..len].iter_mut().enumerate() {
            *b = self.host[off + i].load(Ordering::Relaxed);
        }
        u64::from_le_bytes(v)
    }

    fn poke(&self, addr: u64, v: u64) {
        let off = (addr - GUEST) as usize;
        for (i, b) in v.to_le_bytes().iter().enumerate() {
            self.host[off + i].store(*b, Ordering::Relaxed);
        }
    }
}

impl GuestMemory for TlbMem {
    fn read(&mut self, addr: u64, buf: &mut [u8], oi: MemOpIdx) -> Result<(), MemFault> {
        self.reads += 1;
        self.seen.push(self.insn);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self.byte(addr + i as u64, false, oi)?.load(Ordering::Relaxed);
        }
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8], oi: MemOpIdx) -> Result<(), MemFault> {
        self.writes += 1;
        self.seen.push(self.insn);
        for (i, b) in data.iter().enumerate() {
            self.byte(addr + i as u64, true, oi)?.store(*b, Ordering::Relaxed);
        }
        Ok(())
    }

    fn fast_tlb(&self) -> Option<Arc<FastTlb>> {
        Some(Arc::clone(&self.fast))
    }

    fn insn_start(&mut self, words: &[u64; INSN_START_WORDS]) {
        self.insn = words[0];
    }
}

/// The slot of the page at `guest` in a table of `entries` entries.
fn index(guest: u64, entries: usize) -> usize {
    ((guest >> PAGE_BITS) as usize) & (entries - 1)
}

/// Map page `page` of `mem` for MMU index `mmu_idx`, for reads, and for writes if `write`.
fn map(t: &TlbTables, mem: &TlbMem, mmu_idx: usize, page: usize, write: bool) {
    let guest = GUEST + page as u64 * PAGE;
    let host = mem.host.as_ptr() as u64 + page as u64 * PAGE;
    let w = if write { guest } else { u64::MAX };
    let e = [guest, w, u64::MAX, host.wrapping_sub(guest)];
    // SAFETY: the entry maps the page to `PAGE` bytes of `mem.host`, which are atomics, so
    // writes through a shared reference are fine. Every test keeps `mem`, and so the bytes,
    // alive until it is done running blocks against these tables.
    unsafe { t.set(mmu_idx, index(guest, t.len(mmu_idx)), e) }
}

fn opts() -> CompileOptions<'static> {
    CompileOptions { tlb_page_bits: Some(PAGE_BITS), ..CompileOptions::default() }
}

fn compile(f: &Func) -> CompiledTb {
    let r = CodeRegion::new(1 << 16).expect("code region");
    r.compile_chained(f, &CodegenOptions::default(), &opts()).expect("compile")
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

/// A block with loads and stores of every size into page 0 and 1, through MMU index 1.
fn ldst_block(addr_type: Type) -> Func {
    let mut f = Func::new(FuncConfig { addr_type, ..FuncConfig::default() });
    let env = f.env();
    let r: Vec<TempI64> =
        (0..6).map(|i| f.global_mem_new_i64(env, 0x100 + 8 * i, &format!("r{i}"))).collect();
    let w = f.global_mem_new_i32(env, 0x140, "w");
    let at = |f: &mut Func, a: u64| -> Temp {
        if addr_type == Type::I32 {
            f.constant_i32(a as i32).temp()
        } else {
            f.constant_i64(a as i64).temp()
        }
    };
    let a = at(&mut f, GUEST + 8);
    f.gen_qemu_ld_i64(r[0], a, 1, MemOp::UQ);
    let a = at(&mut f, GUEST + 0x11);
    f.gen_qemu_ld_i64(r[1], a, 1, MemOp::SB);
    let a = at(&mut f, GUEST + 0x22);
    f.gen_qemu_ld_i64(r[2], a, 1, MemOp::SW.or(MemOp::ALIGN));
    let a = at(&mut f, GUEST + 0x34);
    f.gen_qemu_ld_i32(w, a, 1, MemOp::UL);
    let a = at(&mut f, GUEST + PAGE + 0x40);
    f.gen_qemu_st_i64(r[0], a, 1, MemOp::UQ);
    let a = at(&mut f, GUEST + PAGE + 0x48);
    f.gen_qemu_st_i32(w, a, 1, MemOp::UW);
    let c = f.constant_i64(0x7f);
    let a = at(&mut f, GUEST + PAGE + 0x4b);
    f.gen_qemu_st_i64(c, a, 1, MemOp::UB);
    // Not aligned to its size, but not crossing the page: the fast path still applies.
    let a = at(&mut f, GUEST + PAGE + 0x51);
    f.gen_qemu_st_i64(r[1], a, 1, MemOp::UL);
    let a = at(&mut f, GUEST + PAGE + 0x51);
    f.gen_qemu_ld_i64(r[3], a, 1, MemOp::SL);
    f.gen_exit_tb(0, 0);
    f
}

/// Run [`ldst_block`] against a TLB set up by `setup` and check the results, which do not
/// depend on the path each access takes. Returns the slow path reads and writes.
fn run_ldst(addr_type: Type, setup: impl Fn(&TlbTables, &TlbMem)) -> (usize, usize) {
    let tb = compile(&ldst_block(addr_type));
    let t = TlbTables::new(PAGE_BITS, MODES, ENTRIES);
    let mut mem = TlbMem::new(Arc::clone(t.fast()));
    setup(&t, &mem);
    let mut env = vec![0u8; ENV_SIZE];
    let x = tb.run(&mut env, &mut mem, &HelperRegistry::new());
    assert_eq!(x, Ok(Exit::ExitTb(0)));
    assert_eq!(rd64(&env, 0x100), 0x0f0e_0d0c_0b0a_0908);
    assert_eq!(rd64(&env, 0x108), 0x11);
    assert_eq!(rd64(&env, 0x110), 0x2322);
    assert_eq!(rd64(&env, 0x140) as u32, 0x3736_3534);
    assert_eq!(mem.peek(GUEST + PAGE + 0x40, 8), 0x0f0e_0d0c_0b0a_0908);
    assert_eq!(mem.peek(GUEST + PAGE + 0x48, 4), 0x7f4a_3534);
    assert_eq!(mem.peek(GUEST + PAGE + 0x51, 4), 0x11);
    assert_eq!(rd64(&env, 0x118), 0x11);
    (mem.reads, mem.writes)
}

#[test]
fn hits_go_straight_to_host_memory() {
    if !NATIVE {
        return;
    }
    for ty in [Type::I64, Type::I32] {
        let n = run_ldst(ty, |t, mem| {
            map(t, mem, 1, 0, false);
            map(t, mem, 1, 1, true);
        });
        assert_eq!(n, (0, 0), "{ty:?}");
    }
}

#[test]
fn misses_take_the_slow_path() {
    if !NATIVE {
        return;
    }
    // Nothing mapped.
    assert_eq!(run_ldst(Type::I64, |_, _| {}), (5, 4));
    // Mapped for another MMU index only.
    let n = run_ldst(Type::I64, |t, mem| {
        map(t, mem, 0, 0, true);
        map(t, mem, 0, 1, true);
    });
    assert_eq!(n, (5, 4));
    // Page 1 mapped read only: its stores miss, its load hits.
    let n = run_ldst(Type::I64, |t, mem| {
        map(t, mem, 1, 0, false);
        map(t, mem, 1, 1, false);
    });
    assert_eq!(n, (0, 4));
}

#[test]
fn flagged_comparators_take_the_slow_path() {
    if !NATIVE {
        return;
    }
    let n = run_ldst(Type::I64, |t, mem| {
        map(t, mem, 1, 0, false);
        map(t, mem, 1, 1, true);
        // Writes to page 1 are not dirty yet, as for a page holding code.
        t.add_flags(1, index(GUEST + PAGE, ENTRIES), 1, NOTDIRTY);
    });
    assert_eq!(n, (0, 4));
    // An entry for another page in the same slot does not match.
    let n = run_ldst(Type::I64, |t, mem| {
        map(t, mem, 1, 0, false);
        let other = GUEST + PAGE + ENTRIES as u64 * PAGE;
        // SAFETY: as in `map`; `other` is never accessed, as no test address is on it.
        unsafe { t.set(1, index(other, ENTRIES), [other, other, u64::MAX, 0]) }
    });
    assert_eq!(n, (1, 4));
}

/// One load of `memop` at `addr` through MMU index 0 into env word 0x100.
fn one_load(addr: u64, memop: MemOp) -> Func {
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let r = f.global_mem_new_i64(env, 0x100, "r");
    let a = f.constant_i64(addr as i64);
    f.gen_qemu_ld_i64(r, a, 0, memop);
    f.gen_exit_tb(0, 0);
    f
}

#[test]
fn misaligned_and_page_crossing_accesses_take_the_slow_path() {
    if !NATIVE {
        return;
    }
    let t = TlbTables::new(PAGE_BITS, MODES, ENTRIES);
    let mut mem = TlbMem::new(Arc::clone(t.fast()));
    for p in 0..PAGES {
        map(&t, &mem, 0, p, true);
    }
    let run = |mem: &mut TlbMem, f: &Func| {
        let mut env = vec![0u8; ENV_SIZE];
        let x = compile(f).run(&mut env, mem, &HelperRegistry::new());
        (x, rd64(&env, 0x100))
    };
    // Aligned as asked: a hit.
    let (x, v) = run(&mut mem, &one_load(GUEST + 0x40, MemOp::UQ.or(MemOp::ALIGN)));
    assert_eq!((x, v, mem.reads), (Ok(Exit::ExitTb(0)), mem.peek(GUEST + 0x40, 8), 0));
    // Misaligned with MO_ALIGN: the slow path raises the alignment fault.
    let (x, _) = run(&mut mem, &one_load(GUEST + 0x41, MemOp::UL.or(MemOp::ALIGN)));
    match x {
        Ok(Exit::Unwind(Unwind::Mem(m))) => assert_eq!(m.kind, FaultKind::Unaligned),
        other => panic!("{other:?}"),
    }
    // Crossing into the next page: the slow path, which reads both pages.
    mem.poke(GUEST + PAGE - 4, 0x1122_3344_5566_7788);
    let reads = mem.reads;
    let (x, v) = run(&mut mem, &one_load(GUEST + PAGE - 4, MemOp::UQ));
    assert_eq!((x, v), (Ok(Exit::ExitTb(0)), 0x1122_3344_5566_7788));
    assert_eq!(mem.reads, reads + 1);
    // The last 8 bytes of a page do not cross it.
    let (x, _) = run(&mut mem, &one_load(GUEST + PAGE - 8, MemOp::UQ));
    assert_eq!((x, mem.reads), (Ok(Exit::ExitTb(0)), reads + 1));
}

thread_local! {
    /// The TLB of the test running on this thread, for [`helper_resize`].
    static TABLES: RefCell<Option<TlbTables>> = const { RefCell::new(None) };
}

/// Resize MMU index 0's table to `a[0]` entries, as `tlb_flush` after a resize does.
fn helper_resize(_: &mut HelperEnv<'_>, a: &[u64]) -> Result<u128, Unwind> {
    TABLES.with(|t| t.borrow_mut().as_mut().expect("tables").resize(0, a[0] as usize));
    Ok(0)
}

#[test]
fn a_resize_during_a_block_is_seen_by_the_next_access() {
    if !NATIVE {
        return;
    }
    let info = HelperInfo::new("resize", call_flags::NO_RWG, HelperType::Void, &[HelperType::I64]);
    let mut reg = HelperRegistry::new();
    reg.register_info(&info, helper_resize);

    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let r: Vec<TempI64> =
        (0..3).map(|i| f.global_mem_new_i64(env, 0x100 + 8 * i, &format!("r{i}"))).collect();
    let a = f.constant_i64((GUEST + 0x80) as i64);
    f.gen_qemu_ld_i64(r[0], a, 0, MemOp::UQ);
    let h = f.helper(info);
    let n = f.constant_i64(64);
    f.gen_call(h, None, &[n.temp()]);
    f.gen_qemu_ld_i64(r[1], a, 0, MemOp::UQ);
    let b: TempI32 = f.constant_i32(7);
    f.gen_qemu_st_i32(b, a, 0, MemOp::UB);
    f.gen_qemu_ld_i64(r[2], a, 0, MemOp::UQ);
    f.gen_exit_tb(0, 0);
    let tb = compile(&f);

    let t = TlbTables::new(PAGE_BITS, MODES, ENTRIES);
    let mut mem = TlbMem::new(Arc::clone(t.fast()));
    map(&t, &mem, 0, 0, true);
    TABLES.with(|c| *c.borrow_mut() = Some(t));
    let mut env = vec![0u8; ENV_SIZE];
    let x = tb.run(&mut env, &mut mem, &reg);
    let t = TABLES.with(|c| c.borrow_mut().take()).expect("tables");
    assert_eq!(x, Ok(Exit::ExitTb(0)));
    let v = 0x8786_8584_8382_8180;
    assert_eq!((rd64(&env, 0x100), rd64(&env, 0x108)), (v, v));
    assert_eq!(rd64(&env, 0x110), v & !0xff | 7);
    // The first load hit; the new table is empty, so the rest missed.
    assert_eq!((mem.reads, mem.writes), (2, 1));
    assert_eq!(t.len(0), 64);
    assert_eq!(t.get(0, index(GUEST, 64)), TLB_INVALID_ENTRY);
}

#[test]
fn memory_without_a_tlb_of_the_page_size_misses() {
    if !NATIVE {
        return;
    }
    let tb = compile(&ldst_block(Type::I64));
    // A TLB for 16 KiB pages: the block was compiled for 4 KiB ones, so it must not read it.
    let t = TlbTables::new(PAGE_BITS + 2, MODES, ENTRIES);
    let mut mem = TlbMem::new(Arc::clone(t.fast()));
    let guest = GUEST;
    let host = mem.host.as_ptr() as u64;
    // SAFETY: as in `map`, for the first 16 KiB, which is all of `mem.host`.
    unsafe {
        t.set(1, index(guest >> 2, ENTRIES), [guest, guest, u64::MAX, host.wrapping_sub(guest)])
    }
    let mut env = vec![0u8; ENV_SIZE];
    assert_eq!(tb.run(&mut env, &mut mem, &HelperRegistry::new()), Ok(Exit::ExitTb(0)));
    assert_eq!((mem.reads, mem.writes), (5, 4));
}

#[test]
fn a_chain_has_one_page_size() {
    let r = CodeRegion::new(1 << 16).expect("code region");
    let g = CodegenOptions::default();
    r.compile_chained(&one_load(GUEST, MemOp::UB), &g, &opts()).expect("compile");
    let next = r.successor(1 << 16).expect("code region");
    let other = CompileOptions { tlb_page_bits: Some(PAGE_BITS + 1), ..CompileOptions::default() };
    let e = next.compile_chained(&one_load(GUEST, MemOp::UB), &g, &other).map(|_| ());
    assert!(matches!(e, Err(GenCodeError::Unsupported(_))), "{e:?}");
    // Without the lookup, any block goes.
    let plain = CompileOptions::default();
    next.compile_chained(&one_load(GUEST, MemOp::UB), &g, &plain).expect("compile");
}

#[test]
fn a_miss_is_served_in_the_instruction_of_the_access() {
    if !NATIVE {
        return;
    }
    let mut f = Func::new(FuncConfig::default());
    let env = f.env();
    let r: Vec<TempI64> =
        (0..3).map(|i| f.global_mem_new_i64(env, 0x100 + 8 * i, &format!("r{i}"))).collect();
    for (i, addr) in [GUEST + 8, GUEST + PAGE + 0x10, 0x10].into_iter().enumerate() {
        f.gen_insn_start(&[0x10 * (i as u64 + 1), 0, 0]);
        let a = f.constant_i64(addr as i64);
        f.gen_qemu_ld_i64(r[i], a, 0, MemOp::UQ);
    }
    f.gen_insn_start(&[0x40, 0, 0]);
    f.gen_exit_tb(0, 0);
    let tb = compile(&f);
    let t = TlbTables::new(PAGE_BITS, MODES, ENTRIES);
    let mut mem = TlbMem::new(Arc::clone(t.fast()));
    let mut env = vec![0u8; ENV_SIZE];
    let mut last = None;
    // Nothing is mapped, so every load takes the slow path; the last one faults.
    let x = tb.run_traced(&mut env, &mut mem, &HelperRegistry::new(), &mut last);
    match x {
        Ok(Exit::Unwind(Unwind::Mem(m))) => assert_eq!(m.addr, 0x10),
        other => panic!("{other:?}"),
    }
    assert_eq!(mem.seen, [0x10, 0x20, 0x30]);
    assert_eq!(last, Some([0x30, 0, 0]));
    assert_eq!(rd64(&env, 0x108), mem.peek(GUEST + PAGE + 0x10, 8));
}
