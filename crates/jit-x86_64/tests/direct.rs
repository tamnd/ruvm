// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct user mode accesses ([`CompileOptions::guest_base`]): `qemu_ld` and `qemu_st` below
//! the bound are host accesses at the guest address plus the base, with no TLB, and addresses
//! past the bound, accesses less aligned than they must be, and accesses the host refuses go
//! to the guest memory through the slow path or fault, the last ones by way of
//! [`ruvm_jit_x86_64::fault_target`] in a signal handler.
//!
//! Generated code only runs on an x86-64 host; elsewhere these tests only compile and link.

#![cfg(unix)]

use std::sync::atomic::{AtomicU8, Ordering};

use ruvm_jit_core::ir::{FuncConfig, TempI64};
use ruvm_jit_core::types::MemOpIdx;
use ruvm_jit_core::{Func, MemOp, Type};
use ruvm_jit_interp::{Exit, FaultKind, GuestMemory, HelperRegistry, MemFault, Unwind};
use ruvm_jit_x86_64::{CodeRegion, CompileOptions, HostFeatures};

const NATIVE: bool = cfg!(all(unix, target_arch = "x86_64"));
const ENV_SIZE: usize = 0x200;
const PAGE: u64 = 4096;
/// The guest address of the first host page.
const GUEST: u64 = 0x4000_0000;
const PAGES: usize = 4;
/// Guest addresses from `1 << BITS` take the slow path.
const BITS: u32 = 31;
/// What the slow path reads.
const SLOW: u8 = 0xab;

/// [`PAGES`] host pages, page aligned as guest memory is.
#[repr(C, align(4096))]
struct Host([AtomicU8; PAGES * PAGE as usize]);

impl Host {
    fn new() -> Box<Host> {
        Box::new(Host(std::array::from_fn(|i| AtomicU8::new(i as u8))))
    }

    fn base(&self) -> u64 {
        (self.0.as_ptr() as u64).wrapping_sub(GUEST)
    }

    fn peek(&self, guest: u64, len: usize) -> u64 {
        let off = (guest - GUEST) as usize;
        let mut v = [0u8; 8];
        for (i, b) in v[..len].iter_mut().enumerate() {
            *b = self.0[off + i].load(Ordering::Relaxed);
        }
        u64::from_le_bytes(v)
    }

    /// Give page `page` the host protection `prot`. The test puts back read and write before
    /// the pages are freed.
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn protect(&self, page: usize, prot: i32) {
        let p = self.0[page * PAGE as usize..].as_ptr();
        // SAFETY: the page is a whole page of `self`, which only the blocks under test and
        // `peek` touch, and the test does not peek at a page it protected.
        let r = unsafe { libc::mprotect(p as *mut libc::c_void, PAGE as usize, prot) };
        assert_eq!(r, 0);
    }
}

/// The slow path: reads give [`SLOW`] bytes, writes are recorded.
#[derive(Default)]
struct SlowMem {
    reads: Vec<u64>,
    writes: Vec<(u64, Vec<u8>)>,
}

impl GuestMemory for SlowMem {
    fn read(&mut self, addr: u64, buf: &mut [u8], _: MemOpIdx) -> Result<(), MemFault> {
        self.reads.push(addr);
        buf.fill(SLOW);
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8], _: MemOpIdx) -> Result<(), MemFault> {
        self.writes.push((addr, data.to_vec()));
        Ok(())
    }
}

fn rd64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

/// Loads of `loads` (address and memop) into env words from 0x100, then stores of env word
/// 0x180 at `stores`.
fn block(addr_type: Type, loads: &[(u64, MemOp)], stores: &[(u64, MemOp)]) -> Func {
    let mut f = Func::new(FuncConfig { addr_type, ..FuncConfig::default() });
    let env = f.env();
    let at = |f: &mut Func, a: u64| {
        if addr_type == Type::I32 {
            f.constant_i32(a as i32).temp()
        } else {
            f.constant_i64(a as i64).temp()
        }
    };
    let regs: Vec<TempI64> =
        (0..loads.len()).map(|k| f.global_mem_new_i64(env, 0x100 + 8 * k as i64, "r")).collect();
    let v: TempI64 = f.global_mem_new_i64(env, 0x180, "v");
    for (&r, &(a, m)) in regs.iter().zip(loads) {
        let a = at(&mut f, a);
        f.gen_qemu_ld_i64(r, a, 0, m);
    }
    for &(a, m) in stores {
        let a = at(&mut f, a);
        f.gen_qemu_st_i64(v, a, 0, m);
    }
    f.gen_exit_tb(0, 0);
    f
}

/// Run `f` with the guest at `host`, the value 0x8877_6655_4433_2211 in env word 0x180.
fn run(host: &Host, f: &Func, mem: &mut SlowMem) -> Vec<u8> {
    let r = CodeRegion::with_features(1 << 16, HostFeatures::detect()).expect("code region");
    let opts = CompileOptions { guest_base: Some((host.base(), BITS)), ..Default::default() };
    let tb = r.compile_with(f, &opts).expect("compile");
    let mut env = vec![0u8; ENV_SIZE];
    env[0x180..0x188].copy_from_slice(&0x8877_6655_4433_2211u64.to_le_bytes());
    assert_eq!(tb.run(&mut env, mem, &HelperRegistry::new()), Ok(Exit::ExitTb(0)));
    env
}

#[test]
fn accesses_below_the_bound_go_straight_to_the_host() {
    if !NATIVE {
        return;
    }
    let host = Host::new();
    for ty in [Type::I64, Type::I32] {
        let loads = [
            (GUEST + 8, MemOp::UQ),
            (GUEST + 0x11, MemOp::SB),
            (GUEST + 0x22, MemOp::SW.or(MemOp::ALIGN)),
            // Crossing into the next page is one host access too.
            (GUEST + PAGE - 2, MemOp::UL),
        ];
        let stores = [(GUEST + PAGE + 0x40, MemOp::UQ), (GUEST + 2 * PAGE + 0x51, MemOp::UW)];
        let mut mem = SlowMem::default();
        let env = run(&host, &block(ty, &loads, &stores), &mut mem);
        assert!(mem.reads.is_empty() && mem.writes.is_empty(), "{ty:?}");
        assert_eq!(rd64(&env, 0x100), 0x0f0e_0d0c_0b0a_0908);
        assert_eq!(rd64(&env, 0x108), 0x11);
        assert_eq!(rd64(&env, 0x110), 0x2322);
        assert_eq!(rd64(&env, 0x118), 0x0100_fffe);
        assert_eq!(host.peek(GUEST + PAGE + 0x40, 8), 0x8877_6655_4433_2211);
        assert_eq!(host.peek(GUEST + 2 * PAGE + 0x51, 2), 0x2211);
    }
}

#[test]
fn addresses_past_the_bound_take_the_slow_path() {
    if !NATIVE {
        return;
    }
    let host = Host::new();
    for ty in [Type::I64, Type::I32] {
        let far = (1 << BITS) + 0x1000;
        let loads = [(far, MemOp::UQ), (far + 0x41, MemOp::UL)];
        let stores = [(far + 8, MemOp::UL), (far + 0x43, MemOp::UW)];
        let mut mem = SlowMem::default();
        let env = run(&host, &block(ty, &loads, &stores), &mut mem);
        assert_eq!(mem.reads, [far, far + 0x41], "{ty:?}");
        assert_eq!(
            mem.writes,
            [(far + 8, vec![0x11, 0x22, 0x33, 0x44]), (far + 0x43, vec![0x11, 0x22])]
        );
        assert_eq!(rd64(&env, 0x100), u64::from_le_bytes([SLOW; 8]));
        assert_eq!(rd64(&env, 0x108), u64::from(u32::from_le_bytes([SLOW; 4])));
    }
}

#[test]
fn misaligned_accesses_that_must_be_aligned_fault() {
    if !NATIVE {
        return;
    }
    let host = Host::new();
    for ty in [Type::I64, Type::I32] {
        for (loads, stores, write) in [
            (&[(GUEST + 0x41, MemOp::UL.or(MemOp::ALIGN))][..], &[][..], false),
            (&[][..], &[(GUEST + 0x43, MemOp::UW.or(MemOp::ALIGN))][..], true),
        ] {
            let f = block(ty, loads, stores);
            let r = CodeRegion::with_features(1 << 16, HostFeatures::detect()).expect("region");
            let opts =
                CompileOptions { guest_base: Some((host.base(), BITS)), ..Default::default() };
            let tb = r.compile_with(&f, &opts).expect("compile");
            let mut env = vec![0u8; ENV_SIZE];
            let mut mem = SlowMem::default();
            match tb.run(&mut env, &mut mem, &HelperRegistry::new()) {
                Ok(Exit::Unwind(Unwind::Mem(e))) => {
                    assert_eq!((e.kind, e.write), (FaultKind::Unaligned, write), "{ty:?}");
                    assert_eq!(e.addr, if write { GUEST + 0x43 } else { GUEST + 0x41 });
                }
                other => panic!("{ty:?}: {other:?}"),
            }
            assert!(mem.reads.is_empty() && mem.writes.is_empty());
        }
    }
    assert_eq!(host.peek(GUEST + 0x43, 2), 0x4443);
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod faults {
    use ruvm_jit_x86_64::fault_target;

    use super::*;

    /// Resume a faulting direct access at its slow path, as the user mode signal handler does.
    extern "C" fn on_fault(_: i32, _: *mut libc::siginfo_t, uc: *mut libc::c_void) {
        let uc = uc.cast::<libc::ucontext_t>();
        // SAFETY: the kernel passes the interrupted context, which the handler may change.
        let rip = unsafe { &mut (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] };
        match fault_target(*rip as u64) {
            Some(to) => *rip = to as i64,
            // Not ours: a real crash.
            None => std::process::abort(),
        }
    }

    fn install() {
        // SAFETY: the handler only reads and changes the context it is given and calls
        // `fault_target`, which is safe in a signal handler.
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = on_fault as *const () as usize;
            sa.sa_flags = libc::SA_SIGINFO;
            assert_eq!(libc::sigaction(libc::SIGSEGV, &sa, std::ptr::null_mut()), 0);
        }
    }

    #[test]
    fn accesses_the_host_refuses_take_the_slow_path() {
        if !NATIVE {
            return;
        }
        install();
        let host = Host::new();
        host.protect(2, libc::PROT_NONE);
        host.protect(3, libc::PROT_READ);
        for ty in [Type::I64, Type::I32] {
            let loads = [
                (GUEST + 2 * PAGE + 8, MemOp::UQ),
                (GUEST + 3 * PAGE + 8, MemOp::UB),
                // The second page refuses it.
                (GUEST + 2 * PAGE - 4, MemOp::UQ),
            ];
            let stores = [(GUEST + 3 * PAGE + 0x10, MemOp::UL), (GUEST + PAGE + 0x10, MemOp::UL)];
            let mut mem = SlowMem::default();
            let env = run(&host, &block(ty, &loads, &stores), &mut mem);
            assert_eq!(mem.reads, [GUEST + 2 * PAGE + 8, GUEST + 2 * PAGE - 4], "{ty:?}");
            assert_eq!(mem.writes, [(GUEST + 3 * PAGE + 0x10, vec![0x11, 0x22, 0x33, 0x44])]);
            assert_eq!(rd64(&env, 0x100), u64::from_le_bytes([SLOW; 8]));
            assert_eq!(rd64(&env, 0x108), 8);
            assert_eq!(rd64(&env, 0x110), u64::from_le_bytes([SLOW; 8]));
            assert_eq!(host.peek(GUEST + PAGE + 0x10, 4), 0x4433_2211);
        }
        host.protect(2, libc::PROT_READ | libc::PROT_WRITE);
        host.protect(3, libc::PROT_READ | libc::PROT_WRITE);
    }

    #[test]
    fn other_faults_are_not_redirected() {
        let x = 0u64;
        assert_eq!(fault_target(std::ptr::addr_of!(x) as u64 + 7), None);
        assert_eq!(fault_target(on_fault as *const () as usize as u64 + 16), None);
        assert_eq!(fault_target(0), None);
    }
}
