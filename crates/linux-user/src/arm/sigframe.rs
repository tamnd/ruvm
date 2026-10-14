// SPDX-License-Identifier: GPL-2.0-or-later

//! The signal frames of arm, `linux-user/arm/signal.c`: `struct sigframe` for a handler
//! without `SA_SIGINFO` and `struct rt_sigframe` for one with it, both with the VFP state in
//! the register space of the `ucontext`. There is no FDPIC, so the return code on the stack
//! is always that of `write_arm_sigreturn()`.
//!
//! FPEXC, FPINST and FPINST2 are not kept: the frame has FPEXC.EN set and the other two 0,
//! what a user mode CPU resets them to, and sigreturn drops them. A program at PL0 cannot
//! read them anyway.

use std::sync::OnceLock;

use ruvm_jit::Cpu;
use ruvm_target_arm::cpu::{CPSR_E, CPSR_EXEC, CPSR_IT, CPSR_T, CPSR_USER, CpuArmState};
use ruvm_target_arm::tcg::user;
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use super::ARM_SYSCALL_BASE;
use crate::signal::{self, Sigaction, Task, get32, get64, put32, put64};

/// `TARGET_NR_sigreturn` and `TARGET_NR_rt_sigreturn`.
const NR_SIGRETURN: u32 = 119;
const NR_RT_SIGRETURN: u32 = 173;

/// `struct sigframe`: the `ucontext`, then four words of return code.
const FRAME_SIZE: usize = 760;
const RETCODE: usize = 744;
/// `struct rt_sigframe`: the `siginfo`, then a `struct sigframe`.
const RT_FRAME_SIZE: usize = 888;
const RT_UC: usize = 128;
/// `struct target_ucontext`: the flags and link, the `stack_t`, the `sigcontext`, the mask
/// and, 8 byte aligned, the register space.
const UC_STACK: usize = 8;
const UC_MCONTEXT: usize = 20;
const UC_SIGMASK: usize = 104;
const UC_REGSPACE: usize = 232;
/// `struct target_sigcontext`: `trap_no`, `error_code` and `oldmask`, R0 to R15, the CPSR
/// and the fault address.
const SC_OLDMASK: usize = 8;
const SC_R0: usize = 12;
const SC_PC: usize = SC_R0 + 60;
const SC_CPSR: usize = 76;
/// `struct target_vfp_sigframe`: the magic and size, D0 to D31, FPSCR, then FPEXC, FPINST
/// and FPINST2.
const VFP_MAGIC: u32 = 0x5646_5001;
const VFP_SIZE: usize = 288;
const VFP_D0: usize = 8;
const VFP_FPSCR: usize = 264;
const VFP_FPEXC: usize = 272;
/// FPEXC.EN, FPEXC in a user mode CPU.
const FPEXC_EN: u32 = 1 << 30;

/// The address of the sigreturn trampolines.
static SIGTRAMP: OnceLock<u32> = OnceLock::new();

/// `raw_syscall` of the vDSO, in T16: `movs r7, #nr; svc #0`.
fn thumb_sigreturn(nr: u32) -> [u8; 4] {
    let mut c = [0, 0, 0x00, 0xdf];
    c[..2].copy_from_slice(&(0x2700 | nr as u16).to_le_bytes());
    c
}

/// The page of the trampolines, where QEMU loads the vDSO with its `sigreturn_codes`:
/// `sigreturn` at the start and `rt_sigreturn` 8 bytes in, both T16.
pub(super) fn map_sigtramp(space: &GuestSpace) -> Result<(), String> {
    let kind = MapKind { anon: true, ..MapKind::default() };
    let err = |e| format!("mmap: {}", crate::strerror(e));
    let addr = space.mmap(0, PAGE_SIZE, page::READ | page::WRITE, kind, None, 0).map_err(err)?;
    space.write_raw(addr, &thumb_sigreturn(NR_SIGRETURN));
    space.write_raw(addr + 8, &thumb_sigreturn(NR_RT_SIGRETURN));
    space.mprotect(addr, PAGE_SIZE, page::READ | page::EXEC).map_err(err)?;
    let _ = SIGTRAMP.set(addr as u32);
    Ok(())
}

/// `get_sigframe()`: 8 byte aligned room for `size` bytes.
fn get_sigframe(t: &Task, st: &CpuArmState, sa: &Sigaction, size: usize) -> u32 {
    let sp = signal::target_sigsp(t, u64::from(st.regs[13]), sa) as u32;
    sp.wrapping_sub(size as u32) & !7
}

/// `setup_sigframe()` into the `ucontext` `uc`: the `stack_t`, the registers, the VFP state
/// and the mask `old`.
fn setup_sigframe(t: &Task, st: &CpuArmState, uc: &mut [u8], old: u64) {
    uc[..UC_MCONTEXT].fill(0);
    let stack = signal::save_altstack(t, u64::from(st.regs[13]));
    uc[UC_STACK..UC_MCONTEXT].copy_from_slice(&stack[..12]);
    let sc = &mut uc[UC_MCONTEXT..UC_SIGMASK];
    for (i, &r) in st.regs.iter().enumerate() {
        put32(sc, SC_R0 + 4 * i, r);
    }
    put32(sc, SC_PC, st.pc as u32);
    put32(sc, SC_CPSR, st.cpsr_read());
    put32(sc, 0, 0);
    put32(sc, 4, 0);
    put32(sc, SC_CPSR + 4, 0);
    put32(sc, SC_OLDMASK, old as u32);

    let vfp = &mut uc[UC_REGSPACE..];
    put32(vfp, 0, VFP_MAGIC);
    put32(vfp, 4, VFP_SIZE as u32);
    for i in 0..32 {
        put64(vfp, VFP_D0 + 8 * i, st.zregs[i / 2][i % 2]);
    }
    put32(vfp, VFP_FPSCR, user::vfp_get_fpscr(st));
    put32(vfp, VFP_FPEXC, FPEXC_EN);
    put32(vfp, VFP_FPEXC + 4, 0);
    put32(vfp, VFP_FPEXC + 8, 0);
    // The terminating magic word.
    put32(vfp, VFP_SIZE, 0);

    put32(uc, UC_SIGMASK, old as u32);
    put32(uc, UC_SIGMASK + 4, (old >> 32) as u32);
}

/// `setup_return()`: the return code in the `struct sigframe` `frame` at `sp`, and the
/// registers the handler starts with.
fn setup_return(st: &mut CpuArmState, sa: &Sigaction, sig: i32, frame: &mut [u8], sp: u32) {
    let rt = sa.flags & signal::SA_SIGINFO != 0;
    let handler = sa.handler as u32;
    let thumb = handler & 1 != 0;
    let mut cpsr = st.cpsr_read() & !(CPSR_IT | CPSR_T | CPSR_E);
    if thumb {
        cpsr |= CPSR_T;
    }
    let tramp = *SIGTRAMP.get().expect("the vCPU is created first");
    // The trampolines are T16, so the return address has bit 0 set.
    let mut retcode = tramp + 8 * u32::from(rt) + 1;
    // The A32 code the kernel would run, for the ABI, though nothing runs it.
    let nr = if rt { NR_RT_SIGRETURN } else { NR_SIGRETURN };
    put32(frame, RETCODE, 0xe3a0_7000 | nr);
    put32(frame, RETCODE + 4, 0xef00_0000 | nr | ARM_SYSCALL_BASE);
    if sa.flags & signal::SA_RESTORER != 0 {
        retcode = sa.restorer as u32;
    }
    st.regs[0] = sig as u32;
    st.regs[13] = sp;
    st.regs[14] = retcode;
    st.pc = u64::from(handler & if thumb { !1 } else { !3 });
    st.cpsr_write(cpsr, CPSR_IT | CPSR_T | CPSR_E);
}

/// `setup_frame()` and `setup_rt_frame()`: the frame of either kind. `old` is the guest mask
/// to return to.
fn setup(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    info: Option<&signal::Info>,
    old: u64,
) {
    let mut st = CpuArmState::load(cpu.env);
    let (size, uc) = if info.is_some() { (RT_FRAME_SIZE, RT_UC) } else { (FRAME_SIZE, 0) };
    let frame = get_sigframe(t, &st, sa, size);
    let addr = u64::from(frame);
    let mut f = vec![0u8; size];
    if !space.check(addr, size as u64, page::WRITE) || !space.read_raw(addr, &mut f) {
        signal::force_sigsegv(t, sig);
        return;
    }
    if let Some(info) = info {
        f[..RT_UC].copy_from_slice(&signal::guest_siginfo(info));
    }
    setup_sigframe(t, &st, &mut f[uc..], old);
    setup_return(&mut st, sa, sig, &mut f[uc..], frame);
    if info.is_some() {
        st.regs[1] = frame;
        st.regs[2] = frame + RT_UC as u32;
    }
    space.write_raw(addr, &f);
    user::commit(cpu, &mut st);
}

/// `setup_frame()`.
pub(super) fn setup_frame(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    _info: &signal::Info,
    old: u64,
) {
    setup(space, t, cpu, sig, sa, None, old);
}

/// `setup_rt_frame()`.
pub(super) fn setup_rt_frame(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    info: &signal::Info,
    old: u64,
) {
    setup(space, t, cpu, sig, sa, Some(info), old);
}

/// `do_sigframe_return()` from the `ucontext` `uc`: false for a bad frame, which may have
/// restored the registers already.
fn sigframe_return(t: &mut Task, cpu: &Cpu<'_>, st: &mut CpuArmState, uc: &[u8]) -> bool {
    let mask = u64::from(get32(uc, UC_SIGMASK)) | (u64::from(get32(uc, UC_SIGMASK + 4)) << 32);
    signal::set_sigmask(t, signal::t2h_set(mask));

    // restore_sigcontext().
    let sc = &uc[UC_MCONTEXT..UC_SIGMASK];
    for (i, r) in st.regs.iter_mut().enumerate() {
        *r = get32(sc, SC_R0 + 4 * i);
    }
    st.pc = u64::from(get32(sc, SC_PC));
    st.cpsr_write(get32(sc, SC_CPSR), CPSR_USER | CPSR_EXEC);

    // restore_sigframe_vfp().
    let vfp = &uc[UC_REGSPACE..];
    if get32(vfp, 0) != VFP_MAGIC || get32(vfp, 4) as usize != VFP_SIZE {
        return false;
    }
    for i in 0..32 {
        st.zregs[i / 2][i % 2] = get64(vfp, VFP_D0 + 8 * i);
    }
    user::vfp_set_fpscr(cpu, st, get32(vfp, VFP_FPSCR));

    let _ = signal::restore_altstack(t, &uc[UC_STACK..UC_MCONTEXT], u64::from(st.regs[13]));
    true
}

/// `do_sigreturn()` and `do_rt_sigreturn()`.
fn sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>, rt: bool) -> i64 {
    let mut st = CpuArmState::load(cpu.env);
    let (size, uc) = if rt { (RT_FRAME_SIZE, RT_UC) } else { (FRAME_SIZE, 0) };
    // The frame is on a 64-bit boundary; anything else is the program messing with it.
    let frame = st.regs[13];
    let mut f = vec![0u8; size];
    if frame & 7 != 0 || !space.read(u64::from(frame), &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    if !sigframe_return(t, cpu, &mut st, &f[uc..]) {
        signal::force_sig(t, signal::SIGSEGV);
    }
    user::commit(cpu, &mut st);
    signal::ESIGRETURN
}

/// `do_sigreturn()`.
pub(super) fn do_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    sigreturn(space, t, cpu, false)
}

/// `do_rt_sigreturn()`.
pub(super) fn do_rt_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    sigreturn(space, t, cpu, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        // offsetof(struct sigframe, retcode[3]) and offsetof(struct rt_sigframe,
        // sig.retcode[3]), SIGFRAME_RC3_OFFSET and RT_SIGFRAME_RC3_OFFSET.
        assert_eq!(RETCODE + 12, 756);
        assert_eq!(RT_UC + RETCODE + 12, 884);
        // The register space holds the VFP frame and the terminating word.
        const { assert!(UC_REGSPACE + VFP_SIZE + 4 <= RETCODE) };
        assert_eq!(thumb_sigreturn(NR_RT_SIGRETURN), [0xad, 0x27, 0x00, 0xdf]);
    }
}
