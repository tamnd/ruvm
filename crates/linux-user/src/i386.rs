// SPDX-License-Identifier: GPL-2.0-or-later

//! The i386 target, `qemu-i386`: the x86 front end with a 32-bit code segment, the
//! `TARGET_ABI32` parts of `linux-user/i386/cpu_loop.c` and `do_set_thread_area()`. The
//! system calls go through [`compat`](crate::compat), and the signal frames are the
//! `TARGET_X86_64`-less ones of `linux-user/i386/signal.c`.

mod syscalls;

use std::sync::{Arc, OnceLock};

use ruvm_jit::cpu_exec::{cpu_exec_step_atomic, tcg_cpu_exec};
use ruvm_jit::{Cpu, Jit, JitConfig, Vcpu, excp};
use ruvm_mem::AddressSpace;
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_PE_MASK, CR0_PG_MASK, CR0_WP_MASK, CR4_FSGSBASE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK,
    CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK, DESC_DPL_SHIFT, DESC_G_MASK, DESC_P_MASK, DESC_S_MASK,
    DESC_TYPE_SHIFT, HF_CPL_MASK, HF_LMA_MASK, HF_PE_MASK, MSR_EFER_LMA, MSR_EFER_LME, R_CS, R_DS,
    R_EAX, R_EBP, R_EBX, R_ECX, R_EDI, R_EDX, R_ES, R_ESI, R_ESP, R_FS, R_GS, R_SS, SegmentCache,
};
use ruvm_target_x86::tcg::env::{self, EIP, HF_OSFXSR_MASK, SEG_BASE, SEG_SIZE, SEGS};
use ruvm_target_x86::tcg::user::{
    cpu_x86_frstor, cpu_x86_fsave, cpu_x86_fxrstor, cpu_x86_load_seg,
};
use ruvm_target_x86::tcg::{
    EXCP_SYSCALL, EXCP00_DIVZ, EXCP0B_NOSEG, EXCP0C_STACK, EXCP0D_GPF, EXCP0E_PAGE, EXCP01_DB,
    EXCP03_INT3, EXCP04_INTO, EXCP05_BOUND, EXCP06_ILLOP, X86, helper_registry, jit_config,
};
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use crate::compat;
use crate::elf::{Arch, ImageInfo};
use crate::gdb;
use crate::guest::Guest;
use crate::signal::{self, Sigaction, Task, get32, put32};
use crate::start;
use crate::strace;
use crate::syscall::{Proc, THREAD_EXIT};
use crate::x86_64::{
    self, IF_MASK, USER_DS, UserCpu, cpuid, flat_descriptor, gate64, map_rw, reg, set_reg,
};

use syscalls::SYSCALLS;

/// `TASK_UNMAPPED_BASE` for i386.
const TASK_UNMAPPED_BASE: u64 = 0x4000_0000;
/// `ELF_ET_DYN_BASE` for i386.
const ELF_ET_DYN_BASE: u64 = 0x0040_0000;
/// `EM_386` and `EM_486`, which `elf_check_arch()` both takes.
const EM_386: u16 = 3;
const EM_486: u16 = 6;
/// `__USER_CS` of a 32-bit program.
const USER_CS: u32 = 0x23;
/// `TARGET_GDT_ENTRIES`.
const GDT_ENTRIES: u64 = 9;
/// `TARGET_GDT_ENTRY_TLS_MIN` and `TARGET_GDT_ENTRY_TLS_MAX`.
const TLS_MIN: u32 = 6;
const TLS_MAX: u32 = 8;

const EINVAL: i64 = libc::EINVAL as i64;
const EFAULT: i64 = libc::EFAULT as i64;
const ENOSYS: i64 = libc::ENOSYS as i64;

/// The guest's memory, for the GDT that `cpu_set_tls()` writes.
static SPACE: OnceLock<Arc<GuestSpace>> = OnceLock::new();

/// The guest address of GDT entry `n`.
fn gdt_entry(cpu: &Cpu<'_>, n: u32) -> u64 {
    env::ld64(cpu.env, env::GDT + SEG_BASE) + 8 * u64::from(n)
}

/// `do_set_thread_area()`: installs the `struct user_desc` at `ptr` in the GDT, at the entry
/// it names or, for -1, the first free one, whose number it gets back.
fn set_thread_area(space: &GuestSpace, cpu: &Cpu<'_>, ptr: u64) -> i64 {
    let mut d = [0u8; 16];
    if !space.read(ptr, &mut d) {
        return -EFAULT;
    }
    let mut n = get32(&d, 0);
    if n == u32::MAX {
        for i in TLS_MIN..=TLS_MAX {
            let mut e = [0u8; 8];
            space.read_raw(gdt_entry(cpu, i), &mut e);
            if e == [0; 8] {
                n = i;
                if !space.write(ptr, &i.to_le_bytes()) {
                    return -EFAULT;
                }
                break;
            }
        }
    }
    if !(TLS_MIN..=TLS_MAX).contains(&n) {
        return -EINVAL;
    }
    let (base, limit, flags) = (get32(&d, 4), get32(&d, 8), get32(&d, 12));
    let seg_32bit = flags & 1;
    let contents = (flags >> 1) & 3;
    let read_exec_only = (flags >> 3) & 1;
    let limit_in_pages = (flags >> 4) & 1;
    let seg_not_present = (flags >> 5) & 1;
    let useable = (flags >> 6) & 1;
    if contents == 3 && seg_not_present == 0 {
        return -EINVAL;
    }
    // A descriptor of nothing clears the entry.
    let clear = base == 0
        && limit == 0
        && contents == 0
        && read_exec_only == 1
        && seg_32bit == 0
        && limit_in_pages == 0
        && seg_not_present == 1
        && useable == 0;
    let (e1, e2) = if clear {
        (0, 0)
    } else {
        let e1 = ((base & 0xffff) << 16) | (limit & 0xffff);
        let e2 = (base & 0xff00_0000)
            | ((base & 0x00ff_0000) >> 16)
            | (limit & 0xf_0000)
            | ((read_exec_only ^ 1) << 9)
            | (contents << 10)
            | ((seg_not_present ^ 1) << 15)
            | (seg_32bit << 22)
            | (limit_in_pages << 23)
            | (useable << 20)
            | 0x7000;
        (e1, e2)
    };
    let mut e = [0u8; 8];
    put32(&mut e, 0, e1);
    put32(&mut e, 4, e2);
    space.write_raw(gdt_entry(cpu, n), &e);
    0
}

/// `do_get_thread_area()`.
fn get_thread_area(space: &GuestSpace, cpu: &Cpu<'_>, ptr: u64) -> i64 {
    let mut d = [0u8; 16];
    if !space.read(ptr, &mut d) {
        return -EFAULT;
    }
    let n = get32(&d, 0);
    if !(TLS_MIN..=TLS_MAX).contains(&n) {
        return -EINVAL;
    }
    let mut e = [0u8; 8];
    space.read_raw(gdt_entry(cpu, n), &mut e);
    let (e1, e2) = (get32(&e, 0), get32(&e, 4));
    let flags = ((e2 >> 22) & 1)
        | (((e2 >> 10) & 3) << 1)
        | ((((e2 >> 9) & 1) ^ 1) << 3)
        | (((e2 >> 23) & 1) << 4)
        | ((((e2 >> 15) & 1) ^ 1) << 5)
        | (((e2 >> 20) & 1) << 6);
    let limit = (e1 & 0xffff) | (e2 & 0xf_0000);
    let base = (e1 >> 16) | (e2 & 0xff00_0000) | ((e2 & 0xff) << 16);
    put32(&mut d, 4, base);
    put32(&mut d, 8, limit);
    put32(&mut d, 12, flags);
    if space.write(ptr, &d) { 0 } else { -EFAULT }
}

/// `cpu_set_tls()`: `set_thread_area()` of the `struct user_desc` at `tls`, then GS again.
fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
    let space = SPACE.get().expect("the vCPU is created first");
    set_thread_area(space, cpu, tls);
    let gs = env::ld32(cpu.env, SEGS + R_GS * SEG_SIZE);
    let _ = cpu_x86_load_seg(cpu, R_GS, gs);
}

/// The register state a program starts with, `init_main_thread()` with `TARGET_ABI32`: the
/// CPU may be in long mode, but the code segment is 32-bit.
fn initial_state(
    model: &X86Cpu,
    space: &GuestSpace,
    entry: u64,
    sp: u64,
) -> Result<ruvm_target_x86::state::X86CpuState, String> {
    let mut s = model.new_state(true);
    let leaf1 = cpuid(model, 1, 0);
    let has_sse = leaf1[3] & (1 << 25) != 0;
    let has_xsave = leaf1[2] & (1 << 26) != 0;
    let has_fsgsbase = cpuid(model, 7, 0)[1] & 1 != 0;
    let is64 = cpuid(model, 0x8000_0001, 0)[3] & (1 << 29) != 0;

    // x86_cpu_reset_hold() under CONFIG_USER_ONLY.
    let mut xcr0 = 1u64;
    if has_sse {
        xcr0 |= 2;
    }
    let d = cpuid(model, 0xd, 0);
    xcr0 |= (u64::from(d[0]) | (u64::from(d[3]) << 32)) & !3;
    let mut cr4 = 0u64;
    if has_xsave {
        cr4 |= CR4_OSFXSR_MASK | CR4_OSXSAVE_MASK;
    }
    if has_fsgsbase {
        cr4 |= CR4_FSGSBASE_MASK;
    }
    s.xcr0 = xcr0;
    s.cr4 = cr4;

    s.cr0 = CR0_PG_MASK | CR0_WP_MASK | CR0_PE_MASK;
    s.hflags |= HF_PE_MASK | HF_CPL_MASK;
    if has_sse {
        s.cr4 |= CR4_OSFXSR_MASK;
        s.hflags |= HF_OSFXSR_MASK;
    }
    if is64 {
        s.cr4 |= CR4_PAE_MASK;
        s.efer |= MSR_EFER_LMA | MSR_EFER_LME;
        s.hflags |= HF_LMA_MASK;
    }
    s.rflags |= IF_MASK;
    s.regs = [0; 16];
    s.regs[R_ESP] = sp;
    s.rip = entry;

    // The IDT has the limit of 256 legacy gates, but the gates are 64-bit ones in long mode,
    // as QEMU writes them.
    let idt_limit = 255u64;
    let idt = map_rw(space, 8 * (idt_limit + 1))?;
    for n in (0..20).chain([0x80]) {
        let dpl = if n == 3 || n == 4 || n == 0x80 { 3 } else { 0 };
        if is64 {
            space.write_raw(idt + 16 * n, &gate64(dpl));
        } else {
            space.write_raw(idt + 8 * n, &gate64(dpl)[..8]);
        }
    }
    s.idt = SegmentCache { selector: 0, base: idt, limit: idt_limit as u32, flags: 0 };

    let gdt = map_rw(space, 8 * GDT_ENTRIES)?;
    let code = DESC_G_MASK
        | DESC_B_MASK
        | DESC_P_MASK
        | DESC_S_MASK
        | (3 << DESC_DPL_SHIFT)
        | (0xa << DESC_TYPE_SHIFT)
        | DESC_A_MASK;
    let data = DESC_G_MASK
        | DESC_B_MASK
        | DESC_P_MASK
        | DESC_S_MASK
        | (3 << DESC_DPL_SHIFT)
        | (0x2 << DESC_TYPE_SHIFT)
        | DESC_A_MASK;
    // helper_load_seg() sets the accessed bit in the descriptor as it loads it.
    space.write_raw(gdt + u64::from(USER_CS >> 3) * 8, &flat_descriptor(code).to_le_bytes());
    space.write_raw(gdt + u64::from(USER_DS >> 3) * 8, &flat_descriptor(data).to_le_bytes());
    s.gdt = SegmentCache { selector: 0, base: gdt, limit: (8 * GDT_ENTRIES - 1) as u32, flags: 0 };

    s.load_seg_cache(R_CS, USER_CS, 0, 0xffff_ffff, code);
    for seg in [R_SS, R_DS, R_ES, R_FS, R_GS] {
        s.load_seg_cache(seg, USER_DS, 0, 0xffff_ffff, data);
    }
    // "This hack makes Wine work..."
    s.segs[R_FS].selector = 0;
    Ok(s)
}

/// The sigreturn trampolines, `setup_sigtramp()`: QEMU has them in the vDSO.
static SIGTRAMP: OnceLock<u64> = OnceLock::new();
/// Whether the CPU has FXSR, for `get_fpstate_kind()`.
static FXSR: OnceLock<bool> = OnceLock::new();

/// `TARGET_NR_sigreturn` and `TARGET_NR_rt_sigreturn`.
const NR_SIGRETURN: u32 = 119;
const NR_RT_SIGRETURN: u32 = 173;

/// `install_sigtramp()`: `popl %eax; movl $119, %eax; int $0x80`.
fn sigtramp() -> [u8; 8] {
    let mut c = [0x58, 0xb8, 0, 0, 0, 0, 0xcd, 0x80];
    put32(&mut c, 2, NR_SIGRETURN);
    c
}

/// `install_rt_sigtramp()`: `movl $173, %eax; int $0x80`.
fn rt_sigtramp() -> [u8; 8] {
    let mut c = [0xb8, 0, 0, 0, 0, 0xcd, 0x80, 0];
    put32(&mut c, 1, NR_RT_SIGRETURN);
    c
}

/// The page of the trampolines, where QEMU loads the vDSO: `sigreturn` at the start and
/// `rt_sigreturn` 8 bytes in.
fn map_sigtramp(space: &GuestSpace) -> Result<u64, String> {
    let kind = MapKind { anon: true, ..MapKind::default() };
    let err = |e| format!("mmap: {}", crate::strerror(e));
    let addr = space.mmap(0, PAGE_SIZE, page::READ | page::WRITE, kind, None, 0).map_err(err)?;
    space.write_raw(addr, &sigtramp());
    space.write_raw(addr + 8, &rt_sigtramp());
    space.mprotect(addr, PAGE_SIZE, page::READ | page::EXEC).map_err(err)?;
    Ok(addr)
}

/// `struct sigframe`: the return address, the signal, the `sigcontext`, the unused
/// `fpstate`, the high word of the mask and the return code.
const FRAME_SIZE: u32 = 732;
const SIGFRAME_SC: usize = 8;
const EXTRAMASK: usize = 720;
const RETCODE: usize = 724;
/// `struct rt_sigframe`: the return address, the signal, the pointers to the `siginfo` and
/// the `ucontext`, which follow, and the return code.
const RT_FRAME_SIZE: u32 = 268;
const INFO: usize = 16;
const UC: usize = 144;
const UC_STACK: usize = UC + 8;
const UC_MCONTEXT: usize = UC + 20;
const UC_SIGMASK: usize = UC + 108;
const RT_RETCODE: usize = 260;
/// `struct target_sigcontext_32`, after GS, FS, ES and DS: EDI to EAX, in this order.
const SC_REGS: [usize; 8] = [R_EDI, R_ESI, R_EBP, R_ESP, R_EBX, R_EDX, R_ECX, R_EAX];
const SC_TRAPNO: usize = 48;
const SC_ERR: usize = 52;
const SC_EIP: usize = 56;
const SC_CS: usize = 60;
const SC_EFLAGS: usize = 64;
const SC_ESP_AT_SIGNAL: usize = 68;
const SC_SS: usize = 72;
const SC_FPSTATE: usize = 76;
const SC_OLDMASK: usize = 80;
const SC_CR2: usize = 84;
/// `struct target_fregs_state`, what FSAVE writes with `status` and `magic` after it.
const FREGS_SIZE: u32 = 112;

/// `FPStateKind`.
#[derive(Clone, Copy, PartialEq)]
enum FpKind {
    Fsave,
    Fxsave,
    Xsave,
}

/// `get_fpstate_kind()`.
fn fpstate_kind(cpu: &Cpu<'_>) -> FpKind {
    if x86_64::uses_xsave(cpu) {
        FpKind::Xsave
    } else if FXSR.get().copied().unwrap_or(false) {
        FpKind::Fxsave
    } else {
        FpKind::Fsave
    }
}

/// `get_fpstate_size()`.
fn fpstate_size(cpu: &Cpu<'_>, kind: FpKind) -> u32 {
    match kind {
        FpKind::Xsave => x86_64::fpstate_size(cpu, true) as u32,
        FpKind::Fxsave => 512,
        FpKind::Fsave => FREGS_SIZE,
    }
}

/// `get_sigframe()`: the frame, the FSAVE state, the FXSAVE or XSAVE state and the end.
fn get_sigframe(
    t: &Task,
    cpu: &Cpu<'_>,
    sa: &Sigaction,
    size: u32,
    kind: FpKind,
) -> (u32, u32, u32, u32) {
    let mut sp = reg(cpu, R_ESP) as u32;
    if sa.flags & signal::SA_ONSTACK != 0 {
        sp = signal::target_sigsp(t, sp.into(), sa) as u32;
    } else if env::ld32(cpu.env, SEGS + R_SS * SEG_SIZE) & 0xffff != USER_DS
        && sa.flags & signal::SA_RESTORER == 0
        && sa.restorer != 0
    {
        // The legacy stack switch.
        sp = sa.restorer as u32;
    }
    let math = fpstate_size(cpu, kind);
    sp = sp.wrapping_sub(math) & !63;
    let fpend = sp.wrapping_add(math);
    let fxstate = sp;
    if kind != FpKind::Fsave {
        sp = sp.wrapping_sub(FREGS_SIZE);
    }
    let fpstate = sp;
    sp = sp.wrapping_sub(size).wrapping_add(4) & !15;
    (sp.wrapping_sub(4), fpstate, fxstate, fpend)
}

/// `setup_sigcontext()` into `sc`, and the floating point state into guest memory; false
/// when that does not write.
#[allow(clippy::too_many_arguments)]
fn setup_sigcontext(
    space: &GuestSpace,
    cpu: &mut Cpu<'_>,
    sc: &mut [u8],
    mask: u32,
    kind: FpKind,
    fpstate: u32,
    fxstate: u32,
    fpend: u32,
) -> bool {
    let sel = |cpu: &Cpu<'_>, s: usize| env::ld32(cpu.env, SEGS + s * SEG_SIZE);
    for (i, s) in [R_GS, R_FS, R_ES, R_DS].into_iter().enumerate() {
        put32(sc, 4 * i, sel(cpu, s));
    }
    for (i, r) in SC_REGS.into_iter().enumerate() {
        put32(sc, 16 + 4 * i, reg(cpu, r) as u32);
    }
    put32(sc, SC_TRAPNO, cpu.core.exception_index as u32);
    put32(sc, SC_ERR, env::ld32(cpu.env, env::ERROR_CODE));
    put32(sc, SC_EIP, env::ld64(cpu.env, EIP) as u32);
    put32(sc, SC_CS, sel(cpu, R_CS));
    put32(sc, SC_EFLAGS, env::ld64(cpu.env, env::EFLAGS) as u32);
    put32(sc, SC_ESP_AT_SIGNAL, reg(cpu, R_ESP) as u32);
    put32(sc, SC_SS, sel(cpu, R_SS));

    // FSAVE resets the FPU, and QEMU saves the FXSAVE or XSAVE state after it.
    let (fp, fx) = (u64::from(fpstate), u64::from(fxstate));
    // status repeats the status word; magic is 0xffff when FXSAVE or XSAVE state follows.
    let mut tail = [0u8; 4];
    if kind != FpKind::Fsave {
        tail[2..].fill(0xff);
    }
    let ok = cpu_x86_fsave(cpu, fp).is_ok()
        && space.read(fp + 4, &mut tail[..2])
        && space.write_raw(fp + u64::from(FREGS_SIZE) - 4, &tail)
        && (kind == FpKind::Fsave
            || x86_64::save_fpstate(space, cpu, kind == FpKind::Xsave, fp, fx, fpend.into()));
    put32(sc, SC_FPSTATE, fpstate);
    put32(sc, SC_OLDMASK, mask);
    put32(sc, SC_CR2, env::ld64(cpu.env, env::cr(2)) as u32);
    ok
}

/// The registers both frames enter the handler with.
fn enter_handler(cpu: &mut Cpu<'_>, frame: u32, handler: u64, sig: i32, rt: bool) {
    set_reg(cpu, R_ESP, frame.into());
    env::st64(cpu.env, EIP, handler & 0xffff_ffff);
    set_reg(cpu, R_EAX, sig as u64);
    let (edx, ecx) = if rt { (frame + INFO as u32, frame + UC as u32) } else { (0, 0) };
    set_reg(cpu, R_EDX, edx.into());
    set_reg(cpu, R_ECX, ecx.into());
    let segs = if rt {
        [(R_DS, USER_DS), (R_ES, USER_DS), (R_CS, USER_CS), (R_SS, USER_DS)]
    } else {
        [(R_DS, USER_DS), (R_ES, USER_DS), (R_SS, USER_DS), (R_CS, USER_CS)]
    };
    for (seg, s) in segs {
        if cpu_x86_load_seg(cpu, seg, s).is_err() {
            break;
        }
    }
    let fl = env::ld64(cpu.env, env::EFLAGS);
    env::st64(cpu.env, env::EFLAGS, fl & !x86_64::TF_MASK);
}

/// `setup_frame()`: the frame of a handler without `SA_SIGINFO`. `old` is the guest mask to
/// return to.
fn setup_frame(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    _info: &signal::Info,
    old: u64,
) {
    let kind = fpstate_kind(cpu);
    let (frame, fpstate, fxstate, fpend) = get_sigframe(t, cpu, sa, FRAME_SIZE, kind);
    let mut f = [0u8; FRAME_SIZE as usize];
    let addr = u64::from(frame);
    if fpend < frame
        || !space.check(addr, u64::from(fpend - frame), page::WRITE)
        || !space.read_raw(addr, &mut f)
    {
        signal::force_sigsegv(t, sig);
        return;
    }
    let sc = &mut f[SIGFRAME_SC..SIGFRAME_SC + 88];
    if !setup_sigcontext(space, cpu, sc, old as u32, kind, fpstate, fxstate, fpend) {
        cpu.core.exception_index = -1;
        signal::force_sigsegv(t, sig);
        return;
    }
    put32(&mut f, EXTRAMASK, (old >> 32) as u32);
    if sa.flags & signal::SA_RESTORER != 0 {
        put32(&mut f, 0, sa.restorer as u32);
    } else {
        f[RETCODE..].copy_from_slice(&sigtramp());
        put32(&mut f, 0, *SIGTRAMP.get().expect("the vCPU is created first") as u32);
    }
    put32(&mut f, 4, sig as u32);
    space.write_raw(addr, &f);
    enter_handler(cpu, frame, sa.handler, sig, false);
}

/// `setup_rt_frame()`.
fn setup_rt_frame(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    info: &signal::Info,
    old: u64,
) {
    let kind = fpstate_kind(cpu);
    let (frame, fpstate, fxstate, fpend) = get_sigframe(t, cpu, sa, RT_FRAME_SIZE, kind);
    let mut f = [0u8; RT_FRAME_SIZE as usize];
    let addr = u64::from(frame);
    if fpend < frame
        || !space.check(addr, u64::from(fpend - frame), page::WRITE)
        || !space.read_raw(addr, &mut f)
    {
        signal::force_sigsegv(t, sig);
        return;
    }
    if sa.flags & signal::SA_SIGINFO != 0 {
        f[INFO..UC].copy_from_slice(&signal::guest_siginfo(info));
    }
    put32(&mut f, UC, u32::from(kind == FpKind::Xsave));
    put32(&mut f, UC + 4, 0);
    let esp = reg(cpu, R_ESP) & 0xffff_ffff;
    f[UC_STACK..UC_MCONTEXT].copy_from_slice(&signal::save_altstack(t, esp)[..12]);
    let sc = &mut f[UC_MCONTEXT..UC_SIGMASK];
    if !setup_sigcontext(space, cpu, sc, old as u32, kind, fpstate, fxstate, fpend) {
        cpu.core.exception_index = -1;
        signal::force_sigsegv(t, sig);
        return;
    }
    put32(&mut f, UC_SIGMASK, old as u32);
    put32(&mut f, UC_SIGMASK + 4, (old >> 32) as u32);
    if sa.flags & signal::SA_RESTORER != 0 {
        put32(&mut f, 0, sa.restorer as u32);
    } else {
        f[RT_RETCODE..].copy_from_slice(&rt_sigtramp());
        put32(&mut f, 0, *SIGTRAMP.get().expect("the vCPU is created first") as u32 + 8);
    }
    put32(&mut f, 4, sig as u32);
    put32(&mut f, 8, frame + INFO as u32);
    put32(&mut f, 12, frame + UC as u32);
    space.write_raw(addr, &f);
    enter_handler(cpu, frame, sa.handler, sig, true);
}

/// `restore_sigcontext()` from `sc`, false when the floating point state does not load.
fn restore_sigcontext(space: &GuestSpace, cpu: &mut Cpu<'_>, sc: &[u8]) -> bool {
    // A selector that does not load leaves a #GP for the next instruction.
    let seg = |i: usize| u32::from(get32(sc, i) as u16);
    let _ = [R_GS, R_FS, R_ES, R_DS]
        .into_iter()
        .enumerate()
        .try_for_each(|(i, s)| cpu_x86_load_seg(cpu, s, seg(4 * i)));
    for (i, r) in SC_REGS.into_iter().enumerate() {
        set_reg(cpu, r, get32(sc, 16 + 4 * i).into());
    }
    env::st64(cpu.env, EIP, get32(sc, SC_EIP).into());
    if cpu_x86_load_seg(cpu, R_CS, seg(SC_CS) | 3).is_ok() {
        let _ = cpu_x86_load_seg(cpu, R_SS, seg(SC_SS) | 3);
    }
    let fl = env::ld64(cpu.env, env::EFLAGS);
    let tmp = u64::from(get32(sc, SC_EFLAGS));
    let fix = x86_64::FIX_EFLAGS;
    env::st64(cpu.env, env::EFLAGS, (fl & !fix) | (tmp & fix));

    let fp = u64::from(get32(sc, SC_FPSTATE));
    if fp == 0 {
        return true;
    }
    // frstor_sigcontext(): the FXSAVE or XSAVE state, then the FSAVE state over it.
    let kind = fpstate_kind(cpu);
    let mut math = u64::from(fpstate_size(cpu, kind));
    if kind != FpKind::Fsave {
        math += u64::from(FREGS_SIZE);
    }
    if !space.check(fp, math, page::READ) {
        return false;
    }
    let fx = fp + u64::from(FREGS_SIZE);
    let fx_ok = match kind {
        FpKind::Xsave => x86_64::restore_fpstate(space, cpu, fx),
        FpKind::Fxsave => cpu_x86_fxrstor(cpu, fx).is_ok(),
        FpKind::Fsave => true,
    };
    fx_ok && cpu_x86_frstor(cpu, fp).is_ok()
}

/// `do_sigreturn()`.
fn do_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    let frame = reg(cpu, R_ESP).wrapping_sub(8) & 0xffff_ffff;
    let mut f = [0u8; FRAME_SIZE as usize];
    if !space.read(frame, &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    let sc = &f[SIGFRAME_SC..SIGFRAME_SC + 88];
    let mask = u64::from(get32(sc, SC_OLDMASK)) | (u64::from(get32(&f, EXTRAMASK)) << 32);
    signal::set_sigmask(t, signal::t2h_set(mask));
    if !restore_sigcontext(space, cpu, sc) {
        signal::force_sig(t, signal::SIGSEGV);
    }
    signal::ESIGRETURN
}

/// `do_rt_sigreturn()`.
fn do_rt_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    let frame = reg(cpu, R_ESP).wrapping_sub(4) & 0xffff_ffff;
    let mut f = [0u8; RT_FRAME_SIZE as usize];
    if !space.read(frame, &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    let mask = u64::from(get32(&f, UC_SIGMASK)) | (u64::from(get32(&f, UC_SIGMASK + 4)) << 32);
    signal::set_sigmask(t, signal::t2h_set(mask));
    if !restore_sigcontext(space, cpu, &f[UC_MCONTEXT..UC_SIGMASK]) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    let sp = reg(cpu, R_ESP);
    let _ = signal::restore_altstack(t, &f[UC_STACK..UC_MCONTEXT], sp);
    signal::ESIGRETURN
}

/// The strace entry of i386 call `nr`: the host's when the call is the host's, else the name.
fn strace_entry(nr: u64) -> Option<&'static strace::Entry> {
    static NAMED: OnceLock<Vec<strace::Entry>> = OnceLock::new();
    let &(name, host) = SYSCALLS.get(nr as usize)?;
    strace::entry(i64::from(host)).filter(|e| e.name() == name).or_else(|| {
        NAMED.get_or_init(|| strace::named(SYSCALLS.iter().map(|s| s.0))).get(nr as usize)
    })
}

/// `do_syscall()` of i386 call `nr`.
fn do_syscall(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>, nr: u64, a: [u64; 6]) -> i64 {
    let Some(&(name, host)) = SYSCALLS.get(nr as usize) else { return -ENOSYS };
    match name {
        "set_thread_area" => set_thread_area(p.space(), cpu, a[0]),
        "get_thread_area" => get_thread_area(p.space(), cpu, a[0]),
        "sigreturn" => {
            if signal::block_signals(t) {
                return signal::ERESTARTSYS;
            }
            do_sigreturn(p.space(), t, cpu)
        }
        _ => compat::do_syscall(p, t, cpu, name, (host >= 0).then_some(i64::from(host)), a),
    }
}

/// `cpu_loop()`, until the thread calls `exit` with others left.
pub(crate) fn cpu_loop(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
        cpu.process_queued_cpu_work();
        match trapnr {
            0x80 | EXCP_SYSCALL => {
                let nr = reg(cpu, R_EAX) & 0xffff_ffff;
                let args =
                    [R_EBX, R_ECX, R_EDX, R_ESI, R_EDI, R_EBP].map(|r| reg(cpu, r) & 0xffff_ffff);
                gdb::set_orig_ax(nr);
                gdb::syscall_entry(t, cpu, nr);
                let traced =
                    strace::on().then(|| strace::syscall(p.space(), nr, strace_entry(nr), &args));
                let ret = do_syscall(p, t, cpu, nr, args);
                if let Some(e) = traced {
                    strace::syscall_ret(p.space(), e, ret, &args);
                }
                if ret == THREAD_EXIT {
                    return;
                }
                gdb::syscall_return(t, cpu, nr);
                if ret == signal::ERESTARTSYS {
                    let eip = env::ld64(cpu.env, EIP);
                    env::st64(cpu.env, EIP, eip.wrapping_sub(2) & 0xffff_ffff);
                } else if ret != signal::ESIGRETURN {
                    set_reg(cpu, R_EAX, u64::from(ret as u32));
                }
            }
            EXCP0B_NOSEG | EXCP0C_STACK => signal::force_sig(t, signal::SIGBUS),
            EXCP0D_GPF | EXCP04_INTO | EXCP05_BOUND => signal::force_sig(t, signal::SIGSEGV),
            EXCP0E_PAGE => {
                let code = if env::ld32(cpu.env, env::ERROR_CODE) & 1 != 0 {
                    signal::SEGV_ACCERR
                } else {
                    signal::SEGV_MAPERR
                };
                let addr = env::ld64(cpu.env, env::cr(2));
                signal::force_sig_fault(t, signal::SIGSEGV, code, addr);
            }
            EXCP00_DIVZ => {
                let eip = env::ld64(cpu.env, EIP);
                signal::force_sig_fault(t, signal::SIGFPE, signal::FPE_INTDIV, eip);
            }
            EXCP01_DB | excp::DEBUG => {
                let eip = env::ld64(cpu.env, EIP);
                signal::force_sig_fault(t, signal::SIGTRAP, signal::TRAP_BRKPT, eip);
            }
            EXCP03_INT3 => signal::force_sig(t, signal::SIGTRAP),
            EXCP06_ILLOP => {
                let eip = env::ld64(cpu.env, EIP);
                signal::force_sig_fault(t, signal::SIGILL, signal::ILL_ILLOPN, eip);
            }
            excp::INTERRUPT => {}
            excp::ATOMIC => cpu_exec_step_atomic(cpu),
            _ => {
                eprintln!(
                    "qemu: unhandled CPU exception 0x{trapnr:x} - aborting\nEIP={:08x}",
                    env::ld64(cpu.env, EIP)
                );
                std::process::abort();
            }
        }
        let space = Arc::clone(p.space());
        signal::process_pending_signals(&space, t, cpu);
    }
}

/// The i386 target.
pub(crate) static GUEST: Guest = Guest {
    machine: "i686",
    minsigstksz: 2048,
    env_size: env::ENV_SIZE,
    generic_abi: false,
    abi32: true,
    eabi: false,
    sa_restorer: true,
    open_flags: &[],
    cpuinfo: None,
    sp: |cpu| reg(cpu, R_ESP),
    clone_regs: |cpu, newsp| {
        if newsp != 0 {
            set_reg(cpu, R_ESP, newsp);
        }
        set_reg(cpu, R_EAX, 0);
    },
    set_tls,
    cpu_loop,
    setup_frame: Some(setup_frame),
    setup_rt_frame,
    rt_sigreturn: do_rt_sigreturn,
};

/// `qemu-i386`: the model `-cpu` names.
#[derive(Default)]
pub(crate) struct Target {
    model: Option<X86Cpu>,
}

impl Target {
    fn model(&self) -> &X86Cpu {
        self.model.as_ref().expect("the CPU is selected first")
    }
}

impl start::Target for Target {
    fn guest(&self) -> &'static Guest {
        &GUEST
    }

    fn name(&self) -> &'static str {
        "i386"
    }

    fn layout(&self) -> (u64, u64) {
        (TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE)
    }

    fn max_reserve(&self) -> u64 {
        1 << 32
    }

    fn select_cpu(&mut self, cpu: &str) -> Result<(), String> {
        let (model_name, features) = cpu.split_once(',').unwrap_or((cpu, ""));
        let mut c = X86Cpu::new(model_name, Accel::Tcg).map_err(|e| e.to_string())?;
        if !features.is_empty() {
            c.parse_features(features).map_err(|e| e.to_string())?;
        }
        c.realize().map_err(|e| e.to_string())?;
        self.model = Some(c);
        Ok(())
    }

    fn arch(&self) -> Arch {
        // get_elf_platform(): the family, from i386 to i686.
        const PLATFORMS: [&str; 4] = ["i386", "i486", "i586", "i686"];
        let eax = cpuid(self.model(), 1, 0)[0];
        let mut family = (eax >> 8) & 0xf;
        if family == 0xf {
            family += (eax >> 20) & 0xff;
        }
        Arch {
            machines: &[EM_386, EM_486],
            elf32: true,
            platform: Some(PLATFORMS[family.clamp(3, 6) as usize - 3]),
            hwcap: u64::from(cpuid(self.model(), 1, 0)[3]),
            hwcap2: None,
        }
    }

    fn new_jit(&self, config: &dyn Fn(&mut JitConfig)) -> Arc<Jit> {
        let mut c = jit_config();
        config(&mut c);
        let backend = ruvm_jit::host_backend(helper_registry(), c.code_gen_buffer_size);
        Jit::new(c, backend)
    }

    fn create_vcpu(
        &mut self,
        jit: &Arc<Jit>,
        space: &Arc<GuestSpace>,
        as_: Arc<AddressSpace>,
        info: &ImageInfo,
    ) -> Result<Vcpu, String> {
        let model = self.model.take().expect("the CPU is selected first");
        let _ = FXSR.set(cpuid(&model, 1, 0)[3] & (1 << 24) != 0);
        let _ = SIGTRAMP.set(map_sigtramp(space)?);
        let state = initial_state(&model, space, info.entry, info.start_stack)?;
        let _ = SPACE.set(Arc::clone(space));
        let x86 = Arc::new(X86::new(model).with_user_mode());
        let ops = Arc::new(UserCpu { x86, space: Arc::clone(space) });
        let mut v = jit.create_vcpu(ops, as_, env::ENV_SIZE);
        env::load_state(&mut v.env, &state);
        Ok(v)
    }
}
