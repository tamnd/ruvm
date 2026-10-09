// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86_64 target: `linux-user/i386/cpu_loop.c`, `target_cpu_copy_regs()`, the user mode
//! parts of `x86_cpu_reset_hold()`, `do_arch_prctl()` and the signal frames of
//! `linux-user/i386/signal.c`.

use std::fmt;
use std::sync::Arc;

use ruvm_jit::cpu_exec::{cpu_exec_step_atomic, tcg_cpu_exec};
use ruvm_jit::cputlb::tlb_set_page;
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, Jit, JitConfig, MmuAccessType, Ra, Tb, TbCpuState, Vcpu, Watchpoint,
    excp,
};
use ruvm_jit_core::Type;
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_PE_MASK, CR0_PG_MASK, CR0_WP_MASK, CR4_FSGSBASE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK,
    CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK, DESC_DPL_SHIFT, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK,
    DESC_S_MASK, DESC_TYPE_SHIFT, HF_CPL_MASK, HF_LMA_MASK, HF_PE_MASK, MSR_EFER_LMA, MSR_EFER_LME,
    R_CS, R_DS, R_EAX, R_EBP, R_EBX, R_ECX, R_EDI, R_EDX, R_ES, R_ESI, R_ESP, R_FS, R_GS, R_SS,
    SegmentCache,
};
use ruvm_target_x86::tcg::env::{self, EIP, HF_OSFXSR_MASK, REGS, SEG_BASE, SEG_SIZE, SEGS};
use ruvm_target_x86::tcg::user::{
    cpu_x86_fxrstor, cpu_x86_fxsave, cpu_x86_load_seg, cpu_x86_xrstor, cpu_x86_xsave,
    record_sigsegv, xsave_area_size,
};
use ruvm_target_x86::tcg::{
    EXCP_SYSCALL, EXCP00_DIVZ, EXCP0B_NOSEG, EXCP0C_STACK, EXCP0D_GPF, EXCP0E_PAGE, EXCP01_DB,
    EXCP03_INT3, EXCP04_INTO, EXCP05_BOUND, EXCP06_ILLOP, X86, helper_registry, jit_config,
};
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use crate::elf::{Arch, ImageInfo};
use crate::guest::Guest;
use crate::signal::{self, Sigaction, Task, get32, get64, put32, put64};
use crate::start;
use crate::syscall::{self, Proc, THREAD_EXIT};

/// `TASK_UNMAPPED_BASE` for x86_64, `TASK_SIZE / 3` page aligned.
const TASK_UNMAPPED_BASE: u64 = 0x2aaa_aaaa_b000;
/// `ELF_ET_DYN_BASE` for x86_64, `TASK_SIZE / 3 * 2`.
const ELF_ET_DYN_BASE: u64 = 0x5555_5555_6000;
/// `EM_X86_64`.
const EM_X86_64: u16 = 62;
/// `__USER_CS`.
const USER_CS: u32 = 0x33;
/// `__USER_DS`.
const USER_DS: u32 = 0x2b;
/// `TARGET_GDT_ENTRIES`.
const GDT_ENTRIES: u64 = 16;
/// `IF_MASK`.
const IF_MASK: u64 = 1 << 9;

/// The vCPU of a user mode guest: the x86 front end, with guest pages checked against the
/// guest's mappings instead of page tables.
struct UserCpu {
    x86: Arc<X86>,
    space: Arc<GuestSpace>,
}

impl fmt::Debug for UserCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserCpu").field("x86", &self.x86).finish_non_exhaustive()
    }
}

impl CpuOps for UserCpu {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        self.x86.translate_code(cpu, tb)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        self.x86.get_tb_cpu_state(cpu)
    }

    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        self.x86.synchronize_from_tb(cpu, tb);
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; INSN_START_WORDS]) {
        self.x86.restore_state_to_opc(cpu, tb, data);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        self.x86.set_pc(cpu, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        self.x86.get_pc(cpu)
    }

    fn cpu_exec_enter(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_enter(cpu);
    }

    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_exit(cpu);
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        self.x86.cpu_exec_interrupt(cpu, interrupt_request)
    }

    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.x86.cpu_exec_halt(cpu)
    }

    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        self.x86.cpu_exec_reset(cpu);
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.x86.do_interrupt(cpu);
    }

    fn fake_user_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.x86.fake_user_interrupt(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        self.x86.has_work(cpu)
    }

    /// The page is there when the guest mapped it with the access asked for. As on an x86
    /// host, anything mapped can be read.
    fn tlb_fill(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        _size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        probe: bool,
        ra: Ra,
    ) -> Result<bool, CpuLoopExit> {
        let flags = self.space.page_flags(addr) & page::RWX;
        let ok = match access_type {
            MmuAccessType::DataLoad => flags != 0,
            MmuAccessType::DataStore => flags & page::WRITE != 0,
            MmuAccessType::InstFetch => flags & page::EXEC != 0,
        };
        if ok {
            let a = addr & !(PAGE_SIZE - 1);
            tlb_set_page(cpu, a, a, flags | ruvm_jit::page::READ, mmu_idx, PAGE_SIZE);
            return Ok(true);
        }
        if probe {
            return Ok(false);
        }
        // QEMU touches the host address, and the host reports a non-canonical one with a #GP,
        // whose siginfo has no address.
        let canonical = (addr as i64) << 16 >> 16 == addr as i64;
        let fault = if canonical { addr } else { 0 };
        Err(record_sigsegv(cpu, fault, access_type, flags == 0, ra))
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        self.x86.do_unaligned_access(cpu, addr, access_type, mmu_idx, ra)
    }

    fn do_transaction_failed(
        &self,
        cpu: &mut Cpu<'_>,
        physaddr: u64,
        addr: u64,
        size: usize,
        access_type: MmuAccessType,
        mmu_idx: usize,
        attrs: MemTxAttrs,
        response: MemTxResult,
        ra: Ra,
    ) -> Result<(), CpuLoopExit> {
        self.x86.do_transaction_failed(
            cpu,
            physaddr,
            addr,
            size,
            access_type,
            mmu_idx,
            attrs,
            response,
            ra,
        )
    }

    fn mmu_index(&self, cpu: &Cpu<'_>, ifetch: bool) -> usize {
        self.x86.mmu_index(cpu, ifetch)
    }

    fn pointer_wrap(&self, cpu: &Cpu<'_>, mmu_idx: usize, result: u64, base: u64) -> u64 {
        self.x86.pointer_wrap(cpu, mmu_idx, result, base)
    }

    fn debug_excp_handler(&self, cpu: &mut Cpu<'_>) {
        self.x86.debug_excp_handler(cpu);
    }

    fn debug_check_watchpoint(&self, cpu: &mut Cpu<'_>, wp: &Watchpoint) -> bool {
        self.x86.debug_check_watchpoint(cpu, wp)
    }

    fn debug_check_breakpoint(&self, cpu: &mut Cpu<'_>) -> bool {
        self.x86.debug_check_breakpoint(cpu)
    }

    fn adjust_watchpoint_address(&self, cpu: &mut Cpu<'_>, addr: u64, len: u64) -> u64 {
        self.x86.adjust_watchpoint_address(cpu, addr, len)
    }

    fn guest_default_memory_order(&self) -> u32 {
        self.x86.guest_default_memory_order()
    }

    fn addr_type(&self) -> Type {
        self.x86.addr_type()
    }

    fn precise_smc(&self) -> bool {
        self.x86.precise_smc()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.x86.as_any()
    }
}

/// `reg(n)`: general register `n` of the vCPU.
pub(crate) fn reg(cpu: &Cpu<'_>, n: usize) -> u64 {
    env::ld64(cpu.env, REGS + 8 * n)
}

/// Sets general register `n`.
pub(crate) fn set_reg(cpu: &mut Cpu<'_>, n: usize, v: u64) {
    env::st64(cpu.env, REGS + 8 * n, v);
}

/// `do_arch_prctl()`.
pub(crate) fn arch_prctl(space: &GuestSpace, cpu: &mut Cpu<'_>, code: u64, addr: u64) -> i64 {
    const ARCH_SET_GS: u64 = 0x1001;
    const ARCH_SET_FS: u64 = 0x1002;
    const ARCH_GET_FS: u64 = 0x1003;
    const ARCH_GET_GS: u64 = 0x1004;
    let seg = match code {
        ARCH_SET_GS | ARCH_GET_GS => R_GS,
        ARCH_SET_FS | ARCH_GET_FS => R_FS,
        _ => return -i64::from(libc::EINVAL),
    };
    let at = SEGS + seg * SEG_SIZE;
    if code == ARCH_SET_GS || code == ARCH_SET_FS {
        // cpu_x86_load_seg(env, idx, 0), then the base.
        env::st32(cpu.env, at, 0);
        env::st64(cpu.env, at + SEG_BASE, addr);
        env::st32(cpu.env, at + env::SEG_LIMIT, 0);
        env::st32(cpu.env, at + env::SEG_FLAGS, 0);
        0
    } else if space.write(addr, &env::ld64(cpu.env, at + SEG_BASE).to_le_bytes()) {
        0
    } else {
        -i64::from(libc::EFAULT)
    }
}

/// Sets the FS base of a new thread or process, `CLONE_SETTLS`.
fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
    env::st64(cpu.env, SEGS + R_FS * SEG_SIZE + SEG_BASE, tls);
}

/// `set_gate64()` for an interrupt gate with no handler, which is all user mode needs.
fn gate64(dpl: u32) -> [u8; 16] {
    let e2: u32 = 0x8000 | (dpl << 13);
    let mut b = [0u8; 16];
    b[4..8].copy_from_slice(&e2.to_le_bytes());
    b
}

/// `write_dt()` for a flat 4 GiB segment.
fn flat_descriptor(flags: u32) -> u64 {
    let e1: u32 = 0xffff;
    let e2: u32 = 0x000f_0000 | flags;
    u64::from(e1) | (u64::from(e2) << 32)
}

fn map_rw(space: &GuestSpace, len: u64) -> Result<u64, String> {
    let kind = MapKind { anon: true, ..MapKind::default() };
    space
        .mmap(0, len, page::READ | page::WRITE, kind, None, 0)
        .map_err(|e| format!("mmap: {}", crate::strerror(e)))
}

/// `cpuid(index, 0)` of the model as `[eax, ebx, ecx, edx]`.
fn cpuid(model: &X86Cpu, index: u32, count: u32) -> [u32; 4] {
    model.cpuid(index, count)
}

/// The register state a program starts with: the user mode reset of `x86_cpu_reset_hold()`,
/// then `target_cpu_copy_regs()` with its IDT and GDT mapped into the guest.
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
    let has_lm = cpuid(model, 0x8000_0001, 0)[3] & (1 << 29) != 0;

    // x86_cpu_reset_hold() under CONFIG_USER_ONLY: every state component the CPU has.
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

    // target_cpu_copy_regs().
    s.cr0 = CR0_PG_MASK | CR0_WP_MASK | CR0_PE_MASK;
    s.hflags |= HF_PE_MASK | HF_CPL_MASK;
    if has_sse {
        s.cr4 |= CR4_OSFXSR_MASK;
        s.hflags |= HF_OSFXSR_MASK;
    }
    if !has_lm {
        return Err("The selected x86 CPU does not support 64 bit mode".into());
    }
    s.cr4 |= CR4_PAE_MASK;
    s.efer |= MSR_EFER_LMA | MSR_EFER_LME;
    s.hflags |= HF_LMA_MASK;
    s.rflags |= IF_MASK;
    s.regs = [0; 16];
    s.regs[R_ESP] = sp;
    s.rip = entry;

    // The IDT: 256 gates, the ones a program may raise itself with DPL 3.
    let idt_limit = 511u64;
    let idt = map_rw(space, 8 * (idt_limit + 1))?;
    for n in (0..20).chain([0x80]) {
        let dpl = if n == 3 || n == 4 || n == 0x80 { 3 } else { 0 };
        space.write_raw(idt + 16 * n, &gate64(dpl));
    }
    s.idt = SegmentCache { selector: 0, base: idt, limit: idt_limit as u32, flags: 0 };

    let gdt = map_rw(space, 8 * GDT_ENTRIES)?;
    let code = DESC_G_MASK
        | DESC_B_MASK
        | DESC_P_MASK
        | DESC_S_MASK
        | DESC_L_MASK
        | (3 << DESC_DPL_SHIFT)
        | (0xa << DESC_TYPE_SHIFT);
    let data = DESC_G_MASK
        | DESC_B_MASK
        | DESC_P_MASK
        | DESC_S_MASK
        | (3 << DESC_DPL_SHIFT)
        | (0x2 << DESC_TYPE_SHIFT);
    // helper_load_seg() sets the accessed bit in the descriptor as it loads it.
    space.write_raw(
        gdt + u64::from(USER_CS >> 3) * 8,
        &flat_descriptor(code | DESC_A_MASK).to_le_bytes(),
    );
    space.write_raw(
        gdt + u64::from(USER_DS >> 3) * 8,
        &flat_descriptor(data | DESC_A_MASK).to_le_bytes(),
    );
    s.gdt = SegmentCache { selector: 0, base: gdt, limit: (8 * GDT_ENTRIES - 1) as u32, flags: 0 };

    s.load_seg_cache(R_CS, USER_CS, 0, 0xffff_ffff, code | DESC_A_MASK);
    s.load_seg_cache(R_SS, USER_DS, 0, 0xffff_ffff, data | DESC_A_MASK);
    for seg in [R_DS, R_ES, R_FS, R_GS] {
        s.load_seg_cache(seg, 0, 0, 0, 0);
    }
    Ok(s)
}

/// The size of `struct rt_sigframe`: the return address, the `ucontext` and the `siginfo`.
const FRAME_SIZE: u64 = 440;
/// Where the `ucontext`, its `stack_t`, `sigcontext` and signal mask, and the `siginfo` are.
const UC: usize = 8;
const UC_STACK: usize = 24;
const UC_SIGMASK: usize = 304;
const INFO: usize = 312;
/// The `sigcontext` slots of the general registers, in `REGS` order.
const SC_REGS: [usize; 16] =
    [152, 160, 144, 136, 168, 128, 120, 112, 48, 56, 64, 72, 80, 88, 96, 104];
const SC_RIP: usize = 176;
const SC_EFLAGS: usize = 184;
const SC_CS: usize = 192;
const SC_SS: usize = 198;
const SC_ERR: usize = 200;
const SC_TRAPNO: usize = 208;
const SC_OLDMASK: usize = 216;
const SC_CR2: usize = 224;
const SC_FPSTATE: usize = 232;
/// `sw_reserved` in the FXSAVE image, `struct _fpx_sw_bytes`.
const FP_SW: u64 = 464;
const FP_XSTATE_MAGIC1: u32 = 0x4650_5853;
const FP_XSTATE_MAGIC2: u32 = 0x4650_5845;
/// The legacy area and the XSAVE header.
const XSAVE_MIN: u64 = 512 + 64;
/// `TF_MASK`.
const TF_MASK: u64 = 1 << 8;
/// The flags `sigreturn()` takes from the frame: CF, PF, AF, ZF, SF, TF, DF, OF and AC.
const FIX_EFLAGS: u64 = 0x40DD5;

/// `get_fpstate_kind()`: whether frames carry XSAVE state rather than FXSAVE state.
fn uses_xsave(cpu: &Cpu<'_>) -> bool {
    env::ld64(cpu.env, env::cr(4)) & CR4_OSXSAVE_MASK != 0
}

/// `get_fpstate_size()`.
fn fpstate_size(cpu: &Cpu<'_>, xsave: bool) -> u64 {
    if xsave { xsave_area_size(cpu, env::ld64(cpu.env, env::XCR0)) as u64 + 4 } else { 512 }
}

/// `setup_rt_frame()`: the frame for the handler of `sig` on the guest stack, and the
/// registers that enter it. `old` is the guest mask to return to.
fn setup_rt_frame(
    space: &GuestSpace,
    t: &mut Task,
    cpu: &mut Cpu<'_>,
    sig: i32,
    sa: &Sigaction,
    info: &signal::Info,
    old: u64,
) {
    let xsave = uses_xsave(cpu);
    // get_sigframe(): below the red zone, or on the alternate stack.
    let rsp = reg(cpu, R_ESP);
    let mut sp = signal::target_sigsp(t, rsp.wrapping_sub(128), sa);
    let math = fpstate_size(cpu, xsave);
    sp = sp.wrapping_sub(math) & !63;
    let fpstate = sp;
    let fpend = sp.wrapping_add(math);
    sp = sp.wrapping_sub(FRAME_SIZE).wrapping_add(8) & !15;
    let frame = sp.wrapping_sub(8);

    let mut f = [0u8; FRAME_SIZE as usize];
    if fpend < frame
        || !space.check(frame, fpend - frame, page::WRITE)
        || !space.read_raw(frame, &mut f)
    {
        signal::force_sigsegv(t, sig);
        return;
    }
    if sa.flags & signal::SA_SIGINFO != 0 {
        f[INFO..].copy_from_slice(info);
    }
    put64(&mut f, UC, u64::from(xsave));
    put64(&mut f, UC + 8, 0);
    f[UC_STACK..UC_STACK + 24].copy_from_slice(&signal::save_altstack(t, rsp));

    // setup_sigcontext().
    for (n, at) in SC_REGS.iter().enumerate() {
        put64(&mut f, *at, reg(cpu, n));
    }
    put64(&mut f, SC_TRAPNO, cpu.core.exception_index as i64 as u64);
    put64(&mut f, SC_ERR, env::ld32(cpu.env, env::ERROR_CODE) as i32 as i64 as u64);
    put64(&mut f, SC_RIP, env::ld64(cpu.env, EIP));
    put64(&mut f, SC_EFLAGS, env::ld64(cpu.env, env::EFLAGS));
    let sel = |cpu: &Cpu<'_>, s: usize| env::ld32(cpu.env, SEGS + s * SEG_SIZE) as u16;
    f[SC_CS..SC_CS + 2].copy_from_slice(&sel(cpu, R_CS).to_le_bytes());
    f[SC_CS + 2..SC_SS].fill(0);
    f[SC_SS..SC_SS + 2].copy_from_slice(&sel(cpu, R_SS).to_le_bytes());
    let fp_ok = if xsave {
        // xsave_sigcontext(): XSAVE adds to the header, so it starts out zero.
        let xcr0 = env::ld64(cpu.env, env::XCR0);
        let xstate_size = fpend - fpstate - 4;
        space.write_raw(fpstate + 512, &[0u8; 64]) && cpu_x86_xsave(cpu, fpstate, xcr0).is_ok() && {
            let mut sw = [0u8; 24];
            put32(&mut sw, 0, FP_XSTATE_MAGIC1);
            put32(&mut sw, 4, (fpend - fpstate) as u32);
            put64(&mut sw, 8, xcr0);
            put32(&mut sw, 16, xstate_size as u32);
            space.write_raw(fpstate + FP_SW, &sw)
                && space.write_raw(fpstate + xstate_size, &FP_XSTATE_MAGIC2.to_le_bytes())
        }
    } else {
        cpu_x86_fxsave(cpu, fpstate).is_ok() && space.write_raw(fpstate + FP_SW, &[0u8; 4])
    };
    if !fp_ok {
        cpu.core.exception_index = -1;
        signal::force_sigsegv(t, sig);
        return;
    }
    put64(&mut f, SC_FPSTATE, fpstate);
    put64(&mut f, SC_OLDMASK, old);
    put64(&mut f, SC_CR2, env::ld64(cpu.env, env::cr(2)));
    put64(&mut f, UC_SIGMASK, old);

    // SA_RESTORER is required on x86_64.
    if sa.flags & signal::SA_RESTORER == 0 {
        signal::force_sigsegv(t, sig);
        return;
    }
    put64(&mut f, 0, sa.restorer);
    space.write_raw(frame, &f);

    set_reg(cpu, R_ESP, frame);
    env::st64(cpu.env, EIP, sa.handler);
    set_reg(cpu, R_EAX, 0);
    set_reg(cpu, R_EDI, sig as u64);
    set_reg(cpu, R_ESI, frame + INFO as u64);
    set_reg(cpu, R_EDX, frame + UC as u64);
    for (seg, s) in [(R_DS, USER_DS), (R_ES, USER_DS), (R_CS, USER_CS), (R_SS, USER_DS)] {
        if cpu_x86_load_seg(cpu, seg, s).is_err() {
            break;
        }
    }
    let fl = env::ld64(cpu.env, env::EFLAGS);
    env::st64(cpu.env, env::EFLAGS, fl & !TF_MASK);
}

/// `xrstor_sigcontext()` of the state at `fp`, false when the frame is bad.
fn restore_fpstate(space: &GuestSpace, cpu: &mut Cpu<'_>, fp: u64) -> bool {
    let xsave = uses_xsave(cpu);
    let math = fpstate_size(cpu, xsave);
    let mut img = vec![0u8; 512];
    if !space.check(fp, math, page::READ) || !space.read(fp, &mut img) {
        return false;
    }
    if xsave {
        let xcr0 = env::ld64(cpu.env, env::XCR0);
        let sw = &img[FP_SW as usize..];
        let (magic1, ext, xs) = (get32(sw, 0), u64::from(get32(sw, 4)), u64::from(get32(sw, 16)));
        let max = xsave_area_size(cpu, xcr0) as u64;
        if magic1 == FP_XSTATE_MAGIC1 && (XSAVE_MIN..=max).contains(&xs) && xs <= ext {
            let xfeatures = get64(sw, 8) & xcr0;
            if xs < xsave_area_size(cpu, xfeatures) as u64 || !space.check(fp, xs + 4, page::READ) {
                return false;
            }
            let mut m2 = [0u8; 4];
            if space.read(fp + xs, &mut m2) && u32::from_le_bytes(m2) == FP_XSTATE_MAGIC2 {
                return matches!(cpu_x86_xrstor(cpu, fp, xfeatures), Ok(true));
            }
        }
    }
    cpu_x86_fxrstor(cpu, fp).is_ok()
}

/// `do_rt_sigreturn()`.
fn do_rt_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    let frame = reg(cpu, R_ESP).wrapping_sub(8);
    let mut f = [0u8; FRAME_SIZE as usize];
    if !space.read(frame, &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    signal::set_sigmask(t, signal::t2h_set(get64(&f, UC_SIGMASK)));

    // restore_sigcontext().
    for (n, at) in SC_REGS.iter().enumerate() {
        set_reg(cpu, n, get64(&f, *at));
    }
    env::st64(cpu.env, EIP, get64(&f, SC_RIP));
    // A selector that does not load leaves a #GP for the next instruction.
    let cs = u32::from(u16::from_le_bytes([f[SC_CS], f[SC_CS + 1]])) | 3;
    let ss = u32::from(u16::from_le_bytes([f[SC_SS], f[SC_SS + 1]])) | 3;
    if cpu_x86_load_seg(cpu, R_CS, cs).is_ok() {
        let _ = cpu_x86_load_seg(cpu, R_SS, ss);
    }
    let fl = env::ld64(cpu.env, env::EFLAGS);
    let tmp = get64(&f, SC_EFLAGS);
    env::st64(cpu.env, env::EFLAGS, (fl & !FIX_EFLAGS) | (tmp & FIX_EFLAGS));
    let fp = get64(&f, SC_FPSTATE);
    let ok = fp == 0 || restore_fpstate(space, cpu, fp);
    if !ok {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    let sp = reg(cpu, R_ESP);
    let _ = signal::restore_altstack(t, &f[UC_STACK..UC_STACK + 24], sp);
    signal::ESIGRETURN
}

/// `cpu_loop()`, until the thread calls `exit` with others left.
pub(crate) fn cpu_loop(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
        cpu.process_queued_cpu_work();
        match trapnr {
            0x80 | EXCP_SYSCALL => {
                let a = if trapnr == EXCP_SYSCALL {
                    [R_EAX, R_EDI, R_ESI, R_EDX, 10, 8, 9]
                } else {
                    [R_EAX, R_EBX, R_ECX, R_EDX, R_ESI, R_EDI, R_EBP]
                };
                let nr = reg(cpu, a[0]);
                let args = [
                    reg(cpu, a[1]),
                    reg(cpu, a[2]),
                    reg(cpu, a[3]),
                    reg(cpu, a[4]),
                    reg(cpu, a[5]),
                    reg(cpu, a[6]),
                ];
                let ret = syscall::do_syscall(p, t, cpu, nr, args);
                if ret == THREAD_EXIT {
                    return;
                }
                if ret == signal::ERESTARTSYS {
                    let eip = env::ld64(cpu.env, EIP);
                    env::st64(cpu.env, EIP, eip.wrapping_sub(2));
                } else if ret != signal::ESIGRETURN {
                    set_reg(cpu, R_EAX, ret as u64);
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
                    "qemu: unhandled CPU exception 0x{trapnr:x} - aborting\nRIP={:016x}",
                    env::ld64(cpu.env, EIP)
                );
                std::process::abort();
            }
        }
        let space = Arc::clone(p.space());
        signal::process_pending_signals(&space, t, cpu);
    }
}

/// The x86_64 target.
pub(crate) static GUEST: Guest = Guest {
    machine: "x86_64",
    minsigstksz: 2048,
    env_size: env::ENV_SIZE,
    generic_abi: false,
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
    setup_rt_frame,
    rt_sigreturn: do_rt_sigreturn,
};

/// `qemu-x86_64`: the model `-cpu` names.
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
        "x86_64"
    }

    fn layout(&self) -> (u64, u64) {
        (TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE)
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
        Arch {
            machine: EM_X86_64,
            platform: Some("x86_64"),
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
        let state = initial_state(&model, space, info.entry, info.start_stack)?;
        let x86 = Arc::new(X86::new(model).with_user_mode());
        let ops = Arc::new(UserCpu { x86, space: Arc::clone(space) });
        let mut v = jit.create_vcpu(ops, as_, env::ENV_SIZE);
        env::load_state(&mut v.env, &state);
        Ok(v)
    }
}
