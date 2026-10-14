// SPDX-License-Identifier: GPL-2.0-or-later

//! The i386 target, `qemu-i386`: the x86 front end with a 32-bit code segment, the
//! `TARGET_ABI32` parts of `linux-user/i386/cpu_loop.c` and `do_set_thread_area()`. The
//! system calls go through [`compat`](crate::compat).
//!
//! Signal frames are not there yet: a signal with a handler kills the program with `SIGSEGV`,
//! as a frame that does not fit on the stack would.

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
use ruvm_target_x86::tcg::user::cpu_x86_load_seg;
use ruvm_target_x86::tcg::{
    EXCP_SYSCALL, EXCP00_DIVZ, EXCP0B_NOSEG, EXCP0C_STACK, EXCP0D_GPF, EXCP0E_PAGE, EXCP01_DB,
    EXCP03_INT3, EXCP04_INTO, EXCP05_BOUND, EXCP06_ILLOP, X86, helper_registry, jit_config,
};
use ruvm_user_common::GuestSpace;

use crate::compat;
use crate::elf::{Arch, ImageInfo};
use crate::gdb;
use crate::guest::Guest;
use crate::signal::{self, Sigaction, Task, get32, put32};
use crate::start;
use crate::strace;
use crate::syscall::{Proc, THREAD_EXIT};
use crate::x86_64::{
    IF_MASK, USER_DS, UserCpu, cpuid, flat_descriptor, gate64, map_rw, reg, set_reg,
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

/// What a 32-bit program's signal handler would be entered with; the frames are not written
/// yet, so the program gets `SIGSEGV`.
fn setup_rt_frame(
    _space: &GuestSpace,
    t: &mut Task,
    _cpu: &mut Cpu<'_>,
    sig: i32,
    _sa: &Sigaction,
    _info: &signal::Info,
    _old: u64,
) {
    signal::force_sigsegv(t, sig);
}

fn do_rt_sigreturn(_space: &GuestSpace, t: &mut Task, _cpu: &mut Cpu<'_>) -> i64 {
    signal::force_sig(t, signal::SIGSEGV);
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
        let state = initial_state(&model, space, info.entry, info.start_stack)?;
        let _ = SPACE.set(Arc::clone(space));
        let x86 = Arc::new(X86::new(model).with_user_mode());
        let ops = Arc::new(UserCpu { x86, space: Arc::clone(space) });
        let mut v = jit.create_vcpu(ops, as_, env::ENV_SIZE);
        env::load_state(&mut v.env, &state);
        Ok(v)
    }
}
