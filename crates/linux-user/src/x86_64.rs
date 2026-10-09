// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86_64 target: `linux-user/i386/cpu_loop.c`, `target_cpu_copy_regs()`, the user mode
//! parts of `x86_cpu_reset_hold()` and `do_arch_prctl()`, and `main()` for `qemu-x86_64`.

use std::fmt;
use std::fs::File;
use std::process::ExitCode;
use std::sync::Arc;

use ruvm_jit::cpu_exec::{cpu_exec_step_atomic, tcg_cpu_exec};
use ruvm_jit::cputlb::tlb_set_page;
use ruvm_jit::translate::TbBuild;
use ruvm_jit::{
    Cpu, CpuLoopExit, CpuOps, Jit, MmuAccessType, Ra, Tb, TbCpuState, Watchpoint, excp,
};
use ruvm_jit_core::Type;
use ruvm_jit_core::types::INSN_START_WORDS;
use ruvm_mem::{MemTxAttrs, MemTxResult, MemorySystem};
use ruvm_target_x86::cpuid::{Accel, X86Cpu};
use ruvm_target_x86::state::{
    CR0_PE_MASK, CR0_PG_MASK, CR0_WP_MASK, CR4_FSGSBASE_MASK, CR4_OSFXSR_MASK, CR4_OSXSAVE_MASK,
    CR4_PAE_MASK, DESC_A_MASK, DESC_B_MASK, DESC_DPL_SHIFT, DESC_G_MASK, DESC_L_MASK, DESC_P_MASK,
    DESC_S_MASK, DESC_TYPE_SHIFT, HF_CPL_MASK, HF_LMA_MASK, HF_PE_MASK, MSR_EFER_LMA, MSR_EFER_LME,
    R_CS, R_DS, R_EAX, R_EBP, R_EBX, R_ECX, R_EDI, R_EDX, R_ES, R_ESI, R_ESP, R_FS, R_GS, R_SS,
    SegmentCache,
};
use ruvm_target_x86::tcg::env::{self, EIP, HF_OSFXSR_MASK, REGS, SEG_BASE, SEG_SIZE, SEGS};
use ruvm_target_x86::tcg::user::record_sigsegv;
use ruvm_target_x86::tcg::{
    EXCP_SYSCALL, EXCP00_DIVZ, EXCP0B_NOSEG, EXCP0C_STACK, EXCP0D_GPF, EXCP0E_PAGE, EXCP01_DB,
    EXCP03_INT3, EXCP04_INTO, EXCP05_BOUND, EXCP06_ILLOP, X86, helper_registry, jit_config,
};
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use crate::elf::{self, Arch, Creds, Exec};
use crate::host;
use crate::opts::{self, Exit};
use crate::syscall::{self, Proc};

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
        Err(record_sigsegv(cpu, addr, access_type, flags == 0, ra))
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

/// `RSP`.
pub(crate) const RSP: usize = R_ESP;

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
pub(crate) fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
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

/// `dump_core_and_abort()` for a signal the program did not handle: QEMU's message, then the
/// same signal for the emulator, without a host core dump.
fn dump_core_and_abort(sig: i32) -> ! {
    eprintln!("qemu: uncaught target signal {sig} ({}) - core dumped", syscall::strsignal(sig));
    host::die_with_signal(sig)
}

/// `cpu_loop()`.
fn cpu_loop(p: &mut Proc, cpu: &mut Cpu<'_>) -> ! {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
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
                let ret = syscall::do_syscall(p, cpu, nr, args);
                set_reg(cpu, R_EAX, ret as u64);
            }
            EXCP0B_NOSEG | EXCP0C_STACK => dump_core_and_abort(libc::SIGBUS),
            EXCP0D_GPF | EXCP0E_PAGE | EXCP04_INTO | EXCP05_BOUND => {
                dump_core_and_abort(libc::SIGSEGV)
            }
            EXCP00_DIVZ => dump_core_and_abort(libc::SIGFPE),
            EXCP01_DB | EXCP03_INT3 | excp::DEBUG => dump_core_and_abort(libc::SIGTRAP),
            EXCP06_ILLOP => dump_core_and_abort(libc::SIGILL),
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
    }
}

/// The host side of the auxiliary vector.
fn creds() -> Creds {
    let id = |nr| host::sys(nr, &[]) as u64;
    Creds {
        ids: [
            id(libc::SYS_getuid),
            id(libc::SYS_geteuid),
            id(libc::SYS_getgid),
            id(libc::SYS_getegid),
        ],
        clktck: 100,
        secure: 0,
        random: host::random16(),
    }
}

/// `prepare_binprm()` and the format check of `loader_exec()`: whether `file` is a regular,
/// executable file starting with the ELF magic. Every failure is `ENOEXEC` to the user.
fn is_exec(file: &File) -> bool {
    use std::os::unix::fs::{FileExt, PermissionsExt};
    let Ok(m) = file.metadata() else { return false };
    if !m.is_file() || m.permissions().mode() & 0o111 == 0 {
        return false;
    }
    let mut head = [0u8; 4];
    matches!(file.read_at(&mut head, 0), Ok(4)) && elf::is_elf(&head)
}

/// `main()` of `qemu-x86_64`.
pub(crate) fn main(argv0: &str, args: &[String]) -> ExitCode {
    let prog = argv0.rsplit('/').next().unwrap_or(argv0);
    let env: Vec<(String, String)> = std::env::vars_os()
        .map(|(k, v)| (k.to_string_lossy().into_owned(), v.to_string_lossy().into_owned()))
        .collect();
    let mut stack_size = opts::DEFAULT_STACK_SIZE;
    if let Some(cur) = host::stack_rlimit() {
        stack_size = stack_size.max(cur);
    }
    let mut o = match opts::parse("x86_64", env, args) {
        Ok(o) => o,
        Err(Exit::Usage(text, code)) => {
            print!("{text}");
            return code;
        }
        Err(Exit::Error(msg)) => {
            eprintln!("{msg}");
            return ExitCode::FAILURE;
        }
    };
    if o.stack_size == opts::DEFAULT_STACK_SIZE {
        o.stack_size = stack_size;
    }
    host::reset_sigpipe();

    let cpu_name = o.cpu.clone().unwrap_or_else(|| "max".to_string());
    let (model_name, features) = match cpu_name.split_once(',') {
        Some((m, f)) => (m.to_string(), f.to_string()),
        None => (cpu_name.clone(), String::new()),
    };
    let model = X86Cpu::new(&model_name, Accel::Tcg).and_then(|mut c| {
        if !features.is_empty() {
            c.parse_features(&features)?;
        }
        c.realize()?;
        Ok(c)
    });
    let model = match model {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{prog}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let file = match File::open(&o.exec_path) {
        Ok(f) => f,
        Err(e) => {
            println!("Error while loading {}: {}", o.exec_path, crate::strerror_of(&e));
            return ExitCode::FAILURE;
        }
    };
    // real_exec_path, what /proc/self/exe names.
    let real_exec_path = match std::fs::canonicalize(&o.exec_path) {
        Ok(p) => p.to_string_lossy().into_owned(),
        Err(_) => {
            println!("Could not resolve {}", o.exec_path);
            o.exec_path.clone()
        }
    };
    if !is_exec(&file) {
        println!("Error while loading {}: {}", o.exec_path, crate::strerror(libc::ENOEXEC));
        return ExitCode::FAILURE;
    }

    let want =
        if o.reserved_va != 0 { o.reserved_va } else { ruvm_user_common::space::DEFAULT_RESERVE };
    let space = match GuestSpace::new(want, TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            eprintln!("{prog}: Unable to reserve guest address space: {}", crate::strerror_of(&e));
            return ExitCode::FAILURE;
        }
    };

    let arch = Arch {
        machine: EM_X86_64,
        platform: Some("x86_64"),
        hwcap: u64::from(cpuid(&model, 1, 0)[3]),
    };
    let argv: Vec<Vec<u8>> = o.args.iter().map(|a| a.clone().into_bytes()).collect();
    let envp: Vec<Vec<u8>> = o.env.iter().map(|a| a.clone().into_bytes()).collect();
    let exec = Exec {
        arch,
        filename: &o.exec_path,
        file: &file,
        argv: &argv,
        envp: &envp,
        stack_size: o.stack_size,
        ld_prefix: &o.ld_prefix,
        creds: creds(),
    };
    let info = match elf::load_elf_binary(&space, &exec) {
        Ok(i) => i,
        Err(e) => {
            // error_reportf_err() and exit(-1).
            eprintln!("{prog}: {e}");
            return ExitCode::from(255);
        }
    };
    drop(file);

    let state = match initial_state(&model, &space, info.entry, info.start_stack) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let mut config = jit_config();
    config.user_only = true;
    config.mttcg = false;
    if let Some(mb) = o.tb_size {
        if mb != 0 {
            config.code_gen_buffer_size = usize::try_from(mb << 20).unwrap_or(usize::MAX);
        }
    }
    if o.one_insn_per_tb {
        config.one_insn_per_tb = true;
    }
    let backend = ruvm_jit::host_backend(helper_registry(), config.code_gen_buffer_size);
    let jit = Jit::new(config, backend);

    let ms = MemorySystem::new();
    let as_ = (|| {
        let root = ms.new_container("system", 1u128 << 64)?;
        let ram = ms.new_ram_from_block(Arc::clone(space.block()))?;
        ms.add_subregion(root, 0, ram)?;
        ms.address_space_init(root, "memory")
    })();
    let as_ = match as_ {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{prog}: {e}");
            return ExitCode::FAILURE;
        }
    };
    {
        let j = Arc::downgrade(&jit);
        let block = Arc::clone(space.block());
        space.set_code_hook(Box::new(move |s, l| {
            if let Some(j) = j.upgrade() {
                j.tb_invalidate_phys_block(&block, s, l);
            }
        }));
    }

    let x86 = Arc::new(X86::new(model).with_user_mode());
    let ops = Arc::new(UserCpu { x86, space: Arc::clone(&space) });
    let mut v = jit.create_vcpu(ops, as_, env::ENV_SIZE);
    env::load_state(&mut v.env, &state);
    let mut proc = Proc::new(
        Arc::clone(&space),
        info.brk,
        real_exec_path,
        o.uname_release.clone(),
        o.ld_prefix.clone(),
    );
    let mut cpu = v.cpu();
    cpu_loop(&mut proc, &mut cpu)
}
