// SPDX-License-Identifier: GPL-2.0-or-later

//! The arm target, `qemu-arm`: the AArch32 front end at EL0, `linux-user/arm/cpu_loop.c`,
//! `init_main_thread()`, and the hardware capabilities and `/proc/cpuinfo` of `elfload.c` and
//! `target_proc.h`. The system calls are the EABI ones and go through
//! [`compat`](crate::compat), with the EABI's register pairs and structures.
//!
//! There is no commpage: the kuser helpers at 0xffff0000 are not mapped, and a jump there is
//! `SIGSEGV`. There is no vDSO either. Signal frames are not there yet: a signal with a
//! handler kills the program with `SIGSEGV`, as a frame that does not fit on the stack would.

mod syscalls;

use std::mem::offset_of;
use std::sync::{Arc, OnceLock};

use ruvm_jit::cpu_exec::{cpu_exec_step_atomic, tcg_cpu_exec};
use ruvm_jit::{Cpu, Jit, JitConfig, Vcpu, excp};
use ruvm_mem::AddressSpace;
use ruvm_target_arm::cpu::{
    ArmCpuModel, CpuArmState, ENV_SIZE, EXCP_BKPT, EXCP_DATA_ABORT, EXCP_KERNEL_TRAP,
    EXCP_PREFETCH_ABORT, EXCP_SWI, EXCP_UDEF, PC, THUMB, env_off, reg_off,
};
use ruvm_target_arm::tcg::{Arm, helper_registry, jit_config, user};
use ruvm_user_common::GuestSpace;

use crate::aarch64::{UserCpu, format_cpuinfo};
use crate::compat;
use crate::elf::{Arch, ImageInfo};
use crate::gdb;
use crate::guest::Guest;
use crate::signal::{self, Sigaction, Task};
use crate::start;
use crate::strace;
use crate::syscall::{Proc, THREAD_EXIT};

use syscalls::SYSCALLS;

/// `TASK_UNMAPPED_BASE` for arm.
const TASK_UNMAPPED_BASE: u64 = 0x4000_0000;
/// `ELF_ET_DYN_BASE` for arm.
const ELF_ET_DYN_BASE: u64 = 0x0040_0000;
/// `EM_ARM`.
const EM_ARM: u16 = 40;
/// `ARM_SYSCALL_BASE`, the OABI call numbers' prefix in the SVC immediate.
const ARM_SYSCALL_BASE: u32 = 0x90_0000;
/// `ARM_NR_BASE` and the private calls above it.
const ARM_NR_BASE: u32 = 0xf_0000;
const ARM_NR_BREAKPOINT: u32 = ARM_NR_BASE + 1;
const ARM_NR_CACHEFLUSH: u32 = ARM_NR_BASE + 2;
const ARM_NR_SET_TLS: u32 = ARM_NR_BASE + 5;
const ARM_NR_GET_TLS: u32 = ARM_NR_BASE + 6;
/// `TARGET_ILL_ILLTRP`.
const ILL_ILLTRP: i32 = 4;

const ENOSYS: i64 = libc::ENOSYS as i64;

/// The env offset of `TPIDRURO`, which the kernel's TLS is.
const TPIDRURO: usize = env_off(offset_of!(CpuArmState, tpidrro_el0));
/// The env offsets of the syndrome and the fault address of the last exception.
const SYNDROME: usize = env_off(offset_of!(CpuArmState, exception_syndrome));
const FAULT_ADDR: usize = env_off(offset_of!(CpuArmState, exception_vaddress));

/// `get_elf_hwcap()` bits.
const HWCAP_SWP: u64 = 1 << 0;
const HWCAP_HALF: u64 = 1 << 1;
const HWCAP_THUMB: u64 = 1 << 2;
const HWCAP_FAST_MULT: u64 = 1 << 4;
const HWCAP_VFP: u64 = 1 << 6;
const HWCAP_EDSP: u64 = 1 << 7;
const HWCAP_NEON: u64 = 1 << 12;
const HWCAP_VFPV3: u64 = 1 << 13;
const HWCAP_VFPV3D16: u64 = 1 << 14;
const HWCAP_TLS: u64 = 1 << 15;
const HWCAP_VFPV4: u64 = 1 << 16;
const HWCAP_IDIVA: u64 = 1 << 17;
const HWCAP_IDIVT: u64 = 1 << 18;
const HWCAP_VFPD32: u64 = 1 << 19;
const HWCAP_LPAE: u64 = 1 << 20;
const HWCAP_FPHP: u64 = 1 << 22;
const HWCAP_ASIMDHP: u64 = 1 << 23;
const HWCAP_ASIMDDP: u64 = 1 << 24;
const HWCAP_ASIMDFHM: u64 = 1 << 25;
const HWCAP_ASIMDBF16: u64 = 1 << 26;
const HWCAP_I8MM: u64 = 1 << 27;
/// `get_elf_hwcap2()` bits.
const HWCAP2_AES: u64 = 1 << 0;
const HWCAP2_PMULL: u64 = 1 << 1;
const HWCAP2_SHA1: u64 = 1 << 2;
const HWCAP2_SHA2: u64 = 1 << 3;
const HWCAP2_CRC32: u64 = 1 << 4;
const HWCAP2_SB: u64 = 1 << 5;
const HWCAP2_SSBS: u64 = 1 << 6;

/// The capabilities that are Advanced SIMD instructions, which the AArch32 translator does
/// not have yet, so that programs do not pick them.
const HWCAP_SIMD: u64 =
    HWCAP_NEON | HWCAP_ASIMDHP | HWCAP_ASIMDDP | HWCAP_ASIMDFHM | HWCAP_ASIMDBF16 | HWCAP_I8MM;
const HWCAP2_SIMD: u64 = HWCAP2_AES | HWCAP2_PMULL | HWCAP2_SHA1 | HWCAP2_SHA2;

/// Register `n`, R0 to R14.
fn reg(cpu: &Cpu<'_>, n: usize) -> u32 {
    let off = reg_off(n);
    u32::from_le_bytes(cpu.env[off..off + 4].try_into().expect("4 bytes"))
}

fn set_reg(cpu: &mut Cpu<'_>, n: usize, v: u32) {
    let off = reg_off(n);
    cpu.env[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn ld32(cpu: &Cpu<'_>, off: usize) -> u32 {
    u32::from_le_bytes(cpu.env[off..off + 4].try_into().expect("4 bytes"))
}

fn ld64(cpu: &Cpu<'_>, off: usize) -> u64 {
    u64::from_le_bytes(cpu.env[off..off + 8].try_into().expect("8 bytes"))
}

fn st64(cpu: &mut Cpu<'_>, off: usize, v: u64) {
    cpu.env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `regs[15]`.
fn pc(cpu: &Cpu<'_>) -> u32 {
    ld64(cpu, PC) as u32
}

fn set_pc(cpu: &mut Cpu<'_>, pc: u32) {
    st64(cpu, PC, u64::from(pc));
}

/// Whether the CPU is in Thumb state.
fn thumb(cpu: &Cpu<'_>) -> bool {
    ld32(cpu, THUMB) != 0
}

/// `cpu_set_tls()`: TPIDRURO.
fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
    st64(cpu, TPIDRURO, tls & 0xffff_ffff);
}

/// The ID register field at `shift`.
fn field(v: u32, shift: u32) -> u32 {
    (v >> shift) & 0xf
}

/// `get_elf_hwcap()` and `get_elf_hwcap2()` of the model, without the Advanced SIMD ones.
fn hwcaps(m: &ArmCpuModel) -> (u64, u64) {
    let id = m.aa32.unwrap_or_default();
    let [mvfr0, mvfr1, _] = id.mvfr;
    let (isar0, isar5, isar6) = (id.id_isar[0], id.id_isar[5], id.id_isar[6]);
    let mut h = 0u64;
    let mut set = |cond: bool, bits: u64| {
        if cond {
            h |= bits;
        }
    };
    set(true, HWCAP_SWP | HWCAP_HALF | HWCAP_THUMB | HWCAP_FAST_MULT);
    // ARM_FEATURE_V5, ARM_FEATURE_NEON, ARM_FEATURE_V6K and ARM_FEATURE_LPAE, which every
    // ARMv8 CPU has.
    set(true, HWCAP_EDSP | HWCAP_NEON | HWCAP_TLS | HWCAP_LPAE);
    set(field(isar0, 24) > 1, HWCAP_IDIVA);
    set(field(isar0, 24) != 0, HWCAP_IDIVT);
    // aa32_vfp: FPSP or FPDP.
    set(field(mvfr0, 4) != 0 || field(mvfr0, 8) != 0, HWCAP_VFP);
    if field(mvfr0, 4) >= 2 || field(mvfr0, 8) >= 2 {
        set(true, HWCAP_VFPV3);
        // aa32_simd_r32: SIMDReg.
        set(field(mvfr0, 0) >= 2, HWCAP_VFPD32);
        set(field(mvfr0, 0) < 2, HWCAP_VFPV3D16);
    }
    set(field(mvfr1, 28) != 0, HWCAP_VFPV4);
    // aa32_fp16_arith, for both.
    set(field(mvfr1, 24) >= 3, HWCAP_FPHP | HWCAP_ASIMDHP);
    set(field(isar6, 4) != 0, HWCAP_ASIMDDP);
    set(field(isar6, 8) != 0, HWCAP_ASIMDFHM);
    set(field(isar6, 20) != 0, HWCAP_ASIMDBF16);
    set(field(isar6, 24) != 0, HWCAP_I8MM);

    let mut h2 = 0u64;
    let mut set2 = |cond: bool, bits: u64| {
        if cond {
            h2 |= bits;
        }
    };
    set2(field(isar5, 4) != 0, HWCAP2_AES);
    set2(field(isar5, 4) > 1, HWCAP2_PMULL);
    set2(field(isar5, 8) != 0, HWCAP2_SHA1);
    set2(field(isar5, 12) != 0, HWCAP2_SHA2);
    set2(field(isar5, 16) != 0, HWCAP2_CRC32);
    set2(field(isar6, 12) != 0, HWCAP2_SB);
    set2(field(id.id_pfr[2], 4) != 0, HWCAP2_SSBS);
    (h & !HWCAP_SIMD, h2 & !HWCAP2_SIMD)
}

/// `elf_hwcap_str()` names, by bit.
const HWCAP_NAMES: &[&str] = &[
    "swp",
    "half",
    "thumb",
    "26bit",
    "fast_mult",
    "fpa",
    "vfp",
    "edsp",
    "java",
    "iwmmxt",
    "crunch",
    "thumbee",
    "neon",
    "vfpv3",
    "vfpv3d16",
    "tls",
    "vfpv4",
    "idiva",
    "idivt",
    "vfpd32",
    "lpae",
    "evtstrm",
    "fphp",
    "asimdhp",
    "asimddp",
    "asimdfhm",
    "asimdbf16",
    "i8mm",
];

/// `elf_hwcap2_str()` names, by bit.
const HWCAP2_NAMES: &[&str] = &["aes", "pmull", "sha1", "sha2", "crc32", "sb", "ssbs"];

/// HWCAP, HWCAP2 and MIDR of the model, for `/proc/cpuinfo`.
static CPUINFO: OnceLock<(u64, u64, u64)> = OnceLock::new();

/// `open_cpuinfo()` of `target_proc.h` for arm, which treats an ARMv8 CPU as an arm64 kernel
/// with `CONFIG_COMPAT` shows it.
fn cpuinfo() -> String {
    let (h, h2, midr) = CPUINFO.get().copied().unwrap_or_default();
    format_cpuinfo(&[(HWCAP_NAMES, h), (HWCAP2_NAMES, h2)], midr)
}

/// What an arm program's signal handler would be entered with; the frames are not written
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

/// The strace entry of arm call `nr`: the host's when the call is the host's, else the name.
fn strace_entry(nr: u64) -> Option<&'static strace::Entry> {
    static NAMED: OnceLock<Vec<strace::Entry>> = OnceLock::new();
    let &(name, host) = SYSCALLS.get(nr as usize)?;
    strace::entry(i64::from(host)).filter(|e| e.name() == name).or_else(|| {
        NAMED.get_or_init(|| strace::named(SYSCALLS.iter().map(|s| s.0))).get(nr as usize)
    })
}

/// `do_syscall()` of arm EABI call `nr`.
fn do_syscall(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>, nr: u64, a: [u64; 6]) -> i64 {
    let Some(&(name, host)) = SYSCALLS.get(nr as usize) else { return -ENOSYS };
    compat::do_syscall(p, t, cpu, name, (host >= 0).then_some(i64::from(host)), a)
}

/// `insn_is_linux_bkpt()`: the three UDF encodings the kernel takes as breakpoints. A T32
/// instruction is read as two halfwords, the first in the low half.
fn insn_is_linux_bkpt(opcode: u32, thumb: bool) -> bool {
    if thumb {
        opcode & 0xffff == 0xde01 || opcode == 0xa000_f7f0
    } else {
        opcode & 0x0fff_ffff == 0x07f0_01f0
    }
}

/// `get_user_code_u32()`: the word at `addr`, or for a 16-bit Thumb instruction at the end of
/// the mapping, the halfword.
fn code_u32(space: &GuestSpace, addr: u32) -> Option<u32> {
    let mut b = [0u8; 4];
    if space.read(u64::from(addr), &mut b) {
        return Some(u32::from_le_bytes(b));
    }
    let mut h = [0u8; 2];
    space.read(u64::from(addr), &mut h).then(|| u32::from(u16::from_le_bytes(h)))
}

/// The private calls from `ARM_NR_BASE` on, which `cpu_loop()` handles itself.
fn arm_private_call(t: &mut Task, cpu: &mut Cpu<'_>, n: u32) {
    let width = if thumb(cpu) { 2 } else { 4 };
    match n {
        ARM_NR_CACHEFLUSH => set_reg(cpu, 0, 0),
        ARM_NR_SET_TLS => {
            let tls = reg(cpu, 0);
            set_tls(cpu, u64::from(tls));
            set_reg(cpu, 0, 0);
        }
        ARM_NR_BREAKPOINT => {
            let pc = pc(cpu).wrapping_sub(width);
            set_pc(cpu, pc);
            signal::force_sig_fault(t, signal::SIGTRAP, signal::TRAP_BRKPT, u64::from(pc));
        }
        ARM_NR_GET_TLS => set_reg(cpu, 0, ld64(cpu, TPIDRURO) as u32),
        // 0xf0000 to 0xf07ff are ENOSYS rather than SIGILL, usr26 and usr32 among them.
        _ if n < 0xf_0800 => set_reg(cpu, 0, -ENOSYS as u32),
        // As the kernel does, the address is that of the SVC, but the PC stays after it.
        _ => {
            let addr = pc(cpu).wrapping_sub(width);
            signal::force_sig_fault(t, signal::SIGILL, ILL_ILLTRP, u64::from(addr));
        }
    }
}

/// `EXCP_SWI`: the call of the SVC before the PC. The number is in R7 for EABI programs; an
/// ARM SVC with a nonzero immediate is an OABI call, which the kernel's `CONFIG_OABI_COMPAT`
/// takes. Only the OABI numbers of the private calls are taken here, the others are `ENOSYS`.
/// True when the thread has exited.
fn do_swi(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) -> bool {
    let thumb = thumb(cpu);
    let mut oabi = false;
    let n = if thumb {
        reg(cpu, 7)
    } else {
        // The syndrome has the low 16 bits of the immediate, should the SVC not be readable.
        let insn = code_u32(p.space(), pc(cpu).wrapping_sub(4))
            .unwrap_or_else(|| ld32(cpu, SYNDROME) & 0xffff);
        match insn & 0xff_ffff {
            0 => reg(cpu, 7),
            imm => {
                oabi = true;
                imm ^ ARM_SYSCALL_BASE
            }
        }
    };
    if n > ARM_NR_BASE {
        arm_private_call(t, cpu, n);
        return false;
    }
    if oabi {
        set_reg(cpu, 0, -ENOSYS as u32);
        return false;
    }
    let nr = u64::from(n);
    let args: [u64; 6] = std::array::from_fn(|i| u64::from(reg(cpu, i)));
    gdb::syscall_entry(t, cpu, nr);
    let traced = strace::on().then(|| strace::syscall(p.space(), nr, strace_entry(nr), &args));
    let ret = do_syscall(p, t, cpu, nr, args);
    if let Some(e) = traced {
        strace::syscall_ret(p.space(), e, ret, &args);
    }
    if ret == THREAD_EXIT {
        return true;
    }
    gdb::syscall_return(t, cpu, nr);
    if ret == signal::ERESTARTSYS {
        let pc = pc(cpu).wrapping_sub(if thumb { 2 } else { 4 });
        set_pc(cpu, pc);
    } else if ret != signal::ESIGRETURN {
        set_reg(cpu, 0, ret as u32);
    }
    false
}

/// The signal of a prefetch or data abort. QEMU reads the short-descriptor FSR; here EL1 is
/// AArch64, so the fault status is that of the AArch64 syndrome.
fn signal_for_abort(t: &mut Task, cpu: &Cpu<'_>) {
    let fsc = ld32(cpu, SYNDROME) & 0x3f;
    let (sig, code) = match fsc {
        0x04..=0x07 => (signal::SIGSEGV, signal::SEGV_MAPERR),
        0x09..=0x0b | 0x0d..=0x0f => (signal::SIGSEGV, signal::SEGV_ACCERR),
        0x21 => (signal::SIGBUS, signal::BUS_ADRALN),
        _ => {
            eprintln!("qemu: unexpected fault status 0x{fsc:x}");
            std::process::abort();
        }
    };
    signal::force_sig_fault(t, sig, code, ld64(cpu, FAULT_ADDR) & 0xffff_ffff);
}

/// `cpu_loop()`, until the thread calls `exit` with others left.
pub(crate) fn cpu_loop(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
        cpu.process_queued_cpu_work();
        match trapnr {
            EXCP_UDEF => {
                // The FPA11 emulation of QEMU is not here; an FPA instruction is SIGILL.
                let pc = pc(cpu);
                let opcode = code_u32(p.space(), pc).unwrap_or(0);
                let (sig, code) = if insn_is_linux_bkpt(opcode, thumb(cpu)) {
                    (signal::SIGTRAP, signal::TRAP_BRKPT)
                } else {
                    (signal::SIGILL, signal::ILL_ILLOPN)
                };
                signal::force_sig_fault(t, sig, code, u64::from(pc));
            }
            EXCP_SWI => {
                if do_swi(p, t, cpu) {
                    return;
                }
            }
            excp::INTERRUPT | excp::YIELD => {}
            EXCP_PREFETCH_ABORT | EXCP_DATA_ABORT => signal_for_abort(t, cpu),
            excp::DEBUG | EXCP_BKPT => {
                let pc = pc(cpu);
                signal::force_sig_fault(t, signal::SIGTRAP, signal::TRAP_BRKPT, u64::from(pc));
            }
            // do_kernel_trap(): with no commpage, the kuser helpers are not there.
            EXCP_KERNEL_TRAP => {
                let pc = pc(cpu);
                signal::force_sig_fault(t, signal::SIGSEGV, signal::SEGV_MAPERR, u64::from(pc));
            }
            excp::ATOMIC => cpu_exec_step_atomic(cpu),
            _ => {
                eprintln!(
                    "qemu: unhandled CPU exception 0x{trapnr:x} - aborting\nPC={:08x}",
                    pc(cpu)
                );
                std::process::abort();
            }
        }
        let space = Arc::clone(p.space());
        signal::process_pending_signals(&space, t, cpu);
    }
}

/// The arm target. `cpu_to_uname_machine()` says `armv7l` for any ARMv7 or later CPU.
pub(crate) static GUEST: Guest = Guest {
    machine: "armv7l",
    minsigstksz: 2048,
    env_size: ENV_SIZE,
    generic_abi: false,
    abi32: true,
    eabi: true,
    sa_restorer: true,
    // O_DIRECTORY, O_NOFOLLOW, O_DIRECT and O_LARGEFILE.
    open_flags: &[
        (0o40000, 0o200000),
        (0o100000, 0o400000),
        (0o200000, 0o40000),
        (0o400000, 0o100000),
    ],
    cpuinfo: Some(cpuinfo),
    sp: |cpu| u64::from(reg(cpu, 13)),
    clone_regs: |cpu, newsp| {
        if newsp != 0 {
            set_reg(cpu, 13, newsp as u32);
        }
        set_reg(cpu, 0, 0);
    },
    set_tls,
    cpu_loop,
    setup_rt_frame,
    rt_sigreturn: do_rt_sigreturn,
};

/// `qemu-arm`: the model `-cpu` names.
#[derive(Default)]
pub(crate) struct Target {
    model: Option<ArmCpuModel>,
}

impl Target {
    fn model(&self) -> &ArmCpuModel {
        self.model.as_ref().expect("the CPU is selected first")
    }
}

impl start::Target for Target {
    fn guest(&self) -> &'static Guest {
        &GUEST
    }

    fn name(&self) -> &'static str {
        "arm"
    }

    fn layout(&self) -> (u64, u64) {
        (TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE)
    }

    fn max_reserve(&self) -> u64 {
        1 << 32
    }

    /// `max`, and `any`, which is `max` for the user mode emulator, are the only AArch32
    /// models so far.
    fn select_cpu(&mut self, cpu: &str) -> Result<(), String> {
        let (name, _features) = cpu.split_once(',').unwrap_or((cpu, ""));
        let m = match name {
            "max" | "any" => ArmCpuModel::max_aa32(),
            _ => return Err(format!("unable to find CPU model '{name}'")),
        };
        let (h, h2) = hwcaps(&m);
        let _ = CPUINFO.set((h, h2, m.midr));
        self.model = Some(m);
        Ok(())
    }

    fn arch(&self) -> Arch {
        let (h, h2) = hwcaps(self.model());
        Arch {
            machines: &[EM_ARM],
            elf32: true,
            // get_elf_platform() of an ARMv8 CPU.
            platform: Some("v8l"),
            hwcap: h,
            hwcap2: Some(h2),
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
        let arm = Arc::new(Arm::new(model).with_user_mode());
        // init_main_thread(): User mode, in Thumb state for an odd entry point. R0 is the
        // atexit() function, none, and R10 the start of the data for uClinux PIC programs.
        let mut st = user::user_reset_aa32(arm.model(), info.entry & 1 != 0);
        st.pc = info.entry & 0xffff_fffe;
        st.regs[13] = info.start_stack as u32;
        st.regs[0] = 0;
        st.regs[10] = info.start_data as u32;
        let ops = Arc::new(UserCpu { arm, space: Arc::clone(space) });
        let mut v = jit.create_vcpu(ops, as_, ENV_SIZE);
        st.store(&mut v.env);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syscall_table_is_the_eabi_one() {
        let nr = |name: &str| SYSCALLS.iter().position(|s| s.0 == name);
        assert_eq!(nr("exit"), Some(1));
        assert_eq!(nr("ugetrlimit"), Some(191));
        assert_eq!(nr("mmap2"), Some(192));
        assert_eq!(nr("pread64"), Some(180));
        assert_eq!(nr("arm_fadvise64_64"), Some(270));
        assert_eq!(nr("arm_sync_file_range"), Some(341));
        assert_eq!(nr("clone"), Some(120));
        assert_eq!(nr("exit_group"), Some(248));
        assert_eq!(nr("set_tid_address"), Some(256));
        assert_eq!(nr("clock_gettime64"), Some(403));
        assert_eq!(SYSCALLS.last().map(|s| s.0), Some("mseal"));
        // The OABI only calls are not there.
        for name in ["time", "mmap", "select", "socketcall", "ipc", "getrlimit", "umount"] {
            assert_eq!(nr(name), None, "{name}");
        }
    }

    #[test]
    fn syscall_table_host_numbers() {
        let host = |name: &str| SYSCALLS.iter().find(|s| s.0 == name).map(|s| s.1);
        assert_eq!(host("read"), Some(libc::SYS_read as i16));
        assert_eq!(host("openat"), Some(libc::SYS_openat as i16));
        assert_eq!(host("accept"), Some(libc::SYS_accept as i16));
        // The calls with 64-bit arguments or 32-bit structures are converted by name.
        for name in ["pread64", "truncate64", "fstat64", "fcntl64", "_llseek", "readahead"] {
            assert_eq!(host(name), Some(-1), "{name}");
        }
        // Each name is there once, and a number without a call has no host call.
        let mut names: Vec<_> = SYSCALLS.iter().map(|s| s.0).filter(|n| !n.is_empty()).collect();
        let len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), len);
        assert!(SYSCALLS.iter().all(|s| !s.0.is_empty() || s.1 == -1));
    }

    #[test]
    fn max_hwcaps() {
        let (h, h2) = hwcaps(&ArmCpuModel::max_aa32());
        let want = HWCAP_SWP
            | HWCAP_HALF
            | HWCAP_THUMB
            | HWCAP_FAST_MULT
            | HWCAP_VFP
            | HWCAP_EDSP
            | HWCAP_VFPV3
            | HWCAP_TLS
            | HWCAP_VFPV4
            | HWCAP_IDIVA
            | HWCAP_IDIVT
            | HWCAP_VFPD32
            | HWCAP_LPAE
            | HWCAP_FPHP;
        assert_eq!(h, want);
        assert_eq!(h2, HWCAP2_CRC32 | HWCAP2_SB | HWCAP2_SSBS);
        assert_eq!(h & HWCAP_SIMD, 0);
        assert_eq!(HWCAP_NAMES.len(), 28);
    }

    #[test]
    fn linux_breakpoints() {
        assert!(insn_is_linux_bkpt(0xe7f0_01f0, false));
        assert!(!insn_is_linux_bkpt(0xe7f0_00f0, false));
        assert!(insn_is_linux_bkpt(0x0000_de01, true));
        assert!(insn_is_linux_bkpt(0xa000_f7f0, true));
        assert!(!insn_is_linux_bkpt(0x0000_de02, true));
    }

    #[test]
    fn layout() {
        assert_eq!(ELF_ET_DYN_BASE, 0x40_0000);
        assert_eq!(ARM_NR_SET_TLS, 0xf_0005);
        assert_eq!(ARM_SYSCALL_BASE ^ 0x90_0001, 1);
    }
}
