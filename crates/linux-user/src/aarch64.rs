// SPDX-License-Identifier: GPL-2.0-or-later

//! The aarch64 target: `linux-user/aarch64/cpu_loop.c`, `target_cpu_copy_regs()`, the hardware
//! capabilities and `/proc/cpuinfo` of `elfload.c` and `target_proc.h`, and the signal frames
//! of `linux-user/aarch64/signal.c`.

use std::fmt;
use std::mem::offset_of;
use std::sync::{Arc, OnceLock};

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
use ruvm_target_arm::cpu::{
    ArmCpuModel, CpuArmState, ENV_SIZE, EXCLUSIVE_ADDR, EXCP_BKPT, EXCP_DATA_ABORT,
    EXCP_PREFETCH_ABORT, EXCP_SWI, EXCP_UDEF, PC, PSTATE_M, PSTATE_NRW, PSTATE_SP, TFSR_EL,
    env_off, xreg_off,
};
use ruvm_target_arm::syndrome;
use ruvm_target_arm::tcg::{Arm, helper_registry, jit_config, user, vfp_get_fpsr, vfp_set_fpsr};
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use crate::elf::{Arch, ImageInfo};
use crate::generic;
use crate::guest::Guest;
use crate::host;
use crate::signal::{self, Sigaction, Task, get32, get64, put32, put64};
use crate::start;
use crate::syscall::{self, Proc, THREAD_EXIT};

/// `TASK_UNMAPPED_BASE` for aarch64.
const TASK_UNMAPPED_BASE: u64 = 1 << 46;
/// `ELF_ET_DYN_BASE` for aarch64, two thirds of the 48-bit space, page aligned.
const ELF_ET_DYN_BASE: u64 = ((1u64 << 48) / 3 * 2 + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
/// `EM_AARCH64`.
const EM_AARCH64: u16 = 183;
/// `ENOSYS`.
const ENOSYS: i64 = libc::ENOSYS as i64;
/// `SEGV_CPERR`.
const SEGV_CPERR: i32 = 10;
/// `EC_BTITRAP`.
const EC_BTITRAP: u32 = 0x0d;
/// `EC_SMETRAP`.
const EC_SMETRAP: u32 = 0x1d;
/// `EC_MOP`.
const EC_MOP: u32 = 0x27;
/// `EC_GCS`.
const EC_GCS: u32 = 0x2d;

/// The env offset of `TPIDR_EL0`.
const TPIDR_EL0: usize = env_off(offset_of!(CpuArmState, tpidr_el));

/// The vCPU of a user mode guest: the AArch64 front end, with guest pages checked against the
/// guest's mappings instead of page tables.
struct UserCpu {
    arm: Arc<Arm>,
    space: Arc<GuestSpace>,
}

impl fmt::Debug for UserCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserCpu").finish_non_exhaustive()
    }
}

impl CpuOps for UserCpu {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        self.arm.translate_code(cpu, tb)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        self.arm.get_tb_cpu_state(cpu)
    }

    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        self.arm.synchronize_from_tb(cpu, tb);
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; INSN_START_WORDS]) {
        self.arm.restore_state_to_opc(cpu, tb, data);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        self.arm.set_pc(cpu, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        self.arm.get_pc(cpu)
    }

    fn cpu_exec_enter(&self, cpu: &mut Cpu<'_>) {
        self.arm.cpu_exec_enter(cpu);
    }

    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        self.arm.cpu_exec_exit(cpu);
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        self.arm.cpu_exec_interrupt(cpu, interrupt_request)
    }

    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.arm.cpu_exec_halt(cpu)
    }

    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        self.arm.cpu_exec_reset(cpu);
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.arm.do_interrupt(cpu);
    }

    fn fake_user_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.arm.fake_user_interrupt(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        self.arm.has_work(cpu)
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
        Err(user::record_sigsegv(&self.arm, cpu, fault, access_type, flags == 0, ra))
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        self.arm.do_unaligned_access(cpu, addr, access_type, mmu_idx, ra)
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
        self.arm.do_transaction_failed(
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
        self.arm.mmu_index(cpu, ifetch)
    }

    fn pointer_wrap(&self, cpu: &Cpu<'_>, mmu_idx: usize, result: u64, base: u64) -> u64 {
        self.arm.pointer_wrap(cpu, mmu_idx, result, base)
    }

    fn debug_excp_handler(&self, cpu: &mut Cpu<'_>) {
        self.arm.debug_excp_handler(cpu);
    }

    fn debug_check_watchpoint(&self, cpu: &mut Cpu<'_>, wp: &Watchpoint) -> bool {
        self.arm.debug_check_watchpoint(cpu, wp)
    }

    fn debug_check_breakpoint(&self, cpu: &mut Cpu<'_>) -> bool {
        self.arm.debug_check_breakpoint(cpu)
    }

    fn adjust_watchpoint_address(&self, cpu: &mut Cpu<'_>, addr: u64, len: u64) -> u64 {
        self.arm.adjust_watchpoint_address(cpu, addr, len)
    }

    fn guest_default_memory_order(&self) -> u32 {
        self.arm.guest_default_memory_order()
    }

    fn addr_type(&self) -> Type {
        self.arm.addr_type()
    }

    fn precise_smc(&self) -> bool {
        self.arm.precise_smc()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.arm.as_any()
    }
}

fn ld64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// General register `n`, with 31 the stack pointer.
fn xreg(cpu: &Cpu<'_>, n: usize) -> u64 {
    ld64(cpu.env, xreg_off(n))
}

fn set_xreg(cpu: &mut Cpu<'_>, n: usize, v: u64) {
    st64(cpu.env, xreg_off(n), v);
}

/// `cpu_set_tls()`: TPIDR_EL0.
fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
    st64(cpu.env, TPIDR_EL0, tls);
}

/// The ID register field at `shift`.
fn field(v: u64, shift: u32) -> u64 {
    (v >> shift) & 0xf
}

/// `get_elf_hwcap()` and `get_elf_hwcap2()` of the model.
fn hwcaps(m: &ArmCpuModel) -> (u64, u64) {
    let (isar0, isar1, isar2) = (m.id_aa64isar0, m.id_aa64isar1, m.id_aa64isar2);
    let (pfr0, pfr1, zfr0) = (m.id_aa64pfr0, m.id_aa64pfr1, m.id_aa64zfr0);
    let mut h = 0u64;
    let mut set = |cond: bool, bits: u64| {
        if cond {
            h |= bits;
        }
    };
    // FP, ASIMD and CPUID.
    set(true, 1 | 2 | 1 << 11);
    set(field(isar0, 4) != 0, 1 << 3);
    set(field(isar0, 4) > 1, 1 << 4);
    set(field(isar0, 8) != 0, 1 << 5);
    set(field(isar0, 12) != 0, 1 << 6);
    set(field(isar0, 12) > 1, 1 << 21);
    set(field(isar0, 16) != 0, 1 << 7);
    set(field(isar0, 32) != 0, 1 << 17);
    set(field(isar0, 36) != 0, 1 << 18);
    set(field(isar0, 40) != 0, 1 << 19);
    set(field(pfr0, 16) == 1, 1 << 9 | 1 << 10);
    set(field(isar0, 20) >= 2, 1 << 8);
    set(field(m.id_aa64mmfr2, 32) != 0, 1 << 25);
    set(field(isar0, 28) != 0, 1 << 12);
    set(field(isar0, 44) != 0, 1 << 20);
    set(field(isar1, 16) != 0, 1 << 14);
    let sve = field(pfr0, 32) != 0;
    set(sve, 1 << 22);
    let pauth = field(isar1, 4) != 0 || field(isar1, 8) != 0 || field(isar2, 12) != 0;
    set(pauth, 1 << 30 | 1 << 31);
    set(field(isar0, 48) != 0, 1 << 23);
    set(field(pfr0, 48) != 0, 1 << 24);
    set(field(isar1, 12) != 0, 1 << 13);
    set(field(isar1, 36) != 0, 1 << 29);
    set(field(isar0, 52) != 0, 1 << 27);
    set(field(isar1, 0) != 0, 1 << 16);
    set(field(isar1, 20) != 0, 1 << 15);
    set(field(isar1, 20) >= 2, 1 << 26);
    set(field(pfr1, 44) != 0, 1 << 32);
    set(field(isar2, 52) >= 2, 1 << 33);

    let mut h2 = 0u64;
    let mut set2 = |cond: bool, bits: u64| {
        if cond {
            h2 |= bits;
        }
    };
    set2(field(isar1, 0) >= 2, 1);
    set2(field(zfr0, 0) != 0, 1 << 1);
    set2(field(zfr0, 4) != 0, 1 << 2);
    set2(field(zfr0, 4) >= 2, 1 << 3);
    set2(field(zfr0, 16) != 0, 1 << 4);
    set2(field(zfr0, 32) != 0, 1 << 5);
    set2(field(zfr0, 40) != 0, 1 << 6);
    set2(field(isar0, 52) >= 2, 1 << 7);
    set2(field(isar1, 32) != 0, 1 << 8);
    set2(sve && field(zfr0, 44) != 0, 1 << 9);
    set2(field(zfr0, 52) != 0, 1 << 10);
    set2(field(zfr0, 56) != 0, 1 << 11);
    set2(sve && field(zfr0, 20) != 0, 1 << 12);
    set2(field(isar1, 52) != 0, 1 << 13);
    set2(field(isar1, 44) != 0, 1 << 14);
    set2(field(isar0, 60) != 0, 1 << 16);
    set2(field(pfr1, 0) != 0, 1 << 17);
    set2(field(pfr1, 8) >= 2, 1 << 18);
    set2(field(pfr1, 8) >= 3, 1 << 22);
    set2(field(pfr1, 24) != 0, 1 << 23 | 1 << 26 | 1 << 27 | 1 << 28 | 1 << 29);
    set2(field(isar2, 20) != 0, 1 << 44);
    set2(field(isar2, 16) != 0, 1 << 43);
    set2(field(zfr0, 0) >= 2, 1 << 36);
    set2(field(zfr0, 24) != 0, 1 << 45);
    set2(field(isar2, 52) != 0, 1 << 34);
    set2(field(isar0, 20) >= 3, 1 << 47);
    set2(field(isar2, 56) != 0, 1 << 49);
    (h, h2)
}

/// `hwcap_str()` names, by bit.
const HWCAP_NAMES: &[&str] = &[
    "fp",
    "asimd",
    "evtstrm",
    "aes",
    "pmull",
    "sha1",
    "sha2",
    "crc32",
    "atomics",
    "fphp",
    "asimdhp",
    "cpuid",
    "asimdrdm",
    "jscvt",
    "fcma",
    "lrcpc",
    "dcpop",
    "sha3",
    "sm3",
    "sm4",
    "asimddp",
    "sha512",
    "sve",
    "asimdfhm",
    "dit",
    "uscat",
    "ilrcpc",
    "flagm",
    "ssbs",
    "sb",
    "paca",
    "pacg",
    "gcs",
    "cmpbr",
    "fprcvt",
    "f8mm8",
    "f8mm4",
    "svef16mm",
    "sveeltperm",
    "sveaes2",
    "svebfscale",
    "sve2p2",
    "sme2p2",
    "smesbitperm",
    "smeaes",
    "smesfexpa",
    "smestmop",
    "smesmop4",
];

/// `hwcap_str()` names of HWCAP2, by bit.
const HWCAP2_NAMES: &[&str] = &[
    "dcpodp",
    "sve2",
    "sveaes",
    "svepmull",
    "svebitperm",
    "svesha3",
    "svesm4",
    "flagm2",
    "frint",
    "svei8mm",
    "svef32mm",
    "svef64mm",
    "svebf16",
    "i8mm",
    "bf16",
    "dgh",
    "rng",
    "bti",
    "mte",
    "ecv",
    "afp",
    "rpres",
    "mte3",
    "sme",
    "smei16i64",
    "smef64f64",
    "smei8i32",
    "smef16f32",
    "smeb16f32",
    "smef32f32",
    "smefa64",
    "wfxt",
    "ebf16",
    "sveebf16",
    "cssc",
    "rprfm",
    "sve2p1",
    "sme2",
    "sme2p1",
    "smei16i32",
    "smebi32i32",
    "smeb16b16",
    "smef16f16",
    "mops",
    "hbc",
    "sveb16b16",
    "lrcpc3",
    "lse128",
    "fpmr",
    "lut",
    "faminmax",
    "f8cvt",
    "f8fma",
    "f8dp4",
    "f8dp2",
    "f8e4m3",
    "f8e5m2",
    "smelutv2",
    "smef8f16",
    "smef8f32",
    "smesf8fma",
    "smesf8dp4",
    "smesf8dp2",
    "poe",
];

/// HWCAP, HWCAP2 and MIDR of the model, for `/proc/cpuinfo`.
static CPUINFO: OnceLock<(u64, u64, u64)> = OnceLock::new();

/// `open_cpuinfo()` of `target_proc.h` for aarch64.
fn cpuinfo() -> String {
    use std::fmt::Write;
    let (h, h2, midr) = CPUINFO.get().copied().unwrap_or_default();
    let rev = midr & 0xf;
    let mut features = String::new();
    for (names, caps) in [(HWCAP_NAMES, h), (HWCAP2_NAMES, h2)] {
        for (bit, name) in names.iter().enumerate() {
            if caps & (1 << bit) != 0 {
                features.push(' ');
                features.push_str(name);
            }
        }
    }
    let mut s = String::new();
    for i in 0..host::online_cpus() {
        let _ = write!(
            s,
            "processor\t: {i}\nmodel name\t: ARMv8 Processor rev {rev} (v8l)\nBogoMIPS\t: \
             100.00\nFeatures\t:{features}\nCPU implementer\t: 0x{:02x}\nCPU architecture: 8\n\
             CPU variant\t: 0x{:01x}\nCPU part\t: 0x{:03x}\nCPU revision\t: {rev}\n\n",
            (midr >> 24) & 0xff,
            (midr >> 20) & 0xf,
            (midr >> 4) & 0xfff,
        );
    }
    s
}

/// The page with `mov x8, #139; svc #0` that a handler without `SA_RESTORER` returns to,
/// `setup_sigtramp()`.
static SIGTRAMP: OnceLock<u64> = OnceLock::new();

fn map_sigtramp(space: &GuestSpace) -> Result<u64, String> {
    let kind = MapKind { anon: true, ..MapKind::default() };
    let err = |e| format!("mmap: {}", crate::strerror(e));
    let addr = space.mmap(0, PAGE_SIZE, page::READ | page::WRITE, kind, None, 0).map_err(err)?;
    let mut code = [0u8; 8];
    put32(&mut code, 0, 0xd280_1168);
    put32(&mut code, 4, 0xd400_0001);
    space.write_raw(addr, &code);
    space.mprotect(addr, PAGE_SIZE, page::READ | page::EXEC).map_err(err)?;
    Ok(addr)
}

// struct target_rt_sigframe: siginfo, then the ucontext.
const UC: usize = 128;
const UC_STACK: usize = UC + 16;
const UC_SIGMASK: usize = UC + 40;
/// `uc_mcontext`.
const MC: usize = UC + 176;
const MC_FAULT: usize = MC;
const MC_REGS: usize = MC + 8;
const MC_SP: usize = MC + 256;
const MC_PC: usize = MC + 264;
const MC_PSTATE: usize = MC + 272;
/// `uc_mcontext.__reserved`, where the records start.
const RESERVED: usize = MC + 288;
/// `sizeof(struct target_rt_sigframe)`.
const FRAME_SIZE: usize = RESERVED + 4096;
/// The end of the standard space, less the end marker and an extra record.
const STD_SIZE: usize = FRAME_SIZE - 8;

const FPSIMD_MAGIC: u32 = 0x4650_8001;
const FPSIMD_SIZE: usize = 528;
const ESR_MAGIC: u32 = 0x4553_5201;
const ESR_SIZE: usize = 16;
const EXTRA_MAGIC: u32 = 0x4558_5401;
const EXTRA_SIZE: usize = 32;
const SVE_MAGIC: u32 = 0x5356_4501;
const SVE_HEADER: usize = 16;
const SVE_FLAG_SM: u16 = 1;

/// `TARGET_SVE_SIG_ZREG_OFFSET()`.
fn sve_zreg(vq: usize, n: usize) -> usize {
    SVE_HEADER + vq * 16 * n
}

/// `TARGET_SVE_SIG_PREG_OFFSET()`.
fn sve_preg(vq: usize, n: usize) -> usize {
    sve_zreg(vq, 32) + vq * 2 * n
}

/// `TARGET_SVE_SIG_CONTEXT_SIZE()`.
fn sve_size(vq: usize) -> usize {
    sve_preg(vq, 17)
}

/// `target_sigframe_layout`.
#[derive(Default)]
struct Layout {
    total: usize,
    extra_base: usize,
    extra_size: usize,
    std_end: usize,
    extra_ofs: usize,
    extra_end: usize,
}

impl Layout {
    /// `alloc_sigframe_space()`.
    fn alloc(&mut self, this_size: usize) -> usize {
        let mut loc = self.total;
        if self.extra_base != 0 {
            self.extra_size += this_size;
        } else if this_size + loc > STD_SIZE {
            // Too big for the standard space: an extra record with its own end marker there.
            self.extra_ofs = loc;
            self.total += EXTRA_SIZE;
            self.std_end = self.total;
            self.total += 8;
            self.extra_base = self.total;
            loc = self.total;
            self.extra_size = this_size;
        }
        self.total += this_size;
        loc
    }
}

/// `target_setup_frame()`: the frame for the handler of `sig` on the guest stack, and the
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
    let mut st = CpuArmState::load(cpu.env);
    let mut l = Layout { total: RESERVED, ..Layout::default() };
    let fpsimd_ofs = l.alloc(FPSIMD_SIZE);
    let esr_ofs = if st.esr_el[1] != 0 { l.alloc(ESR_SIZE) } else { 0 };
    let vq = user::sve_vq(cpu, &st).map(|v| v as usize);
    let sve = vq.map(|vq| {
        let size = (sve_size(vq) + 15) & !15;
        (l.alloc(size), size, vq)
    });
    if l.extra_ofs != 0 {
        l.extra_end = l.alloc(8);
    } else {
        l.std_end = l.total;
        l.total += 8;
    }
    l.total = l.total.max(FRAME_SIZE);
    let fr_ofs = l.total;
    l.total += 16;

    // get_sigframe().
    let sp = st.xregs[31];
    let frame = signal::target_sigsp(t, sp, sa).wrapping_sub(l.total as u64) & !15;
    let mut f = vec![0u8; l.total];
    if !space.check(frame, l.total as u64, page::WRITE) || !space.read_raw(frame, &mut f) {
        signal::force_sigsegv(t, sig);
        return;
    }
    let return_addr = if sa.flags & signal::SA_RESTORER != 0 {
        sa.restorer
    } else {
        SIGTRAMP.get().copied().unwrap_or(0)
    };

    // target_setup_general_frame().
    put64(&mut f, UC, 0);
    put64(&mut f, UC + 8, 0);
    f[UC_STACK..UC_STACK + 24].copy_from_slice(&signal::save_altstack(t, sp));
    for i in 0..31 {
        put64(&mut f, MC_REGS + 8 * i, st.xregs[i]);
    }
    put64(&mut f, MC_SP, sp);
    put64(&mut f, MC_PC, st.pc);
    put64(&mut f, MC_PSTATE, u64::from(st.pstate_read()));
    put64(&mut f, MC_FAULT, st.exception_vaddress);
    put64(&mut f, UC_SIGMASK, old);

    // target_setup_fpsimd_record().
    put32(&mut f, fpsimd_ofs, FPSIMD_MAGIC);
    put32(&mut f, fpsimd_ofs + 4, FPSIMD_SIZE as u32);
    put32(&mut f, fpsimd_ofs + 8, vfp_get_fpsr(&st));
    put32(&mut f, fpsimd_ofs + 12, st.fpcr);
    for i in 0..32 {
        put64(&mut f, fpsimd_ofs + 16 + 16 * i, st.zregs[i][0]);
        put64(&mut f, fpsimd_ofs + 24 + 16 * i, st.zregs[i][1]);
    }
    if esr_ofs != 0 {
        put32(&mut f, esr_ofs, ESR_MAGIC);
        put32(&mut f, esr_ofs + 4, ESR_SIZE as u32);
        put64(&mut f, esr_ofs + 8, st.esr_el[1]);
        // Leave ESR_EL1 clear while it's not relevant.
        st.esr_el[1] = 0;
    }
    put64(&mut f, l.std_end, 0);
    if l.extra_ofs != 0 {
        put32(&mut f, l.extra_ofs, EXTRA_MAGIC);
        put32(&mut f, l.extra_ofs + 4, EXTRA_SIZE as u32);
        put64(&mut f, l.extra_ofs + 8, frame + l.extra_base as u64);
        put32(&mut f, l.extra_ofs + 16, l.extra_size as u32);
        put64(&mut f, l.extra_end, 0);
    }
    if let Some((ofs, size, vq)) = sve {
        // target_setup_sve_record().
        let r = &mut f[ofs..ofs + size];
        r[..SVE_HEADER].fill(0);
        put32(r, 0, SVE_MAGIC);
        put32(r, 4, size as u32);
        r[8..10].copy_from_slice(&((vq * 16) as u16).to_le_bytes());
        for i in 0..32 {
            for j in 0..vq * 2 {
                put64(r, sve_zreg(vq, i) + 8 * j, st.zregs[i][j]);
            }
        }
        for i in 0..17 {
            for j in 0..vq {
                let v = (st.pregs[i][j >> 2] >> ((j & 3) * 16)) as u16;
                let at = sve_preg(vq, i) + 2 * j;
                r[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    // The frame record for unwinding.
    put64(&mut f, fr_ofs, st.xregs[29]);
    put64(&mut f, fr_ofs + 8, st.xregs[30]);
    f[..UC].copy_from_slice(info);
    space.write_raw(frame, &f);

    st.xregs[0] = sig as u64;
    st.xregs[1] = frame;
    st.xregs[2] = frame + UC as u64;
    st.xregs[29] = frame + fr_ofs as u64;
    st.xregs[30] = return_addr;
    st.xregs[31] = frame;
    st.pc = sa.handler;
    user::commit(cpu, &mut st);
}

/// Reads the 8-byte record header at `at`: magic and size.
fn record_head(space: &GuestSpace, at: u64) -> Option<(u32, usize)> {
    let mut h = [0u8; 8];
    space.read(at, &mut h).then(|| (get32(&h, 0), get32(&h, 4) as usize))
}

/// `target_restore_sigframe()` past the general frame: the FPSIMD and SVE state from the
/// records at `frame + RESERVED`. False when the frame is bad.
fn restore_records(space: &GuestSpace, cpu: &Cpu<'_>, st: &mut CpuArmState, frame: u64) -> bool {
    let mut fpsimd = None;
    let mut sve: Option<(u64, usize)> = None;
    let mut extra: Option<u64> = None;
    let mut used_extra = false;
    let mut at = Some(frame + RESERVED as u64);
    while let Some(ctx) = at {
        let Some((magic, size)) = record_head(space, ctx) else { return false };
        match magic {
            0 => {
                if size != 0 {
                    return false;
                }
                if used_extra {
                    at = None;
                } else {
                    at = extra;
                    used_extra = true;
                }
                continue;
            }
            FPSIMD_MAGIC => {
                if fpsimd.is_some() || size != FPSIMD_SIZE {
                    return false;
                }
                fpsimd = Some(ctx);
            }
            ESR_MAGIC => {}
            SVE_MAGIC => {
                if sve.is_some() || size < SVE_HEADER {
                    return false;
                }
                sve = Some((ctx, size));
            }
            EXTRA_MAGIC => {
                if extra.is_some() || size != EXTRA_SIZE {
                    return false;
                }
                let mut e = [0u8; EXTRA_SIZE];
                if !space.read(ctx, &mut e) {
                    return false;
                }
                let (datap, len) = (get64(&e, 8), u64::from(get32(&e, 16)));
                if !space.check(datap, len, page::READ) {
                    return false;
                }
                extra = Some(datap);
            }
            _ => return false,
        }
        // A record smaller than its header would never end.
        if size < 8 {
            return false;
        }
        at = Some(ctx + size as u64);
    }

    // Require FPSIMD always.
    let Some(fp) = fpsimd else { return false };
    let mut r = [0u8; FPSIMD_SIZE];
    if !space.read(fp, &mut r) {
        return false;
    }
    vfp_set_fpsr(st, get32(&r, 8));
    user::set_fpcr(cpu, st, get32(&r, 12));
    for i in 0..32 {
        st.zregs[i][0] = get64(&r, 16 + 16 * i);
        st.zregs[i][1] = get64(&r, 24 + 16 * i);
    }

    // SVE data, if present, overwrites FPSIMD data.
    if let Some((ctx, size)) = sve {
        let mut h = [0u8; SVE_HEADER];
        if !space.read(ctx, &mut h) {
            return false;
        }
        let vl = u16::from_le_bytes([h[8], h[9]]) as usize;
        let flags = u16::from_le_bytes([h[10], h[11]]);
        // There is no streaming mode without SME.
        if flags & SVE_FLAG_SM != 0 {
            return false;
        }
        let Some(vq) = user::sve_vq(cpu, st).map(|v| v as usize) else { return false };
        if vl != vq * 16 {
            return false;
        }
        // Accept an empty record.
        if size <= SVE_HEADER {
            return true;
        }
        if size < sve_size(vq) {
            return false;
        }
        let mut r = vec![0u8; sve_size(vq)];
        if !space.read(ctx, &mut r) {
            return false;
        }
        for i in 0..32 {
            for j in 0..vq * 2 {
                st.zregs[i][j] = get64(&r, sve_zreg(vq, i) + 8 * j);
            }
        }
        for i in 0..17 {
            for j in 0..vq {
                let at = sve_preg(vq, i) + 2 * j;
                let v = u64::from(u16::from_le_bytes([r[at], r[at + 1]]));
                if j & 3 == 0 {
                    st.pregs[i][j >> 2] = v;
                } else {
                    st.pregs[i][j >> 2] |= v << ((j & 3) * 16);
                }
            }
        }
    }
    true
}

/// `do_rt_sigreturn()`.
fn do_rt_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    let mut st = CpuArmState::load(cpu.env);
    let frame = st.xregs[31];
    let mut f = vec![0u8; FRAME_SIZE];
    if frame & 15 != 0 || !space.read(frame, &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    // target_restore_general_frame().
    signal::set_sigmask(t, signal::t2h_set(get64(&f, UC_SIGMASK)));
    for i in 0..31 {
        st.xregs[i] = get64(&f, MC_REGS + 8 * i);
    }
    st.xregs[31] = get64(&f, MC_SP);
    st.pc = get64(&f, MC_PC);
    // Unlike QEMU, the frame cannot leave EL0t, AArch64 and SP_EL0.
    let pstate = get64(&f, MC_PSTATE) as u32;
    st.pstate_write(pstate & !(PSTATE_M | PSTATE_NRW | PSTATE_SP));
    if !restore_records(space, cpu, &mut st, frame) {
        user::commit(cpu, &mut st);
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    let sp = st.xregs[31];
    user::commit(cpu, &mut st);
    let _ = signal::restore_altstack(t, &f[UC_STACK..UC_STACK + 24], sp);
    signal::ESIGRETURN
}

/// `signal_for_exception()`: the signal of a synchronous exception, with its syndrome kept for
/// the ESR record of the frame.
fn signal_for_exception(t: &mut Task, cpu: &mut Cpu<'_>, addr: u64) {
    let mut st = CpuArmState::load(cpu.env);
    let syn = st.exception_syndrome;
    st.esr_el[1] = u64::from(syn);
    st.store(cpu.env);
    let ec = syn >> syndrome::EC_SHIFT;
    let (sig, code) = match ec {
        syndrome::EC_DATAABORT | syndrome::EC_INSNABORT => match syn & 0x3f {
            0x04..=0x07 => (signal::SIGSEGV, signal::SEGV_MAPERR),
            0x09..=0x0b | 0x0d..=0x0f => (signal::SIGSEGV, signal::SEGV_ACCERR),
            0x11 => (signal::SIGSEGV, signal::SEGV_MTESERR),
            0x21 => (signal::SIGBUS, signal::BUS_ADRALN),
            _ => {
                eprintln!("qemu: unexpected fault status 0x{:x}", syn & 0x3f);
                std::process::abort();
            }
        },
        syndrome::EC_PCALIGNMENT => (signal::SIGBUS, signal::BUS_ADRALN),
        syndrome::EC_UNCATEGORIZED
        | syndrome::EC_SYSTEMREGISTERTRAP
        | EC_SMETRAP
        | EC_BTITRAP
        | syndrome::EC_ILLEGALSTATE => (signal::SIGILL, signal::ILL_ILLOPC),
        syndrome::EC_PACFAIL | EC_MOP => (signal::SIGILL, signal::ILL_ILLOPN),
        EC_GCS => (signal::SIGSEGV, SEGV_CPERR),
        _ => {
            eprintln!("qemu: unexpected exception class 0x{ec:x}");
            std::process::abort();
        }
    };
    signal::force_sig_fault(t, sig, code, addr);
}

/// `cpu_loop()`, until the thread calls `exit` with others left.
pub(crate) fn cpu_loop(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
        cpu.process_queued_cpu_work();
        match trapnr {
            EXCP_SWI => {
                let nr = xreg(cpu, 8);
                let mut args = [0u64; 6];
                for (i, a) in args.iter_mut().enumerate() {
                    *a = xreg(cpu, i);
                }
                let ret = match generic::to_host(nr) {
                    Some(n) => {
                        // CLONE_BACKWARDS: tls before the child's tid pointer.
                        if n == libc::SYS_clone {
                            args.swap(3, 4);
                        }
                        syscall::do_syscall(p, t, cpu, n as u64, args)
                    }
                    None => -ENOSYS,
                };
                if ret == THREAD_EXIT {
                    return;
                }
                if ret == signal::ERESTARTSYS {
                    let pc = ld64(cpu.env, PC);
                    st64(cpu.env, PC, pc.wrapping_sub(4));
                } else if ret != signal::ESIGRETURN {
                    set_xreg(cpu, 0, ret as u64);
                }
            }
            excp::INTERRUPT | excp::YIELD => {}
            EXCP_UDEF => {
                let pc = ld64(cpu.env, PC);
                signal_for_exception(t, cpu, pc);
            }
            EXCP_PREFETCH_ABORT | EXCP_DATA_ABORT => {
                let addr = CpuArmState::load(cpu.env).exception_vaddress;
                signal_for_exception(t, cpu, addr);
            }
            excp::DEBUG | EXCP_BKPT => {
                let pc = ld64(cpu.env, PC);
                signal::force_sig_fault(t, signal::SIGTRAP, signal::TRAP_BRKPT, pc);
            }
            excp::ATOMIC => cpu_exec_step_atomic(cpu),
            _ => {
                eprintln!(
                    "qemu: unhandled CPU exception 0x{trapnr:x} - aborting\nPC={:016x}",
                    ld64(cpu.env, PC)
                );
                std::process::abort();
            }
        }
        // Check for MTE asynchronous faults.
        if ld64(cpu.env, TFSR_EL) != 0 {
            st64(cpu.env, TFSR_EL, 0);
            signal::force_sig_fault(t, signal::SIGSEGV, signal::SEGV_MTEAERR, 0);
        }
        let space = Arc::clone(p.space());
        signal::process_pending_signals(&space, t, cpu);
        // Exception return on AArch64 always clears the exclusive monitor.
        st64(cpu.env, EXCLUSIVE_ADDR, u64::MAX);
    }
}

/// The aarch64 target.
pub(crate) static GUEST: Guest = Guest {
    machine: "aarch64",
    minsigstksz: 2048,
    env_size: ENV_SIZE,
    generic_abi: true,
    sa_restorer: true,
    // O_DIRECTORY, O_NOFOLLOW, O_DIRECT and O_LARGEFILE.
    open_flags: &[
        (0o40000, 0o200000),
        (0o100000, 0o400000),
        (0o200000, 0o40000),
        (0o400000, 0o100000),
    ],
    cpuinfo: Some(cpuinfo),
    sp: |cpu| xreg(cpu, 31),
    clone_regs: |cpu, newsp| {
        if newsp != 0 {
            set_xreg(cpu, 31, newsp);
        }
        set_xreg(cpu, 0, 0);
    },
    set_tls,
    cpu_loop,
    setup_rt_frame,
    rt_sigreturn: do_rt_sigreturn,
};

/// `qemu-aarch64`: the model `-cpu` names.
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
        "aarch64"
    }

    fn layout(&self) -> (u64, u64) {
        (TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE)
    }

    fn select_cpu(&mut self, cpu: &str) -> Result<(), String> {
        let (name, _features) = cpu.split_once(',').unwrap_or((cpu, ""));
        let m = ArmCpuModel::by_name(name)
            .ok_or_else(|| format!("unable to find CPU model '{name}'"))?;
        let (h, h2) = hwcaps(&m);
        let _ = CPUINFO.set((h, h2, m.midr));
        self.model = Some(m);
        Ok(())
    }

    fn arch(&self) -> Arch {
        let (h, h2) = hwcaps(self.model());
        Arch { machine: EM_AARCH64, platform: Some("aarch64"), hwcap: h, hwcap2: Some(h2) }
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
        let mut st = user::user_reset(arm.model());
        // target_cpu_copy_regs().
        st.pc = info.entry & !3;
        st.xregs[31] = info.start_stack;
        if arm.model().features.pauth != 0 {
            // The keys of a new process are random, as the kernel makes them.
            for pair in st.pac_keys.chunks_mut(2) {
                let r = host::random16();
                pair[0] = u64::from_le_bytes(r[..8].try_into().expect("8 bytes"));
                pair[1] = u64::from_le_bytes(r[8..].try_into().expect("8 bytes"));
            }
        }
        let tramp = map_sigtramp(space)?;
        let _ = SIGTRAMP.set(tramp);
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
    fn frame_layout_matches_the_kernel() {
        assert_eq!(RESERVED, 592);
        assert_eq!(FRAME_SIZE, 4688);
        assert_eq!(ELF_ET_DYN_BASE, 0xaaaa_aaaa_b000);
    }

    #[test]
    fn sve_records_spill_into_extra_space() {
        let mut l = Layout { total: RESERVED, ..Layout::default() };
        assert_eq!(l.alloc(FPSIMD_SIZE), RESERVED);
        // A 256-byte vector length does not fit the standard space.
        let size = (sve_size(16) + 15) & !15;
        let at = l.alloc(size);
        assert_ne!(l.extra_ofs, 0);
        assert_eq!(at, l.extra_base);
        assert_eq!(l.extra_size, size);
    }

    #[test]
    fn neoverse_n1_hwcaps() {
        let m = ArmCpuModel::by_name("neoverse-n1").expect("model");
        let (h, _) = hwcaps(&m);
        assert_ne!(h & 1 << 8, 0, "atomics");
        assert_eq!(h & 1 << 22, 0, "no sve");
    }
}
