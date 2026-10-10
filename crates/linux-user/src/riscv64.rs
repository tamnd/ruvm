// SPDX-License-Identifier: GPL-2.0-or-later

//! The riscv64 target: `linux-user/riscv/cpu_loop.c`, the user mode CPU state of
//! `riscv_cpu_reset_hold()`, the hardware capabilities of `elfload.c`, `/proc/cpuinfo` of
//! `target_proc.h`, `riscv_hwprobe()` of `syscall.c` and the signal frames of
//! `linux-user/riscv/signal.c`.

use std::fmt;
use std::mem::offset_of;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

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
use ruvm_target_riscv::cfg::{CpuBuilder, PropError, model_missing};
use ruvm_target_riscv::cpu::{
    BADADDR, CpuRiscvState, ENV_SIZE, EXCP_BREAKPOINT, EXCP_ILLEGAL_INST, EXCP_INST_ACCESS_FAULT,
    EXCP_INST_PAGE_FAULT, EXCP_LOAD_ACCESS_FAULT, EXCP_LOAD_ADDR_MIS, EXCP_LOAD_PAGE_FAULT,
    EXCP_STORE_AMO_ACCESS_FAULT, EXCP_STORE_AMO_ADDR_MIS, EXCP_STORE_PAGE_FAULT, EXCP_U_ECALL,
    MENVCFG_CBCFE, MENVCFG_CBIE, MENVCFG_CBZE, MSTATUS_FS, MSTATUS_VS, PC, PRV_U, RVA, RVC, RVD,
    RVF, RVI, RVM, RVV, RiscvCfg, env_off,
};
use ruvm_target_riscv::tcg::{Riscv, RiscvBoard, helper_registry, jit_config};
use ruvm_user_common::{GuestSpace, MapKind, PAGE_SIZE, page};

use crate::elf::{Arch, ImageInfo};
use crate::generic;
use crate::guest::Guest;
use crate::host;
use crate::signal::{self, Sigaction, Task, get32, get64, put32, put64};
use crate::start;
use crate::syscall::{self, Proc, THREAD_EXIT};

/// `TASK_UNMAPPED_BASE` for riscv64: a third of the 48-bit space, page aligned.
const TASK_UNMAPPED_BASE: u64 = ((1u64 << 47) / 3 + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
/// `ELF_ET_DYN_BASE` for riscv64.
const ELF_ET_DYN_BASE: u64 = TASK_UNMAPPED_BASE * 2;
/// `EM_RISCV`.
const EM_RISCV: u16 = 243;
/// `ENOSYS`.
const ENOSYS: i64 = libc::ENOSYS as i64;
/// `EINVAL`.
const EINVAL: i64 = libc::EINVAL as i64;
/// `EFAULT`.
const EFAULT: i64 = libc::EFAULT as i64;
/// The generic `renameat`, which riscv64 does not have.
const NR_RENAMEAT: u64 = 38;
/// `TARGET_NR_riscv_hwprobe`.
const NR_RISCV_HWPROBE: u64 = 258;
/// `TARGET_NR_riscv_flush_icache`.
const NR_RISCV_FLUSH_ICACHE: u64 = 259;
/// `MSECCFG_USEED`: `seed` readable from U mode.
const MSECCFG_USEED: u64 = 1 << 8;

/// The env offset of `x0`.
const GPR: usize = env_off(offset_of!(CpuRiscvState, gpr));

/// The vCPU of a user mode guest: the RISC-V front end, with guest pages checked against the
/// guest's mappings instead of page tables.
struct UserCpu {
    rv: Arc<Riscv>,
    space: Arc<GuestSpace>,
}

impl fmt::Debug for UserCpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserCpu").finish_non_exhaustive()
    }
}

impl CpuOps for UserCpu {
    fn translate_code(&self, cpu: &mut Cpu<'_>, tb: &mut TbBuild) -> Result<(), CpuLoopExit> {
        self.rv.translate_code(cpu, tb)
    }

    fn get_tb_cpu_state(&self, cpu: &Cpu<'_>) -> TbCpuState {
        self.rv.get_tb_cpu_state(cpu)
    }

    fn synchronize_from_tb(&self, cpu: &mut Cpu<'_>, tb: &Tb) {
        self.rv.synchronize_from_tb(cpu, tb);
    }

    fn restore_state_to_opc(&self, cpu: &mut Cpu<'_>, tb: &Tb, data: &[u64; INSN_START_WORDS]) {
        self.rv.restore_state_to_opc(cpu, tb, data);
    }

    fn set_pc(&self, cpu: &mut Cpu<'_>, pc: u64) {
        self.rv.set_pc(cpu, pc);
    }

    fn get_pc(&self, cpu: &Cpu<'_>) -> u64 {
        self.rv.get_pc(cpu)
    }

    fn cpu_exec_enter(&self, cpu: &mut Cpu<'_>) {
        self.rv.cpu_exec_enter(cpu);
    }

    fn cpu_exec_exit(&self, cpu: &mut Cpu<'_>) {
        self.rv.cpu_exec_exit(cpu);
    }

    fn cpu_exec_interrupt(&self, cpu: &mut Cpu<'_>, interrupt_request: u32) -> bool {
        self.rv.cpu_exec_interrupt(cpu, interrupt_request)
    }

    fn cpu_exec_halt(&self, cpu: &mut Cpu<'_>) -> bool {
        self.rv.cpu_exec_halt(cpu)
    }

    fn cpu_exec_reset(&self, cpu: &mut Cpu<'_>) {
        self.rv.cpu_exec_reset(cpu);
    }

    fn do_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.rv.do_interrupt(cpu);
    }

    fn fake_user_interrupt(&self, cpu: &mut Cpu<'_>) {
        self.rv.fake_user_interrupt(cpu);
    }

    fn has_work(&self, cpu: &Cpu<'_>) -> bool {
        self.rv.has_work(cpu)
    }

    /// The page is there when the guest mapped it with the access asked for. As on an x86
    /// host, anything mapped can be read. A fault leaves its address in `badaddr` and is a
    /// page fault when nothing is mapped there, an access fault otherwise, for `cpu_loop()`
    /// to raise `SIGSEGV` as the host's fault would.
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
        let excp = match (flags == 0, access_type) {
            (true, MmuAccessType::InstFetch) => EXCP_INST_PAGE_FAULT,
            (true, MmuAccessType::DataLoad) => EXCP_LOAD_PAGE_FAULT,
            (true, MmuAccessType::DataStore) => EXCP_STORE_PAGE_FAULT,
            (false, MmuAccessType::InstFetch) => EXCP_INST_ACCESS_FAULT,
            (false, MmuAccessType::DataLoad) => EXCP_LOAD_ACCESS_FAULT,
            (false, MmuAccessType::DataStore) => EXCP_STORE_AMO_ACCESS_FAULT,
        };
        st64(cpu.env, BADADDR, if canonical { addr } else { 0 });
        Err(cpu.raise_exception(excp, ra))
    }

    fn do_unaligned_access(
        &self,
        cpu: &mut Cpu<'_>,
        addr: u64,
        access_type: MmuAccessType,
        mmu_idx: usize,
        ra: Ra,
    ) -> CpuLoopExit {
        self.rv.do_unaligned_access(cpu, addr, access_type, mmu_idx, ra)
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
        self.rv.do_transaction_failed(
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
        self.rv.mmu_index(cpu, ifetch)
    }

    fn pointer_wrap(&self, cpu: &Cpu<'_>, mmu_idx: usize, result: u64, base: u64) -> u64 {
        self.rv.pointer_wrap(cpu, mmu_idx, result, base)
    }

    fn debug_excp_handler(&self, cpu: &mut Cpu<'_>) {
        self.rv.debug_excp_handler(cpu);
    }

    fn debug_check_watchpoint(&self, cpu: &mut Cpu<'_>, wp: &Watchpoint) -> bool {
        self.rv.debug_check_watchpoint(cpu, wp)
    }

    fn debug_check_breakpoint(&self, cpu: &mut Cpu<'_>) -> bool {
        self.rv.debug_check_breakpoint(cpu)
    }

    fn adjust_watchpoint_address(&self, cpu: &mut Cpu<'_>, addr: u64, len: u64) -> u64 {
        self.rv.adjust_watchpoint_address(cpu, addr, len)
    }

    fn guest_default_memory_order(&self) -> u32 {
        self.rv.guest_default_memory_order()
    }

    fn addr_type(&self) -> Type {
        self.rv.addr_type()
    }

    fn precise_smc(&self) -> bool {
        self.rv.precise_smc()
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        self.rv.as_any()
    }
}

/// The board of a user mode hart: `time` counts host time, as `cpu_get_host_ticks()` does.
struct UserBoard(Instant);

impl RiscvBoard for UserBoard {
    fn rdtime(&self) -> Option<u64> {
        Some(u64::try_from(self.0.elapsed().as_nanos()).unwrap_or(u64::MAX))
    }

    fn timebase_freq(&self) -> u64 {
        1_000_000_000
    }
}

fn ld64(env: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(env[off..off + 8].try_into().expect("8 bytes"))
}

fn st64(env: &mut [u8], off: usize, v: u64) {
    env[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// General register `n`.
fn gpr(cpu: &Cpu<'_>, n: usize) -> u64 {
    ld64(cpu.env, GPR + 8 * n)
}

fn set_gpr(cpu: &mut Cpu<'_>, n: usize, v: u64) {
    st64(cpu.env, GPR + 8 * n, v);
}

/// `ra`.
const X_RA: usize = 1;
/// `sp`.
const X_SP: usize = 2;
/// `tp`.
const X_TP: usize = 4;
/// `a0`.
const X_A0: usize = 10;
/// `a7`.
const X_A7: usize = 17;

/// `cpu_set_tls()`: `tp`.
fn set_tls(cpu: &mut Cpu<'_>, tls: u64) {
    set_gpr(cpu, X_TP, tls);
}

/// `get_elf_hwcap()`: the misa letters the kernel reports.
fn hwcap(cfg: &RiscvCfg) -> u64 {
    cfg.misa_ext() & (RVI | RVM | RVA | RVF | RVD | RVC | RVV)
}

/// The configuration of the selected CPU, for `/proc/cpuinfo` and `riscv_hwprobe()`.
static CFG: OnceLock<RiscvCfg> = OnceLock::new();

fn cfg() -> RiscvCfg {
    CFG.get().copied().unwrap_or_default()
}

/// `open_cpuinfo()` of `target_proc.h` for riscv64.
fn cpuinfo() -> String {
    use std::fmt::Write;
    let cfg = cfg();
    let isa = cfg.isa_string();
    let mmu = if cfg.mmu { "sv48" } else { "none" };
    let mut s = String::new();
    for i in 0..host::online_cpus() {
        let _ = write!(
            s,
            "processor\t: {i}\nhart\t\t: {i}\nisa\t\t: {isa}\nmmu\t\t: {mmu}\nuarch\t\t: qemu\n\n"
        );
    }
    s
}

/// `RISCV_HWPROBE_KEY_IMA_EXT_0` of `cfg` with the misa letters `misa`.
fn ima_ext_0(cfg: &RiscvCfg, misa: u64) -> u64 {
    let has = |l| misa & l != 0;
    let bits = [
        has(RVF) && has(RVD),
        has(RVC),
        has(RVV),
        cfg.ext_zba,
        cfg.ext_zbb,
        cfg.ext_zbs,
        cfg.ext_zicboz,
        cfg.ext_zbc,
        cfg.ext_zbkb,
        cfg.ext_zbkc,
        cfg.ext_zbkx,
        cfg.ext_zknd,
        cfg.ext_zkne,
        cfg.ext_zknh,
        cfg.ext_zksed,
        cfg.ext_zksh,
        cfg.ext_zkt,
        cfg.ext_zvbb,
        cfg.ext_zvbc,
        cfg.ext_zvkb,
        cfg.ext_zvkg,
        cfg.ext_zvkned,
        cfg.ext_zvknha,
        cfg.ext_zvknhb,
        cfg.ext_zvksed,
        cfg.ext_zvksh,
        cfg.ext_zvkt,
        cfg.ext_zfh,
        cfg.ext_zfhmin,
        cfg.ext_zihintntl,
        cfg.ext_zvfh,
        cfg.ext_zvfhmin,
        cfg.ext_zfa,
        cfg.ext_ztso,
        cfg.ext_zacas,
        cfg.ext_zicond,
        cfg.ext_zihintpause,
        cfg.ext_zve32x,
        cfg.ext_zve32f,
        cfg.ext_zve64x,
        cfg.ext_zve64f,
        cfg.ext_zve64d,
        cfg.ext_zimop,
        cfg.ext_zca,
        cfg.ext_zcb,
        cfg.ext_zcd,
        cfg.ext_zcf,
        cfg.ext_zcmop,
        cfg.ext_zawrs,
        cfg.ext_supm,
        cfg.ext_zicntr,
        cfg.ext_zihpm,
        cfg.ext_zfbfmin,
        cfg.ext_zvfbfmin,
        cfg.ext_zvfbfwma,
        cfg.ext_zicbom,
        cfg.ext_zaamo,
        cfg.ext_zalrsc,
        cfg.ext_zabha,
    ];
    bits.iter().enumerate().filter(|&(_, &b)| b).fold(0, |v, (i, _)| v | 1 << i)
}

/// `risc_hwprobe_fill_pairs()`: the value of `key`, `None` for a key it does not know.
fn hwprobe_value(cfg: &RiscvCfg, misa: u64, key: i64) -> Option<u64> {
    let has = |l| misa & l != 0;
    Some(match key {
        0 => u64::from(cfg.mvendorid),
        1 => cfg.marchid,
        2 => cfg.mimpid,
        3 => u64::from(has(RVI) && has(RVM) && has(RVA)),
        4 => ima_ext_0(cfg, misa),
        // RISCV_HWPROBE_MISALIGNED_FAST.
        5 => 3,
        6 if cfg.ext_zicboz => u64::from(cfg.cboz_blocksize),
        12 if cfg.ext_zicbom => u64::from(cfg.cbom_blocksize),
        6 | 12 => 0,
        _ => return None,
    })
}

/// `do_riscv_hwprobe(pairs, pair_count, cpusetsize, cpus, flags)`.
fn do_riscv_hwprobe(space: &GuestSpace, cpu: &Cpu<'_>, a: [u64; 6]) -> i64 {
    let [pairs, count, setsize, cpus, flags, _] = a;
    if flags != 0 {
        return -EINVAL;
    }
    if setsize != 0 {
        // nonempty_cpu_set(): only whether some bit is set matters.
        if !space.check(cpus, setsize, page::READ) {
            return -EFAULT;
        }
        let mut nonempty = false;
        let mut buf = [0u8; 4096];
        let mut at = 0;
        while at < setsize && !nonempty {
            let n = (setsize - at).min(buf.len() as u64) as usize;
            if !space.read(cpus + at, &mut buf[..n]) {
                return -EFAULT;
            }
            nonempty = buf[..n].iter().any(|&b| b != 0);
            at += n as u64;
        }
        if !nonempty {
            return -EINVAL;
        }
    } else if cpus != 0 {
        return -EINVAL;
    }
    if count == 0 {
        return 0;
    }
    let Some(len) = count.checked_mul(16).filter(|&l| space.check(pairs, l, page::WRITE)) else {
        return -EFAULT;
    };
    let cfg = cfg();
    let misa = ld64(cpu.env, env_off(offset_of!(CpuRiscvState, misa)));
    for p in (pairs..pairs + len).step_by(16) {
        let mut pair = [0u8; 16];
        if !space.read(p, &mut pair) {
            return -EFAULT;
        }
        let key = get64(&pair, 0) as i64;
        match hwprobe_value(&cfg, misa, key) {
            Some(v) => put64(&mut pair, 8, v),
            None => {
                put64(&mut pair, 0, u64::MAX);
                put64(&mut pair, 8, 0);
            }
        }
        if !space.write(p, &pair) {
            return -EFAULT;
        }
    }
    0
}

/// The page with `li a7, 139; ecall` that a handler returns to, in place of QEMU's vDSO.
static SIGTRAMP: OnceLock<u64> = OnceLock::new();

fn map_sigtramp(space: &GuestSpace) -> Result<u64, String> {
    let kind = MapKind { anon: true, ..MapKind::default() };
    let err = |e| format!("mmap: {}", crate::strerror(e));
    let addr = space.mmap(0, PAGE_SIZE, page::READ | page::WRITE, kind, None, 0).map_err(err)?;
    let mut code = [0u8; 8];
    put32(&mut code, 0, 0x08b0_0893);
    put32(&mut code, 4, 0x0000_0073);
    space.write_raw(addr, &code);
    space.mprotect(addr, PAGE_SIZE, page::READ | page::EXEC).map_err(err)?;
    Ok(addr)
}

// struct target_rt_sigframe: siginfo, then the ucontext.
const UC: usize = 128;
const UC_STACK: usize = UC + 16;
const UC_SIGMASK: usize = UC + 40;
/// `uc_mcontext`, 16-byte aligned after the 1024-bit `uc_sigmask`.
const MC: usize = UC + 176;
const MC_PC: usize = MC;
/// `gpr[0]` of the context, which is x1: x0 is not saved.
const MC_GPR: usize = MC + 8;
const MC_FPR: usize = MC + 256;
const MC_FCSR: usize = MC + 512;
/// `sizeof(struct target_rt_sigframe)`.
const FRAME_SIZE: usize = (MC_FCSR + 8 + 15) & !15;

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
    let mut st = CpuRiscvState::load(cpu.env);
    // get_sigframe(): a frame that would overflow the alternate stack gets a bogus address.
    let sp = st.gpr[X_SP];
    let size = FRAME_SIZE as u64;
    let frame = if signal::on_sig_stack(t, sp) && !signal::on_sig_stack(t, sp.wrapping_sub(size)) {
        u64::MAX
    } else {
        signal::target_sigsp(t, sp, sa).wrapping_sub(size) & !15
    };
    let mut f = [0u8; FRAME_SIZE];
    if !space.check(frame, size, page::WRITE) || !space.read_raw(frame, &mut f) {
        signal::force_sigsegv(t, sig);
        return;
    }

    // setup_ucontext().
    put64(&mut f, UC, 0);
    put64(&mut f, UC + 8, 0);
    f[UC_STACK..UC_STACK + 24].copy_from_slice(&signal::save_altstack(t, sp));
    put64(&mut f, UC_SIGMASK, old);
    // setup_sigcontext().
    put64(&mut f, MC_PC, st.pc);
    for i in 1..32 {
        put64(&mut f, MC_GPR + 8 * (i - 1), st.gpr[i]);
    }
    for i in 0..32 {
        put64(&mut f, MC_FPR + 8 * i, st.fpr[i]);
    }
    put32(&mut f, MC_FCSR, ((st.fflags & 0x1f) | (st.frm << 5)) as u32);
    f[..UC].copy_from_slice(info);
    space.write_raw(frame, &f);

    st.pc = sa.handler;
    st.gpr[X_SP] = frame;
    st.gpr[X_A0] = sig as u64;
    st.gpr[X_A0 + 1] = frame;
    st.gpr[X_A0 + 2] = frame + UC as u64;
    st.gpr[X_RA] = SIGTRAMP.get().copied().unwrap_or(0);
    st.store(cpu.env);
}

/// `do_rt_sigreturn()`.
fn do_rt_sigreturn(space: &GuestSpace, t: &mut Task, cpu: &mut Cpu<'_>) -> i64 {
    let mut st = CpuRiscvState::load(cpu.env);
    let frame = st.gpr[X_SP];
    let mut f = [0u8; FRAME_SIZE];
    if !space.read(frame, &mut f) {
        signal::force_sig(t, signal::SIGSEGV);
        return signal::ESIGRETURN;
    }
    // restore_ucontext().
    signal::set_sigmask(t, signal::t2h_set(get64(&f, UC_SIGMASK)));
    st.pc = get64(&f, MC_PC);
    for i in 1..32 {
        st.gpr[i] = get64(&f, MC_GPR + 8 * (i - 1));
    }
    for i in 0..32 {
        st.fpr[i] = get64(&f, MC_FPR + 8 * i);
    }
    let fcsr = u64::from(get32(&f, MC_FCSR));
    st.frm = (fcsr >> 5) & 7;
    st.fflags = fcsr & 0x1f;
    let sp = st.gpr[X_SP];
    st.store(cpu.env);
    let _ = signal::restore_altstack(t, &f[UC_STACK..UC_STACK + 24], sp);
    signal::ESIGRETURN
}

/// `cpu_loop()`, until the thread calls `exit` with others left.
pub(crate) fn cpu_loop(p: &Arc<Proc>, t: &mut Task, cpu: &mut Cpu<'_>) {
    loop {
        let trapnr = tcg_cpu_exec(cpu);
        cpu.process_queued_cpu_work();
        match trapnr {
            excp::INTERRUPT | excp::YIELD => {}
            excp::ATOMIC => cpu_exec_step_atomic(cpu),
            EXCP_U_ECALL => {
                let pc = ld64(cpu.env, PC);
                st64(cpu.env, PC, pc.wrapping_add(4));
                let nr = gpr(cpu, X_A7);
                let mut args = [0u64; 6];
                for (i, a) in args.iter_mut().enumerate() {
                    *a = gpr(cpu, X_A0 + i);
                }
                let ret = match (nr, generic::to_host(nr)) {
                    // A no-op: self-modifying code is detected anyway.
                    (NR_RISCV_FLUSH_ICACHE, _) => 0,
                    (NR_RISCV_HWPROBE, _) => do_riscv_hwprobe(p.space(), cpu, args),
                    (NR_RENAMEAT, _) | (_, None) => -ENOSYS,
                    (_, Some(n)) => {
                        // CLONE_BACKWARDS: tls before the child's tid pointer.
                        if n == libc::SYS_clone {
                            args.swap(3, 4);
                        }
                        syscall::do_syscall(p, t, cpu, n as u64, args)
                    }
                };
                if ret == THREAD_EXIT {
                    return;
                }
                if ret == signal::ERESTARTSYS {
                    let pc = ld64(cpu.env, PC);
                    st64(cpu.env, PC, pc.wrapping_sub(4));
                } else if ret != signal::ESIGRETURN {
                    set_gpr(cpu, X_A0, ret as u64);
                }
            }
            EXCP_ILLEGAL_INST => {
                let pc = ld64(cpu.env, PC);
                signal::force_sig_fault(t, signal::SIGILL, signal::ILL_ILLOPC, pc);
            }
            EXCP_BREAKPOINT | excp::DEBUG => {
                let pc = ld64(cpu.env, PC);
                signal::force_sig_fault(t, signal::SIGTRAP, signal::TRAP_BRKPT, pc);
            }
            // The faults the host's SIGSEGV and SIGBUS give QEMU.
            EXCP_INST_PAGE_FAULT | EXCP_LOAD_PAGE_FAULT | EXCP_STORE_PAGE_FAULT => {
                let addr = ld64(cpu.env, BADADDR);
                signal::force_sig_fault(t, signal::SIGSEGV, signal::SEGV_MAPERR, addr);
            }
            EXCP_INST_ACCESS_FAULT | EXCP_LOAD_ACCESS_FAULT | EXCP_STORE_AMO_ACCESS_FAULT => {
                let addr = ld64(cpu.env, BADADDR);
                signal::force_sig_fault(t, signal::SIGSEGV, signal::SEGV_ACCERR, addr);
            }
            EXCP_LOAD_ADDR_MIS | EXCP_STORE_AMO_ADDR_MIS => {
                let addr = ld64(cpu.env, BADADDR);
                signal::force_sig_fault(t, signal::SIGBUS, signal::BUS_ADRALN, addr);
            }
            _ => {
                // EXCP_DUMP() prints with %#x, which has no prefix for 0.
                let n = if trapnr == 0 { "0".to_string() } else { format!("{trapnr:#x}") };
                eprintln!(
                    "\nqemu: unhandled CPU exception {n} - aborting\npc       {:016x}",
                    ld64(cpu.env, PC)
                );
                std::process::exit(1);
            }
        }
        let space = Arc::clone(p.space());
        signal::process_pending_signals(&space, t, cpu);
    }
}

/// The riscv64 target.
pub(crate) static GUEST: Guest = Guest {
    machine: "riscv64",
    minsigstksz: 2048,
    env_size: ENV_SIZE,
    generic_abi: true,
    sa_restorer: false,
    open_flags: &[],
    cpuinfo: Some(cpuinfo),
    sp: |cpu| gpr(cpu, X_SP),
    clone_regs: |cpu, newsp| {
        if newsp != 0 {
            set_gpr(cpu, X_SP, newsp);
        }
        set_gpr(cpu, X_A0, 0);
    },
    set_tls,
    cpu_loop,
    setup_rt_frame,
    rt_sigreturn: do_rt_sigreturn,
};

/// `qemu-riscv64`: the model `-cpu` names, with its properties.
#[derive(Default)]
pub(crate) struct Target {
    cfg: Option<RiscvCfg>,
}

impl Target {
    fn cfg(&self) -> &RiscvCfg {
        self.cfg.as_ref().expect("the CPU is selected first")
    }
}

/// `-cpu model,prop=value,...`: the configuration of the model with its properties set.
fn parse_cpu(arg: &str) -> Result<RiscvCfg, String> {
    let mut parts = arg.split(',');
    let name = parts.next().unwrap_or_default();
    let builder = if model_missing(name).is_empty() { CpuBuilder::new(name) } else { None };
    let mut b = builder.ok_or_else(|| format!("unable to find CPU model '{name}'"))?;
    for feat in parts.filter(|f| !f.is_empty()) {
        let Some((prop, value)) = feat.split_once('=') else {
            return Err(format!("Expected key=value format, found {feat}."));
        };
        let global = format!("{name}-riscv-cpu.{prop}");
        match b.set(prop, value) {
            Ok(()) => {}
            Err(PropError::NotFound) => {
                return Err(format!(
                    "can't apply global {global}={value}: Property '{global}' not found"
                ));
            }
            Err(PropError::Invalid(msg) | PropError::Hinted(msg, _)) => {
                return Err(format!("can't apply global {global}={value}: {msg}"));
            }
            Err(PropError::Unsupported) => {
                return Err(format!("CPU property {prop}={value} is not supported by ruvm yet"));
            }
        }
    }
    let mut warn = Vec::new();
    let cfg = b.finalize(0, &mut warn);
    for w in warn {
        eprintln!("qemu-riscv64: warning: {w}");
    }
    cfg
}

impl start::Target for Target {
    fn guest(&self) -> &'static Guest {
        &GUEST
    }

    fn name(&self) -> &'static str {
        "riscv64"
    }

    fn layout(&self) -> (u64, u64) {
        (TASK_UNMAPPED_BASE, ELF_ET_DYN_BASE)
    }

    fn select_cpu(&mut self, cpu: &str) -> Result<(), String> {
        let cfg = parse_cpu(cpu)?;
        let _ = CFG.set(cfg);
        self.cfg = Some(cfg);
        Ok(())
    }

    fn arch(&self) -> Arch {
        Arch { machine: EM_RISCV, platform: None, hwcap: hwcap(self.cfg()), hwcap2: None }
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
        let cfg = self.cfg.take().expect("the CPU is selected first");
        let rv = Arc::new(Riscv::new().with_cfg(cfg));
        rv.set_board(Arc::new(UserBoard(Instant::now())));
        // riscv_cpu_reset_hold() of a user mode CPU: U mode with the FPU and the vector unit
        // on. The checks QEMU leaves out of user mode, of the counters, `seed` and the cache
        // block operations, are passed by enabling them all.
        let mut st = CpuRiscvState::reset_cfg(0, info.entry, &cfg);
        st.priv_lvl = PRV_U;
        st.mstatus |= MSTATUS_FS | MSTATUS_VS;
        st.menvcfg = MENVCFG_CBIE | MENVCFG_CBCFE | MENVCFG_CBZE;
        st.senvcfg = st.menvcfg;
        st.mcounteren = u64::from(u32::MAX);
        st.scounteren = u64::from(u32::MAX);
        st.mseccfg |= MSECCFG_USEED;
        // init_main_thread().
        st.pc = info.entry;
        st.gpr[X_SP] = info.start_stack;
        let tramp = map_sigtramp(space)?;
        let _ = SIGTRAMP.set(tramp);
        let ops = Arc::new(UserCpu { rv, space: Arc::clone(space) });
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
        assert_eq!(MC, 0x130);
        assert_eq!(FRAME_SIZE, 832);
        assert_eq!(TASK_UNMAPPED_BASE, 0x2aaa_aaaa_b000);
        assert_eq!(ELF_ET_DYN_BASE, 0x5555_5555_6000);
    }

    #[test]
    fn max_hwcaps_and_hwprobe() {
        let cfg = parse_cpu("max").expect("max");
        let h = hwcap(&cfg);
        assert_eq!(h & (RVI | RVM | RVA | RVF | RVD | RVC), RVI | RVM | RVA | RVF | RVD | RVC);
        let misa = cfg.misa_ext();
        assert_eq!(hwprobe_value(&cfg, misa, 3), Some(1));
        assert_eq!(hwprobe_value(&cfg, misa, 4).map(|v| v & 3), Some(3));
        assert_eq!(hwprobe_value(&cfg, misa, 5), Some(3));
        assert_eq!(hwprobe_value(&cfg, misa, 7), None);
        assert!(parse_cpu("nope").is_err());
    }
}
