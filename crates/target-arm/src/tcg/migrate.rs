// SPDX-License-Identifier: GPL-2.0-or-later

//! The vCPU state that the `cpu` section of QEMU 11.1's migration stream carries, from
//! `vmstate_arm_cpu` in target/arm/machine.c.
//!
//! [`save`] reads a vCPU into a [`CpuMig`] and [`load`] puts one back, on the vCPU's thread;
//! the board crate lays a [`CpuMig`] out in the stream. The system registers travel as
//! QEMU's cpreg list: pairs of a KVM style register index and a raw value, sorted by index.
//!
//! Differences from QEMU:
//! - The register list is the one QEMU 11.1 builds for `max` without EL2 and EL3, which is
//!   what the `virt` board has by default, whatever the model. With EL2 or EL3 the registers
//!   of those levels are added. The AArch32 views and the AArch64 registers this port does
//!   not have go out with QEMU's reset values for `max` and are ignored when they come in.
//! - The AArch32 core registers, the AArch32 VFP registers (FPSID, the MVFRs and FPEXC) and
//!   the SError, WFxT, event register and FPMR state do not exist here. They go out as QEMU
//!   sends them for `max` after reset and are dropped on the way in.
//! - The timer expiry times are in the generic counter's time base. On the way in the timers
//!   are recalculated from their registers instead.

use ruvm_jit::{Cpu, interrupt};

use super::sysreg::{self, Kind, PL0_R, PL0_W};
use super::{
    Arm, INTERRUPT_FIQ, INTERRUPT_VFIQ, INTERRUPT_VIRQ, PSCI_OFF, PSCI_ON, arm_of, commit, gtimer,
    ld64, st64, vfp,
};
use crate::cpu::{
    CpuArmState, GTIMER_HYP, GTIMER_HYPVIRT, GTIMER_PHYS, GTIMER_SEC, GTIMER_VIRT, NUM_GTIMERS,
};

/// `CP_REG_ARM64 | CP_REG_SIZE_U64 | CP_REG_ARM64_SYSREG`: the index of an AArch64 system
/// register is this with the encoding in the low 16 bits.
const AARCH64_SYSREG: u64 = 0x6030_0000_0013_0000;

/// The non-secure bank of the 64-bit AArch32 PAR. QEMU's `PAR_EL1` is an alias, so PAR_EL1
/// travels in this entry.
const PAR64_NS: u64 = 0x4030_0000_200f_0380;

/// The AArch64 register an entry of the list is kept in, if ruvm keeps it.
fn aa64_key(index: u64) -> Option<u32> {
    match index {
        PAR64_NS => Some(sysreg::key(3, 0, 7, 4, 0)),
        _ if index & !0xffff == AARCH64_SYSREG => Some(index as u32 & 0xffff),
        _ => None,
    }
}

/// The AArch32 FPSCR bits that are FPSR bits, `FPSCR_FPSR_MASK`.
const FPSCR_FPSR_MASK: u32 = 0xf800_009f;
/// The AArch32 FPSCR bits that are FPCR bits, `FPSCR_FPCR_MASK`.
const FPSCR_FPCR_MASK: u32 = 0x07ff_9f00;

/// QEMU's `vfp.xregs[ARM_VFP_FPSID]` for `max`.
const FPSID: u32 = 0x4103_4070;
/// QEMU's `vfp.xregs[2..16]` for `max` after reset: MVFR2, MVFR1 and MVFR0 are set.
const VFP_XREGS: [u32; 14] = [0, 0, 0, 0x43, 0x1321_1111, 0x1011_0222, 0, 0, 0, 0, 0, 0, 0, 0];

/// `ARMCPU` as the `cpu` section sees it, field by field in stream order.
#[derive(Clone, Debug, Default)]
pub struct CpuMig {
    /// `env.regs`, the AArch32 core registers.
    pub regs: [u32; 16],
    /// `env.xregs`.
    pub xregs: [u64; 32],
    /// `env.pc`.
    pub pc: u64,
    /// `cpsr`: `pstate_read()`.
    pub cpsr: u32,
    /// `env.spsr`, the AArch32 SPSR of the current mode.
    pub spsr: u32,
    /// `env.banked_spsr`.
    pub banked_spsr: [u64; 8],
    /// `env.banked_r13`.
    pub banked_r13: [u32; 8],
    /// `env.banked_r14`.
    pub banked_r14: [u32; 8],
    /// `env.usr_regs`.
    pub usr_regs: [u32; 5],
    /// `env.fiq_regs`.
    pub fiq_regs: [u32; 5],
    /// `env.elr_el`.
    pub elr_el: [u64; 4],
    /// `env.sp_el`.
    pub sp_el: [u64; 4],
    /// `cpreg_vmstate_array_len`.
    pub cpreg_len: i32,
    /// `cpreg_vmstate_indexes`.
    pub cpreg_indexes: Vec<u64>,
    /// `cpreg_vmstate_values`.
    pub cpreg_values: Vec<u64>,
    /// `env.exclusive_addr`.
    pub exclusive_addr: u64,
    /// `env.exclusive_val`.
    pub exclusive_val: u64,
    /// `env.exclusive_high`.
    pub exclusive_high: u64,
    /// The low half of `env.exception.syndrome`.
    pub syndrome32: u32,
    /// `env.exception.fsr`.
    pub fsr: u32,
    /// `env.exception.vaddress`.
    pub vaddress: u64,
    /// `gt_timer[GTIMER_PHYS]` and `gt_timer[GTIMER_VIRT]`: the expiry times, -1 when off.
    pub gt_timer: [i64; 2],
    /// `power_state == PSCI_OFF`.
    pub powered_off: bool,
    /// `cpu/vfp`: the low 128 bits of each vector register.
    pub zregs_lo: [[u64; 2]; 32],
    /// `env.vfp.xregs[0]`.
    pub fpsid: u32,
    /// `env.vfp.xregs[2..16]`.
    pub vfp_xregs: [u32; 14],
    /// `fpscr`: the FPSCR view of FPCR and FPSR, 0 when they do not fit.
    pub fpscr: u32,
    /// `cpu/vfp/fpcr_fpsr`.
    pub fpcr: u64,
    /// `cpu/vfp/fpcr_fpsr`.
    pub fpsr: u64,
    /// Whether the CPU has SVE, which `cpu/sve` needs.
    pub sve: bool,
    /// `cpu/sve`: bits 128 to 1023 of each vector register, as QEMU sends them.
    pub zregs_hi: [[u64; 14]; 32],
    /// `cpu/sve`: the predicate registers and FFR.
    pub pregs: [[u64; 4]; 17],
    /// `cpu/irq-line-state`.
    pub irq_line_state: u32,
    /// Whether the model is `max`, which has FEAT_WFxT and FEAT_FPMR in QEMU 11.1.
    pub max: bool,
    /// `cpu/wfxt-timer`.
    pub wfxt_timer: i64,
    /// `cpu/syndrome64`: the whole of `env.exception.syndrome`.
    pub syndrome: u64,
    /// `cpu/pstate64`: the whole of PSTATE.
    pub pstate64: u64,
    /// `cpu/event`.
    pub event_register: bool,
    /// `cpu/fpmr`.
    pub fpmr: u64,
    /// `cpu/serror`: pending, has_esr.
    pub serror: [u8; 2],
    /// `cpu/serror`: the ESR.
    pub serror_esr: u64,
}

impl CpuMig {
    /// Whether `cpu/vfp/fpcr_fpsr` is needed: FPCR or FPSR has bits FPSCR cannot hold.
    pub fn fpcr_fpsr_needed(&self) -> bool {
        self.fpcr & !u64::from(FPSCR_FPCR_MASK) != 0 || self.fpsr & !u64::from(FPSCR_FPSR_MASK) != 0
    }

    /// `cpu_pre_load()`: clear what the stream may not carry. The subsections that hold a
    /// wider copy of a field (`cpu/syndrome64`, `cpu/pstate64`) start at 0, which they never
    /// send.
    pub fn pre_load(&mut self) {
        self.fpscr = 0;
        self.fpcr = 0;
        self.fpsr = 0;
        self.irq_line_state = u32::MAX;
        self.syndrome = 0;
        self.pstate64 = 0;
    }
}

/// The raw value of a register in the list, as `read_raw_cp_reg()` gets it; `default`
/// for a register this port does not have.
fn raw_read(cpu: &mut Cpu<'_>, key: u32, default: u64) -> u64 {
    let ops = cpu.ops();
    let arm = arm_of(&ops);
    if let Some((timer, cval)) = timer_slot(key) {
        let st = CpuArmState::load(cpu.env);
        return if cval { st.gt_cval[timer] } else { st.gt_ctl[timer] };
    }
    let Some(r) = sysreg::lookup(key, arm.features()) else {
        return default;
    };
    match r.kind {
        Kind::Field { off, .. } => ld64(cpu.env, off),
        Kind::Model(get) => get(arm.model()),
        Kind::Const(v) => v,
        Kind::Zero => 0,
        Kind::Special if r.access & PL0_R != 0 => sysreg::read(cpu, key),
        _ => default,
    }
}

/// Write the raw value of a register in the list, as `write_raw_cp_reg()` does. Constants
/// and registers this port does not have are left alone.
fn raw_write(cpu: &mut Cpu<'_>, key: u32, value: u64) {
    let ops = cpu.ops();
    let arm = arm_of(&ops);
    if let Some((timer, cval)) = timer_slot(key) {
        // The timers are recalculated once everything is in.
        let mut st = CpuArmState::load(cpu.env);
        if cval {
            st.gt_cval[timer] = value;
        } else {
            st.gt_ctl[timer] = value;
        }
        st.store(cpu.env);
        return;
    }
    let Some(r) = sysreg::lookup(key, arm.features()) else {
        return;
    };
    match r.kind {
        Kind::Field { off, .. } => st64(cpu.env, off, value),
        Kind::Special if r.access & PL0_W != 0 => sysreg::write(cpu, key, value),
        _ => {}
    }
}

/// The timer behind the CTL (`false`) or CVAL (`true`) register `key`, read and written raw
/// as QEMU does rather than through the accessors, which pick a timer by the current EL.
fn timer_slot(key: u32) -> Option<(usize, bool)> {
    let (op0, op1, crn, crm, op2) =
        (key >> 14, (key >> 11) & 7, (key >> 7) & 0xf, (key >> 3) & 0xf, key & 7);
    if op0 != 3 || crn != 14 || !(1..=2).contains(&op2) {
        return None;
    }
    let timer = match (op1, crm) {
        (3, 2) => GTIMER_PHYS,
        (3, 3) => GTIMER_VIRT,
        (4, 2) => GTIMER_HYP,
        (4, 3) => GTIMER_HYPVIRT,
        (7, 2) => GTIMER_SEC,
        _ => return None,
    };
    Some((timer, op2 == 2))
}

/// Whether the EL2 or EL3 register `key` goes in the list: QEMU leaves out the registers
/// with no raw access (the TVAL views, the EL02 and EL12 aliases and the operations).
fn raw_access(arm: &Arm, key: u32) -> bool {
    let op1 = (key >> 11) & 7;
    if op1 == 5 || timer_reg_tval(key) {
        return false;
    }
    sysreg::lookup(key, arm.features()).is_some_and(|r| {
        r.access & PL0_R != 0
            && r.access & PL0_W != 0
            && matches!(r.kind, Kind::Field { .. } | Kind::Special)
    })
}

/// Whether `key` is a timer's TVAL register.
fn timer_reg_tval(key: u32) -> bool {
    (key >> 14) == 3 && (key >> 7) & 0xf == 14 && matches!((key >> 3) & 0xf, 2 | 3) && key & 7 == 0
}

/// `write_cpustate_to_list()`: the register list of the vCPU, sorted by index.
fn cpreg_list(cpu: &mut Cpu<'_>) -> Vec<(u64, u64)> {
    let ops = cpu.ops();
    let arm = arm_of(&ops);
    let mut list: Vec<(u64, u64)> = MAX_CPREGS
        .iter()
        .map(|&(index, reset)| (index, aa64_key(index).map_or(reset, |k| raw_read(cpu, k, reset))))
        .collect();
    let extra: Vec<u32> = sysreg::el2_el3_keys(arm.features())
        .filter(|&k| raw_access(arm, k))
        .filter(|&k| {
            MAX_CPREGS.binary_search_by_key(&(AARCH64_SYSREG | u64::from(k)), |e| e.0).is_err()
        })
        .collect();
    for k in extra {
        list.push((AARCH64_SYSREG | u64::from(k), raw_read(cpu, k, 0)));
    }
    list.sort_by_key(|e| e.0);
    list.dedup_by_key(|e| e.0);
    list
}

/// `cpu_pre_save()` and the fields of `vmstate_arm_cpu`: the state of the vCPU `cpu`.
pub fn save(cpu: &mut Cpu<'_>) -> CpuMig {
    let ops = cpu.ops();
    let arm = arm_of(&ops);
    let list = cpreg_list(cpu);
    let st = CpuArmState::load(cpu.env);
    let shared = cpu.core.shared();
    let mut m = CpuMig {
        xregs: st.xregs,
        pc: st.pc,
        cpsr: st.pstate_read(),
        elr_el: st.elr_el,
        sp_el: st.sp_el,
        cpreg_len: list.len() as i32,
        cpreg_indexes: list.iter().map(|e| e.0).collect(),
        cpreg_values: list.iter().map(|e| e.1).collect(),
        exclusive_addr: st.exclusive_addr,
        exclusive_val: st.exclusive_val,
        exclusive_high: st.exclusive_high,
        syndrome32: st.exception_syndrome,
        syndrome: u64::from(st.exception_syndrome),
        vaddress: st.exception_vaddress,
        gt_timer: [
            gtimer::expiry_ns(arm, &st, GTIMER_PHYS),
            gtimer::expiry_ns(arm, &st, GTIMER_VIRT),
        ],
        powered_off: arm.power_state(shared.cpu_index) == PSCI_OFF,
        fpsid: FPSID,
        vfp_xregs: VFP_XREGS,
        fpcr: u64::from(vfp::get_fpcr(&st)),
        fpsr: u64::from(vfp::get_fpsr(&st)),
        sve: arm.features().sve,
        max: arm.model().name == "max",
        wfxt_timer: -1,
        pstate64: u64::from(st.pstate_read()),
        ..CpuMig::default()
    };
    // aarch64_banked_spsr_index(): SPSR_EL1, SPSR_EL2 and SPSR_EL3 are banked_spsr[1], [6]
    // and [7]; the AArch32 SPSR_ABT, SPSR_UND, SPSR_IRQ and SPSR_FIQ are [2] to [5].
    m.banked_spsr[1] = st.spsr_el[1];
    m.banked_spsr[2..6].copy_from_slice(&st.spsr_aarch32);
    m.banked_spsr[6] = st.spsr_el[2];
    m.banked_spsr[7] = st.spsr_el[3];
    if !m.fpcr_fpsr_needed() {
        m.fpscr = (m.fpsr as u32 & FPSCR_FPSR_MASK) | (m.fpcr as u32 & FPSCR_FPCR_MASK);
    }
    for (i, z) in st.zregs.iter().enumerate() {
        m.zregs_lo[i].copy_from_slice(&z[..2]);
        m.zregs_hi[i].copy_from_slice(&z[2..16]);
    }
    m.pregs = st.pregs;
    let (g, i) = arm.lines(shared.cpu_index);
    m.irq_line_state = (shared.interrupt_request() & (interrupt::HARD | INTERRUPT_FIQ)) | g[i].gic;
    m
}

/// `cpu_post_load()`: put the state `m` that came in into the vCPU `cpu`. Fails with the
/// index of a register in the stream that the vCPU does not have.
pub fn load(cpu: &mut Cpu<'_>, m: &CpuMig) -> Result<(), u64> {
    let ops = cpu.ops();
    let arm = arm_of(&ops);
    let local = cpreg_list(cpu);
    if let Some(&index) =
        m.cpreg_indexes.iter().find(|&&i| local.binary_search_by_key(&i, |e| e.0).is_err())
    {
        return Err(index);
    }
    let mut st = CpuArmState::load(cpu.env);
    st.xregs = m.xregs;
    st.pc = m.pc;
    // The PSTATE bits above 31 are not implemented here.
    st.pstate_write(if m.pstate64 != 0 { m.pstate64 as u32 } else { m.cpsr });
    st.elr_el = m.elr_el;
    st.sp_el = m.sp_el;
    st.spsr_el[1] = m.banked_spsr[1];
    st.spsr_aarch32.copy_from_slice(&m.banked_spsr[2..6]);
    st.spsr_el[2] = m.banked_spsr[6];
    st.spsr_el[3] = m.banked_spsr[7];
    st.exclusive_addr = m.exclusive_addr;
    st.exclusive_val = m.exclusive_val;
    st.exclusive_high = m.exclusive_high;
    // This port keeps a 32 bit syndrome.
    st.exception_syndrome = if m.syndrome != 0 { m.syndrome as u32 } else { m.syndrome32 };
    st.exception_vaddress = m.vaddress;
    for (i, z) in st.zregs.iter_mut().enumerate() {
        z[..2].copy_from_slice(&m.zregs_lo[i]);
        if m.sve {
            z[2..16].copy_from_slice(&m.zregs_hi[i]);
        }
    }
    if m.sve {
        st.pregs = m.pregs;
    }
    // get_fpscr() only takes a non-zero FPSCR; the subsection sets FPCR and FPSR whole.
    let (fpcr, fpsr) = if m.fpscr != 0 {
        let fpcr = (vfp::get_fpcr(&st) & !FPSCR_FPCR_MASK) | (m.fpscr & FPSCR_FPCR_MASK);
        (fpcr, m.fpscr & FPSCR_FPSR_MASK)
    } else {
        (m.fpcr as u32, m.fpsr as u32)
    };
    vfp::set_fpcr(&mut st, fpcr, arm.features());
    vfp::set_fpsr(&mut st, fpsr);
    st.store(cpu.env);
    for (&index, &value) in m.cpreg_indexes.iter().zip(&m.cpreg_values) {
        if let Some(k) = aa64_key(index) {
            raw_write(cpu, k, value);
        }
    }
    {
        let shared = cpu.core.shared();
        let (mut g, i) = arm.lines(shared.cpu_index);
        g[i].power = if m.powered_off { PSCI_OFF } else { PSCI_ON };
        // cpu_post_load(): without the subsection the lines are what the interrupt
        // requests say.
        let lines = if m.irq_line_state == u32::MAX {
            shared.interrupt_request()
        } else {
            m.irq_line_state
        };
        g[i].gic = lines & (INTERRUPT_VIRQ | INTERRUPT_VFIQ);
    }
    let mut st = CpuArmState::load(cpu.env);
    let f = arm.features();
    for timer in 0..NUM_GTIMERS {
        let has = match timer {
            GTIMER_HYP | GTIMER_HYPVIRT => f.el2,
            GTIMER_SEC => f.el3,
            _ => true,
        };
        if has {
            gtimer::recalc(arm, cpu, &mut st, timer);
        }
    }
    commit(cpu, &mut st);
    arm.update_virt_lines(cpu, &st);
    Ok(())
}

/// The register list QEMU 11.1 builds for `max` on `virt` without EL2 and EL3: each index
/// with the value after reset. The AArch32 entries (`0x4020...` and `0x4030...`) other
/// than PAR64 and the AArch64 ones this port does not have go out with these values.
static MAX_CPREGS: [(u64, u64); 362] = [
    (0x4020_0000_000f_6800, 0x0),
    (0x4020_0000_000f_6801, 0x0),
    (0x4020_0000_200e_0000, 0x351a_d000),
    (0x4020_0000_200e_0002, 0x0),
    (0x4020_0000_200e_0004, 0x0),
    (0x4020_0000_200e_0005, 0x0),
    (0x4020_0000_200e_0006, 0x0),
    (0x4020_0000_200e_0007, 0x0),
    (0x4020_0000_200e_0038, 0x0),
    (0x4020_0000_200e_0084, 0x0),
    (0x4020_0000_200e_0085, 0x0),
    (0x4020_0000_200e_0086, 0x0),
    (0x4020_0000_200e_0087, 0x0),
    (0x4020_0000_200e_0100, 0x0),
    (0x4020_0000_200e_0102, 0x0),
    (0x4020_0000_200e_0104, 0x0),
    (0x4020_0000_200e_0105, 0x0),
    (0x4020_0000_200e_0106, 0x0),
    (0x4020_0000_200e_0107, 0x0),
    (0x4020_0000_200e_0182, 0x0),
    (0x4020_0000_200e_0184, 0x0),
    (0x4020_0000_200e_0185, 0x0),
    (0x4020_0000_200e_0186, 0x0),
    (0x4020_0000_200e_0187, 0x0),
    (0x4020_0000_200e_0204, 0x0),
    (0x4020_0000_200e_0205, 0x0),
    (0x4020_0000_200e_0280, 0x0),
    (0x4020_0000_200e_0284, 0x0),
    (0x4020_0000_200e_0285, 0x0),
    (0x4020_0000_200e_0302, 0x0),
    (0x4020_0000_200e_0380, 0x0),
    (0x4020_0000_200e_0800, 0x0),
    (0x4020_0000_200e_0838, 0x0),
    (0x4020_0000_200e_0884, 0xa),
    (0x4020_0000_200e_0984, 0x0),
    (0x4020_0000_200e_1000, 0x0),
    (0x4020_0000_200e_1038, 0x0),
    (0x4020_0000_200e_3800, 0x0),
    (0x4020_0000_200e_3801, 0x2),
    (0x4020_0000_200e_3802, 0x11_0f13),
    (0x4020_0000_200e_3c86, 0x0),
    (0x4020_0000_200f_0001, 0xb444_c004),
    (0x4020_0000_200f_0002, 0x0),
    (0x4020_0000_200f_0003, 0x0),
    (0x4020_0000_200f_0006, 0x0),
    (0x4020_0000_200f_0009, 0x123),
    (0x4020_0000_200f_000f, 0x0),
    (0x4020_0000_200f_0080, 0x1102_0131),
    (0x4020_0000_200f_0082, 0x600_000a),
    (0x4020_0000_200f_0083, 0x0),
    (0x4020_0000_200f_0084, 0x1010_1105),
    (0x4020_0000_200f_0085, 0x4000_0000),
    (0x4020_0000_200f_0086, 0x126_0000),
    (0x4020_0000_200f_0087, 0x212_2211),
    (0x4020_0000_200f_0100, 0x210_1110),
    (0x4020_0000_200f_0101, 0x1311_2111),
    (0x4020_0000_200f_0102, 0x2123_2042),
    (0x4020_0000_200f_0103, 0x111_2131),
    (0x4020_0000_200f_0104, 0x1_1142),
    (0x4020_0000_200f_0105, 0x1101_1121),
    (0x4020_0000_200f_0106, 0x2002_1110),
    (0x4020_0000_200f_0107, 0x111_1111),
    (0x4020_0000_200f_0180, 0x0),
    (0x4020_0000_200f_0181, 0x0),
    (0x4020_0000_200f_0182, 0x0),
    (0x4020_0000_200f_0183, 0x0),
    (0x4020_0000_200f_0184, 0x11),
    (0x4020_0000_200f_0185, 0x10),
    (0x4020_0000_200f_0186, 0x2),
    (0x4020_0000_200f_0187, 0x0),
    (0x4020_0000_200f_0200, 0x0),
    (0x4020_0000_200f_0280, 0x0),
    (0x4020_0000_200f_0300, 0x0),
    (0x4020_0000_200f_0380, 0x0),
    (0x4020_0000_200f_0400, 0x0),
    (0x4020_0000_200f_0480, 0x0),
    (0x4020_0000_200f_0500, 0x0),
    (0x4020_0000_200f_0580, 0x0),
    (0x4020_0000_200f_0600, 0x0),
    (0x4020_0000_200f_0680, 0x0),
    (0x4020_0000_200f_0700, 0x0),
    (0x4020_0000_200f_0780, 0x0),
    (0x4020_0000_200f_0801, 0x0),
    (0x4020_0000_200f_0802, 0x0),
    (0x4020_0000_200f_0803, 0x0),
    (0x4020_0000_200f_0821, 0x0),
    (0x4020_0000_200f_0823, 0x0),
    (0x4020_0000_200f_0882, 0xc00),
    (0x4020_0000_200f_1800, 0x0),
    (0x4020_0000_200f_2801, 0x0),
    (0x4020_0000_200f_2880, 0x0),
    (0x4020_0000_200f_2881, 0x0),
    (0x4020_0000_200f_2923, 0x0),
    (0x4020_0000_200f_2980, 0x0),
    (0x4020_0000_200f_3000, 0x0),
    (0x4020_0000_200f_3001, 0x0),
    (0x4020_0000_200f_3002, 0x0),
    (0x4020_0000_200f_480a, 0x0),
    (0x4020_0000_200f_480b, 0x0),
    (0x4020_0000_200f_4e03, 0x0),
    (0x4020_0000_200f_4e06, 0x2_0001),
    (0x4020_0000_200f_4e07, 0x1000_0018),
    (0x4020_0000_200f_4f00, 0x0),
    (0x4020_0000_200f_4f04, 0x0),
    (0x4020_0000_200f_4f05, 0x0),
    (0x4020_0000_200f_4f06, 0x0),
    (0x4020_0000_200f_5100, 0x0),
    (0x4020_0000_200f_5101, 0x0),
    (0x4020_0000_200f_5180, 0x0),
    (0x4020_0000_200f_5181, 0x0),
    (0x4020_0000_200f_6001, 0x0),
    (0x4020_0000_200f_6002, 0x1),
    (0x4020_0000_200f_6081, 0x0),
    (0x4020_0000_200f_60a1, 0x0),
    (0x4020_0000_200f_6800, 0x0),
    (0x4020_0000_200f_6801, 0x0),
    (0x4020_0000_200f_6802, 0x0),
    (0x4020_0000_200f_6803, 0x0),
    (0x4020_0000_200f_6804, 0x0),
    (0x4020_0000_200f_7080, 0x0),
    (0x4020_0000_200f_7808, 0x0),
    (0x4020_0000_200f_7988, 0x800_0000),
    (0x4030_0000_200e_0080, 0x0),
    (0x4030_0000_200e_0100, 0x0),
    (0x4030_0000_200f_0380, 0x0),
    (0x4030_0000_200f_0780, 0x0),
    (0x4030_0000_200f_0788, 0x0),
    (0x4030_0000_200f_0790, 0x0),
    (0x4030_0000_200f_0798, 0x0),
    (0x6030_0000_0013_8002, 0x0),
    (0x6030_0000_0013_8004, 0x0),
    (0x6030_0000_0013_8005, 0x0),
    (0x6030_0000_0013_8006, 0x0),
    (0x6030_0000_0013_8007, 0x0),
    (0x6030_0000_0013_800c, 0x0),
    (0x6030_0000_0013_800d, 0x0),
    (0x6030_0000_0013_800e, 0x0),
    (0x6030_0000_0013_800f, 0x0),
    (0x6030_0000_0013_8010, 0x0),
    (0x6030_0000_0013_8012, 0x0),
    (0x6030_0000_0013_8014, 0x0),
    (0x6030_0000_0013_8015, 0x0),
    (0x6030_0000_0013_8016, 0x0),
    (0x6030_0000_0013_8017, 0x0),
    (0x6030_0000_0013_801a, 0x0),
    (0x6030_0000_0013_801c, 0x0),
    (0x6030_0000_0013_801d, 0x0),
    (0x6030_0000_0013_801e, 0x0),
    (0x6030_0000_0013_801f, 0x0),
    (0x6030_0000_0013_8024, 0x0),
    (0x6030_0000_0013_8025, 0x0),
    (0x6030_0000_0013_802c, 0x0),
    (0x6030_0000_0013_802d, 0x0),
    (0x6030_0000_0013_8032, 0x0),
    (0x6030_0000_0013_8080, 0x0),
    (0x6030_0000_0013_808c, 0xa),
    (0x6030_0000_0013_809c, 0x0),
    (0x6030_0000_0013_83ce, 0x0),
    (0x6030_0000_0013_9808, 0x0),
    (0x6030_0000_0013_9820, 0x0),
    (0x6030_0000_0013_9828, 0x0),
    (0x6030_0000_0013_a038, 0x0),
    (0x6030_0000_0013_c006, 0x0),
    (0x6030_0000_0013_c008, 0x1102_0131),
    (0x6030_0000_0013_c00a, 0x600_000a),
    (0x6030_0000_0013_c00b, 0x0),
    (0x6030_0000_0013_c00c, 0x1010_1105),
    (0x6030_0000_0013_c00d, 0x4000_0000),
    (0x6030_0000_0013_c00e, 0x126_0000),
    (0x6030_0000_0013_c00f, 0x212_2211),
    (0x6030_0000_0013_c010, 0x210_1110),
    (0x6030_0000_0013_c011, 0x1311_2111),
    (0x6030_0000_0013_c012, 0x2123_2042),
    (0x6030_0000_0013_c013, 0x111_2131),
    (0x6030_0000_0013_c014, 0x1_1142),
    (0x6030_0000_0013_c015, 0x1101_1121),
    (0x6030_0000_0013_c016, 0x2002_1110),
    (0x6030_0000_0013_c017, 0x111_1111),
    (0x6030_0000_0013_c018, 0x1011_0222),
    (0x6030_0000_0013_c019, 0x1321_1111),
    (0x6030_0000_0013_c01a, 0x43),
    (0x6030_0000_0013_c01b, 0x0),
    (0x6030_0000_0013_c01c, 0x11),
    (0x6030_0000_0013_c01d, 0x10),
    (0x6030_0000_0013_c01e, 0x2),
    (0x6030_0000_0013_c01f, 0x0),
    (0x6030_0000_0013_c021, 0x10_1010_1200_0121),
    (0x6030_0000_0013_c022, 0x1_0000_0111),
    (0x6030_0000_0013_c023, 0x0),
    (0x6030_0000_0013_c024, 0x110_1101_0121_0022),
    (0x6030_0000_0013_c025, 0x92f1_5fff_7180_0001),
    (0x6030_0000_0013_c026, 0x0),
    (0x6030_0000_0013_c027, 0xfc00_0003),
    (0x6030_0000_0013_c028, 0x1000_0000_1030_560a),
    (0x6030_0000_0013_c029, 0x0),
    (0x6030_0000_0013_c02a, 0x0),
    (0x6030_0000_0013_c02b, 0x0),
    (0x6030_0000_0013_c02c, 0x0),
    (0x6030_0000_0013_c02d, 0x0),
    (0x6030_0000_0013_c02e, 0x0),
    (0x6030_0000_0013_c02f, 0x0),
    (0x6030_0000_0013_c031, 0x111_2111_1021_1502),
    (0x6030_0000_0013_c032, 0x1120_0000_0011_0012),
    (0x6030_0000_0013_c033, 0x1000_0010),
    (0x6030_0000_0013_c034, 0x0),
    (0x6030_0000_0013_c035, 0x0),
    (0x6030_0000_0013_c036, 0x0),
    (0x6030_0000_0013_c037, 0x0),
    (0x6030_0000_0013_c038, 0x2100_0323_1020_1126),
    (0x6030_0000_0013_c039, 0x110_1120_1031_2122),
    (0x6030_0000_0013_c03a, 0x1221_0111_1201_1011),
    (0x6030_0000_0013_c03b, 0x1000_0000_1100_1111),
    (0x6030_0000_0013_c03c, 0x100),
    (0x6030_0000_0013_c03d, 0x0),
    (0x6030_0000_0013_c03e, 0x0),
    (0x6030_0000_0013_c03f, 0x0),
    (0x6030_0000_0013_c080, 0xc5_0838),
    (0x6030_0000_0013_c081, 0x0),
    (0x6030_0000_0013_c082, 0x0),
    (0x6030_0000_0013_c083, 0x0),
    (0x6030_0000_0013_c090, 0x0),
    (0x6030_0000_0013_c094, 0x0),
    (0x6030_0000_0013_c096, 0x0),
    (0x6030_0000_0013_c100, 0x0),
    (0x6030_0000_0013_c101, 0x0),
    (0x6030_0000_0013_c102, 0x0),
    (0x6030_0000_0013_c103, 0x0),
    (0x6030_0000_0013_c108, 0x0),
    (0x6030_0000_0013_c109, 0x0),
    (0x6030_0000_0013_c10a, 0x0),
    (0x6030_0000_0013_c10b, 0x0),
    (0x6030_0000_0013_c110, 0x0),
    (0x6030_0000_0013_c111, 0x0),
    (0x6030_0000_0013_c112, 0x0),
    (0x6030_0000_0013_c113, 0x0),
    (0x6030_0000_0013_c118, 0x0),
    (0x6030_0000_0013_c119, 0x0),
    (0x6030_0000_0013_c128, 0x0),
    (0x6030_0000_0013_c129, 0x0),
    (0x6030_0000_0013_c12a, 0x0),
    (0x6030_0000_0013_c288, 0x0),
    (0x6030_0000_0013_c289, 0x0),
    (0x6030_0000_0013_c290, 0x0),
    (0x6030_0000_0013_c298, 0x0),
    (0x6030_0000_0013_c300, 0x0),
    (0x6030_0000_0013_c4f1, 0x0),
    (0x6030_0000_0013_c4f6, 0x0),
    (0x6030_0000_0013_c510, 0x0),
    (0x6030_0000_0013_c511, 0x0),
    (0x6030_0000_0013_c512, 0x0),
    (0x6030_0000_0013_c513, 0x0),
    (0x6030_0000_0013_c518, 0x0),
    (0x6030_0000_0013_c519, 0x0),
    (0x6030_0000_0013_c520, 0x0),
    (0x6030_0000_0013_c521, 0x0),
    (0x6030_0000_0013_c522, 0x0),
    (0x6030_0000_0013_c523, 0x0),
    (0x6030_0000_0013_c527, 0x0),
    (0x6030_0000_0013_c600, 0x0),
    (0x6030_0000_0013_c601, 0x0),
    (0x6030_0000_0013_c602, 0x1),
    (0x6030_0000_0013_c609, 0x0),
    (0x6030_0000_0013_c681, 0x0),
    (0x6030_0000_0013_c684, 0x0),
    (0x6030_0000_0013_c687, 0x0),
    (0x6030_0000_0013_c708, 0x0),
    (0x6030_0000_0013_c801, 0x123),
    (0x6030_0000_0013_c806, 0x0),
    (0x6030_0000_0013_c807, 0x0),
    (0x6030_0000_0013_cd82, 0x0),
    (0x6030_0000_0013_cd83, 0x0),
    (0x6030_0000_0013_cf80, 0x0),
    (0x6030_0000_0013_cf90, 0x0),
    (0x6030_0000_0013_cf91, 0x0),
    (0x6030_0000_0013_cf92, 0x0),
    (0x6030_0000_0013_cf93, 0x0),
    (0x6030_0000_0013_cf98, 0x800_0000),
    (0x6030_0000_0013_d000, 0x0),
    (0x6030_0000_0013_d801, 0xb444_c004),
    (0x6030_0000_0013_d929, 0x0),
    (0x6030_0000_0013_da12, 0x0),
    (0x6030_0000_0013_da17, 0x0),
    (0x6030_0000_0013_da20, 0x0),
    (0x6030_0000_0013_da21, 0x0),
    (0x6030_0000_0013_da22, 0x0),
    (0x6030_0000_0013_dce0, 0x4101_3000),
    (0x6030_0000_0013_dce1, 0x0),
    (0x6030_0000_0013_dce5, 0x0),
    (0x6030_0000_0013_dce6, 0x2_0001),
    (0x6030_0000_0013_dce7, 0x1000_0018),
    (0x6030_0000_0013_dce8, 0x0),
    (0x6030_0000_0013_de82, 0x0),
    (0x6030_0000_0013_de83, 0x0),
    (0x6030_0000_0013_de85, 0x0),
    (0x6030_0000_0013_de87, 0x0),
    (0x6030_0000_0013_df00, 0x3b9a_ca00),
    (0x6030_0000_0013_df11, 0x0),
    (0x6030_0000_0013_df12, 0x0),
    (0x6030_0000_0013_df19, 0x0),
    (0x6030_0000_0013_df1a, 0x0),
    (0x6030_0000_0013_df40, 0x0),
    (0x6030_0000_0013_df41, 0x0),
    (0x6030_0000_0013_df42, 0x0),
    (0x6030_0000_0013_df43, 0x0),
    (0x6030_0000_0013_df44, 0x0),
    (0x6030_0000_0013_df45, 0x0),
    (0x6030_0000_0013_df60, 0x0),
    (0x6030_0000_0013_df61, 0x0),
    (0x6030_0000_0013_df62, 0x0),
    (0x6030_0000_0013_df63, 0x0),
    (0x6030_0000_0013_df64, 0x0),
    (0x6030_0000_0013_df65, 0x0),
    (0x6030_0000_0013_df7f, 0x0),
    (0x6030_0000_0013_e081, 0x0),
    (0x6030_0000_0013_e083, 0x0),
    (0x6030_0000_0013_e08c, 0x0),
    (0x6030_0000_0013_e08d, 0x0),
    (0x6030_0000_0013_e08e, 0x0),
    (0x6030_0000_0013_e090, 0x0),
    (0x6030_0000_0013_e092, 0x0),
    (0x6030_0000_0013_e095, 0x0),
    (0x6030_0000_0013_e096, 0x0),
    (0x6030_0000_0013_e103, 0x0),
    (0x6030_0000_0013_e110, 0x0),
    (0x6030_0000_0013_e128, 0x0),
    (0x6030_0000_0013_e129, 0x0),
    (0x6030_0000_0013_e180, 0x0),
    (0x6030_0000_0013_e18c, 0x0),
    (0x6030_0000_0013_e18d, 0x0),
    (0x6030_0000_0013_e281, 0x0),
    (0x6030_0000_0013_e293, 0x0),
    (0x6030_0000_0013_e509, 0x0),
    (0x6030_0000_0013_e512, 0x0),
    (0x6030_0000_0013_e513, 0x0),
    (0x6030_0000_0013_e515, 0x0),
    (0x6030_0000_0013_e519, 0x0),
    (0x6030_0000_0013_e540, 0x0),
    (0x6030_0000_0013_e541, 0x0),
    (0x6030_0000_0013_e542, 0x0),
    (0x6030_0000_0013_e543, 0x0),
    (0x6030_0000_0013_e547, 0xf),
    (0x6030_0000_0013_e548, 0x0),
    (0x6030_0000_0013_e549, 0x0),
    (0x6030_0000_0013_e609, 0x0),
    (0x6030_0000_0013_e681, 0x0),
    (0x6030_0000_0013_e687, 0x0),
    (0x6030_0000_0013_e706, 0x0),
    (0x6030_0000_0013_f081, 0x0),
    (0x6030_0000_0013_f083, 0x0),
    (0x6030_0000_0013_f090, 0x0),
    (0x6030_0000_0013_f096, 0x0),
    (0x6030_0000_0013_f099, 0x0),
    (0x6030_0000_0013_f128, 0x0),
    (0x6030_0000_0013_f129, 0x0),
    (0x6030_0000_0013_f509, 0x0),
    (0x6030_0000_0013_f510, 0x0),
    (0x6030_0000_0013_f513, 0x0),
    (0x6030_0000_0013_f519, 0x0),
    (0x6030_0000_0013_f551, 0x0),
    (0x6030_0000_0013_f687, 0x0),
    (0x6030_0000_0013_ff11, 0x0),
    (0x6030_0000_0013_ff12, 0x0),
];

#[cfg(test)]
mod tests {
    use super::{MAX_CPREGS, aa64_key, timer_slot};
    use crate::cpu::ArmCpuModel;
    use crate::tcg::sysreg::{Kind, common_regs, key, lookup};

    #[test]
    fn the_list_is_sorted() {
        assert!(MAX_CPREGS.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn every_register_of_max_with_storage_is_in_the_list() {
        let f = ArmCpuModel::max().features;
        let missing: Vec<&str> = common_regs()
            .iter()
            .filter(|r| (r.feat)(&f) && (r.key >> 11) & 7 != 5)
            .filter(|r| lookup(r.key, &f).is_some_and(|l| matches!(l.kind, Kind::Field { .. })))
            // These travel in the core fields, as in QEMU, where they are aliases.
            .filter(|r| !matches!(r.name, "SPSR_EL1" | "ELR_EL1" | "SP_EL0"))
            .filter(|r| !MAX_CPREGS.iter().any(|e| aa64_key(e.0) == Some(r.key)))
            .map(|r| r.name)
            .collect();
        assert!(missing.is_empty(), "not in the list: {missing:?}");
    }

    #[test]
    fn timers_are_found_by_encoding() {
        assert_eq!(timer_slot(key(3, 3, 14, 2, 1)), Some((0, false)));
        assert_eq!(timer_slot(key(3, 3, 14, 3, 2)), Some((1, true)));
        assert_eq!(timer_slot(key(3, 3, 14, 2, 0)), None);
        assert_eq!(timer_slot(key(3, 5, 14, 2, 1)), None);
    }
}
