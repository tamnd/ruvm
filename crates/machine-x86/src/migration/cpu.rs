// SPDX-License-Identifier: GPL-2.0-or-later

//! The `cpu_common` and `cpu` sections of a TCG x86 vCPU, from hw/core/cpu-system.c and
//! target/i386/machine.c.
//!
//! The TCG state buffer only holds the registers the translated code uses. The rest of
//! `CPUX86State` (MTRRs, machine check banks, the KVM paravirtual MSRs, SVM intercepts) lives in
//! a per-vCPU side store that starts from the reset state and keeps what an incoming stream
//! carried, so it goes back out unchanged.

use std::sync::mpsc;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use ruvm_base::{Result, err, error_report};
use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::{Cpu, CpuShared};
use ruvm_target_x86::state::{
    CR0_PE_MASK, DESC_DPL_MASK, DESC_DPL_SHIFT, HF_CPL_MASK, HF_GUEST_MASK, MtrrVar, R_SS,
    SegmentCache, X86CpuState,
};
use ruvm_target_x86::tcg::{X86, env};
use ruvm_vmstate::{EINVAL, VmStateDescription, VmStateField};

/// `EXCP01_DB`.
const EXCP01_DB: i32 = 1;
/// `EXCP0E_PAGE`.
const EXCP0E_PAGE: i32 = 14;
/// `MSR_IA32_MISC_ENABLE_DEFAULT`.
const MISC_ENABLE_DEFAULT: u64 = 1;
/// `V_TPR_MASK`.
const V_TPR_MASK: u32 = 0x0f;

/// Runs `f` on the vCPU's thread and hands back what it returns.
fn on_cpu<R: Send + 'static>(
    shared: &CpuShared,
    f: impl FnOnce(&mut Cpu<'_>) -> R + Send + 'static,
) -> Result<R> {
    let (tx, rx) = mpsc::channel();
    shared.run_on_cpu(move |cpu| {
        let _ = tx.send(f(cpu));
    });
    rx.recv().map_err(|_| err!("vCPU {} did not run the migration request", shared.cpu_index))
}

/// The `CPUState` part of a vCPU.
#[derive(Debug, Clone, Default)]
pub(crate) struct Common {
    halted: u32,
    interrupt_request: u32,
    exception_index: i32,
    crash_occurred: bool,
}

static CPU_COMMON_EXCEPTION_INDEX: LazyLock<VmStateDescription<Common>> = LazyLock::new(|| {
    VmStateDescription::<Common>::new("cpu_common/exception_index")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.exception_index != -1)
        .field(VmStateField::scalar("exception_index", |c: &mut Common| &mut c.exception_index))
});

static CPU_COMMON_CRASH_OCCURRED: LazyLock<VmStateDescription<Common>> = LazyLock::new(|| {
    VmStateDescription::<Common>::new("cpu_common/crash_occurred")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.crash_occurred)
        .field(VmStateField::scalar("crash_occurred", |c: &mut Common| &mut c.crash_occurred))
});

/// `vmstate_cpu_common`.
pub(crate) static VMSTATE_CPU_COMMON: LazyLock<VmStateDescription<Common>> = LazyLock::new(|| {
    VmStateDescription::<Common>::new("cpu_common")
        .version_id(1)
        .minimum_version_id(1)
        .pre_load(|c| {
            c.exception_index = -1;
            0
        })
        .post_load(|c, _| {
            // 0x01 was CPU_INTERRUPT_EXIT.
            c.interrupt_request &= !0x01;
            0
        })
        .fields([
            VmStateField::scalar("halted", |c: &mut Common| &mut c.halted),
            VmStateField::scalar("interrupt_request", |c: &mut Common| &mut c.interrupt_request),
        ])
        .subsection(&CPU_COMMON_EXCEPTION_INDEX)
        .subsection(&CPU_COMMON_CRASH_OCCURRED)
});

/// Reads the `CPUState` part of the vCPU behind `shared`.
pub(crate) fn get_common(shared: &CpuShared) -> Result<Common> {
    on_cpu(shared, |cpu| {
        let s = cpu.shared();
        Common {
            halted: s.halted.load(std::sync::atomic::Ordering::Acquire),
            interrupt_request: s.interrupt_request(),
            exception_index: cpu.core.exception_index,
            crash_occurred: false,
        }
    })
}

/// Loads `c` into the vCPU behind `shared`, with `cpu_common_post_load()`'s TLB flush.
pub(crate) fn put_common(shared: &CpuShared, c: Common) -> Result<()> {
    on_cpu(shared, move |cpu| {
        let s = cpu.shared();
        s.halted.store(c.halted, std::sync::atomic::Ordering::Release);
        s.reset_interrupt(u32::MAX);
        if c.interrupt_request != 0 {
            s.cpu_interrupt(c.interrupt_request);
        }
        cpu.core.exception_index = c.exception_index;
        tlb_flush(cpu);
    })
}

/// `X86CPU` as the `cpu` section sees it: the register state plus the fields that only exist
/// in the stream or that QEMU keeps in a different type.
#[derive(Debug, Clone)]
pub(crate) struct CpuMig {
    s: X86CpuState,
    fpus_vmstate: u16,
    fptag_vmstate: u16,
    fpregs_format: u16,
    vm_vmcb: u64,
    tsc_offset: u64,
    intercept: u64,
    intercept_cr_read: u16,
    intercept_cr_write: u16,
    intercept_dr_read: u16,
    intercept_dr_write: u16,
    intercept_exceptions: u32,
    v_tpr: u8,
    soft_interrupt: u8,
    nmi_injected: u8,
    nmi_pending: u8,
    has_error_code: u8,
    exception_pending: u8,
    exception_injected: u8,
    exception_has_payload: u8,
    triple_fault_pending: u8,
    error_code: i32,
    opmask: [u64; 8],
    nested_cr3: u64,
    nested_pg_mode: u32,
}

impl CpuMig {
    fn new(s: X86CpuState) -> Self {
        let mut c = CpuMig {
            s,
            fpus_vmstate: 0,
            fptag_vmstate: 0,
            fpregs_format: 0,
            vm_vmcb: 0,
            tsc_offset: 0,
            intercept: 0,
            intercept_cr_read: 0,
            intercept_cr_write: 0,
            intercept_dr_read: 0,
            intercept_dr_write: 0,
            intercept_exceptions: 0,
            v_tpr: 0,
            soft_interrupt: 0,
            nmi_injected: 0,
            nmi_pending: 0,
            has_error_code: 0,
            exception_pending: 0,
            exception_injected: 0,
            exception_has_payload: 0,
            triple_fault_pending: 0,
            error_code: 0,
            opmask: [0; 8],
            nested_cr3: 0,
            nested_pg_mode: 0,
        };
        c.copy_out();
        c
    }

    /// Copies the fields QEMU keeps as `uint8_t` or `int32_t` out of the register state.
    fn copy_out(&mut self) {
        let s = &self.s;
        self.soft_interrupt = u8::from(s.soft_interrupt);
        self.nmi_injected = u8::from(s.nmi_injected);
        self.nmi_pending = u8::from(s.nmi_pending);
        self.has_error_code = u8::from(s.has_error_code);
        self.exception_pending = u8::from(s.exception_pending);
        self.exception_injected = u8::from(s.exception_injected);
        self.exception_has_payload = u8::from(s.exception_has_payload);
        self.triple_fault_pending = u8::from(s.triple_fault_pending);
        self.error_code = s.error_code as i32;
    }

    /// The reverse of [`copy_out`](Self::copy_out), after a load.
    fn copy_in(&mut self) {
        let s = &mut self.s;
        s.soft_interrupt = self.soft_interrupt != 0;
        s.nmi_injected = self.nmi_injected != 0;
        s.nmi_pending = self.nmi_pending != 0;
        s.has_error_code = self.has_error_code != 0;
        s.exception_pending = self.exception_pending != 0;
        s.exception_injected = self.exception_injected != 0;
        s.exception_has_payload = self.exception_has_payload != 0;
        s.triple_fault_pending = self.triple_fault_pending != 0;
        s.error_code = self.error_code as u32;
    }

    /// The real mode DPL fix of `cpu_pre_save()` and `cpu_post_load()`.
    fn fix_real_mode_dpl(&mut self) {
        let s = &mut self.s;
        if s.cr0 & CR0_PE_MASK == 0 && (s.segs[1].flags >> DESC_DPL_SHIFT) & 3 != 0 {
            for seg in &mut s.segs {
                seg.flags &= !DESC_DPL_MASK;
            }
        }
    }
}

/// `cpu_pre_save()`.
fn cpu_pre_save(c: &mut CpuMig) -> i32 {
    c.copy_out();
    c.v_tpr = (c.s.int_ctl & V_TPR_MASK) as u8;
    c.fpus_vmstate = (c.s.fpus & !0x3800) | (((c.s.fpstt & 7) as u16) << 11);
    c.fptag_vmstate = 0;
    for (i, &t) in c.s.fptags.iter().enumerate() {
        if t == 0 {
            c.fptag_vmstate |= 1 << i;
        }
    }
    c.fpregs_format = 0;
    c.fix_real_mode_dpl();
    if c.exception_pending != 0 && c.s.hflags & HF_GUEST_MASK == 0 {
        c.exception_pending = 0;
        c.exception_injected = 1;
        if c.exception_has_payload != 0 {
            if c.s.exception_nr == EXCP01_DB {
                c.s.dr[6] = c.s.exception_payload;
            } else if c.s.exception_nr == EXCP0E_PAGE {
                c.s.cr2 = c.s.exception_payload;
            }
        }
    }
    0
}

/// `cpu_post_load()`.
fn cpu_post_load(c: &mut CpuMig, _version_id: i32) -> i32 {
    if c.fpregs_format != 0 {
        error_report("Unsupported old non-softfloat CPU state");
        return -EINVAL;
    }
    c.fix_real_mode_dpl();
    c.s.hflags &= !HF_CPL_MASK;
    c.s.hflags |= (c.s.segs[R_SS].flags >> DESC_DPL_SHIFT) & HF_CPL_MASK;
    if c.s.exception_nr != -1 && c.exception_pending == 0 && c.exception_injected == 0 {
        c.exception_injected = 1;
    }
    c.s.fpstt = u32::from(c.fpus_vmstate >> 11) & 7;
    c.s.fpus = c.fpus_vmstate & !0x3800;
    c.fptag_vmstate ^= 0xff;
    for i in 0..8 {
        c.s.fptags[i] = ((c.fptag_vmstate >> i) & 1) as u8;
    }
    c.copy_in();
    0
}

macro_rules! sc {
    ($name:literal, $($p:tt)+) => {
        VmStateField::scalar($name, |c: &mut CpuMig| &mut c.$($p)+)
    };
}

macro_rules! ar {
    ($name:literal, $($p:tt)+) => {
        VmStateField::array($name, |c: &mut CpuMig| &mut c.$($p)+)
    };
}

/// A subsection with one field, sent when `needed` holds.
macro_rules! sub1 {
    ($stat:ident, $id:literal, $name:literal, |$n:ident| $need:expr, $($p:tt)+) => {
        static $stat: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
            VmStateDescription::new($id)
                .version_id(1)
                .minimum_version_id(1)
                .needed(|$n: &CpuMig| $need)
                .field(sc!($name, $($p)+))
        });
    };
}

static SEGMENT: LazyLock<VmStateDescription<SegmentCache>> = LazyLock::new(|| {
    VmStateDescription::new("segment").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("selector", |s: &mut SegmentCache| &mut s.selector),
        VmStateField::scalar("base", |s: &mut SegmentCache| &mut s.base),
        VmStateField::scalar("limit", |s: &mut SegmentCache| &mut s.limit),
        VmStateField::scalar("flags", |s: &mut SegmentCache| &mut s.flags),
    ])
});

/// `x86_FPReg_tmp`: the 80-bit register as significand and sign plus exponent.
#[derive(Debug, Default)]
struct FpRegTmp {
    mant: u64,
    exp: u16,
}

static FPREG_TMP: LazyLock<VmStateDescription<FpRegTmp>> = LazyLock::new(|| {
    VmStateDescription::new("fpreg_tmp").fields([
        VmStateField::scalar("tmp_mant", |t: &mut FpRegTmp| &mut t.mant),
        VmStateField::scalar("tmp_exp", |t: &mut FpRegTmp| &mut t.exp),
    ])
});

static FPREG: LazyLock<VmStateDescription<[u64; 2]>> = LazyLock::new(|| {
    VmStateDescription::new("fpreg").field(VmStateField::with_tmp(
        &FPREG_TMP,
        |r: &[u64; 2]| FpRegTmp { mant: r[0], exp: r[1] as u16 },
        |r: &mut [u64; 2], t: FpRegTmp| *r = [t.mant, u64::from(t.exp)],
    ))
});

/// A description over the 64-bit lanes `lanes` of a ZMM register.
fn zmm_vmsd(name: &'static str, lanes: &'static [usize]) -> VmStateDescription<[u64; 8]> {
    const NAMES: [&str; 8] = [
        "_q_ZMMReg[0]",
        "_q_ZMMReg[1]",
        "_q_ZMMReg[2]",
        "_q_ZMMReg[3]",
        "_q_ZMMReg[4]",
        "_q_ZMMReg[5]",
        "_q_ZMMReg[6]",
        "_q_ZMMReg[7]",
    ];
    VmStateDescription::new(name).version_id(1).minimum_version_id(1).fields(
        lanes.iter().map(|&l| VmStateField::scalar(NAMES[l], move |z: &mut [u64; 8]| &mut z[l])),
    )
}

static XMM_REG: LazyLock<VmStateDescription<[u64; 8]>> =
    LazyLock::new(|| zmm_vmsd("xmm_reg", &[0, 1]));
static YMMH_REG: LazyLock<VmStateDescription<[u64; 8]>> =
    LazyLock::new(|| zmm_vmsd("ymmh_reg", &[2, 3]));
static ZMMH_REG: LazyLock<VmStateDescription<[u64; 8]>> =
    LazyLock::new(|| zmm_vmsd("zmmh_reg", &[4, 5, 6, 7]));
static HI16_ZMM_REG: LazyLock<VmStateDescription<[u64; 8]>> =
    LazyLock::new(|| zmm_vmsd("hi16_zmm_reg", &[0, 1, 2, 3, 4, 5, 6, 7]));

fn low16(c: &mut CpuMig) -> &mut [[u64; 8]; 16] {
    c.s.xmm_regs.first_chunk_mut().expect("32 vector registers")
}

fn high16(c: &mut CpuMig) -> &mut [[u64; 8]; 16] {
    c.s.xmm_regs.last_chunk_mut().expect("32 vector registers")
}

static MTRR_VAR: LazyLock<VmStateDescription<MtrrVar>> = LazyLock::new(|| {
    VmStateDescription::new("mtrr_var").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("base", |m: &mut MtrrVar| &mut m.base),
        VmStateField::scalar("mask", |m: &mut MtrrVar| &mut m.mask),
    ])
});

static EXCEPTION_INFO: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/exception_info")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.exception_pending != 0 && c.s.hflags & HF_GUEST_MASK != 0)
        .fields([
            sc!("env.exception_pending", exception_pending),
            sc!("env.exception_injected", exception_injected),
            sc!("env.exception_has_payload", exception_has_payload),
            sc!("env.exception_payload", s.exception_payload),
        ])
});

sub1!(ERROR_CODE, "cpu/error_code", "env.error_code", |c| c.has_error_code != 0, error_code);
sub1!(
    STEAL_TIME,
    "cpu/steal_time_msr",
    "env.steal_time_msr",
    |c| c.s.steal_time_msr != 0,
    s.steal_time_msr
);
sub1!(
    ASYNC_PF,
    "cpu/async_pf_msr",
    "env.async_pf_en_msr",
    |c| c.s.async_pf_en_msr != 0,
    s.async_pf_en_msr
);
sub1!(
    ASYNC_PF_INT,
    "cpu/async_pf_int_msr",
    "env.async_pf_int_msr",
    |c| c.s.async_pf_int_msr != 0,
    s.async_pf_int_msr
);
sub1!(
    PV_EOI,
    "cpu/async_pv_eoi_msr",
    "env.pv_eoi_en_msr",
    |c| c.s.pv_eoi_en_msr != 0,
    s.pv_eoi_en_msr
);
sub1!(
    POLL_CONTROL,
    "cpu/poll_control_msr",
    "env.poll_control_msr",
    |c| c.s.poll_control_msr != 1,
    s.poll_control_msr
);

static FPOP_IP_DP: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/fpop_ip_dp")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.s.fpop != 0 || c.s.fpip != 0 || c.s.fpdp != 0)
        .fields([sc!("env.fpop", s.fpop), sc!("env.fpip", s.fpip), sc!("env.fpdp", s.fpdp)])
});

sub1!(TSC_ADJUST, "cpu/msr_tsc_adjust", "env.tsc_adjust", |c| c.s.tsc_adjust != 0, s.tsc_adjust);
sub1!(
    TSC_DEADLINE,
    "cpu/msr_tscdeadline",
    "env.tsc_deadline",
    |c| c.s.tsc_deadline != 0,
    s.tsc_deadline
);
sub1!(
    MISC_ENABLE,
    "cpu/msr_ia32_misc_enable",
    "env.msr_ia32_misc_enable",
    |c| c.s.msr_ia32_misc_enable != MISC_ENABLE_DEFAULT,
    s.msr_ia32_misc_enable
);
sub1!(
    FEATURE_CONTROL,
    "cpu/msr_ia32_feature_control",
    "env.msr_ia32_feature_control",
    |c| c.s.msr_ia32_feature_control != 0,
    s.msr_ia32_feature_control
);

static AVX512: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/avx512")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| {
            c.opmask.iter().any(|&k| k != 0)
                || c.s.xmm_regs[..16].iter().any(|z| z[4..].iter().any(|&q| q != 0))
                || c.s.xmm_regs[16..].iter().any(|z| z.iter().any(|&q| q != 0))
        })
        .fields([
            ar!("env.opmask_regs", opmask),
            VmStateField::struct_array("env.xmm_regs", &ZMMH_REG, low16),
            VmStateField::struct_array("env.xmm_regs", &HI16_ZMM_REG, high16),
        ])
});

sub1!(XSS, "cpu/xss", "env.xss", |c| c.s.xss != 0, s.xss);
sub1!(UMWAIT, "cpu/umwait", "env.umwait", |c| c.s.umwait != 0, s.umwait);
sub1!(
    MSR_SMI_COUNT,
    "cpu/msr_smi_count",
    "env.msr_smi_count",
    |c| c.s.msr_smi_count != 0,
    s.msr_smi_count
);
sub1!(PKRU, "cpu/pkru", "env.pkru", |c| c.s.pkru != 0, s.pkru);
sub1!(PKRS, "cpu/pkrs", "env.pkrs", |c| c.s.pkrs != 0, s.pkrs);
sub1!(SPEC_CTRL, "cpu/spec_ctrl", "env.spec_ctrl", |c| c.s.spec_ctrl != 0, s.spec_ctrl);
sub1!(VIRT_SSBD, "cpu/virt_ssbd", "env.virt_ssbd", |c| c.s.virt_ssbd != 0, s.virt_ssbd);
sub1!(SVM_GUEST, "cpu/svm_guest", "env.int_ctl", |c| c.s.int_ctl != 0, s.int_ctl);
sub1!(MSR_HWCR, "cpu/msr_hwcr", "env.msr_hwcr", |c| c.s.msr_hwcr != 0, s.msr_hwcr);
sub1!(
    TRIPLE_FAULT,
    "cpu/triple_fault",
    "env.triple_fault_pending",
    |c| c.triple_fault_pending != 0,
    triple_fault_pending
);

static SVM_NPT: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    // HF2_NPT_MASK.
    VmStateDescription::<CpuMig>::new("cpu/svn_npt")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.s.hflags2 & (1 << 5) != 0)
        .fields([sc!("env.nested_cr3", nested_cr3), sc!("env.nested_pg_mode", nested_pg_mode)])
});

/// `vmstate_x86_cpu`.
pub(crate) static VMSTATE_X86_CPU: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::new("cpu")
        .version_id(12)
        .minimum_version_id(11)
        .pre_save(cpu_pre_save)
        .post_load(cpu_post_load)
        .fields([
            ar!("env.regs", s.regs),
            sc!("env.eip", s.rip),
            sc!("env.eflags", s.rflags),
            sc!("env.hflags", s.hflags),
            sc!("env.fpuc", s.fpuc),
            sc!("env.fpus_vmstate", fpus_vmstate),
            sc!("env.fptag_vmstate", fptag_vmstate),
            sc!("env.fpregs_format_vmstate", fpregs_format),
            VmStateField::struct_array("env.fpregs", &FPREG, |c: &mut CpuMig| &mut c.s.fpregs),
            VmStateField::struct_array("env.segs", &SEGMENT, |c: &mut CpuMig| &mut c.s.segs),
            VmStateField::structure("env.ldt", &SEGMENT, |c: &mut CpuMig| &mut c.s.ldt),
            VmStateField::structure("env.tr", &SEGMENT, |c: &mut CpuMig| &mut c.s.tr),
            VmStateField::structure("env.gdt", &SEGMENT, |c: &mut CpuMig| &mut c.s.gdt),
            VmStateField::structure("env.idt", &SEGMENT, |c: &mut CpuMig| &mut c.s.idt),
            sc!("env.sysenter_cs", s.sysenter_cs),
            sc!("env.sysenter_esp", s.sysenter_esp),
            sc!("env.sysenter_eip", s.sysenter_eip),
            sc!("env.cr[0]", s.cr0),
            sc!("env.cr[2]", s.cr2),
            sc!("env.cr[3]", s.cr3),
            sc!("env.cr[4]", s.cr4),
            ar!("env.dr", s.dr),
            sc!("env.a20_mask", s.a20_mask),
            sc!("env.mxcsr", s.mxcsr),
            VmStateField::struct_array("env.xmm_regs", &XMM_REG, low16),
            sc!("env.efer", s.efer),
            sc!("env.star", s.star),
            sc!("env.lstar", s.lstar),
            sc!("env.cstar", s.cstar),
            sc!("env.fmask", s.fmask),
            sc!("env.kernelgsbase", s.kernelgsbase),
            sc!("env.smbase", s.smbase),
            sc!("env.pat", s.pat),
            sc!("env.hflags2", s.hflags2),
            sc!("env.vm_hsave", s.vm_hsave),
            sc!("env.vm_vmcb", vm_vmcb),
            sc!("env.tsc_offset", tsc_offset),
            sc!("env.intercept", intercept),
            sc!("env.intercept_cr_read", intercept_cr_read),
            sc!("env.intercept_cr_write", intercept_cr_write),
            sc!("env.intercept_dr_read", intercept_dr_read),
            sc!("env.intercept_dr_write", intercept_dr_write),
            sc!("env.intercept_exceptions", intercept_exceptions),
            sc!("env.v_tpr", v_tpr),
            ar!("env.mtrr_fixed", s.mtrr_fixed),
            sc!("env.mtrr_deftype", s.mtrr_deftype),
            VmStateField::struct_array("env.mtrr_var", &MTRR_VAR, |c: &mut CpuMig| {
                &mut c.s.mtrr_var
            })
            .version(8),
            sc!("env.interrupt_injected", s.interrupt_injected),
            sc!("env.mp_state", s.mp_state),
            sc!("env.tsc", s.tsc),
            sc!("env.exception_nr", s.exception_nr),
            sc!("env.soft_interrupt", soft_interrupt),
            sc!("env.nmi_injected", nmi_injected),
            sc!("env.nmi_pending", nmi_pending),
            sc!("env.has_error_code", has_error_code),
            sc!("env.sipi_vector", s.sipi_vector),
            sc!("env.mcg_cap", s.mcg_cap),
            sc!("env.mcg_status", s.mcg_status),
            sc!("env.mcg_ctl", s.mcg_ctl),
            ar!("env.mce_banks", s.mce_banks),
            sc!("env.tsc_aux", s.tsc_aux),
            sc!("env.system_time_msr", s.system_time_msr),
            sc!("env.wall_clock_msr", s.wall_clock_msr),
            sc!("env.xcr0", s.xcr0).version(12),
            sc!("env.xstate_bv", s.xstate_bv).version(12),
            VmStateField::struct_array("env.xmm_regs", &YMMH_REG, low16).version(12),
        ])
        .subsection(&EXCEPTION_INFO)
        .subsection(&ERROR_CODE)
        .subsection(&ASYNC_PF)
        .subsection(&ASYNC_PF_INT)
        .subsection(&PV_EOI)
        .subsection(&STEAL_TIME)
        .subsection(&POLL_CONTROL)
        .subsection(&FPOP_IP_DP)
        .subsection(&TSC_ADJUST)
        .subsection(&TSC_DEADLINE)
        .subsection(&MISC_ENABLE)
        .subsection(&FEATURE_CONTROL)
        .subsection(&AVX512)
        .subsection(&XSS)
        .subsection(&UMWAIT)
        .subsection(&MSR_SMI_COUNT)
        .subsection(&PKRU)
        .subsection(&PKRS)
        .subsection(&SPEC_CTRL)
        .subsection(&VIRT_SSBD)
        .subsection(&SVM_NPT)
        .subsection(&SVM_GUEST)
        .subsection(&MSR_HWCR)
        .subsection(&TRIPLE_FAULT)
});

/// What `cpu` holds beyond the TCG state buffer, kept between saves and loads.
pub(crate) type SideStore = Arc<Mutex<Option<CpuMig>>>;

/// The reset state of the vCPU `cpu`, which the side store starts from.
fn reset_state(cpu: &Cpu<'_>) -> X86CpuState {
    let ops = cpu.ops();
    let is_bsp = cpu.shared().cpu_index == 0;
    match ops.as_any().and_then(|a| a.downcast_ref::<X86>()) {
        Some(x86) => x86.model().new_state(is_bsp),
        None => X86CpuState::default(),
    }
}

/// Reads the `cpu` section state of the vCPU behind `shared`.
pub(crate) fn get_cpu(shared: &CpuShared, side: &SideStore) -> Result<CpuMig> {
    let side = Arc::clone(side);
    on_cpu(shared, move |cpu| {
        let mut g = side.lock().unwrap_or_else(PoisonError::into_inner);
        let c = g.get_or_insert_with(|| CpuMig::new(reset_state(cpu)));
        env::save_state(cpu.env, &mut c.s);
        c.tsc_offset = env::ld64(cpu.env, env::TSC_OFFSET);
        c.copy_out();
        c.clone()
    })
}

/// Loads `c` into the vCPU behind `shared` and keeps it in the side store. The `tsc_offset`
/// goes in as it came: the `timer` section set `cpu_get_ticks()` to the source's, so the
/// guest's TSC carries on from where it stopped.
pub(crate) fn put_cpu(shared: &CpuShared, side: &SideStore, c: CpuMig) -> Result<()> {
    let side = Arc::clone(side);
    on_cpu(shared, move |cpu| {
        env::load_state(cpu.env, &c.s);
        env::st64(cpu.env, env::TSC_OFFSET, c.tsc_offset);
        *side.lock().unwrap_or_else(PoisonError::into_inner) = Some(c);
        tlb_flush(cpu);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    fn state() -> CpuMig {
        let mut s = X86CpuState::default();
        s.regs[0] = 0x1122;
        s.rip = 0xfff0;
        s.fptags = [1, 1, 1, 1, 1, 1, 0, 1];
        s.fpstt = 6;
        s.fpus = 0x0041;
        s.xmm_regs[3][1] = 7;
        s.xmm_regs[3][2] = 9;
        s.exception_nr = -1;
        s.poll_control_msr = 1;
        s.msr_ia32_misc_enable = 1;
        CpuMig::new(s)
    }

    #[test]
    fn cpu_section_is_qemu_sized_and_round_trips() {
        let mut c = state();
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_X86_CPU, &mut c).unwrap();
        let bytes = f.into_inner();
        // The fields of vmstate_x86_cpu at version 12 and no subsection.
        let fields = 128 + 8 + 8 + 4 + 2 * 4 + 8 * 10 + 10 * 20 + 4 + 8 + 8 + 4 * 8 + 64 + 4 + 4;
        let fields = fields + 16 * 16 + 6 * 8 + 4 + 8 + 4 + 4 * 8 + 4 * 2 + 4 + 1;
        let fields = fields + 11 * 8 + 8 + 8 * 16 + 4 + 4 + 8 + 4 + 4 + 4;
        let fields = fields + 3 * 8 + 40 * 8 + 3 * 8 + 2 * 8 + 16 * 16;
        assert_eq!(bytes.len(), fields);
        // fpus_vmstate carries the top of stack, and fptag_vmstate has a bit per full register.
        assert_eq!(&bytes[150..154], &[0x30, 0x41, 0x00, 0x40]);

        // In a stream the section footer follows, where the loader looks for a subsection.
        let mut stream = bytes.clone();
        stream.push(0x7e);
        let mut back = CpuMig::new(X86CpuState::default());
        vmstate_load_state(&mut StreamReader::new(&stream), &VMSTATE_X86_CPU, &mut back, 12)
            .unwrap();
        assert_eq!(back.s.regs[0], 0x1122);
        assert_eq!(back.s.fpstt, 6);
        assert_eq!(back.s.fpus, 0x0041);
        assert_eq!(back.s.fptags, [1, 1, 1, 1, 1, 1, 0, 1]);
        assert_eq!(back.s.xmm_regs[3][..3], [0, 7, 9]);
    }

    #[test]
    fn cpu_subsections_follow_the_state() {
        let mut c = state();
        c.s.poll_control_msr = 0;
        c.s.fpip = 0x1234;
        c.s.msr_smi_count = 3;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_X86_CPU, &mut c).unwrap();
        let bytes = f.into_inner();
        let find = |name: &[u8]| bytes.windows(name.len()).any(|w| w == name);
        assert!(find(b"cpu/poll_control_msr"));
        assert!(find(b"cpu/fpop_ip_dp"));
        assert!(find(b"cpu/msr_smi_count"));
        assert!(!find(b"cpu/avx512"));

        let mut back = state();
        vmstate_load_state(&mut StreamReader::new(&bytes), &VMSTATE_X86_CPU, &mut back, 12)
            .unwrap();
        assert_eq!(back.s.poll_control_msr, 0);
        assert_eq!(back.s.fpip, 0x1234);
        assert_eq!(back.s.msr_smi_count, 3);
    }
}
