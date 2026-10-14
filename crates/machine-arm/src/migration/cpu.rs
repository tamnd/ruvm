// SPDX-License-Identifier: GPL-2.0-or-later

//! The `cpu_common` and `cpu` sections of an Arm vCPU, from hw/core/cpu-system.c and
//! target/arm/machine.c. The state itself is gathered and put back by
//! [`ruvm_target_arm::tcg::migrate`].

use std::sync::atomic::Ordering;
use std::sync::{LazyLock, mpsc};

use ruvm_base::{Result, err};
use ruvm_jit::cputlb::tlb_flush;
use ruvm_jit::{Cpu, CpuShared};
use ruvm_target_arm::tcg::migrate::{self, CpuMig};
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

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
            halted: s.halted.load(Ordering::Acquire),
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
        s.halted.store(c.halted, Ordering::Release);
        s.reset_interrupt(u32::MAX);
        if c.interrupt_request != 0 {
            s.cpu_interrupt(c.interrupt_request);
        }
        cpu.core.exception_index = c.exception_index;
        tlb_flush(cpu);
    })
}

/// `vmstate_vfp_fpcr_fpsr`.
static VFP_FPCR_FPSR: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/vfp/fpcr_fpsr")
        .version_id(1)
        .minimum_version_id(1)
        .needed(CpuMig::fpcr_fpsr_needed)
        .fields([
            VmStateField::scalar("fpcr", |c: &mut CpuMig| &mut c.fpcr),
            VmStateField::scalar("fpsr", |c: &mut CpuMig| &mut c.fpsr),
        ])
});

/// `vmstate_vfp`: the low 128 bits of each vector register are `env.vfp.zregs[n].d[0..2]`.
static VFP: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    let zregs = (0..32).map(|n| {
        let name: &'static str = Box::leak(format!("env.vfp.zregs[{n}].d").into_boxed_str());
        VmStateField::array(name, move |c: &mut CpuMig| &mut c.zregs_lo[n])
    });
    VmStateDescription::<CpuMig>::new("cpu/vfp")
        .version_id(3)
        .minimum_version_id(3)
        .fields(zregs)
        .fields([
            VmStateField::scalar("env.vfp.xregs[0]", |c: &mut CpuMig| &mut c.fpsid),
            VmStateField::array("env.vfp.xregs", |c: &mut CpuMig| &mut c.vfp_xregs),
            VmStateField::scalar("fpscr", |c: &mut CpuMig| &mut c.fpscr),
        ])
        .subsection(&VFP_FPCR_FPSR)
});

/// `vmstate_zreg_hi_reg`: QEMU sends `d[2..16]`, `ARM_MAX_VQ - 2` words.
static ZREG_HI: LazyLock<VmStateDescription<[u64; 14]>> = LazyLock::new(|| {
    VmStateDescription::<[u64; 14]>::new("cpu/sve/zreg_hi")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::array("d", |d: &mut [u64; 14]| d))
});

/// `vmstate_preg_reg`.
static PREG: LazyLock<VmStateDescription<[u64; 4]>> = LazyLock::new(|| {
    VmStateDescription::<[u64; 4]>::new("cpu/sve/preg")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::array("p", |p: &mut [u64; 4]| p))
});

/// `vmstate_sve`.
static SVE: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/sve")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.sve)
        .fields([
            VmStateField::struct_array("env.vfp.zregs", &ZREG_HI, |c: &mut CpuMig| &mut c.zregs_hi),
            VmStateField::struct_array("env.vfp.pregs", &PREG, |c: &mut CpuMig| &mut c.pregs),
        ])
});

/// `vmstate_serror`: this port has no pending SError state, so it is only ever loaded.
static SERROR: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/serror")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.serror[0] != 0)
        .fields([
            VmStateField::scalar("env.serror.pending", |c: &mut CpuMig| &mut c.serror[0]),
            VmStateField::scalar("env.serror.has_esr", |c: &mut CpuMig| &mut c.serror[1]),
            VmStateField::scalar("env.serror.esr", |c: &mut CpuMig| &mut c.serror_esr),
        ])
});

/// `vmstate_irq_line_state`.
static IRQ_LINE_STATE: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/irq-line-state")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_| true)
        .field(VmStateField::scalar("env.irq_line_state", |c: &mut CpuMig| &mut c.irq_line_state))
});

/// `vmstate_wfxt_timer`: QEMU's `max` has FEAT_WFxT, and its timer is idle here.
static WFXT_TIMER: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/wfxt-timer")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.max)
        .field(VmStateField::single("wfxt_timer", &Timer, |c: &mut CpuMig| &mut c.wfxt_timer))
});

/// `vmstate_syndrome64`.
static SYNDROME64: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/syndrome64")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.syndrome > u64::from(u32::MAX))
        .field(VmStateField::scalar("env.exception.syndrome", |c: &mut CpuMig| &mut c.syndrome))
});

/// `vmstate_pstate64`.
static PSTATE64: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/pstate64")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.pstate64 > u64::from(u32::MAX))
        .field(VmStateField::scalar("pstate", |c: &mut CpuMig| &mut c.pstate64))
});

/// `vmstate_event`.
static EVENT: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/event")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.event_register)
        .field(VmStateField::scalar("env.event_register", |c: &mut CpuMig| &mut c.event_register))
});

/// `vmstate_fpmr`: QEMU's `max` has FEAT_FPMR.
static FPMR: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu/fpmr")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c| c.max)
        .field(VmStateField::scalar("env.vfp.fpmr", |c: &mut CpuMig| &mut c.fpmr))
});

/// `vmstate_arm_cpu`. The M profile and AArch32 only subsections are left out: this port
/// has neither.
pub(crate) static VMSTATE_ARM_CPU: LazyLock<VmStateDescription<CpuMig>> = LazyLock::new(|| {
    VmStateDescription::<CpuMig>::new("cpu")
        .version_id(22)
        .minimum_version_id(22)
        .pre_load(|c| {
            c.pre_load();
            0
        })
        .fields([
            VmStateField::array("env.regs", |c: &mut CpuMig| &mut c.regs),
            VmStateField::array("env.xregs", |c: &mut CpuMig| &mut c.xregs),
            VmStateField::scalar("env.pc", |c: &mut CpuMig| &mut c.pc),
            VmStateField::scalar("cpsr", |c: &mut CpuMig| &mut c.cpsr),
            VmStateField::scalar("env.spsr", |c: &mut CpuMig| &mut c.spsr),
            VmStateField::array("env.banked_spsr", |c: &mut CpuMig| &mut c.banked_spsr),
            VmStateField::array("env.banked_r13", |c: &mut CpuMig| &mut c.banked_r13),
            VmStateField::array("env.banked_r14", |c: &mut CpuMig| &mut c.banked_r14),
            VmStateField::array("env.usr_regs", |c: &mut CpuMig| &mut c.usr_regs),
            VmStateField::array("env.fiq_regs", |c: &mut CpuMig| &mut c.fiq_regs),
            VmStateField::array("env.elr_el", |c: &mut CpuMig| &mut c.elr_el),
            VmStateField::array("env.sp_el", |c: &mut CpuMig| &mut c.sp_el),
            VmStateField::scalar("cpreg_vmstate_array_len", |c: &mut CpuMig| &mut c.cpreg_len),
            VmStateField::varray_alloc(
                "cpreg_vmstate_indexes",
                |c: &CpuMig| usize::try_from(c.cpreg_len).unwrap_or(0),
                |c: &mut CpuMig| &mut c.cpreg_indexes,
            ),
            VmStateField::varray_alloc(
                "cpreg_vmstate_values",
                |c: &CpuMig| usize::try_from(c.cpreg_len).unwrap_or(0),
                |c: &mut CpuMig| &mut c.cpreg_values,
            ),
            VmStateField::scalar("env.exclusive_addr", |c: &mut CpuMig| &mut c.exclusive_addr),
            VmStateField::scalar("env.exclusive_val", |c: &mut CpuMig| &mut c.exclusive_val),
            VmStateField::scalar("env.exclusive_high", |c: &mut CpuMig| &mut c.exclusive_high),
            VmStateField::unused(8),
            VmStateField::scalar("env.exception.syndrome", |c: &mut CpuMig| &mut c.syndrome32),
            VmStateField::scalar("env.exception.fsr", |c: &mut CpuMig| &mut c.fsr),
            VmStateField::scalar("env.exception.vaddress", |c: &mut CpuMig| &mut c.vaddress),
            VmStateField::single("gt_timer[GTIMER_PHYS]", &Timer, |c: &mut CpuMig| {
                &mut c.gt_timer[0]
            }),
            VmStateField::single("gt_timer[GTIMER_VIRT]", &Timer, |c: &mut CpuMig| {
                &mut c.gt_timer[1]
            }),
            VmStateField::scalar("power_state", |c: &mut CpuMig| &mut c.powered_off),
        ])
        .subsection(&VFP)
        .subsection(&SVE)
        .subsection(&SERROR)
        .subsection(&IRQ_LINE_STATE)
        .subsection(&WFXT_TIMER)
        .subsection(&SYNDROME64)
        .subsection(&PSTATE64)
        .subsection(&EVENT)
        .subsection(&FPMR)
});

/// Reads the vCPU behind `shared` for the `cpu` section.
pub(crate) fn get_cpu(shared: &CpuShared) -> Result<CpuMig> {
    on_cpu(shared, migrate::save)
}

/// Loads `m` into the vCPU behind `shared`.
pub(crate) fn put_cpu(shared: &CpuShared, m: CpuMig) -> Result<()> {
    on_cpu(shared, move |cpu| migrate::load(cpu, &m))?.map_err(|index| {
        err!(
            "cpu_post_load: system register 0x{:x} in the incoming stream but unknown on the \
             destination: fail migration",
            index
        )
    })
}
