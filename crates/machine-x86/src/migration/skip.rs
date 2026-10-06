// SPDX-License-Identifier: GPL-2.0-or-later

//! The q35 sections ruvm accepts from QEMU without loading them yet: the stream layout of each
//! device, so the stream is parsed and checked, with the state dropped. They are never sent, and
//! a QEMU destination keeps its reset state for them.

use std::sync::LazyLock;

use ruvm_migration::SaveVm;
use ruvm_vmstate::{MigPriority, VmStateDescription, VmStateField};

/// What a few layouts need to know about the machine to size their arrays.
#[derive(Debug, Default)]
pub(crate) struct Skip {
    max_cpus: usize,
    nirq: i32,
    irq_count: Vec<i32>,
    devs: Vec<u8>,
}

type Vmsd = VmStateDescription<Skip>;

/// A description of `size` bytes at `version`.
fn blob(name: &'static str, version: i32, size: usize) -> Vmsd {
    VmStateDescription::new(name)
        .version_id(version)
        .minimum_version_id(version)
        .field(VmStateField::unused(size))
}

static APIC_SIPI: LazyLock<Vmsd> = LazyLock::new(|| blob("apic_sipi", 1, 8));
static APIC: LazyLock<Vmsd> =
    LazyLock::new(|| blob("apic", 3, 181).priority(MigPriority::Apic).subsection(&APIC_SIPI));

static ICOUNT_WARP: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/warp_timer", 1, 16));
static ICOUNT_TIMERS: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/timers", 1, 16));
static ICOUNT_SHIFT: LazyLock<Vmsd> = LazyLock::new(|| blob("timer/icount/shift", 2, 10));
static ICOUNT: LazyLock<Vmsd> = LazyLock::new(|| {
    blob("timer/icount", 1, 16)
        .subsection(&ICOUNT_WARP)
        .subsection(&ICOUNT_TIMERS)
        .subsection(&ICOUNT_SHIFT)
});
static TIMER: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("timer")
        .version_id(2)
        .minimum_version_id(1)
        .fields([VmStateField::unused(16), VmStateField::unused(8).version(2)])
        .subsection(&ICOUNT)
});

static KVM_TPR_OPT: LazyLock<Vmsd> = LazyLock::new(|| blob("kvm-tpr-opt", 1, 144));

static FW_CFG_DMA: LazyLock<Vmsd> = LazyLock::new(|| blob("fw_cfg/dma", 0, 8));
static FW_CFG_ACPI_MR: LazyLock<Vmsd> = LazyLock::new(|| blob("fw_cfg/acpi_mr", 1, 24));
static FW_CFG: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("fw_cfg")
        .version_id(2)
        .minimum_version_id(1)
        .fields([
            VmStateField::unused(2),
            VmStateField::unused(2).test(|_, v| v == 1),
            VmStateField::unused(4).version(2),
        ])
        .subsection(&FW_CFG_DMA)
        .subsection(&FW_CFG_ACPI_MR)
});

static MCH: LazyLock<Vmsd> = LazyLock::new(|| blob("mch", 1, 277));
static PCI_HOST: LazyLock<Vmsd> = LazyLock::new(|| blob("PCIHost", 1, 4));
static PCIBUS: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("PCIBUS").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("nirq", |s: &mut Skip| &mut s.nirq),
        VmStateField::varray_alloc(
            "irq_count",
            |s: &Skip| usize::try_from(s.nirq).unwrap_or(0),
            |s: &mut Skip| &mut s.irq_count,
        ),
    ])
});
static DMA: LazyLock<Vmsd> = LazyLock::new(|| blob("dma", 1, 75));

static RTC_REINJECT: LazyLock<Vmsd> =
    LazyLock::new(|| blob("mc146818rtc/irq_reinject_on_ack_count", 1, 2));
static RTC: LazyLock<Vmsd> =
    LazyLock::new(|| blob("mc146818rtc", 3, 245).subsection(&RTC_REINJECT));

static PM_MEMHP: LazyLock<Vmsd> = LazyLock::new(|| {
    // The selector, then no slots: ruvm and the QEMU it talks to have no memory hotplug slots.
    blob("ich9_pm/memhp", 1, 4)
});
static PM_TCO: LazyLock<Vmsd> = LazyLock::new(|| blob("ich9_pm/tco", 1, 35));
static PM_CPUHP: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("ich9_pm/cpuhp").version_id(1).minimum_version_id(1).fields([
        VmStateField::unused(5),
        VmStateField::vbuffer_alloc("devs", |s: &Skip| 10 * s.max_cpus, |s: &mut Skip| &mut s.devs),
    ])
});
static PM_PCIHP: LazyLock<Vmsd> = LazyLock::new(|| blob("ich9_pm/pcihp", 1, 2056));
static ICH9_PM: LazyLock<Vmsd> = LazyLock::new(|| {
    blob("ich9_pm", 1, 62)
        .subsection(&PM_MEMHP)
        .subsection(&PM_TCO)
        .subsection(&PM_CPUHP)
        .subsection(&PM_PCIHP)
});
static LPC_RST_CNT: LazyLock<Vmsd> = LazyLock::new(|| blob("ICH9LPC/rst_cnt", 1, 1));
static LPC_SMI_FEAT: LazyLock<Vmsd> = LazyLock::new(|| blob("ICH9LPC/smi_feat", 1, 17));
static ICH9_LPC: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("ICH9LPC")
        .version_id(1)
        .minimum_version_id(1)
        .fields([
            // The PCI device and the APM registers.
            VmStateField::unused(278),
            VmStateField::structure("pm", &ICH9_PM, |s: &mut Skip| s),
            // chip_config and sci_level.
            VmStateField::unused(16388),
        ])
        .subsection(&LPC_RST_CNT)
        .subsection(&LPC_SMI_FEAT)
});

static PIC_LTIM: LazyLock<Vmsd> = LazyLock::new(|| blob("i8259/ltim", 1, 1));
static PIC: LazyLock<Vmsd> = LazyLock::new(|| blob("i8259", 1, 16).subsection(&PIC_LTIM));

static IOAPIC: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("ioapic").version_id(3).minimum_version_id(1).fields([
        VmStateField::unused(2),
        VmStateField::unused(8).version(2),
        VmStateField::unused(4).version(2),
        VmStateField::unused(192),
    ])
});

static ACPI_BUILD: LazyLock<Vmsd> = LazyLock::new(|| blob("acpi_build", 1, 1));

/// Registers the `apic` section of the CPU with APIC ID `apic_id`.
pub(crate) fn register_apic(savevm: &mut SaveVm, apic_id: u32) {
    savevm.register_discard_with("", Some(apic_id), &APIC, Skip::default);
}

/// Registers the `kvm-tpr-opt` section of kvmvapic, which comes with the first APIC.
pub(crate) fn register_vapic(savevm: &mut SaveVm) {
    savevm.register_discard_with("", Some(0), &KVM_TPR_OPT, Skip::default);
}

/// Registers the `timer` section, the first one QEMU registers.
pub(crate) fn register_timer(savevm: &mut SaveVm) {
    savevm.register_discard_with("", Some(0), &TIMER, Skip::default);
}

/// Registers the board's sections, which come after the CPUs.
pub(crate) fn register_board(savevm: &mut SaveVm, max_cpus: usize) {
    let fresh = Skip::default;
    savevm.register_discard_with("", Some(0), &FW_CFG, fresh);
    savevm.register_discard_with("0000:00:00.0/", Some(0), &MCH, fresh);
    savevm.register_discard_with("", Some(0), &PCI_HOST, fresh);
    savevm.register_discard_with("", Some(0), &PCIBUS, fresh);
    savevm.register_discard_with("", Some(0), &DMA, fresh);
    savevm.register_discard_with("", Some(1), &DMA, fresh);
    savevm.register_discard_with("", Some(0), &RTC, fresh);
    savevm.register_discard_with("0000:00:1f.0/", Some(0), &ICH9_LPC, move || Skip {
        max_cpus,
        ..Skip::default()
    });
    savevm.register_discard_with("", Some(0), &PIC, fresh);
    savevm.register_discard_with("", Some(1), &PIC, fresh);
    savevm.register_discard_with("", Some(0), &IOAPIC, fresh);
    savevm.register_discard_with("", Some(0), &ACPI_BUILD, fresh);
}
