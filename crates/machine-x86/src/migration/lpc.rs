// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ICH9LPC` section, from hw/isa/lpc_ich9.c, with the `APM State` (hw/isa/apm.c) and
//! `ich9_pm` (hw/acpi/ich9.c) structures it nests, and the `acpi_build` section from
//! hw/i386/acpi-build.c.
//!
//! The PM timer's overflow timer and the TCO timer are on the virtual clock, whose value
//! travels in the `timer` section.

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_acpi::Ich9PmVmState;
use ruvm_hw_acpi::ich9::{
    ACPI_PCIHP_MAX_HOTPLUG_BUS, AcpiPcihpPciStatusVmState, CpuHotplugDevVmState, CpuHotplugVmState,
    MemHotplugDevVmState, MemHotplugVmState, TcoVmState,
};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use super::pci::VMSTATE_PCI_DEVICE;
use crate::ich9_lpc::{ApmVmState, ICH9_CC_SIZE, Ich9Lpc, Ich9LpcVmState};

/// `vmstate_apm`.
static VMSTATE_APM: LazyLock<VmStateDescription<ApmVmState>> = LazyLock::new(|| {
    type S = ApmVmState;
    VmStateDescription::new("APM State").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("apmc", |s: &mut S| &mut s.apmc),
        VmStateField::scalar("apms", |s: &mut S| &mut s.apms),
    ])
});

/// `vmstate_memhp_sts`.
static VMSTATE_MEMHP_STS: LazyLock<VmStateDescription<MemHotplugDevVmState>> =
    LazyLock::new(|| {
        type S = MemHotplugDevVmState;
        VmStateDescription::new("memory hotplug device state")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("is_enabled", |s: &mut S| &mut s.is_enabled),
                VmStateField::scalar("is_inserting", |s: &mut S| &mut s.is_inserting),
                VmStateField::scalar("ost_event", |s: &mut S| &mut s.ost_event),
                VmStateField::scalar("ost_status", |s: &mut S| &mut s.ost_status),
            ])
    });

/// `vmstate_memory_hotplug`. `devs` has `dev_count` entries, which is not in the stream: the
/// local count is the one used.
pub(crate) static VMSTATE_MEMORY_HOTPLUG: LazyLock<VmStateDescription<MemHotplugVmState>> =
    LazyLock::new(|| {
        type S = MemHotplugVmState;
        VmStateDescription::new("memory hotplug state").version_id(1).minimum_version_id(1).fields(
            [
                VmStateField::scalar("selector", |s: &mut S| &mut s.selector),
                VmStateField::struct_varray(
                    "devs",
                    |s: &S| s.devs.len(),
                    &VMSTATE_MEMHP_STS,
                    |s: &mut S| &mut s.devs,
                ),
            ],
        )
    });

/// `vmstate_tco_io_sts`.
static VMSTATE_TCO_IO_STS: LazyLock<VmStateDescription<TcoVmState>> = LazyLock::new(|| {
    type S = TcoVmState;
    VmStateDescription::new("tco io device status").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("tco.rld", |s: &mut S| &mut s.rld),
        VmStateField::scalar("tco.din", |s: &mut S| &mut s.din),
        VmStateField::scalar("tco.dout", |s: &mut S| &mut s.dout),
        VmStateField::scalar("tco.sts1", |s: &mut S| &mut s.sts1),
        VmStateField::scalar("tco.sts2", |s: &mut S| &mut s.sts2),
        VmStateField::scalar("tco.cnt1", |s: &mut S| &mut s.cnt1),
        VmStateField::scalar("tco.cnt2", |s: &mut S| &mut s.cnt2),
        VmStateField::scalar("tco.msg1", |s: &mut S| &mut s.msg1),
        VmStateField::scalar("tco.msg2", |s: &mut S| &mut s.msg2),
        VmStateField::scalar("tco.wdcnt", |s: &mut S| &mut s.wdcnt),
        VmStateField::scalar("tco.tmr", |s: &mut S| &mut s.tmr),
        VmStateField::scalar("sw_irq_gen", |s: &mut S| &mut s.sw_irq_gen),
        VmStateField::single("tco_timer", &Timer, |s: &mut S| &mut s.tco_timer),
        VmStateField::scalar("expire_time", |s: &mut S| &mut s.expire_time),
        VmStateField::scalar("timeouts_no", |s: &mut S| &mut s.timeouts_no),
    ])
});

/// `vmstate_cpuhp_sts`.
static VMSTATE_CPUHP_STS: LazyLock<VmStateDescription<CpuHotplugDevVmState>> =
    LazyLock::new(|| {
        type S = CpuHotplugDevVmState;
        VmStateDescription::new("CPU hotplug device state")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::scalar("is_inserting", |s: &mut S| &mut s.is_inserting),
                VmStateField::scalar("is_removing", |s: &mut S| &mut s.is_removing),
                VmStateField::scalar("ost_event", |s: &mut S| &mut s.ost_event),
                VmStateField::scalar("ost_status", |s: &mut S| &mut s.ost_status),
            ])
    });

/// `vmstate_cpu_hotplug`. `devs` has one entry per possible CPU, which is not in the stream:
/// the local count, see [`Ich9Pm::set_cpu_hotplug_slots`](ruvm_hw_acpi::Ich9Pm), is the one
/// used.
static VMSTATE_CPU_HOTPLUG: LazyLock<VmStateDescription<CpuHotplugVmState>> = LazyLock::new(|| {
    type S = CpuHotplugVmState;
    VmStateDescription::new("CPU hotplug state").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("selector", |s: &mut S| &mut s.selector),
        VmStateField::scalar("command", |s: &mut S| &mut s.command),
        VmStateField::struct_varray(
            "devs",
            |s: &S| s.devs.len(),
            &VMSTATE_CPUHP_STS,
            |s: &mut S| &mut s.devs,
        ),
    ])
});

/// `vmstate_acpi_pcihp_pci_status`.
static VMSTATE_ACPI_PCIHP_PCI_STATUS: LazyLock<VmStateDescription<AcpiPcihpPciStatusVmState>> =
    LazyLock::new(|| {
        type S = AcpiPcihpPciStatusVmState;
        VmStateDescription::new("acpi_pcihp_pci_status").version_id(1).minimum_version_id(1).fields(
            [
                VmStateField::scalar("up", |s: &mut S| &mut s.up),
                VmStateField::scalar("down", |s: &mut S| &mut s.down),
            ],
        )
    });

/// `vmstate_memhp_state`, always sent: it has no `needed`.
static VMSTATE_ICH9_PM_MEMHP: LazyLock<VmStateDescription<Ich9PmVmState>> = LazyLock::new(|| {
    type S = Ich9PmVmState;
    VmStateDescription::new("ich9_pm/memhp").version_id(1).minimum_version_id(1).field(
        VmStateField::structure("acpi_memory_hotplug", &VMSTATE_MEMORY_HOTPLUG, |s: &mut S| {
            &mut s.acpi_memory_hotplug
        })
        .version(1),
    )
});

/// `vmstate_tco_io_state`. `vmstate_test_use_tco()` is `enable_tco`, which is always set on
/// q35.
static VMSTATE_ICH9_PM_TCO: LazyLock<VmStateDescription<Ich9PmVmState>> = LazyLock::new(|| {
    type S = Ich9PmVmState;
    VmStateDescription::new("ich9_pm/tco")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| true)
        .field(
            VmStateField::structure("tco_regs", &VMSTATE_TCO_IO_STS, |s: &mut S| &mut s.tco_regs)
                .version(1),
        )
});

/// `vmstate_cpuhp_state`. `cpuhp_needed()` is `has_hotpluggable_cpus`, which q35 has.
static VMSTATE_ICH9_PM_CPUHP: LazyLock<VmStateDescription<Ich9PmVmState>> = LazyLock::new(|| {
    type S = Ich9PmVmState;
    VmStateDescription::new("ich9_pm/cpuhp")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| true)
        .field(
            VmStateField::structure("cpuhp_state", &VMSTATE_CPU_HOTPLUG, |s: &mut S| {
                &mut s.cpuhp_state
            })
            .version(1),
        )
});

/// `vmstate_pcihp_state`. `vmstate_test_use_pcihp()` is `use_acpi_hotplug_bridge`, the
/// `acpi-pci-hotplug-with-bridge-support` property, which is on for q35.
static VMSTATE_ICH9_PM_PCIHP: LazyLock<VmStateDescription<Ich9PmVmState>> = LazyLock::new(|| {
    type S = Ich9PmVmState;
    VmStateDescription::new("ich9_pm/pcihp")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|_: &S| true)
        .fields([
            VmStateField::scalar("acpi_pci_hotplug.hotplug_select", |s: &mut S| {
                &mut s.acpi_pci_hotplug.hotplug_select
            }),
            VmStateField::struct_varray(
                "acpi_pci_hotplug.acpi_pcihp_pci_status",
                |_: &S| ACPI_PCIHP_MAX_HOTPLUG_BUS,
                &VMSTATE_ACPI_PCIHP_PCI_STATUS,
                |s: &mut S| &mut s.acpi_pci_hotplug.acpi_pcihp_pci_status,
            )
            .version(1),
            VmStateField::scalar("acpi_pci_hotplug.acpi_index", |s: &mut S| {
                &mut s.acpi_pci_hotplug.acpi_index
            }),
        ])
});

/// `vmstate_ich9_pm`. `ich9_pm_post_load()` is in
/// [`Ich9Pm::vmstate_load`](ruvm_hw_acpi::Ich9Pm::vmstate_load).
static VMSTATE_ICH9_PM: LazyLock<VmStateDescription<Ich9PmVmState>> = LazyLock::new(|| {
    type S = Ich9PmVmState;
    VmStateDescription::new("ich9_pm")
        .version_id(1)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("acpi_regs.pm1.evt.sts", |s: &mut S| &mut s.pm1_evt_sts),
            VmStateField::scalar("acpi_regs.pm1.evt.en", |s: &mut S| &mut s.pm1_evt_en),
            VmStateField::scalar("acpi_regs.pm1.cnt.cnt", |s: &mut S| &mut s.pm1_cnt_cnt),
            VmStateField::single("acpi_regs.tmr.timer", &Timer, |s: &mut S| &mut s.tmr_timer),
            VmStateField::scalar("acpi_regs.tmr.overflow_time", |s: &mut S| {
                &mut s.tmr_overflow_time
            }),
            VmStateField::array("acpi_regs.gpe.sts", |s: &mut S| &mut s.gpe_sts),
            VmStateField::array("acpi_regs.gpe.en", |s: &mut S| &mut s.gpe_en),
            VmStateField::scalar("smi_en", |s: &mut S| &mut s.smi_en),
            VmStateField::scalar("smi_sts", |s: &mut S| &mut s.smi_sts),
        ])
        .subsection(&VMSTATE_ICH9_PM_MEMHP)
        .subsection(&VMSTATE_ICH9_PM_TCO)
        .subsection(&VMSTATE_ICH9_PM_CPUHP)
        .subsection(&VMSTATE_ICH9_PM_PCIHP)
});

/// `vmstate_ich9_rst_cnt`.
static VMSTATE_ICH9_RST_CNT: LazyLock<VmStateDescription<Ich9LpcVmState>> = LazyLock::new(|| {
    type S = Ich9LpcVmState;
    VmStateDescription::new("ICH9LPC/rst_cnt")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.rst_cnt != 0)
        .field(VmStateField::scalar("rst_cnt", |s: &mut S| &mut s.rst_cnt))
});

/// `vmstate_ich9_smi_feat`.
static VMSTATE_ICH9_SMI_FEAT: LazyLock<VmStateDescription<Ich9LpcVmState>> = LazyLock::new(|| {
    type S = Ich9LpcVmState;
    VmStateDescription::new("ICH9LPC/smi_feat")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &S| s.smi_guest_features_le != [0; 8] || s.smi_features_ok != 0)
        .fields([
            VmStateField::array("smi_guest_features_le", |s: &mut S| &mut s.smi_guest_features_le),
            VmStateField::scalar("smi_features_ok", |s: &mut S| &mut s.smi_features_ok),
            VmStateField::scalar("smi_negotiated_features", |s: &mut S| {
                &mut s.smi_negotiated_features
            }),
        ])
});

/// `vmstate_ich9_lpc`. `ich9_lpc_post_load()` is in [`Ich9Lpc::vmstate_load`].
pub(crate) static VMSTATE_ICH9_LPC: LazyLock<VmStateDescription<Ich9LpcVmState>> =
    LazyLock::new(|| {
        type S = Ich9LpcVmState;
        VmStateDescription::new("ICH9LPC")
            .version_id(1)
            .minimum_version_id(1)
            .fields([
                VmStateField::structure("d", &VMSTATE_PCI_DEVICE, |s: &mut S| &mut s.d),
                VmStateField::structure("apm", &VMSTATE_APM, |s: &mut S| &mut s.apm),
                VmStateField::structure("pm", &VMSTATE_ICH9_PM, |s: &mut S| &mut s.pm),
                VmStateField::varray(
                    "chip_config",
                    |_: &S| ICH9_CC_SIZE,
                    |s: &mut S| &mut s.chip_config,
                ),
                VmStateField::scalar("sci_level", |s: &mut S| &mut s.sci_level),
            ])
            .subsection(&VMSTATE_ICH9_RST_CNT)
            .subsection(&VMSTATE_ICH9_SMI_FEAT)
    });

/// Registers the `0000:00:1f.0/ICH9LPC` section of `lpc`, instance 0. `possible_cpus` sizes the
/// `ich9_pm/cpuhp` device array, which QEMU sizes from `possible_cpus` too.
pub(crate) fn register(savevm: &mut SaveVm, lpc: &Arc<Ich9Lpc>, possible_cpus: usize) {
    lpc.pm().set_cpu_hotplug_slots(possible_cpus);
    let (get, put) = (Arc::clone(lpc), Arc::clone(lpc));
    savevm.register_vmsd(
        "0000:00:1f.0/",
        Some(0),
        &VMSTATE_ICH9_LPC,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

/// `AcpiBuildState`'s migrated part: whether the guest already read the ACPI tables, after
/// which QEMU no longer rebuilds them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct AcpiBuildVmState {
    pub(crate) patched: u8,
}

/// `vmstate_acpi_build`.
pub(crate) static VMSTATE_ACPI_BUILD: LazyLock<VmStateDescription<AcpiBuildVmState>> =
    LazyLock::new(|| {
        type S = AcpiBuildVmState;
        VmStateDescription::new("acpi_build")
            .version_id(1)
            .minimum_version_id(1)
            .field(VmStateField::scalar("patched", |s: &mut S| &mut s.patched))
    });

/// Registers the `acpi_build` section, instance 0 with no path prefix as in QEMU. `get` and
/// `set` read and write the machine's "tables were patched" flag.
pub(crate) fn register_acpi_build(
    savevm: &mut SaveVm,
    mut get: impl FnMut() -> bool + Send + 'static,
    mut set: impl FnMut(bool) + Send + 'static,
) {
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_ACPI_BUILD,
        move || Ok(AcpiBuildVmState { patched: u8::from(get()) }),
        move |s| {
            set(s.patched != 0);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_hw_acpi::ich9::PciHotplugVmState;
    use ruvm_hw_pci::PciDeviceVmState;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    /// What q35 with one possible CPU and no memory slots has.
    fn state() -> Ich9LpcVmState {
        let mut config = vec![0; 256];
        config[0] = 0x86;
        let pm = Ich9PmVmState {
            pm1_cnt_cnt: 1,
            tmr_timer: 1234,
            gpe_sts: [3; 16],
            smi_en: 0x20,
            tco_regs: TcoVmState::default(),
            cpuhp_state: CpuHotplugVmState {
                devs: vec![CpuHotplugDevVmState::default()],
                ..Default::default()
            },
            acpi_pci_hotplug: PciHotplugVmState::default(),
            ..Default::default()
        };
        let mut chip_config = vec![0; ICH9_CC_SIZE];
        chip_config[0x3410] = 0x20;
        Ich9LpcVmState {
            d: PciDeviceVmState { version_id: 2, config, irq_state: [0; 4] },
            apm: ApmVmState { apmc: 0xb2, apms: 0 },
            pm,
            chip_config,
            sci_level: 1,
            ..Default::default()
        }
    }

    fn save(s: &mut Ich9LpcVmState) -> Vec<u8> {
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ICH9_LPC, s).unwrap();
        f.into_inner()
    }

    /// A subsection's header: the marker, the name and the version.
    fn sub(name: &str) -> usize {
        2 + name.len() + 4
    }

    #[test]
    fn ich9_lpc_layout_matches_qemu() {
        let mut s = state();
        let b = save(&mut s);
        // The sizes in QEMU's vmdesc for q35: d 276, apm 2, pm 2246 with its four subsections
        // and their headers, chip_config 16384 and sci_level 4.
        let pm = 2 * 3 + 8 + 8 + 16 * 2 + 4 * 2;
        let memhp = 4;
        let tco = 35;
        let cpuhp = 4 + 1 + 10;
        let pcihp = 4 + 256 * 8 + 4;
        let pm_len = pm
            + sub("ich9_pm/memhp")
            + memhp
            + sub("ich9_pm/tco")
            + tco
            + sub("ich9_pm/cpuhp")
            + cpuhp
            + sub("ich9_pm/pcihp")
            + pcihp;
        assert_eq!(pm_len, 2246);
        assert_eq!(b.len(), 276 + 2 + pm_len + ICH9_CC_SIZE + 4);
        assert_eq!(&b[278 + 6..278 + 14], &1234i64.to_be_bytes());
        assert_eq!(b[278 + pm_len + 0x3410], 0x20);
        // The TCO timer is not armed: -1.
        let tco_at = 278 + pm + sub("ich9_pm/memhp") + memhp + sub("ich9_pm/tco");
        assert_eq!(&b[tco_at + 18..tco_at + 26], &(-1i64).to_be_bytes());

        let mut back = state();
        back.pm.tmr_timer = -1;
        back.chip_config.fill(0);
        back.apm.apmc = 0;
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ICH9_LPC, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn ich9_lpc_subsections() {
        let mut s = state();
        let base = save(&mut s).len();

        s.rst_cnt = 4;
        let b = save(&mut s);
        assert_eq!(b.len(), base + sub("ICH9LPC/rst_cnt") + 1);
        assert_eq!(b[b.len() - 1], 4);

        s.smi_features_ok = 1;
        s.smi_negotiated_features = 1 << 2;
        s.smi_guest_features_le = [4, 0, 0, 0, 0, 0, 0, 0];
        let b = save(&mut s);
        assert_eq!(b.len(), base + sub("ICH9LPC/rst_cnt") + 1 + sub("ICH9LPC/smi_feat") + 17);
        let mut back = state();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ICH9_LPC, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn acpi_build_layout_matches_qemu() {
        let mut s = AcpiBuildVmState { patched: 1 };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ACPI_BUILD, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b, [1u8]);
        let mut back = AcpiBuildVmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ACPI_BUILD, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }
}
