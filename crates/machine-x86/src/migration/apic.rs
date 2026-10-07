// SPDX-License-Identifier: GPL-2.0-or-later

//! The `apic` section of a local APIC, `vmstate_apic_common` from hw/intc/apic_common.c.

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_intc::apic::{Apic, ApicVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{MigPriority, VmStateDescription, VmStateField};

type Vmsd = VmStateDescription<ApicVmState>;

/// `vmstate_apic_common_sipi`: only sent for an APIC that waits for a SIPI.
static APIC_SIPI: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("apic_sipi")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &ApicVmState| s.wait_for_sipi != 0)
        .fields([
            VmStateField::scalar("sipi_vector", |s: &mut ApicVmState| &mut s.sipi_vector),
            VmStateField::scalar("wait_for_sipi", |s: &mut ApicVmState| &mut s.wait_for_sipi),
        ])
});

/// `vmstate_apic_common`.
pub(crate) static VMSTATE_APIC: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("apic")
        .version_id(3)
        .minimum_version_id(3)
        .priority(MigPriority::Apic)
        // apic_pre_load(): the value assumed when the apic_sipi subsection is absent.
        .pre_load(|s: &mut ApicVmState| {
            s.wait_for_sipi = 0;
            0
        })
        .fields([
            VmStateField::scalar("apicbase", |s: &mut ApicVmState| &mut s.apicbase),
            VmStateField::scalar("id", |s: &mut ApicVmState| &mut s.id),
            VmStateField::scalar("arb_id", |s: &mut ApicVmState| &mut s.arb_id),
            VmStateField::scalar("tpr", |s: &mut ApicVmState| &mut s.tpr),
            VmStateField::scalar("spurious_vec", |s: &mut ApicVmState| &mut s.spurious_vec),
            VmStateField::scalar("log_dest", |s: &mut ApicVmState| &mut s.log_dest),
            VmStateField::scalar("dest_mode", |s: &mut ApicVmState| &mut s.dest_mode),
            VmStateField::array("isr", |s: &mut ApicVmState| &mut s.isr),
            VmStateField::array("tmr", |s: &mut ApicVmState| &mut s.tmr),
            VmStateField::array("irr", |s: &mut ApicVmState| &mut s.irr),
            VmStateField::array("lvt", |s: &mut ApicVmState| &mut s.lvt),
            VmStateField::scalar("esr", |s: &mut ApicVmState| &mut s.esr),
            VmStateField::array("icr", |s: &mut ApicVmState| &mut s.icr),
            VmStateField::scalar("divide_conf", |s: &mut ApicVmState| &mut s.divide_conf),
            VmStateField::scalar("count_shift", |s: &mut ApicVmState| &mut s.count_shift),
            VmStateField::scalar("initial_count", |s: &mut ApicVmState| &mut s.initial_count),
            VmStateField::scalar("initial_count_load_time", |s: &mut ApicVmState| {
                &mut s.initial_count_load_time
            }),
            VmStateField::scalar("next_time", |s: &mut ApicVmState| &mut s.next_time),
            // The open-coded timer: an INT64, not a VMSTATE_TIMER_PTR.
            VmStateField::scalar("timer_expiry", |s: &mut ApicVmState| &mut s.timer_expiry),
        ])
        .subsection(&APIC_SIPI)
});

/// Registers the `apic` section of `apic`, with its initial APIC ID as the instance id the way
/// `apic_common_realize()` does.
pub(crate) fn register(savevm: &mut SaveVm, apic: &Arc<Apic>) {
    let (get, put) = (Arc::clone(apic), Arc::clone(apic));
    savevm.register_vmsd(
        "",
        Some(apic.initial_apic_id()),
        &VMSTATE_APIC,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}
