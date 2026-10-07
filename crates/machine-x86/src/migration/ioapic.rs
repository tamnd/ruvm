// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ioapic` section, `vmstate_ioapic_common` from hw/intc/ioapic_common.c.

use std::sync::{Arc, LazyLock};

use ruvm_hw_intc::ioapic::{IoApic, IoApicVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// `vmstate_ioapic_common`.
pub(crate) static VMSTATE_IOAPIC: LazyLock<VmStateDescription<IoApicVmState>> =
    LazyLock::new(|| {
        VmStateDescription::new("ioapic").version_id(3).minimum_version_id(1).fields([
            VmStateField::scalar("id", |s: &mut IoApicVmState| &mut s.id),
            VmStateField::scalar("ioregsel", |s: &mut IoApicVmState| &mut s.ioregsel),
            // To account for qemu-kvm's v2 format.
            VmStateField::unused(8).version(2),
            VmStateField::scalar("irr", |s: &mut IoApicVmState| &mut s.irr).version(2),
            VmStateField::array("ioredtbl", |s: &mut IoApicVmState| &mut s.ioredtbl),
        ])
    });

/// Registers the `ioapic` section of `ioapic` as instance `instance_id`, 0 for the first one.
pub(crate) fn register(savevm: &mut SaveVm, instance_id: u32, ioapic: &Arc<IoApic>) {
    let (get, put) = (Arc::clone(ioapic), Arc::clone(ioapic));
    savevm.register_vmsd(
        "",
        Some(instance_id),
        &VMSTATE_IOAPIC,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}
