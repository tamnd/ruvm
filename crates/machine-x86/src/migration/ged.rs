// SPDX-License-Identifier: GPL-2.0-or-later

//! The `acpi-ged` section of microvm, from hw/acpi/generic_event_device.c. Its layout is
//! [`VMSTATE_ACPI_GED`].

use std::sync::Arc;

use ruvm_hw_acpi::AcpiGed;
use ruvm_hw_acpi::ich9::MemHotplugVmState;
use ruvm_hw_acpi::vmstate::{AcpiGedVmState, VMSTATE_ACPI_GED};
use ruvm_migration::SaveVm;

/// Registers the `acpi-ged` section of `ged`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, ged: &Arc<AcpiGed>) {
    let (get, put) = (Arc::clone(ged), Arc::clone(ged));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_ACPI_GED,
        move || Ok(AcpiGedVmState { sel: get.sel(), memhp: MemHotplugVmState::default() }),
        move |s| {
            put.set_sel(s.sel);
            Ok(())
        },
    );
}
