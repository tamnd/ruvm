// SPDX-License-Identifier: GPL-2.0-or-later

//! The `acpi-ged` section of microvm, from hw/acpi/generic_event_device.c.
//!
//! Of the subsections only `acpi-ged/memhp` is sent: it has no `needed`. microvm has no
//! hotpluggable CPUs, no GHES and no PCI hotplug, so the others never are. Its memory hotplug
//! state has no slots and nothing maps its registers without memory hotplug, so the selector
//! goes out as 0 and is dropped on load.

use std::sync::{Arc, LazyLock};

use ruvm_hw_acpi::AcpiGed;
use ruvm_hw_acpi::ich9::MemHotplugVmState;
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use super::lpc::VMSTATE_MEMORY_HOTPLUG;

/// `AcpiGedState`, the parts its section carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct AcpiGedVmState {
    /// `ged_state.sel`.
    pub(crate) sel: u32,
    /// `memhp_state`.
    pub(crate) memhp: MemHotplugVmState,
}

type S = AcpiGedVmState;

/// `vmstate_ged_state`.
static VMSTATE_GED_STATE: LazyLock<VmStateDescription<u32>> = LazyLock::new(|| {
    VmStateDescription::new("acpi-ged-state")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::scalar("sel", |s: &mut u32| s))
});

/// `vmstate_memhp_state`.
static VMSTATE_GED_MEMHP: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("acpi-ged/memhp").version_id(1).minimum_version_id(1).field(
        VmStateField::structure("memhp_state", &VMSTATE_MEMORY_HOTPLUG, |s: &mut S| &mut s.memhp)
            .version(1),
    )
});

/// `vmstate_acpi_ged`.
pub(crate) static VMSTATE_ACPI_GED: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("acpi-ged")
        .version_id(1)
        .minimum_version_id(1)
        .field(
            VmStateField::structure("ged_state", &VMSTATE_GED_STATE, |s: &mut S| &mut s.sel)
                .version(1),
        )
        .subsection(&VMSTATE_GED_MEMHP)
});

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

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn ged_layout_matches_qemu() {
        let mut s = S { sel: 0x8000_0002, memhp: MemHotplugVmState::default() };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ACPI_GED, &mut s).unwrap();
        let b = f.into_inner();
        // sel, then the memhp subsection: its header and the selector, with no slots.
        let memhp = 2 + "acpi-ged/memhp".len() + 4 + 4;
        assert_eq!(b.len(), 4 + memhp);
        assert_eq!(b[..4], [0x80, 0, 0, 2]);
        assert_eq!(&b[6..6 + "acpi-ged/memhp".len()], b"acpi-ged/memhp");

        let mut back = S::default();
        let mut b = b;
        // What follows the section in a stream: a nested structure at the end peeks past it.
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ACPI_GED, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }
}
