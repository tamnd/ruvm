// SPDX-License-Identifier: GPL-2.0-or-later

//! The VMState descriptions of the devices here, shared by the boards that register them: the
//! `memory hotplug state` structure of hw/acpi/memory_hotplug.c and the `acpi-ged` section of
//! hw/acpi/generic_event_device.c.
//!
//! Of the `acpi-ged` subsections only `acpi-ged/memhp` is sent: it has no `needed`. Neither
//! microvm nor virt has hotpluggable CPUs, GHES or PCI hotplug through the GED, so the others
//! never are. Without memory hotplug the GED's memory hotplug state has no slots and nothing maps
//! its registers, so the selector goes out as 0 and is dropped on load.

use std::sync::LazyLock;

use ruvm_vmstate::{VmStateDescription, VmStateField};

use crate::ich9::{MemHotplugDevVmState, MemHotplugVmState};

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
pub static VMSTATE_MEMORY_HOTPLUG: LazyLock<VmStateDescription<MemHotplugVmState>> =
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

/// `AcpiGedState`, the parts its section carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AcpiGedVmState {
    /// `ged_state.sel`.
    pub sel: u32,
    /// `memhp_state`.
    pub memhp: MemHotplugVmState,
}

/// `vmstate_ged_state`.
static VMSTATE_GED_STATE: LazyLock<VmStateDescription<u32>> = LazyLock::new(|| {
    VmStateDescription::new("acpi-ged-state")
        .version_id(1)
        .minimum_version_id(1)
        .field(VmStateField::scalar("sel", |s: &mut u32| s))
});

/// `vmstate_memhp_state`.
static VMSTATE_GED_MEMHP: LazyLock<VmStateDescription<AcpiGedVmState>> = LazyLock::new(|| {
    type S = AcpiGedVmState;
    VmStateDescription::new("acpi-ged/memhp").version_id(1).minimum_version_id(1).field(
        VmStateField::structure("memhp_state", &VMSTATE_MEMORY_HOTPLUG, |s: &mut S| &mut s.memhp)
            .version(1),
    )
});

/// `vmstate_acpi_ged`.
pub static VMSTATE_ACPI_GED: LazyLock<VmStateDescription<AcpiGedVmState>> = LazyLock::new(|| {
    type S = AcpiGedVmState;
    VmStateDescription::new("acpi-ged")
        .version_id(1)
        .minimum_version_id(1)
        .field(
            VmStateField::structure("ged_state", &VMSTATE_GED_STATE, |s: &mut S| &mut s.sel)
                .version(1),
        )
        .subsection(&VMSTATE_GED_MEMHP)
});

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn ged_layout_matches_qemu() {
        let mut s = AcpiGedVmState { sel: 0x8000_0002, memhp: MemHotplugVmState::default() };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ACPI_GED, &mut s).unwrap();
        let b = f.into_inner();
        // sel, then the memhp subsection: its header and the selector, with no slots.
        let memhp = 2 + "acpi-ged/memhp".len() + 4 + 4;
        assert_eq!(b.len(), 4 + memhp);
        assert_eq!(b[..4], [0x80, 0, 0, 2]);
        assert_eq!(&b[6..6 + "acpi-ged/memhp".len()], b"acpi-ged/memhp");

        let mut back = AcpiGedVmState::default();
        let mut b = b;
        // What follows the section in a stream: a nested structure at the end peeks past it.
        b.push(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ACPI_GED, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }
}
