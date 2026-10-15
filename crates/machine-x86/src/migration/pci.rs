// SPDX-License-Identifier: GPL-2.0-or-later

//! The q35 host side sections: `mch` (hw/pci-host/q35.c), `PCIHost` (hw/pci/pci_host.c) and
//! `PCIBUS`. The `PCIDevice` description every PCI function embeds is
//! [`ruvm_hw_pci::vmstate::VMSTATE_PCI_DEVICE`].

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_pci::vmstate::{VMSTATE_PCIBUS, VMSTATE_PCIHOST};
use ruvm_hw_pci::{MchVmState, PciBus, PciHostState, Q35PciHost};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

pub(crate) use ruvm_hw_pci::vmstate::VMSTATE_PCI_DEVICE;

/// `vmstate_mch`. `mch_post_load()` is in [`Q35PciHost::mch_vmstate_load`].
pub(crate) static VMSTATE_MCH: LazyLock<VmStateDescription<MchVmState>> = LazyLock::new(|| {
    type S = MchVmState;
    VmStateDescription::new("mch").version_id(1).minimum_version_id(1).fields([
        VmStateField::structure("parent_obj", &VMSTATE_PCI_DEVICE, |s: &mut S| &mut s.parent_obj),
        // Used to be smm_enabled.
        VmStateField::unused(1),
    ])
});

/// Registers the MCH's `0000:00:00.0/mch` section, instance 0.
pub(crate) fn register_mch(savevm: &mut SaveVm, q35: &Arc<Q35PciHost>) {
    let (get, put) = (Arc::clone(q35), Arc::clone(q35));
    savevm.register_vmsd(
        "0000:00:00.0/",
        Some(0),
        &VMSTATE_MCH,
        move || Ok(get.mch_vmstate_save()),
        move |s| put.mch_vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

/// Registers the `PCIHost` section of the host bridge, instance 0 with no path prefix: QEMU
/// registers it with a NULL device.
pub(crate) fn register_pci_host(savevm: &mut SaveVm, host: &Arc<PciHostState>) {
    let (get, put) = (Arc::clone(host), Arc::clone(host));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PCIHOST,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// Registers the `PCIBUS` section of the root bus, instance 0 with no path prefix as
/// `pci_root_bus_internal_init()` does.
pub(crate) fn register_pci_bus(savevm: &mut SaveVm, bus: &Arc<PciBus>) {
    let (get, put) = (Arc::clone(bus), Arc::clone(bus));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PCIBUS,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_hw_pci::PciDeviceVmState;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    fn pci_device() -> PciDeviceVmState {
        let mut config = vec![0; 256];
        config[0] = 0x86;
        config[255] = 0x5a;
        PciDeviceVmState { version_id: 2, config, irq_state: [0, 1, 0, 1] }
    }

    #[test]
    fn mch_layout_matches_qemu() {
        let mut s = MchVmState { parent_obj: pci_device() };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_MCH, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 277);
        assert_eq!(b[276], 0);

        let mut back = MchVmState { parent_obj: pci_device() };
        back.parent_obj.config.fill(0);
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_MCH, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }
}
