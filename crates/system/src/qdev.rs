// SPDX-License-Identifier: GPL-2.0-or-later

//! The QOM types of the devices the aarch64 and riscv64 `virt` boards plug, with the parents
//! QEMU 11.1 gives them, so `qom-list-types` and `device-list-properties` know them.
//!
//! qos-test, bios-tables-test and cdrom-test decide what to run from
//! `qom-list-types implements=device`, and a qos-test case that fails stops the whole test.
//! So a device is listed here only once its qos-test cases pass under the qtest accelerator.
//! virtio-rng was the first, then virtio-blk. The types carry no properties yet: the boards
//! build their devices from the `-device` options, not from these types.

use ruvm_qom::{Registry, TYPE_OBJECT, TypeInfo};

/// `TYPE_DEVICE`.
const TYPE_DEVICE: &str = "device";
/// `TYPE_SYS_BUS_DEVICE`.
const TYPE_SYS_BUS_DEVICE: &str = "sys-bus-device";
/// `TYPE_PCI_DEVICE`.
const TYPE_PCI_DEVICE: &str = "pci-device";
/// `TYPE_VIRTIO_DEVICE`.
const TYPE_VIRTIO_DEVICE: &str = "virtio-device";
/// `TYPE_VIRTIO_PCI`.
const TYPE_VIRTIO_PCI: &str = "virtio-pci";

/// Each type with its parent and whether it is abstract. A virtio PCI device is a
/// `virtio-*-pci` on a `virtio-*-pci-base`, as `virtio_pci_types_register()` makes them.
const TYPES: &[(&str, &str, bool)] = &[
    (TYPE_DEVICE, TYPE_OBJECT, true),
    (TYPE_SYS_BUS_DEVICE, TYPE_DEVICE, true),
    (TYPE_PCI_DEVICE, TYPE_DEVICE, true),
    (TYPE_VIRTIO_DEVICE, TYPE_DEVICE, true),
    (TYPE_VIRTIO_PCI, TYPE_PCI_DEVICE, true),
    ("virtio-mmio", TYPE_SYS_BUS_DEVICE, false),
    ("virtio-blk-device", TYPE_VIRTIO_DEVICE, false),
    ("virtio-blk-pci-base", TYPE_VIRTIO_PCI, true),
    ("virtio-blk-pci", "virtio-blk-pci-base", false),
    ("virtio-rng-device", TYPE_VIRTIO_DEVICE, false),
    ("virtio-rng-pci-base", TYPE_VIRTIO_PCI, true),
    ("virtio-rng-pci", "virtio-rng-pci-base", false),
];

pub(crate) fn register_types(registry: &Registry) {
    registry.register_all(TYPES.iter().map(|&(name, parent, abstract_)| {
        let info = TypeInfo::new(name).parent(parent);
        if abstract_ { info.abstract_() } else { info }
    }));
}

#[cfg(test)]
mod tests {
    use ruvm_qom::qmp::{device_list_properties, qom_list_types};

    use super::*;

    fn names(registry: &Registry, implements: &str, abstract_: bool) -> Vec<String> {
        let ruvm_qapi::QValue::List(l) = qom_list_types(registry, Some(implements), abstract_)
        else {
            panic!("a list");
        };
        let mut v: Vec<String> =
            l.iter().map(|t| t.as_dict().unwrap().get_str("name").unwrap().to_owned()).collect();
        v.sort();
        v
    }

    #[test]
    fn device_types_match_qemu() {
        let registry = Registry::new();
        register_types(&registry);
        assert_eq!(
            names(&registry, "device", false),
            [
                "virtio-blk-device",
                "virtio-blk-pci",
                "virtio-mmio",
                "virtio-rng-device",
                "virtio-rng-pci"
            ]
        );
        assert_eq!(
            names(&registry, "pci-device", true),
            [
                "pci-device",
                "virtio-blk-pci",
                "virtio-blk-pci-base",
                "virtio-pci",
                "virtio-rng-pci",
                "virtio-rng-pci-base"
            ]
        );
        assert_eq!(names(&registry, "sys-bus-device", true), ["sys-bus-device", "virtio-mmio"]);
        assert!(device_list_properties(&registry, "virtio-rng-pci").is_ok());
        let e = device_list_properties(&registry, "virtio-pci").unwrap_err();
        assert_eq!(e.to_string(), "Parameter 'typename' expects a non-abstract device type");
    }
}
