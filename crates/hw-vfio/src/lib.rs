// SPDX-License-Identifier: GPL-2.0-or-later

//! VFIO PCI, platform, AP and CCW devices, and the vfio-user client.
//!
//! So far this is `vfio-pci` on Linux hosts, QEMU's hw/vfio/pci.c over the legacy container
//! of hw/vfio/container-legacy.c: the config space, BARs and ROM of a host function bound to
//! `vfio-pci` are given to the guest, with the parts QEMU emulates emulated the same way, and the
//! guest's RAM is mapped for the device's DMA, and INTx, MSI and MSI-X reach the guest. DMA
//! through a vIOMMU, iommufd and the device quirks come later. The plan for the rest of this
//! crate is in `spec/24-workspace-layout.md`.

mod host;
#[cfg(target_os = "linux")]
pub mod pci;

pub use host::PciHostDeviceAddress;

/// `PCI_ANY_ID`: the ID properties of `vfio-pci` keep the host's value.
pub const PCI_ANY_ID: u32 = !0;

/// The properties of `vfio-pci`, `vfio_pci_properties` of hw/vfio/pci.c.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfioPciConf {
    /// The qdev `id`.
    pub id: Option<String>,
    /// `host`: the function, in sysfs under `/sys/bus/pci/devices`.
    pub host: Option<PciHostDeviceAddress>,
    /// `sysfsdev`: the sysfs directory of the function, which wins over `host`.
    pub sysfsdev: Option<String>,
    /// `x-pci-vendor-id`, `x-pci-device-id`, `x-pci-sub-vendor-id`, `x-pci-sub-device-id`
    /// and `x-pci-class-code`, or [`PCI_ANY_ID`] for the host's.
    pub vendor_id: u32,
    pub device_id: u32,
    pub sub_vendor_id: u32,
    pub sub_device_id: u32,
    pub class_code: u32,
    /// `x-balloon-allowed`.
    pub balloon_allowed: bool,
    /// `ats`: `None` is auto.
    pub ats: Option<bool>,
    /// `rombar`: -1 by default, 0 hides the ROM BAR.
    pub rombar: i32,
    /// `romfile`: an empty name means no ROM at all.
    pub romfile: Option<String>,
    /// `multifunction`.
    pub multifunction: bool,
    /// `skip-vsc-check`.
    pub skip_vsc_check: bool,
}

impl Default for VfioPciConf {
    fn default() -> Self {
        VfioPciConf {
            id: None,
            host: None,
            sysfsdev: None,
            vendor_id: PCI_ANY_ID,
            device_id: PCI_ANY_ID,
            sub_vendor_id: PCI_ANY_ID,
            sub_device_id: PCI_ANY_ID,
            class_code: PCI_ANY_ID,
            balloon_allowed: false,
            ats: None,
            rombar: -1,
            romfile: None,
            multifunction: false,
            skip_vsc_check: true,
        }
    }
}
