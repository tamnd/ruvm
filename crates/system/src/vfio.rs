// SPDX-License-Identifier: GPL-2.0-or-later

//! `-device vfio-pci` on q35: the properties of hw/vfio/pci.c and the plug.
//!
//! Only the root bus `pcie.0` takes the device. Behind intel-iommu the device does DMA into its
//! own address space, and the vIOMMU is told about it so it can refuse it without caching mode.

use ruvm_base::error::Error;
use ruvm_base::report::Location;
use ruvm_hw_iommu::intel_iommu::OnOffAuto;
use ruvm_hw_vfio::{PciHostDeviceAddress, VfioPciConf};
use ruvm_machine_x86::{FirmwareSearch, X86Board};
use ruvm_qapi::opts::QemuOpts;
use ruvm_qapi::visit::{Visitor, VisitorExt};

use crate::x86::Located;

pub(crate) const TYPE_VFIO_PCI: &str = "vfio-pci";

/// The properties of `vfio-pci` and of `PCIDevice` that ruvm does not take yet.
const UNSUPPORTED: &[&str] = &[
    "vf-token",
    "x-pre-copy-dirty-page-tracking",
    "x-device-dirty-page-tracking",
    "display",
    "xres",
    "yres",
    "x-vga",
    "x-req",
    "x-igd-opregion",
    "x-igd-lpc",
    "x-igd-legacy-mode",
    "enable-migration",
    "x-migration-multifd-transfer",
    "x-migration-load-config-after-iter",
    "x-migration-max-queued-buffers-size",
    "migration-events",
    "x-no-kvm-intx",
    "x-no-kvm-msi",
    "x-no-kvm-msix",
    "x-no-geforce-quirks",
    "x-no-kvm-ioeventfd",
    "x-no-vfio-ioeventfd",
    "x-igd-gms",
    "x-nv-gpudirect-clique",
    "x-msix-relocation",
    "iommufd",
    "x-vpasid-cap-offset",
    "fd",
    "ramfb",
    "use-legacy-x86-rom",
    "x-ramfb-migrate",
    "romsize",
    "failover_pair_id",
    "acpi-index",
];

/// A planned `vfio-pci`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VfioPlug {
    pub conf: VfioPciConf,
    pub devfn: Option<u8>,
    pub loc: Option<Location>,
}

/// `qdev_device_add()` up to realize for `vfio-pci`. `pci` says whether the machine has the
/// root bus `pcie.0`.
pub(crate) fn plan_device(
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
) -> Result<VfioPlug, Located> {
    match opts.get("bus") {
        Some("pcie.0") if pci => {}
        Some(b) => return Err(Located::new(loc, format!("Bus '{b}' not found"))),
        None => {}
    }
    if !pci {
        return Err(Located::new(loc, format!("No 'PCI' bus found for device '{TYPE_VFIO_PCI}'")));
    }
    let mut plug = VfioPlug { conf: VfioPciConf::default(), devfn: None, loc: loc.clone() };
    let conf = &mut plug.conf;
    let at = |e: Error| Located(loc.clone(), e);
    let boolean = |k: &str, v: &str| {
        crate::display::prop(k, v, |vi, n, out| vi.type_bool(Some(n), out)).map_err(at)
    };
    let uint32 = |k: &str, v: &str| {
        crate::display::prop(k, v, |vi, n, out| vi.type_uint32(Some(n), out)).map_err(at)
    };
    let bad_value = |k: &str, v: &str| {
        Located::new(loc, format!("Property '{TYPE_VFIO_PCI}.{k}' doesn't take value '{v}'"))
    };
    for (k, v) in opts.iter() {
        match k {
            "driver" | "bus" => {}
            "id" => conf.id = Some(v.to_string()),
            "host" => {
                conf.host = Some(PciHostDeviceAddress::parse(v).ok_or_else(|| bad_value(k, v))?)
            }
            "sysfsdev" => conf.sysfsdev = Some(v.to_string()),
            "x-pci-vendor-id" => conf.vendor_id = uint32(k, v)?,
            "x-pci-device-id" => conf.device_id = uint32(k, v)?,
            "x-pci-sub-vendor-id" => conf.sub_vendor_id = uint32(k, v)?,
            "x-pci-sub-device-id" => conf.sub_device_id = uint32(k, v)?,
            "x-pci-class-code" => conf.class_code = uint32(k, v)?,
            "x-balloon-allowed" => conf.balloon_allowed = boolean(k, v)?,
            "skip-vsc-check" => conf.skip_vsc_check = boolean(k, v)?,
            "multifunction" => conf.multifunction = boolean(k, v)?,
            "x-no-mmap" => conf.no_mmap = boolean(k, v)?,
            "x-intx-mmap-timeout-ms" => conf.intx_mmap_timeout_ms = uint32(k, v)?,
            "ats" => {
                let ats = OnOffAuto::parse(v).ok_or_else(|| {
                    Located::new(loc, format!("Parameter '{k}' does not accept value '{v}'"))
                })?;
                conf.ats = match ats {
                    OnOffAuto::On => Some(true),
                    OnOffAuto::Off => Some(false),
                    OnOffAuto::Auto => None,
                };
            }
            "rombar" => {
                conf.rombar = crate::display::prop(k, v, |vi, n, out| vi.type_int32(Some(n), out))
                    .map_err(at)?;
            }
            "romfile" => conf.romfile = Some(v.to_string()),
            "addr" => {
                plug.devfn = Some(crate::display::parse_devfn(v).ok_or_else(|| bad_value(k, v))?);
            }
            _ if UNSUPPORTED.contains(&k) => {
                return Err(Located::new(
                    loc,
                    format!("Property '{TYPE_VFIO_PCI}.{k}' is not supported by ruvm yet"),
                ));
            }
            _ => {
                return Err(Located::new(loc, format!("Property '{TYPE_VFIO_PCI}.{k}' not found")));
            }
        }
    }
    // pci_add_option_rom() loads the file through fw_cfg then, which ruvm does not do.
    if conf.rombar == 0 && conf.romfile.as_deref().is_some_and(|f| !f.is_empty()) {
        return Err(Located::new(loc, "romfile with rombar=0 is not supported by ruvm yet"));
    }
    Ok(plug)
}

/// Realizes a planned `vfio-pci` on q35. The `first` one also hooks the system reset.
#[cfg(target_os = "linux")]
pub(crate) fn realize_x86(
    board: &mut X86Board,
    plug: &VfioPlug,
    firmware: &FirmwareSearch,
    clock: &std::sync::Arc<ruvm_hw_core::timer::Clock>,
    first: bool,
) -> Result<(), Located> {
    use std::sync::Arc;

    use ruvm_base::report::push_location;
    use ruvm_hw_pci::regs::PCI_ROM_SLOT;

    // qdev_device_add() runs realize with the option's location current, so what realize
    // reports itself, such as the dma-buf warning, starts with "-device vfio-pci,...: ".
    let _loc = plug.loc.clone().map(push_location);
    let X86Board::Q35(m, _) = board else {
        return Err(Located::new(
            &plug.loc,
            format!("No 'PCI' bus found for device '{TYPE_VFIO_PCI}'"),
        ));
    };
    // pci_device_iommu_address_space() needs the devfn before the device is on the bus.
    let bus = m.pci_bus();
    let dma_as = plug
        .devfn
        .or_else(|| bus.first_free_devfn())
        .and_then(|devfn| m.pci_dma().address_space(u16::from(devfn)))
        .unwrap_or_else(|| Arc::clone(m.memory_as()));
    let viommu = Q35Viommu(m);
    let dev = ruvm_hw_vfio::pci::realize(
        bus,
        plug.devfn,
        true,
        &dma_as,
        m.memory_as(),
        Some(&viommu),
        clock,
        &plug.conf,
    )
    .map_err(|e| Located(plug.loc.clone(), e))?;
    // vfio_address_space_get() registers vfio_device_reset_handler() along with the first
    // device, so it runs after the devices themselves are reset.
    if first {
        m.add_reset_hook(Box::new(ruvm_hw_vfio::pci::reset_handler));
    }
    // pci_add_option_rom()
    if let Some(name) = plug.conf.romfile.as_deref().filter(|f| !f.is_empty()) {
        let at = |e: String| Located::new(&plug.loc, e);
        let Some(data) = firmware.load(name) else {
            return Err(at(format!("failed to find romfile \"{name}\"")));
        };
        if data.is_empty() {
            return Err(at(format!("romfile \"{name}\" is empty")));
        }
        let devfn = dev.devfn();
        let block = format!("0000:00:{:02x}.{:x}/{TYPE_VFIO_PCI}.rom", devfn >> 3, devfn & 7);
        let rom = m.add_device_rom(&block, &data).map_err(at)?;
        dev.register_bar(PCI_ROM_SLOT, 0, rom);
    }
    Ok(())
}

/// The vIOMMU of the q35 root bus as vfio-pci sees it.
#[cfg(target_os = "linux")]
struct Q35Viommu<'a>(&'a ruvm_machine_x86::Q35);

#[cfg(target_os = "linux")]
impl ruvm_hw_vfio::pci::HostIommu for Q35Viommu<'_> {
    fn set_iommu_device(&self, devfn: u8) -> Result<(), String> {
        self.0.set_iommu_device(devfn)
    }

    fn unset_iommu_device(&self, devfn: u8) {
        self.0.unset_iommu_device(devfn);
    }
}

/// Realizes a planned `vfio-pci`, which only Linux plans.
#[cfg(not(target_os = "linux"))]
pub(crate) fn realize_x86(
    _board: &mut X86Board,
    plug: &VfioPlug,
    _firmware: &FirmwareSearch,
    _clock: &std::sync::Arc<ruvm_hw_core::timer::Clock>,
    _first: bool,
) -> Result<(), Located> {
    Err(Located::new(&plug.loc, format!("'{TYPE_VFIO_PCI}' is not a valid device model name")))
}
