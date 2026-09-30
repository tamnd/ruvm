// SPDX-License-Identifier: GPL-2.0-or-later

//! AML shared by the PCI host bridges: `_OSC`, the `EDSM` helper and the `_CRS` range sets,
//! from hw/acpi/pci.c, hw/acpi/pci-bridge.c and the CRS helpers in hw/acpi/aml-build.c.

use super::aml::{self, Aml, Serialize};

/// The PCI Firmware Specification 3.1 `_DSM` UUID.
pub const PCI_DSM_UUID: &str = "E5C937D0-3553-4D7A-9117-EA4D19C3434D";

/// `build_pci_host_bridge_osc_method()`. PME, AER and SHPC are always granted. Native PCIe
/// hotplug is granted only when ACPI PCI hotplug is off.
pub fn host_bridge_osc(enable_native_pcie_hotplug: bool) -> Aml {
    let cdw1 = aml::name("CDW1");
    let ctrl = aml::local(0);
    let mut method = aml::method("_OSC", 4, Serialize::NotSerialized);
    method.append(&aml::create_dword_field(&aml::arg(3), &aml::int(0), "CDW1"));

    let mut if_ctx =
        aml::if_(&aml::equal(&aml::arg(0), &aml::touuid("33DB4D5B-1FF7-401C-9657-7441C03DD766")));
    if_ctx.append(&aml::create_dword_field(&aml::arg(3), &aml::int(4), "CDW2"));
    if_ctx.append(&aml::create_dword_field(&aml::arg(3), &aml::int(8), "CDW3"));
    if_ctx.append(&aml::store(&aml::name("CDW3"), &ctrl));
    let mask = 0x1E | u64::from(enable_native_pcie_hotplug);
    if_ctx.append(&aml::and(&ctrl, &aml::int(mask), Some(&ctrl)));

    // Unknown revision.
    let mut if_ctx2 = aml::if_(&aml::lnot(&aml::equal(&aml::arg(1), &aml::int(1))));
    if_ctx2.append(&aml::or(&cdw1, &aml::int(0x08), Some(&cdw1)));
    if_ctx.append(&if_ctx2);

    // Capabilities bits were masked.
    let mut if_ctx2 = aml::if_(&aml::lnot(&aml::equal(&aml::name("CDW3"), &ctrl)));
    if_ctx2.append(&aml::or(&cdw1, &aml::int(0x10), Some(&cdw1)));
    if_ctx.append(&if_ctx2);

    if_ctx.append(&aml::store(&ctrl, &aml::name("CDW3")));
    method.append(&if_ctx);

    // Unrecognized UUID.
    let mut else_ctx = aml::else_();
    else_ctx.append(&aml::or(&cdw1, &aml::int(4), Some(&cdw1)));
    method.append(&else_ctx);

    method.append(&aml::return_(&aml::arg(3)));
    method
}

/// `build_append_pci_dsm_func0_common()`: set `retvar` to an empty capability buffer and bail
/// out for a UUID or revision this does not handle.
pub fn dsm_func0_common(ctx: &mut Aml, retvar: &Aml) {
    ctx.append(&aml::store(&aml::buffer(1, Some(&[0])), retvar));
    let mut if_ctx = aml::if_(&aml::lnot(&aml::equal(&aml::arg(0), &aml::touuid(PCI_DSM_UUID))));
    if_ctx.append(&aml::return_(retvar));
    ctx.append(&if_ctx);
    let mut if_ctx = aml::if_(&aml::lless(&aml::arg(1), &aml::int(2)));
    if_ctx.append(&aml::return_(retvar));
    ctx.append(&if_ctx);
}

/// `build_pci_bridge_edsm()`, the `_DSM` body for devices with a static `acpi-index`.
pub fn bridge_edsm() -> Aml {
    let zero = aml::int(0);
    let func = aml::arg(2);
    let ret = aml::local(0);
    let aidx = aml::local(1);
    let params = aml::arg(4);
    let mut method = aml::method("EDSM", 5, Serialize::Serialized);

    // Function 0: functions 1 and 7 are supported.
    let mut if_ctx = aml::if_(&aml::equal(&func, &zero));
    dsm_func0_common(&mut if_ctx, &ret);
    if_ctx.append(&aml::store(&aml::int(1 | 1 << 7), &aml::index(&ret, &zero)));
    if_ctx.append(&aml::return_(&ret));
    method.append(&if_ctx);

    // Function 7, the device label. The index is stored at run time because some guests choke
    // on a package initialized with computed data.
    let mut if_ctx = aml::if_(&aml::equal(&func, &aml::int(7)));
    let mut pkg = aml::package(2);
    pkg.append(&zero);
    pkg.append(&aml::string(""));
    if_ctx.append(&aml::store(&pkg, &ret));
    if_ctx.append(&aml::store(&aml::derefof(&aml::index(&params, &aml::int(0))), &aidx));
    if_ctx.append(&aml::store(&aidx, &aml::index(&ret, &zero)));
    if_ctx.append(&aml::return_(&ret));
    method.append(&if_ctx);
    method
}

/// One `CrsRangeEntry`, inclusive on both ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrsRange {
    pub base: u64,
    pub limit: u64,
}

/// `CrsRangeSet`: the I/O and memory windows claimed by expander buses, which PCI0 leaves out
/// of its `_CRS`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrsRangeSet {
    pub io_ranges: Vec<CrsRange>,
    pub mem_ranges: Vec<CrsRange>,
    pub mem_64bit_ranges: Vec<CrsRange>,
}

/// `crs_range_insert()`.
pub fn crs_range_insert(ranges: &mut Vec<CrsRange>, base: u64, limit: u64) {
    ranges.push(CrsRange { base, limit });
}

/// `crs_replace_with_free_ranges()`: given the used ranges inside `[start, end]`, replace them
/// with the gaps between them.
pub fn crs_replace_with_free_ranges(ranges: &mut Vec<CrsRange>, start: u64, end: u64) {
    ranges.sort_by_key(|r| r.base);
    let mut free = Vec::new();
    let mut free_base = start;
    for used in ranges.iter() {
        if free_base < used.base {
            crs_range_insert(&mut free, free_base, used.base - 1);
        }
        free_base = used.limit + 1;
    }
    if free_base < end {
        crs_range_insert(&mut free, free_base, end);
    }
    *ranges = free;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_ranges_are_the_gaps() {
        let mut r = vec![CrsRange { base: 0xc000_0000, limit: 0xcfff_ffff }];
        crs_replace_with_free_ranges(&mut r, 0x8000_0000, 0xfebf_ffff);
        assert_eq!(
            r,
            [
                CrsRange { base: 0x8000_0000, limit: 0xbfff_ffff },
                CrsRange { base: 0xd000_0000, limit: 0xfebf_ffff },
            ]
        );
        let mut r = Vec::new();
        crs_replace_with_free_ranges(&mut r, 0x0d00, 0xffff);
        assert_eq!(r, [CrsRange { base: 0x0d00, limit: 0xffff }]);
    }
}
