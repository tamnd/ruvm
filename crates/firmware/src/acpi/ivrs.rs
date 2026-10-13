// SPDX-License-Identifier: GPL-2.0-or-later

//! The I/O Virtualization Reporting Structure that `build_amd_iommu()` in
//! hw/i386/acpi-build.c makes for an amd-iommu.

use super::aml::append_int_noprefix;
use super::linker::BiosLinker;
use super::table::AcpiTable;

/// `IOAPIC_SB_DEVID`, the device id the IOAPIC special entry names.
const IOAPIC_SB_DEVID: u64 = 0xa0;

/// One IVHD device entry, from `insert_ivhd()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IvhdDevice {
    /// A "Select" entry for one bus, device and function.
    Select(u16),
    /// A "Start of Range" and "End of Range" pair covering a leaf bus.
    Range(u16, u16),
    /// An "Alias Start of Range" from the first id, aliased to the second, and the "End of
    /// Range" with the third.
    Alias(u16, u16, u16),
}

/// What `build_amd_iommu()` needs to know about the machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IvrsInfo {
    /// The `addr` of the AMDVI-PCI function.
    pub devfn: u8,
    /// `capab_offset`, where the secure device capability sits in its config space.
    pub cap_offset: u8,
    /// The MMIO base, `mr_mmio.addr`.
    pub base: u64,
    /// Interrupt remapping is supported, which adds the IOAPIC special entry.
    pub intr: bool,
    /// `xtsup`.
    pub xtsup: bool,
    /// `dma-translation`.
    pub dma_translation: bool,
    /// `amdvi_extended_feature_register()`.
    pub efr: u64,
    /// The device entries from the bus walk, empty for none.
    pub devices: Vec<IvhdDevice>,
}

/// `build_amd_iommu()`.
pub fn build_ivrs(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    ivrs: &IvrsInfo,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("IVRS", 1, oem_id, oem_table_id, tbl);
    // IVinfo: EFRSup and a PASize of 40.
    append_int_noprefix(tbl, 1 | 40 << 8, 4);
    append_int_noprefix(tbl, 0, 8); // Reserved

    let mut blob = Vec::new();
    for d in &ivrs.devices {
        match *d {
            IvhdDevice::Select(id) => append_int_noprefix(&mut blob, u64::from(id) << 8 | 0x2, 4),
            IvhdDevice::Range(start, end) => {
                append_int_noprefix(&mut blob, u64::from(start) << 8 | 0x3, 4);
                append_int_noprefix(&mut blob, u64::from(end) << 8 | 0x4, 4);
            }
            IvhdDevice::Alias(start, alias, end) => {
                append_int_noprefix(&mut blob, u64::from(start) << 8 | 0x43, 4);
                append_int_noprefix(&mut blob, u64::from(alias) << 8, 4);
                append_int_noprefix(&mut blob, u64::from(end) << 8 | 0x4, 4);
            }
        }
    }
    if blob.is_empty() {
        // A type 1 entry that covers all devices.
        append_int_noprefix(&mut blob, 0x1, 4);
    }
    if ivrs.intr {
        // The IOAPIC special entry, which Linux wants before it turns on remapping.
        append_int_noprefix(&mut blob, 1 << 56 | IOAPIC_SB_DEVID << 40 | 0x48, 8);
    }

    // IVHD type 10h
    append_int_noprefix(tbl, 0x10, 1);
    // HtTunEn, iotblSup, PrefSup and PPRSup
    append_int_noprefix(tbl, 1 | 1 << 4 | 1 << 6 | 1 << 7, 1);
    append_int_noprefix(tbl, blob.len() as u64 + 24, 2); // Length
    append_int_noprefix(tbl, u64::from(ivrs.devfn), 2); // DeviceID
    append_int_noprefix(tbl, u64::from(ivrs.cap_offset), 2); // Capability offset
    append_int_noprefix(tbl, ivrs.base, 8); // IOMMU base address
    append_int_noprefix(tbl, 0, 2); // PCI Segment Group
    append_int_noprefix(tbl, 0, 2); // IOMMU info
    // IOMMU Feature Reporting. QEMU also ORs in HATS and GATS as 48 << 30 and 48 << 28, but
    // both land above bit 31 and the field is four bytes, so only GTSup and GASup are left.
    append_int_noprefix(tbl, 1 << 2 | 1 << 6 | u64::from(ivrs.xtsup), 4);
    tbl.extend_from_slice(&blob);

    // IVHD type 11h
    append_int_noprefix(tbl, 0x11, 1);
    append_int_noprefix(tbl, 1 | 1 << 4, 1); // HtTunEn and iotblSup
    append_int_noprefix(tbl, blob.len() as u64 + 40, 2); // Length
    append_int_noprefix(tbl, u64::from(ivrs.devfn), 2); // DeviceID
    append_int_noprefix(tbl, u64::from(ivrs.cap_offset), 2); // Capability offset
    append_int_noprefix(tbl, ivrs.base, 8); // IOMMU base address
    append_int_noprefix(tbl, 0, 2); // PCI Segment Group
    append_int_noprefix(tbl, 0, 2); // IOMMU info
    append_int_noprefix(tbl, u64::from(!ivrs.dma_translation), 4); // Attributes: HATDis
    append_int_noprefix(tbl, ivrs.efr, 8); // EFR Register Image
    append_int_noprefix(tbl, 0, 8); // EFR Register Image 2
    tbl.extend_from_slice(&blob);

    table.end(Some(linker), tbl);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acpi::table::{APPNAME6, APPNAME8, TABLE_FILE, checksum};

    fn expected(path: &str) -> Vec<u8> {
        let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../vendor-qemu/acpi-expected")
            .join(path);
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    }

    fn build(ivrs: &IvrsInfo) -> Vec<u8> {
        let mut tbl = Vec::new();
        let mut linker = BiosLinker::new();
        linker.alloc(TABLE_FILE, 64, false);
        build_ivrs(&mut tbl, &mut linker, ivrs, APPNAME6, APPNAME8);
        tbl[9] = 0;
        tbl[9] = checksum(&tbl);
        tbl
    }

    fn q35() -> IvrsInfo {
        IvrsInfo {
            devfn: 0x10,
            cap_offset: 0x40,
            base: 0xfed8_0000,
            intr: true,
            xtsup: false,
            dma_translation: true,
            efr: 0x29d3,
            devices: [0x00, 0x08, 0x10, 0xf8, 0xfa, 0xfb].map(IvhdDevice::Select).to_vec(),
        }
    }

    #[test]
    fn q35_matches_qemu() {
        // test_acpi_q35_amd_iommu: -device amd-iommu on a q35 with the host bridge, VGA,
        // AMDVI-PCI, LPC, AHCI and SMBus on the root bus.
        assert_eq!(build(&q35()), expected("x86/q35/IVRS.ivrs"));
    }

    #[test]
    fn no_devices_no_intremap() {
        let ivrs = IvrsInfo {
            intr: false,
            xtsup: true,
            dma_translation: false,
            devices: Vec::new(),
            ..q35()
        };
        let t = build(&ivrs);
        assert_eq!(t.len(), 48 + 24 + 4 + 40 + 4);
        assert_eq!(&t[50..52], &[28, 0]);
        assert_eq!(&t[68..72], &[0x45, 0, 0, 0]);
        assert_eq!(&t[72..76], &[1, 0, 0, 0]);
        assert_eq!(&t[96..100], &[1, 0, 0, 0]);
    }

    #[test]
    fn range_and_alias_entries() {
        let ivrs = IvrsInfo {
            intr: false,
            devices: vec![IvhdDevice::Range(0x100, 0x1ff), IvhdDevice::Alias(0x200, 0x18, 0x2ff)],
            ..q35()
        };
        let t = build(&ivrs);
        assert_eq!(&t[72..80], &[0x3, 0, 1, 0, 0x4, 0xff, 1, 0]);
        assert_eq!(&t[80..92], &[0x43, 0, 2, 0, 0, 0x18, 0, 0, 0x4, 0xff, 2, 0]);
    }
}
