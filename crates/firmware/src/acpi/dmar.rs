// SPDX-License-Identifier: GPL-2.0-or-later

//! The DMA Remapping table that `build_dmar_q35()` in hw/i386/acpi-build.c makes for an
//! intel-iommu.

use super::aml::append_int_noprefix;
use super::linker::BiosLinker;
use super::table::AcpiTable;

/// `Q35_HOST_BRIDGE_IOMMU_ADDR`.
const DMAR_REG_BASE: u64 = 0xfed9_0000;
/// `Q35_PSEUDO_BUS_PLATFORM`, the bus of the IOAPIC scope.
const PSEUDO_BUS_PLATFORM: u64 = 0xff;
/// `ACPI_BUILD_IOAPIC_ID`.
const IOAPIC_ID: u64 = 0;

/// One root bus function in the DRHD scope, `dmar_host_bridges()` and
/// `insert_scope()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmarScope {
    pub devfn: u8,
    /// A PCI-PCI bridge, which takes the subhierarchy scope type.
    pub bridge: bool,
}

/// What `build_dmar_q35()` needs to know about the machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DmarInfo {
    /// `aw-bits`.
    pub aw_bits: u8,
    /// Interrupt remapping is on.
    pub intr: bool,
    /// `device-iotlb`, which adds the ATSR structure.
    pub device_iotlb: bool,
    /// The functions on the root bus, in devfn order.
    pub scopes: Vec<DmarScope>,
}

/// `build_dmar_q35()`.
pub fn build_dmar(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    dmar: &DmarInfo,
    oem_id: &str,
    oem_table_id: &str,
) {
    let table = AcpiTable::begin("DMAR", 1, oem_id, oem_table_id, tbl);
    append_int_noprefix(tbl, u64::from(dmar.aw_bits) - 1, 1); // Host Address Width
    append_int_noprefix(tbl, u64::from(dmar.intr), 1); // Flags
    append_int_noprefix(tbl, 0, 10); // Reserved

    // DRHD
    append_int_noprefix(tbl, 0, 2); // Type
    append_int_noprefix(tbl, 16 + 8 * (dmar.scopes.len() as u64 + 1), 2); // Length
    append_int_noprefix(tbl, 0, 1); // Flags
    append_int_noprefix(tbl, 0, 1); // Reserved
    append_int_noprefix(tbl, 0, 2); // Segment Number
    append_int_noprefix(tbl, DMAR_REG_BASE, 8); // Register Base Address

    // Scope definition for the root-complex IOAPIC
    append_int_noprefix(tbl, 3, 1); // Type: IOAPIC
    append_int_noprefix(tbl, 8, 1); // Length
    append_int_noprefix(tbl, 0, 2); // Reserved
    append_int_noprefix(tbl, IOAPIC_ID, 1); // Enumeration ID
    append_int_noprefix(tbl, PSEUDO_BUS_PLATFORM, 1); // Start Bus Number
    append_int_noprefix(tbl, 0, 1); // Path: device
    append_int_noprefix(tbl, 0, 1); // Path: function

    for s in &dmar.scopes {
        append_int_noprefix(tbl, if s.bridge { 2 } else { 1 }, 1); // Type
        append_int_noprefix(tbl, 8, 1); // Length
        append_int_noprefix(tbl, 0, 2); // Reserved
        append_int_noprefix(tbl, 0, 1); // Enumeration ID
        append_int_noprefix(tbl, 0, 1); // Start Bus Number
        append_int_noprefix(tbl, u64::from(s.devfn >> 3), 1); // Path: device
        append_int_noprefix(tbl, u64::from(s.devfn & 7), 1); // Path: function
    }

    if dmar.device_iotlb {
        // ATSR
        append_int_noprefix(tbl, 2, 2); // Type
        append_int_noprefix(tbl, 8, 2); // Length
        append_int_noprefix(tbl, 1, 1); // Flags: ALL_PORTS
        append_int_noprefix(tbl, 0, 1); // Reserved
        append_int_noprefix(tbl, 0, 2); // Segment Number
    }
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

    fn build(dmar: &DmarInfo) -> Vec<u8> {
        let mut tbl = Vec::new();
        let mut linker = BiosLinker::new();
        linker.alloc(TABLE_FILE, 64, false);
        build_dmar(&mut tbl, &mut linker, dmar, APPNAME6, APPNAME8);
        tbl[9] = 0;
        tbl[9] = checksum(&tbl);
        tbl
    }

    #[test]
    fn q35_matches_qemu() {
        // test_acpi_q35_tcg_dmar: intel-iommu,intremap=on,device-iotlb=on on a q35 with the
        // host bridge, VGA, LPC, AHCI and SMBus on the root bus.
        let scopes = [0x00, 0x08, 0xf8, 0xfa, 0xfb]
            .into_iter()
            .map(|devfn| DmarScope { devfn, bridge: false })
            .collect();
        let dmar = DmarInfo { aw_bits: 48, intr: true, device_iotlb: true, scopes };
        assert_eq!(build(&dmar), expected("x86/q35/DMAR.dmar"));
    }

    #[test]
    fn bridges_and_no_atsr() {
        let dmar = DmarInfo {
            aw_bits: 39,
            intr: false,
            device_iotlb: false,
            scopes: vec![DmarScope { devfn: 0x10, bridge: true }],
        };
        let t = build(&dmar);
        assert_eq!(t.len(), 36 + 12 + 16 + 16);
        assert_eq!(&t[36..38], &[38, 0]);
        assert_eq!(&t[56..64], &[2, 8, 0, 0, 0, 0, 2, 0]);
    }
}
