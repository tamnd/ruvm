// SPDX-License-Identifier: GPL-2.0-or-later

//! The Virtual I/O Translation table from hw/acpi/viot.c, which tells the guest which PCI
//! functions sit behind a virtio-iommu.

use super::aml::append_int_noprefix;
use super::linker::BiosLinker;
use super::table::AcpiTable;

/// The virtio-iommu node follows the 48 byte header.
const VIOMMU_OFFSET: u64 = 48;

/// What `build_viot()` needs to know about the machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ViotInfo {
    /// The requester ID of the virtio-iommu-pci function.
    pub bdf: u16,
    /// The bus range of each PCI host bridge that does not bypass the IOMMU, as
    /// `pci_bus_range()` gives it, in any order.
    pub ranges: Vec<(u8, u8)>,
}

fn bdf(bus: u8, devfn: u8) -> u64 {
    (u64::from(bus) << 8) | u64::from(devfn)
}

/// `build_viot()`.
pub fn build_viot(
    tbl: &mut Vec<u8>,
    linker: &mut BiosLinker,
    viot: &ViotInfo,
    oem_id: &str,
    oem_table_id: &str,
) {
    let mut ranges = viot.ranges.clone();
    ranges.sort_by_key(|r| r.0);
    let table = AcpiTable::begin("VIOT", 0, oem_id, oem_table_id, tbl);
    append_int_noprefix(tbl, ranges.len() as u64 + 1, 2); // Node count
    append_int_noprefix(tbl, VIOMMU_OFFSET, 2); // Node offset
    append_int_noprefix(tbl, 0, 8); // Reserved

    // Virtio-iommu node
    append_int_noprefix(tbl, 3, 1); // Type: virtio-pci IOMMU
    append_int_noprefix(tbl, 0, 1); // Reserved
    append_int_noprefix(tbl, 16, 2); // Length
    append_int_noprefix(tbl, 0, 2); // PCI Segment
    append_int_noprefix(tbl, u64::from(viot.bdf), 2); // PCI BDF number
    append_int_noprefix(tbl, 0, 8); // Reserved

    // build_pci_host_range()
    for (min, max) in ranges {
        append_int_noprefix(tbl, 1, 1); // Type: PCI range
        append_int_noprefix(tbl, 0, 1); // Reserved
        append_int_noprefix(tbl, 24, 2); // Length
        append_int_noprefix(tbl, bdf(min, 0), 4); // Endpoint start
        append_int_noprefix(tbl, 0, 2); // PCI Segment start
        append_int_noprefix(tbl, 0, 2); // PCI Segment end
        append_int_noprefix(tbl, bdf(min, 0), 2); // PCI BDF start
        append_int_noprefix(tbl, bdf(max, 0xff), 2); // PCI BDF end
        append_int_noprefix(tbl, VIOMMU_OFFSET, 2); // Output node
        append_int_noprefix(tbl, 0, 6); // Reserved
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

    fn build(viot: &ViotInfo) -> Vec<u8> {
        let mut tbl = Vec::new();
        let mut linker = BiosLinker::new();
        linker.alloc(TABLE_FILE, 64, false);
        build_viot(&mut tbl, &mut linker, viot, APPNAME6, APPNAME8);
        tbl[9] = 0;
        tbl[9] = checksum(&tbl);
        tbl
    }

    #[test]
    fn q35_matches_qemu() {
        // test_acpi_q35_viot: pcie.0 and pcie.200 bypass the IOMMU, pcie.100 and pcie.300 do not.
        let viot = ViotInfo { bdf: 0x10, ranges: vec![(0x30, 0x30), (0x10, 0x10)] };
        assert_eq!(build(&viot), expected("x86/q35/VIOT.viot"));
    }

    #[test]
    fn arm_virt_matches_qemu() {
        let viot = ViotInfo { bdf: 0x08, ranges: vec![(0, 0)] };
        assert_eq!(build(&viot), expected("aarch64/virt/VIOT"));
    }
}
