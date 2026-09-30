// SPDX-License-Identifier: GPL-2.0-or-later

//! ACPI table generation: the AML builder, the BIOS linker and loader script, the generic tables
//! and the per machine table sets.

pub mod aml;
pub mod cpuhp;
pub mod devices;
pub mod linker;
pub mod microvm;
pub mod pci;
pub mod pcihp;
pub mod q35;
pub mod table;
pub mod x86;

use linker::BiosLinker;

/// `AcpiBuildTables`: the blobs a machine exposes through fw_cfg.
#[derive(Debug, Default)]
pub struct BuildTables {
    /// `etc/acpi/tables`.
    pub table_data: Vec<u8>,
    /// `etc/acpi/rsdp`.
    pub rsdp: Vec<u8>,
    /// `etc/tpm/log`.
    pub tcpalog: Vec<u8>,
    /// `etc/vmgenid_guid`.
    pub vmgenid: Vec<u8>,
    /// `etc/hardware_errors`.
    pub hardware_errors: Vec<u8>,
    /// `etc/table-loader`.
    pub linker: BiosLinker,
}

impl BuildTables {
    /// `acpi_build_tables_init()`.
    pub fn new() -> Self {
        Self::default()
    }
}
