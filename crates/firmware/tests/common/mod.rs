// SPDX-License-Identifier: GPL-2.0-or-later

//! Helpers shared by the ACPI golden tests.
//!
//! `load()` handles the tables the way firmware would: it runs the checksum commands from the
//! loader script and cleans up the FADT pointers the same way bios-tables-test does before it
//! compares.

#![allow(dead_code)]

use std::path::PathBuf;

use ruvm_firmware::acpi::BuildTables;
use ruvm_firmware::acpi::linker::Command;
use ruvm_firmware::acpi::table::{TABLE_FILE, checksum};

/// Reads `vendor-qemu/acpi-expected/x86/<machine>/<name>`.
pub(crate) fn expected(machine: &str, name: &str) -> Vec<u8> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor-qemu/acpi-expected/x86")
        .join(machine);
    std::fs::read(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Runs the checksum commands and splits `etc/acpi/tables` into its tables, stopping at the
/// zero padding at the end.
pub(crate) fn load(t: &BuildTables) -> Vec<(String, Vec<u8>)> {
    let mut data = t.table_data.clone();
    for cmd in t.linker.commands() {
        if let Command::AddChecksum { file, offset, start, length } = cmd {
            if file == TABLE_FILE {
                let (s, o) = (start as usize, offset as usize);
                data[o] = 0;
                data[o] = checksum(&data[s..s + length as usize]);
            }
        }
    }
    let mut tables = Vec::new();
    let mut at = 0;
    while at + 8 <= data.len() && data[at..at + 4] != [0; 4] {
        let len = u32::from_le_bytes(data[at + 4..at + 8].try_into().unwrap()) as usize;
        let sig = String::from_utf8(data[at..at + 4].to_vec()).unwrap();
        let mut table = data[at..at + len].to_vec();
        // The FACS has no checksum.
        if sig != "FACS" {
            assert_eq!(checksum(&table), 0, "{sig} checksum");
        }
        if sig == "FACP" {
            // test_acpi_fadt_table() zeroes the pointers the firmware filled in and redoes the sum.
            table[36..44].fill(0);
            if table[8] >= 3 {
                table[132..148].fill(0);
            }
            table[9] = 0;
            table[9] = checksum(&table);
        }
        tables.push((sig, table));
        at += len;
    }
    tables
}

pub(crate) fn table(tables: &[(String, Vec<u8>)], sig: &str) -> Vec<u8> {
    tables.iter().find(|(s, _)| s == sig).unwrap_or_else(|| panic!("no {sig}")).1.clone()
}

pub(crate) fn hex(b: &[u8]) -> String {
    b.chunks(16)
        .enumerate()
        .map(|(i, c)| {
            let bytes: Vec<String> = c.iter().map(|x| format!("{x:02x}")).collect();
            format!("{:04x}: {}\n", i * 16, bytes.join(" "))
        })
        .collect()
}

/// Asserts that table `sig` in `t` matches the expected blob `file`.
pub(crate) fn check(t: &BuildTables, machine: &str, sig: &str, file: &str) {
    let got = table(&load(t), sig);
    let want = expected(machine, file);
    assert!(got == want, "{file} differs\ngot:\n{}\nwant:\n{}", hex(&got), hex(&want));
}
