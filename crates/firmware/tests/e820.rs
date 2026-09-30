// SPDX-License-Identifier: GPL-2.0-or-later

//! The e820 table and the bytes of the `etc/e820` fw_cfg file.

use ruvm_firmware::e820::{E820_ENTRY_SIZE, E820_RAM, E820_RESERVED, E820Entry, E820Table};

#[test]
fn blob_is_packed_little_endian() {
    let mut t = E820Table::new();
    t.add_entry(0, 0x8000_0000, E820_RAM);
    t.add_entry(0xfd_0000_0000, 0x3_0000_0000, E820_RESERVED);
    t.add_entry(0x1_0000_0000, 0x4000_0000, E820_RAM);
    let blob = t.to_blob();
    assert_eq!(blob.len(), 3 * E820_ENTRY_SIZE);
    #[rustfmt::skip]
    let expect: [u8; 60] = [
        0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0x80, 0, 0, 0, 0,  1, 0, 0, 0,
        0, 0, 0, 0, 0xfd, 0, 0, 0,  0, 0, 0, 0, 3, 0, 0, 0,  2, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0,  0, 0, 0, 0x40, 0, 0, 0, 0,  1, 0, 0, 0,
    ];
    assert_eq!(blob, expect);
    let first: &[u8; E820_ENTRY_SIZE] = blob[..E820_ENTRY_SIZE].try_into().unwrap();
    assert_eq!(E820Entry::from_bytes(first), t.entries()[0]);
}

#[test]
fn get_entry_checks_type_and_index() {
    let mut t = E820Table::new();
    t.add_entry(0, 0x9fc00, E820_RAM);
    t.add_entry(0xfeff_c000, 0x4000, E820_RESERVED);
    assert_eq!(t.get_entry(0, E820_RAM), Some((0, 0x9fc00)));
    assert_eq!(t.get_entry(1, E820_RAM), None);
    assert_eq!(t.get_entry(1, E820_RESERVED), Some((0xfeff_c000, 0x4000)));
    assert_eq!(t.get_entry(2, E820_RAM), None);
}

#[test]
fn get_table_seals() {
    let mut t = E820Table::new();
    assert!(t.is_empty());
    t.add_entry(0, 0x1000, E820_RAM);
    assert!(!t.is_done());
    assert_eq!(t.get_table().len(), 1);
    assert!(t.is_done());
    let r = std::panic::catch_unwind(move || {
        let mut t = t;
        t.add_entry(0x1000, 0x1000, E820_RAM);
    });
    assert!(r.is_err());
}
