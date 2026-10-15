// SPDX-License-Identifier: GPL-2.0-or-later

//! Save and load of the state `vmstate_fw_cfg` carries, and the ACPI file sizes it records.

use ruvm_hw_core::fw_cfg::*;

fn device(dma_enabled: bool) -> FwCfgState {
    let props = FwCfgProps { dma_enabled, ..FwCfgProps::default() };
    let s = FwCfgState::new(props, None).unwrap();
    s.add_bytes(FW_CFG_SIGNATURE, b"QEMU".to_vec());
    s.add_file("etc/test", b"0123456789".to_vec()).unwrap();
    s
}

#[test]
fn read_position_survives() {
    let a = device(true);
    let (_, key, _) = a.files().into_iter().find(|(n, _, _)| n == "etc/test").unwrap();
    assert!(a.select(key));
    assert_eq!(a.data_read(1), u64::from(b'0'));
    assert_eq!(a.data_read(1), u64::from(b'1'));
    let v = a.vmstate_save();
    assert_eq!(v.cur_entry, key);
    assert_eq!(v.cur_offset, 2);
    assert!(v.dma_needed());
    assert!(!v.acpi_mr_needed());

    let b = device(true);
    b.reset();
    b.vmstate_load(&v);
    assert_eq!(b.vmstate_save(), v);
    assert_eq!(b.cur_entry(), key);
    assert_eq!(b.data_read(1), u64::from(b'2'));
}

#[test]
fn acpi_mr_sizes_are_kept() {
    let a = device(false);
    let mut v = a.vmstate_save();
    assert!(!v.dma_needed());
    v.table_mr_size = 0x20000;
    v.linker_mr_size = 0x1000;
    assert!(!v.acpi_mr_needed());
    v.rsdp_mr_size = 36;
    assert!(v.acpi_mr_needed());
    a.vmstate_load(&v);
    assert_eq!(a.vmstate_save(), v);
}

#[test]
fn acpi_files_give_the_mr_sizes() {
    // The sizes QEMU 11.1 sends for `virt`.
    let a = device(true);
    a.add_file("etc/acpi/tables", vec![0; 0x20000]).unwrap();
    a.add_file("etc/table-loader", vec![0; 0xc80]).unwrap();
    a.add_file("etc/acpi/rsdp", vec![0; 36]).unwrap();
    let v = a.vmstate_save();
    assert_eq!((v.table_mr_size, v.linker_mr_size, v.rsdp_mr_size), (0x20000, 0xc80, 36));
    assert!(v.acpi_mr_needed());

    a.modify_file("etc/acpi/rsdp", vec![0; 20]).unwrap();
    assert_eq!(a.vmstate_save().rsdp_mr_size, 20);
}
