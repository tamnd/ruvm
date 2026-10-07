// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ich9_smb` section, from hw/i2c/smbus_ich9.c, with the `pmsmb` structure from
//! hw/i2c/pm_smbus.c. The `i2c_bus` section of its bus is in `legacy`.

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_i2c::{Ich9Smbus, Ich9SmbusVmState, PmSmbusVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use super::pci::VMSTATE_PCI_DEVICE;

/// `pmsmb_vmstate`.
static VMSTATE_PMSMB: LazyLock<VmStateDescription<PmSmbusVmState>> = LazyLock::new(|| {
    type S = PmSmbusVmState;
    VmStateDescription::new("pmsmb").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("smb_stat", |s: &mut S| &mut s.smb_stat),
        VmStateField::scalar("smb_ctl", |s: &mut S| &mut s.smb_ctl),
        VmStateField::scalar("smb_cmd", |s: &mut S| &mut s.smb_cmd),
        VmStateField::scalar("smb_addr", |s: &mut S| &mut s.smb_addr),
        VmStateField::scalar("smb_data0", |s: &mut S| &mut s.smb_data0),
        VmStateField::scalar("smb_data1", |s: &mut S| &mut s.smb_data1),
        VmStateField::scalar("smb_index", |s: &mut S| &mut s.smb_index),
        VmStateField::array("smb_data", |s: &mut S| &mut s.smb_data),
        VmStateField::scalar("smb_auxctl", |s: &mut S| &mut s.smb_auxctl),
        VmStateField::scalar("smb_blkdata", |s: &mut S| &mut s.smb_blkdata),
        VmStateField::scalar("i2c_enable", |s: &mut S| &mut s.i2c_enable),
        VmStateField::scalar("op_done", |s: &mut S| &mut s.op_done),
        VmStateField::scalar("in_i2c_block_read", |s: &mut S| &mut s.in_i2c_block_read),
        VmStateField::scalar("start_transaction_on_status_read", |s: &mut S| {
            &mut s.start_transaction_on_status_read
        }),
    ])
});

/// `vmstate_ich9_smbus`.
pub(crate) static VMSTATE_ICH9_SMBUS: LazyLock<VmStateDescription<Ich9SmbusVmState>> =
    LazyLock::new(|| {
        type S = Ich9SmbusVmState;
        VmStateDescription::new("ich9_smb").version_id(1).minimum_version_id(1).fields([
            VmStateField::structure("dev", &VMSTATE_PCI_DEVICE, |s: &mut S| &mut s.dev),
            VmStateField::scalar("irq_enabled", |s: &mut S| &mut s.irq_enabled),
            VmStateField::structure("smb", &VMSTATE_PMSMB, |s: &mut S| &mut s.smb).version(1),
        ])
    });

/// Registers the `0000:00:1f.3/ich9_smb` section of `smbus`, instance 0.
pub(crate) fn register(savevm: &mut SaveVm, smbus: &Arc<Ich9Smbus>) {
    let (get, put) = (Arc::clone(smbus), Arc::clone(smbus));
    savevm.register_vmsd(
        "0000:00:1f.3/",
        Some(0),
        &VMSTATE_ICH9_SMBUS,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_hw_pci::PciDeviceVmState;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn ich9_smb_layout_matches_qemu() {
        let mut config = vec![0; 256];
        config[0] = 0x86;
        let mut smb = PmSmbusVmState {
            smb_stat: 0x42,
            smb_index: 7,
            smb_blkdata: 0x99,
            op_done: true,
            start_transaction_on_status_read: true,
            ..Default::default()
        };
        smb.smb_data[31] = 0x55;
        let mut s = Ich9SmbusVmState {
            dev: PciDeviceVmState { version_id: 2, config, irq_state: [0; 4] },
            irq_enabled: true,
            smb,
        };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ICH9_SMBUS, &mut s).unwrap();
        let b = f.into_inner();
        // QEMU's vmdesc: dev 276, irq_enabled 1, smb 48.
        assert_eq!(b.len(), 276 + 1 + 48);
        assert_eq!(b[276], 1);
        let smb = &b[277..];
        assert_eq!(smb[0], 0x42);
        assert_eq!(&smb[6..10], &7u32.to_be_bytes());
        assert_eq!(smb[10 + 31], 0x55);
        assert_eq!(smb[43], 0x99);
        assert_eq!(&smb[44..], &[0, 1, 0, 1]);

        let mut back = Ich9SmbusVmState {
            dev: PciDeviceVmState { version_id: 2, config: vec![0; 256], irq_state: [0; 4] },
            ..Default::default()
        };
        // What follows the section in a stream: the nested structure at the end peeks past it.
        let b = [&b[..], &[0]].concat();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ICH9_SMBUS, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }
}
