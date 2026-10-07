// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ich9_ahci` section, from hw/ide/ich.c, with the `ahci` structures from hw/ide/ahci.c
//! and the `ide_bus` and `ide_drive` ones from hw/ide/core.c.
//!
//! The layout is QEMU's in full, but only an idle controller loads: see
//! [`Ich9Ahci::vmstate_load`].

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_storage::{
    AhciPortVmState, AhciVmState, IDE_IO_BUFFER_TOTAL_LEN, Ich9Ahci, Ich9AhciVmState,
    IdeBusVmState, IdeDriveVmState, NcqVmState,
};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Uint32Equal;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use super::pci::VMSTATE_PCI_DEVICE;

/// `DRQ_STAT`.
const DRQ_STAT: u8 = 0x08;

/// `vmstate_ide_error_status`. `error_status` is 0 on AHCI here, see [`IdeBusVmState`].
static VMSTATE_IDE_ERROR_STATUS: LazyLock<VmStateDescription<IdeBusVmState>> =
    LazyLock::new(|| {
        type S = IdeBusVmState;
        VmStateDescription::new("ide_bus/error")
            .version_id(2)
            .minimum_version_id(1)
            .needed(|s: &S| s.error_status != 0)
            .fields([
                VmStateField::scalar("error_status", |s: &mut S| &mut s.error_status),
                VmStateField::scalar("retry_sector_num", |s: &mut S| &mut s.retry_sector_num)
                    .version(2),
                VmStateField::scalar("retry_nsector", |s: &mut S| &mut s.retry_nsector).version(2),
                VmStateField::scalar("retry_unit", |s: &mut S| &mut s.retry_unit).version(2),
            ])
    });

/// `vmstate_ide_bus`.
static VMSTATE_IDE_BUS: LazyLock<VmStateDescription<IdeBusVmState>> = LazyLock::new(|| {
    type S = IdeBusVmState;
    VmStateDescription::new("ide_bus")
        .version_id(1)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("cmd", |s: &mut S| &mut s.cmd),
            VmStateField::scalar("unit", |s: &mut S| &mut s.unit),
        ])
        .subsection(&VMSTATE_IDE_ERROR_STATUS)
});

/// `vmstate_ide_drive_pio_state`. QEMU also sends it while the bus has a PIO request waiting
/// to be retried, which never happens here. `ide_drive_pio_pre_save()` is in
/// [`IdeDriveVmState`]'s producer.
static VMSTATE_IDE_DRIVE_PIO_STATE: LazyLock<VmStateDescription<IdeDriveVmState>> =
    LazyLock::new(|| {
        type S = IdeDriveVmState;
        VmStateDescription::new("ide_drive/pio_state")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.status & DRQ_STAT != 0)
            .fields([
                VmStateField::scalar("req_nb_sectors", |s: &mut S| &mut s.req_nb_sectors),
                VmStateField::varray_alloc(
                    "io_buffer",
                    |_: &S| IDE_IO_BUFFER_TOTAL_LEN,
                    |s: &mut S| &mut s.io_buffer,
                )
                .version(1),
                VmStateField::scalar("cur_io_buffer_offset", |s: &mut S| {
                    &mut s.cur_io_buffer_offset
                }),
                VmStateField::scalar("cur_io_buffer_len", |s: &mut S| &mut s.cur_io_buffer_len),
                VmStateField::scalar("end_transfer_fn_idx", |s: &mut S| &mut s.end_transfer_fn_idx),
                VmStateField::scalar("elementary_transfer_size", |s: &mut S| {
                    &mut s.elementary_transfer_size
                }),
                VmStateField::scalar("packet_transfer_size", |s: &mut S| {
                    &mut s.packet_transfer_size
                }),
            ])
    });

/// `vmstate_ide_tray_state`.
static VMSTATE_IDE_TRAY_STATE: LazyLock<VmStateDescription<IdeDriveVmState>> =
    LazyLock::new(|| {
        type S = IdeDriveVmState;
        VmStateDescription::new("ide_drive/tray_state")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.tray_open || s.tray_locked)
            .fields([
                VmStateField::scalar("tray_open", |s: &mut S| &mut s.tray_open),
                VmStateField::scalar("tray_locked", |s: &mut S| &mut s.tray_locked),
            ])
    });

/// `vmstate_ide_atapi_gesn_state`.
static VMSTATE_IDE_ATAPI_GESN_STATE: LazyLock<VmStateDescription<IdeDriveVmState>> =
    LazyLock::new(|| {
        type S = IdeDriveVmState;
        VmStateDescription::new("ide_drive/atapi/gesn_state")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.new_media || s.eject_request)
            .fields([
                VmStateField::scalar("events.new_media", |s: &mut S| &mut s.new_media),
                VmStateField::scalar("events.eject_request", |s: &mut S| &mut s.eject_request),
            ])
    });

/// `vmstate_ide_drive`. `ide_drive_post_load()` is in the drive's load, see
/// [`Ich9Ahci::vmstate_load`].
static VMSTATE_IDE_DRIVE: LazyLock<VmStateDescription<IdeDriveVmState>> = LazyLock::new(|| {
    type S = IdeDriveVmState;
    VmStateDescription::new("ide_drive")
        .version_id(3)
        .minimum_version_id(0)
        .fields([
            VmStateField::scalar("mult_sectors", |s: &mut S| &mut s.mult_sectors),
            VmStateField::scalar("identify_set", |s: &mut S| &mut s.identify_set),
            VmStateField::partial_buffer("identify_data", 512, |s: &mut S| &mut s.identify_data)
                .test(|s: &S, _| s.identify_set != 0),
            VmStateField::scalar("feature", |s: &mut S| &mut s.feature),
            VmStateField::scalar("error", |s: &mut S| &mut s.error),
            VmStateField::scalar("nsector", |s: &mut S| &mut s.nsector),
            VmStateField::scalar("sector", |s: &mut S| &mut s.sector),
            VmStateField::scalar("lcyl", |s: &mut S| &mut s.lcyl),
            VmStateField::scalar("hcyl", |s: &mut S| &mut s.hcyl),
            VmStateField::scalar("hob_feature", |s: &mut S| &mut s.hob_feature),
            VmStateField::scalar("hob_sector", |s: &mut S| &mut s.hob_sector),
            VmStateField::scalar("hob_nsector", |s: &mut S| &mut s.hob_nsector),
            VmStateField::scalar("hob_lcyl", |s: &mut S| &mut s.hob_lcyl),
            VmStateField::scalar("hob_hcyl", |s: &mut S| &mut s.hob_hcyl),
            VmStateField::scalar("select", |s: &mut S| &mut s.select),
            VmStateField::scalar("status", |s: &mut S| &mut s.status),
            VmStateField::scalar("lba48", |s: &mut S| &mut s.lba48),
            VmStateField::scalar("sense_key", |s: &mut S| &mut s.sense_key),
            VmStateField::scalar("asc", |s: &mut S| &mut s.asc),
            VmStateField::scalar("cdrom_changed", |s: &mut S| &mut s.cdrom_changed).version(3),
        ])
        .subsection(&VMSTATE_IDE_DRIVE_PIO_STATE)
        .subsection(&VMSTATE_IDE_TRAY_STATE)
        .subsection(&VMSTATE_IDE_ATAPI_GESN_STATE)
});

/// `vmstate_ncq_tfs`.
static VMSTATE_NCQ_TFS: LazyLock<VmStateDescription<NcqVmState>> = LazyLock::new(|| {
    type S = NcqVmState;
    VmStateDescription::new("ncq state").version_id(1).fields([
        VmStateField::scalar("sector_count", |s: &mut S| &mut s.sector_count),
        VmStateField::scalar("lba", |s: &mut S| &mut s.lba),
        VmStateField::scalar("tag", |s: &mut S| &mut s.tag),
        VmStateField::scalar("cmd", |s: &mut S| &mut s.cmd),
        VmStateField::scalar("slot", |s: &mut S| &mut s.slot),
        VmStateField::scalar("used", |s: &mut S| &mut s.used),
        VmStateField::scalar("halt", |s: &mut S| &mut s.halt),
    ])
});

/// `vmstate_ahci_device`.
static VMSTATE_AHCI_DEVICE: LazyLock<VmStateDescription<AhciPortVmState>> = LazyLock::new(|| {
    type S = AhciPortVmState;
    VmStateDescription::new("ahci port").version_id(1).fields([
        VmStateField::structure("port", &VMSTATE_IDE_BUS, |s: &mut S| &mut s.port).version(1),
        VmStateField::structure("port.ifs[0]", &VMSTATE_IDE_DRIVE, |s: &mut S| &mut s.ifs0)
            .version(1),
        VmStateField::scalar("port_state", |s: &mut S| &mut s.port_state),
        VmStateField::scalar("finished", |s: &mut S| &mut s.finished),
        VmStateField::scalar("port_regs.lst_addr", |s: &mut S| &mut s.lst_addr),
        VmStateField::scalar("port_regs.lst_addr_hi", |s: &mut S| &mut s.lst_addr_hi),
        VmStateField::scalar("port_regs.fis_addr", |s: &mut S| &mut s.fis_addr),
        VmStateField::scalar("port_regs.fis_addr_hi", |s: &mut S| &mut s.fis_addr_hi),
        VmStateField::scalar("port_regs.irq_stat", |s: &mut S| &mut s.irq_stat),
        VmStateField::scalar("port_regs.irq_mask", |s: &mut S| &mut s.irq_mask),
        VmStateField::scalar("port_regs.cmd", |s: &mut S| &mut s.cmd),
        VmStateField::scalar("port_regs.tfdata", |s: &mut S| &mut s.tfdata),
        VmStateField::scalar("port_regs.sig", |s: &mut S| &mut s.sig),
        VmStateField::scalar("port_regs.scr_stat", |s: &mut S| &mut s.scr_stat),
        VmStateField::scalar("port_regs.scr_ctl", |s: &mut S| &mut s.scr_ctl),
        VmStateField::scalar("port_regs.scr_err", |s: &mut S| &mut s.scr_err),
        VmStateField::scalar("port_regs.scr_act", |s: &mut S| &mut s.scr_act),
        VmStateField::scalar("port_regs.cmd_issue", |s: &mut S| &mut s.cmd_issue),
        VmStateField::scalar("done_first_drq", |s: &mut S| &mut s.done_first_drq),
        VmStateField::scalar("busy_slot", |s: &mut S| &mut s.busy_slot),
        VmStateField::scalar("init_d2h_sent", |s: &mut S| &mut s.init_d2h_sent),
        VmStateField::struct_array("ncq_tfs", &VMSTATE_NCQ_TFS, |s: &mut S| &mut s.ncq_tfs)
            .version(1),
    ])
});

/// `vmstate_ahci`. `dev` has `ports` entries, the local count as in QEMU, and `ports` must
/// then match. `ahci_state_post_load()` is in [`Ich9Ahci::vmstate_load`].
static VMSTATE_AHCI: LazyLock<VmStateDescription<AhciVmState>> = LazyLock::new(|| {
    type S = AhciVmState;
    VmStateDescription::new("ahci").version_id(1).fields([
        VmStateField::struct_varray(
            "dev",
            |s: &S| s.ports as usize,
            &VMSTATE_AHCI_DEVICE,
            |s: &mut S| &mut s.dev,
        ),
        VmStateField::scalar("control_regs.cap", |s: &mut S| &mut s.cap),
        VmStateField::scalar("control_regs.ghc", |s: &mut S| &mut s.ghc),
        VmStateField::scalar("control_regs.irqstatus", |s: &mut S| &mut s.irqstatus),
        VmStateField::scalar("control_regs.impl", |s: &mut S| &mut s.impl_),
        VmStateField::scalar("control_regs.version", |s: &mut S| &mut s.version),
        VmStateField::scalar("idp_index", |s: &mut S| &mut s.idp_index),
        VmStateField::single("ports", &Uint32Equal, |s: &mut S| &mut s.ports),
    ])
});

/// `vmstate_ich9_ahci`.
pub(crate) static VMSTATE_ICH9_AHCI: LazyLock<VmStateDescription<Ich9AhciVmState>> =
    LazyLock::new(|| {
        type S = Ich9AhciVmState;
        VmStateDescription::new("ich9_ahci").version_id(1).fields([
            VmStateField::structure("parent_obj", &VMSTATE_PCI_DEVICE, |s: &mut S| {
                &mut s.parent_obj
            }),
            VmStateField::structure("ahci", &VMSTATE_AHCI, |s: &mut S| &mut s.ahci),
        ])
    });

/// Registers the `0000:00:1f.2/ich9_ahci` section of `ahci`, instance 0.
pub(crate) fn register(savevm: &mut SaveVm, ahci: &Arc<Ich9Ahci>) {
    let (get, put) = (Arc::clone(ahci), Arc::clone(ahci));
    savevm.register_vmsd(
        "0000:00:1f.2/",
        Some(0),
        &VMSTATE_ICH9_AHCI,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_hw_pci::PciDeviceVmState;
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    /// One `ahci port` without IDENTIFY data or subsections.
    const PORT_LEN: usize = 2 + 28 + 4 * 2 + 14 * 4 + 1 + 4 + 1 + 32 * 17;

    fn state() -> Ich9AhciVmState {
        let mut config = vec![0; 256];
        config[0] = 0x86;
        let dev = (0..6)
            .map(|i| AhciPortVmState {
                ifs0: IdeDriveVmState { status: 0x50, ..Default::default() },
                sig: 0xeb14_0101 - i,
                busy_slot: -1,
                ..Default::default()
            })
            .collect();
        Ich9AhciVmState {
            parent_obj: PciDeviceVmState { version_id: 2, config, irq_state: [0; 4] },
            ahci: AhciVmState {
                dev,
                cap: 0xc010_1f05,
                ghc: 0x8000_0002,
                impl_: 0x3f,
                version: 0x0001_0000,
                ports: 6,
                ..Default::default()
            },
        }
    }

    fn save(s: &mut Ich9AhciVmState) -> Vec<u8> {
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_ICH9_AHCI, s).unwrap();
        f.into_inner()
    }

    fn load(b: &[u8]) -> ruvm_base::Result<Ich9AhciVmState> {
        let mut back = state();
        for p in &mut back.ahci.dev {
            *p = AhciPortVmState::default();
        }
        // What follows the section in a stream: nested structures at the end peek past it.
        let b = [b, &[0]].concat();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ICH9_AHCI, &mut back, 1)?;
        Ok(back)
    }

    #[test]
    fn ich9_ahci_layout_matches_qemu() {
        assert_eq!(PORT_LEN, 644);
        let mut s = state();
        let b = save(&mut s);
        assert_eq!(b.len(), 276 + 6 * PORT_LEN + 7 * 4);
        assert_eq!(b.len(), 4168);
        // The first port's drive status, after mult_sectors, identify_set, feature, error,
        // nsector and the nine sector and select bytes.
        assert_eq!(b[276 + 2 + 4 + 4 + 1 + 1 + 4 + 9], 0x50);
        // busy_slot of the first port.
        let busy = 276 + 2 + 28 + 4 * 2 + 14 * 4 + 1;
        assert_eq!(&b[busy..busy + 4], &(-1i32).to_be_bytes());
        assert_eq!(&b[b.len() - 4..], &6u32.to_be_bytes());
        assert_eq!(load(&b).unwrap(), s);
    }

    #[test]
    fn ide_drive_identify_and_subsections() {
        let mut s = state();
        s.ahci.dev[0].ifs0.identify_set = 1;
        s.ahci.dev[0].ifs0.identify_data[0] = 0x40;
        s.ahci.dev[0].ifs0.identify_data[511] = 0xa5;
        let b = save(&mut s);
        assert_eq!(b.len(), 4168 + 512);
        assert_eq!(b[276 + 2 + 8], 0x40);
        assert_eq!(load(&b).unwrap(), s);

        // A locked tray and a pending PIO transfer bring their subsections.
        let tray = 2 + "ide_drive/tray_state".len() + 4 + 2;
        let pio = 2 + "ide_drive/pio_state".len() + 4 + 4 + IDE_IO_BUFFER_TOTAL_LEN + 4 * 4 + 1;
        let drive = &mut s.ahci.dev[1].ifs0;
        drive.tray_locked = true;
        drive.status |= DRQ_STAT;
        drive.io_buffer = vec![0x77; IDE_IO_BUFFER_TOTAL_LEN];
        drive.end_transfer_fn_idx = 2;
        let b = save(&mut s);
        assert_eq!(b.len(), 4168 + 512 + tray + pio);
        assert_eq!(load(&b).unwrap(), s);

        // The bus error subsection.
        s.ahci.dev[2].port.error_status = 0x20;
        s.ahci.dev[2].port.retry_sector_num = 99;
        let b = save(&mut s);
        let error = 2 + "ide_bus/error".len() + 4 + 4 + 8 + 4 + 1;
        assert_eq!(b.len(), 4168 + 512 + tray + pio + error);
        assert_eq!(load(&b).unwrap(), s);
    }

    #[test]
    fn ich9_ahci_port_count_must_match() {
        let mut s = state();
        let b = save(&mut s);
        let mut back = state();
        back.ahci.dev.truncate(4);
        back.ahci.ports = 4;
        let r = vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_ICH9_AHCI, &mut back, 1);
        assert!(r.is_err());
    }
}
