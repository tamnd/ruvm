// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-scsi through virtio-mmio. The command tests are ported from
//! `tests/qtest/virtio-scsi-test.c` (`hotplug`, `unaligned_write_same`, `unmap_large_lba`,
//! `write_to_cdrom` and the unit attention check of `qvirtio_scsi_pci_init`). The rest covers
//! task management, asynchronous notification, the event queue and the error paths of
//! `hw/scsi/virtio-scsi.c`.

mod common;

use std::sync::Arc;

use common::{Buf, Guest, SplitRing};
use ruvm_hw_storage::scsi::{ScsiDiskConf, ScsiSense};
use ruvm_hw_storage::{BlockBackend, VecBackend};
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_CONFIG, VIRTIO_MMIO_DEVICE_ID};
use ruvm_hw_virtio::scsi::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{VirtioDeviceClass, VirtioScsi, VirtioScsiConf};

const CDB_SIZE: usize = 32;
const SENSE_SIZE: usize = 96;
const REQ_SIZE: usize = 19 + CDB_SIZE;
const RESP_SIZE: usize = 12 + SENSE_SIZE;
const DISK_SIZE: usize = 1 << 20;

const GOOD: u8 = 0;
const CHECK_CONDITION: u8 = 2;
const UNIT_ATTENTION: u8 = 6;
const DATA_PROTECT: u8 = 7;
const ILLEGAL_REQUEST: u8 = 5;

fn features() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

fn hd(disk: &Arc<VecBackend>, id: u32, lun: u32) -> ScsiDiskConf {
    let drive: Arc<dyn BlockBackend> = disk.clone();
    ScsiDiskConf::hd(drive).at(id, lun)
}

/// A negotiated HBA with the control, event and first command queue ready and `disks` cold
/// plugged.
struct Hba {
    g: Guest,
    ctrl: SplitRing,
    event: SplitRing,
    cmd: SplitRing,
}

fn setup_with(conf: VirtioScsiConf, disks: Vec<ScsiDiskConf>) -> Hba {
    let mut scsi = VirtioScsi::new(conf);
    for d in disks {
        scsi.bus_mut().attach(d).unwrap();
    }
    let dev: Box<dyn VirtioDeviceClass> = Box::new(scsi);
    let mut g = Guest::new(false, Some(dev));
    g.negotiate(features());
    let ctrl = g.setup_queue(0, 0);
    let event = g.setup_queue(1, 0);
    let cmd = g.setup_queue(2, 0);
    g.driver_ok();
    Hba { g, ctrl, event, cmd }
}

fn setup(disk: &Arc<VecBackend>) -> Hba {
    setup_with(VirtioScsiConf::default(), vec![hd(disk, 0, 0)])
}

fn lun(id: u8, lun: u16) -> [u8; 8] {
    let mut l = [0u8; 8];
    l[0] = 1;
    l[1] = id;
    l[2] = 0x40 | (lun >> 8) as u8;
    l[3] = lun as u8;
    l
}

#[derive(Debug)]
struct Resp {
    sense_len: u32,
    resid: u32,
    status: u8,
    response: u8,
    sense: Vec<u8>,
    used_len: u32,
    data: Vec<u8>,
}

impl Hba {
    /// `virtio_scsi_do_command()`: a request, optional data out, the response and optional
    /// data in.
    fn command(&mut self, lun: [u8; 8], cdb: &[u8], data_out: &[u8], data_in: u32) -> Resp {
        let g = &mut self.g;
        let mut req = vec![0u8; REQ_SIZE];
        req[..8].copy_from_slice(&lun);
        req[19..19 + cdb.len()].copy_from_slice(cdb);
        let req_addr = g.alloc(REQ_SIZE as u64, 16);
        g.write_mem(req_addr, &req);
        let mut bufs = vec![Buf::out(req_addr, REQ_SIZE as u32)];
        if !data_out.is_empty() {
            let a = g.alloc(data_out.len() as u64, 16);
            g.write_mem(a, data_out);
            bufs.push(Buf::out(a, data_out.len() as u32));
        }
        let resp_addr = g.alloc(RESP_SIZE as u64, 16);
        let mut init = vec![0u8; RESP_SIZE];
        init[10] = 0xff;
        init[11] = 0xff;
        g.write_mem(resp_addr, &init);
        bufs.push(Buf::inp(resp_addr, RESP_SIZE as u32));
        let in_addr = if data_in > 0 {
            let a = g.alloc(u64::from(data_in), 16);
            g.write_mem(a, &vec![0; data_in as usize]);
            bufs.push(Buf::inp(a, data_in));
            a
        } else {
            0
        };
        self.cmd.submit(g, &bufs);
        g.kick(&self.cmd);
        let (_, used_len) = self.cmd.get_used(g).expect("command completes");
        let r = g.read_mem(resp_addr, RESP_SIZE);
        Resp {
            sense_len: u32::from_le_bytes(r[0..4].try_into().unwrap()),
            resid: u32::from_le_bytes(r[4..8].try_into().unwrap()),
            status: r[10],
            response: r[11],
            sense: r[12..].to_vec(),
            used_len,
            data: if data_in > 0 { g.read_mem(in_addr, data_in as usize) } else { Vec::new() },
        }
    }

    fn cmd0(&mut self, cdb: &[u8], data_out: &[u8], data_in: u32) -> Resp {
        self.command(lun(0, 0), cdb, data_out, data_in)
    }

    /// Clears the power on unit attention with TEST UNIT READY.
    fn clear_ua(&mut self) {
        let r = self.cmd0(&[0; 6], &[], 0);
        assert_eq!(r.status, CHECK_CONDITION);
        let r = self.cmd0(&[0; 6], &[], 0);
        assert_eq!(r.status, GOOD);
    }

    /// Sends a control queue request and returns the response bytes.
    fn ctrl(&mut self, req: &[u8], resp_len: u32) -> (u32, Vec<u8>) {
        let g = &mut self.g;
        let a = g.alloc(req.len() as u64, 16);
        g.write_mem(a, req);
        let r = g.alloc(u64::from(resp_len), 16);
        g.write_mem(r, &vec![0xee; resp_len as usize]);
        self.ctrl.submit(g, &[Buf::out(a, req.len() as u32), Buf::inp(r, resp_len)]);
        g.kick(&self.ctrl);
        let (_, len) = self.ctrl.get_used(g).expect("control request completes");
        (len, g.read_mem(r, resp_len as usize))
    }

    fn tmf(&mut self, subtype: u32, lun: [u8; 8]) -> u8 {
        let mut req = vec![0u8; 24];
        req[0..4].copy_from_slice(&VIRTIO_SCSI_T_TMF.to_le_bytes());
        req[4..8].copy_from_slice(&subtype.to_le_bytes());
        req[8..16].copy_from_slice(&lun);
        let (len, resp) = self.ctrl(&req, 1);
        assert_eq!(len, 1);
        resp[0]
    }

    /// Posts one event buffer of 16 bytes and returns its address.
    fn post_event(&mut self) -> u64 {
        let a = self.g.alloc(16, 16);
        self.g.write_mem(a, &[0xee; 16]);
        self.event.submit(&self.g, &[Buf::inp(a, 16)]);
        a
    }

    fn event(&mut self, addr: u64) -> Option<(u32, [u8; 8], u32)> {
        let (_, len) = self.event.get_used(&self.g)?;
        assert_eq!(len, 16);
        let e = self.g.read_mem(addr, 16);
        Some((
            u32::from_le_bytes(e[0..4].try_into().unwrap()),
            e[4..12].try_into().unwrap(),
            u32::from_le_bytes(e[12..16].try_into().unwrap()),
        ))
    }

    fn with<R>(&self, f: impl FnOnce(&mut VirtIODevice, &mut VirtioScsi) -> R) -> R {
        self.g.mmio.with_device(f).unwrap()
    }
}

#[test]
fn config_space() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let h = setup(&disk);
    let g = &h.g;
    assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), u32::from(VIRTIO_ID_SCSI));
    assert_eq!(g.config_readl(0), 1, "num_queues");
    assert_eq!(g.config_readl(4), 254, "seg_max");
    assert_eq!(g.config_readl(8), 0xffff, "max_sectors");
    assert_eq!(g.config_readl(12), 128, "cmd_per_lun");
    assert_eq!(g.config_readl(16), 16, "event_info_size");
    assert_eq!(g.config_readl(20), 96, "sense_size");
    assert_eq!(g.config_readl(24), 32, "cdb_size");
    assert_eq!(g.config_readw(28), 0, "max_channel");
    assert_eq!(g.config_readw(30), 255, "max_target");
    assert_eq!(g.config_readl(32), 16383, "max_lun");
    let f = g.device_features();
    assert_ne!(f & feature(VIRTIO_SCSI_F_HOTPLUG), 0);
    assert_ne!(f & feature(VIRTIO_SCSI_F_CHANGE), 0);
    assert_eq!(f & feature(VIRTIO_SCSI_F_T10_PI), 0);
}

#[test]
fn queue_count_and_features_follow_properties() {
    let conf = VirtioScsiConf {
        num_queues: Some(4),
        virtqueue_size: 128,
        seg_max_adjust: false,
        hotplug: false,
        param_change: false,
        ..VirtioScsiConf::default()
    };
    let h = setup_with(conf, Vec::new());
    assert_eq!(h.g.config_readl(0), 4);
    assert_eq!(h.g.config_readl(4), 126);
    let f = h.g.device_features();
    assert_eq!(f & (feature(VIRTIO_SCSI_F_HOTPLUG) | feature(VIRTIO_SCSI_F_CHANGE)), 0);
    h.with(|vdev, d| {
        assert_eq!(vdev.num_queues(), 6);
        assert_eq!(vdev.queue_num_max(5), 128);
        assert_eq!(d.conf().num_queues, Some(4));
    });
}

#[test]
fn realize_errors() {
    let bad = |conf: VirtioScsiConf| {
        let dev: Box<dyn VirtioDeviceClass> = Box::new(VirtioScsi::new(conf));
        match Guest::try_new(false, Some(dev)) {
            Ok(_) => panic!("realize should fail"),
            Err(e) => e.to_string(),
        }
    };
    assert_eq!(
        bad(VirtioScsiConf { num_queues: Some(0), ..VirtioScsiConf::default() }),
        "Invalid number of queues (= 0), must be a positive integer less than 1022."
    );
    assert_eq!(
        bad(VirtioScsiConf { num_queues: Some(1023), ..VirtioScsiConf::default() }),
        "Invalid number of queues (= 1023), must be a positive integer less than 1022."
    );
    assert_eq!(
        bad(VirtioScsiConf { virtqueue_size: 2, ..VirtioScsiConf::default() }),
        "invalid virtqueue_size property (= 2), must be > 2"
    );
}

#[test]
fn unit_attention_after_reset() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_OK);
    assert_eq!(r.status, CHECK_CONDITION);
    assert_eq!(r.sense_len, 18);
    assert_eq!(r.sense[0], 0x70);
    assert_eq!(r.sense[2], UNIT_ATTENTION);
    assert_eq!(r.sense[12], 0x29);
    assert_eq!(r.sense[13], 0x00);
    assert_eq!(r.used_len, RESP_SIZE as u32);
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.sense_len, 0);
}

#[test]
fn inquiry_read_write() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    h.clear_ua();

    let r = h.cmd0(&[0x12, 0, 0, 0, 96, 0], &[], 96);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.data[0], 0, "TYPE_DISK");
    assert_eq!(&r.data[8..16], b"QEMU    ");
    assert_eq!(r.resid, 0);

    // WRITE(10) two blocks at LBA 3, then READ(10) them back.
    let data: Vec<u8> = (0..1024).map(|i| (i % 251) as u8).collect();
    let r = h.cmd0(&[0x2a, 0, 0, 0, 0, 3, 0, 0, 2, 0], &data, 0);
    assert_eq!((r.response, r.status), (VIRTIO_SCSI_S_OK, GOOD));
    assert_eq!(&disk.contents()[3 * 512..5 * 512], &data[..]);
    let r = h.cmd0(&[0x28, 0, 0, 0, 0, 3, 0, 0, 2, 0], &[], 1024);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.data, data);
    assert_eq!(r.used_len, (RESP_SIZE + 1024) as u32);

    // A data in buffer larger than the transfer leaves a residual.
    let r = h.cmd0(&[0x28, 0, 0, 0, 0, 3, 0, 0, 1, 0], &[], 1024);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.resid, 512);
    assert_eq!(&r.data[..512], &data[..512]);

    // READ CAPACITY(10).
    let r = h.cmd0(&[0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[], 8);
    assert_eq!(r.status, GOOD);
    let last = u32::from_be_bytes(r.data[0..4].try_into().unwrap());
    assert_eq!(u64::from(last) + 1, (DISK_SIZE / 512) as u64);
    assert_eq!(u32::from_be_bytes(r.data[4..8].try_into().unwrap()), 512);
}

#[test]
fn bad_target_and_lun() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    let r = h.command(lun(5, 0), &[0; 6], &[], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_BAD_TARGET);
    let mut l = lun(0, 0);
    l[0] = 2;
    let r = h.command(l, &[0; 6], &[], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_BAD_TARGET);
    // A LUN that is not there on a target that is answers through LUN 0 with ILLEGAL
    // REQUEST, LOGICAL UNIT NOT SUPPORTED... except for INQUIRY and REPORT LUNS.
    h.clear_ua();
    let r = h.command(lun(0, 3), &[0; 6], &[], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_OK);
    assert_eq!(r.status, CHECK_CONDITION);
    assert_eq!(r.sense[2], ILLEGAL_REQUEST);
    assert_eq!(r.sense[12], 0x25);
}

#[test]
fn overrun_when_buffers_are_too_small() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    h.clear_ua();
    let r = h.cmd0(&[0x28, 0, 0, 0, 0, 0, 0, 0, 2, 0], &[], 512);
    assert_eq!(r.response, VIRTIO_SCSI_S_OVERRUN);
    // Data out where the command wants data in.
    let r = h.cmd0(&[0x28, 0, 0, 0, 0, 0, 0, 0, 1, 0], &[0; 512], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_OVERRUN);
}

/// `test_unaligned_write_same`.
#[test]
fn unaligned_write_same() {
    let disk = Arc::new(VecBackend::new(64 << 20));
    let mut h = setup(&disk);
    h.clear_ua();
    let buf1 = [0u8; 512];
    let mut buf2 = [0u8; 512];
    buf2[..5].copy_from_slice(b"\x01\x02\x03\x04\x05");
    let r = h.cmd0(&[0x41, 0, 0, 0, 0, 1, 0, 0, 2, 0], &buf1, 0);
    assert_eq!(r.response, 0);
    let r = h.cmd0(&[0x41, 0, 0, 0, 0, 1, 0, 0x33, 0, 0], &buf2, 0);
    assert_eq!(r.response, 0);
    assert_eq!(r.status, GOOD);
    let r = h.cmd0(&[0x41, 1, 0, 0, 0, 1, 0, 0x33, 0, 0], &[], 0);
    assert_eq!(r.response, 0);
    assert_eq!(r.status, GOOD);
}

/// `test_unmap_large_lba`.
#[test]
fn unmap_large_lba() {
    // null-co's 1 GiB with 4 KiB blocks, at scsi-id 1. The zeroed image is not touched
    // except where the unmap writes.
    let disk = Arc::new(VecBackend::new(1 << 30));
    let mut conf = hd(&disk, 1, 0);
    conf.logical_block_size = 4096;
    conf.physical_block_size = 4096;
    let mut h = setup_with(VirtioScsiConf::default(), vec![conf]);
    let target = lun(1, 0);
    let r = h.command(target, &[0; 6], &[], 0);
    assert_eq!(r.status, CHECK_CONDITION);
    let params: [u8; 0x18] = [
        0x00, 0x16, 0x00, 0x10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x7f, 0xff, 0x00, 0x00, 0x03, 0xff,
        0, 0, 0, 0,
    ];
    let r = h.command(target, &[0x42, 0, 0, 0, 0, 0, 0, 0, 0x18, 0], &params, 0);
    assert_eq!(r.response, 0);
    assert_ne!(r.status, CHECK_CONDITION);
}

/// `test_write_to_cdrom`.
#[test]
fn write_to_cdrom() {
    let iso = Arc::new(VecBackend::new(1 << 20));
    let drive: Arc<dyn BlockBackend> = iso.clone();
    let mut h = setup_with(VirtioScsiConf::default(), vec![ScsiDiskConf::cd(Some(drive))]);
    // Power on and medium change unit attentions first.
    for _ in 0..4 {
        if h.cmd0(&[0; 6], &[], 0).status == GOOD {
            break;
        }
    }
    let r = h.cmd0(&[0x2a, 0, 0, 0, 0, 0, 0, 0, 1, 0], &[0; 2048], 0);
    assert_eq!(r.response, 0);
    assert_eq!(r.status, CHECK_CONDITION);
    assert_eq!(r.sense[0], 0x70);
    assert_eq!(r.sense[2], DATA_PROTECT);
    assert_eq!(r.sense[12], 0x27);
    assert_eq!(r.sense[13], 0x00);
}

#[test]
fn sense_size_limits_sense_copied() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    // Like a driver, read the config space before writing it: writes go into the copy the
    // last read left, as in QEMU.
    assert_eq!(h.g.config_readl(24), 32);
    h.g.writel(VIRTIO_MMIO_CONFIG + 20, 8);
    assert_eq!(h.g.config_readl(20), 8);
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!(r.status, CHECK_CONDITION);
    assert_eq!(r.sense_len, 8);
    assert_eq!(r.sense[2], UNIT_ATTENTION);
    // Sense bytes past sense_size stay as they were.
    assert_eq!(r.sense[12], 0);
    // A device reset puts the default back.
    h.g.set_status(0);
    assert_eq!(h.g.config_readl(20), 96);
}

#[test]
fn bad_config_write_breaks_the_device() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let h = setup(&disk);
    h.g.writel(VIRTIO_MMIO_CONFIG + 24, 256);
    assert_ne!(h.g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
    assert_eq!(h.g.config_readl(24), 32);
}

#[test]
fn short_request_header_breaks_the_device() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    let a = h.g.alloc(16, 16);
    let r = h.g.alloc(RESP_SIZE as u64, 16);
    h.cmd.submit(&h.g, &[Buf::out(a, 16), Buf::inp(r, RESP_SIZE as u32)]);
    h.g.kick(&h.cmd);
    assert_ne!(h.g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
}

#[test]
fn task_management() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    h.clear_ua();
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_ABORT_TASK, lun(0, 0)), VIRTIO_SCSI_S_OK);
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_ABORT_TASK_SET, lun(0, 0)), VIRTIO_SCSI_S_OK);
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_QUERY_TASK_SET, lun(0, 0)), VIRTIO_SCSI_S_OK);
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_ABORT_TASK, lun(9, 0)), VIRTIO_SCSI_S_BAD_TARGET);
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_ABORT_TASK, lun(0, 7)), VIRTIO_SCSI_S_INCORRECT_LUN);
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_CLEAR_ACA, lun(0, 0)), VIRTIO_SCSI_S_FUNCTION_REJECTED);
    assert_eq!(h.tmf(99, lun(0, 0)), VIRTIO_SCSI_S_FUNCTION_REJECTED);

    // A LUN reset raises the RESET unit attention.
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET, lun(0, 0)), VIRTIO_SCSI_S_OK);
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!((r.status, r.sense[2], r.sense[12]), (CHECK_CONDITION, UNIT_ATTENTION, 0x29));
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!(r.status, GOOD);

    // So does an I_T nexus reset, even through a LUN that does not exist.
    assert_eq!(h.tmf(VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET, lun(0, 5)), VIRTIO_SCSI_S_OK);
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!((r.status, r.sense[12]), (CHECK_CONDITION, 0x29));
}

#[test]
fn asynchronous_notification_and_unknown_control_requests() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    for ty in [VIRTIO_SCSI_T_AN_QUERY, VIRTIO_SCSI_T_AN_SUBSCRIBE] {
        let mut req = vec![0u8; 16];
        req[0..4].copy_from_slice(&ty.to_le_bytes());
        req[4..12].copy_from_slice(&lun(0, 0));
        req[12..16].copy_from_slice(&0xffu32.to_le_bytes());
        let (len, resp) = h.ctrl(&req, 5);
        assert_eq!(len, 5);
        assert_eq!(resp, [0, 0, 0, 0, VIRTIO_SCSI_S_OK]);
    }
    let (len, resp) = h.ctrl(&7u32.to_le_bytes(), 4);
    assert_eq!(len, 0);
    assert_eq!(resp, [0xee; 4]);
    // A TMF whose response does not fit breaks the device.
    let mut req = vec![0u8; 24];
    req[4..8].copy_from_slice(&VIRTIO_SCSI_T_TMF_ABORT_TASK.to_le_bytes());
    let a = h.g.alloc(24, 16);
    h.g.write_mem(a, &req);
    h.ctrl.submit(&h.g, &[Buf::out(a, 24)]);
    h.g.kick(&h.ctrl);
    assert_ne!(h.g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
}

/// `hotplug` from the qtest, plus the events the guest sees.
#[test]
fn hotplug_and_unplug_events() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    h.clear_ua();
    let ev = h.post_event();
    h.g.kick(&h.event);
    let disk2 = Arc::new(VecBackend::new(DISK_SIZE));
    let conf = hd(&disk2, 1, 300);
    let idx = h.with(|vdev, d| d.hotplug(vdev, conf)).unwrap();
    assert!(h.g.irq());
    let (event, l, reason) = h.event(ev).expect("rescan event");
    assert_eq!(event, VIRTIO_SCSI_T_TRANSPORT_RESET);
    assert_eq!(reason, VIRTIO_SCSI_EVT_RESET_RESCAN);
    assert_eq!(l, [1, 1, 0x41, 44, 0, 0, 0, 0]);
    h.with(|_, d| assert_eq!(d.bus().devices()[idx].address(), (0, 1, 300)));

    // The existing disk reports REPORTED LUNS DATA HAS CHANGED.
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!((r.status, r.sense[2], r.sense[12], r.sense[13]), (CHECK_CONDITION, 6, 0x3f, 0x0e));

    let ev = h.post_event();
    h.g.kick(&h.event);
    let gone = h.with(|vdev, d| d.hot_unplug(vdev, 0, 1, 300));
    assert!(gone.is_some());
    let (event, l, reason) = h.event(ev).expect("removed event");
    assert_eq!(event, VIRTIO_SCSI_T_TRANSPORT_RESET);
    assert_eq!(reason, VIRTIO_SCSI_EVT_RESET_REMOVED);
    assert_eq!(&l[..4], &[1, 1, 0x41, 44]);
    let r = h.command(lun(1, 300), &[0; 6], &[], 0);
    assert_eq!(r.response, VIRTIO_SCSI_S_BAD_TARGET);
    assert!(h.with(|vdev, d| d.hot_unplug(vdev, 0, 1, 300)).is_none());
}

#[test]
fn dropped_events_are_reported() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    let conf = hd(&disk, 2, 0);
    h.with(|vdev, d| d.hotplug(vdev, conf)).unwrap();
    assert!(h.with(|_, d| d.events_dropped()));
    // The next buffer the driver adds gets VIRTIO_SCSI_T_NO_EVENT with the missed flag.
    let ev = h.post_event();
    h.g.kick(&h.event);
    let (event, l, reason) = h.event(ev).expect("missed event");
    assert_eq!(event, VIRTIO_SCSI_T_NO_EVENT | VIRTIO_SCSI_T_EVENTS_MISSED);
    assert_eq!(l, [0; 8]);
    assert_eq!(reason, 0);
    assert!(!h.with(|_, d| d.events_dropped()));
    // Kicking the event queue with nothing dropped leaves new buffers alone.
    let _ = h.post_event();
    h.g.kick(&h.event);
    assert!(h.event.get_used(&h.g).is_none());
}

#[test]
fn no_events_before_driver_ok_or_without_the_feature() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let dev: Box<dyn VirtioDeviceClass> = Box::new(VirtioScsi::new(VirtioScsiConf::default()));
    let mut g = Guest::new(false, Some(dev));
    g.negotiate(features() & !feature(VIRTIO_SCSI_F_HOTPLUG));
    let _ = g.setup_queue(0, 0);
    let mut event = g.setup_queue(1, 0);
    g.driver_ok();
    let a = g.alloc(16, 16);
    event.submit(&g, &[Buf::inp(a, 16)]);
    g.kick(&event);
    let conf = hd(&disk, 0, 0);
    g.mmio.with_device(|vdev, d: &mut VirtioScsi| d.hotplug(vdev, conf)).unwrap().unwrap();
    assert!(event.get_used(&g).is_none());
}

#[test]
fn parameter_change_event() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut h = setup(&disk);
    h.clear_ua();
    let ev = h.post_event();
    h.g.kick(&h.event);
    h.with(|vdev, d| d.report_change(vdev, 0, ScsiSense::CAPACITY_CHANGED));
    let (event, l, reason) = h.event(ev).expect("param change event");
    assert_eq!(event, VIRTIO_SCSI_T_PARAM_CHANGE);
    assert_eq!(&l[..4], &[1, 0, 0, 0]);
    assert_eq!(reason, 0x2a | (0x09 << 8));
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!((r.status, r.sense[12], r.sense[13]), (CHECK_CONDITION, 0x2a, 0x09));
}

#[test]
fn legacy_transport_works() {
    let disk = Arc::new(VecBackend::new(DISK_SIZE));
    let mut scsi = VirtioScsi::new(VirtioScsiConf::default());
    scsi.bus_mut().attach(hd(&disk, 0, 0)).unwrap();
    let dev: Box<dyn VirtioDeviceClass> = Box::new(scsi);
    let mut g = Guest::new(true, Some(dev));
    g.negotiate(features());
    let ctrl = g.setup_queue(0, 0);
    let event = g.setup_queue(1, 0);
    let cmd = g.setup_queue(2, 0);
    g.driver_ok();
    let mut h = Hba { g, ctrl, event, cmd };
    let r = h.cmd0(&[0; 6], &[], 0);
    assert_eq!(r.status, CHECK_CONDITION);
    let r = h.cmd0(&[0x12, 0, 0, 0, 36, 0], &[], 36);
    assert_eq!(r.status, GOOD);
    assert_eq!(&r.data[8..16], b"QEMU    ");
}
