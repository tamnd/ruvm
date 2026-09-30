// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the SCSI bus with `scsi-hd` and `scsi-cd`: the commands, sense data and unit
//! attention, following QEMU's scsi-disk emulation and the cases in `tests/qtest/virtio-scsi-test.c`.

#![forbid(unsafe_code)]

use std::io;
use std::sync::Arc;

use ruvm_hw_storage::scsi::*;
use ruvm_hw_storage::{BlockBackend, VecBackend};

const INFO: ScsiBusInfo =
    ScsiBusInfo { tcq: true, max_channel: 0, max_target: 255, max_lun: 16383 };

/// A backend that reads zeros and throws writes away, like QEMU's `null-co` driver.
#[derive(Debug)]
struct NullBackend(u64);

impl BlockBackend for NullBackend {
    fn read_at(&self, _offset: u64, buf: &mut [u8]) -> io::Result<()> {
        buf.fill(0);
        Ok(())
    }

    fn write_at(&self, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    fn len(&self) -> u64 {
        self.0
    }
}

struct Reply {
    status: u8,
    data: Vec<u8>,
    sense: Option<ScsiSense>,
    residual: u64,
}

fn run_lun(bus: &mut ScsiBus, id: u32, lun: u32, cdb: &[u8], out: &[u8]) -> Reply {
    let dev = bus.find(0, id, lun).expect("no device on the target");
    let mut req = bus.new_request(dev, lun, cdb);
    let data = bus.execute(&mut req, out);
    let sense = bus.get_sense(&req, SCSI_SENSE_LEN);
    Reply {
        status: req.status().unwrap(),
        data,
        sense: if sense.is_empty() { None } else { Some(ScsiSense::from_buf(&sense)) },
        residual: req.residual(),
    }
}

fn run(bus: &mut ScsiBus, cdb: &[u8], out: &[u8]) -> Reply {
    run_lun(bus, 0, 0, cdb, out)
}

fn good(bus: &mut ScsiBus, cdb: &[u8], out: &[u8]) -> Vec<u8> {
    let r = run(bus, cdb, out);
    assert_eq!(r.status, GOOD, "cdb {cdb:02x?} failed with {:?}", r.sense);
    r.data
}

fn check(bus: &mut ScsiBus, cdb: &[u8], out: &[u8], sense: ScsiSense) {
    let r = run(bus, cdb, out);
    assert_eq!(r.status, CHECK_CONDITION, "cdb {cdb:02x?}");
    assert_eq!(r.sense, Some(sense), "cdb {cdb:02x?}");
}

/// A bus with a scsi-hd at 0:0 on a backend of `sectors` 512 byte sectors, with the power on
/// unit attention already consumed.
fn hd_bus(sectors: usize) -> (ScsiBus, Arc<VecBackend>) {
    let img = Arc::new(VecBackend::new(sectors * 512));
    let mut bus = ScsiBus::new(INFO);
    let mut conf = ScsiDiskConf::hd(img.clone()).at(0, 0);
    conf.drive_name = Some("drive0".into());
    bus.attach(conf).unwrap();
    check(&mut bus, &[0; 6], &[], ScsiSense::RESET);
    (bus, img)
}

#[test]
fn power_on_unit_attention_then_ready() {
    let img = Arc::new(VecBackend::new(1 << 20));
    let mut bus = ScsiBus::new(INFO);
    bus.attach(ScsiDiskConf::hd(img).at(0, 0)).unwrap();
    // INQUIRY does not report the condition, TEST UNIT READY does, once.
    good(&mut bus, &[INQUIRY_OP, 0, 0, 0, 36, 0], &[]);
    let r = run(&mut bus, &[0; 6], &[]);
    assert_eq!(r.status, CHECK_CONDITION);
    let s = r.sense.unwrap();
    assert_eq!((s.key, s.asc, s.ascq), (6, 0x29, 0));
    good(&mut bus, &[0; 6], &[]);
}

const INQUIRY_OP: u8 = opcode::INQUIRY;

#[test]
fn unit_attention_through_request_sense() {
    let img = Arc::new(VecBackend::new(1 << 20));
    let mut bus = ScsiBus::new(INFO);
    let dev = bus.attach(ScsiDiskConf::hd(img).at(0, 0)).unwrap();
    // Without autosense, the unit attention stays as the device's sense.
    let mut req = bus.new_request(dev, 0, &[0; 6]);
    bus.execute(&mut req, &[]);
    assert_eq!(req.status(), Some(CHECK_CONDITION));
    // REQUEST SENSE returns it instead of reporting it again.
    let r = run(&mut bus, &[opcode::REQUEST_SENSE, 0, 0, 0, 18, 0], &[]);
    assert_eq!(r.status, GOOD);
    assert_eq!(ScsiSense::from_buf(&r.data), ScsiSense::RESET);
    let r = run(&mut bus, &[opcode::REQUEST_SENSE, 0, 0, 0, 18, 0], &[]);
    assert_eq!(ScsiSense::from_buf(&r.data), ScsiSense::NO_SENSE);
    good(&mut bus, &[0; 6], &[]);
}

#[test]
fn request_sense_returns_last_error() {
    let (mut bus, _) = hd_bus(64);
    let dev = bus.find(0, 0, 0).unwrap();
    let mut req = bus.new_request(dev, 0, &[opcode::READ_10, 0, 0, 0, 1, 0, 0, 0, 1, 0]);
    bus.execute(&mut req, &[]);
    assert_eq!(req.sense(), Some(ScsiSense::LBA_OUT_OF_RANGE));
    // Descriptor format on request.
    let r = run(&mut bus, &[opcode::REQUEST_SENSE, 1, 0, 0, 252, 0], &[]);
    assert_eq!(r.data, vec![0x72, 5, 0x21, 0, 0, 0, 0, 0]);
    let r = run(&mut bus, &[opcode::REQUEST_SENSE, 0, 0, 0, 252, 0], &[]);
    // The device answers this one and, like every emulated command, returns the whole
    // allocation length.
    assert_eq!(r.data.len(), 252);
    assert_eq!(ScsiSense::from_buf(&r.data), ScsiSense::NO_SENSE);
}

#[test]
fn standard_inquiry() {
    let (mut bus, _) = hd_bus(64);
    let d = good(&mut bus, &[INQUIRY_OP, 0, 0, 0, 96, 0], &[]);
    assert_eq!(d.len(), 96);
    assert_eq!(d[0], TYPE_DISK);
    assert_eq!(d[1], 0);
    assert_eq!(d[2], 5);
    assert_eq!(d[3], 0x12);
    assert_eq!(d[4], 91);
    assert_eq!(d[7], 0x12);
    assert_eq!(&d[8..16], b"QEMU    ");
    assert_eq!(&d[16..32], b"QEMU HARDDISK   ");
    assert_eq!(&d[32..36], b"2.5+");
    let d = good(&mut bus, &[INQUIRY_OP, 0, 0, 0, 36, 0], &[]);
    assert_eq!(d[4], 31);
    check(&mut bus, &[INQUIRY_OP, 0, 0x80, 0, 36, 0], &[], ScsiSense::INVALID_FIELD);
}

#[test]
fn vpd_pages() {
    let img = Arc::new(VecBackend::new(1 << 20));
    let mut bus = ScsiBus::new(INFO);
    let mut conf = ScsiDiskConf::hd(img).at(0, 0);
    conf.serial = Some("SER123".into());
    conf.wwn = 0x5000_1234_5678_9abc;
    conf.rotation_rate = 1;
    bus.attach(conf).unwrap();
    check(&mut bus, &[0; 6], &[], ScsiSense::RESET);

    let d = good(&mut bus, &[INQUIRY_OP, 1, 0, 0, 255, 0], &[]);
    assert_eq!(&d[..10], &[0, 0, 0, 6, 0x00, 0x80, 0x83, 0xb0, 0xb1, 0xb2]);
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0x80, 0, 255, 0], &[]);
    assert_eq!(&d[..10], b"\x00\x80\x00\x06SER123");
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0x83, 0, 255, 0], &[]);
    assert_eq!(d[3], 10 + 12);
    assert_eq!(&d[4..14], b"\x02\x00\x00\x06SER123");
    assert_eq!(&d[14..18], &[1, 3, 0, 8]);
    assert_eq!(&d[18..26], &0x5000_1234_5678_9abcu64.to_be_bytes());

    let d = good(&mut bus, &[INQUIRY_OP, 1, 0xb0, 0, 255, 0], &[]);
    assert_eq!(d[3], 0x3c);
    assert_eq!(d[4], 1);
    let be32 = |o: usize| u32::from_be_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
    assert_eq!(be32(8), i32::MAX as u32 / 512);
    assert_eq!(be32(20), (1 << 30) / 512);
    assert_eq!(be32(24), 255);
    assert_eq!(be32(28), 4096 / 512);
    assert_eq!(be32(40), i32::MAX as u32 / 512);

    let d = good(&mut bus, &[INQUIRY_OP, 1, 0xb1, 0, 255, 0], &[]);
    assert_eq!(d.len(), 255);
    assert_eq!(&d[..6], &[0, 0xb1, 0, 0x3c, 0, 1]);
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0xb2, 0, 255, 0], &[]);
    assert_eq!(&d[..8], &[0, 0xb2, 0, 4, 0, 0xe0, 2, 0]);
    check(&mut bus, &[INQUIRY_OP, 1, 0x42, 0, 255, 0], &[], ScsiSense::INVALID_FIELD);
}

#[test]
fn device_id_defaults_to_the_drive_name() {
    let (mut bus, _) = hd_bus(64);
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0, 0, 255, 0], &[]);
    // No serial, so no page 0x80.
    assert_eq!(&d[..9], &[0, 0, 0, 5, 0x00, 0x83, 0xb0, 0xb1, 0xb2]);
    check(&mut bus, &[INQUIRY_OP, 1, 0x80, 0, 255, 0], &[], ScsiSense::INVALID_FIELD);
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0x83, 0, 255, 0], &[]);
    assert_eq!(&d[..14], b"\x00\x83\x00\x0a\x02\x00\x00\x06drive0");
}

#[test]
fn read_capacity() {
    let (mut bus, _) = hd_bus(2048);
    let d = good(&mut bus, &[opcode::READ_CAPACITY_10, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[]);
    assert_eq!(d, vec![0, 0, 7, 0xff, 0, 0, 2, 0]);
    let mut cdb = [0u8; 16];
    cdb[0] = opcode::SERVICE_ACTION_IN_16;
    cdb[1] = opcode::SAI_READ_CAPACITY_16;
    cdb[13] = 32;
    let d = good(&mut bus, &cdb, &[]);
    assert_eq!(d.len(), 32);
    assert_eq!(&d[..16], &[0, 0, 0, 0, 0, 0, 7, 0xff, 0, 0, 2, 0, 0, 0, 0x80, 0]);
    // A nonzero LBA without PMI is an error.
    check(
        &mut bus,
        &[opcode::READ_CAPACITY_10, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        &[],
        ScsiSense::INVALID_FIELD,
    );
}

#[test]
fn read_and_write() {
    let (mut bus, img) = hd_bus(64);
    let data: Vec<u8> = (0..1024).map(|i| i as u8).collect();
    good(&mut bus, &[opcode::WRITE_10, 0, 0, 0, 0, 3, 0, 0, 2, 0], &data);
    assert_eq!(&img.contents()[3 * 512..5 * 512], &data[..]);
    let d = good(&mut bus, &[opcode::READ_10, 0, 0, 0, 0, 3, 0, 0, 2, 0], &[]);
    assert_eq!(d, data);
    let d = good(&mut bus, &[opcode::READ_6, 0, 0, 3, 1, 0], &[]);
    assert_eq!(d, data[..512]);
    let mut cdb = [0u8; 16];
    cdb[0] = opcode::READ_16;
    cdb[9] = 4;
    cdb[13] = 1;
    assert_eq!(good(&mut bus, &cdb, &[]), data[512..]);
    // READ(6) of 0 blocks is 256 blocks, past the end of this disk.
    check(&mut bus, &[opcode::READ_6, 0, 0, 0, 0, 0], &[], ScsiSense::LBA_OUT_OF_RANGE);
    check(
        &mut bus,
        &[opcode::READ_10, 0, 0, 0, 0, 63, 0, 0, 2, 0],
        &[],
        ScsiSense::LBA_OUT_OF_RANGE,
    );
    // Protection information is refused.
    check(
        &mut bus,
        &[opcode::READ_10, 0x20, 0, 0, 0, 0, 0, 0, 1, 0],
        &[],
        ScsiSense::INVALID_FIELD,
    );
    // A zero length transfer is fine.
    let r = run(&mut bus, &[opcode::READ_10, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[]);
    assert_eq!((r.status, r.data.len()), (GOOD, 0));
}

#[test]
fn fua_and_synchronize_cache_flush() {
    let (mut bus, img) = hd_bus(64);
    let before = img.flush_count();
    good(&mut bus, &[opcode::WRITE_10, 0, 0, 0, 0, 0, 0, 0, 1, 0], &[1; 512]);
    assert_eq!(img.flush_count(), before);
    good(&mut bus, &[opcode::WRITE_10, 8, 0, 0, 0, 0, 0, 0, 1, 0], &[1; 512]);
    assert_eq!(img.flush_count(), before + 1);
    good(&mut bus, &[opcode::SYNCHRONIZE_CACHE, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[]);
    assert_eq!(img.flush_count(), before + 2);
}

#[test]
fn backend_errors_and_write_protect() {
    let (mut bus, img) = hd_bus(64);
    img.set_failing(true);
    check(&mut bus, &[opcode::READ_10, 0, 0, 0, 0, 0, 0, 0, 1, 0], &[], ScsiSense::IO_ERROR);
    img.set_failing(false);
    img.set_read_only(true);
    check(
        &mut bus,
        &[opcode::WRITE_10, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        &[0; 512],
        ScsiSense::WRITE_PROTECTED,
    );
    // MODE SENSE reports the WP bit.
    let d = good(&mut bus, &[opcode::MODE_SENSE, 0, 0x08, 0, 255, 0], &[]);
    assert_eq!(d[2], 0x90);
}

#[test]
fn test_unit_ready_and_unknown_opcodes() {
    let (mut bus, _) = hd_bus(64);
    good(&mut bus, &[0; 6], &[]);
    check(&mut bus, &[0xc0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[], ScsiSense::INVALID_OPCODE);
    // READ TOC works on disks as well, as in QEMU.
    good(&mut bus, &[opcode::READ_TOC, 0, 0, 0, 0, 0, 0, 0, 12, 0], &[]);
    // The reserved group codes cannot be parsed.
    check(&mut bus, &[0x60, 0, 0, 0, 0, 0], &[], ScsiSense::INVALID_OPCODE);
    check(&mut bus, &[opcode::RESERVE, 1, 0, 0, 0, 0], &[], ScsiSense::INVALID_FIELD);
    good(&mut bus, &[opcode::RESERVE, 0, 0, 0, 0, 0], &[]);
}

#[test]
fn mode_sense_and_select() {
    let (mut bus, img) = hd_bus(2048);
    let d = good(&mut bus, &[opcode::MODE_SENSE, 0, 0x08, 0, 255, 0], &[]);
    // Header, block descriptor, caching page with WCE.
    assert_eq!(d.len(), 255);
    assert_eq!(&d[..4], &[3 + 8 + 20, 0, 0x10, 8]);
    assert_eq!(&d[4..12], &[0, 0, 8, 0, 0, 0, 2, 0]);
    assert_eq!(&d[12..15], &[0x08, 0x12, 0x04]);
    // Changeable values.
    let d = good(&mut bus, &[opcode::MODE_SENSE, 8, 0x48, 0, 255, 0], &[]);
    assert_eq!(&d[..7], &[3 + 20, 0, 0x10, 0, 0x08, 0x12, 0x04]);
    // All pages of a disk: 1, 4, 5 and 8.
    let d = good(&mut bus, &[opcode::MODE_SENSE_10, 8, 0x3f, 0, 0, 0, 0, 1, 0, 0], &[]);
    let len = usize::from(u16::from_be_bytes([d[0], d[1]])) + 2;
    assert_eq!(len, 8 + 12 + 24 + 32 + 20);
    check(&mut bus, &[opcode::MODE_SENSE, 0, 0x0e, 0, 255, 0], &[], ScsiSense::INVALID_FIELD);
    check(
        &mut bus,
        &[opcode::MODE_SENSE, 0, 0xc8, 0, 255, 0],
        &[],
        ScsiSense::SAVING_PARAMS_NOT_SUPPORTED,
    );

    // Turn the write cache off.
    let before = img.flush_count();
    let mut page = vec![0u8, 0, 0, 0, 0x08, 0x12];
    page.extend_from_slice(&[0; 0x12]);
    good(&mut bus, &[opcode::MODE_SELECT, 0x10, 0, 0, page.len() as u8, 0], &page);
    assert!(!bus.device(0).unwrap().write_cache_enabled());
    assert_eq!(img.flush_count(), before + 1);
    let d = good(&mut bus, &[opcode::MODE_SENSE, 8, 0x08, 0, 255, 0], &[]);
    assert_eq!(d[6], 0);
    // Changing a bit that is not changeable is refused, and nothing changes.
    page[6] = 0x05;
    check(
        &mut bus,
        &[opcode::MODE_SELECT, 0x10, 0, 0, page.len() as u8, 0],
        &page,
        ScsiSense::INVALID_PARAM,
    );
    assert!(!bus.device(0).unwrap().write_cache_enabled());
    // SP=1 or PF=0.
    check(
        &mut bus,
        &[opcode::MODE_SELECT, 0x11, 0, 0, page.len() as u8, 0],
        &page,
        ScsiSense::INVALID_FIELD,
    );
    check(&mut bus, &[opcode::MODE_SELECT, 0x10, 0, 0, 2, 0], &page, ScsiSense::INVALID_PARAM_LEN);
}

#[test]
fn report_luns_and_missing_luns() {
    let mut bus = ScsiBus::new(INFO);
    for lun in [0, 3, 300] {
        let img = Arc::new(VecBackend::new(1 << 16));
        bus.attach(ScsiDiskConf::hd(img).at(1, lun)).unwrap();
    }
    let r = run_lun(&mut bus, 1, 0, &[opcode::REPORT_LUNS, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], &[]);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.data.len(), 32);
    assert_eq!(&r.data[..8], &[0, 0, 0, 24, 0, 0, 0, 0]);
    assert_eq!(&r.data[8..16], &[0; 8]);
    assert_eq!(&r.data[16..18], &[0, 3]);
    assert_eq!(&r.data[24..26], &[0x41, 0x2c]);
    assert_eq!(r.residual, 256 - 32);
    // An allocation length under 16 is refused.
    let r = run_lun(&mut bus, 1, 0, &[opcode::REPORT_LUNS, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0], &[]);
    assert_eq!(r.sense, Some(ScsiSense::INVALID_FIELD));

    // LUN 5 does not exist. The first command reports the power on condition of the device the
    // request went through.
    let r = run_lun(&mut bus, 1, 5, &[0; 6], &[]);
    assert_eq!(r.sense, Some(ScsiSense::RESET));
    let r = run_lun(&mut bus, 1, 5, &[0; 6], &[]);
    assert_eq!(r.sense, Some(ScsiSense::LUN_NOT_SUPPORTED));
    let r = run_lun(&mut bus, 1, 5, &[INQUIRY_OP, 0, 0, 0, 36, 0], &[]);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.data[0], TYPE_NO_LUN);
    let r = run_lun(&mut bus, 1, 5, &[opcode::REQUEST_SENSE, 0, 0, 0, 18, 0], &[]);
    assert_eq!(ScsiSense::from_buf(&r.data), ScsiSense::LUN_NOT_SUPPORTED);
}

#[test]
fn target_without_lun_zero() {
    let mut bus = ScsiBus::new(INFO);
    let img = Arc::new(VecBackend::new(1 << 16));
    bus.attach(ScsiDiskConf::hd(img).at(2, 1)).unwrap();
    let r = run_lun(&mut bus, 2, 0, &[INQUIRY_OP, 0, 0, 0, 36, 0], &[]);
    assert_eq!(r.status, GOOD);
    assert_eq!(r.data[0], TYPE_NOT_PRESENT | TYPE_INACTIVE);
    assert_eq!(r.data[4], 31);
    assert_eq!(&r.data[16..32], b"QEMU TARGET     ");
    assert_eq!(&r.data[32..36], b"2.5\0");
}

#[test]
fn reported_luns_changed_is_cleared_by_report_luns() {
    let (mut bus, _) = hd_bus(64);
    bus.report_change(0, ScsiSense::REPORTED_LUNS_CHANGED);
    good(&mut bus, &[opcode::REPORT_LUNS, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], &[]);
    good(&mut bus, &[0; 6], &[]);
    bus.set_ua(ScsiSense::REPORTED_LUNS_CHANGED);
    check(&mut bus, &[0; 6], &[], ScsiSense::REPORTED_LUNS_CHANGED);
    good(&mut bus, &[0; 6], &[]);
}

#[test]
fn unit_attention_precedence() {
    let (mut bus, _) = hd_bus(64);
    let d = bus.device_mut(0).unwrap();
    d.set_ua(ScsiSense::CAPACITY_CHANGED);
    d.set_ua(ScsiSense::RESET);
    // A reset is more important than the capacity change, which it replaces.
    d.set_ua(ScsiSense::MEDIUM_CHANGED);
    check(&mut bus, &[0; 6], &[], ScsiSense::RESET);
    good(&mut bus, &[0; 6], &[]);
}

/// `test_unaligned_write_same()` from virtio-scsi-test.c.
#[test]
fn unaligned_write_same() {
    let (mut bus, img) = hd_bus(8 * 1024 * 1024 / 512);
    let buf1 = [0u8; 512];
    let mut buf2 = [0u8; 512];
    buf2[0] = 1;
    good(&mut bus, &[0x41, 0, 0, 0, 0, 1, 0, 0, 2, 0], &buf1);
    good(&mut bus, &[0x41, 0, 0, 0, 0, 1, 0, 0x33, 0, 0], &buf2);
    let c = img.contents();
    for b in 1..1 + 0x3300 {
        assert_eq!(c[b * 512], 1);
    }
    // With NDOB there is no data and QEMU completes the command without writing anything.
    good(&mut bus, &[0x41, 1, 0, 0, 0, 1, 0, 0x33, 0, 0], &[]);
    assert_eq!(img.contents()[512], 1);
}

#[test]
fn write_same_errors_and_zeroes() {
    let (mut bus, img) = hd_bus(64);
    img.fill(0, &[0xaa; 64 * 512]);
    good(&mut bus, &[opcode::WRITE_SAME_10, 0, 0, 0, 0, 2, 0, 0, 3, 0], &[0; 512]);
    let c = img.contents();
    assert!(c[2 * 512..5 * 512].iter().all(|&b| b == 0));
    assert_eq!(c[5 * 512], 0xaa);
    check(
        &mut bus,
        &[opcode::WRITE_SAME_10, 0, 0, 0, 0, 2, 0, 0, 0, 0],
        &[0; 512],
        ScsiSense::INVALID_FIELD,
    );
    check(
        &mut bus,
        &[opcode::WRITE_SAME_10, 2, 0, 0, 0, 2, 0, 0, 1, 0],
        &[0; 512],
        ScsiSense::INVALID_FIELD,
    );
    check(
        &mut bus,
        &[opcode::WRITE_SAME_10, 0, 0, 0, 0, 60, 0, 0, 8, 0],
        &[0; 512],
        ScsiSense::LBA_OUT_OF_RANGE,
    );
}

fn unmap_params(lba: u64, count: u32) -> Vec<u8> {
    let mut p = vec![0, 0x16, 0, 0x10, 0, 0, 0, 0];
    p.extend_from_slice(&lba.to_be_bytes());
    p.extend_from_slice(&count.to_be_bytes());
    p.extend_from_slice(&[0; 4]);
    p
}

/// `test_iothread_attach_node()` aside, this is `test_unmap_large_LBA()` from virtio-scsi-test.c.
#[test]
fn unmap_large_lba() {
    let mut bus = ScsiBus::new(INFO);
    let mut conf = ScsiDiskConf::hd(Arc::new(NullBackend(1 << 30))).at(0, 0);
    conf.logical_block_size = 4096;
    conf.physical_block_size = 4096;
    bus.attach(conf).unwrap();
    check(&mut bus, &[0; 6], &[], ScsiSense::RESET);
    let r = run(&mut bus, &[0x42, 0, 0, 0, 0, 0, 0, 0, 0x18, 0], &unmap_params(0x7fff, 0x3ff));
    assert_ne!(r.status, CHECK_CONDITION);
    assert_eq!(r.status, GOOD);
}

#[test]
fn unmap_checks() {
    let (mut bus, img) = hd_bus(64);
    img.fill(0, &[0xaa; 64 * 512]);
    good(&mut bus, &[opcode::UNMAP, 0, 0, 0, 0, 0, 0, 0, 0x18, 0], &unmap_params(4, 2));
    let c = img.contents();
    assert!(c[4 * 512..6 * 512].iter().all(|&b| b == 0));
    assert_eq!(c[6 * 512], 0xaa);
    check(
        &mut bus,
        &[opcode::UNMAP, 0, 0, 0, 0, 0, 0, 0, 0x18, 0],
        &unmap_params(63, 2),
        ScsiSense::LBA_OUT_OF_RANGE,
    );
    check(
        &mut bus,
        &[opcode::UNMAP, 0, 0, 0, 0, 0, 0, 0, 0x04, 0],
        &unmap_params(0, 1),
        ScsiSense::INVALID_PARAM_LEN,
    );
    check(
        &mut bus,
        &[opcode::UNMAP, 1, 0, 0, 0, 0, 0, 0, 0x18, 0],
        &unmap_params(0, 1),
        ScsiSense::INVALID_FIELD,
    );
}

fn cd_bus(media: Option<Arc<dyn BlockBackend>>) -> ScsiBus {
    let mut bus = ScsiBus::new(INFO);
    bus.attach(ScsiDiskConf::cd(media).at(0, 0)).unwrap();
    check(&mut bus, &[0; 6], &[], ScsiSense::RESET);
    bus
}

/// `test_write_to_cdrom()` from virtio-scsi-test.c.
#[test]
fn write_to_cdrom() {
    let mut bus = cd_bus(Some(Arc::new(NullBackend(1 << 20))));
    let r = run(&mut bus, &[0x2a, 0, 0, 0, 0, 0, 0, 0, 1, 0], &[0; 2048]);
    assert_eq!(r.status, CHECK_CONDITION);
    let dev = bus.find(0, 0, 0).unwrap();
    let mut req = bus.new_request(dev, 0, &[0x2a, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
    bus.execute(&mut req, &[0; 2048]);
    let sense = bus.get_sense(&req, 96);
    assert_eq!(sense[0], 0x70);
    assert_eq!(sense[2], 7);
    assert_eq!(sense[12], 0x27);
    assert_eq!(sense[13], 0);
}

#[test]
fn cdrom_commands() {
    let img: Arc<dyn BlockBackend> = Arc::new(VecBackend::new(100 * 2048));
    let mut bus = cd_bus(Some(img));
    let d = good(&mut bus, &[INQUIRY_OP, 0, 0, 0, 36, 0], &[]);
    assert_eq!((d[0], d[1]), (TYPE_ROM, 0x80));
    assert_eq!(&d[16..32], b"QEMU CD-ROM     ");
    let d = good(&mut bus, &[INQUIRY_OP, 1, 0, 0, 36, 0], &[]);
    assert_eq!(&d[..6], &[5, 0, 0, 2, 0, 0x83]);
    let d = good(&mut bus, &[opcode::READ_CAPACITY_10, 0, 0, 0, 0, 0, 0, 0, 0, 0], &[]);
    assert_eq!(d, vec![0, 0, 0, 99, 0, 0, 8, 0]);
    let d = good(&mut bus, &[opcode::READ_TOC, 0, 0, 0, 0, 0, 0, 0, 12 + 8, 0], &[]);
    assert_eq!(&d[..4], &[0, 18, 1, 1]);
    assert_eq!(&d[4..12], &[0, 0x14, 1, 0, 0, 0, 0, 0]);
    assert_eq!(&d[12..20], &[0, 0x16, 0xaa, 0, 0, 0, 0, 100]);
    let d = good(&mut bus, &[opcode::GET_CONFIGURATION, 0, 0, 0, 0, 0, 0, 0, 40, 0], &[]);
    assert_eq!(&d[..8], &[0, 0, 0, 36, 0, 0, 0, 8]);
    let d = good(&mut bus, &[opcode::MECHANISM_STATUS, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0], &[]);
    assert_eq!(d, vec![0, 0, 0, 0, 0, 1, 0, 0]);
    let d = good(&mut bus, &[opcode::MODE_SENSE_10, 0, 0x2a, 0, 0, 0, 0, 0, 36, 0], &[]);
    assert_eq!(&d[8..14], &[0x2a, 0x14, 0x3b, 0, 0x7f, 0xff]);
    // DMA reads use 2048 byte blocks.
    assert_eq!(good(&mut bus, &[opcode::READ_10, 0, 0, 0, 0, 1, 0, 0, 1, 0], &[]).len(), 2048);
}

#[test]
fn cdrom_tray_and_media_change() {
    let mut bus = cd_bus(None);
    check(&mut bus, &[0; 6], &[], ScsiSense::NO_MEDIUM);
    // Lock the tray; ejecting is refused.
    good(&mut bus, &[opcode::ALLOW_MEDIUM_REMOVAL, 0, 0, 0, 1, 0], &[]);
    check(
        &mut bus,
        &[opcode::START_STOP, 0, 0, 0, 2, 0],
        &[],
        ScsiSense::NOT_READY_REMOVAL_PREVENTED,
    );
    good(&mut bus, &[opcode::ALLOW_MEDIUM_REMOVAL, 0, 0, 0, 0, 0], &[]);

    let media: Arc<dyn BlockBackend> = Arc::new(VecBackend::new(10 * 2048));
    bus.device_mut(0).unwrap().change_media(Some(media));
    // The guest first sees "no medium", then "medium changed".
    check(&mut bus, &[0; 6], &[], ScsiSense::UNIT_ATTENTION_NO_MEDIUM);
    check(&mut bus, &[0; 6], &[], ScsiSense::MEDIUM_CHANGED);
    good(&mut bus, &[0; 6], &[]);
    // Open and close the tray.
    good(&mut bus, &[opcode::START_STOP, 0, 0, 0, 2, 0], &[]);
    assert!(bus.device(0).unwrap().tray_open());
    check(&mut bus, &[0; 6], &[], ScsiSense::NO_MEDIUM);
    good(&mut bus, &[opcode::START_STOP, 0, 0, 0, 3, 0], &[]);
    good(&mut bus, &[0; 6], &[]);

    // GET EVENT STATUS NOTIFICATION reports the new medium once.
    let gesn = [opcode::GET_EVENT_STATUS_NOTIFICATION, 1, 0, 0, 0x10, 0, 0, 0, 8, 0];
    assert_eq!(good(&mut bus, &gesn, &[]), vec![0, 4, 4, 0x10, 2, 2, 0, 0]);
    assert_eq!(good(&mut bus, &gesn, &[]), vec![0, 4, 4, 0x10, 0, 2, 0, 0]);
    check(&mut bus, &[gesn[0], 0, 0, 0, 0x10, 0, 0, 0, 8, 0], &[], ScsiSense::INVALID_FIELD);

    // Locked with a medium, the error is ILLEGAL REQUEST.
    good(&mut bus, &[opcode::ALLOW_MEDIUM_REMOVAL, 0, 0, 0, 1, 0], &[]);
    check(
        &mut bus,
        &[opcode::START_STOP, 0, 0, 0, 2, 0],
        &[],
        ScsiSense::ILLEGAL_REQ_REMOVAL_PREVENTED,
    );
    assert!(bus.device(0).unwrap().tray_locked());
    bus.reset();
    assert!(!bus.device(0).unwrap().tray_locked());
}

#[test]
fn realize_errors() {
    let err = |conf: ScsiDiskConf| ScsiDisk::new(conf).unwrap_err().to_string();
    let img = || -> Arc<dyn BlockBackend> { Arc::new(VecBackend::new(1 << 16)) };
    let mut c = ScsiDiskConf::default();
    assert!(err(c.clone()).contains("drive property not set"));
    c = ScsiDiskConf::hd(img());
    c.serial = Some("x".repeat(37));
    assert!(err(c.clone()).contains("The serial number can't be longer than 36 characters"));
    c.serial = Some("x".repeat(21));
    assert!(err(c.clone()).contains("when it is also used as the default for device_id"));
    c.device_id = Some("id".into());
    assert!(ScsiDisk::new(c).is_ok());
    c = ScsiDiskConf::hd(img());
    c.logical_block_size = 4096;
    assert!(err(c.clone()).contains("logical_block_size > physical_block_size not supported"));
    c = ScsiDiskConf::hd(img());
    c.min_io_size = 100;
    assert!(err(c).contains("min_io_size must be a multiple of logical_block_size"));
    c = ScsiDiskConf::hd(Arc::new(VecBackend::new(0)));
    assert!(err(c).contains("Device needs media, but drive is empty"));
}

#[test]
fn bus_addressing() {
    let mut bus = ScsiBus::new(INFO);
    let img = || -> Arc<dyn BlockBackend> { Arc::new(VecBackend::new(1 << 16)) };
    let mut c = ScsiDiskConf::hd(img());
    c.id = Some("disk0".into());
    let first = bus.attach(c).unwrap();
    assert_eq!(bus.device(first).unwrap().address(), (0, 0, 0));
    // No address: the next free target at LUN 0.
    let second = bus.attach(ScsiDiskConf::hd(img())).unwrap();
    assert_eq!(bus.device(second).unwrap().address(), (0, 1, 0));
    // A target without a LUN gets the first free LUN.
    let mut c = ScsiDiskConf::hd(img());
    c.scsi_id = Some(0);
    let third = bus.attach(c).unwrap();
    assert_eq!(bus.device(third).unwrap().address(), (0, 0, 1));

    let e = bus.attach(ScsiDiskConf::hd(img()).at(0, 0)).unwrap_err().to_string();
    assert!(e.contains("lun already used by 'disk0'"), "{e}");
    let e = bus.attach(ScsiDiskConf::hd(img()).at(256, 0)).unwrap_err().to_string();
    assert!(e.contains("bad scsi device id: 256"), "{e}");
    let e = bus.attach(ScsiDiskConf::hd(img()).at(0, 16384)).unwrap_err().to_string();
    assert!(e.contains("bad scsi device lun: 16384"), "{e}");
    let mut c = ScsiDiskConf::hd(img());
    c.channel = 1;
    let e = bus.attach(c).unwrap_err().to_string();
    assert!(e.contains("bad scsi channel id: 1"), "{e}");

    assert!(bus.detach(0, 1, 0).is_some());
    assert!(bus.detach(0, 1, 0).is_none());
    assert_eq!(bus.find(0, 0, 7), Some(0));
    assert_eq!(bus.find(0, 5, 0), None);

    let mut small =
        ScsiBus::new(ScsiBusInfo { tcq: false, max_channel: 0, max_target: 0, max_lun: 0 });
    small.attach(ScsiDiskConf::hd(img())).unwrap();
    let e = small.attach(ScsiDiskConf::hd(img())).unwrap_err().to_string();
    assert!(e.contains("no free target"), "{e}");
    let mut c = ScsiDiskConf::hd(img());
    c.scsi_id = Some(0);
    let e = small.attach(c).unwrap_err().to_string();
    assert!(e.contains("no free lun"), "{e}");
}

#[test]
fn sense_conversion() {
    let fixed = ScsiSense::LBA_OUT_OF_RANGE.to_buf(252, true);
    assert_eq!(fixed.len(), 18);
    assert_eq!(convert_sense(&fixed, 252, false), vec![0x72, 5, 0x21, 0, 0, 0, 0, 0]);
    assert_eq!(convert_sense(&[], 8, false), vec![0x72, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(ScsiSense::from_buf(&[0x72, 6, 0x29, 0]), ScsiSense::RESET);
    let (status, sense) = ScsiSense::from_io_error(&io::Error::other("boom"));
    assert_eq!((status, sense), (CHECK_CONDITION, ScsiSense::IO_ERROR));
}
