// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-blk through virtio-mmio. The request tests are ported from
//! `tests/qtest/virtio-blk-test.c` (`test_basic`, `indirect`, `config`, the any-layout and
//! the discard and write zeroes checks), the rest covers the error paths of
//! `hw/block/virtio-blk.c`.

mod common;

use common::{Buf, Guest, SplitRing};
use ruvm_hw_virtio::blk::*;
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_CONFIG, VIRTIO_MMIO_DEVICE_ID, VIRTIO_MMIO_INT_CONFIG};
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{BlockBackend, MemBlockBackend, VirtioBlk, VirtioBlkConf, VirtioDeviceClass};

const DISK_SIZE: usize = 1 << 20;
const SECTORS: u64 = (DISK_SIZE / 512) as u64;

fn blk(disk: &MemBlockBackend, conf: VirtioBlkConf) -> Option<Box<dyn VirtioDeviceClass>> {
    Some(Box::new(VirtioBlk::new(Box::new(disk.clone()), conf)))
}

fn features() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

/// A negotiated device with queue 0 ready.
fn setup(legacy: bool, disk: &MemBlockBackend, conf: VirtioBlkConf) -> (Guest, SplitRing) {
    let mut g = Guest::new(legacy, blk(disk, conf));
    g.negotiate(features());
    let q = g.setup_queue(0, 0);
    g.driver_ok();
    (g, q)
}

fn header(ty: u32, sector: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(16);
    h.extend_from_slice(&ty.to_le_bytes());
    h.extend_from_slice(&0u32.to_le_bytes());
    h.extend_from_slice(&sector.to_le_bytes());
    h
}

fn dwz(sector: u64, num_sectors: u32, flags: u32) -> Vec<u8> {
    let mut d = Vec::with_capacity(16);
    d.extend_from_slice(&sector.to_le_bytes());
    d.extend_from_slice(&num_sectors.to_le_bytes());
    d.extend_from_slice(&flags.to_le_bytes());
    d
}

/// The result of one request.
#[derive(Debug)]
struct Done {
    status: u8,
    used_len: u32,
    data: Vec<u8>,
}

/// Submits a request the way `virtio_blk_request()` plus the qtest helpers do: one device
/// readable buffer for the header, one for `out`, one device writable buffer of `in_len` bytes
/// and one for the status byte. Then kicks and waits for the used element.
fn request(
    g: &mut Guest,
    q: &mut SplitRing,
    ty: u32,
    sector: u64,
    out: &[u8],
    in_len: u32,
) -> Done {
    let hdr = g.alloc(16, 16);
    g.write_mem(hdr, &header(ty, sector));
    let mut bufs = vec![Buf::out(hdr, 16)];
    if !out.is_empty() {
        let d = g.alloc(out.len() as u64, 16);
        g.write_mem(d, out);
        bufs.push(Buf::out(d, out.len() as u32));
    }
    let data = if in_len > 0 {
        let d = g.alloc(u64::from(in_len), 16);
        g.write_mem(d, &vec![0xaa; in_len as usize]);
        bufs.push(Buf::inp(d, in_len));
        Some(d)
    } else {
        None
    };
    let status = g.alloc(1, 1);
    g.write_mem(status, &[0xff]);
    bufs.push(Buf::inp(status, 1));
    let head = q.submit(g, &bufs);
    g.kick(q);
    let (id, used_len) = q.get_used(g).expect("request completed");
    assert_eq!(id, u32::from(head));
    Done {
        status: g.read_mem(status, 1)[0],
        used_len,
        data: data.map(|d| g.read_mem(d, in_len as usize)).unwrap_or_default(),
    }
}

fn sector_of(fill: &[u8]) -> Vec<u8> {
    let mut s = vec![0; 512];
    s[..fill.len()].copy_from_slice(fill);
    s
}

#[test]
fn identity_and_config() {
    for legacy in [true, false] {
        let disk = MemBlockBackend::new(DISK_SIZE);
        let g = Guest::new(legacy, blk(&disk, VirtioBlkConf::default()));
        assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), u32::from(VIRTIO_ID_BLOCK));
        // The qtest `config` test: capacity is the image size in sectors.
        assert_eq!(g.config_readq(0), SECTORS);
        assert_eq!(g.config_readl(12), 256 - 2, "seg_max");
        assert_eq!(g.config_readb(18), 16, "heads");
        assert_eq!(g.config_readb(19), 63, "sectors");
        assert_eq!(g.config_readl(20), 512, "blk_size");
        assert_eq!(g.config_readb(32), 1, "wce");
        assert_eq!(g.config_readw(34), 1, "num_queues");
        assert_eq!(g.config_readl(36), BDRV_REQUEST_MAX_SECTORS, "max_discard_sectors");
        assert_eq!(g.config_readl(40), 1, "max_discard_seg");
        assert_eq!(g.config_readl(44), 1, "discard_sector_alignment");
        assert_eq!(g.config_readl(48), BDRV_REQUEST_MAX_SECTORS, "max_write_zeroes_sectors");
        assert_eq!(g.config_readl(52), 1, "max_write_zeroes_seg");
        assert_eq!(g.config_readb(56), 1, "write_zeroes_may_unmap");
        // The config space ends after write_zeroes_may_unmap.
        assert_eq!(g.config_readb(57), 0xff);
    }
}

#[test]
fn config_size_follows_features() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let conf = VirtioBlkConf { discard: false, write_zeroes: false, ..Default::default() };
    let g = Guest::new(false, blk(&disk, conf));
    assert_eq!(g.config_readw(34), 1);
    assert_eq!(g.config_readl(36), u32::MAX);
    let f = g.device_features();
    assert!(!has_feature(f, VIRTIO_BLK_F_DISCARD));
    assert!(!has_feature(f, VIRTIO_BLK_F_WRITE_ZEROES));

    let conf = VirtioBlkConf { write_zeroes: false, ..Default::default() };
    let g = Guest::new(false, blk(&disk, conf));
    assert_eq!(g.config_readl(44), 1);
    assert_eq!(g.config_readl(48), u32::MAX);
}

#[test]
fn offered_features() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let modern = Guest::new(false, blk(&disk, VirtioBlkConf::default())).device_features();
    for bit in [
        VIRTIO_BLK_F_SEG_MAX,
        VIRTIO_BLK_F_GEOMETRY,
        VIRTIO_BLK_F_TOPOLOGY,
        VIRTIO_BLK_F_BLK_SIZE,
        VIRTIO_BLK_F_FLUSH,
        VIRTIO_BLK_F_CONFIG_WCE,
        VIRTIO_BLK_F_DISCARD,
        VIRTIO_BLK_F_WRITE_ZEROES,
        VIRTIO_F_VERSION_1,
    ] {
        assert!(has_feature(modern, bit), "modern bit {bit}");
    }
    for bit in [VIRTIO_BLK_F_SCSI, VIRTIO_BLK_F_RO, VIRTIO_BLK_F_MQ, VIRTIO_F_ANY_LAYOUT] {
        assert!(!has_feature(modern, bit), "modern bit {bit}");
    }

    let legacy = Guest::new(true, blk(&disk, VirtioBlkConf::default()));
    let f = u64::from(legacy.readl(ruvm_hw_virtio::mmio::VIRTIO_MMIO_DEVICE_FEATURES));
    assert!(has_feature(f, VIRTIO_BLK_F_SCSI));
    assert!(!has_feature(f, VIRTIO_F_ANY_LAYOUT));

    let ro = MemBlockBackend::new(DISK_SIZE).read_only();
    let f = Guest::new(false, blk(&ro, VirtioBlkConf::default())).device_features();
    assert!(has_feature(f, VIRTIO_BLK_F_RO));

    let conf = VirtioBlkConf { num_queues: Some(2), ..Default::default() };
    let g = Guest::new(false, blk(&disk, conf));
    assert!(has_feature(g.device_features(), VIRTIO_BLK_F_MQ));
    assert_eq!(g.config_readw(34), 2);
}

/// `test_basic()` from virtio-blk-test.c.
#[test]
fn basic() {
    for legacy in [true, false] {
        let disk = MemBlockBackend::new(DISK_SIZE);
        let (mut g, mut q) = setup(legacy, &disk, VirtioBlkConf::default());

        // Write and read with three descriptors.
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &sector_of(b"TEST"), 0);
        assert_eq!(d.status, VIRTIO_BLK_S_OK);
        assert_eq!(d.used_len, 1);
        assert!(g.irq());
        g.ack();
        assert_eq!(&disk.data().lock().unwrap()[..4], b"TEST");

        let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 0, &[], 512);
        assert_eq!(d.status, VIRTIO_BLK_S_OK);
        assert_eq!(d.used_len, 513);
        assert_eq!(d.data, sector_of(b"TEST"));
        assert!(g.irq());
        g.ack();

        // Write zeroes over the sector and read it back.
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 1, 0), 0);
        assert_eq!(d.status, VIRTIO_BLK_S_OK);
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 0, &[], 512);
        assert_eq!(d.data, vec![0; 512]);

        // Write zeroes with UNMAP.
        request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &sector_of(b"TEST"), 0);
        let flags = VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP;
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 1, flags), 0);
        assert_eq!(d.status, VIRTIO_BLK_S_OK);
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 0, &[], 512);
        assert_eq!(d.data, vec![0; 512]);

        // Discard.
        let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &dwz(0, 1, 0), 0);
        assert_eq!(d.status, VIRTIO_BLK_S_OK);

        // Any layout: header and data in one buffer, data and status in another.
        if !legacy {
            let data = sector_of(b"TEST");
            let out = g.alloc(16 + 512, 16);
            let mut raw = header(VIRTIO_BLK_T_OUT, 1);
            raw.extend_from_slice(&data);
            g.write_mem(out, &raw);
            let st = g.alloc(1, 1);
            q.submit(&g, &[Buf::out(out, 16 + 512), Buf::inp(st, 1)]);
            g.kick(&q);
            assert_eq!(q.get_used(&g).unwrap().1, 1);
            assert_eq!(g.read_mem(st, 1)[0], VIRTIO_BLK_S_OK);

            let hdr = g.alloc(16, 16);
            g.write_mem(hdr, &header(VIRTIO_BLK_T_IN, 1));
            let inp = g.alloc(513, 16);
            q.submit(&g, &[Buf::out(hdr, 16), Buf::inp(inp, 513)]);
            g.kick(&q);
            assert_eq!(q.get_used(&g).unwrap().1, 513);
            let got = g.read_mem(inp, 513);
            assert_eq!(&got[..512], &data[..]);
            assert_eq!(got[512], VIRTIO_BLK_S_OK);
        }
    }
}

/// The `indirect` test: a write and a read through indirect descriptor tables.
#[test]
fn indirect() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let mut g = Guest::new(false, blk(&disk, VirtioBlkConf::default()));
    let f = g.negotiate(features());
    assert!(has_feature(f, VIRTIO_RING_F_INDIRECT_DESC));
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();

    let hdr = g.alloc(16, 16);
    g.write_mem(hdr, &header(VIRTIO_BLK_T_OUT, 0));
    let data = g.alloc(512, 16);
    g.write_mem(data, &sector_of(b"TEST"));
    let st = g.alloc(1, 1);
    let head = q.add_indirect(&mut g, &[Buf::out(hdr, 16), Buf::out(data, 512), Buf::inp(st, 1)]);
    q.make_available(&g, head);
    g.kick(&q);
    assert_eq!(q.get_used(&g), Some((u32::from(head), 1)));
    assert_eq!(g.read_mem(st, 1)[0], VIRTIO_BLK_S_OK);

    g.write_mem(hdr, &header(VIRTIO_BLK_T_IN, 0));
    let head = q.add_indirect(&mut g, &[Buf::out(hdr, 16), Buf::inp(data, 512), Buf::inp(st, 1)]);
    g.write_mem(data, &[0; 512]);
    q.make_available(&g, head);
    g.kick(&q);
    assert_eq!(q.get_used(&g), Some((u32::from(head), 513)));
    assert_eq!(g.read_mem(data, 512), sector_of(b"TEST"));
}

#[test]
fn several_requests_in_one_kick() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    let mut heads = Vec::new();
    let mut stats = Vec::new();
    for i in 0..4u64 {
        let hdr = g.alloc(16, 16);
        g.write_mem(hdr, &header(VIRTIO_BLK_T_OUT, i));
        let data = g.alloc(512, 16);
        g.write_mem(data, &[i as u8 + 1; 512]);
        let st = g.alloc(1, 1);
        heads.push(q.submit(&g, &[Buf::out(hdr, 16), Buf::out(data, 512), Buf::inp(st, 1)]));
        stats.push(st);
    }
    g.kick(&q);
    for (h, st) in heads.iter().zip(&stats) {
        assert_eq!(q.get_used(&g), Some((u32::from(*h), 1)));
        assert_eq!(g.read_mem(*st, 1)[0], VIRTIO_BLK_S_OK);
    }
    let d = disk.data();
    let d = d.lock().unwrap();
    for i in 0..4 {
        assert!(d[i * 512..(i + 1) * 512].iter().all(|&b| b == i as u8 + 1));
    }
}

#[test]
fn get_id() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let conf = VirtioBlkConf { serial: Some("ruvm-disk".into()), ..Default::default() };
    let (mut g, mut q) = setup(false, &disk, conf);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_GET_ID, 0, &[], 20);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    assert_eq!(&d.data[..10], b"ruvm-disk\0");
    assert_eq!(d.used_len, 21);

    // A serial of 20 characters or more is cut to 20 bytes, with no terminator.
    let conf =
        VirtioBlkConf { serial: Some("0123456789abcdefghijklmn".into()), ..Default::default() };
    let (mut g, mut q) = setup(false, &disk, conf);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_GET_ID, 0, &[], 24);
    assert_eq!(&d.data[..20], b"0123456789abcdefghij");
    assert_eq!(&d.data[20..], [0xaa; 4]);

    // No serial reads as an empty string.
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_GET_ID, 0, &[], 20);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    assert_eq!(d.data[0], 0);
}

#[test]
fn flush_and_write_cache() {
    // Write cache on: writes do not flush, FLUSH does.
    let disk = MemBlockBackend::new(DISK_SIZE);
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &[1; 512], 0);
    assert_eq!(disk.flush_count(), 0);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_FLUSH, 0, &[], 0);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    assert_eq!(disk.flush_count(), 1);

    // The guest turns the cache off through config space, so writes flush.
    assert_eq!(g.config_readb(32), 1);
    g.store(VIRTIO_MMIO_CONFIG + 32, 1, 0);
    assert_eq!(g.config_readb(32), 0);
    request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &[1; 512], 0);
    assert_eq!(disk.flush_count(), 2);
    // A reset brings the property value back.
    g.set_status(0);
    assert_eq!(g.config_readb(32), 1);
}

#[test]
fn writethrough_when_driver_lacks_flush() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let mut g = Guest::new(false, blk(&disk, VirtioBlkConf::default()));
    g.negotiate(features() & !feature(VIRTIO_BLK_F_FLUSH) & !feature(VIRTIO_BLK_F_CONFIG_WCE));
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    assert_eq!(g.config_readb(32), 0);
    request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &[1; 512], 0);
    assert_eq!(disk.flush_count(), 1);
}

#[test]
fn write_cache_property_off() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let conf = VirtioBlkConf { write_cache: false, ..Default::default() };
    let (mut g, mut q) = setup(false, &disk, conf);
    // CONFIG_WCE was negotiated, so the property decides, and it says writethrough.
    assert_eq!(g.config_readb(32), 0);
    request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &[1; 512], 0);
    assert_eq!(disk.flush_count(), 1);
}

#[test]
fn out_of_range_requests_fail() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, SECTORS, &[], 512);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_OUT, SECTORS - 1, &[0; 1024], 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    // Not a whole number of sectors.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 0, &[], 100);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    // The last sector is fine.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, SECTORS - 1, &[], 512);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &dwz(SECTORS, 1, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(SECTORS - 1, 2, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
}

#[test]
fn discard_and_write_zeroes_limits() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let conf =
        VirtioBlkConf { max_discard_sectors: 4, max_write_zeroes_sectors: 8, ..Default::default() };
    let (mut g, mut q) = setup(false, &disk, conf);
    assert_eq!(g.config_readl(36), 4);
    assert_eq!(g.config_readl(48), 8);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &dwz(0, 5, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &dwz(0, 4, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 9, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 8, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
}

#[test]
fn unsupported_requests() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    // Unknown type.
    let d = request(&mut g, &mut q, 0x42, 0, &[], 0);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // SCSI passthrough.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_SCSI_CMD, 0, &[0; 16], 96);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // Discard with UNMAP is not allowed.
    let flags = VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP;
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &dwz(0, 1, flags), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // Unknown flags.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 1, 2), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // Discard without the OUT bit in the type.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD & !VIRTIO_BLK_T_OUT, 0, &dwz(0, 1, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // More than one discard segment.
    let mut two = dwz(0, 1, 0);
    two.extend_from_slice(&dwz(2, 1, 0));
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_DISCARD, 0, &two, 0);
    assert_eq!(d.status, VIRTIO_BLK_S_UNSUPP);
    // The device is still healthy.
    assert_eq!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
}

#[test]
fn read_only_disk() {
    let disk = MemBlockBackend::from_vec(vec![7; DISK_SIZE]).read_only();
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 0, &[1; 512], 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_WRITE_ZEROES, 0, &dwz(0, 1, 0), 0);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 0, &[], 512);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    assert_eq!(d.data, vec![7; 512]);
    assert!(disk.data().lock().unwrap().iter().all(|&b| b == 7));
}

#[test]
fn missing_headers_break_the_device() {
    for legacy in [true, false] {
        let disk = MemBlockBackend::new(DISK_SIZE);
        let (mut g, mut q) = setup(legacy, &disk, VirtioBlkConf::default());
        // Only a header, no status byte.
        let hdr = g.alloc(16, 16);
        g.write_mem(hdr, &header(VIRTIO_BLK_T_IN, 0));
        q.submit(&g, &[Buf::out(hdr, 16)]);
        g.kick(&q);
        assert_eq!(q.get_used(&g), None);
        if legacy {
            // Legacy drivers get no NEEDS_RESET and no interrupt, the device just stops.
            assert_eq!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
            assert!(!g.irq());
        } else {
            assert_ne!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
            assert!(g.irq());
            assert_eq!(g.isr() & VIRTIO_MMIO_INT_CONFIG, VIRTIO_MMIO_INT_CONFIG);
        }
        // A broken device ignores further requests until reset.
        let st = g.alloc(1, 1);
        g.write_mem(hdr, &header(VIRTIO_BLK_T_FLUSH, 0));
        q.submit(&g, &[Buf::out(hdr, 16), Buf::inp(st, 1)]);
        g.kick(&q);
        assert_eq!(q.get_used(&g), None);
        let broken = g.mmio.with_backend(|b| b.vdev().is_broken()).unwrap();
        assert!(broken);
        g.set_status(0);
        let broken = g.mmio.with_backend(|b| b.vdev().is_broken()).unwrap();
        assert!(!broken);
    }
}

#[test]
fn short_header_breaks_the_device() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let (mut g, mut q) = setup(false, &disk, VirtioBlkConf::default());
    let hdr = g.alloc(8, 8);
    let st = g.alloc(1, 1);
    q.submit(&g, &[Buf::out(hdr, 8), Buf::inp(st, 1)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g), None);
    assert_ne!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
}

#[test]
fn realize_checks_properties() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let cases: Vec<(VirtioBlkConf, &str)> = vec![
        (VirtioBlkConf { num_queues: Some(0), ..Default::default() }, "num-queues"),
        (VirtioBlkConf { queue_size: 2, ..Default::default() }, "must be > 2"),
        (VirtioBlkConf { queue_size: 100, ..Default::default() }, "power of 2"),
        (VirtioBlkConf { queue_size: 2048, ..Default::default() }, "power of 2"),
        (VirtioBlkConf { max_discard_sectors: 0, ..Default::default() }, "max-discard-sectors"),
        (
            VirtioBlkConf { max_write_zeroes_sectors: 0, ..Default::default() },
            "max-write-zeroes-sectors",
        ),
        (VirtioBlkConf { heads: 300, cyls: 1, secs: 1, ..Default::default() }, "heads"),
        (
            VirtioBlkConf {
                logical_block_size: 4096,
                physical_block_size: 512,
                ..Default::default()
            },
            "logical_block_size > physical_block_size",
        ),
    ];
    for (conf, msg) in cases {
        let err = Guest::try_new(false, blk(&disk, conf)).err().expect(msg);
        assert!(err.to_string().contains(msg), "{err} does not mention {msg}");
    }
}

#[test]
fn logical_block_size_4k() {
    let disk = MemBlockBackend::new(DISK_SIZE);
    let conf =
        VirtioBlkConf { logical_block_size: 4096, physical_block_size: 4096, ..Default::default() };
    let (mut g, mut q) = setup(false, &disk, conf);
    assert_eq!(g.config_readl(20), 4096);
    assert_eq!(g.config_readl(44), 8, "discard_sector_alignment");
    // Sector numbers stay in 512 byte units but must be block aligned.
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 1, &[], 4096);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_IN, 8, &[], 512);
    assert_eq!(d.status, VIRTIO_BLK_S_IOERR);
    let d = request(&mut g, &mut q, VIRTIO_BLK_T_OUT, 8, &[3; 4096], 0);
    assert_eq!(d.status, VIRTIO_BLK_S_OK);
    assert!(disk.data().lock().unwrap()[4096..8192].iter().all(|&b| b == 3));
}

#[test]
fn mem_backend() {
    let mut disk = MemBlockBackend::new(4096);
    assert_eq!(disk.size(), 4096);
    assert!(disk.is_writable());
    disk.write_at(512, b"abc").unwrap();
    let mut buf = [0; 3];
    disk.read_at(512, &mut buf).unwrap();
    assert_eq!(&buf, b"abc");
    assert!(disk.read_at(4095, &mut buf).is_err());
    assert!(disk.write_at(4094, b"abc").is_err());
    disk.write_zeroes(512, 512, false).unwrap();
    disk.read_at(512, &mut buf).unwrap();
    assert_eq!(buf, [0; 3]);
    disk.flush().unwrap();
    assert_eq!(disk.flush_count(), 1);
}
