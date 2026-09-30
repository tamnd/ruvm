// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-rng through virtio-mmio, in the spirit of `tests/qtest/virtio-rng-test.c`.

mod common;

use std::sync::{Arc, Mutex};

use common::{Buf, Guest};
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{EntropySource, RandomFile, VirtioDeviceClass, VirtioRng, VirtioRngConf};

/// Hands out 0, 1, 2, ... and counts how much it gave, with an optional budget.
#[derive(Debug, Clone, Default)]
struct Counter {
    state: Arc<Mutex<(u8, usize, Option<usize>)>>,
}

impl Counter {
    fn limited(budget: usize) -> Self {
        let c = Counter::default();
        c.state.lock().unwrap().2 = Some(budget);
        c
    }

    fn given(&self) -> usize {
        self.state.lock().unwrap().1
    }

    fn add_budget(&self, n: usize) {
        let mut s = self.state.lock().unwrap();
        s.2 = Some(s.2.unwrap_or(0) + n);
    }
}

impl EntropySource for Counter {
    fn fill(&mut self, buf: &mut [u8]) -> usize {
        let mut s = self.state.lock().unwrap();
        let n = s.2.map_or(buf.len(), |b| b.min(buf.len()));
        for b in &mut buf[..n] {
            *b = s.0;
            s.0 = s.0.wrapping_add(1);
        }
        s.1 += n;
        if let Some(b) = s.2.as_mut() {
            *b -= n;
        }
        n
    }
}

fn rng(src: &Counter, conf: VirtioRngConf) -> Option<Box<dyn VirtioDeviceClass>> {
    Some(Box::new(VirtioRng::new(Box::new(src.clone()), conf)))
}

fn no_event_idx() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

#[test]
fn fills_buffers() {
    for legacy in [true, false] {
        let src = Counter::default();
        let mut g = Guest::new(legacy, rng(&src, VirtioRngConf::default()));
        g.negotiate(no_event_idx());
        let mut q = g.setup_queue(0, 0);
        g.driver_ok();

        let a = g.alloc(16, 8);
        let b = g.alloc(32, 8);
        let h1 = q.submit(&g, &[Buf::inp(a, 16)]);
        let h2 = q.submit(&g, &[Buf::inp(b, 32)]);
        assert!(!g.irq());
        g.kick(&q);
        assert_eq!(q.get_used(&g), Some((u32::from(h1), 16)));
        assert_eq!(q.get_used(&g), Some((u32::from(h2), 32)));
        assert_eq!(q.get_used(&g), None);
        let expect: Vec<u8> = (0..48).collect();
        assert_eq!(g.read_mem(a, 16), expect[..16]);
        assert_eq!(g.read_mem(b, 32), expect[16..]);
        assert_eq!(src.given(), 48);
        assert!(g.irq());
        g.ack();
        assert!(!g.irq());
    }
}

#[test]
fn chained_buffers_are_filled_in_order() {
    let src = Counter::default();
    let mut g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    let a = g.alloc(3, 1);
    let b = g.alloc(4, 1);
    q.submit(&g, &[Buf::inp(a, 3), Buf::inp(b, 4)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g).unwrap().1, 7);
    assert_eq!(g.read_mem(a, 3), [0, 1, 2]);
    assert_eq!(g.read_mem(b, 4), [3, 4, 5, 6]);
}

#[test]
fn buffers_posted_before_driver_ok_are_filled_at_driver_ok() {
    let src = Counter::default();
    let mut g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    let a = g.alloc(8, 8);
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g), None);
    assert_eq!(src.given(), 0);
    g.driver_ok();
    assert_eq!(q.get_used(&g).unwrap().1, 8);
    assert!(g.irq());
}

#[test]
fn no_interrupt_flag_suppresses_the_irq() {
    let src = Counter::default();
    let mut g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    q.set_avail_flags(&g, common::VRING_AVAIL_F_NO_INTERRUPT);
    let a = g.alloc(8, 8);
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g).unwrap().1, 8);
    assert!(!g.irq());
    assert_eq!(g.isr(), 0);
}

#[test]
fn empty_source_does_not_hang_and_later_entropy_is_delivered() {
    let src = Counter::limited(0);
    let mut g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    let a = g.alloc(8, 8);
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g), None);
    assert!(!g.irq());

    // Entropy that turns up later, QEMU's chr_read() path.
    let used = g
        .mmio
        .with_device(|vdev, dev: &mut VirtioRng| dev.entropy_available(vdev, &[9, 8, 7]))
        .unwrap();
    assert_eq!(used, 3);
    assert_eq!(q.get_used(&g).unwrap().1, 3);
    assert_eq!(g.read_mem(a, 3), [9, 8, 7]);
    assert!(g.irq());

    // A source that gives a little at a time fills a buffer partially.
    src.add_budget(5);
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g).unwrap().1, 5);
}

#[test]
fn entropy_is_refused_before_driver_ok() {
    let src = Counter::default();
    let g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    let used = g
        .mmio
        .with_device(|vdev, dev: &mut VirtioRng| dev.entropy_available(vdev, &[1, 2, 3]))
        .unwrap();
    assert_eq!(used, 0);
}

#[test]
fn rate_limit() {
    let src = Counter::default();
    let conf = VirtioRngConf { max_bytes: 10, period_ms: 1000 };
    let mut g = Guest::new(false, rng(&src, conf));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    let armed = |g: &Guest| g.mmio.with_device(|_, d: &mut VirtioRng| d.timer_armed()).unwrap();
    assert!(!armed(&g));
    // As in QEMU, the first processing attempt arms the timer, and DRIVER_OK is one.
    g.driver_ok();
    assert!(armed(&g));

    let a = g.alloc(8, 8);
    let b = g.alloc(8, 8);
    q.submit(&g, &[Buf::inp(a, 8)]);
    q.submit(&g, &[Buf::inp(b, 8)]);
    g.kick(&q);
    // 10 bytes of quota: the first buffer is full, the second gets 2 bytes.
    assert_eq!(q.get_used(&g).unwrap().1, 8);
    assert_eq!(q.get_used(&g).unwrap().1, 2);
    assert_eq!(src.given(), 10);

    // Out of quota, nothing more until the period ends.
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g), None);
    let quota = g.mmio.with_device(|_, d: &mut VirtioRng| d.quota_remaining()).unwrap();
    assert_eq!(quota, 0);

    g.mmio.with_device(|vdev, d: &mut VirtioRng| d.check_rate_limit(vdev)).unwrap();
    assert_eq!(q.get_used(&g).unwrap().1, 8);
    assert_eq!(src.given(), 18);
    assert!(!armed(&g));
    let quota = g.mmio.with_device(|_, d: &mut VirtioRng| d.quota_remaining()).unwrap();
    assert_eq!(quota, 2);

    // The next request arms the timer again.
    q.submit(&g, &[Buf::inp(a, 1)]);
    g.kick(&q);
    assert!(armed(&g));
}

#[test]
fn realize_checks_properties() {
    let src = Counter::default();
    let err = Guest::try_new(false, rng(&src, VirtioRngConf { max_bytes: 1, period_ms: 0 }))
        .err()
        .unwrap();
    assert!(err.to_string().contains("'period' parameter expects a positive integer"));
    for max_bytes in [0, 1 << 63] {
        let err = Guest::try_new(false, rng(&src, VirtioRngConf { max_bytes, period_ms: 1 }))
            .err()
            .unwrap();
        assert!(err.to_string().contains("'max-bytes' parameter must be positive"));
    }
}

#[test]
fn random_file_reads_entropy() {
    let mut f = RandomFile::default();
    let mut buf = [0u8; 64];
    assert_eq!(f.fill(&mut buf), 64);
    assert_ne!(buf, [0u8; 64]);
    let mut missing = RandomFile::new("/nonexistent/ruvm-rng");
    assert_eq!(missing.fill(&mut buf), 0);
}

#[test]
fn reset_forgets_the_queue() {
    let src = Counter::default();
    let mut g = Guest::new(false, rng(&src, VirtioRngConf::default()));
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    g.set_status(0);
    let a = g.alloc(8, 8);
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g), None);
    assert_eq!(src.given(), 0);

    // Set up from scratch and it works again.
    g.negotiate(no_event_idx());
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    q.submit(&g, &[Buf::inp(a, 8)]);
    g.kick(&q);
    assert_eq!(q.get_used(&g).unwrap().1, 8);
}

#[test]
fn packed_ring() {
    use ruvm_hw_virtio::mmio::*;
    use ruvm_hw_virtio::rng::VIRTIO_RNG_QUEUE_SIZE;
    const AVAIL: u16 = 1 << 7;
    const USED: u16 = 1 << 15;
    const WRITE: u16 = 2;

    let src = Counter::default();
    let mut g = Guest::with_backend(false, rng(&src, VirtioRngConf::default()), |b| {
        b.vdev_mut().set_host_feature(34, true);
    })
    .unwrap();
    let f = g.negotiate(!feature(VIRTIO_RING_F_EVENT_IDX));
    assert!(has_feature(f, 34));

    let n = u64::from(VIRTIO_RNG_QUEUE_SIZE);
    let desc = g.alloc(16 * n, 16);
    let driver = g.alloc(4, 4);
    let device = g.alloc(4, 4);
    g.write_mem(desc, &vec![0; 16 * n as usize]);
    g.write_mem(driver, &[0; 4]);
    g.write_mem(device, &[0; 4]);
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
    g.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(VIRTIO_RNG_QUEUE_SIZE));
    g.writel(VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
    g.writel(VIRTIO_MMIO_QUEUE_DESC_HIGH, 0);
    g.writel(VIRTIO_MMIO_QUEUE_AVAIL_LOW, driver as u32);
    g.writel(VIRTIO_MMIO_QUEUE_AVAIL_HIGH, 0);
    g.writel(VIRTIO_MMIO_QUEUE_USED_LOW, device as u32);
    g.writel(VIRTIO_MMIO_QUEUE_USED_HIGH, 0);
    g.writel(VIRTIO_MMIO_QUEUE_READY, 1);
    g.driver_ok();

    // Two single descriptor buffers, buffer ids 5 and 6, first lap so AVAIL set and USED clear.
    let bufs = [g.alloc(4, 4), g.alloc(6, 4)];
    for (i, (addr, len)) in [(bufs[0], 4u32), (bufs[1], 6u32)].into_iter().enumerate() {
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&len.to_le_bytes());
        d.extend_from_slice(&(5 + i as u16).to_le_bytes());
        d.extend_from_slice(&(AVAIL | WRITE).to_le_bytes());
        g.write_mem(desc + 16 * i as u64, &d);
    }
    g.writel(VIRTIO_MMIO_QUEUE_NOTIFY, 0);

    for (i, len) in [(0u64, 4u32), (1, 6)] {
        let d = g.read_mem(desc + 16 * i, 16);
        let used_len = u32::from_le_bytes(d[8..12].try_into().unwrap());
        let id = u16::from_le_bytes(d[12..14].try_into().unwrap());
        let flags = u16::from_le_bytes(d[14..16].try_into().unwrap());
        assert_eq!(id, 5 + i as u16);
        assert_eq!(used_len, len);
        assert_eq!(flags & (AVAIL | USED), AVAIL | USED);
    }
    assert_eq!(g.read_mem(bufs[0], 4), [0, 1, 2, 3]);
    assert_eq!(g.read_mem(bufs[1], 6), [4, 5, 6, 7, 8, 9]);
    assert!(g.irq());
}
