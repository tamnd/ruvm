// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-balloon through virtio-mmio: the checks from `tests/qtest/virtio-balloon-test.c`,
//! plus inflate and deflate, statistics, free page hints and free page reporting.

mod common;

use common::{Buf, Guest, RAM_SIZE};
use ruvm_hw_virtio::balloon::*;
use ruvm_hw_virtio::mmio::{VIRTIO_MMIO_CONFIG, VIRTIO_MMIO_DEVICE_ID};
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{BalloonOp, RecordingBalloonBackend, VirtioBalloon};

fn features() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

fn guest(legacy: bool, balloon: VirtioBalloon) -> Guest {
    Guest::new(legacy, Some(Box::new(balloon)))
}

fn backend() -> RecordingBalloonBackend {
    RecordingBalloonBackend::new().with_ram_size(RAM_SIZE)
}

fn dev<R>(g: &Guest, f: impl FnOnce(&mut VirtIODevice, &mut VirtioBalloon) -> R) -> R {
    g.mmio.with_device(f).unwrap()
}

/// Puts `pfns` in one buffer on `ring` and kicks it.
fn send_pfns(g: &mut Guest, ring: &mut common::SplitRing, pfns: &[u32]) -> u16 {
    let bytes: Vec<u8> = pfns.iter().flat_map(|p| p.to_le_bytes()).collect();
    let addr = g.alloc(bytes.len() as u64, 8);
    g.write_mem(addr, &bytes);
    let head = ring.submit(g, &[Buf::out(addr, bytes.len() as u32)]);
    g.kick(ring);
    head
}

fn stats_buffer(g: &mut Guest, stats: &[(u16, u64)]) -> (u64, u32) {
    let bytes: Vec<u8> = stats
        .iter()
        .flat_map(|(tag, val)| tag.to_le_bytes().into_iter().chain(val.to_le_bytes()))
        .collect();
    let addr = g.alloc(bytes.len() as u64, 8);
    g.write_mem(addr, &bytes);
    (addr, bytes.len() as u32)
}

#[test]
fn identity_features_and_config() {
    for legacy in [true, false] {
        let g = guest(legacy, VirtioBalloon::new(RAM_SIZE, Box::new(backend())));
        assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), 5);
        let f = g.device_features();
        assert!(has_feature(f, VIRTIO_BALLOON_F_STATS_VQ));
        assert!(has_feature(f, VIRTIO_BALLOON_F_PAGE_POISON));
        for bit in [
            VIRTIO_BALLOON_F_DEFLATE_ON_OOM,
            VIRTIO_BALLOON_F_FREE_PAGE_HINT,
            VIRTIO_BALLOON_F_REPORTING,
        ] {
            assert!(!has_feature(f, bit));
        }
        assert_eq!(dev(&g, |vdev, _| (vdev.num_queues(), vdev.config_len())), (3, 16));
        assert_eq!(g.config_readl(0), 0);
        assert_eq!(g.config_readl(4), 0);
        assert_eq!(g.config_readl(8), VIRTIO_BALLOON_CMD_ID_STOP);
    }

    let mut b = VirtioBalloon::new(RAM_SIZE, Box::new(backend()));
    b.set_page_poison(false);
    b.set_deflate_on_oom(true);
    let g = guest(false, b);
    let f = g.device_features();
    assert!(has_feature(f, VIRTIO_BALLOON_F_DEFLATE_ON_OOM));
    assert!(!has_feature(f, VIRTIO_BALLOON_F_PAGE_POISON));
    assert_eq!(dev(&g, |vdev, _| vdev.config_len()), 8);

    let mut b = VirtioBalloon::new(RAM_SIZE, Box::new(backend()));
    b.set_page_poison(false);
    b.set_free_page_hint(true);
    let g = guest(false, b);
    assert_eq!(dev(&g, |vdev, _| vdev.config_len()), 12);
}

/// `query_stats()`: one entry for each statistic the kernel knows, all unset.
#[test]
fn query_stats() {
    let g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(backend())));
    let stats = dev(&g, |_, b| b.guest_stats());
    assert_eq!(stats.len(), VIRTIO_BALLOON_S_NR);
    assert_eq!(stats[0], ("stat-swap-in", u64::MAX));
    assert_eq!(stats[15], ("stat-direct-reclaims", u64::MAX));
    assert!(stats.iter().all(|(_, v)| *v == u64::MAX));
    assert_eq!(dev(&g, |_, b| b.stats_last_update()), 0);
}

#[test]
fn target_actual_and_query() {
    for legacy in [true, false] {
        let g = guest(legacy, VirtioBalloon::new(RAM_SIZE, Box::new(backend())));
        g.negotiate(features());
        g.driver_ok();
        dev(&g, |vdev, b| b.to_target(vdev, RAM_SIZE - (1 << 20)));
        assert_eq!(g.config_readl(0), 256);
        assert_ne!(g.isr() & 2, 0);
        g.ack();

        // 0 is ignored, and anything above RAM means an empty balloon.
        dev(&g, |vdev, b| b.to_target(vdev, 0));
        assert_eq!(g.config_readl(0), 256);
        assert_eq!(g.isr(), 0);
        dev(&g, |vdev, b| b.to_target(vdev, RAM_SIZE * 2));
        assert_eq!(g.config_readl(0), 0);

        assert_eq!(dev(&g, |_, b| b.query()), RAM_SIZE);
        g.store(VIRTIO_MMIO_CONFIG + 4, 4, 256);
        assert_eq!(dev(&g, |_, b| (b.actual(), b.query())), (256, RAM_SIZE - (1 << 20)));
        assert_eq!(g.config_readl(4), 256);
        g.store(VIRTIO_MMIO_CONFIG + 4, 4, 256);
        assert_eq!(dev(&g, |_, b| b.take_balloon_change_events()), vec![RAM_SIZE - (1 << 20)]);
    }
}

#[test]
fn inflate_and_deflate_pfn_batches() {
    for legacy in [true, false] {
        let rec = backend();
        let mut g = guest(legacy, VirtioBalloon::new(RAM_SIZE, Box::new(rec.clone())));
        g.negotiate(features());
        let mut ivq = g.setup_queue(BALLOON_IVQ, 0);
        let mut dvq = g.setup_queue(BALLOON_DVQ, 0);
        g.driver_ok();

        // 0x5000 is past the end of RAM and skipped.
        let head = send_pfns(&mut g, &mut ivq, &[0x100, 0x101, 0x5000, 0x3ff]);
        assert_eq!(ivq.get_used(&g), Some((u32::from(head), 0)));
        assert_eq!(
            rec.take_ops(),
            vec![
                BalloonOp::Discard(0x10_0000, 4096),
                BalloonOp::Discard(0x10_1000, 4096),
                BalloonOp::Discard(0x3f_f000, 4096),
            ]
        );

        let head = send_pfns(&mut g, &mut dvq, &[0x100, 0x101]);
        assert_eq!(dvq.get_used(&g), Some((u32::from(head), 0)));
        assert_eq!(
            rec.take_ops(),
            vec![BalloonOp::Populate(0x10_0000, 4096), BalloonOp::Populate(0x10_1000, 4096)]
        );

        // A trailing partial PFN is ignored, and several buffers are handled in one kick.
        let a = g.alloc(6, 8);
        g.write_mem(a, &[0x10, 0x01, 0, 0, 0xff, 0xff]);
        ivq.submit(&g, &[Buf::out(a, 6)]);
        send_pfns(&mut g, &mut ivq, &[0x120]);
        assert_eq!(std::iter::from_fn(|| ivq.get_used(&g)).count(), 2);
        assert_eq!(
            rec.take_ops(),
            vec![BalloonOp::Discard(0x11_0000, 4096), BalloonOp::Discard(0x12_0000, 4096)]
        );
    }
}

#[test]
fn inflate_with_discard_inhibited() {
    let rec = backend().with_discard_inhibited(true);
    let mut g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(rec.clone())));
    g.negotiate(features());
    let mut ivq = g.setup_queue(BALLOON_IVQ, 0);
    g.driver_ok();
    let head = send_pfns(&mut g, &mut ivq, &[0x100]);
    assert_eq!(ivq.get_used(&g), Some((u32::from(head), 0)));
    assert!(rec.ops().is_empty());
}

#[test]
fn inflate_with_big_host_pages() {
    let rec = backend().with_host_page_size(0x4000);
    let mut g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(rec.clone())));
    g.negotiate(features());
    let mut ivq = g.setup_queue(BALLOON_IVQ, 0);
    let mut dvq = g.setup_queue(BALLOON_DVQ, 0);
    g.driver_ok();

    // Three quarters of a host page, then a page elsewhere: nothing is discarded.
    send_pfns(&mut g, &mut ivq, &[0x100, 0x101, 0x102, 0x200]);
    assert!(rec.take_ops().is_empty());
    // The whole host page, in any order.
    send_pfns(&mut g, &mut ivq, &[0x203, 0x201, 0x200, 0x202]);
    assert_eq!(rec.take_ops(), vec![BalloonOp::Discard(0x20_0000, 0x4000)]);
    // Pieces of one host page in two buffers do not add up.
    send_pfns(&mut g, &mut ivq, &[0x300, 0x301]);
    send_pfns(&mut g, &mut ivq, &[0x302, 0x303]);
    assert!(rec.take_ops().is_empty());

    send_pfns(&mut g, &mut dvq, &[0x301]);
    assert_eq!(rec.take_ops(), vec![BalloonOp::Populate(0x30_0000, 0x4000)]);
}

#[test]
fn stats_flow() {
    let rec = backend();
    let mut g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(rec)));
    let accepted = g.negotiate(features());
    assert!(has_feature(accepted, VIRTIO_BALLOON_F_STATS_VQ));
    let mut svq = g.setup_queue(BALLOON_SVQ, 0);
    g.driver_ok();

    // The polling interval property.
    dev(&g, |_, b| {
        assert_eq!(b.stats_timer(), None);
        assert_eq!(
            b.set_stats_poll_interval(-1).unwrap_err().message(),
            "timer value must be greater than zero"
        );
        assert_eq!(
            b.set_stats_poll_interval(i64::from(u32::MAX) + 1).unwrap_err().message(),
            "timer value is too big"
        );
    });

    // The guest hands over its first statistics and the device keeps the buffer.
    let (addr, len) = stats_buffer(&mut g, &[(4, 1234), (5, 4096), (99, 7)]);
    let head = svq.submit(&g, &[Buf::out(addr, len)]);
    g.kick(&svq);
    assert_eq!(svq.used_idx(&g), 0);
    let stats = dev(&g, |_, b| b.guest_stats());
    assert_eq!(stats[4], ("stat-free-memory", 1234));
    assert_eq!(stats[5], ("stat-total-memory", 4096));
    assert_eq!(stats[0].1, u64::MAX);
    assert!(dev(&g, |_, b| b.stats_last_update()) > 0);
    // Polling is off, so no timer.
    assert_eq!(dev(&g, |_, b| b.stats_timer()), None);

    // Turning polling on fires right away and gives the buffer back.
    dev(&g, |_, b| b.set_stats_poll_interval(2).unwrap());
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(0));
    dev(&g, |vdev, b| b.stats_poll(vdev));
    assert_eq!(svq.get_used(&g), Some((u32::from(head), 0)));
    assert_eq!(dev(&g, |_, b| b.stats_timer()), None);

    // The guest refills it, which re-arms the timer with the interval. Old values are gone.
    let (addr, len) = stats_buffer(&mut g, &[(4, 99)]);
    let head = svq.submit(&g, &[Buf::out(addr, len)]);
    g.kick(&svq);
    let stats = dev(&g, |_, b| b.guest_stats());
    assert_eq!((stats[4].1, stats[5].1), (99, u64::MAX));
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(2));

    // A new interval re-arms, the same one does nothing, 0 stops.
    dev(&g, |_, b| b.set_stats_poll_interval(5).unwrap());
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(5));
    dev(&g, |_, b| b.set_stats_poll_interval(5).unwrap());
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(5));
    dev(&g, |vdev, b| b.stats_poll(vdev));
    assert_eq!(svq.get_used(&g), Some((u32::from(head), 0)));

    // A poll without a buffer just re-arms.
    dev(&g, |_, b| b.set_stats_poll_interval(0).unwrap());
    dev(&g, |_, b| b.set_stats_poll_interval(3).unwrap());
    dev(&g, |vdev, b| b.stats_poll(vdev));
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(3));
    dev(&g, |_, b| b.set_stats_poll_interval(0).unwrap());
    assert_eq!(dev(&g, |_, b| (b.stats_timer(), b.stats_poll_interval())), (None, 0));
}

#[test]
fn reset_drops_the_held_stats_buffer() {
    let mut g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(backend())));
    g.negotiate(features());
    let mut svq = g.setup_queue(BALLOON_SVQ, 0);
    g.driver_ok();
    let (addr, len) = stats_buffer(&mut g, &[(4, 1)]);
    svq.submit(&g, &[Buf::out(addr, len)]);
    g.kick(&svq);
    dev(&g, |_, b| b.set_stats_poll_interval(1).unwrap());
    g.set_status(0);
    // Nothing is held any more, so a poll only re-arms.
    dev(&g, |vdev, b| b.stats_poll(vdev));
    assert_eq!(dev(&g, |_, b| b.stats_timer()), Some(1));
}

#[test]
fn free_page_hints() {
    let rec = backend();
    let mut b = VirtioBalloon::new(RAM_SIZE, Box::new(rec.clone()));
    b.set_free_page_hint(true);
    let mut g = guest(false, b);
    let accepted = g.negotiate(features());
    assert!(has_feature(accepted, VIRTIO_BALLOON_F_FREE_PAGE_HINT));
    assert_eq!(dev(&g, |vdev, b| (b.free_page_vq(), vdev.queue_num_max(3))), (Some(3), 1024));
    let mut fvq = g.setup_queue(3, 0);
    g.driver_ok();

    g.ack();
    dev(&g, |vdev, b| b.free_page_start(vdev));
    let id = VIRTIO_BALLOON_FREE_PAGE_HINT_CMD_ID_MIN + 1;
    assert_eq!(g.config_readl(8), id);
    assert_ne!(g.isr() & 2, 0);

    // Hints before the guest acknowledges the command are not passed on.
    let early = g.alloc(0x1000, 0x1000);
    fvq.submit(&g, &[Buf::inp(early, 0x1000)]);
    g.kick(&fvq);
    assert_eq!(fvq.used_idx(&g), 1);
    assert!(rec.take_ops().is_empty());

    // A stale id does not start hinting.
    let cmd = g.alloc(4, 4);
    g.write_mem(cmd, &(id - 1).to_le_bytes());
    fvq.submit(&g, &[Buf::out(cmd, 4)]);
    g.kick(&fvq);
    assert_eq!(dev(&g, |_, b| b.free_page_hint_status()), FreePageHintStatus::Requested);

    // The right id starts it, and hints in the same and later buffers are passed on.
    let cmd = g.alloc(4, 4);
    g.write_mem(cmd, &id.to_le_bytes());
    let pages = g.alloc(0x2000, 0x1000);
    fvq.submit(&g, &[Buf::out(cmd, 4), Buf::inp(pages, 0x1000)]);
    fvq.submit(&g, &[Buf::inp(pages + 0x1000, 0x1000)]);
    g.kick(&fvq);
    assert_eq!(dev(&g, |_, b| b.free_page_hint_status()), FreePageHintStatus::Start);
    assert_eq!(g.config_readl(8), 0);
    assert_eq!(
        rec.take_ops(),
        vec![
            BalloonOp::FreePageHint(pages, 0x1000),
            BalloonOp::FreePageHint(pages + 0x1000, 0x1000)
        ]
    );
    assert_eq!(fvq.used_idx(&g), 4);

    // Any id stops it.
    let cmd = g.alloc(4, 4);
    g.write_mem(cmd, &VIRTIO_BALLOON_CMD_ID_STOP.to_le_bytes());
    fvq.submit(&g, &[Buf::out(cmd, 4)]);
    g.kick(&fvq);
    assert_eq!(dev(&g, |_, b| b.free_page_hint_status()), FreePageHintStatus::Stop);
    assert_eq!(g.config_readl(8), VIRTIO_BALLOON_CMD_ID_STOP);

    g.ack();
    dev(&g, |vdev, b| b.free_page_done(vdev));
    assert_eq!(g.config_readl(8), VIRTIO_BALLOON_CMD_ID_DONE);
    assert_ne!(g.isr() & 2, 0);
    g.ack();
    dev(&g, |vdev, b| b.free_page_done(vdev));
    assert_eq!(g.isr(), 0, "no change, no interrupt");
    dev(&g, |vdev, b| b.free_page_stop(vdev));
    assert_eq!(g.config_readl(8), VIRTIO_BALLOON_CMD_ID_STOP);

    // Stopping and finishing keep the command id.
    dev(&g, |_, b| assert_eq!(b.free_page_hint_cmd_id(), id));

    // A short command id breaks the device.
    let cmd = g.alloc(4, 4);
    fvq.submit(&g, &[Buf::out(cmd, 2)]);
    g.kick(&fvq);
    assert!(dev(&g, |vdev, _| vdev.is_broken()));
}

#[test]
fn free_page_reporting() {
    for poison in [0u32, 0xaa] {
        let rec = backend();
        let mut b = VirtioBalloon::new(RAM_SIZE, Box::new(rec.clone()));
        b.set_free_page_reporting(true);
        let mut g = guest(false, b);
        g.negotiate(features());
        assert_eq!(dev(&g, |vdev, b| (b.reporting_vq(), vdev.queue_num_max(3))), (Some(3), 32));
        let mut rvq = g.setup_queue(3, 0);
        g.store(VIRTIO_MMIO_CONFIG + 12, 4, u64::from(poison));
        g.driver_ok();
        assert_eq!(dev(&g, |_, b| b.poison_val()), poison);

        let head = rvq.submit(
            &g,
            &[
                Buf::inp(0x10_0000, 0x4000),
                // Not page aligned.
                Buf::inp(0x20_0800, 0x1000),
                Buf::inp(0x20_0000, 0x800),
                // Past the end of RAM.
                Buf::inp(RAM_SIZE - 0x1000, 0x2000),
            ],
        );
        g.kick(&rvq);
        assert_eq!(rvq.get_used(&g), Some((u32::from(head), 0)));
        let want = if poison == 0 { vec![BalloonOp::Discard(0x10_0000, 0x4000)] } else { vec![] };
        assert_eq!(rec.take_ops(), want);
    }
}

#[test]
fn poison_value_needs_the_feature() {
    let g = guest(false, VirtioBalloon::new(RAM_SIZE, Box::new(backend())));
    g.negotiate(features() & !feature(VIRTIO_BALLOON_F_PAGE_POISON));
    g.store(VIRTIO_MMIO_CONFIG + 12, 4, 0x55);
    assert_eq!(dev(&g, |_, b| b.poison_val()), 0);

    g.negotiate(features());
    g.store(VIRTIO_MMIO_CONFIG + 12, 4, 0x55);
    assert_eq!(dev(&g, |_, b| b.poison_val()), 0x55);
    assert_eq!(g.config_readl(12), 0x55);
    // Reset clears it.
    g.set_status(0);
    assert_eq!(dev(&g, |_, b| b.poison_val()), 0);
}
