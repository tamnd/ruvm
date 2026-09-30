// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio-mmio register interface, driven through the address space like a guest would.
//! The checks follow `tests/qtest/libqos/virtio-mmio.c` and `hw/virtio/virtio-mmio.c`.

mod common;

use common::{Buf, Guest};
use ruvm_hw_virtio::mmio::*;
use ruvm_hw_virtio::rng::{VIRTIO_ID_RNG, VIRTIO_RNG_QUEUE_SIZE};
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{EntropySource, VirtioDeviceClass, VirtioRng, VirtioRngConf};

/// Hands out 0, 1, 2, ... so filled buffers can be checked.
#[derive(Debug, Default)]
struct Counter(u8);

impl EntropySource for Counter {
    fn fill(&mut self, buf: &mut [u8]) -> usize {
        for b in buf.iter_mut() {
            *b = self.0;
            self.0 = self.0.wrapping_add(1);
        }
        buf.len()
    }
}

fn rng() -> Option<Box<dyn VirtioDeviceClass>> {
    Some(Box::new(VirtioRng::new(Box::new(Counter::default()), VirtioRngConf::default())))
}

#[test]
fn identification_registers() {
    for legacy in [true, false] {
        let g = Guest::new(legacy, rng());
        assert_eq!(g.readl(VIRTIO_MMIO_MAGIC_VALUE), VIRT_MAGIC);
        assert_eq!(
            g.readl(VIRTIO_MMIO_VERSION),
            if legacy { VIRT_VERSION_LEGACY } else { VIRT_VERSION }
        );
        assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), u32::from(VIRTIO_ID_RNG));
        assert_eq!(g.readl(VIRTIO_MMIO_VENDOR_ID), VIRT_VENDOR);
        assert_eq!(g.status(), 0);
        assert_eq!(g.isr(), 0);
        assert!(!g.irq());
    }
}

#[test]
fn force_legacy_defaults_to_on() {
    const { assert!(VIRTIO_MMIO_FORCE_LEGACY_DEFAULT) };
    let g = Guest::new(VIRTIO_MMIO_FORCE_LEGACY_DEFAULT, rng());
    assert!(g.mmio.is_legacy());
}

#[test]
fn empty_transport_only_answers_identification() {
    for legacy in [true, false] {
        let g = Guest::new(legacy, None);
        assert_eq!(g.readl(VIRTIO_MMIO_MAGIC_VALUE), VIRT_MAGIC);
        assert_eq!(
            g.readl(VIRTIO_MMIO_VERSION),
            if legacy { VIRT_VERSION_LEGACY } else { VIRT_VERSION }
        );
        assert_eq!(g.readl(VIRTIO_MMIO_VENDOR_ID), VIRT_VENDOR);
        // Device ID 0 tells the guest there is nothing here.
        assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), 0);
        assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
        g.writel(VIRTIO_MMIO_STATUS, 7);
        assert_eq!(g.status(), 0);
        assert_eq!(g.config_readl(0), 0);
    }
}

#[test]
fn modern_features_have_version_1_and_hide_legacy_bits() {
    let g = Guest::new(false, rng());
    let f = g.device_features();
    assert!(has_feature(f, VIRTIO_F_VERSION_1));
    assert!(has_feature(f, VIRTIO_RING_F_EVENT_IDX));
    assert!(has_feature(f, VIRTIO_RING_F_INDIRECT_DESC));
    assert_eq!(f & VIRTIO_LEGACY_FEATURES, 0);
}

#[test]
fn legacy_features_are_32_bits_and_include_legacy_bits() {
    let g = Guest::new(true, rng());
    g.writel(VIRTIO_MMIO_DEVICE_FEATURES_SEL, 0);
    let lo = g.readl(VIRTIO_MMIO_DEVICE_FEATURES);
    assert_ne!(lo & (1 << VIRTIO_F_NOTIFY_ON_EMPTY), 0);
    assert_ne!(lo & (1 << VIRTIO_F_ANY_LAYOUT), 0);
    g.writel(VIRTIO_MMIO_DEVICE_FEATURES_SEL, 1);
    assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_FEATURES), 0);
}

#[test]
fn queue_num_max_and_queue_sel() {
    let g = Guest::new(false, rng());
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_NUM_MAX), u32::from(VIRTIO_RNG_QUEUE_SIZE));
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 1);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_NUM_MAX), 0);
    // Out of range selections are ignored, the previous one stays.
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
    g.writel(VIRTIO_MMIO_QUEUE_SEL, VIRTIO_QUEUE_MAX as u32);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_NUM_MAX), u32::from(VIRTIO_RNG_QUEUE_SIZE));
}

#[test]
fn registers_need_32_bit_accesses() {
    let g = Guest::new(false, rng());
    assert_eq!(g.load(VIRTIO_MMIO_MAGIC_VALUE, 2), 0);
    assert_eq!(g.load(VIRTIO_MMIO_MAGIC_VALUE, 1), 0);
    g.store(VIRTIO_MMIO_STATUS, 1, u64::from(VIRTIO_CONFIG_S_ACKNOWLEDGE));
    assert_eq!(g.status(), 0);
    g.store(VIRTIO_MMIO_STATUS, 4, u64::from(VIRTIO_CONFIG_S_ACKNOWLEDGE));
    assert_eq!(g.status(), VIRTIO_CONFIG_S_ACKNOWLEDGE);
}

#[test]
fn modern_only_registers() {
    let g = Guest::new(false, rng());
    assert_eq!(g.readl(VIRTIO_MMIO_SHM_LEN_LOW), u32::MAX);
    assert_eq!(g.readl(VIRTIO_MMIO_SHM_LEN_HIGH), u32::MAX);
    assert_eq!(g.readl(VIRTIO_MMIO_CONFIG_GENERATION), 0);
    // The legacy PFN register does not exist in the modern layout.
    g.writel(VIRTIO_MMIO_QUEUE_PFN, 5);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_PFN), 0);

    let g = Guest::new(true, rng());
    g.writel(VIRTIO_MMIO_QUEUE_READY, 1);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_READY), 0);
}

#[test]
fn legacy_pfn_round_trips_and_zero_resets() {
    let mut g = Guest::new(true, rng());
    g.negotiate(!0);
    let q = g.setup_queue(0, 0);
    g.driver_ok();
    assert_eq!(u64::from(g.readl(VIRTIO_MMIO_QUEUE_PFN)), q.desc / common::PAGE_SIZE);
    g.writel(VIRTIO_MMIO_QUEUE_PFN, 0);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_PFN), 0);
    assert_eq!(g.status(), 0);
}

#[test]
fn modern_queue_ready_round_trips() {
    let mut g = Guest::new(false, rng());
    g.negotiate(!0);
    let _q = g.setup_queue(0, 0);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_READY), 1);
    g.writel(VIRTIO_MMIO_QUEUE_READY, 0);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_READY), 0);
}

#[test]
fn unsupported_driver_features_are_dropped() {
    // QEMU masks what the device does not offer and still accepts FEATURES_OK.
    let g = Guest::new(false, rng());
    g.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
    g.set_driver_features(feature(VIRTIO_F_VERSION_1) | 1);
    g.set_status(
        VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER | VIRTIO_CONFIG_S_FEATURES_OK,
    );
    assert_ne!(g.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);
    let guest = g.mmio.with_backend(|b| b.vdev().guest_features()).unwrap();
    assert_eq!(guest, feature(VIRTIO_F_VERSION_1));
}

#[test]
fn features_ok_refused_without_iommu_platform() {
    let g = Guest::with_backend(false, rng(), |b| {
        b.vdev_mut().set_host_feature(VIRTIO_F_IOMMU_PLATFORM, true);
    })
    .unwrap();
    assert!(has_feature(g.device_features(), VIRTIO_F_IOMMU_PLATFORM));
    g.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
    g.set_driver_features(feature(VIRTIO_F_VERSION_1));
    g.set_status(
        VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER | VIRTIO_CONFIG_S_FEATURES_OK,
    );
    assert_eq!(g.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);

    // Accepting it works.
    g.set_driver_features(feature(VIRTIO_F_VERSION_1) | feature(VIRTIO_F_IOMMU_PLATFORM));
    g.set_status(
        VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER | VIRTIO_CONFIG_S_FEATURES_OK,
    );
    assert_ne!(g.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);
}

#[test]
fn features_are_frozen_after_features_ok() {
    let g = Guest::new(false, rng());
    let f = g.negotiate(feature(VIRTIO_F_VERSION_1));
    assert_eq!(f, feature(VIRTIO_F_VERSION_1));
    let dropped = g.mmio.with_backend(|b| b.set_features(!0)).unwrap();
    assert!(dropped.is_err());
    let guest = g.mmio.with_backend(|b| b.vdev().guest_features()).unwrap();
    assert_eq!(guest, feature(VIRTIO_F_VERSION_1));
}

#[test]
fn legacy_driver_features_are_masked() {
    let g = Guest::new(true, rng());
    g.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
    g.writel(VIRTIO_MMIO_DRIVER_FEATURES_SEL, 0);
    g.writel(VIRTIO_MMIO_DRIVER_FEATURES, u32::MAX);
    let host = g.mmio.with_backend(|b| b.vdev().host_features()).unwrap();
    let guest = g.mmio.with_backend(|b| b.vdev().guest_features()).unwrap();
    assert_eq!(guest, host & 0xffff_ffff);
}

#[test]
fn oversized_queue_needs_reset() {
    let g = Guest::new(false, rng());
    g.negotiate(!0);
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
    // Above VIRTQUEUE_MAX_SIZE the write is silently ignored.
    g.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(VIRTQUEUE_MAX_SIZE) + 1);
    g.writel(VIRTIO_MMIO_QUEUE_READY, 1);
    assert_eq!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
    // Above what the device offers is a device error.
    g.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(VIRTIO_RNG_QUEUE_SIZE) * 2);
    g.writel(VIRTIO_MMIO_QUEUE_READY, 1);
    assert_ne!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
    // The config interrupt that goes with it waits for DRIVER_OK, which is not set yet.
    assert!(!g.irq());
}

#[test]
fn config_space_out_of_range_reads_all_ones() {
    // virtio-rng has no config space at all.
    let g = Guest::new(false, rng());
    assert_eq!(g.config_readl(0), u32::MAX);
    assert_eq!(g.config_readw(0), u16::MAX);
    assert_eq!(g.config_readb(0), u8::MAX);
}

#[test]
fn used_buffer_raises_irq_and_ack_lowers_it() {
    for legacy in [true, false] {
        let mut g = Guest::new(legacy, rng());
        g.negotiate(!feature(VIRTIO_RING_F_EVENT_IDX));
        let mut q = g.setup_queue(0, 0);
        g.driver_ok();
        let buf = g.alloc(16, 16);
        q.submit(&g, &[Buf::inp(buf, 16)]);
        g.kick(&q);
        assert_eq!(q.get_used(&g).map(|e| e.1), Some(16));
        assert!(g.irq());
        assert_eq!(g.isr() & VIRTIO_MMIO_INT_VRING, VIRTIO_MMIO_INT_VRING);
        g.ack();
        assert!(!g.irq());
        assert_eq!(g.isr(), 0);
    }
}

#[test]
fn status_zero_resets_the_device() {
    for legacy in [true, false] {
        let mut g = Guest::new(legacy, rng());
        g.negotiate(!feature(VIRTIO_RING_F_EVENT_IDX));
        let mut q = g.setup_queue(0, 0);
        g.driver_ok();
        let buf = g.alloc(4, 4);
        q.submit(&g, &[Buf::inp(buf, 4)]);
        g.kick(&q);
        assert!(g.irq());

        g.set_status(0);
        assert_eq!(g.status(), 0);
        assert_eq!(g.isr(), 0);
        assert!(!g.irq());
        g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
        if legacy {
            assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_PFN), 0);
        } else {
            assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_READY), 0);
        }
        let guest = g.mmio.with_backend(|b| b.vdev().guest_features()).unwrap();
        assert_eq!(guest, 0);
    }
}

#[test]
fn transport_reset_clears_everything() {
    let mut g = Guest::new(false, rng());
    g.negotiate(!0);
    g.setup_queue(0, 0);
    g.driver_ok();
    g.mmio.reset();
    assert_eq!(g.status(), 0);
    g.writel(VIRTIO_MMIO_QUEUE_SEL, 0);
    assert_eq!(g.readl(VIRTIO_MMIO_QUEUE_READY), 0);
}

#[test]
fn notify_for_missing_queue_is_ignored() {
    let mut g = Guest::new(false, rng());
    g.negotiate(!0);
    g.setup_queue(0, 0);
    g.driver_ok();
    g.writel(VIRTIO_MMIO_QUEUE_NOTIFY, 5);
    g.writel(VIRTIO_MMIO_QUEUE_NOTIFY, 0xffff);
    assert_eq!(g.status() & VIRTIO_CONFIG_S_NEEDS_RESET, 0);
    assert!(!g.irq());
}

#[test]
fn indirect_descriptors() {
    let mut g = Guest::new(false, rng());
    g.negotiate(!feature(VIRTIO_RING_F_EVENT_IDX));
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    let a = g.alloc(3, 1);
    let b = g.alloc(5, 1);
    let head = q.add_indirect(&mut g, &[Buf::inp(a, 3), Buf::inp(b, 5)]);
    q.make_available(&g, head);
    g.kick(&q);
    assert_eq!(q.get_used(&g), Some((u32::from(head), 8)));
    assert_eq!(g.read_mem(a, 3), [0, 1, 2]);
    assert_eq!(g.read_mem(b, 5), [3, 4, 5, 6, 7]);
}

#[test]
fn event_idx_suppresses_interrupts_until_used_event() {
    let mut g = Guest::new(false, rng());
    let f = g.negotiate(!0);
    assert!(has_feature(f, VIRTIO_RING_F_EVENT_IDX));
    let mut q = g.setup_queue(0, 0);
    g.driver_ok();
    let a = g.alloc(4, 4);
    // The first used buffer always interrupts, since nothing was signalled yet.
    q.submit(&g, &[Buf::inp(a, 4)]);
    g.kick(&q);
    assert!(q.get_used(&g).is_some());
    assert!(g.irq());
    g.ack();
    // used_event = 2: no interrupt when used idx goes to 2, one when it goes to 3.
    q.set_used_event(&g, 2);
    q.submit(&g, &[Buf::inp(a, 4)]);
    g.kick(&q);
    assert!(q.get_used(&g).is_some());
    assert!(!g.irq());
    q.submit(&g, &[Buf::inp(a, 4)]);
    g.kick(&q);
    assert!(q.get_used(&g).is_some());
    assert!(g.irq());
}

#[test]
fn packed_ring_is_offered_when_enabled() {
    let g = Guest::with_backend(false, rng(), |b| {
        b.vdev_mut().set_host_feature(34, true);
    })
    .unwrap();
    assert!(has_feature(g.device_features(), 34));
}
