// SPDX-License-Identifier: GPL-2.0-or-later

//! The single port virtio-console through virtio-mmio.

mod common;

use std::sync::{Arc, Mutex};

use common::{Buf, Guest};
use ruvm_hw_virtio::console::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{ConsoleBackend, VirtioConsole, VirtioDeviceClass};

/// Collects what the guest prints.
#[derive(Debug, Clone, Default)]
struct Sink {
    out: Arc<Mutex<Vec<u8>>>,
    writable: Arc<Mutex<usize>>,
}

impl Sink {
    fn output(&self) -> Vec<u8> {
        self.out.lock().unwrap().clone()
    }
}

impl ConsoleBackend for Sink {
    fn write(&mut self, buf: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(buf);
        buf.len()
    }

    fn guest_writable(&mut self) {
        *self.writable.lock().unwrap() += 1;
    }
}

fn console(sink: Option<&Sink>) -> Option<Box<dyn VirtioDeviceClass>> {
    let backend = sink.map(|s| Box::new(s.clone()) as Box<dyn ConsoleBackend>);
    Some(Box::new(VirtioConsole::new(backend)))
}

fn features() -> u64 {
    !feature(VIRTIO_RING_F_EVENT_IDX)
}

#[test]
fn identity_features_and_config() {
    for legacy in [true, false] {
        let g = Guest::new(legacy, console(None));
        assert_eq!(g.readl(ruvm_hw_virtio::mmio::VIRTIO_MMIO_DEVICE_ID), 3);
        let f = g.device_features();
        assert!(has_feature(f, VIRTIO_CONSOLE_F_EMERG_WRITE));
        assert!(!has_feature(f, VIRTIO_CONSOLE_F_MULTIPORT));
        assert!(!has_feature(f, VIRTIO_CONSOLE_F_SIZE));
        // cols, rows, max_nr_ports, emerg_wr
        assert_eq!(g.config_readw(0), 0);
        assert_eq!(g.config_readw(2), 0);
        assert_eq!(g.config_readl(4), 1);
        assert_eq!(g.config_readl(8), 0);
        assert_eq!(g.config_readl(12), u32::MAX);
    }
}

#[test]
fn emergency_write_can_be_turned_off() {
    let mut dev = VirtioConsole::new(None);
    dev.set_emergency_write(false);
    let g = Guest::new(false, Some(Box::new(dev)));
    assert!(!has_feature(g.device_features(), VIRTIO_CONSOLE_F_EMERG_WRITE));
}

#[test]
fn guest_output_reaches_the_backend() {
    for legacy in [true, false] {
        let sink = Sink::default();
        let mut g = Guest::new(legacy, console(Some(&sink)));
        g.negotiate(features());
        let _rx = g.setup_queue(VIRTIO_CONSOLE_RX_QUEUE, 0);
        let mut tx = g.setup_queue(VIRTIO_CONSOLE_TX_QUEUE, 0);
        g.driver_ok();

        let a = g.alloc(6, 1);
        let b = g.alloc(7, 1);
        g.write_mem(a, b"Hello,");
        g.write_mem(b, b" world!");
        let head = tx.submit(&g, &[Buf::out(a, 6), Buf::out(b, 7)]);
        g.kick(&tx);
        assert_eq!(sink.output(), b"Hello, world!");
        // Output buffers come back with nothing written.
        assert_eq!(tx.get_used(&g), Some((u32::from(head), 0)));
        assert!(g.irq());
    }
}

#[test]
fn large_output_is_passed_on_whole() {
    let sink = Sink::default();
    let mut g = Guest::new(false, console(Some(&sink)));
    g.negotiate(features());
    let mut tx = g.setup_queue(VIRTIO_CONSOLE_TX_QUEUE, 0);
    g.driver_ok();
    let data: Vec<u8> = (0..10000u32).map(|i| (i % 251) as u8).collect();
    let a = g.alloc(data.len() as u64, 1);
    g.write_mem(a, &data);
    tx.submit(&g, &[Buf::out(a, data.len() as u32)]);
    g.kick(&tx);
    assert_eq!(sink.output(), data);
}

#[test]
fn output_without_backend_is_discarded() {
    let mut g = Guest::new(false, console(None));
    g.negotiate(features());
    let mut tx = g.setup_queue(VIRTIO_CONSOLE_TX_QUEUE, 0);
    g.driver_ok();
    let a = g.alloc(4, 1);
    tx.submit(&g, &[Buf::out(a, 4)]);
    g.kick(&tx);
    assert_eq!(tx.get_used(&g).map(|e| e.1), Some(0));
}

#[test]
fn host_input_reaches_the_guest() {
    for legacy in [true, false] {
        let sink = Sink::default();
        let mut g = Guest::new(legacy, console(Some(&sink)));
        g.negotiate(features());
        let mut rx = g.setup_queue(VIRTIO_CONSOLE_RX_QUEUE, 0);
        g.driver_ok();

        let ready = |g: &Guest| {
            g.mmio.with_device(|vdev, d: &mut VirtioConsole| d.guest_ready(vdev)).unwrap()
        };
        assert_eq!(ready(&g), 0);

        let a = g.alloc(4, 1);
        let b = g.alloc(8, 1);
        rx.submit(&g, &[Buf::inp(a, 4)]);
        rx.submit(&g, &[Buf::inp(b, 8)]);
        g.kick(&rx);
        // Kicking the receive queue tells the backend it can send more.
        assert_eq!(*sink.writable.lock().unwrap(), 1);
        assert_eq!(ready(&g), 12);

        let sent = g
            .mmio
            .with_device(|vdev, d: &mut VirtioConsole| d.write_to_guest(vdev, b"abcdefghijklmnop"))
            .unwrap();
        assert_eq!(sent, 12);
        assert_eq!(rx.get_used(&g).unwrap().1, 4);
        assert_eq!(rx.get_used(&g).unwrap().1, 8);
        assert_eq!(g.read_mem(a, 4), b"abcd");
        assert_eq!(g.read_mem(b, 8), b"efghijkl");
        assert!(g.irq());
        assert_eq!(ready(&g), 0);
    }
}

#[test]
fn input_needs_an_open_port() {
    let sink = Sink::default();
    let mut g = Guest::new(false, console(Some(&sink)));
    g.negotiate(features());
    let mut rx = g.setup_queue(VIRTIO_CONSOLE_RX_QUEUE, 0);
    let a = g.alloc(4, 1);
    rx.submit(&g, &[Buf::inp(a, 4)]);
    let write = |g: &Guest| {
        g.mmio.with_device(|vdev, d: &mut VirtioConsole| d.write_to_guest(vdev, b"xy")).unwrap()
    };
    // Before DRIVER_OK the guest side is closed.
    assert_eq!(write(&g), 0);
    let connected =
        |g: &Guest| g.mmio.with_device(|_, d: &mut VirtioConsole| d.guest_connected()).unwrap();
    assert!(!connected(&g));
    g.driver_ok();
    assert!(connected(&g));

    // Host side closed.
    g.mmio.with_device(|_, d: &mut VirtioConsole| d.set_host_connected(false)).unwrap();
    assert_eq!(write(&g), 0);
    g.mmio.with_device(|_, d: &mut VirtioConsole| d.set_host_connected(true)).unwrap();
    assert_eq!(write(&g), 2);
    assert_eq!(rx.get_used(&g).unwrap().1, 2);

    // A reset closes the guest side again.
    g.set_status(0);
    assert!(!connected(&g));
}

#[test]
fn emergency_write() {
    let sink = Sink::default();
    let mut g = Guest::new(false, console(Some(&sink)));
    g.negotiate(features() & !feature(VIRTIO_CONSOLE_F_EMERG_WRITE));
    g.setup_queue(VIRTIO_CONSOLE_TX_QUEUE, 0);
    g.driver_ok();
    // Works whenever the device offers it, like QEMU, which checks the host features.
    g.writel(ruvm_hw_virtio::mmio::VIRTIO_MMIO_CONFIG + 8, u32::from(b'!'));
    g.writel(ruvm_hw_virtio::mmio::VIRTIO_MMIO_CONFIG + 8, 0x100 | u32::from(b'?'));
    assert_eq!(sink.output(), b"!?");
    assert_eq!(g.config_readl(8), 0);
}

#[test]
fn control_output_is_returned_unread() {
    let sink = Sink::default();
    let mut g = Guest::new(false, console(Some(&sink)));
    g.negotiate(features());
    let mut ctrl = g.setup_queue(VIRTIO_CONSOLE_CTRL_TX_QUEUE, 0);
    g.driver_ok();
    let a = g.alloc(8, 1);
    g.write_mem(a, &[0; 8]);
    ctrl.submit(&g, &[Buf::out(a, 8)]);
    g.kick(&ctrl);
    assert_eq!(ctrl.get_used(&g).map(|e| e.1), Some(0));
    assert!(sink.output().is_empty());
}

#[test]
fn queue_sizes() {
    let g = Guest::new(false, console(None));
    for (q, size) in [(0, 128), (1, 128), (2, 32), (3, 32), (4, 0)] {
        g.writel(ruvm_hw_virtio::mmio::VIRTIO_MMIO_QUEUE_SEL, q);
        assert_eq!(g.readl(ruvm_hw_virtio::mmio::VIRTIO_MMIO_QUEUE_NUM_MAX), size);
    }
}
