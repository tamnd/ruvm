// SPDX-License-Identifier: GPL-2.0-or-later

//! A minimal virtio-console: `hw/char/virtio-serial-bus.c` reduced to one `virtconsole` port
//! (`hw/char/virtio-console.c`) without `VIRTIO_CONSOLE_F_MULTIPORT`.
//!
//! This is the shape a guest sees from `-device virtio-serial-device,max_ports=1 -device
//! virtconsole`: the device offers only `VIRTIO_CONSOLE_F_EMERG_WRITE`, reports one port in
//! config space, and port 0 is the console. Queue 0 carries host to guest data and queue 1 guest
//! to host data. Queues 2 and 3 are the control queues, which QEMU creates even without
//! multiport and so do we; a driver that has not negotiated multiport never uses them, and any
//! buffers put on the control output queue are returned unread.
//!
//! Without multiport the guest cannot say when it opens the port, so the port counts as open
//! from `DRIVER_OK` until the status goes back to zero, as in QEMU.
//!
//! Guest output goes to a [`ConsoleBackend`]. Host input is pushed in with
//! [`VirtioConsole::write_to_guest`], typically from a chardev read handler that first asks
//! [`VirtioConsole::guest_ready`] how much room there is.
//!
//! Differences from QEMU: the console never throttles (QEMU's console ports drop what the
//! chardev does not take, and so do we), a port with no backend discards guest output rather
//! than pretending to consume it, and `VIRTIO_CONSOLE_F_SIZE` is not offered.
//!
//! Not ported: multiport and the control message protocol, more than one port, generic
//! `virtserialport` ports, port names, guest open and close events, VMState, trace points and
//! QOM registration.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use ruvm_base::Result;

use crate::virtio::{VIRTIO_CONFIG_S_DRIVER_OK, VirtIODevice, VirtioDeviceClass, feature};

/// `TYPE_VIRTIO_SERIAL`.
pub const TYPE_VIRTIO_SERIAL: &str = "virtio-serial-device";

/// `VIRTIO_ID_CONSOLE`.
pub const VIRTIO_ID_CONSOLE: u16 = 3;

/// `VIRTIO_CONSOLE_F_SIZE`: the device reports the console size.
pub const VIRTIO_CONSOLE_F_SIZE: u32 = 0;
/// `VIRTIO_CONSOLE_F_MULTIPORT`: the device has several ports and a control channel.
pub const VIRTIO_CONSOLE_F_MULTIPORT: u32 = 1;
/// `VIRTIO_CONSOLE_F_EMERG_WRITE`: the guest may write single characters through config space.
pub const VIRTIO_CONSOLE_F_EMERG_WRITE: u32 = 2;

/// Size of `struct virtio_console_config`.
pub const VIRTIO_CONSOLE_CONFIG_SIZE: usize = 12;

/// Queue index of port 0's host to guest queue.
pub const VIRTIO_CONSOLE_RX_QUEUE: u16 = 0;
/// Queue index of port 0's guest to host queue.
pub const VIRTIO_CONSOLE_TX_QUEUE: u16 = 1;
/// Queue index of the control queue from host to guest.
pub const VIRTIO_CONSOLE_CTRL_RX_QUEUE: u16 = 2;
/// Queue index of the control queue from guest to host.
pub const VIRTIO_CONSOLE_CTRL_TX_QUEUE: u16 = 3;

/// Where console output goes, the chardev side of `virtconsole`.
pub trait ConsoleBackend: Send + fmt::Debug {
    /// Writes what the guest printed. Returns how much was taken; the rest is dropped, since a
    /// console must not stall the guest.
    fn write(&mut self, buf: &[u8]) -> usize;

    /// The guest added receive buffers, so input that was held back can be sent now
    /// (`qemu_chr_fe_accept_input()`).
    fn guest_writable(&mut self) {}
}

/// The virtio-console device model, `VirtIOSerial` with a single `VirtConsole` port.
#[derive(Debug)]
pub struct VirtioConsole {
    backend: Option<Box<dyn ConsoleBackend>>,
    emergency_write: bool,
    host_connected: bool,
    guest_connected: bool,
}

impl VirtioConsole {
    /// A console writing to `backend`. With no backend the port is not connected on the host
    /// side and guest output is thrown away.
    pub fn new(backend: Option<Box<dyn ConsoleBackend>>) -> Self {
        let host_connected = backend.is_some();
        VirtioConsole { backend, emergency_write: true, host_connected, guest_connected: false }
    }

    /// The `emergency-write` property, on by default.
    pub fn set_emergency_write(&mut self, on: bool) {
        self.emergency_write = on;
    }

    /// Whether the guest side of port 0 is open.
    pub fn guest_connected(&self) -> bool {
        self.guest_connected
    }

    /// Whether the host side of port 0 is open.
    pub fn host_connected(&self) -> bool {
        self.host_connected
    }

    /// Opens or closes the host side of port 0, `virtio_serial_open()` and
    /// `virtio_serial_close()`.
    pub fn set_host_connected(&mut self, connected: bool) {
        self.host_connected = connected;
    }

    /// `virtio_serial_guest_ready()`: how many bytes the guest can take right now, up to 4096.
    pub fn guest_ready(&self, vdev: &VirtIODevice) -> usize {
        let q = VIRTIO_CONSOLE_RX_QUEUE;
        if !vdev.queue_ready(q)
            || vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK == 0
            || vdev.queue_empty(q)
        {
            return 0;
        }
        vdev.avail_bytes(q, 4096, 0).0 as usize
    }

    /// `virtio_serial_write()`: sends host input to the guest. Returns how many bytes fitted in
    /// the buffers the guest has posted.
    pub fn write_to_guest(&mut self, vdev: &mut VirtIODevice, buf: &[u8]) -> usize {
        if !self.host_connected || !self.guest_connected {
            return 0;
        }
        let q = VIRTIO_CONSOLE_RX_QUEUE;
        if !vdev.queue_ready(q) {
            return 0;
        }
        let mem = Arc::clone(vdev.mem());
        let mut offset = 0;
        while offset < buf.len() {
            let Some(chain) = vdev.pop(q) else {
                break;
            };
            let mut w = chain.writer(&*mem);
            let len = w.write(&buf[offset..]).unwrap_or(0);
            offset += len;
            vdev.push(q, &chain, len as u32);
        }
        vdev.notify(q);
        offset
    }

    /// `have_data` of the console port: hands guest output to the backend.
    fn have_data(&mut self, buf: &[u8]) {
        if let Some(b) = self.backend.as_mut() {
            // A console drops what the backend does not take instead of throttling.
            let _ = b.write(buf);
        }
    }

    /// `discard_vq_data()`: returns every buffer on `q` untouched.
    fn discard(vdev: &mut VirtIODevice, q: u16) {
        if !vdev.queue_ready(q) {
            return;
        }
        while let Some(chain) = vdev.pop(q) {
            vdev.push(q, &chain, 0);
        }
        vdev.notify(q);
    }

    /// `handle_output()` for port 0: writes out everything the guest queued.
    fn flush_output(&mut self, vdev: &mut VirtIODevice) {
        let q = VIRTIO_CONSOLE_TX_QUEUE;
        if !self.host_connected {
            Self::discard(vdev, q);
            return;
        }
        let mem = Arc::clone(vdev.mem());
        let mut buf = [0u8; 4096];
        while let Some(chain) = vdev.pop(q) {
            let mut r = chain.reader(&*mem);
            loop {
                match r.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => self.have_data(&buf[..n]),
                }
            }
            vdev.push(q, &chain, 0);
        }
        vdev.notify(q);
    }
}

impl VirtioDeviceClass for VirtioConsole {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        vdev.init(TYPE_VIRTIO_SERIAL, VIRTIO_ID_CONSOLE, VIRTIO_CONSOLE_CONFIG_SIZE);
        vdev.add_queue(128)?;
        vdev.add_queue(128)?;
        vdev.add_queue(32)?;
        vdev.add_queue(32)?;
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        let mut features = features;
        if self.emergency_write {
            features |= feature(VIRTIO_CONSOLE_F_EMERG_WRITE);
        }
        Ok(features)
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        // cols and rows stay 0, max_nr_ports is 1, emerg_wr reads as 0.
        config.fill(0);
        config[4..8].copy_from_slice(&1u32.to_le_bytes());
    }

    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        let emerg_wr = u32::from_le_bytes([config[8], config[9], config[10], config[11]]);
        if !vdev.host_has_feature(VIRTIO_CONSOLE_F_EMERG_WRITE) || emerg_wr == 0 {
            return;
        }
        if self.host_connected {
            // Only the low byte, so a partial config write is not taken for an emergency write.
            self.have_data(&[emerg_wr as u8]);
        }
        config[8..12].fill(0);
    }

    fn set_status(&mut self, _vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        if status & VIRTIO_CONFIG_S_DRIVER_OK != 0 {
            // Non-multiport guests cannot tell us when they open the port.
            self.guest_connected = true;
        } else {
            self.guest_connected = false;
        }
        Ok(())
    }

    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        self.guest_connected = false;
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        match queue {
            VIRTIO_CONSOLE_RX_QUEUE => {
                if let Some(b) = self.backend.as_mut() {
                    b.guest_writable();
                }
            }
            VIRTIO_CONSOLE_TX_QUEUE => self.flush_output(vdev),
            VIRTIO_CONSOLE_CTRL_TX_QUEUE => Self::discard(vdev, queue),
            _ => {}
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
