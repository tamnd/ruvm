// SPDX-License-Identifier: GPL-2.0-or-later

//! What `virtio_serial_save_device()` writes for the one console port.

use ruvm_base::{Error, Result};

use super::VirtioConsole;

/// One port as `virtio_serial_save_device()` writes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioConsolePortVmState {
    pub id: u32,
    pub guest_connected: u8,
    pub host_connected: u8,
    /// Whether the port holds a guest buffer it has not finished with. This device never
    /// does, and refuses a stream where one is held.
    pub elem_popped: u32,
}

/// `VirtIOSerial` as `vdc->save` writes it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtioConsoleVmState {
    pub cols: u16,
    pub rows: u16,
    pub max_nr_ports: u32,
    /// `ports_map`, one word per 32 ports.
    pub ports_map: Vec<u32>,
    pub ports: Vec<VirtioConsolePortVmState>,
}

impl Default for VirtioConsoleVmState {
    fn default() -> Self {
        VirtioConsoleVmState {
            cols: 0,
            rows: 0,
            max_nr_ports: 1,
            ports_map: vec![1],
            ports: Vec::new(),
        }
    }
}

fn load_error(msg: impl std::fmt::Display) -> Error {
    Error::generic(format!("virtio-serial: {msg}"))
}

impl VirtioConsole {
    /// The device model's part of the migration stream: one port, the console at ID 0, which
    /// is also the only bit in the ports map.
    pub fn vmstate_save(&self) -> VirtioConsoleVmState {
        VirtioConsoleVmState {
            ports: vec![VirtioConsolePortVmState {
                id: 0,
                guest_connected: u8::from(self.guest_connected),
                host_connected: u8::from(self.host_connected),
                elem_popped: 0,
            }],
            ..VirtioConsoleVmState::default()
        }
    }

    /// `virtio_serial_load_device()`. The host side of the port stays as this end has it.
    pub fn vmstate_load(&mut self, s: &VirtioConsoleVmState) -> Result<()> {
        if s.max_nr_ports > 1 {
            return Err(load_error(format!("{} ports, this device has 1", s.max_nr_ports)));
        }
        if s.ports_map.first().copied().unwrap_or(1) != 1 {
            return Err(load_error("ports map differs"));
        }
        let mut guest_connected = false;
        for p in &s.ports {
            if p.id != 0 {
                return Err(load_error(format!("no port {}", p.id)));
            }
            if p.elem_popped != 0 {
                return Err(load_error("a port holds a guest buffer"));
            }
            guest_connected = p.guest_connected != 0;
        }
        self.guest_connected = guest_connected;
        Ok(())
    }
}
