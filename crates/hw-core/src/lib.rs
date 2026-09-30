// SPDX-License-Identifier: GPL-2.0-or-later

//! The qdev equivalent: devices, buses, properties, GPIO and IRQ lines, clocks, reset, hotplug
//! and lock domains.
//!
//! So far this is the machine object and the main system bus, which is what machine `none`
//! needs. The rest of the plan is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod bus;
pub mod machine;

pub use machine::{Machine, MachineClassInfo, create_machine, machine_class_info};

/// Registers the bus and machine types.
pub fn register_types(registry: &ruvm_qom::Registry) {
    bus::register_types(registry);
    machine::register_types(registry);
}
