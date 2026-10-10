// SPDX-License-Identifier: GPL-2.0-or-later

//! The qdev equivalent: devices, buses, properties, GPIO and IRQ lines, clocks, reset, hotplug
//! and lock domains.
//!
//! So far this is the machine object, the main system bus, IRQ lines and device timers. The rest of the plan is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod bus;
pub mod fw_cfg;
pub mod irq;
pub mod machine;
pub mod timer;

pub use irq::{IrqLine, IrqPin};
pub use machine::{
    Machine, MachineClassInfo, create_machine, machine_class_info, register_machine_type,
};
pub use timer::{Clock, Timer};

/// Registers the bus and machine types.
pub fn register_types(registry: &ruvm_qom::Registry) {
    bus::register_types(registry);
    machine::register_types(registry);
}
