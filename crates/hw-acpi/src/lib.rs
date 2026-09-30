// SPDX-License-Identifier: GPL-2.0-or-later

//! ACPI hardware: PM timer, GPE, hotplug registers and GED.
//!
//! [`core`] is hw/acpi/core.c (PM1 event and control, the PM timer, GPE blocks), [`ich9`] is the
//! ICH9 LPC power management block from hw/acpi/ich9.c and [`ged`] is the Generic Event Device
//! from hw/acpi/generic_event_device.c. ACPI table building lives in ruvm-firmware.
//!
//! CPU and memory hotplug registers, ACPI PCI hotplug, the ICH9 TCO watchdog and PIIX4 PM are
//! not ported yet. VMState, trace points and QOM registration are not ported for any device here.

#![forbid(unsafe_code)]

pub mod core;
pub mod ged;
pub mod ich9;

pub use crate::core::{
    AcpiPm, AcpiPmConfig, AcpiRegs, PmTimerWidth, SystemRequest, SystemRequestHandler, WakeupReason,
};
pub use ged::{AcpiGed, AcpiGedProps};
pub use ich9::{Ich9Pm, Ich9PmProps};
