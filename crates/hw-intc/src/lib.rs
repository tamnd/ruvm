// SPDX-License-Identifier: GPL-2.0-or-later

//! Interrupt controllers: 8259, IOAPIC, LAPIC, GIC, PLIC, AIA, XICS, XIVE and board controllers.
//!
//! So far the 8259 pair, the IOAPIC and the emulated GICv3 are here. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod gicv3;
pub mod i8259;
pub mod ioapic;
