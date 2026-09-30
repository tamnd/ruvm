// SPDX-License-Identifier: GPL-2.0-or-later

//! Interrupt controllers: 8259, IOAPIC, LAPIC, GIC, PLIC, AIA, XICS, XIVE and board controllers.
//!
//! So far the 8259 pair and the IOAPIC are here. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod i8259;
