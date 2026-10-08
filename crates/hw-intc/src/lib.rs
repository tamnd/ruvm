// SPDX-License-Identifier: GPL-2.0-or-later

//! Interrupt controllers: 8259, IOAPIC, LAPIC, GIC, PLIC, AIA, XICS, XIVE and board controllers.
//!
//! So far the 8259 pair, the IOAPIC, the local APIC, the emulated GICv3, the SiFive PLIC, the
//! RISC-V ACLINT and the RISC-V AIA's APLIC and IMSIC are here. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod apic;
pub mod gicv3;
pub mod i8259;
pub mod ioapic;
pub mod riscv_aclint;
pub mod riscv_aplic;
pub mod riscv_imsic;
pub mod sifive_plic;
