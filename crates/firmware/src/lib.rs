// SPDX-License-Identifier: GPL-2.0-or-later

//! Firmware discovery, fw_cfg files, ACPI tables, SMBIOS, device trees and kernel loading.
//!
//! So far this has the ACPI table builder, the e820 table and the x86 direct kernel boot logic
//! (bzImage, PVH and multiboot detection). The plan for the rest of this crate is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod acpi;
pub mod e820;
pub mod x86_linux;
