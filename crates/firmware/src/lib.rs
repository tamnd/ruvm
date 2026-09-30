// SPDX-License-Identifier: GPL-2.0-or-later

//! Firmware discovery, fw_cfg files, ACPI tables, SMBIOS, device trees and kernel loading.
//!
//! Only the ACPI table builder is here so far. The plan for the rest of this crate is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod acpi;
