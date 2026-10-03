// SPDX-License-Identifier: GPL-2.0-or-later

//! Arm machines: virt, sbsa-ref and every Arm board.
//!
//! So far this is the `virt` board ([`virt`]) with GICv3, and the device tree writer it uses
//! ([`fdt`]).

#![forbid(unsafe_code)]

pub mod fdt;
pub mod virt;
