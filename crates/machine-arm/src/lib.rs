// SPDX-License-Identifier: GPL-2.0-or-later

//! Arm machines: virt, sbsa-ref and every Arm board.
//!
//! So far this is the `virt` board ([`virt`]) with GICv3, the device tree writer it uses
//! ([`fdt`]), its CFI01 flashes ([`pflash`]) and the loop that runs it on TCG ([`tcg_run`]).

#![forbid(unsafe_code)]

pub mod fdt;
pub mod pflash;
pub mod tcg_run;
pub mod virt;
