// SPDX-License-Identifier: GPL-2.0-or-later

//! Arm machines: virt, sbsa-ref and every Arm board.
//!
//! So far these are the `virt` board ([`virt`]) with GICv3 and the `sbsa-ref` board
//! ([`sbsa_ref`]), the device tree writer they use ([`fdt`]), their CFI01 flashes
//! ([`pflash`]) and the loop that runs them on TCG ([`tcg_run`]).

#![forbid(unsafe_code)]

pub mod fdt;
pub mod pflash;
pub mod sbsa_ref;
pub mod tcg_run;
pub mod virt;
