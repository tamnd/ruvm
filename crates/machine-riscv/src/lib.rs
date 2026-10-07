// SPDX-License-Identifier: GPL-2.0-or-later

//! RISC-V machines: virt, spike, sifive and the rest.
//!
//! So far this is the `virt` board ([`virt`]), hw/riscv/virt.c, and the loop that runs it on
//! TCG ([`tcg_run`]). The device tree writer and the CFI flashes are the ones of
//! `ruvm-machine-arm`, which are not Arm specific.

#![forbid(unsafe_code)]

pub mod tcg_run;
pub mod virt;
