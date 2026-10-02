// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm guest: a port of QEMU's `target/arm` TCG front end for AArch64.
//!
//! It covers the A64 base integer instruction set, scalar floating point and AdvSIMD (with AES,
//! SHA1, SHA256 and PMULL) at EL0 to EL3: data processing, loads and stores in every addressing
//! mode (pairs, exclusives and the LSE atomics included), branches, conditional select and
//! compare, bitfield, extract and CRC32, with the system registers of EL0 to EL3 (HCR_EL2 and
//! SCR_EL3 traps and routing, the VHE redirections), the stage 1 and stage 2 VMSAv8-64 page
//! walks for the 4K, 16K and 64K granules and 48 bit addresses, exceptions and interrupts
//! routed across the four ELs, ERET, WFI, HVC and SMC, PSCI over HVC or SMC behind a board
//! hook, and the generic timers reported to the board through a callback.
//!
//! The `max` model adds SVE and SVE2 at a vector length set like QEMU's `sve-max-vq`: the
//! integer, predicate, permute, element count, load, store, first fault, non fault, gather
//! and scatter instructions, with the ZCR_ELx vector length and the CPACR, CPTR ZEN traps.
//!
//! Not yet covered: the SVE floating point, widening, narrowing, dot product and crypto
//! groups, SME, AArch32, Secure EL2, 52 bit addresses, and the debug,
//! PMU, pointer authentication and MTE extensions.
//!
//! - [`cpu`]: `CPUARMState` as a struct the generated code addresses by offset, and the CPU
//!   models.
//! - [`syndrome`]: the exception syndrome encodings of `target/arm/syndrome.h`.
//! - [`tcg`]: the translator, the helpers, the page walk and the `CpuOps` glue.

pub mod cpu;
pub mod syndrome;
pub mod tcg;
