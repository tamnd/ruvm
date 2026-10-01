// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm guest: a port of QEMU's `target/arm` TCG front end for AArch64.
//!
//! This first slice covers the A64 base integer instruction set at EL0 and EL1: data
//! processing, loads and stores in every addressing mode (pairs, exclusives and the LSE
//! atomics included), branches, conditional select and compare, bitfield, extract and CRC32,
//! with the system registers EL0 and EL1 need, the stage 1 VMSAv8-64 page walk for the 4K
//! granule and 48 bit VAs, synchronous exceptions and IRQs taken to EL1, SVC, ERET and WFI.
//!
//! Not in this slice: scalar floating point and AdvSIMD (they raise UNDEF until the second
//! slice brings them in with `ruvm-softfloat`), SVE and SVE2, and EL2 and EL3 (HVC and SMC are
//! UNDEFINED as on a CPU without those levels).
//!
//! - [`cpu`]: `CPUARMState` as a struct the generated code addresses by offset, and the CPU
//!   models.
//! - [`syndrome`]: the exception syndrome encodings of `target/arm/syndrome.h`.
//! - [`tcg`]: the translator, the helpers, the page walk and the `CpuOps` glue.

pub mod cpu;
pub mod syndrome;
pub mod tcg;
