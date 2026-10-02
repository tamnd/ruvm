// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 guest: architectural CPU state, CPU models and CPUID, and MSR
//! definitions.
//!
//! The crate describes what an x86 vCPU looks like after reset and what
//! CPUID it reports, and on Linux x86-64 hosts it also moves that state in
//! and out of a KVM vCPU.
//!
//! - [`state`] holds [`state::X86CpuState`] and the port of
//!   `x86_cpu_reset_hold()`.
//! - [`cpuid`] holds the CPU models, feature words, `+feat,-feat` parsing,
//!   host filtering and `cpu_x86_cpuid()`.
//! - [`msr`] holds MSR indices and the list of MSRs written at reset.
//! - [`kvm_convert`] turns the state into KVM structures and back. It is
//!   plain data work and builds everywhere.
//! - `kvm` (Linux x86-64 only) is the port of `target/i386/kvm/kvm.c`:
//!   vCPU setup and register sync over `ruvm-accel-kvm`.
//! - [`tcg`] is the port of `target/i386/tcg`: the integer instruction set
//!   translated by `ruvm-jit`, with the page walk, exceptions and
//!   interrupts. x87, SSE and AVX raise #UD.

#![deny(unsafe_code)]

pub mod cpuid;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod kvm;
pub mod kvm_convert;
pub mod msr;
pub mod state;
pub mod tcg;
