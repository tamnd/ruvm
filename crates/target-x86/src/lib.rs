// SPDX-License-Identifier: GPL-2.0-or-later

//! The x86 guest: architectural CPU state, CPU models and CPUID, and MSR
//! definitions.
//!
//! This crate knows nothing about KVM. It describes what an x86 vCPU looks
//! like after reset and what CPUID it reports; `ruvm-accel-kvm` turns that
//! into ioctls.
//!
//! - [`state`] holds [`state::X86CpuState`] and the port of
//!   `x86_cpu_reset_hold()`.
//! - [`cpuid`] holds the CPU models, feature words, `+feat,-feat` parsing,
//!   host filtering and `cpu_x86_cpuid()`.
//! - [`msr`] holds MSR indices and the list of MSRs written at reset.

#![forbid(unsafe_code)]

pub mod cpuid;
pub mod msr;
pub mod state;
