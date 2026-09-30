// SPDX-License-Identifier: GPL-2.0-or-later

//! x86 machines: pc, q35, microvm, isapc, nitro and Xen.
//!
//! So far these are the `q35` board ([`q35`]) with its ICH9 LPC bridge ([`ich9_lpc`]), the
//! parts it shares with the other PC boards ([`pc`]) and the `microvm` board ([`microvm`]).
//! The plan for this crate is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod ich9_lpc;
pub mod microvm;
pub mod pc;
pub mod q35;

pub use ich9_lpc::{Ich9Lpc, Ich9LpcConfig};
pub use microvm::{Microvm, MicrovmConfig, MicrovmProps};
pub use q35::{Q35, Q35MachineConfig, Q35Props};
