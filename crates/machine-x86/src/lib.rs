// SPDX-License-Identifier: GPL-2.0-or-later

//! x86 machines: pc, q35, microvm, isapc, nitro and Xen.
//!
//! So far this is the ICH9 LPC bridge of q35 ([`ich9_lpc`]) and the `microvm` board
//! ([`microvm`]). The plan for this crate is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod ich9_lpc;
pub mod microvm;

pub use ich9_lpc::{Ich9Lpc, Ich9LpcConfig};
pub use microvm::{Microvm, MicrovmConfig, MicrovmProps};
