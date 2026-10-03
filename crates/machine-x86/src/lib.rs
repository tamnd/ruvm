// SPDX-License-Identifier: GPL-2.0-or-later

//! x86 machines: pc, q35, microvm, isapc, nitro and Xen.
//!
//! So far these are the `q35` board ([`q35`]) with its ICH9 LPC bridge ([`ich9_lpc`]), the
//! parts it shares with the other PC boards ([`pc`]) and the `microvm` board ([`microvm`]).
//! They run on TCG ([`tcg_run`]) on any host and on KVM (`kvm_run`) on Linux x86 hosts; both
//! report to the owner with the events of [`run_event`]. [`debugcon`] is `isa-debugcon`.
//! The plan for this crate is in `spec/24-workspace-layout.md`.

// The only unsafe code is the KVM_INTERRUPT ioctl in `kvm_run`, which kvm-ioctls does not wrap.
#![deny(unsafe_code)]

pub mod board;
pub mod debugcon;
pub mod file_backend;
pub mod firmware;
pub mod ich9_lpc;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod kvm_run;
pub mod microvm;
pub mod pc;
pub mod pflash;
pub mod q35;
pub mod run_event;
pub mod tcg_run;

pub use board::{BoardKind, BoardSpec, KernelFiles, X86_BOARDS, X86Board, build_board};
pub use file_backend::FileBackend;
pub use firmware::FirmwareSearch;
pub use ich9_lpc::{Ich9Lpc, Ich9LpcConfig};
pub use microvm::{Microvm, MicrovmConfig, MicrovmProps};
pub use pflash::{Pflash, PflashBacking, PflashProps};
pub use q35::{Q35, Q35MachineConfig, Q35Props};
