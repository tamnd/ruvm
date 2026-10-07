// SPDX-License-Identifier: GPL-2.0-or-later

//! pvpanic, watchdogs, IPMI, ivshmem, edu, testdevs and devices that fit nowhere else.
//!
//! What is here so far:
//!
//! - [`pvpanic`]: `-device pvpanic` on the ISA I/O bus and `-device pvpanic-pci`, which let a
//!   guest tell the host that it panicked, loaded a crash kernel or wants to be shut down.
//! - [`debugexit`]: `-device isa-debug-exit`, the port test harnesses write to end a run with a
//!   chosen exit code.
//! - [`sifive_test`]: the SiFive test finisher on the RISC-V boards, which a guest writes to
//!   pass, fail with an exit code or reset.
//!
//! `isa-debugcon` is not here. QEMU keeps it in hw/char/debugcon.c because it is a character
//! device backend front end, so it belongs in the hw-char crate.
//!
//! The rest of the plan for this crate is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod debugexit;
pub mod pvpanic;
pub mod sifive_test;

pub use debugexit::{DebugExitHandler, IsaDebugExit, IsaDebugExitConfig, debug_exit_code};
pub use pvpanic::{
    PVPANIC_CRASH_LOADED, PVPANIC_EVENTS, PVPANIC_PANICKED, PVPANIC_SHUTDOWN, PvPanicEvent,
    PvPanicHandler, PvPanicIsa, PvPanicIsaConfig, PvPanicPci, PvPanicState, pvpanic_port_file,
};
pub use sifive_test::{SiFiveTest, SiFiveTestHandler, SiFiveTestRequest};
