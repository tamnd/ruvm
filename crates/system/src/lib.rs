// SPDX-License-Identifier: GPL-2.0-or-later

//! The system emulator, system/vl.c and its neighbours: option parsing, machine creation, the
//! main loop and runstates.

#![forbid(unsafe_code)]

mod arm;
mod migration;
mod net;
pub mod options;
mod qmp_cmds;
pub mod qtest;
mod riscv;
pub mod runstate;
pub mod vl;
mod x86;
