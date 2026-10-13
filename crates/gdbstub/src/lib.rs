// SPDX-License-Identifier: GPL-2.0-or-later

//! The GDB remote protocol server, `gdbstub/` of QEMU.
//!
//! For now this is the user mode stub that `-g` starts, for an x86_64, aarch64 or riscv64
//! program on an x86_64 Linux host. [`packet`] and [`signal`] are the parts every stub
//! shares.

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod aarch64;
pub mod packet;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod riscv64;
pub mod signal;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod user;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod x86_64;
