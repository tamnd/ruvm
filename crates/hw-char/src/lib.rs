// SPDX-License-Identifier: GPL-2.0-or-later

//! Serial ports, parallel ports and board UARTs.
//!
//! So far this is the 16550 and the Arm PL011. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod pl011;
pub mod serial;
