// SPDX-License-Identifier: GPL-2.0-or-later

//! Timers: PIT, HPET, the MC146818 RTC, the Arm generic timer glue, ACLINT and board timers.
//!
//! So far the three x86 ones, the Arm PL031 and the Goldfish RTC are here. The RISC-V ACLINT
//! timer is in the hw-intc crate with the rest of the ACLINT. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod goldfish_rtc;
pub mod hpet;
pub mod i8254;
pub mod mc146818;
pub mod pl031;
