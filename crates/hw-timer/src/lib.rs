// SPDX-License-Identifier: GPL-2.0-or-later

//! Timers: PIT, HPET, the MC146818 RTC, the Arm generic timer glue, ACLINT and board timers.
//!
//! So far the PIT and the MC146818 RTC are here. The rest of the plan is in `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod i8254;
pub mod mc146818;
