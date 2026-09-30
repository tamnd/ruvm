// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP monitor, a port of monitor/qmp.c, monitor/monitor.c and
//! monitor/qmp-cmds-control.c.
//!
//! [`Qmp`] holds the command tables, the monitors and the in-band dispatcher. A
//! [`MonitorQmp`] is one monitor, fed the bytes its client sends. The wire protocol,
//! capabilities negotiation, out of band execution, descriptor passing, the request queue with its suspend and
//! resume rules and event throttling all follow QEMU 11.1, because clients see each of them.
//! HMP comes later.

#![forbid(unsafe_code)]

pub mod control;
pub mod event;
#[cfg(unix)]
pub mod fds;
pub mod qmp;

pub use qmp::{Commands, MonitorQmp, QMP_REQ_QUEUE_LEN_MAX, Qmp};
