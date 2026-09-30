// SPDX-License-Identifier: MIT OR Apache-2.0

//! Errors and error classes, bit helpers, epoch based RCU, notifier lists and the timer wheel shared by every ruvm crate.
//!
//! Nothing in here knows about guests, devices or QEMU's object model. It is the small set of pieces
//! that every other layer needs and that QEMU keeps in `util/` and `qapi/error.c`.

#![forbid(unsafe_code)]

pub mod bitmap;
pub mod bits;
pub mod error;
pub mod notify;
pub mod rcu;
pub mod report;
pub mod timer;

pub use bitmap::Bitmap;
pub use error::{Error, ErrorClass, Result, ResultExt};
pub use notify::{NotifierId, NotifierList};
pub use rcu::Rcu;
pub use report::{Location, error_report, set_program_name, warn_report};
pub use timer::{ClockType, TimerId, TimerList};

/// Builds a `GenericError` from a format string, the way `error_setg()` is used in QEMU.
#[macro_export]
macro_rules! err {
    ($($t:tt)*) => { $crate::Error::generic(::std::format!($($t)*)) };
}

/// Returns early with a `GenericError`, for the common `error_setg(errp, ...); return` pair.
#[macro_export]
macro_rules! bail {
    ($($t:tt)*) => { return ::core::result::Result::Err($crate::err!($($t)*).into()) };
}

/// Reports a message the way `hw_error()` does and aborts. This is for states that mean ruvm has a
/// bug, never for anything a guest or a user can cause.
#[macro_export]
macro_rules! fatal {
    ($($t:tt)*) => {{
        $crate::report::error_report(&::std::format!($($t)*));
        ::std::process::abort()
    }};
}
