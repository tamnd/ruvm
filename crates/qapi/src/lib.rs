// SPDX-License-Identifier: GPL-2.0-or-later

//! Rust types, visitors, QMP dispatch and introspection generated from QEMU's QAPI schema.
//!
//! This first part is QEMU's QObject layer: [`QValue`] and [`QDict`] in place of `QObject` and its
//! subtypes, and [`json`] in place of the lexer, streamer, parser and writer in qobject/. Everything
//! a QMP client sees passes through the writer, so its output is QEMU's byte for byte, down to the
//! order of keys in an object.

#![forbid(unsafe_code)]

pub mod json;
mod qvalue;

pub use qvalue::{QDict, QType, QValue};
