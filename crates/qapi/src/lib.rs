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

/// The `query-qmp-schema` reply for this build, as QEMU would print it.
///
/// It is computed at build time from the vendored schema, with each `'if'` decided by the table
/// in build.rs, so returning it costs a copy.
pub static QMP_SCHEMA_JSON: &str = include_str!(concat!(env!("OUT_DIR"), "/qmp-schema.json"));

/// [`QMP_SCHEMA_JSON`] as a value.
pub fn qmp_schema() -> QValue {
    json::from_str(QMP_SCHEMA_JSON).expect("the build script writes valid JSON")
}
