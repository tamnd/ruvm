// SPDX-License-Identifier: GPL-2.0-or-later

//! The QAPI schema parser and code generator, ported from QEMU's scripts/qapi.
//!
//! The port keeps the Python code's structure and error messages so that later changes to
//! scripts/qapi can be carried over by reading the diff. [`parser`] reads schema files,
//! [`schema`] builds the entity model, and [`introspect`] produces what `query-qmp-schema`
//! returns. [`rust`] generates the Rust types and their visitors.

#![forbid(unsafe_code)]

pub mod introspect;
pub mod parser;
pub mod rust;
pub mod schema;

pub use introspect::{Annotated, Lit, introspect};
pub use parser::{Error, Result, Value};
pub use schema::{Cond, Schema};
