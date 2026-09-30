// SPDX-License-Identifier: GPL-2.0-or-later

//! The block layer: the graph, formats, protocols, filters, jobs and exports.
//!
//! For now this is the node graph from block.c with the two drivers the QMP conformance tests
//! reach for, `null-co`/`null-aio` from block/null.c and `blkdebug` from block/blkdebug.c. The
//! drivers have no I/O path yet. The plan for the rest of the crate is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

mod graph;

pub use graph::{BlockGraph, NodeInfo};
