// SPDX-License-Identifier: GPL-2.0-or-later

//! The block layer: the graph, formats, protocols, filters, jobs and exports.
//!
//! So far this is the node graph from block.c with these drivers:
//!
//! - `file` from block/file-posix.c, the protocol driver for regular host files, with the
//!   byte range locks QEMU takes on images;
//! - `raw` from block/raw-format.c, with its `offset` and `size` window;
//! - `null-co` and `null-aio` from block/null.c;
//! - `blkdebug` from block/blkdebug.c, as a pass-through without rules.
//!
//! On top sit [`BlockBackend`] from block/block-backend.c, what devices use, and legacy
//! `-drive` from blockdev.c ([`BlockGraph::drive_new`]). Requests are synchronous for now: they
//! run on the caller's thread. The asynchronous path comes with the ruvm-aio integration. The
//! plan for the rest of the crate is in `spec/14-block-layer-and-tools.md` and
//! `spec/24-workspace-layout.md`.

// The only unsafe code is the pair of fcntl() calls in sys.rs.
#![deny(unsafe_code)]

mod backend;
mod drive;
mod file;
mod graph;
mod node;
mod perm;
mod raw;
mod sys;

pub use backend::BlockBackend;
pub use drive::{BlockInterfaceType, DriveInfo};
pub use graph::{BlockGraph, NodeInfo};
pub use perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED, perm_names,
};
