// SPDX-License-Identifier: GPL-2.0-or-later

//! The block layer: the graph, formats, protocols, filters, jobs and exports.
//!
//! So far this is the node graph from block.c: children and their roles, backing chains,
//! format probing, `json:` and plain file names, the graph lock and drained sections,
//! permission updates, `blockdev-reopen`, snapshots and activation. The request layer of
//! block/io.c sits on the nodes (alignment with read-modify-write, serialising requests,
//! `max_transfer` splitting, copy-on-read, block status, write thresholds and accounting).
//! The drivers are:
//!
//! - `file`, `host_device` and `host_cdrom` from block/file-posix.c, with the byte range locks
//!   QEMU takes on images and `aio=threads`, `native` and `io_uring` (the last two on Linux);
//! - `raw` from block/raw-format.c, with its `offset` and `size` window;
//! - `null-co` and `null-aio` from block/null.c;
//! - the filters `blkdebug`, `blkverify`, `throttle` (with throttle groups), `copy-on-read`,
//!   `preallocate` and `compress`;
//! - `nbd` and `luks`, which live in their own modules.
//! - the image formats `qed`, `qcow` (version 1), `parallels`, `dmg`, `cloop` and `bochs`, and
//!   the `vvfat` protocol (`fat:` file names), each in its own module.
//!
//! On top sit [`BlockBackend`] from block/block-backend.c, what devices use, and legacy
//! `-drive` from blockdev.c ([`BlockGraph::drive_new`]). Requests are synchronous: they run on
//! the caller's thread, and the io module explains how a ruvm-aio executor drives them. The
//! plan for the rest of the crate is in `spec/14-block-layer-and-tools.md` and
//! `spec/24-workspace-layout.md`.

// The only unsafe code is in sys.rs (fcntl() and ioctl()) and protocol/aio.rs (linux-aio and
// io_uring submission).
#![deny(unsafe_code)]

pub mod accounting;
mod backend;
mod bitmap;
mod block_copy;
mod bochs;
mod cloop;
mod create;
mod dmg;
mod drain;
mod drive;
mod drivers;
mod event;
#[cfg(unix)]
mod file;
mod filter;
mod graph;
mod graph_lock;
mod imgopts;
mod io;
mod job;
mod luks;
pub mod nbd;
mod node;
mod open;
mod ops;
mod parallels;
mod perm;
mod probe;
mod protocol;
mod qcow;
mod qed;
mod query;
mod raw;
mod reopen;
#[cfg(unix)]
mod sys;
pub mod throttle;
pub mod tools;
mod vdi;
mod vhdx;
mod vmdk;
mod vpc;
mod vvfat;

pub use backend::BlockBackend;
pub use drive::{BlockInterfaceType, DriveInfo};
pub use event::{BlockEvent, BlockEventHook, set_event_hook};
pub use graph::{BlockGraph, NodeInfo};
pub use node::{BDRV_FIX_ERRORS, BDRV_FIX_LEAKS, BlockFragInfo, CheckResult};
pub use ops::MapEntry;
pub use perm::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED, perm_names,
};
