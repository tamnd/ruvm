// SPDX-License-Identifier: GPL-2.0-or-later

//! Live migration: the QEMU 11.1 migration stream, precopy and its channels.
//!
//! This is the part of migration/ that a plain precopy migration between two machines goes
//! through, written so that the other end can be QEMU 11.1:
//!
//! - [`savevm`] is migration/savevm.c: the stream header, the `configuration` section, the
//!   section framing (`QEMU_VM_SECTION_START`, `PART`, `END` and `FULL`, each with its footer),
//!   the commands, `QEMU_VM_EOF` and the vmdesc JSON at the end. A [`SaveVm`] holds the
//!   `SaveStateEntry` list of a machine and both saves and loads a whole stream with it.
//! - [`ram`] is the `ram` section of migration/ram.c, version 4: the RAM block list, zero and
//!   normal pages, and the dirty bitmaps of `ruvm-mem` that tell precopy what to send again.
//!   With the `mapped-ram` capability on a `file` channel, each page has a fixed place in the
//!   file instead, after a bitmap of the pages that are there; [`mapped_ram`] is that layout.
//! - [`channel`] opens the `tcp`, `unix`, `fd`, `exec` and `file` channels, from a URI or from
//!   the `channels` argument of `migrate` and `migrate-incoming`.
//! - [`global_state`] is the `globalstate` section, the run state the destination resumes in.
//! - [`state`] is migration/migration.c: the outgoing state machine with its thread, the
//!   incoming side, the capabilities and parameters, and what `query-migrate` reports.
//! - [`postcopy`] is migration/postcopy-ram.c and the return path: the messages the
//!   destination sends back, the page requests of postcopy and userfaultfd on Linux.
//! - [`multifd`] is migration/multifd.c with the `none` and `zlib` compression methods: the
//!   extra channels RAM pages go over with the `multifd` capability.
//! - [`xbzrle`] is migration/xbzrle.c and migration/page_cache.c: pages sent again as the
//!   difference to their last copy, with the `xbzrle` capability.
//! - [`cpr`] is the CPR state of migration/cpr.c: the descriptors of shared RAM that
//!   `cpr-transfer` hands to the next process over the `cpr` channel, before the migration.
//!
//! The device state itself is described with `ruvm-vmstate`; the machine registers one entry
//! per device and one for RAM, with the same section names and instance ids QEMU uses, so that
//! each side finds what the other sends.
//!
//! Not here yet: postcopy recovery and preempt, zstd and the other multifd compression methods,
//! and the dirty ring.

#![forbid(unsafe_code)]

pub mod channel;
pub mod cpr;
pub mod global_state;
pub mod mapped_ram;
pub mod multifd;
pub mod postcopy;
pub mod ram;
pub mod savevm;
pub mod state;
pub mod write_tracking;
pub mod xbzrle;

pub use channel::{Channel, FdResolver, MigrationAddr, parse_input, parse_uri};
#[cfg(unix)]
pub use channel::{FdsetOpener, set_fdset_opener};
pub use global_state::GlobalState;
pub use ram::{NoHooks, RamHooks, RamSection, RamStats};
pub use savevm::{
    DeviceState, Discard, EntryInfo, IncomingHooks, LiveState, LoadOptions, LoadParams,
    MachineConfig, QemuFile, SaveParams, SaveVm, VmsdState,
};
pub use state::{Migration, MigrationHost, default_parameters};
