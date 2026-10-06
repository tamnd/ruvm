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
//! - [`channel`] opens the `tcp`, `unix`, `fd`, `exec` and `file` channels, from a URI or from
//!   the `channels` argument of `migrate` and `migrate-incoming`.
//! - [`global_state`] is the `globalstate` section, the run state the destination resumes in.
//! - [`state`] is migration/migration.c: the outgoing state machine with its thread, the
//!   incoming side, the capabilities and parameters, and what `query-migrate` reports.
//!
//! The device state itself is described with `ruvm-vmstate`; the machine registers one entry
//! per device and one for RAM, with the same section names and instance ids QEMU uses, so that
//! each side finds what the other sends.
//!
//! Not here yet: postcopy, multifd, compression, XBZRLE, mapped-ram, the return path and the
//! dirty ring.

#![forbid(unsafe_code)]

pub mod channel;
pub mod global_state;
pub mod ram;
pub mod savevm;
pub mod state;

pub use channel::{Channel, FdResolver, MigrationAddr, parse_input, parse_uri};
pub use global_state::GlobalState;
pub use ram::{NoHooks, RamHooks, RamSection, RamStats};
pub use savevm::{
    DeviceState, Discard, EntryInfo, LiveState, MachineConfig, QemuFile, SaveVm, VmsdState,
};
pub use state::{Migration, MigrationHost, default_parameters};
