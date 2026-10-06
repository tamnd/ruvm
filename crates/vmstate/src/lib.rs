// SPDX-License-Identifier: GPL-2.0-or-later

//! VMState descriptions and the QEMU compatible encoder and decoder for them.
//!
//! This is migration/vmstate.c, migration/vmstate-types.c and the parts of migration/qemu-file.c
//! they use. A [`VmStateDescription`] lists the fields of some piece of device state in wire order,
//! with versions, tests, nested descriptions and subsections. [`vmstate_save_state`] turns it into
//! exactly the bytes QEMU 11.1 writes for the same description and values, and
//! [`vmstate_load_state`] reads them back with the same checks and error messages.
//!
//! QEMU reaches each field through a byte offset into the device struct. Here each field has a
//! closure that borrows the value out of the state, so the crate needs no unsafe code and device
//! structs keep whatever layout suits them. What cannot change is the field list itself, which has
//! to match QEMU's entry for entry.
//!
//! [`vmstate_save_state_vmdesc`] also writes the vmdesc JSON that QEMU appends to a migration
//! stream. The list based infos (`qtailq`, `gtree`, `qlist`, `GByteArray`, `bitmap`, `fd`) are
//! not here yet; `timer` is [`info::Timer`] over the expiry time.

#![forbid(unsafe_code)]

mod field;
mod file;
pub mod info;
mod json;
mod vmsd;
mod vmstate;

pub use field::VmStateField;
pub use file::{EINVAL, EIO, StreamReader, StreamWriter};
pub use info::{VmStateInfo, VmStateType};
pub use json::JsonWriter;
pub use vmsd::{MigPriority, VmStateDescription};
pub use vmstate::{vmstate_load_state, vmstate_save_state, vmstate_save_state_vmdesc};

/// `QEMU_VM_EOF`, the byte that ends a migration stream.
pub const QEMU_VM_EOF: u8 = 0x00;
/// `QEMU_VM_SUBSECTION`, the byte in front of each subsection.
pub const QEMU_VM_SUBSECTION: u8 = 0x05;
/// `VMS_MARKER_PTR_NULL`, sent in place of a missing element of a pointer array.
pub const VMS_MARKER_PTR_NULL: u8 = 0x30;
/// `VMS_MARKER_PTR_VALID`, sent in front of a present element when the array is auto allocated.
pub const VMS_MARKER_PTR_VALID: u8 = 0x31;
