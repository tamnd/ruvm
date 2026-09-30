// SPDX-License-Identifier: GPL-2.0-or-later

//! The SCSI core with `scsi-hd` and `scsi-cd`, ported from QEMU's `hw/scsi/scsi-bus.c`,
//! `hw/scsi/scsi-disk.c` and `scsi/utils.c`.
//!
//! A host bus adapter owns a [`ScsiBus`], attaches disks with [`ScsiBus::attach`], and for each
//! command finds the device with [`ScsiBus::find`], builds a request with
//! [`ScsiBus::new_request`] and runs it with [`ScsiBus::execute`]. The bus implements unit
//! attention exactly as QEMU does: a pending condition is reported (and consumed) by the next
//! command other than INQUIRY, REPORT LUNS, GET CONFIGURATION and GET EVENT STATUS
//! NOTIFICATION, and is cleared for good once the guest sees it through autosense
//! ([`ScsiBus::get_sense`]) or REQUEST SENSE.
//!
//! # Differences from QEMU
//!
//! - Every command runs synchronously inside [`ScsiBus::execute`], so there are never requests
//!   outstanding for a task management function to abort or query.
//! - Commands are handed their whole data at once instead of in scatter gather chunks.
//! - READ DISC INFORMATION and READ DVD STRUCTURE are not ported and fail with INVALID
//!   OPCODE. The transfer length of the ATA pass through commands is not computed.
//! - The `rerror` and `werror` policies are not ported; every backend error is reported.
//! - Disk geometry comes from the size alone.
//!
//! # Not ported
//!
//! `scsi-block`, `scsi-generic`, persistent reservations, device quirks and VMState.

mod bus;
mod cdb;
mod disk;
mod sense;

pub use bus::{ScsiBus, ScsiBusInfo, ScsiRequest};
pub use cdb::{
    ScsiCommand, TYPE_DISK, TYPE_INACTIVE, TYPE_NO_LUN, TYPE_NOT_PRESENT, TYPE_ROM, XferMode,
    cdb_lba, cdb_length, cdb_xfer, data_cdb_xfer, opcode,
};
pub use disk::{QEMU_HW_VERSION, ScsiDisk, ScsiDiskConf, ScsiDiskKind};
pub use sense::*;
