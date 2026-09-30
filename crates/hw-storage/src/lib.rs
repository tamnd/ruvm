// SPDX-License-Identifier: GPL-2.0-or-later

//! Storage controllers and drives.
//!
//! For now this is the ICH9 AHCI controller ([`Ich9Ahci`], QEMU's `ich9-ahci`) with SATA hard
//! disks and ATAPI CD-ROM drives, ported from QEMU's `hw/ide/ahci.c`, `hw/ide/ich.c`,
//! `hw/ide/core.c` and `hw/ide/atapi.c`. Drives read and write a [`BlockBackend`];
//! [`VecBackend`] keeps an image in memory.
//!
//! The [`scsi`] module has the SCSI core, `scsi-hd` and `scsi-cd`, for SCSI host adapters such
//! as virtio-scsi. It uses the same [`BlockBackend`].
//!
//! # The ATA and ATAPI subset
//!
//! Hard disks implement IDENTIFY DEVICE, READ and WRITE SECTORS (and MULTIPLE, and the EXT
//! forms), READ and WRITE DMA (EXT), READ and WRITE FPDMA QUEUED (NCQ), FLUSH CACHE (EXT), SET
//! FEATURES, SET MULTIPLE MODE, the VERIFY and native max address commands, and the power
//! management no-ops. The model string is "QEMU HARDDISK", the default serial number "QM%05d"
//! numbered as QEMU does, and the firmware revision "2.5+". CD-ROM drives implement IDENTIFY
//! PACKET DEVICE, DEVICE RESET and PACKET with TEST UNIT READY, REQUEST SENSE, INQUIRY, READ
//! CAPACITY, READ(10), READ(12), MODE SENSE(10), GET CONFIGURATION, SEEK, SET CD SPEED and
//! PREVENT ALLOW MEDIUM REMOVAL, by PIO or DMA.
//!
//! # Differences from QEMU
//!
//! - Every request is synchronous. QEMU submits block I/O asynchronously and finishes commands
//!   from completion callbacks; here a command, NCQ included, runs to completion inside the
//!   guest's register write. The "check the command list again" bottom half runs before the
//!   write returns.
//! - Interrupt changes are collected while the controller state is locked and delivered to the
//!   PCI core, as INTx levels or MSI messages, once the lock is released.
//! - A register access that arrives on the thread that is already inside the controller, for
//!   example a DMA transfer aimed at the controller's own BAR, is dropped (reads return zero)
//!   instead of reentering.
//! - All DMA, FIS writes included, fails while the function is not a bus master.
//! - Mapping the command list and FIS area is a readability check of guest memory at the time
//!   the engines start, not a long lived host mapping.
//! - Disk geometry comes from the size alone; the MBR based guess is not ported.
//! - DATA SET MANAGEMENT (TRIM), SMART, security and the CFA commands abort. NCQ commands other
//!   than READ and WRITE FPDMA QUEUED fail the queue. The ATAPI commands not listed above (GET
//!   EVENT STATUS NOTIFICATION, READ TOC, READ CD, START STOP UNIT, READ DISC INFORMATION, READ
//!   DVD STRUCTURE, MECHANISM STATUS and the rest) are rejected with ILLEGAL REQUEST, and there
//!   is no media change or tray model. The CD audio control mode page is all zeros.
//!
//! # Not ported
//!
//! VMState (migration), trace points, QOM properties and the `rerror` and `werror` error
//! policies. Every backend error is reported to the guest, QEMU's "report" policy.

#![forbid(unsafe_code)]

mod ahci;
mod atapi;
mod block;
mod ich;
mod ide;
pub mod scsi;

pub use ahci::DmaMemory;
pub use block::{BlockBackend, VecBackend};
pub use ich::{
    ICH9_AHCI_PORTS, Ich9Ahci, PCI_CLASS_STORAGE_SATA, PCI_DEVICE_ID_INTEL_82801IR,
    PCI_VENDOR_ID_INTEL,
};
pub use ide::{DriveConfig, DriveKind};
