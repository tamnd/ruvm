// SPDX-License-Identifier: GPL-2.0-or-later

//! Storage controllers and drives.
//!
//! For now this is the ICH9 AHCI controller ([`Ich9Ahci`], QEMU's `ich9-ahci`) with SATA hard
//! disks and ATAPI CD-ROM drives, ported from QEMU's `hw/ide/ahci.c`, `hw/ide/ich.c`,
//! `hw/ide/core.c` and `hw/ide/atapi.c`, and the same controller as a memory mapped device
//! ([`SysbusAhci`], QEMU's `sysbus-ahci` from `hw/ide/ahci-sysbus.c`). Drives read and write
//! a [`BlockBackend`]; [`VecBackend`] keeps an image in memory.
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
//! - VMState: [`Ich9Ahci::vmstate_save`] and [`Ich9Ahci::vmstate_load`] have QEMU's
//!   `ich9_ahci` layout, but only an idle controller loads: a stream with an NCQ tag in use, a
//!   command in progress, a PIO transfer (`ide_drive/pio_state`), a request waiting for retry
//!   (`ide_bus/error`) or an open tray is refused. The media change flags (`cdrom_changed`,
//!   `ide_drive/atapi/gesn_state`) are dropped on load and sent as 0, and free NCQ tags are
//!   sent as zeros where QEMU keeps the last command's fields.
//!
//! # Not ported
//!
//! Trace points, QOM properties and the `rerror` and `werror` error policies. Every backend
//! error is reported to the guest, QEMU's "report" policy.

#![forbid(unsafe_code)]

mod ahci;
mod ahci_sysbus;
mod atapi;
mod block;
mod ich;
mod ide;
pub mod scsi;

pub use ahci::{AhciPortVmState, AhciVmState, DmaMemory, NcqVmState};
pub use ahci_sysbus::{SYSBUS_AHCI_MMIO_SIZE, SysbusAhci, TYPE_SYSBUS_AHCI};
pub use block::{BlockBackend, VecBackend};
pub use ich::{
    ICH9_AHCI_PORTS, Ich9Ahci, Ich9AhciVmState, PCI_CLASS_STORAGE_SATA,
    PCI_DEVICE_ID_INTEL_82801IR, PCI_VENDOR_ID_INTEL,
};
pub use ide::{DriveConfig, DriveKind, IDE_IO_BUFFER_TOTAL_LEN, IdeBusVmState, IdeDriveVmState};
