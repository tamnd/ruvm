// SPDX-License-Identifier: GPL-2.0-or-later

//! Command descriptor blocks: opcodes, lengths, transfer sizes and directions, from
//! `scsi/utils.c` and `scsi_req_parse_cdb()` in `hw/scsi/scsi-bus.c`.

/// The SCSI operation codes this crate knows about, from `include/scsi/constants.h`.
#[allow(missing_docs)]
pub mod opcode {
    pub const TEST_UNIT_READY: u8 = 0x00;
    pub const REWIND: u8 = 0x01;
    pub const REQUEST_SENSE: u8 = 0x03;
    pub const FORMAT_UNIT: u8 = 0x04;
    pub const READ_BLOCK_LIMITS: u8 = 0x05;
    pub const REASSIGN_BLOCKS: u8 = 0x07;
    pub const READ_6: u8 = 0x08;
    pub const WRITE_6: u8 = 0x0a;
    pub const SET_CAPACITY: u8 = 0x0b;
    pub const READ_REVERSE: u8 = 0x0f;
    pub const WRITE_FILEMARKS: u8 = 0x10;
    pub const SPACE: u8 = 0x11;
    pub const INQUIRY: u8 = 0x12;
    pub const MODE_SELECT: u8 = 0x15;
    pub const RESERVE: u8 = 0x16;
    pub const RELEASE: u8 = 0x17;
    pub const COPY: u8 = 0x18;
    pub const ERASE: u8 = 0x19;
    pub const MODE_SENSE: u8 = 0x1a;
    pub const START_STOP: u8 = 0x1b;
    pub const RECEIVE_DIAGNOSTIC: u8 = 0x1c;
    pub const SEND_DIAGNOSTIC: u8 = 0x1d;
    pub const ALLOW_MEDIUM_REMOVAL: u8 = 0x1e;
    pub const SET_WINDOW: u8 = 0x24;
    pub const READ_CAPACITY_10: u8 = 0x25;
    pub const READ_10: u8 = 0x28;
    pub const WRITE_10: u8 = 0x2a;
    pub const SEEK_10: u8 = 0x2b;
    pub const WRITE_VERIFY_10: u8 = 0x2e;
    pub const VERIFY_10: u8 = 0x2f;
    pub const SEARCH_HIGH: u8 = 0x30;
    pub const SEARCH_EQUAL: u8 = 0x31;
    pub const SEARCH_LOW: u8 = 0x32;
    pub const SET_LIMITS: u8 = 0x33;
    pub const PRE_FETCH: u8 = 0x34;
    pub const SYNCHRONIZE_CACHE: u8 = 0x35;
    pub const LOCK_UNLOCK_CACHE: u8 = 0x36;
    pub const MEDIUM_SCAN: u8 = 0x38;
    pub const COMPARE: u8 = 0x39;
    pub const COPY_VERIFY: u8 = 0x3a;
    pub const WRITE_BUFFER: u8 = 0x3b;
    pub const READ_BUFFER: u8 = 0x3c;
    pub const UPDATE_BLOCK: u8 = 0x3d;
    pub const WRITE_LONG_10: u8 = 0x3f;
    pub const CHANGE_DEFINITION: u8 = 0x40;
    pub const WRITE_SAME_10: u8 = 0x41;
    pub const UNMAP: u8 = 0x42;
    pub const READ_TOC: u8 = 0x43;
    pub const GET_CONFIGURATION: u8 = 0x46;
    pub const GET_EVENT_STATUS_NOTIFICATION: u8 = 0x4a;
    pub const LOG_SELECT: u8 = 0x4c;
    pub const READ_DISC_INFORMATION: u8 = 0x51;
    pub const RESERVE_TRACK: u8 = 0x53;
    pub const MODE_SELECT_10: u8 = 0x55;
    pub const RESERVE_10: u8 = 0x56;
    pub const RELEASE_10: u8 = 0x57;
    pub const MODE_SENSE_10: u8 = 0x5a;
    pub const SEND_CUE_SHEET: u8 = 0x5d;
    pub const PERSISTENT_RESERVE_OUT: u8 = 0x5f;
    pub const WRITE_FILEMARKS_16: u8 = 0x80;
    pub const ALLOW_OVERWRITE: u8 = 0x82;
    pub const ATA_PASSTHROUGH_16: u8 = 0x85;
    pub const READ_16: u8 = 0x88;
    pub const WRITE_16: u8 = 0x8a;
    pub const WRITE_VERIFY_16: u8 = 0x8e;
    pub const VERIFY_16: u8 = 0x8f;
    pub const PRE_FETCH_16: u8 = 0x90;
    pub const SYNCHRONIZE_CACHE_16: u8 = 0x91;
    pub const LOCATE_16: u8 = 0x92;
    pub const WRITE_SAME_16: u8 = 0x93;
    pub const SERVICE_ACTION_IN_16: u8 = 0x9e;
    pub const REPORT_LUNS: u8 = 0xa0;
    pub const ATA_PASSTHROUGH_12: u8 = 0xa1;
    pub const MAINTENANCE_IN: u8 = 0xa3;
    pub const MAINTENANCE_OUT: u8 = 0xa4;
    pub const SET_READ_AHEAD: u8 = 0xa7;
    pub const READ_12: u8 = 0xa8;
    pub const WRITE_12: u8 = 0xaa;
    pub const ERASE_12: u8 = 0xac;
    pub const READ_DVD_STRUCTURE: u8 = 0xad;
    pub const WRITE_VERIFY_12: u8 = 0xae;
    pub const VERIFY_12: u8 = 0xaf;
    pub const SEARCH_HIGH_12: u8 = 0xb0;
    pub const SEARCH_EQUAL_12: u8 = 0xb1;
    pub const SEARCH_LOW_12: u8 = 0xb2;
    pub const SEND_VOLUME_TAG: u8 = 0xb6;
    pub const SET_CD_SPEED: u8 = 0xbb;
    pub const MECHANISM_STATUS: u8 = 0xbd;
    pub const READ_CD: u8 = 0xbe;
    pub const SEND_DVD_STRUCTURE: u8 = 0xbf;

    /// The SERVICE ACTION IN(16) service action of READ CAPACITY(16).
    pub const SAI_READ_CAPACITY_16: u8 = 0x10;
}

use opcode::*;

/// `TYPE_DISK`: a direct access block device.
pub const TYPE_DISK: u8 = 0x00;
/// `TYPE_ROM`: a CD or DVD drive.
pub const TYPE_ROM: u8 = 0x05;
/// `TYPE_NOT_PRESENT`.
pub const TYPE_NOT_PRESENT: u8 = 0x1f;
/// `TYPE_INACTIVE`.
pub const TYPE_INACTIVE: u8 = 0x20;
/// `TYPE_NO_LUN`: the peripheral qualifier for a LUN that does not exist.
pub const TYPE_NO_LUN: u8 = 0x7f;

/// Which way the data of a command flows, QEMU's `SCSIXferMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum XferMode {
    /// No data.
    #[default]
    None,
    /// From the device to the initiator, like a READ.
    FromDev,
    /// From the initiator to the device, like a WRITE.
    ToDev,
}

/// A parsed command, QEMU's `SCSICommand`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScsiCommand {
    /// The CDB bytes, `len` of them used.
    pub buf: [u8; 16],
    /// The CDB length.
    pub len: usize,
    /// The number of bytes the command transfers.
    pub xfer: u64,
    /// The logical block address, `u64::MAX` for commands without one.
    pub lba: u64,
    /// The data direction.
    pub mode: XferMode,
}

fn be16(b: &[u8]) -> u32 {
    u32::from(u16::from_be_bytes([b[0], b[1]]))
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// `scsi_cdb_length()`: the length of a CDB from its group code, `None` for the reserved and
/// vendor specific groups.
pub fn cdb_length(buf: &[u8]) -> Option<usize> {
    match buf.first()? >> 5 {
        0 => Some(6),
        1 | 2 => Some(10),
        4 => Some(16),
        5 => Some(12),
        _ => None,
    }
}

/// `scsi_cdb_xfer()`: the transfer length field of a CDB. `buf` must hold the whole CDB.
pub fn cdb_xfer(buf: &[u8]) -> u32 {
    match buf[0] >> 5 {
        0 => u32::from(buf[4]),
        1 | 2 => be16(&buf[7..]),
        4 => be32(&buf[10..]),
        5 => be32(&buf[6..]),
        _ => u32::MAX,
    }
}

/// `scsi_data_cdb_xfer()`: like [`cdb_xfer`], but a 6-byte READ or WRITE of 0 blocks means 256.
pub fn data_cdb_xfer(buf: &[u8]) -> u32 {
    if buf[0] >> 5 == 0 && buf[4] == 0 { 256 } else { cdb_xfer(buf) }
}

/// `scsi_cmd_lba()`: the logical block address field of a CDB.
pub fn cdb_lba(buf: &[u8]) -> u64 {
    match buf[0] >> 5 {
        0 => u64::from(be32(buf) & 0x1f_ffff),
        1 | 2 | 5 => u64::from(be32(&buf[2..])),
        4 => u64::from_be_bytes([buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8], buf[9]]),
        _ => u64::MAX,
    }
}

/// `scsi_req_xfer()` for disks and CD drives.
fn req_xfer(buf: &[u8], blocksize: u32, dev_type: u8) -> u64 {
    let bs = u64::from(blocksize);
    let mut xfer = u64::from(cdb_xfer(buf));
    match buf[0] {
        TEST_UNIT_READY | REWIND | START_STOP | SET_CAPACITY | WRITE_FILEMARKS
        | WRITE_FILEMARKS_16 | SPACE | RESERVE | RELEASE | ERASE | ALLOW_MEDIUM_REMOVAL
        | SEEK_10 | SYNCHRONIZE_CACHE | SYNCHRONIZE_CACHE_16 | LOCATE_16 | LOCK_UNLOCK_CACHE
        | SET_CD_SPEED | SET_LIMITS | WRITE_LONG_10 | UPDATE_BLOCK | RESERVE_TRACK
        | SET_READ_AHEAD | PRE_FETCH | PRE_FETCH_16 | ALLOW_OVERWRITE => xfer = 0,
        VERIFY_10 | VERIFY_12 | VERIFY_16 => {
            if buf[1] & 2 == 0 {
                xfer = 0;
            } else if buf[1] & 4 != 0 {
                xfer = 1;
            }
            xfer *= bs;
        }
        WRITE_SAME_10 | WRITE_SAME_16 => xfer = if buf[1] & 1 != 0 { 0 } else { bs },
        READ_CAPACITY_10 => xfer = 8,
        READ_BLOCK_LIMITS => xfer = 6,
        SEND_VOLUME_TAG => {
            xfer = if dev_type == TYPE_ROM { be16(&buf[9..]) } else { be16(&buf[8..]) }.into();
        }
        WRITE_6 | READ_6 | READ_REVERSE => {
            if xfer == 0 {
                xfer = 256;
            }
            xfer *= bs;
        }
        WRITE_10 | WRITE_VERIFY_10 | WRITE_12 | WRITE_VERIFY_12 | WRITE_16 | WRITE_VERIFY_16
        | READ_10 | READ_12 | READ_16 => xfer *= bs,
        FORMAT_UNIT => {
            xfer = if dev_type == TYPE_ROM && buf[1] & 16 != 0 {
                12
            } else if buf[1] & 16 == 0 {
                0
            } else if buf[1] & 32 != 0 {
                8
            } else {
                4
            };
        }
        INQUIRY | RECEIVE_DIAGNOSTIC | SEND_DIAGNOSTIC => xfer = be16(&buf[3..]).into(),
        READ_CD | READ_BUFFER | WRITE_BUFFER | SEND_CUE_SHEET => {
            xfer = u64::from(buf[8]) | u64::from(buf[7]) << 8 | u64::from(buf[6]) << 16;
        }
        PERSISTENT_RESERVE_OUT => xfer = be32(&buf[5..]).into(),
        MECHANISM_STATUS | READ_DVD_STRUCTURE | SEND_DVD_STRUCTURE | MAINTENANCE_OUT
        | MAINTENANCE_IN
            if dev_type == TYPE_ROM =>
        {
            xfer = be16(&buf[8..]).into();
        }
        ATA_PASSTHROUGH_12 if dev_type == TYPE_ROM => xfer = 0,
        _ => {}
    }
    xfer
}

/// `scsi_cmd_xfer_mode()`.
fn xfer_mode(buf: &[u8], xfer: u64) -> XferMode {
    if xfer == 0 {
        return XferMode::None;
    }
    match buf[0] {
        WRITE_6
        | WRITE_10
        | WRITE_VERIFY_10
        | WRITE_12
        | WRITE_VERIFY_12
        | WRITE_16
        | WRITE_VERIFY_16
        | VERIFY_10
        | VERIFY_12
        | VERIFY_16
        | COPY
        | COPY_VERIFY
        | COMPARE
        | CHANGE_DEFINITION
        | LOG_SELECT
        | MODE_SELECT
        | MODE_SELECT_10
        | SEND_DIAGNOSTIC
        | WRITE_BUFFER
        | FORMAT_UNIT
        | REASSIGN_BLOCKS
        | SEARCH_EQUAL
        | SEARCH_HIGH
        | SEARCH_LOW
        | UPDATE_BLOCK
        | WRITE_LONG_10
        | WRITE_SAME_10
        | WRITE_SAME_16
        | UNMAP
        | SEARCH_HIGH_12
        | SEARCH_EQUAL_12
        | SEARCH_LOW_12
        | MEDIUM_SCAN
        | SEND_VOLUME_TAG
        | SEND_CUE_SHEET
        | SEND_DVD_STRUCTURE
        | PERSISTENT_RESERVE_OUT
        | MAINTENANCE_OUT
        | SET_WINDOW => XferMode::ToDev,
        ATA_PASSTHROUGH_12 | ATA_PASSTHROUGH_16 => {
            if buf[2] & 8 != 0 {
                XferMode::FromDev
            } else {
                XferMode::ToDev
            }
        }
        _ => XferMode::FromDev,
    }
}

impl ScsiCommand {
    /// `scsi_req_parse_cdb()` for a device with the given block size and peripheral type.
    /// `None` when the group code is reserved or `buf` is shorter than the CDB.
    pub fn parse(buf: &[u8], blocksize: u32, dev_type: u8) -> Option<ScsiCommand> {
        let len = cdb_length(buf)?;
        if len > buf.len() {
            return None;
        }
        let mut cmd = ScsiCommand { len, ..ScsiCommand::default() };
        cmd.buf[..len].copy_from_slice(&buf[..len]);
        cmd.xfer = req_xfer(&cmd.buf, blocksize, dev_type);
        cmd.mode = xfer_mode(&cmd.buf, cmd.xfer);
        cmd.lba = cdb_lba(&cmd.buf);
        Some(cmd)
    }

    /// The operation code.
    pub fn opcode(&self) -> u8 {
        self.buf[0]
    }
}
