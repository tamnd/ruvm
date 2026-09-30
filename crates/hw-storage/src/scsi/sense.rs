// SPDX-License-Identifier: GPL-2.0-or-later

//! Sense codes and sense data, from QEMU's `scsi/utils.c`.

use std::io;

/// `SCSI_SENSE_LEN`: the size of fixed format sense data.
pub const SCSI_SENSE_LEN: usize = 18;
/// `SCSI_SENSE_BUF_SIZE`: how much sense a device or request keeps.
pub const SCSI_SENSE_BUF_SIZE: usize = 252;

/// `GOOD`.
pub const GOOD: u8 = 0x00;
/// `CHECK_CONDITION`.
pub const CHECK_CONDITION: u8 = 0x02;
/// `BUSY`.
pub const BUSY: u8 = 0x08;
/// `RESERVATION_CONFLICT`.
pub const RESERVATION_CONFLICT: u8 = 0x18;
/// `TASK_SET_FULL`.
pub const TASK_SET_FULL: u8 = 0x28;

/// Sense key `NO_SENSE`.
pub const SENSE_KEY_NO_SENSE: u8 = 0x00;
/// Sense key `NOT_READY`.
pub const SENSE_KEY_NOT_READY: u8 = 0x02;
/// Sense key `MEDIUM_ERROR`.
pub const SENSE_KEY_MEDIUM_ERROR: u8 = 0x03;
/// Sense key `HARDWARE_ERROR`.
pub const SENSE_KEY_HARDWARE_ERROR: u8 = 0x04;
/// Sense key `ILLEGAL_REQUEST`.
pub const SENSE_KEY_ILLEGAL_REQUEST: u8 = 0x05;
/// Sense key `UNIT_ATTENTION`.
pub const SENSE_KEY_UNIT_ATTENTION: u8 = 0x06;
/// Sense key `DATA_PROTECT`.
pub const SENSE_KEY_DATA_PROTECT: u8 = 0x07;
/// Sense key `ABORTED_COMMAND`.
pub const SENSE_KEY_ABORTED_COMMAND: u8 = 0x0b;

/// A sense key with its additional sense code and qualifier, QEMU's `SCSISense`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScsiSense {
    /// The sense key.
    pub key: u8,
    /// The additional sense code.
    pub asc: u8,
    /// The additional sense code qualifier.
    pub ascq: u8,
}

macro_rules! sense_codes {
    ($($(#[$doc:meta])* $name:ident = ($key:expr, $asc:expr, $ascq:expr);)*) => {
        impl ScsiSense {
            $($(#[$doc])* pub const $name: ScsiSense = ScsiSense::new($key, $asc, $ascq);)*
        }
    };
}

sense_codes! {
    /// No sense data.
    NO_SENSE = (SENSE_KEY_NO_SENSE, 0x00, 0x00);
    /// LUN not ready, manual intervention required.
    LUN_NOT_READY = (SENSE_KEY_NOT_READY, 0x04, 0x03);
    /// Medium not present.
    NO_MEDIUM = (SENSE_KEY_NOT_READY, 0x3a, 0x00);
    /// Medium removal prevented, as a NOT READY condition.
    NOT_READY_REMOVAL_PREVENTED = (SENSE_KEY_NOT_READY, 0x53, 0x02);
    /// Logical unit not ready, cause not reportable.
    NOT_READY = (SENSE_KEY_NOT_READY, 0x04, 0x00);
    /// Internal target failure.
    TARGET_FAILURE = (SENSE_KEY_HARDWARE_ERROR, 0x44, 0x00);
    /// Invalid command operation code.
    INVALID_OPCODE = (SENSE_KEY_ILLEGAL_REQUEST, 0x20, 0x00);
    /// LBA out of range.
    LBA_OUT_OF_RANGE = (SENSE_KEY_ILLEGAL_REQUEST, 0x21, 0x00);
    /// Invalid field in CDB.
    INVALID_FIELD = (SENSE_KEY_ILLEGAL_REQUEST, 0x24, 0x00);
    /// Invalid field in parameter list.
    INVALID_PARAM = (SENSE_KEY_ILLEGAL_REQUEST, 0x26, 0x00);
    /// Parameter value invalid.
    INVALID_PARAM_VALUE = (SENSE_KEY_ILLEGAL_REQUEST, 0x26, 0x01);
    /// Parameter list length error.
    INVALID_PARAM_LEN = (SENSE_KEY_ILLEGAL_REQUEST, 0x1a, 0x00);
    /// Logical unit not supported.
    LUN_NOT_SUPPORTED = (SENSE_KEY_ILLEGAL_REQUEST, 0x25, 0x00);
    /// Saving parameters not supported.
    SAVING_PARAMS_NOT_SUPPORTED = (SENSE_KEY_ILLEGAL_REQUEST, 0x39, 0x00);
    /// Incompatible medium installed.
    INCOMPATIBLE_FORMAT = (SENSE_KEY_ILLEGAL_REQUEST, 0x30, 0x00);
    /// Medium removal prevented, as an ILLEGAL REQUEST.
    ILLEGAL_REQ_REMOVAL_PREVENTED = (SENSE_KEY_ILLEGAL_REQUEST, 0x53, 0x02);
    /// Invalid tag.
    INVALID_TAG = (SENSE_KEY_ILLEGAL_REQUEST, 0x4b, 0x01);
    /// Command aborted, I/O process terminated.
    IO_ERROR = (SENSE_KEY_ABORTED_COMMAND, 0x00, 0x06);
    /// Command aborted, I_T nexus loss occurred.
    I_T_NEXUS_LOSS = (SENSE_KEY_ABORTED_COMMAND, 0x29, 0x07);
    /// Command aborted, logical unit failure.
    LUN_FAILURE = (SENSE_KEY_ABORTED_COMMAND, 0x3e, 0x01);
    /// Command aborted, overlapped commands attempted.
    OVERLAPPED_COMMANDS = (SENSE_KEY_ABORTED_COMMAND, 0x4e, 0x00);
    /// Command aborted, logical unit communication failure.
    LUN_COMM_FAILURE = (SENSE_KEY_ABORTED_COMMAND, 0x08, 0x00);
    /// Command aborted, logical unit does not respond to selection.
    LUN_NOT_RESPONDING = (SENSE_KEY_ABORTED_COMMAND, 0x05, 0x00);
    /// Command aborted, command timeout during processing.
    COMMAND_TIMEOUT = (SENSE_KEY_ABORTED_COMMAND, 0x2e, 0x02);
    /// Command aborted, commands cleared by device server.
    COMMAND_ABORTED = (SENSE_KEY_ABORTED_COMMAND, 0x2f, 0x02);
    /// Medium error, unrecovered read error.
    READ_ERROR = (SENSE_KEY_MEDIUM_ERROR, 0x11, 0x00);
    /// Unit attention, capacity data has changed.
    CAPACITY_CHANGED = (SENSE_KEY_UNIT_ATTENTION, 0x2a, 0x09);
    /// Unit attention, power on, reset or bus device reset occurred.
    RESET = (SENSE_KEY_UNIT_ATTENTION, 0x29, 0x00);
    /// Unit attention, SCSI bus reset occurred.
    SCSI_BUS_RESET = (SENSE_KEY_UNIT_ATTENTION, 0x29, 0x02);
    /// Unit attention, medium not present.
    UNIT_ATTENTION_NO_MEDIUM = (SENSE_KEY_UNIT_ATTENTION, 0x3a, 0x00);
    /// Unit attention, medium may have changed.
    MEDIUM_CHANGED = (SENSE_KEY_UNIT_ATTENTION, 0x28, 0x00);
    /// Unit attention, reported LUNs data has changed.
    REPORTED_LUNS_CHANGED = (SENSE_KEY_UNIT_ATTENTION, 0x3f, 0x0e);
    /// Unit attention, device internal reset.
    DEVICE_INTERNAL_RESET = (SENSE_KEY_UNIT_ATTENTION, 0x29, 0x04);
    /// Data protect, write protected.
    WRITE_PROTECTED = (SENSE_KEY_DATA_PROTECT, 0x27, 0x00);
    /// Data protect, space allocation failed write protect.
    SPACE_ALLOC_FAILED = (SENSE_KEY_DATA_PROTECT, 0x27, 0x07);
}

impl ScsiSense {
    /// A sense code from its three parts.
    pub const fn new(key: u8, asc: u8, ascq: u8) -> Self {
        ScsiSense { key, asc, ascq }
    }

    /// Whether this is a unit attention condition.
    pub fn is_unit_attention(self) -> bool {
        self.key == SENSE_KEY_UNIT_ATTENTION
    }

    /// `scsi_build_sense_buf()`: fixed (18 bytes) or descriptor (8 bytes) format sense data, cut
    /// to `size` bytes.
    pub fn to_buf(self, size: usize, fixed: bool) -> Vec<u8> {
        let mut buf = if fixed {
            let mut b = vec![0u8; SCSI_SENSE_LEN];
            b[0] = 0x70;
            b[2] = self.key;
            b[7] = 10;
            b[12] = self.asc;
            b[13] = self.ascq;
            b
        } else {
            vec![0x72, self.key, self.asc, self.ascq, 0, 0, 0, 0]
        };
        buf.truncate(size);
        buf
    }

    /// `scsi_parse_sense_buf()`: the code in sense data of either format. Data too short to hold
    /// one reads as [`ScsiSense::IO_ERROR`].
    pub fn from_buf(buf: &[u8]) -> Self {
        let Some(&first) = buf.first() else {
            return ScsiSense::IO_ERROR;
        };
        if first & 2 == 0 {
            if buf.len() < 14 {
                return ScsiSense::IO_ERROR;
            }
            ScsiSense::new(buf[2], buf[12], buf[13])
        } else {
            if buf.len() < 4 {
                return ScsiSense::IO_ERROR;
            }
            ScsiSense::new(buf[1], buf[2], buf[3])
        }
    }

    /// `scsi_sense_from_errno()` for a failed backend request: the status and the sense to
    /// report to the guest.
    pub fn from_io_error(err: &io::Error) -> (u8, ScsiSense) {
        match err.kind() {
            io::ErrorKind::InvalidInput => (CHECK_CONDITION, ScsiSense::INVALID_FIELD),
            io::ErrorKind::OutOfMemory => (CHECK_CONDITION, ScsiSense::TARGET_FAILURE),
            io::ErrorKind::StorageFull => (CHECK_CONDITION, ScsiSense::SPACE_ALLOC_FAILED),
            _ => (CHECK_CONDITION, ScsiSense::IO_ERROR),
        }
    }

    /// `scsi_ua_precedence()`: lower numbers win when two unit attentions compete.
    pub(crate) fn ua_precedence(self) -> i32 {
        if self.key != SENSE_KEY_UNIT_ATTENTION {
            return i32::MAX;
        }
        match (self.asc, self.ascq) {
            // DEVICE INTERNAL RESET goes with POWER ON OCCURRED.
            (0x29, 0x04) => 1,
            // MICROCODE HAS BEEN CHANGED goes with SCSI BUS RESET OCCURRED.
            (0x3f, 0x01) => 2,
            // These two go with all the others.
            (0x29, 0x05 | 0x06) => (i32::from(self.asc) << 8) | i32::from(self.ascq),
            (0x29, q) if q <= 0x07 => i32::from(q),
            // COMMANDS CLEARED BY POWER LOSS NOTIFICATION.
            (0x2f, 0x01) => 8,
            (asc, ascq) => (i32::from(asc) << 8) | i32::from(ascq),
        }
    }
}

/// `scsi_convert_sense()`: sense data in the format the caller asked for, at most `len` bytes.
/// Empty input converts to NO SENSE.
pub fn convert_sense(input: &[u8], len: usize, fixed: bool) -> Vec<u8> {
    if input.is_empty() {
        return ScsiSense::NO_SENSE.to_buf(len, fixed);
    }
    let fixed_in = input[0] & 2 == 0;
    if fixed == fixed_in {
        input[..input.len().min(len)].to_vec()
    } else {
        ScsiSense::from_buf(input).to_buf(len, fixed)
    }
}
