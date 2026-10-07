// SPDX-License-Identifier: GPL-2.0-or-later

//! The `scsi-disk` description with the `SCSIDevice` state it starts with.

use ruvm_base::{Error, Result};

use super::super::sense::{SCSI_SENSE_BUF_SIZE, SCSI_SENSE_LEN, ScsiSense};
use super::ScsiDisk;

/// `SCSI_SENSE_BUF_SIZE_OLD`: the part of the sense buffer in the main `SCSIDevice` fields.
/// The rest goes in the `SCSIDevice/sense` subsection, when the sense is longer.
pub const SCSI_SENSE_BUF_SIZE_OLD: usize = 96;

/// `SCSIDiskState` as `scsi-disk` version 1 has it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScsiDiskVmState {
    /// `unit_attention`: key, ASC and ASCQ.
    pub unit_attention: [u8; 3],
    pub sense_is_ua: bool,
    /// [`SCSI_SENSE_BUF_SIZE`] bytes.
    pub sense: Vec<u8>,
    pub sense_len: u32,
    pub media_changed: bool,
    pub media_event: bool,
    pub eject_request: bool,
    pub tray_open: bool,
    pub tray_locked: bool,
}

impl Default for ScsiDiskVmState {
    fn default() -> Self {
        ScsiDiskVmState {
            unit_attention: [0; 3],
            sense_is_ua: false,
            sense: vec![0; SCSI_SENSE_BUF_SIZE],
            sense_len: 0,
            media_changed: false,
            media_event: false,
            eject_request: false,
            tray_open: false,
            tray_locked: false,
        }
    }
}

impl ScsiDisk {
    /// The disk's part of the migration stream. There are never requests in flight, so none
    /// are sent.
    pub fn vmstate_save(&self) -> ScsiDiskVmState {
        let ua = self.unit_attention;
        let mut sense = vec![0; SCSI_SENSE_BUF_SIZE];
        let mut sense_len = 0;
        if let Some(s) = self.sense {
            let buf = s.to_buf(SCSI_SENSE_LEN, true);
            sense[..buf.len()].copy_from_slice(&buf);
            sense_len = buf.len() as u32;
        }
        ScsiDiskVmState {
            unit_attention: [ua.key, ua.asc, ua.ascq],
            sense_is_ua: self.sense_is_ua,
            sense,
            sense_len,
            media_changed: self.media_changed,
            media_event: self.media_event,
            eject_request: self.eject_request,
            tray_open: self.tray_open,
            tray_locked: self.tray_locked,
        }
    }

    /// Takes back what [`vmstate_save`](Self::vmstate_save) returned. Sense data is kept as
    /// the condition it reports.
    pub fn vmstate_load(&mut self, s: &ScsiDiskVmState) -> Result<()> {
        let len = s.sense_len as usize;
        if len > SCSI_SENSE_BUF_SIZE || len > s.sense.len() {
            return Err(Error::generic(format!("scsi-disk: sense length {len}")));
        }
        let [key, asc, ascq] = s.unit_attention;
        self.unit_attention = ScsiSense::new(key, asc, ascq);
        self.sense_is_ua = s.sense_is_ua;
        self.sense = if len == 0 { None } else { Some(ScsiSense::from_buf(&s.sense[..len])) };
        self.media_changed = s.media_changed;
        self.media_event = s.media_event;
        self.eject_request = s.eject_request;
        self.tray_open = s.tray_open;
        self.tray_locked = s.tray_locked;
        Ok(())
    }
}
