// SPDX-License-Identifier: GPL-2.0-or-later

//! The SCSI bus: device addressing, requests, unit attention and the commands a target answers
//! for LUNs that have no device, from QEMU's `hw/scsi/scsi-bus.c`.

use ruvm_base::{Error, Result};

use super::cdb::{ScsiCommand, TYPE_INACTIVE, TYPE_NO_LUN, TYPE_NOT_PRESENT, opcode::*};
use super::disk::{QEMU_HW_VERSION, ScsiDisk, ScsiDiskConf};
use super::sense::{CHECK_CONDITION, GOOD, SCSI_SENSE_LEN, ScsiSense};

/// `SCSI_INQUIRY_LEN`.
const SCSI_INQUIRY_LEN: usize = 36;

/// The limits of a host bus adapter, the parts of QEMU's `SCSIBusInfo` the bus itself uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScsiBusInfo {
    /// Whether the adapter queues tagged commands, reported in the target's INQUIRY data.
    pub tcq: bool,
    /// The highest channel number.
    pub max_channel: u32,
    /// The highest target number.
    pub max_target: u32,
    /// The highest LUN.
    pub max_lun: u32,
}

/// How a request is handled, the `SCSIReqOps` QEMU picks in `scsi_req_new()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReqOps {
    /// The CDB could not be parsed.
    InvalidOpcode,
    /// The transfer length is too large.
    InvalidField,
    /// A unit attention condition is reported instead of running the command.
    UnitAttention,
    /// REPORT LUNS, REQUEST SENSE with sense pending, or a LUN without a device.
    Target,
    /// The device runs the command.
    Device,
}

/// One command on its way through the bus, QEMU's `SCSIRequest`.
///
/// Build it with [`ScsiBus::new_request`] and run it with [`ScsiBus::execute`]. The device
/// index it holds is only meaningful until devices are attached or detached.
#[derive(Clone, Debug)]
pub struct ScsiRequest {
    dev: usize,
    lun: u32,
    cmd: ScsiCommand,
    ops: ReqOps,
    sense: Option<ScsiSense>,
    status: Option<u8>,
    residual: u64,
}

impl ScsiRequest {
    /// The parsed command. A CDB that could not be parsed has length 0 and no data.
    pub fn cmd(&self) -> &ScsiCommand {
        &self.cmd
    }

    /// The LUN the command was sent to.
    pub fn lun(&self) -> u32 {
        self.lun
    }

    /// The index of the device on the bus.
    pub fn device(&self) -> usize {
        self.dev
    }

    /// The SCSI status, `None` until the request ran.
    pub fn status(&self) -> Option<u8> {
        self.status
    }

    /// The bytes of the transfer length that were not transferred.
    pub fn residual(&self) -> u64 {
        self.residual
    }

    /// The sense of a request that ended in CHECK CONDITION.
    pub fn sense(&self) -> Option<ScsiSense> {
        self.sense
    }
}

/// A SCSI bus with the devices attached to it, QEMU's `SCSIBus`.
#[derive(Debug)]
pub struct ScsiBus {
    info: ScsiBusInfo,
    devices: Vec<ScsiDisk>,
    unit_attention: ScsiSense,
}

fn store_lun(out: &mut [u8], lun: u32) {
    if lun < 256 {
        // Simple logical unit addressing.
        out[0] = 0;
        out[1] = lun as u8;
    } else {
        // Flat space addressing.
        out[0] = 0x40 | (lun >> 8) as u8;
        out[1] = lun as u8;
    }
}

impl ScsiBus {
    /// An empty bus for an adapter with the given limits.
    pub fn new(info: ScsiBusInfo) -> Self {
        ScsiBus { info, devices: Vec::new(), unit_attention: ScsiSense::NO_SENSE }
    }

    /// The adapter limits.
    pub fn info(&self) -> &ScsiBusInfo {
        &self.info
    }

    /// The attached devices, in the order they were attached.
    pub fn devices(&self) -> &[ScsiDisk] {
        &self.devices
    }

    /// A device by index.
    pub fn device(&self, index: usize) -> Option<&ScsiDisk> {
        self.devices.get(index)
    }

    /// A device by index, for media changes and the like.
    pub fn device_mut(&mut self, index: usize) -> Option<&mut ScsiDisk> {
        self.devices.get_mut(index)
    }

    /// The bus wide unit attention condition.
    pub fn unit_attention(&self) -> ScsiSense {
        self.unit_attention
    }

    /// `do_scsi_device_find()`: the device at exactly this address, or else the first one on the
    /// same channel and target.
    pub fn find(&self, channel: u32, id: u32, lun: u32) -> Option<usize> {
        let mut found = None;
        for (i, d) in self.devices.iter().enumerate() {
            if d.channel == channel && d.id == id {
                if d.lun == lun {
                    return Some(i);
                }
                if found.is_none() {
                    found = Some(i);
                }
            }
        }
        found
    }

    fn address_user(&self, channel: u32, id: u32, lun: u32) -> Option<usize> {
        self.find(channel, id, lun).filter(|&i| self.devices[i].lun == lun)
    }

    /// Plugs a device in: `scsi_bus_check_address()`, the automatic target and LUN choice of
    /// `scsi_qdev_realize()`, then the device's own realize. Returns the device index.
    pub fn attach(&mut self, conf: ScsiDiskConf) -> Result<usize> {
        let info = self.info;
        let channel = conf.channel;
        if channel > info.max_channel {
            return Err(Error::generic(format!("bad scsi channel id: {channel}")));
        }
        if let Some(id) = conf.scsi_id {
            if id > info.max_target {
                return Err(Error::generic(format!("bad scsi device id: {id}")));
            }
        }
        if let Some(lun) = conf.lun {
            if lun > info.max_lun {
                return Err(Error::generic(format!("bad scsi device lun: {lun}")));
            }
        }
        if let (Some(id), Some(lun)) = (conf.scsi_id, conf.lun) {
            if let Some(i) = self.address_user(channel, id, lun) {
                let name = self.devices[i].qdev_id.clone().unwrap_or_default();
                return Err(Error::generic(format!("lun already used by '{name}'")));
            }
        }

        let (id, lun) = match (conf.scsi_id, conf.lun) {
            (None, lun) => {
                let lun = lun.unwrap_or(0);
                let id = (0..=info.max_target)
                    .find(|&id| self.address_user(channel, id, lun).is_none())
                    .ok_or_else(|| Error::generic("no free target"))?;
                (id, lun)
            }
            (Some(id), None) => {
                let lun = (0..=info.max_lun)
                    .find(|&lun| self.address_user(channel, id, lun).is_none())
                    .ok_or_else(|| Error::generic("no free lun"))?;
                (id, lun)
            }
            (Some(id), Some(lun)) => (id, lun),
        };

        let mut disk = ScsiDisk::new(conf)?;
        disk.channel = channel;
        disk.id = id;
        disk.lun = lun;
        self.devices.push(disk);
        Ok(self.devices.len() - 1)
    }

    /// Unplugs the device at an exact address.
    pub fn detach(&mut self, channel: u32, id: u32, lun: u32) -> Option<ScsiDisk> {
        let i = self.address_user(channel, id, lun)?;
        Some(self.devices.remove(i))
    }

    /// `scsi_bus_set_ua()`: raises a unit attention on the whole bus unless a more important
    /// one is pending.
    pub fn set_ua(&mut self, sense: ScsiSense) {
        if !sense.is_unit_attention() {
            return;
        }
        if sense.ua_precedence() < self.unit_attention.ua_precedence() {
            self.unit_attention = sense;
        }
    }

    /// Resets every device, as a bus reset does.
    pub fn reset(&mut self) {
        for d in &mut self.devices {
            d.reset();
        }
    }

    /// `scsi_req_new()`: parses a CDB sent to LUN `lun` through device `dev` (usually what
    /// [`ScsiBus::find`] returned) and decides how it runs. A pending unit attention is taken
    /// from the device or bus right here, so it is consumed even if the request never runs.
    ///
    /// # Panics
    ///
    /// If `dev` is not a device index.
    pub fn new_request(&mut self, dev: usize, lun: u32, cdb: &[u8]) -> ScsiRequest {
        let bus_ua = self.unit_attention.is_unit_attention();
        let d = &self.devices[dev];
        let mut req = ScsiRequest {
            dev,
            lun,
            cmd: ScsiCommand::default(),
            ops: ReqOps::InvalidOpcode,
            sense: None,
            status: None,
            residual: 0,
        };
        let Some(&op) = cdb.first() else {
            return req;
        };
        let ops = if (d.unit_attention.is_unit_attention() || bus_ua)
            && op != INQUIRY
            && op != REPORT_LUNS
            && op != GET_CONFIGURATION
            && op != GET_EVENT_STATUS_NOTIFICATION
            && !(op == REQUEST_SENSE && d.sense_is_ua)
        {
            ReqOps::UnitAttention
        } else if lun != d.lun || op == REPORT_LUNS || (op == REQUEST_SENSE && d.sense.is_some()) {
            ReqOps::Target
        } else {
            ReqOps::Device
        };
        let Some(cmd) = ScsiCommand::parse(cdb, d.blocksize, d.dev_type) else {
            return req;
        };
        req.cmd = cmd;
        req.residual = cmd.xfer;
        req.ops = if cmd.xfer > i32::MAX as u64 { ReqOps::InvalidField } else { ops };
        if req.ops == ReqOps::UnitAttention {
            // scsi_fetch_unit_attention_sense().
            let d = &mut self.devices[dev];
            let ua = if d.unit_attention.is_unit_attention() {
                &mut d.unit_attention
            } else {
                &mut self.unit_attention
            };
            req.sense = Some(*ua);
            *ua = ScsiSense::NO_SENSE;
        }
        req
    }

    /// Runs a request to completion. `data_out` holds the data of a command that sends some;
    /// missing bytes read as zero. Returns the data for the initiator, at most the transfer
    /// length. The request gets its status, sense and residual, and the device remembers the
    /// sense for REQUEST SENSE, as `scsi_req_complete()` does.
    ///
    /// # Panics
    ///
    /// If the request was built for a device index that no longer exists.
    pub fn execute(&mut self, req: &mut ScsiRequest, data_out: &[u8]) -> Vec<u8> {
        let (status, data) = match req.ops {
            ReqOps::InvalidOpcode => {
                req.sense = Some(ScsiSense::INVALID_OPCODE);
                (CHECK_CONDITION, Vec::new())
            }
            ReqOps::InvalidField => {
                req.sense = Some(ScsiSense::INVALID_FIELD);
                (CHECK_CONDITION, Vec::new())
            }
            ReqOps::UnitAttention => (CHECK_CONDITION, Vec::new()),
            ReqOps::Target => self.target_command(req),
            ReqOps::Device => {
                let outcome = self.devices[req.dev].execute(&req.cmd, data_out);
                req.sense = outcome.sense;
                (outcome.status, outcome.data)
            }
        };
        let transferred = if req.cmd.mode == super::cdb::XferMode::ToDev {
            if status == GOOD { req.cmd.xfer.min(data_out.len() as u64) } else { 0 }
        } else {
            data.len() as u64
        };
        req.residual = req.cmd.xfer.saturating_sub(transferred);
        self.complete(req, status);
        data
    }

    /// `scsi_req_complete()`.
    fn complete(&mut self, req: &mut ScsiRequest, status: u8) {
        req.status = Some(status);
        if status == GOOD {
            req.sense = None;
        }
        let d = &mut self.devices[req.dev];
        d.sense = req.sense;
        d.sense_is_ua = req.sense.is_some() && req.ops == ReqOps::UnitAttention;
    }

    /// `scsi_req_get_sense()`: the autosense data of a finished request in fixed format, at most
    /// `len` bytes, empty if there is none. Reporting a unit attention this way clears it, as it
    /// would for an adapter with autosense.
    pub fn get_sense(&mut self, req: &ScsiRequest, len: usize) -> Vec<u8> {
        let Some(sense) = req.sense else {
            return Vec::new();
        };
        let out = sense.to_buf(len, true);
        if let Some(d) = self.devices.get_mut(req.dev) {
            if d.sense_is_ua {
                d.unit_attention_reported();
                d.sense = None;
                d.sense_is_ua = false;
            }
        }
        out
    }

    /// `scsi_target_send_command()` and the data it returns.
    fn target_command(&mut self, req: &mut ScsiRequest) -> (u8, Vec<u8>) {
        let op = req.cmd.opcode();
        if req.lun != 0 && op != INQUIRY && op != REQUEST_SENSE {
            req.sense = Some(ScsiSense::LUN_NOT_SUPPORTED);
            return (CHECK_CONDITION, Vec::new());
        }
        let fixed = req.cmd.buf[1] & 1 == 0;
        let data = match op {
            REPORT_LUNS => self.report_luns(req),
            INQUIRY => self.target_inquiry(req),
            REQUEST_SENSE => {
                let xfer = req.cmd.xfer as usize;
                let d = &mut self.devices[req.dev];
                let data = if req.lun != 0 {
                    ScsiSense::LUN_NOT_SUPPORTED.to_buf(xfer.min(SCSI_SENSE_LEN), fixed)
                } else {
                    d.sense.unwrap_or_default().to_buf(xfer.min(SCSI_SENSE_LEN), fixed)
                };
                if d.sense_is_ua {
                    d.unit_attention_reported();
                    d.sense = None;
                    d.sense_is_ua = false;
                }
                Some(data)
            }
            TEST_UNIT_READY => Some(Vec::new()),
            _ => {
                req.sense = Some(ScsiSense::INVALID_OPCODE);
                return (CHECK_CONDITION, Vec::new());
            }
        };
        match data {
            Some(data) => (GOOD, data),
            None => {
                req.sense = Some(ScsiSense::INVALID_FIELD);
                (CHECK_CONDITION, Vec::new())
            }
        }
    }

    /// `scsi_target_emulate_report_luns()`.
    fn report_luns(&mut self, req: &ScsiRequest) -> Option<Vec<u8>> {
        if req.cmd.xfer < 16 || req.cmd.buf[2] > 2 {
            return None;
        }
        let (channel, id) = {
            let d = &self.devices[req.dev];
            (d.channel, d.id)
        };
        // The header, then LUN 0 whether or not it exists.
        let mut buf = vec![0u8; 16];
        for d in &self.devices {
            if d.channel == channel && d.id == id && d.lun != 0 {
                let mut entry = [0u8; 8];
                store_lun(&mut entry, d.lun);
                buf.extend_from_slice(&entry);
            }
        }
        let list_len = (buf.len() - 8) as u32;
        buf[..4].copy_from_slice(&list_len.to_be_bytes());
        buf.truncate(buf.len().min((req.cmd.xfer & !7) as usize));

        // scsi_clear_reported_luns_changed().
        let d = &mut self.devices[req.dev];
        let ua = if d.unit_attention.is_unit_attention() {
            Some(&mut d.unit_attention)
        } else if self.unit_attention.is_unit_attention() {
            Some(&mut self.unit_attention)
        } else {
            None
        };
        if let Some(ua) = ua {
            let changed = ScsiSense::REPORTED_LUNS_CHANGED;
            if ua.asc == changed.asc && ua.ascq == changed.ascq {
                *ua = ScsiSense::NO_SENSE;
            }
        }
        Some(buf)
    }

    /// `scsi_target_emulate_inquiry()` for a LUN without a device.
    fn target_inquiry(&self, req: &ScsiRequest) -> Option<Vec<u8>> {
        let buf = &req.cmd.buf;
        let xfer = req.cmd.xfer as usize;
        if buf[1] & 2 != 0 {
            // Command support data, which is optional.
            return None;
        }
        if buf[1] & 1 != 0 {
            // Only the supported pages page. QEMU leaves the page code where the peripheral
            // type belongs.
            if buf[2] != 0 {
                return None;
            }
            let mut out = vec![buf[2], 0, 1, 0];
            out.truncate(xfer.min(4));
            return Some(out);
        }
        if buf[2] != 0 {
            return None;
        }
        let len = xfer.min(SCSI_INQUIRY_LEN);
        let mut out = vec![0u8; SCSI_INQUIRY_LEN];
        if req.lun != 0 {
            out[0] = TYPE_NO_LUN;
        } else {
            out[0] = TYPE_NOT_PRESENT | TYPE_INACTIVE;
            out[2] = 5;
            out[3] = 2 | 0x10;
            out[4] = (len as u8).wrapping_sub(5);
            out[7] = 0x10 | if self.info.tcq { 0x02 } else { 0 };
            out[8..16].copy_from_slice(b"QEMU    ");
            out[16..32].copy_from_slice(b"QEMU TARGET     ");
            // pstrcpy() with a size of 4 keeps three characters and a terminator.
            let v = QEMU_HW_VERSION.as_bytes();
            let n = v.len().min(3);
            out[32..32 + n].copy_from_slice(&v[..n]);
        }
        out.truncate(len);
        Some(out)
    }

    /// `scsi_device_report_change()` without the adapter callback: raises `sense` on the
    /// device. The adapter reports the change to the guest its own way.
    pub fn report_change(&mut self, index: usize, sense: ScsiSense) {
        if let Some(d) = self.devices.get_mut(index) {
            d.set_ua(sense);
        }
    }
}
