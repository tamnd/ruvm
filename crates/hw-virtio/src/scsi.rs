// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-scsi, a port of `hw/scsi/virtio-scsi.c`.
//!
//! The device is a SCSI host bus adapter. It owns a [`ScsiBus`] from `ruvm-hw-storage`, and the
//! `scsi-hd` and `scsi-cd` devices on that bus do the actual SCSI work. Disks talk to images
//! through `ruvm_hw_storage::BlockBackend`, the same trait the IDE and AHCI controllers use;
//! virtio-blk keeps its own smaller trait in [`crate::blk`]. The SCSI side lives in
//! `ruvm-hw-storage` because that is where every other storage device is, so SCSI disks use
//! its backend trait and nothing here adds a third one.
//!
//! The queues are the control queue, the event queue and one or more command queues, in that
//! order. Commands, task management functions, asynchronous notification queries, hotplug and
//! parameter change events are handled as QEMU handles them, including the
//! `VIRTIO_F_ANY_LAYOUT` workaround for old BIOSes and the dropped events flag.
//!
//! Differences from QEMU:
//!
//! - Commands run to completion synchronously while the queue is serviced. Every command in a
//!   batch is prepared first (which is when a unit attention is consumed) and then run, the
//!   order QEMU uses, but there are never outstanding commands, so `ABORT TASK`,
//!   `ABORT TASK SET`, `CLEAR TASK SET`, `QUERY TASK` and `QUERY TASK SET` always complete with
//!   `VIRTIO_SCSI_S_OK`.
//! - The residual of a command that moved data is the guest buffer size minus the bytes moved.
//!   QEMU computes the same thing for most commands, but for reads and writes done through its
//!   DMA helpers it reports the transfer length minus the buffer size, which only differs when
//!   the guest supplies more buffer than the command needs.
//! - Hot plug is a method call ([`VirtioScsi::hotplug`] and [`VirtioScsi::hot_unplug`]) rather
//!   than a hotplug handler.
//!
//! Not ported: iothreads and dataplane, T10 PI, VMState, request migration, trace points and
//! QOM registration.

use std::any::Any;

use ruvm_base::{Error, Result};
use ruvm_hw_storage::scsi::{
    GOOD, ScsiBus, ScsiBusInfo, ScsiDisk, ScsiDiskConf, ScsiRequest, ScsiSense, TYPE_ROM, XferMode,
};
use ruvm_virtio_queue::DescriptorChain;

use crate::virtio::{
    VIRTIO_CONFIG_S_DRIVER_OK, VIRTIO_F_ANY_LAYOUT, VIRTIO_QUEUE_MAX, VirtIODevice,
    VirtioDeviceClass, feature,
};

/// `TYPE_VIRTIO_SCSI`.
pub const TYPE_VIRTIO_SCSI: &str = "virtio-scsi-device";
/// `VIRTIO_ID_SCSI`.
pub const VIRTIO_ID_SCSI: u16 = 8;

/// `VIRTIO_SCSI_F_INOUT`.
pub const VIRTIO_SCSI_F_INOUT: u32 = 0;
/// `VIRTIO_SCSI_F_HOTPLUG`: the device reports hot plug and unplug on the event queue.
pub const VIRTIO_SCSI_F_HOTPLUG: u32 = 1;
/// `VIRTIO_SCSI_F_CHANGE`: the device reports parameter changes on the event queue.
pub const VIRTIO_SCSI_F_CHANGE: u32 = 2;
/// `VIRTIO_SCSI_F_T10_PI`.
pub const VIRTIO_SCSI_F_T10_PI: u32 = 3;

/// `VIRTIO_SCSI_VQ_NUM_FIXED`: the control and event queues.
pub const VIRTIO_SCSI_VQ_NUM_FIXED: u32 = 2;
/// `VIRTIO_SCSI_CDB_DEFAULT_SIZE`.
pub const VIRTIO_SCSI_CDB_DEFAULT_SIZE: u32 = 32;
/// `VIRTIO_SCSI_SENSE_DEFAULT_SIZE`.
pub const VIRTIO_SCSI_SENSE_DEFAULT_SIZE: u32 = 96;
/// `VIRTIO_SCSI_MAX_CHANNEL`.
pub const VIRTIO_SCSI_MAX_CHANNEL: u32 = 0;
/// `VIRTIO_SCSI_MAX_TARGET`.
pub const VIRTIO_SCSI_MAX_TARGET: u32 = 255;
/// `VIRTIO_SCSI_MAX_LUN`.
pub const VIRTIO_SCSI_MAX_LUN: u32 = 16383;
/// The size of `struct virtio_scsi_config`.
pub const VIRTIO_SCSI_CONFIG_SIZE: usize = 36;

/// `VIRTIO_SCSI_S_OK`, which is also "function complete" for task management.
pub const VIRTIO_SCSI_S_OK: u8 = 0;
/// `VIRTIO_SCSI_S_OVERRUN`.
pub const VIRTIO_SCSI_S_OVERRUN: u8 = 1;
/// `VIRTIO_SCSI_S_ABORTED`.
pub const VIRTIO_SCSI_S_ABORTED: u8 = 2;
/// `VIRTIO_SCSI_S_BAD_TARGET`.
pub const VIRTIO_SCSI_S_BAD_TARGET: u8 = 3;
/// `VIRTIO_SCSI_S_RESET`.
pub const VIRTIO_SCSI_S_RESET: u8 = 4;
/// `VIRTIO_SCSI_S_BUSY`.
pub const VIRTIO_SCSI_S_BUSY: u8 = 5;
/// `VIRTIO_SCSI_S_TRANSPORT_FAILURE`.
pub const VIRTIO_SCSI_S_TRANSPORT_FAILURE: u8 = 6;
/// `VIRTIO_SCSI_S_TARGET_FAILURE`.
pub const VIRTIO_SCSI_S_TARGET_FAILURE: u8 = 7;
/// `VIRTIO_SCSI_S_NEXUS_FAILURE`.
pub const VIRTIO_SCSI_S_NEXUS_FAILURE: u8 = 8;
/// `VIRTIO_SCSI_S_FAILURE`.
pub const VIRTIO_SCSI_S_FAILURE: u8 = 9;
/// `VIRTIO_SCSI_S_FUNCTION_SUCCEEDED`.
pub const VIRTIO_SCSI_S_FUNCTION_SUCCEEDED: u8 = 10;
/// `VIRTIO_SCSI_S_FUNCTION_REJECTED`.
pub const VIRTIO_SCSI_S_FUNCTION_REJECTED: u8 = 11;
/// `VIRTIO_SCSI_S_INCORRECT_LUN`.
pub const VIRTIO_SCSI_S_INCORRECT_LUN: u8 = 12;

/// `VIRTIO_SCSI_T_TMF`: a task management function on the control queue.
pub const VIRTIO_SCSI_T_TMF: u32 = 0;
/// `VIRTIO_SCSI_T_AN_QUERY`.
pub const VIRTIO_SCSI_T_AN_QUERY: u32 = 1;
/// `VIRTIO_SCSI_T_AN_SUBSCRIBE`.
pub const VIRTIO_SCSI_T_AN_SUBSCRIBE: u32 = 2;

/// `VIRTIO_SCSI_T_TMF_ABORT_TASK`.
pub const VIRTIO_SCSI_T_TMF_ABORT_TASK: u32 = 0;
/// `VIRTIO_SCSI_T_TMF_ABORT_TASK_SET`.
pub const VIRTIO_SCSI_T_TMF_ABORT_TASK_SET: u32 = 1;
/// `VIRTIO_SCSI_T_TMF_CLEAR_ACA`.
pub const VIRTIO_SCSI_T_TMF_CLEAR_ACA: u32 = 2;
/// `VIRTIO_SCSI_T_TMF_CLEAR_TASK_SET`.
pub const VIRTIO_SCSI_T_TMF_CLEAR_TASK_SET: u32 = 3;
/// `VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET`.
pub const VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET: u32 = 4;
/// `VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET`.
pub const VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET: u32 = 5;
/// `VIRTIO_SCSI_T_TMF_QUERY_TASK`.
pub const VIRTIO_SCSI_T_TMF_QUERY_TASK: u32 = 6;
/// `VIRTIO_SCSI_T_TMF_QUERY_TASK_SET`.
pub const VIRTIO_SCSI_T_TMF_QUERY_TASK_SET: u32 = 7;

/// `VIRTIO_SCSI_T_NO_EVENT`.
pub const VIRTIO_SCSI_T_NO_EVENT: u32 = 0;
/// `VIRTIO_SCSI_T_TRANSPORT_RESET`.
pub const VIRTIO_SCSI_T_TRANSPORT_RESET: u32 = 1;
/// `VIRTIO_SCSI_T_ASYNC_NOTIFY`.
pub const VIRTIO_SCSI_T_ASYNC_NOTIFY: u32 = 2;
/// `VIRTIO_SCSI_T_PARAM_CHANGE`.
pub const VIRTIO_SCSI_T_PARAM_CHANGE: u32 = 3;
/// `VIRTIO_SCSI_T_EVENTS_MISSED`: or-ed into the next event after one was dropped.
pub const VIRTIO_SCSI_T_EVENTS_MISSED: u32 = 0x8000_0000;

/// `VIRTIO_SCSI_EVT_RESET_HARD`.
pub const VIRTIO_SCSI_EVT_RESET_HARD: u32 = 0;
/// `VIRTIO_SCSI_EVT_RESET_RESCAN`: a device appeared.
pub const VIRTIO_SCSI_EVT_RESET_RESCAN: u32 = 1;
/// `VIRTIO_SCSI_EVT_RESET_REMOVED`: a device went away.
pub const VIRTIO_SCSI_EVT_RESET_REMOVED: u32 = 2;

/// Size of `struct virtio_scsi_cmd_req` without the CDB.
const CMD_REQ_SIZE: usize = 19;
/// Size of `struct virtio_scsi_cmd_resp` without the sense data.
const CMD_RESP_SIZE: usize = 12;
/// Size of `struct virtio_scsi_ctrl_tmf_req`.
const TMF_REQ_SIZE: usize = 24;
/// Size of `struct virtio_scsi_ctrl_tmf_resp`.
const TMF_RESP_SIZE: usize = 1;
/// Size of `struct virtio_scsi_ctrl_an_req`.
const AN_REQ_SIZE: usize = 16;
/// Size of `struct virtio_scsi_ctrl_an_resp`.
const AN_RESP_SIZE: usize = 5;
/// Size of `struct virtio_scsi_event`.
const EVENT_SIZE: usize = 16;
/// `SCSI_SENSE_BUF_SIZE`.
const SENSE_BUF_SIZE: usize = 252;

const CTRL_VQ: u16 = 0;
const EVENT_VQ: u16 = 1;

/// The virtio-scsi properties, with QEMU's defaults.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtioScsiConf {
    /// `num_queues`: command queues. `None` is QEMU's automatic choice, which is 1 unless the
    /// PCI transport picks one per vCPU.
    pub num_queues: Option<u32>,
    /// `virtqueue_size`.
    pub virtqueue_size: u32,
    /// `seg_max_adjust`: report `virtqueue_size - 2` segments instead of 126.
    pub seg_max_adjust: bool,
    /// `max_sectors`.
    pub max_sectors: u32,
    /// `cmd_per_lun`.
    pub cmd_per_lun: u32,
    /// `hotplug`: offer `VIRTIO_SCSI_F_HOTPLUG`.
    pub hotplug: bool,
    /// `param_change`: offer `VIRTIO_SCSI_F_CHANGE`.
    pub param_change: bool,
}

impl Default for VirtioScsiConf {
    fn default() -> Self {
        VirtioScsiConf {
            num_queues: None,
            virtqueue_size: 256,
            seg_max_adjust: true,
            max_sectors: 0xFFFF,
            cmd_per_lun: 128,
            hotplug: true,
            param_change: true,
        }
    }
}

/// Why `virtio_scsi_parse_req()` failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParseError {
    /// `-EINVAL`: the headers do not fit.
    Invalid,
    /// `-ENOTSUP`: data in both directions.
    NotSupported,
}

/// A request whose headers were parsed, the parts of `VirtIOSCSIReq` that outlive parsing.
#[derive(Debug)]
struct Parsed {
    chain: DescriptorChain,
    queue: u16,
    /// The request header, `req_size` bytes.
    req: Vec<u8>,
    /// The size of the response header, `resp_iov.size`.
    resp_size: usize,
    /// Where the data starts in the device readable part.
    out_skip: u64,
    /// Where the data starts in the device writable part.
    in_skip: u64,
    out_size: u64,
    in_size: u64,
    mode: XferMode,
}

impl Parsed {
    fn qsgl_size(&self) -> u64 {
        self.out_size + self.in_size
    }
}

/// A command that is ready to run.
#[derive(Debug)]
struct Prepared {
    parsed: Parsed,
    sreq: ScsiRequest,
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn put_u16(cfg: &mut [u8], off: usize, v: u16) {
    cfg[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

fn put_u32(cfg: &mut [u8], off: usize, v: u32) {
    cfg[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

/// `virtio_scsi_get_lun()`.
fn get_lun(lun: &[u8]) -> u32 {
    ((u32::from(lun[2]) << 8) | u32::from(lun[3])) & 0x3FFF
}

/// The virtio-scsi device model, `VirtIOSCSI`.
#[derive(Debug)]
pub struct VirtioScsi {
    conf: VirtioScsiConf,
    bus: ScsiBus,
    host_features: u64,
    num_queues: u32,
    sense_size: u32,
    cdb_size: u32,
    events_dropped: bool,
}

impl VirtioScsi {
    /// A host bus adapter with properties `conf` and an empty bus. Disks can be attached to
    /// [`VirtioScsi::bus_mut`] before the device is realized.
    pub fn new(conf: VirtioScsiConf) -> Self {
        let mut host_features = 0;
        if conf.hotplug {
            host_features |= feature(VIRTIO_SCSI_F_HOTPLUG);
        }
        if conf.param_change {
            host_features |= feature(VIRTIO_SCSI_F_CHANGE);
        }
        VirtioScsi {
            conf,
            bus: ScsiBus::new(ScsiBusInfo {
                tcq: true,
                max_channel: VIRTIO_SCSI_MAX_CHANNEL,
                max_target: VIRTIO_SCSI_MAX_TARGET,
                max_lun: VIRTIO_SCSI_MAX_LUN,
            }),
            host_features,
            num_queues: 0,
            sense_size: VIRTIO_SCSI_SENSE_DEFAULT_SIZE,
            cdb_size: VIRTIO_SCSI_CDB_DEFAULT_SIZE,
            events_dropped: false,
        }
    }

    /// The properties, with the number of queues filled in once realized.
    pub fn conf(&self) -> &VirtioScsiConf {
        &self.conf
    }

    /// The SCSI bus.
    pub fn bus(&self) -> &ScsiBus {
        &self.bus
    }

    /// The SCSI bus, mutably. Devices attached here are cold plugged: the guest is not told.
    pub fn bus_mut(&mut self) -> &mut ScsiBus {
        &mut self.bus
    }

    /// The sense size the driver set in config space.
    pub fn sense_size(&self) -> u32 {
        self.sense_size
    }

    /// The CDB size the driver set in config space.
    pub fn cdb_size(&self) -> u32 {
        self.cdb_size
    }

    /// Whether an event was dropped for lack of an event buffer.
    pub fn events_dropped(&self) -> bool {
        self.events_dropped
    }

    /// `virtio_scsi_hotplug()`: attaches a disk while the guest runs and, if the driver took
    /// `VIRTIO_SCSI_F_HOTPLUG`, sends a rescan event and raises REPORTED LUNS DATA HAS CHANGED
    /// on the bus. Returns the device index.
    pub fn hotplug(&mut self, vdev: &mut VirtIODevice, conf: ScsiDiskConf) -> Result<usize> {
        let index = self.bus.attach(conf)?;
        let (_, id, lun) = self.bus.devices()[index].address();
        if vdev.has_feature(VIRTIO_SCSI_F_HOTPLUG) {
            self.push_event(
                vdev,
                VIRTIO_SCSI_T_TRANSPORT_RESET,
                VIRTIO_SCSI_EVT_RESET_RESCAN,
                id,
                lun,
            );
            self.bus.set_ua(ScsiSense::REPORTED_LUNS_CHANGED);
        }
        Ok(index)
    }

    /// `virtio_scsi_hotunplug()`: detaches a disk and, if the driver took
    /// `VIRTIO_SCSI_F_HOTPLUG`, sends a removal event and raises REPORTED LUNS DATA HAS CHANGED
    /// on the bus.
    pub fn hot_unplug(
        &mut self,
        vdev: &mut VirtIODevice,
        channel: u32,
        id: u32,
        lun: u32,
    ) -> Option<ScsiDisk> {
        let disk = self.bus.detach(channel, id, lun)?;
        if vdev.has_feature(VIRTIO_SCSI_F_HOTPLUG) {
            let (_, id, lun) = disk.address();
            self.push_event(
                vdev,
                VIRTIO_SCSI_T_TRANSPORT_RESET,
                VIRTIO_SCSI_EVT_RESET_REMOVED,
                id,
                lun,
            );
            self.bus.set_ua(ScsiSense::REPORTED_LUNS_CHANGED);
        }
        Some(disk)
    }

    /// `scsi_device_report_change()` with `virtio_scsi_change()`: raises `sense` as a unit
    /// attention on device `index` and, if the driver took `VIRTIO_SCSI_F_CHANGE` and the
    /// device is not a CD-ROM, reports a parameter change event.
    pub fn report_change(&mut self, vdev: &mut VirtIODevice, index: usize, sense: ScsiSense) {
        self.bus.report_change(index, sense);
        let Some(d) = self.bus.device(index) else {
            return;
        };
        if vdev.has_feature(VIRTIO_SCSI_F_CHANGE) && d.device_type() != TYPE_ROM {
            let (_, id, lun) = d.address();
            let reason = u32::from(sense.asc) | (u32::from(sense.ascq) << 8);
            self.push_event(vdev, VIRTIO_SCSI_T_PARAM_CHANGE, reason, id, lun);
        }
    }

    /// `virtio_scsi_device_get()`: the device a request's LUN field addresses.
    fn device_get(&self, lun: &[u8]) -> Option<usize> {
        if lun[0] != 1 {
            return None;
        }
        if lun[2] != 0 && !(0x40..0x80).contains(&lun[2]) {
            return None;
        }
        self.bus.find(0, u32::from(lun[1]), get_lun(lun))
    }

    /// `virtio_scsi_parse_req()`.
    fn parse_req(
        vdev: &VirtIODevice,
        queue: u16,
        chain: DescriptorChain,
        req_size: usize,
        resp_size: usize,
    ) -> std::result::Result<Parsed, (ParseError, Parsed)> {
        let mem = vdev.mem().clone();
        let mut req = vec![0u8; req_size];
        let req_ok = chain.readable_len() >= req_size as u64
            && chain.reader(&*mem).read_exact(&mut req).is_ok();
        let resp_ok = chain.writable_len() >= resp_size as u64;

        let (mut out_skip, mut in_skip) = (req_size as u64, resp_size as u64);
        if !vdev.has_feature(VIRTIO_F_ANY_LAYOUT) {
            if let Some(d) = chain.readable().first() {
                out_skip = u64::from(d.len());
            }
            if let Some(d) = chain.writable().first() {
                in_skip = u64::from(d.len());
            }
        }
        let out_size = chain.readable_len().saturating_sub(out_skip);
        let in_size = chain.writable_len().saturating_sub(in_skip);
        let mode = if out_size != 0 {
            XferMode::ToDev
        } else if in_size != 0 {
            XferMode::FromDev
        } else {
            XferMode::None
        };
        let parsed =
            Parsed { chain, queue, req, resp_size, out_skip, in_skip, out_size, in_size, mode };
        if !req_ok || !resp_ok {
            // The data sizes are never looked at in this case, and QEMU leaves the scatter
            // gather list empty.
            let parsed = Parsed { out_size: 0, in_size: 0, ..parsed };
            return Err((ParseError::Invalid, parsed));
        }
        if out_size != 0 && in_size != 0 {
            return Err((ParseError::NotSupported, parsed));
        }
        Ok(parsed)
    }

    /// `virtio_scsi_bad_req()`.
    fn bad_req(vdev: &mut VirtIODevice, parsed: &Parsed) {
        vdev.error("wrong size for virtio-scsi headers");
        vdev.detach(parsed.queue, &parsed.chain);
    }

    /// `virtio_scsi_complete_req()`: writes `resp` at the start of the response, returns the
    /// chain and notifies the driver.
    fn complete_req(vdev: &mut VirtIODevice, parsed: &Parsed, resp: &[u8]) {
        let mem = vdev.mem().clone();
        let n = resp.len().min(parsed.resp_size);
        let _ = parsed.chain.writer(&*mem).write_all(&resp[..n]);
        let len = parsed.qsgl_size() + parsed.resp_size as u64;
        vdev.push(parsed.queue, &parsed.chain, u32::try_from(len).unwrap_or(u32::MAX));
        vdev.notify(parsed.queue);
    }

    /// `virtio_scsi_complete_cmd_req()` for a command that never reached a device.
    fn complete_cmd_response(vdev: &mut VirtIODevice, parsed: &Parsed, response: u8) {
        let mut resp = [0u8; CMD_RESP_SIZE];
        resp[11] = response;
        Self::complete_req(vdev, parsed, &resp);
    }

    /// `virtio_scsi_handle_cmd_req_prepare()`. `Err` is a request so broken the device stopped.
    fn prepare_cmd(
        &mut self,
        vdev: &mut VirtIODevice,
        queue: u16,
        chain: DescriptorChain,
    ) -> std::result::Result<Option<Prepared>, ()> {
        let cdb_size = self.cdb_size as usize;
        let parsed = match Self::parse_req(
            vdev,
            queue,
            chain,
            CMD_REQ_SIZE + cdb_size,
            CMD_RESP_SIZE + self.sense_size as usize,
        ) {
            Ok(parsed) => parsed,
            Err((ParseError::NotSupported, parsed)) => {
                Self::complete_cmd_response(vdev, &parsed, VIRTIO_SCSI_S_FAILURE);
                return Ok(None);
            }
            Err((ParseError::Invalid, parsed)) => {
                Self::bad_req(vdev, &parsed);
                return Err(());
            }
        };
        let lun = &parsed.req[0..8];
        let Some(dev) = self.device_get(lun) else {
            Self::complete_cmd_response(vdev, &parsed, VIRTIO_SCSI_S_BAD_TARGET);
            return Ok(None);
        };
        let cdb = &parsed.req[CMD_REQ_SIZE..];
        let sreq = self.bus.new_request(dev, get_lun(lun), cdb);
        let cmd = sreq.cmd();
        if cmd.mode != XferMode::None && (cmd.mode != parsed.mode || cmd.xfer > parsed.qsgl_size())
        {
            Self::complete_cmd_response(vdev, &parsed, VIRTIO_SCSI_S_OVERRUN);
            return Ok(None);
        }
        Ok(Some(Prepared { parsed, sreq }))
    }

    /// `virtio_scsi_handle_cmd_req_submit()` through `virtio_scsi_command_complete()`.
    fn submit_cmd(&mut self, vdev: &mut VirtIODevice, prepared: Prepared) {
        let Prepared { parsed, mut sreq } = prepared;
        let mem = vdev.mem().clone();
        let xfer = sreq.cmd().xfer;
        let mut data_out = Vec::new();
        if sreq.cmd().mode == XferMode::ToDev {
            let len = xfer.min(parsed.out_size) as usize;
            data_out.resize(len, 0);
            let mut reader = parsed.chain.reader(&*mem);
            reader.skip(parsed.out_skip);
            let _ = reader.read_exact(&mut data_out);
        }
        let data_in = self.bus.execute(&mut sreq, &data_out);
        if !data_in.is_empty() {
            let n = data_in.len().min(parsed.in_size as usize);
            let mut writer = parsed.chain.writer(&*mem);
            writer.skip(parsed.in_skip);
            let _ = writer.write_all(&data_in[..n]);
        }

        let status = sreq.status().unwrap_or(GOOD);
        let mut resp = vec![0u8; CMD_RESP_SIZE];
        resp[10] = status;
        resp[11] = VIRTIO_SCSI_S_OK;
        if status == GOOD {
            let moved = xfer - sreq.residual();
            let resid = if moved != 0 { parsed.qsgl_size() - moved } else { sreq.residual() };
            put_u32(&mut resp, 4, resid as u32);
        } else {
            let sense = self.bus.get_sense(&sreq, SENSE_BUF_SIZE);
            let n = sense.len().min(parsed.resp_size - CMD_RESP_SIZE);
            put_u32(&mut resp, 0, n as u32);
            resp.extend_from_slice(&sense[..n]);
        }
        Self::complete_req(vdev, &parsed, &resp);
    }

    /// `virtio_scsi_handle_cmd_vq()`.
    fn handle_cmd_vq(&mut self, vdev: &mut VirtIODevice, q: u16) {
        let suppress_notifications = vdev.queue_notification(q);
        let mut reqs = Vec::new();
        let mut broken = false;
        loop {
            if suppress_notifications {
                vdev.set_queue_notification(q, false);
            }
            while let Some(chain) = vdev.pop(q) {
                match self.prepare_cmd(vdev, q, chain) {
                    Ok(Some(prepared)) => reqs.push(prepared),
                    Ok(None) => {}
                    Err(()) => {
                        for prepared in reqs.drain(..) {
                            vdev.detach(q, &prepared.parsed.chain);
                        }
                        broken = true;
                    }
                }
            }
            if suppress_notifications {
                vdev.set_queue_notification(q, true);
            }
            if broken || vdev.queue_empty(q) {
                break;
            }
        }
        for prepared in reqs {
            self.submit_cmd(vdev, prepared);
        }
    }

    /// `virtio_scsi_do_tmf()`: the response code.
    fn do_tmf(&mut self, req: &[u8]) -> u8 {
        let subtype = le32(req, 4);
        let lun = &req[8..16];
        let d = self.device_get(lun);
        let check = |bus: &ScsiBus| -> std::result::Result<usize, u8> {
            let Some(i) = d else {
                return Err(VIRTIO_SCSI_S_BAD_TARGET);
            };
            if bus.devices()[i].address().2 != get_lun(lun) {
                return Err(VIRTIO_SCSI_S_INCORRECT_LUN);
            }
            Ok(i)
        };
        match subtype {
            VIRTIO_SCSI_T_TMF_ABORT_TASK
            | VIRTIO_SCSI_T_TMF_QUERY_TASK
            | VIRTIO_SCSI_T_TMF_ABORT_TASK_SET
            | VIRTIO_SCSI_T_TMF_CLEAR_TASK_SET
            | VIRTIO_SCSI_T_TMF_QUERY_TASK_SET => match check(&self.bus) {
                // Commands never stay outstanding, so there is nothing to abort or find.
                Ok(_) => VIRTIO_SCSI_S_OK,
                Err(response) => response,
            },
            VIRTIO_SCSI_T_TMF_LOGICAL_UNIT_RESET => match check(&self.bus) {
                Ok(i) => {
                    if let Some(d) = self.bus.device_mut(i) {
                        d.reset();
                    }
                    VIRTIO_SCSI_S_OK
                }
                Err(response) => response,
            },
            VIRTIO_SCSI_T_TMF_I_T_NEXUS_RESET => {
                let target = u32::from(lun[1]);
                let matching: Vec<usize> = (0..self.bus.devices().len())
                    .filter(|&i| {
                        let (channel, id, _) = self.bus.devices()[i].address();
                        channel == 0 && id == target
                    })
                    .collect();
                for i in matching {
                    if let Some(d) = self.bus.device_mut(i) {
                        d.reset();
                    }
                }
                VIRTIO_SCSI_S_OK
            }
            _ => VIRTIO_SCSI_S_FUNCTION_REJECTED,
        }
    }

    /// `virtio_scsi_handle_ctrl_req()`.
    fn handle_ctrl_req(&mut self, vdev: &mut VirtIODevice, chain: DescriptorChain) {
        let mem = vdev.mem().clone();
        let mut type_buf = [0u8; 4];
        let type_ok =
            chain.readable_len() >= 4 && chain.reader(&*mem).read_exact(&mut type_buf).is_ok();
        if !type_ok {
            vdev.error("wrong size for virtio-scsi headers");
            vdev.detach(CTRL_VQ, &chain);
            return;
        }
        match u32::from_le_bytes(type_buf) {
            VIRTIO_SCSI_T_TMF => {
                match Self::parse_req(vdev, CTRL_VQ, chain, TMF_REQ_SIZE, TMF_RESP_SIZE) {
                    Ok(parsed) => {
                        let response = self.do_tmf(&parsed.req);
                        Self::complete_req(vdev, &parsed, &[response]);
                    }
                    Err((_, parsed)) => Self::bad_req(vdev, &parsed),
                }
            }
            VIRTIO_SCSI_T_AN_QUERY | VIRTIO_SCSI_T_AN_SUBSCRIBE => {
                match Self::parse_req(vdev, CTRL_VQ, chain, AN_REQ_SIZE, AN_RESP_SIZE) {
                    // event_actual 0, response VIRTIO_SCSI_S_OK.
                    Ok(parsed) => Self::complete_req(vdev, &parsed, &[0u8; AN_RESP_SIZE]),
                    Err((_, parsed)) => Self::bad_req(vdev, &parsed),
                }
            }
            _ => {
                // An unknown type completes with nothing written.
                vdev.push(CTRL_VQ, &chain, 0);
                vdev.notify(CTRL_VQ);
            }
        }
    }

    /// `virtio_scsi_push_event()`.
    fn push_event(&mut self, vdev: &mut VirtIODevice, event: u32, reason: u32, id: u32, lun: u32) {
        if vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK == 0 {
            return;
        }
        let Some(chain) = vdev.pop(EVENT_VQ) else {
            self.events_dropped = true;
            return;
        };
        let mut event = event;
        if self.events_dropped {
            event |= VIRTIO_SCSI_T_EVENTS_MISSED;
            self.events_dropped = false;
        }
        let parsed = match Self::parse_req(vdev, EVENT_VQ, chain, 0, EVENT_SIZE) {
            Ok(parsed) => parsed,
            Err((_, parsed)) => {
                Self::bad_req(vdev, &parsed);
                return;
            }
        };
        let mut evt = [0u8; EVENT_SIZE];
        put_u32(&mut evt, 0, event);
        if event != VIRTIO_SCSI_T_EVENTS_MISSED {
            evt[4] = 1;
            evt[5] = id as u8;
            if lun >= 256 {
                evt[6] = ((lun >> 8) as u8) | 0x40;
            }
            evt[7] = lun as u8;
        }
        put_u32(&mut evt, 12, reason);
        Self::complete_req(vdev, &parsed, &evt);
    }
}

impl VirtioDeviceClass for VirtioScsi {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let num_queues = self.conf.num_queues.unwrap_or(1);
        let max = VIRTIO_QUEUE_MAX as u32 - VIRTIO_SCSI_VQ_NUM_FIXED;
        if num_queues == 0 || num_queues > max {
            return Err(Error::generic(format!(
                "Invalid number of queues (= {num_queues}), must be a positive integer less \
                 than {max}."
            )));
        }
        let size = self.conf.virtqueue_size;
        if size <= 2 {
            return Err(Error::generic(format!(
                "invalid virtqueue_size property (= {size}), must be > 2"
            )));
        }
        self.conf.num_queues = Some(num_queues);
        self.num_queues = num_queues;
        self.sense_size = VIRTIO_SCSI_SENSE_DEFAULT_SIZE;
        self.cdb_size = VIRTIO_SCSI_CDB_DEFAULT_SIZE;
        vdev.init(TYPE_VIRTIO_SCSI, VIRTIO_ID_SCSI, VIRTIO_SCSI_CONFIG_SIZE);
        let size = u16::try_from(size).unwrap_or(u16::MAX);
        for _ in 0..num_queues + VIRTIO_SCSI_VQ_NUM_FIXED {
            vdev.add_queue(size)?;
        }
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        Ok(features | self.host_features)
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let mut cfg = [0u8; VIRTIO_SCSI_CONFIG_SIZE];
        let seg_max = if self.conf.seg_max_adjust { self.conf.virtqueue_size - 2 } else { 128 - 2 };
        put_u32(&mut cfg, 0, self.num_queues);
        put_u32(&mut cfg, 4, seg_max);
        put_u32(&mut cfg, 8, self.conf.max_sectors);
        put_u32(&mut cfg, 12, self.conf.cmd_per_lun);
        put_u32(&mut cfg, 16, EVENT_SIZE as u32);
        put_u32(&mut cfg, 20, self.sense_size);
        put_u32(&mut cfg, 24, self.cdb_size);
        put_u16(&mut cfg, 28, VIRTIO_SCSI_MAX_CHANNEL as u16);
        put_u16(&mut cfg, 30, VIRTIO_SCSI_MAX_TARGET as u16);
        put_u32(&mut cfg, 32, VIRTIO_SCSI_MAX_LUN);
        let n = config.len().min(cfg.len());
        config[..n].copy_from_slice(&cfg[..n]);
    }

    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        if config.len() < VIRTIO_SCSI_CONFIG_SIZE {
            return;
        }
        let sense_size = le32(config, 20);
        let cdb_size = le32(config, 24);
        if sense_size >= 65536 || cdb_size >= 256 {
            vdev.error("bad data written to virtio-scsi configuration space");
            return;
        }
        self.sense_size = sense_size;
        self.cdb_size = cdb_size;
    }

    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        self.bus.reset();
        self.sense_size = VIRTIO_SCSI_SENSE_DEFAULT_SIZE;
        self.cdb_size = VIRTIO_SCSI_CDB_DEFAULT_SIZE;
        self.events_dropped = false;
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        match queue {
            CTRL_VQ => {
                while let Some(chain) = vdev.pop(CTRL_VQ) {
                    self.handle_ctrl_req(vdev, chain);
                }
            }
            EVENT_VQ => {
                if self.events_dropped {
                    self.push_event(vdev, VIRTIO_SCSI_T_NO_EVENT, 0, 0, 0);
                }
            }
            _ => self.handle_cmd_vq(vdev, queue),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
