// SPDX-License-Identifier: GPL-2.0-or-later

//! `scsi-hd` and `scsi-cd`, from QEMU's `hw/scsi/scsi-disk.c`.

use std::io;
use std::sync::Arc;

use ruvm_base::{Error, Result};

mod vmstate;

pub use vmstate::{SCSI_SENSE_BUF_SIZE_OLD, ScsiDiskVmState};

use super::cdb::{ScsiCommand, TYPE_DISK, TYPE_ROM, XferMode, data_cdb_xfer, opcode::*};
use super::sense::{CHECK_CONDITION, GOOD, ScsiSense};
use crate::block::BlockBackend;

/// `QEMU_HW_VERSION`, the default INQUIRY revision.
pub const QEMU_HW_VERSION: &str = "2.5+";

const SCSI_WRITE_SAME_MAX: u64 = 512 * 1024;
const SCSI_MAX_INQUIRY_LEN: usize = 256;
const DEFAULT_DISCARD_GRANULARITY: u32 = 4 * 1024;
const DEFAULT_MAX_UNMAP_SIZE: u64 = 1 << 30;
const DEFAULT_MAX_IO_SIZE: u64 = i32::MAX as u64;
const MAX_SERIAL_LEN: usize = 36;
const MAX_SERIAL_LEN_FOR_DEVID: usize = 20;
const BDRV_SECTOR_SIZE: u64 = 512;
const CD_MAX_SECTORS: u64 = 80 * 60 * 75 * 2048 / 512;

const MODE_PAGE_R_W_ERROR: usize = 0x01;
const MODE_PAGE_HD_GEOMETRY: usize = 0x04;
const MODE_PAGE_FLEXIBLE_DISK_GEOMETRY: usize = 0x05;
const MODE_PAGE_CACHING: usize = 0x08;
const MODE_PAGE_AUDIO_CTL: usize = 0x0e;
const MODE_PAGE_CAPABILITIES: usize = 0x2a;
const MODE_PAGE_ALLS: usize = 0x3f;

const MMC_PROFILE_NONE: u16 = 0x0000;
const MMC_PROFILE_CD_ROM: u16 = 0x0008;
const MMC_PROFILE_DVD_ROM: u16 = 0x0010;
const GESN_MEDIA: u8 = 4;
const MS_TRAY_OPEN: u8 = 1;
const MS_MEDIA_PRESENT: u8 = 2;
const MEC_NO_CHANGE: u8 = 0;
const MEC_EJECT_REQUESTED: u8 = 1;
const MEC_NEW_MEDIA: u8 = 2;

/// Which of the two device types a [`ScsiDisk`] is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ScsiDiskKind {
    /// `scsi-hd`.
    #[default]
    Hd,
    /// `scsi-cd`.
    Cd,
}

/// The qdev properties of `scsi-hd` and `scsi-cd`.
#[derive(Clone, Debug)]
pub struct ScsiDiskConf {
    /// Hard disk or CD drive.
    pub kind: ScsiDiskKind,
    /// `drive`. A CD drive may be empty.
    pub drive: Option<Arc<dyn BlockBackend>>,
    /// The block backend's name, the default device identification when there is no serial.
    pub drive_name: Option<String>,
    /// The qdev id, used in the "lun already used" error.
    pub id: Option<String>,
    /// `channel`.
    pub channel: u32,
    /// `scsi-id`: `None` takes the first free target.
    pub scsi_id: Option<u32>,
    /// `lun`: `None` takes the first free LUN (or LUN 0 with an automatic target).
    pub lun: Option<u32>,
    /// `ver`.
    pub version: Option<String>,
    /// `serial`.
    pub serial: Option<String>,
    /// `vendor`.
    pub vendor: Option<String>,
    /// `product`.
    pub product: Option<String>,
    /// `device_id`.
    pub device_id: Option<String>,
    /// `removable` (scsi-hd only, scsi-cd always is).
    pub removable: bool,
    /// `dpofua` (scsi-hd only).
    pub dpofua: bool,
    /// `wwn`.
    pub wwn: u64,
    /// `port_wwn`.
    pub port_wwn: u64,
    /// `port_index`.
    pub port_index: u16,
    /// `max_unmap_size` in bytes.
    pub max_unmap_size: u64,
    /// `max_io_size` in bytes.
    pub max_io_size: u64,
    /// `rotation_rate`.
    pub rotation_rate: u16,
    /// `scsi_version`.
    pub scsi_version: i32,
    /// `logical_block_size`, 0 for the 512 byte default.
    pub logical_block_size: u32,
    /// `physical_block_size`, 0 for the 512 byte default.
    pub physical_block_size: u32,
    /// `min_io_size`.
    pub min_io_size: u32,
    /// `opt_io_size`.
    pub opt_io_size: u32,
    /// `discard_granularity`, `None` for the automatic default.
    pub discard_granularity: Option<u32>,
    /// `write-cache`: whether the emulated write cache starts enabled.
    pub write_cache: bool,
    /// `cyls`, `heads` and `secs`, all 0 to guess from the size.
    pub chs: (u32, u32, u32),
}

impl Default for ScsiDiskConf {
    fn default() -> Self {
        ScsiDiskConf {
            kind: ScsiDiskKind::Hd,
            drive: None,
            drive_name: None,
            id: None,
            channel: 0,
            scsi_id: None,
            lun: None,
            version: None,
            serial: None,
            vendor: None,
            product: None,
            device_id: None,
            removable: false,
            dpofua: true,
            wwn: 0,
            port_wwn: 0,
            port_index: 0,
            max_unmap_size: DEFAULT_MAX_UNMAP_SIZE,
            max_io_size: DEFAULT_MAX_IO_SIZE,
            rotation_rate: 0,
            scsi_version: 5,
            logical_block_size: 0,
            physical_block_size: 0,
            min_io_size: 0,
            opt_io_size: 0,
            discard_granularity: None,
            write_cache: true,
            chs: (0, 0, 0),
        }
    }
}

impl ScsiDiskConf {
    /// `-device scsi-hd,drive=...`.
    pub fn hd(drive: Arc<dyn BlockBackend>) -> Self {
        ScsiDiskConf { drive: Some(drive), ..Self::default() }
    }

    /// `-device scsi-cd`, with or without a medium.
    pub fn cd(drive: Option<Arc<dyn BlockBackend>>) -> Self {
        ScsiDiskConf { kind: ScsiDiskKind::Cd, drive, ..Self::default() }
    }

    /// Sets the SCSI address, `scsi-id` and `lun`.
    pub fn at(mut self, scsi_id: u32, lun: u32) -> Self {
        self.scsi_id = Some(scsi_id);
        self.lun = Some(lun);
        self
    }
}

/// What a command left behind: its status, the sense for CHECK CONDITION and the data for the
/// initiator.
#[derive(Debug, Default)]
pub(crate) struct Outcome {
    pub(crate) status: u8,
    pub(crate) sense: Option<ScsiSense>,
    pub(crate) data: Vec<u8>,
}

impl Outcome {
    fn good(data: Vec<u8>) -> Self {
        Outcome { status: GOOD, sense: None, data }
    }

    fn check(sense: ScsiSense) -> Self {
        Outcome { status: CHECK_CONDITION, sense: Some(sense), data: Vec::new() }
    }

    fn io_error(err: &io::Error) -> Self {
        let (status, sense) = ScsiSense::from_io_error(err);
        Outcome { status, sense: Some(sense), data: Vec::new() }
    }
}

/// How an emulated command failed before completing: `Illegal` is QEMU's `illegal_request`
/// label, which reports INVALID FIELD unless a status is already set.
enum Fail {
    Illegal,
    Done(Outcome),
}

impl From<ScsiSense> for Fail {
    fn from(sense: ScsiSense) -> Self {
        Fail::Done(Outcome::check(sense))
    }
}

fn be16(b: &[u8]) -> usize {
    usize::from(u16::from_be_bytes([b[0], b[1]]))
}

fn strpadcpy(dst: &mut [u8], src: &str) {
    let src = src.as_bytes();
    for (i, d) in dst.iter_mut().enumerate() {
        *d = src.get(i).copied().unwrap_or(b' ');
    }
}

fn lba_to_msf(buf: &mut [u8], lba: u32) {
    let lba = lba + 150;
    buf[0] = ((lba / 75) / 60) as u8;
    buf[1] = ((lba / 75) % 60) as u8;
    buf[2] = (lba % 75) as u8;
}

/// `cdrom_read_toc()` from `hw/block/cdrom.c`.
fn cdrom_read_toc(nb_sectors: u32, buf: &mut [u8], msf: bool, start_track: u8) -> Option<usize> {
    if start_track > 1 && start_track != 0xaa {
        return None;
    }
    let mut q = 2;
    let mut put = |q: &mut usize, bytes: &[u8]| {
        buf[*q..*q + bytes.len()].copy_from_slice(bytes);
        *q += bytes.len();
    };
    put(&mut q, &[1, 1]);
    let mut track = |q: &mut usize, control: u8, number: u8, lba: u32| {
        put(q, &[0, control, number, 0]);
        if msf {
            let mut m = [0u8; 4];
            lba_to_msf(&mut m[1..], lba);
            put(q, &m);
        } else {
            put(q, &lba.to_be_bytes());
        }
    };
    if start_track <= 1 {
        track(&mut q, 0x14, 1, 0);
    }
    // The lead out track.
    track(&mut q, 0x16, 0xaa, nb_sectors);
    buf[..2].copy_from_slice(&((q - 2) as u16).to_be_bytes());
    Some(q)
}

/// `cdrom_read_toc_raw()` from `hw/block/cdrom.c`.
fn cdrom_read_toc_raw(nb_sectors: u32, buf: &mut [u8], msf: bool) -> usize {
    let mut out = vec![1u8, 1];
    out.extend_from_slice(&[1, 0x14, 0, 0xa0, 0, 0, 0, 0, 1, 0, 0]);
    out.extend_from_slice(&[1, 0x14, 0, 0xa1, 0, 0, 0, 0, 1, 0, 0]);
    out.extend_from_slice(&[1, 0x14, 0, 0xa2, 0, 0, 0]);
    let addr = |lba: u32| -> [u8; 4] {
        if msf {
            let mut m = [0u8; 4];
            lba_to_msf(&mut m[1..], lba);
            m
        } else {
            lba.to_be_bytes()
        }
    };
    out.extend_from_slice(&addr(nb_sectors));
    out.extend_from_slice(&[1, 0x14, 0, 1, 0, 0, 0]);
    out.extend_from_slice(&addr(0));
    let len = out.len() + 2;
    buf[..2].copy_from_slice(&((len - 2) as u16).to_be_bytes());
    buf[2..len].copy_from_slice(&out);
    len
}

/// A SCSI hard disk or CD drive, `SCSIDiskState` together with the generic `SCSIDevice` state.
#[derive(Debug)]
pub struct ScsiDisk {
    pub(crate) channel: u32,
    pub(crate) id: u32,
    pub(crate) lun: u32,
    pub(crate) qdev_id: Option<String>,
    /// The pending unit attention condition, NO SENSE when there is none.
    pub(crate) unit_attention: ScsiSense,
    /// The sense of the last command, what REQUEST SENSE returns.
    pub(crate) sense: Option<ScsiSense>,
    pub(crate) sense_is_ua: bool,

    kind: ScsiDiskKind,
    blk: Option<Arc<dyn BlockBackend>>,
    pub(crate) dev_type: u8,
    pub(crate) blocksize: u32,
    max_lba: u64,
    removable: bool,
    dpofua: bool,
    write_cache: bool,
    tray_open: bool,
    tray_locked: bool,
    media_changed: bool,
    media_event: bool,
    eject_request: bool,
    scsi_version: i32,
    default_scsi_version: i32,

    vendor: String,
    product: String,
    version: String,
    serial: Option<String>,
    device_id: Option<String>,
    wwn: u64,
    port_wwn: u64,
    port_index: u16,
    max_unmap_size: u64,
    max_io_size: u64,
    rotation_rate: u16,
    logical_block_size: u32,
    physical_block_size: u32,
    min_io_size: u32,
    opt_io_size: u32,
    discard_granularity: u32,
    cyls: u32,
    heads: u32,
    secs: u32,
}

impl ScsiDisk {
    /// `scsi_hd_realize()` or `scsi_cd_realize()`, then the reset every new device gets. The
    /// SCSI address is checked and completed when the disk is attached to a bus.
    pub fn new(conf: ScsiDiskConf) -> Result<ScsiDisk> {
        let is_cd = conf.kind == ScsiDiskKind::Cd;
        if !is_cd && conf.drive.is_none() {
            return Err(Error::generic("drive property not set"));
        }
        let removable = is_cd || conf.removable;
        let has_media = conf.drive.as_ref().is_some_and(|b| !b.is_empty() || is_cd);
        if !removable && !has_media {
            return Err(Error::generic("Device needs media, but drive is empty"));
        }

        // blkconf_blocksizes().
        let physical = if conf.physical_block_size == 0 { 512 } else { conf.physical_block_size };
        let logical = if conf.logical_block_size == 0 { 512 } else { conf.logical_block_size };
        if logical > physical {
            return Err(Error::generic("logical_block_size > physical_block_size not supported"));
        }
        if conf.min_io_size % logical != 0 {
            return Err(Error::generic("min_io_size must be a multiple of logical_block_size"));
        }
        if conf.min_io_size / logical > u32::from(u16::MAX) {
            return Err(Error::generic(format!(
                "min_io_size must not exceed {} logical blocks",
                u16::MAX
            )));
        }
        if conf.opt_io_size % logical != 0 {
            return Err(Error::generic("opt_io_size must be a multiple of logical_block_size"));
        }
        if conf.discard_granularity.is_some_and(|g| g % logical != 0) {
            return Err(Error::generic(
                "discard_granularity must be a multiple of logical_block_size",
            ));
        }

        let (dev_type, blocksize, product) = if is_cd {
            let bs = if conf.physical_block_size != 0 { conf.physical_block_size } else { 2048 };
            (TYPE_ROM, bs, "QEMU CD-ROM")
        } else {
            (TYPE_DISK, logical, "QEMU HARDDISK")
        };

        if let Some(serial) = &conf.serial {
            if serial.len() > MAX_SERIAL_LEN {
                return Err(Error::generic(format!(
                    "The serial number can't be longer than {MAX_SERIAL_LEN} characters"
                )));
            }
        }
        let mut device_id = conf.device_id.clone();
        if device_id.is_none() {
            if let Some(serial) = &conf.serial {
                if serial.len() > MAX_SERIAL_LEN_FOR_DEVID {
                    return Err(Error::generic(format!(
                        "The serial number can't be longer than {MAX_SERIAL_LEN_FOR_DEVID} \
                         characters when it is also used as the default for device_id"
                    )));
                }
                device_id = Some(serial.clone());
            } else {
                device_id = conf.drive_name.clone().filter(|n| !n.is_empty());
            }
        }

        // blkconf_geometry() with the size based guess of hd_geometry_guess().
        let nb512 = conf.drive.as_ref().map_or(0, |b| b.len() / BDRV_SECTOR_SIZE);
        let (cyls, heads, secs) = if dev_type == TYPE_DISK && conf.chs == (0, 0, 0) {
            ((nb512 / (16 * 63)).clamp(2, 16383) as u32, 16, 63)
        } else {
            conf.chs
        };

        let mut disk = ScsiDisk {
            channel: conf.channel,
            id: conf.scsi_id.unwrap_or(0),
            lun: conf.lun.unwrap_or(0),
            qdev_id: conf.id,
            unit_attention: ScsiSense::NO_SENSE,
            sense: None,
            sense_is_ua: false,
            kind: conf.kind,
            blk: conf.drive,
            dev_type,
            blocksize,
            max_lba: 0,
            removable,
            dpofua: !is_cd && conf.dpofua,
            write_cache: conf.write_cache,
            tray_open: false,
            tray_locked: false,
            media_changed: false,
            media_event: false,
            eject_request: false,
            scsi_version: conf.scsi_version,
            default_scsi_version: conf.scsi_version,
            vendor: conf.vendor.unwrap_or_else(|| "QEMU".to_string()),
            product: conf.product.unwrap_or_else(|| product.to_string()),
            version: conf.version.unwrap_or_else(|| QEMU_HW_VERSION.to_string()),
            serial: conf.serial,
            device_id,
            wwn: conf.wwn,
            port_wwn: conf.port_wwn,
            port_index: conf.port_index,
            max_unmap_size: conf.max_unmap_size,
            max_io_size: conf.max_io_size,
            rotation_rate: conf.rotation_rate,
            logical_block_size: logical,
            physical_block_size: physical,
            min_io_size: conf.min_io_size,
            opt_io_size: conf.opt_io_size,
            discard_granularity: conf
                .discard_granularity
                .unwrap_or_else(|| logical.max(DEFAULT_DISCARD_GRANULARITY)),
            cyls,
            heads,
            secs,
        };
        disk.reset();
        Ok(disk)
    }

    /// `scsi-hd` or `scsi-cd`.
    pub fn kind(&self) -> ScsiDiskKind {
        self.kind
    }

    /// The SCSI address as (channel, target, LUN).
    pub fn address(&self) -> (u32, u32, u32) {
        (self.channel, self.id, self.lun)
    }

    /// The qdev id.
    pub fn qdev_id(&self) -> Option<&str> {
        self.qdev_id.as_deref()
    }

    /// The peripheral device type, `TYPE_DISK` or `TYPE_ROM`.
    pub fn device_type(&self) -> u8 {
        self.dev_type
    }

    /// The block size the guest sees.
    pub fn blocksize(&self) -> u32 {
        self.blocksize
    }

    /// The pending unit attention condition, [`ScsiSense::NO_SENSE`] if there is none.
    pub fn unit_attention(&self) -> ScsiSense {
        self.unit_attention
    }

    /// Whether the emulated write cache is on (the WCE bit of the caching mode page).
    pub fn write_cache_enabled(&self) -> bool {
        self.write_cache
    }

    /// Whether the tray is open, for removable devices.
    pub fn tray_open(&self) -> bool {
        self.tray_open
    }

    /// Whether the guest locked the tray with PREVENT ALLOW MEDIUM REMOVAL.
    pub fn tray_locked(&self) -> bool {
        self.tray_locked
    }

    /// `scsi_disk_reset()`: requests are gone (there are none outstanding here), a RESET unit
    /// attention is raised and the tray is closed and unlocked.
    pub fn reset(&mut self) {
        self.set_ua(ScsiSense::RESET);
        let nb = self.nb_sectors();
        self.max_lba = nb.saturating_sub(1);
        self.tray_locked = false;
        self.tray_open = false;
        self.scsi_version = self.default_scsi_version;
    }

    /// `scsi_device_set_ua()`: raises a unit attention unless a more important one is pending.
    pub fn set_ua(&mut self, sense: ScsiSense) {
        if !sense.is_unit_attention() {
            return;
        }
        if sense.ua_precedence() < self.unit_attention.ua_precedence() {
            self.unit_attention = sense;
        }
    }

    /// `scsi_cd_change_media_cb()`: a medium was inserted into (`Some`) or taken out of (`None`)
    /// a removable drive.
    pub fn change_media(&mut self, drive: Option<Arc<dyn BlockBackend>>) {
        let load = drive.is_some();
        self.blk = drive;
        self.media_changed = load;
        self.tray_open = !load;
        self.set_ua(ScsiSense::UNIT_ATTENTION_NO_MEDIUM);
        self.media_event = true;
        self.eject_request = false;
    }

    /// `scsi_cd_eject_request_cb()`: the host asks the guest to eject the medium.
    pub fn eject_request(&mut self, force: bool) {
        self.eject_request = true;
        if force {
            self.tray_locked = false;
        }
    }

    /// `scsi_disk_unit_attention_reported()`: once the "no medium" unit attention of a media
    /// change was seen, the guest is told the medium changed.
    pub(crate) fn unit_attention_reported(&mut self) {
        if self.media_changed {
            self.media_changed = false;
            self.set_ua(ScsiSense::MEDIUM_CHANGED);
        }
    }

    /// `blk_get_geometry()`: the size in 512 byte sectors, 0 without a medium.
    fn nb_sectors512(&self) -> u64 {
        self.blk.as_ref().map_or(0, |b| b.len() / BDRV_SECTOR_SIZE)
    }

    /// The size in guest blocks.
    fn nb_sectors(&self) -> u64 {
        self.nb_sectors512() / (u64::from(self.blocksize) / BDRV_SECTOR_SIZE)
    }

    /// `blk_is_inserted()`.
    fn is_inserted(&self) -> bool {
        self.blk.is_some()
    }

    /// `blk_is_available()`: inserted and the tray closed.
    fn is_available(&self) -> bool {
        self.is_inserted() && !self.tray_open
    }

    /// `blk_is_writable()`.
    fn is_writable(&self) -> bool {
        self.dev_type != TYPE_ROM && self.blk.as_ref().is_some_and(|b| !b.is_read_only())
    }

    fn backend(&self) -> io::Result<&Arc<dyn BlockBackend>> {
        self.blk.as_ref().ok_or_else(|| io::Error::other("no medium"))
    }

    /// `check_lba_range()`, with `lba` and `n` in guest blocks.
    fn check_lba_range(&self, lba: u64, n: u64) -> bool {
        lba.checked_add(n).is_some_and(|end| end <= self.max_lba.saturating_add(1))
    }

    /// Runs a command addressed to this device's own LUN, `scsi_new_request()` with the
    /// dispatch table. `data_out` holds the `cmd.xfer` bytes of a command that sends data.
    pub(crate) fn execute(&mut self, cmd: &ScsiCommand, data_out: &[u8]) -> Outcome {
        match cmd.opcode() {
            READ_6 | READ_10 | READ_12 | READ_16 | WRITE_6 | WRITE_10 | WRITE_12 | WRITE_16
            | WRITE_VERIFY_10 | WRITE_VERIFY_12 | WRITE_VERIFY_16 => {
                self.dma_command(cmd, data_out)
            }
            _ => self.emulate_command(cmd, data_out),
        }
    }

    /// `scsi_disk_emulate_command()` plus the data phase and `scsi_disk_emulate_write_data()`.
    fn emulate_command(&mut self, cmd: &ScsiCommand, data_out: &[u8]) -> Outcome {
        let op = cmd.opcode();
        let media_optional = matches!(
            op,
            INQUIRY
                | MODE_SENSE
                | MODE_SENSE_10
                | RESERVE
                | RESERVE_10
                | RELEASE
                | RELEASE_10
                | START_STOP
                | ALLOW_MEDIUM_REMOVAL
                | GET_CONFIGURATION
                | GET_EVENT_STATUS_NOTIFICATION
                | MECHANISM_STATUS
                | REQUEST_SENSE
        );
        if !media_optional && !self.is_available() {
            return Outcome::check(ScsiSense::NO_MEDIUM);
        }

        // QEMU refuses allocation lengths above 64 KiB rather than return more than it built.
        if cmd.xfer > 65536 {
            return Outcome::check(ScsiSense::INVALID_FIELD);
        }
        let buflen = (cmd.xfer as usize).max(4096);
        let mut outbuf = vec![0u8; buflen];

        match self.emulate_op(cmd, &mut outbuf) {
            Ok(()) => {}
            Err(Fail::Illegal) => return Outcome::check(ScsiSense::INVALID_FIELD),
            Err(Fail::Done(outcome)) => return outcome,
        }

        let len = buflen.min(cmd.xfer as usize);
        if len == 0 {
            return Outcome::good(Vec::new());
        }
        if cmd.mode == XferMode::ToDev {
            let mut inbuf = data_out[..data_out.len().min(len)].to_vec();
            inbuf.resize(len, 0);
            return self.emulate_write_data(cmd, &inbuf);
        }
        outbuf.truncate(len);
        Outcome::good(outbuf)
    }

    /// The opcode switch of `scsi_disk_emulate_command()`.
    fn emulate_op(&mut self, cmd: &ScsiCommand, outbuf: &mut [u8]) -> Result<(), Fail> {
        let buf = &cmd.buf;
        match cmd.opcode() {
            TEST_UNIT_READY => {}
            INQUIRY => {
                self.emulate_inquiry(cmd, outbuf).ok_or(Fail::Illegal)?;
            }
            MODE_SENSE | MODE_SENSE_10 => self.emulate_mode_sense(cmd, outbuf)?,
            READ_TOC => {
                self.emulate_read_toc(cmd, outbuf).ok_or(Fail::Illegal)?;
            }
            RESERVE | RELEASE => {
                if buf[1] & 1 != 0 {
                    return Err(Fail::Illegal);
                }
            }
            RESERVE_10 | RELEASE_10 => {
                if buf[1] & 3 != 0 {
                    return Err(Fail::Illegal);
                }
            }
            START_STOP => self.emulate_start_stop(cmd)?,
            ALLOW_MEDIUM_REMOVAL => self.tray_locked = buf[4] & 1 != 0,
            READ_CAPACITY_10 => {
                let nb = self.nb_sectors();
                if nb == 0 {
                    return Err(ScsiSense::LUN_NOT_READY.into());
                }
                if buf[8] & 1 == 0 && cmd.lba != 0 {
                    return Err(Fail::Illegal);
                }
                // The address of the last block, clipped to 2 TB rather than wrapped.
                self.max_lba = nb - 1;
                let last = self.max_lba.min(u64::from(u32::MAX)) as u32;
                outbuf[..4].copy_from_slice(&last.to_be_bytes());
                outbuf[4..8].fill(0);
                outbuf[6] = (self.blocksize >> 8) as u8;
            }
            REQUEST_SENSE => {
                // Just return NO SENSE.
                let sense = ScsiSense::NO_SENSE.to_buf(outbuf.len(), buf[1] & 1 == 0);
                outbuf[..sense.len()].copy_from_slice(&sense);
            }
            MECHANISM_STATUS => {
                if self.dev_type != TYPE_ROM {
                    return Err(Fail::Illegal);
                }
                outbuf[..8].fill(0);
                outbuf[5] = 1;
            }
            GET_CONFIGURATION => self.get_configuration(outbuf)?,
            GET_EVENT_STATUS_NOTIFICATION => self.get_event_status_notification(cmd, outbuf)?,
            SERVICE_ACTION_IN_16 => {
                if buf[1] & 31 != SAI_READ_CAPACITY_16 {
                    return Err(Fail::Illegal);
                }
                let nb = self.nb_sectors();
                if nb == 0 {
                    return Err(ScsiSense::LUN_NOT_READY.into());
                }
                if buf[14] & 1 == 0 && cmd.lba != 0 {
                    return Err(Fail::Illegal);
                }
                self.max_lba = nb - 1;
                outbuf[..8].copy_from_slice(&self.max_lba.to_be_bytes());
                outbuf[8..16].fill(0);
                outbuf[10] = (self.blocksize >> 8) as u8;
                outbuf[13] = self.physical_block_exp();
                // The TPE bit, when the format supports discard.
                if self.discard_granularity != 0 {
                    outbuf[14] = 0x80;
                }
            }
            SYNCHRONIZE_CACHE => {
                let res = self.backend().and_then(|b| b.flush());
                return Err(Fail::Done(match res {
                    Ok(()) => Outcome::good(Vec::new()),
                    Err(e) => Outcome::io_error(&e),
                }));
            }
            SEEK_10 => {
                if cmd.lba > self.max_lba {
                    return Err(ScsiSense::LBA_OUT_OF_RANGE.into());
                }
            }
            MODE_SELECT | MODE_SELECT_10 | UNMAP | WRITE_SAME_10 | WRITE_SAME_16 | FORMAT_UNIT => {}
            VERIFY_10 | VERIFY_12 | VERIFY_16 => {
                if buf[1] & 6 != 0 {
                    return Err(Fail::Illegal);
                }
            }
            _ => return Err(ScsiSense::INVALID_OPCODE.into()),
        }
        Ok(())
    }

    /// `get_physical_block_exp()`.
    fn physical_block_exp(&self) -> u8 {
        let mut exp = 0;
        let mut pbs = self.physical_block_size;
        while pbs > self.logical_block_size {
            pbs >>= 1;
            exp += 1;
        }
        exp
    }

    /// `scsi_disk_emulate_write_data()` once the data has arrived.
    fn emulate_write_data(&mut self, cmd: &ScsiCommand, inbuf: &[u8]) -> Outcome {
        match cmd.opcode() {
            MODE_SELECT | MODE_SELECT_10 => self.emulate_mode_select(cmd, inbuf),
            UNMAP => self.emulate_unmap(cmd, inbuf),
            WRITE_SAME_10 | WRITE_SAME_16 => self.emulate_write_same(cmd, inbuf),
            FORMAT_UNIT => Outcome::good(Vec::new()),
            // VERIFY with BYTCHK, which scsi-disk does not implement.
            _ => Outcome::check(ScsiSense::INVALID_FIELD),
        }
    }

    /// `scsi_disk_emulate_inquiry()`. Returns the length of the data built.
    fn emulate_inquiry(&self, cmd: &ScsiCommand, outbuf: &mut [u8]) -> Option<usize> {
        if cmd.buf[1] & 1 != 0 {
            return self.emulate_vpd_page(cmd, outbuf);
        }
        // Standard INQUIRY data.
        if cmd.buf[2] != 0 {
            return None;
        }
        let buflen = (cmd.xfer as usize).min(SCSI_MAX_INQUIRY_LEN);
        outbuf[0] = self.dev_type & 0x1f;
        outbuf[1] = if self.removable { 0x80 } else { 0 };
        strpadcpy(&mut outbuf[16..32], &self.product);
        strpadcpy(&mut outbuf[8..16], &self.vendor);
        outbuf[32..36].fill(0);
        let ver = self.version.as_bytes();
        let n = ver.len().min(4);
        outbuf[32..32 + n].copy_from_slice(&ver[..n]);
        // SPC-3 conformance, so guests ask for READ CAPACITY(16) and the VPD pages.
        outbuf[2] = self.default_scsi_version as u8;
        // Response data format 2, HiSup.
        outbuf[3] = 2 | 0x10;
        // A too small allocation length does not shrink the additional length.
        outbuf[4] = if buflen > 36 { (buflen - 5) as u8 } else { 36 - 5 };
        // Sync data transfer and command queueing.
        outbuf[7] = 0x10 | 0x02;
        Some(buflen)
    }

    /// `scsi_disk_emulate_vpd_page()`.
    fn emulate_vpd_page(&self, cmd: &ScsiCommand, outbuf: &mut [u8]) -> Option<usize> {
        let page_code = cmd.buf[2];
        outbuf[0] = self.dev_type & 0x1f;
        outbuf[1] = page_code;
        outbuf[2] = 0;
        outbuf[3] = 0;
        let start = 4;
        let mut buflen = start;
        let mut push = |outbuf: &mut [u8], bytes: &[u8]| {
            outbuf[buflen..buflen + bytes.len()].copy_from_slice(bytes);
            buflen += bytes.len();
        };
        match page_code {
            // Supported pages.
            0x00 => {
                push(outbuf, &[0x00]);
                if self.serial.is_some() {
                    push(outbuf, &[0x80]);
                }
                push(outbuf, &[0x83]);
                if self.dev_type == TYPE_DISK {
                    push(outbuf, &[0xb0, 0xb1, 0xb2]);
                }
            }
            // Unit serial number.
            0x80 => {
                let serial = self.serial.as_ref()?.as_bytes();
                push(outbuf, &serial[..serial.len().min(MAX_SERIAL_LEN)]);
            }
            // Device identification.
            0x83 => {
                if let Some(id) = &self.device_id {
                    let id = &id.as_bytes()[..id.len().min(255 - 8)];
                    if !id.is_empty() {
                        push(outbuf, &[0x2, 0, 0, id.len() as u8]);
                        push(outbuf, id);
                    }
                }
                if self.wwn != 0 {
                    push(outbuf, &[0x1, 0x3, 0, 8]);
                    push(outbuf, &self.wwn.to_be_bytes());
                }
                if self.port_wwn != 0 {
                    push(outbuf, &[0x61, 0x93, 0, 8]);
                    push(outbuf, &self.port_wwn.to_be_bytes());
                }
                if self.port_index != 0 {
                    push(outbuf, &[0x61, 0x94, 0, 4, 0, 0]);
                    push(outbuf, &self.port_index.to_be_bytes());
                }
            }
            // Block limits.
            0xb0 => {
                if self.dev_type == TYPE_ROM {
                    return None;
                }
                let bs = u64::from(self.blocksize);
                let min_io = u64::from(self.min_io_size) / bs;
                let opt_io = u64::from(self.opt_io_size) / bs;
                let max_io = self.max_io_size / bs;
                let mut bl = [0u8; 0x3c];
                bl[0] = 1; // wsnz
                if max_io != 0 {
                    // The granularity and optimal length cannot exceed the maximum.
                    bl[2..4].copy_from_slice(&(min_io.min(max_io) as u16).to_be_bytes());
                    bl[4..8].copy_from_slice(&(max_io as u32).to_be_bytes());
                    bl[8..12].copy_from_slice(&(opt_io.min(max_io) as u32).to_be_bytes());
                } else {
                    bl[2..4].copy_from_slice(&(min_io as u16).to_be_bytes());
                    bl[8..12].copy_from_slice(&(opt_io as u32).to_be_bytes());
                }
                let max_unmap = (self.max_unmap_size / bs) as u32;
                bl[16..20].copy_from_slice(&max_unmap.to_be_bytes());
                // 255 descriptors fit in 4 KiB with an 8 byte header.
                bl[20..24].copy_from_slice(&255u32.to_be_bytes());
                let unmap_sectors = (u64::from(self.discard_granularity) / bs) as u32;
                bl[24..28].copy_from_slice(&unmap_sectors.to_be_bytes());
                // The maximum WRITE SAME length is the maximum transfer length.
                bl[36..40].copy_from_slice(&(max_io as u32).to_be_bytes());
                push(outbuf, &bl);
            }
            // Block device characteristics.
            0xb1 => {
                buflen = 0x40;
                outbuf[4..6].copy_from_slice(&self.rotation_rate.to_be_bytes());
                outbuf[6..9].fill(0);
            }
            // Logical block provisioning.
            0xb2 => {
                buflen = 8;
                outbuf[4] = 0;
                // UNMAP and WRITE SAME(10) and (16) are all supported.
                outbuf[5] = 0xe0;
                outbuf[6] = if self.discard_granularity != 0 { 2 } else { 1 };
                outbuf[7] = 0;
            }
            _ => return None,
        }
        outbuf[start - 1] = (buflen - start) as u8;
        Some(buflen)
    }

    /// `mode_sense_page()`: page `page` with its two byte header, `None` if the device type does
    /// not have it. `page_control` 1 asks for the mask of changeable bits.
    fn mode_sense_page(&self, page: usize, page_control: u8) -> Option<Vec<u8>> {
        let disk = self.dev_type == TYPE_DISK;
        let rom = self.dev_type == TYPE_ROM;
        let length = match page {
            MODE_PAGE_HD_GEOMETRY if disk => 0x16,
            MODE_PAGE_FLEXIBLE_DISK_GEOMETRY if disk => 0x1e,
            MODE_PAGE_CACHING if disk || rom => 0x12,
            MODE_PAGE_R_W_ERROR if disk || rom => 10,
            MODE_PAGE_AUDIO_CTL if rom => 14,
            MODE_PAGE_CAPABILITIES if rom => 0x14,
            _ => return None,
        };
        let mut out = vec![0u8; length + 2];
        out[0] = page as u8;
        out[1] = length as u8;
        // The offsets below leave out the two byte header, which keeps them the same as in
        // MODE SELECT.
        let p = &mut out[2..];
        let changeable = page_control == 1;
        let (cyls, heads, secs) = (self.cyls, self.heads, self.secs);
        match page {
            MODE_PAGE_HD_GEOMETRY if !changeable => {
                let c = [(cyls >> 16) as u8, (cyls >> 8) as u8, cyls as u8];
                p[0..3].copy_from_slice(&c);
                p[3] = heads as u8;
                // Write precompensation and reduced current start cylinders, disabled.
                p[4..7].copy_from_slice(&c);
                p[7..10].copy_from_slice(&c);
                // Device step rate, 200 ns.
                p[11] = 200;
                // Landing zone cylinder.
                p[12..15].fill(0xff);
                // Medium rotation rate, 5400 rpm.
                p[18..20].copy_from_slice(&5400u16.to_be_bytes());
            }
            MODE_PAGE_FLEXIBLE_DISK_GEOMETRY if !changeable => {
                // Transfer rate, 5 Mbit/s.
                p[0..2].copy_from_slice(&5000u16.to_be_bytes());
                p[2] = heads as u8;
                p[3] = secs as u8;
                p[4] = (self.blocksize >> 8) as u8;
                let c = [(cyls >> 8) as u8, cyls as u8];
                p[6..8].copy_from_slice(&c);
                p[8..10].copy_from_slice(&c);
                p[10..12].copy_from_slice(&c);
                // Step rate, step pulse width, head settle delay, motor on and off delays.
                p[13] = 1;
                p[14] = 1;
                p[16] = 1;
                p[17] = 1;
                p[18] = 1;
                p[26..28].copy_from_slice(&5400u16.to_be_bytes());
            }
            MODE_PAGE_CACHING => {
                if changeable || self.write_cache {
                    p[0] = 4; // WCE
                }
            }
            MODE_PAGE_R_W_ERROR => {
                if changeable {
                    if rom {
                        p[0] = 0x80;
                    }
                } else {
                    // Automatic write reallocation.
                    p[0] = 0x80;
                    if rom {
                        // Read retry count.
                        p[1] = 0x20;
                    }
                }
            }
            MODE_PAGE_CAPABILITIES if !changeable => {
                p[0] = 0x3b; // CD-R and CD-RW read
                p[1] = 0; // no writing
                p[2] = 0x7f; // audio, composite, digital out, mode 2 form 1 and 2, multi session
                p[3] = 0xff; // CD DA, accurate, RW supported and corrected, C2, ISRC, UPC, barcode
                p[4] = 0x2d | if self.tray_locked { 2 } else { 0 };
                p[5] = 0;
                p[6..8].copy_from_slice(&(50u16 * 176).to_be_bytes());
                p[8..10].copy_from_slice(&2u16.to_be_bytes());
                p[10..12].copy_from_slice(&2048u16.to_be_bytes());
                p[12..14].copy_from_slice(&(16u16 * 176).to_be_bytes());
                p[16..18].copy_from_slice(&(16u16 * 176).to_be_bytes());
                p[18..20].copy_from_slice(&(16u16 * 176).to_be_bytes());
            }
            _ => {}
        }
        Some(out)
    }

    /// `scsi_disk_emulate_mode_sense()`.
    fn emulate_mode_sense(&self, cmd: &ScsiCommand, outbuf: &mut [u8]) -> Result<(), Fail> {
        let buf = &cmd.buf;
        let six = buf[0] == MODE_SENSE;
        let mut dbd = buf[1] & 0x8 != 0;
        let page = usize::from(buf[2] & 0x3f);
        let page_control = (buf[2] & 0xc0) >> 6;

        let dev_specific_param = if self.dev_type == TYPE_DISK {
            let mut v = if self.dpofua { 0x10 } else { 0 };
            if !self.is_writable() {
                v |= 0x80;
            }
            v
        } else {
            // MMC drives have no block descriptors and no device specific parameter.
            dbd = true;
            0
        };

        let mut out = Vec::with_capacity(256);
        if six {
            out.extend_from_slice(&[0, 0, dev_specific_param, 0]);
        } else {
            out.extend_from_slice(&[0, 0, 0, dev_specific_param, 0, 0, 0, 0]);
        }

        let nb512 = self.nb_sectors512();
        if !dbd && nb512 != 0 {
            if six {
                out[3] = 8;
            } else {
                out[7] = 8;
            }
            let mut nb = nb512 / (u64::from(self.blocksize) / BDRV_SECTOR_SIZE);
            if nb > 0xff_ffff {
                nb = 0;
            }
            out.extend_from_slice(&[
                0,
                (nb >> 16) as u8,
                (nb >> 8) as u8,
                nb as u8,
                0,
                0,
                (self.blocksize >> 8) as u8,
                0,
            ]);
        }

        if page_control == 3 {
            return Err(ScsiSense::SAVING_PARAMS_NOT_SUPPORTED.into());
        }
        if page == MODE_PAGE_ALLS {
            for page in 0..MODE_PAGE_ALLS {
                if let Some(p) = self.mode_sense_page(page, page_control) {
                    out.extend_from_slice(&p);
                }
            }
        } else {
            let p = self.mode_sense_page(page, page_control).ok_or(Fail::Illegal)?;
            out.extend_from_slice(&p);
        }

        // The mode data length does not count itself.
        let buflen = out.len();
        if six {
            out[0] = (buflen - 1) as u8;
        } else {
            out[0..2].copy_from_slice(&((buflen - 2) as u16).to_be_bytes());
        }
        let n = buflen.min(outbuf.len());
        outbuf[..n].copy_from_slice(&out[..n]);
        Ok(())
    }

    /// `scsi_disk_emulate_read_toc()`.
    fn emulate_read_toc(&self, cmd: &ScsiCommand, outbuf: &mut [u8]) -> Option<usize> {
        let msf = cmd.buf[1] & 2 != 0;
        let format = cmd.buf[2] & 0xf;
        let start_track = cmd.buf[6];
        let nb = self.nb_sectors() as u32;
        match format {
            0 => cdrom_read_toc(nb, outbuf, msf, start_track),
            // Multi session: only a single session is defined.
            1 => {
                outbuf[..12].fill(0);
                outbuf[1] = 0x0a;
                outbuf[2] = 0x01;
                outbuf[3] = 0x01;
                Some(12)
            }
            2 => Some(cdrom_read_toc_raw(nb, outbuf, msf)),
            _ => None,
        }
    }

    /// `scsi_disk_emulate_start_stop()`.
    fn emulate_start_stop(&mut self, cmd: &ScsiCommand) -> Result<(), Fail> {
        let start = cmd.buf[4] & 1 != 0;
        // Load on start, eject on stop.
        let loej = cmd.buf[4] & 2 != 0;
        // Eject and load only happen for power condition 0.
        if cmd.buf[4] & 0xf0 != 0 {
            return Ok(());
        }
        if self.removable && loej {
            if !start && !self.tray_open && self.tray_locked {
                return Err(if self.is_inserted() {
                    ScsiSense::ILLEGAL_REQ_REMOVAL_PREVENTED
                } else {
                    ScsiSense::NOT_READY_REMOVAL_PREVENTED
                }
                .into());
            }
            self.tray_open = !start;
        }
        Ok(())
    }

    /// `scsi_get_configuration()`.
    fn get_configuration(&self, outbuf: &mut [u8]) -> Result<(), Fail> {
        if self.dev_type != TYPE_ROM {
            return Err(Fail::Illegal);
        }
        let nb = self.nb_sectors512();
        let current = if !self.is_available() {
            MMC_PROFILE_NONE
        } else if nb > CD_MAX_SECTORS {
            MMC_PROFILE_DVD_ROM
        } else {
            MMC_PROFILE_CD_ROM
        };
        outbuf[..40].fill(0);
        // Bytes after the data length field.
        outbuf[0..4].copy_from_slice(&36u32.to_be_bytes());
        outbuf[6..8].copy_from_slice(&current.to_be_bytes());
        // Feature 0, the profile list: persistent, current, two profiles.
        outbuf[10] = 0x03;
        outbuf[11] = 8;
        outbuf[12..14].copy_from_slice(&MMC_PROFILE_DVD_ROM.to_be_bytes());
        outbuf[14] = u8::from(current == MMC_PROFILE_DVD_ROM);
        outbuf[16..18].copy_from_slice(&MMC_PROFILE_CD_ROM.to_be_bytes());
        outbuf[18] = u8::from(current == MMC_PROFILE_CD_ROM);
        // Feature 1, the core feature: version 2, persistent, current, SCSI, DBE.
        outbuf[20..22].copy_from_slice(&1u16.to_be_bytes());
        outbuf[22] = 0x08 | 0x03;
        outbuf[23] = 8;
        outbuf[24..28].copy_from_slice(&1u32.to_be_bytes());
        outbuf[28] = 1;
        // Feature 3, removable medium: tray, load, eject, unlocked at power up, lock.
        outbuf[32..34].copy_from_slice(&3u16.to_be_bytes());
        outbuf[34] = 0x08 | 0x03;
        outbuf[35] = 4;
        outbuf[36] = 0x39;
        Ok(())
    }

    /// `scsi_get_event_status_notification()`.
    fn get_event_status_notification(
        &mut self,
        cmd: &ScsiCommand,
        outbuf: &mut [u8],
    ) -> Result<(), Fail> {
        if self.dev_type != TYPE_ROM || cmd.buf[1] & 1 == 0 {
            // Not a CD drive, or the asynchronous form.
            return Err(Fail::Illegal);
        }
        let mut size = 4;
        outbuf[0] = 0;
        outbuf[1] = 0;
        outbuf[3] = 1 << GESN_MEDIA;
        if cmd.buf[4] & (1 << GESN_MEDIA) != 0 {
            outbuf[2] = GESN_MEDIA;
            // scsi_event_status_media().
            let media_status = if self.tray_open {
                MS_TRAY_OPEN
            } else if self.is_inserted() {
                MS_MEDIA_PRESENT
            } else {
                0
            };
            let mut event_code = MEC_NO_CHANGE;
            if media_status != MS_TRAY_OPEN {
                if self.media_event {
                    event_code = MEC_NEW_MEDIA;
                    self.media_event = false;
                } else if self.eject_request {
                    event_code = MEC_EJECT_REQUESTED;
                    self.eject_request = false;
                }
            }
            outbuf[4..8].copy_from_slice(&[event_code, media_status, 0, 0]);
            size += 4;
        } else {
            outbuf[2] = 0x80;
        }
        outbuf[0..2].copy_from_slice(&((size - 4) as u16).to_be_bytes());
        Ok(())
    }

    /// `scsi_disk_check_mode_select()`: the page may only change bits MODE SENSE reports as
    /// changeable, and may be truncated but not longer.
    fn check_mode_select(&self, page: usize, inbuf: &[u8]) -> bool {
        let expected_len = inbuf.len() + 2;
        if expected_len > 256 || page == MODE_PAGE_ALLS {
            return false;
        }
        let Some(current) = self.mode_sense_page(page, 0) else {
            return false;
        };
        if expected_len > current.len() {
            return false;
        }
        let Some(changeable) = self.mode_sense_page(page, 1) else {
            return false;
        };
        (2..expected_len).all(|i| (current[i] ^ inbuf[i - 2]) & !changeable[i] == 0)
    }

    /// `mode_select_pages()`: checks every page (`change` false) or applies them (true).
    fn mode_select_pages(&mut self, mut p: &[u8], change: bool) -> Result<(), ScsiSense> {
        while !p.is_empty() {
            // Both forms of the page header.
            let page = usize::from(p[0] & 0x3f);
            let (subpage, page_len) = if p[0] & 0x40 != 0 {
                if p.len() < 4 {
                    return Err(ScsiSense::INVALID_PARAM_LEN);
                }
                let r = (p[1], be16(&p[2..]));
                p = &p[4..];
                r
            } else {
                if p.len() < 2 {
                    return Err(ScsiSense::INVALID_PARAM_LEN);
                }
                let r = (0, usize::from(p[1]));
                p = &p[2..];
                r
            };
            if subpage != 0 {
                return Err(ScsiSense::INVALID_PARAM);
            }
            if page_len > p.len() {
                return Err(ScsiSense::INVALID_PARAM_LEN);
            }
            let body = &p[..page_len];
            if !change {
                if !self.check_mode_select(page, body) {
                    return Err(ScsiSense::INVALID_PARAM);
                }
            } else if page == MODE_PAGE_CACHING && !body.is_empty() {
                // scsi_disk_apply_mode_select(): only WCE can change.
                self.write_cache = body[0] & 4 != 0;
            }
            p = &p[page_len..];
        }
        Ok(())
    }

    /// `scsi_disk_emulate_mode_select()`.
    fn emulate_mode_select(&mut self, cmd: &ScsiCommand, inbuf: &[u8]) -> Outcome {
        // Only PF=1, SP=0.
        if cmd.buf[1] & 0x11 != 0x10 {
            return Outcome::check(ScsiSense::INVALID_FIELD);
        }
        let hdr_len = if cmd.opcode() == MODE_SELECT { 4 } else { 8 };
        let mut p = inbuf;
        if p.len() < hdr_len {
            return Outcome::check(ScsiSense::INVALID_PARAM_LEN);
        }
        let bd_len = if cmd.opcode() == MODE_SELECT { usize::from(p[3]) } else { be16(&p[6..]) };
        p = &p[hdr_len..];
        if p.len() < bd_len {
            return Outcome::check(ScsiSense::INVALID_PARAM_LEN);
        }
        if bd_len != 0 && bd_len != 8 {
            return Outcome::check(ScsiSense::INVALID_PARAM);
        }
        // The block size may change, in the bits the block descriptor of MODE SENSE reports.
        if bd_len != 0 {
            let bs = u32::from(p[5]) << 16 | u32::from(p[6]) << 8 | u32::from(p[7]);
            if bs != 0 && bs & !0xfe00 == 0 && bs != self.blocksize {
                self.blocksize = bs;
            }
        }
        p = &p[bd_len..];

        // Nothing changes unless every page is valid.
        if let Err(sense) = self.mode_select_pages(p, false) {
            return Outcome::check(sense);
        }
        let _ = self.mode_select_pages(p, true);
        if !self.write_cache {
            if let Err(e) = self.backend().and_then(|b| b.flush()) {
                return Outcome::io_error(&e);
            }
        }
        Outcome::good(Vec::new())
    }

    /// `scsi_disk_emulate_unmap()` and its completion loop.
    fn emulate_unmap(&mut self, cmd: &ScsiCommand, p: &[u8]) -> Outcome {
        let len = p.len();
        // ANCHOR=1 is not supported.
        if cmd.buf[1] & 1 != 0 {
            return Outcome::check(ScsiSense::INVALID_FIELD);
        }
        if len < 8 || len < be16(&p[0..]) + 2 || len < be16(&p[2..]) + 8 || be16(&p[2..]) & 15 != 0
        {
            return Outcome::check(ScsiSense::INVALID_PARAM_LEN);
        }
        if !self.is_writable() {
            return Outcome::check(ScsiSense::WRITE_PROTECTED);
        }
        let count = be16(&p[2..]) >> 4;
        let bs = u64::from(self.blocksize);
        for d in p[8..].chunks_exact(16).take(count) {
            let lba = u64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
            let n = u64::from(u32::from_be_bytes([d[8], d[9], d[10], d[11]]));
            if !self.check_lba_range(lba, n) {
                return Outcome::check(ScsiSense::LBA_OUT_OF_RANGE);
            }
            if let Err(e) = self.backend().and_then(|b| b.discard(lba * bs, n * bs)) {
                return Outcome::io_error(&e);
            }
        }
        Outcome::good(Vec::new())
    }

    /// `scsi_disk_emulate_write_same()`.
    fn emulate_write_same(&mut self, cmd: &ScsiCommand, inbuf: &[u8]) -> Outcome {
        let nb = u64::from(data_cdb_xfer(&cmd.buf));
        // PBDATA, LBDATA and ANCHOR are not supported.
        if nb == 0 || cmd.buf[1] & 0x16 != 0 {
            return Outcome::check(ScsiSense::INVALID_FIELD);
        }
        if !self.is_writable() {
            return Outcome::check(ScsiSense::WRITE_PROTECTED);
        }
        if !self.check_lba_range(cmd.lba, nb) {
            return Outcome::check(ScsiSense::LBA_OUT_OF_RANGE);
        }
        let bs = u64::from(self.blocksize);
        let Ok(blk) = self.backend() else {
            return Outcome::check(ScsiSense::NO_MEDIUM);
        };
        let offset = cmd.lba * bs;
        let total = nb * bs;
        let block = &inbuf[..inbuf.len().min(bs as usize)];
        let res = if cmd.buf[1] & 1 != 0 || block.iter().all(|&b| b == 0) {
            blk.write_zeroes(offset, total)
        } else {
            // The block repeated, in chunks of at most 512 KiB.
            let chunk_len = total.min(SCSI_WRITE_SAME_MAX) as usize;
            let mut chunk = vec![0u8; chunk_len];
            for piece in chunk.chunks_mut(bs as usize) {
                piece.copy_from_slice(&block[..piece.len()]);
            }
            let mut done = 0;
            let mut res = Ok(());
            while done < total {
                let n = (total - done).min(chunk_len as u64) as usize;
                res = blk.write_at(offset + done, &chunk[..n]);
                if res.is_err() {
                    break;
                }
                done += n as u64;
            }
            res
        };
        match res {
            Ok(()) => Outcome::good(Vec::new()),
            Err(e) => Outcome::io_error(&e),
        }
    }

    /// `scsi_disk_dma_command()` with the reads and writes it starts.
    fn dma_command(&mut self, cmd: &ScsiCommand, data_out: &[u8]) -> Outcome {
        if !self.is_available() {
            return Outcome::check(ScsiSense::NO_MEDIUM);
        }
        let len = u64::from(data_cdb_xfer(&cmd.buf));
        let is_write = cmd.mode == XferMode::ToDev
            || matches!(
                cmd.opcode(),
                WRITE_6
                    | WRITE_10
                    | WRITE_12
                    | WRITE_16
                    | WRITE_VERIFY_10
                    | WRITE_VERIFY_12
                    | WRITE_VERIFY_16
            );
        if is_write && !self.is_writable() {
            return Outcome::check(ScsiSense::WRITE_PROTECTED);
        }
        // Protection information is not supported. SCSI-2 and older have no RDPROTECT field.
        if self.scsi_version > 2 && cmd.buf[1] & 0xe0 != 0 {
            return Outcome::check(ScsiSense::INVALID_FIELD);
        }
        if !self.check_lba_range(cmd.lba, len) {
            return Outcome::check(ScsiSense::LBA_OUT_OF_RANGE);
        }
        if len == 0 {
            return Outcome::good(Vec::new());
        }
        let fua = match cmd.opcode() {
            READ_10 | READ_12 | READ_16 | WRITE_10 | WRITE_12 | WRITE_16 => cmd.buf[1] & 8 != 0,
            WRITE_VERIFY_10 | WRITE_VERIFY_12 | WRITE_VERIFY_16 => true,
            _ => false,
        };
        let bs = u64::from(self.blocksize);
        let Ok(blk) = self.backend() else {
            return Outcome::check(ScsiSense::NO_MEDIUM);
        };
        let offset = cmd.lba * bs;
        let bytes = (len * bs) as usize;
        let res = if is_write {
            let mut data = data_out[..data_out.len().min(bytes)].to_vec();
            data.resize(bytes, 0);
            blk.write_at(offset, &data).and_then(|()| if fua { blk.flush() } else { Ok(()) })
        } else {
            let mut data = vec![0u8; bytes];
            let res = if fua { blk.flush() } else { Ok(()) };
            match res.and_then(|()| blk.read_at(offset, &mut data)) {
                Ok(()) => return Outcome::good(data),
                Err(e) => Err(e),
            }
        };
        match res {
            Ok(()) => Outcome::good(Vec::new()),
            Err(e) => Outcome::io_error(&e),
        }
    }
}
