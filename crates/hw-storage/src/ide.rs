// SPDX-License-Identifier: GPL-2.0-or-later

//! The ATA side of a drive, ported from the parts of QEMU's `hw/ide/core.c` that an AHCI port
//! uses: the task file registers, IDENTIFY, the PIO and DMA read and write commands, FLUSH CACHE,
//! SET FEATURES and the handful of housekeeping commands.
//!
//! QEMU runs these commands as chains of AIO callbacks. Here every command runs to completion
//! before [`IdeDrive::exec_cmd`] returns, so each chain became a loop. The host adapter (the AHCI
//! port) is reached through [`IdeHost`], the equivalent of `IDEDMAOps`.

use std::sync::Arc;

use crate::block::BlockBackend;

pub(crate) const ERR_STAT: u8 = 0x01;
pub(crate) const DRQ_STAT: u8 = 0x08;
pub(crate) const SEEK_STAT: u8 = 0x10;
pub(crate) const WRERR_STAT: u8 = 0x20;
pub(crate) const READY_STAT: u8 = 0x40;
pub(crate) const BUSY_STAT: u8 = 0x80;

pub(crate) const ABRT_ERR: u8 = 0x04;
pub(crate) const MC_ERR: u8 = 0x20;

const ATA_DEV_LBA: u8 = 0x40;
const ATA_DEV_ALWAYS_ON: u8 = 0xa0;
const ATA_DEV_HS: u8 = 0x0f;

const MAX_MULT_SECTORS: u32 = 16;
pub(crate) const SECTOR_SIZE: u64 = 512;
/// `IDE_DMA_BUF_SECTORS`, which sizes the I/O buffer.
pub(crate) const IDE_DMA_BUF_SECTORS: usize = 256;

pub(crate) const WIN_DEVICE_RESET: u8 = 0x08;
const WIN_READ_EXT: u8 = 0x24;
const WIN_READDMA_EXT: u8 = 0x25;
const WIN_READ_NATIVE_MAX_EXT: u8 = 0x27;
const WIN_MULTREAD_EXT: u8 = 0x29;
const WIN_WRITE_EXT: u8 = 0x34;
const WIN_WRITEDMA_EXT: u8 = 0x35;
const WIN_MULTWRITE_EXT: u8 = 0x39;
const WIN_VERIFY_EXT: u8 = 0x42;

/// What kind of device sits on a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveKind {
    /// A hard disk, `ide-hd`.
    Hd,
    /// An ATAPI CD or DVD drive, `ide-cd`.
    Cd,
}

impl DriveKind {
    fn mask(self) -> u8 {
        match self {
            DriveKind::Hd => HD_OK,
            DriveKind::Cd => CD_OK,
        }
    }
}

/// The user visible properties of a drive, the `ide-hd` and `ide-cd` device properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DriveConfig {
    /// A hard disk or a CD-ROM drive.
    pub kind: DriveKind,
    /// The serial number. When unset the controller picks `QM%05d` the way QEMU numbers drives.
    pub serial: Option<String>,
    /// The model string. Defaults to "QEMU HARDDISK" or "QEMU DVD-ROM".
    pub model: Option<String>,
    /// The firmware revision. Defaults to "2.5+", QEMU's `QEMU_HW_VERSION`.
    pub version: Option<String>,
    /// Whether the volatile write cache starts enabled. QEMU's default is on.
    pub write_cache: bool,
    /// The `cyls`, `heads` and `secs` of a hard disk, when they were set. Without them the
    /// geometry is guessed from the size, as `blkconf_geometry()` does when all three are 0.
    pub geometry: Option<(u32, u32, u32)>,
}

impl DriveConfig {
    /// A hard disk with default properties.
    pub fn hd() -> Self {
        DriveConfig {
            kind: DriveKind::Hd,
            serial: None,
            model: None,
            version: None,
            write_cache: true,
            geometry: None,
        }
    }

    /// A CD-ROM drive with default properties.
    pub fn cdrom() -> Self {
        DriveConfig { kind: DriveKind::Cd, ..DriveConfig::hd() }
    }
}

impl Default for DriveConfig {
    fn default() -> Self {
        DriveConfig::hd()
    }
}

/// A scatter-gather list of guest physical ranges, `QEMUSGList`.
#[derive(Clone, Debug, Default)]
pub(crate) struct SgList {
    pub(crate) entries: Vec<(u64, u64)>,
    pub(crate) size: u64,
}

impl SgList {
    pub(crate) fn add(&mut self, addr: u64, len: u64) {
        self.entries.push((addr, len));
        self.size += len;
    }
}

/// What the drive needs from the adapter it sits behind, `IDEDMAOps`.
pub(crate) trait IdeHost {
    /// Moves `s.io_buffer[start..start + len]` between the guest and the drive for a PIO data
    /// phase, `pio_transfer`.
    fn pio_transfer(&mut self, s: &mut IdeDrive, start: usize, len: usize);

    /// Builds a scatter list of at most `limit` bytes that starts `offset` bytes into the guest's
    /// buffer. `None` if the guest described no usable buffer.
    fn sglist(&mut self, limit: u64, offset: u64) -> Option<SgList>;

    /// Copies `data` into the guest ranges of `sg`.
    fn write_guest(&mut self, sg: &SgList, data: &[u8]);

    /// Fills `data` from the guest ranges of `sg`.
    fn read_guest(&mut self, sg: &SgList, data: &mut [u8]);

    /// Adds `tx_bytes` to the adapter's transferred byte count, `commit_buf`.
    fn commit_buf(&mut self, tx_bytes: u32);

    /// The command is finished, `cmd_done`.
    fn cmd_done(&mut self, s: &mut IdeDrive);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DmaCmd {
    Read,
    Write,
    Atapi,
}

/// One drive and its task file, `IDEState`.
#[derive(Debug)]
pub(crate) struct IdeDrive {
    pub(crate) kind: DriveKind,
    /// The image. A CD drive with no disc has none.
    pub(crate) blk: Option<Arc<dyn BlockBackend>>,
    pub(crate) nb_sectors: u64,
    pub(crate) cylinders: u32,
    pub(crate) heads: u32,
    pub(crate) sectors: u32,
    pub(crate) drive_heads: u32,
    pub(crate) drive_sectors: u32,
    pub(crate) serial: String,
    pub(crate) model: String,
    pub(crate) version: String,
    pub(crate) write_cache: bool,
    pub(crate) identify_data: [u8; 512],
    pub(crate) identify_set: bool,
    pub(crate) mult_sectors: u32,
    pub(crate) ncq_queues: u32,
    pub(crate) reset_reverts: bool,

    pub(crate) feature: u8,
    pub(crate) error: u8,
    pub(crate) nsector: u32,
    pub(crate) sector: u8,
    pub(crate) lcyl: u8,
    pub(crate) hcyl: u8,
    pub(crate) hob_feature: u8,
    pub(crate) hob_nsector: u8,
    pub(crate) hob_sector: u8,
    pub(crate) hob_lcyl: u8,
    pub(crate) hob_hcyl: u8,
    pub(crate) select: u8,
    pub(crate) status: u8,
    pub(crate) lba48: bool,

    pub(crate) req_nb_sectors: u32,
    pub(crate) io_buffer: Vec<u8>,
    pub(crate) io_buffer_offset: u64,
    pub(crate) io_buffer_index: usize,
    pub(crate) io_buffer_size: usize,
    pub(crate) dma_cmd: DmaCmd,

    pub(crate) sense_key: u8,
    pub(crate) asc: u8,
    pub(crate) tray_locked: bool,
    pub(crate) atapi_dma: bool,
    pub(crate) lba: i64,
    pub(crate) packet_transfer_size: i64,
    pub(crate) elementary_transfer_size: i64,
    pub(crate) cd_sector_size: usize,
}

const HD_OK: u8 = 1;
const CD_OK: u8 = 2;
const ALL_OK: u8 = HD_OK | CD_OK;

/// The permission and `SET_DSC` columns of `ide_cmd_table`. CompactFlash only commands are
/// left out, so they abort like any other unknown opcode.
fn cmd_info(cmd: u8) -> (u8, bool) {
    match cmd {
        0x06 => (HD_OK, false),
        0x08 => (CD_OK, false),
        0x10 => (HD_OK, true),
        0x20 => (ALL_OK, false),
        0x21 | 0x24 | 0x25 => (HD_OK, false),
        0x27 => (HD_OK, true),
        0x29 | 0x30 | 0x31 | 0x34 | 0x35 | 0x39 | 0x3c => (HD_OK, false),
        0x40..=0x42 | 0x70 | 0x91 => (HD_OK, true),
        0x90 => (ALL_OK, false),
        0x94..=0x97 | 0x99 => (HD_OK, false),
        0x98 => (HD_OK, true),
        0xa0 | 0xa1 => (CD_OK, false),
        0xb0 | 0xc6 => (HD_OK, true),
        0xc4 | 0xc5 | 0xc8..=0xcb => (HD_OK, false),
        0xe0..=0xe3 | 0xe6 => (HD_OK, false),
        0xe5 => (HD_OK, true),
        0xe7 | 0xec => (ALL_OK, false),
        0xea => (HD_OK, false),
        0xef => (ALL_OK, true),
        0xf5 | 0xf8 => (HD_OK, true),
        _ => (0, false),
    }
}

fn put_le16(buf: &mut [u8], word: usize, v: u32) {
    buf[word * 2..word * 2 + 2].copy_from_slice(&(v as u16).to_le_bytes());
}

/// `padstr()`: an ATA string, space padded, with the bytes of each word swapped.
fn padstr(buf: &mut [u8], word: usize, src: &str, len: usize) {
    let src = src.as_bytes();
    let out = &mut buf[word * 2..word * 2 + len];
    for i in 0..len {
        out[i ^ 1] = src.get(i).copied().unwrap_or(b' ');
    }
}

impl IdeDrive {
    /// `ide_init_drive()`. `blk` must be present for a hard disk.
    pub(crate) fn new(
        config: &DriveConfig,
        blk: Option<Arc<dyn BlockBackend>>,
        default_serial: String,
    ) -> Self {
        let nb_sectors = blk.as_ref().map_or(0, |b| b.len() / SECTOR_SIZE);
        // blkconf_geometry(), and guess_chs_for_size() when no geometry was given. The MBR
        // based guess is not ported.
        let (cylinders, heads, sectors) = config
            .geometry
            .unwrap_or_else(|| ((nb_sectors / (16 * 63)).clamp(2, 16383) as u32, 16, 63));
        let model = match config.kind {
            DriveKind::Hd => "QEMU HARDDISK",
            DriveKind::Cd => "QEMU DVD-ROM",
        };
        let mut s = IdeDrive {
            kind: config.kind,
            blk,
            nb_sectors,
            cylinders,
            heads,
            sectors,
            drive_heads: heads,
            drive_sectors: sectors,
            serial: config.serial.clone().unwrap_or(default_serial),
            model: config.model.clone().unwrap_or_else(|| model.to_string()),
            version: config.version.clone().unwrap_or_else(|| "2.5+".to_string()),
            write_cache: config.write_cache,
            identify_data: [0; 512],
            identify_set: false,
            mult_sectors: MAX_MULT_SECTORS,
            ncq_queues: 0,
            reset_reverts: false,
            feature: 0,
            error: 0,
            nsector: 0,
            sector: 0,
            lcyl: 0,
            hcyl: 0,
            hob_feature: 0,
            hob_nsector: 0,
            hob_sector: 0,
            hob_lcyl: 0,
            hob_hcyl: 0,
            select: ATA_DEV_ALWAYS_ON,
            status: READY_STAT | SEEK_STAT,
            lba48: false,
            req_nb_sectors: 0,
            io_buffer: vec![0; IDE_DMA_BUF_SECTORS * SECTOR_SIZE as usize + 4],
            io_buffer_offset: 0,
            io_buffer_index: 0,
            io_buffer_size: 0,
            dma_cmd: DmaCmd::Read,
            sense_key: 0,
            asc: 0,
            tray_locked: false,
            atapi_dma: false,
            lba: -1,
            packet_transfer_size: 0,
            elementary_transfer_size: 0,
            cd_sector_size: 0,
        };
        s.reset();
        s
    }

    /// `ide_reset()`.
    pub(crate) fn reset(&mut self) {
        if self.reset_reverts {
            self.reset_reverts = false;
            self.heads = self.drive_heads;
            self.sectors = self.drive_sectors;
        }
        self.mult_sectors = MAX_MULT_SECTORS;
        self.feature = 0;
        self.error = 0;
        self.nsector = 0;
        self.sector = 0;
        self.lcyl = 0;
        self.hcyl = 0;
        self.hob_feature = 0;
        self.hob_sector = 0;
        self.hob_nsector = 0;
        self.hob_lcyl = 0;
        self.hob_hcyl = 0;
        self.select = ATA_DEV_ALWAYS_ON;
        self.status = READY_STAT | SEEK_STAT;
        self.lba48 = false;
        self.sense_key = 0;
        self.asc = 0;
        self.packet_transfer_size = 0;
        self.elementary_transfer_size = 0;
        self.io_buffer_index = 0;
        self.cd_sector_size = 0;
        self.atapi_dma = false;
        self.tray_locked = false;
        self.io_buffer_size = 0;
        self.req_nb_sectors = 0;
        self.set_signature();
    }

    /// `ide_set_signature()`.
    pub(crate) fn set_signature(&mut self) {
        self.select &= !ATA_DEV_HS;
        self.nsector = 1;
        self.sector = 1;
        if self.kind == DriveKind::Cd {
            self.lcyl = 0x14;
            self.hcyl = 0xeb;
        } else if self.blk.is_some() {
            self.lcyl = 0;
            self.hcyl = 0;
        } else {
            self.lcyl = 0xff;
            self.hcyl = 0xff;
        }
    }

    /// `ide_get_sector()`.
    pub(crate) fn get_sector(&self) -> u64 {
        if self.select & ATA_DEV_LBA != 0 {
            if self.lba48 {
                u64::from(self.hob_hcyl) << 40
                    | u64::from(self.hob_lcyl) << 32
                    | u64::from(self.hob_sector) << 24
                    | u64::from(self.hcyl) << 16
                    | u64::from(self.lcyl) << 8
                    | u64::from(self.sector)
            } else {
                u64::from(self.select & ATA_DEV_HS) << 24
                    | u64::from(self.hcyl) << 16
                    | u64::from(self.lcyl) << 8
                    | u64::from(self.sector)
            }
        } else {
            let cyl = u64::from(self.hcyl) << 8 | u64::from(self.lcyl);
            let heads = u64::from(self.heads);
            let secs = u64::from(self.sectors);
            (cyl * heads * secs + u64::from(self.select & ATA_DEV_HS) * secs)
                .wrapping_add(u64::from(self.sector))
                .wrapping_sub(1)
        }
    }

    /// `ide_set_sector()`.
    pub(crate) fn set_sector(&mut self, n: u64) {
        if self.select & ATA_DEV_LBA != 0 {
            if self.lba48 {
                self.sector = n as u8;
                self.lcyl = (n >> 8) as u8;
                self.hcyl = (n >> 16) as u8;
                self.hob_sector = (n >> 24) as u8;
                self.hob_lcyl = (n >> 32) as u8;
                self.hob_hcyl = (n >> 40) as u8;
            } else {
                self.select = (self.select & !ATA_DEV_HS) | ((n >> 24) as u8 & ATA_DEV_HS);
                self.hcyl = (n >> 16) as u8;
                self.lcyl = (n >> 8) as u8;
                self.sector = n as u8;
            }
        } else {
            let track = u64::from(self.heads) * u64::from(self.sectors);
            // QEMU divides by zero here if the guest set a zero geometry; leave the registers.
            if track == 0 {
                return;
            }
            let cyl = n / track;
            let r = n % track;
            self.hcyl = (cyl >> 8) as u8;
            self.lcyl = cyl as u8;
            self.select =
                (self.select & !ATA_DEV_HS) | ((r / u64::from(self.sectors)) as u8 & ATA_DEV_HS);
            self.sector = (r % u64::from(self.sectors) + 1) as u8;
        }
    }

    /// `ide_sect_range_ok()`.
    fn sect_range_ok(&self, sector: u64, n: u64) -> bool {
        let total = self.nb_sectors;
        sector <= total && n <= total - sector
    }

    /// `ide_cmd_lba48_transform()`.
    fn lba48_transform(&mut self, lba48: bool) {
        self.lba48 = lba48;
        if !lba48 {
            if self.nsector == 0 {
                self.nsector = 256;
            }
        } else if self.nsector == 0 && self.hob_nsector == 0 {
            self.nsector = 65536;
        } else {
            self.nsector |= u32::from(self.hob_nsector) << 8;
        }
    }

    /// `ide_abort_command()`.
    pub(crate) fn abort_command(&mut self, h: &mut dyn IdeHost) {
        self.status = READY_STAT | ERR_STAT;
        self.error = ABRT_ERR;
        self.transfer_stop(h);
    }

    /// `ide_transfer_start_norecurse()` for an adapter with a `pio_transfer` hook: the data
    /// moves right away and the caller carries on with what QEMU would run as the end transfer
    /// function.
    pub(crate) fn transfer_start(&mut self, h: &mut dyn IdeHost, start: usize, len: usize) {
        if self.status & ERR_STAT == 0 {
            self.status |= DRQ_STAT;
        }
        h.pio_transfer(self, start, len);
    }

    /// `ide_transfer_halt()`.
    pub(crate) fn transfer_halt(&mut self) {
        self.status &= !DRQ_STAT;
    }

    /// `ide_transfer_stop()`.
    pub(crate) fn transfer_stop(&mut self, h: &mut dyn IdeHost) {
        self.transfer_halt();
        h.cmd_done(self);
    }

    /// `ide_dma_buf_commit()`.
    pub(crate) fn dma_buf_commit(&mut self, h: &mut dyn IdeHost, tx_bytes: u32) {
        h.commit_buf(tx_bytes);
        self.io_buffer_offset += u64::from(tx_bytes);
    }

    /// `ide_set_inactive()`.
    pub(crate) fn set_inactive(&mut self, h: &mut dyn IdeHost) {
        h.cmd_done(self);
    }

    /// `ide_dma_error()`.
    fn dma_error(&mut self, h: &mut dyn IdeHost) {
        self.dma_buf_commit(h, 0);
        self.abort_command(h);
        self.set_inactive(h);
    }

    /// `ide_bus_exec_cmd()`: runs ATA command `val` with the task file as the guest set it.
    pub(crate) fn exec_cmd(&mut self, h: &mut dyn IdeHost, val: u8) {
        // Only DEVICE RESET is allowed while BSY or DRQ is set, and only to ATAPI devices.
        if self.status & (BUSY_STAT | DRQ_STAT) != 0
            && (val != WIN_DEVICE_RESET || self.kind != DriveKind::Cd)
        {
            return;
        }
        let (ok, set_dsc) = cmd_info(val);
        if ok & self.kind.mask() == 0 {
            self.abort_command(h);
            return;
        }
        self.status = READY_STAT | BUSY_STAT;
        self.error = 0;
        self.io_buffer_offset = 0;

        if self.run_handler(h, val) {
            self.status &= !BUSY_STAT;
            if set_dsc && self.error == 0 {
                self.status |= SEEK_STAT;
            }
            h.cmd_done(self);
        }
    }

    /// The handler column of `ide_cmd_table`. Returns whether the completion code should run.
    fn run_handler(&mut self, h: &mut dyn IdeHost, cmd: u8) -> bool {
        match cmd {
            // DATA SET MANAGEMENT: TRIM is not implemented, and nothing else is defined.
            0x06 => {
                self.abort_command(h);
                true
            }
            WIN_DEVICE_RESET => self.cmd_device_reset(),
            0x10 | 0x70 | 0x94..=0x97 | 0x99 | 0xe0..=0xe3 | 0xe6 => true,
            0x20 | 0x21 | WIN_READ_EXT => self.cmd_read_pio(h, cmd == WIN_READ_EXT),
            WIN_READDMA_EXT | 0xc8 | 0xc9 => self.cmd_dma(h, cmd == WIN_READDMA_EXT, DmaCmd::Read),
            WIN_READ_NATIVE_MAX_EXT | 0xf8 => self.cmd_read_native_max(h, cmd),
            WIN_MULTREAD_EXT | 0xc4 => self.cmd_read_multiple(h, cmd == WIN_MULTREAD_EXT),
            0x30 | 0x31 | WIN_WRITE_EXT | 0x3c => self.cmd_write_pio(h, cmd == WIN_WRITE_EXT),
            WIN_WRITEDMA_EXT | 0xca | 0xcb => {
                self.cmd_dma(h, cmd == WIN_WRITEDMA_EXT, DmaCmd::Write)
            }
            WIN_MULTWRITE_EXT | 0xc5 => self.cmd_write_multiple(h, cmd == WIN_MULTWRITE_EXT),
            0x40..=0x42 => {
                self.lba48_transform(cmd == WIN_VERIFY_EXT);
                true
            }
            0x90 => self.cmd_exec_dev_diagnostic(),
            0x91 => self.cmd_specify(h),
            0x98 | 0xe5 => {
                self.nsector = 0xff;
                true
            }
            0xa0 => self.cmd_packet(h),
            0xa1 => {
                self.atapi_identify();
                self.status = READY_STAT | SEEK_STAT;
                self.transfer_start(h, 0, 512);
                self.transfer_stop(h);
                false
            }
            0xc6 => self.cmd_set_multiple_mode(h),
            0xe7 | 0xea => {
                self.flush_cache(h);
                false
            }
            0xec => self.cmd_identify(h),
            0xef => self.cmd_set_features(h),
            // SECURITY FREEZE LOCK, which QEMU treats as CFA WEAR LEVEL.
            0xf5 => {
                self.nsector = 0;
                true
            }
            // SMART is not implemented and aborts. So does anything not listed.
            _ => {
                self.abort_command(h);
                true
            }
        }
    }

    fn cmd_device_reset(&mut self) -> bool {
        self.transfer_halt();
        self.reset();
        self.status = 0;
        false
    }

    fn cmd_identify(&mut self, h: &mut dyn IdeHost) -> bool {
        if self.kind == DriveKind::Hd {
            self.identify();
            self.status = READY_STAT | SEEK_STAT;
            self.transfer_start(h, 0, 512);
            self.transfer_stop(h);
            false
        } else {
            self.set_signature();
            self.abort_command(h);
            true
        }
    }

    /// `ide_identify()`.
    pub(crate) fn identify(&mut self) {
        if !self.identify_set {
            let p = &mut self.identify_data;
            p.fill(0);
            put_le16(p, 0, 0x0040);
            put_le16(p, 1, self.cylinders);
            put_le16(p, 3, self.heads);
            put_le16(p, 4, 512 * self.sectors);
            put_le16(p, 5, 512);
            put_le16(p, 6, self.sectors);
            padstr(p, 10, &self.serial, 20);
            put_le16(p, 20, 3);
            put_le16(p, 21, 512);
            put_le16(p, 22, 4);
            padstr(p, 23, &self.version, 8);
            padstr(p, 27, &self.model, 40);
            put_le16(p, 47, 0x8000 | MAX_MULT_SECTORS);
            put_le16(p, 48, 1);
            put_le16(p, 49, (1 << 11) | (1 << 9) | (1 << 8));
            put_le16(p, 51, 0x200);
            put_le16(p, 52, 0x200);
            put_le16(p, 53, 1 | (1 << 1) | (1 << 2));
            put_le16(p, 54, self.cylinders);
            put_le16(p, 55, self.heads);
            put_le16(p, 56, self.sectors);
            let oldsize = self.cylinders.wrapping_mul(self.heads).wrapping_mul(self.sectors);
            put_le16(p, 57, oldsize);
            put_le16(p, 58, oldsize >> 16);
            if self.mult_sectors != 0 {
                put_le16(p, 59, 0x100 | self.mult_sectors);
            }
            put_le16(p, 62, 0x07);
            put_le16(p, 63, 0x07);
            put_le16(p, 64, 0x03);
            put_le16(p, 65, 120);
            put_le16(p, 66, 120);
            put_le16(p, 67, 120);
            put_le16(p, 68, 120);
            if self.ncq_queues != 0 {
                put_le16(p, 75, self.ncq_queues - 1);
                put_le16(p, 76, 1 << 8);
            }
            put_le16(p, 80, 0xf0);
            put_le16(p, 81, 0x16);
            put_le16(p, 82, (1 << 14) | (1 << 5) | 1);
            put_le16(p, 83, (1 << 14) | (1 << 13) | (1 << 12) | (1 << 10));
            put_le16(p, 84, 1 << 14);
            if self.write_cache {
                put_le16(p, 85, (1 << 14) | (1 << 5) | 1);
            } else {
                put_le16(p, 85, (1 << 14) | 1);
            }
            put_le16(p, 86, (1 << 13) | (1 << 12) | (1 << 10));
            put_le16(p, 87, 1 << 14);
            put_le16(p, 88, 0x3f | (1 << 13));
            put_le16(p, 93, 1 | (1 << 14) | 0x2000);
            // A 512 byte physical block.
            put_le16(p, 106, 0x6000);
            put_le16(p, 217, 0);

            // ide_identify_size()
            let nb = self.nb_sectors;
            let lba28 = nb.min((1 << 28) - 1);
            put_le16(p, 60, lba28 as u32);
            put_le16(p, 61, (lba28 >> 16) as u32);
            put_le16(p, 100, nb as u32);
            put_le16(p, 101, (nb >> 16) as u32);
            put_le16(p, 102, (nb >> 32) as u32);
            put_le16(p, 103, (nb >> 48) as u32);
            self.identify_set = true;
        }
        self.io_buffer[..512].copy_from_slice(&self.identify_data);
    }

    /// `ide_atapi_identify()`, with `USE_DMA_CDROM` as QEMU builds it.
    pub(crate) fn atapi_identify(&mut self) {
        if !self.identify_set {
            let p = &mut self.identify_data;
            p.fill(0);
            put_le16(p, 0, (2 << 14) | (5 << 8) | (1 << 7) | (2 << 5));
            padstr(p, 10, &self.serial, 20);
            put_le16(p, 20, 3);
            put_le16(p, 21, 512);
            put_le16(p, 22, 4);
            padstr(p, 23, &self.version, 8);
            padstr(p, 27, &self.model, 40);
            put_le16(p, 48, 1);
            put_le16(p, 49, (1 << 9) | (1 << 8));
            put_le16(p, 53, 7);
            put_le16(p, 62, 7);
            put_le16(p, 63, 7);
            put_le16(p, 64, 3);
            put_le16(p, 65, 0xb4);
            put_le16(p, 66, 0xb4);
            put_le16(p, 67, 0x12c);
            put_le16(p, 68, 0xb4);
            put_le16(p, 71, 30);
            put_le16(p, 72, 30);
            if self.ncq_queues != 0 {
                put_le16(p, 75, self.ncq_queues - 1);
                put_le16(p, 76, 1 << 8);
            }
            put_le16(p, 80, 0x1e);
            put_le16(p, 88, 0x3f | (1 << 13));
            self.identify_set = true;
        }
        self.io_buffer[..512].copy_from_slice(&self.identify_data);
    }

    fn cmd_read_pio(&mut self, h: &mut dyn IdeHost, lba48: bool) -> bool {
        if self.kind == DriveKind::Cd {
            // Odd, but ATA4 8.27.5.2 requires it.
            self.set_signature();
            self.abort_command(h);
            return true;
        }
        self.lba48_transform(lba48);
        self.req_nb_sectors = 1;
        self.sector_read(h);
        false
    }

    fn cmd_read_multiple(&mut self, h: &mut dyn IdeHost, lba48: bool) -> bool {
        if self.mult_sectors == 0 {
            self.abort_command(h);
            return true;
        }
        self.lba48_transform(lba48);
        self.req_nb_sectors = self.mult_sectors;
        self.sector_read(h);
        false
    }

    fn cmd_write_pio(&mut self, h: &mut dyn IdeHost, lba48: bool) -> bool {
        self.lba48_transform(lba48);
        self.req_nb_sectors = 1;
        self.status = SEEK_STAT | READY_STAT;
        self.transfer_start(h, 0, 512);
        self.sector_write(h);
        false
    }

    fn cmd_write_multiple(&mut self, h: &mut dyn IdeHost, lba48: bool) -> bool {
        if self.mult_sectors == 0 {
            self.abort_command(h);
            return true;
        }
        self.lba48_transform(lba48);
        self.req_nb_sectors = self.mult_sectors;
        let n = self.nsector.min(self.req_nb_sectors) as usize;
        self.status = SEEK_STAT | READY_STAT;
        self.transfer_start(h, 0, 512 * n);
        self.sector_write(h);
        false
    }

    /// `ide_rw_error()`.
    fn rw_error(&mut self, h: &mut dyn IdeHost) {
        self.abort_command(h);
    }

    /// `ide_sector_read()` and its completion callback, as one loop.
    fn sector_read(&mut self, h: &mut dyn IdeHost) {
        loop {
            self.status = READY_STAT | SEEK_STAT;
            self.error = 0;
            let sector = self.get_sector();
            let n = self.nsector;
            if n == 0 {
                self.transfer_stop(h);
                return;
            }
            self.status |= BUSY_STAT;
            let n = n.min(self.req_nb_sectors);
            if !self.sect_range_ok(sector, u64::from(n)) {
                self.rw_error(h);
                return;
            }
            let len = n as usize * SECTOR_SIZE as usize;
            let ok = match &self.blk {
                Some(b) => b.read_at(sector * SECTOR_SIZE, &mut self.io_buffer[..len]).is_ok(),
                None => false,
            };
            self.status &= !BUSY_STAT;
            if !ok {
                self.rw_error(h);
                return;
            }
            self.set_sector(sector + u64::from(n));
            self.nsector -= n;
            self.transfer_start(h, 0, len);
        }
    }

    /// `ide_sector_write()` and its completion callback, as one loop. The first chunk of data
    /// is already in the buffer.
    fn sector_write(&mut self, h: &mut dyn IdeHost) {
        loop {
            self.status = READY_STAT | SEEK_STAT | BUSY_STAT;
            let sector = self.get_sector();
            let n = self.nsector.min(self.req_nb_sectors);
            if !self.sect_range_ok(sector, u64::from(n)) {
                self.rw_error(h);
                return;
            }
            let len = n as usize * SECTOR_SIZE as usize;
            let ok = match &self.blk {
                Some(b) => b.write_at(sector * SECTOR_SIZE, &self.io_buffer[..len]).is_ok(),
                None => false,
            };
            self.status &= !BUSY_STAT;
            if !ok {
                self.rw_error(h);
                return;
            }
            self.nsector -= n;
            self.set_sector(sector + u64::from(n));
            if self.nsector == 0 {
                self.transfer_stop(h);
                return;
            }
            let n1 = self.nsector.min(self.req_nb_sectors) as usize;
            self.transfer_start(h, 0, n1 * SECTOR_SIZE as usize);
        }
    }

    fn cmd_dma(&mut self, h: &mut dyn IdeHost, lba48: bool, cmd: DmaCmd) -> bool {
        self.lba48_transform(lba48);
        // ide_sector_start_dma(), ide_start_dma() and ahci_start_dma().
        self.status = READY_STAT | SEEK_STAT | DRQ_STAT;
        self.io_buffer_size = 0;
        self.dma_cmd = cmd;
        self.io_buffer_index = 0;
        self.io_buffer_offset = 0;
        self.dma_loop(h);
        false
    }

    /// `ide_dma_cb()` as a loop, one pass per scatter list the guest's PRDT yields.
    fn dma_loop(&mut self, h: &mut dyn IdeHost) {
        loop {
            let n = (self.io_buffer_size >> 9) as u32;
            if n > 0 {
                self.dma_buf_commit(h, n * SECTOR_SIZE as u32);
                let sector = self.get_sector() + u64::from(n);
                self.set_sector(sector);
                self.nsector -= n;
            }

            if self.nsector == 0 {
                self.status = READY_STAT | SEEK_STAT;
                self.set_inactive(h);
                return;
            }

            let n = self.nsector;
            let want = u64::from(n) * SECTOR_SIZE;
            self.io_buffer_index = 0;
            let Some(sg) = h.sglist(want, self.io_buffer_offset) else {
                self.dma_error(h);
                return;
            };
            self.io_buffer_size = sg.size as usize;

            if sg.size < want {
                // The PRDs are too short for this request. Stop without an error, and let the
                // guest see the short byte count.
                self.status = READY_STAT | SEEK_STAT;
                self.dma_buf_commit(h, 0);
                self.set_inactive(h);
                return;
            }

            let sector = self.get_sector();
            if !self.sect_range_ok(sector, u64::from(n)) {
                self.dma_error(h);
                return;
            }

            let Some(blk) = self.blk.clone() else {
                self.dma_error(h);
                return;
            };
            let mut buf = vec![0; sg.size as usize];
            let offset = sector * SECTOR_SIZE;
            let ok = match self.dma_cmd {
                DmaCmd::Read => {
                    let ok = blk.read_at(offset, &mut buf).is_ok();
                    if ok {
                        h.write_guest(&sg, &buf);
                    }
                    ok
                }
                _ => {
                    h.read_guest(&sg, &mut buf);
                    blk.write_at(offset, &buf).is_ok()
                }
            };
            if !ok {
                self.dma_error(h);
                return;
            }
        }
    }

    /// `ide_flush_cache()` and `ide_flush_cb()`.
    fn flush_cache(&mut self, h: &mut dyn IdeHost) {
        if let Some(blk) = self.blk.clone() {
            self.status |= BUSY_STAT;
            if blk.flush().is_err() {
                self.rw_error(h);
                return;
            }
        }
        self.status = READY_STAT | SEEK_STAT;
        h.cmd_done(self);
    }

    fn cmd_read_native_max(&mut self, h: &mut dyn IdeHost, cmd: u8) -> bool {
        if self.nb_sectors == 0 {
            self.abort_command(h);
            return true;
        }
        let (aheads, asectors) = (self.heads, self.sectors);
        self.heads = self.drive_heads;
        self.sectors = self.drive_sectors;
        self.lba48_transform(cmd == WIN_READ_NATIVE_MAX_EXT);
        self.set_sector(self.nb_sectors - 1);
        self.heads = aheads;
        self.sectors = asectors;
        true
    }

    fn cmd_specify(&mut self, h: &mut dyn IdeHost) -> bool {
        if self.kind == DriveKind::Hd {
            self.heads = u32::from(self.select & ATA_DEV_HS) + 1;
            self.sectors = self.nsector & 0xff;
        } else {
            self.abort_command(h);
        }
        true
    }

    fn cmd_set_multiple_mode(&mut self, h: &mut dyn IdeHost) -> bool {
        let n = self.nsector & 0xff;
        if n != 0 && (n > MAX_MULT_SECTORS || self.nsector & self.nsector.wrapping_sub(1) != 0) {
            self.abort_command(h);
        } else {
            self.mult_sectors = n;
        }
        true
    }

    fn cmd_exec_dev_diagnostic(&mut self) -> bool {
        self.select = ATA_DEV_ALWAYS_ON;
        self.set_signature();
        if self.kind == DriveKind::Cd {
            self.status = 0;
        } else {
            self.status = READY_STAT | SEEK_STAT;
        }
        self.error = 0x01;
        false
    }

    fn cmd_set_features(&mut self, h: &mut dyn IdeHost) -> bool {
        let p = &mut self.identify_data;
        match self.feature {
            0x02 => {
                self.write_cache = true;
                put_le16(p, 85, (1 << 14) | (1 << 5) | 1);
                return true;
            }
            0x82 => {
                self.write_cache = false;
                put_le16(p, 85, (1 << 14) | 1);
                self.flush_cache(h);
                return false;
            }
            0xcc => {
                self.reset_reverts = true;
                return true;
            }
            0x66 => {
                self.reset_reverts = false;
                return true;
            }
            0xaa | 0x55 | 0x05 | 0x85 | 0x69 | 0x67 | 0x96 | 0x9a | 0x42 | 0xc2 => return true,
            0x03 => {
                let val = self.nsector & 0x07;
                let (w62, w63, w88) = match self.nsector >> 3 {
                    0x00 | 0x01 => (0x07, 0x07, 0x3f),
                    0x02 => (0x07 | (1 << (val + 8)), 0x07, 0x3f),
                    0x04 => (0x07, 0x07 | (1 << (val + 8)), 0x3f),
                    0x08 => (0x07, 0x07, 0x3f | (1 << (val + 8))),
                    _ => {
                        self.abort_command(h);
                        return true;
                    }
                };
                put_le16(p, 62, w62);
                put_le16(p, 63, w63);
                put_le16(p, 88, w88);
                return true;
            }
            _ => {}
        }
        self.abort_command(h);
        true
    }

    fn cmd_packet(&mut self, h: &mut dyn IdeHost) -> bool {
        // Overlapping commands are not supported.
        if self.feature & 0x02 != 0 {
            self.abort_command(h);
            return true;
        }
        self.status = READY_STAT | SEEK_STAT;
        self.atapi_dma = self.feature & 1 != 0;
        if self.atapi_dma {
            self.dma_cmd = DmaCmd::Atapi;
        }
        self.nsector = 1;
        self.transfer_start(h, 0, 12);
        self.atapi_cmd(h);
        false
    }
}

/// The length of an IDE I/O buffer, `io_buffer_total_len`, which is how many bytes the
/// `ide_drive/pio_state` subsection carries.
pub const IDE_IO_BUFFER_TOTAL_LEN: usize = IDE_DMA_BUF_SECTORS * SECTOR_SIZE as usize + 4;

/// `ide_transfer_stop`'s index in QEMU's `transfer_end_table`.
const END_TRANSFER_STOP_IDX: u8 = 2;

/// `vmstate_ide_bus` (version 1) with its `ide_bus/error` subsection.
///
/// The one-drive bus of an AHCI port has no state of its own here: `cmd` and `unit` stay 0 on
/// AHCI, and with the "report" error policy no request ever waits to be retried.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdeBusVmState {
    pub cmd: u8,
    pub unit: u8,
    /// `ide_bus/error`, sent when not 0.
    pub error_status: i32,
    /// `ide_bus/error`, version 2.
    pub retry_sector_num: i64,
    /// `ide_bus/error`, version 2.
    pub retry_nsector: u32,
    /// `ide_bus/error`, version 2.
    pub retry_unit: u8,
}

/// `vmstate_ide_drive` (version 3) with its subsections.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IdeDriveVmState {
    pub mult_sectors: i32,
    pub identify_set: i32,
    /// 512 bytes, sent only when `identify_set` is not 0.
    pub identify_data: Vec<u8>,
    pub feature: u8,
    pub error: u8,
    pub nsector: u32,
    pub sector: u8,
    pub lcyl: u8,
    pub hcyl: u8,
    pub hob_feature: u8,
    pub hob_sector: u8,
    pub hob_nsector: u8,
    pub hob_lcyl: u8,
    pub hob_hcyl: u8,
    pub select: u8,
    pub status: u8,
    pub lba48: u8,
    pub sense_key: u8,
    pub asc: u8,
    /// Version 3. There is no media change model here, so it is sent as 0 and dropped on load.
    pub cdrom_changed: u8,

    /// `ide_drive/pio_state`, sent while DRQ is set.
    pub req_nb_sectors: i32,
    /// `ide_drive/pio_state`: [`IDE_IO_BUFFER_TOTAL_LEN`] bytes.
    pub io_buffer: Vec<u8>,
    /// `ide_drive/pio_state`.
    pub cur_io_buffer_offset: i32,
    /// `ide_drive/pio_state`.
    pub cur_io_buffer_len: i32,
    /// `ide_drive/pio_state`.
    pub end_transfer_fn_idx: u8,
    /// `ide_drive/pio_state`.
    pub elementary_transfer_size: i32,
    /// `ide_drive/pio_state`.
    pub packet_transfer_size: i32,

    /// `ide_drive/tray_state`, sent when the tray is open or locked.
    pub tray_open: bool,
    /// `ide_drive/tray_state`.
    pub tray_locked: bool,

    /// `ide_drive/atapi/gesn_state` (`events.new_media`), sent when either flag is set.
    pub new_media: bool,
    /// `ide_drive/atapi/gesn_state` (`events.eject_request`).
    pub eject_request: bool,
}

impl Default for IdeDriveVmState {
    fn default() -> Self {
        IdeDriveVmState {
            mult_sectors: 0,
            identify_set: 0,
            identify_data: vec![0; 512],
            feature: 0,
            error: 0,
            nsector: 0,
            sector: 0,
            lcyl: 0,
            hcyl: 0,
            hob_feature: 0,
            hob_sector: 0,
            hob_nsector: 0,
            hob_lcyl: 0,
            hob_hcyl: 0,
            select: 0,
            status: 0,
            lba48: 0,
            sense_key: 0,
            asc: 0,
            cdrom_changed: 0,
            req_nb_sectors: 0,
            io_buffer: Vec::new(),
            cur_io_buffer_offset: 0,
            cur_io_buffer_len: 0,
            end_transfer_fn_idx: 0,
            elementary_transfer_size: 0,
            packet_transfer_size: 0,
            tray_open: false,
            tray_locked: false,
            new_media: false,
            eject_request: false,
        }
    }
}

impl IdeDriveVmState {
    /// Whether the drive is in a state [`IdeDrive::vmstate_load`] takes: no PIO transfer in
    /// progress and the tray closed.
    pub(crate) fn check(&self) -> Result<(), String> {
        if self.status & DRQ_STAT != 0 {
            return Err("ide: a PIO transfer in progress (ide_drive/pio_state) is not supported"
                .to_string());
        }
        if self.tray_open {
            return Err("ide: an open tray is not supported".to_string());
        }
        if self.identify_set != 0 && self.identify_data.len() != 512 {
            return Err(format!("ide: {} bytes of identify data", self.identify_data.len()));
        }
        Ok(())
    }
}

impl IdeDrive {
    /// The drive's part of the stream. `ide_drive_pio_pre_save()` runs only if DRQ is set,
    /// which a synchronous transfer never leaves behind; the buffer position is then unknown
    /// and goes out as the start, with `ide_transfer_stop` as the end transfer function.
    pub(crate) fn vmstate_save(&self) -> IdeDriveVmState {
        let mut v = IdeDriveVmState {
            mult_sectors: self.mult_sectors as i32,
            identify_set: i32::from(self.identify_set),
            identify_data: self.identify_data.to_vec(),
            feature: self.feature,
            error: self.error,
            nsector: self.nsector,
            sector: self.sector,
            lcyl: self.lcyl,
            hcyl: self.hcyl,
            hob_feature: self.hob_feature,
            hob_sector: self.hob_sector,
            hob_nsector: self.hob_nsector,
            hob_lcyl: self.hob_lcyl,
            hob_hcyl: self.hob_hcyl,
            select: self.select,
            status: self.status,
            lba48: u8::from(self.lba48),
            sense_key: self.sense_key,
            asc: self.asc,
            tray_locked: self.tray_locked,
            ..IdeDriveVmState::default()
        };
        if self.status & DRQ_STAT != 0 {
            let mut buf = self.io_buffer.clone();
            buf.resize(IDE_IO_BUFFER_TOTAL_LEN, 0);
            v.req_nb_sectors = self.req_nb_sectors as i32;
            v.io_buffer = buf;
            v.end_transfer_fn_idx = END_TRANSFER_STOP_IDX;
            v.elementary_transfer_size = self.elementary_transfer_size as i32;
            v.packet_transfer_size = self.packet_transfer_size as i32;
        }
        v
    }

    /// Loads the drive's part of the stream, then `ide_drive_post_load()`. Fails, changing
    /// nothing, for the states [`IdeDriveVmState::check`] refuses. The media change events
    /// (`cdrom_changed` and the GESN flags) are dropped.
    pub(crate) fn vmstate_load(&mut self, v: &IdeDriveVmState) -> Result<(), String> {
        v.check()?;
        self.mult_sectors = v.mult_sectors as u32;
        self.identify_set = v.identify_set != 0;
        if self.identify_set {
            self.identify_data.copy_from_slice(&v.identify_data);
        }
        self.feature = v.feature;
        self.error = v.error;
        self.nsector = v.nsector;
        self.sector = v.sector;
        self.lcyl = v.lcyl;
        self.hcyl = v.hcyl;
        self.hob_feature = v.hob_feature;
        self.hob_sector = v.hob_sector;
        self.hob_nsector = v.hob_nsector;
        self.hob_lcyl = v.hob_lcyl;
        self.hob_hcyl = v.hob_hcyl;
        self.select = v.select;
        self.status = v.status;
        self.lba48 = v.lba48 != 0;
        self.sense_key = v.sense_key;
        self.asc = v.asc;
        self.tray_locked = v.tray_locked;
        // ide_drive_post_load(): the write cache follows IDENTIFY word 85 bit 5. QEMU tests
        // byte 85 of the buffer instead; this reads the word.
        if self.blk.is_some() && self.identify_set {
            let w85 = u16::from_le_bytes([self.identify_data[170], self.identify_data[171]]);
            self.write_cache = w85 & (1 << 5) != 0;
        }
        Ok(())
    }
}
