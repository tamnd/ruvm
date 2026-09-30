// SPDX-License-Identifier: GPL-2.0-or-later

//! The ATAPI packet interface of a CD drive, ported from a subset of QEMU's `hw/ide/atapi.c`.
//!
//! The ported commands are TEST UNIT READY, REQUEST SENSE, INQUIRY, PREVENT ALLOW MEDIUM
//! REMOVAL, READ CAPACITY, READ(10), READ(12), SEEK, GET CONFIGURATION, MODE SENSE(10) and SET
//! CD SPEED. Everything else, such as READ TOC, GET EVENT STATUS NOTIFICATION, READ CD or START
//! STOP UNIT, fails with ILLEGAL REQUEST and "invalid command operation code". Media changes are
//! not modeled: the disc is the backend the drive was created with, for the life of the drive.

use crate::ide::{BUSY_STAT, DRQ_STAT, ERR_STAT, IdeDrive, IdeHost, MC_ERR, READY_STAT, SEEK_STAT};

const ATAPI_SECTOR_SIZE: usize = 2048;

const ATAPI_INT_REASON_CD: u32 = 0x01;
const ATAPI_INT_REASON_IO: u32 = 0x02;

const NOT_READY: u8 = 0x02;
const ILLEGAL_REQUEST: u8 = 0x05;
const UNIT_ATTENTION: u8 = 0x06;

const ASC_ILLEGAL_OPCODE: u8 = 0x20;
const ASC_LOGICAL_BLOCK_OOR: u8 = 0x21;
const ASC_INV_FIELD_IN_CMD_PACKET: u8 = 0x24;
const ASC_SAVING_PARAMETERS_NOT_SUPPORTED: u8 = 0x39;
const ASC_MEDIUM_NOT_PRESENT: u8 = 0x3a;
const ASC_DATA_PHASE_ERROR: u8 = 0x4b;

const MMC_PROFILE_CD_ROM: u16 = 0x0008;
const MMC_PROFILE_DVD_ROM: u16 = 0x0010;

/// `CD_MAX_SECTORS`: the most 512 byte sectors a CD holds. Anything larger is a DVD.
const CD_MAX_SECTORS: u64 = 80 * 60 * 75 * 2048 / 512;

/// The command may run while a unit attention condition is pending.
const ALLOW_UA: u8 = 0x01;
/// The command needs a disc.
const CHECK_READY: u8 = 0x02;
/// The command moves no data, so a zero byte count limit is fine.
const NONDATA: u8 = 0x04;

const GPCMD_READ_10: u8 = 0x28;

#[derive(Clone, Copy)]
enum Cmd {
    TestUnitReady,
    RequestSense,
    Inquiry,
    PreventAllow,
    ReadCapacity,
    Read,
    Seek,
    GetConfiguration,
    ModeSense,
    SetSpeed,
}

/// The ported rows of `atapi_cmd_table`.
fn lookup(op: u8) -> Option<(Cmd, u8)> {
    Some(match op {
        0x00 => (Cmd::TestUnitReady, CHECK_READY | NONDATA),
        0x03 => (Cmd::RequestSense, ALLOW_UA),
        0x12 => (Cmd::Inquiry, ALLOW_UA),
        0x1e => (Cmd::PreventAllow, NONDATA),
        0x25 => (Cmd::ReadCapacity, CHECK_READY),
        0x28 | 0xa8 => (Cmd::Read, CHECK_READY),
        0x2b => (Cmd::Seek, CHECK_READY | NONDATA),
        0x46 => (Cmd::GetConfiguration, ALLOW_UA),
        0x5a => (Cmd::ModeSense, 0),
        0xbb => (Cmd::SetSpeed, NONDATA),
        _ => return None,
    })
}

fn be16(b: &[u8]) -> u32 {
    u32::from(u16::from_be_bytes([b[0], b[1]]))
}

fn be32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn put_be16(b: &mut [u8], v: u16) {
    b[..2].copy_from_slice(&v.to_be_bytes());
}

fn put_be32(b: &mut [u8], v: u32) {
    b[..4].copy_from_slice(&v.to_be_bytes());
}

/// `padstr8()`: a space padded SCSI string, without byte swapping.
fn padstr8(buf: &mut [u8], src: &str) {
    let src = src.as_bytes();
    for (i, b) in buf.iter_mut().enumerate() {
        *b = src.get(i).copied().unwrap_or(b' ');
    }
}

impl IdeDrive {
    fn media_present(&self) -> bool {
        self.nb_sectors > 0
    }

    /// `ide_atapi_cmd_ok()`.
    fn atapi_cmd_ok(&mut self, h: &mut dyn IdeHost) {
        self.error = 0;
        self.status = READY_STAT | SEEK_STAT;
        self.nsector = (self.nsector & !7) | ATAPI_INT_REASON_IO | ATAPI_INT_REASON_CD;
        self.transfer_stop(h);
    }

    /// `ide_atapi_cmd_error()`.
    fn atapi_cmd_error(&mut self, h: &mut dyn IdeHost, sense_key: u8, asc: u8) {
        self.error = sense_key << 4;
        self.status = READY_STAT | ERR_STAT;
        self.nsector = (self.nsector & !7) | ATAPI_INT_REASON_IO | ATAPI_INT_REASON_CD;
        self.sense_key = sense_key;
        self.asc = asc;
        self.transfer_stop(h);
    }

    /// `ide_atapi_io_error()`.
    fn atapi_io_error(&mut self, h: &mut dyn IdeHost) {
        if self.blk.is_none() {
            self.atapi_cmd_error(h, NOT_READY, ASC_MEDIUM_NOT_PRESENT);
        } else {
            self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_LOGICAL_BLOCK_OOR);
        }
    }

    /// `atapi_byte_count_limit()`.
    fn byte_count_limit(&self) -> i64 {
        let bcl = u16::from(self.lcyl) | u16::from(self.hcyl) << 8;
        if bcl == 0xffff { 0xfffe } else { i64::from(bcl) }
    }

    /// Reads 2048 byte sector `self.lba` into the start of the I/O buffer.
    fn cd_read_sector(&mut self) -> bool {
        let Some(blk) = self.blk.clone() else {
            return false;
        };
        let Ok(lba) = u64::try_from(self.lba) else {
            return false;
        };
        blk.read_at(lba * ATAPI_SECTOR_SIZE as u64, &mut self.io_buffer[..ATAPI_SECTOR_SIZE])
            .is_ok()
    }

    /// `ide_atapi_cmd_reply_end()`: the PIO data phase, one DRQ block per pass.
    fn atapi_reply_end(&mut self, h: &mut dyn IdeHost) {
        while self.packet_transfer_size > 0 {
            // See if a new sector must be read.
            if self.lba != -1 && self.io_buffer_index >= self.cd_sector_size {
                self.status |= BUSY_STAT;
                if !self.cd_read_sector() {
                    self.atapi_io_error(h);
                    return;
                }
                self.lba += 1;
                self.io_buffer_index = 0;
                self.status &= !BUSY_STAT;
            }
            let mut size;
            if self.elementary_transfer_size > 0 {
                // Data left to move in this elementary transfer.
                size = (self.cd_sector_size - self.io_buffer_index) as i64;
                size = size.min(self.elementary_transfer_size);
            } else {
                // A new transfer is needed.
                self.nsector = (self.nsector & !7) | ATAPI_INT_REASON_IO;
                let mut bcl = self.byte_count_limit();
                size = self.packet_transfer_size;
                if size > bcl {
                    // The byte count limit must be even in this case.
                    if bcl & 1 != 0 {
                        bcl -= 1;
                    }
                    size = bcl;
                }
                self.lcyl = size as u8;
                self.hcyl = (size >> 8) as u8;
                self.elementary_transfer_size = size;
                // No more than one sector at a time.
                if self.lba != -1 {
                    size = size.min((self.cd_sector_size - self.io_buffer_index) as i64);
                }
            }
            if size <= 0 {
                // A byte count limit of one rounds down to zero, which would spin forever.
                self.abort_command(h);
                return;
            }
            self.packet_transfer_size -= size;
            self.elementary_transfer_size -= size;
            let size = size as usize;
            self.io_buffer_index += size;
            let start = self.io_buffer_index - size;
            self.transfer_start(h, start, size);
        }
        self.atapi_cmd_ok(h);
    }

    /// `ide_atapi_cmd_reply()`.
    fn atapi_reply(&mut self, h: &mut dyn IdeHost, size: usize, max_size: usize) {
        let size = size.min(max_size);
        self.lba = -1;
        self.packet_transfer_size = size as i64;
        self.io_buffer_size = size;
        self.elementary_transfer_size = 0;
        if self.atapi_dma {
            self.status = READY_STAT | SEEK_STAT | DRQ_STAT;
            self.start_atapi_dma(h);
        } else {
            self.status = READY_STAT | SEEK_STAT;
            self.io_buffer_index = 0;
            self.atapi_reply_end(h);
        }
    }

    /// `ide_start_dma()` and `ahci_start_dma()` for ATAPI.
    fn start_atapi_dma(&mut self, h: &mut dyn IdeHost) {
        self.io_buffer_index = 0;
        self.io_buffer_offset = 0;
        self.atapi_dma_loop(h);
    }

    /// `ahci_dma_rw_buf()` for a device to host transfer.
    fn dma_rw_buf(&mut self, h: &mut dyn IdeHost) -> bool {
        let start = self.io_buffer_index;
        let l = self.io_buffer_size.saturating_sub(start);
        let Some(sg) = h.sglist(l as u64, self.io_buffer_offset) else {
            return false;
        };
        let data = self.io_buffer[start..start + l].to_vec();
        h.write_guest(&sg, &data);
        self.dma_buf_commit(h, l as u32);
        self.io_buffer_index += l;
        true
    }

    /// `ide_atapi_cmd_read_dma_cb()` as a loop.
    fn atapi_dma_loop(&mut self, h: &mut dyn IdeHost) {
        loop {
            if self.io_buffer_size > 0 {
                if self.lba != -1 {
                    self.lba += (self.io_buffer_size >> 11) as i64;
                }
                self.packet_transfer_size -= self.io_buffer_size as i64;
                if !self.dma_rw_buf(h) {
                    self.set_inactive(h);
                    return;
                }
            }

            if self.packet_transfer_size <= 0 {
                self.status = READY_STAT | SEEK_STAT;
                self.nsector = (self.nsector & !7) | ATAPI_INT_REASON_IO | ATAPI_INT_REASON_CD;
                self.set_inactive(h);
                return;
            }

            self.io_buffer_index = 0;
            let n = ((self.packet_transfer_size >> 11) as usize)
                .min(crate::ide::IDE_DMA_BUF_SECTORS / 4);
            self.io_buffer_size = n * ATAPI_SECTOR_SIZE;
            let ok = match (&self.blk, u64::try_from(self.lba)) {
                (Some(blk), Ok(lba)) => blk
                    .read_at(
                        lba * ATAPI_SECTOR_SIZE as u64,
                        &mut self.io_buffer[..n * ATAPI_SECTOR_SIZE],
                    )
                    .is_ok(),
                _ => false,
            };
            if !ok {
                // What ide_handle_rw_error() does with the default "report" policy.
                self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_LOGICAL_BLOCK_OOR);
                self.set_inactive(h);
                return;
            }
        }
    }

    /// `ide_atapi_cmd()`: runs the packet the guest just sent, which sits at the start of the
    /// I/O buffer.
    pub(crate) fn atapi_cmd(&mut self, h: &mut dyn IdeHost) {
        let mut packet = [0u8; 16];
        packet.copy_from_slice(&self.io_buffer[..16]);
        let Some((cmd, flags)) = lookup(packet[0]) else {
            self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_ILLEGAL_OPCODE);
            return;
        };

        // With a unit attention condition pending, only ALLOW_UA commands run.
        if self.sense_key == UNIT_ATTENTION && flags & ALLOW_UA == 0 {
            // ide_atapi_cmd_check_status()
            self.error = MC_ERR | (UNIT_ATTENTION << 4);
            self.status = ERR_STAT;
            self.nsector = 0;
            return;
        }

        if flags & CHECK_READY != 0 && !self.media_present() {
            self.atapi_cmd_error(h, NOT_READY, ASC_MEDIUM_NOT_PRESENT);
            return;
        }

        // A PIO data command with a zero byte count limit aborts at the ATA level.
        if flags & NONDATA == 0 && !self.atapi_dma && self.byte_count_limit() == 0 {
            self.abort_command(h);
            return;
        }

        match cmd {
            Cmd::TestUnitReady | Cmd::SetSpeed => self.atapi_cmd_ok(h),
            Cmd::RequestSense => self.cmd_request_sense(h, &packet),
            Cmd::Inquiry => self.cmd_inquiry(h, &packet),
            Cmd::PreventAllow => {
                self.tray_locked = packet[4] & 1 != 0;
                self.atapi_cmd_ok(h);
            }
            Cmd::ReadCapacity => {
                let total = self.nb_sectors >> 2;
                put_be32(&mut self.io_buffer[0..], total.wrapping_sub(1) as u32);
                put_be32(&mut self.io_buffer[4..], ATAPI_SECTOR_SIZE as u32);
                self.atapi_reply(h, 8, 8);
            }
            Cmd::Read => self.cmd_read(h, &packet),
            Cmd::Seek => {
                let lba = u64::from(be32(&packet[2..]));
                if lba >= self.nb_sectors >> 2 {
                    self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_LOGICAL_BLOCK_OOR);
                } else {
                    self.atapi_cmd_ok(h);
                }
            }
            Cmd::GetConfiguration => self.cmd_get_configuration(h, &packet),
            Cmd::ModeSense => self.cmd_mode_sense(h, &packet),
        }
    }

    fn cmd_request_sense(&mut self, h: &mut dyn IdeHost, packet: &[u8; 16]) {
        let max_len = usize::from(packet[4]);
        let buf = &mut self.io_buffer;
        buf[..18].fill(0);
        buf[0] = 0x70 | (1 << 7);
        buf[2] = self.sense_key;
        buf[7] = 10;
        buf[12] = self.asc;
        if self.sense_key == UNIT_ATTENTION {
            self.sense_key = 0;
        }
        self.atapi_reply(h, 18, max_len);
    }

    fn cmd_inquiry(&mut self, h: &mut dyn IdeHost, packet: &[u8; 16]) {
        let page_code = packet[2];
        let max_len = usize::from(packet[4]);
        let serial = self.serial.clone();
        let model = self.model.clone();
        let version = self.version.clone();
        let buf = &mut self.io_buffer;
        let mut idx;
        let size_idx;
        let preamble_len;

        if packet[1] & 1 != 0 {
            // Enable Vital Product Data: byte 2 selects the page.
            preamble_len = 4;
            size_idx = 3;
            buf[0] = 0x05;
            buf[1] = page_code;
            buf[2] = 0;
            idx = 4;
            match page_code {
                0x00 => {
                    buf[4] = 0x00;
                    buf[5] = 0x83;
                    idx = 6;
                }
                0x83 => {
                    if idx + 24 > max_len {
                        self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_DATA_PHASE_ERROR);
                        return;
                    }
                    buf[idx..idx + 4].copy_from_slice(&[0x02, 0x00, 0x00, 20]);
                    idx += 4;
                    padstr8(&mut buf[idx..idx + 20], &serial);
                    idx += 20;
                    if idx + 72 <= max_len {
                        buf[idx..idx + 4].copy_from_slice(&[0x02, 0x01, 0x00, 68]);
                        idx += 4;
                        padstr8(&mut buf[idx..idx + 8], "ATA");
                        idx += 8;
                        padstr8(&mut buf[idx..idx + 40], &model);
                        idx += 40;
                        padstr8(&mut buf[idx..idx + 20], &serial);
                        idx += 20;
                    }
                }
                _ => {
                    self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_INV_FIELD_IN_CMD_PACKET);
                    return;
                }
            }
        } else {
            preamble_len = 5;
            size_idx = 4;
            buf[0] = 0x05; // CD-ROM
            buf[1] = 0x80; // removable
            buf[2] = 0x00; // ISO
            buf[3] = 0x21; // ATAPI-2
            buf[5] = 0;
            buf[6] = 0;
            buf[7] = 0;
            padstr8(&mut buf[8..16], "QEMU");
            padstr8(&mut buf[16..32], "QEMU DVD-ROM");
            padstr8(&mut buf[32..36], &version);
            idx = 36;
        }
        buf[size_idx] = (idx - preamble_len) as u8;
        self.atapi_reply(h, idx, max_len);
    }

    fn cmd_read(&mut self, h: &mut dyn IdeHost, packet: &[u8; 16]) {
        let total = self.nb_sectors >> 2;
        let nb_sectors =
            if packet[0] == GPCMD_READ_10 { be16(&packet[7..]) } else { be32(&packet[6..]) };
        if nb_sectors == 0 {
            self.atapi_cmd_ok(h);
            return;
        }
        let lba = u64::from(be32(&packet[2..]));
        if lba >= total || lba + u64::from(nb_sectors) > total {
            self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_LOGICAL_BLOCK_OOR);
            return;
        }
        self.lba = lba as i64;
        self.packet_transfer_size = i64::from(nb_sectors) * ATAPI_SECTOR_SIZE as i64;
        self.cd_sector_size = ATAPI_SECTOR_SIZE;
        if self.atapi_dma {
            // ide_atapi_cmd_read_dma()
            self.io_buffer_size = 0;
            self.status = READY_STAT | SEEK_STAT | DRQ_STAT | BUSY_STAT;
            self.start_atapi_dma(h);
        } else {
            // ide_atapi_cmd_read_pio()
            self.elementary_transfer_size = 0;
            self.io_buffer_index = ATAPI_SECTOR_SIZE;
            self.atapi_reply_end(h);
        }
    }

    fn cmd_get_configuration(&mut self, h: &mut dyn IdeHost, packet: &[u8; 16]) {
        // Only feature 0 is supported.
        if packet[2] != 0 || packet[3] != 0 {
            self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_INV_FIELD_IN_CMD_PACKET);
            return;
        }
        let max_len = (be16(&packet[7..]) as usize).min(512);
        let present = self.media_present();
        let dvd = present && self.nb_sectors > CD_MAX_SECTORS;
        let buf = &mut self.io_buffer;
        buf[..max_len.max(20)].fill(0);
        if present {
            let profile = if dvd { MMC_PROFILE_DVD_ROM } else { MMC_PROFILE_CD_ROM };
            put_be16(&mut buf[6..], profile);
        }
        buf[10] = 0x02 | 0x01; // persistent and current
        let mut len = 12;
        for (index, profile) in [MMC_PROFILE_DVD_ROM, MMC_PROFILE_CD_ROM].into_iter().enumerate() {
            // ide_atapi_set_profile()
            let p = 12 + index * 4;
            put_be16(&mut buf[p..], profile);
            buf[p + 2] = u8::from(buf[p] == buf[6] && buf[p + 1] == buf[7]);
            buf[11] += 4;
            len += 4;
        }
        put_be32(&mut buf[0..], (len - 4) as u32);
        self.atapi_reply(h, len, max_len);
    }

    fn cmd_mode_sense(&mut self, h: &mut dyn IdeHost, packet: &[u8; 16]) {
        let max_len = be16(&packet[7..]) as usize;
        let action = packet[2] >> 6;
        let code = packet[2] & 0x3f;
        let tray_locked = self.tray_locked;
        match action {
            0 => {}
            1 | 2 => {
                self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_INV_FIELD_IN_CMD_PACKET);
                return;
            }
            _ => {
                self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_SAVING_PARAMETERS_NOT_SUPPORTED);
                return;
            }
        }
        let buf = &mut self.io_buffer;
        let len = match code {
            0x01 => 16,
            0x0e => 24,
            0x2a => 30,
            _ => {
                self.atapi_cmd_error(h, ILLEGAL_REQUEST, ASC_INV_FIELD_IN_CMD_PACKET);
                return;
            }
        };
        put_be16(&mut buf[0..], (len - 2) as u16);
        buf[2] = 0x70;
        buf[3..8].fill(0);
        buf[8] = code;
        buf[9] = (len - 10) as u8;
        match code {
            // Read/write error recovery.
            0x01 => buf[10..16].copy_from_slice(&[0x00, 0x05, 0x00, 0x00, 0x00, 0x00]),
            // Audio control. QEMU leaves the other bytes as they were; they are zeroed here.
            0x0e => buf[10..24].fill(0),
            // Capabilities.
            _ => {
                buf[10] = 0x3b; // read CDR/CDRW/DVDROM/DVDR/DVDRAM
                buf[11] = 0x00;
                buf[12] = 0x71;
                buf[13] = 3 << 5;
                buf[14] = (1 << 0) | (1 << 3) | (1 << 5) | (u8::from(tray_locked) << 1);
                buf[15] = 0x00;
                put_be16(&mut buf[16..], 704);
                buf[18] = 0;
                buf[19] = 2;
                put_be16(&mut buf[20..], 512);
                put_be16(&mut buf[22..], 704);
                buf[24..30].fill(0);
            }
        }
        self.atapi_reply(h, len, max_len);
    }
}
