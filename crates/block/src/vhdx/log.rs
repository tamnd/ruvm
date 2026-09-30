// SPDX-License-Identifier: GPL-2.0-or-later

//! The VHDX log, a port of block/vhdx-log.c.
//!
//! The log is a circular buffer of 4 KiB sectors. Each entry starts with a header and the
//! descriptors, 126 in the first sector and 128 in every further one, followed by one data
//! sector per data descriptor:
//!
//! ```text
//! [ hdr, desc ][   desc   ][ ... ][ data ][ ... ]
//! ```
//!
//! A data descriptor holds the first 8 and the last 4 bytes of the 4 KiB it describes, the
//! data sector the 4084 bytes in between. A zero descriptor says that a range reads as zeroes.
//!
//! When an image is opened, the newest run of valid entries with increasing sequence numbers
//! is written to the image (the log is replayed) and the log is emptied. A metadata update
//! first goes to the log, which is then flushed right away, so the log never holds more than
//! one entry.

use std::io;

use ruvm_base::Error;
use ruvm_qapi::types::PreallocMode;

use super::{
    Guid, MIB, OpenError, State, ZERO_GUID, checksum_calc, einval, guid_at, guid_generate, le32,
    le64, put32, put64, to_io, update_checksum, update_headers, user_visible_write,
};
use crate::node::{Node, errno};

const LOG_MIN_SIZE: u64 = MIB;
const LOG_SECTOR_SIZE: u32 = 4096;
const LOG_SIGNATURE: u32 = 0x6567_6f6c;
const LOG_DESC_SIGNATURE: u32 = 0x6373_6564;
const LOG_ZERO_SIGNATURE: u32 = 0x6f72_657a;
const LOG_DATA_SIGNATURE: u32 = 0x6174_6164;

/// `sizeof(VHDXLogEntryHeader)`.
const LOG_HDR_SIZE: usize = 64;
/// `sizeof(VHDXLogDescriptor)`.
const LOG_DESC_SIZE: usize = 32;

/// `VHDXLogEntries`: where the log is, and the read and write positions in it.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LogEntries {
    pub(super) offset: u64,
    pub(super) length: u64,
    pub(super) write: u32,
    pub(super) read: u32,
    pub(super) sequence: u64,
    pub(super) tail: u32,
}

/// `VHDXLogEntryHeader`.
#[derive(Clone, Copy, Debug, Default)]
struct LogEntryHeader {
    signature: u32,
    checksum: u32,
    entry_length: u32,
    tail: u32,
    sequence_number: u64,
    descriptor_count: u32,
    log_guid: Guid,
    flushed_file_offset: u64,
    last_file_offset: u64,
}

impl LogEntryHeader {
    fn parse(b: &[u8]) -> LogEntryHeader {
        LogEntryHeader {
            signature: le32(b, 0),
            checksum: le32(b, 4),
            entry_length: le32(b, 8),
            tail: le32(b, 12),
            sequence_number: le64(b, 16),
            descriptor_count: le32(b, 24),
            log_guid: guid_at(b, 32),
            flushed_file_offset: le64(b, 48),
            last_file_offset: le64(b, 56),
        }
    }

    fn write(&self, b: &mut [u8]) {
        put32(b, 0, self.signature);
        put32(b, 4, self.checksum);
        put32(b, 8, self.entry_length);
        put32(b, 12, self.tail);
        put64(b, 16, self.sequence_number);
        put32(b, 24, self.descriptor_count);
        put32(b, 28, 0);
        b[32..48].copy_from_slice(&self.log_guid);
        put64(b, 48, self.flushed_file_offset);
        put64(b, 56, self.last_file_offset);
    }
}

/// `VHDXLogDescriptor`. `leading_bytes` doubles as `zero_length` in a zero descriptor.
#[derive(Clone, Copy, Debug, Default)]
struct LogDescriptor {
    signature: u32,
    trailing_bytes: [u8; 4],
    leading_bytes: [u8; 8],
    file_offset: u64,
    sequence_number: u64,
}

impl LogDescriptor {
    fn parse(b: &[u8]) -> LogDescriptor {
        LogDescriptor {
            signature: le32(b, 0),
            trailing_bytes: b[4..8].try_into().unwrap(),
            leading_bytes: b[8..16].try_into().unwrap(),
            file_offset: le64(b, 16),
            sequence_number: le64(b, 24),
        }
    }

    fn write(&self, b: &mut [u8]) {
        put32(b, 0, self.signature);
        b[4..8].copy_from_slice(&self.trailing_bytes);
        b[8..16].copy_from_slice(&self.leading_bytes);
        put64(b, 16, self.file_offset);
        put64(b, 24, self.sequence_number);
    }

    fn zero_length(&self) -> u64 {
        u64::from_le_bytes(self.leading_bytes)
    }
}

/// `VHDXLogSequence`: a run of valid log entries.
#[derive(Clone, Copy, Debug, Default)]
struct LogSequence {
    valid: bool,
    count: u32,
    log: LogEntries,
    hdr: LogEntryHeader,
}

/// `vhdx_log_peek_hdr()`: the header of the entry at the read position, which stays put.
fn peek_hdr(file: &Node, log: &LogEntries) -> io::Result<LogEntryHeader> {
    // Peeking only works on sector boundaries.
    if log.read % LOG_SECTOR_SIZE != 0 {
        return Err(errno(libc::EFAULT));
    }
    let mut read = log.read;
    // Log sectors are 4 KiB and the log length is a multiple of 1 MiB, so there is always a
    // whole number of sectors in the buffer.
    if read as u64 + LOG_HDR_SIZE as u64 > log.length {
        read = 0;
    }
    if read == log.write {
        return Err(einval());
    }
    let mut b = [0u8; LOG_HDR_SIZE];
    file.pread(log.offset + read as u64, &mut b)?;
    Ok(LogEntryHeader::parse(&b))
}

/// `vhdx_log_inc_idx()`: the next sector, wrapping at the end of the log.
fn inc_idx(idx: u32, length: u64) -> u32 {
    let idx = idx.wrapping_add(LOG_SECTOR_SIZE);
    if idx as u64 >= length { 0 } else { idx }
}

/// `vhdx_log_reset()`: empties the log. A zero log guid tells any v0 log parser that there is
/// no log.
fn reset(file: &Node, s: &mut State) {
    s.log.read = 0;
    s.log.write = 0;
    // QEMU ignores failures here too.
    let _ = update_headers(file, s, false, Some(&ZERO_GUID));
}

/// `vhdx_log_read_sectors()`: reads up to `buf.len() / 4096` sectors from the read position,
/// stopping early when the log is empty. Returns how many sectors it read. Without `peek` the
/// read position moves past them.
fn read_sectors(file: &Node, log: &mut LogEntries, buf: &mut [u8], peek: bool) -> io::Result<u32> {
    let mut read = log.read;
    let mut sectors_read = 0u32;
    let mut ret = Ok(());
    for sector in buf.chunks_exact_mut(LOG_SECTOR_SIZE as usize) {
        if read == log.write {
            // Empty.
            break;
        }
        if let Err(e) = file.pread(log.offset + read as u64, sector) {
            ret = Err(e);
            break;
        }
        read = inc_idx(read, log.length);
        sectors_read += 1;
    }
    if !peek {
        log.read = read;
    }
    ret.map(|()| sectors_read)
}

/// `vhdx_log_write_sectors()`: writes the sectors of `buf` at the write position, stopping
/// early if the log is full. Returns how many sectors it wrote.
fn write_sectors(file: &Node, s: &mut State, buf: &[u8]) -> io::Result<u32> {
    user_visible_write(file, s)?;
    let mut sectors_written = 0u32;
    let mut write = s.log.write;
    for sector in buf.chunks_exact(LOG_SECTOR_SIZE as usize) {
        let offset = s.log.offset + write as u64;
        write = inc_idx(write, s.log.length);
        if write == s.log.read {
            // Full.
            break;
        }
        file.pwrite(offset, sector)?;
        s.log.write = write;
        sectors_written += 1;
    }
    Ok(sectors_written)
}

/// `vhdx_log_hdr_is_valid()`.
fn hdr_is_valid(log: &LogEntries, hdr: &LogEntryHeader, active_log_guid: &Guid) -> bool {
    hdr.signature == LOG_SIGNATURE
        // An entry larger than the whole log is obviously invalid.
        && log.length >= hdr.entry_length as u64
        // The entry is made of whole log sectors.
        && hdr.entry_length % LOG_SECTOR_SIZE == 0
        // The spec says sequence numbers start at 1.
        && hdr.sequence_number != 0
        // Entries only count if they match the log guid in the active header.
        && hdr.log_guid == *active_log_guid
        && hdr.descriptor_count as u64 * LOG_DESC_SIZE as u64 <= hdr.entry_length as u64
}

/// `vhdx_log_desc_is_valid()`: the sequence number matches the entry, the file offset is
/// sector aligned and the signature is one of the two kinds.
fn desc_is_valid(desc: &LogDescriptor, hdr: &LogEntryHeader) -> bool {
    if desc.sequence_number != hdr.sequence_number || desc.file_offset % LOG_SECTOR_SIZE as u64 != 0
    {
        return false;
    }
    match desc.signature {
        LOG_ZERO_SIGNATURE => desc.zero_length() % LOG_SECTOR_SIZE as u64 == 0,
        LOG_DESC_SIGNATURE => true,
        _ => false,
    }
}

/// `vhdx_compute_desc_sectors()`: the sectors that hold the header and `desc_cnt`
/// descriptors. Never 0, even for no descriptors.
fn compute_desc_sectors(desc_cnt: u32) -> u32 {
    // The header takes the room of two descriptors.
    desc_cnt.wrapping_add(2).div_ceil(128)
}

/// `vhdx_log_read_desc()`: reads the header and the descriptor sectors of the entry at the
/// read position and checks every descriptor. Returns the header, the raw sectors and the
/// descriptors.
fn read_desc(
    file: &Node,
    active_log_guid: &Guid,
    log: &mut LogEntries,
) -> io::Result<(LogEntryHeader, Vec<u8>, Vec<LogDescriptor>)> {
    let hdr = peek_hdr(file, log)?;
    if !hdr_is_valid(log, &hdr, active_log_guid) {
        return Err(einval());
    }
    let desc_sectors = compute_desc_sectors(hdr.descriptor_count);
    let mut buffer = Vec::new();
    let len = desc_sectors as usize * LOG_SECTOR_SIZE as usize;
    if buffer.try_reserve_exact(len).is_err() {
        return Err(errno(libc::ENOMEM));
    }
    buffer.resize(len, 0);
    let sectors_read = read_sectors(file, log, &mut buffer, false)?;
    if sectors_read != desc_sectors {
        return Err(einval());
    }
    let mut descs = Vec::with_capacity(hdr.descriptor_count as usize);
    for i in 0..hdr.descriptor_count as usize {
        let off = LOG_HDR_SIZE + i * LOG_DESC_SIZE;
        let desc = LogDescriptor::parse(&buffer[off..off + LOG_DESC_SIZE]);
        if !desc_is_valid(&desc, &hdr) {
            return Err(einval());
        }
        descs.push(desc);
    }
    Ok((hdr, buffer, descs))
}

/// `vhdx_log_flush_desc()`: writes what `desc` describes to the image. A data descriptor
/// needs its data sector in `data`; a zero descriptor writes real zeroes, it does not leave
/// a hole.
fn flush_desc(file: &Node, desc: &LogDescriptor, data: Option<&[u8]>) -> io::Result<()> {
    let mut buffer = vec![0u8; LOG_SECTOR_SIZE as usize];
    let mut count = 1;
    match desc.signature {
        LOG_DESC_SIGNATURE => {
            let Some(data) = data else { return Err(errno(libc::EFAULT)) };
            // The data sector must have the sequence number of the descriptor.
            let seq = ((le32(data, 4) as u64) << 32) | le32(data, 4092) as u64;
            if seq != desc.sequence_number {
                return Err(einval());
            }
            // The first 8 and the last 4 bytes of the sector are in the descriptor.
            buffer[..8].copy_from_slice(&desc.leading_bytes);
            buffer[8..4092].copy_from_slice(&data[8..4092]);
            buffer[4092..].copy_from_slice(&desc.trailing_bytes);
        }
        LOG_ZERO_SIGNATURE => count = desc.zero_length() / LOG_SECTOR_SIZE as u64,
        sig => {
            ruvm_base::report::error_report(&format!(
                "Invalid VHDX log descriptor entry signature 0x{sig:x}"
            ));
            return Err(einval());
        }
    }
    let mut file_offset = desc.file_offset;
    for _ in 0..count {
        file.pwrite(file_offset, &buffer)?;
        file.flush()?;
        file_offset += LOG_SECTOR_SIZE as u64;
    }
    Ok(())
}

/// `vhdx_log_flush()`: writes the entries of `logs`, which must have been validated, to the
/// image and then empties the log.
fn flush(file: &Node, s: &mut State, logs: &mut LogSequence) -> io::Result<()> {
    let mut data = vec![0u8; LOG_SECTOR_SIZE as usize];
    let mut have_data = false;
    user_visible_write(file, s)?;

    // Each round is one entry, which can span several sectors.
    for _ in 0..logs.count {
        let hdr_tmp = peek_hdr(file, &logs.log)?;
        let file_length = file.getlength()?;
        // A flushed file offset past the end of the file means the file has been truncated
        // or is corrupt, and must not be used.
        if hdr_tmp.flushed_file_offset > file_length {
            return Err(einval());
        }
        let log_guid = s.header().log_guid;
        let (hdr, _, descs) = read_desc(file, &log_guid, &mut logs.log)?;
        for desc in &descs {
            if desc.signature == LOG_DESC_SIGNATURE {
                // A data descriptor, so read the sector to write.
                if read_sectors(file, &mut logs.log, &mut data, false)? != 1 {
                    return Err(einval());
                }
                have_data = true;
            }
            flush_desc(file, desc, have_data.then_some(&data[..]))?;
        }
        if file_length < hdr.last_file_offset {
            let mut new_file_size = hdr.last_file_offset;
            if new_file_size % MIB != 0 {
                // Round up to the next 1 MiB boundary.
                new_file_size = new_file_size.div_ceil(MIB) * MIB;
                if new_file_size > i64::MAX as u64 {
                    return Err(einval());
                }
                file.truncate_full(new_file_size as i64, false, PreallocMode::Off, 0)
                    .map_err(to_io)?;
            }
        }
    }

    file.flush()?;
    // The log is fully flushed, so mark it empty. That sets the log guid to 0, too.
    reset(file, s);
    Ok(())
}

/// `vhdx_validate_log_entry()`: checks the entry at the read position, and that its sequence
/// number follows `seq` unless that is 0. Returns the header of a valid entry.
///
/// The read position moves past the entry, or by one sector if there is no entry there.
fn validate_log_entry(
    file: &Node,
    active_log_guid: &Guid,
    log: &mut LogEntries,
    seq: u64,
) -> io::Result<Option<LogEntryHeader>> {
    let inc = |log: &mut LogEntries| log.read = inc_idx(log.read, log.length);
    let hdr = match peek_hdr(file, log) {
        Ok(h) => h,
        Err(e) => {
            inc(log);
            return Err(e);
        }
    };
    if !hdr_is_valid(log, &hdr, active_log_guid) || (seq > 0 && hdr.sequence_number != seq + 1) {
        inc(log);
        return Ok(None);
    }

    let desc_sectors = compute_desc_sectors(hdr.descriptor_count);
    // Read all the sectors of the entry, for the checksum.
    let total_sectors = hdr.entry_length / LOG_SECTOR_SIZE;

    // This moves the read position past the descriptors.
    let (_, desc_buffer, _) = read_desc(file, active_log_guid, log)?;
    let mut crc = checksum_calc(0xffff_ffff, &desc_buffer, Some(4)) ^ 0xffff_ffff;

    let mut buffer = vec![0u8; LOG_SECTOR_SIZE as usize];
    if total_sectors > desc_sectors {
        for _ in 0..total_sectors - desc_sectors {
            if read_sectors(file, log, &mut buffer, false)? != 1 {
                return Ok(None);
            }
            crc = checksum_calc(crc, &buffer, None) ^ 0xffff_ffff;
        }
    }
    crc ^= 0xffff_ffff;
    if crc != hdr.checksum {
        return Ok(None);
    }
    Ok(Some(hdr))
}

/// `vhdx_log_search()`: goes through the whole log, sector by sector, for the newest run of
/// valid entries.
fn search(file: &Node, s: &mut State) -> io::Result<LogSequence> {
    let log_guid = s.header().log_guid;
    let mut candidate = LogSequence::default();
    let mut curr_log = s.log;
    // Assume the log is full.
    curr_log.write = curr_log.length as u32;
    curr_log.read = 0;

    loop {
        let mut curr_seq = 0u64;
        let mut current = LogSequence::default();
        let tail = curr_log.read;

        if let Some(hdr) = validate_log_entry(file, &log_guid, &mut curr_log, curr_seq)? {
            current.valid = true;
            current.log = curr_log;
            current.log.read = tail;
            current.log.write = curr_log.read;
            current.count = 1;
            current.hdr = hdr;

            while let Some(hdr) = validate_log_entry(file, &log_guid, &mut curr_log, curr_seq)? {
                current.log.write = curr_log.read;
                current.count += 1;
                curr_seq = hdr.sequence_number;
            }
        }

        if current.valid
            && (!candidate.valid || current.hdr.sequence_number > candidate.hdr.sequence_number)
        {
            candidate = current;
        }

        if curr_log.read < tail {
            break;
        }
    }

    if candidate.valid {
        // The next sequence number, for writes.
        s.log.sequence = candidate.hdr.sequence_number + 1;
    }
    Ok(candidate)
}

/// `vhdx_parse_log()`: the spec says a log must be replayed before the image is used, even
/// read-only. Returns whether it replayed one.
pub(super) fn parse_log(file: &Node, s: &mut State, read_only: bool) -> Result<bool, OpenError> {
    let hdr = *s.header();
    s.log.offset = hdr.log_offset;
    s.log.length = hdr.log_length as u64;

    if s.log.offset < LOG_MIN_SIZE || s.log.offset % LOG_MIN_SIZE != 0 {
        return Err(einval().into());
    }
    // The spec only knows log version 0.
    if hdr.log_version != 0 {
        return Err(einval().into());
    }
    // A zero log guid or log length means there is no log to replay.
    if hdr.log_guid == ZERO_GUID || hdr.log_length == 0 {
        return Ok(false);
    }
    if hdr.log_length as u64 % LOG_MIN_SIZE != 0 {
        return Err(einval().into());
    }

    // There is a log: look for an active run of valid entries in it.
    let mut logs = search(file, s)?;
    if !logs.valid {
        return Ok(false);
    }
    if read_only {
        let filename = file.filename().unwrap_or_default();
        return Err(Error::generic(format!(
            "VHDX image file '{filename}' opened read-only, but contains a log that needs to \
             be replayed"
        ))
        .hint(format!("To replay the log, run:\nqemu-img check -r all '{filename}'\n"))
        .into());
    }
    flush(file, s, &mut logs)?;
    Ok(true)
}

/// `vhdx_log_raw_to_le_sector()`: splits the 4 KiB at `data` between `desc` and the data
/// sector `sector`.
fn raw_to_le_sector(desc: &mut LogDescriptor, sector: &mut [u8], data: &[u8], seq: u64) {
    desc.leading_bytes.copy_from_slice(&data[..8]);
    sector[8..4092].copy_from_slice(&data[8..4092]);
    desc.trailing_bytes.copy_from_slice(&data[4092..4096]);
    put32(sector, 0, LOG_DATA_SIGNATURE);
    put32(sector, 4, (seq >> 32) as u32);
    put32(sector, 4092, seq as u32);
}

/// `vhdx_log_write()`: puts `data`, which goes to `offset` in the image, into a new log entry.
fn log_write(file: &Node, s: &mut State, data: &[u8], offset: u64) -> io::Result<()> {
    let length = data.len() as u32;
    let header = *s.header();
    if length > header.log_length {
        // There is no log. One could be made here instead of failing.
        return Err(einval());
    }
    if header.log_guid == ZERO_GUID {
        let new_guid = guid_generate()?;
        let _ = update_headers(file, s, false, Some(&new_guid));
    } else {
        // For now the log must be flushed after every write.
        return Err(errno(libc::ENOTSUP));
    }

    // 0 is not a valid sequence number, but it is also what a first write or a wrapped
    // sequence number sees.
    if s.log.sequence == 0 {
        s.log.sequence = 1;
    }

    let sector_offset = (offset % LOG_SECTOR_SIZE as u64) as u32;
    let mut file_offset = offset - sector_offset as u64;

    let mut aligned_length = length;
    let mut leading_length = 0;
    let mut partial_sectors = 0;
    // Count the unaligned head and tail.
    if sector_offset != 0 {
        leading_length = (LOG_SECTOR_SIZE - sector_offset).min(length);
        aligned_length -= leading_length;
        partial_sectors += 1;
    }
    let mut sectors = aligned_length / LOG_SECTOR_SIZE;
    let trailing_length = aligned_length - sectors * LOG_SECTOR_SIZE;
    if trailing_length != 0 {
        partial_sectors += 1;
    }
    // The sectors of the data, without the header and the descriptors.
    sectors += partial_sectors;

    let file_length = file.getlength()?;

    let header = *s.header();
    let mut new_hdr = LogEntryHeader {
        signature: LOG_SIGNATURE,
        tail: s.log.tail,
        sequence_number: s.log.sequence,
        descriptor_count: sectors,
        flushed_file_offset: file_length,
        last_file_offset: file_length,
        log_guid: header.log_guid,
        ..Default::default()
    };
    let desc_sectors = compute_desc_sectors(new_hdr.descriptor_count);
    let total_length = (desc_sectors + sectors) * LOG_SECTOR_SIZE;
    new_hdr.entry_length = total_length;

    let mut buffer = vec![0u8; total_length as usize];
    new_hdr.write(&mut buffer);

    // Log sectors are all 4 KiB, so partial sectors are merged with what the file already has
    // at the destination.
    let mut merged_sector = vec![0u8; LOG_SECTOR_SIZE as usize];
    let data_start = desc_sectors as usize * LOG_SECTOR_SIZE as usize;
    let mut data_pos = 0usize;
    for i in 0..sectors {
        let mut new_desc = LogDescriptor {
            signature: LOG_DESC_SIGNATURE,
            sequence_number: s.log.sequence,
            file_offset,
            ..Default::default()
        };
        let bytes_written;
        let sector_write: &[u8] = if i == 0 && leading_length != 0 {
            // A partial sector at the front.
            file.pread(file_offset, &mut merged_sector)?;
            let so = sector_offset as usize;
            let ll = leading_length as usize;
            merged_sector[so..so + ll].copy_from_slice(&data[data_pos..data_pos + ll]);
            bytes_written = ll;
            &merged_sector
        } else if i == sectors - 1 && trailing_length != 0 {
            // A partial sector at the end.
            let tl = trailing_length as usize;
            file.pread(file_offset + tl as u64, &mut merged_sector[tl..])?;
            merged_sector[..tl].copy_from_slice(&data[data_pos..data_pos + tl]);
            bytes_written = tl;
            &merged_sector
        } else {
            bytes_written = LOG_SECTOR_SIZE as usize;
            &data[data_pos..data_pos + LOG_SECTOR_SIZE as usize]
        };

        let so = data_start + i as usize * LOG_SECTOR_SIZE as usize;
        let mut sector = [0u8; LOG_SECTOR_SIZE as usize];
        raw_to_le_sector(&mut new_desc, &mut sector, sector_write, s.log.sequence);
        buffer[so..so + LOG_SECTOR_SIZE as usize].copy_from_slice(&sector);
        let doff = LOG_HDR_SIZE + i as usize * LOG_DESC_SIZE;
        new_desc.write(&mut buffer[doff..doff + LOG_DESC_SIZE]);

        data_pos += bytes_written;
        file_offset += LOG_SECTOR_SIZE as u64;
    }

    // The checksum covers the whole entry, from the header to the last data sector.
    update_checksum(&mut buffer, 4);

    let sectors_written = write_sectors(file, s, &buffer)?;
    if sectors_written != desc_sectors + sectors {
        // The log could be flushed here instead of failing.
        return Err(einval());
    }

    s.log.sequence += 1;
    // The new tail.
    s.log.tail = s.log.write;
    Ok(())
}

/// `vhdx_log_write_and_flush()`: writes a log entry and flushes the log to the image right
/// away.
pub(super) fn write_and_flush(
    file: &Node,
    s: &mut State,
    data: &[u8],
    offset: u64,
) -> io::Result<()> {
    // New and changed blocks must be stable before the log entry exists.
    file.flush()?;
    log_write(file, s, data, offset)?;
    let mut logs = LogSequence { valid: true, count: 1, log: s.log, ..Default::default() };
    // The log must be stable too.
    file.flush()?;
    flush(file, s, &mut logs)?;
    s.log = logs.log;
    Ok(())
}
