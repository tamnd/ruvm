// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vhdx` format driver, a port of block/vhdx.c, block/vhdx-endian.c and block/vhdx.h.
//!
//! VHDX is Microsoft's successor to VHD. A file starts with a 1 MiB header section that holds
//! the file identifier, two copies of the image header and two copies of the region table.
//! The region table points at the block allocation table (BAT) and the metadata region; the
//! header points at the log. Every other chunk of the BAT is interleaved with an entry for a
//! sector bitmap block, which only differencing images use. Metadata updates (for us, BAT
//! entries) go through the log first, see [`log`].
//!
//! Like QEMU, the driver supports the dynamic and fixed subformats with 512 byte logical
//! sectors. Differencing images are refused when they are opened: a parent locator gives
//! `ENOTSUP`, `has_parent` without one gives `EINVAL`.
//!
//! Differences from QEMU:
//!
//! - ruvm has no `BDRV_O_CHECK` open flag, so the BAT entries are always checked when an image
//!   is opened. An image with broken BAT entries cannot be opened for `qemu-img check`; QEMU
//!   opens it and counts the entries as corruptions.
//! - There is no migration blocker, because the block layer has no migration to block.
//! - QEMU's `vhdx_log_read_sectors()` reads every sector into the start of the buffer, so the
//!   descriptors of a log entry with more than 126 of them are read from the wrong place and
//!   QEMU rejects the entry, which fails the open. Here every sector goes where it belongs.
//! - Requests hold the driver lock across their I/O on the file; QEMU drops its coroutine
//!   lock around the data reads and writes.
//! - The creator field names ruvm's QEMU version, `QEMU v11.1.0`.
//! - Where QEMU passes an errno up from a failed truncation of the file, a guest request here
//!   gets an error that carries the truncation's message instead.

mod log;

use std::io;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result, report};
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::types::{
    BlockdevCreateOptionsU, BlockdevCreateOptionsVhdx, BlockdevOptionsU, BlockdevVhdxSubformat,
    PreallocMode,
};
use ruvm_qapi::visit::{parse_option_size, qapi_bool_parse};
use ruvm_qapi::{QDict, QValue};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::node::{
    BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BDRV_REQ_ZERO_WRITE, BDRV_SECTOR_SIZE, BlockDriverInfo,
    BlockLimits, CheckResult, Driver, Node, ReopenState, errno,
};

use log::LogEntries;

pub(crate) static VHDX: DriverDef = DriverDef::format("vhdx", open)
    .with_probe(probe)
    .with_create_opts(create_opts)
    .with_create(create)
    .with_create_opts_list(&CREATE_OPTS);

/// `vhdx_create_opts`.
static CREATE_OPTS: [QemuOptDesc; 5] = [
    QemuOptDesc::new("size", QemuOptType::Size).help("Virtual disk size; max of 64TB."),
    QemuOptDesc::new("log_size", QemuOptType::Size)
        .help("Log size; min 1MB.")
        .default_value("1048576"),
    QemuOptDesc::new("block_size", QemuOptType::Size)
        .help("Block Size; min 1MB, max 256MB. 0 means auto-calculate based on image size.")
        .default_value("0"),
    QemuOptDesc::new("subformat", QemuOptType::String)
        .help("VHDX format type, can be either 'dynamic' or 'fixed'. Default is 'dynamic'."),
    QemuOptDesc::new("block_state_zero", QemuOptType::Bool).help(
        "Force use of payload blocks of type 'ZERO'. Non-standard, but default.  Do not set to \
         'off' when using 'qemu-img convert' with subformat=dynamic.",
    ),
];

const KIB: u64 = 1024;
const MIB: u64 = 1024 * KIB;
const GIB: u64 = 1024 * MIB;
const TIB: u64 = 1024 * GIB;

const DEFAULT_LOG_SIZE: u32 = 1024 * 1024;

const HEADER_BLOCK_SIZE: u64 = 64 * KIB;
const FILE_ID_OFFSET: u64 = 0;
const HEADER1_OFFSET: u64 = HEADER_BLOCK_SIZE;
const HEADER2_OFFSET: u64 = HEADER_BLOCK_SIZE * 2;
const REGION_TABLE_OFFSET: u64 = HEADER_BLOCK_SIZE * 3;
const REGION_TABLE2_OFFSET: u64 = HEADER_BLOCK_SIZE * 4;
const HEADER_SECTION_END: u64 = MIB;

const FILE_SIGNATURE: u64 = 0x656C_6966_7864_6876;
const HEADER_SIGNATURE: u32 = 0x6461_6568;
const REGION_SIGNATURE: u32 = 0x6967_6572;
const METADATA_SIGNATURE: u64 = 0x6174_6164_6174_656D;

/// The header checksum covers this much, not just the 80 bytes of `VHDXHeader`.
const HEADER_SIZE: usize = 4096;
/// `sizeof(VHDXHeader)`.
const HEADER_STRUCT_SIZE: usize = 80;

const REGION_ENTRY_REQUIRED: u32 = 0x01;

const PAYLOAD_BLOCK_NOT_PRESENT: u64 = 0;
const PAYLOAD_BLOCK_UNDEFINED: u64 = 1;
const PAYLOAD_BLOCK_ZERO: u64 = 2;
const PAYLOAD_BLOCK_UNMAPPED: u64 = 3;
const PAYLOAD_BLOCK_UNMAPPED_V095: u64 = 5;
const PAYLOAD_BLOCK_FULLY_PRESENT: u64 = 6;

const MAX_SECTORS_PER_BLOCK: u64 = 1 << 23;
const BAT_STATE_BIT_MASK: u64 = 0x07;
const BAT_FILE_OFF_MASK: u64 = 0xFFFF_FFFF_FFF0_0000;

const METADATA_ENTRY_SIZE: usize = 32;
const METADATA_TABLE_MAX_SIZE: usize = METADATA_ENTRY_SIZE * 2048;
const METADATA_HEADER_SIZE: usize = 32;

const META_FLAGS_IS_VIRTUAL_DISK: u32 = 0x02;
const META_FLAGS_IS_REQUIRED: u32 = 0x04;

const PARAMS_LEAVE_BLOCKS_ALLOCED: u32 = 0x01;
const PARAMS_HAS_PARENT: u32 = 0x02;
const BLOCK_SIZE_MIN: u32 = MIB as u32;
const BLOCK_SIZE_MAX: u32 = 256 * MIB as u32;

const MAX_IMAGE_SIZE: u64 = 64 * TIB;

const META_FILE_PARAMETER_PRESENT: u32 = 0x01;
const META_VIRTUAL_DISK_SIZE_PRESENT: u32 = 0x02;
const META_PAGE_83_PRESENT: u32 = 0x04;
const META_LOGICAL_SECTOR_SIZE_PRESENT: u32 = 0x08;
const META_PHYS_SECTOR_SIZE_PRESENT: u32 = 0x10;
const META_PARENT_LOCATOR_PRESENT: u32 = 0x20;
const META_ALL_PRESENT: u32 = META_FILE_PARAMETER_PRESENT
    | META_VIRTUAL_DISK_SIZE_PRESENT
    | META_PAGE_83_PRESENT
    | META_LOGICAL_SECTOR_SIZE_PRESENT
    | META_PHYS_SECTOR_SIZE_PRESENT;

/// What `QEMU_VERSION` is for the creator field.
const QEMU_VERSION: &str = "11.1.0";

/// An `MSGUID` as it sits in the file: `data1`, `data2` and `data3` little endian, then the
/// eight bytes of `data4`.
type Guid = [u8; 16];

const ZERO_GUID: Guid = [0; 16];

const fn ms_guid(d1: u32, d2: u16, d3: u16, d4: [u8; 8]) -> Guid {
    let a = d1.to_le_bytes();
    let b = d2.to_le_bytes();
    let c = d3.to_le_bytes();
    [
        a[0], a[1], a[2], a[3], b[0], b[1], c[0], c[1], d4[0], d4[1], d4[2], d4[3], d4[4], d4[5],
        d4[6], d4[7],
    ]
}

/// 2DC27766-F623-4200-9D64-115E9BFD4A08
const BAT_GUID: Guid =
    ms_guid(0x2dc2_7766, 0xf623, 0x4200, [0x9d, 0x64, 0x11, 0x5e, 0x9b, 0xfd, 0x4a, 0x08]);
/// 8B7CA206-4790-4B9A-B8FE-575F050F886E
const METADATA_GUID: Guid =
    ms_guid(0x8b7c_a206, 0x4790, 0x4b9a, [0xb8, 0xfe, 0x57, 0x5f, 0x05, 0x0f, 0x88, 0x6e]);
/// CAA16737-FA36-4D43-B3B6-33F0AA44E76B
const FILE_PARAM_GUID: Guid =
    ms_guid(0xcaa1_6737, 0xfa36, 0x4d43, [0xb3, 0xb6, 0x33, 0xf0, 0xaa, 0x44, 0xe7, 0x6b]);
/// 2FA54224-CD1B-4876-B211-5DBED83BF4B8
const VIRTUAL_SIZE_GUID: Guid =
    ms_guid(0x2FA5_4224, 0xcd1b, 0x4876, [0xb2, 0x11, 0x5d, 0xbe, 0xd8, 0x3b, 0xf4, 0xb8]);
/// BECA12AB-B2E6-4523-93EF-C309E000C746
const PAGE83_GUID: Guid =
    ms_guid(0xbeca_12ab, 0xb2e6, 0x4523, [0x93, 0xef, 0xc3, 0x09, 0xe0, 0x00, 0xc7, 0x46]);
/// CDA348C7-445D-4471-9CC9-E9885251C556
const PHYS_SECTOR_GUID: Guid =
    ms_guid(0xcda3_48c7, 0x445d, 0x4471, [0x9c, 0xc9, 0xe9, 0x88, 0x52, 0x51, 0xc5, 0x56]);
/// A8D35F2D-B30B-454D-ABF7-D3D84834AB0C
const PARENT_LOCATOR_GUID: Guid =
    ms_guid(0xa8d3_5f2d, 0xb30b, 0x454d, [0xab, 0xf7, 0xd3, 0xd8, 0x48, 0x34, 0xab, 0x0c]);
/// 8141BF1D-A96F-4709-BA47-F233A8FAAB5F
const LOGICAL_SECTOR_GUID: Guid =
    ms_guid(0x8141_bf1d, 0xa96f, 0x4709, [0xba, 0x47, 0xf2, 0x33, 0xa8, 0xfa, 0xab, 0x5f]);

// Little endian field access, the job of vhdx-endian.c.

fn le16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes(b[off..off + 2].try_into().unwrap())
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn guid_at(b: &[u8], off: usize) -> Guid {
    b[off..off + 16].try_into().unwrap()
}

fn put16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// The CRC-32C (Castagnoli) table, for the reflected polynomial 0x82F63B78.
static CRC32C_TABLE: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// `crc32c()` from util/crc32c.c: the caller passes the starting value, and the result is
/// inverted. Chaining calls means inverting the result again before passing it back in.
fn crc32c(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc = CRC32C_TABLE[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

/// `vhdx_update_checksum()`: zeroes the checksum field at `crc_offset`, then stores the CRC of
/// `buf` there.
fn update_checksum(buf: &mut [u8], crc_offset: usize) -> u32 {
    put32(buf, crc_offset, 0);
    let crc = crc32c(0xffff_ffff, buf);
    put32(buf, crc_offset, crc);
    crc
}

/// `vhdx_checksum_calc()`: the CRC of `buf` with the checksum field at `crc_offset` read as
/// zero. `None` means there is no checksum field in `buf`.
fn checksum_calc(crc: u32, buf: &[u8], crc_offset: Option<usize>) -> u32 {
    match crc_offset {
        Some(off) => {
            let crc = crc32c(crc, &buf[..off]) ^ 0xffff_ffff;
            let crc = crc32c(crc, &[0; 4]) ^ 0xffff_ffff;
            crc32c(crc, &buf[off + 4..])
        }
        None => crc32c(crc, buf),
    }
}

/// `vhdx_checksum_is_valid()`.
fn checksum_is_valid(buf: &[u8], crc_offset: usize) -> bool {
    checksum_calc(0xffff_ffff, buf, Some(crc_offset)) == le32(buf, crc_offset)
}

/// `vhdx_guid_generate()`: a version 4 UUID, as `qemu_uuid_generate()` makes it, used as an
/// `MSGUID`.
fn guid_generate() -> io::Result<Guid> {
    let mut g = [0u8; 16];
    ruvm_crypto::random::random_bytes(&mut g).map_err(|e| io::Error::other(e.message()))?;
    // time_high_and_version is a host order uint16_t in QemuUUID.
    g[7] = (g[7] & 0x0f) | 0x40;
    g[8] = (g[8] & 0x3f) | 0x80;
    Ok(g)
}

/// `g_random_int()`.
fn random_u32() -> io::Result<u32> {
    let mut b = [0u8; 4];
    ruvm_crypto::random::random_bytes(&mut b).map_err(|e| io::Error::other(e.message()))?;
    Ok(u32::from_le_bytes(b))
}

/// `VHDXHeader`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Header {
    signature: u32,
    checksum: u32,
    sequence_number: u64,
    file_write_guid: Guid,
    data_write_guid: Guid,
    log_guid: Guid,
    log_version: u16,
    version: u16,
    log_length: u32,
    log_offset: u64,
}

impl Header {
    /// `vhdx_header_le_import()`.
    fn parse(b: &[u8]) -> Header {
        Header {
            signature: le32(b, 0),
            checksum: le32(b, 4),
            sequence_number: le64(b, 8),
            file_write_guid: guid_at(b, 16),
            data_write_guid: guid_at(b, 32),
            log_guid: guid_at(b, 48),
            log_version: le16(b, 64),
            version: le16(b, 66),
            log_length: le32(b, 68),
            log_offset: le64(b, 72),
        }
    }

    /// `vhdx_header_le_export()`.
    fn write(&self, b: &mut [u8]) {
        put32(b, 0, self.signature);
        put32(b, 4, self.checksum);
        put64(b, 8, self.sequence_number);
        b[16..32].copy_from_slice(&self.file_write_guid);
        b[32..48].copy_from_slice(&self.data_write_guid);
        b[48..64].copy_from_slice(&self.log_guid);
        put16(b, 64, self.log_version);
        put16(b, 66, self.version);
        put32(b, 68, self.log_length);
        put64(b, 72, self.log_offset);
    }
}

/// `VHDXRegionTableEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct RegionTableEntry {
    guid: Guid,
    file_offset: u64,
    length: u32,
    data_bits: u32,
}

impl RegionTableEntry {
    fn parse(b: &[u8]) -> RegionTableEntry {
        RegionTableEntry {
            guid: guid_at(b, 0),
            file_offset: le64(b, 16),
            length: le32(b, 24),
            data_bits: le32(b, 28),
        }
    }

    fn write(&self, b: &mut [u8]) {
        b[..16].copy_from_slice(&self.guid);
        put64(b, 16, self.file_offset);
        put32(b, 24, self.length);
        put32(b, 28, self.data_bits);
    }
}

/// `VHDXMetadataTableEntry`.
#[derive(Clone, Copy, Debug, Default)]
struct MetadataTableEntry {
    item_id: Guid,
    offset: u32,
    length: u32,
    data_bits: u32,
}

impl MetadataTableEntry {
    fn parse(b: &[u8]) -> MetadataTableEntry {
        MetadataTableEntry {
            item_id: guid_at(b, 0),
            offset: le32(b, 16),
            length: le32(b, 20),
            data_bits: le32(b, 24),
        }
    }

    fn write(&self, b: &mut [u8]) {
        b[..16].copy_from_slice(&self.item_id);
        put32(b, 16, self.offset);
        put32(b, 20, self.length);
        put32(b, 24, self.data_bits);
    }
}

/// `VHDXRegionEntry`: a range of the file some structure uses, for overlap checks.
#[derive(Clone, Copy, Debug)]
struct Region {
    start: u64,
    end: u64,
}

/// The registered regions, newest first like QEMU's `QLIST_INSERT_HEAD()` list.
#[derive(Clone, Debug, Default)]
struct Regions(Vec<Region>);

impl Regions {
    /// `vhdx_region_register()`.
    fn register(&mut self, start: u64, length: u64) {
        self.0.insert(0, Region { start, end: start.wrapping_add(length) });
    }

    /// `vhdx_region_check()`.
    fn check(&self, start: u64, length: u64) -> io::Result<()> {
        let end = start.wrapping_add(length);
        for r in &self.0 {
            if !(start >= r.end || end <= r.start) {
                // QEMU prints the last number with "%.lu", a precision of zero, which prints
                // nothing at all for zero.
                let r_end = if r.end == 0 { String::new() } else { r.end.to_string() };
                report::error_report(&format!(
                    "VHDX region {start}-{end} overlaps with region {}-{r_end}",
                    r.start
                ));
                return Err(errno(libc::EINVAL));
            }
        }
        Ok(())
    }
}

/// The parts of `BDRVVHDXState` that say how the BAT maps the virtual disk. They do not
/// change once the image is open.
#[derive(Clone, Copy, Debug, Default)]
struct Geometry {
    block_size: u32,
    sectors_per_block: u32,
    sectors_per_block_bits: u32,
    chunk_ratio: u64,
    chunk_ratio_bits: u32,
    logical_sector_size_bits: u32,
    bat_entries: u32,
    bat_offset: u64,
}

impl Geometry {
    /// `vhdx_set_shift_bits()` and `vhdx_calc_bat_entries()` for a disk without a parent.
    fn new(block_size: u32, logical_sector_size: u32, virtual_disk_size: u64) -> Geometry {
        let sectors_per_block = block_size / logical_sector_size;
        let chunk_ratio = MAX_SECTORS_PER_BLOCK * logical_sector_size as u64 / block_size as u64;
        let chunk_ratio_bits = chunk_ratio.trailing_zeros();
        // These are uint32_t in QEMU and wrap the same way, for an empty disk in particular.
        let data_blocks_cnt = virtual_disk_size.div_ceil(block_size as u64) as u32;
        let bat_entries =
            data_blocks_cnt.wrapping_add(data_blocks_cnt.wrapping_sub(1) >> chunk_ratio_bits);
        Geometry {
            block_size,
            sectors_per_block,
            sectors_per_block_bits: sectors_per_block.trailing_zeros(),
            chunk_ratio,
            chunk_ratio_bits,
            logical_sector_size_bits: logical_sector_size.trailing_zeros(),
            bat_entries,
            bat_offset: 0,
        }
    }

    /// `vhdx_block_translate()`.
    fn translate(&self, bat: &[u64], sector_num: u64, nb_sectors: u64) -> SectorInfo {
        let mut bat_idx = sector_num >> self.sectors_per_block_bits;
        let block_offset = (sector_num - (bat_idx << self.sectors_per_block_bits)) as u32;
        bat_idx += bat_idx >> self.chunk_ratio_bits;
        let mut sectors_avail = (self.sectors_per_block - block_offset) as u64;
        if sectors_avail > nb_sectors {
            sectors_avail = nb_sectors;
        }
        let bytes_avail = sectors_avail << self.logical_sector_size_bits;
        let entry = bat.get(bat_idx as usize).copied().unwrap_or(0);
        let block_offset = (block_offset as u64) << self.logical_sector_size_bits;
        let mut file_offset = entry & BAT_FILE_OFF_MASK;
        // The file offset must be past the header section, so a zero offset means none.
        if file_offset != 0 {
            file_offset += block_offset;
        }
        SectorInfo {
            bat_idx: bat_idx as usize,
            sectors_avail,
            bytes_avail,
            file_offset,
            block_offset,
        }
    }

    /// `vhdx_update_bat_table_entry()`: sets the BAT entry of `sinfo` and returns the entry as
    /// it goes to the file with the offset of the entry in the file.
    fn update_bat_table_entry(
        &self,
        bat: &mut [u64],
        sinfo: &SectorInfo,
        state: u64,
    ) -> (u64, u64) {
        let e = match state {
            // For a ZERO block the offset field is reserved in the v1.0 spec, and Hyper-V
            // fails to read the image if it is not zero.
            PAYLOAD_BLOCK_ZERO
            | PAYLOAD_BLOCK_UNDEFINED
            | PAYLOAD_BLOCK_NOT_PRESENT
            | PAYLOAD_BLOCK_UNMAPPED => 0,
            _ => sinfo.file_offset,
        } | (state & BAT_STATE_BIT_MASK);
        bat[sinfo.bat_idx] = e;
        (e, self.bat_offset + sinfo.bat_idx as u64 * 8)
    }
}

/// `VHDXSectorInfo`.
#[derive(Clone, Copy, Debug, Default)]
struct SectorInfo {
    bat_idx: usize,
    sectors_avail: u64,
    bytes_avail: u64,
    file_offset: u64,
    block_offset: u64,
}

/// The mutable part of `BDRVVHDXState`, behind the driver's lock.
struct State {
    headers: [Header; 2],
    curr_header: usize,
    session_guid: Guid,
    first_visible_write: bool,
    log: LogEntries,
    bat: Vec<u64>,
}

impl State {
    fn header(&self) -> &Header {
        &self.headers[self.curr_header]
    }
}

/// `vhdx_write_header()`: writes `hdr` at `offset`. The checksum covers the 4 KiB header
/// area, so with `read` the rest of that area comes from the file, otherwise it is zeroes.
fn write_header(file: &Node, hdr: &Header, offset: u64, read: bool) -> io::Result<()> {
    let mut buffer = vec![0u8; HEADER_SIZE];
    if read {
        file.pread(offset, &mut buffer)?;
    }
    hdr.write(&mut buffer);
    update_checksum(&mut buffer, 4);
    file.pwrite(offset, &buffer[..HEADER_STRUCT_SIZE])?;
    file.flush()
}

/// `vhdx_update_header()`: writes the inactive header with the next sequence number, which
/// makes it the active one.
fn update_header(
    file: &Node,
    s: &mut State,
    generate_data_write_guid: bool,
    log_guid: Option<&Guid>,
) -> io::Result<()> {
    let (hdr_idx, header_offset) =
        if s.curr_header == 0 { (1, HEADER2_OFFSET) } else { (0, HEADER1_OFFSET) };
    let seq = s.headers[s.curr_header].sequence_number.wrapping_add(1);
    let session_guid = s.session_guid;
    let inactive = &mut s.headers[hdr_idx];
    inactive.sequence_number = seq;
    // A new file guid must be generated before any file write, including headers.
    inactive.file_write_guid = session_guid;
    // A new data guid only needs to be generated before guest visible writes.
    if generate_data_write_guid {
        inactive.data_write_guid = guid_generate()?;
    }
    if let Some(g) = log_guid {
        inactive.log_guid = *g;
    }
    write_header(file, &s.headers[hdr_idx], header_offset, true)?;
    s.curr_header = hdr_idx;
    Ok(())
}

/// `vhdx_update_headers()`: the spec wants both headers updated, so that both are valid.
fn update_headers(
    file: &Node,
    s: &mut State,
    generate_data_write_guid: bool,
    log_guid: Option<&Guid>,
) -> io::Result<()> {
    update_header(file, s, generate_data_write_guid, log_guid)?;
    update_header(file, s, generate_data_write_guid, log_guid)
}

/// `vhdx_user_visible_write()`: the first guest visible write gets a new data write guid.
fn user_visible_write(file: &Node, s: &mut State) -> io::Result<()> {
    if s.first_visible_write {
        s.first_visible_write = false;
        update_headers(file, s, true, None)?;
    }
    Ok(())
}

/// `vhdx_probe()`.
fn probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() >= 8 && &buf[..8] == b"vhdxfile" { 100 } else { 0 }
}

/// Why opening failed: a message of its own, or an errno that `bdrv_open_common()` turns into
/// "Could not open".
enum OpenError {
    Msg(Error),
    Errno(io::Error),
}

impl From<io::Error> for OpenError {
    fn from(e: io::Error) -> Self {
        OpenError::Errno(e)
    }
}

impl From<Error> for OpenError {
    fn from(e: Error) -> Self {
        OpenError::Msg(e)
    }
}

fn einval() -> io::Error {
    errno(libc::EINVAL)
}

fn enotsup() -> io::Error {
    errno(libc::ENOTSUP)
}

/// `vhdx_parse_header()`: reads both headers and picks the active one, then registers the
/// log region.
fn parse_header(file: &Node, s: &mut State, regions: &mut Regions) -> Result<()> {
    let fail = |e: Option<io::Error>| match e {
        Some(e) => Error::from_io("No valid VHDX header found", e),
        None => Error::generic("No valid VHDX header found"),
    };
    let mut buffer = vec![0u8; HEADER_SIZE];
    let mut valid = [false; 2];
    for (i, offset) in [HEADER1_OFFSET, HEADER2_OFFSET].into_iter().enumerate() {
        file.pread(offset, &mut buffer).map_err(|e| fail(Some(e)))?;
        s.headers[i] = Header::parse(&buffer);
        if checksum_is_valid(&buffer, 4) {
            let h = &s.headers[i];
            if h.signature == HEADER_SIGNATURE && h.version == 1 {
                valid[i] = true;
            }
        }
    }
    // With only one valid header, the sequence numbers do not matter.
    s.curr_header = match valid {
        [true, false] => 0,
        [false, true] => 1,
        [false, false] => return Err(fail(None)),
        [true, true] => {
            let (h1, h2) = (s.headers[0].sequence_number, s.headers[1].sequence_number);
            if h1 > h2 {
                0
            } else if h2 > h1 {
                1
            } else if s.headers[0] == s.headers[1] {
                // Microsoft's Disk2VHD writes two identical headers with the same sequence
                // number. That is not a corrupt file.
                0
            } else {
                return Err(fail(None));
            }
        }
    };
    let h = s.header();
    regions.register(h.log_offset, h.log_length as u64);
    Ok(())
}

/// `vhdx_open_region_tables()`: finds the BAT and metadata regions.
fn open_region_tables(
    file: &Node,
    regions: &mut Regions,
) -> io::Result<(RegionTableEntry, RegionTableEntry)> {
    let mut buffer = vec![0u8; HEADER_BLOCK_SIZE as usize];
    file.pread(REGION_TABLE_OFFSET, &mut buffer)?;
    if !checksum_is_valid(&buffer, 4) {
        return Err(einval());
    }
    if le32(&buffer, 0) != REGION_SIGNATURE {
        return Err(einval());
    }
    // The spec allows at most 2047 entries.
    let entry_count = le32(&buffer, 8);
    if entry_count > 2047 {
        return Err(einval());
    }
    let mut bat_rt = None;
    let mut metadata_rt = None;
    for i in 0..entry_count as usize {
        let e = RegionTableEntry::parse(&buffer[16 + i * 32..]);
        // The entries must not overlap each other or anything else in the file.
        regions.check(e.file_offset, e.length as u64)?;
        regions.register(e.file_offset, e.length as u64);
        if e.guid == BAT_GUID {
            if bat_rt.is_some() {
                return Err(einval());
            }
            bat_rt = Some(e);
            continue;
        }
        if e.guid == METADATA_GUID {
            if metadata_rt.is_some() {
                return Err(einval());
            }
            metadata_rt = Some(e);
            continue;
        }
        // A required entry we do not know: the spec says we must fail.
        if e.data_bits & REGION_ENTRY_REQUIRED != 0 {
            return Err(enotsup());
        }
    }
    match (bat_rt, metadata_rt) {
        (Some(b), Some(m)) => Ok((b, m)),
        _ => Err(einval()),
    }
}

/// What `vhdx_parse_metadata()` finds.
struct Metadata {
    block_size: u32,
    data_bits: u32,
    virtual_disk_size: u64,
    logical_sector_size: u32,
}

fn pread_u32(file: &Node, offset: u64) -> io::Result<u32> {
    let mut b = [0u8; 4];
    file.pread(offset, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn pread_u64(file: &Node, offset: u64) -> io::Result<u64> {
    let mut b = [0u8; 8];
    file.pread(offset, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// `vhdx_parse_metadata()`: reads the five required metadata items. A differencing image
/// also has a parent locator, which QEMU does not support yet.
fn parse_metadata(file: &Node, metadata_rt: &RegionTableEntry) -> io::Result<Metadata> {
    let mut buffer = vec![0u8; METADATA_TABLE_MAX_SIZE];
    file.pread(metadata_rt.file_offset, &mut buffer)?;
    if le64(&buffer, 0) != METADATA_SIGNATURE {
        return Err(einval());
    }
    let entry_count = le16(&buffer, 10) as usize;
    if entry_count * METADATA_ENTRY_SIZE > METADATA_TABLE_MAX_SIZE - METADATA_HEADER_SIZE {
        return Err(einval());
    }
    let items = [
        (FILE_PARAM_GUID, META_FILE_PARAMETER_PRESENT),
        (VIRTUAL_SIZE_GUID, META_VIRTUAL_DISK_SIZE_PRESENT),
        (PAGE83_GUID, META_PAGE_83_PRESENT),
        (LOGICAL_SECTOR_GUID, META_LOGICAL_SECTOR_SIZE_PRESENT),
        (PHYS_SECTOR_GUID, META_PHYS_SECTOR_SIZE_PRESENT),
        (PARENT_LOCATOR_GUID, META_PARENT_LOCATOR_PRESENT),
    ];
    let mut present = 0u32;
    let mut entries = [MetadataTableEntry::default(); 6];
    for i in 0..entry_count {
        let off = METADATA_HEADER_SIZE + i * METADATA_ENTRY_SIZE;
        let e = MetadataTableEntry::parse(&buffer[off..]);
        if let Some(k) = items.iter().position(|(g, _)| *g == e.item_id) {
            let bit = items[k].1;
            if present & bit != 0 {
                return Err(einval());
            }
            entries[k] = e;
            present |= bit;
            continue;
        }
        // A required item we do not know: the spec says we must fail.
        if e.data_bits & META_FLAGS_IS_REQUIRED != 0 {
            return Err(enotsup());
        }
    }
    // A parent locator makes this fail too, before has_parent is even looked at.
    if present != META_ALL_PRESENT {
        return Err(enotsup());
    }
    let base = metadata_rt.file_offset;
    let mut params = [0u8; 8];
    file.pread(entries[0].offset as u64 + base, &mut params)?;
    let block_size = le32(&params, 0);
    let data_bits = le32(&params, 4);

    // The parent locator is there if and only if has_parent is set.
    if data_bits & PARAMS_HAS_PARENT != 0 {
        if present & META_PARENT_LOCATOR_PRESENT != 0 {
            // Differencing files are not supported yet.
            return Err(enotsup());
        }
        return Err(einval());
    }

    let virtual_disk_size = pread_u64(file, entries[1].offset as u64 + base)?;
    let logical_sector_size = pread_u32(file, entries[3].offset as u64 + base)?;
    let _physical_sector_size = pread_u32(file, entries[4].offset as u64 + base)?;

    if !(BLOCK_SIZE_MIN..=BLOCK_SIZE_MAX).contains(&block_size) {
        return Err(einval());
    }
    // Only 512 byte logical sectors are supported.
    if logical_sector_size != 512 {
        return Err(enotsup());
    }
    let sectors_per_block = block_size / logical_sector_size;
    let chunk_ratio = MAX_SECTORS_PER_BLOCK * logical_sector_size as u64 / block_size as u64;
    if !sectors_per_block.is_power_of_two()
        || !chunk_ratio.is_power_of_two()
        || !block_size.is_power_of_two()
    {
        return Err(einval());
    }
    Ok(Metadata { block_size, data_bits, virtual_disk_size, logical_sector_size })
}

/// `vhdx_check_bat_entries()`: every fully present payload block must lie inside the file and
/// clear of the other regions. With `errcnt`, errors are counted there and the check goes
/// on; without, it stops at the first.
fn check_bat_entries(
    file: &Node,
    geo: &Geometry,
    bat: &[u64],
    regions: &Regions,
    total_sectors: u64,
    mut errcnt: Option<&mut i64>,
) -> io::Result<()> {
    let image_file_size = match file.getlength() {
        Ok(l) => l,
        Err(e) => {
            report::error_report("Could not determinate VHDX image file size.");
            return Err(e);
        }
    };
    let block_size = geo.block_size as u64;
    let mut payblocks = geo.chunk_ratio;
    let mut ret = Ok(());
    for i in 0..geo.bat_entries as u64 {
        let entry = bat.get(i as usize).copied().unwrap_or(0);
        if entry & BAT_STATE_BIT_MASK != PAYLOAD_BLOCK_FULLY_PRESENT {
            continue;
        }
        let offset = entry & BAT_FILE_OFF_MASK;
        // The last block may exist only partially: QEMU used to create images like that.
        let block_length = block_size
            .min((total_sectors * BDRV_SECTOR_SIZE).wrapping_sub(i.wrapping_mul(block_size)))
            as u32 as u64;
        let mut fail = |ret: &mut io::Result<()>| -> bool {
            *ret = Err(einval());
            match errcnt.as_deref_mut() {
                Some(c) => {
                    *c += 1;
                    false
                }
                None => true,
            }
        };
        if offset > i64::MAX as u64 - block_size {
            report::error_report(&format!("VHDX BAT entry {i} offset overflow."));
            if fail(&mut ret) {
                break;
            }
        }
        if offset >= image_file_size {
            report::error_report(&format!(
                "VHDX BAT entry {i} start offset {offset} points after end of file \
                 ({image_file_size}). Image has probably been truncated."
            ));
            if fail(&mut ret) {
                break;
            }
        } else if offset.wrapping_add(block_length) > image_file_size {
            report::error_report(&format!(
                "VHDX BAT entry {i} end offset {} points after end of file \
                 ({image_file_size}). Image has probably been truncated.",
                offset.wrapping_add(block_length).wrapping_sub(1)
            ));
            if fail(&mut ret) {
                break;
            }
        }
        // Check the offsets of payload blocks against the regions and the log.
        if payblocks > 0 {
            payblocks -= 1;
            if regions.check(offset, block_size).is_err() && fail(&mut ret) {
                break;
            }
        } else {
            payblocks = geo.chunk_ratio;
            // Sector bitmap blocks are for differencing files, which are not supported.
        }
    }
    ret
}

/// The open `vhdx` image, `BDRVVHDXState`.
struct VhdxDriver {
    geo: Geometry,
    /// `params.data_bits`.
    data_bits: u32,
    virtual_disk_size: u64,
    regions: Regions,
    log_replayed_on_open: bool,
    state: Mutex<State>,
}

/// `vhdx_open()`.
fn open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Vhdx(o) = opts else { unreachable!("vhdx driver with other options") };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    match open_image(&file, args.flags.read_only) {
        Ok(d) => Ok(Box::new(d)),
        Err(OpenError::Msg(e)) => Err(e),
        Err(OpenError::Errno(e)) => Err(match file.filename() {
            Some(f) if !f.is_empty() => Error::from_io(format!("Could not open '{f}'"), e),
            _ => Error::from_io("Could not open image", e),
        }),
    }
}

fn open_image(file: &Arc<Node>, read_only: bool) -> std::result::Result<VhdxDriver, OpenError> {
    // Validate the file signature.
    let mut signature = [0u8; 8];
    file.pread(0, &mut signature)?;
    if &signature != b"vhdxfile" {
        return Err(einval().into());
    }

    // The file write guid of any header update, a new one for this session as the spec says.
    let mut s = State {
        headers: [Header::default(); 2],
        curr_header: 0,
        session_guid: guid_generate()?,
        first_visible_write: true,
        log: LogEntries::default(),
        bat: Vec::new(),
    };
    let mut regions = Regions::default();

    parse_header(file, &mut s, &mut regions)?;
    let log_replayed_on_open = log::parse_log(file, &mut s, read_only)?;
    let (bat_rt, metadata_rt) = open_region_tables(file, &mut regions)?;
    let md = parse_metadata(file, &metadata_rt)?;

    let mut geo = Geometry::new(md.block_size, md.logical_sector_size, md.virtual_disk_size);
    // The spec says virtual_disk_size is a multiple of the logical sector size.
    let total_sectors = md.virtual_disk_size >> geo.logical_sector_size_bits;
    geo.bat_offset = bat_rt.file_offset;

    // The BAT region must be large enough for all entries.
    if geo.bat_entries as u64 > bat_rt.length as u64 / 8 {
        return Err(einval().into());
    }
    let mut raw = Vec::new();
    if raw.try_reserve_exact(bat_rt.length as usize).is_err() {
        return Err(errno(libc::ENOMEM).into());
    }
    raw.resize(bat_rt.length as usize, 0);
    file.pread(geo.bat_offset, &mut raw)?;
    s.bat = raw.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    drop(raw);

    check_bat_entries(file, &geo, &s.bat, &regions, total_sectors, None)?;

    Ok(VhdxDriver {
        geo,
        data_bits: md.data_bits,
        virtual_disk_size: md.virtual_disk_size,
        regions,
        log_replayed_on_open,
        state: Mutex::new(s),
    })
}

/// A block layer [`Error`] as the errno-style error of a guest request.
fn to_io(e: Error) -> io::Error {
    io::Error::other(e.message().to_string())
}

impl VhdxDriver {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap()
    }

    fn total_sectors(&self) -> u64 {
        self.virtual_disk_size >> self.geo.logical_sector_size_bits
    }

    /// `vhdx_allocate_block()`: a new payload block at the end of the file, 1 MiB aligned.
    ///
    /// If `need_zero` is still set on return, truncating could not promise that the new block
    /// reads as zeroes and the caller must write them.
    fn allocate_block(&self, file: &Node, need_zero: &mut bool) -> io::Result<u64> {
        let current_len = file.getlength()?;
        // Per the spec, blocks are placed in units of 1 MiB.
        let new_offset = current_len.div_ceil(MIB) * MIB;
        if new_offset > i64::MAX as u64 {
            return Err(einval());
        }
        let end = (new_offset + self.geo.block_size as u64) as i64;
        if *need_zero {
            match file.truncate_full(end, false, PreallocMode::Off, BDRV_REQ_ZERO_WRITE) {
                // What QEMU gets as ENOTSUP for flags the protocol driver cannot handle.
                Err(e) if e.message() == "Block driver does not support requested flags" => {}
                r => {
                    *need_zero = false;
                    return r.map(|()| new_offset).map_err(to_io);
                }
            }
        }
        file.truncate_full(end, false, PreallocMode::Off, 0).map_err(to_io)?;
        Ok(new_offset)
    }

    /// `vhdx_co_writev()` for `nb_sectors` sectors from `sector_num`.
    fn writev(&self, file: &Node, mut sector_num: u64, buf: &[u8]) -> io::Result<()> {
        let geo = &self.geo;
        let mut s = self.lock();
        user_visible_write(file, &mut s)?;

        let mut nb_sectors = buf.len() as u64 / BDRV_SECTOR_SIZE;
        let mut bytes_done = 0usize;
        while nb_sectors > 0 {
            if self.data_bits & PARAMS_HAS_PARENT != 0 {
                // Differencing files are not supported yet.
                return Err(enotsup());
            }
            let mut sinfo = geo.translate(&s.bat, sector_num, nb_sectors);
            let data = &buf[bytes_done..bytes_done + sinfo.bytes_avail as usize];
            let bat_state = s.bat[sinfo.bat_idx] & BAT_STATE_BIT_MASK;
            let mut bat_update = None;
            let mut bat_prior_offset = 0;
            let mut padded = None;
            match bat_state {
                PAYLOAD_BLOCK_ZERO
                | PAYLOAD_BLOCK_NOT_PRESENT
                | PAYLOAD_BLOCK_UNMAPPED
                | PAYLOAD_BLOCK_UNMAPPED_V095
                | PAYLOAD_BLOCK_UNDEFINED => {
                    // A ZERO block must keep reading zeroes outside this write.
                    let mut use_zero_buffers = bat_state == PAYLOAD_BLOCK_ZERO;
                    bat_prior_offset = sinfo.file_offset;
                    sinfo.file_offset = self.allocate_block(file, &mut use_zero_buffers)?;
                    bat_update = Some(geo.update_bat_table_entry(
                        &mut s.bat,
                        &sinfo,
                        PAYLOAD_BLOCK_FULLY_PRESENT,
                    ));
                    // file_offset is the start of the new block. Write at the offset in the
                    // block, unless truncating could not give zeroes for the whole block, in
                    // which case the whole block is written.
                    if !use_zero_buffers {
                        sinfo.file_offset += sinfo.block_offset;
                    } else {
                        let mut v = vec![0u8; sinfo.block_offset as usize];
                        v.extend_from_slice(data);
                        let bs = geo.block_size as u64;
                        if sinfo.bytes_avail.wrapping_sub(sinfo.block_offset) < bs {
                            let tail = bs - (sinfo.bytes_avail + sinfo.block_offset);
                            v.resize(v.len() + tail as usize, 0);
                        }
                        padded = Some(v);
                    }
                }
                PAYLOAD_BLOCK_FULLY_PRESENT => {}
                // PARTIALLY_PRESENT is for differencing files, which are not supported.
                _ => return Err(errno(libc::EIO)),
            }

            // An offset in the header section means something is badly wrong.
            let r = if sinfo.file_offset < MIB {
                Err(errno(libc::EFAULT))
            } else {
                file.pwrite(sinfo.file_offset, padded.as_deref().unwrap_or(data))
            };
            if let Err(e) = r {
                if bat_update.is_some() {
                    // Keep the BAT in memory in line with the file.
                    sinfo.file_offset = bat_prior_offset;
                    geo.update_bat_table_entry(&mut s.bat, &sinfo, bat_state);
                }
                return Err(e);
            }

            if let Some((bat_entry, bat_entry_offset)) = bat_update {
                // The BAT entry goes into the log, which is then flushed to the file.
                log::write_and_flush(file, &mut s, &bat_entry.to_le_bytes(), bat_entry_offset)?;
            }

            nb_sectors -= sinfo.sectors_avail;
            sector_num += sinfo.sectors_avail;
            bytes_done += sinfo.bytes_avail as usize;
        }
        Ok(())
    }
}

impl Driver for VhdxDriver {
    /// `vhdx_co_readv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        assert!(offset % BDRV_SECTOR_SIZE == 0 && buf.len() as u64 % BDRV_SECTOR_SIZE == 0);
        let file = bs.file();
        let s = self.lock();
        let mut sector_num = offset / BDRV_SECTOR_SIZE;
        let mut nb_sectors = buf.len() as u64 / BDRV_SECTOR_SIZE;
        let mut bytes_done = 0usize;
        while nb_sectors > 0 {
            if self.data_bits & PARAMS_HAS_PARENT != 0 {
                // Differencing files would need the sector bitmap; not supported yet.
                return Err(enotsup());
            }
            let sinfo = self.geo.translate(&s.bat, sector_num, nb_sectors);
            let chunk = &mut buf[bytes_done..bytes_done + sinfo.bytes_avail as usize];
            match s.bat[sinfo.bat_idx] & BAT_STATE_BIT_MASK {
                PAYLOAD_BLOCK_NOT_PRESENT
                | PAYLOAD_BLOCK_UNDEFINED
                | PAYLOAD_BLOCK_UNMAPPED
                | PAYLOAD_BLOCK_UNMAPPED_V095
                | PAYLOAD_BLOCK_ZERO => chunk.fill(0),
                PAYLOAD_BLOCK_FULLY_PRESENT => file.pread(sinfo.file_offset, chunk)?,
                // PARTIALLY_PRESENT is for differencing files, which are not supported.
                _ => return Err(errno(libc::EIO)),
            }
            nb_sectors -= sinfo.sectors_avail;
            sector_num += sinfo.sectors_avail;
            bytes_done += sinfo.bytes_avail as usize;
        }
        Ok(())
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        assert!(offset % BDRV_SECTOR_SIZE == 0 && buf.len() as u64 % BDRV_SECTOR_SIZE == 0);
        self.writev(&bs.file(), offset / BDRV_SECTOR_SIZE, buf)
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.total_sectors() * BDRV_SECTOR_SIZE)
    }

    /// QEMU's `.bdrv_co_readv` and `.bdrv_co_writev` work in whole sectors.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        bl.request_alignment = BDRV_SECTOR_SIZE as u32;
        Ok(())
    }

    /// `vhdx_reopen_prepare()`: nothing to check.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    /// `vhdx_co_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        Some(Ok(BlockDriverInfo { cluster_size: self.geo.block_size as u64, ..Default::default() }))
    }

    /// `vhdx_has_zero_init()`: fixed images have every BAT entry present and dynamic ones none,
    /// right after creation, so the first entry tells which this is.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        if self.geo.bat_entries == 0 {
            return Some(true);
        }
        let state = self.lock().bat.first().copied().unwrap_or(0) & BAT_STATE_BIT_MASK;
        if state == PAYLOAD_BLOCK_FULLY_PRESENT {
            return Some(bs.file().has_zero_init());
        }
        Some(true)
    }

    /// `vhdx_co_check()`. A log that needed replaying was replayed when the image was opened
    /// read-write, which is how `qemu-img check -r` opens it, so that is all there is to fix.
    fn check(&self, bs: &Node, _fix: u32) -> Option<Result<CheckResult>> {
        let mut result = CheckResult::default();
        if self.log_replayed_on_open {
            result.corruptions_fixed += 1;
        }
        let s = self.lock();
        let _ = check_bat_entries(
            &bs.file(),
            &self.geo,
            &s.bat,
            &self.regions,
            self.total_sectors(),
            Some(&mut result.corruptions),
        );
        Some(Ok(result))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// `vhdx_create_new_headers()`: both headers, the second with the next sequence number.
fn create_new_headers(node: &Node, log_size: u32) -> io::Result<()> {
    let mut hdr = Header {
        signature: HEADER_SIGNATURE,
        sequence_number: random_u32()? as u64,
        log_version: 0,
        version: 1,
        log_length: log_size,
        log_offset: HEADER_SECTION_END,
        file_write_guid: guid_generate()?,
        data_write_guid: guid_generate()?,
        ..Default::default()
    };
    write_header(node, &hdr, HEADER1_OFFSET, false)?;
    hdr.sequence_number += 1;
    write_header(node, &hdr, HEADER2_OFFSET, false)
}

/// `vhdx_create_new_metadata()`: the metadata table with the five required items.
fn create_new_metadata(
    node: &Node,
    image_size: u64,
    block_size: u32,
    sector_size: u32,
    metadata_offset: u64,
    fixed: bool,
) -> io::Result<()> {
    // File parameters, virtual disk size, page 83 data, logical and physical sector size.
    let sizes = [8u32, 8, 16, 4, 4];
    let mut entry_buffer = [0u8; 40];
    put32(&mut entry_buffer, 0, block_size);
    if fixed {
        put32(&mut entry_buffer, 4, PARAMS_LEAVE_BLOCKS_ALLOCED);
    }
    put64(&mut entry_buffer, 8, image_size);
    entry_buffer[16..32].copy_from_slice(&guid_generate()?);
    put32(&mut entry_buffer, 32, sector_size);
    put32(&mut entry_buffer, 36, sector_size);

    let mut buffer = vec![0u8; HEADER_BLOCK_SIZE as usize];
    put64(&mut buffer, 0, METADATA_SIGNATURE);
    put16(&mut buffer, 10, 5);

    // The items themselves go after the 64 KiB reserved for the table.
    let ids =
        [FILE_PARAM_GUID, VIRTUAL_SIZE_GUID, PAGE83_GUID, LOGICAL_SECTOR_GUID, PHYS_SECTOR_GUID];
    let mut offset = 64 * KIB as u32;
    for (i, (id, length)) in ids.into_iter().zip(sizes).enumerate() {
        let mut data_bits = META_FLAGS_IS_REQUIRED;
        if i > 0 {
            data_bits |= META_FLAGS_IS_VIRTUAL_DISK;
        }
        let e = MetadataTableEntry { item_id: id, offset, length, data_bits };
        e.write(&mut buffer[METADATA_HEADER_SIZE + i * METADATA_ENTRY_SIZE..]);
        offset += length;
    }

    node.pwrite(metadata_offset, &buffer)?;
    node.pwrite(metadata_offset + 64 * KIB, &entry_buffer)
}

/// `vhdx_create_bat()`: a dynamic image gets an all zero BAT, which a file that reads as
/// zeroes already has; a fixed image gets every block present.
#[allow(clippy::too_many_arguments)]
fn create_bat(
    node: &Node,
    geo: &Geometry,
    image_size: u64,
    fixed: bool,
    use_zero_blocks: bool,
    file_offset: u64,
    length: u32,
) -> Result<()> {
    // Data starts after the BAT and well past the metadata, leaving 4 MiB spare for later.
    let data_file_offset = file_offset + length as u64 + 5 * MIB;
    let total_sectors = image_size >> geo.logical_sector_size_bits;

    let new_len = if fixed { data_file_offset + image_size } else { data_file_offset };
    node.truncate_full(new_len as i64, false, PreallocMode::Off, 0)?;

    if fixed || use_zero_blocks || !node.has_zero_init() {
        let mut bat = Vec::new();
        if bat.try_reserve_exact(length as usize / 8).is_err() {
            return Err(Error::generic("Failed to allocate memory for the BAT"));
        }
        bat.resize(length as usize / 8, 0u64);
        let mut block_state =
            if fixed { PAYLOAD_BLOCK_FULLY_PRESENT } else { PAYLOAD_BLOCK_NOT_PRESENT };
        if use_zero_blocks {
            block_state = PAYLOAD_BLOCK_ZERO;
        }
        // Fill the BAT as if writing whole blocks.
        let mut sector_num = 0u64;
        while sector_num < total_sectors {
            let mut sinfo = geo.translate(&bat, sector_num, geo.sectors_per_block as u64);
            sinfo.file_offset = data_file_offset + (sector_num << geo.logical_sector_size_bits);
            sinfo.file_offset = sinfo.file_offset.div_ceil(MIB) * MIB;
            geo.update_bat_table_entry(&mut bat, &sinfo, block_state);
            sector_num += geo.sectors_per_block as u64;
        }
        let mut raw = vec![0u8; length as usize];
        for (c, e) in raw.chunks_exact_mut(8).zip(&bat) {
            c.copy_from_slice(&e.to_le_bytes());
        }
        node.pwrite(file_offset, &raw).map_err(|e| Error::from_io("Failed to write the BAT", e))?;
    }
    Ok(())
}

/// `vhdx_create_new_region_table()`: both region tables, and the BAT they point at. Returns
/// the offset of the metadata region.
#[allow(clippy::too_many_arguments)]
fn create_new_region_table(
    node: &Node,
    image_size: u64,
    block_size: u32,
    sector_size: u32,
    log_size: u32,
    use_zero_blocks: bool,
    fixed: bool,
) -> Result<u64> {
    let mut geo = Geometry::new(block_size, sector_size, image_size);

    let mut buffer = vec![0u8; HEADER_BLOCK_SIZE as usize];
    put32(&mut buffer, 0, REGION_SIGNATURE);
    // The BAT and the metadata.
    put32(&mut buffer, 8, 2);

    let bat_length = (geo.bat_entries as u64 * 8).div_ceil(MIB) * MIB;
    let rt_bat = RegionTableEntry {
        guid: BAT_GUID,
        length: bat_length as u32,
        file_offset: (HEADER_SECTION_END + log_size as u64).div_ceil(MIB) * MIB,
        data_bits: 0,
    };
    geo.bat_offset = rt_bat.file_offset;
    let rt_metadata = RegionTableEntry {
        guid: METADATA_GUID,
        file_offset: (rt_bat.file_offset + rt_bat.length as u64).div_ceil(MIB) * MIB,
        // The minimum, and more than enough.
        length: MIB as u32,
        data_bits: 0,
    };
    rt_bat.write(&mut buffer[16..]);
    rt_metadata.write(&mut buffer[48..]);
    update_checksum(&mut buffer, 4);

    create_bat(node, &geo, image_size, fixed, use_zero_blocks, rt_bat.file_offset, rt_bat.length)?;

    node.pwrite(REGION_TABLE_OFFSET, &buffer)
        .map_err(|e| Error::from_io("Failed to write first region table", e))?;
    node.pwrite(REGION_TABLE2_OFFSET, &buffer)
        .map_err(|e| Error::from_io("Failed to write second region table", e))?;
    Ok(rt_metadata.file_offset)
}

/// `vhdx_co_create()` on the node that holds the image.
///
/// The layout: file identifier at 0, the headers at 64 and 128 KiB, the region tables at 192
/// and 256 KiB, then from 1 MiB the log, the BAT, the metadata and the data blocks.
fn co_create_on(node: &Node, o: &BlockdevCreateOptionsVhdx) -> Result<()> {
    let image_size = o.size;
    let log_size = o.log_size.unwrap_or(DEFAULT_LOG_SIZE as u64);
    let use_zero_blocks = o.block_state_zero.unwrap_or(true);
    let fixed = o.subformat.unwrap_or_default() == BlockdevVhdxSubformat::Fixed;

    // These keep the BAT a reasonable size to hold in memory. QEMU stores the size in a
    // uint32_t, so a larger one wraps.
    let block_size = match o.block_size {
        Some(b) => b as u32,
        None if image_size > 32 * TIB => 64 * MIB as u32,
        None if image_size > 100 * GIB => 32 * MIB as u32,
        None if image_size > GIB => 16 * MIB as u32,
        None => 8 * MIB as u32,
    };
    let log_size = log_size as u32;

    // The creator field is optional, and there for diagnostics.
    let creator: Vec<u8> =
        format!("QEMU v{QEMU_VERSION}").encode_utf16().flat_map(|c| c.to_le_bytes()).collect();
    node.pwrite(FILE_ID_OFFSET, &FILE_SIGNATURE.to_le_bytes())
        .map_err(|e| Error::from_io("Failed to write file signature", e))?;
    node.pwrite(FILE_ID_OFFSET + 8, &creator)
        .map_err(|e| Error::from_io("Failed to write creator field", e))?;

    create_new_headers(node, log_size)
        .map_err(|e| Error::from_io("Failed to write image headers", e))?;

    let metadata_offset = create_new_region_table(
        node,
        image_size,
        block_size,
        512,
        log_size,
        use_zero_blocks,
        fixed,
    )?;

    create_new_metadata(node, image_size, block_size, 512, metadata_offset, fixed)
        .map_err(|e| Error::from_io("Failed to initialize metadata", e))
}

/// The option checks at the start of `vhdx_co_create()`.
fn check_create(o: &BlockdevCreateOptionsVhdx) -> Result<()> {
    if o.size > MAX_IMAGE_SIZE {
        return Err(Error::generic("Image size too large; max of 64TB"));
    }
    let log_size = match o.log_size {
        Some(l) if l > u32::MAX as u64 => {
            return Err(Error::generic("Log size must be smaller than 4 GB"));
        }
        Some(l) => l,
        None => DEFAULT_LOG_SIZE as u64,
    };
    if log_size < MIB || log_size % MIB != 0 {
        return Err(Error::generic("Log size must be a multiple of 1 MB"));
    }
    if let Some(b) = o.block_size {
        let b = b as u32;
        if (b as u64) < MIB || b as u64 % MIB != 0 {
            return Err(Error::generic("Block size must be a multiple of 1 MB"));
        }
        if !b.is_power_of_two() {
            return Err(Error::generic("Block size must be a power of two"));
        }
        if b > BLOCK_SIZE_MAX {
            return Err(Error::generic(format!("Block size must not exceed {BLOCK_SIZE_MAX}")));
        }
    }
    Ok(())
}

/// `vhdx_co_create()`: `blockdev-create` with `driver: vhdx`.
fn create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Vhdx(o) = options else {
        unreachable!("vhdx driver with other create options")
    };
    check_create(&o)?;
    let blk = graph.open_create_blk(o.file.clone())?;
    let node = blk.root().expect("a new backend has its node");
    co_create_on(&node, &o)
}

/// Takes `key` out of the `-o` options as a string.
fn take(options: &mut QDict, key: &str) -> Option<String> {
    options.remove(key).and_then(|v| match v {
        QValue::Str(s) => Some(s),
        v => v.as_str().map(str::to_owned),
    })
}

/// `vhdx_co_create_opts()`: `qemu-img create -f vhdx`.
fn create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let size = take(options, "size").map(|v| parse_option_size("size", &v)).transpose()?;
    let log_size =
        take(options, "log_size").map(|v| parse_option_size("log_size", &v)).transpose()?;
    let block_size =
        take(options, "block_size").map(|v| parse_option_size("block_size", &v)).transpose()?;
    let subformat = take(options, "subformat");
    let block_state_zero = take(options, "block_state_zero")
        .map(|v| qapi_bool_parse("block_state_zero", &v))
        .transpose()?;

    // Create and open the file (protocol layer).
    let graph = BlockGraph::new();
    graph.create_file(filename, options)?;
    let blk = graph.open_protocol_blk(filename)?;
    let node = blk.root().expect("a new backend has its node");

    let Some(size) = size else { return Err(Error::generic("Parameter 'size' is missing")) };
    let subformat = match subformat {
        Some(v) => Some(BlockdevVhdxSubformat::from_name(&v).ok_or_else(|| {
            Error::generic(format!("Parameter 'subformat' does not accept value '{v}'"))
        })?),
        None => None,
    };

    // Round the sizes up quietly: the image size to 512 bytes, the block and log sizes to
    // 1 MiB. A block size of 0 means automatic, which QAPI says by leaving it out.
    let mut block_size = block_size.map(|b| round_up(b, MIB));
    if block_size == Some(0) {
        block_size = None;
    }
    if let Some(b) = block_size {
        if b > BLOCK_SIZE_MAX as u64 {
            block_size = Some(BLOCK_SIZE_MAX as u64);
        }
    }
    let o = BlockdevCreateOptionsVhdx {
        file: ruvm_qapi::types::BlockdevRef::Reference(String::new()),
        size: round_up(size, BDRV_SECTOR_SIZE),
        log_size: log_size.map(|l| round_up(l, MIB)),
        block_size,
        subformat,
        block_state_zero,
    };
    check_create(&o)?;
    co_create_on(&node, &o)
}

/// `ROUND_UP()`, which wraps like the C macro does.
fn round_up(n: u64, d: u64) -> u64 {
    n.wrapping_add(d - 1) & !(d - 1)
}

#[cfg(test)]
mod tests;
