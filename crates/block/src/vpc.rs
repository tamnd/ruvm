// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vpc` format driver from block/vpc.c: Connectix / Microsoft Virtual PC images, better
//! known as VHD.
//!
//! A VHD image ends with a 512 byte footer ("conectix") that gives the size, the disk geometry
//! and the type. A fixed image is the raw disk followed by the footer. A dynamic image has a
//! copy of the footer at offset 0, then a dynamic disk header ("cxsparse") that points at the
//! block allocation table (BAT). Each allocated block is a sector bitmap followed by the block
//! data, 2 MiB by default, and the footer moves to the end of the file whenever a block is
//! added. Differencing images are opened like dynamic ones, without their parent, as QEMU does.
//!
//! The guest visible size follows QEMU's rules: images whose creator application is "vpc " or
//! "qemu" use the CHS geometry, all others (Hyper-V, disk2vhd, XenServer, Azure, QEMU with
//! `force_size`, which writes "qem2") use `current_size`, and so does an image whose geometry
//! is the maximum 65535/16/255. Creating an image rounds the size up to the next size the CHS
//! geometry can represent unless `force_size` is on.
//!
//! Differences from QEMU:
//!
//! - The runtime option `force_size_calc` (`chs` or `current_size`) is not taken: the typed
//!   `BlockdevOptions` of `vpc` have no such member, and QEMU only accepts it from `-drive`
//!   style option dictionaries. The creator application decides, as when QEMU is not given
//!   the option.
//! - QEMU registers a migration blocker for every open vpc node; there is no migration here.
//! - QEMU drops the state lock around the data transfer of a request. Here reads drop it too,
//!   while writes hold it to the end, which only serialises writes to the same image.

use std::io;
use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::types::{
    BlockdevCreateOptionsU, BlockdevCreateOptionsVpc, BlockdevOptionsU, BlockdevRef,
    BlockdevVpcSubformat, PreallocMode,
};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::imgopts::{take_bool, take_size, take_str};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RECURSE, BDRV_BLOCK_ZERO,
    BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BDRV_SECTOR_SIZE, BlockDriverInfo, BlockStatus, Driver,
    Node, ReopenState, errno,
};

/// `bdrv_vpc`.
pub(crate) static VPC: DriverDef = DriverDef::format("vpc", vpc_open_node)
    .with_probe(vpc_probe)
    .with_create_opts(vpc_co_create_opts)
    .with_create_opts_list(&VPC_CREATE_OPTS)
    .with_create(vpc_co_create)
    .with_strong_opts(&["force_size_calc"]);

/// `vpc_create_opts`, in QEMU's order.
static VPC_CREATE_OPTS: [QemuOptDesc; 3] = [
    QemuOptDesc::new("size", QemuOptType::Size).help("Virtual disk size"),
    QemuOptDesc::new("subformat", QemuOptType::String).help(
        "Type of virtual hard disk format. Supported formats are {dynamic (default) | fixed} ",
    ),
    QemuOptDesc::new("force_size", QemuOptType::Bool).help(
        "Force disk size calculation to use the actual size specified, rather than using the \
         nearest CHS-based calculation",
    ),
];

const VHD_FIXED: u32 = 2;
const VHD_DYNAMIC: u32 = 3;

/// Seconds since Jan 1, 2000 0:00:00 (UTC).
const VHD_TIMESTAMP_BASE: u64 = 946_684_800;

const VHD_CHS_MAX_C: u16 = 65535;
const VHD_CHS_MAX_H: u8 = 16;
const VHD_CHS_MAX_S: u8 = 255;

/// 2040 GiB, the largest image.
const VHD_MAX_SECTORS: u64 = 0xff00_0000;
const VHD_MAX_GEOMETRY: u64 = VHD_CHS_MAX_C as u64 * VHD_CHS_MAX_H as u64 * VHD_CHS_MAX_S as u64;

const FOOTER_SIZE: usize = 512;
const DYNDISK_HEADER_SIZE: usize = 1024;

/// Offsets into `VHDFooter`, all fields big-endian.
mod footer {
    pub(super) const CREATOR: usize = 0;
    pub(super) const FEATURES: usize = 8;
    pub(super) const VERSION: usize = 12;
    pub(super) const DATA_OFFSET: usize = 16;
    pub(super) const TIMESTAMP: usize = 24;
    pub(super) const CREATOR_APP: usize = 28;
    pub(super) const MAJOR: usize = 32;
    pub(super) const MINOR: usize = 34;
    pub(super) const CREATOR_OS: usize = 36;
    pub(super) const ORIG_SIZE: usize = 40;
    pub(super) const CURRENT_SIZE: usize = 48;
    pub(super) const CYLS: usize = 56;
    pub(super) const HEADS: usize = 58;
    pub(super) const SECS_PER_CYL: usize = 59;
    pub(super) const TYPE: usize = 60;
    pub(super) const CHECKSUM: usize = 64;
    pub(super) const UUID: usize = 68;
}

/// Offsets into `VHDDynDiskHeader`, all fields big-endian.
mod dyndisk {
    pub(super) const MAGIC: usize = 0;
    pub(super) const DATA_OFFSET: usize = 8;
    pub(super) const TABLE_OFFSET: usize = 16;
    pub(super) const VERSION: usize = 24;
    pub(super) const MAX_TABLE_ENTRIES: usize = 28;
    pub(super) const BLOCK_SIZE: usize = 32;
    pub(super) const CHECKSUM: usize = 36;
}

fn be16(b: &[u8], off: usize) -> u16 {
    u16::from_be_bytes([b[off], b[off + 1]])
}

fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn be64(b: &[u8], off: usize) -> u64 {
    u64::from_be_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

fn put_be16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_be_bytes());
}

fn put_be32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_be_bytes());
}

fn put_be64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_be_bytes());
}

/// `vpc_checksum()`: the one's complement of the byte sum.
fn vpc_checksum(buf: &[u8]) -> u32 {
    !buf.iter().fold(0u32, |a, &b| a.wrapping_add(u32::from(b)))
}

/// `vpc_probe()`.
fn vpc_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() >= 8 && &buf[..8] == b"conectix" { 100 } else { 0 }
}

/// `vpc_ignore_current_size()`: Virtual PC and old QEMU versions size the disk by its CHS
/// geometry.
fn vpc_ignore_current_size(footer: &[u8]) -> bool {
    let app = &footer[footer::CREATOR_APP..footer::CREATOR_APP + 4];
    app == b"vpc " || app == b"qemu"
}

/// The parts of `BDRVVPCState` requests change, under `s->lock`.
#[derive(Debug)]
struct VpcDynamic {
    /// `pagetable`, in host order.
    pagetable: Vec<u32>,
    free_data_block_offset: u64,
    /// `u64::MAX` for none, QEMU's `(int64_t) -1`.
    last_bitmap_offset: u64,
}

/// `BDRVVPCState`.
#[derive(Debug)]
pub(crate) struct VpcDriver {
    /// The footer as it is on disk, with its checksum.
    footer: [u8; FOOTER_SIZE],
    fixed: bool,
    total_sectors: u64,
    max_table_entries: u32,
    bat_offset: u64,
    block_size: u32,
    bitmap_size: u32,
    lock: Mutex<VpcDynamic>,
}

fn open_err(file: &Node, e: io::Error) -> Error {
    let name = file.filename().unwrap_or_default();
    Error::from_io(format!("Could not open '{name}'"), e)
}

/// `vpc_open()`.
fn vpc_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Vpc(o) = opts else { unreachable!("vpc driver with other options") };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    Ok(Box::new(vpc_open(&file)?))
}

fn vpc_open(file: &Node) -> Result<VpcDriver> {
    let mut footer = [0u8; FOOTER_SIZE];
    file.pread(0, &mut footer).map_err(|_| Error::generic("Unable to read VHD header"))?;

    let mut disk_type = VHD_DYNAMIC;
    if &footer[..8] != b"conectix" {
        let offset = file.getlength().map_err(|_| Error::generic("Invalid file size"))?;
        if offset < FOOTER_SIZE as u64 {
            return Err(Error::generic("File too small for a VHD header"));
        }
        // A fixed disk has its footer only at the end of the file.
        file.pread(offset - FOOTER_SIZE as u64, &mut footer).map_err(|e| open_err(file, e))?;
        if &footer[..8] != b"conectix" || be32(&footer, footer::TYPE) != VHD_FIXED {
            return Err(Error::generic("invalid VPC image"));
        }
        disk_type = VHD_FIXED;
    }

    let checksum = be32(&footer, footer::CHECKSUM);
    let mut zeroed = footer;
    put_be32(&mut zeroed, footer::CHECKSUM, 0);
    if vpc_checksum(&zeroed) != checksum {
        return Err(Error::generic("Incorrect header checksum"));
    }

    // The visible size in Virtual PC comes from the geometry rather than from the size in the
    // footer, which is usually too large.
    let mut total_sectors = u64::from(be16(&footer, footer::CYLS))
        * u64::from(footer[footer::HEADS])
        * u64::from(footer[footer::SECS_PER_CYL]);
    let use_chs = vpc_ignore_current_size(&footer);
    if !use_chs || total_sectors == VHD_MAX_GEOMETRY {
        total_sectors = be64(&footer, footer::CURRENT_SIZE) / BDRV_SECTOR_SIZE;
    }
    if total_sectors > VHD_MAX_SECTORS {
        return Err(open_err(file, errno(libc::EFBIG)));
    }

    let mut s = VpcDriver {
        footer,
        fixed: disk_type == VHD_FIXED,
        total_sectors,
        max_table_entries: 0,
        bat_offset: 0,
        block_size: 0,
        bitmap_size: 0,
        lock: Mutex::new(VpcDynamic {
            pagetable: Vec::new(),
            free_data_block_offset: 0,
            last_bitmap_offset: u64::MAX,
        }),
    };
    if disk_type != VHD_DYNAMIC {
        return Ok(s);
    }

    let mut hdr = [0u8; DYNDISK_HEADER_SIZE];
    file.pread(be64(&footer, footer::DATA_OFFSET), &mut hdr)
        .map_err(|_| Error::generic("Error reading dynamic VHD header"))?;
    if &hdr[dyndisk::MAGIC..dyndisk::MAGIC + 8] != b"cxsparse" {
        return Err(Error::generic("Invalid header magic"));
    }
    let block_size = be32(&hdr, dyndisk::BLOCK_SIZE);
    if !block_size.is_power_of_two() || u64::from(block_size) < BDRV_SECTOR_SIZE {
        return Err(Error::generic(format!("Invalid block size {block_size}")));
    }
    let bitmap_size = ((block_size / (8 * 512)) + 511) & !511;
    // An int in QEMU.
    let max_table_entries = be32(&hdr, dyndisk::MAX_TABLE_ENTRIES) as i32;

    if (total_sectors * 512) / u64::from(block_size) > 0xffff_ffff {
        return Err(Error::generic("Too many blocks"));
    }
    let computed_size = (max_table_entries as i64 as u64).wrapping_mul(u64::from(block_size));
    if computed_size < total_sectors * 512 {
        return Err(Error::generic("Page table too small"));
    }
    if max_table_entries as i64 as u64 > (usize::MAX / 4) as u64 || max_table_entries > i32::MAX / 4
    {
        return Err(Error::generic(format!("Max Table Entries too large ({max_table_entries})")));
    }
    let max_table_entries = max_table_entries as u32;
    let pagetable_size = u64::from(max_table_entries) * 4;
    let bat_offset = be64(&hdr, dyndisk::TABLE_OFFSET);
    let mut raw = vec![0u8; pagetable_size as usize];
    file.pread(bat_offset, &mut raw).map_err(|_| Error::generic("Error reading pagetable"))?;

    let mut free = (bat_offset + pagetable_size).div_ceil(512) * 512;
    let pagetable: Vec<u32> = raw.chunks_exact(4).map(|c| be32(c, 0)).collect();
    for &e in &pagetable {
        if e != 0xffff_ffff {
            let next = 512 * u64::from(e) + u64::from(bitmap_size) + u64::from(block_size);
            free = free.max(next);
        }
    }
    let bs_size = file.getlength().map_err(|e| Error::from_io("Unable to learn image size", e))?;
    if free > bs_size {
        return Err(Error::generic(
            "block-vpc: free_data_block_offset points after the end of file. The image has been \
             truncated.",
        ));
    }

    s.max_table_entries = max_table_entries;
    s.bat_offset = bat_offset;
    s.block_size = block_size;
    s.bitmap_size = bitmap_size;
    s.lock = Mutex::new(VpcDynamic {
        pagetable,
        free_data_block_offset: free,
        last_bitmap_offset: u64::MAX,
    });
    Ok(s)
}

/// `bdrv_co_pwrite_sync()`: a write and a flush of the node.
fn pwrite_sync(file: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
    file.pwrite(offset, buf)?;
    file.flush()
}

impl VpcDriver {
    fn state(&self) -> MutexGuard<'_, VpcDynamic> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `get_image_offset()`: where the byte at `offset` is in the file, or `None` when its
    /// block is not allocated. For a write, the block bitmap is set to all ones first.
    fn get_image_offset(
        &self,
        s: &mut VpcDynamic,
        file: &Node,
        offset: u64,
        write: bool,
    ) -> io::Result<Option<u64>> {
        let index = offset / u64::from(self.block_size);
        let in_block = offset % u64::from(self.block_size);
        if index >= u64::from(self.max_table_entries) || s.pagetable[index as usize] == 0xffff_ffff
        {
            return Ok(None);
        }
        let bitmap_offset = 512 * u64::from(s.pagetable[index as usize]);
        let block_offset = bitmap_offset + u64::from(self.bitmap_size) + in_block;

        // Never write to sectors the bitmap marks as unused: every sector of a block we write
        // to is marked used, which may cost Virtual PC its sparse read optimisation but is
        // correct.
        if write && s.last_bitmap_offset != bitmap_offset {
            s.last_bitmap_offset = bitmap_offset;
            let bitmap = vec![0xffu8; self.bitmap_size as usize];
            pwrite_sync(file, bitmap_offset, &bitmap)?;
        }
        Ok(Some(block_offset))
    }

    /// `rewrite_footer()`: the footer goes after the last block.
    fn rewrite_footer(&self, s: &VpcDynamic, file: &Node) -> io::Result<()> {
        pwrite_sync(file, s.free_data_block_offset, &self.footer)
    }

    /// `alloc_block()`: a new block at the end of the file, where the footer was.
    fn alloc_block(&self, s: &mut VpcDynamic, file: &Node, offset: u64) -> io::Result<u64> {
        if offset > self.total_sectors * BDRV_SECTOR_SIZE {
            return Err(errno(libc::EINVAL));
        }
        let index = (offset / u64::from(self.block_size)) as usize;
        assert_eq!(s.pagetable[index], 0xffff_ffff);
        s.pagetable[index] = (s.free_data_block_offset / 512) as u32;

        let bitmap = vec![0xffu8; self.bitmap_size as usize];
        // On failure QEMU leaves the in-memory entry behind, and so does this.
        pwrite_sync(file, s.free_data_block_offset, &bitmap)?;

        let grow = u64::from(self.block_size) + u64::from(self.bitmap_size);
        s.free_data_block_offset += grow;
        let r = self.rewrite_footer(s, file).and_then(|()| {
            let bat_offset = self.bat_offset + 4 * index as u64;
            pwrite_sync(file, bat_offset, &s.pagetable[index].to_be_bytes())
        });
        if let Err(e) = r {
            s.free_data_block_offset -= grow;
            return Err(e);
        }
        Ok(self.get_image_offset(s, file, offset, false)?.expect("just allocated"))
    }
}

impl Driver for VpcDriver {
    /// `vpc_co_preadv()`.
    fn pread(&self, bs: &Node, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let file = bs.file();
        if self.fixed {
            return file.pread(offset, buf);
        }
        let mut done = 0usize;
        while done < buf.len() {
            let image_offset = {
                let mut s = self.state();
                self.get_image_offset(&mut s, &file, offset, false)?
            };
            let n = (buf.len() - done)
                .min((u64::from(self.block_size) - offset % u64::from(self.block_size)) as usize);
            let chunk = &mut buf[done..done + n];
            match image_offset {
                None => chunk.fill(0),
                Some(o) => file.pread(o, chunk)?,
            }
            done += n;
            offset += n as u64;
        }
        Ok(())
    }

    /// `vpc_co_pwritev()`.
    fn pwrite(&self, bs: &Node, mut offset: u64, buf: &[u8]) -> io::Result<()> {
        let file = bs.file();
        if self.fixed {
            return file.pwrite(offset, buf);
        }
        let mut s = self.state();
        let mut done = 0usize;
        while done < buf.len() {
            let image_offset = self.get_image_offset(&mut s, &file, offset, true)?;
            let n = (buf.len() - done)
                .min((u64::from(self.block_size) - offset % u64::from(self.block_size)) as usize);
            let image_offset = match image_offset {
                Some(o) => o,
                None => self.alloc_block(&mut s, &file, offset)?,
            };
            file.pwrite(image_offset, &buf[done..done + n])?;
            done += n;
            offset += n as u64;
        }
        Ok(())
    }

    /// The disk size, from the geometry or the footer.
    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.total_sectors * BDRV_SECTOR_SIZE)
    }

    /// `vpc_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        mut offset: u64,
        mut bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let file = bs.file();
        if self.fixed {
            return Some(Ok(BlockStatus {
                ret: BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | BDRV_BLOCK_RECURSE,
                pnum: bytes,
                map: offset,
                file: Some(file),
            }));
        }
        let mut s = self.state();
        let bsz = u64::from(self.block_size);
        let mut image_offset = match self.get_image_offset(&mut s, &file, offset, false) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let allocated = image_offset.is_some();
        let mut st = BlockStatus { ret: BDRV_BLOCK_ZERO, pnum: 0, map: 0, file: None };
        loop {
            // All sectors of a block are contiguous, the bitmap is not looked at.
            let n = ((offset + 1).div_ceil(bsz) * bsz - offset).min(bytes);
            st.pnum += n;
            offset += n;
            bytes -= n;
            // An allocated extent never spans blocks, there is a bitmap in between.
            if allocated {
                st.file = Some(file.clone());
                st.map = image_offset.expect("allocated");
                st.ret = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID;
                break;
            }
            if bytes == 0 {
                break;
            }
            image_offset = match self.get_image_offset(&mut s, &file, offset, false) {
                Ok(o) => o,
                Err(e) => return Some(Err(e)),
            };
            if image_offset.is_some() {
                break;
            }
        }
        Some(Ok(st))
    }

    /// `vpc_co_get_info()`: dynamic images have clusters of the block size.
    fn get_info(&self, _bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        let mut bdi = BlockDriverInfo::default();
        if !self.fixed {
            bdi.cluster_size = u64::from(self.block_size);
        }
        Some(Ok(bdi))
    }

    /// `vpc_has_zero_init()`.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        if self.fixed { Some(bs.file().has_zero_init()) } else { Some(true) }
    }

    /// `vpc_reopen_prepare()`: nothing to check.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// `calculate_geometry()`: the CHS geometry for `total_sectors` as the VHD specification
/// describes it, allowing up to 255 sectors per track. The geometry may round the size down.
fn calculate_geometry(total_sectors: u64) -> (u16, u8, u8) {
    let total_sectors = total_sectors.min(VHD_MAX_GEOMETRY);
    let mut secs_per_cyl: u32;
    let mut heads: u32;
    let mut cyls_times_heads: u32;
    if total_sectors >= 65535 * 16 * 63 {
        secs_per_cyl = 255;
        heads = 16;
        cyls_times_heads = (total_sectors / u64::from(secs_per_cyl)) as u32;
    } else {
        secs_per_cyl = 17;
        cyls_times_heads = (total_sectors / u64::from(secs_per_cyl)) as u32;
        // A uint8_t in QEMU.
        heads = cyls_times_heads.div_ceil(1024) & 0xff;
        if heads < 4 {
            heads = 4;
        }
        if cyls_times_heads >= heads * 1024 || heads > 16 {
            secs_per_cyl = 31;
            heads = 16;
            cyls_times_heads = (total_sectors / u64::from(secs_per_cyl)) as u32;
        }
        if cyls_times_heads >= heads * 1024 {
            secs_per_cyl = 63;
            heads = 16;
            cyls_times_heads = (total_sectors / u64::from(secs_per_cyl)) as u32;
        }
    }
    ((cyls_times_heads / heads) as u16, heads as u8, secs_per_cyl as u8)
}

/// `calculate_rounded_image_size()`: the geometry and the number of sectors of a new image of
/// `size` bytes.
fn calculate_rounded_image_size(size: u64, force_size: bool) -> Result<((u16, u8, u8), u64)> {
    let mut geo = (0u16, 0u8, 0u8);
    let chs = |g: (u16, u8, u8)| u64::from(g.0) * u64::from(g.1) * u64::from(g.2);
    if force_size {
        // This makes the size below come from `size`.
        geo = (VHD_CHS_MAX_C, VHD_CHS_MAX_H, VHD_CHS_MAX_S);
    } else {
        // Grow the sector count until the geometry covers it, so that a conversion rounds up
        // rather than truncates.
        let total_sectors = VHD_MAX_GEOMETRY.min(size / BDRV_SECTOR_SIZE);
        let mut i = 0;
        while total_sectors > chs(geo) {
            geo = calculate_geometry(total_sectors + i);
            i += 1;
        }
    }
    let total_sectors = if chs(geo) == VHD_MAX_GEOMETRY {
        let t = size / BDRV_SECTOR_SIZE;
        if t > VHD_MAX_SECTORS {
            return Err(Error::generic("Disk size is too large, max size is 2040 GiB"));
        }
        t
    } else {
        chs(geo)
    };
    Ok((geo, total_sectors))
}

/// `qemu_uuid_generate()`: a random version 4 UUID.
fn uuid_generate() -> Result<[u8; 16]> {
    let mut u = [0u8; 16];
    ruvm_crypto::random::random_bytes(&mut u)?;
    u[6] = (u[6] & 0x0f) | 0x40;
    u[8] = (u[8] & 0x3f) | 0x80;
    Ok(u)
}

/// `create_dynamic_disk()`.
fn create_dynamic_disk(node: &Node, footer: &[u8], total_sectors: u64) -> io::Result<()> {
    let block_size: u64 = 0x20_0000;
    let num_bat_entries = total_sectors.div_ceil(block_size / 512);

    // The footer goes at the start and at the end.
    node.pwrite(0, footer)?;
    let offset = 1536 + ((num_bat_entries * 4 + 511) & !511);
    node.pwrite(offset, footer)?;

    // The initial BAT: nothing allocated.
    let bat_sector = [0xffu8; 512];
    let mut offset = 3 * 512;
    for _ in 0..(num_bat_entries * 4).div_ceil(512) {
        node.pwrite(offset, &bat_sector)?;
        offset += 512;
    }

    let mut hdr = [0u8; DYNDISK_HEADER_SIZE];
    hdr[dyndisk::MAGIC..dyndisk::MAGIC + 8].copy_from_slice(b"cxsparse");
    // The specification says 0xFFFFFFFF, but Microsoft's tools want all 64 bits set.
    put_be64(&mut hdr, dyndisk::DATA_OFFSET, u64::MAX);
    put_be64(&mut hdr, dyndisk::TABLE_OFFSET, 3 * 512);
    put_be32(&mut hdr, dyndisk::VERSION, 0x0001_0000);
    put_be32(&mut hdr, dyndisk::BLOCK_SIZE, block_size as u32);
    put_be32(&mut hdr, dyndisk::MAX_TABLE_ENTRIES, num_bat_entries as u32);
    let sum = vpc_checksum(&hdr);
    put_be32(&mut hdr, dyndisk::CHECKSUM, sum);
    node.pwrite(512, &hdr)
}

/// `create_fixed_disk()`.
fn create_fixed_disk(node: &Node, footer: &[u8], total_size: u64) -> Result<()> {
    let total_size = total_size + FOOTER_SIZE as u64;
    node.truncate_full(total_size as i64, false, PreallocMode::Off, 0)?;
    node.pwrite(total_size - FOOTER_SIZE as u64, footer)
        .map_err(|e| Error::from_io("Unable to write VHD header", e))
}

/// `vpc_co_create()`: `blockdev-create` with `driver: vpc`.
fn vpc_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Vpc(o) = options else {
        unreachable!("vpc driver with other create options")
    };
    do_create(graph, o)
}

fn do_create(graph: &BlockGraph, o: BlockdevCreateOptionsVpc) -> Result<()> {
    let total_size = o.size;
    let disk_type = match o.subformat.unwrap_or(BlockdevVpcSubformat::Dynamic) {
        BlockdevVpcSubformat::Dynamic => VHD_DYNAMIC,
        BlockdevVpcSubformat::Fixed => VHD_FIXED,
    };
    let force_size = o.force_size.unwrap_or(false);

    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");

    let ((cyls, heads, secs_per_cyl), total_sectors) =
        calculate_rounded_image_size(total_size, force_size)?;
    if total_size != total_sectors * BDRV_SECTOR_SIZE {
        return Err(Error::generic(
            "The requested image size cannot be represented in CHS geometry",
        )
        .hint(format!(
            "Try size={} or force-size=on (the latter makes the image incompatible with \
             Virtual PC)",
            total_sectors * BDRV_SECTOR_SIZE
        )));
    }

    let mut footer = [0u8; FOOTER_SIZE];
    footer[footer::CREATOR..footer::CREATOR + 8].copy_from_slice(b"conectix");
    footer[footer::CREATOR_APP..footer::CREATOR_APP + 4].copy_from_slice(if force_size {
        b"qem2"
    } else {
        b"qemu"
    });
    footer[footer::CREATOR_OS..footer::CREATOR_OS + 4].copy_from_slice(b"Wi2k");
    put_be32(&mut footer, footer::FEATURES, 0x02);
    put_be32(&mut footer, footer::VERSION, 0x0001_0000);
    put_be64(
        &mut footer,
        footer::DATA_OFFSET,
        if disk_type == VHD_DYNAMIC { FOOTER_SIZE as u64 } else { u64::MAX },
    );
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    put_be32(&mut footer, footer::TIMESTAMP, now.wrapping_sub(VHD_TIMESTAMP_BASE) as u32);
    // The version of Virtual PC 2007.
    put_be16(&mut footer, footer::MAJOR, 0x0005);
    put_be16(&mut footer, footer::MINOR, 0x0003);
    put_be64(&mut footer, footer::ORIG_SIZE, total_size);
    put_be64(&mut footer, footer::CURRENT_SIZE, total_size);
    put_be16(&mut footer, footer::CYLS, cyls);
    footer[footer::HEADS] = heads;
    footer[footer::SECS_PER_CYL] = secs_per_cyl;
    put_be32(&mut footer, footer::TYPE, disk_type);
    footer[footer::UUID..footer::UUID + 16].copy_from_slice(&uuid_generate()?);
    let sum = vpc_checksum(&footer);
    put_be32(&mut footer, footer::CHECKSUM, sum);

    if disk_type == VHD_DYNAMIC {
        create_dynamic_disk(&node, &footer, total_sectors)
            .map_err(|_| Error::generic("Unable to create or write VHD header"))
    } else {
        create_fixed_disk(&node, &footer, total_size)
    }
}

/// `vpc_co_create_opts()`: `qemu-img create -f vpc`.
fn vpc_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let size = take_size(options, "size")?.unwrap_or(0);
    let subformat = take_str(options, "subformat");
    let force_size = take_bool(options, "force_size")?;

    // The protocol layer first, as QEMU does, even when the options turn out bad below.
    let graph = BlockGraph::new();
    graph.create_file(filename, options)?;
    let blk = graph.open_protocol_blk(filename)?;
    let node = blk.root().expect("a new backend has its node");

    let subformat = match subformat {
        Some(s) => Some(BlockdevVpcSubformat::from_name(&s).ok_or_else(|| {
            Error::generic(format!("Parameter 'subformat' does not accept value '{s}'"))
        })?),
        None => None,
    };
    // Silently round up the size, and to the geometry unless the size is forced.
    let mut size = size.div_ceil(BDRV_SECTOR_SIZE) * BDRV_SECTOR_SIZE;
    if !force_size.unwrap_or(false) {
        let (_, total_sectors) = calculate_rounded_image_size(size, false)?;
        size = total_sectors * BDRV_SECTOR_SIZE;
    }
    do_create(
        &graph,
        BlockdevCreateOptionsVpc {
            file: BlockdevRef::Reference(node.name.clone()),
            size,
            subformat,
            force_size,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe() {
        assert_eq!(vpc_probe(b"conectix\0\0", None), 100);
        assert_eq!(vpc_probe(b"conecti", None), 0);
        assert_eq!(vpc_probe(b"cxsparse", None), 0);
    }

    #[test]
    fn checksum() {
        assert_eq!(vpc_checksum(&[]), 0xffff_ffff);
        assert_eq!(vpc_checksum(&[1, 2, 3]), !6);
    }

    #[test]
    fn geometry() {
        // 10 MiB: 20480 sectors.
        assert_eq!(calculate_geometry(20480), (301, 4, 17));
        let ((c, h, s), t) = calculate_rounded_image_size(10 << 20, false).unwrap();
        assert_eq!((c, h, s), (302, 4, 17));
        assert_eq!(t, 20536);
        // Large disks use 255 sectors per track.
        assert_eq!(calculate_geometry(65535 * 16 * 63).2, 255);
        // Forced sizes keep their size and get the maximum geometry.
        let (g, t) = calculate_rounded_image_size(10 << 20, true).unwrap();
        assert_eq!(g, (65535, 16, 255));
        assert_eq!(t, 20480);
        assert_eq!(
            calculate_rounded_image_size(3 << 40, true).unwrap_err().message(),
            "Disk size is too large, max size is 2040 GiB"
        );
    }
}
