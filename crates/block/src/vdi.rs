// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vdi` format driver from block/vdi.c: VirtualBox disk images.
//!
//! A VDI image starts with a 512 byte header, then the block map, then the data blocks of
//! 1 MiB each. The block map entry of a guest block is the index of its data block in the file,
//! or `VDI_UNALLOCATED` / `VDI_DISCARDED` for blocks that read as zeros. A dynamic image
//! allocates data blocks at the end of the file on the first write; a static image has all of
//! them from the start. Images with a parent (differencing images) are refused, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - QEMU registers a migration blocker for every open vdi node; there is no migration here.
//! - `vdi_co_check()` with a fix mode returns `-ENOTSUP`, which qemu-img reports as "This image
//!   format does not support checks"; here the driver reports the same by answering `None`.
//! - The header is kept as the bytes read from the file and only `disk_size` and
//!   `blocks_allocated` are patched in when it is written back. QEMU converts the whole header
//!   to host order and back, which gives the same bytes.

use std::io;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::types::{
    BlockdevCreateOptionsU, BlockdevCreateOptionsVdi, BlockdevOptionsU, BlockdevRef, PreallocMode,
};

use crate::drivers::{DriverDef, OpenArgs};
use crate::graph::BlockGraph;
use crate::imgopts::{take_bool, take_size};
use crate::node::{
    BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RECURSE, BDRV_BLOCK_ZERO,
    BDRV_CHILD_IMAGE, BDRV_CHILD_PRIMARY, BDRV_SECTOR_SIZE, BlockDriverInfo, BlockStatus,
    CheckResult, Driver, Node, ReopenState,
};

/// `bdrv_vdi`.
pub(crate) static VDI: DriverDef = DriverDef::format("vdi", vdi_open_node)
    .with_probe(vdi_probe)
    .with_create_opts(vdi_co_create_opts)
    .with_create_opts_list(&VDI_CREATE_OPTS)
    .with_create(vdi_co_create);

/// `vdi_create_opts`. The `cluster_size` option only exists in builds with
/// `CONFIG_VDI_BLOCK_SIZE`, which is off by default.
static VDI_CREATE_OPTS: [QemuOptDesc; 2] = [
    QemuOptDesc::new("size", QemuOptType::Size).help("Virtual disk size"),
    QemuOptDesc::new("static", QemuOptType::Bool)
        .help("VDI static (pre-allocated) image")
        .default_value("off"),
];

const SECTOR_SIZE: u64 = 512;
const VDI_TEXT: &[u8] = b"<<< QEMU VM Virtual Disk Image >>>\n";
const VDI_SIGNATURE: u32 = 0xbeda_107f;
const VDI_VERSION_1_1: u32 = 0x0001_0001;
const VDI_TYPE_DYNAMIC: u32 = 1;
const VDI_TYPE_STATIC: u32 = 2;

/// A block that was never written.
const VDI_UNALLOCATED: u32 = 0xffff_ffff;
/// A block that was discarded, which reads as zeros too.
#[allow(dead_code)]
const VDI_DISCARDED: u32 = 0xffff_fffe;

fn vdi_is_allocated(x: u32) -> bool {
    x < 0xffff_fffe
}

/// The size of a data block, the only one QEMU supports.
const DEFAULT_CLUSTER_SIZE: u32 = 1 << 20;
/// The most blocks a bmap can have, so that it stays within 2 GiB.
const VDI_BLOCKS_IN_IMAGE_MAX: u32 = 0x1fff_ff80;
const VDI_DISK_SIZE_MAX: u64 = VDI_BLOCKS_IN_IMAGE_MAX as u64 * DEFAULT_CLUSTER_SIZE as u64;

const HEADER_SIZE: usize = 512;

/// Offsets into `VdiHeader`, all fields little-endian.
mod hdr {
    pub(super) const TEXT: usize = 0;
    pub(super) const SIGNATURE: usize = 64;
    pub(super) const VERSION: usize = 68;
    pub(super) const HEADER_SIZE: usize = 72;
    pub(super) const IMAGE_TYPE: usize = 76;
    pub(super) const OFFSET_BMAP: usize = 340;
    pub(super) const OFFSET_DATA: usize = 344;
    pub(super) const SECTOR_SIZE: usize = 360;
    pub(super) const DISK_SIZE: usize = 368;
    pub(super) const BLOCK_SIZE: usize = 376;
    pub(super) const BLOCKS_IN_IMAGE: usize = 384;
    pub(super) const BLOCKS_ALLOCATED: usize = 388;
    pub(super) const UUID_IMAGE: usize = 392;
    pub(super) const UUID_LAST_SNAP: usize = 408;
    pub(super) const UUID_LINK: usize = 424;
    pub(super) const UUID_PARENT: usize = 440;
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

fn put_le32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn put_le64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// `vdi_probe()`.
fn vdi_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() >= HEADER_SIZE && le32(buf, hdr::SIGNATURE) == VDI_SIGNATURE { 100 } else { 0 }
}

/// What writes change, under `s->bmap_lock`.
#[derive(Debug)]
struct VdiState {
    /// The block map in host order, padded to whole sectors.
    bmap: Vec<u32>,
    blocks_allocated: u32,
}

/// `BDRVVdiState`.
#[derive(Debug)]
pub(crate) struct VdiDriver {
    /// The header as read from the file.
    header: [u8; HEADER_SIZE],
    /// The disk size rounded up to whole sectors.
    disk_size: u64,
    image_type: u32,
    offset_data: u64,
    block_size: u32,
    bmap_sector: u64,
    blocks_in_image: u32,
    lock: RwLock<VdiState>,
}

/// `vdi_open()`.
fn vdi_open_node(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Vdi(o) = opts else { unreachable!("vdi driver with other options") };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_IMAGE | BDRV_CHILD_PRIMARY)?;
    Ok(Box::new(vdi_open(&file)?))
}

fn vdi_open(file: &Node) -> Result<VdiDriver> {
    let mut header = [0u8; HEADER_SIZE];
    file.pread(0, &mut header).map_err(|e| {
        let name = file.filename().unwrap_or_default();
        Error::from_io(format!("Could not open '{name}'"), e)
    })?;

    let mut disk_size = le64(&header, hdr::DISK_SIZE);
    if disk_size > VDI_DISK_SIZE_MAX {
        return Err(Error::generic(format!(
            "Unsupported VDI image size (size is 0x{disk_size:x}, max supported is \
             0x{VDI_DISK_SIZE_MAX:x})"
        )));
    }
    // 'VBoxManage convertfromraw' can create images with odd disk sizes. They are accepted,
    // with the size rounded up to whole sectors.
    disk_size = disk_size.div_ceil(SECTOR_SIZE) * SECTOR_SIZE;

    let signature = le32(&header, hdr::SIGNATURE);
    let version = le32(&header, hdr::VERSION);
    let offset_bmap = le32(&header, hdr::OFFSET_BMAP);
    let offset_data = le32(&header, hdr::OFFSET_DATA);
    let sector_size = le32(&header, hdr::SECTOR_SIZE);
    let block_size = le32(&header, hdr::BLOCK_SIZE);
    let blocks_in_image = le32(&header, hdr::BLOCKS_IN_IMAGE);
    let room = u64::from(blocks_in_image) * u64::from(block_size);
    let is_null = |off: usize| header[off..off + 16].iter().all(|&b| b == 0);
    if signature != VDI_SIGNATURE {
        return Err(Error::generic(format!(
            "Image not in VDI format (bad signature {signature:08x})"
        )));
    } else if version != VDI_VERSION_1_1 {
        return Err(Error::generic(format!(
            "unsupported VDI image (version {}.{})",
            version >> 16,
            version & 0xffff
        )));
    } else if u64::from(offset_bmap) % SECTOR_SIZE != 0 {
        return Err(Error::generic(format!(
            "unsupported VDI image (unaligned block map offset 0x{offset_bmap:x})"
        )));
    } else if u64::from(offset_data) % SECTOR_SIZE != 0 {
        return Err(Error::generic(format!(
            "unsupported VDI image (unaligned data offset 0x{offset_data:x})"
        )));
    } else if u64::from(sector_size) != SECTOR_SIZE {
        return Err(Error::generic(format!(
            "unsupported VDI image (sector size {sector_size} is not {SECTOR_SIZE})"
        )));
    } else if block_size != DEFAULT_CLUSTER_SIZE {
        return Err(Error::generic(format!(
            "unsupported VDI image (block size {block_size} is not {DEFAULT_CLUSTER_SIZE})"
        )));
    } else if disk_size > room {
        return Err(Error::generic(format!(
            "unsupported VDI image (disk size {disk_size}, image bitmap has room for {room})"
        )));
    } else if !is_null(hdr::UUID_LINK) {
        return Err(Error::generic("unsupported VDI image (non-NULL link UUID)"));
    } else if !is_null(hdr::UUID_PARENT) {
        return Err(Error::generic("unsupported VDI image (non-NULL parent UUID)"));
    } else if blocks_in_image > VDI_BLOCKS_IN_IMAGE_MAX {
        return Err(Error::generic(format!(
            "unsupported VDI image (too many blocks {blocks_in_image}, max is \
             {VDI_BLOCKS_IN_IMAGE_MAX})"
        )));
    }

    let bmap_sectors = (u64::from(blocks_in_image) * 4).div_ceil(SECTOR_SIZE);
    let mut raw = vec![0u8; (bmap_sectors * SECTOR_SIZE) as usize];
    file.pread(u64::from(offset_bmap), &mut raw).map_err(|e| {
        let name = file.filename().unwrap_or_default();
        Error::from_io(format!("Could not open '{name}'"), e)
    })?;
    let bmap = raw.chunks_exact(4).map(|c| le32(c, 0)).collect();

    Ok(VdiDriver {
        header,
        disk_size,
        image_type: le32(&header, hdr::IMAGE_TYPE),
        offset_data: u64::from(offset_data),
        block_size,
        bmap_sector: u64::from(offset_bmap) / SECTOR_SIZE,
        blocks_in_image,
        lock: RwLock::new(VdiState {
            bmap,
            blocks_allocated: le32(&header, hdr::BLOCKS_ALLOCATED),
        }),
    })
}

impl VdiDriver {
    fn read_state(&self) -> RwLockReadGuard<'_, VdiState> {
        self.lock.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write_state(&self) -> RwLockWriteGuard<'_, VdiState> {
        self.lock.write().unwrap_or_else(|e| e.into_inner())
    }

    fn data_offset(&self, entry: u32) -> u64 {
        self.offset_data + u64::from(entry) * u64::from(self.block_size)
    }

    /// The block index, the offset in the block and the length of the part of a request at
    /// `offset` with `left` bytes to go that is in one block.
    fn split(&self, offset: u64, left: usize) -> (usize, u64, usize) {
        let bs = u64::from(self.block_size);
        let in_block = offset % bs;
        ((offset / bs) as usize, in_block, left.min((bs - in_block) as usize))
    }
}

impl Driver for VdiDriver {
    /// `vdi_co_preadv()`.
    fn pread(&self, bs: &Node, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let file = bs.file();
        let mut done = 0;
        while done < buf.len() {
            let (index, in_block, n) = self.split(offset, buf.len() - done);
            let entry = self.read_state().bmap[index];
            let chunk = &mut buf[done..done + n];
            if vdi_is_allocated(entry) {
                file.pread(self.data_offset(entry) + in_block, chunk)?;
            } else {
                // Not allocated, zeros.
                chunk.fill(0);
            }
            done += n;
            offset += n as u64;
        }
        Ok(())
    }

    /// `vdi_co_pwritev()`.
    fn pwrite(&self, bs: &Node, mut offset: u64, buf: &[u8]) -> io::Result<()> {
        let file = bs.file();
        let mut first_last: Option<(usize, usize)> = None;
        let mut done = 0;
        while done < buf.len() {
            let (index, in_block, n) = self.split(offset, buf.len() - done);
            let data = &buf[done..done + n];
            let mut entry = self.read_state().bmap[index];
            if !vdi_is_allocated(entry) {
                let mut s = self.write_state();
                entry = s.bmap[index];
                if !vdi_is_allocated(entry) {
                    // Allocate a new block and write all of it, under the write lock so that
                    // no partial write of the same block can overlap.
                    entry = s.blocks_allocated;
                    s.bmap[index] = entry;
                    s.blocks_allocated += 1;
                    first_last = Some(first_last.map_or((index, index), |(f, _)| (f, index)));
                    let mut block = vec![0u8; self.block_size as usize];
                    block[in_block as usize..in_block as usize + n].copy_from_slice(data);
                    file.pwrite(self.data_offset(entry), &block)?;
                    done += n;
                    offset += n as u64;
                    continue;
                }
                // A concurrent allocation did the work.
            }
            file.pwrite(self.data_offset(entry) + in_block, data)?;
            done += n;
            offset += n as u64;
        }

        let Some((first, last)) = first_last else { return Ok(()) };
        // One or more new blocks: write the header and the changed sectors of the bmap.
        let (header, bmap) = {
            let s = self.read_state();
            let mut header = self.header;
            put_le64(&mut header, hdr::DISK_SIZE, self.disk_size);
            put_le32(&mut header, hdr::BLOCKS_ALLOCATED, s.blocks_allocated);
            let per_sector = SECTOR_SIZE as usize / 4;
            let (first, last) = (first / per_sector, last / per_sector);
            let bmap: Vec<u8> = s.bmap[first * per_sector..(last + 1) * per_sector]
                .iter()
                .flat_map(|e| e.to_le_bytes())
                .collect();
            (header, (first as u64, bmap))
        };
        file.pwrite(0, &header)?;
        file.pwrite((self.bmap_sector + bmap.0) * SECTOR_SIZE, &bmap.1)
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.disk_size)
    }

    /// `vdi_co_block_status()`.
    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let (index, in_block, _) = self.split(offset, 0);
        let entry = self.read_state().bmap[index];
        let pnum = (u64::from(self.block_size) - in_block).min(bytes);
        if !vdi_is_allocated(entry) {
            return Some(Ok(BlockStatus { ret: BDRV_BLOCK_ZERO, pnum, map: 0, file: None }));
        }
        let recurse = if self.image_type == VDI_TYPE_STATIC { BDRV_BLOCK_RECURSE } else { 0 };
        Some(Ok(BlockStatus {
            ret: BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | recurse,
            pnum,
            map: self.data_offset(entry) + in_block,
            file: Some(bs.file()),
        }))
    }

    /// `vdi_co_check()`: the bmap must not use a data block twice or one past the end, and
    /// `blocks_allocated` must match. Nothing can be fixed.
    fn check(&self, _bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        if fix != 0 {
            return None;
        }
        let s = self.read_state();
        let mut res = CheckResult::default();
        let mut blocks_allocated = 0u32;
        let mut used = vec![VDI_UNALLOCATED; self.blocks_in_image as usize];
        for block in 0..self.blocks_in_image {
            let entry = s.bmap[block as usize];
            if !vdi_is_allocated(entry) {
                continue;
            }
            if entry < self.blocks_in_image {
                blocks_allocated += 1;
                if !vdi_is_allocated(used[entry as usize]) {
                    used[entry as usize] = entry;
                } else {
                    eprintln!("ERROR: block index {} also used by {}", used[entry as usize], entry);
                    res.corruptions += 1;
                }
            } else {
                eprintln!("ERROR: block index {block} too large, is {entry}");
                res.corruptions += 1;
            }
        }
        if blocks_allocated != s.blocks_allocated {
            eprintln!(
                "ERROR: allocated blocks mismatch, is {blocks_allocated}, should be {}",
                s.blocks_allocated
            );
            res.corruptions += 1;
        }
        Some(Ok(res))
    }

    /// `vdi_co_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<io::Result<BlockDriverInfo>> {
        Some(Ok(BlockDriverInfo {
            cluster_size: u64::from(self.block_size),
            ..BlockDriverInfo::default()
        }))
    }

    /// `vdi_make_empty()`: not written in QEMU either, and it must succeed.
    fn make_empty(&self, _bs: &Node) -> Option<io::Result<()>> {
        Some(Ok(()))
    }

    /// `vdi_has_zero_init()`.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        if self.image_type == VDI_TYPE_STATIC {
            Some(bs.file().has_zero_init())
        } else {
            Some(true)
        }
    }

    /// `vdi_reopen_prepare()`: nothing to check.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// `qemu_uuid_generate()` and `qemu_uuid_bswap()`: a random version 4 UUID with its first
/// three fields little-endian, as VirtualBox stores them.
fn uuid_generate_le() -> Result<[u8; 16]> {
    let mut u = [0u8; 16];
    ruvm_crypto::random::random_bytes(&mut u)?;
    u[6] = (u[6] & 0x0f) | 0x40;
    u[8] = (u[8] & 0x3f) | 0x80;
    u[0..4].reverse();
    u[4..6].reverse();
    u[6..8].reverse();
    Ok(u)
}

/// `vdi_co_create()`: `blockdev-create` with `driver: vdi`.
fn vdi_co_create(graph: &BlockGraph, options: BlockdevCreateOptionsU) -> Result<()> {
    let BlockdevCreateOptionsU::Vdi(o) = options else {
        unreachable!("vdi driver with other create options")
    };
    vdi_co_do_create(graph, o, DEFAULT_CLUSTER_SIZE)
}

/// `vdi_co_do_create()`.
fn vdi_co_do_create(
    graph: &BlockGraph,
    o: BlockdevCreateOptionsVdi,
    block_size: u32,
) -> Result<()> {
    let bytes = o.size;
    let image_type = match o.preallocation.unwrap_or(PreallocMode::Off) {
        PreallocMode::Off => VDI_TYPE_DYNAMIC,
        PreallocMode::Metadata => VDI_TYPE_STATIC,
        _ => return Err(Error::generic("Preallocation mode not supported for vdi")),
    };
    if bytes > VDI_DISK_SIZE_MAX {
        return Err(Error::generic(format!(
            "Unsupported VDI image size (size is 0x{bytes:x}, max supported is \
             0x{VDI_DISK_SIZE_MAX:x})"
        )));
    }

    let blk = graph.open_create_blk(o.file)?;
    let node = blk.root().expect("a new backend has its node");

    // Enough blocks for the whole disk.
    let blocks = bytes.div_ceil(u64::from(block_size)) as u32;
    let bmap_size = (u64::from(blocks) * 4).div_ceil(SECTOR_SIZE) * SECTOR_SIZE;

    let mut header = [0u8; HEADER_SIZE];
    header[hdr::TEXT..hdr::TEXT + VDI_TEXT.len()].copy_from_slice(VDI_TEXT);
    put_le32(&mut header, hdr::SIGNATURE, VDI_SIGNATURE);
    put_le32(&mut header, hdr::VERSION, VDI_VERSION_1_1);
    put_le32(&mut header, hdr::HEADER_SIZE, 0x180);
    put_le32(&mut header, hdr::IMAGE_TYPE, image_type);
    put_le32(&mut header, hdr::OFFSET_BMAP, 0x200);
    put_le32(&mut header, hdr::OFFSET_DATA, 0x200 + bmap_size as u32);
    put_le32(&mut header, hdr::SECTOR_SIZE, SECTOR_SIZE as u32);
    put_le64(&mut header, hdr::DISK_SIZE, bytes);
    put_le32(&mut header, hdr::BLOCK_SIZE, block_size);
    put_le32(&mut header, hdr::BLOCKS_IN_IMAGE, blocks);
    if image_type == VDI_TYPE_STATIC {
        put_le32(&mut header, hdr::BLOCKS_ALLOCATED, blocks);
    }
    header[hdr::UUID_IMAGE..hdr::UUID_IMAGE + 16].copy_from_slice(&uuid_generate_le()?);
    header[hdr::UUID_LAST_SNAP..hdr::UUID_LAST_SNAP + 16].copy_from_slice(&uuid_generate_le()?);
    // uuid_link and uuid_parent stay zero.
    node.pwrite(0, &header).map_err(|_| Error::generic("Error writing header"))?;
    let mut offset = HEADER_SIZE as u64;

    if bmap_size > 0 {
        let mut bmap = vec![0u8; bmap_size as usize];
        for i in 0..blocks {
            let v = if image_type == VDI_TYPE_STATIC { i } else { VDI_UNALLOCATED };
            put_le32(&mut bmap, i as usize * 4, v);
        }
        node.pwrite(offset, &bmap).map_err(|_| Error::generic("Error writing bmap"))?;
        offset += bmap_size;
    }

    if image_type == VDI_TYPE_STATIC {
        let end = offset + u64::from(blocks) * u64::from(block_size);
        node.truncate_full(end as i64, false, PreallocMode::Off, 0)
            .map_err(|e| e.prepend("Failed to statically allocate file"))?;
    }
    Ok(())
}

/// `vdi_co_create_opts()`: `qemu-img create -f vdi`.
fn vdi_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    let is_static = take_bool(options, "static")?.unwrap_or(false);
    let size = take_size(options, "size")?.unwrap_or(0);

    // The protocol layer.
    let graph = BlockGraph::new();
    graph.create_file(filename, options)?;
    let blk = graph.open_protocol_blk(filename)?;
    let node = blk.root().expect("a new backend has its node");

    vdi_co_do_create(
        &graph,
        BlockdevCreateOptionsVdi {
            file: BlockdevRef::Reference(node.name.clone()),
            // Silently round up.
            size: size.div_ceil(BDRV_SECTOR_SIZE) * BDRV_SECTOR_SIZE,
            preallocation: is_static.then_some(PreallocMode::Metadata),
        },
        DEFAULT_CLUSTER_SIZE,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe() {
        let mut h = [0u8; HEADER_SIZE];
        assert_eq!(vdi_probe(&h, None), 0);
        put_le32(&mut h, hdr::SIGNATURE, VDI_SIGNATURE);
        assert_eq!(vdi_probe(&h, None), 100);
        assert_eq!(vdi_probe(&h[..511], None), 0);
    }

    #[test]
    fn allocated() {
        assert!(vdi_is_allocated(0));
        assert!(vdi_is_allocated(0xffff_fffd));
        assert!(!vdi_is_allocated(VDI_DISCARDED));
        assert!(!vdi_is_allocated(VDI_UNALLOCATED));
    }

    #[test]
    fn uuid_is_v4_in_guid_order() {
        let u = uuid_generate_le().unwrap();
        // The version nibble is in the high half of the third field, byte 7 once swapped.
        assert_eq!(u[7] >> 4, 4);
        assert_eq!(u[8] >> 6, 2);
    }
}
