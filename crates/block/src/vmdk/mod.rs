// SPDX-License-Identifier: GPL-2.0-or-later

//! The `vmdk` format driver from block/vmdk.c: VMware virtual disks.
//!
//! A VMDK image is a text descriptor plus one or more extents. The descriptor names the
//! create type, the content IDs (`CID` and `parentCID`) that tie an overlay to its parent,
//! the parent (`parentFileNameHint`) and one line per extent. A monolithic sparse or
//! stream-optimized image keeps the descriptor inside its only extent, at sector 1; the other
//! create types keep it in a small text file of its own and the data in separate extent
//! files. The extent types are:
//!
//! - `FLAT` and `VMFS`: a raw range of a file.
//! - `SPARSE` and `VMFSSPARSE`: a hosted sparse extent ("KDMV", VMDK4) or an ESX sparse
//!   extent ("COWD", VMDK3). Both map grains (clusters) through a grain directory (L1) and
//!   grain tables (L2). VMDK4 extents may keep a redundant grain directory, may mark grains
//!   as zeroed (`zeroed_grain`) and may hold deflate compressed grains behind a grain
//!   marker (`streamOptimized`, which also allows its header at the end of the file).
//! - `SESPARSE`: the ESXi space efficient sparse format, which is read only here as it is
//!   in QEMU.
//!
//! Writing to a sparse extent allocates grains at the end of its file. The first write after
//! opening an image gives it a new random `CID` in the descriptor.
//!
//! Differences from QEMU:
//!
//! - For a stream-optimized extent whose grain directory is at the end, QEMU reads the footer
//!   relative to the end of the descriptor file (`bs->file`) even when the extent is another
//!   file. Here it is read from the end of the extent's own file, which is the same for
//!   every image QEMU can make.
//! - QEMU copies the backing data for a newly allocated grain from the offset of the grain
//!   within its extent, which is the wrong place for any extent but the first. Here the copy
//!   comes from the guest offset of the grain.
//! - Where QEMU would go on with an uninitialised grain table position (a write to a grain
//!   whose grain table is not allocated, which no image QEMU makes has), this driver fails
//!   the request: `EINVAL` for data and `ENOTSUP`, so that plain zeroes get written, for a
//!   zero write.
//! - QEMU divides by zero or loops forever on a sparse extent whose grain size is 0. Such an
//!   extent is refused here with "Invalid granularity, image may be corrupt".
//! - Options for the extent files (`extents.N.*`) are not taken, since the typed
//!   `BlockdevOptions` of `vmdk` have no member for them.
//! - QEMU registers a migration blocker for every open vmdk node; there is no migration here.
//! - An image without extents (a descriptor whose extent lines are all skipped) gives
//!   `ENOTSUP` from `get_info` where QEMU fails an assertion.
//! - A lookup in an extent whose grains per grain directory entry wrap to 0 in 32 bits
//!   fails the request here where QEMU divides by zero. VMDK4 extents like that are refused
//!   on open in both.
//! - When writing the backup grain directory of a new sparse extent fails, QEMU sets the
//!   error but still reports success. Here the create fails with that error.

mod create;
mod io;
mod open;
#[cfg(test)]
mod tests;

use std::io as stdio;
use std::sync::{Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType};
use ruvm_qapi::types::{
    ImageInfoSpecific, ImageInfoSpecificU, ImageInfoSpecificVmdk, ImageInfoSpecificVmdkWrapper,
    VmdkExtentInfo,
};

use crate::drivers::DriverDef;
use crate::node::{
    BlockDriverInfo, BlockLimits, BlockStatus, CheckResult, Driver, Node, ReopenState, errno,
};

/// `bdrv_vmdk`.
pub(crate) static VMDK: DriverDef = DriverDef::format("vmdk", open::vmdk_open)
    .with_probe(vmdk_probe)
    .with_backing()
    .with_create_opts(create::vmdk_co_create_opts)
    .with_create_opts_list(&VMDK_CREATE_OPTS)
    .with_create(create::vmdk_co_create);

/// `vmdk_create_opts`, in QEMU's order.
static VMDK_CREATE_OPTS: [QemuOptDesc; 9] = [
    QemuOptDesc::new("size", QemuOptType::Size).help("Virtual disk size"),
    QemuOptDesc::new("adapter_type", QemuOptType::String)
        .help("Virtual adapter type, can be one of ide (default), lsilogic, buslogic or legacyESX"),
    QemuOptDesc::new("backing_file", QemuOptType::String).help("File name of a base image"),
    QemuOptDesc::new("backing_fmt", QemuOptType::String).help("Must be 'vmdk' if present"),
    QemuOptDesc::new("compat6", QemuOptType::Bool)
        .help("VMDK version 6 image")
        .default_value("off"),
    QemuOptDesc::new("hwversion", QemuOptType::String)
        .help("VMDK hardware version")
        .default_value("undefined"),
    QemuOptDesc::new("toolsversion", QemuOptType::String).help("VMware guest tools version"),
    QemuOptDesc::new("subformat", QemuOptType::String).help(
        "VMDK flat extent format, can be one of {monolithicSparse (default) | monolithicFlat | \
         twoGbMaxExtentSparse | twoGbMaxExtentFlat | streamOptimized} ",
    ),
    QemuOptDesc::new("zeroed_grain", QemuOptType::Bool)
        .help("Enable efficient zero writes using the zeroed-grain GTE feature"),
];

/// "COWD", big-endian.
pub(super) const VMDK3_MAGIC: u32 = u32::from_be_bytes(*b"COWD");
/// "KDMV", big-endian.
pub(super) const VMDK4_MAGIC: u32 = u32::from_be_bytes(*b"KDMV");
pub(super) const VMDK4_COMPRESSION_DEFLATE: u16 = 1;
pub(super) const VMDK4_FLAG_NL_DETECT: u32 = 1 << 0;
pub(super) const VMDK4_FLAG_RGD: u32 = 1 << 1;
/// Zeroed-grain enable bit.
pub(super) const VMDK4_FLAG_ZERO_GRAIN: u32 = 1 << 2;
pub(super) const VMDK4_FLAG_COMPRESS: u32 = 1 << 16;
pub(super) const VMDK4_FLAG_MARKER: u32 = 1 << 17;
pub(super) const VMDK4_GD_AT_END: u64 = u64::MAX;

pub(super) const VMDK_EXTENT_MAX_SECTORS: u64 = 1 << 32;

pub(super) const VMDK_GTE_ZEROED: u32 = 0x1;

pub(super) const SECTOR_SIZE: u64 = 512;
/// 20 sectors of 512 bytes each.
pub(super) const DESC_SIZE: usize = 20 * 512;

pub(super) const L2_CACHE_SIZE: usize = 16;

/// `MARKER_END_OF_STREAM`.
pub(super) const MARKER_END_OF_STREAM: u32 = 0;
/// `MARKER_FOOTER`.
pub(super) const MARKER_FOOTER: u32 = 3;

/// The size of `VmdkGrainMarker` without its data.
pub(super) const GRAIN_MARKER_SIZE: usize = 12;

pub(super) fn le16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

pub(super) fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

pub(super) fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().expect("8 bytes"))
}

pub(super) fn be32(b: &[u8], off: usize) -> u32 {
    u32::from_be_bytes(b[off..off + 4].try_into().expect("4 bytes"))
}

/// QEMU's `ROUND_UP()`, which masks and so only works for powers of two; this keeps the
/// result it gives for other divisors, 0 included.
pub(super) fn round_up_mask(n: u64, d: u64) -> u64 {
    n.wrapping_add(d).wrapping_sub(1) & d.wrapping_neg()
}

/// `vmdk_probe()`.
fn vmdk_probe(buf: &[u8], _filename: Option<&str>) -> i32 {
    if buf.len() < 4 {
        return 0;
    }
    let magic = be32(buf, 0);
    if magic == VMDK3_MAGIC || magic == VMDK4_MAGIC {
        return 100;
    }
    let end = buf.len();
    let mut p = 0;
    while p < end {
        if buf[p] == b'#' {
            // Skip a comment line.
            while p < end && buf[p] != b'\n' {
                p += 1;
            }
            p += 1;
            continue;
        }
        if buf[p] == b' ' {
            while p < end && buf[p] == b' ' {
                p += 1;
            }
            // Skip '\r' if windows line endings are used.
            if p < end && buf[p] == b'\r' {
                p += 1;
            }
            // Only blank lines may come before the 'version=' line.
            if p == end || buf[p] != b'\n' {
                return 0;
            }
            p += 1;
            continue;
        }
        let rest = &buf[p..];
        for v in [b"version=1\n", b"version=2\n", b"version=3\n"] {
            if rest.starts_with(v) {
                return 100;
            }
        }
        for v in [b"version=1\r\n", b"version=2\r\n", b"version=3\r\n"] {
            if rest.starts_with(v) {
                return 100;
            }
        }
        return 0;
    }
    0
}

/// `VmdkExtent`.
#[derive(Debug, Default)]
pub(super) struct Extent {
    /// The name of the child that holds the extent: `file` or `extents.N`. QEMU keeps the
    /// `BdrvChild` and updates it on reopen; looking the child up by name does the same.
    pub child: String,
    pub flat: bool,
    pub compressed: bool,
    pub has_marker: bool,
    pub has_zero_grain: bool,
    pub sesparse: bool,
    pub sesparse_l2_tables_offset: u64,
    pub sesparse_clusters_offset: u64,
    /// 4, or 8 for seSparse.
    pub entry_size: u32,
    pub version: u32,
    pub sectors: u64,
    pub end_sector: u64,
    pub flat_start_offset: u64,
    pub l1_table_offset: u64,
    pub l1_backup_table_offset: u64,
    pub l1_table: Vec<u64>,
    pub l1_backup_table: Vec<u32>,
    pub l1_size: u32,
    pub l1_entry_sectors: u32,
    pub l2_size: u32,
    /// `L2_CACHE_SIZE` grain tables as they are on disk.
    pub l2_cache: Vec<u8>,
    pub l2_cache_offsets: [u32; L2_CACHE_SIZE],
    pub l2_cache_counts: [u32; L2_CACHE_SIZE],
    pub cluster_sectors: u64,
    pub next_cluster_sector: u64,
    /// The extent type from the descriptor, `SPARSE` for a monolithic image.
    pub ty: String,
}

/// The parts of `BDRVVmdkState` requests change, under `s->lock`.
#[derive(Debug)]
pub(super) struct State {
    /// Ordered by guest address.
    pub extents: Vec<Extent>,
    pub cid_updated: bool,
    pub cid_checked: bool,
}

/// `BDRVVmdkState`.
#[derive(Debug)]
pub(crate) struct VmdkDriver {
    pub(super) desc_offset: u64,
    pub(super) cid: u32,
    pub(super) parent_cid: u32,
    pub(super) create_type: String,
    /// `bs->total_sectors`: the end of the last extent.
    pub(super) total_sectors: u64,
    pub(super) lock: Mutex<State>,
}

impl VmdkDriver {
    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        self.lock.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The node of the child called `name`, or `EIO` when there is none.
pub(super) fn child_node(bs: &Node, name: &str) -> stdio::Result<std::sync::Arc<Node>> {
    bs.child(name).map(|c| c.node).ok_or_else(|| errno(libc::EIO))
}

/// `vmdk_extents_type_eq()`.
fn extents_type_eq(a: &Extent, b: &Extent) -> bool {
    a.flat == b.flat
        && a.compressed == b.compressed
        && (a.flat || a.cluster_sectors == b.cluster_sectors)
}

impl Driver for VmdkDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> stdio::Result<()> {
        self.co_preadv(bs, offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> stdio::Result<()> {
        let mut s = self.state();
        self.pwritev(bs, &mut s, offset, buf.len() as u64, Some(buf), false, false)
    }

    /// `vmdk_co_pwrite_zeroes()`: a dry run first, since writing zeroes fails unless the
    /// request covers whole grains with the zeroed-grain feature.
    fn pwrite_zeroes(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        _may_unmap: bool,
    ) -> stdio::Result<()> {
        let mut s = self.state();
        self.pwritev(bs, &mut s, offset, bytes, None, true, true)?;
        self.pwritev(bs, &mut s, offset, bytes, None, true, false)
    }

    /// `vmdk_co_pwritev_compressed()`.
    fn pwrite_compressed(&self, bs: &Node, offset: u64, buf: &[u8]) -> Option<stdio::Result<()>> {
        if buf.is_empty() {
            // The caller writes 0 bytes to signal the end; align the files to a sector then.
            let names: Vec<String> = self.state().extents.iter().map(|e| e.child.clone()).collect();
            for name in names {
                let r = child_node(bs, &name).and_then(|file| {
                    let length = file.getlength()?.div_ceil(SECTOR_SIZE) * SECTOR_SIZE;
                    file.truncate_full(length as i64, false, ruvm_qapi::types::PreallocMode::Off, 0)
                        .map_err(to_io)
                });
                if let Err(e) = r {
                    return Some(Err(e));
                }
            }
            return Some(Ok(()));
        }
        Some(self.pwrite(bs, offset, buf))
    }

    fn can_compress(&self) -> bool {
        true
    }

    fn getlength(&self, _bs: &Node) -> stdio::Result<u64> {
        Ok(self.total_sectors * SECTOR_SIZE)
    }

    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<stdio::Result<BlockStatus>> {
        Some(self.co_block_status(bs, offset, bytes))
    }

    /// `vmdk_refresh_limits()`.
    fn refresh_limits(&self, _bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        for e in &self.state().extents {
            if !e.flat {
                let c = (e.cluster_sectors * SECTOR_SIZE).min(u64::from(u32::MAX)) as u32;
                bl.pwrite_zeroes_alignment = bl.pwrite_zeroes_alignment.max(c);
            }
        }
        Ok(())
    }

    /// `vmdk_reopen_prepare()`: the extents follow their children by name, nothing to do.
    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    /// `vmdk_co_get_info()`.
    fn get_info(&self, _bs: &Node) -> Option<stdio::Result<BlockDriverInfo>> {
        let s = self.state();
        let Some(first) = s.extents.first() else {
            return Some(Err(errno(libc::ENOTSUP)));
        };
        // See if there are several extents of different kinds.
        if s.extents[1..].iter().any(|e| !extents_type_eq(first, e)) {
            return Some(Err(errno(libc::ENOTSUP)));
        }
        let mut bdi =
            BlockDriverInfo { needs_compressed_writes: first.compressed, ..Default::default() };
        if !first.flat {
            bdi.cluster_size = first.cluster_sectors * SECTOR_SIZE;
        }
        Some(Ok(bdi))
    }

    /// `vmdk_get_specific_info()`.
    fn get_specific_info(&self, bs: &Node) -> Result<Option<ImageInfoSpecific>> {
        let s = self.state();
        let mut extents = Vec::with_capacity(s.extents.len());
        for e in &s.extents {
            // vmdk_get_extent_info()
            let filename = match bs.child(&e.child) {
                Some(c) => {
                    c.node.refresh_filename();
                    c.node.filename().unwrap_or_default()
                }
                None => String::new(),
            };
            extents.push(VmdkExtentInfo {
                filename,
                format: e.ty.clone(),
                virtual_size: (e.sectors.wrapping_mul(SECTOR_SIZE)) as i64,
                cluster_size: (!e.flat).then_some((e.cluster_sectors * SECTOR_SIZE) as i64),
                compressed: e.compressed.then_some(true),
            });
        }
        Ok(Some(ImageInfoSpecific {
            u: ImageInfoSpecificU::Vmdk(ImageInfoSpecificVmdkWrapper {
                data: ImageInfoSpecificVmdk {
                    create_type: self.create_type.clone(),
                    cid: i64::from(self.cid),
                    parent_cid: i64::from(self.parent_cid),
                    extents,
                },
            }),
        }))
    }

    /// `vmdk_co_get_allocated_file_size()`.
    fn get_allocated_file_size(&self, bs: &Node) -> Option<stdio::Result<u64>> {
        let names: Vec<String> = self.state().extents.iter().map(|e| e.child.clone()).collect();
        let r = (|| {
            let mut ret = bs.file().allocated_file_size()?;
            for name in names {
                if name == "file" {
                    continue;
                }
                ret += child_node(bs, &name)?.allocated_file_size()?;
            }
            Ok(ret)
        })();
        Some(r)
    }

    /// `vmdk_has_zero_init()`: not when a flat extent's storage lacks it.
    fn has_zero_init(&self, bs: &Node) -> Option<bool> {
        let s = self.state();
        for e in s.extents.iter().filter(|e| e.flat) {
            if let Some(c) = bs.child(&e.child) {
                if !c.node.has_zero_init() {
                    return Some(false);
                }
            }
        }
        Some(true)
    }

    /// `vmdk_co_check()`.
    fn check(&self, bs: &Node, fix: u32) -> Option<Result<CheckResult>> {
        if fix != 0 {
            return None;
        }
        Some(self.co_check(bs))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

/// The errno of a block layer error, for the callbacks that return `-errno`.
pub(super) fn to_io(e: Error) -> stdio::Error {
    let code = std::error::Error::source(&e)
        .and_then(|c| c.downcast_ref::<stdio::Error>())
        .and_then(stdio::Error::raw_os_error);
    match code {
        Some(c) => errno(c),
        None => stdio::Error::other(e.message().to_string()),
    }
}
