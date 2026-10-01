// SPDX-License-Identifier: GPL-2.0-or-later

//! Reads, writes, block status and check for VMDK images.

use std::io;

use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};
use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlkdebugEvent;

use super::open::{cstr, read_cid, strstr};
use super::{
    DESC_SIZE, Extent, GRAIN_MARKER_SIZE, L2_CACHE_SIZE, SECTOR_SIZE, State,
    VMDK_EXTENT_MAX_SECTORS, VMDK_GTE_ZEROED, VmdkDriver, child_node, le32, le64,
};
use crate::node::{
    BDRV_BLOCK_COMPRESSED, BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RECURSE,
    BDRV_BLOCK_ZERO, BlockStatus, CheckResult, Node, errno,
};

/// What `get_cluster_offset()` finds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Lookup {
    /// `VMDK_OK`: the grain is at this byte offset of the extent file.
    Ok(u64),
    /// `VMDK_ERROR`.
    Error,
    /// `VMDK_UNALLOC`.
    Unalloc,
    /// `VMDK_ZEROED`.
    Zeroed,
}

/// `VmdkMetaData`: where the grain table entry of a grain is.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct MetaData {
    l1_index: u32,
    l2_index: u32,
    l2_offset: u32,
    /// The slot of the grain table in the L2 cache.
    cache_slot: usize,
    new_allocation: bool,
    /// Whether the fields above were filled in. QEMU leaves them uninitialised for flat
    /// extents and for grains whose grain table is not allocated.
    valid: bool,
}

/// `find_extent()`: the first extent from `hint` on that holds `sector_num`.
fn find_extent(extents: &[Extent], sector_num: u64, hint: usize) -> Option<usize> {
    (hint..extents.len()).find(|&i| sector_num < extents[i].end_sector)
}

/// `vmdk_find_offset_in_cluster()`.
fn find_offset_in_cluster(e: &Extent, offset: u64) -> u64 {
    let cluster_size = e.cluster_sectors * SECTOR_SIZE;
    let begin = (e.end_sector - e.sectors) * SECTOR_SIZE;
    offset.wrapping_sub(begin) % cluster_size
}

/// `vmdk_L2update()`: points the grain table entry `m` at `value`, in both grain tables.
fn l2_update(e: &mut Extent, file: &Node, m: &MetaData, value: u32) -> bool {
    let bytes = value.to_le_bytes();
    let at = u64::from(m.l2_offset) * 512 + u64::from(m.l2_index) * 4;
    file.debug_event(BlkdebugEvent::L2Update);
    if file.pwrite(at, &bytes).is_err() {
        return false;
    }
    if e.l1_backup_table_offset != 0 {
        let l2_offset = e.l1_backup_table[m.l1_index as usize];
        let at = u64::from(l2_offset) * 512 + u64::from(m.l2_index) * 4;
        if file.pwrite(at, &bytes).is_err() {
            return false;
        }
    }
    if file.flush().is_err() {
        return false;
    }
    let l2_size_bytes = e.l2_size as usize * e.entry_size as usize;
    let at = m.cache_slot * l2_size_bytes + m.l2_index as usize * 4;
    e.l2_cache[at..at + 4].copy_from_slice(&bytes);
    true
}

/// `vmdk_read_extent()`.
fn read_extent(
    e: &Extent,
    file: &Node,
    cluster_offset: u64,
    offset_in_cluster: u64,
    out: &mut [u8],
) -> io::Result<()> {
    if !e.compressed {
        file.debug_event(BlkdebugEvent::ReadAio);
        return file.pread(cluster_offset + offset_in_cluster, out);
    }
    let cluster_bytes = (e.cluster_sectors * 512) as usize;
    // Read two clusters in case the grain marker and the compressed data are larger than one.
    let mut cluster_buf = vec![0u8; cluster_bytes * 2];
    file.debug_event(BlkdebugEvent::ReadCompressed);
    file.pread(cluster_offset, &mut cluster_buf)?;
    let mut data: &[u8] = &cluster_buf[..cluster_bytes];
    if e.has_marker {
        let data_len = le32(&cluster_buf, 8) as usize;
        if data_len == 0 || data_len > cluster_bytes * 2 - GRAIN_MARKER_SIZE {
            return Err(errno(libc::EINVAL));
        }
        data = &cluster_buf[GRAIN_MARKER_SIZE..GRAIN_MARKER_SIZE + data_len];
    }
    let mut uncomp = vec![0u8; cluster_bytes];
    let mut d = Decompress::new(true);
    match d.decompress(data, &mut uncomp, FlushDecompress::Finish) {
        Ok(Status::StreamEnd) => {}
        _ => return Err(errno(libc::EINVAL)),
    }
    let buf_len = d.total_out();
    if offset_in_cluster + out.len() as u64 > buf_len {
        return Err(errno(libc::EINVAL));
    }
    let start = offset_in_cluster as usize;
    out.copy_from_slice(&uncomp[start..start + out.len()]);
    Ok(())
}

/// zlib's `compress()` into a buffer of `cap` bytes, `None` when it does not fit.
pub(super) fn zlib_compress(data: &[u8], cap: usize) -> Option<Vec<u8>> {
    let mut out = vec![0u8; cap];
    let mut c = Compress::new(Compression::default(), true);
    match c.compress(data, &mut out, FlushCompress::Finish) {
        Ok(Status::StreamEnd) => {
            out.truncate(c.total_out() as usize);
            Some(out)
        }
        _ => None,
    }
}

/// `vmdk_write_extent()`: `data` goes to `offset_in_cluster` of the grain at
/// `cluster_offset`; `offset` is the guest offset.
fn write_extent(
    e: &mut Extent,
    file: &Node,
    cluster_offset: u64,
    offset_in_cluster: u64,
    data: &[u8],
    offset: u64,
) -> io::Result<()> {
    let n_bytes = data.len() as u64;
    let cluster_size = e.cluster_sectors * SECTOR_SIZE;
    let compressed;
    let payload: &[u8] = if e.compressed {
        // Only whole grains, or the tail of the extent.
        if offset_in_cluster != 0
            || n_bytes > cluster_size
            || (n_bytes < cluster_size && offset + n_bytes != e.end_sector * SECTOR_SIZE)
        {
            return Err(errno(libc::EINVAL));
        }
        if !e.has_marker {
            return Err(errno(libc::EINVAL));
        }
        let Some(z) = zlib_compress(data, (cluster_size * 2) as usize) else {
            return Err(errno(libc::EINVAL));
        };
        if z.is_empty() {
            return Err(errno(libc::EINVAL));
        }
        let mut m = Vec::with_capacity(GRAIN_MARKER_SIZE + z.len());
        m.extend_from_slice(&(offset >> 9).to_le_bytes());
        m.extend_from_slice(&(z.len() as u32).to_le_bytes());
        m.extend_from_slice(&z);
        compressed = m;
        file.debug_event(BlkdebugEvent::WriteCompressed);
        &compressed
    } else {
        file.debug_event(BlkdebugEvent::WriteAio);
        data
    };

    let write_offset = cluster_offset + offset_in_cluster;
    let ret = file.pwrite(write_offset, payload);
    let write_end_sector = (write_offset + payload.len() as u64).div_ceil(SECTOR_SIZE);
    if e.compressed {
        e.next_cluster_sector = write_end_sector;
    } else {
        e.next_cluster_sector = e.next_cluster_sector.max(write_end_sector);
    }
    ret
}

impl VmdkDriver {
    /// `vmdk_is_cid_valid()`: whether the parent's `CID` is still the `parentCID` of this
    /// image. The answer is kept once it is yes.
    fn is_cid_valid(&self, bs: &Node, cid_checked: &mut bool) -> bool {
        if !*cid_checked {
            if let Some(b) = bs.backing() {
                let p = b.node;
                if p.driver_name != "vmdk" {
                    // A backing file in another format has no CID, so the overlay's parent
                    // CID cannot be right.
                    return false;
                }
                let Some(pd) = p.driver.as_any().and_then(|a| a.downcast_ref::<VmdkDriver>())
                else {
                    return false;
                };
                let Some(pfile) = p.child("file") else { return false };
                match read_cid(&pfile.node, pd.desc_offset, false) {
                    Ok(cur) if cur == self.parent_cid => {}
                    _ => return false,
                }
            }
        }
        *cid_checked = true;
        true
    }

    /// `vmdk_write_cid()`: puts `cid` into the descriptor.
    fn write_cid(&self, bs: &Node, cid: u32) -> io::Result<()> {
        let file = bs.file();
        let size = if self.desc_offset == 0 {
            match file.getlength() {
                Ok(l) if l <= 16 << 20 => l as usize,
                _ => {
                    error_report("VMDK description file too big");
                    return Err(errno(libc::EFBIG));
                }
            }
        } else {
            DESC_SIZE
        };
        if size == 0 {
            return Err(errno(libc::EINVAL));
        }
        let mut desc = vec![0u8; size];
        file.pread(self.desc_offset, &mut desc)?;
        write_cid_buf(&mut desc, cid)?;
        file.pwrite(self.desc_offset, &desc)?;
        file.flush()
    }

    /// `get_whole_cluster()`: fills the new grain at `cluster_offset` around the range
    /// `[skip_start, skip_end)` the caller is about to write, from the backing file or with
    /// zeroes.
    #[allow(clippy::too_many_arguments, reason = "the arguments of get_whole_cluster()")]
    fn get_whole_cluster(
        &self,
        bs: &Node,
        e: &Extent,
        file: &Node,
        cid_checked: &mut bool,
        cluster_offset: u64,
        offset: u64,
        skip_start: u64,
        skip_end: u64,
        zeroed: bool,
    ) -> bool {
        let cluster_bytes = e.cluster_sectors * SECTOR_SIZE;
        // For copy on write, align the request to the start of the grain.
        let offset = offset - offset % cluster_bytes;
        let mut grain = vec![0u8; cluster_bytes as usize];
        let backing = bs.backing().map(|c| c.node);
        let copy_from_backing = backing.is_some() && !zeroed;
        debug_assert!(skip_end <= cluster_bytes);

        // This is the first write to a grain that does not exist yet; read it from the
        // parent image if there is one.
        if backing.is_some() && !self.is_cid_valid(bs, cid_checked) {
            return false;
        }

        let (ss, se) = (skip_start as usize, skip_end as usize);
        if skip_start > 0 {
            if let (true, Some(b)) = (copy_from_backing, &backing) {
                // qcow2 emits this on bs->file instead of bs->backing
                file.debug_event(BlkdebugEvent::CowRead);
                if b.pread(offset, &mut grain[..ss]).is_err() {
                    return false;
                }
            }
            file.debug_event(BlkdebugEvent::CowWrite);
            if file.pwrite(cluster_offset, &grain[..ss]).is_err() {
                return false;
            }
        }
        if skip_end < cluster_bytes {
            if let (true, Some(b)) = (copy_from_backing, &backing) {
                file.debug_event(BlkdebugEvent::CowRead);
                if b.pread(offset + skip_end, &mut grain[se..]).is_err() {
                    return false;
                }
            }
            file.debug_event(BlkdebugEvent::CowWrite);
            if file.pwrite(cluster_offset + skip_end, &grain[se..]).is_err() {
                return false;
            }
        }
        true
    }

    /// `get_cluster_offset()`: where the grain holding guest `offset` is in its extent file.
    ///
    /// For a flat extent that is the start of the extent. For a sparse one it comes from the
    /// grain tables; with `allocate`, a missing grain gets allocated at the end of the file
    /// and filled around `[skip_start, skip_end)` before this returns.
    #[allow(clippy::too_many_arguments, reason = "the arguments of get_cluster_offset()")]
    fn get_cluster_offset(
        &self,
        bs: &Node,
        e: &mut Extent,
        file: &Node,
        cid_checked: &mut bool,
        mut m: Option<&mut MetaData>,
        guest_offset: u64,
        allocate: bool,
        skip_start: u64,
        skip_end: u64,
    ) -> Lookup {
        if let Some(m) = m.as_deref_mut() {
            m.new_allocation = false;
            m.valid = false;
        }
        if e.flat {
            return Lookup::Ok(e.flat_start_offset);
        }

        let l2_size_bytes = e.l2_size as usize * e.entry_size as usize;
        let offset = guest_offset.wrapping_sub((e.end_sector - e.sectors) * SECTOR_SIZE);
        if e.l1_entry_sectors == 0 {
            return Lookup::Error;
        }
        let l1_index = ((offset >> 9) / u64::from(e.l1_entry_sectors)) as u32;
        if l1_index >= e.l1_size {
            return Lookup::Error;
        }
        let l2_offset: u32 = if e.sesparse {
            let l2_offset_u64 = e.l1_table[l1_index as usize];
            if l2_offset_u64 == 0 {
                0
            } else if l2_offset_u64 & 0xffff_ffff_0000_0000 != 0x1000_0000_0000_0000 {
                // The top nibble is 1 for an allocated grain table. The check is strict: the
                // top 4 bytes must be 0x10000000, since at most 64 TB with 16 MB per grain
                // table fits in 32 bits.
                return Lookup::Error;
            } else {
                let idx = l2_offset_u64 & 0x0000_0000_ffff_ffff;
                let o = e
                    .sesparse_l2_tables_offset
                    .wrapping_add(idx.wrapping_mul(l2_size_bytes as u64) / SECTOR_SIZE);
                if o > 0x0000_0000_ffff_ffff {
                    return Lookup::Error;
                }
                o as u32
            }
        } else {
            e.l1_table[l1_index as usize] as u32
        };
        if l2_offset == 0 {
            return Lookup::Unalloc;
        }

        let slot = match e.l2_cache_offsets.iter().position(|&o| o == l2_offset) {
            Some(i) => {
                // Increment the hit count.
                e.l2_cache_counts[i] = e.l2_cache_counts[i].wrapping_add(1);
                if e.l2_cache_counts[i] == 0xffff_ffff {
                    for c in &mut e.l2_cache_counts {
                        *c >>= 1;
                    }
                }
                i
            }
            None => {
                // Not found: load the grain table into the least used slot.
                let mut min_index = 0;
                let mut min_count = 0xffff_ffff;
                for i in 0..L2_CACHE_SIZE {
                    if e.l2_cache_counts[i] < min_count {
                        min_count = e.l2_cache_counts[i];
                        min_index = i;
                    }
                }
                let at = min_index * l2_size_bytes;
                let table = &mut e.l2_cache[at..at + l2_size_bytes];
                file.debug_event(BlkdebugEvent::L2Load);
                if file.pread(u64::from(l2_offset) * 512, table).is_err() {
                    return Lookup::Error;
                }
                e.l2_cache_offsets[min_index] = l2_offset;
                e.l2_cache_counts[min_index] = 1;
                min_index
            }
        };

        let l2_index = ((offset >> 9) / e.cluster_sectors % u64::from(e.l2_size)) as u32;
        if let Some(m) = m.as_deref_mut() {
            m.l1_index = l1_index;
            m.l2_index = l2_index;
            m.l2_offset = l2_offset;
            m.cache_slot = slot;
            m.valid = true;
        }

        let table = &e.l2_cache[slot * l2_size_bytes..(slot + 1) * l2_size_bytes];
        let mut zeroed = false;
        let mut cluster_sector: u64;
        if e.sesparse {
            cluster_sector = le64(table, l2_index as usize * 8);
            match cluster_sector & 0xf000_0000_0000_0000 {
                0x0000_0000_0000_0000 => {
                    // An unallocated grain.
                    if cluster_sector != 0 {
                        return Lookup::Error;
                    }
                }
                // A SCSI unmapped grain, then a zero grain.
                0x1000_0000_0000_0000 | 0x2000_0000_0000_0000 => zeroed = true,
                0x3000_0000_0000_0000 => {
                    // An allocated grain.
                    cluster_sector = ((cluster_sector & 0x0fff_0000_0000_0000) >> 48)
                        | ((cluster_sector & 0x0000_ffff_ffff_ffff) << 12);
                    cluster_sector = e
                        .sesparse_clusters_offset
                        .wrapping_add(cluster_sector.wrapping_mul(e.cluster_sectors));
                }
                _ => return Lookup::Error,
            }
        } else {
            cluster_sector = u64::from(le32(table, l2_index as usize * 4));
            if e.has_zero_grain && cluster_sector == u64::from(VMDK_GTE_ZEROED) {
                zeroed = true;
            }
        }

        if cluster_sector == 0 || zeroed {
            if !allocate {
                return if zeroed { Lookup::Zeroed } else { Lookup::Unalloc };
            }
            debug_assert!(!e.sesparse);
            if e.next_cluster_sector >= VMDK_EXTENT_MAX_SECTORS {
                return Lookup::Error;
            }
            cluster_sector = e.next_cluster_sector;
            e.next_cluster_sector += e.cluster_sectors;

            // Write the grain itself first, so that running out of space or stopping at the
            // wrong time does not corrupt the image.
            if !self.get_whole_cluster(
                bs,
                e,
                file,
                cid_checked,
                cluster_sector * SECTOR_SIZE,
                guest_offset,
                skip_start,
                skip_end,
                zeroed,
            ) {
                return Lookup::Error;
            }
            if let Some(m) = m {
                m.new_allocation = true;
            }
        }
        Lookup::Ok(cluster_sector.wrapping_shl(9))
    }

    /// `vmdk_co_preadv()`.
    pub(super) fn co_preadv(&self, bs: &Node, mut offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut s = self.state();
        let State { extents, cid_checked, .. } = &mut *s;
        let mut hint = 0;
        let mut done = 0usize;
        while done < buf.len() {
            let idx = find_extent(extents, offset >> 9, hint).ok_or_else(|| errno(libc::EIO))?;
            hint = idx;
            let e = &mut extents[idx];
            let file = child_node(bs, &e.child)?;
            let ret = self.get_cluster_offset(bs, e, &file, cid_checked, None, offset, false, 0, 0);
            let oic = find_offset_in_cluster(e, offset);
            let n = ((buf.len() - done) as u64).min(e.cluster_sectors * SECTOR_SIZE - oic) as usize;
            let out = &mut buf[done..done + n];

            match ret {
                Lookup::Ok(cluster_offset) => read_extent(e, &file, cluster_offset, oic, out)?,
                _ => {
                    // Not allocated: read from the parent image if there is one.
                    let backing = bs.backing();
                    match backing {
                        Some(b) if ret != Lookup::Zeroed => {
                            if !self.is_cid_valid(bs, cid_checked) {
                                return Err(errno(libc::EINVAL));
                            }
                            // qcow2 emits this on bs->file instead of bs->backing
                            bs.file().debug_event(BlkdebugEvent::ReadBackingAio);
                            b.node.pread(offset, out)?;
                        }
                        _ => out.fill(0),
                    }
                }
            }
            offset += n as u64;
            done += n;
        }
        Ok(())
    }

    /// `vmdk_pwritev()`. With `zeroed`, `data` is ignored and the write uses zeroed grains if
    /// it can and fails with `ENOTSUP` otherwise; `zero_dry_run` then only checks whether it
    /// can, without changing the image.
    #[allow(clippy::too_many_arguments, reason = "the arguments of vmdk_pwritev()")]
    pub(super) fn pwritev(
        &self,
        bs: &Node,
        s: &mut State,
        mut offset: u64,
        mut bytes: u64,
        data: Option<&[u8]>,
        zeroed: bool,
        zero_dry_run: bool,
    ) -> io::Result<()> {
        if offset.div_ceil(SECTOR_SIZE) > self.total_sectors {
            error_report(&format!(
                "Wrong offset: offset=0x{offset:x} total_sectors=0x{:x}",
                self.total_sectors
            ));
            return Err(errno(libc::EIO));
        }

        let mut hint = 0;
        let mut done = 0usize;
        while bytes > 0 {
            {
                let State { extents, cid_checked, .. } = &mut *s;
                let idx =
                    find_extent(extents, offset >> 9, hint).ok_or_else(|| errno(libc::EIO))?;
                hint = idx;
                let e = &mut extents[idx];
                if e.sesparse {
                    return Err(errno(libc::ENOTSUP));
                }
                let file = child_node(bs, &e.child)?;
                let oic = find_offset_in_cluster(e, offset);
                let cluster_size = e.cluster_sectors * SECTOR_SIZE;
                let mut n = bytes.min(cluster_size - oic);

                let mut m = MetaData::default();
                let mut ret = self.get_cluster_offset(
                    bs,
                    e,
                    &file,
                    cid_checked,
                    Some(&mut m),
                    offset,
                    !(e.compressed || zeroed),
                    oic,
                    oic + n,
                );
                if e.compressed {
                    if let Lookup::Ok(_) = ret {
                        // Refuse to write to an allocated grain of a stream-optimized image.
                        error_report("Could not write to allocated cluster for streamOptimized");
                        return Err(errno(libc::EIO));
                    } else if !zeroed {
                        // Allocate.
                        ret = self.get_cluster_offset(
                            bs,
                            e,
                            &file,
                            cid_checked,
                            Some(&mut m),
                            offset,
                            true,
                            0,
                            0,
                        );
                    }
                }
                if ret == Lookup::Error {
                    return Err(errno(libc::EINVAL));
                }
                if zeroed {
                    // A zero write; there is no data.
                    if e.has_zero_grain && oic == 0 && n >= cluster_size {
                        n = cluster_size;
                        if !m.valid {
                            return Err(errno(libc::ENOTSUP));
                        }
                        if !zero_dry_run
                            && ret != Lookup::Zeroed
                            && !l2_update(e, &file, &m, VMDK_GTE_ZEROED)
                        {
                            return Err(errno(libc::EIO));
                        }
                    } else {
                        return Err(errno(libc::ENOTSUP));
                    }
                } else {
                    let Lookup::Ok(cluster_offset) = ret else {
                        return Err(errno(libc::EINVAL));
                    };
                    let data = data.ok_or_else(|| errno(libc::EINVAL))?;
                    let chunk = &data[done..done + n as usize];
                    write_extent(e, &file, cluster_offset, oic, chunk, offset)?;
                    if m.new_allocation && !l2_update(e, &file, &m, (cluster_offset >> 9) as u32) {
                        return Err(errno(libc::EIO));
                    }
                }
                bytes -= n;
                offset += n;
                done += n as usize;
            }

            // Give the image a new CID on the first write after every open.
            if !s.cid_updated {
                let mut b = [0u8; 4];
                ruvm_crypto::random::random_bytes(&mut b).ok();
                self.write_cid(bs, u32::from_ne_bytes(b))?;
                s.cid_updated = true;
            }
        }
        Ok(())
    }

    /// `vmdk_co_block_status()`.
    pub(super) fn co_block_status(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
    ) -> io::Result<BlockStatus> {
        let mut s = self.state();
        let State { extents, cid_checked, .. } = &mut *s;
        let idx = find_extent(extents, offset >> 9, 0).ok_or_else(|| errno(libc::EIO))?;
        let e = &mut extents[idx];
        let file = child_node(bs, &e.child)?;
        let ret = self.get_cluster_offset(bs, e, &file, cid_checked, None, offset, false, 0, 0);
        let oic = find_offset_in_cluster(e, offset);
        let mut st = BlockStatus {
            ret: 0,
            pnum: (e.cluster_sectors * SECTOR_SIZE - oic).min(bytes),
            map: 0,
            file: None,
        };
        match ret {
            Lookup::Error => return Err(errno(libc::EIO)),
            Lookup::Unalloc => {}
            Lookup::Zeroed => st.ret = BDRV_BLOCK_ZERO,
            Lookup::Ok(cluster_offset) => {
                st.ret = BDRV_BLOCK_DATA;
                if !e.compressed {
                    st.ret |= BDRV_BLOCK_OFFSET_VALID;
                    st.map = cluster_offset + oic;
                    if e.flat {
                        st.ret |= BDRV_BLOCK_RECURSE;
                    }
                } else {
                    st.ret |= BDRV_BLOCK_COMPRESSED;
                }
                st.file = Some(file);
            }
        }
        Ok(st)
    }

    /// `vmdk_co_check()`: every allocated grain must be inside its extent file.
    pub(super) fn co_check(&self, bs: &Node) -> Result<CheckResult> {
        let mut s = self.state();
        let State { extents, cid_checked, .. } = &mut *s;
        let total_sectors = self.total_sectors;
        let mut sector_num: u64 = 0;
        let mut hint = 0;
        let fail = |e: io::Error| Err(Error::generic(strerror(&e)));
        loop {
            if sector_num >= total_sectors {
                return Ok(CheckResult::default());
            }
            let Some(idx) = find_extent(extents, sector_num, hint) else {
                eprintln!("ERROR: could not find extent for sector {sector_num}");
                return fail(errno(libc::EINVAL));
            };
            hint = idx;
            let e = &mut extents[idx];
            let file = child_node(bs, &e.child).map_err(|e| Error::generic(strerror(&e)))?;
            let ret = self.get_cluster_offset(
                bs,
                e,
                &file,
                cid_checked,
                None,
                sector_num << 9,
                false,
                0,
                0,
            );
            match ret {
                Lookup::Error => {
                    eprintln!("ERROR: could not get cluster_offset for sector {sector_num}");
                    // VMDK_ERROR is -1, which the caller reads as -EPERM.
                    return fail(errno(libc::EPERM));
                }
                Lookup::Ok(cluster_offset) => {
                    let extent_len = match file.getlength() {
                        Ok(l) => l,
                        Err(err) => {
                            eprintln!(
                                "ERROR: could not get extent file length for sector {sector_num}"
                            );
                            return fail(err);
                        }
                    };
                    if cluster_offset >= extent_len {
                        eprintln!("ERROR: cluster offset for sector {sector_num} points after EOF");
                        return fail(errno(libc::EINVAL));
                    }
                }
                _ => {}
            }
            sector_num += e.cluster_sectors;
        }
    }
}

/// The part of `vmdk_write_cid()` that edits the descriptor in `desc`: the new `CID` goes
/// after "CID=" and the text from "parentCID" on follows it. The bytes after the new end of
/// the text stay as they were.
pub(super) fn write_cid_buf(desc: &mut [u8], cid: u32) -> io::Result<()> {
    let size = desc.len();
    desc[size - 1] = 0;
    let len = cstr(desc).len();
    let tmp_pos = strstr(&desc[..len], b"parentCID").ok_or_else(|| errno(libc::EINVAL))?;
    let tmp = desc[tmp_pos..len].to_vec();
    if let Some(p) = strstr(&desc[..len], b"CID") {
        // sizeof("CID") skips the name and the '='.
        let p = p + 4;
        if p < size {
            // snprintf()
            let text = format!("{cid:x}\n");
            let n = text.len().min(size - p - 1);
            desc[p..p + n].copy_from_slice(&text.as_bytes()[..n]);
            desc[p + n] = 0;
        }
        // pstrcat()
        let len = cstr(desc).len();
        if len < size {
            let n = tmp.len().min(size - len - 1);
            desc[len..len + n].copy_from_slice(&tmp[..n]);
            desc[len + n] = 0;
        }
    }
    Ok(())
}
