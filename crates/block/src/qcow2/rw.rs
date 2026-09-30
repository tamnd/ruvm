// SPDX-License-Identifier: GPL-2.0-or-later

//! Guest I/O, from block/qcow2.c and block/qcow2-threads.c: reads, writes with allocation
//! and copy on write, compressed clusters, zeroing, discard, block status and the VM state
//! area.
//!
//! QEMU splits large requests into tasks that run in parallel and does the (de)compression
//! and encryption in a thread pool. Here a request runs its pieces one after the other on the
//! caller's thread, in the same order and with the same results.

use std::io;

use super::cluster::{L2Meta, REQUEST_MAX_BYTES};
use super::compress;
use super::state::*;
use crate::node::{
    BDRV_BLOCK_COMPRESSED, BDRV_BLOCK_DATA, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RECURSE,
    BDRV_BLOCK_ZERO, errno,
};

/// What [`State::block_status`] found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BlockStatus {
    /// The `BDRV_BLOCK_*` bits.
    pub status: u32,
    /// How many bytes from the start share the status.
    pub pnum: u64,
    /// With `BDRV_BLOCK_OFFSET_VALID`, the offset in the data file.
    pub map: u64,
}

impl State {
    /// `qcow2_co_encrypt()`: IVs come from the host offset for LUKS, from the guest offset for
    /// legacy AES.
    pub(crate) fn encrypt(
        &self,
        host_offset: u64,
        guest_offset: u64,
        buf: &mut [u8],
    ) -> io::Result<()> {
        self.encdec(host_offset, guest_offset, buf, true)
    }

    /// `qcow2_co_decrypt()`.
    pub(crate) fn decrypt(
        &self,
        host_offset: u64,
        guest_offset: u64,
        buf: &mut [u8],
    ) -> io::Result<()> {
        self.encdec(host_offset, guest_offset, buf, false)
    }

    fn encdec(
        &self,
        host_offset: u64,
        guest_offset: u64,
        buf: &mut [u8],
        enc: bool,
    ) -> io::Result<()> {
        let crypto = self.crypto.as_ref().expect("image is encrypted");
        let sector_size = crypto.sector_size();
        assert_eq!(guest_offset % sector_size, 0);
        assert_eq!(host_offset % sector_size, 0);
        assert_eq!(buf.len() as u64 % sector_size, 0);
        if buf.is_empty() {
            return Ok(());
        }
        let offset = if self.crypt_physical_offset { host_offset } else { guest_offset };
        let r = if enc { crypto.encrypt(offset, buf) } else { crypto.decrypt(offset, buf) };
        r.map_err(|_| errno(libc::EIO))
    }

    /// Reads from the backing image. Past its end, the data reads as zeroes, as the block
    /// layer makes it.
    fn backing_read(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let backing = self.backing.as_ref().expect("image has a backing file");
        let len = backing.len()?;
        if offset >= len {
            buf.fill(0);
            return Ok(());
        }
        let n = (len - offset).min(buf.len() as u64) as usize;
        backing.pread(offset, &mut buf[..n])?;
        buf[n..].fill(0);
        Ok(())
    }

    /// `qcow2_co_preadv_compressed()`.
    fn read_compressed(&mut self, l2_entry: u64, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let (coffset, csize) = self.parse_compressed_l2_entry(l2_entry);
        let mut cbuf = Vec::new();
        cbuf.try_reserve_exact(csize as usize).map_err(|_| errno(libc::ENOMEM))?;
        cbuf.resize(csize as usize, 0);
        self.file.pread(coffset, &mut cbuf)?;
        let mut out = vec![0u8; self.cluster_size as usize];
        self.decompress(&mut out, &cbuf).map_err(|_| errno(libc::EIO))?;
        let oic = self.offset_into_cluster(offset) as usize;
        buf.copy_from_slice(&out[oic..oic + buf.len()]);
        Ok(())
    }

    /// `qcow2_co_decompress()`.
    fn decompress(&self, dest: &mut [u8], src: &[u8]) -> io::Result<()> {
        match self.compression_type {
            QCOW2_COMPRESSION_TYPE_ZLIB => compress::zlib_decompress(dest, src),
            // Refused at open time.
            _ => Err(errno(libc::ENOTSUP)),
        }
    }

    /// `qcow2_co_compress()`.
    fn compress(&self, dest: &mut [u8], src: &[u8]) -> io::Result<usize> {
        match self.compression_type {
            QCOW2_COMPRESSION_TYPE_ZLIB => compress::zlib_compress(dest, src),
            _ => Err(errno(libc::ENOTSUP)),
        }
    }

    /// `qcow2_co_preadv_part()`.
    pub(crate) fn preadv(&mut self, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        self.check_usable()?;
        self.maybe_clean_caches();
        while !buf.is_empty() {
            let mut cur_bytes = (buf.len() as u64).min(i32::MAX as u64);
            if self.crypto.is_some() {
                cur_bytes = cur_bytes.min(QCOW_MAX_CRYPT_CLUSTERS * self.cluster_size);
            }
            let (host_offset, ty) = self.get_host_offset(offset, &mut cur_bytes)?;
            let (cur, rest) = std::mem::take(&mut buf).split_at_mut(cur_bytes as usize);
            match ty {
                SubclusterType::ZeroPlain | SubclusterType::ZeroAlloc => cur.fill(0),
                SubclusterType::UnallocatedPlain | SubclusterType::UnallocatedAlloc => {
                    if self.backing.is_some() {
                        self.backing_read(offset, cur)?;
                    } else {
                        cur.fill(0);
                    }
                }
                SubclusterType::Compressed => self.read_compressed(host_offset, offset, cur)?,
                SubclusterType::Normal => {
                    self.data().pread(host_offset, cur)?;
                    if self.crypto.is_some() {
                        self.decrypt(host_offset, offset, cur)?;
                    }
                }
                SubclusterType::Invalid => unreachable!(),
            }
            offset += cur_bytes;
            buf = rest;
        }
        Ok(())
    }

    /// `merge_cow()`: the index of the allocation whose COW regions sit right around the
    /// write, so they can go out together.
    fn merge_cow(offset: u64, bytes: u64, metas: &[L2Meta]) -> Option<usize> {
        for (i, m) in metas.iter().enumerate() {
            if (m.cow_start.nb_bytes == 0 && m.cow_end.nb_bytes == 0) || m.skip_cow {
                continue;
            }
            // A request can span allocated and unallocated clusters, so it does not always
            // line up with the regions.
            if m.cow_start_offset() + m.cow_start.nb_bytes != offset {
                assert!(offset < m.cow_start_offset());
                assert_eq!(m.cow_start.nb_bytes, 0);
                continue;
            }
            if m.offset + m.cow_end.offset != offset + bytes {
                assert!(offset + bytes > m.offset + m.cow_end.offset);
                assert_eq!(m.cow_end.nb_bytes, 0);
                continue;
            }
            return Some(i);
        }
        None
    }

    /// `qcow2_co_pwritev_task()` together with `qcow2_handle_l2meta()`.
    fn pwritev_task(
        &mut self,
        host_offset: u64,
        offset: u64,
        data: &[u8],
        mut metas: Vec<L2Meta>,
    ) -> io::Result<()> {
        let crypt_buf;
        let data = if self.crypto.is_some() {
            assert!(data.len() as u64 <= QCOW_MAX_CRYPT_CLUSTERS * self.cluster_size);
            let mut b = data.to_vec();
            if let Err(e) = self.encrypt(host_offset, offset, &mut b) {
                for m in &metas {
                    self.alloc_cluster_abort(m);
                }
                return Err(e);
            }
            crypt_buf = b;
            &crypt_buf[..]
        } else {
            data
        };
        let r = (|| {
            // With COW to do, the guest data may go out together with the copied regions.
            let merged = Self::merge_cow(offset, data.len() as u64, &metas);
            if let Some(i) = merged {
                metas[i].merge_data = true;
            } else {
                self.data().pwrite(host_offset, data)?;
            }
            // Link the new clusters; what is left over when one fails is aborted below.
            while !metas.is_empty() {
                let m = metas[0].clone();
                let d = if m.merge_data { Some(data) } else { None };
                self.alloc_cluster_link_l2(&m, d)?;
                metas.remove(0);
            }
            Ok(())
        })();
        for m in &metas {
            self.alloc_cluster_abort(m);
        }
        r
    }

    /// `qcow2_co_pwritev_part()`.
    pub(crate) fn pwritev(&mut self, mut offset: u64, mut buf: &[u8]) -> io::Result<()> {
        self.check_usable()?;
        self.maybe_clean_caches();
        while !buf.is_empty() {
            let oic = self.offset_into_cluster(offset);
            let mut cur_bytes = (buf.len() as u64).min(i32::MAX as u64);
            if self.crypto.is_some() {
                cur_bytes = cur_bytes.min(QCOW_MAX_CRYPT_CLUSTERS * self.cluster_size - oic);
            }
            let mut metas = Vec::new();
            let r = self.alloc_host_offset(offset, &mut cur_bytes, &mut metas).and_then(|h| {
                self.pre_write_overlap_check(0, h, cur_bytes, true)?;
                Ok(h)
            });
            let host_offset = match r {
                Ok(h) => h,
                Err(e) => {
                    for m in &metas {
                        self.alloc_cluster_abort(m);
                    }
                    return Err(e);
                }
            };
            let (cur, rest) = buf.split_at(cur_bytes as usize);
            self.pwritev_task(host_offset, offset, cur, metas)?;
            offset += cur_bytes;
            buf = rest;
        }
        Ok(())
    }

    /// `qcow2_co_pwritev_compressed_task()`.
    fn pwrite_compressed_cluster(&mut self, offset: u64, data: &[u8]) -> io::Result<()> {
        let cs = self.cluster_size as usize;
        assert!(
            data.len() == cs
                || (data.len() < cs && offset + data.len() as u64 == self.disk_size_sectors())
        );
        // The last cluster of an image whose size is not aligned is padded with zeroes.
        let mut buf = vec![0u8; cs];
        buf[..data.len()].copy_from_slice(data);
        let mut out = vec![0u8; cs];
        let out_len = match self.compress(&mut out[..cs - 1], &buf) {
            Ok(n) => n as u64,
            Err(e) if e.raw_os_error() == Some(libc::ENOMEM) => {
                // Does not compress: write a normal cluster.
                return self.pwritev(offset, data);
            }
            Err(_) => return Err(errno(libc::EINVAL)),
        };
        let cluster_offset = self.alloc_compressed_cluster_offset(offset, out_len)?;
        self.pre_write_overlap_check(0, cluster_offset, out_len, true)?;
        self.data().pwrite(cluster_offset, &out[..out_len as usize])?;
        Ok(())
    }

    /// `qcow2_co_pwritev_compressed_part()`. The range must start on a cluster and cover whole
    /// clusters, except for the last cluster of the image. An empty write pads the file to a
    /// whole sector.
    pub(crate) fn pwrite_compressed(&mut self, mut offset: u64, mut buf: &[u8]) -> io::Result<()> {
        self.check_usable()?;
        if self.has_data_file() {
            return Err(errno(libc::ENOTSUP));
        }
        if buf.is_empty() {
            // Align the end of the file to a sector for sector based readers.
            let len = self.file.len()?;
            return self.file.truncate(len).map_err(|_| errno(libc::EIO));
        }
        if self.offset_into_cluster(offset) != 0 {
            return Err(errno(libc::EINVAL));
        }
        if self.offset_into_cluster(buf.len() as u64) != 0
            && offset + buf.len() as u64 != self.disk_size_sectors()
        {
            return Err(errno(libc::EINVAL));
        }
        while !buf.is_empty() {
            let chunk = buf.len().min(self.cluster_size as usize);
            let (cur, rest) = buf.split_at(chunk);
            self.pwrite_compressed_cluster(offset, cur)?;
            offset += chunk as u64;
            buf = rest;
        }
        Ok(())
    }

    /// `is_zero()`: whether the whole range reads as zeroes, looking down the backing chain.
    fn is_zero(&mut self, mut offset: u64, mut bytes: u64) -> io::Result<bool> {
        let end = self.disk_size_sectors();
        if offset + bytes > end {
            bytes = end.saturating_sub(offset);
        }
        while bytes > 0 {
            let mut n = bytes;
            let (_, ty) = self.get_host_offset(offset, &mut n)?;
            match ty {
                SubclusterType::ZeroPlain | SubclusterType::ZeroAlloc => {}
                SubclusterType::UnallocatedPlain | SubclusterType::UnallocatedAlloc => {
                    if self.backing.is_some() {
                        let mut b = vec![0u8; n as usize];
                        self.backing_read(offset, &mut b)?;
                        if b.iter().any(|&x| x != 0) {
                            return Ok(false);
                        }
                    }
                }
                _ => return Ok(false),
            }
            offset += n;
            bytes -= n;
        }
        Ok(true)
    }

    /// `qcow2_co_pwrite_zeroes()`: ENOTSUP when the range cannot be zeroed through metadata,
    /// in which case the caller writes a buffer of zeroes.
    pub(crate) fn pwrite_zeroes(
        &mut self,
        mut offset: u64,
        mut bytes: u64,
        may_unmap: bool,
    ) -> io::Result<()> {
        self.check_usable()?;
        let head = self.offset_into_subcluster(offset);
        let mut tail = (offset + bytes).next_multiple_of(self.subcluster_size) - (offset + bytes);
        if offset + bytes == self.disk_size_sectors() {
            tail = 0;
        }
        if head != 0 || tail != 0 {
            assert!(head + bytes + tail <= self.subcluster_size);
            // The rest of the subcluster must read as zeroes already.
            if !(self.is_zero(offset - head, head)? && self.is_zero(offset + bytes, tail)?) {
                return Err(errno(libc::ENOTSUP));
            }
            offset -= head;
            bytes = self.subcluster_size;
            let mut nr = self.subcluster_size;
            let (_, ty) = self.get_host_offset(offset, &mut nr)?;
            if !matches!(
                ty,
                SubclusterType::UnallocatedPlain
                    | SubclusterType::UnallocatedAlloc
                    | SubclusterType::ZeroPlain
                    | SubclusterType::ZeroAlloc
            ) {
                return Err(errno(libc::ENOTSUP));
            }
        }
        self.subcluster_zeroize(offset, bytes, may_unmap)?;
        Ok(())
    }

    /// `qcow2_co_pdiscard()`.
    pub(crate) fn pdiscard(&mut self, offset: u64, bytes: u64) -> io::Result<()> {
        self.check_usable()?;
        // Without the zero flag, discarding could uncover stale backing data.
        if self.qcow_version < 3 && self.backing.is_some() {
            return Err(errno(libc::ENOTSUP));
        }
        if (offset | bytes) & (self.cluster_size - 1) != 0 {
            assert!(bytes < self.cluster_size);
            // Partial clusters are ignored, except for the complete partial cluster at the end
            // of an image whose size is not aligned.
            if self.offset_into_cluster(offset) != 0 || offset + bytes != self.disk_size_sectors() {
                return Err(errno(libc::ENOTSUP));
            }
        }
        self.cluster_discard(offset, bytes, DiscardType::Request, false)?;
        Ok(())
    }

    /// `qcow2_detect_metadata_preallocation()`: whether noticeably more clusters are in use
    /// than the file has allocated. `None` when that cannot be told.
    fn detect_metadata_preallocation(&mut self) -> io::Result<bool> {
        let file_length = self.file.len()?;
        let Some(real_allocation) = self.file.allocated_size() else {
            return Ok(false);
        };
        let real_clusters = real_allocation / self.cluster_size;
        let threshold = (real_clusters * 10 / 9).max(real_clusters + 2);
        let end_cluster = self.size_to_clusters(file_length);
        let mut cluster_count = 0;
        let mut i = 0;
        while i < end_cluster && cluster_count < threshold {
            if self.get_refcount(i)? != 0 {
                cluster_count += 1;
            }
            i += 1;
        }
        Ok(cluster_count >= threshold)
    }

    /// `qcow2_co_block_status()`.
    pub(crate) fn block_status(&mut self, offset: u64, count: u64) -> io::Result<BlockStatus> {
        self.check_usable()?;
        if !self.metadata_preallocation_checked {
            self.metadata_preallocation = self.detect_metadata_preallocation().unwrap_or(false);
            self.metadata_preallocation_checked = true;
        }
        let mut bytes = count.min(i32::MAX as u64);
        let (host_offset, ty) = self.get_host_offset(offset, &mut bytes)?;
        let mut st = BlockStatus { status: 0, pnum: bytes, map: 0 };
        if matches!(
            ty,
            SubclusterType::Normal | SubclusterType::ZeroAlloc | SubclusterType::UnallocatedAlloc
        ) && self.crypto.is_none()
        {
            st.map = host_offset;
            st.status |= BDRV_BLOCK_OFFSET_VALID;
        }
        if matches!(ty, SubclusterType::ZeroPlain | SubclusterType::ZeroAlloc) {
            st.status |= BDRV_BLOCK_ZERO;
        } else if !matches!(ty, SubclusterType::UnallocatedPlain | SubclusterType::UnallocatedAlloc)
        {
            st.status |= BDRV_BLOCK_DATA;
        }
        if self.metadata_preallocation
            && st.status & BDRV_BLOCK_DATA != 0
            && st.status & BDRV_BLOCK_OFFSET_VALID != 0
        {
            st.status |= BDRV_BLOCK_RECURSE;
        }
        if ty == SubclusterType::Compressed {
            st.status |= BDRV_BLOCK_COMPRESSED;
        }
        Ok(st)
    }

    /// `qcow2_check_vmstate_request()`.
    fn vmstate_offset(&self, pos: u64, len: usize) -> io::Result<u64> {
        let vmstate_offset = self.vm_state_offset();
        if i64::MAX as u64 - pos < vmstate_offset {
            return Err(errno(libc::EIO));
        }
        let off = pos + vmstate_offset;
        if off
            .checked_add(len as u64)
            .is_none_or(|e| e > REQUEST_MAX_BYTES.max(i64::MAX as u64 >> 9 << 9))
        {
            return Err(errno(libc::EIO));
        }
        Ok(off)
    }

    /// `qcow2_co_save_vmstate()`.
    pub(crate) fn save_vmstate(&mut self, pos: u64, buf: &[u8]) -> io::Result<()> {
        let off = self.vmstate_offset(pos, buf.len())?;
        self.pwritev(off, buf)
    }

    /// `qcow2_co_load_vmstate()`.
    pub(crate) fn load_vmstate(&mut self, pos: u64, buf: &mut [u8]) -> io::Result<()> {
        let off = self.vmstate_offset(pos, buf.len())?;
        self.preadv(off, buf)
    }
}
