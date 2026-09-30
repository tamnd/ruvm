// SPDX-License-Identifier: GPL-2.0-or-later

//! Block status through the backing chain, from block/io.c: `bdrv_co_do_block_status()`,
//! `bdrv_co_common_block_status_above()` and the helpers built on them.

use std::io;
use std::sync::Arc;

use crate::node::{
    BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_DATA, BDRV_BLOCK_EOF, BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_RAW,
    BDRV_BLOCK_RECURSE, BDRV_BLOCK_ZERO, BDRV_WANT_ALLOCATED, BDRV_WANT_PRECISE, BDRV_WANT_ZERO,
    BlockStatus, Node,
};

use super::buffer_is_zero;

/// `MAX_ZERO_CHECK_BUFFER` from block/io.c.
const MAX_ZERO_CHECK_BUFFER: u64 = 128 * 1024;

impl Node {
    /// `bdrv_co_do_block_status()`: the status of the start of the range in this node alone.
    /// A `pnum` of 0 means the offset is at or past the end.
    pub(crate) fn do_block_status(
        &self,
        mode: u32,
        offset: u64,
        bytes: u64,
    ) -> io::Result<BlockStatus> {
        let mut out = BlockStatus { ret: 0, pnum: 0, map: 0, file: None };
        let total_size = self.getlength()?;
        if offset >= total_size {
            out.ret = BDRV_BLOCK_EOF;
            return Ok(out);
        }
        if bytes == 0 {
            return Ok(out);
        }
        let bytes = bytes.min(total_size - offset);

        let filtered = self.filter_child();
        // Probe whether the driver has a block_status callback. `None` from the driver means
        // it has none.
        let align = self.request_alignment();
        let aligned_offset = offset / align * align;
        let aligned_bytes = (offset + bytes).div_ceil(align) * align - aligned_offset;

        self.inc_in_flight();
        let r = self.driver.block_status(self, mode, aligned_offset, aligned_bytes);
        let r = match r {
            None if filtered.is_none() => {
                self.dec_in_flight();
                out.pnum = bytes;
                out.ret = BDRV_BLOCK_DATA | BDRV_BLOCK_ALLOCATED;
                if offset + bytes == total_size {
                    out.ret |= BDRV_BLOCK_EOF;
                }
                if self.is_protocol() {
                    out.ret |= BDRV_BLOCK_OFFSET_VALID;
                    out.map = offset;
                    out.file = Some(self.arc());
                }
                return Ok(out);
            }
            None => {
                // The default for filters.
                Ok(BlockStatus {
                    ret: BDRV_BLOCK_RAW | BDRV_BLOCK_OFFSET_VALID,
                    pnum: aligned_bytes,
                    map: aligned_offset,
                    file: filtered.map(|c| c.node),
                })
            }
            Some(r) => r,
        };
        let r = r.and_then(|st| self.block_status_finish(st, mode, offset, bytes, aligned_offset));
        self.dec_in_flight();
        let mut st = r?;
        if offset + st.pnum == total_size {
            st.ret |= BDRV_BLOCK_EOF;
        }
        Ok(st)
    }

    fn block_status_finish(
        &self,
        mut st: BlockStatus,
        mode: u32,
        offset: u64,
        bytes: u64,
        aligned_offset: u64,
    ) -> io::Result<BlockStatus> {
        // The driver's answer is a non-zero multiple of the alignment. Clamp it and adjust
        // the mapping to the original request.
        debug_assert!(st.pnum > 0);
        st.pnum -= offset - aligned_offset;
        st.pnum = st.pnum.min(bytes);
        if st.ret & BDRV_BLOCK_OFFSET_VALID != 0 {
            st.map += offset - aligned_offset;
        }

        if st.ret & BDRV_BLOCK_RAW != 0 {
            let file = st.file.clone().expect("BDRV_BLOCK_RAW comes with a file");
            return file.do_block_status(mode, st.map, st.pnum);
        }

        if st.ret & (BDRV_BLOCK_DATA | BDRV_BLOCK_ZERO) != 0 {
            st.ret |= BDRV_BLOCK_ALLOCATED;
        } else if self.def.is_some_and(|d| d.supports_backing) {
            match self.cow_child() {
                None => st.ret |= BDRV_BLOCK_ZERO,
                Some(c) if mode == BDRV_WANT_PRECISE => {
                    if let Ok(size2) = c.node.getlength() {
                        if offset >= size2 {
                            st.ret |= BDRV_BLOCK_ZERO;
                        }
                    }
                }
                Some(_) => {}
            }
        }

        let is_self = st.file.as_ref().is_some_and(|f| std::ptr::eq(Arc::as_ptr(f), self));
        if mode == BDRV_WANT_PRECISE
            && st.ret & BDRV_BLOCK_RECURSE != 0
            && st.file.is_some()
            && !is_self
            && st.ret & BDRV_BLOCK_DATA != 0
            && st.ret & BDRV_BLOCK_ZERO == 0
            && st.ret & BDRV_BLOCK_OFFSET_VALID != 0
        {
            let file = st.file.clone().expect("checked");
            // Errors are ignored: this only adds information.
            if let Ok(st2) = file.do_block_status(mode, st.map, st.pnum) {
                if st2.ret & BDRV_BLOCK_EOF != 0
                    && (st2.pnum == 0 || st2.ret & BDRV_BLOCK_ZERO != 0)
                {
                    // A format may read past the end of its file; that area reads as zero.
                    st.ret |= BDRV_BLOCK_ZERO;
                } else {
                    st.pnum = st2.pnum;
                    st.ret |= st2.ret & BDRV_BLOCK_ZERO;
                }
            }
            st.ret &= !BDRV_BLOCK_RECURSE;
        }
        Ok(st)
    }

    /// `bdrv_co_common_block_status_above()`: returns the status and the depth at which it
    /// was found.
    pub(crate) fn common_block_status_above(
        self: &Arc<Self>,
        base: Option<&Arc<Node>>,
        include_base: bool,
        mode: u32,
        offset: u64,
        bytes: u64,
    ) -> io::Result<(BlockStatus, u32)> {
        let is_base = |n: &Arc<Node>| base.is_some_and(|b| Arc::ptr_eq(b, n));
        if !include_base && is_base(self) {
            return Ok((BlockStatus { ret: 0, pnum: bytes, map: 0, file: None }, 0));
        }
        let mut depth = 1;
        let mut st = self.do_block_status(mode, offset, bytes)?;
        if st.pnum == 0 || st.ret & BDRV_BLOCK_ALLOCATED != 0 || is_base(self) {
            return Ok((st, depth));
        }
        let eof = if st.ret & BDRV_BLOCK_EOF != 0 { Some(offset + st.pnum) } else { None };
        let mut bytes = st.pnum;

        let mut p = self.filter_or_cow_bs();
        while let Some(n) = p {
            if !include_base && is_base(&n) {
                break;
            }
            st = n.do_block_status(mode, offset, bytes)?;
            depth += 1;
            if st.pnum == 0 {
                // The layer above deferred to this shorter one; the zeroes past its end
                // count as allocated here.
                st = BlockStatus {
                    ret: BDRV_BLOCK_ZERO | BDRV_BLOCK_ALLOCATED,
                    pnum: bytes,
                    map: st.map,
                    file: Some(n.clone()),
                };
                break;
            }
            if st.ret & BDRV_BLOCK_ALLOCATED != 0 {
                // Found it. The EOF is not the upper layer's, which may be longer.
                st.ret &= !BDRV_BLOCK_EOF;
                break;
            }
            if is_base(&n) {
                break;
            }
            bytes = st.pnum;
            p = n.filter_or_cow_bs();
        }
        if eof == Some(offset + st.pnum) {
            st.ret |= BDRV_BLOCK_EOF;
        }
        Ok((st, depth))
    }

    /// `bdrv_co_block_status_above()`.
    pub(crate) fn block_status_above(
        self: &Arc<Self>,
        base: Option<&Arc<Node>>,
        offset: u64,
        bytes: u64,
    ) -> io::Result<BlockStatus> {
        Ok(self.common_block_status_above(base, false, BDRV_WANT_PRECISE, offset, bytes)?.0)
    }

    /// `bdrv_co_block_status()`: the status of this node alone, with what shows through from
    /// the backing file counted as unallocated.
    pub(crate) fn block_status(&self, offset: u64, bytes: u64) -> io::Result<BlockStatus> {
        let me = self.arc();
        let base = self.filter_or_cow_bs();
        me.block_status_above(base.as_ref(), offset, bytes)
    }

    /// `bdrv_co_is_zero_fast()`: whether the range is known to read as zeroes.
    pub(crate) fn is_zero_fast(&self, mut offset: u64, mut bytes: u64) -> io::Result<bool> {
        let me = self.arc();
        while bytes > 0 {
            let (st, _) =
                me.common_block_status_above(None, false, BDRV_WANT_ZERO, offset, bytes)?;
            if st.ret & BDRV_BLOCK_ZERO == 0 || st.pnum == 0 {
                return Ok(false);
            }
            offset += st.pnum;
            bytes -= st.pnum;
        }
        Ok(true)
    }

    /// `bdrv_co_is_all_zeroes()`: whether the whole node is known to read as zeroes.
    pub(crate) fn is_all_zeroes(&self) -> io::Result<bool> {
        let me = self.arc();
        let bytes = self.getlength()?;
        let (st, _) = me.common_block_status_above(None, false, BDRV_WANT_ZERO, 0, bytes)?;
        let pnum = st.pnum;
        if st.ret & BDRV_BLOCK_ZERO != 0 {
            return self.is_zero_fast(pnum, bytes - pnum);
        }
        // Raw files are often created with a small non-sparse region at the front. If only
        // that is allocated, read it.
        if pnum > MAX_ZERO_CHECK_BUFFER {
            return Ok(false);
        }
        if !self.is_zero_fast(pnum, bytes - pnum)? {
            return Ok(false);
        }
        let mut buf = vec![0u8; pnum as usize];
        self.driver.pread(self, 0, &mut buf)?;
        Ok(buffer_is_zero(&buf))
    }

    /// `bdrv_co_is_allocated()`: whether the start of the range is allocated in this node,
    /// and for how many bytes that holds.
    pub(crate) fn is_allocated(&self, offset: u64, bytes: u64) -> io::Result<(bool, u64)> {
        let me = self.arc();
        let (st, _) =
            me.common_block_status_above(Some(&me), true, BDRV_WANT_ALLOCATED, offset, bytes)?;
        Ok((st.ret & BDRV_BLOCK_ALLOCATED != 0, st.pnum))
    }

    /// `bdrv_co_is_allocated_above()`: the depth (1 is this node) of the first layer between
    /// this node and `base` that allocates the start of the range, 0 if none does, and the
    /// number of bytes that holds for.
    pub(crate) fn is_allocated_above(
        self: &Arc<Self>,
        base: Option<&Arc<Node>>,
        include_base: bool,
        offset: u64,
        bytes: u64,
    ) -> io::Result<(u32, u64)> {
        let (st, depth) =
            self.common_block_status_above(base, include_base, BDRV_WANT_ALLOCATED, offset, bytes)?;
        let d = if st.ret & BDRV_BLOCK_ALLOCATED != 0 { depth } else { 0 };
        Ok((d, st.pnum))
    }
}
