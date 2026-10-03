// SPDX-License-Identifier: GPL-2.0-or-later

//! Enough of libfdt (subprojects/dtc/libfdt) and of QEMU's system/device_tree.c to build the
//! virt device tree byte for byte and to edit a user's `-dtb` the way `arm_load_dtb()` does.
//!
//! This grew out of the microvm writer in `ruvm-machine-x86` and works the same way: the tree
//! stays one flat buffer and every edit is the libfdt edit. A new subnode goes right after its
//! parent's properties (in front of older siblings), a new property right after the node name
//! (in front of older properties), and property names go into the strings block, reusing any
//! earlier string that ends with the same bytes. libfdt opens a gap with `memmove()` and only
//! copies the value into it, so alignment padding keeps whatever bytes were there before; QEMU's
//! dumps have those bytes and so do ours.
//!
//! On top of the microvm writer this has what a loaded blob needs: any block layout
//! (`fdt_open_into()`), `FDT_NOP` tags (skipped by every walk, written by `fdt_nop_node()`),
//! unit-address matching in paths (`/intc` finds `intc@8000000`), reads (`fdt_getprop()`,
//! `fdt_get_phandle()`, `fdt_get_path()`), and errors instead of panics. Every walk is bounds
//! checked the way libfdt checks it, so a malformed blob gives libfdt's error code.
//!
//! The `qemu_fdt_*` wrappers return QEMU's fatal messages as errors, without the
//! `qemu-system-aarch64: ` prefix `error_report()` adds.
//!
//! # Differences from libfdt
//!
//! - The buffer has no address, so `FDT_ERR_ALIGNMENT` never happens.
//! - Only absolute paths are looked up; libfdt also resolves a leading alias. QEMU's callers
//!   always pass absolute paths.
//! - A reserve map that runs off the end of the blob is refused with `FDT_ERR_TRUNCATED` by
//!   [`Fdt::open_into`]; libfdt goes on with a negative block size.

use std::fmt;

/// `FDT_MAGIC`.
pub const FDT_MAGIC: u32 = 0xd00d_feed;
/// `FDT_SW_MAGIC`, an unfinished sequential-write blob.
const FDT_SW_MAGIC: u32 = !FDT_MAGIC;
const FDT_BEGIN_NODE: u32 = 0x1;
const FDT_END_NODE: u32 = 0x2;
const FDT_PROP: u32 = 0x3;
const FDT_NOP: u32 = 0x4;
const FDT_END: u32 = 0x9;
/// `FDT_MAX_SIZE` in system/device_tree.c, the size of a tree `create_device_tree()` makes.
pub const FDT_MAX_SIZE: usize = 0x10_0000;
/// `sizeof(struct fdt_header)` rounded up to a reserve entry, where `fdt_create()` puts the
/// reserve map.
const CREATE_OFF_MEM_RSVMAP: usize = 0x30;
/// `sizeof(struct fdt_header)`, which is also `FDT_ALIGN(sizeof(struct fdt_header), 8)`.
const HEADER_SIZE: usize = 40;
/// `sizeof(struct fdt_reserve_entry)`.
const RSV_ENTRY_SIZE: usize = 16;
/// `FDT_FIRST_SUPPORTED_VERSION`.
const FIRST_SUPPORTED_VERSION: u32 = 0x02;
/// `FDT_LAST_SUPPORTED_VERSION`.
const LAST_SUPPORTED_VERSION: u32 = 0x11;
/// The first phandle `qemu_fdt_alloc_phandle()` hands out without `phandle-start`.
const PHANDLE_START: u32 = 0x8000;

// Header field offsets.
const H_MAGIC: usize = 0;
const H_TOTALSIZE: usize = 4;
const H_OFF_DT_STRUCT: usize = 8;
const H_OFF_DT_STRINGS: usize = 12;
const H_OFF_MEM_RSVMAP: usize = 16;
const H_VERSION: usize = 20;
const H_LAST_COMP_VERSION: usize = 24;
const H_BOOT_CPUID_PHYS: usize = 28;
const H_SIZE_DT_STRINGS: usize = 32;
const H_SIZE_DT_STRUCT: usize = 36;

/// A libfdt error code. [`fmt::Display`] gives `fdt_strerror()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FdtError {
    /// `FDT_ERR_NOTFOUND`.
    NotFound,
    /// `FDT_ERR_EXISTS`.
    Exists,
    /// `FDT_ERR_NOSPACE`.
    NoSpace,
    /// `FDT_ERR_BADOFFSET`.
    BadOffset,
    /// `FDT_ERR_BADPATH`.
    BadPath,
    /// `FDT_ERR_BADSTATE`.
    BadState,
    /// `FDT_ERR_TRUNCATED`.
    Truncated,
    /// `FDT_ERR_BADMAGIC`.
    BadMagic,
    /// `FDT_ERR_BADVERSION`.
    BadVersion,
    /// `FDT_ERR_BADSTRUCTURE`.
    BadStructure,
    /// `FDT_ERR_INTERNAL`.
    Internal,
}

impl fmt::Display for FdtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FdtError::NotFound => "FDT_ERR_NOTFOUND",
            FdtError::Exists => "FDT_ERR_EXISTS",
            FdtError::NoSpace => "FDT_ERR_NOSPACE",
            FdtError::BadOffset => "FDT_ERR_BADOFFSET",
            FdtError::BadPath => "FDT_ERR_BADPATH",
            FdtError::BadState => "FDT_ERR_BADSTATE",
            FdtError::Truncated => "FDT_ERR_TRUNCATED",
            FdtError::BadMagic => "FDT_ERR_BADMAGIC",
            FdtError::BadVersion => "FDT_ERR_BADVERSION",
            FdtError::BadStructure => "FDT_ERR_BADSTRUCTURE",
            FdtError::Internal => "FDT_ERR_INTERNAL",
        })
    }
}

impl std::error::Error for FdtError {}

type FdtResult<T> = Result<T, FdtError>;

/// `FDT_TAGALIGN()`.
fn tag_align(n: usize) -> usize {
    (n + 3) & !3
}

/// `qemu_fdt_setprop_sized_cells()` without the setprop: each `(ncells, value)` becomes one
/// or two big endian cells. `None` where QEMU returns -1: `ncells` is not 1 or 2, or a value
/// does not fit in one cell.
pub fn sized_cells(values: &[(u32, u64)]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for &(ncells, value) in values {
        let hi = (value >> 32) as u32;
        match ncells {
            2 => out.extend_from_slice(&hi.to_be_bytes()),
            1 if hi == 0 => {}
            _ => return None,
        }
        out.extend_from_slice(&(value as u32).to_be_bytes());
    }
    Some(out)
}

/// A flattened device tree open for editing, libfdt's `void *fdt` with `fdt_totalsize()`
/// equal to the buffer length.
#[derive(Clone)]
pub struct Fdt {
    buf: Vec<u8>,
    next_phandle: u32,
}

impl fmt::Debug for Fdt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Fdt")
            .field("totalsize", &self.buf.len())
            .field("size_dt_struct", &self.hdr(H_SIZE_DT_STRUCT))
            .field("size_dt_strings", &self.hdr(H_SIZE_DT_STRINGS))
            .finish_non_exhaustive()
    }
}

impl Default for Fdt {
    fn default() -> Self {
        Fdt::new()
    }
}

impl Fdt {
    /// `create_device_tree()`: an [`FDT_MAX_SIZE`] blob with an empty root node.
    pub fn new() -> Fdt {
        let mut f = Fdt { buf: vec![0; FDT_MAX_SIZE], next_phandle: PHANDLE_START };
        let off_struct = CREATE_OFF_MEM_RSVMAP + RSV_ENTRY_SIZE;
        for (field, v) in [
            (H_MAGIC, FDT_MAGIC),
            (H_TOTALSIZE, FDT_MAX_SIZE as u32),
            (H_OFF_DT_STRUCT, off_struct as u32),
            (H_OFF_DT_STRINGS, off_struct as u32 + 16),
            (H_OFF_MEM_RSVMAP, CREATE_OFF_MEM_RSVMAP as u32),
            (H_VERSION, 17),
            (H_LAST_COMP_VERSION, 16),
            (H_BOOT_CPUID_PHYS, 0),
            (H_SIZE_DT_STRINGS, 0),
            (H_SIZE_DT_STRUCT, 16),
        ] {
            f.set_hdr(field, v);
        }
        for (i, w) in [FDT_BEGIN_NODE, 0, FDT_END_NODE, FDT_END].iter().enumerate() {
            f.put_word(off_struct + i * 4, *w);
        }
        f
    }

    /// `fdt_open_into(fdt, buf, bufsize)` with `buf` a zeroed `bufsize` buffer that already
    /// holds `blob`, as `load_device_tree()` calls it: the tree is reordered if its blocks are
    /// out of libfdt's order, and becomes version 17 with `bufsize` bytes of room.
    pub fn open_into(blob: &[u8], bufsize: usize) -> FdtResult<Fdt> {
        let mut buf = vec![0; bufsize.max(HEADER_SIZE)];
        let n = blob.len().min(buf.len());
        buf[..n].copy_from_slice(&blob[..n]);
        let mut f = Fdt { buf, next_phandle: PHANDLE_START };
        f.ro_probe()?;

        let mem_rsv_size = (f.num_mem_rsv()? + 1) * RSV_ENTRY_SIZE;
        let version = f.hdr(H_VERSION);
        let struct_size = if version >= 17 {
            f.hdr(H_SIZE_DT_STRUCT) as usize
        } else if version == 16 {
            let mut size = 0;
            loop {
                let (tag, next) = f.next_tag(size);
                size = next?;
                if tag == FDT_END {
                    break;
                }
            }
            size
        } else {
            return Err(FdtError::BadVersion);
        };

        let totalsize = f.hdr(H_TOTALSIZE) as usize;
        if !f.blocks_misordered(mem_rsv_size, struct_size) {
            // fdt_move()
            if totalsize > bufsize {
                return Err(FdtError::NoSpace);
            }
            f.set_hdr(H_VERSION, 17);
            f.set_hdr(H_SIZE_DT_STRUCT, struct_size as u32);
            f.set_hdr(H_TOTALSIZE, bufsize as u32);
            return Ok(f);
        }

        let strings_size = f.hdr(H_SIZE_DT_STRINGS) as usize;
        let newsize = HEADER_SIZE + mem_rsv_size + struct_size + strings_size;
        if bufsize < newsize {
            return Err(FdtError::NoSpace);
        }
        // The buffer is the old tree, so the converted one is built right after it.
        let tmp = totalsize;
        if tmp + newsize > bufsize {
            return Err(FdtError::NoSpace);
        }
        // fdt_packblocks_()
        let rsv_off = HEADER_SIZE;
        let struct_off = rsv_off + mem_rsv_size;
        let strings_off = struct_off + struct_size;
        for (from, to, len) in [
            (f.hdr(H_OFF_MEM_RSVMAP) as usize, rsv_off, mem_rsv_size),
            (f.hdr(H_OFF_DT_STRUCT) as usize, struct_off, struct_size),
            (f.hdr(H_OFF_DT_STRINGS) as usize, strings_off, strings_size),
        ] {
            if from + len > f.buf.len() {
                return Err(FdtError::Truncated);
            }
            f.buf.copy_within(from..from + len, tmp + to);
        }
        for (field, v) in [
            (H_OFF_MEM_RSVMAP, rsv_off),
            (H_OFF_DT_STRUCT, struct_off),
            (H_SIZE_DT_STRUCT, struct_size),
            (H_OFF_DT_STRINGS, strings_off),
            (H_SIZE_DT_STRINGS, strings_size),
        ] {
            f.put_word(tmp + field, v as u32);
        }
        f.buf.copy_within(tmp..tmp + newsize, 0);
        f.set_hdr(H_MAGIC, FDT_MAGIC);
        f.set_hdr(H_TOTALSIZE, bufsize as u32);
        f.set_hdr(H_VERSION, 17);
        f.set_hdr(H_LAST_COMP_VERSION, 16);
        // libfdt copies boot_cpuid_phys from the old header, which the memmove() just
        // overwrote with the converted one, so it keeps what is there now.
        Ok(f)
    }

    /// `fdt_check_header()`.
    pub fn check_header(&self) -> FdtResult<()> {
        if self.hdr(H_MAGIC) != FDT_MAGIC {
            return Err(FdtError::BadMagic);
        }
        let version = self.hdr(H_VERSION);
        let last_comp = self.hdr(H_LAST_COMP_VERSION);
        if version < FIRST_SUPPORTED_VERSION
            || last_comp > LAST_SUPPORTED_VERSION
            || version < last_comp
        {
            return Err(FdtError::BadVersion);
        }
        let hdrsize: u64 = match version {
            ..=1 => 28,
            2 => 32,
            3..=16 => 36,
            _ => 40,
        };
        let total = u64::from(self.hdr(H_TOTALSIZE));
        if total < hdrsize || total > i32::MAX as u64 {
            return Err(FdtError::Truncated);
        }
        let check_off = |off: u64| off >= hdrsize && off <= total;
        let check_block = |base: u64, size: u64| check_off(base) && check_off(base + size);
        if !check_off(u64::from(self.hdr(H_OFF_MEM_RSVMAP))) {
            return Err(FdtError::Truncated);
        }
        let off_struct = u64::from(self.hdr(H_OFF_DT_STRUCT));
        let struct_ok = if version < 17 {
            check_off(off_struct)
        } else {
            check_block(off_struct, u64::from(self.hdr(H_SIZE_DT_STRUCT)))
        };
        if !struct_ok
            || !check_block(
                u64::from(self.hdr(H_OFF_DT_STRINGS)),
                u64::from(self.hdr(H_SIZE_DT_STRINGS)),
            )
        {
            return Err(FdtError::Truncated);
        }
        Ok(())
    }

    /// The blob, `fdt_totalsize()` bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// The blob, `fdt_totalsize()` bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    fn hdr(&self, field: usize) -> u32 {
        self.word(field)
    }

    fn set_hdr(&mut self, field: usize, v: u32) {
        self.put_word(field, v);
    }

    fn word(&self, off: usize) -> u32 {
        let b = &self.buf[off..off + 4];
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    }

    fn put_word(&mut self, off: usize, v: u32) {
        self.buf[off..off + 4].copy_from_slice(&v.to_be_bytes());
    }

    fn off_struct(&self) -> usize {
        self.hdr(H_OFF_DT_STRUCT) as usize
    }

    /// `fdt_ro_probe_()`.
    fn ro_probe(&self) -> FdtResult<()> {
        if self.buf.len() < HEADER_SIZE {
            return Err(FdtError::Truncated);
        }
        match self.hdr(H_MAGIC) {
            FDT_MAGIC => {
                if self.hdr(H_VERSION) < FIRST_SUPPORTED_VERSION
                    || self.hdr(H_LAST_COMP_VERSION) > LAST_SUPPORTED_VERSION
                {
                    return Err(FdtError::BadVersion);
                }
            }
            FDT_SW_MAGIC => {
                if self.hdr(H_SIZE_DT_STRUCT) == 0 {
                    return Err(FdtError::BadState);
                }
            }
            _ => return Err(FdtError::BadMagic),
        }
        if self.hdr(H_TOTALSIZE) >= i32::MAX as u32 {
            return Err(FdtError::Truncated);
        }
        Ok(())
    }

    /// `fdt_num_mem_rsv()`.
    fn num_mem_rsv(&self) -> FdtResult<usize> {
        let base = self.hdr(H_OFF_MEM_RSVMAP) as usize;
        let total = (self.hdr(H_TOTALSIZE) as usize).min(self.buf.len());
        let mut i = 0;
        loop {
            let off = base + i * RSV_ENTRY_SIZE;
            if total < RSV_ENTRY_SIZE || off > total - RSV_ENTRY_SIZE {
                return Err(FdtError::Truncated);
            }
            if self.buf[off + 8..off + 16].iter().all(|&b| b == 0) {
                return Ok(i);
            }
            i += 1;
        }
    }

    /// `fdt_blocks_misordered_()`.
    fn blocks_misordered(&self, mem_rsv_size: usize, struct_size: usize) -> bool {
        let rsv = self.hdr(H_OFF_MEM_RSVMAP) as usize;
        let st = self.off_struct();
        let strings = self.hdr(H_OFF_DT_STRINGS) as usize;
        rsv < HEADER_SIZE
            || st < rsv + mem_rsv_size
            || strings < st + struct_size
            || (self.hdr(H_TOTALSIZE) as usize) < strings + self.hdr(H_SIZE_DT_STRINGS) as usize
    }

    /// `fdt_offset_ptr()`: the buffer offset of `len` bytes at structure offset `off`, if
    /// they are inside the structure block.
    fn offset_ptr(&self, off: usize, len: usize) -> Option<usize> {
        let abs = self.off_struct() + off;
        let total = (self.hdr(H_TOTALSIZE) as usize).min(self.buf.len());
        if abs + len > total {
            return None;
        }
        if self.hdr(H_VERSION) >= 0x11 && off + len > self.hdr(H_SIZE_DT_STRUCT) as usize {
            return None;
        }
        Some(abs)
    }

    /// `fdt_next_tag()`: the tag at structure offset `start` and where the next one starts.
    /// A tag that runs off the structure block reads as `FDT_END` with an error.
    fn next_tag(&self, start: usize) -> (u32, FdtResult<usize>) {
        let Some(p) = self.offset_ptr(start, 4) else {
            return (FDT_END, Err(FdtError::Truncated));
        };
        let tag = self.word(p);
        let mut off = start + 4;
        let bad = (FDT_END, Err(FdtError::BadStructure));
        match tag {
            FDT_BEGIN_NODE => loop {
                let Some(p) = self.offset_ptr(off, 1) else { return bad };
                off += 1;
                if self.buf[p] == 0 {
                    break;
                }
            },
            FDT_PROP => {
                let Some(p) = self.offset_ptr(off, 4) else { return bad };
                let len = self.word(p) as usize;
                if len + off >= i32::MAX as usize {
                    return bad;
                }
                off += 8 + len;
                if self.hdr(H_VERSION) < 0x10 && len >= 8 && (off - len) % 8 != 0 {
                    off += 4;
                }
            }
            FDT_END | FDT_END_NODE | FDT_NOP => {}
            _ => return bad,
        }
        if self.offset_ptr(start, off - start).is_none() {
            return bad;
        }
        (tag, Ok(tag_align(off)))
    }

    /// `fdt_check_node_offset_()`: where the node at `off` starts its properties.
    fn check_node_offset(&self, off: usize) -> FdtResult<usize> {
        if off % 4 != 0 {
            return Err(FdtError::BadOffset);
        }
        match self.next_tag(off) {
            (FDT_BEGIN_NODE, Ok(next)) => Ok(next),
            _ => Err(FdtError::BadOffset),
        }
    }

    /// `fdt_check_prop_offset_()`.
    fn check_prop_offset(&self, off: usize) -> FdtResult<usize> {
        if off % 4 != 0 {
            return Err(FdtError::BadOffset);
        }
        match self.next_tag(off) {
            (FDT_PROP, Ok(next)) => Ok(next),
            _ => Err(FdtError::BadOffset),
        }
    }

    /// `fdt_next_node()`: the next node after `offset` (the root if `None`), keeping track of
    /// the depth if asked.
    pub fn next_node(
        &self,
        offset: Option<usize>,
        mut depth: Option<&mut i32>,
    ) -> FdtResult<usize> {
        let mut next = match offset {
            Some(o) => self.check_node_offset(o)?,
            None => 0,
        };
        loop {
            let off = next;
            let (tag, n) = self.next_tag(off);
            match tag {
                FDT_PROP | FDT_NOP => {}
                FDT_BEGIN_NODE => {
                    if let Some(d) = depth.as_deref_mut() {
                        *d += 1;
                    }
                }
                FDT_END_NODE => {
                    if let Some(d) = depth.as_deref_mut() {
                        *d -= 1;
                        if *d < 0 {
                            return n;
                        }
                    }
                }
                _ => {
                    return match n {
                        Ok(_) => Err(FdtError::NotFound),
                        Err(FdtError::Truncated) if depth.is_none() => Err(FdtError::NotFound),
                        Err(e) => Err(e),
                    };
                }
            }
            next = n?;
            if tag == FDT_BEGIN_NODE {
                return Ok(off);
            }
        }
    }

    /// `fdt_get_name()`.
    pub fn get_name(&self, node: usize) -> FdtResult<&[u8]> {
        self.check_node_offset(node)?;
        let name = &self.buf[self.off_struct() + node + 4..];
        // next_tag() found the terminator inside the structure block.
        let len = name.iter().position(|&b| b == 0).ok_or(FdtError::Internal)?;
        Ok(&name[..len])
    }

    /// `fdt_nodename_eq_()`: `s` names the node exactly, or has no unit address and the node
    /// name is `s@unit`.
    fn nodename_eq(&self, node: usize, s: &[u8]) -> bool {
        let Ok(p) = self.get_name(node) else { return false };
        if p.len() < s.len() || &p[..s.len()] != s {
            return false;
        }
        p.len() == s.len() || (!s.contains(&b'@') && p[s.len()] == b'@')
    }

    /// `fdt_subnode_offset_namelen()`.
    pub fn subnode_offset(&self, parent: usize, name: &[u8]) -> FdtResult<usize> {
        let mut depth = 0;
        let mut offset = Ok(parent);
        loop {
            if depth < 0 {
                return Err(FdtError::NotFound);
            }
            let off = offset?;
            if depth == 1 && self.nodename_eq(off, name) {
                return Ok(off);
            }
            offset = self.next_node(Some(off), Some(&mut depth));
        }
    }

    /// `fdt_path_offset()` for an absolute path.
    pub fn path_offset(&self, path: &str) -> FdtResult<usize> {
        if !path.starts_with('/') {
            return Err(FdtError::BadPath);
        }
        let mut offset = 0;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            offset = self.subnode_offset(offset, part.as_bytes())?;
        }
        Ok(offset)
    }

    /// `fdt_get_string()`.
    fn get_string(&self, stroffset: u32) -> FdtResult<&[u8]> {
        let total = self.buf.len();
        let abs = self.hdr(H_OFF_DT_STRINGS) as usize + stroffset as usize;
        if abs >= total {
            return Err(FdtError::BadOffset);
        }
        let size = self.hdr(H_SIZE_DT_STRINGS) as usize;
        if stroffset as usize >= size {
            return Err(FdtError::BadOffset);
        }
        let len = (total - abs).min(size - stroffset as usize);
        let s = &self.buf[abs..abs + len];
        let n = s.iter().position(|&b| b == 0).ok_or(FdtError::Truncated)?;
        Ok(&s[..n])
    }

    /// `nextprop_()`.
    fn nextprop(&self, mut off: usize) -> FdtResult<usize> {
        loop {
            let (tag, next) = self.next_tag(off);
            match tag {
                FDT_END => return Err(next.err().unwrap_or(FdtError::BadStructure)),
                FDT_PROP => return Ok(off),
                FDT_NOP => off = next?,
                _ => return Err(FdtError::NotFound),
            }
        }
    }

    /// `fdt_get_property_namelen_()`: the offset and length of property `name` of `node`.
    fn get_property(&self, node: usize, name: &[u8]) -> FdtResult<(usize, usize)> {
        let mut offset = self.nextprop(self.check_node_offset(node)?);
        loop {
            let off = offset?;
            let abs = self.off_struct() + off;
            if self.get_string(self.word(abs + 8)).is_ok_and(|s| s == name) {
                return Ok((off, self.word(abs + 4) as usize));
            }
            offset = self.nextprop(self.check_prop_offset(off)?);
        }
    }

    /// `fdt_getprop()`.
    pub fn getprop_at(&self, node: usize, name: &str) -> FdtResult<&[u8]> {
        let (off, len) = self.get_property(node, name.as_bytes())?;
        let abs = self.off_struct() + off + 12;
        Ok(&self.buf[abs..abs + len])
    }

    /// `fdt_get_phandle()`: the node's `phandle` (or `linux,phandle`), 0 if it has none.
    pub fn get_phandle_at(&self, node: usize) -> u32 {
        for name in ["phandle", "linux,phandle"] {
            if let Ok(v) = self.getprop_at(node, name) {
                if v.len() == 4 {
                    return u32::from_be_bytes([v[0], v[1], v[2], v[3]]);
                }
            }
        }
        0
    }

    /// `fdt_get_path()`.
    pub fn get_path(&self, node: usize) -> FdtResult<String> {
        let mut names: Vec<&[u8]> = Vec::new();
        let mut depth = 0;
        let mut offset = Ok(0);
        loop {
            let off = match offset {
                Ok(o) if o <= node => o,
                Ok(_) | Err(FdtError::NotFound) => return Err(FdtError::BadOffset),
                Err(FdtError::BadOffset) => return Err(FdtError::BadStructure),
                Err(e) => return Err(e),
            };
            names.truncate(depth.max(0) as usize);
            names.push(self.get_name(off)?);
            if off == node {
                let path: Vec<String> =
                    names[1..].iter().map(|n| String::from_utf8_lossy(n).into_owned()).collect();
                return Ok(format!("/{}", path.join("/")));
            }
            offset = self.next_node(Some(off), Some(&mut depth));
        }
    }

    /// `fdt_splice_()`: makes the `oldlen` bytes at buffer offset `p` `newlen` long by moving
    /// everything after them up to the end of the strings block. A new gap keeps its bytes.
    fn splice(&mut self, p: usize, oldlen: usize, newlen: usize) -> FdtResult<()> {
        let end = self.hdr(H_OFF_DT_STRINGS) as usize + self.hdr(H_SIZE_DT_STRINGS) as usize;
        if p + oldlen > end {
            return Err(FdtError::BadOffset);
        }
        if end - oldlen + newlen > self.buf.len() {
            return Err(FdtError::NoSpace);
        }
        self.buf.copy_within(p + oldlen..end, p + newlen);
        Ok(())
    }

    /// `fdt_splice_struct_()` at structure offset `off`.
    fn splice_struct(&mut self, off: usize, oldlen: usize, newlen: usize) -> FdtResult<()> {
        self.splice(self.off_struct() + off, oldlen, newlen)?;
        let delta = |v: u32| (v as usize + newlen - oldlen) as u32;
        self.set_hdr(H_SIZE_DT_STRUCT, delta(self.hdr(H_SIZE_DT_STRUCT)));
        self.set_hdr(H_OFF_DT_STRINGS, delta(self.hdr(H_OFF_DT_STRINGS)));
        Ok(())
    }

    /// `fdt_find_add_string_()`: the offset of `s` in the strings block and whether it was
    /// added.
    fn find_add_string(&mut self, s: &str) -> FdtResult<(u32, bool)> {
        let mut needle = s.as_bytes().to_vec();
        needle.push(0);
        let base = self.hdr(H_OFF_DT_STRINGS) as usize;
        let size = self.hdr(H_SIZE_DT_STRINGS) as usize;
        let strings = &self.buf[base..base + size];
        if let Some(pos) = strings.windows(needle.len()).position(|w| w == needle.as_slice()) {
            return Ok((pos as u32, false));
        }
        self.splice(base + size, 0, needle.len())?;
        self.buf[base + size..base + size + needle.len()].copy_from_slice(&needle);
        self.set_hdr(H_SIZE_DT_STRINGS, (size + needle.len()) as u32);
        Ok((size as u32, true))
    }

    /// `fdt_add_subnode_namelen()`: the offset of the new node.
    pub fn add_subnode_at(&mut self, parent: usize, name: &[u8]) -> FdtResult<usize> {
        match self.subnode_offset(parent, name) {
            Ok(_) => return Err(FdtError::Exists),
            Err(FdtError::NotFound) => {}
            Err(e) => return Err(e),
        }
        // After the parent's properties.
        let (tag, mut next) = self.next_tag(parent);
        if tag != FDT_BEGIN_NODE {
            return Err(FdtError::Internal);
        }
        let off = loop {
            let off = next?;
            let (tag, n) = self.next_tag(off);
            next = n;
            if tag != FDT_PROP && tag != FDT_NOP {
                break off;
            }
        };
        let namelen = tag_align(name.len() + 1);
        let nodelen = 4 + namelen + 4;
        self.splice_struct(off, 0, nodelen)?;
        let abs = self.off_struct() + off;
        self.put_word(abs, FDT_BEGIN_NODE);
        self.buf[abs + 4..abs + 4 + namelen].fill(0);
        self.buf[abs + 4..abs + 4 + name.len()].copy_from_slice(name);
        self.put_word(abs + nodelen - 4, FDT_END_NODE);
        Ok(off)
    }

    /// `fdt_setprop()`: resizes the property if the node has it, adds it in front of the
    /// others if not, then copies the value in.
    pub fn setprop_at(&mut self, node: usize, name: &str, value: &[u8]) -> FdtResult<()> {
        let prop = match self.get_property(node, name.as_bytes()) {
            // fdt_resize_property_()
            Ok((prop, oldlen)) => {
                self.splice_struct(prop + 12, tag_align(oldlen), tag_align(value.len()))?;
                prop
            }
            // fdt_add_property_()
            Err(FdtError::NotFound) => {
                let next = self.check_node_offset(node)?;
                let (nameoff, allocated) = self.find_add_string(name)?;
                if let Err(e) = self.splice_struct(next, 0, 12 + tag_align(value.len())) {
                    if allocated {
                        // fdt_del_last_string_()
                        let size = self.hdr(H_SIZE_DT_STRINGS) as usize - name.len() - 1;
                        self.set_hdr(H_SIZE_DT_STRINGS, size as u32);
                    }
                    return Err(e);
                }
                let abs = self.off_struct() + next;
                self.put_word(abs, FDT_PROP);
                self.put_word(abs + 8, nameoff);
                next
            }
            Err(e) => return Err(e),
        };
        let abs = self.off_struct() + prop;
        self.put_word(abs + 4, value.len() as u32);
        self.buf[abs + 12..abs + 12 + value.len()].copy_from_slice(value);
        Ok(())
    }

    /// `fdt_nop_node()`: turns the node and everything in it into `FDT_NOP` tags.
    pub fn nop_node_at(&mut self, node: usize) -> FdtResult<()> {
        // fdt_node_end_offset_()
        let mut depth = 0;
        let mut offset = Ok(node);
        let end = loop {
            let off = offset?;
            if depth < 0 {
                break off;
            }
            offset = self.next_node(Some(off), Some(&mut depth));
        };
        let base = self.off_struct();
        for off in (node..end).step_by(4) {
            self.put_word(base + off, FDT_NOP);
        }
        Ok(())
    }

    // The qemu_fdt_* wrappers. Their errors are the messages QEMU prints before exiting.

    /// `findnode_nofail()`.
    pub fn findnode(&self, path: &str) -> Result<usize, String> {
        self.path_offset(path)
            .map_err(|e| format!("findnode_nofail Couldn't find node {path}: {e}"))
    }

    /// Whether `fdt_path_offset(fdt, path) >= 0`.
    pub fn exists(&self, path: &str) -> bool {
        self.path_offset(path).is_ok()
    }

    /// `qemu_fdt_add_subnode()`.
    pub fn add_subnode(&mut self, path: &str) -> Result<usize, String> {
        let (parent, name) = path.rsplit_once('/').ok_or_else(|| {
            format!("qemu_fdt_add_subnode: Failed to create subnode {path}: FDT_ERR_BADPATH")
        })?;
        let parent = if parent.is_empty() { 0 } else { self.findnode(parent)? };
        self.add_subnode_at(parent, name.as_bytes())
            .map_err(|e| format!("qemu_fdt_add_subnode: Failed to create subnode {path}: {e}"))
    }

    /// `qemu_fdt_add_path()`: adds every missing node on `path`.
    pub fn add_path(&mut self, path: &str) -> Result<usize, String> {
        let Some(rest) = path.strip_prefix('/') else {
            return Err(format!("qemu_fdt_add_path: Failed to create subnode {path}: bad path"));
        };
        let mut parent = 0;
        for name in rest.split('/') {
            parent = match self.subnode_offset(parent, name.as_bytes()) {
                Ok(o) => o,
                Err(FdtError::NotFound) => {
                    self.add_subnode_at(parent, name.as_bytes()).map_err(|e| {
                        format!("qemu_fdt_add_path: Failed to create subnode {name}: {e}")
                    })?
                }
                Err(e) => {
                    return Err(format!(
                        "qemu_fdt_add_path: Unexpected error in finding subnode {name}: {e}"
                    ));
                }
            };
        }
        Ok(parent)
    }

    /// `qemu_fdt_setprop()`.
    pub fn setprop(&mut self, path: &str, name: &str, value: &[u8]) -> Result<(), String> {
        let node = self.findnode(path)?;
        self.setprop_at(node, name, value)
            .map_err(|e| format!("qemu_fdt_setprop: Couldn't set {path}/{name}: {e}"))
    }

    /// `qemu_fdt_setprop_cell()`.
    pub fn setprop_cell(&mut self, path: &str, name: &str, value: u32) -> Result<(), String> {
        let node = self.findnode(path)?;
        self.setprop_at(node, name, &value.to_be_bytes()).map_err(|e| {
            format!("qemu_fdt_setprop_cell: Couldn't set {path}/{name} = {value:#08x}: {e}")
        })
    }

    /// `qemu_fdt_setprop_cells()`.
    pub fn setprop_cells(&mut self, path: &str, name: &str, values: &[u32]) -> Result<(), String> {
        let v: Vec<u8> = values.iter().flat_map(|c| c.to_be_bytes()).collect();
        self.setprop(path, name, &v)
    }

    /// `qemu_fdt_setprop_u64()`.
    pub fn setprop_u64(&mut self, path: &str, name: &str, value: u64) -> Result<(), String> {
        self.setprop(path, name, &value.to_be_bytes())
    }

    /// `qemu_fdt_setprop_string()`.
    pub fn setprop_string(&mut self, path: &str, name: &str, value: &str) -> Result<(), String> {
        let node = self.findnode(path)?;
        let mut v = value.as_bytes().to_vec();
        v.push(0);
        self.setprop_at(node, name, &v).map_err(|e| {
            format!("qemu_fdt_setprop_string: Couldn't set {path}/{name} = {value}: {e}")
        })
    }

    /// `qemu_fdt_getprop()`.
    pub fn getprop(&self, path: &str, name: &str) -> Result<&[u8], String> {
        let node = self.findnode(path)?;
        self.getprop_at(node, name)
            .map_err(|e| format!("qemu_fdt_getprop: Couldn't get {path}/{name}: {e}"))
    }

    /// `qemu_fdt_getprop_cell()`.
    pub fn getprop_cell(&self, path: &str, name: &str) -> Result<u32, String> {
        let v = self.getprop(path, name)?;
        if v.len() != 4 {
            return Err(format!(
                "qemu_fdt_getprop_cell: {path}/{name} not 4 bytes long (not a cell?)"
            ));
        }
        Ok(u32::from_be_bytes([v[0], v[1], v[2], v[3]]))
    }

    /// `qemu_fdt_get_phandle()`.
    pub fn get_phandle(&self, path: &str) -> Result<u32, String> {
        let p = self.get_phandle_at(self.findnode(path)?);
        if p == 0 {
            return Err(format!(
                "qemu_fdt_get_phandle: Couldn't get phandle for {path}: <valid offset/length>"
            ));
        }
        Ok(p)
    }

    /// `qemu_fdt_nop_node()`.
    pub fn nop_node(&mut self, path: &str) -> Result<(), String> {
        let node = self.findnode(path)?;
        self.nop_node_at(node)
            .map_err(|e| format!("qemu_fdt_nop_node: Couldn't nop node {path}: {e}"))
    }

    /// `qemu_fdt_node_unit_path()`: the paths of every node called `name` or `name@...`, in
    /// document order.
    pub fn node_unit_path(&self, name: &str) -> Result<Vec<String>, String> {
        let prefix = format!("{name}@");
        let mut out = Vec::new();
        let mut offset = self.next_node(None, None);
        let err = loop {
            let off = match offset {
                Ok(o) => o,
                Err(e) => break e,
            };
            let n = match self.get_name(off) {
                Ok(n) => n,
                Err(e) => break e,
            };
            if n == name.as_bytes() || n.starts_with(prefix.as_bytes()) {
                out.push(self.get_path(off).unwrap_or_default());
            }
            offset = self.next_node(Some(off), None);
        };
        if err != FdtError::NotFound {
            return Err(format!(
                "qemu_fdt_node_unit_path: abort parsing dt for {name} node units: {err}"
            ));
        }
        Ok(out)
    }

    /// `qemu_fdt_alloc_phandle()` with no `phandle-start`.
    pub fn alloc_phandle(&mut self) -> u32 {
        let p = self.next_phandle;
        self.next_phandle += 1;
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(b: &[u8], o: usize) -> u32 {
        u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
    }

    #[test]
    fn empty_tree_matches_create_device_tree() {
        let f = Fdt::new();
        let blob = f.as_bytes();
        assert_eq!(blob.len(), FDT_MAX_SIZE);
        assert_eq!(word(blob, 0), FDT_MAGIC);
        // BEGIN_NODE "" END_NODE END
        assert_eq!(&blob[0x40..0x50], &[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 9]);
        assert_eq!(word(blob, 12), 0x50);
        assert_eq!(word(blob, 36), 16);
        assert_eq!(f.check_header(), Ok(()));
        assert_eq!(f.path_offset("/"), Ok(0));
        assert_eq!(f.get_path(0), Ok("/".to_string()));
    }

    #[test]
    fn newest_first_and_suffix_strings() {
        let mut f = Fdt::new();
        f.add_subnode("/a").unwrap();
        f.add_subnode("/b").unwrap();
        f.setprop_cell("/a", "linux,phandle", 1).unwrap();
        f.setprop_cell("/a", "phandle", 1).unwrap();
        // "b" comes first.
        assert_eq!(f.next_node(Some(0), None), f.path_offset("/b"));
        let a = f.path_offset("/a").unwrap();
        let (first, _) = f.get_property(a, b"phandle").unwrap();
        assert_eq!(first, f.check_node_offset(a).unwrap());
        // "phandle" is the tail of "linux,phandle".
        assert_eq!(word(f.as_bytes(), f.off_struct() + first + 8), 6);
        assert_eq!(f.get_phandle("/a"), Ok(1));
        assert_eq!(f.get_path(a), Ok("/a".to_string()));
    }

    #[test]
    fn padding_keeps_the_old_bytes() {
        // The gap for "q" opens where "p" was, and only one byte of it is written, so the
        // padding keeps the last three bytes of the old value of "p".
        let mut f = Fdt::new();
        f.add_subnode("/a").unwrap();
        f.setprop("/a", "p", &[1, 2, 3, 4]).unwrap();
        f.setprop("/a", "q", &[0xaa]).unwrap();
        let a = f.path_offset("/a").unwrap();
        let (prop, len) = f.get_property(a, b"q").unwrap();
        assert_eq!(len, 1);
        let abs = f.off_struct() + prop;
        assert_eq!(&f.as_bytes()[abs + 12..abs + 16], &[0xaa, 2, 3, 4]);
    }

    #[test]
    fn unit_names_nops_and_paths() {
        let mut f = Fdt::new();
        f.add_subnode("/intc@8000000").unwrap();
        f.setprop_cell("/intc@8000000", "phandle", 0x8001).unwrap();
        f.add_subnode("/memory@40000000").unwrap();
        f.add_path("/cpus/cpu-map/socket0/cluster0/core0").unwrap();
        assert_eq!(f.get_phandle("/intc"), Ok(0x8001));
        assert!(f.exists("/memory"));
        assert!(!f.exists("/intc@0"));
        assert_eq!(f.node_unit_path("memory"), Ok(vec!["/memory@40000000".to_string()]));
        assert_eq!(
            f.node_unit_path("core0"),
            Ok(vec!["/cpus/cpu-map/socket0/cluster0/core0".to_string()])
        );
        assert_eq!(
            f.add_subnode("/memory@40000000"),
            Err("qemu_fdt_add_subnode: Failed to create subnode /memory@40000000: \
                 FDT_ERR_EXISTS"
                .to_string())
        );
        f.nop_node("/memory").unwrap();
        assert!(!f.exists("/memory"));
        assert_eq!(f.node_unit_path("memory"), Ok(Vec::new()));
        // A node added after a NOPed one goes in front of it, past the root's properties.
        f.setprop_cell("/", "#size-cells", 2).unwrap();
        f.add_subnode("/memory@40000000").unwrap();
        assert!(f.exists("/memory@40000000"));
        assert_eq!(
            f.getprop("/", "model"),
            Err("qemu_fdt_getprop: Couldn't get //model: FDT_ERR_NOTFOUND".to_string())
        );
        assert_eq!(
            f.setprop("/nope", "x", &[]),
            Err("findnode_nofail Couldn't find node /nope: FDT_ERR_NOTFOUND".to_string())
        );
        assert_eq!(f.getprop_cell("/", "#size-cells"), Ok(2));
    }

    #[test]
    fn open_into_keeps_or_repacks() {
        let mut f = Fdt::new();
        f.add_subnode("/chosen").unwrap();
        f.setprop_string("/chosen", "bootargs", "quiet").unwrap();
        // A packed copy: header, reserve map, structure, strings.
        let src = f.as_bytes();
        let st = word(src, 8) as usize;
        let st_size = word(src, 36) as usize;
        let strs = word(src, 12) as usize;
        let strs_size = word(src, 32) as usize;
        let mut packed = src[..strs + strs_size].to_vec();
        packed[4..8].copy_from_slice(&((strs + strs_size) as u32).to_be_bytes());
        let g = Fdt::open_into(&packed, 4096).unwrap();
        assert_eq!(g.as_bytes().len(), 4096);
        assert_eq!(g.check_header(), Ok(()));
        assert_eq!(g.getprop("/chosen", "bootargs"), Ok(&b"quiet\0"[..]));
        assert_eq!(word(g.as_bytes(), 8) as usize, st);

        // Strings before the structure block: misordered, so the blocks are repacked.
        let mut mis = vec![0u8; HEADER_SIZE];
        let rsv = vec![0u8; 16];
        let strings = &src[strs..strs + strs_size];
        let structure = &src[st..st + st_size];
        let strings_off = HEADER_SIZE + rsv.len();
        let struct_off = strings_off + strings.len().next_multiple_of(4);
        mis.extend_from_slice(&rsv);
        mis.extend_from_slice(strings);
        mis.resize(struct_off, 0);
        mis.extend_from_slice(structure);
        for (field, v) in [
            (H_MAGIC, FDT_MAGIC),
            (H_TOTALSIZE, mis.len() as u32),
            (H_OFF_DT_STRUCT, struct_off as u32),
            (H_OFF_DT_STRINGS, strings_off as u32),
            (H_OFF_MEM_RSVMAP, HEADER_SIZE as u32),
            (H_VERSION, 17),
            (H_LAST_COMP_VERSION, 16),
            (H_SIZE_DT_STRINGS, strings.len() as u32),
            (H_SIZE_DT_STRUCT, structure.len() as u32),
        ] {
            mis[field..field + 4].copy_from_slice(&v.to_be_bytes());
        }
        let g = Fdt::open_into(&mis, 4096).unwrap();
        assert_eq!(g.check_header(), Ok(()));
        assert_eq!(word(g.as_bytes(), 16) as usize, HEADER_SIZE);
        assert_eq!(word(g.as_bytes(), 8) as usize, HEADER_SIZE + 16);
        assert_eq!(g.getprop("/chosen", "bootargs"), Ok(&b"quiet\0"[..]));

        assert_eq!(Fdt::open_into(b"not a dtb", 4096).unwrap_err(), FdtError::BadMagic);
        assert_eq!(Fdt::open_into(&packed, 64).unwrap_err(), FdtError::NoSpace);
    }

    #[test]
    fn broken_structure_is_an_error() {
        let mut f = Fdt::new();
        f.add_subnode("/a").unwrap();
        // Make the property length of a new property run past the block.
        f.setprop("/a", "p", &[1]).unwrap();
        let a = f.path_offset("/a").unwrap();
        let (prop, _) = f.get_property(a, b"p").unwrap();
        let abs = f.off_struct() + prop;
        f.put_word(abs + 4, 0x1000);
        assert_eq!(f.getprop_at(a, "p"), Err(FdtError::BadStructure));
        assert!(f.node_unit_path("a").unwrap_err().ends_with("FDT_ERR_BADSTRUCTURE"));
    }

    #[test]
    fn sized_cells_like_qemu() {
        assert_eq!(
            sized_cells(&[(2, 0x4000_0000), (1, 5)]).unwrap(),
            [0, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 5]
        );
        assert_eq!(sized_cells(&[(1, 1 << 32)]), None);
        assert_eq!(sized_cells(&[(3, 0)]), None);
    }
}
