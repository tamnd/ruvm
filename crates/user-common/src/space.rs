// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest address space: the reservation, the page flags of `accel/tcg/user-exec.c` and the
//! mmap engine of `linux-user/mmap.c` for a reserved guest (`reserved_va`).
//!
//! The whole guest lives in one host reservation from [`HostMemory::reserve`], and guest
//! address `g` is byte `g` of it, so `guest_base` is the reservation's host address. The
//! emulator owns every page of it: `mmap()` finds room the way QEMU does with `reserved_va`
//! (first fit from `task_unmapped_base`, then from `mmap_min_addr`) and replaces the pages
//! with [`HostMemory::map_fixed`], and `munmap()` puts fresh zero pages back.
//!
//! The guest's protection lives in the page flags, which the CPU checks when it fills its TLB
//! and the syscall layer checks for every pointer it is given, as `access_ok()` does. The host
//! pages follow them too, as `target_mprotect()` makes them: nothing where the guest has
//! nothing mapped or mapped `PROT_NONE`, read where it may read or run, and read and write
//! where it may write. Generated code that reaches guest memory directly, without the TLB,
//! then faults on the host exactly where the guest must, and the emulator finds out why from
//! the flags. A page with translated code on it stays read only on the host while the code
//! lives ([`GuestSpace::protect_code`]), so that stores to it fault and drop the code first,
//! as `page_protect()` arranges. The reservation ends with a page the guest never gets, so
//! that an access at its last byte faults rather than run past it.
//!
//! Deliberate differences from QEMU:
//!
//! - QEMU only reserves the guest space for 64-bit guests with `-R`; here it is always
//!   reserved, as large as the host allows up to [`DEFAULT_RESERVE`]. Addresses therefore
//!   follow QEMU's `-R` layout, not that of a plain `qemu-x86_64`.
//! - A shared file mapping the guest cannot write is a private one, which only differs if
//!   someone else writes the file.
//! - Pages of a file mapping past the end of the file read as zero where Linux raises SIGBUS.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::os::fd::RawFd;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};

use ruvm_mem::RamBlock;
use ruvm_sys::HostMemory;

/// `TARGET_PAGE_SIZE` for now: every guest and host so far has 4 KiB pages.
pub const PAGE_SIZE: u64 = 4096;

/// The guest space reserved when nothing else is asked for: 64 TiB, half the user space of
/// an x86_64 or aarch64 Linux process, so the emulator keeps the other half.
pub const DEFAULT_RESERVE: u64 = 1 << 46;

/// The smallest reservation tried before giving up.
const MIN_RESERVE: u64 = 1 << 32;

/// The inaccessible bytes past the end of the guest space, enough for the widest access that
/// starts inside it.
const GUARD: u64 = PAGE_SIZE;

/// `PAGE_READ`, `PAGE_WRITE`, `PAGE_EXEC` and the bookkeeping flags of a guest page.
pub mod page {
    /// The guest may read the page.
    pub const READ: u32 = 1;
    /// The guest may write the page.
    pub const WRITE: u32 = 2;
    /// The guest may run code from the page.
    pub const EXEC: u32 = 4;
    /// `PAGE_RWX`.
    pub const RWX: u32 = READ | WRITE | EXEC;
    /// `PAGE_ANON`: anonymous memory.
    pub const ANON: u32 = 0x80;
    /// A `MAP_SHARED` mapping, of a file or anonymous.
    pub const SHARED: u32 = 0x100;
}

/// `PAGE_ALIGN()`, or `None` when it overflows.
pub fn page_align(v: u64) -> Option<u64> {
    v.checked_add(PAGE_SIZE - 1).map(|v| v & !(PAGE_SIZE - 1))
}

/// What kind of `mmap()` the guest asked for, the host meaning of its flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MapKind {
    /// `MAP_FIXED`.
    pub fixed: bool,
    /// `MAP_FIXED_NOREPLACE`.
    pub noreplace: bool,
    /// `MAP_SHARED` or `MAP_SHARED_VALIDATE`.
    pub shared: bool,
    /// `MAP_ANONYMOUS`.
    pub anon: bool,
}

/// One mapped range of guest pages, `PageFlagsNode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Range {
    end: u64,
    flags: u32,
}

struct Inner {
    /// Non-overlapping ranges keyed by their start.
    ranges: BTreeMap<u64, Range>,
    /// The pages with translated code on them, which the host keeps read only.
    code: BTreeSet<u64>,
}

impl Inner {
    /// Splits the range that has `addr` strictly inside it in two.
    fn split_at(&mut self, addr: u64) {
        let Some((&start, &r)) = self.ranges.range(..addr).next_back() else { return };
        if r.end > addr {
            self.ranges.insert(start, Range { end: addr, flags: r.flags });
            self.ranges.insert(addr, Range { end: r.end, flags: r.flags });
        }
    }

    /// Drops every page in `start..end`.
    fn clear(&mut self, start: u64, end: u64) {
        self.split_at(start);
        self.split_at(end);
        let keys: Vec<u64> = self.ranges.range(start..end).map(|(&k, _)| k).collect();
        for k in keys {
            self.ranges.remove(&k);
        }
    }

    /// `page_set_flags()` over `start..end`, merging with equal neighbours.
    fn set(&mut self, start: u64, end: u64, flags: u32) {
        self.clear(start, end);
        let mut start = start;
        let mut end = end;
        if let Some((&ps, &p)) = self.ranges.range(..start).next_back() {
            if p.end == start && p.flags == flags {
                self.ranges.remove(&ps);
                start = ps;
            }
        }
        if let Some(&n) = self.ranges.get(&end) {
            if n.flags == flags {
                self.ranges.remove(&end);
                end = n.end;
            }
        }
        self.ranges.insert(start, Range { end, flags });
    }

    /// The first range that overlaps `start..end`, `pageflags_find()`.
    fn first_overlap(&self, start: u64, end: u64) -> Option<(u64, Range)> {
        if let Some((&s, &r)) = self.ranges.range(..=start).next_back() {
            if r.end > start {
                return Some((s, r));
            }
        }
        self.ranges.range(start..end).next().map(|(&s, &r)| (s, r))
    }

    /// Whether every page of `start..end` is mapped with all of `need`.
    fn check(&self, start: u64, end: u64, need: u32) -> bool {
        self.all(start, end, |f| f & need == need)
    }

    /// Whether every page of `start..end` is mapped with flags for which `ok` holds.
    fn all(&self, start: u64, end: u64, ok: impl Fn(u32) -> bool) -> bool {
        let mut at = start;
        while at < end {
            match self.first_overlap(at, at + 1) {
                Some((_, r)) if ok(r.flags) => at = r.end,
                _ => return false,
            }
        }
        true
    }

    /// Whether the host lets every page of `start..end` be written as it is.
    fn host_writable(&self, start: u64, end: u64) -> bool {
        self.check(start, end, page::WRITE) && self.code.range(start..end).next().is_none()
    }

    /// `page_find_range_empty()`: the lowest `align`ed address from `min` where `len` bytes
    /// are free and end at or below `max` (inclusive).
    fn find_empty(&self, min: u64, max: u64, len: u64, align: u64) -> Option<u64> {
        let mut min = min;
        loop {
            min = min.checked_add(align - 1)? & !(align - 1);
            if min > max || len - 1 > max - min {
                return None;
            }
            match self.first_overlap(min, min + len) {
                None => return Some(min),
                Some((_, r)) => {
                    if max < r.end {
                        return None;
                    }
                    min = r.end;
                }
            }
        }
    }
}

/// Called with the guest range whose bytes changed behind the CPU's back, to drop translated
/// code there.
pub type CodeHook = Box<dyn Fn(u64, u64) + Send + Sync>;

/// The address space of a user mode guest.
pub struct GuestSpace {
    block: Arc<RamBlock>,
    /// One past the last guest address, `reserved_va + 1`.
    size: u64,
    inner: Mutex<Inner>,
    /// `mmap_min_addr`.
    min_addr: u64,
    /// `task_unmapped_base`, where `mmap()` without a hint starts looking.
    unmapped_base: u64,
    /// `elf_et_dyn_base`, where a PIE executable with an interpreter goes.
    et_dyn_base: u64,
    code_hook: OnceLock<CodeHook>,
}

impl fmt::Debug for GuestSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuestSpace")
            .field("size", &self.size)
            .field("ranges", &self.lock().ranges.len())
            .finish_non_exhaustive()
    }
}

fn mmap_min_addr() -> u64 {
    let v = std::fs::read_to_string("/proc/sys/vm/mmap_min_addr")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(PAGE_SIZE);
    page_align(v.max(PAGE_SIZE)).unwrap_or(PAGE_SIZE)
}

fn errno_of(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(ENOMEM)
}

/// `EINVAL`.
const EINVAL: i32 = 22;
/// `ENOMEM`.
const ENOMEM: i32 = 12;
/// `EEXIST`.
const EEXIST: i32 = 17;

impl GuestSpace {
    /// Reserves up to `want` bytes of guest space, halving the size until the host gives it,
    /// and lays it out as `main.c` does for `reserved_va`, from the target's
    /// `TASK_UNMAPPED_BASE` and `ELF_ET_DYN_BASE`.
    pub fn new(
        want: u64,
        task_unmapped_base: u64,
        elf_et_dyn_base: u64,
    ) -> std::io::Result<GuestSpace> {
        let mut size = want.max(MIN_RESERVE);
        let mem = loop {
            match HostMemory::reserve((size + GUARD) as usize) {
                Ok(m) => break m,
                Err(e) if size / 2 < MIN_RESERVE => return Err(e),
                Err(_) => size /= 2,
            }
        };
        // Nothing is mapped yet.
        mem.protect(0, mem.len(), false, false)?;
        let block = Arc::new(RamBlock::from_memory("guest", mem, 12));
        let reserved_va = size - 1;
        let third = page_align(reserved_va / 3).unwrap_or(0);
        // QEMU falls back for each base on its own, but when only ELF_ET_DYN_BASE is out of
        // range (x86_64 and riscv64 in the default reservation) its replacement lands right on
        // TASK_UNMAPPED_BASE: the stack then follows the executable and leaves brk no room.
        // Falling back together keeps the two apart, in the kernel's order.
        let fits = task_unmapped_base < reserved_va && elf_et_dyn_base < reserved_va;
        let (unmapped_base, et_dyn_base) =
            if fits { (task_unmapped_base, elf_et_dyn_base) } else { (third, third * 2) };
        Ok(GuestSpace {
            block,
            size,
            inner: Mutex::new(Inner { ranges: BTreeMap::new(), code: BTreeSet::new() }),
            min_addr: mmap_min_addr(),
            unmapped_base,
            et_dyn_base,
            code_hook: OnceLock::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The RAM block that holds the guest, with guest address `g` at offset `g`.
    pub fn block(&self) -> &Arc<RamBlock> {
        &self.block
    }

    /// The size of the guest space, one past its last address.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// `g2h()`: the host address of guest address `g`.
    pub fn g2h(&self, g: u64) -> usize {
        self.block.host_addr() + g as usize
    }

    /// `guest_base` and the number of low address bits that are always inside the space:
    /// an access at an address below `1 << bits` that faults on the host faults for the guest.
    pub fn guest_base(&self) -> (u64, u32) {
        (self.g2h(0) as u64, 63 - self.size.leading_zeros())
    }

    /// `task_unmapped_base`.
    pub fn unmapped_base(&self) -> u64 {
        self.unmapped_base
    }

    /// `elf_et_dyn_base`.
    pub fn et_dyn_base(&self) -> u64 {
        self.et_dyn_base
    }

    /// Sets what to call when bytes of the guest change outside the CPU. Only the first call
    /// counts.
    pub fn set_code_hook(&self, hook: CodeHook) {
        let _ = self.code_hook.set(hook);
    }

    /// Tells the code hook that `len` bytes at `start` changed, as the kernel does when a
    /// system call writes guest memory.
    pub fn notify_write(&self, start: u64, len: u64) {
        if len == 0 {
            return;
        }
        if let Some(h) = self.code_hook.get() {
            h(start, len);
        }
    }

    /// `guest_range_valid_untagged()`.
    pub fn range_valid(&self, start: u64, len: u64) -> bool {
        start.checked_add(len).is_some_and(|end| end <= self.size)
    }

    /// `page_get_flags()`: the flags of the page at `addr`, 0 when nothing is mapped there.
    pub fn page_flags(&self, addr: u64) -> u32 {
        let a = addr & !(PAGE_SIZE - 1);
        self.lock().first_overlap(a, a + 1).map_or(0, |(_, r)| r.flags)
    }

    /// `page_check_range()`: whether every byte of `len` at `addr` is mapped with all of the
    /// [`page`] bits in `need`. An empty range always is. When `need` has [`page::WRITE`],
    /// translated code in the range is dropped, so that the host lets the bytes be written.
    pub fn check(&self, addr: u64, len: u64, need: u32) -> bool {
        if len == 0 {
            return true;
        }
        let Some(end) = addr.checked_add(len) else { return false };
        if end > self.size {
            return false;
        }
        let start = addr & !(PAGE_SIZE - 1);
        let ok = self.lock().check(start, end, need);
        if ok && need & page::WRITE != 0 {
            self.drop_code(start, end);
        }
        ok
    }

    /// `page_unprotect()` for every page of `start..end` with translated code on it: the code
    /// goes, and with it the host's write protection.
    fn drop_code(&self, start: u64, end: u64) {
        let pages: Vec<u64> = self.lock().code.range(start..end).copied().collect();
        for p in pages {
            self.notify_write(p, PAGE_SIZE);
        }
    }

    /// `page_protect()` when `protect`, else the end of it: translated code from the page at
    /// `addr` now exists, or no longer does. While it does, the host keeps the page read only,
    /// so that a store from generated code faults and the emulator drops the code before it
    /// lets the store happen.
    pub fn protect_code(&self, addr: u64, protect: bool) {
        let page = addr & !(PAGE_SIZE - 1);
        if page >= self.size {
            return;
        }
        let mut inner = self.lock();
        let changed = if protect { inner.code.insert(page) } else { inner.code.remove(&page) };
        if changed {
            let _ = self.sync_host(&inner, page, page + PAGE_SIZE);
        }
    }

    /// Gives the host pages of `start..end` the protection the flags and the translated code
    /// call for.
    fn sync_host(&self, inner: &Inner, start: u64, end: u64) -> Result<(), i32> {
        let mem = self.block.host_memory();
        let set = |s: u64, e: u64, flags: u32| {
            let read = flags & page::RWX != 0;
            mem.protect(s as usize, (e - s) as usize, read, flags & page::WRITE != 0)
                .map_err(|e| errno_of(&e))
        };
        let mut at = start;
        while at < end {
            match inner.first_overlap(at, end) {
                Some((s, r)) if s <= at => {
                    let e = r.end.min(end);
                    set(at, e, r.flags)?;
                    at = e;
                }
                Some((s, _)) => {
                    set(at, s, 0)?;
                    at = s;
                }
                None => {
                    set(at, end, 0)?;
                    at = end;
                }
            }
        }
        for &p in inner.code.range(start..end) {
            if let Some((_, r)) = inner.first_overlap(p, p + 1) {
                if r.flags & page::WRITE != 0 {
                    set(p, p + PAGE_SIZE, page::READ)?;
                }
            }
        }
        Ok(())
    }

    /// `page_check_range_empty()`.
    pub fn is_empty(&self, start: u64, len: u64) -> bool {
        self.lock().first_overlap(start, start + len).is_none()
    }

    /// The mapped ranges with their flags, in address order, for `/proc/self/maps`.
    pub fn ranges(&self) -> Vec<(u64, u64, u32)> {
        self.lock().ranges.iter().map(|(&s, r)| (s, r.end, r.flags)).collect()
    }

    /// `page_set_flags()` for memory the caller filled itself.
    pub fn set_flags(&self, start: u64, len: u64, flags: u32) {
        let _ = self.try_set_flags(start, len, flags);
    }

    /// [`set_flags`](Self::set_flags), failing when the host cannot protect the pages so.
    fn try_set_flags(&self, start: u64, len: u64, flags: u32) -> Result<(), i32> {
        if len == 0 {
            return Ok(());
        }
        let mut inner = self.lock();
        inner.set(start, start + len, flags);
        self.sync_host(&inner, start, start + len)
    }

    /// `mmap_find_vma()` with `reserved_va`: room for `size` bytes `align`ed, searching up
    /// from `start` (or `task_unmapped_base` for 0) and then from `mmap_min_addr`.
    pub fn find_vma(&self, start: u64, size: u64, align: u64) -> Option<u64> {
        let align = align.max(PAGE_SIZE);
        let start = if start == 0 { self.unmapped_base } else { start & !(PAGE_SIZE - 1) };
        let start = start.checked_add(align - 1)? & !(align - 1);
        let size = page_align(size)?;
        if size == 0 {
            return None;
        }
        let reserved_va = self.size - 1;
        let inner = self.lock();
        let mut ret = None;
        if start <= reserved_va {
            ret = inner.find_empty(start, reserved_va, size, align);
        }
        if ret.is_none() && start > self.min_addr {
            ret = inner.find_empty(self.min_addr, (start - 1).min(reserved_va), size, align);
        }
        ret
    }

    fn replace(
        &self,
        start: u64,
        len: u64,
        fd: Option<RawFd>,
        off: u64,
        shared: bool,
    ) -> Result<(), i32> {
        self.block
            .host_memory()
            .map_fixed(start as usize, len as usize, fd, off, shared)
            .map_err(|e| errno_of(&e))?;
        self.notify_write(start, len);
        Ok(())
    }

    /// `target_mmap()`: maps `len` bytes with the guest protection `prot` ([`page`] bits),
    /// anonymous or from `fd` at `offset`. Returns the guest address or a positive errno.
    pub fn mmap(
        &self,
        start: u64,
        len: u64,
        prot: u32,
        kind: MapKind,
        fd: Option<RawFd>,
        offset: u64,
    ) -> Result<u64, i32> {
        if len == 0 || prot & !page::RWX != 0 {
            return Err(EINVAL);
        }
        let len = match page_align(len) {
            Some(l) if l != 0 => l,
            _ => return Err(ENOMEM),
        };
        if !kind.anon && offset % PAGE_SIZE != 0 {
            return Err(EINVAL);
        }
        let mut start = start;
        if kind.fixed || kind.noreplace {
            if start % PAGE_SIZE != 0 {
                return Err(EINVAL);
            }
            if !self.range_valid(start, len) {
                return Err(ENOMEM);
            }
            if kind.noreplace && !self.is_empty(start, len) {
                return Err(EEXIST);
            }
        } else {
            start = self.find_vma(start, len, PAGE_SIZE).ok_or(ENOMEM)?;
        }
        let file = if kind.anon { None } else { Some(fd.ok_or(9)?) };
        let host_shared = kind.shared && (kind.anon || prot & page::WRITE != 0);
        self.replace(start, len, file, if kind.anon { 0 } else { offset }, host_shared)?;
        let mut flags = prot;
        if kind.anon {
            flags |= page::ANON;
        }
        if kind.shared {
            flags |= page::SHARED;
        }
        self.try_set_flags(start, len, flags)?;
        Ok(start)
    }

    /// `target_munmap()`.
    pub fn munmap(&self, start: u64, len: u64) -> Result<(), i32> {
        if start % PAGE_SIZE != 0 || len == 0 {
            return Err(EINVAL);
        }
        let len = page_align(len).ok_or(EINVAL)?;
        if !self.range_valid(start, len) {
            return Err(EINVAL);
        }
        self.unmap_mapped(start, len)
    }

    /// Puts zero pages back over whatever is mapped in `start..start + len` and forgets it.
    fn unmap_mapped(&self, start: u64, len: u64) -> Result<(), i32> {
        let end = start + len;
        let mapped: Vec<(u64, u64)> = {
            let inner = self.lock();
            let mut v = Vec::new();
            let mut at = start;
            while let Some((s, r)) = inner.first_overlap(at, end) {
                v.push((s.max(start), r.end.min(end)));
                at = r.end;
                if at >= end {
                    break;
                }
            }
            v
        };
        for (s, e) in mapped {
            self.replace(s, e - s, None, 0, false)?;
        }
        let mut inner = self.lock();
        inner.clear(start, end);
        self.sync_host(&inner, start, end)
    }

    /// `target_mprotect()`.
    pub fn mprotect(&self, start: u64, len: u64, prot: u32) -> Result<(), i32> {
        if start % PAGE_SIZE != 0 || prot & !page::RWX != 0 {
            return Err(EINVAL);
        }
        let len = page_align(len).ok_or(ENOMEM)?;
        if len == 0 {
            return Ok(());
        }
        if !self.range_valid(start, len) {
            return Err(ENOMEM);
        }
        let end = start + len;
        let mut inner = self.lock();
        if !inner.check(start, end, 0) {
            return Err(ENOMEM);
        }
        inner.split_at(start);
        inner.split_at(end);
        let parts: Vec<(u64, Range)> =
            inner.ranges.range(start..end).map(|(&s, &r)| (s, r)).collect();
        for (s, r) in parts {
            inner.set(s, r.end, (r.flags & !page::RWX) | prot);
        }
        self.sync_host(&inner, start, end)
    }

    /// `target_mremap()` with `reserved_va`. `new_addr` is only used with `fixed`.
    pub fn mremap(
        &self,
        old: u64,
        old_size: u64,
        new_size: u64,
        maymove: bool,
        fixed: bool,
        new_addr: u64,
    ) -> Result<u64, i32> {
        if old % PAGE_SIZE != 0 || (fixed && (!maymove || new_addr % PAGE_SIZE != 0)) {
            return Err(EINVAL);
        }
        let old_size = page_align(old_size).ok_or(EINVAL)?;
        let new_size = page_align(new_size).ok_or(EINVAL)?;
        if new_size == 0 || !self.range_valid(old, old_size) {
            return Err(EINVAL);
        }
        if fixed && !self.range_valid(new_addr, new_size) {
            return Err(EINVAL);
        }
        if !self.check(old, old_size.max(1), 0) {
            return Err(14); // EFAULT, as the kernel says for an unmapped old range.
        }
        let flags = self.page_flags(old);
        if fixed || maymove {
            let to = if fixed {
                if new_addr < old + old_size && old < new_addr + new_size {
                    return Err(EINVAL);
                }
                self.unmap_mapped(new_addr, new_size)?;
                new_addr
            } else {
                self.find_vma(0, new_size, PAGE_SIZE).ok_or(ENOMEM)?
            };
            let moved = old_size.min(new_size);
            self.move_range(old, to, moved);
            // The old range ends up unmapped and the new one holds its pages. It may start or
            // end inside a mapping merged with its neighbours, so split those first.
            let parts: Vec<(u64, u64, u32)> = {
                let mut inner = self.lock();
                inner.split_at(old);
                inner.split_at(old + moved);
                inner
                    .ranges
                    .range(old..old + moved)
                    .map(|(&s, r)| (s, r.end.min(old + moved), r.flags))
                    .collect()
            };
            self.unmap_mapped(old, old_size)?;
            for (s, e, f) in parts {
                self.set_flags(to + (s - old), e - s, f);
            }
            if new_size > moved {
                self.set_flags(to + moved, new_size - moved, flags);
            }
            return Ok(to);
        }
        if new_size > old_size {
            let tail = old + old_size;
            if !self.range_valid(tail, new_size - old_size)
                || !self.is_empty(tail, new_size - old_size)
            {
                return Err(ENOMEM);
            }
            self.replace(tail, new_size - old_size, None, 0, false)?;
            self.set_flags(tail, new_size - old_size, flags);
        } else if new_size < old_size {
            self.unmap_mapped(old + new_size, old_size - new_size)?;
        }
        Ok(old)
    }

    /// Moves the bytes of `len` at `from` to `to`, by moving the host pages when the kernel
    /// can and by copying otherwise. `to` holds zero pages beforehand.
    fn move_range(&self, from: u64, to: u64, len: u64) {
        let mem = self.block.host_memory();
        if mem.move_pages(from as usize, to as usize, len as usize).is_err() {
            // The caller gives both ranges the protection of their flags afterwards.
            let _ = mem.protect(from as usize, len as usize, true, false);
            let _ = mem.protect(to as usize, len as usize, true, true);
            let mut buf = vec![0u8; 1 << 20];
            let mut done = 0;
            while done < len {
                let n = (len - done).min(buf.len() as u64) as usize;
                let _ = self.block.read(from + done, &mut buf[..n]);
                let _ = self.block.write(to + done, &buf[..n]);
                done += n as u64;
            }
        }
        self.notify_write(to, len);
    }

    /// Reads guest memory the guest may not read, as the emulator itself may. Only pages the
    /// host can read, mapped with some access, can be read so.
    pub fn read_raw(&self, addr: u64, buf: &mut [u8]) -> bool {
        let len = buf.len() as u64;
        if !self.range_valid(addr, len) {
            return false;
        }
        if len == 0 {
            return true;
        }
        let inner = self.lock();
        inner.all(addr & !(PAGE_SIZE - 1), addr + len, |f| f & page::RWX != 0)
            && self.block.read(addr, buf).is_ok()
    }

    /// Writes mapped guest memory the guest may not write. The host protection is lifted
    /// while the bytes are written, so another thread of the guest could write the pages
    /// then too; only the loader and the signal code use this.
    pub fn write_raw(&self, addr: u64, buf: &[u8]) -> bool {
        let len = buf.len() as u64;
        if !self.range_valid(addr, len) {
            return false;
        }
        if len == 0 {
            return true;
        }
        let (start, end) = (addr & !(PAGE_SIZE - 1), page_align(addr + len).unwrap_or(self.size));
        self.drop_code(start, end);
        let ok = {
            let inner = self.lock();
            if !inner.check(start, end, 0) {
                return false;
            }
            let lift = !inner.host_writable(start, end);
            let mem = self.block.host_memory();
            if lift && mem.protect(start as usize, (end - start) as usize, true, true).is_err() {
                return false;
            }
            let ok = self.block.write(addr, buf).is_ok();
            if lift {
                let _ = self.sync_host(&inner, start, end);
            }
            ok
        };
        if ok {
            self.notify_write(addr, len);
        }
        ok
    }

    /// `copy_from_user()`: reads guest memory the guest may read.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.check(addr, buf.len() as u64, page::READ) && self.read_raw(addr, buf)
    }

    /// `copy_to_user()`: writes guest memory the guest may write.
    pub fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.check(addr, buf.len() as u64, page::WRITE) && self.write_raw(addr, buf)
    }

    /// Reads a NUL terminated string the guest may read, of at most `max` bytes.
    pub fn read_cstr(&self, addr: u64, max: usize) -> Option<Vec<u8>> {
        let mut out = Vec::new();
        let mut at = addr;
        while out.len() < max {
            let page_end = (at | (PAGE_SIZE - 1)) + 1;
            let n = ((page_end - at) as usize).min(max - out.len());
            let mut chunk = vec![0u8; n];
            if !self.read(at, &mut chunk) {
                return None;
            }
            if let Some(i) = chunk.iter().position(|&b| b == 0) {
                out.extend_from_slice(&chunk[..i]);
                return Some(out);
            }
            out.extend_from_slice(&chunk);
            at = page_end;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TUB: u64 = 0x2aaa_aaaa_b000;

    fn space() -> GuestSpace {
        GuestSpace::new(1 << 36, TUB, TUB * 2).unwrap()
    }

    const ANON: MapKind = MapKind { fixed: false, noreplace: false, shared: false, anon: true };
    const FIXED_ANON: MapKind = MapKind { fixed: true, ..ANON };

    #[test]
    fn layout_follows_reserved_va() {
        let s = space();
        assert_eq!(s.size(), 1 << 36);
        assert_eq!(s.unmapped_base(), page_align(((1u64 << 36) - 1) / 3).unwrap());
        assert_eq!(s.et_dyn_base(), s.unmapped_base() * 2);
        let big = GuestSpace::new(1 << 46, TUB, TUB * 2).unwrap();
        if big.size() == 1 << 46 {
            assert_eq!(big.unmapped_base(), 0x1555_5555_6000);
            assert_eq!(big.et_dyn_base(), 0x2aaa_aaaa_c000);
        }
        drop(big);
        let fits = GuestSpace::new(1 << 46, TUB / 2, TUB).unwrap();
        if fits.size() == 1 << 46 {
            assert_eq!((fits.unmapped_base(), fits.et_dyn_base()), (TUB / 2, TUB));
        }
    }

    #[test]
    fn mmap_first_fit_and_flags() {
        let s = space();
        let a = s.mmap(0, 3 * PAGE_SIZE, page::READ | page::WRITE, ANON, None, 0).unwrap();
        assert_eq!(a, s.unmapped_base());
        let b = s.mmap(0, 100, page::READ, ANON, None, 0).unwrap();
        assert_eq!(b, a + 3 * PAGE_SIZE);
        assert!(s.write(a + 5, b"hi"));
        assert!(!s.write(b, b"no"));
        assert!(s.check(a, 4 * PAGE_SIZE, page::READ));
        assert!(!s.check(a, 5 * PAGE_SIZE, page::READ));
        // A hole is reused first.
        s.munmap(a + PAGE_SIZE, PAGE_SIZE).unwrap();
        assert_eq!(s.page_flags(a + PAGE_SIZE), 0);
        let c = s.mmap(0, PAGE_SIZE, page::READ, ANON, None, 0).unwrap();
        assert_eq!(c, a + PAGE_SIZE);
        let mut buf = [1u8; 2];
        assert!(s.read(c, &mut buf));
        assert_eq!(buf, [0, 0]);
        assert_eq!(s.mmap(0, 0, page::READ, ANON, None, 0), Err(EINVAL));
        assert_eq!(s.mmap(0, 1, 8, ANON, None, 0), Err(EINVAL));
    }

    #[test]
    fn fixed_and_noreplace() {
        let s = space();
        let k = MapKind { fixed: true, ..ANON };
        assert_eq!(s.mmap(0x10000, PAGE_SIZE, page::READ, k, None, 0), Ok(0x10000));
        let nr = MapKind { noreplace: true, ..ANON };
        assert_eq!(s.mmap(0x10000, PAGE_SIZE, page::READ, nr, None, 0), Err(EEXIST));
        assert_eq!(s.mmap(0x11000, PAGE_SIZE, page::READ, nr, None, 0), Ok(0x11000));
        assert_eq!(s.mmap(0x11001, PAGE_SIZE, page::READ, k, None, 0), Err(EINVAL));
        assert_eq!(s.mmap(s.size(), PAGE_SIZE, page::READ, k, None, 0), Err(ENOMEM));
        // The two ranges merged.
        assert_eq!(s.ranges().len(), 1);
    }

    #[test]
    fn shared_anon_is_shared() {
        let s = space();
        let k = MapKind { shared: true, ..ANON };
        let a = s.mmap(0, PAGE_SIZE, page::READ | page::WRITE, k, None, 0).unwrap();
        assert_eq!(s.page_flags(a), page::READ | page::WRITE | page::ANON | page::SHARED);
        // The host's mapping behind it is shared, so a fork() keeps it so.
        let h = s.g2h(a) as u64;
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        let line = maps
            .lines()
            .find(|l| {
                let (lo, hi) = l.split_once(' ').unwrap().0.split_once('-').unwrap();
                u64::from_str_radix(lo, 16).unwrap() <= h
                    && h < u64::from_str_radix(hi, 16).unwrap()
            })
            .unwrap();
        assert_eq!(&line.split(' ').nth(1).unwrap()[3..], "s", "{line}");
    }

    #[test]
    fn mprotect_splits() {
        let s = space();
        let a = s.mmap(0, 4 * PAGE_SIZE, page::READ, ANON, None, 0).unwrap();
        s.mprotect(a + PAGE_SIZE, PAGE_SIZE, page::READ | page::WRITE).unwrap();
        assert_eq!(s.page_flags(a), page::READ | page::ANON);
        assert_eq!(s.page_flags(a + PAGE_SIZE), page::READ | page::WRITE | page::ANON);
        assert_eq!(s.ranges().len(), 3);
        assert_eq!(s.mprotect(a, 8 * PAGE_SIZE, page::READ), Err(ENOMEM));
    }

    #[test]
    fn mremap_moves_and_grows() {
        let s = space();
        let a = s.mmap(0, 2 * PAGE_SIZE, page::READ | page::WRITE, ANON, None, 0).unwrap();
        let _guard = s.mmap(0, PAGE_SIZE, page::READ, ANON, None, 0).unwrap();
        assert!(s.write(a + 10, b"data"));
        assert_eq!(s.mremap(a, 2 * PAGE_SIZE, 4 * PAGE_SIZE, false, false, 0), Err(ENOMEM));
        let b = s.mremap(a, 2 * PAGE_SIZE, 4 * PAGE_SIZE, true, false, 0).unwrap();
        assert_ne!(a, b);
        let mut buf = [0u8; 4];
        assert!(s.read(b + 10, &mut buf));
        assert_eq!(&buf, b"data");
        assert!(s.check(b, 4 * PAGE_SIZE, page::WRITE));
        assert_eq!(s.page_flags(a), 0);
        // Shrinking in place.
        assert_eq!(s.mremap(b, 4 * PAGE_SIZE, PAGE_SIZE, false, false, 0), Ok(b));
        assert_eq!(s.page_flags(b + PAGE_SIZE), 0);
    }

    #[test]
    fn mremap_moves_part_of_a_merged_mapping() {
        let s = space();
        let rw = page::READ | page::WRITE;
        let a = s.mmap(0, 2 * PAGE_SIZE, rw, ANON, None, 0).unwrap();
        let c = s.mmap(a + 2 * PAGE_SIZE, 2 * PAGE_SIZE, rw, FIXED_ANON, None, 0).unwrap();
        assert!(s.write(c + 10, b"data"));
        let b = s.mremap(c, 2 * PAGE_SIZE, 3 * PAGE_SIZE, true, false, 0).unwrap();
        assert_ne!(b, c);
        let mut buf = [0u8; 4];
        assert!(s.read(b + 10, &mut buf));
        assert_eq!(&buf, b"data");
        assert!(s.check(b, 3 * PAGE_SIZE, page::WRITE));
        assert!(s.check(a, 2 * PAGE_SIZE, page::WRITE));
        assert_eq!(s.page_flags(c), 0);
    }

    #[test]
    fn strings() {
        let s = space();
        let a = s.mmap(0, 2 * PAGE_SIZE, page::READ | page::WRITE, ANON, None, 0).unwrap();
        assert!(s.write(a + PAGE_SIZE - 3, b"abcdef\0"));
        assert_eq!(s.read_cstr(a + PAGE_SIZE - 3, 100).unwrap(), b"abcdef");
        assert!(s.read_cstr(a + PAGE_SIZE - 3, 4).is_none());
    }
    /// The host's permissions of the page holding `h`, as `/proc/self/maps` shows them.
    fn host_perms(h: usize) -> String {
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
        for line in maps.lines() {
            let (range, rest) = line.split_once(' ').unwrap();
            let (lo, hi) = range.split_once('-').unwrap();
            let (lo, hi) =
                (usize::from_str_radix(lo, 16).unwrap(), usize::from_str_radix(hi, 16).unwrap());
            if lo <= h && h < hi {
                return rest[..3].to_string();
            }
        }
        panic!("{h:#x} is not mapped");
    }

    #[test]
    fn host_protection_follows_the_flags() {
        let s = Arc::new(space());
        let w = Arc::downgrade(&s);
        s.set_code_hook(Box::new(move |a, l| {
            // What the translator does when the code of a page goes.
            if let Some(s) = w.upgrade() {
                let mut p = a & !(PAGE_SIZE - 1);
                while p < a + l {
                    s.protect_code(p, false);
                    p += PAGE_SIZE;
                }
            }
        }));
        let rw = page::READ | page::WRITE;
        let a = s.mmap(0, 3 * PAGE_SIZE, rw, ANON, None, 0).unwrap();
        let h = |g: u64| host_perms(s.g2h(g));
        assert_eq!(h(a), "rw-");
        assert_eq!(h(a + 3 * PAGE_SIZE), "---");
        let (base, bits) = s.guest_base();
        assert_eq!(base as usize, s.g2h(0));
        assert_eq!(1u64 << bits, s.size());
        assert_eq!(host_perms(s.g2h(s.size())), "---");
        s.mprotect(a + PAGE_SIZE, PAGE_SIZE, page::READ).unwrap();
        s.mprotect(a + 2 * PAGE_SIZE, PAGE_SIZE, page::EXEC).unwrap();
        assert_eq!(
            (h(a), h(a + PAGE_SIZE), h(a + 2 * PAGE_SIZE)),
            ("rw-".into(), "r--".into(), "r--".into())
        );
        // The loader may still write what the guest cannot.
        assert!(!s.write(a + PAGE_SIZE, b"no"));
        assert!(s.write_raw(a + PAGE_SIZE + 1, b"ro"));
        assert_eq!(h(a + PAGE_SIZE), "r--");
        let mut buf = [0u8; 2];
        assert!(s.read(a + PAGE_SIZE + 1, &mut buf));
        assert_eq!(&buf, b"ro");
        // Code on a writable page keeps it read only until something writes it.
        s.protect_code(a, true);
        assert_eq!(h(a), "r--");
        s.mprotect(a, PAGE_SIZE, rw).unwrap();
        assert_eq!(h(a), "r--");
        assert!(s.write(a + 8, b"x"));
        assert_eq!(h(a), "rw-");
        s.protect_code(a, true);
        assert!(s.check(a, 1, page::WRITE));
        assert_eq!(h(a), "rw-");
        s.mprotect(a, PAGE_SIZE, 0).unwrap();
        assert_eq!(h(a), "---");
        assert!(!s.read_raw(a, &mut buf));
        assert!(!s.write_raw(a + 3 * PAGE_SIZE, b"no"));
        s.munmap(a, 3 * PAGE_SIZE).unwrap();
        assert_eq!(h(a + PAGE_SIZE), "---");
    }
}
