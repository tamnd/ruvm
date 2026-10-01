// SPDX-License-Identifier: GPL-2.0-or-later

//! The softmmu TLB, ported from `accel/tcg/cputlb.c`.
//!
//! Each MMU index has a fast table that resizes itself on flush, a victim table of
//! [`CPU_VTLB_SIZE`] entries, and the full entries beside both. A miss calls the target's
//! `tlb_fill`, which installs a translation with [`tlb_set_page_full`]. Loads and stores take the
//! slow path through [`do_ld_bytes`] and [`do_st_bytes`], which handle watchpoints, notdirty
//! (self modifying code and dirty logging), MMIO and accesses that cross a page.
//!
//! Differences from QEMU:
//!
//! - The "host address" of RAM is its `ram_addr` (see the crate docs). An entry's `addend` turns
//!   a virtual address into the `ram_addr`, and the bytes are read from the `RamBlock` the full
//!   entry names.
//! - MMIO goes through the CPU's address space by physical address, not a saved section. Writes
//!   to a ROM device go the same way.
//! - Alignment is checked by the interpreter before the softmmu is called, and
//!   `TLB_CHECK_ALIGNED` does not add the atomicity requirement of the access.
//! - The used entry count saturates at zero instead of underflowing when the victim table
//!   flush counts an entry twice.
//! - The TLB lock is a mutex that the owning vCPU also takes on every access. It is never held
//!   while calling into the target or a device.

use std::sync::Arc;

use ruvm_jit_core::types::{MemOp, MemOpIdx};
use ruvm_mem::{DirtyClient, DirtyMask, Endian, MemTxAttrs, MemTxResult, RamBlock, RegionType};

use crate::cpu::{Cpu, CpuLoopExit, MmuAccessType, Ra};
use crate::jit::Jit;
use crate::tb::lock;
use crate::tb_maint::{
    tb_invalidate_phys_range_fast, tb_jmp_cache_clear_page, tcg_flush_jmp_cache,
};
use crate::{
    CPU_TLB_DYN_DEFAULT_BITS, CPU_TLB_DYN_MIN_BITS, CPU_TLB_ENTRY_BITS, CPU_VTLB_SIZE,
    TB_JMP_CACHE_SIZE, bp, page, tlb,
};

/// Where the memory behind a TLB entry is, QEMU's `MemoryRegionSection` pointer.
#[derive(Clone, Debug, Default)]
pub enum TlbSection {
    /// Device memory, reached through the address space.
    #[default]
    Io,
    /// RAM or ROM.
    Ram {
        /// The block.
        block: Arc<RamBlock>,
        /// The `ram_addr` of its first byte.
        ram_base: u64,
    },
    /// A ROM device in romd mode: reads from the block, writes to the device.
    Romd {
        /// The block.
        block: Arc<RamBlock>,
        /// The `ram_addr` of its first byte.
        ram_base: u64,
    },
}

/// `CPUTLBEntryFull`.
#[derive(Clone, Debug, Default)]
pub struct TlbEntryFull {
    /// The physical address of the page, or of the access when given to
    /// [`tlb_set_page_full`].
    pub phys_addr: u64,
    /// The transaction attributes.
    pub attrs: MemTxAttrs,
    /// `PAGE_*` protection bits.
    pub prot: u32,
    /// log2 of the translation's page size.
    pub lg_page_size: u8,
    /// Extra `TLB_*` flags from `tlb_fill`, such as [`tlb::BSWAP`].
    pub tlb_fill_flags: u64,
    /// The slow flags per access type. Set by [`tlb_set_page_full`].
    pub slow_flags: [u32; 3],
    /// Added to a virtual address to give the `ram_addr`. Set by [`tlb_set_page_full`].
    pub xlat_offset: u64,
    /// The memory. Set by [`tlb_set_page_full`].
    pub section: TlbSection,
}

/// `CPUTLBEntry`: the comparators per access type (`u64::MAX` is invalid) and the addend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TlbEntry {
    pub(crate) addr: [u64; 3],
    pub(crate) addend: u64,
}

const EMPTY: TlbEntry = TlbEntry { addr: [u64::MAX; 3], addend: u64::MAX };

impl TlbEntry {
    fn is_empty(&self) -> bool {
        self.addr == [u64::MAX; 3]
    }
}

pub(crate) struct TlbDesc {
    large_page_addr: u64,
    large_page_mask: u64,
    window_begin_ns: i64,
    window_max_entries: usize,
    n_used_entries: usize,
    vindex: usize,
    vtable: [TlbEntry; CPU_VTLB_SIZE],
    vfulltlb: Vec<TlbEntryFull>,
    fulltlb: Vec<TlbEntryFull>,
}

/// One vCPU's TLB, `CPUTLB`.
pub(crate) struct CpuTlb {
    dirty: u32,
    d: Vec<TlbDesc>,
    table: Vec<Vec<TlbEntry>>,
    pub(crate) full_flush_count: u64,
    pub(crate) part_flush_count: u64,
}

impl CpuTlb {
    pub(crate) fn new(nb_mmu_modes: usize, now: i64) -> CpuTlb {
        let n = 1usize << CPU_TLB_DYN_DEFAULT_BITS;
        let mut t = CpuTlb {
            dirty: 0,
            d: Vec::new(),
            table: Vec::new(),
            full_flush_count: 0,
            part_flush_count: 0,
        };
        for _ in 0..nb_mmu_modes {
            t.d.push(TlbDesc {
                large_page_addr: u64::MAX,
                large_page_mask: u64::MAX,
                window_begin_ns: now,
                window_max_entries: 0,
                n_used_entries: 0,
                vindex: 0,
                vtable: [EMPTY; CPU_VTLB_SIZE],
                vfulltlb: vec![TlbEntryFull::default(); CPU_VTLB_SIZE],
                fulltlb: vec![TlbEntryFull::default(); n],
            });
            t.table.push(vec![EMPTY; n]);
        }
        t
    }

    fn index(&self, page_bits: u32, mmu_idx: usize, addr: u64) -> usize {
        ((addr >> page_bits) as usize) & (self.table[mmu_idx].len() - 1)
    }

    /// `tlb_mmu_resize_locked()`.
    fn resize(&mut self, jit: &Jit, mmu_idx: usize, now: i64) {
        let max_bits = 22.min(jit.config.target_long_bits - jit.config.page_bits);
        let desc = &mut self.d[mmu_idx];
        let old_size = self.table[mmu_idx].len();
        let mut new_size = old_size;
        let window_len_ns = 100 * 1000 * 1000;
        let window_expired = now > desc.window_begin_ns + window_len_ns;
        if desc.n_used_entries > desc.window_max_entries {
            desc.window_max_entries = desc.n_used_entries;
        }
        let rate = desc.window_max_entries * 100 / old_size;
        if rate > 70 {
            new_size = (old_size << 1).min(1 << max_bits);
        } else if rate < 30 && window_expired {
            let mut ceil = desc.window_max_entries.next_power_of_two().max(1);
            let expected_rate = desc.window_max_entries * 100 / ceil;
            // Avoid undersizing when the max number of entries seen is just below a pow2.
            if expected_rate > 70 {
                ceil *= 2;
            }
            new_size = ceil.max(1 << CPU_TLB_DYN_MIN_BITS);
        }
        if new_size == old_size {
            if window_expired {
                desc.window_begin_ns = now;
                desc.window_max_entries = desc.n_used_entries;
            }
            return;
        }
        desc.window_begin_ns = now;
        desc.window_max_entries = 0;
        desc.fulltlb = vec![TlbEntryFull::default(); new_size];
        self.table[mmu_idx] = vec![EMPTY; new_size];
    }

    /// `tlb_mmu_flush_locked()`.
    fn flush_locked(&mut self, mmu_idx: usize) {
        let desc = &mut self.d[mmu_idx];
        desc.n_used_entries = 0;
        desc.large_page_addr = u64::MAX;
        desc.large_page_mask = u64::MAX;
        desc.vindex = 0;
        desc.vtable = [EMPTY; CPU_VTLB_SIZE];
        self.table[mmu_idx].fill(EMPTY);
    }

    /// `tlb_flush_one_mmuidx_locked()`.
    fn flush_one_mmuidx_locked(&mut self, jit: &Jit, mmu_idx: usize, now: i64) {
        self.resize(jit, mmu_idx, now);
        self.flush_locked(mmu_idx);
    }

    fn n_used_dec(&mut self, mmu_idx: usize) {
        let d = &mut self.d[mmu_idx];
        d.n_used_entries = d.n_used_entries.saturating_sub(1);
    }

    /// `tlb_flush_vtlb_page_mask_locked()`.
    fn flush_vtlb_page_mask_locked(&mut self, jit: &Jit, mmu_idx: usize, page: u64, mask: u64) {
        for k in 0..CPU_VTLB_SIZE {
            if flush_entry_mask_locked(jit, &mut self.d[mmu_idx].vtable[k], page, mask) {
                self.n_used_dec(mmu_idx);
            }
        }
    }

    /// `tlb_flush_page_locked()`.
    fn flush_page_locked(&mut self, jit: &Jit, mmu_idx: usize, page: u64) {
        let lp_addr = self.d[mmu_idx].large_page_addr;
        let lp_mask = self.d[mmu_idx].large_page_mask;
        // Check if we need to flush due to large pages.
        if page & lp_mask == lp_addr {
            self.flush_one_mmuidx_locked(jit, mmu_idx, jit.now_ns());
        } else {
            let i = self.index(jit.config.page_bits, mmu_idx, page);
            if flush_entry_mask_locked(jit, &mut self.table[mmu_idx][i], page, u64::MAX) {
                self.n_used_dec(mmu_idx);
            }
            self.flush_vtlb_page_mask_locked(jit, mmu_idx, page, u64::MAX);
        }
    }

    /// `tlb_flush_range_locked()`.
    fn flush_range_locked(&mut self, jit: &Jit, mmu_idx: usize, addr: u64, len: u64, bits: u32) {
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let f_mask = ((self.table[mmu_idx].len() as u64) - 1) << CPU_TLB_ENTRY_BITS;
        // If bits is smaller than the tlb size, there may be multiple entries within the TLB;
        // if len is larger than the tlb size, testing every entry costs more than a flush.
        if mask < f_mask || len > f_mask {
            self.flush_one_mmuidx_locked(jit, mmu_idx, jit.now_ns());
            return;
        }
        // Check if we need to flush due to large pages. Because large_page_mask contains all
        // 1's from the msb, we only need to test the end of the range.
        let d = &self.d[mmu_idx];
        if (addr.wrapping_add(len).wrapping_sub(1)) & d.large_page_mask == d.large_page_addr {
            self.flush_one_mmuidx_locked(jit, mmu_idx, jit.now_ns());
            return;
        }
        let mut i = 0;
        while i < len {
            let page = addr.wrapping_add(i);
            let idx = self.index(jit.config.page_bits, mmu_idx, page);
            if flush_entry_mask_locked(jit, &mut self.table[mmu_idx][idx], page, mask) {
                self.n_used_dec(mmu_idx);
            }
            self.flush_vtlb_page_mask_locked(jit, mmu_idx, page, mask);
            i += jit.page_size();
        }
    }
}

fn tlb_hit_page(jit: &Jit, tlb_addr: u64, page: u64) -> bool {
    page == tlb_addr & (jit.page_mask() | tlb::INVALID_MASK)
}

fn tlb_hit(jit: &Jit, tlb_addr: u64, addr: u64) -> bool {
    tlb_hit_page(jit, tlb_addr, addr & jit.page_mask())
}

fn tlb_hit_page_anyprot(jit: &Jit, e: &TlbEntry, page: u64) -> bool {
    e.addr.iter().any(|&a| tlb_hit_page(jit, a, page))
}

/// `tlb_flush_entry_mask_locked()`.
fn flush_entry_mask_locked(jit: &Jit, e: &mut TlbEntry, page: u64, mask: u64) -> bool {
    let page = page & mask;
    let mask = mask & (jit.page_mask() | tlb::INVALID_MASK);
    if e.addr.iter().any(|&a| page == a & mask) {
        *e = EMPTY;
        true
    } else {
        false
    }
}

fn all_mmuidx_bits(jit: &Jit) -> u32 {
    (1u32 << jit.config.nb_mmu_modes) - 1
}

/// `tlb_flush_by_mmuidx_async_work()`.
fn tlb_flush_by_mmuidx_async_work(cpu: &Cpu<'_>, idxmap: u32) {
    let jit = &cpu.core.jit;
    let shared = &cpu.core.shared;
    let now = jit.now_ns();
    {
        let mut t = lock(&shared.tlb);
        let all_dirty = t.dirty;
        let to_clean = idxmap & all_dirty;
        t.dirty = all_dirty & !to_clean;
        for mmu_idx in 0..jit.config.nb_mmu_modes {
            if to_clean & (1 << mmu_idx) != 0 {
                t.flush_one_mmuidx_locked(jit, mmu_idx, now);
            }
        }
        if to_clean == all_mmuidx_bits(jit) {
            t.full_flush_count += 1;
        } else {
            t.part_flush_count += u64::from(to_clean.count_ones());
        }
    }
    tcg_flush_jmp_cache(shared);
}

/// `tlb_flush_by_mmuidx()`: flush the MMU indexes in `idxmap`.
pub fn tlb_flush_by_mmuidx(cpu: &mut Cpu<'_>, idxmap: u32) {
    tlb_flush_by_mmuidx_async_work(cpu, idxmap);
}

/// `tlb_flush()`.
pub fn tlb_flush(cpu: &mut Cpu<'_>) {
    let all = all_mmuidx_bits(&cpu.core.jit);
    tlb_flush_by_mmuidx(cpu, all);
}

fn tlb_flush_page_by_mmuidx_async_0(cpu: &Cpu<'_>, addr: u64, idxmap: u32) {
    let jit = &cpu.core.jit;
    let shared = &cpu.core.shared;
    {
        let mut t = lock(&shared.tlb);
        for mmu_idx in 0..jit.config.nb_mmu_modes {
            if idxmap & (1 << mmu_idx) != 0 {
                t.flush_page_locked(jit, mmu_idx, addr);
            }
        }
    }
    // Discard jump cache entries for any tb which might potentially overlap the flushed page.
    tb_jmp_cache_clear_page(jit, shared, addr.wrapping_sub(jit.page_size()));
    tb_jmp_cache_clear_page(jit, shared, addr);
}

/// `tlb_flush_page_by_mmuidx()`.
pub fn tlb_flush_page_by_mmuidx(cpu: &mut Cpu<'_>, addr: u64, idxmap: u32) {
    let addr = addr & cpu.core.jit.page_mask();
    tlb_flush_page_by_mmuidx_async_0(cpu, addr, idxmap);
}

/// `tlb_flush_page()`.
pub fn tlb_flush_page(cpu: &mut Cpu<'_>, addr: u64) {
    let all = all_mmuidx_bits(&cpu.core.jit);
    tlb_flush_page_by_mmuidx(cpu, addr, all);
}

fn tlb_flush_range_by_mmuidx_async_0(cpu: &Cpu<'_>, addr: u64, len: u64, idxmap: u32, bits: u32) {
    let jit = &cpu.core.jit;
    let shared = &cpu.core.shared;
    {
        let mut t = lock(&shared.tlb);
        for mmu_idx in 0..jit.config.nb_mmu_modes {
            if idxmap & (1 << mmu_idx) != 0 {
                t.flush_range_locked(jit, mmu_idx, addr, len, bits);
            }
        }
    }
    // If the length is larger than the jump cache size, then it will take longer to clear
    // each entry individually than it will to clear it all.
    if len >= jit.page_size() * TB_JMP_CACHE_SIZE as u64 {
        tcg_flush_jmp_cache(shared);
        return;
    }
    // Discard jump cache entries for any tb which might potentially overlap the flushed pages,
    // which includes the previous.
    let mut a = addr.wrapping_sub(jit.page_size());
    for _ in 0..len / jit.page_size() + 1 {
        tb_jmp_cache_clear_page(jit, shared, a);
        a = a.wrapping_add(jit.page_size());
    }
}

/// `tlb_flush_range_by_mmuidx()`: flush `len` bytes from `addr`, comparing only the low `bits`
/// bits of the address.
pub fn tlb_flush_range_by_mmuidx(cpu: &mut Cpu<'_>, addr: u64, len: u64, idxmap: u32, bits: u32) {
    let jit = cpu.jit();
    // If all bits are significant, and len is small, this devolves to tlb_flush_page.
    if len <= jit.page_size() && bits >= jit.config.target_long_bits {
        tlb_flush_page_by_mmuidx(cpu, addr, idxmap);
        return;
    }
    // If no page bits are significant, this devolves to tlb_flush.
    if bits < jit.config.page_bits {
        tlb_flush_by_mmuidx(cpu, idxmap);
        return;
    }
    tlb_flush_range_by_mmuidx_async_0(cpu, addr & jit.page_mask(), len, idxmap, bits);
}

/// `tlb_flush_page_bits_by_mmuidx()`.
pub fn tlb_flush_page_bits_by_mmuidx(cpu: &mut Cpu<'_>, addr: u64, idxmap: u32, bits: u32) {
    let size = cpu.core.jit.page_size();
    tlb_flush_range_by_mmuidx(cpu, addr, size, idxmap, bits);
}

/// Run `f` on every other vCPU and then, as safe work, on `src`. The caller must leave the
/// execution loop for the flush to complete before it continues.
fn flush_all_cpus_synced(src: &Cpu<'_>, f: impl Fn(&mut Cpu<'_>) + Clone + Send + 'static) {
    for cpu in src.core.jit.cpu_list() {
        if !Arc::ptr_eq(&cpu, &src.core.shared) {
            let g = f.clone();
            cpu.async_run_on_cpu(move |c| g(c));
        }
    }
    src.core.shared.async_safe_run_on_cpu(move |c| f(c));
}

/// `tlb_flush_by_mmuidx_all_cpus_synced()`.
pub fn tlb_flush_by_mmuidx_all_cpus_synced(src: &mut Cpu<'_>, idxmap: u32) {
    flush_all_cpus_synced(src, move |c| tlb_flush_by_mmuidx_async_work(c, idxmap));
}

/// `tlb_flush_all_cpus_synced()`.
pub fn tlb_flush_all_cpus_synced(src: &mut Cpu<'_>) {
    let all = all_mmuidx_bits(&src.core.jit);
    tlb_flush_by_mmuidx_all_cpus_synced(src, all);
}

/// `tlb_flush_page_by_mmuidx_all_cpus_synced()`.
pub fn tlb_flush_page_by_mmuidx_all_cpus_synced(src: &mut Cpu<'_>, addr: u64, idxmap: u32) {
    let addr = addr & src.core.jit.page_mask();
    flush_all_cpus_synced(src, move |c| tlb_flush_page_by_mmuidx_async_0(c, addr, idxmap));
}

/// `tlb_flush_page_all_cpus_synced()`.
pub fn tlb_flush_page_all_cpus_synced(src: &mut Cpu<'_>, addr: u64) {
    let all = all_mmuidx_bits(&src.core.jit);
    tlb_flush_page_by_mmuidx_all_cpus_synced(src, addr, all);
}

/// `tlb_flush_range_by_mmuidx_all_cpus_synced()`.
pub fn tlb_flush_range_by_mmuidx_all_cpus_synced(
    src: &mut Cpu<'_>,
    addr: u64,
    len: u64,
    idxmap: u32,
    bits: u32,
) {
    let jit = src.jit();
    // If all bits are significant, and len is small, this devolves to tlb_flush_page.
    if bits >= jit.config.target_long_bits && len <= jit.page_size() {
        tlb_flush_page_by_mmuidx_all_cpus_synced(src, addr, idxmap);
        return;
    }
    // If no page bits are significant, this devolves to tlb_flush.
    if bits < jit.config.page_bits {
        tlb_flush_by_mmuidx_all_cpus_synced(src, idxmap);
        return;
    }
    let addr = addr & jit.page_mask();
    flush_all_cpus_synced(src, move |c| {
        tlb_flush_range_by_mmuidx_async_0(c, addr, len, idxmap, bits);
    });
}

/// `tlb_flush_page_bits_by_mmuidx_all_cpus_synced()`.
pub fn tlb_flush_page_bits_by_mmuidx_all_cpus_synced(
    src: &mut Cpu<'_>,
    addr: u64,
    idxmap: u32,
    bits: u32,
) {
    let size = src.core.jit.page_size();
    tlb_flush_range_by_mmuidx_all_cpus_synced(src, addr, size, idxmap, bits);
}

/// `tlb_reset_dirty_range_locked()`.
fn reset_dirty_range_locked(
    jit: &Jit,
    e: &mut TlbEntry,
    full: &TlbEntryFull,
    start: u64,
    len: u64,
) {
    let addr = e.addr[MmuAccessType::DataStore as usize];
    let flags = (addr | u64::from(full.slow_flags[MmuAccessType::DataStore as usize]))
        & (tlb::INVALID_MASK | u64::from(tlb::MMIO | tlb::DISCARD_WRITE) | tlb::NOTDIRTY);
    if flags == 0 {
        let host = (addr & jit.page_mask()).wrapping_add(e.addend);
        if host.wrapping_sub(start) < len {
            e.addr[MmuAccessType::DataStore as usize] = addr | tlb::NOTDIRTY;
        }
    }
}

/// `tlb_reset_dirty_range_all()`: make every vCPU's writes to `ram_addr` range
/// `start..start + len` take the notdirty path.
pub(crate) fn tlb_reset_dirty_range_all(jit: &Jit, start: u64, len: u64) {
    for cpu in jit.cpu_list() {
        let mut t = lock(&cpu.tlb);
        let t = &mut *t;
        for mmu_idx in 0..t.d.len() {
            for (e, full) in t.table[mmu_idx].iter_mut().zip(&t.d[mmu_idx].fulltlb) {
                reset_dirty_range_locked(jit, e, full, start, len);
            }
            let d = &mut t.d[mmu_idx];
            for (e, full) in d.vtable.iter_mut().zip(&d.vfulltlb) {
                reset_dirty_range_locked(jit, e, full, start, len);
            }
        }
    }
}

/// `tlb_set_dirty()`.
fn tlb_set_dirty(cpu: &Cpu<'_>, addr: u64) {
    let jit = &cpu.core.jit;
    let addr = addr & jit.page_mask();
    let mut t = lock(&cpu.core.shared.tlb);
    let st = MmuAccessType::DataStore as usize;
    for mmu_idx in 0..t.d.len() {
        let i = t.index(jit.config.page_bits, mmu_idx, addr);
        let e = &mut t.table[mmu_idx][i];
        if e.addr[st] == addr | tlb::NOTDIRTY {
            e.addr[st] = addr;
        }
    }
    for d in &mut t.d {
        for e in &mut d.vtable {
            if e.addr[st] == addr | tlb::NOTDIRTY {
                e.addr[st] = addr;
            }
        }
    }
}

fn page_dirty(block: &RamBlock, off: u64, size: u64, client: DirtyClient) -> bool {
    !block.is_dirty_logging(client) || block.get_dirty(off, size, client)
}

/// `cpu_physical_memory_is_clean()`: some client still needs to see writes to the page.
fn ram_is_clean(jit: &Jit, block: &RamBlock, off: u64, ram_addr: u64) -> bool {
    let size = jit.page_size();
    let vga = page_dirty(block, off & jit.page_mask(), size, DirtyClient::Vga);
    let code = !jit.page_has_code(ram_addr);
    let migration = page_dirty(block, off & jit.page_mask(), size, DirtyClient::Migration);
    !(vga && code && migration)
}

/// `tlb_add_large_page()`.
fn tlb_add_large_page(t: &mut CpuTlb, mmu_idx: usize, addr: u64, size: u64) {
    let d = &mut t.d[mmu_idx];
    let mut lp_addr = d.large_page_addr;
    let mut lp_mask = !(size - 1);
    if lp_addr == u64::MAX {
        // No previous large page.
        lp_addr = addr;
    } else {
        // Extend the existing region to include the new page.
        lp_mask &= d.large_page_mask;
        while (lp_addr ^ addr) & lp_mask != 0 {
            lp_mask <<= 1;
        }
    }
    d.large_page_addr = lp_addr & lp_mask;
    d.large_page_mask = lp_mask;
}

/// `tlb_set_page_full()`: install the translation of `addr` for `mmu_idx`. Called from the
/// target's `tlb_fill`.
pub fn tlb_set_page_full(cpu: &mut Cpu<'_>, mmu_idx: usize, addr: u64, full: &TlbEntryFull) {
    let jit = cpu.jit();
    let page_bits = jit.config.page_bits;
    let page_size = jit.page_size();
    let lg = u32::from(full.lg_page_size);
    let large = if lg > page_bits { Some(1u64 << lg) } else { None };
    let addr_page = addr & jit.page_mask();
    let paddr_page = full.phys_addr & jit.page_mask();
    let prot = full.prot;

    // address_space_translate_for_iotlb(): only a range covering the whole page is RAM.
    let fv = cpu.core.as_.flatview();
    let mut section = TlbSection::Io;
    let mut readonly = false;
    let mut page_ram_addr = 0;
    if let Some(r) = fv.lookup(paddr_page) {
        let covers = u128::from(paddr_page) + u128::from(page_size) <= r.end();
        if let (true, Some(block)) = (covers, r.ram_block()) {
            let off = r.offset_in_region() + (paddr_page - r.addr());
            let ram_base = jit.ram_addr_base(block);
            match r.region_type() {
                RegionType::Ram => {
                    page_ram_addr = ram_base + off;
                    readonly = r.readonly();
                    section = TlbSection::Ram { block: block.clone(), ram_base };
                }
                RegionType::RomDevice if r.romd_mode() => {
                    page_ram_addr = ram_base + off;
                    section = TlbSection::Romd { block: block.clone(), ram_base };
                }
                _ => {}
            }
        }
    }
    let is_ram = matches!(section, TlbSection::Ram { .. });
    let is_romd = matches!(section, TlbSection::Romd { .. });

    let mut read_flags = full.tlb_fill_flags;
    if lg < page_bits {
        // Repeat the MMU check and TLB fill on every access.
        read_flags |= tlb::INVALID_MASK;
    }
    // RAM and ROMD both have associated host memory; IO does not.
    let addend = if is_ram || is_romd { page_ram_addr } else { 0 };
    let mut write_flags = read_flags;
    let iotlb;
    let mut check_clean = false;
    if is_ram {
        iotlb = page_ram_addr;
        if prot & page::WRITE != 0 {
            if readonly {
                write_flags |= u64::from(tlb::DISCARD_WRITE);
            } else {
                check_clean = true;
            }
        }
    } else {
        iotlb = paddr_page;
        // Writes to romd devices must go through MMIO to enable write. Reads to romd devices
        // go through the ram_ptr found above, but of course reads to I/O must go through MMIO.
        write_flags |= u64::from(tlb::MMIO);
        if !is_romd {
            read_flags = write_flags;
        }
    }

    let wp_flags = cpu.watchpoint_address_matches(addr_page, page_size);

    let mut t = lock(&cpu.core.shared.tlb);
    if let Some(sz) = large {
        tlb_add_large_page(&mut t, mmu_idx, addr, sz);
    }
    if check_clean {
        // Tested under the TLB lock, so that tlb_protect_code() either is seen here or sees
        // the new entry.
        if let TlbSection::Ram { block, ram_base } = &section {
            if ram_is_clean(&jit, block, page_ram_addr - ram_base, page_ram_addr) {
                write_flags |= tlb::NOTDIRTY;
            }
        }
    }

    // Note that the tlb is no longer clean.
    t.dirty |= 1 << mmu_idx;
    // Make sure there's no cached translation for the new page.
    t.flush_vtlb_page_mask_locked(&jit, mmu_idx, addr_page, u64::MAX);

    let index = t.index(page_bits, mmu_idx, addr_page);
    let te = t.table[mmu_idx][index];
    // Only evict the old entry to the victim tlb if it's for a different page; otherwise just
    // overwrite the stale data.
    if !tlb_hit_page_anyprot(&jit, &te, addr_page) && !te.is_empty() {
        let d = &mut t.d[mmu_idx];
        let vidx = d.vindex % CPU_VTLB_SIZE;
        d.vindex = d.vindex.wrapping_add(1);
        d.vtable[vidx] = te;
        d.vfulltlb[vidx] = d.fulltlb[index].clone();
        t.n_used_dec(mmu_idx);
    }

    let mut nf = full.clone();
    nf.xlat_offset = iotlb.wrapping_sub(addr_page);
    nf.phys_addr = paddr_page;
    nf.section = section;
    let mut tn = TlbEntry { addr: [u64::MAX; 3], addend: addend.wrapping_sub(addr_page) };

    let set_compare =
        |nf: &mut TlbEntryFull, tn: &mut TlbEntry, flags: u64, at: MmuAccessType, enable: bool| {
            let (address, slow) = if enable {
                let mut a = addr_page | (flags & tlb::FLAGS_MASK);
                let slow = (flags & u64::from(tlb::SLOW_FLAGS_MASK)) as u32;
                if slow != 0 {
                    a |= tlb::FORCE_SLOW;
                }
                (a, slow)
            } else {
                (u64::MAX, 0)
            };
            tn.addr[at as usize] = address;
            nf.slow_flags[at as usize] = slow;
        };

    set_compare(&mut nf, &mut tn, read_flags, MmuAccessType::InstFetch, prot & page::EXEC != 0);
    if wp_flags & bp::MEM_READ != 0 {
        read_flags |= u64::from(tlb::WATCHPOINT);
    }
    set_compare(&mut nf, &mut tn, read_flags, MmuAccessType::DataLoad, prot & page::READ != 0);
    if prot & page::WRITE_INV != 0 {
        write_flags |= tlb::INVALID_MASK;
    }
    if wp_flags & bp::MEM_WRITE != 0 {
        write_flags |= u64::from(tlb::WATCHPOINT);
    }
    set_compare(&mut nf, &mut tn, write_flags, MmuAccessType::DataStore, prot & page::WRITE != 0);

    t.table[mmu_idx][index] = tn;
    t.d[mmu_idx].fulltlb[index] = nf;
    t.d[mmu_idx].n_used_entries += 1;
}

/// `tlb_set_page_with_attrs()`.
#[allow(clippy::too_many_arguments)]
pub fn tlb_set_page_with_attrs(
    cpu: &mut Cpu<'_>,
    addr: u64,
    paddr: u64,
    attrs: MemTxAttrs,
    prot: u32,
    mmu_idx: usize,
    size: u64,
) {
    let full = TlbEntryFull {
        phys_addr: paddr,
        attrs,
        prot,
        lg_page_size: size.trailing_zeros() as u8,
        ..TlbEntryFull::default()
    };
    assert!(size.is_power_of_two());
    tlb_set_page_full(cpu, mmu_idx, addr, &full);
}

/// `tlb_set_page()`.
pub fn tlb_set_page(
    cpu: &mut Cpu<'_>,
    addr: u64,
    paddr: u64,
    prot: u32,
    mmu_idx: usize,
    size: u64,
) {
    tlb_set_page_with_attrs(cpu, addr, paddr, MemTxAttrs::default(), prot, mmu_idx, size);
}

/// `victim_tlb_hit()`.
fn victim_tlb_hit(
    t: &mut CpuTlb,
    mmu_idx: usize,
    index: usize,
    at: MmuAccessType,
    page: u64,
) -> bool {
    for vidx in 0..CPU_VTLB_SIZE {
        let cmp = t.d[mmu_idx].vtable[vidx].addr[at as usize];
        if cmp == page {
            // Found entry in victim tlb, swap tlb and iotlb.
            let d = &mut t.d[mmu_idx];
            std::mem::swap(&mut t.table[mmu_idx][index], &mut d.vtable[vidx]);
            std::mem::swap(&mut d.fulltlb[index], &mut d.vfulltlb[vidx]);
            return true;
        }
    }
    false
}

/// One page of an access, `MMULookupPageData`.
struct PageData {
    addr: u64,
    size: usize,
    full: TlbEntryFull,
    flags: u64,
    haddr: u64,
}

/// Look up `addr` in the TLB, without filling. Returns the comparator, addend and full entry.
fn tlb_lookup_nofill(
    cpu: &Cpu<'_>,
    addr: u64,
    at: MmuAccessType,
    mmu_idx: usize,
) -> Option<(u64, u64, TlbEntryFull)> {
    let jit = &cpu.core.jit;
    let mut t = lock(&cpu.core.shared.tlb);
    let index = t.index(jit.config.page_bits, mmu_idx, addr);
    let e = t.table[mmu_idx][index];
    let tlb_addr = e.addr[at as usize];
    if tlb_hit(jit, tlb_addr, addr)
        || victim_tlb_hit(&mut t, mmu_idx, index, at, addr & jit.page_mask())
    {
        let e = t.table[mmu_idx][index];
        Some((e.addr[at as usize], e.addend, t.d[mmu_idx].fulltlb[index].clone()))
    } else {
        None
    }
}

fn tlb_read_entry(
    cpu: &Cpu<'_>,
    addr: u64,
    at: MmuAccessType,
    mmu_idx: usize,
) -> (u64, u64, TlbEntryFull) {
    let jit = &cpu.core.jit;
    let t = lock(&cpu.core.shared.tlb);
    let index = t.index(jit.config.page_bits, mmu_idx, addr);
    let e = t.table[mmu_idx][index];
    (e.addr[at as usize], e.addend, t.d[mmu_idx].fulltlb[index].clone())
}

/// `mmu_lookup1()`.
fn mmu_lookup1(
    cpu: &mut Cpu<'_>,
    addr: u64,
    size: usize,
    mmu_idx: usize,
    at: MmuAccessType,
    ra: Ra,
) -> Result<PageData, CpuLoopExit> {
    let (tlb_addr, addend, full) = match tlb_lookup_nofill(cpu, addr, at, mmu_idx) {
        Some(x) => x,
        None => {
            let ops = cpu.ops();
            let ok = ops.tlb_fill(cpu, addr, size, at, mmu_idx, false, ra)?;
            assert!(ok, "tlb_fill must not fail without probe");
            let (a, addend, full) = tlb_read_entry(cpu, addr, at, mmu_idx);
            (a & !tlb::INVALID_MASK, addend, full)
        }
    };
    let flags =
        (tlb_addr & (tlb::FLAGS_MASK & !tlb::FORCE_SLOW)) | u64::from(full.slow_flags[at as usize]);
    Ok(PageData { addr, size, full, flags, haddr: addr.wrapping_add(addend) })
}

/// `notdirty_write()`.
fn notdirty_write(
    cpu: &mut Cpu<'_>,
    vaddr: u64,
    size: usize,
    full: &TlbEntryFull,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let jit = cpu.jit();
    let ram_addr = vaddr.wrapping_add(full.xlat_offset);
    if jit.page_has_code(ram_addr) {
        tb_invalidate_phys_range_fast(cpu, ram_addr, size as u64, ra)?;
    }
    let TlbSection::Ram { block, ram_base } = &full.section else { return Ok(()) };
    let off = ram_addr - ram_base;
    // Set both VGA and migration bits for simplicity and to remove the notdirty callback
    // faster.
    block.set_dirty(
        off,
        size as u64,
        DirtyMask::NONE.with(DirtyClient::Vga).with(DirtyClient::Migration),
    );
    // We remove the notdirty callback only if the code has been flushed.
    if !ram_is_clean(&jit, block, off, ram_addr) {
        tlb_set_dirty(cpu, vaddr);
    }
    Ok(())
}

/// `mmu_watch_or_dirty()`.
fn mmu_watch_or_dirty(
    cpu: &mut Cpu<'_>,
    p: &mut PageData,
    at: MmuAccessType,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    // On watchpoint hit, this will longjmp out.
    if p.flags & u64::from(tlb::WATCHPOINT) != 0 {
        let wp = if at == MmuAccessType::DataStore { bp::MEM_WRITE } else { bp::MEM_READ };
        cpu.check_watchpoint(p.addr, p.size as u64, p.full.attrs, wp, ra)?;
        p.flags &= !u64::from(tlb::WATCHPOINT);
    }
    // Note that notdirty is only set for writes.
    if p.flags & tlb::NOTDIRTY != 0 {
        notdirty_write(cpu, p.addr, p.size, &p.full, ra)?;
        p.flags &= !tlb::NOTDIRTY;
    }
    Ok(())
}

/// `mmu_lookup()`: the one or two pages of an access.
fn mmu_lookup(
    cpu: &mut Cpu<'_>,
    addr: u64,
    size: usize,
    mmu_idx: usize,
    at: MmuAccessType,
    ra: Ra,
) -> Result<(PageData, Option<PageData>), CpuLoopExit> {
    let jit = cpu.jit();
    let page1_addr = addr.wrapping_add(size as u64 - 1) & jit.page_mask();
    let crosspage = (addr ^ page1_addr) & jit.page_mask() != 0;
    if !crosspage {
        let mut p0 = mmu_lookup1(cpu, addr, size, mmu_idx, at, ra)?;
        if p0.flags & (u64::from(tlb::WATCHPOINT) | tlb::NOTDIRTY) != 0 {
            mmu_watch_or_dirty(cpu, &mut p0, at, ra)?;
        }
        return Ok((p0, None));
    }
    let size0 = page1_addr.wrapping_sub(addr) as usize;
    let ops = cpu.ops();
    let page1_addr = ops.pointer_wrap(cpu, mmu_idx, page1_addr, addr);
    // Lookup both pages, recognizing exceptions from either.
    let mut p0 = mmu_lookup1(cpu, addr, size0, mmu_idx, at, ra)?;
    let mut p1 = mmu_lookup1(cpu, page1_addr, size - size0, mmu_idx, at, ra)?;
    let flags = p0.flags | p1.flags;
    if flags & (u64::from(tlb::WATCHPOINT) | tlb::NOTDIRTY) != 0 {
        mmu_watch_or_dirty(cpu, &mut p0, at, ra)?;
        mmu_watch_or_dirty(cpu, &mut p1, at, ra)?;
    }
    // target/sparc is the only user of TLB_BSWAP, and all its accesses are aligned.
    assert!(flags & u64::from(tlb::BSWAP) == 0, "TLB_BSWAP on an access crossing a page");
    Ok((p0, Some(p1)))
}

/// `io_failed()`.
#[allow(clippy::too_many_arguments)]
fn io_failed(
    cpu: &mut Cpu<'_>,
    full: &TlbEntryFull,
    addr: u64,
    size: usize,
    at: MmuAccessType,
    mmu_idx: usize,
    response: MemTxResult,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    if !cpu.core.ignore_memory_transaction_failures {
        let jit = cpu.jit();
        let physaddr = full.phys_addr | (addr & !jit.page_mask());
        let ops = cpu.ops();
        ops.do_transaction_failed(
            cpu, physaddr, addr, size, at, mmu_idx, full.attrs, response, ra,
        )?;
    }
    Ok(())
}

/// `io_prepare()`: an IO access in a block must be its last instruction.
fn io_prepare(cpu: &mut Cpu<'_>, ra: Ra) -> Result<(), CpuLoopExit> {
    if !cpu.can_do_io() {
        return Err(crate::translate::cpu_io_recompile(cpu, ra));
    }
    Ok(())
}

fn ram_of(full: &TlbEntryFull, haddr: u64) -> Option<(&Arc<RamBlock>, u64)> {
    match &full.section {
        TlbSection::Ram { block, ram_base } | TlbSection::Romd { block, ram_base } => {
            Some((block, haddr - ram_base))
        }
        TlbSection::Io => None,
    }
}

/// The access sizes for MMIO: aligned pieces of up to 8 bytes.
fn io_piece(addr: u64, remaining: usize) -> usize {
    1usize << (remaining as u64 | addr | 8).trailing_zeros()
}

fn do_ld_piece(
    cpu: &mut Cpu<'_>,
    p: &PageData,
    buf: &mut [u8],
    mmu_idx: usize,
    at: MmuAccessType,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    if p.flags & u64::from(tlb::MMIO) != 0 {
        io_prepare(cpu, ra)?;
        let jit = cpu.jit();
        let as_ = cpu.core.as_.clone();
        let mut phys = p.full.phys_addr | (p.addr & !jit.page_mask());
        let mut addr = p.addr;
        let mut done = 0;
        while done < buf.len() {
            let n = io_piece(addr, buf.len() - done);
            let (val, r) = as_.load(phys, n as u32, Endian::Little, p.full.attrs);
            if !r.is_ok() {
                io_failed(cpu, &p.full, addr, n, at, mmu_idx, r, ra)?;
            }
            buf[done..done + n].copy_from_slice(&val.to_le_bytes()[..n]);
            done += n;
            addr = addr.wrapping_add(n as u64);
            phys = phys.wrapping_add(n as u64);
        }
        return Ok(());
    }
    let (block, off) = ram_of(&p.full, p.haddr).expect("RAM TLB entry without a block");
    block.read(off, buf).expect("RAM TLB entry outside its block");
    Ok(())
}

fn do_st_piece(
    cpu: &mut Cpu<'_>,
    p: &PageData,
    data: &[u8],
    mmu_idx: usize,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    if p.flags & u64::from(tlb::MMIO) != 0 {
        io_prepare(cpu, ra)?;
        let jit = cpu.jit();
        let as_ = cpu.core.as_.clone();
        let mut phys = p.full.phys_addr | (p.addr & !jit.page_mask());
        let mut addr = p.addr;
        let mut done = 0;
        while done < data.len() {
            let n = io_piece(addr, data.len() - done);
            let mut b = [0u8; 8];
            b[..n].copy_from_slice(&data[done..done + n]);
            let r = as_.store(phys, n as u32, u64::from_le_bytes(b), Endian::Little, p.full.attrs);
            if !r.is_ok() {
                io_failed(cpu, &p.full, addr, n, MmuAccessType::DataStore, mmu_idx, r, ra)?;
            }
            done += n;
            addr = addr.wrapping_add(n as u64);
            phys = phys.wrapping_add(n as u64);
        }
        return Ok(());
    }
    if p.flags & u64::from(tlb::DISCARD_WRITE) != 0 {
        return Ok(());
    }
    let (block, off) = ram_of(&p.full, p.haddr).expect("RAM TLB entry without a block");
    block.write(off, data).expect("RAM TLB entry outside its block");
    Ok(())
}

/// Load `buf.len()` bytes at `addr` through the softmmu, in memory order. This is the slow
/// path of every `qemu_ld`.
pub fn do_ld_bytes(
    cpu: &mut Cpu<'_>,
    addr: u64,
    buf: &mut [u8],
    oi: MemOpIdx,
    at: MmuAccessType,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let mmu_idx = oi.mmu_idx() as usize;
    let (p0, p1) = mmu_lookup(cpu, addr, buf.len(), mmu_idx, at, ra)?;
    let n0 = p0.size;
    do_ld_piece(cpu, &p0, &mut buf[..n0], mmu_idx, at, ra)?;
    match p1 {
        Some(p1) => do_ld_piece(cpu, &p1, &mut buf[n0..], mmu_idx, at, ra)?,
        None => {
            if p0.flags & u64::from(tlb::BSWAP) != 0 {
                buf.reverse();
            }
        }
    }
    Ok(())
}

/// Store `data` at `addr` through the softmmu, in memory order. This is the slow path of every
/// `qemu_st`.
pub fn do_st_bytes(
    cpu: &mut Cpu<'_>,
    addr: u64,
    data: &[u8],
    oi: MemOpIdx,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let mmu_idx = oi.mmu_idx() as usize;
    let (p0, p1) = mmu_lookup(cpu, addr, data.len(), mmu_idx, MmuAccessType::DataStore, ra)?;
    let n0 = p0.size;
    match p1 {
        Some(p1) => {
            do_st_piece(cpu, &p0, &data[..n0], mmu_idx, ra)?;
            do_st_piece(cpu, &p1, &data[n0..], mmu_idx, ra)?;
        }
        None => {
            if p0.flags & u64::from(tlb::BSWAP) != 0 {
                let mut swapped = data.to_vec();
                swapped.reverse();
                do_st_piece(cpu, &p0, &swapped, mmu_idx, ra)?;
            } else {
                do_st_piece(cpu, &p0, data, mmu_idx, ra)?;
            }
        }
    }
    Ok(())
}

/// `cpu_ld*_mmu()`: load the value `oi` describes, for target helpers.
pub fn cpu_ld_mmu(cpu: &mut Cpu<'_>, addr: u64, oi: MemOpIdx, ra: Ra) -> Result<u64, CpuLoopExit> {
    let mop = oi.memop();
    let n = mop.size_bytes() as usize;
    assert!(n <= 8);
    let mut b = [0u8; 8];
    do_ld_bytes(cpu, addr, &mut b[..n], oi, MmuAccessType::DataLoad, ra)?;
    if mop.is_bswap() {
        b[..n].reverse();
    }
    let v = u64::from_le_bytes(b);
    if mop.is_signed() && n < 8 {
        let sh = 64 - 8 * n as u32;
        return Ok((((v << sh) as i64) >> sh) as u64);
    }
    Ok(v)
}

/// `cpu_st*_mmu()`: store the value `oi` describes, for target helpers.
pub fn cpu_st_mmu(
    cpu: &mut Cpu<'_>,
    addr: u64,
    val: u64,
    oi: MemOpIdx,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let mop = oi.memop();
    let n = mop.size_bytes() as usize;
    assert!(n <= 8);
    let mut b = val.to_le_bytes();
    if mop.is_bswap() {
        b[..n].reverse();
    }
    do_st_bytes(cpu, addr, &b[..n], oi, ra)
}

/// Fetch code bytes through the softmmu, `cpu_ld*_code_mmu()`, in memory order.
pub fn cpu_ld_code(
    cpu: &mut Cpu<'_>,
    addr: u64,
    buf: &mut [u8],
    mmu_idx: usize,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let oi = MemOpIdx::new(MemOp::UB, mmu_idx as u32);
    do_ld_bytes(cpu, addr, buf, oi, MmuAccessType::InstFetch, ra)
}

/// `probe_access_internal()`: the flags, the host (`ram_addr`) address if the page is RAM,
/// and the full entry. With `nonfault`, a failed fill gives [`tlb::INVALID_MASK`].
#[allow(clippy::too_many_arguments)]
fn probe_access_internal(
    cpu: &mut Cpu<'_>,
    addr: u64,
    fault_size: usize,
    at: MmuAccessType,
    mmu_idx: usize,
    nonfault: bool,
    ra: Ra,
) -> Result<(u64, Option<u64>, Option<TlbEntryFull>), CpuLoopExit> {
    let mut flags = tlb::FLAGS_MASK & !tlb::FORCE_SLOW;
    let (tlb_addr, addend, full) = match tlb_lookup_nofill(cpu, addr, at, mmu_idx) {
        Some(x) => x,
        None => {
            let ops = cpu.ops();
            if !ops.tlb_fill(cpu, addr, fault_size, at, mmu_idx, nonfault, ra)? {
                // Non-faulting page table read failed.
                return Ok((tlb::INVALID_MASK, None, None));
            }
            // With PAGE_WRITE_INV the entry is marked invalid at once to force the next access
            // through tlb_fill, but we know that this entry is valid.
            flags &= !tlb::INVALID_MASK;
            tlb_read_entry(cpu, addr, at, mmu_idx)
        }
    };
    flags &= tlb_addr;
    flags |= u64::from(full.slow_flags[at as usize]);
    // Fold all "mmio-like" bits into TLB_MMIO. This is not RAM.
    let ram_like = u64::from(tlb::WATCHPOINT | tlb::CHECK_ALIGNED) | tlb::NOTDIRTY;
    if flags & !ram_like != 0 {
        return Ok((u64::from(tlb::MMIO), None, Some(full)));
    }
    // Everything else is RAM.
    Ok((flags, Some(addr.wrapping_add(addend)), Some(full)))
}

/// `probe_access()`: make sure `size` bytes at `addr` can be accessed, raising the guest fault
/// if not, and handle watchpoints and notdirty. Returns the `ram_addr` if the page is RAM.
pub fn probe_access(
    cpu: &mut Cpu<'_>,
    addr: u64,
    size: usize,
    at: MmuAccessType,
    mmu_idx: usize,
    ra: Ra,
) -> Result<Option<u64>, CpuLoopExit> {
    let (flags, host, full) = probe_access_internal(cpu, addr, size, at, mmu_idx, false, ra)?;
    // Per the interface, size == 0 merely faults the access.
    if size == 0 {
        return Ok(None);
    }
    if flags & (tlb::NOTDIRTY | u64::from(tlb::WATCHPOINT)) != 0 {
        let full = full.expect("a filled entry");
        if flags & u64::from(tlb::WATCHPOINT) != 0 {
            let wp = if at == MmuAccessType::DataStore { bp::MEM_WRITE } else { bp::MEM_READ };
            cpu.check_watchpoint(addr, size as u64, full.attrs, wp, ra)?;
        }
        if flags & tlb::NOTDIRTY != 0 {
            notdirty_write(cpu, addr, size, &full, ra)?;
        }
    }
    Ok(host)
}

/// `get_page_addr_code_hostp()`: the `ram_addr` of the code at `addr`, or `u64::MAX` if it is
/// not in RAM. A fault in the fill is raised.
pub fn get_page_addr_code(cpu: &mut Cpu<'_>, addr: u64) -> Result<u64, CpuLoopExit> {
    let ops = cpu.ops();
    let mmu_idx = ops.mmu_index(cpu, true);
    let (_, host, full) =
        probe_access_internal(cpu, addr, 1, MmuAccessType::InstFetch, mmu_idx, false, Ra::None)?;
    let Some(host) = host else { return Ok(u64::MAX) };
    let jit = cpu.jit();
    if full.is_some_and(|f| u32::from(f.lg_page_size) < jit.config.page_bits) {
        return Ok(u64::MAX);
    }
    Ok(host)
}

/// The number of full and partial TLB flushes `cpu` did, `tlb_flush_count()`.
pub fn tlb_flush_counts(cpu: &Cpu<'_>) -> (u64, u64) {
    let t = lock(&cpu.core.shared.tlb);
    (t.full_flush_count, t.part_flush_count)
}

/// The number of entries in the fast table of `mmu_idx`.
pub fn tlb_n_entries(cpu: &Cpu<'_>, mmu_idx: usize) -> usize {
    lock(&cpu.core.shared.tlb).table[mmu_idx].len()
}

impl Cpu<'_> {
    /// [`tlb_set_page_full`] as a method, for `tlb_fill` hooks.
    pub fn tlb_set_page_full(&mut self, mmu_idx: usize, addr: u64, full: &TlbEntryFull) {
        tlb_set_page_full(self, mmu_idx, addr, full);
    }

    /// [`tlb_set_page`] as a method, for `tlb_fill` hooks.
    pub fn tlb_set_page(&mut self, addr: u64, paddr: u64, prot: u32, mmu_idx: usize, size: u64) {
        tlb_set_page(self, addr, paddr, prot, mmu_idx, size);
    }
}
