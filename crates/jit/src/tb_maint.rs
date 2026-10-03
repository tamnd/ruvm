// SPDX-License-Identifier: GPL-2.0-or-later

//! Translation block maintenance, from `tb-maint.c` and the lookup half of `cpu-exec.c`: the
//! hash table and jump cache lookups, linking a new block into its pages, chaining with
//! `goto_tb`, invalidation by physical range including self modifying code, and `tb_flush`.
//!
//! Lock order: the page lists, then the hash table, then a vCPU's TLB, then the code page set.
//! Separately, a destination's `jmp_lock`, then a source's `jmp_dest`. A vCPU's jump cache lock
//! is only ever held on its own.
//!
//! Differences from QEMU:
//!
//! - The jump cache is a mutex protected array per vCPU instead of an array of atomics.
//! - Pages keep their (possibly empty) block list once they had code, as QEMU's page descriptors
//!   stay allocated, so a write to a page that lost its code still turns the slow path off.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::cpu::{Cpu, CpuLoopExit, CpuShared, Ra};
use crate::cputlb;
use crate::jit::Jit;
use crate::tb::{Tb, TbCpuState, TbKey, lock};
use crate::{TB_JMP_CACHE_SIZE, cf};

/// One slot of the jump cache, `CPUJumpCache`'s array entry.
#[derive(Clone, Debug, Default)]
pub(crate) struct JcEntry {
    pub(crate) tb: Option<Arc<Tb>>,
    pub(crate) pc: u64,
}

const TB_JMP_PAGE_BITS: u32 = crate::TB_JMP_CACHE_BITS / 2;
const TB_JMP_PAGE_SIZE: u64 = 1 << TB_JMP_PAGE_BITS;
const TB_JMP_ADDR_MASK: u64 = TB_JMP_PAGE_SIZE - 1;
const TB_JMP_PAGE_MASK: u64 = (TB_JMP_CACHE_SIZE as u64 - 1) & !TB_JMP_ADDR_MASK;

/// `tb_jmp_cache_hash_page()`.
pub(crate) fn tb_jmp_cache_hash_page(jit: &Jit, pc: u64) -> usize {
    let sh = jit.config.page_bits - TB_JMP_PAGE_BITS;
    let tmp = pc ^ (pc >> sh);
    ((tmp >> sh) & TB_JMP_PAGE_MASK) as usize
}

/// `tb_jmp_cache_hash_func()`.
pub(crate) fn tb_jmp_cache_hash_func(jit: &Jit, pc: u64) -> usize {
    let sh = jit.config.page_bits - TB_JMP_PAGE_BITS;
    let tmp = pc ^ (pc >> sh);
    (((tmp >> sh) & TB_JMP_PAGE_MASK) | (tmp & TB_JMP_ADDR_MASK)) as usize
}

/// `tcg_flush_jmp_cache()`.
pub(crate) fn tcg_flush_jmp_cache(cpu: &CpuShared) {
    for e in lock(&cpu.jmp_cache).iter_mut() {
        e.tb = None;
    }
}

/// `tb_jmp_cache_clear_page()`.
pub(crate) fn tb_jmp_cache_clear_page(jit: &Jit, cpu: &CpuShared, page_addr: u64) {
    let i0 = tb_jmp_cache_hash_page(jit, page_addr);
    let mut jc = lock(&cpu.jmp_cache);
    for e in &mut jc[i0..i0 + TB_JMP_PAGE_SIZE as usize] {
        e.tb = None;
    }
}

fn key_for(s: &TbCpuState, phys_pc: u64) -> TbKey {
    TbKey {
        phys_pc,
        pc: if s.cflags & cf::PCREL != 0 { 0 } else { s.pc },
        flags: s.flags,
        cs_base: s.cs_base,
        cflags: s.cflags,
    }
}

/// `tb_htable_lookup()`.
pub(crate) fn tb_htable_lookup(
    cpu: &mut Cpu<'_>,
    s: TbCpuState,
) -> Result<Option<Arc<Tb>>, CpuLoopExit> {
    let phys_pc = cputlb::get_page_addr_code(cpu, s.pc)?;
    if phys_pc == u64::MAX {
        return Ok(None);
    }
    let key = key_for(&s, phys_pc);
    let tb = cpu.core.jit.htable.read().unwrap_or_else(|e| e.into_inner()).get(&key).cloned();
    let Some(tb) = tb else { return Ok(None) };
    // tb_lookup_cmp(): check the second page too.
    if tb.page_addr[1] == u64::MAX {
        return Ok(Some(tb));
    }
    let jit = cpu.jit();
    let virt_page1 = (s.pc.wrapping_add(jit.page_size() - 1)) & jit.page_mask();
    let phys_page1 = cputlb::get_page_addr_code(cpu, virt_page1)?;
    Ok(if tb.page_addr[1] == phys_page1 { Some(tb) } else { None })
}

/// `tb_lookup()`: the jump cache, then the hash table.
pub(crate) fn tb_lookup(cpu: &mut Cpu<'_>, s: TbCpuState) -> Result<Option<Arc<Tb>>, CpuLoopExit> {
    // We should never be trying to look up an INVALID tb.
    debug_assert!(s.cflags & cf::INVALID == 0);
    let jit = cpu.jit();
    let shared = cpu.shared();
    let hash = tb_jmp_cache_hash_func(&jit, s.pc);
    {
        let jc = lock(&shared.jmp_cache);
        let e = &jc[hash];
        if let Some(tb) = &e.tb {
            if e.pc == s.pc
                && tb.cs_base == s.cs_base
                && tb.flags == s.flags
                && tb.cflags() == s.cflags
            {
                let tb = tb.clone();
                assert!(tb.cflags() & cf::PCREL != 0 || tb.pc == s.pc);
                return Ok(Some(tb));
            }
        }
    }
    let Some(tb) = tb_htable_lookup(cpu, s)? else { return Ok(None) };
    let mut jc = lock(&shared.jmp_cache);
    jc[hash] = JcEntry { tb: Some(tb.clone()), pc: s.pc };
    drop(jc);
    assert!(tb.cflags() & cf::PCREL != 0 || tb.pc == s.pc);
    Ok(Some(tb))
}

/// Set the jump cache entry for `pc`, as `cpu_exec_loop()` does after `tb_gen_code()`.
pub(crate) fn jc_set(cpu: &Cpu<'_>, pc: u64, tb: &Arc<Tb>) {
    let hash = tb_jmp_cache_hash_func(&cpu.core.jit, pc);
    lock(&cpu.core.shared.jmp_cache)[hash] = JcEntry { tb: Some(tb.clone()), pc };
}

impl Jit {
    fn page_index(&self, addr: u64) -> u64 {
        addr >> self.config.page_bits
    }

    /// `tlb_protect_code()`: writes to the page must go through `notdirty_write()`.
    pub(crate) fn tlb_protect_code(&self, ram_addr: u64) {
        let page = ram_addr & self.page_mask();
        self.code_pages.write().unwrap_or_else(|e| e.into_inner()).insert(self.page_index(page));
        cputlb::tlb_reset_dirty_range_all(self, page, self.page_size());
    }

    /// `tlb_unprotect_code()`.
    pub(crate) fn tlb_unprotect_code(&self, ram_addr: u64) {
        let idx = self.page_index(ram_addr);
        self.code_pages.write().unwrap_or_else(|e| e.into_inner()).remove(&idx);
    }

    /// Whether the page holding `ram_addr` holds translated code, the clear
    /// `DIRTY_MEMORY_CODE` bit.
    pub(crate) fn page_has_code(&self, ram_addr: u64) -> bool {
        let idx = self.page_index(ram_addr);
        self.code_pages.read().unwrap_or_else(|e| e.into_inner()).contains(&idx)
    }

    /// `tb_link_page()`: add `tb` to its pages and the hash table. Returns `tb`, or a block
    /// that another thread linked first for the same code.
    pub(crate) fn tb_link_page(&self, tb: &Arc<Tb>) -> Arc<Tb> {
        debug_assert!(!tb.is_invalid());
        let mut pages = lock(&self.pages);
        // tb_record()
        let p0 = self.page_index(tb.page_addr[0]);
        self.tb_page_add(&mut pages, p0, tb, 0);
        if tb.page_addr[1] != u64::MAX {
            let p1 = self.page_index(tb.page_addr[1]);
            if p1 != p0 {
                self.tb_page_add(&mut pages, p1, tb, 1);
            }
        }
        let mut ht = self.htable.write().unwrap_or_else(|e| e.into_inner());
        let key = tb.key();
        if let Some(existing) = ht.get(&key) {
            let existing = existing.clone();
            drop(ht);
            self.tb_remove(&mut pages, tb);
            return existing;
        }
        ht.insert(key, tb.clone());
        tb.clone()
    }

    fn tb_page_add(
        &self,
        pages: &mut HashMap<u64, Vec<Arc<Tb>>>,
        idx: u64,
        tb: &Arc<Tb>,
        n: usize,
    ) {
        let list = pages.entry(idx).or_default();
        let page_already_protected = !list.is_empty();
        list.push(tb.clone());
        if !page_already_protected {
            self.tlb_protect_code(tb.page_addr[n] & self.page_mask());
        }
    }

    /// `tb_remove()`: take `tb` off its page lists.
    fn tb_remove(&self, pages: &mut HashMap<u64, Vec<Arc<Tb>>>, tb: &Arc<Tb>) {
        for n in 0..2 {
            if tb.page_addr[n] == u64::MAX {
                continue;
            }
            if let Some(list) = pages.get_mut(&self.page_index(tb.page_addr[n])) {
                list.retain(|t| !Arc::ptr_eq(t, tb));
            }
        }
    }

    /// `tb_jmp_cache_inval_tb()`.
    fn tb_jmp_cache_inval_tb(&self, tb: &Arc<Tb>) {
        if tb.cflags() & cf::PCREL != 0 {
            // A TB may be at any virtual address.
            for cpu in self.cpu_list() {
                tcg_flush_jmp_cache(&cpu);
            }
        } else {
            let h = tb_jmp_cache_hash_func(self, tb.pc);
            for cpu in self.cpu_list() {
                let mut jc = lock(&cpu.jmp_cache);
                if jc[h].tb.as_ref().is_some_and(|t| Arc::ptr_eq(t, tb)) {
                    jc[h].tb = None;
                }
            }
        }
    }

    /// `tb_remove_from_jmp_list()`.
    fn tb_remove_from_jmp_list(&self, orig: &Arc<Tb>, n_orig: usize) {
        // Mark the slot so that no further jumps can be inserted.
        let dest_weak = {
            let mut d = lock(&orig.jmp_dest[n_orig]);
            d.mark = true;
            d.dest.clone()
        };
        let Some(dest) = dest_weak.as_ref().and_then(std::sync::Weak::upgrade) else { return };
        let mut list = lock(&dest.jmp_lock);
        // While acquiring the lock, the jump might have been removed if the destination TB was
        // invalidated; check again.
        if !lock(&orig.jmp_dest[n_orig]).same(&dest_weak, true) {
            return;
        }
        if Arc::ptr_eq(&dest, orig) {
            // A TB that links to itself: removing the entry means tb_jmp_unlink won't see it.
            self.backend.set_jmp_target(orig, n_orig, None);
        }
        if let Some(i) = list
            .iter()
            .position(|(t, n)| *n == n_orig && std::ptr::eq(t.as_ptr(), Arc::as_ptr(orig)))
        {
            list.remove(i);
        }
    }

    /// `tb_jmp_unlink()`: remove every jump into `dest`.
    fn tb_jmp_unlink(&self, dest: &Arc<Tb>) {
        let mut list = lock(&dest.jmp_lock);
        for (src, n) in list.drain(..) {
            if let Some(src) = src.upgrade() {
                self.backend.set_jmp_target(&src, n, None);
                lock(&src.jmp_dest[n]).dest = None;
            }
        }
    }

    /// `do_tb_phys_invalidate()`. With `pages`, the block is also taken off its page lists.
    pub(crate) fn do_tb_phys_invalidate(
        &self,
        tb: &Arc<Tb>,
        pages: Option<&mut HashMap<u64, Vec<Arc<Tb>>>>,
    ) {
        // Make sure no further incoming jumps will be chained to this TB.
        {
            let _g = lock(&tb.jmp_lock);
            tb.set_invalid();
        }
        // Remove the TB from the hash list.
        let removed = {
            let mut ht = self.htable.write().unwrap_or_else(|e| e.into_inner());
            let key = tb.key();
            match ht.get(&key) {
                Some(t) if Arc::ptr_eq(t, tb) => {
                    ht.remove(&key);
                    true
                }
                _ => false,
            }
        };
        if removed {
            if let Some(pages) = pages {
                self.tb_remove(pages, tb);
            }
            self.tb_jmp_cache_inval_tb(tb);
            self.tb_remove_from_jmp_list(tb, 0);
            self.tb_remove_from_jmp_list(tb, 1);
            self.tb_jmp_unlink(tb);
            self.tb_phys_invalidate_count.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// `tb_phys_invalidate()`.
    pub fn tb_phys_invalidate(&self, tb: &Arc<Tb>, page_addr: u64) {
        if page_addr == u64::MAX && tb.page_addr[0] != u64::MAX {
            let mut pages = lock(&self.pages);
            self.do_tb_phys_invalidate(tb, Some(&mut pages));
        } else {
            self.do_tb_phys_invalidate(tb, None);
        }
    }

    /// `tb_invalidate_phys_range()`: invalidate every block with code in the `ram_addr` range
    /// `start..=last`.
    pub fn tb_invalidate_phys_range(&self, start: u64, last: u64) {
        let mut pages = lock(&self.pages);
        let index_last = self.page_index(last);
        let mut index = self.page_index(start);
        while index <= index_last {
            if pages.contains_key(&index) {
                let page_start = index << self.config.page_bits;
                let page_last = (page_start | !self.page_mask()).min(last);
                let r = tb_invalidate_phys_page_range_locked(
                    self,
                    None,
                    &mut pages,
                    page_start.max(start),
                    page_last,
                    Ra::None,
                );
                debug_assert!(r.is_ok());
                let _ = r;
            }
            index += 1;
        }
    }

    /// `tb_flush__exclusive_or_serial()`: drop every block. Call it with no other vCPU
    /// running generated code.
    pub fn tb_flush_exclusive_or_serial(&self) {
        for cpu in self.cpu_list() {
            tcg_flush_jmp_cache(&cpu);
        }
        self.htable.write().unwrap_or_else(|e| e.into_inner()).clear();
        // tb_remove_all(): the page lists empty, the pages stay protected.
        for list in lock(&self.pages).values_mut() {
            list.clear();
        }
        // tcg_region_reset_all()
        {
            let mut r = lock(&self.region);
            r.used = 0;
            r.full = false;
            r.tbs.clear();
        }
        self.tb_flush_count.fetch_add(1, Ordering::AcqRel);
        self.plugin_flush();
    }

    /// `tb_add_jump()`: chain slot `n` of `tb` to `tb_next`.
    pub(crate) fn tb_add_jump(&self, tb: &Arc<Tb>, n: usize, tb_next: &Arc<Tb>) {
        assert!(n < 2);
        let mut list = lock(&tb_next.jmp_lock);
        // Make sure the destination TB is valid.
        if tb_next.is_invalid() {
            return;
        }
        {
            // Atomically claim the jump destination slot only if it was NULL.
            let mut d = lock(&tb.jmp_dest[n]);
            if d.dest.is_some() || d.mark {
                return;
            }
            d.dest = Some(Arc::downgrade(tb_next));
        }
        // Patch the native jump address.
        self.backend.set_jmp_target(tb, n, Some(tb_next));
        list.push((Arc::downgrade(tb), n));
    }
}

/// `tb_invalidate_phys_page_range__locked()`: invalidate the blocks with code in
/// `start..=last`, which must be in one page. With a CPU and `ra`, a block that modifies itself
/// is stopped when the target has precise SMC.
pub(crate) fn tb_invalidate_phys_page_range_locked(
    jit: &Jit,
    mut cpu: Option<&mut Cpu<'_>>,
    pages: &mut HashMap<u64, Vec<Arc<Tb>>>,
    start: u64,
    last: u64,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    // Range may not cross a page.
    debug_assert!((start ^ last) & jit.page_mask() == 0);
    let idx = start >> jit.config.page_bits;
    let precise = cpu.as_ref().is_some_and(|c| c.core.ops.precise_smc());
    let current_tb = match (&cpu, ra) {
        (Some(c), Ra::Tb) if precise => c.core.current_tb.clone(),
        _ => None,
    };
    let mut current_tb_modified = false;
    let list: Vec<Arc<Tb>> = pages.get(&idx).cloned().unwrap_or_default();
    for tb in &list {
        // A TB may span two physical pages.
        let n = if tb.page_addr[0] >> jit.config.page_bits == idx { 0 } else { 1 };
        let mut tb_start = tb.page_addr[0];
        let mut tb_last = tb_start + u64::from(tb.size).max(1) - 1;
        if n == 0 {
            tb_last = tb_last.min(tb_start | !jit.page_mask());
        } else {
            tb_start = tb.page_addr[1];
            tb_last = tb_start + (tb_last & !jit.page_mask());
        }
        if !(tb_last < start || tb_start > last) {
            if let Some(cur) = &current_tb {
                if Arc::ptr_eq(cur, tb) && cur.cflags() & cf::COUNT_MASK != 1 {
                    // We are modifying the current TB, so stop its execution.
                    current_tb_modified = true;
                    if let Some(c) = cpu.as_mut() {
                        c.cpu_restore_state(ra);
                    }
                }
            }
            jit.do_tb_phys_invalidate(tb, Some(pages));
        }
    }
    // If no code remaining, no need to continue to use slow writes.
    if pages.get(&idx).is_none_or(Vec::is_empty) {
        jit.tlb_unprotect_code(start);
    }
    if current_tb_modified {
        let c = cpu.expect("current_tb_modified needs a CPU");
        // Force execution of one insn next time.
        c.core.cflags_next_tb = 1 | cf::NOIRQ | c.curr_cflags();
        return Err(c.cpu_loop_exit_noexc());
    }
    Ok(())
}

/// `tb_invalidate_phys_range_fast()`: a write of `len` bytes at `ram_addr` from the softmmu.
pub(crate) fn tb_invalidate_phys_range_fast(
    cpu: &mut Cpu<'_>,
    ram_addr: u64,
    len: u64,
    ra: Ra,
) -> Result<(), CpuLoopExit> {
    let jit = cpu.jit();
    let mut pages = lock(&jit.pages);
    if !pages.contains_key(&(ram_addr >> jit.config.page_bits)) {
        return Ok(());
    }
    let last = ram_addr + len - 1;
    // The access may cross into the next page; the second page is a separate access in QEMU.
    let page_last = last.min(ram_addr | !jit.page_mask());
    let r =
        tb_invalidate_phys_page_range_locked(&jit, Some(cpu), &mut pages, ram_addr, page_last, ra);
    drop(pages);
    r
}

/// `queue_tb_flush()`: flush the code buffer from `cpu` with every vCPU stopped.
pub fn queue_tb_flush(cpu: &Cpu<'_>) {
    let jit = cpu.jit();
    let tb_flush_count = jit.tb_flush_count();
    cpu.core.shared.async_safe_run_on_cpu(move |c| {
        // If it is already been done on request of another CPU, just retry.
        let jit = c.jit();
        if jit.tb_flush_count() == tb_flush_count {
            jit.tb_flush_exclusive_or_serial();
        }
    });
}
