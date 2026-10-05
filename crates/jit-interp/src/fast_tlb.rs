// SPDX-License-Identifier: MIT OR Apache-2.0

//! The TLB tables that generated code reads for its inline softmmu lookups, QEMU's
//! `CPUTLBDescFast` and `CPUTLBEntry`.
//!
//! For each MMU index there is a table of entries of [`TLB_ENTRY_WORDS`] words: the read, write
//! and code comparators, then the addend. A comparator is the page address of the entry with
//! flag bits in its low bits, or `u64::MAX` when that kind of access is not allowed. The addend
//! added to a guest virtual address gives the host address of the byte.
//!
//! A host backend that inlines the lookup loads the two descriptor words of the access's MMU
//! index from [`FastTlb::desc`]: the mask `(entries - 1) << TLB_ENTRY_BITS` and the address of
//! the table. The entry is at `table + ((addr >> (page_bits - TLB_ENTRY_BITS)) & mask)`. When
//! the page of the access, with the low address bits that must be zero for its alignment, equals
//! the comparator, the access goes straight to host memory at `addr + addend`; any flag bit in
//! the comparator makes the compare fail and sends the access to the slow path, as in QEMU.
//!
//! Generated code reads the tables without any lock while the softmmu changes them, so every word
//! is atomic. [`TlbTables::set`] is the only way to install an entry that can match, and it is
//! unsafe: its caller promises that the host memory of every page it maps stays there as long as
//! the entry does. Everything else only invalidates entries, or adds or removes flag bits, which
//! never makes an entry match a page other than its own.
//!
//! A table is replaced (on a resize) only on the thread that runs generated code against it,
//! between or during its runs, and a table replaced during a run is kept until the run ends.
//! So an address that generated code read from the descriptor stays good for the whole run.
//!
//! Differences from QEMU:
//!
//! - Generated code finds the descriptor through its run context rather than at a fixed negative
//!   offset from `env`, because the CPU state here belongs to the target and has no room for it.
//! - MMU indexes that have no table point at a one entry table that never matches, so an access
//!   with any MMU index below [`TLB_MAX_MMU_MODES`] can be looked up.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, ThreadId};

/// log2 of the size of an entry in bytes, `CPU_TLB_ENTRY_BITS`.
pub const TLB_ENTRY_BITS: u32 = 5;
/// The words of an entry: three comparators and the addend.
pub const TLB_ENTRY_WORDS: usize = 4;
/// The word of an entry that holds the addend.
pub const TLB_ADDEND_WORD: usize = 3;
/// The MMU indexes a descriptor covers, `NB_MMU_MODES` at its largest. A `MemOpIdx` has four
/// bits for the index.
pub const TLB_MAX_MMU_MODES: usize = 16;
/// The descriptor words per MMU index: the mask, then the table address.
pub const TLB_DESC_WORDS: usize = 2;

/// The lowest bit a comparator flag may use, `TARGET_PAGE_BITS_MIN - 3`. The bits below it are
/// zero in every comparator that can match, so a backend that inlines an access needing at most
/// this many low address bits clear for its alignment can include them in its compare.
pub const TLB_FLAGS_SHIFT: u32 = 6;

/// An entry no access matches.
pub const TLB_INVALID_ENTRY: [u64; TLB_ENTRY_WORDS] = [u64::MAX; TLB_ENTRY_WORDS];

type Table = Arc<[AtomicU64]>;

fn new_table(entries: usize) -> Table {
    (0..entries * TLB_ENTRY_WORDS).map(|_| AtomicU64::new(u64::MAX)).collect()
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Which tables generated code may still be using.
#[derive(Debug)]
struct Keep {
    /// The table each descriptor entry names.
    cur: Vec<Table>,
    /// Tables replaced during the current run.
    retired: Vec<Table>,
    /// The thread of the current run and how many runs it has open.
    runner: Option<(ThreadId, usize)>,
}

/// The part of a softmmu TLB that generated code reads, shared between the TLB, which changes
/// it through [`TlbTables`], and the runs that read it.
#[derive(Debug)]
pub struct FastTlb {
    page_bits: u32,
    desc: Box<[AtomicU64]>,
    keep: Mutex<Keep>,
}

impl FastTlb {
    /// log2 of the guest page size the tables are for, `TARGET_PAGE_BITS`.
    pub fn page_bits(&self) -> u32 {
        self.page_bits
    }

    /// The descriptor: [`TLB_DESC_WORDS`] words per MMU index, [`TLB_MAX_MMU_MODES`] indexes.
    /// The table addresses in it stay valid while a run is open ([`FastTlb::enter_run`]).
    pub fn desc(&self) -> &[AtomicU64] {
        &self.desc
    }

    /// A descriptor whose every entry fails to match, for a run without a TLB to read.
    pub fn miss_desc() -> &'static [AtomicU64] {
        static MISS: OnceLock<(Table, Box<[AtomicU64]>)> = OnceLock::new();
        let (_, desc) = MISS.get_or_init(|| {
            let table = new_table(1);
            let desc = desc_for(&vec![table.clone(); TLB_MAX_MMU_MODES]);
            (table, desc)
        });
        desc
    }

    /// Start a run of generated code that reads the descriptor. Until the guard is dropped,
    /// tables replaced by [`TlbTables::resize`] are kept. Runs nest on one thread.
    ///
    /// # Panics
    ///
    /// If another thread has a run open: a vCPU's TLB is read by that vCPU's thread only.
    pub fn enter_run(&self) -> RunGuard<'_> {
        let me = thread::current().id();
        let mut k = lock(&self.keep);
        k.runner = match k.runner {
            None => Some((me, 1)),
            Some((t, n)) if t == me => Some((t, n + 1)),
            Some(_) => panic!("a TLB is being run against on two threads"),
        };
        RunGuard { tlb: self }
    }
}

/// An open run, from [`FastTlb::enter_run`].
#[derive(Debug)]
pub struct RunGuard<'a> {
    tlb: &'a FastTlb,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        let mut k = lock(&self.tlb.keep);
        k.runner = match k.runner {
            Some((t, n)) if n > 1 => Some((t, n - 1)),
            _ => None,
        };
        if k.runner.is_none() {
            k.retired.clear();
        }
    }
}

fn desc_for(tables: &[Table]) -> Box<[AtomicU64]> {
    let desc: Box<[AtomicU64]> =
        (0..TLB_MAX_MMU_MODES * TLB_DESC_WORDS).map(|_| AtomicU64::new(0)).collect();
    for (i, t) in tables.iter().enumerate() {
        write_desc(&desc, i, t);
    }
    desc
}

fn write_desc(desc: &[AtomicU64], mmu_idx: usize, t: &Table) {
    let entries = (t.len() / TLB_ENTRY_WORDS) as u64;
    let mask = (entries - 1) << TLB_ENTRY_BITS;
    let addr = t.as_ptr() as u64;
    // A run on this thread reads both words after this; no other thread reads them during
    // a run (see FastTlb::enter_run).
    desc[mmu_idx * TLB_DESC_WORDS + 1].store(addr, Ordering::Release);
    desc[mmu_idx * TLB_DESC_WORDS].store(mask, Ordering::Release);
}

/// The tables of a softmmu TLB, owned by the TLB. Reads and changes here need no lock beyond
/// the one the TLB holds.
#[derive(Debug)]
pub struct TlbTables {
    cur: Vec<Table>,
    fast: Arc<FastTlb>,
}

impl TlbTables {
    /// Tables of `entries` entries each (a power of two above 1) for `nb_mmu_modes` MMU indexes, all
    /// invalid, for pages of `1 << page_bits` bytes.
    ///
    /// # Panics
    ///
    /// If `nb_mmu_modes` is above [`TLB_MAX_MMU_MODES`], `entries` is not a power of two above
    /// 1, or `page_bits` is not above [`TLB_FLAGS_SHIFT`] + 3.
    pub fn new(page_bits: u32, nb_mmu_modes: usize, entries: usize) -> TlbTables {
        assert!(nb_mmu_modes <= TLB_MAX_MMU_MODES, "too many MMU indexes");
        assert!(entries.is_power_of_two() && entries > 1, "bad TLB size {entries}");
        assert!((TLB_FLAGS_SHIFT + 3..64).contains(&page_bits), "bad page size");
        let cur: Vec<Table> = (0..nb_mmu_modes).map(|_| new_table(entries)).collect();
        let miss = new_table(1);
        let mut named = cur.clone();
        named.resize(TLB_MAX_MMU_MODES, miss);
        let desc = desc_for(&named);
        let keep = Mutex::new(Keep { cur: named, retired: Vec::new(), runner: None });
        TlbTables { cur, fast: Arc::new(FastTlb { page_bits, desc, keep }) }
    }

    /// The part generated code reads.
    pub fn fast(&self) -> &Arc<FastTlb> {
        &self.fast
    }

    /// The number of entries of `mmu_idx`'s table.
    pub fn len(&self, mmu_idx: usize) -> usize {
        self.cur[mmu_idx].len() / TLB_ENTRY_WORDS
    }

    fn word(&self, mmu_idx: usize, index: usize, w: usize) -> &AtomicU64 {
        &self.cur[mmu_idx][index * TLB_ENTRY_WORDS + w]
    }

    /// Entry `index` of `mmu_idx`'s table.
    pub fn get(&self, mmu_idx: usize, index: usize) -> [u64; TLB_ENTRY_WORDS] {
        std::array::from_fn(|w| self.word(mmu_idx, index, w).load(Ordering::Relaxed))
    }

    /// Install entry `index` of `mmu_idx`'s table.
    ///
    /// # Safety
    ///
    /// For each comparator that is not `u64::MAX`, `page` being that comparator with its bits
    /// below `1 << page_bits` cleared, the host bytes from `page + addend` for a whole page must
    /// stay valid, for reads (and for writes if it is the write comparator, word 1), until the
    /// entry is replaced or invalidated or its table resized. Generated code accesses them
    /// without any other check.
    ///
    /// # Panics
    ///
    /// If a comparator that is not `u64::MAX` has flag bits below [`TLB_FLAGS_SHIFT`] or is for
    /// a page that does not belong at `index`.
    #[allow(unsafe_code)]
    pub unsafe fn set(&self, mmu_idx: usize, index: usize, e: [u64; TLB_ENTRY_WORDS]) {
        let page_bits = self.fast.page_bits;
        let n = self.len(mmu_idx) as u64;
        for &c in &e[..TLB_ADDEND_WORD] {
            if c != u64::MAX {
                assert!(c & ((1 << TLB_FLAGS_SHIFT) - 1) == 0, "TLB comparator {c:#x}");
                assert!((c >> page_bits) & (n - 1) == index as u64, "TLB entry in the wrong slot");
            }
        }
        for (w, &v) in e.iter().enumerate() {
            self.word(mmu_idx, index, w).store(v, Ordering::Relaxed);
        }
    }

    /// Make entry `index` of `mmu_idx`'s table match nothing.
    pub fn invalidate(&self, mmu_idx: usize, index: usize) {
        for w in 0..TLB_ENTRY_WORDS {
            self.word(mmu_idx, index, w).store(u64::MAX, Ordering::Relaxed);
        }
    }

    /// Make every entry of `mmu_idx`'s table match nothing.
    pub fn invalidate_all(&self, mmu_idx: usize) {
        for w in self.cur[mmu_idx].iter() {
            w.store(u64::MAX, Ordering::Relaxed);
        }
    }

    fn check_flags(&self, flags: u64) {
        let ok = flags >> self.fast.page_bits == 0 && flags & ((1 << TLB_FLAGS_SHIFT) - 1) == 0;
        assert!(ok, "TLB flags {flags:#x} outside the flag bits");
    }

    /// Set `flags`, which must be flag bits (from [`TLB_FLAGS_SHIFT`] up to the page bits), in
    /// comparator `w` of an entry. A comparator of `u64::MAX` stays as it is.
    pub fn add_flags(&self, mmu_idx: usize, index: usize, w: usize, flags: u64) {
        assert!(w < TLB_ADDEND_WORD);
        self.check_flags(flags);
        self.word(mmu_idx, index, w).fetch_or(flags, Ordering::Relaxed);
    }

    /// Store `new` in comparator `w` of an entry if it holds `old` and the two differ only in
    /// flag bits. True if it was stored.
    pub fn replace_flags(
        &self,
        mmu_idx: usize,
        index: usize,
        w: usize,
        old: u64,
        new: u64,
    ) -> bool {
        assert!(w < TLB_ADDEND_WORD);
        self.check_flags(old ^ new);
        if old == u64::MAX {
            return false;
        }
        self.word(mmu_idx, index, w)
            .compare_exchange(old, new, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    }

    /// Replace `mmu_idx`'s table with one of `entries` entries (a power of two above 1), all
    /// invalid.
    ///
    /// # Panics
    ///
    /// If `entries` is not a power of two above 1, or a run is open on another thread.
    pub fn resize(&mut self, mmu_idx: usize, entries: usize) {
        assert!(entries.is_power_of_two() && entries > 1, "bad TLB size {entries}");
        let t = new_table(entries);
        let mut k = lock(&self.fast.keep);
        let open = match k.runner {
            Some((thread, _)) => {
                assert!(
                    thread == thread::current().id(),
                    "TLB resized during another thread's run"
                );
                true
            }
            None => false,
        };
        write_desc(&self.fast.desc, mmu_idx, &t);
        let old = std::mem::replace(&mut k.cur[mmu_idx], t.clone());
        if open {
            k.retired.push(old);
        }
        drop(k);
        self.cur[mmu_idx] = t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(f: &FastTlb, mmu_idx: usize) -> (u64, u64) {
        let d = f.desc();
        (
            d[mmu_idx * TLB_DESC_WORDS].load(Ordering::Relaxed),
            d[mmu_idx * TLB_DESC_WORDS + 1].load(Ordering::Relaxed),
        )
    }

    #[test]
    #[allow(unsafe_code)]
    fn the_descriptor_names_each_table() {
        let mut t = TlbTables::new(12, 3, 256);
        let f = Arc::clone(t.fast());
        assert_eq!(desc(&f, 1).0, 255 << TLB_ENTRY_BITS);
        // Indexes without a table have a one entry table that never matches.
        assert_eq!(desc(&f, 3).0, 0);
        assert_ne!(desc(&f, 15).1, 0);
        // SAFETY: no generated code runs in this test.
        unsafe { t.set(1, 7, [0x7000, 0x7000 | 0x80, u64::MAX, 42]) };
        assert_eq!(t.get(1, 7), [0x7000, 0x7080, u64::MAX, 42]);
        assert!(t.replace_flags(1, 7, 1, 0x7080, 0x7000));
        assert!(!t.replace_flags(1, 7, 1, 0x7080, 0x7000));
        t.add_flags(1, 7, 0, 0x100);
        assert_eq!(t.get(1, 7)[0], 0x7100);
        t.resize(1, 64);
        assert_eq!(t.len(1), 64);
        assert_eq!(desc(&f, 1).0, 63 << TLB_ENTRY_BITS);
        assert_eq!(t.get(1, 7), TLB_INVALID_ENTRY);
        t.invalidate_all(0);
    }

    #[test]
    fn a_table_replaced_during_a_run_lives_until_it_ends() {
        let mut t = TlbTables::new(12, 1, 16);
        let f = Arc::clone(t.fast());
        let run = f.enter_run();
        let inner = f.enter_run();
        t.resize(0, 32);
        drop(inner);
        assert_eq!(lock(&f.keep).retired.len(), 1);
        drop(run);
        assert!(lock(&f.keep).retired.is_empty());
        t.resize(0, 16);
        assert!(lock(&f.keep).retired.is_empty());
    }

    #[test]
    #[should_panic(expected = "outside the flag bits")]
    fn flags_stay_below_the_page() {
        let t = TlbTables::new(12, 1, 16);
        t.add_flags(0, 0, 0, 0x1000);
    }

    #[test]
    #[should_panic(expected = "wrong slot")]
    #[allow(unsafe_code)]
    fn an_entry_goes_in_the_slot_of_its_page() {
        let t = TlbTables::new(12, 1, 16);
        // SAFETY: the entry is refused before it is stored.
        unsafe { t.set(0, 3, [0x4000, u64::MAX, u64::MAX, 0]) };
    }
}
