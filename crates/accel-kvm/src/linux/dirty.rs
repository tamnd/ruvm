// SPDX-License-Identifier: GPL-2.0-or-later

//! Harvesting the KVM dirty ring, `kvm_dirty_ring_reap()` and its helpers in
//! accel/kvm/kvm-all.c.
//!
//! With `dirty-ring-size` set, KVM pushes the guest frame of every page the guest dirties in a
//! logged slot onto the ring of the vCPU that wrote it. Reaping walks each ring from where the
//! last reap stopped, sets the page's bit in its slot's bitmap, marks the entry collected and,
//! once all rings are walked, issues `KVM_RESET_DIRTY_RINGS` so the kernel write protects the
//! pages again and reuses the entries. The slot bitmaps are copied into the RAM blocks when the
//! memory system syncs, which is where migration takes its dirty pages from.
//!
//! Three things reap, as in QEMU: the `kvm-reaper` thread once a second, a vCPU whose ring
//! filled up (`KVM_EXIT_DIRTY_RING_FULL`), and the global sync, which first kicks every vCPU out
//! of `KVM_RUN` so that pages the CPU still holds in its own buffers (Intel's PML) reach the
//! ring.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use kvm_ioctls::{VcpuFd, VmFd};
use ruvm_base::error_report;
use ruvm_mem::{DirtyMask, RamBlock};
use ruvm_sys::kvm::{DirtyGfn, DirtyRingMap, reset_dirty_rings};

use super::vcpu::VcpuKick;

/// The host page size on x86, which sizes the slots and places the ring in the vCPU mapping.
pub(crate) const HOST_PAGE: u64 = 4096;

/// The number of KVM address spaces. ruvm has no SMM address space, so every slot is in 0.
const NR_AS: u32 = 1;

/// The period of the reaper thread, the `sleep(1)` in `kvm_dirty_ring_reaper_thread()`.
const REAP_PERIOD: Duration = Duration::from_secs(1);

/// How long a synchronous kick waits for a vCPU to come out of `KVM_RUN`. QEMU waits for good
/// through `run_on_cpu()`; a vCPU that does not answer in this long has nothing to flush that
/// the next sync will not pick up.
const KICK_WAIT: Duration = Duration::from_secs(1);

/// A slot's dirty bitmap, `KVMSlot.dirty_bmap`: one bit per host page, in 64 bit words.
#[derive(Debug)]
pub(crate) struct SlotBitmap {
    pages: u64,
    bits: Vec<u64>,
}

impl SlotBitmap {
    /// `kvm_slot_init_dirty_bitmap()`: the bitmap is rounded up to whole words.
    fn new(pages: u64) -> Self {
        SlotBitmap { pages, bits: vec![0; pages.div_ceil(64) as usize] }
    }

    fn mark(&mut self, page: u64) {
        if page < self.pages {
            self.bits[(page / 64) as usize] |= 1 << (page % 64);
        }
    }

    /// Takes the bits and leaves the bitmap clean, the sync and the `memset()` after it.
    fn take(&mut self) -> Vec<u64> {
        std::mem::replace(&mut self.bits, vec![0; self.pages.div_ceil(64) as usize])
    }
}

/// Where a vCPU's ring lives.
#[derive(Debug)]
enum RingMem {
    /// Mapped from the vCPU descriptor.
    Mapped(DirtyRingMap),
    /// In ordinary memory, for the tests.
    #[cfg(test)]
    Owned(Box<[DirtyGfn]>),
}

/// A vCPU's dirty ring with the state QEMU keeps for it in `CPUState`.
#[derive(Debug)]
pub(crate) struct VcpuRing {
    mem: RingMem,
    /// `kvm_fetch_index`: the next entry to look at. Only used with the harvest lock held.
    fetch: AtomicU32,
    /// Whether the vCPU thread is inside `KVM_RUN`.
    in_run: AtomicBool,
    /// How many times the vCPU thread came back from `KVM_RUN`.
    exits: AtomicU64,
    /// The kick for the vCPU thread, set the first time it runs.
    kick: OnceLock<VcpuKick>,
}

impl VcpuRing {
    fn new(mem: RingMem) -> Self {
        VcpuRing {
            mem,
            fetch: AtomicU32::new(0),
            in_run: AtomicBool::new(false),
            exits: AtomicU64::new(0),
            kick: OnceLock::new(),
        }
    }

    fn entries(&self) -> &[DirtyGfn] {
        match &self.mem {
            RingMem::Mapped(m) => m.entries(),
            #[cfg(test)]
            RingMem::Owned(e) => e,
        }
    }

    /// Records how to kick the vCPU thread. Only the first call counts.
    pub(crate) fn set_kick(&self, kick: impl FnOnce() -> VcpuKick) {
        self.kick.get_or_init(kick);
    }

    /// The vCPU thread is about to enter `KVM_RUN`.
    pub(crate) fn enter_run(&self) {
        self.in_run.store(true, Ordering::SeqCst);
    }

    /// The vCPU thread came back from `KVM_RUN`.
    pub(crate) fn leave_run(&self) {
        self.exits.fetch_add(1, Ordering::SeqCst);
        self.in_run.store(false, Ordering::SeqCst);
    }
}

/// `kvm_dirty_ring_mark_page()`: sets the bit of page `offset` of slot `slot_id` in address
/// space `as_id`. Entries for an address space or slot that is not logged, or past the end of
/// the slot, are dropped, as QEMU drops them.
fn mark_page(slots: &mut [Option<SlotBitmap>], as_id: u32, slot_id: u32, offset: u64) {
    if as_id >= NR_AS {
        return;
    }
    if let Some(Some(bmap)) = slots.get_mut(slot_id as usize) {
        bmap.mark(offset);
    }
}

/// `kvm_dirty_ring_reap_one()`: walks `ring` from its fetch index until the first entry the
/// kernel has not published, marks each page and hands the entry back. Returns how many entries
/// it took.
fn reap_one(ring: &VcpuRing, slots: &mut [Option<SlotBitmap>]) -> u32 {
    let entries = ring.entries();
    let size = entries.len() as u32;
    let mut fetch = ring.fetch.load(Ordering::Relaxed);
    let mut count = 0;
    loop {
        let cur = &entries[(fetch % size) as usize];
        if !cur.is_dirtied() {
            break;
        }
        let slot = cur.slot.load(Ordering::Relaxed);
        let offset = cur.offset.load(Ordering::Relaxed);
        mark_page(slots, slot >> 16, slot & 0xffff, offset);
        cur.set_collected();
        fetch = fetch.wrapping_add(1);
        count += 1;
    }
    ring.fetch.store(fetch, Ordering::Relaxed);
    count
}

/// The rings and the slot bitmaps they are reaped into, which QEMU keeps in `KVMState` and the
/// `KVMSlot`s under `kvm_slots_lock()`.
#[derive(Debug, Default)]
struct Harvest {
    rings: Vec<Arc<VcpuRing>>,
    /// Indexed by slot id; `None` for a slot that is not logged.
    slots: Vec<Option<SlotBitmap>>,
}

impl Harvest {
    /// The walk of `kvm_dirty_ring_reap_locked()`, over `only` or over every ring.
    fn reap(&mut self, only: Option<&VcpuRing>) -> u32 {
        match only {
            Some(ring) => reap_one(ring, &mut self.slots),
            None => self.rings.iter().map(|r| reap_one(r, &mut self.slots)).sum(),
        }
    }
}

/// The dirty rings of a VM with the slot bitmaps they feed.
#[derive(Debug)]
pub(crate) struct DirtyRings {
    vm: Arc<VmFd>,
    /// `kvm_dirty_ring_size`: entries per ring, 0 with the dirty bitmap instead.
    size: u32,
    /// `kvm_dirty_ring_with_bitmap`.
    with_bitmap: bool,
    harvest: Mutex<Harvest>,
}

impl DirtyRings {
    /// The rings of `vm` with `size` entries each, none of them mapped yet. When the ring is on
    /// this also starts the reaper thread, `kvm_dirty_ring_reaper_init()`.
    pub(crate) fn new(vm: Arc<VmFd>, size: u32, with_bitmap: bool) -> io::Result<Arc<Self>> {
        let rings =
            Arc::new(DirtyRings { vm, size, with_bitmap, harvest: Mutex::new(Harvest::default()) });
        if size != 0 {
            let weak = Arc::downgrade(&rings);
            std::thread::Builder::new().name("kvm-reaper".to_string()).spawn(move || {
                loop {
                    std::thread::sleep(REAP_PERIOD);
                    let Some(rings) = weak.upgrade() else { return };
                    rings.reap(None);
                }
            })?;
        }
        Ok(rings)
    }

    fn lock(&self) -> MutexGuard<'_, Harvest> {
        self.harvest.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Whether the dirty ring is in use, `kvm_dirty_ring_enabled()`.
    pub(crate) fn enabled(&self) -> bool {
        self.size != 0
    }

    /// `kvm_dirty_ring_size`.
    pub(crate) fn size(&self) -> u32 {
        self.size
    }

    /// `kvm_dirty_ring_with_bitmap`.
    pub(crate) fn with_bitmap(&self) -> bool {
        self.with_bitmap
    }

    /// `map_kvm_dirty_gfns()`: maps the ring of a new vCPU and adds it to the ones reaped.
    /// Gives `None` when the ring is off.
    pub(crate) fn add_vcpu(&self, fd: &VcpuFd) -> io::Result<Option<Arc<VcpuRing>>> {
        if !self.enabled() {
            return Ok(None);
        }
        let map = DirtyRingMap::map(fd, self.size as usize, HOST_PAGE as usize)?;
        let ring = Arc::new(VcpuRing::new(RingMem::Mapped(map)));
        self.lock().rings.push(Arc::clone(&ring));
        Ok(Some(ring))
    }

    /// `kvm_dirty_ring_reap_locked()`: reaps, then gives the collected entries back to the
    /// kernel.
    fn reap_locked(&self, h: &mut Harvest, only: Option<&VcpuRing>) -> u32 {
        let total = h.reap(only);
        if total > 0 {
            match reset_dirty_rings(&*self.vm) {
                Ok(n) if n == total => {}
                Ok(n) => error_report(&format!(
                    "KVM_RESET_DIRTY_RINGS reset {n} entries where {total} were collected"
                )),
                Err(e) => {
                    error_report(&format!("KVM_RESET_DIRTY_RINGS failed: {}", crate::strerror(&e)))
                }
            }
        }
        total
    }

    /// `kvm_dirty_ring_reap()`: reaps the ring of `only`, or every ring.
    pub(crate) fn reap(&self, only: Option<&VcpuRing>) -> u32 {
        let mut h = self.lock();
        self.reap_locked(&mut h, only)
    }

    /// `kvm_dirty_ring_flush()`: once this returns, every page dirtied before the call is in
    /// its slot's bitmap.
    pub(crate) fn flush(&self) {
        self.kick_all_synchronously();
        self.reap(None);
    }

    /// `kvm_cpu_synchronize_kick_all()`: kicks every vCPU that is inside `KVM_RUN` and waits
    /// until it has come out once, which empties the CPU's own dirty buffers into the ring. A
    /// vCPU outside `KVM_RUN` has nothing buffered.
    fn kick_all_synchronously(&self) {
        let rings = self.lock().rings.clone();
        let mut waits = Vec::new();
        for ring in &rings {
            if !ring.in_run.load(Ordering::SeqCst) {
                continue;
            }
            let seen = ring.exits.load(Ordering::SeqCst);
            if ring.kick.get().is_some_and(|k| k.kick().is_ok()) {
                waits.push((ring, seen));
            }
        }
        let deadline = Instant::now() + KICK_WAIT;
        for (ring, seen) in waits {
            while ring.in_run.load(Ordering::SeqCst)
                && ring.exits.load(Ordering::SeqCst) == seen
                && Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_micros(50));
            }
        }
    }

    /// `kvm_slot_init_dirty_bitmap()` for slot `id` of `pages` pages. A slot that already has a
    /// bitmap keeps it.
    pub(crate) fn init_slot(&self, id: usize, pages: u64) {
        let mut h = self.lock();
        if h.slots.len() <= id {
            h.slots.resize_with(id + 1, || None);
        }
        h.slots[id].get_or_insert_with(|| SlotBitmap::new(pages));
    }

    /// Takes the bits collected for slot `id` and leaves its bitmap clean.
    pub(crate) fn take_slot(&self, id: usize) -> Option<Vec<u64>> {
        self.lock().slots.get_mut(id)?.as_mut().map(SlotBitmap::take)
    }

    /// Before slot `id` goes away: reaps every ring, so nothing still queued for the slot is
    /// lost, then takes its bits and drops its bitmap.
    pub(crate) fn retire_slot(&self, id: usize) -> Option<Vec<u64>> {
        let mut h = self.lock();
        self.reap_locked(&mut h, None);
        h.slots.get_mut(id)?.take().map(|mut b| b.take())
    }

    /// Drops the bitmap of slot `id`.
    pub(crate) fn drop_slot(&self, id: usize) {
        if let Some(s) = self.lock().slots.get_mut(id) {
            *s = None;
        }
    }
}

/// `physical_memory_set_dirty_lebitmap()` for one slot: marks dirty in `block`, for the clients
/// in `mask`, the first `pages` host pages from `offset` on whose bits are set in `bits`. Each
/// run of set bits goes in as one range. Returns how many pages were set.
pub(crate) fn set_dirty_lebitmap(
    block: &RamBlock,
    offset: u64,
    bits: &[u64],
    pages: u64,
    mask: DirtyMask,
) -> u64 {
    let mut run: Option<u64> = None;
    let mut count = 0;
    let mut flush = |start: u64, end: u64| {
        block.set_dirty(offset + start * HOST_PAGE, (end - start) * HOST_PAGE, mask);
        count += end - start;
    };
    for (i, &word) in bits.iter().enumerate() {
        let base = i as u64 * 64;
        if base >= pages {
            break;
        }
        if run.is_none() && word == 0 {
            continue;
        }
        if let (Some(_), u64::MAX) = (run, word) {
            continue;
        }
        for bit in 0..64 {
            let page = base + bit;
            if page >= pages {
                break;
            }
            match ((word >> bit) & 1 != 0, run) {
                (true, None) => run = Some(page),
                (false, Some(start)) => {
                    flush(start, page);
                    run = None;
                }
                _ => {}
            }
        }
    }
    if let Some(start) = run {
        let end = pages.min(bits.len() as u64 * 64);
        flush(start, end);
    }
    count
}

#[cfg(test)]
mod tests {
    use ruvm_mem::{DirtyClient, MemorySystem};
    use ruvm_sys::kvm::{KVM_DIRTY_GFN_F_DIRTY, KVM_DIRTY_GFN_F_RESET};

    use super::*;

    fn ring(entries: usize) -> Arc<VcpuRing> {
        let mem = (0..entries).map(|_| DirtyGfn::default()).collect();
        Arc::new(VcpuRing::new(RingMem::Owned(mem)))
    }

    /// What the kernel does when the guest dirties a page: fill the entry, then publish it.
    fn push(ring: &VcpuRing, index: usize, as_id: u32, slot: u32, offset: u64) {
        let e = &ring.entries()[index % ring.entries().len()];
        e.slot.store((as_id << 16) | slot, Ordering::Relaxed);
        e.offset.store(offset, Ordering::Relaxed);
        e.flags.store(KVM_DIRTY_GFN_F_DIRTY, Ordering::Release);
    }

    fn bit(h: &Harvest, slot: usize, page: u64) -> bool {
        let b = h.slots[slot].as_ref().unwrap();
        (b.bits[(page / 64) as usize] >> (page % 64)) & 1 != 0
    }

    #[test]
    fn reaping_marks_pages_and_collects_entries() {
        let r = ring(8);
        let mut h = Harvest {
            rings: vec![Arc::clone(&r)],
            slots: vec![Some(SlotBitmap::new(16)), None, Some(SlotBitmap::new(8))],
        };
        push(&r, 0, 0, 0, 3);
        push(&r, 1, 0, 2, 7);
        // A slot that is not logged, another address space, a page past the end of the slot
        // and a slot that does not exist: all taken off the ring, none marked.
        push(&r, 2, 0, 1, 0);
        push(&r, 3, 1, 0, 4);
        push(&r, 4, 0, 0, 16);
        push(&r, 5, 0, 9, 0);

        assert_eq!(h.reap(None), 6);
        assert!(bit(&h, 0, 3));
        assert!(bit(&h, 2, 7));
        assert!(!bit(&h, 0, 4));
        assert_eq!(h.slots[0].as_ref().unwrap().bits, vec![1 << 3]);
        for e in &r.entries()[..6] {
            assert_eq!(e.flags.load(Ordering::Relaxed), KVM_DIRTY_GFN_F_RESET);
        }
        assert_eq!(r.fetch.load(Ordering::Relaxed), 6);
        // Collected entries are not taken twice.
        assert_eq!(h.reap(None), 0);
    }

    #[test]
    fn reaping_stops_at_the_first_unpublished_entry() {
        let r = ring(4);
        let mut h = Harvest { rings: vec![Arc::clone(&r)], slots: vec![Some(SlotBitmap::new(8))] };
        push(&r, 0, 0, 0, 1);
        push(&r, 2, 0, 0, 2);
        assert_eq!(h.reap(None), 1);
        assert!(!bit(&h, 0, 2));
        push(&r, 1, 0, 0, 5);
        assert_eq!(h.reap(None), 2);
        assert!(bit(&h, 0, 2) && bit(&h, 0, 5));
    }

    #[test]
    fn the_fetch_index_wraps_around_the_ring() {
        let r = ring(4);
        let mut h = Harvest { rings: vec![Arc::clone(&r)], slots: vec![Some(SlotBitmap::new(64))] };
        for i in 0..3 {
            push(&r, i, 0, 0, i as u64);
        }
        assert_eq!(h.reap(None), 3);
        // The kernel reuses entries after the reset and carries on from index 3.
        for i in 3..7 {
            push(&r, i, 0, 0, 10 + i as u64);
        }
        assert_eq!(h.reap(None), 4);
        assert_eq!(r.fetch.load(Ordering::Relaxed), 7);
        assert_eq!(h.slots[0].as_ref().unwrap().bits, vec![0b111 | (0b1111 << 13)]);
    }

    #[test]
    fn a_full_ring_reaps_one_vcpu_or_all() {
        let (a, b) = (ring(4), ring(4));
        let mut h = Harvest {
            rings: vec![Arc::clone(&a), Arc::clone(&b)],
            slots: vec![Some(SlotBitmap::new(8))],
        };
        push(&a, 0, 0, 0, 1);
        push(&b, 0, 0, 0, 2);
        assert_eq!(h.reap(Some(&a)), 1);
        assert!(!bit(&h, 0, 2));
        assert_eq!(h.reap(None), 1);
        assert!(bit(&h, 0, 2));
    }

    #[test]
    fn slot_bitmaps_round_up_and_take_clears() {
        let mut b = SlotBitmap::new(65);
        assert_eq!(b.bits.len(), 2);
        b.mark(64);
        b.mark(65);
        assert_eq!(b.take(), vec![0, 1]);
        assert_eq!(b.bits, vec![0, 0]);
    }

    #[test]
    fn bitmaps_land_in_the_ram_block() {
        let mem = MemorySystem::new();
        let ram = mem.new_ram("ram", 128 * HOST_PAGE).unwrap();
        let block = mem.ram_block(ram).unwrap();
        block.start_dirty_log(DirtyClient::Migration);
        block.start_dirty_log(DirtyClient::Vga);
        let bits = [0b1011 | (1 << 63), 0b1 | (1 << 10)];
        // The slot starts 8 pages into the block and has 70 pages, so bit 74 is out of it.
        let n = set_dirty_lebitmap(&block, 8 * HOST_PAGE, &bits, 70, DirtyClient::Migration.mask());
        assert_eq!(n, 5);
        let dirty: Vec<u64> = (0..128)
            .filter(|p| block.get_dirty(p * HOST_PAGE, HOST_PAGE, DirtyClient::Migration))
            .collect();
        assert_eq!(dirty, vec![8, 9, 11, 71, 72]);
        assert_eq!(block.dirty_pages(DirtyClient::Vga), 0);

        // A run that reaches the end of the slot.
        let n = set_dirty_lebitmap(&block, 0, &[u64::MAX, u64::MAX], 100, DirtyMask::ALL);
        assert_eq!(n, 100);
        assert_eq!(block.dirty_pages(DirtyClient::Vga), 100);
    }
}
