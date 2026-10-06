// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ram` section, version 4, migration/ram.c.
//!
//! The `QEMU_VM_SECTION_START` payload lists the RAM blocks: the total size with
//! `RAM_SAVE_FLAG_MEM_SIZE`, then each block's name and length. Every section after that is a run
//! of pages, each a big-endian word of the page offset and its flags, the block name unless
//! `RAM_SAVE_FLAG_CONTINUE` says it is the block of the page before, and either one zero byte
//! (`RAM_SAVE_FLAG_ZERO`) or the whole page (`RAM_SAVE_FLAG_PAGE`). `RAM_SAVE_FLAG_EOS` ends
//! each section.
//!
//! Precopy sends every page once, then keeps sending the pages the guest wrote since, which the
//! `DIRTY_MEMORY_MIGRATION` bitmaps of the RAM blocks record.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ruvm_base::{Error, Result, bail, warn_report};
use ruvm_mem::{DirtyClient, RamBlock};
use ruvm_vmstate::StreamReader;

use crate::savevm::{LiveState, QemuFile};

/// `RAM_SAVE_FLAG_FULL`, obsolete.
pub const RAM_SAVE_FLAG_FULL: u64 = 0x01;
/// `RAM_SAVE_FLAG_ZERO`: a page of zeros, sent as one byte.
pub const RAM_SAVE_FLAG_ZERO: u64 = 0x02;
/// `RAM_SAVE_FLAG_MEM_SIZE`: the block list follows.
pub const RAM_SAVE_FLAG_MEM_SIZE: u64 = 0x04;
/// `RAM_SAVE_FLAG_PAGE`: a whole page follows.
pub const RAM_SAVE_FLAG_PAGE: u64 = 0x08;
/// `RAM_SAVE_FLAG_EOS`: the end of the section.
pub const RAM_SAVE_FLAG_EOS: u64 = 0x10;
/// `RAM_SAVE_FLAG_CONTINUE`: the page is in the block of the page before.
pub const RAM_SAVE_FLAG_CONTINUE: u64 = 0x20;
/// `RAM_SAVE_FLAG_XBZRLE`: a page encoded against the last one sent.
pub const RAM_SAVE_FLAG_XBZRLE: u64 = 0x40;
/// `RAM_SAVE_FLAG_HOOK`: RDMA registration.
pub const RAM_SAVE_FLAG_HOOK: u64 = 0x80;
/// `RAM_SAVE_FLAG_MULTIFD_FLUSH`: wait for the multifd channels.
pub const RAM_SAVE_FLAG_MULTIFD_FLUSH: u64 = 0x200;

/// The version of the `ram` section.
pub const RAM_SECTION_VERSION: i32 = 4;

/// What the RAM saver asks of the machine around the dirty bitmaps.
pub trait RamHooks: Send {
    /// `global_dirty_log_start(GLOBAL_DIRTY_MIGRATION)`: tells the memory listeners (an
    /// accelerator that tracks writes itself) that migration logging is on.
    fn log_start(&self) {}

    /// `global_dirty_log_stop(GLOBAL_DIRTY_MIGRATION)`.
    fn log_stop(&self) {}

    /// `memory_global_dirty_log_sync()`: moves what the listeners collected into the block
    /// bitmaps, before they are read.
    fn log_sync(&self) {}

    /// Called after bits were taken out of the bitmaps, before any page is read for sending. A
    /// TCG machine flushes the TLBs here so that the next write to such a page marks it again.
    fn after_clear(&self) {}

    /// Called when an incoming migration has loaded RAM, to drop translated code and caches.
    fn load_done(&self) {}
}

/// The hooks of a machine with nothing to do.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoHooks;

impl RamHooks for NoHooks {}

/// `MigrationRAMStats`, kept with atomics so `query-migrate` can read them while the migration
/// thread works.
#[derive(Debug, Default)]
pub struct RamStats {
    /// Bytes of the `ram` sections sent.
    pub transferred: AtomicU64,
    /// Pages sent whole.
    pub normal: AtomicU64,
    /// Pages sent as zero pages.
    pub duplicate: AtomicU64,
    /// Bitmap syncs so far.
    pub dirty_sync_count: AtomicU64,
    /// Pages dirty at the last sync and not sent yet.
    pub remaining_pages: AtomicU64,
    /// The size of all blocks.
    pub total: AtomicU64,
    /// Pages found dirty by the last sync, per second since the one before.
    pub dirty_pages_rate: AtomicU64,
    /// Bytes sent while the guest was stopped.
    pub downtime_bytes: AtomicU64,
}

/// The page size migration works with, `TARGET_PAGE_SIZE` on x86.
const PAGE_BITS: u32 = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS;

enum Recv {
    Block(usize),
    Dropped,
}

/// The `ram` live entry over the RAM blocks of a machine.
pub struct RamSection {
    blocks: Vec<Arc<RamBlock>>,
    hooks: Box<dyn RamHooks>,
    stats: Arc<RamStats>,
    // Prefixes of block names that are accepted and dropped when the source has them and this
    // side does not.
    droppable: Vec<String>,
    // One bitmap per block, of pages still to send.
    pending: Vec<Vec<u64>>,
    pending_count: u64,
    // Where the scan for dirty pages goes on.
    cursor: (usize, u64),
    last_sent: Option<usize>,
    last_sync: Option<std::time::Instant>,
    logging: bool,
    last_recv: Option<Recv>,
}

impl std::fmt::Debug for RamSection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RamSection")
            .field("blocks", &self.blocks.iter().map(|b| b.name()).collect::<Vec<_>>())
            .field("pending", &self.pending_count)
            .finish()
    }
}

impl RamSection {
    /// The `ram` entry over `blocks`, in the order they are sent.
    pub fn new(blocks: Vec<Arc<RamBlock>>, hooks: impl RamHooks + 'static) -> Self {
        RamSection {
            blocks,
            hooks: Box::new(hooks),
            stats: Arc::new(RamStats::default()),
            droppable: vec!["/rom@".to_string()],
            pending: Vec::new(),
            pending_count: 0,
            cursor: (0, 0),
            last_sent: None,
            last_sync: None,
            logging: false,
            last_recv: None,
        }
    }

    /// Blocks whose name starts with one of `prefixes` are dropped when the source sends them
    /// and this machine has no such block. By default that is `/rom@`, the fw_cfg blobs QEMU
    /// keeps in RAM blocks: the guest copied what it needed out of them at boot, and this side
    /// has its own.
    pub fn droppable(mut self, prefixes: &[&str]) -> Self {
        self.droppable = prefixes.iter().map(|p| p.to_string()).collect();
        self
    }

    /// The statistics, to share with `query-migrate`.
    pub fn stats(&self) -> Arc<RamStats> {
        self.stats.clone()
    }

    fn pages(block: &RamBlock) -> u64 {
        block.len() >> PAGE_BITS
    }

    /// `migration_bitmap_sync()`: takes the bits the guest set since the last sync.
    fn sync(&mut self) {
        self.hooks.log_sync();
        let mut found = 0;
        for (block, pending) in self.blocks.iter().zip(&mut self.pending) {
            let mut words = vec![0u64; pending.len()];
            let n = block.take_dirty_words(DirtyClient::Migration, 0, &mut words);
            for (p, w) in pending.iter_mut().zip(&words[..n]) {
                found += u64::from((w & !*p).count_ones());
                *p |= w;
            }
        }
        self.hooks.after_clear();
        self.pending_count += found;
        let now = std::time::Instant::now();
        if let Some(last) = self.last_sync {
            let secs = now.duration_since(last).as_secs_f64();
            if secs > 0.0 {
                self.stats.dirty_pages_rate.store((found as f64 / secs) as u64, Ordering::Relaxed);
            }
        }
        self.last_sync = Some(now);
        self.stats.dirty_sync_count.fetch_add(1, Ordering::Relaxed);
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
    }

    /// The next page to send at or after the cursor, clearing its bit.
    fn next_dirty(&mut self) -> Option<(usize, u64)> {
        while self.cursor.0 < self.blocks.len() {
            let (b, start) = self.cursor;
            let bitmap = &mut self.pending[b];
            let npages = Self::pages(&self.blocks[b]);
            let mut word = (start / 64) as usize;
            let mut mask = !0u64 << (start % 64);
            while word < bitmap.len() {
                let bits = bitmap[word] & mask;
                if bits != 0 {
                    let page = word as u64 * 64 + u64::from(bits.trailing_zeros());
                    if page >= npages {
                        bitmap[word] = 0;
                        break;
                    }
                    bitmap[word] &= !(1 << (page % 64));
                    self.cursor = (b, page + 1);
                    self.pending_count -= 1;
                    return Some((b, page));
                }
                word += 1;
                mask = !0;
            }
            self.cursor = (b + 1, 0);
        }
        None
    }

    /// `ram_save_target_page()` with zero page detection.
    fn send_page(
        &mut self,
        f: &mut QemuFile<'_>,
        b: usize,
        page: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        let block = &self.blocks[b];
        let offset = page << PAGE_BITS;
        block.read(offset, buf).map_err(|e| {
            Error::generic(format!("Failed to read RAM block {}: {e}", block.name()))
        })?;
        let zero = buf.iter().all(|&x| x == 0);
        let mut flags = if zero { RAM_SAVE_FLAG_ZERO } else { RAM_SAVE_FLAG_PAGE };
        let same = self.last_sent == Some(b);
        if same {
            flags |= RAM_SAVE_FLAG_CONTINUE;
        }
        let before = f.transferred();
        f.put_be64(offset | flags);
        if !same {
            let name = block.name();
            f.put_byte(name.len() as u8);
            f.put_buffer(name.as_bytes());
            self.last_sent = Some(b);
        }
        if zero {
            f.put_byte(0);
            self.stats.duplicate.fetch_add(1, Ordering::Relaxed);
        } else {
            f.put_buffer(buf);
            self.stats.normal.fetch_add(1, Ordering::Relaxed);
        }
        self.stats.transferred.fetch_add(f.transferred() - before, Ordering::Relaxed);
        f.maybe_flush()
    }

    fn end_section(&mut self, f: &mut QemuFile<'_>) {
        f.put_be64(RAM_SAVE_FLAG_EOS);
        self.stats.transferred.fetch_add(8, Ordering::Relaxed);
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
    }

    fn stop_logging(&mut self) {
        if self.logging {
            for b in &self.blocks {
                b.stop_dirty_log(DirtyClient::Migration);
            }
            self.hooks.log_stop();
            self.logging = false;
        }
    }

    fn find_block(&self, name: &str) -> Option<usize> {
        self.blocks.iter().position(|b| b.name() == name)
    }

    fn droppable_name(&self, name: &str) -> bool {
        self.droppable.iter().any(|p| name.starts_with(p.as_str()))
    }

    fn read_name(f: &mut StreamReader<'_>) -> String {
        let len = usize::from(f.get_byte());
        let mut id = vec![0; len];
        f.get_buffer(&mut id);
        String::from_utf8_lossy(&id).into_owned()
    }

    /// `parse_ramblocks()`.
    fn parse_blocks(&mut self, f: &mut StreamReader<'_>, mut total: u64) -> Result<()> {
        while total != 0 {
            let id = Self::read_name(f);
            let length = f.get_be64();
            if f.get_error() != 0 {
                bail!("Failed to read the RAM block list: stream error {}", f.get_error());
            }
            match self.find_block(&id) {
                Some(i) => {
                    let block = &self.blocks[i];
                    if length != block.len() {
                        // qemu_ram_resize() on a block that cannot be resized.
                        bail!(
                            "Size mismatch: {}: 0x{:x} != 0x{:x}: Invalid argument",
                            id,
                            length,
                            block.len()
                        );
                    }
                }
                None if self.droppable_name(&id) => {
                    warn_report(&format!(
                        "RAM block \"{id}\" has no counterpart here, dropping it"
                    ));
                }
                None => bail!("Unknown ramblock \"{}\", cannot accept migration", id),
            }
            total = total.wrapping_sub(length);
        }
        Ok(())
    }

    /// `ram_block_from_stream()`.
    fn block_from_stream(&mut self, f: &mut StreamReader<'_>, flags: u64) -> Result<()> {
        if flags & RAM_SAVE_FLAG_CONTINUE != 0 {
            if self.last_recv.is_none() {
                bail!("Ack, bad migration stream!");
            }
            return Ok(());
        }
        let id = Self::read_name(f);
        self.last_recv = Some(match self.find_block(&id) {
            Some(i) => Recv::Block(i),
            None if self.droppable_name(&id) => Recv::Dropped,
            None => bail!("Can't find block {}", id),
        });
        Ok(())
    }
}

impl LiveState for RamSection {
    /// `ram_save_setup()`: starts dirty logging, marks every page to send and writes the block
    /// list.
    fn save_setup(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        let mut total = 0;
        self.pending.clear();
        self.pending_count = 0;
        for b in &self.blocks {
            if b.len() % PAGE_SIZE as u64 != 0 || b.page_bits() != PAGE_BITS {
                bail!("RAM block {} does not use {} byte pages", b.name(), PAGE_SIZE);
            }
            b.start_dirty_log(DirtyClient::Migration);
            let pages = Self::pages(b);
            let mut bitmap = vec![!0u64; pages.div_ceil(64) as usize];
            if pages % 64 != 0 {
                if let Some(last) = bitmap.last_mut() {
                    *last = (1u64 << (pages % 64)) - 1;
                }
            }
            self.pending.push(bitmap);
            self.pending_count += pages;
            total += b.len();
        }
        self.logging = true;
        self.hooks.log_start();
        // Pages the guest wrote before logging started are all marked already; this re-arms
        // write tracking for what comes after.
        self.hooks.after_clear();
        self.cursor = (0, 0);
        self.last_sent = None;
        self.last_sync = Some(std::time::Instant::now());
        self.stats.total.store(total, Ordering::Relaxed);
        self.stats.dirty_sync_count.store(1, Ordering::Relaxed);
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);

        f.put_be64(total | RAM_SAVE_FLAG_MEM_SIZE);
        for b in &self.blocks {
            let name = b.name();
            f.put_byte(name.len() as u8);
            f.put_buffer(name.as_bytes());
            f.put_be64(b.len());
        }
        self.end_section(f);
        Ok(())
    }

    /// `ram_save_iterate()`.
    fn save_iterate(&mut self, f: &mut QemuFile<'_>, max_bytes: u64) -> Result<bool> {
        let start = f.transferred();
        let mut buf = vec![0u8; PAGE_SIZE];
        // QEMU's destination keeps its last block across sections, but a fresh name costs
        // little and keeps every section self-contained.
        self.last_sent = None;
        let mut done = false;
        while f.transferred() - start < max_bytes {
            match self.next_dirty() {
                Some((b, page)) => self.send_page(f, b, page, &mut buf)?,
                None => {
                    // The round is complete; the next sync starts another.
                    self.cursor = (0, 0);
                    done = true;
                    break;
                }
            }
        }
        self.end_section(f);
        Ok(done)
    }

    /// `ram_save_complete()`: with the guest stopped, a last sync and every page left.
    fn save_complete(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        let before = f.transferred();
        self.sync();
        self.cursor = (0, 0);
        self.last_sent = None;
        let mut buf = vec![0u8; PAGE_SIZE];
        while let Some((b, page)) = self.next_dirty() {
            self.send_page(f, b, page, &mut buf)?;
        }
        self.end_section(f);
        self.stats.downtime_bytes.fetch_add(f.transferred() - before, Ordering::Relaxed);
        self.stop_logging();
        Ok(())
    }

    /// `ram_state_pending_estimate()` and `ram_state_pending_exact()`.
    fn pending(&mut self, exact: bool) -> u64 {
        if exact && self.logging {
            self.sync();
        }
        self.pending_count << PAGE_BITS
    }

    fn save_cleanup(&mut self) {
        self.stop_logging();
        self.pending.clear();
        self.pending_count = 0;
    }

    fn load_setup(&mut self) -> Result<()> {
        self.last_recv = None;
        Ok(())
    }

    /// `ram_load_precopy()`.
    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()> {
        if version_id != RAM_SECTION_VERSION {
            bail!("Unsupported RAM section version {}", version_id);
        }
        let mut buf = vec![0u8; PAGE_SIZE];
        loop {
            let word = f.get_be64();
            if f.get_error() != 0 {
                bail!("Getting RAM address failed");
            }
            let flags = word & (PAGE_SIZE as u64 - 1);
            let addr = word & !(PAGE_SIZE as u64 - 1);

            let mut target = None;
            if flags & (RAM_SAVE_FLAG_ZERO | RAM_SAVE_FLAG_PAGE | RAM_SAVE_FLAG_XBZRLE) != 0 {
                self.block_from_stream(f, flags)?;
                if let Some(Recv::Block(i)) = self.last_recv {
                    if addr + PAGE_SIZE as u64 > self.blocks[i].len() {
                        bail!("Illegal RAM offset {:x}", addr);
                    }
                    target = Some(i);
                }
            }

            match flags & !RAM_SAVE_FLAG_CONTINUE {
                RAM_SAVE_FLAG_MEM_SIZE => self.parse_blocks(f, addr)?,
                RAM_SAVE_FLAG_ZERO => {
                    let ch = f.get_byte();
                    if ch != 0 {
                        bail!("Found a zero page with value {}", ch);
                    }
                    if let Some(i) = target {
                        // ram_handle_zero(): a page that is already zero is left alone, so the
                        // host does not have to back it.
                        let block = &self.blocks[i];
                        block.read(addr, &mut buf).map_err(|e| {
                            Error::generic(format!("Illegal RAM offset {addr:x}: {e}"))
                        })?;
                        if buf.iter().any(|&b| b != 0) {
                            block.fill(addr, PAGE_SIZE as u64, 0).map_err(|e| {
                                Error::generic(format!("Illegal RAM offset {addr:x}: {e}"))
                            })?;
                        }
                    }
                }
                RAM_SAVE_FLAG_PAGE => {
                    f.get_buffer(&mut buf);
                    if let Some(i) = target {
                        self.blocks[i].write(addr, &buf).map_err(|e| {
                            Error::generic(format!("Illegal RAM offset {addr:x}: {e}"))
                        })?;
                    }
                }
                RAM_SAVE_FLAG_XBZRLE => bail!("Failed to decompress XBZRLE page at {:x}", addr),
                // Without multifd there is nothing to wait for.
                RAM_SAVE_FLAG_MULTIFD_FLUSH => {}
                RAM_SAVE_FLAG_EOS => break,
                RAM_SAVE_FLAG_HOOK => bail!("RDMA is not supported"),
                _ => bail!("Unknown combination of migration flags: 0x{:x}", flags),
            }
            let ret = f.get_error();
            if ret != 0 {
                bail!("RAM load failed: stream error {}", ret);
            }
        }
        Ok(())
    }

    fn load_cleanup(&mut self) {
        self.last_recv = None;
        self.hooks.load_done();
    }
}

impl Drop for RamSection {
    fn drop(&mut self) {
        self.stop_logging();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::savevm::{EntryInfo, MachineConfig, SaveVm};

    fn config() -> MachineConfig {
        MachineConfig {
            name: "pc-q35-11.1".into(),
            page_bits: 12,
            legacy_page_bits: 12,
            uuid: None,
        }
    }

    fn vm(blocks: Vec<Arc<RamBlock>>) -> SaveVm {
        let mut s = SaveVm::new(config());
        s.register_live(
            EntryInfo::new("ram", RAM_SECTION_VERSION),
            RamSection::new(blocks, NoHooks),
        );
        s
    }

    fn block(name: &str, pages: u64) -> Arc<RamBlock> {
        Arc::new(RamBlock::new(name, pages << 12, 12).unwrap())
    }

    #[test]
    fn precopy_round_trip_with_dirty_pages() {
        let src = block("pc.ram", 200);
        src.write(0x1000, b"hello").unwrap();
        src.write(199 << 12, &[0xaa; 4096]).unwrap();
        let mut s = vm(vec![src.clone()]);
        let mut out = Vec::new();
        {
            let mut f = QemuFile::new(&mut out);
            s.save_header(&mut f).unwrap();
            s.save_setup(&mut f).unwrap();
            // A small budget takes several passes over the first round.
            let mut rounds = 0;
            while !s.save_iterate(&mut f, 4 << 10).unwrap() {
                rounds += 1;
            }
            assert!(rounds >= 1);
            assert_eq!(s.pending(false), 0);
            // The guest writes after its pages went out.
            src.write(0x1000, b"world").unwrap();
            src.set_dirty(0x1000, 5, DirtyClient::Migration.mask());
            src.write(0x5000, &[1; 16]).unwrap();
            src.set_dirty(0x5000, 16, DirtyClient::Migration.mask());
            assert_eq!(s.pending(true), 2 << 12);
            s.save_complete(&mut f).unwrap();
        }
        assert!(!src.is_dirty_logging(DirtyClient::Migration));

        let dst = block("pc.ram", 200);
        dst.write(0x9000, &[7; 8]).unwrap();
        vm(vec![dst.clone()]).load_state(&mut StreamReader::new(&out)).unwrap();
        let (mut a, mut b) = (vec![0; 200 << 12], vec![0; 200 << 12]);
        src.read(0, &mut a).unwrap();
        dst.read(0, &mut b).unwrap();
        assert!(a == b);
        assert_eq!(&b[0x1000..0x1005], b"world");
    }

    #[test]
    fn unknown_blocks() {
        let src = vec![block("pc.ram", 4), block("/rom@etc/acpi/tables", 2)];
        src[1].write(0, &[5; 64]).unwrap();
        let mut out = Vec::new();
        vm(src).save_state(&mut QemuFile::new(&mut out)).unwrap();

        // The fw_cfg blob is dropped.
        let dst = block("pc.ram", 4);
        vm(vec![dst]).load_state(&mut StreamReader::new(&out)).unwrap();

        // Anything else is an error, as in QEMU.
        let src = vec![block("pc.ram", 4), block("vga.vram", 2)];
        let mut out = Vec::new();
        vm(src).save_state(&mut QemuFile::new(&mut out)).unwrap();
        let e = vm(vec![block("pc.ram", 4)]).load_state(&mut StreamReader::new(&out)).unwrap_err();
        assert!(
            e.message().ends_with("Unknown ramblock \"vga.vram\", cannot accept migration"),
            "{}",
            e.message()
        );

        let mut out = Vec::new();
        vm(vec![block("pc.ram", 4)]).save_state(&mut QemuFile::new(&mut out)).unwrap();
        let e = vm(vec![block("pc.ram", 8)]).load_state(&mut StreamReader::new(&out)).unwrap_err();
        assert!(
            e.message().ends_with("Size mismatch: pc.ram: 0x4000 != 0x8000: Invalid argument"),
            "{}",
            e.message()
        );
    }
}
