// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ram` section, version 4, migration/ram.c.
//!
//! The `QEMU_VM_SECTION_START` payload lists the RAM blocks: the total size with
//! `RAM_SAVE_FLAG_MEM_SIZE`, then each block's name and length. Every section after that is a run
//! of pages, each a big-endian word of the page offset and its flags, the block name unless
//! `RAM_SAVE_FLAG_CONTINUE` says it is the block of the page before, and either one zero byte
//! (`RAM_SAVE_FLAG_ZERO`), the whole page (`RAM_SAVE_FLAG_PAGE`) or its XBZRLE encoding
//! (`RAM_SAVE_FLAG_XBZRLE`). `RAM_SAVE_FLAG_EOS` ends each section.
//!
//! Precopy sends every page once, then keeps sending the pages the guest wrote since, which the
//! `DIRTY_MEMORY_MIGRATION` bitmaps of the RAM blocks record. With the `xbzrle` capability the
//! pages of the later rounds go out as the difference to the copy sent before, when that copy
//! is still in the cache.
//!
//! With `mapped-ram` the pages do not go in the stream: each block has a fixed region of the file
//! after its entry in the block list, the pages are written there and the bitmap of those that
//! are not zero is written at the end; the destination reads them back in the block list. See
//! [`crate::mapped_ram`].
//!
//! In postcopy the source tells the destination to drop the pages it still has to send, then
//! sends them once more each, the ones the destination asks for first. The destination fills
//! them in with userfaultfd, which wakes whoever touched such a page before it came.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;

use ruvm_base::{Error, Result, bail, error_report, warn_report};
use ruvm_mem::{DirtyClient, RamBlock, buffer_is_zero};
use ruvm_qapi::types::ZeroPageDetection;
use ruvm_vmstate::StreamReader;

use crate::channel::FileChannel;
use crate::mapped_ram::{self, FileBlock, HDR_LEN, HDR_VERSION, Header, LOAD_BUF_SIZE};
use crate::multifd::{MultifdRecv, MultifdSend};
use crate::postcopy::uffd::Uffd;
use crate::postcopy::{PageRequests, ReturnPath};
use crate::savevm::{LiveState, LoadParams, QemuFile, SaveParams};
use crate::write_tracking::WriteTracking;
use crate::xbzrle::{self, ENCODING_FLAG_XBZRLE, Encoded, PageCache, XbzrleCounters};

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
    /// The iteration the migration is in, from 1, `mig_stats.dirty_sync_count`. Also the
    /// generation of the XBZRLE cache.
    pub dirty_sync_count: AtomicU64,
    /// Pages dirty at the last sync and not sent yet.
    pub remaining_pages: AtomicU64,
    /// The size of all blocks.
    pub total: AtomicU64,
    /// Pages found dirty by the last sync, per second since the one before.
    pub dirty_pages_rate: AtomicU64,
    /// Bytes sent while the guest ran.
    pub precopy_bytes: AtomicU64,
    /// Bytes sent while the guest was stopped, outside postcopy.
    pub downtime_bytes: AtomicU64,
    /// Bytes sent in postcopy, after the destination started the guest.
    pub postcopy_bytes: AtomicU64,
    /// Bytes sent on the multifd channels.
    pub multifd_bytes: AtomicU64,
    /// The XBZRLE counters, which unlike the others are never reset.
    pub xbzrle: XbzrleCounters,
    /// Whether the guest runs, `runstate_is_running()`, which decides where the bytes sent
    /// count. The migration thread keeps it up to date.
    pub guest_running: AtomicBool,
}

impl RamStats {
    /// The part of `migrate_init()` that clears `mig_stats` for a new migration.
    pub fn reset(&self) {
        for c in [
            &self.transferred,
            &self.normal,
            &self.duplicate,
            &self.dirty_sync_count,
            &self.remaining_pages,
            &self.dirty_pages_rate,
            &self.precopy_bytes,
            &self.downtime_bytes,
            &self.postcopy_bytes,
            &self.multifd_bytes,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// The page size migration works with, `TARGET_PAGE_SIZE` on x86.
const PAGE_BITS: u32 = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS;
/// `MAX_WAIT`: how long one iteration may send for.
const MAX_WAIT: std::time::Duration = std::time::Duration::from_millis(50);

enum Recv {
    Block(usize),
    Dropped,
}

/// The XBZRLE state of an outgoing migration, `XBZRLE` in ram.c.
struct XbzrleSave {
    // The size migrate-set-parameters sets, which the cache follows.
    size: Arc<AtomicU64>,
    cache: PageCache,
    encoded: Vec<u8>,
    // `xbzrle_started`: pages are only encoded from the second round on.
    started: bool,
}

/// The counters at the start of the rate period, for `migration_update_rates()`.
struct RatePeriod {
    start: Option<std::time::Instant>,
    pages: u64,
    cache_miss: u64,
    xbzrle_pages: u64,
    xbzrle_bytes: u64,
}

/// `RAMBlock.receivedmap` of every block: the pages the destination has, which the fault thread
/// does not ask for again.
#[derive(Debug)]
struct RecvMap(Vec<Vec<AtomicU64>>);

impl RecvMap {
    fn new(blocks: &[Arc<RamBlock>]) -> Self {
        RecvMap(
            blocks
                .iter()
                .map(|b| {
                    (0..RamSection::pages(b).div_ceil(64)).map(|_| AtomicU64::new(0)).collect()
                })
                .collect(),
        )
    }

    fn set(&self, b: usize, page: u64) {
        self.0[b][(page / 64) as usize].fetch_or(1 << (page % 64), Ordering::Release);
    }

    fn test(&self, b: usize, page: u64) -> bool {
        self.0[b][(page / 64) as usize].load(Ordering::Acquire) & (1 << (page % 64)) != 0
    }

    fn clear(&self, b: usize, first: u64, end: u64) {
        for page in first..end {
            self.0[b][(page / 64) as usize].fetch_and(!(1 << (page % 64)), Ordering::Release);
        }
    }
}

/// The destination between `MIG_CMD_POSTCOPY_LISTEN` and the end of the stream.
struct Listening {
    uffd: Arc<Uffd>,
    quit: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// `postcopy_ram_fault_thread()`: asks the source for every page someone touches before it is
/// here.
fn fault_thread(
    uffd: &Uffd,
    blocks: &[Arc<RamBlock>],
    recv: &RecvMap,
    rp: &ReturnPath,
    quit: &AtomicBool,
) {
    let mut last = None;
    while !quit.load(Ordering::Acquire) {
        let addr = match uffd.wait_fault(100) {
            Ok(Some(addr)) => addr,
            Ok(None) => continue,
            Err(e) => {
                error_report(&format!("postcopy_ram_fault_thread: userfault poll: {e}"));
                break;
            }
        };
        let Some(b) = blocks
            .iter()
            .position(|b| addr >= b.host_addr() && addr - b.host_addr() < b.len() as usize)
        else {
            error_report(&format!("postcopy_ram_fault_thread: Fault outside guest: {addr:x}"));
            break;
        };
        let offset = ((addr - blocks[b].host_addr()) & !(PAGE_SIZE - 1)) as u64;
        // migrate_send_rp_req_pages(): the page may have come since the fault.
        if recv.test(b, offset >> PAGE_BITS) {
            continue;
        }
        let name = (last != Some(b)).then(|| blocks[b].name());
        if let Err(e) = rp.req_pages(name, offset, PAGE_SIZE as u32) {
            error_report(e.message());
            break;
        }
        last = Some(b);
    }
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
    // The pages the destination asked for, which go out first in postcopy.
    requests: Arc<PageRequests>,
    in_postcopy: bool,
    // Destination: what came, once postcopy was advised.
    recv_map: Option<Arc<RecvMap>>,
    listening: Option<Listening>,
    multifd_send: Option<Arc<MultifdSend>>,
    multifd_recv: Option<Arc<MultifdRecv>>,
    zero_page_detection: ZeroPageDetection,
    xbzrle_size: Option<Arc<AtomicU64>>,
    xbzrle: Option<XbzrleSave>,
    // `last_stage`: the guest is stopped and these are the last pages.
    last_stage: bool,
    // The `mapped-ram` capability of the outgoing migration, and where each block goes in the
    // file.
    mapped_ram: bool,
    file_blocks: Vec<Arc<FileBlock>>,
    // The `background-snapshot` capability: one pass without dirty logging, and the write
    // protection of the pages not sent yet once the guest runs again.
    background: bool,
    // Every block is ignored (`cpr-transfer`): listed, but no page goes out.
    ignore: bool,
    tracking: Option<WriteTracking>,
    // The same for a load: the pages are in the file the stream comes from.
    load_mapped_ram: bool,
    load_file: Option<Arc<FileChannel>>,
    // Where each block starts in one address space over all of them, like `RAMBlock.offset`,
    // for the XBZRLE cache.
    block_base: Vec<u64>,
    // `target_page_count`: the pages sent, for the rates.
    page_count: u64,
    rates: RatePeriod,
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
        let requests =
            PageRequests::new(blocks.iter().map(|b| (b.name().to_string(), b.len())).collect());
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
            requests: Arc::new(requests),
            in_postcopy: false,
            recv_map: None,
            listening: None,
            multifd_send: None,
            multifd_recv: None,
            zero_page_detection: ZeroPageDetection::Multifd,
            xbzrle_size: None,
            xbzrle: None,
            last_stage: false,
            mapped_ram: false,
            file_blocks: Vec::new(),
            background: false,
            ignore: false,
            tracking: None,
            load_mapped_ram: false,
            load_file: None,
            block_base: Vec::new(),
            page_count: 0,
            rates: RatePeriod {
                start: None,
                pages: 0,
                cache_miss: 0,
                xbzrle_pages: 0,
                xbzrle_bytes: 0,
            },
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
        self.update_rates(now);
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
    }

    /// The rate period of `migration_bitmap_sync()`, a second or more, and
    /// `migration_update_rates()` for XBZRLE at its end.
    fn update_rates(&mut self, now: std::time::Instant) {
        let r = &mut self.rates;
        let start = *r.start.get_or_insert(now);
        if now.duration_since(start) <= std::time::Duration::from_secs(1) {
            return;
        }
        let pages = self.page_count - r.pages;
        if pages != 0 && self.xbzrle.is_some() {
            let c = &self.stats.xbzrle;
            let miss = c.cache_miss.load(Ordering::Relaxed);
            let xpages = c.pages.load(Ordering::Relaxed);
            let xbytes = c.bytes.load(Ordering::Relaxed);
            let miss_rate = (miss - r.cache_miss) as f64 / pages as f64;
            let unencoded = ((xpages - r.xbzrle_pages) * PAGE_SIZE as u64) as f64;
            let encoded = (xbytes - r.xbzrle_bytes) as f64;
            let encoding_rate =
                if xpages == r.xbzrle_pages || encoded == 0.0 { 0.0 } else { unencoded / encoded };
            c.set_rates(miss_rate, encoding_rate);
            r.cache_miss = miss;
            r.xbzrle_pages = xpages;
            r.xbzrle_bytes = xbytes;
        }
        r.pages = self.page_count;
        r.start = Some(now);
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
                    // ram_bytes_remaining() goes down page by page.
                    self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
                    return Some((b, page));
                }
                word += 1;
                mask = !0;
            }
            self.cursor = (b + 1, 0);
        }
        None
    }

    /// `ram_save_target_page()`: a zero page goes out as one byte on the main stream unless the
    /// multifd threads look for them or detection is off, and with multifd every other page goes
    /// to the channels until postcopy starts. Without multifd, `ram_save_page()` tries XBZRLE
    /// before it sends the whole page.
    fn send_page(
        &mut self,
        f: &mut QemuFile<'_>,
        b: usize,
        page: u64,
        buf: &mut [u8],
    ) -> Result<()> {
        let block = &self.blocks[b];
        let offset = page << PAGE_BITS;
        let detect = self.zero_page_detection;
        let multifd = self.multifd_send.as_ref().filter(|_| !self.in_postcopy);
        let check_zero = detect != ZeroPageDetection::None
            && (self.multifd_send.is_none() || detect == ZeroPageDetection::Legacy);
        let fb = self.file_blocks.get(b);
        if let Some(m) = multifd {
            if !check_zero {
                self.page_count += 1;
                return m.queue_page(block, fb, offset);
            }
        }
        block.read(offset, buf).map_err(|e| {
            Error::generic(format!("Failed to read RAM block {}: {e}", block.name()))
        })?;
        let zero = check_zero && buffer_is_zero(buf);
        if let Some(m) = multifd {
            if !zero {
                self.page_count += 1;
                return m.queue_page(block, fb, offset);
            }
        }
        if let Some(fb) = fb {
            // Zero pages are not written with mapped-ram; the bitmap leaves them out.
            if zero {
                fb.set(page, false);
                self.stats.duplicate.fetch_add(1, Ordering::Relaxed);
            } else {
                f.put_buffer_at(buf, fb.pages_offset + offset)?;
                fb.set(page, true);
                self.stats.normal.fetch_add(1, Ordering::Relaxed);
                self.account(PAGE_SIZE as u64);
            }
            self.page_count += 1;
            return Ok(());
        }
        if zero {
            let len = self.page_header(f, b, offset | RAM_SAVE_FLAG_ZERO);
            f.put_byte(0);
            self.stats.duplicate.fetch_add(1, Ordering::Relaxed);
            self.page_count += 1;
            self.account(len + 1);
            // xbzrle_cache_zero_page(): a stale copy in the cache would be wrong now.
            let age = self.stats.dirty_sync_count.load(Ordering::Relaxed);
            let addr = self.block_base.get(b).map_or(0, |base| base + offset);
            if let Some(x) = self.xbzrle.as_mut().filter(|x| x.started) {
                x.cache.insert(addr, &[0; PAGE_SIZE], age);
            }
        } else if !self.save_xbzrle_page(f, b, offset, buf) {
            let len = self.page_header(f, b, offset | RAM_SAVE_FLAG_PAGE);
            f.put_buffer(buf);
            self.stats.normal.fetch_add(1, Ordering::Relaxed);
            self.page_count += 1;
            self.account(len + PAGE_SIZE as u64);
        }
        f.maybe_flush()
    }

    /// `save_page_header()`: the page offset with its flags and, unless the page is in the
    /// block of the page before, the block name. Returns the bytes written.
    fn page_header(&mut self, f: &mut QemuFile<'_>, b: usize, word: u64) -> u64 {
        let same = self.last_sent == Some(b);
        f.put_be64(if same { word | RAM_SAVE_FLAG_CONTINUE } else { word });
        if same {
            return 8;
        }
        let name = self.blocks[b].name();
        f.put_byte(name.len() as u8);
        f.put_buffer(name.as_bytes());
        self.last_sent = Some(b);
        8 + 1 + name.len() as u64
    }

    /// `save_xbzrle_page()`, once XBZRLE started and outside postcopy: sends the page as the
    /// difference to its cached copy. Returns false when the whole page has to go instead,
    /// because it was not in the cache or its encoding is too long; a page that did not change
    /// since its last copy is not sent at all.
    fn save_xbzrle_page(
        &mut self,
        f: &mut QemuFile<'_>,
        b: usize,
        offset: u64,
        buf: &[u8],
    ) -> bool {
        if self.in_postcopy {
            return false;
        }
        let Some(x) = self.xbzrle.as_mut().filter(|x| x.started) else { return false };
        let c = &self.stats.xbzrle;
        let addr = self.block_base[b] + offset;
        let age = self.stats.dirty_sync_count.load(Ordering::Relaxed);
        if !x.cache.is_cached(addr, age) {
            c.cache_miss.fetch_add(1, Ordering::Relaxed);
            if !self.last_stage {
                x.cache.insert(addr, buf, age);
            }
            return false;
        }
        // A page found in the cache counts as encoded whatever comes of it, so that the encoding
        // rate shows skipped pages too.
        c.pages.fetch_add(1, Ordering::Relaxed);
        let Some(prev) = x.cache.get(addr) else { return false };
        let encoded = xbzrle::encode_buffer(prev, buf, &mut x.encoded);
        // The cache holds what the destination has, except for a page left out.
        if !self.last_stage && encoded != Encoded::Unchanged {
            prev.copy_from_slice(buf);
        }
        let len = match encoded {
            Encoded::Unchanged => return true,
            Encoded::Overflow => {
                c.overflow.fetch_add(1, Ordering::Relaxed);
                c.bytes.fetch_add(PAGE_SIZE as u64, Ordering::Relaxed);
                return false;
            }
            Encoded::Len(len) => len,
        };
        let data = std::mem::take(&mut x.encoded);
        let mut bytes = self.page_header(f, b, offset | RAM_SAVE_FLAG_XBZRLE);
        f.put_byte(ENCODING_FLAG_XBZRLE);
        f.put_be16(len as u16);
        f.put_buffer(&data[..len]);
        bytes += len as u64 + 1 + 2;
        if let Some(x) = self.xbzrle.as_mut() {
            x.encoded = data;
        }
        // The XBZRLE bytes leave out the 8 byte page word.
        self.stats.xbzrle.bytes.fetch_add(bytes - 8, Ordering::Relaxed);
        self.page_count += 1;
        self.account(bytes);
        true
    }

    /// `xbzrle_cache_resize()`: a new and empty cache when `xbzrle-cache-size` changed.
    fn xbzrle_resize(&mut self) -> Result<()> {
        if let Some(x) = self.xbzrle.as_mut() {
            let size = x.size.load(Ordering::Relaxed);
            if size != x.cache.size() {
                x.cache = PageCache::new(size, PAGE_SIZE)?;
            }
        }
        Ok(())
    }

    /// `ram_transferred_add()`.
    fn account(&self, bytes: u64) {
        self.stats.transferred.fetch_add(bytes, Ordering::Relaxed);
        let to = if self.stats.guest_running.load(Ordering::Relaxed) {
            &self.stats.precopy_bytes
        } else if self.in_postcopy {
            &self.stats.postcopy_bytes
        } else {
            &self.stats.downtime_bytes
        };
        to.fetch_add(bytes, Ordering::Relaxed);
    }

    /// `multifd_ram_flush_and_sync()`: every page queued so far is out on the channels, and the
    /// destination waits for them on `RAM_SAVE_FLAG_MULTIFD_FLUSH`. With mapped-ram the
    /// destination reads the pages from the file once, so only the threads here are waited for.
    fn multifd_sync(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        let Some(m) = self.multifd_send.as_ref().filter(|_| !self.in_postcopy) else {
            return Ok(());
        };
        m.flush_and_sync()?;
        if self.mapped_ram {
            return Ok(());
        }
        f.put_be64(RAM_SAVE_FLAG_MULTIFD_FLUSH);
        self.account(8);
        f.fflush()
    }

    /// `migration_transferred_bytes()`, which leaves out what is still buffered.
    fn flushed(&self, f: &QemuFile<'_>) -> u64 {
        f.flushed() + self.multifd_send.as_ref().map_or(0, |m| m.bytes())
    }

    /// Ends a `ram` section. Only the end of an iteration counts as RAM sent, as in QEMU.
    fn end_section(&mut self, f: &mut QemuFile<'_>, counted: bool) {
        f.put_be64(RAM_SAVE_FLAG_EOS);
        if counted {
            self.account(8);
        }
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
    }

    /// Clears the pending bit of a page, saying whether it was set.
    fn take_page(&mut self, b: usize, page: u64) -> bool {
        let Some(word) = self.pending.get_mut(b).and_then(|p| p.get_mut((page / 64) as usize))
        else {
            return false;
        };
        let bit = 1 << (page % 64);
        if *word & bit == 0 {
            return false;
        }
        *word &= !bit;
        self.pending_count -= 1;
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
        true
    }

    /// `get_queued_page()`: sends the pages of the oldest request that still have to go.
    /// Returns false when there was no request.
    fn send_requested(&mut self, f: &mut QemuFile<'_>, buf: &mut [u8]) -> Result<bool> {
        let Some((b, start, len)) = self.requests.pop() else { return Ok(false) };
        let mut sent = false;
        for page in (start >> PAGE_BITS)..((start + len) >> PAGE_BITS) {
            if self.take_page(b, page) {
                self.send_page(f, b, page, buf)?;
                sent = true;
            }
        }
        // Someone on the destination is waiting for these.
        if sent {
            f.fflush()?;
        }
        Ok(true)
    }

    /// `poll_fault_page()` in `get_queued_page()`: with background snapshots, the page a guest
    /// write waits for goes out first. Returns false when nobody waits.
    fn send_faulted(&mut self, f: &mut QemuFile<'_>, buf: &mut [u8]) -> Result<bool> {
        let Some(t) = &self.tracking else { return Ok(false) };
        let fault = t.poll_fault().map_err(|e| Error::from_io("Failed to read write fault", e))?;
        let Some((b, offset)) = fault else { return Ok(false) };
        let page = offset >> PAGE_BITS;
        if self.take_page(b, page) {
            self.send_page(f, b, page, buf)?;
        }
        self.release_protection(b, page)?;
        Ok(true)
    }

    /// `ram_save_release_protection()`: the page went out, so the guest may write it now. The
    /// stream holds a copy of it, so unlike QEMU nothing has to be flushed first.
    fn release_protection(&self, b: usize, page: u64) -> Result<()> {
        let Some(t) = &self.tracking else { return Ok(()) };
        t.release(b, page << PAGE_BITS, PAGE_SIZE as u64)
            .map_err(|e| Error::from_io("Failed to release write protection", e))
    }

    /// The pending pages of each block as byte ranges.
    fn pending_ranges(&self) -> Vec<(String, Vec<(u64, u64)>)> {
        let mut out = Vec::new();
        for (block, bitmap) in self.blocks.iter().zip(&self.pending) {
            let npages = Self::pages(block);
            let mut ranges = Vec::new();
            let mut run = None;
            for page in 0..npages {
                let set = bitmap[(page / 64) as usize] & (1 << (page % 64)) != 0;
                match (set, run) {
                    (true, None) => run = Some(page),
                    (false, Some(s)) => {
                        ranges.push((s << PAGE_BITS, (page - s) << PAGE_BITS));
                        run = None;
                    }
                    _ => {}
                }
            }
            if let Some(s) = run {
                ranges.push((s << PAGE_BITS, (npages - s) << PAGE_BITS));
            }
            if !ranges.is_empty() {
                out.push((block.name().to_string(), ranges));
            }
        }
        out
    }

    /// `postcopy_place_page()` and `postcopy_place_page_zero()`.
    fn place(&self, i: usize, addr: u64, data: Option<&[u8]>) -> Result<()> {
        let Some(l) = &self.listening else { return Ok(()) };
        let mem = self.blocks[i].host_memory();
        let ret = match data {
            Some(d) => l.uffd.copy(mem, addr as usize, d),
            None => l.uffd.zeropage(mem, addr as usize, PAGE_SIZE),
        };
        if let Err(e) = ret {
            let host = mem.host_addr() + addr as usize;
            let errno = e.raw_os_error().unwrap_or(0);
            match data {
                Some(d) => error_report(&format!(
                    "uffd_copy_page() failed: dst_addr={host:#x} src_addr={:p} length={PAGE_SIZE} \
                     mode=0 errno={errno}",
                    d.as_ptr()
                )),
                None => error_report(&format!(
                    "uffd_zero_page() failed: addr={host:#x} length={PAGE_SIZE} mode=0 \
                     errno={errno}"
                )),
            }
            bail!(
                "Failed to place postcopy page {:x} of {}: {}",
                addr,
                self.blocks[i].name(),
                -errno
            );
        }
        Ok(())
    }

    fn mark_received(&self, i: usize, addr: u64) {
        if let Some(r) = &self.recv_map {
            r.set(i, addr >> PAGE_BITS);
        }
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

    /// `load_xbzrle()`: applies an encoded page to the page in `block`, which is None for a
    /// block that is dropped.
    fn load_xbzrle(
        f: &mut StreamReader<'_>,
        block: Option<&Arc<RamBlock>>,
        addr: u64,
        buf: &mut [u8],
    ) -> bool {
        let flags = f.get_byte();
        let len = usize::from(f.get_be16());
        if flags != ENCODING_FLAG_XBZRLE {
            error_report("Failed to load XBZRLE page - wrong compression!");
            return false;
        }
        if len > PAGE_SIZE {
            error_report("Failed to load XBZRLE page - len overflow!");
            return false;
        }
        let mut data = [0u8; PAGE_SIZE];
        f.get_buffer(&mut data[..len]);
        let Some(block) = block else { return true };
        if f.get_error() != 0 || block.read(addr, buf).is_err() {
            return true;
        }
        if xbzrle::decode_buffer(&data[..len], buf).is_none() {
            error_report("Failed to load XBZRLE page - decode error!");
            return false;
        }
        if let Err(e) = block.write(addr, buf) {
            error_report(&format!("Illegal RAM offset {addr:x}: {e}"));
            return false;
        }
        true
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
            let target = match self.find_block(&id) {
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
                    Some(i)
                }
                None if self.droppable_name(&id) => {
                    warn_report(&format!(
                        "RAM block \"{id}\" has no counterpart here, dropping it"
                    ));
                    None
                }
                None => bail!("Unknown ramblock \"{}\", cannot accept migration", id),
            };
            if self.load_mapped_ram {
                self.parse_mapped_ram(f, target, length)?;
            }
            total = total.wrapping_sub(length);
        }
        Ok(())
    }

    /// `parse_ramblock_mapped_ram()`: the header after the block in the list, the bitmap, the
    /// pages it marks, and the stream goes on after the region of the block. The pages of a
    /// block this side drops are skipped.
    fn parse_mapped_ram(
        &mut self,
        f: &mut StreamReader<'_>,
        target: Option<usize>,
        length: u64,
    ) -> Result<()> {
        let Some(fc) = self.load_file.clone() else {
            bail!("Migration requires seekable transport (e.g. file)");
        };
        // mapped_ram_read_header()
        let mut hb = [0u8; HDR_LEN];
        let got = f.get_buffer(&mut hb);
        if got != HDR_LEN {
            bail!(
                "Could not read whole mapped-ram migration header (expected {}, got {} bytes)",
                HDR_LEN,
                got
            );
        }
        let h = Header::parse(&hb);
        if h.version > HDR_VERSION {
            bail!(
                "Migration mapped-ram capability version not supported (expected <= {}, got {})",
                HDR_VERSION,
                h.version as i32
            );
        }
        if h.page_size != PAGE_SIZE as u64 {
            bail!(
                "Migration mapped-ram header has invalid page_size {} (expected {})",
                h.page_size,
                PAGE_SIZE
            );
        }
        if let Some(i) = target {
            let block = self.blocks[i].clone();
            if h.pages_offset % PAGE_SIZE as u64 != 0 {
                bail!("Error reading ramblock {} pages, region has bad alignment", block.name());
            }
            let pages = length / h.page_size;
            let mut raw = vec![0u8; mapped_ram::bitmap_size(pages) as usize];
            if fc.read_at(&mut raw, h.bitmap_offset).is_err() {
                f.set_error(-ruvm_vmstate::EIO);
                bail!("Error reading dirty bitmap");
            }
            let bitmap = mapped_ram::bitmap_from_bytes(&raw, pages);
            self.read_mapped_pages(f, &fc, &block, h.pages_offset, &bitmap, pages)?;
        }
        // Skip the pages.
        Self::reader_set_offset(f, &fc, h.pages_offset + length)
    }

    /// `read_ramblock_mapped_ram()`: each run of pages in the bitmap, in pieces of
    /// `MAPPED_RAM_LOAD_BUF_SIZE`, read here or by the multifd channels. The pages left out
    /// stay as they are, since an incoming machine has nothing in them yet.
    fn read_mapped_pages(
        &self,
        f: &mut StreamReader<'_>,
        fc: &FileChannel,
        block: &Arc<RamBlock>,
        pages_offset: u64,
        bitmap: &[u64],
        pages: u64,
    ) -> Result<()> {
        let mut buf = Vec::new();
        for (first, end) in mapped_ram::runs(bitmap, pages) {
            let mut offset = first << PAGE_BITS;
            let mut unread = (end - first) << PAGE_BITS;
            while unread > 0 {
                if offset >= block.len() {
                    bail!("page outside of ramblock {} range", block.name());
                }
                let size = unread.min(LOAD_BUF_SIZE as u64) as usize;
                let at = pages_offset + offset;
                let res = match &self.multifd_recv {
                    Some(m) => m.recv_file(block, offset, at, size),
                    None => {
                        buf.resize(size, 0);
                        fc.read_at(&mut buf, at).inspect_err(|_| f.set_error(-ruvm_vmstate::EIO))
                    }
                };
                res.map_err(|e| {
                    e.prepend(format!(
                        "({}) failed to read page {offset:x}from file offset {at:x}: ",
                        block.name()
                    ))
                })?;
                if self.multifd_recv.is_none() {
                    block.write(offset, &buf).map_err(|e| {
                        Error::generic(format!("Illegal RAM offset {offset:x}: {e}"))
                    })?;
                }
                offset += size as u64;
                unread -= size as u64;
            }
        }
        Ok(())
    }

    /// `qemu_set_offset()` on the incoming stream: what is buffered goes, and reading goes on
    /// at `target` in the file. A target within the buffer just skips to it.
    fn reader_set_offset(f: &mut StreamReader<'_>, fc: &FileChannel, target: u64) -> Result<()> {
        let buffered = f.remaining().len() as u64;
        let at = fc.offset().inspect_err(|_| f.set_error(-ruvm_vmstate::EIO))?;
        let cur = at.saturating_sub(buffered);
        if target >= cur && target - cur <= buffered {
            f.skip((target - cur) as usize);
            return Ok(());
        }
        f.skip(buffered as usize);
        fc.set_offset(target).inspect_err(|_| f.set_error(-ruvm_vmstate::EIO))
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
        // ram_bytes_total_with_ignored(), for the header.
        let mut listed = 0;
        self.pending.clear();
        self.pending_count = 0;
        self.requests.reset();
        self.in_postcopy = false;
        self.last_stage = false;
        self.block_base.clear();
        for b in &self.blocks {
            self.block_base.push(total);
            if b.len() % PAGE_SIZE as u64 != 0 || b.page_bits() != PAGE_BITS {
                bail!("RAM block {} does not use {} byte pages", b.name(), PAGE_SIZE);
            }
            let pages = Self::pages(b);
            listed += b.len();
            if self.ignore {
                // ram_init_bitmaps() leaves ignored blocks out: nothing to send or to log.
                self.pending.push(vec![0; pages.div_ceil(64) as usize]);
                continue;
            }
            // There is no dirty log with background snapshots.
            if !self.background {
                b.start_dirty_log(DirtyClient::Migration);
            }
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
        if !self.background && !self.ignore {
            self.logging = true;
            self.hooks.log_start();
            // The first sync of ram_init_bitmaps(): every page goes out anyway, so the bits
            // the guest set since boot are dropped rather than sending those pages twice.
            for b in &self.blocks {
                let mut words = vec![0u64; Self::pages(b).div_ceil(64) as usize];
                b.take_dirty_words(DirtyClient::Migration, 0, &mut words);
            }
            // This re-arms write tracking for what comes after.
            self.hooks.after_clear();
        }
        self.cursor = (0, 0);
        self.last_sent = None;
        self.last_sync = Some(std::time::Instant::now());
        self.stats.total.store(total, Ordering::Relaxed);
        self.stats.dirty_sync_count.store(1, Ordering::Relaxed);
        self.stats.remaining_pages.store(self.pending_count, Ordering::Relaxed);
        self.page_count = 0;
        self.rates =
            RatePeriod { start: None, pages: 0, cache_miss: 0, xbzrle_pages: 0, xbzrle_bytes: 0 };
        // xbzrle_init()
        self.xbzrle = match &self.xbzrle_size {
            Some(size) => Some(XbzrleSave {
                cache: PageCache::new(size.load(Ordering::Relaxed), PAGE_SIZE)?,
                size: size.clone(),
                encoded: vec![0; PAGE_SIZE],
                started: false,
            }),
            None => None,
        };

        f.put_be64(listed | RAM_SAVE_FLAG_MEM_SIZE);
        self.file_blocks.clear();
        for b in &self.blocks {
            let name = b.name();
            f.put_byte(name.len() as u8);
            f.put_buffer(name.as_bytes());
            f.put_be64(b.len());
            if self.mapped_ram {
                // mapped_ram_setup_ramblock(): the header, then room for the pages.
                let fb = FileBlock::new(f.offset()? + HDR_LEN as u64, Self::pages(b));
                f.put_buffer(&fb.header(PAGE_SIZE as u64).to_bytes());
                f.set_offset(fb.pages_offset + b.len())?;
                self.file_blocks.push(Arc::new(fb));
            }
        }
        self.multifd_sync(f)?;
        self.end_section(f, false);
        Ok(())
    }

    /// `ram_save_iterate()`.
    fn save_iterate(&mut self, f: &mut QemuFile<'_>, max_bytes: u64) -> Result<bool> {
        self.xbzrle_resize()?;
        // migration_rate_exceeded() only sees what reached the channel, so the iteration goes
        // on until a flush takes it over the budget.
        let start = self.flushed(f);
        let mut buf = vec![0u8; PAGE_SIZE];
        // QEMU's destination keeps its last block across sections, but a fresh name costs
        // little and keeps every section self-contained.
        self.last_sent = None;
        let mut done = false;
        let t0 = std::time::Instant::now();
        let mut pages = 0u64;
        while self.flushed(f) - start < max_bytes {
            // An iteration ends after MAX_WAIT, which is checked after the first page, in case
            // a sync took long, and then every 64 pages.
            if pages % 64 == 1 && t0.elapsed() > MAX_WAIT {
                break;
            }
            pages += 1;
            if self.in_postcopy && self.send_requested(f, &mut buf)? {
                continue;
            }
            if self.send_faulted(f, &mut buf)? {
                continue;
            }
            match self.next_dirty() {
                Some((b, page)) => {
                    self.send_page(f, b, page, &mut buf)?;
                    self.release_protection(b, page)?;
                }
                None => {
                    // The round is complete; the next sync starts another, and XBZRLE starts
                    // with it.
                    self.cursor = (0, 0);
                    self.multifd_sync(f)?;
                    if let Some(x) = &mut self.xbzrle {
                        x.started = true;
                    }
                    done = true;
                    break;
                }
            }
        }
        self.end_section(f, true);
        Ok(done)
    }

    /// `ram_save_complete()`: with the guest stopped, a last sync and every page left. In
    /// postcopy there is nothing new to sync.
    fn save_complete(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        self.xbzrle_resize()?;
        self.last_stage = true;
        if !self.in_postcopy && !self.ignore {
            self.sync();
        }
        self.last_sent = None;
        let mut buf = vec![0u8; PAGE_SIZE];
        // The round goes on from where the last iteration stopped. When it wraps, the multifd
        // channels sync first, as find_dirty_block() has them do, so that no page overtakes an
        // older copy of itself still on another channel.
        let mid_round = self.cursor != (0, 0);
        while let Some((b, page)) = self.next_dirty() {
            self.send_page(f, b, page, &mut buf)?;
        }
        if mid_round {
            self.multifd_sync(f)?;
            self.cursor = (0, 0);
            while let Some((b, page)) = self.next_dirty() {
                self.send_page(f, b, page, &mut buf)?;
            }
        }
        self.multifd_sync(f)?;
        // ram_save_file_bmap(), now that no channel writes pages any more.
        for fb in std::mem::take(&mut self.file_blocks) {
            let bitmap = fb.bitmap_bytes();
            f.put_buffer_at(&bitmap, fb.bitmap_offset)
                .map_err(|e| e.prepend("Failed to write bitmap to file: "))?;
            self.account(bitmap.len() as u64);
        }
        self.end_section(f, false);
        self.stop_logging();
        Ok(())
    }

    /// `ram_state_pending_estimate()` and `ram_state_pending_exact()`.
    fn pending(&mut self, exact: bool) -> u64 {
        if exact && self.logging && !self.in_postcopy {
            self.sync();
        }
        self.pending_count << PAGE_BITS
    }

    fn save_cleanup(&mut self) {
        self.tracking = None;
        self.stop_logging();
        self.pending.clear();
        self.pending_count = 0;
        self.in_postcopy = false;
        self.multifd_send = None;
        self.file_blocks.clear();
        // xbzrle_cleanup()
        self.xbzrle = None;
    }

    fn set_save_params(&mut self, p: &SaveParams) {
        self.multifd_send = p.multifd.clone();
        self.zero_page_detection = p.zero_page_detection;
        self.xbzrle_size = p.xbzrle_cache_size.clone();
        self.mapped_ram = p.mapped_ram;
        self.background = p.background_snapshot;
        self.ignore = p.ignore_ram;
    }

    /// `ram_write_tracking_start()`.
    fn write_tracking_start(&mut self) -> Result<()> {
        self.tracking = Some(
            WriteTracking::start(&self.blocks)
                .map_err(|e| Error::from_io("ram_write_tracking_start() failed", e))?,
        );
        Ok(())
    }

    /// `ram_write_tracking_stop()`.
    fn write_tracking_stop(&mut self) {
        self.tracking = None;
    }

    fn set_load_params(&mut self, p: &LoadParams) {
        self.multifd_recv = p.multifd.clone();
        self.load_mapped_ram = p.mapped_ram;
        self.load_file = p.file.clone();
    }

    fn ram_blocks(&self) -> Vec<Arc<RamBlock>> {
        self.blocks.clone()
    }

    fn droppable_blocks(&self) -> Vec<String> {
        self.droppable.clone()
    }

    fn has_postcopy(&self) -> bool {
        true
    }

    /// `ram_save_postcopy_prepare()`.
    fn save_postcopy_prepare(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        self.multifd_sync(f)?;
        self.end_section(f, false);
        Ok(())
    }

    /// `ram_postcopy_send_discard_bitmap()`, which syncs a last time with the guest stopped and
    /// starts the scan for pages over.
    fn postcopy_discard_ranges(&mut self) -> Vec<(String, Vec<(u64, u64)>)> {
        if self.logging {
            self.sync();
        }
        self.in_postcopy = true;
        self.cursor = (0, 0);
        self.last_sent = None;
        self.pending_ranges()
    }

    fn page_requests(&self) -> Option<Arc<PageRequests>> {
        Some(self.requests.clone())
    }

    fn has_ram_block(&self, name: &str) -> bool {
        self.find_block(name).is_some()
    }

    /// `ram_postcopy_incoming_init()`: everything that came at boot goes, the source sends all
    /// of RAM anyway, and only an empty page makes userfaultfd report a fault.
    fn postcopy_advise(&mut self) -> Result<()> {
        for b in &self.blocks {
            b.discard_range(0, b.len()).map_err(|e| {
                Error::generic(format!("init_range: Failed to discard {}: {e}", b.name()))
            })?;
        }
        self.recv_map = Some(Arc::new(RecvMap::new(&self.blocks)));
        Ok(())
    }

    /// `ram_discard_range()`.
    fn postcopy_discard(&mut self, name: &str, start: u64, len: u64) -> Option<Result<()>> {
        let Some(i) = self.find_block(name) else {
            return self.droppable_name(name).then_some(Ok(()));
        };
        let block = &self.blocks[i];
        let ret = block.discard_range(start, len).map_err(|e| {
            Error::generic(format!(
                "ram_block_discard_range: Bad range {name} {start:x}+{len:x}: {e}"
            ))
        });
        if ret.is_ok() {
            if let Some(r) = &self.recv_map {
                r.clear(i, start >> PAGE_BITS, (start + len) >> PAGE_BITS);
            }
        }
        Some(ret)
    }

    /// `postcopy_ram_incoming_setup()`: registers every block with userfaultfd and starts the
    /// fault thread.
    fn postcopy_listen(&mut self, rp: Option<&Arc<ReturnPath>>) -> Result<()> {
        let Some(rp) = rp.cloned() else {
            bail!("Postcopy needs a return path to ask for pages");
        };
        let Some(recv) = self.recv_map.clone() else {
            bail!("Postcopy RAM was not advised");
        };
        let uffd = Uffd::new().map_err(|e| Error::from_io("Userfaultfd not available", e))?;
        for b in &self.blocks {
            // Closing the descriptor on failure unregisters what went before.
            uffd.register(b.host_memory())
                .map_err(|e| Error::from_io("UFFDIO_REGISTER failed", e))?;
        }
        // Device state comes next, and RAM changes under the guest from now on.
        self.hooks.load_done();
        let uffd = Arc::new(uffd);
        let quit = Arc::new(AtomicBool::new(false));
        let (u, q, blocks) = (uffd.clone(), quit.clone(), self.blocks.clone());
        let thread = std::thread::Builder::new()
            .name("mig/dst/fault".to_string())
            .spawn(move || fault_thread(&u, &blocks, &recv, &rp, &q))
            .map_err(|e| Error::from_io("failed to create the postcopy fault thread", e))?;
        self.listening = Some(Listening { uffd, quit, thread: Some(thread) });
        Ok(())
    }

    /// `postcopy_ram_incoming_cleanup()`.
    fn postcopy_end(&mut self) {
        if let Some(mut l) = self.listening.take() {
            l.quit.store(true, Ordering::Release);
            if let Some(t) = l.thread.take() {
                let _ = t.join();
            }
            for b in &self.blocks {
                if let Err(e) = l.uffd.unregister(b.host_memory()) {
                    error_report(&format!(
                        "postcopy_ram_incoming_cleanup: userfault unregister {e}"
                    ));
                }
            }
        }
    }

    fn load_setup(&mut self) -> Result<()> {
        self.last_recv = None;
        self.recv_map = None;
        Ok(())
    }

    /// `ram_load_precopy()`.
    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()> {
        if version_id != RAM_SECTION_VERSION {
            bail!("Unsupported RAM section version {}", version_id);
        }
        let mut buf = vec![0u8; PAGE_SIZE];
        // ram_load_postcopy(): after LISTEN every page goes in with userfaultfd.
        let postcopy = self.listening.is_some();
        // Pages come from their place in the file with mapped-ram, never in the stream.
        let invalid_flags = if self.load_mapped_ram {
            RAM_SAVE_FLAG_HOOK
                | RAM_SAVE_FLAG_MULTIFD_FLUSH
                | RAM_SAVE_FLAG_PAGE
                | RAM_SAVE_FLAG_XBZRLE
                | RAM_SAVE_FLAG_ZERO
        } else {
            0
        };
        loop {
            let word = f.get_be64();
            if f.get_error() != 0 {
                bail!("Getting RAM address failed");
            }
            let flags = word & (PAGE_SIZE as u64 - 1);
            let addr = word & !(PAGE_SIZE as u64 - 1);
            if flags & invalid_flags != 0 {
                bail!("Unexpected RAM flags: {}", flags & invalid_flags);
            }

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

            let what = flags & !RAM_SAVE_FLAG_CONTINUE;
            if postcopy
                && what != RAM_SAVE_FLAG_ZERO
                && what != RAM_SAVE_FLAG_PAGE
                && what != RAM_SAVE_FLAG_EOS
            {
                bail!("Unknown combination of migration flags: 0x{:x} (postcopy mode)", flags);
            }
            match what {
                RAM_SAVE_FLAG_MEM_SIZE => {
                    self.parse_blocks(f, addr)?;
                    // With mapped-ram the multifd channels read all of RAM in the block list;
                    // wait for them once and for all.
                    if self.load_mapped_ram {
                        if let Some(m) = &self.multifd_recv {
                            m.sync_main()?;
                        }
                    }
                }
                RAM_SAVE_FLAG_ZERO => {
                    let ch = f.get_byte();
                    if ch != 0 {
                        bail!("Found a zero page with value {}", ch);
                    }
                    if let Some(i) = target.filter(|_| postcopy) {
                        self.place(i, addr, None)?;
                    } else if let Some(i) = target {
                        // ram_handle_zero(): a page that is already zero is left alone, so the
                        // host does not have to back it.
                        let block = &self.blocks[i];
                        let illegal =
                            |e| Error::generic(format!("Illegal RAM offset {addr:x}: {e}"));
                        if !block.is_zero(addr, PAGE_SIZE as u64).map_err(illegal)? {
                            block.fill(addr, PAGE_SIZE as u64, 0).map_err(illegal)?;
                        }
                    }
                }
                RAM_SAVE_FLAG_PAGE => {
                    f.get_buffer(&mut buf);
                    if let Some(i) = target.filter(|_| postcopy) {
                        if f.get_error() == 0 {
                            self.place(i, addr, Some(&buf))?;
                        }
                    } else if let Some(i) = target {
                        self.blocks[i].write(addr, &buf).map_err(|e| {
                            Error::generic(format!("Illegal RAM offset {addr:x}: {e}"))
                        })?;
                    }
                }
                RAM_SAVE_FLAG_XBZRLE => {
                    if !Self::load_xbzrle(f, target.map(|i| &self.blocks[i]), addr, &mut buf) {
                        bail!("Failed to decompress XBZRLE page at {:x}", addr);
                    }
                }
                // multifd_recv_sync_main(): the pages of this round on every channel are in.
                RAM_SAVE_FLAG_MULTIFD_FLUSH => {
                    if let Some(m) = &self.multifd_recv {
                        m.sync_main()?;
                    }
                }
                RAM_SAVE_FLAG_EOS => break,
                RAM_SAVE_FLAG_HOOK => bail!("RDMA is not supported"),
                _ => bail!("Unknown combination of migration flags: 0x{:x}", flags),
            }
            let ret = f.get_error();
            if ret != 0 {
                bail!("RAM load failed: stream error {}", ret);
            }
            if let Some(i) = target {
                self.mark_received(i, addr);
            }
        }
        Ok(())
    }

    fn load_cleanup(&mut self) {
        self.postcopy_end();
        self.last_recv = None;
        self.recv_map = None;
        self.hooks.load_done();
    }
}

impl Drop for RamSection {
    fn drop(&mut self) {
        self.tracking = None;
        self.postcopy_end();
        self.stop_logging();
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::types::MigrationCapability;

    use super::*;
    use crate::savevm::{EntryInfo, LoadInfo, LoadOptions, MachineConfig, SaveVm};

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
        src.write(180 << 12, &[0xaa; 20 << 12]).unwrap();
        let mut s = vm(vec![src.clone()]);
        let mut out = Vec::new();
        {
            let mut f = QemuFile::new(&mut out);
            s.save_header(&mut f).unwrap();
            s.save_setup(&mut f).unwrap();
            // A small budget takes several passes over the first round. Only what got flushed
            // counts against it, so it takes more pages than fit in the buffer.
            let mut rounds = 0;
            while !s.save_iterate(&mut f, 4 << 10, false).unwrap() {
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
    fn writes_from_before_the_migration_go_once() {
        // Like a block of a memory system, which logs from the start.
        let src = block("pc.ram", 16);
        src.start_dirty_log(DirtyClient::Migration);
        src.write(0x3000, b"boot").unwrap();
        src.set_dirty(0x3000, 4, DirtyClient::Migration.mask());
        let mut s = vm(vec![src.clone()]);
        let mut out = Vec::new();
        let mut f = QemuFile::new(&mut out);
        s.save_setup(&mut f).unwrap();
        while !s.save_iterate(&mut f, 1 << 20, false).unwrap() {}
        // The page went out in the first round, and nothing wrote it since.
        assert_eq!(s.pending(true), 0);
    }

    #[test]
    fn ignored_ram_is_listed_but_not_sent() {
        // cpr-transfer: the destination maps the same memory, so only the block list goes out.
        let src = block("pc.ram", 64);
        src.write(0x2000, b"shared").unwrap();
        let mut s = vm(vec![src.clone()]);
        s.set_save_params(&SaveParams {
            multifd: None,
            zero_page_detection: ZeroPageDetection::Multifd,
            xbzrle_cache_size: None,
            mapped_ram: false,
            background_snapshot: false,
            ignore_ram: true,
        });
        let mut out = Vec::new();
        {
            let mut f = QemuFile::new(&mut out);
            s.save_header(&mut f).unwrap();
            s.save_setup(&mut f).unwrap();
            assert!(!src.is_dirty_logging(DirtyClient::Migration));
            assert_eq!(s.pending(false), 0);
            while !s.save_iterate(&mut f, 4 << 10, false).unwrap() {}
            s.save_complete(&mut f).unwrap();
        }
        assert!(out.len() < 4096, "{} bytes", out.len());
        let dst = block("pc.ram", 64);
        vm(vec![dst.clone()]).load_state(&mut StreamReader::new(&out)).unwrap();
        let mut b = [0; 6];
        dst.read(0x2000, &mut b).unwrap();
        assert_eq!(b, [0; 6]);
    }

    #[test]
    fn xbzrle_pages() {
        let src = block("pc.ram", 64);
        for page in 0..32u64 {
            src.fill(page << 12, 4096, page as u8 + 1).unwrap();
        }
        let ram = RamSection::new(vec![src.clone()], NoHooks);
        let stats = ram.stats();
        let mut s = SaveVm::new(config());
        s.register_live(EntryInfo::new("ram", RAM_SECTION_VERSION), ram);
        s.set_save_params(&SaveParams {
            multifd: None,
            zero_page_detection: ZeroPageDetection::Multifd,
            xbzrle_cache_size: Some(Arc::new(AtomicU64::new(16 << 12))),
            mapped_ram: false,
            background_snapshot: false,
            ignore_ram: false,
        });
        let dirty = |page: u64| src.set_dirty(page << 12, 4096, DirtyClient::Migration.mask());
        let x = &stats.xbzrle;
        let mut out = Vec::new();
        {
            let mut f = QemuFile::new(&mut out);
            s.save_header(&mut f).unwrap();
            s.save_setup(&mut f).unwrap();
            // The first round sends everything whole; XBZRLE starts after it.
            assert!(s.save_iterate(&mut f, 1 << 20, false).unwrap());
            assert_eq!(x.cache_miss.load(Ordering::Relaxed), 0);

            // The second round finds nothing in the cache yet and fills it.
            src.write(0x1000, b"hello").unwrap();
            for page in 1..4 {
                dirty(page);
            }
            assert_eq!(s.pending(true), 3 << 12);
            assert!(s.save_iterate(&mut f, 1 << 20, false).unwrap());
            assert_eq!(x.cache_miss.load(Ordering::Relaxed), 3);
            assert_eq!(x.pages.load(Ordering::Relaxed), 0);

            // Now page 1 goes out encoded, page 2 did not change and is left out, page 3
            // changed too much and goes whole, and page 5 is a zero page.
            src.write(0x1010, b"world").unwrap();
            let mut page = [0u8; 4096];
            for b in page.iter_mut().step_by(2) {
                *b = 0xee;
            }
            src.write(3 << 12, &page).unwrap();
            src.fill(5 << 12, 4096, 0).unwrap();
            for page in [1, 2, 3, 5] {
                dirty(page);
            }
            let normal = stats.normal.load(Ordering::Relaxed);
            s.save_complete(&mut f).unwrap();
            assert_eq!(stats.normal.load(Ordering::Relaxed), normal + 1);
        }
        assert_eq!(x.cache_miss.load(Ordering::Relaxed), 3);
        assert_eq!(x.pages.load(Ordering::Relaxed), 3);
        assert_eq!(x.overflow.load(Ordering::Relaxed), 1);
        // The header with the block name, zrun 16, nzrun 5 and "world", the flag and the
        // length, less the page word; and a whole page for the overflow.
        assert_eq!(x.bytes.load(Ordering::Relaxed), (15 + 7 + 3 - 8) + 4096);

        let dst = block("pc.ram", 64);
        vm(vec![dst.clone()]).load_state(&mut StreamReader::new(&out)).unwrap();
        let (mut a, mut b) = (vec![0; 64 << 12], vec![0; 64 << 12]);
        src.read(0, &mut a).unwrap();
        dst.read(0, &mut b).unwrap();
        assert!(a == b);
        assert_eq!(&b[0x1010..0x1015], b"world");

        // A page with an unknown encoding fails the load.
        let mut word = ((1u64 << 12) | RAM_SAVE_FLAG_XBZRLE).to_be_bytes().to_vec();
        word.extend_from_slice(b"\x06pc.ram\x01");
        let at = out.windows(word.len()).position(|w| w == word).unwrap();
        out[at + word.len() - 1] = 2;
        let err =
            vm(vec![block("pc.ram", 64)]).load_state(&mut StreamReader::new(&out)).unwrap_err();
        assert!(err.message().contains("Failed to decompress XBZRLE page at 1000"), "{err}");
    }

    #[test]
    fn postcopy_discard_and_requests() {
        let src = block("pc.ram", 200);
        let mut ram = RamSection::new(vec![src.clone()], NoHooks);
        let mut out = Vec::new();
        ram.save_setup(&mut QemuFile::new(&mut out)).unwrap();
        // The first 100 pages went out, then the guest wrote page 10 again.
        for _ in 0..100 {
            ram.next_dirty().unwrap();
        }
        src.set_dirty(10 << 12, 1, DirtyClient::Migration.mask());
        assert_eq!(
            ram.postcopy_discard_ranges(),
            vec![("pc.ram".to_string(), vec![(10 << 12, 1 << 12), (100 << 12, 100 << 12)])]
        );
        assert_eq!(ram.pending(true), 101 << 12);

        // A requested page goes first.
        let requests = ram.page_requests().unwrap();
        requests.queue(Some("pc.ram"), 150 << 12, 4096).unwrap();
        let n = out.len();
        assert!(!ram.save_iterate(&mut QemuFile::new(&mut out), 1).unwrap());
        assert_eq!(out[n..n + 8], ((150u64 << 12) | RAM_SAVE_FLAG_ZERO).to_be_bytes());
        assert_eq!(ram.pending(false), 100 << 12);
        assert_eq!(requests.requests(), 1);
        assert!(ram.stats().postcopy_bytes.load(Ordering::Relaxed) > 0);
        // Asking again for a page that went out sends nothing.
        requests.queue(None, 150 << 12, 4096).unwrap();
        while !ram.save_iterate(&mut QemuFile::new(&mut out), 1 << 20).unwrap() {}
        assert_eq!(ram.pending(true), 0);
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

    fn load_mapped(uri: &str, blocks: Vec<Arc<RamBlock>>) -> Result<LoadInfo> {
        use crate::channel::{Channel, parse_uri};
        let l = Channel::listen(&parse_uri(uri).unwrap(), None).unwrap();
        let file = l.file();
        let mut f = StreamReader::from_reader(l.accept().unwrap());
        let mut d = vm(blocks);
        d.capabilities = vec![MigrationCapability::MappedRam];
        d.load_state_with(&mut f, LoadOptions { file, ..Default::default() })
    }

    #[test]
    fn mapped_ram_round_trip() {
        use crate::channel::{Channel, parse_uri};
        let path = std::env::temp_dir().join(format!("ruvm-ram-mapped-{}", std::process::id()));
        let uri = format!("file:{},offset=0x1000", path.display());
        let src = vec![block("pc.ram", 300), block("/rom@etc/acpi/tables", 2)];
        for p in 0..300u64 {
            if p % 4 != 0 {
                src[0].fill(p << 12, 4096, p as u8 | 1).unwrap();
            }
        }
        src[1].write(0, &[5; 64]).unwrap();
        let mut s = vm(src.clone());
        s.capabilities = vec![MigrationCapability::MappedRam];
        s.set_save_params(&SaveParams {
            multifd: None,
            zero_page_detection: ZeroPageDetection::Legacy,
            xbzrle_cache_size: None,
            mapped_ram: true,
            background_snapshot: false,
            ignore_ram: false,
        });
        let c = Channel::connect_socket(&parse_uri(&uri).unwrap(), None).unwrap();
        {
            let mut f = QemuFile::with_file(c.out, c.file);
            s.save_header(&mut f).unwrap();
            s.save_setup(&mut f).unwrap();
            while !s.save_iterate(&mut f, 64 << 10, false).unwrap() {}
            // A page sent before changes, and one becomes zero: both land on their old place.
            src[0].fill(0x5000, 4096, 0xee).unwrap();
            src[0].set_dirty(0x5000, 4096, DirtyClient::Migration.mask());
            src[0].fill(0x6000, 4096, 0).unwrap();
            src[0].set_dirty(0x6000, 4096, DirtyClient::Migration.mask());
            s.save_complete(&mut f).unwrap();
            f.fflush().unwrap();
        }
        s.save_cleanup();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 300 << 12 && len < (300 << 12) + (4 << 20), "{len}");

        // The fw_cfg blob is dropped on load, and its pages skipped.
        let dst = block("pc.ram", 300);
        load_mapped(&uri, vec![dst.clone()]).unwrap();
        let (mut a, mut b) = (vec![0; 300 << 12], vec![0; 300 << 12]);
        src[0].read(0, &mut a).unwrap();
        dst.read(0, &mut b).unwrap();
        assert!(a == b);
        assert_eq!(b[0x5000], 0xee);

        // Without the file on the incoming side there is nothing to read the pages from.
        let mut raw = std::fs::read(&path).unwrap();
        let mut d = vm(vec![block("pc.ram", 300)]);
        d.capabilities = vec![MigrationCapability::MappedRam];
        let e = d.load_state(&mut StreamReader::new(&raw[0x1000..])).unwrap_err();
        assert!(
            e.message().contains("Migration requires seekable transport (e.g. file)"),
            "{}",
            e.message()
        );

        // A header from a newer QEMU.
        let pat = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0x10, 0];
        let at = raw.windows(pat.len()).position(|w| w == pat).unwrap();
        raw[at + 3] = 2;
        std::fs::write(&path, &raw).unwrap();
        let e = load_mapped(&uri, vec![block("pc.ram", 300)]).unwrap_err();
        assert!(
            e.message().contains(
                "Migration mapped-ram capability version not supported (expected <= 1, got 2)"
            ),
            "{}",
            e.message()
        );
        let _ = std::fs::remove_file(path);
    }
}
