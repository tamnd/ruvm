// SPDX-License-Identifier: GPL-2.0-or-later

//! Multifd, migration/multifd.c, multifd-nocomp.c and multifd-zlib.c.
//!
//! With the `multifd` capability RAM pages leave the main stream and go over extra sockets,
//! `multifd-channels` of them, each with its own thread on both sides. A channel starts with a
//! 64 byte `MultiFDInit_t` that says which channel it is. After that it carries packets: a
//! `MultiFDPacket_t` header naming a RAM block and up to 128 page offsets, the non-zero pages
//! first and the zero pages after them, followed by the data of the non-zero pages, raw or as one
//! piece of a zlib stream that runs for the whole life of the channel.
//!
//! The main stream still decides when RAM is consistent. At the end of each pass over RAM, and
//! before the device state, the source sends a packet with `MULTIFD_FLAG_SYNC` on every channel
//! and then `RAM_SAVE_FLAG_MULTIFD_FLUSH` on the main stream. The destination waits on that flag
//! until every channel has seen its sync packet, so no page from an earlier pass can land after
//! a later one.
//!
//! The destination tells its channels apart by their first four bytes: `QEVM` is the main
//! stream and `MULTIFD_MAGIC` a multifd channel, so they may connect in any order.
//!
//! With `mapped-ram` on a `file:` channel the channels are more handles on the same file, and
//! there are no packets at all: the sending threads write each page at its place in the file
//! (multifd_file_write_ramblock_iov()) and the receiving threads read the ranges the main thread
//! hands them. The syncs only wait for the local threads then.
//!
//! Not here: zstd, qatzip, QPL and UADK compression, TLS, zero copy and device state sent over
//! multifd (no ruvm device has a `save_live_complete_precopy_thread`).

use std::fs::File;
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use flate2::{Compress, Decompress, FlushCompress, FlushDecompress, Status};
use ruvm_base::{Error, Result, bail};
use ruvm_mem::{RamBlock, buffer_is_zero};

use crate::channel::{Accepted, Incoming, Listener, Socket, read_exact_at, write_all_at};
use crate::mapped_ram::{AlignedBuf, FileBlock};
use crate::ram::RamStats;

/// `MULTIFD_MAGIC`, the first word of every channel and packet.
pub const MULTIFD_MAGIC: u32 = 0x1122_3344;
/// `MULTIFD_VERSION`.
pub const MULTIFD_VERSION: u32 = 1;
/// `QEMU_VM_FILE_MAGIC`, how the main stream starts.
const QEMU_VM_FILE_MAGIC: u32 = 0x5145_564d;

const FLAG_SYNC: u32 = 1;
const FLAG_COMPRESSION_MASK: u32 = 0x1f << 1;
const FLAG_NOCOMP: u32 = 0;
const FLAG_ZLIB: u32 = 1 << 1;
const FLAG_DEVICE_STATE: u32 = 32 << 1;

/// `MULTIFD_PACKET_SIZE`: the most page data one packet carries.
const PACKET_SIZE: usize = 512 << 10;
const PAGE_SIZE: usize = 4096;
/// `multifd_ram_page_count()`.
const PAGE_COUNT: usize = PACKET_SIZE / PAGE_SIZE;
/// `sizeof(MultiFDInit_t)`.
const INIT_LEN: usize = 64;
/// `sizeof(MultiFDPacketHdr_t)`.
const HDR_LEN: usize = 12;
/// `sizeof(MultiFDPacket_t)` with room for a full packet of offsets.
const PACKET_LEN: usize = 320 + 8 * PAGE_COUNT;
/// `sizeof(MultiFDPacketDeviceState_t)`.
const DEVICE_STATE_LEN: usize = HDR_LEN + 256 + 4 + 4;
/// Where the fields of `MultiFDPacket_t` are.
const OFF_PAGES_ALLOC: usize = 12;
const OFF_NORMAL_PAGES: usize = 16;
const OFF_NEXT_PACKET_SIZE: usize = 20;
const OFF_PACKET_NUM: usize = 24;
const OFF_ZERO_PAGES: usize = 32;
const OFF_RAMBLOCK: usize = 64;
const OFF_OFFSET: usize = 320;

/// How the page data of a packet is encoded, `multifd-compression`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// The pages as they are.
    None,
    /// A zlib stream at this level, `multifd-zlib-level`.
    Zlib(u32),
}

/// The settings of one multifd migration.
#[derive(Debug, Clone, Copy)]
pub struct MultifdParams {
    /// `multifd-channels`.
    pub channels: u8,
    /// `multifd-compression`.
    pub compression: Compression,
    /// Whether the channel threads find the zero pages, `zero-page-detection=multifd`.
    pub zero_pages: bool,
    /// `qemu_uuid`, which every channel announces and the destination compares with its own.
    pub uuid: [u8; 16],
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Waits on `cond` for a wakeup or 100 ms, whichever comes first, so the caller can look at the
/// cancel flag again.
fn wait_a_while<'a, T>(cond: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cond.wait_timeout(g, Duration::from_millis(100)).map_or_else(|e| e.into_inner().0, |r| r.0)
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn be64(b: &[u8], at: usize) -> u64 {
    let mut w = [0; 8];
    w.copy_from_slice(&b[at..at + 8]);
    u64::from_be_bytes(w)
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_be_bytes());
}

fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_be_bytes());
}

/// `qemu_uuid_unparse()`.
fn uuid_unparse(u: &[u8]) -> String {
    let h: String = u.iter().map(|b| format!("{b:02x}")).collect();
    format!("{}-{}-{}-{}-{}", &h[..8], &h[8..12], &h[12..16], &h[16..20], &h[20..])
}

/// `MultiFDInit_t` of channel `id`.
fn init_packet(uuid: &[u8; 16], id: u8) -> [u8; INIT_LEN] {
    let mut b = [0; INIT_LEN];
    put32(&mut b, 0, MULTIFD_MAGIC);
    put32(&mut b, 4, MULTIFD_VERSION);
    b[8..24].copy_from_slice(uuid);
    b[24] = id;
    b
}

/// `multifd_recv_initial_packet()`: the channel id, checked against this side.
fn parse_init(b: &[u8; INIT_LEN], uuid: &[u8; 16], channels: u8) -> Result<u8> {
    let (magic, version) = (be32(b, 0), be32(b, 4));
    if magic != MULTIFD_MAGIC {
        bail!("multifd: received packet magic {:x} expected {:x}", magic, MULTIFD_MAGIC);
    }
    if version != MULTIFD_VERSION {
        bail!("multifd: received packet version {} expected {}", version, MULTIFD_VERSION);
    }
    let id = b[24];
    if b[8..24] != uuid[..] {
        bail!(
            "multifd: received uuid '{}' and expected uuid '{}' for channel {}",
            uuid_unparse(&b[8..24]),
            uuid_unparse(uuid),
            id as i8
        );
    }
    if id >= channels {
        bail!("multifd: received channel id {} exceeds channel count {}", id, channels);
    }
    Ok(id)
}

/// `multifd_send_fill_packet()`. A sync packet has no RAM fields.
fn fill_packet(
    p: &mut [u8],
    flags: u32,
    next_packet_size: u32,
    packet_num: u64,
    pages: Option<(&str, u32, &[u64])>,
) {
    p.fill(0);
    put32(p, 0, MULTIFD_MAGIC);
    put32(p, 4, MULTIFD_VERSION);
    put32(p, 8, flags);
    put32(p, OFF_NEXT_PACKET_SIZE, next_packet_size);
    put64(p, OFF_PACKET_NUM, packet_num);
    if let Some((name, normal, offsets)) = pages {
        // multifd_ram_fill_packet()
        put32(p, OFF_PAGES_ALLOC, PAGE_COUNT as u32);
        put32(p, OFF_NORMAL_PAGES, normal);
        put32(p, OFF_ZERO_PAGES, offsets.len() as u32 - normal);
        // pstrcpy() into 256 bytes keeps at most 255 and the terminator.
        let n = name.len().min(255);
        p[OFF_RAMBLOCK..OFF_RAMBLOCK + n].copy_from_slice(&name.as_bytes()[..n]);
        for (i, &o) in offsets.iter().enumerate() {
            put64(p, OFF_OFFSET + 8 * i, o);
        }
    }
}

/// Reads until `buf` is full or the stream ends, and says how much came.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

/// `qio_channel_read_all()`.
fn read_all(r: &mut impl Read, buf: &mut [u8]) -> Result<()> {
    match read_full(r, buf) {
        Ok(n) if n == buf.len() => Ok(()),
        Ok(_) => bail!("Unexpected end-of-file before all data were read"),
        Err(e) => Err(Error::from_io("Unable to read from socket", e)),
    }
}

/// zlib's return codes, for the messages.
fn zcode<E>(r: &std::result::Result<Status, E>, err: i32) -> i32 {
    match r {
        Ok(Status::Ok) => 0,
        Ok(Status::StreamEnd) => 1,
        Ok(Status::BufError) => -5,
        Err(_) => err,
    }
}

/// `compressBound()`.
fn compress_bound(n: usize) -> usize {
    n + (n >> 12) + (n >> 14) + (n >> 25) + 13
}

enum Work {
    // The pages, and with mapped-ram where the block is in the file.
    Pages(Arc<RamBlock>, Option<Arc<FileBlock>>, Vec<u64>),
    Sync,
}

/// What a channel goes over.
enum Wire {
    Socket(Socket),
    // A handle on the file of a mapped-ram migration.
    File(File),
}

struct SendState {
    // The job of each channel, until its thread takes it.
    slots: Vec<Option<Work>>,
    busy: Vec<bool>,
    next: usize,
    error: Option<String>,
    exiting: bool,
}

impl SendState {
    fn idle(&self, i: usize) -> bool {
        self.slots[i].is_none() && !self.busy[i]
    }
}

struct SendShared {
    state: Mutex<SendState>,
    cond: Condvar,
    packet_num: AtomicU64,
    stats: Arc<RamStats>,
    cancel: Arc<AtomicBool>,
}

impl SendShared {
    /// `multifd_send_set_error()` and `multifd_send_kick_main()`.
    fn set_error(&self, e: &Error) {
        let mut s = lock(&self.state);
        if s.error.is_none() && !s.exiting {
            s.error = Some(e.message().to_string());
        }
        drop(s);
        self.cond.notify_all();
    }
}

/// The pages `multifd_queue_page()` collected for the next packet.
#[derive(Default)]
struct Queue {
    block: Option<Arc<RamBlock>>,
    file: Option<Arc<FileBlock>>,
    offsets: Vec<u64>,
}

/// The sending side: one thread per channel and the page queue of the migration thread.
pub struct MultifdSend {
    shared: Arc<SendShared>,
    queue: Mutex<Queue>,
    // Other handles on the sockets, to wake the threads up. A file channel has none.
    ctl: Vec<Socket>,
    channels: usize,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for MultifdSend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultifdSend").field("channels", &self.channels).finish()
    }
}

/// What one sending thread keeps to itself.
struct SendChannel {
    id: u8,
    wire: Wire,
    zlib: Option<Compress>,
    zero_pages: bool,
    pages: Vec<u8>,
    packet: Vec<u8>,
    aligned: AlignedBuf,
}

impl SendChannel {
    /// `multifd_send_thread()`.
    fn run(&mut self, shared: &SendShared, uuid: &[u8; 16]) {
        // multifd_send_initial_packet(), which a file does without.
        if let Wire::Socket(sock) = &mut self.wire {
            let init = init_packet(uuid, self.id);
            if let Err(e) = sock.write_all(&init) {
                shared.set_error(&Error::from_io("Unable to write to socket", e));
                return;
            }
            shared.stats.multifd_bytes.fetch_add(INIT_LEN as u64, Ordering::Relaxed);
        }
        let id = usize::from(self.id);
        loop {
            let work = {
                let mut s = lock(&shared.state);
                loop {
                    if s.exiting {
                        return;
                    }
                    if let Some(w) = s.slots[id].take() {
                        s.busy[id] = true;
                        break w;
                    }
                    s = shared.cond.wait(s).unwrap_or_else(|e| e.into_inner());
                }
            };
            let res = match (work, &self.wire) {
                (Work::Pages(block, Some(fb), offsets), Wire::File(_)) => {
                    self.write_pages(shared, &block, &fb, offsets)
                }
                (Work::Pages(block, _, offsets), _) => self.send_pages(shared, &block, offsets),
                // MULTIFD_SYNC_LOCAL: a file has nobody to tell.
                (Work::Sync, Wire::File(_)) => Ok(()),
                (Work::Sync, Wire::Socket(_)) => self.send_sync(shared),
            };
            if let Err(e) = res {
                shared.set_error(&e);
                return;
            }
            lock(&shared.state).busy[id] = false;
            shared.cond.notify_all();
        }
    }

    fn write(&mut self, shared: &SendShared, len: usize) -> Result<()> {
        let Wire::Socket(sock) = &mut self.wire else {
            bail!("multifd {}: a file channel has no packets", self.id);
        };
        sock.write_all(&self.packet[..len])
            .map_err(|e| Error::from_io("Unable to write to socket", e))?;
        shared.stats.multifd_bytes.fetch_add(len as u64, Ordering::Relaxed);
        Ok(())
    }

    fn send_sync(&mut self, shared: &SendShared) -> Result<()> {
        let num = shared.packet_num.fetch_add(1, Ordering::Relaxed);
        self.packet.resize(PACKET_LEN, 0);
        fill_packet(&mut self.packet[..PACKET_LEN], FLAG_SYNC, 0, num, None);
        self.write(shared, PACKET_LEN)
    }

    /// `multifd_send_zero_page_detect()`: moves the zero pages to the end of `offsets` the way
    /// QEMU does and returns the number of the others.
    fn zero_page_detect(&mut self, block: &RamBlock, offsets: &mut [u64]) -> Result<usize> {
        let n = offsets.len();
        self.pages.resize(n * PAGE_SIZE, 0);
        for (i, &o) in offsets.iter().enumerate() {
            block.read(o, &mut self.pages[i * PAGE_SIZE..(i + 1) * PAGE_SIZE]).map_err(|e| {
                Error::generic(format!("Failed to read RAM block {}: {e}", block.name()))
            })?;
        }
        if !self.zero_pages {
            return Ok(n);
        }
        // Where the data of each offset is in `pages`.
        let mut slot: Vec<usize> = (0..n).collect();
        let (mut i, mut j) = (0usize, n);
        while i < j {
            let s = slot[i];
            if !buffer_is_zero(&self.pages[s * PAGE_SIZE..(s + 1) * PAGE_SIZE]) {
                i += 1;
                continue;
            }
            offsets.swap(i, j - 1);
            slot.swap(i, j - 1);
            j -= 1;
        }
        // Put the data of the normal pages in their order at the start.
        let mut data = vec![0u8; i * PAGE_SIZE];
        for (k, &s) in slot[..i].iter().enumerate() {
            data[k * PAGE_SIZE..(k + 1) * PAGE_SIZE]
                .copy_from_slice(&self.pages[s * PAGE_SIZE..(s + 1) * PAGE_SIZE]);
        }
        self.pages[..i * PAGE_SIZE].copy_from_slice(&data);
        Ok(i)
    }

    /// The mapped-ram `send_prepare` of `nocomp`, multifd_set_file_bitmap() and
    /// file_write_ramblock_iov(): every page that is not zero goes to its place in the file,
    /// one write per run of pages that follow each other.
    fn write_pages(
        &mut self,
        shared: &SendShared,
        block: &RamBlock,
        fb: &FileBlock,
        mut offsets: Vec<u64>,
    ) -> Result<()> {
        let normal = self.zero_page_detect(block, &mut offsets)?;
        shared.stats.normal.fetch_add(normal as u64, Ordering::Relaxed);
        shared.stats.duplicate.fetch_add((offsets.len() - normal) as u64, Ordering::Relaxed);
        for &o in &offsets[..normal] {
            fb.set(o >> PAGE_SIZE.trailing_zeros(), true);
        }
        for &o in &offsets[normal..] {
            fb.set(o >> PAGE_SIZE.trailing_zeros(), false);
        }
        let Wire::File(file) = &self.wire else {
            bail!("multifd {}: mapped-ram needs a file channel", self.id);
        };
        // With O_DIRECT the data has to come from an aligned buffer.
        let data = self.aligned.get(normal * PAGE_SIZE);
        data.copy_from_slice(&self.pages[..normal * PAGE_SIZE]);
        let mut i = 0;
        while i < normal {
            let mut j = i + 1;
            while j < normal && offsets[j] == offsets[j - 1] + PAGE_SIZE as u64 {
                j += 1;
            }
            if offsets[i] >= block.len() {
                bail!("offset {:x}outside of ramblock {} range", offsets[i], block.name());
            }
            write_all_at(file, &data[i * PAGE_SIZE..j * PAGE_SIZE], fb.pages_offset + offsets[i])?;
            i = j;
        }
        shared.stats.multifd_bytes.fetch_add((normal * PAGE_SIZE) as u64, Ordering::Relaxed);
        Ok(())
    }

    /// `send_prepare` of the compression method, then the write.
    fn send_pages(
        &mut self,
        shared: &SendShared,
        block: &RamBlock,
        mut offsets: Vec<u64>,
    ) -> Result<()> {
        let normal = self.zero_page_detect(block, &mut offsets)?;
        shared.stats.normal.fetch_add(normal as u64, Ordering::Relaxed);
        shared.stats.duplicate.fetch_add((offsets.len() - normal) as u64, Ordering::Relaxed);
        self.packet.resize(PACKET_LEN, 0);
        let (flags, size) = match self.zlib.as_mut() {
            None => {
                let size = normal * PAGE_SIZE;
                self.packet.extend_from_slice(&self.pages[..size]);
                (FLAG_NOCOMP, size)
            }
            Some(z) => {
                // multifd_zlib_send_prepare()
                let bound = compress_bound(PACKET_SIZE);
                self.packet.resize(PACKET_LEN + bound, 0);
                let out = &mut self.packet[PACKET_LEN..];
                let mut out_size = 0;
                for i in 0..normal {
                    let flush =
                        if i + 1 == normal { FlushCompress::Sync } else { FlushCompress::None };
                    let page = &self.pages[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
                    let mut used = 0;
                    let ret = loop {
                        let (in0, out0) = (z.total_in(), z.total_out());
                        let r = z.compress(&page[used..], &mut out[out_size..], flush);
                        used += (z.total_in() - in0) as usize;
                        out_size += (z.total_out() - out0) as usize;
                        if !matches!(r, Ok(Status::Ok)) || used == PAGE_SIZE || out_size == bound {
                            break r;
                        }
                    };
                    if matches!(ret, Ok(Status::Ok)) && used < PAGE_SIZE {
                        bail!("multifd {}: deflate failed to compress all input", self.id);
                    }
                    if !matches!(ret, Ok(Status::Ok)) {
                        bail!(
                            "multifd {}: deflate returned {} instead of Z_OK",
                            self.id,
                            zcode(&ret, -2)
                        );
                    }
                }
                self.packet.truncate(PACKET_LEN + out_size);
                (FLAG_ZLIB, out_size)
            }
        };
        let num = shared.packet_num.fetch_add(1, Ordering::Relaxed);
        let pages = Some((block.name(), normal as u32, &offsets[..]));
        fill_packet(&mut self.packet[..PACKET_LEN], flags, size as u32, num, pages);
        self.write(shared, PACKET_LEN + size)
    }
}

impl MultifdSend {
    /// `multifd_send_setup()` over sockets already connected: starts a thread per channel,
    /// which announces itself first.
    pub fn start(
        sockets: Vec<Socket>,
        params: &MultifdParams,
        stats: Arc<RamStats>,
        cancel: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        let mut ctl = Vec::with_capacity(sockets.len());
        for s in &sockets {
            ctl.push(s.try_clone().map_err(|e| Error::from_io("multifd: socket", e))?);
        }
        let wires = sockets.into_iter().map(Wire::Socket).collect();
        Self::start_wires(wires, ctl, params, stats, cancel)
    }

    /// [`start`](Self::start) for mapped-ram, over more handles on the migration file. Pages
    /// must then be queued with their [`FileBlock`].
    pub fn start_file(
        files: Vec<File>,
        params: &MultifdParams,
        stats: Arc<RamStats>,
        cancel: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        let wires = files.into_iter().map(Wire::File).collect();
        Self::start_wires(wires, Vec::new(), params, stats, cancel)
    }

    fn start_wires(
        wires: Vec<Wire>,
        ctl: Vec<Socket>,
        params: &MultifdParams,
        stats: Arc<RamStats>,
        cancel: Arc<AtomicBool>,
    ) -> Result<Arc<Self>> {
        let n = wires.len();
        let shared = Arc::new(SendShared {
            state: Mutex::new(SendState {
                slots: (0..n).map(|_| None).collect(),
                busy: vec![false; n],
                next: 0,
                error: None,
                exiting: false,
            }),
            cond: Condvar::new(),
            packet_num: AtomicU64::new(0),
            stats,
            cancel,
        });
        let me = Arc::new(MultifdSend {
            shared: shared.clone(),
            queue: Mutex::new(Queue::default()),
            ctl,
            channels: n,
            threads: Mutex::new(Vec::new()),
        });
        for (i, wire) in wires.into_iter().enumerate() {
            let zlib = match params.compression {
                Compression::None => None,
                Compression::Zlib(level) => {
                    Some(Compress::new(flate2::Compression::new(level), true))
                }
            };
            let mut ch = SendChannel {
                id: i as u8,
                wire,
                zlib,
                zero_pages: params.zero_pages,
                pages: Vec::new(),
                packet: Vec::with_capacity(PACKET_LEN + compress_bound(PACKET_SIZE)),
                aligned: AlignedBuf::default(),
            };
            let (sh, uuid) = (shared.clone(), params.uuid);
            let t = std::thread::Builder::new()
                .name(format!("mig/src/send_{i}"))
                .spawn(move || ch.run(&sh, &uuid));
            match t {
                Ok(t) => lock(&me.threads).push(t),
                Err(e) => {
                    me.shutdown();
                    return Err(Error::from_io("failed to create the multifd send thread", e));
                }
            }
        }
        Ok(me)
    }

    fn check(&self, s: &SendState) -> Result<()> {
        if let Some(e) = &s.error {
            return Err(Error::generic(e.clone()));
        }
        if s.exiting || self.shared.cancel.load(Ordering::Relaxed) {
            bail!("multifd: the migration is being cancelled");
        }
        Ok(())
    }

    /// `multifd_send()`: gives the work to the next idle channel, round robin.
    fn send(&self, work: Work) -> Result<()> {
        let sh = &self.shared;
        let mut s = lock(&sh.state);
        loop {
            self.check(&s)?;
            let n = s.slots.len();
            if let Some(i) = (0..n).map(|k| (s.next + k) % n).find(|&i| s.idle(i)) {
                s.slots[i] = Some(work);
                s.next = (i + 1) % n;
                drop(s);
                sh.cond.notify_all();
                return Ok(());
            }
            s = wait_a_while(&sh.cond, s);
        }
    }

    fn send_queue(&self, q: &mut Queue) -> Result<()> {
        if let Some(block) = q.block.take() {
            let offsets = std::mem::take(&mut q.offsets);
            if !offsets.is_empty() {
                self.send(Work::Pages(block, q.file.take(), offsets))?;
            }
        }
        Ok(())
    }

    /// `multifd_queue_page()`. With mapped-ram, `file` is where the block goes in the file.
    pub fn queue_page(
        &self,
        block: &Arc<RamBlock>,
        file: Option<&Arc<FileBlock>>,
        offset: u64,
    ) -> Result<()> {
        let mut q = lock(&self.queue);
        if q.block.as_ref().is_some_and(|b| !Arc::ptr_eq(b, block)) {
            self.send_queue(&mut q)?;
        }
        q.block = Some(block.clone());
        q.file = file.cloned();
        q.offsets.push(offset);
        if q.offsets.len() == PAGE_COUNT {
            self.send_queue(&mut q)?;
        }
        Ok(())
    }

    /// The part of `multifd_ram_flush_and_sync()` before `RAM_SAVE_FLAG_MULTIFD_FLUSH`: sends
    /// what is queued, then a sync packet on every channel, and waits until all of them went
    /// out. On a file there are no packets, and this only waits for the threads.
    pub fn flush_and_sync(&self) -> Result<()> {
        self.send_queue(&mut lock(&self.queue))?;
        // multifd_send_sync_main(MULTIFD_SYNC_ALL)
        let sh = &self.shared;
        let mut s = lock(&sh.state);
        let n = s.slots.len();
        let mut pending: Vec<usize> = (0..n).collect();
        while !pending.is_empty() {
            self.check(&s)?;
            pending.retain(|&i| {
                if s.idle(i) {
                    s.slots[i] = Some(Work::Sync);
                    false
                } else {
                    true
                }
            });
            sh.cond.notify_all();
            if !pending.is_empty() {
                s = wait_a_while(&sh.cond, s);
            }
        }
        while !(0..n).all(|i| s.idle(i)) {
            self.check(&s)?;
            s = wait_a_while(&sh.cond, s);
        }
        self.check(&s)
    }

    /// `mig_stats.multifd_bytes`: what the channels sent so far.
    pub fn bytes(&self) -> u64 {
        self.shared.stats.multifd_bytes.load(Ordering::Relaxed)
    }

    /// `multifd_send_shutdown()`: stops the threads, waking any that waits on its socket.
    pub fn shutdown(&self) {
        lock(&self.shared.state).exiting = true;
        self.shared.cond.notify_all();
        for s in &self.ctl {
            s.shutdown();
        }
        for t in lock(&self.threads).drain(..) {
            let _ = t.join();
        }
    }
}

/// What the receiving threads need to know about the machine.
#[derive(Debug, Clone, Default)]
pub struct RecvConfig {
    /// The RAM blocks pages may go to.
    pub blocks: Vec<Arc<RamBlock>>,
    /// Prefixes of block names whose pages are read and dropped when this side has no such
    /// block.
    pub droppable: Vec<String>,
    /// `(idstr, instance_id)` of every entry, for the device state packets.
    pub entries: Vec<(String, u32)>,
    /// Whether this side expects zlib data, from its own `multifd-compression`.
    pub zlib: bool,
    /// The `postcopy-ram` capability: zero pages are always written then.
    pub postcopy_ram: bool,
}

/// A range of pages the main thread gives a channel to read from the file, `MultiFDRecvData`.
struct FileJob {
    block: Arc<RamBlock>,
    offset: u64,
    file_offset: u64,
    size: usize,
}

struct RecvState {
    // With mapped-ram, the job of each channel until its thread takes it, and whether it is
    // still at one.
    jobs: Vec<Option<FileJob>>,
    busy: Vec<bool>,
    next: usize,
    // Channels that saw a sync packet and wait to be let go.
    synced: usize,
    // Per channel, bumped to let it go.
    release: Vec<u64>,
    // Channels that ended at the end of their stream.
    closed: usize,
    error: Option<String>,
    exiting: bool,
}

struct RecvShared {
    state: Mutex<RecvState>,
    cond: Condvar,
    ctl: Vec<Socket>,
    channels: usize,
    // The channels are handles on a mapped-ram file.
    file: bool,
}

impl RecvShared {
    /// `multifd_recv_terminate_threads()`.
    fn terminate(&self, e: Option<&Error>) {
        let mut s = lock(&self.state);
        if let Some(e) = e {
            if s.error.is_none() && !s.exiting {
                s.error = Some(e.message().to_string());
            }
        }
        let first = !s.exiting;
        s.exiting = true;
        drop(s);
        self.cond.notify_all();
        if first {
            for c in &self.ctl {
                c.shutdown();
            }
        }
    }

    fn exiting(&self) -> bool {
        lock(&self.state).exiting
    }
}

/// The receiving side: one thread per channel.
pub struct MultifdRecv {
    shared: Arc<RecvShared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for MultifdRecv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultifdRecv").field("channels", &self.shared.channels).finish()
    }
}

/// Where the pages of a packet go.
enum Target {
    Block(usize),
    Dropped,
}

/// A `MultiFDPacket_t` after `multifd_ram_unfill_packet()`.
struct RamPacket {
    flags: u32,
    next_packet_size: u32,
    target: Option<Target>,
    normal: Vec<u64>,
    zero: Vec<u64>,
}

struct RecvChannel {
    id: u8,
    sock: Socket,
    zlib: Option<Decompress>,
    packet: Vec<u8>,
    zbuf: Vec<u8>,
    data: Vec<u8>,
}

impl RecvChannel {
    /// `multifd_recv_thread()`.
    fn run(&mut self, cfg: &RecvConfig, shared: &RecvShared) {
        if let Err(e) = self.recv_loop(cfg, shared) {
            shared.terminate(Some(&e));
        }
    }

    fn recv_loop(&mut self, cfg: &RecvConfig, shared: &RecvShared) -> Result<()> {
        let mut hdr = [0u8; HDR_LEN];
        loop {
            if shared.exiting() {
                return Ok(());
            }
            match read_full(&mut self.sock, &mut hdr) {
                Ok(0) => {
                    lock(&shared.state).closed += 1;
                    shared.cond.notify_all();
                    return Ok(());
                }
                Ok(HDR_LEN) => {}
                Ok(_) => bail!("Unexpected end-of-file before all data were read"),
                Err(_) if shared.exiting() => return Ok(()),
                Err(e) => return Err(Error::from_io("Unable to read from socket", e)),
            }
            // multifd_recv_unfill_packet_header()
            let (magic, version, flags) = (be32(&hdr, 0), be32(&hdr, 4), be32(&hdr, 8));
            if magic != MULTIFD_MAGIC {
                bail!("multifd: received packet magic {:x}, expected {:x}", magic, MULTIFD_MAGIC);
            }
            if version != MULTIFD_VERSION {
                bail!("multifd: received packet version {}, expected {}", version, MULTIFD_VERSION);
            }
            let len = if flags & FLAG_DEVICE_STATE != 0 { DEVICE_STATE_LEN } else { PACKET_LEN };
            self.packet.resize(len, 0);
            self.packet[..HDR_LEN].copy_from_slice(&hdr);
            match read_full(&mut self.sock, &mut self.packet[HDR_LEN..len]) {
                Ok(0) => bail!("multifd: unexpected EOF after packet header"),
                Ok(n) if n == len - HDR_LEN => {}
                Ok(_) => bail!("Unexpected end-of-file before all data were read"),
                Err(e) => return Err(Error::from_io("Unable to read from socket", e)),
            }
            if flags & FLAG_DEVICE_STATE != 0 {
                self.device_state(cfg)?;
                if flags & FLAG_SYNC != 0 {
                    bail!("multifd: received SYNC device state packet");
                }
                continue;
            }
            let mut p = self.unfill(cfg)?;
            p.flags &= !FLAG_SYNC;
            if !p.normal.is_empty() || !p.zero.is_empty() {
                self.ram_recv(cfg, &p)?;
            }
            if flags & FLAG_SYNC != 0 {
                let mut s = lock(&shared.state);
                s.synced += 1;
                let round = s.release[usize::from(self.id)];
                shared.cond.notify_all();
                while s.release[usize::from(self.id)] == round && !s.exiting {
                    s = shared.cond.wait(s).unwrap_or_else(|e| e.into_inner());
                }
            }
        }
    }

    /// `multifd_device_state_recv()`. No entry here has `load_state_buffer`, so a packet with
    /// data is an error once it was read.
    fn device_state(&mut self, cfg: &RecvConfig) -> Result<()> {
        let p = &self.packet;
        let instance_id = be32(p, HDR_LEN + 256);
        let size = be32(p, HDR_LEN + 260);
        if size == 0 {
            bail!("multifd: received empty device state packet");
        }
        let got = io::copy(&mut (&mut self.sock).take(u64::from(size)), &mut io::sink())
            .map_err(|e| Error::from_io("Unable to read from socket", e))?;
        if got != u64::from(size) {
            bail!("Unexpected end-of-file before all data were read");
        }
        let idstr = &p[HDR_LEN..HDR_LEN + 256];
        if idstr[255] != 0 {
            bail!("unterminated multifd device state idstr");
        }
        let end = idstr.iter().position(|&b| b == 0).unwrap_or(255);
        let idstr = String::from_utf8_lossy(&idstr[..end]).into_owned();
        if !cfg.entries.iter().any(|(id, inst)| *id == idstr && *inst == instance_id) {
            bail!("Unknown idstr {} or instance id {} for load state buffer", idstr, instance_id);
        }
        bail!("idstr {} / instance {} has no load state buffer operation", idstr, instance_id)
    }

    /// `multifd_ram_unfill_packet()`.
    fn unfill(&self, cfg: &RecvConfig) -> Result<RamPacket> {
        let p = &self.packet;
        let flags = be32(p, 8);
        let next_packet_size = be32(p, OFF_NEXT_PACKET_SIZE);
        let pages_alloc = be32(p, OFF_PAGES_ALLOC);
        if pages_alloc as usize > PAGE_COUNT {
            bail!("multifd: received packet with {} pages, expected {}", pages_alloc, PAGE_COUNT);
        }
        let normal = be32(p, OFF_NORMAL_PAGES);
        if normal > pages_alloc {
            bail!(
                "multifd: received packet with {} non-zero pages, which exceeds maximum expected \
                 pages {}",
                normal,
                pages_alloc
            );
        }
        let zero = be32(p, OFF_ZERO_PAGES);
        if zero > pages_alloc - normal {
            bail!(
                "multifd: received packet with {} zero pages, expected maximum {}",
                zero,
                pages_alloc - normal
            );
        }
        let mut out = RamPacket {
            flags,
            next_packet_size,
            target: None,
            normal: Vec::new(),
            zero: Vec::new(),
        };
        if normal == 0 && zero == 0 {
            return Ok(out);
        }
        let raw = &p[OFF_RAMBLOCK..OFF_RAMBLOCK + 255];
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let name = String::from_utf8_lossy(&raw[..end]);
        let target = match cfg.blocks.iter().position(|b| b.name() == name) {
            Some(i) => Target::Block(i),
            None if cfg.droppable.iter().any(|d| name.starts_with(d.as_str())) => Target::Dropped,
            None => bail!("multifd: unknown ram block {}", name),
        };
        let used_length = match target {
            Target::Block(i) => Some(cfg.blocks[i].len()),
            Target::Dropped => None,
        };
        for i in 0..(normal + zero) as usize {
            let offset = be64(p, OFF_OFFSET + 8 * i);
            if let Some(len) = used_length {
                if offset > len.saturating_sub(PAGE_SIZE as u64) {
                    bail!("multifd: offset too long {} (max {:x})", offset, len);
                }
            }
            if i < normal as usize {
                out.normal.push(offset);
            } else {
                out.zero.push(offset);
            }
        }
        out.target = Some(target);
        Ok(out)
    }

    /// `recv` of the compression method.
    fn ram_recv(&mut self, cfg: &RecvConfig, p: &RamPacket) -> Result<()> {
        let expected = if self.zlib.is_some() { FLAG_ZLIB } else { FLAG_NOCOMP };
        let flags = p.flags & FLAG_COMPRESSION_MASK;
        if flags != expected {
            bail!("multifd {}: flags received {:x} flags expected {:x}", self.id, flags, expected);
        }
        let block = match p.target {
            Some(Target::Block(i)) => Some(&cfg.blocks[i]),
            _ => None,
        };
        let werr = |b: &RamBlock, o: u64, e: ruvm_mem::MemError| {
            Error::generic(format!("multifd: failed to write page {o:x} of {}: {e}", b.name()))
        };
        // multifd_recv_zero_page_process(): a page that is already zero is left alone, so the
        // host does not have to back it.
        if let Some(b) = block {
            for &o in &p.zero {
                let write = cfg.postcopy_ram
                    || !b.is_zero(o, PAGE_SIZE as u64).map_err(|e| werr(b, o, e))?;
                if write {
                    b.fill(o, PAGE_SIZE as u64, 0).map_err(|e| werr(b, o, e))?;
                }
            }
        }
        if p.normal.is_empty() {
            return Ok(());
        }
        let total = p.normal.len() * PAGE_SIZE;
        self.data.resize(total, 0);
        match self.zlib.as_mut() {
            None => read_all(&mut self.sock, &mut self.data)?,
            Some(z) => {
                // multifd_zlib_recv()
                let in_size = p.next_packet_size as usize;
                if in_size > 2 * PACKET_SIZE {
                    bail!(
                        "multifd {}: packet size received {} is bigger than {}",
                        self.id,
                        in_size,
                        2 * PACKET_SIZE
                    );
                }
                self.zbuf.resize(in_size, 0);
                read_all(&mut self.sock, &mut self.zbuf)?;
                let start_out = z.total_out();
                let mut used = 0;
                for i in 0..p.normal.len() {
                    let flush = if i + 1 == p.normal.len() {
                        FlushDecompress::Sync
                    } else {
                        FlushDecompress::None
                    };
                    let page = &mut self.data[i * PAGE_SIZE..(i + 1) * PAGE_SIZE];
                    let mut made = 0;
                    let ret = loop {
                        let (in0, out0) = (z.total_in(), z.total_out());
                        let r = z.decompress(&self.zbuf[used..], &mut page[made..], flush);
                        used += (z.total_in() - in0) as usize;
                        made += (z.total_out() - out0) as usize;
                        if !matches!(r, Ok(Status::Ok)) || used == in_size || made == PAGE_SIZE {
                            break r;
                        }
                    };
                    if matches!(ret, Ok(Status::Ok)) && made < PAGE_SIZE {
                        bail!("multifd {}: inflate generated too few output", self.id);
                    }
                    if !matches!(ret, Ok(Status::Ok)) {
                        bail!(
                            "multifd {}: inflate returned {} instead of Z_OK",
                            self.id,
                            zcode(&ret, -3)
                        );
                    }
                }
                let out_size = z.total_out() - start_out;
                if out_size != total as u64 {
                    bail!(
                        "multifd {}: packet size received {} size expected {}",
                        self.id,
                        out_size,
                        total
                    );
                }
            }
        }
        if let Some(b) = block {
            for (i, &o) in p.normal.iter().enumerate() {
                b.write(o, &self.data[i * PAGE_SIZE..(i + 1) * PAGE_SIZE])
                    .map_err(|e| werr(b, o, e))?;
            }
        }
        Ok(())
    }
}

/// `multifd_recv_thread()` on a mapped-ram file: reads the ranges it is given into RAM.
fn file_recv_thread(id: u8, file: &File, shared: &RecvShared) {
    let i = usize::from(id);
    let mut buf = AlignedBuf::default();
    loop {
        let job = {
            let mut s = lock(&shared.state);
            loop {
                if s.exiting {
                    return;
                }
                if let Some(j) = s.jobs[i].take() {
                    s.busy[i] = true;
                    break j;
                }
                s = shared.cond.wait(s).unwrap_or_else(|e| e.into_inner());
            }
        };
        // multifd_file_recv_data()
        let data = buf.get(job.size);
        let res = read_exact_at(file, data, job.file_offset).and_then(|()| {
            job.block.write(job.offset, data).map_err(|e| {
                Error::generic(format!(
                    "multifd: failed to write page {:x} of {}: {e}",
                    job.offset,
                    job.block.name()
                ))
            })
        });
        if let Err(e) = res {
            shared.terminate(Some(&e.prepend(format!("multifd recv ({id}): "))));
            return;
        }
        lock(&shared.state).busy[i] = false;
        shared.cond.notify_all();
    }
}

impl MultifdRecv {
    fn new_shared(n: usize, ctl: Vec<Socket>, file: bool) -> Arc<Self> {
        let shared = Arc::new(RecvShared {
            state: Mutex::new(RecvState {
                jobs: (0..n).map(|_| None).collect(),
                busy: vec![false; n],
                next: 0,
                synced: 0,
                release: vec![0; n],
                closed: 0,
                error: None,
                exiting: false,
            }),
            cond: Condvar::new(),
            ctl,
            channels: n,
            file,
        });
        Arc::new(MultifdRecv { shared, threads: Mutex::new(Vec::new()) })
    }

    /// `multifd_recv_setup()` and `multifd_recv_new_channel()`: starts a thread on each channel,
    /// `channels[id]` being the one that announced `id`.
    pub fn start(channels: Vec<Socket>, cfg: RecvConfig) -> Result<Arc<Self>> {
        let n = channels.len();
        let mut ctl = Vec::with_capacity(n);
        for s in &channels {
            ctl.push(s.try_clone().map_err(|e| Error::from_io("multifd: socket", e))?);
        }
        let me = Self::new_shared(n, ctl, false);
        let shared = me.shared.clone();
        let cfg = Arc::new(cfg);
        for (i, sock) in channels.into_iter().enumerate() {
            let mut ch = RecvChannel {
                id: i as u8,
                sock,
                zlib: cfg.zlib.then(|| Decompress::new(true)),
                packet: Vec::with_capacity(PACKET_LEN),
                zbuf: Vec::new(),
                data: Vec::new(),
            };
            let (sh, c) = (shared.clone(), cfg.clone());
            let t = std::thread::Builder::new()
                .name(format!("mig/dst/recv_{i}"))
                .spawn(move || ch.run(&c, &sh));
            match t {
                Ok(t) => lock(&me.threads).push(t),
                Err(e) => {
                    me.shutdown();
                    return Err(Error::from_io("failed to create the multifd recv thread", e));
                }
            }
        }
        Ok(me)
    }

    /// [`start`](Self::start) for mapped-ram, over more handles on the migration file. The
    /// threads read what [`recv_file`](Self::recv_file) gives them.
    pub fn start_file(files: Vec<File>) -> Result<Arc<Self>> {
        let me = Self::new_shared(files.len(), Vec::new(), true);
        for (i, file) in files.into_iter().enumerate() {
            let sh = me.shared.clone();
            let t = std::thread::Builder::new()
                .name(format!("mig/dst/recv_{i}"))
                .spawn(move || file_recv_thread(i as u8, &file, &sh));
            match t {
                Ok(t) => lock(&me.threads).push(t),
                Err(e) => {
                    me.shutdown();
                    return Err(Error::from_io("failed to create the multifd recv thread", e));
                }
            }
        }
        Ok(me)
    }

    fn stopped(s: &RecvState) -> Result<()> {
        if let Some(e) = &s.error {
            return Err(Error::generic(e.clone()));
        }
        if s.exiting {
            bail!("multifd: the incoming migration is being stopped");
        }
        Ok(())
    }

    /// `multifd_recv()` with mapped-ram: has the next idle channel read `size` bytes at
    /// `file_offset` into `block` at `offset`.
    pub fn recv_file(
        &self,
        block: &Arc<RamBlock>,
        offset: u64,
        file_offset: u64,
        size: usize,
    ) -> Result<()> {
        let sh = &self.shared;
        let mut s = lock(&sh.state);
        loop {
            Self::stopped(&s)?;
            let n = sh.channels;
            let idle = |s: &RecvState, i: usize| s.jobs[i].is_none() && !s.busy[i];
            if let Some(i) = (0..n).map(|k| (s.next + k) % n).find(|&i| idle(&s, i)) {
                s.jobs[i] = Some(FileJob { block: block.clone(), offset, file_offset, size });
                s.next = (i + 1) % n;
                drop(s);
                sh.cond.notify_all();
                return Ok(());
            }
            s = wait_a_while(&sh.cond, s);
        }
    }

    /// `multifd_recv_sync_main()`: waits until every channel saw its sync packet, then lets
    /// them all go on. QEMU waits forever when a channel failed; this returns its error. With
    /// mapped-ram it waits until every channel is done with what it was given.
    pub fn sync_main(&self) -> Result<()> {
        let sh = &self.shared;
        let n = sh.channels;
        let mut s = lock(&sh.state);
        if sh.file {
            while !(0..n).all(|i| s.jobs[i].is_none() && !s.busy[i]) {
                Self::stopped(&s)?;
                s = wait_a_while(&sh.cond, s);
            }
            return Self::stopped(&s);
        }
        while s.synced < n {
            if let Some(e) = &s.error {
                return Err(Error::generic(e.clone()));
            }
            if s.exiting {
                bail!("multifd: the incoming migration is being stopped");
            }
            if s.closed > 0 {
                bail!("multifd: a channel closed before the sync");
            }
            s = sh.cond.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.synced -= n;
        for r in &mut s.release {
            *r += 1;
        }
        drop(s);
        sh.cond.notify_all();
        Ok(())
    }

    /// `multifd_recv_shutdown()` and `multifd_recv_cleanup()`.
    pub fn shutdown(&self) {
        self.shared.terminate(None);
        for t in lock(&self.threads).drain(..) {
            let _ = t.join();
        }
    }

    /// The first error of a channel.
    pub fn error(&self) -> Option<Error> {
        lock(&self.shared.state).error.clone().map(Error::generic)
    }
}

/// The channels of one incoming migration.
pub struct IncomingChannels {
    /// The main stream.
    pub main: Incoming,
    /// Another handle on the main stream's socket, for the return path.
    pub socket: Option<Socket>,
    /// The multifd channels, by id.
    pub multifd: Vec<Socket>,
}

impl std::fmt::Debug for IncomingChannels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IncomingChannels")
            .field("socket", &self.socket)
            .field("multifd", &self.multifd.len())
            .finish()
    }
}

/// `migration_ioc_process_incoming()`: accepts connections until the main stream and, with
/// `multifd` set to `(channels, uuid)`, every multifd channel are in, telling them apart by their
/// first word.
pub fn accept_channels(
    listener: &mut Listener,
    multifd: Option<(u8, [u8; 16])>,
) -> Result<IncomingChannels> {
    let n = multifd.map_or(0, |m| usize::from(m.0));
    let mut main: Option<(Incoming, Option<Socket>)> = None;
    let mut chans: Vec<Option<Socket>> = (0..n).map(|_| None).collect();
    let mut count = 0;
    while main.is_none() || count < n {
        let mut s = match listener.accept_next()? {
            Accepted::Stream(r) => {
                if main.is_some() {
                    bail!("non-peekable channel used without multifd");
                }
                main = Some((r, None));
                continue;
            }
            Accepted::Socket(s) => s,
        };
        // migration_channel_read_peek(), which reads here and puts the word back for the main
        // stream.
        let mut magic = [0u8; 4];
        match read_full(&mut s, &mut magic) {
            Ok(4) => {}
            Ok(_) => bail!("Failed to peek at channel"),
            Err(e) => return Err(Error::from_io("Failed to peek at channel", e)),
        }
        match (u32::from_be_bytes(magic), multifd) {
            (QEMU_VM_FILE_MAGIC, _) if main.is_none() => {
                let back = s.try_clone().ok();
                main = Some((Box::new(io::Cursor::new(magic).chain(s)), back));
            }
            (MULTIFD_MAGIC, Some((channels, uuid))) => {
                let prefix = format!("failed to receive packet via multifd channel {count}: ");
                let mut init = [0u8; INIT_LEN];
                init[..4].copy_from_slice(&magic);
                read_all(&mut s, &mut init[4..]).map_err(|e| e.prepend(&prefix))?;
                let id = parse_init(&init, &uuid, channels).map_err(|e| e.prepend(&prefix))?;
                let slot = &mut chans[usize::from(id)];
                if slot.is_some() {
                    bail!("multifd: received id '{}' already setup'", id);
                }
                *slot = Some(s);
                count += 1;
            }
            (m, _) => bail!("unknown channel magic: {}", m),
        }
    }
    let Some((main, socket)) = main else { unreachable!("the loop ends with the main channel") };
    Ok(IncomingChannels { main, socket, multifd: chans.into_iter().flatten().collect() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{TcpListener, TcpStream};

    fn pairs(n: usize) -> (Vec<Socket>, Vec<Socket>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let mut a = Vec::new();
        let mut b = Vec::new();
        for _ in 0..n {
            a.push(Socket::Tcp(TcpStream::connect(addr).unwrap()));
            b.push(Socket::Tcp(l.accept().unwrap().0));
        }
        (a, b)
    }

    fn round_trip(compression: Compression) {
        const PAGES: u64 = 700;
        let src = Arc::new(RamBlock::new("pc.ram", PAGES << 12, 12).unwrap());
        let dst = Arc::new(RamBlock::new("pc.ram", PAGES << 12, 12).unwrap());
        for p in 0..PAGES {
            if p % 4 != 0 {
                src.fill(p << 12, 4096, (p % 251) as u8 + 1).unwrap();
            }
            // Stale data the zero pages have to clear.
            dst.fill(p << 12, 4096, 0xee).unwrap();
        }
        let (tx, mut rx) = pairs(3);
        let uuid = [7u8; 16];
        let params = MultifdParams { channels: 3, compression, zero_pages: true, uuid };
        let stats = Arc::new(RamStats::default());
        let send = MultifdSend::start(tx, &params, stats.clone(), Arc::new(AtomicBool::new(false)))
            .unwrap();
        // Each channel announces itself first.
        let mut ids = Vec::new();
        for s in &mut rx {
            let mut init = [0u8; INIT_LEN];
            read_all(s, &mut init).unwrap();
            ids.push(parse_init(&init, &uuid, 3).unwrap());
        }
        assert_eq!(ids, [0, 1, 2]);
        let cfg = RecvConfig {
            blocks: vec![dst.clone()],
            zlib: compression != Compression::None,
            ..Default::default()
        };
        let recv = MultifdRecv::start(rx, cfg).unwrap();
        for round in 0..2 {
            for p in 0..PAGES {
                send.queue_page(&src, None, p << 12).unwrap();
            }
            send.flush_and_sync().unwrap();
            recv.sync_main().unwrap();
            if round == 0 {
                // A zero page that is not one any more.
                src.fill(4 << 12, 4096, 0x55).unwrap();
            }
        }
        send.shutdown();
        recv.shutdown();
        assert!(recv.error().is_none());
        let (mut a, mut b) = (vec![0; (PAGES << 12) as usize], vec![0; (PAGES << 12) as usize]);
        src.read(0, &mut a).unwrap();
        dst.read(0, &mut b).unwrap();
        assert!(a == b);
        let zero = PAGES.div_ceil(4);
        assert_eq!(stats.duplicate.load(Ordering::Relaxed), 2 * zero - 1);
        assert_eq!(stats.normal.load(Ordering::Relaxed), 2 * PAGES - 2 * zero + 1);
        assert!(stats.multifd_bytes.load(Ordering::Relaxed) > 3 * INIT_LEN as u64);
    }

    #[test]
    fn nocomp_round_trip() {
        round_trip(Compression::None);
    }

    #[test]
    fn zlib_round_trip() {
        round_trip(Compression::Zlib(1));
    }

    #[test]
    fn bad_packets() {
        let uuid = [0u8; 16];
        let mut init = init_packet(&uuid, 4);
        assert_eq!(
            parse_init(&init, &uuid, 2).unwrap_err().message(),
            "multifd: received channel id 4 exceeds channel count 2"
        );
        assert_eq!(
            parse_init(&init, &[1; 16], 8).unwrap_err().message(),
            "multifd: received uuid '00000000-0000-0000-0000-000000000000' and expected uuid \
             '01010101-0101-0101-0101-010101010101' for channel 4"
        );
        init[0] = 0;
        assert_eq!(
            parse_init(&init, &uuid, 8).unwrap_err().message(),
            "multifd: received packet magic 223344 expected 11223344"
        );

        // A packet for a block this side does not have fails the channel.
        let dst = Arc::new(RamBlock::new("pc.ram", 4 << 12, 12).unwrap());
        let (mut tx, rx) = pairs(1);
        let cfg = RecvConfig { blocks: vec![dst], ..Default::default() };
        let recv = MultifdRecv::start(rx, cfg).unwrap();
        let mut p = vec![0u8; PACKET_LEN];
        fill_packet(&mut p, FLAG_SYNC, 0, 0, Some(("vga.vram", 0, &[0x1000])));
        tx[0].write_all(&p).unwrap();
        assert_eq!(recv.sync_main().unwrap_err().message(), "multifd: unknown ram block vga.vram");
        recv.shutdown();

        let dst = Arc::new(RamBlock::new("pc.ram", 4 << 12, 12).unwrap());
        let (mut tx, rx) = pairs(1);
        let cfg = RecvConfig { blocks: vec![dst], ..Default::default() };
        let recv = MultifdRecv::start(rx, cfg).unwrap();
        fill_packet(&mut p, FLAG_SYNC, 0, 0, Some(("pc.ram", 0, &[0x4000])));
        tx[0].write_all(&p).unwrap();
        assert_eq!(
            recv.sync_main().unwrap_err().message(),
            "multifd: offset too long 16384 (max 4000)"
        );
        recv.shutdown();
    }
}
