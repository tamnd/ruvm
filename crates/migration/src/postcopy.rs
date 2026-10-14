// SPDX-License-Identifier: GPL-2.0-or-later

//! Postcopy live migration and the return path, migration/postcopy-ram.c and the return path
//! parts of migration/migration.c.
//!
//! With the `postcopy-ram` capability the source can switch over before all of RAM went out:
//! `migrate-start-postcopy` makes it stop the guest, tell the destination which pages it has to
//! throw away again (`MIG_CMD_POSTCOPY_RAM_DISCARD`), and send the device state in one
//! `MIG_CMD_PACKAGED` blob. The destination starts the guest on that and fetches every page the
//! guest touches before it arrived. On Linux it registers its RAM with userfaultfd, so that such
//! an access sleeps until the page is there, and a fault thread asks the source for the page on
//! the return path (`MIG_RP_MSG_REQ_PAGES`). Meanwhile the source keeps sending the rest in the
//! background and puts the requested pages first.
//!
//! The return path is the second direction of a socket channel. Besides page requests it
//! carries `PONG` replies to the source's `PING`s and the final `SHUT`.
//!
//! [`ReturnPath`] is the destination end of it, [`PageRequests`] the queue of requested pages on
//! the source, and [`source_return_path`] the source thread that reads the messages.
//!
//! With the `postcopy-blocktime` capability the destination also keeps [`Blocktime`], how long
//! its vCPUs waited for pages, which `query-migrate` reports once the migration completed.

use std::collections::{HashMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result, bail};

use crate::channel::Socket;

/// `enum mig_rp_message_type`: the messages on the return path.
pub mod rp {
    /// `MIG_RP_MSG_INVALID`.
    pub const INVALID: u16 = 0;
    /// `MIG_RP_MSG_SHUT`: the destination sends nothing more; nonzero data is an error.
    pub const SHUT: u16 = 1;
    /// `MIG_RP_MSG_PONG`: the answer to a `PING`, with its value.
    pub const PONG: u16 = 2;
    /// `MIG_RP_MSG_REQ_PAGES_ID`: a page request with the block name.
    pub const REQ_PAGES_ID: u16 = 3;
    /// `MIG_RP_MSG_REQ_PAGES`: a page request in the block of the request before.
    pub const REQ_PAGES: u16 = 4;
    /// `MIG_RP_MSG_RECV_BITMAP`: the received bitmap, for postcopy recovery.
    pub const RECV_BITMAP: u16 = 5;
    /// `MIG_RP_MSG_RESUME_ACK`: the destination is ready to resume, for postcopy recovery.
    pub const RESUME_ACK: u16 = 6;
    /// `MIG_RP_MSG_SWITCHOVER_ACK`.
    pub const SWITCHOVER_ACK: u16 = 7;
    /// `MIG_RP_MSG_MAX`.
    pub const MAX: u16 = 8;
}

/// `rp_cmd_args[]`: the name and fixed length, or -1, of each return path message.
const RP_CMD_ARGS: [(&str, i32); rp::MAX as usize + 1] = [
    ("INVALID", -1),
    ("SHUT", 4),
    ("PONG", 4),
    ("REQ_PAGES_ID", -1),
    ("REQ_PAGES", 12),
    ("RECV_BITMAP", -1),
    ("RESUME_ACK", 4),
    ("SWITCHOVER_ACK", 0),
    ("MAX", -1),
];

/// `QEMU_VM_PING_PACKAGED_LOADED`: the `PING` at the end of the postcopy package, whose `PONG`
/// tells the source that the destination loaded the device state.
pub const PING_PACKAGED_LOADED: u32 = 0x42;

/// `MAX_VM_CMD_PACKAGED_SIZE`.
pub const MAX_PACKAGED_SIZE: usize = 1 << 24;

/// `MIGRATION_RESUME_ACK_VALUE`.
const RESUME_ACK_VALUE: u32 = 1;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The userfaultfd wrapper postcopy uses: the real one on Linux, and elsewhere one that cannot
/// be created, so that the rest of the code is the same everywhere.
pub(crate) mod uffd {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(crate) use ruvm_sys::userfaultfd::Userfaultfd as Uffd;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(crate) use ruvm_sys::userfaultfd::{FEATURE_PAGEFAULT_FLAG_WP, FEATURE_THREAD_ID};

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub(crate) use stub::Uffd;

    /// `UFFD_FEATURE_PAGEFAULT_FLAG_WP`.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub(crate) const FEATURE_PAGEFAULT_FLAG_WP: u64 = 1;

    /// `UFFD_FEATURE_THREAD_ID`.
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    pub(crate) const FEATURE_THREAD_ID: u64 = 1 << 8;

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    mod stub {
        use std::convert::Infallible;
        use std::io;

        use ruvm_sys::HostMemory;

        /// A userfaultfd on a host without one: [`Uffd::new`] always fails.
        #[derive(Debug)]
        pub(crate) struct Uffd(Infallible);

        impl Uffd {
            pub(crate) fn new() -> io::Result<Self> {
                Err(io::Error::from(io::ErrorKind::Unsupported))
            }

            pub(crate) fn with_features(_: u64) -> io::Result<Self> {
                Self::new()
            }

            pub(crate) fn features() -> io::Result<u64> {
                Err(io::Error::from(io::ErrorKind::Unsupported))
            }

            pub(crate) fn register_write_protect(&self, _: &HostMemory) -> io::Result<bool> {
                match self.0 {}
            }

            pub(crate) fn write_protect(
                &self,
                _: &HostMemory,
                _: usize,
                _: usize,
                _: bool,
            ) -> io::Result<()> {
                match self.0 {}
            }

            pub(crate) fn register(&self, _: &HostMemory) -> io::Result<()> {
                match self.0 {}
            }

            pub(crate) fn unregister(&self, _: &HostMemory) -> io::Result<()> {
                match self.0 {}
            }

            pub(crate) fn copy(&self, _: &HostMemory, _: usize, _: &[u8]) -> io::Result<()> {
                match self.0 {}
            }

            pub(crate) fn zeropage(&self, _: &HostMemory, _: usize, _: usize) -> io::Result<()> {
                match self.0 {}
            }

            pub(crate) fn wait_fault(&self, _: i32) -> io::Result<Option<usize>> {
                match self.0 {}
            }

            pub(crate) fn wait_fault_thread(&self, _: i32) -> io::Result<Option<(usize, u32)>> {
                match self.0 {}
            }
        }
    }
}

/// `postcopy_ram_supported_by_host()`: whether this host can take a postcopy migration, which
/// needs userfaultfd.
pub fn supported_by_host() -> Result<()> {
    uffd::Uffd::new().map(drop).map_err(|e| {
        if e.kind() == io::ErrorKind::Unsupported {
            Error::generic("postcopy_ram_supported_by_host: No OS support")
        } else {
            Error::from_io("Userfaultfd not available", e)
        }
    })
}

/// The destination end of the return path, `MigrationIncomingState.to_src_file`.
#[derive(Debug)]
pub struct ReturnPath {
    out: Mutex<Socket>,
    // A second handle for shutting the channel down while a thread writes or reads on it.
    ctl: Socket,
}

impl ReturnPath {
    /// The return path over the other direction of the incoming socket.
    pub fn new(sock: Socket) -> io::Result<Self> {
        let ctl = sock.try_clone()?;
        Ok(ReturnPath { out: Mutex::new(sock), ctl })
    }

    /// `migrate_send_rp_message()`.
    pub fn send(&self, ty: u16, data: &[u8]) -> Result<()> {
        let mut msg = Vec::with_capacity(4 + data.len());
        msg.extend_from_slice(&ty.to_be_bytes());
        msg.extend_from_slice(&(data.len() as u16).to_be_bytes());
        msg.extend_from_slice(data);
        lock(&self.out).write_all(&msg).map_err(|e| Error::from_io("Unable to write to socket", e))
    }

    /// `migrate_send_rp_pong()`.
    pub fn pong(&self, value: u32) -> Result<()> {
        self.send(rp::PONG, &value.to_be_bytes())
    }

    /// `migrate_send_rp_shut()`.
    pub fn shut(&self, value: u32) -> Result<()> {
        self.send(rp::SHUT, &value.to_be_bytes())
    }

    /// `migrate_send_rp_message_req_pages()`: asks for `len` bytes at `start` of the block
    /// `name`, or of the block of the last request when `name` is `None`.
    pub fn req_pages(&self, name: Option<&str>, start: u64, len: u32) -> Result<()> {
        let mut data = Vec::with_capacity(13 + name.map_or(0, str::len));
        data.extend_from_slice(&start.to_be_bytes());
        data.extend_from_slice(&len.to_be_bytes());
        match name {
            Some(n) => {
                data.push(n.len() as u8);
                data.extend_from_slice(n.as_bytes());
                self.send(rp::REQ_PAGES_ID, &data)
            }
            None => self.send(rp::REQ_PAGES, &data),
        }
    }

    /// `qemu_file_shutdown()`, which wakes up a thread blocked on the channel.
    pub fn shutdown(&self) {
        self.ctl.shutdown();
    }
}

#[derive(Debug, Default)]
struct Queue {
    pages: VecDeque<(usize, u64, u64)>,
    // `RAMState.last_req_rb`.
    last: Option<usize>,
    kicked: bool,
}

/// The pages the destination asked for, `RAMState.src_page_requests`, which the RAM saver sends
/// before anything else.
#[derive(Debug)]
pub struct PageRequests {
    blocks: Vec<(String, u64)>,
    queue: Mutex<Queue>,
    cond: Condvar,
    count: AtomicU64,
}

impl PageRequests {
    /// The queue for RAM blocks with these names and lengths, in the order the saver numbers
    /// them.
    pub fn new(blocks: Vec<(String, u64)>) -> Self {
        PageRequests {
            blocks,
            queue: Mutex::new(Queue::default()),
            cond: Condvar::new(),
            count: AtomicU64::new(0),
        }
    }

    /// Forgets everything, at the start of a migration.
    pub fn reset(&self) {
        *lock(&self.queue) = Queue::default();
        self.count.store(0, Ordering::Relaxed);
    }

    /// `postcopy_requests` of `query-migrate`.
    pub fn requests(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// `ram_save_queue_pages()`.
    pub fn queue(&self, name: Option<&str>, start: u64, len: u64) -> Result<()> {
        self.count.fetch_add(1, Ordering::Relaxed);
        let mut q = lock(&self.queue);
        let b = match name {
            None => match q.last {
                Some(b) => b,
                None => bail!("MIG_RP_MSG_REQ_PAGES has no previous block"),
            },
            Some(n) => match self.blocks.iter().position(|(name, _)| name == n) {
                Some(b) => {
                    q.last = Some(b);
                    b
                }
                None => bail!("MIG_RP_MSG_REQ_PAGES has no block '{}'", n),
            },
        };
        let blocklen = self.blocks[b].1;
        if start.wrapping_add(len).wrapping_sub(1) >= blocklen {
            bail!(
                "MIG_RP_MSG_REQ_PAGES request overrun, start={:x} len={:x} blocklen={:x}",
                start,
                len,
                blocklen
            );
        }
        q.pages.push_back((b, start, len));
        drop(q);
        self.cond.notify_all();
        Ok(())
    }

    /// `unqueue_page()`: the oldest request, as the block index, offset and length.
    pub fn pop(&self) -> Option<(usize, u64, u64)> {
        lock(&self.queue).pages.pop_front()
    }

    /// Sleeps up to `timeout`, less if a request comes in or [`kick`](Self::kick) is called.
    pub fn wait(&self, timeout: Duration) {
        let q = lock(&self.queue);
        if !q.pages.is_empty() || q.kicked {
            drop(q);
            lock(&self.queue).kicked = false;
            return;
        }
        let (mut q, _) = self
            .cond
            .wait_timeout_while(q, timeout, |q| q.pages.is_empty() && !q.kicked)
            .unwrap_or_else(|e| e.into_inner());
        q.kicked = false;
    }

    /// `migration_rp_kick()`: wakes up [`wait`](Self::wait).
    pub fn kick(&self) {
        lock(&self.queue).kicked = true;
        self.cond.notify_all();
    }
}

/// What the source return path thread shares with the migration thread, `rp_state`.
#[derive(Debug, Default)]
pub struct SourceRp {
    /// `postcopy_package_loaded`: the destination answered the `PING` at the end of the
    /// package.
    pub package_loaded: AtomicBool,
    /// Bumped on every `PONG`, `rp_pong_acks`.
    pub pongs: AtomicU64,
    /// `switchover_ack_pending_num`: the acknowledgements the destination still owes before
    /// the source may switch over.
    pub switchover_ack_pending: AtomicU32,
}

/// Reads exactly `buf.len()` bytes, or as many as come before the end of the stream. Returns
/// the count.
fn read_full(r: &mut impl Read, buf: &mut [u8]) -> io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(n) => got += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(got)
}

/// `source_return_path_thread()`: reads the return path until `SHUT`, the end of the channel
/// or `running` turning false. Page requests go to `requests`; `wake` is called whenever the
/// migration thread may have something new to look at.
pub fn source_return_path(
    sock: &mut impl Read,
    state: &SourceRp,
    requests: Option<&PageRequests>,
    running: &dyn Fn() -> bool,
) -> Result<()> {
    let mut buf = [0u8; 512];
    while running() {
        let mut head = [0u8; 4];
        match read_full(sock, &mut head) {
            Ok(4) => {}
            // The channel ended without an error: the destination has gone.
            Ok(_) => return Ok(()),
            Err(e) => return Err(Error::from_io("Unable to read from socket", e)),
        }
        let ty = u16::from_be_bytes([head[0], head[1]]);
        let len = u16::from_be_bytes([head[2], head[3]]);
        if ty >= rp::MAX || ty == rp::INVALID {
            bail!("Received invalid message 0x{:04x} length 0x{:04x}", ty, len);
        }
        let (name, want) = RP_CMD_ARGS[usize::from(ty)];
        if (want != -1 && i32::from(len) != want) || usize::from(len) > buf.len() {
            // QEMU prints the expected length as a size_t, so -1 comes out as its maximum.
            bail!(
                "Received '{}' message (0x{:04x}) withincorrect length {} expecting {}",
                name,
                ty,
                len,
                want as isize as usize
            );
        }
        let len = usize::from(len);
        let got = read_full(sock, &mut buf[..len])
            .map_err(|e| Error::from_io("Unable to read from socket", e))?;
        if got != len {
            bail!("Failed reading data for message 0x{:04x} read {} expected {}", ty, got, len);
        }
        let be32 = |b: &[u8]| u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        let be64 = |b: &[u8]| u64::from_be_bytes(b[..8].try_into().expect("8 bytes"));
        match ty {
            rp::SHUT => {
                let sibling_error = be32(&buf);
                if sibling_error != 0 {
                    bail!("Sibling indicated error {}", sibling_error);
                }
                return Ok(());
            }
            rp::PONG => {
                state.pongs.fetch_add(1, Ordering::Relaxed);
                if be32(&buf) == PING_PACKAGED_LOADED {
                    state.package_loaded.store(true, Ordering::Release);
                    if let Some(r) = requests {
                        r.kick();
                    }
                }
            }
            rp::REQ_PAGES => {
                handle_req_pages(requests, None, be64(&buf), be32(&buf[8..]))?;
            }
            rp::REQ_PAGES_ID => {
                let mut expected = 13;
                let mut req = None;
                if len >= expected {
                    let n = usize::from(buf[12]);
                    expected += n;
                    req = Some((be64(&buf), be32(&buf[8..]), n));
                }
                if len != expected {
                    bail!("Req_Page_id with length {} expecting {}", len, expected);
                }
                let (start, rlen, n) = req.expect("checked above");
                let id = String::from_utf8_lossy(&buf[13..13 + n]).into_owned();
                handle_req_pages(requests, Some(&id), start, rlen)?;
            }
            rp::RECV_BITMAP => {
                if len < 1 {
                    bail!("MIG_RP_MSG_RECV_BITMAP missing block name");
                }
                // ram_dirty_bitmap_reload(): only in postcopy-recover, which this side never
                // enters.
                bail!("Reload bitmap in incorrect state postcopy-active");
            }
            rp::RESUME_ACK => {
                let v = be32(&buf);
                if v != RESUME_ACK_VALUE {
                    bail!("illegal resume_ack value {}", v);
                }
            }
            rp::SWITCHOVER_ACK => {
                let left = state.switchover_ack_pending.fetch_sub(1, Ordering::AcqRel);
                if left == 0 {
                    bail!("Switchover ack pending num underflowed");
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// `migrate_handle_rp_req_pages()`.
fn handle_req_pages(
    requests: Option<&PageRequests>,
    name: Option<&str>,
    start: u64,
    len: u32,
) -> Result<()> {
    const HOST_PAGE: u64 = 4096;
    if start % HOST_PAGE != 0 || u64::from(len) % HOST_PAGE != 0 {
        bail!("MIG_RP_MSG_REQ_PAGES: Misaligned page request, start:{:x} len: {}", start, len);
    }
    match requests {
        Some(r) => r.queue(name, start, u64::from(len)),
        None => bail!("MIG_RP_MSG_REQ_PAGES has no previous block"),
    }
}

/// `BLOCKTIME_LATENCY_BUCKET_N`: bucket `n` of the latency distribution counts the faults that
/// took from 2^n to 2^(n+1) microseconds, and the last one also all the longer ones.
pub const LATENCY_BUCKETS: usize = 24;

/// `PostcopyBlocktimeContext`: how long the threads of the destination waited for the pages
/// they faulted on in postcopy. The times are host nanoseconds, `QEMU_CLOCK_REALTIME`.
#[derive(Debug)]
pub struct Blocktime {
    /// `tid_to_vcpu_hash`: the vCPU index of each vCPU thread.
    vcpus: HashMap<u32, usize>,
    /// `smp.cpus`.
    cpus: usize,
    start: Instant,
    /// `page_request_mutex`, which also covers the received bitmap the faults are checked
    /// against.
    inner: Mutex<BlocktimeInner>,
}

#[derive(Debug)]
struct BlocktimeInner {
    /// `vcpu_blocktime_total`.
    vcpu_total: Vec<u64>,
    /// `vcpu_faults_count`.
    vcpu_faults: Vec<u64>,
    /// `vcpu_faults_current`: the faults of each vCPU still waiting.
    vcpu_waiting: Vec<u8>,
    /// `vcpu_addr_hash`: the faults waiting on each page, oldest first, with the vCPU (`None`
    /// for another thread) and the time.
    pages: HashMap<usize, Vec<(Option<usize>, u64)>>,
    /// `latency_buckets`.
    buckets: [u64; LATENCY_BUCKETS],
    /// `total_blocktime`: the time all vCPUs waited at once.
    total: u64,
    /// `last_begin`.
    last_begin: u64,
    /// `smp_cpus_down`.
    cpus_down: usize,
    /// `non_vcpu_faults`.
    other_faults: u64,
    /// `non_vcpu_blocktime_total`.
    other_total: u64,
}

/// The `postcopy-*` members of `MigrationInfo` that `fill_destination_postcopy_migration_info()`
/// sets. The blocktimes are in milliseconds and the latencies in nanoseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BlocktimeInfo {
    /// `postcopy-blocktime`.
    pub blocktime: u32,
    /// `postcopy-vcpu-blocktime`.
    pub vcpu_blocktime: Vec<u32>,
    /// `postcopy-latency`.
    pub latency: u64,
    /// `postcopy-latency-dist`.
    pub latency_dist: Vec<u64>,
    /// `postcopy-vcpu-latency`.
    pub vcpu_latency: Vec<u64>,
    /// `postcopy-non-vcpu-latency`.
    pub non_vcpu_latency: u64,
}

/// `SCALE_MS`.
const NS_PER_MS: u64 = 1_000_000;

impl Blocktime {
    /// `blocktime_context_new()` for vCPUs whose threads have the ids `thread_ids`, by vCPU
    /// index. Where vCPUs share a thread, the thread counts as the last of them, as in QEMU.
    pub fn new(thread_ids: &[i64]) -> Self {
        let cpus = thread_ids.len();
        Blocktime {
            vcpus: thread_ids.iter().enumerate().map(|(i, &t)| (t as u32, i)).collect(),
            cpus,
            start: Instant::now(),
            inner: Mutex::new(BlocktimeInner {
                vcpu_total: vec![0; cpus],
                vcpu_faults: vec![0; cpus],
                vcpu_waiting: vec![0; cpus],
                pages: HashMap::new(),
                buckets: [0; LATENCY_BUCKETS],
                total: 0,
                last_begin: 0,
                cpus_down: 0,
                other_faults: 0,
                other_total: 0,
            }),
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// The part of `migrate_send_rp_req_pages()` under `page_request_mutex`: if `missing`
    /// says the page at `addr` is not here yet, `mark_postcopy_blocktime_begin()` records the
    /// fault of thread `tid` (0 when the kernel does not say) and this returns true.
    pub fn begin(&self, addr: usize, tid: u32, missing: impl FnOnce() -> bool) -> bool {
        self.begin_at(addr, tid, missing, || self.now())
    }

    fn begin_at(
        &self,
        addr: usize,
        tid: u32,
        missing: impl FnOnce() -> bool,
        now: impl FnOnce() -> u64,
    ) -> bool {
        let mut g = lock(&self.inner);
        if !missing() {
            return false;
        }
        if tid == 0 {
            return true;
        }
        let now = now();
        let cpu = self.vcpus.get(&tid).copied();
        match cpu {
            Some(c) => {
                g.vcpu_faults[c] += 1;
                if g.vcpu_waiting[c] == 0 {
                    g.cpus_down += 1;
                    g.last_begin = now;
                }
                g.vcpu_waiting[c] = g.vcpu_waiting[c].saturating_add(1);
            }
            None => g.other_faults += 1,
        }
        g.pages.entry(addr).or_default().push((cpu, now));
        true
    }

    /// The page at `addr` is in place: `mark`, which sets it in the received bitmap, and
    /// `mark_postcopy_blocktime_end()`, both under `page_request_mutex`.
    pub fn end(&self, addr: usize, mark: impl FnOnce()) {
        self.end_at(addr, mark, || self.now());
    }

    fn end_at(&self, addr: usize, mark: impl FnOnce(), now: impl FnOnce() -> u64) {
        let mut g = lock(&self.inner);
        mark();
        let Some(faults) = g.pages.remove(&addr) else { return };
        let now = now();
        let mut freed = 0;
        // blocktime_cpu_list_iter_fn(), newest first.
        for (cpu, at) in faults.into_iter().rev() {
            let passed = now.saturating_sub(at);
            let us = passed / 1000;
            let bucket = if us == 0 { 0 } else { 63 - us.leading_zeros() as usize };
            g.buckets[bucket.min(LATENCY_BUCKETS - 1)] += 1;
            match cpu {
                Some(c) => {
                    g.vcpu_waiting[c] -= 1;
                    if g.vcpu_waiting[c] == 0 {
                        g.vcpu_total[c] += passed;
                        freed += 1;
                    }
                }
                None => g.other_total += passed,
            }
        }
        // If all vCPUs waited, the time they all did ends here.
        if g.cpus_down == self.cpus && freed > 0 {
            g.total += now.saturating_sub(g.last_begin);
        }
        g.cpus_down -= freed;
    }

    /// `fill_destination_postcopy_migration_info()`.
    pub fn info(&self) -> BlocktimeInfo {
        let g = lock(&self.inner);
        let (mut latency_total, mut faults) = (g.other_total, g.other_faults);
        let mut info = BlocktimeInfo::default();
        for (&total, &count) in g.vcpu_total.iter().zip(&g.vcpu_faults) {
            info.vcpu_blocktime.push((total / NS_PER_MS) as u32);
            info.vcpu_latency.push(total.checked_div(count).unwrap_or(0));
            latency_total += total;
            faults += count;
        }
        info.blocktime = (g.total / NS_PER_MS) as u32;
        info.latency = latency_total.checked_div(faults).unwrap_or(0);
        info.latency_dist = g.buckets.to_vec();
        info.non_vcpu_latency = g.other_total.checked_div(g.other_faults).unwrap_or(0);
        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(msgs: &[u8], requests: &PageRequests) -> Result<SourceRp> {
        let state = SourceRp::default();
        source_return_path(&mut &msgs[..], &state, Some(requests), &|| true)?;
        Ok(state)
    }

    fn msg(ty: u16, data: &[u8]) -> Vec<u8> {
        let mut m = ty.to_be_bytes().to_vec();
        m.extend_from_slice(&(data.len() as u16).to_be_bytes());
        m.extend_from_slice(data);
        m
    }

    fn req(start: u64, len: u32, name: Option<&str>) -> Vec<u8> {
        let mut d = start.to_be_bytes().to_vec();
        d.extend_from_slice(&len.to_be_bytes());
        match name {
            Some(n) => {
                d.push(n.len() as u8);
                d.extend_from_slice(n.as_bytes());
                msg(rp::REQ_PAGES_ID, &d)
            }
            None => msg(rp::REQ_PAGES, &d),
        }
    }

    fn requests() -> PageRequests {
        PageRequests::new(vec![("pc.ram".into(), 0x10000), ("vga.vram".into(), 0x4000)])
    }

    #[test]
    fn page_requests() {
        let r = requests();
        let mut m = msg(rp::PONG, &1u32.to_be_bytes());
        m.extend(req(0x2000, 0x1000, Some("vga.vram")));
        m.extend(req(0x3000, 0x1000, None));
        m.extend(msg(rp::PONG, &PING_PACKAGED_LOADED.to_be_bytes()));
        m.extend(msg(rp::SHUT, &0u32.to_be_bytes()));
        // Nothing after SHUT is read.
        m.extend(msg(0x99, &[]));
        let s = run(&m, &r).unwrap();
        assert!(s.package_loaded.load(Ordering::Relaxed));
        assert_eq!(s.pongs.load(Ordering::Relaxed), 2);
        assert_eq!(r.requests(), 2);
        assert_eq!(r.pop(), Some((1, 0x2000, 0x1000)));
        assert_eq!(r.pop(), Some((1, 0x3000, 0x1000)));
        assert_eq!(r.pop(), None);
    }

    #[test]
    fn errors_match_qemu() {
        let e = |m: Vec<u8>| run(&m, &requests()).err().unwrap().message().to_string();
        assert_eq!(e(msg(9, &[])), "Received invalid message 0x0009 length 0x0000");
        assert_eq!(
            e(msg(rp::PONG, &[0; 2])),
            "Received 'PONG' message (0x0002) withincorrect length 2 expecting 4"
        );
        assert_eq!(
            e(msg(rp::REQ_PAGES_ID, &[0; 513])),
            format!(
                "Received 'REQ_PAGES_ID' message (0x0003) withincorrect length 513 expecting {}",
                usize::MAX
            )
        );
        let mut short = msg(rp::PONG, &[0; 4]);
        short.truncate(6);
        assert_eq!(e(short), "Failed reading data for message 0x0002 read 2 expected 4");
        assert_eq!(e(msg(rp::SHUT, &1u32.to_be_bytes())), "Sibling indicated error 1");
        assert_eq!(e(req(0, 0x1000, None)), "MIG_RP_MSG_REQ_PAGES has no previous block");
        assert_eq!(e(req(0, 0x1000, Some("nope"))), "MIG_RP_MSG_REQ_PAGES has no block 'nope'");
        assert_eq!(
            e(req(0x10, 0x1000, Some("pc.ram"))),
            "MIG_RP_MSG_REQ_PAGES: Misaligned page request, start:10 len: 4096"
        );
        assert_eq!(
            e(req(0xf000, 0x2000, Some("pc.ram"))),
            "MIG_RP_MSG_REQ_PAGES request overrun, start=f000 len=2000 blocklen=10000"
        );
        let mut bad_id = req(0, 0x1000, Some("pc.ram"));
        bad_id.push(0);
        bad_id[3] += 1;
        assert_eq!(e(bad_id), "Req_Page_id with length 20 expecting 19");
        // The end of the channel is not an error.
        run(&[], &requests()).unwrap();
    }

    #[test]
    fn blocktime_follows_qemu() {
        // Two vCPUs on threads 10 and 11, and thread 99 that is no vCPU.
        let bt = Blocktime::new(&[10, 11]);
        let begin = |addr, tid, ns| assert!(bt.begin_at(addr, tid, || true, || ns));
        let end = |addr, ns| bt.end_at(addr, || {}, || ns);
        // vCPU 0 waits on page A from 1000, vCPU 1 on page B from 3000: both are down from
        // 3000 until A comes at 5000.
        begin(0xa000, 10, 1000);
        begin(0xb000, 11, 3000);
        begin(0xb000, 99, 3500);
        end(0xa000, 5000);
        // vCPU 0 then faults twice on B, which comes at 2_003_000.
        begin(0xb000, 10, 6000);
        begin(0xb000, 10, 7000);
        end(0xb000, 2_003_000);
        // A page nobody waited on, and a fault without a thread id, change nothing.
        end(0xc000, 2_004_000);
        begin(0xd000, 0, 2_005_000);
        let info = bt.info();
        // vCPU 0: 4000 on A, then 1_997_000 from its first fault on B. vCPU 1: 2_000_000.
        assert_eq!(info.vcpu_latency, vec![(4000 + 1_997_000) / 3, 2_000_000]);
        assert_eq!(info.vcpu_blocktime, vec![2, 2]);
        // Both were down from 3000 to 5000 and from 6000 to 2_003_000.
        assert_eq!(info.blocktime, 1);
        assert_eq!(info.non_vcpu_latency, 1_999_500);
        assert_eq!(info.latency, (4000 + 1_997_000 + 2_000_000 + 1_999_500) / 5);
        // 4 us goes in bucket 2, the waits of about 2 ms in bucket 10.
        let mut dist = vec![0; LATENCY_BUCKETS];
        dist[2] = 1;
        dist[10] = 4;
        assert_eq!(info.latency_dist, dist);
    }

    #[test]
    fn blocktime_checks_the_page_under_its_lock() {
        let bt = Blocktime::new(&[10]);
        assert!(!bt.begin(0x1000, 10, || false));
        assert_eq!(bt.info().vcpu_latency, vec![0]);
        assert!(bt.begin(0x1000, 10, || true));
        let mut marked = false;
        bt.end(0x1000, || marked = true);
        assert!(marked);
        assert_eq!(bt.info().latency_dist.iter().sum::<u64>(), 1);
    }

    #[test]
    fn wait_returns_on_request() {
        let r = std::sync::Arc::new(requests());
        let r2 = r.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            r2.queue(Some("pc.ram"), 0, 0x1000).unwrap();
        });
        let start = Instant::now();
        r.wait(Duration::from_secs(10));
        assert!(start.elapsed() < Duration::from_secs(5));
        t.join().unwrap();
        r.kick();
        r.wait(Duration::from_secs(10));
        assert!(start.elapsed() < Duration::from_secs(5));
    }
}
