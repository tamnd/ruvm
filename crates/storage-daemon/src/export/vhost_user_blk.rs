// SPDX-License-Identifier: GPL-2.0-or-later

//! The vhost-user-blk export: block/export/vhost-user-blk-server.c, the server of
//! util/vhost-user-server.c and the device side of the vhost-user protocol that QEMU takes from
//! subprojects/libvhost-user.
//!
//! The export listens on a Unix socket and serves one frontend at a time, like QEMU: a second
//! frontend waits in the listen backlog until the first one disconnects. The frontend's messages
//! are handled on the export's thread, and every ring that has a kick file descriptor gets a
//! thread of its own that waits for kicks and runs the requests through
//! [`VirtioBlkHandler`](super::virtio_blk::VirtioBlkHandler).
//!
//! Differences from QEMU:
//!
//! - Guest memory is not mapped. Each region's file is read and written with `pread` and
//!   `pwrite`, which works for the memfd and shared memory backends a vhost-user frontend
//!   normally uses but not for hugetlbfs, which refuses `write`.
//! - The protocol features offered are `MQ`, `REPLY_ACK`, `CONFIG` and `CONFIGURE_MEM_SLOTS`.
//!   There is no dirty page log, backend request channel, host notifier, postcopy or in-flight
//!   tracking, so the frontend neither migrates the device live nor reconnects with requests in
//!   flight, and there is no config change message after a resize. The capacity in the config
//!   space is instead read again on every `GET_CONFIG`.
//! - Only `unix` addresses can be listened on. An `fd` address fails the way the NBD server's
//!   does, since taking over an inherited descriptor by number is not possible here.
//! - A kick descriptor that reports a hang up (the write end of a pipe was closed) stops that
//!   ring's thread rather than making it spin.

use std::fs::File;
use std::io;
use std::os::fd::OwnedFd;
use std::os::unix::fs::FileExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::thread::JoinHandle;

use rustix::event::{PollFd, PollFlags, poll};
use rustix::io::Errno;
use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::types::{
    BlockExportOptionsVhostUserBlk, SocketAddress, SocketAddressU, UnixSocketAddress,
};
use ruvm_vhost::VHOST_USER_F_PROTOCOL_FEATURES;
use ruvm_vhost::user::message::{
    Header, MAX_REGIONS, REGION_SIZE, REPLY_FLAG, VERSION, VRING_ADDR_SIZE, VRING_IDX_MASK,
    VRING_NOFD_MASK, protocol, request, u32_at, u64_at,
};
use ruvm_vhost::user::{Connection, Message};
use ruvm_virtio_queue::{
    GuestMemory, MemoryError, RingAddresses, SplitQueue, VIRTIO_F_EVENT_IDX, VIRTIO_F_INDIRECT_DESC,
};

use super::virtio_blk::{
    ReqError, VIRTIO_BLK_CONFIG_SIZE, VIRTIO_BLK_CONFIG_WCE, VIRTIO_BLK_F_BLK_SIZE,
    VIRTIO_BLK_F_CONFIG_WCE, VIRTIO_BLK_F_DISCARD, VIRTIO_BLK_F_FLUSH, VIRTIO_BLK_F_MQ,
    VIRTIO_BLK_F_RO, VIRTIO_BLK_F_SEG_MAX, VIRTIO_BLK_F_TOPOLOGY, VIRTIO_BLK_F_WRITE_ZEROES,
    VIRTIO_BLK_MAX_DISCARD_SECTORS, VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS, VIRTIO_BLK_SECTOR_BITS,
    VIRTIO_BLK_SECTOR_SIZE, VIRTIO_F_VERSION_1, VirtioBlkConfig, VirtioBlkHandler,
    check_block_size,
};
use super::{ExportArgs, ExportDriver};

/// `VHOST_USER_BLK_NUM_QUEUES_DEFAULT`
const VHOST_USER_BLK_NUM_QUEUES_DEFAULT: u16 = 1;
/// `VHOST_USER_MAX_RAM_SLOTS`
const VHOST_USER_MAX_RAM_SLOTS: usize = 509;
/// `VHOST_USER_MEM_REG_SIZE`: the padding word and one region.
const VHOST_USER_MEM_REG_SIZE: usize = 8 + REGION_SIZE;
/// `VHOST_SET_CONFIG_TYPE_FRONTEND`
const VHOST_SET_CONFIG_TYPE_FRONTEND: u32 = 0;
/// `VHOST_USER_NONE`
const VHOST_USER_NONE: u32 = 0;

#[cfg(target_os = "linux")]
const SUN_PATH_LEN: usize = 108;
#[cfg(not(target_os = "linux"))]
const SUN_PATH_LEN: usize = 104;

/// A running vhost-user-blk export.
#[derive(Debug)]
pub struct VhostUserBlkExport {
    server: Arc<Server>,
    thread: Mutex<Option<JoinHandle<()>>>,
    /// The socket file to remove on shutdown, none for an abstract socket.
    path: Option<String>,
}

/// What the export thread, the rings and the export handle share.
#[derive(Debug)]
struct Server {
    handler: VirtioBlkHandler,
    config: Mutex<VirtioBlkConfig>,
    max_queues: u16,
    stopping: AtomicBool,
    /// A clone of the connected frontend's socket, for shutting it down.
    client: Mutex<Option<UnixStream>>,
    /// Written to wake the export thread when it waits for a connection.
    wake: UnixStream,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn read_lock<T>(m: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    m.read().unwrap_or_else(|e| e.into_inner())
}

fn write_lock<T>(m: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    m.write().unwrap_or_else(|e| e.into_inner())
}

impl ExportDriver for VhostUserBlkExport {
    /// QEMU's vhost-user-blk export never takes a reference of its own on the export, so a
    /// connected frontend does not keep `block-export-del` from going ahead.
    fn in_use(&self) -> bool {
        false
    }

    /// `vu_blk_exp_request_shutdown()`, which is `vhost_user_server_stop()`, and then the
    /// listener's cleanup, which removes the socket file.
    fn shutdown(&self) {
        let Some(thread) = lock(&self.thread).take() else {
            return;
        };
        self.server.stopping.store(true, Ordering::SeqCst);
        if let Some(client) = lock(&self.server.client).as_ref() {
            let _ = client.shutdown(std::net::Shutdown::Both);
        }
        let _ = (&self.server.wake).write_all_nonblocking();
        let _ = thread.join();
        if let Some(path) = &self.path {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Drop for VhostUserBlkExport {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Writing one byte to a socket that is only read to be woken up.
trait Wake {
    fn write_all_nonblocking(self) -> io::Result<()>;
}

impl Wake for &UnixStream {
    fn write_all_nonblocking(self) -> io::Result<()> {
        use std::io::Write;
        match (&*self).write(&[1]) {
            Ok(_) => Ok(()),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(()),
            Err(e) => Err(e),
        }
    }
}

/// `vu_blk_exp_create()`.
pub fn create(
    args: &ExportArgs<'_>,
    opts: &BlockExportOptionsVhostUserBlk,
) -> Result<Arc<dyn ExportDriver>> {
    let logical_block_size = opts.logical_block_size.unwrap_or(VIRTIO_BLK_SECTOR_SIZE);
    check_block_size("logical-block-size", logical_block_size)?;
    let num_queues = opts.num_queues.unwrap_or(VHOST_USER_BLK_NUM_QUEUES_DEFAULT);
    if num_queues == 0 {
        return Err(Error::generic("num-queues must be greater than 0"));
    }
    // check_block_size() keeps it at 2 MiB or below.
    let logical_block_size = logical_block_size as u32;

    let handler = VirtioBlkHandler {
        blk: args.blk.clone(),
        serial: "vhost_user_blk".into(),
        logical_block_size,
        writable: args.writable,
    };
    let config = initial_config(&handler, num_queues);

    let (listener, path) = listen(&opts.addr)?;
    let (wake, wake_rx) =
        UnixStream::pair().map_err(|e| Error::from_io("Failed to create socket", e))?;
    let _ = wake.set_nonblocking(true);
    let server = Arc::new(Server {
        handler,
        config: Mutex::new(config),
        max_queues: num_queues,
        stopping: AtomicBool::new(false),
        client: Mutex::new(None),
        wake,
    });
    let srv = server.clone();
    let thread = std::thread::Builder::new()
        .name(format!("vhost-user-blk {}", args.id))
        .spawn(move || serve(&srv, &listener, &wake_rx))
        .map_err(|e| Error::from_io("Failed to create thread", e))?;
    Ok(Arc::new(VhostUserBlkExport { server, thread: Mutex::new(Some(thread)), path }))
}

/// `vu_blk_initialize_config()`.
fn initial_config(handler: &VirtioBlkHandler, num_queues: u16) -> VirtioBlkConfig {
    let blk_size = handler.logical_block_size;
    VirtioBlkConfig {
        capacity: handler.blk.getlength().unwrap_or(0) >> VIRTIO_BLK_SECTOR_BITS,
        blk_size,
        size_max: 0,
        seg_max: 128 - 2,
        min_io_size: 0,
        opt_io_size: 0,
        wce: 0,
        num_queues,
        max_discard_sectors: VIRTIO_BLK_MAX_DISCARD_SECTORS,
        max_discard_seg: 1,
        discard_sector_alignment: blk_size >> VIRTIO_BLK_SECTOR_BITS,
        max_write_zeroes_sectors: VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS,
        max_write_zeroes_seg: 1,
    }
}

/// `vu_blk_get_features()`.
fn device_features(writable: bool) -> u64 {
    let mut features = 1u64 << VIRTIO_BLK_F_SEG_MAX
        | 1 << VIRTIO_BLK_F_TOPOLOGY
        | 1 << VIRTIO_BLK_F_BLK_SIZE
        | 1 << VIRTIO_BLK_F_FLUSH
        | 1 << VIRTIO_BLK_F_DISCARD
        | 1 << VIRTIO_BLK_F_WRITE_ZEROES
        | 1 << VIRTIO_BLK_F_CONFIG_WCE
        | 1 << VIRTIO_BLK_F_MQ
        | 1 << VIRTIO_F_VERSION_1
        | 1 << VIRTIO_F_INDIRECT_DESC
        | 1 << VIRTIO_F_EVENT_IDX
        | VHOST_USER_F_PROTOCOL_FEATURES;
    if !writable {
        features |= 1 << VIRTIO_BLK_F_RO;
    }
    features
}

/// The protocol features offered, see the module documentation.
const PROTOCOL_FEATURES: u64 =
    protocol::MQ | protocol::REPLY_ACK | protocol::CONFIG | protocol::CONFIGURE_MEM_SLOTS;

fn is_abstract(saddr: &UnixSocketAddress) -> bool {
    #[cfg(target_os = "linux")]
    return saddr.abstract_ == Some(true);
    #[cfg(not(target_os = "linux"))]
    {
        let _ = saddr;
        false
    }
}

/// The listening half of `vhost_user_server_start()`: `qio_net_listener_open_sync()` on a
/// `unix` address, as `unix_listen_saddr()` does it.
fn listen(addr: &SocketAddress) -> Result<(UnixListener, Option<String>)> {
    let saddr = match &addr.u {
        SocketAddressU::Unix(u) => u,
        SocketAddressU::Fd(f) => {
            if f.str.parse::<i32>().is_err() {
                return Err(Error::from_io(
                    format!("Unable to parse FD number {}", f.str),
                    io::Error::from_raw_os_error(libc::EINVAL),
                ));
            }
            return Err(Error::generic(format!("File descriptor '{}' is not a socket", f.str)));
        }
        _ => {
            return Err(Error::generic("Only socket address types 'unix' and 'fd' are supported"));
        }
    };
    let abstract_ = is_abstract(saddr);
    let path =
        if saddr.path.is_empty() && !abstract_ { temp_socket_path() } else { saddr.path.clone() };
    let max = if abstract_ { SUN_PATH_LEN - 1 } else { SUN_PATH_LEN };
    if path.len() > max {
        return Err(Error::generic(format!("UNIX socket path '{path}' is too long"))
            .hint(format!("Path must be less than {max} bytes\n")));
    }
    if !abstract_ {
        if let Err(e) = std::fs::remove_file(&path) {
            if e.kind() != io::ErrorKind::NotFound {
                return Err(Error::from_io(format!("Failed to unlink socket {path}"), e));
            }
        }
    }
    #[cfg(target_os = "linux")]
    let r = if abstract_ {
        use std::os::linux::net::SocketAddrExt;
        let mut name = path.as_bytes().to_vec();
        if saddr.tight == Some(false) {
            name.resize(SUN_PATH_LEN - 1, 0);
        }
        std::os::unix::net::SocketAddr::from_abstract_name(&name)
            .and_then(|a| UnixListener::bind_addr(&a))
    } else {
        UnixListener::bind(&path)
    };
    #[cfg(not(target_os = "linux"))]
    let r = UnixListener::bind(&path);
    let listener = r.map_err(|e| Error::from_io(format!("Failed to bind socket to {path}"), e))?;
    listener.set_nonblocking(true).map_err(|e| Error::from_io("Failed to listen on socket", e))?;
    Ok((listener, if abstract_ { None } else { Some(path) }))
}

/// `qemu-socket-XXXXXX` in the temporary directory, for a `unix` address with an empty path.
fn temp_socket_path() -> String {
    use std::sync::atomic::AtomicU32;
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir();
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let name = format!("qemu-socket-{:06x}", ((std::process::id() << 8) ^ n) & 0xff_ffff);
        let p = dir.join(name);
        if !p.exists() {
            return p.to_string_lossy().into_owned();
        }
    }
}

/// The export thread: wait for a frontend, serve it until it goes away, and wait for the next
/// one, until the export is shut down.
fn serve(srv: &Arc<Server>, listener: &UnixListener, wake: &UnixStream) {
    while !srv.stopping.load(Ordering::SeqCst) {
        let mut fds = [PollFd::new(listener, PollFlags::IN), PollFd::new(wake, PollFlags::IN)];
        match poll(&mut fds, None) {
            Ok(_) | Err(Errno::INTR) => {}
            Err(_) => return,
        }
        if srv.stopping.load(Ordering::SeqCst) {
            return;
        }
        let stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(_) => continue,
        };
        // vu_accept()
        if stream.set_nonblocking(false).is_err() {
            continue;
        }
        let Ok(clone) = stream.try_clone() else {
            continue;
        };
        *lock(&srv.client) = Some(clone);
        if srv.stopping.load(Ordering::SeqCst) {
            break;
        }
        Session::new(srv.clone(), stream).run();
        *lock(&srv.client) = None;
    }
}

/// One memory region the frontend shared.
#[derive(Debug)]
struct Region {
    gpa: u64,
    size: u64,
    /// The frontend's own address for the region, which ring addresses are given in.
    qva: u64,
    /// Where the region starts in `file`.
    offset: u64,
    file: File,
}

/// The guest memory the frontend shared, kept sorted by guest address.
#[derive(Debug, Default)]
struct MemTable {
    regions: Vec<Region>,
}

impl MemTable {
    fn find(&self, gpa: u64) -> Option<&Region> {
        self.regions.iter().find(|r| gpa >= r.gpa && gpa - r.gpa < r.size)
    }

    /// `qva_to_va()`, turned into a guest address.
    fn qva_to_gpa(&self, qva: u64) -> Option<u64> {
        self.regions
            .iter()
            .find(|r| qva >= r.qva && qva - r.qva < r.size)
            .map(|r| r.gpa + (qva - r.qva))
    }

    /// `_vu_add_mem_reg()` without the mapping.
    fn add(&mut self, region: Region) -> std::result::Result<(), String> {
        let end = region.gpa.saturating_add(region.size);
        if self.regions.iter().any(|r| region.gpa < r.gpa.saturating_add(r.size) && r.gpa < end) {
            return Err("regions with overlapping guest physical addresses".into());
        }
        let idx = self.regions.partition_point(|r| r.gpa < region.gpa);
        self.regions.insert(idx, region);
        Ok(())
    }

    /// Runs `f` on each piece of `addr..addr + len` with the region it falls in and the offset
    /// into the region's file.
    fn each(
        &self,
        addr: u64,
        len: usize,
        mut f: impl FnMut(&File, u64, std::ops::Range<usize>) -> io::Result<()>,
    ) -> std::result::Result<(), MemoryError> {
        let fail = || MemoryError::OutOfRange { addr, len: len as u64 };
        let mut done = 0usize;
        while done < len {
            let at = addr.checked_add(done as u64).ok_or_else(fail)?;
            let r = self.find(at).ok_or_else(fail)?;
            let within = at - r.gpa;
            let n = ((r.size - within).min((len - done) as u64)) as usize;
            f(&r.file, r.offset + within, done..done + n).map_err(|_| fail())?;
            done += n;
        }
        Ok(())
    }
}

impl GuestMemory for MemTable {
    fn read(&self, addr: u64, buf: &mut [u8]) -> std::result::Result<(), MemoryError> {
        let len = buf.len();
        self.each(addr, len, |file, off, range| file.read_exact_at(&mut buf[range], off))
    }

    fn write(&self, addr: u64, buf: &[u8]) -> std::result::Result<(), MemoryError> {
        self.each(addr, buf.len(), |file, off, range| file.write_all_at(&buf[range], off))
    }
}

/// `VuVirtq`, the part the ring's thread shares with the message loop.
#[derive(Debug, Default)]
struct VringState {
    num: u16,
    /// Ring addresses, already turned into guest addresses.
    addrs: Option<RingAddresses>,
    /// Where the driver's next request is when the ring (re)starts.
    last_avail: u16,
    queue: Option<SplitQueue>,
    call: Option<File>,
    enabled: bool,
    started: bool,
}

/// A ring's thread. Writing to `ctl` makes it look at the ring again; dropping `ctl` stops it.
#[derive(Debug)]
struct Worker {
    ctl: UnixStream,
    thread: JoinHandle<()>,
}

#[derive(Debug, Default)]
struct Vring {
    state: Arc<Mutex<VringState>>,
    worker: Option<Worker>,
}

impl Vring {
    fn stop(&mut self) {
        if let Some(w) = self.worker.take() {
            drop(w.ctl);
            let _ = w.thread.join();
        }
    }

    fn poke(&self) {
        if let Some(w) = &self.worker {
            let _ = (&w.ctl).write_all_nonblocking();
        }
    }
}

/// One connected frontend: `VuDev` and the message loop of `vu_client_trip()`.
struct Session {
    srv: Arc<Server>,
    conn: Connection,
    mem: Arc<RwLock<MemTable>>,
    vrings: Vec<Vring>,
    features: u64,
    protocol_features: u64,
    broken: bool,
}

/// The reply to a message, when it has one of its own.
type Reply = Option<Vec<u8>>;

fn reply_u64(v: u64) -> Reply {
    Some(v.to_ne_bytes().to_vec())
}

impl Session {
    fn new(srv: Arc<Server>, stream: UnixStream) -> Self {
        let vrings = (0..srv.max_queues).map(|_| Vring::default()).collect();
        Session {
            srv,
            conn: Connection::new(stream),
            mem: Arc::default(),
            vrings,
            features: 0,
            protocol_features: 0,
            broken: false,
        }
    }

    /// `vu_panic()` with the server's `panic_cb()`.
    fn panic(&mut self, msg: &str) {
        self.broken = true;
        error_report(&format!("vu_panic: {msg}"));
    }

    fn run(mut self) {
        while !self.broken {
            let msg = match self.conn.recv() {
                Ok(m) => m,
                Err(ruvm_vhost::Error::Disconnected) => break,
                Err(ruvm_vhost::Error::TooManyFds { max, .. }) => {
                    error_report(&format!(
                        "A maximum of {max} fds are allowed, however got more fds now"
                    ));
                    break;
                }
                Err(ruvm_vhost::Error::PayloadTooLarge(size)) => {
                    error_report(&format!("Error: too big message request, size: {size}"));
                    break;
                }
                Err(_) => break,
            };
            if !self.dispatch(msg) {
                break;
            }
        }
        for v in &mut self.vrings {
            v.stop();
        }
    }

    /// `vu_dispatch()`. Returns false when the connection is gone.
    fn dispatch(&mut self, msg: Message) -> bool {
        let header = msg.header;
        let reply =
            self.process(msg).or_else(|| header.needs_reply().then(|| 0u64.to_ne_bytes().to_vec()));
        let Some(payload) = reply else {
            return true;
        };
        let size = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        let h = Header { request: header.request, flags: VERSION | REPLY_FLAG, size };
        if let Err(e) = self.conn.send(h, &payload, &[]) {
            self.panic(&format!("Error while writing: {}", vhost_strerror(&e)));
            return false;
        }
        true
    }

    fn vring_index(&mut self, index: u32, what: &str) -> Option<usize> {
        if index >= u32::from(self.srv.max_queues) {
            self.panic(&format!("Invalid {what} index: {index}"));
            return None;
        }
        Some(index as usize)
    }

    /// `vu_process_message()`.
    fn process(&mut self, msg: Message) -> Reply {
        let p = &msg.payload;
        let u64_payload = || if p.len() >= 8 { u64_at(p, 0) } else { 0 };
        let state = || {
            if p.len() >= 8 { (u32_at(p, 0), u32_at(p, 4)) } else { (u32::MAX, 0) }
        };
        match msg.header.request {
            // vu_blk_process_msg(): the panic callback alone, which does not break the device.
            VHOST_USER_NONE => {
                error_report("vu_panic: disconnect");
                None
            }
            request::GET_FEATURES => reply_u64(device_features(self.srv.handler.writable)),
            request::SET_FEATURES => {
                self.features = u64_payload();
                if self.features & (1 << VIRTIO_F_VERSION_1) == 0 {
                    self.panic("virtio legacy devices aren't supported by libvhost-user");
                    return None;
                }
                if self.features & VHOST_USER_F_PROTOCOL_FEATURES == 0 {
                    for v in &self.vrings {
                        lock(&v.state).enabled = true;
                    }
                }
                None
            }
            request::GET_PROTOCOL_FEATURES => reply_u64(PROTOCOL_FEATURES),
            request::SET_PROTOCOL_FEATURES => {
                self.protocol_features = u64_payload();
                None
            }
            request::SET_OWNER => None,
            request::RESET_OWNER | request::RESET_DEVICE => {
                self.reset();
                None
            }
            request::SET_MEM_TABLE => {
                self.set_mem_table(msg);
                None
            }
            request::SET_VRING_NUM => {
                let (index, num) = state();
                let i = self.vring_index(index, "vring_num")?;
                lock(&self.vrings[i].state).num = num as u16;
                None
            }
            request::SET_VRING_ADDR => {
                self.set_vring_addr(p);
                None
            }
            request::SET_VRING_BASE => {
                let (index, num) = state();
                let i = self.vring_index(index, "vring_base")?;
                lock(&self.vrings[i].state).last_avail = num as u16;
                None
            }
            request::GET_VRING_BASE => {
                let (index, _) = state();
                let num = self.get_vring_base(index);
                let mut out = index.to_ne_bytes().to_vec();
                out.extend_from_slice(&num.to_ne_bytes());
                Some(out)
            }
            request::SET_VRING_KICK => {
                self.set_vring_kick(msg);
                None
            }
            request::SET_VRING_CALL => {
                self.set_vring_call(msg);
                None
            }
            request::SET_VRING_ERR => {
                self.check_queue_msg_file(&msg);
                None
            }
            request::GET_QUEUE_NUM => reply_u64(u64::from(self.srv.max_queues)),
            request::SET_VRING_ENABLE => {
                let (index, enable) = state();
                let i = self.vring_index(index, "vring_enable")?;
                lock(&self.vrings[i].state).enabled = enable != 0;
                self.vrings[i].poke();
                None
            }
            request::SET_BACKEND_REQ_FD => {
                if msg.fds.len() != 1 {
                    self.panic(&format!("Invalid backend_req_fd message ({} fd's)", msg.fds.len()));
                }
                None
            }
            request::GET_CONFIG => Some(self.get_config(p)),
            request::SET_CONFIG => {
                if !self.set_config(p) {
                    self.panic("Set virtio configuration space failed");
                }
                None
            }
            request::GET_MAX_MEM_SLOTS => reply_u64(VHOST_USER_MAX_RAM_SLOTS as u64),
            request::ADD_MEM_REG => {
                self.add_mem_reg(msg);
                None
            }
            request::REM_MEM_REG => {
                self.rem_mem_reg(msg);
                None
            }
            other => {
                self.panic(&format!("Unhandled request: {other}"));
                None
            }
        }
    }

    /// `vu_reset_device_exec()`: every ring goes back to its initial state.
    fn reset(&mut self) {
        for v in &mut self.vrings {
            v.stop();
            *lock(&v.state) = VringState::default();
        }
    }

    fn region(p: &[u8], at: usize, file: OwnedFd) -> Region {
        Region {
            gpa: u64_at(p, at),
            size: u64_at(p, at + 8),
            qva: u64_at(p, at + 16),
            offset: u64_at(p, at + 24),
            file: File::from(file),
        }
    }

    /// `vu_set_mem_table_exec()`.
    fn set_mem_table(&mut self, msg: Message) {
        let p = &msg.payload;
        let count = if p.len() >= 8 { u32_at(p, 0) as usize } else { 0 };
        let count = count.min(MAX_REGIONS).min((p.len().saturating_sub(8)) / REGION_SIZE);
        let mut table = MemTable::default();
        let mut fds = msg.fds.into_iter();
        for i in 0..count {
            let Some(fd) = fds.next() else {
                self.panic("region mmap error: Bad file descriptor");
                break;
            };
            if let Err(e) = table.add(Self::region(p, 8 + i * REGION_SIZE, fd)) {
                self.panic(&e);
                break;
            }
        }
        *write_lock(&self.mem) = table;
    }

    /// `vu_add_mem_reg()`.
    fn add_mem_reg(&mut self, msg: Message) {
        if msg.fds.len() != 1 {
            self.panic(&format!(
                "VHOST_USER_ADD_MEM_REG received {} fds - only 1 fd should be sent for this \
                 message type",
                msg.fds.len()
            ));
            return;
        }
        if msg.payload.len() < VHOST_USER_MEM_REG_SIZE {
            self.panic(&format!(
                "VHOST_USER_ADD_MEM_REG requires a message size of at least \
                 {VHOST_USER_MEM_REG_SIZE} bytes and only {} bytes were received",
                msg.payload.len()
            ));
            return;
        }
        let mut mem = write_lock(&self.mem);
        if mem.regions.len() == VHOST_USER_MAX_RAM_SLOTS {
            drop(mem);
            self.panic(
                "failing attempt to hot add memory via VHOST_USER_ADD_MEM_REG message because \
                 the backend has no free ram slots available",
            );
            return;
        }
        let fd = msg.fds.into_iter().next();
        let r = fd.map(|fd| Self::region(&msg.payload, 8, fd)).map(|r| mem.add(r));
        drop(mem);
        if let Some(Err(e)) = r {
            self.panic(&e);
        }
    }

    /// `vu_rem_mem_reg()`.
    fn rem_mem_reg(&mut self, msg: Message) {
        if msg.fds.len() > 1 {
            self.panic(&format!(
                "VHOST_USER_REM_MEM_REG received {} fds - at most 1 fd should be sent for this \
                 message type",
                msg.fds.len()
            ));
            return;
        }
        let p = &msg.payload;
        if p.len() < VHOST_USER_MEM_REG_SIZE {
            self.panic(&format!(
                "VHOST_USER_REM_MEM_REG requires a message size of at least \
                 {VHOST_USER_MEM_REG_SIZE} bytes and only {} bytes were received",
                p.len()
            ));
            return;
        }
        let (gpa, size, qva) = (u64_at(p, 8), u64_at(p, 16), u64_at(p, 24));
        let mut mem = write_lock(&self.mem);
        let found = mem.regions.iter().position(|r| gpa >= r.gpa && gpa - r.gpa < r.size);
        match found {
            Some(i)
                if mem.regions[i].gpa == gpa
                    && mem.regions[i].qva == qva
                    && mem.regions[i].size == size =>
            {
                mem.regions.remove(i);
            }
            _ => {
                drop(mem);
                self.panic("Specified region not found\n");
            }
        }
    }

    /// `vu_set_vring_addr_exec()`. The ring addresses are turned into guest addresses here,
    /// which is when libvhost-user maps them.
    fn set_vring_addr(&mut self, p: &[u8]) {
        if p.len() < VRING_ADDR_SIZE {
            self.panic("Invalid vring_addr message");
            return;
        }
        let Some(i) = self.vring_index(u32_at(p, 0), "vring_addr") else {
            return;
        };
        let (desc, used, avail) = (u64_at(p, 8), u64_at(p, 16), u64_at(p, 24));
        let mem = read_lock(&self.mem);
        let addrs = match (mem.qva_to_gpa(desc), mem.qva_to_gpa(avail), mem.qva_to_gpa(used)) {
            (Some(desc_table), Some(driver_area), Some(device_area)) => {
                RingAddresses { desc_table, driver_area, device_area }
            }
            _ => {
                drop(mem);
                self.panic("Invalid vring_addr message");
                return;
            }
        };
        drop(mem);
        lock(&self.vrings[i].state).addrs = Some(addrs);
    }

    /// `vu_get_vring_base_exec()`: stop the ring and say where it stopped.
    fn get_vring_base(&mut self, index: u32) -> u32 {
        let Some(i) = self.vring_index(index, "vring_base") else {
            return 0;
        };
        let v = &mut self.vrings[i];
        v.stop();
        let mut st = lock(&v.state);
        if let Some(q) = st.queue.take() {
            st.last_avail = q.next_avail();
        }
        st.started = false;
        st.call = None;
        u32::from(st.last_avail)
    }

    /// `vu_check_queue_msg_file()`: the ring index, and whether a descriptor came with it.
    fn check_queue_msg_file(&mut self, msg: &Message) -> Option<(usize, bool)> {
        let v = if msg.payload.len() >= 8 { u64_at(&msg.payload, 0) } else { 0 };
        let index = (v & VRING_IDX_MASK) as u32;
        let nofd = v & VRING_NOFD_MASK != 0;
        let i = self.vring_index(index, "queue")?;
        if nofd {
            return Some((i, false));
        }
        if msg.fds.len() != 1 {
            self.panic(&format!("Invalid fds in request: {}", msg.header.request));
            return None;
        }
        Some((i, true))
    }

    /// `vu_set_vring_kick_exec()`, then `vu_blk_queue_set_started()`: the ring starts and a
    /// thread waits for its kicks.
    fn set_vring_kick(&mut self, mut msg: Message) {
        let Some((i, has_fd)) = self.check_queue_msg_file(&msg) else {
            return;
        };
        let event_idx = self.features & (1 << VIRTIO_F_EVENT_IDX) != 0;
        let indirect = self.features & (1 << VIRTIO_F_INDIRECT_DESC) != 0;
        self.vrings[i].stop();
        {
            let mut st = lock(&self.vrings[i].state);
            st.started = true;
            let Some(addrs) = st.addrs else {
                drop(st);
                self.panic("Invalid vring_addr message");
                return;
            };
            let queue = match SplitQueue::new(st.num, addrs) {
                Ok(q) => q,
                Err(_) => {
                    drop(st);
                    self.panic("Invalid vring_addr message");
                    return;
                }
            };
            let mut queue = queue;
            queue.set_event_idx(event_idx);
            queue.set_indirect_desc(indirect);
            queue.set_next_avail(st.last_avail);
            let used = read_lock(&self.mem).read_u16(addrs.device_area + 2);
            queue.set_next_used(used.unwrap_or(0));
            st.queue = Some(queue);
        }
        if !has_fd {
            return;
        }
        let kick = msg.fds.remove(0);
        let Ok((ctl, ctl_rx)) = UnixStream::pair() else {
            return;
        };
        let Some(client) = self.conn.stream().try_clone().ok() else {
            return;
        };
        let ring = RingThread {
            srv: self.srv.clone(),
            state: self.vrings[i].state.clone(),
            mem: self.mem.clone(),
            kick,
            ctl: ctl_rx,
            client,
        };
        let spawned = std::thread::Builder::new()
            .name(format!("vhost-user-blk vq{i}"))
            .spawn(move || ring.run());
        if let Ok(thread) = spawned {
            let _ = ctl.set_nonblocking(true);
            self.vrings[i].worker = Some(Worker { ctl, thread });
        }
    }

    /// `vu_set_vring_call_exec()`.
    fn set_vring_call(&mut self, mut msg: Message) {
        let Some((i, has_fd)) = self.check_queue_msg_file(&msg) else {
            return;
        };
        let call = has_fd.then(|| File::from(msg.fds.remove(0)));
        // In case of I/O hang after reconnecting.
        if let Some(f) = &call {
            let _ = signal(f);
        }
        lock(&self.vrings[i].state).call = call;
    }

    /// `vu_blk_get_config()`. As in QEMU the offset is ignored and the bytes always come from
    /// the start of the config space. A size larger than the config space gets an empty reply.
    fn get_config(&self, p: &[u8]) -> Vec<u8> {
        if p.len() < 12 {
            return Vec::new();
        }
        let size = u32_at(p, 4) as usize;
        if size > VIRTIO_BLK_CONFIG_SIZE || p.len() < 12 + size {
            return Vec::new();
        }
        let mut cfg = lock(&self.srv.config);
        if let Ok(len) = self.srv.handler.blk.getlength() {
            cfg.capacity = len >> VIRTIO_BLK_SECTOR_BITS;
        }
        let bytes = cfg.to_bytes();
        let mut out = p.to_vec();
        out[12..12 + size].copy_from_slice(&bytes[..size]);
        out
    }

    /// `vu_blk_set_config()`: only the write cache byte can be written.
    fn set_config(&self, p: &[u8]) -> bool {
        if p.len() < 13 {
            return false;
        }
        let (offset, size, flags) = (u32_at(p, 0), u32_at(p, 4), u32_at(p, 8));
        // Don't support live migration.
        if flags != VHOST_SET_CONFIG_TYPE_FRONTEND {
            return false;
        }
        if offset as usize != VIRTIO_BLK_CONFIG_WCE || size != 1 {
            return false;
        }
        let wce = p[12];
        lock(&self.srv.config).wce = wce;
        self.srv.handler.blk.set_enable_write_cache(wce != 0);
        true
    }
}

/// `strerror()` of the I/O error under a vhost error.
fn vhost_strerror(e: &ruvm_vhost::Error) -> String {
    match e {
        ruvm_vhost::Error::Io(io) => ruvm_base::error::strerror(io),
        other => other.to_string(),
    }
}

/// `eventfd_write(fd, 1)`. With a pipe in place of an eventfd the eight bytes are just data.
fn signal(f: &File) -> io::Result<()> {
    use std::io::Write;
    let mut f = f;
    f.write_all(&1u64.to_ne_bytes())
}

/// What a ring's thread needs.
struct RingThread {
    srv: Arc<Server>,
    state: Arc<Mutex<VringState>>,
    mem: Arc<RwLock<MemTable>>,
    kick: OwnedFd,
    ctl: UnixStream,
    client: UnixStream,
}

impl RingThread {
    /// Waits for kicks and runs the ring each time, `vu_kick_cb()` and `vu_blk_process_vq()`.
    /// The first pass is the kick libvhost-user injects when the ring starts.
    fn run(self) {
        loop {
            if let Err(msg) = self.process() {
                error_report(&format!("vu_panic: {msg}"));
                // kick_handler(): a broken device ends the connection.
                let _ = self.client.shutdown(std::net::Shutdown::Both);
                return;
            }
            match self.wait() {
                Ok(true) => {}
                Ok(false) => return,
                Err(msg) => {
                    error_report(&format!("vu_panic: {msg}"));
                    let _ = self.client.shutdown(std::net::Shutdown::Both);
                    return;
                }
            }
        }
    }

    /// Waits for a kick or a poke. False means stop.
    fn wait(&self) -> std::result::Result<bool, String> {
        loop {
            let mut fds =
                [PollFd::new(&self.kick, PollFlags::IN), PollFd::new(&self.ctl, PollFlags::IN)];
            match poll(&mut fds, None) {
                Ok(_) => {}
                Err(Errno::INTR) => continue,
                Err(_) => return Ok(false),
            }
            let (kick, ctl) = (fds[0].revents(), fds[1].revents());
            if !ctl.is_empty() {
                let mut b = [0u8; 64];
                match rustix::io::read(&self.ctl, &mut b) {
                    Ok(0) | Err(_) => return Ok(false),
                    Ok(_) => return Ok(true),
                }
            }
            if kick.contains(PollFlags::IN) {
                let mut b = [0u8; 8];
                return match rustix::io::read(&self.kick, &mut b) {
                    Ok(0) => Ok(false),
                    Ok(_) | Err(Errno::AGAIN) | Err(Errno::INTR) => Ok(true),
                    Err(e) => Err(format!(
                        "kick eventfd_read(): {}",
                        ruvm_base::error::strerror(&io::Error::from(e))
                    )),
                };
            }
            if !kick.is_empty() {
                return Ok(false);
            }
        }
    }

    /// `vu_blk_process_vq()`: every available request, each completed and notified on its own.
    fn process(&self) -> std::result::Result<(), String> {
        let mut st = lock(&self.state);
        if !st.started || !st.enabled {
            return Ok(());
        }
        let mem = read_lock(&self.mem);
        let st = &mut *st;
        let Some(queue) = st.queue.as_mut() else {
            return Ok(());
        };
        loop {
            let chain = match queue.pop(&*mem) {
                Ok(Some(c)) => c,
                Ok(None) => return Ok(()),
                Err(e) => return Err(format!("virtio: {e}")),
            };
            let in_len = match self.srv.handler.process_req(&*mem, &chain) {
                Ok(n) => n,
                Err(ReqError::Malformed) => continue,
                Err(ReqError::Memory(_)) => {
                    return Err("virtio: invalid address for buffers".into());
                }
            };
            // vu_blk_req_complete(): vu_queue_push() and vu_queue_notify().
            queue.add_used(&*mem, chain.head(), in_len).map_err(|e| format!("virtio: {e}"))?;
            let notify = queue.needs_notification(&*mem).map_err(|e| format!("virtio: {e}"))?;
            if notify {
                if let Some(call) = &st.call {
                    signal(call).map_err(|e| {
                        format!("Error writing eventfd: {}", ruvm_base::error::strerror(&e))
                    })?;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `unix` address; the Linux type has more fields than the others.
    fn unix_addr(path: String) -> UnixSocketAddress {
        #[cfg(target_os = "linux")]
        return UnixSocketAddress { path, abstract_: None, tight: None };
        #[cfg(not(target_os = "linux"))]
        UnixSocketAddress { path }
    }

    #[test]
    fn features_follow_writable() {
        assert_eq!(device_features(false) & (1 << VIRTIO_BLK_F_RO), 1 << VIRTIO_BLK_F_RO);
        assert_eq!(device_features(true) & (1 << VIRTIO_BLK_F_RO), 0);
        assert_ne!(device_features(true) & VHOST_USER_F_PROTOCOL_FEATURES, 0);
    }

    #[test]
    fn only_unix_addresses() {
        let inet = SocketAddress {
            u: SocketAddressU::Inet(ruvm_qapi::types::InetSocketAddress {
                host: "localhost".into(),
                port: "0".into(),
                ..Default::default()
            }),
        };
        let e = listen(&inet).unwrap_err();
        assert_eq!(e.message(), "Only socket address types 'unix' and 'fd' are supported");
        let long = SocketAddress { u: SocketAddressU::Unix(unix_addr("x".repeat(200))) };
        let e = listen(&long).unwrap_err();
        assert!(e.message().starts_with("UNIX socket path '"), "{}", e.message());
    }

    #[test]
    fn mem_table_splits_accesses() {
        let dir = std::env::temp_dir();
        let make = |name: &str, len: usize| {
            let p = dir.join(format!("ruvm-vub-mem-{}-{name}", std::process::id()));
            std::fs::write(&p, vec![0u8; len]).unwrap();
            let f = std::fs::OpenOptions::new().read(true).write(true).open(&p).unwrap();
            let _ = std::fs::remove_file(&p);
            f
        };
        let mut t = MemTable::default();
        t.add(Region {
            gpa: 0x1000,
            size: 0x1000,
            qva: 0x7000,
            offset: 0,
            file: make("b", 0x1000),
        })
        .unwrap();
        t.add(Region { gpa: 0, size: 0x1000, qva: 0x5000, offset: 0x10, file: make("a", 0x1010) })
            .unwrap();
        assert!(
            t.add(Region { gpa: 0x800, size: 0x10, qva: 0, offset: 0, file: make("c", 0x10) })
                .is_err()
        );
        assert_eq!(t.qva_to_gpa(0x7010), Some(0x1010));
        assert_eq!(t.qva_to_gpa(0x9000), None);
        t.write(0xffe, &[1, 2, 3, 4]).unwrap();
        let mut b = [0u8; 4];
        t.read(0xffe, &mut b).unwrap();
        assert_eq!(b, [1, 2, 3, 4]);
        assert!(t.read(0x1ffe, &mut b).is_err());
    }
}
