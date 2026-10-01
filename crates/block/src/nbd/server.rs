// SPDX-License-Identifier: GPL-2.0-or-later

//! The NBD server from nbd/server.c, with the server lifetime from blockdev-nbd.c and the
//! export bookkeeping from block/export/export.c.
//!
//! [`NbdServer`] listens on a socket and runs every connection on a thread of its own: the
//! handshake first, bounded by the handshake timer, then the requests one after the other.
//! Exports belong to the server they were added to. The QMP style entry points
//! ([`nbd_server_start`] and friends) keep one server per process, like QEMU's `nbd_server`.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{
    BlockDirtyBitmapOrStr, BlockExportOptions, BlockExportOptionsNbd, BlockExportOptionsU,
    BlockExportRemoveMode, NbdServerAddOptions, NbdServerOptions, SocketAddress,
};

use super::proto::*;
use super::sock::{Listener, socket_listen};
use crate::backend::BlockBackend;
use crate::graph::{BlockGraph, id_wellformed};
use crate::node::{BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO, Node};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE};

const NBD_META_ID_BASE_ALLOCATION: u32 = 0;
const NBD_META_ID_ALLOCATION_DEPTH: u32 = 1;
const NBD_META_ID_DIRTY_BITMAP: u32 = 2;

/// `NBD_MAX_BLOCK_STATUS_EXTENTS`: 1 MiB of extents.
const NBD_MAX_BLOCK_STATUS_EXTENTS: usize = 1024 * 1024 / 8;

const BDRV_SECTOR_SIZE: u64 = 512;

/// What `bdrv_block_status_above()` says about the start of a range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NbdBlockStatus {
    /// `BDRV_BLOCK_DATA`.
    pub data: bool,
    /// `BDRV_BLOCK_ZERO`.
    pub zero: bool,
    /// How many bytes from the start of the range this applies to, at least 1.
    pub bytes: u64,
}

/// Where an export gets its block status and allocation depth from.
///
/// By default an export asks the block layer about the node it exports, like QEMU. A creator
/// can pass another source in [`NbdExportExtras`], to test or to serve something the block
/// layer does not know about.
pub trait NbdStatusSource: Send + Sync + fmt::Debug {
    /// `blk_co_block_status_above()` for `[offset, offset + bytes)`.
    fn block_status(&self, offset: u64, bytes: u64) -> io::Result<NbdBlockStatus>;

    /// `blk_co_is_allocated_above()`: the depth of the layer that decides the content of the
    /// start of the range (1 for the top, 0 for unallocated), and how many bytes that covers.
    fn allocation_depth(&self, offset: u64, bytes: u64) -> io::Result<(u32, u64)> {
        let _ = offset;
        Ok((1, bytes))
    }
}

/// A dirty bitmap an export can publish as `qemu:dirty-bitmap:NAME`.
pub trait NbdDirtyBitmap: Send + Sync + fmt::Debug {
    /// `bdrv_dirty_bitmap_name()`.
    fn name(&self) -> String;

    /// `bdrv_dirty_bitmap_enabled()`.
    fn enabled(&self) -> bool;

    /// `bdrv_dirty_bitmap_next_dirty_area()`: the first dirty run in `[start, end)` as
    /// `(offset, length)`, with the length capped at `max`.
    fn next_dirty_area(&self, start: u64, end: u64, max: u64) -> Option<(u64, u64)>;
}

/// What an export needs beyond `BlockExportOptions`, for the parts the block layer does not
/// have yet.
#[derive(Clone, Debug, Default)]
pub struct NbdExportExtras {
    /// Block status and allocation depth. `None` reports everything as allocated data.
    pub status: Option<Arc<dyn NbdStatusSource>>,
    /// The dirty bitmaps that exist, with the node each one belongs to.
    pub bitmaps: Vec<(String, Arc<dyn NbdDirtyBitmap>)>,
}

/// The block status of the exported node, as QEMU's server asks the block layer.
#[derive(Debug)]
struct NodeStatus {
    node: Arc<Node>,
}

impl NbdStatusSource for NodeStatus {
    fn block_status(&self, offset: u64, bytes: u64) -> io::Result<NbdBlockStatus> {
        // blockstatus_to_extents(): bdrv_co_block_status_above(bs, NULL, ...).
        let st = self.node.block_status_above(None, offset, bytes)?;
        Ok(NbdBlockStatus {
            data: st.ret & BDRV_BLOCK_DATA != 0,
            zero: st.ret & BDRV_BLOCK_ZERO != 0,
            bytes: st.pnum,
        })
    }

    fn allocation_depth(&self, offset: u64, bytes: u64) -> io::Result<(u32, u64)> {
        // blockalloc_to_extents(): bdrv_co_is_allocated_above(bs, NULL, false, ...).
        self.node.is_allocated_above(None, false, offset, bytes)
    }
}

#[derive(Debug)]
struct AllData;

impl NbdStatusSource for AllData {
    fn block_status(&self, _offset: u64, bytes: u64) -> io::Result<NbdBlockStatus> {
        Ok(NbdBlockStatus { data: true, zero: false, bytes })
    }
}

/// `NBDExport` together with its `BlockExport`.
#[derive(Debug)]
struct Export {
    id: String,
    name: String,
    description: Option<String>,
    blk: Arc<BlockBackend>,
    size: u64,
    nbdflags: u16,
    bitmaps: Vec<Arc<dyn NbdDirtyBitmap>>,
    bitmap_names: Vec<String>,
    allocation_depth: bool,
    status: Arc<dyn NbdStatusSource>,
}

/// An accepted connection, as the server keeps track of it.
#[derive(Debug)]
struct Conn {
    stream: NbdStream,
    export: Option<Arc<Export>>,
}

#[derive(Debug, Default)]
struct State {
    exports: Vec<Arc<Export>>,
    conns: HashMap<u64, Conn>,
    next_conn: u64,
    stopping: bool,
}

#[derive(Debug)]
struct Inner {
    listener: Option<Listener>,
    addr: Option<SocketAddress>,
    handshake_max_secs: u32,
    max_connections: u32,
    state: Mutex<State>,
    cond: Condvar,
    accept_thread: Mutex<Option<thread::JoinHandle<()>>>,
    on_close: CloseNotify,
}

/// What [`NbdServer::set_close_notify`] installed: the `close_fn` of `nbd_client_new()`.
#[derive(Default)]
struct CloseNotify(Mutex<Option<CloseFn>>);

/// The callback [`NbdServer::set_close_notify`] takes.
type CloseFn = Arc<dyn Fn(bool) + Send + Sync>;

impl fmt::Debug for CloseNotify {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CloseNotify")
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn find_export(&self, name: &str) -> Option<Arc<Export>> {
        self.lock().exports.iter().find(|e| e.name == name).cloned()
    }
}

/// A running NBD server: `NBDServerData` and the exports added to it.
#[derive(Clone, Debug)]
pub struct NbdServer {
    inner: Arc<Inner>,
}

impl NbdServer {
    /// `nbd_server_start_options()`: listen on `opts.addr`.
    pub fn start(opts: &NbdServerOptions) -> Result<NbdServer> {
        Self::start_addr(
            &opts.addr,
            opts.handshake_max_seconds.unwrap_or(NBD_DEFAULT_HANDSHAKE_MAX_SECS),
            opts.tls_creds.as_deref(),
            opts.max_connections.unwrap_or(NBD_DEFAULT_MAX_CONNECTIONS),
        )
    }

    /// `nbd_server_start()`. A `max_connections` of 0 means no limit, and a
    /// `handshake_max_secs` of 0 means no handshake timeout.
    pub fn start_addr(
        addr: &SocketAddress,
        handshake_max_secs: u32,
        tls_creds: Option<&str>,
        max_connections: u32,
    ) -> Result<NbdServer> {
        let s = Self::bind_addr(addr, handshake_max_secs, tls_creds, max_connections)?;
        s.start_accepting()?;
        Ok(s)
    }

    /// [`NbdServer::start_addr`] without accepting connections yet: they wait in the
    /// listen backlog until [`NbdServer::start_accepting`]. qemu-nbd listens before it opens
    /// the image and only accepts once the export is there.
    pub fn bind_addr(
        addr: &SocketAddress,
        handshake_max_secs: u32,
        tls_creds: Option<&str>,
        max_connections: u32,
    ) -> Result<NbdServer> {
        let (listener, bound) = socket_listen(addr)?;
        if let Some(id) = tls_creds {
            return Err(Error::generic(format!("No TLS credentials with id '{id}'")));
        }
        let inner = Arc::new(Inner {
            listener: Some(listener),
            addr: Some(bound),
            handshake_max_secs,
            max_connections,
            state: Mutex::new(State::default()),
            cond: Condvar::new(),
            accept_thread: Mutex::new(None),
            on_close: CloseNotify::default(),
        });
        Ok(NbdServer { inner })
    }

    /// Starts accepting connections on a server made by [`NbdServer::bind_addr`]. Does
    /// nothing if it already does or has no listening socket.
    pub fn start_accepting(&self) -> Result<()> {
        let mut at = self.inner.accept_thread.lock().unwrap_or_else(|e| e.into_inner());
        if at.is_some() || self.inner.listener.is_none() {
            return Ok(());
        }
        let i2 = self.inner.clone();
        let t = thread::Builder::new()
            .name("nbd-listener".into())
            .spawn(move || accept_loop(&i2))
            .map_err(|e| Error::from_io("Failed to start the NBD listener", e))?;
        *at = Some(t);
        Ok(())
    }

    /// A server without a listening socket, for a caller that accepts connections itself and
    /// hands them to [`NbdServer::serve_stream`].
    pub fn detached(handshake_max_secs: u32, max_connections: u32) -> NbdServer {
        NbdServer {
            inner: Arc::new(Inner {
                listener: None,
                addr: None,
                handshake_max_secs,
                max_connections,
                state: Mutex::new(State::default()),
                cond: Condvar::new(),
                accept_thread: Mutex::new(None),
                on_close: CloseNotify::default(),
            }),
        }
    }

    /// The address the server listens on, with the port the system picked if the caller
    /// asked for port 0.
    pub fn local_addr(&self) -> Option<&SocketAddress> {
        self.inner.addr.as_ref()
    }

    /// `nbd_client_new()` on a connection accepted elsewhere. The connection runs on a new
    /// thread.
    pub fn serve_stream(&self, stream: NbdStream) {
        spawn_client(&self.inner, stream);
    }

    /// Calls `f` whenever a connection goes away, with whether its handshake had finished:
    /// the `close_fn` qemu-nbd passes to `nbd_client_new()`. It runs on the connection's
    /// thread after the connection was removed from [`NbdServer::connections`].
    pub fn set_close_notify(&self, f: impl Fn(bool) + Send + Sync + 'static) {
        *self.inner.on_close.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
    }

    /// The number of connections open right now.
    pub fn connections(&self) -> usize {
        self.inner.lock().conns.len()
    }

    /// The ids of the exports, oldest first.
    pub fn export_ids(&self) -> Vec<String> {
        self.inner.lock().exports.iter().map(|e| e.id.clone()).collect()
    }

    /// Whether a connection uses the export with id `id`: its `refcount > 1`.
    pub fn export_in_use(&self, id: &str) -> bool {
        self.inner.lock().conns.values().any(|c| c.export.as_ref().is_some_and(|e| e.id == id))
    }

    /// `blk_exp_add()` with `nbd_export_create()`: `block-export-add` for type `nbd`.
    pub fn export_add(&self, graph: &BlockGraph, opts: &BlockExportOptions) -> Result<()> {
        self.export_add_with(graph, opts, NbdExportExtras::default())
    }

    /// [`NbdServer::export_add`] with the block status source and dirty bitmaps to use.
    pub fn export_add_with(
        &self,
        graph: &BlockGraph,
        opts: &BlockExportOptions,
        extras: NbdExportExtras,
    ) -> Result<()> {
        let BlockExportOptionsU::Nbd(arg) = &opts.u;
        if opts.fixed_iothread == Some(true)
            && matches!(opts.iothread, Some(ruvm_qapi::types::BlockExportIothreads::Multi(_)))
        {
            return Err(Error::generic("Cannot use fixed-iothread for a multi-threaded export"));
        }
        if !id_wellformed(&opts.id) {
            return Err(Error::generic("Invalid block export id"));
        }
        if self.inner.lock().exports.iter().any(|e| e.id == opts.id) {
            return Err(Error::generic(format!("Block export id '{}' is already in use", opts.id)));
        }
        let Some(node) = graph.node(&opts.node_name) else {
            return Err(Error::generic(format!(
                "Cannot find device='' nor node-name='{}'",
                opts.node_name
            )));
        };
        let writable = opts.writable.unwrap_or(false);
        if node.read_only && writable {
            return Err(Error::generic("Cannot export read-only node as writable"));
        }
        if let Some(iot) = &opts.iothread {
            let id = match iot {
                ruvm_qapi::types::BlockExportIothreads::Single(s) => Some(s.as_str()),
                ruvm_qapi::types::BlockExportIothreads::Multi(v) => {
                    if v.is_empty() {
                        return Err(Error::generic("The set of I/O threads must not be empty"));
                    }
                    Some(v[0].as_str())
                }
            };
            if let Some(id) = id {
                // There are no iothread objects yet. QEMU ignores a missing iothread
                // unless fixed-iothread is set, but a list must name existing ones.
                let multi = matches!(iot, ruvm_qapi::types::BlockExportIothreads::Multi(_));
                if opts.fixed_iothread == Some(true) || multi {
                    return Err(Error::generic(format!("iothread \"{id}\" not found")));
                }
            }
        }
        let mut perm = BLK_PERM_CONSISTENT_READ;
        if writable {
            perm |= BLK_PERM_WRITE;
        }
        let blk = BlockBackend::new(graph, &opts.node_name, perm, BLK_PERM_ALL)?;
        blk.set_enable_write_cache(!opts.writethrough.unwrap_or(false));
        let mut extras = extras;
        if extras.status.is_none() {
            if let Some(n) = graph.find_node(&opts.node_name) {
                extras.status = Some(Arc::new(NodeStatus { node: n }));
            }
        }
        let exp =
            self.export_create(&opts.id, &opts.node_name, node.read_only, arg, blk, extras)?;
        let mut st = self.inner.lock();
        // The id and name checks ran without the lock held; run them again now that the
        // export is about to go in.
        if st.exports.iter().any(|e| e.id == exp.id) {
            return Err(Error::generic(format!("Block export id '{}' is already in use", exp.id)));
        }
        if st.exports.iter().any(|e| e.name == exp.name) {
            return Err(Error::generic(format!(
                "NBD server already has export named '{}'",
                exp.name
            )));
        }
        st.exports.push(Arc::new(exp));
        Ok(())
    }

    /// `nbd_export_create()`.
    fn export_create(
        &self,
        id: &str,
        node_name: &str,
        node_read_only: bool,
        arg: &BlockExportOptionsNbd,
        blk: Arc<BlockBackend>,
        extras: NbdExportExtras,
    ) -> Result<Export> {
        let name = arg.name.clone().unwrap_or_else(|| node_name.to_string());
        let readonly = !blk.perm().0 & BLK_PERM_WRITE != 0;
        if name.len() > NBD_MAX_STRING_SIZE {
            return Err(Error::generic(format!("export name '{name}' too long")));
        }
        if let Some(d) = &arg.description {
            if d.len() > NBD_MAX_STRING_SIZE {
                return Err(Error::generic(format!("description '{d}' too long")));
            }
        }
        if self.inner.find_export(&name).is_some() {
            return Err(Error::generic(format!("NBD server already has export named '{name}'")));
        }
        let size = blk
            .getlength()
            .map_err(|e| Error::from_io("Failed to determine the NBD export's length", e))?;
        let (perm, shared) = blk.perm();
        blk.set_perm(perm, shared & !BLK_PERM_RESIZE)?;

        let mut nbdflags =
            NBD_FLAG_HAS_FLAGS | NBD_FLAG_SEND_FLUSH | NBD_FLAG_SEND_FUA | NBD_FLAG_SEND_CACHE;
        if self.inner.max_connections != 1 {
            nbdflags |= NBD_FLAG_CAN_MULTI_CONN;
        }
        if readonly {
            nbdflags |= NBD_FLAG_READ_ONLY;
        } else {
            nbdflags |= NBD_FLAG_SEND_TRIM | NBD_FLAG_SEND_WRITE_ZEROES | NBD_FLAG_SEND_FAST_ZERO;
        }

        let mut bitmaps = Vec::new();
        for b in arg.bitmaps.iter().flatten() {
            let bm = match b {
                BlockDirtyBitmapOrStr::Local(bitmap) => {
                    let Some((_, bm)) = extras
                        .bitmaps
                        .iter()
                        .find(|(n, bm)| n == node_name && bm.name() == *bitmap)
                    else {
                        return Err(Error::generic(format!("Bitmap '{bitmap}' is not found")));
                    };
                    if readonly && !node_read_only && bm.enabled() {
                        return Err(Error::generic(format!(
                            "Enabled bitmap '{bitmap}' incompatible with readonly export"
                        )));
                    }
                    bm.clone()
                }
                BlockDirtyBitmapOrStr::External(ext) => {
                    let found = extras.bitmaps.iter().find(|(n, _)| *n == ext.node);
                    if found.is_none() {
                        return Err(Error::generic(format!("Node '{}' not found", ext.node)));
                    }
                    let Some((_, bm)) = extras
                        .bitmaps
                        .iter()
                        .find(|(n, bm)| *n == ext.node && bm.name() == ext.name)
                    else {
                        return Err(Error::generic(format!(
                            "Dirty bitmap '{}' not found",
                            ext.name
                        )));
                    };
                    bm.clone()
                }
            };
            bitmaps.push(bm);
        }
        let bitmap_names = bitmaps.iter().map(|b| b.name()).collect();

        Ok(Export {
            id: id.to_string(),
            name,
            description: arg.description.clone(),
            blk,
            size: size / BDRV_SECTOR_SIZE * BDRV_SECTOR_SIZE,
            nbdflags,
            bitmaps,
            bitmap_names,
            allocation_depth: arg.allocation_depth.unwrap_or(false),
            status: extras.status.unwrap_or_else(|| Arc::new(AllData)),
        })
    }

    /// `qmp_block_export_del()`: remove the export with id `id`. In safe mode (the default)
    /// this fails while clients use the export; in hard mode they are disconnected. Returns
    /// once no connection uses the export any more.
    pub fn export_remove(&self, id: &str, mode: Option<BlockExportRemoveMode>) -> Result<()> {
        let mut st = self.inner.lock();
        let Some(pos) = st.exports.iter().position(|e| e.id == id) else {
            return Err(Error::generic(format!("Export '{id}' is not found")));
        };
        let exp = st.exports[pos].clone();
        let in_use = |st: &State| {
            st.conns.values().any(|c| c.export.as_ref().is_some_and(|e| Arc::ptr_eq(e, &exp)))
        };
        if mode != Some(BlockExportRemoveMode::Hard) && in_use(&st) {
            return Err(Error::generic(format!("export '{id}' still in use"))
                .hint("Use mode='hard' to force client disconnect\n"));
        }
        st.exports.remove(pos);
        for c in st.conns.values() {
            if c.export.as_ref().is_some_and(|e| Arc::ptr_eq(e, &exp)) {
                c.stream.shutdown();
            }
        }
        while in_use(&st) {
            st = self.inner.cond.wait(st).unwrap_or_else(|e| e.into_inner());
        }
        Ok(())
    }

    /// `qmp_nbd_server_add()`: the older way to add an export.
    pub fn nbd_server_add(&self, graph: &BlockGraph, arg: &NbdServerAddOptions) -> Result<()> {
        let Some(node_name) = resolve_device(graph, &arg.device) else {
            let d = &arg.device;
            return Err(Error::generic(format!("Cannot find device='{d}' nor node-name='{d}'")));
        };
        let name = arg.name.clone().unwrap_or_else(|| arg.device.clone());
        let mut writable = arg.writable;
        if graph.node(&node_name).is_some_and(|n| n.read_only) {
            writable = Some(false);
        }
        let opts = BlockExportOptions {
            id: name.clone(),
            node_name,
            writable,
            u: BlockExportOptionsU::Nbd(BlockExportOptionsNbd {
                name: Some(name),
                description: arg.description.clone(),
                bitmaps: arg.bitmap.clone().map(|b| vec![BlockDirtyBitmapOrStr::Local(b)]),
                allocation_depth: None,
            }),
            ..BlockExportOptions::default()
        };
        self.export_add(graph, &opts)
    }

    /// `nbd_server_free()` after `blk_exp_close_all_type()`: close the listener, drop every
    /// export and connection, and wait for the connection threads to finish.
    pub fn stop(&self) {
        {
            let mut st = self.inner.lock();
            st.stopping = true;
            st.exports.clear();
            for c in st.conns.values() {
                c.stream.shutdown();
            }
            self.inner.cond.notify_all();
        }
        let t = self.inner.accept_thread.lock().unwrap().take();
        if let Some(t) = t {
            let _ = t.join();
        }
        let mut st = self.inner.lock();
        while !st.conns.is_empty() {
            st = self.inner.cond.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// `bdrv_lookup_bs(device, device)`: the node a device or node name refers to.
fn resolve_device(graph: &BlockGraph, device: &str) -> Option<String> {
    if let Some(blk) = graph.backend(device) {
        return blk.node_name();
    }
    graph.node(device).map(|n| n.node_name)
}

fn accept_loop(inner: &Arc<Inner>) {
    let Some(listener) = inner.listener.as_ref() else {
        return;
    };
    loop {
        {
            let mut st = inner.lock();
            // nbd_update_server_watch(): stop accepting while at the connection limit.
            while !st.stopping
                && inner.max_connections != 0
                && st.conns.len() >= inner.max_connections as usize
            {
                st = inner.cond.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.stopping {
                return;
            }
        }
        match listener.accept() {
            Ok(Some(s)) => spawn_client(inner, s),
            // QIONetListener ignores failed accepts.
            Ok(None) | Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }
}

fn spawn_client(inner: &Arc<Inner>, stream: NbdStream) {
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let id = {
        let mut st = inner.lock();
        if st.stopping {
            stream.shutdown();
            return;
        }
        let id = st.next_conn;
        st.next_conn += 1;
        st.conns.insert(id, Conn { stream: clone, export: None });
        id
    };
    let i2 = inner.clone();
    let r = thread::Builder::new().name("nbd-client".into()).spawn(move || {
        let mut c = Client::new(i2.clone(), id, stream);
        let negotiated = c.run();
        drop(c);
        let mut st = i2.lock();
        st.conns.remove(&id);
        i2.cond.notify_all();
        drop(st);
        let hook = i2.on_close.0.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some(hook) = hook {
            hook(negotiated);
        }
    });
    if r.is_err() {
        let mut st = inner.lock();
        if let Some(c) = st.conns.remove(&id) {
            c.stream.shutdown();
        }
        inner.cond.notify_all();
    }
}

/// `NBDMetaContexts`: the contexts a client selected, for one export.
#[derive(Clone, Debug, Default)]
struct Meta {
    exp: Option<Arc<Export>>,
    count: usize,
    base_allocation: bool,
    allocation_depth: bool,
    bitmaps: Vec<bool>,
}

fn same_export(a: &Option<Arc<Export>>, b: &Arc<Export>) -> bool {
    a.as_ref().is_some_and(|a| Arc::ptr_eq(a, b))
}

/// A request error: the errno and, except for a quiet disconnect, the message.
#[derive(Debug)]
struct Fail {
    errno: i32,
    err: Option<Error>,
}

impl Fail {
    fn new(errno: i32, msg: impl Into<String>) -> Fail {
        Fail { errno, err: Some(Error::generic(msg)) }
    }

    fn eio(err: Error) -> Fail {
        Fail { errno: libc::EIO, err: Some(err) }
    }
}

/// `NBDRequest` with the server side extras.
#[derive(Debug, Default)]
struct Request {
    cookie: u64,
    from: u64,
    len: u64,
    flags: u16,
    typ: u16,
    /// `None`: the contexts the client negotiated; `Some`: the ones a payload asked for.
    contexts: Option<Meta>,
}

/// `NBDClient`.
#[derive(Debug)]
struct Client {
    inner: Arc<Inner>,
    id: u64,
    ioc: NbdStream,
    mode: NbdMode,
    opt: u32,
    optlen: u32,
    exp: Option<Arc<Export>>,
    check_align: u32,
    contexts: Meta,
}

/// `nbd_sanitize_name()`.
fn sanitize_name(name: &str) -> String {
    let b = name.as_bytes();
    if b.len() < 80 || !b[..80].iter().all(|&c| c != 0) {
        return name.to_string();
    }
    format!("{}...", String::from_utf8_lossy(&b[..80]))
}

impl Client {
    fn new(inner: Arc<Inner>, id: u64, ioc: NbdStream) -> Client {
        Client {
            inner,
            id,
            ioc,
            mode: NbdMode::Oldstyle,
            opt: 0,
            optlen: 0,
            exp: None,
            check_align: 0,
            contexts: Meta::default(),
        }
    }

    /// `nbd_co_client_start()` followed by the request loop. Returns whether the handshake
    /// finished, the `negotiated` of `client_close()`.
    fn run(&mut self) -> bool {
        let timer = self.start_handshake_timer();
        let r = self.negotiate();
        if let Some(t) = timer {
            let (m, c) = &*t;
            *m.lock().unwrap() = true;
            c.notify_all();
        }
        let negotiated = match r {
            Ok(true) => {
                self.transmission();
                true
            }
            Ok(false) => false,
            Err(e) => {
                report(&e);
                false
            }
        };
        self.ioc.shutdown();
        negotiated
    }

    #[allow(clippy::type_complexity)]
    fn start_handshake_timer(&self) -> Option<Arc<(Mutex<bool>, Condvar)>> {
        if self.inner.handshake_max_secs == 0 {
            return None;
        }
        let stream = self.ioc.try_clone().ok()?;
        let t = Arc::new((Mutex::new(false), Condvar::new()));
        let t2 = t.clone();
        let dur = Duration::from_secs(u64::from(self.inner.handshake_max_secs));
        let r = thread::Builder::new().name("nbd-handshake-timer".into()).spawn(move || {
            let (m, c) = &*t2;
            let g = m.lock().unwrap();
            let (g, _) = c.wait_timeout_while(g, dur, |done| !*done).unwrap();
            if !*g {
                // nbd_handshake_timer_cb().
                stream.shutdown();
            }
        });
        r.ok().map(|_| t)
    }

    fn attach(&mut self, exp: Arc<Export>) {
        let mut st = self.inner.lock();
        if let Some(c) = st.conns.get_mut(&self.id) {
            c.export = Some(exp.clone());
        }
        self.exp = Some(exp);
    }

    fn find_export(&self, name: &str) -> Option<Arc<Export>> {
        self.inner.find_export(name)
    }

    // The option phase.

    /// `nbd_negotiate_send_rep_len()`.
    fn send_rep_len(&mut self, typ: u32, len: u32) -> Result<()> {
        assert!(len < NBD_MAX_BUFFER_SIZE);
        let mut b = Vec::with_capacity(20);
        b.extend_from_slice(&NBD_REP_MAGIC.to_be_bytes());
        b.extend_from_slice(&self.opt.to_be_bytes());
        b.extend_from_slice(&typ.to_be_bytes());
        b.extend_from_slice(&len.to_be_bytes());
        nbd_write(&mut self.ioc, &b)
    }

    fn send_rep(&mut self, typ: u32) -> Result<()> {
        self.send_rep_len(typ, 0)
    }

    /// `nbd_negotiate_send_rep_err()`.
    fn send_rep_err(&mut self, typ: u32, msg: &str) -> Result<()> {
        assert!(msg.len() < NBD_MAX_STRING_SIZE);
        self.send_rep_len(typ, msg.len() as u32)?;
        nbd_write(&mut self.ioc, msg.as_bytes())
            .map_err(|e| e.prepend("write failed (error message): "))
    }

    /// `nbd_opt_drop()`: skip the rest of the option and send an error. `Ok(false)` is QEMU's 0.
    fn opt_drop(&mut self, typ: u32, msg: &str) -> Result<bool> {
        let r = nbd_drop(&mut self.ioc, u64::from(self.optlen));
        self.optlen = 0;
        r?;
        self.send_rep_err(typ, msg)?;
        Ok(false)
    }

    fn opt_invalid(&mut self, msg: &str) -> Result<bool> {
        self.opt_drop(NBD_REP_ERR_INVALID, msg)
    }

    /// `nbd_opt_read()`.
    fn opt_read(&mut self, buf: &mut [u8], check_nul: bool) -> Result<bool> {
        if buf.len() > self.optlen as usize {
            let m = format!("Inconsistent lengths in option {}", nbd_opt_lookup(self.opt));
            return self.opt_invalid(&m);
        }
        self.optlen -= buf.len() as u32;
        read_all(&mut self.ioc, buf)?;
        if check_nul && buf.contains(&0) {
            let m = format!("Unexpected embedded NUL in option {}", nbd_opt_lookup(self.opt));
            return self.opt_invalid(&m);
        }
        Ok(true)
    }

    /// `nbd_opt_skip()`.
    fn opt_skip(&mut self, size: u32) -> Result<bool> {
        if size > self.optlen {
            let m = format!("Inconsistent lengths in option {}", nbd_opt_lookup(self.opt));
            return self.opt_invalid(&m);
        }
        self.optlen -= size;
        nbd_drop(&mut self.ioc, u64::from(size))?;
        Ok(true)
    }

    fn opt_read_u16(&mut self) -> Result<Option<u16>> {
        let mut b = [0u8; 2];
        Ok(self.opt_read(&mut b, false)?.then(|| u16::from_be_bytes(b)))
    }

    fn opt_read_u32(&mut self) -> Result<Option<u32>> {
        let mut b = [0u8; 4];
        Ok(self.opt_read(&mut b, false)?.then(|| u32::from_be_bytes(b)))
    }

    /// `nbd_opt_read_name()`: the name as sent, `None` if an error reply went out instead.
    fn opt_read_name(&mut self) -> Result<Option<Vec<u8>>> {
        let Some(len) = self.opt_read_u32()? else {
            return Ok(None);
        };
        if len as usize > NBD_MAX_STRING_SIZE {
            return self.opt_invalid(&format!("Invalid name length: {len}")).map(|_| None);
        }
        let mut name = vec![0u8; len as usize];
        if !self.opt_read(&mut name, true)? {
            return Ok(None);
        }
        Ok(Some(name))
    }

    /// `nbd_negotiate_handle_list()`.
    fn handle_list(&mut self) -> Result<bool> {
        let exports = self.inner.lock().exports.clone();
        for exp in exports {
            let desc = exp.description.as_deref().unwrap_or("");
            let len = (exp.name.len() + desc.len() + 4) as u32;
            self.send_rep_len(NBD_REP_SERVER, len)?;
            nbd_write(&mut self.ioc, &(exp.name.len() as u32).to_be_bytes())
                .map_err(|e| e.prepend("write failed (name length): "))?;
            nbd_write(&mut self.ioc, exp.name.as_bytes())
                .map_err(|e| e.prepend("write failed (name buffer): "))?;
            nbd_write(&mut self.ioc, desc.as_bytes())
                .map_err(|e| e.prepend("write failed (description buffer): "))?;
        }
        self.send_rep(NBD_REP_ACK)?;
        Ok(false)
    }

    /// `nbd_check_meta_export()`.
    fn check_meta_export(&mut self, exp: &Arc<Export>) {
        if !same_export(&self.contexts.exp, exp) {
            self.contexts.count = 0;
        }
    }

    /// `nbd_negotiate_handle_export_name()`.
    fn handle_export_name(&mut self, no_zeroes: bool) -> Result<bool> {
        if self.mode >= NbdMode::Extended {
            return Err(Error::generic("Extended headers already negotiated"));
        }
        if self.optlen as usize > NBD_MAX_STRING_SIZE {
            return Err(Error::generic("Bad length received"));
        }
        let mut name = vec![0u8; self.optlen as usize];
        nbd_read(&mut self.ioc, &mut name, Some("export name"))?;
        self.optlen = 0;
        // The name is a C string on QEMU's side: it ends at the first NUL.
        if let Some(p) = name.iter().position(|&c| c == 0) {
            name.truncate(p);
        }
        let Some(exp) = self.find_export(&String::from_utf8_lossy(&name)) else {
            return Err(Error::generic("export not found"));
        };
        self.check_meta_export(&exp);
        let mut myflags = exp.nbdflags;
        if self.mode >= NbdMode::Structured {
            myflags |= NBD_FLAG_SEND_DF;
        }
        if self.mode >= NbdMode::Extended && self.contexts.count > 0 {
            myflags |= NBD_FLAG_BLOCK_STAT_PAYLOAD;
        }
        let mut buf = [0u8; 10 + 124];
        buf[..8].copy_from_slice(&exp.size.to_be_bytes());
        buf[8..10].copy_from_slice(&myflags.to_be_bytes());
        let len = if no_zeroes { 10 } else { buf.len() };
        nbd_write(&mut self.ioc, &buf[..len]).map_err(|e| e.prepend("write failed: "))?;
        self.attach(exp);
        Ok(true)
    }

    /// `nbd_negotiate_send_info()`.
    fn send_info(&mut self, info: u16, buf: &[u8]) -> Result<()> {
        self.send_rep_len(NBD_REP_INFO, 2 + buf.len() as u32)?;
        nbd_write(&mut self.ioc, &info.to_be_bytes())?;
        nbd_write(&mut self.ioc, buf)
    }

    /// `nbd_reject_length()`.
    fn reject_length(&mut self, fatal: bool) -> Result<bool> {
        assert!(self.optlen != 0);
        let m = format!("option '{}' has unexpected length", nbd_opt_lookup(self.opt));
        let r = self.opt_invalid(&m)?;
        if fatal {
            return Err(Error::generic(m));
        }
        Ok(r)
    }

    /// `nbd_negotiate_handle_info()`: `Ok(true)` once `NBD_OPT_GO` succeeded.
    fn handle_info(&mut self) -> Result<bool> {
        let Some(name) = self.opt_read_name()? else {
            return Ok(false);
        };
        let Some(mut requests) = self.opt_read_u16()? else {
            return Ok(false);
        };
        let mut sendname = false;
        let mut blocksize = false;
        while requests > 0 {
            requests -= 1;
            let Some(request) = self.opt_read_u16()? else {
                return Ok(false);
            };
            match request {
                NBD_INFO_NAME => sendname = true,
                NBD_INFO_BLOCK_SIZE => blocksize = true,
                _ => {}
            }
        }
        if self.optlen != 0 {
            return self.reject_length(false);
        }
        let name_str = String::from_utf8_lossy(&name).into_owned();
        let Some(exp) = self.find_export(&name_str) else {
            let m = format!("export '{}' not present", sanitize_name(&name_str));
            self.send_rep_err(NBD_REP_ERR_UNKNOWN, &m)?;
            return Ok(false);
        };
        if self.opt == NBD_OPT_GO {
            self.check_meta_export(&exp);
        }
        if sendname {
            self.send_info(NBD_INFO_NAME, &name)?;
        }
        if let Some(d) = &exp.description {
            self.send_info(NBD_INFO_DESCRIPTION, d.as_bytes())?;
        }
        // The block layer has no alignment restrictions to report yet.
        let request_alignment: u32 = 1;
        let mut check_align = 0;
        let min = if self.opt == NBD_OPT_INFO || blocksize {
            check_align = request_alignment;
            request_alignment
        } else {
            1
        };
        let sizes = [min, min.max(4096), NBD_MAX_BUFFER_SIZE];
        let mut b = Vec::with_capacity(12);
        for s in sizes {
            b.extend_from_slice(&s.to_be_bytes());
        }
        self.send_info(NBD_INFO_BLOCK_SIZE, &b)?;

        let mut myflags = exp.nbdflags;
        if self.mode >= NbdMode::Structured {
            myflags |= NBD_FLAG_SEND_DF;
        }
        if self.mode >= NbdMode::Extended && (self.contexts.count > 0 || self.opt == NBD_OPT_INFO) {
            myflags |= NBD_FLAG_BLOCK_STAT_PAYLOAD;
        }
        let mut b = Vec::with_capacity(10);
        b.extend_from_slice(&exp.size.to_be_bytes());
        b.extend_from_slice(&myflags.to_be_bytes());
        self.send_info(NBD_INFO_EXPORT, &b)?;

        if self.opt == NBD_OPT_INFO && !blocksize && request_alignment > 1 {
            self.send_rep_err(
                NBD_REP_ERR_BLOCK_SIZE_REQD,
                "request NBD_INFO_BLOCK_SIZE to use this export",
            )?;
            return Ok(false);
        }
        self.send_rep(NBD_REP_ACK)?;
        if self.opt == NBD_OPT_GO {
            self.check_align = check_align;
            self.attach(exp);
            return Ok(true);
        }
        Ok(false)
    }

    /// `nbd_negotiate_send_meta_context()`.
    fn send_meta_context(&mut self, context: &str, mut id: u32) -> Result<()> {
        assert!(context.len() <= NBD_MAX_STRING_SIZE);
        if self.opt == NBD_OPT_LIST_META_CONTEXT {
            id = 0;
        }
        let mut b = Vec::with_capacity(24 + context.len());
        b.extend_from_slice(&NBD_REP_MAGIC.to_be_bytes());
        b.extend_from_slice(&self.opt.to_be_bytes());
        b.extend_from_slice(&NBD_REP_META_CONTEXT.to_be_bytes());
        b.extend_from_slice(&(4 + context.len() as u32).to_be_bytes());
        b.extend_from_slice(&id.to_be_bytes());
        b.extend_from_slice(context.as_bytes());
        nbd_write(&mut self.ioc, &b)
    }

    /// `nbd_meta_empty_or_pattern()`.
    fn empty_or_pattern(&self, pattern: &str, query: &str) -> bool {
        if query.is_empty() {
            return self.opt == NBD_OPT_LIST_META_CONTEXT;
        }
        query == pattern
    }

    /// `nbd_meta_base_query()`.
    fn meta_base_query(&self, meta: &mut Meta, query: &str) -> bool {
        let Some(q) = query.strip_prefix("base:") else {
            return false;
        };
        if self.empty_or_pattern("allocation", q) {
            meta.base_allocation = true;
        }
        true
    }

    /// `nbd_meta_qemu_query()`.
    fn meta_qemu_query(&self, meta: &mut Meta, query: &str) -> bool {
        let Some(q) = query.strip_prefix("qemu:") else {
            return false;
        };
        let exp = meta.exp.clone().expect("meta context export");
        let list = self.opt == NBD_OPT_LIST_META_CONTEXT;
        if q.is_empty() {
            if list {
                meta.allocation_depth = exp.allocation_depth;
                meta.bitmaps.iter_mut().for_each(|b| *b = true);
            }
            return true;
        }
        if q == "allocation-depth" {
            meta.allocation_depth = exp.allocation_depth;
            return true;
        }
        if let Some(q) = q.strip_prefix("dirty-bitmap:") {
            if q.is_empty() {
                if list {
                    meta.bitmaps.iter_mut().for_each(|b| *b = true);
                }
                return true;
            }
            if let Some(i) = exp.bitmap_names.iter().position(|n| n == q) {
                meta.bitmaps[i] = true;
            }
        }
        true
    }

    /// `nbd_negotiate_meta_query()`.
    fn meta_query(&mut self, meta: &mut Meta) -> Result<bool> {
        let Some(len) = self.opt_read_u32()? else {
            return Ok(false);
        };
        if len as usize > NBD_MAX_STRING_SIZE {
            return self.opt_skip(len);
        }
        let mut q = vec![0u8; len as usize];
        if !self.opt_read(&mut q, true)? {
            return Ok(false);
        }
        let q = String::from_utf8_lossy(&q);
        if self.meta_base_query(meta, &q) {
            return Ok(true);
        }
        self.meta_qemu_query(meta, &q);
        Ok(true)
    }

    /// `nbd_negotiate_meta_queries()`.
    fn meta_queries(&mut self) -> Result<bool> {
        if self.opt == NBD_OPT_SET_META_CONTEXT && self.mode < NbdMode::Structured {
            let m = format!(
                "request option '{}' when structured reply is not negotiated",
                nbd_opt_lookup(self.opt)
            );
            return self.opt_invalid(&m);
        }
        let set = self.opt == NBD_OPT_SET_META_CONTEXT;
        if set {
            self.contexts = Meta::default();
        }
        let mut meta = Meta::default();
        let r = self.meta_queries_into(&mut meta);
        if set {
            self.contexts = meta;
        }
        r
    }

    fn meta_queries_into(&mut self, meta: &mut Meta) -> Result<bool> {
        let Some(name) = self.opt_read_name()? else {
            return Ok(false);
        };
        let name = String::from_utf8_lossy(&name).into_owned();
        let Some(exp) = self.find_export(&name) else {
            let m = format!("export '{}' not present", sanitize_name(&name));
            return self.opt_drop(NBD_REP_ERR_UNKNOWN, &m);
        };
        meta.exp = Some(exp.clone());
        meta.bitmaps = vec![false; exp.bitmaps.len()];
        let Some(nb_queries) = self.opt_read_u32()? else {
            return Ok(false);
        };
        if self.opt == NBD_OPT_LIST_META_CONTEXT && nb_queries == 0 {
            meta.base_allocation = true;
            meta.allocation_depth = exp.allocation_depth;
            meta.bitmaps.iter_mut().for_each(|b| *b = true);
        } else {
            for _ in 0..nb_queries {
                if !self.meta_query(meta)? {
                    return Ok(false);
                }
            }
        }
        let mut count = 0;
        if meta.base_allocation {
            self.send_meta_context("base:allocation", NBD_META_ID_BASE_ALLOCATION)?;
            count += 1;
        }
        if meta.allocation_depth {
            self.send_meta_context("qemu:allocation-depth", NBD_META_ID_ALLOCATION_DEPTH)?;
            count += 1;
        }
        for (i, name) in exp.bitmap_names.iter().enumerate() {
            if !meta.bitmaps[i] {
                continue;
            }
            let ctx = format!("qemu:dirty-bitmap:{name}");
            self.send_meta_context(&ctx, NBD_META_ID_DIRTY_BITMAP + i as u32)?;
            count += 1;
        }
        self.send_rep(NBD_REP_ACK)?;
        meta.count = count;
        Ok(false)
    }

    /// `nbd_negotiate_options()`: `Ok(true)` to go on to transmission, `Ok(false)` to close
    /// the connection quietly.
    fn negotiate_options(&mut self) -> Result<bool> {
        // A failure to read the flags is not reported: it is probably a port probe.
        let Ok(mut flags) = nbd_read32(&mut self.ioc, "flags") else {
            return Ok(false);
        };
        self.mode = NbdMode::ExportName;
        let mut fixed = false;
        if flags & NBD_FLAG_C_FIXED_NEWSTYLE != 0 {
            fixed = true;
            flags &= !NBD_FLAG_C_FIXED_NEWSTYLE;
            self.mode = NbdMode::Simple;
        }
        let mut no_zeroes = false;
        if flags & NBD_FLAG_C_NO_ZEROES != 0 {
            no_zeroes = true;
            flags &= !NBD_FLAG_C_NO_ZEROES;
        }
        if flags != 0 {
            return Err(Error::generic(format!("Unknown client flags 0x{flags:x} received")));
        }
        loop {
            let magic = nbd_read64(&mut self.ioc, "opts magic")?;
            if magic != NBD_OPTS_MAGIC {
                return Err(Error::generic("Bad magic received"));
            }
            let option = nbd_read32(&mut self.ioc, "option")?;
            self.opt = option;
            let length = nbd_read32(&mut self.ioc, "option length")?;
            assert_eq!(self.optlen, 0);
            self.optlen = length;
            if length > NBD_MAX_BUFFER_SIZE {
                return Err(Error::generic(format!(
                    "len ({length}) is larger than max len ({NBD_MAX_BUFFER_SIZE})"
                )));
            }
            if !fixed {
                if option == NBD_OPT_EXPORT_NAME {
                    return self.handle_export_name(no_zeroes);
                }
                return Err(Error::generic(format!(
                    "Unsupported option {option} ({})",
                    nbd_opt_lookup(option)
                )));
            }
            match option {
                NBD_OPT_LIST => {
                    if length != 0 {
                        self.reject_length(false)?;
                    } else {
                        self.handle_list()?;
                    }
                }
                NBD_OPT_ABORT => {
                    let _ = self.send_rep(NBD_REP_ACK);
                    return Ok(false);
                }
                NBD_OPT_EXPORT_NAME => return self.handle_export_name(no_zeroes),
                NBD_OPT_INFO | NBD_OPT_GO => {
                    if self.handle_info()? {
                        return Ok(true);
                    }
                }
                NBD_OPT_STARTTLS => {
                    if length != 0 {
                        self.reject_length(false)?;
                    } else {
                        self.send_rep_err(NBD_REP_ERR_POLICY, "TLS not configured")?;
                    }
                }
                NBD_OPT_STRUCTURED_REPLY => {
                    if length != 0 {
                        self.reject_length(false)?;
                    } else if self.mode >= NbdMode::Extended {
                        self.send_rep_err(
                            NBD_REP_ERR_EXT_HEADER_REQD,
                            "extended headers already negotiated",
                        )?;
                    } else if self.mode >= NbdMode::Structured {
                        self.send_rep_err(
                            NBD_REP_ERR_INVALID,
                            "structured reply already negotiated",
                        )?;
                    } else {
                        self.send_rep(NBD_REP_ACK)?;
                        self.mode = NbdMode::Structured;
                    }
                }
                NBD_OPT_LIST_META_CONTEXT | NBD_OPT_SET_META_CONTEXT => {
                    self.meta_queries()?;
                }
                NBD_OPT_EXTENDED_HEADERS => {
                    if length != 0 {
                        self.reject_length(false)?;
                    } else if self.mode >= NbdMode::Extended {
                        self.send_rep_err(
                            NBD_REP_ERR_INVALID,
                            "extended headers already negotiated",
                        )?;
                    } else {
                        self.send_rep(NBD_REP_ACK)?;
                        self.mode = NbdMode::Extended;
                    }
                }
                _ => {
                    let m = format!("Unsupported option {option} ({})", nbd_opt_lookup(option));
                    self.opt_drop(NBD_REP_ERR_UNSUP, &m)?;
                }
            }
        }
    }

    /// `nbd_negotiate()`.
    fn negotiate(&mut self) -> Result<bool> {
        let mut buf = [0u8; 18];
        buf[..8].copy_from_slice(b"NBDMAGIC");
        buf[8..16].copy_from_slice(&NBD_OPTS_MAGIC.to_be_bytes());
        buf[16..].copy_from_slice(&(NBD_FLAG_FIXED_NEWSTYLE | NBD_FLAG_NO_ZEROES).to_be_bytes());
        // Failing to send the greeting is not worth a message either.
        if nbd_write(&mut self.ioc, &buf).is_err() {
            return Ok(false);
        }
        let r = self.negotiate_options().map_err(|e| e.prepend("option negotiation failed: "))?;
        if r {
            assert_eq!(self.optlen, 0);
        }
        Ok(r)
    }

    // The transmission phase.

    fn exp(&self) -> &Arc<Export> {
        self.exp.as_ref().expect("export")
    }

    /// `nbd_receive_request()`. `Ok(None)` on a clean end-of-file.
    fn receive_request(&mut self, req: &mut Request) -> std::result::Result<(), Fail> {
        let ext = self.mode >= NbdMode::Extended;
        let size = if ext { NBD_EXTENDED_REQUEST_SIZE } else { NBD_REQUEST_SIZE };
        let mut buf = [0u8; NBD_EXTENDED_REQUEST_SIZE];
        match nbd_read_eof(&mut self.ioc, &mut buf[..size]) {
            Ok(true) => {}
            Ok(false) => return Err(Fail { errno: libc::EIO, err: None }),
            // qio_channel_readv() failures carry no message here, only the errno.
            Err(e) => {
                let quiet = e.message().starts_with("Unable to read from socket");
                return Err(Fail { errno: libc::EIO, err: (!quiet).then_some(e) });
            }
        }
        let magic = be32(&buf);
        req.flags = be16(&buf[4..]);
        req.typ = be16(&buf[6..]);
        req.cookie = be64(&buf[8..]);
        req.from = be64(&buf[16..]);
        let expect = if ext {
            req.len = be64(&buf[24..]);
            NBD_EXTENDED_REQUEST_MAGIC
        } else {
            req.len = u64::from(be32(&buf[24..]));
            NBD_REQUEST_MAGIC
        };
        if magic != expect {
            return Err(Fail::new(
                libc::EINVAL,
                format!("invalid magic (got 0x{magic:x}, expected 0x{expect:x})"),
            ));
        }
        Ok(())
    }

    /// `nbd_co_block_status_payload_read()`.
    fn block_status_payload_read(&mut self, req: &mut Request) -> std::result::Result<(), Fail> {
        let mut payload_len = req.len;
        if payload_len > u64::from(NBD_MAX_BUFFER_SIZE) {
            return Err(Fail::new(
                libc::EINVAL,
                format!("len ({}) is larger than max len ({NBD_MAX_BUFFER_SIZE})", req.len),
            ));
        }
        let exp = self.exp().clone();
        let nr_bitmaps = exp.bitmaps.len();
        let mut rc = Meta { exp: Some(exp), ..Meta::default() };
        let valid = payload_len % 4 == 0
            && payload_len >= 8
            && payload_len <= 8 + 4 * self.contexts.count as u64;
        let mut skip = !valid;
        if valid {
            let mut buf = vec![0u8; payload_len as usize];
            nbd_read(&mut self.ioc, &mut buf, Some("CMD_BLOCK_STATUS data")).map_err(Fail::eio)?;
            rc.bitmaps = vec![false; nr_bitmaps];
            let count = (buf.len() - 8) / 4;
            payload_len = 0;
            for i in 0..count {
                let id = be32(&buf[8 + 4 * i..]);
                if id == NBD_META_ID_BASE_ALLOCATION {
                    if !self.contexts.base_allocation || rc.base_allocation {
                        skip = true;
                        break;
                    }
                    rc.base_allocation = true;
                } else if id == NBD_META_ID_ALLOCATION_DEPTH {
                    if !self.contexts.allocation_depth || rc.allocation_depth {
                        skip = true;
                        break;
                    }
                    rc.allocation_depth = true;
                } else {
                    let idx = id.wrapping_sub(NBD_META_ID_DIRTY_BITMAP) as usize;
                    if idx >= nr_bitmaps || !self.contexts.bitmaps[idx] || rc.bitmaps[idx] {
                        skip = true;
                        break;
                    }
                    rc.bitmaps[idx] = true;
                }
            }
            if !skip {
                req.len = be64(&buf);
                rc.count = count;
                req.contexts = Some(rc);
                return Ok(());
            }
        }
        debug_assert!(skip);
        req.len = 0;
        rc.count = 0;
        req.contexts = Some(rc);
        nbd_drop(&mut self.ioc, payload_len).map_err(Fail::eio)
    }

    /// `nbd_co_receive_request()`. `complete` says whether the whole request, payload
    /// included, was read.
    fn co_receive_request(
        &mut self,
        req: &mut Request,
        complete: &mut bool,
        data: &mut Vec<u8>,
    ) -> std::result::Result<(), Fail> {
        self.receive_request(req)?;
        let mut check_length = false;
        let mut check_rofs = false;
        let mut allocate_buffer = false;
        let mut payload_okay = false;
        let mut payload_len = 0u64;
        let mut valid_flags = NBD_CMD_FLAG_FUA;
        let extended_with_payload =
            self.mode >= NbdMode::Extended && req.flags & NBD_CMD_FLAG_PAYLOAD_LEN != 0;
        if extended_with_payload {
            payload_len = req.len;
            check_length = true;
        }
        match req.typ {
            NBD_CMD_DISC => {
                *complete = true;
                return Err(Fail { errno: libc::EIO, err: None });
            }
            NBD_CMD_READ => {
                if self.mode >= NbdMode::Structured {
                    valid_flags |= NBD_CMD_FLAG_DF;
                }
                check_length = true;
                allocate_buffer = true;
            }
            NBD_CMD_WRITE => {
                if self.mode >= NbdMode::Extended {
                    valid_flags |= NBD_CMD_FLAG_PAYLOAD_LEN;
                }
                payload_okay = true;
                payload_len = req.len;
                check_length = true;
                allocate_buffer = true;
                check_rofs = true;
            }
            NBD_CMD_FLUSH => {}
            NBD_CMD_TRIM => check_rofs = true,
            NBD_CMD_CACHE => check_length = true,
            NBD_CMD_WRITE_ZEROES => {
                valid_flags |= NBD_CMD_FLAG_NO_HOLE | NBD_CMD_FLAG_FAST_ZERO;
                check_rofs = true;
            }
            NBD_CMD_BLOCK_STATUS => {
                if extended_with_payload {
                    self.block_status_payload_read(req)?;
                    check_length = false;
                    payload_len = 0;
                    valid_flags |= NBD_CMD_FLAG_PAYLOAD_LEN;
                }
                valid_flags |= NBD_CMD_FLAG_REQ_ONE;
            }
            _ => {}
        }

        if payload_len == 0 {
            *complete = true;
        }
        if check_length && req.len > u64::from(NBD_MAX_BUFFER_SIZE) {
            return Err(Fail::new(
                libc::EINVAL,
                format!("len ({}) is larger than max len ({NBD_MAX_BUFFER_SIZE})", req.len),
            ));
        }
        if payload_len != 0 && !payload_okay {
            // A payload on a command that takes none: skip it, the flag check below fails
            // the command.
            req.len = 0;
        }
        if allocate_buffer {
            *data = vec![0u8; req.len as usize];
        }
        if payload_len != 0 {
            if payload_okay {
                nbd_read(&mut self.ioc, data, Some("CMD_WRITE data")).map_err(Fail::eio)?;
            } else {
                nbd_drop(&mut self.ioc, payload_len).map_err(Fail::eio)?;
            }
            *complete = true;
        }

        let exp = self.exp().clone();
        if exp.nbdflags & NBD_FLAG_READ_ONLY != 0 && check_rofs {
            return Err(Fail::new(libc::EROFS, "Export is read-only"));
        }
        if req.from > exp.size || req.len > exp.size - req.from {
            let errno = if req.typ == NBD_CMD_WRITE || req.typ == NBD_CMD_WRITE_ZEROES {
                libc::ENOSPC
            } else {
                libc::EINVAL
            };
            return Err(Fail::new(
                errno,
                format!(
                    "operation past EOF; From: {}, Len: {}, Size: {}",
                    req.from, req.len, exp.size
                ),
            ));
        }
        if req.flags & !valid_flags != 0 {
            return Err(Fail::new(
                libc::EINVAL,
                format!(
                    "unsupported flags for command {} (got 0x{:x})",
                    nbd_cmd_lookup(req.typ),
                    req.flags
                ),
            ));
        }
        Ok(())
    }

    /// `nbd_co_send_iov()`.
    fn send_iov(&mut self, parts: &[&[u8]]) -> Result<()> {
        let total: usize = parts.iter().map(|p| p.len()).sum();
        if total <= 65536 {
            let mut v = Vec::with_capacity(total);
            for p in parts {
                v.extend_from_slice(p);
            }
            nbd_write(&mut self.ioc, &v)
        } else {
            for p in parts {
                nbd_write(&mut self.ioc, p)?;
            }
            Ok(())
        }
    }

    /// `nbd_co_send_simple_reply()`. `error` is a positive errno.
    fn send_simple_reply(&mut self, req: &Request, error: i32, data: &[u8]) -> Result<()> {
        let nbd_err = system_errno_to_nbd_errno(error);
        let mut h = Vec::with_capacity(16);
        h.extend_from_slice(&NBD_SIMPLE_REPLY_MAGIC.to_be_bytes());
        h.extend_from_slice(&nbd_err.to_be_bytes());
        h.extend_from_slice(&req.cookie.to_be_bytes());
        self.send_iov(&[&h, data])
    }

    /// `set_be_chunk()`.
    fn chunk_header(&self, req: &Request, flags: u16, typ: u16, length: usize) -> Vec<u8> {
        let mut h = Vec::with_capacity(32);
        if self.mode >= NbdMode::Extended {
            h.extend_from_slice(&NBD_EXTENDED_REPLY_MAGIC.to_be_bytes());
            h.extend_from_slice(&flags.to_be_bytes());
            h.extend_from_slice(&typ.to_be_bytes());
            h.extend_from_slice(&req.cookie.to_be_bytes());
            h.extend_from_slice(&req.from.to_be_bytes());
            h.extend_from_slice(&(length as u64).to_be_bytes());
        } else {
            h.extend_from_slice(&NBD_STRUCTURED_REPLY_MAGIC.to_be_bytes());
            h.extend_from_slice(&flags.to_be_bytes());
            h.extend_from_slice(&typ.to_be_bytes());
            h.extend_from_slice(&req.cookie.to_be_bytes());
            h.extend_from_slice(&(length as u32).to_be_bytes());
        }
        h
    }

    fn send_chunk_done(&mut self, req: &Request) -> Result<()> {
        let h = self.chunk_header(req, NBD_REPLY_FLAG_DONE, NBD_REPLY_TYPE_NONE, 0);
        self.send_iov(&[&h])
    }

    fn send_chunk_read(
        &mut self,
        req: &Request,
        offset: u64,
        data: &[u8],
        last: bool,
    ) -> Result<()> {
        let flags = if last { NBD_REPLY_FLAG_DONE } else { 0 };
        let h = self.chunk_header(req, flags, NBD_REPLY_TYPE_OFFSET_DATA, 8 + data.len());
        self.send_iov(&[&h, &offset.to_be_bytes(), data])
    }

    /// `nbd_co_send_chunk_error()`. `error` is a positive errno.
    fn send_chunk_error(&mut self, req: &Request, error: i32, msg: &str) -> Result<()> {
        let nbd_err = system_errno_to_nbd_errno(error);
        assert!(nbd_err != 0);
        let h = self.chunk_header(req, NBD_REPLY_FLAG_DONE, NBD_REPLY_TYPE_ERROR, 6 + msg.len());
        let mut p = Vec::with_capacity(6);
        p.extend_from_slice(&nbd_err.to_be_bytes());
        p.extend_from_slice(&(msg.len() as u16).to_be_bytes());
        self.send_iov(&[&h, &p, msg.as_bytes()])
    }

    /// `nbd_send_generic_reply()`. `ret` is 0 or a negative errno.
    fn send_generic_reply(&mut self, req: &Request, ret: i32, msg: &str) -> Result<()> {
        if self.mode >= NbdMode::Structured && ret < 0 {
            self.send_chunk_error(req, -ret, msg)
        } else if self.mode >= NbdMode::Extended {
            self.send_chunk_done(req)
        } else {
            self.send_simple_reply(req, if ret < 0 { -ret } else { 0 }, &[])
        }
    }

    fn io_ret(r: io::Result<()>) -> i32 {
        match r {
            Ok(()) => 0,
            Err(e) => -io_errno(&e),
        }
    }

    /// `nbd_co_send_sparse_read()`.
    fn send_sparse_read(&mut self, req: &Request, data: &mut [u8]) -> Result<()> {
        let exp = self.exp().clone();
        let offset = req.from;
        let size = data.len() as u64;
        let mut progress = 0u64;
        while progress < size {
            let st = match exp.status.block_status(offset + progress, size - progress) {
                Ok(st) => st,
                Err(e) => {
                    let msg =
                        format!("unable to check for holes: {}", ruvm_base::error::strerror(&e));
                    return self.send_chunk_error(req, io_errno(&e), &msg);
                }
            };
            let pnum = st.bytes.clamp(1, size - progress);
            let last = progress + pnum == size;
            if st.zero {
                let flags = if last { NBD_REPLY_FLAG_DONE } else { 0 };
                let h = self.chunk_header(req, flags, NBD_REPLY_TYPE_OFFSET_HOLE, 12);
                let mut p = Vec::with_capacity(12);
                p.extend_from_slice(&(offset + progress).to_be_bytes());
                p.extend_from_slice(&(pnum as u32).to_be_bytes());
                self.send_iov(&[&h, &p])?;
            } else {
                let buf = &mut data[progress as usize..(progress + pnum) as usize];
                if let Err(e) = exp.blk.pread(offset + progress, buf) {
                    return Err(Error::from_io("reading from file failed", e));
                }
                let buf = &data[progress as usize..(progress + pnum) as usize];
                self.send_chunk_read(req, offset + progress, buf, last)?;
            }
            progress += pnum;
        }
        Ok(())
    }

    /// `nbd_do_cmd_read()`.
    fn do_cmd_read(&mut self, req: &Request, data: &mut [u8]) -> Result<()> {
        let exp = self.exp().clone();
        if req.flags & NBD_CMD_FLAG_FUA != 0 {
            let r = Self::io_ret(exp.blk.flush());
            if r < 0 {
                return self.send_generic_reply(req, r, "flush failed");
            }
        }
        if self.mode >= NbdMode::Structured && req.flags & NBD_CMD_FLAG_DF == 0 && req.len > 0 {
            return self.send_sparse_read(req, data);
        }
        let r = Self::io_ret(exp.blk.pread(req.from, data));
        if r < 0 {
            return self.send_generic_reply(req, r, "reading from file failed");
        }
        if self.mode >= NbdMode::Structured {
            if req.len > 0 {
                self.send_chunk_read(req, req.from, data, true)
            } else {
                self.send_chunk_done(req)
            }
        } else {
            self.send_simple_reply(req, 0, data)
        }
    }

    /// `nbd_co_send_extents()`.
    fn send_extents(&mut self, req: &Request, ea: &ExtentArray, last: bool, id: u32) -> Result<()> {
        let flags = if last { NBD_REPLY_FLAG_DONE } else { 0 };
        let mut p = Vec::new();
        p.extend_from_slice(&id.to_be_bytes());
        let typ = if self.mode >= NbdMode::Extended {
            p.extend_from_slice(&(ea.extents.len() as u32).to_be_bytes());
            for &(len, fl) in &ea.extents {
                p.extend_from_slice(&len.to_be_bytes());
                p.extend_from_slice(&fl.to_be_bytes());
            }
            NBD_REPLY_TYPE_BLOCK_STATUS_EXT
        } else {
            for &(len, fl) in &ea.extents {
                p.extend_from_slice(&(len as u32).to_be_bytes());
                p.extend_from_slice(&(fl as u32).to_be_bytes());
            }
            NBD_REPLY_TYPE_BLOCK_STATUS
        };
        let h = self.chunk_header(req, flags, typ, p.len());
        self.send_iov(&[&h, &p])
    }

    /// `nbd_co_send_block_status()`.
    fn send_block_status(&mut self, req: &Request, df: bool, last: bool, id: u32) -> Result<()> {
        let exp = self.exp().clone();
        let mut ea = ExtentArray::new(df, self.mode);
        let mut offset = req.from;
        let mut bytes = req.len;
        let mut err = None;
        while bytes > 0 {
            let r = if id == NBD_META_ID_BASE_ALLOCATION {
                // blockstatus_to_extents().
                exp.status.block_status(offset, bytes).map(|s| {
                    let fl = (if s.data { 0 } else { NBD_STATE_HOLE })
                        | (if s.zero { NBD_STATE_ZERO } else { 0 });
                    (s.bytes, fl)
                })
            } else {
                // blockalloc_to_extents().
                exp.status.allocation_depth(offset, bytes).map(|(d, n)| (n, u64::from(d)))
            };
            match r {
                Ok((num, fl)) => {
                    let num = num.clamp(1, bytes);
                    if !ea.add(num, fl) {
                        break;
                    }
                    offset += num;
                    bytes -= num;
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = err {
            return self.send_chunk_error(req, io_errno(&e), "can't get block status");
        }
        self.send_extents(req, &ea, last, id)
    }

    /// `nbd_co_send_bitmap()` with `bitmap_to_extents()`.
    fn send_bitmap(&mut self, req: &Request, idx: usize, df: bool, last: bool) -> Result<()> {
        let exp = self.exp().clone();
        let bm = &exp.bitmaps[idx];
        let mut ea = ExtentArray::new(df, self.mode);
        let end = req.from + req.len;
        let bound = if ea.extended { i64::MAX as u64 } else { i32::MAX as u64 };
        let mut start = req.from;
        let mut full = false;
        while let Some((dirty_start, dirty_count)) = bm.next_dirty_area(start, end, bound) {
            if !ea.add(dirty_start - start, 0) || !ea.add(dirty_count, NBD_STATE_DIRTY) {
                full = true;
                break;
            }
            start = dirty_start + dirty_count;
        }
        if !full {
            ea.add(end - start, 0);
        }
        self.send_extents(req, &ea, last, NBD_META_ID_DIRTY_BITMAP + idx as u32)
    }

    /// `nbd_handle_request()`.
    fn handle_request(&mut self, req: &Request, data: &mut [u8]) -> Result<()> {
        let exp = self.exp().clone();
        let blk = &exp.blk;
        match req.typ {
            NBD_CMD_CACHE => {
                // A prefetch without copy-on-read reads nothing.
                self.send_generic_reply(req, 0, "caching data failed")
            }
            NBD_CMD_READ => self.do_cmd_read(req, data),
            NBD_CMD_WRITE => {
                let mut r = Self::io_ret(blk.pwrite(req.from, data));
                if r == 0 && req.flags & NBD_CMD_FLAG_FUA != 0 {
                    r = Self::io_ret(blk.flush());
                }
                self.send_generic_reply(req, r, "writing to file failed")
            }
            NBD_CMD_WRITE_ZEROES => {
                let may_unmap = req.flags & NBD_CMD_FLAG_NO_HOLE == 0;
                let mut r = Self::io_ret(blk.pwrite_zeroes(req.from, req.len, may_unmap));
                if r == 0 && req.flags & NBD_CMD_FLAG_FUA != 0 {
                    r = Self::io_ret(blk.flush());
                }
                self.send_generic_reply(req, r, "writing to file failed")
            }
            NBD_CMD_FLUSH => {
                let r = Self::io_ret(blk.flush());
                self.send_generic_reply(req, r, "flush failed")
            }
            NBD_CMD_TRIM => {
                let mut r = Self::io_ret(blk.pdiscard(req.from, req.len));
                if r >= 0 && req.flags & NBD_CMD_FLAG_FUA != 0 {
                    r = Self::io_ret(blk.flush());
                }
                self.send_generic_reply(req, r, "discard failed")
            }
            NBD_CMD_BLOCK_STATUS => {
                let ctx = req.contexts.clone().unwrap_or_else(|| self.contexts.clone());
                if ctx.count > 0 {
                    let df = req.flags & NBD_CMD_FLAG_REQ_ONE != 0;
                    let mut remaining = ctx.count;
                    if req.len == 0 {
                        return self.send_generic_reply(req, -libc::EINVAL, "need non-zero length");
                    }
                    if ctx.base_allocation {
                        remaining -= 1;
                        self.send_block_status(
                            req,
                            df,
                            remaining == 0,
                            NBD_META_ID_BASE_ALLOCATION,
                        )?;
                    }
                    if ctx.allocation_depth {
                        remaining -= 1;
                        self.send_block_status(
                            req,
                            df,
                            remaining == 0,
                            NBD_META_ID_ALLOCATION_DEPTH,
                        )?;
                    }
                    for i in 0..exp.bitmaps.len() {
                        if !ctx.bitmaps.get(i).copied().unwrap_or(false) {
                            continue;
                        }
                        remaining -= 1;
                        self.send_bitmap(req, i, df, remaining == 0)?;
                    }
                    assert_eq!(remaining, 0);
                    Ok(())
                } else if self.contexts.count > 0 {
                    self.send_generic_reply(
                        req,
                        -libc::EINVAL,
                        "CMD_BLOCK_STATUS payload not valid",
                    )
                } else {
                    self.send_generic_reply(req, -libc::EINVAL, "CMD_BLOCK_STATUS not negotiated")
                }
            }
            t => {
                let m = format!("invalid request type ({t}) received");
                self.send_generic_reply(req, -libc::EINVAL, &m)
            }
        }
    }

    /// The loop of `nbd_trip()` calls.
    fn transmission(&mut self) {
        loop {
            let mut req = Request::default();
            let mut complete = false;
            let mut data = Vec::new();
            let r = match self.co_receive_request(&mut req, &mut complete, &mut data) {
                Err(Fail { errno, err }) if errno == libc::EIO => {
                    if let Some(e) = err {
                        report(&e.prepend("Disconnect client, due to: "));
                    }
                    return;
                }
                Err(Fail { err, .. }) => {
                    let msg = err.map(|e| e.message().to_string()).unwrap_or_default();
                    self.send_generic_reply(&req, -libc::EINVAL, &msg)
                }
                Ok(()) => self.handle_request(&req, &mut data),
            };
            if let Err(e) = r {
                report(&e.prepend("Failed to send reply: ").prepend("Disconnect client, due to: "));
                return;
            }
            if !complete {
                error_report(
                    "Disconnect client, due to: Request handling failed in intermediate state",
                );
                return;
            }
        }
    }
}

fn report(e: &Error) {
    ruvm_base::report::report_error(e);
}

/// `NBDExtentArray`.
#[derive(Debug)]
struct ExtentArray {
    extents: Vec<(u64, u64)>,
    nb_alloc: usize,
    extended: bool,
    can_add: bool,
}

impl ExtentArray {
    fn new(dont_fragment: bool, mode: NbdMode) -> ExtentArray {
        ExtentArray {
            extents: Vec::new(),
            nb_alloc: if dont_fragment { 1 } else { NBD_MAX_BLOCK_STATUS_EXTENTS },
            extended: mode >= NbdMode::Extended,
            can_add: true,
        }
    }

    /// `nbd_extent_array_add()`: `false` once the array is full.
    fn add(&mut self, length: u64, flags: u64) -> bool {
        assert!(self.can_add);
        if length == 0 {
            return true;
        }
        if let Some(last) = self.extents.last_mut() {
            if last.1 == flags {
                let sum = last.0 + length;
                if sum <= u64::from(u32::MAX) || self.extended {
                    last.0 = sum;
                    return true;
                }
            }
        }
        if self.extents.len() >= self.nb_alloc {
            self.can_add = false;
            return false;
        }
        self.extents.push((length, flags));
        true
    }
}

static SERVER: Mutex<Option<NbdServer>> = Mutex::new(None);

fn global() -> MutexGuard<'static, Option<NbdServer>> {
    SERVER.lock().unwrap_or_else(|e| e.into_inner())
}

/// `nbd_server_is_running()`.
pub fn nbd_server_is_running() -> bool {
    global().is_some()
}

/// `nbd-server-start`: start the process wide server.
pub fn nbd_server_start(opts: &NbdServerOptions) -> Result<()> {
    let mut g = global();
    if g.is_some() {
        return Err(Error::generic("NBD server already running"));
    }
    *g = Some(NbdServer::start(opts)?);
    Ok(())
}

/// The process wide server, if one is running.
pub fn nbd_server() -> Option<NbdServer> {
    global().clone()
}

/// `nbd-server-stop`.
pub fn nbd_server_stop() -> Result<()> {
    let s = global().take();
    let Some(s) = s else {
        return Err(Error::generic("NBD server not running"));
    };
    s.stop();
    Ok(())
}

/// `nbd-server-add`.
pub fn nbd_server_add(graph: &BlockGraph, arg: &NbdServerAddOptions) -> Result<()> {
    // qmp_nbd_server_add() looks the device up before anything else, and
    // nbd_export_create() only then notices that no server runs.
    if resolve_device(graph, &arg.device).is_none() {
        let d = &arg.device;
        return Err(Error::generic(format!("Cannot find device='{d}' nor node-name='{d}'")));
    }
    let Some(s) = nbd_server() else {
        return Err(Error::generic("NBD server not running"));
    };
    s.nbd_server_add(graph, arg)
}

/// `block-export-add` for an NBD export on the process wide server.
pub fn block_export_add(graph: &BlockGraph, opts: &BlockExportOptions) -> Result<()> {
    let Some(s) = nbd_server() else {
        return Err(Error::generic("NBD server not running"));
    };
    s.export_add(graph, opts)
}

/// `nbd-server-remove`.
pub fn nbd_server_remove(name: &str, mode: Option<BlockExportRemoveMode>) -> Result<()> {
    let Some(s) = nbd_server() else {
        return Err(Error::generic(format!("Export '{name}' is not found")));
    };
    s.export_remove(name, mode)
}
