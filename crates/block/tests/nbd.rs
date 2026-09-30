// SPDX-License-Identifier: GPL-2.0-or-later

//! NBD end to end: byte traces of the client handshake against scripted servers, byte traces
//! of the server's answers (compared with qemu-nbd's when it is installed), the ruvm client
//! against the ruvm server over Unix sockets and TCP, the `nbd` block driver, the ruvm client
//! against qemu-nbd, and qemu-img against the ruvm server.
//!
//! The tests that need QEMU's tools look for them in `$QEMU_BIN_DIR`, /opt/homebrew/bin,
//! /usr/local/bin and /usr/bin, and skip with a message when they are not there.

#![cfg(unix)]

use std::fs;
use std::io::{self, Cursor, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ruvm_block::nbd::{
    NBD_FLAG_READ_ONLY, NBD_FLAG_SEND_FAST_ZERO, NBD_FLAG_SEND_WRITE_ZEROES, NbdClient,
    NbdExportInfo, NbdMode, NbdServer, nbd_receive_export_list, nbd_receive_negotiate,
};
use ruvm_block::{BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph};
use ruvm_qapi::json;
use ruvm_qapi::types::{
    BlockExportOptions, BlockExportOptionsNbd, BlockExportOptionsU, BlockdevOptions,
    BlockdevOptionsNbd, InetSocketAddress, SocketAddress, SocketAddressU, UnixSocketAddress,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

// The protocol numbers, written out here so the traces do not depend on the code under test.
const NBDMAGIC: &[u8] = b"NBDMAGIC";
const IHAVEOPT: &[u8] = b"IHAVEOPT";
const OLDSTYLE_MAGIC: u64 = 0x0000_4202_8186_1253;
const REP_MAGIC: u64 = 0x0003_e889_0455_65a9;
const OPT_EXPORT_NAME: u32 = 1;
const OPT_ABORT: u32 = 2;
const OPT_LIST: u32 = 3;
const OPT_STARTTLS: u32 = 5;
const OPT_INFO: u32 = 6;
const OPT_GO: u32 = 7;
const OPT_STRUCTURED_REPLY: u32 = 8;
const OPT_LIST_META_CONTEXT: u32 = 9;
const OPT_SET_META_CONTEXT: u32 = 10;
const OPT_EXTENDED_HEADERS: u32 = 11;
const REP_ACK: u32 = 1;
const REP_SERVER: u32 = 2;
const REP_INFO: u32 = 3;
const REP_META_CONTEXT: u32 = 4;
const REP_ERR_UNSUP: u32 = 0x8000_0001;
const REP_ERR_UNKNOWN: u32 = 0x8000_0006;

// ---------------------------------------------------------------------------------------------
// Helpers

/// A fresh directory for one test in the system temporary directory, where socket paths stay
/// short enough.
fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ruvm-nbd-{test}"));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// An image of `len` bytes where byte `i` is `i % 251`, with a zeroed middle.
fn image(dir: &Path, name: &str, len: usize) -> String {
    let path = dir.join(name);
    let mut data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let q = len / 4;
    data[q..2 * q].fill(0);
    fs::write(&path, data).unwrap();
    path.to_str().unwrap().to_string()
}

fn opts<T: Default + Visit>(s: &str) -> T {
    let mut v = QObjectInputVisitor::new(json::from_str(s).unwrap());
    let mut o = T::default();
    T::visit(&mut v, None, &mut o).unwrap();
    o
}

/// A graph with a raw node `name` over the file `path`.
fn graph_with(name: &str, path: &str, read_only: bool) -> BlockGraph {
    let g = BlockGraph::new();
    let s = format!(
        r#"{{"driver": "raw", "node-name": "{name}", "read-only": {read_only},
            "file": {{"driver": "file", "filename": "{path}", "read-only": {read_only}}}}}"#
    );
    g.blockdev_add(opts::<BlockdevOptions>(&s)).unwrap();
    g
}

// Linux has more members in the socket addresses.
#[allow(clippy::needless_update)]
fn unix_addr(path: &Path) -> SocketAddress {
    SocketAddress {
        u: SocketAddressU::Unix(UnixSocketAddress {
            path: path.to_str().unwrap().to_string(),
            ..UnixSocketAddress::default()
        }),
    }
}

fn tcp_addr(port: &str) -> SocketAddress {
    SocketAddress {
        u: SocketAddressU::Inet(InetSocketAddress {
            host: "127.0.0.1".into(),
            port: port.into(),
            ..InetSocketAddress::default()
        }),
    }
}

fn export(
    id: &str,
    node: &str,
    writable: bool,
    desc: Option<&str>,
    depth: bool,
) -> BlockExportOptions {
    BlockExportOptions {
        id: id.into(),
        node_name: node.into(),
        writable: Some(writable),
        u: BlockExportOptionsU::Nbd(BlockExportOptionsNbd {
            name: Some(id.into()),
            description: desc.map(str::to_string),
            bitmaps: None,
            allocation_depth: Some(depth),
        }),
        ..BlockExportOptions::default()
    }
}

fn client_opts(server: SocketAddress, export: &str) -> BlockdevOptionsNbd {
    BlockdevOptionsNbd { server, export: Some(export.into()), ..BlockdevOptionsNbd::default() }
}

fn open_client(o: &BlockdevOptionsNbd) -> ruvm_base::Result<NbdClient> {
    let mut ro = false;
    NbdClient::open(o, None, &mut ro, true)
}

/// A channel that reads a scripted server and records what the client writes.
struct Script {
    input: Cursor<Vec<u8>>,
    output: Vec<u8>,
}

impl Script {
    fn new(server: Vec<u8>) -> Script {
        Script { input: Cursor::new(server), output: Vec::new() }
    }
}

impl Read for Script {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.input.read(buf)
    }
}

impl Write for Script {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.output.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Byte builder.
#[derive(Default)]
struct B(Vec<u8>);

impl B {
    fn raw(mut self, b: &[u8]) -> B {
        self.0.extend_from_slice(b);
        self
    }
    fn u16(self, v: u16) -> B {
        self.raw(&v.to_be_bytes())
    }
    fn u32(self, v: u32) -> B {
        self.raw(&v.to_be_bytes())
    }
    fn u64(self, v: u64) -> B {
        self.raw(&v.to_be_bytes())
    }
    fn str32(self, s: &str) -> B {
        self.u32(s.len() as u32).raw(s.as_bytes())
    }
    /// A client option request.
    fn opt(self, opt: u32, data: &[u8]) -> B {
        self.raw(IHAVEOPT).u32(opt).u32(data.len() as u32).raw(data)
    }
    /// A server option reply.
    fn rep(self, opt: u32, typ: u32, data: &[u8]) -> B {
        self.u64(REP_MAGIC).u32(opt).u32(typ).u32(data.len() as u32).raw(data)
    }
    fn greeting(self, flags: u16) -> B {
        self.raw(NBDMAGIC).raw(IHAVEOPT).u16(flags)
    }
    fn done(self) -> Vec<u8> {
        self.0
    }
}

fn info_export(size: u64, flags: u16) -> Vec<u8> {
    B::default().u16(0).u64(size).u16(flags).done()
}

fn info_block_size(min: u32, pref: u32, max: u32) -> Vec<u8> {
    B::default().u16(3).u32(min).u32(pref).u32(max).done()
}

/// The payload of `NBD_OPT_GO` or `NBD_OPT_INFO` asking for block sizes.
fn go_payload(name: &str) -> Vec<u8> {
    B::default().str32(name).u16(1).u16(3).done()
}

fn meta_query(name: &str, queries: &[&str]) -> Vec<u8> {
    let mut b = B::default().str32(name).u32(queries.len() as u32);
    for q in queries {
        b = b.str32(q);
    }
    b.done()
}

fn meta_reply(id: u32, name: &str) -> Vec<u8> {
    B::default().u32(id).raw(name.as_bytes()).done()
}

fn want(name: &str) -> NbdExportInfo {
    NbdExportInfo {
        request_sizes: true,
        mode: NbdMode::Extended,
        base_allocation: true,
        name: name.into(),
        ..NbdExportInfo::default()
    }
}

// ---------------------------------------------------------------------------------------------
// Client handshake traces

/// Extended headers, a meta context and `NBD_OPT_GO`: everything the driver asks for.
#[test]
fn trace_extended_go() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ACK, &[])
        .rep(OPT_SET_META_CONTEXT, REP_META_CONTEXT, &meta_reply(7, "base:allocation"))
        .rep(OPT_SET_META_CONTEXT, REP_ACK, &[])
        .rep(OPT_GO, REP_INFO, &info_export(1 << 20, 0x0d))
        .rep(OPT_GO, REP_INFO, &info_block_size(1, 4096, 1 << 25))
        .rep(OPT_GO, REP_ACK, &[])
        .done();
    let mut s = Script::new(server);
    let mut info = want("disk");
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    let expect = B::default()
        .u32(3)
        .opt(OPT_EXTENDED_HEADERS, &[])
        .opt(OPT_SET_META_CONTEXT, &meta_query("disk", &["base:allocation"]))
        .opt(OPT_GO, &go_payload("disk"))
        .done();
    assert_eq!(s.output, expect);
    assert_eq!(info.mode, NbdMode::Extended);
    assert!(info.base_allocation);
    assert_eq!(info.context_id, 7);
    assert_eq!((info.size, info.flags), (1 << 20, 0x0d));
    assert_eq!((info.min_block, info.opt_block, info.max_block), (1, 4096, 1 << 25));
}

/// A server without extended headers: structured replies instead, and an `x-dirty-bitmap`
/// context.
#[test]
fn trace_structured_fallback() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ERR_UNSUP, &[])
        .rep(OPT_STRUCTURED_REPLY, REP_ACK, &[])
        .rep(OPT_SET_META_CONTEXT, REP_META_CONTEXT, &meta_reply(2, "qemu:dirty-bitmap:b0"))
        .rep(OPT_SET_META_CONTEXT, REP_ACK, &[])
        .rep(OPT_GO, REP_INFO, &info_export(4096, 0x03))
        .rep(OPT_GO, REP_ACK, &[])
        .done();
    let mut s = Script::new(server);
    let mut info = want("");
    info.x_dirty_bitmap = Some("qemu:dirty-bitmap:b0".into());
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    let expect = B::default()
        .u32(3)
        .opt(OPT_EXTENDED_HEADERS, &[])
        .opt(OPT_STRUCTURED_REPLY, &[])
        .opt(OPT_SET_META_CONTEXT, &meta_query("", &["qemu:dirty-bitmap:b0"]))
        .opt(OPT_GO, &go_payload(""))
        .done();
    assert_eq!(s.output, expect);
    assert_eq!(info.mode, NbdMode::Structured);
    assert_eq!(info.context_id, 2);
    assert!(info.base_allocation);
}

/// Neither extended headers nor structured replies: simple replies and no meta context.
#[test]
fn trace_simple() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ERR_UNSUP, &[])
        .rep(OPT_STRUCTURED_REPLY, REP_ERR_UNSUP, &[])
        .rep(OPT_GO, REP_INFO, &info_export(512, 0x01))
        .rep(OPT_GO, REP_ACK, &[])
        .done();
    let mut s = Script::new(server);
    let mut info = want("x");
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    let expect = B::default()
        .u32(3)
        .opt(OPT_EXTENDED_HEADERS, &[])
        .opt(OPT_STRUCTURED_REPLY, &[])
        .opt(OPT_GO, &go_payload("x"))
        .done();
    assert_eq!(s.output, expect);
    assert_eq!(info.mode, NbdMode::Simple);
    assert!(!info.base_allocation);
}

/// A server without `NBD_OPT_GO`: the client checks the export list and falls back to
/// `NBD_OPT_EXPORT_NAME`.
#[test]
fn trace_export_name_fallback() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ERR_UNSUP, &[])
        .rep(OPT_STRUCTURED_REPLY, REP_ERR_UNSUP, &[])
        .rep(OPT_GO, REP_ERR_UNSUP, &[])
        .rep(OPT_LIST, REP_SERVER, &B::default().str32("other").done())
        .rep(OPT_LIST, REP_SERVER, &B::default().str32("x").raw(b"desc").done())
        .rep(OPT_LIST, REP_ACK, &[])
        .u64(8192)
        .u16(0x03)
        .done();
    let mut s = Script::new(server);
    let mut info = want("x");
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    let expect = B::default()
        .u32(3)
        .opt(OPT_EXTENDED_HEADERS, &[])
        .opt(OPT_STRUCTURED_REPLY, &[])
        .opt(OPT_GO, &go_payload("x"))
        .opt(OPT_LIST, &[])
        .opt(OPT_EXPORT_NAME, b"x")
        .done();
    assert_eq!(s.output, expect);
    assert_eq!((info.size, info.flags), (8192, 0x03));

    // The export is not in the list.
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ERR_UNSUP, &[])
        .rep(OPT_STRUCTURED_REPLY, REP_ERR_UNSUP, &[])
        .rep(OPT_GO, REP_ERR_UNSUP, &[])
        .rep(OPT_LIST, REP_SERVER, &B::default().str32("other").done())
        .rep(OPT_LIST, REP_ACK, &[])
        .done();
    let mut s = Script::new(server);
    let e = nbd_receive_negotiate(&mut s, &mut want("x")).unwrap_err();
    assert_eq!(e.message(), "No export with name 'x' available");
    assert!(s.output.ends_with(&B::default().opt(OPT_ABORT, &[]).done()));
}

/// Newstyle without the fixed flag: straight to `NBD_OPT_EXPORT_NAME`, with 124 zero bytes
/// after the export flags.
#[test]
fn trace_unfixed_newstyle() {
    let server = B::default().greeting(0).u64(1024).u16(0x01).raw(&[0; 124]).done();
    let mut s = Script::new(server);
    let mut info = want("e");
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    assert_eq!(s.output, B::default().u32(0).opt(OPT_EXPORT_NAME, b"e").done());
    assert_eq!(info.mode, NbdMode::ExportName);
    assert_eq!(info.size, 1024);
}

/// Oldstyle: the server talks, the client only listens.
#[test]
fn trace_oldstyle() {
    let server =
        || B::default().raw(NBDMAGIC).u64(OLDSTYLE_MAGIC).u64(2048).u32(0x03).raw(&[0; 124]).done();
    let mut s = Script::new(server());
    let mut info = want("");
    nbd_receive_negotiate(&mut s, &mut info).unwrap();
    assert!(s.output.is_empty());
    assert_eq!(info.mode, NbdMode::Oldstyle);
    assert_eq!((info.size, info.flags), (2048, 0x03));

    let e = nbd_receive_negotiate(&mut Script::new(server()), &mut want("n")).unwrap_err();
    assert_eq!(e.message(), "Server does not support non-empty export names");

    let bad = B::default().raw(NBDMAGIC).u64(OLDSTYLE_MAGIC).u64(1).u32(0x10000).done();
    let e = nbd_receive_negotiate(&mut Script::new(bad), &mut want("")).unwrap_err();
    assert_eq!(e.message(), "Unexpected export flags 10000x");
}

/// `NBD_OPT_LIST`, `NBD_OPT_INFO` and `NBD_OPT_LIST_META_CONTEXT`, as `qemu-nbd --list` asks.
#[test]
fn trace_export_list() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ACK, &[])
        .rep(OPT_LIST, REP_SERVER, &B::default().str32("a").raw(b"first").done())
        .rep(OPT_LIST, REP_ACK, &[])
        .rep(OPT_INFO, REP_INFO, &info_export(4096, 0x01))
        .rep(OPT_INFO, REP_ACK, &[])
        .rep(OPT_LIST_META_CONTEXT, REP_META_CONTEXT, &meta_reply(0, "base:allocation"))
        .rep(OPT_LIST_META_CONTEXT, REP_ACK, &[])
        // qemu 3.0 did not list "qemu:" contexts for an empty query, so the client asks again.
        .rep(OPT_LIST_META_CONTEXT, REP_META_CONTEXT, &meta_reply(0, "qemu:allocation-depth"))
        .rep(OPT_LIST_META_CONTEXT, REP_ACK, &[])
        .done();
    let mut s = Script::new(server);
    let list = nbd_receive_export_list(&mut s).unwrap();
    let expect = B::default()
        .u32(3)
        .opt(OPT_EXTENDED_HEADERS, &[])
        .opt(OPT_LIST, &[])
        .opt(OPT_INFO, &go_payload("a"))
        .opt(OPT_LIST_META_CONTEXT, &meta_query("a", &[]))
        .opt(OPT_LIST_META_CONTEXT, &meta_query("a", &["qemu:"]))
        .opt(OPT_ABORT, &[])
        .done();
    assert_eq!(s.output, expect);
    assert_eq!(list.len(), 1);
    assert_eq!(list[0].name, "a");
    assert_eq!(list[0].description.as_deref(), Some("first"));
    assert_eq!(list[0].contexts, ["base:allocation", "qemu:allocation-depth"]);
}

/// Errors in the handshake, with QEMU's text and the `NBD_OPT_ABORT` that follows them.
#[test]
fn trace_errors() {
    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ACK, &[])
        .rep(OPT_SET_META_CONTEXT, REP_ACK, &[])
        .rep(OPT_GO, REP_ERR_UNKNOWN, b"no such export")
        .done();
    let mut s = Script::new(server);
    let e = nbd_receive_negotiate(&mut s, &mut want("nope")).unwrap_err();
    assert_eq!(e.message(), "Requested export not available");
    assert_eq!(e.hint_text(), Some("server reported: no such export\n"));
    assert!(s.output.ends_with(&B::default().opt(OPT_ABORT, &[]).done()));

    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ACK, &[])
        .rep(OPT_SET_META_CONTEXT, REP_META_CONTEXT, &meta_reply(1, "base:other"))
        .done();
    let e = nbd_receive_negotiate(&mut Script::new(server), &mut want("")).unwrap_err();
    assert_eq!(
        e.message(),
        "Failed to negotiate meta context 'base:allocation', server answered with different \
         context 'base:other'"
    );

    let e = nbd_receive_negotiate(
        &mut Script::new(b"NBDMAGIC\0\0\0\0\0\0\0\x01".to_vec()),
        &mut want(""),
    )
    .unwrap_err();
    assert_eq!(e.message(), "Bad server magic received: 0x1");

    let server = B::default()
        .greeting(3)
        .rep(OPT_EXTENDED_HEADERS, REP_ACK, &[])
        .rep(OPT_SET_META_CONTEXT, REP_ACK, &[])
        .rep(OPT_GO, REP_ACK, &[])
        .done();
    let e = nbd_receive_negotiate(&mut Script::new(server), &mut want("")).unwrap_err();
    assert_eq!(e.message(), "broken server omitted NBD_INFO_EXPORT");
}

// ---------------------------------------------------------------------------------------------
// Server traces

/// A ruvm server with one export `exp` of a 1 MiB image, on a Unix socket.
struct RuvmServer {
    _graph: BlockGraph,
    server: NbdServer,
    sock: PathBuf,
    img: String,
}

impl RuvmServer {
    fn start(dir: &Path, writable: bool, desc: Option<&str>, depth: bool) -> RuvmServer {
        let img = image(dir, "img.raw", 1 << 20);
        let graph = graph_with("n0", &img, !writable);
        let sock = dir.join("s");
        let server = NbdServer::start_addr(&unix_addr(&sock), 10, None, 0).unwrap();
        server.export_add(&graph, &export("exp", "n0", writable, desc, depth)).unwrap();
        RuvmServer { _graph: graph, server, sock, img }
    }
}

impl Drop for RuvmServer {
    fn drop(&mut self) {
        self.server.stop();
    }
}

/// Runs a raw option exchange: `send` after the client flags, reading the greeting first.
/// Returns everything the server sent after the greeting, until it closes or goes quiet.
fn exchange(sock: &Path, send: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut s = UnixStream::connect(sock).unwrap();
    s.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
    let mut greeting = [0u8; 18];
    s.read_exact(&mut greeting).unwrap();
    s.write_all(&3u32.to_be_bytes()).unwrap();
    s.write_all(send).unwrap();
    let mut out = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        match s.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => out.extend_from_slice(&buf[..n]),
            Err(_) => break,
        }
    }
    (greeting.to_vec(), out)
}

/// The option exchanges every server trace test runs. The last one ends the handshake.
fn server_scripts() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("starttls", B::default().opt(OPT_STARTTLS, &[]).opt(OPT_ABORT, &[]).done()),
        ("list", B::default().opt(OPT_LIST, &[]).opt(OPT_ABORT, &[]).done()),
        ("unknown", B::default().opt(99, b"xyz").opt(OPT_ABORT, &[]).done()),
        (
            "structured",
            B::default()
                .opt(OPT_STRUCTURED_REPLY, &[])
                .opt(OPT_STRUCTURED_REPLY, &[])
                .opt(OPT_EXTENDED_HEADERS, &[])
                .opt(OPT_ABORT, &[])
                .done(),
        ),
        (
            "extended",
            B::default()
                .opt(OPT_EXTENDED_HEADERS, &[])
                .opt(OPT_STRUCTURED_REPLY, &[])
                .opt(OPT_ABORT, &[])
                .done(),
        ),
        (
            "info",
            B::default()
                .opt(OPT_INFO, &go_payload("exp"))
                .opt(OPT_INFO, &B::default().str32("exp").u16(2).u16(1).u16(2).done())
                .opt(OPT_INFO, &go_payload("missing"))
                .opt(OPT_ABORT, &[])
                .done(),
        ),
        (
            "list-meta",
            B::default()
                .opt(OPT_STRUCTURED_REPLY, &[])
                .opt(OPT_LIST_META_CONTEXT, &meta_query("exp", &[]))
                .opt(OPT_LIST_META_CONTEXT, &meta_query("exp", &["qemu:"]))
                .opt(OPT_LIST_META_CONTEXT, &meta_query("exp", &["base:"]))
                .opt(OPT_ABORT, &[])
                .done(),
        ),
        (
            "set-meta",
            B::default()
                .opt(OPT_SET_META_CONTEXT, &meta_query("exp", &["base:allocation"]))
                .opt(OPT_STRUCTURED_REPLY, &[])
                .opt(
                    OPT_SET_META_CONTEXT,
                    &meta_query("exp", &["base:allocation", "qemu:allocation-depth", "x:y"]),
                )
                .opt(OPT_SET_META_CONTEXT, &meta_query("nope", &["base:allocation"]))
                .opt(OPT_ABORT, &[])
                .done(),
        ),
        ("go", B::default().opt(OPT_EXTENDED_HEADERS, &[]).opt(OPT_GO, &go_payload("exp")).done()),
        ("export-name", B::default().opt(OPT_EXPORT_NAME, b"exp").done()),
        ("transmission-structured", transmission(false)),
        ("transmission-extended", transmission(true)),
    ]
}

/// A request in the transmission phase, with compact or extended headers.
fn req(b: B, ext: bool, flags: u16, typ: u16, cookie: u64, from: u64, len: u64) -> B {
    if ext {
        b.u32(0x21e4_1c71).u16(flags).u16(typ).u64(cookie).u64(from).u64(len)
    } else {
        b.u32(0x2560_9513).u16(flags).u16(typ).u64(cookie).u64(from).u32(len as u32)
    }
}

/// Negotiates both block status contexts and runs the read-only commands, including some
/// that fail.
fn transmission(ext: bool) -> Vec<u8> {
    let mode = if ext { OPT_EXTENDED_HEADERS } else { OPT_STRUCTURED_REPLY };
    let mut b = B::default()
        .opt(mode, &[])
        .opt(
            OPT_SET_META_CONTEXT,
            &meta_query("exp", &["base:allocation", "qemu:allocation-depth"]),
        )
        .opt(OPT_GO, &go_payload("exp"));
    // NBD_CMD_READ, NBD_CMD_BLOCK_STATUS (all contexts, then with REQ_ONE), NBD_CMD_CACHE,
    // NBD_CMD_FLUSH, a read past the end, a read with DF, an unknown command.
    b = req(b, ext, 0, 0, 1, 100, 64);
    b = req(b, ext, 0, 7, 2, 0, 1 << 20);
    b = req(b, ext, 8, 7, 3, 4096, 1 << 19);
    b = req(b, ext, 0, 5, 4, 0, 65536);
    b = req(b, ext, 0, 3, 5, 0, 0);
    b = req(b, ext, 0, 0, 6, (1 << 20) - 4, 8);
    b = req(b, ext, 4, 0, 7, 0, 4096);
    // No NBD_CMD_DISC at the end: qemu-nbd may close before answering the requests still in
    // flight. The exchange ends when the server has nothing more to say.
    req(b, ext, 0, 42, 8, 0, 512).done()
}

/// What the ruvm server says to each script, checked against hand-written expectations for
/// the parts that do not depend on the image.
#[test]
fn server_traces() {
    let dir = scratch("strace");
    let srv = RuvmServer::start(&dir, true, Some("the disk"), true);
    let (greeting, out) = exchange(&srv.sock, &B::default().opt(OPT_ABORT, &[]).done());
    assert_eq!(greeting, B::default().greeting(3).done());
    assert_eq!(out, B::default().rep(OPT_ABORT, REP_ACK, &[]).done());

    let (_, out) =
        exchange(&srv.sock, &B::default().opt(OPT_STARTTLS, &[]).opt(OPT_ABORT, &[]).done());
    let expect = B::default()
        .rep(OPT_STARTTLS, 0x8000_0002, b"TLS not configured")
        .rep(OPT_ABORT, REP_ACK, &[])
        .done();
    assert_eq!(out, expect);

    let (_, out) = exchange(&srv.sock, &B::default().opt(OPT_LIST, &[]).opt(OPT_ABORT, &[]).done());
    let expect = B::default()
        .rep(OPT_LIST, REP_SERVER, &B::default().str32("exp").raw(b"the disk").done())
        .rep(OPT_LIST, REP_ACK, &[])
        .rep(OPT_ABORT, REP_ACK, &[])
        .done();
    assert_eq!(out, expect);

    let (_, out) = exchange(&srv.sock, &B::default().opt(99, b"xyz").opt(OPT_ABORT, &[]).done());
    let expect = B::default()
        .rep(99, REP_ERR_UNSUP, b"Unsupported option 99 (<unknown>)")
        .rep(OPT_ABORT, REP_ACK, &[])
        .done();
    assert_eq!(out, expect);

    // NBD_OPT_EXPORT_NAME: size, flags and no padding since the client set NO_ZEROES, then
    // the transmission phase. Close with NBD_CMD_DISC.
    let mut s = UnixStream::connect(&srv.sock).unwrap();
    let mut greeting = [0u8; 18];
    s.read_exact(&mut greeting).unwrap();
    s.write_all(&B::default().u32(3).opt(OPT_EXPORT_NAME, b"exp").done()).unwrap();
    let mut b = [0u8; 10];
    s.read_exact(&mut b).unwrap();
    assert_eq!(u64::from_be_bytes(b[..8].try_into().unwrap()), 1 << 20);
    let flags = u16::from_be_bytes([b[8], b[9]]);
    assert_eq!(flags & NBD_FLAG_READ_ONLY, 0);
    assert_ne!(flags & NBD_FLAG_SEND_WRITE_ZEROES, 0);
    assert_ne!(flags & NBD_FLAG_SEND_FAST_ZERO, 0);
}

/// The ruvm server and qemu-nbd give the same bytes for the same option exchanges.
#[test]
fn server_traces_match_qemu_nbd() {
    let Some(qemu_nbd) = tool("qemu-nbd") else {
        return;
    };
    let dir = scratch("sqemu");
    let srv = RuvmServer::start(&dir, true, Some("the disk"), true);
    let qsock = dir.join("q");
    let _q = QemuNbd::start(
        &qemu_nbd,
        &qsock,
        &srv.img,
        &["--export-name=exp", "--description=the disk", "--allocation-depth", "--shared=0"],
    );
    for (name, script) in server_scripts() {
        let (g1, ours) = exchange(&srv.sock, &script);
        let (g2, theirs) = exchange(&qsock, &script);
        assert_eq!(g1, g2, "greeting");
        let (ours, theirs) = if name.starts_with("transmission") {
            (by_cookie(&ours), by_cookie(&theirs))
        } else {
            (ours, theirs)
        };
        if ours != theirs {
            fs::write(dir.join(format!("{name}.ours")), &ours).unwrap();
            fs::write(dir.join(format!("{name}.theirs")), &theirs).unwrap();
        }
        assert_eq!(hex(&ours), hex(&theirs), "{name}");
    }
}

fn be(b: &[u8], at: usize, n: usize) -> u64 {
    b[at..at + n].iter().fold(0, |v, &x| v << 8 | u64::from(x))
}

/// The replies of the transmission phase sorted by cookie, keeping the order within one
/// cookie. qemu-nbd answers requests in parallel and so in any order; the replies to each one
/// are what has to match. The option phase before is left alone.
fn by_cookie(b: &[u8]) -> Vec<u8> {
    let mut at = 0;
    while b.len() >= at + 20 && be(b, at, 8) == REP_MAGIC {
        at += 20 + be(b, at + 16, 4) as usize;
    }
    let mut out = b[..at].to_vec();
    let mut chunks: Vec<(u64, &[u8])> = Vec::new();
    while at < b.len() {
        let (cookie, len) = match be(b, at, 4) {
            // A simple reply. Reads in the tests always get structured replies.
            0x6744_6698 => (be(b, at + 8, 8), 16),
            // A structured reply chunk.
            0x668e_33ef => (be(b, at + 8, 8), 20 + be(b, at + 16, 4) as usize),
            // An extended reply chunk.
            0x6e8a_278c => (be(b, at + 8, 8), 32 + be(b, at + 24, 8) as usize),
            m => panic!("bad reply magic {m:#x} at {at}"),
        };
        chunks.push((cookie, &b[at..at + len]));
        at += len;
    }
    chunks.sort_by_key(|c| c.0);
    for (_, c) in chunks {
        out.extend_from_slice(c);
    }
    out
}

fn hex(b: &[u8]) -> String {
    b.chunks(16)
        .map(|c| c.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------------------------
// Client and server together

fn check_io(c: &NbdClient, img: &str) {
    let file = fs::read(img).unwrap();
    assert_eq!(c.size(), file.len() as u64);
    let mut buf = vec![0u8; 70000];
    c.pread(1000, &mut buf).unwrap();
    assert_eq!(buf, file[1000..71000]);

    // A read past the end of the export is padded with zeroes.
    let mut tail = vec![1u8; 1024];
    c.pread(file.len() as u64 - 512, &mut tail).unwrap();
    assert_eq!(tail[..512], file[file.len() - 512..]);
    assert!(tail[512..].iter().all(|&b| b == 0));

    c.pwrite(4096, &[0xab; 8192], false).unwrap();
    c.pwrite(3, &[0xcd; 5], true).unwrap();
    c.flush().unwrap();
    c.pwrite_zeroes(65536, 4096, false, false, false).unwrap();
    // A server may refuse a fast zero it cannot do quickly (qemu-nbd over a macOS file does),
    // and then the caller writes the zeroes the slow way.
    if let Err(e) = c.pwrite_zeroes(69632, 4096, true, true, true) {
        assert_eq!(e.raw_os_error(), Some(libc::ENOTSUP));
        c.pwrite_zeroes(69632, 4096, true, true, false).unwrap();
    }
    c.pdiscard(0, 0).unwrap();
    c.cache(0, 65536).unwrap();

    let now = fs::read(img).unwrap();
    assert!(now[4096..12288].iter().all(|&b| b == 0xab));
    assert_eq!(now[3..8], [0xcd; 5]);
    assert!(now[65536..73728].iter().all(|&b| b == 0));
    let mut back = vec![0u8; 16];
    c.pread(4090, &mut back).unwrap();
    assert_eq!(back, now[4090..4106]);

    // The zeroed quarter of the image (see image()) reads as zero, whatever the host file
    // system says about allocation.
    let q = file.len() as u64 / 4;
    let st = c.block_status(q, q).unwrap();
    assert!(st.bytes >= 1 && st.bytes <= q);
    let st = c.block_status(0, 4096).unwrap();
    assert!(st.data);
    assert!(!st.zero);

    c.truncate(file.len() as u64, true).unwrap();
    assert_eq!(c.truncate(1, true).unwrap_err().message(), "Cannot resize NBD nodes");
    assert_eq!(c.truncate(1 << 30, false).unwrap_err().message(), "Cannot grow NBD nodes");
    c.close();
}

#[test]
fn client_server_unix() {
    let dir = scratch("unix");
    let srv = RuvmServer::start(&dir, true, None, false);
    let c = open_client(&client_opts(unix_addr(&srv.sock), "exp")).unwrap();
    let info = c.info();
    assert_eq!(info.mode, NbdMode::Extended);
    assert!(info.base_allocation);
    assert_eq!(
        c.exact_filename().unwrap(),
        format!("nbd+unix:///exp?socket={}", srv.sock.display())
    );
    check_io(&c, &srv.img);
}

#[test]
fn client_server_tcp() {
    let dir = scratch("tcp");
    let img = image(&dir, "img.raw", 1 << 20);
    let graph = graph_with("n0", &img, false);
    let server = NbdServer::start_addr(&tcp_addr("0"), 10, None, 0).unwrap();
    server.export_add(&graph, &export("exp", "n0", true, None, false)).unwrap();
    let addr = server.local_addr().unwrap().clone();
    let SocketAddressU::Inet(i) = &addr.u else { panic!() };
    assert_ne!(i.port, "0");
    let c = open_client(&client_opts(addr.clone(), "exp")).unwrap();
    assert_eq!(c.exact_filename().unwrap(), format!("nbd://127.0.0.1:{}/exp", i.port));
    check_io(&c, &img);
    server.stop();
}

#[test]
fn client_errors() {
    let dir = scratch("cerr");
    let srv = RuvmServer::start(&dir, false, None, false);

    // A read-only export makes the node read-only when auto-read-only allows it.
    let o = client_opts(unix_addr(&srv.sock), "exp");
    let mut ro = false;
    let c = NbdClient::open(&o, None, &mut ro, true).unwrap();
    assert!(ro);
    assert_eq!(c.pwrite(0, &[1], false).unwrap_err().raw_os_error(), Some(libc::EACCES));
    assert_eq!(
        c.reopen_check(false).unwrap_err().message(),
        "Can't reopen read-only NBD mount as read/write"
    );
    drop(c);
    let mut ro = false;
    let e = NbdClient::open(&o, Some("nbd:x"), &mut ro, false).unwrap_err();
    assert_eq!(e.message(), "Could not open 'nbd:x': Permission denied");

    let e = open_client(&client_opts(unix_addr(&srv.sock), "nope")).unwrap_err();
    assert_eq!(e.message(), "Requested export not available");

    let mut o = client_opts(unix_addr(&srv.sock), "exp");
    o.tls_creds = Some("tls0".into());
    assert_eq!(open_client(&o).unwrap_err().message(), "No TLS credentials with id 'tls0'");

    let mut o = client_opts(unix_addr(&srv.sock), "exp");
    o.x_dirty_bitmap = Some("qemu:dirty-bitmap:nope".into());
    let e = open_client(&o).unwrap_err();
    assert_eq!(e.message(), "Could not open image: Invalid argument");

    let o = client_opts(unix_addr(&dir.join("absent")), "exp");
    let e = open_client(&o).unwrap_err();
    assert!(
        e.message()
            .starts_with(&format!("Failed to connect to '{}'", dir.join("absent").display()))
    );
}

/// Socket options: TCP keep-alive tuning, and a Unix socket with no path, which QEMU puts in
/// the temporary directory.
#[test]
fn socket_options() {
    let dir = scratch("sockopt");
    let img = image(&dir, "img.raw", 1 << 16);
    let graph = graph_with("n0", &img, true);
    let server = NbdServer::start_addr(&tcp_addr("0"), 10, None, 0).unwrap();
    server.export_add(&graph, &export("exp", "n0", false, None, false)).unwrap();
    let mut addr = server.local_addr().unwrap().clone();
    if let SocketAddressU::Inet(i) = &mut addr.u {
        i.keep_alive = Some(true);
        i.keep_alive_count = Some(3);
        i.keep_alive_idle = Some(30);
        i.keep_alive_interval = Some(5);
    }
    let c = open_client(&client_opts(addr, "exp")).unwrap();
    let mut b = [0u8; 2];
    c.pread(251, &mut b).unwrap();
    assert_eq!(b, [0, 1]);
    c.close();
    server.stop();

    let server = NbdServer::start_addr(&unix_addr(Path::new("")), 10, None, 0).unwrap();
    server.export_add(&graph, &export("exp", "n0", false, None, false)).unwrap();
    let addr = server.local_addr().unwrap().clone();
    let SocketAddressU::Unix(u) = &addr.u else { panic!() };
    let tmp = std::env::temp_dir().join("qemu-socket-");
    assert!(u.path.starts_with(tmp.to_str().unwrap()), "{}", u.path);
    let path = PathBuf::from(&u.path);
    assert!(path.exists());
    let c = open_client(&client_opts(addr.clone(), "exp")).unwrap();
    c.close();
    server.stop();
    drop(server);
    assert!(!path.exists());

    let long = "x".repeat(200);
    let e = NbdServer::start_addr(&unix_addr(Path::new(&long)), 10, None, 0).unwrap_err();
    assert_eq!(e.message(), format!("UNIX socket path '{long}' is too long"));
}

/// Linux abstract sockets, tight and padded.
#[cfg(target_os = "linux")]
#[test]
fn abstract_sockets() {
    let dir = scratch("abstract");
    let img = image(&dir, "img.raw", 1 << 16);
    let graph = graph_with("n0", &img, true);
    for tight in [None, Some(false)] {
        let name = format!("ruvm-nbd-test-{}-{tight:?}", std::process::id());
        let mut addr = unix_addr(Path::new(&name));
        if let SocketAddressU::Unix(u) = &mut addr.u {
            u.abstract_ = Some(true);
            u.tight = tight;
        }
        let server = NbdServer::start_addr(&addr, 10, None, 0).unwrap();
        server.export_add(&graph, &export("exp", "n0", false, None, false)).unwrap();
        assert!(!Path::new(&name).exists());
        let c = open_client(&client_opts(addr.clone(), "exp")).unwrap();
        assert_eq!(c.size(), 1 << 16);
        c.close();
        server.stop();
    }
}

/// `reconnect-delay`: requests during a server outage wait for the server to come back.
#[test]
fn client_reconnect() {
    let dir = scratch("recon");
    let img = image(&dir, "img.raw", 1 << 16);
    let graph = graph_with("n0", &img, false);
    let sock = dir.join("s");
    let server = NbdServer::start_addr(&unix_addr(&sock), 10, None, 0).unwrap();
    server.export_add(&graph, &export("exp", "n0", false, None, false)).unwrap();
    let mut o = client_opts(unix_addr(&sock), "exp");
    o.reconnect_delay = Some(10);
    let c = open_client(&o).unwrap();
    let mut b = [0u8; 4];
    c.pread(0, &mut b).unwrap();

    server.stop();
    drop(server);
    let graph2 = graph_with("n1", &img, true);
    let restart = thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        let s = NbdServer::start_addr(&unix_addr(&sock), 10, None, 0).unwrap();
        s.export_add(&graph2, &export("exp", "n1", false, None, false)).unwrap();
        (s, graph2)
    });
    let t = Instant::now();
    c.pread(4, &mut b).unwrap();
    assert_eq!(b, [4, 5, 6, 7]);
    assert!(t.elapsed() < Duration::from_secs(10));
    let (s, _g) = restart.join().unwrap();
    c.close();
    s.stop();
    drop(graph);
}

/// The `nbd` driver through `blockdev-add` and a block backend.
#[test]
fn driver_blockdev_add() {
    let dir = scratch("drv");
    let srv = RuvmServer::start(&dir, true, None, false);
    let g = BlockGraph::new();
    let s = format!(
        r#"{{"driver": "nbd", "node-name": "remote", "export": "exp",
            "server": {{"type": "unix", "path": "{}"}}}}"#,
        srv.sock.display()
    );
    g.blockdev_add(opts::<BlockdevOptions>(&s)).unwrap();
    let blk: Arc<BlockBackend> = BlockBackend::new(
        &g,
        "remote",
        BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,
        BLK_PERM_CONSISTENT_READ,
    )
    .unwrap();
    assert_eq!(blk.getlength().unwrap(), 1 << 20);
    blk.pwrite(100, b"hello").unwrap();
    let mut b = [0u8; 5];
    blk.pread(100, &mut b).unwrap();
    assert_eq!(&b, b"hello");
    assert_eq!(&fs::read(&srv.img).unwrap()[100..105], b"hello");
}

// ---------------------------------------------------------------------------------------------
// QEMU's tools

fn tool(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(d) = std::env::var_os("QEMU_BIN_DIR") {
        dirs.push(d.into());
    }
    dirs.extend(["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].map(PathBuf::from));
    let found = dirs.into_iter().map(|d| d.join(name)).find(|p| p.is_file());
    if found.is_none() {
        eprintln!("skipping: {name} not found");
    }
    found
}

/// A qemu-nbd serving an image on a Unix socket, killed on drop.
struct QemuNbd(Child);

impl QemuNbd {
    fn start(bin: &Path, sock: &Path, img: &str, extra: &[&str]) -> QemuNbd {
        let child = Command::new(bin)
            .arg("--persistent")
            .arg("--format=raw")
            .arg(format!("--socket={}", sock.display()))
            .args(extra)
            .arg(img)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let t = Instant::now();
        while UnixStream::connect(sock).is_err() {
            assert!(t.elapsed() < Duration::from_secs(10), "qemu-nbd did not start");
            thread::sleep(Duration::from_millis(20));
        }
        QemuNbd(child)
    }
}

impl Drop for QemuNbd {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn client_against_qemu_nbd() {
    let Some(qemu_nbd) = tool("qemu-nbd") else {
        return;
    };
    let dir = scratch("qnbd");
    let img = image(&dir, "img.raw", 1 << 20);
    let sock = dir.join("q");
    let _q = QemuNbd::start(&qemu_nbd, &sock, &img, &["--export-name=exp", "--allocation-depth"]);
    let c = open_client(&client_opts(unix_addr(&sock), "exp")).unwrap();
    assert_eq!(c.info().mode, NbdMode::Extended);
    check_io(&c, &img);

    // qemu:allocation-depth instead of base:allocation: everything is in the top layer.
    let mut o = client_opts(unix_addr(&sock), "exp");
    o.x_dirty_bitmap = Some("qemu:allocation-depth".into());
    let c = open_client(&o).unwrap();
    let st = c.block_status(0, 1 << 20).unwrap();
    assert!(st.bytes > 0);
    c.close();

    let e = open_client(&client_opts(unix_addr(&sock), "nope")).unwrap_err();
    assert_eq!(e.message(), "Requested export not available");
}

#[test]
fn client_against_qemu_nbd_read_only() {
    let Some(qemu_nbd) = tool("qemu-nbd") else {
        return;
    };
    let dir = scratch("qnbdro");
    let img = image(&dir, "img.raw", 1 << 16);
    let sock = dir.join("q");
    let _q = QemuNbd::start(&qemu_nbd, &sock, &img, &["--read-only", "--export-name="]);
    let mut ro = false;
    let c = NbdClient::open(&client_opts(unix_addr(&sock), ""), None, &mut ro, true).unwrap();
    assert!(ro);
    assert_ne!(c.info().flags & NBD_FLAG_READ_ONLY, 0);
    let mut b = [0u8; 3];
    c.pread(250, &mut b).unwrap();
    assert_eq!(b, [250, 0, 1]);
}

fn qemu_img(bin: &Path, args: &[&str]) -> (bool, String, String) {
    let out = Command::new(bin).args(args).output().unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn qemu_img_against_ruvm_server() {
    let Some(qemu_img_bin) = tool("qemu-img") else {
        return;
    };
    let dir = scratch("qimg");
    let srv = RuvmServer::start(&dir, false, Some("desc"), true);
    let uri = format!("nbd+unix:///exp?socket={}", srv.sock.display());

    let (ok, out, err) = qemu_img(&qemu_img_bin, &["info", "--output=json", &uri]);
    assert!(ok, "{err}");
    assert!(out.contains("\"virtual-size\": 1048576"), "{out}");
    assert!(out.contains("\"format\": \"raw\""), "{out}");

    let (ok, out, err) =
        qemu_img(&qemu_img_bin, &["compare", "-f", "raw", "-F", "raw", &srv.img, &uri]);
    assert!(ok, "{err}");
    assert_eq!(out, "Images are identical.\n");

    // The same map through qemu-nbd and through the ruvm server.
    if let Some(qemu_nbd) = tool("qemu-nbd") {
        let qsock = dir.join("q");
        let _q = QemuNbd::start(&qemu_nbd, &qsock, &srv.img, &["--export-name=exp", "--read-only"]);
        let quri = format!("nbd+unix:///exp?socket={}", qsock.display());
        let (ok1, ours, e1) = qemu_img(&qemu_img_bin, &["map", "--output=json", "-f", "raw", &uri]);
        let (ok2, theirs, e2) =
            qemu_img(&qemu_img_bin, &["map", "--output=json", "-f", "raw", &quri]);
        assert!(ok1 && ok2, "{e1}{e2}");
        assert_eq!(ours, theirs);
    }

    let (ok, _, err) = qemu_img(
        &qemu_img_bin,
        &["info", &format!("nbd+unix:///nope?socket={}", srv.sock.display())],
    );
    assert!(!ok);
    assert!(err.contains("Requested export not available"), "{err}");
}
