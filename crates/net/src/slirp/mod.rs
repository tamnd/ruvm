// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev user`, the user mode network stack, ported from net/slirp.c.
//!
//! The stack itself is libslirp, loaded when the first `user` backend is made (see [`ffi`]). Each
//! backend owns one `Slirp` instance and a thread that does what QEMU's main loop does for it:
//! polling the sockets libslirp asks for, firing its timers and passing frames on to the peer.
//! libslirp is not thread safe, so every call into an instance happens with its lock held.
//!
//! Frames libslirp emits are queued and delivered once the lock is released, so that a peer
//! that answers right away can call back into the stack without deadlocking.

#![allow(unsafe_code)]

mod ffi;

use std::collections::{HashMap, VecDeque};
use std::ffi::{CString, c_char, c_int, c_void};
use std::io::{Read, Write};
use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, TcpStream};
use std::os::fd::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::thread::JoinHandle;
use std::time::Instant;

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::cutils::strtoi64;
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevU, NetdevUserOptions};

use crate::client::{NetClient, NetClientOps, eth_pad_short_frame, lock};
use crate::hub::hub_id_for_client;
use crate::net::Net;
use crate::poll::Waker;
use crate::util::inet_aton;

use ffi::{InAddr, Lib, SlirpCb, SlirpConfig, SlirpPtr, TimerCb};

/// `CONFIG_SMBD_COMMAND`, the default meson picks.
const SMBD_COMMAND: &str = "/usr/sbin/smbd";

/// The longest wait between two looks at the timers, in milliseconds.
const MAX_POLL_MS: u32 = 1000;

/// `sizeof(((struct sockaddr_un *)0)->sun_path)`.
#[cfg(target_os = "linux")]
const SUN_PATH_LEN: usize = 108;
#[cfg(not(target_os = "linux"))]
const SUN_PATH_LEN: usize = 104;

/// Which end of the string [`get_str_sep`] searches from.
#[derive(Clone, Copy)]
enum Sep {
    First(u8),
    Last(u8),
}

/// `get_str_sep()`: takes the text up to `sep` off the front of `p`, cut to `size - 1` bytes as
/// the C buffer would. `None` when there is no separator.
fn get_str_sep(p: &mut &str, size: usize, sep: Sep) -> Option<String> {
    let at = match sep {
        Sep::First(c) => p.bytes().position(|b| b == c)?,
        Sep::Last(c) => p.bytes().rposition(|b| b == c)?,
    };
    let bytes = &p.as_bytes()[..at.min(size - 1)];
    let buf = String::from_utf8_lossy(bytes).into_owned();
    *p = &p[at + 1..];
    Some(buf)
}

/// `strtol(s, &end, 10)`: the value and the index where the number ends (0 without digits).
fn strtol10(s: &str) -> (i64, usize) {
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() && matches!(b[i], b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r') {
        i += 1;
    }
    let neg = i < b.len() && b[i] == b'-';
    if i < b.len() && (b[i] == b'-' || b[i] == b'+') {
        i += 1;
    }
    let start = i;
    let mut v: i64 = 0;
    while i < b.len() && b[i].is_ascii_digit() {
        let d = i64::from(b[i] - b'0');
        v = if neg {
            v.saturating_mul(10).saturating_sub(d)
        } else {
            v.saturating_mul(10).saturating_add(d)
        };
        i += 1;
    }
    if i == start { (0, 0) } else { (v, i) }
}

/// `qemu_strtoi()`. With `need_all` false the number may be followed by anything, as when the
/// caller passes an end pointer.
fn qemu_strtoi(s: &str, base: u32, need_all: bool) -> Option<i32> {
    match strtoi64(s, base, need_all) {
        Ok((v, _)) => i32::try_from(v).ok(),
        Err(_) => None,
    }
}

/// `in6_equal_net()`.
fn in6_equal_net(a: &Ipv6Addr, b: &Ipv6Addr, prefix_len: i32) -> bool {
    let (a, b) = (a.octets(), b.octets());
    let n = (prefix_len / 8) as usize;
    if a[..n] != b[..n] {
        return false;
    }
    let bits = prefix_len % 8;
    if bits == 0 {
        return true;
    }
    let mask = 0xffu8 << (8 - bits);
    a[n] & mask == b[n] & mask
}

/// A string for libslirp, cut at the first NUL as C would see it.
fn cstring(s: &str) -> CString {
    let end = s.bytes().position(|b| b == 0).unwrap_or(s.len());
    CString::new(&s.as_bytes()[..end]).unwrap_or_default()
}

/// `pstrcpy()` into a buffer of `size` bytes.
fn truncated(s: &str, size: usize) -> String {
    if s.len() < size {
        return s.to_string();
    }
    String::from_utf8_lossy(&s.as_bytes()[..size - 1]).into_owned()
}

fn err(msg: impl Into<String>) -> Error {
    Error::generic(msg.into())
}

/// A forwarding rule from the command line.
#[derive(Debug)]
enum Fwd {
    Host(String),
    Guest(String),
}

/// What `net_init_slirp()` and the first half of `net_slirp_init()` work out from the options,
/// before anything is created.
#[derive(Debug)]
struct Config {
    restricted: bool,
    ipv4: bool,
    ipv6: bool,
    net: u32,
    mask: u32,
    host: u32,
    dhcp: u32,
    dns: u32,
    smbsrv: u32,
    prefix6: Ipv6Addr,
    prefix6_len: i32,
    host6: Ipv6Addr,
    dns6: Ipv6Addr,
    hostname: Option<CString>,
    tftp_server_name: Option<CString>,
    tftp: Option<CString>,
    bootfile: Option<CString>,
    domainname: Option<CString>,
    dnssearch: Vec<CString>,
    smb: Option<String>,
    fwds: Vec<Fwd>,
}

fn parse_v4(s: &str, what: &str) -> Result<u32> {
    inet_aton(s).map(u32::from).ok_or_else(|| err(what))
}

impl Config {
    fn parse(user: &NetdevUserOptions) -> Result<Config> {
        let mut ipv4 = true;
        let mut ipv6 = true;
        if (user.ipv6 == Some(true) && user.ipv4.is_none()) || user.ipv4 == Some(false) {
            ipv4 = false;
        }
        if (user.ipv4 == Some(true) && user.ipv6.is_none()) || user.ipv6 == Some(false) {
            ipv6 = false;
        }
        let vnet = match (&user.net, &user.ip) {
            (Some(n), _) => Some(n.clone()),
            (None, Some(ip)) => Some(format!("{ip}/24")),
            (None, None) => None,
        };
        let restricted = user.restrict.unwrap_or(false);

        // net_init_slirp_configs_host() and _guest() push to the front of one list, host rules
        // first, so the guest rules run first and each group runs backwards.
        let mut fwds = Vec::new();
        for g in user.guestfwd.iter().flatten().rev() {
            fwds.push(Fwd::Guest(truncated(&g.str, 1024)));
        }
        for h in user.hostfwd.iter().flatten().rev() {
            fwds.push(Fwd::Host(truncated(&h.str, 1024)));
        }

        if !ipv4 && (vnet.is_some() || user.host.is_some() || user.dns.is_some()) {
            return Err(err("IPv4 disabled but netmask/host/dns provided"));
        }
        if !ipv6
            && (user.ipv6_prefix.is_some() || user.ipv6_host.is_some() || user.ipv6_dns.is_some())
        {
            return Err(err("IPv6 disabled but prefix/host6/dns6 provided"));
        }
        if !ipv4 && !ipv6 {
            return Err(err("IPv4 and IPv6 disabled"));
        }

        let mut net: u32 = 0x0a00_0200;
        let mut mask: u32 = 0xffff_ff00;
        let mut host: u32 = 0x0a00_0202;
        let mut dhcp: u32 = 0x0a00_020f;
        let mut dns: u32 = 0x0a00_0203;

        if let Some(vnet) = &vnet {
            let mut p = vnet.as_str();
            match get_str_sep(&mut p, 20, Sep::First(b'/')) {
                None => {
                    net = parse_v4(p, "Failed to parse netmask")?;
                    // The same order of tests as slirp's net_init, kept even where two give
                    // the same mask.
                    #[allow(clippy::if_same_then_else)]
                    let m = if net & 0x8000_0000 == 0 {
                        0xff00_0000
                    } else if net & 0xfff0_0000 == 0xac10_0000 {
                        0xfff0_0000
                    } else if net & 0xc000_0000 == 0x8000_0000 {
                        0xffff_0000
                    } else if net & 0xffff_0000 == 0xc0a8_0000 {
                        0xffff_0000
                    } else if net & 0xffff_0000 == 0xc612_0000 {
                        0xfffe_0000
                    } else if net & 0xe000_0000 == 0xe000_0000 {
                        0xffff_ff00
                    } else {
                        0xffff_fff0
                    };
                    mask = m;
                }
                Some(buf) => {
                    net = parse_v4(&buf, "Failed to parse netmask")?;
                    let (shift, end) = strtol10(p);
                    let shift = shift as i32;
                    if end != p.len() {
                        mask = parse_v4(p, "Failed to parse netmask (trailing chars)")?;
                    } else if !(4..=32).contains(&shift) {
                        return Err(err("Invalid netmask provided (must be in range 4-32)"));
                    } else {
                        mask = 0xffff_ffffu32 << (32 - shift);
                    }
                }
            }
            net &= mask;
            host = net | (0x0202 & !mask);
            dhcp = net | (0x020f & !mask);
            dns = net | (0x0203 & !mask);
        }

        if let Some(h) = &user.host {
            host = parse_v4(h, "Failed to parse host")?;
        }
        if host & mask != net {
            return Err(err("Host doesn't belong to network"));
        }
        if let Some(d) = &user.dns {
            dns = parse_v4(d, "Failed to parse DNS")?;
        }
        if restricted && dns & mask != net {
            return Err(err("DNS doesn't belong to network"));
        }
        if dns == host {
            return Err(err("DNS must be different from host"));
        }
        if let Some(d) = &user.dhcpstart {
            dhcp = parse_v4(d, "Failed to parse DHCP start address")?;
        }
        if dhcp & mask != net {
            return Err(err("DHCP doesn't belong to network"));
        }
        if dhcp == host || dhcp == dns {
            return Err(err("DHCP must be different from host and DNS"));
        }
        let mut smbsrv = 0;
        if let Some(s) = &user.smbserver {
            smbsrv = parse_v4(s, "Failed to parse SMB address")?;
        }

        let prefix6: Ipv6Addr = user
            .ipv6_prefix
            .as_deref()
            .unwrap_or("fec0::")
            .parse()
            .map_err(|_| err("Failed to parse IPv6 prefix"))?;
        let mut prefix6_len = user.ipv6_prefixlen.unwrap_or(0) as i32;
        if prefix6_len == 0 {
            prefix6_len = 64;
        }
        if !(0..=126).contains(&prefix6_len) {
            return Err(err(
                "Invalid IPv6 prefix provided (IPv6 prefix length must be between 0 and 126)",
            ));
        }
        let host6 = match &user.ipv6_host {
            Some(h) => {
                let a: Ipv6Addr = h.parse().map_err(|_| err("Failed to parse IPv6 host"))?;
                if !in6_equal_net(&prefix6, &a, prefix6_len) {
                    return Err(err("IPv6 Host doesn't belong to network"));
                }
                a
            }
            None => {
                let mut o = prefix6.octets();
                o[15] |= 2;
                Ipv6Addr::from(o)
            }
        };
        let dns6 = match &user.ipv6_dns {
            Some(d) => {
                let a: Ipv6Addr = d.parse().map_err(|_| err("Failed to parse IPv6 DNS"))?;
                if restricted && !in6_equal_net(&prefix6, &a, prefix6_len) {
                    return Err(err("IPv6 DNS doesn't belong to network"));
                }
                a
            }
            None => {
                let mut o = prefix6.octets();
                o[15] |= 3;
                Ipv6Addr::from(o)
            }
        };

        if let Some(d) = &user.domainname {
            if d.is_empty() {
                return Err(err("'domainname' parameter cannot be empty"));
            }
            if d.len() > 255 {
                return Err(err("'domainname' parameter cannot exceed 255 bytes"));
            }
        }
        if user.hostname.as_ref().is_some_and(|h| h.len() > 255) {
            return Err(err("'vhostname' parameter cannot exceed 255 bytes"));
        }
        if user.tftp_server_name.as_ref().is_some_and(|h| h.len() > 255) {
            return Err(err("'tftp-server-name' parameter cannot exceed 255 bytes"));
        }

        let c = |s: &Option<String>| s.as_deref().map(cstring);
        Ok(Config {
            restricted,
            ipv4,
            ipv6,
            net,
            mask,
            host,
            dhcp,
            dns,
            smbsrv,
            prefix6,
            prefix6_len,
            host6,
            dns6,
            hostname: c(&user.hostname),
            tftp_server_name: c(&user.tftp_server_name),
            tftp: c(&user.tftp),
            bootfile: c(&user.bootfile),
            domainname: c(&user.domainname),
            dnssearch: user.dnssearch.iter().flatten().map(|d| cstring(&d.str)).collect(),
            smb: user.smb.clone(),
            fwds,
        })
    }
}

fn in_addr(host_order: u32) -> InAddr {
    InAddr::from(Ipv4Addr::from(host_order))
}

/// What a timer does when it fires.
#[derive(Clone, Copy, Debug)]
enum TimerKind {
    /// `slirp_handle_timer(slirp, id, cb_opaque)`, libslirp 4.7 and later.
    Id { id: c_int, cb_opaque: usize },
    /// `cb(cb_opaque)`, older libslirp.
    Cb { cb: TimerCb, cb_opaque: usize },
}

#[derive(Debug)]
struct Timer {
    kind: TimerKind,
    /// Deadline in milliseconds on [`Inner::now_ms`]'s clock, or `None` while not armed.
    expire: Option<i64>,
}

#[derive(Debug, Default)]
struct Timers {
    next: usize,
    map: HashMap<usize, Timer>,
}

/// `struct GuestFwd`: a guest forwarding rule whose host end is a TCP connection.
#[derive(Debug)]
struct GuestFwd {
    stream: TcpStream,
    server: InAddr,
    port: c_int,
    eof: AtomicBool,
}

/// What lives behind the lock that serializes calls into libslirp.
#[derive(Debug, Default)]
struct Stack {
    alive: bool,
    /// Boxed because libslirp keeps a pointer to each one as its opaque.
    #[allow(clippy::vec_box)]
    guestfwds: Vec<Box<GuestFwd>>,
    smb_dir: Option<PathBuf>,
}

/// `SlirpState`. Its address is the opaque pointer libslirp hands back to the callbacks.
#[derive(Debug)]
struct Inner {
    lib: &'static Lib,
    slirp: AtomicPtr<c_void>,
    nc: OnceLock<Weak<NetClient>>,
    pending: Mutex<VecDeque<Vec<u8>>>,
    delivering: Mutex<()>,
    timers: Mutex<Timers>,
    waker: Arc<Waker>,
    clock_base: Instant,
    stack: Mutex<Stack>,
}

/// The [`Inner`] a callback was registered with.
fn from_opaque<'a>(opaque: *mut c_void) -> &'a Inner {
    // SAFETY: libslirp only calls the callbacks in CALLBACKS with the opaque pointer given to
    // slirp_new(), which is the address of an Inner kept alive by an Arc until after
    // slirp_cleanup() returned, and nothing calls into that instance afterwards.
    unsafe { &*opaque.cast::<Inner>() }
}

impl Inner {
    fn now_ms(&self) -> i64 {
        (self.clock_base.elapsed().as_nanos() / 1_000_000) as i64
    }

    fn slirp(&self) -> SlirpPtr {
        self.slirp.load(Ordering::SeqCst)
    }

    /// Sends the frames libslirp produced on to the peer. Only one thread does that at a time;
    /// the others leave their frames to it.
    fn deliver(&self) {
        loop {
            let Ok(guard) = self.delivering.try_lock() else {
                return;
            };
            while let Some(frame) = lock(&self.pending).pop_front() {
                let Some(nc) = self.nc.get().and_then(Weak::upgrade) else {
                    continue;
                };
                // net_slirp_send_packet()
                match nc.peer_needs_padding().then(|| eth_pad_short_frame(&frame)).flatten() {
                    Some(padded) => nc.send_packet(&padded),
                    None => nc.send_packet(&frame),
                };
            }
            drop(guard);
            if lock(&self.pending).is_empty() {
                return;
            }
        }
    }

    /// The soonest timer deadline, in milliseconds from now.
    fn next_timer_ms(&self) -> Option<i64> {
        let now = self.now_ms();
        lock(&self.timers).map.values().filter_map(|t| t.expire).min().map(|e| (e - now).max(0))
    }

    fn fire_timers(&self) {
        loop {
            let now = self.now_ms();
            let due = {
                let mut t = lock(&self.timers);
                t.map.values_mut().find(|t| t.expire.is_some_and(|e| e <= now)).map(|t| {
                    t.expire = None;
                    t.kind
                })
            };
            let Some(kind) = due else {
                return;
            };
            let st = lock(&self.stack);
            if !st.alive {
                return;
            }
            match kind {
                TimerKind::Id { id, cb_opaque } => {
                    if let Some(handle_timer) = self.lib.handle_timer {
                        let p = std::ptr::with_exposed_provenance_mut(cb_opaque);
                        // SAFETY: the instance is alive (checked under the lock), and id and
                        // cb_opaque are what libslirp passed to timer_new_opaque.
                        unsafe { handle_timer(self.slirp(), id, p) };
                    }
                }
                TimerKind::Cb { cb, cb_opaque } => {
                    let p = std::ptr::with_exposed_provenance_mut(cb_opaque);
                    // SAFETY: cb and its argument came from libslirp's timer_new call for an
                    // instance that is still alive, and the lock serializes it with other calls.
                    unsafe { cb(p) };
                }
            }
            drop(st);
        }
    }
}

extern "C" fn cb_send_packet(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize {
    let inner = from_opaque(opaque);
    // SAFETY: libslirp passes a frame of len bytes that stays valid for the call.
    let frame = unsafe { std::slice::from_raw_parts(buf.cast::<u8>(), len) }.to_vec();
    lock(&inner.pending).push_back(frame);
    inner.waker.wake();
    len as isize
}

/// `net_slirp_guest_error()`: QEMU logs these only with `-d guest_errors`, which ruvm does not
/// have, so they are dropped.
extern "C" fn cb_guest_error(_msg: *const c_char, _opaque: *mut c_void) {}

extern "C" fn cb_clock_get_ns(opaque: *mut c_void) -> i64 {
    from_opaque(opaque).clock_base.elapsed().as_nanos() as i64
}

fn new_timer(inner: &Inner, kind: TimerKind) -> *mut c_void {
    let mut t = lock(&inner.timers);
    t.next += 1;
    let id = t.next;
    t.map.insert(id, Timer { kind, expire: None });
    std::ptr::without_provenance_mut(id)
}

extern "C" fn cb_timer_new(
    cb: TimerCb,
    cb_opaque: *mut c_void,
    opaque: *mut c_void,
) -> *mut c_void {
    new_timer(from_opaque(opaque), TimerKind::Cb { cb, cb_opaque: cb_opaque.expose_provenance() })
}

extern "C" fn cb_timer_new_opaque(
    id: c_int,
    cb_opaque: *mut c_void,
    opaque: *mut c_void,
) -> *mut c_void {
    new_timer(from_opaque(opaque), TimerKind::Id { id, cb_opaque: cb_opaque.expose_provenance() })
}

extern "C" fn cb_timer_free(timer: *mut c_void, opaque: *mut c_void) {
    lock(&from_opaque(opaque).timers).map.remove(&timer.addr());
}

extern "C" fn cb_timer_mod(timer: *mut c_void, expire_ms: i64, opaque: *mut c_void) {
    let inner = from_opaque(opaque);
    if let Some(t) = lock(&inner.timers).map.get_mut(&timer.addr()) {
        t.expire = Some(expire_ms);
    }
    inner.waker.wake();
}

extern "C" fn cb_register_poll(_fd: c_int, _opaque: *mut c_void) {}

extern "C" fn cb_notify(opaque: *mut c_void) {
    from_opaque(opaque).waker.wake();
}

extern "C" fn cb_init_completed(slirp: SlirpPtr, opaque: *mut c_void) {
    from_opaque(opaque).slirp.store(slirp, Ordering::SeqCst);
}

/// `slirp_cb`. libslirp keeps the pointer, so it lives for the whole run.
static CALLBACKS: SlirpCb = SlirpCb {
    send_packet: Some(cb_send_packet),
    guest_error: Some(cb_guest_error),
    clock_get_ns: Some(cb_clock_get_ns),
    timer_new: Some(cb_timer_new),
    timer_free: Some(cb_timer_free),
    timer_mod: Some(cb_timer_mod),
    register_poll_fd: Some(cb_register_poll),
    unregister_poll_fd: Some(cb_register_poll),
    notify: Some(cb_notify),
    init_completed: Some(cb_init_completed),
    timer_new_opaque: Some(cb_timer_new_opaque),
    register_poll_socket: Some(cb_register_poll),
    unregister_poll_socket: Some(cb_register_poll),
};

fn slirp_to_poll(events: c_int) -> i16 {
    let mut r = 0;
    for (s, p) in POLL_MAP {
        if events & s != 0 {
            r |= p;
        }
    }
    r
}

const POLL_MAP: [(c_int, i16); 5] = [
    (ffi::POLL_IN, libc::POLLIN),
    (ffi::POLL_OUT, libc::POLLOUT),
    (ffi::POLL_PRI, libc::POLLPRI),
    (ffi::POLL_ERR, libc::POLLERR),
    (ffi::POLL_HUP, libc::POLLHUP),
];

/// `net_slirp_add_poll()`.
extern "C" fn cb_add_poll(fd: c_int, events: c_int, opaque: *mut c_void) -> c_int {
    // SAFETY: the opaque pointer is the &mut Vec the I/O thread passes to pollfds_fill, which
    // stays borrowed for the whole call.
    let fds = unsafe { &mut *opaque.cast::<Vec<libc::pollfd>>() };
    fds.push(libc::pollfd { fd, events: slirp_to_poll(events), revents: 0 });
    (fds.len() - 1) as c_int
}

/// `net_slirp_get_revents()`.
extern "C" fn cb_get_revents(idx: c_int, opaque: *mut c_void) -> c_int {
    // SAFETY: as in cb_add_poll, for the Vec passed to pollfds_poll.
    let fds = unsafe { &*opaque.cast::<Vec<libc::pollfd>>() };
    let revents = usize::try_from(idx).ok().and_then(|i| fds.get(i)).map_or(0, |p| p.revents);
    let mut r = 0;
    for (s, p) in POLL_MAP {
        if revents & p != 0 {
            r |= s;
        }
    }
    r
}

/// `guestfwd_write()`: what the guest sends to a forwarded address goes to the connection.
extern "C" fn cb_guestfwd_write(buf: *const c_void, len: usize, opaque: *mut c_void) -> isize {
    // SAFETY: opaque is the boxed GuestFwd registered with slirp_add_guestfwd(), which lives
    // until after slirp_cleanup(), and buf holds len bytes for the call.
    let (fwd, data) =
        unsafe { (&*opaque.cast::<GuestFwd>(), std::slice::from_raw_parts(buf.cast::<u8>(), len)) };
    match (&fwd.stream).write_all(data) {
        Ok(()) => len as isize,
        Err(_) => -1,
    }
}

/// The client side of a `user` backend.
#[derive(Debug)]
pub(crate) struct SlirpOps {
    inner: Arc<Inner>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl NetClientOps for SlirpOps {
    /// `net_slirp_receive()`.
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        let data = iov.concat();
        let inner = &self.inner;
        {
            let st = lock(&inner.stack);
            if st.alive {
                // SAFETY: the instance is alive and the lock is held; data outlives the call.
                unsafe { (inner.lib.input)(inner.slirp(), data.as_ptr(), data.len() as c_int) };
            }
        }
        inner.waker.wake();
        inner.deliver();
        data.len() as isize
    }

    /// `net_slirp_cleanup()`.
    fn cleanup(&self, _nc: &NetClient) {
        let inner = &self.inner;
        inner.waker.request_stop();
        if let Some(h) = lock(&self.thread).take() {
            if h.thread().id() != std::thread::current().id() {
                let _ = h.join();
            }
        }
        let mut st = lock(&inner.stack);
        if !st.alive {
            return;
        }
        st.alive = false;
        let slirp = inner.slirp.swap(std::ptr::null_mut(), Ordering::SeqCst);
        if !slirp.is_null() {
            // SAFETY: the instance came from slirp_new() and is freed once, under the lock; the
            // I/O thread has stopped and every other caller checks `alive` first.
            unsafe { (inner.lib.cleanup)(slirp) };
        }
        st.guestfwds.clear();
        smb_cleanup(&mut st);
        drop(st);
        lock(&inner.pending).clear();
    }

    fn as_any(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        Some(self)
    }
}

/// The I/O thread: QEMU's main loop poll notifier, timers and chardev handlers for one stack.
fn run(inner: Arc<Inner>) {
    let mut fds: Vec<libc::pollfd> = Vec::new();
    let mut buf = vec![0u8; 4096];
    while !inner.waker.stopped() {
        fds.clear();
        fds.push(libc::pollfd { fd: inner.waker.read_fd(), events: libc::POLLIN, revents: 0 });
        let mut timeout: u32 = MAX_POLL_MS;
        let guest_start;
        {
            let st = lock(&inner.stack);
            if !st.alive {
                return;
            }
            let opaque: *mut c_void = (&raw mut fds).cast();
            // SAFETY: the instance is alive and the lock is held; opaque is the Vec cb_add_poll
            // expects, borrowed for the call.
            unsafe { (inner.lib.pollfds_fill)(inner.slirp(), &mut timeout, cb_add_poll, opaque) };
            guest_start = fds.len();
            for g in &st.guestfwds {
                let want = !g.eof.load(Ordering::SeqCst)
                    // SAFETY: as above.
                    && unsafe { (inner.lib.socket_can_recv)(inner.slirp(), g.server, g.port) } > 0;
                let events = if want { libc::POLLIN } else { 0 };
                fds.push(libc::pollfd { fd: g.stream.as_raw_fd(), events, revents: 0 });
            }
        }
        let mut wait = i64::from(timeout.min(MAX_POLL_MS));
        if let Some(t) = inner.next_timer_ms() {
            wait = wait.min(t);
        }
        // SAFETY: fds is a live array of fds.len() pollfd entries.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, wait as c_int) };
        if fds[0].revents != 0 {
            inner.waker.drain();
        }
        {
            let st = lock(&inner.stack);
            if !st.alive {
                return;
            }
            let opaque: *mut c_void = (&raw mut fds).cast();
            // SAFETY: as for pollfds_fill; the indices libslirp asks about are the ones
            // cb_add_poll gave it.
            unsafe {
                (inner.lib.pollfds_poll)(inner.slirp(), c_int::from(n < 0), cb_get_revents, opaque)
            };
            for (i, g) in st.guestfwds.iter().enumerate() {
                if n <= 0 || fds[guest_start + i].revents == 0 {
                    continue;
                }
                // SAFETY: as above.
                let room = unsafe { (inner.lib.socket_can_recv)(inner.slirp(), g.server, g.port) };
                if room == 0 {
                    continue;
                }
                let len = room.min(buf.len());
                match (&g.stream).read(&mut buf[..len]) {
                    Ok(0) | Err(_) => g.eof.store(true, Ordering::SeqCst),
                    Ok(got) => {
                        // SAFETY: as above; buf holds got bytes.
                        unsafe {
                            (inner.lib.socket_recv)(
                                inner.slirp(),
                                g.server,
                                g.port,
                                buf.as_ptr(),
                                got as c_int,
                            )
                        };
                    }
                }
            }
        }
        inner.fire_timers();
        inner.deliver();
    }
}

/// `slirp_smb_cleanup()`.
fn smb_cleanup(st: &mut Stack) {
    if let Some(dir) = st.smb_dir.take() {
        if std::fs::remove_dir_all(&dir).is_err() {
            error_report(&format!("'rm -rf {}' failed.", dir.display()));
        }
    }
}

/// The user name of the effective user, `getpwuid(geteuid())->pw_name`.
fn user_name() -> Option<String> {
    let mut pw = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut buf = vec![0 as c_char; 4096];
    let mut out: *mut libc::passwd = std::ptr::null_mut();
    // SAFETY: getpwuid_r() fills pw and buf, whose sizes are passed along, and sets out to pw
    // on success. pw_name then points into buf, which is still alive when it is copied.
    unsafe {
        let r = libc::getpwuid_r(
            libc::geteuid(),
            pw.as_mut_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
            &mut out,
        );
        if r != 0 || out.is_null() || (*out).pw_name.is_null() {
            return None;
        }
        Some(std::ffi::CStr::from_ptr((*out).pw_name).to_string_lossy().into_owned())
    }
}

/// A `user` backend, as the monitor commands find it.
struct Slirp<'a> {
    inner: &'a Inner,
}

impl Slirp<'_> {
    /// `slirp_hostfwd()`.
    fn hostfwd(&self, st: &Stack, redir: &str) -> Result<()> {
        let syntax = |why: &str| err(format!("Invalid host forwarding rule '{redir}' ({why})"));
        let lib = self.inner.lib;
        let mut p = redir;
        let Some(proto) = get_str_sep(&mut p, 256, Sep::First(b':')) else {
            return Err(syntax("No : separators"));
        };
        let unix_ok = lib.at_least(4, 7);
        let (is_udp, is_unix) = match proto.as_str() {
            "tcp" | "" => (false, false),
            "udp" => (true, false),
            "unix" if unix_ok => (false, true),
            _ => return Err(syntax("Bad protocol name")),
        };
        let host_addr: rustix::net::SocketAddrAny = if is_unix {
            let Some(path) = get_str_sep(&mut p, 256, Sep::Last(b'-')) else {
                return Err(syntax("Missing - separator"));
            };
            if path.is_empty() {
                return Err(syntax("Missing unix socket path"));
            }
            if !path.starts_with('/') {
                return Err(syntax("unix socket path must be absolute"));
            }
            if path.len() > SUN_PATH_LEN - 1 {
                return Err(syntax("Unix socket path is too long"));
            }
            if let Ok(md) = std::fs::metadata(&path) {
                use std::os::unix::fs::FileTypeExt;
                if !md.file_type().is_socket() {
                    return Err(syntax("file exists and it's not unix socket"));
                }
                if let Err(e) = std::fs::remove_file(&path) {
                    return Err(err(format!("Failed to unlink '{path}': {}", strerror(&e))));
                }
            }
            match rustix::net::SocketAddrUnix::new(path.as_str()) {
                Ok(a) => a.into(),
                Err(_) => return Err(syntax("Unix socket path is too long")),
            }
        } else {
            let Some(addr) = get_str_sep(&mut p, 256, Sep::First(b':')) else {
                return Err(syntax("Missing : separator"));
            };
            let mut ip = Ipv4Addr::UNSPECIFIED;
            if !addr.is_empty() {
                ip = inet_aton(&addr).ok_or_else(|| syntax("Bad host address"))?;
            }
            let Some(port) = get_str_sep(&mut p, 256, Sep::First(b'-')) else {
                return Err(syntax("Bad host port separator"));
            };
            let port = match qemu_strtoi(&port, 0, false) {
                Some(v) if (0..=65535).contains(&v) => v as u16,
                _ => return Err(syntax("Bad host port")),
            };
            SocketAddrV4::new(ip, port).into()
        };
        let Some(gaddr) = get_str_sep(&mut p, 256, Sep::First(b':')) else {
            return Err(syntax("Missing guest address"));
        };
        let mut gip = Ipv4Addr::UNSPECIFIED;
        if !gaddr.is_empty() {
            gip = inet_aton(&gaddr).ok_or_else(|| syntax("Bad guest address"))?;
        }
        let gport = match qemu_strtoi(p, 0, false) {
            Some(v) if (1..=65535).contains(&v) => v as u16,
            _ => return Err(syntax("Bad guest port")),
        };
        let guest_addr: rustix::net::SocketAddrAny = SocketAddrV4::new(gip, gport).into();
        if !st.alive {
            return Err(err(format!("Could not set up host forwarding rule '{redir}'")));
        }
        let flags = if is_udp { ffi::HOSTFWD_UDP } else { 0 };
        // SAFETY: the instance is alive and its lock is held by the caller (st); both addresses
        // are valid sockaddrs of the given lengths for the duration of the call.
        let r = unsafe {
            (lib.add_hostxfwd)(
                self.inner.slirp(),
                host_addr.as_ptr().cast(),
                host_addr.addr_len(),
                guest_addr.as_ptr().cast(),
                guest_addr.addr_len(),
                flags,
            )
        };
        if r < 0 {
            return Err(err(format!("Could not set up host forwarding rule '{redir}'")));
        }
        Ok(())
    }

    /// The second half of `hmp_hostfwd_remove()`: the line it prints.
    fn hostfwd_remove(&self, st: &Stack, src: &str) -> String {
        let invalid = || "invalid format".to_string();
        let mut p = src;
        let Some(proto) = get_str_sep(&mut p, 256, Sep::First(b':')) else {
            return invalid();
        };
        let is_udp = match proto.as_str() {
            "tcp" | "" => false,
            "udp" => true,
            _ => return invalid(),
        };
        let Some(addr) = get_str_sep(&mut p, 256, Sep::First(b':')) else {
            return invalid();
        };
        let mut ip = Ipv4Addr::UNSPECIFIED;
        if !addr.is_empty() {
            match inet_aton(&addr) {
                Some(a) => ip = a,
                None => return invalid(),
            }
        }
        let Some(port) = qemu_strtoi(p, 10, true) else {
            return invalid();
        };
        let host_addr: rustix::net::SocketAddrAny = SocketAddrV4::new(ip, port as u16).into();
        let mut r = -1;
        if st.alive {
            let flags = if is_udp { ffi::HOSTFWD_UDP } else { 0 };
            // SAFETY: as in hostfwd().
            r = unsafe {
                (self.inner.lib.remove_hostxfwd)(
                    self.inner.slirp(),
                    host_addr.as_ptr().cast(),
                    host_addr.addr_len(),
                    flags,
                )
            };
        }
        format!("host forwarding rule for {src} {}", if r != 0 { "not found" } else { "removed" })
    }

    /// `slirp_guestfwd()`. The host end is either `cmd:` or a `tcp:host:port` connection;
    /// QEMU takes any chardev there.
    fn guestfwd(&self, st: &mut Stack, config: &str) -> Result<()> {
        let syntax = || err(format!("Invalid guest forwarding rule '{config}'"));
        let conflict =
            || err(format!("Conflicting/invalid host:port in guest forwarding rule '{config}'"));
        let mut p = config;
        let proto = get_str_sep(&mut p, 128, Sep::First(b':')).ok_or_else(syntax)?;
        if proto != "tcp" && !proto.is_empty() {
            return Err(syntax());
        }
        let addr = get_str_sep(&mut p, 128, Sep::First(b':')).ok_or_else(syntax)?;
        let mut server = Ipv4Addr::UNSPECIFIED;
        if !addr.is_empty() {
            server = inet_aton(&addr).ok_or_else(syntax)?;
        }
        let port_str = get_str_sep(&mut p, 128, Sep::First(b'-')).ok_or_else(syntax)?;
        let (port, end) = strtol10(&port_str);
        if end != port_str.len() || !(1..=65535).contains(&port) {
            return Err(syntax());
        }
        let port = port as c_int;
        let mut server_addr = InAddr::from(server);
        let lib = self.inner.lib;
        if let Some(cmd) = p.strip_prefix("cmd:") {
            let cmd = cstring(cmd);
            // SAFETY: the instance is alive and locked (st); the strings and the address are
            // valid for the call, and libslirp copies the command line.
            let r =
                unsafe { (lib.add_exec)(self.inner.slirp(), cmd.as_ptr(), &mut server_addr, port) };
            if r < 0 {
                return Err(conflict());
            }
            return Ok(());
        }
        let label = format!("guestfwd.tcp.{port}");
        let open_failed = || err(format!("Could not open guest forwarding device '{label}'"));
        let Some(target) = p.strip_prefix("tcp:") else {
            return Err(open_failed());
        };
        let stream = match TcpStream::connect(target) {
            Ok(s) => s,
            Err(e) => {
                error_report(&format!("Failed to connect to '{target}': {}", strerror(&e)));
                return Err(open_failed());
            }
        };
        let fwd =
            Box::new(GuestFwd { stream, server: server_addr, port, eof: AtomicBool::new(false) });
        let opaque: *mut c_void = std::ptr::from_ref::<GuestFwd>(&fwd).cast_mut().cast();
        // SAFETY: as for add_exec; opaque is the boxed GuestFwd, which is kept in the stack
        // until slirp_cleanup() and so outlives every call of the write callback.
        let r = unsafe {
            (lib.add_guestfwd)(
                self.inner.slirp(),
                cb_guestfwd_write,
                opaque,
                &mut server_addr,
                port,
            )
        };
        if r < 0 {
            return Err(conflict());
        }
        st.guestfwds.push(fwd);
        Ok(())
    }

    /// `slirp_smb()`.
    fn smb(&self, st: &mut Stack, exported_dir: &str, server: u32) -> Result<()> {
        let Some(user) = user_name() else {
            return Err(err("Failed to retrieve user name"));
        };
        if std::fs::metadata(SMBD_COMMAND).is_err() {
            return Err(err(format!("Could not find '{SMBD_COMMAND}', please install it")));
        }
        let access = rustix::fs::Access::READ_OK | rustix::fs::Access::EXEC_OK;
        if let Err(e) = rustix::fs::access(exported_dir, access) {
            let e = std::io::Error::from(e);
            return Err(err(format!(
                "Error accessing shared directory '{exported_dir}': {}",
                strerror(&e)
            )));
        }
        let Ok(dir) = crate::util::make_temp("qemu-smb.", "", true) else {
            return Err(err("Could not create samba server dir"));
        };
        let d = dir.display().to_string();
        st.smb_dir = Some(dir);
        let smb_conf = format!("{d}/smb.conf");
        let text = format!(
            "[global]\n\
             private dir={d}\n\
             interfaces=127.0.0.1\n\
             bind interfaces only=yes\n\
             pid directory={d}\n\
             lock directory={d}\n\
             state directory={d}\n\
             cache directory={d}\n\
             ncalrpc dir={d}/ncalrpc\n\
             log file={d}/log.smbd\n\
             smb passwd file={d}/smbpasswd\n\
             security = user\n\
             map to guest = Bad User\n\
             load printers = no\n\
             printing = bsd\n\
             disable spoolss = yes\n\
             usershare max shares = 0\n\
             [qemu]\n\
             path={exported_dir}\n\
             read only=no\n\
             guest ok=yes\n\
             force user={user}\n"
        );
        if let Err(e) = std::fs::write(&smb_conf, text) {
            smb_cleanup(st);
            return Err(err(format!(
                "Could not create samba server configuration file '{smb_conf}': {}",
                strerror(&e)
            )));
        }
        let cmdline = cstring(&format!("{SMBD_COMMAND} -l {d} -s {smb_conf}"));
        let mut addr = in_addr(server);
        for port in [139, 445] {
            // SAFETY: as in guestfwd().
            let r = unsafe {
                (self.inner.lib.add_exec)(self.inner.slirp(), cmdline.as_ptr(), &mut addr, port)
            };
            if r < 0 {
                smb_cleanup(st);
                return Err(err("Conflicting/invalid smbserver address"));
            }
        }
        Ok(())
    }

    /// `slirp_connection_info()`.
    fn connection_info(&self, st: &Stack) -> String {
        if !st.alive {
            return String::new();
        }
        // SAFETY: the instance is alive and locked; the result is a g_malloc'd string we own.
        unsafe { ffi::take_gstring((self.inner.lib.connection_info)(self.inner.slirp())) }
    }
}

/// The forwarding rules and the SMB share, set up right after the stack is made.
fn setup(s: &Slirp<'_>, st: &mut Stack, cfg: &Config) -> Result<()> {
    for f in &cfg.fwds {
        match f {
            Fwd::Host(r) => s.hostfwd(st, r)?,
            Fwd::Guest(g) => s.guestfwd(st, g)?,
        }
    }
    if let Some(dir) = &cfg.smb {
        s.smb(st, dir, cfg.smbsrv)?;
    }
    Ok(())
}

/// `net_init_slirp()` and `net_slirp_init()`.
pub(crate) fn net_init_slirp(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::User(user) = &netdev.u else {
        unreachable!("net_init_slirp called for {:?}", netdev.u.tag());
    };
    let cfg = Config::parse(user)?;
    let lib = ffi::lib().map_err(|e| {
        err(format!(
            "network backend 'user' is not compiled into this binary (libslirp could not be \
             loaded: {e})"
        ))
    })?;
    let waker = Waker::new().map_err(|e| {
        err(format!("Could not start the user mode network stack: {}", strerror(&e)))
    })?;
    let inner = Arc::new(Inner {
        lib,
        slirp: AtomicPtr::default(),
        nc: OnceLock::new(),
        pending: Mutex::new(VecDeque::new()),
        delivering: Mutex::new(()),
        timers: Mutex::new(Timers::default()),
        waker,
        clock_base: Instant::now(),
        stack: Mutex::new(Stack::default()),
    });
    let ops = Arc::new(SlirpOps { inner: inner.clone(), thread: Mutex::new(None) });
    let (i, o) = (inner.clone(), ops.clone());
    let nc = net.new_client(NetClientDriver::User, peer.as_ref(), "user", Some(name), move |w| {
        let _ = i.nc.set(w.clone());
        o
    });
    nc.set_info_str(&format!(
        "net={},restrict={}",
        Ipv4Addr::from(cfg.net),
        if cfg.restricted { "on" } else { "off" }
    ));

    let opt = |c: &Option<CString>| c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut dnssearch: Vec<*const c_char> = cfg.dnssearch.iter().map(|c| c.as_ptr()).collect();
    let vdnssearch = if dnssearch.is_empty() {
        std::ptr::null()
    } else {
        dnssearch.push(std::ptr::null());
        dnssearch.as_ptr()
    };
    let scfg = SlirpConfig {
        version: if lib.at_least(4, 9) {
            6
        } else if lib.at_least(4, 7) {
            4
        } else {
            1
        },
        restricted: c_int::from(cfg.restricted),
        in_enabled: cfg.ipv4,
        vnetwork: in_addr(cfg.net),
        vnetmask: in_addr(cfg.mask),
        vhost: in_addr(cfg.host),
        in6_enabled: cfg.ipv6,
        vprefix_addr6: cfg.prefix6.into(),
        vprefix_len: cfg.prefix6_len as u8,
        vhost6: cfg.host6.into(),
        vhostname: opt(&cfg.hostname),
        tftp_server_name: opt(&cfg.tftp_server_name),
        tftp_path: opt(&cfg.tftp),
        bootfile: opt(&cfg.bootfile),
        vdhcp_start: in_addr(cfg.dhcp),
        vnameserver: in_addr(cfg.dns),
        vnameserver6: cfg.dns6.into(),
        vdnssearch,
        vdomainname: opt(&cfg.domainname),
        if_mtu: 0,
        if_mru: 0,
        disable_host_loopback: false,
        enable_emu: false,
        outbound_addr: std::ptr::null(),
        outbound_addr6: std::ptr::null(),
        disable_dns: false,
        disable_dhcp: false,
        mfr_id: 0,
        oob_eth_addr: [0; 6],
    };

    let r = {
        let mut st = lock(&inner.stack);
        let opaque: *mut c_void = Arc::as_ptr(&inner).cast_mut().cast();
        // SAFETY: the config and the strings it points to live until slirp_new() returns, and
        // libslirp copies what it keeps. CALLBACKS is static, and opaque is the Inner that the
        // callbacks expect, kept alive by the client until slirp_cleanup().
        let slirp = unsafe { (lib.new)(&scfg, &CALLBACKS, opaque) };
        if slirp.is_null() {
            Err(err("Could not start the user mode network stack"))
        } else {
            inner.slirp.store(slirp, Ordering::SeqCst);
            st.alive = true;
            setup(&Slirp { inner: &inner }, &mut st, &cfg)
        }
    };
    let r = r.and_then(|()| {
        let i = inner.clone();
        std::thread::Builder::new().name("net-user".into()).spawn(move || run(i)).map_err(|e| {
            err(format!("Could not start the user mode network stack: {}", strerror(&e)))
        })
    });
    match r {
        Ok(h) => {
            *lock(&ops.thread) = Some(h);
            Ok(())
        }
        Err(e) => {
            net.del_client(&nc);
            Err(e)
        }
    }
}

fn ops_of(nc: &NetClient) -> Option<&SlirpOps> {
    nc.ops().as_any()?.downcast_ref::<SlirpOps>()
}

/// `slirp_lookup()`: the backend called `id`, or the first one.
fn lookup(net: &Net, id: Option<&str>) -> Result<Arc<NetClient>> {
    match id {
        Some(id) => {
            let nc =
                net.find_netdev(id).ok_or_else(|| err(format!("unrecognized netdev id '{id}'")))?;
            if nc.model() != "user" || ops_of(&nc).is_none() {
                return Err(err("invalid device specified"));
            }
            Ok(nc)
        }
        None => net
            .clients()
            .iter()
            .find(|c| ops_of(c).is_some())
            .cloned()
            .ok_or_else(|| err("user mode network stack not in use")),
    }
}

/// `hmp_hostfwd_add()`.
pub(crate) fn hostfwd_add(net: &Net, id: Option<&str>, redir: &str) -> Result<()> {
    let nc = lookup(net, id)?;
    let ops = ops_of(&nc).expect("lookup checked the type");
    let st = lock(&ops.inner.stack);
    Slirp { inner: &ops.inner }.hostfwd(&st, redir)
}

/// `hmp_hostfwd_remove()`.
pub(crate) fn hostfwd_remove(net: &Net, id: Option<&str>, src: &str) -> Result<String> {
    let nc = lookup(net, id)?;
    let ops = ops_of(&nc).expect("lookup checked the type");
    let st = lock(&ops.inner.stack);
    Ok(Slirp { inner: &ops.inner }.hostfwd_remove(&st, src))
}

/// `hmp_info_usernet()`.
pub(crate) fn info_usernet(net: &Net) -> String {
    let mut out = String::new();
    for nc in net.clients() {
        let Some(ops) = ops_of(nc) else {
            continue;
        };
        let info = {
            let st = lock(&ops.inner.stack);
            Slirp { inner: &ops.inner }.connection_info(&st)
        };
        let id = hub_id_for_client(nc).unwrap_or(-1);
        out.push_str(&format!("Hub {id} ({}):\n{info}", nc.name()));
    }
    out
}

/// The version of the libslirp `-netdev user` uses, loading it if that has not happened yet,
/// or why it cannot be loaded. The library is looked up under the name in `RUVM_LIBSLIRP` first,
/// then under the usual names.
pub fn libslirp_version() -> std::result::Result<String, String> {
    ffi::lib().map(|l| l.version.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn str_sep() {
        let mut p = "tcp:127.0.0.1:22-:22";
        assert_eq!(get_str_sep(&mut p, 256, Sep::First(b':')).as_deref(), Some("tcp"));
        assert_eq!(get_str_sep(&mut p, 256, Sep::First(b':')).as_deref(), Some("127.0.0.1"));
        assert_eq!(get_str_sep(&mut p, 4, Sep::First(b'-')).as_deref(), Some("22"));
        assert_eq!(p, ":22");
        let mut p = "/a-b/s-c:1";
        assert_eq!(get_str_sep(&mut p, 256, Sep::Last(b'-')).as_deref(), Some("/a-b/s"));
        assert_eq!(get_str_sep(&mut p, 3, Sep::First(b'x')), None);
        let mut p = "abcdef/1";
        assert_eq!(get_str_sep(&mut p, 4, Sep::First(b'/')).as_deref(), Some("abc"));
    }

    #[test]
    fn numbers() {
        assert_eq!(strtol10("24"), (24, 2));
        assert_eq!(strtol10("24x"), (24, 2));
        assert_eq!(strtol10(""), (0, 0));
        assert_eq!(strtol10(" -3"), (-3, 3));
        assert_eq!(qemu_strtoi("0x10", 0, false), Some(16));
        assert_eq!(qemu_strtoi("22abc", 0, false), Some(22));
        assert_eq!(qemu_strtoi("22abc", 10, true), None);
        assert_eq!(qemu_strtoi("", 0, false), None);
    }

    #[test]
    fn ipv6_nets() {
        let p: Ipv6Addr = "fec0::".parse().unwrap();
        assert!(in6_equal_net(&p, &"fec0::2".parse().unwrap(), 64));
        assert!(!in6_equal_net(&p, &"fec1::2".parse().unwrap(), 64));
        assert!(in6_equal_net(&p, &"fec1::2".parse().unwrap(), 15));
        assert!(!in6_equal_net(&p, &"fec1::2".parse().unwrap(), 16));
    }
}
