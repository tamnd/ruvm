// SPDX-License-Identifier: GPL-2.0-or-later

//! The tap backend, net/tap.c.
//!
//! Frames go to and from a tap interface, one per read or write. With `vnet_hdr` the kernel
//! puts a virtio-net header in front of each frame. A NIC that understands those headers turns
//! them on with [`NetClient::set_vnet_hdr_len`]; otherwise the backend adds and strips an empty
//! one itself.
//!
//! Making tap interfaces only works on Linux. `fd=` and `fds=` work anywhere, which is what the
//! tests use on other hosts.

use std::io::IoSlice;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::process::ExitStatusExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevTapOptions, NetdevU};

use crate::client::{
    NET_BUFSIZE, NetClient, NetClientOps, NetOffloads, VNET_HDR_LEN, eth_pad_short_frame, lock,
};
use crate::net::Net;
use crate::poll::{Interest, IoHandler, IoThread, set_nonblocking};

#[cfg(target_os = "linux")]
use crate::tap_linux as host;

/// What other hosts do: what tap-bsd.c and tap-stub.c do for descriptors that were passed in,
/// and a plain error for making an interface.
#[cfg(not(target_os = "linux"))]
mod host {
    use std::io;
    use std::os::fd::{AsRawFd, OwnedFd};

    use ruvm_base::{Error, Result};

    use crate::client::NetOffloads;

    pub(crate) fn tap_open(
        _ifname: &mut String,
        _vnet_hdr: &mut bool,
        _vnet_hdr_required: bool,
        _mq_required: bool,
    ) -> Result<OwnedFd> {
        Err(Error::generic("tap interfaces can only be created on Linux hosts; use fd= instead"))
    }

    pub(crate) fn set_sndbuf(_fd: &impl AsRawFd, _sndbuf: i32) -> Result<()> {
        Ok(())
    }

    pub(crate) fn probe_vnet_hdr(_fd: &impl AsRawFd) -> Result<bool> {
        Ok(false)
    }

    pub(crate) fn probe_has_ufo(_fd: &impl AsRawFd) -> bool {
        false
    }

    pub(crate) fn probe_has_uso(_fd: &impl AsRawFd) -> bool {
        false
    }

    pub(crate) fn probe_has_tunnel(_fd: &impl AsRawFd) -> bool {
        false
    }

    pub(crate) fn set_vnet_hdr_len(_fd: &impl AsRawFd, _len: usize) {}

    pub(crate) fn set_vnet_le(_fd: &impl AsRawFd, _on: bool) -> io::Result<()> {
        Err(io::Error::from_raw_os_error(libc::EINVAL))
    }

    pub(crate) fn set_vnet_be(_fd: &impl AsRawFd, _on: bool) -> io::Result<()> {
        Err(io::Error::from_raw_os_error(libc::EINVAL))
    }

    pub(crate) fn set_offload(_fd: &impl AsRawFd, _ol: &NetOffloads) {}

    pub(crate) fn fd_enable(_fd: &impl AsRawFd) -> io::Result<()> {
        Ok(())
    }

    pub(crate) fn fd_disable(_fd: &impl AsRawFd) -> io::Result<()> {
        Ok(())
    }

    pub(crate) fn get_ifname(_fd: &impl AsRawFd) -> io::Result<String> {
        Err(io::Error::from_raw_os_error(libc::ENOTSUP))
    }
}

/// `DEFAULT_NETWORK_SCRIPT`.
pub const DEFAULT_NETWORK_SCRIPT: &str = "/etc/qemu-ifup";
/// `DEFAULT_NETWORK_DOWN_SCRIPT`.
pub const DEFAULT_NETWORK_DOWN_SCRIPT: &str = "/etc/qemu-ifdown";

/// The most packets one wakeup reads before letting others run.
const SEND_BATCH: usize = 50;

/// `TAPState`.
#[derive(Debug)]
pub struct TapState {
    this: Weak<TapState>,
    nc: Weak<NetClient>,
    fd: Mutex<Option<Arc<OwnedFd>>>,
    raw_fd: RawFd,
    host_vnet_hdr_len: AtomicUsize,
    using_vnet_hdr: AtomicBool,
    has_ufo: bool,
    has_uso: bool,
    has_tunnel: bool,
    enabled: AtomicBool,
    read_poll: AtomicBool,
    write_poll: AtomicBool,
    down_script: Mutex<Option<(String, String)>>,
    vhost: AtomicBool,
    vhostfd: Mutex<Option<OwnedFd>>,
    io: OnceLock<IoThread>,
}

impl TapState {
    fn fd(&self) -> Option<Arc<OwnedFd>> {
        lock(&self.fd).clone()
    }

    fn wake(&self) {
        if let Some(io) = self.io.get() {
            io.wake();
        }
    }

    /// `tap_read_poll()`.
    fn read_poll(&self, enable: bool) {
        self.read_poll.store(enable, Ordering::SeqCst);
        self.wake();
    }

    /// `tap_write_poll()`.
    fn write_poll(&self, enable: bool) {
        self.write_poll.store(enable, Ordering::SeqCst);
        self.wake();
    }

    /// The descriptor number, `tap_get_fd()`.
    pub fn raw_fd(&self) -> RawFd {
        self.raw_fd
    }

    /// Whether vhost-net was asked for. It is only recorded; frames always go through here.
    pub fn vhost_requested(&self) -> bool {
        self.vhost.load(Ordering::Relaxed)
    }

    /// `tap_enable()`.
    pub fn enable(&self) -> std::io::Result<()> {
        if self.enabled.load(Ordering::SeqCst) {
            return Ok(());
        }
        let fd = self.fd().ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
        host::fd_enable(fd.as_ref())?;
        self.enabled.store(true, Ordering::SeqCst);
        self.wake();
        Ok(())
    }

    /// `tap_disable()`.
    pub fn disable(&self) -> std::io::Result<()> {
        if !self.enabled.load(Ordering::SeqCst) {
            return Ok(());
        }
        let fd = self.fd().ok_or_else(|| std::io::Error::from_raw_os_error(libc::EBADF))?;
        host::fd_disable(fd.as_ref())?;
        if let Some(nc) = self.nc.upgrade() {
            nc.purge_queued_packets();
        }
        self.enabled.store(false, Ordering::SeqCst);
        self.wake();
        Ok(())
    }

    /// `tap_send()`.
    fn send(&self, fd: &OwnedFd) {
        let Some(nc) = self.nc.upgrade() else {
            return;
        };
        let mut buf = vec![0u8; NET_BUFSIZE];
        for _ in 0..SEND_BATCH {
            let size = match rustix::io::read(fd, &mut buf) {
                Ok(n) if n > 0 => n,
                _ => break,
            };
            let hdr_len = self.host_vnet_hdr_len.load(Ordering::Relaxed);
            if hdr_len != 0 && size <= hdr_len {
                break;
            }
            let mut pkt = &buf[..size];
            if hdr_len != 0 && !self.using_vnet_hdr.load(Ordering::Relaxed) {
                pkt = &pkt[hdr_len..];
            }
            let padded;
            if nc.peer_needs_padding() {
                if let Some(p) = eth_pad_short_frame(pkt) {
                    padded = p;
                    pkt = &padded;
                }
            }
            let me = self.this.clone();
            let ret = nc.send_packet_async(
                pkt,
                Some(Box::new(move |_| {
                    if let Some(s) = me.upgrade() {
                        s.read_poll(true);
                    }
                })),
            );
            if ret == 0 {
                self.read_poll.store(false, Ordering::SeqCst);
                break;
            } else if ret < 0 {
                break;
            }
        }
    }

    /// `tap_exit_notify()`: runs the down script once.
    fn run_down_script(&self) {
        let script = lock(&self.down_script).take();
        if let Some((script, ifname)) = script {
            if let Err(e) = launch_script(&script, &ifname) {
                error_report(e.message());
            }
        }
    }
}

impl IoHandler for TapState {
    fn interest(&self) -> Interest {
        let enabled = self.enabled.load(Ordering::SeqCst);
        Interest {
            fd: self.fd(),
            read: enabled && self.read_poll.load(Ordering::SeqCst),
            write: enabled && self.write_poll.load(Ordering::SeqCst),
            timeout: None,
        }
    }

    fn readable(&self, fd: &Arc<OwnedFd>) {
        self.send(fd);
    }

    /// `tap_writable()`.
    fn writable(&self, _fd: &Arc<OwnedFd>) {
        self.write_poll.store(false, Ordering::SeqCst);
        if let Some(nc) = self.nc.upgrade() {
            nc.flush_queued_packets();
        }
    }
}

impl NetClientOps for TapState {
    /// `tap_receive_iov()`.
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        let Some(fd) = self.fd() else {
            return -(libc::EBADF as isize);
        };
        let hdr = [0u8; crate::client::VNET_HDR_V1_HASH_TUNNEL_LEN];
        let hdr_len = self.host_vnet_hdr_len.load(Ordering::Relaxed);
        let mut slices = Vec::with_capacity(iov.len() + 1);
        if hdr_len != 0 && !self.using_vnet_hdr.load(Ordering::Relaxed) {
            slices.push(IoSlice::new(&hdr[..hdr_len]));
        }
        slices.extend(iov.iter().map(|b| IoSlice::new(b)));
        loop {
            match rustix::io::writev(fd.as_ref(), &slices) {
                Ok(n) => return n as isize,
                Err(rustix::io::Errno::INTR) => continue,
                Err(rustix::io::Errno::AGAIN) => {
                    self.write_poll(true);
                    return 0;
                }
                Err(e) => return -(e.raw_os_error() as isize),
            }
        }
    }

    /// `tap_cleanup()`.
    fn cleanup(&self, nc: &NetClient) {
        nc.purge_queued_packets();
        self.run_down_script();
        self.read_poll.store(false, Ordering::SeqCst);
        self.write_poll.store(false, Ordering::SeqCst);
        if let Some(io) = self.io.get() {
            io.stop();
        }
        lock(&self.fd).take();
        lock(&self.vhostfd).take();
    }

    /// `tap_poll()`.
    fn poll(&self, _nc: &NetClient, enable: bool) {
        self.read_poll(enable);
        self.write_poll(enable);
    }

    fn has_ufo(&self, _nc: &NetClient) -> bool {
        self.has_ufo
    }

    fn has_uso(&self, _nc: &NetClient) -> bool {
        self.has_uso
    }

    fn has_tunnel(&self, _nc: &NetClient) -> bool {
        self.has_tunnel
    }

    fn has_vnet_hdr(&self, _nc: &NetClient) -> bool {
        self.host_vnet_hdr_len.load(Ordering::Relaxed) != 0
    }

    fn has_vnet_hdr_len(&self, _nc: &NetClient, _len: usize) -> bool {
        self.host_vnet_hdr_len.load(Ordering::Relaxed) != 0
    }

    /// `tap_set_vnet_hdr_len()`.
    fn set_vnet_hdr_len(&self, _nc: &NetClient, len: usize) -> bool {
        if let Some(fd) = self.fd() {
            host::set_vnet_hdr_len(fd.as_ref(), len);
        }
        self.host_vnet_hdr_len.store(len, Ordering::Relaxed);
        self.using_vnet_hdr.store(true, Ordering::Relaxed);
        true
    }

    fn set_offload(&self, _nc: &NetClient, ol: &NetOffloads) {
        if let Some(fd) = self.fd() {
            host::set_offload(fd.as_ref(), ol);
        }
    }

    fn set_vnet_le(&self, _nc: &NetClient, is_le: bool) -> Option<std::io::Result<()>> {
        Some(self.fd().map_or(Ok(()), |fd| host::set_vnet_le(fd.as_ref(), is_le)))
    }

    fn set_vnet_be(&self, _nc: &NetClient, is_be: bool) -> Option<std::io::Result<()>> {
        Some(self.fd().map_or(Ok(()), |fd| host::set_vnet_be(fd.as_ref(), is_be)))
    }
}

/// `launch_script()`: runs `script ifname` and waits for it.
fn launch_script(script: &str, ifname: &str) -> Result<()> {
    let status = match std::process::Command::new(script).arg(ifname).status() {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::OutOfMemory => {
            return Err(Error::from_io(format!("could not launch network script {script}"), e));
        }
        // The exec failing in QEMU's child makes it exit with 1.
        Err(_) => std::process::ExitStatus::from_raw(1 << 8),
    };
    if status.success() {
        return Ok(());
    }
    Err(Error::generic(format!("network script {script} failed with status {}", status.into_raw())))
}

/// `tap_parse_script()`.
fn parse_script(arg: Option<&str>, default: &str) -> Option<String> {
    let s = arg.unwrap_or(default);
    if s.is_empty() || s == "no" { None } else { Some(s.to_string()) }
}

/// Makes the client of one queue, `net_tap_fd_init()`.
fn fd_init(
    net: &mut Net,
    peer: Option<&Arc<NetClient>>,
    model: &str,
    name: &str,
    fd: OwnedFd,
    vnet_hdr: bool,
) -> Result<(Arc<NetClient>, Arc<TapState>)> {
    let raw_fd = fd.as_raw_fd();
    let has_ufo = host::probe_has_ufo(&fd);
    let has_uso = host::probe_has_uso(&fd);
    let has_tunnel = host::probe_has_tunnel(&fd);
    let fd = Arc::new(fd);
    let mut state = None;
    let nc = net.new_client(NetClientDriver::Tap, peer, model, Some(name), |w| {
        let s = Arc::new_cyclic(|this| TapState {
            this: this.clone(),
            nc: w.clone(),
            fd: Mutex::new(Some(fd.clone())),
            raw_fd,
            host_vnet_hdr_len: AtomicUsize::new(if vnet_hdr { VNET_HDR_LEN } else { 0 }),
            using_vnet_hdr: AtomicBool::new(false),
            has_ufo,
            has_uso,
            has_tunnel,
            enabled: AtomicBool::new(true),
            read_poll: AtomicBool::new(false),
            write_poll: AtomicBool::new(false),
            down_script: Mutex::new(None),
            vhost: AtomicBool::new(false),
            vhostfd: Mutex::new(None),
            io: OnceLock::new(),
        });
        state = Some(s.clone());
        s
    });
    let s = state.expect("make_ops ran");
    host::set_offload(fd.as_ref(), &NetOffloads::default());
    if vnet_hdr {
        host::set_vnet_hdr_len(fd.as_ref(), VNET_HDR_LEN);
    }
    s.read_poll.store(true, Ordering::SeqCst);
    match IoThread::spawn(name, s.clone()) {
        Ok(io) => {
            let _ = s.io.set(io);
        }
        Err(e) => {
            net.del_client(&nc);
            return Err(Error::from_io("could not start the tap I/O thread", e));
        }
    }
    Ok((nc, s))
}

/// `net_init_tap_one()`.
#[allow(clippy::too_many_arguments)]
fn init_one(
    net: &mut Net,
    tap: &NetdevTapOptions,
    peer: Option<&Arc<NetClient>>,
    name: &str,
    ifname: &str,
    script: Option<&str>,
    downscript: Option<&str>,
    vhostfd: Option<OwnedFd>,
    vnet_hdr: bool,
    fd: OwnedFd,
) -> Result<()> {
    let raw = fd.as_raw_fd();
    let model = if tap.helper.is_some() { "bridge" } else { "tap" };
    let (nc, s) = fd_init(net, peer, model, name, fd, vnet_hdr)?;
    let sndbuf_required = tap.sndbuf.is_some();
    let sndbuf = match tap.sndbuf {
        Some(v) if v != 0 => v.min(i32::MAX as u64) as i32,
        _ => i32::MAX,
    };
    if let Some(fd) = s.fd() {
        if let Err(e) = host::set_sndbuf(fd.as_ref(), sndbuf) {
            if sndbuf_required {
                net.del_client(&nc);
                return Err(e);
            }
        }
    }
    if tap.fd.is_some() || tap.fds.is_some() {
        nc.set_info_str(&format!("fd={raw}"));
    } else if let Some(helper) = &tap.helper {
        nc.set_info_str(&format!("helper={helper}"));
    } else {
        nc.set_info_str(&format!(
            "ifname={ifname},script={},downscript={}",
            script.unwrap_or("no"),
            downscript.unwrap_or("no")
        ));
        if let Some(d) = downscript {
            *lock(&s.down_script) = Some((d.to_string(), ifname.to_string()));
        }
    }
    let vhost = match tap.vhost {
        Some(v) => v,
        None => vhostfd.is_some() || tap.vhostforce == Some(true),
    };
    s.vhost.store(vhost, Ordering::Relaxed);
    *lock(&s.vhostfd) = vhostfd;
    Ok(())
}

/// Adopts descriptors given by number and makes them non-blocking, `unblock_fds()`.
fn unblock_fds(fds: Vec<RawFd>) -> Result<Vec<OwnedFd>> {
    let mut out = Vec::with_capacity(fds.len());
    let mut err = None;
    for fd in fds {
        match crate::fd::adopt(fd) {
            Ok(owned) => {
                if err.is_none() {
                    if let Err(e) = set_nonblocking(&owned) {
                        err = Some(crate::poll::nonblocking_error(fd, &e));
                    }
                }
                out.push(owned);
            }
            Err(e) => {
                if err.is_none() {
                    err = Some(crate::poll::nonblocking_error(fd, &e));
                }
            }
        }
    }
    match err {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// `net_init_tap()`.
pub(crate) fn net_init_tap(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::Tap(tap) = &netdev.u else {
        unreachable!("tap init with another type");
    };
    if tap.vhost == Some(false) && (tap.vhostfds.is_some() || tap.vhostfd.is_some()) {
        return Err(Error::generic("vhostfd(s)= is not valid without vhost"));
    }
    let exclusive = usize::from(tap.queues.is_some())
        + usize::from(tap.helper.is_some())
        + usize::from(tap.fds.is_some())
        + usize::from(tap.fd.is_some());
    if exclusive > 1 {
        return Err(Error::generic("queues=, helper=, fds= and fd= are mutual exclusive"));
    }
    if (tap.fd.is_some() || tap.fds.is_some() || tap.helper.is_some())
        && (tap.ifname.is_some()
            || tap.script.is_some()
            || tap.downscript.is_some()
            || tap.vnet_hdr.is_some())
    {
        return Err(Error::generic(
            "ifname=, script=, downscript=, vnet_hdr= are invalid with fd=/fds=/helper=",
        ));
    }

    // tap_parse_fds_and_queues()
    let mut queues = 1usize;
    let mut fds: Option<Vec<OwnedFd>> = None;
    if let Some(q) = tap.queues {
        if q > i32::MAX as u32 {
            return Err(Error::generic(format!("queues exceeds maximum {}", i32::MAX)));
        }
        if q == 0 {
            return Err(Error::generic("queues must be greater than zero"));
        }
        queues = q as usize;
    } else if let Some(param) = tap.fd.as_deref().or(tap.fds.as_deref()) {
        let raw = net.parse_fds(param, if tap.fd.is_some() { 1 } else { 0 })?;
        queues = raw.len();
        fds = Some(unblock_fds(raw)?);
    } else if tap.helper.is_some() {
        return Err(Error::generic("the bridge helper (helper=) is not supported"));
    }

    if peer.is_some() && queues > 1 {
        return Err(Error::generic("Multiqueue tap cannot be used with hubs"));
    }

    // tap_parse_vhost_fds()
    let mut vhost_fds: Vec<Option<OwnedFd>> = Vec::new();
    if let Some(param) = tap.vhostfd.as_deref().or(tap.vhostfds.as_deref()) {
        let raw = net.parse_fds(param, queues)?;
        vhost_fds = unblock_fds(raw)?.into_iter().map(Some).collect();
    }
    vhost_fds.resize_with(queues, || None);
    let mut vhost_fds = vhost_fds.into_iter();

    if let Some(fds) = fds {
        let mut vnet_hdr = false;
        for (i, fd) in fds.into_iter().enumerate() {
            if i == 0 {
                vnet_hdr = host::probe_vnet_hdr(&fd)?;
            } else if host::probe_vnet_hdr(&fd).ok() != Some(vnet_hdr) {
                return Err(Error::generic("vnet_hdr not consistent across given tap fds"));
            }
            init_one(
                net,
                tap,
                peer.as_ref(),
                name,
                "",
                None,
                None,
                vhost_fds.next().flatten(),
                vnet_hdr,
                fd,
            )?;
        }
        return Ok(());
    }

    let script = parse_script(tap.script.as_deref(), DEFAULT_NETWORK_SCRIPT);
    let downscript = parse_script(tap.downscript.as_deref(), DEFAULT_NETWORK_DOWN_SCRIPT);
    let mut ifname = tap.ifname.clone().unwrap_or_default();
    for i in 0..queues {
        let first = i == 0;
        // net_tap_init()
        let (mut vnet_hdr, required) = match tap.vnet_hdr {
            Some(v) => (v, v),
            None => (true, false),
        };
        let fd = host::tap_open(&mut ifname, &mut vnet_hdr, required, queues > 1)?;
        if first {
            if let Some(s) = &script {
                launch_script(s, &ifname)?;
            }
        }
        if queues > 1 && first && tap.ifname.is_none() {
            ifname = host::get_ifname(&fd).map_err(|_| Error::generic("Fail to get ifname"))?;
        }
        init_one(
            net,
            tap,
            peer.as_ref(),
            name,
            &ifname,
            if first { script.as_deref() } else { None },
            if first { downscript.as_deref() } else { None },
            vhost_fds.next().flatten(),
            vnet_hdr,
            fd,
        )?;
    }
    Ok(())
}
