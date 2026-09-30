// SPDX-License-Identifier: GPL-2.0-or-later

//! Helpers the integration tests share: a NIC that records what reaches it, and waiting for
//! things that happen on I/O threads.

#![allow(dead_code, unreachable_pub)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use ruvm_net::{MacAddr, Net, NetClient, NetClientOps, Nic, NicConf};

/// A receiver that keeps every frame, and can be told to refuse them.
#[derive(Debug, Default)]
pub struct Recorder {
    pub frames: Mutex<Vec<Vec<u8>>>,
    cv: Condvar,
    /// `can_receive` answers false while this is set.
    pub closed: AtomicBool,
    /// `receive` answers 0 (full) while this is set.
    pub full: AtomicBool,
    /// `set_vnet_hdr_len` is refused while this is set.
    pub no_vnet_hdr: AtomicBool,
    pub link_changes: AtomicUsize,
}

impl Recorder {
    pub fn new() -> Arc<Self> {
        Arc::new(Recorder::default())
    }

    pub fn take(&self) -> Vec<Vec<u8>> {
        std::mem::take(&mut *self.frames.lock().unwrap())
    }

    pub fn count(&self) -> usize {
        self.frames.lock().unwrap().len()
    }

    /// Waits until `n` frames arrived, and returns them.
    pub fn wait_for(&self, n: usize) -> Vec<Vec<u8>> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut g = self.frames.lock().unwrap();
        while g.len() < n {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(!left.is_zero(), "timed out waiting for {n} frames, got {}", g.len());
            g = self.cv.wait_timeout(g, left).unwrap().0;
        }
        std::mem::take(&mut *g)
    }
}

impl NetClientOps for Recorder {
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        if self.full.load(Ordering::SeqCst) {
            return 0;
        }
        let data = iov.concat();
        let len = data.len() as isize;
        self.frames.lock().unwrap().push(data);
        self.cv.notify_all();
        len
    }

    fn can_receive(&self, _nc: &NetClient) -> bool {
        !self.closed.load(Ordering::SeqCst)
    }

    fn link_status_changed(&self, _nc: &NetClient) {
        self.link_changes.fetch_add(1, Ordering::SeqCst);
    }

    fn set_vnet_hdr_len(&self, _nc: &NetClient, _len: usize) -> bool {
        !self.no_vnet_hdr.load(Ordering::SeqCst)
    }
}

/// Plugs a recording NIC into the backend called `id`.
pub fn attach_nic(net: &mut Net, id: &str) -> (Arc<Nic>, Arc<Recorder>) {
    let peer = net.find_netdev(id).unwrap_or_else(|| panic!("no netdev {id}"));
    let rec = Recorder::new();
    let conf = NicConf { macaddr: MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]), peers: vec![peer] };
    let r = rec.clone();
    let nic = net.new_nic(&conf, "test-nic", Some(&format!("nic-{id}")), move |_, _| r.clone());
    (nic, rec)
}

/// Makes a backend-like client with a recorder behind it, not joined to anything.
pub fn plain_client(net: &mut Net, name: &str) -> (Arc<NetClient>, Arc<Recorder>) {
    let rec = Recorder::new();
    let r = rec.clone();
    let nc =
        net.new_client(ruvm_qapi::types::NetClientDriver::Socket, None, "test", Some(name), |_| r);
    (nc, rec)
}

/// Polls `f` until it holds or ten seconds pass.
pub fn wait_until(what: &str, mut f: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !f() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The info string of client `name`.
pub fn info(net: &Net, name: &str) -> String {
    net.find_netdev(name).map(|c| c.info_str()).unwrap_or_default()
}

/// A TCP port nobody listens on right now.
pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A UDP port nobody uses right now.
pub fn free_udp_port() -> u16 {
    std::net::UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

/// A fresh directory for socket files, short enough for `sun_path`.
pub fn tmpdir(tag: &str) -> std::path::PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::SeqCst);
    let d = std::path::PathBuf::from(format!("/tmp/rn-{}-{tag}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Creates a backend from `-netdev` style options right away, as `netdev_add` does, and
/// returns the error message if that fails.
pub fn netdev(net: &mut Net, opts: &str) -> Result<(), String> {
    net.netdev_add_opts(opts).map_err(|e| e.message().to_string())
}

/// `netdev` on a fresh `Net`, for the error message.
pub fn netdev_err(opts: &str) -> String {
    let mut net = Net::new();
    match netdev(&mut net, opts) {
        Ok(()) => panic!("-netdev {opts} should have failed"),
        Err(e) => e,
    }
}

/// A frame of `len` bytes that says which test sent it.
pub fn frame(tag: u8, len: usize) -> Vec<u8> {
    (0..len).map(|i| tag.wrapping_add(i as u8)).collect()
}
