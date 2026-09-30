// SPDX-License-Identifier: GPL-2.0-or-later

//! The client list and the command line and monitor side of net/net.c.
//!
//! [`Net`] owns what QEMU keeps in globals: the list of clients, the hubs, the NIC table the
//! old `-net nic` and `-nic` options fill, and the `netdev`, `nic` and `net` option lists.

use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::BuildHasher;
#[cfg(unix)]
use std::os::fd::{IntoRawFd, OwnedFd, RawFd};
#[cfg(windows)]
use std::os::windows::io::OwnedSocket as OwnedFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_base::error::ErrorClass;
use ruvm_base::{Error, Result, warn_report};
use ruvm_qapi::cutils::id_wellformed;
use ruvm_qapi::opts::{QemuOpts, QemuOptsList, is_help_option};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevHubPortOptions, NetdevU};
use ruvm_qapi::visit::QObjectInputVisitor;
use ruvm_qapi::visit::Visit;

use crate::client::{MAX_QUEUE_NUM, NetClient, NetClientOps, lock};
use crate::hub::{Hub, HubPortOps, hub_id_for_client};
use crate::opts_visitor::OptsVisitor;
use crate::util::{MacAddr, parse_macaddr};

/// `MAX_NICS`: the size of the table `-net nic` and `-nic` fill.
pub const MAX_NICS: usize = 8;

/// `DEV_NVECTORS_UNSPECIFIED`.
pub const DEV_NVECTORS_UNSPECIFIED: i32 = -1;

/// The backends `-netdev help` lists, in QEMU's order.
#[cfg(target_os = "linux")]
pub const AVAILABLE_NETDEVS: &[&str] = &[
    "socket",
    "stream",
    "dgram",
    "hubport",
    "tap",
    "passt",
    #[cfg(feature = "slirp")]
    "user",
    "vhost-user",
];
/// The backends `-netdev help` lists, in QEMU's order.
#[cfg(all(unix, not(target_os = "linux")))]
pub const AVAILABLE_NETDEVS: &[&str] = &[
    "socket",
    "stream",
    "dgram",
    "hubport",
    "tap",
    #[cfg(feature = "slirp")]
    "user",
    "vhost-user",
];
/// The backends `-netdev help` lists, in QEMU's order.
#[cfg(not(unix))]
pub const AVAILABLE_NETDEVS: &[&str] = &["socket", "stream", "dgram", "hubport", "tap"];

/// A descriptor number as `fd=` gives it. Windows has no such numbers, but the parsing is the same.
#[cfg(windows)]
type RawFd = i32;

#[cfg(unix)]
fn into_raw(fd: OwnedFd) -> RawFd {
    fd.into_raw_fd()
}

#[cfg(windows)]
fn into_raw(fd: OwnedFd) -> RawFd {
    use std::os::windows::io::IntoRawSocket;
    fd.into_raw_socket() as RawFd
}

/// Looks up a descriptor the monitor was given under a name, `monitor_get_fd()`.
pub type FdResolver = Box<dyn FnMut(&str) -> Result<OwnedFd> + Send>;

/// `NICInfo`: one NIC asked for with `-net nic` or `-nic`, for the machine to create.
#[derive(Clone, Debug, Default)]
pub struct NicInfo {
    pub macaddr: MacAddr,
    pub model: Option<String>,
    pub name: Option<String>,
    pub devaddr: Option<String>,
    pub netdev: Option<Arc<NetClient>>,
    pub used: bool,
    pub instantiated: bool,
    pub nvectors: i32,
}

/// `NICConf`: what a NIC model is configured with.
#[derive(Clone, Debug, Default)]
pub struct NicConf {
    pub macaddr: MacAddr,
    /// One backend per queue. Empty means one queue with no peer.
    pub peers: Vec<Arc<NetClient>>,
}

/// `NICState`: a NIC model's queues.
pub struct Nic {
    queues: Vec<Arc<NetClient>>,
    macaddr: MacAddr,
    peer_deleted: AtomicBool,
    /// Backends deleted while this NIC was their peer. QEMU hands them to the NIC, which frees
    /// them together with itself.
    deleted_peers: Mutex<Vec<Arc<NetClient>>>,
}

impl fmt::Debug for Nic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nic")
            .field("name", &self.queues[0].name())
            .field("queues", &self.queues.len())
            .field("macaddr", &self.macaddr.to_string())
            .finish()
    }
}

impl Nic {
    /// `qemu_get_queue()`.
    pub fn queue(&self) -> &Arc<NetClient> {
        &self.queues[0]
    }

    /// `qemu_get_subqueue()`.
    pub fn subqueue(&self, index: usize) -> &Arc<NetClient> {
        &self.queues[index]
    }

    pub fn queues(&self) -> &[Arc<NetClient>] {
        &self.queues
    }

    pub fn macaddr(&self) -> MacAddr {
        self.macaddr
    }

    /// Whether the backend was deleted under the NIC, `peer_deleted`.
    pub fn peer_deleted(&self) -> bool {
        self.peer_deleted.load(Ordering::SeqCst)
    }

    /// `qemu_format_nic_info_str()`.
    pub fn format_info_str(&self, macaddr: MacAddr) {
        let nc = self.queue();
        nc.set_info_str(&format!("model={},macaddr={}", nc.model(), macaddr));
    }
}

type InitFn = fn(&mut Net, &Netdev, &str, Option<Arc<NetClient>>) -> Result<()>;

/// `net_client_init_fun[]`.
fn init_fun(driver: NetClientDriver) -> Option<InitFn> {
    match driver {
        NetClientDriver::Nic => Some(net_init_nic),
        NetClientDriver::Hubport => Some(net_init_hubport),
        #[cfg(unix)]
        NetClientDriver::Tap => Some(crate::tap::net_init_tap),
        #[cfg(unix)]
        NetClientDriver::Socket => Some(crate::socket::net_init_socket),
        #[cfg(unix)]
        NetClientDriver::Stream => Some(crate::stream::net_init_stream),
        #[cfg(unix)]
        NetClientDriver::Dgram => Some(crate::dgram::net_init_dgram),
        #[cfg(all(unix, feature = "slirp"))]
        NetClientDriver::User => Some(crate::slirp::net_init_slirp),
        #[cfg(unix)]
        NetClientDriver::VhostUser => Some(crate::vhost_user::net_init_vhost_user),
        _ => None,
    }
}

/// The netdev subsystem: every client, the hubs and the NIC table.
pub struct Net {
    clients: Vec<Arc<NetClient>>,
    hubs: Vec<Arc<Hub>>,
    nics: Vec<Arc<Nic>>,
    nd_table: [NicInfo; MAX_NICS],
    nb_nics: usize,
    mac_table: [i32; 256],
    id_counter: u64,
    netdev_opts: QemuOptsList,
    nic_opts: QemuOptsList,
    net_opts: QemuOptsList,
    modern: Vec<Netdev>,
    vm_running: Arc<AtomicBool>,
    fd_resolver: Option<FdResolver>,
    #[cfg(unix)]
    pub(crate) chardev_resolver: Option<crate::vhost_user::ChardevResolver>,
    qtest: bool,
}

impl fmt::Debug for Net {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Net")
            .field("clients", &self.clients.len())
            .field("hubs", &self.hubs.len())
            .field("nb_nics", &self.nb_nics)
            .finish_non_exhaustive()
    }
}

impl Default for Net {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Net {
    fn drop(&mut self) {
        self.cleanup();
        for nic in std::mem::take(&mut self.nics) {
            self.del_nic(&nic);
        }
    }
}

fn opts_list(name: &'static str) -> QemuOptsList {
    QemuOptsList::new(name, &[]).with_implied_opt_name("type")
}

impl Net {
    pub fn new() -> Self {
        Net {
            clients: Vec::new(),
            hubs: Vec::new(),
            nics: Vec::new(),
            nd_table: Default::default(),
            nb_nics: 0,
            mac_table: [0; 256],
            id_counter: 0,
            netdev_opts: opts_list("netdev"),
            nic_opts: opts_list("nic"),
            net_opts: opts_list("net"),
            modern: Vec::new(),
            vm_running: Arc::new(AtomicBool::new(true)),
            fd_resolver: None,
            #[cfg(unix)]
            chardev_resolver: None,
            qtest: false,
        }
    }

    /// Installs the lookup for named descriptors, which `fd=` and `fds=` use for names that do
    /// not start with a digit.
    pub fn set_fd_resolver(&mut self, resolver: Option<FdResolver>) {
        self.fd_resolver = resolver;
    }

    /// Leaves out the "not connected to host network" hub warning, as QEMU does under qtest.
    pub fn set_qtest(&mut self, qtest: bool) {
        self.qtest = qtest;
    }

    /// The `-netdev` options that have been parsed.
    pub fn netdev_opts(&self) -> &QemuOptsList {
        &self.netdev_opts
    }

    pub fn nic_opts(&self) -> &QemuOptsList {
        &self.nic_opts
    }

    pub fn net_opts(&self) -> &QemuOptsList {
        &self.net_opts
    }

    /// Every client, in creation order.
    pub fn clients(&self) -> &[Arc<NetClient>] {
        &self.clients
    }

    /// The hubs, newest first.
    pub fn hubs(&self) -> &[Arc<Hub>] {
        &self.hubs
    }

    /// `nd_table`.
    pub fn nd_table(&self) -> &[NicInfo; MAX_NICS] {
        &self.nd_table
    }

    pub fn nb_nics(&self) -> usize {
        self.nb_nics
    }

    pub fn vm_running(&self) -> bool {
        self.vm_running.load(Ordering::SeqCst)
    }

    /// `net_vm_change_state_handler()`.
    pub fn vm_state_change(&mut self, running: bool) {
        self.vm_running.store(running, Ordering::SeqCst);
        for nc in self.clients.clone() {
            if running {
                if let Some(peer) = nc.peer() {
                    if nc.can_send_packet() {
                        peer.flush_queued_packets();
                    }
                }
            } else {
                nc.flush_or_purge_queued_packets(true);
            }
        }
    }

    // Clients.

    fn assign_name(&self, model: &str) -> String {
        let id = self.clients.iter().filter(|c| c.model() == model).count();
        format!("{model}.{id}")
    }

    /// `qemu_new_net_client()`: makes a client named `name`, or `model.N` without a name,
    /// joined to `peer`.
    pub fn new_client(
        &mut self,
        driver: NetClientDriver,
        peer: Option<&Arc<NetClient>>,
        model: &str,
        name: Option<&str>,
        make_ops: impl FnOnce(&Weak<NetClient>) -> Arc<dyn NetClientOps>,
    ) -> Arc<NetClient> {
        let name = match name {
            Some(n) => n.to_string(),
            None => self.assign_name(model),
        };
        let nc = NetClient::with_runstate(driver, model, &name, self.vm_running.clone(), make_ops);
        if let Some(peer) = peer {
            NetClient::connect(&nc, peer);
        }
        self.clients.push(nc.clone());
        nc
    }

    /// `qemu_new_nic()`: makes the queues of a NIC model, one per peer in `conf`. `make_ops` is
    /// called once per queue with the queue index.
    pub fn new_nic(
        &mut self,
        conf: &NicConf,
        model: &str,
        name: Option<&str>,
        mut make_ops: impl FnMut(&Weak<NetClient>, usize) -> Arc<dyn NetClientOps>,
    ) -> Arc<Nic> {
        let queues = conf.peers.len().max(1);
        let mut ncs = Vec::with_capacity(queues);
        for i in 0..queues {
            let nc = self.new_client(NetClientDriver::Nic, conf.peers.get(i), model, name, |w| {
                make_ops(w, i)
            });
            nc.set_queue_index(i as u32);
            ncs.push(nc);
        }
        let nic = Arc::new(Nic {
            queues: ncs,
            macaddr: conf.macaddr,
            peer_deleted: AtomicBool::new(false),
            deleted_peers: Mutex::new(Vec::new()),
        });
        self.nics.push(nic.clone());
        nic
    }

    fn nic_of(&self, nc: &NetClient) -> Option<Arc<Nic>> {
        self.nics.iter().find(|n| n.queues.iter().any(|q| std::ptr::eq(q.as_ref(), nc))).cloned()
    }

    /// `qemu_find_netdev()`.
    pub fn find_netdev(&self, id: &str) -> Option<Arc<NetClient>> {
        self.clients.iter().find(|c| c.driver() != NetClientDriver::Nic && c.name() == id).cloned()
    }

    /// `qemu_find_net_clients_except()`: every client named `id` not of type `except`.
    pub fn find_clients_except(
        &self,
        id: &str,
        except: Option<NetClientDriver>,
    ) -> Vec<Arc<NetClient>> {
        self.clients
            .iter()
            .filter(|c| Some(c.driver()) != except && c.name() == id)
            .take(MAX_QUEUE_NUM)
            .cloned()
            .collect()
    }

    fn remove_from_list(&mut self, nc: &NetClient) {
        self.clients.retain(|c| !std::ptr::eq(c.as_ref(), nc));
    }

    /// `qemu_free_net_client()`: unlinks the peer. Packets still queued are dropped without
    /// telling anybody, as `qemu_del_net_queue()` does.
    fn free_client(nc: &NetClient) {
        if let Some(peer) = nc.peer() {
            peer.clear_peer();
        }
    }

    /// `qemu_del_net_client()`: deletes a backend and its other queues.
    pub fn del_client(&mut self, nc: &Arc<NetClient>) {
        assert_ne!(nc.driver(), NetClientDriver::Nic, "use del_nic for NICs");
        let ncs = self.find_clients_except(nc.name(), Some(NetClientDriver::Nic));
        assert!(!ncs.is_empty());

        if let Some(peer) = nc.peer() {
            if peer.driver() == NetClientDriver::Nic {
                let Some(nic) = self.nic_of(&peer) else {
                    return;
                };
                if nic.peer_deleted.swap(true, Ordering::SeqCst) {
                    return;
                }
                for q in &ncs {
                    if let Some(p) = q.peer() {
                        p.set_link_down(true);
                    }
                    self.remove_from_list(q);
                }
                lock(&nic.deleted_peers).extend(ncs);
                peer.ops().link_status_changed(&peer);
                return;
            }
        }

        for q in &ncs {
            self.remove_from_list(q);
            q.ops().cleanup(q);
            Self::free_client(q);
        }
    }

    /// `qemu_del_nic()`.
    pub fn del_nic(&mut self, nic: &Arc<Nic>) {
        self.macaddr_set_free(nic.macaddr);
        let peer_deleted = nic.peer_deleted();
        for nc in &nic.queues {
            let Some(peer) = nc.peer() else {
                continue;
            };
            if peer_deleted {
                peer.ops().cleanup(&peer);
                Self::free_client(&peer);
            } else {
                peer.purge_queued_packets();
            }
        }
        lock(&nic.deleted_peers).clear();
        for nc in nic.queues.iter().rev() {
            self.remove_from_list(nc);
            nc.ops().cleanup(nc);
            Self::free_client(nc);
        }
        self.nics.retain(|n| !Arc::ptr_eq(n, nic));
    }

    /// `net_cleanup()`: deletes every backend. NICs belong to their devices and stay.
    pub fn cleanup(&mut self) {
        loop {
            let Some(nc) =
                self.clients.iter().find(|c| c.driver() != NetClientDriver::Nic).cloned()
            else {
                break;
            };
            self.del_client(&nc);
        }
    }

    /// `qmp_set_link()`.
    pub fn set_link(&mut self, name: &str, up: bool) -> Result<()> {
        let ncs = self.find_clients_except(name, None);
        if ncs.is_empty() {
            return Err(Error::new(
                ErrorClass::DeviceNotFound,
                format!("Device '{name}' not found"),
            ));
        }
        net_client_set_link(&ncs, up);
        Ok(())
    }

    /// `qmp_netdev_del()`.
    pub fn netdev_del(&mut self, id: &str) -> Result<()> {
        let Some(nc) = self.find_netdev(id) else {
            return Err(Error::new(ErrorClass::DeviceNotFound, format!("Device '{id}' not found")));
        };
        if !nc.is_netdev() {
            return Err(Error::generic(format!("Device '{id}' is not a netdev")));
        }
        self.del_client(&nc);
        if let Some(h) = self.netdev_opts.find(Some(id)).map(QemuOpts::handle) {
            self.netdev_opts.del(h);
        }
        Ok(())
    }

    // Hubs.

    /// `net_hub_add_port()`: adds a port to hub `hub_id`, making the hub if needed.
    pub fn hub_add_port(
        &mut self,
        hub_id: i32,
        name: Option<&str>,
        hubpeer: Option<&Arc<NetClient>>,
    ) -> Arc<NetClient> {
        let hub = match self.hubs.iter().find(|h| h.id() == hub_id) {
            Some(h) => h.clone(),
            None => {
                let h = Hub::new(hub_id);
                self.hubs.insert(0, h.clone());
                h
            }
        };
        let (port_id, default_name) = hub.next_port();
        let name = name.map_or(default_name, str::to_string);
        let weak_hub = Arc::downgrade(&hub);
        let nc = self.new_client(NetClientDriver::Hubport, hubpeer, "hub", Some(&name), |_| {
            Arc::new(HubPortOps { hub: weak_hub.clone() })
        });
        let _ = nc.hub.set((weak_hub, port_id));
        hub.list_port(&nc);
        nc
    }

    fn hub_check_clients(&self, warnings: &mut Vec<String>) {
        for hub in &self.hubs {
            let (mut has_nic, mut has_host_dev) = (false, false);
            for port in hub.ports() {
                let Some(peer) = port.peer() else {
                    warnings.push(format!("hub port {} has no peer", port.name()));
                    continue;
                };
                match peer.driver() {
                    NetClientDriver::Nic => has_nic = true,
                    NetClientDriver::User
                    | NetClientDriver::Tap
                    | NetClientDriver::Socket
                    | NetClientDriver::Stream
                    | NetClientDriver::Dgram
                    | NetClientDriver::Vde
                    | NetClientDriver::VhostUser => has_host_dev = true,
                    _ => {}
                }
            }
            if has_host_dev && !has_nic {
                warnings.push(format!("hub {} with no nics", hub.id()));
            }
            if has_nic && !has_host_dev && !self.qtest {
                warnings.push(format!("hub {} is not connected to host network", hub.id()));
            }
        }
    }

    /// `net_check_clients()`: reports each warning with `warn_report` and returns them too.
    pub fn check_clients(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        self.hub_check_clients(&mut warnings);
        for nc in &self.clients {
            if nc.peer().is_none() {
                let kind = if nc.driver() == NetClientDriver::Nic { "nic" } else { "netdev" };
                warnings.push(format!("{kind} {} has no peer", nc.name()));
            }
        }
        for nd in &self.nd_table {
            if nd.used && !nd.instantiated {
                warnings.push(format!(
                    "requested NIC ({}, model {}) was not created (not supported by this machine?)",
                    nd.name.as_deref().unwrap_or("anonymous"),
                    nd.model.as_deref().unwrap_or("unspecified")
                ));
            }
        }
        for w in &warnings {
            warn_report(w);
        }
        warnings
    }

    /// `hmp_info_network()`.
    pub fn info_network(&self) -> String {
        let mut out = String::new();
        for hub in &self.hubs {
            out.push_str(&format!("hub {}\n", hub.id()));
            for port in hub.ports() {
                out.push_str(&format!(" \\ {}", port.name()));
                match port.peer() {
                    Some(peer) => {
                        out.push_str(": ");
                        print_net_client(&mut out, &peer);
                    }
                    None => out.push('\n'),
                }
            }
        }
        for nc in &self.clients {
            if hub_id_for_client(nc).is_some() {
                continue;
            }
            let peer = nc.peer();
            let is_nic = nc.driver() == NetClientDriver::Nic;
            if peer.is_none() || is_nic {
                print_net_client(&mut out, nc);
            }
            if let (Some(peer), true) = (peer, is_nic) {
                out.push_str(" \\ ");
                print_net_client(&mut out, &peer);
            }
        }
        out
    }

    // MAC addresses.

    fn macaddr_set_used(&mut self, mac: MacAddr) {
        let i = usize::from(mac.0[5]);
        if (0x56..0xFF).contains(&i) {
            self.mac_table[i] += 1;
        }
    }

    fn macaddr_set_free(&mut self, mac: MacAddr) {
        if mac.0[..5] != MAC_BASE[..5] {
            return;
        }
        let i = usize::from(mac.0[5]);
        if (0x56..0xFF).contains(&i) {
            self.mac_table[i] -= 1;
        }
    }

    /// `qemu_macaddr_default_if_unset()`: gives a zero address the next free
    /// `52:54:00:12:34:xx`, and counts an address of that form as taken.
    pub fn macaddr_default_if_unset(&mut self, mac: &mut MacAddr) {
        if !mac.is_zero() {
            if mac.0[..5] == MAC_BASE[..5] {
                self.macaddr_set_used(*mac);
            }
            return;
        }
        let free = (0x56..0xFF).find(|&i| self.mac_table[i] == 0).unwrap_or(0xFF);
        *mac = MacAddr([0x52, 0x54, 0x00, 0x12, 0x34, free as u8]);
        self.macaddr_set_used(*mac);
    }

    /// `qemu_find_nic_info()`: the first NIC asked for on the command line that matches
    /// `typename` (or `alias`, or any model with `match_default` if none was given) and has not
    /// been created yet. The caller sets `instantiated` once it made the device.
    pub fn find_nic_info(
        &mut self,
        typename: &str,
        match_default: bool,
        alias: Option<&str>,
    ) -> Option<&mut NicInfo> {
        let nb = self.nb_nics;
        self.nd_table[..nb].iter_mut().find(|nd| {
            nd.used
                && !nd.instantiated
                && ((match_default && nd.model.is_none())
                    || nd.model.as_deref() == Some(typename)
                    || (alias.is_some() && nd.model.as_deref() == alias))
        })
    }

    fn nic_get_free_idx(&self) -> Option<usize> {
        self.nd_table.iter().position(|nd| !nd.used)
    }

    /// `id_generate(ID_NET)`.
    fn id_generate(&mut self) -> String {
        let rnd = RandomState::new().hash_one(self.id_counter) % 100;
        let id = format!("#net{}{:02}", self.id_counter, rnd);
        self.id_counter += 1;
        id
    }

    // Descriptors.

    /// `monitor_fd_param()`: a descriptor number, or a name the monitor knows.
    pub fn fd_param(&mut self, name: &str) -> Result<RawFd> {
        let digit = name.as_bytes().first().is_some_and(u8::is_ascii_digit);
        if !digit {
            if let Some(resolve) = self.fd_resolver.as_mut() {
                return resolve(name).map(into_raw);
            }
        }
        match name.parse::<i64>() {
            Ok(fd) if name.bytes().all(|b| b.is_ascii_digit()) && fd <= i64::from(i32::MAX) => {
                Ok(fd as RawFd)
            }
            _ => Err(Error::generic(format!("Invalid file descriptor number '{name}'"))),
        }
    }

    /// `net_parse_fds()`: descriptors separated by colons, exactly `expected` of them unless it
    /// is 0.
    pub fn parse_fds(&mut self, param: &str, expected: usize) -> Result<Vec<RawFd>> {
        let names: Vec<&str> =
            if param.is_empty() { Vec::new() } else { param.split(':').collect() };
        if expected != 0 && names.len() != expected {
            return Err(Error::generic(format!(
                "expected {expected} socket fds, got {}",
                names.len()
            )));
        }
        if names.is_empty() {
            return Err(Error::generic("no fds passed"));
        }
        let mut fds = Vec::with_capacity(names.len());
        for n in names {
            match self.fd_param(n) {
                Ok(fd) => fds.push(fd),
                Err(e) => {
                    #[cfg(unix)]
                    for fd in fds {
                        drop(crate::fd::adopt(fd));
                    }
                    return Err(e);
                }
            }
        }
        Ok(fds)
    }

    // Options.

    /// `-netdev`: JSON and the `stream` and `dgram` types go the modern way, the rest into the
    /// `netdev` option list.
    pub fn parse_netdev(&mut self, optarg: &str) -> Result<()> {
        if netdev_is_modern(optarg) {
            let nd = parse_modern(optarg)?;
            self.modern.push(nd);
            return Ok(());
        }
        self.netdev_opts.parse(optarg, true)?;
        Ok(())
    }

    /// `-nic`.
    pub fn parse_nic(&mut self, optarg: &str) -> Result<()> {
        self.nic_opts.parse(optarg, true)?;
        Ok(())
    }

    /// `-net`.
    pub fn parse_net(&mut self, optarg: &str) -> Result<()> {
        self.net_opts.parse(optarg, true)?;
        Ok(())
    }

    /// `net_init_clients()`: creates what the options asked for, modern `-netdev` first, then
    /// the other `-netdev`s, then `-nic`, then `-net`. Stops at the first error.
    pub fn init_clients(&mut self) -> Result<()> {
        for nd in std::mem::take(&mut self.modern) {
            self.client_init1(&nd, true)?;
        }
        self.foreach_opts(OptsKind::Netdev)?;
        self.foreach_opts(OptsKind::Nic)?;
        self.foreach_opts(OptsKind::Net)?;
        Ok(())
    }

    fn foreach_opts(&mut self, kind: OptsKind) -> Result<()> {
        let slot = match kind {
            OptsKind::Netdev => &mut self.netdev_opts,
            OptsKind::Nic => &mut self.nic_opts,
            OptsKind::Net => &mut self.net_opts,
        };
        let name = slot.name().unwrap_or("netdev");
        let mut list = std::mem::replace(slot, opts_list(name));
        let mut result = Ok(());
        for opts in list.iter_mut() {
            result = match kind {
                OptsKind::Netdev => self.init_netdev(opts),
                OptsKind::Nic => self.param_nic(opts),
                OptsKind::Net => self.init_net(opts),
            };
            if result.is_err() {
                break;
            }
        }
        let slot = match kind {
            OptsKind::Netdev => &mut self.netdev_opts,
            OptsKind::Nic => &mut self.nic_opts,
            OptsKind::Net => &mut self.net_opts,
        };
        *slot = list;
        result
    }

    /// `net_init_netdev()`.
    fn init_netdev(&mut self, opts: &mut QemuOpts) -> Result<()> {
        if opts.get("type").is_some_and(is_help_option) {
            show_netdevs();
            std::process::exit(0);
        }
        self.client_init(opts, true)
    }

    /// `net_init_client()`.
    fn init_net(&mut self, opts: &mut QemuOpts) -> Result<()> {
        if opts.get("model").is_some_and(is_help_option) {
            return Ok(());
        }
        self.client_init(opts, false)
    }

    /// `net_param_nic()`.
    fn param_nic(&mut self, opts: &mut QemuOpts) -> Result<()> {
        let ty = opts.get("type").map(str::to_string);
        if let Some(t) = &ty {
            if t == "none" {
                return Ok(());
            }
            if is_help_option(t) {
                show_netdevs();
                std::process::exit(0);
            }
        }
        let Some(idx) = self.nic_get_free_idx().filter(|_| self.nb_nics < MAX_NICS) else {
            return Err(Error::generic("no more on-board/default NIC slots available"));
        };
        if ty.is_none() {
            opts.set("type", "user")?;
        }
        let mut ni = NicInfo { model: opts.get_del("model"), ..NicInfo::default() };
        if ni.model.as_deref().is_some_and(is_help_option) {
            self.nd_table[idx] = ni;
            return Ok(());
        }
        let nd_id = match opts.id() {
            Some(id) => id.to_string(),
            None => {
                let id = self.id_generate();
                opts.set_id(Some(id.clone()));
                id
            }
        };
        if let Some(mac) = opts.get_del("mac") {
            if !parse_macaddr(&mut ni.macaddr, &mac) {
                self.nd_table[idx] = ni;
                return Err(Error::generic("invalid syntax for ethernet address"));
            }
            if ni.macaddr.is_multicast() {
                self.nd_table[idx] = ni;
                return Err(Error::generic("NIC cannot have multicast MAC address"));
            }
        }
        self.macaddr_default_if_unset(&mut ni.macaddr);
        let r = self.client_init(opts, true);
        if r.is_ok() {
            ni.netdev = self.find_netdev(&nd_id);
            ni.used = true;
            self.nb_nics += 1;
        }
        self.nd_table[idx] = ni;
        r
    }

    /// `net_client_init()`.
    fn client_init(&mut self, opts: &mut QemuOpts, is_netdev: bool) -> Result<()> {
        if let Some(ip6_net) = opts.get("ipv6-net").map(str::to_string) {
            let (prefix, len) = match ip6_net.split_once('/') {
                Some((p, l)) => (p.to_string(), Some(l.to_string())),
                None => (ip6_net.clone(), None),
            };
            let mut prefix_len = 64u64;
            if let Some(l) = len {
                match ruvm_qapi::cutils::strtou64(&l, 10, false) {
                    Ok((v, _)) => prefix_len = v,
                    Err(_) => {
                        return Err(Error::generic(
                            "parameter 'ipv6-net' expects a number after '/'",
                        ));
                    }
                }
            }
            opts.set("ipv6-prefix", &prefix)?;
            opts.set_number("ipv6-prefixlen", prefix_len as i64)?;
            opts.unset("ipv6-net");
        }
        if !is_netdev && opts.id().is_none() {
            let id = self.id_generate();
            opts.set_id(Some(id));
        }
        let mut v = OptsVisitor::new(opts);
        #[cfg(target_os = "linux")]
        if opts.get("type") == Some("passt") {
            let (id, passt) = crate::passt::visit_netdev(&mut v)?;
            return self.passt_init1(&id, &passt, is_netdev);
        }
        let mut nd = Netdev::default();
        Netdev::visit(&mut v, None, &mut nd)?;
        self.client_init1(&nd, is_netdev)
    }

    /// `net_client_init1()` for `type=passt`, which has no `Netdev` branch here.
    #[cfg(target_os = "linux")]
    fn passt_init1(
        &mut self,
        id: &str,
        passt: &crate::passt::PasstOptions,
        is_netdev: bool,
    ) -> Result<()> {
        let peer = if is_netdev { None } else { Some(self.hub_add_port(0, None, None)) };
        if self.find_netdev(id).is_some() {
            return Err(Error::generic(format!("Duplicate ID '{id}'")));
        }
        crate::passt::net_init_passt(self, passt, id, peer)?;
        if is_netdev {
            let nc = self.find_netdev(id).expect("backend registered its client");
            nc.set_is_netdev();
        }
        Ok(())
    }

    /// `net_client_init1()`.
    fn client_init1(&mut self, netdev: &Netdev, is_netdev: bool) -> Result<()> {
        let ty = netdev.u.tag();
        let not_compiled = || {
            Error::generic(format!(
                "network backend '{}' is not compiled into this binary",
                ty.as_str()
            ))
        };
        let mut peer = None;
        if is_netdev {
            if ty == NetClientDriver::Nic || init_fun(ty).is_none() {
                return Err(not_compiled());
            }
        } else {
            if ty == NetClientDriver::None {
                return Ok(());
            }
            if ty == NetClientDriver::Hubport {
                return Err(Error::generic(format!(
                    "network backend '{}' is only supported with -netdev/-nic",
                    ty.as_str()
                )));
            }
            if init_fun(ty).is_none() {
                return Err(not_compiled());
            }
            let nic_with_netdev = matches!(&netdev.u, NetdevU::Nic(n) if n.netdev.is_some());
            if !nic_with_netdev {
                peer = Some(self.hub_add_port(0, None, None));
            }
        }
        if self.find_netdev(&netdev.id).is_some() {
            return Err(Error::generic(format!("Duplicate ID '{}'", netdev.id)));
        }
        let init = init_fun(ty).ok_or_else(not_compiled)?;
        init(self, netdev, &netdev.id, peer)?;
        if is_netdev {
            let nc = self.find_netdev(&netdev.id).expect("backend registered its client");
            nc.set_is_netdev();
        }
        Ok(())
    }

    /// `netdev_add` from the monitor with options, as HMP gives them: parsed into the
    /// `netdev` list and created.
    pub fn netdev_add_opts(&mut self, optarg: &str) -> Result<()> {
        if netdev_is_modern(optarg) {
            let nd = parse_modern(optarg)?;
            return self.netdev_add(&nd);
        }
        let handle = self.netdev_opts.parse(optarg, true)?.handle();
        let mut list = std::mem::replace(&mut self.netdev_opts, opts_list("netdev"));
        let r = match list.get_mut(handle) {
            Some(opts) => self.client_init(opts, true),
            None => Ok(()),
        };
        if r.is_err() {
            list.del(handle);
        }
        self.netdev_opts = list;
        r
    }

    /// `qmp_netdev_add()`.
    pub fn netdev_add(&mut self, netdev: &Netdev) -> Result<()> {
        if !id_wellformed(&netdev.id) {
            return Err(Error::generic("Parameter 'id' expects an identifier"));
        }
        self.client_init1(netdev, true)
    }

    /// `hostfwd_add [netdev_id] [tcp|udp|unix]:[hostaddr]:hostport-[guestaddr]:guestport`:
    /// adds a host forwarding rule to the `user` backend `id`, or to the first one.
    pub fn hostfwd_add(&self, id: Option<&str>, redir: &str) -> Result<()> {
        #[cfg(all(unix, feature = "slirp"))]
        return crate::slirp::hostfwd_add(self, id, redir);
        #[cfg(not(all(unix, feature = "slirp")))]
        {
            let _ = redir;
            Err(self.no_usernet(id))
        }
    }

    /// `hostfwd_remove [netdev_id] [tcp|udp]:[hostaddr]:hostport`. A backend that cannot be
    /// found is an error; otherwise the result is the line the monitor prints, which says
    /// whether the rule was removed, or "invalid format".
    pub fn hostfwd_remove(&self, id: Option<&str>, src: &str) -> Result<String> {
        #[cfg(all(unix, feature = "slirp"))]
        return crate::slirp::hostfwd_remove(self, id, src);
        #[cfg(not(all(unix, feature = "slirp")))]
        {
            let _ = src;
            Err(self.no_usernet(id))
        }
    }

    /// `info usernet`: the connections of every `user` backend.
    pub fn info_usernet(&self) -> String {
        #[cfg(all(unix, feature = "slirp"))]
        return crate::slirp::info_usernet(self);
        #[cfg(not(all(unix, feature = "slirp")))]
        String::new()
    }

    /// The error `slirp_lookup()` gives when there are no `user` backends at all.
    #[cfg(not(all(unix, feature = "slirp")))]
    fn no_usernet(&self, id: Option<&str>) -> Error {
        match id {
            Some(id) if self.find_netdev(id).is_none() => {
                Error::generic(format!("unrecognized netdev id '{id}'"))
            }
            Some(_) => Error::generic("invalid device specified"),
            None => Error::generic("user mode network stack not in use"),
        }
    }
}

#[derive(Clone, Copy)]
enum OptsKind {
    Netdev,
    Nic,
    Net,
}

const MAC_BASE: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x00];

/// `net_client_set_link()`.
fn net_client_set_link(ncs: &[Arc<NetClient>], up: bool) {
    let nc = &ncs[0];
    for q in ncs {
        q.set_link_down(!up);
    }
    nc.ops().link_status_changed(nc);
    if let Some(peer) = nc.peer() {
        if peer.driver() == NetClientDriver::Nic {
            for q in ncs {
                if let Some(p) = q.peer() {
                    p.set_link_down(!up);
                }
            }
        }
        peer.ops().link_status_changed(&peer);
    }
}

/// `print_net_client()`.
fn print_net_client(out: &mut String, nc: &NetClient) {
    out.push_str(&format!(
        "{}: index={},type={},{}\n",
        nc.name(),
        nc.queue_index(),
        nc.driver().as_str(),
        nc.info_str()
    ));
}

/// `show_netdevs()`.
pub fn show_netdevs() {
    println!("Available netdev backend types:");
    for n in AVAILABLE_NETDEVS {
        println!("{n}");
    }
}

/// `netdev_is_modern()`.
pub fn netdev_is_modern(optstr: &str) -> bool {
    if optstr.starts_with('{') {
        return true;
    }
    let mut list = opts_list("netdev");
    let Ok(opts) = list.create(None, false) else {
        return false;
    };
    if opts.do_parse(optstr, Some("type")).is_err() {
        return false;
    }
    matches!(opts.get("type"), Some("stream" | "dgram"))
}

/// `netdev_parse_modern()` without queueing: JSON or dotted keys, visited as a `Netdev`.
pub fn parse_modern(optstr: &str) -> Result<Netdev> {
    let mut v = if optstr.starts_with('{') {
        QObjectInputVisitor::new(ruvm_qapi::json::from_str(optstr)?)
    } else {
        let dict = ruvm_qapi::keyval::keyval_parse(optstr, Some("type"), None)?;
        QObjectInputVisitor::new_keyval(ruvm_qapi::QValue::Dict(dict))
    };
    let mut nd = Netdev::default();
    Netdev::visit(&mut v, None, &mut nd)?;
    Ok(nd)
}

/// `net_init_nic()`.
fn net_init_nic(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::Nic(nic) = &netdev.u else {
        unreachable!("nic init with another type");
    };
    let Some(idx) = net.nic_get_free_idx().filter(|_| net.nb_nics < MAX_NICS) else {
        return Err(Error::generic("too many NICs"));
    };
    let mut nd = NicInfo::default();
    net.nd_table[idx] = NicInfo::default();
    nd.netdev = match &nic.netdev {
        Some(id) => Some(
            net.find_netdev(id)
                .ok_or_else(|| Error::generic(format!("netdev '{id}' not found")))?,
        ),
        None => Some(peer.expect("-net nic gets a hub port")),
    };
    nd.name = Some(name.to_string());
    nd.model = nic.model.clone();
    nd.devaddr = nic.addr.clone();
    if let Some(mac) = &nic.macaddr {
        if !parse_macaddr(&mut nd.macaddr, mac) {
            return Err(Error::generic("invalid syntax for ethernet address"));
        }
        if nd.macaddr.is_multicast() {
            return Err(Error::generic("NIC cannot have multicast MAC address (odd 1st byte)"));
        }
    }
    net.macaddr_default_if_unset(&mut nd.macaddr);
    nd.nvectors = match nic.vectors {
        Some(v) if v > 0x7ff_ffff => {
            return Err(Error::generic(format!("invalid # of vectors: {v}")));
        }
        Some(v) => v as i32,
        None => DEV_NVECTORS_UNSPECIFIED,
    };
    nd.used = true;
    net.nd_table[idx] = nd;
    net.nb_nics += 1;
    Ok(())
}

/// `net_init_hubport()`.
fn net_init_hubport(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    assert!(peer.is_none());
    let NetdevU::Hubport(NetdevHubPortOptions { hubid, netdev: hub_netdev }) = &netdev.u else {
        unreachable!("hubport init with another type");
    };
    let hubpeer = match hub_netdev {
        Some(id) => Some(
            net.find_netdev(id)
                .ok_or_else(|| Error::generic(format!("netdev '{id}' not found")))?,
        ),
        None => None,
    };
    net.hub_add_port(*hubid, Some(name), hubpeer.as_ref());
    Ok(())
}
