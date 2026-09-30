// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev vhost-user`, ported from net/vhost-user.c.
//!
//! The backend does no packet work of its own: the frames go between the guest and the
//! vhost-user process through shared rings that the virtio-net device sets up. What happens
//! here is the part QEMU does when the netdev is created: taking the chardev, connecting, and
//! the first requests of the protocol (features, protocol features, queue count, owner). The
//! device model then gets the connection through [`NetClient::vhost_user`].
//!
//! ruvm has no chardev layer this crate can reach, so `chardev=` names are looked up through a
//! resolver the embedder installs with [`Net::set_chardev_resolver`]. Tests use
//! [`VhostUserChardev::connect`] to hand in a connected Unix socket.

use std::fmt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Arc, Mutex};

use ruvm_base::{Error, Result, error_report};
use ruvm_qapi::types::{NetClientDriver, Netdev, NetdevU};
use ruvm_vhost::user::Frontend;
use ruvm_vhost::user::message::protocol;

use crate::client::{MAX_QUEUE_NUM, NetClient, NetClientOps};
use crate::net::Net;

/// The protocol features the frontend takes up when the backend offers them.
pub const VHOST_USER_NET_PROTOCOL_FEATURES: u64 = protocol::MQ
    | protocol::LOG_SHMFD
    | protocol::RARP
    | protocol::REPLY_ACK
    | protocol::MTU
    | protocol::CROSS_ENDIAN
    | protocol::CONFIGURE_MEM_SLOTS
    | protocol::STATUS
    | protocol::RESET_DEVICE;

/// A character device as `-netdev vhost-user,chardev=` needs it: a connected socket, plus the
/// two chardev features QEMU checks before it takes one.
#[derive(Debug)]
pub struct VhostUserChardev {
    label: String,
    stream: UnixStream,
    reconnectable: bool,
    fd_pass: bool,
}

impl VhostUserChardev {
    /// Wraps a connected socket. `reconnectable` and `fd_pass` are
    /// `QEMU_CHAR_FEATURE_RECONNECTABLE` and `QEMU_CHAR_FEATURE_FD_PASS`.
    pub fn new(label: &str, stream: UnixStream, reconnectable: bool, fd_pass: bool) -> Self {
        VhostUserChardev { label: label.to_string(), stream, reconnectable, fd_pass }
    }

    /// Connects to a vhost-user backend listening on `path`, like `-chardev socket,path=`
    /// does. Such a chardev can reconnect and pass descriptors. This stands in for the chardev
    /// layer and is meant for embedders and tests.
    pub fn connect(label: &str, path: impl AsRef<Path>) -> std::io::Result<Self> {
        Ok(Self::new(label, UnixStream::connect(path)?, true, true))
    }

    /// The chardev id.
    pub fn label(&self) -> &str {
        &self.label
    }
}

/// Looks up the chardev `chardev=` names, `qemu_chr_find()`. The chardev is handed over, as
/// QEMU's backend claims it.
pub type ChardevResolver = Box<dyn FnMut(&str) -> Option<VhostUserChardev> + Send>;

/// What a `vhost-user` netdev set up with its backend, shared by all its queues.
pub struct VhostUserNet {
    frontend: Arc<Mutex<Frontend>>,
    features: u64,
    protocol_features: u64,
    max_queues: u64,
    queues: u32,
    chardev: String,
}

impl fmt::Debug for VhostUserNet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VhostUserNet")
            .field("features", &format_args!("{:#x}", self.features))
            .field("protocol_features", &format_args!("{:#x}", self.protocol_features))
            .field("max_queues", &self.max_queues)
            .field("queues", &self.queues)
            .field("chardev", &self.chardev)
            .finish_non_exhaustive()
    }
}

impl VhostUserNet {
    /// The connection, for the device model to set up memory and rings.
    pub fn frontend(&self) -> Arc<Mutex<Frontend>> {
        self.frontend.clone()
    }

    /// The virtio and vhost-user features the backend offered with `GET_FEATURES`.
    pub fn features(&self) -> u64 {
        self.features
    }

    /// The protocol features both sides agreed on.
    pub fn protocol_features(&self) -> u64 {
        self.protocol_features
    }

    /// The queue pairs the backend can do: `GET_QUEUE_NUM`, or 1 without `MQ`.
    pub fn max_queues(&self) -> u64 {
        self.max_queues
    }

    /// The queue pairs asked for with `queues=`.
    pub fn queues(&self) -> u32 {
        self.queues
    }

    /// The label of the chardev the connection came from.
    pub fn chardev(&self) -> &str {
        &self.chardev
    }
}

/// `NetVhostUserState`, one per queue pair.
#[derive(Debug)]
struct VhostUserOps {
    shared: Arc<VhostUserNet>,
}

impl NetClientOps for VhostUserOps {
    /// `vhost_user_receive()`. Frames only get here when the rings are not in use; the backend
    /// has nowhere to take them, so they are accepted and dropped. QEMU also asks the backend
    /// to announce the guest when it sees a 60 byte RARP frame, which is left out.
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        iov.iter().map(|b| b.len()).sum::<usize>() as isize
    }

    fn has_vnet_hdr(&self, _nc: &NetClient) -> bool {
        true
    }

    fn has_ufo(&self, _nc: &NetClient) -> bool {
        true
    }

    fn set_vnet_le(&self, _nc: &NetClient, _is_le: bool) -> Option<std::io::Result<()>> {
        Some(Ok(()))
    }

    fn set_vnet_be(&self, _nc: &NetClient, _is_be: bool) -> Option<std::io::Result<()>> {
        Some(Ok(()))
    }

    fn as_any(&self) -> Option<&(dyn std::any::Any + Send + Sync)> {
        Some(self)
    }
}

impl NetClient {
    /// The vhost-user connection behind a `vhost-user` backend queue, `get_vhost_net()`.
    pub fn vhost_user(&self) -> Option<Arc<VhostUserNet>> {
        let ops = self.ops().as_any()?.downcast_ref::<VhostUserOps>()?;
        Some(ops.shared.clone())
    }
}

/// The start of the session `vhost_net_init()` and `vhost_user_start()` do.
fn start(frontend: &mut Frontend, queues: u32) -> std::result::Result<(u64, u64, u64), String> {
    let init_failed = |e: ruvm_vhost::Error| format!("vhost_backend_init failed: {e}");
    let features = frontend.get_features().map_err(init_failed)?;
    let protocol_features = frontend
        .negotiate_protocol_features(VHOST_USER_NET_PROTOCOL_FEATURES)
        .map_err(init_failed)?;
    let max_queues = if protocol_features & protocol::MQ != 0 {
        frontend.get_queue_num().map_err(init_failed)?
    } else {
        1
    };
    frontend.set_owner().map_err(|e| format!("vhost_set_owner failed: {e}"))?;
    if u64::from(queues) > max_queues {
        return Err(format!("you are asking more queues than supported: {max_queues}"));
    }
    Ok((features, protocol_features, max_queues))
}

/// `net_init_vhost_user()` and `net_vhost_user_init()`.
pub(crate) fn net_init_vhost_user(
    net: &mut Net,
    netdev: &Netdev,
    name: &str,
    peer: Option<Arc<NetClient>>,
) -> Result<()> {
    let NetdevU::VhostUser(opts) = &netdev.u else {
        unreachable!("net_init_vhost_user called for {:?}", netdev.u.tag());
    };
    // net_vhost_claim_chardev()
    let Some(chr) = net.resolve_chardev(&opts.chardev) else {
        return Err(Error::generic(format!("chardev \"{}\" not found", opts.chardev)));
    };
    if !chr.reconnectable {
        return Err(Error::generic(format!("chardev \"{}\" is not reconnectable", opts.chardev)));
    }
    if !chr.fd_pass {
        return Err(Error::generic(format!(
            "chardev \"{}\" does not support FD passing",
            opts.chardev
        )));
    }
    let queues = opts.queues.map_or(1, |q| q as i32);
    if queues < 1 || queues as usize > MAX_QUEUE_NUM {
        return Err(Error::generic(format!(
            "vhost-user number of queues must be in range [1, {MAX_QUEUE_NUM}]"
        )));
    }
    let queues = queues as u32;

    let mut frontend = Frontend::new(chr.stream);
    let (features, protocol_features, max_queues) = match start(&mut frontend, queues) {
        Ok(v) => v,
        Err(e) => {
            error_report(&e);
            return Err(Error::generic("Device 'vhost-user' could not be initialized"));
        }
    };
    let shared = Arc::new(VhostUserNet {
        frontend: Arc::new(Mutex::new(frontend)),
        features,
        protocol_features,
        max_queues,
        queues,
        chardev: chr.label.clone(),
    });
    for i in 0..queues {
        let s = shared.clone();
        let nc = net.new_client(
            NetClientDriver::VhostUser,
            peer.as_ref(),
            "vhost_user",
            Some(name),
            move |_| Arc::new(VhostUserOps { shared: s }),
        );
        nc.set_info_str(&format!("vhost-user{i} to {}", chr.label));
        nc.set_queue_index(i);
    }
    Ok(())
}

impl Net {
    /// Installs the chardev lookup `-netdev vhost-user,chardev=` uses.
    pub fn set_chardev_resolver(&mut self, resolver: Option<ChardevResolver>) {
        self.chardev_resolver = resolver;
    }

    fn resolve_chardev(&mut self, label: &str) -> Option<VhostUserChardev> {
        self.chardev_resolver.as_mut().and_then(|r| r(label))
    }
}
