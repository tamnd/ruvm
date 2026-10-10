// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev` for the x86 boards, through ruvm-net's [`Net`], and the link between a virtio-net
//! and its netdev, which `set_netdev()` and `qemu_new_nic()` make in QEMU.
//!
//! Frames that reach a NIC wait in a short queue for a virtual clock timer, which hands them to
//! the device. So delivery stops with the VM, as `virtio_net_can_receive()` makes it in QEMU, a
//! migration finds the devices still, and no device lock is taken on the thread that sent the
//! frame, which may hold the lock of the sending NIC. When the queue is full the frames stay in
//! ruvm-net until the NIC drains.

use std::collections::VecDeque;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use ruvm_base::report::Location;
use ruvm_hw_core::{Clock, Timer};
use ruvm_hw_virtio::{NetPeer, RxOutcome, VirtioNet, VirtioNetHdr};
use ruvm_machine_x86::VirtioHandle;
use ruvm_net::{MacAddr, Net, NetClient, NetClientOps, NicConf};
use ruvm_qapi::types::NetClientDriver;

use crate::x86::Located;

/// Frames waiting for one NIC before ruvm-net has to hold them.
const NIC_QUEUE_MAX: usize = 256;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The netdevs and NICs of a machine.
pub(crate) struct Network {
    net: Mutex<Net>,
    clock: Arc<Clock>,
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Network").finish_non_exhaustive()
    }
}

impl Network {
    /// Parses the `-netdev` options and makes the backends, `net_init_clients()`. The VM
    /// counts as stopped until [`Network::vm_state_change`] says otherwise.
    pub(crate) fn new(
        netdevs: &[(String, Option<Location>)],
        clock: &Arc<Clock>,
    ) -> Result<Network, Located> {
        let mut net = Net::new();
        for (arg, loc) in netdevs {
            net.parse_netdev(arg).map_err(|e| Located(loc.clone(), e))?;
        }
        net.init_clients().map_err(|e| Located(None, e))?;
        net.vm_state_change(false);
        Ok(Network { net: Mutex::new(net), clock: Arc::clone(clock) })
    }

    /// The NIC of a virtio-net of type `typename`, joined to netdev `netdev` if given, as
    /// `set_netdev()` and `qemu_new_nic()` make it. A zero `mac` gets the next default address.
    /// Returns the peer for the device and the port to [`NicPort::connect`] once it is plugged.
    pub(crate) fn new_nic(
        &self,
        typename: &str,
        id: Option<&str>,
        netdev: Option<&str>,
        mac: &mut [u8; 6],
    ) -> Result<(Box<dyn NetPeer>, NicPort), String> {
        let mut net = lock(&self.net);
        let mut peers = Vec::new();
        if let Some(name) = netdev {
            peers = net.find_clients_except(name, Some(NetClientDriver::Nic));
            if peers.is_empty() {
                return Err(format!("Property '{typename}.netdev' can't find value '{name}'"));
            }
            // The device has one queue pair, so it takes one peer.
            peers.truncate(1);
            if peers[0].peer().is_some() {
                return Err(format!(
                    "Property '{typename}.netdev' can't take value '{name}', it's in use"
                ));
            }
        }
        let mut addr = MacAddr(*mac);
        net.macaddr_default_if_unset(&mut addr);
        *mac = addr.0;
        let inner = Arc::new(NicInner {
            handle: OnceLock::new(),
            queue: Mutex::new(VecDeque::new()),
            refused: AtomicBool::new(false),
            clock: Arc::clone(&self.clock),
            timer: OnceLock::new(),
            nc: OnceLock::new(),
        });
        let conf = NicConf { macaddr: addr, peers };
        let ops = Arc::clone(&inner);
        let nic = net.new_nic(&conf, typename, id, move |w, _| {
            let _ = ops.nc.set(w.clone());
            Arc::new(NicOps(Arc::clone(&ops)))
        });
        let weak = Arc::downgrade(&inner);
        let timer = self.clock.new_timer(move || {
            if let Some(nic) = weak.upgrade() {
                nic.run();
            }
        });
        let _ = inner.timer.set(timer);
        let peer = NicPeer { nc: Arc::clone(nic.queue()), inner: Arc::clone(&inner) };
        Ok((Box::new(peer), NicPort(inner)))
    }

    /// `net_check_clients()`: warns about the clients left without a peer.
    pub(crate) fn check_clients(&self) {
        lock(&self.net).check_clients();
    }

    /// `hmp_info_network()`.
    pub(crate) fn info_network(&self) -> String {
        lock(&self.net).info_network()
    }

    /// `net_vm_change_state_handler()`: a stopped VM sends and takes nothing.
    pub(crate) fn vm_state_change(&self, running: bool) {
        lock(&self.net).vm_state_change(running);
    }
}

/// A NIC made by [`Network::new_nic`], waiting for its device.
#[derive(Debug)]
pub(crate) struct NicPort(Arc<NicInner>);

impl NicPort {
    /// Delivers to the plugged device from now on.
    pub(crate) fn connect(&self, handle: VirtioHandle) {
        let _ = self.0.handle.set(handle);
        self.0.kick();
    }
}

#[derive(Debug)]
struct NicInner {
    handle: OnceLock<VirtioHandle>,
    queue: Mutex<VecDeque<Vec<u8>>>,
    /// A frame was turned away and waits in ruvm-net for a flush.
    refused: AtomicBool,
    clock: Arc<Clock>,
    timer: OnceLock<Timer>,
    nc: OnceLock<Weak<NetClient>>,
}

impl NicInner {
    fn kick(&self) {
        if let Some(t) = self.timer.get() {
            t.modify(self.clock.get_ns());
        }
    }

    /// Hands the waiting frames to the device while it has buffers for them.
    fn run(&self) {
        let Some(nic) = self.handle.get() else { return };
        loop {
            let Some(frame) = lock(&self.queue).pop_front() else { break };
            match nic.with_device(|vdev, n: &mut VirtioNet| n.receive(vdev, &frame)) {
                // Wait for rx_ready().
                Some(RxOutcome::NoBuffers) | None => {
                    lock(&self.queue).push_front(frame);
                    break;
                }
                // Delivered, or dropped by a NIC that is not ready, as in QEMU.
                Some(_) => {}
            }
        }
        if lock(&self.queue).len() < NIC_QUEUE_MAX && self.refused.swap(false, Ordering::SeqCst) {
            if let Some(nc) = self.nc.get().and_then(Weak::upgrade) {
                nc.flush_queued_packets();
            }
        }
    }
}

/// What ruvm-net calls with the frames for the NIC.
struct NicOps(Arc<NicInner>);

impl NetClientOps for NicOps {
    fn receive(&self, _nc: &NetClient, iov: &[&[u8]]) -> isize {
        let mut q = lock(&self.0.queue);
        if q.len() >= NIC_QUEUE_MAX {
            self.0.refused.store(true, Ordering::SeqCst);
            return 0;
        }
        let frame = iov.concat();
        let len = frame.len() as isize;
        q.push_back(frame);
        drop(q);
        self.0.kick();
        len
    }

    fn can_receive(&self, _nc: &NetClient) -> bool {
        let room = lock(&self.0.queue).len() < NIC_QUEUE_MAX;
        if !room {
            self.0.refused.store(true, Ordering::SeqCst);
        }
        room
    }
}

/// The device's side: the frames the guest sends go to the netdev.
#[derive(Debug)]
struct NicPeer {
    nc: Arc<NetClient>,
    inner: Arc<NicInner>,
}

impl NetPeer for NicPeer {
    fn send(&mut self, _pair: u16, _hdr: Option<&VirtioNetHdr>, frame: &[u8]) {
        self.nc.send_packet(frame);
    }

    fn rx_ready(&mut self, _pair: u16) {
        self.inner.kick();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(a: &[&str]) -> Vec<(String, Option<Location>)> {
        a.iter().map(|s| (s.to_string(), None)).collect()
    }

    #[test]
    fn nics_join_their_netdevs() {
        let clock = Clock::manual(ruvm_base::ClockType::Virtual);
        let net =
            Network::new(&args(&["hubport,id=a,hubid=0", "hubport,id=b,hubid=0"]), &clock).unwrap();
        let mut mac = [0; 6];
        let (_, port) = net.new_nic("virtio-net-pci", None, Some("a"), &mut mac).unwrap();
        assert_eq!(mac, [0x52, 0x54, 0, 0x12, 0x34, 0x56]);
        let err = net.new_nic("virtio-net-pci", None, Some("a"), &mut [0; 6]).unwrap_err();
        assert_eq!(err, "Property 'virtio-net-pci.netdev' can't take value 'a', it's in use");
        let err = net.new_nic("virtio-net-pci", None, Some("c"), &mut [0; 6]).unwrap_err();
        assert_eq!(err, "Property 'virtio-net-pci.netdev' can't find value 'c'");
        let mut mac = [0; 6];
        let (mut b, _) = net.new_nic("virtio-net-pci", None, Some("b"), &mut mac).unwrap();
        assert_eq!(mac[5], 0x57);

        // Nothing moves while the VM is stopped.
        b.send(0, None, &[0xff; 60]);
        assert!(lock(&port.0.queue).is_empty());
        // Then the frame reaches a's queue, where it waits for a device.
        net.vm_state_change(true);
        assert_eq!(lock(&port.0.queue).len(), 1);
        port.0.run();
        assert_eq!(lock(&port.0.queue).len(), 1);

        assert!(Network::new(&args(&["bogus,id=x"]), &clock).is_err());
    }
}
