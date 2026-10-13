// SPDX-License-Identifier: GPL-2.0-or-later

//! Self announcement, net/announce.c.
//!
//! `announce-self` sends a RARP frame from every NIC, or from the ones it names, so that
//! switches learn where the guest is after a migration. The frames go out once right away and
//! then `rounds - 1` more times, further apart each round, on a named timer of the realtime
//! clock. There is no realtime timer list yet, so each named timer runs on its own thread.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::thread;
use std::time::{Duration, Instant};

use ruvm_qapi::types::AnnounceParameters;

use crate::client::ETH_ZLEN;
use crate::net::Nic;
use crate::util::MacAddr;

/// `ETH_P_RARP`.
const ETH_P_RARP: u16 = 0x8035;
/// `ARP_HTYPE_ETH`.
const ARP_HTYPE_ETH: u16 = 0x0001;
/// `ARP_PTYPE_IP`.
const ARP_PTYPE_IP: u16 = 0x0800;
/// `ARP_OP_REQUEST_REV`.
const ARP_OP_REQUEST_REV: u16 = 0x0003;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `announce_self_create()`: a broadcast RARP request from `mac`, padded to the shortest
/// Ethernet frame.
pub fn announce_self_create(mac: MacAddr) -> [u8; ETH_ZLEN] {
    let mut buf = [0u8; ETH_ZLEN];
    buf[0..6].fill(0xff);
    buf[6..12].copy_from_slice(&mac.0);
    buf[12..14].copy_from_slice(&ETH_P_RARP.to_be_bytes());
    buf[14..16].copy_from_slice(&ARP_HTYPE_ETH.to_be_bytes());
    buf[16..18].copy_from_slice(&ARP_PTYPE_IP.to_be_bytes());
    buf[18] = 6;
    buf[19] = 4;
    buf[20..22].copy_from_slice(&ARP_OP_REQUEST_REV.to_be_bytes());
    // Sender and target hardware addresses, with both IP addresses left zero.
    buf[22..28].copy_from_slice(&mac.0);
    buf[32..38].copy_from_slice(&mac.0);
    buf
}

/// `qemu_announce_self_iter()`: announces `nic` unless `interfaces` leaves it out.
fn announce_nic(nic: &Nic, interfaces: Option<&[String]>) {
    let nc = nic.queue();
    if interfaces.is_some_and(|names| !names.iter().any(|n| n == nc.name())) {
        return;
    }
    nc.send_packet_raw(&announce_self_create(nic.macaddr()));
    // A NIC that can have the guest announce itself does that as well.
    nc.ops().announce(nc);
}

/// `qemu_announce_timer_step()`: the milliseconds until the next round, when `round` rounds
/// are left.
fn step_ms(params: &AnnounceParameters, round: i64) -> u64 {
    let n = params.rounds.saturating_sub(round).saturating_sub(1);
    let mut step = params.initial.saturating_add(n.saturating_mul(params.step));
    if step < 0 || step > params.max {
        step = params.max;
    }
    step.max(0) as u64
}

/// The named announce timers, the `named_timers` of net/announce.c.
#[derive(Default)]
pub struct AnnounceTimers {
    /// Bumped each time a timer is set again, which stops the thread of the old setting.
    generations: Mutex<HashMap<String, u64>>,
    changed: Condvar,
}

impl fmt::Debug for AnnounceTimers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnounceTimers").finish_non_exhaustive()
    }
}

impl AnnounceTimers {
    pub fn new() -> Arc<AnnounceTimers> {
        Arc::default()
    }

    /// `qmp_announce_self()`: (re)sets the timer named by `params.id` and announces the NIC
    /// that `nics` returns at each round. `rounds` of 0 only stops the timer.
    pub fn announce_self<F>(self: &Arc<Self>, params: AnnounceParameters, nics: F)
    where
        F: Fn() -> Vec<Arc<Nic>> + Send + 'static,
    {
        let id = params.id.clone().unwrap_or_default();
        let generation = {
            let mut gens = lock(&self.generations);
            let g = gens.entry(id.clone()).or_default();
            *g += 1;
            *g
        };
        self.changed.notify_all();
        if params.rounds == 0 {
            return;
        }
        let once = move |params: &AnnounceParameters| {
            for nic in nics() {
                announce_nic(&nic, params.interfaces.as_deref());
            }
        };
        once(&params);
        let weak = Arc::downgrade(self);
        // Without a thread only the first round goes out.
        let _ = thread::Builder::new().name("announce".into()).spawn(move || {
            let mut round = params.rounds;
            loop {
                round -= 1;
                if round == 0 {
                    break;
                }
                let deadline = Instant::now() + Duration::from_millis(step_ms(&params, round));
                if !wait(&weak, &id, generation, deadline) {
                    break;
                }
                once(&params);
            }
        });
    }
}

/// Sleeps until `deadline`. False if the timer was set again first, or is gone.
fn wait(timers: &Weak<AnnounceTimers>, id: &str, generation: u64, deadline: Instant) -> bool {
    let Some(timers) = timers.upgrade() else { return false };
    let mut gens = lock(&timers.generations);
    loop {
        if gens.get(id) != Some(&generation) {
            return false;
        }
        let now = Instant::now();
        if now >= deadline {
            return true;
        }
        gens = timers
            .changed
            .wait_timeout(gens, deadline - now)
            .unwrap_or_else(PoisonError::into_inner)
            .0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(rounds: i64) -> AnnounceParameters {
        AnnounceParameters {
            initial: 20,
            max: 100,
            rounds,
            step: 10,
            interfaces: None,
            id: Some("bob".into()),
        }
    }

    #[test]
    fn rarp_frame() {
        let mac = MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]);
        let buf = announce_self_create(mac);
        assert_eq!(&buf[0..6], &[0xff; 6]);
        assert_eq!(&buf[6..12], &mac.0);
        assert_eq!(&buf[12..22], &[0x80, 0x35, 0, 1, 8, 0, 6, 4, 0, 3]);
        assert_eq!(&buf[22..28], &mac.0);
        assert_eq!(&buf[28..32], &[0; 4]);
        assert_eq!(&buf[32..38], &mac.0);
        assert!(buf[38..].iter().all(|&b| b == 0));
    }

    #[test]
    fn steps() {
        let p = params(300);
        // The second round waits `initial`, then each one waits `step` more, up to `max`.
        assert_eq!(step_ms(&p, 299), 20);
        assert_eq!(step_ms(&p, 298), 30);
        assert_eq!(step_ms(&p, 291), 100);
        assert_eq!(step_ms(&p, 1), 100);
        let p = AnnounceParameters { step: -50, ..params(5) };
        assert_eq!(step_ms(&p, 3), 100);
    }

    #[test]
    fn zero_rounds_stops_the_timer() {
        let t = AnnounceTimers::new();
        let calls = Arc::new(Mutex::new(0));
        let c = Arc::clone(&calls);
        t.announce_self(params(300), move || {
            *lock(&c) += 1;
            Vec::new()
        });
        thread::sleep(Duration::from_millis(60));
        t.announce_self(params(0), Vec::new);
        let n = *lock(&calls);
        assert!(n >= 2, "{n}");
        thread::sleep(Duration::from_millis(150));
        assert_eq!(*lock(&calls), n);
    }
}
