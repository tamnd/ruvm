// SPDX-License-Identifier: GPL-2.0-or-later

//! Event rate limiting, `monitor_qapi_event_queue_no_reenter()` and its timer in
//! monitor/monitor.c.
//!
//! Guest triggerable events are limited to one per second per key. The first event in a quiet
//! period goes out at once and starts the period. Events during the period replace one pending
//! slot, and when the period ends the pending event goes out and starts a new period. A period
//! that ends with nothing pending ends the state. The key is the event, plus one data member for
//! the events where QEMU hashes one in.
//!
//! Time is passed in, in nanoseconds, so the caller picks the clock: realtime normally and the
//! virtual clock under qtest, as `monitor_get_event_clock()` does.

use std::collections::HashMap;

use ruvm_qapi::events::QapiEvent;
use ruvm_qapi::{QDict, QValue};

const SCALE_MS: i64 = 1_000_000;

/// `monitor_qapi_event_conf`: the period of each throttled event, in nanoseconds.
pub fn event_rate(event: QapiEvent) -> Option<i64> {
    match event {
        QapiEvent::RtcChange
        | QapiEvent::BlockIoError
        | QapiEvent::Watchdog
        | QapiEvent::BalloonChange
        | QapiEvent::QuorumReportBad
        | QapiEvent::QuorumFailure
        | QapiEvent::VserportChange
        | QapiEvent::MemoryDeviceSizeChange
        | QapiEvent::HvBalloonStatusReport => Some(1000 * SCALE_MS),
        _ => None,
    }
}

/// The data member that is part of the throttle key, `qapi_event_throttle_hash()`.
fn key_member(event: QapiEvent) -> Option<&'static str> {
    match event {
        QapiEvent::VserportChange => Some("id"),
        QapiEvent::QuorumReportBad => Some("node-name"),
        QapiEvent::MemoryDeviceSizeChange | QapiEvent::BlockIoError => Some("qom-path"),
        _ => None,
    }
}

fn data(qdict: &QDict) -> Option<&QDict> {
    match qdict.get("data") {
        Some(QValue::Dict(d)) => Some(d),
        _ => None,
    }
}

type Key = (QapiEvent, Option<String>);

#[derive(Debug)]
struct State {
    deadline: i64,
    pending: Option<QDict>,
}

/// The throttle state of every event key that sent something within its period.
#[derive(Debug, Default)]
pub struct EventThrottle {
    states: HashMap<Key, State>,
}

impl EventThrottle {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes an event at time `now` and returns it if it goes out now. Otherwise it is kept
    /// until [`EventThrottle::expire`] sends it.
    pub fn queue(&mut self, event: QapiEvent, qdict: QDict, now: i64) -> Option<QDict> {
        let Some(rate) = event_rate(event) else {
            return Some(qdict);
        };
        // BLOCK_IO_ERROR stops the VM when the action is "stop", and the management tool has
        // to see every one of those.
        if event == QapiEvent::BlockIoError
            && data(&qdict).and_then(|d| d.get_str("action")) == Some("stop")
        {
            return Some(qdict);
        }
        let member = key_member(event)
            .and_then(|m| data(&qdict).and_then(|d| d.get_str(m)))
            .map(str::to_string);
        let key = (event, member);
        if let Some(state) = self.states.get_mut(&key) {
            state.pending = Some(qdict);
            None
        } else {
            self.states.insert(key, State { deadline: now + rate, pending: None });
            Some(qdict)
        }
    }

    /// Ends every period that is over at `now`, `monitor_qapi_event_handler()`, and returns the
    /// events that go out.
    pub fn expire(&mut self, now: i64) -> Vec<QDict> {
        let mut out = Vec::new();
        let mut due: Vec<(Key, i64)> = self
            .states
            .iter()
            .filter(|(_, s)| s.deadline <= now)
            .map(|(k, s)| (k.clone(), s.deadline))
            .collect();
        // Timers fire in deadline order.
        due.sort_by_key(|(_, d)| *d);
        for (key, _) in due {
            let state = self.states.get_mut(&key).expect("collected above");
            match state.pending.take() {
                Some(qdict) => {
                    out.push(qdict);
                    let rate = event_rate(key.0).expect("only throttled events have state");
                    state.deadline = now + rate;
                }
                None => {
                    self.states.remove(&key);
                }
            }
        }
        out
    }

    /// When [`EventThrottle::expire`] next has work.
    pub fn next_deadline(&self) -> Option<i64> {
        self.states.values().map(|s| s.deadline).min()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(name: &str, data: QDict) -> QDict {
        QDict::new().with("event", name).with("data", data)
    }

    const S: i64 = 1000 * SCALE_MS;

    #[test]
    fn unthrottled_events_pass() {
        let mut t = EventThrottle::new();
        for _ in 0..3 {
            assert!(t.queue(QapiEvent::Stop, ev("STOP", QDict::new()), 0).is_some());
        }
        assert_eq!(t.next_deadline(), None);
    }

    #[test]
    fn first_goes_out_last_is_kept() {
        let mut t = EventThrottle::new();
        let rtc = |n: i64| ev("RTC_CHANGE", QDict::new().with("offset", n));
        assert!(t.queue(QapiEvent::RtcChange, rtc(1), 0).is_some());
        assert!(t.queue(QapiEvent::RtcChange, rtc(2), 10).is_none());
        assert!(t.queue(QapiEvent::RtcChange, rtc(3), 20).is_none());
        assert_eq!(t.next_deadline(), Some(S));
        assert!(t.expire(S - 1).is_empty());
        let out = t.expire(S);
        assert_eq!(out, vec![rtc(3)]);
        // A new period started, and it ends with nothing pending.
        assert_eq!(t.next_deadline(), Some(2 * S));
        assert!(t.expire(2 * S).is_empty());
        assert_eq!(t.next_deadline(), None);
        assert!(t.queue(QapiEvent::RtcChange, rtc(4), 2 * S).is_some());
    }

    #[test]
    fn keys_include_a_member() {
        let mut t = EventThrottle::new();
        let port = |id: &str| ev("VSERPORT_CHANGE", QDict::new().with("id", id).with("open", true));
        assert!(t.queue(QapiEvent::VserportChange, port("a"), 0).is_some());
        assert!(t.queue(QapiEvent::VserportChange, port("b"), 0).is_some());
        assert!(t.queue(QapiEvent::VserportChange, port("a"), 1).is_none());
    }

    #[test]
    fn block_io_error_stop_is_never_throttled() {
        let mut t = EventThrottle::new();
        let err = |action: &str| {
            ev("BLOCK_IO_ERROR", QDict::new().with("qom-path", "/x").with("action", action))
        };
        assert!(t.queue(QapiEvent::BlockIoError, err("report"), 0).is_some());
        assert!(t.queue(QapiEvent::BlockIoError, err("report"), 1).is_none());
        assert!(t.queue(QapiEvent::BlockIoError, err("stop"), 2).is_some());
        assert!(t.queue(QapiEvent::BlockIoError, err("stop"), 3).is_some());
    }
}
