// SPDX-License-Identifier: GPL-2.0-or-later

//! Throttle groups from block/throttle-groups.c: members that share one [`ThrottleState`] and
//! take turns, round robin, when they are throttled.
//!
//! Groups live in one registry for the process, as QEMU's `throttle_groups` list does, and are
//! looked up by name. A group is created by `object-add throttle-group` ([`throttle_group_add`])
//! or on the fly when a member registers with a name no group has yet (what `-drive
//! throttling.group=...` does). The registry functions are free functions; [`BlockGraph`] has
//! thin wrappers for the monitor.
//!
//! Differences from QEMU:
//!
//! - Requests are synchronous. A throttled request blocks its thread on the group's condition
//!   variable instead of a coroutine queue. Waiters are woken in the order they queued.
//! - There is no event loop to fire the timers. A thread that waits in a group fires any timer
//!   of the group that has expired, running `timer_cb()` itself, and sleeps at most until the
//!   next deadline (in slices of 10 ms, so that a [`VirtualClock`](super::VirtualClock) moved
//!   from another thread is noticed). A timer is only armed while some member of the group has
//!   a request waiting, so there is always a thread to fire it.
//! - `throttle_group_restart_queue()` runs the restart at once, under the group lock, rather
//!   than in a new coroutine; `restart_pending` is not needed.
//! - The properties of `object-add` are applied `x-*` first, in schema order, then `limits`.
//!   QEMU applies them in the order of its option dictionary.
//! - `qtest` has no say over the clock: groups use the real clock, or the one a test passes
//!   to [`throttle_group_add_with_clock`].

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::{ThrottleGroupProperties, ThrottleLimits};

use super::{
    BucketType, Clock, THROTTLE_MAX, ThrottleConfig, ThrottleDirection, ThrottleState,
    ThrottleTimers, realtime_clock,
};
use crate::graph::BlockGraph;

/// The longest a waiting thread sleeps before it looks at the clock again.
const WAIT_SLICE: Duration = Duration::from_millis(10);

/// The group side of a `ThrottleGroupMember`, protected by the group lock.
#[derive(Debug)]
struct Slot {
    id: u64,
    io_limits_disabled: Arc<AtomicU32>,
    pending_reqs: [u32; THROTTLE_MAX],
    /// `throttled_reqs[]`: the tickets handed out to waiters, and how many were woken. The
    /// queue is empty when both are equal.
    queued: [u64; THROTTLE_MAX],
    served: [u64; THROTTLE_MAX],
    timers: ThrottleTimers,
}

/// The part of `ThrottleGroup` that `tg->lock` protects.
#[derive(Debug)]
struct GroupState {
    ts: ThrottleState,
    /// `head`: newest member first.
    members: Vec<Slot>,
    tokens: [Option<u64>; THROTTLE_MAX],
    any_timer_armed: [bool; THROTTLE_MAX],
}

/// `ThrottleGroup`.
#[derive(Debug)]
pub struct ThrottleGroup {
    name: String,
    clock: Arc<dyn Clock>,
    lock: Mutex<GroupState>,
    cond: Condvar,
}

#[derive(Debug)]
struct Entry {
    group: Arc<ThrottleGroup>,
    /// The QOM reference count: one for the object of `object-add`, one per member.
    refs: usize,
    /// Made by `object-add`, so `object-del` may remove it.
    user_created: bool,
}

/// `throttle_groups`.
static GROUPS: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

fn next_member_id() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl GroupState {
    fn idx(&self, id: u64) -> usize {
        self.members.iter().position(|s| s.id == id).expect("the member is in its group")
    }

    fn slot(&mut self, id: u64) -> &mut Slot {
        let i = self.idx(id);
        &mut self.members[i]
    }

    /// `throttle_group_next_tgm()`: the next member, as in a circular list.
    fn next_tgm(&self, id: u64) -> u64 {
        let i = self.idx(id) + 1;
        self.members[if i == self.members.len() { 0 } else { i }].id
    }

    /// `tgm_has_pending_reqs()`.
    fn has_pending(&self, id: u64, d: usize) -> bool {
        self.members[self.idx(id)].pending_reqs[d] > 0
    }

    /// `next_throttle_token()`: the next member in the round robin with pending requests,
    /// or `id` if there is none.
    fn next_throttle_token(&self, id: u64, d: usize) -> u64 {
        // A member being drained skips the round robin, or it could wait for the throttled
        // requests of the others.
        let me = &self.members[self.idx(id)];
        if me.pending_reqs[d] > 0 && me.io_limits_disabled.load(Ordering::SeqCst) > 0 {
            return id;
        }

        let start = self.tokens[d].expect("a group with members has a token");
        let mut token = self.next_tgm(start);
        while token != start && !self.has_pending(token, d) {
            token = self.next_tgm(token);
        }

        // With nothing queued, the token goes to this member, which most likely has the
        // request that is being queued.
        if token == start && !self.has_pending(token, d) {
            token = id;
        }
        debug_assert!(token == id || self.has_pending(token, d));
        token
    }

    /// `throttle_group_schedule_timer()`: whether the next request of `id` must wait. Arms a
    /// timer and hands `id` the token if none is armed in the group yet.
    fn schedule_timer(&mut self, id: u64, dir: ThrottleDirection) -> bool {
        let d = dir as usize;
        let i = self.idx(id);
        if self.members[i].io_limits_disabled.load(Ordering::SeqCst) > 0 {
            return false;
        }
        if self.any_timer_armed[d] {
            return true;
        }
        let GroupState { ts, members, .. } = self;
        let must_wait = ts.schedule_timer(&mut members[i].timers, dir);
        if must_wait {
            self.tokens[d] = Some(id);
            self.any_timer_armed[d] = true;
        }
        must_wait
    }

    /// `throttle_group_co_restart_queue()`: wakes the first waiter of `id`, if any.
    fn restart_queue(&mut self, id: u64, d: usize) -> bool {
        let s = self.slot(id);
        if s.served[d] < s.queued[d] {
            s.served[d] += 1;
            true
        } else {
            false
        }
    }

    /// `schedule_next_request()`, always from a request's context (QEMU's coroutine case).
    fn schedule_next_request(&mut self, id: u64, dir: ThrottleDirection) {
        let d = dir as usize;
        let mut token = self.next_throttle_token(id, d);
        if !self.has_pending(token, d) {
            return;
        }

        if !self.schedule_timer(token, dir) {
            // Requests of this member go first.
            if self.restart_queue(id, d) {
                token = id;
            } else {
                let s = self.slot(token);
                let now = s.timers.clock.now_ns();
                s.timers.timers[d].as_mut().expect("the throttle timer exists").modify(now);
                self.any_timer_armed[d] = true;
            }
            self.tokens[d] = Some(token);
        }
    }

    /// `throttle_group_restart_queue_entry()`.
    fn restart_queue_entry(&mut self, id: u64, dir: ThrottleDirection, reset_timer_armed: bool) {
        let d = dir as usize;
        if reset_timer_armed {
            self.any_timer_armed[d] = false;
        }
        // With nobody to wake, the next request is up to this member to schedule.
        if !self.restart_queue(id, d) {
            self.schedule_next_request(id, dir);
        }
    }

    /// Runs `timer_cb()` for every timer that expired by `now`. Returns whether any did.
    fn fire_expired(&mut self, now: i64) -> bool {
        let mut fired = false;
        loop {
            let due = self.members.iter().find_map(|s| {
                ThrottleDirection::ALL.into_iter().find_map(|dir| {
                    let t = s.timers.timers[dir as usize]?;
                    t.expire_time().filter(|&e| e <= now).map(|_| (s.id, dir))
                })
            });
            let Some((id, dir)) = due else {
                return fired;
            };
            self.slot(id).timers.timers[dir as usize].as_mut().unwrap().del();
            self.restart_queue_entry(id, dir, true);
            fired = true;
        }
    }

    /// The earliest deadline of the armed timers.
    fn next_deadline(&self) -> Option<i64> {
        self.members
            .iter()
            .flat_map(|s| s.timers.timers.iter().filter_map(|t| t.and_then(|t| t.expire_time())))
            .min()
    }
}

impl ThrottleGroup {
    fn new(name: &str, clock: Arc<dyn Clock>, cfg: &ThrottleConfig) -> Self {
        let mut ts = ThrottleState::new();
        ts.config(clock.as_ref(), cfg);
        ThrottleGroup {
            name: name.to_string(),
            clock,
            lock: Mutex::new(GroupState {
                ts,
                members: Vec::new(),
                tokens: [None; THROTTLE_MAX],
                any_timer_armed: [false; THROTTLE_MAX],
            }),
            cond: Condvar::new(),
        }
    }

    /// The group's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Waits until the waiter with `ticket` in the queue of `id` is woken, firing the
    /// group's timers as they expire.
    fn wait<'a>(
        &'a self,
        mut g: MutexGuard<'a, GroupState>,
        id: u64,
        d: usize,
        ticket: u64,
    ) -> MutexGuard<'a, GroupState> {
        loop {
            if g.slot(id).served[d] > ticket {
                return g;
            }
            let now = self.clock.now_ns();
            if g.fire_expired(now) {
                self.cond.notify_all();
                continue;
            }
            let timeout = match g.next_deadline() {
                Some(t) => Duration::from_nanos((t - now).max(0) as u64).min(WAIT_SLICE),
                None => WAIT_SLICE,
            };
            g = self.cond.wait_timeout(g, timeout).unwrap().0;
        }
    }
}

/// `throttle_group_by_name()`.
fn by_name<'a>(groups: &'a mut [Entry], name: &str) -> Option<&'a mut Entry> {
    groups.iter_mut().find(|e| e.group.name == name)
}

/// `throttle_group_exists()`.
pub fn throttle_group_exists(name: &str) -> bool {
    GROUPS.lock().unwrap().iter().any(|e| e.group.name == name)
}

/// `throttle_group_incref()`: the group `name`, created without limits if there is none.
fn incref(name: &str) -> Arc<ThrottleGroup> {
    let mut groups = GROUPS.lock().unwrap();
    if let Some(e) = by_name(&mut groups, name) {
        e.refs += 1;
        return e.group.clone();
    }
    let group = Arc::new(ThrottleGroup::new(name, realtime_clock(), &ThrottleConfig::new()));
    groups.push(Entry { group: group.clone(), refs: 1, user_created: false });
    group
}

/// `throttle_group_unref()`: the group goes with its last reference.
fn unref(group: &Arc<ThrottleGroup>) {
    let mut groups = GROUPS.lock().unwrap();
    if let Some(i) = groups.iter().position(|e| Arc::ptr_eq(&e.group, group)) {
        groups[i].refs -= 1;
        if groups[i].refs == 0 {
            groups.remove(i);
        }
    }
}

/// What `throttle_group_set()` does to one `x-*` property of a group being created.
fn set_property(cfg: &mut ThrottleConfig, name: &str, value: i64) -> Result<()> {
    if value < 0 {
        return Err(Error::generic("Property values cannot be negative"));
    }
    let Some(rest) = name.strip_prefix("x-") else {
        return Err(Error::generic(format!("Property '{name}' not found")));
    };
    if rest == "iops-size" {
        cfg.op_size = value as u64;
        return Ok(());
    }
    let (bucket, category) = match rest.split_once('-') {
        Some((kind, rest)) => {
            let (dir, category) = rest.split_once('-').unwrap_or((rest, ""));
            let t = match (kind, dir) {
                ("iops", "total") => BucketType::OpsTotal,
                ("iops", "read") => BucketType::OpsRead,
                ("iops", "write") => BucketType::OpsWrite,
                ("bps", "total") => BucketType::BpsTotal,
                ("bps", "read") => BucketType::BpsRead,
                ("bps", "write") => BucketType::BpsWrite,
                _ => return Err(Error::generic(format!("Property '{name}' not found"))),
            };
            (t, category)
        }
        None => return Err(Error::generic(format!("Property '{name}' not found"))),
    };
    let b = cfg.bucket_mut(bucket);
    match category {
        "" => b.avg = value as u64,
        "max" => b.max = value as u64,
        "max-length" => {
            if value > i64::from(u32::MAX) {
                // Sic: QEMU's message lacks the space.
                return Err(Error::generic(format!(
                    "{name} value must be in therange [0, {}]",
                    u32::MAX
                )));
            }
            b.burst_length = value as u64;
        }
        _ => return Err(Error::generic(format!("Property '{name}' not found"))),
    }
    Ok(())
}

/// The `x-*` properties of `props`, in schema order.
fn x_properties(p: &ThrottleGroupProperties) -> [(&'static str, Option<i64>); 19] {
    [
        ("x-iops-total", p.x_iops_total),
        ("x-iops-total-max", p.x_iops_total_max),
        ("x-iops-total-max-length", p.x_iops_total_max_length),
        ("x-iops-read", p.x_iops_read),
        ("x-iops-read-max", p.x_iops_read_max),
        ("x-iops-read-max-length", p.x_iops_read_max_length),
        ("x-iops-write", p.x_iops_write),
        ("x-iops-write-max", p.x_iops_write_max),
        ("x-iops-write-max-length", p.x_iops_write_max_length),
        ("x-bps-total", p.x_bps_total),
        ("x-bps-total-max", p.x_bps_total_max),
        ("x-bps-total-max-length", p.x_bps_total_max_length),
        ("x-bps-read", p.x_bps_read),
        ("x-bps-read-max", p.x_bps_read_max),
        ("x-bps-read-max-length", p.x_bps_read_max_length),
        ("x-bps-write", p.x_bps_write),
        ("x-bps-write-max", p.x_bps_write_max),
        ("x-bps-write-max-length", p.x_bps_write_max_length),
        ("x-iops-size", p.x_iops_size),
    ]
}

/// `object-add qom-type=throttle-group`: sets the properties on a new group, then
/// `throttle_group_obj_complete()`.
pub fn throttle_group_add(id: &str, props: &ThrottleGroupProperties) -> Result<()> {
    throttle_group_add_with_clock(id, props, realtime_clock())
}

/// [`throttle_group_add`] with the clock the group throttles by.
pub fn throttle_group_add_with_clock(
    id: &str,
    props: &ThrottleGroupProperties,
    clock: Arc<dyn Clock>,
) -> Result<()> {
    // throttle_group_obj_init() starts from throttle_init().
    let mut cfg = ThrottleState::new().cfg;
    for (name, v) in x_properties(props) {
        if let Some(v) = v {
            set_property(&mut cfg, name, v)?;
        }
    }
    if let Some(l) = &props.limits {
        cfg.apply_limits(l)?;
    }

    // throttle_group_obj_complete().
    let mut groups = GROUPS.lock().unwrap();
    if by_name(&mut groups, id).is_some() {
        return Err(Error::generic("A group with this name already exists"));
    }
    cfg.is_valid()?;
    let group = Arc::new(ThrottleGroup::new(id, clock, &cfg));
    groups.push(Entry { group, refs: 1, user_created: true });
    Ok(())
}

/// `object-del` of a throttle group: `throttle_group_prepare_delete()`, then the monitor's
/// reference goes.
pub fn throttle_group_del(id: &str) -> Result<()> {
    let mut groups = GROUPS.lock().unwrap();
    let Some(i) = groups.iter().position(|e| e.user_created && e.group.name == id) else {
        return Err(Error::generic(format!("object '{id}' not found")));
    };
    if groups[i].refs > 1 {
        return Err(Error::generic(format!(
            "Cannot delete throttle group '{id}' with active references"
        )));
    }
    groups.remove(i);
    Ok(())
}

fn find(name: &str) -> Option<Arc<ThrottleGroup>> {
    GROUPS.lock().unwrap().iter().find(|e| e.group.name == name).map(|e| e.group.clone())
}

/// `qom-get` of `limits`: `throttle_group_get_limits()`.
pub fn throttle_group_get_limits(name: &str) -> Option<ThrottleLimits> {
    let g = find(name)?;
    let cfg = g.lock.lock().unwrap().ts.get_config();
    Some(cfg.to_limits())
}

/// `qom-set` of `limits`: `throttle_group_set_limits()`. Unset fields keep their value.
pub fn throttle_group_set_limits(name: &str, limits: &ThrottleLimits) -> Result<()> {
    let Some(g) = find(name) else {
        return Err(Error::generic(format!("Device '{name}' not found")));
    };
    let mut st = g.lock.lock().unwrap();
    let mut cfg = st.ts.get_config();
    cfg.apply_limits(limits)?;
    st.ts.config(g.clock.as_ref(), &cfg);
    Ok(())
}

/// `qom-set` of an `x-*` property: always refused once the group exists.
pub fn throttle_group_set_property(name: &str, _property: &str, _value: i64) -> Result<()> {
    if find(name).is_none() {
        return Err(Error::generic(format!("Device '{name}' not found")));
    }
    Err(Error::generic("Property cannot be set after initialization"))
}

impl BlockGraph {
    /// `object-add qom-type=throttle-group`, see [`throttle_group_add`]. Throttle groups are
    /// global to the process, as in QEMU, not kept per graph.
    pub fn throttle_group_add(&self, id: &str, props: &ThrottleGroupProperties) -> Result<()> {
        throttle_group_add(id, props)
    }

    /// `object-del` of a throttle group, see [`throttle_group_del`].
    pub fn throttle_group_del(&self, id: &str) -> Result<()> {
        throttle_group_del(id)
    }

    /// `qom-get <group> limits`.
    pub fn throttle_group_limits(&self, id: &str) -> Option<ThrottleLimits> {
        throttle_group_get_limits(id)
    }

    /// `qom-set <group> limits`.
    pub fn throttle_group_set_limits(&self, id: &str, limits: &ThrottleLimits) -> Result<()> {
        throttle_group_set_limits(id, limits)
    }
}

#[derive(Clone, Debug)]
struct Registration {
    group: Arc<ThrottleGroup>,
    id: u64,
}

/// `ThrottleGroupMember`: what a backend or a `throttle` node holds to take part in a group.
#[derive(Debug, Default)]
pub struct ThrottleGroupMember {
    /// `throttle_state`, `None` when not in a group.
    reg: Mutex<Option<Registration>>,
    io_limits_disabled: Arc<AtomicU32>,
}

impl ThrottleGroupMember {
    /// A member of no group.
    pub fn new() -> Self {
        Self::default()
    }

    fn reg(&self) -> Option<Registration> {
        self.reg.lock().unwrap().clone()
    }

    /// Whether the member is in a group (`throttle_state != NULL`).
    pub fn is_registered(&self) -> bool {
        self.reg.lock().unwrap().is_some()
    }

    /// `throttle_group_get_name()`.
    pub fn group_name(&self) -> Option<String> {
        self.reg().map(|r| r.group.name.clone())
    }

    /// Whether both members share one group.
    pub fn same_group(&self, other: &ThrottleGroupMember) -> bool {
        match (self.reg(), other.reg()) {
            (Some(a), Some(b)) => Arc::ptr_eq(&a.group, &b.group),
            _ => false,
        }
    }

    /// `throttle_group_register_tgm()`: joins the group `groupname`, which is created if it
    /// does not exist.
    pub fn register(&self, groupname: &str) {
        let mut reg = self.reg.lock().unwrap();
        assert!(reg.is_none(), "the throttle group member is already registered");
        let group = incref(groupname);
        let id = next_member_id();
        {
            let mut st = group.lock.lock().unwrap();
            for t in &mut st.tokens {
                if t.is_none() {
                    *t = Some(id);
                }
            }
            let slot = Slot {
                id,
                io_limits_disabled: self.io_limits_disabled.clone(),
                pending_reqs: [0; THROTTLE_MAX],
                queued: [0; THROTTLE_MAX],
                served: [0; THROTTLE_MAX],
                timers: ThrottleTimers::new(group.clock.clone(), true, true),
            };
            st.members.insert(0, slot);
        }
        *reg = Some(Registration { group, id });
    }

    /// `throttle_group_unregister_tgm()`: leaves the group, which goes if this was its last
    /// reference. The member must have no requests waiting.
    pub fn unregister(&self) {
        let Some(Registration { group, id }) = self.reg.lock().unwrap().take() else {
            return;
        };
        {
            let mut st = group.lock.lock().unwrap();
            for d in 0..THROTTLE_MAX {
                let s = st.slot(id);
                assert_eq!(s.pending_reqs[d], 0);
                assert_eq!(s.served[d], s.queued[d]);
                assert!(!s.timers.timers[d].is_some_and(|t| t.pending()));
                if st.tokens[d] == Some(id) {
                    let token = st.next_tgm(id);
                    st.tokens[d] = if token == id { None } else { Some(token) };
                }
            }
            let i = st.idx(id);
            st.members.remove(i);
        }
        unref(&group);
    }

    /// `throttle_group_config()`: the new limits apply to the whole group, and the member's
    /// queue is restarted.
    pub fn config(&self, cfg: &ThrottleConfig) {
        let Some(r) = self.reg() else {
            return;
        };
        r.group.lock.lock().unwrap().ts.config(r.group.clock.as_ref(), cfg);
        self.restart();
    }

    /// `throttle_group_get_config()`.
    pub fn get_config(&self) -> Option<ThrottleConfig> {
        let r = self.reg()?;
        let cfg = r.group.lock.lock().unwrap().ts.get_config();
        Some(cfg)
    }

    /// `throttle_group_co_io_limits_intercept()`: waits until a request of `bytes` in
    /// direction `dir` may go, accounts it and schedules the next one. Returns at once when
    /// the member is in no group.
    pub fn io_limits_intercept(&self, bytes: u64, dir: ThrottleDirection) {
        let Some(Registration { group, id }) = self.reg() else {
            return;
        };
        let d = dir as usize;
        let mut st = group.lock.lock().unwrap();

        let token = st.next_throttle_token(id, d);
        let must_wait = st.schedule_timer(token, dir);

        // Wait if there is a timer set or requests of this type are queued.
        if must_wait || st.slot(id).pending_reqs[d] > 0 {
            let s = st.slot(id);
            s.pending_reqs[d] += 1;
            let ticket = s.queued[d];
            s.queued[d] += 1;
            st = group.wait(st, id, d, ticket);
            st.slot(id).pending_reqs[d] -= 1;
        }

        st.ts.account(dir, bytes);
        st.schedule_next_request(id, dir);
        drop(st);
        group.cond.notify_all();
    }

    /// `throttle_group_restart_tgm()`: lets the queued requests of the member go now.
    pub fn restart(&self) {
        let Some(Registration { group, id }) = self.reg() else {
            return;
        };
        for dir in ThrottleDirection::ALL {
            let d = dir as usize;
            let mut st = group.lock.lock().unwrap();
            let t = st.slot(id).timers.timers[d].as_mut().expect("the throttle timer exists");
            let reset_timer_armed = if t.pending() {
                // This member's timer is armed: fire it now.
                t.del();
                true
            } else if st.any_timer_armed[d] {
                // Another member's timer is armed: leave it be.
                false
            } else {
                // No timer: pretend one fires now, so that nobody arms one meanwhile.
                st.any_timer_armed[d] = true;
                true
            };
            st.restart_queue_entry(id, dir, reset_timer_armed);
        }
        group.cond.notify_all();
    }

    /// The start of a drained section: `io_limits_disabled` goes up and, the first time,
    /// the queued requests go.
    pub fn io_limits_disable_begin(&self) {
        if self.io_limits_disabled.fetch_add(1, Ordering::SeqCst) == 0 {
            self.restart();
        }
    }

    /// The end of a drained section.
    pub fn io_limits_disable_end(&self) {
        let old = self.io_limits_disabled.fetch_sub(1, Ordering::SeqCst);
        assert!(old > 0, "unbalanced end of a drained section on a throttle group member");
    }

    /// How many requests of `dir` wait.
    pub fn pending_reqs(&self, dir: ThrottleDirection) -> u32 {
        let Some(Registration { group, id }) = self.reg() else {
            return 0;
        };
        let mut st = group.lock.lock().unwrap();
        st.slot(id).pending_reqs[dir as usize]
    }
}

impl Drop for ThrottleGroupMember {
    fn drop(&mut self) {
        self.unregister();
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::*;

    /// The members of `name`'s group in list order and the token of `dir`, for tests of the
    /// round robin, with `pending` requests set per member id.
    pub(crate) fn next_token_with(
        m: &ThrottleGroupMember,
        pending: &[(&ThrottleGroupMember, u32)],
        dir: ThrottleDirection,
    ) -> u64 {
        let r = m.reg().unwrap();
        let mut st = r.group.lock.lock().unwrap();
        let d = dir as usize;
        for s in &mut st.members {
            s.pending_reqs[d] = 0;
        }
        for (p, n) in pending {
            let id = p.reg().unwrap().id;
            st.slot(id).pending_reqs[d] = *n;
        }
        let t = st.next_throttle_token(r.id, d);
        for s in &mut st.members {
            s.pending_reqs[d] = 0;
        }
        t
    }

    pub(crate) fn member_id(m: &ThrottleGroupMember) -> u64 {
        m.reg().unwrap().id
    }

    pub(crate) fn set_token(m: &ThrottleGroupMember, dir: ThrottleDirection) {
        let r = m.reg().unwrap();
        r.group.lock.lock().unwrap().tokens[dir as usize] = Some(r.id);
    }

    pub(crate) fn has_timers(m: &ThrottleGroupMember) -> bool {
        let r = m.reg().unwrap();
        let mut st = r.group.lock.lock().unwrap();
        st.slot(r.id).timers.are_initialized()
    }
}
