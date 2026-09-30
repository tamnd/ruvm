// SPDX-License-Identifier: GPL-2.0-or-later

//! Draining from block/io.c: `bdrv_drained_begin()`, `bdrv_drained_end()` and
//! `bdrv_drain_all_begin()`, and the `AIO_WAIT_WHILE()` they poll with.
//!
//! A drained node has no requests in flight and its parents send no new ones until the matching
//! end. As in QEMU, draining a node quiesces its parents, recursively up to the block backends
//! and jobs, and waits until neither the node nor any parent has requests in flight. Children
//! are not quiesced: they only see what the drained node itself still sends while it finishes.
//!
//! QEMU polls the AioContext of the node until the condition turns false. Requests here
//! complete on whatever thread runs them, so [`aio_wait_while`] waits on a condition variable
//! that every completion kicks ([`aio_wait_kick`]), with a timeout as a safety net for
//! conditions that change without a kick.
//!
//! Differences from QEMU:
//!
//! - QEMU runs every `bdrv_drain_all_begin()` in the main loop, so two of them never
//!   overlap. Here any thread may call it, and the drain-all sections of different threads
//!   are serialised: a second thread waits in [`drain_all_begin`] until the first one's
//!   section ends. Sections of one thread nest as in QEMU.
//! - [`DrainAllOwner`] holds that serialisation without draining, for tests that count
//!   quiesce levels and must not see the drain-all of a test running next to them.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::ThreadId;
use std::time::Duration;

use crate::node::Node;

static WAIT: Mutex<u64> = Mutex::new(0);
static WAIT_COND: Condvar = Condvar::new();

/// `aio_wait_kick()`: something a waiter may be waiting for happened.
pub(crate) fn aio_wait_kick() {
    *WAIT.lock().unwrap() += 1;
    WAIT_COND.notify_all();
}

/// `AIO_WAIT_WHILE()`: waits while `cond` holds.
pub(crate) fn aio_wait_while(mut cond: impl FnMut() -> bool) {
    loop {
        let g = WAIT.lock().unwrap();
        let seen = *g;
        drop(g);
        // The main loop runs its bottom halves while it waits.
        crate::job::main_loop::poll_bhs();
        if !cond() {
            return;
        }
        let g = WAIT.lock().unwrap();
        if *g == seen {
            let _ = WAIT_COND.wait_timeout(g, Duration::from_millis(10)).unwrap();
        }
    }
}

/// `bdrv_drain_all_count`: how many `bdrv_drain_all_begin()` are active. New nodes start out
/// drained that many times.
pub(crate) static DRAIN_ALL_COUNT: AtomicU32 = AtomicU32::new(0);

/// Every node there is, `all_bdrv_states`, for `bdrv_drain_all()` and `bdrv_next_all_states()`.
static ALL_STATES: Mutex<Vec<Weak<Node>>> = Mutex::new(Vec::new());

/// Held while the drain-all count changes along with the drained sections it stands for,
/// and while a new node is registered, so a new node is drained exactly as many times as
/// the drain-all sections it will see end.
static DRAIN_ALL_NODES: Mutex<()> = Mutex::new(());

/// The thread in a drain-all section and how deep, see the module documentation.
struct Owner {
    thread: Option<ThreadId>,
    depth: u32,
}

static OWNER: Mutex<Owner> = Mutex::new(Owner { thread: None, depth: 0 });
static OWNER_COND: Condvar = Condvar::new();

fn owner_acquire() {
    let me = std::thread::current().id();
    let mut o = OWNER.lock().unwrap();
    while o.thread.is_some_and(|t| t != me) {
        o = OWNER_COND.wait(o).unwrap();
    }
    o.thread = Some(me);
    o.depth += 1;
}

fn owner_release() {
    let mut o = OWNER.lock().unwrap();
    assert!(o.depth > 0 && o.thread == Some(std::thread::current().id()));
    o.depth -= 1;
    if o.depth == 0 {
        o.thread = None;
        OWNER_COND.notify_all();
    }
}

/// Keeps other threads out of drain-all sections while it lives, without draining
/// anything. The owning thread may still start drain-all sections of its own. Tests hold
/// it so that a drain-all of another test does not disturb what they count.
#[cfg(test)]
pub(crate) struct DrainAllOwner(());

#[cfg(test)]
impl DrainAllOwner {
    pub(crate) fn acquire() -> Self {
        owner_acquire();
        DrainAllOwner(())
    }
}

#[cfg(test)]
impl Drop for DrainAllOwner {
    fn drop(&mut self) {
        owner_release();
    }
}

/// Adds a new node to the list of all nodes. As `bdrv_new()` does, it starts out drained
/// once for every drain-all section in progress.
pub(crate) fn register_node(bs: &Arc<Node>) {
    let _g = DRAIN_ALL_NODES.lock().unwrap();
    {
        let mut all = ALL_STATES.lock().unwrap();
        all.retain(|w| w.strong_count() > 0);
        all.push(Arc::downgrade(bs));
    }
    for _ in 0..DRAIN_ALL_COUNT.load(Ordering::SeqCst) {
        bs.do_drained_begin(None, false);
    }
}

/// `bdrv_next_all_states()` as a list.
pub(crate) fn all_nodes() -> Vec<Arc<Node>> {
    ALL_STATES.lock().unwrap().iter().filter_map(Weak::upgrade).collect()
}

impl Node {
    /// `bdrv_parent_drained_begin()`: quiesce every parent but `ignore`.
    fn parent_drained_begin(&self, ignore: Option<u64>) {
        for ops in self.mark_parents_quiesced(ignore, true) {
            ops.drained_begin();
        }
    }

    /// `bdrv_parent_drained_end()`.
    fn parent_drained_end(&self, ignore: Option<u64>) {
        for ops in self.mark_parents_quiesced(ignore, false) {
            ops.drained_end();
        }
    }

    /// `bdrv_drain_poll()`: whether a parent other than `ignore` or the node itself still has
    /// requests in flight.
    pub(crate) fn drain_poll(&self, ignore: Option<u64>) -> bool {
        let busy = self
            .parents()
            .iter()
            .filter(|p| Some(p.id) != ignore)
            .filter_map(|p| p.ops.clone())
            .any(|ops| ops.drained_poll());
        busy || self.io.in_flight() > 0
    }

    /// `bdrv_do_drained_begin()`.
    pub(crate) fn do_drained_begin(&self, parent: Option<u64>, poll: bool) {
        // Stop things in parent-to-child order.
        if self.quiesce_counter.fetch_add(1, Ordering::SeqCst) == 0 {
            let _g = crate::graph_lock::rdlock();
            self.parent_drained_begin(parent);
            self.driver.drain_begin(self);
        }
        if poll {
            aio_wait_while(|| self.drain_poll(parent));
        }
    }

    /// `bdrv_do_drained_end()`.
    pub(crate) fn do_drained_end(&self, parent: Option<u64>) {
        let old = self.quiesce_counter.fetch_sub(1, Ordering::SeqCst);
        assert!(old > 0, "drained_end without drained_begin");
        // Re-enable things in child-to-parent order.
        if old == 1 {
            let _g = crate::graph_lock::rdlock();
            self.driver.drain_end(self);
            self.parent_drained_end(parent);
        }
    }

    /// `bdrv_drained_begin()`: quiesce the parents and wait for every request to finish.
    pub(crate) fn drained_begin(&self) {
        self.do_drained_begin(None, true);
    }

    /// `bdrv_drained_end()`.
    pub(crate) fn drained_end(&self) {
        self.do_drained_end(None);
    }

    /// `bs->quiesce_counter`.
    #[cfg(test)]
    pub(crate) fn quiesce_count(&self) -> u32 {
        self.quiesce_counter.load(Ordering::SeqCst)
    }

    /// A guard for a drained section, `bdrv_drained_begin()` until it is dropped.
    pub(crate) fn drained(self: &Arc<Self>) -> DrainedSection {
        self.drained_begin();
        DrainedSection(self.clone())
    }
}

/// A drained section of one node. Dropping it ends the section.
pub(crate) struct DrainedSection(Arc<Node>);

impl Drop for DrainedSection {
    fn drop(&mut self) {
        self.0.drained_end();
    }
}

/// `bdrv_drain_all_begin()`: quiesce every node, then wait for all of them.
pub(crate) fn drain_all_begin() {
    owner_acquire();
    let nodes = {
        let _g = DRAIN_ALL_NODES.lock().unwrap();
        DRAIN_ALL_COUNT.fetch_add(1, Ordering::SeqCst);
        let nodes = all_nodes();
        for bs in &nodes {
            bs.do_drained_begin(None, false);
        }
        nodes
    };
    aio_wait_while(|| nodes.iter().any(|bs| bs.io.in_flight() > 0 || all_parents_busy(bs)));
}

/// `bdrv_drain_poll(bs, NULL, true)`: the parents that are not nodes.
fn all_parents_busy(bs: &Node) -> bool {
    bs.parents()
        .iter()
        .filter_map(|p| p.ops.clone())
        .filter(|o| o.node().is_none())
        .any(|o| o.drained_poll())
}

/// `bdrv_drain_all_end()`.
pub(crate) fn drain_all_end() {
    {
        let _g = DRAIN_ALL_NODES.lock().unwrap();
        // Every node there is was drained by this section, when it began or when the node
        // was made.
        for bs in all_nodes() {
            bs.do_drained_end(None);
        }
        DRAIN_ALL_COUNT.fetch_sub(1, Ordering::SeqCst);
    }
    owner_release();
}

/// `bdrv_drain_all()`.
pub(crate) fn drain_all() {
    drain_all_begin();
    drain_all_end();
}

#[cfg(test)]
mod tests;
