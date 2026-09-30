// SPDX-License-Identifier: MIT OR Apache-2.0

//! The timer list for one clock, the data structure behind `QEMUTimerList` in util/qemu-timer.c.
//!
//! QEMU keeps a sorted linked list per clock, which makes arming O(n). This is a binary heap with
//! lazy deletion instead. Every arm pushes an entry tagged with the timer's generation, and entries
//! whose generation is stale are skipped when they reach the top. Timers live in a slab and keep
//! their slot, so rearming a timer never allocates once the heap has grown to its working size.
//!
//! The list does not own a clock. The reactor asks it for the next deadline, reads its clock, and
//! calls `expire()` with the current time.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// QEMU's four clocks. Their meanings matter to device models, see spec/03.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClockType {
    /// Monotonic host time. Runs while the VM is stopped.
    Realtime,
    /// Guest time. Stops with the VM and follows icount when it is on.
    Virtual,
    /// Host wall clock time. May jump.
    Host,
    /// Realtime used for icount warps. Equal to `Virtual` without icount.
    VirtualRt,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimerId(u32);

struct Slot<C> {
    generation: u32,
    deadline: Option<i64>,
    callback: Option<C>,
}

pub struct TimerList<C> {
    slots: Vec<Slot<C>>,
    free: Vec<u32>,
    // (deadline, arm sequence, slot, generation), smallest deadline first. Equal deadlines fire in
    // the order they were armed, which is what QEMU's sorted list does.
    heap: BinaryHeap<Reverse<(i64, u64, u32, u32)>>,
    seq: u64,
}

impl<C> std::fmt::Debug for TimerList<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let armed: Vec<_> =
            self.slots.iter().enumerate().filter_map(|(i, s)| s.deadline.map(|d| (i, d))).collect();
        f.debug_struct("TimerList")
            .field("timers", &(self.slots.len() - self.free.len()))
            .field("armed", &armed)
            .finish()
    }
}

impl<C> Default for TimerList<C> {
    fn default() -> Self {
        TimerList { slots: Vec::new(), free: Vec::new(), heap: BinaryHeap::new(), seq: 0 }
    }
}

impl<C> TimerList<C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `timer_new()`: makes a timer that is not armed.
    pub fn insert(&mut self, callback: C) -> TimerId {
        if let Some(i) = self.free.pop() {
            let slot = &mut self.slots[i as usize];
            slot.callback = Some(callback);
            slot.deadline = None;
            TimerId(i)
        } else {
            self.slots.push(Slot { generation: 0, deadline: None, callback: Some(callback) });
            TimerId(self.slots.len() as u32 - 1)
        }
    }

    /// `timer_free()`. The slot is reused by a later insert.
    pub fn remove(&mut self, id: TimerId) -> Option<C> {
        let slot = self.slots.get_mut(id.0 as usize)?;
        let cb = slot.callback.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        slot.deadline = None;
        self.free.push(id.0);
        Some(cb)
    }

    /// `timer_mod()`: arms the timer for `deadline`, replacing any earlier arming.
    pub fn arm(&mut self, id: TimerId, deadline: i64) {
        let slot = &mut self.slots[id.0 as usize];
        assert!(slot.callback.is_some(), "arming a freed timer");
        slot.generation = slot.generation.wrapping_add(1);
        slot.deadline = Some(deadline);
        self.seq += 1;
        self.heap.push(Reverse((deadline, self.seq, id.0, slot.generation)));
        // Stale entries pile up when a timer is rearmed much more often than it fires. Rebuild once
        // they outnumber live timers by a wide margin so the heap stays bounded.
        if self.heap.len() > 64 && self.heap.len() > 4 * self.slots.len() {
            self.compact();
        }
    }

    /// `timer_mod_anticipate()`: arms the timer only if that makes it fire sooner.
    pub fn arm_earlier(&mut self, id: TimerId, deadline: i64) {
        match self.slots[id.0 as usize].deadline {
            Some(d) if d <= deadline => {}
            _ => self.arm(id, deadline),
        }
    }

    /// `timer_del()`.
    pub fn disarm(&mut self, id: TimerId) {
        let slot = &mut self.slots[id.0 as usize];
        if slot.deadline.take().is_some() {
            slot.generation = slot.generation.wrapping_add(1);
        }
    }

    /// `timer_pending()`.
    pub fn is_armed(&self, id: TimerId) -> bool {
        self.slots.get(id.0 as usize).is_some_and(|s| s.deadline.is_some())
    }

    /// `timer_expire_time_ns()`.
    pub fn deadline_of(&self, id: TimerId) -> Option<i64> {
        self.slots.get(id.0 as usize).and_then(|s| s.deadline)
    }

    /// The earliest armed deadline, `timerlist_deadline_ns()` without the subtraction.
    pub fn next_deadline(&mut self) -> Option<i64> {
        self.drop_stale();
        self.heap.peek().map(|Reverse((d, ..))| *d)
    }

    /// Pops every timer whose deadline is at or before `now`, earliest first, and hands each one to
    /// `run` with its callback. The timer is disarmed before `run` sees it, so the callback can
    /// rearm it. Timers armed by `run` for a time at or before `now` also fire in this call, the
    /// way `timerlist_run_timers()` behaves.
    pub fn expire(&mut self, now: i64, mut run: impl FnMut(&mut Self, TimerId)) -> usize {
        let mut fired = 0;
        loop {
            self.drop_stale();
            match self.heap.peek() {
                Some(Reverse((d, ..))) if *d <= now => {}
                _ => return fired,
            }
            let Reverse((_, _, slot, _)) = self.heap.pop().expect("peeked");
            let s = &mut self.slots[slot as usize];
            s.deadline = None;
            s.generation = s.generation.wrapping_add(1);
            fired += 1;
            run(self, TimerId(slot));
        }
    }

    pub fn callback(&self, id: TimerId) -> Option<&C> {
        self.slots.get(id.0 as usize)?.callback.as_ref()
    }

    pub fn callback_mut(&mut self, id: TimerId) -> Option<&mut C> {
        self.slots.get_mut(id.0 as usize)?.callback.as_mut()
    }

    fn drop_stale(&mut self) {
        while let Some(Reverse((_, _, slot, generation))) = self.heap.peek() {
            let s = &self.slots[*slot as usize];
            if s.generation == *generation && s.deadline.is_some() {
                return;
            }
            self.heap.pop();
        }
    }

    fn compact(&mut self) {
        let slots = &self.slots;
        self.heap.retain(|Reverse((_, _, slot, generation))| {
            let s = &slots[*slot as usize];
            s.generation == *generation && s.deadline.is_some()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fires_in_deadline_order_and_ties_in_arm_order() {
        let mut t = TimerList::new();
        let a = t.insert('a');
        let b = t.insert('b');
        let c = t.insert('c');
        t.arm(a, 30);
        t.arm(b, 10);
        t.arm(c, 10);
        assert_eq!(t.next_deadline(), Some(10));
        let mut seen = String::new();
        t.expire(20, |t, id| seen.push(*t.callback(id).unwrap()));
        assert_eq!(seen, "bc");
        assert_eq!(t.next_deadline(), Some(30));
        assert!(t.is_armed(a) && !t.is_armed(b));
    }

    #[test]
    fn rearm_and_disarm_replace_earlier_arming() {
        let mut t = TimerList::new();
        let a = t.insert(());
        t.arm(a, 5);
        t.arm(a, 50);
        assert_eq!(t.expire(10, |_, _| panic!("fired at the stale deadline")), 0);
        t.disarm(a);
        assert_eq!(t.next_deadline(), None);
        t.arm(a, 50);
        t.arm_earlier(a, 60);
        assert_eq!(t.deadline_of(a), Some(50));
        t.arm_earlier(a, 40);
        assert_eq!(t.deadline_of(a), Some(40));
    }

    #[test]
    fn a_callback_can_rearm_its_own_timer() {
        let mut t = TimerList::new();
        let a = t.insert(0u32);
        t.arm(a, 0);
        let fired = t.expire(100, |t, id| {
            let n = t.callback_mut(id).unwrap();
            *n += 1;
            let next = *n as i64 * 30;
            t.arm(id, next);
        });
        assert_eq!(fired, 4);
        assert_eq!(t.deadline_of(a), Some(120));
    }

    #[test]
    fn heavy_rearming_keeps_the_heap_bounded() {
        let mut t = TimerList::new();
        let a = t.insert(());
        for i in 0..100_000 {
            t.arm(a, i);
        }
        assert!(t.heap.len() <= 64);
        let b = t.insert(());
        t.remove(a);
        assert_eq!(t.next_deadline(), None);
        assert_eq!(t.insert(()), a);
        assert_ne!(a, b);
    }
}
