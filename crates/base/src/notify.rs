// SPDX-License-Identifier: MIT OR Apache-2.0

//! Notifier lists, QEMU's `NotifierList` from util/notify.c. Machine init done, reset, shutdown,
//! migration state changes and memory listeners all hand out one of these.
//!
//! QEMU inserts new notifiers at the head of the list, so they run in the reverse of the order they
//! were added. Code ported from QEMU sometimes depends on that, so the same order is kept here.

use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NotifierId(u64);

type Callback<T> = Box<dyn FnMut(&T) + Send>;

pub struct NotifierList<T> {
    inner: Mutex<Inner<T>>,
}

struct Inner<T> {
    next: u64,
    // Newest first, as in QEMU.
    entries: Vec<(NotifierId, Callback<T>)>,
}

impl<T> std::fmt::Debug for NotifierList<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NotifierList").field("len", &self.len()).finish()
    }
}

impl<T> Default for NotifierList<T> {
    fn default() -> Self {
        NotifierList { inner: Mutex::new(Inner { next: 0, entries: Vec::new() }) }
    }
}

impl<T> NotifierList<T> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `notifier_list_add()`.
    pub fn add(&self, f: impl FnMut(&T) + Send + 'static) -> NotifierId {
        let mut inner = self.lock();
        let id = NotifierId(inner.next);
        inner.next += 1;
        inner.entries.insert(0, (id, Box::new(f)));
        id
    }

    /// `notifier_remove()`. Returns false if the notifier was already gone.
    pub fn remove(&self, id: NotifierId) -> bool {
        let mut inner = self.lock();
        let before = inner.entries.len();
        inner.entries.retain(|(i, _)| *i != id);
        inner.entries.len() != before
    }

    /// `notifier_list_notify()`. Notifiers run with the list locked, so a notifier must not add
    /// to or remove from the list it is on. QEMU forbids that too, it just does not check.
    pub fn notify(&self, data: &T) {
        let mut inner = self.lock();
        for (_, f) in inner.entries.iter_mut() {
            f(data);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.lock().entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner<T>> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::NotifierList;
    use std::sync::{Arc, Mutex};

    #[test]
    fn newest_runs_first_and_removal_works() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let list = NotifierList::<u32>::new();
        let ids: Vec<_> = (0..3)
            .map(|n| {
                let seen = Arc::clone(&seen);
                list.add(move |v| seen.lock().unwrap().push((n, *v)))
            })
            .collect();
        list.notify(&7);
        assert_eq!(*seen.lock().unwrap(), [(2, 7), (1, 7), (0, 7)]);
        assert!(list.remove(ids[1]));
        assert!(!list.remove(ids[1]));
        seen.lock().unwrap().clear();
        list.notify(&8);
        assert_eq!(*seen.lock().unwrap(), [(2, 8), (0, 8)]);
    }
}
