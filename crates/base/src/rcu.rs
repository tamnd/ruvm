// SPDX-License-Identifier: MIT OR Apache-2.0

//! Read copy update for data that is read on every guest access and replaced rarely, such as the
//! FlatView of an address space and the translation block cache.
//!
//! A reader takes a snapshot, which is an `Arc` of the current value, and keeps using it for as long
//! as it likes. A writer builds a new value and publishes it, and the old one is freed when the last
//! reader drops its snapshot. That is the grace period QEMU's `call_rcu()` waits for, expressed as
//! a reference count.
//!
//! This version publishes through a lock that readers hold only long enough to clone the `Arc`. It
//! is correct and has no unsafe code. The epoch based version described in spec/03 replaces the
//! inside of this type without changing its interface, once there is a benchmark that shows the
//! reader side matters.

use std::sync::{Arc, RwLock};

pub struct Rcu<T> {
    current: RwLock<Arc<T>>,
}

impl<T> Rcu<T> {
    pub fn new(value: T) -> Self {
        Rcu { current: RwLock::new(Arc::new(value)) }
    }

    /// A snapshot of the current value. It stays valid however many times the value is replaced.
    pub fn read(&self) -> Arc<T> {
        Arc::clone(&self.current.read().unwrap_or_else(|p| p.into_inner()))
    }

    /// Publishes a new value and returns the old one. Readers that already hold a snapshot keep
    /// seeing the old value, new readers see the new one.
    pub fn replace(&self, value: T) -> Arc<T> {
        self.publish(Arc::new(value))
    }

    pub fn publish(&self, value: Arc<T>) -> Arc<T> {
        let mut cur = self.current.write().unwrap_or_else(|p| p.into_inner());
        std::mem::replace(&mut *cur, value)
    }

    /// Builds the next value from the current one and publishes it. Writers are serialized, so an
    /// update is never lost to a concurrent one.
    pub fn update(&self, f: impl FnOnce(&T) -> T) -> Arc<T> {
        let mut cur = self.current.write().unwrap_or_else(|p| p.into_inner());
        let next = Arc::new(f(&cur));
        std::mem::replace(&mut *cur, next)
    }
}

impl<T: Default> Default for Rcu<T> {
    fn default() -> Self {
        Rcu::new(T::default())
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Rcu<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Rcu").field(&self.read()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Rcu;
    use std::sync::Arc;

    #[test]
    fn readers_keep_their_snapshot() {
        let r = Rcu::new(vec![1, 2, 3]);
        let old = r.read();
        r.replace(vec![4]);
        assert_eq!(*old, vec![1, 2, 3]);
        assert_eq!(*r.read(), vec![4]);
    }

    #[test]
    fn concurrent_updates_are_not_lost() {
        let r = Arc::new(Rcu::new(0u64));
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let r = Arc::clone(&r);
                std::thread::spawn(move || {
                    for _ in 0..1000 {
                        r.update(|v| v + 1);
                        assert!(*r.read() > 0);
                    }
                })
            })
            .collect();
        threads.into_iter().for_each(|t| t.join().unwrap());
        assert_eq!(*r.read(), 8000);
    }
}
