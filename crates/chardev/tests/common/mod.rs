// SPDX-License-Identifier: GPL-2.0-or-later

//! What the backend tests share: a frontend that remembers what it saw, like `FeHandler` in
//! QEMU's tests/unit/test-char.c, and a way to wait for the chardev threads.

#![allow(dead_code)]

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ruvm_chardev::{ChrEvent, Connection, Frontend};

/// Everything a frontend read, the events it got and how many connections it had.
#[derive(Debug, Default)]
pub(crate) struct Rec {
    pub(crate) data: Mutex<Vec<u8>>,
    pub(crate) events: Mutex<Vec<ChrEvent>>,
    pub(crate) opens: AtomicUsize,
    pub(crate) closes: AtomicUsize,
}

impl Rec {
    pub(crate) fn new() -> Arc<Rec> {
        Arc::default()
    }

    pub(crate) fn take(&self) -> Vec<u8> {
        std::mem::take(&mut *self.data.lock().unwrap())
    }

    pub(crate) fn len(&self) -> usize {
        self.data.lock().unwrap().len()
    }

    pub(crate) fn last_event(&self) -> Option<ChrEvent> {
        self.events.lock().unwrap().last().copied()
    }

    /// Waits until the frontend read `n` bytes and takes them.
    pub(crate) fn wait_bytes(&self, n: usize) -> Vec<u8> {
        wait_until(|| self.len() >= n);
        self.take()
    }
}

impl Frontend for Rec {
    fn serve(&self, conn: &mut Connection) -> io::Result<()> {
        self.opens.fetch_add(1, Ordering::SeqCst);
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                self.closes.fetch_add(1, Ordering::SeqCst);
                return Ok(());
            }
            self.data.lock().unwrap().extend_from_slice(&buf[..n]);
        }
    }

    fn event(&self, event: ChrEvent) {
        self.events.lock().unwrap().push(event);
    }
}

/// Waits up to five seconds for `cond`, and fails the test if it never holds.
pub(crate) fn wait_until(cond: impl Fn() -> bool) {
    let end = Instant::now() + Duration::from_secs(5);
    while !cond() {
        assert!(Instant::now() < end, "timed out");
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Gives the chardev threads time to do what they would do, for checks that nothing happens.
pub(crate) fn settle() {
    std::thread::sleep(Duration::from_millis(300));
}

/// A fresh directory for one test, removed when dropped.
#[derive(Debug)]
pub(crate) struct TempDir(pub(crate) PathBuf);

impl TempDir {
    pub(crate) fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("ruvm-chardev-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    pub(crate) fn path(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_string()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
