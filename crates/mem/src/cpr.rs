// SPDX-License-Identifier: MIT OR Apache-2.0

//! The file descriptors CPR hands to the next process, the fd list of `cpr_state` with
//! `cpr_save_fd()`, `cpr_find_fd()` and `cpr_delete_fd()` from migration/cpr.c.
//!
//! Shared RAM is the reason the list lives here: a block made with `share=on` or
//! `aux-ram-share=on` is a memfd, saved under the block's name when it is made. With
//! `cpr-transfer` the migration sends the whole list over a UNIX socket before anything else,
//! and the next process fills its list from it before it builds the machine, so that making the
//! same block there finds the descriptor and maps the very same memory instead of a new one.
//!
//! Like QEMU's, the list is global to the process. Each entry owns its descriptor; whoever
//! finds one gets a duplicate.

use std::os::fd::OwnedFd;
use std::sync::{Mutex, MutexGuard};

/// One saved descriptor, `CprFd`.
#[derive(Debug)]
pub struct CprFd {
    /// What the descriptor is for: the block name for RAM.
    pub name: String,
    /// Tells apart several descriptors under one name; 0 for RAM.
    pub id: i32,
    pub fd: OwnedFd,
}

static FDS: Mutex<Vec<CprFd>> = Mutex::new(Vec::new());

/// Whether this process came up from a `cpr-transfer`, `cpr_is_incoming()`.
static INCOMING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn fds() -> MutexGuard<'static, Vec<CprFd>> {
    FDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// `cpr_save_fd()`: keeps `fd` under `name` and `id`, replacing what was there.
pub fn save_fd(name: &str, id: i32, fd: OwnedFd) {
    let mut l = fds();
    l.retain(|e| e.name != name || e.id != id);
    l.push(CprFd { name: name.to_string(), id, fd });
}

/// `cpr_find_fd()`: a duplicate of the descriptor saved under `name` and `id`.
pub fn find_fd(name: &str, id: i32) -> Option<OwnedFd> {
    fds().iter().find(|e| e.name == name && e.id == id).and_then(|e| e.fd.try_clone().ok())
}

/// `cpr_delete_fd()`.
pub fn delete_fd(name: &str, id: i32) {
    fds().retain(|e| e.name != name || e.id != id);
}

/// Duplicates of every saved descriptor in the order they were saved, what `cpr_state_save()`
/// sends.
pub fn saved_fds() -> std::io::Result<Vec<CprFd>> {
    fds()
        .iter()
        .map(|e| Ok(CprFd { name: e.name.clone(), id: e.id, fd: e.fd.try_clone()? }))
        .collect()
}

/// `cpr_fd_post_load()`: takes in what the previous process sent. When a name and id come
/// twice, the first one wins.
pub fn load_fds(list: Vec<CprFd>) {
    let mut l = fds();
    for e in list {
        if !l.iter().any(|o| o.name == e.name && o.id == e.id) {
            l.push(e);
        }
    }
}

/// `cpr_set_incoming_mode()`, for whether this process loaded CPR state.
pub fn set_incoming(on: bool) {
    INCOMING.store(on, std::sync::atomic::Ordering::Release);
}

/// `cpr_is_incoming()`.
pub fn is_incoming() -> bool {
    INCOMING.load(std::sync::atomic::Ordering::Acquire)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some_fd() -> OwnedFd {
        std::fs::File::open("/dev/null").unwrap().into()
    }

    #[test]
    fn save_find_delete() {
        save_fd("cpr-test-a", 0, some_fd());
        save_fd("cpr-test-a", 1, some_fd());
        assert!(find_fd("cpr-test-a", 0).is_some());
        assert!(find_fd("cpr-test-a", 2).is_none());
        assert!(saved_fds().unwrap().iter().any(|e| e.name == "cpr-test-a" && e.id == 1));
        // The first of two loaded under one key wins, and one saved here already stays.
        load_fds(vec![
            CprFd { name: "cpr-test-b".into(), id: 0, fd: some_fd() },
            CprFd { name: "cpr-test-b".into(), id: 0, fd: some_fd() },
            CprFd { name: "cpr-test-a".into(), id: 0, fd: some_fd() },
        ]);
        assert_eq!(saved_fds().unwrap().iter().filter(|e| e.name == "cpr-test-b").count(), 1);
        delete_fd("cpr-test-a", 0);
        delete_fd("cpr-test-a", 1);
        delete_fd("cpr-test-b", 0);
        assert!(find_fd("cpr-test-a", 0).is_none());
        assert!(find_fd("cpr-test-b", 0).is_none());
    }
}
