// SPDX-License-Identifier: GPL-2.0-or-later

//! File descriptor passing, monitor/fds.c.
//!
//! A client sends descriptors with SCM_RIGHTS alongside the bytes of a command, and the command
//! decides what they are for. `getfd` keeps one under a name in the monitor that received it,
//! for a later command such as `netdev_add` to take. `add-fd` puts one in a numbered fd set that
//! every monitor shares, and the block layer opens `/dev/fdset/N` by duplicating a member whose
//! access mode matches. A set lives until it has no members left and nothing holds a duplicate of
//! one, which is why removing the last member of a set that is in use does not end the set.

use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use rustix::fs::OFlags;
use ruvm_base::{Error, Result};
use ruvm_qapi::commands::{
    register_add_fd, register_closefd, register_getfd, register_query_fdsets, register_remove_fd,
};
use ruvm_qapi::types::{
    AddFdArg, AddfdInfo, ClosefdArg, FdsetFdInfo, FdsetInfo, GetfdArg, RemoveFdArg,
};

use crate::qmp::{Commands, MonitorQmp};

/// `QERR_INVALID_PARAMETER_VALUE`.
fn invalid_value(name: &str, expected: &str) -> Error {
    Error::generic(format!("Parameter '{name}' expects {expected}"))
}

fn no_msgfd() -> Error {
    Error::generic("No file descriptor supplied via SCM_RIGHTS")
}

/// The descriptors a monitor holds by name, `mon->fds`.
#[derive(Debug, Default)]
pub struct NamedFds {
    /// Newest first, the order of QEMU's list.
    fds: Vec<(String, OwnedFd)>,
}

impl NamedFds {
    /// `monitor_add_fd()`. A name that is already taken gets the new descriptor and the old one
    /// is closed.
    pub fn add(&mut self, name: &str, fd: OwnedFd) -> Result<()> {
        if name.starts_with(|c: char| c.is_ascii_digit()) {
            return Err(invalid_value("fdname", "a name not starting with a digit"));
        }
        match self.fds.iter_mut().find(|(n, _)| n == name) {
            Some((_, old)) => *old = fd,
            None => self.fds.insert(0, (name.to_string(), fd)),
        }
        Ok(())
    }

    /// `qmp_closefd()`.
    pub fn close(&mut self, name: &str) -> Result<()> {
        match self.fds.iter().position(|(n, _)| n == name) {
            Some(i) => {
                self.fds.remove(i);
                Ok(())
            }
            None => Err(Error::generic(format!("File descriptor named '{name}' not found"))),
        }
    }

    /// `monitor_get_fd()`: the caller takes the descriptor and the name is free again.
    pub fn take(&mut self, name: &str) -> Result<OwnedFd> {
        match self.fds.iter().position(|(n, _)| n == name) {
            Some(i) => Ok(self.fds.remove(i).1),
            None => {
                Err(Error::generic(format!("File descriptor named '{name}' has not been found")))
            }
        }
    }

    pub fn len(&self) -> usize {
        self.fds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fds.is_empty()
    }
}

#[derive(Debug)]
struct Member {
    fd: OwnedFd,
    opaque: Option<String>,
}

#[derive(Debug)]
struct FdSet {
    id: i64,
    /// In the order they were added. QEMU keeps them newest first and reverses them again when
    /// it builds the `query-fdsets` reply, so this is the reply order.
    fds: Vec<Member>,
    /// Descriptors handed out by [`FdSets::dup_fd_add`] that have not been given back yet.
    dup_fds: Vec<RawFd>,
}

impl FdSet {
    fn is_unused(&self) -> bool {
        self.fds.is_empty() && self.dup_fds.is_empty()
    }
}

/// The fd sets of the process, `mon_fdsets`.
#[derive(Debug, Default)]
pub struct FdSets {
    /// Ordered by id.
    sets: Vec<FdSet>,
}

impl FdSets {
    pub fn new() -> Self {
        Self::default()
    }

    /// `monitor_fdset_add_fd()`. Without an id the descriptor goes in a new set with the lowest
    /// free id. On error the descriptor is closed.
    pub fn add_fd(
        &mut self,
        fd: OwnedFd,
        fdset_id: Option<i64>,
        opaque: Option<String>,
    ) -> Result<AddfdInfo> {
        let pos = match fdset_id {
            Some(id) => match self.sets.binary_search_by_key(&id, |s| s.id) {
                Ok(pos) => pos,
                Err(_) if id < 0 => {
                    return Err(invalid_value("fdset-id", "a non-negative value"));
                }
                Err(pos) => {
                    self.sets.insert(pos, FdSet { id, fds: Vec::new(), dup_fds: Vec::new() });
                    pos
                }
            },
            None => {
                // The first id that is not taken, counting up from 0 through the sorted list.
                let pos =
                    self.sets.iter().enumerate().take_while(|(i, s)| s.id == *i as i64).count();
                let id = pos as i64;
                self.sets.insert(pos, FdSet { id, fds: Vec::new(), dup_fds: Vec::new() });
                pos
            }
        };
        let set = &mut self.sets[pos];
        let info = AddfdInfo { fdset_id: set.id, fd: fd.as_raw_fd().into() };
        set.fds.push(Member { fd, opaque });
        Ok(info)
    }

    /// `qmp_remove_fd()`: one member, or every member when `fd` is not given.
    pub fn remove_fd(&mut self, fdset_id: i64, fd: Option<i64>) -> Result<()> {
        let not_found = || {
            let what = match fd {
                Some(fd) => format!("fdset-id:{fdset_id}, fd:{fd}"),
                None => format!("fdset-id:{fdset_id}"),
            };
            Error::generic(format!("File descriptor named '{what}' not found"))
        };
        let pos = self.sets.iter().position(|s| s.id == fdset_id).ok_or_else(not_found)?;
        let set = &mut self.sets[pos];
        match fd {
            Some(fd) => {
                let i = set
                    .fds
                    .iter()
                    .position(|m| i64::from(m.fd.as_raw_fd()) == fd)
                    .ok_or_else(not_found)?;
                set.fds.remove(i);
            }
            None => set.fds.clear(),
        }
        if set.is_unused() {
            self.sets.remove(pos);
        }
        Ok(())
    }

    /// `qmp_query_fdsets()`. QEMU prepends each set to the reply, so the highest id comes first.
    pub fn query(&self) -> Vec<FdsetInfo> {
        self.sets
            .iter()
            .rev()
            .map(|s| FdsetInfo {
                fdset_id: s.id,
                fds: s
                    .fds
                    .iter()
                    .map(|m| FdsetFdInfo { fd: m.fd.as_raw_fd().into(), opaque: m.opaque.clone() })
                    .collect(),
            })
            .collect()
    }

    /// `monitor_fdset_dup_fd_add()`: what opening `/dev/fdset/N` with `flags` returns. The
    /// first member whose access mode matches is duplicated, and the duplicate keeps the set
    /// alive until [`FdSets::dup_fd_remove`] is called for it.
    pub fn dup_fd_add(&mut self, fdset_id: i64, flags: OFlags) -> Result<OwnedFd> {
        let Some(set) = self.sets.iter_mut().find(|s| s.id == fdset_id) else {
            return Err(Error::generic(format!("Failed to find fdset /dev/fdset/{fdset_id}")));
        };
        let mask = access_mask();
        let mut found = None;
        for m in &set.fds {
            let fl = rustix::fs::fcntl_getfl(&m.fd).map_err(|_| {
                Error::generic(format!(
                    "Failed to read file status flags for fd={}",
                    m.fd.as_raw_fd()
                ))
            })?;
            if flags & mask == fl & mask {
                found = Some(&m.fd);
                break;
            }
        }
        let Some(fd) = found else {
            return Err(Error::generic(format!(
                "Failed to find file descriptor with matching flags=0x{:x}",
                flags.bits()
            )));
        };
        let dup = dup_flags(fd, flags).map_err(|_| {
            Error::generic(format!("Failed to dup() given file descriptor fd={}", fd.as_raw_fd()))
        })?;
        set.dup_fds.insert(0, dup.as_raw_fd());
        Ok(dup)
    }

    /// `monitor_fdset_dup_fd_remove()`: the holder of a duplicate is about to close it.
    pub fn dup_fd_remove(&mut self, dup_fd: RawFd) {
        for pos in 0..self.sets.len() {
            let set = &mut self.sets[pos];
            if let Some(i) = set.dup_fds.iter().position(|&d| d == dup_fd) {
                set.dup_fds.remove(i);
                if set.is_unused() {
                    self.sets.remove(pos);
                }
                return;
            }
        }
    }

    /// `monitor_fdsets_cleanup()`, run when a monitor disconnects. The members belong to the
    /// client and stay until it removes them, so this only drops sets nothing uses.
    pub fn cleanup(&mut self) {
        self.sets.retain(|s| !s.is_unused());
    }

    pub fn len(&self) -> usize {
        self.sets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sets.is_empty()
    }
}

#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "netbsd"
))]
fn access_mask() -> OFlags {
    OFlags::ACCMODE | OFlags::DIRECT
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "netbsd"
)))]
fn access_mask() -> OFlags {
    OFlags::ACCMODE
}

/// `qemu_dup_flags()`.
fn dup_flags(fd: &OwnedFd, flags: OFlags) -> rustix::io::Result<OwnedFd> {
    let dup = rustix::io::fcntl_dupfd_cloexec(fd, 0)?;
    let dup_flags = rustix::fs::fcntl_getfl(&dup)?;
    if flags & OFlags::SYNC != dup_flags & OFlags::SYNC {
        return Err(rustix::io::Errno::INVAL);
    }
    rustix::fs::fcntl_setfl(&dup, flags)?;
    if flags.contains(OFlags::TRUNC) || flags.contains(OFlags::CREATE | OFlags::EXCL) {
        rustix::fs::ftruncate(&dup, 0)?;
    }
    Ok(dup)
}

fn getfd(mon: &MonitorQmp, arg: GetfdArg) -> Result<()> {
    let fd = mon.take_msgfd().ok_or_else(no_msgfd)?;
    mon.named_fds().add(&arg.fdname, fd)
}

fn closefd(mon: &MonitorQmp, arg: ClosefdArg) -> Result<()> {
    mon.named_fds().close(&arg.fdname)
}

fn add_fd(mon: &MonitorQmp, arg: AddFdArg) -> Result<AddfdInfo> {
    let fd = mon.take_msgfd().ok_or_else(no_msgfd)?;
    let Some(qmp) = mon.qmp() else { return Err(no_msgfd()) };
    qmp.fdsets().add_fd(fd, arg.fdset_id, arg.opaque)
}

fn remove_fd(mon: &MonitorQmp, arg: RemoveFdArg) -> Result<()> {
    let Some(qmp) = mon.qmp() else { return Ok(()) };
    qmp.fdsets().remove_fd(arg.fdset_id, arg.fd)
}

fn query_fdsets(mon: &MonitorQmp) -> Result<Vec<FdsetInfo>> {
    Ok(mon.qmp().map(|q| q.fdsets().query()).unwrap_or_default())
}

pub(crate) fn register(cmds: &mut Commands) {
    register_getfd(cmds, getfd);
    register_closefd(cmds, closefd);
    register_add_fd(cmds, add_fd);
    register_remove_fd(cmds, remove_fd);
    register_query_fdsets(cmds, query_fdsets);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fd() -> OwnedFd {
        std::fs::File::open("/dev/null").unwrap().into()
    }

    #[test]
    fn named_fds() {
        let mut n = NamedFds::default();
        let err = n.add("1x", fd()).unwrap_err();
        assert_eq!(err.to_string(), "Parameter 'fdname' expects a name not starting with a digit");
        n.add("a", fd()).unwrap();
        n.add("a", fd()).unwrap();
        assert_eq!(n.len(), 1);
        n.add("b", fd()).unwrap();
        n.take("a").unwrap();
        assert_eq!(
            n.take("a").unwrap_err().to_string(),
            "File descriptor named 'a' has not been found"
        );
        n.close("b").unwrap();
        assert_eq!(n.close("b").unwrap_err().to_string(), "File descriptor named 'b' not found");
        assert!(n.is_empty());
    }

    #[test]
    fn set_ids() {
        let mut s = FdSets::new();
        assert_eq!(s.add_fd(fd(), None, None).unwrap().fdset_id, 0);
        assert_eq!(s.add_fd(fd(), Some(2), None).unwrap().fdset_id, 2);
        assert_eq!(s.add_fd(fd(), None, None).unwrap().fdset_id, 1);
        assert_eq!(s.add_fd(fd(), None, None).unwrap().fdset_id, 3);
        assert_eq!(s.add_fd(fd(), Some(2), Some("x".into())).unwrap().fdset_id, 2);
        let err = s.add_fd(fd(), Some(-1), None).unwrap_err();
        assert_eq!(err.to_string(), "Parameter 'fdset-id' expects a non-negative value");
        let q = s.query();
        assert_eq!(q.iter().map(|i| i.fdset_id).collect::<Vec<_>>(), [3, 2, 1, 0]);
        assert_eq!(q[1].fds.len(), 2);
        assert_eq!(q[1].fds[1].opaque.as_deref(), Some("x"));
    }

    #[test]
    fn remove() {
        let mut s = FdSets::new();
        let a = s.add_fd(fd(), Some(5), None).unwrap();
        s.add_fd(fd(), Some(5), None).unwrap();
        assert_eq!(
            s.remove_fd(4, None).unwrap_err().to_string(),
            "File descriptor named 'fdset-id:4' not found"
        );
        assert_eq!(
            s.remove_fd(5, Some(-3)).unwrap_err().to_string(),
            "File descriptor named 'fdset-id:5, fd:-3' not found"
        );
        s.remove_fd(5, Some(a.fd)).unwrap();
        assert_eq!(s.query()[0].fds.len(), 1);
        s.remove_fd(5, None).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn dup_keeps_the_set() {
        let mut s = FdSets::new();
        let info = s.add_fd(fd(), None, None).unwrap();
        assert_eq!(
            s.dup_fd_add(1, OFlags::RDONLY).unwrap_err().to_string(),
            "Failed to find fdset /dev/fdset/1"
        );
        assert!(
            s.dup_fd_add(0, OFlags::RDWR)
                .unwrap_err()
                .to_string()
                .starts_with("Failed to find file descriptor with matching flags=0x")
        );
        let dup = s.dup_fd_add(0, OFlags::RDONLY).unwrap();
        assert_ne!(i64::from(dup.as_raw_fd()), info.fd);
        s.remove_fd(0, None).unwrap();
        // The duplicate is still out, so the set stays, empty.
        assert_eq!(s.len(), 1);
        s.cleanup();
        assert_eq!(s.len(), 1);
        s.dup_fd_remove(dup.as_raw_fd());
        assert!(s.is_empty());
    }
}
