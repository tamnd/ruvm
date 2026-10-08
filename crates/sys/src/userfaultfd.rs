// SPDX-License-Identifier: MIT OR Apache-2.0

//! Linux userfaultfd, which postcopy migration uses to fetch guest pages on demand and a
//! background snapshot uses to see guest writes coming.
//!
//! The incoming side of postcopy registers its guest RAM in `UFFDIO_REGISTER_MODE_MISSING` mode.
//! From then on a thread that touches a page with nothing mapped (vCPU, device or the kernel on
//! their behalf) sleeps in the kernel, and the fault shows up as a message on the descriptor.
//! The migration code asks the source for that page and installs it with `UFFDIO_COPY` (or
//! `UFFDIO_ZEROPAGE`), which maps it atomically and wakes the sleepers.
//!
//! A background snapshot registers RAM in `UFFDIO_REGISTER_MODE_WP` mode and write-protects it
//! with `UFFDIO_WRITEPROTECT`. A write to a protected page then sleeps the same way until the
//! snapshot saved the page and lifted the protection.
//!
//! This is the subset of `linux/userfaultfd.h` and QEMU's util/userfaultfd.c that these need.

use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use crate::HostMemory;

const UFFD_API: u64 = 0xaa;
const UFFDIO_REGISTER_MODE_MISSING: u64 = 1;
const UFFDIO_REGISTER_MODE_WP: u64 = 1 << 1;
const UFFDIO_WRITEPROTECT_MODE_WP: u64 = 1;
const UFFDIO_WRITEPROTECT_NR: u64 = 0x06;
/// `UFFD_FEATURE_PAGEFAULT_FLAG_WP`: write-protect faults.
pub const FEATURE_PAGEFAULT_FLAG_WP: u64 = 1;
const UFFD_EVENT_PAGEFAULT: u8 = 0x12;
const PAGE_SIZE: usize = 4096;

#[cfg(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc64"
))]
const fn ioc(dir_read: bool, dir_write: bool, nr: u64, size: u64) -> u64 {
    let dir = (if dir_read { 2 } else { 0 }) | (if dir_write { 4 } else { 0 });
    (dir << 29) | (size << 16) | (0xaa << 8) | nr
}

#[cfg(not(any(
    target_arch = "powerpc",
    target_arch = "powerpc64",
    target_arch = "mips",
    target_arch = "mips64",
    target_arch = "sparc64"
)))]
const fn ioc(dir_read: bool, dir_write: bool, nr: u64, size: u64) -> u64 {
    let dir = (if dir_read { 2 } else { 0 }) | (if dir_write { 1 } else { 0 });
    (dir << 30) | (size << 16) | (0xaa << 8) | nr
}

const UFFDIO_API: u64 = ioc(true, true, 0x3f, 24);
const UFFDIO_REGISTER: u64 = ioc(true, true, 0x00, 32);
const UFFDIO_UNREGISTER: u64 = ioc(true, false, 0x01, 16);
const UFFDIO_COPY: u64 = ioc(true, true, 0x03, 40);
const UFFDIO_ZEROPAGE: u64 = ioc(true, true, 0x04, 32);
const UFFDIO_WRITEPROTECT: u64 = ioc(true, true, UFFDIO_WRITEPROTECT_NR, 24);

#[repr(C)]
#[derive(Default)]
struct UffdioApi {
    api: u64,
    features: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioRange {
    start: u64,
    len: u64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioRegister {
    range: UffdioRange,
    mode: u64,
    ioctls: u64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioCopy {
    dst: u64,
    src: u64,
    len: u64,
    mode: u64,
    copy: i64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioZeropage {
    range: UffdioRange,
    mode: u64,
    zeropage: i64,
}

#[repr(C)]
#[derive(Default)]
struct UffdioWriteprotect {
    range: UffdioRange,
    mode: u64,
}

/// The argument structures of the ioctls this module issues.
trait Arg {}
impl Arg for UffdioWriteprotect {}
impl Arg for UffdioApi {}
impl Arg for UffdioRange {}
impl Arg for UffdioRegister {}
impl Arg for UffdioCopy {}
impl Arg for UffdioZeropage {}

/// A userfaultfd descriptor.
#[derive(Debug)]
pub struct Userfaultfd {
    fd: OwnedFd,
}

impl Userfaultfd {
    /// `uffd_open()` and the `UFFDIO_API` handshake, with no optional features. The descriptor
    /// is non-blocking; [`wait_fault`](Self::wait_fault) polls it.
    pub fn new() -> io::Result<Self> {
        Self::with_features(0)
    }

    /// `uffd_create_fd()`: like [`new`](Self::new), asking for the `UFFD_FEATURE_*` bits in
    /// `features`. Fails if the kernel lacks any of them.
    pub fn with_features(features: u64) -> io::Result<Self> {
        let (uffd, _) = Self::open(features)?;
        Ok(uffd)
    }

    /// `uffd_query_features()`: the `UFFD_FEATURE_*` bits the kernel has.
    pub fn features() -> io::Result<u64> {
        Self::open(0).map(|(_, features)| features)
    }

    /// Opens a descriptor and does the `UFFDIO_API` handshake asking for `features`, returning
    /// the features the kernel reports back.
    fn open(features: u64) -> io::Result<(Self, u64)> {
        // SAFETY: userfaultfd(2) takes flags and returns a new descriptor or -1.
        let ret =
            unsafe { libc::syscall(libc::SYS_userfaultfd, libc::O_CLOEXEC | libc::O_NONBLOCK) };
        if ret < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: the kernel just returned this descriptor and nothing else owns it.
        let fd = unsafe { OwnedFd::from_raw_fd(ret as i32) };
        let uffd = Userfaultfd { fd };
        let mut api = UffdioApi { api: UFFD_API, features, ..Default::default() };
        uffd.ioctl(UFFDIO_API, &mut api)?;
        Ok((uffd, api.features))
    }

    fn ioctl<T: Arg>(&self, req: u64, arg: &mut T) -> io::Result<()> {
        // SAFETY: `req` is one of the userfaultfd ioctls whose argument is `T` (the `Arg` types
        // mirror the kernel structures and every caller pairs them right), and `arg` is valid
        // for reads and writes for the whole call. The ioctls only touch guest memory through
        // the ranges in `arg`, which the callers keep inside a `HostMemory` they borrow.
        let ret = unsafe { libc::ioctl(self.fd.as_raw_fd(), req as _, arg as *mut T) };
        if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
    }

    fn range(mem: &HostMemory, offset: usize, len: usize) -> io::Result<UffdioRange> {
        let end = offset.checked_add(len);
        if end.is_none_or(|e| e > mem.len()) || offset % PAGE_SIZE != 0 || len % PAGE_SIZE != 0 {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        Ok(UffdioRange { start: (mem.host_addr() + offset) as u64, len: len as u64 })
    }

    /// Registers all of `mem` for missing page faults.
    pub fn register(&self, mem: &HostMemory) -> io::Result<()> {
        let mut reg = UffdioRegister {
            range: Self::range(mem, 0, mem.len())?,
            mode: UFFDIO_REGISTER_MODE_MISSING,
            ioctls: 0,
        };
        self.ioctl(UFFDIO_REGISTER, &mut reg)
    }

    /// Registers all of `mem` for write-protect faults. Returns whether `UFFDIO_WRITEPROTECT`
    /// works on it, which is what the registration reports in `ioctls`.
    pub fn register_write_protect(&self, mem: &HostMemory) -> io::Result<bool> {
        let mut reg = UffdioRegister {
            range: Self::range(mem, 0, mem.len())?,
            mode: UFFDIO_REGISTER_MODE_WP,
            ioctls: 0,
        };
        self.ioctl(UFFDIO_REGISTER, &mut reg)?;
        Ok(reg.ioctls & (1 << UFFDIO_WRITEPROTECT_NR) != 0)
    }

    /// `uffd_change_protection()`: write-protects `len` bytes at `offset` of `mem`, or lifts the
    /// protection and wakes the threads that wait for it.
    pub fn write_protect(
        &self,
        mem: &HostMemory,
        offset: usize,
        len: usize,
        protect: bool,
    ) -> io::Result<()> {
        let mut wp = UffdioWriteprotect {
            range: Self::range(mem, offset, len)?,
            mode: if protect { UFFDIO_WRITEPROTECT_MODE_WP } else { 0 },
        };
        self.ioctl(UFFDIO_WRITEPROTECT, &mut wp)
    }

    /// Stops faults on `mem` coming here.
    pub fn unregister(&self, mem: &HostMemory) -> io::Result<()> {
        let mut range = Self::range(mem, 0, mem.len())?;
        self.ioctl(UFFDIO_UNREGISTER, &mut range)
    }

    /// `UFFDIO_COPY`: maps a copy of `data`, whole pages, at `offset` of `mem` and wakes the
    /// threads waiting for it. Fails with `EEXIST` if a page is there already.
    pub fn copy(&self, mem: &HostMemory, offset: usize, data: &[u8]) -> io::Result<()> {
        let range = Self::range(mem, offset, data.len())?;
        let mut copy = UffdioCopy {
            dst: range.start,
            src: data.as_ptr() as u64,
            len: range.len,
            ..Default::default()
        };
        self.ioctl(UFFDIO_COPY, &mut copy)
    }

    /// `UFFDIO_ZEROPAGE`: maps zero pages over `len` bytes at `offset` of `mem`.
    pub fn zeropage(&self, mem: &HostMemory, offset: usize, len: usize) -> io::Result<()> {
        let mut zero =
            UffdioZeropage { range: Self::range(mem, offset, len)?, ..Default::default() };
        self.ioctl(UFFDIO_ZEROPAGE, &mut zero)
    }

    /// Waits up to `timeout_ms` for a page fault and returns the host address it hit, or `None`
    /// if nothing came. Messages other than page faults are skipped.
    pub fn wait_fault(&self, timeout_ms: i32) -> io::Result<Option<usize>> {
        let mut pfd = libc::pollfd { fd: self.fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: `pfd` is one valid pollfd for the call.
        let n = unsafe { libc::poll(&mut pfd, 1, timeout_ms) };
        if n < 0 {
            let e = io::Error::last_os_error();
            return if e.kind() == io::ErrorKind::Interrupted { Ok(None) } else { Err(e) };
        }
        if n == 0 {
            return Ok(None);
        }
        // struct uffd_msg: the event in byte 0, and for a page fault the flags at 8 and the
        // address at 16.
        let mut msg = [0u8; 32];
        // SAFETY: `msg` is valid for writes of its length.
        let got = unsafe { libc::read(self.fd.as_raw_fd(), msg.as_mut_ptr().cast(), msg.len()) };
        if got < 0 {
            let e = io::Error::last_os_error();
            return if e.kind() == io::ErrorKind::WouldBlock { Ok(None) } else { Err(e) };
        }
        if got as usize != msg.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Read {got} bytes from userfaultfd expected {}", msg.len()),
            ));
        }
        if msg[0] != UFFD_EVENT_PAGEFAULT {
            return Ok(None);
        }
        let addr = u64::from_ne_bytes(msg[16..24].try_into().expect("8 bytes"));
        Ok(Some(addr as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn ioctl_numbers() {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
        {
            assert_eq!(UFFDIO_API, 0xc018_aa3f);
            assert_eq!(UFFDIO_REGISTER, 0xc020_aa00);
            assert_eq!(UFFDIO_UNREGISTER, 0x8010_aa01);
            assert_eq!(UFFDIO_COPY, 0xc028_aa03);
            assert_eq!(UFFDIO_ZEROPAGE, 0xc020_aa04);
            assert_eq!(UFFDIO_WRITEPROTECT, 0xc018_aa06);
        }
    }

    #[test]
    fn fault_and_fill() {
        // Unprivileged userfaultfd may be off on the host running the tests.
        let Ok(uffd) = Userfaultfd::new() else { return };
        let mem = HostMemory::new(4 * PAGE_SIZE).unwrap();
        uffd.register(&mem).unwrap();
        // Fill page 1 before anyone touches it, then fault page 2 from another thread.
        uffd.copy(&mem, PAGE_SIZE, &[7u8; PAGE_SIZE]).unwrap();
        assert_eq!(
            uffd.copy(&mem, PAGE_SIZE, &[8u8; PAGE_SIZE]).unwrap_err().raw_os_error(),
            Some(libc::EEXIST)
        );
        std::thread::scope(|s| {
            let t = s.spawn(|| mem.as_slice()[2 * PAGE_SIZE + 5].load(Ordering::Relaxed));
            let addr = loop {
                if let Some(a) = uffd.wait_fault(1000).unwrap() {
                    break a;
                }
            };
            assert_eq!(addr & !(PAGE_SIZE - 1), mem.host_addr() + 2 * PAGE_SIZE);
            uffd.zeropage(&mem, 2 * PAGE_SIZE, PAGE_SIZE).unwrap();
            assert_eq!(t.join().unwrap(), 0);
        });
        assert_eq!(mem.as_slice()[PAGE_SIZE + 9].load(Ordering::Relaxed), 7);
        assert!(uffd.copy(&mem, 4 * PAGE_SIZE, &[0; PAGE_SIZE]).is_err());
        uffd.unregister(&mem).unwrap();
    }

    #[test]
    fn write_protect() {
        // Unprivileged userfaultfd may be off, and old kernels lack write protection.
        if !Userfaultfd::features().is_ok_and(|f| f & FEATURE_PAGEFAULT_FLAG_WP != 0) {
            return;
        }
        let uffd = Userfaultfd::with_features(FEATURE_PAGEFAULT_FLAG_WP).unwrap();
        let mem = HostMemory::new(2 * PAGE_SIZE).unwrap();
        // Pages with nothing mapped cannot be protected, so map them first.
        mem.as_slice()[0].store(3, Ordering::Relaxed);
        mem.as_slice()[PAGE_SIZE].store(4, Ordering::Relaxed);
        assert!(uffd.register_write_protect(&mem).unwrap());
        uffd.write_protect(&mem, 0, 2 * PAGE_SIZE, true).unwrap();
        // Reads go on as before.
        assert_eq!(mem.as_slice()[PAGE_SIZE].load(Ordering::Relaxed), 4);
        std::thread::scope(|s| {
            let t = s.spawn(|| mem.as_slice()[PAGE_SIZE + 7].store(9, Ordering::Relaxed));
            let addr = loop {
                if let Some(a) = uffd.wait_fault(1000).unwrap() {
                    break a;
                }
            };
            assert_eq!(addr & !(PAGE_SIZE - 1), mem.host_addr() + PAGE_SIZE);
            // The writer waits until the protection goes.
            assert_eq!(mem.as_slice()[PAGE_SIZE + 7].load(Ordering::Relaxed), 0);
            uffd.write_protect(&mem, PAGE_SIZE, PAGE_SIZE, false).unwrap();
            t.join().unwrap();
        });
        assert_eq!(mem.as_slice()[PAGE_SIZE + 7].load(Ordering::Relaxed), 9);
        uffd.unregister(&mem).unwrap();
        // Unregistering lifts the protection of page 0 as well.
        mem.as_slice()[1].store(5, Ordering::Relaxed);
    }
}
