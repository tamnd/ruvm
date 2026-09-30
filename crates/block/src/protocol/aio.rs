// SPDX-License-Identifier: GPL-2.0-or-later

//! The I/O back ends of file-posix: `aio=threads`, `aio=native` (block/linux-aio.c) and the
//! io_uring engine (block/io_uring.c).
//!
//! Every request here is synchronous: it is submitted and then waited for on the caller's
//! thread, like the rest of the block layer until the ruvm-aio integration. The engines keep
//! the shape the asynchronous path needs, though: a request is one submission queue entry or
//! one `iocb`, and the contexts live in a pool, so ruvm-aio can later take the submission and
//! the completion apart without changing what reaches the kernel.
//!
//! Differences from QEMU:
//!
//! - QEMU's schema has `aio=io_uring`; this build's schema does not, as if QEMU was built
//!   without liburing. io_uring is still used underneath: `aio=native` submits through
//!   io_uring on kernels that have it and through linux-aio only when io_uring cannot be set
//!   up, and a node on Linux with `cache.direct=on` and no `aio` option uses io_uring too,
//!   falling back to plain positioned reads and writes. The node still reports the `aio` value
//!   the user gave, so management tools see QEMU's defaults.
//! - QEMU keeps one linux-aio context and one io_uring per `AioContext`, with up to 128
//!   requests in flight on each. Here each context has one request in flight at a time, and
//!   concurrent requests from different threads take different contexts from the pool.
//!   `aio-max-batch` has no effect because nothing is ever batched.
//! - When linux-aio cannot be set up either, QEMU reports "Unable to use Linux AIO, falling
//!   back to thread pool: " at the first request. Here the same message comes when the node
//!   is opened.

#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

/// How a node's reads and writes reach the kernel.
pub(crate) enum AioEngine {
    /// `aio=threads`: `pread()` and `pwrite()`. QEMU runs them in its thread pool, here they
    /// run on the caller's thread.
    Threads,
    /// linux-aio, `io_submit()` and `io_getevents()`.
    #[cfg(target_os = "linux")]
    LinuxAio(linux_aio::Pool),
    /// io_uring.
    #[cfg(target_os = "linux")]
    IoUring(uring::Pool),
}

impl AioEngine {
    /// The engine for `aio=native`: io_uring when the kernel has it, linux-aio otherwise and
    /// the thread engine when neither can be set up, with QEMU's message.
    pub(crate) fn native() -> AioEngine {
        #[cfg(target_os = "linux")]
        {
            if let Ok(p) = uring::Pool::new() {
                return AioEngine::IoUring(p);
            }
            match linux_aio::Pool::new() {
                Ok(p) => AioEngine::LinuxAio(p),
                Err(e) => {
                    let e = ruvm_base::Error::from_io("failed to create linux AIO context", e);
                    ruvm_base::report::error_report(&format!(
                        "Unable to use Linux AIO, falling back to thread pool: {}",
                        e.message()
                    ));
                    AioEngine::Threads
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        AioEngine::Threads
    }

    /// The engine for a node without an `aio` option: io_uring on Linux with `cache.direct=on`
    /// when the kernel lets us have it, the thread engine otherwise.
    pub(crate) fn default_for(direct: bool) -> AioEngine {
        #[cfg(target_os = "linux")]
        if direct {
            if let Ok(p) = uring::Pool::new() {
                return AioEngine::IoUring(p);
            }
        }
        let _ = direct;
        AioEngine::Threads
    }

    /// The engine's name, for tests and traces.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            AioEngine::Threads => "threads",
            #[cfg(target_os = "linux")]
            AioEngine::LinuxAio(_) => "native",
            #[cfg(target_os = "linux")]
            AioEngine::IoUring(_) => "io_uring",
        }
    }

    /// One read at `offset`. Like `pread()` it may return fewer bytes than asked for.
    pub(crate) fn read_at(&self, file: &File, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        match self {
            AioEngine::Threads => file.read_at(buf, offset),
            #[cfg(target_os = "linux")]
            AioEngine::LinuxAio(p) => p.rw(file, linux_aio::Op::Read(buf), offset),
            #[cfg(target_os = "linux")]
            AioEngine::IoUring(p) => p.rw(file, uring::Op::Read(buf), offset),
        }
    }

    /// One write at `offset`. Like `pwrite()` it may write fewer bytes than asked for.
    pub(crate) fn write_at(&self, file: &File, buf: &[u8], offset: u64) -> io::Result<usize> {
        match self {
            AioEngine::Threads => file.write_at(buf, offset),
            #[cfg(target_os = "linux")]
            AioEngine::LinuxAio(p) => p.rw(file, linux_aio::Op::Write(buf), offset),
            #[cfg(target_os = "linux")]
            AioEngine::IoUring(p) => p.rw(file, uring::Op::Write(buf), offset),
        }
    }
}

/// linux-aio through the raw system calls, the way block/linux-aio.c uses libaio.
#[cfg(target_os = "linux")]
pub(crate) mod linux_aio {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::sync::Mutex;

    /// `IOCB_CMD_PREAD` and `IOCB_CMD_PWRITE`.
    const IOCB_CMD_PREAD: u16 = 0;
    const IOCB_CMD_PWRITE: u16 = 1;

    /// `struct iocb` from linux/aio_abi.h. The key and the flags swap places on big endian
    /// hosts.
    #[repr(C)]
    #[derive(Default)]
    struct Iocb {
        aio_data: u64,
        #[cfg(target_endian = "little")]
        aio_key: u32,
        aio_rw_flags: i32,
        #[cfg(target_endian = "big")]
        aio_key: u32,
        aio_lio_opcode: u16,
        aio_reqprio: i16,
        aio_fildes: u32,
        aio_buf: u64,
        aio_nbytes: u64,
        aio_offset: i64,
        aio_reserved2: u64,
        aio_flags: u32,
        aio_resfd: u32,
    }

    /// `struct io_event`.
    #[repr(C)]
    #[derive(Default)]
    struct IoEvent {
        data: u64,
        obj: u64,
        res: i64,
        res2: i64,
    }

    /// One request.
    pub(crate) enum Op<'a> {
        Read(&'a mut [u8]),
        Write(&'a [u8]),
    }

    /// A linux-aio context, `aio_context_t`.
    struct Ctx(libc::c_ulong);

    impl Ctx {
        /// `io_setup()` for one request in flight.
        fn new() -> io::Result<Ctx> {
            let mut ctx: libc::c_ulong = 0;
            // SAFETY: io_setup() writes the new context to the pointer, which points at a
            // live, zeroed `aio_context_t` as the call requires.
            let ret = unsafe { libc::syscall(libc::SYS_io_setup, 1 as libc::c_long, &raw mut ctx) };
            if ret < 0 {
                return Err(io::Error::last_os_error());
            }
            Ok(Ctx(ctx))
        }

        /// Submits one request and waits for it. The buffer stays borrowed until the kernel
        /// has completed the request: on an error after the submission the context is
        /// destroyed, and `io_destroy()` waits for what is in flight.
        fn rw(self, file: &File, op: Op<'_>, offset: u64) -> (Option<Ctx>, io::Result<usize>) {
            let (opcode, buf, len) = match op {
                Op::Read(b) => (IOCB_CMD_PREAD, b.as_mut_ptr() as u64, b.len()),
                Op::Write(b) => (IOCB_CMD_PWRITE, b.as_ptr() as u64, b.len()),
            };
            let mut iocb = Iocb {
                aio_lio_opcode: opcode,
                aio_fildes: file.as_raw_fd() as u32,
                aio_buf: buf,
                aio_nbytes: len as u64,
                aio_offset: offset as i64,
                ..Iocb::default()
            };
            let mut list = [&raw mut iocb];
            loop {
                // SAFETY: the context is live, `list` holds one pointer to an initialised
                // `iocb` that stays in place until the request completes, and the buffer it
                // names is borrowed for `len` bytes, mutably for a read, until this function
                // returns, which is after the completion or after `io_destroy()` below has
                // waited for it.
                let ret = unsafe {
                    libc::syscall(libc::SYS_io_submit, self.0, 1 as libc::c_long, list.as_mut_ptr())
                };
                if ret == 1 {
                    break;
                }
                let e = io::Error::last_os_error();
                if ret < 0 && e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                // Nothing was submitted, the context can be used again.
                let e = if ret < 0 { e } else { io::Error::from_raw_os_error(libc::EIO) };
                return (Some(self), Err(e));
            }
            let mut ev = IoEvent::default();
            loop {
                // SAFETY: the context is live and `ev` has room for the one event asked for;
                // a null timeout waits for ever.
                let ret = unsafe {
                    libc::syscall(
                        libc::SYS_io_getevents,
                        self.0,
                        1 as libc::c_long,
                        1 as libc::c_long,
                        &raw mut ev,
                        std::ptr::null_mut::<libc::timespec>(),
                    )
                };
                if ret == 1 {
                    break;
                }
                let e = io::Error::last_os_error();
                if ret < 0 && e.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                // Dropping the context waits for the request.
                drop(self);
                return (None, Err(e));
            }
            let r = if ev.res < 0 {
                Err(io::Error::from_raw_os_error(-ev.res as i32))
            } else {
                Ok(ev.res as usize)
            };
            (Some(self), r)
        }
    }

    impl Drop for Ctx {
        fn drop(&mut self) {
            // SAFETY: the context is live and nothing uses it after this. io_destroy() waits
            // for requests still in flight, so no buffer is written after it returns.
            unsafe {
                libc::syscall(libc::SYS_io_destroy, self.0);
            }
        }
    }

    /// The contexts not in use, `LinuxAioState` per thread in QEMU.
    pub(crate) struct Pool(Mutex<Vec<Ctx>>);

    impl Pool {
        /// `laio_init()`: sets up the first context, which also tells whether the kernel
        /// allows linux-aio at all.
        pub(crate) fn new() -> io::Result<Pool> {
            Ok(Pool(Mutex::new(vec![Ctx::new()?])))
        }

        /// `laio_co_submit()` and the wait for its completion.
        pub(crate) fn rw(&self, file: &File, op: Op<'_>, offset: u64) -> io::Result<usize> {
            let ctx = self.0.lock().unwrap().pop();
            let ctx = match ctx {
                Some(c) => c,
                None => Ctx::new()?,
            };
            let (ctx, r) = ctx.rw(file, op, offset);
            if let Some(ctx) = ctx {
                self.0.lock().unwrap().push(ctx);
            }
            r
        }
    }
}

/// io_uring through the `io-uring` crate, the way block/io_uring.c uses liburing.
#[cfg(target_os = "linux")]
pub(crate) mod uring {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::sync::Mutex;

    use io_uring::{IoUring, opcode, types};

    /// One request.
    pub(crate) enum Op<'a> {
        Read(&'a mut [u8]),
        Write(&'a [u8]),
    }

    /// The rings not in use.
    pub(crate) struct Pool(Mutex<Vec<IoUring>>);

    /// Room for one request in flight, rounded up by the kernel.
    const RING_ENTRIES: u32 = 2;

    impl Pool {
        /// `luring_init()`: sets up the first ring, which also tells whether the kernel
        /// allows io_uring at all.
        pub(crate) fn new() -> io::Result<Pool> {
            Ok(Pool(Mutex::new(vec![IoUring::new(RING_ENTRIES)?])))
        }

        /// `luring_co_submit()` and the wait for its completion.
        pub(crate) fn rw(&self, file: &File, op: Op<'_>, offset: u64) -> io::Result<usize> {
            let ring = self.0.lock().unwrap().pop();
            let mut ring = match ring {
                Some(r) => r,
                None => IoUring::new(RING_ENTRIES)?,
            };
            let (reuse, r) = submit_wait(&mut ring, file, op, offset);
            if reuse {
                self.0.lock().unwrap().push(ring);
            }
            r
        }
    }

    /// Submits one request on an idle ring and waits for its completion. The flag says
    /// whether the ring is idle again and can go back to the pool.
    fn submit_wait(
        ring: &mut IoUring,
        file: &File,
        op: Op<'_>,
        offset: u64,
    ) -> (bool, io::Result<usize>) {
        let fd = types::Fd(file.as_raw_fd());
        // A request longer than `u32::MAX` bytes is cut short; the callers loop on short
        // reads and writes as they do with `pread()`.
        let entry = match op {
            Op::Read(b) => {
                let len = b.len().min(u32::MAX as usize) as u32;
                opcode::Read::new(fd, b.as_mut_ptr(), len).offset(offset).build()
            }
            Op::Write(b) => {
                let len = b.len().min(u32::MAX as usize) as u32;
                opcode::Write::new(fd, b.as_ptr(), len).offset(offset).build()
            }
        };
        // SAFETY: the entry names a live descriptor and a buffer borrowed, mutably for a read,
        // for its full length until this function returns, and it does not return before the
        // completion arrives: the loop below only leaves early when the kernel never took the
        // entry. The ring is idle, so there is room for it.
        let pushed = unsafe { ring.submission().push(&entry.user_data(0)) };
        if pushed.is_err() {
            return (false, Err(io::Error::from_raw_os_error(libc::EBUSY)));
        }
        loop {
            match ring.submit_and_wait(1) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    if !ring.submission().is_empty() {
                        // The kernel never took the entry, so the buffer is free again. The
                        // entry is still queued, so the ring is not reused.
                        return (false, Err(e));
                    }
                    // The kernel has the request: keep waiting for it.
                    std::thread::yield_now();
                }
            }
            if let Some(cqe) = ring.completion().next() {
                let res = cqe.result();
                return (
                    true,
                    if res < 0 {
                        Err(io::Error::from_raw_os_error(-res))
                    } else {
                        Ok(res as usize)
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(engine: &AioEngine, name: &str) {
        let path = std::env::temp_dir().join(format!("ruvm-aio-{}-{name}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        let data: Vec<u8> = (0..8192u32).map(|i| i as u8).collect();
        assert_eq!(engine.write_at(&file, &data, 4096).unwrap(), data.len());
        let mut back = vec![0u8; data.len()];
        assert_eq!(engine.read_at(&file, &mut back, 4096).unwrap(), data.len());
        assert_eq!(back, data);
        // Reading past the end gives 0 bytes, as pread() does.
        assert_eq!(engine.read_at(&file, &mut back, 1 << 20).unwrap(), 0);
        drop(file);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn threads_round_trip() {
        let e = AioEngine::Threads;
        assert_eq!(e.name(), "threads");
        round_trip(&e, "threads");
        assert_eq!(AioEngine::default_for(false).name(), "threads");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn linux_aio_round_trip() {
        // Containers may forbid io_setup(), then there is nothing to test.
        let Ok(p) = linux_aio::Pool::new() else { return };
        let e = AioEngine::LinuxAio(p);
        assert_eq!(e.name(), "native");
        round_trip(&e, "linux-aio");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn io_uring_round_trip() {
        // Kernels before 5.1 and seccomp profiles may not allow io_uring.
        let Ok(p) = uring::Pool::new() else { return };
        let e = AioEngine::IoUring(p);
        assert_eq!(e.name(), "io_uring");
        round_trip(&e, "io_uring");
    }
}
