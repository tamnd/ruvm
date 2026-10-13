// SPDX-License-Identifier: MIT OR Apache-2.0

//! The host thread id QEMU reports for a vCPU thread, `qemu_get_thread_id()`.

/// `qemu_get_thread_id()`: the kernel's id of the calling thread on Linux, the process id
/// elsewhere, as QEMU's POSIX fallback gives. QEMU on Windows gives `GetCurrentThreadId()`.
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn thread_id() -> i64 {
    i64::from(rustix::thread::gettid().as_raw_nonzero().get())
}

/// `qemu_get_thread_id()`: the kernel's id of the calling thread on Linux, the process id
/// elsewhere, as QEMU's POSIX fallback gives. QEMU on Windows gives `GetCurrentThreadId()`.
#[cfg(not(any(target_os = "linux", target_os = "android")))]
pub fn thread_id() -> i64 {
    i64::from(std::process::id())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_differ_on_linux() {
        let main = thread_id();
        assert!(main > 0);
        let other = std::thread::spawn(thread_id).join().unwrap();
        if cfg!(target_os = "linux") {
            assert_ne!(main, other);
        } else {
            assert_eq!(main, other);
        }
    }
}
