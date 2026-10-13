// SPDX-License-Identifier: GPL-2.0-or-later

//! The vCPU threads of the qtest accelerator, `accel/dummy-cpus.c`: one thread per vCPU that
//! runs no guest code and only waits until it is told to end. They exist so that a vCPU has a
//! host thread id to report, as in QEMU.

use std::io;
use std::sync::mpsc::channel;
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;

/// The dummy vCPU threads of a machine. Dropping it ends them.
#[derive(Debug)]
pub struct DummyVcpus {
    thread_ids: Vec<i64>,
    quit: Arc<(Mutex<bool>, Condvar)>,
    handles: Vec<JoinHandle<()>>,
}

impl DummyVcpus {
    /// `dummy_start_vcpu_thread()` for `n` vCPUs: the threads `CPU <i>/DUMMY`. Returns once
    /// each thread has its id, as `qemu_init_vcpu()` waits for `cpu->created`.
    pub fn start(n: usize) -> io::Result<DummyVcpus> {
        let quit = Arc::new((Mutex::new(false), Condvar::new()));
        let mut v = DummyVcpus { thread_ids: Vec::new(), quit, handles: Vec::new() };
        for i in 0..n {
            let (created, wait) = channel();
            let quit = Arc::clone(&v.quit);
            let h =
                std::thread::Builder::new().name(format!("CPU {i}/DUMMY")).spawn(move || {
                    let _ = created.send(ruvm_sys::thread::thread_id());
                    let (m, cv) = &*quit;
                    let mut q = m.lock().unwrap_or_else(PoisonError::into_inner);
                    while !*q {
                        q = cv.wait(q).unwrap_or_else(PoisonError::into_inner);
                    }
                })?;
            v.handles.push(h);
            v.thread_ids.push(wait.recv().map_err(io::Error::other)?);
        }
        Ok(v)
    }

    /// `cpu->thread_id` of each vCPU, in `cpu_index` order.
    pub fn thread_ids(&self) -> &[i64] {
        &self.thread_ids
    }
}

impl Drop for DummyVcpus {
    fn drop(&mut self) {
        let (m, cv) = &*self.quit;
        *m.lock().unwrap_or_else(PoisonError::into_inner) = true;
        cv.notify_all();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_start_and_end() {
        let v = DummyVcpus::start(3).unwrap();
        let ids = v.thread_ids().to_vec();
        assert_eq!(ids.len(), 3);
        assert!(ids.iter().all(|&id| id > 0));
        if cfg!(target_os = "linux") {
            assert!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2], "{ids:?}");
        }
        drop(v);
    }
}
