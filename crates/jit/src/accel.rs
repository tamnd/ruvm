// SPDX-License-Identifier: GPL-2.0-or-later

//! The vCPU threads, `accel/tcg/tcg-accel-ops-mttcg.c` and `tcg-accel-ops-rr.c`, with
//! `qemu_wait_io_event()` and `cpu_thread_is_idle()` from `system/cpus.c`.
//!
//! [`start_vcpus`] starts one thread per vCPU when [`crate::JitConfig::mttcg`] is set, and one
//! round robin thread for all of them otherwise.
//!
//! Differences from QEMU:
//!
//! - There is no big lock and no run state. A vCPU waits on its `halt` condition variable, which
//!   is shared by all vCPUs in round robin mode, and [`crate::CpuShared::kick`] wakes it. A
//!   vCPU is stopped only by its own `stop` and `stopped` flags.
//! - `EXCP_DEBUG` sets `stopped`, standing for `cpu_handle_guest_debug()` without a gdb stub.
//! - The vCPUs run as soon as their thread starts, without waiting for a machine start.
//! - The round robin kick timer is a thread that kicks every 100 ms of host time.
//! - The round robin thread ends once every vCPU was unplugged, and every thread hands its
//!   vCPUs back when it ends.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::cpu::{Cpu, CpuShared, Vcpu};
use crate::cpu_exec::{cpu_exec_step_atomic, tcg_cpu_exec};
use crate::excp;
use crate::jit::Jit;
use crate::tb::lock;

/// `TCG_KICK_PERIOD`.
const TCG_KICK_PERIOD: Duration = Duration::from_millis(100);

/// `cpu_is_stopped()` without a run state.
fn cpu_is_stopped(cpu: &CpuShared) -> bool {
    cpu.stopped.load(Ordering::Acquire)
}

/// `cpu_can_run()`.
fn cpu_can_run(cpu: &CpuShared) -> bool {
    !cpu.stop.load(Ordering::Acquire) && !cpu_is_stopped(cpu)
}

/// `cpu_thread_is_idle()`.
fn cpu_thread_is_idle(cpu: &Cpu<'_>) -> bool {
    let shared = &cpu.core.shared;
    if shared.stop.load(Ordering::Acquire) || !shared.work_list_empty() {
        return false;
    }
    if cpu_is_stopped(shared) {
        return true;
    }
    if shared.halted.load(Ordering::Acquire) == 0 || cpu.has_work() {
        return false;
    }
    true
}

/// `cpu_handle_guest_debug()`.
fn cpu_handle_guest_debug(cpu: &CpuShared) {
    cpu.stopped.store(true, Ordering::Release);
}

/// `qemu_wait_io_event_common()`.
fn qemu_wait_io_event_common(cpu: &mut Cpu<'_>) {
    let shared = cpu.shared();
    shared.thread_kicked.store(false, Ordering::SeqCst);
    if shared.stop.load(Ordering::Acquire) {
        // qemu_cpu_stop()
        shared.stop.store(false, Ordering::Release);
        shared.stopped.store(true, Ordering::Release);
    }
    cpu.process_queued_cpu_work();
}

/// `qemu_wait_io_event()`.
fn qemu_wait_io_event(cpu: &mut Cpu<'_>) {
    let halt = cpu.core.shared.halt.clone();
    let mut slept = false;
    {
        let mut g = lock(&halt.0);
        while cpu_thread_is_idle(cpu) {
            if !slept {
                slept = true;
                if crate::plugin::enabled(cpu) {
                    // The plugin runs without the lock; look again before waiting.
                    drop(g);
                    crate::plugin::vcpu_idle(cpu);
                    g = lock(&halt.0);
                    continue;
                }
            }
            g = halt.1.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
    if slept {
        crate::plugin::vcpu_resume(cpu);
    }
    qemu_wait_io_event_common(cpu);
}

fn register_thread(cpu: &CpuShared) {
    *lock(&cpu.thread) = Some(std::thread::current().id());
}

/// `mttcg_cpu_thread_fn()`.
fn mttcg_cpu_thread_fn(mut vcpu: Vcpu) -> Vec<Vcpu> {
    let shared = vcpu.shared().clone();
    register_thread(&shared);
    let mut cpu = vcpu.cpu();
    cpu.set_can_do_io(true);

    // process any pending work
    shared.exit_request.store(true, Ordering::SeqCst);

    loop {
        if cpu_can_run(&shared) {
            let r = tcg_cpu_exec(&mut cpu);
            match r {
                excp::DEBUG => cpu_handle_guest_debug(&shared),
                // Usually halted is set, but may have already been reset by another thread by
                // the time we arrive here.
                excp::HALTED => {}
                excp::ATOMIC => cpu_exec_step_atomic(&mut cpu),
                // Ignore everything else?
                _ => {}
            }
        }
        shared.exit_request.store(false, Ordering::SeqCst);
        qemu_wait_io_event(&mut cpu);
        if shared.unplug.load(Ordering::Acquire) && !cpu_can_run(&shared) {
            break;
        }
    }
    *lock(&shared.thread) = None;
    vec![vcpu]
}

/// `all_cpu_threads_idle()` over the round robin thread's vCPUs.
fn all_cpu_threads_idle(vcpus: &mut [Vcpu]) -> bool {
    vcpus.iter_mut().all(|v| cpu_thread_is_idle(&v.cpu()))
}

/// The round robin kick timer.
struct KickTimer {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<()>,
}

fn rr_start_kick_timer(jit: &Arc<Jit>, timer: &mut Option<KickTimer>, ncpus: usize) {
    if timer.is_some() || ncpus < 2 {
        return;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let jit: Weak<Jit> = Arc::downgrade(jit);
    let s = stop.clone();
    let handle = std::thread::spawn(move || {
        loop {
            std::thread::park_timeout(TCG_KICK_PERIOD);
            if s.load(Ordering::Acquire) {
                break;
            }
            match jit.upgrade() {
                Some(jit) => jit.rr_kick_next_cpu(),
                None => break,
            }
        }
    });
    *timer = Some(KickTimer { stop, handle });
}

fn rr_stop_kick_timer(timer: &mut Option<KickTimer>) {
    if let Some(t) = timer.take() {
        t.stop.store(true, Ordering::Release);
        t.handle.thread().unpark();
        let _ = t.handle.join();
    }
}

/// `rr_wait_io_event()`.
fn rr_wait_io_event(jit: &Arc<Jit>, vcpus: &mut [Vcpu], timer: &mut Option<KickTimer>) {
    {
        let halt = jit.rr_halt.clone();
        let mut g = lock(&halt.0);
        while all_cpu_threads_idle(vcpus) {
            drop(g);
            rr_stop_kick_timer(timer);
            g = lock(&halt.0);
            if !all_cpu_threads_idle(vcpus) {
                break;
            }
            g = halt.1.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }
    rr_start_kick_timer(jit, timer, vcpus.len());
    for v in vcpus.iter_mut() {
        qemu_wait_io_event_common(&mut v.cpu());
    }
}

/// `rr_deal_with_unplugged_cpus()`.
fn rr_deal_with_unplugged_cpus(vcpus: &mut Vec<Vcpu>, gone: &mut Vec<Vcpu>) {
    let mut i = 0;
    while i < vcpus.len() {
        let s = vcpus[i].shared();
        if s.unplug.load(Ordering::Acquire) && !cpu_can_run(s) {
            let v = vcpus.remove(i);
            *lock(&v.shared().thread) = None;
            gone.push(v);
        } else {
            i += 1;
        }
    }
}

/// `rr_cpu_thread_fn()`.
fn rr_cpu_thread_fn(jit: Arc<Jit>, mut vcpus: Vec<Vcpu>) -> Vec<Vcpu> {
    for v in &mut vcpus {
        register_thread(v.shared());
        v.cpu().set_can_do_io(true);
    }
    let mut timer = None;
    let mut gone = Vec::new();
    rr_start_kick_timer(&jit, &mut timer, vcpus.len());

    let mut cur: Option<usize> = Some(0);
    // process any pending work
    if let Some(v) = vcpus.first() {
        v.shared().exit_request.store(true, Ordering::SeqCst);
    }

    while !vcpus.is_empty() {
        let mut i = cur.unwrap_or(0).min(vcpus.len() - 1);
        cur = Some(i);
        loop {
            let shared = vcpus[i].shared().clone();
            if !shared.work_list_empty() || shared.exit_request.load(Ordering::Acquire) {
                break;
            }
            // Store rr_current_cpu before evaluating cpu_can_run().
            *lock(&jit.rr_current_cpu) = Some(Arc::downgrade(&shared));

            if cpu_can_run(&shared) {
                let r = tcg_cpu_exec(&mut vcpus[i].cpu());
                if r == excp::DEBUG {
                    cpu_handle_guest_debug(&shared);
                    break;
                } else if r == excp::ATOMIC {
                    cpu_exec_step_atomic(&mut vcpus[i].cpu());
                    break;
                }
            } else if shared.stop.load(Ordering::Acquire) {
                if shared.unplug.load(Ordering::Acquire) {
                    cur = if i + 1 < vcpus.len() { Some(i + 1) } else { None };
                }
                break;
            }

            if i + 1 < vcpus.len() {
                i += 1;
                cur = Some(i);
            } else {
                cur = None;
                break;
            }
        }

        // Does not need a memory barrier because a spurious wakeup is okay.
        *lock(&jit.rr_current_cpu) = None;

        if let Some(c) = cur {
            let s = vcpus[c].shared();
            if s.exit_request.load(Ordering::Acquire) {
                s.exit_request.store(false, Ordering::SeqCst);
            }
        }

        rr_wait_io_event(&jit, &mut vcpus, &mut timer);
        rr_deal_with_unplugged_cpus(&mut vcpus, &mut gone);
    }
    rr_stop_kick_timer(&mut timer);
    gone
}

/// The running vCPU threads.
#[derive(Debug)]
pub struct VcpuThreads {
    cpus: Vec<Arc<CpuShared>>,
    handles: Vec<JoinHandle<Vec<Vcpu>>>,
}

/// Start the vCPU threads for `vcpus`, `tcg_start_vcpu_thread()`: one per vCPU with MTTCG,
/// one for all of them in round robin mode.
pub fn start_vcpus(jit: &Arc<Jit>, vcpus: Vec<Vcpu>) -> VcpuThreads {
    let cpus = vcpus.iter().map(|v| v.shared().clone()).collect();
    let mut handles = Vec::new();
    if jit.config.mttcg {
        for v in vcpus {
            let name = format!("CPU {}/TCG", v.shared().cpu_index);
            let h = std::thread::Builder::new()
                .name(name)
                .spawn(move || mttcg_cpu_thread_fn(v))
                .expect("cannot create the vCPU thread");
            handles.push(h);
        }
    } else if !vcpus.is_empty() {
        let jit = jit.clone();
        let h = std::thread::Builder::new()
            .name("ALL CPUs/TCG".to_string())
            .spawn(move || rr_cpu_thread_fn(jit, vcpus))
            .expect("cannot create the vCPU thread");
        handles.push(h);
    }
    VcpuThreads { cpus, handles }
}

impl VcpuThreads {
    /// The vCPUs the threads run.
    pub fn cpus(&self) -> &[Arc<CpuShared>] {
        &self.cpus
    }

    /// Wait until every vCPU is stopped, polling. Returns `false` on timeout.
    pub fn wait_all_stopped(&self, timeout: Duration) -> bool {
        let start = std::time::Instant::now();
        loop {
            if self.cpus.iter().all(|c| c.stopped.load(Ordering::Acquire)) {
                return true;
            }
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Unplug every vCPU, `cpu_remove_sync()`, wait for the threads to end and hand the vCPUs
    /// back in `cpu_index` order.
    pub fn stop_and_join(self) -> Vec<Vcpu> {
        for c in &self.cpus {
            c.stop.store(true, Ordering::Release);
            c.unplug.store(true, Ordering::Release);
            c.kick();
        }
        let mut out = Vec::new();
        for h in self.handles {
            match h.join() {
                Ok(v) => out.extend(v),
                Err(e) => std::panic::resume_unwind(e),
            }
        }
        out.sort_by_key(|v| v.shared().cpu_index);
        out
    }
}
