// SPDX-License-Identifier: GPL-2.0-or-later

//! The run state and shutdown requests, system/runstate.c and the `vm_start()` and `vm_stop()`
//! parts of system/cpus.c.

use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_monitor::Qmp;
use ruvm_qapi::events::{event_resume, event_shutdown, event_stop};
use ruvm_qapi::types::{RunState, ShutdownArg, ShutdownCause, StatusInfo};

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `runstate_transitions_def`.
const TRANSITIONS: &[(RunState, RunState)] = {
    use RunState::*;
    &[
        (Debug, Running),
        (Debug, FinishMigrate),
        (Debug, Prelaunch),
        (Inmigrate, InternalError),
        (Inmigrate, IoError),
        (Inmigrate, Paused),
        (Inmigrate, Running),
        (Inmigrate, Shutdown),
        (Inmigrate, Suspended),
        (Inmigrate, Watchdog),
        (Inmigrate, GuestPanicked),
        (Inmigrate, FinishMigrate),
        (Inmigrate, Prelaunch),
        (Inmigrate, Postmigrate),
        (Inmigrate, Colo),
        (InternalError, Paused),
        (InternalError, FinishMigrate),
        (InternalError, Prelaunch),
        (IoError, Running),
        (IoError, FinishMigrate),
        (IoError, Prelaunch),
        (Paused, Running),
        (Paused, FinishMigrate),
        (Paused, Postmigrate),
        (Paused, Prelaunch),
        (Paused, Colo),
        (Paused, Suspended),
        (Postmigrate, Running),
        (Postmigrate, FinishMigrate),
        (Postmigrate, Prelaunch),
        (Prelaunch, Running),
        (Prelaunch, FinishMigrate),
        (Prelaunch, Inmigrate),
        (Prelaunch, Suspended),
        (FinishMigrate, Running),
        (FinishMigrate, Paused),
        (FinishMigrate, Postmigrate),
        (FinishMigrate, Prelaunch),
        (FinishMigrate, Colo),
        (FinishMigrate, InternalError),
        (FinishMigrate, IoError),
        (FinishMigrate, Shutdown),
        (FinishMigrate, Suspended),
        (FinishMigrate, Watchdog),
        (FinishMigrate, GuestPanicked),
        (RestoreVm, Running),
        (RestoreVm, Prelaunch),
        (RestoreVm, Suspended),
        (Colo, Running),
        (Colo, Prelaunch),
        (Colo, Shutdown),
        (Running, Debug),
        (Running, InternalError),
        (Running, IoError),
        (Running, Paused),
        (Running, FinishMigrate),
        (Running, RestoreVm),
        (Running, SaveVm),
        (Running, Shutdown),
        (Running, Watchdog),
        (Running, GuestPanicked),
        (Running, Colo),
        (SaveVm, Running),
        (SaveVm, Suspended),
        (Shutdown, Paused),
        (Shutdown, FinishMigrate),
        (Shutdown, Prelaunch),
        (Shutdown, Colo),
        (Debug, Suspended),
        (Running, Suspended),
        (Suspended, Running),
        (Suspended, FinishMigrate),
        (Suspended, Prelaunch),
        (Suspended, Colo),
        (Suspended, Paused),
        (Suspended, SaveVm),
        (Suspended, RestoreVm),
        (Suspended, Shutdown),
        (Watchdog, Running),
        (Watchdog, FinishMigrate),
        (Watchdog, Prelaunch),
        (Watchdog, Colo),
        (GuestPanicked, Running),
        (GuestPanicked, FinishMigrate),
        (GuestPanicked, Prelaunch),
    ]
};

/// `runstate_is_live()`: the vCPU clock ticks in these states.
pub fn is_live(state: RunState) -> bool {
    matches!(state, RunState::Running | RunState::Suspended)
}

/// `shutdown_caused_by_guest()`.
pub fn caused_by_guest(cause: ShutdownCause) -> bool {
    cause as usize >= ShutdownCause::GuestShutdown as usize
}

/// A termination signal behind a shutdown request, `shutdown_signal` and `shutdown_pid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Killed {
    pub signo: i32,
    pub pid: i32,
}

#[derive(Debug)]
struct Inner {
    state: RunState,
    vm_was_suspended: bool,
    shutdown_requested: ShutdownCause,
    /// `shutdown_exit_code`.
    shutdown_exit_code: i32,
    killed: Option<Killed>,
}

/// What `pause_all_vcpus()` and `resume_all_vcpus()` do for the machine: called with false
/// to stop the vCPUs and with true to let them run.
pub type CpuHook = Arc<dyn Fn(bool) + Send + Sync>;

/// The run state of the machine, with the events QMP clients see when it changes.
pub struct Runstate {
    inner: Mutex<Inner>,
    qmp: Arc<Qmp>,
    cpus: Mutex<Option<CpuHook>>,
}

impl std::fmt::Debug for Runstate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runstate").field("inner", &self.inner).finish_non_exhaustive()
    }
}

impl Runstate {
    /// A machine in `prelaunch`, where `qemu_init()` leaves it.
    pub fn new(qmp: Arc<Qmp>) -> Arc<Self> {
        Arc::new(Runstate {
            inner: Mutex::new(Inner {
                state: RunState::Prelaunch,
                vm_was_suspended: false,
                shutdown_requested: ShutdownCause::None,
                shutdown_exit_code: 0,
                killed: None,
            }),
            qmp,
            cpus: Mutex::new(None),
        })
    }

    /// Installs what starts and stops the vCPUs. Without one there are no vCPUs to run.
    pub fn set_cpu_hook(&self, hook: Option<CpuHook>) {
        *lock(&self.cpus) = hook;
    }

    fn cpus(&self, run: bool) {
        let hook = lock(&self.cpus).clone();
        if let Some(h) = hook {
            h(run);
        }
    }

    fn emit(&self, event: Option<ruvm_qapi::QDict>) {
        if let Some(e) = event {
            self.qmp.emit_event(e);
        }
    }

    /// `runstate_get()`.
    pub fn get(&self) -> RunState {
        lock(&self.inner).state
    }

    /// `runstate_is_running()`.
    pub fn is_running(&self) -> bool {
        self.get() == RunState::Running
    }

    /// `runstate_needs_reset()`.
    pub fn needs_reset(&self) -> bool {
        matches!(self.get(), RunState::InternalError | RunState::Shutdown)
    }

    fn set_locked(inner: &mut Inner, new: RunState) {
        if inner.state == new {
            return;
        }
        if !TRANSITIONS.contains(&(inner.state, new)) {
            error_report(&format!(
                "invalid runstate transition: '{}' -> '{}'",
                inner.state.as_str(),
                new.as_str()
            ));
            std::process::abort();
        }
        inner.state = new;
    }

    /// `runstate_set()`. An invalid transition is a bug, and aborts as it does in QEMU.
    pub fn set(&self, new: RunState) {
        Self::set_locked(&mut lock(&self.inner), new);
    }

    /// `query-status`.
    pub fn status(&self) -> StatusInfo {
        let state = self.get();
        StatusInfo { running: state == RunState::Running, status: state }
    }

    /// `vm_stop()`, which is `do_vm_stop(state, true)`. On a vCPU thread the vCPUs are only
    /// asked to stop.
    pub fn vm_stop(&self, state: RunState) {
        if !is_live(self.get()) {
            return;
        }
        self.cpus(false);
        let mut inner = lock(&self.inner);
        let old = inner.state;
        if !is_live(old) {
            return;
        }
        inner.vm_was_suspended = old == RunState::Suspended;
        Self::set_locked(&mut inner, state);
        drop(inner);
        self.emit(event_stop(&self.qmp.policy()));
    }

    /// `vm_shutdown()`, `do_vm_stop(RUN_STATE_SHUTDOWN, false)`: no STOP event, since the
    /// process is on its way out.
    pub fn vm_shutdown(&self) {
        if is_live(self.get()) {
            self.cpus(false);
        }
        let mut inner = lock(&self.inner);
        if is_live(inner.state) {
            inner.vm_was_suspended = inner.state == RunState::Suspended;
            Self::set_locked(&mut inner, RunState::Shutdown);
        }
    }

    /// `vm_start()`: `vm_prepare_start()` and then `resume_all_vcpus()`.
    pub fn vm_start(&self) {
        let mut inner = lock(&self.inner);
        if inner.state == RunState::Running {
            return;
        }
        let state = if inner.vm_was_suspended { RunState::Suspended } else { RunState::Running };
        drop(inner);
        self.emit(event_resume(&self.qmp.policy()));
        inner = lock(&self.inner);
        Self::set_locked(&mut inner, state);
        inner.vm_was_suspended = false;
        drop(inner);
        self.cpus(true);
    }

    /// `qmp_stop()`.
    pub fn qmp_stop(&self) -> Result<()> {
        self.vm_stop(RunState::Paused);
        Ok(())
    }

    /// `qmp_cont()`.
    pub fn qmp_cont(&self) -> Result<()> {
        match self.get() {
            RunState::InternalError | RunState::Shutdown => {
                Err(Error::generic("Resetting the Virtual Machine is required"))
            }
            RunState::Suspended => Ok(()),
            RunState::FinishMigrate => Err(Error::generic("Migration is not finalized yet")),
            RunState::Colo => Err(Error::generic("COLO checkpoint in progress")),
            _ => {
                self.vm_start();
                Ok(())
            }
        }
    }

    /// `qemu_system_shutdown_request()`. The main loop picks the request up once the
    /// dispatcher has finished the command it is running.
    pub fn shutdown_request(&self, cause: ShutdownCause) {
        lock(&self.inner).shutdown_requested = cause;
        self.qmp.shutdown();
    }

    /// `qemu_system_shutdown_request_with_code()`: a shutdown request that also sets the
    /// exit status of the process, as `isa-debug-exit` makes.
    pub fn shutdown_request_with_code(&self, cause: ShutdownCause, code: i32) {
        lock(&self.inner).shutdown_exit_code = code;
        self.shutdown_request(cause);
    }

    /// The exit status `qemu_main_loop()` returns: the code of the last
    /// [`Runstate::shutdown_request_with_code`], 0 without one.
    pub fn exit_code(&self) -> i32 {
        lock(&self.inner).shutdown_exit_code
    }

    /// `qemu_system_killed()`, called from the thread that watches for signals.
    pub fn killed(&self, killed: Killed) {
        let mut inner = lock(&self.inner);
        inner.killed = Some(killed);
        inner.shutdown_requested = ShutdownCause::HostSignal;
        drop(inner);
        self.qmp.shutdown();
    }

    /// `qemu_shutdown_requested()`: takes the pending request, if any.
    pub fn take_shutdown_request(&self) -> ShutdownCause {
        std::mem::take(&mut lock(&self.inner).shutdown_requested)
    }

    /// The signal behind the last shutdown request, cleared by reading it, as
    /// `qemu_kill_report()` clears `shutdown_signal`.
    pub fn take_killed(&self) -> Option<Killed> {
        lock(&self.inner).killed.take()
    }

    /// `qemu_system_shutdown()`: the SHUTDOWN event.
    pub fn send_shutdown_event(&self, cause: ShutdownCause) {
        let arg = ShutdownArg { guest: caused_by_guest(cause), reason: cause };
        self.emit(event_shutdown(&self.qmp.policy(), arg));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitions() {
        let rs = Runstate::new(Qmp::new());
        assert_eq!(rs.get(), RunState::Prelaunch);
        rs.vm_stop(RunState::Paused);
        assert_eq!(rs.get(), RunState::Prelaunch);
        rs.qmp_cont().unwrap();
        assert!(rs.status().running);
        rs.qmp_stop().unwrap();
        assert_eq!(rs.status(), StatusInfo { running: false, status: RunState::Paused });
        rs.qmp_cont().unwrap();
        rs.set(RunState::Shutdown);
        let e = rs.qmp_cont().unwrap_err();
        assert_eq!(e.message(), "Resetting the Virtual Machine is required");
    }

    #[test]
    fn shutdown_requests() {
        let rs = Runstate::new(Qmp::new());
        assert_eq!(rs.take_shutdown_request(), ShutdownCause::None);
        rs.killed(Killed { signo: 15, pid: 42 });
        assert_eq!(rs.take_shutdown_request(), ShutdownCause::HostSignal);
        assert_eq!(rs.take_shutdown_request(), ShutdownCause::None);
        assert_eq!(rs.take_killed(), Some(Killed { signo: 15, pid: 42 }));
        assert!(!caused_by_guest(ShutdownCause::HostQmpQuit));
        assert!(caused_by_guest(ShutdownCause::GuestShutdown));
        assert_eq!(rs.exit_code(), 0);
        rs.shutdown_request_with_code(ShutdownCause::GuestShutdown, 3);
        assert_eq!(rs.take_shutdown_request(), ShutdownCause::GuestShutdown);
        assert_eq!(rs.exit_code(), 3);
    }

    #[test]
    fn the_cpu_hook_follows_the_runstate() {
        let rs = Runstate::new(Qmp::new());
        let calls = Arc::new(Mutex::new(Vec::new()));
        let c = Arc::clone(&calls);
        rs.set_cpu_hook(Some(Arc::new(move |run| c.lock().unwrap().push(run))));
        rs.qmp_cont().unwrap();
        rs.qmp_stop().unwrap();
        rs.qmp_cont().unwrap();
        rs.vm_shutdown();
        rs.vm_shutdown();
        assert_eq!(*calls.lock().unwrap(), vec![true, false, true, false]);
    }
}
