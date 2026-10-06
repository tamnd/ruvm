// SPDX-License-Identifier: GPL-2.0-or-later

//! What migration needs from the system emulator: the run state changes around the switchover
//! and after an incoming migration (migration_stop_vm(), migration_iteration_finish() and
//! process_incoming_migration_bh()), and the MIGRATION and MIGRATION_PASS events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use ruvm_base::report::report_error;
use ruvm_base::{Error, Result};
use ruvm_migration::{GlobalState, MigrationHost};
use ruvm_monitor::Qmp;
use ruvm_qapi::events::{event_migration, event_migration_pass};
use ruvm_qapi::types::{MigrationArg, MigrationPassArg, MigrationStatus, RunState};

use crate::runstate::{Runstate, is_live};

/// The machine side of a migration.
#[derive(Debug)]
pub(crate) struct Host {
    runstate: Arc<Runstate>,
    qmp: Arc<Qmp>,
    global_state: Arc<GlobalState>,
    /// `autostart`: `-S` clears it, and `cont` during `inmigrate` sets it.
    autostart: Arc<AtomicBool>,
    /// `MigrationState.vm_old_state`.
    old_state: Mutex<RunState>,
}

impl Host {
    pub(crate) fn new(
        runstate: Arc<Runstate>,
        qmp: Arc<Qmp>,
        global_state: Arc<GlobalState>,
        autostart: Arc<AtomicBool>,
    ) -> Self {
        Host { runstate, qmp, global_state, autostart, old_state: Mutex::new(RunState::Running) }
    }
}

impl MigrationHost for Host {
    fn is_running(&self) -> bool {
        self.runstate.is_running()
    }

    fn is_incoming(&self) -> bool {
        self.runstate.get() == RunState::Inmigrate
    }

    fn is_postmigrate(&self) -> bool {
        self.runstate.get() == RunState::Postmigrate
    }

    fn stop_for_switchover(&self) -> Result<()> {
        // migration_stop_vm(): global_state_store() and vm_stop_force_state().
        let old = self.runstate.get();
        *self.old_state.lock().unwrap_or_else(PoisonError::into_inner) = old;
        self.global_state.store(old, self.runstate.vm_was_suspended());
        self.runstate.vm_stop_force_state(RunState::FinishMigrate);
        Ok(())
    }

    fn resume_after_failure(&self, was_running: bool) {
        // migration_iteration_finish() for a failed or cancelled migration.
        if was_running {
            if self.runstate.get() != RunState::Shutdown {
                self.runstate.vm_start();
            }
        } else if self.runstate.get() == RunState::FinishMigrate {
            let old = *self.old_state.lock().unwrap_or_else(PoisonError::into_inner);
            self.runstate.set(old);
        }
    }

    fn set_postmigrate(&self) {
        self.runstate.set(RunState::Postmigrate);
    }

    fn incoming_done(&self) {
        // global_state_post_load() and process_incoming_migration_bh().
        self.runstate.set_vm_was_suspended(self.global_state.vm_was_suspended());
        let target = self.global_state.received().unwrap_or(RunState::Running);
        if is_live(target) {
            if self.autostart.load(Ordering::Acquire) {
                self.runstate.vm_start();
            } else {
                self.runstate.set(RunState::Paused);
            }
        } else {
            self.runstate.set(target);
        }
    }

    fn incoming_failed(&self, err: &Error, exit_on_error: bool) {
        if exit_on_error {
            report_error(err);
            ruvm_chardev::stdio::term_exit();
            std::process::exit(1);
        }
    }

    fn status_event(&self, status: MigrationStatus) {
        if let Some(e) = event_migration(&self.qmp.policy(), MigrationArg { status }) {
            self.qmp.emit_event(e);
        }
    }

    fn pass_event(&self, pass: i64) {
        if let Some(e) = event_migration_pass(&self.qmp.policy(), MigrationPassArg { pass }) {
            self.qmp.emit_event(e);
        }
    }
}
