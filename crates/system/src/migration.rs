// SPDX-License-Identifier: GPL-2.0-or-later

//! What migration needs from the system emulator: the run state changes around the switchover
//! and after an incoming migration (migration_stop_vm(), migration_iteration_finish() and
//! process_incoming_migration_bh()), the MIGRATION and MIGRATION_PASS events, and the reset
//! and virtual clock that snapshots need.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use ruvm_base::report::report_error;
use ruvm_base::{Error, Result};
use ruvm_migration::{GlobalState, Migration, MigrationHost};
use ruvm_monitor::Qmp;
use ruvm_qapi::QValue;
use ruvm_qapi::events::{event_migration, event_migration_pass};
use ruvm_qapi::json;
use ruvm_qapi::keyval::keyval_parse;
use ruvm_qapi::types::{
    MigrationArg, MigrationChannel, MigrationPassArg, MigrationStatus, RunState,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};

use crate::runstate::{Runstate, is_live};

/// `qemu_system_reset()` done right away, for `loadvm`.
pub(crate) type ResetFn = Box<dyn Fn() -> std::result::Result<(), String> + Send + Sync>;
/// `qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL)`.
pub(crate) type ClockFn = Box<dyn Fn() -> u64 + Send + Sync>;

/// What snapshots need from the machine.
pub(crate) struct SnapshotHooks {
    pub(crate) reset: ResetFn,
    pub(crate) vm_clock_ns: ClockFn,
}

impl std::fmt::Debug for SnapshotHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SnapshotHooks")
    }
}

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
    snapshot: Option<SnapshotHooks>,
}

impl Host {
    pub(crate) fn new(
        runstate: Arc<Runstate>,
        qmp: Arc<Qmp>,
        global_state: Arc<GlobalState>,
        autostart: Arc<AtomicBool>,
    ) -> Self {
        Host {
            runstate,
            qmp,
            global_state,
            autostart,
            old_state: Mutex::new(RunState::Running),
            snapshot: None,
        }
    }

    /// The host with the reset and clock `savevm` and `loadvm` use.
    pub(crate) fn with_snapshot_hooks(mut self, hooks: SnapshotHooks) -> Self {
        self.snapshot = Some(hooks);
        self
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

    fn stop_for_snapshot(&self) -> Result<()> {
        // migration_stop_vm(s, RUN_STATE_PAUSED) for a background snapshot.
        let old = self.runstate.get();
        *self.old_state.lock().unwrap_or_else(PoisonError::into_inner) = old;
        self.global_state.store(old, self.runstate.vm_was_suspended());
        self.runstate.vm_stop_force_state(RunState::Paused);
        Ok(())
    }

    fn resume_after_snapshot(&self, _was_running: bool) {
        // vm_resume(s->vm_old_state)
        let old = *self.old_state.lock().unwrap_or_else(PoisonError::into_inner);
        if is_live(old) {
            self.runstate.vm_start();
        } else {
            self.runstate.set(old);
        }
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

    fn global_state_store(&self) {
        self.global_state.store(self.runstate.get(), self.runstate.vm_was_suspended());
    }

    fn snapshot_reset(&self) -> Result<()> {
        // qemu_system_reset(SHUTDOWN_CAUSE_SNAPSHOT_LOAD), which sends no RESET event.
        match &self.snapshot {
            Some(h) => (h.reset)().map_err(Error::generic),
            None => Err(Error::generic("snapshots are not supported with this machine by ruvm")),
        }
    }

    fn snapshot_loaded(&self) {
        self.runstate.set_vm_was_suspended(self.global_state.vm_was_suspended());
    }

    fn vm_clock_ns(&self) -> u64 {
        self.snapshot.as_ref().map_or(0, |h| (h.vm_clock_ns)())
    }
}

/// `migrate_is_uri()`: letters up to the first colon.
fn is_uri(s: &str) -> bool {
    s.split_once(':').is_some_and(|(scheme, _)| scheme.chars().all(|c| c.is_ascii_alphabetic()))
}

/// The `-incoming` argument parsed as `incoming_option_parse()` does: `defer` and URIs give
/// no channel (but a bad URI is an error), anything else is a `MigrationChannel` in JSON or in
/// keyval form with `channel-type` as the implied key.
pub(crate) fn incoming_channel(arg: &str) -> Result<Option<MigrationChannel>> {
    if arg == "defer" {
        return Ok(None);
    }
    if is_uri(arg) {
        ruvm_migration::parse_uri(arg)?;
        return Ok(None);
    }
    // qobject_input_visitor_new_str()
    let mut v = if arg.starts_with('{') {
        QObjectInputVisitor::new(json::from_str(arg)?)
    } else {
        QObjectInputVisitor::new_keyval(QValue::Dict(keyval_parse(
            arg,
            Some("channel-type"),
            None,
        )?))
    };
    let mut channel = MigrationChannel::default();
    MigrationChannel::visit(&mut v, None, &mut channel)?;
    Ok(Some(channel))
}

/// The `qmp_migrate_incoming()` that `-incoming` makes once the machine is ready: on the
/// main channel, a URI or a channel.
pub(crate) fn start_incoming(m: &Migration, arg: &str) -> Result<()> {
    match incoming_channel(arg)? {
        Some(c) => m.incoming(None, Some(std::slice::from_ref(&c)), true),
        None => m.incoming(Some(arg), None, true),
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::types::{MigrationAddressU, MigrationChannelType, SocketAddressU};

    use super::*;

    #[test]
    fn incoming_arguments() {
        assert!(incoming_channel("defer").unwrap().is_none());
        assert!(incoming_channel("unix:/tmp/m.sock").unwrap().is_none());
        assert_eq!(
            incoming_channel("foo:bar").unwrap_err().message(),
            "unknown migration protocol: foo:bar"
        );
        let c = incoming_channel(
            r#"{"channel-type":"cpr","addr":{"transport":"socket","type":"unix","path":"c.sock"}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.channel_type, MigrationChannelType::Cpr);
        let MigrationAddressU::Socket(s) = &c.addr.u else { panic!("{c:?}") };
        assert!(matches!(&s.u, SocketAddressU::Unix(u) if u.path == "c.sock"));
        let c = incoming_channel("main,addr.transport=socket,addr.type=unix,addr.path=m.sock")
            .unwrap()
            .unwrap();
        assert_eq!(c.channel_type, MigrationChannelType::Main);
        assert!(incoming_channel("cpr,addr.transport=nope").is_err());
    }
}
