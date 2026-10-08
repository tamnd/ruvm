// SPDX-License-Identifier: GPL-2.0-or-later

//! Snapshots of the whole VM: `save_snapshot()`, `load_snapshot()` and `delete_snapshot()` of
//! migration/savevm.c, the `snapshot-save`, `snapshot-load` and `snapshot-delete` jobs and the
//! job commands, and the `savevm`, `loadvm`, `delvm` and `info snapshots` HMP commands that
//! `human-monitor-command` runs.
//!
//! Differences from QEMU:
//!
//! - The human monitor has only the four snapshot commands. Any other command fails with a
//!   QMP error instead of running.
//! - `-loadvm` is not supported yet.
//! - There are no migration blockers, so `save_snapshot()` skips `migration_is_blocked()`, and
//!   no record/replay, so a snapshot has no icount.
//! - Loading a snapshot of a suspended guest does not wake it up, since ruvm has no S3.
//! - Snapshots need a machine ruvm can migrate, which for now is x86 on TCG.

use std::sync::Arc;

use ruvm_base::{Error, Result, bail};
use ruvm_block::{BlockEvent, SnapshotParams};
use ruvm_migration::Migration;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::*;
use ruvm_qapi::events::event_job_status_change;
use ruvm_qapi::types::{JobStatusChangeArg, JobType, RunState};

use crate::runstate::is_live;
use crate::vl::Vm;

fn migration(vm: &Vm) -> Result<&Migration> {
    vm.migration
        .get()
        .ok_or_else(|| Error::generic("snapshots are not supported with this machine by ruvm yet"))
}

/// `vm_resume()`.
fn vm_resume(vm: &Vm, state: RunState) {
    if is_live(state) {
        vm.runstate.vm_start();
    } else {
        vm.runstate.set(state);
    }
}

/// `load_snapshot_resume()`. QEMU also wakes a guest that stayed suspended; ruvm has no S3.
fn load_snapshot_resume(vm: &Vm, state: RunState) {
    vm_resume(vm, state);
}

/// `save_snapshot()`: takes the snapshot `name`, or one named after the time, with the VM
/// state on `vmstate` or the first node that can hold it. `devices` are the nodes to snapshot,
/// `None` for all writable disks.
fn save_snapshot(
    vm: &Vm,
    name: Option<&str>,
    overwrite: bool,
    vmstate: Option<&str>,
    devices: Option<&[String]>,
) -> Result<()> {
    let saved_state = vm.runstate.get();
    let m = migration(vm)?;
    m.can_snapshot()?;
    let block = &vm.block;
    block.all_can_snapshot(devices)?;
    // Delete old snapshots of the same name.
    if let Some(name) = name {
        if overwrite {
            block.all_delete_snapshot(name, devices)?;
        } else if block.all_has_snapshot(name, devices)? {
            bail!("Snapshot '{name}' already exists in one or more devices");
        }
    }
    let bs = block.all_find_vmstate_bs(vmstate, devices)?;

    m.global_state_store();
    vm.runstate.vm_stop(RunState::SaveVm);
    let drain = block.drain_all_section();
    let sn = SnapshotParams::now(name, m.vm_clock_ns());
    let res = (|| {
        let mut f = bs.channel();
        let ret = m.save_snapshot_state(&mut f);
        let ret2 = f.close();
        let size = ret?;
        ret2?;
        block.all_create_snapshot(&sn, &bs, size, devices).inspect_err(|_| {
            let _ = block.all_delete_snapshot(&sn.name, devices);
        })
    })();
    drop(drain);
    vm_resume(vm, saved_state);
    res
}

/// `load_snapshot()`: reverts the disks to the snapshot `name` and loads its VM state. The
/// guest must be stopped.
fn load_snapshot(
    vm: &Vm,
    name: &str,
    vmstate: Option<&str>,
    devices: Option<&[String]>,
) -> Result<()> {
    let m = migration(vm)?;
    m.can_snapshot()?;
    let block = &vm.block;
    block.all_can_snapshot(devices)?;
    if !block.all_has_snapshot(name, devices)? {
        bail!("Snapshot '{name}' does not exist in one or more devices");
    }
    let bs = block.all_find_vmstate_bs(vmstate, devices)?;
    // Don't even try to load empty VM states.
    match bs.snapshot_vm_state_size(name) {
        None => bail!("Snapshot can not be found"),
        Some(0) => bail!("This is a disk-only snapshot. Revert to it  offline using qemu-img"),
        Some(_) => {}
    }
    // Flush all IO requests so they don't interfere with the new state.
    let _drain = block.drain_all_section();
    block.all_goto_snapshot(name, devices)?;
    m.load_snapshot_state(bs.channel())
}

/// `delete_snapshot()`.
fn delete_snapshot(vm: &Vm, name: &str, devices: Option<&[String]>) -> Result<()> {
    vm.block.all_can_snapshot(devices)?;
    vm.block.all_delete_snapshot(name, devices)
}

/// `hmp_loadvm()` and `snapshot_load_job_bh()`: stops the guest, loads the snapshot and, if
/// that worked, puts the guest back in the state it was in.
fn loadvm(vm: &Vm, name: &str, vmstate: Option<&str>, devices: Option<&[String]>) -> Result<()> {
    let saved_state = vm.runstate.get();
    vm.runstate.vm_stop(RunState::RestoreVm);
    load_snapshot(vm, name, vmstate, devices)?;
    load_snapshot_resume(vm, saved_state);
    Ok(())
}

/// `get_str()`: a word, or a string in double quotes with `\n`, `\r`, `\\`, `\'` and `\"`
/// escapes. Gives the string and the rest of the line, or `None` with nothing to read.
fn get_str(p: &str) -> Option<(String, &str)> {
    let p = p.trim_start_matches(is_space);
    if p.is_empty() {
        return None;
    }
    let mut out = String::new();
    if let Some(rest) = p.strip_prefix('"') {
        let mut it = rest.char_indices();
        while let Some((i, c)) = it.next() {
            match c {
                '"' => return Some((out, &rest[i + 1..])),
                '\\' => match it.next().map(|e| e.1) {
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some(c @ ('\\' | '\'' | '"')) => out.push(c),
                    // QEMU prints "unsupported escape code" to its stdout here.
                    _ => return None,
                },
                c => out.push(c),
            }
        }
        // QEMU prints "unterminated string" to its stdout here.
        return None;
    }
    let end = p.find(is_space).unwrap_or(p.len());
    out.push_str(&p[..end]);
    Some((out, &p[end..]))
}

fn is_space(c: char) -> bool {
    // qemu_isspace()
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// An HMP command of this module and its one string argument, `name:s` or `name:s?`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hmp {
    Savevm,
    Loadvm,
    Delvm,
    InfoSnapshots,
}

/// `handle_hmp_command()` for the commands of this module: what the monitor prints.
fn hmp(vm: &Vm, cmdline: &str) -> Result<String> {
    let start = cmdline;
    // get_command_name()
    let name_of = |p: &str| -> Option<(usize, usize)> {
        let s = p.len() - p.trim_start_matches(is_space).len();
        if s == p.len() {
            return None;
        }
        let len = p[s..].find(|c: char| c == '/' || is_space(c)).unwrap_or(p.len() - s);
        Some((s, s + len))
    };
    let Some((s, e)) = name_of(start) else { return Ok(String::new()) };
    let cmd = match &start[s..e] {
        "savevm" => Hmp::Savevm,
        "loadvm" => Hmp::Loadvm,
        "delvm" => Hmp::Delvm,
        "info" => {
            let rest = start[e..].trim_start_matches(is_space);
            let Some((s2, e2)) = name_of(rest) else {
                bail!("ruvm's human monitor has no 'info' listing yet");
            };
            let off = start.len() - rest.len();
            if &rest[s2..e2] != "snapshots" {
                bail!("ruvm's human monitor does not have 'info {}' yet", &rest[s2..e2]);
            }
            return parse_args(vm, start, off + e2, Hmp::InfoSnapshots);
        }
        other => bail!("ruvm's human monitor does not have '{other}' yet"),
    };
    parse_args(vm, start, e, cmd)
}

/// `monitor_parse_arguments()` and the command itself. `args` is where the arguments of
/// `cmdline` start.
fn parse_args(vm: &Vm, cmdline: &str, args: usize, cmd: Hmp) -> Result<String> {
    let name = match cmd {
        Hmp::Savevm => "savevm",
        Hmp::Loadvm => "loadvm",
        Hmp::Delvm => "delvm",
        Hmp::InfoSnapshots => "snapshots",
    };
    // monitor_parse_command() skips the space after the name.
    let after = &cmdline[args..];
    let args = cmdline.len() - after.trim_start_matches(is_space).len();
    let try_help = || {
        let shown = cmdline[..args].trim_end_matches(is_space);
        format!("Try \"help {shown}\" for more information\n")
    };
    let mut p = &cmdline[args..];
    let mut arg = None;
    if cmd != Hmp::InfoSnapshots {
        let optional = cmd == Hmp::Savevm;
        if !(optional && p.trim_start_matches(is_space).is_empty()) {
            match get_str(p) {
                Some((s, rest)) => {
                    arg = Some(s);
                    p = rest;
                }
                None => return Ok(format!("{name}: string expected\n{}", try_help())),
            }
        }
    }
    if !p.trim_start_matches(is_space).is_empty() {
        return Ok(format!("{name}: extraneous characters at the end of line\n{}", try_help()));
    }
    let res = match cmd {
        Hmp::Savevm => save_snapshot(vm, arg.as_deref(), true, None, None),
        Hmp::Loadvm => loadvm(vm, arg.as_deref().unwrap_or_default(), None, None),
        Hmp::Delvm => delete_snapshot(vm, arg.as_deref().unwrap_or_default(), None),
        Hmp::InfoSnapshots => return Ok(vm.block.info_snapshots()),
    };
    // hmp_handle_error()
    Ok(match res {
        Ok(()) => String::new(),
        Err(e) => format!("Error: {}\n", e.message()),
    })
}

/// Registers `human-monitor-command`, the `snapshot-*` jobs and the job commands, and sends
/// `JOB_STATUS_CHANGE` for the jobs.
pub(crate) fn register(vm: &Arc<Vm>, cmds: &mut Commands) {
    let v = vm.clone();
    // monitor_puts() puts a carriage return before every newline.
    register_human_monitor_command(cmds, move |_: &MonitorQmp, arg| {
        hmp(&v, &arg.command_line).map(|out| out.replace('\n', "\r\n"))
    });

    let v = vm.clone();
    register_snapshot_save(cmds, move |_: &MonitorQmp, arg| {
        let (w, id) = (v.clone(), arg.job_id.clone());
        v.block.start_oneshot_job(
            &id,
            JobType::SnapshotSave,
            Box::new(move || {
                save_snapshot(&w, Some(&arg.tag), false, Some(&arg.vmstate), Some(&arg.devices))
            }),
        )
    });
    let v = vm.clone();
    register_snapshot_load(cmds, move |_: &MonitorQmp, arg| {
        let (w, id) = (v.clone(), arg.job_id.clone());
        v.block.start_oneshot_job(
            &id,
            JobType::SnapshotLoad,
            Box::new(move || loadvm(&w, &arg.tag, Some(&arg.vmstate), Some(&arg.devices))),
        )
    });
    let v = vm.clone();
    register_snapshot_delete(cmds, move |_: &MonitorQmp, arg| {
        let (w, id) = (v.clone(), arg.job_id.clone());
        v.block.start_oneshot_job(
            &id,
            JobType::SnapshotDelete,
            Box::new(move || delete_snapshot(&w, &arg.tag, Some(&arg.devices))),
        )
    });

    let v = vm.clone();
    register_query_jobs(cmds, move |_: &MonitorQmp| Ok(v.block.query_jobs()));
    let v = vm.clone();
    register_job_cancel(cmds, move |_: &MonitorQmp, a| v.block.job_cancel(&a.id));
    let v = vm.clone();
    register_job_pause(cmds, move |_: &MonitorQmp, a| v.block.job_pause(&a.id));
    let v = vm.clone();
    register_job_resume(cmds, move |_: &MonitorQmp, a| v.block.job_resume(&a.id));
    let v = vm.clone();
    register_job_complete(cmds, move |_: &MonitorQmp, a| v.block.job_complete(&a.id));
    let v = vm.clone();
    register_job_finalize(cmds, move |_: &MonitorQmp, a| v.block.job_finalize(&a.id));
    let v = vm.clone();
    register_job_dismiss(cmds, move |_: &MonitorQmp, a| v.block.job_dismiss(&a.id));

    let qmp = vm.qmp.clone();
    ruvm_block::set_event_hook(Some(Arc::new(move |ev: &BlockEvent| {
        if let BlockEvent::JobStatusChange { id, status } = ev {
            let arg = JobStatusChangeArg { id: id.clone(), status: *status };
            if let Some(e) = event_job_status_change(&qmp.policy(), arg) {
                qmp.emit_event(e);
            }
        }
    })));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings() {
        assert_eq!(get_str("  ab cd"), Some(("ab".into(), " cd")));
        assert_eq!(get_str(r#" "a \"b\"\n" x"#), Some(("a \"b\"\n".into(), " x")));
        assert_eq!(get_str("   "), None);
        assert_eq!(get_str(r#""open"#), None);
        assert_eq!(get_str(r#""\q""#), None);
    }
}
