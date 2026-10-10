// SPDX-License-Identifier: GPL-2.0-or-later

//! The human monitor that `human-monitor-command` runs: the command lookup and argument
//! parsing of monitor/hmp.c, `help`, the `info` commands ruvm has, and the snapshot commands
//! of [`crate::snapshot`].
//!
//! Differences from QEMU:
//!
//! - Only the commands in [`HMP_CMDS`] and [`INFO_CMDS`] run. Another command QEMU has prints
//!   that ruvm's human monitor does not have it yet, the way a command QEMU does not know
//!   prints `unknown command`, so `human-monitor-command` still succeeds.
//! - `help` lists only the commands ruvm has, the way QEMU leaves out the ones that are not
//!   available, and there is no `help log`.
//! - There is no `-preconfig`, so every command is available.

use std::sync::Arc;

use ruvm_monitor::control::version_info;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::register_human_monitor_command;
use ruvm_qapi::types::{Accelerator, RunState};

use crate::snapshot::{delete_snapshot, loadvm, save_snapshot};
use crate::vl::{Vm, accels};

/// What a command runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cmd {
    Help,
    Info,
    Savevm,
    Loadvm,
    Delvm,
    InfoVersion,
    InfoKvm,
    InfoSnapshots,
    InfoStatus,
    InfoName,
    InfoNetwork,
    InfoUuid,
}

/// The arguments of a command, its `args_type`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Args {
    /// `""`.
    None,
    /// `name:s`.
    Str,
    /// `name:s?`.
    OptStr,
    /// `name:S?`, the rest of the line.
    OptRest,
}

/// An entry of `hmp_cmds` or `hmp_info_cmds`.
struct HmpCommand {
    name: &'static str,
    args: Args,
    params: &'static str,
    help: &'static str,
    cmd: Cmd,
}

const fn command(
    name: &'static str,
    args: Args,
    params: &'static str,
    help: &'static str,
    cmd: Cmd,
) -> HmpCommand {
    HmpCommand { name, args, params, help, cmd }
}

/// The commands of hmp-commands.hx that ruvm has, sorted by name the way `sortcmdlist()`
/// sorts them.
const HMP_CMDS: &[HmpCommand] = &[
    command("delvm", Args::Str, "tag", "delete a VM snapshot from its tag", Cmd::Delvm),
    command("help|?", Args::OptRest, "[cmd]", "show the help", Cmd::Help),
    command(
        "info",
        Args::OptStr,
        "[subcommand]",
        "show various information about the system state",
        Cmd::Info,
    ),
    command("loadvm", Args::Str, "tag", "restore a VM snapshot from its tag", Cmd::Loadvm),
    command(
        "savevm",
        Args::OptStr,
        "tag",
        "save a VM snapshot. If no tag is provided, a new snapshot is created",
        Cmd::Savevm,
    ),
];

/// The commands of hmp-commands-info.hx that ruvm has, sorted the same way.
const INFO_CMDS: &[HmpCommand] = &[
    command("kvm", Args::None, "", "show KVM information", Cmd::InfoKvm),
    command("name", Args::None, "", "show the current VM name", Cmd::InfoName),
    command("network", Args::None, "", "show the network state", Cmd::InfoNetwork),
    command(
        "snapshots",
        Args::None,
        "",
        "show the currently saved VM snapshots",
        Cmd::InfoSnapshots,
    ),
    command(
        "status",
        Args::None,
        "",
        "show the current VM status (running|paused)",
        Cmd::InfoStatus,
    ),
    command("uuid", Args::None, "", "show the current VM UUID", Cmd::InfoUuid),
    command("version", Args::None, "", "show the version of QEMU", Cmd::InfoVersion),
];

/// The names of every command of hmp-commands.hx, so ruvm can tell the QEMU commands it does
/// not have from the ones nobody has.
const QEMU_CMDS: &[&str] = &[
    "help|?",
    "clear",
    "commit",
    "quit|q",
    "exit_preconfig",
    "block_resize",
    "block_stream",
    "block_job_set_speed",
    "block_job_cancel",
    "block_job_complete",
    "block_job_pause",
    "block_job_resume",
    "eject",
    "drive_del",
    "change",
    "screendump",
    "logfile",
    "trace-event",
    "trace-file",
    "log",
    "savevm",
    "loadvm",
    "delvm",
    "one-insn-per-tb",
    "stop|s",
    "cont|c",
    "system_wakeup",
    "gdbserver",
    "x",
    "xp",
    "gpa2hva",
    "gpa2hpa",
    "gva2gpa",
    "print|p",
    "i",
    "o",
    "sendkey",
    "sync-profile",
    "system_reset",
    "system_powerdown",
    "sum",
    "device_add",
    "device_del",
    "cpu",
    "mouse_move",
    "mouse_button",
    "mouse_set",
    "wavcapture",
    "stopcapture",
    "memsave",
    "pmemsave",
    "boot_set",
    "nmi",
    "ringbuf_write",
    "ringbuf_read",
    "announce_self",
    "migrate",
    "migrate_cancel",
    "migrate_continue",
    "migrate_incoming",
    "migrate_recover",
    "migrate_pause",
    "migrate_set_capability",
    "migrate_set_parameter",
    "migrate_start_postcopy",
    "x_colo_lost_heartbeat",
    "client_migrate_info",
    "dump-guest-memory",
    "dump-skeys",
    "migration_mode",
    "snapshot_blkdev",
    "snapshot_blkdev_internal",
    "snapshot_delete_blkdev_internal",
    "drive_mirror",
    "drive_backup",
    "drive_add",
    "pcie_aer_inject_error",
    "netdev_add",
    "netdev_del",
    "object_add",
    "object_del",
    "hostfwd_add",
    "hostfwd_remove",
    "balloon",
    "set_link",
    "watchdog_action",
    "nbd_server_start",
    "nbd_server_add",
    "nbd_server_remove",
    "nbd_server_stop",
    "mce",
    "getfd",
    "closefd",
    "block_set_io_throttle",
    "set_password",
    "expire_password",
    "chardev-add",
    "chardev-change",
    "chardev-remove",
    "chardev-send-break",
    "qemu-io",
    "qom-list",
    "qom-get",
    "qom-set",
    "replay_break",
    "replay_delete_break",
    "replay_seek",
    "calc_dirty_rate",
    "set_vcpu_dirty_limit",
    "cancel_vcpu_dirty_limit",
    "dumpdtb",
    "xen-event-inject",
    "xen-event-list",
    "info",
];

/// The names of every command of hmp-commands-info.hx.
const QEMU_INFO_CMDS: &[&str] = &[
    "version",
    "network",
    "chardev",
    "block",
    "blockstats",
    "block-jobs",
    "registers",
    "lapic",
    "cpus",
    "history",
    "irq",
    "pic",
    "pci",
    "tlb",
    "mem",
    "mtree",
    "jit",
    "sync-profile",
    "accel",
    "kvm",
    "accelerators",
    "numa",
    "usb",
    "usbhost",
    "capture",
    "snapshots",
    "status",
    "mice",
    "vnc",
    "spice",
    "name",
    "uuid",
    "usernet",
    "migrate",
    "migrate_capabilities",
    "migrate_parameters",
    "balloon",
    "qtree",
    "qdm",
    "qom-tree",
    "roms",
    "trace-events",
    "tpm",
    "memdev",
    "memory-devices",
    "iothreads",
    "rocker",
    "rocker-ports",
    "rocker-of-dpa-flows",
    "rocker-of-dpa-groups",
    "skeys",
    "cmma",
    "dump",
    "ramblock",
    "hotpluggable-cpus",
    "vm-generation-id",
    "memory_size_summary",
    "sev",
    "replay",
    "dirty_rate",
    "vcpu_dirty_limit",
    "sgx",
    "via",
    "stats",
    "virtio",
    "virtio-status",
    "virtio-queue-status",
    "virtio-vhost-queue-status",
    "virtio-queue-element",
    "cryptodev",
    "firmware-log",
];

fn is_space(c: char) -> bool {
    // qemu_isspace()
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c')
}

/// `hmp_compare_cmd()`: whether `name` is one of the `|` separated names of `list`.
fn compare_cmd(name: &str, list: &str) -> bool {
    list.split('|').any(|n| n == name)
}

/// What the monitor prints for a command it has no entry for: the words of the command line up
/// to and including its name.
fn missing(qemu: &[&str], name: &str, words: &str) -> String {
    if qemu.iter().any(|c| compare_cmd(name, c)) {
        let words: Vec<&str> = words.split(is_space).filter(|w| !w.is_empty()).collect();
        format!("ruvm's human monitor does not have '{}' yet\n", words.join(" "))
    } else {
        format!("unknown command: '{words}'\n")
    }
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

/// `monitor_parse_command()`: the command `cmdline` names and where its arguments start, or
/// what the monitor prints when there is none.
fn parse_command(cmdline: &str) -> Result<(&'static HmpCommand, usize), String> {
    let (mut table, mut qemu) = (HMP_CMDS, QEMU_CMDS);
    let skip = |p: usize| cmdline.len() - cmdline[p..].trim_start_matches(is_space).len();
    let mut p = 0;
    loop {
        // get_command_name()
        let s = skip(p);
        if s == cmdline.len() {
            return Err(String::new());
        }
        let e =
            cmdline[s..].find(|c: char| c == '/' || is_space(c)).map_or(cmdline.len(), |n| s + n);
        let name = &cmdline[s..e];
        let Some(cmd) = table.iter().find(|c| compare_cmd(name, c.name)) else {
            return Err(missing(qemu, name, &cmdline[..e]));
        };
        p = skip(e);
        // Search the sub command.
        if cmd.cmd != Cmd::Info || p == cmdline.len() {
            return Ok((cmd, p));
        }
        (table, qemu) = (INFO_CMDS, QEMU_INFO_CMDS);
    }
}

/// `monitor_parse_arguments()`: the one argument of `cmd`, which starts at `args` of
/// `cmdline`, or what the monitor prints when it does not parse.
fn parse_args(cmdline: &str, args: usize, cmd: &HmpCommand) -> Result<Option<String>, String> {
    let try_help = || {
        let shown = cmdline[..args].trim_end_matches(is_space);
        format!("Try \"help {shown}\" for more information\n")
    };
    let mut p = &cmdline[args..];
    let mut arg = None;
    match cmd.args {
        Args::None => {}
        Args::Str | Args::OptStr => {
            if !(cmd.args == Args::OptStr && p.trim_start_matches(is_space).is_empty()) {
                let Some((s, rest)) = get_str(p) else {
                    return Err(format!("{}: string expected\n{}", cmd.name, try_help()));
                };
                arg = Some(s);
                p = rest;
            }
        }
        Args::OptRest => {
            let rest = p.trim_start_matches(is_space);
            if !rest.is_empty() {
                arg = Some(rest.to_owned());
            }
            p = "";
        }
    }
    if !p.trim_start_matches(is_space).is_empty() {
        return Err(format!(
            "{}: extraneous characters at the end of line\n{}",
            cmd.name,
            try_help()
        ));
    }
    Ok(arg)
}

/// `help_cmd_dump_one()`.
fn help_cmd_dump_one(out: &mut String, cmd: &HmpCommand, prefix: &[String]) {
    for p in prefix {
        out.push_str(p);
        out.push(' ');
    }
    out.push_str(&format!("{} {} -- {}\n", cmd.name, cmd.params, cmd.help));
}

/// `hmp_help_cmd()`: the help of the commands `name` names, or of all of them.
fn help_cmd(name: Option<&str>) -> String {
    // parse_cmdline()
    let mut args = Vec::new();
    if let Some(mut p) = name {
        while let Some((s, rest)) = get_str(p) {
            args.push(s);
            p = rest;
        }
        if !p.trim_start_matches(is_space).is_empty() {
            return String::new();
        }
    }
    // help_cmd_dump()
    let mut out = String::new();
    let (mut table, mut qemu) = (HMP_CMDS, QEMU_CMDS);
    for i in 0..=args.len() {
        let Some(arg) = args.get(i) else {
            table.iter().for_each(|c| help_cmd_dump_one(&mut out, c, &args));
            break;
        };
        let Some(cmd) = table.iter().find(|c| compare_cmd(arg, c.name)) else {
            return missing(qemu, arg, &args[..=i].join(" "));
        };
        if cmd.cmd != Cmd::Info {
            help_cmd_dump_one(&mut out, cmd, &args[..i]);
            break;
        }
        (table, qemu) = (INFO_CMDS, QEMU_INFO_CMDS);
    }
    out
}

/// `handle_hmp_command()`: what the monitor prints for `cmdline`. `kvm_present` is whether
/// the personality has KVM.
fn hmp(vm: &Vm, kvm_present: bool, cmdline: &str) -> String {
    let (cmd, args) = match parse_command(cmdline) {
        Ok(c) => c,
        Err(out) => return out,
    };
    let arg = match parse_args(cmdline, args, cmd) {
        Ok(a) => a,
        Err(out) => return out,
    };
    let res = match cmd.cmd {
        Cmd::Help => return help_cmd(arg.as_deref()),
        // hmp_info_help()
        Cmd::Info => return help_cmd(Some("info")),
        Cmd::Savevm => save_snapshot(vm, arg.as_deref(), true, None, None),
        Cmd::Loadvm => loadvm(vm, arg.as_deref().unwrap_or_default(), None, None),
        Cmd::Delvm => delete_snapshot(vm, arg.as_deref().unwrap_or_default(), None),
        Cmd::InfoVersion => {
            let v = version_info();
            return format!("{}.{}.{}{}\n", v.qemu.major, v.qemu.minor, v.qemu.micro, v.package);
        }
        Cmd::InfoKvm => {
            let state = match (kvm_present, vm.accel.get() == Some(&Accelerator::Kvm)) {
                (false, _) => "not compiled",
                (true, true) => "enabled",
                (true, false) => "disabled",
            };
            return format!("kvm support: {state}\n");
        }
        Cmd::InfoSnapshots => return vm.block.info_snapshots(),
        Cmd::InfoStatus => {
            let info = vm.runstate.status();
            let mut out = format!("VM status: {}", if info.running { "running" } else { "paused" });
            if !info.running && info.status != RunState::Paused {
                out.push_str(&format!(" ({})", info.status.as_str()));
            }
            out.push('\n');
            return out;
        }
        Cmd::InfoName => return vm.name.as_ref().map(|n| format!("{n}\n")).unwrap_or_default(),
        Cmd::InfoNetwork => {
            return vm.network.get().map(|n| n.info_network()).unwrap_or_default();
        }
        Cmd::InfoUuid => return format!("{}\n", crate::display::qemu_uuid_string()),
    };
    // hmp_handle_error()
    match res {
        Ok(()) => String::new(),
        Err(e) => format!("Error: {}\n", e.message()),
    }
}

/// Registers `human-monitor-command`. `target` is the one of the personality, which decides
/// whether `info kvm` says KVM is there.
pub(crate) fn register(vm: &Arc<Vm>, target: &str, cmds: &mut Commands) {
    let v = vm.clone();
    let kvm_present = accels(target).contains(&"kvm");
    // monitor_puts() puts a carriage return before every newline.
    register_human_monitor_command(cmds, move |_: &MonitorQmp, arg| {
        Ok(hmp(&v, kvm_present, &arg.command_line).replace('\n', "\r\n"))
    });
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

    #[test]
    fn commands() {
        let name = |l: &str| parse_command(l).map(|(c, p)| (c.cmd, p));
        assert_eq!(name("  "), Err(String::new()));
        assert_eq!(name("savevm  a"), Ok((Cmd::Savevm, 8)));
        assert_eq!(name("? info"), Ok((Cmd::Help, 2)));
        assert_eq!(name("info"), Ok((Cmd::Info, 4)));
        assert_eq!(name(" info  status "), Ok((Cmd::InfoStatus, 14)));
        assert_eq!(name(" foo bar"), Err("unknown command: ' foo'\n".into()));
        assert_eq!(name("info foo"), Err("unknown command: 'info foo'\n".into()));
        let lacks = "ruvm's human monitor does not have 'info qtree' yet\n";
        assert_eq!(name(" info  qtree"), Err(lacks.into()));
        let lacks = "ruvm's human monitor does not have 'x' yet\n";
        assert_eq!(name("x/8i 0x100"), Err(lacks.into()));
    }

    #[test]
    fn arguments() {
        let args = |l: &str| {
            let (c, p) = parse_command(l).unwrap();
            parse_args(l, p, c)
        };
        assert_eq!(args("savevm"), Ok(None));
        assert_eq!(args("savevm \"a b\""), Ok(Some("a b".into())));
        assert_eq!(args("help  info status "), Ok(Some("info status ".into())));
        let want = "loadvm: string expected\nTry \"help loadvm\" for more information\n";
        assert_eq!(args("loadvm"), Err(want.into()));
        let want = "status: extraneous characters at the end of line\n\
                    Try \"help info status\" for more information\n";
        assert_eq!(args("info status x"), Err(want.into()));
    }

    #[test]
    fn help() {
        for table in [HMP_CMDS, INFO_CMDS] {
            assert!(table.windows(2).all(|w| w[0].name < w[1].name));
        }
        let all = help_cmd(None);
        assert!(all.starts_with("delvm tag -- delete a VM snapshot from its tag\n"), "{all}");
        assert!(all.ends_with("a new snapshot is created\n"), "{all}");
        let info = help_cmd(Some("info"));
        assert!(info.starts_with("info kvm  -- show KVM information\n"), "{info}");
        assert_eq!(info.lines().count(), INFO_CMDS.len());
        assert!(info.lines().all(|l| l.starts_with("info ")));
        assert_eq!(help_cmd(Some("info uuid")), "info uuid  -- show the current VM UUID\n");
        assert_eq!(help_cmd(Some("loadvm")), "loadvm tag -- restore a VM snapshot from its tag\n");
        let lacks = "ruvm's human monitor does not have 'info qtree' yet\n";
        assert_eq!(help_cmd(Some("info qtree")), lacks);
        assert_eq!(help_cmd(Some("frob")), "unknown command: 'frob'\n");
        assert_eq!(help_cmd(Some("\"open")), "");
    }
}
