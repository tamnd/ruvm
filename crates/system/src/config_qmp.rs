// SPDX-License-Identifier: GPL-2.0-or-later

//! `query-command-line-options`, from monitor/qemu-config-qmp.c.
//!
//! QEMU answers from `vm_config_groups`, the option groups it registers at startup. ruvm lists
//! the groups of the options its command line takes, in the order QEMU registers them, with
//! the parameters of each. The `drive` parameters are QEMU's tables, which `-drive` accepts.

use ruvm_qapi::opts::{QemuOptDesc, QemuOptType, QemuOptsList};
use ruvm_qapi::types::{CommandLineOptionInfo, CommandLineParameterInfo, CommandLineParameterType};
use ruvm_qom::Registry;

/// `qemu_smp_opts`.
fn smp_opts() -> QemuOptsList {
    QemuOptsList::new(
        "smp-opts",
        &[
            QemuOptDesc::new("cpus", QemuOptType::Number),
            QemuOptDesc::new("drawers", QemuOptType::Number),
            QemuOptDesc::new("books", QemuOptType::Number),
            QemuOptDesc::new("sockets", QemuOptType::Number),
            QemuOptDesc::new("dies", QemuOptType::Number),
            QemuOptDesc::new("clusters", QemuOptType::Number),
            QemuOptDesc::new("modules", QemuOptType::Number),
            QemuOptDesc::new("cores", QemuOptType::Number),
            QemuOptDesc::new("threads", QemuOptType::Number),
            QemuOptDesc::new("maxcpus", QemuOptType::Number),
        ],
    )
    .with_implied_opt_name("cpus")
    .with_merge_lists()
}

/// `qemu_legacy_drive_opts` from blockdev.c.
const LEGACY_DRIVE_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("bus", QemuOptType::Number).help("bus number"),
    QemuOptDesc::new("unit", QemuOptType::Number).help("unit number (i.e. lun for scsi)"),
    QemuOptDesc::new("index", QemuOptType::Number).help("index number"),
    QemuOptDesc::new("media", QemuOptType::String).help("media type (disk, cdrom)"),
    QemuOptDesc::new("if", QemuOptType::String)
        .help("interface (ide, scsi, sd, mtd, floppy, pflash, virtio)"),
    QemuOptDesc::new("file", QemuOptType::String).help("file name"),
    // Options that are passed on, but have special semantics with -drive.
    QemuOptDesc::new("read-only", QemuOptType::Bool).help("open drive file as read-only"),
    QemuOptDesc::new("rerror", QemuOptType::String).help("read error action"),
    QemuOptDesc::new("werror", QemuOptType::String).help("write error action"),
    QemuOptDesc::new("copy-on-read", QemuOptType::Bool)
        .help("copy read data from backing file into image file"),
];

/// `qemu_common_drive_opts` from blockdev.c, with `THROTTLE_OPTS` from
/// include/qemu/throttle-options.h spelled out.
const COMMON_DRIVE_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("snapshot", QemuOptType::Bool).help("enable/disable snapshot mode"),
    QemuOptDesc::new("aio", QemuOptType::String)
        .help("host AIO implementation (threads, native, io_uring)"),
    QemuOptDesc::new("cache.writeback", QemuOptType::Bool).help("Enable writeback mode"),
    QemuOptDesc::new("format", QemuOptType::String).help("disk format (raw, qcow2, ...)"),
    QemuOptDesc::new("rerror", QemuOptType::String).help("read error action"),
    QemuOptDesc::new("werror", QemuOptType::String).help("write error action"),
    QemuOptDesc::new("read-only", QemuOptType::Bool).help("open drive file as read-only"),
    QemuOptDesc::new("throttling.iops-total", QemuOptType::Number)
        .help("limit total I/O operations per second"),
    QemuOptDesc::new("throttling.iops-read", QemuOptType::Number)
        .help("limit read operations per second"),
    QemuOptDesc::new("throttling.iops-write", QemuOptType::Number)
        .help("limit write operations per second"),
    QemuOptDesc::new("throttling.bps-total", QemuOptType::Number)
        .help("limit total bytes per second"),
    QemuOptDesc::new("throttling.bps-read", QemuOptType::Number)
        .help("limit read bytes per second"),
    QemuOptDesc::new("throttling.bps-write", QemuOptType::Number)
        .help("limit write bytes per second"),
    QemuOptDesc::new("throttling.iops-total-max", QemuOptType::Number).help("I/O operations burst"),
    QemuOptDesc::new("throttling.iops-read-max", QemuOptType::Number)
        .help("I/O operations read burst"),
    QemuOptDesc::new("throttling.iops-write-max", QemuOptType::Number)
        .help("I/O operations write burst"),
    QemuOptDesc::new("throttling.bps-total-max", QemuOptType::Number).help("total bytes burst"),
    QemuOptDesc::new("throttling.bps-read-max", QemuOptType::Number).help("total bytes read burst"),
    QemuOptDesc::new("throttling.bps-write-max", QemuOptType::Number)
        .help("total bytes write burst"),
    QemuOptDesc::new("throttling.iops-total-max-length", QemuOptType::Number)
        .help("length of the iops-total-max burst period, in seconds"),
    QemuOptDesc::new("throttling.iops-read-max-length", QemuOptType::Number)
        .help("length of the iops-read-max burst period, in seconds"),
    QemuOptDesc::new("throttling.iops-write-max-length", QemuOptType::Number)
        .help("length of the iops-write-max burst period, in seconds"),
    QemuOptDesc::new("throttling.bps-total-max-length", QemuOptType::Number)
        .help("length of the bps-total-max burst period, in seconds"),
    QemuOptDesc::new("throttling.bps-read-max-length", QemuOptType::Number)
        .help("length of the bps-read-max burst period, in seconds"),
    QemuOptDesc::new("throttling.bps-write-max-length", QemuOptType::Number)
        .help("length of the bps-write-max burst period, in seconds"),
    QemuOptDesc::new("throttling.iops-size", QemuOptType::Number)
        .help("when limiting by iops max size of an I/O in bytes"),
    QemuOptDesc::new("throttling.group", QemuOptType::String)
        .help("name of the block throttling group"),
    QemuOptDesc::new("copy-on-read", QemuOptType::Bool)
        .help("copy read data from backing file into image file"),
    QemuOptDesc::new("detect-zeroes", QemuOptType::String)
        .help("try to optimize zero writes (off, on, unmap)"),
    QemuOptDesc::new("stats-account-invalid", QemuOptType::Bool)
        .help("whether to account for invalid I/O operations in the statistics"),
    QemuOptDesc::new("stats-account-failed", QemuOptType::Bool)
        .help("whether to account for failed I/O operations in the statistics"),
];

/// `bdrv_runtime_opts` from block.c.
const BDRV_RUNTIME_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("node-name", QemuOptType::String).help("Node name of the block device node"),
    QemuOptDesc::new("driver", QemuOptType::String).help("Block driver to use for the node"),
    QemuOptDesc::new("cache.direct", QemuOptType::Bool)
        .help("Bypass software writeback cache on the host"),
    QemuOptDesc::new("cache.no-flush", QemuOptType::Bool).help("Ignore flush requests"),
    QemuOptDesc::new("active", QemuOptType::Bool).help("Node is activated"),
    QemuOptDesc::new("read-only", QemuOptType::Bool).help("Node is opened in read-only mode"),
    QemuOptDesc::new("auto-read-only", QemuOptType::Bool)
        .help("Node can become read-only if opening read-write fails"),
    QemuOptDesc::new("detect-zeroes", QemuOptType::String)
        .help("try to optimize zero writes (off, on, unmap)"),
    QemuOptDesc::new("discard", QemuOptType::String)
        .help("discard operation (ignore/off, unmap/on)"),
    QemuOptDesc::new("force-share", QemuOptType::Bool)
        .help("always accept other writers (default: off)"),
];

/// The groups in the order `qemu_init()` registers them, the ones from `opts_init()` last.
/// `drive` stands for `drive_config_groups`.
fn config_groups() -> Vec<QemuOptsList> {
    let mut groups = vec![
        QemuOptsList::new("drive", &[]),
        ruvm_chardev::opts::chardev_opts(),
        QemuOptsList::new("device", &[]),
        QemuOptsList::new("netdev", &[]),
        crate::vl::mon_opts(),
        QemuOptsList::new("accel", &[]),
        crate::vl::memory_opts(),
        smp_opts(),
        QemuOptsList::new("object", &[]),
        crate::vl::name_opts(),
        crate::arm::semihosting_config_opts(),
    ];
    #[cfg(unix)]
    groups.push(crate::vl::run_with_opts());
    groups.push(ruvm_ui::vnc::opts_list());
    groups.push(QemuOptsList::new("smbios", &[]));
    groups
}

fn param_type(ty: QemuOptType) -> CommandLineParameterType {
    match ty {
        QemuOptType::String => CommandLineParameterType::String,
        QemuOptType::Bool => CommandLineParameterType::Boolean,
        QemuOptType::Number => CommandLineParameterType::Number,
        QemuOptType::Size => CommandLineParameterType::Size,
    }
}

/// `query_option_descs()`, which prepends, so the list is in reverse table order.
fn query_option_descs(desc: &[QemuOptDesc]) -> Vec<CommandLineParameterInfo> {
    desc.iter()
        .rev()
        .map(|d| CommandLineParameterInfo {
            name: d.name.to_string(),
            type_: param_type(d.ty),
            help: d.help.map(str::to_string),
            default: d.def_value_str.map(str::to_string),
        })
        .collect()
}

/// `get_drive_infolist()`: the drive tables one after another, then `cleanup_infolist()`.
///
/// `cleanup_infolist()` moves on after it drops an entry without looking at the one that took
/// its place, so a name twice in a row survives. QEMU lists `werror` twice that way, and so
/// does ruvm.
fn drive_infolist() -> Vec<CommandLineParameterInfo> {
    let mut list: Vec<_> = [LEGACY_DRIVE_OPTS, COMMON_DRIVE_OPTS, BDRV_RUNTIME_OPTS]
        .into_iter()
        .flat_map(query_option_descs)
        .collect();
    let mut cur = 0;
    while cur + 1 < list.len() {
        if list[..=cur].iter().any(|p| p.name == list[cur + 1].name) {
            list.remove(cur + 1);
        }
        cur += 1;
    }
    list
}

/// `query_all_machine_properties()`: the settable properties of every machine class, each
/// name once, then `type` in front.
fn machine_properties(registry: &Registry) -> Vec<CommandLineParameterInfo> {
    let mut params: Vec<CommandLineParameterInfo> = Vec::new();
    for klass in registry.class_get_list(Some("machine"), false) {
        for prop in klass.properties() {
            if !prop.is_writable() || params.iter().any(|p| p.name == prop.name()) {
                continue;
            }
            let type_ = match prop.type_name() {
                "bool" | "OnOffAuto" => CommandLineParameterType::Boolean,
                "int" => CommandLineParameterType::Number,
                "size" => CommandLineParameterType::Size,
                _ => CommandLineParameterType::String,
            };
            params.push(CommandLineParameterInfo {
                name: prop.name().to_string(),
                type_,
                help: prop.get_description(),
                default: None,
            });
        }
    }
    params.push(CommandLineParameterInfo {
        name: "type".into(),
        type_: CommandLineParameterType::String,
        help: Some("machine type".into()),
        default: None,
    });
    params.reverse();
    params
}

/// `qmp_query_command_line_options()`.
pub(crate) fn query_command_line_options(
    registry: &Registry,
    option: Option<&str>,
) -> ruvm_base::Result<Vec<CommandLineOptionInfo>> {
    let mut list = Vec::new();
    for group in config_groups() {
        let name = group.name().unwrap_or_default();
        if option.is_some_and(|o| o != name) {
            continue;
        }
        let parameters =
            if name == "drive" { drive_infolist() } else { query_option_descs(group.desc()) };
        list.push(CommandLineOptionInfo { option: name.to_string(), parameters });
    }
    if option.is_none_or(|o| o == "machine") {
        list.push(CommandLineOptionInfo {
            option: "machine".into(),
            parameters: machine_properties(registry),
        });
    }
    if let (true, Some(o)) = (list.is_empty(), option) {
        return Err(ruvm_base::Error::generic(format!("invalid option name: {o}")));
    }
    list.reverse();
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drive_parameters_match_qemu() {
        // QEMU 11.1's reply, `werror` twice included.
        let names: Vec<String> = drive_infolist().into_iter().map(|p| p.name).collect();
        assert_eq!(names.len(), 46);
        assert_eq!(names[..4], ["copy-on-read", "werror", "rerror", "read-only"]);
        assert_eq!(names[32..35], ["throttling.iops-total", "werror", "format"]);
        assert_eq!(names[45], "node-name");
    }
}
