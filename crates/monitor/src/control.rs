// SPDX-License-Identifier: GPL-2.0-or-later

//! The commands the monitor implements itself, monitor/qmp-cmds-control.c.

use ruvm_base::Result;
use ruvm_qapi::commands::{
    register_qmp_capabilities, register_query_commands, register_query_qmp_schema,
    register_query_version,
};
use ruvm_qapi::types::{
    CommandInfo, QmpCapabilitiesArg, SchemaInfo, SchemaInfoU, VersionInfo, VersionTriple,
};
use ruvm_qapi::visit::{
    CompatPolicyOutput, QObjectInputVisitor, QObjectOutputVisitor, Visit, VisitorExt,
};
use ruvm_qapi::{QValue, qmp_schema};

use crate::qmp::{Commands, MonitorQmp};

/// The QEMU release ruvm answers `query-version` with. libvirt gates features on it.
pub const QEMU_VERSION: (i64, i64, i64) = (11, 1, 0);

/// `qmp_query_version()`. The package is where QEMU builds put the distribution's version,
/// and where ruvm puts its own.
pub fn version_info() -> VersionInfo {
    let (major, minor, micro) = QEMU_VERSION;
    VersionInfo {
        qemu: VersionTriple { major, minor, micro },
        package: format!("ruvm {}", env!("CARGO_PKG_VERSION")),
    }
}

pub(crate) fn version_value() -> QValue {
    let mut v = QObjectOutputVisitor::new();
    let mut info = version_info();
    VersionInfo::visit(&mut v, None, &mut info).expect("output cannot fail");
    v.complete()
}

fn qmp_capabilities(mon: &MonitorQmp, arg: QmpCapabilitiesArg) -> Result<()> {
    let enable: Vec<&str> = arg.enable.iter().flatten().map(|c| c.as_str()).collect();
    mon.accept_capabilities(&enable)
}

/// `qmp_query_commands()`. QEMU prepends each name to the list, so the reply lists the
/// commands in the reverse of the order they were registered.
fn query_commands(mon: &MonitorQmp) -> Result<Vec<CommandInfo>> {
    let Some(qmp) = mon.qmp() else { return Ok(Vec::new()) };
    let cmds = qmp.commands();
    let mut list: Vec<CommandInfo> =
        cmds.iter().filter(|c| c.enabled).map(|c| CommandInfo { name: c.name.clone() }).collect();
    list.reverse();
    Ok(list)
}

fn is_deprecated(features: &Option<Vec<String>>) -> bool {
    features.iter().flatten().any(|f| f == "deprecated")
}

/// `zap_deprecated()`: what `-compat deprecated-output=hide` leaves of the schema.
fn zap_deprecated(schema: &mut Vec<SchemaInfo>) {
    schema.retain(|e| !is_deprecated(&e.features));
    for ent in schema {
        if let SchemaInfoU::Object(obj) = &mut ent.u {
            obj.members.retain(|m| !is_deprecated(&m.features));
        }
    }
}

/// `qmp_query_qmp_schema()`. Like QEMU it goes through the generated types, so the reply is
/// built by the same visitor every other reply is.
fn query_qmp_schema(mon: &MonitorQmp) -> Result<Vec<SchemaInfo>> {
    let mut v = QObjectInputVisitor::new(qmp_schema());
    let mut schema = Vec::new();
    v.visit_list(None, &mut schema, |v, e| SchemaInfo::visit(v, None, e))?;
    let hide = mon.qmp().is_some_and(|q| q.policy().deprecated_output == CompatPolicyOutput::Hide);
    if hide {
        zap_deprecated(&mut schema);
    }
    Ok(schema)
}

/// The commands of `qmp_commands` that the monitor provides.
pub(crate) fn register(cmds: &mut Commands) {
    register_qmp_capabilities(cmds, qmp_capabilities);
    register_query_version(cmds, |_: &MonitorQmp| Ok(version_info()));
    register_query_commands(cmds, query_commands);
    register_query_qmp_schema(cmds, query_qmp_schema);
}

/// `qmp_cap_negotiation_commands`, which holds only `qmp_capabilities`.
pub(crate) fn register_negotiation(cmds: &mut Commands) {
    register_qmp_capabilities(cmds, qmp_capabilities);
}
