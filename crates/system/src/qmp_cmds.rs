// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP commands the system emulator provides: the run state from monitor/qmp-cmds.c, the
//! chardev commands from chardev/char.c, the QOM commands from qom/qom-qmp-cmds.c and
//! `x-exit-preconfig` from system/vl.c.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_chardev::BACKENDS;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::*;
use ruvm_qapi::types::{
    ChardevBackendInfo, ChardevReturn, MachineInfo, NameInfo, ObjectOptions,
    ObjectPropertiesValues, ObjectPropertyInfo, ObjectTypeInfo, ShutdownCause,
};
use ruvm_qapi::visit::{QObjectInputVisitor, QObjectOutputVisitor, Visit, VisitorExt};
use ruvm_qapi::{QDict, QValue};
use ruvm_qom::qmp as qom;

use crate::vl::Vm;

/// Reads a reply the QOM code built as a [`QValue`] back into the generated type, so the
/// marshaller can write it out again the way QEMU's does.
fn list_of<T: Visit>(value: QValue) -> Result<Vec<T>> {
    let mut v = QObjectInputVisitor::new(value);
    let mut out = Vec::new();
    v.visit_list(None, &mut out, |v, e| T::visit(v, None, e))?;
    Ok(out)
}

/// `user_creatable_add_qapi()`: the options as the dictionary the QOM code takes.
pub(crate) fn object_options_dict(opts: &mut ObjectOptions) -> Result<QDict> {
    let mut ov = QObjectOutputVisitor::new();
    ObjectOptions::visit(&mut ov, None, opts)?;
    match ov.complete() {
        QValue::Dict(d) => Ok(d),
        _ => Err(Error::generic("ObjectOptions did not visit as a dictionary")),
    }
}

/// Registers every command in this module with `vm`'s dispatcher.
pub(crate) fn register(vm: &Arc<Vm>, cmds: &mut Commands) {
    let v = vm.clone();
    register_query_status(cmds, move |_: &MonitorQmp| Ok(v.runstate.status()));
    let v = vm.clone();
    register_stop(cmds, move |_: &MonitorQmp| v.runstate.qmp_stop());
    let v = vm.clone();
    register_cont(cmds, move |_: &MonitorQmp| v.runstate.qmp_cont());
    let v = vm.clone();
    register_quit(cmds, move |_: &MonitorQmp| {
        v.runstate.shutdown_request(ShutdownCause::HostQmpQuit);
        Ok(())
    });
    let v = vm.clone();
    register_query_name(cmds, move |_: &MonitorQmp| Ok(NameInfo { name: v.name.clone() }));
    let v = vm.clone();
    register_x_exit_preconfig(cmds, move |_: &MonitorQmp| v.exit_preconfig());
    // Machine "none" has no CPUs.
    register_query_cpus_fast(cmds, |_: &MonitorQmp| Ok(Vec::new()));
    // `qmp_query_machines()` over the one machine ruvm has, with the values QEMU gives `none`.
    register_query_machines(cmds, |_: &MonitorQmp, arg| {
        Ok(vec![MachineInfo {
            name: "none".into(),
            alias: None,
            is_default: None,
            cpu_max: 1,
            hotpluggable_cpus: false,
            numa_mem_supported: false,
            deprecated: false,
            default_cpu_type: None,
            default_ram_id: Some("ram".into()),
            acpi: false,
            compat_props: arg.compat_props.unwrap_or(false).then(Vec::new),
        }])
    });
    register_human_monitor_command(cmds, |_: &MonitorQmp, _| {
        Err(Error::generic("ruvm has no human monitor yet"))
    });

    let v = vm.clone();
    register_chardev_add(cmds, move |_: &MonitorQmp, arg| {
        v.chardevs.add(&arg.id, &arg.backend)?;
        Ok(ChardevReturn::default())
    });
    let v = vm.clone();
    register_chardev_remove(cmds, move |_: &MonitorQmp, arg| v.chardevs.remove(&arg.id));
    let v = vm.clone();
    register_query_chardev(cmds, move |_: &MonitorQmp| Ok(v.chardevs.query()));
    register_query_chardev_backends(cmds, |_: &MonitorQmp| {
        // QEMU prepends each backend to the list as it walks the types.
        Ok(BACKENDS
            .iter()
            .rev()
            .map(|name| ChardevBackendInfo { name: name.to_string() })
            .collect())
    });

    let v = vm.clone();
    register_object_add(cmds, move |_: &MonitorQmp, mut opts| {
        let dict = object_options_dict(&mut opts)?;
        qom::object_add(&v.registry, &dict)
    });
    let v = vm.clone();
    register_object_del(cmds, move |_: &MonitorQmp, arg| qom::object_del(&v.registry, &arg.id));
    let v = vm.clone();
    register_qom_list(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertyInfo>(qom::qom_list(&v.registry, &arg.path)?)
    });
    let v = vm.clone();
    register_qom_list_get(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertiesValues>(qom::qom_list_get(&v.registry, &arg.paths)?)
    });
    let v = vm.clone();
    register_qom_get(cmds, move |_: &MonitorQmp, arg| {
        qom::qom_get(&v.registry, &arg.path, &arg.property)
    });
    let v = vm.clone();
    register_qom_set(cmds, move |_: &MonitorQmp, arg| {
        qom::qom_set(&v.registry, &arg.path, &arg.property, arg.value)
    });
    let v = vm.clone();
    register_qom_list_types(cmds, move |_: &MonitorQmp, arg| {
        let abstract_ = arg.abstract_.unwrap_or(false);
        let types = qom::qom_list_types(&v.registry, arg.implements.as_deref(), abstract_);
        list_of::<ObjectTypeInfo>(types)
    });
    let v = vm.clone();
    register_qom_list_properties(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertyInfo>(qom::qom_list_properties(&v.registry, &arg.typename)?)
    });
    let v = vm.clone();
    register_device_list_properties(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertyInfo>(qom::device_list_properties(&v.registry, &arg.typename)?)
    });
}
