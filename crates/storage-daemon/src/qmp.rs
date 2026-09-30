// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP commands of the storage daemon: the block, export, NBD server, job, dirty bitmap,
//! chardev, QOM and `quit` commands of storage-daemon/qapi/qapi-schema.json, and the events
//! the block layer raises.
//!
//! Differences from QEMU:
//!
//! - Only the commands listed in [`register`] exist. The storage daemon's schema has more,
//!   such as `blockdev-reopen`, `transaction`, `blockdev-snapshot`, `block-latency-histogram-set`
//!   and the crypto and authz object commands; the ruvm block layer does not have them yet.
//! - `block-export-add` and `query-block-exports` are registered by hand, as they take the
//!   Linux only export types the generated schema leaves out.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_block::BlockEvent;
use ruvm_block::BlockGraph;
use ruvm_chardev::{BACKENDS, Chardevs};
use ruvm_monitor::{Commands, MonitorQmp, Qmp};
use ruvm_qapi::commands::*;
use ruvm_qapi::dispatch::{QmpCommandFunc, QmpCommandOptions};
use ruvm_qapi::events::*;
use ruvm_qapi::types::*;
use ruvm_qapi::visit::{
    CompatPolicy, QObjectInputVisitor, QObjectOutputVisitor, Visit, Visitor, VisitorExt,
};
use ruvm_qapi::{QDict, QValue};
use ruvm_qom::Registry;
use ruvm_qom::qmp as qom;

use crate::export::{self, ExportOptions, ExportType, Exports};

/// What the commands work on.
pub struct State {
    pub graph: Arc<BlockGraph>,
    pub exports: Arc<Exports>,
    pub qmp: Arc<Qmp>,
    pub registry: Registry,
    pub chardevs: Arc<Chardevs>,
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("State").finish_non_exhaustive()
    }
}

impl State {
    /// Sets up the export list, sending `BLOCK_EXPORT_DELETED` through `qmp`.
    pub fn new(qmp: Arc<Qmp>, registry: Registry, chardevs: Arc<Chardevs>) -> Arc<State> {
        let q = qmp.clone();
        let exports = Exports::new(Box::new(move |id: &str| {
            let arg = BlockExportDeletedArg { id: id.to_string() };
            if let Some(ev) = event_block_export_deleted(&q.policy(), arg) {
                q.emit_event(ev);
            }
        }));
        Arc::new(State {
            graph: Arc::new(BlockGraph::new()),
            exports: Arc::new(exports),
            qmp,
            registry,
            chardevs,
        })
    }

    /// `qmp_block_export_add()`.
    pub fn block_export_add(&self, opts: &ExportOptions) -> Result<()> {
        self.exports.add(&self.graph, opts)
    }

    /// `nbd_server_start_options()`.
    pub fn nbd_server_start(&self, opts: &NbdServerOptions) -> Result<()> {
        ruvm_block::nbd::nbd_server_start(opts)
    }

    /// `qmp_nbd_server_add()`.
    pub fn nbd_server_add(&self, arg: &NbdServerAddOptions) -> Result<()> {
        let dev = &arg.device;
        let node = match self.graph.backend(dev) {
            Some(blk) => blk.node_name(),
            None => self.graph.node(dev).map(|n| n.node_name),
        };
        if node.is_none() {
            return Err(Error::generic(format!(
                "Cannot find device='{dev}' nor node-name='{dev}'"
            )));
        }
        let name = arg.name.as_deref().unwrap_or(dev);
        if self.exports.contains(name) {
            return Err(Error::generic(format!("Block export id '{name}' is already in use")));
        }
        let (id, node_name, driver) = export::nbd::server_add(&self.graph, arg)?;
        self.exports.insert(&id, ExportType::Nbd, &node_name, driver);
        Ok(())
    }

    /// `qmp_nbd_server_remove()`.
    pub fn nbd_server_remove(&self, name: &str, mode: Option<BlockExportRemoveMode>) -> Result<()> {
        if self.exports.type_of(name).is_some_and(|t| t != ExportType::Nbd) {
            return Err(Error::generic(format!("Block export '{name}' is not an NBD export")));
        }
        self.exports.del(name, mode)
    }

    /// `qmp_nbd_server_stop()`.
    pub fn nbd_server_stop(&self) -> Result<()> {
        if !ruvm_block::nbd::nbd_server_is_running() {
            return Err(Error::generic("NBD server not running"));
        }
        self.exports.close_all(Some(ExportType::Nbd));
        ruvm_block::nbd::nbd_server_stop()
    }

    /// The end of `main()`: `blk_exp_close_all()`, `job_cancel_sync_all()` and
    /// `bdrv_close_all()`.
    pub fn cleanup(&self) {
        self.exports.close_all(None);
        if ruvm_block::nbd::nbd_server_is_running() {
            let _ = ruvm_block::nbd::nbd_server_stop();
        }
        for job in self.graph.query_jobs() {
            let _ = self.graph.job_cancel(&job.id);
        }
        self.registry.user_creatable_cleanup();
    }
}

/// `socket_address_flatten()`.
pub fn socket_address_flatten(addr: SocketAddressLegacy) -> SocketAddress {
    let u = match addr.u {
        SocketAddressLegacyU::Inet(w) => SocketAddressU::Inet(w.data),
        SocketAddressLegacyU::Unix(w) => SocketAddressU::Unix(w.data),
        SocketAddressLegacyU::Vsock(w) => SocketAddressU::Vsock(w.data),
        SocketAddressLegacyU::Fd(w) => SocketAddressU::Fd(w.data),
    };
    SocketAddress { u }
}

/// `user_creatable_add_qapi()`: the options as the dictionary the QOM code takes.
fn object_options_dict(opts: &mut ObjectOptions) -> Result<QDict> {
    let mut ov = QObjectOutputVisitor::new();
    ObjectOptions::visit(&mut ov, None, opts)?;
    match ov.complete() {
        QValue::Dict(d) => Ok(d),
        _ => Err(Error::generic("ObjectOptions did not visit as a dictionary")),
    }
}

/// Reads a reply the QOM code built as a [`QValue`] back into the generated type.
fn list_of<T: Visit>(value: QValue) -> Result<Vec<T>> {
    let mut v = QObjectInputVisitor::new(value);
    let mut out = Vec::new();
    v.visit_list(None, &mut out, |v, e| T::visit(v, None, e))?;
    Ok(out)
}

/// Sends the QMP event for a block layer event.
pub fn forward_block_event(qmp: &Qmp, ev: &BlockEvent) {
    let p = qmp.policy();
    let dict = match ev.clone() {
        BlockEvent::WriteThreshold { node_name, amount_exceeded, write_threshold } => {
            event_block_write_threshold(
                &p,
                BlockWriteThresholdArg { node_name, amount_exceeded, write_threshold },
            )
        }
        BlockEvent::DeviceTrayMoved { .. } => None,
        BlockEvent::JobStatusChange { id, status } => {
            event_job_status_change(&p, JobStatusChangeArg { id, status })
        }
        BlockEvent::BlockJobCompleted { job_type, device, len, offset, speed, error } => {
            event_block_job_completed(
                &p,
                BlockJobCompletedArg {
                    type_: job_type,
                    device,
                    len: len as i64,
                    offset: offset as i64,
                    speed: speed as i64,
                    error,
                },
            )
        }
        BlockEvent::BlockJobCancelled { job_type, device, len, offset, speed } => {
            event_block_job_cancelled(
                &p,
                BlockJobCancelledArg {
                    type_: job_type,
                    device,
                    len: len as i64,
                    offset: offset as i64,
                    speed: speed as i64,
                },
            )
        }
        BlockEvent::BlockJobReady { job_type, device, len, offset, speed } => {
            event_block_job_ready(
                &p,
                BlockJobReadyArg {
                    type_: job_type,
                    device,
                    len: len as i64,
                    offset: offset as i64,
                    speed: speed as i64,
                },
            )
        }
        BlockEvent::BlockJobPending { job_type, id } => {
            event_block_job_pending(&p, BlockJobPendingArg { type_: job_type, id })
        }
        BlockEvent::BlockJobError { device, operation, action } => {
            event_block_job_error(&p, BlockJobErrorArg { device, operation, action })
        }
    };
    if let Some(d) = dict {
        qmp.emit_event(d);
    }
}

/// Parses the arguments of `block-export-add` the way its marshaller does.
fn export_options_from_qmp(args: QDict, policy: CompatPolicy) -> Result<ExportOptions> {
    let mut iv = QObjectInputVisitor::new_qmp(QValue::Dict(args), policy);
    let v: &mut dyn Visitor = &mut iv;
    let mut arg = ExportOptions::default();
    v.start_struct(None)?;
    let r = ExportOptions::visit_members(v, &mut arg).and_then(|()| v.check_struct());
    v.end_struct();
    r.map(|()| arg)
}

/// Checks that a command without arguments got none.
fn no_args(args: QDict, policy: CompatPolicy) -> Result<()> {
    let mut iv = QObjectInputVisitor::new_qmp(QValue::Dict(args), policy);
    let v: &mut dyn Visitor = &mut iv;
    v.start_struct(None)?;
    let r = v.check_struct();
    v.end_struct();
    r
}

/// Registers every command with `cmds`. `quit` calls `on_quit`.
pub fn register(st: &Arc<State>, cmds: &mut Commands, on_quit: Arc<dyn Fn() + Send + Sync>) {
    register_quit(cmds, move |_: &MonitorQmp| {
        on_quit();
        Ok(())
    });

    // Block devices.
    let s = st.clone();
    register_blockdev_add(cmds, move |_: &MonitorQmp, opts| s.graph.blockdev_add(opts));
    let s = st.clone();
    register_blockdev_del(cmds, move |_: &MonitorQmp, arg| s.graph.blockdev_del(&arg.node_name));
    let s = st.clone();
    register_blockdev_create(cmds, move |_: &MonitorQmp, arg| {
        s.graph.blockdev_create_job(&arg.job_id, arg.options)
    });
    let s = st.clone();
    register_query_named_block_nodes(cmds, move |_: &MonitorQmp, arg| {
        s.graph.query_named_block_nodes(arg.flat)
    });
    let s = st.clone();
    register_block_set_write_threshold(cmds, move |_: &MonitorQmp, arg| {
        s.graph.block_set_write_threshold(&arg.node_name, arg.write_threshold)
    });

    // Exports.
    let s = st.clone();
    let add: QmpCommandFunc<MonitorQmp> = Arc::new(move |_, args, policy| {
        let opts = export_options_from_qmp(args, *policy)?;
        s.block_export_add(&opts)?;
        Ok(None)
    });
    cmds.register("block-export-add", add, QmpCommandOptions::ALLOW_PRECONFIG, 0);
    let s = st.clone();
    register_block_export_del(cmds, move |_: &MonitorQmp, arg| s.exports.del(&arg.id, arg.mode));
    let s = st.clone();
    let query: QmpCommandFunc<MonitorQmp> = Arc::new(move |_, args, policy| {
        no_args(args, *policy)?;
        let list = s.exports.query().iter().map(|e| QValue::Dict(e.to_qdict())).collect();
        Ok(Some(QValue::List(list)))
    });
    cmds.register("query-block-exports", query, QmpCommandOptions::ALLOW_PRECONFIG, 0);

    // The NBD server.
    let s = st.clone();
    register_nbd_server_start(cmds, move |_: &MonitorQmp, arg| {
        let opts = NbdServerOptions {
            handshake_max_seconds: arg.handshake_max_seconds,
            tls_creds: arg.tls_creds,
            tls_authz: arg.tls_authz,
            max_connections: arg.max_connections,
            addr: socket_address_flatten(arg.addr),
        };
        s.nbd_server_start(&opts)
    });
    let s = st.clone();
    register_nbd_server_add(cmds, move |_: &MonitorQmp, arg| s.nbd_server_add(&arg));
    let s = st.clone();
    register_nbd_server_remove(cmds, move |_: &MonitorQmp, arg| {
        s.nbd_server_remove(&arg.name, arg.mode)
    });
    let s = st.clone();
    register_nbd_server_stop(cmds, move |_: &MonitorQmp| s.nbd_server_stop());

    register_jobs(st, cmds);
    register_bitmaps(st, cmds);
    register_objects(st, cmds);
}

fn register_jobs(st: &Arc<State>, cmds: &mut Commands) {
    let s = st.clone();
    register_job_cancel(cmds, move |_: &MonitorQmp, a| s.graph.job_cancel(&a.id));
    let s = st.clone();
    register_job_pause(cmds, move |_: &MonitorQmp, a| s.graph.job_pause(&a.id));
    let s = st.clone();
    register_job_resume(cmds, move |_: &MonitorQmp, a| s.graph.job_resume(&a.id));
    let s = st.clone();
    register_job_complete(cmds, move |_: &MonitorQmp, a| s.graph.job_complete(&a.id));
    let s = st.clone();
    register_job_finalize(cmds, move |_: &MonitorQmp, a| s.graph.job_finalize(&a.id));
    let s = st.clone();
    register_job_dismiss(cmds, move |_: &MonitorQmp, a| s.graph.job_dismiss(&a.id));
    let s = st.clone();
    register_query_jobs(cmds, move |_: &MonitorQmp| Ok(s.graph.query_jobs()));

    let s = st.clone();
    register_block_job_set_speed(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_job_set_speed(&a.device, a.speed)
    });
    let s = st.clone();
    register_block_job_cancel(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_job_cancel(&a.device, a.force)
    });
    let s = st.clone();
    register_block_job_pause(cmds, move |_: &MonitorQmp, a| s.graph.block_job_pause(&a.device));
    let s = st.clone();
    register_block_job_resume(cmds, move |_: &MonitorQmp, a| s.graph.block_job_resume(&a.device));
    let s = st.clone();
    register_block_job_complete(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_job_complete(&a.device)
    });
    let s = st.clone();
    register_block_job_finalize(cmds, move |_: &MonitorQmp, a| s.graph.block_job_finalize(&a.id));
    let s = st.clone();
    register_block_job_dismiss(cmds, move |_: &MonitorQmp, a| s.graph.block_job_dismiss(&a.id));
    let s = st.clone();
    register_block_job_change(cmds, move |_: &MonitorQmp, a| s.graph.block_job_change(&a));
    let s = st.clone();
    register_query_block_jobs(cmds, move |_: &MonitorQmp| s.graph.query_block_jobs());

    let s = st.clone();
    register_blockdev_backup(cmds, move |_: &MonitorQmp, a| s.graph.blockdev_backup(&a));
    let s = st.clone();
    register_blockdev_mirror(cmds, move |_: &MonitorQmp, a| s.graph.blockdev_mirror(&a));
    let s = st.clone();
    register_block_stream(cmds, move |_: &MonitorQmp, a| s.graph.block_stream(&a));
    let s = st.clone();
    register_block_commit(cmds, move |_: &MonitorQmp, a| s.graph.block_commit(&a));
}

fn register_bitmaps(st: &Arc<State>, cmds: &mut Commands) {
    let s = st.clone();
    register_block_dirty_bitmap_add(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_add(&a)
    });
    let s = st.clone();
    register_block_dirty_bitmap_remove(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_remove(&a)
    });
    let s = st.clone();
    register_block_dirty_bitmap_clear(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_clear(&a)
    });
    let s = st.clone();
    register_block_dirty_bitmap_enable(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_enable(&a)
    });
    let s = st.clone();
    register_block_dirty_bitmap_disable(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_disable(&a)
    });
    let s = st.clone();
    register_block_dirty_bitmap_merge(cmds, move |_: &MonitorQmp, a| {
        s.graph.block_dirty_bitmap_merge(&a)
    });
}

fn register_objects(st: &Arc<State>, cmds: &mut Commands) {
    let s = st.clone();
    register_chardev_add(cmds, move |_: &MonitorQmp, arg| {
        s.chardevs.add(&arg.id, &arg.backend)?;
        Ok(ChardevReturn::default())
    });
    let s = st.clone();
    register_chardev_remove(cmds, move |_: &MonitorQmp, arg| s.chardevs.remove(&arg.id));
    let s = st.clone();
    register_query_chardev(cmds, move |_: &MonitorQmp| Ok(s.chardevs.query()));
    register_query_chardev_backends(cmds, |_: &MonitorQmp| {
        Ok(BACKENDS
            .iter()
            .rev()
            .map(|name| ChardevBackendInfo { name: name.to_string() })
            .collect())
    });

    let s = st.clone();
    register_object_add(cmds, move |_: &MonitorQmp, mut opts| {
        let dict = object_options_dict(&mut opts)?;
        qom::object_add(&s.registry, &dict)
    });
    let s = st.clone();
    register_object_del(cmds, move |_: &MonitorQmp, arg| qom::object_del(&s.registry, &arg.id));
    let s = st.clone();
    register_qom_list(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertyInfo>(qom::qom_list(&s.registry, &arg.path)?)
    });
    let s = st.clone();
    register_qom_get(cmds, move |_: &MonitorQmp, arg| {
        qom::qom_get(&s.registry, &arg.path, &arg.property)
    });
    let s = st.clone();
    register_qom_set(cmds, move |_: &MonitorQmp, arg| {
        qom::qom_set(&s.registry, &arg.path, &arg.property, arg.value)
    });
    let s = st.clone();
    register_qom_list_types(cmds, move |_: &MonitorQmp, arg| {
        let abstract_ = arg.abstract_.unwrap_or(false);
        let types = qom::qom_list_types(&s.registry, arg.implements.as_deref(), abstract_);
        list_of::<ObjectTypeInfo>(types)
    });
    let s = st.clone();
    register_qom_list_properties(cmds, move |_: &MonitorQmp, arg| {
        list_of::<ObjectPropertyInfo>(qom::qom_list_properties(&s.registry, &arg.typename)?)
    });
}
