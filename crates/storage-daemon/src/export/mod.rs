// SPDX-License-Identifier: GPL-2.0-or-later

//! The block export layer of block/export/export.c: `blk_exp_add()`, `block-export-del`,
//! `query-block-exports` and closing every export at exit, over the NBD, vhost-user-blk, FUSE
//! and VDUSE drivers.
//!
//! Differences from QEMU:
//!
//! - ruvm's QAPI schema is generated with `CONFIG_FUSE`, `CONFIG_VHOST_USER_BLK_SERVER` and
//!   `CONFIG_VDUSE_BLK_EXPORT` off, as the system emulator is configured. The storage daemon
//!   parses `BlockExportOptions` with its own visitor, [`ExportOptions`], so on Linux it takes
//!   the `vhost-user-blk`, `fuse` and `vduse-blk` types a Linux build of QEMU takes. On other
//!   hosts it takes only `nbd`, and the others fail as QEMU's macOS build fails them:
//!   `Parameter 'type' does not accept value 'fuse'`. `query-qmp-schema` does not list the
//!   Linux only types.
//! - Deleting an export waits for its clients to go away before the command returns, and
//!   `BLOCK_EXPORT_DELETED` goes out before the reply. QEMU replies first and sends the event
//!   from a bottom half once the last reference is gone.
//! - ruvm has no `iothread` objects, so an export naming one fails with QEMU's
//!   `iothread "ID" not found`.

use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_block::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockBackend, BlockGraph,
};
use ruvm_qapi::types::{
    BlockExportIothreads, BlockExportOptions, BlockExportOptionsNbd, BlockExportOptionsU,
    BlockExportOptionsVduseBlk, BlockExportOptionsVhostUserBlk, BlockExportRemoveMode,
    FuseExportAllowOther,
};
use ruvm_qapi::visit::{QEnumLookup, Visit, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue};

pub mod nbd;

#[cfg(target_os = "linux")]
pub mod fuse;
#[cfg(target_os = "linux")]
pub mod vduse_blk;
#[cfg(unix)]
pub mod vhost_user_blk;
#[cfg(unix)]
pub mod virtio_blk;

/// `BlockExportType`, with the values of a Linux build of QEMU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExportType {
    #[default]
    Nbd,
    VhostUserBlk,
    Fuse,
    VduseBlk,
}

impl ExportType {
    /// The types this host takes, in the schema's order.
    #[cfg(target_os = "linux")]
    const ALL: &'static [ExportType] =
        &[ExportType::Nbd, ExportType::VhostUserBlk, ExportType::Fuse, ExportType::VduseBlk];
    #[cfg(target_os = "linux")]
    const LOOKUP: QEnumLookup = QEnumLookup::new(&["nbd", "vhost-user-blk", "fuse", "vduse-blk"]);
    #[cfg(not(target_os = "linux"))]
    const ALL: &'static [ExportType] = &[ExportType::Nbd];
    #[cfg(not(target_os = "linux"))]
    const LOOKUP: QEnumLookup = QEnumLookup::new(&["nbd"]);

    /// The name QMP uses.
    pub fn as_str(self) -> &'static str {
        match self {
            ExportType::Nbd => "nbd",
            ExportType::VhostUserBlk => "vhost-user-blk",
            ExportType::Fuse => "fuse",
            ExportType::VduseBlk => "vduse-blk",
        }
    }

    fn visit(v: &mut dyn Visitor, name: Option<&str>, obj: &mut Self) -> Result<()> {
        let mut value = Self::ALL.iter().position(|t| t == obj).unwrap_or(0);
        v.type_enum(name, &mut value, &Self::LOOKUP)?;
        *obj = Self::ALL[value];
        Ok(())
    }
}

/// `BlockExportOptionsFuse`, which ruvm's schema leaves out.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BlockExportOptionsFuse {
    /// `mountpoint`
    pub mountpoint: String,
    /// `growable`
    pub growable: Option<bool>,
    /// `allow-other`
    pub allow_other: Option<FuseExportAllowOther>,
}

impl BlockExportOptionsFuse {
    fn visit_members(v: &mut dyn Visitor, obj: &mut Self) -> Result<()> {
        v.type_str(Some("mountpoint"), &mut obj.mountpoint)?;
        if v.optional(Some("growable"), obj.growable.is_some()) {
            let p = obj.growable.get_or_insert_with(Default::default);
            v.type_bool(Some("growable"), p)?;
        }
        if v.optional(Some("allow-other"), obj.allow_other.is_some()) {
            let p = obj.allow_other.get_or_insert_with(Default::default);
            FuseExportAllowOther::visit(v, Some("allow-other"), p)?;
        }
        Ok(())
    }
}

/// The type specific part of [`ExportOptions`].
#[derive(Clone, Debug, PartialEq)]
pub enum ExportKind {
    Nbd(BlockExportOptionsNbd),
    VhostUserBlk(BlockExportOptionsVhostUserBlk),
    Fuse(BlockExportOptionsFuse),
    VduseBlk(BlockExportOptionsVduseBlk),
}

impl ExportKind {
    pub fn tag(&self) -> ExportType {
        match self {
            ExportKind::Nbd(_) => ExportType::Nbd,
            ExportKind::VhostUserBlk(_) => ExportType::VhostUserBlk,
            ExportKind::Fuse(_) => ExportType::Fuse,
            ExportKind::VduseBlk(_) => ExportType::VduseBlk,
        }
    }

    fn for_tag(tag: ExportType) -> Self {
        match tag {
            ExportType::Nbd => ExportKind::Nbd(Default::default()),
            ExportType::VhostUserBlk => ExportKind::VhostUserBlk(Default::default()),
            ExportType::Fuse => ExportKind::Fuse(Default::default()),
            ExportType::VduseBlk => ExportKind::VduseBlk(Default::default()),
        }
    }
}

impl Default for ExportKind {
    fn default() -> Self {
        ExportKind::Nbd(Default::default())
    }
}

/// `BlockExportOptions` with every export type of a Linux build.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExportOptions {
    pub id: String,
    pub fixed_iothread: Option<bool>,
    pub iothread: Option<BlockExportIothreads>,
    pub node_name: String,
    pub writable: Option<bool>,
    pub writethrough: Option<bool>,
    pub allow_inactive: Option<bool>,
    pub kind: ExportKind,
}

impl ExportOptions {
    /// `visit_type_BlockExportOptions_members()`.
    pub fn visit_members(v: &mut dyn Visitor, obj: &mut Self) -> Result<()> {
        let mut tag = obj.kind.tag();
        ExportType::visit(v, Some("type"), &mut tag)?;
        if tag != obj.kind.tag() {
            obj.kind = ExportKind::for_tag(tag);
        }
        v.type_str(Some("id"), &mut obj.id)?;
        if v.optional(Some("fixed-iothread"), obj.fixed_iothread.is_some()) {
            let p = obj.fixed_iothread.get_or_insert_with(Default::default);
            v.type_bool(Some("fixed-iothread"), p)?;
        }
        if v.optional(Some("iothread"), obj.iothread.is_some()) {
            let p = obj.iothread.get_or_insert_with(Default::default);
            BlockExportIothreads::visit(v, Some("iothread"), p)?;
        }
        v.type_str(Some("node-name"), &mut obj.node_name)?;
        if v.optional(Some("writable"), obj.writable.is_some()) {
            let p = obj.writable.get_or_insert_with(Default::default);
            v.type_bool(Some("writable"), p)?;
        }
        if v.optional(Some("writethrough"), obj.writethrough.is_some()) {
            let p = obj.writethrough.get_or_insert_with(Default::default);
            v.type_bool(Some("writethrough"), p)?;
        }
        if v.optional(Some("allow-inactive"), obj.allow_inactive.is_some()) {
            let p = obj.allow_inactive.get_or_insert_with(Default::default);
            v.type_bool(Some("allow-inactive"), p)?;
        }
        match &mut obj.kind {
            ExportKind::Nbd(u) => BlockExportOptionsNbd::visit_members(v, u),
            ExportKind::VhostUserBlk(u) => BlockExportOptionsVhostUserBlk::visit_members(v, u),
            ExportKind::Fuse(u) => BlockExportOptionsFuse::visit_members(v, u),
            ExportKind::VduseBlk(u) => BlockExportOptionsVduseBlk::visit_members(v, u),
        }
    }

    /// `visit_type_BlockExportOptions()` on the whole value `v` visits.
    pub fn visit(v: &mut dyn Visitor) -> Result<Self> {
        let mut obj = ExportOptions::default();
        v.start_struct(None)?;
        let r = Self::visit_members(v, &mut obj).and_then(|()| v.check_struct());
        v.end_struct();
        r.map(|()| obj)
    }

    /// The options as ruvm's QAPI type, for an NBD export.
    pub fn to_nbd(&self) -> Option<BlockExportOptions> {
        let ExportKind::Nbd(nbd) = &self.kind else { return None };
        Some(BlockExportOptions {
            id: self.id.clone(),
            fixed_iothread: self.fixed_iothread,
            iothread: self.iothread.clone(),
            node_name: self.node_name.clone(),
            writable: self.writable,
            writethrough: self.writethrough,
            allow_inactive: self.allow_inactive,
            u: BlockExportOptionsU::Nbd(nbd.clone()),
        })
    }
}

/// What a driver gets to create an export from: the block backend `blk_exp_add()` made.
#[derive(Debug)]
pub struct ExportArgs<'a> {
    pub graph: &'a BlockGraph,
    pub id: &'a str,
    pub node_name: &'a str,
    /// The backend, with `BLK_PERM_CONSISTENT_READ`, `BLK_PERM_WRITE` if writable, and every
    /// permission shared.
    pub blk: Arc<BlockBackend>,
    pub writable: bool,
}

/// A running export: `BlockExportDriver` and the driver's part of `BlockExport`.
pub trait ExportDriver: Send + Sync {
    /// Whether a client still holds the export, `refcount > 1`.
    fn in_use(&self) -> bool;

    /// `drv->request_shutdown()` followed by `drv->delete()`: disconnect the clients, stop
    /// taking new ones and return once nothing uses the export any more.
    fn shutdown(&self);
}

/// One entry of `block_exports`.
struct Entry {
    id: String,
    ty: ExportType,
    node_name: String,
    /// `!user_owned`.
    shutting_down: bool,
    driver: Arc<dyn ExportDriver>,
}

/// The function that sends `BLOCK_EXPORT_DELETED` for an id.
pub type DeletedHook = Box<dyn Fn(&str) + Send + Sync>;

/// `block_exports`, newest first.
pub struct Exports {
    list: Mutex<Vec<Entry>>,
    on_deleted: DeletedHook,
}

impl std::fmt::Debug for Exports {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exports").finish_non_exhaustive()
    }
}

/// One line of `query-block-exports`, `BlockExportInfo`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportInfo {
    pub id: String,
    pub ty: ExportType,
    pub node_name: String,
    pub shutting_down: bool,
}

impl ExportInfo {
    /// The reply `qmp_marshal_query_block_exports()` builds for this entry.
    pub fn to_qdict(&self) -> QDict {
        QDict::new()
            .with("id", self.id.as_str())
            .with("type", self.ty.as_str())
            .with("node-name", self.node_name.as_str())
            .with("shutting-down", QValue::Bool(self.shutting_down))
    }
}

impl Exports {
    /// An empty list. `on_deleted` runs for each export that goes away.
    pub fn new(on_deleted: DeletedHook) -> Self {
        Exports { list: Mutex::new(Vec::new()), on_deleted }
    }

    fn lock(&self) -> MutexGuard<'_, Vec<Entry>> {
        self.list.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// `blk_exp_find()`: whether an export with this id exists.
    pub fn contains(&self, id: &str) -> bool {
        self.lock().iter().any(|e| e.id == id)
    }

    /// The type of the export `id`.
    pub fn type_of(&self, id: &str) -> Option<ExportType> {
        self.lock().iter().find(|e| e.id == id).map(|e| e.ty)
    }

    /// Puts an export created elsewhere on the list, as `nbd-server-add` does.
    pub fn insert(&self, id: &str, ty: ExportType, node_name: &str, driver: Arc<dyn ExportDriver>) {
        self.lock().insert(
            0,
            Entry {
                id: id.to_string(),
                ty,
                node_name: node_name.to_string(),
                shutting_down: false,
                driver,
            },
        );
    }

    /// `blk_exp_add()`.
    pub fn add(&self, graph: &BlockGraph, export: &ExportOptions) -> Result<()> {
        let fixed_iothread = export.fixed_iothread == Some(true);
        let multithread = matches!(export.iothread, Some(BlockExportIothreads::Multi(_)));
        if fixed_iothread && multithread {
            return Err(Error::generic("Cannot use fixed-iothread for a multi-threaded export"));
        }
        if !ruvm_qapi::cutils::id_wellformed(&export.id) {
            return Err(Error::generic("Invalid block export id"));
        }
        if self.contains(&export.id) {
            return Err(Error::generic(format!(
                "Block export id '{}' is already in use",
                export.id
            )));
        }
        let Some(node) = graph.node(&export.node_name) else {
            // bdrv_lookup_bs(NULL, node_name)
            return Err(Error::generic(format!(
                "Cannot find device='' nor node-name='{}'",
                export.node_name
            )));
        };
        let writable = export.writable.unwrap_or(false);
        if node.read_only && writable {
            return Err(Error::generic("Cannot export read-only node as writable"));
        }
        // ruvm has no iothread objects: a named one is never found.
        match &export.iothread {
            Some(BlockExportIothreads::Single(id)) => {
                return Err(Error::generic(format!("iothread \"{id}\" not found")));
            }
            Some(BlockExportIothreads::Multi(ids)) => {
                let Some(id) = ids.first() else {
                    return Err(Error::generic("The set of I/O threads must not be empty"));
                };
                return Err(Error::generic(format!("iothread \"{id}\" not found")));
            }
            None => {}
        }
        let ty = export.kind.tag();
        if export.allow_inactive == Some(true) && ty != ExportType::Nbd {
            return Err(Error::generic("Export type does not support inactive exports"));
        }

        let driver: Arc<dyn ExportDriver> = if let Some(opts) = export.to_nbd() {
            nbd::create(graph, &opts)?
        } else {
            let mut perm = BLK_PERM_CONSISTENT_READ;
            if writable {
                perm |= BLK_PERM_WRITE;
            }
            let blk = BlockBackend::new(graph, &export.node_name, perm, BLK_PERM_ALL)?;
            blk.set_enable_write_cache(!export.writethrough.unwrap_or(false));
            let args =
                ExportArgs { graph, id: &export.id, node_name: &export.node_name, blk, writable };
            create_other(&args, &export.kind)?
        };
        self.insert(&export.id, ty, &export.node_name, driver);
        Ok(())
    }

    /// `qmp_block_export_del()`.
    pub fn del(&self, id: &str, mode: Option<BlockExportRemoveMode>) -> Result<()> {
        let driver = {
            let mut list = self.lock();
            let Some(e) = list.iter_mut().find(|e| e.id == id) else {
                return Err(Error::generic(format!("Export '{id}' is not found")));
            };
            if e.shutting_down {
                return Err(Error::generic(format!("Export '{id}' is already shutting down")));
            }
            let mode = mode.unwrap_or(BlockExportRemoveMode::Safe);
            if mode == BlockExportRemoveMode::Safe && e.driver.in_use() {
                return Err(Error::generic(format!("export '{id}' still in use"))
                    .hint("Use mode='hard' to force client disconnect\n"));
            }
            e.shutting_down = true;
            e.driver.clone()
        };
        self.finish(id, &driver);
        Ok(())
    }

    /// `blk_exp_request_shutdown()` and the deletion that follows once it is unused.
    fn finish(&self, id: &str, driver: &Arc<dyn ExportDriver>) {
        driver.shutdown();
        self.lock().retain(|e| e.id != id);
        (self.on_deleted)(id);
    }

    /// `blk_exp_close_all_type()`: shut down every export of type `ty`, or every export.
    pub fn close_all(&self, ty: Option<ExportType>) {
        let victims: Vec<(String, Arc<dyn ExportDriver>)> = {
            let mut list = self.lock();
            list.iter_mut()
                .filter(|e| !e.shutting_down && ty.is_none_or(|t| e.ty == t))
                .map(|e| {
                    e.shutting_down = true;
                    (e.id.clone(), e.driver.clone())
                })
                .collect()
        };
        for (id, driver) in victims {
            self.finish(&id, &driver);
        }
    }

    /// `qmp_query_block_exports()`.
    pub fn query(&self) -> Vec<ExportInfo> {
        self.lock()
            .iter()
            .map(|e| ExportInfo {
                id: e.id.clone(),
                ty: e.ty,
                node_name: e.node_name.clone(),
                shutting_down: e.shutting_down,
            })
            .collect()
    }
}

/// `drv->create()` for the types other than NBD.
#[cfg(target_os = "linux")]
fn create_other(args: &ExportArgs<'_>, kind: &ExportKind) -> Result<Arc<dyn ExportDriver>> {
    match kind {
        ExportKind::Nbd(_) => unreachable!("NBD exports are created by the NBD server"),
        ExportKind::VhostUserBlk(o) => vhost_user_blk::create(args, o),
        ExportKind::Fuse(o) => fuse::create(args, o),
        ExportKind::VduseBlk(o) => vduse_blk::create(args, o),
    }
}

/// `blk_exp_find_driver()` finds nothing: the visitor only takes `nbd` on this host.
#[cfg(not(target_os = "linux"))]
fn create_other(_args: &ExportArgs<'_>, _kind: &ExportKind) -> Result<Arc<dyn ExportDriver>> {
    Err(Error::generic("No driver found for the requested export type"))
}
