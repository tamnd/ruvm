// SPDX-License-Identifier: GPL-2.0-or-later

//! The NBD export driver, `blk_exp_nbd` of nbd/server.c, over the process wide server of
//! blockdev-nbd.c.

use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_block::BlockGraph;
use ruvm_block::nbd::NbdServer;
use ruvm_qapi::types::{BlockExportOptions, BlockExportRemoveMode, NbdServerAddOptions};

use super::ExportDriver;

/// One export on the NBD server.
#[derive(Debug)]
pub struct NbdExport {
    server: NbdServer,
    id: String,
}

impl ExportDriver for NbdExport {
    fn in_use(&self) -> bool {
        self.server.export_in_use(&self.id)
    }

    fn shutdown(&self) {
        let _ = self.server.export_remove(&self.id, Some(BlockExportRemoveMode::Hard));
    }
}

fn running() -> Result<NbdServer> {
    ruvm_block::nbd::nbd_server().ok_or_else(|| Error::generic("NBD server not running"))
}

/// `nbd_export_create()`, after `blk_exp_add()` did its checks.
pub fn create(graph: &BlockGraph, opts: &BlockExportOptions) -> Result<Arc<dyn ExportDriver>> {
    let server = running()?;
    server.export_add(graph, opts)?;
    Ok(Arc::new(NbdExport { server, id: opts.id.clone() }))
}

/// `qmp_nbd_server_add()`. Returns the id of the new export.
pub fn server_add(
    graph: &BlockGraph,
    arg: &NbdServerAddOptions,
) -> Result<(String, String, Arc<dyn ExportDriver>)> {
    ruvm_block::nbd::nbd_server_add(graph, arg)?;
    let server = running()?;
    let id = arg.name.clone().unwrap_or_else(|| arg.device.clone());
    let node_name = match graph.backend(&arg.device) {
        Some(blk) => blk.node_name().unwrap_or_default(),
        None => arg.device.clone(),
    };
    Ok((id.clone(), node_name, Arc::new(NbdExport { server, id })))
}
