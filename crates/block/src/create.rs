// SPDX-License-Identifier: GPL-2.0-or-later

//! Image creation: `blockdev-create` from block/create.c, `bdrv_co_create_file()`,
//! `bdrv_co_create_opts_simple()` for the protocol drivers that can only "create" on something
//! that exists (host devices and NBD), and the helpers the format drivers' create functions use
//! from block.c and block-backend.c.
//!
//! Difference from QEMU: `blockdev-create` is a job in QEMU. Here [`BlockGraph::blockdev_create`]
//! runs the driver's create function to the end on the caller's thread; the job wrapper comes
//! with the job layer.

use std::io;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockdevCreateOptions, BlockdevRef, PreallocMode};
use ruvm_qapi::visit::parse_option_size;

use crate::backend::BlockBackend;
use crate::drivers::{self, DriverDef};
use crate::graph::{BlockGraph, OpenCtx};
use crate::node::{BDRV_SECTOR_SIZE, Node, is_enotsup};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_RESIZE, BLK_PERM_WRITE};

impl BlockGraph {
    /// `qmp_blockdev_create()`, run to completion.
    pub fn blockdev_create(&self, options: BlockdevCreateOptions) -> Result<()> {
        let fmt = options.u.tag().as_str();
        let Some(drv) = drivers::find_format(fmt) else {
            return Err(Error::generic(format!("Block driver '{fmt}' not found or not supported")));
        };
        let Some(create) = drv.create else {
            return Err(Error::generic("Driver does not support blockdev-create"));
        };
        create(self, options.u)
    }

    /// `bdrv_create()` as `qemu-img create` calls it: runs the `.bdrv_co_create_opts` of the
    /// format driver `format` on `filename`. `options` holds the `-o` options as strings, with
    /// `size`; the driver (and the protocol driver below it) take out what they know, and what
    /// is left is for the caller to report.
    pub fn create_image(&self, format: &str, filename: &str, options: &mut QDict) -> Result<()> {
        let Some(drv) = drivers::find_format(format) else {
            return Err(Error::generic(format!("Unknown file format '{format}'")));
        };
        let Some(create_opts) = drv.create_opts else {
            return Err(Error::generic(format!(
                "Driver '{}' does not support image creation",
                drv.format_name
            )));
        };
        create_opts(filename, options)
    }

    /// The `create_opts` list of the format driver `format`, in QEMU's declaration order, for
    /// `qemu-img create -o help` and the `Formatting ...` line. `None` for an unknown format.
    pub fn create_opts_list(format: &str) -> Option<&'static [ruvm_qapi::opts::QemuOptDesc]> {
        drivers::find_format(format).map(|d| d.create_opts_list)
    }

    /// `bdrv_co_create_file()`: creates `filename` with the protocol driver its name picks,
    /// from `qemu-img create` style options. The driver takes out the options it knows;
    /// `options` keeps the rest.
    pub(crate) fn create_file(&self, filename: &str, options: &mut QDict) -> Result<()> {
        let drv = drivers::find_protocol(filename, true)?;
        let Some(create_opts) = drv.create_opts else {
            return Err(Error::generic(format!(
                "Driver '{}' does not support image creation",
                drv.format_name
            )));
        };
        create_opts(filename, options)
    }

    /// `bdrv_co_open_blockdev_ref()`: the node `r` names, or a new one from its definition,
    /// opened read-write with `bdrv_open_blockdev_ref()`'s defaults (`cache.direct`,
    /// `cache.no-flush`, `read-only` and `auto-read-only` all off). The node is not in the
    /// monitor's hands: it goes away with the last reference.
    pub(crate) fn open_blockdev_ref(&self, r: BlockdevRef) -> Result<Arc<Node>> {
        match r {
            BlockdevRef::Reference(name) => self.lookup_bs(&name),
            BlockdevRef::Definition(opts) => {
                self.open_nodes(*opts, OpenCtx::default()).map(|r| r.0)
            }
        }
    }

    /// `bdrv_co_open_blockdev_ref()` then `blk_co_new_with_bs(bs, BLK_PERM_WRITE |
    /// BLK_PERM_RESIZE, BLK_PERM_ALL)`, what the create functions of the formats write their
    /// new image through.
    pub(crate) fn open_create_blk(&self, r: BlockdevRef) -> Result<BlockBackend> {
        let bs = self.open_blockdev_ref(r)?;
        BlockBackend::with_node(None, bs, BLK_PERM_WRITE | BLK_PERM_RESIZE, BLK_PERM_ALL)
    }

    /// `blk_co_new_open(filename, NULL, NULL, BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_PROTOCOL)`:
    /// the protocol node for `filename` in a backend that may write and resize it, as the
    /// `.bdrv_co_create_opts` functions of the formats open the file they just created.
    pub(crate) fn open_protocol_blk(&self, filename: &str) -> Result<BlockBackend> {
        let mut options = QDict::new();
        options.put("read-only", "off");
        let ctx = OpenCtx { protocol: true, ..OpenCtx::default() };
        let (bs, _, _) = self.open_nodes_qdict(Some(filename), options, ctx)?;
        BlockBackend::with_node(None, bs, BLK_PERM_WRITE | BLK_PERM_RESIZE, BLK_PERM_ALL)
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn create_file_and_open() {
        let dir = std::env::temp_dir().join(format!("ruvm-create-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a.img");
        let path = path.to_str().unwrap();
        let g = BlockGraph::new();
        let mut o = QDict::new();
        o.put("size", "1000");
        g.create_file(path, &mut o).unwrap();
        assert!(o.is_empty());
        // Rounded up to whole sectors.
        assert_eq!(std::fs::metadata(path).unwrap().len(), 1024);

        let mut o = QDict::new();
        o.put("size", "1M");
        o.put("preallocation", "full");
        g.create_file(&format!("file:{path}"), &mut o).unwrap();
        assert_eq!(std::fs::metadata(path).unwrap().len(), 1 << 20);

        let mut o = QDict::new();
        o.put("preallocation", "bogus");
        let e = g.create_file(path, &mut o).unwrap_err();
        assert_eq!(e.message(), "invalid parameter value: bogus");

        let blk = g.open_protocol_blk(path).unwrap();
        blk.truncate(4096).unwrap();
        blk.pwrite(0, b"hello").unwrap();
        drop(blk);
        assert_eq!(std::fs::metadata(path).unwrap().len(), 4096);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn blockdev_create_errors() {
        let g = BlockGraph::new();
        let o = BlockdevCreateOptions {
            u: ruvm_qapi::types::BlockdevCreateOptionsU::File(
                ruvm_qapi::types::BlockdevCreateOptionsFile {
                    filename: "/nonexistent-dir/x.img".into(),
                    size: 0,
                    preallocation: None,
                    nocow: None,
                    extent_size_hint: None,
                },
            ),
        };
        let e = g.blockdev_create(o).unwrap_err();
        assert_eq!(
            e.message(),
            "Could not create '/nonexistent-dir/x.img': No such file or directory"
        );
    }
}

fn error_is_enotsup(e: &Error) -> bool {
    std::error::Error::source(e).and_then(|c| c.downcast_ref::<io::Error>()).is_some_and(is_enotsup)
}

/// `bdrv_co_create_opts_simple()`: "creating" an image on a device opens it, checks that it
/// is large enough and zeroes its first sector.
pub(crate) fn create_opts_simple(
    drv: &DriverDef,
    filename: &str,
    options: &mut QDict,
) -> Result<()> {
    let take = |o: &mut QDict, k: &str| o.remove(k).and_then(|v| v.as_str().map(str::to_owned));
    let size = match take(options, "size") {
        Some(v) => parse_option_size("size", &v)?,
        None => 0,
    };
    let prealloc = match take(options, "preallocation") {
        Some(v) => PreallocMode::from_name(&v)
            .ok_or_else(|| Error::generic(format!("invalid parameter value: {v}")))?,
        None => PreallocMode::Off,
    };
    if prealloc != PreallocMode::Off {
        return Err(Error::generic(format!(
            "Unsupported preallocation mode '{}'",
            prealloc.as_str()
        )));
    }

    let graph = BlockGraph::new();
    let mut open_opts = QDict::new();
    open_opts.put("driver", drv.format_name);
    open_opts.put("read-only", "off");
    let ctx = OpenCtx { protocol: true, ..OpenCtx::default() };
    let blk = graph
        .open_nodes_qdict(Some(filename), open_opts, ctx)
        .and_then(|(bs, _, _)| {
            BlockBackend::with_node(None, bs, BLK_PERM_WRITE | BLK_PERM_RESIZE, BLK_PERM_ALL)
        })
        .map_err(|e| {
            e.prepend(format_args!(
                "Protocol driver '{}' does not support creating new images, so an existing \
                 image must be selected as the target; however, opening the given target as \
                 an existing image failed: ",
                drv.format_name
            ))
        })?;
    let Some(bs) = blk.root() else {
        return Err(Error::generic("No medium inserted"));
    };

    // create_file_fallback_truncate()
    let truncated = bs.truncate_full(size as i64, false, PreallocMode::Off, 0);
    if let Err(e) = &truncated {
        if !error_is_enotsup(e) {
            return truncated;
        }
    }
    let len = blk
        .getlength()
        .map_err(|e| Error::from_io("Failed to inquire the new image file's length", e))?;
    if len < size {
        return match truncated {
            Err(e) => Err(e),
            Ok(()) => Err(Error::with_cause(
                "Failed to inquire the new image file's length",
                crate::node::errno(libc::ENOTSUP),
            )),
        };
    }

    // create_file_fallback_zero_first_sector()
    let alignment = u64::from(bs.limits().pwrite_zeroes_alignment);
    let bytes_to_clear = len.min(BDRV_SECTOR_SIZE.max(alignment));
    if bytes_to_clear != 0 {
        blk.pwrite_zeroes(0, bytes_to_clear, true)
            .map_err(|e| Error::from_io("Failed to clear the new image's first sector", e))?;
    }
    Ok(())
}
