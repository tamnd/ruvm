// SPDX-License-Identifier: GPL-2.0-or-later

//! The `-drive` images by id, for the QMP commands that name a drive, as `blk_by_name()`
//! finds the `BlockBackend` of a drive in QEMU.
//!
//! The boards that keep this list are aarch64 `virt` and `sbsa-ref`, riscv64 `virt` and machine
//! `none`. On them a drive is on a virtio-blk device, on a flash or disk ruvm cannot resize yet,
//! or on nothing (`if=none` with no device).

use std::io;
use std::sync::{Arc, Weak};

use ruvm_base::report::{Location, push_location};
use ruvm_base::{Error, Result};
use ruvm_block::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE,
    BLK_PERM_WRITE_UNCHANGED, BlockBackend, BlockGraph, BlockInterfaceType,
};
use ruvm_hw_virtio::{VirtioMmio, VirtioPci};
use ruvm_machine_x86::VirtioHandle;

use crate::x86::{Drive, DriveIf, Located};

/// `drive_new()` past its checks: the block layer opens the image of each planned drive and
/// adds its block backend under the drive id, in command line order. `args` are the `-drive`
/// arguments the drives came from.
pub(crate) fn open_drives(
    graph: &BlockGraph,
    drives: &[Drive],
    args: &[(String, Option<Location>)],
) -> std::result::Result<(), Located> {
    for (d, (arg, loc)) in drives.iter().zip(args) {
        let _loc = loc.clone().map(push_location);
        let default_if = BlockInterfaceType::parse(d.iface.name()).expect("an if= name");
        graph.drive_new(arg, default_if).map_err(|e| Located(loc.clone(), e))?;
    }
    Ok(())
}

/// `blkconf_apply_backend_options()`: the image of drive `d` for a device, which holds the
/// permissions a device that writes unless `readonly`, and lets others resize the image if
/// `resizable`.
pub(crate) fn device_image(
    graph: &BlockGraph,
    d: &Drive,
    readonly: bool,
    resizable: bool,
) -> Result<DriveImage> {
    let blk = graph.backend(&d.id).expect("drive_new() added it");
    let mut perm = BLK_PERM_CONSISTENT_READ;
    if !readonly {
        perm |= BLK_PERM_WRITE;
    }
    let mut shared = BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE_UNCHANGED;
    if resizable {
        shared |= BLK_PERM_RESIZE;
    }
    blk.set_perm(perm, shared)?;
    Ok(DriveImage(blk))
}

/// The block backend of a drive as the image of a device model.
#[derive(Clone, Debug)]
pub(crate) struct DriveImage(Arc<BlockBackend>);

impl ruvm_hw_virtio::BlockBackend for DriveImage {
    fn size(&self) -> u64 {
        self.0.getlength().unwrap_or(0)
    }

    fn is_writable(&self) -> bool {
        self.0.is_writable()
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.pread(offset, buf)
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.0.pwrite(offset, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()
    }
}

impl ruvm_hw_storage::BlockBackend for DriveImage {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.pread(offset, buf)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.0.pwrite(offset, buf)
    }

    fn flush(&self) -> io::Result<()> {
        self.0.flush()
    }

    fn len(&self) -> u64 {
        self.0.getlength().unwrap_or(0)
    }

    fn is_read_only(&self) -> bool {
        self.0.is_read_only()
    }
}

/// `BlockDevOps.resize_cb` of a device: tells the guest the size of its image changed.
pub(crate) type ResizeCb = Arc<dyn Fn() + Send + Sync>;

/// A drive a board put on a device, with what the device does when the image is resized.
pub(crate) struct Plugged {
    /// The index of the drive in the `-drive` list.
    pub(crate) drive: usize,
    pub(crate) resize_cb: ResizeCb,
}

/// How a drive is used.
enum Use {
    /// On no device.
    Free,
    /// On a device that follows a change of size.
    Device(ResizeCb),
    /// On a device ruvm cannot resize yet, such as a flash.
    Other,
}

/// The drives of the machine, or `None` on a machine that does not keep the list yet.
#[derive(Default)]
pub(crate) struct Drives(Option<Vec<(String, Use)>>);

impl std::fmt::Debug for Drives {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Option<Vec<&str>> =
            self.0.as_ref().map(|v| v.iter().map(|(id, _)| id.as_str()).collect());
        f.debug_tuple("Drives").field(&ids).finish()
    }
}

impl Drives {
    /// The list for `drives`, of which the board put `plugged` on devices.
    pub(crate) fn new(drives: &[Drive], plugged: Vec<Plugged>) -> Drives {
        let mut list: Vec<(String, Use)> = drives
            .iter()
            .map(|d| {
                let u = match d.iface {
                    DriveIf::None => Use::Free,
                    _ => Use::Other,
                };
                (d.id.clone(), u)
            })
            .collect();
        for p in plugged {
            list[p.drive].1 = Use::Device(p.resize_cb);
        }
        Drives(Some(list))
    }

    /// `qmp_block_resize()` for a drive of `graph`. `node_exists` says whether
    /// `blockdev-add` made a node of that name, which ruvm cannot resize yet.
    pub(crate) fn block_resize(
        &self,
        graph: &BlockGraph,
        device: Option<&str>,
        node_name: Option<&str>,
        size: i64,
        node_exists: impl Fn(&str) -> bool,
    ) -> Result<()> {
        let Some(list) = &self.0 else {
            return Err(Error::generic(
                "block_resize is not supported with this machine by ruvm yet",
            ));
        };
        // bdrv_lookup_bs()
        let found = device.and_then(|id| list.iter().find(|(d, _)| d == id));
        let Some((id, u)) = found else {
            if node_name.is_some_and(node_exists) {
                return Err(Error::generic(
                    "block_resize of a block node is not supported by ruvm yet",
                ));
            }
            return Err(Error::generic(format!(
                "Cannot find device='{}' nor node-name='{}'",
                device.unwrap_or(""),
                node_name.unwrap_or("")
            )));
        };
        let node = graph.backend(id).and_then(|blk| blk.node_name());
        let Some(node) = node else {
            return Err(Error::generic(format!("Device '{id}' has no medium")));
        };
        if size < 0 {
            return Err(Error::generic("Parameter 'size' expects a >0 size"));
        }
        if matches!(u, Use::Other) {
            return Err(Error::generic(format!(
                "block_resize of drive '{id}' is not supported with its device by ruvm yet"
            )));
        }
        // A backend of its own that may resize the node, as qmp_block_resize() makes.
        BlockBackend::new(graph, &node, BLK_PERM_RESIZE, BLK_PERM_ALL)?.truncate(size as u64)?;
        if let Use::Device(resize_cb) = u {
            resize_cb();
        }
        Ok(())
    }
}

/// `virtio_blk_resize()` of a plugged virtio-blk, or a callback that does nothing if the board
/// lost the transport.
pub(crate) fn resize_cb(handle: Option<&VirtioHandle>) -> ResizeCb {
    match handle {
        Some(VirtioHandle::Pci(dev)) => pci_resize_cb(dev),
        Some(VirtioHandle::Mmio(t)) => mmio_resize_cb(Some(Arc::clone(t))),
        None => mmio_resize_cb(None),
    }
}

/// `virtio_blk_resize()` of a virtio-blk on PCI: `virtio_notify_config()`.
fn pci_resize_cb(dev: &VirtioPci) -> ResizeCb {
    let dev = dev.downgrade();
    Arc::new(move || {
        if let Some(dev) = dev.upgrade() {
            dev.with_backend(|b| b.vdev_mut().notify_config());
        }
    })
}

/// `virtio_blk_resize()` of a virtio-blk on virtio-mmio.
fn mmio_resize_cb(t: Option<Arc<VirtioMmio>>) -> ResizeCb {
    let t = t.map(|t| Arc::downgrade(&t));
    Arc::new(move || {
        if let Some(t) = t.as_ref().and_then(Weak::upgrade) {
            t.with_backend(|b| b.vdev_mut().notify_config());
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use ruvm_hw_storage::BlockBackend as _;

    use super::*;
    use crate::x86::{drive_opts, parse_drive};

    fn drives(graph: &BlockGraph, args: &[&str]) -> Vec<Drive> {
        let mut list = drive_opts();
        let mut out = Vec::new();
        for a in args {
            let d = parse_drive(&mut list, a, None, &out, None).unwrap();
            out.push(d);
        }
        let args: Vec<_> = args.iter().map(|a| (a.to_string(), None)).collect();
        open_drives(graph, &out, &args).unwrap();
        out
    }

    #[test]
    fn block_resize_is_qemus() {
        let dir = std::env::temp_dir().join(format!("ruvm-drives-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.img"), dir.join("b.img"));
        std::fs::write(&a, [0u8; 4096]).unwrap();
        std::fs::write(&b, [0u8; 4096]).unwrap();
        let graph = BlockGraph::new();
        let d = drives(
            &graph,
            &[
                &format!("if=none,id=d0,file={},format=raw", a.display()),
                &format!("if=none,id=d1,file={},format=raw", b.display()),
                "if=none,id=d2,file=null-co://,format=raw",
                "if=none,id=d3",
            ],
        );
        let resized = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&resized);
        let device = device_image(&graph, &d[0], false, true).unwrap();
        let cb: ResizeCb = Arc::new(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        let drives = Drives::new(&d, vec![Plugged { drive: 0, resize_cb: cb }]);
        let resize = |dev: Option<&str>, node: Option<&str>, size| {
            drives
                .block_resize(&graph, dev, node, size, |n| n == "n0")
                .map_err(|e| e.message().to_string())
        };

        resize(Some("d0"), None, 8192).unwrap();
        assert_eq!(device.len(), 8192);
        assert_eq!(resized.load(Ordering::Relaxed), 1);
        resize(Some("d1"), None, 1000).unwrap();
        assert_eq!(std::fs::metadata(&b).unwrap().len(), 1000);
        let e = resize(Some("d2"), None, 1).unwrap_err();
        assert_eq!(e, "Image format driver does not support resize");
        assert_eq!(resize(Some("d3"), None, 1).unwrap_err(), "Device 'd3' has no medium");
        assert_eq!(resize(Some("d0"), None, -1).unwrap_err(), "Parameter 'size' expects a >0 size");
        let e = resize(Some("zz"), None, 1).unwrap_err();
        assert_eq!(e, "Cannot find device='zz' nor node-name=''");
        let e = resize(None, Some("zz"), 1).unwrap_err();
        assert_eq!(e, "Cannot find device='' nor node-name='zz'");
        let e = resize(None, Some("n0"), 1).unwrap_err();
        assert_eq!(e, "block_resize of a block node is not supported by ruvm yet");
        let e = Drives::default().block_resize(&graph, Some("d0"), None, 1, |_| false).unwrap_err();
        assert_eq!(e.message(), "block_resize is not supported with this machine by ruvm yet");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
