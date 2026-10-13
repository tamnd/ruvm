// SPDX-License-Identifier: GPL-2.0-or-later

//! The `-drive` images by id, for the QMP commands that name a drive, as `blk_by_name()`
//! finds the `BlockBackend` of a drive in QEMU.
//!
//! The boards that keep this list are aarch64 `virt` and `sbsa-ref`, riscv64 `virt` and machine
//! `none`. On them a drive is on a virtio-blk device, on a flash or disk ruvm cannot resize yet,
//! or on nothing (`if=none` with no device).

use std::sync::{Arc, Weak};

use ruvm_base::{Error, Result};
use ruvm_hw_virtio::{VirtioMmio, VirtioPci};
use ruvm_machine_x86::FileBackend;

use crate::x86::{Drive, DriveIf};

/// `BlockDevOps.resize_cb` of a device: tells the guest the size of its image changed.
pub(crate) type ResizeCb = Arc<dyn Fn() + Send + Sync>;

/// A drive a board opened for a device, with what the device does when the image is resized.
pub(crate) struct Plugged {
    /// The index of the drive in the `-drive` list.
    pub(crate) drive: usize,
    /// The image, shared with the device.
    pub(crate) image: FileBackend,
    pub(crate) resize_cb: ResizeCb,
}

/// How a drive is used.
enum Use {
    /// On no device. The image is opened when a command needs it, and `None` has no medium.
    Free(Option<Drive>),
    /// On a device that follows a change of size.
    Device(FileBackend, ResizeCb),
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
                    DriveIf::None => Use::Free(d.file.is_some().then(|| d.clone())),
                    _ => Use::Other,
                };
                (d.id.clone(), u)
            })
            .collect();
        for p in plugged {
            list[p.drive].1 = Use::Device(p.image, p.resize_cb);
        }
        Drives(Some(list))
    }

    /// `qmp_block_resize()` for a drive. `node_exists` says whether `blockdev-add` made a
    /// node of that name, which ruvm cannot resize yet.
    pub(crate) fn block_resize(
        &self,
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
        if matches!(u, Use::Free(None)) {
            return Err(Error::generic(format!("Device '{id}' has no medium")));
        }
        let Ok(size) = u64::try_from(size) else {
            return Err(Error::generic("Parameter 'size' expects a >0 size"));
        };
        match u {
            Use::Free(Some(d)) => d.open().and_then(|image| image.truncate(size)),
            Use::Device(image, resize_cb) => image.truncate(size).map(|()| resize_cb()),
            Use::Free(None) | Use::Other => {
                return Err(Error::generic(format!(
                    "block_resize of drive '{id}' is not supported with its device by ruvm yet"
                )));
            }
        }
        .map_err(Error::generic)
    }
}

/// `virtio_blk_resize()` of a virtio-blk on PCI: `virtio_notify_config()`.
pub(crate) fn pci_resize_cb(dev: &VirtioPci) -> ResizeCb {
    let dev = dev.downgrade();
    Arc::new(move || {
        if let Some(dev) = dev.upgrade() {
            dev.with_backend(|b| b.vdev_mut().notify_config());
        }
    })
}

/// `virtio_blk_resize()` of a virtio-blk on virtio-mmio.
pub(crate) fn mmio_resize_cb(t: Option<Arc<VirtioMmio>>) -> ResizeCb {
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

    use super::*;
    use crate::x86::{drive_opts, parse_drive};

    fn drives(args: &[&str]) -> Vec<Drive> {
        let mut list = drive_opts();
        let mut out = Vec::new();
        for a in args {
            let d = parse_drive(&mut list, a, None, &out, None).unwrap();
            out.push(d);
        }
        out
    }

    #[test]
    fn block_resize_is_qemus() {
        let dir = std::env::temp_dir().join(format!("ruvm-drives-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (a, b) = (dir.join("a.img"), dir.join("b.img"));
        std::fs::write(&a, [0u8; 4096]).unwrap();
        std::fs::write(&b, [0u8; 4096]).unwrap();
        let d = drives(&[
            &format!("if=none,id=d0,file={},format=raw", a.display()),
            &format!("if=none,id=d1,file={},format=raw", b.display()),
            "if=none,id=d2,file=null-co://,format=raw",
            "if=none,id=d3",
        ]);
        let resized = Arc::new(AtomicUsize::new(0));
        let r = Arc::clone(&resized);
        let image = d[0].open().unwrap();
        let cb: ResizeCb = Arc::new(move || {
            r.fetch_add(1, Ordering::Relaxed);
        });
        let device = image.clone();
        let drives = Drives::new(&d, vec![Plugged { drive: 0, image, resize_cb: cb }]);
        let resize = |dev: Option<&str>, node: Option<&str>, size| {
            drives.block_resize(dev, node, size, |n| n == "n0").map_err(|e| e.message().to_string())
        };

        resize(Some("d0"), None, 8192).unwrap();
        assert_eq!(device.size(), 8192);
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
        let e = Drives::default().block_resize(Some("d0"), None, 1, |_| false).unwrap_err();
        assert_eq!(e.message(), "block_resize is not supported with this machine by ruvm yet");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
