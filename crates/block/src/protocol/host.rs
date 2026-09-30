// SPDX-License-Identifier: GPL-2.0-or-later

//! The `host_device` and `host_cdrom` protocol drivers from block/file-posix.c: probing,
//! file names, opening and `bdrv_co_create_opts_simple()` from block.c. The request handling
//! is the `file` driver's, in `file.rs`.
//!
//! Differences from QEMU:
//!
//! - `host_device` is built for Linux and macOS, `host_cdrom` for Linux only. QEMU also has
//!   them on FreeBSD, where `host_cdrom` has its own ioctls.
//! - On macOS, `/dev/cdrom` is not looked up in IOKit to find the optical drive's BSD path, so
//!   it has to name a real device node.
//! - SCSI generic devices (`hdev_is_sg()`), dm-multipath path probing and DASD block sizes are
//!   not supported. `probe_blocksizes` is therefore always `ENOTSUP`, as it is in QEMU on any
//!   host but s390x.

use std::io;

use ruvm_base::{Error, Result, report};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockdevOptionsU, PreallocMode};
use ruvm_qapi::visit::parse_option_size;

use crate::backend::BlockBackend;
use crate::drivers::{DriverDef, OpenArgs};
use crate::file::{FileKind, MUTABLE_OPTS, file_open, parse_filename_strip_prefix};
use crate::graph::{BlockGraph, OpenCtx};
use crate::node::{BDRV_SECTOR_SIZE, Driver, is_enotsup};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_RESIZE, BLK_PERM_WRITE};

/// The `host_device` protocol driver, `bdrv_host_device`.
pub(crate) static HOST_DEVICE: DriverDef =
    DriverDef::protocol("host_device", "host_device", hdev_open)
        .with_parse_filename(hdev_parse_filename)
        .with_needs_filename()
        .with_probe_device(hdev_probe_device)
        .with_create_opts(hdev_co_create_opts)
        .with_mutable_opts(MUTABLE_OPTS);

/// The `host_cdrom` protocol driver, `bdrv_host_cdrom`.
#[cfg(target_os = "linux")]
pub(crate) static HOST_CDROM: DriverDef =
    DriverDef::protocol("host_cdrom", "host_cdrom", cdrom_open)
        .with_parse_filename(cdrom_parse_filename)
        .with_needs_filename()
        .with_probe_device(cdrom_probe_device)
        .with_create_opts(cdrom_co_create_opts)
        .with_mutable_opts(MUTABLE_OPTS);

/// `hdev_probe_device()`.
fn hdev_probe_device(filename: &str) -> i32 {
    // Allow a dedicated CD-ROM driver to match with a higher priority.
    if filename.starts_with("/dev/cdrom") {
        return 50;
    }
    match std::fs::metadata(filename) {
        Ok(md) if is_device(&md) => 100,
        _ => 0,
    }
}

fn is_device(md: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::FileTypeExt;
    md.file_type().is_char_device() || md.file_type().is_block_device()
}

/// `cdrom_probe_device()`: a block device that answers `CDROM_DRIVE_STATUS`.
#[cfg(target_os = "linux")]
fn cdrom_probe_device(filename: &str) -> i32 {
    use std::os::fd::AsFd;
    use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
    let Ok(f) =
        std::fs::OpenOptions::new().read(true).custom_flags(libc::O_NONBLOCK).open(filename)
    else {
        return 0;
    };
    if !f.metadata().is_ok_and(|m| m.file_type().is_block_device()) {
        return 0;
    }
    // Attempt to detect via a CD-ROM specific ioctl.
    match crate::sys::dev_ioctl(f.as_fd(), crate::sys::DevIoctl::CdromDriveStatus) {
        Ok(_) => 100,
        Err(_) => 0,
    }
}

/// `hdev_parse_filename()`.
fn hdev_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    parse_filename_strip_prefix(filename, "host_device:", options);
    Ok(())
}

/// `cdrom_parse_filename()`.
#[cfg(target_os = "linux")]
fn cdrom_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    parse_filename_strip_prefix(filename, "host_cdrom:", options);
    Ok(())
}

/// Opens a node of one of the device drivers, the common part of `hdev_open()` and
/// `cdrom_open()`.
fn open_kind(
    kind: FileKind,
    args: &mut OpenArgs<'_>,
    opts: BlockdevOptionsU,
) -> Result<Box<dyn Driver>> {
    let (BlockdevOptionsU::HostDevice(mut o) | BlockdevOptionsU::HostCdrom(mut o)) = opts else {
        unreachable!("host device driver with other options")
    };
    if o.aio.is_none() && args.ctx.inherit.native_aio {
        o.aio = Some(ruvm_qapi::types::BlockdevAioOptions::Native);
    }
    let auto_read_only = args.flags.auto_read_only;
    let d = match file_open(kind, &o, &mut args.flags, auto_read_only) {
        Ok(d) => d,
        Err(e) => {
            // If a physical device experienced an error while being opened.
            if cfg!(target_os = "macos") && o.filename.starts_with("/dev/") {
                print_unmounting_directions(&o.filename);
            }
            return Err(e);
        }
    };
    args.meta.filename = o.filename.clone();
    Ok(Box::new(d))
}

/// `print_unmounting_directions()`.
fn print_unmounting_directions(file_name: &str) {
    report::error_report(&format!(
        "If device {file_name} is mounted on the desktop, unmount it first before using it in QEMU"
    ));
    report::error_report(&format!("Command to unmount device: diskutil unmountDisk {file_name}"));
    report::error_report(&format!("Command to mount device: diskutil mountDisk {file_name}"));
}

/// `hdev_open()`.
fn hdev_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    open_kind(FileKind::HostDevice, args, opts)
}

/// `cdrom_open()`: `raw_open_common()` with `O_NONBLOCK`, so that an empty drive opens.
#[cfg(target_os = "linux")]
fn cdrom_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    open_kind(FileKind::HostCdrom, args, opts)
}

fn hdev_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    create_opts_simple(&HOST_DEVICE, filename, options)
}

#[cfg(target_os = "linux")]
fn cdrom_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    create_opts_simple(&HOST_CDROM, filename, options)
}

fn error_is_enotsup(e: &Error) -> bool {
    std::error::Error::source(e).and_then(|c| c.downcast_ref::<io::Error>()).is_some_and(is_enotsup)
}

/// `bdrv_co_create_opts_simple()`: "creating" an image on a device opens it, checks that it
/// is large enough and zeroes its first sector.
fn create_opts_simple(drv: &DriverDef, filename: &str, options: &mut QDict) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeFlags;
    use ruvm_qapi::types::BlockdevOptionsFile;

    fn temp_file(name: &str, len: u64) -> String {
        let path = std::env::temp_dir().join(format!("ruvm-host-{}-{name}", std::process::id()));
        let f = std::fs::File::create(&path).unwrap();
        f.set_len(len).unwrap();
        path.to_str().unwrap().to_owned()
    }

    fn options(filename: &str) -> BlockdevOptionsFile {
        BlockdevOptionsFile { filename: filename.to_owned(), ..Default::default() }
    }

    #[test]
    fn probe_device_names() {
        assert_eq!(hdev_probe_device("/dev/cdrom"), 50);
        assert_eq!(hdev_probe_device("/dev/cdrom1"), 50);
        assert_eq!(hdev_probe_device("/dev/null"), 100);
        let f = temp_file("probe", 0);
        assert_eq!(hdev_probe_device(&f), 0);
        assert_eq!(hdev_probe_device("/nonexistent/ruvm"), 0);
        std::fs::remove_file(&f).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cdrom_probe_not_a_cdrom() {
        // A character device and a regular file are not CD-ROM drives.
        assert_eq!(cdrom_probe_device("/dev/null"), 0);
        let f = temp_file("cdprobe", 0);
        assert_eq!(cdrom_probe_device(&f), 0);
        std::fs::remove_file(&f).unwrap();
    }

    #[test]
    fn find_protocol_picks_host_device() {
        let d = crate::drivers::find_protocol("/dev/null", true).unwrap();
        assert_eq!(d.format_name, "host_device");
        let d = crate::drivers::find_protocol("host_device:/x", true).unwrap();
        assert_eq!(d.format_name, "host_device");
    }

    #[test]
    fn parse_filename_prefix() {
        let mut o = QDict::new();
        hdev_parse_filename("host_device:/dev/sda", &mut o).unwrap();
        assert_eq!(o.get("filename").and_then(|v| v.as_str()), Some("/dev/sda"));
    }

    #[test]
    fn host_device_needs_a_device() {
        let f = temp_file("regular", 4096);
        let mut flags = NodeFlags::default();
        let e = file_open(FileKind::HostDevice, &options(&f), &mut flags, false).err().unwrap();
        assert_eq!(
            e.message(),
            format!("'host_device' driver requires '{f}' to be either a character or block device")
        );
        std::fs::remove_file(&f).unwrap();

        let mut flags = NodeFlags { read_only: true, ..NodeFlags::default() };
        let e = file_open(FileKind::File, &options("/dev/null"), &mut flags, false).err().unwrap();
        assert_eq!(e.message(), "'file' driver requires '/dev/null' to be a regular file");
    }

    #[test]
    fn host_device_opens_char_device() {
        let mut flags = NodeFlags::default();
        let d = file_open(FileKind::HostDevice, &options("/dev/null"), &mut flags, false).unwrap();
        assert_eq!(d.kind(), FileKind::HostDevice);
        assert_eq!(d.engine_name(), "threads");
    }

    #[test]
    fn logical_blocksize_of_regular_file() {
        let f = temp_file("blksz", 0);
        let file = std::fs::File::open(&f).unwrap();
        assert!(crate::file::probe_logical_blocksize(&file).is_err());
        assert!(crate::file::probe_physical_blocksize(&file).is_err());
        std::fs::remove_file(&f).unwrap();
    }
}
