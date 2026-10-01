// SPDX-License-Identifier: GPL-2.0-or-later

//! The `host_device` and `host_cdrom` protocol drivers from block/file-posix.c: probing,
//! file names and opening. Creating an image on one is `bdrv_co_create_opts_simple()`, in
//! `create.rs`, and the request handling is the `file` driver's, in `file.rs`.
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

use ruvm_base::{Result, report};
use ruvm_qapi::QDict;
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::file::{FileKind, MUTABLE_OPTS, file_open, parse_filename_strip_prefix};
use crate::node::Driver;

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
    crate::create::create_opts_simple(&HOST_DEVICE, filename, options)
}

#[cfg(target_os = "linux")]
fn cdrom_co_create_opts(filename: &str, options: &mut QDict) -> Result<()> {
    crate::create::create_opts_simple(&HOST_CDROM, filename, options)
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
