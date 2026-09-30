// SPDX-License-Identifier: GPL-2.0-or-later

//! Legacy `-drive`: `drive_new()` and `blockdev_init()` from blockdev.c, and the `DriveInfo`
//! table from block/block-backend.c that boards look drives up in.
//!
//! A `-drive` turns into a node tree (a format node over a `file` node) and a named block
//! backend on top, as in QEMU. The board then creates the device for the drive, the
//! [`DriveInfo`] says which one.

use std::sync::Arc;

use ruvm_base::{Error, Result, report};
use ruvm_qapi::opts::QemuOptsList;
use ruvm_qapi::types::{BlockdevDetectZeroesOptions, BlockdevOnError, BlockdevOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, qapi_bool_parse};
use ruvm_qapi::{QDict, QValue};

use crate::backend::BlockBackend;
use crate::graph::{BlockGraph, Inherited, OpenCtx};
use crate::perm::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ};

/// `BlockInterfaceType`: the kind of controller a `-drive` is for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum BlockInterfaceType {
    /// `if=none`: the drive only makes a backend, a `-device` picks it up.
    None,
    /// `if=ide`.
    #[default]
    Ide,
    /// `if=scsi`.
    Scsi,
    /// `if=floppy`.
    Floppy,
    /// `if=pflash`.
    Pflash,
    /// `if=mtd`.
    Mtd,
    /// `if=sd`.
    Sd,
    /// `if=virtio`.
    Virtio,
    /// `if=xen`.
    Xen,
}

impl BlockInterfaceType {
    const ALL: [BlockInterfaceType; 9] = [
        Self::None,
        Self::Ide,
        Self::Scsi,
        Self::Floppy,
        Self::Pflash,
        Self::Mtd,
        Self::Sd,
        Self::Virtio,
        Self::Xen,
    ];

    /// `if_name[]`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Ide => "ide",
            Self::Scsi => "scsi",
            Self::Floppy => "floppy",
            Self::Pflash => "pflash",
            Self::Mtd => "mtd",
            Self::Sd => "sd",
            Self::Virtio => "virtio",
            Self::Xen => "xen",
        }
    }

    /// `if_max_devs[]`: units per bus, 0 for interfaces where only the unit counts.
    pub fn max_devs(self) -> u32 {
        match self {
            Self::Ide => 2,
            Self::Scsi => 7,
            _ => 0,
        }
    }

    /// Parses an `if=` value.
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.as_str() == s)
    }
}

/// `DriveInfo`, plus what `drive_new()` worked out for the board.
#[derive(Clone, Debug, PartialEq)]
pub struct DriveInfo {
    /// The drive id, which is also the name of its block backend.
    pub id: String,
    /// The controller type.
    pub interface: BlockInterfaceType,
    /// The bus number on the controller.
    pub bus: u32,
    /// The unit number on the bus.
    pub unit: u32,
    /// `media=cdrom` on an interface that tells disks and CD-ROMs apart.
    pub media_cd: bool,
    /// Whether the drive was opened read-only (`read-only=on` or `media=cdrom`).
    pub read_only: bool,
    /// `copy-on-read`, off after the read-only check.
    pub copy_on_read: bool,
    /// The guest visible write cache, off for `cache=writethrough` and `cache=directsync`.
    pub write_cache: bool,
    /// `rerror`.
    pub rerror: BlockdevOnError,
    /// `werror`.
    pub werror: BlockdevOnError,
    /// The device the board adds for this drive by itself: `virtio-blk` for `if=virtio`,
    /// `xen-disk` or `xen-cdrom` for `if=xen`. Other interfaces are wired up by the board.
    pub device: Option<&'static str>,
    /// The node name of the root node, `None` for a drive without a medium.
    pub node_name: Option<String>,
    /// The options the node tree was opened with, `None` for a drive without a medium.
    pub blockdev: Option<BlockdevOptions>,
    /// The warnings printed while setting the drive up, in order.
    pub warnings: Vec<String>,
}

/// The `-drive` options as `QemuOpts` holds them: in order, repeated keys allowed, the last
/// one wins.
#[derive(Debug, Default)]
struct DriveOpts(Vec<(String, String)>);

impl DriveOpts {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn has(&self, key: &str) -> bool {
        self.0.iter().any(|(k, _)| k == key)
    }

    fn set(&mut self, key: &str, value: &str) {
        self.0.push((key.to_string(), value.to_string()));
    }

    fn take(&mut self, key: &str) -> Option<String> {
        let v = self.get(key).map(str::to_string);
        self.0.retain(|(k, _)| k != key);
        v
    }

    /// `qemu_opt_rename()`.
    fn rename(&mut self, from: &str, to: &str) -> Result<()> {
        if self.has(from) && self.has(to) {
            return Err(Error::generic(format!(
                "'{to}' and its alias '{from}' can't be used at the same time"
            )));
        }
        for (k, _) in &mut self.0 {
            if k == from {
                *k = to.to_string();
            }
        }
        Ok(())
    }

    fn take_bool(&mut self, key: &str) -> Result<Option<bool>> {
        self.take(key).map(|v| qapi_bool_parse(key, &v)).transpose()
    }

    fn take_number(&mut self, key: &str) -> Result<Option<u64>> {
        use ruvm_qapi::cutils::{Errno, strtou64};
        self.take(key)
            .map(|v| match strtou64(&v, 0, true) {
                Ok((n, _)) => Ok(n),
                Err((Errno::Range, _)) => {
                    Err(Error::generic(format!("Value '{v}' is too large for parameter '{key}'")))
                }
                Err(_) => Err(Error::generic(format!("Parameter '{key}' expects a number"))),
            })
            .transpose()
    }
}

/// `bdrv_parse_cache_mode()`: (`cache.writeback`, `cache.direct`, `cache.no-flush`).
fn parse_cache_mode(mode: &str) -> Option<(bool, bool, bool)> {
    Some(match mode {
        "off" | "none" => (true, true, false),
        "directsync" => (false, true, false),
        "writeback" => (true, false, false),
        "unsafe" => (true, false, true),
        "writethrough" => (false, false, false),
        _ => return None,
    })
}

/// `parse_block_error_action()`.
fn parse_error_action(s: &str, is_read: bool) -> Result<BlockdevOnError> {
    Ok(match s {
        "ignore" => BlockdevOnError::Ignore,
        "enospc" if !is_read => BlockdevOnError::Enospc,
        "stop" => BlockdevOnError::Stop,
        "report" => BlockdevOnError::Report,
        _ => {
            let what = if is_read { "read" } else { "write" };
            return Err(Error::generic(format!("'{s}' invalid {what} error action")));
        }
    })
}

/// Puts `value` at the dotted `path` in `dict`, the way `qdict_crumple()` nests flat keys.
fn put_path(dict: &mut QDict, path: &str, value: &str) -> Result<()> {
    match path.split_once('.') {
        None => {
            if dict.get(path).is_some_and(|v| v.as_dict().is_some()) {
                return Err(Error::generic(format!("Parameter '{path}' used inconsistently")));
            }
            dict.put(path, value);
        }
        Some((head, rest)) => {
            if !dict.contains_key(head) {
                dict.put(head, QDict::new());
            }
            let Some(sub) = dict.get_mut(head).and_then(QValue::as_dict_mut) else {
                return Err(Error::generic(format!("Parameter '{head}' used inconsistently")));
            };
            put_path(sub, rest, value)?;
        }
    }
    Ok(())
}

/// The options `blockdev_init()` keeps for itself, `qemu_common_drive_opts`, apart from the
/// ones it reads by name.
fn is_common_opt(k: &str) -> bool {
    k.starts_with("throttling.") || k.starts_with("stats-")
}

impl BlockGraph {
    /// `drive_new()`: parses a `-drive` argument, opens its node tree and adds its block
    /// backend under the drive id. `default_if` is the machine's `block_default_type`.
    ///
    /// The warnings QEMU prints go to stderr here too, and are also in
    /// [`DriveInfo::warnings`].
    pub fn drive_new(&self, params: &str, default_if: BlockInterfaceType) -> Result<DriveInfo> {
        let mut list = QemuOptsList::new("drive", &[]);
        let parsed = list.parse(params, false)?;
        let user_id = parsed.id().map(str::to_string);
        let mut o = DriveOpts(parsed.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect());
        let mut warnings = Vec::new();

        // Change legacy command line options into QMP ones.
        const RENAMES: [(&str, &str); 15] = [
            ("iops", "throttling.iops-total"),
            ("iops_rd", "throttling.iops-read"),
            ("iops_wr", "throttling.iops-write"),
            ("bps", "throttling.bps-total"),
            ("bps_rd", "throttling.bps-read"),
            ("bps_wr", "throttling.bps-write"),
            ("iops_max", "throttling.iops-total-max"),
            ("iops_rd_max", "throttling.iops-read-max"),
            ("iops_wr_max", "throttling.iops-write-max"),
            ("bps_max", "throttling.bps-total-max"),
            ("bps_rd_max", "throttling.bps-read-max"),
            ("bps_wr_max", "throttling.bps-write-max"),
            ("iops_size", "throttling.iops-size"),
            ("group", "throttling.group"),
            ("readonly", "read-only"),
        ];
        for (from, to) in RENAMES {
            o.rename(from, to)?;
        }

        if let Some(mode) = o.take("cache") {
            let Some((wb, direct, no_flush)) = parse_cache_mode(&mode) else {
                return Err(Error::generic("invalid cache option"));
            };
            // Specific options take precedence.
            let on_off = |b: bool| if b { "on" } else { "off" };
            if !o.has("cache.writeback") {
                o.set("cache.writeback", on_off(wb));
            }
            if !o.has("cache.direct") {
                o.set("cache.direct", on_off(direct));
            }
            if !o.has("cache.no-flush") {
                o.set("cache.no-flush", on_off(no_flush));
            }
        }

        // qemu_legacy_drive_opts.
        let bus = o.take_number("bus")?;
        let unit = o.take_number("unit")?;
        let index = o.take_number("index")?;
        let media = o.take("media");
        let if_name = o.take("if");
        let filename = o.take("file");
        let ro_opt = o.take_bool("read-only")?;
        let rerror = o.take("rerror");
        let werror = o.take("werror");
        let cor_opt = o.take_bool("copy-on-read")?;

        let mut read_only = false;
        let mut media_cd = false;
        if let Some(m) = &media {
            match m.as_str() {
                "disk" => {}
                "cdrom" => {
                    media_cd = true;
                    read_only = true;
                }
                _ => return Err(Error::generic(format!("'{m}' invalid media"))),
            }
        }
        read_only |= ro_opt.unwrap_or(false);
        let mut copy_on_read = cor_opt.unwrap_or(false);
        if read_only && copy_on_read {
            let w = "disabling copy-on-read on read-only drive";
            report::warn_report(w);
            warnings.push(format!("warning: {w}"));
            copy_on_read = false;
        }

        let ty = match &if_name {
            Some(v) => BlockInterfaceType::parse(v)
                .ok_or_else(|| Error::generic(format!("unsupported bus type '{v}'")))?,
            None => default_if,
        };

        // Where the drive sits, by bus and unit or by index, or else the first free slot.
        let max = ty.max_devs();
        let to_int = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        let mut bus_id = bus.map_or(0, to_int);
        let mut unit_id = unit.map_or(-1, to_int);
        let index_id = index.map_or(-1, to_int);
        if index.is_some() {
            if bus_id != 0 || unit_id != -1 {
                return Err(Error::generic("index cannot be used with bus and unit"));
            }
            let max = i64::from(max);
            (bus_id, unit_id) = match index_id.checked_div(max) {
                Some(bus) => (bus, index_id % max),
                None => (0, index_id),
            };
        }
        if unit_id == -1 {
            unit_id = 0;
            while self.drive_get(ty, bus_id as u32, unit_id as u32).is_some() {
                unit_id += 1;
                if max != 0 && unit_id >= i64::from(max) {
                    unit_id -= i64::from(max);
                    bus_id += 1;
                }
            }
        }
        if max != 0 && unit_id >= i64::from(max) {
            return Err(Error::generic(format!("unit {unit_id} too big (max is {})", max - 1)));
        }
        let (bus_id, unit_id) = (bus_id as u32, unit_id as u32);
        if self.drive_get(ty, bus_id, unit_id).is_some() {
            return Err(Error::generic(format!(
                "drive with bus={bus_id}, unit={unit_id} (index={index_id}) exists"
            )));
        }

        let id = user_id.unwrap_or_else(|| {
            let mediastr = match ty {
                BlockInterfaceType::Ide | BlockInterfaceType::Scsi if media_cd => "-cd",
                BlockInterfaceType::Ide | BlockInterfaceType::Scsi => "-hd",
                _ => "",
            };
            if max != 0 {
                format!("{}{bus_id}{mediastr}{unit_id}", ty.as_str())
            } else {
                format!("{}{mediastr}{unit_id}", ty.as_str())
            }
        });

        let device = match ty {
            BlockInterfaceType::Virtio => Some("virtio-blk"),
            BlockInterfaceType::Xen if media_cd => Some("xen-cdrom"),
            BlockInterfaceType::Xen => Some("xen-disk"),
            _ => None,
        };

        let error_bus = matches!(
            ty,
            BlockInterfaceType::Ide
                | BlockInterfaceType::Scsi
                | BlockInterfaceType::Virtio
                | BlockInterfaceType::None
        );
        if werror.is_some() && !error_bus {
            return Err(Error::generic("werror is not supported by this bus type"));
        }
        if rerror.is_some() && !error_bus {
            return Err(Error::generic("rerror is not supported by this bus type"));
        }

        // blockdev_init(): the common options first.
        let snapshot = o.take_bool("snapshot")?.unwrap_or(false);
        let writethrough = !o.take_bool("cache.writeback")?.unwrap_or(true);
        let aio = o.take("aio");
        let native_aio = match aio.as_deref() {
            None | Some("threads") => false,
            Some("native") => true,
            // io_uring is not in this build, as in QEMU without CONFIG_LINUX_IO_URING.
            Some(_) => return Err(Error::generic("invalid aio option")),
        };
        let detect_zeroes = match o.take("detect-zeroes") {
            None => BlockdevDetectZeroesOptions::Off,
            Some(v) => match v.as_str() {
                "off" => BlockdevDetectZeroesOptions::Off,
                "on" => BlockdevDetectZeroesOptions::On,
                "unmap" => BlockdevDetectZeroesOptions::Unmap,
                _ => return Err(Error::generic(format!("invalid parameter value: {v}"))),
            },
        };
        let throttled =
            o.0.iter()
                .any(|(k, v)| k.starts_with("throttling.") && k != "throttling.group" && v != "0");
        o.0.retain(|(k, _)| !is_common_opt(k));
        let format = o.take("format");
        if format.is_some() && o.has("driver") {
            return Err(Error::generic("Cannot specify both 'driver' and 'format'"));
        }
        let werror =
            werror.map_or(Ok(BlockdevOnError::Enospc), |v| parse_error_action(&v, false))?;
        let rerror =
            rerror.map_or(Ok(BlockdevOnError::Report), |v| parse_error_action(&v, true))?;
        if snapshot {
            return Err(Error::generic("snapshot=on is not supported yet"));
        }
        if copy_on_read {
            return Err(Error::generic("copy-on-read=on is not supported yet"));
        }
        if throttled {
            return Err(Error::generic("I/O throttling is not supported yet"));
        }

        let filename = filename.filter(|f| !f.is_empty());
        let (blk, node_name, blockdev) = if filename.is_none() && format.is_none() && o.0.is_empty()
        {
            // No medium.
            let blk = BlockBackend::new_empty(Some(id.clone()), 0, BLK_PERM_ALL, read_only);
            (blk, None, None)
        } else {
            let probed = format.is_none() && !o.has("driver");
            let opts = self.drive_options(o, filename, format, read_only, native_aio)?;
            let ctx = OpenCtx {
                root: false,
                inherit: Inherited { auto_read_only: true, ..Inherited::default() },
                probed,
                detect_zeroes: Some(detect_zeroes),
            };
            let (node, w) = self.open_nodes(opts.clone(), ctx)?;
            warnings.extend(w);
            let name = node.name.clone();
            // blk_new_open(): without BDRV_O_RDWR in the flags only consistent read.
            let blk = BlockBackend::with_node(
                Some(id.clone()),
                node,
                BLK_PERM_CONSISTENT_READ,
                BLK_PERM_ALL,
            )?;
            (blk, Some(name), Some(opts))
        };
        blk.set_enable_write_cache(!writethrough);
        let blk = Arc::new(blk);
        self.monitor_add_blk(&id, blk)?;

        let info = DriveInfo {
            id,
            interface: ty,
            bus: bus_id,
            unit: unit_id,
            media_cd: media_cd
                && matches!(
                    ty,
                    BlockInterfaceType::Ide
                        | BlockInterfaceType::Scsi
                        | BlockInterfaceType::Xen
                        | BlockInterfaceType::None
                ),
            read_only,
            copy_on_read,
            write_cache: !writethrough,
            rerror,
            werror,
            device,
            node_name,
            blockdev,
            warnings,
        };
        self.drives.lock().unwrap().push(info.clone());
        Ok(info)
    }

    /// The node options `bdrv_open()` gets from `blockdev_init()`: the format node, or the
    /// protocol node when `driver` names one, with the file name in the right place.
    fn drive_options(
        &self,
        o: DriveOpts,
        filename: Option<String>,
        format: Option<String>,
        read_only: bool,
        native_aio: bool,
    ) -> Result<BlockdevOptions> {
        let mut d = QDict::new();
        for (k, v) in &o.0 {
            put_path(&mut d, k, v)?;
        }
        // blockdev_init() makes these the defaults rather than bdrv_open()'s.
        let cache = match d.get_mut("cache").and_then(QValue::as_dict_mut) {
            Some(c) => c,
            None => {
                d.put("cache", QDict::new());
                d.get_mut("cache").and_then(QValue::as_dict_mut).expect("just put")
            }
        };
        for k in ["direct", "no-flush"] {
            if !cache.contains_key(k) {
                cache.put(k, "off");
            }
        }
        d.put("read-only", if read_only { "on" } else { "off" });
        if !d.contains_key("auto-read-only") {
            d.put("auto-read-only", "on");
        }
        let driver = match (format, d.get_str("driver")) {
            (Some(f), _) => f,
            (None, Some(drv)) => drv.to_string(),
            // Probed, only raw is known so far. The open code checks the guess.
            (None, None) => "raw".to_string(),
        };
        d.put("driver", driver.as_str());
        if is_protocol(&driver) {
            if let Some(f) = filename {
                d.put("filename", f);
            }
            if native_aio && driver == "file" && !d.contains_key("aio") {
                d.put("aio", "native");
            }
        } else if let Some(f) = filename {
            if !d.contains_key("file") {
                d.put("file", QDict::new());
            }
            let Some(file) = d.get_mut("file").and_then(QValue::as_dict_mut) else {
                return Err(Error::generic("Cannot specify both 'file' and 'file.*' options"));
            };
            if !file.contains_key("driver") {
                file.put("driver", "file");
            }
            file.put("filename", f);
            if native_aio && file.get_str("driver") == Some("file") && !file.contains_key("aio") {
                file.put("aio", "native");
            }
        } else if !d.contains_key("file") {
            return Err(Error::generic("A block device must be specified for \"file\""));
        }
        let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
        let mut opts = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut opts)?;
        Ok(opts)
    }

    /// `drive_get()`: the drive at `bus` and `unit` of an interface.
    pub fn drive_get(&self, ty: BlockInterfaceType, bus: u32, unit: u32) -> Option<DriveInfo> {
        self.drives
            .lock()
            .unwrap()
            .iter()
            .find(|d| d.interface == ty && d.bus == bus && d.unit == unit)
            .cloned()
    }

    /// `drive_get_by_index()`: the drive at `index` counted the way `index=` counts.
    pub fn drive_get_by_index(&self, ty: BlockInterfaceType, index: u32) -> Option<DriveInfo> {
        let max = ty.max_devs();
        let (bus, unit) = match index.checked_div(max) {
            Some(bus) => (bus, index % max),
            None => (0, index),
        };
        self.drive_get(ty, bus, unit)
    }

    /// Every `-drive`, in the order they were added.
    pub fn drives(&self) -> Vec<DriveInfo> {
        self.drives.lock().unwrap().clone()
    }
}

/// Whether `driver` is a protocol driver that takes a `filename` itself.
fn is_protocol(driver: &str) -> bool {
    matches!(driver, "file" | "host_device" | "host_cdrom" | "null-co" | "null-aio")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_modes() {
        assert_eq!(parse_cache_mode("none"), Some((true, true, false)));
        assert_eq!(parse_cache_mode("unsafe"), Some((true, false, true)));
        assert_eq!(parse_cache_mode("directsync"), Some((false, true, false)));
        assert_eq!(parse_cache_mode("bogus"), None);
    }

    #[test]
    fn legacy_errors() {
        let g = BlockGraph::new();
        let ide = BlockInterfaceType::Ide;
        let err = |p: &str| g.drive_new(p, ide).unwrap_err().message().to_string();
        assert_eq!(err("cache=bogus"), "invalid cache option");
        assert_eq!(err("media=tape"), "'tape' invalid media");
        assert_eq!(err("if=usb"), "unsupported bus type 'usb'");
        assert_eq!(err("index=1,unit=0"), "index cannot be used with bus and unit");
        assert_eq!(err("unit=2"), "unit 2 too big (max is 1)");
        assert_eq!(err("if=floppy,werror=stop"), "werror is not supported by this bus type");
        assert_eq!(err("rerror=enospc"), "'enospc' invalid read error action");
        assert_eq!(
            err("readonly=on,read-only=on"),
            "'read-only' and its alias 'readonly' can't \
                                                      be used at the same time"
        );
        assert_eq!(err("aio=io_uring,file=x"), "invalid aio option");
        assert_eq!(err("format=raw,driver=raw"), "Cannot specify both 'driver' and 'format'");
        assert_eq!(err("bus=x"), "Parameter 'bus' expects a number");
    }

    #[test]
    fn empty_drives_and_ids() {
        let g = BlockGraph::new();
        let d = g.drive_new("if=ide,media=cdrom", BlockInterfaceType::Ide).unwrap();
        assert_eq!((d.id.as_str(), d.bus, d.unit), ("ide0-cd0", 0, 0));
        assert!(d.media_cd && d.read_only && d.node_name.is_none());
        let d = g.drive_new("if=ide", BlockInterfaceType::Ide).unwrap();
        assert_eq!(d.id, "ide0-hd1");
        let d = g.drive_new("index=2", BlockInterfaceType::Ide).unwrap();
        assert_eq!((d.id.as_str(), d.bus, d.unit), ("ide1-hd0", 1, 0));
        let e = g.drive_new("index=2", BlockInterfaceType::Ide).unwrap_err();
        assert_eq!(e.message(), "drive with bus=1, unit=0 (index=2) exists");
        let d = g.drive_new("if=virtio", BlockInterfaceType::Ide).unwrap();
        assert_eq!((d.id.as_str(), d.device), ("virtio0", Some("virtio-blk")));
        let d = g.drive_new("if=none,id=cd,readonly=on,copy-on-read=on", BlockInterfaceType::Ide);
        let d = d.unwrap();
        assert_eq!(d.warnings, ["warning: disabling copy-on-read on read-only drive"]);
        assert!(!d.copy_on_read);
        let blk = g.backend("cd").unwrap();
        assert!(!blk.is_inserted() && blk.is_read_only());
        let e = g.drive_new("if=none,id=cd", BlockInterfaceType::Ide).unwrap_err();
        assert_eq!(e.message(), "Device with id 'cd' already exists");
    }
}
