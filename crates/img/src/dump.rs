// SPDX-License-Identifier: GPL-2.0-or-later

//! The human readable dumps of block/qapi.c that `qemu-img info` and `qemu-img snapshot -l`
//! print: `bdrv_node_info_dump()`, `bdrv_snapshot_dump()` and `dump_qobject()`.

use ruvm_qapi::cutils::{format_g, size_to_str};
use ruvm_qapi::types::{BlockGraphInfo, BlockLimitsInfo, ImageInfoSpecific, SnapshotInfo};
use ruvm_qapi::visit::{QObjectOutputVisitor, Visit};
use ruvm_qapi::{QDict, QValue, json};

use crate::common::localtime;

/// A QAPI value as the output visitor turns it into a QObject.
pub(crate) fn to_qobject<T: Visit + Clone>(value: &T) -> QValue {
    let mut v = QObjectOutputVisitor::new();
    let mut value = value.clone();
    T::visit(&mut v, None, &mut value).expect("output visits do not fail");
    v.complete()
}

/// `qobject_to_json_pretty()` of a QAPI value, the way the `--output=json` dumps print it.
pub(crate) fn to_json_pretty<T: Visit + Clone>(value: &T) -> String {
    json::to_string(&to_qobject(value), true)
}

/// `bdrv_snapshot_dump()`: one row, or the header with `None`, without the newline.
pub(crate) fn snapshot_dump(sn: Option<&SnapshotInfo>) -> String {
    let Some(sn) = sn else {
        return format!(
            "{:<7} {:<16} {:>8} {:>19} {:>15} {:>10}",
            "ID", "TAG", "VM_SIZE", "DATE", "VM_CLOCK", "ICOUNT"
        );
    };
    let (y, mo, d, h, mi, s) = localtime(sn.date_sec);
    let date = format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}");
    let nsec = sn.vm_clock_sec as u64 * 1_000_000_000 + sn.vm_clock_nsec as u64;
    let secs = nsec / 1_000_000_000;
    let clock = format!(
        "{:04}:{:02}:{:02}.{:03}",
        (secs / 3600) as i32,
        ((secs / 60) % 60) as i32,
        (secs % 60) as i32,
        ((nsec / 1_000_000) % 1000) as i32
    );
    let icount = match sn.icount {
        Some(i) => i.to_string(),
        None => "--".to_string(),
    };
    format!(
        "{:<7} {:<16} {:>8} {:>19} {:>15} {:>10}",
        sn.id,
        sn.name,
        size_to_str(sn.vm_state_size as u64),
        date,
        clock,
        icount
    )
}

/// `dump_qobject()`.
fn dump_qobject(out: &mut String, comp_indent: usize, obj: &QValue) {
    match obj {
        QValue::Int(i) => out.push_str(&i.to_string()),
        QValue::Uint(u) => out.push_str(&u.to_string()),
        QValue::Double(d) => out.push_str(&format_g(*d, 17)),
        QValue::Str(s) => out.push_str(s),
        QValue::Dict(d) => dump_qdict(out, comp_indent, d),
        QValue::List(l) => dump_qlist(out, comp_indent, l),
        QValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        QValue::Null => {}
    }
}

fn is_composite(v: &QValue) -> bool {
    matches!(v, QValue::Dict(_) | QValue::List(_))
}

/// `dump_qlist()`.
fn dump_qlist(out: &mut String, indentation: usize, list: &[QValue]) {
    for (i, v) in list.iter().enumerate() {
        let composite = is_composite(v);
        out.push_str(&format!(
            "{:w$}[{i}]:{}",
            "",
            if composite { '\n' } else { ' ' },
            w = indentation * 4
        ));
        dump_qobject(out, indentation + 1, v);
        if !composite {
            out.push('\n');
        }
    }
}

/// `dump_qdict()`: dashes in the keys become spaces.
fn dump_qdict(out: &mut String, indentation: usize, dict: &QDict) {
    for (key, v) in dict.iter() {
        let composite = is_composite(v);
        out.push_str(&format!(
            "{:w$}{}:{}",
            "",
            key.replace('-', " "),
            if composite { '\n' } else { ' ' },
            w = indentation * 4
        ));
        dump_qobject(out, indentation + 1, v);
        if !composite {
            out.push('\n');
        }
    }
}

/// `qobject_is_empty_dump()`.
fn is_empty_dump(obj: &QValue) -> bool {
    match obj {
        QValue::Dict(d) => d.is_empty(),
        QValue::List(l) => l.is_empty(),
        _ => false,
    }
}

/// `bdrv_image_info_specific_dump()`.
pub(crate) fn image_info_specific_dump(
    out: &mut String,
    info: &ImageInfoSpecific,
    prefix: Option<&str>,
    indentation: usize,
) {
    let obj = to_qobject(info);
    let data = obj.as_dict().and_then(|d| d.get("data")).cloned().unwrap_or_default();
    if !is_empty_dump(&data) {
        if let Some(p) = prefix {
            out.push_str(&format!("{:w$}{p}", "", w = indentation * 4));
        }
        dump_qobject(out, indentation + 1, &data);
    }
}

/// `bdrv_image_info_limits_dump()`.
fn limits_dump(out: &mut String, limits: &BlockLimitsInfo, prefix: &str, indentation: usize) {
    let obj = to_qobject(limits);
    if !is_empty_dump(&obj) {
        out.push_str(&format!("{:w$}{prefix}", "", w = indentation * 4));
        dump_qobject(out, indentation + 1, &obj);
    }
}

/// `bdrv_node_info_dump()`.
pub(crate) fn node_info_dump(
    out: &mut String,
    info: &BlockGraphInfo,
    indentation: usize,
    protocol: bool,
) {
    let ind = " ".repeat(indentation * 4);
    let protocol = protocol && indentation != 0;
    let dsize = match info.actual_size {
        Some(s) => size_to_str(s as u64),
        None => "unavailable".to_string(),
    };
    out.push_str(&format!(
        "{ind}{}: {}\n{ind}{}: {}\n{ind}{}: {} ({} bytes)\n{ind}disk size: {dsize}\n",
        if protocol { "filename" } else { "image" },
        info.filename,
        if protocol { "protocol type" } else { "file format" },
        info.format,
        if protocol { "file length" } else { "virtual size" },
        size_to_str(info.virtual_size as u64),
        info.virtual_size
    ));
    if info.encrypted == Some(true) {
        out.push_str(&format!("{ind}encrypted: yes\n"));
    }
    if let Some(c) = info.cluster_size {
        out.push_str(&format!("{ind}cluster_size: {c}\n"));
    }
    if info.dirty_flag == Some(true) {
        out.push_str(&format!("{ind}cleanly shut down: no\n"));
    }
    if let Some(b) = &info.backing_filename {
        out.push_str(&format!("{ind}backing file: {b}"));
        match &info.full_backing_filename {
            None => out.push_str(" (cannot determine actual path)"),
            Some(f) if f != b => out.push_str(&format!(" (actual path: {f})")),
            Some(_) => {}
        }
        out.push('\n');
        if let Some(f) = &info.backing_filename_format {
            out.push_str(&format!("{ind}backing file format: {f}\n"));
        }
    }
    if let Some(l) = &info.limits {
        limits_dump(out, l, "Block limits:\n", indentation);
    }
    if let Some(sns) = &info.snapshots {
        out.push_str(&format!("{ind}Snapshot list:\n{ind}{}\n", snapshot_dump(None)));
        for sn in sns {
            out.push_str(&format!("{ind}{}\n", snapshot_dump(Some(sn))));
        }
    }
    if let Some(fs) = &info.format_specific {
        image_info_specific_dump(out, fs, Some("Format specific information:\n"), indentation);
    }
}

/// `dump_human_image_info()`.
pub(crate) fn human_image_info(
    out: &mut String,
    info: &BlockGraphInfo,
    indentation: usize,
    path: &str,
) {
    node_info_dump(out, info, indentation, info.children.is_empty());
    for child in &info.children {
        out.push_str(&format!(
            "{:w$}Child node '{path}{}':\n",
            "",
            child.name,
            w = indentation * 4
        ));
        let child_path = format!("{path}{}/", child.name);
        human_image_info(out, &child.info, indentation + 1, &child_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_header() {
        assert_eq!(
            snapshot_dump(None),
            "ID      TAG               VM_SIZE                DATE        VM_CLOCK     ICOUNT"
        );
    }

    #[test]
    fn qdict_dump() {
        let mut d = QDict::new();
        d.put("a-b", 1i64);
        let mut out = String::new();
        dump_qobject(&mut out, 1, &QValue::Dict(d));
        assert_eq!(out, "    a b: 1\n");
    }
}
