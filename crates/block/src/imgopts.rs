// SPDX-License-Identifier: GPL-2.0-or-later

//! What the `.bdrv_co_create_opts` functions of the image formats have in common: taking their
//! `-o` options out of the option dictionary, creating and opening the image file, and turning
//! the options into `BlockdevCreateOptions` the way `qobject_input_visitor_new_flat_confused()`
//! does in QEMU. Also the `QemuOptDesc` shorthand for the `create_opts` lists.

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevCreateOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue};

use crate::backend::BlockBackend;
use crate::graph::BlockGraph;

/// One entry of a `create_opts` list: `opt_desc!(name, Type, help)` or with a default value as
/// the fourth argument.
macro_rules! opt_desc {
    ($name:expr, $ty:ident, $help:expr) => {
        ruvm_qapi::opts::QemuOptDesc::new($name, ruvm_qapi::opts::QemuOptType::$ty).help($help)
    };
    ($name:expr, $ty:ident, $help:expr, $def:expr) => {
        ruvm_qapi::opts::QemuOptDesc::new($name, ruvm_qapi::opts::QemuOptType::$ty)
            .help($help)
            .default_value($def)
    };
}
pub(crate) use opt_desc;

/// Takes `key` out of `options` as a string, whatever type it had.
pub(crate) fn take_str(options: &mut QDict, key: &str) -> Option<String> {
    options.remove(key).map(|v| match v {
        QValue::Str(s) => s,
        QValue::Bool(b) => if b { "on" } else { "off" }.to_string(),
        other => other.to_json(),
    })
}

/// Takes `key` out of `options` as a size, `qemu_opt_get_size_del()`.
pub(crate) fn take_size(options: &mut QDict, key: &str) -> Result<Option<u64>> {
    match options.remove(key) {
        None => Ok(None),
        Some(QValue::Str(s)) => ruvm_qapi::visit::parse_option_size(key, &s).map(Some),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            Error::generic(format!("Invalid parameter type for '{key}', expected: size"))
        }),
    }
}

/// Takes `key` out of `options` as a boolean, `qemu_opt_get_bool_del()`.
pub(crate) fn take_bool(options: &mut QDict, key: &str) -> Result<Option<bool>> {
    match options.remove(key) {
        None => Ok(None),
        Some(QValue::Str(s)) => ruvm_qapi::visit::qapi_bool_parse(key, &s).map(Some),
        Some(QValue::Bool(b)) => Ok(Some(b)),
        Some(_) => {
            Err(Error::generic(format!("Invalid parameter type for '{key}', expected: boolean")))
        }
    }
}

/// Whether `key` is one of `names`, where a name ending in `.` stands for every key with that
/// prefix.
fn matches(names: &[&str], key: &str) -> bool {
    names.iter().any(|n| if n.ends_with('.') { key.starts_with(n) } else { key == *n })
}

/// The start of a format's `.bdrv_co_create_opts`: takes the options called `names` out of
/// `options` (with the `-o` spelling, `backing_file` rather than `backing-file`), creates
/// `filename` with the protocol driver its name picks from what is left (the protocol takes
/// out what it knows), and opens it read-write as `bdrv_co_open(filename, NULL, NULL,
/// BDRV_O_RDWR | BDRV_O_RESIZE | BDRV_O_PROTOCOL)` does.
///
/// Returns the graph the file node is in, a backend holding the node, and the taken options
/// with `renames` applied (`qdict_rename_keys()`) and `driver` and `file` (the node name) put
/// in, ready for [`visit_create_options`] once the driver has adjusted them.
pub(crate) fn create_opts_open(
    filename: &str,
    options: &mut QDict,
    driver: &str,
    names: &[&str],
    renames: &[(&str, &str)],
) -> Result<(BlockGraph, BlockBackend, QDict)> {
    let mut taken = QDict::new();
    let keys: Vec<String> =
        options.keys().filter(|k| matches(names, k)).map(str::to_string).collect();
    for k in keys {
        if let Some(v) = options.remove(&k) {
            let k = renames.iter().find(|(from, _)| *from == k).map_or(k, |(_, to)| to.to_string());
            taken.put(k, v);
        }
    }
    let graph = BlockGraph::new();
    graph.create_file(filename, options)?;
    let blk = graph.open_protocol_blk(filename)?;
    let node = blk.root().expect("a new backend has its node");
    taken.put("driver", driver);
    taken.put("file", node.name.as_str());
    Ok((graph, blk, taken))
}

/// `qobject_input_visitor_new_flat_confused()` and `visit_type_BlockdevCreateOptions()`: the
/// flat string options as typed create options.
pub(crate) fn visit_create_options(qdict: QDict) -> Result<BlockdevCreateOptions> {
    let crumpled = crate::open::crumple(qdict)?;
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(crumpled));
    let mut opts = BlockdevCreateOptions::default();
    BlockdevCreateOptions::visit(&mut v, None, &mut opts)?;
    Ok(opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_helpers() {
        let mut o = QDict::new();
        o.put("size", "1M");
        o.put("lazy", "on");
        o.put("name", "x");
        assert_eq!(take_size(&mut o, "size").unwrap(), Some(1 << 20));
        assert_eq!(take_bool(&mut o, "lazy").unwrap(), Some(true));
        assert_eq!(take_str(&mut o, "name").as_deref(), Some("x"));
        assert!(o.is_empty());
        assert!(matches(&["encrypt."], "encrypt.format"));
        assert!(!matches(&["encrypt"], "encrypt.format"));
    }
}
