// SPDX-License-Identifier: GPL-2.0-or-later

//! Checks introspection against the reply of a real QEMU.
//!
//! tests/data/qmp-schema-macos-homebrew.json is what Homebrew's qemu-system-x86_64 11.1 on macOS
//! returns for `query-qmp-schema` with `-machine none`, minus the `{"return": ...}` around it.

use std::path::Path;

use ruvm_qapi::{QDict, QValue};
use ruvm_qapi_gen::{Lit, Schema, introspect};

/// The build conditions Homebrew's QEMU was configured with.
const HOMEBREW_MACOS: &[&str] = &[
    "CONFIG_AUDIO_COREAUDIO",
    "CONFIG_COCOA",
    "CONFIG_CURSES",
    "CONFIG_DBUS_DISPLAY",
    "CONFIG_FDT",
    "CONFIG_PIXMAN",
    "CONFIG_POSIX",
    "CONFIG_REPLICATION",
    "CONFIG_SPICE_PROTOCOL",
    "CONFIG_TCG",
    "CONFIG_TPM",
    "CONFIG_VMNET",
    "CONFIG_VNC",
    "CONFIG_ZSTD",
    "HAVE_CHARDEV_SERIAL",
    "HAVE_HOST_BLOCK_DEVICE",
    "HAVE_TCP_KEEPCNT",
    "HAVE_TCP_KEEPIDLE",
    "HAVE_TCP_KEEPINTVL",
];

fn to_qvalue(l: &Lit) -> QValue {
    match l {
        Lit::Null => QValue::Null,
        Lit::Bool(b) => QValue::Bool(*b),
        Lit::Str(s) => QValue::str(s),
        Lit::List(items) => QValue::List(items.iter().map(|a| to_qvalue(&a.value)).collect()),
        Lit::Dict(d) => {
            let mut d: Vec<_> = d.iter().collect();
            d.sort_by(|a, b| a.0.cmp(&b.0));
            let mut q = QDict::new();
            for (k, v) in d {
                q.put(k.clone(), to_qvalue(v));
            }
            QValue::Dict(q)
        }
    }
}

fn schema() -> Schema {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor-qemu/qapi/qapi-schema.json");
    Schema::load(&root).unwrap_or_else(|e| panic!("{e}"))
}

#[test]
fn matches_homebrew_qemu_byte_for_byte() {
    let trees = introspect(&schema(), false);
    let list = introspect::resolve(&trees, &|s| HOMEBREW_MACOS.contains(&s));
    let got = QValue::List(list.iter().map(to_qvalue).collect()).to_json();
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/qmp-schema-macos-homebrew.json");
    let want = std::fs::read_to_string(path).unwrap();
    let want = want.trim_end();
    if got != want {
        let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
        let from = at.saturating_sub(200);
        panic!(
            "introspection differs at byte {at}\n ruvm: {}\n qemu: {}",
            &got[from..(at + 200).min(got.len())],
            &want[from..(at + 200).min(want.len())]
        );
    }
}

#[test]
fn built_in_reply_parses_and_has_the_basics() {
    let v = ruvm_qapi::qmp_schema();
    let list = v.as_list().unwrap();
    let names: Vec<&str> = list.iter().filter_map(|e| e.as_dict()?.get_str("name")).collect();
    for n in ["qmp_capabilities", "query-qmp-schema", "query-version", "SHUTDOWN", "str", "int"] {
        assert!(names.contains(&n), "{n} missing");
    }
    assert_eq!(v.to_json(), ruvm_qapi::QMP_SCHEMA_JSON);
}

#[test]
fn numbering_does_not_depend_on_conditions() {
    let trees = introspect(&schema(), false);
    let none = introspect::resolve(&trees, &|_| false);
    let all = introspect::resolve(&trees, &|_| true);
    let find = |l: &[Lit], name: &str| {
        l.iter()
            .find(|e| matches!(e, Lit::Dict(d) if d.iter().any(|(k, v)| k == "name" && *v == Lit::Str(name.into()))))
            .cloned()
    };
    // query-version comes after conditional entities, and its types keep their numbers.
    assert_eq!(find(&none, "query-version"), find(&all, "query-version"));
}

#[test]
fn unmasked_names() {
    let trees = introspect(&schema(), true);
    let list = introspect::resolve(&trees, &|_| false);
    let qv = list
        .iter()
        .map(to_qvalue)
        .find(|v| v.as_dict().and_then(|d| d.get_str("name")) == Some("query-version"))
        .unwrap();
    assert_eq!(qv.as_dict().unwrap().get_str("ret-type"), Some("VersionInfo"));
}
