// SPDX-License-Identifier: GPL-2.0-or-later

//! Builds the `query-qmp-schema` reply and the Rust types from the vendored QAPI schema.
//!
//! The QObject, JSON and number formatting code is shared with the library through `#[path]`, so the reply is
//! printed by the same writer QMP uses and comes out byte for byte as QEMU prints it.

use std::path::Path;

#[allow(dead_code, unreachable_pub)]
#[path = "src/cutils.rs"]
mod cutils;
#[allow(dead_code, unreachable_pub)]
#[path = "src/json.rs"]
mod json;
#[allow(dead_code, unreachable_pub)]
#[path = "src/qvalue.rs"]
mod qvalue;

use qvalue::{QDict, QValue};
use ruvm_qapi_gen::config::{self, Config};
use ruvm_qapi_gen::{Lit, Schema, introspect};

fn to_qvalue(l: &Lit) -> QValue {
    match l {
        Lit::Null => QValue::Null,
        Lit::Bool(b) => QValue::Bool(*b),
        Lit::Str(s) => QValue::str(s),
        Lit::List(items) => QValue::List(items.iter().map(|a| to_qvalue(&a.value)).collect()),
        Lit::Dict(d) => {
            // QEMU's C literal lists keys sorted, and qobject_from_qlit() inserts them in that order.
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

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let qapi = Path::new(&manifest).join("../../vendor-qemu/qapi");
    println!("cargo::rerun-if-changed={}", qapi.display());
    let config = Config::from_cargo_env();

    let schema = Schema::load(&qapi.join("qapi-schema.json")).unwrap_or_else(|e| panic!("{e}"));
    let trees = introspect(&schema, false);
    for sym in introspect::symbols(&trees) {
        if config::rule(&sym).is_none() {
            panic!(
                "the QAPI schema uses condition {sym}, which crates/qapi-gen/src/config.rs does not know about"
            );
        }
    }
    let is_set = |sym: &str| config.is_set(sym);
    let list = introspect::resolve(&trees, &is_set);
    let json = QValue::List(list.iter().map(to_qvalue).collect()).to_json();
    let out =
        Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("qmp-schema.json");
    std::fs::write(out, json).expect("write the schema");

    let types = ruvm_qapi_gen::rust::gen_types(&schema, &is_set);
    let out =
        Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("qapi-types.rs");
    std::fs::write(out, types).expect("write the types");
    let dir = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).to_path_buf();
    let commands = ruvm_qapi_gen::rust::gen_commands(&schema, &is_set);
    std::fs::write(dir.join("qapi-commands.rs"), commands).expect("write the commands");
    let events = ruvm_qapi_gen::rust::gen_events(&schema, &is_set);
    std::fs::write(dir.join("qapi-events.rs"), events).expect("write the events");
}
