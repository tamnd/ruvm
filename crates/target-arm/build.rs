// SPDX-License-Identifier: GPL-2.0-or-later

//! Generates the A64 decoder from QEMU's a64.decode, run with the options target/arm/tcg's
//! meson.build passes (`--static-decode=disas_a64`).
//!
//! ruvm-decode makes every `trans_` method a required trait method. This slice implements only
//! the base integer patterns, so the build script gives every `trans_` declaration a default body
//! that returns false, which the decoder turns into UNDEF. Implementing a method that is not in
//! the file is still a compile error, so a typo in a pattern name cannot slip through.

use std::path::{Path, PathBuf};

use ruvm_decode::{Invocation, generate_files};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let decode = manifest.join("../../vendor-qemu/decode/arm/tcg/a64.decode");
    println!("cargo::rerun-if-changed={}", decode.display());
    println!("cargo::rerun-if-changed=build.rs");
    let text =
        std::fs::read_to_string(&decode).unwrap_or_else(|e| panic!("{}: {e}", decode.display()));
    let name = "a64.decode";
    let inv = Invocation::parse(&["--static-decode=disas_a64", name]).expect("decode options");
    let rust = generate_files(&[(name, &text)], &inv.options)
        .unwrap_or_else(|e| panic!("{}", e.render(false)));
    let mut out = String::with_capacity(rust.len() + 4096);
    for line in rust.lines() {
        let t = line.trim_start();
        if t.starts_with("fn trans_") && t.ends_with("-> bool;") {
            let indent = &line[..line.len() - t.len()];
            out.push_str(indent);
            out.push_str("#[allow(unused_variables)]\n");
            out.push_str(&line[..line.len() - 1]);
            out.push_str(" {\n");
            out.push_str(indent);
            out.push_str("    false\n");
            out.push_str(indent);
            out.push_str("}\n");
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    let dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    write_if_changed(&dir.join("a64_decode.rs"), &out);
}

fn write_if_changed(path: &Path, text: &str) {
    if std::fs::read_to_string(path).is_ok_and(|old| old == text) {
        return;
    }
    std::fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}
