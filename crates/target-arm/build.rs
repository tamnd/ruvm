// SPDX-License-Identifier: GPL-2.0-or-later

//! Generates the A64 decoders from QEMU's a64.decode and sve.decode, run with the options
//! target/arm/tcg's meson.build passes (`--static-decode=disas_a64` and
//! `--decode=disas_sve`).
//!
//! ruvm-decode makes every `trans_` method a required trait method. Not every pattern is
//! implemented, so the build script gives every `trans_` declaration a default body
//! that returns false, which the decoder turns into UNDEF. Implementing a method that is not in
//! the file is still a compile error, so a typo in a pattern name cannot slip through.

use std::path::{Path, PathBuf};

use ruvm_decode::{Invocation, generate_files};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    println!("cargo::rerun-if-changed=build.rs");
    for (name, opt, out) in [
        ("a64.decode", "--static-decode=disas_a64", "a64_decode.rs"),
        ("sve.decode", "--decode=disas_sve", "sve_decode.rs"),
    ] {
        let decode = manifest.join("../../vendor-qemu/decode/arm/tcg").join(name);
        println!("cargo::rerun-if-changed={}", decode.display());
        let text = std::fs::read_to_string(&decode)
            .unwrap_or_else(|e| panic!("{}: {e}", decode.display()));
        let inv = Invocation::parse(&[opt, name]).expect("decode options");
        let rust = generate_files(&[(name, &text)], &inv.options)
            .unwrap_or_else(|e| panic!("{}", e.render(false)));
        write_if_changed(&dir.join(out), &default_trans(&rust));
    }
}

/// Give every `trans_` declaration a default body that returns false.
fn default_trans(rust: &str) -> String {
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
    out
}

fn write_if_changed(path: &Path, text: &str) {
    if std::fs::read_to_string(path).is_ok_and(|old| old == text) {
        return;
    }
    std::fs::write(path, text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}
