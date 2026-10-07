// SPDX-License-Identifier: GPL-2.0-or-later

//! Generates the RISC-V decoders from QEMU's insn32.decode, insn16.decode and xlrbr.decode,
//! run with the options target/riscv/tcg's meson.build passes (`--static-decode=decode_insn32`,
//! `--static-decode=decode_insn16 --insnwidth=16` and `--static-decode=decode_xlrbr`).
//!
//! ruvm-decode makes every `trans_` method a required trait method. Not every pattern is
//! implemented, so the build script gives every `trans_` declaration a default body that
//! returns false, which the decoder turns into an illegal instruction.
//!
//! In C the 16-bit decoder calls the same `trans_` functions as the 32-bit one (`c.addi` is
//! `trans_addi`). Here the 16-bit trait extends the 32-bit one instead, and the methods both
//! files name are left out of the 16-bit trait so that the generated code reaches the 32-bit
//! implementation. The argument sets the 16-bit file marks `!extern` come from the 32-bit
//! module.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ruvm_decode::{Invocation, generate_files};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    println!("cargo::rerun-if-changed=build.rs");
    let gen32 = generate(&manifest, "insn32.decode", &["--static-decode=decode_insn32"]);
    let names32 = method_names(&gen32);
    write_if_changed(&dir.join("insn32.rs"), &default_trans(&gen32));

    let gen16 =
        generate(&manifest, "insn16.decode", &["--static-decode=decode_insn16", "--insnwidth=16"]);
    let gen16 = extend_insn32(&gen16, "DecodeInsn16", &names32);
    write_if_changed(&dir.join("insn16.rs"), &default_trans(&gen16));

    let genx = generate(&manifest, "xlrbr.decode", &["--static-decode=decode_xlrbr"]);
    write_if_changed(&dir.join("xlrbr.rs"), &default_trans(&genx));
}

fn generate(manifest: &Path, name: &str, opts: &[&str]) -> String {
    let decode = manifest.join("../../vendor-qemu/decode/riscv").join(name);
    println!("cargo::rerun-if-changed={}", decode.display());
    let text =
        std::fs::read_to_string(&decode).unwrap_or_else(|e| panic!("{}: {e}", decode.display()));
    let mut args: Vec<&str> = opts.to_vec();
    args.push(name);
    let inv = Invocation::parse(&args).expect("decode options");
    generate_files(&[(name, &text)], &inv.options).unwrap_or_else(|e| panic!("{}", e.render(false)))
}

/// The name of the trait method a generated line declares, if it declares one.
fn method_name(line: &str) -> Option<&str> {
    let t = line.trim_start().strip_prefix("fn ")?;
    if !line.trim_end().ends_with(';') {
        return None;
    }
    t.split('(').next()
}

/// The `trans_` and `!function` methods of a generated trait.
fn method_names(rust: &str) -> HashSet<String> {
    rust.lines().filter_map(method_name).map(str::to_owned).collect()
}

/// Make the trait `name` of a generated decoder extend `DecodeInsn32`, without the methods
/// that trait already has.
fn extend_insn32(rust: &str, name: &str, names32: &HashSet<String>) -> String {
    let decl = format!("pub trait {name} {{");
    assert!(rust.contains(&decl), "no {decl} in the generated decoder");
    let mut out = String::with_capacity(rust.len());
    for line in rust.lines() {
        if line == decl {
            out.push_str(&format!("pub trait {name}: DecodeInsn32 {{\n"));
            continue;
        }
        if method_name(line).is_some_and(|n| names32.contains(n)) {
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
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
