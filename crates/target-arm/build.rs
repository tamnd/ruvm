// SPDX-License-Identifier: GPL-2.0-or-later

//! Generates the A64 decoders from QEMU's a64.decode and sve.decode, and the AArch32 ones from
//! vfp.decode, vfp-uncond.decode, a32.decode, a32-uncond.decode, t32.decode and t16.decode,
//! run with the options target/arm/tcg's meson.build passes (`--static-decode=disas_a64`,
//! `--decode=disas_sve`, `--decode=disas_vfp`, `--decode=disas_vfp_uncond`,
//! `--static-decode=disas_a32`, `--static-decode=disas_a32_uncond`,
//! `--static-decode=disas_t32` and `--static-decode=disas_t16 --insnwidth=16`).
//!
//! ruvm-decode makes every `trans_` method a required trait method. Not every pattern is
//! implemented, so the build script gives every `trans_` declaration a default body
//! that returns false, which the decoder turns into UNDEF. Implementing a method that is not in
//! the file is still a compile error, so a typo in a pattern name cannot slip through.
//!
//! In C the four AArch32 decoders call the same `trans_` functions (T16 `ADD_rrri` is the A32
//! one). Here each trait extends the one before it, in the order A32, A32 unconditional, T32,
//! T16, and leaves out the methods an earlier trait has, so that the generated code reaches
//! the one implementation. The argument sets a file marks `!extern` come from the earlier
//! modules.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ruvm_decode::{Invocation, generate_files};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("out dir"));
    println!("cargo::rerun-if-changed=build.rs");
    for (name, opt, out) in [
        ("a64.decode", "--static-decode=disas_a64", "a64_decode.rs"),
        ("sve.decode", "--decode=disas_sve", "sve_decode.rs"),
        ("vfp.decode", "--decode=disas_vfp", "vfp_decode.rs"),
        ("vfp-uncond.decode", "--decode=disas_vfp_uncond", "vfp_uncond_decode.rs"),
    ] {
        let rust = generate(&manifest, name, &[opt]);
        write_if_changed(&dir.join(out), &default_trans(&rust));
    }

    let mut names = HashSet::new();
    let mut base: Option<&str> = None;
    for (name, opts, tr, out) in [
        ("a32.decode", &["--static-decode=disas_a32"][..], "DisasA32", "a32_decode.rs"),
        (
            "a32-uncond.decode",
            &["--static-decode=disas_a32_uncond"][..],
            "DisasA32Uncond",
            "a32_uncond_decode.rs",
        ),
        ("t32.decode", &["--static-decode=disas_t32"][..], "DisasT32", "t32_decode.rs"),
        (
            "t16.decode",
            &["--static-decode=disas_t16", "--insnwidth=16"][..],
            "DisasT16",
            "t16_decode.rs",
        ),
    ] {
        let mut rust = generate(&manifest, name, opts);
        if let Some(b) = base {
            rust = extend(&rust, tr, b, &names);
        }
        names.extend(method_names(&rust));
        base = Some(tr);
        write_if_changed(&dir.join(out), &default_trans(&rust));
    }
}

fn generate(manifest: &Path, name: &str, opts: &[&str]) -> String {
    let decode = manifest.join("../../vendor-qemu/decode/arm/tcg").join(name);
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

/// Make the trait `name` of a generated decoder extend `base`, without the methods `base`
/// and the traits it extends already have.
fn extend(rust: &str, name: &str, base: &str, names: &HashSet<String>) -> String {
    let decl = format!("pub trait {name} {{");
    assert!(rust.contains(&decl), "no {decl} in the generated decoder");
    let mut out = String::with_capacity(rust.len());
    for line in rust.lines() {
        if line == decl {
            out.push_str(&format!("pub trait {name}: {base} {{\n"));
            continue;
        }
        if method_name(line).is_some_and(|n| names.contains(n)) {
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
