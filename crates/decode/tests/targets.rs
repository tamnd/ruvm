// SPDX-License-Identifier: GPL-2.0-or-later

//! Every .decode file under QEMU 11.1's target/ tree, as vendored in vendor-qemu/decode, run with
//! the options its meson.build passes. Each must generate, and all the generated files together
//! must compile as one crate with `-D warnings`, each file in its own module. Files that use an
//! `!extern` argument set import the module generated from the file that defines it, which is
//! how QEMU compiles them too (t32.c includes the a32 decoder, and so on).

use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_decode::{Invocation, generate_files};

/// (file under vendor-qemu/decode, module name, meson options, modules whose sets it uses)
const TARGETS: &[(&str, &str, &[&str], &[&str])] = &[
    ("arm/tcg/a64.decode", "a64", &["--static-decode=disas_a64"], &[]),
    ("arm/tcg/sve.decode", "sve", &["--decode=disas_sve"], &[]),
    ("arm/tcg/sme.decode", "sme", &["--decode=disas_sme"], &[]),
    ("arm/tcg/sme-fa64.decode", "sme_fa64", &["--static-decode=disas_sme_fa64"], &[]),
    ("arm/tcg/neon-shared.decode", "neon_shared", &["--decode=disas_neon_shared"], &[]),
    ("arm/tcg/neon-dp.decode", "neon_dp", &["--decode=disas_neon_dp"], &[]),
    ("arm/tcg/neon-ls.decode", "neon_ls", &["--decode=disas_neon_ls"], &[]),
    ("arm/tcg/vfp.decode", "vfp", &["--decode=disas_vfp"], &[]),
    ("arm/tcg/vfp-uncond.decode", "vfp_uncond", &["--decode=disas_vfp_uncond"], &[]),
    ("arm/tcg/m-nocp.decode", "m_nocp", &["--decode=disas_m_nocp"], &[]),
    ("arm/tcg/mve.decode", "mve", &["--decode=disas_mve"], &[]),
    ("arm/tcg/a32.decode", "a32", &["--static-decode=disas_a32"], &[]),
    ("arm/tcg/a32-uncond.decode", "a32_uncond", &["--static-decode=disas_a32_uncond"], &["a32"]),
    ("arm/tcg/t32.decode", "t32", &["--static-decode=disas_t32"], &["a32", "a32_uncond"]),
    (
        "arm/tcg/t16.decode",
        "t16",
        &["-w", "16", "--static-decode=disas_t16"],
        &["a32", "a32_uncond", "t32"],
    ),
    ("avr/insn.decode", "avr", &["--decode", "decode_insn", "--insnwidth", "16"], &[]),
    ("hppa/insns.decode", "hppa", &[], &[]),
    ("loongarch/insns.decode", "loongarch", &[], &[]),
    ("microblaze/insns.decode", "microblaze", &[], &[]),
    ("or1k/insns.decode", "or1k", &[], &[]),
    ("sparc/insns.decode", "sparc", &[], &[]),
    ("ppc/insn32.decode", "ppc32", &["--static-decode=decode_insn32"], &[]),
    (
        "ppc/insn64.decode",
        "ppc64",
        &["--static-decode=decode_insn64", "--insnwidth=64"],
        &["ppc32"],
    ),
    ("rx/insns.decode", "rx", &["--varinsnwidth", "32"], &[]),
    (
        "riscv/insn16.decode",
        "rv16",
        &["--static-decode=decode_insn16", "--insnwidth=16"],
        &["rv32"],
    ),
    ("riscv/insn32.decode", "rv32", &["--static-decode=decode_insn32"], &[]),
    ("riscv/xthead.decode", "xthead", &["--static-decode=decode_xthead"], &["rv32"]),
    (
        "riscv/XVentanaCondOps.decode",
        "xventana",
        &["--static-decode=decode_XVentanaCodeOps"],
        &["rv32"],
    ),
    ("riscv/xmips.decode", "xmips", &["--static-decode=decode_xmips"], &[]),
    ("riscv/xlrbr.decode", "xlrbr", &["--static-decode=decode_xlrbr"], &["rv32"]),
    ("mips/tcg/rel6.decode", "mips_rel6", &["--decode=decode_isa_rel6"], &[]),
    ("mips/tcg/msa.decode", "mips_msa", &["--decode=decode_ase_msa"], &[]),
    ("mips/tcg/tx79.decode", "mips_tx79", &["--static-decode=decode_tx79"], &[]),
    ("mips/tcg/vr54xx.decode", "mips_vr54xx", &["--decode=decode_ext_vr54xx"], &[]),
    ("mips/tcg/octeon.decode", "mips_octeon", &["--decode=decode_ext_octeon"], &[]),
    ("mips/tcg/lcsr.decode", "mips_lcsr", &["--decode=decode_ase_lcsr"], &[]),
    ("mips/tcg/godson2.decode", "mips_godson2", &["--static-decode=decode_godson2"], &[]),
    (
        "mips/tcg/loong-ext.decode",
        "mips_loong_ext",
        &["--static-decode=decode_loong_ext"],
        &["mips_godson2"],
    ),
];

fn vendor_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../vendor-qemu/decode")
}

fn generate_one(file: &str, args: &[&str]) -> String {
    let path = vendor_dir().join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut argv: Vec<&str> = args.to_vec();
    argv.push(file);
    let inv = Invocation::parse(&argv).unwrap();
    generate_files(&[(file, &text)], &inv.options)
        .unwrap_or_else(|e| panic!("{file}: {}", e.render(false)))
}

#[test]
fn every_target_file_is_listed() {
    let mut found = Vec::new();
    let mut stack = vec![vendor_dir()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "decode") {
                let rel = path.strip_prefix(vendor_dir()).unwrap();
                found.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
    }
    found.sort();
    let mut listed: Vec<String> = TARGETS.iter().map(|t| t.0.to_string()).collect();
    listed.sort();
    assert_eq!(found, listed);
}

#[test]
fn every_target_file_generates_and_compiles() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("decode-targets");
    std::fs::create_dir_all(&dir).unwrap();
    let mut root = String::from("#![forbid(unsafe_code)]\n");
    for (file, module, args, uses) in TARGETS {
        let rust = generate_one(file, args);
        std::fs::write(dir.join(format!("{module}.rs")), rust).unwrap();
        root += &format!("pub mod {module} {{\n");
        for u in *uses {
            root += &format!("    #[allow(unused_imports)]\n    use super::{u}::*;\n");
        }
        root += &format!("    include!(\"{module}.rs\");\n}}\n");
    }
    let lib = dir.join("lib.rs");
    std::fs::write(&lib, root).unwrap();

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(rustc)
        .args(["--crate-type=lib", "--crate-name=decode_targets", "--edition=2024"])
        .args(["--emit=metadata", "-D", "warnings", "-o"])
        .arg(dir.join("libdecode_targets.rmeta"))
        .arg(&lib)
        .output()
        .expect("run rustc");
    assert!(
        out.status.success(),
        "generated code does not compile:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
