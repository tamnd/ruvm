// SPDX-License-Identifier: GPL-2.0-or-later

//! QEMU's tests/decode, run the way its meson.build runs them. The .decode files in
//! tests/decode are copied unchanged from QEMU 11.1 (they carry QEMU's LGPL notice and are test
//! data only). Every `err_` file must fail with exactly the message decodetree.py prints, which
//! is recorded below, and every `succ_` file must generate code that compiles.

use std::path::{Path, PathBuf};
use std::process::Command;

use ruvm_decode::{Options, generate_files};

/// What `decodetree.py --output-null --test-for-error <file>` prints to stderr for each file,
/// captured from QEMU 11.1.
const ERRORS: &[(&str, &str)] = &[
    ("err_argset1.decode", "err_argset1.decode:5: detected: duplicate argument \"a\""),
    ("err_argset2.decode", "err_argset2.decode:5: detected: invalid argument set token \"0e\""),
    ("err_field1.decode", "err_field1.decode:5: detected: invalid field token \"asdf\""),
    (
        "err_field10.decode",
        "err_field10.decode:7: detected: format refers to undefined field field2",
    ),
    ("err_field2.decode", "err_field2.decode:5: detected: field 0:33 too large"),
    ("err_field3.decode", "err_field3.decode:5: detected: field 31:2 too large"),
    ("err_field4.decode", "err_field4.decode:6: detected: duplicate field field"),
    ("err_field5.decode", "err_field5.decode:5: detected: duplicate function"),
    ("err_field6.decode", "err_field6.decode:5: detected: field with no value"),
    (
        "err_field7.decode",
        "err_field7.decode: detected: field definitions form a cycle: field1 => field2",
    ),
    (
        "err_field8.decode",
        "err_field8.decode:8: detected: pattern refers to undefined field field2",
    ),
    (
        "err_field9.decode",
        "err_field9.decode:14: detected: pattern that uses fields defined in format cannot use format that uses fields defined in pattern",
    ),
    ("err_init1.decode", "err_init1.decode:6: detected: field a not initialized"),
    ("err_init2.decode", "err_init2.decode:6: detected: duplicate field a"),
    ("err_init3.decode", "err_init3.decode:7: detected: field a set by format and pattern"),
    ("err_init4.decode", "err_init4.decode:7: detected: field b not initialized"),
    (
        "err_overlap1.decode",
        "err_overlap1.decode:6: detected: fieldmask overlaps fixedmask  (0x00000001 & 0xffffffff)",
    ),
    (
        "err_overlap2.decode",
        "err_overlap2.decode:6: detected: fieldmask overlaps fixedmask  (0x00000001 & 0xffffffff)",
    ),
    (
        "err_overlap3.decode",
        "err_overlap3.decode:6: detected: fieldmask overlaps undefmask  (0x00000001 & 0x000000ff)",
    ),
    (
        "err_overlap4.decode",
        "err_overlap4.decode:6: detected: fixedmask overlaps undefmask  (0xffffffff & 0x00000001)",
    ),
    ("err_overlap5.decode", "err_overlap5.decode:5: detected: field components overlap"),
    (
        "err_overlap6.decode",
        "err_overlap6.decode:6: detected: pattern fixed bits overlap format fixed bits",
    ),
    (
        "err_overlap7.decode",
        "err_overlap7.decode:5: detected: overlapping patterns:\nerr_overlap7.decode:5: insn1 00000000 00000000 00000000 00000000\nerr_overlap7.decode:6: insn2 00000000 00000000 00000000 00000000",
    ),
    ("err_overlap8.decode", "err_overlap8.decode:5: detected: bits left unspecified  (0x00000001)"),
    ("err_overlap9.decode", "err_overlap9.decode:6: detected: bits left unspecified  (0x00000001)"),
    (
        "err_pattern_group_empty.decode",
        "err_pattern_group_empty.decode:5: detected: empty pattern group",
    ),
    (
        "err_pattern_group_ident1.decode",
        "err_pattern_group_ident1.decode:8: detected: indentation  4  !=  2",
    ),
    (
        "err_pattern_group_ident2.decode",
        "err_pattern_group_ident2.decode:10: detected: indentation  0  !=  2",
    ),
    (
        "err_pattern_group_nest1.decode",
        "err_pattern_group_nest1.decode:13: detected: mismatched close brace",
    ),
    (
        "err_pattern_group_nest2.decode",
        "err_pattern_group_nest2.decode:6: detected: missing close brace",
    ),
    (
        "err_pattern_group_nest3.decode",
        "err_pattern_group_nest3.decode:11: detected: overlapping patterns:\nerr_pattern_group_nest3.decode:11: sub1 00000000 00000000 00000000 ........\nerr_pattern_group_nest3.decode:12: sub2 00000000 00000000 ........ ........",
    ),
    (
        "err_pattern_group_overlap1.decode",
        "err_pattern_group_overlap1.decode:1: detected: overlapping patterns:\nerr_pattern_group_overlap1.decode:1: one 00000000 00000000 00000000 00000000\nerr_pattern_group_overlap1.decode:2: group 00000000 00000000 00000000 000000..",
    ),
    ("err_width1.decode", "err_width1.decode:5: detected: definition has 33 bits"),
    ("err_width2.decode", "err_width2.decode:5: detected: definition has 31 bits"),
    ("err_width3.decode", "err_width3.decode:5: detected: field s exceeds insnwidth"),
    ("err_width4.decode", "err_width4.decode:5: detected: definition has 31 bits"),
];

const SUCCESSES: &[&str] = &[
    "succ_argset_type1.decode",
    "succ_function.decode",
    "succ_ident1.decode",
    "succ_infer1.decode",
    "succ_named_field.decode",
    "succ_pattern_group_nest1.decode",
    "succ_pattern_group_nest2.decode",
    "succ_pattern_group_nest3.decode",
    "succ_pattern_group_nest4.decode",
];

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/decode")
}

fn read(name: &str) -> String {
    std::fs::read_to_string(dir().join(name)).unwrap()
}

#[test]
fn every_file_is_listed() {
    let mut found: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".decode"))
        .collect();
    found.sort();
    let mut listed: Vec<String> =
        ERRORS.iter().map(|e| e.0).chain(SUCCESSES.iter().copied()).map(String::from).collect();
    listed.sort();
    assert_eq!(found, listed);
}

#[test]
fn error_files_fail_with_qemus_message() {
    for (name, want) in ERRORS {
        let text = read(name);
        let err = generate_files(&[(name, &text)], &Options::default())
            .expect_err(&format!("{name} should fail"));
        assert_eq!(err.render(true), *want, "{name}");
    }
}

#[test]
fn success_files_generate_code_that_compiles() {
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("decode-succ");
    std::fs::create_dir_all(&out).unwrap();
    let mut root = String::from("#![forbid(unsafe_code)]\n");
    for (i, name) in SUCCESSES.iter().enumerate() {
        let text = read(name);
        let rust = generate_files(&[(name, &text)], &Options::default())
            .unwrap_or_else(|e| panic!("{}", e.render(false)));
        std::fs::write(out.join(format!("m{i}.rs")), rust).unwrap();
        root += &format!("pub mod m{i} {{\n    include!(\"m{i}.rs\");\n}}\n");
    }
    std::fs::write(out.join("lib.rs"), root).unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let res = Command::new(rustc)
        .args(["--crate-type=lib", "--crate-name=decode_succ", "--edition=2024"])
        .args(["--emit=metadata", "-D", "warnings", "-o"])
        .arg(out.join("libdecode_succ.rmeta"))
        .arg(out.join("lib.rs"))
        .output()
        .expect("run rustc");
    assert!(res.status.success(), "{}", String::from_utf8_lossy(&res.stderr));
    let _ = std::fs::remove_dir_all(&out);
}

/// The command line behaves like the script under meson: exit status 0 and the message on
/// stderr for an expected error, exit status 0 and nothing printed for a good file.
#[test]
fn command_line_matches_meson_runs() {
    let bin = env!("CARGO_BIN_EXE_ruvm-decodetree");
    for (name, want) in ERRORS {
        let res = Command::new(bin)
            .current_dir(dir())
            .args(["--output-null", "--test-for-error", name])
            .output()
            .unwrap();
        assert!(res.status.success(), "{name}");
        assert_eq!(String::from_utf8_lossy(&res.stderr), format!("{want}\n"), "{name}");
    }
    for name in SUCCESSES {
        let res =
            Command::new(bin).current_dir(dir()).args(["--output-null", name]).output().unwrap();
        assert!(res.status.success(), "{name}: {}", String::from_utf8_lossy(&res.stderr));
        assert!(res.stdout.is_empty() && res.stderr.is_empty(), "{name}");
        let res = Command::new(bin)
            .current_dir(dir())
            .args(["--output-null", "--test-for-error", name])
            .output()
            .unwrap();
        assert_eq!(res.status.code(), Some(1), "{name}");
    }
}
