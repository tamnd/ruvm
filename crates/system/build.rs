// SPDX-License-Identifier: GPL-2.0-or-later

//! Generates the option table and the `-help` text from the vendored qemu-options.hx, the way
//! QEMU's build includes qemu-options.def into system/vl.c three times.

use std::fmt::Write as _;
use std::path::Path;

use ruvm_qapi_gen::config::{self, Config};
use ruvm_qapi_gen::hx::{self, OptionsEntry};

/// `QEMU_OPTION_add_fd` becomes `AddFd`. Single letter options differ only in case (`-m` and
/// `-M`, `-s` and `-S`, `-d` and `-D`), so a lower case letter other than `h` becomes
/// `LowerM` and so on, and an upper case one keeps its name.
fn variant(enum_name: &str) -> String {
    let base = enum_name.strip_prefix("QEMU_OPTION_").expect("QEMU_OPTION_ prefix");
    if base.len() == 1 && base != "h" && base.bytes().all(|b| b.is_ascii_lowercase()) {
        return format!("Lower{}", base.to_ascii_uppercase());
    }
    base.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| {
            let lower = w.to_ascii_lowercase();
            let mut c = lower.chars();
            c.next().map(|f| f.to_ascii_uppercase().to_string() + c.as_str()).unwrap_or_default()
        })
        .collect()
}

fn arch_list(arch: &[String]) -> String {
    let names: Vec<String> = arch
        .iter()
        .map(|a| format!("Arch::{}", a.strip_prefix("QEMU_ARCH_").expect("QEMU_ARCH_ prefix")))
        .collect();
    format!("&[{}]", names.join(", "))
}

/// The macros the help strings use. The paths are the ones a QEMU built with the default
/// prefix of /usr/local prints.
fn macros(name: &str) -> Option<String> {
    let text = match name {
        "DEFAULT_NETWORK_SCRIPT" => "/usr/local/etc/qemu-ifup",
        "DEFAULT_NETWORK_DOWN_SCRIPT" => "/usr/local/etc/qemu-ifdown",
        "DEFAULT_BRIDGE_INTERFACE" => "br0",
        "DEFAULT_BRIDGE_HELPER" => "/usr/local/libexec/qemu-bridge-helper",
        "DEFAULT_GDBSTUB_PORT" => "1234",
        _ => return None,
    };
    Some(text.to_string())
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR");
    let path = Path::new(&manifest).join("../../vendor-qemu/hx/qemu-options.hx");
    println!("cargo::rerun-if-changed={}", path.display());
    let text = std::fs::read_to_string(&path).expect("read qemu-options.hx");
    let config = Config::from_cargo_env();
    let is_set = |sym: &str| config::rule(sym).map(|_| config.is_set(sym));
    let entries = hx::parse_options(&text, &is_set, &macros)
        .unwrap_or_else(|e| panic!("qemu-options.hx: {e}"));

    let mut out = String::new();
    let mut variants: Vec<String> = vec!["H".into()];
    for e in &entries {
        if let OptionsEntry::Def(d) = e {
            let v = variant(&d.enum_name);
            if !variants.contains(&v) {
                variants.push(v);
            }
        }
    }
    out.push_str("/// The `QEMU_OPTION_*` enumerators.\n");
    out.push_str("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]\npub enum Opt {\n");
    for v in &variants {
        let _ = writeln!(out, "    {v},");
    }
    out.push_str("}\n\n");

    out.push_str("/// `qemu_options[]`, with the `-h` entry vl.c puts in front of the table.\n");
    out.push_str("pub static QEMU_OPTIONS: &[QemuOption] = &[\n");
    out.push_str(
        "    QemuOption { name: \"h\", has_arg: false, index: Opt::H, arch: &[Arch::ALL] },\n",
    );
    for e in &entries {
        if let OptionsEntry::Def(d) = e {
            let _ = writeln!(
                out,
                "    QemuOption {{ name: {}, has_arg: {}, index: Opt::{}, arch: {} }},",
                hx::rust_str(&d.name),
                d.has_arg,
                variant(&d.enum_name),
                arch_list(&d.arch)
            );
        }
    }
    out.push_str("];\n\n");

    out.push_str("/// What `help()` prints between the usage lines and the key list.\n");
    out.push_str("pub static HELP: &[HelpItem] = &[\n");
    for e in &entries {
        match e {
            OptionsEntry::Heading { text, arch } => {
                let _ = writeln!(
                    out,
                    "    HelpItem {{ text: {}, heading: true, arch: {} }},",
                    hx::rust_str(text),
                    arch_list(arch)
                );
            }
            OptionsEntry::Def(d) => {
                let _ = writeln!(
                    out,
                    "    HelpItem {{ text: {}, heading: false, arch: {} }},",
                    hx::rust_str(&d.help),
                    arch_list(&d.arch)
                );
            }
        }
    }
    out.push_str("];\n");

    let dest = Path::new(&std::env::var("OUT_DIR").expect("cargo sets OUT_DIR")).join("options.rs");
    std::fs::write(dest, out).expect("write options.rs");
}
