// SPDX-License-Identifier: GPL-2.0-or-later

//! Export the `qemu_plugin_*` symbols from this crate's test binaries, so that the plugins the
//! tests load can find them. A program that loads plugins needs the same flag; see the crate
//! docs.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "linux" | "android" | "freebsd" | "netbsd" | "openbsd" | "dragonfly" | "illumos"
        | "solaris" => println!("cargo:rustc-link-arg-tests=-rdynamic"),
        "macos" | "ios" => println!("cargo:rustc-link-arg-tests=-Wl,-export_dynamic"),
        _ => {}
    }
}
