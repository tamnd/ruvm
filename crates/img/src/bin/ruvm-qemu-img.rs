// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-img on its own, for the tests of this crate and the iotests harness. The `ruvm` binary is
//! the real entry point; this one calls the same [`ruvm_img::main`] with the name `qemu-img`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let argv0 = argv0("qemu-img");
    let version = format!(
        "qemu-img version 11.1.0 (ruvm {})\nCopyright (c) 2003-2026 Fabrice Bellard and the QEMU \
         Project developers\n",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(ruvm_img::main(&argv0, &args, &version))
}

/// `argv[0]` as given when the program runs under the name `qemu-img`, through a link as the
/// iotests harness runs it, so that messages print it exactly as QEMU's `qemu-img` does; the plain
/// name when it runs as `ruvm-qemu-img`.
fn argv0(name: &str) -> String {
    let given = std::env::args().next().unwrap_or_default();
    let base = given.rsplit('/').next().unwrap_or(&given);
    if base == name { given } else { name.to_string() }
}
