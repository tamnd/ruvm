// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-io on its own, for the iotests harness. The `ruvm` binary is the real entry point; this
//! one calls the same [`ruvm_io::main`] with the name `qemu-io`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let argv0 = argv0("qemu-io");
    let version = format!(
        "qemu-io version 11.1.0 (ruvm {})\nCopyright (c) 2003-2026 Fabrice Bellard and the QEMU \
         Project developers\n",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(ruvm_io::main(&argv0, &args, &version))
}

/// `argv[0]` as given when the program runs under the name `qemu-io`, through a link as the
/// iotests harness runs it, so that messages print it exactly as QEMU's `qemu-io` does; the plain
/// name when it runs as `ruvm-qemu-io`.
fn argv0(name: &str) -> String {
    let given = std::env::args().next().unwrap_or_default();
    let base = given.rsplit('/').next().unwrap_or(&given);
    if base == name { given } else { name.to_string() }
}
