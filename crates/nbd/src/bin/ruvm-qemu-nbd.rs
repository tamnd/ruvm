// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-nbd on its own, for the tests of this crate and the iotests harness. The `ruvm` binary is
//! the real entry point; this one calls the same [`ruvm_nbd::main`] with the name `qemu-nbd`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let argv0 = argv0("qemu-nbd");
    let version = format!(
        "{argv0} 11.1.0 (ruvm {})\nWritten by Anthony Liguori.\n\nCopyright (c) 2003-2026 \
         Fabrice Bellard and the QEMU Project developers\nThis is free software; see the source \
         for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or \
         FITNESS FOR A PARTICULAR PURPOSE.\n",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(ruvm_nbd::main(&argv0, &args, &version))
}

/// `argv[0]` as given when the program runs under the name `qemu-nbd`, through a link as the
/// iotests harness runs it, so that messages print it exactly as QEMU's `qemu-nbd` does; the plain
/// name when it runs as `ruvm-qemu-nbd`.
fn argv0(name: &str) -> String {
    let given = std::env::args().next().unwrap_or_default();
    let base = given.rsplit('/').next().unwrap_or(&given);
    if base == name { given } else { name.to_string() }
}
