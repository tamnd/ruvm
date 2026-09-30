// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-nbd on its own, for the tests of this crate. The `ruvm` binary is the real entry
//! point; this one calls the same [`ruvm_nbd::main`] with the name `qemu-nbd`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let version = format!(
        "qemu-nbd 11.1.0 (ruvm {})\nWritten by Anthony Liguori.\n\nCopyright (c) 2003-2026 \
         Fabrice Bellard and the QEMU Project developers\nThis is free software; see the source \
         for copying conditions.  There is NO\nwarranty; not even for MERCHANTABILITY or \
         FITNESS FOR A PARTICULAR PURPOSE.\n",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(ruvm_nbd::main("qemu-nbd", &args, &version))
}
