// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-storage-daemon on its own, for the tests of this crate. The `ruvm` binary is the real
//! entry point; this one calls the same [`ruvm_storage_daemon::main`] with the name
//! `qemu-storage-daemon`.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let version = format!(
        "qemu-storage-daemon version 11.1.0 (ruvm {})\nCopyright (c) 2003-2026 Fabrice Bellard \
         and the QEMU Project developers\n",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(ruvm_storage_daemon::main("qemu-storage-daemon", &args, &version))
}
