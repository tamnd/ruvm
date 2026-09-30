// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest agent, qemu-ga, as its own small binary.
//!
//! It runs inside guests, often on minimal images, so it stays out of the multi-call binary. For
//! now it only answers its version option, in the format qga/main.c prints, which is the plain
//! QEMU version without the package suffix.

use std::process::ExitCode;

const QEMU_VERSION: &str = "11.1.0";

fn main() -> ExitCode {
    let wants_version = std::env::args()
        .skip(1)
        .take_while(|a| a.starts_with('-'))
        .any(|a| a == "-V" || a == "--version");
    if wants_version {
        println!("QEMU Guest Agent {QEMU_VERSION}");
        return ExitCode::SUCCESS;
    }
    eprintln!(
        "qemu-ga: ruvm {} cannot run the guest agent yet, see https://github.com/tamnd/ruvm#status",
        env!("CARGO_PKG_VERSION")
    );
    ExitCode::from(1)
}
