// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qemu-system-*` personality.

use std::process::ExitCode;

use ruvm_base::report::set_program_name;
use ruvm_system::vl::{Personality, qemu_main};

use crate::names;
use crate::version;

/// Runs the system emulator for `target`, as `qemu-system-<target>` would.
pub(crate) fn run(target: &'static str, argv0: &str, args: &[String]) -> ExitCode {
    // error_init() keeps the part of argv[0] after the last slash.
    let prgname = argv0.rsplit('/').next().unwrap_or(argv0);
    set_program_name(prgname);
    let version_text =
        version::text(&names::Personality::System(target), argv0).unwrap_or_default();
    let p = Personality { target, prgname, version_text: &version_text };
    ExitCode::from(qemu_main(&p, args))
}
