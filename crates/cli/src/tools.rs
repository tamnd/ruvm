// SPDX-License-Identifier: GPL-2.0-or-later

//! The tools that are ported: `qemu-img` and `qemu-io` so far.

use std::process::ExitCode;

use crate::names::{Personality, Tool};
use crate::version;

/// Runs `personality` if it is one of the ported tools; `None` for the others.
pub(crate) fn run(personality: &Personality, argv0: &str, args: &[String]) -> Option<ExitCode> {
    let Personality::Tool(tool) = personality else { return None };
    let main: fn(&str, &[String], &str) -> u8 = match tool {
        Tool::Img => ruvm_img::main,
        Tool::Io => ruvm_io::main,
        _ => return None,
    };
    let version_text = version::text(personality, argv0).unwrap_or_default();
    Some(ExitCode::from(main(argv0, args, &version_text)))
}
