// SPDX-License-Identifier: GPL-2.0-or-later

//! The `qemu-system-*` personality.

use std::process::ExitCode;

use ruvm_base::report::{error_report, set_program_name};
use ruvm_system::options::{Opt, help_text, lookup_opt};

use crate::names::Personality;
use crate::version;

/// Walks the command line the way `qemu_init()` does. The options that print something and exit
/// work, and so do QEMU's errors for options that do not exist or lack their argument. Starting
/// a machine does not work yet.
pub(crate) fn run(target: &'static str, argv0: &str, args: &[String]) -> ExitCode {
    // error_init() keeps the part of argv[0] after the last slash.
    let prgname = argv0.rsplit('/').next().unwrap_or(argv0);
    set_program_name(prgname);
    let personality = Personality::System(target);
    let version_text = version::text(&personality, argv0).unwrap_or_default();
    let mut optind = 0;
    while optind < args.len() {
        if !args[optind].starts_with('-') {
            // A disk image for IDE hard disk 0.
            optind += 1;
            continue;
        }
        let found = match lookup_opt(args, &mut optind) {
            Ok(found) => found,
            Err(e) => {
                eprintln!("{prgname}: {e}");
                return ExitCode::from(1);
            }
        };
        match found.option.index {
            Opt::H => {
                print!("{}", help_text(target, prgname, &version_text));
                return ExitCode::SUCCESS;
            }
            Opt::Version => {
                print!("{version_text}");
                return ExitCode::SUCCESS;
            }
            _ => {}
        }
    }
    error_report(&format!(
        "ruvm {} cannot start a machine yet, see https://github.com/tamnd/ruvm#status",
        version::RUVM_VERSION
    ));
    ExitCode::from(1)
}
