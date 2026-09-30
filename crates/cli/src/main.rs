// SPDX-License-Identifier: GPL-2.0-or-later

//! The ruvm multi-call binary, answering to every QEMU binary name.
//!
//! An installer puts `ruvm` on disk once and symlinks every QEMU name to it. At startup the file
//! name in `argv[0]` decides which program this is, the way busybox does it. `ruvm --list` prints
//! the names, and `ruvm <name> [args]` runs a personality without a symlink, which is how the
//! tests reach them.

use std::process::ExitCode;

mod names;
mod system;
mod tools;
mod version;

use names::Personality;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let argv0 = args.first().map(String::as_str).unwrap_or("ruvm");
    match Personality::from_argv0(argv0) {
        Some(Personality::Ruvm) | None => ruvm(&args[1.min(args.len())..]),
        Some(personality) => run(&personality, argv0, &args[1..]),
    }
}

/// `ruvm` under its own name, or under a name it does not know.
fn ruvm(args: &[String]) -> ExitCode {
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            print!("{}", version::text(&Personality::Ruvm, "ruvm").unwrap_or_default());
            ExitCode::SUCCESS
        }
        Some("--list") => {
            for p in Personality::all().iter().skip(1) {
                println!("{}", p.name());
            }
            ExitCode::SUCCESS
        }
        Some("--help" | "-h") | None => {
            println!(
                "Usage: ruvm <name> [args...]\n       ruvm --list | --version\n\nruvm runs as whichever QEMU program it is named after. Symlink a QEMU name to it,\nor pass the name as the first argument. `ruvm --list` prints every name it answers to."
            );
            ExitCode::SUCCESS
        }
        Some(name) => match Personality::from_name(name) {
            Some(Personality::Ruvm) => ruvm(&args[1..]),
            Some(personality) => run(&personality, name, &args[1..]),
            None => {
                eprintln!("ruvm: {name} is not a QEMU program name, see ruvm --list");
                ExitCode::from(1)
            }
        },
    }
}

fn run(personality: &Personality, argv0: &str, args: &[String]) -> ExitCode {
    if let Personality::System(target) = personality {
        return system::run(target, argv0, args);
    }
    if let Some(code) = tools::run(personality, argv0, args) {
        return code;
    }
    if wants_version(personality, args) {
        if let Some(text) = version::text(personality, argv0) {
            print!("{text}");
            return ExitCode::SUCCESS;
        }
    }
    eprintln!(
        "{}: ruvm {} cannot run this program yet, see https://github.com/tamnd/ruvm#status",
        personality.name(),
        version::RUVM_VERSION
    );
    ExitCode::from(1)
}

/// Whether the arguments ask for the version, spelled the way that program spells it.
///
/// The system emulator and user mode take single or double dash long options and act on the first
/// one they meet. The tools use getopt, which stops at the first argument that is not an option,
/// and they take `-V` as well.
fn wants_version(personality: &Personality, args: &[String]) -> bool {
    let options = args.iter().take_while(|a| a.starts_with('-'));
    match personality {
        Personality::System(_) | Personality::User(_) => {
            options.map(String::as_str).any(|a| a == "-version" || a == "--version")
        }
        Personality::Tool(_) => options.map(String::as_str).any(|a| a == "-V" || a == "--version"),
        Personality::Ruvm => false,
    }
}
