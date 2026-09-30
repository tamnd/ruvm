// SPDX-License-Identifier: MIT OR Apache-2.0

//! Check and maintenance tasks for the ruvm workspace.
//!
//! Everything here is a check that reads the whole tree or shells out, which is why it is not a
//! unit test. Running it through cargo rather than a shell script means it behaves the same on a
//! laptop and on a Windows runner.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};
use std::time::Instant;

mod layers;
mod provenance;
mod style;
mod unsafe_audit;
mod upstream;
mod workspace;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = root();
    let result = match args.first().map(String::as_str) {
        Some("layers") => layers::check(&root),
        Some("provenance") => provenance::check(&root),
        Some("unsafe-audit") => unsafe_audit::check(&root),
        Some("style") => style::check(&root),
        Some("upstream-sync") => upstream::sync(&root, &args[1..]),
        Some("vendor-check") => upstream::check(&root),
        Some("ci") => ci(&root),
        Some("help" | "--help" | "-h") | None => {
            usage();
            Ok(())
        }
        Some(other) => {
            usage();
            Err(format!("no task called {other}"))
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!(
        "cargo xtask <task>

  layers         check the layer rule against xtask/layers.toml
  provenance     check that no permissive crate depends on a GPL crate, and the SPDX headers
  unsafe-audit   count unsafe items per crate against the budget in its manifest
  style          check the prose rules in every markdown file
  upstream-sync  refresh vendor-qemu/ from a QEMU tag and print what changed
                 cargo xtask upstream-sync <tag> [--from <qemu checkout>]
  vendor-check   check vendor-qemu/ against its MANIFEST
  ci             everything CI runs, in the order it runs it"
    );
}

/// The workspace root, which is the parent of this crate's directory.
fn root() -> PathBuf {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent().map(Path::to_path_buf).unwrap_or(here)
}

fn ci(root: &Path) -> Result<(), String> {
    let whole = Instant::now();
    step("layers", || layers::check(root))?;
    step("provenance", || provenance::check(root))?;
    step("unsafe-audit", || unsafe_audit::check(root))?;
    step("style", || style::check(root))?;
    step("vendor-check", || upstream::check(root))?;
    step("fmt", || cargo(&["fmt", "--all", "--check"]))?;
    step("clippy", || cargo(&["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]))?;
    step("test", || cargo(&["test", "--workspace"]))?;
    step("doc", || cargo(&["doc", "--workspace", "--no-deps"]))?;
    println!("everything CI runs is green, in {:.1}s", whole.elapsed().as_secs_f64());
    Ok(())
}

fn step(name: &str, run: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    println!("== {name}");
    let start = Instant::now();
    run().map_err(|e| format!("{name}: {e}"))?;
    println!("== {name} took {:.1}s", start.elapsed().as_secs_f64());
    Ok(())
}

fn cargo(args: &[&str]) -> Result<(), String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let status = Command::new(cargo)
        .args(args)
        .status()
        .map_err(|e| format!("could not run cargo {}: {e}", args.join(" ")))?;
    if status.success() { Ok(()) } else { Err(format!("cargo {} failed", args.join(" "))) }
}
