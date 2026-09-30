// SPDX-License-Identifier: GPL-2.0-or-later

//! `ruvm-decodetree`, a command line with the same options as QEMU's scripts/decodetree.py that
//! writes Rust instead of C. Build scripts should call the library; this is for looking at the
//! output and for running QEMU's tests/decode files by hand.

#![forbid(unsafe_code)]

use std::io::Write as _;
use std::process::ExitCode;

use ruvm_decode::{Error, Invocation, generate_files};

fn run(inv: &Invocation) -> Result<(), Error> {
    let mut texts = Vec::new();
    for f in &inv.files {
        match std::fs::read_to_string(f) {
            Ok(t) => texts.push((f.as_str(), t)),
            Err(e) => {
                return Err(Error { file: f.clone(), line: 0, message: e.to_string() });
            }
        }
    }
    let inputs: Vec<(&str, &str)> = texts.iter().map(|(n, t)| (*n, t.as_str())).collect();
    let out = generate_files(&inputs, &inv.options)?;
    if inv.output_null {
        return Ok(());
    }
    let written = match &inv.output {
        Some(path) => std::fs::write(path, &out),
        None => std::io::stdout().lock().write_all(out.as_bytes()),
    };
    written.map_err(|e| Error {
        file: inv.output.clone().unwrap_or_default(),
        line: 0,
        message: e.to_string(),
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let inv = match Invocation::parse(&args) {
        Ok(inv) => inv,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&inv) {
        Ok(()) if inv.test_for_error => ExitCode::FAILURE,
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{}", e.render(inv.test_for_error));
            if inv.test_for_error { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
    }
}
