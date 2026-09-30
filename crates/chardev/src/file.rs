// SPDX-License-Identifier: GPL-2.0-or-later

//! The `file` and `pipe` chardevs, chardev/char-file.c and chardev/char-pipe.c.

use std::fs::{File, OpenOptions};

use ruvm_base::error::strerror;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{ChardevFile, ChardevHostdev};

use crate::local::{Local, LocalKind};

/// `error_setg_file_open()`.
fn open_error(path: &str, e: &std::io::Error) -> Error {
    Error::generic(format!("Could not open '{path}': {}", strerror(e)))
}

#[cfg(not(windows))]
fn out_error(path: &str, e: &std::io::Error) -> Error {
    open_error(path, e)
}

/// Windows QEMU says less about it.
#[cfg(windows)]
fn out_error(path: &str, _e: &std::io::Error) -> Error {
    Error::generic(format!("open {path} failed"))
}

/// `file_chr_open()`: output goes to `out`, created if need be and emptied unless `append`
/// is set. Input comes from `in` when there is one, which Windows does not have.
pub(crate) fn open_file(file: &ChardevFile) -> Result<Local> {
    #[cfg(windows)]
    if file.in_.is_some() {
        return Err(Error::generic("input file not supported"));
    }
    let mut o = OpenOptions::new();
    o.write(true).create(true);
    if file.append == Some(true) {
        o.append(true);
    } else {
        o.truncate(true);
    }
    let out = o.open(&file.out).map_err(|e| out_error(&file.out, &e))?;
    #[cfg(unix)]
    let input = match &file.in_ {
        Some(p) => {
            let f = File::open(p).map_err(|e| open_error(p, &e))?;
            Some(Box::new(crate::local::FdSource(f)) as Box<dyn crate::local::Source>)
        }
        None => None,
    };
    #[cfg(not(unix))]
    let input = None;
    Ok(Local::new(LocalKind::File, input, Box::new(out)))
}

fn open_rw(path: &str) -> std::io::Result<File> {
    OpenOptions::new().read(true).write(true).open(path)
}

/// `pipe_chr_open()`: the two fifos `path.in` and `path.out`, or `path` itself for both ways
/// when either of those is missing.
#[cfg(unix)]
pub(crate) fn open_pipe(dev: &ChardevHostdev) -> Result<Local> {
    use crate::local::FdSource;

    let path = &dev.device;
    let (input, output) = match (open_rw(&format!("{path}.in")), open_rw(&format!("{path}.out"))) {
        (Ok(i), Ok(o)) => (i, o),
        _ => {
            let f = open_rw(path).map_err(|e| open_error(path, &e))?;
            let o = f.try_clone().map_err(|e| open_error(path, &e))?;
            (f, o)
        }
    };
    Ok(Local::new(LocalKind::Pipe, Some(Box::new(FdSource(input))), Box::new(output)))
}

/// QEMU makes a named pipe server under `\\.\pipe\` here, which ruvm cannot do yet.
#[cfg(not(unix))]
pub(crate) fn open_pipe(dev: &ChardevHostdev) -> Result<Local> {
    let _ = (open_rw, open_error, &dev.device);
    Err(Error::generic("chardev backend 'pipe' is not supported by ruvm on Windows yet"))
}
