// SPDX-License-Identifier: GPL-2.0-or-later

//! Where firmware and option ROM files come from, `qemu_find_file()`.
//!
//! QEMU first tries the name as a path, then every data directory in order. The data
//! directories are the `-L` ones in command line order followed by the install directory.
//! ruvm does not install its own blobs yet, so after `-L` it looks where distributions put
//! them:
//!
//! 1. `/usr/share/qemu`
//! 2. `/usr/share/seabios`
//! 3. `/usr/local/share/qemu`
//!
//! and, as a last resort, in the data directory of a QEMU found on `PATH`
//! (`<dir of qemu-system-x86_64>/../share/qemu`). That covers Homebrew and source installs,
//! which is where `linuxboot_dma.bin` and the other option ROMs usually are.

use std::env;
use std::path::{Path, PathBuf};

/// The directories searched after the `-L` ones, in order.
pub const DEFAULT_FIRMWARE_DIRS: &[&str] =
    &["/usr/share/qemu", "/usr/share/seabios", "/usr/local/share/qemu"];

/// The QEMU binary whose data directory is the last fallback.
pub const QEMU_BINARY: &str = "qemu-system-x86_64";

/// A firmware search path.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FirmwareSearch {
    dirs: Vec<PathBuf>,
}

impl FirmwareSearch {
    /// The `-L` directories, then [`DEFAULT_FIRMWARE_DIRS`], then the data directory of the
    /// QEMU on `PATH` if there is one.
    pub fn new(l_dirs: &[PathBuf]) -> FirmwareSearch {
        Self::with_fallback(l_dirs, qemu_data_dir(env::var_os("PATH").as_deref()))
    }

    /// Like [`FirmwareSearch::new`] with an explicit last fallback directory.
    pub fn with_fallback(l_dirs: &[PathBuf], fallback: Option<PathBuf>) -> FirmwareSearch {
        let mut dirs: Vec<PathBuf> = l_dirs.to_vec();
        dirs.extend(DEFAULT_FIRMWARE_DIRS.iter().map(PathBuf::from));
        dirs.extend(fallback);
        FirmwareSearch { dirs }
    }

    /// Exactly these directories, nothing else.
    pub fn from_dirs(dirs: Vec<PathBuf>) -> FirmwareSearch {
        FirmwareSearch { dirs }
    }

    /// The directories, in search order.
    pub fn dirs(&self) -> &[PathBuf] {
        &self.dirs
    }

    /// `qemu_find_file()`: `name` itself if it can be read, otherwise the first data directory
    /// that has it.
    pub fn find(&self, name: &str) -> Option<PathBuf> {
        let direct = Path::new(name);
        if direct.is_file() {
            return Some(direct.to_path_buf());
        }
        self.dirs.iter().map(|d| d.join(name)).find(|p| p.is_file())
    }

    /// The contents of the file [`FirmwareSearch::find`] picks.
    pub fn load(&self, name: &str) -> Option<Vec<u8>> {
        std::fs::read(self.find(name)?).ok()
    }
}

/// `<dir>/../share/qemu` for the first [`QEMU_BINARY`] in the `PATH` value `path`, following
/// symlinks so that Homebrew's `bin` links end up in the Cellar.
pub fn qemu_data_dir(path: Option<&std::ffi::OsStr>) -> Option<PathBuf> {
    let path = path?;
    for dir in env::split_paths(path) {
        let bin = dir.join(QEMU_BINARY);
        if !bin.is_file() {
            continue;
        }
        let bin = bin.canonicalize().unwrap_or(bin);
        let share = bin.parent()?.parent()?.join("share").join("qemu");
        if share.is_dir() {
            return Some(share);
        }
    }
    None
}
