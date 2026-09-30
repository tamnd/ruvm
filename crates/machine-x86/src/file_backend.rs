// SPDX-License-Identifier: GPL-2.0-or-later

//! A raw image file as the backend of a disk, for `-drive file=...,format=raw`.
//!
//! The block layer proper (formats, protocols, caching modes) lives elsewhere. This is the
//! small adapter the x86 boards need so that `-drive` works today: one host file, read and
//! written at the offsets the guest asks for. It implements both the virtio-blk backend trait
//! and the one of the IDE/AHCI emulation.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// A raw image in a host file.
#[derive(Debug)]
pub struct FileBackend {
    path: PathBuf,
    file: Mutex<File>,
    size: u64,
    writable: bool,
}

impl FileBackend {
    /// Opens `path`, read-only if `read_only` is set. The error is QEMU's "Could not open"
    /// message, without the option location in front.
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> Result<FileBackend, String> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(!read_only)
            .open(path)
            .map_err(|e| format!("Could not open '{}': {}", path.display(), strerror(&e)))?;
        let size = file
            .metadata()
            .map_err(|e| format!("Could not open '{}': {}", path.display(), strerror(&e)))?
            .len();
        Ok(FileBackend {
            path: path.to_path_buf(),
            file: Mutex::new(file),
            size,
            writable: !read_only,
        })
    }

    /// The file the image lives in.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The size of the image in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether the image was opened for writing.
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    fn lock(&self) -> MutexGuard<'_, File> {
        self.file.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn check(&self, offset: u64, len: usize) -> io::Result<()> {
        match offset.checked_add(len as u64) {
            Some(end) if end <= self.size => Ok(()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "access beyond the end of the image",
            )),
        }
    }

    /// Fills `buf` with the bytes at `offset`.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.check(offset, buf.len())?;
        let mut f = self.lock();
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }

    /// Writes `buf` at `offset`.
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only image"));
        }
        self.check(offset, buf.len())?;
        let mut f = self.lock();
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(buf)
    }

    /// Makes earlier writes durable.
    pub fn flush(&self) -> io::Result<()> {
        if !self.writable {
            return Ok(());
        }
        self.lock().sync_data()
    }
}

/// The C library's text for an I/O error, without Rust's " (os error N)" suffix.
fn strerror(e: &io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

impl ruvm_hw_virtio::BlockBackend for FileBackend {
    fn size(&self) -> u64 {
        self.size
    }

    fn is_writable(&self) -> bool {
        self.writable
    }

    fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        FileBackend::read_at(self, offset, buf)
    }

    fn write_at(&mut self, offset: u64, buf: &[u8]) -> io::Result<()> {
        FileBackend::write_at(self, offset, buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        FileBackend::flush(self)
    }
}

impl ruvm_hw_storage::BlockBackend for FileBackend {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        FileBackend::read_at(self, offset, buf)
    }

    fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        FileBackend::write_at(self, offset, buf)
    }

    fn flush(&self) -> io::Result<()> {
        FileBackend::flush(self)
    }

    fn len(&self) -> u64 {
        self.size
    }
}
