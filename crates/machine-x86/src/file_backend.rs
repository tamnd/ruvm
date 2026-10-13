// SPDX-License-Identifier: GPL-2.0-or-later

//! A raw image file as the backend of a disk, for `-drive file=...,format=raw`.
//!
//! The block layer proper (formats, protocols, caching modes) lives elsewhere. This is the
//! small adapter the boards need so that `-drive` works today: one host file, read and
//! written at the offsets the guest asks for, or the `null-co` protocol of block/null.c,
//! which has no file. It implements both the virtio-blk backend trait and the one of the
//! IDE/AHCI emulation.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

/// A raw image in a host file, or a `null-co` image. Clones share the image, as the device
/// and the monitor share the `BlockBackend` of a drive in QEMU.
#[derive(Clone, Debug)]
pub struct FileBackend {
    path: PathBuf,
    /// `None` for `null-co`.
    file: Option<Arc<Mutex<File>>>,
    size: Arc<AtomicU64>,
    writable: bool,
}

/// `null-co://`, the file name of a `null-co` image.
pub const NULL_CO: &str = "null-co://";

/// `BLOCK_OPT_SIZE` of `null-co` when it is not given: 1 GiB.
pub const NULL_CO_DEFAULT_SIZE: u64 = 1 << 30;

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
            file: Some(Arc::new(Mutex::new(file))),
            size: Arc::new(AtomicU64::new(size)),
            writable: !read_only,
        })
    }

    /// A `null-co` image of `size` bytes: writes go nowhere and reads give zeroes. block/null.c
    /// leaves the buffer as it is unless `read-zeroes` is on, and zeroes are one valid way to
    /// do that.
    pub fn null(size: u64, read_only: bool) -> FileBackend {
        let size = Arc::new(AtomicU64::new(size));
        FileBackend { path: PathBuf::from(NULL_CO), file: None, size, writable: !read_only }
    }

    /// The file the image lives in.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The size of the image in bytes.
    pub fn size(&self) -> u64 {
        self.size.load(Ordering::Acquire)
    }

    /// `bdrv_co_truncate()` of a raw image with no preallocation, as `block_resize` does it:
    /// the file grows or shrinks to `size` bytes. The error is QEMU's.
    pub fn truncate(&self, size: u64) -> Result<(), String> {
        let Some(file) = &self.file else {
            return Err("Image format driver does not support resize".to_string());
        };
        if !self.writable {
            return Err("Image is read-only".to_string());
        }
        let f = Self::lock(file);
        f.set_len(size).map_err(|e| format!("Could not resize file: {}", strerror(&e)))?;
        self.size.store(size, Ordering::Release);
        Ok(())
    }

    /// Whether the image was opened for writing.
    pub fn is_writable(&self) -> bool {
        self.writable
    }

    fn lock(file: &Mutex<File>) -> MutexGuard<'_, File> {
        file.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn check(&self, offset: u64, len: usize) -> io::Result<()> {
        match offset.checked_add(len as u64) {
            Some(end) if end <= self.size() => Ok(()),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "access beyond the end of the image",
            )),
        }
    }

    /// Fills `buf` with the bytes at `offset`.
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.check(offset, buf.len())?;
        let Some(file) = &self.file else {
            buf.fill(0);
            return Ok(());
        };
        let mut f = Self::lock(file);
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }

    /// Writes `buf` at `offset`.
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only image"));
        }
        self.check(offset, buf.len())?;
        let Some(file) = &self.file else {
            return Ok(());
        };
        let mut f = Self::lock(file);
        f.seek(SeekFrom::Start(offset))?;
        f.write_all(buf)
    }

    /// Makes earlier writes durable.
    pub fn flush(&self) -> io::Result<()> {
        match &self.file {
            Some(file) if self.writable => Self::lock(file).sync_data(),
            _ => Ok(()),
        }
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
        FileBackend::size(self)
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
        FileBackend::size(self)
    }
}
