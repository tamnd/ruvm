// SPDX-License-Identifier: GPL-2.0-or-later

//! Where a qcow2 image keeps its bytes: the `file` child, the optional `data-file` child and the
//! optional backing image. They are all nodes of the block graph, behind the small
//! [`Storage`] and [`Backing`] traits so that the format code does not depend on the graph.

use std::fmt;
use std::io;
use std::sync::Arc;

use ruvm_base::Result;

use crate::node::{Node, is_enotsup};

/// The byte store under a qcow2 image, what `BdrvChild` gives the driver in QEMU.
pub(crate) trait Storage: Send + Sync + fmt::Debug {
    /// `bdrv_pread()`. Bytes past the end of the store read as zeroes.
    fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// `bdrv_pwrite()`.
    fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()>;

    /// `bdrv_pwrite_zeroes()`. The default writes a buffer of zeroes.
    fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        let _ = may_unmap;
        write_zero_buffer(self, offset, bytes)
    }

    /// `bdrv_pdiscard()`. Discarding is advice, so the default does nothing.
    fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        let _ = (offset, bytes);
        Ok(())
    }

    /// `bdrv_getlength()`: the length rounded up to whole sectors.
    fn len(&self) -> io::Result<u64>;

    /// `bdrv_truncate()` with `exact=true` and no preallocation.
    fn truncate(&self, len: u64) -> Result<()>;

    /// `bdrv_flush()`.
    fn flush(&self) -> io::Result<()>;

    /// The file name, for messages and `data-file` in the image information.
    fn filename(&self) -> Option<String> {
        None
    }

    /// `bdrv_get_allocated_file_size()`, or `None` where that is unknown.
    fn allocated_size(&self) -> Option<u64> {
        None
    }
}

/// Writes `bytes` zeroes at `offset` through `pwrite`, a chunk at a time.
pub(crate) fn write_zero_buffer<S: Storage + ?Sized>(
    s: &S,
    offset: u64,
    bytes: u64,
) -> io::Result<()> {
    const CHUNK: u64 = 1 << 20;
    let zeroes = vec![0u8; bytes.min(CHUNK) as usize];
    let mut done = 0;
    while done < bytes {
        let n = (bytes - done).min(CHUNK) as usize;
        s.pwrite(offset + done, &zeroes[..n])?;
        done += n as u64;
    }
    Ok(())
}

/// A child node of the qcow2 node in the block graph.
#[derive(Debug, Clone)]
pub(crate) struct NodeStorage(pub(crate) Arc<Node>);

impl Storage for NodeStorage {
    fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.pread(offset, buf)
    }

    fn pwrite(&self, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.0.pwrite(offset, buf)
    }

    fn pwrite_zeroes(&self, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        match self.0.pwrite_zeroes(offset, bytes, may_unmap) {
            Err(e) if is_enotsup(&e) => write_zero_buffer(self, offset, bytes),
            r => r,
        }
    }

    fn pdiscard(&self, offset: u64, bytes: u64) -> io::Result<()> {
        self.0.pdiscard(offset, bytes)
    }

    fn len(&self) -> io::Result<u64> {
        self.0.getlength()
    }

    fn truncate(&self, len: u64) -> Result<()> {
        self.0.truncate(len)
    }

    fn flush(&self) -> io::Result<()> {
        self.0.flush()
    }

    fn filename(&self) -> Option<String> {
        self.0.filename()
    }

    fn allocated_size(&self) -> Option<u64> {
        self.0.allocated_file_size().ok()
    }
}

/// The image a qcow2 image reads unallocated clusters from.
pub(crate) trait Backing: Send + Sync + fmt::Debug {
    /// Reads guest data. The caller keeps requests inside [`Backing::len`].
    fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;

    /// The virtual size of the backing image.
    fn len(&self) -> io::Result<u64>;
}

/// A backing node in the block graph.
#[derive(Debug, Clone)]
pub(crate) struct NodeBacking(pub(crate) Arc<Node>);

impl Backing for NodeBacking {
    fn pread(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.0.pread(offset, buf)
    }

    fn len(&self) -> io::Result<u64> {
        self.0.getlength()
    }
}
