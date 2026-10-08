// SPDX-License-Identifier: GPL-2.0-or-later

//! The RAM write tracking of migration/ram.c that background snapshots use.
//!
//! A background snapshot saves the device state with the guest stopped and then lets the guest
//! run while RAM goes out. To keep RAM as it was at the stop, every page is write-protected with
//! userfaultfd first. A guest write to a page that has not gone out yet blocks its thread and
//! comes here as a fault; the migration thread sends that page next and lifts the protection,
//! which wakes the writer. Pages that went out are unprotected as they go, so each page faults
//! at most once.

use std::io;
use std::sync::Arc;

use ruvm_mem::RamBlock;

use crate::postcopy::uffd::{FEATURE_PAGEFAULT_FLAG_WP, Uffd};

/// `ram_write_tracking_available()`: whether the kernel can write-protect anonymous memory
/// through userfaultfd.
pub fn available() -> bool {
    Uffd::features().is_ok_and(|f| f & FEATURE_PAGEFAULT_FLAG_WP != 0)
}

/// `ram_write_tracking_compatible()`: whether every block takes write protection, tried on a
/// descriptor of its own that is closed again.
pub fn compatible(blocks: &[Arc<RamBlock>]) -> bool {
    let Ok(uffd) = Uffd::with_features(FEATURE_PAGEFAULT_FLAG_WP) else { return false };
    blocks.iter().all(|b| {
        let ok = uffd.register_write_protect(b.host_memory()).unwrap_or(false);
        let _ = uffd.unregister(b.host_memory());
        ok
    })
}

/// `ram_write_tracking_prepare()`: reads a byte of every page. Write protection skips pages
/// that were never touched, so they all have to be mapped first.
pub fn prepare(blocks: &[Arc<RamBlock>]) {
    let mut byte = [0u8];
    for b in blocks {
        let page = 1u64 << b.page_bits();
        let mut offset = 0;
        while offset < b.len() {
            let _ = b.read(offset, &mut byte);
            offset += page;
        }
    }
}

/// Write protection over a set of RAM blocks, `rs->uffdio_fd` with the blocks that have
/// `RAM_UF_WRITEPROTECT`. Dropping it unprotects them and wakes any writer still waiting.
#[derive(Debug)]
pub struct WriteTracking {
    uffd: Uffd,
    blocks: Vec<Arc<RamBlock>>,
}

impl WriteTracking {
    /// `ram_write_tracking_start()`: registers every block and protects all of it.
    pub fn start(blocks: &[Arc<RamBlock>]) -> io::Result<Self> {
        let uffd = Uffd::with_features(FEATURE_PAGEFAULT_FLAG_WP)?;
        let mut t = WriteTracking { uffd, blocks: Vec::with_capacity(blocks.len()) };
        for b in blocks {
            let mem = b.host_memory();
            // On failure the blocks registered so far are released by the drop.
            t.uffd.register_write_protect(mem)?;
            t.blocks.push(b.clone());
            t.uffd.write_protect(mem, 0, mem.len(), true)?;
        }
        Ok(t)
    }

    /// `poll_fault_page()`: the next page a writer waits for, as the index of its block in the
    /// list given to [`start`](Self::start) and the byte offset of the page, without blocking.
    pub fn poll_fault(&self) -> io::Result<Option<(usize, u64)>> {
        let Some(addr) = self.uffd.wait_fault(0)? else { return Ok(None) };
        for (i, b) in self.blocks.iter().enumerate() {
            let base = b.host_memory().host_addr();
            if addr >= base && addr < base + b.host_memory().len() {
                let offset = (addr - base) as u64 & !((1u64 << b.page_bits()) - 1);
                return Ok(Some((i, offset)));
            }
        }
        Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Write fault at {addr:#x} outside guest RAM"),
        ))
    }

    /// `ram_save_release_protection()`: lifts the protection of `len` bytes at `offset` of
    /// block `b`, which went out. Writers waiting on them go on.
    pub fn release(&self, b: usize, offset: u64, len: u64) -> io::Result<()> {
        let Some(block) = self.blocks.get(b) else { return Ok(()) };
        self.uffd.write_protect(block.host_memory(), offset as usize, len as usize, false)
    }
}

impl Drop for WriteTracking {
    /// `ram_write_tracking_stop()`.
    fn drop(&mut self) {
        for b in &self.blocks {
            let _ = self.uffd.unregister(b.host_memory());
        }
    }
}
