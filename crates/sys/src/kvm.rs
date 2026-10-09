// SPDX-License-Identifier: MIT OR Apache-2.0

//! The parts of the KVM dirty ring API that kvm-ioctls does not wrap.
//!
//! With `KVM_CAP_DIRTY_LOG_RING` each vCPU has a ring of `struct kvm_dirty_gfn` entries that the
//! kernel fills as the guest writes to logged memory. Userspace maps it from the vCPU descriptor
//! at page `KVM_DIRTY_LOG_PAGE_OFFSET`, reads the entries the kernel marked dirty, marks them
//! collected, and then asks the kernel with `KVM_RESET_DIRTY_RINGS` to write protect those pages
//! again and take the entries back. This module holds the mapping, the entry type with the
//! atomics the protocol needs, and that ioctl.

use std::io;
use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// `KVM_DIRTY_LOG_PAGE_OFFSET`: the page of the vCPU mapping where the dirty ring starts.
pub const KVM_DIRTY_LOG_PAGE_OFFSET: usize = 64;

/// `KVM_DIRTY_GFN_F_DIRTY`: the kernel published the entry.
pub const KVM_DIRTY_GFN_F_DIRTY: u32 = 1;

/// `KVM_DIRTY_GFN_F_RESET`: userspace collected the entry and the next reset may take it back.
pub const KVM_DIRTY_GFN_F_RESET: u32 = 2;

/// `KVM_RESET_DIRTY_RINGS`, which is `_IO(KVMIO, 0xc7)` with `KVMIO` 0xAE.
const KVM_RESET_DIRTY_RINGS: libc::c_ulong = (0xAE << 8) | 0xc7;

/// One ring entry, `struct kvm_dirty_gfn`. The kernel writes `slot` and `offset` and then
/// publishes the entry by storing [`KVM_DIRTY_GFN_F_DIRTY`] into `flags` with release semantics,
/// so every field is an atomic and the flags are read with acquire.
#[repr(C)]
#[derive(Debug, Default)]
pub struct DirtyGfn {
    /// `KVM_DIRTY_GFN_F_*`.
    pub flags: AtomicU32,
    /// The address space id in the upper 16 bits and the slot id in the lower 16.
    pub slot: AtomicU32,
    /// The page number inside the slot.
    pub offset: AtomicU64,
}

impl DirtyGfn {
    /// `dirty_gfn_is_dirtied()`: the kernel published the entry and userspace has not collected
    /// it yet.
    pub fn is_dirtied(&self) -> bool {
        self.flags.load(Ordering::Acquire) == KVM_DIRTY_GFN_F_DIRTY
    }

    /// `dirty_gfn_set_collected()`. The release store pairs with the kernel's read in
    /// `KVM_RESET_DIRTY_RINGS`, so the reset sees the entry as collected only after userspace
    /// read its slot and offset.
    pub fn set_collected(&self) {
        self.flags.store(KVM_DIRTY_GFN_F_RESET, Ordering::Release);
    }
}

/// A vCPU's dirty ring, mapped shared from the vCPU descriptor, `cpu->kvm_dirty_gfns`.
#[derive(Debug)]
pub struct DirtyRingMap {
    /// The address of the mapping. A plain address keeps the type Send and Sync without an
    /// unsafe impl; it is only turned back into a pointer in [`DirtyRingMap::entries`].
    addr: usize,
    entries: usize,
}

impl DirtyRingMap {
    /// `map_kvm_dirty_gfns()`: maps the `entries` entries of the dirty ring of the vCPU behind
    /// `vcpu`. `page_size` is the host page size, which places the ring in the vCPU mapping.
    /// The mapping holds its own reference to the file, so it stays valid if the descriptor is
    /// closed first.
    pub fn map(vcpu: &impl AsRawFd, entries: usize, page_size: usize) -> io::Result<Self> {
        let invalid = || io::Error::from_raw_os_error(libc::EINVAL);
        let len = entries.checked_mul(size_of::<DirtyGfn>()).ok_or_else(invalid)?;
        if len == 0 {
            return Err(invalid());
        }
        let offset = page_size.checked_mul(KVM_DIRTY_LOG_PAGE_OFFSET).ok_or_else(invalid)?;
        let offset = libc::off_t::try_from(offset).map_err(|_| invalid())?;
        // SAFETY: a new mapping at an address the kernel picks aliases nothing in this process.
        // The kernel writes the ring concurrently, which the atomic fields of `DirtyGfn` allow.
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                vcpu.as_raw_fd(),
                offset,
            )
        };
        if p == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(DirtyRingMap { addr: p as usize, entries })
    }

    /// The entries of the ring.
    pub fn entries(&self) -> &[DirtyGfn] {
        // SAFETY: the mapping is `entries * 16` bytes long, page aligned and lives as long as
        // `self`. `DirtyGfn` is `repr(C)` with the layout of `struct kvm_dirty_gfn`, and every
        // bit pattern is a valid value of its atomic fields.
        unsafe { std::slice::from_raw_parts(self.addr as *const DirtyGfn, self.entries) }
    }
}

impl Drop for DirtyRingMap {
    fn drop(&mut self) {
        // SAFETY: the mapping came from `mmap` in `map` with this length, and no reference into
        // it outlives `self`.
        unsafe {
            libc::munmap(self.addr as *mut libc::c_void, self.entries * size_of::<DirtyGfn>())
        };
    }
}

/// `KVM_RESET_DIRTY_RINGS` on the VM behind `vm`: write protects the pages of every collected
/// entry again and gives the entries back to the kernel. Returns how many entries it reset.
pub fn reset_dirty_rings(vm: &impl AsRawFd) -> io::Result<u32> {
    // SAFETY: the ioctl takes no argument, so it reads and writes no memory of this process.
    // On a descriptor that is not a KVM VM it fails with ENOTTY.
    let ret = unsafe { libc::ioctl(vm.as_raw_fd(), KVM_RESET_DIRTY_RINGS as _) };
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(ret as u32) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_has_the_kernel_layout() {
        assert_eq!(size_of::<DirtyGfn>(), 16);
        assert_eq!(std::mem::offset_of!(DirtyGfn, slot), 4);
        assert_eq!(std::mem::offset_of!(DirtyGfn, offset), 8);
        assert_eq!(KVM_RESET_DIRTY_RINGS, 0xaec7);
    }

    #[test]
    fn collected_entries_are_not_dirtied() {
        let gfn = DirtyGfn::default();
        assert!(!gfn.is_dirtied());
        gfn.flags.store(KVM_DIRTY_GFN_F_DIRTY, Ordering::Release);
        assert!(gfn.is_dirtied());
        gfn.set_collected();
        assert!(!gfn.is_dirtied());
        assert_eq!(gfn.flags.load(Ordering::Relaxed), KVM_DIRTY_GFN_F_RESET);
    }

    /// A memfd stands in for the vCPU descriptor: the mapping comes from the right page and
    /// writes through one mapping show up in another, as the kernel's do.
    #[test]
    fn maps_the_ring_page_of_a_descriptor() {
        let fd = rustix::fs::memfd_create("ring", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let file = std::fs::File::from(fd);
        file.set_len((KVM_DIRTY_LOG_PAGE_OFFSET as u64 + 1) * 4096).unwrap();
        let a = DirtyRingMap::map(&file, 256, 4096).unwrap();
        let b = DirtyRingMap::map(&file, 256, 4096).unwrap();
        assert_eq!(a.entries().len(), 256);
        b.entries()[255].offset.store(7, Ordering::Relaxed);
        assert_eq!(a.entries()[255].offset.load(Ordering::Relaxed), 7);
        assert!(DirtyRingMap::map(&file, 0, 4096).is_err());
    }

    #[test]
    fn reset_needs_a_vm() {
        let fd = rustix::fs::memfd_create("vm", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let err = reset_dirty_rings(&fd).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(libc::ENOTTY));
    }
}
