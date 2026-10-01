// SPDX-License-Identifier: GPL-2.0-or-later

//! The translation block, `TranslationBlock` from `include/exec/translation-block.h`.
//!
//! QEMU keeps the incoming jump list as tagged pointers threaded through the source blocks and
//! guards it with the destination's `jmp_lock`; `jmp_dest[n]` is an atomic pointer whose low bit
//! marks a slot that may no longer be linked. Here the list is a vector of weak references under
//! `jmp_lock`, and each `jmp_dest[n]` is a small mutex holding the pointer and the mark. The lock
//! order is the same as QEMU's: a destination's `jmp_lock` first, then a source's `jmp_dest`.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use crate::cf;

/// The state that selects a translation, `TCGTBCPUState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TbCpuState {
    /// The guest PC.
    pub pc: u64,
    /// Target flags that change how code is translated.
    pub flags: u32,
    /// Compile flags, `CF_*`.
    pub cflags: u32,
    /// The code segment base, or whatever else the target keeps there.
    pub cs_base: u64,
}

/// The key of the block hash table: what `tb_hash_func()` hashes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TbKey {
    pub(crate) phys_pc: u64,
    pub(crate) pc: u64,
    pub(crate) flags: u32,
    pub(crate) cs_base: u64,
    pub(crate) cflags: u32,
}

/// One slot of `jmp_dest`: where the slot is chained to, and the mark bit.
#[derive(Debug, Default)]
pub(crate) struct JmpDest {
    pub(crate) dest: Option<Weak<Tb>>,
    pub(crate) mark: bool,
}

impl JmpDest {
    pub(crate) fn same(&self, other_dest: &Option<Weak<Tb>>, other_mark: bool) -> bool {
        let same_ptr = match (&self.dest, other_dest) {
            (None, None) => true,
            (Some(a), Some(b)) => Weak::ptr_eq(a, b),
            _ => false,
        };
        same_ptr && self.mark == other_mark
    }
}

/// A translated block of guest code.
pub struct Tb {
    /// A unique nonzero multiple of four. Generated code uses it as the block pointer in
    /// `exit_tb`, so the exit index fits in the low two bits.
    pub id: u64,
    /// The guest PC of the first instruction; zero with `CF_PCREL`.
    pub pc: u64,
    /// `cs_base`.
    pub cs_base: u64,
    /// Target flags.
    pub flags: u32,
    cflags: AtomicU32,
    /// Bytes of guest code covered.
    pub size: u32,
    /// Number of guest instructions.
    pub icount: u16,
    /// The `ram_addr` of the first byte, and of the second page if the block crosses into one;
    /// `u64::MAX` stands for QEMU's -1.
    pub page_addr: [u64; 2],
    /// Which `goto_tb` slots the code uses, from `jmp_reset_offset`.
    pub goto_tb_used: [bool; 2],
    /// The backend's code.
    pub(crate) code: Box<dyn Any + Send + Sync>,
    /// Size charged to the code region.
    pub(crate) code_size: usize,
    /// Incoming jumps: the source block and its slot.
    pub(crate) jmp_lock: Mutex<Vec<(Weak<Tb>, usize)>>,
    /// Outgoing jumps.
    pub(crate) jmp_dest: [Mutex<JmpDest>; 2],
}

impl fmt::Debug for Tb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tb")
            .field("id", &self.id)
            .field("pc", &format_args!("{:#x}", self.pc))
            .field("cs_base", &self.cs_base)
            .field("flags", &format_args!("{:#x}", self.flags))
            .field("cflags", &format_args!("{:#x}", self.cflags()))
            .field("size", &self.size)
            .field("icount", &self.icount)
            .field("page_addr", &self.page_addr)
            .finish_non_exhaustive()
    }
}

pub(crate) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Tb {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u64,
        s: TbCpuState,
        size: u32,
        icount: u16,
        page_addr: [u64; 2],
        goto_tb_used: [bool; 2],
        code: Box<dyn Any + Send + Sync>,
        code_size: usize,
    ) -> Tb {
        Tb {
            id,
            pc: if s.cflags & cf::PCREL != 0 { 0 } else { s.pc },
            cs_base: s.cs_base,
            flags: s.flags,
            cflags: AtomicU32::new(s.cflags),
            size,
            icount,
            page_addr,
            goto_tb_used,
            code,
            code_size,
            jmp_lock: Mutex::new(Vec::new()),
            jmp_dest: [Mutex::new(JmpDest::default()), Mutex::new(JmpDest::default())],
        }
    }

    /// `tb_cflags()`.
    pub fn cflags(&self) -> u32 {
        self.cflags.load(Ordering::Acquire)
    }

    pub(crate) fn set_invalid(&self) {
        self.cflags.fetch_or(cf::INVALID, Ordering::AcqRel);
    }

    /// Whether the block has been invalidated.
    pub fn is_invalid(&self) -> bool {
        self.cflags() & cf::INVALID != 0
    }

    /// `tb_page_addr0()`.
    pub fn page_addr0(&self) -> u64 {
        self.page_addr[0]
    }

    /// `tb_page_addr1()`.
    pub fn page_addr1(&self) -> u64 {
        self.page_addr[1]
    }

    /// The backend's code, for the backend to downcast.
    pub fn code(&self) -> &(dyn Any + Send + Sync) {
        &*self.code
    }

    /// The block `goto_tb` slot `n` is chained to, if any.
    pub fn jmp_dest(&self, n: usize) -> Option<Arc<Tb>> {
        lock(&self.jmp_dest[n]).dest.as_ref().and_then(Weak::upgrade)
    }

    pub(crate) fn key(&self) -> TbKey {
        let cflags = self.cflags() & !cf::INVALID;
        TbKey {
            phys_pc: self.page_addr[0],
            pc: if cflags & cf::PCREL != 0 { 0 } else { self.pc },
            flags: self.flags,
            cs_base: self.cs_base,
            cflags,
        }
    }
}
