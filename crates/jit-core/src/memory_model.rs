// SPDX-License-Identifier: MIT OR Apache-2.0

//! How a guest's memory ordering is kept on a host with a weaker one: the fence mapping.
//!
//! QEMU's `tcg_gen_req_mo` puts a barrier in front of every guest load and store, asking for
//! the orders the guest needs and the host does not give (`guest_mo & !TCG_TARGET_DEFAULT_MO`).
//! That is [`FenceMapping::Qemu`], the default. For an x86 guest on an Arm host it puts a
//! `dmb ishld` before each load and a `dmb ish` before each store, because the barrier before a
//! store has to order earlier loads (`LD_ST`) as well as earlier stores (`ST_ST`).
//!
//! Two other mappings are available for guests whose order needs nothing from store to load,
//! which covers x86-TSO (`mo::ALL & !mo::ST_LD`):
//!
//! - [`FenceMapping::Risotto`], from Risotto (Gao et al., ASPLOS 2023, "Risotto: A Dynamic
//!   Binary Translator for Weak Memory Model Architectures", doi 10.1145/3567955.3567962). A
//!   load is followed by a barrier ordering it before every later access (`LD_LD | LD_ST`,
//!   `dmb ishld` on Arm), and a store is preceded by a barrier ordering earlier stores before
//!   it (`ST_ST`, `dmb ishst`). The optimizer's `fold_mb` merges a load's trailing barrier with
//!   the next store's leading one when no memory access, call or block end sits between them,
//!   giving one barrier of the union strength (`ld; F(rm); F(ww); st` becomes
//!   `ld; F(rm|ww); st`, a single `dmb ish`). Atomic read-modify-write and compare and swap
//!   ops are bracketed by full barriers.
//! - [`FenceMapping::AranciniRcpc`], after Arancini (Reimers et al., ASPLOS 2026,
//!   doi 10.1145/3779212.3790127) and the RCpc refinement that the spec keeps behind a flag.
//!   No barrier ops are emitted for plain accesses. Instead each `qemu_ld` that must be ordered
//!   before later accesses carries [`ldst_flags::ACQUIRE_PC`] and each `qemu_st` that must be
//!   ordered after earlier ones carries [`ldst_flags::RELEASE`] in [`crate::ir::Op::flags`]. An
//!   Arm backend lowers them to `ldapr` (FEAT_LRCPC) and `stlr`. A load-acquire-PC orders its
//!   load before every later access, a store-release orders every earlier access before its
//!   store, and RCpc lets a store-release pass a later load-acquire-PC, which is exactly the
//!   store to load reordering TSO allows. Atomics are bracketed by full barriers as for
//!   Risotto.
//!
//! Which mapping a block uses is [`FuncConfig::fence_mapping`](crate::ir::FuncConfig), set by
//! the runtime from [`FenceMapping::select`] for the guest and host pair. The builder checks it
//! again with [`FenceMapping::effective`], so a mapping that does not fit the guest silently
//! falls back to QEMU's.
//!
//! # Conservative choices
//!
//! - Only guests that need no `ST_LD` order use the alternative mappings; anything else (a
//!   sequentially consistent guest, say) keeps QEMU's mapping.
//! - [`FenceMapping::AranciniRcpc`] is opt in: [`FenceMapping::preferred`] never returns it,
//!   because the spec notes that the RCpc lowering is a ruvm addition the Arancini proofs do not
//!   cover. Without FEAT_LRCPC on the host, [`FenceMapping::select`] turns it into
//!   [`FenceMapping::Risotto`].
//! - A load gets [`ldst_flags::ACQUIRE_PC`] when the guest needs `LD_LD` or `LD_ST`, and a
//!   store gets [`ldst_flags::RELEASE`] when it needs `ST_ST` or `LD_ST`; `LD_ST` is covered
//!   twice.
//! - Both alternative mappings bracket every atomic read-modify-write and compare and swap with
//!   full barriers (`mb ALL`), in parallel blocks where the op is a helper call and in serial
//!   blocks where it is a load and a store, as Risotto prescribes. QEMU's mapping is left as it
//!   is: no barrier around the atomic helpers.
//! - The non-atomic compare and swap (`gen_nonatomic_cmpxchg_*`) is a plain load and store and
//!   gets the plain access barriers only, as in QEMU.
//!
//! # Differences from QEMU
//!
//! QEMU only has the first mapping. The others, the flag bits on `qemu_ld` and `qemu_st` and
//! the barriers around atomics are ruvm additions; with [`FenceMapping::Qemu`] the builder emits
//! exactly what QEMU does.

use crate::types::mo;

/// How the orders a guest needs are enforced on the host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FenceMapping {
    /// `tcg_gen_req_mo`: a barrier before each access. The default.
    #[default]
    Qemu,
    /// A barrier after each load and before each store, full barriers around atomics.
    Risotto,
    /// Acquire-PC loads and release stores, full barriers around atomics. Needs FEAT_LRCPC on
    /// an Arm host.
    AranciniRcpc,
}

impl FenceMapping {
    /// Every mapping.
    pub const ALL: [FenceMapping; 3] =
        [FenceMapping::Qemu, FenceMapping::Risotto, FenceMapping::AranciniRcpc];

    /// A short lower case name: `qemu`, `risotto` or `arancini-rcpc`.
    pub const fn name(self) -> &'static str {
        match self {
            FenceMapping::Qemu => "qemu",
            FenceMapping::Risotto => "risotto",
            FenceMapping::AranciniRcpc => "arancini-rcpc",
        }
    }

    /// The mapping called `s`, as [`FenceMapping::name`] prints it. `arancini` is accepted for
    /// `arancini-rcpc`.
    pub fn from_name(s: &str) -> Option<FenceMapping> {
        match s {
            "qemu" => Some(FenceMapping::Qemu),
            "risotto" => Some(FenceMapping::Risotto),
            "arancini" | "arancini-rcpc" => Some(FenceMapping::AranciniRcpc),
            _ => None,
        }
    }

    /// True if this mapping can enforce `needed`, a set of `mo` order bits. The alternative
    /// mappings need some order to enforce and cannot enforce `ST_LD`.
    pub const fn fits(self, needed: u32) -> bool {
        match self {
            FenceMapping::Qemu => true,
            FenceMapping::Risotto | FenceMapping::AranciniRcpc => {
                needed & mo::ALL != 0 && needed & mo::ST_LD == 0
            }
        }
    }

    /// The mapping a block actually uses when this one is asked for: `self` if it
    /// [`fits`](FenceMapping::fits) the orders the guest needs on this host, else
    /// [`FenceMapping::Qemu`].
    pub const fn effective(self, guest_mo: u32, host_mo: u32) -> FenceMapping {
        if self.fits(needed_mo(guest_mo, host_mo)) { self } else { FenceMapping::Qemu }
    }

    /// The mapping to use when `self` is asked for, for a guest with order `guest_mo` on a host
    /// that gives `host_mo` for free and has (`host_rcpc`) or lacks FEAT_LRCPC. Without RCpc,
    /// [`FenceMapping::AranciniRcpc`] becomes [`FenceMapping::Risotto`]; a mapping that does
    /// not fit becomes [`FenceMapping::Qemu`].
    pub const fn select(self, guest_mo: u32, host_mo: u32, host_rcpc: bool) -> FenceMapping {
        let m = match self {
            FenceMapping::AranciniRcpc if !host_rcpc => FenceMapping::Risotto,
            m => m,
        };
        m.effective(guest_mo, host_mo)
    }

    /// The mapping ruvm picks by itself for a guest and host pair: [`FenceMapping::Risotto`]
    /// where it fits, else [`FenceMapping::Qemu`]. [`FenceMapping::AranciniRcpc`] has to be
    /// asked for.
    pub const fn preferred(guest_mo: u32, host_mo: u32) -> FenceMapping {
        FenceMapping::Risotto.effective(guest_mo, host_mo)
    }
}

impl std::fmt::Display for FenceMapping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The orders a guest with `guest_mo` needs barriers for on a host that gives `host_mo`, the
/// mask `tcg_gen_req_mo` applies.
pub const fn needed_mo(guest_mo: u32, host_mo: u32) -> u32 {
    guest_mo & !host_mo & mo::ALL
}

/// The x86 guest order, `TCG_MO_ALL & ~TCG_MO_ST_LD` in `target/i386/tcg/tcg-cpu.c`.
pub const X86_TSO: u32 = mo::ALL & !mo::ST_LD;

/// Bits of [`crate::ir::Op::flags`] on `qemu_ld`, `qemu_st`, `qemu_ld2` and `qemu_st2`.
pub mod ldst_flags {
    /// The access size, `MemOp::SIZE`, as QEMU keeps it in `TCGOP_FLAGS`.
    pub const SIZE: u8 = 0x07;
    /// The load must be ordered before every later access: a load-acquire-PC (`ldapr`).
    pub const ACQUIRE_PC: u8 = 0x40;
    /// Every earlier access must be ordered before the store: a store-release (`stlr`).
    pub const RELEASE: u8 = 0x80;
}

/// What the host barrier for an `mb` op with `bar` must order, reduced to the three kinds of
/// barrier Arm has. This is the table of QEMU's aarch64 `tcg_out_mb`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FenceKind {
    /// Orders earlier loads before later loads and stores, `dmb ishld`.
    Load,
    /// Orders earlier stores before later stores, `dmb ishst`.
    Store,
    /// Orders everything, `dmb ish`.
    Full,
}

impl FenceKind {
    /// The barrier kind for `mb` argument `bar`: `ST_ST` alone is [`FenceKind::Store`], any
    /// non-empty mix of `LD_LD` and `LD_ST` is [`FenceKind::Load`], and anything else,
    /// including no order bits at all, is [`FenceKind::Full`].
    pub const fn for_bar(bar: u32) -> FenceKind {
        let a = bar & mo::ALL;
        if a == mo::ST_ST {
            FenceKind::Store
        } else if a != 0 && a & !(mo::LD_LD | mo::LD_ST) == 0 {
            FenceKind::Load
        } else {
            FenceKind::Full
        }
    }
}
