// SPDX-License-Identifier: GPL-2.0-or-later

//! Memory ordering on an AArch64 host: the host features that matter, the fence mapping to use
//! for a guest, and the instructions the mappings of [`ruvm_jit_core::memory_model`] lower to.
//!
//! - An `mb` op becomes `dmb ishld`, `dmb ishst` or `dmb ish` by QEMU's `tcg_out_mb` table
//!   ([`FenceKind::for_bar`]).
//! - A `qemu_ld` with [`ldst_flags::ACQUIRE_PC`] becomes `ldapr` (FEAT_LRCPC), or `ldapur*` for
//!   a sign extending load when FEAT_LRCPC2 is there. On a host without FEAT_LRCPC it becomes
//!   `ldar`, which is stronger (RCsc), so a block built for the RCpc mapping still runs
//!   correctly there; [`select_fence_mapping`] does not pick that mapping on such a host.
//! - A `qemu_st` with [`ldst_flags::RELEASE`] becomes `stlr`, which every ARMv8.0 host has.
//! - These single instructions are only used for accesses that generated code makes itself,
//!   through the host window of [`crate::CompiledTb::run_with_window`]. An access that goes to
//!   the service routine instead (no window, outside the window, misaligned for its size, byte
//!   swapped or 128 bits wide) is fenced conservatively: `dmb ishld` after an acquire load and
//!   `dmb ish` before a release store.
//! - With either alternative mapping, a helper call that may have side effects, and so may
//!   access guest memory, gets `dmb ishst` before it and `dmb ishld` after it. That keeps the
//!   helper's accesses ordered with the block's under the same rules (earlier stores before its
//!   stores, its loads before later accesses). QEMU's mapping is left alone: QEMU does not fence
//!   helpers. The `dmb ishst` is left out when a full barrier from an `mb` op has run since the
//!   last guest access, helper call or label, which is the case for the atomic helpers the
//!   alternative mappings already bracket.
//!
//! Differences from QEMU: QEMU's aarch64 backend never emits `ldapr` or `stlr` for plain guest
//! accesses and does not detect FEAT_LRCPC, and it puts no barriers around helper calls.

use std::sync::OnceLock;

#[cfg(doc)]
use ruvm_jit_core::memory_model::ldst_flags;
use ruvm_jit_core::memory_model::{FenceKind, FenceMapping};

use crate::asm::i;

/// `TCG_TARGET_DEFAULT_MO` for AArch64: the host orders nothing for free.
pub const TARGET_DEFAULT_MO: u32 = 0;

/// The optional AArch64 features the backend can use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostFeatures {
    /// FEAT_LRCPC: `ldapr`.
    pub lrcpc: bool,
    /// FEAT_LRCPC2: `ldapur` and `stlur`, with sign extending forms.
    pub lrcpc2: bool,
}

impl HostFeatures {
    /// No optional feature: an ARMv8.0 host.
    pub const BASELINE: HostFeatures = HostFeatures { lrcpc: false, lrcpc2: false };

    /// Both features, as on an Apple M1 or later.
    pub const ALL: HostFeatures = HostFeatures { lrcpc: true, lrcpc2: true };

    /// The features of the host this runs on, detected once. Every feature is off on a host
    /// that is not AArch64.
    pub fn detect() -> HostFeatures {
        static FEATURES: OnceLock<HostFeatures> = OnceLock::new();
        *FEATURES.get_or_init(detect_now)
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_now() -> HostFeatures {
    let lrcpc = std::arch::is_aarch64_feature_detected!("rcpc");
    let lrcpc2 = lrcpc && std::arch::is_aarch64_feature_detected!("rcpc2");
    HostFeatures { lrcpc, lrcpc2 }
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_now() -> HostFeatures {
    HostFeatures::BASELINE
}

/// The fence mapping for a guest with order `guest_mo` when `requested` is asked for, on this
/// host: [`FenceMapping::AranciniRcpc`] needs FEAT_LRCPC and becomes
/// [`FenceMapping::Risotto`] without it, and a mapping that does not fit the guest becomes
/// [`FenceMapping::Qemu`]. See [`FenceMapping::select`].
pub fn select_fence_mapping(requested: FenceMapping, guest_mo: u32) -> FenceMapping {
    requested.select(guest_mo, TARGET_DEFAULT_MO, HostFeatures::detect().lrcpc)
}

/// The `dmb` instruction for an `mb` op with argument `bar`, QEMU's `tcg_out_mb`.
pub fn dmb_for(bar: u32) -> u32 {
    match FenceKind::for_bar(bar) {
        FenceKind::Load => DMB_ISHLD,
        FenceKind::Store => DMB_ISHST,
        FenceKind::Full => DMB_ISH_FULL,
    }
}

/// `dmb ishld`.
pub const DMB_ISHLD: u32 = i::DMB_ISH | i::DMB_LD;
/// `dmb ishst`.
pub const DMB_ISHST: u32 = i::DMB_ISH | i::DMB_ST;
/// `dmb ish`.
pub const DMB_ISH_FULL: u32 = i::DMB_ISH | i::DMB_LD | i::DMB_ST;

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_jit_core::memory_model::X86_TSO;
    use ruvm_jit_core::types::mo;

    #[test]
    fn dmb_words() {
        assert_eq!(DMB_ISHLD, 0xd50339bf);
        assert_eq!(DMB_ISHST, 0xd5033abf);
        assert_eq!(DMB_ISH_FULL, 0xd5033bbf);
        assert_eq!(dmb_for(mo::LD_LD | mo::LD_ST | mo::BAR_SC), DMB_ISHLD);
        assert_eq!(dmb_for(mo::ST_ST | mo::BAR_SC), DMB_ISHST);
        assert_eq!(dmb_for(mo::LD_ST | mo::ST_ST | mo::BAR_SC), DMB_ISH_FULL);
        assert_eq!(dmb_for(mo::ALL | mo::BAR_SC), DMB_ISH_FULL);
        assert_eq!(dmb_for(0), DMB_ISH_FULL);
    }

    #[test]
    fn selection_on_this_host() {
        let f = HostFeatures::detect();
        assert!(!f.lrcpc2 || f.lrcpc);
        let want = if f.lrcpc { FenceMapping::AranciniRcpc } else { FenceMapping::Risotto };
        assert_eq!(select_fence_mapping(FenceMapping::AranciniRcpc, X86_TSO), want);
        assert_eq!(select_fence_mapping(FenceMapping::Risotto, X86_TSO), FenceMapping::Risotto);
        assert_eq!(select_fence_mapping(FenceMapping::Qemu, X86_TSO), FenceMapping::Qemu);
        assert_eq!(select_fence_mapping(FenceMapping::Risotto, 0), FenceMapping::Qemu);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    fn apple_silicon_has_rcpc() {
        // Every Apple core since the M1 implements FEAT_LRCPC and FEAT_LRCPC2.
        assert_eq!(HostFeatures::detect(), HostFeatures::ALL);
    }
}
