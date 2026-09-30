// SPDX-License-Identifier: GPL-2.0-or-later

//! XSAVE state components, as `x86_ext_save_areas[]` in
//! `target/i386/cpu.c`, plus the per-accelerator offset setup from
//! `tcg/tcg-cpu.c` and `kvm/kvm-cpu.c`.

use super::words::{
    CPUID_7_0_EBX_AVX512F, CPUID_7_0_EBX_MPX, CPUID_7_0_ECX_CET_SHSTK, CPUID_7_0_ECX_PKU,
    CPUID_7_0_EDX_AMX_TILE, CPUID_7_0_EDX_ARCH_LBR, CPUID_7_0_EDX_CET_IBT, CPUID_7_1_EDX_APXF,
    CPUID_7_1_EDX_AVX10, CPUID_EXT_AVX, CPUID_EXT_XSAVE, FEAT_1_ECX, FEAT_7_0_EBX, FEAT_7_0_ECX,
    FEAT_7_0_EDX, FEAT_7_1_EDX, FeatureWordArray,
};

/// `XSAVE_STATE_AREA_COUNT`.
pub const XSAVE_STATE_AREA_COUNT: usize = 20;

/// Size of the legacy region plus the XSAVE header.
pub const XSAVE_LEGACY_SIZE: u32 = 512 + 64;

/// One XSAVE state component, as `ExtSaveArea`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtSaveArea {
    /// Features that enable the component; any one of them is enough.
    pub features: &'static [(usize, u64)],
    /// Offset in the standard (non-compacted) layout.
    pub offset: u32,
    /// Size in bytes. Zero means the component is not supported.
    pub size: u32,
    /// `CPUID[0xd,i].ECX` as reported by the host.
    pub ecx: u32,
}

/// Set of state components, indexed by component number.
pub type ExtSaveAreas = [ExtSaveArea; XSAVE_STATE_AREA_COUNT];

const NONE: ExtSaveArea = ExtSaveArea { features: &[], offset: 0, size: 0, ecx: 0 };

const fn area(features: &'static [(usize, u64)], size: u32) -> ExtSaveArea {
    ExtSaveArea { features, offset: 0, size, ecx: 0 }
}

const XSAVE_FEAT: &[(usize, u64)] = &[(FEAT_1_ECX, CPUID_EXT_XSAVE)];
const MPX_FEAT: &[(usize, u64)] = &[(FEAT_7_0_EBX, CPUID_7_0_EBX_MPX)];
const AVX512_FEAT: &[(usize, u64)] =
    &[(FEAT_7_0_EBX, CPUID_7_0_EBX_AVX512F), (FEAT_7_1_EDX, CPUID_7_1_EDX_AVX10)];
const CET_FEAT: &[(usize, u64)] =
    &[(FEAT_7_0_ECX, CPUID_7_0_ECX_CET_SHSTK), (FEAT_7_0_EDX, CPUID_7_0_EDX_CET_IBT)];
const AMX_FEAT: &[(usize, u64)] = &[(FEAT_7_0_EDX, CPUID_7_0_EDX_AMX_TILE)];

/// `x86_ext_save_areas[]` before any accelerator fills in offsets.
pub const DEFAULT_EXT_SAVE_AREAS: ExtSaveAreas = [
    area(XSAVE_FEAT, XSAVE_LEGACY_SIZE),
    area(XSAVE_FEAT, XSAVE_LEGACY_SIZE),
    area(&[(FEAT_1_ECX, CPUID_EXT_AVX)], 256),
    area(MPX_FEAT, 64),
    area(MPX_FEAT, 64),
    area(AVX512_FEAT, 64),
    area(AVX512_FEAT, 512),
    area(AVX512_FEAT, 1024),
    NONE,
    area(&[(FEAT_7_0_ECX, CPUID_7_0_ECX_PKU)], 8),
    NONE,
    area(CET_FEAT, 16),
    area(CET_FEAT, 24),
    NONE,
    NONE,
    area(&[(FEAT_7_0_EDX, CPUID_7_0_EDX_ARCH_LBR)], 0x328),
    NONE,
    area(AMX_FEAT, 64),
    area(AMX_FEAT, 0x2000),
    area(&[(FEAT_7_1_EDX, CPUID_7_1_EDX_APXF)], 128),
];

/// `x86_cpu_init_xsave()`: drop components (above SSE) that the
/// accelerator cannot handle at all.
pub fn trim_unsupported(areas: &mut ExtSaveAreas, supported_xcr0_xss: u64) {
    for (i, esa) in areas.iter_mut().enumerate().skip(2) {
        if supported_xcr0_xss & (1 << i) == 0 {
            esa.size = 0;
        }
    }
}

/// `x86_tcg_cpu_xsave_init()`: offsets of the fields in `X86XSaveArea`.
pub fn tcg_init_offsets(areas: &mut ExtSaveAreas) {
    const OFFSETS: [(usize, u32); 9] =
        [(0, 0), (1, 0), (2, 576), (3, 960), (4, 1024), (5, 1088), (6, 1152), (7, 1664), (9, 2688)];
    for (i, off) in OFFSETS {
        areas[i].offset = off;
    }
}

/// `kvm_cpu_xsave_init()`: offsets and ECX from the host's
/// `CPUID[0xd,i]`. `host` returns EAX, EBX, ECX for a subleaf.
pub fn kvm_init_offsets(areas: &mut ExtSaveAreas, host: impl Fn(u32) -> (u32, u32, u32)) {
    areas[0].offset = 0;
    areas[1].offset = 0;
    for (i, esa) in areas.iter_mut().enumerate().skip(2) {
        if esa.size == 0 {
            continue;
        }
        let (eax, ebx, ecx) = host(i as u32);
        if eax != 0 {
            esa.offset = ebx;
            esa.ecx = ecx;
        }
    }
}

/// `cpuid_has_xsave_feature()`.
pub fn has_xsave_feature(features: &FeatureWordArray, esa: &ExtSaveArea) -> bool {
    esa.size != 0 && esa.features.iter().any(|&(w, m)| features[w] & m != 0)
}

/// `xsave_area_size()`.
pub fn xsave_area_size(areas: &ExtSaveAreas, mask: u64, compacted: bool) -> u32 {
    let mut ret = u64::from(areas[0].size);
    for (i, esa) in areas.iter().enumerate().skip(2) {
        if (mask >> i) & 1 != 0 {
            let offset = if compacted { ret } else { u64::from(esa.offset) };
            ret = ret.max(offset + u64::from(esa.size));
        }
    }
    ret as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tcg_sizes() {
        let mut a = DEFAULT_EXT_SAVE_AREAS;
        tcg_init_offsets(&mut a);
        // x87 + SSE + AVX: 576 + 256.
        assert_eq!(xsave_area_size(&a, 0x7, false), 832);
        // Up to PKRU in the standard layout: 2688 + 8.
        assert_eq!(xsave_area_size(&a, 0x2ff, false), 2696);
        // Compacted, just x87/SSE/AVX/PKRU: 576 + 256 + 8.
        assert_eq!(xsave_area_size(&a, 0x207, true), 840);
    }
}
