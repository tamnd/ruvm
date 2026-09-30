// SPDX-License-Identifier: GPL-2.0-or-later

//! Cache descriptions and the CPUID encoders for leaves 2, 4, 0x80000005,
//! 0x80000006 and 0x8000001D, from `target/i386/cpu.c`.

use super::models::ModelCache;
use super::topo::{TopoLevel, X86CpuTopoInfo};

const KIB: u32 = 1024;
const MIB: u32 = 1024 * 1024;

/// Cache type, as `CacheType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// Data cache.
    Data,
    /// Instruction cache.
    Instruction,
    /// Unified cache.
    Unified,
}

impl CacheType {
    /// `CACHE_TYPE()` encoding for leaf 4 and 0x8000001D.
    fn leaf4(self) -> u32 {
        match self {
            CacheType::Data => 1,
            CacheType::Instruction => 2,
            CacheType::Unified => 3,
        }
    }
}

/// One cache, as `CPUCacheInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheInfo {
    /// Type.
    pub kind: CacheType,
    /// Level, 1 to 3.
    pub level: u32,
    /// Size in bytes.
    pub size: u32,
    /// Line size in bytes.
    pub line_size: u32,
    /// Ways of associativity.
    pub associativity: u32,
    /// Physical line partitions.
    pub partitions: u32,
    /// Number of sets.
    pub sets: u32,
    /// Lines per tag (AMD only).
    pub lines_per_tag: u32,
    /// Self-initializing.
    pub self_init: bool,
    /// WBINVD/INVD is not guaranteed to act on lower levels of sharing
    /// threads.
    pub no_invd_sharing: bool,
    /// Inclusive of lower levels.
    pub inclusive: bool,
    /// Uses a complex function to index the cache.
    pub complex_indexing: bool,
    /// Level of the topology the cache is shared at.
    pub share_level: TopoLevel,
}

/// A set of caches, as `CPUCaches`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuCaches {
    /// L1 data.
    pub l1d: CacheInfo,
    /// L1 instruction.
    pub l1i: CacheInfo,
    /// L2.
    pub l2: CacheInfo,
    /// L3.
    pub l3: CacheInfo,
}

const fn cache(
    kind: CacheType,
    level: u32,
    size: u32,
    associativity: u32,
    sets: u32,
    lines_per_tag: u32,
    share_level: TopoLevel,
) -> CacheInfo {
    CacheInfo {
        kind,
        level,
        size,
        line_size: 64,
        associativity,
        partitions: 1,
        sets,
        lines_per_tag,
        self_init: false,
        no_invd_sharing: false,
        inclusive: false,
        complex_indexing: false,
        share_level,
    }
}

const fn flags(
    mut c: CacheInfo,
    self_init: bool,
    no_invd: bool,
    incl: bool,
    cplx: bool,
) -> CacheInfo {
    c.self_init = self_init;
    c.no_invd_sharing = no_invd;
    c.inclusive = incl;
    c.complex_indexing = cplx;
    c
}

const LEGACY_L3: CacheInfo = flags(
    cache(CacheType::Unified, 3, 16 * MIB, 16, 16384, 1, TopoLevel::Die),
    true,
    false,
    true,
    true,
);

/// `legacy_amd_cache_info`.
pub const LEGACY_AMD_CACHES: CpuCaches = CpuCaches {
    l1d: flags(
        cache(CacheType::Data, 1, 64 * KIB, 2, 512, 1, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l1i: flags(
        cache(CacheType::Instruction, 1, 64 * KIB, 2, 512, 1, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l2: cache(CacheType::Unified, 2, 512 * KIB, 16, 512, 1, TopoLevel::Core),
    l3: LEGACY_L3,
};

/// `legacy_intel_cache_info`.
pub const LEGACY_INTEL_CACHES: CpuCaches = CpuCaches {
    l1d: flags(
        cache(CacheType::Data, 1, 32 * KIB, 8, 64, 0, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l1i: flags(
        cache(CacheType::Instruction, 1, 32 * KIB, 8, 64, 0, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l2: flags(
        cache(CacheType::Unified, 2, 4 * MIB, 16, 4096, 0, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l3: LEGACY_L3,
};

/// `legacy_intel_cpuid2_cache_info`, used for leaf 2 when
/// `x-consistent-cache` is off.
pub const LEGACY_INTEL_CPUID2_CACHES: CpuCaches = CpuCaches {
    l2: flags(
        cache(CacheType::Unified, 2, 2 * MIB, 8, 4096, 0, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    ..LEGACY_INTEL_CACHES
};

/// `epyc_cache_info`.
pub const EPYC_CACHES: CpuCaches = CpuCaches {
    l1d: flags(
        cache(CacheType::Data, 1, 32 * KIB, 8, 64, 1, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l1i: flags(
        cache(CacheType::Instruction, 1, 64 * KIB, 4, 256, 1, TopoLevel::Core),
        true,
        true,
        false,
        false,
    ),
    l2: cache(CacheType::Unified, 2, 512 * KIB, 8, 1024, 1, TopoLevel::Core),
    l3: flags(
        cache(CacheType::Unified, 3, 8 * MIB, 16, 8192, 1, TopoLevel::Die),
        true,
        false,
        true,
        true,
    ),
};

/// `epyc_v4_cache_info`: EPYC with `complex_indexing` off on L3.
pub const EPYC_V4_CACHES: CpuCaches =
    CpuCaches { l3: flags(EPYC_CACHES.l3, true, false, true, false), ..EPYC_CACHES };

/// `epyc_v5_cache_info`.
pub const EPYC_V5_CACHES: CpuCaches = CpuCaches {
    l1d: flags(EPYC_CACHES.l1d, true, false, false, false),
    l1i: flags(EPYC_CACHES.l1i, true, false, false, false),
    l2: flags(EPYC_CACHES.l2, true, false, true, false),
    l3: flags(EPYC_CACHES.l3, true, true, false, false),
};

impl ModelCache {
    /// The caches this name refers to.
    pub fn caches(self) -> CpuCaches {
        match self {
            ModelCache::Epyc => EPYC_CACHES,
            ModelCache::EpycV4 => EPYC_V4_CACHES,
            ModelCache::EpycV5 => EPYC_V5_CACHES,
        }
    }
}

/// `cpuid2_cache_descriptors[]` as (descriptor, level, type, size,
/// associativity, line size), in index order.
const CPUID2_DESCRIPTORS: &[(u8, u32, CacheType, u32, u32, u32)] = &[
    (0x06, 1, CacheType::Instruction, 8192, 4, 32),
    (0x08, 1, CacheType::Instruction, 16384, 4, 32),
    (0x09, 1, CacheType::Instruction, 32768, 4, 64),
    (0x0A, 1, CacheType::Data, 8192, 2, 32),
    (0x0C, 1, CacheType::Data, 16384, 4, 32),
    (0x0D, 1, CacheType::Data, 16384, 4, 64),
    (0x0E, 1, CacheType::Data, 24576, 6, 64),
    (0x1D, 2, CacheType::Unified, 131072, 2, 64),
    (0x21, 2, CacheType::Unified, 262144, 8, 64),
    (0x24, 2, CacheType::Unified, 1048576, 16, 64),
    (0x2C, 1, CacheType::Data, 32768, 8, 64),
    (0x30, 1, CacheType::Instruction, 32768, 8, 64),
    (0x41, 2, CacheType::Unified, 131072, 4, 32),
    (0x42, 2, CacheType::Unified, 262144, 4, 32),
    (0x43, 2, CacheType::Unified, 524288, 4, 32),
    (0x44, 2, CacheType::Unified, 1048576, 4, 32),
    (0x45, 2, CacheType::Unified, 2097152, 4, 32),
    (0x46, 3, CacheType::Unified, 4194304, 4, 64),
    (0x47, 3, CacheType::Unified, 8388608, 8, 64),
    (0x48, 2, CacheType::Unified, 3145728, 12, 64),
    (0x49, 2, CacheType::Unified, 4194304, 16, 64),
    (0x4A, 3, CacheType::Unified, 6291456, 12, 64),
    (0x4B, 3, CacheType::Unified, 8388608, 16, 64),
    (0x4C, 3, CacheType::Unified, 12582912, 12, 64),
    (0x4D, 3, CacheType::Unified, 16777216, 16, 64),
    (0x4E, 2, CacheType::Unified, 6291456, 24, 64),
    (0x60, 1, CacheType::Data, 16384, 8, 64),
    (0x66, 1, CacheType::Data, 8192, 4, 64),
    (0x67, 1, CacheType::Data, 16384, 4, 64),
    (0x68, 1, CacheType::Data, 32768, 4, 64),
    (0x78, 2, CacheType::Unified, 1048576, 4, 64),
    (0x7D, 2, CacheType::Unified, 2097152, 8, 64),
    (0x7F, 2, CacheType::Unified, 524288, 2, 64),
    (0x80, 2, CacheType::Unified, 524288, 8, 64),
    (0x82, 2, CacheType::Unified, 262144, 8, 32),
    (0x83, 2, CacheType::Unified, 524288, 8, 32),
    (0x84, 2, CacheType::Unified, 1048576, 8, 32),
    (0x85, 2, CacheType::Unified, 2097152, 8, 32),
    (0x86, 2, CacheType::Unified, 524288, 4, 64),
    (0x87, 2, CacheType::Unified, 1048576, 8, 64),
    (0xD0, 3, CacheType::Unified, 524288, 4, 64),
    (0xD1, 3, CacheType::Unified, 1048576, 4, 64),
    (0xD2, 3, CacheType::Unified, 2097152, 4, 64),
    (0xD6, 3, CacheType::Unified, 1048576, 8, 64),
    (0xD7, 3, CacheType::Unified, 2097152, 8, 64),
    (0xD8, 3, CacheType::Unified, 4194304, 8, 64),
    (0xDC, 3, CacheType::Unified, 1572864, 12, 64),
    (0xDD, 3, CacheType::Unified, 3145728, 12, 64),
    (0xDE, 3, CacheType::Unified, 6291456, 12, 64),
    (0xE2, 3, CacheType::Unified, 2097152, 16, 64),
    (0xE3, 3, CacheType::Unified, 4194304, 16, 64),
    (0xE4, 3, CacheType::Unified, 8388608, 16, 64),
    (0xEA, 3, CacheType::Unified, 12582912, 24, 64),
    (0xEB, 3, CacheType::Unified, 18874368, 24, 64),
    (0xEC, 3, CacheType::Unified, 25165824, 24, 64),
];

/// `CACHE_DESCRIPTOR_UNAVAILABLE`.
pub const CACHE_DESCRIPTOR_UNAVAILABLE: u32 = 0xff;

/// `cpuid2_cache_descriptor()`: the first matching descriptor, or
/// [`CACHE_DESCRIPTOR_UNAVAILABLE`] with `unmatched` set.
pub fn cpuid2_descriptor(c: &CacheInfo, unmatched: &mut bool) -> u32 {
    for &(d, level, kind, size, assoc, line) in CPUID2_DESCRIPTORS {
        if level == c.level
            && kind == c.kind
            && size == c.size
            && line == c.line_size
            && assoc == c.associativity
        {
            return u32::from(d);
        }
    }
    *unmatched = true;
    CACHE_DESCRIPTOR_UNAVAILABLE
}

/// Register values of one CPUID query.
pub type Regs = [u32; 4];

/// `encode_cache_cpuid2()`. `consistent` is `x-consistent-cache`, `l3` is
/// `l3-cache`, `min_level` is `env->cpuid_min_level`.
pub fn encode_cpuid2(caches: &CpuCaches, consistent: bool, l3: bool, min_level: u32) -> Regs {
    let mut unmatched = false;
    let l1d = cpuid2_descriptor(&caches.l1d, &mut unmatched);
    let l1i = cpuid2_descriptor(&caches.l1i, &mut unmatched);
    let l2 = cpuid2_descriptor(&caches.l2, &mut unmatched);
    let l3d = cpuid2_descriptor(&caches.l3, &mut unmatched);
    if !consistent || (min_level < 4 && !unmatched) {
        let ecx = if l3 { l3d } else { 0 };
        [1, 0, ecx, (l1d << 16) | (l1i << 8) | l2]
    } else {
        [1, 0, 0, CACHE_DESCRIPTOR_UNAVAILABLE]
    }
}

fn encode_common(c: &CacheInfo, topo: &X86CpuTopoInfo) -> Regs {
    let eax = c.kind.leaf4()
        | (c.level << 5)
        | if c.self_init { 1 << 8 } else { 0 }
        | (topo.max_thread_ids_for_cache(c.share_level).min(4095) << 14);
    let ebx = (c.line_size - 1) | ((c.partitions - 1) << 12) | ((c.associativity - 1) << 22);
    let edx = u32::from(c.no_invd_sharing)
        | (u32::from(c.inclusive) << 1)
        | (u32::from(c.complex_indexing) << 2);
    [eax, ebx, c.sets - 1, edx]
}

/// `encode_cache_cpuid4()`.
pub fn encode_cpuid4(c: &CacheInfo, topo: &X86CpuTopoInfo) -> Regs {
    let mut r = encode_common(c, topo);
    r[0] |= topo.max_core_ids_in_package().min(63) << 26;
    r
}

/// `encode_cache_cpuid8000001d()`.
pub fn encode_cpuid8000001d(c: &CacheInfo, topo: &X86CpuTopoInfo) -> Regs {
    encode_common(c, topo)
}

/// `encode_cache_cpuid80000005()`.
pub fn encode_cpuid80000005(c: &CacheInfo) -> u32 {
    ((c.size / 1024) << 24) | (c.associativity << 16) | (c.lines_per_tag << 8) | c.line_size
}

/// `X86_ENC_ASSOC()`.
fn enc_assoc(a: u32) -> u32 {
    match a {
        0 | 1 => a,
        2 => 0x2,
        4 => 0x4,
        8 => 0x6,
        16 => 0x8,
        32 => 0xa,
        48 => 0xb,
        64 => 0xc,
        96 => 0xd,
        128 => 0xe,
        0xff => 0xf,
        _ => 0,
    }
}

/// `encode_cache_cpuid80000006()`, returning (ECX, EDX).
pub fn encode_cpuid80000006(l2: &CacheInfo, l3: Option<&CacheInfo>) -> (u32, u32) {
    let ecx = ((l2.size / 1024) << 16)
        | (enc_assoc(l2.associativity) << 12)
        | (l2.lines_per_tag << 8)
        | l2.line_size;
    let edx = l3.map_or(0, |l3| {
        ((l3.size / (512 * 1024)) << 18)
            | (enc_assoc(l3.associativity) << 12)
            | (l3.lines_per_tag << 8)
            | l3.line_size
    });
    (ecx, edx)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(c: &CacheInfo) {
        assert_eq!(c.size, c.line_size * c.associativity * c.partitions * c.sets);
    }

    #[test]
    fn geometry_is_consistent() {
        for cs in [
            LEGACY_AMD_CACHES,
            LEGACY_INTEL_CACHES,
            LEGACY_INTEL_CPUID2_CACHES,
            EPYC_CACHES,
            EPYC_V4_CACHES,
            EPYC_V5_CACHES,
        ] {
            check(&cs.l1d);
            check(&cs.l1i);
            check(&cs.l2);
            check(&cs.l3);
        }
    }

    #[test]
    fn legacy_intel_leaf2() {
        // L1d 0x2C, L1i 0x30, L2 4M/16 = 0x49, L3 16M/16 = 0x4D.
        assert_eq!(
            encode_cpuid2(&LEGACY_INTEL_CACHES, false, true, 0xd),
            [1, 0, 0x4d, 0x002c_3049]
        );
        // The cpuid2 variant has a 2M/8 L2 = 0x7D.
        assert_eq!(
            encode_cpuid2(&LEGACY_INTEL_CPUID2_CACHES, false, true, 0xd),
            [1, 0, 0x4d, 0x002c_307d]
        );
        // Consistent cache on a model with leaf 4 reports "use leaf 4".
        assert_eq!(encode_cpuid2(&LEGACY_INTEL_CACHES, true, true, 0xd), [1, 0, 0, 0xff]);
    }
}
