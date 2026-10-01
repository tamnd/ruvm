// SPDX-License-Identifier: GPL-2.0-or-later

//! The instruction set extensions the backend may use, QEMU's `have_*` flags that
//! `tcg_target_init` sets from `cpuid` in `tcg/x86_64/tcg-target.c.inc`.

/// Optional x86-64 extensions. SSE2 is part of x86-64 and always used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HostFeatures {
    /// SSSE3: `pabs*`.
    pub ssse3: bool,
    /// SSE4.1: `pmulld`, `pcmpeqq` and the remaining `pmin*` and `pmax*`.
    pub sse41: bool,
    /// SSE4.2: `pcmpgtq`.
    pub sse42: bool,
    /// AVX: VEX encoded three operand forms for 64 and 128-bit vectors, `have_avx1`.
    pub avx: bool,
    /// AVX2: 256-bit integer vectors, `vpbroadcast*` and the shifts by vector, `have_avx2`.
    pub avx2: bool,
    /// BMI1: `andn`, `have_bmi1`.
    pub bmi1: bool,
    /// BMI2: `shlx`, `shrx` and `sarx`, `have_bmi2`.
    pub bmi2: bool,
    /// LZCNT and TZCNT, `have_lzcnt` (QEMU tests TZCNT through BMI1; this port asks for both).
    pub lzcnt: bool,
    /// POPCNT, `have_popcnt`.
    pub popcnt: bool,
}

impl HostFeatures {
    /// Plain x86-64: SSE2 and nothing else.
    pub const BASELINE: HostFeatures = HostFeatures {
        ssse3: false,
        sse41: false,
        sse42: false,
        avx: false,
        avx2: false,
        bmi1: false,
        bmi2: false,
        lzcnt: false,
        popcnt: false,
    };

    /// Every extension the backend knows about.
    pub const ALL: HostFeatures = HostFeatures {
        ssse3: true,
        sse41: true,
        sse42: true,
        avx: true,
        avx2: true,
        bmi1: true,
        bmi2: true,
        lzcnt: true,
        popcnt: true,
    };

    /// The extensions of the CPU this runs on; [`HostFeatures::BASELINE`] on other hosts.
    pub fn detect() -> HostFeatures {
        #[cfg(target_arch = "x86_64")]
        {
            HostFeatures {
                ssse3: std::arch::is_x86_feature_detected!("ssse3"),
                sse41: std::arch::is_x86_feature_detected!("sse4.1"),
                sse42: std::arch::is_x86_feature_detected!("sse4.2"),
                avx: std::arch::is_x86_feature_detected!("avx"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                bmi1: std::arch::is_x86_feature_detected!("bmi1"),
                bmi2: std::arch::is_x86_feature_detected!("bmi2"),
                lzcnt: std::arch::is_x86_feature_detected!("lzcnt")
                    && std::arch::is_x86_feature_detected!("bmi1"),
                popcnt: std::arch::is_x86_feature_detected!("popcnt"),
            }
            .normalized()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            HostFeatures::BASELINE
        }
    }

    /// The same set with every extension whose prerequisites are missing turned off: AVX2
    /// needs AVX, AVX needs SSE4.2, SSE4.2 needs SSE4.1 and SSE4.1 needs SSSE3.
    pub fn normalized(self) -> HostFeatures {
        let mut f = self;
        f.sse41 &= f.ssse3;
        f.sse42 &= f.sse41;
        f.avx &= f.sse42;
        f.avx2 &= f.avx;
        f
    }

    /// The extensions in both sets.
    pub fn intersect(self, other: HostFeatures) -> HostFeatures {
        HostFeatures {
            ssse3: self.ssse3 && other.ssse3,
            sse41: self.sse41 && other.sse41,
            sse42: self.sse42 && other.sse42,
            avx: self.avx && other.avx,
            avx2: self.avx2 && other.avx2,
            bmi1: self.bmi1 && other.bmi1,
            bmi2: self.bmi2 && other.bmi2,
            lzcnt: self.lzcnt && other.lzcnt,
            popcnt: self.popcnt && other.popcnt,
        }
        .normalized()
    }
}
