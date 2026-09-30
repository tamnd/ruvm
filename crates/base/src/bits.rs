// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit field helpers with the semantics of include/qemu/bitops.h. Device models and decoders use
//! these constantly, and keeping QEMU's argument order makes ported code easy to check.

/// `extract32()`: `length` bits of `value` starting at bit `start`.
#[inline]
pub const fn extract32(value: u32, start: u32, length: u32) -> u32 {
    debug_assert!(length > 0 && length <= 32 - start);
    (value >> start) & (u32::MAX >> (32 - length))
}

#[inline]
pub const fn extract64(value: u64, start: u32, length: u32) -> u64 {
    debug_assert!(length > 0 && length <= 64 - start);
    (value >> start) & (u64::MAX >> (64 - length))
}

/// `sextract32()`: like `extract32()` with the result sign extended from the top bit of the field.
#[inline]
pub const fn sextract32(value: u32, start: u32, length: u32) -> i32 {
    debug_assert!(length > 0 && length <= 32 - start);
    ((value << (32 - length - start)) as i32) >> (32 - length)
}

#[inline]
pub const fn sextract64(value: u64, start: u32, length: u32) -> i64 {
    debug_assert!(length > 0 && length <= 64 - start);
    ((value << (64 - length - start)) as i64) >> (64 - length)
}

/// `deposit32()`: `value` with `length` bits starting at `start` replaced by the low bits of
/// `field`.
#[inline]
pub const fn deposit32(value: u32, start: u32, length: u32, field: u32) -> u32 {
    debug_assert!(length > 0 && length <= 32 - start);
    let mask = (u32::MAX >> (32 - length)) << start;
    (value & !mask) | ((field << start) & mask)
}

#[inline]
pub const fn deposit64(value: u64, start: u32, length: u32, field: u64) -> u64 {
    debug_assert!(length > 0 && length <= 64 - start);
    let mask = (u64::MAX >> (64 - length)) << start;
    (value & !mask) | ((field << start) & mask)
}

/// `MAKE_64BIT_MASK(shift, length)`.
#[inline]
pub const fn mask64(shift: u32, length: u32) -> u64 {
    (u64::MAX >> (64 - length)) << shift
}

/// `QEMU_ALIGN_DOWN` and `QEMU_ALIGN_UP` for power of two alignments.
#[inline]
pub const fn align_down(n: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    n & !(align - 1)
}

#[inline]
pub const fn align_up(n: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (n + align - 1) & !(align - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_and_deposit_round_trip() {
        assert_eq!(extract32(0xdead_beef, 8, 8), 0xbe);
        assert_eq!(extract32(0xdead_beef, 0, 32), 0xdead_beef);
        assert_eq!(extract64(u64::MAX, 63, 1), 1);
        assert_eq!(deposit32(0xdead_beef, 8, 8, 0x12), 0xdead_12ef);
        assert_eq!(deposit64(0, 60, 4, 0xff), 0xf000_0000_0000_0000);
        for v in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
            for start in 0..32 {
                for len in 1..=(32 - start) {
                    assert_eq!(deposit32(v, start, len, extract32(v, start, len)), v);
                }
            }
        }
    }

    #[test]
    fn sign_extension() {
        assert_eq!(sextract32(0x0000_0f00, 8, 4), -1);
        assert_eq!(sextract32(0x0000_0700, 8, 4), 7);
        assert_eq!(sextract64(0x8000_0000_0000_0000, 63, 1), -1);
        assert_eq!(sextract64(0xffff_ffff, 0, 32), -1);
    }

    #[test]
    fn masks_and_alignment() {
        assert_eq!(mask64(4, 8), 0xff0);
        assert_eq!(mask64(0, 64), u64::MAX);
        assert_eq!(align_down(0x1fff, 0x1000), 0x1000);
        assert_eq!(align_up(0x1001, 0x1000), 0x2000);
        assert_eq!(align_up(0x1000, 0x1000), 0x1000);
    }
}
