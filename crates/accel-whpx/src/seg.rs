// SPDX-License-Identifier: GPL-2.0-or-later

//! Segment registers between QEMU's `SegmentCache` and `WHV_X64_SEGMENT_REGISTER`,
//! `whpx_seg_q2h()` and `whpx_seg_h2q()`.

/// `DESC_TYPE_SHIFT`: where the access byte starts in `SegmentCache.flags`.
pub const DESC_TYPE_SHIFT: u32 = 8;

/// Attribute bits of `WHV_X64_SEGMENT_REGISTER`, the descriptor's access byte and flags.
pub mod attr {
    /// `SegmentType`, bits 0 to 3.
    pub const TYPE: u16 = 0xf;
    /// `NonSystemSegment`.
    pub const S: u16 = 1 << 4;
    /// `DescriptorPrivilegeLevel`, bits 5 and 6.
    pub const DPL_SHIFT: u16 = 5;
    /// `Present`.
    pub const P: u16 = 1 << 7;
    /// `Available`.
    pub const AVL: u16 = 1 << 12;
    /// `Long`.
    pub const L: u16 = 1 << 13;
    /// `Default`.
    pub const DB: u16 = 1 << 14;
    /// `Granularity`.
    pub const G: u16 = 1 << 15;
}

/// QEMU's `SegmentCache`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Segment {
    /// The selector.
    pub selector: u16,
    /// The base.
    pub base: u64,
    /// The limit, in bytes.
    pub limit: u32,
    /// The descriptor flags as they sit in the descriptor's high word.
    pub flags: u32,
}

/// `WHV_X64_SEGMENT_REGISTER`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HvSegment {
    /// `Base`.
    pub base: u64,
    /// `Limit`.
    pub limit: u32,
    /// `Selector`.
    pub selector: u16,
    /// `Attributes`.
    pub attributes: u16,
}

impl HvSegment {
    /// The 128 bit register value, low half then high half.
    pub fn pack(&self) -> [u64; 2] {
        let hi = u64::from(self.limit)
            | u64::from(self.selector) << 32
            | u64::from(self.attributes) << 48;
        [self.base, hi]
    }

    /// Reads a 128 bit register value.
    pub fn unpack(v: [u64; 2]) -> HvSegment {
        HvSegment {
            base: v[0],
            limit: v[1] as u32,
            selector: (v[1] >> 32) as u16,
            attributes: (v[1] >> 48) as u16,
        }
    }
}

impl Segment {
    /// `whpx_seg_q2h()`. In virtual 8086 mode every data and code segment is a present,
    /// accessed, writable, DPL 3 data segment.
    pub fn to_hv(&self, v86: bool) -> HvSegment {
        let attributes = if v86 {
            3 | attr::S | 3 << attr::DPL_SHIFT | attr::P
        } else {
            (self.flags >> DESC_TYPE_SHIFT) as u16
        };
        HvSegment { base: self.base, limit: self.limit, selector: self.selector, attributes }
    }

    /// `whpx_seg_h2q()`.
    pub fn from_hv(hs: &HvSegment) -> Segment {
        Segment {
            selector: hs.selector,
            base: hs.base,
            limit: hs.limit,
            flags: u32::from(hs.attributes) << DESC_TYPE_SHIFT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_mode_code_segment() {
        // A 64 bit code segment: type 0xb, S, P, L, G.
        let flags = 0x00a0_9b00;
        let s = Segment { selector: 0x10, base: 0, limit: 0xffff_ffff, flags };
        let hv = s.to_hv(false);
        assert_eq!(hv.attributes, 0xa09b);
        assert_eq!(hv.attributes & attr::TYPE, 0xb);
        assert!(hv.attributes & attr::L != 0 && hv.attributes & attr::G != 0);
        assert_eq!(Segment::from_hv(&hv), s);
    }

    #[test]
    fn v86_and_pack() {
        let s = Segment { selector: 0xb800, base: 0xb8000, limit: 0xffff, flags: 0 };
        let hv = s.to_hv(true);
        assert_eq!(hv.attributes, 0xf3);
        let p = hv.pack();
        assert_eq!(p, [0xb8000, 0xffff | 0xb800 << 32 | 0xf3 << 48]);
        assert_eq!(HvSegment::unpack(p), hv);
    }
}
