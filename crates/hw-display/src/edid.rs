// SPDX-License-Identifier: GPL-2.0-or-later

//! The EDID blob display devices hand to the guest, from QEMU's `hw/display/edid-generate.c` and
//! `edid-region.c`.
//!
//! [`edid_generate`] fills a 128, 256 or 384 byte buffer the way `qemu_edid_generate()` does: the
//! base block, a CTA extension when there is room for one, and a DisplayID extension for screens
//! too large for the base block's detailed timing. [`EdidRegion`] is the read only MMIO window
//! the PCI display devices put at offset 0 of their register BAR.

use std::sync::Arc;

use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `EDID_NAME_MAX_LENGTH`.
pub const EDID_NAME_MAX_LENGTH: usize = 12;

/// `qemu_edid_info`. Zero means "use the default" for every number.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EdidInfo {
    /// The three letter PNP vendor id, "RHT" when unset or not three letters long.
    pub vendor: Option<String>,
    /// The monitor name, "QEMU Monitor" when unset.
    pub name: Option<String>,
    pub serial: Option<String>,
    pub width_mm: u16,
    pub height_mm: u16,
    /// The preferred mode, 1280x800 by default.
    pub prefx: u32,
    pub prefy: u32,
    /// The largest mode to list, unlimited when zero.
    pub maxx: u32,
    pub maxy: u32,
    /// In mHz, 75000 by default.
    pub refresh_rate: u32,
}

struct EdidMode {
    xres: u32,
    yres: u32,
    byte: usize,
    xtra3: usize,
    bit: u32,
    dta: u8,
}

const fn m(xres: u32, yres: u32, byte: usize, xtra3: usize, bit: u32, dta: u8) -> EdidMode {
    EdidMode { xres, yres, byte, xtra3, bit, dta }
}

const MODES: [EdidMode; 22] = [
    // dea/dta extension timings (all at 50 Hz)
    m(5120, 2160, 0, 0, 0, 125),
    m(4096, 2160, 0, 0, 0, 101),
    m(3840, 2160, 0, 0, 0, 96),
    m(2560, 1080, 0, 0, 0, 89),
    m(2048, 1152, 0, 0, 0, 0),
    m(1920, 1080, 0, 0, 0, 31),
    // dea/dta extension timings (all at 60 Hz)
    m(3840, 2160, 0, 0, 0, 97),
    // additional standard timings 3 (all at 60 Hz)
    m(1920, 1200, 0, 10, 0, 0),
    m(1600, 1200, 0, 9, 2, 0),
    m(1680, 1050, 0, 9, 5, 0),
    m(1440, 900, 0, 8, 5, 0),
    m(1280, 1024, 0, 7, 1, 0),
    m(1280, 960, 0, 7, 3, 0),
    m(1280, 768, 0, 7, 6, 0),
    m(1920, 1440, 0, 11, 5, 0),
    m(1856, 1392, 0, 10, 3, 0),
    m(1792, 1344, 0, 10, 5, 0),
    m(1440, 1050, 0, 8, 1, 0),
    m(1360, 768, 0, 8, 7, 0),
    // established timings (all at 60 Hz)
    m(1024, 768, 36, 0, 3, 0),
    m(800, 600, 35, 0, 0, 0),
    m(640, 480, 35, 0, 5, 0),
];

#[derive(Clone, Copy, Debug, Default)]
struct Timings {
    xfront: u32,
    xsync: u32,
    xblank: u32,
    yfront: u32,
    ysync: u32,
    yblank: u32,
    clock: u64,
}

fn generate_timings(refresh_rate: u32, xres: u32, yres: u32) -> Timings {
    let xblank = xres.wrapping_mul(35) / 100;
    let yblank = yres.wrapping_mul(35) / 1000;
    Timings {
        xfront: xres.wrapping_mul(25) / 100,
        xsync: xres.wrapping_mul(3) / 100,
        xblank,
        yfront: yres.wrapping_mul(5) / 1000,
        ysync: yres.wrapping_mul(5) / 1000,
        yblank,
        clock: (u64::from(refresh_rate)
            * u64::from(xres.wrapping_add(xblank))
            * u64::from(yres.wrapping_add(yblank)))
            / 10_000_000,
    }
}

fn edid_ext_dta(dta: &mut [u8]) {
    dta[0] = 0x02;
    dta[1] = 0x03;
    dta[2] = 0x05;
    dta[3] = 0x00;
    // video data block
    dta[4] = 0x40;
}

fn edid_ext_dta_mode(dta: &mut [u8], nr: u8) {
    let at = usize::from(dta[2]);
    dta[at] = nr;
    dta[2] += 1;
    dta[4] += 1;
}

fn edid_std_mode(mode: &mut [u8], xres: u32, yres: u32) -> bool {
    let aspect: u8 = if xres == 0 || yres == 0 {
        mode[0] = 0x01;
        mode[1] = 0x01;
        return true;
    } else if xres * 10 == yres * 16 {
        0
    } else if xres * 3 == yres * 4 {
        1
    } else if xres * 4 == yres * 5 {
        2
    } else if xres * 9 == yres * 16 {
        3
    } else {
        return false;
    };
    if (xres / 8).wrapping_sub(31) > 255 {
        return false;
    }
    mode[0] = ((xres / 8) - 31) as u8;
    mode[1] = aspect << 6;
    true
}

/// A descriptor position: in the base block or in the CTA extension, as a byte offset into the
/// whole buffer.
type Desc = Option<usize>;

fn edid_fill_modes(edid: &mut [u8], xtra3: Desc, dta: Option<usize>, maxx: u32, maxy: u32) {
    let mut std = 38;
    for mode in &MODES {
        if (maxx != 0 && mode.xres > maxx) || (maxy != 0 && mode.yres > maxy) {
            continue;
        }
        if mode.byte != 0 {
            edid[mode.byte] |= 1 << mode.bit;
        } else if std < 54 {
            if edid_std_mode(&mut edid[std..], mode.xres, mode.yres) {
                std += 2;
            }
        } else if mode.xtra3 != 0 {
            if let Some(x) = xtra3 {
                edid[x + mode.xtra3] |= 1 << mode.bit;
            }
        }
        if mode.dta != 0 {
            if let Some(d) = dta {
                edid_ext_dta_mode(&mut edid[d..], mode.dta);
            }
        }
    }
    while std < 54 {
        edid_std_mode(&mut edid[std..], 0, 0);
        std += 2;
    }
}

/// `edid_checksum()`: makes the `len` bytes plus the one after them sum to zero.
fn edid_checksum(edid: &mut [u8], len: usize) {
    let sum = edid[..len].iter().fold(0u32, |s, &b| s + u32::from(b)) & 0xff;
    if sum != 0 {
        edid[len] = (0x100 - sum) as u8;
    }
}

fn edid_desc_next(edid: &[u8], dta: Option<usize>, desc: Desc) -> Desc {
    let desc = desc?;
    if desc + 18 + 18 < 127 {
        return Some(desc + 18);
    }
    if let Some(d) = dta {
        if desc < 127 {
            return Some(d + usize::from(edid[d + 2]));
        }
        if desc + 18 + 18 < d + 127 {
            return Some(desc + 18);
        }
    }
    None
}

fn edid_desc_type(desc: &mut [u8], ty: u8) {
    desc[..5].copy_from_slice(&[0, 0, 0, ty, 0]);
}

fn edid_desc_text(desc: &mut [u8], ty: u8, text: &str) {
    edid_desc_type(desc, ty);
    desc[5..18].fill(b' ');
    let text = text.as_bytes();
    let len = text.len().min(EDID_NAME_MAX_LENGTH);
    desc[5..5 + len].copy_from_slice(&text[..len]);
    desc[5 + len] = b'\n';
}

fn edid_desc_ranges(desc: &mut [u8]) {
    edid_desc_type(desc, 0xfd);
    // vertical (50 to 125 Hz)
    desc[5] = 50;
    desc[6] = 125;
    // horizontal (30 to 160 kHz)
    desc[7] = 30;
    desc[8] = 160;
    // max dot clock (2550 MHz)
    desc[9] = (2550 / 10) as u8;
    // no extended timing information
    desc[10] = 0x01;
    // padding
    desc[11] = b'\n';
    desc[12..18].fill(b' ');
}

/// Additional standard timings 3.
fn edid_desc_xtra3_std(desc: &mut [u8]) {
    edid_desc_type(desc, 0xf7);
    desc[5] = 10;
}

fn edid_desc_dummy(desc: &mut [u8]) {
    edid_desc_type(desc, 0x10);
}

fn edid_desc_timing(desc: &mut [u8], t: &Timings, xres: u32, yres: u32, xmm: u32, ymm: u32) {
    desc[0..2].copy_from_slice(&(t.clock as u16).to_le_bytes());

    desc[2] = xres as u8;
    desc[3] = t.xblank as u8;
    desc[4] = (((xres & 0xf00) >> 4) | ((t.xblank & 0xf00) >> 8)) as u8;

    desc[5] = yres as u8;
    desc[6] = t.yblank as u8;
    desc[7] = (((yres & 0xf00) >> 4) | ((t.yblank & 0xf00) >> 8)) as u8;

    desc[8] = t.xfront as u8;
    desc[9] = t.xsync as u8;

    desc[10] = (((t.yfront & 0x00f) << 4) | (t.ysync & 0x00f)) as u8;
    desc[11] = (((t.xfront & 0x300) >> 2)
        | ((t.xsync & 0x300) >> 4)
        | ((t.yfront & 0x030) >> 2)
        | ((t.ysync & 0x030) >> 4)) as u8;

    desc[12] = xmm as u8;
    desc[13] = ymm as u8;
    desc[14] = (((xmm & 0xf00) >> 4) | ((ymm & 0xf00) >> 8)) as u8;

    desc[17] = 0x18;
}

fn edid_to_10bit(value: f32) -> u32 {
    (value * 1024.0 + 0.5) as u32
}

fn edid_colorspace(edid: &mut [u8], c: [f32; 8]) {
    let [rx, ry, gx, gy, bx, by, wx, wy] = c.map(edid_to_10bit);
    edid[25] = (((rx & 0x03) << 6) | ((ry & 0x03) << 4) | ((gx & 0x03) << 2) | (gy & 0x03)) as u8;
    edid[26] = (((bx & 0x03) << 6) | ((by & 0x03) << 4) | ((wx & 0x03) << 2) | (wy & 0x03)) as u8;
    edid[27] = (rx >> 2) as u8;
    edid[28] = (ry >> 2) as u8;
    edid[29] = (gx >> 2) as u8;
    edid[30] = (gy >> 2) as u8;
    edid[31] = (bx >> 2) as u8;
    edid[32] = (by >> 2) as u8;
    edid[33] = (wx >> 2) as u8;
    edid[34] = (wy >> 2) as u8;
}

/// `qemu_edid_dpi_to_mm()`.
pub fn edid_dpi_to_mm(dpi: u32, res: u32) -> u32 {
    res * 254 / 10 / dpi
}

fn init_displayid(did: &mut [u8]) {
    did[0] = 0x70; // display id extension
    did[1] = 0x13; // version 1.3
    did[2] = 4; // length
    did[3] = 0x03; // product type (standalone display device)
    let len = usize::from(did[2]) + 4;
    edid_checksum(&mut did[1..], len);
}

fn displayid_generate(did: &mut [u8], t: &Timings, xres: u32, yres: u32) {
    did[0] = 0x70; // display id extension
    did[1] = 0x13; // version 1.3
    did[2] = 23; // length
    did[3] = 0x03; // product type (standalone display device)

    did[5] = 0x03; // Detailed Timings Data Block
    did[6] = 0x00; // revision
    did[7] = 0x14; // block length

    did[8] = t.clock as u8;
    did[9] = (t.clock >> 8) as u8;
    did[10] = (t.clock >> 16) as u8;

    did[11] = 0x88; // leave aspect ratio undefined

    let words = [
        xres.wrapping_sub(1),
        t.xblank.wrapping_sub(1),
        t.xfront.wrapping_sub(1),
        t.xsync.wrapping_sub(1),
        yres.wrapping_sub(1),
        t.yblank.wrapping_sub(1),
        t.yfront.wrapping_sub(1),
        t.ysync.wrapping_sub(1),
    ];
    for (i, w) in words.iter().enumerate() {
        did[12 + 2 * i..14 + 2 * i].copy_from_slice(&(*w as u16).to_le_bytes());
    }
    let len = usize::from(did[2]) + 4;
    edid_checksum(&mut did[1..], len);
}

/// `atoi()`: leading blanks, an optional sign and the digits that follow.
fn atoi(s: &str) -> u32 {
    let s = s.trim_start();
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let mut v: i64 = 0;
    for b in digits.bytes().take_while(u8::is_ascii_digit) {
        v = v.wrapping_mul(10).wrapping_add(i64::from(b - b'0'));
    }
    (if neg { v.wrapping_neg() } else { v }) as i32 as u32
}

/// `qemu_edid_generate()`: fills `edid` (128, 256 or 384 bytes, zeroed by the caller) and writes
/// the defaults it picked back into `info`.
pub fn edid_generate(edid: &mut [u8], info: &mut EdidInfo) {
    let size = edid.len();
    let mut desc: Desc = Some(54);
    let refresh_rate = if info.refresh_rate != 0 { info.refresh_rate } else { 75000 };
    // 100 dpi if there is no width_mm/height_mm. QEMU also computes the dpi from the size it is
    // given, and then does not use it.
    let dpi = 100;

    // set defaults

    if info.vendor.as_ref().is_none_or(|v| v.len() != 3) {
        info.vendor = Some("RHT".into());
    }
    if info.name.is_none() {
        info.name = Some("QEMU Monitor".into());
    }
    if info.prefx == 0 {
        info.prefx = 1280;
    }
    if info.prefy == 0 {
        info.prefy = 800;
    }
    let (width_mm, height_mm) = if info.width_mm != 0 && info.height_mm != 0 {
        (u32::from(info.width_mm), u32::from(info.height_mm))
    } else {
        (edid_dpi_to_mm(dpi, info.prefx), edid_dpi_to_mm(dpi, info.prefy))
    };

    let timings = generate_timings(refresh_rate, info.prefx, info.prefy);
    let large_screen = info.prefx >= 4096 || info.prefy >= 4096 || timings.clock >= 65536;

    // extensions

    let mut dta = None;
    let mut did = None;
    if size >= 256 {
        dta = Some(128);
        edid[126] += 1;
        edid_ext_dta(&mut edid[128..]);
    }
    if size >= 384 && large_screen {
        did = Some(256);
        edid[126] += 1;
        init_displayid(&mut edid[256..]);
    }

    // header information

    edid[0..8].copy_from_slice(&[0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00]);

    // manufacturer id, product code, serial number
    let v = info.vendor.as_deref().unwrap_or("RHT").as_bytes();
    let c = |b: u8| u16::from(b.wrapping_sub(b'@') & 0x1f);
    let vendor_id = (c(v[0]) << 10) | (c(v[1]) << 5) | c(v[2]);
    let model_nr: u16 = 0x1234;
    let serial_nr = info.serial.as_deref().map_or(0, atoi);
    edid[8..10].copy_from_slice(&vendor_id.to_be_bytes());
    edid[10..12].copy_from_slice(&model_nr.to_le_bytes());
    edid[12..16].copy_from_slice(&serial_nr.to_le_bytes());

    // manufacture week and year
    edid[16] = 42;
    edid[17] = (2014 - 1990) as u8;

    // edid version
    edid[18] = 1;
    edid[19] = 4;

    // basic display parameters

    // video input: digital, 8bpc, displayport
    edid[20] = 0xa5;
    // screen size
    edid[21] = (width_mm / 10) as u8;
    edid[22] = (height_mm / 10) as u8;
    // display gamma: 2.2
    edid[23] = 220 - 100;
    // supported features bitmap: std sRGB, preferred timing
    edid[24] = 0x06;

    // chromaticity coordinates: standard sRGB, red, green, blue and the white point
    edid_colorspace(edid, [0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600, 0.3127, 0.3290]);

    // descriptor blocks

    if !large_screen {
        // The DTD section has only 12 bits to store the resolution.
        if let Some(d) = desc {
            edid_desc_timing(&mut edid[d..], &timings, info.prefx, info.prefy, width_mm, height_mm);
        }
        desc = edid_desc_next(edid, dta, desc);
    }

    let xtra3 = desc;
    if let Some(x) = xtra3 {
        edid_desc_xtra3_std(&mut edid[x..]);
    }
    desc = edid_desc_next(edid, dta, desc);
    edid_fill_modes(edid, xtra3, dta, info.maxx, info.maxy);
    // The dta video data block is finished at this point, so dta descriptor offsets do not move
    // any more.

    if let Some(d) = desc {
        edid_desc_ranges(&mut edid[d..]);
    }
    desc = edid_desc_next(edid, dta, desc);

    if let (Some(d), Some(name)) = (desc, info.name.clone()) {
        edid_desc_text(&mut edid[d..], 0xfc, &name);
        desc = edid_desc_next(edid, dta, desc);
    }

    if let (Some(d), Some(serial)) = (desc, info.serial.clone()) {
        edid_desc_text(&mut edid[d..], 0xff, &serial);
        desc = edid_desc_next(edid, dta, desc);
    }

    while let Some(d) = desc {
        edid_desc_dummy(&mut edid[d..]);
        desc = edid_desc_next(edid, dta, desc);
    }

    // display id extensions

    if let Some(d) = did {
        displayid_generate(&mut edid[d..], &timings, info.prefx, info.prefy);
    }

    // finish up

    edid_checksum(edid, 127);
    if let Some(d) = dta {
        edid_checksum(&mut edid[d..], 127);
    }
    if let Some(d) = did {
        edid_checksum(&mut edid[d..], 127);
    }
}

/// `qemu_edid_size()`: 128 bytes per block, or 0 when `edid` does not start like an EDID.
pub fn edid_size(edid: &[u8]) -> usize {
    if edid.len() < 127 || edid[0] != 0x00 || edid[1] != 0xff {
        return 0;
    }
    128 * (usize::from(edid[126]) + 1)
}

/// `qemu_edid_region_io()`: the blob as a read only byte wide register window.
#[derive(Debug)]
pub struct EdidRegion {
    blob: Arc<[u8]>,
}

impl EdidRegion {
    pub fn new(blob: Arc<[u8]>) -> EdidRegion {
        EdidRegion { blob }
    }
}

impl MmioOps for EdidRegion {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.blob.get(offset as usize).map_or(0, |&b| u64::from(b)))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, _v: u64) -> MemResult<()> {
        // read only
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(b: &[u8]) -> u8 {
        b.iter().fold(0u8, |s, &x| s.wrapping_add(x))
    }

    #[test]
    fn default_edid_has_valid_checksums_and_one_extension() {
        let mut e = [0u8; 256];
        let mut info = EdidInfo::default();
        edid_generate(&mut e, &mut info);
        assert_eq!(&e[..8], &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
        assert_eq!(edid_size(&e), 256);
        assert_eq!(sum(&e[..128]), 0);
        assert_eq!(sum(&e[128..]), 0);
        // RHT
        assert_eq!(&e[8..10], &[0x49, 0x14]);
        // 1280x800 at 100 dpi is 325 x 203 mm.
        assert_eq!((e[21], e[22]), (32, 20));
        // The preferred timing: 1280 wide with 448 blanking.
        assert_eq!((e[56], e[57], e[58]), (0x00, 0xc0, 0x51));
        // The monitor name descriptor.
        let name = e[..128].windows(12).position(|w| w == b"QEMU Monitor");
        assert!(name.is_some());
    }

    #[test]
    fn large_screens_get_a_displayid_block() {
        let mut e = [0u8; 384];
        let mut info = EdidInfo { prefx: 5120, prefy: 2160, ..EdidInfo::default() };
        edid_generate(&mut e, &mut info);
        assert_eq!(e[126], 2);
        assert_eq!(e[256], 0x70);
        assert_eq!(sum(&e[256..384]), 0);
    }

    #[test]
    fn std_modes_skip_unknown_aspects() {
        let mut b = [0u8; 2];
        assert!(edid_std_mode(&mut b, 1280, 800));
        assert_eq!(b, [129, 0]);
        assert!(!edid_std_mode(&mut b, 1360, 768));
    }
}
