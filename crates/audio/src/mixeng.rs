// SPDX-License-Identifier: GPL-2.0-or-later

//! The mixing engine: sample conversion, clipping, volume and the linear resampler.
//!
//! This is QEMU's `audio/mixeng.c` with the integer mixing engine, which is what every QEMU build
//! uses except the macOS one with CoreAudio. Samples are 64-bit integers with the full scale at
//! 32 bits, so the integer formats convert by shifting and the float format by scaling with 2^31.
//! Arithmetic that can overflow in C wraps here, which is what the C code does on every host QEMU
//! runs on.

use ruvm_qapi::types::AudioFormat;

/// One stereo frame in the mixing format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StSample {
    /// The left channel.
    pub l: i64,
    /// The right channel.
    pub r: i64,
}

/// The software volume a voice applies in the mixing engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MixengVolume {
    /// Mute both channels.
    pub mute: bool,
    /// The right channel gain, with unity at 1 << 32.
    pub r: i64,
    /// The left channel gain, with unity at 1 << 32.
    pub l: i64,
}

/// Unity gain, QEMU's `nominal_volume`.
pub const NOMINAL_VOLUME: MixengVolume = MixengVolume { mute: false, r: 1 << 32, l: 1 << 32 };

/// Zeroes `buf`, like `mixeng_clear()`.
pub fn clear(buf: &mut [StSample]) {
    buf.fill(StSample::default());
}

/// Applies `vol` to every frame in `buf`, like `mixeng_volume()`.
pub fn volume(buf: &mut [StSample], vol: &MixengVolume) {
    if vol.mute {
        clear(buf);
        return;
    }
    for s in buf {
        s.l = s.l.wrapping_mul(vol.l) >> 32;
        s.r = s.r.wrapping_mul(vol.r) >> 32;
    }
}

/// The bits per sample of a format.
pub fn format_bits(af: AudioFormat) -> usize {
    match af {
        AudioFormat::U8 | AudioFormat::S8 => 8,
        AudioFormat::U16 | AudioFormat::S16 => 16,
        AudioFormat::U32 | AudioFormat::S32 | AudioFormat::F32 => 32,
    }
}

/// Whether a format is signed. Float counts as signed, as in `audio_format_is_signed()`.
pub fn format_is_signed(af: AudioFormat) -> bool {
    matches!(af, AudioFormat::S8 | AudioFormat::S16 | AudioFormat::S32 | AudioFormat::F32)
}

/// Picks the conversion and clip routines for a PCM layout, the way QEMU indexes
/// `mixeng_conv` and `mixeng_clip` by `[stereo][signed][swap][size]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SampleFmt {
    af: AudioFormat,
    stereo: bool,
    swap: bool,
}

impl SampleFmt {
    /// The layout for `nchannels` samples of `af`. Any channel count other than two takes the
    /// mono routines, which read one sample per frame.
    pub fn new(af: AudioFormat, nchannels: i32, swap: bool) -> Self {
        // 8-bit samples have no byte order, and QEMU's tables use the natural routine for them.
        let swap = swap && format_bits(af) != 8;
        SampleFmt { af, stereo: nchannels == 2, swap }
    }

    fn sample_bytes(&self) -> usize {
        format_bits(self.af) / 8
    }

    fn conv_one(&self, b: &[u8]) -> i64 {
        match self.af {
            AudioFormat::S8 => i64::from(b[0] as i8) << 24,
            AudioFormat::U8 => (i64::from(b[0]) - 0x80) << 24,
            AudioFormat::S16 => i64::from(self.load16(b) as i16) << 16,
            AudioFormat::U16 => (i64::from(self.load16(b)) - 0x8000) << 16,
            AudioFormat::S32 => i64::from(self.load32(b) as i32),
            AudioFormat::U32 => i64::from(self.load32(b)) - 0x8000_0000,
            AudioFormat::F32 => {
                let f = f32::from_bits(self.load32(b));
                (f * 2_147_483_648.0f32) as i64
            }
        }
    }

    fn clip_one(&self, v: i64, out: &mut [u8]) {
        match self.af {
            AudioFormat::F32 => {
                let f = (v as f32) / 2_147_483_648.0f32;
                self.store32(out, f.to_bits());
            }
            _ => {
                let signed = format_is_signed(self.af);
                let bits = format_bits(self.af) as u32;
                let raw: u32 = if v >= 0x7fff_ffff {
                    // IN_MAX
                    if signed { (1u32 << (bits - 1)) - 1 } else { u32::MAX >> (32 - bits) }
                } else if v < -2_147_483_648 {
                    // IN_MIN
                    if signed { 1u32 << (bits - 1) } else { 0 }
                } else {
                    let s = v >> (32 - bits);
                    if signed { s as u32 } else { (s + (1i64 << (bits - 1))) as u32 }
                };
                match bits {
                    8 => out[0] = raw as u8,
                    16 => self.store16(out, raw as u16),
                    _ => self.store32(out, raw),
                }
            }
        }
    }

    fn load16(&self, b: &[u8]) -> u16 {
        let v = u16::from_ne_bytes([b[0], b[1]]);
        if self.swap { v.swap_bytes() } else { v }
    }

    fn load32(&self, b: &[u8]) -> u32 {
        let v = u32::from_ne_bytes([b[0], b[1], b[2], b[3]]);
        if self.swap { v.swap_bytes() } else { v }
    }

    fn store16(&self, out: &mut [u8], v: u16) {
        let v = if self.swap { v.swap_bytes() } else { v };
        out[..2].copy_from_slice(&v.to_ne_bytes());
    }

    fn store32(&self, out: &mut [u8], v: u32) {
        let v = if self.swap { v.swap_bytes() } else { v };
        out[..4].copy_from_slice(&v.to_ne_bytes());
    }

    /// Converts `dst.len()` frames of PCM from `src` into the mixing format, like the
    /// `conv_*_to_stereo` and `conv_*_to_mono` routines.
    pub fn conv(&self, dst: &mut [StSample], src: &[u8]) {
        let sb = self.sample_bytes();
        let step = if self.stereo { 2 * sb } else { sb };
        for (i, out) in dst.iter_mut().enumerate() {
            let p = &src[i * step..];
            out.l = self.conv_one(p);
            out.r = if self.stereo { self.conv_one(&p[sb..]) } else { out.l };
        }
    }

    /// Clips `src.len()` frames from the mixing format into PCM in `dst`, like the
    /// `clip_*_from_stereo` and `clip_*_from_mono` routines. Mono writes the sum of both channels.
    pub fn clip(&self, dst: &mut [u8], src: &[StSample]) {
        let sb = self.sample_bytes();
        let step = if self.stereo { 2 * sb } else { sb };
        for (i, s) in src.iter().enumerate() {
            let p = &mut dst[i * step..];
            if self.stereo {
                self.clip_one(s.l, p);
                self.clip_one(s.r, &mut p[sb..]);
            } else {
                self.clip_one(s.l.wrapping_add(s.r), p);
            }
        }
    }
}

/// The linear interpolating resampler, QEMU's `struct rate`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rate {
    opos: u64,
    opos_inc: u64,
    ipos: u32,
    ilast: StSample,
}

impl Rate {
    /// Starts a resampler from `inrate` to `outrate`, like `st_rate_start()`.
    pub fn new(inrate: i32, outrate: i32) -> Self {
        Rate {
            opos: 0,
            opos_inc: ((inrate as u32 as u64) << 32) / (outrate as u32 as u64),
            ipos: 0,
            ilast: StSample::default(),
        }
    }

    /// Resamples `ibuf` into `obuf`, adding to the output when `mix` is set and storing it
    /// otherwise. Returns the input frames consumed and the output frames produced, like
    /// `st_rate_flow()` and `st_rate_flow_mix()`.
    pub fn flow(&mut self, ibuf: &[StSample], obuf: &mut [StSample], mix: bool) -> (usize, usize) {
        let put = |o: &mut StSample, v: StSample| {
            if mix {
                o.l = o.l.wrapping_add(v.l);
                o.r = o.r.wrapping_add(v.r);
            } else {
                *o = v;
            }
        };
        if self.opos_inc == 1u64 << 32 {
            let n = ibuf.len().min(obuf.len());
            for i in 0..n {
                put(&mut obuf[i], ibuf[i]);
            }
            return (n, n);
        }
        if ibuf.is_empty() {
            return (0, 0);
        }
        let mut ilast = self.ilast;
        let mut i = 0;
        let mut o = 0;
        'outer: loop {
            while u64::from(self.ipos) <= (self.opos >> 32) {
                ilast = ibuf[i];
                i += 1;
                self.ipos = self.ipos.wrapping_add(1);
                if i >= ibuf.len() {
                    break 'outer;
                }
            }
            if o >= obuf.len() {
                break;
            }
            let icur = ibuf[i];
            if self.ipos >= 0x10001 {
                self.ipos = 1;
                self.opos &= 0xffff_ffff;
            }
            let t = (self.opos & 0xffff_ffff) as i64;
            let w = u32::MAX as i64 - t;
            let out = StSample {
                l: ilast.l.wrapping_mul(w).wrapping_add(icur.l.wrapping_mul(t)) >> 32,
                r: ilast.r.wrapping_mul(w).wrapping_add(icur.r.wrapping_mul(t)) >> 32,
            };
            put(&mut obuf[o], out);
            o += 1;
            self.opos = self.opos.wrapping_add(self.opos_inc);
        }
        self.ilast = ilast;
        (i, o)
    }

    /// How many output frames `frames_in` input frames produce, like `st_rate_frames_out()`.
    pub fn frames_out(&self, frames_in: u32) -> u32 {
        if self.opos_inc == 1u64 << 32 {
            return frames_in;
        }
        if frames_in == 0 {
            return 0;
        }
        let ipos_end = self.ipos.wrapping_sub(1).wrapping_add(frames_in);
        let opos_end = u64::from(ipos_end) << 32;
        if opos_end.wrapping_add(self.opos_inc) <= self.opos {
            return 0;
        }
        let delta = opos_end.wrapping_sub(self.opos).wrapping_add(self.opos_inc);
        let frames_out = (delta / self.opos_inc) as u32;
        if delta % self.opos_inc != 0 { frames_out } else { frames_out.wrapping_sub(1) }
    }

    /// How many input frames are needed for `frames_out` output frames, like
    /// `st_rate_frames_in()`.
    pub fn frames_in(&self, frames_out: u32) -> u32 {
        if self.opos_inc == 1u64 << 32 {
            return frames_out;
        }
        let (opos_start, ipos_start) = if frames_out != 0 {
            (self.opos, self.ipos)
        } else {
            let offset = (self.opos_inc + (1u64 << 32) - 1) & !((1u64 << 32) - 1);
            (self.opos.wrapping_add(offset), self.ipos.wrapping_add((offset >> 32) as u32))
        };
        let opos_end = opos_start
            .wrapping_sub(self.opos_inc)
            .wrapping_add(self.opos_inc.wrapping_mul(u64::from(frames_out)));
        let ipos_end = ((opos_end >> 32) as u32).wrapping_add(1);
        if ipos_end.wrapping_add(1) > ipos_start {
            ipos_end.wrapping_add(1).wrapping_sub(ipos_start)
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s16_round_trip() {
        let f = SampleFmt::new(AudioFormat::S16, 2, false);
        let pcm: Vec<u8> =
            [1000i16, -1000, i16::MAX, i16::MIN].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut st = [StSample::default(); 2];
        f.conv(&mut st, &pcm);
        assert_eq!(st[0], StSample { l: 1000 << 16, r: -1000 << 16 });
        let mut out = vec![0u8; 8];
        f.clip(&mut out, &st);
        assert_eq!(out, pcm);
    }

    #[test]
    fn u8_and_clip_limits() {
        let f = SampleFmt::new(AudioFormat::U8, 1, true);
        let mut st = [StSample::default(); 2];
        f.conv(&mut st, &[0x80, 0xff]);
        assert_eq!(st[0], StSample { l: 0, r: 0 });
        assert_eq!(st[1].l, 0x7f << 24);
        // Mono sums both channels, which overflows the range and clips to IN_MAX.
        let mut out = [0u8; 2];
        f.clip(&mut out, &st);
        assert_eq!(out, [0x80, 0xff]);
        let s = SampleFmt::new(AudioFormat::S16, 2, false);
        let mut o = [0u8; 4];
        s.clip(&mut o, &[StSample { l: 0x7fff_ffff, r: -0x8000_0001 }]);
        assert_eq!(o[..2], i16::MAX.to_ne_bytes());
        assert_eq!(o[2..], i16::MIN.to_ne_bytes());
    }

    #[test]
    fn swapped_u16() {
        let f = SampleFmt::new(AudioFormat::U16, 2, true);
        let mut st = [StSample::default(); 1];
        let src: Vec<u8> = [0x1234u16.swap_bytes(), 0x8000u16.swap_bytes()]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        f.conv(&mut st, &src);
        assert_eq!(st[0].l, (0x1234i64 - 0x8000) << 16);
        assert_eq!(st[0].r, 0);
        let mut out = [0u8; 4];
        f.clip(&mut out, &st);
        assert_eq!(out.to_vec(), src);
    }

    #[test]
    fn float_scale() {
        let f = SampleFmt::new(AudioFormat::F32, 2, false);
        let src: Vec<u8> = [0.5f32, -1.0].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let mut st = [StSample::default(); 1];
        f.conv(&mut st, &src);
        assert_eq!(st[0], StSample { l: 1 << 30, r: -(1 << 31) });
        let mut out = [0u8; 8];
        f.clip(&mut out, &st);
        assert_eq!(out.to_vec(), src);
    }

    #[test]
    fn volume_scales() {
        let mut b = [StSample { l: 1 << 20, r: -(1 << 20) }];
        volume(&mut b, &MixengVolume { mute: false, l: 1 << 31, r: 1 << 32 });
        assert_eq!(b[0], StSample { l: 1 << 19, r: -(1 << 20) });
        volume(&mut b, &MixengVolume { mute: true, ..NOMINAL_VOLUME });
        assert_eq!(b[0], StSample::default());
    }

    #[test]
    fn rate_identity_and_counts() {
        let mut r = Rate::new(44100, 44100);
        let i = [StSample { l: 5, r: 6 }; 3];
        let mut o = [StSample { l: 1, r: 1 }; 2];
        assert_eq!(r.flow(&i, &mut o, true), (2, 2));
        assert_eq!(o[0], StSample { l: 6, r: 7 });
        assert_eq!(r.frames_in(7), 7);

        let r = Rate::new(22050, 44100);
        assert_eq!(r.frames_in(1024), 513);
        assert_eq!(r.frames_out(513), 1024);
        let r = Rate::new(48000, 44100);
        assert_eq!(r.frames_in(1024), 1115);
        assert_eq!(r.frames_in(0), 0);
    }

    #[test]
    fn rate_upsample_interpolates() {
        let mut r = Rate::new(1, 2);
        let i = [StSample { l: 0, r: 0 }, StSample { l: 1 << 16, r: 1 << 16 }, StSample::default()];
        let mut o = [StSample::default(); 8];
        let (ci, co) = r.flow(&i, &mut o, false);
        assert_eq!((ci, co), (3, 4));
        assert_eq!(o[0].l, 0);
        // Halfway between 0 and 1 << 16.
        assert_eq!(o[1].l, 0x8000);
    }
}
