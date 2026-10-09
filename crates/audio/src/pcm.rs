// SPDX-License-Identifier: GPL-2.0-or-later

//! The PCM layer between the mixing engine and a host driver.
//!
//! This is the part of QEMU's `audio_int.h` and `audio-mixeng-be.c` a driver sees: the stream
//! settings, the per-voice buffer QEMU keeps in `HWVoiceOut` and `HWVoiceIn`, the generic
//! buffer helpers most drivers plug in, and the rate limiter the file and null drivers use to
//! consume audio at the speed of the virtual clock.

use std::fmt;
use std::sync::{Arc, OnceLock};

use ruvm_hw_core::timer::{Clock, muldiv64};
use ruvm_qapi::types::{AudioFormat, Audiodev, AudiodevPerDirectionOptions};

use crate::mixeng::{SampleFmt, format_bits};

/// Whether the host stores samples big endian.
pub const HOST_BIG_ENDIAN: bool = cfg!(target_endian = "big");

/// The format of a stream as a device asks for it, QEMU's `struct audsettings`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudSettings {
    /// Frames per second.
    pub freq: i32,
    /// Samples per frame.
    pub nchannels: i32,
    /// The sample format.
    pub fmt: AudioFormat,
    /// Whether samples are stored big endian.
    pub big_endian: bool,
}

impl AudSettings {
    /// Whether the settings describe a stream, like `audio_validate_settings()`.
    pub fn is_valid(&self) -> bool {
        self.nchannels >= 1 && self.freq > 0
    }
}

impl fmt::Display for AudSettings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "frequency={} nchannels={} fmt={} endian={}",
            self.freq,
            self.nchannels,
            self.fmt.as_str(),
            if self.big_endian { "big" } else { "little" }
        )
    }
}

/// The derived layout of a stream, QEMU's `struct audio_pcm_info`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcmInfo {
    /// The sample format.
    pub af: AudioFormat,
    /// Frames per second.
    pub freq: i32,
    /// Samples per frame.
    pub nchannels: i32,
    /// Bytes per frame.
    pub bytes_per_frame: usize,
    /// Bytes per second.
    pub bytes_per_second: i32,
    /// Whether samples are in the other byte order from the host.
    pub swap_endianness: bool,
}

impl PcmInfo {
    /// The layout of `as_`, like `audio_pcm_init_info()`.
    pub fn new(as_: &AudSettings) -> Self {
        let bpf = as_.nchannels.wrapping_mul(format_bits(as_.fmt) as i32) / 8;
        PcmInfo {
            af: as_.fmt,
            freq: as_.freq,
            nchannels: as_.nchannels,
            bytes_per_frame: bpf as usize,
            bytes_per_second: as_.freq.wrapping_mul(bpf),
            swap_endianness: as_.big_endian != HOST_BIG_ENDIAN,
        }
    }

    /// Whether the layout matches `as_`, like `audio_pcm_info_eq()`.
    pub fn eq_settings(&self, as_: &AudSettings) -> bool {
        self.af == as_.fmt
            && self.freq == as_.freq
            && self.nchannels == as_.nchannels
            && self.swap_endianness == (as_.big_endian != HOST_BIG_ENDIAN)
    }

    /// Fills `frames` frames of `buf` with silence, like `audio_pcm_info_clear_buf()`. The
    /// unsigned 16 and 32-bit formats get the signed maximum, which is what QEMU writes.
    pub fn clear_buf(&self, buf: &mut [u8], frames: usize) {
        if frames == 0 {
            return;
        }
        let bytes = frames * self.bytes_per_frame;
        match self.af {
            AudioFormat::U8 => buf[..bytes].fill(0x80),
            AudioFormat::U16 => {
                let mut s = i16::MAX as u16;
                if self.swap_endianness {
                    s = s.swap_bytes();
                }
                let n = frames * self.nchannels as usize;
                for c in buf[..n * 2].chunks_exact_mut(2) {
                    c.copy_from_slice(&s.to_ne_bytes());
                }
            }
            AudioFormat::U32 => {
                let mut s = i32::MAX as u32;
                if self.swap_endianness {
                    s = s.swap_bytes();
                }
                let n = frames * self.nchannels as usize;
                for c in buf[..n * 4].chunks_exact_mut(4) {
                    c.copy_from_slice(&s.to_ne_bytes());
                }
            }
            AudioFormat::S8 | AudioFormat::S16 | AudioFormat::S32 | AudioFormat::F32 => {
                buf[..bytes].fill(0)
            }
        }
    }

    pub(crate) fn sample_fmt(&self) -> SampleFmt {
        SampleFmt::new(self.af, self.nchannels, self.swap_endianness)
    }
}

/// A device volume setting, QEMU's `Volume`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Volume {
    /// Mute the stream.
    pub mute: bool,
    /// How many entries of `vol` are used.
    pub channels: usize,
    /// Per-channel volume, 255 for full.
    pub vol: [u8; 16],
}

/// One direction of an audiodev after validation, with every default filled in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pdo {
    /// Mix voices in software.
    pub mixing_engine: bool,
    /// Run the host stream at the fixed settings below instead of the guest's.
    pub fixed_settings: bool,
    /// The fixed frequency.
    pub frequency: u32,
    /// The fixed channel count.
    pub channels: u32,
    /// The number of host voices.
    pub voices: u32,
    /// The fixed sample format.
    pub format: AudioFormat,
    /// The host buffer length in microseconds, if set.
    pub buffer_length: Option<u32>,
}

impl Pdo {
    pub(crate) fn from_qapi(p: &AudiodevPerDirectionOptions) -> Self {
        let mixing_engine = p.mixing_engine.unwrap_or(true);
        Pdo {
            mixing_engine,
            fixed_settings: p.fixed_settings.unwrap_or(mixing_engine),
            frequency: p.frequency.unwrap_or(44100),
            channels: p.channels.unwrap_or(2),
            voices: p.voices.unwrap_or(if mixing_engine { 1 } else { i32::MAX as u32 }),
            format: p.format.unwrap_or(AudioFormat::S16),
            buffer_length: p.buffer_length,
        }
    }

    /// The fixed settings in host byte order, like `audiodev_to_audsettings()`.
    pub fn settings(&self) -> AudSettings {
        AudSettings {
            freq: self.frequency as i32,
            nchannels: self.channels as i32,
            fmt: self.format,
            big_endian: HOST_BIG_ENDIAN,
        }
    }

    /// The buffer length in frames, rounded, like `audio_buffer_frames()`.
    pub fn buffer_frames(&self, as_: &AudSettings, def_usecs: u32) -> i32 {
        let usecs = u64::from(self.buffer_length.unwrap_or(def_usecs));
        ((as_.freq as u64).wrapping_mul(usecs).wrapping_add(500_000) / 1_000_000) as i32
    }
}

/// The virtual clock a backend paces itself by. It reads as zero until a machine attaches its
/// clock.
#[derive(Clone, Default)]
pub struct VClock(Arc<OnceLock<Arc<Clock>>>);

impl fmt::Debug for VClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VClock").field(&self.0.get().is_some()).finish()
    }
}

impl VClock {
    /// The current virtual time in nanoseconds.
    pub fn get_ns(&self) -> i64 {
        self.0.get().map_or(0, |c| c.get_ns())
    }

    pub(crate) fn set(&self, clock: Arc<Clock>) -> bool {
        self.0.set(clock).is_ok()
    }
}

/// Consumes bytes at the stream's real rate against the virtual clock, QEMU's `RateCtl`.
#[derive(Clone, Debug)]
pub struct RateCtl {
    clock: VClock,
    start_ticks: i64,
    bytes_sent: i64,
    peeked_frames: i64,
}

impl RateCtl {
    /// A limiter started now.
    pub fn new(clock: VClock) -> Self {
        let mut r = RateCtl { clock, start_ticks: 0, bytes_sent: 0, peeked_frames: 0 };
        r.start();
        r
    }

    /// Restarts the count, like `audio_rate_start()`.
    pub fn start(&mut self) {
        self.bytes_sent = 0;
        self.peeked_frames = 0;
        self.start_ticks = self.clock.get_ns();
    }

    /// The bytes due since the start that have not been used, like `audio_rate_peek_bytes()`.
    pub fn peek_bytes(&mut self, info: &PcmInfo) -> usize {
        let ticks = self.clock.get_ns().wrapping_sub(self.start_ticks);
        let bytes = muldiv64(ticks as u64, info.bytes_per_second as u32, 1_000_000_000) as i64;
        let frames = (bytes - self.bytes_sent) / info.bytes_per_frame as i64;
        self.peeked_frames = frames;
        if frames < 0 { 0 } else { (frames * info.bytes_per_frame as i64) as usize }
    }

    /// Accounts for `bytes_used` bytes, like `audio_rate_add_bytes()`.
    pub fn add_bytes(&mut self, bytes_used: usize) {
        if self.peeked_frames < 0 || self.peeked_frames > 65536 {
            self.start();
        }
        self.bytes_sent += bytes_used as i64;
    }

    /// Takes up to `bytes_avail` of the bytes due, like `audio_rate_get_bytes()`.
    pub fn get_bytes(&mut self, info: &PcmInfo, bytes_avail: usize) -> usize {
        let bytes = self.peek_bytes(info).min(bytes_avail);
        self.add_bytes(bytes);
        bytes
    }
}

/// The state QEMU keeps in every `HWVoiceOut` and `HWVoiceIn` for a driver: the stream layout,
/// the buffer size in frames and the emulated ring buffer the generic helpers use.
#[derive(Debug)]
pub struct HwCore {
    /// The host stream layout.
    pub info: PcmInfo,
    /// The host buffer size in frames.
    pub samples: usize,
    buf_emul: Option<Vec<u8>>,
    pos_emul: usize,
    pending_emul: usize,
    size_emul: usize,
}

impl HwCore {
    pub(crate) fn new(info: PcmInfo, samples: usize) -> Self {
        HwCore { info, samples, buf_emul: None, pos_emul: 0, pending_emul: 0, size_emul: 0 }
    }

    fn initialize_buffer(&mut self) {
        self.size_emul = self.samples * self.info.bytes_per_frame;
        self.buf_emul = Some(vec![0; self.size_emul]);
        self.pos_emul = 0;
        self.pending_emul = 0;
    }
}

/// `audio_ring_posb()`: the position `dist` bytes before `pos` in a ring of `len`.
pub(crate) fn ring_posb(pos: usize, dist: usize, len: usize) -> usize {
    if pos >= dist { pos - dist } else { len - dist + pos }
}

/// `audio_generic_buffer_get_free()`.
pub fn generic_buffer_get_free(hw: &HwCore) -> usize {
    if hw.buf_emul.is_some() {
        hw.size_emul - hw.pending_emul
    } else {
        hw.samples * hw.info.bytes_per_frame
    }
}

/// `audio_generic_run_buffer_out()`: hands the pending bytes of the ring to `write`.
pub fn generic_run_buffer_out<P: PcmOut + ?Sized>(pcm: &mut P, hw: &mut HwCore) {
    while hw.pending_emul > 0 {
        let start = ring_posb(hw.pos_emul, hw.pending_emul, hw.size_emul);
        let write_len = hw.pending_emul.min(hw.size_emul - start);
        let Some(buf) = hw.buf_emul.as_ref() else { return };
        let written = pcm.write(&hw.info, &buf[start..start + write_len]);
        hw.pending_emul -= written;
        if written < write_len {
            break;
        }
    }
}

/// `audio_generic_get_buffer_out()`: the free space at the write position of the ring.
pub fn generic_get_buffer_out(hw: &mut HwCore) -> &mut [u8] {
    if hw.buf_emul.is_none() {
        hw.initialize_buffer();
    }
    let size = (hw.size_emul - hw.pending_emul).min(hw.size_emul - hw.pos_emul);
    let pos = hw.pos_emul;
    match hw.buf_emul.as_mut() {
        Some(b) => &mut b[pos..pos + size],
        None => &mut [],
    }
}

/// `audio_generic_put_buffer_out()`.
pub fn generic_put_buffer_out(hw: &mut HwCore, size: usize) -> usize {
    assert!(size + hw.pending_emul <= hw.size_emul);
    hw.pending_emul += size;
    hw.pos_emul = (hw.pos_emul + size) % hw.size_emul;
    size
}

/// `audio_generic_run_buffer_in()`: fills the ring from `read`.
pub fn generic_run_buffer_in<P: PcmIn + ?Sized>(pcm: &mut P, hw: &mut HwCore) {
    if hw.buf_emul.is_none() {
        hw.initialize_buffer();
    }
    while hw.pending_emul < hw.size_emul {
        let read_len = (hw.size_emul - hw.pos_emul).min(hw.size_emul - hw.pending_emul);
        let pos = hw.pos_emul;
        let info = hw.info;
        let Some(buf) = hw.buf_emul.as_mut() else { return };
        let read = pcm.read(&info, &mut buf[pos..pos + read_len]);
        hw.pending_emul += read;
        hw.pos_emul = (hw.pos_emul + read) % hw.size_emul;
        if read < read_len {
            break;
        }
    }
}

/// `audio_generic_get_buffer_in()`: up to `size` captured bytes from the read position.
pub fn generic_get_buffer_in(hw: &mut HwCore, size: usize) -> &[u8] {
    if hw.buf_emul.is_none() {
        return &[];
    }
    let start = ring_posb(hw.pos_emul, hw.pending_emul, hw.size_emul);
    assert!(start < hw.size_emul);
    let size = size.min(hw.pending_emul).min(hw.size_emul - start);
    match hw.buf_emul.as_ref() {
        Some(b) => &b[start..start + size],
        None => &[],
    }
}

/// `audio_generic_put_buffer_in()`.
pub fn generic_put_buffer_in(hw: &mut HwCore, size: usize) {
    assert!(size <= hw.pending_emul);
    hw.pending_emul -= size;
}

/// A host playback stream, the driver half of QEMU's `HWVoiceOut`. The defaults are the
/// generic helpers QEMU installs when a driver leaves a hook empty.
pub trait PcmOut: Send {
    /// Plays `buf` and returns how many bytes were taken.
    fn write(&mut self, info: &PcmInfo, buf: &[u8]) -> usize;

    /// The free bytes in the host buffer, or `None` for no limit.
    fn buffer_get_free(&mut self, _hw: &HwCore) -> Option<usize> {
        None
    }

    /// Flushes buffered audio to the host.
    fn run_buffer_out(&mut self, _hw: &mut HwCore) {}

    /// A buffer of up to `size` bytes to clip the mix into, and its size. The buffer is `None`
    /// when the driver wants the size counted without any data.
    fn get_buffer_out<'a>(
        &'a mut self,
        hw: &'a mut HwCore,
        _size: usize,
    ) -> (Option<&'a mut [u8]>, usize) {
        let b = generic_get_buffer_out(hw);
        let n = b.len();
        (Some(b), n)
    }

    /// Commits `size` bytes of the buffer from `get_buffer_out` and returns how many were taken.
    fn put_buffer_out(&mut self, hw: &mut HwCore, size: usize) -> usize {
        generic_put_buffer_out(hw, size)
    }

    /// Starts or stops the host stream.
    fn enable_out(&mut self, _enable: bool) {}

    /// Applies a device volume in the host, for drivers that report `volume_out`.
    fn volume_out(&mut self, _vol: &Volume) {}

    /// Closes the host stream.
    fn fini_out(&mut self, _hw: &HwCore) {}
}

/// A host capture stream, the driver half of QEMU's `HWVoiceIn`.
pub trait PcmIn: Send {
    /// Captures into `buf` and returns how many bytes were filled.
    fn read(&mut self, info: &PcmInfo, buf: &mut [u8]) -> usize;

    /// Pulls captured audio from the host into the buffer.
    fn run_buffer_in(&mut self, _hw: &mut HwCore) {}

    /// Up to `size` captured bytes.
    fn get_buffer_in<'a>(&'a mut self, hw: &'a mut HwCore, size: usize) -> &'a [u8] {
        generic_get_buffer_in(hw, size)
    }

    /// Releases `size` bytes returned by `get_buffer_in`.
    fn put_buffer_in(&mut self, hw: &mut HwCore, size: usize) {
        generic_put_buffer_in(hw, size)
    }

    /// Starts or stops the host stream.
    fn enable_in(&mut self, _enable: bool) {}

    /// Applies a device volume in the host, for drivers that report `volume_in`.
    fn volume_in(&mut self, _vol: &Volume) {}

    /// Closes the host stream.
    fn fini_in(&mut self, _hw: &HwCore) {}
}

/// What a driver gets when it opens a host stream.
#[derive(Debug)]
pub struct InitCtx<'a> {
    /// The audiodev, for driver-specific options.
    pub dev: &'a Audiodev,
    /// The options for this direction.
    pub pdo: &'a Pdo,
    /// The virtual clock.
    pub clock: &'a VClock,
}

/// A host stream a driver opened.
#[derive(Debug)]
pub struct HwInit<P> {
    /// The stream.
    pub pcm: P,
    /// The layout the host runs at.
    pub info: PcmInfo,
    /// The host buffer size in frames.
    pub samples: usize,
    /// Whether the driver runs the voice itself instead of the audio timer.
    pub poll_mode: bool,
}

/// A host audio driver, the class half of QEMU's `AudioMixengBackendClass`.
pub trait Driver: Send + Sync + fmt::Debug {
    /// The QOM type name, such as `audio-wav`.
    fn type_name(&self) -> &'static str;
    /// The most playback voices the driver can open.
    fn max_voices_out(&self) -> i32;
    /// The most capture voices the driver can open.
    fn max_voices_in(&self) -> i32;
    /// Whether the driver can capture at all.
    fn has_init_in(&self) -> bool {
        true
    }
    /// Whether the driver applies playback volume itself.
    fn volume_out(&self) -> bool {
        false
    }
    /// Whether the driver applies capture volume itself.
    fn volume_in(&self) -> bool {
        false
    }
    /// Opens a playback stream. The driver reports its own errors.
    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>>;
    /// Opens a capture stream. The driver reports its own errors.
    fn init_in(&self, _ctx: &InitCtx<'_>, _as: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn info_and_clear() {
        let as_ =
            AudSettings { freq: 44100, nchannels: 2, fmt: AudioFormat::U16, big_endian: true };
        let info = PcmInfo::new(&as_);
        assert_eq!(info.bytes_per_frame, 4);
        assert_eq!(info.bytes_per_second, 176400);
        assert_eq!(info.swap_endianness, !HOST_BIG_ENDIAN);
        assert!(info.eq_settings(&as_));
        let mut b = [0u8; 8];
        info.clear_buf(&mut b, 2);
        assert_eq!(b, [0x7f, 0xff, 0x7f, 0xff, 0x7f, 0xff, 0x7f, 0xff]);
        assert_eq!(as_.to_string(), "frequency=44100 nchannels=2 fmt=u16 endian=big");
    }

    #[test]
    fn rate_ctl_follows_clock() {
        let clock = VClock::default();
        let c = Clock::manual(ruvm_base::ClockType::Virtual);
        assert!(clock.set(c.clone()));
        let info = PcmInfo::new(&AudSettings {
            freq: 44100,
            nchannels: 2,
            fmt: AudioFormat::S16,
            big_endian: false,
        });
        let mut r = RateCtl::new(clock);
        c.set_ns(10_000_000);
        assert_eq!(r.get_bytes(&info, 100_000), 1764);
        assert_eq!(r.get_bytes(&info, 100_000), 0);
        c.set_ns(20_000_000);
        assert_eq!(r.get_bytes(&info, 1000), 1000);
        assert_eq!(r.get_bytes(&info, 100_000), 764);
    }
}
