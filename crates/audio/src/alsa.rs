// SPDX-License-Identifier: GPL-2.0-or-later

//! The `alsa` driver, QEMU's `audio/alsaaudio.c`, over the `alsa` crate. It opens one PCM per
//! voice in non-blocking mode and is run by the audio timer. QEMU can also run a voice from the
//! poll descriptors of the PCM when `try-poll` is on; that mode is not here, so the timer runs
//! every voice whatever `try-poll` says.

use std::io::Write;

use alsa::pcm::{Access, Format, HwParams, PCM};
use alsa::{Direction, ValueOr};
use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, warn_report};
use ruvm_qapi::types::{AudioFormat, AudiodevAlsaPerDirectionOptions, AudiodevU};

use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, generic_buffer_get_free,
    generic_run_buffer_in, generic_run_buffer_out,
};

/// The `alsa` driver.
#[derive(Debug, Default)]
pub struct AlsaDriver;

/// `audio_alsa_realize()` fills these in when the user leaves them out, without marking them
/// as set, so `alsa_open()` can still tell.
const DEFAULT_PERIOD_LENGTH: u32 = 5805;
const DEFAULT_BUFFER_LENGTH: u32 = 92880;

/// `snd_strerror()`: the C library text for an errno, or ALSA's own for its two codes.
fn snd_strerror(errno: i32) -> String {
    match errno {
        500_000 => "Sound protocol is not compatible".to_string(),
        500_001 => "Lisp encountered an error during acall".to_string(),
        e => strerror(&std::io::Error::from_raw_os_error(e)),
    }
}

/// `error_printf()`: text on stderr with no program name in front.
fn error_printf(s: &str) {
    let mut err = std::io::stderr().lock();
    let _ = err.write_all(s.as_bytes());
    let _ = err.flush();
}

/// `alsa_logerr()`.
fn logerr(err: &alsa::Error, msg: &str) {
    error_printf(&format!("alsa: {msg} Reason: {}\n", snd_strerror(err.errno())));
}

/// `alsa_logerr2()`.
fn logerr2(err: &alsa::Error, typ: &str, msg: &str) {
    error_printf(&format!(
        "alsa: Could not initialize {typ}:{msg} Reason: {}\n",
        snd_strerror(err.errno())
    ));
}

/// `alsa_recover()`.
fn recover(pcm: &PCM) -> bool {
    match pcm.prepare() {
        Ok(()) => true,
        Err(e) => {
            logerr(&e, &format!("Failed to prepare handle {pcm:p}"));
            false
        }
    }
}

/// `alsa_resume()`.
fn resume(pcm: &PCM) -> bool {
    match pcm.resume() {
        Ok(()) => true,
        Err(e) => {
            logerr(&e, &format!("Failed to resume handle {pcm:p}"));
            false
        }
    }
}

/// `aud_to_alsafmt()`.
fn aud_to_alsafmt(fmt: AudioFormat, big_endian: bool) -> Format {
    match (fmt, big_endian) {
        (AudioFormat::S8, _) => Format::S8,
        (AudioFormat::U8, _) => Format::U8,
        (AudioFormat::S16, false) => Format::S16LE,
        (AudioFormat::S16, true) => Format::S16BE,
        (AudioFormat::U16, false) => Format::U16LE,
        (AudioFormat::U16, true) => Format::U16BE,
        (AudioFormat::S32, false) => Format::S32LE,
        (AudioFormat::S32, true) => Format::S32BE,
        (AudioFormat::U32, false) => Format::U32LE,
        (AudioFormat::U32, true) => Format::U32BE,
        (AudioFormat::F32, false) => Format::FloatLE,
        (AudioFormat::F32, true) => Format::FloatBE,
    }
}

/// `alsa_to_audfmt()`: the format and whether it is big endian.
fn alsa_to_audfmt(fmt: Format) -> Option<(AudioFormat, bool)> {
    Some(match fmt {
        Format::S8 => (AudioFormat::S8, false),
        Format::U8 => (AudioFormat::U8, false),
        Format::S16LE => (AudioFormat::S16, false),
        Format::U16LE => (AudioFormat::U16, false),
        Format::S16BE => (AudioFormat::S16, true),
        Format::U16BE => (AudioFormat::U16, true),
        Format::S32LE => (AudioFormat::S32, false),
        Format::U32LE => (AudioFormat::U32, false),
        Format::S32BE => (AudioFormat::S32, true),
        Format::U32BE => (AudioFormat::U32, true),
        Format::FloatLE => (AudioFormat::F32, false),
        Format::FloatBE => (AudioFormat::F32, true),
        _ => {
            warn_report(&format!("alsa: Unrecognized audio format {}", fmt as i32));
            return None;
        }
    })
}

/// `alsa_set_threshold()`.
fn set_threshold(pcm: &PCM, threshold: i64) {
    let swp = match pcm.sw_params_current() {
        Ok(p) => p,
        Err(e) => {
            error_report("alsa: Could not fully initialize DAC");
            logerr(&e, "Failed to get current software parameters");
            return;
        }
    };
    if let Err(e) = swp.set_start_threshold(threshold as alsa::pcm::Frames) {
        error_report("alsa: Could not fully initialize DAC");
        logerr(&e, &format!("Failed to set software threshold to {threshold}"));
        return;
    }
    if let Err(e) = pcm.sw_params(&swp) {
        error_report("alsa: Could not fully initialize DAC");
        logerr(&e, "Failed to set software parameters");
    }
}

/// What `alsa_open()` got from the device, QEMU's `struct alsa_params_obt`.
struct Obtained {
    settings: AudSettings,
    samples: usize,
}

/// `alsa_open()`.
fn alsa_open(in_: bool, as_: &AudSettings, ctx: &InitCtx<'_>) -> Option<(PCM, Obtained)> {
    let (apdo, threshold) = match &ctx.dev.u {
        AudiodevU::Alsa(o) => (if in_ { o.in_.as_ref() } else { o.out.as_ref() }, o.threshold),
        _ => (None, None),
    };
    let default = AudiodevAlsaPerDirectionOptions::default();
    let apdo = apdo.unwrap_or(&default);
    // A C string ends at the first NUL.
    let pcm_name = apdo.dev.as_deref().unwrap_or("default").split('\0').next().unwrap_or("");
    let typ = if in_ { "ADC" } else { "DAC" };
    let req_fmt = aud_to_alsafmt(as_.fmt, as_.big_endian);

    let dir = if in_ { Direction::Capture } else { Direction::Playback };
    let pcm = match PCM::new(pcm_name, dir, true) {
        Ok(p) => p,
        Err(e) => {
            logerr2(&e, typ, &format!("Failed to open `{pcm_name}'"));
            return None;
        }
    };

    let obtained = {
        let hwp = match HwParams::any(&pcm) {
            Ok(p) => p,
            Err(e) => {
                logerr2(&e, typ, "Failed to initialize hardware parameters");
                return None;
            }
        };
        if let Err(e) = hwp.set_access(Access::RWInterleaved) {
            logerr2(&e, typ, "Failed to set access type");
            return None;
        }
        if let Err(e) = hwp.set_format(req_fmt) {
            logerr2(&e, typ, &format!("Failed to set format {}", req_fmt as i32));
        }
        let freq = match hwp.set_rate_near(as_.freq as u32, ValueOr::Nearest) {
            Ok(f) => f,
            Err(e) => {
                logerr2(&e, typ, &format!("Failed to set frequency {}", as_.freq));
                return None;
            }
        };
        let nchannels = match hwp.set_channels_near(as_.nchannels as u32) {
            Ok(n) => n,
            Err(e) => {
                logerr2(&e, typ, &format!("Failed to set number of channels {}", as_.nchannels));
                return None;
            }
        };

        let buffer_length = apdo.buffer_length.unwrap_or(DEFAULT_BUFFER_LENGTH);
        if buffer_length != 0 {
            let btime = match hwp.set_buffer_time_near(buffer_length, ValueOr::Nearest) {
                Ok(t) => t,
                Err(e) => {
                    logerr2(
                        &e,
                        typ,
                        &format!("Failed to set buffer time to {}", buffer_length as i32),
                    );
                    return None;
                }
            };
            if apdo.buffer_length.is_some() && btime != buffer_length {
                warn_report(&format!(
                    "alsa: Requested buffer time {} was rejected, using {btime}",
                    buffer_length as i32
                ));
            }
        }

        let period_length = apdo.period_length.unwrap_or(DEFAULT_PERIOD_LENGTH);
        if period_length != 0 {
            let ptime = match hwp.set_period_time_near(period_length, ValueOr::Nearest) {
                Ok(t) => t,
                Err(e) => {
                    logerr2(
                        &e,
                        typ,
                        &format!("Failed to set period time to {}", period_length as i32),
                    );
                    return None;
                }
            };
            if apdo.period_length.is_some() && ptime != period_length {
                warn_report(&format!(
                    "alsa: Requested period time {} was rejected, using {}",
                    period_length as i32, ptime as i32
                ));
            }
        }

        if let Err(e) = pcm.hw_params(&hwp) {
            logerr2(&e, typ, "Failed to apply audio parameters");
            return None;
        }
        let samples = match hwp.get_buffer_size() {
            Ok(s) => s,
            Err(e) => {
                logerr2(&e, typ, "Failed to get buffer size");
                return None;
            }
        };
        let obtfmt = match hwp.get_format() {
            Ok(f) => f,
            Err(e) => {
                logerr2(&e, typ, "Failed to get format");
                return None;
            }
        };
        let Some((fmt, big_endian)) = alsa_to_audfmt(obtfmt) else {
            error_report(&format!("alsa: Invalid format was returned {}", obtfmt as i32));
            return None;
        };
        Obtained {
            settings: AudSettings {
                freq: freq as i32,
                nchannels: nchannels as i32,
                fmt,
                big_endian,
            },
            samples: samples as usize,
        }
    };

    if let Err(e) = pcm.prepare() {
        logerr2(&e, typ, &format!("Could not prepare handle {:p}", &pcm));
        return None;
    }

    if let Some(threshold) = threshold.filter(|&t| !in_ && t != 0) {
        let as_ = AudSettings { freq: obtained.settings.freq, ..*as_ };
        set_threshold(&pcm, i64::from(ctx.pdo.buffer_frames(&as_, threshold)));
    }

    Some((pcm, obtained))
}

/// What `alsa_voice_ctl()` does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VoiceCtl {
    Pause,
    Prepare,
    Start,
}

/// `alsa_voice_ctl()`.
fn voice_ctl(pcm: &PCM, typ: &str, ctl: VoiceCtl) {
    if ctl == VoiceCtl::Pause {
        if let Err(e) = pcm.drop() {
            logerr(&e, &format!("Could not stop {typ}"));
        }
        return;
    }
    if let Err(e) = pcm.prepare() {
        logerr(&e, &format!("Could not prepare handle for {typ}"));
        return;
    }
    if ctl == VoiceCtl::Start {
        if let Err(e) = pcm.start() {
            logerr(&e, &format!("Could not start handle for {typ}"));
        }
    }
}

struct AlsaVoiceOut {
    pcm: PCM,
}

struct AlsaVoiceIn {
    pcm: PCM,
}

impl PcmOut for AlsaVoiceOut {
    fn write(&mut self, info: &PcmInfo, buf: &[u8]) -> usize {
        let bpf = info.bytes_per_frame;
        let mut pos = 0;
        let mut len_frames = buf.len() / bpf;
        while len_frames > 0 {
            let src = &buf[pos..pos + len_frames * bpf];
            let written = match self.pcm.io_bytes().writei(src) {
                Ok(0) => return pos,
                Ok(n) => n,
                Err(e) => match e.errno() {
                    libc::EPIPE => {
                        if !recover(&self.pcm) {
                            logerr(&e, &format!("Failed to write {len_frames} frames"));
                            return pos;
                        }
                        continue;
                    }
                    libc::ESTRPIPE => {
                        // The stream is suspended and waits for the application to recover it.
                        if !resume(&self.pcm) {
                            logerr(&e, &format!("Failed to write {len_frames} frames"));
                            return pos;
                        }
                        continue;
                    }
                    libc::EAGAIN => return pos,
                    _ => {
                        logerr(
                            &e,
                            &format!("Failed to write {len_frames} frames from {:p}", src.as_ptr()),
                        );
                        return pos;
                    }
                },
            };
            pos += written * bpf;
            if written < len_frames {
                break;
            }
            len_frames -= written;
        }
        pos
    }

    fn buffer_get_free(&mut self, hw: &HwCore) -> Option<usize> {
        let mut avail = self.pcm.avail_update();
        if let Err(e) = &avail {
            if e.errno() == libc::EPIPE && recover(&self.pcm) {
                avail = self.pcm.avail_update();
            }
        }
        let avail = avail.unwrap_or_else(|e| {
            logerr(&e, "Could not obtain number of available frames");
            0
        });
        let bpf = hw.info.bytes_per_frame;
        let alsa_free = avail as usize * bpf;
        let generic_in_use = hw.samples * bpf - generic_buffer_get_free(hw);
        // Only reached when avail_update() promised more frames than writei() took. What is
        // left in the generic buffer has to fit first.
        Some(alsa_free.saturating_sub(generic_in_use))
    }

    fn run_buffer_out(&mut self, hw: &mut HwCore) {
        generic_run_buffer_out(self, hw);
    }

    fn enable_out(&mut self, enable: bool) {
        let ctl = if enable { VoiceCtl::Prepare } else { VoiceCtl::Pause };
        voice_ctl(&self.pcm, "playback", ctl);
    }
}

impl PcmIn for AlsaVoiceIn {
    fn read(&mut self, info: &PcmInfo, buf: &mut [u8]) -> usize {
        let bpf = info.bytes_per_frame;
        let mut pos = 0;
        let mut len = buf.len();
        while len > 0 {
            let frames = len / bpf;
            let dst = &mut buf[pos..pos + frames * bpf];
            let nread = match self.pcm.io_bytes().readi(dst) {
                Ok(0) => return pos,
                Ok(n) => n,
                Err(e) => match e.errno() {
                    libc::EPIPE => {
                        if !recover(&self.pcm) {
                            logerr(&e, &format!("Failed to read {len} frames"));
                            return pos;
                        }
                        continue;
                    }
                    libc::EAGAIN => return pos,
                    _ => {
                        logerr(&e, &format!("Failed to read {len} frames to {:p}", dst.as_ptr()));
                        return pos;
                    }
                },
            };
            pos += nread * bpf;
            len -= nread * bpf;
        }
        pos
    }

    fn run_buffer_in(&mut self, hw: &mut HwCore) {
        generic_run_buffer_in(self, hw);
    }

    fn enable_in(&mut self, enable: bool) {
        let ctl = if enable { VoiceCtl::Start } else { VoiceCtl::Pause };
        voice_ctl(&self.pcm, "capture", ctl);
    }
}

impl Driver for AlsaDriver {
    fn type_name(&self) -> &'static str {
        "audio-alsa"
    }

    fn max_voices_out(&self) -> i32 {
        i32::MAX
    }

    fn max_voices_in(&self) -> i32 {
        i32::MAX
    }

    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let (pcm, obt) = alsa_open(false, as_, ctx)?;
        Some(HwInit {
            pcm: Box::new(AlsaVoiceOut { pcm }),
            info: PcmInfo::new(&obt.settings),
            samples: obt.samples,
            poll_mode: false,
        })
    }

    fn init_in(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        let (pcm, obt) = alsa_open(true, as_, ctx)?;
        Some(HwInit {
            pcm: Box::new(AlsaVoiceIn { pcm }),
            info: PcmInfo::new(&obt.settings),
            samples: obt.samples,
            poll_mode: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_round_trip() {
        let all = [
            AudioFormat::S8,
            AudioFormat::U8,
            AudioFormat::S16,
            AudioFormat::U16,
            AudioFormat::S32,
            AudioFormat::U32,
            AudioFormat::F32,
        ];
        for fmt in all {
            for be in [false, true] {
                let (f, e) = alsa_to_audfmt(aud_to_alsafmt(fmt, be)).unwrap();
                assert_eq!(f, fmt);
                let one_byte = matches!(fmt, AudioFormat::S8 | AudioFormat::U8);
                assert_eq!(e, be && !one_byte);
            }
        }
        assert_eq!(snd_strerror(libc::ENOENT), "No such file or directory");
        assert_eq!(snd_strerror(500_000), "Sound protocol is not compatible");
    }
}
