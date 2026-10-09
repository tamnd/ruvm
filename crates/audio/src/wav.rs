// SPDX-License-Identifier: GPL-2.0-or-later

//! The `wav` driver, QEMU's `audio/wavaudio.c`. It writes playback to a WAVE file at the real
//! rate of the stream, in the fixed format of the audiodev, and patches the lengths in the
//! header when the stream closes.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_qapi::types::{AudioFormat, AudiodevU};

use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmInfo, PcmOut, RateCtl,
    generic_buffer_get_free, generic_run_buffer_out,
};

/// The `wav` driver.
#[derive(Debug, Default)]
pub struct WavDriver;

struct WavVoice {
    f: Option<File>,
    rate: RateCtl,
    total_samples: i32,
}

fn le_store(buf: &mut [u8], val: u32, len: usize) {
    buf[..len].copy_from_slice(&val.to_le_bytes()[..len]);
}

impl PcmOut for WavVoice {
    fn write(&mut self, info: &PcmInfo, buf: &[u8]) -> usize {
        let bytes = self.rate.get_bytes(info, buf.len());
        assert!(bytes % info.bytes_per_frame == 0);
        if bytes > 0 {
            if let Some(f) = self.f.as_mut() {
                if let Err(e) = f.write_all(&buf[..bytes]) {
                    error_report(&format!("wav: fwrite of {bytes} bytes failed: {}", strerror(&e)));
                }
            }
        }
        self.total_samples = self.total_samples.wrapping_add((bytes / info.bytes_per_frame) as i32);
        bytes
    }

    fn buffer_get_free(&mut self, hw: &HwCore) -> Option<usize> {
        Some(generic_buffer_get_free(hw))
    }

    fn run_buffer_out(&mut self, hw: &mut HwCore) {
        generic_run_buffer_out(self, hw);
    }

    fn enable_out(&mut self, enable: bool) {
        if enable {
            self.rate.start();
        }
    }

    fn fini_out(&mut self, hw: &HwCore) {
        let Some(mut f) = self.f.take() else { return };
        let datalen = (self.total_samples as u32).wrapping_mul(hw.info.bytes_per_frame as u32);
        let rifflen = datalen.wrapping_add(36);
        let steps = [
            (SeekFrom::Start(4), rifflen, "fseek to rlen failed", "failed to write rlen"),
            (SeekFrom::Current(32), datalen, "fseek to dlen failed", "failed to write dlen"),
        ];
        for (pos, val, seek_msg, write_msg) in steps {
            if let Err(e) = f.seek(pos) {
                error_report(&format!("wav: {seek_msg}: {}", strerror(&e)));
                break;
            }
            if let Err(e) = f.write_all(&val.to_le_bytes()) {
                error_report(&format!("wav: {write_msg}: {}", strerror(&e)));
                break;
            }
        }
    }
}

impl Driver for WavDriver {
    fn type_name(&self) -> &'static str {
        "audio-wav"
    }

    fn max_voices_out(&self) -> i32 {
        1
    }

    fn max_voices_in(&self) -> i32 {
        0
    }

    fn has_init_in(&self) -> bool {
        false
    }

    fn init_out(&self, ctx: &InitCtx<'_>, _as: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let mut hdr: [u8; 44] = [
            0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d,
            0x74, 0x20, 0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x44, 0xac, 0x00, 0x00,
            0x10, 0xb1, 0x02, 0x00, 0x04, 0x00, 0x10, 0x00, 0x64, 0x61, 0x74, 0x61, 0x00, 0x00,
            0x00, 0x00,
        ];
        let path = match &ctx.dev.u {
            AudiodevU::Wav(w) => w.path.as_deref(),
            _ => None,
        };
        let path = path.unwrap_or("qemu.wav");
        let mut wav_as = ctx.pdo.settings();
        let stereo = u32::from(wav_as.nchannels == 2);
        let bits16 = match wav_as.fmt {
            AudioFormat::S8 | AudioFormat::U8 => 0,
            AudioFormat::S16 | AudioFormat::U16 => 1,
            AudioFormat::S32 | AudioFormat::U32 => {
                error_report("wav: WAVE files cannot handle 32-bit formats");
                return None;
            }
            AudioFormat::F32 => {
                error_report("wav: WAVE files cannot handle float formats");
                return None;
            }
        };
        hdr[34] = if bits16 == 1 { 0x10 } else { 0x08 };
        wav_as.big_endian = false;
        let info = PcmInfo::new(&wav_as);
        le_store(&mut hdr[22..], info.nchannels as u32, 2);
        le_store(&mut hdr[24..], info.freq as u32, 4);
        le_store(&mut hdr[28..], (info.freq as u32) << (bits16 + stereo), 4);
        le_store(&mut hdr[32..], 1 << (bits16 + stereo), 2);

        let mut f = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                error_report(&format!("wav: failed to open wave file '{path}': {}", strerror(&e)));
                return None;
            }
        };
        if let Err(e) = f.write_all(&hdr) {
            error_report(&format!("wav: failed to write header: {}", strerror(&e)));
            return None;
        }
        Some(HwInit {
            pcm: Box::new(WavVoice {
                f: Some(f),
                rate: RateCtl::new(ctx.clock.clone()),
                total_samples: 0,
            }),
            info,
            samples: 1024,
            poll_mode: false,
        })
    }
}
