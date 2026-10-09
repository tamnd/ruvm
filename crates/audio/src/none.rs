// SPDX-License-Identifier: GPL-2.0-or-later

//! The `none` driver, QEMU's `audio/noaudio.c`. Playback is thrown away and capture returns
//! silence, both at the real rate of the stream.

use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, RateCtl,
    generic_buffer_get_free, generic_run_buffer_in, generic_run_buffer_out,
};

/// The `none` driver.
#[derive(Debug, Default)]
pub struct NoneDriver;

struct NoVoice {
    rate: RateCtl,
}

impl PcmOut for NoVoice {
    fn write(&mut self, info: &PcmInfo, buf: &[u8]) -> usize {
        self.rate.get_bytes(info, buf.len())
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
}

impl PcmIn for NoVoice {
    fn read(&mut self, info: &PcmInfo, buf: &mut [u8]) -> usize {
        let bytes = self.rate.get_bytes(info, buf.len());
        info.clear_buf(buf, bytes / info.bytes_per_frame);
        bytes
    }

    fn run_buffer_in(&mut self, hw: &mut HwCore) {
        generic_run_buffer_in(self, hw);
    }

    fn enable_in(&mut self, enable: bool) {
        if enable {
            self.rate.start();
        }
    }
}

impl Driver for NoneDriver {
    fn type_name(&self) -> &'static str {
        "audio-none"
    }

    fn max_voices_out(&self) -> i32 {
        i32::MAX
    }

    fn max_voices_in(&self) -> i32 {
        i32::MAX
    }

    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        Some(HwInit {
            pcm: Box::new(NoVoice { rate: RateCtl::new(ctx.clock.clone()) }),
            info: PcmInfo::new(as_),
            samples: 1024,
            poll_mode: false,
        })
    }

    fn init_in(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        Some(HwInit {
            pcm: Box::new(NoVoice { rate: RateCtl::new(ctx.clock.clone()) }),
            info: PcmInfo::new(as_),
            samples: 1024,
            poll_mode: false,
        })
    }
}
