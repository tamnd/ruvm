// SPDX-License-Identifier: GPL-2.0-or-later

//! The sound of the PC speaker, the audio half of QEMU's hw/audio/pcspk.c. Port 0x61 itself
//! belongs to the board, which passes the gate and data bits it is written with to
//! [`PcSpkAudio::io_write`]. While channel 2 of the PIT runs in mode 3, the voice plays a
//! square wave at the channel's frequency, 32 kHz unsigned 8-bit mono.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_audio::{AudSettings, AudioBackend, SwVoiceOut};
use ruvm_base::error_report;
use ruvm_hw_timer::i8254::{I8254, PIT_FREQ};
use ruvm_qapi::types::AudioFormat;

const PCSPK_BUF_LEN: usize = 1792;
const PCSPK_SAMPLE_RATE: u32 = 32000;
const PCSPK_MAX_FREQ: u32 = PCSPK_SAMPLE_RATE >> 1;
/// The smallest count whose frequency the sample rate can still reproduce.
const PCSPK_MIN_COUNT: u32 = PIT_FREQ.div_ceil(PCSPK_MAX_FREQ);

/// The voice and the wave it loops.
struct State {
    voice: Option<SwVoiceOut>,
    sample_buf: [u8; PCSPK_BUF_LEN],
    pit_count: u32,
    samples: usize,
    play_pos: usize,
}

impl State {
    /// `generate_samples()`: a whole number of periods of the square wave, so the buffer
    /// loops without a gap, or silence.
    fn generate_samples(&mut self) {
        if self.pit_count != 0 {
            let m = PCSPK_SAMPLE_RATE * self.pit_count;
            let n = ((u64::from(PIT_FREQ) << 32) / u64::from(m)) as u32;
            let total = PCSPK_BUF_LEN as u64 * u64::from(PIT_FREQ);
            let aligned = total / u64::from(m) * u64::from(m);
            self.samples = ((aligned / u64::from(PIT_FREQ >> 1) + 1) >> 1) as usize;
            for i in 0..self.samples {
                let phase = n.wrapping_mul(i as u32) >> 25;
                self.sample_buf[i] = ((64 & phase) as u8).wrapping_sub(32);
            }
        } else {
            self.samples = PCSPK_BUF_LEN;
            self.sample_buf = [128; PCSPK_BUF_LEN];
        }
    }
}

struct Inner {
    pit: Arc<I8254>,
    be: Arc<AudioBackend>,
    s: Mutex<State>,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `pcspk_callback()`: tops the voice up with the wave for the current count.
    fn callback(&self, mut free: usize) {
        let ch = self.pit.get_channel_info(2);
        if ch.mode != 3 {
            return;
        }
        let mut n = ch.initial_count as u32;
        // Frequencies the sample rate cannot reproduce play as silence.
        if n < PCSPK_MIN_COUNT {
            n = 0;
        }
        let mut s = self.lock();
        if s.pit_count != n {
            s.pit_count = n;
            s.play_pos = 0;
            s.generate_samples();
        }
        while free > 0 {
            let pos = s.play_pos;
            let len = (s.samples - pos).min(free);
            let written = self.be.write(s.voice, &s.sample_buf[pos..pos + len]);
            if written == 0 {
                break;
            }
            s.play_pos = (pos + written) % s.samples;
            free -= written;
        }
    }
}

/// The speaker's voice, which `-machine pcspk-audiodev=` asks for.
pub struct PcSpkAudio {
    inner: Arc<Inner>,
}

impl fmt::Debug for PcSpkAudio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PcSpkAudio").finish_non_exhaustive()
    }
}

impl PcSpkAudio {
    /// `pcspk_audio_init()`: opens the voice on `be`. The voice reads the count of channel 2
    /// of `pit`.
    pub fn new(pit: Arc<I8254>, be: Arc<AudioBackend>) -> PcSpkAudio {
        let inner = Arc::new_cyclic(|me: &Weak<Inner>| {
            let me = me.clone();
            let cb = Arc::new(move |free: usize| {
                if let Some(me) = me.upgrade() {
                    me.callback(free);
                }
            });
            let as_ = AudSettings {
                freq: PCSPK_SAMPLE_RATE as i32,
                nchannels: 1,
                fmt: AudioFormat::U8,
                big_endian: false,
            };
            let voice = be.open_out(None, "pcspk", cb, &as_);
            if voice.is_none() {
                error_report("pcspk: Could not open voice");
            }
            let s = State {
                voice,
                sample_buf: [0; PCSPK_BUF_LEN],
                pit_count: 0,
                samples: 0,
                play_pos: 0,
            };
            Inner { pit, be, s: Mutex::new(s) }
        });
        PcSpkAudio { inner }
    }

    /// The audio side of `pcspk_io_write()`: setting the gate restarts the wave, and the
    /// voice plays while both the gate and the data bit are set.
    pub fn io_write(&self, gate: bool, data_on: bool) {
        let mut s = self.inner.lock();
        if s.voice.is_none() {
            return;
        }
        if gate {
            s.play_pos = 0;
        }
        self.inner.be.set_active_out(s.voice, gate && data_on);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wave(count: u32) -> Vec<u8> {
        let mut s = State {
            voice: None,
            sample_buf: [0; PCSPK_BUF_LEN],
            pit_count: count,
            samples: 0,
            play_pos: 0,
        };
        s.generate_samples();
        s.sample_buf[..s.samples].to_vec()
    }

    #[test]
    fn the_wave_holds_whole_periods() {
        assert_eq!(PCSPK_MIN_COUNT, 75);
        // 1193 is close to 1 kHz, 32 samples a period, and the buffer holds 56 of them.
        let w = wave(1193);
        assert_eq!(w.len(), 1792);
        assert_eq!(&w[..4], &[0xe0; 4]);
        assert!(w.iter().all(|&b| b == 0x20 || b == 0xe0));
        let flips = w.windows(2).filter(|p| p[0] != p[1]).count();
        assert_eq!(flips, 111);
        assert_eq!(wave(0), vec![128; PCSPK_BUF_LEN]);
    }
}
