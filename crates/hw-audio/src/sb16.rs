// SPDX-License-Identifier: GPL-2.0-or-later

//! The Creative Sound Blaster 16, a port of QEMU's hw/audio/sb16.c. The DSP takes commands
//! on the ISA ports from `iobase`, plays 8-bit transfers from one channel of the i8257 pair
//! and 16-bit ones from another, and raises its ISA interrupt at the end of each block. The
//! mixer only keeps the registers it is written.
//!
//! The `LOG_GUEST_ERROR` and `LOG_UNIMP` messages are not printed, since the workspace has no
//! `-d` log mask, and there is no migration state yet.

use std::fmt;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_audio::{AudSettings, AudioBackend, AudioCallback, SwVoiceOut};
use ruvm_base::warn_report;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::{Clock, Timer};
use ruvm_mem::MmioOps;
use ruvm_qapi::types::AudioFormat;

use crate::i8257::{I8257, IsaDma};
use crate::portio::{Portio, portio_list};

/// `TYPE_SB16`.
pub const TYPE_SB16: &str = "sb16";

const E3: &[u8] = b"COPYRIGHT (C) CREATIVE TECHNOLOGY LTD, 1992.\0";

const SAMPLE_RATE_MIN: i32 = 5000;
const SAMPLE_RATE_MAX: i32 = 45000;

const DMA8_AUTO: i32 = 1;
const DMA8_HIGH: i32 = 2;

const NANOSECONDS_PER_SECOND: i64 = 1_000_000_000;

/// The qdev properties of the card.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sb16Config {
    /// `version`, what command 0xe1 answers.
    pub ver: u32,
    /// `iobase`.
    pub port: u32,
    /// `irq`.
    pub irq: u32,
    /// `dma`, the 8-bit channel.
    pub dma: u32,
    /// `dma16`, the 16-bit channel.
    pub hdma: u32,
}

impl Default for Sb16Config {
    fn default() -> Self {
        Sb16Config { ver: 0x0405, port: 0x220, irq: 5, dma: 1, hdma: 5 }
    }
}

/// The part of `SB16State` that changes.
struct State {
    in_index: i32,
    out_data_len: i32,
    fmt_stereo: i32,
    fmt_signed: i32,
    fmt_bits: i32,
    fmt: AudioFormat,
    dma_auto: i32,
    block_size: i32,
    fifo: i32,
    freq: i32,
    time_const: i32,
    speaker: i32,
    needed_bytes: i32,
    cmd: i32,
    use_hdma: i32,
    highspeed: i32,
    can_write: i32,
    v2x6: i32,

    csp_param: u8,
    csp_value: u8,
    csp_mode: u8,
    csp_regs: [u8; 256],
    csp_reg83: [u8; 4],
    csp_reg83r: i32,
    csp_reg83w: i32,

    in2_data: [u8; 10],
    out_data: [u8; 50],
    test_reg: u8,
    last_read_byte: u8,
    nzero: i32,

    left_till_irq: i32,

    dma_running: i32,
    bytes_per_second: i32,
    align: i32,
    voice: Option<SwVoiceOut>,

    mixer_nreg: u8,
    mixer_regs: [u8; 256],
}

impl State {
    /// `sb16_initfn()`: everything zero but the command.
    fn new() -> State {
        State {
            in_index: 0,
            out_data_len: 0,
            fmt_stereo: 0,
            fmt_signed: 0,
            fmt_bits: 0,
            fmt: AudioFormat::U8,
            dma_auto: 0,
            block_size: 0,
            fifo: 0,
            freq: 0,
            time_const: 0,
            speaker: 0,
            needed_bytes: 0,
            cmd: -1,
            use_hdma: 0,
            highspeed: 0,
            can_write: 0,
            v2x6: 0,
            csp_param: 0,
            csp_value: 0,
            csp_mode: 0,
            csp_regs: [0; 256],
            csp_reg83: [0; 4],
            csp_reg83r: 0,
            csp_reg83w: 0,
            in2_data: [0; 10],
            out_data: [0; 50],
            test_reg: 0,
            last_read_byte: 0,
            nzero: 0,
            left_till_irq: 0,
            dma_running: 0,
            bytes_per_second: 0,
            align: 0,
            voice: None,
            mixer_nreg: 0,
            mixer_regs: [0; 256],
        }
    }

    /// `reset_mixer()`.
    fn reset_mixer(&mut self) {
        self.mixer_regs[..0x7f].fill(0xff);
        self.mixer_regs[0x83..].fill(0xff);
        // Master and MIDI volume, 3 bits.
        self.mixer_regs[0x02] = 4;
        self.mixer_regs[0x06] = 4;
        // CD volume, 3 bits, and voice volume, 2 bits.
        self.mixer_regs[0x08] = 0;
        self.mixer_regs[0x0a] = 0;
        // The input filters and source, then the output filter and stereo switch.
        self.mixer_regs[0x0c] = 0;
        self.mixer_regs[0x0e] = 0;
        // Voice, master and MIDI volume, left in d5 to d7 and right in d1 to d3.
        self.mixer_regs[0x04] = (4 << 5) | (4 << 1);
        self.mixer_regs[0x22] = (4 << 5) | (4 << 1);
        self.mixer_regs[0x26] = (4 << 5) | (4 << 1);
        self.mixer_regs[0x30..0x48].fill(0x20);
    }

    /// `dsp_out_data()`.
    fn dsp_out_data(&mut self, val: u8) {
        let n = self.out_data_len as usize;
        if n < self.out_data.len() {
            self.out_data[n] = val;
            self.out_data_len += 1;
        }
    }

    /// `dsp_get_data()`: the bytes of a command come out last first.
    fn dsp_get_data(&mut self) -> u8 {
        if self.in_index != 0 {
            self.in_index -= 1;
            self.in2_data[self.in_index as usize]
        } else {
            warn_report("sb16: buffer underflow");
            0
        }
    }

    /// `dsp_get_lohi()`.
    fn dsp_get_lohi(&mut self) -> i32 {
        let hi = i32::from(self.dsp_get_data());
        let lo = i32::from(self.dsp_get_data());
        (hi << 8) | lo
    }

    /// `dsp_get_hilo()`.
    fn dsp_get_hilo(&mut self) -> i32 {
        let lo = i32::from(self.dsp_get_data());
        let hi = i32::from(self.dsp_get_data());
        (hi << 8) | lo
    }

    /// `speaker()`, which does not reach the voice in QEMU either.
    fn speaker(&mut self, on: i32) {
        self.speaker = on;
    }
}

/// `magic_of_irq()`.
fn magic_of_irq(irq: u32) -> u8 {
    match irq {
        5 => 2,
        7 => 4,
        9 => 1,
        10 => 8,
        _ => 2,
    }
}

/// `restrict_sampling_rate()`.
fn restrict_sampling_rate(freq: i32) -> i32 {
    freq.clamp(SAMPLE_RATE_MIN, SAMPLE_RATE_MAX)
}

struct Inner {
    be: Arc<AudioBackend>,
    cfg: Sb16Config,
    isa_dma: Arc<I8257>,
    isa_hdma: Arc<I8257>,
    pic: IrqLine,
    aux_ts: Timer,
    /// What the voice callback last said it can take. It lives outside the lock, as the
    /// backend calls back while the card may be writing to it.
    audio_free: AtomicI32,
    callback: AudioCallback,
    s: Mutex<State>,
}

/// The card's state while one of its handlers runs. The lock is let go around calls into the
/// DMA controller, which can run the card's transfer handler before it returns.
struct Cx<'a> {
    d: &'a Inner,
    g: Option<MutexGuard<'a, State>>,
}

impl std::ops::Deref for Cx<'_> {
    type Target = State;

    fn deref(&self) -> &State {
        self.g.as_ref().expect("sb16 state is locked")
    }
}

impl std::ops::DerefMut for Cx<'_> {
    fn deref_mut(&mut self) -> &mut State {
        self.g.as_mut().expect("sb16 state is locked")
    }
}

impl Inner {
    fn cx(&self) -> Cx<'_> {
        Cx { d: self, g: Some(self.s.lock().unwrap_or_else(PoisonError::into_inner)) }
    }

    /// `aux_timer()`.
    fn aux_timer(&self) {
        self.cx().can_write = 1;
        self.pic.raise();
    }
}

impl Cx<'_> {
    /// Runs `f` with the lock let go.
    fn unlocked<R>(&mut self, f: impl FnOnce() -> R) -> R {
        self.g = None;
        let r = f();
        self.g = Some(self.d.s.lock().unwrap_or_else(PoisonError::into_inner));
        r
    }

    /// `audio_be_open_out()` on the card's voice.
    fn open_voice(&mut self, nchannels: i32) {
        self.d.audio_free.store(0, Ordering::SeqCst);
        let as_ = AudSettings { freq: self.freq, nchannels, fmt: self.fmt, big_endian: false };
        let cb = Arc::clone(&self.d.callback);
        self.voice = self.d.be.open_out(self.voice, "sb16", cb, &as_);
    }

    /// `control()`: holds or lets go of the DMA request of the channel in use, and starts or
    /// stops the voice.
    fn control(&mut self, hold: bool) {
        let d = self.d;
        let (dma, isa_dma) =
            if self.use_hdma != 0 { (d.cfg.hdma, &d.isa_hdma) } else { (d.cfg.dma, &d.isa_dma) };
        self.dma_running = i32::from(hold);
        self.unlocked(|| {
            if hold {
                isa_dma.hold_dreq(dma);
            } else {
                isa_dma.release_dreq(dma);
            }
        });
        d.be.set_active_out(self.voice, hold);
    }

    /// `continue_dma8()`.
    fn continue_dma8(&mut self) {
        if self.freq > 0 {
            let nch = 1 << self.fmt_stereo;
            self.open_voice(nch);
        }
        self.control(true);
    }

    /// `dma_cmd8()`.
    fn dma_cmd8(&mut self, mask: i32, dma_len: i32) {
        self.fmt = AudioFormat::U8;
        self.use_hdma = 0;
        self.fmt_bits = 8;
        self.fmt_signed = 0;
        self.fmt_stereo = i32::from(self.mixer_regs[0x0e] & 2 != 0);
        if self.time_const == -1 {
            if self.freq <= 0 {
                self.freq = 11025;
            }
        } else {
            let tmp = 256 - self.time_const;
            self.freq = (1_000_000 + (tmp / 2)) / tmp;
        }
        self.freq = restrict_sampling_rate(self.freq);

        if dma_len != -1 {
            self.block_size = dma_len << self.fmt_stereo;
        } else {
            // Act1/PL sets an odd block size with command 0x48 and SecondReality/FC an even
            // one, both in stereo, and this is what makes both work.
            self.block_size &= !self.fmt_stereo;
        }

        self.freq >>= self.fmt_stereo;
        self.left_till_irq = self.block_size;
        self.bytes_per_second = self.freq << self.fmt_stereo;
        self.dma_auto = i32::from(mask & DMA8_AUTO != 0);
        self.align = (1 << self.fmt_stereo) - 1;

        self.continue_dma8();
        self.speaker(1);
    }

    /// `dma_cmd()`, for the 0xb0 to 0xcf commands.
    fn dma_cmd(&mut self, cmd: u8, d0: u8, dma_len: i32) {
        self.use_hdma = i32::from(cmd < 0xc0);
        self.fifo = i32::from((cmd >> 1) & 1);
        self.dma_auto = i32::from((cmd >> 2) & 1);
        self.fmt_signed = i32::from((d0 >> 4) & 1);
        self.fmt_stereo = i32::from((d0 >> 5) & 1);

        match cmd >> 4 {
            11 => self.fmt_bits = 16,
            12 => self.fmt_bits = 8,
            _ => {}
        }

        if self.time_const != -1 {
            let tmp = 256 - self.time_const;
            self.freq = (1_000_000 + (tmp / 2)) / tmp;
            self.time_const = -1;
        }

        let word = i32::from(self.fmt_bits == 16);
        self.block_size = dma_len + 1;
        self.block_size <<= word;
        if self.dma_auto == 0 {
            // DOOM in auto-init mode wants the stereo bit left out and the single transfers
            // of setsound.exe from the Miles Sound System want it in.
            self.block_size <<= self.fmt_stereo;
        }

        self.fmt = match (self.fmt_bits == 16, self.fmt_signed != 0) {
            (true, true) => AudioFormat::S16,
            (true, false) => AudioFormat::U16,
            (false, true) => AudioFormat::S8,
            (false, false) => AudioFormat::U8,
        };

        self.left_till_irq = self.block_size;
        self.bytes_per_second = (self.freq << self.fmt_stereo) << word;
        self.highspeed = 0;
        self.align = (1 << (self.fmt_stereo + word)) - 1;

        if self.freq != 0 {
            let nch = 1 << self.fmt_stereo;
            self.open_voice(nch);
        }

        self.control(true);
        self.speaker(1);
    }

    /// `command()`: starts a command, which may wait for argument bytes.
    fn command(&mut self, cmd: u8) {
        if (0xb0..0xd0).contains(&cmd) {
            self.needed_bytes = 3;
        } else {
            self.needed_bytes = 0;
            match cmd {
                0x03 => self.dsp_out_data(0x10),
                0x04 | 0x0f | 0x10 | 0xe0 | 0xe2 | 0xe4 | 0xf9 => self.needed_bytes = 1,
                0x05 | 0x0e | 0x48 | 0x74..=0x77 | 0x80 => self.needed_bytes = 2,
                0x09 => self.dsp_out_data(0xf8),
                0x14 => {
                    self.needed_bytes = 2;
                    self.block_size = 0;
                }
                // Auto-initialize DMA DAC, 8-bit.
                0x1c => self.dma_cmd8(DMA8_AUTO, -1),
                // Direct ADC, Juice/PL.
                0x20 => self.dsp_out_data(0xff),
                0x40 => {
                    self.freq = -1;
                    self.time_const = -1;
                    self.needed_bytes = 1;
                }
                0x41 | 0x42 => {
                    self.freq = -1;
                    self.time_const = -1;
                    self.needed_bytes = 2;
                }
                0x45 | 0xf2 | 0xf3 => {
                    self.dsp_out_data(0xaa);
                    if cmd != 0x45 {
                        self.mixer_regs[0x82] |= if cmd == 0xf2 { 1 } else { 2 };
                        self.d.pic.raise();
                    }
                }
                0x90 | 0x91 => self.dma_cmd8(i32::from(cmd & 1 == 0) | DMA8_HIGH, -1),
                // Halt DMA, 8-bit and 16-bit.
                0xd0 | 0xd5 => self.control(false),
                0xd1 => self.speaker(1),
                0xd3 => self.speaker(0),
                // Continue 8-bit DMA. KQ6, or Sierra's audblst.drv, changes the frequency
                // between halt and continue.
                0xd4 => self.continue_dma8(),
                0xd6 => self.control(true),
                // Exit auto-init DMA after this block, 16-bit and 8-bit.
                0xd9 | 0xda => self.dma_auto = 0,
                0xe1 => {
                    let ver = self.d.cfg.ver;
                    self.dsp_out_data(ver as u8);
                    self.dsp_out_data((ver >> 8) as u8);
                }
                0xe3 => {
                    for &b in E3.iter().rev() {
                        self.dsp_out_data(b);
                    }
                }
                0xe8 => {
                    let t = self.test_reg;
                    self.dsp_out_data(t);
                }
                0xfa | 0xfc => self.dsp_out_data(0),
                _ => {}
            }
        }
        self.cmd = if self.needed_bytes == 0 { -1 } else { i32::from(cmd) };
    }

    /// `complete()`: runs a command once its argument bytes are in.
    fn complete(&mut self) {
        if (0xb0..0xd0).contains(&self.cmd) {
            let d2 = self.dsp_get_data();
            let d1 = self.dsp_get_data();
            let d0 = self.dsp_get_data();
            if self.cmd & 8 != 0 {
                warn_report(&format!(
                    "sb16: ADC params cmd = 0x{:x} d0 = {d0}, d1 = {d1}, d2 = {d2}",
                    self.cmd
                ));
            } else {
                let cmd = self.cmd as u8;
                self.dma_cmd(cmd, d0, i32::from(d1) + (i32::from(d2) << 8));
            }
        } else {
            match self.cmd {
                0x04 => {
                    self.csp_mode = self.dsp_get_data();
                    self.csp_reg83r = 0;
                    self.csp_reg83w = 0;
                }
                0x05 => {
                    self.csp_param = self.dsp_get_data();
                    self.csp_value = self.dsp_get_data();
                }
                0x0e => {
                    let d0 = self.dsp_get_data();
                    let d1 = self.dsp_get_data();
                    if d1 == 0x83 {
                        let i = (self.csp_reg83r % 4) as usize;
                        self.csp_reg83[i] = d0;
                        self.csp_reg83r += 1;
                    } else {
                        self.csp_regs[usize::from(d1)] = d0;
                    }
                }
                0x0f => {
                    let d0 = self.dsp_get_data();
                    if d0 == 0x83 {
                        let v = self.csp_reg83[(self.csp_reg83w % 4) as usize];
                        self.dsp_out_data(v);
                        self.csp_reg83w += 1;
                    } else {
                        let v = self.csp_regs[usize::from(d0)];
                        self.dsp_out_data(v);
                    }
                }
                0x10 => {
                    let d0 = self.dsp_get_data();
                    warn_report(&format!("sb16: cmd 0x10 d0=0x{d0:x}"));
                }
                0x14 => {
                    let len = self.dsp_get_lohi() + 1;
                    self.dma_cmd8(0, len);
                }
                0x40 => self.time_const = i32::from(self.dsp_get_data()),
                // 0x41 sets the output rate and 0x42 the input rate, but the card has one
                // rate for both, and FT2 sets the output rate with 0x42.
                0x41 | 0x42 => {
                    let freq = self.dsp_get_hilo();
                    self.freq = restrict_sampling_rate(freq);
                }
                0x48 => self.block_size = self.dsp_get_lohi() + 1,
                // ADPCM, ignored.
                0x74..=0x77 => {}
                0x80 => {
                    let freq = if self.freq > 0 { self.freq } else { 11025 };
                    let samples = self.dsp_get_lohi() + 1;
                    let bytes = samples << self.fmt_stereo << i32::from(self.fmt_bits == 16);
                    // muldiv64() of a signed byte count.
                    let ticks = (i128::from(bytes) * i128::from(NANOSECONDS_PER_SECOND)
                        / i128::from(freq)) as i64;
                    if ticks < NANOSECONDS_PER_SECOND / 1024 {
                        self.d.pic.raise();
                    } else if let Some(clock) = self.d.aux_ts.clock() {
                        self.d.aux_ts.modify(clock.get_ns() + ticks);
                    }
                }
                0xe0 => {
                    let d0 = self.dsp_get_data();
                    self.out_data_len = 0;
                    self.dsp_out_data(!d0);
                }
                0xe2 => {}
                0xe4 => self.test_reg = self.dsp_get_data(),
                0xf9 => {
                    let d0 = self.dsp_get_data();
                    let v = match d0 {
                        0x0e => 0xff,
                        0x0f => 0x07,
                        0x37 => 0x38,
                        _ => 0x00,
                    };
                    self.dsp_out_data(v);
                }
                _ => return,
            }
        }
        self.cmd = -1;
    }

    /// `legacy_reset()`: the voice back at 11025 Hz, unsigned 8-bit mono.
    fn legacy_reset(&mut self) {
        self.freq = 11025;
        self.fmt_signed = 0;
        self.fmt_bits = 8;
        self.fmt_stereo = 0;
        self.fmt = AudioFormat::U8;
        let as_ =
            AudSettings { freq: 11025, nchannels: 1, fmt: AudioFormat::U8, big_endian: false };
        let cb = Arc::clone(&self.d.callback);
        self.voice = self.d.be.open_out(self.voice, "sb16", cb, &as_);
    }

    /// `reset()`, the DSP reset a guest asks for through port 6.
    fn reset(&mut self) {
        self.d.pic.lower();
        if self.dma_auto != 0 {
            self.d.pic.raise();
            self.d.pic.lower();
        }

        self.mixer_regs[0x82] = 0;
        self.dma_auto = 0;
        self.in_index = 0;
        self.out_data_len = 0;
        self.left_till_irq = 0;
        self.needed_bytes = 0;
        self.block_size = -1;
        self.nzero = 0;
        self.highspeed = 0;
        self.v2x6 = 0;
        self.cmd = -1;

        self.dsp_out_data(0xaa);
        self.speaker(0);
        self.control(false);
        self.legacy_reset();
    }

    /// `dsp_write()`.
    fn dsp_write(&mut self, iport: u32, val: u32) {
        match iport {
            0x06 => match val {
                0x00 => {
                    if self.v2x6 == 1 {
                        self.reset();
                    }
                    self.v2x6 = 0;
                }
                // 3 is a FreeBSD kludge.
                0x01 | 0x03 => self.v2x6 = 1,
                // Prince of Persia, csp.sys, diagnose.exe.
                0xc6 => self.v2x6 = 0,
                // Panic.
                0xb8 => self.reset(),
                0x39 => {
                    self.dsp_out_data(0x38);
                    self.reset();
                    self.v2x6 = 0x39;
                }
                _ => self.v2x6 = val as i32,
            },
            0x0c => {
                if self.needed_bytes == 0 {
                    self.command(val as u8);
                } else if self.in_index as usize == self.in2_data.len() {
                    warn_report("sb16: in data overrun");
                } else {
                    let i = self.in_index as usize;
                    self.in2_data[i] = val as u8;
                    self.in_index += 1;
                    if self.in_index == self.needed_bytes {
                        self.needed_bytes = 0;
                        self.complete();
                    }
                }
            }
            _ => {}
        }
    }

    /// `dsp_read()`.
    fn dsp_read(&mut self, nport: u32, iport: u32) -> u32 {
        match iport {
            // Reset.
            0x06 => 0xff,
            // Read data.
            0x0a => {
                if self.out_data_len != 0 {
                    self.out_data_len -= 1;
                    let v = self.out_data[self.out_data_len as usize];
                    self.last_read_byte = v;
                    u32::from(v)
                } else {
                    if self.cmd != -1 {
                        warn_report(&format!(
                            "sb16: empty output buffer for command 0x{:x}",
                            self.cmd
                        ));
                    }
                    u32::from(self.last_read_byte)
                }
            }
            // Zero when the DSP can take a write.
            0x0c => {
                if self.can_write != 0 {
                    0
                } else {
                    0x80
                }
            }
            // Timer interrupt clear.
            0x0d => 0,
            // Data available, and the ack of an 8-bit interrupt.
            0x0e => {
                let v = if self.out_data_len == 0 || self.highspeed != 0 { 0 } else { 0x80 };
                if self.mixer_regs[0x82] & 1 != 0 {
                    self.mixer_regs[0x82] &= !1;
                    self.d.pic.lower();
                }
                v
            }
            // The ack of a 16-bit interrupt.
            0x0f => {
                if self.mixer_regs[0x82] & 2 != 0 {
                    self.mixer_regs[0x82] &= !2;
                    self.d.pic.lower();
                }
                0xff
            }
            _ => {
                warn_report(&format!("sb16: dsp_read 0x{nport:x} error"));
                0xff
            }
        }
    }

    /// `mixer_write_datab()`.
    fn mixer_write_datab(&mut self, val: u32) {
        // A write to 0x80 changes QEMU's `irq`, which nothing reads after realize, so the
        // interrupt stays where it was and only the register changes.
        match self.mixer_nreg {
            0x00 => self.reset_mixer(),
            0x82 => return,
            _ => {}
        }
        let n = usize::from(self.mixer_nreg);
        self.mixer_regs[n] = val as u8;
    }

    /// `write_audio()`: moves up to `len` bytes from the transfer at `dma_pos` to the voice.
    fn write_audio(&mut self, nchan: u32, mut dma_pos: i32, dma_len: i32, len: i32) -> i32 {
        let d = self.d;
        let isa_dma = if nchan == d.cfg.dma { &d.isa_dma } else { &d.isa_hdma };
        let mut tmpbuf = [0u8; 4096];
        let mut temp = len;
        let mut net = 0;
        while temp != 0 {
            let left = dma_len - dma_pos;
            let to_copy = (temp.min(left).max(0) as usize).min(tmpbuf.len());
            let copied = isa_dma.read_memory(nchan, &mut tmpbuf[..to_copy], dma_pos);
            let copied = d.be.write(self.voice, &tmpbuf[..copied]) as i32;
            temp -= copied;
            dma_pos = (dma_pos + copied) % dma_len;
            net += copied;
            if copied == 0 {
                break;
            }
        }
        net
    }

    /// `SB_read_DMA()`, the transfer handler of both channels.
    fn read_dma(&mut self, nchan: u32, mut dma_pos: i32, dma_len: i32) -> i32 {
        if self.block_size <= 0 {
            return dma_pos;
        }
        if self.left_till_irq < 0 {
            self.left_till_irq = self.block_size;
        }

        let free = if self.voice.is_some() {
            let free = self.d.audio_free.load(Ordering::SeqCst) & !self.align;
            if free <= 0 || dma_len == 0 {
                return dma_pos;
            }
            free
        } else {
            dma_len
        };

        let mut copy = free;
        let till = self.left_till_irq;
        if till <= copy && self.dma_auto == 0 {
            copy = till;
        }

        let written = self.write_audio(nchan, dma_pos, dma_len, copy);
        dma_pos = (dma_pos + written) % dma_len;
        self.left_till_irq -= written;

        if self.left_till_irq <= 0 {
            self.mixer_regs[0x82] |= if nchan & 4 != 0 { 2 } else { 1 };
            self.d.pic.raise();
            if self.dma_auto == 0 {
                self.control(false);
                self.speaker(0);
            }
        }

        while self.left_till_irq <= 0 {
            self.left_till_irq += self.block_size;
        }
        dma_pos
    }
}

fn mixer_write_indexb(d: &Inner, _nport: u32, val: u32) {
    d.cx().mixer_nreg = val as u8;
}

fn mixer_write_datab(d: &Inner, _nport: u32, val: u32) {
    d.cx().mixer_write_datab(val);
}

fn mixer_read(d: &Inner, _nport: u32) -> u32 {
    let s = d.cx();
    u32::from(s.mixer_regs[usize::from(s.mixer_nreg)])
}

fn dsp_write(d: &Inner, nport: u32, val: u32) {
    d.cx().dsp_write(nport.wrapping_sub(d.cfg.port), val);
}

fn dsp_read(d: &Inner, nport: u32) -> u32 {
    d.cx().dsp_read(nport, nport.wrapping_sub(d.cfg.port))
}

/// `sb16_ioport_list`.
const SB16_IOPORT_LIST: [Portio<Inner>; 6] = [
    Portio { offset: 4, len: 1, size: 1, read: None, write: Some(mixer_write_indexb) },
    Portio { offset: 5, len: 1, size: 1, read: Some(mixer_read), write: Some(mixer_write_datab) },
    Portio { offset: 6, len: 1, size: 1, read: Some(dsp_read), write: Some(dsp_write) },
    Portio { offset: 10, len: 1, size: 1, read: Some(dsp_read), write: None },
    Portio { offset: 12, len: 1, size: 1, read: None, write: Some(dsp_write) },
    Portio { offset: 12, len: 4, size: 1, read: Some(dsp_read), write: None },
];

/// A Sound Blaster 16 on the ISA bus.
pub struct Sb16 {
    inner: Arc<Inner>,
}

impl fmt::Debug for Sb16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sb16").field("cfg", &self.inner.cfg).finish_non_exhaustive()
    }
}

impl Sb16 {
    /// `sb16_realizefn()` after the backend check: takes the two DMA channels from `dma`,
    /// which is `None` on a bus without DMA, and the interrupt from the ISA `irqs`, then
    /// hooks the transfer handler to both channels. The ports come from
    /// [`Sb16::io_regions`].
    pub fn realize(
        be: Arc<AudioBackend>,
        cfg: Sb16Config,
        dma: Option<&IsaDma>,
        irqs: &[IrqLine],
        clock: &Arc<Clock>,
    ) -> Result<Sb16, String> {
        let Some(dma) = dma else {
            return Err("ISA controller does not support DMA".into());
        };
        // QEMU asserts here.
        let Some(pic) = irqs.get(cfg.irq as usize).filter(|_| cfg.irq < 16) else {
            return Err(format!("ISA interrupt {} does not exist", cfg.irq));
        };
        let isa_hdma = Arc::clone(dma.get(cfg.hdma));
        let isa_dma = Arc::clone(dma.get(cfg.dma));

        let mut s = State::new();
        s.mixer_regs[0x80] = magic_of_irq(cfg.irq);
        s.mixer_regs[0x81] = (1u32.wrapping_shl(cfg.dma) | 1u32.wrapping_shl(cfg.hdma)) as u8;
        s.mixer_regs[0x82] = 2 << 5;
        s.csp_regs[5] = 1;
        s.csp_regs[9] = 0xf8;
        s.reset_mixer();
        s.can_write = 1;

        let inner = Arc::new_cyclic(|me: &Weak<Inner>| {
            let w = me.clone();
            let aux_ts = clock.new_timer(move || {
                if let Some(d) = w.upgrade() {
                    d.aux_timer();
                }
            });
            let w = me.clone();
            // SB_audio_callback().
            let callback: AudioCallback = Arc::new(move |free: usize| {
                if let Some(d) = w.upgrade() {
                    d.audio_free.store(free.min(i32::MAX as usize) as i32, Ordering::SeqCst);
                }
            });
            Inner {
                be,
                cfg,
                isa_dma,
                isa_hdma,
                pic: pic.clone(),
                aux_ts,
                audio_free: AtomicI32::new(0),
                callback,
                s: Mutex::new(s),
            }
        });

        for (iso, nchan) in [(&inner.isa_hdma, cfg.hdma), (&inner.isa_dma, cfg.dma)] {
            let w = Arc::downgrade(&inner);
            iso.register_channel(
                nchan,
                Arc::new(move |nchan: u32, pos: i32, len: i32| match w.upgrade() {
                    Some(d) => d.cx().read_dma(nchan, pos, len),
                    None => pos,
                }),
            );
        }
        Ok(Sb16 { inner })
    }

    /// The card's ports, `isa_register_portio_list()` at `iobase`: each region with the port
    /// it starts at and its size.
    pub fn io_regions(&self) -> Vec<(u32, u64, Arc<dyn MmioOps>)> {
        portio_list(&self.inner, &SB16_IOPORT_LIST, self.inner.cfg.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mixer_resets_like_qemus() {
        let mut s = State::new();
        s.reset_mixer();
        assert_eq!(s.mixer_regs[0x00], 0xff);
        assert_eq!(s.mixer_regs[0x04], 0x88);
        assert_eq!(s.mixer_regs[0x0e], 0);
        assert_eq!(s.mixer_regs[0x30], 0x20);
        assert_eq!(s.mixer_regs[0x47], 0x20);
        assert_eq!(s.mixer_regs[0x48], 0xff);
        assert_eq!(s.mixer_regs[0x7f], 0);
        assert_eq!(s.mixer_regs[0x82], 0);
        assert_eq!(s.mixer_regs[0x83], 0xff);
    }

    #[test]
    fn command_bytes_come_out_last_first() {
        let mut s = State::new();
        s.in2_data[..2].copy_from_slice(&[0x34, 0x12]);
        s.in_index = 2;
        assert_eq!(s.dsp_get_lohi(), 0x1234);
        s.in_index = 2;
        assert_eq!(s.dsp_get_hilo(), 0x3412);
    }
}
