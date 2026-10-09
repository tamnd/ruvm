// SPDX-License-Identifier: GPL-2.0-or-later

//! The Intel 82801AA AC97 controller with a SigmaTel STAC9700 codec, QEMU's `hw/audio/ac97.c`.
//!
//! BAR 0 is the native audio mixer, the codec registers. BAR 1 is the native audio bus master:
//! three DMA engines (PCM in, PCM out and the microphone), each walking a list of 32 buffer
//! descriptors in guest memory. The backend asks for data from its timer, and the matching
//! engine copies as much as it can between the descriptors and the voice.
//!
//! Migration of the device state is not ported yet.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_audio::{AudSettings, AudioBackend, SwVoiceIn, SwVoiceOut, Volume};
use ruvm_base::{Error, Result, error_report};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_pci::regs::{
    PCI_BASE_ADDRESS_SPACE_IO, PCI_CLASS_PROG, PCI_INTERRUPT_LINE, PCI_INTERRUPT_PIN, PCI_STATUS,
    PCI_STATUS_FAST_BACK,
};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};
use ruvm_qapi::types::AudioFormat;

use crate::{PCI_CLASS_MULTIMEDIA_AUDIO, PCI_STATUS_DEVSEL_MEDIUM, PCI_VENDOR_ID_INTEL};

/// `TYPE_AC97`.
pub const TYPE_AC97: &str = "AC97";

const PCI_DEVICE_ID_INTEL_82801AA_5: u16 = 0x2415;

const SR_FIFOE: u16 = 16;
const SR_BCIS: u16 = 8;
const SR_LVBCI: u16 = 4;
const SR_CELV: u16 = 2;
const SR_DCH: u16 = 1;
const SR_WCLEAR_MASK: u16 = SR_FIFOE | SR_BCIS | SR_LVBCI;
const SR_RO_MASK: u16 = SR_DCH | SR_CELV;
const SR_INT_MASK: u16 = SR_FIFOE | SR_BCIS | SR_LVBCI;

const CR_IOCE: u8 = 16;
const CR_FEIE: u8 = 8;
const CR_LVBIE: u8 = 4;
const CR_RR: u8 = 2;
const CR_RPBM: u8 = 1;
const CR_VALID_MASK: u8 = (1 << 5) - 1;
const CR_DONT_CLEAR_MASK: u8 = CR_IOCE | CR_FEIE | CR_LVBIE;

const GC_WR: u32 = 4;
const GC_CR: u32 = 2;
const GC_VALID_MASK: u32 = (1 << 6) - 1;

const GS_MD3: u32 = 1 << 17;
const GS_AD3: u32 = 1 << 16;
const GS_RCS: u32 = 1 << 15;
const GS_B3S12: u32 = 1 << 14;
const GS_B2S12: u32 = 1 << 13;
const GS_B1S12: u32 = 1 << 12;
const GS_S1R1: u32 = 1 << 11;
const GS_S0R1: u32 = 1 << 10;
const GS_S1CR: u32 = 1 << 9;
const GS_S0CR: u32 = 1 << 8;
const GS_MINT: u32 = 1 << 7;
const GS_POINT: u32 = 1 << 6;
const GS_PIINT: u32 = 1 << 5;
const GS_RSRVD: u32 = (1 << 4) | (1 << 3);
const GS_MOINT: u32 = 1 << 2;
const GS_MIINT: u32 = 1 << 1;
const GS_GSCI: u32 = 1;
const GS_RO_MASK: u32 = GS_B3S12
    | GS_B2S12
    | GS_B1S12
    | GS_S1CR
    | GS_S0CR
    | GS_MINT
    | GS_POINT
    | GS_PIINT
    | GS_RSRVD
    | GS_MOINT
    | GS_MIINT;
const GS_VALID_MASK: u32 = (1 << 18) - 1;
const GS_WCLEAR_MASK: u32 = GS_RCS | GS_S1R1 | GS_S0R1 | GS_GSCI;

// Not used by the model, but part of the register layout.
const _: u32 = GS_MD3 | GS_AD3;

const BD_IOC: u32 = 1 << 31;
const BD_BUP: u32 = 1 << 30;

const REC_MASK: u32 = 7;

const BUP_SET: u32 = 1;
const BUP_LAST: u32 = 2;

const PI_INDEX: usize = 0;
const PO_INDEX: usize = 1;
const MC_INDEX: usize = 2;
const LAST_INDEX: usize = 3;

// The bus master registers of each engine, at `index * 16`.
const BM_BDBAR: u32 = 0;
const BM_CIV: u32 = 4;
const BM_LVI: u32 = 5;
const BM_SR: u32 = 6;
const BM_PICB: u32 = 8;
const BM_PIV: u32 = 10;
const BM_CR: u32 = 11;

const GLOB_CNT: u32 = 0x2c;
const GLOB_STA: u32 = 0x30;
const CAS: u32 = 0x34;

// The codec registers, `ac97.h`.
const AC97_RESET: u32 = 0x00;
const AC97_MASTER_VOLUME_MUTE: u32 = 0x02;
const AC97_HEADPHONE_VOLUME_MUTE: u32 = 0x04;
const AC97_MASTER_VOLUME_MONO_MUTE: u32 = 0x06;
const AC97_MASTER_TONE_RL: u32 = 0x08;
const AC97_PC_BEEP_VOLUME_MUTE: u32 = 0x0a;
const AC97_PHONE_VOLUME_MUTE: u32 = 0x0c;
const AC97_MIC_VOLUME_MUTE: u32 = 0x0e;
const AC97_LINE_IN_VOLUME_MUTE: u32 = 0x10;
const AC97_CD_VOLUME_MUTE: u32 = 0x12;
const AC97_VIDEO_VOLUME_MUTE: u32 = 0x14;
const AC97_AUX_VOLUME_MUTE: u32 = 0x16;
const AC97_PCM_OUT_VOLUME_MUTE: u32 = 0x18;
const AC97_RECORD_SELECT: u32 = 0x1a;
const AC97_RECORD_GAIN_MUTE: u32 = 0x1c;
const AC97_RECORD_GAIN_MIC_MUTE: u32 = 0x1e;
const AC97_GENERAL_PURPOSE: u32 = 0x20;
const AC97_3D_CONTROL: u32 = 0x22;
const AC97_POWERDOWN_CTRL_STAT: u32 = 0x26;
const AC97_EXTENDED_AUDIO_ID: u32 = 0x28;
const AC97_EXTENDED_AUDIO_CTRL_STAT: u32 = 0x2a;
const AC97_PCM_FRONT_DAC_RATE: u32 = 0x2c;
const AC97_PCM_SURROUND_DAC_RATE: u32 = 0x2e;
const AC97_PCM_LFE_DAC_RATE: u32 = 0x30;
const AC97_PCM_LR_ADC_RATE: u32 = 0x32;
const AC97_MIC_ADC_RATE: u32 = 0x34;
const AC97_SIGMATEL_ANALOG: u32 = 0x6c;
const AC97_SIGMATEL_DAC2INVERT: u32 = 0x6e;
const AC97_VENDOR_ID1: u32 = 0x7c;
const AC97_VENDOR_ID2: u32 = 0x7e;

const EACS_VRA: u32 = 1;
const EACS_VRM: u32 = 8;
const MUTE_SHIFT: u32 = 15;

/// `GET_BM()`.
fn get_bm(addr: u32) -> usize {
    ((addr >> 4) & 3) as usize
}

/// One buffer descriptor.
#[derive(Clone, Copy, Debug, Default)]
struct Bd {
    addr: u32,
    ctl_len: u32,
}

/// `AC97BusMasterRegs`.
#[derive(Clone, Copy, Debug, Default)]
struct BmRegs {
    bdbar: u32,
    civ: u8,
    lvi: u8,
    sr: u16,
    picb: u16,
    piv: u8,
    cr: u8,
    bd_valid: bool,
    bd: Bd,
}

/// The mutable part of `AC97LinkState`.
struct State {
    glob_cnt: u32,
    glob_sta: u32,
    cas: u32,
    last_samp: u32,
    bm_regs: [BmRegs; LAST_INDEX],
    mixer_data: [u8; 256],
    voice_pi: Option<SwVoiceIn>,
    voice_po: Option<SwVoiceOut>,
    voice_mc: Option<SwVoiceIn>,
    invalid_freq: [i32; LAST_INDEX],
    silence: [u8; 128],
    bup_flag: u32,
}

struct Inner {
    me: Weak<Inner>,
    dev: Arc<PciDevice>,
    be: Arc<AudioBackend>,
    dma: Arc<dyn DmaMemory>,
    s: Mutex<State>,
}

/// An `AC97` PCI function.
pub struct Ac97 {
    inner: Arc<Inner>,
}

impl fmt::Debug for Ac97 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ac97").field("dev", &self.inner.dev.name()).finish_non_exhaustive()
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `pci_dma_read()`: nothing reaches memory while bus mastering is off.
    fn dma_read(&self, addr: u32, buf: &mut [u8]) {
        if !self.dev.is_bus_master() || !self.dma.read(u64::from(addr), buf) {
            buf.fill(0);
        }
    }

    /// `pci_dma_write()`.
    fn dma_write(&self, addr: u32, buf: &[u8]) {
        if self.dev.is_bus_master() {
            self.dma.write(u64::from(addr), buf);
        }
    }

    fn fetch_bd(&self, r: &mut BmRegs) {
        let mut b = [0u8; 8];
        self.dma_read(r.bdbar.wrapping_add(u32::from(r.civ) * 8), &mut b);
        r.bd_valid = true;
        r.bd.addr = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & !3;
        r.bd.ctl_len = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        r.picb = (r.bd.ctl_len & 0xffff) as u16;
    }

    fn update_sr(&self, s: &mut State, index: usize, new_sr: u16) {
        let masks = [GS_PIINT, GS_POINT, GS_MINT];
        let r = &mut s.bm_regs[index];
        let new_mask = new_sr & SR_INT_MASK;
        let old_mask = r.sr & SR_INT_MASK;
        let mut event = false;
        let mut level = false;
        if new_mask ^ old_mask != 0 {
            if new_mask == 0 {
                event = true;
            } else {
                if new_mask & SR_LVBCI != 0 && r.cr & CR_LVBIE != 0 {
                    event = true;
                    level = true;
                }
                if new_mask & SR_BCIS != 0 && r.cr & CR_IOCE != 0 {
                    event = true;
                    level = true;
                }
            }
        }
        r.sr = new_sr;
        if !event {
            return;
        }
        if level {
            s.glob_sta |= masks[index];
            self.dev.set_irq(1);
        } else {
            s.glob_sta &= !masks[index];
            self.dev.set_irq(0);
        }
    }

    fn voice_set_active(&self, s: &State, index: usize, on: bool) {
        match index {
            PI_INDEX => self.be.set_active_in(s.voice_pi, on),
            PO_INDEX => self.be.set_active_out(s.voice_po, on),
            _ => self.be.set_active_in(s.voice_mc, on),
        }
    }

    fn reset_bm_regs(&self, s: &mut State, index: usize) {
        {
            let r = &mut s.bm_regs[index];
            r.bdbar = 0;
            r.civ = 0;
            r.lvi = 0;
        }
        self.update_sr(s, index, SR_DCH);
        let r = &mut s.bm_regs[index];
        r.picb = 0;
        r.piv = 0;
        r.cr &= CR_DONT_CLEAR_MASK;
        r.bd_valid = false;
        self.voice_set_active(s, index, false);
        s.silence = [0; 128];
    }

    fn open_voice(&self, s: &mut State, index: usize, freq: i32) {
        let as_ = AudSettings { freq, nchannels: 2, fmt: AudioFormat::S16, big_endian: false };
        if freq > 0 {
            s.invalid_freq[index] = 0;
            let me = self.me.clone();
            let cb = Arc::new(move |avail: usize| {
                if let Some(me) = me.upgrade() {
                    me.transfer_audio(index, i32::try_from(avail).unwrap_or(i32::MAX));
                }
            });
            match index {
                PI_INDEX => s.voice_pi = self.be.open_in(s.voice_pi, "ac97.pi", cb, &as_),
                PO_INDEX => s.voice_po = self.be.open_out(s.voice_po, "ac97.po", cb, &as_),
                _ => s.voice_mc = self.be.open_in(s.voice_mc, "ac97.mc", cb, &as_),
            }
        } else {
            s.invalid_freq[index] = freq;
            match index {
                PI_INDEX => self.be.close_in(s.voice_pi.take()),
                PO_INDEX => self.be.close_out(s.voice_po.take()),
                _ => self.be.close_in(s.voice_mc.take()),
            }
        }
    }

    fn reset_voices(&self, s: &mut State, active: [bool; LAST_INDEX]) {
        let freq = mixer_load(s, AC97_PCM_LR_ADC_RATE);
        self.open_voice(s, PI_INDEX, i32::from(freq));
        self.be.set_active_in(s.voice_pi, active[PI_INDEX]);

        let freq = mixer_load(s, AC97_PCM_FRONT_DAC_RATE);
        self.open_voice(s, PO_INDEX, i32::from(freq));
        self.be.set_active_out(s.voice_po, active[PO_INDEX]);

        let freq = mixer_load(s, AC97_MIC_ADC_RATE);
        self.open_voice(s, MC_INDEX, i32::from(freq));
        self.be.set_active_in(s.voice_mc, active[MC_INDEX]);
    }

    fn update_combined_volume_out(&self, s: &State) {
        let (mute, lvol, rvol) = get_volume(mixer_load(s, AC97_MASTER_VOLUME_MUTE), 0x3f, true);
        let (pmute, plvol, prvol) = get_volume(mixer_load(s, AC97_PCM_OUT_VOLUME_MUTE), 0x1f, true);
        let lvol = (u32::from(lvol) * u32::from(plvol) / 255) as u8;
        let rvol = (u32::from(rvol) * u32::from(prvol) / 255) as u8;
        self.be.set_volume_out(s.voice_po, &volume_lr(mute | pmute, lvol, rvol));
    }

    fn update_volume_in(&self, s: &State) {
        let (mute, lvol, rvol) = get_volume(mixer_load(s, AC97_RECORD_GAIN_MUTE), 0x0f, false);
        self.be.set_volume_in(s.voice_pi, &volume_lr(mute, lvol, rvol));
    }

    fn set_volume(&self, s: &mut State, index: u32, val: u32) {
        match index {
            AC97_MASTER_VOLUME_MUTE => {
                mixer_store(s, index, (val & 0xbf3f) as u16);
                self.update_combined_volume_out(s);
            }
            AC97_PCM_OUT_VOLUME_MUTE => {
                mixer_store(s, index, (val & 0x9f1f) as u16);
                self.update_combined_volume_out(s);
            }
            AC97_RECORD_GAIN_MUTE => {
                mixer_store(s, index, (val & 0x8f0f) as u16);
                self.update_volume_in(s);
            }
            _ => {}
        }
    }

    fn mixer_reset(&self, s: &mut State) {
        s.mixer_data = [0; 256];
        for reg in [
            AC97_RESET,
            AC97_HEADPHONE_VOLUME_MUTE,
            AC97_MASTER_VOLUME_MONO_MUTE,
            AC97_MASTER_TONE_RL,
            AC97_PC_BEEP_VOLUME_MUTE,
            AC97_PHONE_VOLUME_MUTE,
            AC97_MIC_VOLUME_MUTE,
            AC97_LINE_IN_VOLUME_MUTE,
            AC97_CD_VOLUME_MUTE,
            AC97_VIDEO_VOLUME_MUTE,
            AC97_AUX_VOLUME_MUTE,
            AC97_RECORD_GAIN_MIC_MUTE,
            AC97_GENERAL_PURPOSE,
            AC97_3D_CONTROL,
        ] {
            mixer_store(s, reg, 0);
        }
        mixer_store(s, AC97_POWERDOWN_CTRL_STAT, 0x000f);

        // SigmaTel 9700 (STAC9700).
        mixer_store(s, AC97_VENDOR_ID1, 0x8384);
        mixer_store(s, AC97_VENDOR_ID2, 0x7600);

        mixer_store(s, AC97_EXTENDED_AUDIO_ID, 0x0809);
        mixer_store(s, AC97_EXTENDED_AUDIO_CTRL_STAT, 0x0009);
        mixer_store(s, AC97_PCM_FRONT_DAC_RATE, 0xbb80);
        mixer_store(s, AC97_PCM_SURROUND_DAC_RATE, 0xbb80);
        mixer_store(s, AC97_PCM_LFE_DAC_RATE, 0xbb80);
        mixer_store(s, AC97_PCM_LR_ADC_RATE, 0xbb80);
        mixer_store(s, AC97_MIC_ADC_RATE, 0xbb80);

        record_select(s, 0);
        self.set_volume(s, AC97_MASTER_VOLUME_MUTE, 0x8000);
        self.set_volume(s, AC97_PCM_OUT_VOLUME_MUTE, 0x8808);
        self.set_volume(s, AC97_RECORD_GAIN_MUTE, 0x8808);

        self.reset_voices(s, [false; LAST_INDEX]);
    }

    fn nam_writew(&self, s: &mut State, addr: u32, mut val: u32) {
        s.cas = 0;
        match addr {
            AC97_RESET => self.mixer_reset(s),
            AC97_POWERDOWN_CTRL_STAT => {
                val &= !0x800f;
                val |= u32::from(mixer_load(s, addr)) & 0xf;
                mixer_store(s, addr, val as u16);
            }
            AC97_PCM_OUT_VOLUME_MUTE | AC97_MASTER_VOLUME_MUTE | AC97_RECORD_GAIN_MUTE => {
                self.set_volume(s, addr, val);
            }
            AC97_RECORD_SELECT => record_select(s, val),
            AC97_VENDOR_ID1 | AC97_VENDOR_ID2 | AC97_EXTENDED_AUDIO_ID => {}
            AC97_EXTENDED_AUDIO_CTRL_STAT => {
                if val & EACS_VRA == 0 {
                    mixer_store(s, AC97_PCM_FRONT_DAC_RATE, 0xbb80);
                    mixer_store(s, AC97_PCM_LR_ADC_RATE, 0xbb80);
                    self.open_voice(s, PI_INDEX, 48000);
                    self.open_voice(s, PO_INDEX, 48000);
                }
                if val & EACS_VRM == 0 {
                    mixer_store(s, AC97_MIC_ADC_RATE, 0xbb80);
                    self.open_voice(s, MC_INDEX, 48000);
                }
                mixer_store(s, AC97_EXTENDED_AUDIO_CTRL_STAT, val as u16);
            }
            AC97_PCM_FRONT_DAC_RATE => {
                if u32::from(mixer_load(s, AC97_EXTENDED_AUDIO_CTRL_STAT)) & EACS_VRA != 0 {
                    mixer_store(s, addr, val as u16);
                    self.open_voice(s, PO_INDEX, val as i32);
                }
            }
            AC97_MIC_ADC_RATE => {
                if u32::from(mixer_load(s, AC97_EXTENDED_AUDIO_CTRL_STAT)) & EACS_VRM != 0 {
                    mixer_store(s, addr, val as u16);
                    self.open_voice(s, MC_INDEX, val as i32);
                }
            }
            AC97_PCM_LR_ADC_RATE => {
                if u32::from(mixer_load(s, AC97_EXTENDED_AUDIO_CTRL_STAT)) & EACS_VRA != 0 {
                    mixer_store(s, addr, val as u16);
                    self.open_voice(s, PI_INDEX, val as i32);
                }
            }
            // None of the features in these registers are emulated, so they are read-only.
            AC97_HEADPHONE_VOLUME_MUTE
            | AC97_MASTER_VOLUME_MONO_MUTE
            | AC97_MASTER_TONE_RL
            | AC97_PC_BEEP_VOLUME_MUTE
            | AC97_PHONE_VOLUME_MUTE
            | AC97_MIC_VOLUME_MUTE
            | AC97_LINE_IN_VOLUME_MUTE
            | AC97_CD_VOLUME_MUTE
            | AC97_VIDEO_VOLUME_MUTE
            | AC97_AUX_VOLUME_MUTE
            | AC97_RECORD_GAIN_MIC_MUTE
            | AC97_GENERAL_PURPOSE
            | AC97_3D_CONTROL
            | AC97_SIGMATEL_ANALOG
            | AC97_SIGMATEL_DAC2INVERT => {}
            _ => mixer_store(s, addr, val as u16),
        }
    }

    fn nabm_readb(&self, s: &mut State, addr: u32) -> u32 {
        if addr == CAS {
            let val = s.cas;
            s.cas = 1;
            return val;
        }
        if addr >= 0x30 {
            return !0;
        }
        let r = &s.bm_regs[get_bm(addr)];
        match addr & 15 {
            BM_CIV => u32::from(r.civ),
            BM_LVI => u32::from(r.lvi),
            BM_PIV => u32::from(r.piv),
            BM_CR => u32::from(r.cr),
            BM_SR => u32::from(r.sr & 0xff),
            _ => !0,
        }
    }

    fn nabm_readw(&self, s: &State, addr: u32) -> u32 {
        if addr >= 0x30 {
            return !0;
        }
        let r = &s.bm_regs[get_bm(addr)];
        match addr & 15 {
            BM_SR => u32::from(r.sr),
            BM_PICB => u32::from(r.picb),
            _ => !0,
        }
    }

    fn nabm_readl(&self, s: &State, addr: u32) -> u32 {
        match addr {
            GLOB_CNT => s.glob_cnt,
            GLOB_STA => s.glob_sta | GS_S0CR,
            _ if addr >= 0x30 => !0,
            _ => {
                let r = &s.bm_regs[get_bm(addr)];
                match addr & 15 {
                    BM_BDBAR => r.bdbar,
                    BM_CIV => u32::from(r.civ) | (u32::from(r.lvi) << 8) | (u32::from(r.sr) << 16),
                    BM_PICB => {
                        u32::from(r.picb) | (u32::from(r.piv) << 16) | (u32::from(r.cr) << 24)
                    }
                    _ => !0,
                }
            }
        }
    }

    fn write_sr(&self, s: &mut State, index: usize, val: u32) {
        let r = &mut s.bm_regs[index];
        r.sr |= (val as u16) & !(SR_RO_MASK | SR_WCLEAR_MASK);
        let new_sr = r.sr & !((val as u16) & SR_WCLEAR_MASK);
        self.update_sr(s, index, new_sr);
    }

    fn nabm_writeb(&self, s: &mut State, addr: u32, val: u32) {
        if addr >= 0x30 {
            return;
        }
        let index = get_bm(addr);
        match addr & 15 {
            BM_LVI => {
                let r = &mut s.bm_regs[index];
                if r.cr & CR_RPBM != 0 && r.sr & SR_DCH != 0 {
                    r.sr &= !(SR_DCH | SR_CELV);
                    r.civ = r.piv;
                    r.piv = (r.piv + 1) % 32;
                    self.fetch_bd(r);
                }
                r.lvi = (val % 32) as u8;
            }
            BM_CR => {
                if val as u8 & CR_RR != 0 {
                    self.reset_bm_regs(s, index);
                } else {
                    let r = &mut s.bm_regs[index];
                    r.cr = val as u8 & CR_VALID_MASK;
                    if r.cr & CR_RPBM == 0 {
                        self.voice_set_active(s, index, false);
                        s.bm_regs[index].sr |= SR_DCH;
                    } else {
                        r.civ = r.piv;
                        r.piv = (r.piv + 1) % 32;
                        self.fetch_bd(r);
                        r.sr &= !SR_DCH;
                        self.voice_set_active(s, index, true);
                    }
                }
            }
            BM_SR => self.write_sr(s, index, val),
            _ => {}
        }
    }

    fn nabm_writew(&self, s: &mut State, addr: u32, val: u32) {
        if addr < 0x30 && addr & 15 == BM_SR {
            self.write_sr(s, get_bm(addr), val);
        }
    }

    fn nabm_writel(&self, s: &mut State, addr: u32, val: u32) {
        match addr {
            GLOB_CNT => {
                // Warm and cold reset requests are not handled.
                if val & (GC_WR | GC_CR) == 0 {
                    s.glob_cnt = val & GC_VALID_MASK;
                }
            }
            GLOB_STA => {
                s.glob_sta &= !(val & GS_WCLEAR_MASK);
                s.glob_sta |= (val & !(GS_WCLEAR_MASK | GS_RO_MASK)) & GS_VALID_MASK;
            }
            _ if addr < 0x30 && addr & 15 == BM_BDBAR => {
                s.bm_regs[get_bm(addr)].bdbar = val & !3;
            }
            _ => {}
        }
    }

    fn write_audio(&self, s: &mut State, max: i32, stop: &mut bool) -> i32 {
        let mut tmpbuf = [0u8; 4096];
        let r = &s.bm_regs[PO_INDEX];
        let mut addr = r.bd.addr;
        let mut temp = (u32::from(r.picb) << 1).min(max as u32);
        let mut written = 0u32;
        let mut to_copy = 0usize;
        if temp == 0 {
            *stop = true;
            return 0;
        }
        while temp != 0 {
            to_copy = (temp as usize).min(tmpbuf.len());
            self.dma_read(addr, &mut tmpbuf[..to_copy]);
            let copied = self.be.write(s.voice_po, &tmpbuf[..to_copy]) as u32;
            if copied == 0 {
                *stop = true;
                break;
            }
            temp -= copied;
            addr = addr.wrapping_add(copied);
            written += copied;
        }
        if temp == 0 {
            s.last_samp = if to_copy < 4 {
                0
            } else {
                let b = &tmpbuf[to_copy - 4..to_copy];
                u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
            };
        }
        s.bm_regs[PO_INDEX].bd.addr = addr;
        written as i32
    }

    fn write_bup(&self, s: &mut State, mut elapsed: i32) {
        if s.bup_flag & BUP_SET == 0 {
            if s.bup_flag & BUP_LAST != 0 {
                let samp = s.last_samp.to_ne_bytes();
                for c in s.silence.chunks_exact_mut(4) {
                    c.copy_from_slice(&samp);
                }
            } else {
                s.silence = [0; 128];
            }
            s.bup_flag |= BUP_SET;
        }
        while elapsed != 0 {
            let mut temp = elapsed.min(s.silence.len() as i32);
            while temp != 0 {
                let copied = self.be.write(s.voice_po, &s.silence[..temp as usize]) as i32;
                if copied == 0 {
                    return;
                }
                temp -= copied;
                elapsed -= copied;
            }
        }
    }

    fn read_audio(&self, s: &mut State, index: usize, max: i32, stop: &mut bool) -> i32 {
        let mut tmpbuf = [0u8; 4096];
        let r = &s.bm_regs[index];
        let mut addr = r.bd.addr;
        let mut temp = (u32::from(r.picb) << 1).min(max as u32);
        let mut nread = 0u32;
        let voice = if index == MC_INDEX { s.voice_mc } else { s.voice_pi };
        if temp == 0 {
            *stop = true;
            return 0;
        }
        while temp != 0 {
            let to_copy = (temp as usize).min(tmpbuf.len());
            let acquired = self.be.read(voice, &mut tmpbuf[..to_copy]);
            if acquired == 0 {
                *stop = true;
                break;
            }
            self.dma_write(addr, &tmpbuf[..acquired]);
            let acquired = acquired as u32;
            temp -= acquired;
            addr = addr.wrapping_add(acquired);
            nread += acquired;
        }
        s.bm_regs[index].bd.addr = addr;
        nread as i32
    }

    /// The voice callback: moves up to `elapsed` bytes between the descriptors and the voice.
    fn transfer_audio(&self, index: usize, mut elapsed: i32) {
        let mut guard = self.lock();
        let s = &mut *guard;
        let mut stop = false;

        if s.invalid_freq[index] != 0 {
            error_report(&format!(
                "ac97: attempt to use voice {index} with invalid frequency {}",
                s.invalid_freq[index]
            ));
            return;
        }

        if s.bm_regs[index].sr & SR_DCH != 0 {
            if s.bm_regs[index].cr & CR_RPBM != 0 && index == PO_INDEX {
                self.write_bup(s, elapsed);
            }
            return;
        }

        while (elapsed >> 1) != 0 && !stop {
            {
                let r = &mut s.bm_regs[index];
                if !r.bd_valid {
                    self.fetch_bd(r);
                }
                if r.picb == 0 {
                    if r.civ == r.lvi {
                        r.sr |= SR_DCH;
                        s.bup_flag = 0;
                        break;
                    }
                    r.sr &= !SR_CELV;
                    r.civ = r.piv;
                    r.piv = (r.piv + 1) % 32;
                    self.fetch_bd(r);
                    return;
                }
            }

            let temp = if index == PO_INDEX {
                self.write_audio(s, elapsed, &mut stop)
            } else {
                self.read_audio(s, index, elapsed, &mut stop)
            };
            elapsed -= temp;
            let r = &mut s.bm_regs[index];
            r.picb = r.picb.wrapping_sub((temp >> 1) as u16);

            if r.picb == 0 {
                let mut new_sr = r.sr & !SR_CELV;
                if r.bd.ctl_len & BD_IOC != 0 {
                    new_sr |= SR_BCIS;
                }
                if r.civ == r.lvi {
                    new_sr |= SR_LVBCI | SR_DCH | SR_CELV;
                    stop = true;
                    s.bup_flag = if r.bd.ctl_len & BD_BUP != 0 { BUP_LAST } else { 0 };
                } else {
                    r.civ = r.piv;
                    r.piv = (r.piv + 1) % 32;
                    self.fetch_bd(r);
                }
                self.update_sr(s, index, new_sr);
            }
        }
    }

    /// `ac97_on_reset()`.
    fn on_reset(&self) {
        let mut s = self.lock();
        for i in 0..LAST_INDEX {
            self.reset_bm_regs(&mut s, i);
        }
        // The mixer too: the Windows XP driver reads the vendor ID before it resets the codec.
        self.mixer_reset(&mut s);
    }
}

fn mixer_store(s: &mut State, i: u32, v: u16) {
    let i = i as usize;
    if i + 2 > s.mixer_data.len() {
        return;
    }
    s.mixer_data[i..i + 2].copy_from_slice(&v.to_le_bytes());
}

fn mixer_load(s: &State, i: u32) -> u16 {
    let i = i as usize;
    if i + 2 > s.mixer_data.len() {
        return 0xffff;
    }
    u16::from_le_bytes([s.mixer_data[i], s.mixer_data[i + 1]])
}

fn record_select(s: &mut State, val: u32) {
    let rs = val & REC_MASK;
    let ls = (val >> 8) & REC_MASK;
    mixer_store(s, AC97_RECORD_SELECT, (rs | (ls << 8)) as u16);
}

/// `get_volume()`: the mute bit and the two channel volumes scaled to 0..=255.
fn get_volume(vol: u16, mask: u16, inverse: bool) -> (bool, u8, u8) {
    let mute = (vol >> MUTE_SHIFT) & 1 != 0;
    let mut rvol = (255 * u32::from(vol & mask) / u32::from(mask)) as u8;
    let mut lvol = (255 * u32::from((vol >> 8) & mask) / u32::from(mask)) as u8;
    if inverse {
        rvol = 255 - rvol;
        lvol = 255 - lvol;
    }
    (mute, lvol, rvol)
}

/// The `Volume` of `audio_be_set_volume_out_lr()`.
fn volume_lr(mute: bool, lvol: u8, rvol: u8) -> Volume {
    let mut v = Volume { mute, channels: 2, vol: [0; 16] };
    v.vol[0] = lvol;
    v.vol[1] = rvol;
    v
}

/// BAR 0, `ac97_io_nam_ops`.
struct Nam(Arc<Inner>);

impl MmioOps for Nam {
    fn read(&self, _cx: &AccessCtx, addr: u64, size: AccessSize) -> MemResult<u64> {
        let mask = size.mask();
        let size = u64::from(size.bytes());
        if addr / size > 256 {
            return Ok(mask);
        }
        let mut s = self.0.lock();
        s.cas = 0;
        Ok(match size {
            2 => u64::from(mixer_load(&s, addr as u32)),
            _ => mask,
        })
    }

    fn write(&self, _cx: &AccessCtx, addr: u64, size: AccessSize, val: u64) -> MemResult<()> {
        let size = u64::from(size.bytes());
        if addr / size > 256 {
            return Ok(());
        }
        let mut s = self.0.lock();
        match size {
            2 => self.0.nam_writew(&mut s, addr as u32, val as u32),
            _ => s.cas = 0,
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}

/// BAR 1, `ac97_io_nabm_ops`.
struct Nabm(Arc<Inner>);

impl MmioOps for Nabm {
    fn read(&self, _cx: &AccessCtx, addr: u64, size: AccessSize) -> MemResult<u64> {
        let mask = size.mask();
        let size = u64::from(size.bytes());
        if addr / size > 64 {
            return Ok(mask);
        }
        let mut s = self.0.lock();
        let addr = addr as u32;
        let val = match size {
            1 => self.0.nabm_readb(&mut s, addr),
            2 => self.0.nabm_readw(&s, addr),
            _ => self.0.nabm_readl(&s, addr),
        };
        Ok(u64::from(val) & mask)
    }

    fn write(&self, _cx: &AccessCtx, addr: u64, size: AccessSize, val: u64) -> MemResult<()> {
        let size = u64::from(size.bytes());
        if addr / size > 64 {
            return Ok(());
        }
        let mut s = self.0.lock();
        let (addr, val) = (addr as u32, val as u32);
        match size {
            1 => self.0.nabm_writeb(&mut s, addr, val),
            2 => self.0.nabm_writew(&mut s, addr, val),
            _ => self.0.nabm_writel(&mut s, addr, val),
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}

/// The legacy reset hook.
struct Ops(Weak<Inner>);

impl PciDeviceOps for Ops {
    fn reset(&self, _dev: &PciDevice) {
        if let Some(s) = self.0.upgrade() {
            s.on_reset();
        }
    }
}

impl Ac97 {
    /// `ac97_realize()`: plugs the function into `bus` with `be` already resolved by
    /// `audio_be_check()`. `dma` is the bus master address space.
    pub fn realize(
        bus: &PciBus,
        devfn: Option<u8>,
        id: Option<String>,
        be: Arc<AudioBackend>,
        dma: Arc<dyn DmaMemory>,
    ) -> Result<Ac97> {
        let info = PciDeviceInfo {
            name: TYPE_AC97.to_string(),
            id,
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: PCI_DEVICE_ID_INTEL_82801AA_5,
            revision: 0x01,
            class_id: PCI_CLASS_MULTIMEDIA_AUDIO,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;
        dev.with_config(|c| {
            c.config[PCI_STATUS] = PCI_STATUS_FAST_BACK as u8;
            c.config[PCI_STATUS + 1] = (PCI_STATUS_DEVSEL_MEDIUM >> 8) as u8;
            c.config[PCI_CLASS_PROG] = 0x00;
            c.config[PCI_INTERRUPT_LINE] = 0x00;
            c.config[PCI_INTERRUPT_PIN] = 0x01;
        });
        let inner = Arc::new_cyclic(|me| Inner {
            me: me.clone(),
            dev: Arc::clone(&dev),
            be,
            dma,
            s: Mutex::new(State {
                glob_cnt: 0,
                glob_sta: 0,
                cas: 0,
                last_samp: 0,
                bm_regs: [BmRegs::default(); LAST_INDEX],
                mixer_data: [0; 256],
                voice_pi: None,
                voice_po: None,
                voice_mc: None,
                invalid_freq: [0; LAST_INDEX],
                silence: [0; 128],
                bup_flag: 0,
            }),
        });
        let mem = bus.memory();
        let mem_err = |e: ruvm_mem::MemError| Error::generic(e.to_string());
        let nam =
            mem.new_io("ac97-nam", 1024, Arc::new(Nam(Arc::clone(&inner)))).map_err(mem_err)?;
        let nabm =
            mem.new_io("ac97-nabm", 256, Arc::new(Nabm(Arc::clone(&inner)))).map_err(mem_err)?;
        dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_IO, nam);
        dev.register_bar(1, PCI_BASE_ADDRESS_SPACE_IO, nabm);
        dev.set_ops(Arc::new(Ops(Arc::downgrade(&inner))));
        inner.on_reset();
        Ok(Ac97 { inner })
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.inner.dev
    }
}
