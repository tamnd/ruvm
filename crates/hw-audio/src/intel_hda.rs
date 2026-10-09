// SPDX-License-Identifier: GPL-2.0-or-later

//! The Intel HD Audio controller, QEMU's `hw/audio/intel-hda.c`: `intel-hda` (ICH6) and
//! `ich9-intel-hda`.
//!
//! BAR 0 holds the controller registers, mirrored once at 0x2000. The guest sends codec
//! verbs through the CORB ring or the immediate command registers and gets the answers in
//! the RIRB ring. Eight DMA engines, four for input and four for output, walk buffer
//! descriptor lists; the codecs on the HDA bus pull and push stream data through them from
//! their own timers.
//!
//! Migration of the device state is not ported yet.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_audio::AudioBackend;
use ruvm_base::{Error, Result};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_pci::regs::{PCI_BASE_ADDRESS_SPACE_MEMORY, PCI_INTERRUPT_PIN};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::hda_codec::{Event, HdaAudio, HdaCodecKind, Hook};
use crate::{PCI_CLASS_MULTIMEDIA_HD_AUDIO, PCI_VENDOR_ID_INTEL};

/// `TYPE_INTEL_HDA_GENERIC` with the ICH6 IDs.
pub const TYPE_INTEL_HDA: &str = "intel-hda";
/// The ICH9 flavour.
pub const TYPE_ICH9_INTEL_HDA: &str = "ich9-intel-hda";

const ICH6_GCTL_RESET: u32 = 1 << 0;
const ICH6_CORBCTL_RUN: u32 = 1 << 1;
const ICH6_RIRBWP_RST: u32 = 1 << 15;
const ICH6_RBCTL_IRQ_EN: u32 = 1 << 0;
const ICH6_RBCTL_DMA_EN: u32 = 1 << 1;
const ICH6_RBCTL_OVERRUN_EN: u32 = 1 << 2;
const ICH6_RBSTS_IRQ: u32 = 1 << 0;
const ICH6_RBSTS_OVERRUN: u32 = 1 << 2;
const ICH6_IRS_BUSY: u32 = 1 << 0;
const ICH6_IRS_VALID: u32 = 1 << 1;
const SD_CTL_STREAM_RESET: u32 = 1 << 0;
const SD_STS_FIFO_READY: u32 = 1 << 5;
const HDA_BUFFER_SIZE: u32 = 256;

/// The highest codec address a codec can take, plus one.
const HDA_MAX_CAD: u32 = 15;

/// One buffer descriptor, `bpl`.
#[derive(Clone, Copy, Default)]
struct Bpl {
    addr: u64,
    len: u32,
    flags: u32,
}

/// `IntelHDAStream`.
#[derive(Default)]
struct HdaStream {
    ctl: u32,
    lpib: u32,
    cbl: u32,
    lvi: u32,
    fmt: u32,
    bdlp_lbase: u32,
    bdlp_ubase: u32,
    /// The descriptors read when the stream last started, `None` before that.
    bpl: Option<Vec<Bpl>>,
    bentries: u32,
    bsize: u32,
    be: u32,
    bp: u32,
}

/// The registers of `IntelHDAState`.
#[derive(Default)]
struct Ctrl {
    g_ctl: u32,
    wake_en: u32,
    state_sts: u32,
    int_ctl: u32,
    int_sts: u32,
    wall_clk: u32,
    corb_lbase: u32,
    corb_ubase: u32,
    corb_rp: u32,
    corb_wp: u32,
    corb_ctl: u32,
    corb_sts: u32,
    corb_size: u32,
    rirb_lbase: u32,
    rirb_ubase: u32,
    rirb_wp: u32,
    rirb_cnt: u32,
    rirb_ctl: u32,
    rirb_sts: u32,
    rirb_size: u32,
    dp_lbase: u32,
    dp_ubase: u32,
    icw: u32,
    irr: u32,
    ics: u32,
    st: [HdaStream; 8],
    rirb_count: u32,
    wall_base_ns: i64,
}

/// The register a table entry stores to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field {
    GCtl,
    WakeEn,
    StateSts,
    IntCtl,
    IntSts,
    WallClk,
    CorbLbase,
    CorbUbase,
    CorbWp,
    CorbRp,
    CorbCtl,
    CorbSts,
    CorbSize,
    RirbLbase,
    RirbUbase,
    RirbWp,
    RirbCnt,
    RirbCtl,
    RirbSts,
    RirbSize,
    DpLbase,
    DpUbase,
    Icw,
    Irr,
    Ics,
    StCtl,
    StLpib,
    StCbl,
    StLvi,
    StFmt,
    StBdlpL,
    StBdlpU,
}

/// The write handlers of the register table.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Handler {
    GCtl,
    Irq,
    CorbRun,
    RirbWp,
    RirbSts,
    Ics,
    StCtl,
}

/// `IntelHDAReg`. A register without a field is a constant that reads as `reset`.
#[derive(Clone, Copy)]
struct Reg {
    stream: usize,
    field: Option<Field>,
    shift: u32,
    wmask: u32,
    wclear: u32,
    reset: u32,
    whandler: Option<Handler>,
}

impl Reg {
    const fn konst(reset: u32) -> Reg {
        Reg { stream: 0, field: None, shift: 0, wmask: 0, wclear: 0, reset, whandler: None }
    }

    const fn rw(field: Field, wmask: u32) -> Reg {
        Reg { field: Some(field), wmask, ..Reg::konst(0) }
    }

    const fn wclear(self, wclear: u32) -> Reg {
        Reg { wclear, ..self }
    }

    const fn handler(self, h: Handler) -> Reg {
        Reg { whandler: Some(h), ..self }
    }

    const fn reset(self, reset: u32) -> Reg {
        Reg { reset, ..self }
    }
}

/// `ST_REG(0, 0)`, the first stream register.
const ST_BASE: u64 = 0x80;
/// The size of `regtab`: one past the last stream's BDLPU.
const REGTAB_SIZE: u64 = ST_BASE + 7 * 0x20 + 0x1c + 1;

/// `intel_hda_reg_find()`: the `regtab` entry at `addr`.
fn reg_find(addr: u64) -> Option<Reg> {
    use Field as F;
    use Handler as H;
    if addr >= REGTAB_SIZE {
        return None;
    }
    let reg = match addr {
        0x00 => Reg::konst(0x4401),
        0x02 => Reg::konst(0),
        0x03 => Reg::konst(1),
        0x04 => Reg::konst(0x3c),
        0x06 => Reg::konst(0x1d),
        0x08 => Reg::rw(F::GCtl, 0x0103).handler(H::GCtl),
        0x0c => Reg::rw(F::WakeEn, 0x7fff).handler(H::Irq),
        0x0e => Reg::rw(F::StateSts, 0x7fff).wclear(0x7fff).handler(H::Irq),
        0x20 => Reg::rw(F::IntCtl, 0xc000_00ff).handler(H::Irq),
        0x24 => Reg::rw(F::IntSts, 0xc000_00ff).wclear(0xc000_00ff),
        0x30 => Reg::rw(F::WallClk, 0),
        0x40 => Reg::rw(F::CorbLbase, 0xffff_ff80),
        0x44 => Reg::rw(F::CorbUbase, 0xffff_ffff),
        0x48 => Reg::rw(F::CorbWp, 0xff).handler(H::CorbRun),
        0x4a => Reg::rw(F::CorbRp, 0x80ff),
        0x4c => Reg::rw(F::CorbCtl, 0x03).handler(H::CorbRun),
        0x4d => Reg::rw(F::CorbSts, 0x01).wclear(0x01),
        0x4e => Reg::rw(F::CorbSize, 0).reset(0x42),
        0x50 => Reg::rw(F::RirbLbase, 0xffff_ff80),
        0x54 => Reg::rw(F::RirbUbase, 0xffff_ffff),
        0x58 => Reg::rw(F::RirbWp, 0x8000).handler(H::RirbWp),
        0x5a => Reg::rw(F::RirbCnt, 0xff),
        0x5c => Reg::rw(F::RirbCtl, 0x07),
        0x5d => Reg::rw(F::RirbSts, 0x05).wclear(0x05).handler(H::RirbSts),
        0x5e => Reg::rw(F::RirbSize, 0).reset(0x42),
        0x60 => Reg::rw(F::Icw, 0xffff_ffff),
        0x64 => Reg::rw(F::Irr, 0),
        0x68 => Reg::rw(F::Ics, 0x0003).wclear(0x0002).handler(H::Ics),
        0x70 => Reg::rw(F::DpLbase, 0xffff_ff81),
        0x74 => Reg::rw(F::DpUbase, 0xffff_ffff),
        ST_BASE.. => {
            let stream = ((addr - ST_BASE) / 0x20) as usize;
            let reg = match (addr - ST_BASE) % 0x20 {
                0x00 => Reg::rw(F::StCtl, 0x1cff_001f).handler(H::StCtl),
                0x02 => Reg { shift: 16, ..Reg::rw(F::StCtl, 0x00ff_0000).handler(H::StCtl) },
                0x03 => Reg { shift: 24, ..Reg::rw(F::StCtl, 0x1c00_0000) }
                    .wclear(0x1c00_0000)
                    .handler(H::StCtl)
                    .reset(SD_STS_FIFO_READY << 24),
                0x04 => Reg::rw(F::StLpib, 0),
                0x08 => Reg::rw(F::StCbl, 0xffff_ffff),
                0x0c => Reg::rw(F::StLvi, 0x00ff),
                0x10 => Reg::konst(HDA_BUFFER_SIZE),
                0x12 => Reg::rw(F::StFmt, 0x7f7f),
                0x18 => Reg::rw(F::StBdlpL, 0xffff_ff80),
                0x1c => Reg::rw(F::StBdlpU, 0xffff_ffff),
                _ => return None,
            };
            Reg { stream, ..reg }
        }
        _ => return None,
    };
    Some(reg)
}

impl Ctrl {
    /// `intel_hda_reg_addr()`.
    fn field(&mut self, f: Field, stream: usize) -> &mut u32 {
        let st = &mut self.st[stream];
        match f {
            Field::GCtl => &mut self.g_ctl,
            Field::WakeEn => &mut self.wake_en,
            Field::StateSts => &mut self.state_sts,
            Field::IntCtl => &mut self.int_ctl,
            Field::IntSts => &mut self.int_sts,
            Field::WallClk => &mut self.wall_clk,
            Field::CorbLbase => &mut self.corb_lbase,
            Field::CorbUbase => &mut self.corb_ubase,
            Field::CorbWp => &mut self.corb_wp,
            Field::CorbRp => &mut self.corb_rp,
            Field::CorbCtl => &mut self.corb_ctl,
            Field::CorbSts => &mut self.corb_sts,
            Field::CorbSize => &mut self.corb_size,
            Field::RirbLbase => &mut self.rirb_lbase,
            Field::RirbUbase => &mut self.rirb_ubase,
            Field::RirbWp => &mut self.rirb_wp,
            Field::RirbCnt => &mut self.rirb_cnt,
            Field::RirbCtl => &mut self.rirb_ctl,
            Field::RirbSts => &mut self.rirb_sts,
            Field::RirbSize => &mut self.rirb_size,
            Field::DpLbase => &mut self.dp_lbase,
            Field::DpUbase => &mut self.dp_ubase,
            Field::Icw => &mut self.icw,
            Field::Irr => &mut self.irr,
            Field::Ics => &mut self.ics,
            Field::StCtl => &mut st.ctl,
            Field::StLpib => &mut st.lpib,
            Field::StCbl => &mut st.cbl,
            Field::StLvi => &mut st.lvi,
            Field::StFmt => &mut st.fmt,
            Field::StBdlpL => &mut st.bdlp_lbase,
            Field::StBdlpU => &mut st.bdlp_ubase,
        }
    }

    /// `intel_hda_update_int_sts()`.
    fn update_int_sts(&mut self) {
        let mut sts = 0;
        // Controller status.
        if self.rirb_sts & ICH6_RBSTS_IRQ != 0 {
            sts |= 1 << 30;
        }
        if self.rirb_sts & ICH6_RBSTS_OVERRUN != 0 {
            sts |= 1 << 30;
        }
        if self.state_sts & self.wake_en != 0 {
            sts |= 1 << 30;
        }
        // Stream status: the buffer completion interrupts.
        for (i, st) in self.st.iter().enumerate() {
            if st.ctl & (1 << 26) != 0 {
                sts |= 1 << i;
            }
        }
        // Global status.
        if sts & self.int_ctl != 0 {
            sts |= 1 << 31;
        }
        self.int_sts = sts;
    }

    /// `intel_hda_regs_reset()`: every register with storage goes back to its reset value,
    /// in table order.
    fn regs_reset(&mut self) {
        for addr in 0..REGTAB_SIZE {
            if let Some(Reg { field: Some(f), stream, reset, .. }) = reg_find(addr) {
                *self.field(f, stream) = reset;
            }
        }
    }
}

fn hda_addr(lbase: u32, ubase: u32) -> u64 {
    (u64::from(ubase) << 32) | u64::from(lbase)
}

/// The mutable part of `IntelHDAState`: the registers and the codecs on the HDA bus, oldest
/// first.
struct State {
    c: Ctrl,
    codecs: Vec<HdaAudio>,
    next_cad: u32,
}

struct Inner {
    me: Weak<Inner>,
    dev: Arc<PciDevice>,
    dma: Arc<dyn DmaMemory>,
    clock: Arc<Clock>,
    s: Mutex<State>,
}

/// An `intel-hda` or `ich9-intel-hda` PCI function with its HDA bus.
pub struct IntelHda {
    inner: Arc<Inner>,
}

impl fmt::Debug for IntelHda {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntelHda").field("dev", &self.inner.dev.name()).finish_non_exhaustive()
    }
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.s.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `pci_dma_read()`. Without bus mastering, or outside of memory, the data reads as 0.
    fn dma_read(&self, addr: u64, buf: &mut [u8]) {
        if !self.dev.is_bus_master() || !self.dma.read(addr, buf) {
            buf.fill(0);
        }
    }

    /// `pci_dma_write()`, whether it reached memory.
    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.dev.is_bus_master() && self.dma.write(addr, buf)
    }

    /// `intel_hda_update_irq()`. With MSI a message goes out on every update that finds the
    /// interrupt pending.
    fn update_irq(&self, c: &mut Ctrl) {
        c.update_int_sts();
        let level = c.int_sts & (1 << 31) != 0 && c.int_ctl & (1 << 31) != 0;
        if self.dev.msi_enabled() {
            if level {
                self.dev.msi_notify(0);
            }
        } else {
            self.dev.set_irq(i32::from(level));
        }
    }

    /// `intel_hda_response()`: a codec answers, into the immediate response register or the
    /// RIRB ring.
    fn response(&self, c: &mut Ctrl, cad: u32, solicited: bool, response: u32) {
        if c.ics & ICH6_IRS_BUSY != 0 {
            c.irr = response;
            c.ics &= !(ICH6_IRS_BUSY | 0xf0);
            c.ics |= ICH6_IRS_VALID | (cad << 4);
            return;
        }
        if c.rirb_ctl & ICH6_RBCTL_DMA_EN == 0 {
            // RIRB DMA is off, the response is dropped.
            return;
        }
        let ex = if solicited { 0 } else { 1 << 4 } | cad;
        let wp = (c.rirb_wp + 1) & 0xff;
        let addr = hda_addr(c.rirb_lbase, c.rirb_ubase) + 8 * u64::from(wp);
        let ok1 = self.dma_write(addr, &response.to_le_bytes());
        let ok2 = self.dma_write(addr + 4, &ex.to_le_bytes());
        if !(ok1 && ok2) && c.rirb_ctl & ICH6_RBCTL_OVERRUN_EN != 0 {
            c.rirb_sts |= ICH6_RBSTS_OVERRUN;
            self.update_irq(c);
        }
        c.rirb_wp = wp;
        c.rirb_count += 1;
        if c.rirb_count == c.rirb_cnt || (c.corb_rp & 0xff) == c.corb_wp {
            // The RIRB count is reached, or the CORB ring is empty.
            if c.rirb_ctl & ICH6_RBCTL_IRQ_EN != 0 {
                c.rirb_sts |= ICH6_RBSTS_IRQ;
                self.update_irq(c);
            }
        }
    }

    /// `intel_hda_send_command()`.
    fn send_command(&self, c: &mut Ctrl, codecs: &mut [HdaAudio], verb: u32) {
        let cad = (verb >> 28) & 0x0f;
        if verb & (1 << 27) != 0 {
            // Indirect node addressing, not specified in HDA 1.0.
            return;
        }
        let nid = (verb >> 20) & 0x7f;
        let data = verb & 0xfffff;
        // The bus lists its children newest first.
        let Some(codec) = codecs.iter_mut().rev().find(|a| a.cad == cad) else { return };
        let response = codec.command(nid, data);
        self.response(c, cad, true, response);
    }

    /// `intel_hda_corb_run()`.
    fn corb_run(&self, s: &mut State) {
        let State { c, codecs, .. } = s;
        if c.ics & ICH6_IRS_BUSY != 0 {
            let icw = c.icw;
            self.send_command(c, codecs, icw);
            return;
        }
        loop {
            if c.corb_ctl & ICH6_CORBCTL_RUN == 0 {
                return;
            }
            if (c.corb_rp & 0xff) == c.corb_wp {
                // The CORB ring is empty.
                return;
            }
            if c.rirb_count == c.rirb_cnt {
                // The RIRB count is reached.
                return;
            }
            let rp = (c.corb_rp + 1) & 0xff;
            let mut verb = [0; 4];
            self.dma_read(hda_addr(c.corb_lbase, c.corb_ubase) + 4 * u64::from(rp), &mut verb);
            c.corb_rp = rp;
            self.send_command(c, codecs, u32::from_le_bytes(verb));
        }
    }

    /// `intel_hda_xfer()`: moves `buf` between the codec and the DMA engine running stream
    /// `stnr`, if there is one.
    fn xfer(&self, c: &mut Ctrl, stnr: u32, output: bool, buf: &mut [u8]) -> bool {
        let first = if output { 4 } else { 0 };
        let Some(si) = (first..first + 4).find(|&i| (c.st[i].ctl >> 20) & 0x0f == stnr) else {
            return false;
        };
        let st = &mut c.st[si];
        let Some(bpl) = &st.bpl else { return false };
        let mut irq = false;
        let mut off = 0;
        let mut left = buf.len() as u32;
        let mut s = st.bentries;
        while left > 0 && s > 0 {
            s -= 1;
            let e = bpl[st.be as usize];
            let copy = left.min(st.bsize.wrapping_sub(st.lpib)).min(e.len.wrapping_sub(st.bp));
            let addr = e.addr.wrapping_add(u64::from(st.bp));
            let chunk = &mut buf[off..off + copy as usize];
            if output {
                self.dma_read(addr, chunk);
            } else {
                self.dma_write(addr, chunk);
            }
            st.lpib = st.lpib.wrapping_add(copy);
            st.bp = st.bp.wrapping_add(copy);
            off += copy as usize;
            left -= copy;
            if e.len == st.bp {
                // The descriptor is done.
                if e.flags & 0x01 != 0 {
                    irq = true;
                }
                st.bp = 0;
                st.be += 1;
                if st.be == st.bentries {
                    // Wrap around the list.
                    st.be = 0;
                    st.lpib = 0;
                }
            }
        }
        let lpib = st.lpib;
        if c.dp_lbase & 0x01 != 0 {
            let addr = hda_addr(c.dp_lbase & !0x01, c.dp_ubase) + 8 * si as u64;
            self.dma_write(addr, &lpib.to_le_bytes());
        }
        if irq {
            // Buffer completion interrupt.
            c.st[si].ctl |= 1 << 26;
            self.update_irq(c);
        }
        true
    }

    /// `intel_hda_parse_bdl()`.
    fn parse_bdl(&self, st: &mut HdaStream) {
        let mut addr = hda_addr(st.bdlp_lbase, st.bdlp_ubase);
        st.bentries = st.lvi + 1;
        let mut bpl = Vec::with_capacity(st.bentries as usize);
        for _ in 0..st.bentries {
            let mut b = [0; 16];
            self.dma_read(addr, &mut b);
            bpl.push(Bpl {
                addr: u64::from_le_bytes(b[0..8].try_into().unwrap_or_default()),
                len: u32::from_le_bytes(b[8..12].try_into().unwrap_or_default()),
                flags: u32::from_le_bytes(b[12..16].try_into().unwrap_or_default()),
            });
            addr = addr.wrapping_add(16);
        }
        st.bpl = Some(bpl);
        st.bsize = st.cbl;
        st.lpib = 0;
        st.be = 0;
        st.bp = 0;
    }

    /// `intel_hda_set_st_ctl()`.
    fn set_st_ctl(&self, s: &mut State, stream: usize, old: u32) {
        let output = stream >= 4;
        let State { c, codecs, .. } = s;
        let st = &mut c.st[stream];
        if st.ctl & SD_CTL_STREAM_RESET != 0 {
            st.ctl = (SD_STS_FIFO_READY << 24) | SD_CTL_STREAM_RESET;
        }
        if (st.ctl & 0x02) != (old & 0x02) {
            // The run bit flipped.
            let stnr = (st.ctl >> 20) & 0x0f;
            let running = st.ctl & 0x02 != 0;
            if running {
                self.parse_bdl(st);
            }
            for a in codecs.iter_mut().rev() {
                a.stream(stnr, running, output);
            }
        }
        self.update_irq(c);
    }

    /// `intel_hda_reset()`, after the codecs' resets as `device_cold_reset()` orders them.
    fn reset(&self, s: &mut State) {
        for a in &mut s.codecs {
            a.reset();
        }
        let c = &mut s.c;
        c.regs_reset();
        c.wall_base_ns = self.clock.get_ns();
        for a in &s.codecs {
            c.state_sts |= 1 << a.cad;
        }
        self.update_irq(c);
    }

    /// `intel_hda_reg_write()`.
    fn reg_write(&self, s: &mut State, reg: Reg, val: u32, wmask: u32) {
        let Some(f) = reg.field else { return };
        if reg.wmask == 0 {
            // A write to a read-only register.
            return;
        }
        let r = s.c.field(f, reg.stream);
        let old = *r;
        let val = val.wrapping_shl(reg.shift);
        let wmask = wmask.wrapping_shl(reg.shift) & reg.wmask;
        *r &= !wmask;
        *r |= wmask & val;
        *r &= !(val & reg.wclear);
        match reg.whandler {
            None => {}
            Some(Handler::GCtl) => {
                if s.c.g_ctl & ICH6_GCTL_RESET == 0 {
                    self.reset(s);
                }
            }
            Some(Handler::Irq) => self.update_irq(&mut s.c),
            Some(Handler::CorbRun) => self.corb_run(s),
            Some(Handler::RirbWp) => {
                if s.c.rirb_wp & ICH6_RIRBWP_RST != 0 {
                    s.c.rirb_wp = 0;
                }
            }
            Some(Handler::RirbSts) => {
                self.update_irq(&mut s.c);
                if old & ICH6_RBSTS_IRQ != 0 && s.c.rirb_sts & ICH6_RBSTS_IRQ == 0 {
                    // ICH6_RBSTS_IRQ was cleared.
                    s.c.rirb_count = 0;
                    self.corb_run(s);
                }
            }
            Some(Handler::Ics) => {
                if s.c.ics & ICH6_IRS_BUSY != 0 {
                    self.corb_run(s);
                }
            }
            Some(Handler::StCtl) => self.set_st_ctl(s, reg.stream, old),
        }
    }

    /// `intel_hda_reg_read()`.
    fn reg_read(&self, s: &mut State, reg: Reg, rmask: u32) -> u32 {
        let Some(f) = reg.field else {
            // A constant register.
            return reg.reset;
        };
        if f == Field::WallClk {
            // intel_hda_get_wall_clk(), at 24 MHz.
            let ns = self.clock.get_ns().wrapping_sub(s.c.wall_base_ns);
            s.c.wall_clk = (ns.wrapping_mul(24) / 1000) as u32;
        }
        (*s.c.field(f, reg.stream) >> reg.shift) & rmask
    }

    /// A codec's timer or voice wants the controller.
    fn codec_event(&self, index: usize, ev: Event) {
        let mut s = self.lock();
        let State { c, codecs, .. } = &mut *s;
        let Some(codec) = codecs.get_mut(index) else { return };
        match ev {
            Event::Timer(si) => {
                codec.timer(si, &mut |stnr, output, buf| self.xfer(c, stnr, output, buf));
            }
            Event::Voice(si, avail) => codec.voice(si, avail),
        }
    }
}

/// BAR 0, `intel_hda_mmio_ops`: the registers at 0 and their alias at 0x2000.
struct Mmio(Arc<Inner>);

impl MmioOps for Mmio {
    fn read(&self, _cx: &AccessCtx, addr: u64, size: AccessSize) -> MemResult<u64> {
        let Some(reg) = reg_find(addr & 0x1fff) else { return Ok(0) };
        let mut s = self.0.lock();
        let val = self.0.reg_read(&mut s, reg, size.mask() as u32);
        Ok(u64::from(val) & size.mask())
    }

    fn write(&self, _cx: &AccessCtx, addr: u64, size: AccessSize, val: u64) -> MemResult<()> {
        let Some(reg) = reg_find(addr & 0x1fff) else { return Ok(()) };
        let mut s = self.0.lock();
        self.0.reg_write(&mut s, reg, val as u32, size.mask() as u32);
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
        if let Some(i) = self.0.upgrade() {
            let mut s = i.lock();
            i.reset(&mut s);
        }
    }
}

impl IntelHda {
    /// `intel_hda_realize()`. `ich9` picks `ich9-intel-hda` over `intel-hda`. `msi` is the
    /// `msi` property, `None` for `auto`. `clock` is the virtual clock and `dma` the bus master
    /// address space.
    #[allow(clippy::too_many_arguments)]
    pub fn realize(
        bus: &PciBus,
        devfn: Option<u8>,
        id: Option<String>,
        ich9: bool,
        msi: Option<bool>,
        old_msi_addr: bool,
        clock: Arc<Clock>,
        dma: Arc<dyn DmaMemory>,
    ) -> Result<IntelHda> {
        let (name, device_id, revision) =
            if ich9 { (TYPE_ICH9_INTEL_HDA, 0x293e, 3) } else { (TYPE_INTEL_HDA, 0x2668, 1) };
        let info = PciDeviceInfo {
            name: name.to_string(),
            id,
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id,
            revision,
            class_id: PCI_CLASS_MULTIMEDIA_HD_AUDIO,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;
        dev.with_config(|c| {
            c.config[PCI_INTERRUPT_PIN] = 1;
            // HDCTL bit 0 selects the signaling mode, 1 for HDA and 0 for AC97.
            c.config[0x40] = 0x01;
        });
        if msi != Some(false) {
            let offset = if old_msi_addr { 0x50 } else { 0x60 };
            if let Err(e) = dev.msi_init(offset, 1, true, false) {
                // With msi=auto MSI is quietly left off.
                if msi == Some(true) {
                    bus.unregister_device(&dev);
                    return Err(e.hint(
                        "You have to use msi=auto (default) or msi=off with this machine type.\n",
                    ));
                }
            }
        }
        let inner = Arc::new_cyclic(|me| Inner {
            me: me.clone(),
            dev: Arc::clone(&dev),
            dma,
            clock,
            s: Mutex::new(State { c: Ctrl::default(), codecs: Vec::new(), next_cad: 0 }),
        });
        let mmio = bus
            .memory()
            .new_io(name, 0x4000, Arc::new(Mmio(Arc::clone(&inner))))
            .map_err(|e| Error::generic(e.to_string()))?;
        dev.register_bar(0, PCI_BASE_ADDRESS_SPACE_MEMORY, mmio);
        dev.set_ops(Arc::new(Ops(Arc::downgrade(&inner))));
        {
            let mut s = inner.lock();
            inner.reset(&mut s);
        }
        Ok(IntelHda { inner })
    }

    /// `hda_codec_dev_realize()` and the codec's init: plugs a codec into the HDA bus. `cad`
    /// is the `cad` property, where `u32::MAX` (QEMU's -1) takes the next free address. `be`
    /// resolves the backend, `audio_be_check()`, once the address is known to fit.
    pub fn add_codec(
        &self,
        kind: HdaCodecKind,
        cad: u32,
        mixer: bool,
        be: impl FnOnce() -> Result<Arc<AudioBackend>>,
    ) -> Result<()> {
        let inner = &self.inner;
        let cad = {
            let mut s = inner.lock();
            let cad = if cad == u32::MAX { s.next_cad } else { cad };
            if cad >= HDA_MAX_CAD {
                return Err(Error::generic("HDA audio codec address is full"));
            }
            s.next_cad = cad + 1;
            cad
        };
        let be = be()?;
        let mut s = inner.lock();
        let index = s.codecs.len();
        let me = inner.me.clone();
        let hook: Hook = Arc::new(move |ev| {
            if let Some(i) = me.upgrade() {
                i.codec_event(index, ev);
            }
        });
        let codec = HdaAudio::new(kind, cad, mixer, be, Arc::clone(&inner.clock), hook);
        s.codecs.push(codec);
        // QEMU resets the machine once every device is in; ruvm's devices start out reset,
        // so the controller sees its new codec the same way.
        inner.reset(&mut s);
        Ok(())
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.inner.dev
    }
}
