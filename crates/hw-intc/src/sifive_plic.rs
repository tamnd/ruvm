// SPDX-License-Identifier: GPL-2.0-or-later

//! The SiFive platform level interrupt controller, hw/intc/sifive_plic.c and
//! include/hw/intc/sifive_plic.h.
//!
//! [`SiFivePlic`] is `SiFivePLICState`. Every source has a priority, a pending bit and a claimed
//! bit, and every context (one hart in one privilege mode, from the `hart-config` string) has an
//! enable bitmap and a priority threshold. A context's output is high while some source is
//! pending, enabled for it, not claimed and of a priority above the threshold. Reading the
//! context's claim register returns the best such source, clears its pending bit and marks it
//! claimed; writing the source number back completes it.
//!
//! The outputs are QEMU's gpio out array: [`SiFivePlic::s_external_irq`] for each hart (first
//! `num_harts` lines, `s_external_irqs`) and [`SiFivePlic::m_external_irq`] for each hart (next
//! `num_harts` lines, `m_external_irqs`). `sifive_plic_create()` wires the M mode one to the
//! hart's `IRQ_M_EXT` input (MEIP) and the S mode one to `IRQ_S_EXT` (SEIP); the board does the
//! same with [`SiFivePlic::addr_config`]. U mode contexts have no output, as in QEMU.
//! Also as in QEMU, a write to an enable register does not reevaluate the outputs; the next
//! source, priority, threshold, claim or completion event does.
//!
//! The register block is `aperture-size` bytes and only takes aligned 4 byte accesses
//! (`sifive_plic_ops.valid`), little endian.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState, QOM properties and registration, and the KVM side.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - QEMU keeps `pending` and `claimed` in atomics because its gpio inputs can be raised without
//!   the BQL. Here one mutex covers the whole state and the outputs are driven with it held, so
//!   the lines connected to the outputs must not call back into the PLIC.
//! - `char_to_mode()` and `parse_hart_config()` print an error and `exit(1)` on a bad
//!   `hart-config`, and realize fails with "plic: invalid number of interrupt sources". Here
//!   [`SiFivePlic::new`] returns those messages as an error and the board decides.
//! - Realize claims `MIP_SEIP` on every hart with `riscv_cpu_claim_interrupts()` (and fails with
//!   "SEIP already claimed"), and sets `msi_nonbroken`. Both are CPU and machine state the board
//!   owns, so the board must do them.
//! - The asserts in `sifive_plic_create()` that the strides are powers of two are checks in
//!   [`SiFivePlic::new`] with the same effect.
//! - A `num-priorities` of `u32::MAX` makes QEMU divide by zero in the WARL case. Here the
//!   value is stored unreduced instead.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_base::{Error, Result};
use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_SIFIVE_PLIC`.
pub const TYPE_SIFIVE_PLIC: &str = "riscv.sifive.plic";

/// `PLICMode`: the privilege mode of one context.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PlicMode {
    /// `PLICMode_U`.
    U = 0,
    /// `PLICMode_S`.
    S = 1,
    /// `PLICMode_M`.
    M = 2,
}

/// `PLICAddr`: what one context is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlicAddr {
    /// The context number.
    pub addrid: u32,
    /// The hart, counted from 0 for the whole machine (so `hartid-base` is already added).
    pub hartid: u32,
    /// The privilege mode.
    pub mode: PlicMode,
}

/// The properties of `riscv.sifive.plic`, as `sifive_plic_create()` sets them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiFivePlicConfig {
    /// `hart-config`, such as `"MS,MS"`: one comma separated entry per hart, each a set of the
    /// modes `U`, `S` and `M` that hart has a context for.
    pub hart_config: String,
    /// `hartid-base`.
    pub hartid_base: u32,
    /// `num-sources`, counting the reserved source 0.
    pub num_sources: u32,
    /// `num-priorities`.
    pub num_priorities: u32,
    /// `priority-base`.
    pub priority_base: u32,
    /// `pending-base`.
    pub pending_base: u32,
    /// `enable-base`.
    pub enable_base: u32,
    /// `enable-stride`, a power of two.
    pub enable_stride: u32,
    /// `context-base`.
    pub context_base: u32,
    /// `context-stride`, a power of two.
    pub context_stride: u32,
    /// `aperture-size`, the size of the MMIO region.
    pub aperture_size: u32,
}

/// `char_to_mode()`.
fn char_to_mode(c: char) -> Result<PlicMode> {
    match c {
        'U' => Ok(PlicMode::U),
        'S' => Ok(PlicMode::S),
        'M' => Ok(PlicMode::M),
        _ => Err(Error::generic(format!("plic: invalid mode '{c}'"))),
    }
}

/// `parse_hart_config()`: the contexts of `hart_config` and the number of harts it names.
pub fn parse_hart_config(hart_config: &str, hartid_base: u32) -> Result<(Vec<PlicAddr>, u32)> {
    // First pass: validate and count, as QEMU does.
    let mut num_harts = 0u32;
    let mut modes = 0u32;
    for c in hart_config.chars() {
        if c == ',' {
            if modes != 0 {
                num_harts += 1;
                modes = 0;
            }
        } else {
            let m = 1u32 << (char_to_mode(c)? as u32);
            if modes == (modes | m) {
                return Err(Error::generic(format!(
                    "plic: duplicate mode '{c}' in config: {hart_config}"
                )));
            }
            modes |= m;
        }
    }
    if modes != 0 {
        num_harts += 1;
    }

    // Second pass: store the hart and mode of every context.
    let mut addrs = Vec::new();
    let mut hartid = hartid_base;
    let mut modes = 0u32;
    for c in hart_config.chars() {
        if c == ',' {
            if modes != 0 {
                hartid += 1;
                modes = 0;
            }
        } else {
            let mode = char_to_mode(c)?;
            addrs.push(PlicAddr { addrid: addrs.len() as u32, hartid, mode });
            modes |= 1 << (mode as u32);
        }
    }
    Ok((addrs, num_harts))
}

/// `addr_between()`, in the 32 bit arithmetic QEMU uses.
fn addr_between(addr: u32, base: u32, num: u32) -> bool {
    addr >= base && addr - base < num
}

/// The register state of `SiFivePLICState`.
#[derive(Debug)]
struct PlicState {
    source_priority: Vec<u32>,
    target_priority: Vec<u32>,
    pending: Vec<u32>,
    claimed: Vec<u32>,
    enable: Vec<u32>,
}

/// `SiFivePLICState`, the `riscv.sifive.plic` device.
pub struct SiFivePlic {
    config: SiFivePlicConfig,
    addr_config: Vec<PlicAddr>,
    num_harts: u32,
    bitfield_words: u32,
    state: Mutex<PlicState>,
    m_external_irqs: Vec<IrqPin>,
    s_external_irqs: Vec<IrqPin>,
    weak: Weak<SiFivePlic>,
}

impl fmt::Debug for SiFivePlic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SiFivePlic")
            .field("config", &self.config)
            .field("num_harts", &self.num_harts)
            .field("state", &*self.lock())
            .finish_non_exhaustive()
    }
}

fn set_masked(word: &mut u32, mask: u32, level: bool) {
    if level {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

impl SiFivePlic {
    /// `sifive_plic_realize()` with the properties `sifive_plic_create()` sets. The device
    /// starts in its reset state. The outputs are disconnected.
    pub fn new(config: SiFivePlicConfig) -> Result<Arc<SiFivePlic>> {
        if !config.enable_stride.is_power_of_two() && config.enable_stride != 0 {
            return Err(Error::generic("plic: enable-stride is not a power of two"));
        }
        if !config.context_stride.is_power_of_two() && config.context_stride != 0 {
            return Err(Error::generic("plic: context-stride is not a power of two"));
        }
        let (addr_config, num_harts) = parse_hart_config(&config.hart_config, config.hartid_base)?;
        if config.num_sources == 0 {
            return Err(Error::generic("plic: invalid number of interrupt sources"));
        }
        let bitfield_words = config.num_sources.div_ceil(32);
        let num_addrs = addr_config.len();
        let state = PlicState {
            source_priority: vec![0; config.num_sources as usize],
            target_priority: vec![0; num_addrs],
            pending: vec![0; bitfield_words as usize],
            claimed: vec![0; bitfield_words as usize],
            enable: vec![0; bitfield_words as usize * num_addrs],
        };
        Ok(Arc::new_cyclic(|weak| SiFivePlic {
            config,
            addr_config,
            num_harts,
            bitfield_words,
            state: Mutex::new(state),
            m_external_irqs: (0..num_harts).map(|_| IrqPin::new()).collect(),
            s_external_irqs: (0..num_harts).map(|_| IrqPin::new()).collect(),
            weak: weak.clone(),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, PlicState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The configuration the device was built with.
    pub fn config(&self) -> &SiFivePlicConfig {
        &self.config
    }

    /// The contexts, in context number order.
    pub fn addr_config(&self) -> &[PlicAddr] {
        &self.addr_config
    }

    /// `num_harts`: how many harts `hart-config` names.
    pub fn num_harts(&self) -> u32 {
        self.num_harts
    }

    /// `num_addrs`: how many contexts there are.
    pub fn num_addrs(&self) -> u32 {
        self.addr_config.len() as u32
    }

    /// The size of the MMIO region, `aperture-size`.
    pub fn mmio_size(&self) -> u64 {
        u64::from(self.config.aperture_size)
    }

    /// The M mode external interrupt output of hart `hartid_base + n`, gpio out
    /// `num_harts + n`. Connect it to the hart's MEIP input.
    pub fn m_external_irq(&self, n: usize) -> &IrqPin {
        &self.m_external_irqs[n]
    }

    /// The S mode external interrupt output of hart `hartid_base + n`, gpio out `n`. Connect it
    /// to the hart's SEIP input.
    pub fn s_external_irq(&self, n: usize) -> &IrqPin {
        &self.s_external_irqs[n]
    }

    /// Gpio out `n` in QEMU's numbering: the S mode outputs, then the M mode outputs.
    pub fn gpio_out(&self, n: usize) -> &IrqPin {
        let harts = self.num_harts as usize;
        if n < harts { &self.s_external_irqs[n] } else { &self.m_external_irqs[n - harts] }
    }

    /// The output that context `addrid` drives, or `None` for a U mode context.
    pub fn context_irq(&self, addrid: usize) -> Option<&IrqPin> {
        let a = self.addr_config.get(addrid)?;
        let n = (a.hartid - self.config.hartid_base) as usize;
        match a.mode {
            PlicMode::M => Some(&self.m_external_irqs[n]),
            PlicMode::S => Some(&self.s_external_irqs[n]),
            PlicMode::U => None,
        }
    }

    /// Input `n`, from `qdev_init_gpio_in(dev, sifive_plic_irq_request, num_sources)`.
    pub fn input(&self, n: u32) -> IrqLine {
        let w = self.weak.clone();
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = w.upgrade() {
                    s.set_irq(n, level);
                }
            }),
            n,
        )
    }

    /// All `num-sources` inputs.
    pub fn inputs(&self) -> Vec<IrqLine> {
        (0..self.config.num_sources).map(|n| self.input(n)).collect()
    }

    /// `sifive_plic_irq_request()`: a source going high latches its pending bit. A source going
    /// low does nothing.
    pub fn set_irq(&self, irq: u32, level: i32) {
        if level > 0 {
            let mut s = self.lock();
            set_masked(&mut s.pending[(irq >> 5) as usize], 1 << (irq & 31), true);
            self.update(&s);
        }
    }

    /// `sifive_plic_claimed()`: the best source context `addrid` could claim, or 0.
    fn claimed(&self, s: &PlicState, addrid: u32) -> u32 {
        let words = self.bitfield_words as usize;
        let mut max_irq = 0;
        let mut max_prio = s.target_priority[addrid as usize];
        let mut num_irq_in_word = 32;
        for i in 0..words {
            let pending_enabled_not_claimed =
                (s.pending[i] & !s.claimed[i]) & s.enable[addrid as usize * words + i];
            if pending_enabled_not_claimed == 0 {
                continue;
            }
            if i == words - 1 {
                // The last word holds fewer than 32 sources if num-sources is not a multiple
                // of 32.
                num_irq_in_word = self.config.num_sources - (((words - 1) as u32) << 5);
            }
            for j in 0..num_irq_in_word {
                let irq = ((i as u32) << 5) + j;
                let prio = s.source_priority[irq as usize];
                let enabled = pending_enabled_not_claimed & (1 << j) != 0;
                if enabled && prio > max_prio {
                    max_irq = irq;
                    max_prio = prio;
                }
            }
        }
        max_irq
    }

    /// `sifive_plic_update()`: drives the output of every M and S mode context.
    fn update(&self, s: &PlicState) {
        for a in &self.addr_config {
            let level = self.claimed(s, a.addrid) != 0;
            let n = (a.hartid - self.config.hartid_base) as usize;
            match a.mode {
                PlicMode::M => self.m_external_irqs[n].set_bool(level),
                PlicMode::S => self.s_external_irqs[n].set_bool(level),
                PlicMode::U => {}
            }
        }
    }

    /// Whether `num-priorities + 1` is a power of two, in which case the priority registers
    /// are WARL.
    fn priorities_warl(&self) -> bool {
        let n = self.config.num_priorities;
        (n.wrapping_add(1) & n) == 0
    }

    /// The value a WARL priority register keeps.
    fn warl(&self, value: u64) -> u32 {
        let modulus = u64::from(self.config.num_priorities.wrapping_add(1));
        value.checked_rem(modulus).unwrap_or(value) as u32
    }

    /// `sifive_plic_read()`.
    pub fn reg_read(&self, addr: u64) -> u64 {
        let c = &self.config;
        let addr = addr as u32;
        let num_addrs = self.num_addrs();
        let mut s = self.lock();
        if addr_between(addr, c.priority_base, c.num_sources << 2) {
            let irq = (addr - c.priority_base) >> 2;
            return u64::from(s.source_priority[irq as usize]);
        } else if addr_between(addr, c.pending_base, c.num_sources.wrapping_add(31) >> 3) {
            let word = (addr - c.pending_base) >> 2;
            return u64::from(s.pending[word as usize]);
        } else if addr_between(addr, c.enable_base, num_addrs.wrapping_mul(c.enable_stride)) {
            let addrid = (addr - c.enable_base) / c.enable_stride;
            let wordid = (addr & (c.enable_stride - 1)) >> 2;
            if wordid < self.bitfield_words {
                return u64::from(s.enable[(addrid * self.bitfield_words + wordid) as usize]);
            }
        } else if addr_between(addr, c.context_base, num_addrs.wrapping_mul(c.context_stride)) {
            let addrid = (addr - c.context_base) / c.context_stride;
            let contextid = addr & (c.context_stride - 1);
            if contextid == 0 {
                return u64::from(s.target_priority[addrid as usize]);
            } else if contextid == 4 {
                let max_irq = self.claimed(&s, addrid);
                if max_irq != 0 {
                    let (word, bit) = ((max_irq >> 5) as usize, 1 << (max_irq & 31));
                    set_masked(&mut s.pending[word], bit, false);
                    set_masked(&mut s.claimed[word], bit, true);
                }
                self.update(&s);
                return u64::from(max_irq);
            }
        }
        // QEMU logs "sifive_plic_read: Invalid register read 0x%x".
        0
    }

    /// `sifive_plic_write()`.
    pub fn reg_write(&self, addr: u64, value: u64) {
        let c = &self.config;
        let addr = addr as u32;
        let num_addrs = self.num_addrs();
        let mut s = self.lock();
        if addr_between(addr, c.priority_base, c.num_sources << 2) {
            let irq = ((addr - c.priority_base) >> 2) as usize;
            if irq == 0 {
                // Source 0 is reserved. QEMU logs "sifive_plic_write: Invalid source priority
                // write 0x%x".
            } else if self.priorities_warl() {
                s.source_priority[irq] = self.warl(value);
                self.update(&s);
            } else if value <= u64::from(c.num_priorities) {
                s.source_priority[irq] = value as u32;
                self.update(&s);
            }
        } else if addr_between(addr, c.pending_base, c.num_sources.wrapping_add(31) >> 3) {
            // QEMU logs "sifive_plic_write: invalid pending write: 0x%x".
        } else if addr_between(addr, c.enable_base, num_addrs.wrapping_mul(c.enable_stride)) {
            let addrid = (addr - c.enable_base) / c.enable_stride;
            let wordid = (addr & (c.enable_stride - 1)) >> 2;
            if wordid < self.bitfield_words {
                s.enable[(addrid * self.bitfield_words + wordid) as usize] = value as u32;
            }
            // Otherwise QEMU logs "sifive_plic_write: Invalid enable write 0x%x".
        } else if addr_between(addr, c.context_base, num_addrs.wrapping_mul(c.context_stride)) {
            let addrid = ((addr - c.context_base) / c.context_stride) as usize;
            let contextid = addr & (c.context_stride - 1);
            if contextid == 0 {
                if self.priorities_warl() {
                    s.target_priority[addrid] = self.warl(value);
                    self.update(&s);
                } else if value <= u64::from(c.num_priorities) {
                    s.target_priority[addrid] = value as u32;
                    self.update(&s);
                }
            } else if contextid == 4 && value < u64::from(c.num_sources) {
                // Completion. A source number out of range is ignored.
                let irq = value as u32;
                set_masked(&mut s.claimed[(irq >> 5) as usize], 1 << (irq & 31), false);
                self.update(&s);
            }
            // At other offsets QEMU logs "sifive_plic_write: Invalid context write 0x%x".
        }
        // Anything else QEMU logs as "sifive_plic_write: Invalid register write 0x%x".
    }

    /// `sifive_plic_reset()`: clears every register and lowers every output.
    pub fn reset(&self) {
        let mut s = self.lock();
        s.source_priority.fill(0);
        s.target_priority.fill(0);
        s.pending.fill(0);
        s.claimed.fill(0);
        s.enable.fill(0);
        for i in 0..self.num_harts as usize {
            self.m_external_irqs[i].lower();
            self.s_external_irqs[i].lower();
        }
    }

    /// The priority of source `irq`.
    pub fn source_priority(&self, irq: u32) -> u32 {
        self.lock().source_priority[irq as usize]
    }

    /// Whether source `irq` is pending.
    pub fn is_pending(&self, irq: u32) -> bool {
        self.lock().pending[(irq >> 5) as usize] & (1 << (irq & 31)) != 0
    }

    /// Whether source `irq` is claimed and not yet completed.
    pub fn is_claimed(&self, irq: u32) -> bool {
        self.lock().claimed[(irq >> 5) as usize] & (1 << (irq & 31)) != 0
    }
}

/// `sifive_plic_ops`.
impl MmioOps for SiFivePlic {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.reg_read(offset))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    const PENDING: u64 = 0x1000;
    const ENABLE: u64 = 0x2000;
    const CONTEXT: u64 = 0x20_0000;

    /// The virt board's PLIC for two harts.
    fn virt_plic() -> Arc<SiFivePlic> {
        SiFivePlic::new(SiFivePlicConfig {
            hart_config: "MS,MS".into(),
            hartid_base: 0,
            num_sources: 96,
            num_priorities: 7,
            priority_base: 0,
            pending_base: 0x1000,
            enable_base: 0x2000,
            enable_stride: 0x80,
            context_base: 0x20_0000,
            context_stride: 0x1000,
            aperture_size: 0x20_0000 + 4 * 0x1000,
        })
        .unwrap()
    }

    fn watch(pin: &IrqPin) -> Arc<AtomicI32> {
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        pin.connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        level
    }

    #[test]
    fn hart_config_parsing() {
        let (addrs, harts) = parse_hart_config("M,MS,,S", 4).unwrap();
        assert_eq!(harts, 3);
        let got: Vec<_> = addrs.iter().map(|a| (a.addrid, a.hartid, a.mode)).collect();
        assert_eq!(
            got,
            [(0, 4, PlicMode::M), (1, 5, PlicMode::M), (2, 5, PlicMode::S), (3, 6, PlicMode::S)]
        );
        let e = parse_hart_config("MM", 0).unwrap_err();
        assert!(e.to_string().contains("plic: duplicate mode 'M' in config: MM"), "{e}");
        let e = parse_hart_config("MX", 0).unwrap_err();
        assert!(e.to_string().contains("plic: invalid mode 'X'"), "{e}");
    }

    #[test]
    fn claim_complete_threshold_and_enable() {
        let plic = virt_plic();
        assert_eq!(plic.num_addrs(), 4);
        // Context 1 is hart 0 in S mode.
        let seip0 = watch(plic.s_external_irq(0));
        let meip0 = watch(plic.m_external_irq(0));
        let src = plic.inputs();

        plic.reg_write(10 * 4, 3);
        plic.reg_write(11 * 4, 5);
        assert_eq!(plic.reg_read(10 * 4), 3);

        // Pending, but not enabled: no interrupt.
        src[10].raise();
        assert!(plic.is_pending(10));
        assert_eq!(plic.reg_read(PENDING), 1 << 10);
        assert_eq!(seip0.load(Ordering::SeqCst), 0);

        // Enable 10 and 11 for context 1.
        plic.reg_write(ENABLE + 0x80, (1 << 10) | (1 << 11));
        assert_eq!(plic.reg_read(ENABLE + 0x80), (1 << 10) | (1 << 11));
        // As in QEMU, an enable write does not update the outputs; the next event does.
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        plic.reg_write(CONTEXT + 0x1000, 0);
        assert_eq!(seip0.load(Ordering::SeqCst), 1);
        assert_eq!(meip0.load(Ordering::SeqCst), 0);

        // A threshold at the priority masks it.
        plic.reg_write(CONTEXT + 0x1000, 3);
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        plic.reg_write(CONTEXT + 0x1000, 2);
        assert_eq!(seip0.load(Ordering::SeqCst), 1);

        // The higher priority source wins the claim.
        src[11].raise();
        assert_eq!(plic.reg_read(CONTEXT + 0x1004), 11);
        assert!(plic.is_claimed(11));
        assert!(!plic.is_pending(11));
        assert_eq!(seip0.load(Ordering::SeqCst), 1);
        assert_eq!(plic.reg_read(CONTEXT + 0x1004), 10);
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        assert_eq!(plic.reg_read(CONTEXT + 0x1004), 0);

        // A line held high does not pend again until the device raises it again.
        src[11].lower();
        plic.reg_write(CONTEXT + 0x1004, 11);
        assert!(!plic.is_claimed(11));
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        src[11].raise();
        assert_eq!(seip0.load(Ordering::SeqCst), 1);

        // While claimed, a source raised again stays pending but does not interrupt.
        assert_eq!(plic.reg_read(CONTEXT + 0x1004), 11);
        src[11].raise();
        assert!(plic.is_pending(11));
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        plic.reg_write(CONTEXT + 0x1004, 11);
        assert_eq!(seip0.load(Ordering::SeqCst), 1);

        plic.reset();
        assert_eq!(seip0.load(Ordering::SeqCst), 0);
        assert!(!plic.is_pending(11));
    }

    #[test]
    fn priorities_are_warl_and_source_zero_is_reserved() {
        let plic = virt_plic();
        plic.reg_write(4, 0xf);
        assert_eq!(plic.reg_read(4), 7);
        plic.reg_write(4, 9);
        assert_eq!(plic.reg_read(4), 1);
        plic.reg_write(0, 5);
        assert_eq!(plic.reg_read(0), 0);
        plic.reg_write(CONTEXT, 0xa);
        assert_eq!(plic.reg_read(CONTEXT), 2);

        // Without WARL, out of range values are ignored.
        let plic =
            SiFivePlic::new(SiFivePlicConfig { num_priorities: 5, ..virt_plic().config().clone() })
                .unwrap();
        plic.reg_write(4, 3);
        plic.reg_write(4, 6);
        assert_eq!(plic.reg_read(4), 3);
    }

    #[test]
    fn pending_is_read_only_and_holes_read_zero() {
        let plic = virt_plic();
        plic.input(33).raise();
        plic.reg_write(PENDING + 4, 0);
        assert_eq!(plic.reg_read(PENDING + 4), 2);
        // Enable words past the last source word read as zero and ignore writes.
        plic.reg_write(ENABLE + 0xc, 0xffff_ffff);
        assert_eq!(plic.reg_read(ENABLE + 0xc), 0);
        assert_eq!(plic.reg_read(CONTEXT + 8), 0);
        assert_eq!(plic.reg_read(CONTEXT + 4 * 0x1000), 0);
    }

    #[test]
    fn mode_outputs_follow_gpio_numbering() {
        let plic = virt_plic();
        assert!(std::ptr::eq(plic.gpio_out(1), plic.s_external_irq(1)));
        assert!(std::ptr::eq(plic.gpio_out(2), plic.m_external_irq(0)));
        assert!(std::ptr::eq(plic.context_irq(2).unwrap(), plic.m_external_irq(1)));
        assert_eq!(plic.valid(), AccessConstraints::exact(4));
    }
}
