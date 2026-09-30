// SPDX-License-Identifier: GPL-2.0-or-later

//! The High Precision Event Timer, from hw/timer/hpet.c and include/hw/timer/hpet.h.
//!
//! The main counter runs at 100 MHz (a 10 ns period) and is computed from the virtual clock plus
//! `hpet_offset`. Each comparator has its own [`Timer`] on that clock. Interrupts go out on 32
//! routed lines, or as an FSB message through [`Hpet::set_msi_handler`] when `msi` is on.
//! In legacy replacement mode timer 0 drives IRQ0 and timer 1 drives the RTC's IRQ8, the PIT
//! and RTC inputs are cut off and `pit_enabled` is lowered.
//!
//! The counter is read without taking the device lock, using the same seqlock scheme as QEMU
//! built from atomics. Everything else runs under one mutex.
//!
//! VMState, trace points and QOM registration are not ported.

use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_base::Error;
use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

pub const HPET_BASE: u64 = 0xfed0_0000;
pub const HPET_LEN: u64 = 0x400;
/// 10 ns.
pub const HPET_CLK_PERIOD: u64 = 10;

/// 1000000 femtoseconds == 1 ns.
pub const FS_PER_NS: u64 = 1_000_000;
pub const HPET_MIN_TIMERS: u8 = 3;
pub const HPET_MAX_TIMERS: u8 = 24;

pub const HPET_NUM_IRQ_ROUTES: usize = 32;

pub const HPET_LEGACY_PIT_INT: u32 = 0;
pub const HPET_LEGACY_RTC_INT: u32 = 1;

pub const HPET_CFG_ENABLE: u64 = 0x001;
pub const HPET_CFG_LEGACY: u64 = 0x002;

pub const HPET_ID: u64 = 0x000;
pub const HPET_PERIOD: u64 = 0x004;
pub const HPET_CFG: u64 = 0x010;
pub const HPET_STATUS: u64 = 0x020;
pub const HPET_COUNTER: u64 = 0x0f0;
pub const HPET_TN_CFG: u64 = 0x000;
pub const HPET_TN_CMP: u64 = 0x008;
pub const HPET_TN_ROUTE: u64 = 0x010;
pub const HPET_CFG_WRITE_MASK: u64 = 0x3;

pub const HPET_ID_NUM_TIM_SHIFT: u32 = 8;
pub const HPET_ID_NUM_TIM_MASK: u64 = 0x1f00;

pub const HPET_TN_TYPE_LEVEL: u64 = 0x002;
pub const HPET_TN_ENABLE: u64 = 0x004;
pub const HPET_TN_PERIODIC: u64 = 0x008;
pub const HPET_TN_PERIODIC_CAP: u64 = 0x010;
pub const HPET_TN_SIZE_CAP: u64 = 0x020;
pub const HPET_TN_SETVAL: u64 = 0x040;
pub const HPET_TN_32BIT: u64 = 0x100;
pub const HPET_TN_INT_ROUTE_MASK: u64 = 0x3e00;
pub const HPET_TN_FSB_ENABLE: u64 = 0x4000;
pub const HPET_TN_FSB_CAP: u64 = 0x8000;
pub const HPET_TN_CFG_WRITE_MASK: u64 = 0x7f4e;
pub const HPET_TN_INT_ROUTE_SHIFT: u32 = 9;
pub const HPET_TN_INT_ROUTE_CAP_SHIFT: u32 = 32;

/// `TYPE_HPET`.
pub const TYPE_HPET: &str = "hpet";
/// `HPET_INTCAP`, the name of the interrupt capability property.
pub const HPET_INTCAP: &str = "hpet-intcap";

/// Bit of `flags` for the `msi` property.
const HPET_MSI_SUPPORT: u32 = 0;

/// `RTC_ISA_IRQ`.
const RTC_ISA_IRQ: usize = 8;

/// `struct hpet_fw_entry`, one HPET block as the firmware sees it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct HpetFwEntry {
    pub event_timer_block_id: u32,
    pub address: u64,
    pub min_tick: u16,
    pub page_prot: u8,
}

/// `struct hpet_fw_config`, what the board puts in the `etc/hpet` style fw_cfg data and the
/// ACPI HPET table. QEMU keeps one global copy, `hpet_fw_cfg`; here the board owns it and
/// passes it to [`Hpet::realize`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HpetFwConfig {
    pub count: u8,
    pub hpet: [HpetFwEntry; 8],
}

impl Default for HpetFwConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl HpetFwConfig {
    /// The size of the packed C struct.
    pub const PACKED_SIZE: usize = 1 + 8 * 15;

    /// The initial value, `{.count = UINT8_MAX}`, meaning no HPET was created yet.
    pub const fn new() -> Self {
        HpetFwConfig {
            count: u8::MAX,
            hpet: [HpetFwEntry { event_timer_block_id: 0, address: 0, min_tick: 0, page_prot: 0 };
                8],
        }
    }

    /// The `QEMU_PACKED` little endian layout of the struct.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(Self::PACKED_SIZE);
        v.push(self.count);
        for e in &self.hpet {
            v.extend_from_slice(&e.event_timer_block_id.to_le_bytes());
            v.extend_from_slice(&e.address.to_le_bytes());
            v.extend_from_slice(&e.min_tick.to_le_bytes());
            v.push(e.page_prot);
        }
        v
    }
}

/// The properties of the `hpet` device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HpetProperties {
    /// `timers`, the number of comparators.
    pub timers: u8,
    /// `msi`: advertise FSB delivery in each timer's capabilities.
    pub msi: bool,
    /// `hpet-intcap`: which routes each timer may use. The board must set it.
    pub intcap: u32,
}

impl Default for HpetProperties {
    fn default() -> Self {
        HpetProperties { timers: HPET_MIN_TIMERS, msi: false, intcap: 0 }
    }
}

/// Where FSB interrupts go: `address_space_stl_le(&address_space_memory, addr, data)`.
pub type HpetMsiHandler = Arc<dyn Fn(u64, u32) + Send + Sync>;

/// `HPETTimer`, without the `QEMUTimer`, which lives in [`Hpet::qemu_timers`].
#[derive(Copy, Clone, Debug, Default)]
struct HpetTimer {
    /// Timer number.
    tn: u8,
    /// Configuration and capabilities.
    config: u64,
    /// Comparator.
    cmp: u64,
    /// FSB route.
    fsb: u64,
    /// Comparator extended to counter width.
    cmp64: u64,
    /// Last value written to the comparator.
    period: u64,
    /// The next pop is the one shot 32 bit wrap interrupt, the one after is the real expiration.
    wrap_flag: u8,
    /// Last value armed, to avoid timer storms.
    last: u64,
}

impl HpetTimer {
    /// `timer_int_route()`.
    fn int_route(&self) -> usize {
        ((self.config & HPET_TN_INT_ROUTE_MASK) >> HPET_TN_INT_ROUTE_SHIFT) as usize
    }

    /// `timer_fsb_route()`.
    fn fsb_route(&self) -> bool {
        self.config & HPET_TN_FSB_ENABLE != 0
    }

    /// `timer_is_periodic()`.
    fn is_periodic(&self) -> bool {
        self.config & HPET_TN_PERIODIC != 0
    }

    /// `timer_enabled()`.
    fn enabled(&self) -> bool {
        self.config & HPET_TN_ENABLE != 0
    }
}

/// The part of `HPETState` behind `s->lock`.
#[derive(Debug)]
struct HpetState {
    timer: Vec<HpetTimer>,
    /// Interrupt status register.
    isr: u64,
}

/// `HPETState`.
pub struct Hpet {
    clock: Arc<Clock>,
    lock: Mutex<HpetState>,
    /// `state_version`, a seqlock around `config`, `hpet_offset` and `hpet_counter` so the
    /// counter can be read without the lock. Those three are only written with `lock` held.
    state_version: AtomicU32,
    config: AtomicU64,
    hpet_offset: AtomicU64,
    hpet_counter: AtomicU64,
    rtc_irq_level: AtomicU8,
    irqs: Vec<IrqPin>,
    pit_enabled: IrqPin,
    msi_handler: Mutex<Option<HpetMsiHandler>>,
    qemu_timers: Vec<Timer>,
    num_timers: u8,
    flags: u32,
    intcap: u32,
    capability: u64,
    hpet_id: u8,
    address: u64,
}

impl fmt::Debug for Hpet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hpet")
            .field("hpet_id", &self.hpet_id)
            .field("num_timers", &self.num_timers)
            .field("config", &self.config.load(Ordering::SeqCst))
            .finish_non_exhaustive()
    }
}

/// `hpet_time_after()`.
fn hpet_time_after(a: u64, b: u64) -> bool {
    (b.wrapping_sub(a) as i64) < 0
}

/// `ticks_to_ns()`.
fn ticks_to_ns(value: u64) -> u64 {
    value.wrapping_mul(HPET_CLK_PERIOD)
}

/// `ns_to_ticks()`.
fn ns_to_ticks(value: u64) -> u64 {
    value / HPET_CLK_PERIOD
}

/// `hpet_fixup_reg()`.
fn hpet_fixup_reg(new: u64, old: u64, mask: u64) -> u64 {
    (new & mask) | (old & !mask)
}

/// `activating_bit()`.
fn activating_bit(old: u64, new: u64, mask: u64) -> bool {
    old & mask == 0 && new & mask != 0
}

/// `deactivating_bit()`.
fn deactivating_bit(old: u64, new: u64, mask: u64) -> bool {
    old & mask != 0 && new & mask == 0
}

/// `deposit64()`.
fn deposit64(value: u64, start: u32, length: u32, field: u64) -> u64 {
    let mask = if length >= 64 { u64::MAX } else { ((1u64 << length) - 1) << start };
    (value & !mask) | ((field << start) & mask)
}

/// `hpet_calculate_cmp64()`: the next value of the main counter that matches `target`, either
/// entirely or only in the low 32 bits depending on the timer mode.
fn hpet_calculate_cmp64(t: &HpetTimer, cur_tick: u64, target: u64) -> u64 {
    if t.config & HPET_TN_32BIT != 0 {
        let mut result = deposit64(cur_tick, 0, 32, target);
        if result < cur_tick {
            result = result.wrapping_add(0x1_0000_0000);
        }
        result
    } else {
        target
    }
}

/// `hpet_next_wrap()`.
fn hpet_next_wrap(cur_tick: u64) -> u64 {
    (cur_tick | 0xffff_ffff).wrapping_add(1)
}

impl Hpet {
    /// `hpet_realize()` followed by `hpet_reset()`. `fw` is the board's `hpet_fw_cfg`, which
    /// hands out the instance id, and `address` is where the board maps the registers.
    pub fn realize(
        clock: &Arc<Clock>,
        props: HpetProperties,
        fw: &mut HpetFwConfig,
        address: u64,
    ) -> Result<Arc<Hpet>, Error> {
        if props.timers < HPET_MIN_TIMERS || props.timers > HPET_MAX_TIMERS {
            return Err(Error::generic(format!(
                "hpet.num_timers must be between {HPET_MIN_TIMERS} and {HPET_MAX_TIMERS}"
            )));
        }
        if props.intcap == 0 {
            return Err(Error::generic("hpet.hpet-intcap not initialized"));
        }
        if fw.count == u8::MAX {
            // first instance
            fw.count = 0;
        }
        if fw.count == 8 {
            return Err(Error::generic("Only 8 instances of HPET are allowed"));
        }
        let hpet_id = fw.count;
        fw.count += 1;

        // 64-bit General Capabilities and ID Register; LegacyReplacementRoute.
        let mut capability = 0x8086_a001u64;
        capability |= u64::from(props.timers - 1) << HPET_ID_NUM_TIM_SHIFT;
        capability |= (HPET_CLK_PERIOD * FS_PER_NS) << 32;

        let flags = u32::from(props.msi) << HPET_MSI_SUPPORT;
        let s = Arc::new_cyclic(|weak: &Weak<Hpet>| {
            let qemu_timers = (0..HPET_MAX_TIMERS)
                .map(|i| {
                    let w = weak.clone();
                    clock.new_timer(move || {
                        if let Some(s) = w.upgrade() {
                            s.hpet_timer(usize::from(i));
                        }
                    })
                })
                .collect();
            let timer =
                (0..HPET_MAX_TIMERS).map(|i| HpetTimer { tn: i, ..HpetTimer::default() }).collect();
            Hpet {
                clock: clock.clone(),
                lock: Mutex::new(HpetState { timer, isr: 0 }),
                state_version: AtomicU32::new(0),
                config: AtomicU64::new(0),
                hpet_offset: AtomicU64::new(0),
                hpet_counter: AtomicU64::new(0),
                rtc_irq_level: AtomicU8::new(0),
                irqs: (0..HPET_NUM_IRQ_ROUTES).map(|_| IrqPin::new()).collect(),
                pit_enabled: IrqPin::new(),
                msi_handler: Mutex::new(None),
                qemu_timers,
                num_timers: props.timers,
                flags,
                intcap: props.intcap,
                capability,
                hpet_id,
                address,
            }
        });
        s.reset(fw);
        Ok(s)
    }

    fn state(&self) -> MutexGuard<'_, HpetState> {
        self.lock.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Output line `n` of the 32 interrupt routes, `sysbus_init_irq()`.
    pub fn irq(&self, n: usize) -> &IrqPin {
        &self.irqs[n]
    }

    /// The `pit_enabled` output, low while legacy replacement mode is on.
    pub fn pit_enabled(&self) -> &IrqPin {
        &self.pit_enabled
    }

    /// Input `n` of the two legacy inputs, `HPET_LEGACY_PIT_INT` or `HPET_LEGACY_RTC_INT`.
    pub fn legacy_irq_in(self: &Arc<Self>, n: u32) -> IrqLine {
        let w = Arc::downgrade(self);
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = w.upgrade() {
                    s.hpet_handle_legacy_irq(n, level);
                }
            }),
            n,
        )
    }

    /// Where FSB interrupt messages are written. Without a handler they are dropped.
    pub fn set_msi_handler(&self, handler: Option<HpetMsiHandler>) {
        *self.msi_handler.lock().unwrap_or_else(|p| p.into_inner()) = handler;
    }

    /// The instance number handed out by `hpet_fw_cfg`.
    pub fn hpet_id(&self) -> u8 {
        self.hpet_id
    }

    /// The General Capabilities and ID register.
    pub fn capability(&self) -> u64 {
        self.capability
    }

    /// The `hpet_fw_cfg` entry of this instance as `hpet_reset()` fills it in.
    pub fn fw_entry(&self) -> HpetFwEntry {
        HpetFwEntry {
            event_timer_block_id: self.capability as u32,
            address: self.address,
            min_tick: 0,
            page_prot: 0,
        }
    }

    /// `hpet_in_legacy_mode()`.
    fn in_legacy_mode(&self) -> bool {
        self.config.load(Ordering::SeqCst) & HPET_CFG_LEGACY != 0
    }

    /// `hpet_enabled()`.
    fn enabled(&self) -> bool {
        self.config.load(Ordering::SeqCst) & HPET_CFG_ENABLE != 0
    }

    fn now_ns(&self) -> u64 {
        self.clock.get_ns() as u64
    }

    /// `hpet_get_ticks()`.
    fn get_ticks(&self) -> u64 {
        ns_to_ticks(self.now_ns().wrapping_add(self.hpet_offset.load(Ordering::SeqCst)))
    }

    /// `hpet_get_ns()`.
    fn get_ns(&self, tick: u64) -> u64 {
        ticks_to_ns(tick).wrapping_sub(self.hpet_offset.load(Ordering::SeqCst))
    }

    fn seqlock_write_begin(&self) {
        self.state_version.fetch_add(1, Ordering::SeqCst);
    }

    fn seqlock_write_end(&self) {
        self.state_version.fetch_add(1, Ordering::SeqCst);
    }

    /// `update_irq()`.
    fn update_irq(&self, s: &mut HpetState, tn: usize, set: bool) {
        let timer = s.timer[tn];
        let route = if tn <= 1 && self.in_legacy_mode() {
            // If LegacyReplacementRoute bit is set, HPET specification requires timer0 be routed
            // to IRQ0 in NON-APIC or IRQ2 in the I/O APIC, timer1 be routed to IRQ8 in NON-APIC
            // or IRQ8 in the I/O APIC.
            if tn == 0 { 0 } else { RTC_ISA_IRQ }
        } else {
            timer.int_route()
        };
        let mask = 1u64 << timer.tn;

        if set && timer.config & HPET_TN_TYPE_LEVEL != 0 {
            // If HPET_TN_ENABLE bit is 0, "the timer will still operate and generate appropriate
            // status bits, but will not cause an interrupt"
            s.isr |= mask;
        } else {
            s.isr &= !mask;
        }

        if set && timer.enabled() && self.enabled() {
            if timer.fsb_route() {
                let handler = self.msi_handler.lock().unwrap_or_else(|p| p.into_inner()).clone();
                if let Some(h) = handler {
                    h(timer.fsb >> 32, timer.fsb as u32);
                }
            } else if timer.config & HPET_TN_TYPE_LEVEL != 0 {
                self.irqs[route].raise();
            } else {
                self.irqs[route].pulse();
            }
        } else if !timer.fsb_route() {
            self.irqs[route].lower();
        }
    }

    /// `hpet_arm()`.
    fn arm(&self, s: &mut HpetState, tn: usize, tick: u64) {
        let t = &mut s.timer[tn];
        let mut ns = self.get_ns(tick);

        // Clamp period to reasonable min value (1 us)
        if t.is_periodic() && ns.wrapping_sub(t.last) < 1000 {
            ns = t.last.wrapping_add(1000);
        }

        t.last = ns;
        self.qemu_timers[tn].modify(ns as i64);
    }

    /// `hpet_timer()`: the comparator timer expired.
    fn hpet_timer(&self, tn: usize) {
        let mut guard = self.state();
        let s = &mut *guard;
        let period = s.timer[tn].period;
        let cur_tick = self.get_ticks();

        let t = &mut s.timer[tn];
        if t.is_periodic() && period != 0 {
            while hpet_time_after(cur_tick, t.cmp64) {
                t.cmp64 = t.cmp64.wrapping_add(period);
            }
            if t.config & HPET_TN_32BIT != 0 {
                t.cmp = u64::from(t.cmp64 as u32);
            } else {
                t.cmp = t.cmp64;
            }
            let cmp64 = t.cmp64;
            self.arm(s, tn, cmp64);
        } else if t.wrap_flag != 0 {
            t.wrap_flag = 0;
            let cmp64 = t.cmp64;
            self.arm(s, tn, cmp64);
        }
        self.update_irq(s, tn, true);
    }

    /// `hpet_set_timer()`.
    fn set_timer(&self, s: &mut HpetState, tn: usize) {
        let cur_tick = self.get_ticks();
        let t = &mut s.timer[tn];

        t.wrap_flag = 0;
        t.cmp64 = hpet_calculate_cmp64(t, cur_tick, t.cmp);
        if t.config & HPET_TN_32BIT != 0 {
            // hpet spec says in one-shot 32-bit mode, generate an interrupt when counter wraps
            // in addition to an interrupt with comparator match.
            if !t.is_periodic() && t.cmp64 > hpet_next_wrap(cur_tick) {
                t.wrap_flag = 1;
                self.arm(s, tn, hpet_next_wrap(cur_tick));
                return;
            }
        }
        let cmp64 = t.cmp64;
        self.arm(s, tn, cmp64);
    }

    /// `hpet_del_timer()`.
    fn del_timer(&self, s: &mut HpetState, tn: usize) {
        self.qemu_timers[tn].del();

        if s.isr & (1u64 << tn) != 0 {
            // For level-triggered interrupt, this leaves ISR set but lowers irq.
            self.update_irq(s, tn, true);
        }
    }

    /// `hpet_ram_read()`. The value is cut to `size` bytes.
    pub fn mmio_read(&self, addr: u64, size: u32) -> u64 {
        let v = self.hpet_ram_read(addr);
        if size >= 8 { v } else { v & ((1u64 << (size * 8)) - 1) }
    }

    fn hpet_ram_read(&self, addr: u64) -> u64 {
        let shift = (addr & 4) * 8;
        let addr = addr & !4;

        if addr == HPET_COUNTER {
            // Write update is rare, so busywait here is unlikely to happen
            let cur_tick = loop {
                let version = self.state_version.load(Ordering::SeqCst);
                if version & 1 != 0 {
                    std::hint::spin_loop();
                    continue;
                }
                let cur_tick = if !self.enabled() {
                    self.hpet_counter.load(Ordering::SeqCst)
                } else {
                    self.get_ticks()
                };
                if self.state_version.load(Ordering::SeqCst) == version {
                    break cur_tick;
                }
            };
            return cur_tick >> shift;
        }

        let s = self.state();
        // address range of all global regs
        if addr <= 0xff {
            match addr {
                // including HPET_PERIOD
                HPET_ID => self.capability >> shift,
                HPET_CFG => self.config.load(Ordering::SeqCst) >> shift,
                HPET_STATUS => s.isr >> shift,
                _ => 0,
            }
        } else {
            let timer_id = ((addr - 0x100) / 0x20) as u8;
            if timer_id >= self.num_timers {
                return 0;
            }
            let timer = &s.timer[usize::from(timer_id)];
            match addr & 0x1f {
                // including interrupt capabilities
                HPET_TN_CFG => timer.config >> shift,
                // comparator register
                HPET_TN_CMP => timer.cmp >> shift,
                HPET_TN_ROUTE => timer.fsb >> shift,
                _ => 0,
            }
        }
    }

    /// `hpet_ram_write()`.
    pub fn mmio_write(&self, addr: u64, size: u32, mut value: u64) {
        let shift = ((addr & 4) * 8) as u32;
        let mut len = (size * 8).min(64 - shift);
        let mut guard = self.state();
        let s = &mut *guard;
        let addr = addr & !4;

        // address range of all global regs
        if addr <= 0xff {
            match addr {
                HPET_ID => {}
                HPET_CFG => {
                    let old_val = self.config.load(Ordering::SeqCst);
                    let new_val = deposit64(old_val, shift, len, value);
                    let new_val = hpet_fixup_reg(new_val, old_val, HPET_CFG_WRITE_MASK);
                    self.seqlock_write_begin();
                    self.config.store(new_val, Ordering::SeqCst);
                    if activating_bit(old_val, new_val, HPET_CFG_ENABLE) {
                        // Enable main counter and interrupt generation.
                        let counter = self.hpet_counter.load(Ordering::SeqCst);
                        self.hpet_offset.store(
                            ticks_to_ns(counter).wrapping_sub(self.now_ns()),
                            Ordering::SeqCst,
                        );
                        for i in 0..usize::from(self.num_timers) {
                            if s.timer[i].enabled() && s.isr & (1u64 << i) != 0 {
                                self.update_irq(s, i, true);
                            }
                            self.set_timer(s, i);
                        }
                    } else if deactivating_bit(old_val, new_val, HPET_CFG_ENABLE) {
                        // Halt main counter and disable interrupt generation.
                        self.hpet_counter.store(self.get_ticks(), Ordering::SeqCst);
                        for i in 0..usize::from(self.num_timers) {
                            self.del_timer(s, i);
                        }
                    }
                    self.seqlock_write_end();

                    // i8254 and RTC output pins are disabled when HPET is in legacy mode
                    if activating_bit(old_val, new_val, HPET_CFG_LEGACY) {
                        self.pit_enabled.set(0);
                        self.irqs[0].lower();
                        self.irqs[RTC_ISA_IRQ].lower();
                    } else if deactivating_bit(old_val, new_val, HPET_CFG_LEGACY) {
                        self.irqs[0].lower();
                        self.pit_enabled.set(1);
                        self.irqs[RTC_ISA_IRQ]
                            .set(i32::from(self.rtc_irq_level.load(Ordering::SeqCst)));
                    }
                }
                HPET_STATUS => {
                    let new_val = value << shift;
                    let cleared = new_val & s.isr;
                    for i in 0..usize::from(self.num_timers) {
                        if cleared & (1u64 << i) != 0 {
                            self.update_irq(s, i, false);
                        }
                    }
                }
                HPET_COUNTER => {
                    // Writing while enabled is allowed but has no visible effect.
                    let c = self.hpet_counter.load(Ordering::SeqCst);
                    self.hpet_counter.store(deposit64(c, shift, len, value), Ordering::SeqCst);
                }
                _ => {}
            }
        } else {
            let timer_id = ((addr - 0x100) / 0x20) as u8;
            if timer_id >= self.num_timers {
                return;
            }
            let tn = usize::from(timer_id);
            match addr & 0x18 {
                HPET_TN_CFG => {
                    let old_val = s.timer[tn].config;
                    let new_val = deposit64(old_val, shift, len, value);
                    let new_val = hpet_fixup_reg(new_val, old_val, HPET_TN_CFG_WRITE_MASK);
                    if deactivating_bit(old_val, new_val, HPET_TN_TYPE_LEVEL) {
                        // Do this before changing timer->config; otherwise, if HPET_TN_FSB is
                        // set, update_irq will not lower the qemu_irq.
                        self.update_irq(s, tn, false);
                    }
                    s.timer[tn].config = new_val;
                    if activating_bit(old_val, new_val, HPET_TN_ENABLE) && s.isr & (1u64 << tn) != 0
                    {
                        self.update_irq(s, tn, true);
                    }
                    if new_val & HPET_TN_32BIT != 0 {
                        let t = &mut s.timer[tn];
                        t.cmp = u64::from(t.cmp as u32);
                        t.period = u64::from(t.period as u32);
                    }
                    if self.enabled() {
                        self.set_timer(s, tn);
                    }
                }
                // comparator register
                HPET_TN_CMP => {
                    let t = &mut s.timer[tn];
                    if t.config & HPET_TN_32BIT != 0 {
                        // High 32-bits are zero, leave them untouched.
                        if shift != 0 {
                            return;
                        }
                        len = 64;
                        value = u64::from(value as u32);
                    }
                    if !t.is_periodic() || t.config & HPET_TN_SETVAL != 0 {
                        t.cmp = deposit64(t.cmp, shift, len, value);
                    }
                    if t.is_periodic() {
                        t.period = deposit64(t.period, shift, len, value);
                    }
                    t.config &= !HPET_TN_SETVAL;
                    if self.enabled() {
                        self.set_timer(s, tn);
                    }
                }
                HPET_TN_ROUTE => {
                    let t = &mut s.timer[tn];
                    t.fsb = deposit64(t.fsb, shift, len, value);
                }
                _ => {}
            }
        }
    }

    /// `hpet_reset()`. Also refreshes this instance's entry in `fw`.
    pub fn reset(&self, fw: &mut HpetFwConfig) {
        {
            let mut guard = self.state();
            let s = &mut *guard;
            for i in 0..usize::from(self.num_timers) {
                self.del_timer(s, i);
                let t = &mut s.timer[i];
                t.cmp = !0;
                t.config = HPET_TN_PERIODIC_CAP | HPET_TN_SIZE_CAP;
                if self.flags & (1 << HPET_MSI_SUPPORT) != 0 {
                    t.config |= HPET_TN_FSB_CAP;
                }
                // advertise availability of ioapic int
                t.config |= u64::from(self.intcap) << HPET_TN_INT_ROUTE_CAP_SHIFT;
                t.period = 0;
                t.wrap_flag = 0;
            }

            self.pit_enabled.set(1);
            self.seqlock_write_begin();
            self.hpet_counter.store(0, Ordering::SeqCst);
            self.hpet_offset.store(0, Ordering::SeqCst);
            self.config.store(0, Ordering::SeqCst);
            self.seqlock_write_end();
        }
        fw.hpet[usize::from(self.hpet_id)] = self.fw_entry();

        // to document that the RTC lowers its output on reset as well
        self.rtc_irq_level.store(0, Ordering::SeqCst);
    }

    /// `hpet_handle_legacy_irq()`: the PIT (n = 0) or RTC (n = 1) output, passed through unless
    /// legacy replacement mode is on.
    pub fn hpet_handle_legacy_irq(&self, n: u32, level: i32) {
        if n == HPET_LEGACY_PIT_INT {
            if !self.in_legacy_mode() {
                self.irqs[0].set(level);
            }
        } else {
            self.rtc_irq_level.store(level as u8, Ordering::SeqCst);
            if !self.in_legacy_mode() {
                self.irqs[RTC_ISA_IRQ].set(level);
            }
        }
    }

    /// `hpet_pre_save()` without the migration part: latches the counter while enabled.
    pub fn pre_save(&self) {
        let _s = self.state();
        if self.enabled() {
            self.seqlock_write_begin();
            self.hpet_counter.store(self.get_ticks(), Ordering::SeqCst);
            self.seqlock_write_end();
        }
    }

    /// `hpet_post_load()`: recomputes the hidden comparator state after the registers were
    /// restored.
    pub fn post_load(&self) {
        let mut s = self.state();
        let counter = self.hpet_counter.load(Ordering::SeqCst);
        let last = self.now_ns().wrapping_sub(NANOSECONDS_PER_SECOND as u64);
        for t in s.timer.iter_mut().take(usize::from(self.num_timers)) {
            t.cmp64 = hpet_calculate_cmp64(t, counter, t.cmp);
            t.last = last;
        }
    }
}

/// `hpet_ram_ops`: 4 and 8 byte accesses, little endian.
impl MmioOps for Hpet {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.mmio_read(offset, size.bytes()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.mmio_write(offset, size.bytes(), value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmp64_in_32bit_mode_goes_past_the_counter() {
        let t = HpetTimer { config: HPET_TN_32BIT, ..HpetTimer::default() };
        assert_eq!(hpet_calculate_cmp64(&t, 0x1_8000_0000, 0x10), 0x2_0000_0010);
        assert_eq!(hpet_calculate_cmp64(&t, 0x1_0000_0000, 0x10), 0x1_0000_0010);
        assert_eq!(hpet_next_wrap(0x1_2345_6789), 0x2_0000_0000);
    }

    #[test]
    fn deposit() {
        assert_eq!(deposit64(0xffff_ffff_ffff_ffff, 32, 32, 0x1234), 0x0000_1234_ffff_ffff);
        assert_eq!(deposit64(0, 0, 64, 0xabcd), 0xabcd);
    }

    #[test]
    fn fw_cfg_packs_like_c() {
        let mut fw = HpetFwConfig::new();
        fw.count = 1;
        fw.hpet[0] = HpetFwEntry {
            event_timer_block_id: 0x8086a201,
            address: HPET_BASE,
            ..HpetFwEntry::default()
        };
        let b = fw.to_bytes();
        assert_eq!(b.len(), HpetFwConfig::PACKED_SIZE);
        assert_eq!(&b[..5], &[1, 0x01, 0xa2, 0x86, 0x80]);
        assert_eq!(&b[5..13], &HPET_BASE.to_le_bytes());
    }
}
