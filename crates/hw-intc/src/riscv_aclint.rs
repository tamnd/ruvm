// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V advanced core local interruptor, hw/intc/riscv_aclint.c and
//! include/hw/intc/riscv_aclint.h.
//!
//! Two devices. [`RiscvAclintMtimer`] is `RISCVAclintMTimerState`: one `mtime` counter for all
//! harts, derived from the virtual clock at `timebase-freq` plus a guest set `time_delta`, and an
//! `mtimecmp` register per hart whose output (MTIP) is high once `mtime` has reached it.
//! [`RiscvAclintSwi`] is `RISCVAclintSwiState`: a 4 byte register per hart that drives MSIP, or
//! for the SSWI variant one that only ever raises SSIP.
//!
//! The SiFive CLINT layout the virt board uses is an MSWI at the CLINT base
//! (`RISCV_ACLINT_SWI_SIZE` bytes) followed by an MTIMER of `RISCV_ACLINT_DEFAULT_MTIMER_SIZE`
//! (0x8000) bytes with `mtimecmp` at 0 and `mtime` at 0x7ff8.
//!
//! The MTIMER takes aligned 4 and 8 byte accesses (`riscv_aclint_mtimer_ops.valid` and `.impl`)
//! and the SWI aligned 4 byte ones (`riscv_aclint_swi_ops.valid`), little endian.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState, QOM properties and registration, and the KVM side.
//! - The `qemu_log_mask()` guest error and unimplemented messages are not printed, since the
//!   workspace has no `-d` log yet.
//! - QEMU asks `cpu_by_arch_id()` whether a hart exists. Here the board lists the harts in the
//!   device's range that have no CPU in `absent_harts`; by default every hart exists.
//! - The CPU side is the board's: realize claims `MIP_MTIP` and `MIP_MSIP` with
//!   `riscv_cpu_claim_interrupts()`, and `riscv_aclint_mtimer_create()` installs
//!   `cpu_riscv_read_rtc` as each hart's `rdtime_fn` when `provide_rdtime` is set. Here that
//!   function is [`RiscvAclintMtimer::time`].
//! - After a write to `mtime`, QEMU reprograms each hart's Sstc `stimecmp` and `vstimecmp`
//!   timers with `riscv_timer_write_timecmp()`. Here that is the handler set with
//!   [`RiscvAclintMtimer::set_time_changed_handler`], called once per existing hart, in hart
//!   order, after the device lock is dropped (so after every `mtimecmp` was reprogrammed rather
//!   than interleaved with them).
//! - The MSWI reads MSIP back from the hart's `mip`. Since the MSWI claims that bit, nothing else
//!   changes it, so here the device returns the level it last drove instead.
//! - A `timebase-freq` of 0 panics where QEMU divides by zero.
//! - Register callbacks and timer expiries are serialized by a device lock instead of the BQL.
//!   The `mtimecmp` outputs are driven with it held, so the lines connected to them must not
//!   call back into the MTIMER registers ([`RiscvAclintMtimer::time`] is fine).

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_hw_core::irq::IrqPin;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer, muldiv64};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_RISCV_ACLINT_MTIMER`.
pub const TYPE_RISCV_ACLINT_MTIMER: &str = "riscv.aclint.mtimer";
/// `TYPE_RISCV_ACLINT_SWI`.
pub const TYPE_RISCV_ACLINT_SWI: &str = "riscv.aclint.swi";

/// `RISCV_ACLINT_DEFAULT_MTIMECMP`.
pub const RISCV_ACLINT_DEFAULT_MTIMECMP: u32 = 0x0;
/// `RISCV_ACLINT_DEFAULT_MTIME`.
pub const RISCV_ACLINT_DEFAULT_MTIME: u32 = 0x7ff8;
/// `RISCV_ACLINT_DEFAULT_MTIMER_SIZE`.
pub const RISCV_ACLINT_DEFAULT_MTIMER_SIZE: u32 = 0x8000;
/// `RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ`.
pub const RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ: u32 = 10_000_000;
/// `RISCV_ACLINT_MAX_HARTS`.
pub const RISCV_ACLINT_MAX_HARTS: u32 = 4095;
/// `RISCV_ACLINT_SWI_SIZE`.
pub const RISCV_ACLINT_SWI_SIZE: u32 = 0x4000;

type TimeChanged = Arc<dyn Fn(u32) + Send + Sync>;

/// The properties of `riscv.aclint.mtimer`, as `riscv_aclint_mtimer_create()` sets them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AclintMtimerConfig {
    /// `hartid-base`.
    pub hartid_base: u32,
    /// `num-harts`, at most [`RISCV_ACLINT_MAX_HARTS`].
    pub num_harts: u32,
    /// `timecmp-base`, 8 byte aligned.
    pub timecmp_base: u32,
    /// `time-base`, 8 byte aligned.
    pub time_base: u32,
    /// `aperture-size`, the size of the MMIO region.
    pub aperture_size: u32,
    /// `timebase-freq` in Hz.
    pub timebase_freq: u32,
    /// Hart ids in `hartid_base..hartid_base + num_harts` that have no CPU.
    pub absent_harts: Vec<u32>,
}

impl Default for AclintMtimerConfig {
    /// One hart with the default layout and timebase.
    fn default() -> Self {
        AclintMtimerConfig {
            hartid_base: 0,
            num_harts: 1,
            timecmp_base: RISCV_ACLINT_DEFAULT_MTIMECMP,
            time_base: RISCV_ACLINT_DEFAULT_MTIME,
            aperture_size: RISCV_ACLINT_DEFAULT_MTIMER_SIZE,
            timebase_freq: RISCV_ACLINT_DEFAULT_TIMEBASE_FREQ,
            absent_harts: Vec::new(),
        }
    }
}

/// `RISCVAclintMTimerState`, the `riscv.aclint.mtimer` device.
pub struct RiscvAclintMtimer {
    config: AclintMtimerConfig,
    present: Vec<bool>,
    clock: Arc<Clock>,
    /// `time_delta`, atomic so that [`RiscvAclintMtimer::time`] takes no device lock.
    time_delta: AtomicU64,
    /// `timecmp`. The lock also serializes the register callbacks.
    timecmp: Mutex<Vec<u64>>,
    /// `timers`, `None` for an absent hart.
    timers: Vec<Option<Timer>>,
    /// `timer_irqs`, gpio out `n` is hart `hartid_base + n`.
    timer_irqs: Vec<IrqPin>,
    time_changed: Mutex<Option<TimeChanged>>,
}

impl fmt::Debug for RiscvAclintMtimer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiscvAclintMtimer")
            .field("config", &self.config)
            .field("time_delta", &self.time_delta.load(Ordering::Relaxed))
            .field("timecmp", &*self.lock())
            .finish_non_exhaustive()
    }
}

impl RiscvAclintMtimer {
    /// `riscv_aclint_mtimer_create()` without the mapping and the CPU side: the timers run on
    /// `clock`, the virtual clock. Every `mtimecmp` starts at 0 and the outputs are
    /// disconnected; connect output `n` to the MTIP input of hart `hartid_base + n`.
    pub fn new(clock: Arc<Clock>, config: AclintMtimerConfig) -> Arc<RiscvAclintMtimer> {
        assert!(config.num_harts <= RISCV_ACLINT_MAX_HARTS);
        assert!(config.timecmp_base & 0x7 == 0);
        assert!(config.time_base & 0x7 == 0);
        let present: Vec<bool> = (0..config.num_harts)
            .map(|i| !config.absent_harts.contains(&(config.hartid_base + i)))
            .collect();
        Arc::new_cyclic(|weak: &Weak<RiscvAclintMtimer>| {
            let timers = present
                .iter()
                .enumerate()
                .map(|(i, &p)| {
                    p.then(|| {
                        let w = weak.clone();
                        // `riscv_aclint_mtimer_cb()`. QEMU runs it under the BQL, so it cannot
                        // race with a register write. Here a write may rearm the timer between
                        // the clock taking it off its list and this callback running; the
                        // timer is then pending again and the stale expiry is dropped.
                        clock.new_timer(move || {
                            if let Some(s) = w.upgrade() {
                                let _guard = s.lock();
                                let rearmed = s.timers[i].as_ref().is_some_and(Timer::pending);
                                if !rearmed {
                                    s.timer_irqs[i].raise();
                                }
                            }
                        })
                    })
                })
                .collect();
            RiscvAclintMtimer {
                timecmp: Mutex::new(vec![0; config.num_harts as usize]),
                timer_irqs: (0..config.num_harts).map(|_| IrqPin::new()).collect(),
                present,
                clock,
                time_delta: AtomicU64::new(0),
                timers,
                time_changed: Mutex::new(None),
                config,
            }
        })
    }

    fn lock(&self) -> MutexGuard<'_, Vec<u64>> {
        self.timecmp.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The configuration the device was built with.
    pub fn config(&self) -> &AclintMtimerConfig {
        &self.config
    }

    /// The size of the MMIO region, `aperture-size`.
    pub fn mmio_size(&self) -> u64 {
        u64::from(self.config.aperture_size)
    }

    /// `timebase_freq`: the frequency of `mtime` in Hz, what the device tree advertises as
    /// `timebase-frequency`.
    pub fn timebase_freq(&self) -> u32 {
        self.config.timebase_freq
    }

    /// The MTIP output of hart `hartid_base + n`, gpio out `n`.
    pub fn timer_irq(&self, n: usize) -> &IrqPin {
        &self.timer_irqs[n]
    }

    /// Called after a guest write to `mtime` with each existing hart id, for the CPU to
    /// reprogram its `stimecmp` and `vstimecmp` timers as `riscv_timer_write_timecmp()` does.
    pub fn set_time_changed_handler(&self, f: impl Fn(u32) + Send + Sync + 'static) {
        *self.time_changed.lock().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(f));
    }

    /// `cpu_riscv_read_rtc_raw()`.
    fn read_rtc_raw(&self) -> u64 {
        muldiv64(
            self.clock.get_ns() as u64,
            self.config.timebase_freq,
            NANOSECONDS_PER_SECOND as u32,
        )
    }

    /// `cpu_riscv_read_rtc()`: the current `mtime`, which is also what the `time` CSR reads.
    /// It takes no device lock and can be called from any thread.
    pub fn time(&self) -> u64 {
        self.read_rtc_raw().wrapping_add(self.time_delta.load(Ordering::SeqCst))
    }

    /// The `mtimecmp` of hart `hartid_base + n`.
    pub fn timecmp(&self, n: usize) -> u64 {
        self.lock()[n]
    }

    /// `riscv_aclint_mtimer_write_timecmp()` for hart `hartid_base + n`.
    fn write_timecmp(&self, timecmp: &mut [u64], n: usize, value: u64) {
        let freq = self.config.timebase_freq;
        let rtc = self.time();
        timecmp[n] = value;
        if value <= rtc {
            // A compare value in the past raises the interrupt at once.
            self.timer_irqs[n].raise();
            return;
        }

        // Otherwise set up the future timer interrupt.
        self.timer_irqs[n].lower();
        let diff = value - rtc;
        // Back to nanoseconds, with the arguments of muldiv64 switched.
        let ns_diff = muldiv64(diff, NANOSECONDS_PER_SECOND as u32, freq);
        let next = if (NANOSECONDS_PER_SECOND as u64 > u64::from(freq) && ns_diff < diff)
            || ns_diff > i64::MAX as u64
        {
            i64::MAX as u64
        } else {
            (self.clock.get_ns() as u64).wrapping_add(ns_diff).min(i64::MAX as u64)
        };
        if let Some(t) = &self.timers[n] {
            t.modify(next as i64);
        }
    }

    /// `riscv_aclint_mtimer_read()`.
    pub fn reg_read(&self, addr: u64, size: u32) -> u64 {
        let c = &self.config;
        let timecmp_base = u64::from(c.timecmp_base);
        let time_base = u64::from(c.time_base);
        if addr >= timecmp_base && addr < timecmp_base + (u64::from(c.num_harts) << 3) {
            let n = ((addr - timecmp_base) >> 3) as usize;
            if !self.present[n] {
                // QEMU logs "aclint-mtimer: invalid hartid: %zu".
            } else if addr & 0x7 == 0 {
                // timecmp_lo for RV32/RV64 or timecmp for RV64.
                let timecmp = self.lock()[n];
                return if size == 4 { timecmp & 0xffff_ffff } else { timecmp };
            } else if addr & 0x7 == 4 {
                // timecmp_hi.
                return (self.lock()[n] >> 32) & 0xffff_ffff;
            } else {
                // QEMU logs "aclint-mtimer: invalid read: %08x".
                return 0;
            }
        } else if addr == time_base {
            // time_lo for RV32/RV64 or time for RV64.
            let rtc = self.time();
            return if size == 4 { rtc & 0xffff_ffff } else { rtc };
        } else if addr == time_base + 4 {
            // time_hi.
            return (self.time() >> 32) & 0xffff_ffff;
        }
        // QEMU logs "aclint-mtimer: invalid read: %08x".
        0
    }

    /// `riscv_aclint_mtimer_write()`.
    pub fn reg_write(&self, addr: u64, value: u64, size: u32) {
        let c = &self.config;
        let timecmp_base = u64::from(c.timecmp_base);
        let time_base = u64::from(c.time_base);
        if addr >= timecmp_base && addr < timecmp_base + (u64::from(c.num_harts) << 3) {
            self.write_timecmp_reg(((addr - timecmp_base) >> 3) as usize, addr, value, size);
        } else if addr == time_base || addr == time_base + 4 {
            self.write_time_reg(addr == time_base, value, size);
        }
        // Anything else QEMU logs as "aclint-mtimer: invalid write: %08x".
    }

    /// The `mtimecmp` part of `riscv_aclint_mtimer_write()`, for hart `hartid_base + n`.
    fn write_timecmp_reg(&self, n: usize, addr: u64, value: u64, size: u32) {
        let mut timecmp = self.lock();
        if !self.present[n] {
            // QEMU logs "aclint-mtimer: invalid hartid: %zu".
        } else if addr & 0x7 == 0 {
            if size == 4 {
                // timecmp_lo for RV32/RV64.
                let hi = timecmp[n] >> 32;
                self.write_timecmp(&mut timecmp, n, (hi << 32) | (value & 0xffff_ffff));
            } else {
                // timecmp for RV64.
                self.write_timecmp(&mut timecmp, n, value);
            }
        } else if addr & 0x7 == 4 && size == 4 {
            // timecmp_hi for RV32/RV64.
            let lo = timecmp[n];
            self.write_timecmp(&mut timecmp, n, (value << 32) | (lo & 0xffff_ffff));
        }
        // An 8 byte write at offset 4 QEMU logs as "aclint-mtimer: invalid timecmp_hi write:
        // %08x", and other offsets as "aclint-mtimer: invalid timecmp write: %08x".
    }

    /// The `mtime` part of `riscv_aclint_mtimer_write()`: `low` for the register at
    /// `time-base`, otherwise the one 4 bytes above.
    fn write_time_reg(&self, low: bool, value: u64, size: u32) {
        let c = &self.config;
        let mut timecmp = self.lock();
        let rtc_r = self.read_rtc_raw();
        let rtc = rtc_r.wrapping_add(self.time_delta.load(Ordering::SeqCst));
        let new = match (low, size == 4) {
            // time_lo for RV32/RV64.
            (true, true) => (rtc & !0xffff_ffff) | value,
            // time for RV64.
            (true, false) => value,
            // time_hi for RV32/RV64.
            (false, true) => (value << 32) | (rtc & 0xffff_ffff),
            // QEMU logs "aclint-mtimer: invalid time_hi write: %08x".
            (false, false) => return,
        };
        self.time_delta.store(new.wrapping_sub(rtc_r), Ordering::SeqCst);

        // Check whether the timer interrupt fires for each hart.
        for n in 0..c.num_harts as usize {
            if self.present[n] {
                let v = timecmp[n];
                self.write_timecmp(&mut timecmp, n, v);
            }
        }
        drop(timecmp);
        let hook = self.time_changed.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(hook) = hook {
            for n in 0..c.num_harts {
                if self.present[n as usize] {
                    hook(c.hartid_base + n);
                }
            }
        }
    }

    /// `riscv_aclint_mtimer_reset_enter()`: `mtime` goes back to zero, which reevaluates every
    /// hart's interrupt against its unchanged `mtimecmp`.
    pub fn reset(&self) {
        self.reg_write(u64::from(self.config.time_base), 0, 8);
    }
}

/// `riscv_aclint_mtimer_ops`.
impl MmioOps for RiscvAclintMtimer {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.reg_read(offset, size.bytes()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value, size.bytes());
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 8)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

/// The properties of `riscv.aclint.swi`, as `riscv_aclint_swi_create()` sets them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AclintSwiConfig {
    /// `hartid-base`.
    pub hartid_base: u32,
    /// `num-harts`, at most [`RISCV_ACLINT_MAX_HARTS`].
    pub num_harts: u32,
    /// `sswi`: an SSWI, whose registers raise SSIP and read as zero, rather than an MSWI.
    pub sswi: bool,
    /// Hart ids in `hartid_base..hartid_base + num_harts` that have no CPU.
    pub absent_harts: Vec<u32>,
}

/// `RISCVAclintSwiState`, the `riscv.aclint.swi` device.
pub struct RiscvAclintSwi {
    config: AclintSwiConfig,
    present: Vec<bool>,
    /// The level last driven on each output, standing in for the hart's `mip.MSIP`.
    level: Vec<AtomicBool>,
    /// `soft_irqs`, gpio out `n` is hart `hartid_base + n`.
    soft_irqs: Vec<IrqPin>,
    lock: Mutex<()>,
}

impl fmt::Debug for RiscvAclintSwi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RiscvAclintSwi").field("config", &self.config).finish_non_exhaustive()
    }
}

impl RiscvAclintSwi {
    /// `riscv_aclint_swi_create()` without the mapping and the CPU side. Connect output `n`
    /// to the MSIP input of hart `hartid_base + n`, or its SSIP input for an SSWI.
    pub fn new(config: AclintSwiConfig) -> Arc<RiscvAclintSwi> {
        assert!(config.num_harts <= RISCV_ACLINT_MAX_HARTS);
        let present = (0..config.num_harts)
            .map(|i| !config.absent_harts.contains(&(config.hartid_base + i)))
            .collect();
        Arc::new(RiscvAclintSwi {
            level: (0..config.num_harts).map(|_| AtomicBool::new(false)).collect(),
            soft_irqs: (0..config.num_harts).map(|_| IrqPin::new()).collect(),
            present,
            config,
            lock: Mutex::new(()),
        })
    }

    /// The configuration the device was built with.
    pub fn config(&self) -> &AclintSwiConfig {
        &self.config
    }

    /// The size of the MMIO region, `RISCV_ACLINT_SWI_SIZE`.
    pub fn mmio_size(&self) -> u64 {
        u64::from(RISCV_ACLINT_SWI_SIZE)
    }

    /// The MSIP (or for an SSWI, SSIP) output of hart `hartid_base + n`, gpio out `n`.
    pub fn soft_irq(&self, n: usize) -> &IrqPin {
        &self.soft_irqs[n]
    }

    fn set(&self, n: usize, level: bool) {
        self.level[n].store(level, Ordering::SeqCst);
        self.soft_irqs[n].set_bool(level);
    }

    /// `riscv_aclint_swi_read()`.
    pub fn reg_read(&self, addr: u64) -> u64 {
        if addr < u64::from(self.config.num_harts) << 2 {
            let n = (addr >> 2) as usize;
            if !self.present[n] {
                // QEMU logs "aclint-swi: invalid hartid: %zu".
            } else if addr & 0x3 == 0 {
                return if self.config.sswi {
                    0
                } else {
                    u64::from(self.level[n].load(Ordering::SeqCst))
                };
            }
        }
        // QEMU logs "aclint-swi: invalid read: %08x".
        0
    }

    /// `riscv_aclint_swi_write()`.
    pub fn reg_write(&self, addr: u64, value: u64) {
        let _guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        let n = (addr >> 2) as usize;
        if addr < u64::from(self.config.num_harts) << 2 && self.present[n] && addr & 0x3 == 0 {
            if value & 0x1 != 0 {
                self.set(n, true);
            } else if !self.config.sswi {
                self.set(n, false);
            }
        }
        // For a hart with no CPU QEMU logs "aclint-swi: invalid hartid: %zu", and for anything
        // else outside the registers "aclint-swi: invalid write: %08x".
    }

    /// `riscv_aclint_swi_reset_enter()`: an MSWI clears every MSIP. An SSWI does nothing.
    pub fn reset(&self) {
        let _guard = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        if !self.config.sswi {
            for n in 0..self.config.num_harts as usize {
                self.set(n, false);
            }
        }
    }
}

/// `riscv_aclint_swi_ops`.
impl MmioOps for RiscvAclintSwi {
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
    use ruvm_base::ClockType;
    use ruvm_hw_core::irq::IrqLine;
    use std::sync::atomic::AtomicI32;

    const MTIME: u64 = RISCV_ACLINT_DEFAULT_MTIME as u64;

    fn watch(pin: &IrqPin) -> Arc<AtomicI32> {
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        pin.connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        level
    }

    fn mtimer(harts: u32) -> (Arc<Clock>, Arc<RiscvAclintMtimer>) {
        let clock = Clock::manual(ClockType::Virtual);
        let m = RiscvAclintMtimer::new(
            clock.clone(),
            AclintMtimerConfig { num_harts: harts, ..Default::default() },
        );
        (clock, m)
    }

    #[test]
    fn mtime_follows_the_virtual_clock() {
        let (clock, m) = mtimer(1);
        assert_eq!(m.timebase_freq(), 10_000_000);
        clock.advance_to(1_000_000_000);
        assert_eq!(m.time(), 10_000_000);
        assert_eq!(m.reg_read(MTIME, 8), 10_000_000);
        clock.advance_to(1_000_000_150);
        // 150 ns is one and a half ticks: rounded down.
        assert_eq!(m.time(), 10_000_001);

        // Setting mtime moves time_delta.
        m.reg_write(MTIME, 0x1_0000_0005, 8);
        assert_eq!(m.time(), 0x1_0000_0005);
        assert_eq!(m.reg_read(MTIME, 4), 5);
        assert_eq!(m.reg_read(MTIME + 4, 4), 1);
        m.reg_write(MTIME + 4, 2, 4);
        assert_eq!(m.time(), 0x2_0000_0005);
        m.reg_write(MTIME, 7, 4);
        assert_eq!(m.time(), 0x2_0000_0007);
        clock.advance_to(1_000_000_250);
        assert_eq!(m.time(), 0x2_0000_0008);
    }

    #[test]
    fn mtimecmp_fires_mtip() {
        let (clock, m) = mtimer(2);
        let mtip0 = watch(m.timer_irq(0));
        let mtip1 = watch(m.timer_irq(1));

        // 100 ticks from now is 10 us.
        m.reg_write(0, 100, 8);
        assert_eq!(mtip0.load(Ordering::SeqCst), 0);
        assert_eq!(m.reg_read(0, 8), 100);
        clock.advance_to(9_999);
        assert_eq!(mtip0.load(Ordering::SeqCst), 0);
        clock.advance_to(10_000);
        assert_eq!(mtip0.load(Ordering::SeqCst), 1);
        assert_eq!(mtip1.load(Ordering::SeqCst), -1);

        // A compare value in the past fires at once, a future one lowers the line.
        m.reg_write(8, 50, 8);
        assert_eq!(mtip1.load(Ordering::SeqCst), 1);
        m.reg_write(0, u64::MAX, 8);
        assert_eq!(mtip0.load(Ordering::SeqCst), 0);

        // 32 bit halves.
        m.reg_write(12, 0, 4);
        m.reg_write(8, 300, 4);
        assert_eq!(m.timecmp(1), 300);
        assert_eq!(mtip1.load(Ordering::SeqCst), 0);
        m.reg_write(12, 1, 4);
        assert_eq!(m.reg_read(12, 4), 1);
        assert_eq!(m.reg_read(8, 4), 300);
        assert_eq!(m.timecmp(1), (1 << 32) | 300);

        // Moving mtime past mtimecmp fires it.
        m.reg_write(MTIME, (1 << 32) | 300, 8);
        assert_eq!(mtip1.load(Ordering::SeqCst), 1);
        assert_eq!(mtip0.load(Ordering::SeqCst), 0);

        // Reset zeroes mtime and so fires every hart whose mtimecmp is zero.
        m.reg_write(0, 0, 8);
        m.reset();
        assert_eq!(m.time(), 0);
        assert_eq!(mtip0.load(Ordering::SeqCst), 1);
        assert_eq!(mtip1.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn time_writes_notify_each_hart() {
        let (_clock, m) = mtimer(3);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let s = seen.clone();
        m.set_time_changed_handler(move |h| s.lock().unwrap().push(h));
        m.reg_write(MTIME, 0, 8);
        // A 64 bit write to time_hi is ignored.
        m.reg_write(MTIME + 4, 0, 8);
        assert_eq!(*seen.lock().unwrap(), [0, 1, 2]);
    }

    #[test]
    fn absent_harts_are_skipped() {
        let clock = Clock::manual(ClockType::Virtual);
        let m = RiscvAclintMtimer::new(
            clock,
            AclintMtimerConfig { num_harts: 2, absent_harts: vec![1], ..Default::default() },
        );
        m.reg_write(8, 5, 8);
        assert_eq!(m.timecmp(1), 0);
        assert_eq!(m.reg_read(8, 8), 0);
        assert_eq!(m.valid(), AccessConstraints::any_size(4, 8));
    }

    #[test]
    fn mswi_drives_msip() {
        let swi = RiscvAclintSwi::new(AclintSwiConfig { num_harts: 2, ..Default::default() });
        let msip1 = watch(swi.soft_irq(1));
        swi.reg_write(4, 1);
        assert_eq!(msip1.load(Ordering::SeqCst), 1);
        assert_eq!(swi.reg_read(4), 1);
        assert_eq!(swi.reg_read(0), 0);
        swi.reg_write(4, 2);
        assert_eq!(msip1.load(Ordering::SeqCst), 0);
        assert_eq!(swi.reg_read(4), 0);
        swi.reg_write(4, 1);
        swi.reset();
        assert_eq!(msip1.load(Ordering::SeqCst), 0);
        // Past the last hart.
        swi.reg_write(8, 1);
        assert_eq!(swi.reg_read(8), 0);
    }

    #[test]
    fn sswi_only_raises() {
        let swi =
            RiscvAclintSwi::new(AclintSwiConfig { num_harts: 1, sswi: true, ..Default::default() });
        let count = Arc::new(AtomicI32::new(0));
        let c = count.clone();
        swi.soft_irq(0).connect(IrqLine::from_fn(move |v| {
            assert_eq!(v, 1);
            c.fetch_add(1, Ordering::SeqCst);
        }));
        swi.reg_write(0, 1);
        swi.reg_write(0, 0);
        swi.reg_write(0, 1);
        swi.reset();
        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert_eq!(swi.reg_read(0), 0);
    }
}
