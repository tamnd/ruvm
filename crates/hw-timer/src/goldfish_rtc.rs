// SPDX-License-Identifier: GPL-2.0-or-later

//! The Goldfish virtual platform real time clock, hw/rtc/goldfish_rtc.c.
//!
//! [`GoldfishRtc`] is `GoldfishRTCState`. The counter is nanoseconds since the epoch:
//! `tick_offset` plus the RTC clock (the clock `-rtc clock=` picks). Reading `TIME_LOW` latches
//! the upper half for the following `TIME_HIGH` read. The alarm is a timer on the RTC clock that
//! sets the one pending bit when the counter reaches `alarm_next`. The register block is 0x24
//! bytes and only takes aligned 4 byte accesses, little endian unless the `big-endian` property
//! is set.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState (and so the conversion from the old virtual clock relative
//!   `tick_offset_vmstate`), trace points and QOM registration.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - QEMU reads the start date from the global `-rtc base=` state with `qemu_get_timedate()`.
//!   Here the board passes the date the RTC clock's zero corresponds to, as for the PL031.
//! - Like QEMU, reset stops the alarm and clears the alarm and interrupt registers without
//!   lowering the output, and keeps the counter.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use ruvm_hw_core::irq::IrqPin;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_GOLDFISH_RTC`.
pub const TYPE_GOLDFISH_RTC: &str = "goldfish_rtc";

/// Size of the register block, the `goldfish_rtc` MMIO region.
pub const GOLDFISH_RTC_MMIO_SIZE: u64 = 0x24;

/// Low half of the counter. Reading it latches the high half.
pub const RTC_TIME_LOW: u64 = 0x00;
/// High half of the counter, as latched by the last `TIME_LOW` read.
pub const RTC_TIME_HIGH: u64 = 0x04;
/// Low half of the alarm. Writing it arms the alarm.
pub const RTC_ALARM_LOW: u64 = 0x08;
/// High half of the alarm.
pub const RTC_ALARM_HIGH: u64 = 0x0c;
/// Interrupt enable.
pub const RTC_IRQ_ENABLED: u64 = 0x10;
/// Write to stop the alarm.
pub const RTC_CLEAR_ALARM: u64 = 0x14;
/// Whether the alarm is armed.
pub const RTC_ALARM_STATUS: u64 = 0x18;
/// Write to clear the interrupt.
pub const RTC_CLEAR_INTERRUPT: u64 = 0x1c;

/// The register state of `GoldfishRTCState`.
#[derive(Debug)]
struct GoldfishRtcState {
    tick_offset: u64,
    alarm_next: u64,
    alarm_running: u32,
    irq_pending: u32,
    irq_enabled: u32,
    time_high: u32,
}

/// `GoldfishRTCState`, the `goldfish_rtc` device.
pub struct GoldfishRtc {
    state: Mutex<GoldfishRtcState>,
    irq: IrqPin,
    clock: Arc<Clock>,
    timer: Timer,
    big_endian: bool,
}

impl fmt::Debug for GoldfishRtc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GoldfishRtc")
            .field("state", &*self.lock())
            .field("irq", &self.irq)
            .field("big_endian", &self.big_endian)
            .finish_non_exhaustive()
    }
}

fn unix_seconds(t: SystemTime) -> i64 {
    match t.duration_since(UNIX_EPOCH) {
        Ok(d) => d.as_secs() as i64,
        Err(e) => {
            let d = e.duration();
            -(d.as_secs() as i64) - i64::from(d.subsec_nanos() > 0)
        }
    }
}

/// `deposit64(value, start, 32, field)`.
fn deposit32_in_64(value: u64, start: u32, field: u64) -> u64 {
    let mask = 0xffff_ffffu64 << start;
    (value & !mask) | ((field << start) & mask)
}

impl GoldfishRtc {
    /// `goldfish_rtc_realize()`. `rtc_clock` is the clock chosen by `-rtc clock=`, and
    /// `host_date` is the date `qemu_get_timedate()` returns when that clock reads zero. The
    /// counter starts at the whole second of `host_date` plus the current clock time, and
    /// `big_endian` is the `big-endian` property (false on the RISC-V virt board).
    pub fn new(rtc_clock: Arc<Clock>, host_date: SystemTime, big_endian: bool) -> Arc<GoldfishRtc> {
        let ref_start = unix_seconds(host_date);
        Arc::new_cyclic(|weak: &Weak<GoldfishRtc>| {
            let w = weak.clone();
            let timer = rtc_clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    let mut st = s.lock();
                    s.interrupt(&mut st);
                }
            });
            let today = ref_start + rtc_clock.get_ms() / 1000;
            let tick_offset = (today as u64)
                .wrapping_mul(NANOSECONDS_PER_SECOND as u64)
                .wrapping_sub(rtc_clock.get_ns() as u64);
            GoldfishRtc {
                state: Mutex::new(GoldfishRtcState {
                    tick_offset,
                    alarm_next: 0,
                    alarm_running: 0,
                    irq_pending: 0,
                    irq_enabled: 0,
                    time_high: 0,
                }),
                irq: IrqPin::new(),
                clock: rtc_clock,
                timer,
                big_endian,
            }
        })
    }

    fn lock(&self) -> MutexGuard<'_, GoldfishRtcState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The interrupt output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// The offset between the counter and the RTC clock in nanoseconds.
    pub fn tick_offset(&self) -> u64 {
        self.lock().tick_offset
    }

    /// The current counter, nanoseconds since the epoch.
    pub fn count(&self) -> u64 {
        let s = self.lock();
        self.get_count(&s)
    }

    /// `goldfish_rtc_update()`.
    fn update(&self, s: &GoldfishRtcState) {
        self.irq.set_bool((s.irq_pending & s.irq_enabled) != 0);
    }

    /// `goldfish_rtc_interrupt()`.
    fn interrupt(&self, s: &mut GoldfishRtcState) {
        s.alarm_running = 0;
        s.irq_pending = 1;
        self.update(s);
    }

    /// `goldfish_rtc_get_count()`.
    fn get_count(&self, s: &GoldfishRtcState) -> u64 {
        s.tick_offset.wrapping_add(self.clock.get_ns() as u64)
    }

    /// `goldfish_rtc_clear_alarm()`.
    fn clear_alarm(&self, s: &mut GoldfishRtcState) {
        self.timer.del();
        s.alarm_running = 0;
    }

    /// `goldfish_rtc_set_alarm()`.
    fn set_alarm(&self, s: &mut GoldfishRtcState) {
        let ticks = self.get_count(s);
        let event = s.alarm_next;
        if event <= ticks {
            self.clear_alarm(s);
            self.interrupt(s);
        } else {
            // The expiry is the clock now plus (event - ticks), which is event - tick_offset.
            self.timer.modify(event.wrapping_sub(s.tick_offset) as i64);
            s.alarm_running = 1;
        }
    }

    /// `goldfish_rtc_read()`.
    pub fn reg_read(&self, offset: u64) -> u32 {
        let mut s = self.lock();
        // TIME_LOW returns the unsigned low half, and the TIME_HIGH read after it the signed
        // high half of the same value.
        match offset {
            RTC_TIME_LOW => {
                let r = self.get_count(&s);
                s.time_high = (r >> 32) as u32;
                r as u32
            }
            RTC_TIME_HIGH => s.time_high,
            RTC_ALARM_LOW => s.alarm_next as u32,
            RTC_ALARM_HIGH => (s.alarm_next >> 32) as u32,
            RTC_IRQ_ENABLED => s.irq_enabled,
            RTC_ALARM_STATUS => s.alarm_running,
            // QEMU logs "goldfish_rtc_read: offset 0x%x is UNIMP.".
            _ => 0,
        }
    }

    /// `goldfish_rtc_write()`.
    pub fn reg_write(&self, offset: u64, value: u64) {
        let mut guard = self.lock();
        let s = &mut *guard;
        match offset {
            RTC_TIME_LOW | RTC_TIME_HIGH => {
                let start = if offset == RTC_TIME_LOW { 0 } else { 32 };
                let current_tick = self.get_count(s);
                let new_tick = deposit32_in_64(current_tick, start, value);
                s.tick_offset = s.tick_offset.wrapping_add(new_tick.wrapping_sub(current_tick));
            }
            RTC_ALARM_LOW => {
                s.alarm_next = deposit32_in_64(s.alarm_next, 0, value);
                self.set_alarm(s);
            }
            RTC_ALARM_HIGH => s.alarm_next = deposit32_in_64(s.alarm_next, 32, value),
            RTC_IRQ_ENABLED => {
                s.irq_enabled = (value & 0x1) as u32;
                self.update(s);
            }
            RTC_CLEAR_ALARM => self.clear_alarm(s),
            RTC_CLEAR_INTERRUPT => {
                s.irq_pending = 0;
                self.update(s);
            }
            // QEMU logs "goldfish_rtc_write: offset 0x%x is UNIMP.".
            _ => {}
        }
    }

    /// `goldfish_rtc_reset()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        self.timer.del();
        s.alarm_next = 0;
        s.alarm_running = 0;
        s.irq_pending = 0;
        s.irq_enabled = 0;
    }
}

/// `goldfish_rtc_ops`.
impl MmioOps for GoldfishRtc {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.reg_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }

    fn endianness(&self) -> Endian {
        if self.big_endian { Endian::Big } else { Endian::Little }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;
    use ruvm_hw_core::irq::IrqLine;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::time::Duration;

    const NS: u64 = NANOSECONDS_PER_SECOND as u64;
    const DATE: u64 = 1_700_000_000;

    fn rtc() -> (Arc<Clock>, Arc<GoldfishRtc>, Arc<AtomicI32>) {
        let clock = Clock::manual(ClockType::Virtual);
        clock.advance_to(2_500_000_000);
        let rtc = GoldfishRtc::new(clock.clone(), UNIX_EPOCH + Duration::from_secs(DATE), false);
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        rtc.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        (clock, rtc, level)
    }

    fn read_time(rtc: &GoldfishRtc) -> u64 {
        let lo = u64::from(rtc.reg_read(RTC_TIME_LOW));
        (u64::from(rtc.reg_read(RTC_TIME_HIGH)) << 32) | lo
    }

    #[test]
    fn time_reads_the_date_in_nanoseconds() {
        let (clock, rtc, _) = rtc();
        // The date is taken in whole seconds at realize, the clock runs on in nanoseconds.
        assert_eq!(read_time(&rtc), (DATE + 2) * NS);
        clock.advance_to(3_000_000_007);
        let t = (DATE + 2) * NS + 500_000_007;
        assert_eq!(read_time(&rtc), t);

        // TIME_HIGH is latched by TIME_LOW.
        assert_eq!(u64::from(rtc.reg_read(RTC_TIME_LOW)), t & 0xffff_ffff);
        clock.advance_to(3_000_000_000 + (1 << 33));
        assert_eq!(u64::from(rtc.reg_read(RTC_TIME_HIGH)), t >> 32);
        clock.advance_to(3_000_000_007 + (1 << 33));

        // Setting the time.
        rtc.reg_write(RTC_TIME_HIGH, 1);
        rtc.reg_write(RTC_TIME_LOW, 5);
        assert_eq!(read_time(&rtc), (1 << 32) | 5);
        clock.advance_to(3_000_000_010 + (1 << 33));
        assert_eq!(read_time(&rtc), (1 << 32) | 8);
        assert_eq!(rtc.reg_read(0x20), 0);
    }

    #[test]
    fn alarm_fires_and_clears() {
        let (clock, rtc, irq) = rtc();
        let now = rtc.count();
        let alarm = now + 1000;
        rtc.reg_write(RTC_IRQ_ENABLED, 1);
        assert_eq!(irq.load(Ordering::SeqCst), 0);
        rtc.reg_write(RTC_ALARM_HIGH, alarm >> 32);
        rtc.reg_write(RTC_ALARM_LOW, alarm & 0xffff_ffff);
        assert_eq!(rtc.reg_read(RTC_ALARM_STATUS), 1);
        assert_eq!(u64::from(rtc.reg_read(RTC_ALARM_LOW)), alarm & 0xffff_ffff);
        assert_eq!(u64::from(rtc.reg_read(RTC_ALARM_HIGH)), alarm >> 32);
        clock.advance_to(2_500_000_999);
        assert_eq!(irq.load(Ordering::SeqCst), 0);
        clock.advance_to(2_500_001_000);
        assert_eq!(irq.load(Ordering::SeqCst), 1);
        assert_eq!(rtc.reg_read(RTC_ALARM_STATUS), 0);
        rtc.reg_write(RTC_CLEAR_INTERRUPT, 0);
        assert_eq!(irq.load(Ordering::SeqCst), 0);

        // An alarm in the past fires at once; a disabled interrupt stays low.
        rtc.reg_write(RTC_IRQ_ENABLED, 0);
        rtc.reg_write(RTC_ALARM_LOW, 0);
        assert_eq!(irq.load(Ordering::SeqCst), 0);
        rtc.reg_write(RTC_IRQ_ENABLED, 1);
        assert_eq!(irq.load(Ordering::SeqCst), 1);
        rtc.reg_write(RTC_CLEAR_INTERRUPT, 0);

        // CLEAR_ALARM stops an armed alarm.
        let alarm = rtc.count() + 50;
        rtc.reg_write(RTC_ALARM_HIGH, alarm >> 32);
        rtc.reg_write(RTC_ALARM_LOW, alarm & 0xffff_ffff);
        rtc.reg_write(RTC_CLEAR_ALARM, 0);
        assert_eq!(rtc.reg_read(RTC_ALARM_STATUS), 0);
        clock.advance_to(2_500_002_000);
        assert_eq!(irq.load(Ordering::SeqCst), 0);

        rtc.reset();
        assert_eq!(rtc.reg_read(RTC_IRQ_ENABLED), 0);
        assert_eq!(rtc.reg_read(RTC_ALARM_LOW), 0);
    }

    #[test]
    fn endianness_follows_the_property() {
        let clock = Clock::manual(ClockType::Virtual);
        let be = GoldfishRtc::new(clock.clone(), UNIX_EPOCH, true);
        let le = GoldfishRtc::new(clock, UNIX_EPOCH, false);
        assert_eq!(be.endianness(), Endian::Big);
        assert_eq!(le.endianness(), Endian::Little);
        assert_eq!(le.valid(), AccessConstraints::exact(4));
    }
}
