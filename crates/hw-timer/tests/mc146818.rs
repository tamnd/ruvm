// SPDX-License-Identifier: GPL-2.0-or-later

//! tests/qtest/rtc-test.c, driving the RTC through its ports on a manual virtual clock, the way
//! `-rtc clock=vm` runs under qtest.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND};
use ruvm_hw_timer::mc146818::*;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

/// 2023-11-14 22:13:20 UTC, the host date the guest starts from.
const START: i64 = 1_700_000_000;

struct Rtc {
    clock: Arc<Clock>,
    rtc: Mc146818Rtc,
    irq: Arc<AtomicI32>,
}

impl Rtc {
    /// `qtest_start("-rtc clock=vm")` on a pc machine, which uses base_year 2000.
    fn new() -> Self {
        Self::with_policy(LostTickPolicy::Discard)
    }

    fn with_policy(lost_tick_policy: LostTickPolicy) -> Self {
        let clock = Clock::manual(ClockType::Virtual);
        let props = Mc146818Props { base_year: 2000, lost_tick_policy, ..Default::default() };
        let start = UNIX_EPOCH + Duration::from_secs(START as u64);
        let rtc = Mc146818Rtc::new(props, clock.clone(), start).unwrap();
        let irq = Arc::new(AtomicI32::new(0));
        let level = irq.clone();
        rtc.connect_irq(IrqLine::from_fn(move |l| level.store(l, Ordering::SeqCst)));
        Rtc { clock, rtc, irq }
    }

    fn outb(&self, port: u64, val: u8) {
        let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
        self.rtc.write(&cx, port, AccessSize::B1, u64::from(val)).unwrap();
    }

    fn inb(&self, port: u64) -> u8 {
        let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
        self.rtc.read(&cx, port, AccessSize::B1).unwrap() as u8
    }

    fn cmos_read(&self, reg: usize) -> u8 {
        self.outb(0, reg as u8);
        self.inb(1)
    }

    fn cmos_write(&self, reg: usize, val: u8) {
        self.outb(0, reg as u8);
        self.outb(1, val);
    }

    fn get_irq(&self) -> bool {
        self.irq.load(Ordering::SeqCst) != 0
    }

    fn clock_step(&self, ns: i64) -> i64 {
        self.clock.advance_to(self.clock.get_ns() + ns)
    }

    fn clock_step_next(&self) -> i64 {
        let d = self.clock.deadline_ns();
        assert!(d >= 0, "no timer armed");
        self.clock_step(d)
    }

    fn assert_time(&self, h: u8, m: u8, s: u8) {
        assert_eq!(self.cmos_read(RTC_HOURS), h);
        assert_eq!(self.cmos_read(RTC_MINUTES), m);
        assert_eq!(self.cmos_read(RTC_SECONDS), s);
    }

    fn assert_datetime_bcd(&self, h: u8, min: u8, s: u8, d: u8, m: u8, y: u16) {
        self.assert_time(h, min, s);
        assert_eq!(self.cmos_read(RTC_DAY_OF_MONTH), d);
        assert_eq!(self.cmos_read(RTC_MONTH), m);
        assert_eq!(self.cmos_read(RTC_YEAR), (y & 0xff) as u8);
        assert_eq!(self.cmos_read(RTC_CENTURY), (y >> 8) as u8);
    }

    fn cmos_get_date_time(&self) -> i64 {
        let base_year = 2000;
        let mut sec = self.cmos_read(RTC_SECONDS);
        let mut min = self.cmos_read(RTC_MINUTES);
        let mut hour = self.cmos_read(RTC_HOURS);
        let mut mday = self.cmos_read(RTC_DAY_OF_MONTH);
        let mut mon = self.cmos_read(RTC_MONTH);
        let mut year = self.cmos_read(RTC_YEAR);
        let hour_offset;

        if self.cmos_read(RTC_REG_B) & REG_B_DM == 0 {
            sec = bcd2dec(sec);
            min = bcd2dec(min);
            hour = bcd2dec(hour);
            mday = bcd2dec(mday);
            mon = bcd2dec(mon);
            year = bcd2dec(year);
            hour_offset = 80;
        } else {
            hour_offset = 0x80;
        }
        if self.cmos_read(RTC_REG_B) & REG_B_24H == 0 && hour >= hour_offset {
            hour = hour - hour_offset + 12;
        }
        mktimegm(&Tm {
            sec: i32::from(sec),
            min: i32::from(min),
            hour: i32::from(hour),
            mday: i32::from(mday),
            mon: i32::from(mon) - 1,
            year: base_year + i32::from(year) - 1900,
            wday: 0,
        })
    }

    /// The host clock here is `START` plus the virtual clock, so the RTC must match it exactly.
    fn check_time(&self) {
        let start = START + self.clock.get_ns() / NANOSECONDS_PER_SECOND;
        let date: Vec<i64> = (0..4).map(|_| self.cmos_get_date_time()).collect();
        let end = START + self.clock.get_ns() / NANOSECONDS_PER_SECOND;
        let t = date.windows(2).find(|w| w[0] == w[1]).expect("no two readings match")[0];
        assert!(start <= t && t <= end, "RTC is {} seconds off", t - start);
    }

    fn set_time(&self, mode: u8, h: u8, m: u8, s: u8) {
        self.cmos_write(RTC_REG_B, mode);
        self.cmos_write(RTC_REG_A, 0x76);
        self.cmos_write(RTC_HOURS, h);
        self.cmos_write(RTC_MINUTES, m);
        self.cmos_write(RTC_SECONDS, s);
        self.cmos_write(RTC_REG_A, 0x26);
    }

    fn set_datetime_bcd(&self, h: u8, min: u8, s: u8, d: u8, m: u8, y: u16) {
        self.cmos_write(RTC_HOURS, h);
        self.cmos_write(RTC_MINUTES, min);
        self.cmos_write(RTC_SECONDS, s);
        self.cmos_write(RTC_YEAR, (y & 0xff) as u8);
        self.cmos_write(RTC_CENTURY, (y >> 8) as u8);
        self.cmos_write(RTC_MONTH, m);
        self.cmos_write(RTC_DAY_OF_MONTH, d);
    }

    #[allow(clippy::too_many_arguments)]
    fn set_datetime(&self, mode: u8, h: u8, min: u8, s: u8, d: u8, m: u8, y: u16) {
        self.cmos_write(RTC_REG_B, mode);
        self.cmos_write(RTC_REG_A, 0x76);
        self.set_datetime_bcd(h, min, s, d, m, y);
        self.cmos_write(RTC_REG_A, 0x26);
    }

    fn wait_periodic_interrupt(&self, mut real_time: i64) -> i64 {
        while !self.get_irq() {
            real_time = self.clock_step_next();
        }
        assert_ne!(self.cmos_read(RTC_REG_C) & REG_C_PF, 0);
        real_time
    }
}

fn bcd2dec(value: u8) -> u8 {
    ((value >> 4) & 0x0f) * 10 + (value & 0x0f)
}

#[test]
fn check_time_bcd() {
    let t = Rtc::new();
    t.cmos_write(RTC_REG_B, REG_B_24H);
    t.check_time();
    t.clock_step(3 * NANOSECONDS_PER_SECOND + 1);
    t.check_time();
}

#[test]
fn check_time_dec() {
    let t = Rtc::new();
    t.cmos_write(RTC_REG_B, REG_B_24H | REG_B_DM);
    t.check_time();
    t.clock_step(3600 * NANOSECONDS_PER_SECOND);
    t.check_time();
}

#[test]
fn alarm_interrupt() {
    let t = Rtc::new();
    let wiggle = 2;
    let now = gmtime(START + t.clock.get_ns() / NANOSECONDS_PER_SECOND);

    // set DEC mode
    t.cmos_write(RTC_REG_B, REG_B_24H | REG_B_DM);

    assert!(!t.get_irq());
    t.cmos_read(RTC_REG_C);

    t.cmos_write(RTC_SECONDS_ALARM, ((now.sec + 2) % 60) as u8);
    t.cmos_write(RTC_MINUTES_ALARM, RTC_ALARM_DONT_CARE);
    t.cmos_write(RTC_HOURS_ALARM, RTC_ALARM_DONT_CARE);
    t.cmos_write(RTC_REG_B, t.cmos_read(RTC_REG_B) | REG_B_AIE);

    for _ in 0..2 + wiggle {
        if t.get_irq() {
            break;
        }
        t.clock_step(NANOSECONDS_PER_SECOND);
    }

    assert!(t.get_irq());
    assert_ne!(t.cmos_read(RTC_REG_C) & REG_C_AF, 0);
    assert_eq!(t.cmos_read(RTC_REG_C), 0);
}

#[test]
fn alarm_am_pm() {
    let t = Rtc::new();
    t.cmos_write(RTC_MINUTES_ALARM, 0xc0);
    t.cmos_write(RTC_SECONDS_ALARM, 0xc0);

    // set BCD 12 hour mode
    t.cmos_write(RTC_REG_B, 0);

    // Set time and alarm hour.
    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_HOURS_ALARM, 0x82);
    t.cmos_write(RTC_HOURS, 0x81);
    t.cmos_write(RTC_MINUTES, 0x59);
    t.cmos_write(RTC_SECONDS, 0x00);
    t.cmos_read(RTC_REG_C);
    t.cmos_write(RTC_REG_A, 0x26);

    // Check that alarm triggers when AM/PM is set.
    t.clock_step(60 * NANOSECONDS_PER_SECOND);
    assert_eq!(t.cmos_read(RTC_HOURS), 0x82);
    assert_ne!(t.cmos_read(RTC_REG_C) & REG_C_AF, 0);

    // The slow part of the C test, which is cheap here.
    // set DEC 12 hour mode
    t.cmos_write(RTC_REG_B, REG_B_DM);

    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_HOURS_ALARM, 0x82);
    t.cmos_write(RTC_HOURS, 3);
    t.cmos_write(RTC_MINUTES, 0);
    t.cmos_write(RTC_SECONDS, 0);
    t.cmos_read(RTC_REG_C);
    t.cmos_write(RTC_REG_A, 0x26);

    // Check that alarm triggers.
    t.clock_step(3600 * 11 * NANOSECONDS_PER_SECOND);
    assert_eq!(t.cmos_read(RTC_HOURS), 0x82);
    assert_ne!(t.cmos_read(RTC_REG_C) & REG_C_AF, 0);

    // Same as above, with inverted HOURS and HOURS_ALARM.
    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_HOURS_ALARM, 2);
    t.cmos_write(RTC_HOURS, 3);
    t.cmos_write(RTC_MINUTES, 0);
    t.cmos_write(RTC_SECONDS, 0);
    t.cmos_read(RTC_REG_C);
    t.cmos_write(RTC_REG_A, 0x26);

    // Check that alarm does not trigger if hours differ only by AM/PM.
    t.clock_step(3600 * 11 * NANOSECONDS_PER_SECOND);
    assert_eq!(t.cmos_read(RTC_HOURS), 0x82);
    assert_eq!(t.cmos_read(RTC_REG_C) & REG_C_AF, 0);
}

#[test]
fn basic_24h_dec() {
    let t = Rtc::new();
    // set decimal 24 hour mode
    t.set_time(REG_B_24H | REG_B_DM, 9, 59, 0);
    t.clock_step(1_000_000_000);
    t.assert_time(9, 59, 1);
    t.clock_step(59_000_000_000);
    t.assert_time(10, 0, 0);

    // test BCD wraparound
    t.set_time(REG_B_24H | REG_B_DM, 9, 59, 0);
    t.clock_step(60_000_000_000);
    t.assert_time(10, 0, 0);

    t.set_time(REG_B_24H | REG_B_DM, 23, 59, 0);
    t.clock_step(60_000_000_000);
    t.assert_time(0, 0, 0);
}

#[test]
fn basic_24h_bcd() {
    let t = Rtc::new();
    // set BCD 24 hour mode
    t.set_time(REG_B_24H, 0x09, 0x59, 0x00);
    t.clock_step(1_000_000_000);
    t.assert_time(0x09, 0x59, 0x01);
    t.clock_step(59_000_000_000);
    t.assert_time(0x10, 0x00, 0x00);

    // test BCD wraparound
    t.set_time(REG_B_24H, 0x09, 0x59, 0x00);
    t.clock_step(60_000_000_000);
    t.assert_time(0x10, 0x00, 0x00);

    t.set_time(REG_B_24H, 0x23, 0x59, 0x00);
    t.clock_step(60_000_000_000);
    t.assert_time(0x00, 0x00, 0x00);
}

#[test]
fn basic_12h_dec() {
    let t = Rtc::new();
    // set decimal 12 hour mode
    t.set_time(REG_B_DM, 0x81, 59, 0);
    t.clock_step(1_000_000_000);
    t.assert_time(0x81, 59, 1);
    t.clock_step(59_000_000_000);
    t.assert_time(0x82, 0, 0);

    // 12 PM -> 1 PM
    t.set_time(REG_B_DM, 0x8c, 59, 59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x81, 0, 0);

    // 12 AM -> 1 AM
    t.set_time(REG_B_DM, 0x0c, 59, 59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x01, 0, 0);

    // 11 AM -> 12 PM
    t.set_time(REG_B_DM, 0x0b, 59, 59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x8c, 0, 0);

    // 11 PM -> 12 AM
    t.set_time(REG_B_DM, 0x8b, 59, 59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x0c, 0, 0);
}

#[test]
fn basic_12h_bcd() {
    let t = Rtc::new();
    // set BCD 12 hour mode
    t.set_time(0, 0x81, 0x59, 0x00);
    t.clock_step(1_000_000_000);
    t.assert_time(0x81, 0x59, 0x01);
    t.clock_step(59_000_000_000);
    t.assert_time(0x82, 0x00, 0x00);

    // test BCD wraparound
    t.set_time(0, 0x09, 0x59, 0x59);
    t.clock_step(60_000_000_000);
    t.assert_time(0x10, 0x00, 0x59);

    // 12 AM -> 1 AM
    t.set_time(0, 0x12, 0x59, 0x59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x01, 0x00, 0x00);

    // 12 PM -> 1 PM
    t.set_time(0, 0x92, 0x59, 0x59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x81, 0x00, 0x00);

    // 11 AM -> 12 PM
    t.set_time(0, 0x11, 0x59, 0x59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x92, 0x00, 0x00);

    // 11 PM -> 12 AM
    t.set_time(0, 0x91, 0x59, 0x59);
    t.clock_step(1_000_000_000);
    t.assert_time(0x12, 0x00, 0x00);
}

fn assert_set_year(t: &Rtc, year: u8, century: u8) {
    assert_eq!(t.cmos_read(RTC_HOURS), 0x02);
    assert_eq!(t.cmos_read(RTC_MINUTES), 0x04);
    assert!(t.cmos_read(RTC_SECONDS) >= 0x58);
    assert_eq!(t.cmos_read(RTC_DAY_OF_MONTH), 0x02);
    assert_eq!(t.cmos_read(RTC_MONTH), 0x02);
    assert_eq!(t.cmos_read(RTC_YEAR), year);
    assert_eq!(t.cmos_read(RTC_CENTURY), century);
}

#[test]
fn set_year_20xx() {
    let t = Rtc::new();
    // Set BCD mode
    t.cmos_write(RTC_REG_B, REG_B_24H);
    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_YEAR, 0x11);
    t.cmos_write(RTC_CENTURY, 0x20);
    t.cmos_write(RTC_MONTH, 0x02);
    t.cmos_write(RTC_DAY_OF_MONTH, 0x02);
    t.cmos_write(RTC_HOURS, 0x02);
    t.cmos_write(RTC_MINUTES, 0x04);
    t.cmos_write(RTC_SECONDS, 0x58);
    t.cmos_write(RTC_REG_A, 0x26);
    assert_set_year(&t, 0x11, 0x20);

    // Set a date in 2080 to ensure there is no year-2038 overflow.
    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_YEAR, 0x80);
    t.cmos_write(RTC_REG_A, 0x26);
    assert_set_year(&t, 0x80, 0x20);

    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_YEAR, 0x11);
    t.cmos_write(RTC_REG_A, 0x26);
    assert_set_year(&t, 0x11, 0x20);
}

#[test]
fn set_year_1980() {
    let t = Rtc::new();
    // Set BCD mode
    t.cmos_write(RTC_REG_B, REG_B_24H);
    t.cmos_write(RTC_REG_A, 0x76);
    t.cmos_write(RTC_YEAR, 0x80);
    t.cmos_write(RTC_CENTURY, 0x19);
    t.cmos_write(RTC_MONTH, 0x02);
    t.cmos_write(RTC_DAY_OF_MONTH, 0x02);
    t.cmos_write(RTC_HOURS, 0x02);
    t.cmos_write(RTC_MINUTES, 0x04);
    t.cmos_write(RTC_SECONDS, 0x58);
    t.cmos_write(RTC_REG_A, 0x26);
    assert_set_year(&t, 0x80, 0x19);
}

#[test]
fn register_b_set_flag() {
    let t = Rtc::new();
    if t.cmos_read(RTC_REG_A) & REG_A_UIP != 0 {
        t.clock_step(UIP_HOLD_LENGTH + NANOSECONDS_PER_SECOND / 5);
    }
    assert_eq!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);

    // Enable binary-coded decimal (BCD) mode and SET flag in Register B
    t.cmos_write(RTC_REG_B, REG_B_24H | REG_B_SET);

    t.set_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // Since SET flag is still enabled, time does not advance.
    t.clock_step(1_000_000_000);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // Disable SET flag in Register B
    t.cmos_write(RTC_REG_B, t.cmos_read(RTC_REG_B) & !REG_B_SET);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // Since SET flag is disabled, the clock now advances.
    t.clock_step(1_000_000_000);
    t.assert_datetime_bcd(0x02, 0x04, 0x59, 0x02, 0x02, 0x2011);
}

#[test]
fn divider_reset() {
    let t = Rtc::new();
    // Enable binary-coded decimal (BCD) mode in Register B
    t.cmos_write(RTC_REG_B, REG_B_24H);

    // Enter divider reset
    t.cmos_write(RTC_REG_A, 0x76);
    t.set_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // Since divider reset flag is still enabled, these are equality checks.
    t.clock_step(1_000_000_000);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // The first update ends 500 ms after divider reset
    t.cmos_write(RTC_REG_A, 0x26);
    t.clock_step(500_000_000 - UIP_HOLD_LENGTH - 1);
    assert_eq!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);
    t.assert_datetime_bcd(0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    t.clock_step(1);
    assert_ne!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);
    t.clock_step(UIP_HOLD_LENGTH);
    assert_eq!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);

    t.assert_datetime_bcd(0x02, 0x04, 0x59, 0x02, 0x02, 0x2011);
}

#[test]
fn uip_stuck() {
    let t = Rtc::new();
    t.set_datetime(REG_B_24H, 0x02, 0x04, 0x58, 0x02, 0x02, 0x2011);

    // The first update ends 500 ms after divider reset
    t.cmos_read(RTC_REG_C);
    t.clock_step(500_000_000);
    assert_eq!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);
    t.assert_datetime_bcd(0x02, 0x04, 0x59, 0x02, 0x02, 0x2011);

    // UF is now set.
    t.cmos_write(RTC_HOURS_ALARM, 0x02);
    t.cmos_write(RTC_MINUTES_ALARM, 0xc0);
    t.cmos_write(RTC_SECONDS_ALARM, 0xc0);

    // Because the alarm will fire soon, reading register A will latch UIP.
    t.clock_step(1_000_000_000 - UIP_HOLD_LENGTH / 2);
    assert_ne!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);

    // Move the alarm far away. This must not cause UIP to remain stuck!
    t.cmos_write(RTC_HOURS_ALARM, 0x03);
    t.clock_step(UIP_HOLD_LENGTH);
    assert_eq!(t.cmos_read(RTC_REG_A) & REG_A_UIP, 0);
}

/// Success if no crash or abort.
#[test]
fn fuzz_registers() {
    let t = Rtc::new();
    let mut seed: u32 = 0x1234_5678;
    let mut rand = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    for _ in 0..1000 {
        let reg = (rand() % 16) as usize;
        let val = (rand() % 256) as u8;
        t.cmos_write(reg, val);
        t.cmos_read(reg);
    }
    // Time must still be able to move with whatever state the fuzzing left.
    t.clock_step(10 * NANOSECONDS_PER_SECOND);
}

const RTC_PERIOD_CODE1: u8 = 13; // 8 Hz
const RTC_PERIOD_CODE2: u8 = 15; // 2 Hz
const RTC_PERIOD_TEST_NR: u64 = 50;

fn periodic_interrupt(t: &Rtc) {
    // disable all interrupts.
    t.cmos_write(RTC_REG_B, t.cmos_read(RTC_REG_B) & !(REG_B_PIE | REG_B_AIE | REG_B_UIE));
    t.cmos_write(RTC_REG_A, RTC_PERIOD_CODE1);
    // enable periodic interrupt after properly configure the period.
    t.cmos_write(RTC_REG_B, t.cmos_read(RTC_REG_B) | REG_B_PIE);

    let start_time = t.clock_step_next();
    let mut real_time = start_time;

    for _ in 0..RTC_PERIOD_TEST_NR {
        t.cmos_write(RTC_REG_A, RTC_PERIOD_CODE1);
        real_time = t.wait_periodic_interrupt(real_time);
        t.cmos_write(RTC_REG_A, RTC_PERIOD_CODE2);
        real_time = t.wait_periodic_interrupt(real_time);
    }

    let period_clocks = (u64::from(periodic_period_to_clock(i32::from(RTC_PERIOD_CODE1)))
        + u64::from(periodic_period_to_clock(i32::from(RTC_PERIOD_CODE2))))
        * RTC_PERIOD_TEST_NR;
    let period_time = periodic_clock_to_ns(period_clocks as i64);

    let real_time = real_time - start_time;
    assert!((real_time - period_time).abs() <= NANOSECONDS_PER_SECOND / 2);
}

#[test]
fn periodic_timer() {
    periodic_interrupt(&Rtc::new());
}

/// With the slew policy and every tick reaching the CPU, the timing is the same as discard.
#[test]
fn periodic_timer_slew() {
    let t = Rtc::with_policy(LostTickPolicy::Slew);
    let delivered = IrqDelivered::default();
    t.rtc.set_irq_delivered(delivered.clone());
    let level = t.irq.clone();
    t.rtc.connect_irq(IrqLine::from_fn(move |l| {
        level.store(l, Ordering::SeqCst);
        delivered.report(l);
    }));
    periodic_interrupt(&t);
}

/// Ticks that were not delivered are coalesced and injected again when REG_C is read.
#[test]
fn periodic_timer_slew_reinjects() {
    let t = Rtc::with_policy(LostTickPolicy::Slew);
    t.cmos_write(RTC_REG_A, RTC_PERIOD_CODE2);
    t.cmos_write(RTC_REG_B, t.cmos_read(RTC_REG_B) | REG_B_PIE);
    // Three ticks at 2 Hz, none of them reported as delivered.
    t.clock_step(3 * NANOSECONDS_PER_SECOND / 2 + 1_000_000);
    assert!(t.get_irq());
    assert_ne!(t.cmos_read(RTC_REG_C) & REG_C_PF, 0);
    // The read lowered the line and injected a coalesced tick right away.
    assert!(t.get_irq());
    assert_eq!(t.cmos_read(RTC_REG_C) & (REG_C_IRQF | REG_C_PF), REG_C_IRQF | REG_C_PF);

    t.rtc.reset_reinjection();
    t.cmos_read(RTC_REG_C);
    assert!(!t.get_irq());
}

#[test]
fn date_property_and_reset() {
    let t = Rtc::new();
    assert_eq!(mktimegm(&t.rtc.date()), START);
    t.clock_step(5 * NANOSECONDS_PER_SECOND);
    assert_eq!(mktimegm(&t.rtc.date()), START + 5);

    t.rtc.set_date(SystemTime::UNIX_EPOCH + Duration::from_secs(951_782_400));
    assert_eq!(t.rtc.date(), gmtime(951_782_400));
    // base_year 2000 means the century byte holds the real century.
    assert_eq!(t.rtc.get_cmos_data(RTC_CENTURY), 0x20);

    t.rtc.notify_suspend();
    t.rtc.set_cmos_data(RTC_REG_B, REG_B_24H | REG_B_PIE | REG_B_AIE);
    t.rtc.reset();
    assert_eq!(t.rtc.get_cmos_data(0x0f), 0xfe);
    assert_eq!(t.rtc.get_cmos_data(RTC_REG_B), REG_B_24H);
    assert!(!t.get_irq());
}
