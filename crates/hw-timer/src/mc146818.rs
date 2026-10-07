// SPDX-License-Identifier: GPL-2.0-or-later

//! The MC146818 real time clock and its CMOS RAM, from hw/rtc/mc146818rtc.c.
//!
//! The guest date is kept as `base_rtc` seconds plus the time the RTC clock has run since
//! `last_update`, and the CMOS time registers are refreshed from that when read. Two timers
//! drive the interrupts: the periodic timer (with the lost tick slew policy) and the update
//! ended timer, which also handles the alarm.
//!
//! [`Mc146818VmState`] is the device as QEMU's `mc146818rtc` VMState section carries it. Not
//! ported yet: trace points, QOM registration, the coalesced PIO index subregion and the ACPI
//! `_CRS` hook.

use std::fmt;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use ruvm_base::{ClockType, Error, Result};
use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer, muldiv64};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_MC146818_RTC`.
pub const TYPE_MC146818_RTC: &str = "mc146818rtc";

pub const RTC_SECONDS: usize = 0;
pub const RTC_SECONDS_ALARM: usize = 1;
pub const RTC_MINUTES: usize = 2;
pub const RTC_MINUTES_ALARM: usize = 3;
pub const RTC_HOURS: usize = 4;
pub const RTC_HOURS_ALARM: usize = 5;
pub const RTC_ALARM_DONT_CARE: u8 = 0xc0;

pub const RTC_DAY_OF_WEEK: usize = 6;
pub const RTC_DAY_OF_MONTH: usize = 7;
pub const RTC_MONTH: usize = 8;
pub const RTC_YEAR: usize = 9;

pub const RTC_REG_A: usize = 10;
pub const RTC_REG_B: usize = 11;
pub const RTC_REG_C: usize = 12;
pub const RTC_REG_D: usize = 13;

// PC cmos mappings
pub const RTC_CENTURY: usize = 0x32;
pub const RTC_IBM_PS2_CENTURY_BYTE: usize = 0x37;

pub const REG_A_UIP: u8 = 0x80;

pub const REG_B_SET: u8 = 0x80;
pub const REG_B_PIE: u8 = 0x40;
pub const REG_B_AIE: u8 = 0x20;
pub const REG_B_UIE: u8 = 0x10;
pub const REG_B_SQWE: u8 = 0x08;
pub const REG_B_DM: u8 = 0x04;
pub const REG_B_24H: u8 = 0x02;

pub const REG_C_UF: u8 = 0x10;
pub const REG_C_IRQF: u8 = 0x80;
pub const REG_C_PF: u8 = 0x40;
pub const REG_C_AF: u8 = 0x20;
pub const REG_C_MASK: u8 = 0x70;

pub const RTC_CLOCK_RATE: u32 = 32768;
pub const RTC_REINJECT_ON_ACK_COUNT: u16 = 20;
pub const UIP_HOLD_LENGTH: i64 = 8 * NANOSECONDS_PER_SECOND / 32768;

pub const RTC_ISA_BASE: u16 = 0x70;
pub const RTC_ISA_IRQ: u8 = 8;
const ISA_NUM_IRQS: u8 = 16;

const SEC_PER_MIN: i32 = 60;
const MIN_PER_HOUR: i32 = 60;
const HOUR_PER_DAY: i32 = 24;
const SEC_PER_DAY: i32 = 86400;

/// `get_max_clock_jump()`: how far the RTC clock may have moved past the next periodic
/// interrupt before `rtc_post_load()` reprograms the periodic timer.
const MAX_CLOCK_JUMP: u64 = 60 * NANOSECONDS_PER_SECOND as u64;

const NS_U32: u32 = NANOSECONDS_PER_SECOND as u32;
const NS_U64: u64 = NANOSECONDS_PER_SECOND as u64;

/// `periodic_period_to_clock()`: the period of a rate code in 32 kHz cycles.
pub fn periodic_period_to_clock(period_code: i32) -> u32 {
    if period_code == 0 {
        return 0;
    }
    let code = if period_code <= 2 { period_code + 7 } else { period_code };
    1 << (code - 1)
}

/// `periodic_clock_to_ns()`.
pub fn periodic_clock_to_ns(clocks: i64) -> i64 {
    muldiv64(clocks as u64, NS_U32, RTC_CLOCK_RATE) as i64
}

/// `LostTickPolicy`. The RTC accepts `Discard` and `Slew`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LostTickPolicy {
    #[default]
    Discard,
    Delay,
    Slew,
}

/// The qdev properties of the device.
#[derive(Clone, Copy, Debug)]
pub struct Mc146818Props {
    pub base_year: i32,
    pub iobase: u16,
    pub irq: u8,
    pub lost_tick_policy: LostTickPolicy,
}

impl Default for Mc146818Props {
    fn default() -> Self {
        Mc146818Props {
            base_year: 1980,
            iobase: RTC_ISA_BASE,
            irq: RTC_ISA_IRQ,
            lost_tick_policy: LostTickPolicy::Discard,
        }
    }
}

/// The counter of hw/intc/kvm_irqcount.c. Interrupt controllers call [`IrqDelivered::report`]
/// when a raised line reaches a CPU, the slew policy uses it to count coalesced ticks.
#[derive(Clone, Debug, Default)]
pub struct IrqDelivered(Arc<AtomicI32>);

impl IrqDelivered {
    /// `kvm_report_irq_delivered()`.
    pub fn report(&self, delivered: i32) {
        self.0.fetch_add(delivered, Ordering::SeqCst);
    }

    /// `kvm_reset_irq_delivered()`.
    pub fn reset(&self) {
        self.0.store(0, Ordering::SeqCst);
    }

    /// `kvm_get_irq_delivered()`.
    pub fn get(&self) -> i32 {
        self.0.load(Ordering::SeqCst)
    }
}

/// The fields of `struct tm` the RTC uses. `year` counts from 1900 and `mon` from 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tm {
    pub sec: i32,
    pub min: i32,
    pub hour: i32,
    pub mday: i32,
    pub mon: i32,
    pub year: i32,
    pub wday: i32,
}

/// `mktimegm()` from util/cutils.c.
pub fn mktimegm(tm: &Tm) -> i64 {
    let mut y = i64::from(tm.year) + 1900;
    let mut m = i64::from(tm.mon) + 1;
    let d = i64::from(tm.mday);
    if m < 3 {
        m += 12;
        y -= 1;
    }
    let mut t = 86400 * (d + (153 * m - 457) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 719_469);
    t += 3600 * i64::from(tm.hour) + 60 * i64::from(tm.min) + i64::from(tm.sec);
    t
}

/// `gmtime_r()`.
pub fn gmtime(t: i64) -> Tm {
    let days = t.div_euclid(86400);
    let secs = t.rem_euclid(86400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let mday = doy - (153 * mp + 2) / 5 + 1;
    let mon = if mp < 10 { mp + 2 } else { mp - 10 };
    let year = yoe + era * 400 + i64::from(mon < 2);
    Tm {
        sec: (secs % 60) as i32,
        min: (secs / 60 % 60) as i32,
        hour: (secs / 3600) as i32,
        mday: mday as i32,
        mon: mon as i32,
        year: (year - 1900) as i32,
        wday: (days + 4).rem_euclid(7) as i32,
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

type Hook<T> = Arc<dyn Fn(T) + Send + Sync>;

/// The `mc146818rtc` VMState section, version 3. The timers and times are on the RTC clock
/// (`-rtc clock=`, the host clock by default), in nanoseconds; a timer is -1 when not armed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mc146818VmState {
    pub cmos_data: [u8; 128],
    pub cmos_index: u8,
    pub periodic_timer: i64,
    pub next_periodic_time: i64,
    pub irq_coalesced: u32,
    pub period: u32,
    pub base_rtc: u64,
    pub last_update: u64,
    pub offset: i64,
    pub update_timer: i64,
    pub next_alarm_time: u64,
    /// Subsection `mc146818rtc/irq_reinject_on_ack_count`, sent when it is not 0.
    pub irq_reinject_on_ack_count: u16,
}

impl Default for Mc146818VmState {
    fn default() -> Self {
        Mc146818VmState {
            cmos_data: [0; 128],
            cmos_index: 0,
            periodic_timer: -1,
            next_periodic_time: 0,
            irq_coalesced: 0,
            period: 0,
            base_rtc: 0,
            last_update: 0,
            offset: 0,
            update_timer: -1,
            next_alarm_time: 0,
            irq_reinject_on_ack_count: 0,
        }
    }
}

/// `timer_put()`: arms `timer` for `expire`, or deletes it for -1.
fn timer_put(timer: &Timer, expire: i64) {
    if expire == -1 {
        timer.del();
    } else {
        timer.modify(expire);
    }
}

/// `MC146818RtcState`.
struct RtcState {
    cmos_data: [u8; 128],
    cmos_index: usize,
    isairq: u8,
    io_base: u16,
    base_year: i32,
    base_rtc: u64,
    last_update: u64,
    offset: i64,
    irq: IrqPin,
    irq_delivered: IrqDelivered,
    clock: Arc<Clock>,
    // periodic timer
    periodic_timer: Timer,
    next_periodic_time: i64,
    // update-ended timer
    update_timer: Timer,
    next_alarm_time: u64,
    irq_reinject_on_ack_count: u16,
    irq_coalesced: u32,
    period: u32,
    coalesced_timer: Option<Timer>,
    lost_tick_policy: LostTickPolicy,
    rtc_change: Option<Hook<i64>>,
    wakeup: Option<Hook<()>>,
}

fn lock(s: &Mutex<RtcState>) -> MutexGuard<'_, RtcState> {
    s.lock().unwrap_or_else(PoisonError::into_inner)
}

fn timer_callback(
    weak: &Weak<Mutex<RtcState>>,
    f: fn(&mut RtcState),
) -> impl Fn() + Send + Sync + 'static {
    let weak = weak.clone();
    move || {
        if let Some(s) = weak.upgrade() {
            f(&mut lock(&s));
        }
    }
}

impl RtcState {
    fn now(&self) -> i64 {
        self.clock.get_ns()
    }

    /// `rtc_running()`.
    fn running(&self) -> bool {
        self.cmos_data[RTC_REG_B] & REG_B_SET == 0 && (self.cmos_data[RTC_REG_A] & 0x70) <= 0x20
    }

    /// `get_rtc_ns_since_last_update()`. This does not include the `base_rtc` seconds, which
    /// does not matter if the caller only needs the nanoseconds part.
    fn ns_since_last_update(&self) -> u64 {
        (self.now() as u64).wrapping_sub(self.last_update).wrapping_add(self.offset as u64)
    }

    /// `rtc_coalesced_timer_update()`.
    fn coalesced_timer_update(&self) {
        let Some(timer) = &self.coalesced_timer else {
            return;
        };
        if self.irq_coalesced == 0 {
            timer.del();
        } else {
            // divide each RTC interval to 2 - 8 smaller intervals
            let c = self.irq_coalesced.min(7) + 1;
            let next_clock = self.now() + periodic_clock_to_ns(i64::from(self.period / c));
            timer.modify(next_clock);
        }
    }

    /// `rtc_policy_slew_deliver_irq()`.
    fn slew_deliver_irq(&self) -> bool {
        self.irq_delivered.reset();
        self.irq.raise();
        self.irq_delivered.get() != 0
    }

    /// `rtc_coalesced_timer()`.
    fn coalesced_timer(&mut self) {
        if self.irq_coalesced != 0 {
            self.cmos_data[RTC_REG_C] |= 0xc0;
            if self.slew_deliver_irq() {
                self.irq_coalesced -= 1;
            }
        }
        self.coalesced_timer_update();
    }

    /// `rtc_periodic_clock_ticks()`.
    fn periodic_clock_ticks(&self) -> u32 {
        if self.cmos_data[RTC_REG_B] & REG_B_PIE == 0 {
            return 0;
        }
        periodic_period_to_clock(i32::from(self.cmos_data[RTC_REG_A] & 0x0f))
    }

    /// `periodic_timer_update()`. `period_change` says the update is only due to a new period.
    fn periodic_timer_update(&mut self, current_time: i64, old_period: u32, period_change: bool) {
        let period = self.periodic_clock_ticks();
        self.period = period;

        if period == 0 {
            self.irq_coalesced = 0;
            self.periodic_timer.del();
            return;
        }

        // compute 32 khz clock
        let cur_clock = muldiv64(current_time as u64, RTC_CLOCK_RATE, NS_U32) as i64;
        let mut lost_clock: i64 = 0;

        // If the update is due to a period change, count the clock since the last interrupt.
        if old_period != 0 && period_change {
            let next_periodic_clock =
                muldiv64(self.next_periodic_time as u64, RTC_CLOCK_RATE, NS_U32) as i64;
            let last_periodic_clock = next_periodic_clock - i64::from(old_period);
            lost_clock = cur_clock - last_periodic_clock;
            assert!(lost_clock >= 0);
        }

        // irq_coalesced changes if interrupts were lost, or when the period changes: the OS
        // treats delayed ticks as the new period, so the missing ticks are scaled to it and the
        // leftover goes back into lost_clock.
        if self.lost_tick_policy == LostTickPolicy::Slew {
            let old_irq_coalesced = self.irq_coalesced;
            lost_clock += i64::from(old_irq_coalesced.wrapping_mul(old_period));
            self.irq_coalesced = (lost_clock / i64::from(self.period)) as u32;
            lost_clock %= i64::from(self.period);
            if old_irq_coalesced != self.irq_coalesced || old_period != self.period {
                self.coalesced_timer_update();
            }
        } else {
            // No way to make up for the lost interrupts, but time must move on anyway.
            lost_clock = lost_clock.min(i64::from(period));
        }

        assert!(lost_clock >= 0 && lost_clock <= i64::from(period));

        let next_irq_clock = cur_clock + i64::from(period) - lost_clock;
        self.next_periodic_time = periodic_clock_to_ns(next_irq_clock) + 1;
        self.periodic_timer.modify(self.next_periodic_time);
    }

    /// `rtc_periodic_timer()`.
    fn periodic_timer(&mut self) {
        let (time, period) = (self.next_periodic_time, self.period);
        self.periodic_timer_update(time, period, false);
        self.cmos_data[RTC_REG_C] |= REG_C_PF;
        if self.cmos_data[RTC_REG_B] & REG_B_PIE != 0 {
            self.cmos_data[RTC_REG_C] |= REG_C_IRQF;
            if self.lost_tick_policy == LostTickPolicy::Slew {
                if self.irq_reinject_on_ack_count >= RTC_REINJECT_ON_ACK_COUNT {
                    self.irq_reinject_on_ack_count = 0;
                }
                if !self.slew_deliver_irq() {
                    self.irq_coalesced = self.irq_coalesced.wrapping_add(1);
                    self.coalesced_timer_update();
                }
            } else {
                self.irq.raise();
            }
        }
    }

    /// `check_update_timer()`: arms the update-ended timer.
    fn check_update_timer(&mut self) {
        // From the data sheet: "Holding the dividers in reset prevents interrupts from
        // operating, while setting the SET bit allows them to occur."
        if self.cmos_data[RTC_REG_A] & 0x60 == 0x60 {
            assert!(self.cmos_data[RTC_REG_A] & REG_A_UIP == 0);
            self.update_timer.del();
            return;
        }

        let guest_nsec = self.ns_since_last_update() % NS_U64;
        let mut next_update_time =
            (self.now() as u64).wrapping_add(NS_U64).wrapping_sub(guest_nsec);

        // Compute time of next alarm. One second is already accounted for in next_update_time.
        let next_alarm_sec = self.get_next_alarm();
        self.next_alarm_time = next_update_time
            .wrapping_add(((i64::from(next_alarm_sec) - 1) * NANOSECONDS_PER_SECOND) as u64);

        // If update_in_progress latched the UIP bit, keep the timer programmed to the next
        // second so that UIP is cleared. Otherwise, if UF is already set, we might optimize.
        if self.cmos_data[RTC_REG_A] & REG_A_UIP == 0 && self.cmos_data[RTC_REG_C] & REG_C_UF != 0 {
            // If AF cannot change (it is set already, or SET=1 and the time is not updated),
            // nothing to do.
            if self.cmos_data[RTC_REG_B] & REG_B_SET != 0
                || self.cmos_data[RTC_REG_C] & REG_C_AF != 0
            {
                self.update_timer.del();
                return;
            }
            // UF is set, but AF is clear. Program the timer to target the alarm time.
            next_update_time = self.next_alarm_time;
        }
        if next_update_time != self.update_timer.expire_time().unwrap_or(-1) as u64 {
            self.update_timer.modify(next_update_time as i64);
        }
    }

    /// `convert_hour()`.
    fn convert_hour(&self, mut hour: u8) -> u8 {
        if self.cmos_data[RTC_REG_B] & REG_B_24H == 0 {
            hour %= 12;
            if self.cmos_data[RTC_HOURS] & 0x80 != 0 {
                hour += 12;
            }
        }
        hour
    }

    /// `get_next_alarm()`: seconds from now to the next alarm.
    fn get_next_alarm(&mut self) -> i32 {
        self.update_time();

        let mut alarm_sec = self.bcd_to_int(self.cmos_data[RTC_SECONDS_ALARM]);
        let mut alarm_min = self.bcd_to_int(self.cmos_data[RTC_MINUTES_ALARM]);
        let mut alarm_hour = self.bcd_to_int(self.cmos_data[RTC_HOURS_ALARM]);
        if alarm_hour != -1 {
            alarm_hour = i32::from(self.convert_hour(alarm_hour as u8));
        }

        let cur_sec = self.bcd_to_int(self.cmos_data[RTC_SECONDS]);
        let cur_min = self.bcd_to_int(self.cmos_data[RTC_MINUTES]);
        let cur_hour = self.bcd_to_int(self.cmos_data[RTC_HOURS]);
        let cur_hour = i32::from(self.convert_hour(cur_hour as u8));

        if alarm_hour == -1 {
            alarm_hour = cur_hour;
            if alarm_min == -1 {
                alarm_min = cur_min;
                if alarm_sec == -1 {
                    alarm_sec = cur_sec + 1;
                } else if cur_sec > alarm_sec {
                    alarm_min += 1;
                }
            } else if cur_min == alarm_min {
                if alarm_sec == -1 {
                    alarm_sec = cur_sec + 1;
                } else if cur_sec > alarm_sec {
                    alarm_hour += 1;
                }
                if alarm_sec == SEC_PER_MIN {
                    // wrap to next hour, minutes is not in don't care mode
                    alarm_sec = 0;
                    alarm_hour += 1;
                }
            } else if cur_min > alarm_min {
                alarm_hour += 1;
            }
        } else if cur_hour == alarm_hour {
            if alarm_min == -1 {
                alarm_min = cur_min;
                if alarm_sec == -1 {
                    alarm_sec = cur_sec + 1;
                } else if cur_sec > alarm_sec {
                    alarm_min += 1;
                }
                if alarm_sec == SEC_PER_MIN {
                    alarm_sec = 0;
                    alarm_min += 1;
                }
                // wrap to next day, hour is not in don't care mode
                alarm_min %= MIN_PER_HOUR;
            } else if cur_min == alarm_min {
                if alarm_sec == -1 {
                    alarm_sec = cur_sec + 1;
                }
                // wrap to next day, hours+minutes not in don't care mode
                alarm_sec %= SEC_PER_MIN;
            }
        }

        // values that are still don't care fire at the next min/sec
        if alarm_min == -1 {
            alarm_min = 0;
        }
        if alarm_sec == -1 {
            alarm_sec = 0;
        }

        // keep values in range
        if alarm_sec == SEC_PER_MIN {
            alarm_sec = 0;
            alarm_min += 1;
        }
        if alarm_min == MIN_PER_HOUR {
            alarm_min = 0;
            alarm_hour += 1;
        }
        alarm_hour %= HOUR_PER_DAY;

        let hour = alarm_hour - cur_hour;
        let min = hour * MIN_PER_HOUR + alarm_min - cur_min;
        let sec = min * SEC_PER_MIN + alarm_sec - cur_sec;
        if sec <= 0 { sec + SEC_PER_DAY } else { sec }
    }

    /// `rtc_update_timer()`.
    fn update_timer(&mut self) {
        assert!(self.cmos_data[RTC_REG_A] & 0x60 != 0x60);
        let mut irqs = REG_C_UF;

        // UIP might have been latched, update time and clear it.
        self.update_time();
        self.cmos_data[RTC_REG_A] &= !REG_A_UIP;

        if self.now() as u64 >= self.next_alarm_time {
            irqs |= REG_C_AF;
            if self.cmos_data[RTC_REG_B] & REG_B_AIE != 0 {
                if let Some(wakeup) = &self.wakeup {
                    wakeup(());
                }
            }
        }

        let new_irqs = irqs & !self.cmos_data[RTC_REG_C];
        self.cmos_data[RTC_REG_C] |= irqs;
        if new_irqs & self.cmos_data[RTC_REG_B] != 0 {
            self.cmos_data[RTC_REG_C] |= REG_C_IRQF;
            self.irq.raise();
        }
        self.check_update_timer();
    }

    /// `cmos_ioport_write()`.
    fn ioport_write(&mut self, addr: u64, data: u8) {
        if addr & 1 == 0 {
            self.cmos_index = usize::from(data & 0x7f);
            return;
        }
        let mut data = data;
        match self.cmos_index {
            RTC_SECONDS_ALARM | RTC_MINUTES_ALARM | RTC_HOURS_ALARM => {
                self.cmos_data[self.cmos_index] = data;
                self.check_update_timer();
            }
            RTC_IBM_PS2_CENTURY_BYTE
            | RTC_CENTURY
            | RTC_SECONDS
            | RTC_MINUTES
            | RTC_HOURS
            | RTC_DAY_OF_WEEK
            | RTC_DAY_OF_MONTH
            | RTC_MONTH
            | RTC_YEAR => {
                if self.cmos_index == RTC_IBM_PS2_CENTURY_BYTE {
                    self.cmos_index = RTC_CENTURY;
                }
                self.cmos_data[self.cmos_index] = data;
                // if in set mode, do not update the time
                if self.running() {
                    self.set_time();
                    self.check_update_timer();
                }
            }
            RTC_REG_A => {
                let update_periodic_timer = (self.cmos_data[RTC_REG_A] ^ data) & 0x0f != 0;
                let old_period = self.periodic_clock_ticks();

                if data & 0x60 == 0x60 {
                    if self.running() {
                        self.update_time();
                    }
                    // What happens to UIP when divider reset is enabled is unclear from the
                    // datasheet. Shouldn't matter much though.
                    self.cmos_data[RTC_REG_A] &= !REG_A_UIP;
                } else if self.cmos_data[RTC_REG_A] & 0x60 == 0x60 && (data & 0x70) <= 0x20 {
                    // when the divider reset is removed, the first update cycle begins
                    // one-half second later
                    if self.cmos_data[RTC_REG_B] & REG_B_SET == 0 {
                        self.offset = 500_000_000;
                        self.set_time();
                    }
                    self.cmos_data[RTC_REG_A] &= !REG_A_UIP;
                }
                // UIP bit is read only
                self.cmos_data[RTC_REG_A] =
                    (data & !REG_A_UIP) | (self.cmos_data[RTC_REG_A] & REG_A_UIP);

                if update_periodic_timer {
                    let now = self.now();
                    self.periodic_timer_update(now, old_period, true);
                }
                self.check_update_timer();
            }
            RTC_REG_B => {
                let update_periodic_timer = (self.cmos_data[RTC_REG_B] ^ data) & REG_B_PIE != 0;
                let old_period = self.periodic_clock_ticks();

                if data & REG_B_SET != 0 {
                    // update cmos to when the rtc was stopping
                    if self.running() {
                        self.update_time();
                    }
                    // set mode: reset UIP mode
                    self.cmos_data[RTC_REG_A] &= !REG_A_UIP;
                    data &= !REG_B_UIE;
                } else if self.cmos_data[RTC_REG_B] & REG_B_SET != 0
                    && (self.cmos_data[RTC_REG_A] & 0x70) <= 0x20
                {
                    // if disabling set mode, update the time
                    self.offset = (self.ns_since_last_update() % NS_U64) as i64;
                    self.set_time();
                }
                // If an interrupt flag is already set when the interrupt becomes enabled,
                // raise an interrupt immediately.
                if data & self.cmos_data[RTC_REG_C] & REG_C_MASK != 0 {
                    self.cmos_data[RTC_REG_C] |= REG_C_IRQF;
                    self.irq.raise();
                } else {
                    self.cmos_data[RTC_REG_C] &= !REG_C_IRQF;
                    self.irq.lower();
                }
                self.cmos_data[RTC_REG_B] = data;

                if update_periodic_timer {
                    let now = self.now();
                    self.periodic_timer_update(now, old_period, true);
                }
                self.check_update_timer();
            }
            RTC_REG_C | RTC_REG_D => {
                // cannot write to them
            }
            _ => self.cmos_data[self.cmos_index] = data,
        }
    }

    /// `rtc_to_bcd()`.
    fn int_to_bcd(&self, a: i32) -> u8 {
        if self.cmos_data[RTC_REG_B] & REG_B_DM != 0 {
            a as u8
        } else {
            (((a / 10) << 4) | (a % 10)) as u8
        }
    }

    /// `rtc_from_bcd()`. Returns -1 for a don't care value.
    fn bcd_to_int(&self, a: u8) -> i32 {
        if a & 0xc0 == 0xc0 {
            return -1;
        }
        let a = i32::from(a);
        if self.cmos_data[RTC_REG_B] & REG_B_DM != 0 { a } else { ((a >> 4) * 10) + (a & 0x0f) }
    }

    /// `rtc_get_time()`.
    fn get_time(&self) -> Tm {
        let mut hour = self.bcd_to_int(self.cmos_data[RTC_HOURS] & 0x7f);
        if self.cmos_data[RTC_REG_B] & REG_B_24H == 0 {
            hour %= 12;
            if self.cmos_data[RTC_HOURS] & 0x80 != 0 {
                hour += 12;
            }
        }
        Tm {
            sec: self.bcd_to_int(self.cmos_data[RTC_SECONDS]),
            min: self.bcd_to_int(self.cmos_data[RTC_MINUTES]),
            hour,
            wday: self.bcd_to_int(self.cmos_data[RTC_DAY_OF_WEEK]) - 1,
            mday: self.bcd_to_int(self.cmos_data[RTC_DAY_OF_MONTH]),
            mon: self.bcd_to_int(self.cmos_data[RTC_MONTH]) - 1,
            year: self.bcd_to_int(self.cmos_data[RTC_YEAR])
                + self.base_year
                + self.bcd_to_int(self.cmos_data[RTC_CENTURY]) * 100
                - 1900,
        }
    }

    /// `rtc_set_time()`.
    fn set_time(&mut self) {
        let tm = self.get_time();
        let secs = mktimegm(&tm);
        self.base_rtc = secs as u64;
        self.last_update = self.now() as u64;
        if let Some(event) = &self.rtc_change {
            event(secs);
        }
    }

    /// `rtc_set_cmos()`.
    fn set_cmos(&mut self, tm: &Tm) {
        self.cmos_data[RTC_SECONDS] = self.int_to_bcd(tm.sec);
        self.cmos_data[RTC_MINUTES] = self.int_to_bcd(tm.min);
        if self.cmos_data[RTC_REG_B] & REG_B_24H != 0 {
            // 24 hour format
            self.cmos_data[RTC_HOURS] = self.int_to_bcd(tm.hour);
        } else {
            // 12 hour format
            let h = if tm.hour % 12 != 0 { tm.hour % 12 } else { 12 };
            self.cmos_data[RTC_HOURS] = self.int_to_bcd(h);
            if tm.hour >= 12 {
                self.cmos_data[RTC_HOURS] |= 0x80;
            }
        }
        self.cmos_data[RTC_DAY_OF_WEEK] = self.int_to_bcd(tm.wday + 1);
        self.cmos_data[RTC_DAY_OF_MONTH] = self.int_to_bcd(tm.mday);
        self.cmos_data[RTC_MONTH] = self.int_to_bcd(tm.mon + 1);
        let year = tm.year + 1900 - self.base_year;
        self.cmos_data[RTC_YEAR] = self.int_to_bcd(year % 100);
        self.cmos_data[RTC_CENTURY] = self.int_to_bcd(year / 100);
    }

    /// `rtc_update_time()`.
    fn update_time(&mut self) {
        let guest_sec = self.base_rtc.wrapping_add(self.ns_since_last_update() / NS_U64) as i64;
        let ret = gmtime(guest_sec);

        // Is SET flag of Register B disabled?
        if self.cmos_data[RTC_REG_B] & REG_B_SET == 0 {
            self.set_cmos(&ret);
        }
    }

    /// `update_in_progress()`.
    fn update_in_progress(&mut self) -> bool {
        if !self.running() {
            return false;
        }
        if let Some(next_update_time) = self.update_timer.expire_time() {
            // Latch UIP until the timer expires.
            if self.now() >= next_update_time - UIP_HOLD_LENGTH {
                self.cmos_data[RTC_REG_A] |= REG_A_UIP;
                return true;
            }
        }

        // UIP bit will be set at last 244us of every second.
        let guest_nsec = self.ns_since_last_update();
        guest_nsec % NS_U64 >= NS_U64 - UIP_HOLD_LENGTH as u64
    }

    /// `cmos_ioport_read()`.
    fn ioport_read(&mut self, addr: u64) -> u8 {
        if addr & 1 == 0 {
            return 0xff;
        }
        match self.cmos_index {
            RTC_IBM_PS2_CENTURY_BYTE
            | RTC_CENTURY
            | RTC_SECONDS
            | RTC_MINUTES
            | RTC_HOURS
            | RTC_DAY_OF_WEEK
            | RTC_DAY_OF_MONTH
            | RTC_MONTH
            | RTC_YEAR => {
                if self.cmos_index == RTC_IBM_PS2_CENTURY_BYTE {
                    self.cmos_index = RTC_CENTURY;
                }
                // if not in set mode, calibrate cmos before reading
                if self.running() {
                    self.update_time();
                }
                self.cmos_data[self.cmos_index]
            }
            RTC_REG_A => {
                let mut ret = self.cmos_data[RTC_REG_A];
                if self.update_in_progress() {
                    ret |= REG_A_UIP;
                }
                ret
            }
            RTC_REG_C => {
                let ret = self.cmos_data[RTC_REG_C];
                self.irq.lower();
                self.cmos_data[RTC_REG_C] = 0x00;
                if ret & (REG_C_UF | REG_C_AF) != 0 {
                    self.check_update_timer();
                }

                if self.irq_coalesced != 0
                    && self.cmos_data[RTC_REG_B] & REG_B_PIE != 0
                    && self.irq_reinject_on_ack_count < RTC_REINJECT_ON_ACK_COUNT
                {
                    self.irq_reinject_on_ack_count += 1;
                    self.cmos_data[RTC_REG_C] |= REG_C_IRQF | REG_C_PF;
                    if self.slew_deliver_irq() {
                        self.irq_coalesced -= 1;
                    }
                }
                ret
            }
            _ => self.cmos_data[self.cmos_index],
        }
    }

    /// `rtc_set_date_from_host()`, with the host time passed in.
    fn set_date_from_host(&mut self, secs: i64) {
        let tm = gmtime(secs);
        self.base_rtc = mktimegm(&tm) as u64;
        self.last_update = self.now() as u64;
        self.offset = 0;

        // set the CMOS date
        self.set_cmos(&tm);
    }
}

/// The MC146818 RTC, `MC146818RtcState`. Cloning gives another handle to the same device.
#[derive(Clone)]
pub struct Mc146818Rtc {
    state: Arc<Mutex<RtcState>>,
}

impl fmt::Debug for Mc146818Rtc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = lock(&self.state);
        f.debug_struct("Mc146818Rtc")
            .field("io_base", &s.io_base)
            .field("isairq", &s.isairq)
            .field("cmos_index", &s.cmos_index)
            .finish_non_exhaustive()
    }
}

impl Mc146818Rtc {
    /// `rtc_realizefn()`. `rtc_clock` is the clock chosen by `-rtc clock=`, and `host_date` is
    /// the date `qemu_get_timedate()` returns when the clock reads zero. The CMOS date starts
    /// at `host_date` plus the current clock time.
    pub fn new(props: Mc146818Props, rtc_clock: Arc<Clock>, host_date: SystemTime) -> Result<Self> {
        if props.irq >= ISA_NUM_IRQS {
            return Err(Error::generic(format!(
                "Maximum value for \"irq\" is: {}",
                ISA_NUM_IRQS - 1
            )));
        }
        if !matches!(props.lost_tick_policy, LostTickPolicy::Discard | LostTickPolicy::Slew) {
            return Err(Error::generic("Invalid lost tick policy."));
        }

        let state = Arc::new_cyclic(|weak: &Weak<Mutex<RtcState>>| {
            let coalesced_timer = (props.lost_tick_policy == LostTickPolicy::Slew)
                .then(|| rtc_clock.new_timer(timer_callback(weak, RtcState::coalesced_timer)));
            let periodic_timer =
                rtc_clock.new_timer(timer_callback(weak, RtcState::periodic_timer));
            let update_timer = rtc_clock.new_timer(timer_callback(weak, RtcState::update_timer));
            let mut cmos_data = [0; 128];
            cmos_data[RTC_REG_A] = 0x26;
            cmos_data[RTC_REG_B] = 0x02;
            cmos_data[RTC_REG_C] = 0x00;
            cmos_data[RTC_REG_D] = 0x80;
            Mutex::new(RtcState {
                cmos_data,
                cmos_index: 0,
                isairq: props.irq,
                io_base: props.iobase,
                // For historical reasons: the default base year was 2000 on most machine types
                // before the century byte was implemented. The century byte is always 0 (until
                // 2079) for base_year 1980, but correct for base_year 2000.
                base_year: if props.base_year == 2000 { 0 } else { props.base_year },
                base_rtc: 0,
                last_update: 0,
                offset: 0,
                irq: IrqPin::new(),
                irq_delivered: IrqDelivered::default(),
                clock: rtc_clock.clone(),
                periodic_timer,
                next_periodic_time: 0,
                update_timer,
                next_alarm_time: 0,
                irq_reinject_on_ack_count: 0,
                irq_coalesced: 0,
                period: 0,
                coalesced_timer,
                lost_tick_policy: props.lost_tick_policy,
                rtc_change: None,
                wakeup: None,
            })
        });

        let rtc = Mc146818Rtc { state };
        {
            let mut s = rtc.lock();
            let secs = unix_seconds(host_date) + s.clock.get_ms() / 1000;
            s.set_date_from_host(secs);
            s.check_update_timer();
        }
        Ok(rtc)
    }

    fn lock(&self) -> MutexGuard<'_, RtcState> {
        lock(&self.state)
    }

    /// The `iobase` property.
    pub fn iobase(&self) -> u16 {
        self.lock().io_base
    }

    /// The `irq` property, the ISA IRQ the board wires the output to.
    pub fn isairq(&self) -> u8 {
        self.lock().isairq
    }

    /// Connects the interrupt output, `qdev_connect_gpio_out(dev, 0, irq)`.
    pub fn connect_irq(&self, line: IrqLine) {
        self.lock().irq.connect(line);
    }

    /// Uses `delivered` for the slew policy's delivery reports instead of the device's own.
    pub fn set_irq_delivered(&self, delivered: IrqDelivered) {
        self.lock().irq_delivered = delivered;
    }

    /// Called with the new guest time in seconds since the epoch whenever the guest sets the
    /// clock. `RTC_CHANGE` carries its difference to the host reference time.
    pub fn set_rtc_change_handler(&self, f: impl Fn(i64) + Send + Sync + 'static) {
        self.lock().rtc_change = Some(Arc::new(f));
    }

    /// Called when the alarm fires with AIE set, `qemu_system_wakeup_request()`.
    pub fn set_wakeup_handler(&self, f: impl Fn() + Send + Sync + 'static) {
        self.lock().wakeup = Some(Arc::new(move |()| f()));
    }

    /// `cmos_ioport_read()`. `addr` is the offset in the 2 byte port range.
    pub fn ioport_read(&self, addr: u64) -> u8 {
        self.lock().ioport_read(addr)
    }

    /// `cmos_ioport_write()`.
    pub fn ioport_write(&self, addr: u64, data: u8) {
        self.lock().ioport_write(addr, data);
    }

    /// `mc146818rtc_set_cmos_data()`.
    pub fn set_cmos_data(&self, addr: usize, val: u8) {
        self.lock().cmos_data[addr] = val;
    }

    /// `mc146818rtc_get_cmos_data()`.
    pub fn get_cmos_data(&self, addr: usize) -> u8 {
        self.lock().cmos_data[addr]
    }

    /// `rtc_reset_reinjection()`.
    pub fn reset_reinjection(&self) {
        self.lock().irq_coalesced = 0;
    }

    /// `rtc_notify_suspend()`: sets the CMOS shutdown status register (index 0xF) to
    /// S3_resume (0xFE), so the BIOS starts S3 resume at POST.
    pub fn notify_suspend(&self) {
        self.set_cmos_data(0xf, 0xfe);
    }

    /// `rtc_reset_enter()` and `rtc_reset_hold()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        // A guest suspending itself sets 0xfe, keep that and reset anything else.
        if s.cmos_data[0x0f] != 0xfe {
            s.cmos_data[0x0f] = 0x00;
        }

        s.cmos_data[RTC_REG_B] &= !(REG_B_PIE | REG_B_AIE | REG_B_SQWE);
        s.cmos_data[RTC_REG_C] &= !(REG_C_UF | REG_C_IRQF | REG_C_PF | REG_C_AF);
        s.check_update_timer();

        if s.lost_tick_policy == LostTickPolicy::Slew {
            s.irq_coalesced = 0;
            s.irq_reinject_on_ack_count = 0;
        }

        s.irq.lower();
    }

    /// `rtc_get_date()`, the `date` property.
    pub fn date(&self) -> Tm {
        let mut s = self.lock();
        s.update_time();
        s.get_time()
    }

    /// `rtc_set_date_from_host()` with the given host time.
    pub fn set_date(&self, host_date: SystemTime) {
        self.lock().set_date_from_host(unix_seconds(host_date));
    }

    /// `rtc_pre_save()`, which brings the CMOS time registers up to date, and the section's
    /// fields.
    pub fn vmstate_save(&self) -> Mc146818VmState {
        let mut s = self.lock();
        s.update_time();
        Mc146818VmState {
            cmos_data: s.cmos_data,
            cmos_index: s.cmos_index as u8,
            periodic_timer: s.periodic_timer.expire_time().unwrap_or(-1),
            next_periodic_time: s.next_periodic_time,
            irq_coalesced: s.irq_coalesced,
            period: s.period,
            base_rtc: s.base_rtc,
            last_update: s.last_update,
            offset: s.offset,
            update_timer: s.update_timer.expire_time().unwrap_or(-1),
            next_alarm_time: s.next_alarm_time,
            irq_reinject_on_ack_count: s.irq_reinject_on_ack_count,
        }
    }

    /// Takes over a loaded `mc146818rtc` section, version 3: the registers, the two timers
    /// armed for their loaded expiry, and then `rtc_post_load()`. On the realtime clock the
    /// date is taken from the CMOS registers again; the periodic timer is reprogrammed when the
    /// clock is before the next periodic interrupt or more than a minute past it; the slew
    /// policy's coalesced timer is restarted. The IRQ output is left alone, as in QEMU.
    pub fn vmstate_load(&self, v: &Mc146818VmState) {
        let mut s = self.lock();
        s.cmos_data = v.cmos_data;
        // cmos_index is a uint8_t in QEMU too; only the low 7 bits select a register.
        s.cmos_index = usize::from(v.cmos_index) & 0x7f;
        timer_put(&s.periodic_timer, v.periodic_timer);
        s.next_periodic_time = v.next_periodic_time;
        s.irq_coalesced = v.irq_coalesced;
        s.period = v.period;
        s.base_rtc = v.base_rtc;
        s.last_update = v.last_update;
        s.offset = v.offset;
        timer_put(&s.update_timer, v.update_timer);
        s.next_alarm_time = v.next_alarm_time;
        s.irq_reinject_on_ack_count = v.irq_reinject_on_ack_count;

        if s.clock.kind() == ClockType::Realtime {
            s.set_time();
            s.offset = 0;
            s.check_update_timer();
        }
        s.period = s.periodic_clock_ticks();
        let now = s.now() as u64;
        let next = s.next_periodic_time as u64;
        if now < next || now > next.wrapping_add(MAX_CLOCK_JUMP) {
            let period = s.period;
            s.periodic_timer_update(now as i64, period, false);
        }
        if s.lost_tick_policy == LostTickPolicy::Slew {
            s.coalesced_timer_update();
        }
    }
}

/// `cmos_ops`.
impl MmioOps for Mc146818Rtc {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.ioport_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.ioport_write(offset, value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gmtime_round_trips() {
        for t in [0, 1_700_000_000, 951_782_400, -86_401, 3_476_000_000, 4_102_444_800] {
            assert_eq!(mktimegm(&gmtime(t)), t);
        }
        let tm = gmtime(951_782_400);
        assert_eq!((tm.year, tm.mon, tm.mday, tm.wday), (100, 1, 29, 2));
        assert_eq!(gmtime(0).wday, 4);
    }

    #[test]
    fn period_codes() {
        assert_eq!(periodic_period_to_clock(0), 0);
        assert_eq!(periodic_period_to_clock(1), 128);
        assert_eq!(periodic_period_to_clock(3), 4);
        assert_eq!(periodic_period_to_clock(15), 16384);
        assert_eq!(periodic_clock_to_ns(32768), NANOSECONDS_PER_SECOND);
    }

    fn rtc_at(now: i64) -> (Arc<Clock>, Mc146818Rtc) {
        let clock = Clock::manual(ClockType::Host);
        clock.advance_to(now);
        let date = UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let rtc = Mc146818Rtc::new(Mc146818Props::default(), clock.clone(), date).unwrap();
        (clock, rtc)
    }

    #[test]
    fn vmstate_round_trip_keeps_the_timers() {
        let (clock, src) = rtc_at(3 * NANOSECONDS_PER_SECOND + 1234);
        // Periodic interrupts at the default 1024 Hz rate.
        src.ioport_write(0, RTC_REG_B as u8);
        src.ioport_write(1, REG_B_24H | REG_B_PIE);
        clock.advance_to(3 * NANOSECONDS_PER_SECOND + 500_000);
        let saved = src.vmstate_save();
        assert_eq!(saved.period, 32);
        assert_eq!(saved.periodic_timer, saved.next_periodic_time);
        assert!(saved.periodic_timer > 3 * NANOSECONDS_PER_SECOND + 500_000);
        assert_ne!(saved.update_timer, -1);
        assert_eq!(saved.cmos_index, RTC_REG_B as u8);

        // Just past the next periodic interrupt the loaded deadline stands, as in QEMU.
        let now = saved.next_periodic_time + 10;
        let (_clock2, dst) = rtc_at(now);
        dst.vmstate_load(&saved);
        assert_eq!(dst.vmstate_save(), saved);
        assert_eq!(dst.date(), src.date());

        // A periodic deadline ahead of the clock is reprogrammed from the current time.
        let mut ahead = saved.clone();
        ahead.next_periodic_time = now + 10 * NANOSECONDS_PER_SECOND;
        ahead.periodic_timer = ahead.next_periodic_time;
        dst.vmstate_load(&ahead);
        let again = dst.vmstate_save();
        assert!(again.next_periodic_time > now);
        assert!(again.next_periodic_time <= now + periodic_clock_to_ns(32) + 2);
        assert_eq!(again.periodic_timer, again.next_periodic_time);

        // With PIE clear, a reprogrammed periodic timer goes away.
        let mut off = ahead.clone();
        off.cmos_data[RTC_REG_B] = REG_B_24H;
        dst.vmstate_load(&off);
        let off = dst.vmstate_save();
        assert_eq!((off.period, off.periodic_timer), (0, -1));
    }
}
