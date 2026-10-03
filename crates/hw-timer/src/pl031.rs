// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm PrimeCell PL031 real time clock, hw/rtc/pl031.c.
//!
//! [`Pl031`] is `PL031State`. The counter is `tick_offset` plus the seconds the RTC clock (the
//! clock `-rtc clock=` picks) has run, and the alarm is a timer on that clock that sets the one
//! raw interrupt bit when the counter reaches the match register. The register block is 0x1000
//! bytes with the default access sizes, as `pl031_ops` has no constraints.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState (and so the `pl031/tick-offset` subsection and the conversion to and
//!   from the old virtual clock relative offset), trace points and QOM registration.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - QEMU reads the start date from the global `-rtc base=` state with `qemu_get_timedate()`.
//!   Here the board passes the date the RTC clock's zero corresponds to, as for the MC146818.
//! - The `RTC_CHANGE` QAPI event is a handler, [`Pl031::set_rtc_change_handler`].
//! - `pl031_ops` is `DEVICE_NATIVE_ENDIAN`. Every Arm target QEMU builds is little endian, so
//!   the registers are little endian here.
//! - Like QEMU, the device has no reset handler: a system reset keeps the counter, the match
//!   register and the interrupt state.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use ruvm_hw_core::irq::IrqPin;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_PL031`.
pub const TYPE_PL031: &str = "pl031";

/// Size of the register block, the `pl031` MMIO region.
pub const PL031_MMIO_SIZE: u64 = 0x1000;

/// Data read register.
pub const RTC_DR: u64 = 0x00;
/// Match register.
pub const RTC_MR: u64 = 0x04;
/// Data load register.
pub const RTC_LR: u64 = 0x08;
/// Control register.
pub const RTC_CR: u64 = 0x0c;
/// Interrupt mask and set register.
pub const RTC_IMSC: u64 = 0x10;
/// Raw interrupt status register.
pub const RTC_RIS: u64 = 0x14;
/// Masked interrupt status register.
pub const RTC_MIS: u64 = 0x18;
/// Interrupt clear register.
pub const RTC_ICR: u64 = 0x1c;
/// RTCPeriphID0, the first of the eight ID registers.
pub const RTC_PERIPHID0: u64 = 0xfe0;

/// `pl031_id`: the device ID, then the cell ID.
pub const PL031_ID: [u8; 8] = [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

type Hook<T> = Arc<dyn Fn(T) + Send + Sync>;

/// The register state of `PL031State`.
struct Pl031State {
    tick_offset: u32,
    mr: u32,
    lr: u32,
    cr: u32,
    im: u32,
    is: u32,
    /// The date, in seconds since the epoch, that the RTC clock's zero corresponds to.
    ref_start: i64,
    rtc_change: Option<Hook<i64>>,
}

impl fmt::Debug for Pl031State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pl031State")
            .field("tick_offset", &self.tick_offset)
            .field("mr", &self.mr)
            .field("lr", &self.lr)
            .field("cr", &self.cr)
            .field("im", &self.im)
            .field("is", &self.is)
            .finish_non_exhaustive()
    }
}

/// `PL031State`, the `pl031` device.
pub struct Pl031 {
    state: Mutex<Pl031State>,
    irq: IrqPin,
    clock: Arc<Clock>,
    timer: Timer,
}

impl fmt::Debug for Pl031 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pl031").field("state", &*self.lock()).field("irq", &self.irq).finish()
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

impl Pl031 {
    /// `pl031_init()`. `rtc_clock` is the clock chosen by `-rtc clock=`, and `host_date` is
    /// the date `qemu_get_timedate()` returns when that clock reads zero. The counter starts
    /// at `host_date` plus the current clock time.
    pub fn new(rtc_clock: Arc<Clock>, host_date: SystemTime) -> Arc<Pl031> {
        let ref_start = unix_seconds(host_date);
        Arc::new_cyclic(|weak: &Weak<Pl031>| {
            let w = weak.clone();
            let timer = rtc_clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    let mut st = s.lock();
                    s.interrupt(&mut st);
                }
            });
            let today = ref_start + rtc_clock.get_ms() / 1000;
            let tick_offset = (today - rtc_clock.get_ns() / NANOSECONDS_PER_SECOND) as u32;
            Pl031 {
                state: Mutex::new(Pl031State {
                    tick_offset,
                    mr: 0,
                    lr: 0,
                    cr: 0,
                    im: 0,
                    is: 0,
                    ref_start,
                    rtc_change: None,
                }),
                irq: IrqPin::new(),
                clock: rtc_clock,
                timer,
            }
        })
    }

    fn lock(&self) -> MutexGuard<'_, Pl031State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The interrupt output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// The offset between the guest counter and the RTC clock in seconds.
    pub fn tick_offset(&self) -> u32 {
        self.lock().tick_offset
    }

    /// Called whenever the guest writes the load register, with the seconds since the epoch
    /// that QEMU hands `qemu_timedate_diff()` for the `RTC_CHANGE` event. As in QEMU, that is
    /// `qemu_get_timedate()` with `tick_offset` as the offset, so the reference date is
    /// counted twice; the event's difference to the host date is taken from it unchanged.
    pub fn set_rtc_change_handler(&self, f: impl Fn(i64) + Send + Sync + 'static) {
        self.lock().rtc_change = Some(Arc::new(f));
    }

    /// `pl031_update()`.
    fn update(&self, s: &Pl031State) {
        self.irq.set_bool((s.is & s.im) != 0);
    }

    /// `pl031_interrupt()`.
    fn interrupt(&self, s: &mut Pl031State) {
        s.is = 1;
        self.update(s);
    }

    /// `pl031_get_count()`.
    fn get_count(&self, s: &Pl031State) -> u32 {
        let now = self.clock.get_ns();
        (i64::from(s.tick_offset) + now / NANOSECONDS_PER_SECOND) as u32
    }

    /// `pl031_set_alarm()`. The counter wraps, and so does the subtraction, which gives the
    /// right answer when the match value is behind the counter.
    fn set_alarm(&self, s: &mut Pl031State) {
        let ticks = s.mr.wrapping_sub(self.get_count(s));
        if ticks == 0 {
            self.timer.del();
            self.interrupt(s);
        } else {
            let now = self.clock.get_ns();
            self.timer.modify(now + i64::from(ticks) * NANOSECONDS_PER_SECOND);
        }
    }

    /// `pl031_read()`.
    pub fn reg_read(&self, offset: u64) -> u32 {
        let s = self.lock();
        match offset {
            RTC_DR => self.get_count(&s),
            RTC_MR => s.mr,
            RTC_IMSC => s.im,
            RTC_RIS => s.is,
            RTC_LR => s.lr,
            // The RTC is permanently enabled.
            RTC_CR => 1,
            RTC_MIS => s.is & s.im,
            0xfe0..=0xfff => u32::from(PL031_ID[((offset - RTC_PERIPHID0) >> 2) as usize]),
            // QEMU logs "pl031: read of write-only register at offset 0x%x" for ICR and
            // "pl031_read: Bad offset 0x%x" for the rest.
            _ => 0,
        }
    }

    /// `pl031_write()`.
    pub fn reg_write(&self, offset: u64, value: u64) {
        let mut guard = self.lock();
        let s = &mut *guard;
        match offset {
            RTC_LR => {
                s.lr = value as u32;
                let count = self.get_count(s);
                s.tick_offset = s.tick_offset.wrapping_add((value as u32).wrapping_sub(count));
                if let Some(event) = &s.rtc_change {
                    let today = s.ref_start + self.clock.get_ms() / 1000;
                    event(today + i64::from(s.tick_offset));
                }
                self.set_alarm(s);
            }
            RTC_MR => {
                s.mr = value as u32;
                self.set_alarm(s);
            }
            RTC_IMSC => {
                s.im = (value & 1) as u32;
                self.update(s);
            }
            RTC_ICR => {
                s.is &= !(value as u32);
                self.update(s);
            }
            // The written value is ignored.
            RTC_CR => {}
            // QEMU logs "pl031: write to read-only register at offset 0x%x" for DR, MIS and
            // RIS and "pl031_write: Bad offset 0x%x" for the rest.
            _ => {}
        }
    }
}

/// `pl031_ops`.
impl MmioOps for Pl031 {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.reg_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }
}
