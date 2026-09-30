// SPDX-License-Identifier: GPL-2.0-or-later

//! The 8253/8254 interval timer, `isa-pit`, from hw/timer/i8254.c and hw/timer/i8254_common.c.
//!
//! Three channels count down at [`PIT_FREQ`] on the virtual clock. Nothing ticks: the count and
//! the OUT pin are computed from the time the count was loaded. Channel 0 drives the IRQ output
//! through a timer armed at each OUT transition, channel 2 is the PC speaker and its gate is set
//! by port 0x61 through [`I8254::set_gate`].
//!
//! VMState, trace points, QOM registration and the KVM in-kernel `kvm-pit` are not ported.
//!
//! The channel state sits behind a mutex. The IRQ output is set after the lock is dropped, so
//! whatever the line is wired to (the PIC, the HPET) may call back into the PIT.

use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer, muldiv64};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, Endian, MemResult, MmioOps};

/// `TYPE_I8254`.
pub const TYPE_I8254: &str = "isa-pit";

/// `PIT_FREQ`: the input clock of the counters in Hz.
pub const PIT_FREQ: u32 = 1193182;

/// `RW_STATE_LSB`.
pub const RW_STATE_LSB: u8 = 1;
/// `RW_STATE_MSB`.
pub const RW_STATE_MSB: u8 = 2;
/// `RW_STATE_WORD0`.
pub const RW_STATE_WORD0: u8 = 3;
/// `RW_STATE_WORD1`.
pub const RW_STATE_WORD1: u8 = 4;

/// `PITChannelInfo`, what the speaker and port 0x61 read back.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PitChannelInfo {
    pub gate: i32,
    pub mode: i32,
    pub initial_count: i32,
    pub out: i32,
}

/// `PITChannelState` without the timer and IRQ, which live in [`I8254`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PitChannelState {
    /// Can be 65536.
    pub count: i32,
    pub latched_count: u16,
    pub count_latched: u8,
    pub status_latched: u8,
    pub status: u8,
    pub read_state: u8,
    pub write_state: u8,
    pub write_latch: u8,
    pub rw_mode: u8,
    pub mode: u8,
    /// Not supported.
    pub bcd: u8,
    /// Timer start.
    pub gate: u8,
    pub count_load_time: i64,
    /// IRQ handling, only used on channel 0.
    pub next_transition_time: i64,
    pub irq_disabled: u32,
}

/// Counter ticks since the count was loaded.
fn ticks(s: &PitChannelState, current_time: i64) -> u64 {
    muldiv64(
        current_time.wrapping_sub(s.count_load_time) as u64,
        PIT_FREQ,
        NANOSECONDS_PER_SECOND as u32,
    )
}

/// `pit_get_out()`: the level of the OUT pin at `current_time`.
pub fn pit_get_out(s: &PitChannelState, current_time: i64) -> i32 {
    let d = ticks(s, current_time);
    let count = s.count as u64;
    let out = match s.mode {
        2 => d % count == 0 && d != 0,
        3 => d % count < (count + 1) >> 1,
        4 | 5 => d == count,
        // 0, 1 and the undefined modes 6 and 7.
        _ => d >= count,
    };
    i32::from(out)
}

/// `pit_get_next_transition_time()`: when OUT changes next, or -1 if it never does.
pub fn pit_get_next_transition_time(s: &PitChannelState, current_time: i64) -> i64 {
    let d = ticks(s, current_time);
    let count = s.count as u64;
    let next_time = match s.mode {
        2 => {
            let base = d / count * count;
            if d - base == 0 && d != 0 { base + count } else { base + count + 1 }
        }
        3 => {
            let base = d / count * count;
            let period2 = (count + 1) >> 1;
            if d - base < period2 { base + period2 } else { base + count }
        }
        4 | 5 => {
            if d < count {
                count
            } else if d == count {
                count + 1
            } else {
                return -1;
            }
        }
        _ => {
            if d < count {
                count
            } else {
                return -1;
            }
        }
    };
    // Convert to timer units.
    let mut next_time = (s.count_load_time as u64).wrapping_add(muldiv64(
        next_time,
        NANOSECONDS_PER_SECOND as u32,
        PIT_FREQ,
    ));
    // Fix potential rounding problems.
    if next_time <= current_time as u64 {
        next_time = current_time as u64 + 1;
    }
    next_time as i64
}

/// `pit_get_count()`.
fn pit_get_count(s: &PitChannelState, now: i64) -> i32 {
    let d = ticks(s, now);
    let count = s.count as u64;
    let counter = match s.mode {
        0 | 1 | 4 | 5 => count.wrapping_sub(d) & 0xffff,
        // XXX: may be incorrect for odd counts.
        3 => count - (2 * d) % count,
        _ => count - d % count,
    };
    counter as i32
}

/// `PITCommonState` minus the I/O region.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PitCommonState {
    pub channels: [PitChannelState; 3],
}

/// The `isa-pit` device.
#[derive(Debug)]
pub struct I8254 {
    /// The `iobase` property, 0x40 on a PC.
    pub iobase: u32,
    clock: Arc<Clock>,
    state: Mutex<PitCommonState>,
    /// The output of channel 0, gpio out 0.
    pub irq: IrqPin,
    /// The timer behind channel 0's IRQ.
    irq_timer: Timer,
}

impl I8254 {
    /// Creates and resets the device. `clock` is the virtual clock.
    pub fn new(clock: &Arc<Clock>, iobase: u32) -> Arc<Self> {
        let pit = Arc::new_cyclic(|weak: &Weak<I8254>| {
            let w = weak.clone();
            // The timer 0 is connected to an IRQ.
            let irq_timer = clock.new_timer(move || {
                if let Some(pit) = w.upgrade() {
                    pit.pit_irq_timer();
                }
            });
            I8254 {
                iobase,
                clock: clock.clone(),
                state: Mutex::new(PitCommonState::default()),
                irq: IrqPin::new(),
                irq_timer,
            }
        });
        pit.reset();
        pit
    }

    fn lock(&self) -> MutexGuard<'_, PitCommonState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn now(&self) -> i64 {
        self.clock.get_ns()
    }

    /// A copy of the channel state.
    pub fn state(&self) -> PitCommonState {
        self.lock().clone()
    }

    /// `pit_irq_timer_update()`. Only channel 0 has a timer. Returns the level to put on the IRQ
    /// output once the lock is released.
    fn pit_irq_timer_update(
        &self,
        pit: &mut PitCommonState,
        channel: usize,
        current_time: i64,
    ) -> Option<i32> {
        let s = &mut pit.channels[channel];
        if channel != 0 || s.irq_disabled != 0 {
            return None;
        }
        let expire_time = pit_get_next_transition_time(s, current_time);
        let irq_level = pit_get_out(s, current_time);
        s.next_transition_time = expire_time;
        if expire_time != -1 {
            self.irq_timer.modify(expire_time);
        } else {
            self.irq_timer.del();
        }
        Some(irq_level)
    }

    fn set_irq(&self, level: Option<i32>) {
        if let Some(level) = level {
            self.irq.set(level);
        }
    }

    /// `pit_irq_timer()`.
    fn pit_irq_timer(&self) {
        let level = {
            let mut pit = self.lock();
            let t = pit.channels[0].next_transition_time;
            self.pit_irq_timer_update(&mut pit, 0, t)
        };
        self.set_irq(level);
    }

    /// `pit_load_count()`.
    fn pit_load_count(&self, pit: &mut PitCommonState, channel: usize, val: i32) -> Option<i32> {
        let val = if val == 0 { 0x10000 } else { val };
        let now = self.now();
        let s = &mut pit.channels[channel];
        s.count_load_time = now;
        s.count = val;
        self.pit_irq_timer_update(pit, channel, now)
    }

    /// `pit_latch_count()`. If already latched, do not latch again.
    fn pit_latch_count(&self, s: &mut PitChannelState) {
        if s.count_latched == 0 {
            s.latched_count = pit_get_count(s, self.now()) as u16;
            s.count_latched = s.rw_mode;
        }
    }

    /// `pit_ioport_write()`.
    pub fn ioport_write(&self, addr: u64, val: u8) {
        let level = {
            let mut pit = self.lock();
            self.write_locked(&mut pit, (addr & 3) as usize, val)
        };
        self.set_irq(level);
    }

    fn write_locked(&self, pit: &mut PitCommonState, addr: usize, val: u8) -> Option<i32> {
        if addr == 3 {
            let channel = usize::from(val >> 6);
            if channel == 3 {
                // Read back command.
                for (channel, s) in pit.channels.iter_mut().enumerate() {
                    if val & (2 << channel) == 0 {
                        continue;
                    }
                    if val & 0x20 == 0 {
                        self.pit_latch_count(s);
                    }
                    if val & 0x10 == 0 && s.status_latched == 0 {
                        // Status latch. XXX: add BCD and null count.
                        let out = pit_get_out(s, self.now()) as u8;
                        s.status = (out << 7) | (s.rw_mode << 4) | (s.mode << 1) | s.bcd;
                        s.status_latched = 1;
                    }
                }
            } else {
                let s = &mut pit.channels[channel];
                let access = (val >> 4) & 3;
                if access == 0 {
                    self.pit_latch_count(s);
                } else {
                    s.rw_mode = access;
                    s.read_state = access;
                    s.write_state = access;
                    s.mode = (val >> 1) & 7;
                    s.bcd = val & 1;
                    // XXX: update irq timer ?
                }
            }
            None
        } else {
            let s = &mut pit.channels[addr];
            match s.write_state {
                RW_STATE_MSB => self.pit_load_count(pit, addr, i32::from(val) << 8),
                RW_STATE_WORD0 => {
                    s.write_latch = val;
                    s.write_state = RW_STATE_WORD1;
                    None
                }
                RW_STATE_WORD1 => {
                    let v = i32::from(s.write_latch) | (i32::from(val) << 8);
                    s.write_state = RW_STATE_WORD0;
                    self.pit_load_count(pit, addr, v)
                }
                // RW_STATE_LSB
                _ => self.pit_load_count(pit, addr, i32::from(val)),
            }
        }
    }

    /// `pit_ioport_read()`.
    pub fn ioport_read(&self, addr: u64) -> u8 {
        let addr = (addr & 3) as usize;
        if addr == 3 {
            // Mode/Command register is write only, read is ignored.
            return 0;
        }
        let now = self.now();
        let mut pit = self.lock();
        let s = &mut pit.channels[addr];
        let ret: i32 = if s.status_latched != 0 {
            s.status_latched = 0;
            i32::from(s.status)
        } else if s.count_latched != 0 {
            match s.count_latched {
                RW_STATE_MSB => {
                    s.count_latched = 0;
                    i32::from(s.latched_count >> 8)
                }
                RW_STATE_WORD0 => {
                    s.count_latched = RW_STATE_MSB;
                    i32::from(s.latched_count & 0xff)
                }
                // RW_STATE_LSB
                _ => {
                    s.count_latched = 0;
                    i32::from(s.latched_count & 0xff)
                }
            }
        } else {
            match s.read_state {
                RW_STATE_MSB => (pit_get_count(s, now) >> 8) & 0xff,
                RW_STATE_WORD0 => {
                    s.read_state = RW_STATE_WORD1;
                    pit_get_count(s, now) & 0xff
                }
                RW_STATE_WORD1 => {
                    s.read_state = RW_STATE_WORD0;
                    (pit_get_count(s, now) >> 8) & 0xff
                }
                // RW_STATE_LSB
                _ => pit_get_count(s, now) & 0xff,
            }
        };
        ret as u8
    }

    /// `pit_set_gate()` with `pit_set_channel_gate()`. `val` must be 0 or 1.
    pub fn set_gate(&self, channel: usize, val: i32) {
        let level = {
            let mut pit = self.lock();
            let sc = &mut pit.channels[channel];
            let mut level = None;
            // Modes 0 and 4 only disable/enable counting, which is not done (XXX, as in modes
            // 2 and 3). Modes 1, 2, 3 and 5 restart counting on a rising edge.
            if matches!(sc.mode, 1 | 2 | 3 | 5) && i32::from(sc.gate) < val {
                let now = self.now();
                sc.count_load_time = now;
                level = self.pit_irq_timer_update(&mut pit, channel, now);
            }
            pit.channels[channel].gate = val as u8;
            level
        };
        self.set_irq(level);
    }

    /// `pit_get_channel_info()` with `pit_get_channel_info_common()`.
    pub fn get_channel_info(&self, channel: usize) -> PitChannelInfo {
        let now = self.now();
        let pit = self.lock();
        let sc = &pit.channels[channel];
        PitChannelInfo {
            gate: i32::from(sc.gate),
            mode: i32::from(sc.mode),
            initial_count: sc.count,
            out: pit_get_out(sc, now),
        }
    }

    /// `pit_reset()` with `pit_reset_common()`.
    pub fn reset(&self) {
        let now = self.now();
        let mut pit = self.lock();
        for (i, s) in pit.channels.iter_mut().enumerate() {
            s.mode = 3;
            s.gate = u8::from(i != 2);
            s.count_load_time = now;
            s.count = 0x10000;
            if i == 0 && s.irq_disabled == 0 {
                s.next_transition_time = pit_get_next_transition_time(s, s.count_load_time);
            }
        }
        let s = &pit.channels[0];
        if s.irq_disabled == 0 {
            self.irq_timer.modify(s.next_transition_time);
        }
    }

    /// `pit_irq_control()`. When the HPET is in legacy mode it suppresses the ignored timer
    /// IRQ, and enables it again when legacy mode is left.
    pub fn irq_control(&self, enable: bool) {
        let level = {
            let mut pit = self.lock();
            if enable {
                pit.channels[0].irq_disabled = 0;
                let now = self.now();
                self.pit_irq_timer_update(&mut pit, 0, now)
            } else {
                pit.channels[0].irq_disabled = 1;
                self.irq_timer.del();
                None
            }
        };
        self.set_irq(level);
    }

    /// gpio in 0, wired to `pit_irq_control()`.
    pub fn irq_control_in(self: &Arc<Self>) -> IrqLine {
        let w = Arc::downgrade(self);
        IrqLine::from_fn(move |level| {
            if let Some(pit) = w.upgrade() {
                pit.irq_control(level != 0);
            }
        })
    }

    /// Whether channel 0's IRQ timer is armed, and for when.
    pub fn irq_timer_expire_time(&self) -> Option<i64> {
        self.irq_timer.expire_time()
    }
}

/// `pit_ioport_ops`: four byte wide ports.
impl MmioOps for I8254 {
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

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}
