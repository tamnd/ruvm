// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm PrimeCell PL061 GPIO controller, hw/gpio/pl061.c.
//!
//! [`Pl061`] is `PL061State`: eight pins, each an input ([`Pl061::gpio_in`]) and an output
//! ([`Pl061::out`]), plus one interrupt line that is set when a masked pin event is pending.
//! The register block is 0x1000 bytes with the default access sizes, as `pl061_ops` has no
//! constraints.
//!
//! Pins that are inputs are pulled up, pulled down or left floating as the `pullups` and
//! `pulldowns` properties say ([`Pl061Props`]). A pulled up input reads as 1 on its output
//! line, a floating one keeps the level it had, which is what lets the `virt` and `sbsa-ref`
//! boards wire a power button to a pin.
//!
//! Differences from QEMU:
//!
//! - Not ported: the Stellaris `pl061_luminary` variant with its extra drive, pull and lock
//!   registers at 0x500 to 0x528, VMState, trace points and QOM registration. Those offsets
//!   are bad offsets here, as they are on the plain PL061.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - `pl061_ops` is `DEVICE_NATIVE_ENDIAN`. Every Arm target QEMU builds is little endian, so
//!   the registers are little endian here.
//! - The board calls [`Pl061::reset`] for the enter and hold reset phases.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_PL061`.
pub const TYPE_PL061: &str = "pl061";

/// Size of the register block, the `pl061` MMIO region.
pub const PL061_MMIO_SIZE: u64 = 0x1000;

/// `N_GPIOS`.
pub const PL061_NUM_GPIOS: usize = 8;

/// `pl061_id`: the ID registers at 0xfd0 to 0xffc.
pub const PL061_ID: [u8; 12] =
    [0x00, 0x00, 0x00, 0x00, 0x61, 0x10, 0x04, 0x00, 0x0d, 0xf0, 0x05, 0xb1];

/// Direction register.
pub const GPIODIR: u64 = 0x400;
/// Interrupt sense register.
pub const GPIOIS: u64 = 0x404;
/// Interrupt both edges register.
pub const GPIOIBE: u64 = 0x408;
/// Interrupt event register.
pub const GPIOIEV: u64 = 0x40c;
/// Interrupt mask register.
pub const GPIOIE: u64 = 0x410;
/// Raw interrupt status register.
pub const GPIORIS: u64 = 0x414;
/// Masked interrupt status register.
pub const GPIOMIS: u64 = 0x418;
/// Interrupt clear register.
pub const GPIOIC: u64 = 0x41c;
/// Alternate function select register.
pub const GPIOAFSEL: u64 = 0x420;

/// The `pl061` properties.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Pl061Props {
    /// `pullups`: input pins pulled up to 1. Defaults to all of them.
    pub pullups: u8,
    /// `pulldowns`: input pins pulled down to 0.
    pub pulldowns: u8,
}

impl Default for Pl061Props {
    fn default() -> Self {
        Pl061Props { pullups: 0xff, pulldowns: 0 }
    }
}

/// The registers of `PL061State` that the plain PL061 uses. The luminary only ones are left
/// out.
#[derive(Debug, Default)]
struct Pl061State {
    data: u32,
    old_out_data: u32,
    old_in_data: u32,
    dir: u32,
    isense: u32,
    ibe: u32,
    iev: u32,
    im: u32,
    istate: u32,
    afsel: u32,
    /// The commit register, which only the luminary variant lets a guest change, so it stays
    /// at 0xff and every AFSEL bit is writable.
    cr: u32,
}

/// `PL061State`, the `pl061` device.
pub struct Pl061 {
    state: Mutex<Pl061State>,
    props: Pl061Props,
    irq: IrqPin,
    out: [IrqPin; PL061_NUM_GPIOS],
}

impl fmt::Debug for Pl061 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pl061")
            .field("state", &*self.lock())
            .field("props", &self.props)
            .field("irq", &self.irq)
            .finish_non_exhaustive()
    }
}

impl Pl061 {
    /// `pl061_init()` and `pl061_realize()`. Fails as QEMU's realize does when a pin is both
    /// pulled up and pulled down. The device starts in its reset state.
    pub fn new(props: Pl061Props) -> Result<Arc<Pl061>, String> {
        if props.pullups & props.pulldowns != 0 {
            return Err("no bit may be set both in pullups and pulldowns".to_string());
        }
        let dev = Arc::new(Pl061 {
            state: Mutex::new(Pl061State::default()),
            props,
            irq: IrqPin::new(),
            out: std::array::from_fn(|_| IrqPin::new()),
        });
        dev.reset();
        Ok(dev)
    }

    fn lock(&self) -> MutexGuard<'_, Pl061State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The interrupt output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// Output pin `n`, `s->out[n]`.
    pub fn out(&self, n: usize) -> &IrqPin {
        &self.out[n]
    }

    /// Input pin `n`, the `qdev_get_gpio_in()` line that `pl061_set_irq()` serves.
    pub fn gpio_in(self: &Arc<Self>, n: u32) -> IrqLine {
        let weak = Arc::downgrade(self);
        IrqLine::new(
            Arc::new(move |n, level| {
                if let Some(s) = weak.upgrade() {
                    s.set_irq(n, level);
                }
            }),
            n,
        )
    }

    /// `pl061_floating()`: input pins that are neither pulled up nor down.
    fn floating(&self, s: &Pl061State) -> u32 {
        u32::from(!(self.props.pullups | self.props.pulldowns)) & !s.dir & 0xff
    }

    /// `pl061_pullups()`: input pins that are pulled up.
    fn pullups(&self, s: &Pl061State) -> u32 {
        u32::from(self.props.pullups) & !s.dir & 0xff
    }

    /// `pl061_update()`.
    fn update(&self, s: &mut Pl061State) {
        let pullups = self.pullups(s);
        let floating = self.floating(s);

        // Pins configured as output are driven from the data register. Otherwise a pin that
        // is pulled up is 1, and a floating one keeps the value it had before, so no change
        // is reported to the other end.
        let out = (s.data & s.dir & 0xff) | pullups | (s.old_out_data & floating);
        let changed = s.old_out_data ^ out;
        if changed != 0 {
            s.old_out_data = out;
            for (i, pin) in self.out.iter().enumerate() {
                let mask = 1 << i;
                if changed & mask != 0 {
                    pin.set_bool(out & mask != 0);
                }
            }
        }

        // Inputs.
        let changed = (s.old_in_data ^ s.data) & !s.dir & 0xff;
        if changed != 0 {
            s.old_in_data = s.data;
            for i in 0..PL061_NUM_GPIOS {
                let mask = 1 << i;
                if changed & mask != 0 && s.isense & mask == 0 {
                    // An edge interrupt: any edge with IBE set, otherwise the edge IEV picks.
                    if s.ibe & mask != 0 {
                        s.istate |= mask;
                    } else {
                        s.istate |= !(s.data ^ s.iev) & mask;
                    }
                }
            }
        }

        // Level interrupts.
        s.istate |= !(s.data ^ s.iev) & s.isense;

        self.irq.set_bool(s.istate & s.im != 0);
    }

    /// `pl061_read()`.
    pub fn reg_read(&self, offset: u64) -> u32 {
        let s = self.lock();
        match offset {
            0x0..=0x3ff => s.data & (offset >> 2) as u32,
            GPIODIR => s.dir,
            GPIOIS => s.isense,
            GPIOIBE => s.ibe,
            GPIOIEV => s.iev,
            GPIOIE => s.im,
            GPIORIS => s.istate,
            GPIOMIS => s.istate & s.im,
            GPIOAFSEL => s.afsel,
            0xfd0..=0xfff => u32::from(PL061_ID[((offset - 0xfd0) >> 2) as usize]),
            // QEMU logs "pl061_read: Bad offset %x".
            _ => 0,
        }
    }

    /// `pl061_write()`.
    pub fn reg_write(&self, offset: u64, value: u64) {
        let mut guard = self.lock();
        let s = &mut *guard;
        let byte = (value & 0xff) as u32;
        match offset {
            0x0..=0x3ff => {
                let mask = ((offset >> 2) as u32 & s.dir) & 0xff;
                s.data = (s.data & !mask) | (value as u32 & mask);
            }
            GPIODIR => s.dir = byte,
            GPIOIS => s.isense = byte,
            GPIOIBE => s.ibe = byte,
            GPIOIEV => s.iev = byte,
            GPIOIE => s.im = byte,
            GPIOIC => s.istate &= !(value as u32),
            GPIOAFSEL => {
                let mask = s.cr & 0xff;
                s.afsel = (s.afsel & !mask) | (value as u32 & mask);
            }
            // QEMU logs "pl061_write: Bad offset %x" and does not update.
            _ => return,
        }
        self.update(s);
    }

    /// `pl061_set_irq()`: a level on input pin `n`. Pins configured as outputs ignore it.
    pub fn set_irq(&self, n: u32, level: i32) {
        let mut s = self.lock();
        let mask = 1u32 << n;
        if s.dir & mask == 0 {
            s.data &= !mask;
            if level != 0 {
                s.data |= mask;
            }
            self.update(&mut s);
        }
    }

    /// `pl061_enter_reset()` then `pl061_hold_reset()`: the registers go back to their reset
    /// values and the input pins that are not floating drive their pulled levels.
    pub fn reset(&self) {
        let mut s = self.lock();
        *s = Pl061State { cr: 0xff, ..Pl061State::default() };
        let floating = self.floating(&s);
        let pullups = self.pullups(&s);
        for (i, pin) in self.out.iter().enumerate() {
            if floating & (1 << i) != 0 {
                continue;
            }
            pin.set_bool(pullups & (1 << i) != 0);
        }
        s.old_out_data = pullups;
    }
}

/// `pl061_ops`.
impl MmioOps for Pl061 {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.reg_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    fn level(pin: &IrqPin) -> Arc<AtomicI32> {
        let l = Arc::new(AtomicI32::new(-1));
        let c = l.clone();
        pin.connect(IrqLine::from_fn(move |v| c.store(v, Ordering::SeqCst)));
        l
    }

    #[test]
    fn id_registers() {
        let g = Pl061::new(Pl061Props::default()).unwrap();
        let id: Vec<u32> = (0..12).map(|i| g.reg_read(0xfd0 + 4 * i)).collect();
        assert_eq!(id, PL061_ID.iter().map(|&b| u32::from(b)).collect::<Vec<_>>());
    }

    #[test]
    fn realize_rejects_overlap() {
        let err = Pl061::new(Pl061Props { pullups: 0x81, pulldowns: 0x01 }).unwrap_err();
        assert_eq!(err, "no bit may be set both in pullups and pulldowns");
    }

    #[test]
    fn data_is_masked_by_address_and_direction() {
        let g = Pl061::new(Pl061Props::default()).unwrap();
        let o0 = level(g.out(0));
        g.reg_write(GPIODIR, 0x0f);
        // Address bits 9:2 select the pins a write may change.
        g.reg_write(0x3fc, 0xff);
        assert_eq!(g.reg_read(0x3fc), 0x0f);
        g.reg_write(0x3fc, 0x00);
        assert_eq!(o0.load(Ordering::SeqCst), 0);
        g.reg_write(0x004, 0xff);
        assert_eq!(o0.load(Ordering::SeqCst), 1);
        assert_eq!(g.reg_read(0x3fc), 0x01);
        assert_eq!(g.reg_read(0x004), 0x01);
    }

    #[test]
    fn edge_interrupt_on_rising_input() {
        let g = Pl061::new(Pl061Props { pullups: 0, pulldowns: 0 }).unwrap();
        let irq = level(g.irq());
        let pin = g.gpio_in(3);
        g.reg_write(GPIOIEV, 0x08);
        g.reg_write(GPIOIE, 0x08);
        pin.set(1);
        assert_eq!(irq.load(Ordering::SeqCst), 1);
        assert_eq!(g.reg_read(GPIORIS), 0x08);
        assert_eq!(g.reg_read(GPIOMIS), 0x08);
        pin.set(0);
        g.reg_write(GPIOIC, 0x08);
        assert_eq!(irq.load(Ordering::SeqCst), 0);
        assert_eq!(g.reg_read(GPIORIS), 0);
    }

    #[test]
    fn level_interrupt_follows_iev() {
        let g = Pl061::new(Pl061Props::default()).unwrap();
        let irq = level(g.irq());
        g.reg_write(GPIOIS, 0x01);
        g.reg_write(GPIOIE, 0x01);
        // IEV is 0, so a low input is the active level.
        assert_eq!(irq.load(Ordering::SeqCst), 1);
        g.reg_write(GPIOIC, 0x01);
        assert_eq!(g.reg_read(GPIORIS), 0x01);
    }

    #[test]
    fn reset_drives_pullups() {
        let g = Pl061::new(Pl061Props { pullups: 0x01, pulldowns: 0x02 }).unwrap();
        let o0 = level(g.out(0));
        let o1 = level(g.out(1));
        let o2 = level(g.out(2));
        g.reset();
        assert_eq!(o0.load(Ordering::SeqCst), 1);
        assert_eq!(o1.load(Ordering::SeqCst), 0);
        // Pin 2 floats, so reset leaves it alone.
        assert_eq!(o2.load(Ordering::SeqCst), -1);
    }

    #[test]
    fn output_pins_ignore_inputs() {
        let g = Pl061::new(Pl061Props::default()).unwrap();
        g.reg_write(GPIODIR, 0x01);
        g.set_irq(0, 1);
        assert_eq!(g.reg_read(0x3fc), 0);
        g.set_irq(1, 1);
        assert_eq!(g.reg_read(0x3fc), 0x02);
    }

    #[test]
    fn afsel_and_bad_offsets() {
        let g = Pl061::new(Pl061Props::default()).unwrap();
        g.reg_write(GPIOAFSEL, 0x1ff);
        assert_eq!(g.reg_read(GPIOAFSEL), 0xff);
        g.reg_write(0x500, 0x12);
        assert_eq!(g.reg_read(0x500), 0);
    }
}
