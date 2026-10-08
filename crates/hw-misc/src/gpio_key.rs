// SPDX-License-Identifier: GPL-2.0-or-later

//! A GPIO key, hw/gpio/gpio_key.c.
//!
//! [`GpioKey`] is `GPIOKEYState`, a button that a board presses for the guest. Any level on its
//! input ([`GpioKey::press`] or [`GpioKey::gpio_in`]) raises the output and arms a 100 ms timer
//! on the virtual clock, and the output drops again when the timer fires. The `virt` and
//! `sbsa-ref` boards wire the output to a PL061 pin and press it for `system_powerdown`.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState and QOM registration.
//! - The board calls [`GpioKey::reset`] for the legacy reset handler.

use std::fmt;
use std::sync::{Arc, Weak};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_hw_core::timer::{Clock, Timer};

/// `TYPE_GPIOKEY`.
pub const TYPE_GPIOKEY: &str = "gpio-key";

/// `GPIO_KEY_LATENCY`: how long the key stays pressed, in milliseconds.
pub const GPIO_KEY_LATENCY_MS: i64 = 100;

/// `GPIOKEYState`, the `gpio-key` device.
pub struct GpioKey {
    irq: IrqPin,
    clock: Arc<Clock>,
    timer: Timer,
}

impl fmt::Debug for GpioKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpioKey").field("irq", &self.irq).finish_non_exhaustive()
    }
}

impl GpioKey {
    /// `gpio_key_realize()`. `clock` is the virtual clock.
    pub fn new(clock: Arc<Clock>) -> Arc<GpioKey> {
        Arc::new_cyclic(|weak: &Weak<GpioKey>| {
            let w = weak.clone();
            let timer = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    s.timer_expired();
                }
            });
            GpioKey { irq: IrqPin::new(), clock, timer }
        })
    }

    /// The output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// The input, `qdev_get_gpio_in(dev, 0)`.
    pub fn gpio_in(self: &Arc<Self>) -> IrqLine {
        let weak = Arc::downgrade(self);
        IrqLine::from_fn(move |_| {
            if let Some(s) = weak.upgrade() {
                s.press();
            }
        })
    }

    /// `gpio_key_set_irq()`: the level is ignored, every call presses the key.
    pub fn press(&self) {
        self.irq.raise();
        let now_ms = self.clock.get_ms();
        self.timer.modify((now_ms + GPIO_KEY_LATENCY_MS) * 1_000_000);
    }

    /// `gpio_key_timer_expired()`.
    fn timer_expired(&self) {
        self.irq.lower();
        self.timer.del();
    }

    /// `gpio_key_reset()`. Like QEMU it only stops the timer and leaves the output as it is.
    pub fn reset(&self) {
        self.timer.del();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;
    use std::sync::atomic::{AtomicI32, Ordering};

    #[test]
    fn press_releases_after_100ms() {
        let clock = Clock::manual(ClockType::Virtual);
        let key = GpioKey::new(clock.clone());
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        key.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        key.gpio_in().set(0);
        assert_eq!(level.load(Ordering::SeqCst), 1);
        clock.advance_to(99_999_999);
        assert_eq!(level.load(Ordering::SeqCst), 1);
        clock.advance_to(100_000_000);
        assert_eq!(level.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reset_cancels_release() {
        let clock = Clock::manual(ClockType::Virtual);
        let key = GpioKey::new(clock.clone());
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        key.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        key.press();
        key.reset();
        clock.advance_to(1_000_000_000);
        assert_eq!(level.load(Ordering::SeqCst), 1);
    }
}
