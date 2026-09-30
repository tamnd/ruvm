// SPDX-License-Identifier: GPL-2.0-or-later

//! IRQ lines, `qemu_irq` from hw/core/irq.c.
//!
//! An [`IrqLine`] is the input end: a handler plus the line number it passes to that handler.
//! It is a cheap handle and can be cloned into every device that drives the line. A line that
//! was never connected does nothing, like a NULL `qemu_irq`.
//!
//! Devices keep their outputs in an [`IrqPin`], which starts disconnected and is wired later by
//! the board, the way `sysbus_connect_irq()` and `qdev_connect_gpio_out()` fill in a device's
//! output array after it was created.

use std::fmt;
use std::sync::{Arc, RwLock};

/// The handler behind a set of lines, `qemu_irq_handler`. It gets the line number and the level.
pub type IrqHandler = Arc<dyn Fn(u32, i32) + Send + Sync>;

/// `qemu_irq`.
#[derive(Clone, Default)]
pub struct IrqLine(Option<(IrqHandler, u32)>);

impl fmt::Debug for IrqLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Some((_, n)) => write!(f, "IrqLine({n})"),
            None => f.write_str("IrqLine(disconnected)"),
        }
    }
}

impl IrqLine {
    /// `qemu_allocate_irq()`: line `n` of `handler`.
    pub fn new(handler: IrqHandler, n: u32) -> Self {
        IrqLine(Some((handler, n)))
    }

    /// A line built from a closure that only cares about the level.
    pub fn from_fn(f: impl Fn(i32) + Send + Sync + 'static) -> Self {
        Self::new(Arc::new(move |_, level| f(level)), 0)
    }

    /// Whether the line goes anywhere.
    pub fn is_connected(&self) -> bool {
        self.0.is_some()
    }

    /// `qemu_set_irq()`.
    pub fn set(&self, level: i32) {
        if let Some((handler, n)) = &self.0 {
            handler(*n, level);
        }
    }

    /// `qemu_set_irq()` with a boolean level.
    pub fn set_bool(&self, level: bool) {
        self.set(i32::from(level));
    }

    /// `qemu_irq_raise()`.
    pub fn raise(&self) {
        self.set(1);
    }

    /// `qemu_irq_lower()`.
    pub fn lower(&self) {
        self.set(0);
    }

    /// `qemu_irq_pulse()`.
    pub fn pulse(&self) {
        self.set(1);
        self.set(0);
    }
}

/// `qemu_allocate_irqs()`: lines `0..n` of one handler.
pub fn allocate(handler: IrqHandler, n: u32) -> Vec<IrqLine> {
    (0..n).map(|i| IrqLine::new(handler.clone(), i)).collect()
}

/// `qemu_irq_invert()`.
pub fn invert(line: IrqLine) -> IrqLine {
    IrqLine::from_fn(move |level| line.set(i32::from(level == 0)))
}

/// The `split-irq` device: one input driving several outputs, first to last.
pub fn split(lines: Vec<IrqLine>) -> IrqLine {
    IrqLine::from_fn(move |level| {
        for l in &lines {
            l.set(level);
        }
    })
}

/// An output pin of a device. It can be connected after the device exists and read from any
/// thread.
#[derive(Default)]
pub struct IrqPin(RwLock<IrqLine>);

impl fmt::Debug for IrqPin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IrqPin({:?})", self.line())
    }
}

impl IrqPin {
    pub fn new() -> Self {
        Self::default()
    }

    /// `qdev_connect_gpio_out()`. Connecting again replaces the old line.
    pub fn connect(&self, line: IrqLine) {
        *self.0.write().unwrap_or_else(|p| p.into_inner()) = line;
    }

    /// The line the pin is wired to.
    pub fn line(&self) -> IrqLine {
        self.0.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn set(&self, level: i32) {
        self.line().set(level);
    }

    pub fn set_bool(&self, level: bool) {
        self.set(i32::from(level));
    }

    pub fn raise(&self) {
        self.set(1);
    }

    pub fn lower(&self) {
        self.set(0);
    }

    pub fn pulse(&self) {
        let line = self.line();
        line.raise();
        line.lower();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type Log = Arc<Mutex<Vec<(u32, i32)>>>;

    fn recorder() -> (IrqHandler, Log) {
        let log = Arc::new(Mutex::new(Vec::new()));
        let l = log.clone();
        (Arc::new(move |n, level| l.lock().unwrap().push((n, level))), log)
    }

    #[test]
    fn lines_carry_their_number() {
        let (h, log) = recorder();
        let lines = allocate(h, 3);
        lines[2].raise();
        lines[0].pulse();
        assert_eq!(*log.lock().unwrap(), [(2, 1), (0, 1), (0, 0)]);
    }

    #[test]
    fn disconnected_lines_do_nothing() {
        let pin = IrqPin::new();
        pin.raise();
        assert!(!pin.line().is_connected());
        let (h, log) = recorder();
        pin.connect(IrqLine::new(h, 7));
        pin.set(5);
        assert_eq!(*log.lock().unwrap(), [(7, 5)]);
    }

    #[test]
    fn invert_and_split() {
        let (h, log) = recorder();
        let out = split(vec![IrqLine::new(h.clone(), 0), invert(IrqLine::new(h, 1))]);
        out.raise();
        out.lower();
        assert_eq!(*log.lock().unwrap(), [(0, 1), (1, 0), (0, 0), (1, 1)]);
    }
}
