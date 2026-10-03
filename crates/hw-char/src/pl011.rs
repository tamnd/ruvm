// SPDX-License-Identifier: GPL-2.0-or-later

//! The Arm PrimeCell PL011 UART, hw/char/pl011.c.
//!
//! [`Pl011`] is `PL011State`: the registers, the receive FIFO (16 entries, or 1 when the FIFO is
//! disabled), loopback of data, break and the modem control lines, and the six interrupt
//! outputs. The character device is the same abstract [`SerialBackend`] the 16550 uses. The
//! device transmits through it, and whatever feeds it input calls [`Pl011::can_receive`],
//! [`Pl011::receive`] and [`Pl011::receive_break`], the `qemu_chr_fe_set_handlers()` callbacks.
//!
//! The interrupt outputs are, in order, UARTINTR (the combined line, the only one `virt`
//! wires), UARTRXINTR, UARTTXINTR, UARTRTINTR, UARTMSINTR and UARTEINTR. The register block is
//! 0x1000 bytes and implements 32-bit accesses only, as `pl011_ops` does; the memory core
//! widens narrower accesses.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState (and so `pl011_post_load()` and the `migrate-clk` property), trace
//!   points, QOM registration and `pl011_create()`.
//! - The `qemu_log_mask()` guest error and unimplemented messages are not printed, since the
//!   workspace has no `-d` log yet. The once-only "disabled UART" state is still tracked.
//! - The `clk` input is a frequency set with [`Pl011::set_clock_hz`] rather than a `Clock`
//!   object. QEMU only uses it to trace the baud rate, which [`Pl011::baudrate`] returns.
//! - `qemu_chr_fe_write_all()` retries a busy backend every 100 microseconds, as QEMU does on
//!   `EAGAIN`; here busy means [`SerialBackend::write`] returned 0.
//! - [`Pl011::can_receive`] returns 0 when loopback has pushed the FIFO past its depth. QEMU
//!   computes the room in unsigned arithmetic there and hands the chardev layer a negative
//!   count, which it also treats as no room.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, RwLock};
use std::time::Duration;

use ruvm_hw_core::IrqPin;
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::serial::SerialBackend;

/// `TYPE_PL011`.
pub const TYPE_PL011: &str = "pl011";
/// `TYPE_PL011_LUMINARY`.
pub const TYPE_PL011_LUMINARY: &str = "pl011_luminary";

/// `PL011_FIFO_DEPTH`: the receive FIFO depth when the FIFO is enabled.
pub const PL011_FIFO_DEPTH: usize = 16;

/// Size of the register block, the `pl011` MMIO region.
pub const PL011_MMIO_SIZE: u64 = 0x1000;

/// Number of interrupt outputs.
pub const PL011_NUM_IRQS: usize = 6;

// Register offsets.

/// UARTDR, data.
pub const UARTDR: u64 = 0x00;
/// UARTRSR on read, UARTECR on write.
pub const UARTRSR: u64 = 0x04;
/// UARTFR, flags.
pub const UARTFR: u64 = 0x18;
/// UARTILPR, IrDA low power counter.
pub const UARTILPR: u64 = 0x20;
/// UARTIBRD, integer baud rate divisor.
pub const UARTIBRD: u64 = 0x24;
/// UARTFBRD, fractional baud rate divisor.
pub const UARTFBRD: u64 = 0x28;
/// UARTLCR_H, line control.
pub const UARTLCR_H: u64 = 0x2c;
/// UARTCR, control.
pub const UARTCR: u64 = 0x30;
/// UARTIFLS, interrupt FIFO level select.
pub const UARTIFLS: u64 = 0x34;
/// UARTIMSC, interrupt mask set and clear.
pub const UARTIMSC: u64 = 0x38;
/// UARTRIS, raw interrupt status.
pub const UARTRIS: u64 = 0x3c;
/// UARTMIS, masked interrupt status.
pub const UARTMIS: u64 = 0x40;
/// UARTICR, interrupt clear.
pub const UARTICR: u64 = 0x44;
/// UARTDMACR, DMA control.
pub const UARTDMACR: u64 = 0x48;
/// UARTPeriphID0, the first of the eight ID registers.
pub const UARTPERIPHID0: u64 = 0xfe0;

// Flag Register, UARTFR.

pub const PL011_FLAG_RI: u32 = 0x100;
pub const PL011_FLAG_TXFE: u32 = 0x80;
pub const PL011_FLAG_RXFF: u32 = 0x40;
pub const PL011_FLAG_TXFF: u32 = 0x20;
pub const PL011_FLAG_RXFE: u32 = 0x10;
pub const PL011_FLAG_DCD: u32 = 0x04;
pub const PL011_FLAG_DSR: u32 = 0x02;
pub const PL011_FLAG_CTS: u32 = 0x01;

/// Data Register, UARTDR: break error.
pub const DR_BE: u32 = 1 << 10;

// Interrupt status bits in UARTRIS, UARTMIS and UARTIMSC.

pub const INT_OE: u32 = 1 << 10;
pub const INT_BE: u32 = 1 << 9;
pub const INT_PE: u32 = 1 << 8;
pub const INT_FE: u32 = 1 << 7;
pub const INT_RT: u32 = 1 << 6;
pub const INT_TX: u32 = 1 << 5;
pub const INT_RX: u32 = 1 << 4;
pub const INT_DSR: u32 = 1 << 3;
pub const INT_DCD: u32 = 1 << 2;
pub const INT_CTS: u32 = 1 << 1;
pub const INT_RI: u32 = 1 << 0;
pub const INT_E: u32 = INT_OE | INT_BE | INT_PE | INT_FE;
pub const INT_MS: u32 = INT_RI | INT_DSR | INT_DCD | INT_CTS;

// Line Control Register, UARTLCR_H.

pub const LCR_FEN: u32 = 1 << 4;
pub const LCR_BRK: u32 = 1 << 0;

// Control Register, UARTCR.

pub const CR_OUT2: u32 = 1 << 13;
pub const CR_OUT1: u32 = 1 << 12;
pub const CR_RTS: u32 = 1 << 11;
pub const CR_DTR: u32 = 1 << 10;
pub const CR_RXE: u32 = 1 << 9;
pub const CR_TXE: u32 = 1 << 8;
pub const CR_LBE: u32 = 1 << 7;
pub const CR_UARTEN: u32 = 1 << 0;

/// Integer Baud Rate Divider, UARTIBRD.
pub const IBRD_MASK: u32 = 0xffff;
/// Fractional Baud Rate Divider, UARTFBRD.
pub const FBRD_MASK: u32 = 0x3f;

/// `pl011_id_arm`.
pub const PL011_ID_ARM: [u8; 8] = [0x11, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];
/// `pl011_id_luminary`.
pub const PL011_ID_LUMINARY: [u8; 8] = [0x11, 0x00, 0x18, 0x01, 0x0d, 0xf0, 0x05, 0xb1];

/// Which bits in the interrupt status matter for each outbound IRQ line, `irqmask`.
const IRQMASK: [u32; PL011_NUM_IRQS] = [
    INT_E | INT_MS | INT_RT | INT_TX | INT_RX, // combined IRQ
    INT_RX,
    INT_TX,
    INT_RT,
    INT_MS,
    INT_E,
];

/// The register state of `PL011State`.
#[derive(Debug)]
struct Pl011State {
    flags: u32,
    lcr: u32,
    rsr: u32,
    cr: u32,
    dmacr: u32,
    int_enabled: u32,
    int_level: u32,
    read_fifo: [u32; PL011_FIFO_DEPTH],
    ilpr: u32,
    ibrd: u32,
    fbrd: u32,
    ifl: u32,
    read_pos: i32,
    read_count: i32,
    read_trigger: i32,
    clk_hz: u64,
    logged_disabled_uart: bool,
}

impl Pl011State {
    fn loopback_enabled(&self) -> bool {
        (self.cr & CR_LBE) != 0
    }

    fn is_fifo_enabled(&self) -> bool {
        (self.lcr & LCR_FEN) != 0
    }

    /// `pl011_get_fifo_depth()`. The depth is a power of 2.
    fn fifo_depth(&self) -> i32 {
        if self.is_fifo_enabled() { PL011_FIFO_DEPTH as i32 } else { 1 }
    }

    fn reset_rx_fifo(&mut self) {
        self.read_count = 0;
        self.read_pos = 0;
        self.flags &= !PL011_FLAG_RXFF;
        self.flags |= PL011_FLAG_RXFE;
    }

    fn reset_tx_fifo(&mut self) {
        self.flags &= !PL011_FLAG_TXFF;
        self.flags |= PL011_FLAG_TXFE;
    }

    /// `pl011_set_read_trigger()`. The documented trigger is the IFLS level, but Linux only
    /// reads the FIFO in response to an interrupt, so QEMU interrupts as soon as the FIFO is not
    /// empty.
    fn set_read_trigger(&mut self) {
        self.read_trigger = 1;
    }

    /// `pl011_get_baudrate()`.
    fn baudrate(&self) -> u32 {
        if self.ibrd == 0 {
            return 0;
        }
        let div = (u64::from(self.ibrd) << 6) + u64::from(self.fbrd);
        ((self.clk_hz / div) << 2) as u32
    }
}

/// `PL011State`, the `pl011` and `pl011_luminary` devices.
pub struct Pl011 {
    state: Mutex<Pl011State>,
    irq: [IrqPin; PL011_NUM_IRQS],
    id: &'static [u8; 8],
    backend: RwLock<Option<Arc<dyn SerialBackend>>>,
}

impl fmt::Debug for Pl011 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pl011")
            .field("state", &*self.lock())
            .field("irq", &self.irq)
            .field("id", &self.id)
            .field("backend", &self.backend().is_some())
            .finish()
    }
}

impl Pl011 {
    /// The `pl011` device, realized and reset. `backend` is the `chardev` property.
    pub fn new(backend: Option<Arc<dyn SerialBackend>>) -> Arc<Pl011> {
        Self::with_id(&PL011_ID_ARM, backend)
    }

    /// The `pl011_luminary` device, which differs only in its ID registers.
    pub fn new_luminary(backend: Option<Arc<dyn SerialBackend>>) -> Arc<Pl011> {
        Self::with_id(&PL011_ID_LUMINARY, backend)
    }

    fn with_id(id: &'static [u8; 8], backend: Option<Arc<dyn SerialBackend>>) -> Arc<Pl011> {
        let s = Arc::new(Pl011 {
            state: Mutex::new(Pl011State {
                flags: 0,
                lcr: 0,
                rsr: 0,
                cr: 0,
                dmacr: 0,
                int_enabled: 0,
                int_level: 0,
                read_fifo: [0; PL011_FIFO_DEPTH],
                ilpr: 0,
                ibrd: 0,
                fbrd: 0,
                ifl: 0,
                read_pos: 0,
                read_count: 0,
                read_trigger: 1,
                clk_hz: 0,
                logged_disabled_uart: false,
            }),
            irq: std::array::from_fn(|_| IrqPin::new()),
            id,
            backend: RwLock::new(backend),
        });
        s.reset();
        s
    }

    fn lock(&self) -> MutexGuard<'_, Pl011State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn backend(&self) -> Option<Arc<dyn SerialBackend>> {
        self.backend.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// Interrupt output `n`, sysbus IRQ `n`: 0 is UARTINTR, then RX, TX, RT, MS and E.
    ///
    /// # Panics
    ///
    /// If `n` is 6 or more.
    pub fn irq(&self, n: usize) -> &IrqPin {
        &self.irq[n]
    }

    /// All six interrupt outputs.
    pub fn irqs(&self) -> &[IrqPin; PL011_NUM_IRQS] {
        &self.irq
    }

    /// The eight bytes the ID registers at 0xfe0 return.
    pub fn id(&self) -> &'static [u8; 8] {
        self.id
    }

    /// Sets the frequency of the `clk` input, `UARTCLK`, in Hz.
    pub fn set_clock_hz(&self, hz: u64) {
        self.lock().clk_hz = hz;
    }

    /// The frequency of the `clk` input in Hz.
    pub fn clock_hz(&self) -> u64 {
        self.lock().clk_hz
    }

    /// `pl011_get_baudrate()`: the line speed the divisors select, or 0 if UARTIBRD is 0.
    pub fn baudrate(&self) -> u32 {
        self.lock().baudrate()
    }

    /// Changes the chardev, or detaches it.
    pub fn set_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        *self.backend.write().unwrap_or_else(|p| p.into_inner()) = backend;
    }

    /// `pl011_update()`.
    fn update(&self, s: &Pl011State) {
        let flags = s.int_level & s.int_enabled;
        for (pin, mask) in self.irq.iter().zip(IRQMASK) {
            pin.set_bool((flags & mask) != 0);
        }
    }

    /// `pl011_fifo_rx_put()`.
    fn fifo_rx_put(&self, s: &mut Pl011State, value: u32) {
        let pipe_depth = s.fifo_depth();
        let slot = (s.read_pos + s.read_count) & (pipe_depth - 1);
        s.read_fifo[slot as usize] = value;
        s.read_count += 1;
        s.flags &= !PL011_FLAG_RXFE;
        if s.read_count == pipe_depth {
            s.flags |= PL011_FLAG_RXFF;
        }
        if s.read_count == s.read_trigger {
            s.int_level |= INT_RX;
            self.update(s);
        }
    }

    /// `pl011_loopback_tx()`. Real hardware loops back after the TX FIFO at the frame rate, so
    /// a full RX FIFO might drop different bytes; that is not emulated.
    fn loopback_tx(&self, s: &mut Pl011State, value: u32) {
        if !s.loopback_enabled() {
            return;
        }
        self.fifo_rx_put(s, value);
    }

    /// `pl011_write_txdata()`.
    fn write_txdata(&self, s: &mut Pl011State, data: u8) {
        if (s.cr & CR_UARTEN) == 0 && !s.logged_disabled_uart {
            // QEMU logs "PL011 data written to disabled UART" once here.
            s.logged_disabled_uart = true;
        }
        // QEMU logs "PL011 data written to disabled TX UART" when CR.TXE is clear, and sends
        // the byte anyway.
        if let Some(be) = self.backend() {
            write_all(&*be, &[data]);
        }
        self.loopback_tx(s, u32::from(data));
        s.int_level |= INT_TX;
        self.update(s);
    }

    /// `pl011_read_rxdata()`.
    fn read_rxdata(&self, s: &mut Pl011State) -> u32 {
        let fifo_depth = s.fifo_depth();
        s.flags &= !PL011_FLAG_RXFF;
        let c = s.read_fifo[s.read_pos as usize];
        if s.read_count > 0 {
            s.read_count -= 1;
            s.read_pos = (s.read_pos + 1) & (fifo_depth - 1);
        }
        if s.read_count == 0 {
            s.flags |= PL011_FLAG_RXFE;
        }
        if s.read_count == s.read_trigger - 1 {
            s.int_level &= !INT_RX;
        }
        s.rsr = c >> 8;
        self.update(s);
        if let Some(be) = self.backend() {
            be.accept_input();
        }
        c
    }

    /// `pl011_loopback_mdmctrl()`: in loopback, the modem control outputs drive the modem
    /// status inputs right away, even when the write only set CR.LBE.
    fn loopback_mdmctrl(&self, s: &mut Pl011State) {
        if !s.loopback_enabled() {
            return;
        }
        let cr = s.cr;
        let mut fr = s.flags & !(PL011_FLAG_RI | PL011_FLAG_DCD | PL011_FLAG_DSR | PL011_FLAG_CTS);
        fr |= if (cr & CR_OUT2) != 0 { PL011_FLAG_RI } else { 0 };
        fr |= if (cr & CR_OUT1) != 0 { PL011_FLAG_DCD } else { 0 };
        fr |= if (cr & CR_RTS) != 0 { PL011_FLAG_CTS } else { 0 };
        fr |= if (cr & CR_DTR) != 0 { PL011_FLAG_DSR } else { 0 };

        let mut il = s.int_level & !(INT_DSR | INT_DCD | INT_CTS | INT_RI);
        il |= if (fr & PL011_FLAG_DSR) != 0 { INT_DSR } else { 0 };
        il |= if (fr & PL011_FLAG_DCD) != 0 { INT_DCD } else { 0 };
        il |= if (fr & PL011_FLAG_CTS) != 0 { INT_CTS } else { 0 };
        il |= if (fr & PL011_FLAG_RI) != 0 { INT_RI } else { 0 };

        s.flags = fr;
        s.int_level = il;
        self.update(s);
    }

    /// `pl011_read()`: a 32-bit register read.
    pub fn reg_read(&self, offset: u64) -> u32 {
        let mut guard = self.lock();
        let s = &mut *guard;
        match offset >> 2 {
            0 => self.read_rxdata(s),
            1 => s.rsr,
            6 => s.flags,
            8 => s.ilpr,
            9 => s.ibrd,
            10 => s.fbrd,
            11 => s.lcr,
            12 => s.cr,
            13 => s.ifl,
            14 => s.int_enabled,
            15 => s.int_level,
            16 => s.int_level & s.int_enabled,
            18 => s.dmacr,
            0x3f8..=0x3ff => u32::from(self.id[((offset - UARTPERIPHID0) >> 2) as usize]),
            // QEMU logs "pl011_read: Bad offset 0x%x".
            _ => 0,
        }
    }

    /// `pl011_write()`: a 32-bit register write.
    pub fn reg_write(&self, offset: u64, value: u32) {
        let mut guard = self.lock();
        let s = &mut *guard;
        match offset >> 2 {
            0 => self.write_txdata(s, value as u8),
            // UARTECR: any write clears the error flags.
            1 => s.rsr = 0,
            // Writes to the flag register are ignored.
            6 => {}
            8 => s.ilpr = value,
            9 => s.ibrd = value & IBRD_MASK,
            10 => s.fbrd = value & FBRD_MASK,
            11 => {
                // Reset the FIFO state on FIFO enable or disable.
                if ((s.lcr ^ value) & LCR_FEN) != 0 {
                    s.reset_rx_fifo();
                    s.reset_tx_fifo();
                }
                if ((s.lcr ^ value) & LCR_BRK) != 0 {
                    let break_enable = (value & LCR_BRK) != 0;
                    if let Some(be) = self.backend() {
                        be.set_break(break_enable);
                    }
                    // pl011_loopback_break()
                    if break_enable {
                        self.loopback_tx(s, DR_BE);
                    }
                }
                s.lcr = value;
                s.set_read_trigger();
            }
            12 => {
                // The enable bit is not implemented; toggling it re-arms the log message.
                if ((s.cr ^ value) & CR_UARTEN) != 0 {
                    s.logged_disabled_uart = false;
                }
                s.cr = value;
                self.loopback_mdmctrl(s);
            }
            13 => {
                s.ifl = value;
                s.set_read_trigger();
            }
            14 => {
                s.int_enabled = value;
                self.update(s);
            }
            17 => {
                s.int_level &= !value;
                self.update(s);
            }
            // DMA is not implemented; QEMU logs that when bits 0 or 1 are set.
            18 => s.dmacr = value,
            // QEMU logs "pl011_write: Bad offset 0x%x".
            _ => {}
        }
    }

    /// `pl011_can_receive()`: how many bytes [`Pl011::receive`] takes now. The UART and RX
    /// enable bits are not checked, because QEMU never enforced them and guests rely on that.
    pub fn can_receive(&self) -> usize {
        let s = self.lock();
        (s.fifo_depth() - s.read_count).max(0) as usize
    }

    /// `pl011_receive()`. In loopback mode the RX input is disconnected and this does nothing.
    pub fn receive(&self, buf: &[u8]) {
        let mut guard = self.lock();
        let s = &mut *guard;
        if s.loopback_enabled() {
            return;
        }
        for &b in buf {
            self.fifo_rx_put(s, u32::from(b));
        }
    }

    /// `pl011_event()` with `CHR_EVENT_BREAK`: queues a break, unless in loopback.
    pub fn receive_break(&self) {
        let mut guard = self.lock();
        let s = &mut *guard;
        if !s.loopback_enabled() {
            self.fifo_rx_put(s, DR_BE);
        }
    }

    /// `pl011_reset()`.
    pub fn reset(&self) {
        let mut guard = self.lock();
        let s = &mut *guard;
        s.lcr = 0;
        s.rsr = 0;
        s.dmacr = 0;
        s.int_enabled = 0;
        s.int_level = 0;
        s.ilpr = 0;
        s.ibrd = 0;
        s.fbrd = 0;
        s.read_trigger = 1;
        s.ifl = 0x12;
        s.cr = 0x300;
        s.flags = 0;
        s.logged_disabled_uart = false;
        s.reset_rx_fifo();
        s.reset_tx_fifo();
    }
}

/// `qemu_chr_fe_write_all()`: keeps going until the backend took everything, waiting 100
/// microseconds whenever it is busy.
fn write_all(be: &dyn SerialBackend, mut bytes: &[u8]) {
    while !bytes.is_empty() {
        let n = be.write(bytes);
        if n == 0 {
            std::thread::sleep(Duration::from_micros(100));
            continue;
        }
        bytes = &bytes[n.min(bytes.len())..];
    }
}

/// `pl011_ops`.
impl MmioOps for Pl011 {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.reg_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.reg_write(offset, value as u32);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}
