// SPDX-License-Identifier: GPL-2.0-or-later

//! The 16550A UART, hw/char/serial.c, and its ISA front end, hw/char/serial-isa.c.
//!
//! [`Serial`] is `SerialState`: the registers, the 16 byte receive and transmit FIFOs, the
//! receive timeout timer and the modem status poll timer. The character device is abstract.
//! The device transmits through a [`SerialBackend`], and whatever feeds it input calls
//! [`Serial::can_receive`] and [`Serial::receive`], the `qemu_chr_fe_set_handlers()` pair.
//! [`IsaSerial`] is `isa-serial`, which picks the COM port base and IRQ from its index.
//!
//! [`Serial::vmstate_save`] and [`Serial::vmstate_load`] move the state the `serial` VMState
//! carries, as a [`SerialVmState`] whose fields are named after QEMU's.
//!
//! Not ported: trace points, QOM registration, the `wakeup` property, the ACPI description of
//! `isa-serial`, and the `serial-mm` and `serial-pci` front ends.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, Weak};

use ruvm_base::{Error, Result};
use ruvm_hw_core::timer::NANOSECONDS_PER_SECOND;
use ruvm_hw_core::{Clock, IrqLine, IrqPin, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `UART_FIFO_LENGTH`: 16550A FIFO length.
pub const UART_FIFO_LENGTH: usize = 16;

/// Divisor latch access bit.
pub const UART_LCR_DLAB: u8 = 0x80;
/// Set break.
pub const UART_LCR_SB: u8 = 0x40;
/// Even parity select.
pub const UART_LCR_EPS: u8 = 0x10;
/// Parity enable.
pub const UART_LCR_PEN: u8 = 0x08;
/// Number of stop bits.
pub const UART_LCR_NSTB: u8 = 0x04;
/// Word length select.
pub const UART_LCR_WLS: u8 = 0x03;

/// Enable modem status interrupt.
pub const UART_IER_MSI: u8 = 0x08;
/// Enable receiver line status interrupt.
pub const UART_IER_RLSI: u8 = 0x04;
/// Enable transmitter holding register interrupt.
pub const UART_IER_THRI: u8 = 0x02;
/// Enable receiver data interrupt.
pub const UART_IER_RDI: u8 = 0x01;

/// No interrupts pending.
pub const UART_IIR_NO_INT: u8 = 0x01;
/// Mask for the interrupt ID.
pub const UART_IIR_ID: u8 = 0x06;

/// Modem status interrupt.
pub const UART_IIR_MSI: u8 = 0x00;
/// Transmitter holding register empty.
pub const UART_IIR_THRI: u8 = 0x02;
/// Receiver data interrupt.
pub const UART_IIR_RDI: u8 = 0x04;
/// Receiver line status interrupt.
pub const UART_IIR_RLSI: u8 = 0x06;
/// Character timeout indication.
pub const UART_IIR_CTI: u8 = 0x0C;

/// FIFO enabled, but not functioning.
pub const UART_IIR_FENF: u8 = 0x80;
/// FIFO enabled.
pub const UART_IIR_FE: u8 = 0xC0;

/// Enable loopback test mode.
pub const UART_MCR_LOOP: u8 = 0x10;
/// Out2 complement.
pub const UART_MCR_OUT2: u8 = 0x08;
/// Out1 complement.
pub const UART_MCR_OUT1: u8 = 0x04;
/// RTS complement.
pub const UART_MCR_RTS: u8 = 0x02;
/// DTR complement.
pub const UART_MCR_DTR: u8 = 0x01;

/// Data carrier detect.
pub const UART_MSR_DCD: u8 = 0x80;
/// Ring indicator.
pub const UART_MSR_RI: u8 = 0x40;
/// Data set ready.
pub const UART_MSR_DSR: u8 = 0x20;
/// Clear to send.
pub const UART_MSR_CTS: u8 = 0x10;
/// Delta DCD.
pub const UART_MSR_DDCD: u8 = 0x08;
/// Trailing edge ring indicator.
pub const UART_MSR_TERI: u8 = 0x04;
/// Delta DSR.
pub const UART_MSR_DDSR: u8 = 0x02;
/// Delta CTS.
pub const UART_MSR_DCTS: u8 = 0x01;
/// Any of the delta bits.
pub const UART_MSR_ANY_DELTA: u8 = 0x0F;

/// Transmitter empty.
pub const UART_LSR_TEMT: u8 = 0x40;
/// Transmit holding register empty.
pub const UART_LSR_THRE: u8 = 0x20;
/// Break interrupt indicator.
pub const UART_LSR_BI: u8 = 0x10;
/// Frame error indicator.
pub const UART_LSR_FE: u8 = 0x08;
/// Parity error indicator.
pub const UART_LSR_PE: u8 = 0x04;
/// Overrun error indicator.
pub const UART_LSR_OE: u8 = 0x02;
/// Receiver data ready.
pub const UART_LSR_DR: u8 = 0x01;
/// Any of the LSR interrupt triggering status bits.
pub const UART_LSR_INT_ANY: u8 = 0x1E;

// Interrupt trigger levels. The byte counts are for the 16550A, newer UARTs have higher ones.

/// 1 byte ITL.
pub const UART_FCR_ITL_1: u8 = 0x00;
/// 4 bytes ITL.
pub const UART_FCR_ITL_2: u8 = 0x40;
/// 8 bytes ITL.
pub const UART_FCR_ITL_3: u8 = 0x80;
/// 14 bytes ITL.
pub const UART_FCR_ITL_4: u8 = 0xC0;

/// DMA mode select.
pub const UART_FCR_DMS: u8 = 0x08;
/// Transmit FIFO reset.
pub const UART_FCR_XFR: u8 = 0x04;
/// Receive FIFO reset.
pub const UART_FCR_RFR: u8 = 0x02;
/// FIFO enable.
pub const UART_FCR_FE: u8 = 0x01;

/// `MAX_XMIT_RETRY`.
pub const MAX_XMIT_RETRY: u32 = 4;

/// Modem line bits of the TIOCM ioctls, from include/chardev/char-serial.h.
pub const CHR_TIOCM_CTS: u32 = 0x020;
pub const CHR_TIOCM_CAR: u32 = 0x040;
pub const CHR_TIOCM_DSR: u32 = 0x100;
pub const CHR_TIOCM_RI: u32 = 0x080;
pub const CHR_TIOCM_DTR: u32 = 0x002;
pub const CHR_TIOCM_RTS: u32 = 0x004;

/// The `baudbase` property default.
pub const SERIAL_BAUDBASE_DEFAULT: u32 = 115200;

/// Size of the register block, the `serial` I/O region.
pub const SERIAL_IO_SIZE: u64 = 8;

/// Line settings handed to the backend, `QEMUSerialSetParams`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SerialParams {
    pub speed: i32,
    /// `'N'`, `'E'` or `'O'`.
    pub parity: char,
    pub data_bits: i32,
    pub stop_bits: i32,
}

/// The character device side of the UART, the ioctls and writes serial.c makes through
/// `CharFrontend`.
///
/// The device calls these with its register lock held, so they must not call back into the
/// same [`Serial`].
pub trait SerialBackend: Send + Sync {
    /// `qemu_chr_fe_write()`: sends what it can and returns how many bytes it took. 0 means
    /// try again later, and the owner then calls [`Serial::write_ready`].
    fn write(&self, bytes: &[u8]) -> usize;

    /// `CHR_IOCTL_SERIAL_SET_PARAMS`.
    fn set_params(&self, params: &SerialParams) {
        let _ = params;
    }

    /// `CHR_IOCTL_SERIAL_SET_BREAK`.
    fn set_break(&self, enable: bool) {
        let _ = enable;
    }

    /// `CHR_IOCTL_SERIAL_GET_TIOCM`: the `CHR_TIOCM_*` lines, or `None` when the backend is
    /// not a real serial port (`-ENOTSUP`).
    fn get_tiocm(&self) -> Option<u32> {
        None
    }

    /// `CHR_IOCTL_SERIAL_SET_TIOCM`.
    fn set_tiocm(&self, flags: u32) {
        let _ = flags;
    }

    /// `qemu_chr_fe_accept_input()`: the guest took a byte, so there may be room for more.
    fn accept_input(&self) {}
}

/// `Fifo8` with the 16 byte depth of the UART.
#[derive(Debug, Default)]
struct Fifo8(VecDeque<u8>);

impl Fifo8 {
    fn new() -> Self {
        Fifo8(VecDeque::with_capacity(UART_FIFO_LENGTH))
    }

    fn is_full(&self) -> bool {
        self.0.len() >= UART_FIFO_LENGTH
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn num_used(&self) -> usize {
        self.0.len()
    }

    fn push(&mut self, v: u8) {
        assert!(!self.is_full(), "fifo8_push: FIFO full");
        self.0.push_back(v);
    }

    fn pop(&mut self) -> u8 {
        self.0.pop_front().expect("fifo8_pop: FIFO empty")
    }

    fn reset(&mut self) {
        self.0.clear();
    }

    /// The ring as `vmstate_fifo8` sends it: the bytes in order from slot 0.
    fn to_vmstate(&self) -> Fifo8VmState {
        let mut v = Fifo8VmState::default();
        for (d, &b) in v.data.iter_mut().zip(self.0.iter()) {
            *d = b;
        }
        v.num = self.0.len() as u32;
        v
    }

    /// Refills the FIFO from what `vmstate_fifo8` loaded. QEMU takes `head` and `num` as they
    /// come; out of range values are refused here rather than indexing out of the ring.
    fn load_vmstate(&mut self, v: &Fifo8VmState, name: &str) -> Result<()> {
        if v.head as usize >= UART_FIFO_LENGTH || v.num as usize > UART_FIFO_LENGTH {
            return Err(Error::generic(format!(
                "serial: {name} head {} num {} out of range",
                v.head, v.num
            )));
        }
        self.0.clear();
        for i in 0..v.num as usize {
            self.0.push_back(v.data[(v.head as usize + i) % UART_FIFO_LENGTH]);
        }
        Ok(())
    }
}

/// `Fifo8` as `vmstate_fifo8` carries it: the whole ring, the read position and the count.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Fifo8VmState {
    /// `data`, `capacity` bytes.
    pub data: [u8; UART_FIFO_LENGTH],
    pub head: u32,
    pub num: u32,
}

/// The fields of `vmstate_serial` (version 3) and its subsections, named as in QEMU.
///
/// `fifo_timeout_timer` and `modem_status_poll` are expiry times on the device's clock, the
/// virtual clock, or -1 when the timer is not armed. `thr_ipending` and `poll_msl` are -1 when
/// the stream did not carry them, what `serial_pre_load()` sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SerialVmState {
    pub divider: u16,
    pub rbr: u8,
    pub ier: u8,
    pub iir: u8,
    pub lcr: u8,
    pub mcr: u8,
    pub lsr: u8,
    pub msr: u8,
    pub scr: u8,
    /// `fcr`, copied by `serial_pre_save()`.
    pub fcr_vmstate: u8,
    /// `serial/thr_ipending`.
    pub thr_ipending: i32,
    /// `serial/tsr`.
    pub tsr_retry: u32,
    pub thr: u8,
    pub tsr: u8,
    /// `serial/recv_fifo`.
    pub recv_fifo: Fifo8VmState,
    /// `serial/xmit_fifo`.
    pub xmit_fifo: Fifo8VmState,
    /// `serial/fifo_timeout_timer`.
    pub fifo_timeout_timer: i64,
    /// `serial/timeout_ipending`.
    pub timeout_ipending: i32,
    /// `serial/poll`.
    pub poll_msl: i32,
    pub modem_status_poll: i64,
}

impl SerialVmState {
    /// `serial_thr_ipending_needed()`.
    pub fn thr_ipending_needed(&self) -> bool {
        if (self.ier & UART_IER_THRI) != 0 {
            let expected = i32::from((self.iir & UART_IIR_ID) == UART_IIR_THRI);
            self.thr_ipending != expected
        } else {
            // LSR.THRE is sampled again when the interrupt is enabled.
            false
        }
    }

    /// `serial_tsr_needed()`.
    pub fn tsr_needed(&self) -> bool {
        self.tsr_retry != 0
    }

    /// `serial_recv_fifo_needed()`.
    pub fn recv_fifo_needed(&self) -> bool {
        self.recv_fifo.num != 0
    }

    /// `serial_xmit_fifo_needed()`.
    pub fn xmit_fifo_needed(&self) -> bool {
        self.xmit_fifo.num != 0
    }

    /// `serial_fifo_timeout_timer_needed()`.
    pub fn fifo_timeout_timer_needed(&self) -> bool {
        self.fifo_timeout_timer != -1
    }

    /// `serial_timeout_ipending_needed()`.
    pub fn timeout_ipending_needed(&self) -> bool {
        self.timeout_ipending != 0
    }

    /// `serial_poll_needed()`.
    pub fn poll_needed(&self) -> bool {
        self.poll_msl >= 0
    }
}

/// `timer_put()`: the expiry time, -1 when the timer is not armed.
fn timer_vmstate(t: &Timer) -> i64 {
    t.expire_time().unwrap_or(-1)
}

/// `timer_get()`: arms the timer at `expire`, or stops it for -1.
fn timer_load(t: &Timer, expire: i64) {
    if expire == -1 {
        t.del();
    } else {
        t.modify(expire);
    }
}

/// The register state of `SerialState`, behind the device lock.
#[derive(Debug)]
struct SerialState {
    divider: u16,
    /// Receive register.
    rbr: u8,
    /// Transmit holding register.
    thr: u8,
    /// Transmit shift register.
    tsr: u8,
    ier: u8,
    /// Read only.
    iir: u8,
    lcr: u8,
    mcr: u8,
    /// Read only.
    lsr: u8,
    /// Read only.
    msr: u8,
    scr: u8,
    fcr: u8,
    /// Hidden state needed for tx irq generation, since it can be reset while reading IIR.
    thr_ipending: bool,
    last_break_enable: bool,
    baudbase: u32,
    tsr_retry: u32,
    /// Time when the last byte was successfully sent out of the TSR.
    last_xmit_ts: i64,
    recv_fifo: Fifo8,
    xmit_fifo: Fifo8,
    /// Interrupt trigger level for `recv_fifo`.
    recv_fifo_itl: u8,
    /// Timeout interrupt pending state.
    timeout_ipending: bool,
    /// Time to transmit a char in ns.
    char_transmit_time: u64,
    poll_msl: i32,
}

/// `SerialState`, the 16550A core that `isa-serial` and friends wrap.
pub struct Serial {
    state: Mutex<SerialState>,
    irq: IrqPin,
    clock: Arc<Clock>,
    backend: RwLock<Option<Arc<dyn SerialBackend>>>,
    fifo_timeout_timer: Timer,
    modem_status_poll: Timer,
}

impl fmt::Debug for Serial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Serial")
            .field("state", &*self.lock())
            .field("irq", &self.irq)
            .field("backend", &self.backend().is_some())
            .finish()
    }
}

impl Serial {
    /// `serial_realize()` followed by `serial_reset()`. Timers run on `clock`, the virtual
    /// clock, and `backend` is the `chardev` property.
    pub fn new(
        clock: Arc<Clock>,
        baudbase: u32,
        backend: Option<Arc<dyn SerialBackend>>,
    ) -> Arc<Serial> {
        let s = Arc::new_cyclic(|weak: &Weak<Serial>| {
            let w = weak.clone();
            let modem_status_poll = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    let mut st = s.lock();
                    s.update_msl(&mut st);
                }
            });
            let w = weak.clone();
            let fifo_timeout_timer = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    s.fifo_timeout_int();
                }
            });
            Serial {
                state: Mutex::new(SerialState {
                    divider: 0,
                    rbr: 0,
                    thr: 0,
                    tsr: 0,
                    ier: 0,
                    iir: 0,
                    lcr: 0,
                    mcr: 0,
                    lsr: 0,
                    msr: 0,
                    scr: 0,
                    fcr: 0,
                    thr_ipending: false,
                    last_break_enable: false,
                    baudbase,
                    tsr_retry: 0,
                    last_xmit_ts: 0,
                    recv_fifo: Fifo8::new(),
                    xmit_fifo: Fifo8::new(),
                    recv_fifo_itl: 0,
                    timeout_ipending: false,
                    char_transmit_time: 0,
                    poll_msl: 0,
                }),
                irq: IrqPin::new(),
                clock,
                backend: RwLock::new(backend),
                fifo_timeout_timer,
                modem_status_poll,
            }
        });
        s.reset();
        s
    }

    fn lock(&self) -> MutexGuard<'_, SerialState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn backend(&self) -> Option<Arc<dyn SerialBackend>> {
        self.backend.read().unwrap_or_else(|p| p.into_inner()).clone()
    }

    /// The interrupt output, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// The `baudbase` property.
    pub fn baudbase(&self) -> u32 {
        self.lock().baudbase
    }

    /// Time to send one character at the current settings, in nanoseconds.
    pub fn char_transmit_time(&self) -> u64 {
        self.lock().char_transmit_time
    }

    /// Virtual time at which the last byte left the transmit shift register.
    pub fn last_xmit_ts(&self) -> i64 {
        self.lock().last_xmit_ts
    }

    /// `recv_fifo_put()`. Receive overruns do not overwrite FIFO contents.
    fn recv_fifo_put(s: &mut SerialState, chr: u8) {
        if !s.recv_fifo.is_full() {
            s.recv_fifo.push(chr);
        } else {
            s.lsr |= UART_LSR_OE;
        }
    }

    /// `serial_update_irq()`.
    fn update_irq(&self, s: &mut SerialState) {
        let mut tmp_iir = UART_IIR_NO_INT;

        if (s.ier & UART_IER_RLSI) != 0 && (s.lsr & UART_LSR_INT_ANY) != 0 {
            tmp_iir = UART_IIR_RLSI;
        } else if (s.ier & UART_IER_RDI) != 0 && s.timeout_ipending {
            // IER.RDI can mask this interrupt. This is not in the specification but is
            // observed on existing hardware.
            tmp_iir = UART_IIR_CTI;
        } else if (s.ier & UART_IER_RDI) != 0
            && (s.lsr & UART_LSR_DR) != 0
            && ((s.fcr & UART_FCR_FE) == 0
                || s.recv_fifo.num_used() >= usize::from(s.recv_fifo_itl))
        {
            tmp_iir = UART_IIR_RDI;
        } else if (s.ier & UART_IER_THRI) != 0 && s.thr_ipending {
            tmp_iir = UART_IIR_THRI;
        } else if (s.ier & UART_IER_MSI) != 0 && (s.msr & UART_MSR_ANY_DELTA) != 0 {
            tmp_iir = UART_IIR_MSI;
        }

        s.iir = tmp_iir | (s.iir & 0xF0);

        if tmp_iir != UART_IIR_NO_INT {
            self.irq.raise();
        } else {
            self.irq.lower();
        }
    }

    /// `serial_update_parameters()`.
    fn update_parameters(&self, s: &mut SerialState) {
        // Start bit.
        let mut frame_size = 1;
        let parity = if (s.lcr & UART_LCR_PEN) != 0 {
            // Parity bit.
            frame_size += 1;
            if (s.lcr & UART_LCR_EPS) != 0 { 'E' } else { 'O' }
        } else {
            'N'
        };
        let stop_bits = if (s.lcr & UART_LCR_NSTB) != 0 { 2 } else { 1 };

        let data_bits = i32::from(s.lcr & UART_LCR_WLS) + 5;
        frame_size += data_bits + stop_bits;
        // Zero divisor should give about 3500 baud. The C code does this in float.
        let speed: f32 =
            if s.divider == 0 { 3500.0 } else { s.baudbase as f32 / f32::from(s.divider) };
        let ssp = SerialParams { speed: speed as i32, parity, data_bits, stop_bits };
        s.char_transmit_time = ((NANOSECONDS_PER_SECOND as f32 / speed) * frame_size as f32) as u64;
        if let Some(be) = self.backend() {
            be.set_params(&ssp);
        }
    }

    /// `serial_update_msl()`, also the modem status poll timer.
    fn update_msl(&self, s: &mut SerialState) {
        self.modem_status_poll.del();

        let Some(flags) = self.backend().and_then(|be| be.get_tiocm()) else {
            s.poll_msl = -1;
            return;
        };

        let omsr = s.msr;
        let set = |msr: u8, line: u32, bit: u8| {
            if (flags & line) != 0 { msr | bit } else { msr & !bit }
        };
        s.msr = set(s.msr, CHR_TIOCM_CTS, UART_MSR_CTS);
        s.msr = set(s.msr, CHR_TIOCM_DSR, UART_MSR_DSR);
        s.msr = set(s.msr, CHR_TIOCM_CAR, UART_MSR_DCD);
        s.msr = set(s.msr, CHR_TIOCM_RI, UART_MSR_RI);

        if s.msr != omsr {
            // Set delta bits.
            s.msr |= (s.msr >> 4) ^ (omsr >> 4);
            // UART_MSR_TERI only if change was from 1 to 0.
            if (s.msr & UART_MSR_TERI) != 0 && (omsr & UART_MSR_RI) == 0 {
                s.msr &= !UART_MSR_TERI;
            }
            self.update_irq(s);
        }

        // The real 16550A apparently has a 250ns response latency to line status changes.
        // We poll only every 10ms, and only if MSI interrupts are turned on.
        if s.poll_msl != 0 {
            self.modem_status_poll.modify(self.clock.get_ns() + NANOSECONDS_PER_SECOND / 100);
        }
    }

    /// `serial_watch_cb()`: the backend can take output again after [`SerialBackend::write`]
    /// returned 0.
    pub fn write_ready(&self) {
        let mut s = self.lock();
        if s.tsr_retry > 0 {
            self.xmit(&mut s);
        }
    }

    /// `serial_xmit()`.
    fn xmit(&self, s: &mut SerialState) {
        loop {
            assert!((s.lsr & UART_LSR_TEMT) == 0);
            if s.tsr_retry == 0 {
                assert!((s.lsr & UART_LSR_THRE) == 0);

                if (s.fcr & UART_FCR_FE) != 0 {
                    assert!(!s.xmit_fifo.is_empty());
                    s.tsr = s.xmit_fifo.pop();
                    if s.xmit_fifo.is_empty() {
                        s.lsr |= UART_LSR_THRE;
                    }
                } else {
                    s.tsr = s.thr;
                    s.lsr |= UART_LSR_THRE;
                }
                if (s.lsr & UART_LSR_THRE) != 0 && !s.thr_ipending {
                    s.thr_ipending = true;
                    self.update_irq(s);
                }
            }

            if (s.mcr & UART_MCR_LOOP) != 0 {
                // In loopback mode, say that we just received a char.
                let tsr = s.tsr;
                self.receive1(s, &[tsr]);
            } else {
                // With no chardev the write gives 0 and there is no watch to retry with, so the
                // byte is dropped as in QEMU.
                let rc = self.backend().map(|be| be.write(&[s.tsr]));
                if rc == Some(0) && s.tsr_retry < MAX_XMIT_RETRY {
                    s.tsr_retry += 1;
                    return;
                }
            }
            s.tsr_retry = 0;

            // Transmit another byte if it is already available. That is only possible when the
            // FIFO is enabled and not empty.
            if (s.lsr & UART_LSR_THRE) != 0 {
                break;
            }
        }

        s.last_xmit_ts = self.clock.get_ns();
        s.lsr |= UART_LSR_TEMT;
    }

    /// `serial_write_fcr()`. `val` only has the bits that are supposed to stick.
    fn write_fcr(s: &mut SerialState, val: u8) {
        s.fcr = val;

        if (val & UART_FCR_FE) != 0 {
            s.iir |= UART_IIR_FE;
            // Set the recv_fifo trigger level.
            s.recv_fifo_itl = match val & 0xC0 {
                UART_FCR_ITL_1 => 1,
                UART_FCR_ITL_2 => 4,
                UART_FCR_ITL_3 => 8,
                _ => 14,
            };
        } else {
            s.iir &= !UART_IIR_FE;
        }
    }

    /// `serial_update_tiocm()`.
    fn update_tiocm(&self, s: &SerialState) {
        let Some(be) = self.backend() else {
            return;
        };
        let mut flags = be.get_tiocm().unwrap_or(0);

        flags &= !(CHR_TIOCM_RTS | CHR_TIOCM_DTR);

        if (s.mcr & UART_MCR_RTS) != 0 {
            flags |= CHR_TIOCM_RTS;
        }
        if (s.mcr & UART_MCR_DTR) != 0 {
            flags |= CHR_TIOCM_DTR;
        }

        be.set_tiocm(flags);
    }

    /// `serial_ioport_write()`. `addr` is the register offset, 0 to 7.
    pub fn ioport_write(&self, addr: u64, val: u8) {
        debug_assert!(addr < SERIAL_IO_SIZE);
        let mut guard = self.lock();
        let s = &mut *guard;
        match addr & 7 {
            1 => {
                if (s.lcr & UART_LCR_DLAB) != 0 {
                    s.divider = (s.divider & 0x00ff) | (u16::from(val) << 8);
                    self.update_parameters(s);
                } else {
                    let changed = (s.ier ^ val) & 0x0f;
                    s.ier = val & 0x0f;
                    // If the backend is a real serial port, poll its modem status lines only
                    // while UART_IER_MSI is set.
                    if (changed & UART_IER_MSI) != 0 && s.poll_msl >= 0 {
                        if (s.ier & UART_IER_MSI) != 0 {
                            s.poll_msl = 1;
                            self.update_msl(s);
                        } else {
                            self.modem_status_poll.del();
                            s.poll_msl = 0;
                        }
                    }

                    // Turning on the THRE interrupt in IER can trigger the interrupt if
                    // LSR.THRE=1, even if it had been masked before by reading IIR. This is
                    // not in the datasheet, but Windows relies on it. Like Bochs we only
                    // resample on the rising edge; Windows toggles IER to all zeroes and back.
                    //
                    // If IER.THRI is zero, thr_ipending is not used, so clear it.
                    if (changed & UART_IER_THRI) != 0 {
                        s.thr_ipending =
                            (s.ier & UART_IER_THRI) != 0 && (s.lsr & UART_LSR_THRE) != 0;
                    }

                    if changed != 0 {
                        self.update_irq(s);
                    }
                }
            }
            2 => {
                let mut val = val;
                // Did the enable/disable flag change? If so, make sure the FIFOs get flushed.
                if ((val ^ s.fcr) & UART_FCR_FE) != 0 {
                    val |= UART_FCR_XFR | UART_FCR_RFR;
                }

                // FIFO clear.
                if (val & UART_FCR_RFR) != 0 {
                    s.lsr &= !(UART_LSR_DR | UART_LSR_BI);
                    self.fifo_timeout_timer.del();
                    s.timeout_ipending = false;
                    s.recv_fifo.reset();
                }

                if (val & UART_FCR_XFR) != 0 {
                    s.lsr |= UART_LSR_THRE;
                    s.thr_ipending = true;
                    s.xmit_fifo.reset();
                }

                Self::write_fcr(s, val & 0xC9);
                self.update_irq(s);
            }
            3 => {
                s.lcr = val;
                self.update_parameters(s);
                let break_enable = (val & UART_LCR_SB) != 0;
                if break_enable != s.last_break_enable {
                    s.last_break_enable = break_enable;
                    if let Some(be) = self.backend() {
                        be.set_break(break_enable);
                    }
                }
            }
            4 => {
                let old_mcr = s.mcr;
                s.mcr = val & 0x1f;
                if (val & UART_MCR_LOOP) != 0 {
                    return;
                }

                if s.poll_msl >= 0 && old_mcr != s.mcr {
                    self.update_tiocm(s);
                    // Update the modem status after a one character send wait, since there may
                    // be a response from the other end of the line.
                    self.modem_status_poll
                        .modify(self.clock.get_ns() + s.char_transmit_time as i64);
                }
            }
            5 | 6 => {}
            7 => s.scr = val,
            _ => {
                if (s.lcr & UART_LCR_DLAB) != 0 {
                    s.divider = (s.divider & 0xff00) | u16::from(val);
                    self.update_parameters(s);
                } else {
                    s.thr = val;
                    if (s.fcr & UART_FCR_FE) != 0 {
                        // Transmit overruns overwrite data, so make space if needed.
                        if s.xmit_fifo.is_full() {
                            s.xmit_fifo.pop();
                        }
                        s.xmit_fifo.push(s.thr);
                    }
                    s.thr_ipending = false;
                    s.lsr &= !UART_LSR_THRE;
                    s.lsr &= !UART_LSR_TEMT;
                    self.update_irq(s);
                    if s.tsr_retry == 0 {
                        self.xmit(s);
                    }
                }
            }
        }
    }

    /// `serial_ioport_read()`. `addr` is the register offset, 0 to 7.
    pub fn ioport_read(&self, addr: u64) -> u8 {
        debug_assert!(addr < SERIAL_IO_SIZE);
        let mut guard = self.lock();
        let s = &mut *guard;
        match addr & 7 {
            1 => {
                if (s.lcr & UART_LCR_DLAB) != 0 {
                    (s.divider >> 8) as u8
                } else {
                    s.ier
                }
            }
            2 => {
                let ret = s.iir;
                if (ret & UART_IIR_ID) == UART_IIR_THRI {
                    s.thr_ipending = false;
                    self.update_irq(s);
                }
                ret
            }
            3 => s.lcr,
            4 => s.mcr,
            5 => {
                let ret = s.lsr;
                // Clear break and overrun interrupts.
                if (s.lsr & (UART_LSR_BI | UART_LSR_OE)) != 0 {
                    s.lsr &= !(UART_LSR_BI | UART_LSR_OE);
                    self.update_irq(s);
                }
                ret
            }
            6 => {
                if (s.mcr & UART_MCR_LOOP) != 0 {
                    // In loopback, the modem output pins are connected to the inputs.
                    let mut ret = (s.mcr & 0x0c) << 4;
                    ret |= (s.mcr & 0x02) << 3;
                    ret |= (s.mcr & 0x01) << 5;
                    ret
                } else {
                    if s.poll_msl >= 0 {
                        self.update_msl(s);
                    }
                    let ret = s.msr;
                    // Clear delta bits and the msr interrupt after the read, if they were set.
                    if (s.msr & UART_MSR_ANY_DELTA) != 0 {
                        s.msr &= 0xF0;
                        self.update_irq(s);
                    }
                    ret
                }
            }
            7 => s.scr,
            _ => {
                if (s.lcr & UART_LCR_DLAB) != 0 {
                    return s.divider as u8;
                }
                let ret;
                if (s.fcr & UART_FCR_FE) != 0 {
                    ret = if s.recv_fifo.is_empty() { 0 } else { s.recv_fifo.pop() };
                    if s.recv_fifo.is_empty() {
                        s.lsr &= !(UART_LSR_DR | UART_LSR_BI);
                    } else {
                        self.fifo_timeout_timer
                            .modify(self.clock.get_ns() + s.char_transmit_time as i64 * 4);
                    }
                    s.timeout_ipending = false;
                } else {
                    ret = s.rbr;
                    s.lsr &= !(UART_LSR_DR | UART_LSR_BI);
                }
                self.update_irq(s);
                if (s.mcr & UART_MCR_LOOP) == 0 {
                    // In loopback mode, don't receive any data.
                    if let Some(be) = self.backend() {
                        be.accept_input();
                    }
                }
                ret
            }
        }
    }

    /// `serial_can_receive()`: how many bytes [`Serial::receive`] takes now.
    pub fn can_receive(&self) -> usize {
        let s = self.lock();
        if (s.fcr & UART_FCR_FE) != 0 {
            if !s.recv_fifo.is_full() {
                // Advertise (itl - count) bytes when count < ITL, and 1 above it. Advertising
                // UART_FIFO_LENGTH - count would almost always fill the FIFO before the guest
                // has a chance to respond, overriding the ITL the guest has set.
                let used = s.recv_fifo.num_used();
                let itl = usize::from(s.recv_fifo_itl);
                if used <= itl { itl - used } else { 1 }
            } else {
                0
            }
        } else {
            usize::from((s.lsr & UART_LSR_DR) == 0)
        }
    }

    /// `serial_receive_break()`, what `CHR_EVENT_BREAK` does.
    pub fn receive_break(&self) {
        let mut s = self.lock();
        s.rbr = 0;
        // When LSR.DR is set a null byte is pushed into the FIFO.
        Self::recv_fifo_put(&mut s, 0);
        s.lsr |= UART_LSR_BI | UART_LSR_DR;
        self.update_irq(&mut s);
    }

    /// `fifo_timeout_int()`: there is data in the receive FIFO and RBR has not been read for
    /// 4 character times.
    fn fifo_timeout_int(&self) {
        let mut s = self.lock();
        if !s.recv_fifo.is_empty() {
            s.timeout_ipending = true;
            self.update_irq(&mut s);
        }
    }

    /// `serial_receive1()`: bytes from the chardev. Send no more than
    /// [`Serial::can_receive`] allows.
    pub fn receive(&self, buf: &[u8]) {
        if buf.is_empty() {
            return;
        }
        let mut s = self.lock();
        self.receive1(&mut s, buf);
    }

    /// `serial_receive1()` with the lock held.
    fn receive1(&self, s: &mut SerialState, buf: &[u8]) {
        if (s.fcr & UART_FCR_FE) != 0 {
            for &b in buf {
                Self::recv_fifo_put(s, b);
            }
            s.lsr |= UART_LSR_DR;
            // Call the timeout receive callback in 4 char transmit times.
            self.fifo_timeout_timer.modify(self.clock.get_ns() + s.char_transmit_time as i64 * 4);
        } else {
            if (s.lsr & UART_LSR_DR) != 0 {
                s.lsr |= UART_LSR_OE;
            }
            s.rbr = buf[0];
            s.lsr |= UART_LSR_DR;
        }
        self.update_irq(s);
    }

    /// The state `vmstate_serial` sends, with `serial_pre_save()` applied.
    pub fn vmstate_save(&self) -> SerialVmState {
        let s = self.lock();
        SerialVmState {
            divider: s.divider,
            rbr: s.rbr,
            ier: s.ier,
            iir: s.iir,
            lcr: s.lcr,
            mcr: s.mcr,
            lsr: s.lsr,
            msr: s.msr,
            scr: s.scr,
            fcr_vmstate: s.fcr,
            thr_ipending: i32::from(s.thr_ipending),
            tsr_retry: s.tsr_retry,
            thr: s.thr,
            tsr: s.tsr,
            recv_fifo: s.recv_fifo.to_vmstate(),
            xmit_fifo: s.xmit_fifo.to_vmstate(),
            fifo_timeout_timer: timer_vmstate(&self.fifo_timeout_timer),
            timeout_ipending: i32::from(s.timeout_ipending),
            poll_msl: s.poll_msl,
            modem_status_poll: timer_vmstate(&self.modem_status_poll),
        }
    }

    /// Loads what `vmstate_serial` carried, then does `serial_post_load()`. The caller has
    /// already reset `fcr_vmstate` for streams older than version 3.
    ///
    /// The timers are armed at the loaded expiry times, as `timer_get()` does. The interrupt
    /// line is left alone, as in QEMU: the interrupt controllers carry their input levels.
    /// QEMU adds a write watch when a byte is stuck in the transmit shift register; there is
    /// no watch here, so the byte is retried right away, as [`Serial::set_backend`] does.
    pub fn vmstate_load(&self, v: &SerialVmState) -> Result<()> {
        // serial_post_load() checks these before anything changes.
        if v.tsr_retry > 0 {
            if (v.lsr & UART_LSR_TEMT) != 0 {
                return Err(Error::generic(format!(
                    "inconsistent state in serial device (tsr empty, tsr_retry={}",
                    v.tsr_retry
                )));
            }
        } else if (v.lsr & UART_LSR_TEMT) == 0 {
            return Err(Error::generic(
                "inconsistent state in serial device (tsr not empty, tsr_retry=0".to_string(),
            ));
        }
        let mut recv_fifo = Fifo8::new();
        recv_fifo.load_vmstate(&v.recv_fifo, "recv_fifo")?;
        let mut xmit_fifo = Fifo8::new();
        xmit_fifo.load_vmstate(&v.xmit_fifo, "xmit_fifo")?;

        let mut guard = self.lock();
        let s = &mut *guard;
        s.divider = v.divider;
        s.rbr = v.rbr;
        s.ier = v.ier;
        s.iir = v.iir;
        s.lcr = v.lcr;
        s.mcr = v.mcr;
        s.lsr = v.lsr;
        s.msr = v.msr;
        s.scr = v.scr;
        s.thr_ipending = if v.thr_ipending == -1 {
            (s.iir & UART_IIR_ID) == UART_IIR_THRI
        } else {
            v.thr_ipending != 0
        };
        s.tsr_retry = v.tsr_retry.min(MAX_XMIT_RETRY);
        s.thr = v.thr;
        s.tsr = v.tsr;
        s.recv_fifo = recv_fifo;
        s.xmit_fifo = xmit_fifo;
        timer_load(&self.fifo_timeout_timer, v.fifo_timeout_timer);
        s.timeout_ipending = v.timeout_ipending != 0;
        s.poll_msl = v.poll_msl;
        timer_load(&self.modem_status_poll, v.modem_status_poll);

        s.last_break_enable = (s.lcr & UART_LCR_SB) != 0;
        // Initialize fcr via setter to perform essential side-effects.
        Self::write_fcr(s, v.fcr_vmstate);
        self.update_parameters(s);

        if s.tsr_retry > 0 {
            self.xmit(s);
        }
        Ok(())
    }

    /// `serial_reset()`.
    pub fn reset(&self) {
        let mut guard = self.lock();
        let s = &mut *guard;

        s.rbr = 0;
        s.ier = 0;
        s.iir = UART_IIR_NO_INT;
        s.lcr = 0;
        s.lsr = UART_LSR_TEMT | UART_LSR_THRE;
        s.msr = UART_MSR_DCD | UART_MSR_DSR | UART_MSR_CTS;
        // Default to 9600 baud, 1 start bit, 8 data bits, 1 stop bit, no parity.
        s.divider = 0x0C;
        s.mcr = UART_MCR_OUT2;
        s.scr = 0;
        s.tsr_retry = 0;
        s.char_transmit_time = (NANOSECONDS_PER_SECOND as u64 / 9600) * 10;
        s.poll_msl = 0;

        s.timeout_ipending = false;
        self.fifo_timeout_timer.del();
        self.modem_status_poll.del();

        s.recv_fifo.reset();
        s.xmit_fifo.reset();

        s.last_xmit_ts = self.clock.get_ns();

        s.thr_ipending = false;
        s.last_break_enable = false;
        self.irq.lower();

        self.update_msl(s);
        s.msr &= !UART_MSR_ANY_DELTA;
    }

    /// `serial_be_change()`: switches to another backend, or none, and brings it up to date.
    pub fn set_backend(&self, backend: Option<Arc<dyn SerialBackend>>) {
        *self.backend.write().unwrap_or_else(|p| p.into_inner()) = backend;
        let mut guard = self.lock();
        let s = &mut *guard;

        self.update_parameters(s);

        if let Some(be) = self.backend() {
            be.set_break(s.last_break_enable);
        }

        s.poll_msl = if (s.ier & UART_IER_MSI) != 0 { 1 } else { 0 };
        self.update_msl(s);

        if s.poll_msl >= 0 && (s.mcr & UART_MCR_LOOP) == 0 {
            self.update_tiocm(s);
        }

        // QEMU moves its write watch to the new backend. We have no watch, so try the stuck
        // byte right away.
        if s.tsr_retry > 0 {
            self.xmit(s);
        }
    }
}

/// `serial_io_ops`.
impl MmioOps for Serial {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.ioport_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.ioport_write(offset, value as u8);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::default().allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

/// `TYPE_ISA_SERIAL`.
pub const TYPE_ISA_SERIAL: &str = "isa-serial";

/// `MAX_ISA_SERIAL_PORTS`.
pub const MAX_ISA_SERIAL_PORTS: usize = 4;

/// `isa_serial_io`: the COM1 to COM4 port bases.
pub const ISA_SERIAL_IO: [u32; MAX_ISA_SERIAL_PORTS] = [0x3f8, 0x2f8, 0x3e8, 0x2e8];

/// `isa_serial_irq`: the COM1 to COM4 ISA IRQs.
pub const ISA_SERIAL_IRQ: [u32; MAX_ISA_SERIAL_PORTS] = [4, 3, 4, 3];

/// `ISASerialState`, the `isa-serial` device. The `index`, `iobase` and `irq` properties are
/// `u32::MAX` (QEMU's -1) until [`IsaSerial::realize`] fills them in.
#[derive(Debug)]
pub struct IsaSerial {
    pub index: u32,
    pub iobase: u32,
    pub isairq: u32,
    state: Arc<Serial>,
}

impl IsaSerial {
    /// `serial_isa_initfn()` with the property defaults.
    pub fn new(state: Arc<Serial>) -> Self {
        IsaSerial { index: u32::MAX, iobase: u32::MAX, isairq: u32::MAX, state }
    }

    /// `serial_isa_realizefn()`. `next_index` is the static counter that numbers the ports
    /// created without an index, and `get_irq` is `isa_get_irq()`. The caller maps
    /// [`IsaSerial::state`] at [`IsaSerial::iobase`] with a size of [`SERIAL_IO_SIZE`].
    pub fn realize(
        &mut self,
        next_index: &mut u32,
        get_irq: impl FnOnce(u32) -> IrqLine,
    ) -> Result<()> {
        if self.index == u32::MAX {
            self.index = *next_index;
        }
        let Some(&io) = ISA_SERIAL_IO.get(self.index as usize) else {
            return Err(Error::generic(format!(
                "Max. supported number of ISA serial ports is {MAX_ISA_SERIAL_PORTS}."
            )));
        };
        if self.iobase == u32::MAX {
            self.iobase = io;
        }
        if self.isairq == u32::MAX {
            self.isairq = ISA_SERIAL_IRQ[self.index as usize];
        }
        *next_index += 1;

        self.state.irq().connect(get_irq(self.isairq));
        Ok(())
    }

    /// The 16550 behind the port, also the ops of its I/O region.
    pub fn state(&self) -> &Arc<Serial> {
        &self.state
    }

    /// `isa_serial_set_iobase()`. Moving the region is up to the caller.
    pub fn set_iobase(&mut self, iobase: u32) {
        self.iobase = iobase;
    }
}
