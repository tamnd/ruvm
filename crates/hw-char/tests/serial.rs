// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the 16550A, plus the COM1 sequences QEMU's tests drive through
//! port 0x3f8 (tests/qtest/migration/i386/a-b-bootblock.S and tests/tcg/alpha/system/boot.S).

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_char::serial::*;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MmioOps};

const RBR: u64 = 0;
const THR: u64 = 0;
const DLL: u64 = 0;
const IER: u64 = 1;
const DLM: u64 = 1;
const IIR: u64 = 2;
const FCR: u64 = 2;
const LCR: u64 = 3;
const MCR: u64 = 4;
const LSR: u64 = 5;
const MSR: u64 = 6;
const SCR: u64 = 7;

/// A chardev that records what the UART does to it.
#[derive(Default)]
struct Recorder {
    out: Mutex<Vec<u8>>,
    params: Mutex<Vec<SerialParams>>,
    breaks: Mutex<Vec<bool>>,
    tiocm: Mutex<Option<u32>>,
    set_tiocm: Mutex<Vec<u32>>,
    /// When set, writes take nothing.
    busy: AtomicBool,
    accepted: AtomicI32,
}

impl SerialBackend for Recorder {
    fn write(&self, bytes: &[u8]) -> usize {
        if self.busy.load(Ordering::SeqCst) {
            return 0;
        }
        self.out.lock().unwrap().extend_from_slice(bytes);
        bytes.len()
    }

    fn set_params(&self, params: &SerialParams) {
        self.params.lock().unwrap().push(*params);
    }

    fn set_break(&self, enable: bool) {
        self.breaks.lock().unwrap().push(enable);
    }

    fn get_tiocm(&self) -> Option<u32> {
        *self.tiocm.lock().unwrap()
    }

    fn set_tiocm(&self, flags: u32) {
        self.set_tiocm.lock().unwrap().push(flags);
    }

    fn accept_input(&self) {
        self.accepted.fetch_add(1, Ordering::SeqCst);
    }
}

struct Rig {
    clock: Arc<Clock>,
    serial: Arc<Serial>,
    be: Arc<Recorder>,
    level: Arc<AtomicI32>,
}

impl Rig {
    fn new() -> Self {
        Self::with_backend(Recorder::default())
    }

    fn with_backend(be: Recorder) -> Self {
        let clock = Clock::manual(ClockType::Virtual);
        let be = Arc::new(be);
        let serial =
            Serial::new(clock.clone(), SERIAL_BAUDBASE_DEFAULT, Some(be.clone() as Arc<_>));
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        serial.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        Rig { clock, serial, be, level }
    }

    fn rd(&self, reg: u64) -> u8 {
        self.serial.ioport_read(reg)
    }

    fn wr(&self, reg: u64, val: u8) {
        self.serial.ioport_write(reg, val);
    }

    fn irq(&self) -> i32 {
        self.level.load(Ordering::SeqCst)
    }

    fn out(&self) -> Vec<u8> {
        self.be.out.lock().unwrap().clone()
    }

    fn advance(&self, ns: i64) {
        let now = self.clock.get_ns();
        self.clock.advance_to(now + ns);
    }
}

#[test]
fn reset_values() {
    let r = Rig::new();
    assert_eq!(r.rd(IER), 0);
    assert_eq!(r.rd(IIR), UART_IIR_NO_INT);
    assert_eq!(r.rd(LCR), 0);
    assert_eq!(r.rd(MCR), UART_MCR_OUT2);
    assert_eq!(r.rd(LSR), UART_LSR_TEMT | UART_LSR_THRE);
    assert_eq!(r.rd(MSR), UART_MSR_DCD | UART_MSR_DSR | UART_MSR_CTS);
    assert_eq!(r.rd(SCR), 0);
    assert_eq!(r.rd(RBR), 0);
    assert_eq!(r.serial.char_transmit_time(), 1_041_660);
    assert_eq!(r.irq(), 0);
    r.wr(LCR, UART_LCR_DLAB);
    assert_eq!(r.rd(DLL), 0x0c);
    assert_eq!(r.rd(DLM), 0);
}

#[test]
fn reset_restores_registers() {
    let r = Rig::new();
    r.wr(IER, 0x0f);
    r.wr(LCR, 0x1b);
    r.wr(MCR, 0x1f);
    r.wr(SCR, 0x55);
    r.serial.reset();
    assert_eq!(r.rd(IER), 0);
    assert_eq!(r.rd(LCR), 0);
    assert_eq!(r.rd(MCR), UART_MCR_OUT2);
    assert_eq!(r.rd(SCR), 0);
    assert_eq!(r.irq(), 0);
}

#[test]
fn scratch_register() {
    let r = Rig::new();
    for v in [0x00, 0x55, 0xaa, 0xff] {
        r.wr(SCR, v);
        assert_eq!(r.rd(SCR), v);
    }
}

#[test]
fn divisor_latch_and_parameters() {
    let r = Rig::new();
    r.wr(LCR, UART_LCR_DLAB | 0x03);
    r.wr(DLL, 0x01);
    r.wr(DLM, 0x00);
    assert_eq!(r.rd(DLL), 1);
    assert_eq!(r.rd(DLM), 0);
    r.wr(LCR, 0x03);
    // With DLAB clear, offset 1 is IER again and offset 0 no longer shows the divisor.
    assert_eq!(r.rd(IER), 0);
    let p = *r.be.params.lock().unwrap().last().unwrap();
    assert_eq!(p, SerialParams { speed: 115200, parity: 'N', data_bits: 8, stop_bits: 1 });
    // 10 bit frames at 115200 baud.
    assert_eq!(r.serial.char_transmit_time(), 86_805);

    r.wr(LCR, UART_LCR_DLAB);
    r.wr(DLM, 0x01);
    r.wr(DLL, 0x80);
    assert_eq!(r.rd(DLM), 1);
    assert_eq!(r.rd(DLL), 0x80);
    // 7 data bits, even parity, 2 stop bits.
    r.wr(LCR, UART_LCR_PEN | UART_LCR_EPS | UART_LCR_NSTB | 0x02);
    let p = *r.be.params.lock().unwrap().last().unwrap();
    assert_eq!(p, SerialParams { speed: 300, parity: 'E', data_bits: 7, stop_bits: 2 });
    r.wr(LCR, UART_LCR_PEN);
    let p = *r.be.params.lock().unwrap().last().unwrap();
    assert_eq!((p.parity, p.data_bits, p.stop_bits), ('O', 5, 1));

    // A zero divisor gives about 3500 baud.
    r.wr(LCR, UART_LCR_DLAB);
    r.wr(DLL, 0);
    r.wr(DLM, 0);
    assert_eq!(r.be.params.lock().unwrap().last().unwrap().speed, 3500);
}

#[test]
fn break_goes_to_backend_on_change_only() {
    let r = Rig::new();
    r.wr(LCR, 0x03);
    r.wr(LCR, UART_LCR_SB | 0x03);
    r.wr(LCR, UART_LCR_SB | 0x03);
    r.wr(LCR, 0x03);
    assert_eq!(*r.be.breaks.lock().unwrap(), [true, false]);
}

#[test]
fn receive_break() {
    let r = Rig::new();
    r.wr(IER, UART_IER_RLSI);
    r.serial.receive_break();
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_RLSI);
    // Reading LSR clears BI and with it the line status interrupt.
    assert_eq!(r.rd(LSR) & (UART_LSR_BI | UART_LSR_DR), UART_LSR_BI | UART_LSR_DR);
    assert_eq!(r.rd(LSR) & (UART_LSR_BI | UART_LSR_DR), UART_LSR_DR);
    assert_eq!(r.irq(), 0);
    assert_eq!(r.rd(RBR), 0);
}

#[test]
fn transmit_reaches_backend() {
    let r = Rig::new();
    for &b in b"hello" {
        assert_ne!(r.rd(LSR) & UART_LSR_THRE, 0);
        r.wr(THR, b);
    }
    assert_eq!(r.out(), b"hello");
    assert_eq!(r.rd(LSR), UART_LSR_TEMT | UART_LSR_THRE);

    // The same with the FIFO on.
    r.wr(FCR, UART_FCR_FE);
    r.wr(THR, b'!');
    assert_eq!(r.out(), b"hello!");
}

#[test]
fn transmit_retry() {
    let be = Recorder::default();
    be.busy.store(true, Ordering::SeqCst);
    let r = Rig::with_backend(be);
    r.wr(THR, b'x');
    // The byte sits in TSR: THR is free again but the transmitter is not empty.
    assert_eq!(r.rd(LSR) & (UART_LSR_TEMT | UART_LSR_THRE), UART_LSR_THRE);
    r.serial.write_ready();
    assert_eq!(r.rd(LSR) & UART_LSR_TEMT, 0);
    r.be.busy.store(false, Ordering::SeqCst);
    r.serial.write_ready();
    assert_eq!(r.out(), b"x");
    assert_eq!(r.rd(LSR), UART_LSR_TEMT | UART_LSR_THRE);

    // After MAX_XMIT_RETRY attempts the byte is dropped.
    r.be.busy.store(true, Ordering::SeqCst);
    r.wr(THR, b'y');
    for _ in 0..MAX_XMIT_RETRY {
        r.serial.write_ready();
    }
    assert_eq!(r.rd(LSR), UART_LSR_TEMT | UART_LSR_THRE);
    assert_eq!(r.out(), b"x");
}

#[test]
fn no_backend_drops_output() {
    let clock = Clock::manual(ClockType::Virtual);
    let s = Serial::new(clock, SERIAL_BAUDBASE_DEFAULT, None);
    s.ioport_write(THR, b'a');
    assert_eq!(s.ioport_read(LSR), UART_LSR_TEMT | UART_LSR_THRE);
}

#[test]
fn thre_interrupt_raise_and_clear_on_iir_read() {
    let r = Rig::new();
    // Enabling THRI with THR empty raises the interrupt right away.
    r.wr(IER, UART_IER_THRI);
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_THRI);
    // Reading IIR with THRI pending clears it.
    assert_eq!(r.irq(), 0);
    assert_eq!(r.rd(IIR), UART_IIR_NO_INT);

    // Writing THR sends the byte and raises THRI again.
    r.wr(THR, b'a');
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_THRI);
    assert_eq!(r.irq(), 0);

    // Toggling IER.THRI resamples LSR.THRE.
    r.wr(IER, 0);
    r.wr(IER, UART_IER_THRI);
    assert_eq!(r.irq(), 1);
    r.wr(IER, 0);
    assert_eq!(r.irq(), 0);
}

#[test]
fn receive_without_fifo() {
    let r = Rig::new();
    r.wr(IER, UART_IER_RDI | UART_IER_RLSI);
    assert_eq!(r.serial.can_receive(), 1);
    r.serial.receive(b"a");
    assert_eq!(r.serial.can_receive(), 0);
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_RDI);
    assert_ne!(r.rd(LSR) & UART_LSR_DR, 0);
    // A second byte overruns.
    r.serial.receive(b"b");
    assert_eq!(r.rd(IIR), UART_IIR_RLSI);
    assert_eq!(r.rd(LSR) & (UART_LSR_OE | UART_LSR_DR), UART_LSR_OE | UART_LSR_DR);
    assert_eq!(r.rd(IIR), UART_IIR_RDI);
    assert_eq!(r.rd(RBR), b'b');
    assert_eq!(r.rd(LSR) & UART_LSR_DR, 0);
    assert_eq!(r.irq(), 0);
    assert_eq!(r.be.accepted.load(Ordering::SeqCst), 1);
}

#[test]
fn fifo_enable_and_trigger_levels() {
    for (itl, level) in
        [(UART_FCR_ITL_1, 1), (UART_FCR_ITL_2, 4), (UART_FCR_ITL_3, 8), (UART_FCR_ITL_4, 14)]
    {
        let r = Rig::new();
        r.wr(FCR, itl | UART_FCR_FE);
        assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_NO_INT);
        r.wr(IER, UART_IER_RDI);
        assert_eq!(r.serial.can_receive(), level);

        let data: Vec<u8> = (0..level as u8).collect();
        r.serial.receive(&data[..level - 1]);
        if level > 1 {
            // Below the trigger level: data ready but no interrupt yet.
            assert_ne!(r.rd(LSR) & UART_LSR_DR, 0);
            assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_NO_INT);
            assert_eq!(r.irq(), 0);
            assert_eq!(r.serial.can_receive(), 1);
        }
        r.serial.receive(&data[level - 1..]);
        assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_RDI);
        assert_eq!(r.irq(), 1);
        // At the trigger level nothing more is advertised, above it one byte at a time.
        assert_eq!(r.serial.can_receive(), 0);
        r.serial.receive(b"+");
        assert_eq!(r.serial.can_receive(), 1);

        for &b in &data {
            assert_eq!(r.rd(RBR), b);
        }
        assert_eq!(r.rd(RBR), b'+');
        assert_eq!(r.rd(LSR) & UART_LSR_DR, 0);
        assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_NO_INT);
        assert_eq!(r.irq(), 0);
    }
}

#[test]
fn fifo_overrun_and_reset() {
    let r = Rig::new();
    r.wr(FCR, UART_FCR_FE);
    let data: Vec<u8> = (0..20).collect();
    r.serial.receive(&data);
    assert_eq!(r.serial.can_receive(), 0);
    assert_ne!(r.rd(LSR) & UART_LSR_OE, 0);
    // Overruns do not overwrite the FIFO.
    assert_eq!(r.rd(RBR), 0);
    // Clearing the receive FIFO drops the rest.
    r.wr(FCR, UART_FCR_FE | UART_FCR_RFR);
    assert_eq!(r.rd(LSR) & UART_LSR_DR, 0);
    assert_eq!(r.rd(RBR), 0);
    // Turning the FIFO off flushes it and clears the IIR bits.
    r.serial.receive(b"ab");
    r.wr(FCR, 0);
    assert_eq!(r.rd(IIR), UART_IIR_NO_INT);
    assert_eq!(r.rd(LSR) & UART_LSR_DR, 0);
}

#[test]
fn receive_timeout_fires() {
    let r = Rig::new();
    r.wr(FCR, UART_FCR_ITL_4 | UART_FCR_FE);
    r.wr(IER, UART_IER_RDI);
    r.serial.receive(b"abc");
    assert_eq!(r.irq(), 0);
    let ctt = r.serial.char_transmit_time() as i64;
    r.advance(ctt * 4 - 1);
    assert_eq!(r.irq(), 0);
    r.advance(1);
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_CTI);
    // Reading a byte clears the timeout and rearms it.
    assert_eq!(r.rd(RBR), b'a');
    assert_eq!(r.irq(), 0);
    r.advance(ctt * 4);
    assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_CTI);
    assert_eq!(r.rd(RBR), b'b');
    assert_eq!(r.rd(RBR), b'c');
    // Empty now, so the timer stays quiet.
    r.advance(ctt * 8);
    assert_eq!(r.irq(), 0);
    assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_NO_INT);
}

#[test]
fn receive_timeout_masked_by_ier() {
    let r = Rig::new();
    r.wr(FCR, UART_FCR_ITL_4 | UART_FCR_FE);
    r.serial.receive(b"a");
    r.advance(r.serial.char_transmit_time() as i64 * 4);
    assert_eq!(r.irq(), 0);
    // The timeout is pending, and shows once RDI is enabled.
    r.wr(IER, UART_IER_RDI);
    assert_eq!(r.rd(IIR), UART_IIR_FE | UART_IIR_CTI);
}

#[test]
fn loopback_echoes_thr_and_modem_lines() {
    let r = Rig::new();
    r.wr(IER, UART_IER_RDI);
    r.wr(MCR, UART_MCR_LOOP);
    r.wr(THR, 0x5a);
    assert!(r.out().is_empty());
    assert_eq!(r.rd(LSR), UART_LSR_TEMT | UART_LSR_THRE | UART_LSR_DR);
    assert_eq!(r.rd(IIR), UART_IIR_RDI);
    assert_eq!(r.rd(RBR), 0x5a);
    // No accept_input in loopback.
    assert_eq!(r.be.accepted.load(Ordering::SeqCst), 0);

    // OUT2, OUT1, RTS and DTR come back as DCD, RI, CTS and DSR.
    r.wr(MCR, UART_MCR_LOOP);
    assert_eq!(r.rd(MSR), 0);
    r.wr(MCR, UART_MCR_LOOP | UART_MCR_OUT2);
    assert_eq!(r.rd(MSR), UART_MSR_DCD);
    r.wr(MCR, UART_MCR_LOOP | UART_MCR_OUT1);
    assert_eq!(r.rd(MSR), UART_MSR_RI);
    r.wr(MCR, UART_MCR_LOOP | UART_MCR_RTS);
    assert_eq!(r.rd(MSR), UART_MSR_CTS);
    r.wr(MCR, UART_MCR_LOOP | UART_MCR_DTR);
    assert_eq!(r.rd(MSR), UART_MSR_DSR);
    r.wr(MCR, 0xff);
    assert_eq!(r.rd(MCR), 0x1f);
    assert_eq!(r.rd(MSR), 0xf0);

    // With the FIFO the whole burst comes back in order.
    r.wr(FCR, UART_FCR_FE);
    for &b in b"xyz" {
        r.wr(THR, b);
    }
    assert_eq!(r.rd(RBR), b'x');
    assert_eq!(r.rd(RBR), b'y');
    assert_eq!(r.rd(RBR), b'z');
    assert!(r.out().is_empty());
}

#[test]
fn modem_status_from_backend() {
    let be = Recorder::default();
    *be.tiocm.lock().unwrap() = Some(CHR_TIOCM_CTS | CHR_TIOCM_DSR | CHR_TIOCM_CAR);
    let r = Rig::with_backend(be);
    assert_eq!(r.rd(MSR), UART_MSR_DCD | UART_MSR_DSR | UART_MSR_CTS);

    r.wr(IER, UART_IER_MSI);
    *r.be.tiocm.lock().unwrap() = Some(CHR_TIOCM_DSR | CHR_TIOCM_CAR | CHR_TIOCM_RI);
    // The poll timer notices the change every 10ms.
    r.advance(NANOS_10MS);
    assert_eq!(r.irq(), 1);
    assert_eq!(r.rd(IIR), UART_IIR_MSI);
    assert_eq!(r.rd(MSR), UART_MSR_DCD | UART_MSR_DSR | UART_MSR_RI | UART_MSR_DCTS);
    assert_eq!(r.irq(), 0);

    // RI going away sets TERI.
    *r.be.tiocm.lock().unwrap() = Some(CHR_TIOCM_DSR | CHR_TIOCM_CAR);
    assert_eq!(r.rd(MSR), UART_MSR_DCD | UART_MSR_DSR | UART_MSR_TERI);

    // MCR changes go out as RTS and DTR.
    r.wr(MCR, UART_MCR_OUT2 | UART_MCR_RTS | UART_MCR_DTR);
    let last = *r.be.set_tiocm.lock().unwrap().last().unwrap();
    assert_eq!(last & (CHR_TIOCM_RTS | CHR_TIOCM_DTR), CHR_TIOCM_RTS | CHR_TIOCM_DTR);
}

const NANOS_10MS: i64 = 10_000_000;

#[test]
fn mmio_ops() {
    let r = Rig::new();
    let cx = AccessCtx::default();
    assert_eq!(r.serial.impl_constraints(), AccessConstraints::exact(1));
    assert!(r.serial.valid().unaligned);
    r.serial.write(&cx, SCR, AccessSize::B1, 0x1234).unwrap();
    assert_eq!(r.serial.read(&cx, SCR, AccessSize::B1).unwrap(), 0x34);
}

#[test]
fn isa_defaults() {
    let clock = Clock::manual(ClockType::Virtual);
    let mut next = 0;
    let mut ports = Vec::new();
    for _ in 0..MAX_ISA_SERIAL_PORTS {
        let mut isa = IsaSerial::new(Serial::new(clock.clone(), SERIAL_BAUDBASE_DEFAULT, None));
        isa.realize(&mut next, |_| IrqLine::default()).unwrap();
        ports.push((isa.index, isa.iobase, isa.isairq));
    }
    assert_eq!(ports, [(0, 0x3f8, 4), (1, 0x2f8, 3), (2, 0x3e8, 4), (3, 0x2e8, 3)]);

    let mut isa = IsaSerial::new(Serial::new(clock.clone(), SERIAL_BAUDBASE_DEFAULT, None));
    let err = isa.realize(&mut next, |_| IrqLine::default()).unwrap_err();
    assert!(err.to_string().contains("Max. supported number of ISA serial ports is 4."));

    // Explicit properties win, and the IRQ reaches the line isa_get_irq() gives.
    let mut next = 0;
    let mut isa = IsaSerial::new(Serial::new(clock, SERIAL_BAUDBASE_DEFAULT, None));
    isa.index = 1;
    isa.iobase = 0x100;
    let level = Arc::new(AtomicI32::new(-1));
    let l = level.clone();
    let mut asked = 0;
    isa.realize(&mut next, |n| {
        asked = n;
        IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst))
    })
    .unwrap();
    assert_eq!((isa.iobase, isa.isairq, asked, next), (0x100, 3, 3, 1));
    isa.state().ioport_write(IER, UART_IER_THRI);
    assert_eq!(level.load(Ordering::SeqCst), 1);
}

/// a-b-bootblock.S: an `outb` of 'A' and 'B' to 0x3f8 with no setup at all reaches the chardev.
#[test]
fn bootblock_output_without_init() {
    let r = Rig::new();
    r.wr(THR, b'A');
    r.wr(THR, b'B');
    assert_eq!(r.out(), b"AB");
}

/// The COM1 setup in tests/tcg/alpha/system/boot.S, then polled output.
#[test]
fn alpha_boot_com1_init_and_poll() {
    let r = Rig::new();
    r.wr(LCR, 0x87);
    r.wr(DLM, 0);
    r.wr(DLL, 12);
    r.wr(LCR, 0x07);
    r.wr(MCR, 0x0f);
    assert_eq!(r.be.params.lock().unwrap().last().unwrap().speed, 9600);
    for &c in b"Hello\n" {
        while r.rd(LSR) & 0x20 == 0 {}
        r.wr(THR, c);
    }
    assert_eq!(r.out(), b"Hello\n");
}
