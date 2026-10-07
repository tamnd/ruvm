// SPDX-License-Identifier: GPL-2.0-or-later

//! Save and load of the 16550A state `vmstate_serial` carries.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_char::serial::*;
use ruvm_hw_core::{Clock, IrqLine};

const RBR: u64 = 0;
const DLL: u64 = 0;
const IER: u64 = 1;
const IIR: u64 = 2;
const FCR: u64 = 2;
const LCR: u64 = 3;
const LSR: u64 = 5;
const SCR: u64 = 7;

#[derive(Default)]
struct Sink {
    out: Mutex<Vec<u8>>,
}

impl SerialBackend for Sink {
    fn write(&self, bytes: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(bytes);
        bytes.len()
    }
}

struct Rig {
    clock: Arc<Clock>,
    serial: Arc<Serial>,
    be: Arc<Sink>,
    level: Arc<AtomicI32>,
}

impl Rig {
    fn new() -> Self {
        Self::on(Clock::manual(ClockType::Virtual))
    }

    fn on(clock: Arc<Clock>) -> Self {
        let be = Arc::new(Sink::default());
        let serial =
            Serial::new(clock.clone(), SERIAL_BAUDBASE_DEFAULT, Some(be.clone() as Arc<_>));
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        serial.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        Rig { clock, serial, be, level }
    }

    fn advance(&self, ns: i64) {
        let now = self.clock.get_ns();
        self.clock.advance_to(now + ns);
    }
}

#[test]
fn reset_state_sends_no_subsections() {
    let r = Rig::new();
    let v = r.serial.vmstate_save();
    assert_eq!(v.divider, 0x0c);
    assert_eq!(v.iir, UART_IIR_NO_INT);
    assert_eq!(v.lsr, UART_LSR_TEMT | UART_LSR_THRE);
    assert_eq!(v.fifo_timeout_timer, -1);
    assert_eq!(v.modem_status_poll, -1);
    assert!(!v.thr_ipending_needed());
    assert!(!v.tsr_needed());
    assert!(!v.recv_fifo_needed());
    assert!(!v.xmit_fifo_needed());
    assert!(!v.fifo_timeout_timer_needed());
    assert!(!v.timeout_ipending_needed());
    // The sink has no modem lines, so serial_update_msl() turned polling off.
    assert!(!v.poll_needed());
}

#[test]
fn round_trip_with_fifo_and_timer() {
    let a = Rig::new();
    a.serial.ioport_write(LCR, UART_LCR_DLAB);
    a.serial.ioport_write(DLL, 0x01);
    a.serial.ioport_write(LCR, 0x1b);
    a.serial.ioport_write(SCR, 0x5a);
    a.serial.ioport_write(FCR, UART_FCR_FE | UART_FCR_ITL_3);
    a.serial.ioport_write(IER, UART_IER_RDI);
    a.serial.receive(b"abc");
    let v = a.serial.vmstate_save();
    assert_eq!(v.recv_fifo.num, 3);
    assert_eq!(&v.recv_fifo.data[..3], b"abc");
    assert!(v.recv_fifo_needed());
    assert!(v.fifo_timeout_timer_needed());

    let clock = Clock::manual(ClockType::Virtual);
    clock.set_ns(a.clock.get_ns());
    let b = Rig::on(clock);
    b.serial.vmstate_load(&v).unwrap();
    assert_eq!(b.serial.vmstate_save(), v);
    assert_eq!(b.serial.ioport_read(SCR), 0x5a);
    assert_eq!(b.serial.ioport_read(LCR), 0x1b);

    // The character timeout fires on the destination.
    b.advance(v.fifo_timeout_timer - b.clock.get_ns());
    assert_eq!(b.serial.ioport_read(IIR) & 0x0f, UART_IIR_CTI);
    assert_eq!(b.level.load(Ordering::SeqCst), 1);
    assert_eq!(b.serial.ioport_read(RBR), b'a');
    assert_eq!(b.serial.ioport_read(RBR), b'b');
    assert_eq!(b.serial.ioport_read(RBR), b'c');
    assert_eq!(b.serial.ioport_read(LSR) & UART_LSR_DR, 0);
}

#[test]
fn fifo_head_wraps() {
    let r = Rig::new();
    let mut v = r.serial.vmstate_save();
    v.fcr_vmstate = UART_FCR_FE;
    v.lsr |= UART_LSR_DR;
    v.recv_fifo.head = 14;
    v.recv_fifo.num = 3;
    v.recv_fifo.data[14] = b'x';
    v.recv_fifo.data[15] = b'y';
    v.recv_fifo.data[0] = b'z';
    r.serial.vmstate_load(&v).unwrap();
    assert_eq!(r.serial.ioport_read(RBR), b'x');
    assert_eq!(r.serial.ioport_read(RBR), b'y');
    assert_eq!(r.serial.ioport_read(RBR), b'z');
}

#[test]
fn thr_ipending_from_iir_when_absent() {
    let r = Rig::new();
    let mut v = r.serial.vmstate_save();
    v.ier = UART_IER_THRI;
    v.iir = UART_IIR_THRI;
    v.thr_ipending = -1;
    r.serial.vmstate_load(&v).unwrap();
    assert_eq!(r.serial.vmstate_save().thr_ipending, 1);

    v.iir = UART_IIR_NO_INT;
    r.serial.vmstate_load(&v).unwrap();
    let saved = r.serial.vmstate_save();
    assert_eq!(saved.thr_ipending, 0);
    assert!(!saved.thr_ipending_needed());
}

#[test]
fn stuck_byte_is_sent_after_load() {
    let r = Rig::new();
    let mut v = r.serial.vmstate_save();
    v.lsr &= !UART_LSR_TEMT;
    v.tsr_retry = 9;
    v.tsr = b'q';
    r.serial.vmstate_load(&v).unwrap();
    assert_eq!(*r.be.out.lock().unwrap(), b"q");
    let saved = r.serial.vmstate_save();
    assert_eq!(saved.tsr_retry, 0);
    assert_ne!(saved.lsr & UART_LSR_TEMT, 0);
}

#[test]
fn inconsistent_state_is_refused() {
    let r = Rig::new();
    let good = r.serial.vmstate_save();

    let mut v = good.clone();
    v.tsr_retry = 1;
    assert!(r.serial.vmstate_load(&v).is_err());

    let mut v = good.clone();
    v.lsr &= !UART_LSR_TEMT;
    assert!(r.serial.vmstate_load(&v).is_err());

    let mut v = good.clone();
    v.xmit_fifo.head = 16;
    assert!(r.serial.vmstate_load(&v).is_err());

    let mut v = good.clone();
    v.recv_fifo.num = 17;
    assert!(r.serial.vmstate_load(&v).is_err());

    assert_eq!(r.serial.vmstate_save(), good);
}
