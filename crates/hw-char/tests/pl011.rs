// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the PL011, plus the way the `virt` board maps it at 0x09000000.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_char::pl011::*;
use ruvm_hw_char::serial::SerialBackend;
use ruvm_hw_core::IrqLine;
use ruvm_mem::{
    AccessCtx, AccessSize, AddressSpace, MemTxAttrs, MemTxResult, MemorySystem, MmioOps,
};

/// A chardev that records what the UART does to it.
#[derive(Default)]
struct Recorder {
    out: Mutex<Vec<u8>>,
    breaks: Mutex<Vec<bool>>,
    accepted: AtomicI32,
}

impl SerialBackend for Recorder {
    fn write(&self, bytes: &[u8]) -> usize {
        self.out.lock().unwrap().extend_from_slice(bytes);
        bytes.len()
    }

    fn set_break(&self, enable: bool) {
        self.breaks.lock().unwrap().push(enable);
    }

    fn accept_input(&self) {
        self.accepted.fetch_add(1, Ordering::SeqCst);
    }
}

struct Rig {
    uart: Arc<Pl011>,
    be: Arc<Recorder>,
    levels: Vec<Arc<AtomicI32>>,
}

impl Rig {
    fn new() -> Self {
        let be = Arc::new(Recorder::default());
        let uart = Pl011::new(Some(be.clone() as Arc<_>));
        let mut levels = Vec::new();
        for n in 0..PL011_NUM_IRQS {
            let level = Arc::new(AtomicI32::new(-1));
            let l = level.clone();
            uart.irq(n).connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
            levels.push(level);
        }
        Rig { uart, be, levels }
    }

    fn rd(&self, reg: u64) -> u32 {
        self.uart.reg_read(reg)
    }

    fn wr(&self, reg: u64, val: u32) {
        self.uart.reg_write(reg, val);
    }

    /// The level of output `n`, or -1 if it was never driven.
    fn irq(&self, n: usize) -> i32 {
        self.levels[n].load(Ordering::SeqCst)
    }

    fn out(&self) -> Vec<u8> {
        self.be.out.lock().unwrap().clone()
    }
}

#[test]
fn reset_values() {
    let r = Rig::new();
    assert_eq!(r.rd(UARTRSR), 0);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    assert_eq!(r.rd(UARTILPR), 0);
    assert_eq!(r.rd(UARTIBRD), 0);
    assert_eq!(r.rd(UARTFBRD), 0);
    assert_eq!(r.rd(UARTLCR_H), 0);
    assert_eq!(r.rd(UARTCR), CR_RXE | CR_TXE);
    assert_eq!(r.rd(UARTIFLS), 0x12);
    assert_eq!(r.rd(UARTIMSC), 0);
    assert_eq!(r.rd(UARTRIS), 0);
    assert_eq!(r.rd(UARTMIS), 0);
    assert_eq!(r.rd(UARTDMACR), 0);
    // Unknown offsets and the write-only ICR read as 0.
    assert_eq!(r.rd(UARTICR), 0);
    assert_eq!(r.rd(0x80), 0);
    assert_eq!(r.uart.can_receive(), 1);
    assert_eq!(r.uart.baudrate(), 0);
    // Reset does not drive the outputs.
    assert_eq!(r.irq(0), -1);
}

#[test]
fn id_registers() {
    let r = Rig::new();
    let ids: Vec<u32> = (0..8).map(|i| r.rd(UARTPERIPHID0 + 4 * i)).collect();
    assert_eq!(ids, [0x11, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
    let lum = Pl011::new_luminary(None);
    let ids: Vec<u32> = (0..8).map(|i| lum.reg_read(UARTPERIPHID0 + 4 * i)).collect();
    assert_eq!(ids, [0x11, 0x00, 0x18, 0x01, 0x0d, 0xf0, 0x05, 0xb1]);
    assert_eq!(lum.id(), &PL011_ID_LUMINARY);
}

#[test]
fn transmit_reaches_backend_and_sets_txris() {
    let r = Rig::new();
    for &b in b"Hello" {
        r.wr(UARTDR, u32::from(b));
    }
    // Only the low byte of a DR write is sent.
    r.wr(UARTDR, 0x1_0021);
    assert_eq!(r.out(), b"Hello!");
    assert_eq!(r.rd(UARTRIS), INT_TX);
    assert_eq!(r.rd(UARTMIS), 0);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(2), 0);

    r.wr(UARTIMSC, INT_TX);
    assert_eq!(r.rd(UARTMIS), INT_TX);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.irq(2), 1);
    assert_eq!(r.irq(3), 0);

    r.wr(UARTICR, INT_TX);
    assert_eq!(r.rd(UARTRIS), 0);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(2), 0);
}

#[test]
fn transmit_while_disabled_still_sends() {
    let r = Rig::new();
    r.wr(UARTCR, 0);
    r.wr(UARTDR, u32::from(b'x'));
    assert_eq!(r.out(), b"x");
}

#[test]
fn receive_without_fifo() {
    let r = Rig::new();
    r.wr(UARTIMSC, INT_RX);
    assert_eq!(r.uart.can_receive(), 1);
    r.uart.receive(b"a");
    assert_eq!(r.uart.can_receive(), 0);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFF);
    assert_eq!(r.rd(UARTRIS), INT_RX);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(1), 1);

    assert_eq!(r.rd(UARTDR), u32::from(b'a'));
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    assert_eq!(r.rd(UARTRIS), 0);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.be.accepted.load(Ordering::SeqCst), 1);

    // Reading an empty FIFO returns the last entry again and changes nothing.
    assert_eq!(r.rd(UARTDR), u32::from(b'a'));
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
}

#[test]
fn receive_with_fifo() {
    let r = Rig::new();
    r.wr(UARTLCR_H, LCR_FEN | 0x60);
    r.wr(UARTIMSC, INT_RX);
    assert_eq!(r.uart.can_receive(), PL011_FIFO_DEPTH);

    // The RX interrupt goes up at the first byte, whatever IFLS says.
    r.uart.receive(b"0");
    assert_eq!(r.irq(1), 1);
    let data: Vec<u8> = (b'1'..).take(PL011_FIFO_DEPTH - 1).collect();
    r.uart.receive(&data);
    assert_eq!(r.uart.can_receive(), 0);
    assert_eq!(r.rd(UARTFR) & (PL011_FLAG_RXFF | PL011_FLAG_RXFE), PL011_FLAG_RXFF);

    let mut got = Vec::new();
    for _ in 0..PL011_FIFO_DEPTH {
        assert_eq!(r.irq(1), 1);
        got.push(r.rd(UARTDR) as u8);
        assert_eq!(r.rd(UARTFR) & PL011_FLAG_RXFF, 0);
    }
    let want: Vec<u8> = (b'0'..).take(PL011_FIFO_DEPTH).collect();
    assert_eq!(got, want);
    assert_eq!(r.rd(UARTFR) & PL011_FLAG_RXFE, PL011_FLAG_RXFE);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.uart.can_receive(), PL011_FIFO_DEPTH);

    // The FIFO wraps around.
    r.uart.receive(b"xyz");
    assert_eq!(r.rd(UARTDR), u32::from(b'x'));
    assert_eq!(r.rd(UARTDR), u32::from(b'y'));
    assert_eq!(r.rd(UARTDR), u32::from(b'z'));
}

#[test]
fn toggling_fifo_enable_drops_received_data() {
    let r = Rig::new();
    r.wr(UARTLCR_H, LCR_FEN);
    r.uart.receive(b"abc");
    assert_eq!(r.uart.can_receive(), PL011_FIFO_DEPTH - 3);
    // Changing other LCR bits keeps the FIFO.
    r.wr(UARTLCR_H, LCR_FEN | 0x60);
    assert_eq!(r.uart.can_receive(), PL011_FIFO_DEPTH - 3);
    r.wr(UARTLCR_H, 0x60);
    assert_eq!(r.uart.can_receive(), 1);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    // The RX interrupt status is not cleared by the FIFO reset.
    assert_eq!(r.rd(UARTRIS), INT_RX);
}

#[test]
fn break_from_backend() {
    let r = Rig::new();
    r.uart.receive_break();
    assert_eq!(r.rd(UARTDR), DR_BE);
    assert_eq!(r.rd(UARTRSR), DR_BE >> 8);
    // Any write to ECR clears RSR.
    r.wr(UARTRSR, 0);
    assert_eq!(r.rd(UARTRSR), 0);
}

#[test]
fn break_output_goes_to_backend() {
    let r = Rig::new();
    r.wr(UARTLCR_H, LCR_BRK);
    r.wr(UARTLCR_H, LCR_BRK | 0x60);
    r.wr(UARTLCR_H, 0);
    assert_eq!(*r.be.breaks.lock().unwrap(), [true, false]);
    // Not in loopback, so nothing came back.
    assert_eq!(r.rd(UARTFR) & PL011_FLAG_RXFE, PL011_FLAG_RXFE);
}

#[test]
fn loopback_data_and_break() {
    let r = Rig::new();
    r.wr(UARTLCR_H, LCR_FEN);
    r.wr(UARTCR, CR_UARTEN | CR_TXE | CR_RXE | CR_LBE);
    r.wr(UARTDR, u32::from(b'L'));
    assert_eq!(r.out(), b"L");
    // Input from the backend is ignored in loopback.
    r.uart.receive(b"no");
    r.uart.receive_break();
    r.wr(UARTLCR_H, LCR_FEN | LCR_BRK);
    assert_eq!(r.rd(UARTDR), u32::from(b'L'));
    assert_eq!(r.rd(UARTDR), DR_BE);
    assert_eq!(r.rd(UARTRSR), DR_BE >> 8);
    assert_eq!(r.rd(UARTFR) & PL011_FLAG_RXFE, PL011_FLAG_RXFE);
}

#[test]
fn loopback_modem_lines() {
    let r = Rig::new();
    r.wr(UARTIMSC, INT_MS);
    // Without LBE the outputs do not reach the inputs.
    r.wr(UARTCR, CR_OUT2 | CR_OUT1 | CR_RTS | CR_DTR | CR_TXE | CR_RXE);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    assert_eq!(r.rd(UARTRIS), 0);

    r.wr(UARTCR, CR_OUT2 | CR_OUT1 | CR_RTS | CR_DTR | CR_TXE | CR_RXE | CR_LBE);
    let lines = PL011_FLAG_RI | PL011_FLAG_DCD | PL011_FLAG_CTS | PL011_FLAG_DSR;
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE | lines);
    assert_eq!(r.rd(UARTRIS), INT_MS);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(4), 1);
    assert_eq!(r.irq(5), 0);

    r.wr(UARTCR, CR_RTS | CR_LBE);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE | PL011_FLAG_CTS);
    assert_eq!(r.rd(UARTRIS), INT_CTS);
    assert_eq!(r.irq(4), 1);
    r.wr(UARTCR, CR_LBE);
    assert_eq!(r.rd(UARTRIS), 0);
    assert_eq!(r.irq(4), 0);
}

#[test]
fn loopback_overrun_stops_input() {
    let r = Rig::new();
    r.wr(UARTCR, CR_TXE | CR_RXE | CR_LBE);
    r.wr(UARTDR, u32::from(b'1'));
    r.wr(UARTDR, u32::from(b'2'));
    // QEMU does not bound loopback; the one-entry FIFO keeps the newest byte.
    assert_eq!(r.uart.can_receive(), 0);
    r.wr(UARTCR, CR_TXE | CR_RXE);
    assert_eq!(r.uart.can_receive(), 0);
    assert_eq!(r.rd(UARTDR), u32::from(b'2'));
    assert_eq!(r.uart.can_receive(), 0);
    assert_eq!(r.rd(UARTDR), u32::from(b'2'));
    assert_eq!(r.uart.can_receive(), 1);
}

#[test]
fn baud_rate_registers() {
    let r = Rig::new();
    r.uart.set_clock_hz(24_000_000);
    assert_eq!(r.uart.clock_hz(), 24_000_000);
    // 24 MHz / (16 * 115200) = 13.02, so IBRD 13 and FBRD 1.
    r.wr(UARTIBRD, 0xf_000d);
    r.wr(UARTFBRD, 0xc1);
    assert_eq!(r.rd(UARTIBRD), 13);
    assert_eq!(r.rd(UARTFBRD), 1);
    assert_eq!(r.uart.baudrate(), 115_244);
    // FR ignores writes.
    r.wr(UARTFR, 0xffff);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    r.wr(UARTILPR, 0x1ff);
    r.wr(UARTIFLS, 0x3f);
    r.wr(UARTDMACR, 7);
    assert_eq!(r.rd(UARTILPR), 0x1ff);
    assert_eq!(r.rd(UARTIFLS), 0x3f);
    assert_eq!(r.rd(UARTDMACR), 7);
}

#[test]
fn reset_restores_registers() {
    let r = Rig::new();
    r.wr(UARTLCR_H, LCR_FEN | 0x70);
    r.wr(UARTCR, CR_UARTEN | CR_LBE);
    r.wr(UARTIMSC, 0x7ff);
    r.wr(UARTIBRD, 1);
    r.wr(UARTDR, 0x41);
    r.uart.reset();
    assert_eq!(r.rd(UARTLCR_H), 0);
    assert_eq!(r.rd(UARTCR), CR_RXE | CR_TXE);
    assert_eq!(r.rd(UARTIMSC), 0);
    assert_eq!(r.rd(UARTRIS), 0);
    assert_eq!(r.rd(UARTIBRD), 0);
    assert_eq!(r.rd(UARTFR), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    assert_eq!(r.uart.can_receive(), 1);
}

#[test]
fn backend_can_be_swapped() {
    let r = Rig::new();
    r.uart.set_backend(None);
    r.wr(UARTDR, u32::from(b'a'));
    let other = Arc::new(Recorder::default());
    r.uart.set_backend(Some(other.clone() as Arc<_>));
    r.wr(UARTDR, u32::from(b'b'));
    assert!(r.out().is_empty());
    assert_eq!(*other.out.lock().unwrap(), b"b");
}

#[test]
fn mmio_ops() {
    let r = Rig::new();
    let ops: &dyn MmioOps = &*r.uart;
    assert_eq!(ops.impl_constraints().min, 4);
    assert_eq!(ops.impl_constraints().max, 4);
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    ops.write(&cx, UARTDR, AccessSize::B4, u64::from(b'Q')).unwrap();
    assert_eq!(r.out(), b"Q");
    assert_eq!(ops.read(&cx, UARTCR, AccessSize::B4).unwrap(), 0x300);
}

/// The PL011 the way `virt` maps it, accessed through the memory core.
#[test]
fn mapped_at_virt_uart_base() {
    const BASE: u64 = 0x0900_0000;
    let r = Rig::new();
    let mem = Arc::new(MemorySystem::new());
    let sys = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_io(TYPE_PL011, u128::from(PL011_MMIO_SIZE), r.uart.clone()).unwrap();
    mem.add_subregion(sys, BASE, io).unwrap();
    let space: Arc<AddressSpace> = mem.address_space_init(sys, "memory").unwrap();

    // A byte store to DR is widened to a 32-bit write.
    assert_eq!(space.write(BASE, MemTxAttrs::UNSPECIFIED, b"h"), MemTxResult::OK);
    assert_eq!(space.write_u32(BASE, MemTxAttrs::UNSPECIFIED, u32::from(b'i')), MemTxResult::OK);
    assert_eq!(r.out(), b"hi");

    let mut b = [0u8; 4];
    assert_eq!(space.read(BASE + UARTFR, MemTxAttrs::UNSPECIFIED, &mut b), MemTxResult::OK);
    assert_eq!(u32::from_le_bytes(b), PL011_FLAG_TXFE | PL011_FLAG_RXFE);
    let mut b = [0u8; 1];
    assert_eq!(space.read(BASE + 0xfe0, MemTxAttrs::UNSPECIFIED, &mut b), MemTxResult::OK);
    assert_eq!(b[0], 0x11);
    assert_eq!(space.read(BASE + 0xff0, MemTxAttrs::UNSPECIFIED, &mut b), MemTxResult::OK);
    assert_eq!(b[0], 0x0d);
}
