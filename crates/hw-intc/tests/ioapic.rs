// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the IOAPIC from hw/intc/ioapic.c, driven through IOREGSEL/IOWIN
//! with a manual virtual clock for the interrupt storm timer.

use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_intc::ioapic::*;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

type Msgs = Arc<Mutex<Vec<(u64, u32)>>>;

struct Rig {
    clock: Arc<Clock>,
    ioapics: IoApics,
    s: Arc<IoApic>,
    msgs: Msgs,
}

fn rig_version(version: u8) -> Rig {
    let clock = Clock::manual(ClockType::Virtual);
    let ioapics = IoApics::new();
    let msgs: Msgs = Arc::default();
    let m = msgs.clone();
    let s = IoApic::realize(
        &clock,
        version,
        &ioapics,
        Arc::new(move |a, d| m.lock().unwrap().push((a, d))),
    )
    .unwrap();
    Rig { clock, ioapics, s, msgs }
}

fn rig() -> Rig {
    rig_version(IOAPIC_VER_DEF)
}

impl Rig {
    fn read_reg(&self, sel: u8) -> u32 {
        self.s.mmio_write(IOAPIC_IOREGSEL, 4, u64::from(sel));
        self.s.mmio_read(IOAPIC_IOWIN, 4) as u32
    }

    fn write_reg(&self, sel: u8, v: u32) {
        self.s.mmio_write(IOAPIC_IOREGSEL, 4, u64::from(sel));
        self.s.mmio_write(IOAPIC_IOWIN, 4, u64::from(v));
    }

    fn set_entry(&self, pin: u8, entry: u64) {
        self.write_reg(IOAPIC_REG_REDTBL_BASE + 2 * pin + 1, (entry >> 32) as u32);
        self.write_reg(IOAPIC_REG_REDTBL_BASE + 2 * pin, entry as u32);
    }

    fn entry(&self, pin: u8) -> u64 {
        let lo = self.read_reg(IOAPIC_REG_REDTBL_BASE + 2 * pin);
        let hi = self.read_reg(IOAPIC_REG_REDTBL_BASE + 2 * pin + 1);
        (u64::from(hi) << 32) | u64::from(lo)
    }

    fn take(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.msgs.lock().unwrap())
    }
}

#[test]
fn version_register() {
    let r = rig();
    // 24 entries (23 in bits 16..23), version 0x20.
    assert_eq!(r.read_reg(IOAPIC_REG_VER), 0x0017_0020);
    assert_eq!(rig_version(0x11).read_reg(IOAPIC_REG_VER), 0x0017_0011);
    // Read only.
    r.write_reg(IOAPIC_REG_VER, 0);
    assert_eq!(r.read_reg(IOAPIC_REG_VER), 0x0017_0020);

    let clock = Clock::manual(ClockType::Virtual);
    let e = IoApic::realize(&clock, 0x12, &IoApics::new(), Arc::new(|_, _| {})).unwrap_err();
    assert_eq!(e.message(), "IOAPIC only supports version 0x11 or 0x20 (default: 0x20).");

    let set = IoApics::new();
    let _a = IoApic::realize(&clock, 0x20, &set, Arc::new(|_, _| {})).unwrap();
    let _b = IoApic::realize(&clock, 0x20, &set, Arc::new(|_, _| {})).unwrap();
    let e = IoApic::realize(&clock, 0x20, &set, Arc::new(|_, _| {})).unwrap_err();
    assert_eq!(e.message(), "Only 2 ioapics allowed");
}

#[test]
fn id_and_ioregsel() {
    let r = rig();
    r.write_reg(IOAPIC_REG_ID, 0xfa00_0000);
    assert_eq!(r.read_reg(IOAPIC_REG_ID), 0x0a00_0000);
    // The arbitration register reads as the ID.
    assert_eq!(r.read_reg(IOAPIC_REG_ARB), 0x0a00_0000);
    r.s.mmio_write(IOAPIC_IOREGSEL, 4, 0x1234);
    assert_eq!(r.s.mmio_read(IOAPIC_IOREGSEL, 4), 0x34);
    // Past the table and at the gap below it reads zero.
    assert_eq!(r.read_reg(0x40), 0);
    assert_eq!(r.read_reg(0x03), 0);
    // IOWIN only takes 4 byte accesses.
    r.s.mmio_write(IOAPIC_IOREGSEL, 4, u64::from(IOAPIC_REG_VER));
    assert_eq!(r.s.mmio_read(IOAPIC_IOWIN, 2), 0);
    // The window repeats every 256 bytes.
    assert_eq!(r.s.mmio_read(0x110, 4), 0x0017_0020);
}

#[test]
fn reset_masks_every_entry() {
    let r = rig();
    for pin in 0..IOAPIC_NUM_PINS as u8 {
        assert_eq!(r.entry(pin), IOAPIC_LVT_MASKED);
    }
    r.set_entry(3, 0x0500_0000_0000_0031);
    r.s.reset();
    assert_eq!(r.entry(3), IOAPIC_LVT_MASKED);
    assert_eq!(r.read_reg(IOAPIC_REG_ID), 0);
}

#[test]
fn redirection_entry_programming() {
    let r = rig();
    r.set_entry(5, 0xff00_0000_0001_a9ff);
    // Delivery status and remote IRR are read only.
    assert_eq!(r.entry(5), 0xff00_0000_0001_a9ff & IOAPIC_RW_BITS);
    assert_eq!(r.entry(5) & IOAPIC_RO_BITS, 0);
    assert_eq!(r.s.redirection_entry(5), r.entry(5));
}

#[test]
fn edge_delivery_message_layout() {
    let r = rig();
    // Fixed, physical, edge, vector 0x30, destination 5.
    r.set_entry(4, 0x0500_0000_0000_0030);
    assert!(r.take().is_empty());
    let pin = r.s.input(4);
    pin.raise();
    assert_eq!(r.take(), [(0xfee0_5000, 0x0030)]);
    // Edge: no remote IRR, and holding the line high does nothing more.
    assert_eq!(r.entry(4) & IOAPIC_LVT_REMOTE_IRR, 0);
    assert_eq!(r.s.irr(), 0);
    pin.raise();
    assert_eq!(r.take(), [(0xfee0_5000, 0x0030)]);
    pin.lower();
    assert!(r.take().is_empty());

    // Lowest priority, logical destination 0x0f, vector 0x41.
    r.set_entry(6, 0x0f00_0000_0000_0941);
    r.s.input(6).pulse();
    assert_eq!(r.take(), [(0xfee0_f004, 0x0141)]);

    // NMI keeps its delivery mode in the data word.
    r.set_entry(7, 0x0100_0000_0000_0400);
    r.s.input(7).pulse();
    assert_eq!(r.take(), [(0xfee0_1000, 0x0400)]);

    // The whole 16 bit dest_idx field ends up in address bits 4 to 19.
    r.set_entry(8, 0x1234_0000_0000_0050);
    r.s.input(8).pulse();
    assert_eq!(r.take(), [(0xfee0_0000 | (0x1234 << 4), 0x0050)]);
}

#[test]
fn irq0_goes_to_pin2() {
    let r = rig();
    r.set_entry(2, 0x0000_0000_0000_0020);
    r.s.input(0).pulse();
    assert_eq!(r.take(), [(0xfee0_0000, 0x20)]);
    assert_eq!(r.s.irq_count()[0], 1);
    assert_eq!(r.s.irq_count()[2], 0);
}

#[test]
fn masked_edge_is_dropped() {
    let r = rig();
    r.set_entry(9, IOAPIC_LVT_MASKED | 0x33);
    r.s.input(9).pulse();
    assert!(r.take().is_empty());
    assert_eq!(r.s.irr(), 0);
    // Unmasking later does not bring it back.
    r.set_entry(9, 0x33);
    assert!(r.take().is_empty());
}

#[test]
fn level_remote_irr_until_eoi() {
    let r = rig();
    // Level, vector 0x40, destination 1.
    r.set_entry(10, 0x0100_0000_0000_8040);
    let pin = r.s.input(10);
    pin.raise();
    assert_eq!(r.take(), [(0xfee0_1000, 0x8040)]);
    assert_ne!(r.entry(10) & IOAPIC_LVT_REMOTE_IRR, 0);
    assert_eq!(r.s.irr(), 1 << 10);

    // Remote IRR holds off new deliveries.
    pin.lower();
    pin.raise();
    assert!(r.take().is_empty());

    // A write to the entry does not clear remote IRR for a level entry.
    r.set_entry(10, 0x0100_0000_0000_8040);
    assert_ne!(r.entry(10) & IOAPIC_LVT_REMOTE_IRR, 0);
    assert!(r.take().is_empty());

    // EOI for another vector changes nothing.
    r.ioapics.eoi_broadcast(0x41);
    assert!(r.take().is_empty());

    // EOI with the line still high delivers again.
    r.ioapics.eoi_broadcast(0x40);
    assert_eq!(r.take(), [(0xfee0_1000, 0x8040)]);
    assert_ne!(r.entry(10) & IOAPIC_LVT_REMOTE_IRR, 0);

    // With the line low, EOI just clears remote IRR.
    pin.lower();
    r.ioapics.eoi_broadcast(0x40);
    assert!(r.take().is_empty());
    assert_eq!(r.entry(10) & IOAPIC_LVT_REMOTE_IRR, 0);

    // Switching the entry to edge clears remote IRR on the write.
    pin.raise();
    assert_eq!(r.take().len(), 1);
    r.set_entry(10, 0x0100_0000_0000_0040);
    assert_eq!(r.entry(10) & IOAPIC_LVT_REMOTE_IRR, 0);
}

#[test]
fn masked_level_is_latched_and_delivered_on_unmask() {
    let r = rig();
    r.set_entry(11, IOAPIC_LVT_MASKED | IOAPIC_LVT_TRIGGER_MODE | 0x51);
    r.s.input(11).raise();
    assert!(r.take().is_empty());
    assert_eq!(r.s.irr(), 1 << 11);
    assert_eq!(r.entry(11) & IOAPIC_LVT_REMOTE_IRR, 0);
    r.set_entry(11, IOAPIC_LVT_TRIGGER_MODE | 0x51);
    assert_eq!(r.take(), [(0xfee0_0000, 0x8051)]);
}

#[test]
fn eoi_register_only_in_version_0x20() {
    for (version, redeliver) in [(0x20, true), (0x11, false)] {
        let r = rig_version(version);
        r.set_entry(12, IOAPIC_LVT_TRIGGER_MODE | 0x60);
        r.s.input(12).raise();
        assert_eq!(r.take().len(), 1);
        // Only 4 byte writes count.
        r.s.mmio_write(IOAPIC_EOI, 1, 0x60);
        assert!(r.take().is_empty());
        r.s.mmio_write(IOAPIC_EOI, 4, 0x60);
        assert_eq!(r.take().len(), usize::from(redeliver));
    }
}

#[test]
fn eoi_reaches_every_ioapic() {
    let clock = Clock::manual(ClockType::Virtual);
    let set = IoApics::new();
    let msgs: Msgs = Arc::default();
    let m1 = msgs.clone();
    let m2 = msgs.clone();
    let a =
        IoApic::realize(&clock, 0x20, &set, Arc::new(move |x, d| m1.lock().unwrap().push((x, d))))
            .unwrap();
    let b =
        IoApic::realize(&clock, 0x20, &set, Arc::new(move |x, d| m2.lock().unwrap().push((x, d))))
            .unwrap();
    for s in [&a, &b] {
        s.mmio_write(IOAPIC_IOREGSEL, 4, 0x12);
        s.mmio_write(IOAPIC_IOWIN, 4, 0x8070);
        s.input(1).raise();
    }
    assert_eq!(msgs.lock().unwrap().len(), 2);
    // An EOI written to the first one clears the second one as well.
    a.mmio_write(IOAPIC_EOI, 4, 0x70);
    assert_eq!(msgs.lock().unwrap().len(), 4);
}

#[test]
fn interrupt_storm_is_delayed() {
    let r = rig();
    r.set_entry(13, IOAPIC_LVT_TRIGGER_MODE | 0x77);
    r.s.input(13).raise();
    assert_eq!(r.take().len(), 1);
    for _ in 0..SUCCESSIVE_IRQ_MAX_COUNT - 1 {
        r.ioapics.eoi_broadcast(0x77);
    }
    assert_eq!(r.take().len(), (SUCCESSIVE_IRQ_MAX_COUNT - 1) as usize);
    // The next EOI in a row is not answered right away...
    r.ioapics.eoi_broadcast(0x77);
    assert!(r.take().is_empty());
    assert_eq!(r.entry(13) & IOAPIC_LVT_REMOTE_IRR, 0);
    // ...but 10 ms later.
    r.clock.advance_to(9_999_999);
    assert!(r.take().is_empty());
    r.clock.advance_to(10_000_000);
    assert_eq!(r.take(), [(0xfee0_0000, 0x8077)]);
    assert_ne!(r.entry(13) & IOAPIC_LVT_REMOTE_IRR, 0);
}

#[test]
fn handler_may_call_back_in() {
    // A LAPIC that EOIs at once, from inside the delivery.
    let clock = Clock::manual(ClockType::Virtual);
    let set = IoApics::new();
    let count = Arc::new(Mutex::new(0));
    let c = count.clone();
    let s2 = set.clone();
    let s = IoApic::realize(
        &clock,
        0x20,
        &set,
        Arc::new(move |_, data| {
            let mut n = c.lock().unwrap();
            *n += 1;
            if *n < 3 {
                drop(n);
                s2.eoi_broadcast((data & 0xff) as i32);
            }
        }),
    )
    .unwrap();
    s.mmio_write(IOAPIC_IOREGSEL, 4, 0x12);
    s.mmio_write(IOAPIC_IOWIN, 4, 0x8090);
    s.input(1).raise();
    assert_eq!(*count.lock().unwrap(), 3);
}

#[test]
fn mmio_ops_and_info() {
    let r = rig();
    let ops: &dyn MmioOps = &*r.s;
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    ops.write(&cx, IOAPIC_IOREGSEL, AccessSize::B4, 1).unwrap();
    assert_eq!(ops.read(&cx, IOAPIC_IOWIN, AccessSize::B4).unwrap(), 0x0017_0020);
    assert_eq!(ops.impl_constraints().max, 4);

    r.set_entry(1, 0x0200_0000_0000_a031);
    let info = r.s.print_info();
    assert!(info.starts_with("ioapic0: ver=0x20 id=0x00 sel=0x12 (redir[1])\n"), "{info}");
    assert!(
        info.contains(
            "  pin 1  0x020000000000a031 dest=2 vec=49  active-lo level        fixed  physical\n"
        ),
        "{info}"
    );
    assert!(info.ends_with("  IRR      (none)\n  Remote IRR (none)\n"), "{info}");
}
