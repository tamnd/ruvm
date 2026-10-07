// SPDX-License-Identifier: GPL-2.0-or-later

//! Behaviour of the cascaded i8259 pair, programmed the way SeaBIOS and Linux do it.

use std::sync::{Arc, Mutex};

use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_intc::i8259::{I8259, I8259Pair, i8259_init};
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

type Log = Arc<Mutex<Vec<i32>>>;

struct Pics {
    pair: I8259Pair,
    int: Log,
}

impl Pics {
    fn master(&self) -> &I8259 {
        &self.pair.master
    }

    fn slave(&self) -> &I8259 {
        &self.pair.slave
    }

    /// The level of the INT line to the CPU.
    fn int(&self) -> i32 {
        self.int.lock().unwrap().last().copied().unwrap_or(0)
    }

    fn raise(&self, irq: usize) {
        self.pair.irq_set[irq].raise();
    }

    fn lower(&self, irq: usize) {
        self.pair.irq_set[irq].lower();
    }

    fn pulse(&self, irq: usize) {
        self.pair.irq_set[irq].pulse();
    }

    fn ack(&self) -> u8 {
        self.pair.master.pic_read_irq()
    }
}

/// The pair with nothing programmed yet.
fn raw() -> Pics {
    let int: Log = Arc::default();
    let l = int.clone();
    let pair = i8259_init(IrqLine::from_fn(move |level| l.lock().unwrap().push(level)));
    Pics { pair, int }
}

/// The ICW sequence of SeaBIOS (`pic_setup`) and Linux (`init_8259A`): edge triggered,
/// cascade on IRQ2, 8086 mode, all lines unmasked.
fn init(master_base: u8, slave_base: u8) -> Pics {
    let p = raw();
    p.master().ioport_write(0, 0x11);
    p.master().ioport_write(1, master_base);
    p.master().ioport_write(1, 0x04);
    p.master().ioport_write(1, 0x01);
    p.slave().ioport_write(0, 0x11);
    p.slave().ioport_write(1, slave_base);
    p.slave().ioport_write(1, 0x02);
    p.slave().ioport_write(1, 0x01);
    p
}

fn pc() -> Pics {
    init(0x08, 0x70)
}

/// Reads the ISR through OCW3.
fn isr(pic: &I8259) -> u8 {
    pic.ioport_write(0, 0x0b);
    pic.ioport_read(0)
}

/// Reads the IRR through OCW3.
fn irr(pic: &I8259) -> u8 {
    pic.ioport_write(0, 0x0a);
    pic.ioport_read(0)
}

#[test]
fn init_sequence() {
    let p = pc();
    let m = p.master().state();
    assert_eq!((m.irq_base, m.init_state, m.init4, m.single_mode), (0x08, 0, 1, 0));
    assert_eq!((m.imr, m.isr, m.irr, m.auto_eoi), (0, 0, 0, 0));
    let s = p.slave().state();
    assert_eq!((s.irq_base, s.init_state), (0x70, 0));
    // The data port now reads and writes IMR.
    p.master().ioport_write(1, 0xfb);
    assert_eq!(p.master().ioport_read(1), 0xfb);
    p.slave().ioport_write(1, 0xff);
    assert_eq!(p.slave().ioport_read(1), 0xff);
    // The low three bits of the vector base are dropped.
    let q = init(0x27, 0x2f);
    assert_eq!(q.master().state().irq_base, 0x20);
    assert_eq!(q.slave().state().irq_base, 0x28);
    assert_eq!(
        q.master().print_info(),
        "pic0: irr=00 imr=00 isr=00 hprio=0 irq_base=20 rr_sel=0 elcr=00 fnm=0\n"
    );
}

#[test]
fn single_mode_without_icw4_skips_icw3() {
    let p = raw();
    p.master().ioport_write(0, 0x12);
    p.master().ioport_write(1, 0x40);
    assert_eq!(p.master().state().init_state, 0);
    p.master().ioport_write(1, 0x55);
    assert_eq!(p.master().state().imr, 0x55);
}

#[test]
fn edge_irq_ack_and_eoi() {
    let p = pc();
    assert_eq!(p.int(), 0);
    p.pulse(1);
    assert_eq!(p.int(), 1);
    assert_eq!(irr(p.master()), 0x02);
    assert_eq!(p.ack(), 0x09);
    // Acknowledged: IRR clear, ISR set, INT down.
    assert_eq!(p.int(), 0);
    assert_eq!(irr(p.master()), 0);
    assert_eq!(isr(p.master()), 0x02);
    // A second edge on the same line waits for the EOI.
    p.pulse(1);
    assert_eq!(p.int(), 0);
    p.master().ioport_write(0, 0x20);
    assert_eq!(isr(p.master()), 0);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x09);
}

#[test]
fn edge_needs_a_low_level_first() {
    let p = pc();
    p.raise(4);
    assert_eq!(p.ack(), 0x0c);
    p.master().ioport_write(0, 0x20);
    // Still high: no new edge.
    p.raise(4);
    assert_eq!(p.int(), 0);
    p.lower(4);
    p.raise(4);
    assert_eq!(p.int(), 1);
}

#[test]
fn masking_holds_the_request() {
    let p = pc();
    p.master().ioport_write(1, 0x08);
    p.pulse(3);
    assert_eq!(p.int(), 0);
    assert_eq!(irr(p.master()), 0x08);
    p.master().ioport_write(1, 0x00);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x0b);
}

#[test]
fn priority_and_nesting() {
    let p = pc();
    p.pulse(3);
    p.pulse(1);
    assert_eq!(p.ack(), 0x09);
    // IRQ3 has a lower priority than IRQ1 in service.
    assert_eq!(p.int(), 0);
    // IRQ0 has a higher one and nests.
    p.pulse(0);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x08);
    assert_eq!(isr(p.master()), 0x03);
    // Non-specific EOI clears the highest priority in service first.
    p.master().ioport_write(0, 0x20);
    assert_eq!(isr(p.master()), 0x02);
    assert_eq!(p.int(), 0);
    p.master().ioport_write(0, 0x20);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x0b);
}

#[test]
fn specific_eoi_and_rotation() {
    let p = pc();
    p.pulse(5);
    p.pulse(6);
    assert_eq!(p.ack(), 0x0d);
    // Specific EOI for IRQ5.
    p.master().ioport_write(0, 0x65);
    assert_eq!(isr(p.master()), 0);
    assert_eq!(p.ack(), 0x0e);
    // Rotate on non-specific EOI: IRQ6 becomes the lowest priority.
    p.master().ioport_write(0, 0xa0);
    assert_eq!(p.master().state().priority_add, 7);
    p.pulse(6);
    p.pulse(7);
    assert_eq!(p.ack(), 0x0f);
    p.master().ioport_write(0, 0x20);
    // Set priority: IRQ3 lowest, so IRQ4 highest.
    p.master().ioport_write(0, 0xc3);
    assert_eq!(p.master().state().priority_add, 4);
    p.pulse(3);
    p.pulse(4);
    assert_eq!(p.ack(), 0x0c);
    // Rotate on specific EOI.
    p.master().ioport_write(0, 0xe4);
    assert_eq!(p.master().state().priority_add, 5);
    assert_eq!(p.ack(), 0x0e);
    p.master().ioport_write(0, 0x20);
    assert_eq!(p.ack(), 0x0b);
}

#[test]
fn cascade_through_irq2() {
    let p = pc();
    p.pulse(10);
    assert_eq!(p.int(), 1);
    assert_eq!(irr(p.master()), 0x04);
    assert_eq!(irr(p.slave()), 0x04);
    assert_eq!(p.ack(), 0x72);
    assert_eq!(p.int(), 0);
    assert_eq!(isr(p.slave()), 0x04);
    assert_eq!(isr(p.master()), 0x04);
    // Master IRQs above the cascade still get through, lower ones wait.
    p.pulse(3);
    assert_eq!(p.int(), 0);
    p.pulse(1);
    assert_eq!(p.ack(), 0x09);
    p.master().ioport_write(0, 0x20);
    // EOI to the slave and then the master, as Linux does.
    p.slave().ioport_write(0, 0x20);
    p.master().ioport_write(0, 0x20);
    assert_eq!(isr(p.master()), 0);
    assert_eq!(isr(p.slave()), 0);
    assert_eq!(p.ack(), 0x0b);
}

#[test]
fn slave_priority_orders_slave_lines() {
    let p = pc();
    p.pulse(14);
    p.pulse(8);
    assert_eq!(p.ack(), 0x70);
    p.slave().ioport_write(0, 0x20);
    p.master().ioport_write(0, 0x20);
    // The slave dropped its output on the acknowledge and raised it again on the EOI, a new
    // edge on IRQ2 that the master delivers after its own EOI.
    assert_eq!(irr(p.slave()), 0x40);
    assert_eq!(irr(p.master()), 0x04);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x76);
}

#[test]
fn spurious_irq7_on_master() {
    let p = pc();
    assert_eq!(p.ack(), 0x0f);
    assert_eq!(isr(p.master()), 0);
    // A level triggered request that goes away before the acknowledge.
    p.master().elcr_ioport_write(0x20);
    p.raise(5);
    assert_eq!(p.int(), 1);
    p.lower(5);
    assert_eq!(p.int(), 0);
    assert_eq!(p.ack(), 0x0f);
    assert_eq!(isr(p.master()), 0);
}

#[test]
fn spurious_irq7_on_slave() {
    let p = pc();
    p.slave().elcr_ioport_write(0x08);
    p.raise(11);
    // The slave line goes away but the master latched the edge on IRQ2.
    p.lower(11);
    assert_eq!(irr(p.master()), 0x04);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x77);
    assert_eq!(isr(p.slave()), 0);
    // The master still acknowledged its cascade input.
    assert_eq!(isr(p.master()), 0x04);
}

#[test]
fn elcr_level_mode() {
    let p = pc();
    // Writes are limited by elcr_mask: IRQ0, 1 and 2 on the master and IRQ8 and 13 on the
    // slave are always edge triggered.
    p.master().elcr_ioport_write(0xff);
    assert_eq!(p.master().elcr_ioport_read(), 0xf8);
    p.slave().elcr_ioport_write(0xff);
    assert_eq!(p.slave().elcr_ioport_read(), 0xde);
    p.master().elcr_ioport_write(0x08);
    p.slave().elcr_ioport_write(0x00);

    p.raise(3);
    assert_eq!(p.ack(), 0x0b);
    // A level request stays in IRR through the acknowledge.
    assert_eq!(irr(p.master()), 0x08);
    assert_eq!(p.int(), 0);
    // And comes back after the EOI while the line is still high.
    p.master().ioport_write(0, 0x20);
    assert_eq!(p.int(), 1);
    p.lower(3);
    assert_eq!(p.int(), 0);
    assert_eq!(irr(p.master()), 0);

    // Level mode for the whole chip through LTIM in ICW1.
    p.slave().ioport_write(0, 0x19);
    p.slave().ioport_write(1, 0x70);
    p.slave().ioport_write(1, 0x02);
    p.slave().ioport_write(1, 0x01);
    p.raise(12);
    assert_eq!(p.ack(), 0x74);
    assert_eq!(irr(p.slave()), 0x10);
}

#[test]
fn elcr_mmio_region() {
    let p = pc();
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    let elcr = p.pair.master.elcr_io();
    elcr.write(&cx, 0, AccessSize::B1, 0x28).unwrap();
    assert_eq!(elcr.read(&cx, 0, AccessSize::B1).unwrap(), 0x28);
    let base: &dyn MmioOps = &*p.pair.master;
    base.write(&cx, 1, AccessSize::B1, 0x5a).unwrap();
    assert_eq!(base.read(&cx, 1, AccessSize::B1).unwrap(), 0x5a);
    assert_eq!(base.impl_constraints().max, 1);
}

#[test]
fn poll_mode() {
    let p = pc();
    p.master().ioport_write(1, 0xff);
    p.pulse(6);
    p.master().ioport_write(0, 0x0c);
    // Poll ignores nothing but IMR, so a masked line is not reported.
    assert_eq!(p.master().ioport_read(0), 0);
    assert_eq!(p.master().state().poll, 0);
    p.master().ioport_write(1, 0x00);
    p.master().ioport_write(0, 0x0c);
    assert_eq!(p.master().ioport_read(0), 0x86);
    // The poll read acknowledged it.
    assert_eq!(isr(p.master()), 0x40);
    assert_eq!(irr(p.master()), 0);
    // Without poll, port 0 reads the register chosen by OCW3.
    assert_eq!(p.master().ioport_read(0), 0);
}

#[test]
fn auto_eoi() {
    let p = raw();
    p.master().ioport_write(0, 0x11);
    p.master().ioport_write(1, 0x08);
    p.master().ioport_write(1, 0x04);
    p.master().ioport_write(1, 0x03);
    p.pulse(4);
    assert_eq!(p.ack(), 0x0c);
    assert_eq!(isr(p.master()), 0);
    // Rotate in automatic EOI mode.
    p.master().ioport_write(0, 0x80);
    p.pulse(4);
    assert_eq!(p.ack(), 0x0c);
    assert_eq!(p.master().state().priority_add, 5);
    p.master().ioport_write(0, 0x00);
    assert_eq!(p.master().state().rotate_on_auto_eoi, 0);
}

#[test]
fn special_mask_mode() {
    let p = pc();
    p.pulse(1);
    assert_eq!(p.ack(), 0x09);
    p.pulse(3);
    assert_eq!(p.int(), 0);
    // Mask the line in service and enter special mask mode: lower priorities get through.
    p.master().ioport_write(0, 0x68);
    p.master().ioport_write(1, 0x02);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x0b);
    assert_eq!(isr(p.master()), 0x0a);
    // Leave it again.
    p.master().ioport_write(0, 0x48);
    assert_eq!(p.master().state().special_mask, 0);
}

#[test]
fn special_fully_nested_mode() {
    let p = raw();
    p.master().ioport_write(0, 0x11);
    p.master().ioport_write(1, 0x08);
    p.master().ioport_write(1, 0x04);
    p.master().ioport_write(1, 0x11);
    p.slave().ioport_write(0, 0x11);
    p.slave().ioport_write(1, 0x70);
    p.slave().ioport_write(1, 0x02);
    p.slave().ioport_write(1, 0x01);
    assert_eq!(p.master().state().special_fully_nested_mode, 1);
    p.pulse(13);
    assert_eq!(p.ack(), 0x75);
    // A higher priority slave request while IRQ2 is in service on the master.
    p.pulse(9);
    assert_eq!(p.int(), 1);
    assert_eq!(p.ack(), 0x71);
    assert_eq!(isr(p.slave()), 0x22);
}

#[test]
fn reset_clears_state() {
    let p = pc();
    p.master().elcr_ioport_write(0x08);
    p.raise(3);
    p.master().ioport_write(1, 0x40);
    p.master().reset();
    let s = p.master().state();
    assert_eq!((s.elcr, s.irr, s.imr, s.irq_base, s.last_irr), (0, 0, 0, 0, 0));
    assert_eq!(p.int(), 0);
    // ICW1 keeps the ELCR and the level requests it covers.
    p.master().elcr_ioport_write(0x08);
    p.raise(3);
    p.master().ioport_write(0, 0x11);
    assert_eq!(p.master().state().irr, 0x08);
    assert_eq!(p.master().state().init_state, 1);
}

#[test]
fn output_follows_pic_get_output() {
    let p = pc();
    assert!(!p.master().pic_get_output());
    p.pulse(0);
    assert!(p.master().pic_get_output());
    p.ack();
    assert!(!p.master().pic_get_output());
}

#[test]
fn vmstate_round_trip() {
    let p = pc();
    p.master().ioport_write(1, 0x02);
    p.master().elcr_ioport_write(0x08);
    p.pulse(4);
    assert_eq!(p.ack(), 0x0c);
    p.pulse(10);
    let (m, s) = (p.master().state(), p.slave().state());

    let q = raw();
    q.master().vmstate_load(&m);
    q.slave().vmstate_load(&s);
    assert_eq!(q.master().state(), m);
    assert_eq!(q.slave().state(), s);
    assert!(!q.slave().state().master);
    // Loading does not drive the INT line; the CPU brings its own interrupt request.
    assert_eq!(q.int(), 0);
    // The loaded pair acknowledges what the old one had pending, IRQ10 through the slave.
    assert_eq!(q.ack(), 0x72);
    assert_eq!(isr(q.master()), 0x14);
    assert_eq!(isr(q.slave()), 0x04);
    assert_eq!(q.master().ioport_read(1), 0x02);
    assert_eq!(q.master().elcr_ioport_read(), 0x08);

    // A master state loaded into the slave chip keeps the slave a slave.
    q.slave().vmstate_load(&m);
    assert!(!q.slave().state().master);
}
