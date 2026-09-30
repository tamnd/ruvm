// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the HPET on a manual virtual clock. QEMU has no hpet qtest of its
//! own, so these check the behaviour of hw/timer/hpet.c directly.

use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_timer::hpet::*;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

/// What pc machines pass as `hpet-intcap`.
const PC_INTCAP: u32 = 0x00ff_0104;

type Log = Arc<Mutex<Vec<(u32, i32)>>>;

struct Rig {
    clock: Arc<Clock>,
    hpet: Arc<Hpet>,
    irqs: Log,
    pit: Log,
    msi: Arc<Mutex<Vec<(u64, u32)>>>,
}

fn rig_with(props: HpetProperties) -> Rig {
    let clock = Clock::manual(ClockType::Virtual);
    let mut fw = HpetFwConfig::new();
    let hpet = Hpet::realize(&clock, props, &mut fw, HPET_BASE).unwrap();
    let irqs: Log = Arc::default();
    let l = irqs.clone();
    let handler: Arc<dyn Fn(u32, i32) + Send + Sync> =
        Arc::new(move |n, level| l.lock().unwrap().push((n, level)));
    for n in 0..HPET_NUM_IRQ_ROUTES {
        hpet.irq(n).connect(IrqLine::new(handler.clone(), n as u32));
    }
    let pit: Log = Arc::default();
    let p = pit.clone();
    hpet.pit_enabled().connect(IrqLine::from_fn(move |level| p.lock().unwrap().push((0, level))));
    let msi: Arc<Mutex<Vec<(u64, u32)>>> = Arc::default();
    let m = msi.clone();
    hpet.set_msi_handler(Some(Arc::new(move |a, d| m.lock().unwrap().push((a, d)))));
    Rig { clock, hpet, irqs, pit, msi }
}

fn rig() -> Rig {
    rig_with(HpetProperties { intcap: PC_INTCAP, ..HpetProperties::default() })
}

fn tn(n: u64, reg: u64) -> u64 {
    0x100 + n * 0x20 + reg
}

impl Rig {
    fn rd(&self, addr: u64) -> u64 {
        self.hpet.mmio_read(addr, 8)
    }

    fn wr(&self, addr: u64, v: u64) {
        self.hpet.mmio_write(addr, 8, v);
    }

    fn step(&self, ns: i64) {
        self.clock.advance_to(self.clock.get_ns() + ns);
    }

    /// Every edge seen on the routed lines since the last call.
    fn take_all(&self) -> Vec<(u32, i32)> {
        std::mem::take(&mut *self.irqs.lock().unwrap())
    }

    /// The same without line 0. Enabling the HPET arms the timers left at their reset
    /// comparator of all ones, which QEMU turns into a deadline in the past, so they pop at once
    /// and lower their default route 0.
    fn take(&self) -> Vec<(u32, i32)> {
        self.take_all().into_iter().filter(|e| e.0 != 0).collect()
    }
}

#[test]
fn capability_and_defaults_match_qemu() {
    let r = rig();
    // 3 timers, LegacyReplacementRoute, vendor 8086, rev 1, period 10 ns in femtoseconds.
    assert_eq!(r.rd(HPET_ID), 0x0098_9680_8086_a201);
    assert_eq!(r.hpet.mmio_read(HPET_PERIOD, 4), 10_000_000);
    assert_eq!(r.hpet.mmio_read(HPET_ID, 4), 0x8086_a201);
    assert_eq!(r.rd(HPET_CFG), 0);
    assert_eq!(r.rd(HPET_STATUS), 0);
    for n in 0..3 {
        assert_eq!(r.rd(tn(n, HPET_TN_CFG)), (u64::from(PC_INTCAP) << 32) | 0x30);
        assert_eq!(r.rd(tn(n, HPET_TN_CMP)), u64::MAX);
        assert_eq!(r.rd(tn(n, HPET_TN_ROUTE)), 0);
    }
    // Timers beyond `timers` read as zero.
    assert_eq!(r.rd(tn(3, HPET_TN_CFG)), 0);
    // The ID register is read only.
    r.wr(HPET_ID, 0);
    assert_eq!(r.hpet.mmio_read(HPET_ID, 4), 0x8086_a201);
    // pit_enabled was raised by reset before anything was connected.
    assert!(r.pit.lock().unwrap().is_empty());

    let r = rig_with(HpetProperties { timers: 24, msi: true, intcap: PC_INTCAP });
    assert_eq!(r.rd(HPET_ID) & HPET_ID_NUM_TIM_MASK, 23 << 8);
    assert_eq!(r.rd(tn(23, HPET_TN_CFG)) & 0xffff, 0x8030);
}

#[test]
fn realize_errors_and_fw_cfg() {
    let clock = Clock::manual(ClockType::Virtual);
    let mut fw = HpetFwConfig::new();
    let bad = HpetProperties { timers: 2, intcap: 4, msi: false };
    let e = Hpet::realize(&clock, bad, &mut fw, HPET_BASE).unwrap_err();
    assert_eq!(e.message(), "hpet.num_timers must be between 3 and 24");
    let e = Hpet::realize(&clock, HpetProperties::default(), &mut fw, HPET_BASE).unwrap_err();
    assert_eq!(e.message(), "hpet.hpet-intcap not initialized");
    assert_eq!(fw.count, u8::MAX);

    let props = HpetProperties { intcap: 4, ..HpetProperties::default() };
    let mut all = Vec::new();
    for i in 0..8u64 {
        all.push(Hpet::realize(&clock, props, &mut fw, HPET_BASE + i * 0x1000).unwrap());
    }
    assert_eq!(fw.count, 8);
    assert_eq!(all[3].hpet_id(), 3);
    assert_eq!(
        fw.hpet[3],
        HpetFwEntry {
            event_timer_block_id: 0x8086_a201,
            address: HPET_BASE + 0x3000,
            ..HpetFwEntry::default()
        }
    );
    let e = Hpet::realize(&clock, props, &mut fw, HPET_BASE).unwrap_err();
    assert_eq!(e.message(), "Only 8 instances of HPET are allowed");
    assert_eq!(fw.to_bytes().len(), 121);
}

#[test]
fn counter_runs_at_10ns_per_tick() {
    let r = rig();
    r.step(1000);
    // Stopped until enabled.
    assert_eq!(r.rd(HPET_COUNTER), 0);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(1000);
    assert_eq!(r.rd(HPET_COUNTER), 100);
    r.step(5);
    assert_eq!(r.rd(HPET_COUNTER), 100);
    r.step(5);
    assert_eq!(r.rd(HPET_COUNTER), 101);

    // Disabling freezes it, and it resumes from where it stopped.
    r.wr(HPET_CFG, 0);
    r.step(10_000);
    assert_eq!(r.rd(HPET_COUNTER), 101);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(100);
    assert_eq!(r.rd(HPET_COUNTER), 111);

    // A counter write while halted, in two halves.
    r.wr(HPET_CFG, 0);
    r.hpet.mmio_write(HPET_COUNTER, 4, 0x89ab_cdef);
    r.hpet.mmio_write(HPET_COUNTER + 4, 4, 0x0123_4567);
    assert_eq!(r.rd(HPET_COUNTER), 0x0123_4567_89ab_cdef);
    assert_eq!(r.hpet.mmio_read(HPET_COUNTER + 4, 4), 0x0123_4567);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(20);
    assert_eq!(r.rd(HPET_COUNTER), 0x0123_4567_89ab_cdf1);
}

#[test]
fn one_shot_edge_interrupt() {
    let r = rig();
    r.wr(tn(0, HPET_TN_CMP), 100);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_ENABLE | (2 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(999);
    assert!(r.take().is_empty());
    r.step(1);
    assert_eq!(r.take(), [(2, 1), (2, 0)]);
    // Edge interrupts leave no status bit.
    assert_eq!(r.rd(HPET_STATUS), 0);
    // A one shot does not come back.
    r.step(1_000_000);
    assert!(r.take().is_empty());
}

#[test]
fn level_interrupt_sets_status_until_cleared() {
    let r = rig();
    r.wr(tn(1, HPET_TN_CMP), 50);
    r.wr(tn(1, HPET_TN_CFG), HPET_TN_ENABLE | HPET_TN_TYPE_LEVEL | (5 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(500);
    assert_eq!(r.take(), [(5, 1)]);
    assert_eq!(r.rd(HPET_STATUS), 2);
    // Writing zero bits clears nothing.
    r.wr(HPET_STATUS, 1);
    assert_eq!(r.rd(HPET_STATUS), 2);
    r.wr(HPET_STATUS, 2);
    assert_eq!(r.rd(HPET_STATUS), 0);
    assert_eq!(r.take(), [(5, 0)]);
}

#[test]
fn disabled_timer_still_sets_status() {
    let r = rig();
    r.wr(tn(0, HPET_TN_CMP), 10);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_TYPE_LEVEL | (3 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(100);
    assert_eq!(r.rd(HPET_STATUS), 1);
    assert_eq!(r.take(), [(3, 0)]);
    // Enabling the timer with the status bit set raises the line.
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_ENABLE | HPET_TN_TYPE_LEVEL | (3 << HPET_TN_INT_ROUTE_SHIFT));
    assert_eq!(r.take(), [(3, 1)]);
}

#[test]
fn periodic_interrupts() {
    let r = rig();
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.wr(
        tn(2, HPET_TN_CFG),
        HPET_TN_ENABLE | HPET_TN_PERIODIC | HPET_TN_SETVAL | (7 << HPET_TN_INT_ROUTE_SHIFT),
    );
    // SETVAL lets the comparator be written along with the period.
    r.wr(tn(2, HPET_TN_CMP), 1000);
    assert_eq!(r.rd(tn(2, HPET_TN_CFG)) & HPET_TN_SETVAL, 0);
    assert_eq!(r.rd(tn(2, HPET_TN_CMP)), 1000);

    // The manual clock stops exactly on each deadline, so when `hpet_timer()` runs the counter
    // equals the comparator and is not after it. QEMU then leaves the comparator alone and the
    // 1 us clamp in `hpet_arm()` re-arms the same match one microsecond later, where the
    // comparator finally moves on by one period. Each period therefore pops twice, the same as
    // under qtest.
    let mut pops = Vec::new();
    for _ in 0..25 {
        r.step(1000);
        for e in r.take() {
            if e.1 == 1 {
                pops.push((r.clock.get_ns(), r.rd(tn(2, HPET_TN_CMP))));
            }
        }
    }
    assert_eq!(pops, [(10_000, 1000), (11_000, 2000), (20_000, 2000), (21_000, 3000)]);
}

#[test]
fn periodic_is_clamped_to_one_microsecond() {
    let r = rig();
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.wr(
        tn(0, HPET_TN_CFG),
        HPET_TN_ENABLE | HPET_TN_PERIODIC | HPET_TN_SETVAL | (6 << HPET_TN_INT_ROUTE_SHIFT),
    );
    r.wr(tn(0, HPET_TN_CMP), 1);
    r.step(10_000);
    // One pop per microsecond, not one per tick.
    let pulses = r.take().iter().filter(|e| e.1 == 1).count();
    assert_eq!(pulses, 10);
}

#[test]
fn thirty_two_bit_one_shot_fires_on_wrap_too() {
    let r = rig();
    r.hpet.mmio_write(HPET_COUNTER, 8, 0xffff_ff00);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_32BIT);
    // Switching to 32 bit mode truncates the comparator.
    assert_eq!(r.rd(tn(0, HPET_TN_CMP)), 0xffff_ffff);
    // The high half of the comparator cannot be written in 32 bit mode.
    r.hpet.mmio_write(tn(0, HPET_TN_CMP) + 4, 4, 0x1234);
    assert_eq!(r.rd(tn(0, HPET_TN_CMP)), 0xffff_ffff);
    r.hpet.mmio_write(tn(0, HPET_TN_CMP), 8, 0x5555_0000_0000_0100);
    assert_eq!(r.rd(tn(0, HPET_TN_CMP)), 0x100);

    r.wr(tn(0, HPET_TN_CFG), HPET_TN_32BIT | HPET_TN_ENABLE | (4 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    // 0x100 ticks to the wrap.
    r.step(0x100 * 10 - 1);
    assert!(r.take().is_empty());
    r.step(1);
    assert_eq!(r.take(), [(4, 1), (4, 0)]);
    assert_eq!(r.rd(HPET_COUNTER), 0x1_0000_0000);
    // Then another 0x100 ticks to the match.
    r.step(0x100 * 10 - 1);
    assert!(r.take().is_empty());
    r.step(1);
    assert_eq!(r.take(), [(4, 1), (4, 0)]);
    r.step(100_000_000);
    assert!(r.take().is_empty());
}

#[test]
fn thirty_two_bit_periodic_wraps_the_comparator() {
    let r = rig();
    r.hpet.mmio_write(HPET_COUNTER, 8, 0xffff_ff00);
    r.wr(
        tn(0, HPET_TN_CFG),
        HPET_TN_32BIT
            | HPET_TN_ENABLE
            | HPET_TN_PERIODIC
            | HPET_TN_SETVAL
            | (9 << HPET_TN_INT_ROUTE_SHIFT),
    );
    r.wr(tn(0, HPET_TN_CMP), 0xffff_ff80);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    // The match pops, then the clamp pop moves the comparator on by the period, which wraps.
    r.step(0x80 * 10);
    assert_eq!(r.take(), [(9, 1), (9, 0)]);
    assert_eq!(r.rd(tn(0, HPET_TN_CMP)), 0xffff_ff80);
    r.step(1000);
    assert_eq!(r.take(), [(9, 1), (9, 0)]);
    assert_eq!(r.rd(tn(0, HPET_TN_CMP)), 0xffff_ff00);
}

#[test]
fn legacy_replacement_routing() {
    let r = rig();
    let pit_in = r.hpet.legacy_irq_in(HPET_LEGACY_PIT_INT);
    let rtc_in = r.hpet.legacy_irq_in(HPET_LEGACY_RTC_INT);

    // Outside legacy mode the PIT and RTC pass through to IRQ0 and IRQ8.
    pit_in.raise();
    rtc_in.raise();
    assert_eq!(r.take_all(), [(0, 1), (8, 1)]);

    r.wr(HPET_CFG, HPET_CFG_ENABLE | HPET_CFG_LEGACY);
    assert_eq!(*r.pit.lock().unwrap(), [(0, 0)]);
    assert_eq!(r.take_all(), [(0, 0), (8, 0)]);
    // Now they are cut off.
    pit_in.pulse();
    rtc_in.lower();
    rtc_in.raise();
    assert!(r.take_all().is_empty());

    // Timer 0 goes to IRQ0 and timer 1 to IRQ8 whatever their route field says.
    r.wr(tn(0, HPET_TN_CMP), 10);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_ENABLE | (20 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(tn(1, HPET_TN_CMP), 20);
    r.wr(tn(1, HPET_TN_CFG), HPET_TN_ENABLE | (21 << HPET_TN_INT_ROUTE_SHIFT));
    r.wr(tn(2, HPET_TN_CMP), 30);
    r.wr(tn(2, HPET_TN_CFG), HPET_TN_ENABLE | (22 << HPET_TN_INT_ROUTE_SHIFT));
    r.step(1000);
    assert_eq!(r.take_all(), [(0, 1), (0, 0), (8, 1), (8, 0), (22, 1), (22, 0)]);

    // Leaving legacy mode gives the PIT its line back and restores the RTC level.
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    assert_eq!(*r.pit.lock().unwrap(), [(0, 0), (0, 1)]);
    assert_eq!(r.take_all(), [(0, 0), (8, 1)]);
}

#[test]
fn fsb_delivery() {
    let r = rig_with(HpetProperties { msi: true, intcap: PC_INTCAP, ..HpetProperties::default() });
    assert_ne!(r.rd(tn(0, HPET_TN_CFG)) & HPET_TN_FSB_CAP, 0);
    // Data in the low half, address in the high half.
    r.hpet.mmio_write(tn(0, HPET_TN_ROUTE), 4, 0x4031);
    r.hpet.mmio_write(tn(0, HPET_TN_ROUTE) + 4, 4, 0xfee0_1000);
    assert_eq!(r.rd(tn(0, HPET_TN_ROUTE)), 0xfee0_1000_0000_4031);
    r.wr(tn(0, HPET_TN_CMP), 5);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_ENABLE | HPET_TN_FSB_ENABLE);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(100);
    assert_eq!(*r.msi.lock().unwrap(), [(0xfee0_1000, 0x4031)]);
    assert!(r.take().is_empty());
}

#[test]
fn reset_stops_everything() {
    let r = rig();
    r.wr(tn(0, HPET_TN_CMP), 100);
    r.wr(tn(0, HPET_TN_CFG), HPET_TN_ENABLE);
    r.wr(HPET_CFG, HPET_CFG_ENABLE);
    r.step(500);
    let mut fw = HpetFwConfig::new();
    r.hpet.reset(&mut fw);
    assert_eq!(fw.hpet[0].event_timer_block_id, 0x8086_a201);
    assert_eq!(fw.hpet[0].address, HPET_BASE);
    assert_eq!(r.rd(HPET_CFG), 0);
    assert_eq!(r.rd(HPET_COUNTER), 0);
    r.take_all();
    r.step(10_000);
    assert!(r.take_all().is_empty());
}

#[test]
fn mmio_ops_sizes() {
    let r = rig();
    let ops: &dyn MmioOps = &*r.hpet;
    assert_eq!(ops.valid().min, 4);
    assert_eq!(ops.valid().max, 8);
    assert_eq!(ops.impl_constraints().max, 8);
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    assert_eq!(ops.read(&cx, HPET_ID, AccessSize::B4).unwrap(), 0x8086_a201);
    assert_eq!(ops.read(&cx, HPET_PERIOD, AccessSize::B4).unwrap(), 10_000_000);
    ops.write(&cx, HPET_CFG, AccessSize::B4, HPET_CFG_ENABLE).unwrap();
    r.step(100);
    assert_eq!(ops.read(&cx, HPET_COUNTER, AccessSize::B8).unwrap(), 10);
}
