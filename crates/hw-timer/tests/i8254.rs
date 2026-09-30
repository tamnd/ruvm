// SPDX-License-Identifier: GPL-2.0-or-later

//! Behaviour of the i8254 PIT against a manual virtual clock.

use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_timer::i8254::{I8254, PIT_FREQ, PitChannelInfo};
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

type Log = Arc<Mutex<Vec<(i64, i32)>>>;

/// The first nanosecond at which `n` PIT ticks have elapsed since time 0.
fn tick_ns(n: u64) -> i64 {
    (n * 1_000_000_000).div_ceil(u64::from(PIT_FREQ)) as i64
}

/// Ticks elapsed at `ns` since time 0, as the device counts them.
fn ns_ticks(ns: i64) -> u64 {
    (u128::from(ns as u64) * u128::from(PIT_FREQ) / 1_000_000_000) as u64
}

/// A PIT at 0x40 on a fresh clock, with channel 0's IRQ recorded as (time, level).
fn setup() -> (Arc<Clock>, Arc<I8254>, Log) {
    let clock = Clock::manual(ClockType::Virtual);
    let pit = I8254::new(&clock, 0x40);
    let log: Log = Arc::default();
    let (l, c) = (log.clone(), Arc::downgrade(&clock));
    pit.irq.connect(IrqLine::from_fn(move |level| {
        let now = c.upgrade().map_or(0, |c| c.get_ns());
        l.lock().unwrap().push((now, level));
    }));
    (clock, pit, log)
}

/// Programs `channel` with a control word and a 16 bit count, LSB then MSB.
fn program(pit: &I8254, channel: u8, mode: u8, count: u16) {
    pit.ioport_write(3, (channel << 6) | 0x30 | (mode << 1));
    pit.ioport_write(u64::from(channel), count as u8);
    pit.ioport_write(u64::from(channel), (count >> 8) as u8);
}

/// Latches `channel`'s count and reads it back as a word.
fn latch_read(pit: &I8254, channel: u8) -> u16 {
    pit.ioport_write(3, channel << 6);
    let lo = pit.ioport_read(u64::from(channel));
    let hi = pit.ioport_read(u64::from(channel));
    u16::from(lo) | (u16::from(hi) << 8)
}

/// Rising edges in the log.
fn rising_edges(log: &Log) -> Vec<i64> {
    let log = log.lock().unwrap();
    let mut last = 0;
    let mut edges = Vec::new();
    for &(t, level) in log.iter() {
        if level != 0 && last == 0 {
            edges.push(t);
        }
        last = level;
    }
    edges
}

#[test]
fn reset_state() {
    let (_clock, pit, _log) = setup();
    assert_eq!(
        pit.get_channel_info(0),
        PitChannelInfo { gate: 1, mode: 3, initial_count: 0x10000, out: 1 }
    );
    assert_eq!(pit.get_channel_info(1).gate, 1);
    assert_eq!(pit.get_channel_info(2).gate, 0);
    // Channel 0 runs its IRQ timer from reset on.
    assert!(pit.irq_timer_expire_time().is_some());
    // The command port is write only.
    assert_eq!(pit.ioport_read(3), 0);
}

#[test]
fn mode2_rate_generator_irq() {
    let (clock, pit, log) = setup();
    clock.advance_to(1000);
    log.lock().unwrap().clear();
    program(&pit, 0, 2, 100);
    // Loading the count puts OUT low and arms the timer.
    assert_eq!(*log.lock().unwrap(), [(1000, 0)]);
    assert!(pit.irq_timer_expire_time().is_some());

    let period = tick_ns(100);
    clock.advance_to(1000 + 10 * period + period / 2);
    let edges = rising_edges(&log);
    assert_eq!(edges.len(), 10, "{:?}", log.lock().unwrap());
    for (k, &t) in edges.iter().enumerate() {
        // Each pulse starts when the count reaches a multiple of the reload value.
        assert_eq!(ns_ticks(t - 1000), 100 * (k as u64 + 1), "edge {k} at {t}");
    }
}

#[test]
fn mode2_count_and_latch() {
    let (clock, pit, _log) = setup();
    program(&pit, 0, 2, 100);
    clock.advance_to(tick_ns(30));
    assert_eq!(latch_read(&pit, 0), 70);
    // The count reloads after reaching 1.
    clock.advance_to(tick_ns(130));
    assert_eq!(latch_read(&pit, 0), 70);

    // A latched value holds until both bytes have been read, and a second latch is ignored.
    clock.advance_to(tick_ns(140));
    pit.ioport_write(3, 0x00);
    clock.advance_to(tick_ns(150));
    pit.ioport_write(3, 0x00);
    assert_eq!(pit.ioport_read(0), 60);
    assert_eq!(pit.ioport_read(0), 0);
    // Unlatched reads follow the counter, LSB then MSB.
    assert_eq!(pit.ioport_read(0), 50);
    assert_eq!(pit.ioport_read(0), 0);
}

#[test]
fn mode0_interrupt_on_terminal_count() {
    let (clock, pit, log) = setup();
    log.lock().unwrap().clear();
    program(&pit, 0, 0, 1000);
    assert_eq!(pit.get_channel_info(0).out, 0);
    clock.advance_to(tick_ns(999));
    assert_eq!(pit.get_channel_info(0).out, 0);
    assert_eq!(latch_read(&pit, 0), 1);
    clock.advance_to(tick_ns(1000) + 5);
    assert_eq!(pit.get_channel_info(0).out, 1);
    assert_eq!(*log.lock().unwrap().last().unwrap(), (tick_ns(1000), 1));
    // No more transitions: the timer is idle and the counter wraps.
    assert!(pit.irq_timer_expire_time().is_none());
    clock.advance_to(tick_ns(1010));
    assert_eq!(latch_read(&pit, 0), 0xfff6);
    clock.advance_to(tick_ns(100_000));
    assert_eq!(log.lock().unwrap().last().unwrap().1, 1);
}

#[test]
fn mode3_square_wave() {
    let (clock, pit, log) = setup();
    log.lock().unwrap().clear();
    program(&pit, 0, 3, 100);
    assert_eq!(*log.lock().unwrap(), [(0, 1)]);
    // Counts down by two.
    clock.advance_to(tick_ns(10));
    assert_eq!(latch_read(&pit, 0), 80);
    clock.advance_to(tick_ns(49));
    assert_eq!(pit.get_channel_info(0).out, 1);
    clock.advance_to(tick_ns(50));
    assert_eq!(pit.get_channel_info(0).out, 0);
    clock.advance_to(tick_ns(100));
    assert_eq!(pit.get_channel_info(0).out, 1);
    clock.advance_to(tick_ns(1000) - 1);
    let mut levels: Vec<i32> = log.lock().unwrap().iter().map(|&(_, l)| l).collect();
    levels.dedup();
    // High and low halves alternate, 10 periods minus the last edge.
    assert_eq!(levels.len(), 20);
    assert!(levels.iter().enumerate().all(|(i, &l)| l == i32::from(i % 2 == 0)));
}

#[test]
fn mode4_software_strobe() {
    let (clock, pit, _log) = setup();
    program(&pit, 0, 4, 10);
    clock.advance_to(tick_ns(9));
    assert_eq!(pit.get_channel_info(0).out, 0);
    clock.advance_to(tick_ns(10));
    assert_eq!(pit.get_channel_info(0).out, 1);
    clock.advance_to(tick_ns(11));
    assert_eq!(pit.get_channel_info(0).out, 0);
    assert!(pit.irq_timer_expire_time().is_none());
}

#[test]
fn read_back_command() {
    let (clock, pit, _log) = setup();
    program(&pit, 0, 2, 100);
    program(&pit, 1, 0, 50);
    clock.advance_to(tick_ns(20));
    // Latch count and status of channels 0 and 1.
    pit.ioport_write(3, 0xc6);
    clock.advance_to(tick_ns(40));
    // Status first: OUT, access mode 3 (word), mode, binary.
    assert_eq!(pit.ioport_read(0), 0x34);
    assert_eq!(pit.ioport_read(0), 80);
    assert_eq!(pit.ioport_read(0), 0);
    assert_eq!(pit.ioport_read(1), 0x30);
    assert_eq!(pit.ioport_read(1), 30);
    assert_eq!(pit.ioport_read(1), 0);

    // Status only, with OUT high after the terminal count of channel 1.
    clock.advance_to(tick_ns(60));
    pit.ioport_write(3, 0xe4);
    assert_eq!(pit.ioport_read(1), 0xb0);
    // Count only.
    pit.ioport_write(3, 0xd4);
    assert_eq!(latch_count_word(&pit, 1), 0xfff6);
}

fn latch_count_word(pit: &I8254, channel: u64) -> u16 {
    let lo = pit.ioport_read(channel);
    let hi = pit.ioport_read(channel);
    u16::from(lo) | (u16::from(hi) << 8)
}

#[test]
fn lsb_and_msb_access_modes() {
    let (clock, pit, _log) = setup();
    // LSB only, mode 0.
    pit.ioport_write(3, 0x10);
    pit.ioport_write(0, 0x80);
    assert_eq!(pit.get_channel_info(0).initial_count, 0x80);
    clock.advance_to(tick_ns(0x10));
    assert_eq!(pit.ioport_read(0), 0x70);
    assert_eq!(pit.ioport_read(0), 0x70);
    // MSB only.
    pit.ioport_write(3, 0x20);
    pit.ioport_write(0, 0x01);
    assert_eq!(pit.get_channel_info(0).initial_count, 0x100);
    clock.advance_to(tick_ns(0x10 + 0x20));
    assert_eq!(pit.ioport_read(0), 0);
    // A zero count means 65536.
    pit.ioport_write(0, 0);
    assert_eq!(pit.get_channel_info(0).initial_count, 0x10000);
    // A latch in LSB mode reads out one byte.
    pit.ioport_write(3, 0x10);
    pit.ioport_write(0, 0x40);
    clock.advance_to(clock.get_ns() + tick_ns(4));
    pit.ioport_write(3, 0x00);
    let latched = pit.ioport_read(0);
    assert!(latched == 0x3c || latched == 0x3d, "{latched:#x}");
    assert_eq!(pit.state().channels[0].count_latched, 0);
}

#[test]
fn speaker_gate_restarts_channel2() {
    let (clock, pit, _log) = setup();
    program(&pit, 2, 3, 1000);
    clock.advance_to(tick_ns(700));
    assert_eq!(pit.get_channel_info(2).out, 0);
    // The rising edge of the gate reloads the counter.
    pit.set_gate(2, 1);
    let info = pit.get_channel_info(2);
    assert_eq!(info, PitChannelInfo { gate: 1, mode: 3, initial_count: 1000, out: 1 });
    assert_eq!(pit.state().channels[2].count_load_time, tick_ns(700));
    // Setting it again is not an edge.
    clock.advance_to(tick_ns(800));
    pit.set_gate(2, 1);
    assert_eq!(pit.state().channels[2].count_load_time, tick_ns(700));
    pit.set_gate(2, 0);
    assert_eq!(pit.get_channel_info(2).gate, 0);
}

#[test]
fn irq_control_from_hpet() {
    let (clock, pit, log) = setup();
    program(&pit, 0, 2, 100);
    let control = pit.irq_control_in();
    control.lower();
    assert!(pit.irq_timer_expire_time().is_none());
    log.lock().unwrap().clear();
    clock.advance_to(tick_ns(1000));
    assert!(log.lock().unwrap().is_empty());
    // Loading a count while disabled does not touch the line either.
    program(&pit, 0, 2, 100);
    assert!(log.lock().unwrap().is_empty());
    control.raise();
    assert_eq!(log.lock().unwrap().len(), 1);
    assert!(pit.irq_timer_expire_time().is_some());
}

#[test]
fn mmio_ops() {
    let (clock, pit, _log) = setup();
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    let ops: &dyn MmioOps = &*pit;
    assert_eq!(ops.impl_constraints().max, 1);
    ops.write(&cx, 3, AccessSize::B1, 0x34).unwrap();
    ops.write(&cx, 0, AccessSize::B1, 0x10).unwrap();
    ops.write(&cx, 0, AccessSize::B1, 0x27).unwrap();
    assert_eq!(pit.get_channel_info(0).initial_count, 0x2710);
    clock.advance_to(tick_ns(0x10));
    assert_eq!(ops.read(&cx, 0, AccessSize::B1).unwrap(), 0x00);
    assert_eq!(ops.read(&cx, 0, AccessSize::B1).unwrap(), 0x27);
}
