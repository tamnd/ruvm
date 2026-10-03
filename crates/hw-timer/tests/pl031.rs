// SPDX-License-Identifier: GPL-2.0-or-later

//! The PL031 on a manual virtual clock, the way `-rtc clock=vm` runs under qtest, plus the
//! `virt` mapping at 0x09010000.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, UNIX_EPOCH};

use ruvm_base::ClockType;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND};
use ruvm_hw_timer::pl031::*;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MemTxResult, MemorySystem, MmioOps};

/// 2023-11-14 22:13:20 UTC, the host date the guest starts from.
const START: u32 = 1_700_000_000;

const SEC: i64 = NANOSECONDS_PER_SECOND;

struct Rig {
    clock: Arc<Clock>,
    rtc: Arc<Pl031>,
    irq: Arc<AtomicI32>,
}

impl Rig {
    fn new() -> Self {
        Self::at(0)
    }

    /// A PL031 created when the clock already reads `now` nanoseconds.
    fn at(now: i64) -> Self {
        let clock = Clock::manual(ClockType::Virtual);
        clock.advance_to(now);
        let start = UNIX_EPOCH + Duration::from_secs(u64::from(START));
        let rtc = Pl031::new(clock.clone(), start);
        let irq = Arc::new(AtomicI32::new(-1));
        let level = irq.clone();
        rtc.irq().connect(IrqLine::from_fn(move |l| level.store(l, Ordering::SeqCst)));
        Rig { clock, rtc, irq }
    }

    fn rd(&self, reg: u64) -> u32 {
        self.rtc.reg_read(reg)
    }

    fn wr(&self, reg: u64, val: u32) {
        self.rtc.reg_write(reg, u64::from(val));
    }

    fn irq(&self) -> i32 {
        self.irq.load(Ordering::SeqCst)
    }

    fn advance(&self, ns: i64) {
        let now = self.clock.get_ns();
        self.clock.advance_to(now + ns);
    }
}

#[test]
fn initial_registers() {
    let r = Rig::new();
    assert_eq!(r.rd(RTC_DR), START);
    assert_eq!(r.rd(RTC_MR), 0);
    assert_eq!(r.rd(RTC_LR), 0);
    assert_eq!(r.rd(RTC_CR), 1);
    assert_eq!(r.rd(RTC_IMSC), 0);
    assert_eq!(r.rd(RTC_RIS), 0);
    assert_eq!(r.rd(RTC_MIS), 0);
    // Write-only and unknown registers read as 0.
    assert_eq!(r.rd(RTC_ICR), 0);
    assert_eq!(r.rd(0x20), 0);
    assert_eq!(r.rtc.tick_offset(), START);
    assert_eq!(r.irq(), -1);
}

#[test]
fn id_registers() {
    let r = Rig::new();
    let ids: Vec<u32> = (0..8).map(|i| r.rd(RTC_PERIPHID0 + 4 * i)).collect();
    assert_eq!(ids, [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
    // Any offset inside a word reads that word's byte.
    assert_eq!(r.rd(0xfe5), 0x10);
    assert_eq!(r.rd(0xfff), 0xb1);
}

#[test]
fn counter_follows_the_clock() {
    let r = Rig::new();
    r.advance(SEC - 1);
    assert_eq!(r.rd(RTC_DR), START);
    r.advance(1);
    assert_eq!(r.rd(RTC_DR), START + 1);
    r.advance(59 * SEC + SEC / 2);
    assert_eq!(r.rd(RTC_DR), START + 60);
}

#[test]
fn start_date_counts_from_clock_zero() {
    // Created 10.5 s into the run: the date is START plus the 10 whole seconds.
    let r = Rig::at(10 * SEC + SEC / 2);
    assert_eq!(r.rtc.tick_offset(), START);
    assert_eq!(r.rd(RTC_DR), START + 10);
}

#[test]
fn writes_to_read_only_registers_are_ignored() {
    let r = Rig::new();
    r.wr(RTC_DR, 5);
    r.wr(RTC_RIS, 1);
    r.wr(RTC_MIS, 1);
    r.wr(RTC_CR, 0);
    assert_eq!(r.rd(RTC_DR), START);
    assert_eq!(r.rd(RTC_RIS), 0);
    assert_eq!(r.rd(RTC_CR), 1);
}

#[test]
fn load_register_sets_the_count() {
    let r = Rig::new();
    let changes = Arc::new(Mutex::new(Vec::new()));
    let c = changes.clone();
    r.rtc.set_rtc_change_handler(move |secs| c.lock().unwrap().push(secs));
    r.advance(3 * SEC + 7);
    r.wr(RTC_LR, 1000);
    assert_eq!(r.rd(RTC_LR), 1000);
    assert_eq!(r.rd(RTC_DR), 1000);
    assert_eq!(r.rtc.tick_offset(), 997);
    // QEMU passes qemu_get_timedate(tick_offset), which adds the reference date again.
    assert_eq!(*changes.lock().unwrap(), [i64::from(START) + 3 + 997]);
    r.advance(SEC);
    assert_eq!(r.rd(RTC_DR), 1001);

    // The counter wraps.
    r.wr(RTC_LR, u32::MAX);
    r.advance(SEC);
    assert_eq!(r.rd(RTC_DR), 0);
}

#[test]
fn alarm_raises_irq_when_unmasked() {
    let r = Rig::new();
    r.wr(RTC_IMSC, 0xff);
    assert_eq!(r.rd(RTC_IMSC), 1);
    assert_eq!(r.irq(), 0);
    r.wr(RTC_MR, START + 5);
    r.advance(5 * SEC - 1);
    assert_eq!(r.rd(RTC_RIS), 0);
    assert_eq!(r.irq(), 0);
    r.advance(1);
    assert_eq!(r.rd(RTC_RIS), 1);
    assert_eq!(r.rd(RTC_MIS), 1);
    assert_eq!(r.irq(), 1);

    r.wr(RTC_ICR, 1);
    assert_eq!(r.rd(RTC_RIS), 0);
    assert_eq!(r.irq(), 0);
}

#[test]
fn masked_alarm_sets_only_the_raw_status() {
    let r = Rig::new();
    r.wr(RTC_MR, START + 2);
    r.advance(2 * SEC);
    assert_eq!(r.rd(RTC_RIS), 1);
    assert_eq!(r.rd(RTC_MIS), 0);
    assert_eq!(r.irq(), 0);
    // Unmasking a pending alarm raises the line.
    r.wr(RTC_IMSC, 1);
    assert_eq!(r.irq(), 1);
    r.wr(RTC_IMSC, 0);
    assert_eq!(r.irq(), 0);
}

#[test]
fn match_equal_to_count_fires_at_once() {
    let r = Rig::new();
    r.wr(RTC_IMSC, 1);
    r.wr(RTC_MR, START);
    assert_eq!(r.rd(RTC_RIS), 1);
    assert_eq!(r.irq(), 1);
    assert_eq!(r.clock.next_deadline(), None);
}

#[test]
fn match_behind_the_count_waits_for_the_wrap() {
    let r = Rig::new();
    r.wr(RTC_MR, START - 1);
    let wait = i64::from(u32::MAX) * SEC;
    assert_eq!(r.clock.next_deadline(), Some(wait));
}

#[test]
fn loading_the_counter_rearms_the_alarm() {
    let r = Rig::new();
    r.wr(RTC_IMSC, 1);
    r.wr(RTC_MR, 100);
    // The match is far behind START; loading 98 puts it 2 s ahead.
    r.wr(RTC_LR, 98);
    assert_eq!(r.clock.next_deadline(), Some(2 * SEC));
    r.advance(2 * SEC);
    assert_eq!(r.irq(), 1);
    r.wr(RTC_ICR, 1);
    // Loading the match value itself fires right away.
    r.wr(RTC_LR, 100);
    assert_eq!(r.irq(), 1);
}

#[test]
fn mmio_ops_and_virt_mapping() {
    const BASE: u64 = 0x0901_0000;
    let r = Rig::new();
    let ops: &dyn MmioOps = &*r.rtc;
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    assert_eq!(ops.read(&cx, RTC_DR, AccessSize::B4).unwrap(), u64::from(START));
    ops.write(&cx, RTC_LR, AccessSize::B4, 42).unwrap();
    assert_eq!(ops.read(&cx, RTC_DR, AccessSize::B4).unwrap(), 42);

    let mem = Arc::new(MemorySystem::new());
    let sys = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_io(TYPE_PL031, u128::from(PL031_MMIO_SIZE), r.rtc.clone()).unwrap();
    mem.add_subregion(sys, BASE, io).unwrap();
    let space = mem.address_space_init(sys, "memory").unwrap();
    let mut b = [0u8; 4];
    assert_eq!(space.read(BASE + RTC_DR, MemTxAttrs::UNSPECIFIED, &mut b), MemTxResult::OK);
    assert_eq!(u32::from_le_bytes(b), 42);
    assert_eq!(space.write_u32(BASE + RTC_LR, MemTxAttrs::UNSPECIFIED, 7), MemTxResult::OK);
    assert_eq!(r.rd(RTC_DR), 7);
    let mut b = [0u8; 1];
    assert_eq!(space.read(BASE + 0xfe0, MemTxAttrs::UNSPECIFIED, &mut b), MemTxResult::OK);
    assert_eq!(b[0], 0x31);
}

#[test]
fn dropping_the_device_drops_its_alarm() {
    let Rig { clock, rtc, irq } = Rig::new();
    rtc.reg_write(RTC_IMSC, 1);
    rtc.reg_write(RTC_MR, u64::from(START + 1));
    assert_eq!(clock.next_deadline(), Some(SEC));
    let weak = Arc::downgrade(&rtc);
    drop(rtc);
    assert!(weak.upgrade().is_none());
    assert_eq!(clock.next_deadline(), None);
    clock.advance_to(2 * SEC);
    assert_eq!(irq.load(Ordering::SeqCst), 0);
}
