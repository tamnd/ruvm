// SPDX-License-Identifier: GPL-2.0-or-later

//! tests/unit/test-throttle.c, and tests of the throttle groups.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ruvm_qapi::types::{ThrottleGroupProperties, ThrottleLimits};

use super::groups::{self, ThrottleGroupMember, test_util};
use super::*;

const NS: i64 = NANOSECONDS_PER_SECOND;

fn double_cmp(x: f64, y: f64) -> bool {
    (x - y).abs() < 1e-6
}

fn virtual_clock() -> Arc<dyn Clock> {
    Arc::new(VirtualClock::new(1))
}

#[test]
fn leak_bucket() {
    let cfg = ThrottleConfig::new();
    let mut bkt = cfg.buckets[BucketType::BpsTotal as usize];
    bkt.avg = 150;
    bkt.max = 15;
    bkt.level = 1.5;

    // Leak an operation's worth of time.
    bkt.leak(NS / 150);
    assert_eq!((bkt.avg, bkt.max), (150, 15));
    assert!(double_cmp(bkt.level, 0.5));

    // Leak again, emptying the bucket.
    bkt.leak(NS / 150);
    assert!(double_cmp(bkt.level, 0.0));

    // The level does not go lower.
    bkt.leak(NS / 150);
    assert_eq!((bkt.avg, bkt.max), (150, 15));
    assert!(double_cmp(bkt.level, 0.0));

    // burst_level leaks too, with a burst length over 1.
    bkt.burst_level = 6.0;
    bkt.max = 250;
    bkt.burst_length = 2;
    bkt.leak(NS / 100);
    assert!(double_cmp(bkt.burst_level, 3.5));
    bkt.leak(NS / 100);
    assert!(double_cmp(bkt.burst_level, 1.0));
    bkt.leak(NS / 100);
    assert!(double_cmp(bkt.burst_level, 0.0));
    bkt.leak(NS / 100);
    assert!(double_cmp(bkt.burst_level, 0.0));
}

#[test]
fn compute_wait() {
    let cfg = ThrottleConfig::new();
    let mut bkt = cfg.buckets[BucketType::BpsTotal as usize];

    // No limit.
    bkt.avg = 0;
    bkt.max = 15;
    bkt.level = 1.5;
    assert_eq!(bkt.compute_wait(), 0);

    // Zero delta.
    bkt.avg = 150;
    bkt.max = 15;
    bkt.level = 15.0;
    assert_eq!(bkt.compute_wait(), 0);

    // Below zero delta.
    bkt.level = 9.0;
    assert_eq!(bkt.compute_wait(), 0);

    // Half an operation above max: the time of half an operation.
    bkt.level = 15.5;
    assert_eq!(bkt.compute_wait(), NS / 150 / 2);

    // I/O for 2.2 seconds at the rate of bkt.max.
    bkt.burst_length = 2;
    bkt.level = 0.0;
    bkt.avg = 10;
    bkt.max = 200;
    for i in 0..22 {
        let units = bkt.max as f64 / 10.0;
        bkt.level += units;
        bkt.burst_level += units;
        bkt.leak(NS / 10);
        let wait = bkt.compute_wait();
        assert!(double_cmp(bkt.burst_level, 0.0));
        assert!(double_cmp(bkt.level, f64::from(i + 1) * (bkt.max - bkt.avg) as f64 / 10.0));
        // Bursts last the 2 seconds of burst_length, plus 100 ms because the bucket leaked
        // meanwhile. Then the request has to wait.
        let result = if i < 21 { 0 } else { (1.8 * NS as f64) as i64 };
        assert_eq!(wait, result, "iteration {i}");
    }
}

fn check_cleared(ts: &ThrottleState) {
    assert_eq!(ts.previous_leak, 0);
    assert_eq!(ts.cfg.op_size, 0);
    for b in &ts.cfg.buckets {
        assert_eq!((b.avg, b.max, b.level), (0, 0, 0.0));
    }
}

#[test]
fn init() {
    let ts = ThrottleState::new();
    let tt = ThrottleTimers::new(virtual_clock(), true, true);
    assert!(tt.timers[0].is_some());
    assert!(tt.timers[1].is_some());
    check_cleared(&ts);
}

#[test]
fn init_readonly() {
    let ts = ThrottleState::new();
    let tt = ThrottleTimers::new(virtual_clock(), true, false);
    assert!(tt.timers[0].is_some());
    assert!(tt.timers[1].is_none());
    check_cleared(&ts);
}

#[test]
fn init_writeonly() {
    let ts = ThrottleState::new();
    let tt = ThrottleTimers::new(virtual_clock(), false, true);
    assert!(tt.timers[0].is_none());
    assert!(tt.timers[1].is_some());
    check_cleared(&ts);
}

#[test]
fn destroy() {
    let mut tt = ThrottleTimers::new(virtual_clock(), true, true);
    tt.destroy();
    assert!(tt.timers.iter().all(Option::is_none));
}

#[test]
fn have_timer() {
    let mut tt = ThrottleTimers::new(virtual_clock(), false, false);
    assert!(!tt.are_initialized());
    tt = ThrottleTimers::new(virtual_clock(), true, true);
    assert!(tt.are_initialized());
    tt.destroy();
    assert!(!tt.are_initialized());
}

#[test]
fn detach_attach() {
    let mut tt = ThrottleTimers::new(virtual_clock(), true, true);
    assert!(tt.are_initialized());
    tt.detach();
    assert!(!tt.are_initialized());
    tt.attach();
    assert!(tt.are_initialized());
}

#[test]
fn config_functions() {
    use BucketType::*;
    let mut orig = ThrottleConfig::default();
    let set = |c: &mut ThrottleConfig, t, avg, max, level| {
        let b = c.bucket_mut(t);
        b.avg = avg;
        b.max = max;
        b.level = level;
    };
    set(&mut orig, BpsTotal, 153, 0, 45.0);
    set(&mut orig, BpsRead, 56, 56, 65.0);
    set(&mut orig, BpsWrite, 1, 120, 23.0);
    set(&mut orig, OpsTotal, 150, 150, 1.0);
    set(&mut orig, OpsRead, 69, 400, 90.0);
    set(&mut orig, OpsWrite, 23, 500, 75.0);
    orig.op_size = 1;

    let clock = virtual_clock();
    let mut ts = ThrottleState::new();
    assert_eq!(ts.previous_leak, 0);
    ts.config(clock.as_ref(), &orig);
    // throttle_config() set previous_leak.
    assert_ne!(ts.previous_leak, 0);

    let fin = ts.get_config();
    for (t, avg, max) in [
        (BpsTotal, 153, 0),
        (BpsRead, 56, 56),
        (BpsWrite, 1, 120),
        (OpsTotal, 150, 150),
        (OpsRead, 69, 400),
        (OpsWrite, 23, 500),
    ] {
        assert_eq!((fin.bucket(t).avg, fin.bucket(t).max), (avg, max));
    }
    assert_eq!(fin.op_size, 1);
    // The buckets were emptied.
    assert!(fin.buckets.iter().all(|b| b.level == 0.0));
}

fn set_cfg_value(cfg: &mut ThrottleConfig, is_max: bool, index: usize, value: i64) {
    let b = &mut cfg.buckets[index];
    if is_max {
        b.max = value as u64;
        // With max set, avg is never 0.
        b.avg = b.avg.max(1);
    } else {
        b.avg = value as u64;
    }
}

#[test]
fn enabled() {
    let cfg = ThrottleConfig::new();
    assert!(!cfg.enabled());
    for i in 0..BUCKETS_COUNT {
        let mut cfg = ThrottleConfig::new();
        set_cfg_value(&mut cfg, false, i, 150);
        assert!(cfg.is_valid().is_ok());
        assert!(cfg.enabled());
    }
    for i in 0..BUCKETS_COUNT {
        let mut cfg = ThrottleConfig::new();
        set_cfg_value(&mut cfg, false, i, -150);
        assert!(cfg.is_valid().is_err());
    }
}

fn conflicts_for_one_set(is_max: bool, total: BucketType, read: BucketType, write: BucketType) {
    let (total, read, write) = (total as usize, read as usize, write as usize);
    let check = |sets: &[usize], ok: bool| {
        let mut cfg = ThrottleConfig::new();
        for &i in sets {
            set_cfg_value(&mut cfg, is_max, i, 1);
        }
        assert_eq!(cfg.is_valid().is_ok(), ok);
        if !ok {
            assert_eq!(
                cfg.is_valid().unwrap_err().message(),
                "bps/iops/max total values and read/write values cannot be used at the same time"
            );
        }
    };
    check(&[], true);
    check(&[total, read], false);
    check(&[total, write], false);
    check(&[total, read, write], false);
    check(&[total], true);
    check(&[read, write], true);
}

#[test]
fn conflicting_config() {
    use BucketType::*;
    conflicts_for_one_set(false, BpsTotal, BpsRead, BpsWrite);
    conflicts_for_one_set(false, OpsTotal, OpsRead, OpsWrite);
    conflicts_for_one_set(true, BpsTotal, BpsRead, BpsWrite);
    conflicts_for_one_set(true, OpsTotal, OpsRead, OpsWrite);
}

#[test]
fn is_valid() {
    for (value, ok) in [(-1, false), (0, true), (1, true)] {
        for is_max in [false, true] {
            for i in 0..BUCKETS_COUNT {
                let mut cfg = ThrottleConfig::new();
                set_cfg_value(&mut cfg, is_max, i, value);
                assert_eq!(cfg.is_valid().is_ok(), ok);
            }
        }
    }
}

#[test]
fn ranges() {
    const MAX: u64 = THROTTLE_VALUE_MAX;
    for i in 0..BUCKETS_COUNT {
        let mut cfg = ThrottleConfig::new();
        let valid = |cfg: &ThrottleConfig| cfg.is_valid().is_ok();
        let msg = |cfg: &ThrottleConfig| cfg.is_valid().unwrap_err().message().to_string();

        // avg = 0 disables throttling but is valid.
        cfg.buckets[i].avg = 0;
        assert!(valid(&cfg));
        assert!(!cfg.enabled());

        cfg.buckets[i].avg = 1;
        assert!(valid(&cfg));
        cfg.buckets[i].avg = MAX;
        assert!(valid(&cfg));
        cfg.buckets[i].max = MAX;
        assert!(valid(&cfg));

        // Values over THROTTLE_VALUE_MAX are not allowed.
        cfg.buckets[i].avg = MAX + 1;
        assert_eq!(msg(&cfg), "bps/iops/max values must be within [0, 1000000000000000]");
        cfg.buckets[i].avg = MAX;
        cfg.buckets[i].max = MAX + 1;
        assert!(!valid(&cfg));

        // burst_length must be between 1 and THROTTLE_VALUE_MAX.
        let mut set = |avg, max, len| {
            cfg.buckets[i].avg = avg;
            cfg.buckets[i].max = max;
            cfg.buckets[i].burst_length = len;
            cfg
        };
        assert_eq!(msg(&set(1, 1, 0)), "the burst length cannot be 0");
        assert!(valid(&set(1, 1, 1)));
        assert!(valid(&set(1, 1, MAX)));
        assert_eq!(msg(&set(1, 1, MAX + 1)), "burst length too high for this burst rate");

        // burst_length * max cannot exceed THROTTLE_VALUE_MAX.
        assert!(valid(&set(1, 2, MAX / 2)));
        assert!(!valid(&set(1, 3, MAX / 2)));
        assert!(valid(&set(1, MAX, 1)));
        assert!(!valid(&set(1, MAX, 2)));
        assert_eq!(msg(&set(1, 0, 2)), "burst length set without burst rate");
    }
}

#[test]
fn max_is_missing_limit() {
    for i in 0..BUCKETS_COUNT {
        let mut cfg = ThrottleConfig::new();
        let mut check = |max, avg| {
            cfg.buckets[i].max = max;
            cfg.buckets[i].avg = avg;
            cfg.is_valid().map_err(|e| e.message().to_string())
        };
        assert_eq!(
            check(100, 0).unwrap_err(),
            "bps_max/iops_max require corresponding bps/iops values"
        );
        assert!(check(0, 0).is_ok());
        assert!(check(0, 100).is_ok());
        assert_eq!(check(30, 100).unwrap_err(), "bps_max/iops_max cannot be lower than bps/iops");
        assert!(check(100, 100).is_ok());
    }
}

#[test]
fn iops_size_is_missing_limit() {
    let mut cfg = ThrottleConfig::new();
    cfg.op_size = 4096;
    assert_eq!(cfg.is_valid().unwrap_err().message(), "iops size requires an iops value to be set");
}

fn do_test_accounting(
    is_ops: bool,
    size: u64,
    avg: u64,
    op_size: u64,
    total_result: f64,
    read_result: f64,
    write_result: f64,
) -> bool {
    use BucketType::*;
    let to_test =
        if is_ops { [OpsTotal, OpsRead, OpsWrite] } else { [BpsTotal, BpsRead, BpsWrite] };
    let mut cfg = ThrottleConfig::new();
    for t in to_test {
        cfg.bucket_mut(t).avg = avg;
    }
    cfg.op_size = op_size;

    let mut ts = ThrottleState::new();
    ts.config(virtual_clock().as_ref(), &cfg);
    ts.account(ThrottleDirection::Read, size);
    ts.account(ThrottleDirection::Write, size);

    double_cmp(ts.cfg.bucket(to_test[0]).level, total_result)
        && double_cmp(ts.cfg.bucket(to_test[1]).level, read_result)
        && double_cmp(ts.cfg.bucket(to_test[2]).level, write_result)
}

#[test]
fn accounting() {
    // bps
    assert!(do_test_accounting(false, 512, 150, 0, 1024.0, 512.0, 512.0));
    assert!(do_test_accounting(false, 2 * 512, 150, 0, 2048.0, 1024.0, 1024.0));
    assert!(do_test_accounting(false, 2 * 512, 150, 17, 2048.0, 1024.0, 1024.0));

    // ops
    assert!(do_test_accounting(true, 512, 150, 0, 2.0, 1.0, 1.0));
    assert!(do_test_accounting(true, 2 * 512, 150, 0, 2.0, 1.0, 1.0));
    // A jumbo request of 64 units with an operation size of 13 units.
    let (t, r) = ((64.0 * 2.0) / 13.0, 64.0 / 13.0);
    assert!(do_test_accounting(true, 64 * 512, 150, 13 * 512, t, r, r));
    assert!(do_test_accounting(true, 64 * 512, 300, 13 * 512, t, r, r));
}

#[test]
fn schedule_timer() {
    let clock = Arc::new(VirtualClock::new(NS));
    let mut tt = ThrottleTimers::new(clock.clone(), true, true);
    let mut cfg = ThrottleConfig::new();
    cfg.bucket_mut(BucketType::OpsTotal).avg = 10;
    let mut ts = ThrottleState::new();
    ts.config(clock.as_ref(), &cfg);

    // The bucket holds avg / 10 = 1 operation before throttling.
    ts.account(ThrottleDirection::Read, 512);
    assert!(!ts.schedule_timer(&mut tt, ThrottleDirection::Read));
    ts.account(ThrottleDirection::Read, 512);
    assert!(ts.schedule_timer(&mut tt, ThrottleDirection::Read));
    assert_eq!(tt.timers[0].unwrap().expire_time(), Some(NS + NS / 10));
    assert!(!tt.pending(ThrottleDirection::Write));

    // An armed timer stays as it is.
    clock.advance(NS / 20);
    assert!(ts.schedule_timer(&mut tt, ThrottleDirection::Read));
    assert_eq!(tt.timers[0].unwrap().expire_time(), Some(NS + NS / 10));

    clock.advance(NS / 20);
    assert!(!ts.schedule_timer(&mut tt, ThrottleDirection::Read));
}

#[test]
fn limits_to_config() {
    let mut cfg = ThrottleConfig::new();
    let l = ThrottleLimits { bps_read: Some(100), iops_write: Some(20), ..Default::default() };
    cfg.apply_limits(&l).unwrap();
    assert_eq!(cfg.bucket(BucketType::BpsRead).avg, 100);
    assert_eq!(cfg.bucket(BucketType::OpsWrite).avg, 20);

    let back = cfg.to_limits();
    assert_eq!(back.bps_read, Some(100));
    assert_eq!(back.bps_total_max_length, Some(1));
    assert_eq!(back.iops_size, Some(0));

    let l = ThrottleLimits { iops_read_max_length: Some(1 << 32), ..Default::default() };
    assert_eq!(
        cfg.apply_limits(&l).unwrap_err().message(),
        "iops-read-max-length value must be in the range [0, 4294967295]"
    );
    let l = ThrottleLimits { bps_total: Some(1), ..Default::default() };
    assert_eq!(
        cfg.apply_limits(&l).unwrap_err().message(),
        "bps/iops/max total values and read/write values cannot be used at the same time"
    );
    let mut cfg = ThrottleConfig::new();
    let l = ThrottleLimits { bps_total: Some(-1), ..Default::default() };
    assert_eq!(
        cfg.apply_limits(&l).unwrap_err().message(),
        "bps/iops/max values must be within [0, 1000000000000000]"
    );
}

#[test]
fn groups() {
    let m1 = ThrottleGroupMember::new();
    let m2 = ThrottleGroupMember::new();
    let m3 = ThrottleGroupMember::new();
    assert!(!m1.is_registered() && !m2.is_registered() && !m3.is_registered());

    m1.register("test-groups-bar");
    m2.register("test-groups-foo");
    m3.register("test-groups-bar");
    assert!(m1.is_registered() && m2.is_registered() && m3.is_registered());
    assert_eq!(m1.group_name().as_deref(), Some("test-groups-bar"));
    assert_eq!(m2.group_name().as_deref(), Some("test-groups-foo"));
    assert!(m1.same_group(&m3));
    assert!(!m1.same_group(&m2));
    assert!(test_util::has_timers(&m1));

    // The config of one member is the config of the group.
    let mut cfg1 = ThrottleConfig::new();
    cfg1.bucket_mut(BucketType::BpsRead).avg = 500000;
    cfg1.bucket_mut(BucketType::BpsWrite).avg = 285000;
    cfg1.bucket_mut(BucketType::OpsRead).avg = 20000;
    cfg1.bucket_mut(BucketType::OpsWrite).avg = 12000;
    m1.config(&cfg1);
    let cfg1 = m1.get_config().unwrap();
    let mut cfg2 = m3.get_config().unwrap();
    assert_eq!(cfg1, cfg2);

    cfg2.bucket_mut(BucketType::BpsRead).avg = 4547;
    m3.config(&cfg2);
    assert_eq!(m1.get_config().unwrap(), m3.get_config().unwrap());
    assert_eq!(m1.get_config().unwrap().bucket(BucketType::BpsRead).avg, 4547);

    assert!(groups::throttle_group_exists("test-groups-bar"));
    m1.unregister();
    m2.unregister();
    assert!(groups::throttle_group_exists("test-groups-bar"));
    assert!(!groups::throttle_group_exists("test-groups-foo"));
    m3.unregister();
    assert!(!groups::throttle_group_exists("test-groups-bar"));
    assert!(!m1.is_registered() && !m2.is_registered() && !m3.is_registered());
}

#[test]
fn round_robin_token() {
    let a = ThrottleGroupMember::new();
    let b = ThrottleGroupMember::new();
    let c = ThrottleGroupMember::new();
    // Registered newest first: the list is c, b, a.
    a.register("test-rr");
    b.register("test-rr");
    c.register("test-rr");
    let (ia, ib, ic) =
        (test_util::member_id(&a), test_util::member_id(&b), test_util::member_id(&c));
    let rd = ThrottleDirection::Read;

    // The first member holds the token. Nobody waits, so the token goes to the asker.
    test_util::set_token(&a, rd);
    assert_eq!(test_util::next_token_with(&b, &[], rd), ib);
    // From a, the circular list goes on at c.
    assert_eq!(test_util::next_token_with(&b, &[(&c, 1), (&b, 1)], rd), ic);
    assert_eq!(test_util::next_token_with(&b, &[(&b, 1)], rd), ib);
    // Only the token holder waits: it is found after a full turn.
    assert_eq!(test_util::next_token_with(&b, &[(&a, 1)], rd), ia);
    test_util::set_token(&c, rd);
    assert_eq!(test_util::next_token_with(&a, &[(&b, 2), (&a, 1)], rd), ib);
}

fn add_group(name: &str, limits: ThrottleLimits, clock: Arc<dyn Clock>) {
    let props = ThrottleGroupProperties { limits: Some(limits), ..Default::default() };
    groups::throttle_group_add_with_clock(name, &props, clock).unwrap();
}

/// Waits until `f` holds, for the threads under test.
fn wait_until(f: impl Fn() -> bool) {
    for _ in 0..2000 {
        if f() {
            return;
        }
        thread::sleep(Duration::from_millis(1));
    }
    panic!("timed out");
}

#[test]
fn intercept_waits_for_the_clock() {
    let clock = Arc::new(VirtualClock::new(NS));
    add_group(
        "test-intercept",
        ThrottleLimits { iops_total: Some(10), ..Default::default() },
        clock.clone(),
    );
    let m = Arc::new(ThrottleGroupMember::new());
    m.register("test-intercept");

    // avg / 10 = 1 request fits in the bucket; the second one is over it and the third waits
    // for a tenth of a second.
    m.io_limits_intercept(512, ThrottleDirection::Read);
    m.io_limits_intercept(512, ThrottleDirection::Read);
    let t = {
        let m = m.clone();
        thread::spawn(move || m.io_limits_intercept(512, ThrottleDirection::Read))
    };
    wait_until(|| m.pending_reqs(ThrottleDirection::Read) == 1);
    // Half the wait is not enough.
    clock.advance(NS / 20);
    thread::sleep(Duration::from_millis(30));
    assert!(!t.is_finished());
    clock.advance(NS / 20);
    t.join().unwrap();
    assert_eq!(m.pending_reqs(ThrottleDirection::Read), 0);

    // A drained section lets requests through without waiting.
    m.io_limits_disable_begin();
    for _ in 0..5 {
        m.io_limits_intercept(512, ThrottleDirection::Write);
    }
    m.io_limits_disable_end();

    m.unregister();
    groups::throttle_group_del("test-intercept").unwrap();
}

#[test]
fn restart_releases_waiters() {
    let clock = Arc::new(VirtualClock::new(NS));
    add_group(
        "test-restart",
        ThrottleLimits { bps_write: Some(1000), ..Default::default() },
        clock.clone(),
    );
    let a = Arc::new(ThrottleGroupMember::new());
    let b = Arc::new(ThrottleGroupMember::new());
    a.register("test-restart");
    b.register("test-restart");

    // Fill the bucket, far beyond what the clock will ever leak.
    a.io_limits_intercept(1 << 20, ThrottleDirection::Write);
    let spawn = |m: &Arc<ThrottleGroupMember>| {
        let m = m.clone();
        thread::spawn(move || m.io_limits_intercept(100, ThrottleDirection::Write))
    };
    let ta = spawn(&a);
    let tb = spawn(&b);
    wait_until(|| {
        a.pending_reqs(ThrottleDirection::Write) == 1
            && b.pending_reqs(ThrottleDirection::Write) == 1
    });

    // Draining one member lets its request go; the other still waits.
    a.io_limits_disable_begin();
    ta.join().unwrap();
    thread::sleep(Duration::from_millis(30));
    assert!(!tb.is_finished());
    assert_eq!(b.pending_reqs(ThrottleDirection::Write), 1);
    a.io_limits_disable_end();

    // New limits reset the buckets and restart the queue.
    let mut cfg = ThrottleConfig::new();
    cfg.bucket_mut(BucketType::BpsWrite).avg = 1000;
    b.config(&cfg);
    tb.join().unwrap();

    // The group has references, so object-del refuses.
    assert_eq!(
        groups::throttle_group_del("test-restart").unwrap_err().message(),
        "Cannot delete throttle group 'test-restart' with active references"
    );
    a.unregister();
    b.unregister();
    groups::throttle_group_del("test-restart").unwrap();
    assert!(!groups::throttle_group_exists("test-restart"));
}

#[test]
fn object_add_errors() {
    let add = |name: &str, props: ThrottleGroupProperties| {
        groups::throttle_group_add(name, &props).map_err(|e| e.message().to_string())
    };
    let p = ThrottleGroupProperties { x_iops_total: Some(-1), ..Default::default() };
    assert_eq!(add("test-obj-a", p).unwrap_err(), "Property values cannot be negative");
    let p = ThrottleGroupProperties { x_bps_read_max_length: Some(1 << 32), ..Default::default() };
    assert_eq!(
        add("test-obj-a", p).unwrap_err(),
        "x-bps-read-max-length value must be in therange [0, 4294967295]"
    );
    let p = ThrottleGroupProperties { x_iops_total_max: Some(10), ..Default::default() };
    assert_eq!(
        add("test-obj-a", p).unwrap_err(),
        "bps_max/iops_max require corresponding bps/iops values"
    );
    let p = ThrottleGroupProperties {
        limits: Some(ThrottleLimits {
            bps_read: Some(1),
            bps_total: Some(1),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert_eq!(
        add("test-obj-a", p).unwrap_err(),
        "bps/iops/max total values and read/write values cannot be used at the same time"
    );

    let p = ThrottleGroupProperties {
        limits: Some(ThrottleLimits { iops_total: Some(100), ..Default::default() }),
        ..Default::default()
    };
    add("test-obj-a", p.clone()).unwrap();
    assert_eq!(add("test-obj-a", p).unwrap_err(), "A group with this name already exists");
    assert_eq!(
        groups::throttle_group_set_property("test-obj-a", "x-iops-total", 5).unwrap_err().message(),
        "Property cannot be set after initialization"
    );
    let l = groups::throttle_group_get_limits("test-obj-a").unwrap();
    assert_eq!(l.iops_total, Some(100));
    assert_eq!(l.bps_read_max_length, Some(1));
    groups::throttle_group_set_limits(
        "test-obj-a",
        &ThrottleLimits { iops_total_max: Some(200), ..Default::default() },
    )
    .unwrap();
    let l = groups::throttle_group_get_limits("test-obj-a").unwrap();
    assert_eq!((l.iops_total, l.iops_total_max), (Some(100), Some(200)));

    // A group a member created on the fly is not an object.
    let m = ThrottleGroupMember::new();
    m.register("test-obj-b");
    assert_eq!(
        groups::throttle_group_del("test-obj-b").unwrap_err().message(),
        "object 'test-obj-b' not found"
    );
    drop(m);
    groups::throttle_group_del("test-obj-a").unwrap();
    assert!(!groups::throttle_group_exists("test-obj-a"));
}

fn drive_err(g: &crate::graph::BlockGraph, params: &str) -> String {
    let ide = crate::drive::BlockInterfaceType::Ide;
    g.drive_new(params, ide).unwrap_err().message().to_string()
}

#[test]
fn drive_throttling_errors() {
    let g = crate::graph::BlockGraph::new();
    let err = |p: &str| drive_err(&g, p);
    assert_eq!(
        err("if=none,throttling.bps-total=x"),
        "Parameter 'throttling.bps-total' expects a number"
    );
    assert_eq!(
        err("if=none,bps=1,bps_rd=1"),
        "bps/iops/max total values and read/write values cannot be used at the same time"
    );
    assert_eq!(
        err("if=none,iops_max=10"),
        "bps_max/iops_max require corresponding bps/iops values"
    );
    assert_eq!(err("if=none,iops_size=512"), "iops size requires an iops value to be set");
    assert_eq!(
        err("if=none,bps=1,throttling.bps-total-max-length=2"),
        "burst length set without burst rate"
    );
    assert_eq!(
        err("if=none,bps=1,throttling.bps-total-max-length=0"),
        "the burst length cannot be 0"
    );
    assert_eq!(
        err("if=none,bps=1000000000000001"),
        "bps/iops/max values must be within [0, 1000000000000000]"
    );
    assert_eq!(err("if=none,iops=1,iops_rd=1"), err("if=none,bps=1,bps_wr=1"));
    assert_eq!(
        err("if=none,iops=1,throttling.iops-total=2"),
        "'throttling.iops-total' and its alias 'iops' can't be used at the same time"
    );
    assert_eq!(err("if=none,stats-intervals.x=1"), "Invalid option stats-intervals.x");
    assert_eq!(
        err("if=none,stats-intervals.0=1,stats-intervals.2=1"),
        "Invalid option stats-intervals.2"
    );
    assert_eq!(
        err("if=none,driver=null-co,stats-intervals.0=10,stats-intervals.1=0"),
        "Invalid interval length: 0"
    );
    assert_eq!(
        err("if=none,driver=null-co,stats-intervals.0=4294967296"),
        "Invalid interval length: 4294967296"
    );
    assert_eq!(
        err("if=none,stats-account-failed=maybe"),
        "Parameter 'stats-account-failed' expects 'on' or 'off'"
    );
    // A failed drive leaves no group behind.
    assert!(!groups::throttle_group_exists("drv-fail"));
}

#[test]
fn drive_throttling_groups() {
    let g = crate::graph::BlockGraph::new();
    let ide = crate::drive::BlockInterfaceType::Ide;
    g.drive_new("if=none,id=drv-t0,driver=null-co,iops=100,iops_max=200", ide).unwrap();
    let (name, cfg) = g.backend("drv-t0").unwrap().io_limits().unwrap();
    assert_eq!(name, "drv-t0");
    assert_eq!(cfg.bucket(BucketType::OpsTotal).avg, 100);
    assert_eq!(cfg.bucket(BucketType::OpsTotal).max, 200);

    // Two drives in one group; the second drive's limits are the group's.
    g.drive_new("if=none,id=drv-t1,driver=null-co,bps=1000,group=drv-shared", ide).unwrap();
    g.drive_new("if=none,id=drv-t2,bps_rd=5000,throttling.group=drv-shared", ide).unwrap();
    let b1 = g.backend("drv-t1").unwrap();
    let b2 = g.backend("drv-t2").unwrap();
    assert!(b1.io_limits().is_some() && b2.io_limits().is_some());
    let (name, cfg) = b1.io_limits().unwrap();
    assert_eq!(name, "drv-shared");
    assert_eq!(cfg.bucket(BucketType::BpsTotal).avg, 0);
    assert_eq!(cfg.bucket(BucketType::BpsRead).avg, 5000);
    // A group made by a drive is not the monitor's to delete.
    assert_eq!(
        groups::throttle_group_del("drv-shared").unwrap_err().message(),
        "object 'drv-shared' not found"
    );

    // Zero limits are no throttling.
    g.drive_new("if=none,id=drv-t3,driver=null-co,iops=0", ide).unwrap();
    assert!(g.backend("drv-t3").unwrap().io_limits().is_none());

    b1.io_limits_update_group("drv-other");
    assert_eq!(b1.io_limits().unwrap().0, "drv-other");
    b1.io_limits_disable();
    assert!(b1.io_limits().is_none());
    assert!(!groups::throttle_group_exists("drv-other"));
}

#[test]
fn backend_accounting() {
    let g = crate::graph::BlockGraph::new();
    let ide = crate::drive::BlockInterfaceType::Ide;
    g.drive_new(
        "if=none,id=acct0,driver=null-co,size=1048576,stats-account-invalid=off,stats-intervals.0=60",
        ide,
    )
    .unwrap();
    let blk = g.backend("acct0").unwrap();
    let mut buf = [0u8; 4096];
    blk.pread(0, &mut buf).unwrap();
    blk.pread(4096, &mut buf[..512]).unwrap();
    assert!(blk.pread(1 << 20, &mut buf).is_err());
    blk.flush().unwrap();
    // The drive has no write permission: the write fails after its range check.
    assert!(blk.pdiscard(0, 512).is_err());

    let s = blk.stats();
    assert_eq!((s.rd_bytes, s.rd_operations), (4608, 2));
    assert_eq!(s.invalid_rd_operations, 1);
    assert_eq!(s.flush_operations, 1);
    assert_eq!(
        (s.unmap_operations, s.failed_unmap_operations, s.invalid_unmap_operations),
        (0, 0, 0)
    );
    assert!(!s.account_invalid && s.account_failed);
    assert!(s.idle_time_ns.is_some());
    assert_eq!(s.timed_stats.len(), 1);
    assert_eq!(s.timed_stats[0].interval_length, 60);
}

#[test]
fn backend_drain_releases_throttled_requests() {
    let clock = Arc::new(VirtualClock::new(NS));
    add_group(
        "test-blk-drain",
        ThrottleLimits { iops_total: Some(10), ..Default::default() },
        clock,
    );
    let g = crate::graph::BlockGraph::new();
    let ide = crate::drive::BlockInterfaceType::Ide;
    g.drive_new("if=none,id=drain0,driver=null-co,iops=10,group=test-blk-drain", ide).unwrap();
    let blk = g.backend("drain0").unwrap();
    let mut buf = [0u8; 512];
    blk.pread(0, &mut buf).unwrap();
    blk.pread(0, &mut buf).unwrap();

    // The clock does not move, so only a drained section lets the third read go.
    let t = {
        let blk = blk.clone();
        thread::spawn(move || {
            let mut buf = [0u8; 512];
            blk.pread(0, &mut buf).unwrap();
        })
    };
    // A drain_all() of another test running at the same time (a reopen) may let it go early,
    // so only a read still waiting shows that the drain releases it; if it did not, the drain
    // would never end.
    wait_until(|| blk.in_flight() == 1 || t.is_finished());
    blk.drain();
    t.join().unwrap();
    assert_eq!(blk.stats().rd_operations, 3);

    blk.io_limits_disable();
    groups::throttle_group_del("test-blk-drain").unwrap();
}
