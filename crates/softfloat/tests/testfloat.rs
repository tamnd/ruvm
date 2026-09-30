// SPDX-License-Identifier: GPL-2.0-or-later

//! Checks ruvm-softfloat against Berkeley TestFloat 3e.
//!
//! `testfloat_gen` writes test cases, each with the operands, the result Berkeley SoftFloat 3e
//! computes and its exception flags. Every case is run through ruvm-softfloat with an x86
//! style status (x87 for extF80) and must give the same result and flags. This mirrors what
//! QEMU's own `tests/fp/fp-test` checks:
//!
//! * a NaN result matches any NaN, since NaN payloads are implementation defined;
//! * the integer result of an invalid conversion is not compared, only the flags;
//! * only the five IEEE flags are compared;
//! * `0 * inf + qNaN` uses Arm's rule (default NaN and invalid), which is what SoftFloat's
//!   8086-SSE specialization does, where QEMU's x86 rule returns the quiet NaN quietly.
//!
//! TestFloat's `_r_minMag` integer conversions are covered by running the plain conversions
//! with `-rminMag`, since `testfloat_gen` does not generate them. Functions QEMU lacks (for
//! example `extF80_mulAdd`, `f16_rem` and `f128_to_f16`) are not run. At level 1 the full run
//! is about 313 million cases over 120 functions and takes about ten minutes.
//!
//! Round to odd is not run for roundToInt, the integer conversions and extF80, where
//! TestFloat means something else by it (round toward zero) or QEMU does not implement it.
//! QEMU always raises inexact when rounding to an integer, so those run with `-exact`.
//!
//! The test is ignored because it needs `testfloat_gen`. Build it with
//! `tests/testfloat/build.sh`, which prints its path, and run:
//!
//! ```text
//! RUVM_TESTFLOAT_GEN=/path/to/testfloat_gen \
//!     cargo test -p ruvm-softfloat --test testfloat -- --ignored --nocapture
//! ```
//!
//! `RUVM_TESTFLOAT_LEVEL` picks the TestFloat level (1 or 2, default 1) and
//! `RUVM_TESTFLOAT_FILTER` limits the run to functions whose name contains the given text.

#![cfg(unix)]

mod common;

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

use common::Req;
use common::fmt::*;
use common::op::*;
use ruvm_softfloat::{FloatStatus, FloatX80RoundPrec, InfZeroNanRule, RoundMode};

/// A TestFloat operand or result type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ty {
    F16,
    F32,
    F64,
    X80,
    F128,
    I32,
    I64,
    U32,
    U64,
    Bool,
}

impl Ty {
    fn name(self) -> &'static str {
        match self {
            Ty::F16 => "f16",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
            Ty::X80 => "extF80",
            Ty::F128 => "f128",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::U32 => "ui32",
            Ty::U64 => "ui64",
            Ty::Bool => "bool",
        }
    }

    /// The ruvm format number of a float type.
    fn fmt(self) -> u16 {
        match self {
            Ty::F16 => F16,
            Ty::F32 => F32,
            Ty::F64 => F64,
            Ty::X80 => X80,
            Ty::F128 => F128,
            _ => unreachable!(),
        }
    }

    fn is_float(self) -> bool {
        matches!(self, Ty::F16 | Ty::F32 | Ty::F64 | Ty::X80 | Ty::F128)
    }

    /// Whether `v` is a NaN of this float type.
    fn is_nan(self, v: u128) -> bool {
        match self {
            Ty::F16 => v & 0x7c00 == 0x7c00 && v & 0x3ff != 0,
            Ty::F32 => v as u32 & 0x7f80_0000 == 0x7f80_0000 && v & 0x7f_ffff != 0,
            Ty::F64 => {
                let v = v as u64;
                v & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 && v << 12 != 0
            }
            Ty::X80 => (v >> 64) as u16 & 0x7fff == 0x7fff && (v as u64) << 1 != 0,
            Ty::F128 => (v >> 112) as u16 & 0x7fff == 0x7fff && v << 16 != 0,
            _ => false,
        }
    }

    /// The mask of the bits a value of this type uses.
    fn mask(self) -> u128 {
        match self {
            Ty::F16 => 0xffff,
            Ty::F32 | Ty::I32 | Ty::U32 => 0xffff_ffff,
            Ty::F64 | Ty::I64 | Ty::U64 => u64::MAX.into(),
            Ty::X80 => (1 << 80) - 1,
            Ty::F128 => u128::MAX,
            Ty::Bool => 1,
        }
    }
}

/// The comparison a compare function makes.
#[derive(Clone, Copy, Debug)]
enum Cmp {
    Eq,
    Le,
    Lt,
}

/// One TestFloat function.
#[derive(Clone, Debug)]
struct Func {
    name: String,
    /// The ruvm request operation, `fmt * 64 + op`.
    op: u16,
    args: Vec<Ty>,
    result: Ty,
    /// `imm` for the request.
    imm: i32,
    cmp: Option<Cmp>,
    /// Sensitive to the rounding mode.
    rounds: bool,
    /// Sensitive to extF80 rounding precision.
    x80_prec: bool,
    /// Needs `-exact`, and round to odd means something else to TestFloat.
    to_integer: bool,
}

fn func(name: String, op: u16, args: Vec<Ty>, result: Ty) -> Func {
    Func {
        name,
        op,
        args,
        result,
        imm: 0,
        cmp: None,
        rounds: true,
        x80_prec: false,
        to_integer: false,
    }
}

/// Every TestFloat function QEMU has an equivalent for.
fn functions() -> Vec<Func> {
    let floats = [Ty::F16, Ty::F32, Ty::F64, Ty::X80, Ty::F128];
    let mut v = Vec::new();
    for &t in &floats {
        let f = t.fmt() * 64;
        let n = t.name();
        for (s, o) in [("add", ADD), ("sub", SUB), ("mul", MUL), ("div", DIV)] {
            let mut x = func(format!("{n}_{s}"), f + o, vec![t, t], t);
            x.x80_prec = t == Ty::X80;
            v.push(x);
        }
        let mut x = func(format!("{n}_sqrt"), f + SQRT, vec![t], t);
        x.x80_prec = t == Ty::X80;
        v.push(x);
        if t != Ty::F16 {
            v.push(func(format!("{n}_rem"), f + REM, vec![t, t], t));
        }
        if t != Ty::X80 {
            v.push(func(format!("{n}_mulAdd"), f + MULADD, vec![t, t, t], t));
        }
        let mut x = func(format!("{n}_roundToInt"), f + ROUND_TO_INT, vec![t], t);
        x.to_integer = true;
        v.push(x);
        for (s, o, c) in [
            ("eq", COMPARE_QUIET, Cmp::Eq),
            ("le", COMPARE, Cmp::Le),
            ("lt", COMPARE, Cmp::Lt),
            ("eq_signaling", COMPARE, Cmp::Eq),
            ("le_quiet", COMPARE_QUIET, Cmp::Le),
            ("lt_quiet", COMPARE_QUIET, Cmp::Lt),
        ] {
            let mut x = func(format!("{n}_{s}"), f + o, vec![t, t], Ty::Bool);
            x.cmp = Some(c);
            x.rounds = false;
            v.push(x);
        }
        let ints: &[(Ty, u16)] = if t == Ty::X80 {
            &[(Ty::I32, TO_I32), (Ty::I64, TO_I64)]
        } else {
            &[(Ty::I32, TO_I32), (Ty::I64, TO_I64), (Ty::U32, TO_U32), (Ty::U64, TO_U64)]
        };
        for &(i, o) in ints {
            let mut x = func(format!("{n}_to_{}", i.name()), f + o, vec![t], i);
            x.to_integer = true;
            v.push(x);
        }
        let froms: &[(Ty, u16)] = if t == Ty::X80 {
            &[(Ty::I32, FROM_I64), (Ty::I64, FROM_I64)]
        } else {
            &[(Ty::I32, FROM_I64), (Ty::I64, FROM_I64), (Ty::U32, FROM_U64), (Ty::U64, FROM_U64)]
        };
        for &(i, o) in froms {
            v.push(func(format!("{}_to_{n}", i.name()), f + o, vec![i], t));
        }
    }
    let convs = [
        (Ty::F16, Ty::F32, TO_F32),
        (Ty::F16, Ty::F64, TO_F64),
        (Ty::F32, Ty::F16, TO_F16),
        (Ty::F32, Ty::F64, TO_F64),
        (Ty::F32, Ty::X80, TO_X80),
        (Ty::F32, Ty::F128, TO_F128),
        (Ty::F64, Ty::F16, TO_F16),
        (Ty::F64, Ty::F32, TO_F32),
        (Ty::F64, Ty::X80, TO_X80),
        (Ty::F64, Ty::F128, TO_F128),
        (Ty::X80, Ty::F32, TO_F32),
        (Ty::X80, Ty::F64, TO_F64),
        (Ty::X80, Ty::F128, TO_F128),
        (Ty::F128, Ty::F32, TO_F32),
        (Ty::F128, Ty::F64, TO_F64),
        (Ty::F128, Ty::X80, TO_X80),
    ];
    for (a, z, o) in convs {
        let mut x = func(format!("{}_to_{}", a.name(), z.name()), a.fmt() * 64 + o, vec![a], z);
        // IEEE half precision, not the Arm alternative format.
        x.imm = i32::from(a == Ty::F16 || z == Ty::F16);
        v.push(x);
    }
    v
}

/// Berkeley's flag bits from QEMU's.
fn berkeley_flags(q: u16) -> u8 {
    use ruvm_softfloat::flags::*;
    let mut b = 0;
    for (qf, bf) in [(INEXACT, 1), (UNDERFLOW, 2), (OVERFLOW, 4), (DIVBYZERO, 8), (INVALID, 16)] {
        if q & qf != 0 {
            b |= bf;
        }
    }
    b
}

fn parse(tok: &str) -> u128 {
    u128::from_str_radix(tok, 16).unwrap_or_else(|_| panic!("bad hex {tok:?}"))
}

/// Widen an integer operand for the request.
fn int_operand(t: Ty, v: u128) -> u128 {
    match t {
        Ty::I32 => v as u32 as i32 as i64 as u64 as u128,
        _ => v,
    }
}

struct Run<'a> {
    f: &'a Func,
    rm: RoundMode,
    tininess_before: bool,
    x80_prec: u8,
}

impl Run<'_> {
    fn req(&self) -> Req {
        let mut s = if self.f.args[0] == Ty::X80 || self.f.result == Ty::X80 {
            FloatStatus::x87()
        } else {
            FloatStatus::x86_sse()
        };
        s.rounding_mode = self.rm;
        s.tininess_before_rounding = self.tininess_before;
        s.exception_flags = 0;
        // Berkeley raises invalid for 0 * inf + qNaN and returns the default NaN, like Arm.
        s.float_infzeronan_rule = InfZeroNanRule::DNAN_IF_QNAN;
        s.floatx80_rounding_precision =
            [FloatX80RoundPrec::X, FloatX80RoundPrec::D, FloatX80RoundPrec::S]
                [usize::from(self.x80_prec)];
        let mut r = Req { op: self.f.op, oprm: self.rm as u8, imm: self.f.imm, ..Req::default() };
        r.set_status(&s);
        r
    }

    fn args(&self, level: u32) -> Vec<String> {
        let mut a = vec!["-level".to_string(), level.to_string()];
        a.push(
            match self.rm {
                RoundMode::NearestEven => "-rnear_even",
                RoundMode::ToZero => "-rminMag",
                RoundMode::Down => "-rmin",
                RoundMode::Up => "-rmax",
                RoundMode::TiesAway => "-rnear_maxMag",
                RoundMode::ToOdd => "-rodd",
                _ => unreachable!(),
            }
            .to_string(),
        );
        a.push(if self.tininess_before { "-tininessbefore" } else { "-tininessafter" }.into());
        a.push(["-precision80", "-precision64", "-precision32"][usize::from(self.x80_prec)].into());
        if self.f.to_integer {
            a.push("-exact".into());
        }
        a.push(self.f.name.clone());
        a
    }

    /// Run one `testfloat_gen` invocation; return (cases, failures).
    fn check(&self, tf_gen: &str, level: u32, shown: &AtomicU32) -> (u64, u64) {
        let args = self.args(level);
        let mut child = Command::new(tf_gen)
            .args(&args)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("run {tf_gen}: {e}"));
        let out = BufReader::with_capacity(1 << 16, child.stdout.take().unwrap());
        let base = self.req();
        let f = self.f;
        let (mut n, mut bad) = (0, 0);
        let mut line_buf = String::new();
        let mut lines = out;
        loop {
            line_buf.clear();
            if lines.read_line(&mut line_buf).expect("read testfloat_gen output") == 0 {
                break;
            }
            let toks: Vec<&str> = line_buf.split_whitespace().collect();
            assert_eq!(toks.len(), f.args.len() + 2, "unexpected line {line_buf:?}");
            let mut r = base;
            let ops: Vec<u128> = toks[..f.args.len()].iter().map(|t| parse(t)).collect();
            let want = parse(toks[f.args.len()]);
            let want_flags = parse(toks[f.args.len() + 1]) as u8;
            if f.args[0].is_float() {
                r.a = ops[0];
            } else {
                r.a = int_operand(f.args[0], ops[0]);
            }
            if ops.len() > 1 {
                r.b = ops[1];
            }
            if ops.len() > 2 {
                r.c = ops[2];
            }
            let res = common::eval(&r);
            let got_flags = berkeley_flags(res.flags);
            let raw = u128::from(res.lo) | u128::from(res.hi) << 64;
            let got = match f.cmp {
                Some(c) => {
                    let rel = res.lo as i64;
                    u128::from(match c {
                        Cmp::Eq => rel == 0,
                        Cmp::Le => rel == 0 || rel == -1,
                        Cmp::Lt => rel == -1,
                    })
                }
                None => raw & f.result.mask(),
            };
            let value_ok = if f.result.is_float() && f.result.is_nan(want) {
                f.result.is_nan(got)
            } else if !f.result.is_float() && f.result != Ty::Bool && want_flags & 16 != 0 {
                true
            } else {
                got == want
            };
            n += 1;
            if !value_ok || got_flags != want_flags {
                bad += 1;
                if shown.fetch_add(1, Ordering::Relaxed) < 40 {
                    eprintln!(
                        "MISMATCH {} {}: {} want {want:x} flags {want_flags:02x}, got {got:x} flags {got_flags:02x}",
                        f.name,
                        args[..args.len() - 1].join(" "),
                        toks[..f.args.len()].join(" "),
                    );
                }
            }
        }
        let status = child.wait().expect("wait for testfloat_gen");
        assert!(status.success(), "testfloat_gen {} failed", args.join(" "));
        (n, bad)
    }
}

#[test]
#[ignore = "needs Berkeley TestFloat 3e's testfloat_gen, see tests/testfloat/build.sh"]
fn berkeley_testfloat() {
    let tf_gen = std::env::var("RUVM_TESTFLOAT_GEN")
        .expect("set RUVM_TESTFLOAT_GEN to testfloat_gen, see tests/testfloat/build.sh");
    let level: u32 = std::env::var("RUVM_TESTFLOAT_LEVEL").map_or(1, |v| v.parse().unwrap());
    let filter = std::env::var("RUVM_TESTFLOAT_FILTER").unwrap_or_default();
    let modes = [
        RoundMode::NearestEven,
        RoundMode::ToZero,
        RoundMode::Down,
        RoundMode::Up,
        RoundMode::TiesAway,
        RoundMode::ToOdd,
    ];

    let funcs: Vec<Func> =
        functions().into_iter().filter(|f| f.name.contains(filter.as_str())).collect();
    let next = AtomicU32::new(0);
    let shown = AtomicU32::new(0);
    // (name, cases, mismatches) per function.
    let results = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed) as usize;
                    let Some(f) = funcs.get(i) else { break };
                    let (n, bad) = check_function(f, &modes, &tf_gen, level, &shown);
                    eprintln!("{:24} {n:10} cases {bad:8} mismatches", f.name);
                    results.lock().unwrap().push((f.name.clone(), n, bad));
                }
            });
        }
    });

    let results = results.into_inner().unwrap();
    let total: u64 = results.iter().map(|r| r.1).sum();
    let total_bad: u64 = results.iter().map(|r| r.2).sum();
    let failed: Vec<&str> = results.iter().filter(|r| r.2 != 0).map(|r| r.0.as_str()).collect();
    eprintln!("{total} TestFloat cases, {total_bad} mismatches");
    assert!(total > 0, "no function matched {filter:?}");
    assert_eq!(total_bad, 0, "ruvm-softfloat differs from TestFloat in {failed:?}");
}

/// Run every rounding mode, tininess mode and extF80 precision that affects `f`.
fn check_function(
    f: &Func,
    modes: &[RoundMode],
    tf_gen: &str,
    level: u32,
    shown: &AtomicU32,
) -> (u64, u64) {
    let x80 = f.args.contains(&Ty::X80) || f.result == Ty::X80;
    let (mut n, mut bad) = (0, 0);
    for &rm in modes {
        if (!f.rounds && rm != RoundMode::NearestEven)
            || (rm == RoundMode::ToOdd && (x80 || f.to_integer))
        {
            continue;
        }
        for tininess_before in [false, true] {
            if tininess_before && (f.cmp.is_some() || f.to_integer) {
                continue;
            }
            for x80_prec in 0..if f.x80_prec { 3 } else { 1 } {
                let run = Run { f, rm, tininess_before, x80_prec };
                let (a, b) = run.check(tf_gen, level, shown);
                n += a;
                bad += b;
            }
        }
    }
    (n, bad)
}
