// SPDX-License-Identifier: GPL-2.0-or-later

//! The output helpers of qemu-io-cmds.c: sizes, times, statistics lines and hex dumps, and
//! `bdrv_image_info_specific_dump()` of block/qapi.c for `info`.

use std::time::Duration;

use ruvm_qapi::cutils::format_g;
use ruvm_qapi::types::ImageInfoSpecific;
use ruvm_qapi::visit::{QObjectOutputVisitor, Visit};
use ruvm_qapi::{QDict, QValue};

/// `%.<prec>f` as C prints it, including `inf` and `nan`.
fn c_fixed(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan" } else { "nan" }.to_string();
    }
    format!("{v:.prec$}")
}

/// `snprintf()` into a buffer of `size` bytes.
fn truncated(mut s: String, size: usize) -> String {
    if s.len() >= size {
        s.truncate(size - 1);
    }
    s
}

/// `cvtstr()`: a byte count the way qemu-io prints sizes, like `1.500 KiB`, `2 MiB` or
/// `512 bytes`.
pub fn cvtstr(value: f64) -> String {
    const UNITS: [(u32, &str); 6] =
        [(60, " EiB"), (50, " PiB"), (40, " TiB"), (30, " GiB"), (20, " MiB"), (10, " KiB")];
    let (mut s, suffix) = match UNITS.iter().find(|(shift, _)| value >= (1u64 << shift) as f64) {
        Some(&(shift, suffix)) => {
            (truncated(c_fixed(value / (1u64 << shift) as f64, 3), 60), suffix)
        }
        None => (truncated(c_fixed(value, 6), 58), " bytes"),
    };
    match s.find(".000") {
        Some(i) => {
            s.truncate(i);
            s.push_str(suffix);
        }
        None => s.push_str(suffix),
    }
    s
}

/// `timestr()` with `VERBOSE_FIXED_TIME` when `verbose`, else `DEFAULT_TIME`.
pub fn timestr(t: Duration, verbose: bool) -> String {
    let sec = t.as_secs();
    let frac = f64::from(t.subsec_nanos()) / 1e9;
    if verbose || sec != 0 {
        let secs = (sec % 60) as f64 + frac;
        format!("{}:{:02}:{}", (sec / 3600) as u32, ((sec % 3600) / 60) as u32, zero_pad(secs))
    } else {
        format!("{} sec", zero_pad(frac))
    }
}

/// `%05.2f`.
fn zero_pad(v: f64) -> String {
    format!("{v:05.2}")
}

/// `tdiv()`: `value` per second of `t`.
fn tdiv(value: f64, t: Duration) -> f64 {
    value / t.as_secs_f64()
}

/// `print_report()`: the statistics of a request of `count` bytes at `offset` that moved
/// `total` bytes in `cnt` operations taking `t`. `machine` is `-C`.
pub fn report(
    op: &str,
    t: Duration,
    offset: u64,
    count: u64,
    total: u64,
    cnt: u32,
    machine: bool,
) -> String {
    let ts = timestr(t, machine);
    if machine {
        format!(
            "{total},{cnt},{ts},{},{}\n",
            c_fixed(tdiv(total as f64, t), 3),
            c_fixed(tdiv(f64::from(cnt), t), 3)
        )
    } else {
        format!(
            "{op} {total}/{count} bytes at offset {offset}\n{}, {cnt} ops; {ts} ({}/sec and {} \
             ops/sec)\n",
            cvtstr(total as f64),
            cvtstr(tdiv(total as f64, t)),
            c_fixed(tdiv(f64::from(cnt), t), 4)
        )
    }
}

/// `dump_buffer()`: 16 bytes a line, in hex and as letters and digits.
pub fn dump_buffer(buf: &[u8], offset: u64) -> String {
    let mut out = String::new();
    for (i, chunk) in buf.chunks(16).enumerate() {
        out.push_str(&format!("{:08x}:  ", offset + i as u64 * 16));
        for b in chunk {
            out.push_str(&format!("{b:02x} "));
        }
        out.push(' ');
        for &b in chunk {
            out.push(if b.is_ascii_alphanumeric() { char::from(b) } else { '.' });
        }
        out.push('\n');
    }
    out
}

/// `dump_qobject()`.
fn dump_qobject(out: &mut String, comp_indent: usize, obj: &QValue) {
    match obj {
        QValue::Int(i) => out.push_str(&i.to_string()),
        QValue::Uint(u) => out.push_str(&u.to_string()),
        QValue::Double(d) => out.push_str(&format_g(*d, 17)),
        QValue::Str(s) => out.push_str(s),
        QValue::Dict(d) => dump_qdict(out, comp_indent, d),
        QValue::List(l) => dump_qlist(out, comp_indent, l),
        QValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        QValue::Null => {}
    }
}

fn is_composite(v: &QValue) -> bool {
    matches!(v, QValue::Dict(_) | QValue::List(_))
}

/// `dump_qlist()`.
fn dump_qlist(out: &mut String, indentation: usize, list: &[QValue]) {
    for (i, v) in list.iter().enumerate() {
        let composite = is_composite(v);
        out.push_str(&format!(
            "{:w$}[{i}]:{}",
            "",
            if composite { '\n' } else { ' ' },
            w = indentation * 4
        ));
        dump_qobject(out, indentation + 1, v);
        if !composite {
            out.push('\n');
        }
    }
}

/// `dump_qdict()`: dashes in the keys become spaces.
fn dump_qdict(out: &mut String, indentation: usize, dict: &QDict) {
    for (key, v) in dict.iter() {
        let composite = is_composite(v);
        out.push_str(&format!(
            "{:w$}{}:{}",
            "",
            key.replace('-', " "),
            if composite { '\n' } else { ' ' },
            w = indentation * 4
        ));
        dump_qobject(out, indentation + 1, v);
        if !composite {
            out.push('\n');
        }
    }
}

/// `bdrv_image_info_specific_dump()` with the prefix `info` uses, at indentation 0.
pub fn image_info_specific_dump(info: &ImageInfoSpecific) -> String {
    let mut v = QObjectOutputVisitor::new();
    let mut value = info.clone();
    ImageInfoSpecific::visit(&mut v, None, &mut value).expect("output visits do not fail");
    let obj = v.complete();
    let data = obj.as_dict().and_then(|d| d.get("data")).cloned().unwrap_or_default();
    let empty = match &data {
        QValue::Dict(d) => d.is_empty(),
        QValue::List(l) => l.is_empty(),
        _ => false,
    };
    let mut out = String::new();
    if !empty {
        out.push_str("Format specific information:\n");
        dump_qobject(&mut out, 1, &data);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        assert_eq!(cvtstr(0.0), "0 bytes");
        assert_eq!(cvtstr(512.0), "512 bytes");
        assert_eq!(cvtstr(1536.0), "1.500 KiB");
        assert_eq!(cvtstr(2097152.0), "2 MiB");
        assert_eq!(cvtstr(1.0e3), "1000 bytes");
        assert_eq!(cvtstr(0.5), "0.500000 bytes");
        assert_eq!(cvtstr(f64::INFINITY), "inf EiB");
        assert_eq!(cvtstr((1u64 << 40) as f64 * 3.25), "3.250 TiB");
    }

    #[test]
    fn times() {
        assert_eq!(timestr(Duration::from_micros(1234), false), "00.00 sec");
        assert_eq!(timestr(Duration::from_millis(250), false), "00.25 sec");
        assert_eq!(timestr(Duration::from_millis(250), true), "0:00:00.25");
        assert_eq!(timestr(Duration::from_millis(3_723_500), false), "1:02:03.50");
    }

    #[test]
    fn reports() {
        let t = Duration::from_millis(500);
        assert_eq!(
            report("read", t, 0, 1024, 1024, 1, false),
            "read 1024/1024 bytes at offset 0\n1 KiB, 1 ops; 00.50 sec (2 KiB/sec and 2.0000 \
             ops/sec)\n"
        );
        assert_eq!(report("read", t, 0, 1024, 1024, 1, true), "1024,1,0:00:00.50,2048.000,2.000\n");
    }

    #[test]
    fn dumps() {
        let buf: Vec<u8> = (0x41..0x41 + 18).chain([0u8]).collect();
        assert_eq!(
            dump_buffer(&buf, 0x200),
            "00000200:  41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50  ABCDEFGHIJKLMNOP\n\
             00000210:  51 52 00  QR.\n"
        );
    }

    #[test]
    fn qdict_dump() {
        let mut d = QDict::new();
        d.put("a-b", 1i64);
        d.put("l", QValue::List(vec![QValue::from("x")]));
        let mut out = String::new();
        dump_qobject(&mut out, 1, &QValue::Dict(d));
        assert_eq!(out, "    a b: 1\n    l:\n        [0]: x\n");
    }
}
