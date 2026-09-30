// SPDX-License-Identifier: GPL-2.0-or-later

//! The number parsers from QEMU's util/cutils.c, which decide what the command line and the string
//! visitors accept.
//!
//! They are wrappers around C's `strtoll`, `strtoull` and `strtod`, so the C rules come along:
//! leading white space, an optional sign, base 0 meaning `0x` for hex and a leading `0` for octal,
//! and `strtoull` quietly negating a value that starts with `-`. Each function returns the value and
//! the number of bytes consumed, or an [`Errno`].

/// The two failures the C helpers report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Errno {
    /// `-EINVAL`: nothing was converted, or text was left over when the whole string had to parse.
    Inval,
    /// `-ERANGE`: the value does not fit.
    Range,
}

pub type StrtoResult<T> = Result<(T, usize), (Errno, T)>;

fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

fn digit_value(b: u8) -> Option<u32> {
    match b {
        b'0'..=b'9' => Some(u32::from(b - b'0')),
        b'a'..=b'z' => Some(u32::from(b - b'a') + 10),
        b'A'..=b'Z' => Some(u32::from(b - b'A') + 10),
        _ => None,
    }
}

/// The shared scanner of `strtoll` and `strtoull`: the magnitude, whether it overflowed a `u64`,
/// whether a minus sign was seen, and where parsing stopped. Consumed is 0 when nothing converted.
fn scan_integer(s: &[u8], base: u32) -> (u64, bool, bool, usize) {
    let mut i = 0;
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut base = base;
    let hex_prefix = i + 1 < s.len()
        && s[i] == b'0'
        && (s[i + 1] == b'x' || s[i + 1] == b'X')
        && s.get(i + 2).and_then(|&b| digit_value(b)).is_some_and(|d| d < 16);
    if (base == 0 || base == 16) && hex_prefix {
        i += 2;
        base = 16;
    } else if base == 0 {
        base = if s.get(i) == Some(&b'0') { 8 } else { 10 };
    }
    let start = i;
    let mut val: u64 = 0;
    let mut overflow = false;
    while let Some(d) = s.get(i).and_then(|&b| digit_value(b)) {
        if d >= base {
            break;
        }
        match val.checked_mul(u64::from(base)).and_then(|v| v.checked_add(u64::from(d))) {
            Some(v) => val = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, false, false, 0);
    }
    (val, overflow, neg, i)
}

fn finish<T>(
    value: T,
    used: usize,
    len: usize,
    need_all: bool,
    err: Option<Errno>,
) -> StrtoResult<T> {
    if let Some(e) = err {
        return Err((e, value));
    }
    if used == 0 || (need_all && used != len) {
        return Err((Errno::Inval, value));
    }
    Ok((value, used))
}

/// `qemu_strtoi64()`. With `need_all` it behaves as if `endptr` were NULL.
pub fn strtoi64(s: &str, base: u32, need_all: bool) -> StrtoResult<i64> {
    let (mag, overflow, neg, used) = scan_integer(s.as_bytes(), base);
    let (value, err) = if neg {
        if overflow || mag > 1u64 << 63 {
            (i64::MIN, Some(Errno::Range))
        } else {
            ((mag as i64).wrapping_neg(), None)
        }
    } else if overflow || mag > i64::MAX as u64 {
        (i64::MAX, Some(Errno::Range))
    } else {
        (mag as i64, None)
    };
    finish(value, used, s.len(), need_all, err)
}

/// `qemu_strtou64()`. A leading minus negates in two's complement, as `strtoull` does.
pub fn strtou64(s: &str, base: u32, need_all: bool) -> StrtoResult<u64> {
    let (mag, overflow, neg, used) = scan_integer(s.as_bytes(), base);
    let (value, err) = if overflow {
        (u64::MAX, Some(Errno::Range))
    } else if neg {
        (mag.wrapping_neg(), None)
    } else {
        (mag, None)
    };
    finish(value, used, s.len(), need_all, err)
}

/// `parse_uint()`: like [`strtou64`] but a negative number is `Range` instead of wrapping.
pub fn parse_uint(s: &str, base: u32, need_all: bool) -> StrtoResult<u64> {
    let (mag, overflow, neg, used) = scan_integer(s.as_bytes(), base);
    if used == 0 {
        return Err((Errno::Inval, 0));
    }
    if overflow {
        return Err((Errno::Range, u64::MAX));
    }
    if neg {
        return Err((Errno::Range, 0));
    }
    if need_all && used != s.len() {
        return Err((Errno::Inval, 0));
    }
    Ok((mag, used))
}

/// C `strtod()`: the longest prefix that is a decimal or hex float, `inf`, `infinity` or `nan`.
fn scan_double(s: &[u8]) -> (f64, usize, Option<Errno>) {
    let mut i = 0;
    while i < s.len() && is_c_space(s[i]) {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let rest = &s[i..];
    let lower: Vec<u8> = rest.iter().take(8).map(u8::to_ascii_lowercase).collect();
    let signed = |v: f64| if neg { -v } else { v };
    if lower.starts_with(b"infinity") {
        return (signed(f64::INFINITY), i + 8, None);
    }
    if lower.starts_with(b"inf") {
        return (signed(f64::INFINITY), i + 3, None);
    }
    if lower.starts_with(b"nan") {
        let mut end = i + 3;
        if s.get(end) == Some(&b'(') {
            let mut j = end + 1;
            while j < s.len() && (s[j].is_ascii_alphanumeric() || s[j] == b'_') {
                j += 1;
            }
            if s.get(j) == Some(&b')') {
                end = j + 1;
            }
        }
        return (f64::NAN, end, None);
    }
    let hex = rest.len() > 2
        && rest[0] == b'0'
        && (rest[1] == b'x' || rest[1] == b'X')
        && (rest[2].is_ascii_hexdigit()
            || (rest[2] == b'.' && rest.get(3).is_some_and(u8::is_ascii_hexdigit)));
    if hex {
        return scan_hex_double(s, i + 2, neg);
    }

    let start = i;
    let mut j = i;
    let mut digits = 0;
    while j < s.len() && s[j].is_ascii_digit() {
        j += 1;
        digits += 1;
    }
    if j < s.len() && s[j] == b'.' {
        j += 1;
        while j < s.len() && s[j].is_ascii_digit() {
            j += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return (0.0, 0, None);
    }
    if j < s.len() && (s[j] == b'e' || s[j] == b'E') {
        let mut k = j + 1;
        if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
            k += 1;
        }
        if k < s.len() && s[k].is_ascii_digit() {
            while k < s.len() && s[k].is_ascii_digit() {
                k += 1;
            }
            j = k;
        }
    }
    let text = std::str::from_utf8(&s[start..j]).expect("ASCII digits");
    let text = if text.starts_with('.') { format!("0{text}") } else { text.to_string() };
    let text = if text.ends_with('.') { format!("{text}0") } else { text };
    let text = text.replace(".e", ".0e").replace(".E", ".0E");
    let v: f64 = text.parse().unwrap_or(0.0);
    let v = signed(v);
    let nonzero = s[start..j]
        .iter()
        .take_while(|b| **b != b'e' && **b != b'E')
        .any(|b| (b'1'..=b'9').contains(b));
    let err = if v.is_infinite() || (nonzero && (v == 0.0 || v.is_subnormal())) {
        Some(Errno::Range)
    } else {
        None
    };
    (v, j, err)
}

fn scan_hex_double(s: &[u8], mut j: usize, neg: bool) -> (f64, usize, Option<Errno>) {
    let mut mant: u64 = 0;
    let mut exp: i64 = 0;
    let mut seen_nonzero_lost = false;
    let mut seen_point = false;
    while j < s.len() {
        let b = s[j];
        if b == b'.' && !seen_point {
            seen_point = true;
        } else if let Some(d) = (b as char).to_digit(16) {
            if mant >> 60 == 0 {
                mant = (mant << 4) | u64::from(d);
                if seen_point {
                    exp -= 4;
                }
            } else {
                seen_nonzero_lost |= d != 0;
                if !seen_point {
                    exp += 4;
                }
            }
        } else {
            break;
        }
        j += 1;
    }
    if j < s.len() && (s[j] == b'p' || s[j] == b'P') {
        let mut k = j + 1;
        let mut eneg = false;
        if k < s.len() && (s[k] == b'+' || s[k] == b'-') {
            eneg = s[k] == b'-';
            k += 1;
        }
        if k < s.len() && s[k].is_ascii_digit() {
            let mut e: i64 = 0;
            while k < s.len() && s[k].is_ascii_digit() {
                e = (e * 10 + i64::from(s[k] - b'0')).min(1 << 20);
                k += 1;
            }
            exp += if eneg { -e } else { e };
            j = k;
        }
    }
    let m = mant | u64::from(seen_nonzero_lost);
    let mut v = m as f64 * 2f64.powi(exp.clamp(-2000, 2000) as i32);
    if neg {
        v = -v;
    }
    let err = if v.is_infinite() || (m != 0 && (v == 0.0 || v.is_subnormal())) {
        Some(Errno::Range)
    } else {
        None
    };
    (v, j, err)
}

/// `qemu_strtod()`.
pub fn strtod(s: &str, need_all: bool) -> StrtoResult<f64> {
    let (v, used, err) = scan_double(s.as_bytes());
    if used == 0 {
        return Err((Errno::Inval, 0.0));
    }
    if let Some(e) = err {
        return Err((e, v));
    }
    if need_all && used != s.len() {
        return Err((Errno::Inval, v));
    }
    Ok((v, used))
}

/// `qemu_strtod_finite()`: [`strtod`] that also refuses infinities and NaN.
pub fn strtod_finite(s: &str, need_all: bool) -> StrtoResult<f64> {
    let (v, used, err) = scan_double(s.as_bytes());
    if used == 0 || !v.is_finite() {
        return Err((Errno::Inval, 0.0));
    }
    if let Some(e) = err {
        return Err((e, v));
    }
    if need_all && used != s.len() {
        return Err((Errno::Inval, v));
    }
    Ok((v, used))
}

fn suffix_mul(c: u8, unit: u64) -> Option<u64> {
    let n = match c.to_ascii_uppercase() {
        b'B' => 0,
        b'K' => 1,
        b'M' => 2,
        b'G' => 3,
        b'T' => 4,
        b'P' => 5,
        b'E' => 6,
        _ => return None,
    };
    Some(unit.pow(n))
}

/// The body of `do_strtosz()`: the status, the value and where parsing stopped.
fn do_strtosz(s: &str, default_suffix: u8, unit: u64) -> (Option<Errno>, u64, usize) {
    let b = s.as_bytes();
    let at = |i: usize| b.get(i).copied().unwrap_or(0);
    let mut valf: u64 = 0;
    let (mut retval, mut val, mut end) = {
        let (mag, overflow, neg, used) = scan_integer(b, 10);
        if used == 0 {
            (Some(Errno::Inval), 0, 0)
        } else if overflow {
            (Some(Errno::Range), u64::MAX, used)
        } else if neg {
            (Some(Errno::Range), 0, used)
        } else {
            (None, mag, used)
        }
    };
    if retval == Some(Errno::Range) {
        return (retval, 0, end);
    }
    if retval.is_none() && val == 0 && (at(end) == b'x' || at(end) == b'X') {
        match strtou64(s, 16, false) {
            Ok((v, used)) => {
                val = v;
                end = used;
            }
            Err((e, _)) => return (Some(e), 0, end),
        }
        if at(end) == b'.' || suffix_mul(at(end), unit).is_some() {
            return (Some(Errno::Inval), 0, 0);
        }
    } else if at(end) == b'.' || (end == 0 && s.contains('.')) {
        let mut fraction = 0.0;
        let mut underflow = false;
        if retval.is_none() && at(end) == b'.' && !at(end + 1).is_ascii_digit() {
            end += 1;
        } else {
            let tail = &s[end..];
            let copy = &tail[..tail.find(['e', 'E']).unwrap_or(tail.len())];
            let (f, used, err) = scan_double(copy.as_bytes());
            if used == 0 || !f.is_finite() {
                retval = Some(Errno::Inval);
            } else {
                fraction = f;
                end += used;
                retval = None;
                underflow = err == Some(Errno::Range);
            }
            if fraction.is_sign_negative() {
                return (Some(Errno::Range), 0, end);
            }
        }
        if retval.is_none() {
            if fraction == 1.0 {
                if val == u64::MAX {
                    return (Some(Errno::Range), 0, end);
                }
                val += 1;
            } else if underflow {
                valf = 1;
            } else {
                valf = (fraction * 18446744073709551616.0) as u64;
                if valf == 0 && fraction > 0.0 {
                    valf = 1;
                }
            }
        }
    }
    if retval.is_some() {
        return (retval, 0, end);
    }
    let mul = match suffix_mul(at(end), unit) {
        Some(m) => {
            end += 1;
            m
        }
        None => suffix_mul(default_suffix, unit).expect("valid default suffix"),
    };
    if mul == 1 {
        if valf != 0 {
            return (Some(Errno::Inval), 0, 0);
        }
    } else {
        // 64.64 fixed point times 64.0 gives 128.64, rounded half up at the binary point.
        let part = u128::from(valf) * u128::from(mul);
        let total = u128::from(val) * u128::from(mul) + (part >> 64) + ((part >> 63) & 1);
        if total > u128::from(u64::MAX) {
            return (Some(Errno::Range), 0, end);
        }
        val = total as u64;
    }
    (None, val, end)
}

/// A size parse that may stop early, as with a non-NULL `endptr`: the result and the offset
/// parsing stopped at, which is 0 after `Inval`.
pub fn strtosz_prefix_with(s: &str, default_suffix: u8, unit: u64) -> (Result<u64, Errno>, usize) {
    match do_strtosz(s, default_suffix, unit) {
        (None, v, end) => (Ok(v), end),
        (Some(Errno::Inval), _, _) => (Err(Errno::Inval), 0),
        (Some(e), _, end) => (Err(e), end),
    }
}

/// A size parse of the whole string, as with a NULL `endptr`. Trailing text is `Inval` even when
/// the number itself was out of range.
pub fn strtosz_with(s: &str, default_suffix: u8, unit: u64) -> Result<u64, Errno> {
    let (err, v, end) = do_strtosz(s, default_suffix, unit);
    if end != s.len() {
        return Err(Errno::Inval);
    }
    match err {
        None => Ok(v),
        Some(e) => Err(e),
    }
}

/// `qemu_strtosz()`, binary units with bytes as the default.
pub fn strtosz(s: &str) -> Result<u64, Errno> {
    strtosz_with(s, b'B', 1024)
}

/// `qemu_strtosz_MiB()`, where a bare number means mebibytes.
pub fn strtosz_mib(s: &str) -> Result<u64, Errno> {
    strtosz_with(s, b'M', 1024)
}

/// `qemu_strtosz_metric()`, powers of 1000.
pub fn strtosz_metric(s: &str) -> Result<u64, Errno> {
    strtosz_with(s, b'B', 1000)
}

/// `qapi_bool_parse()` without the error: the eight spellings QEMU takes for a boolean.
pub fn bool_parse(value: &str) -> Option<bool> {
    match value {
        "on" | "yes" | "true" | "y" => Some(true),
        "off" | "no" | "false" | "n" => Some(false),
        _ => None,
    }
}

/// `id_wellformed()`: a letter, then letters, digits, `-`, `.` and `_`.
pub fn id_wellformed(id: &str) -> bool {
    let mut chars = id.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'))
}

/// C's `printf("%.*g", precision, v)`.
pub fn format_g(v: f64, precision: usize) -> String {
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let p = precision.max(1) as i32;
    let sci = format!("{:.*e}", (p - 1) as usize, v);
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    if (-4..p).contains(&exp) {
        let fixed = format!("{:.*}", (p - 1 - exp) as usize, v);
        strip_zeros(&fixed).to_string()
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip_zeros(mantissa), sign, exp.abs())
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

/// `iec_binary_prefix()`.
fn iec_binary_prefix(exp2: u32) -> &'static str {
    ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"][(exp2 / 10) as usize]
}

/// The exponent `frexp()` returns: `x` is a fraction in [0.5, 1) times two to this power.
fn frexp_exp(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale up into the normal range first.
        return frexp_exp(x * 2f64.powi(64)) - 64;
    }
    biased - 1022
}

/// `size_to_str()`: "1 KiB", "1.5 GiB", with three significant digits.
pub fn size_to_str(val: u64) -> String {
    let i = frexp_exp(val as f64 / (1000.0 / 1024.0));
    // C division truncates toward zero, so (0 - 1) / 10 is 0.
    let i = ((i - 1) / 10 * 10) as u32;
    let div = 1u64 << i;
    format!("{} {}B", format_g(val as f64 / div as f64, 3), iec_binary_prefix(i))
}

/// `freq_to_str()`: "100 MHz", with three significant digits.
pub fn freq_to_str(freq_hz: u64) -> String {
    let mut freq = freq_hz as f64;
    let mut exp10 = 0;
    while freq >= 1000.0 {
        freq /= 1000.0;
        exp10 += 3;
    }
    let prefix = ["", "K", "M", "G", "T", "P", "E"][exp10 / 3];
    format!("{} {prefix}Hz", format_g(freq, 3))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integers_follow_c() {
        assert_eq!(strtoi64("0x10", 0, true), Ok((16, 4)));
        assert_eq!(strtoi64("010", 0, true), Ok((8, 3)));
        assert_eq!(strtoi64("  -5", 0, true), Ok((-5, 4)));
        assert_eq!(strtoi64("0x", 0, false), Ok((0, 1)));
        assert_eq!(strtoi64("0x", 0, true), Err((Errno::Inval, 0)));
        assert_eq!(strtoi64("", 0, true), Err((Errno::Inval, 0)));
        assert_eq!(strtoi64("9223372036854775808", 0, true), Err((Errno::Range, i64::MAX)));
        assert_eq!(strtoi64("-9223372036854775808", 0, true), Ok((i64::MIN, 20)));
        assert_eq!(strtou64("-1", 0, true), Ok((u64::MAX, 2)));
        assert_eq!(strtou64("18446744073709551616", 0, true), Err((Errno::Range, u64::MAX)));
        assert_eq!(strtou64("12abc", 0, true), Err((Errno::Inval, 12)));
        assert_eq!(parse_uint("-1", 0, true), Err((Errno::Range, 0)));
    }

    #[test]
    fn doubles_follow_c() {
        assert_eq!(strtod("1.5", true), Ok((1.5, 3)));
        assert_eq!(strtod(".5", true), Ok((0.5, 2)));
        assert_eq!(strtod("5.", true), Ok((5.0, 2)));
        assert_eq!(strtod("1e3", true), Ok((1000.0, 3)));
        assert_eq!(strtod("1e", false), Ok((1.0, 1)));
        assert_eq!(strtod("0x1p3", true), Ok((8.0, 5)));
        assert_eq!(strtod("-inf", true).map(|(v, _)| v), Ok(f64::NEG_INFINITY));
        assert_eq!(strtod_finite("inf", true), Err((Errno::Inval, 0.0)));
        assert_eq!(strtod("1e999", true).unwrap_err().0, Errno::Range);
        assert_eq!(strtod("abc", true), Err((Errno::Inval, 0.0)));
    }

    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const EIB: u64 = MIB * MIB * MIB;

    /// `do_strtosz_full()` from tests/unit/test-cutils.c.
    fn full(s: &str, unit: (u8, u64), ptr: (Result<u64, Errno>, usize), null: Result<u64, Errno>) {
        assert_eq!(strtosz_prefix_with(s, unit.0, unit.1), ptr, "{s:?} with endptr");
        assert_eq!(strtosz_with(s, unit.0, unit.1), null, "{s:?} without endptr");
    }

    fn sz(s: &str, r: Result<u64, Errno>, off: usize) {
        full(s, (b'B', 1024), (r, off), r);
    }

    #[test]
    fn sizes_follow_qemu_unit_tests() {
        const B: (u8, u64) = (b'B', 1024);
        const M: (u8, u64) = (b'M', 1024);
        const METRIC: (u8, u64) = (b'B', 1000);
        let long_zeros = "0".repeat(350);
        let inval = Err(Errno::Inval);
        let range = Err(Errno::Range);

        sz("0", Ok(0), 1);
        sz("08", Ok(8), 2);
        sz(" +12345", Ok(12345), 7);
        sz("9007199254740993", Ok(0x20000000000001), 16);
        sz("18446744073709550591", Ok(0xfffffffffffffbff), 20);
        sz("18446744073709551615", Ok(u64::MAX), 20);
        sz("0x0", Ok(0), 3);
        sz("0xab", Ok(171), 4);
        sz(" +0xae", Ok(174), 6);
        full("1", M, (Ok(MIB), 1), Ok(MIB));
        full("1B", M, (Ok(1), 2), Ok(1));
        full("1K", METRIC, (Ok(1000), 2), Ok(1000));
        sz("1K", Ok(KIB), 2);
        sz("1E", Ok(EIB), 2);
        sz("0.5E", Ok(EIB / 2), 4);
        full("0.5", M, (Ok(MIB / 2), 3), Ok(MIB / 2));
        sz("1.0B", Ok(1), 4);
        sz("1.k", Ok(1024), 3);
        sz(" .5k", Ok(512), 4);
        sz("12.345M", Ok((12.345 * MIB as f64 + 0.5) as u64), 7);
        sz("1.9999k", Ok(2048), 7);
        sz(&format!("1.{long_zeros}1k"), Ok(1024), 354);

        for s in ["", " \t ", ".", " .", " .k", "inf", "NaN", "k", " M", "1.1B", "1.1", "1.00001B"]
        {
            sz(s, inval, 0);
        }
        sz(&format!("1.{long_zeros}1B"), inval, 0);
        for s in ["0x1.8k", "0x1.k", "0x18M", "0x1p1", "1.1.k", "1.1."] {
            sz(s, inval, 0);
        }

        full("1k ", B, (Ok(1024), 2), inval);
        full("123xxx", B, (Ok(123), 3), inval);
        full("123xxx", M, (Ok(123 * MIB), 3), inval);
        full("1.5.k", M, (Ok(MIB * 3 / 2), 3), inval);
        full("1kiB", B, (Ok(1024), 2), inval);
        full("0x", B, (Ok(0), 1), inval);
        full("00x1", B, (Ok(0), 2), inval);
        full("0b1000", B, (Ok(0), 2), inval);
        full("0.NaN", B, (Ok(0), 2), inval);
        full("123-45", B, (Ok(123), 3), inval);
        full(" 123 - 45", B, (Ok(123), 4), inval);
        full("1.5e1k", B, (Ok(EIB / 2 * 3), 4), inval);
        full("1.5E+0k", B, (Ok(EIB / 2 * 3), 4), inval);
        full("1.5E999", B, (Ok(EIB / 2 * 3), 4), inval);

        sz(" -0", range, 3);
        sz("-1", range, 2);
        full("-2M", B, (range, 2), inval);
        sz(" -.0", range, 4);
        full("-.1k", B, (range, 3), inval);
        full(&format!(" -.{long_zeros}1M"), B, (range, 354), inval);
        sz("18446744073709551616", range, 20);
        sz("20E", range, 3);
        sz("15.9999999999999999999999999999999999999999999999999999E", range, 56);
        full("100000Pjunk", B, (range, 7), inval);

        full("12345k", METRIC, (Ok(12345000), 6), Ok(12345000));
        full("12.345M", METRIC, (Ok(12345000), 7), Ok(12345000));
        full(
            "18.446744073709550591E",
            METRIC,
            (Ok(0xfffffffffffffc0c), 22),
            Ok(0xfffffffffffffc0c),
        );
    }

    #[test]
    fn ids() {
        assert!(id_wellformed("net0"));
        assert!(id_wellformed("a-b.c_d"));
        assert!(!id_wellformed("0net"));
        assert!(!id_wellformed(""));
        assert!(!id_wellformed("a b"));
    }

    #[test]
    fn human_sizes() {
        assert_eq!(size_to_str(0), "0 B");
        assert_eq!(size_to_str(1), "1 B");
        assert_eq!(size_to_str(1000), "0.977 KiB");
        assert_eq!(size_to_str(1024), "1 KiB");
        assert_eq!(size_to_str(1536), "1.5 KiB");
        assert_eq!(size_to_str(128 << 20), "128 MiB");
        assert_eq!(size_to_str(u64::MAX), "16 EiB");
        assert_eq!(freq_to_str(100_000_000), "100 MHz");
        assert_eq!(format_g(0.0001, 3), "0.0001");
        assert_eq!(format_g(123456.0, 3), "1.23e+05");
    }
}
