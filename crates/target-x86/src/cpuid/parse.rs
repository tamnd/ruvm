// SPDX-License-Identifier: GPL-2.0-or-later

//! Parsing of the `-cpu model,+feat,-feat,key=value` feature string and of
//! property values, with QEMU's error messages.
//!
//! This covers `x86_cpu_parse_featurestr()` from `target/i386/cpu.c`,
//! `qemu_strtosz_metric()` and `qemu_strtou64()` from `util/cutils.c`, and
//! the scalar parsers of `qapi/string-input-visitor.c`.

/// Result of splitting a feature string, before anything is applied.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FeatureString {
    /// Names given as `+name`, kept verbatim.
    pub plus: Vec<String>,
    /// Names given as `-name`, kept verbatim.
    pub minus: Vec<String>,
    /// `key=value` pairs (a bare `key` means `key=on`), with `_` in the key
    /// turned into `-`. These become global properties in QEMU.
    pub globals: Vec<(String, String)>,
    /// Warnings QEMU prints while parsing.
    pub warnings: Vec<String>,
}

/// `feat2prop()`: property names use `-`, not `_`.
pub fn feat2prop(s: &str) -> String {
    s.replace('_', "-")
}

/// `x86_cpu_parse_featurestr()`.
///
/// Errors carry the text QEMU reports.
pub fn parse_featurestr(features: &str) -> Result<FeatureString, String> {
    let mut out = FeatureString::default();
    let mut ambiguous = false;
    // strtok() skips empty tokens.
    for item in features.split(',').filter(|s| !s.is_empty()) {
        if let Some(rest) = item.strip_prefix('+') {
            out.plus.push(rest.to_string());
            continue;
        }
        if let Some(rest) = item.strip_prefix('-') {
            out.minus.push(rest.to_string());
            continue;
        }
        let (key, val) = match item.split_once('=') {
            Some((k, v)) => (k, v.to_string()),
            None => (item, "on".to_string()),
        };
        let mut name = feat2prop(key);
        let mut val = val;
        if out.plus.contains(&name) {
            out.warnings.push(format!(
                "Ambiguous CPU model string. Don't mix both \"+{name}\" and \"{name}={val}\""
            ));
            ambiguous = true;
        }
        if out.minus.contains(&name) {
            out.warnings.push(format!(
                "Ambiguous CPU model string. Don't mix both \"-{name}\" and \"{name}={val}\""
            ));
            ambiguous = true;
        }
        if name == "tsc-freq" {
            match strtosz_metric(&val) {
                Some(v) if v <= i64::MAX as u64 => {
                    val = v.to_string();
                    name = "tsc-frequency".to_string();
                }
                _ => return Err(format!("bad numerical value {val}")),
            }
        }
        out.globals.push((name, val));
    }
    if ambiguous {
        out.warnings.push(
            "Compatibility of ambiguous CPU model strings won't be kept on future QEMU versions"
                .to_string(),
        );
    }
    Ok(out)
}

fn suffix_mul(c: Option<u8>, unit: u64) -> Option<u64> {
    let pow = match c?.to_ascii_uppercase() {
        b'B' => 0,
        b'K' => 1,
        b'M' => 2,
        b'G' => 3,
        b'T' => 4,
        b'P' => 5,
        b'E' => 6,
        _ => return None,
    };
    Some(unit.pow(pow))
}

/// Parses a C `strtoull()` style unsigned number at the start of `s` in
/// `base` (0 means auto-detect). Returns the value (saturated on
/// overflow), whether it overflowed, and the number of bytes consumed
/// (0 when nothing parsed).
fn c_strtoull(s: &[u8], base: u32) -> (u64, bool, usize) {
    let mut i = 0;
    while i < s.len() && s[i].is_ascii_whitespace() {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut base = base;
    let has_hex_prefix = i + 1 < s.len()
        && s[i] == b'0'
        && (s[i + 1] == b'x' || s[i + 1] == b'X')
        && s.get(i + 2).is_some_and(|c| c.is_ascii_hexdigit());
    if (base == 0 || base == 16) && has_hex_prefix {
        i += 2;
        base = 16;
    } else if base == 0 {
        base = if i < s.len() && s[i] == b'0' { 8 } else { 10 };
    }
    let start = i;
    let mut val: u64 = 0;
    let mut overflow = false;
    while i < s.len() {
        let Some(d) = char::from(s[i]).to_digit(base) else {
            break;
        };
        match val.checked_mul(u64::from(base)).and_then(|v| v.checked_add(u64::from(d))) {
            Some(v) => val = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start {
        return (0, false, 0);
    }
    if overflow {
        return (u64::MAX, true, i);
    }
    (if neg { val.wrapping_neg() } else { val }, false, i)
}

/// `qemu_strtou64(s, NULL, 0, ...)`: the whole string must be a number.
/// A leading minus sign wraps, as with `strtoull()`.
pub fn strtou64(s: &str) -> Option<u64> {
    let (v, overflow, used) = c_strtoull(s.as_bytes(), 0);
    if used == 0 || used != s.len() || overflow {
        return None;
    }
    Some(v)
}

/// `qemu_strtoi64(s, NULL, 0, ...)`.
pub fn strtoi64(s: &str) -> Option<i64> {
    let b = s.as_bytes();
    let (v, overflow, used) = c_strtoull(b, 0);
    if used == 0 || used != s.len() || overflow {
        return None;
    }
    let neg = b.iter().find(|c| !c.is_ascii_whitespace()) == Some(&b'-');
    if neg {
        // v is the two's complement of the magnitude.
        let mag = v.wrapping_neg();
        if mag > i64::MAX as u64 + 1 {
            return None;
        }
        Some((mag as i64).wrapping_neg())
    } else if v > i64::MAX as u64 {
        None
    } else {
        Some(v as i64)
    }
}

/// `qemu_strtosz_metric()`: sizes with decimal (1000-based) suffixes.
/// Returns `None` wherever QEMU returns an error.
pub fn strtosz_metric(s: &str) -> Option<u64> {
    const UNIT: u64 = 1000;
    let b = s.as_bytes();
    // parse_uint(): decimal, negative numbers rejected.
    let (int_val, overflow, int_len) = c_strtoull(b, 10);
    if overflow {
        return None;
    }
    let first = b.iter().position(|c| !c.is_ascii_whitespace()).unwrap_or(b.len());
    if int_len > 0 && b[first] == b'-' {
        return None;
    }
    let int_ok = int_len > 0;
    let mut val = int_val;
    let mut valf: u64 = 0;
    let mut end = int_len;
    if int_ok && val == 0 && matches!(b.get(end), Some(b'x' | b'X')) {
        // Looks like hex: no fraction and no suffix allowed.
        let (v, overflow, used) = c_strtoull(b, 16);
        if used == 0 || overflow {
            return None;
        }
        end = used;
        if b.get(end) == Some(&b'.') || suffix_mul(b.get(end).copied(), UNIT).is_some() {
            return None;
        }
        val = v;
    } else if b.get(end) == Some(&b'.') || (!int_ok && s.contains('.')) {
        let next_is_digit = b.get(end + 1).is_some_and(u8::is_ascii_digit);
        if int_ok && b.get(end) == Some(&b'.') && !next_is_digit {
            end += 1;
        } else {
            // strtod() on the rest, with any exponent cut off.
            let rest = &s[end..];
            let cut = rest.find(['e', 'E']).unwrap_or(rest.len());
            let rest = &rest[..cut];
            let lead = rest.len() - rest.trim_start().len();
            let body = &rest[lead..];
            let blen = body
                .char_indices()
                .take_while(|&(i, c)| c.is_ascii_digit() || c == '.' || (i == 0 && c == '+'))
                .count();
            let text = &body[..blen];
            if text.starts_with('-') || !text.bytes().any(|c| c.is_ascii_digit()) {
                return None;
            }
            let fraction: f64 = text.parse().ok()?;
            end += lead + blen;
            if fraction == 1.0 {
                val = val.checked_add(1)?;
            } else {
                valf = (fraction * 18_446_744_073_709_551_616.0) as u64;
                if valf == 0 && fraction > 0.0 {
                    valf = 1;
                }
            }
        }
    } else if !int_ok {
        return None;
    }
    let mul = match suffix_mul(b.get(end).copied(), UNIT) {
        Some(m) => {
            end += 1;
            m
        }
        None => 1,
    };
    if mul == 1 {
        if valf != 0 {
            return None;
        }
    } else {
        let whole = u128::from(val) * u128::from(mul);
        let frac = u128::from(valf) * u128::from(mul);
        // Round 0.5 upward.
        let low = frac as u64;
        let total = whole + (frac >> 64) + u128::from(low >> 63);
        if total > u128::from(u64::MAX) {
            return None;
        }
        val = total as u64;
    }
    if end != b.len() {
        return None;
    }
    Some(val)
}

/// `qapi_bool_parse()`.
pub fn parse_bool(name: &str, value: &str) -> Result<bool, String> {
    match value {
        "on" | "yes" | "true" | "y" => Ok(true),
        "off" | "no" | "false" | "n" => Ok(false),
        _ => Err(format!("Parameter '{name}' expects 'on' or 'off'")),
    }
}

/// String input visitor for `uint64`.
pub fn parse_u64(name: &str, value: &str) -> Result<u64, String> {
    strtou64(value).ok_or_else(|| format!("Parameter '{name}' expects uint64"))
}

/// String input visitor for `int64`.
pub fn parse_i64(name: &str, value: &str) -> Result<i64, String> {
    strtoi64(value).ok_or_else(|| format!("Parameter '{name}' expects int64"))
}

/// `visit_type_uint32()`.
pub fn parse_u32(name: &str, value: &str) -> Result<u32, String> {
    let v = parse_u64(name, value)?;
    u32::try_from(v).map_err(|_| format!("Parameter '{name}' expects uint32_t"))
}

/// `visit_type_uint8()`.
pub fn parse_u8(name: &str, value: &str) -> Result<u8, String> {
    let v = parse_u64(name, value)?;
    u8::try_from(v).map_err(|_| format!("Parameter '{name}' expects uint8_t"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn featurestr_split() {
        let f = parse_featurestr("+avx,-x2apic,,lahf_lm,level=0x14,sse4_1=off").unwrap();
        assert_eq!(f.plus, vec!["avx"]);
        assert_eq!(f.minus, vec!["x2apic"]);
        assert_eq!(
            f.globals,
            vec![
                ("lahf-lm".to_string(), "on".to_string()),
                ("level".to_string(), "0x14".to_string()),
                ("sse4-1".to_string(), "off".to_string()),
            ]
        );
        assert!(f.warnings.is_empty());
    }

    #[test]
    fn featurestr_ambiguous() {
        let f = parse_featurestr("-x2apic,x2apic=on").unwrap();
        assert_eq!(
            f.warnings,
            vec![
                "Ambiguous CPU model string. Don't mix both \"-x2apic\" and \"x2apic=on\""
                    .to_string(),
                "Compatibility of ambiguous CPU model strings won't be kept on future QEMU versions"
                    .to_string(),
            ]
        );
    }

    #[test]
    fn tsc_freq() {
        let f = parse_featurestr("tsc_freq=2.5G").unwrap();
        assert_eq!(f.globals, vec![("tsc-frequency".to_string(), "2500000000".to_string())]);
        assert_eq!(parse_featurestr("tsc-freq=abc").unwrap_err(), "bad numerical value abc");
    }

    #[test]
    fn strtosz() {
        assert_eq!(strtosz_metric("0"), Some(0));
        assert_eq!(strtosz_metric("12345"), Some(12345));
        assert_eq!(strtosz_metric("1k"), Some(1000));
        assert_eq!(strtosz_metric("1.5M"), Some(1_500_000));
        assert_eq!(strtosz_metric("1.M"), Some(1_000_000));
        assert_eq!(strtosz_metric(".5K"), Some(500));
        assert_eq!(strtosz_metric("0x1b"), Some(27));
        assert_eq!(strtosz_metric("1B"), Some(1));
        assert_eq!(strtosz_metric("0x20M"), None);
        assert_eq!(strtosz_metric("1.5"), None);
        assert_eq!(strtosz_metric("-1"), None);
        assert_eq!(strtosz_metric("1e3"), None);
        assert_eq!(strtosz_metric(""), None);
        assert_eq!(strtosz_metric("20E"), None);
    }

    #[test]
    fn numbers() {
        assert_eq!(strtou64("0x10"), Some(16));
        assert_eq!(strtou64("010"), Some(8));
        assert_eq!(strtou64("-1"), Some(u64::MAX));
        assert_eq!(strtou64("1x"), None);
        assert_eq!(strtoi64("-5"), Some(-5));
        assert_eq!(
            parse_u32("level", "0x100000000").unwrap_err(),
            "Parameter 'level' expects uint32_t"
        );
        assert_eq!(
            parse_bool("avx", "maybe").unwrap_err(),
            "Parameter 'avx' expects 'on' or 'off'"
        );
    }
}
