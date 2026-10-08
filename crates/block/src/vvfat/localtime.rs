// SPDX-License-Identifier: GPL-2.0-or-later

//! The part of `localtime_r()` that `fat_datetime()` needs: the broken down local time of a
//! host timestamp. The crate has no unsafe budget for calling the C library, so this reads the
//! time zone the way the C library does: `TZ` when it is set (a zone file name, possibly after
//! a `:`, or a POSIX TZ string), `/etc/localtime` otherwise, and UTC when neither works. Zone
//! files are TZif files; times after their last transition follow the POSIX TZ string in the
//! footer. Leap second zones are read without their leap seconds. On Windows the time is UTC.

/// The fields of a `struct tm` that FAT timestamps use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tm {
    pub sec: i64,
    pub min: i64,
    pub hour: i64,
    pub mday: i64,
    /// 0 to 11.
    pub mon: i64,
    /// Years since 1900.
    pub year: i64,
}

/// A POSIX TZ rule date.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuleDate {
    /// `Jn`: day 1 to 365, February 29 never counted.
    Julian1(i64),
    /// `n`: day 0 to 365, February 29 counted.
    Julian0(i64),
    /// `Mm.w.d`: day `d` (0 is Sunday) of week `w` (5 is the last) of month `m`.
    Month(i64, i64, i64),
}

/// A POSIX TZ string. Offsets are seconds east of UTC, the opposite of the string's sign.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PosixTz {
    std_off: i64,
    /// The DST offset and the rules for its start and end, with the times of day.
    dst: Option<(i64, RuleDate, i64, RuleDate, i64)>,
}

/// A time zone: TZif transitions with a POSIX TZ string for later times, or just the string.
#[derive(Debug, Clone, Default)]
pub(crate) struct Zone {
    transitions: Vec<(i64, usize)>,
    /// The UTC offsets of the local time types.
    types: Vec<i64>,
    footer: Option<PosixTz>,
}

impl Zone {
    /// The zone `localtime_r()` would use in this process.
    pub(crate) fn local() -> Zone {
        if cfg!(windows) {
            return Zone::default();
        }
        match std::env::var("TZ") {
            Err(_) => Zone::from_file("/etc/localtime").unwrap_or_default(),
            Ok(tz) => Zone::from_tz(&tz),
        }
    }

    /// The zone for a `TZ` value.
    pub(super) fn from_tz(tz: &str) -> Zone {
        if tz.is_empty() {
            return Zone::default();
        }
        let name = tz.strip_prefix(':').unwrap_or(tz);
        if name.starts_with('/') {
            if let Some(z) = Zone::from_file(name) {
                return z;
            }
        } else if !name.contains("..") {
            let dirs = std::env::var("TZDIR")
                .into_iter()
                .chain(["/usr/share/zoneinfo", "/var/db/timezone/zoneinfo"].map(str::to_string));
            for dir in dirs {
                if let Some(z) = Zone::from_file(&format!("{dir}/{name}")) {
                    return z;
                }
            }
        }
        match parse_posix(name) {
            Some(p) => Zone { footer: Some(p), ..Zone::default() },
            None => Zone::default(),
        }
    }

    /// Reads a TZif file.
    fn from_file(path: &str) -> Option<Zone> {
        parse_tzif(&std::fs::read(path).ok()?)
    }

    /// The UTC offset in seconds at time `t`.
    fn offset(&self, t: i64) -> i64 {
        match self.transitions.last() {
            Some(&(last, _)) if t >= last && self.footer.is_some() => {
                self.footer.as_ref().unwrap().offset(t)
            }
            Some(_) if t >= self.transitions[0].0 => {
                let k = self.transitions.partition_point(|&(when, _)| when <= t) - 1;
                self.types.get(self.transitions[k].1).copied().unwrap_or(0)
            }
            _ => match &self.footer {
                Some(p) if self.types.is_empty() => p.offset(t),
                _ => self.types.first().copied().unwrap_or(0),
            },
        }
    }

    /// `localtime_r()`.
    pub(crate) fn localtime(&self, t: i64) -> Tm {
        let local = t + self.offset(t);
        let days = local.div_euclid(86400);
        let secs = local.rem_euclid(86400);
        let (y, m, d) = civil_from_days(days);
        Tm {
            sec: secs % 60,
            min: secs / 60 % 60,
            hour: secs / 3600,
            mday: d,
            mon: m - 1,
            year: y - 1900,
        }
    }
}

impl PosixTz {
    fn offset(&self, t: i64) -> i64 {
        let Some((dst_off, start, start_time, end, end_time)) = self.dst else {
            return self.std_off;
        };
        let (year, _, _) = civil_from_days((t + self.std_off).div_euclid(86400));
        let at = |rule: RuleDate, time: i64, off: i64| {
            (days_from_civil(year, 1, 1) + rule_day(rule, year)) * 86400 + time - off
        };
        let begins = at(start, start_time, self.std_off);
        let ends = at(end, end_time, dst_off);
        let in_dst =
            if begins < ends { begins <= t && t < ends } else { !(ends <= t && t < begins) };
        if in_dst { dst_off } else { self.std_off }
    }
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// The day of the year, from 0, a rule falls on in `year`.
fn rule_day(rule: RuleDate, year: i64) -> i64 {
    match rule {
        RuleDate::Julian1(n) => n - 1 + i64::from(is_leap(year) && n >= 60),
        RuleDate::Julian0(n) => n,
        RuleDate::Month(m, w, d) => {
            let first = days_from_civil(year, m, 1);
            // 1970-01-01 was a Thursday.
            let wday = (first + 4).rem_euclid(7);
            let mut day = (d - wday).rem_euclid(7) + (w - 1) * 7;
            let len = days_from_civil(year + i64::from(m == 12), m % 12 + 1, 1) - first;
            while day >= len {
                day -= 7;
            }
            first + day - days_from_civil(year, 1, 1)
        }
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// The date of a day counted from 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

fn be32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn be64(b: &[u8], at: usize) -> Option<i64> {
    Some(i64::from_be_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// Parses a TZif file, the 64-bit part and the footer when the version has them.
fn parse_tzif(b: &[u8]) -> Option<Zone> {
    if b.get(..4)? != b"TZif" {
        return None;
    }
    let counts = |at: usize| -> Option<[usize; 6]> {
        let mut c = [0usize; 6];
        for (i, v) in c.iter_mut().enumerate() {
            *v = be32(b, at + 20 + 4 * i)? as usize;
        }
        Some(c)
    };
    let [isut, isstd, leap, timecnt, typecnt, charcnt] = counts(0)?;
    let v1_len = timecnt * 5 + typecnt * 6 + charcnt + leap * 8 + isstd + isut;
    let (hdr, tsize) = if b[4] >= b'2' { (44 + v1_len, 8) } else { (0, 4) };
    let [isut, isstd, leap, timecnt, typecnt, charcnt] = counts(hdr)?;
    let mut at = hdr + 44;
    let mut transitions = Vec::with_capacity(timecnt);
    for i in 0..timecnt {
        let when =
            if tsize == 8 { be64(b, at + 8 * i)? } else { i64::from(be32(b, at + 4 * i)? as i32) };
        let ty = *b.get(at + timecnt * tsize + i)? as usize;
        transitions.push((when, ty));
    }
    at += timecnt * (tsize + 1);
    let mut types = Vec::with_capacity(typecnt);
    for i in 0..typecnt {
        types.push(i64::from(be32(b, at + 6 * i)? as i32));
    }
    at += typecnt * 6 + charcnt + leap * (tsize + 4) + isstd + isut;
    let footer = if tsize == 8 {
        b.get(at..)
            .and_then(|rest| rest.strip_prefix(b"\n"))
            .and_then(|rest| rest.split(|&c| c == b'\n').next())
            .and_then(|s| std::str::from_utf8(s).ok())
            .filter(|s| !s.is_empty())
            .and_then(parse_posix)
    } else {
        None
    };
    Some(Zone { transitions, types, footer })
}

/// A cursor over a POSIX TZ string.
struct Cursor<'a> {
    s: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.at).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn name(&mut self) -> Option<()> {
        if self.eat(b'<') {
            while self.peek()? != b'>' {
                self.at += 1;
            }
            self.at += 1;
            return Some(());
        }
        let start = self.at;
        while self.peek().is_some_and(|c| c.is_ascii_alphabetic()) {
            self.at += 1;
        }
        (self.at - start >= 3).then_some(())
    }

    fn num(&mut self) -> Option<i64> {
        let start = self.at;
        let mut v = 0i64;
        while let Some(c) = self.peek().filter(u8::is_ascii_digit) {
            v = v * 10 + i64::from(c - b'0');
            self.at += 1;
        }
        (self.at > start).then_some(v)
    }

    /// `[+-]hh[:mm[:ss]]` in seconds.
    fn time(&mut self) -> Option<i64> {
        let neg = self.eat(b'-');
        if !neg {
            self.eat(b'+');
        }
        let mut v = self.num()? * 3600;
        if self.eat(b':') {
            v += self.num()? * 60;
            if self.eat(b':') {
                v += self.num()?;
            }
        }
        Some(if neg { -v } else { v })
    }

    fn rule(&mut self) -> Option<(RuleDate, i64)> {
        let date = if self.eat(b'J') {
            RuleDate::Julian1(self.num()?)
        } else if self.eat(b'M') {
            let m = self.num()?;
            self.eat(b'.').then_some(())?;
            let w = self.num()?;
            self.eat(b'.').then_some(())?;
            RuleDate::Month(m, w, self.num()?)
        } else {
            RuleDate::Julian0(self.num()?)
        };
        let time = if self.eat(b'/') { self.time()? } else { 7200 };
        Some((date, time))
    }
}

/// Parses a POSIX TZ string such as `EST5EDT,M3.2.0,M11.1.0` or `<+07>-7`.
fn parse_posix(s: &str) -> Option<PosixTz> {
    let mut c = Cursor { s: s.as_bytes(), at: 0 };
    c.name()?;
    let std_off = -c.time()?;
    if c.peek().is_none() {
        return Some(PosixTz { std_off, dst: None });
    }
    c.name()?;
    let dst_off = match c.peek() {
        Some(b',') | None => std_off + 3600,
        _ => -c.time()?,
    };
    let (start, end) = if c.eat(b',') {
        let start = c.rule()?;
        c.eat(b',').then_some(())?;
        (start, c.rule()?)
    } else {
        ((RuleDate::Month(3, 2, 0), 7200), (RuleDate::Month(11, 1, 0), 7200))
    };
    if c.peek().is_some() {
        return None;
    }
    Some(PosixTz { std_off, dst: Some((dst_off, start.0, start.1, end.0, end.1)) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_round_trip() {
        for days in [-800_000, -1, 0, 1, 11_016, 19_000, 2_932_896] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
    }

    #[test]
    fn posix_strings() {
        let z = Zone::from_tz("EST5EDT,M3.2.0,M11.1.0");
        // 2024-07-01 12:00 UTC is EDT, 2024-01-01 12:00 UTC is EST.
        assert_eq!(z.offset(1_719_835_200), -4 * 3600);
        assert_eq!(z.offset(1_704_110_400), -5 * 3600);
        // DST started 2024-03-10 07:00 UTC.
        assert_eq!(z.offset(1_710_054_000 - 1), -5 * 3600);
        assert_eq!(z.offset(1_710_054_000), -4 * 3600);
        let z = Zone::from_tz("<+07>-7");
        assert_eq!(z.offset(0), 7 * 3600);
        let z = Zone::from_tz("");
        assert_eq!(
            z.localtime(86_400 + 3661),
            Tm { sec: 1, min: 1, hour: 1, mday: 2, mon: 0, year: 70 }
        );
        // Southern hemisphere rules wrap around the new year.
        let z = Zone::from_tz("AEST-10AEDT,M10.1.0,M4.1.0/3");
        assert_eq!(z.offset(1_704_110_400), 11 * 3600);
        assert_eq!(z.offset(1_719_835_200), 10 * 3600);
    }

    /// Checks the zone files against `date` where it is there.
    #[cfg(unix)]
    #[test]
    fn matches_date() {
        let times =
            [0i64, 1_000_000_000, 1_710_054_000, 1_719_835_200, 1_900_000_000, 2_200_000_000];
        for tz in
            ["America/New_York", "Europe/Berlin", "Australia/Sydney", "Asia/Ho_Chi_Minh", "UTC"]
        {
            let z = Zone::from_tz(tz);
            for t in times {
                let bsd = std::process::Command::new("date")
                    .env("TZ", tz)
                    .args(["-r", &t.to_string(), "+%Y %m %d %H %M %S"])
                    .output();
                let out = match bsd {
                    Ok(o) if o.status.success() => o,
                    _ => match std::process::Command::new("date")
                        .env("TZ", tz)
                        .args(["-d", &format!("@{t}"), "+%Y %m %d %H %M %S"])
                        .output()
                    {
                        Ok(o) if o.status.success() => o,
                        _ => return,
                    },
                };
                let v: Vec<i64> = String::from_utf8_lossy(&out.stdout)
                    .split_whitespace()
                    .map(|s| s.parse().unwrap())
                    .collect();
                let tm = z.localtime(t);
                assert_eq!(
                    [tm.year + 1900, tm.mon + 1, tm.mday, tm.hour, tm.min, tm.sec],
                    v[..],
                    "{tz} at {t}"
                );
            }
        }
    }
}
