// SPDX-License-Identifier: GPL-2.0-or-later

//! The pieces of the remote protocol that do not depend on the target: hex and binary
//! encoding, the number parsing of `qemu_strtoul()` and the parameter schemas of
//! `cmd_parse_params()` in `gdbstub/gdbstub.c`.

/// `MAX_PACKET_LENGTH`: 2 * 256 * 256 + 32, room for a 64 KiB register or memory block.
pub const MAX_PACKET_LENGTH: usize = 131_104;

/// `tohex()`.
pub fn tohex(v: u8) -> u8 {
    if v < 10 { b'0' + v } else { b'a' + v - 10 }
}

/// `fromhex()`: anything that is not a hex digit is 0.
pub fn fromhex(v: u8) -> u8 {
    match v {
        b'0'..=b'9' => v - b'0',
        b'A'..=b'F' => v - b'A' + 10,
        b'a'..=b'f' => v - b'a' + 10,
        _ => 0,
    }
}

/// `gdb_memtohex()`.
pub fn memtohex(out: &mut Vec<u8>, mem: &[u8]) {
    for &c in mem {
        out.push(tohex(c >> 4));
        out.push(tohex(c & 0xf));
    }
}

/// `gdb_hextomem()`: `len` bytes from the first `2 * len` characters of `buf`.
pub fn hextomem(out: &mut Vec<u8>, buf: &[u8], len: usize) {
    for p in buf.chunks_exact(2).take(len) {
        out.push(fromhex(p[0]) << 4 | fromhex(p[1]));
    }
}

/// `gdb_memtox()`: the binary encoding of `x` packets and `qXfer` replies.
pub fn memtox(out: &mut Vec<u8>, mem: &[u8]) {
    for &c in mem {
        match c {
            b'#' | b'$' | b'*' | b'}' => {
                out.push(b'}');
                out.push(c ^ 0x20);
            }
            _ => out.push(c),
        }
    }
}

/// The error of `qemu_strtoul()`: no digits, or a number too large.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumError {
    /// `-EINVAL`.
    Invalid,
    /// `-ERANGE`.
    Range,
}

/// `qemu_strtoul()` with an end pointer, which is `strtoul()`: leading white space, a sign,
/// `0x` in base 16, then digits. Returns the value, ULONG_MAX when out of range, and how much
/// of `s` it took, 0 when there were no digits.
pub fn strtoul(s: &[u8], base: u32) -> (Result<u64, NumError>, usize) {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let digit = |c: u8| (c as char).to_digit(base);
    if base == 16
        && s.get(i) == Some(&b'0')
        && matches!(s.get(i + 1), Some(b'x' | b'X'))
        && s.get(i + 2).copied().and_then(digit).is_some()
    {
        i += 2;
    }
    let start = i;
    let mut v: u64 = 0;
    let mut over = false;
    while let Some(d) = s.get(i).copied().and_then(digit) {
        match v.checked_mul(u64::from(base)).and_then(|x| x.checked_add(u64::from(d))) {
            Some(x) => v = x,
            None => over = true,
        }
        i += 1;
    }
    if i == start {
        return (Err(NumError::Invalid), 0);
    }
    if over {
        return (Err(NumError::Range), i);
    }
    (Ok(if neg { v.wrapping_neg() } else { v }), i)
}

/// What `read_thread_id()` found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadId {
    /// `GDB_READ_THREAD_ERR`.
    Err,
    /// `GDB_ALL_PROCESSES`: `p-1`.
    AllProcesses,
    /// `GDB_ALL_THREADS`: `-1` in a process.
    AllThreads {
        /// The process, 0 for any.
        pid: u32,
    },
    /// `GDB_ONE_THREAD`.
    One {
        /// The process, 0 for any.
        pid: u32,
        /// The thread, 0 for any.
        tid: u32,
    },
}

/// `read_thread_id()`: `[p<pid>.]<tid>`, both in hex, -1 meaning all. Returns how much of
/// `buf` it took, nothing when it failed.
pub fn read_thread_id(buf: &[u8]) -> (ThreadId, usize) {
    let mut i = 0;
    let mut p = 0u64;
    if buf.first() == Some(&b'p') {
        i = 1;
        let (r, n) = strtoul(&buf[i..], 16);
        match r {
            Ok(v) => p = v,
            Err(_) => return (ThreadId::Err, 0),
        }
        // The '.', whatever it is.
        i += n + 1;
    }
    let (r, n) = strtoul(buf.get(i..).unwrap_or(&[]), 16);
    let Ok(t) = r else { return (ThreadId::Err, 0) };
    i += n;
    if p == u64::MAX {
        return (ThreadId::AllProcesses, i);
    }
    if t == u64::MAX {
        return (ThreadId::AllThreads { pid: p as u32 }, i);
    }
    (ThreadId::One { pid: p as u32, tid: t as u32 }, i)
}

/// One parsed parameter, `GdbCmdVariant`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Param<'a> {
    /// `l` and `L`: a hex number.
    Num(u64),
    /// `s`: the rest of the packet from here, delimiters and all.
    Str(&'a [u8]),
    /// `o`: one character.
    Op(u8),
    /// `t`: a thread ID.
    Thread(ThreadId),
}

impl<'a> Param<'a> {
    /// The number, 0 for anything else.
    pub fn num(self) -> u64 {
        if let Param::Num(v) = self { v } else { 0 }
    }

    /// The string, empty for anything else.
    pub fn str(self) -> &'a [u8] {
        if let Param::Str(s) = self { s } else { &[] }
    }

    /// The character, 0 for anything else.
    pub fn op(self) -> u8 {
        if let Param::Op(c) = self { c } else { 0 }
    }

    /// The thread ID, an error for anything else.
    pub fn thread(self) -> ThreadId {
        if let Param::Thread(t) = self { t } else { ThreadId::Err }
    }
}

/// `cmd_next_param()`: past the next `delimiter`, `?` meaning any of `,;:=`, `0` the end and
/// `.` one character.
fn next_param(s: &[u8], delimiter: u8) -> usize {
    match delimiter {
        b'0' => s.len(),
        b'.' if !s.is_empty() => 1,
        _ => {
            let any: &[u8] =
                if delimiter == b'?' { b",;:=" } else { std::slice::from_ref(&delimiter) };
            match s.iter().position(|c| any.contains(c)) {
                Some(i) => i + 1,
                None => s.len(),
            }
        }
    }
}

/// `cmd_parse_params()`: the parameters `schema` describes, a type and the delimiter after it
/// for each, until either runs out. `None` when a number does not parse.
pub fn parse_params<'a>(data: &'a [u8], schema: &str) -> Option<Vec<Param<'a>>> {
    let schema = schema.as_bytes();
    let mut params = Vec::new();
    let mut d = data;
    let mut k = 0;
    while k + 1 < schema.len() && !d.is_empty() {
        let delim = schema[k + 1];
        let used = match schema[k] {
            b'l' | b'L' => {
                let (r, n) = strtoul(d, 16);
                params.push(Param::Num(r.ok()?));
                n
            }
            b's' => {
                params.push(Param::Str(d));
                0
            }
            b'o' => {
                params.push(Param::Op(d[0]));
                0
            }
            b't' => {
                let (t, n) = read_thread_id(d);
                params.push(Param::Thread(t));
                n
            }
            b'?' => 0,
            _ => return None,
        };
        d = &d[used..];
        d = &d[next_param(d, delim)..];
        k += 2;
    }
    Some(params)
}

/// `host_to_gdb_errno()`.
pub fn host_to_gdb_errno(err: i32) -> u32 {
    match err {
        1 | 2 | 4 | 5 | 9 | 13 | 14 | 16 | 17 | 19 | 20 | 21 | 22 | 23 | 24 | 27 | 28 | 29 | 30 => {
            err as u32
        }
        38 => 88,
        36 => 91,
        _ => 9999,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_parse_as_strtoul_does() {
        assert_eq!(strtoul(b"1f,", 16), (Ok(0x1f), 2));
        assert_eq!(strtoul(b"0x10", 16), (Ok(0x10), 4));
        assert_eq!(strtoul(b"0x", 16), (Ok(0), 1));
        assert_eq!(strtoul(b"-1", 16), (Ok(u64::MAX), 2));
        assert_eq!(strtoul(b" 7", 16), (Ok(7), 2));
        assert_eq!(strtoul(b",", 16), (Err(NumError::Invalid), 0));
        assert_eq!(strtoul(b"10000000000000000", 16), (Err(NumError::Range), 17));
        assert_eq!(strtoul(b"1234abc", 10), (Ok(1234), 4));
    }

    #[test]
    fn thread_ids() {
        assert_eq!(read_thread_id(b"p1f.2a;"), (ThreadId::One { pid: 0x1f, tid: 0x2a }, 6));
        assert_eq!(read_thread_id(b"-1"), (ThreadId::AllThreads { pid: 0 }, 2));
        assert_eq!(read_thread_id(b"p-1.-1"), (ThreadId::AllProcesses, 6));
        assert_eq!(read_thread_id(b"0"), (ThreadId::One { pid: 0, tid: 0 }, 1));
        assert_eq!(read_thread_id(b"px.1").0, ThreadId::Err);
    }

    #[test]
    fn schemas() {
        let p = parse_params(b"0,401000,1", "l?L?L0").unwrap();
        assert_eq!(p, [Param::Num(0), Param::Num(0x401000), Param::Num(1)]);
        let p = parse_params(b"401000,4:deadbeef", "L,L:s0").unwrap();
        assert_eq!(p, [Param::Num(0x401000), Param::Num(4), Param::Str(b"deadbeef")]);
        let p = parse_params(b"target.xml:0,ffb", "s:l,l0").unwrap();
        assert_eq!(p, [Param::Str(b"target.xml:0,ffb"), Param::Num(0), Param::Num(0xffb)]);
        let p = parse_params(b"gp1.2", "o.t0").unwrap();
        assert_eq!(p, [Param::Op(b'g'), Param::Thread(ThreadId::One { pid: 1, tid: 2 })]);
        let p = parse_params(b";1f", "?.l0").unwrap();
        assert_eq!(p, [Param::Num(0x1f)]);
        assert_eq!(parse_params(b"", "L0").unwrap(), []);
        assert!(parse_params(b"zz", "L0").is_none());
    }

    #[test]
    fn encodings() {
        let mut v = Vec::new();
        memtox(&mut v, b"a#b}");
        assert_eq!(v, b"a}\x03b}]");
        let mut m = Vec::new();
        hextomem(&mut m, b"2f61,0", 3);
        assert_eq!(m, b"/a\0");
        let mut h = Vec::new();
        memtohex(&mut h, &[0xde, 0x01]);
        assert_eq!(h, b"de01");
    }
}
