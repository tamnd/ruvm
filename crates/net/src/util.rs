// SPDX-License-Identifier: GPL-2.0-or-later

//! Helpers from net/util.c and the small parsers of net/net.c.

use std::fmt;
use std::net::{Ipv4Addr, SocketAddrV4, ToSocketAddrs};

use ruvm_base::{Error, Result};
use ruvm_qapi::cutils::strtoi64;

/// `MACAddr`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MacAddr(pub [u8; 6]);

impl MacAddr {
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 6]
    }

    pub fn is_multicast(&self) -> bool {
        self.0[0] & 1 != 0
    }
}

impl fmt::Display for MacAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.0;
        write!(f, "{:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}", a[0], a[1], a[2], a[3], a[4], a[5])
    }
}

/// C `strtol(p, &end, base)`, answering the value and the number of bytes used. Overflow
/// saturates like strtol does; no digits at all uses nothing.
fn strtol_prefix(s: &str, base: u32) -> (i64, usize) {
    match strtoi64(s, base, false) {
        Ok((v, used)) => (v, used),
        Err((_, v)) => {
            // A range error still consumed the digits; find out how many.
            let used = s
                .char_indices()
                .skip_while(|(_, c)| c.is_ascii_whitespace())
                .skip_while(|(_, c)| *c == '+' || *c == '-')
                .find(|(_, c)| !c.is_ascii_alphanumeric())
                .map_or(s.len(), |(i, _)| i);
            if v == 0 { (0, 0) } else { (v, used) }
        }
    }
}

/// `net_parse_macaddr()`: either a number up to 0xffffff, which only replaces the last three
/// bytes, or six hex bytes separated by `:` or `-`. Returns false on bad syntax, possibly after
/// changing some bytes, as QEMU does.
pub fn parse_macaddr(mac: &mut MacAddr, s: &str) -> bool {
    if let Ok((offset, _)) = strtoi64(s, 0, true) {
        if (0..=0xFF_FFFF).contains(&offset) {
            mac.0[3] = (offset >> 16) as u8;
            mac.0[4] = (offset >> 8) as u8;
            mac.0[5] = offset as u8;
            return true;
        }
    }
    let mut p = s;
    for i in 0..6 {
        let (v, used) = strtol_prefix(p, 16);
        mac.0[i] = v as u8;
        p = &p[used..];
        if i == 5 {
            if !p.is_empty() {
                return false;
            }
        } else {
            if !p.starts_with([':', '-']) {
                return false;
            }
            p = &p[1..];
        }
    }
    true
}

/// `inet_aton()`: one to four numbers in C notation (decimal, octal or hex) separated by dots.
pub fn inet_aton(s: &str) -> Option<Ipv4Addr> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.is_empty() || parts.len() > 4 {
        return None;
    }
    let mut vals = Vec::with_capacity(4);
    for p in &parts {
        if p.is_empty() || !p.as_bytes()[0].is_ascii_digit() {
            return None;
        }
        let (radix, digits) = if let Some(h) = p.strip_prefix("0x").or(p.strip_prefix("0X")) {
            (16, h)
        } else if p.len() > 1 && p.starts_with('0') {
            (8, &p[1..])
        } else {
            (10, *p)
        };
        let v = if digits.is_empty() {
            if radix == 16 {
                return None;
            }
            0
        } else {
            u32::from_str_radix(digits, radix).ok()?
        };
        vals.push(v);
    }
    let last = *vals.last()?;
    let n = vals.len();
    let max_last: u64 = 1 << (8 * (5 - n as u32));
    if u64::from(last) >= max_last && n > 1 || vals[..n - 1].iter().any(|&v| v > 255) {
        return None;
    }
    let mut addr: u32 = 0;
    for (i, v) in vals[..n - 1].iter().enumerate() {
        addr |= v << (24 - 8 * i);
    }
    addr |= last;
    Some(Ipv4Addr::from(addr))
}

/// `gethostbyname()` for IPv4.
fn resolve_v4(host: &str) -> Option<Ipv4Addr> {
    let addrs = (host, 0u16).to_socket_addrs().ok()?;
    for a in addrs {
        if let std::net::SocketAddr::V4(v4) = a {
            return Some(*v4.ip());
        }
    }
    None
}

/// `convert_host_port()`.
pub fn convert_host_port(host: &str, port: &str) -> Result<SocketAddrV4> {
    let ip = if host.is_empty() {
        Ipv4Addr::UNSPECIFIED
    } else if host.as_bytes()[0].is_ascii_digit() {
        inet_aton(host).ok_or_else(|| {
            Error::generic(format!("host address '{host}' is not a valid IPv4 address"))
        })?
    } else {
        resolve_v4(host)
            .ok_or_else(|| Error::generic(format!("can't resolve host address '{host}'")))?
    };
    let p = match strtoi64(port, 0, false) {
        Ok((v, _)) => v,
        Err(_) => return Err(Error::generic(format!("port number '{port}' is invalid"))),
    };
    Ok(SocketAddrV4::new(ip, p as u16))
}

/// `parse_host_port()`: `host:port`, split at the first colon.
pub fn parse_host_port(s: &str) -> Result<SocketAddrV4> {
    let Some((host, port)) = s.split_once(':') else {
        return Err(Error::generic(format!(
            "host address '{s}' doesn't contain ':' separating host from port"
        )));
    };
    convert_host_port(host, port)
}

/// `SocketReadState`: puts packets back together from the stream format socket and stream
/// backends use, a 4 byte big-endian length (and with `vnet_hdr` a 4 byte header length) and
/// then the data.
#[derive(Debug)]
pub struct SocketReadState {
    state: u8,
    vnet_hdr: bool,
    index: usize,
    packet_len: usize,
    /// The header length of the last packet.
    pub vnet_hdr_len: u32,
    buf: Vec<u8>,
}

impl SocketReadState {
    /// `net_socket_rs_init()`.
    pub fn new(vnet_hdr: bool) -> Self {
        SocketReadState {
            state: 0,
            vnet_hdr,
            index: 0,
            packet_len: 0,
            vnet_hdr_len: 0,
            buf: vec![0; crate::client::NET_BUFSIZE],
        }
    }

    /// Starts over, as after `net_socket_rs_init()`.
    pub fn reset(&mut self) {
        self.state = 0;
        self.index = 0;
        self.packet_len = 0;
        self.vnet_hdr_len = 0;
    }

    /// `net_fill_rstate()`: feeds bytes in and calls `finalize` with every whole packet. An
    /// oversized packet is an error, after which the state starts over.
    pub fn fill(&mut self, mut data: &[u8], mut finalize: impl FnMut(&[u8])) -> Result<()> {
        while !data.is_empty() {
            match self.state {
                0 | 1 => {
                    let l = (4 - self.index).min(data.len());
                    self.buf[self.index..self.index + l].copy_from_slice(&data[..l]);
                    data = &data[l..];
                    self.index += l;
                    if self.index == 4 {
                        let v = u32::from_be_bytes([
                            self.buf[0],
                            self.buf[1],
                            self.buf[2],
                            self.buf[3],
                        ]);
                        self.index = 0;
                        if self.state == 0 {
                            self.packet_len = v as usize;
                            if self.vnet_hdr {
                                self.state = 1;
                            } else {
                                self.state = 2;
                                self.vnet_hdr_len = 0;
                            }
                        } else {
                            self.vnet_hdr_len = v;
                            self.state = 2;
                        }
                    }
                }
                _ => {
                    let l = (self.packet_len - self.index).min(data.len());
                    if self.index + l > self.buf.len() {
                        self.index = 0;
                        self.state = 0;
                        return Err(Error::generic(
                            "serious error: oversized packet received,connection terminated.",
                        ));
                    }
                    self.buf[self.index..self.index + l].copy_from_slice(&data[..l]);
                    self.index += l;
                    data = &data[l..];
                    if self.index >= self.packet_len {
                        let len = self.packet_len;
                        self.index = 0;
                        self.state = 0;
                        finalize(&self.buf[..len]);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Makes a new file or directory in the temporary directory, named `prefix` plus six random
/// characters plus `suffix`, the way `g_file_open_tmp()` and `g_dir_make_tmp()` fill in their
/// `XXXXXX`. Only the owner may use it.
#[cfg(unix)]
pub(crate) fn make_temp(
    prefix: &str,
    suffix: &str,
    dir: bool,
) -> std::io::Result<std::path::PathBuf> {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};

    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let base = std::env::temp_dir();
    let mut last = std::io::Error::from(std::io::ErrorKind::AlreadyExists);
    for attempt in 0..100u32 {
        let mut n = RandomState::new().hash_one((std::process::id(), attempt));
        let mut name = String::from(prefix);
        for _ in 0..6 {
            name.push(char::from(CHARS[(n % CHARS.len() as u64) as usize]));
            n /= CHARS.len() as u64;
        }
        name.push_str(suffix);
        let path = base.join(name);
        let r = if dir {
            std::fs::DirBuilder::new().mode(0o700).create(&path)
        } else {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .map(drop)
        };
        match r {
            Ok(()) => return Ok(path),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = e,
            Err(e) => return Err(e),
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macaddr_forms() {
        let mut m = MacAddr([0x52, 0x54, 0, 0x12, 0x34, 0x56]);
        assert!(parse_macaddr(&mut m, "0x123456"));
        assert_eq!(m.to_string(), "52:54:00:12:34:56");
        assert!(parse_macaddr(&mut m, "1"));
        assert_eq!(m.to_string(), "52:54:00:00:00:01");
        assert!(parse_macaddr(&mut m, "02:aa-bb:cc:dd:ee"));
        assert_eq!(m.to_string(), "02:aa:bb:cc:dd:ee");
        assert!(!parse_macaddr(&mut m, "02:aa:bb:cc:dd"));
        assert!(!parse_macaddr(&mut m, "02:aa:bb:cc:dd:ee:ff"));
        assert!(!parse_macaddr(&mut m, "zz"));
    }

    #[test]
    fn inet_aton_forms() {
        assert_eq!(inet_aton("127.0.0.1"), Some(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(inet_aton("127.1"), Some(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(inet_aton("0x7f.1"), Some(Ipv4Addr::new(127, 0, 0, 1)));
        assert_eq!(inet_aton("1.2.3.256"), None);
        assert_eq!(inet_aton("1..2"), None);
    }

    #[test]
    fn host_port() {
        assert_eq!(parse_host_port(":1234").unwrap().to_string(), "0.0.0.0:1234");
        assert_eq!(parse_host_port("230.0.0.1:0x10").unwrap().to_string(), "230.0.0.1:16");
        assert_eq!(
            parse_host_port("1234").unwrap_err().message(),
            "host address '1234' doesn't contain ':' separating host from port"
        );
        assert_eq!(
            parse_host_port("1.2.3.999:1").unwrap_err().message(),
            "host address '1.2.3.999' is not a valid IPv4 address"
        );
        assert_eq!(
            parse_host_port("127.0.0.1:x").unwrap_err().message(),
            "port number 'x' is invalid"
        );
    }

    #[test]
    fn read_state_reassembles() {
        let mut rs = SocketReadState::new(false);
        let mut got = Vec::new();
        let mut wire = Vec::new();
        for p in [&b"hello"[..], b"", b"world!"] {
            wire.extend_from_slice(&(p.len() as u32).to_be_bytes());
            wire.extend_from_slice(p);
        }
        for chunk in wire.chunks(3) {
            rs.fill(chunk, |p| got.push(p.to_vec())).unwrap();
        }
        assert_eq!(got, vec![b"hello".to_vec(), Vec::new(), b"world!".to_vec()]);

        let mut rs = SocketReadState::new(false);
        let big = (crate::client::NET_BUFSIZE as u32 + 1).to_be_bytes();
        let mut wire = big.to_vec();
        wire.extend(std::iter::repeat_n(0u8, crate::client::NET_BUFSIZE + 1));
        assert!(rs.fill(&wire, |_| {}).is_err());
    }
}
