// SPDX-License-Identifier: GPL-2.0-or-later

//! NBD file names: `nbd://host[:port]/export`, `nbd+unix:///export?socket=path` and the old
//! `nbd:host:port[:exportname=name]` and `nbd:unix:path[:exportname=name]` forms, from
//! block/nbd.c, with `inet_parse()` from util/qemu-sockets.c for the host and port.
//!
//! Like QEMU, the parsers put flat options (`server.type`, `server.host`, `export` and so on)
//! into a [`QDict`], and [`nbd_options_from_qdict`] turns such a dict, plus the legacy `host`,
//! `port` and `path` options, into [`BlockdevOptionsNbd`].

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType, QemuOptsList};
use ruvm_qapi::types::{
    BlockdevOptionsNbd, FdSocketAddress, InetSocketAddress, SocketAddress, SocketAddressU,
    UnixSocketAddress, VsockSocketAddress,
};
use ruvm_qapi::{QDict, QValue};

use super::proto::NBD_DEFAULT_PORT;

/// `EN_OPTSTR`.
const EN_OPTSTR: &str = ":exportname=";

/// The parts of a URI `g_uri_parse()` returns, decoded.
#[derive(Debug, Default)]
struct Uri {
    scheme: String,
    host: Option<String>,
    port: i32,
    path: String,
    query: Option<String>,
}

fn hexval(c: u8) -> Option<u8> {
    (c as char).to_digit(16).map(|d| d as u8)
}

/// `uri_decoder()` without `G_URI_FLAGS_PARSE_RELAXED`: `%XX` escapes are decoded, a bad
/// escape, a decoded NUL or a raw character that is not printable ASCII is an error.
fn uri_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'%' {
            if i + 2 >= b.len() {
                return None;
            }
            let (Some(h), Some(l)) = (hexval(b[i + 1]), hexval(b[i + 2])) else {
                return None;
            };
            let v = (h << 4) | l;
            if v == 0 {
                return None;
            }
            out.push(v);
            i += 3;
        } else {
            if !c.is_ascii_graphic() {
                return None;
            }
            out.push(c);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

/// `g_uri_parse()` for the absolute URIs NBD uses: `scheme://[userinfo@]host[:port]/path?query`.
fn uri_parse(s: &str) -> Option<Uri> {
    let colon = s.find(':')?;
    let scheme = &s[..colon];
    let mut chars = scheme.chars();
    if !chars.next().is_some_and(|c| c.is_ascii_alphabetic())
        || !chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    {
        return None;
    }
    let mut uri = Uri { scheme: scheme.to_ascii_lowercase(), port: -1, ..Uri::default() };
    let mut rest = &s[colon + 1..];
    if let Some(i) = rest.find('#') {
        uri_decode(&rest[i + 1..])?;
        rest = &rest[..i];
    }
    if let Some(i) = rest.find('?') {
        uri.query = Some(uri_decode(&rest[i + 1..])?);
        rest = &rest[..i];
    }
    if let Some(auth) = rest.strip_prefix("//") {
        let end = auth.find('/').unwrap_or(auth.len());
        let (authority, path) = auth.split_at(end);
        rest = path;
        let hostport = match authority.rfind('@') {
            Some(i) => {
                uri_decode(&authority[..i])?;
                &authority[i + 1..]
            }
            None => authority,
        };
        let (host, port) = if let Some(v6) = hostport.strip_prefix('[') {
            let close = v6.find(']')?;
            let after = &v6[close + 1..];
            let port = match after {
                "" => None,
                p => Some(p.strip_prefix(':')?),
            };
            (v6[..close].to_string(), port)
        } else {
            match hostport.rfind(':') {
                Some(i) => (uri_decode(&hostport[..i])?, Some(&hostport[i + 1..])),
                None => (uri_decode(hostport)?, None),
            }
        };
        if let Some(p) = port {
            if !p.is_empty() {
                if !p.bytes().all(|c| c.is_ascii_digit()) {
                    return None;
                }
                let v: u32 = p.parse().ok()?;
                if v > 65535 {
                    return None;
                }
                uri.port = v as i32;
            }
        }
        uri.host = Some(host);
    }
    uri.path = uri_decode(rest)?;
    Some(uri)
}

/// `g_uri_parse_params()` with `&` as separator and no flags: every parameter needs an `=`,
/// both halves are decoded, and a later value for a key replaces an earlier one.
fn uri_parse_params(q: &str) -> Option<Vec<(String, String)>> {
    let mut out: Vec<(String, String)> = Vec::new();
    for p in q.split('&') {
        if p.is_empty() && q.is_empty() {
            break;
        }
        let eq = p.find('=')?;
        let k = uri_decode(&p[..eq])?;
        let v = uri_decode(&p[eq + 1..])?;
        match out.iter_mut().find(|(ok, _)| *ok == k) {
            Some(e) => e.1 = v,
            None => out.push((k, v)),
        }
    }
    Some(out)
}

/// `nbd_parse_uri()`. QEMU only cares whether this fails, not why.
fn nbd_parse_uri(filename: &str, options: &mut QDict) -> std::result::Result<(), ()> {
    let uri = uri_parse(filename).ok_or(())?;
    let is_unix = match uri.scheme.as_str() {
        "nbd" | "nbd+tcp" => false,
        "nbd+unix" => true,
        _ => return Err(()),
    };
    let p = uri.path.strip_prefix('/').unwrap_or(&uri.path);
    if !p.is_empty() {
        options.put("export", QValue::str(p));
    }
    let mut qp = None;
    if let Some(q) = &uri.query {
        let params = uri_parse_params(q).ok_or(())?;
        let n = params.len();
        if n > 1 || (is_unix && n == 0) || (!is_unix && n > 0) {
            return Err(());
        }
        qp = Some(params);
    }
    let server = uri.host.filter(|h| !h.is_empty());
    if is_unix {
        let socket =
            qp.as_ref().and_then(|q| q.iter().find(|(k, _)| k == "socket")).map(|(_, v)| v.clone());
        let Some(socket) = socket else {
            return Err(());
        };
        if server.is_some() || uri.port != -1 {
            return Err(());
        }
        options.put("server.type", QValue::str("unix"));
        options.put("server.path", QValue::str(socket));
    } else {
        let Some(server) = server else {
            return Err(());
        };
        options.put("server.type", QValue::str("inet"));
        options.put("server.host", QValue::str(server));
        let port = if uri.port > 0 { uri.port } else { i32::from(NBD_DEFAULT_PORT) };
        options.put("server.port", QValue::str(port.to_string()));
    }
    Ok(())
}

/// `nbd_has_filename_options_conflict()`.
fn filename_options_conflict(options: &QDict) -> Result<()> {
    for k in options.keys() {
        if matches!(k, "host" | "port" | "path" | "export") || k.starts_with("server.") {
            return Err(Error::generic(format!("Option '{k}' cannot be used with a file name")));
        }
    }
    Ok(())
}

/// `nbd_parse_filename()`: adds the options `filename` stands for to `options`.
pub fn nbd_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    filename_options_conflict(options)?;
    if filename.contains("://") {
        return nbd_parse_uri(filename, options)
            .map_err(|()| Error::generic("No valid URL specified"));
    }
    let mut file = filename;
    if let Some(i) = file.find(EN_OPTSTR) {
        let name = &file[i + EN_OPTSTR.len()..];
        if name.is_empty() {
            return Ok(());
        }
        options.put("export", QValue::str(name));
        file = &file[..i];
    }
    let Some(host_spec) = file.strip_prefix("nbd:") else {
        return Err(Error::generic("File name string for NBD must start with 'nbd:'"));
    };
    if host_spec.is_empty() {
        return Ok(());
    }
    if let Some(path) = host_spec.strip_prefix("unix:") {
        options.put("server.type", QValue::str("unix"));
        options.put("server.path", QValue::str(path));
    } else {
        let addr = inet_parse(host_spec)?;
        options.put("server.type", QValue::str("inet"));
        options.put("server.host", QValue::str(addr.host));
        options.put("server.port", QValue::str(addr.port));
    }
    Ok(())
}

fn inet_opts() -> QemuOptsList {
    let mut desc = vec![
        QemuOptDesc::new("addr", QemuOptType::String),
        QemuOptDesc::new("numeric", QemuOptType::Bool),
        QemuOptDesc::new("to", QemuOptType::Number),
        QemuOptDesc::new("ipv4", QemuOptType::Bool),
        QemuOptDesc::new("ipv6", QemuOptType::Bool),
        QemuOptDesc::new("keep-alive", QemuOptType::Bool),
    ];
    // HAVE_TCP_KEEPCNT, HAVE_TCP_KEEPIDLE (TCP_KEEPALIVE on macOS) and HAVE_TCP_KEEPINTVL.
    if cfg!(unix) {
        desc.push(QemuOptDesc::new("keep-alive-count", QemuOptType::Number));
        desc.push(QemuOptDesc::new("keep-alive-idle", QemuOptType::Number));
        desc.push(QemuOptDesc::new("keep-alive-interval", QemuOptType::Number));
    }
    // HAVE_IPPROTO_MPTCP.
    if cfg!(target_os = "linux") {
        desc.push(QemuOptDesc::new("mptcp", QemuOptType::Bool));
    }
    QemuOptsList::new("InetSocketAddress", &desc).with_implied_opt_name("addr")
}

/// `inet_parse()`: `host:port` or `[ipv6]:port`, then comma separated options.
pub fn inet_parse(s: &str) -> Result<InetSocketAddress> {
    let mut list = inet_opts();
    let opts = list.parse(s, true)?;
    let Some(addr_str) = opts.get("addr") else {
        return Err(Error::generic("error parsing address ''"));
    };
    let mut addr = InetSocketAddress::default();
    if s.starts_with('[') {
        let end = addr_str.find("]:");
        match end {
            Some(e) if e >= 2 && addr_str.len() - e >= 3 => {
                addr.host = addr_str[1..e].to_string();
                addr.port = addr_str[e + 2..].to_string();
            }
            _ => {
                return Err(Error::generic(format!("error parsing IPv6 address '{addr_str}'")));
            }
        }
    } else {
        match addr_str.find(':') {
            Some(i) if addr_str.len() - i >= 2 => {
                addr.host = addr_str[..i].to_string();
                addr.port = addr_str[i + 1..].to_string();
            }
            _ => return Err(Error::generic(format!("error parsing address '{addr_str}'"))),
        }
    }
    let flag = |name: &str| opts.find(name).map(|_| opts.get_bool(name, false));
    let num = |name: &str| opts.find(name).map(|_| opts.get_number(name, 0));
    addr.numeric = flag("numeric");
    addr.to = num("to").map(|v| v as u16);
    addr.ipv4 = flag("ipv4");
    addr.ipv6 = flag("ipv6");
    addr.keep_alive = flag("keep-alive");
    #[cfg(unix)]
    {
        addr.keep_alive_count = num("keep-alive-count").map(|v| v as u32);
        addr.keep_alive_idle = num("keep-alive-idle").map(|v| v as u32);
        addr.keep_alive_interval = num("keep-alive-interval").map(|v| v as u32);
    }
    #[cfg(target_os = "linux")]
    {
        addr.mptcp = flag("mptcp");
    }
    Ok(addr)
}

fn take_str(d: &mut QDict, key: &str) -> Result<Option<String>> {
    match d.remove(key) {
        None => Ok(None),
        Some(v) => match v.as_str() {
            Some(s) => Ok(Some(s.to_string())),
            None => {
                Err(Error::generic(format!("Invalid parameter type for '{key}', expected: string")))
            }
        },
    }
}

fn take_num(d: &mut QDict, key: &str) -> Result<Option<u64>> {
    match d.remove(key) {
        None => Ok(None),
        Some(v) => {
            if let Some(n) = v.as_u64() {
                return Ok(Some(n));
            }
            match v.as_str().and_then(|s| s.parse::<u64>().ok()) {
                Some(n) => Ok(Some(n)),
                None => Err(Error::generic(format!("Parameter '{key}' expects uint32"))),
            }
        }
    }
}

fn take_bool(d: &mut QDict, key: &str) -> Result<Option<bool>> {
    match d.remove(key) {
        None => Ok(None),
        Some(v) => {
            if let Some(b) = v.as_bool() {
                return Ok(Some(b));
            }
            match v.as_str() {
                Some("on" | "yes" | "true" | "y") => Ok(Some(true)),
                Some("off" | "no" | "false" | "n") => Ok(Some(false)),
                _ => Err(Error::generic(format!("Parameter '{key}' expects 'on' or 'off'"))),
            }
        }
    }
}

fn u32_opt(key: &str, v: Option<u64>) -> Result<Option<u32>> {
    match v {
        Some(n) if n > u64::from(u32::MAX) => {
            Err(Error::generic(format!("Parameter '{key}' expects uint32")))
        }
        v => Ok(v.map(|n| n as u32)),
    }
}

/// `nbd_process_legacy_socket_options()` and `nbd_config()`: the options of an `nbd` node from
/// flat keys, as `-drive` and file names give them. Keys that are not NBD options are left in
/// `options`.
pub fn nbd_options_from_qdict(options: &mut QDict) -> Result<BlockdevOptionsNbd> {
    let path = take_str(options, "path")?;
    let host = take_str(options, "host")?;
    let port = take_str(options, "port")?;
    if path.is_some() || host.is_some() || port.is_some() {
        if options.keys().any(|k| k.starts_with("server.")) {
            return Err(Error::generic("Cannot use 'server' and path/host/port at the same time"));
        }
        if path.is_some() && host.is_some() {
            return Err(Error::generic("path and host may not be used at the same time"));
        }
        if let Some(path) = path {
            if port.is_some() {
                return Err(Error::generic("port may not be used without host"));
            }
            options.put("server.type", QValue::str("unix"));
            options.put("server.path", QValue::str(path));
        } else if let Some(host) = host {
            options.put("server.type", QValue::str("inet"));
            options.put("server.host", QValue::str(host));
            let port = port.unwrap_or_else(|| NBD_DEFAULT_PORT.to_string());
            options.put("server.port", QValue::str(port));
        }
    }

    let server_keys: Vec<String> =
        options.keys().filter(|k| k.starts_with("server.")).map(str::to_string).collect();
    if server_keys.is_empty() {
        return Err(Error::generic("NBD server address missing"));
    }
    let Some(ty) = take_str(options, "server.type")? else {
        return Err(Error::generic("Parameter 'server.type' is missing"));
    };
    let missing = |k: &str| Error::generic(format!("Parameter '{k}' is missing"));
    let u = match ty.as_str() {
        "inet" => {
            let host = take_str(options, "server.host")?.ok_or_else(|| missing("server.host"))?;
            let port = take_str(options, "server.port")?.ok_or_else(|| missing("server.port"))?;
            let to = take_num(options, "server.to")?;
            if to.is_some_and(|t| t > 65535) {
                return Err(Error::generic("Parameter 'server.to' expects uint16"));
            }
            // Which members there are depends on the host; some are set below.
            #[allow(clippy::needless_update)]
            #[cfg_attr(not(unix), allow(unused_mut))]
            let mut inet = InetSocketAddress {
                host,
                port,
                numeric: take_bool(options, "server.numeric")?,
                to: to.map(|t| t as u16),
                ipv4: take_bool(options, "server.ipv4")?,
                ipv6: take_bool(options, "server.ipv6")?,
                keep_alive: take_bool(options, "server.keep-alive")?,
                ..InetSocketAddress::default()
            };
            #[cfg(unix)]
            {
                inet.keep_alive_count = u32_opt(
                    "server.keep-alive-count",
                    take_num(options, "server.keep-alive-count")?,
                )?;
                inet.keep_alive_idle = u32_opt(
                    "server.keep-alive-idle",
                    take_num(options, "server.keep-alive-idle")?,
                )?;
                inet.keep_alive_interval = u32_opt(
                    "server.keep-alive-interval",
                    take_num(options, "server.keep-alive-interval")?,
                )?;
            }
            #[cfg(target_os = "linux")]
            {
                inet.mptcp = take_bool(options, "server.mptcp")?;
            }
            SocketAddressU::Inet(inet)
        }
        "unix" => {
            let path = take_str(options, "server.path")?.ok_or_else(|| missing("server.path"))?;
            // Linux has more members; they are set below.
            #[allow(clippy::needless_update)]
            #[cfg_attr(not(target_os = "linux"), allow(unused_mut))]
            let mut unix = UnixSocketAddress { path, ..UnixSocketAddress::default() };
            #[cfg(target_os = "linux")]
            {
                unix.abstract_ = take_bool(options, "server.abstract")?;
                unix.tight = take_bool(options, "server.tight")?;
            }
            SocketAddressU::Unix(unix)
        }
        "vsock" => {
            let cid = take_str(options, "server.cid")?.ok_or_else(|| missing("server.cid"))?;
            let port = take_str(options, "server.port")?.ok_or_else(|| missing("server.port"))?;
            SocketAddressU::Vsock(VsockSocketAddress { cid, port })
        }
        "fd" => {
            let s = take_str(options, "server.str")?.ok_or_else(|| missing("server.str"))?;
            SocketAddressU::Fd(FdSocketAddress { str: s })
        }
        other => {
            return Err(Error::generic(format!(
                "Parameter 'server.type' does not accept value '{other}'"
            )));
        }
    };
    if let Some(k) = options.keys().find(|k| k.starts_with("server.")) {
        return Err(Error::generic(format!("Parameter '{k}' is unexpected")));
    }
    Ok(BlockdevOptionsNbd {
        server: SocketAddress { u },
        export: take_str(options, "export")?,
        tls_creds: take_str(options, "tls-creds")?,
        tls_hostname: take_str(options, "tls-hostname")?,
        x_dirty_bitmap: take_str(options, "x-dirty-bitmap")?,
        reconnect_delay: u32_opt("reconnect-delay", take_num(options, "reconnect-delay")?)?,
        open_timeout: u32_opt("open-timeout", take_num(options, "open-timeout")?)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(f: &str) -> Result<Vec<(String, String)>> {
        let mut d = QDict::new();
        nbd_parse_filename(f, &mut d)?;
        Ok(d.iter_inserted()
            .map(|(k, v)| (k.to_string(), v.as_str().unwrap_or("").to_string()))
            .collect())
    }

    fn kv(v: &[(&str, &str)]) -> Vec<(String, String)> {
        v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn uris() {
        assert_eq!(
            parse("nbd://localhost/foo").unwrap(),
            kv(&[
                ("export", "foo"),
                ("server.type", "inet"),
                ("server.host", "localhost"),
                ("server.port", "10809")
            ])
        );
        assert_eq!(
            parse("nbd+tcp://127.0.0.1:1234").unwrap(),
            kv(&[("server.type", "inet"), ("server.host", "127.0.0.1"), ("server.port", "1234")])
        );
        assert_eq!(
            parse("NBD://[::1]:0/a%20b").unwrap(),
            kv(&[
                ("export", "a b"),
                ("server.type", "inet"),
                ("server.host", "::1"),
                ("server.port", "10809")
            ])
        );
        assert_eq!(
            parse("nbd+unix:///exp?socket=/tmp/s%2Cock").unwrap(),
            kv(&[("export", "exp"), ("server.type", "unix"), ("server.path", "/tmp/s,ock")])
        );
        assert_eq!(
            parse("nbd+unix://?socket=/tmp/s").unwrap(),
            kv(&[("server.type", "unix"), ("server.path", "/tmp/s")])
        );
        for bad in [
            "nbd+unix://host/?socket=/s",
            "nbd+unix:///e",
            "nbd+unix:///e?socket=/s&x=1",
            "nbd+unix:///e?path=/s",
            "nbd+unix://:10/?socket=/s",
            "nbd:///e",
            "nbd://h/e?socket=/s",
            "nbd://h:99999/",
            "nbd://h:x/",
            "http://h/",
            "nbd://h/%zz",
            "nbd://h/a b",
            "1nbd://h/",
        ] {
            assert_eq!(parse(bad).unwrap_err().message(), "No valid URL specified", "{bad}");
        }
    }

    #[test]
    fn legacy_names() {
        assert_eq!(
            parse("nbd:localhost:10810:exportname=disk").unwrap(),
            kv(&[
                ("export", "disk"),
                ("server.type", "inet"),
                ("server.host", "localhost"),
                ("server.port", "10810")
            ])
        );
        assert_eq!(
            parse("nbd:unix:/tmp/sock").unwrap(),
            kv(&[("server.type", "unix"), ("server.path", "/tmp/sock")])
        );
        assert_eq!(
            parse("nbd:[::1]:5000").unwrap(),
            kv(&[("server.type", "inet"), ("server.host", "::1"), ("server.port", "5000")])
        );
        assert_eq!(parse("nbd:").unwrap(), kv(&[]));
        assert_eq!(parse("nbd:h:1:exportname=").unwrap(), kv(&[]));
        let e = |f| parse(f).unwrap_err().message().to_string();
        assert_eq!(e("foo:h:1"), "File name string for NBD must start with 'nbd:'");
        assert_eq!(e("nbd:localhost"), "error parsing address 'localhost'");
        assert_eq!(e("nbd:localhost:"), "error parsing address 'localhost:'");
        assert_eq!(e("nbd:[::1]"), "error parsing IPv6 address '[::1]'");
        assert_eq!(e("nbd:[::1]:"), "error parsing IPv6 address '[::1]:'");

        let mut d = QDict::new();
        d.put("server.host", QValue::str("x"));
        let err = nbd_parse_filename("nbd:h:1", &mut d).unwrap_err();
        assert_eq!(err.message(), "Option 'server.host' cannot be used with a file name");
    }

    #[test]
    fn inet() {
        let a = inet_parse("host:80,ipv4=on,to=90,keep-alive=off").unwrap();
        assert_eq!(a.host, "host");
        assert_eq!(a.port, "80");
        assert_eq!(a.ipv4, Some(true));
        assert_eq!(a.ipv6, None);
        assert_eq!(a.to, Some(90));
        assert_eq!(a.keep_alive, Some(false));
        assert_eq!(inet_parse(":80").unwrap().host, "");
        assert_eq!(inet_parse("ipv4=on").unwrap_err().message(), "error parsing address ''");
    }

    #[test]
    fn options() {
        let mut d = QDict::new();
        d.put("host", QValue::str("h"));
        d.put("export", QValue::str("e"));
        let o = nbd_options_from_qdict(&mut d).unwrap();
        let SocketAddressU::Inet(i) = &o.server.u else { panic!() };
        assert_eq!((i.host.as_str(), i.port.as_str()), ("h", "10809"));
        assert_eq!(o.export.as_deref(), Some("e"));

        let msg = |pairs: &[(&str, &str)]| {
            let mut d = QDict::new();
            for (k, v) in pairs {
                d.put(*k, QValue::str(*v));
            }
            nbd_options_from_qdict(&mut d).unwrap_err().message().to_string()
        };
        assert_eq!(msg(&[]), "NBD server address missing");
        assert_eq!(
            msg(&[("path", "/p"), ("host", "h")]),
            "path and host may not be used at the same time"
        );
        assert_eq!(msg(&[("path", "/p"), ("port", "1")]), "port may not be used without host");
        assert_eq!(
            msg(&[("host", "h"), ("server.type", "inet")]),
            "Cannot use 'server' and path/host/port at the same time"
        );
    }
}
