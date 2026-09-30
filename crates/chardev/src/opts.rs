// SPDX-License-Identifier: GPL-2.0-or-later

//! `-chardev` and the old `-serial`/`-monitor` strings: the `chardev` option list,
//! `qemu_chr_parse_opts()` and `qemu_chr_parse_compat()` from chardev/char.c, with the socket
//! backends' `chr_parse` hooks.

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{OptsHandle, QemuOptDesc, QemuOptType, QemuOpts, QemuOptsList};
use ruvm_qapi::types::{
    ChardevBackend, ChardevBackendU, ChardevCommon, ChardevCommonWrapper, ChardevFile,
    ChardevFileWrapper, ChardevHostdev, ChardevHostdevWrapper, ChardevMux, ChardevMuxWrapper,
    ChardevRingbuf, ChardevRingbufWrapper, ChardevSocket, ChardevSocketWrapper, ChardevStdio,
    ChardevStdioWrapper, FdSocketAddress, FdSocketAddressWrapper, InetSocketAddress,
    InetSocketAddressWrapper, SocketAddressLegacy, SocketAddressLegacyU, UnixSocketAddress,
    UnixSocketAddressWrapper,
};

#[cfg(unix)]
use ruvm_qapi::types::{ChardevPty, ChardevPtyWrapper};

use QemuOptType::{Bool, Number, Size, String as Str};

const DESC: &[QemuOptDesc] = &[
    QemuOptDesc::new("backend", Str),
    QemuOptDesc::new("path", Str),
    QemuOptDesc::new("input-path", Str),
    QemuOptDesc::new("host", Str),
    QemuOptDesc::new("port", Str),
    QemuOptDesc::new("fd", Str),
    QemuOptDesc::new("localaddr", Str),
    QemuOptDesc::new("localport", Str),
    QemuOptDesc::new("to", Number),
    QemuOptDesc::new("ipv4", Bool),
    QemuOptDesc::new("ipv6", Bool),
    QemuOptDesc::new("wait", Bool),
    QemuOptDesc::new("server", Bool),
    QemuOptDesc::new("delay", Bool),
    QemuOptDesc::new("nodelay", Bool),
    QemuOptDesc::new("reconnect-ms", Number),
    QemuOptDesc::new("telnet", Bool),
    QemuOptDesc::new("tn3270", Bool),
    QemuOptDesc::new("tls-creds", Str),
    QemuOptDesc::new("tls-authz", Str),
    QemuOptDesc::new("websocket", Bool),
    QemuOptDesc::new("width", Number),
    QemuOptDesc::new("height", Number),
    QemuOptDesc::new("cols", Number),
    QemuOptDesc::new("rows", Number),
    QemuOptDesc::new("encoding", Str),
    QemuOptDesc::new("mux", Bool),
    QemuOptDesc::new("signal", Bool),
    QemuOptDesc::new("name", Str),
    QemuOptDesc::new("debug", Number),
    QemuOptDesc::new("size", Size),
    QemuOptDesc::new("chardev", Str),
    QemuOptDesc::new("chardevs.0", Str),
    QemuOptDesc::new("chardevs.1", Str),
    QemuOptDesc::new("chardevs.2", Str),
    QemuOptDesc::new("chardevs.3", Str),
    QemuOptDesc::new("append", Bool),
    QemuOptDesc::new("logfile", Str),
    QemuOptDesc::new("logappend", Bool),
    QemuOptDesc::new("logtimestamp", Bool),
    QemuOptDesc::new("mouse", Bool),
    QemuOptDesc::new("clipboard", Bool),
    #[cfg(target_os = "linux")]
    QemuOptDesc::new("tight", Bool).default_value("on"),
    #[cfg(target_os = "linux")]
    QemuOptDesc::new("abstract", Bool),
];

/// `qemu_chardev_opts`, the list `-chardev` fills.
pub fn chardev_opts() -> QemuOptsList {
    QemuOptsList::new("chardev", DESC).with_implied_opt_name("backend")
}

/// The chardev types QEMU has that a user can name. Those ruvm lacks fail when opened rather
/// than being unknown.
const QEMU_BACKENDS: &[&str] = &[
    "braille",
    #[cfg(windows)]
    "console",
    "dbus",
    "file",
    "hub",
    "memory",
    "msmouse",
    "mux",
    "null",
    "parallel",
    "pipe",
    #[cfg(unix)]
    "pty",
    "qemu-vdagent",
    "ringbuf",
    "serial",
    "socket",
    "spiceport",
    "spicevmc",
    "stdio",
    "testdev",
    "udp",
    "vc",
    "wctablet",
];

/// `char_get_class()`.
fn check_driver(name: &str) -> Result<()> {
    if QEMU_BACKENDS.contains(&name) {
        Ok(())
    } else {
        Err(Error::generic(format!("'{name}' is not a valid char driver name")))
    }
}

/// What `-chardev help` prints.
pub fn backend_help() -> String {
    let mut s = String::from("Available chardev backend types: ");
    for name in crate::BACKENDS {
        s.push_str("\n  ");
        s.push_str(name);
    }
    s
}

fn opts_id(opts: &QemuOpts) -> &str {
    opts.id().unwrap_or("(null)")
}

/// `qemu_chr_parse_common()`.
fn parse_common(opts: &QemuOpts) -> ChardevCommon {
    ChardevCommon {
        logfile: opts.get("logfile").map(str::to_string),
        logappend: Some(opts.get_bool("logappend", false)),
        logtimestamp: Some(opts.get_bool("logtimestamp", false)),
    }
}

/// `tcp_chr_parse()`.
fn parse_socket(opts: &QemuOpts) -> Result<ChardevSocket> {
    let path = opts.get("path");
    let host = opts.get("host");
    let port = opts.get("port");
    let fd = opts.get("fd");
    if path.is_some() as u8 + fd.is_some() as u8 + host.is_some() as u8 > 1 {
        return Err(Error::generic("None or one of 'path', 'fd' or 'host' option required."));
    }
    if host.is_some() && port.is_none() {
        return Err(Error::generic("chardev: socket: no port given"));
    }
    let common = parse_common(opts);
    let delay = opts.get("delay").is_some();
    let nodelay = opts.get("nodelay").is_some();
    if delay && nodelay {
        return Err(Error::generic("'delay' and 'nodelay' are mutually exclusive"));
    }
    let server = opts.get_bool("server", false);
    let flag = |name: &str| opts.get(name).map(|_| opts.get_bool(name, false));

    let u = if let Some(path) = path {
        #[cfg(target_os = "linux")]
        let data = UnixSocketAddress {
            path: path.to_string(),
            tight: Some(opts.get_bool("tight", true)),
            abstract_: Some(opts.get_bool("abstract", false)),
        };
        #[cfg(not(target_os = "linux"))]
        let data = UnixSocketAddress { path: path.to_string() };
        SocketAddressLegacyU::Unix(UnixSocketAddressWrapper { data })
    } else if let Some(host) = host {
        SocketAddressLegacyU::Inet(InetSocketAddressWrapper {
            data: InetSocketAddress {
                host: host.to_string(),
                port: port.unwrap_or_default().to_string(),
                to: opts.get("to").map(|_| opts.get_number("to", 0) as u16),
                ipv4: flag("ipv4"),
                ipv6: flag("ipv6"),
                ..Default::default()
            },
        })
    } else {
        SocketAddressLegacyU::Fd(FdSocketAddressWrapper {
            data: FdSocketAddress { str: fd.unwrap_or_default().to_string() },
        })
    };

    Ok(ChardevSocket {
        logfile: common.logfile,
        logappend: common.logappend,
        logtimestamp: common.logtimestamp,
        addr: SocketAddressLegacy { u },
        tls_creds: opts.get("tls-creds").map(str::to_string),
        tls_authz: opts.get("tls-authz").map(str::to_string),
        // QMP defaults both of these differently, so they are always set.
        server: Some(server),
        wait: (opts.find("wait").is_some() || server).then(|| opts.get_bool("wait", true)),
        nodelay: (delay || nodelay)
            .then(|| !opts.get_bool("delay", true) || opts.get_bool("nodelay", false)),
        telnet: flag("telnet"),
        tn3270: flag("tn3270"),
        websocket: flag("websocket"),
        reconnect_ms: opts.find("reconnect-ms").map(|_| opts.get_number("reconnect-ms", 0) as i64),
    })
}

/// `file_chr_parse()`.
fn parse_file(opts: &QemuOpts) -> Result<ChardevFile> {
    let Some(path) = opts.get("path") else {
        return Err(Error::generic("chardev: file: no filename given"));
    };
    let inpath = opts.get("input-path");
    #[cfg(windows)]
    if inpath.is_some() {
        return Err(Error::generic("chardev: file: input-path not supported on Windows"));
    }
    let c = parse_common(opts);
    Ok(ChardevFile {
        logfile: c.logfile,
        logappend: c.logappend,
        logtimestamp: c.logtimestamp,
        in_: inpath.map(str::to_string),
        out: path.to_string(),
        append: Some(opts.get_bool("append", false)),
    })
}

/// `pipe_chr_parse()`.
fn parse_pipe(opts: &QemuOpts) -> Result<ChardevHostdev> {
    let Some(device) = opts.get("path") else {
        return Err(Error::generic("chardev: pipe: no device path given"));
    };
    let c = parse_common(opts);
    Ok(ChardevHostdev {
        logfile: c.logfile,
        logappend: c.logappend,
        logtimestamp: c.logtimestamp,
        device: device.to_string(),
    })
}

/// `mux_chr_parse()`.
fn parse_mux(opts: &QemuOpts) -> Result<ChardevMux> {
    let Some(chardev) = opts.get("chardev") else {
        return Err(Error::generic("chardev: mux: no chardev given"));
    };
    let c = parse_common(opts);
    Ok(ChardevMux {
        logfile: c.logfile,
        logappend: c.logappend,
        logtimestamp: c.logtimestamp,
        chardev: chardev.to_string(),
    })
}

/// `ringbuf_chr_parse()`. QEMU keeps the size in an `int` on the way, so it does too.
fn parse_ringbuf(opts: &QemuOpts) -> ChardevRingbuf {
    let c = parse_common(opts);
    let size = opts.get_size("size", 0) as i32;
    ChardevRingbuf {
        logfile: c.logfile,
        logappend: c.logappend,
        logtimestamp: c.logtimestamp,
        size: (size != 0).then_some(i64::from(size)),
    }
}

/// `qemu_chr_parse_opts()`: the backend `-chardev` describes. Backends ruvm does not have are
/// accepted here and refused when opened.
pub fn parse_opts(opts: &QemuOpts) -> Result<ChardevBackend> {
    let Some(name) = opts.get("backend") else {
        return Err(Error::generic(format!("chardev: \"{}\" missing backend", opts_id(opts))));
    };
    check_driver(name)?;
    let has_size = name == "vc";
    if !has_size && opts.has_any(&["width", "height", "cols", "rows"]) {
        return Err(Error::generic(format!(
            "chardev '{}' does not support size options",
            opts_id(opts)
        )));
    }
    if !has_size && opts.get("encoding").is_some() {
        return Err(Error::generic(format!(
            "chardev '{}' does not support encoding option",
            opts_id(opts)
        )));
    }
    let u = match name {
        "socket" => ChardevBackendU::Socket(ChardevSocketWrapper { data: parse_socket(opts)? }),
        "null" => ChardevBackendU::Null(ChardevCommonWrapper { data: parse_common(opts) }),
        "file" => ChardevBackendU::File(ChardevFileWrapper { data: parse_file(opts)? }),
        "pipe" => ChardevBackendU::Pipe(ChardevHostdevWrapper { data: parse_pipe(opts)? }),
        "mux" => ChardevBackendU::Mux(ChardevMuxWrapper { data: parse_mux(opts)? }),
        "ringbuf" => ChardevBackendU::Ringbuf(ChardevRingbufWrapper { data: parse_ringbuf(opts) }),
        "memory" => ChardevBackendU::Memory(ChardevRingbufWrapper { data: parse_ringbuf(opts) }),
        "stdio" => {
            let c = parse_common(opts);
            ChardevBackendU::Stdio(ChardevStdioWrapper {
                data: ChardevStdio {
                    logfile: c.logfile,
                    logappend: c.logappend,
                    logtimestamp: c.logtimestamp,
                    signal: Some(opts.get_bool("signal", true)),
                },
            })
        }
        #[cfg(unix)]
        "pty" => {
            let c = parse_common(opts);
            ChardevBackendU::Pty(ChardevPtyWrapper {
                data: ChardevPty {
                    logfile: c.logfile,
                    logappend: c.logappend,
                    logtimestamp: c.logtimestamp,
                    path: opts.get("path").map(str::to_string),
                },
            })
        }
        _ => {
            return Err(Error::generic(format!(
                "chardev backend '{name}' is not supported by ruvm yet"
            )));
        }
    };
    Ok(ChardevBackend { u })
}

/// Up to `max` characters at the start of `s` that are not in `stops`, like `%32[^,]`, and
/// the rest. `None` when there are none.
fn scan_until<'a>(s: &'a str, stops: &[char], max: usize) -> Option<(&'a str, &'a str)> {
    let end = s
        .char_indices()
        .take(max)
        .find(|(_, c)| stops.contains(c))
        .map(|(i, _)| i)
        .unwrap_or_else(|| s.char_indices().nth(max).map_or(s.len(), |(i, _)| i));
    (end > 0).then(|| s.split_at(end))
}

/// Up to seven digits at the start of `s`, like `%7[0-9]`, and the rest.
fn scan_digits(s: &str) -> Option<(&str, &str)> {
    let end = s.bytes().take(7).take_while(u8::is_ascii_digit).count();
    (end > 0).then(|| s.split_at(end))
}

/// `%64[^:]:%32[^,]` with `:%32[^,]` as the fallback, as `qemu_chr_parse_compat()` reads a
/// host and port. Returns the host, the port and the rest.
fn host_port<'a>(p: &'a str, stops: &[char]) -> Option<(&'a str, &'a str, &'a str)> {
    let full = scan_until(p, &[':'], 64).and_then(|(host, rest)| {
        let (port, rest) = scan_until(rest.strip_prefix(':')?, stops, 32)?;
        Some((host, port, rest))
    });
    full.or_else(|| {
        let (port, rest) = scan_until(p.strip_prefix(':')?, stops, 32)?;
        Some(("", port, rest))
    })
}

/// The `vc:WxH` and `vc:WCxHC` sizes.
fn vc_size(opts: &mut QemuOpts, size: &str) -> Option<Result<()>> {
    let (w, rest) = scan_digits(size)?;
    if let Some(h) = rest.strip_prefix('x') {
        let (h, _) = scan_digits(h)?;
        return Some(opts.set("width", w).and_then(|()| opts.set("height", h)));
    }
    let (h, _) = scan_digits(rest.strip_prefix("Cx")?)?;
    Some(opts.set("cols", w).and_then(|()| opts.set("rows", h)))
}

/// Fails with `None` where QEMU gives up without saying why.
fn fill_compat(
    opts: &mut QemuOpts,
    filename: &str,
    permit_mux_mon: bool,
) -> std::result::Result<(), Option<Error>> {
    let orig = filename;
    let mut filename = filename;
    if let Some(p) = filename.strip_prefix("mon:") {
        if !permit_mux_mon {
            return Err(Some(Error::generic("mon: isn't supported in this context")));
        }
        filename = p;
        opts.set("mux", "on")?;
        if filename == "stdio" {
            // Ctrl+C goes to the guest when the monitor is muxed onto stdio this way.
            opts.set("signal", "off")?;
        }
    }

    if ["null", "pty", "msmouse", "wctablet", "braille", "testdev", "stdio"].contains(&filename) {
        return Ok(opts.set("backend", filename)?);
    }
    if let Some(p) = filename.strip_prefix("vc") {
        opts.set("backend", "vc")?;
        if let Some(size) = p.strip_prefix(':') {
            return vc_size(opts, size).ok_or(None)?.map_err(Some);
        }
        return Ok(());
    }
    if filename == "con:" {
        return Ok(opts.set("backend", "console")?);
    }
    if filename.starts_with("COM") {
        opts.set("backend", "serial")?;
        return Ok(opts.set("path", filename)?);
    }
    for (prefix, backend) in [("file:", "file"), ("pipe:", "pipe"), ("pty:", "pty")] {
        if let Some(p) = filename.strip_prefix(prefix) {
            opts.set("backend", backend)?;
            return Ok(opts.set("path", p)?);
        }
    }
    for prefix in ["tcp:", "telnet:", "tn3270:", "websocket:"] {
        let Some(p) = filename.strip_prefix(prefix) else { continue };
        let (host, port, rest) = host_port(p, &[',']).ok_or(None)?;
        opts.set("backend", "socket")?;
        opts.set("host", host)?;
        opts.set("port", port)?;
        if let Some(more) = rest.strip_prefix(',') {
            opts.do_parse(more, None)?;
        }
        match prefix {
            "telnet:" => opts.set("telnet", "on")?,
            "tn3270:" => opts.set("tn3270", "on")?,
            "websocket:" => opts.set("websocket", "on")?,
            _ => {}
        }
        return Ok(());
    }
    if let Some(p) = filename.strip_prefix("udp:") {
        opts.set("backend", "udp")?;
        let (host, port, rest) = host_port(p, &['@', ',']).ok_or(None)?;
        opts.set("host", host)?;
        opts.set("port", port)?;
        if let Some(local) = rest.strip_prefix('@') {
            let (host, port, _) = host_port(local, &[',']).ok_or(None)?;
            opts.set("localaddr", host)?;
            opts.set("localport", port)?;
        }
        return Ok(());
    }
    if let Some(p) = filename.strip_prefix("unix:") {
        opts.set("backend", "socket")?;
        return Ok(opts.do_parse(p, Some("path"))?);
    }
    if filename.starts_with("/dev/parport") || filename.starts_with("/dev/ppi") {
        opts.set("backend", "parallel")?;
        return Ok(opts.set("path", filename)?);
    }
    if filename.starts_with("/dev/") {
        opts.set("backend", "serial")?;
        return Ok(opts.set("path", filename)?);
    }
    Err(Some(Error::generic(format!("'{orig}' is not a valid char driver"))))
}

/// `qemu_chr_parse_compat()`: a `chardev` set named `label` for an old style string such as
/// `tcp:localhost:4444,server=on` or `mon:stdio`. On error the set is gone again. The error is
/// `None` for the strings QEMU refuses without a message of its own, where the caller's
/// "parse error" is all the user gets.
pub fn parse_compat(
    list: &mut QemuOptsList,
    label: &str,
    filename: &str,
    permit_mux_mon: bool,
) -> std::result::Result<OptsHandle, Option<Error>> {
    let handle = list.create(Some(label), true).map_err(Some)?.handle();
    let opts = list.get_mut(handle).expect("just made");
    match fill_compat(opts, filename, permit_mux_mon) {
        Ok(()) => Ok(handle),
        Err(e) => {
            list.del(handle);
            Err(e)
        }
    }
}

/// The old style strings `-nographic` gives the default serial port, monitor and parallel
/// port, as `qemu_create_default_devices()` picks them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NographicDefaults {
    pub serial: Option<&'static str>,
    pub monitor: Option<&'static str>,
    pub parallel: Option<&'static str>,
}

/// Which of the defaults are still wanted decides where they go: a serial port and a monitor
/// share stdio through a mux, and either one alone gets stdio for itself.
pub fn nographic_defaults(serial: bool, monitor: bool, parallel: bool) -> NographicDefaults {
    let mut d = NographicDefaults { parallel: parallel.then_some("null"), ..Default::default() };
    if serial && monitor {
        d.serial = Some("mon:stdio");
    } else {
        d.serial = serial.then_some("stdio");
        d.monitor = monitor.then_some("stdio");
    }
    d
}
