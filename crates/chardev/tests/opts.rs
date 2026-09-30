// SPDX-License-Identifier: GPL-2.0-or-later

use ruvm_chardev::Chardevs;
use ruvm_chardev::opts::{chardev_opts, parse_compat, parse_opts};
use ruvm_qapi::opts::QemuOptsList;
use ruvm_qapi::types::{ChardevBackend, ChardevBackendU, ChardevSocket, SocketAddressLegacyU};

fn parse(params: &str) -> Result<ChardevBackend, String> {
    let mut list = chardev_opts();
    let opts = list.parse(params, true).map_err(|e| e.message().to_string())?;
    parse_opts(opts).map_err(|e| e.message().to_string())
}

fn socket(params: &str) -> ChardevSocket {
    match parse(params).unwrap().u {
        ChardevBackendU::Socket(s) => s.data,
        _ => panic!("{params} is not a socket"),
    }
}

#[test]
fn socket_options() {
    let s = socket("socket,id=s,path=/tmp/ruvm.sock,server=on,wait=off");
    match &s.addr.u {
        SocketAddressLegacyU::Unix(u) => assert_eq!(u.data.path, "/tmp/ruvm.sock"),
        _ => panic!("not a unix address"),
    }
    assert_eq!((s.server, s.wait, s.nodelay, s.telnet), (Some(true), Some(false), None, None));
    assert_eq!((s.logappend, s.logtimestamp), (Some(false), Some(false)));

    // A server waits unless told not to, a client has no wait at all.
    assert_eq!(socket("socket,id=s,path=x,server=on").wait, Some(true));
    let s = socket("socket,id=s,host=localhost,port=4444,delay=off,reconnect-ms=500");
    assert_eq!((s.server, s.wait, s.nodelay), (Some(false), None, Some(true)));
    assert_eq!(s.reconnect_ms, Some(500));
    match &s.addr.u {
        SocketAddressLegacyU::Inet(i) => {
            assert_eq!((i.data.host.as_str(), i.data.port.as_str()), ("localhost", "4444"));
        }
        _ => panic!("not an inet address"),
    }
    assert!(matches!(socket("socket,id=s,fd=3").addr.u, SocketAddressLegacyU::Fd(_)));
}

#[test]
fn bad_options() {
    for (params, msg) in [
        ("id=a", "chardev: \"a\" missing backend"),
        ("nope,id=a", "'nope' is not a valid char driver name"),
        ("null,id=a,cols=80", "chardev 'a' does not support size options"),
        ("null,id=a,encoding=utf8", "chardev 'a' does not support encoding option"),
        ("file,id=a,path=x", "chardev backend 'file' is not supported by ruvm yet"),
        ("socket,id=a,path=x,host=y", "None or one of 'path', 'fd' or 'host' option required."),
        ("socket,id=a,host=y", "chardev: socket: no port given"),
        ("socket,id=a,path=x,delay=on,nodelay=on", "'delay' and 'nodelay' are mutually exclusive"),
    ] {
        assert_eq!(parse(params).unwrap_err(), msg, "{params}");
    }
}

fn compat(filename: &str) -> Result<Vec<(String, String)>, Option<String>> {
    let mut list = chardev_opts();
    let handle = parse_compat(&mut list, "c", filename, true)
        .map_err(|e| e.map(|e| e.message().to_string()))?;
    let opts = list.get(handle).unwrap();
    Ok(opts.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect())
}

fn pairs(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[test]
fn compat_strings() {
    for (filename, want) in [
        ("null", &[("backend", "null")][..]),
        ("mon:stdio", &[("mux", "on"), ("signal", "off"), ("backend", "stdio")]),
        ("vc:800x600", &[("backend", "vc"), ("width", "800"), ("height", "600")]),
        ("vc:80Cx24C", &[("backend", "vc"), ("cols", "80"), ("rows", "24")]),
        ("file:/tmp/log", &[("backend", "file"), ("path", "/tmp/log")]),
        (
            "tcp:localhost:4444,server=on,wait=off",
            &[
                ("backend", "socket"),
                ("host", "localhost"),
                ("port", "4444"),
                ("server", "on"),
                ("wait", "off"),
            ],
        ),
        ("telnet::23", &[("backend", "socket"), ("host", ""), ("port", "23"), ("telnet", "on")]),
        ("unix:/tmp/m,server=on", &[("backend", "socket"), ("path", "/tmp/m"), ("server", "on")]),
        (
            "udp:10.0.0.1:5000@:6000",
            &[
                ("backend", "udp"),
                ("host", "10.0.0.1"),
                ("port", "5000"),
                ("localaddr", ""),
                ("localport", "6000"),
            ],
        ),
        ("/dev/ttyS0", &[("backend", "serial"), ("path", "/dev/ttyS0")]),
        ("/dev/parport0", &[("backend", "parallel"), ("path", "/dev/parport0")]),
    ] {
        assert_eq!(compat(filename).unwrap(), pairs(want), "{filename}");
    }
    assert_eq!(compat("bogus").unwrap_err().unwrap(), "'bogus' is not a valid char driver");
    assert_eq!(compat("tcp:nope").unwrap_err(), None);
    assert_eq!(compat("vc:big").unwrap_err(), None);
    assert!(compat("unix:/tmp/m,frob=on").unwrap_err().is_some());

    let mut list = chardev_opts();
    assert!(parse_compat(&mut list, "c", "stdio", false).is_ok());
    let dup = parse_compat(&mut list, "c", "null", true).unwrap_err().unwrap();
    assert_eq!(dup.message(), "Duplicate ID 'c' for chardev");
    let e = parse_compat(&mut list, "d", "mon:stdio", false).unwrap_err().unwrap();
    assert_eq!(e.message(), "mon: isn't supported in this context");
    assert!(list.find(Some("d")).is_none());
}

#[test]
fn new_from_opts() {
    let chardevs = Chardevs::new();
    let mut list: QemuOptsList = chardev_opts();
    let opts = list.parse("null,id=n0", true).unwrap();
    let chr = chardevs.new_from_opts(opts).unwrap().unwrap();
    assert_eq!(chr.label(), "n0");
    assert_eq!(chardevs.query().len(), 1);

    let opts = list.parse("null", true).unwrap();
    assert_eq!(chardevs.new_from_opts(opts).unwrap_err().message(), "chardev: no id specified");
    let opts = list.parse("null,id=m,mux=on", true).unwrap();
    let e = chardevs.new_from_opts(opts).unwrap_err();
    assert_eq!(e.message(), "chardev backend 'mux' is not supported by ruvm yet");
}
