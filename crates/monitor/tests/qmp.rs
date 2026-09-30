// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP server against the behavior of monitor/qmp.c, driven the way a chardev drives it.

use std::io::Write;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_monitor::{MonitorQmp, QMP_REQ_QUEUE_LEN_MAX, Qmp};
use ruvm_qapi::commands::register_migrate_recover;
use ruvm_qapi::events::{event_rtc_change, event_stop};
use ruvm_qapi::types::{MigrateRecoverArg, RtcChangeArg};
use ruvm_qapi::visit::{CompatPolicy, CompatPolicyOutput};
use ruvm_qapi::{QMP_SCHEMA_JSON, QValue, json};

#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Sink {
    /// The lines written since the last call.
    fn take(&self) -> Vec<String> {
        let bytes = std::mem::take(&mut *self.0.lock().unwrap());
        String::from_utf8(bytes).unwrap().lines().map(str::to_string).collect()
    }

    fn take_values(&self) -> Vec<QValue> {
        self.take().iter().map(|l| json::from_str(l).unwrap()).collect()
    }
}

fn setup(oob: bool) -> (Arc<Qmp>, Arc<MonitorQmp>, Sink) {
    let qmp = Qmp::new();
    qmp.register(|c| register_migrate_recover(c, |_: &MonitorQmp, _: MigrateRecoverArg| Ok(())));
    let mon = qmp.add_monitor("mon0", false, oob);
    let sink = Sink::default();
    mon.set_output(Some(Box::new(sink.clone())));
    mon.open();
    (qmp, mon, sink)
}

fn j(text: &str) -> QValue {
    json::from_str(text).unwrap()
}

/// Feeds all of `text` and runs the dispatcher, the way the main loop and the I/O thread
/// take turns.
fn send(qmp: &Arc<Qmp>, mon: &MonitorQmp, text: &str) {
    let mut rest = text.as_bytes();
    while !rest.is_empty() {
        let n = mon.feed(rest);
        rest = &rest[n..];
        qmp.dispatch_pending();
    }
}

fn negotiate(qmp: &Arc<Qmp>, mon: &MonitorQmp, sink: &Sink, oob: bool) {
    let req = if oob {
        r#"{"execute": "qmp_capabilities", "arguments": {"enable": ["oob"]}}"#
    } else {
        r#"{"execute": "qmp_capabilities"}"#
    };
    send(qmp, mon, req);
    assert_eq!(sink.take_values(), vec![j(r#"{"return": {}}"#)]);
}

#[test]
fn greeting() {
    let pkg = format!("ruvm {}", env!("CARGO_PKG_VERSION"));
    let (_, _, sink) = setup(true);
    let want = format!(
        r#"{{"QMP": {{"version": {{"qemu": {{"micro": 0, "minor": 1, "major": 11}}, "package": "{pkg}"}}, "capabilities": ["oob"]}}}}"#
    );
    assert_eq!(sink.take(), vec![want]);
    let (_, _, sink) = setup(false);
    assert_eq!(
        sink.take_values(),
        vec![j(&format!(
            r#"{{"QMP": {{"version": {{"qemu": {{"micro": 0, "minor": 1, "major": 11}}, "package": "{pkg}"}}, "capabilities": []}}}}"#
        ))]
    );
}

#[test]
fn capabilities_negotiation() {
    let (qmp, mon, sink) = setup(false);
    sink.take();
    send(&qmp, &mon, r#"{"execute": "query-version", "id": 1}"#);
    assert_eq!(
        sink.take_values(),
        vec![j(
            r#"{"error": {"class": "CommandNotFound", "desc": "Expecting capabilities negotiation with 'qmp_capabilities'"}, "id": 1}"#
        )]
    );
    send(&qmp, &mon, r#"{"execute": "qmp_capabilities", "arguments": {"enable": ["oob"]}}"#);
    assert_eq!(
        sink.take_values(),
        vec![j(r#"{"error": {"class": "GenericError", "desc": "Capability oob not available"}}"#)]
    );
    send(&qmp, &mon, r#"{"execute": "qmp_capabilities", "arguments": {"enable": ["nope"]}}"#);
    assert_eq!(
        sink.take_values(),
        vec![j(
            r#"{"error": {"class": "GenericError", "desc": "Parameter 'null' does not accept value 'nope'"}}"#
        )]
    );
    assert!(!mon.negotiated());
    negotiate(&qmp, &mon, &sink, false);
    assert!(mon.negotiated() && !mon.oob_enabled());
    send(&qmp, &mon, r#"{"execute": "qmp_capabilities"}"#);
    assert_eq!(
        sink.take_values(),
        vec![j(
            r#"{"error": {"class": "CommandNotFound", "desc": "Capabilities negotiation is already complete, command ignored"}}"#
        )]
    );
    // A new connection starts over.
    mon.close();
    mon.open();
    sink.take();
    assert!(!mon.negotiated());
}

#[test]
fn control_commands() {
    let (qmp, mon, sink) = setup(false);
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    send(&qmp, &mon, r#"{"execute": "query-commands"}"#);
    let rsp = sink.take_values();
    let QValue::Dict(d) = &rsp[0] else { panic!() };
    let Some(QValue::List(l)) = d.get("return") else { panic!() };
    let names: Vec<String> = l
        .iter()
        .map(|c| match c {
            QValue::Dict(c) => c.get_str("name").unwrap().to_string(),
            _ => panic!(),
        })
        .collect();
    // The fd passing commands only exist where SCM_RIGHTS does.
    let fd_cmds: &[&str] =
        if cfg!(unix) { &["query-fdsets", "remove-fd", "add-fd", "closefd", "getfd"] } else { &[] };
    let names: Vec<&str> =
        names.iter().map(String::as_str).filter(|n| !fd_cmds.contains(n)).collect();
    assert_eq!(
        names,
        [
            "migrate-recover",
            "query-qmp-schema",
            "query-commands",
            "query-version",
            "qmp_capabilities"
        ]
    );
    send(&qmp, &mon, r#"{"execute": "query-version"}"#);
    let rsp = sink.take_values();
    let QValue::Dict(d) = &rsp[0] else { panic!() };
    assert!(d.contains_key("return"));
}

#[test]
fn query_qmp_schema_is_byte_identical() {
    let (qmp, mon, sink) = setup(false);
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    send(&qmp, &mon, r#"{"execute": "query-qmp-schema"}"#);
    let lines = sink.take();
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0], format!(r#"{{"return": {QMP_SCHEMA_JSON}}}"#));
}

#[test]
fn query_qmp_schema_hides_deprecated() {
    let (qmp, mon, sink) = setup(false);
    qmp.set_policy(CompatPolicy {
        deprecated_output: CompatPolicyOutput::Hide,
        ..Default::default()
    });
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    send(&qmp, &mon, r#"{"execute": "query-qmp-schema"}"#);
    let text = sink.take().remove(0);
    assert!(text.len() < QMP_SCHEMA_JSON.len());
    // Entities and object members marked deprecated are gone. Enum members keep theirs, as
    // in QEMU.
    let deprecated = |v: &QValue| match v {
        QValue::Dict(d) => match d.get("features") {
            Some(QValue::List(f)) => f.contains(&QValue::str("deprecated")),
            _ => false,
        },
        _ => false,
    };
    let QValue::Dict(rsp) = j(&text) else { panic!() };
    let Some(QValue::List(ents)) = rsp.get("return") else { panic!() };
    for ent in ents {
        assert!(!deprecated(ent));
        let QValue::Dict(e) = ent else { panic!() };
        if e.get_str("meta-type") == Some("object") {
            let Some(QValue::List(members)) = e.get("members") else { panic!() };
            assert!(!members.iter().any(deprecated));
        }
    }
}

#[test]
fn without_oob_one_request_at_a_time() {
    let (qmp, mon, sink) = setup(false);
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    let first = r#"{"execute": "query-version", "id": 1}"#;
    let two = format!(r#"{first}{{"execute": "query-version", "id": 2}}"#);
    // The monitor suspends as soon as the first request is queued, so the rest waits.
    let n = mon.feed(two.as_bytes());
    assert_eq!(n, first.len());
    assert!(!mon.can_read());
    qmp.dispatch_pending();
    assert!(mon.can_read());
    assert_eq!(mon.feed(&two.as_bytes()[n..]), two.len() - n);
    qmp.dispatch_pending();
    let ids: Vec<QValue> = sink
        .take_values()
        .into_iter()
        .map(|v| match v {
            QValue::Dict(d) => d.get("id").cloned().unwrap(),
            _ => panic!(),
        })
        .collect();
    assert_eq!(ids, vec![j("1"), j("2")]);
}

#[test]
fn oob_queue_and_overtaking() {
    let (qmp, mon, sink) = setup(true);
    sink.take();
    negotiate(&qmp, &mon, &sink, true);
    assert!(mon.oob_enabled());
    let req = |i: usize| format!(r#"{{"execute": "query-version", "id": {i}}}"#);

    // Seven queued requests leave room for one more, so the monitor keeps reading, and an OOB
    // request overtakes all of them.
    for i in 0..QMP_REQ_QUEUE_LEN_MAX - 1 {
        mon.feed(req(i).as_bytes());
    }
    assert!(mon.can_read());
    let oob = r#"{"exec-oob": "migrate-recover", "arguments": {"uri": "tcp:x"}, "id": "oob"}"#;
    assert_eq!(mon.feed(oob.as_bytes()), oob.len());
    assert_eq!(sink.take_values(), vec![j(r#"{"return": {}, "id": "oob"}"#)]);

    // The eighth fills the queue and the monitor stops reading.
    let text = format!("{}{}", req(7), req(8));
    let n = mon.feed(text.as_bytes());
    assert_eq!(n, req(7).len());
    assert!(!mon.can_read());
    qmp.dispatch_pending();
    assert!(mon.can_read());
    assert_eq!(sink.take_values().len(), QMP_REQ_QUEUE_LEN_MAX);
    send(&qmp, &mon, &text[n..]);
    assert_eq!(sink.take_values().len(), 1);
}

#[test]
fn exec_oob_needs_negotiation() {
    let (qmp, mon, sink) = setup(true);
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    send(&qmp, &mon, r#"{"exec-oob": "migrate-recover", "arguments": {"uri": "x"}}"#);
    assert_eq!(
        sink.take_values(),
        vec![j(
            r#"{"error": {"class": "GenericError", "desc": "QMP input member 'exec-oob' is unexpected"}}"#
        )]
    );
}

#[test]
fn parse_errors_are_replies() {
    let (qmp, mon, sink) = setup(false);
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    send(&qmp, &mon, "{\"execute\": }\n{\"execute\": \"query-version\", \"id\": 5}");
    let rsp = sink.take_values();
    assert_eq!(rsp.len(), 2, "{rsp:?}");
    let QValue::Dict(e) = &rsp[0] else { panic!() };
    let Some(QValue::Dict(e)) = e.get("error") else { panic!("{:?}", rsp[0]) };
    assert_eq!(e.get_str("class"), Some("GenericError"));
    let QValue::Dict(ok) = &rsp[1] else { panic!() };
    assert_eq!(ok.get("id"), Some(&j("5")));
}

#[test]
fn events_reach_negotiated_monitors() {
    let (qmp, mon, sink) = setup(false);
    let other = qmp.add_monitor("mon1", false, false);
    let other_sink = Sink::default();
    other.set_output(Some(Box::new(other_sink.clone())));
    other.open();
    other_sink.take();
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    qmp.emit_event(event_stop(&qmp.policy()).unwrap());
    let ev = sink.take_values();
    assert_eq!(ev.len(), 1);
    let QValue::Dict(d) = &ev[0] else { panic!() };
    assert_eq!(d.get_str("event"), Some("STOP"));
    // Still negotiating, so no events.
    assert!(other_sink.take().is_empty());
}

#[test]
fn throttled_events() {
    let (qmp, mon, sink) = setup(false);
    let now = Arc::new(AtomicI64::new(0));
    let clock = now.clone();
    qmp.set_event_clock(move || clock.load(Ordering::SeqCst));
    sink.take();
    negotiate(&qmp, &mon, &sink, false);
    let rtc = |offset: i64| {
        event_rtc_change(
            &CompatPolicy::default(),
            RtcChangeArg { offset, qom_path: "/machine/rtc".into() },
        )
        .unwrap()
    };
    qmp.emit_event(rtc(1));
    qmp.emit_event(rtc(2));
    qmp.emit_event(rtc(3));
    assert_eq!(sink.take_values().len(), 1);
    now.store(999_999_999, Ordering::SeqCst);
    qmp.dispatch_pending();
    assert!(sink.take().is_empty());
    now.store(1_000_000_000, Ordering::SeqCst);
    qmp.dispatch_pending();
    let ev = sink.take_values();
    assert_eq!(ev.len(), 1);
    let QValue::Dict(d) = &ev[0] else { panic!() };
    assert_eq!(d.get("data"), Some(&j(r#"{"offset": 3, "qom-path": "/machine/rtc"}"#)));
}

#[cfg(unix)]
#[test]
fn serve_over_a_socket() {
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixStream;

    let qmp = Qmp::new();
    let mon = qmp.add_monitor("mon0", false, true);
    let (client, server) = UnixStream::pair().unwrap();
    let dispatcher = {
        let qmp = qmp.clone();
        std::thread::spawn(move || qmp.run_dispatcher())
    };
    let io = {
        let mon = mon.clone();
        let writer = server.try_clone().unwrap();
        std::thread::spawn(move || mon.serve(server, Box::new(writer)))
    };
    let mut lines = BufReader::new(client.try_clone().unwrap()).lines();
    let mut client = client;
    assert!(lines.next().unwrap().unwrap().starts_with(r#"{"QMP": "#));
    client
        .write_all(br#"{"execute": "qmp_capabilities", "arguments": {"enable": ["oob"]}}"#)
        .unwrap();
    assert_eq!(lines.next().unwrap().unwrap(), r#"{"return": {}}"#);
    for i in 0..20 {
        client
            .write_all(format!(r#"{{"execute": "query-version", "id": {i}}}"#).as_bytes())
            .unwrap();
    }
    for i in 0..20 {
        let rsp = json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        let QValue::Dict(d) = rsp else { panic!() };
        assert_eq!(d.get("id"), Some(&j(&i.to_string())));
    }
    client.shutdown(std::net::Shutdown::Both).unwrap();
    io.join().unwrap().unwrap();
    qmp.shutdown();
    dispatcher.join().unwrap();
}

#[test]
fn serve_through_a_tcp_chardev() {
    use std::io::{BufRead, BufReader};
    use std::net::TcpStream;

    use ruvm_chardev::Chardevs;
    use ruvm_qapi::types::{
        ChardevBackend, ChardevBackendU, ChardevSocket, ChardevSocketWrapper, InetSocketAddress,
        InetSocketAddressWrapper, SocketAddressLegacy, SocketAddressLegacyU,
    };

    let addr = SocketAddressLegacyU::Inet(InetSocketAddressWrapper {
        data: InetSocketAddress {
            host: "127.0.0.1".into(),
            port: "0".into(),
            ..Default::default()
        },
    });
    let data = ChardevSocket {
        addr: SocketAddressLegacy { u: addr },
        server: Some(true),
        wait: Some(false),
        ..Default::default()
    };
    let backend = ChardevBackend { u: ChardevBackendU::Socket(ChardevSocketWrapper { data }) };
    let chardevs = Chardevs::new();
    let chr = chardevs.add("qmp", &backend).unwrap();
    let name = chr.filename();
    let port: u16 =
        name.rsplit(':').next().unwrap().trim_end_matches(",server=on").parse().unwrap();

    let qmp = Qmp::new();
    let mon = qmp.add_monitor("mon0", false, true);
    let fe = chr.attach(mon.clone()).unwrap();
    let dispatcher = {
        let qmp = qmp.clone();
        std::thread::spawn(move || qmp.run_dispatcher())
    };
    // Each client gets its own greeting and has to negotiate again.
    for _ in 0..2 {
        let mut client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        let mut lines = BufReader::new(client.try_clone().unwrap()).lines();
        assert!(lines.next().unwrap().unwrap().starts_with(r#"{"QMP": "#));
        client.write_all(br#"{"execute": "query-version"}"#).unwrap();
        let rsp = lines.next().unwrap().unwrap();
        assert!(rsp.contains("CommandNotFound"), "{rsp}");
        client.write_all(br#"{"execute": "qmp_capabilities"}"#).unwrap();
        assert_eq!(lines.next().unwrap().unwrap(), r#"{"return": {}}"#);
        assert!(mon.negotiated());
    }
    fe.join();
    qmp.shutdown();
    dispatcher.join().unwrap();
}
