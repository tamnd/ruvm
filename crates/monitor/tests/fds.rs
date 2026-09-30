// SPDX-License-Identifier: GPL-2.0-or-later

//! Descriptor passing over a real Unix socket, the way libvirt hands QEMU its tap devices and
//! image files, checked against the replies of monitor/fds.c.

#![cfg(unix)]

use std::io::{BufRead, BufReader, IoSlice, Lines, Write};
use std::mem::MaybeUninit;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::net::UnixStream;
use std::sync::Arc;
use std::thread::JoinHandle;

use rustix::net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags, sendmsg};
use ruvm_monitor::{MonitorQmp, Qmp};
use ruvm_qapi::{QDict, QValue, json};

struct Client {
    qmp: Arc<Qmp>,
    mon: Arc<MonitorQmp>,
    stream: UnixStream,
    lines: Lines<BufReader<UnixStream>>,
    io: Option<JoinHandle<std::io::Result<()>>>,
    dispatcher: Option<JoinHandle<()>>,
}

impl Client {
    fn connect(qmp: &Arc<Qmp>) -> Client {
        let mon = qmp.add_monitor("mon0", false, true);
        let (stream, server) = UnixStream::pair().unwrap();
        let dispatcher = {
            let qmp = qmp.clone();
            std::thread::spawn(move || qmp.run_dispatcher())
        };
        let io = {
            let mon = mon.clone();
            std::thread::spawn(move || mon.serve_unix(server))
        };
        let lines = BufReader::new(stream.try_clone().unwrap()).lines();
        let mut c = Client {
            qmp: qmp.clone(),
            mon,
            stream,
            lines,
            io: Some(io),
            dispatcher: Some(dispatcher),
        };
        c.lines.next().unwrap().unwrap();
        assert_eq!(c.cmd(r#"{"execute": "qmp_capabilities"}"#, &[]), j(r#"{"return": {}}"#));
        c
    }

    /// Sends one command, with descriptors riding on its bytes, and reads the reply.
    fn cmd(&mut self, text: &str, fds: &[BorrowedFd<'_>]) -> QValue {
        if fds.is_empty() {
            self.stream.write_all(text.as_bytes()).unwrap();
        } else {
            let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(4))];
            let mut control = SendAncillaryBuffer::new(&mut space);
            assert!(control.push(SendAncillaryMessage::ScmRights(fds)));
            let n = sendmsg(
                &self.stream,
                &[IoSlice::new(text.as_bytes())],
                &mut control,
                SendFlags::empty(),
            )
            .unwrap();
            assert_eq!(n, text.len());
        }
        json::from_str(&self.lines.next().unwrap().unwrap()).unwrap()
    }

    fn close(mut self) {
        self.stream.shutdown(std::net::Shutdown::Both).unwrap();
        self.io.take().unwrap().join().unwrap().unwrap();
        self.qmp.remove_monitor(&self.mon);
        self.qmp.shutdown();
        self.dispatcher.take().unwrap().join().unwrap();
    }
}

fn j(text: &str) -> QValue {
    json::from_str(text).unwrap()
}

fn error(desc: &str) -> QValue {
    let e = QDict::new().with("class", "GenericError").with("desc", desc);
    QValue::Dict(QDict::new().with("error", e))
}

fn file() -> std::fs::File {
    std::fs::File::open("/dev/null").unwrap()
}

fn ret(v: &QValue) -> &QValue {
    let QValue::Dict(d) = v else { panic!("{}", v.to_json()) };
    d.get("return").unwrap_or_else(|| panic!("{}", v.to_json()))
}

#[test]
fn getfd_and_closefd() {
    let qmp = Qmp::new();
    let mut c = Client::connect(&qmp);
    assert_eq!(
        c.cmd(r#"{"execute": "getfd", "arguments": {"fdname": "tap0"}}"#, &[]),
        error("No file descriptor supplied via SCM_RIGHTS")
    );
    let f = file();
    assert_eq!(
        c.cmd(r#"{"execute": "getfd", "arguments": {"fdname": "tap0"}}"#, &[f.as_fd()]),
        j(r#"{"return": {}}"#)
    );
    // The descriptor was used up by the first getfd, so a second one has nothing to take.
    assert_eq!(
        c.cmd(r#"{"execute": "getfd", "arguments": {"fdname": "tap1"}}"#, &[]),
        error("No file descriptor supplied via SCM_RIGHTS")
    );
    assert_eq!(
        c.cmd(r#"{"execute": "getfd", "arguments": {"fdname": "0tap"}}"#, &[f.as_fd()]),
        error("Parameter 'fdname' expects a name not starting with a digit")
    );
    assert_eq!(c.mon.named_fds().len(), 1);
    assert!(c.mon.named_fds().take("tap0").is_ok());
    assert_eq!(
        c.cmd(r#"{"execute": "closefd", "arguments": {"fdname": "tap0"}}"#, &[]),
        error("File descriptor named 'tap0' not found")
    );
    c.close();
}

#[test]
fn fd_sets() {
    let qmp = Qmp::new();
    let mut c = Client::connect(&qmp);
    let (a, b) = (file(), file());
    let r = c.cmd(r#"{"execute": "add-fd", "arguments": {"opaque": "rdonly"}}"#, &[a.as_fd()]);
    let QValue::Dict(info) = ret(&r).clone() else { panic!() };
    assert_eq!(info.get("fdset-id"), Some(&j("0")));
    let fd0 = info.get("fd").cloned().unwrap();
    let r = c.cmd(r#"{"execute": "add-fd", "arguments": {"fdset-id": 0}}"#, &[b.as_fd()]);
    let QValue::Dict(info) = ret(&r).clone() else { panic!() };
    let fd1 = info.get("fd").cloned().unwrap();
    assert_ne!(fd0, fd1);

    let r = c.cmd(r#"{"execute": "query-fdsets"}"#, &[]);
    let want = format!(
        r#"[{{"fdset-id": 0, "fds": [{{"fd": {}, "opaque": "rdonly"}}, {{"fd": {}}}]}}]"#,
        fd0.to_json(),
        fd1.to_json()
    );
    assert_eq!(ret(&r), &j(&want));

    let rm = format!(
        r#"{{"execute": "remove-fd", "arguments": {{"fdset-id": 0, "fd": {}}}}}"#,
        fd0.to_json()
    );
    assert_eq!(c.cmd(&rm, &[]), j(r#"{"return": {}}"#));
    assert_eq!(
        c.cmd(&rm, &[]),
        error(&format!("File descriptor named 'fdset-id:0, fd:{}' not found", fd0.to_json()))
    );
    assert_eq!(
        c.cmd(r#"{"execute": "remove-fd", "arguments": {"fdset-id": 0}}"#, &[]),
        j(r#"{"return": {}}"#)
    );
    assert_eq!(ret(&c.cmd(r#"{"execute": "query-fdsets"}"#, &[])), &j("[]"));
    assert_eq!(
        c.cmd(r#"{"execute": "add-fd"}"#, &[]),
        error("No file descriptor supplied via SCM_RIGHTS")
    );
    c.close();
}

#[test]
fn sets_outlive_the_monitor() {
    let qmp = Qmp::new();
    let mut c = Client::connect(&qmp);
    let f = file();
    c.cmd(r#"{"execute": "add-fd", "arguments": {"fdset-id": 3}}"#, &[f.as_fd()]);
    c.close();
    // The members belong to the client and stay after it hangs up, for the next connection.
    assert_eq!(qmp.fdsets().len(), 1);
}
