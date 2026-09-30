// SPDX-License-Identifier: GPL-2.0-or-later

//! The backends that are not sockets or the mux. Several tests are ports of QEMU's
//! tests/unit/test-char.c, and say which.

mod common;

use std::sync::Arc;

use common::{Rec, TempDir};
use ruvm_chardev::Chardevs;
use ruvm_chardev::opts::chardev_opts;
use ruvm_qapi::types::{
    ChardevBackend, ChardevBackendU, ChardevCommonWrapper, ChardevFile, ChardevFileWrapper,
    ChardevRingbuf, ChardevRingbufWrapper, DataFormat,
};

fn file(out: &str, in_: Option<&str>, append: Option<bool>) -> ChardevBackend {
    let data = ChardevFile {
        out: out.to_string(),
        in_: in_.map(str::to_string),
        append,
        ..Default::default()
    };
    ChardevBackend { u: ChardevBackendU::File(ChardevFileWrapper { data }) }
}

fn ringbuf(size: Option<i64>) -> ChardevBackend {
    let data = ChardevRingbuf { size, ..Default::default() };
    ChardevBackend { u: ChardevBackendU::Ringbuf(ChardevRingbufWrapper { data }) }
}

fn from_opts(chardevs: &Chardevs, params: &str) -> ruvm_base::Result<Arc<ruvm_chardev::Chardev>> {
    let mut list = chardev_opts();
    let opts = list.parse(params, true).unwrap();
    chardevs.new_from_opts(opts).map(Option::unwrap)
}

/// `char_null_test`.
#[test]
fn null() {
    let chardevs = Chardevs::new();
    assert!(chardevs.find("label-null").is_none());
    let chr = from_opts(&chardevs, "null,id=label-null").unwrap();
    assert!(chr.socket().is_none());
    assert_eq!(chr.typename(), "chardev-null");

    // One frontend at a time.
    let fe = chr.attach(Rec::new()).unwrap();
    assert!(chr.attach(Rec::new()).is_err());
    drop(fe);
    let fe = chr.attach(Rec::new()).unwrap();
    assert_eq!(fe.write(b"buf\0").unwrap(), 4);
    assert_eq!(fe.write_all(b"buf\0").unwrap(), 4);
}

/// `char_file_test`, and `append` both ways.
#[test]
fn file_output() {
    let dir = TempDir::new("file-out");
    let out = dir.path("out");
    let chardevs = Chardevs::new();
    let chr = chardevs.add("f", &file(&out, None, None)).unwrap();
    assert_eq!(chr.write_all(b"hello!").unwrap(), 6);
    assert_eq!(std::fs::read(&out).unwrap(), b"hello!");
    assert_eq!(chr.filename(), "file");
    assert_eq!(chr.typename(), "chardev-file");
    chardevs.remove("f").unwrap();
    drop(chr);

    // Without append the file starts over, with it the output goes at the end.
    let chr = chardevs.add("f", &file(&out, None, Some(false))).unwrap();
    chr.write_all(b"one").unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), b"one");
    chardevs.remove("f").unwrap();
    drop(chr);
    let chr = from_opts(&chardevs, &format!("file,id=f,path={out},append=on")).unwrap();
    chr.write_all(b"two").unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), b"onetwo");
}

#[test]
fn file_errors() {
    let chardevs = Chardevs::new();
    let e = chardevs.add("f", &file("/nonexistent/dir/out", None, None)).unwrap_err();
    #[cfg(unix)]
    assert_eq!(
        e.message(),
        "Failed to add chardev 'f': Could not open '/nonexistent/dir/out': No such file or directory"
    );
    #[cfg(windows)]
    assert_eq!(e.message(), "Failed to add chardev 'f': open /nonexistent/dir/out failed");

    let dir = TempDir::new("file-errors");
    let e = chardevs.add("f", &file(&dir.path("out"), Some(&dir.path("missing")), None));
    #[cfg(unix)]
    assert_eq!(
        e.unwrap_err().message(),
        format!(
            "Failed to add chardev 'f': Could not open '{}': No such file or directory",
            dir.path("missing")
        )
    );
    #[cfg(windows)]
    assert_eq!(e.unwrap_err().message(), "Failed to add chardev 'f': input file not supported");
}

/// A plain input file: the frontend reads it to the end and the connection ends there.
#[cfg(unix)]
#[test]
fn file_input() {
    let dir = TempDir::new("file-in");
    let input = dir.path("in");
    std::fs::write(&input, b"from the file").unwrap();
    let chardevs = Chardevs::new();
    let chr = chardevs.add("f", &file(&dir.path("out"), Some(&input), None)).unwrap();
    let rec = Rec::new();
    let _fe = chr.attach(rec.clone()).unwrap();
    common::wait_until(|| rec.closes.load(std::sync::atomic::Ordering::SeqCst) == 1);
    assert_eq!(rec.take(), b"from the file");
}

#[cfg(unix)]
fn mkfifo(path: &str) {
    let ok = std::process::Command::new("mkfifo").arg(path).status().unwrap().success();
    assert!(ok, "mkfifo {path}");
}

/// `char_file_fifo_test`: input from a fifo, and `chardev-send-break`.
#[cfg(unix)]
#[test]
fn file_fifo() {
    use std::io::Write;

    let dir = TempDir::new("file-fifo");
    let fifo = dir.path("fifo");
    mkfifo(&fifo);
    let mut w = std::fs::OpenOptions::new().read(true).write(true).open(&fifo).unwrap();
    assert_eq!(w.write(b"fifo-in\0").unwrap(), 8);

    let chardevs = Chardevs::new();
    let chr = chardevs.add("label-file", &file(&dir.path("out"), Some(&fifo), None)).unwrap();
    let rec = Rec::new();
    let _fe = chr.attach(rec.clone()).unwrap();

    assert_eq!(
        chardevs.send_break("label-foo").unwrap_err().message(),
        "Chardev 'label-foo' not found"
    );
    assert_ne!(rec.last_event(), Some(ruvm_chardev::ChrEvent::Break));
    chardevs.send_break("label-file").unwrap();
    assert_eq!(rec.last_event(), Some(ruvm_chardev::ChrEvent::Break));

    assert_eq!(rec.wait_bytes(8), b"fifo-in\0");
}

/// `char_pipe_test`: `pipe:path` with `path.in` and `path.out`.
#[cfg(unix)]
#[test]
fn pipe() {
    use std::io::{Read, Write};

    let dir = TempDir::new("pipe");
    let pipe = dir.path("pipe");
    let (inp, out) = (format!("{pipe}.in"), format!("{pipe}.out"));
    mkfifo(&inp);
    mkfifo(&out);

    let chardevs = Chardevs::new();
    let mut list = chardev_opts();
    let (chr, mux) =
        chardevs.new_from_name(&mut list, "pipe", &format!("pipe:{pipe}"), true).unwrap();
    assert!(!mux);
    assert_eq!(chr.typename(), "chardev-pipe");
    assert_eq!(chr.write(b"pipe-out\0").unwrap(), 9);
    let mut r = std::fs::OpenOptions::new().read(true).write(true).open(&out).unwrap();
    let mut buf = [0u8; 10];
    assert_eq!(r.read(&mut buf).unwrap(), 9);
    assert_eq!(&buf[..9], b"pipe-out\0");

    let mut w = std::fs::OpenOptions::new().write(true).open(&inp).unwrap();
    assert_eq!(w.write(b"pipe-in\0").unwrap(), 8);
    drop(w);
    let rec = Rec::new();
    let fe = chr.attach(rec.clone()).unwrap();
    assert_eq!(rec.wait_bytes(8), b"pipe-in\0");
    fe.join();
    chardevs.remove("pipe").unwrap();

    // Without the two fifos the path itself is used both ways.
    let e = chardevs.add(
        "p",
        &ruvm_chardev::opts::parse_opts(
            chardev_opts().parse(&format!("pipe,id=p,path={}", dir.path("none")), true).unwrap(),
        )
        .unwrap(),
    );
    assert_eq!(
        e.unwrap_err().message(),
        format!(
            "Failed to add chardev 'p': Could not open '{}': No such file or directory",
            dir.path("none")
        )
    );
}

/// `char_ringbuf_test`, and the rest of `ringbuf-read` and `ringbuf-write`.
#[test]
fn ringbuf_backend() {
    let chardevs = Chardevs::new();
    let e = from_opts(&chardevs, "ringbuf,id=ringbuf-label,size=5").unwrap_err();
    assert_eq!(e.message(), "size of ringbuf chardev must be power of two");
    assert!(chardevs.find("ringbuf-label").is_none());

    let chr = from_opts(&chardevs, "ringbuf,id=ringbuf-label,size=2").unwrap();
    let fe = chr.attach(Rec::new()).unwrap();
    assert_eq!(fe.write(b"buff").unwrap(), 4);
    assert_eq!(chardevs.ringbuf_read("ringbuf-label", 4, None).unwrap(), "ff");
    assert_eq!(chardevs.ringbuf_read("ringbuf-label", 4, None).unwrap(), "");
    drop(fe);

    // The old name.
    let chr = from_opts(&chardevs, "memory,id=memory-label,size=2").unwrap();
    assert_eq!(chr.typename(), "chardev-memory");
    assert_eq!(chr.filename(), "memory");

    let chardevs = Chardevs::new();
    chardevs.add("r", &ringbuf(None)).unwrap();
    chardevs
        .add("n", &ChardevBackend { u: ChardevBackendU::Null(ChardevCommonWrapper::default()) })
        .unwrap();
    chardevs.ringbuf_write("r", "aGVsbG8=", Some(DataFormat::Base64)).unwrap();
    chardevs.ringbuf_write("r", " world", Some(DataFormat::Utf8)).unwrap();
    assert_eq!(chardevs.ringbuf_read("r", 3, None).unwrap(), "hel");
    assert_eq!(chardevs.ringbuf_read("r", 100, Some(DataFormat::Base64)).unwrap(), "bG8gd29ybGQ=");
    // The UTF-8 form stops at a NUL, as the C string in QEMU does.
    chardevs.ringbuf_write("r", "AGI=", Some(DataFormat::Base64)).unwrap();
    assert_eq!(chardevs.ringbuf_read("r", 100, None).unwrap(), "");

    let err = |r: ruvm_base::Result<_>| r.map(|_: ()| ()).unwrap_err().message().to_string();
    assert_eq!(err(chardevs.ringbuf_write("x", "", None)), "Device 'x' not found");
    assert_eq!(err(chardevs.ringbuf_write("n", "", None)), "n is not a ringbuf device");
    assert_eq!(
        err(chardevs.ringbuf_write("r", "a*b", Some(DataFormat::Base64))),
        "Base64 data contains invalid characters"
    );
    let e = chardevs.ringbuf_read("r", 0, None).unwrap_err();
    assert_eq!(e.message(), "size must be greater than zero");
    let e = chardevs.add("z", &ringbuf(Some(0))).unwrap_err();
    assert_eq!(
        e.message(),
        "Failed to add chardev 'z': size of ringbuf chardev must be power of two"
    );

    // What the frontend gets with `be_write`.
    let chr = chardevs.find("r").unwrap();
    let rec = Rec::new();
    let _fe = chr.attach(rec.clone()).unwrap();
    chr.be_write(b"input");
    assert_eq!(rec.wait_bytes(5), b"input");
}

#[test]
fn backends_list() {
    let chardevs = Chardevs::new();
    for name in ["file", "memory", "mux", "null", "ringbuf", "socket", "stdio"] {
        assert!(ruvm_chardev::BACKENDS.contains(&name), "{name}");
    }
    #[cfg(unix)]
    for name in ["pipe", "pty"] {
        assert!(ruvm_chardev::BACKENDS.contains(&name), "{name}");
    }
    assert!(chardevs.query().is_empty());
}

#[cfg(unix)]
mod unix {
    use std::fs::File;
    use std::io::{Read, Write};
    use std::os::fd::OwnedFd;

    use rustix::fs::{Mode, OFlags, open};
    use rustix::termios::{InputModes, LocalModes, OptionalActions, tcgetattr, tcsetattr};
    use ruvm_chardev::Chardevs;
    use ruvm_chardev::pty::{openpty_raw, redirected_message};
    use ruvm_chardev::stdio::Terminal;
    use ruvm_qapi::types::{ChardevBackend, ChardevBackendU, ChardevPty, ChardevPtyWrapper};

    use super::common::{Rec, TempDir, settle, wait_until};

    fn pty(path: Option<&str>) -> ChardevBackend {
        let data = ChardevPty { path: path.map(str::to_string), ..Default::default() };
        ChardevBackend { u: ChardevBackendU::Pty(ChardevPtyWrapper { data }) }
    }

    fn open_tty(name: &str) -> OwnedFd {
        open(name, OFlags::RDWR | OFlags::NOCTTY, Mode::empty()).unwrap()
    }

    /// Opens the slave the way a terminal program would, in raw mode. Some systems forget
    /// the mode QEMU set once nobody has the slave open.
    fn open_raw(name: &str) -> File {
        let fd = open_tty(name);
        let mut t = tcgetattr(&fd).unwrap();
        t.make_raw();
        tcsetattr(&fd, OptionalActions::Now, &t).unwrap();
        File::from(fd)
    }

    /// Reads from `f` until `n` bytes came.
    fn read_n(f: &mut File, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        f.read_exact(&mut out).unwrap();
        out
    }

    #[test]
    fn pty_backend() {
        let dir = TempDir::new("pty");
        let link = dir.path("link");
        let chardevs = Chardevs::new();
        let chr = chardevs.add("serial0", &pty(Some(&link))).unwrap();
        let name = chr.chardev_return().pty.unwrap();
        assert!(name.starts_with("/dev/"), "{name}");
        assert_eq!(chr.pty_name().as_deref(), Some(name.as_str()));
        assert_eq!(chr.filename(), format!("pty:{name}"));
        assert_eq!(std::fs::read_link(&link).unwrap().to_str().unwrap(), name);
        assert_eq!(
            redirected_message(&name, "serial0"),
            format!("char device redirected to {name} (label serial0)")
        );

        // Nobody has the slave open yet, so output is dropped.
        assert_eq!(chr.write(b"lost").unwrap(), 4);
        let rec = Rec::new();
        let _fe = chr.attach(rec.clone()).unwrap();
        settle();
        assert!(!chr.pty_connected());

        let mut slave = open_raw(&name);
        wait_until(|| chr.pty_connected());
        slave.write_all(b"typed").unwrap();
        assert_eq!(rec.wait_bytes(5), b"typed");
        chr.write_all(b"shown").unwrap();
        assert_eq!(read_n(&mut slave, 5), b"shown");

        // Closing the slave ends the connection, opening it again starts another.
        drop(slave);
        wait_until(|| rec.closes.load(std::sync::atomic::Ordering::SeqCst) == 1);
        let mut slave = open_raw(&name);
        slave.write_all(b"again").unwrap();
        assert_eq!(rec.wait_bytes(5), b"again");
        // A hangup can show a moment late, which may make one more short connection.
        assert!(rec.opens.load(std::sync::atomic::Ordering::SeqCst) >= 2);

        drop(_fe);
        chardevs.remove("serial0").unwrap();
        assert!(std::fs::symlink_metadata(&link).is_err());

        let e = chardevs.add("p2", &pty(Some(&dir.path("no/such/dir")))).unwrap_err();
        assert!(
            e.message().starts_with("Failed to add chardev 'p2': Failed to create PTY symlink: "),
            "{}",
            e.message()
        );
    }

    /// The terminal handling of `stdio`, on a pty instead of the real terminal.
    #[test]
    fn terminal_modes() {
        let (_master, name) = openpty_raw().unwrap();
        let slave = open_tty(&name);
        let mut cooked = tcgetattr(&slave).unwrap();
        cooked.local_modes |= LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG;
        cooked.input_modes |= InputModes::ICRNL;
        tcsetattr(&slave, OptionalActions::Now, &cooked).unwrap();

        let t = Terminal::new(slave.try_clone().unwrap(), true);
        let now = tcgetattr(&slave).unwrap();
        assert!(!now.local_modes.intersects(LocalModes::ICANON | LocalModes::ECHO));
        assert!(!now.input_modes.contains(InputModes::ICRNL));
        assert!(now.local_modes.contains(LocalModes::ISIG));
        assert!(!t.echo());

        t.set_echo(true);
        let now = tcgetattr(&slave).unwrap();
        assert!(now.local_modes.contains(LocalModes::ICANON | LocalModes::ECHO));
        assert!(t.echo());
        drop(t);

        // signal=off keeps Ctrl-C from the host, echo or not.
        let t = Terminal::new(slave.try_clone().unwrap(), false);
        assert!(!tcgetattr(&slave).unwrap().local_modes.contains(LocalModes::ISIG));
        t.set_echo(true);
        assert!(!tcgetattr(&slave).unwrap().local_modes.contains(LocalModes::ISIG));
        drop(t);
        let back = tcgetattr(&slave).unwrap();
        assert!(
            back.local_modes.contains(LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG)
        );
        assert!(back.input_modes.contains(InputModes::ICRNL));
    }

    /// `char_stdio_test`: the child writes through a stdio chardev, here with a pty for its
    /// standard input so the terminal handling runs too.
    #[test]
    fn stdio_backend() {
        use std::process::{Command, Stdio};

        let (master, name) = openpty_raw().unwrap();
        let slave = open_tty(&name);
        let mut cooked = tcgetattr(&slave).unwrap();
        cooked.local_modes |= LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG;
        tcsetattr(&slave, OptionalActions::Now, &cooked).unwrap();

        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["unix::stdio_child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            .env("RUVM_CHARDEV_STDIO_CHILD", "1")
            .stdin(Stdio::from(File::from(slave.try_clone().unwrap())))
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut out = child.stdout.take().unwrap();
        let mut seen = Vec::new();
        let mut buf = [0u8; 256];
        while !String::from_utf8_lossy(&seen).contains("ready") {
            let n = out.read(&mut buf).unwrap();
            assert_ne!(n, 0, "child ended early: {}", String::from_utf8_lossy(&seen));
            seen.extend_from_slice(&buf[..n]);
        }
        assert!(String::from_utf8_lossy(&seen).contains("buf"));
        let raw = tcgetattr(&slave).unwrap();
        assert!(!raw.local_modes.intersects(LocalModes::ICANON | LocalModes::ECHO));
        // signal=off came with the backend options.
        assert!(!raw.local_modes.contains(LocalModes::ISIG));

        File::from(master.try_clone().unwrap()).write_all(b"q").unwrap();
        out.read_to_end(&mut seen).unwrap();
        assert!(child.wait().unwrap().success());
        assert!(
            String::from_utf8_lossy(&seen).contains("got q"),
            "{}",
            String::from_utf8_lossy(&seen)
        );
        let back = tcgetattr(&slave).unwrap();
        assert!(
            back.local_modes.contains(LocalModes::ICANON | LocalModes::ECHO | LocalModes::ISIG)
        );
        drop(master);
    }

    #[test]
    #[ignore = "run by stdio_backend in a child process"]
    fn stdio_child() {
        if std::env::var_os("RUVM_CHARDEV_STDIO_CHILD").is_none() {
            return;
        }
        let chardevs = Chardevs::new();
        let mut list = ruvm_chardev::opts::chardev_opts();
        let opts = list.parse("stdio,id=s,signal=off", true).unwrap();
        let chr = chardevs.new_from_opts(opts).unwrap().unwrap();
        let mut list = ruvm_chardev::opts::chardev_opts();
        let opts = list.parse("stdio,id=t", true).unwrap();
        let e = chardevs.new_from_opts(opts).unwrap_err();
        assert_eq!(e.message(), "cannot use stdio by multiple character devices");

        let rec = Rec::new();
        let fe = chr.attach(rec.clone()).unwrap();
        assert_eq!(fe.write_all(b"buf\0").unwrap(), 4);
        fe.write_all(b"ready\n").unwrap();
        let got = rec.wait_bytes(1);
        fe.write_all(format!("got {}\n", String::from_utf8_lossy(&got)).as_bytes()).unwrap();
        drop(fe);
        chardevs.remove("s").unwrap();
    }
}
