// SPDX-License-Identifier: GPL-2.0-or-later

//! The system emulator started the way QEMU's tests start it: errors from startup, the help
//! texts, and a machine driven over QMP and qtest sockets like libqtest does.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn ruvm() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ruvm"))
}

/// Runs `qemu-system-x86_64` with `args`, returning the exit code, stdout and stderr.
fn system(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(ruvm()).arg("qemu-system-x86_64").args(args).output().unwrap();
    let text = |b: Vec<u8>| String::from_utf8(b).unwrap();
    (out.status.code().unwrap_or(-1), text(out.stdout), text(out.stderr))
}

#[test]
fn startup_errors() {
    let p = "qemu-system-x86_64: ";
    let hint = "Use -machine help to list supported machines\n";
    for (args, want) in [
        (&[][..], format!("{p}No machine specified, and there is no default\n{hint}")),
        (&["-machine", "pc"], format!("{p}unsupported machine type: \"pc\"\n{hint}")),
        (
            &["-machine", "none"],
            format!("{p}No accelerator selected and no default accelerator available\n"),
        ),
        (&["-machine", "none", "-accel", "kvm"], format!("{p}invalid accelerator kvm\n")),
        (&["-m", "512"], format!("{p}-m 512: this option is not supported by ruvm yet\n")),
        (&["-qmp", "tcp:nope"], format!("{p}-qmp tcp:nope: parse error: tcp:nope\n")),
        (
            &["-display", "gtk"],
            format!("{p}-display gtk: Parameter 'type' does not accept value 'gtk'\n"),
        ),
        (
            &["-machine", "none", "-accel", "qtest", "-object", "nope,id=x"],
            format!("{p}-object nope,id=x: Parameter 'qom-type' does not accept value 'nope'\n"),
        ),
    ] {
        let (code, _, err) = system(args);
        assert_eq!((code, err), (1, want), "{args:?}");
    }

    let (code, _, err) = system(&["-machine", "none", "-accel", "qtest", "-mon", "chardev=c"]);
    assert_eq!(code, 1);
    assert_eq!(
        err,
        "qemu-system-x86_64: -mon chardev=c: warning: '-mon' is deprecated, use '-object' with 'monitor-hmp' or 'monitor-qmp' types instead\nqemu-system-x86_64: -mon chardev=c: chardev \"c\" not found\n"
    );
}

#[test]
fn help_options() {
    let (code, out, _) = system(&["-machine", "help"]);
    assert_eq!(
        (code, out.as_str()),
        (0, "Supported machines are:\nnone                 empty machine\n")
    );
    let (code, out, _) = system(&["-accel", "help"]);
    assert_eq!((code, out.as_str()), (0, "Accelerators supported in QEMU binary:\n"));
    let (code, out, _) = system(&["-object", "help"]);
    assert_eq!(
        (code, out.as_str()),
        (
            0,
            "List of user creatable objects:\n  memory-backend-ram\n  monitor-hmp\n  monitor-qmp\n  qtest\n"
        )
    );
    let (code, out, _) = system(&["-display", "help"]);
    assert_eq!(code, 0);
    assert!(out.starts_with("Available display backend types:\nnone\n\n"), "{out}");
    let (code, out, _) = system(&["-audio", "help"]);
    assert_eq!((code, out.as_str()), (0, "Available audio drivers:\nnone\nwav\n"));
}

#[cfg(unix)]
mod sockets {
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::time::Duration;

    use super::*;

    /// Kills the machine if a test fails before it quits.
    struct Machine(std::process::Child);

    impl Drop for Machine {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    struct Qmp {
        reader: BufReader<UnixStream>,
        writer: UnixStream,
        events: Vec<String>,
    }

    impl Qmp {
        fn line(&mut self) -> String {
            let mut l = String::new();
            self.reader.read_line(&mut l).unwrap();
            l
        }

        /// Sends a command and returns its reply, keeping the events that came first.
        fn cmd(&mut self, c: &str) -> String {
            writeln!(self.writer, "{c}").unwrap();
            loop {
                let l = self.line();
                if l.contains("\"event\"") {
                    self.events.push(l);
                } else {
                    return l.trim_end().to_string();
                }
            }
        }
    }

    #[test]
    fn a_libqtest_session() {
        let dir = std::env::temp_dir().join(format!("ruvm-system-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (qs, ms) = (dir.join("qtest.sock"), dir.join("qmp.sock"));
        let _ = std::fs::remove_file(&qs);
        let _ = std::fs::remove_file(&ms);
        let lq = UnixListener::bind(&qs).unwrap();
        let lm = UnixListener::bind(&ms).unwrap();

        // What qtest_qemu_args() passes, with -machine none from the test.
        let child = Command::new(ruvm())
            .arg("qemu-system-x86_64")
            .args(["-qtest", &format!("unix:{}", qs.display()), "-qtest-log", "/dev/null"])
            .args(["-chardev", &format!("socket,path={},id=char0", ms.display())])
            .args(["-object", "monitor-qmp,id=qmp0,chardev=char0"])
            .args(["-display", "none", "-audio", "none", "-machine", "none", "-accel", "qtest"])
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut child = Machine(child);
        let (mut qtest, _) = lq.accept().unwrap();
        let (m, _) = lm.accept().unwrap();
        m.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut qmp =
            Qmp { reader: BufReader::new(m.try_clone().unwrap()), writer: m, events: Vec::new() };

        assert!(qmp.line().starts_with("{\"QMP\": {\"version\": {\"qemu\": {\"micro\": 0,"));
        assert_eq!(qmp.cmd(r#"{"execute": "qmp_capabilities"}"#), r#"{"return": {}}"#);
        let running = r#"{"return": {"status": "running", "running": true}}"#;
        assert_eq!(qmp.cmd(r#"{"execute": "query-status"}"#), running);
        assert_eq!(qmp.cmd(r#"{"execute": "stop"}"#), r#"{"return": {}}"#);
        assert_eq!(
            qmp.cmd(r#"{"execute": "query-status"}"#),
            r#"{"return": {"status": "paused", "running": false}}"#
        );
        assert_eq!(qmp.cmd(r#"{"execute": "cont"}"#), r#"{"return": {}}"#);
        let events: Vec<bool> = ["\"STOP\"", "\"RESUME\""]
            .iter()
            .zip(&qmp.events)
            .map(|(name, e)| e.contains(name))
            .collect();
        assert_eq!(events, [true, true]);
        assert_eq!(
            qmp.cmd(r#"{"execute": "x-exit-preconfig"}"#),
            r#"{"error": {"class": "GenericError", "desc": "The command is permitted only before machine initialization"}}"#
        );

        qtest.write_all(b"readb 0x1000\ninl 0x60\nclock_step 100\nclock_step\n").unwrap();
        // Machine none has no timers, so there is no deadline to step to.
        let want = "OK 0x0000000000000000\nOK 0xffffffff\nOK 100\nFAIL cannot advance clock to the next deadline because there is no pending deadline\n";
        let mut got = vec![0; want.len()];
        qtest.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        qtest.read_exact(&mut got).unwrap();
        assert_eq!(String::from_utf8(got).unwrap(), want);

        assert_eq!(qmp.cmd(r#"{"execute": "quit"}"#), r#"{"return": {}}"#);
        let last = qmp.line();
        let shutdown =
            r#""event": "SHUTDOWN", "data": {"guest": false, "reason": "host-qmp-quit"}"#;
        assert!(last.contains(shutdown), "{last}");
        let status = child.0.wait().unwrap();
        assert!(status.success());
        let mut err = String::new();
        child.0.stderr.take().unwrap().read_to_string(&mut err).unwrap();
        assert_eq!(err, "");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_signal_ends_the_machine() {
        let dir = std::env::temp_dir().join(format!("ruvm-signal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ms = dir.join("qmp.sock");
        let _ = std::fs::remove_file(&ms);
        let child = Command::new(ruvm())
            .arg("qemu-system-x86_64")
            .args(["-machine", "none", "-accel", "qtest"])
            .args(["-qmp", &format!("unix:{},server=on,wait=off", ms.display())])
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut child = Machine(child);
        let m = loop {
            match UnixStream::connect(&ms) {
                Ok(m) => break m,
                Err(_) => std::thread::sleep(Duration::from_millis(20)),
            }
        };
        let mut qmp =
            Qmp { reader: BufReader::new(m.try_clone().unwrap()), writer: m, events: Vec::new() };
        qmp.line();
        assert_eq!(qmp.cmd(r#"{"execute": "qmp_capabilities"}"#), r#"{"return": {}}"#);
        Command::new("kill").args(["-TERM", &child.0.id().to_string()]).status().unwrap();
        let last = qmp.line();
        assert!(last.contains(r#""data": {"guest": false, "reason": "host-signal"}"#), "{last}");
        assert!(child.0.wait().unwrap().success());
        let mut err = String::new();
        child.0.stderr.take().unwrap().read_to_string(&mut err).unwrap();
        assert!(err.starts_with("qemu-system-x86_64: terminating on signal 15 from pid "), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
