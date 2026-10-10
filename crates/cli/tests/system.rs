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
    system_for("x86_64", args)
}

/// Runs `qemu-system-<target>` with `args`.
fn system_for(target: &str, args: &[&str]) -> (i32, String, String) {
    let out =
        Command::new(ruvm()).arg(format!("qemu-system-{target}")).args(args).output().unwrap();
    let text = |b: Vec<u8>| String::from_utf8(b).unwrap();
    (out.status.code().unwrap_or(-1), text(out.stdout), text(out.stderr))
}

#[test]
fn startup_errors() {
    let p = "qemu-system-x86_64: ";
    let hint = "Use -machine help to list supported machines\n";
    // The x86 targets have TCG, and KVM on Linux x86_64 hosts; a target without a front
    // end has neither.
    #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
    {
        let (code, _, err) = system(&["-machine", "none", "-accel", "kvm"]);
        assert_eq!((code, err), (1, format!("{p}-accel kvm: invalid accelerator kvm\n")));
    }
    let (code, _, err) = system_for("alpha", &["-machine", "none"]);
    assert_eq!(
        (code, err.as_str()),
        (1, "qemu-system-alpha: No accelerator selected and no default accelerator available\n")
    );
    let (code, _, err) = system_for("alpha", &["-machine", "none", "-accel", "tcg"]);
    assert_eq!(
        (code, err.as_str()),
        (1, "qemu-system-alpha: -accel tcg: invalid accelerator tcg\n")
    );
    for (args, want) in [
        (
            &["-accel", "tcg,thread=bogus"][..],
            format!("{p}-accel tcg,thread=bogus: Invalid 'thread' setting bogus\n"),
        ),
        (
            &["-accel", "tcg,tb-size=x"],
            format!("{p}-accel tcg,tb-size=x: Parameter 'tb-size' expects uint64\n"),
        ),
        (
            &["-accel", "tcg,nope=1"],
            format!("{p}-accel tcg,nope=1: Property 'tcg-accel.nope' not found\n"),
        ),
    ] {
        let args: Vec<&str> = ["-machine", "none"].iter().chain(args).copied().collect();
        let (code, _, err) = system(&args);
        assert_eq!((code, err), (1, want), "{args:?}");
    }
    for (args, want) in [
        (&[][..], format!("{p}No machine specified, and there is no default\n{hint}")),
        (&["-machine", "pc"], format!("{p}unsupported machine type: \"pc\"\n{hint}")),
        (&["-m", "512"], format!("{p}No machine specified, and there is no default\n{hint}")),
        (&["-hda", "x.img"], format!("{p}-hda x.img: this option is not supported by ruvm yet\n")),
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
        (
            0,
            "Supported machines are:\n\
             microvm              microvm (i386)\n\
             none                 empty machine\n\
             q35                  Standard PC (Q35 + ICH9, 2009) (alias of pc-q35-11.1)\n\
             pc-q35-11.1          Standard PC (Q35 + ICH9, 2009)\n\
             pc-q35-11.0          Standard PC (Q35 + ICH9, 2009)\n\
             pc-q35-10.2          Standard PC (Q35 + ICH9, 2009)\n"
        )
    );
    let (code, out, _) = system(&["-L", "/a", "-L", "/b", "-L", "help"]);
    assert_eq!(code, 0);
    assert!(
        out.starts_with("/a\n/b\n/usr/share/qemu\n/usr/share/seabios\n/usr/local/share/qemu\n"),
        "{out}"
    );
    let (code, out, _) = system(&["-accel", "help"]);
    let kvm = if cfg!(all(target_os = "linux", target_arch = "x86_64")) { "kvm\n" } else { "" };
    assert_eq!((code, out), (0, format!("Accelerators supported in QEMU binary:\n{kvm}tcg\n")));
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

/// The x86 boards on the command line. Everything here fails before a vCPU would run, so it
/// needs no KVM: with `-accel qtest` the boards stop at the accelerator check, and the errors
/// QEMU finds earlier come first.
#[test]
fn x86_board_errors() {
    let p = "qemu-system-x86_64: ";
    let q = ["-accel", "qtest"];
    let only_kvm =
        format!("{p}this machine type is only supported with -accel kvm or tcg by ruvm yet\n");
    for (args, want) in [
        (&["-M", "microvm"][..], only_kvm.clone()),
        (&["-M", "q35", "-m", "256", "-smp", "2", "-nographic"], only_kvm.clone()),
        (&["-M", "q35", "-append", "x"], format!("{p}-append only allowed with -kernel option\n")),
        (&["-M", "q35", "-initrd", "x"], format!("{p}-initrd only allowed with -kernel option\n")),
        (&["-M", "microvm,bogus=on"], format!("{p}Property 'microvm-machine.bogus' not found\n")),
        (
            &["-M", "microvm", "-smp", "300"],
            format!(
                "{p}Invalid SMP CPUs 300. The max CPUs supported by machine 'microvm' is 288\n"
            ),
        ),
        (
            &["-M", "q35", "-smp", "cpus=4,sockets=3,cores=1"],
            format!(
                "{p}Invalid CPU topology: product of the hierarchy must match maxcpus: sockets (3) * dies (1) * modules (1) * cores (1) * threads (1) != maxcpus (4)\n"
            ),
        ),
        (
            &["-M", "q35", "-drive", "file=a.img,if=usb"],
            format!("{p}-drive file=a.img,if=usb: unsupported bus type 'usb'\n"),
        ),
        (
            &["-M", "q35", "-drive", "file=a.img,index=0", "-drive", "file=b.img,index=0"],
            format!("{p}-drive file=b.img,index=0: drive with bus=0, unit=0 (index=0) exists\n"),
        ),
        (
            &["-M", "microvm", "-device", "help"],
            format!("{p}-device help: -device help is not supported by ruvm yet\n"),
        ),
    ] {
        let args: Vec<&str> = args.iter().chain(&q).copied().collect();
        let (code, _, err) = system(&args);
        assert_eq!((code, err), (1, want), "{args:?}");
    }
}

/// A 64 KiB BIOS whose reset vector writes "ok" to `isa-debugcon` and 1 to
/// `isa-debug-exit` at 0xf4, then halts.
fn exit_bios() -> Vec<u8> {
    let mut b = vec![0xf4u8; 0x10000];
    let code = [
        0xb0, b'o', 0xe6, 0xe9, // mov al, 'o'; out 0xe9, al
        0xb0, b'k', 0xe6, 0xe9, // mov al, 'k'; out 0xe9, al
        0xb0, 0x01, 0xe6, 0xf4, // mov al, 1; out 0xf4, al
        0xf4, 0xeb, 0xfd, // hlt; jmp $-1
    ];
    b[0xfff0..0xfff0 + code.len()].copy_from_slice(&code);
    b
}

/// Both x86 boards run guest code on TCG (on any host), with the debugcon output and the
/// `isa-debug-exit` status, `(1 << 1) | 1`, that QEMU gives. `RUVM_TEST_FIRMWARE_DIR` is
/// passed as `-L`.
#[test]
fn tcg_runs_guest_code() {
    let dir = std::env::temp_dir().join(format!("ruvm-cli-tcg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bios = dir.join("exit.bin");
    std::fs::write(&bios, exit_bios()).unwrap();
    for (machine, accel) in [
        ("q35", "tcg"),
        ("microvm", "tcg"),
        ("q35", "tcg,thread=single"),
        ("microvm", "tcg,thread=multi"),
    ] {
        let out = dir.join(format!("{machine}.out"));
        let chardev = format!("file,path={},id=out", out.display());
        let fw = std::env::var("RUVM_TEST_FIRMWARE_DIR").ok();
        let mut args: Vec<&str> = fw.iter().flat_map(|d| ["-L", d.as_str()]).collect();
        args.extend([
            "-M",
            machine,
            "-accel",
            accel,
            "-display",
            "none",
            "-nodefaults",
            "-bios",
            bios.to_str().unwrap(),
            "-chardev",
            &chardev,
            "-device",
            "isa-debugcon,chardev=out",
            "-device",
            "isa-debug-exit,iobase=0xf4,iosize=4",
        ]);
        let (code, _, err) = system(&args);
        // Without firmware installed q35 warns that kvmvapic.bin is missing, as QEMU does.
        let err: String = err
            .lines()
            .filter(|l| !l.starts_with("qemu-system-x86_64: warning: rom: file kvmvapic.bin "))
            .map(|l| format!("{l}\n"))
            .collect();
        assert_eq!((code, err.as_str()), (3, ""), "{machine} {accel}");
        assert_eq!(std::fs::read(&out).unwrap(), b"ok", "{machine} {accel}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Boots Linux on KVM with the command line of the docs and waits for the kernel banner on
/// stdout. Skipped without `RUVM_TEST_KERNEL` (a bzImage), and without a usable `/dev/kvm`
/// unless `RUVM_REQUIRE_KVM` is set. `RUVM_TEST_FIRMWARE_DIR` is passed as `-L`.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod boot {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    fn kvm_usable() -> bool {
        match std::fs::OpenOptions::new().read(true).write(true).open("/dev/kvm") {
            Ok(_) => true,
            Err(e) => {
                if std::env::var_os("RUVM_REQUIRE_KVM").is_some() {
                    panic!("RUVM_REQUIRE_KVM is set but /dev/kvm cannot be opened: {e}");
                }
                eprintln!("skipping: /dev/kvm: {e}");
                false
            }
        }
    }

    fn boot(machine: &str, serial: &[&str]) {
        if !kvm_usable() {
            return;
        }
        let Some(kernel) = std::env::var_os("RUVM_TEST_KERNEL") else {
            eprintln!("skipping: RUVM_TEST_KERNEL is not set");
            return;
        };
        let timeout: u64 =
            std::env::var("RUVM_TEST_BOOT_TIMEOUT").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
        let mut cmd = Command::new(ruvm());
        cmd.arg("qemu-system-x86_64")
            .args(["-M", machine, "-accel", "kvm", "-m", "256", "-no-reboot"])
            .arg("-kernel")
            .arg(&kernel)
            .args(["-append", "console=ttyS0 panic=-1"])
            .args(serial);
        if let Some(dir) = std::env::var_os("RUVM_TEST_FIRMWARE_DIR") {
            cmd.arg("-L").arg(dir);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut stdout = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = [0u8; 4096];
            let mut seen = Vec::new();
            loop {
                match stdout.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => seen.extend_from_slice(&buf[..n]),
                }
                if String::from_utf8_lossy(&seen).contains("Linux version") {
                    let _ = tx.send(true);
                    return;
                }
            }
            let _ = tx.send(false);
        });
        let ok = rx.recv_timeout(Duration::from_secs(timeout)).unwrap_or(false);
        let _ = child.kill();
        let out = child.wait_with_output().unwrap();
        assert!(
            ok,
            "no \"Linux version\" on stdout within {timeout}s; stderr:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn microvm_nographic() {
        boot("microvm", &["-nographic"]);
    }

    #[test]
    fn microvm_serial_stdio() {
        boot("microvm", &["-serial", "stdio", "-display", "none"]);
    }

    #[test]
    fn q35_nographic() {
        boot("q35", &["-nographic"]);
    }
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
        // qtest_resolve_machine_alias() finds the q35 the tests ask for in query-machines.
        let machines = qmp.cmd(r#"{"execute": "query-machines"}"#);
        let q35 = r#"{"hotpluggable-cpus": true, "name": "pc-q35-11.1", "numa-mem-supported": false, "default-cpu-type": "qemu64-x86_64-cpu", "acpi": true, "cpu-max": 4096, "deprecated": false, "default-ram-id": "pc.ram", "alias": "q35"}"#;
        assert!(machines.contains(q35), "{machines}");
        for name in ["none", "microvm", "pc-q35-11.0", "pc-q35-10.2"] {
            assert!(machines.contains(&format!(r#""name": "{name}""#)), "{machines}");
        }
        // The queries qmp-cmd-test runs on machine none, with QEMU's answers.
        for (c, want) in [
            ("query-target", r#"{"return": {"arch": "x86_64"}}"#),
            ("query-uuid", r#"{"return": {"UUID": "00000000-0000-0000-0000-000000000000"}}"#),
            (
                "query-yank",
                r#"{"return": [{"type": "chardev", "id": "char0"}, {"type": "chardev", "id": "qtest"}]}"#,
            ),
            ("query-memory-size-summary", r#"{"return": {"base-memory": 0, "plugged-memory": 0}}"#),
            ("query-current-machine", r#"{"return": {"wakeup-suspend-support": false}}"#),
            ("query-replay", r#"{"return": {"icount": 0, "mode": "none"}}"#),
            ("query-memdev", r#"{"return": []}"#),
            (
                "query-hotpluggable-cpus",
                r#"{"error": {"class": "GenericError", "desc": "machine does not support hot-plugging CPUs"}}"#,
            ),
            (
                "query-balloon",
                r#"{"error": {"class": "DeviceNotActive", "desc": "No balloon device has been activated"}}"#,
            ),
        ] {
            assert_eq!(qmp.cmd(&format!(r#"{{"execute": "{c}"}}"#)), want, "{c}");
        }
        let accels = qmp.cmd(r#"{"execute": "query-accelerators"}"#);
        assert!(accels.contains(r#""enabled": "qtest""#), "{accels}");
        let params = qmp.cmd(r#"{"execute": "query-migrate-parameters"}"#);
        assert!(params.contains(r#""cpr-exec-command": []"#), "{params}");
        for (c, want) in [
            ("x-accel-stats", r#"{"return": {"human-readable-text": ""}}"#),
            ("query-block", r#"{"return": []}"#),
            ("query-named-block-nodes", r#"{"return": []}"#),
            ("x-debug-query-block-graph", r#"{"return": {"edges": [], "nodes": []}}"#),
            ("query-rx-filter", r#"{"return": []}"#),
            ("query-stats-schemas", r#"{"return": []}"#),
            (
                "query-firmware-log",
                r#"{"error": {"class": "GenericError", "desc": "firmware log buffer not found"}}"#,
            ),
            (
                "xen-event-list",
                r#"{"error": {"class": "GenericError", "desc": "Xen event channel emulation not enabled"}}"#,
            ),
        ] {
            assert_eq!(qmp.cmd(&format!(r#"{{"execute": "{c}"}}"#)), want, "{c}");
        }
        let rate = qmp.cmd(r#"{"execute": "query-dirty-rate"}"#);
        assert!(rate.contains(r#""status": "unstarted""#), "{rate}");
        let opts = qmp.cmd(r#"{"execute": "query-command-line-options"}"#);
        assert!(opts.starts_with(r#"{"return": [{"parameters": [{"name": "type""#), "{opts}");
        assert!(opts.contains(r#""option": "drive""#), "{opts}");
        let bad =
            qmp.cmd(r#"{"execute": "query-command-line-options", "arguments": {"option": "foo"}}"#);
        assert!(bad.contains("invalid option name: foo"), "{bad}");
        for (line, want) in [
            ("help info uuid", r#"{"return": "info uuid  -- show the current VM UUID\r\n"}"#),
            ("frob", r#"{"return": "unknown command: 'frob'\r\n"}"#),
            (
                "info qtree",
                r#"{"return": "ruvm's human monitor does not have 'info qtree' yet\r\n"}"#,
            ),
        ] {
            let c = format!(
                r#"{{"execute": "human-monitor-command", "arguments": {{"command-line": "{line}"}}}}"#
            );
            assert_eq!(qmp.cmd(&c), want, "{line}");
        }
        let c =
            r#"{"execute": "human-monitor-command", "arguments": {"command-line": "help info"}}"#;
        let info = qmp.cmd(c);
        assert!(
            info.starts_with(r#"{"return": "info kvm  -- show KVM information\r\ninfo name "#),
            "{info}"
        );
        let c =
            r#"{"execute": "human-monitor-command", "arguments": {"command-line": "info status"}}"#;
        let status = qmp.cmd(c);
        assert!(status.starts_with(r#"{"return": "VM status: "#), "{status}");

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

    /// The reply goes out over QMP exactly as built. crates/qapi/tests/introspect.rs checks the
    /// built reply against QEMU's after the normalization list.
    #[test]
    fn query_qmp_schema_is_the_built_in_reply() {
        let dir = std::env::temp_dir().join(format!("ruvm-schema-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ms = dir.join("qmp.sock");
        let _ = std::fs::remove_file(&ms);
        let child = Command::new(ruvm())
            .arg("qemu-system-x86_64")
            .args(["-machine", "none", "-accel", "qtest", "-display", "none", "-nodefaults"])
            .args(["-qmp", &format!("unix:{},server=on,wait=off", ms.display())])
            .spawn()
            .unwrap();
        let _child = Machine(child);
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
        // Not cmd(): the reply mentions "event" all over, and there are no events to skip here.
        writeln!(qmp.writer, r#"{{"execute": "query-qmp-schema"}}"#).unwrap();
        let reply = qmp.line();
        let reply = reply.trim_end();
        let want = format!("{{\"return\": {}}}", ruvm_qapi::QMP_SCHEMA_JSON);
        assert!(reply == want, "the schema reply differs from the built-in one");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
