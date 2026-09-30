// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev passt`: the command line on every host, and on Linux the backend itself against a
//! stand-in script that echoes frames back, plus the real passt when it is installed.

mod common;

use ruvm_net::{PasstOptions, passt_args};

fn args(optarg: &str) -> Vec<String> {
    let (_, opts) = PasstOptions::parse(optarg).unwrap();
    passt_args(&opts, "/tmp/passt-XXXXXX.pid")
}

fn parse_err(optarg: &str) -> String {
    PasstOptions::parse(optarg).unwrap_err().message().to_string()
}

#[test]
fn default_command_line() {
    let (id, opts) = PasstOptions::parse("passt,id=p0").unwrap();
    assert_eq!(id, "p0");
    assert_eq!(opts, PasstOptions::default());
    assert_eq!(
        args("passt,id=p0"),
        ["passt", "--quiet", "--pid", "/tmp/passt-XXXXXX.pid", "--fd", "3"]
    );
}

#[test]
fn every_option() {
    let optarg = "passt,id=p0,path=/opt/passt,quiet=off,vhost-user=on,mtu=1400,\
                  address=10.0.0.2,netmask=255.255.255.0,mac=52:54:00:12:34:56,\
                  gateway=10.0.0.1,interface=eth0,outbound=10.0.0.3,outbound-if4=eth1,\
                  outbound-if6=eth2,dns=10.0.0.4,search=a.example,search=b.example,\
                  fqdn=vm.example,dhcp-dns=off,dhcp-search=off,map-host-loopback=10.0.0.5,\
                  map-guest-addr=10.0.0.6,dns-forward=10.0.0.7,dns-host=10.0.0.8,\
                  tcp=off,udp=off,icmp=off,dhcp=off,ndp=off,dhcpv6=off,ra=off,freebind=on,\
                  ipv4=off,ipv6=off,tcp-ports=22,tcp-ports=80:8080,udp-ports=53,\
                  param=--debug,param=--trace";
    assert_eq!(
        args(optarg),
        [
            "/opt/passt",
            "--vhost-user",
            "--mtu",
            "1400",
            "--address",
            "10.0.0.2",
            "--netmask",
            "255.255.255.0",
            "--mac-addr",
            "52:54:00:12:34:56",
            "--gateway",
            "10.0.0.1",
            "--interface",
            "eth0",
            "--outbound",
            "10.0.0.3",
            "--outbound-if4",
            "eth1",
            "--outbound-if6",
            "eth2",
            "--dns",
            "10.0.0.4",
            "--fqdn",
            "vm.example",
            "--no-dhcp-dns",
            "--no-dhcp-search",
            "--map-host-loopback",
            "10.0.0.5",
            "--map-guest-addr",
            "10.0.0.6",
            "--dns-forward",
            "10.0.0.7",
            "--dns-host",
            "10.0.0.8",
            "--no-tcp",
            "--no-udp",
            "--no-icmp",
            "--no-dhcp",
            "--no-ndp",
            "--no-dhcpv6",
            "--no-ra",
            "--freebind",
            "--ipv6-only",
            "--ipv4-only",
            "--search",
            "a.example b.example",
            "--tcp-ports",
            "22,80:8080",
            "--udp-ports",
            "53",
            "--debug",
            "--trace",
            "--pid",
            "/tmp/passt-XXXXXX.pid",
            "--fd",
            "3",
        ]
    );
}

#[test]
fn options_that_add_nothing() {
    // Switching on what is on by default, or off what is off, leaves the command line alone.
    let a = args(
        "passt,id=p0,quiet=on,vhost-user=off,dhcp-dns=on,tcp=on,udp=on,icmp=on,dhcp=on,\
         ndp=on,dhcpv6=on,ra=on,freebind=off,ipv4=on,ipv6=on",
    );
    assert_eq!(a, args("passt,id=p0"));
}

#[test]
fn option_errors() {
    assert_eq!(parse_err("passt"), "Parameter 'id' is missing");
    assert_eq!(parse_err("passt,id=p0,bogus=1"), "Invalid parameter 'bogus'");
    assert_eq!(parse_err("passt,id=p0,mtu=big"), "Parameter 'mtu' expects an int64 value");
    assert_eq!(parse_err("passt,id=p0,tcp=maybe"), "Parameter 'tcp' expects 'on' or 'off'");
    assert_eq!(parse_err("user,id=p0"), "Parameter 'type' does not accept value 'user'");
}

#[cfg(not(target_os = "linux"))]
#[test]
fn not_available_off_linux() {
    assert_eq!(common::netdev_err("passt,id=p0"), "Parameter 'type' does not accept value 'passt'");
    assert!(!ruvm_net::AVAILABLE_NETDEVS.contains(&"passt"));
}

#[cfg(target_os = "linux")]
mod linux {
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use ruvm_net::Net;

    use crate::common::{attach_nic, frame, info, netdev, netdev_err, tmpdir, wait_until};

    /// A stand-in for passt: it records its arguments, leaves `cat` behind echoing the socket
    /// on descriptor 3, and writes the pid of that to the `--pid` file.
    fn fake_passt(dir: &Path) -> PathBuf {
        let path = dir.join("passt");
        std::fs::write(
            &path,
            "#!/bin/sh\n\
             printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/args\"\n\
             pidfile=\n\
             while [ $# -gt 0 ]; do\n\
             \x20 if [ \"$1\" = --pid ]; then pidfile=\"$2\"; fi\n\
             \x20 shift\n\
             done\n\
             cat <&3 >&3 3>&- 2>/dev/null &\n\
             echo $! > \"$pidfile\"\n",
        )
        .unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    fn pid_of(net: &Net, id: &str) -> Option<u32> {
        info(net, id).strip_prefix("stream,connected to pid ")?.parse().ok()
    }

    fn alive(pid: u32) -> bool {
        Command::new("kill")
            .arg("-0")
            .arg(pid.to_string())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap()
            .success()
    }

    #[test]
    fn frames_restart_and_removal() {
        let dir = tmpdir("passt");
        let prog = fake_passt(&dir);
        let mut net = Net::new();
        netdev(&mut net, &format!("passt,id=p0,path={},mtu=1500", prog.display())).unwrap();
        let nc = net.find_netdev("p0").unwrap();
        assert_eq!(nc.model(), "passt");
        assert!(nc.is_netdev());
        let pid = pid_of(&net, "p0").expect("connected");
        assert!(alive(pid));

        let argv = std::fs::read_to_string(dir.join("args")).unwrap();
        let argv: Vec<&str> = argv.lines().collect();
        assert_eq!(argv[..3], ["--quiet", "--mtu", "1500"]);
        assert_eq!(argv[3], "--pid");
        let pidfile = PathBuf::from(argv[4]);
        assert!(pidfile.file_name().unwrap().to_str().unwrap().starts_with("passt-"));
        assert_eq!(argv[5..], ["--fd", "3"]);
        assert!(pidfile.exists());

        let (nic, rec) = attach_nic(&mut net, "p0");
        wait_until("link up", || !nc.link_down());
        nic.queue().send_packet(&frame(1, 80));
        assert_eq!(rec.wait_for(1), vec![frame(1, 80)]);

        // passt going away starts a new one.
        assert!(Command::new("kill").arg(pid.to_string()).status().unwrap().success());
        wait_until("a new passt", || pid_of(&net, "p0").is_some_and(|p| p != pid));
        let pid2 = pid_of(&net, "p0").unwrap();
        wait_until("link up", || !nc.link_down());
        nic.queue().send_packet(&frame(2, 90));
        assert_eq!(rec.wait_for(1), vec![frame(2, 90)]);

        // With a NIC attached, the backend goes when the NIC does, as in QEMU.
        net.netdev_del("p0").unwrap();
        assert!(alive(pid2));
        net.del_nic(&nic);
        wait_until("passt to be killed", || !alive(pid2));
        assert!(!pidfile.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn start_errors() {
        let dir = tmpdir("passt-err");
        let e = netdev_err("passt,id=p0,path=/nonexistent/passt");
        assert!(e.starts_with("Error creating daemon: Failed to execute child process"), "{e}");
        let prog = script(&dir, "fails", "exit 3");
        assert_eq!(
            netdev_err(&format!("passt,id=p0,path={}", prog.display())),
            "Passt exited with code 3"
        );
        let prog = script(&dir, "killed", "kill -9 $$");
        assert_eq!(
            netdev_err(&format!("passt,id=p0,path={}", prog.display())),
            "Passt killed with signal 9"
        );
        let prog = script(&dir, "nopid", "exit 0");
        let e = netdev_err(&format!("passt,id=p0,path={}", prog.display()));
        assert!(e.starts_with("File '") && e.ends_with("' did not contain a valid PID."), "{e}");

        let prog = fake_passt(&dir);
        let mut net = Net::new();
        let e = netdev(&mut net, &format!("passt,id=p0,path={},vhost-user=on", prog.display()));
        assert_eq!(e.unwrap_err(), "passt vhost-user mode is not supported yet");
        assert!(net.find_netdev("p0").is_none());
        netdev(&mut net, &format!("passt,id=p0,path={}", prog.display())).unwrap();
        let e = netdev(&mut net, &format!("passt,id=p0,path={}", prog.display()));
        assert_eq!(e.unwrap_err(), "Duplicate ID 'p0' for netdev");
        net.parse_net(&format!("passt,id=p0,path={}", prog.display())).unwrap();
        assert_eq!(net.init_clients().unwrap_err().message(), "Duplicate ID 'p0'");
        net.netdev_del("p0").unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn net_passt_joins_a_hub() {
        let dir = tmpdir("passt-hub");
        let prog = fake_passt(&dir);
        let mut net = Net::new();
        net.parse_net(&format!("passt,path={}", prog.display())).unwrap();
        net.init_clients().unwrap();
        let nc = net.clients().iter().find(|c| c.model() == "passt").unwrap().clone();
        assert!(!nc.is_netdev());
        assert_eq!(nc.peer().unwrap().model(), "hub");
        net.cleanup();
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The real passt, when it is installed.
    #[test]
    fn live_passt() {
        let found = std::env::var_os("PATH")
            .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join("passt").is_file()));
        if !found {
            eprintln!("passt is not in PATH; skipping");
            return;
        }
        let mut net = Net::new();
        netdev(&mut net, "passt,id=p0").unwrap();
        let pid = pid_of(&net, "p0").expect("connected");
        net.netdev_del("p0").unwrap();
        wait_until("passt to exit", || !alive(pid));
    }
}
