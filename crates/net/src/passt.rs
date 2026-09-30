// SPDX-License-Identifier: GPL-2.0-or-later

//! `-netdev passt`, ported from net/passt.c.
//!
//! passt is a separate program that gives the guest user mode networking. QEMU starts it with
//! one end of a socket pair as descriptor 3 and `--fd 3`; passt goes into the background and
//! writes its pid to a file, and from then on frames go over the socket with a 4 byte length
//! in front, the way the `stream` backend sends them. When passt hangs up it is killed and
//! started again, and deleting the netdev kills it.
//!
//! QEMU only builds this backend on Linux, and the generated QAPI types here are made without
//! it, so the options are declared in this module and `-netdev passt` is picked out before the
//! generic `Netdev` visit. [`passt_args`] builds the command line on every host so it can be
//! checked without passt around.

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::QemuOptsList;
use ruvm_qapi::visit::{Visitor, VisitorExt};

use crate::opts_visitor::OptsVisitor;

/// `NetdevPasstOptions`. `None` means the option was not given.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PasstOptions {
    pub path: Option<String>,
    pub quiet: Option<bool>,
    pub vhost_user: Option<bool>,
    pub mtu: Option<i64>,
    pub address: Option<String>,
    pub netmask: Option<String>,
    pub mac: Option<String>,
    pub gateway: Option<String>,
    pub interface: Option<String>,
    pub outbound: Option<String>,
    pub outbound_if4: Option<String>,
    pub outbound_if6: Option<String>,
    pub dns: Option<String>,
    /// `PasstSearch` entries.
    pub search: Option<Vec<String>>,
    pub fqdn: Option<String>,
    pub dhcp_dns: Option<bool>,
    pub dhcp_search: Option<bool>,
    pub map_host_loopback: Option<String>,
    pub map_guest_addr: Option<String>,
    pub dns_forward: Option<String>,
    pub dns_host: Option<String>,
    pub tcp: Option<bool>,
    pub udp: Option<bool>,
    pub icmp: Option<bool>,
    pub dhcp: Option<bool>,
    pub ndp: Option<bool>,
    pub dhcpv6: Option<bool>,
    pub ra: Option<bool>,
    pub freebind: Option<bool>,
    pub ipv4: Option<bool>,
    pub ipv6: Option<bool>,
    /// `PasstPortForward` entries.
    pub tcp_ports: Option<Vec<String>>,
    /// `PasstPortForward` entries.
    pub udp_ports: Option<Vec<String>>,
    /// `PasstParameter` entries, passed to passt as they are.
    pub param: Option<Vec<String>>,
}

fn opt_str(v: &mut dyn Visitor, name: &str, obj: &mut Option<String>) -> Result<()> {
    if v.optional(Some(name), obj.is_some()) {
        v.type_str(Some(name), obj.get_or_insert_with(String::new))?;
    }
    Ok(())
}

fn opt_bool(v: &mut dyn Visitor, name: &str, obj: &mut Option<bool>) -> Result<()> {
    if v.optional(Some(name), obj.is_some()) {
        v.type_bool(Some(name), obj.get_or_insert(false))?;
    }
    Ok(())
}

/// A list of `{ 'str': 'str' }` structs, which on the command line is the key repeated.
fn opt_str_list(v: &mut dyn Visitor, name: &str, obj: &mut Option<Vec<String>>) -> Result<()> {
    if v.optional(Some(name), obj.is_some()) {
        let list = obj.get_or_insert_with(Vec::new);
        v.visit_list(Some(name), list, |v, e: &mut String| {
            v.start_struct(None)?;
            let r = v.type_str(Some("str"), e).and_then(|()| v.check_struct());
            v.end_struct();
            r
        })?;
    }
    Ok(())
}

impl PasstOptions {
    /// `visit_type_NetdevPasstOptions_members()`.
    pub fn visit_members(v: &mut dyn Visitor, obj: &mut Self) -> Result<()> {
        opt_str(v, "path", &mut obj.path)?;
        opt_bool(v, "quiet", &mut obj.quiet)?;
        opt_bool(v, "vhost-user", &mut obj.vhost_user)?;
        if v.optional(Some("mtu"), obj.mtu.is_some()) {
            v.type_int64(Some("mtu"), obj.mtu.get_or_insert(0))?;
        }
        opt_str(v, "address", &mut obj.address)?;
        opt_str(v, "netmask", &mut obj.netmask)?;
        opt_str(v, "mac", &mut obj.mac)?;
        opt_str(v, "gateway", &mut obj.gateway)?;
        opt_str(v, "interface", &mut obj.interface)?;
        opt_str(v, "outbound", &mut obj.outbound)?;
        opt_str(v, "outbound-if4", &mut obj.outbound_if4)?;
        opt_str(v, "outbound-if6", &mut obj.outbound_if6)?;
        opt_str(v, "dns", &mut obj.dns)?;
        opt_str_list(v, "search", &mut obj.search)?;
        opt_str(v, "fqdn", &mut obj.fqdn)?;
        opt_bool(v, "dhcp-dns", &mut obj.dhcp_dns)?;
        opt_bool(v, "dhcp-search", &mut obj.dhcp_search)?;
        opt_str(v, "map-host-loopback", &mut obj.map_host_loopback)?;
        opt_str(v, "map-guest-addr", &mut obj.map_guest_addr)?;
        opt_str(v, "dns-forward", &mut obj.dns_forward)?;
        opt_str(v, "dns-host", &mut obj.dns_host)?;
        opt_bool(v, "tcp", &mut obj.tcp)?;
        opt_bool(v, "udp", &mut obj.udp)?;
        opt_bool(v, "icmp", &mut obj.icmp)?;
        opt_bool(v, "dhcp", &mut obj.dhcp)?;
        opt_bool(v, "ndp", &mut obj.ndp)?;
        opt_bool(v, "dhcpv6", &mut obj.dhcpv6)?;
        opt_bool(v, "ra", &mut obj.ra)?;
        opt_bool(v, "freebind", &mut obj.freebind)?;
        opt_bool(v, "ipv4", &mut obj.ipv4)?;
        opt_bool(v, "ipv6", &mut obj.ipv6)?;
        opt_str_list(v, "tcp-ports", &mut obj.tcp_ports)?;
        opt_str_list(v, "udp-ports", &mut obj.udp_ports)?;
        opt_str_list(v, "param", &mut obj.param)?;
        Ok(())
    }

    /// Reads `-netdev passt,...` options the way `net_client_init()` would visit them as a
    /// `Netdev`, returning the id with the options. The type must be `passt`.
    pub fn parse(optarg: &str) -> Result<(String, PasstOptions)> {
        let mut list = QemuOptsList::new("netdev", &[]).with_implied_opt_name("type");
        let opts = list.parse(optarg, true)?;
        visit_netdev(&mut OptsVisitor::new(opts))
    }
}

/// The `Netdev` visit for `type=passt`: `id`, `type`, then the branch members.
pub(crate) fn visit_netdev(v: &mut dyn Visitor) -> Result<(String, PasstOptions)> {
    let mut id = String::new();
    let mut opts = PasstOptions::default();
    v.start_struct(None)?;
    let r = (|| {
        v.type_str(Some("id"), &mut id)?;
        let mut ty = String::new();
        v.type_str(Some("type"), &mut ty)?;
        if ty != "passt" {
            return Err(Error::generic(format!("Parameter 'type' does not accept value '{ty}'")));
        }
        PasstOptions::visit_members(v, &mut opts)?;
        v.check_struct()
    })();
    v.end_struct();
    r.map(|()| (id, opts))
}

/// `net_passt_decode_args()`: the passt command line, program name first.
pub fn passt_args(passt: &PasstOptions, pidfile: &str) -> Vec<String> {
    let mut args = vec![passt.path.clone().unwrap_or_else(|| "passt".to_string())];
    let flag = |args: &mut Vec<String>, name: &str, value: &Option<String>| {
        if let Some(value) = value {
            args.push(name.to_string());
            args.push(value.clone());
        }
    };
    if passt.vhost_user == Some(true) {
        args.push("--vhost-user".to_string());
    }
    // By default, be quiet.
    if passt.quiet != Some(false) {
        args.push("--quiet".to_string());
    }
    if let Some(mtu) = passt.mtu {
        args.push("--mtu".to_string());
        args.push(mtu.to_string());
    }
    flag(&mut args, "--address", &passt.address);
    flag(&mut args, "--netmask", &passt.netmask);
    flag(&mut args, "--mac-addr", &passt.mac);
    flag(&mut args, "--gateway", &passt.gateway);
    flag(&mut args, "--interface", &passt.interface);
    flag(&mut args, "--outbound", &passt.outbound);
    flag(&mut args, "--outbound-if4", &passt.outbound_if4);
    flag(&mut args, "--outbound-if6", &passt.outbound_if6);
    flag(&mut args, "--dns", &passt.dns);
    flag(&mut args, "--fqdn", &passt.fqdn);
    let off = |args: &mut Vec<String>, name: &str, value: Option<bool>| {
        if value == Some(false) {
            args.push(name.to_string());
        }
    };
    off(&mut args, "--no-dhcp-dns", passt.dhcp_dns);
    off(&mut args, "--no-dhcp-search", passt.dhcp_search);
    flag(&mut args, "--map-host-loopback", &passt.map_host_loopback);
    flag(&mut args, "--map-guest-addr", &passt.map_guest_addr);
    flag(&mut args, "--dns-forward", &passt.dns_forward);
    flag(&mut args, "--dns-host", &passt.dns_host);
    off(&mut args, "--no-tcp", passt.tcp);
    off(&mut args, "--no-udp", passt.udp);
    off(&mut args, "--no-icmp", passt.icmp);
    off(&mut args, "--no-dhcp", passt.dhcp);
    off(&mut args, "--no-ndp", passt.ndp);
    off(&mut args, "--no-dhcpv6", passt.dhcpv6);
    off(&mut args, "--no-ra", passt.ra);
    if passt.freebind == Some(true) {
        args.push("--freebind".to_string());
    }
    off(&mut args, "--ipv6-only", passt.ipv4);
    off(&mut args, "--ipv4-only", passt.ipv6);
    let joined = |args: &mut Vec<String>, name: &str, list: &Option<Vec<String>>, sep| {
        if let Some(list) = list.as_ref().filter(|l| !l.is_empty()) {
            args.push(name.to_string());
            args.push(list.join(sep));
        }
    };
    joined(&mut args, "--search", &passt.search, " ");
    joined(&mut args, "--tcp-ports", &passt.tcp_ports, ",");
    joined(&mut args, "--udp-ports", &passt.udp_ports, ",");
    if let Some(param) = &passt.param {
        args.extend(param.iter().cloned());
    }
    // A pid file to be able to kill passt on exit.
    args.push("--pid".to_string());
    args.push(pidfile.to_string());
    // The socket goes to passt as descriptor 3.
    args.push("--fd".to_string());
    args.push("3".to_string());
    args
}

#[cfg(target_os = "linux")]
pub(crate) use runtime::net_init_passt;

#[cfg(target_os = "linux")]
#[allow(unsafe_code)]
mod runtime {
    use std::os::fd::{AsRawFd, OwnedFd};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::path::PathBuf;
    use std::process::Command;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::{Arc, Mutex};

    use rustix::net::{AddressFamily, SocketFlags, SocketType};
    use rustix::process::{Pid, Signal};
    use ruvm_base::error::strerror;
    use ruvm_base::{Error, Result, error_report, warn_report};
    use ruvm_qapi::types::NetClientDriver;

    use super::{PasstOptions, passt_args};
    use crate::client::{NetClient, lock};
    use crate::net::Net;
    use crate::sock::{Flavour, Framing, SockConfig, SockHooks, fd_cloexec, new_sock};

    /// What `NetPasstState` keeps besides the stream: the command line, the pid file and the
    /// pid of the running passt.
    #[derive(Debug)]
    struct Passt {
        args: Vec<String>,
        pidfile: PathBuf,
        pid: AtomicI32,
        /// Serializes restarts with cleanup.
        busy: Mutex<()>,
    }

    fn kill(pid: i32) {
        if let Some(pid) = Pid::from_raw(pid) {
            let _ = rustix::process::kill_process(pid, Signal::TERM);
        }
    }

    /// `g_ascii_strtoll(contents, NULL, 10)`: leading blanks, a sign and the digits that
    /// follow, 0 when there are none.
    fn parse_pid(s: &str) -> i64 {
        let s = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']);
        let (neg, rest) = match s.as_bytes().first() {
            Some(b'-') => (true, &s[1..]),
            Some(b'+') => (false, &s[1..]),
            _ => (false, s),
        };
        let digits: &str = &rest[..rest.bytes().take_while(u8::is_ascii_digit).count()];
        let v = digits.parse::<i64>().unwrap_or(if digits.is_empty() { 0 } else { i64::MAX });
        if neg { -v } else { v }
    }

    impl Passt {
        /// `net_passt_start_daemon()`: runs passt with `sock` as descriptor 3 and waits for it
        /// to go into the background, then reads its pid.
        fn start_daemon(&self, nc: &NetClient, sock: OwnedFd) -> Result<()> {
            nc.set_info_str("launching passt");
            let raw = sock.as_raw_fd();
            let mut cmd = Command::new(&self.args[0]);
            cmd.args(&self.args[1..]);
            // SAFETY: the hook runs in the child between fork and exec and only makes the two
            // async-signal-safe calls dup2() and fcntl() on a descriptor the child inherited
            // (`sock` stays open in the parent until the child has been waited for).
            unsafe {
                cmd.pre_exec(move || {
                    if raw == 3 {
                        let flags = libc::fcntl(3, libc::F_GETFD);
                        if flags < 0 || libc::fcntl(3, libc::F_SETFD, flags & !libc::FD_CLOEXEC) < 0
                        {
                            return Err(std::io::Error::last_os_error());
                        }
                    } else if libc::dup2(raw, 3) < 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    Ok(())
                });
            }
            let status = cmd.status().map_err(|e| {
                Error::generic(format!(
                    "Error creating daemon: Failed to execute child process \u{201c}{}\u{201d} ({})",
                    self.args[0],
                    strerror(&e)
                ))
            })?;
            drop(sock);
            if let Some(code) = status.code() {
                if code != 0 {
                    return Err(Error::generic(format!("Passt exited with code {code}")));
                }
            }
            if let Some(sig) = status.signal() {
                return Err(Error::generic(format!("Passt killed with signal {sig}")));
            }
            let contents = std::fs::read(&self.pidfile).map_err(|e| {
                Error::generic(format!(
                    "Cannot read passt pid: Failed to open file \u{201c}{}\u{201d}: {}",
                    self.pidfile.display(),
                    strerror(&e)
                ))
            })?;
            let pid = parse_pid(&String::from_utf8_lossy(&contents));
            // pid_t is 32 bits; the conversion truncates like the C cast.
            let pid = pid as i32;
            self.pid.store(pid, Ordering::SeqCst);
            if pid <= 0 {
                return Err(Error::generic(format!(
                    "File '{}' did not contain a valid PID.",
                    self.pidfile.display()
                )));
            }
            Ok(())
        }

        /// `net_passt_stream_start()`: a new socket pair and a new passt on the other end.
        fn stream_start(&self, nc: &NetClient) -> Result<OwnedFd> {
            let (ours, theirs) = rustix::net::socketpair(
                AddressFamily::UNIX,
                SocketType::STREAM,
                SocketFlags::empty(),
                None,
            )
            .map_err(|e| Error::from_io("socketpair() failed", e.into()))?;
            fd_cloexec(&ours)
                .and_then(|()| fd_cloexec(&theirs))
                .map_err(|e| Error::from_io("socketpair() failed", e))?;
            nc.set_info_str("connecting to passt");
            nc.set_link_down(true);
            self.start_daemon(nc, theirs)?;
            nc.set_info_str(&format!(
                "stream,connected to pid {}",
                self.pid.load(Ordering::SeqCst)
            ));
            Ok(ours)
        }

        fn remove_pidfile(&self) {
            if let Err(e) = std::fs::remove_file(&self.pidfile) {
                warn_report(&format!(
                    "Failed to remove passt pidfile {}: {}",
                    self.pidfile.display(),
                    strerror(&e)
                ));
            }
        }
    }

    impl SockHooks for Passt {
        /// `net_passt_send()` after the end of the stream: passt is restarted.
        fn restart(&self, nc: &NetClient) -> Option<OwnedFd> {
            let _busy = lock(&self.busy);
            kill(self.pid.load(Ordering::SeqCst));
            match self.stream_start(nc) {
                Ok(fd) => Some(fd),
                Err(e) => {
                    error_report(e.message());
                    None
                }
            }
        }

        /// `net_passt_cleanup()`.
        fn cleanup(&self) {
            let _busy = lock(&self.busy);
            let pid = self.pid.swap(0, Ordering::SeqCst);
            if pid > 0 {
                kill(pid);
            }
            self.remove_pidfile();
        }
    }

    /// `net_init_passt()`.
    pub(crate) fn net_init_passt(
        net: &mut Net,
        passt: &PasstOptions,
        name: &str,
        peer: Option<Arc<NetClient>>,
    ) -> Result<()> {
        let pidfile = crate::util::make_temp("passt-", ".pid", false).map_err(|e| {
            Error::generic(format!("Failed to create temporary file: {}", strerror(&e)))
        })?;
        let state = Arc::new(Passt {
            args: passt_args(passt, &pidfile.to_string_lossy()),
            pidfile,
            pid: AtomicI32::new(0),
            busy: Mutex::new(()),
        });
        let mut cfg = SockConfig::new(Framing::Stream, Flavour::Stream);
        cfg.hooks = Some(state.clone());
        let (nc, sock) = new_sock(net, NetClientDriver::Stream, peer.as_ref(), "passt", name, cfg)?;
        if passt.vhost_user == Some(true) {
            net.del_client(&nc);
            return Err(Error::generic("passt vhost-user mode is not supported yet"));
        }
        match state.stream_start(&nc) {
            Ok(fd) => {
                sock.attach(fd);
                Ok(())
            }
            Err(e) => {
                net.del_client(&nc);
                Err(e)
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::parse_pid;

        #[test]
        fn pids() {
            assert_eq!(parse_pid("1234\n"), 1234);
            assert_eq!(parse_pid("  42abc"), 42);
            assert_eq!(parse_pid(""), 0);
            assert_eq!(parse_pid("-5"), -5);
            assert_eq!(parse_pid("x1"), 0);
        }
    }
}
