// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-nbd, the NBD server for disk images: qemu-nbd.c.
//!
//! [`main`] takes the arguments after `argv[0]` and returns the exit status. It opens the image
//! named on the command line, exports it over NBD on a TCP port or a Unix socket, and runs until
//! the last client goes away (or forever with `--persistent`) or a termination signal arrives.
//! `--list` connects to a server instead and prints its exports. `--connect` and `--disconnect`
//! drive the Linux kernel NBD client through `/dev/nbdN`. Every option of QEMU 11.1 is there,
//! with the same help text, messages and exit statuses.
//!
//! [`os`] has the process plumbing (PID files, `daemon()`, `os_daemonize()` and systemd socket
//! activation) that qemu-storage-daemon shares.
//!
//! Differences from QEMU:
//!
//! - There is no TLS: no object type is TLS credentials, so `--tls-creds` fails with QEMU's
//!   "Failed to get TLS creds: No TLS credentials with id '<id>'" (or "Object with id '<id>' is
//!   not TLS credentials" for an object of another type).
//! - `-T`/`--trace` is accepted and ignored; there are no trace events, so QEMU's warning about
//!   an unknown event is not printed either.
//! - `--bitmap` fails with "Bitmap '<name>' is not found": the export only sees the dirty
//!   bitmaps its creator hands in, and the block layer offers none to qemu-nbd yet.
//! - `--selinux-label` always fails with "SELinux support not enabled in this binary", as in a
//!   QEMU built without libselinux.
//! - `--aio=io_uring` is refused as in a QEMU built without liburing.
//! - `--snapshot` makes the temporary qcow2 overlay itself, the way `bdrv_append_temp_snapshot()`
//!   does, and removes the file as soon as it is open (at exit on Windows), where QEMU removes it
//!   when the node closes. It needs the qcow2 driver.
//! - The listening socket accepts connections on a thread of its own and every client is served
//!   on its own thread, instead of from the main loop. A client over the `--shared` limit waits
//!   in the listen backlog, as in QEMU.
//! - The PID file lock is the one described in [`os`].
//! - `--connect` hands the socket to the kernel with the same ioctls, from a thread, and ends the
//!   server when `NBD_DO_IT` returns by the same route a signal takes, rather than by sending
//!   itself SIGTERM.
//! - On Windows there are no termination signals to catch; Ctrl-C ends the process at once.

#![deny(unsafe_code)]

use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};

use ruvm_base::Error;
use ruvm_base::report::{error_report, report_error, set_program_name};
use ruvm_block::BlockBackend;
use ruvm_block::nbd::{
    NBD_DEFAULT_HANDSHAKE_MAX_SECS, NBD_DEFAULT_PORT, NBD_MAX_STRING_SIZE, NbdServer,
};
use ruvm_block::tools::OpenFlags;
use ruvm_img::getopt::{Getopt, HasArg, LongOpt, Opt, lopt};
use ruvm_img::shared::{graph, object_add, parse_cache_mode, register_object_types};
use ruvm_img::{Exit, Flow};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptType, QemuOptsList};
use ruvm_qapi::types::{
    BlockDirtyBitmapOrStr, BlockExportOptions, BlockExportOptionsNbd, BlockExportOptionsU,
    BlockdevDetectZeroesOptions, InetSocketAddress, SocketAddress, SocketAddressU,
    UnixSocketAddress,
};

#[cfg(target_os = "linux")]
mod kernel;
mod list;
#[cfg(unix)]
pub mod os;

pub use list::format_export_list;

/// `SOCKET_PATH`.
const SOCKET_PATH: &str = "/var/lock/qemu-nbd-";

const OPT_CACHE: i32 = 256;
const OPT_AIO: i32 = 257;
const OPT_DISCARD: i32 = 258;
const OPT_DETECT_ZEROES: i32 = 259;
const OPT_OBJECT: i32 = 260;
const OPT_TLSCREDS: i32 = 261;
const OPT_IMAGE_OPTS: i32 = 262;
const OPT_FORK: i32 = 263;
const OPT_TLSAUTHZ: i32 = 264;
const OPT_PID_FILE: i32 = 265;
const OPT_SELINUX_LABEL: i32 = 266;
const OPT_TLSHOSTNAME: i32 = 267;
const OPT_HANDSHAKE_LIMIT: i32 = 268;

/// `QEMU_HELP_BOTTOM`.
const QEMU_HELP_BOTTOM: &str = "See <https://qemu.org/contribute/report-a-bug> for how to report \
                                bugs.\nMore information on the QEMU project at \
                                <https://qemu.org>.";

/// `SNAPSHOT_OPT_BASE`.
const SNAPSHOT_OPT_BASE: &str = "snapshot.";

/// `internal_snapshot_opts`.
const INTERNAL_SNAPSHOT_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("snapshot.id", QemuOptType::String).help("snapshot id"),
    QemuOptDesc::new("snapshot.name", QemuOptType::String).help("snapshot name"),
];

const SOPT: &str = "hVb:o:p:rsnc:dvk:e:f:tl:x:T:D:AB:L";

const LONGS: &[LongOpt] = &[
    lopt("help", HasArg::No, b'h' as i32),
    lopt("version", HasArg::No, b'V' as i32),
    lopt("bind", HasArg::Required, b'b' as i32),
    lopt("port", HasArg::Required, b'p' as i32),
    lopt("socket", HasArg::Required, b'k' as i32),
    lopt("offset", HasArg::Required, b'o' as i32),
    lopt("read-only", HasArg::No, b'r' as i32),
    lopt("allocation-depth", HasArg::No, b'A' as i32),
    lopt("bitmap", HasArg::Required, b'B' as i32),
    lopt("connect", HasArg::Required, b'c' as i32),
    lopt("disconnect", HasArg::No, b'd' as i32),
    lopt("list", HasArg::No, b'L' as i32),
    lopt("snapshot", HasArg::No, b's' as i32),
    lopt("load-snapshot", HasArg::Required, b'l' as i32),
    lopt("nocache", HasArg::No, b'n' as i32),
    lopt("cache", HasArg::Required, OPT_CACHE),
    lopt("aio", HasArg::Required, OPT_AIO),
    lopt("discard", HasArg::Required, OPT_DISCARD),
    lopt("detect-zeroes", HasArg::Required, OPT_DETECT_ZEROES),
    lopt("shared", HasArg::Required, b'e' as i32),
    lopt("format", HasArg::Required, b'f' as i32),
    lopt("persistent", HasArg::No, b't' as i32),
    lopt("verbose", HasArg::No, b'v' as i32),
    lopt("object", HasArg::Required, OPT_OBJECT),
    lopt("export-name", HasArg::Required, b'x' as i32),
    lopt("description", HasArg::Required, b'D' as i32),
    lopt("handshake-limit", HasArg::Required, OPT_HANDSHAKE_LIMIT),
    lopt("tls-creds", HasArg::Required, OPT_TLSCREDS),
    lopt("tls-hostname", HasArg::Required, OPT_TLSHOSTNAME),
    lopt("tls-authz", HasArg::Required, OPT_TLSAUTHZ),
    lopt("image-opts", HasArg::No, OPT_IMAGE_OPTS),
    lopt("trace", HasArg::Required, b'T' as i32),
    lopt("fork", HasArg::No, OPT_FORK),
    lopt("pid-file", HasArg::Required, OPT_PID_FILE),
    lopt("selinux-label", HasArg::Required, OPT_SELINUX_LABEL),
];

/// Writes to standard output the way `printf()` does, ignoring a closed pipe.
fn out(s: &str) {
    let mut o = std::io::stdout();
    let _ = o.write_all(s.as_bytes());
    let _ = o.flush();
}

/// `usage()`.
fn usage_text(name: &str) -> String {
    let kernel = if cfg!(target_os = "linux") {
        "\nKernel NBD client support:\n  -c, --connect=DEV         connect FILE to the local NBD \
         device DEV\n  -d, --disconnect          disconnect the specified device\n"
    } else {
        ""
    };
    format!(
        "Usage: {name} [OPTIONS] FILE\n  or:  {name} -L [OPTIONS]\nQEMU Disk Network Block \
         Device Utility\n\n  -h, --help                display this help and exit\n  -V, \
         --version             output version information and exit\n\nConnection \
         properties:\n  -p, --port=PORT           port to listen on (default \
         `{NBD_DEFAULT_PORT}')\n  -b, --bind=IFACE          interface to bind to (default \
         `0.0.0.0')\n  -k, --socket=PATH         path to the unix socket\n                     \
         \x20      (default '{SOCKET_PATH}DEVICE')\n  -e, --shared=NUM          device can be \
         shared by NUM clients (default '1')\n  -t, --persistent          don't exit on the \
         last connection\n  -v, --verbose             display extra debugging information\n  \
         -x, --export-name=NAME    expose export by name (default is empty string)\n  -D, \
         --description=TEXT    export a human-readable description\n      \
         --handshake-limit=N   limit client's handshake to N seconds (default 10)\n\nExposing \
         part of the image:\n  -o, --offset=OFFSET       offset into the image\n  -A, \
         --allocation-depth    expose the allocation depth\n  -B, --bitmap=NAME         expose \
         a persistent dirty bitmap\n\nGeneral purpose options:\n  -L, --list                \
         list exports available from another NBD server\n  --object type,id=ID,...   define \
         an object such as 'secret' for providing\n                            passwords \
         and/or encryption keys\n  --tls-creds=ID            use id of an earlier --object to \
         provide TLS\n  --tls-authz=ID            use id of an earlier --object to provide\n  \
         \x20                         authorization\n  --tls-hostname=HOSTNAME   override \
         hostname used to check x509 certificate\n  -T, --trace \
         [[enable=]<pattern>][,events=<file>][,file=<file>]\n                            \
         specify tracing options\n  --fork                    fork off the server process and \
         exit the parent\n                            once the server is running\n  \
         --pid-file=PATH           store the server's process ID in the given \
         file\n{kernel}\nBlock device options:\n  -f, --format=FORMAT       set image format \
         (raw, qcow2, ...)\n  -r, --read-only           export read-only\n  -s, --snapshot    \
         \x20       use FILE as an external snapshot, create a temporary\n                     \
         \x20      file with backing_file=FILE, redirect the write to\n                        \
         \x20   the temporary one\n  -l, --load-snapshot=SNAPSHOT_PARAM\n                     \
         \x20      load an internal snapshot inside FILE and export it\n                       \
         \x20    as an read-only device, SNAPSHOT_PARAM format is\n                            \
         'snapshot.id=[ID],snapshot.name=[NAME]', or\n                            \
         '[ID_OR_NAME]'\n  -n, --nocache             disable host cache\n      --cache=MODE \
         \x20        set cache mode used to access the disk image, the\n                       \
         \x20    valid options are: 'none', 'writeback' (default),\n                           \
         \x20'writethrough', 'directsync' and 'unsafe'\n      --aio=MODE            set AIO \
         mode (native, io_uring or threads)\n      --discard=MODE        set discard mode \
         (ignore, unmap)\n      --detect-zeroes=MODE  set detect-zeroes mode (off, on, \
         unmap)\n      --image-opts          treat FILE as a full set of image \
         options\n\n{QEMU_HELP_BOTTOM}\n"
    )
}

/// `bdrv_parse_discard_flags()`.
fn parse_discard(mode: &str, flags: &mut OpenFlags) -> bool {
    match mode {
        "off" | "ignore" => flags.unmap = false,
        "on" | "unmap" => flags.unmap = true,
        _ => return false,
    }
    true
}

/// `bdrv_parse_aio()` in a build without io_uring.
fn parse_aio(mode: &str, flags: &mut OpenFlags) -> bool {
    match mode {
        "native" => flags.native_aio = true,
        "threads" => {}
        _ => return false,
    }
    true
}

/// `qemu_strtoi(s, NULL, 0, &v)`.
fn strtoi(s: &str) -> Option<i32> {
    let (v, _) = ruvm_qapi::cutils::strtoi64(s, 0, true).ok()?;
    i32::try_from(v).ok()
}

/// `nbd_build_socket_address()`.
fn build_socket_address(sockpath: Option<&str>, bindto: &str, port: Option<&str>) -> SocketAddress {
    let u = match sockpath {
        // Linux adds `abstract` and `tight` to the type; they keep their defaults.
        #[allow(clippy::needless_update)]
        Some(p) => SocketAddressU::Unix(UnixSocketAddress {
            path: p.to_string(),
            ..UnixSocketAddress::default()
        }),
        None => SocketAddressU::Inet(InetSocketAddress {
            host: bindto.to_string(),
            port: port.map_or_else(|| NBD_DEFAULT_PORT.to_string(), str::to_string),
            ..InetSocketAddress::default()
        }),
    };
    SocketAddress { u }
}

/// `nbd_get_tls_creds()`. No object type is TLS credentials, so this always fails.
fn get_tls_creds(id: &str) -> Error {
    let root = ruvm_qom::Registry::global().objects_root();
    match root.resolve_path_component(id) {
        None => Error::generic(format!("No TLS credentials with id '{id}'")),
        Some(_) => Error::generic(format!("Object with id '{id}' is not TLS credentials")),
    }
}

/// `socket_activation_validate_opts()`.
fn socket_activation_validate_opts(
    device: Option<&str>,
    sockpath: Option<&str>,
    address: Option<&str>,
    port: Option<&str>,
    selinux: Option<&str>,
    list: bool,
) -> Option<&'static str> {
    if device.is_some() {
        return Some("NBD device can't be set when using socket activation");
    }
    if sockpath.is_some() {
        return Some("Unix socket can't be set when using socket activation");
    }
    if address.is_some() {
        return Some("The interface can't be set when using socket activation");
    }
    if port.is_some() {
        return Some("TCP port number can't be set when using socket activation");
    }
    if selinux.is_some() {
        return Some("SELinux label can't be set when using socket activation");
    }
    if list {
        return Some("List mode is incompatible with socket activation");
    }
    None
}

/// `state`: whether the server runs or should end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunState {
    Running,
    Terminate,
}

/// What the connection threads, the signal thread and the kernel client share with `main()`.
#[derive(Debug)]
struct Shared {
    state: Mutex<RunState>,
    cond: Condvar,
    persistent: bool,
    server: Mutex<Option<NbdServer>>,
}

impl Shared {
    fn terminate(&self) {
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = RunState::Terminate;
        self.cond.notify_all();
    }

    fn wait(&self) {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        while *st == RunState::Running {
            st = self.cond.wait(st).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// `nbd_client_closed()`.
    fn client_closed(&self, negotiated: bool) {
        if !negotiated || self.persistent {
            return;
        }
        let server = self.server.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if server.is_some_and(|s| s.connections() == 0) {
            self.terminate();
        }
    }
}

/// The options that matter after parsing.
#[derive(Debug, Default)]
struct Args {
    dev_offset: u64,
    readonly: bool,
    disconnect: bool,
    bindto: Option<String>,
    port: Option<String>,
    sockpath: Option<String>,
    sn_opts: Option<(Option<String>, Option<String>)>,
    sn_id_or_name: Option<String>,
    flags: OpenFlags,
    snapshot: bool,
    seen_cache: bool,
    seen_discard: bool,
    seen_aio: bool,
    fmt: Option<String>,
    detect_zeroes: Option<BlockdevDetectZeroesOptions>,
    export_name: Option<String>,
    export_description: Option<String>,
    bitmaps: Vec<String>,
    alloc_depth: bool,
    tlscredsid: Option<String>,
    tlshostname: Option<String>,
    tlsauthz: Option<String>,
    image_opts: bool,
    writethrough: bool,
    list: bool,
    pid_file: Option<String>,
    selinux_label: Option<String>,
    device: Option<String>,
    fork_process: bool,
    verbose: bool,
    persistent: bool,
    shared: i32,
    handshake_limit: i32,
}

/// qemu-nbd's `main()`. `argv0` is how the program was called, `args` what came after it and
/// `version` the text of `-V` (`<argv0> 11.1.0 ...`, the author line and the copyright). Returns
/// the exit status.
pub fn main(argv0: &str, args: &[String], version: &str) -> u8 {
    let prgname = argv0.rsplit(['/', '\\']).next().unwrap_or(argv0);
    set_program_name(prgname);
    register_object_types();
    let r = run(argv0, args, version);
    let _ = std::io::stdout().flush();
    match r {
        Ok(code) => code,
        Err(Exit(code)) => code,
    }
}

fn fail(msg: &str) -> Exit {
    error_report(msg);
    Exit(1)
}

fn parse_args(argv0: &str, args: &[String], version: &str) -> Flow<(Args, Vec<String>)> {
    let mut argv = vec![argv0.to_string()];
    argv.extend_from_slice(args);
    let mut g = Getopt::new(argv, SOPT, LONGS);
    let mut a = Args {
        flags: OpenFlags { rdwr: true, ..OpenFlags::default() },
        shared: 1,
        handshake_limit: NBD_DEFAULT_HANDSHAKE_MAX_SECS as i32,
        ..Args::default()
    };
    let mut sn_list = QemuOptsList::new("snapshot", INTERNAL_SNAPSHOT_OPTS);
    while let Some(o) = g.next() {
        let (c, arg) = match o {
            Opt::Opt(c, arg) => (c, arg.unwrap_or_default()),
            _ => {
                return Err(fail(&format!("Try `{argv0} --help' for more information.")));
            }
        };
        match c {
            OPT_CACHE => set_cache(&mut a, &arg)?,
            OPT_AIO => {
                if a.seen_aio {
                    return Err(fail("--aio can only be specified once"));
                }
                a.seen_aio = true;
                if !parse_aio(&arg, &mut a.flags) {
                    return Err(fail(&format!("Invalid aio mode '{arg}'")));
                }
            }
            OPT_DISCARD => {
                if a.seen_discard {
                    return Err(fail("--discard can only be specified once"));
                }
                a.seen_discard = true;
                if !parse_discard(&arg, &mut a.flags) {
                    return Err(fail(&format!("Invalid discard mode `{arg}'")));
                }
            }
            OPT_DETECT_ZEROES => {
                let Some(dz) = BlockdevDetectZeroesOptions::from_name(&arg) else {
                    return Err(fail(&format!(
                        "Failed to parse detect_zeroes mode: invalid parameter value: {arg}"
                    )));
                };
                if dz == BlockdevDetectZeroesOptions::Unmap && !a.flags.unmap {
                    return Err(fail(
                        "setting detect-zeroes to unmap is not allowed without setting discard \
                         operation to unmap",
                    ));
                }
                a.detect_zeroes = Some(dz);
            }
            OPT_OBJECT => object_add(&arg)?,
            OPT_TLSCREDS => a.tlscredsid = Some(arg),
            OPT_TLSHOSTNAME => a.tlshostname = Some(arg),
            OPT_IMAGE_OPTS => a.image_opts = true,
            OPT_TLSAUTHZ => a.tlsauthz = Some(arg),
            OPT_FORK => a.fork_process = true,
            OPT_PID_FILE => a.pid_file = Some(arg),
            OPT_SELINUX_LABEL => a.selinux_label = Some(arg),
            OPT_HANDSHAKE_LIMIT => match strtoi(&arg) {
                Some(v) if v >= 0 => a.handshake_limit = v,
                _ => return Err(fail(&format!("Invalid handshake limit '{arg}'"))),
            },
            _ => match u8::try_from(c).map(char::from).unwrap_or('?') {
                's' => a.snapshot = true,
                'n' => set_cache(&mut a, "none")?,
                'b' => a.bindto = Some(arg),
                'p' => a.port = Some(arg),
                'o' => match ruvm_qapi::cutils::strtou64(&arg, 0, true) {
                    Ok((v, _)) => a.dev_offset = v,
                    Err(_) => return Err(fail(&format!("Invalid offset '{arg}'"))),
                },
                'l' | 'r' => {
                    if c == i32::from(b'l') {
                        if arg.starts_with(SNAPSHOT_OPT_BASE) {
                            let Some(o) = sn_list.parse_noisily(&arg, false) else {
                                return Err(fail(&format!(
                                    "Failed in parsing snapshot param `{arg}'"
                                )));
                            };
                            let id = o.get("snapshot.id").map(str::to_string);
                            let name = o.get("snapshot.name").map(str::to_string);
                            a.sn_opts = Some((id, name));
                        } else {
                            a.sn_id_or_name = Some(arg);
                        }
                    }
                    a.readonly = true;
                    a.flags.rdwr = false;
                }
                'A' => a.alloc_depth = true,
                'B' => a.bitmaps.insert(0, arg),
                'k' => {
                    if !arg.starts_with('/') {
                        return Err(fail("socket path must be absolute"));
                    }
                    a.sockpath = Some(arg);
                }
                'd' => a.disconnect = true,
                'c' => a.device = Some(arg),
                'e' => match strtoi(&arg) {
                    Some(v) if v >= 0 => a.shared = v,
                    _ => return Err(fail(&format!("Invalid shared device number '{arg}'"))),
                },
                'f' => a.fmt = Some(arg),
                't' => a.persistent = true,
                'x' => {
                    if arg.len() > NBD_MAX_STRING_SIZE {
                        return Err(fail(&format!("export name '{arg}' too long")));
                    }
                    a.export_name = Some(arg);
                }
                'D' => {
                    if arg.len() > NBD_MAX_STRING_SIZE {
                        return Err(fail(&format!("export description '{arg}' too long")));
                    }
                    a.export_description = Some(arg);
                }
                'v' => a.verbose = true,
                'V' => {
                    out(version);
                    return Err(Exit(0));
                }
                'h' => {
                    out(&usage_text(argv0));
                    return Err(Exit(0));
                }
                'L' => a.list = true,
                // -T: there are no trace events.
                _ => {}
            },
        }
    }
    Ok((a, g.rest().to_vec()))
}

fn set_cache(a: &mut Args, mode: &str) -> Flow<()> {
    if a.seen_cache {
        return Err(fail("-n and --cache can only be specified once"));
    }
    a.seen_cache = true;
    let Some(cm) = parse_cache_mode(mode) else {
        return Err(fail(&format!("Invalid cache mode `{mode}'")));
    };
    cm.apply(&mut a.flags);
    a.writethrough = cm.writethrough;
    Ok(())
}

fn run(argv0: &str, args: &[String], version: &str) -> Flow<u8> {
    let (mut a, rest) = parse_args(argv0, args, version)?;

    if a.list {
        if !rest.is_empty() {
            return Err(fail("List mode is incompatible with a file name"));
        }
        if a.export_name.is_some()
            || a.export_description.is_some()
            || a.dev_offset != 0
            || a.device.is_some()
            || a.disconnect
            || a.fmt.is_some()
            || a.sn_id_or_name.is_some()
            || !a.bitmaps.is_empty()
            || a.alloc_depth
            || a.seen_aio
            || a.seen_discard
            || a.seen_cache
        {
            return Err(fail("List mode is incompatible with per-device settings"));
        }
        if a.fork_process {
            return Err(fail("List mode is incompatible with forking"));
        }
    } else if rest.len() != 1 {
        error_report("Invalid number of arguments");
        eprintln!("Try `{argv0} --help' for more information.");
        return Err(Exit(1));
    } else if a.export_name.is_none() {
        a.export_name = Some(String::new());
    }

    #[cfg(unix)]
    let activation = os::check_socket_activation().map_err(|m| fail(&m))?;
    #[cfg(not(unix))]
    let activation: Vec<()> = Vec::new();
    if activation.is_empty() {
        if a.sockpath.is_none() {
            a.bindto.get_or_insert_with(|| "0.0.0.0".to_string());
            a.port.get_or_insert_with(|| NBD_DEFAULT_PORT.to_string());
        }
    } else {
        if let Some(m) = socket_activation_validate_opts(
            a.device.as_deref(),
            a.sockpath.as_deref(),
            a.bindto.as_deref(),
            a.port.as_deref(),
            a.selinux_label.as_deref(),
            a.list,
        ) {
            return Err(fail(m));
        }
        if activation.len() > 1 {
            return Err(fail("qemu-nbd does not support socket activation with LISTEN_FDS > 1"));
        }
    }

    if let Some(id) = &a.tlscredsid {
        if a.device.is_some() {
            return Err(fail("TLS is not supported with a host device"));
        }
        if a.tlsauthz.is_some() && a.list {
            return Err(fail("TLS authorization is incompatible with export list"));
        }
        if a.tlshostname.is_some() && !a.list {
            return Err(fail("TLS hostname is only supported with export list"));
        }
        report_error(&get_tls_creds(id).prepend("Failed to get TLS creds: "));
        return Err(Exit(1));
    }
    if a.tlsauthz.is_some() {
        return Err(fail("--tls-authz is not permitted without --tls-creds"));
    }
    if a.tlshostname.is_some() {
        return Err(fail("--tls-hostname is not permitted without --tls-creds"));
    }
    if a.selinux_label.is_some() {
        return Err(fail("SELinux support not enabled in this binary"));
    }

    if a.list {
        let saddr = build_socket_address(
            a.sockpath.as_deref(),
            a.bindto.as_deref().unwrap_or("0.0.0.0"),
            a.port.as_deref(),
        );
        return Ok(client_list(&saddr));
    }

    let srcpath = rest[0].clone();
    #[cfg(not(target_os = "linux"))]
    if a.disconnect || a.device.is_some() {
        return Err(fail("Kernel /dev/nbdN support not available"));
    }
    #[cfg(target_os = "linux")]
    if a.disconnect {
        let dev = match std::fs::OpenOptions::new().read(true).write(true).open(&srcpath) {
            Ok(f) => f,
            Err(e) => {
                return Err(fail(&format!(
                    "Cannot open {srcpath}: {}",
                    ruvm_base::error::strerror(&e)
                )));
            }
        };
        kernel::nbd_disconnect(&dev);
        drop(dev);
        out(&format!("{srcpath} disconnected\n"));
        return Ok(0);
    }

    // The descriptor that takes standard error back once the server runs: the original one
    // with --verbose, else standard output.
    #[cfg(unix)]
    let mut old_stderr: Option<std::os::fd::OwnedFd> = None;
    if (a.device.is_some() && !a.verbose) || a.fork_process {
        #[cfg(unix)]
        {
            old_stderr = fork_with_pipe(a.verbose)?;
        }
        #[cfg(not(unix))]
        return Err(fail("Unable to fork into background on Windows hosts"));
    }

    if let Some(dev) = &a.device {
        if a.sockpath.is_none() {
            let base = dev.rsplit('/').next().unwrap_or(dev);
            a.sockpath = Some(format!("{SOCKET_PATH}{base}"));
        }
    }

    let handshake = a.handshake_limit as u32;
    let shared_n = a.shared as u32;
    let saddr = build_socket_address(
        a.sockpath.as_deref(),
        a.bindto.as_deref().unwrap_or("0.0.0.0"),
        a.port.as_deref(),
    );
    #[cfg(unix)]
    let activated = activation.into_iter().next();
    #[cfg(unix)]
    let server = match &activated {
        Some(_) => NbdServer::detached(handshake, shared_n),
        None => NbdServer::bind_addr(&saddr, handshake, None, shared_n).map_err(|e| fatal(&e))?,
    };
    #[cfg(not(unix))]
    let server = NbdServer::bind_addr(&saddr, handshake, None, shared_n).map_err(|e| fatal(&e))?;

    let opened = open_image(&a, &srcpath)?;
    let blk = opened.blk.clone();
    let node_name = blk.node_name().unwrap_or_default();

    blk.set_enable_write_cache(!a.writethrough);

    let loaded = if let Some((id, name)) = &a.sn_opts {
        graph().snapshot_load_tmp(&node_name, id.as_deref(), name.as_deref())
    } else if let Some(id_or_name) = &a.sn_id_or_name {
        graph().snapshot_load_tmp_by_id_or_name(&node_name, id_or_name)
    } else {
        Ok(())
    };
    if let Err(e) = loaded {
        report_error(&e.prepend("Failed to load snapshot: "));
        return Err(Exit(1));
    }

    let export = BlockExportOptions {
        id: "qemu-nbd-export".to_string(),
        node_name: node_name.clone(),
        writethrough: Some(a.writethrough),
        writable: Some(!a.readonly),
        u: BlockExportOptionsU::Nbd(BlockExportOptionsNbd {
            name: a.export_name.clone(),
            description: a.export_description.clone(),
            bitmaps: if a.bitmaps.is_empty() {
                None
            } else {
                Some(a.bitmaps.iter().cloned().map(BlockDirtyBitmapOrStr::Local).collect())
            },
            allocation_depth: if a.alloc_depth { Some(true) } else { None },
        }),
        ..BlockExportOptions::default()
    };
    server.export_add(graph(), &export).map_err(|e| fatal(&e))?;

    let shared = Arc::new(Shared {
        state: Mutex::new(RunState::Running),
        cond: Condvar::new(),
        persistent: a.persistent,
        server: Mutex::new(Some(server.clone())),
    });
    {
        let s2 = shared.clone();
        server.set_close_notify(move |negotiated| s2.client_closed(negotiated));
    }
    #[cfg(unix)]
    {
        let s2 = shared.clone();
        let _ = ruvm_sys::signal::on_termination(move |_| s2.terminate());
    }

    #[cfg(target_os = "linux")]
    let client_thread = match &a.device {
        Some(dev) => {
            let job = KernelClient {
                device: dev.clone(),
                srcpath: srcpath.clone(),
                saddr: saddr.clone(),
                old_stderr: match &old_stderr {
                    Some(fd) => Some(fd.try_clone().map_err(|e| {
                        fail(&format!(
                            "Could not dup original stderr: {}",
                            ruvm_base::error::strerror(&e)
                        ))
                    })?),
                    None => None,
                },
                release: !(a.verbose && !a.fork_process),
                shared: shared.clone(),
            };
            match std::thread::Builder::new().name("nbd-client".into()).spawn(move || job.run()) {
                Ok(t) => Some(t),
                Err(e) => {
                    return Err(fail(&format!(
                        "Failed to create client thread: {}",
                        ruvm_base::error::strerror(&e)
                    )));
                }
            }
        }
        None => None,
    };

    #[cfg(unix)]
    match activated {
        Some(fd) => accept_activated(fd, &server, shared_n).map_err(|e| fatal(&e))?,
        None => server.start_accepting().map_err(|e| fatal(&e))?,
    }
    #[cfg(not(unix))]
    server.start_accepting().map_err(|e| fatal(&e))?;

    #[cfg(unix)]
    let _pidfile = match &a.pid_file {
        Some(p) => Some(os::write_pidfile(p).map_err(|e| fatal(&e))?),
        None => None,
    };
    #[cfg(not(unix))]
    if let Some(p) = &a.pid_file {
        std::fs::write(p, format!("{}\n", std::process::id()))
            .map_err(|e| fatal(&Error::from_io(format!("Could not create '{p}'"), e)))?;
    }

    if let Err(e) = std::env::set_current_dir("/") {
        return Err(fail(&format!(
            "Could not chdir to root directory: {}",
            ruvm_base::error::strerror(&e)
        )));
    }

    #[cfg(unix)]
    if a.fork_process {
        release_pipe(old_stderr.take())?;
    }

    shared.wait();
    // blk_exp_close_all()
    let server = shared.server.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(s) = server {
        s.stop();
    }
    drop(blk);
    drop(opened);
    if let Some(p) = &a.sockpath {
        let _ = std::fs::remove_file(p);
    }

    #[cfg(target_os = "linux")]
    if let Some(t) = client_thread {
        return Ok(t.join().unwrap_or(1));
    }
    Ok(0)
}

/// `error_report_err()` and `exit(1)`.
fn fatal(e: &Error) -> Exit {
    report_error(e);
    Exit(1)
}

/// `qemu_nbd_client_list()`.
fn client_list(saddr: &SocketAddress) -> u8 {
    let mut s = match ruvm_block::nbd::nbd_socket_connect(saddr) {
        Ok(s) => s,
        Err(e) => {
            report_error(&e);
            return 1;
        }
    };
    match ruvm_block::nbd::nbd_receive_export_list(&mut s) {
        Ok(list) => {
            out(&format_export_list(&list));
            0
        }
        Err(e) => {
            report_error(&e);
            1
        }
    }
}

/// The image as it is exported, and what has to live as long as it.
struct Opened {
    blk: Arc<BlockBackend>,
    /// The temporary overlay of `--snapshot`, removed at exit where it could not be removed
    /// while open.
    temp: Option<String>,
}

impl Drop for Opened {
    fn drop(&mut self) {
        if let Some(t) = self.temp.take() {
            let _ = std::fs::remove_file(t);
        }
    }
}

/// Opens the image the way `main()` does it with `blk_new_open()`, including `--snapshot`,
/// `--offset` and `--detect-zeroes`.
fn open_image(a: &Args, srcpath: &str) -> Flow<Opened> {
    let dz = a.detect_zeroes.map(|d| d.as_str().to_string());
    let top_is_main = !a.snapshot && a.dev_offset == 0;
    let mut options = if a.image_opts {
        if a.fmt.is_some() {
            return Err(fail("--image-opts and -f are mutually exclusive"));
        }
        let mut file_opts = QemuOptsList::new("file", &[]).with_implied_opt_name("file");
        let Some(o) = file_opts.parse_noisily(srcpath, true) else {
            return Err(Exit(1));
        };
        o.to_qdict()
    } else {
        let mut d = QDict::new();
        if let Some(f) = &a.fmt {
            d.put("driver", f.as_str());
        }
        d
    };
    if top_is_main {
        if let Some(dz) = &dz {
            options.put("detect-zeroes", dz.as_str());
        }
    }
    let filename = if a.image_opts { None } else { Some(srcpath) };
    let open_fail = |e: Error| fatal(&e.prepend(format!("Failed to blk_new_open '{srcpath}': ")));

    let mut opened = if a.snapshot {
        // bdrv_open_inherit() with BDRV_O_SNAPSHOT: the image itself is opened read-only
        // and a temporary qcow2 overlay on top takes the writes.
        let base_flags = OpenFlags { rdwr: false, ..a.flags };
        let base = graph().blk_new_open(filename, options, base_flags).map_err(open_fail)?;
        let base_node = base.node_name().unwrap_or_default();
        let (tmp, overlay) =
            temp_snapshot(&base, &base_node, a, dz.as_deref()).map_err(open_fail)?;
        drop(base);
        let mut o = Opened { blk: overlay, temp: Some(tmp) };
        // The overlay is open, so its name can go now; it goes away with the last descriptor.
        #[cfg(unix)]
        if let Some(t) = o.temp.take() {
            let _ = std::fs::remove_file(t);
        }
        #[cfg(not(unix))]
        let _ = &mut o;
        o
    } else {
        let blk = graph().blk_new_open(filename, options, a.flags).map_err(open_fail)?;
        Opened { blk, temp: None }
    };

    if a.dev_offset != 0 {
        let mut raw = QDict::new();
        raw.put("driver", "raw");
        raw.put("file", opened.blk.node_name().unwrap_or_default());
        raw.put("offset", a.dev_offset as i64);
        if let Some(dz) = &dz {
            raw.put("detect-zeroes", dz.as_str());
        }
        let blk = graph().blk_new_open(None, raw, a.flags).map_err(|e| fatal(&e))?;
        opened.blk = blk;
    }
    Ok(opened)
}

/// `create_tmp_file()`: a new empty file `<tmpdir>/vl.XXXXXX`.
fn create_tmp_file() -> Result<String, Error> {
    let mut dir = std::env::temp_dir().to_string_lossy().trim_end_matches('/').to_string();
    if cfg!(unix) && dir == "/tmp" {
        dir = "/var/tmp".to_string();
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ u64::from(std::process::id()) << 32;
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut last = None;
    for attempt in 0u64..100 {
        let mut v = seed.wrapping_add(attempt.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut suffix = String::new();
        for _ in 0..6 {
            suffix.push(char::from(CHARS[(v % CHARS.len() as u64) as usize]));
            v /= CHARS.len() as u64;
        }
        let name = format!("{dir}/vl.{suffix}");
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&name) {
            Ok(_) => return Ok(name),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => last = Some((name, e)),
            Err(e) => {
                return Err(Error::from_io(format!("Could not open temporary file '{name}'"), e));
            }
        }
    }
    let (name, e) = last.expect("tried at least once");
    Err(Error::from_io(format!("Could not open temporary file '{name}'"), e))
}

/// `bdrv_append_temp_snapshot()`.
fn temp_snapshot(
    base: &BlockBackend,
    base_node: &str,
    a: &Args,
    dz: Option<&str>,
) -> Result<(String, Arc<BlockBackend>), Error> {
    let size = base.getlength().map_err(|e| Error::from_io("Could not get image size", e))?;
    let tmp = create_tmp_file()?;
    let mut o = QDict::new();
    o.put("size", size.to_string());
    if let Err(e) = graph().create_image("qcow2", &tmp, &mut o) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e.prepend(format!("Could not create temporary overlay '{tmp}': ")));
    }
    let mut d = QDict::new();
    d.put("driver", "qcow2");
    d.put("file.driver", "file");
    d.put("file.filename", tmp.as_str());
    d.put("backing", base_node);
    // bdrv_temp_snapshot_options()
    d.put("cache.direct", "off");
    d.put("cache.no-flush", "on");
    if let Some(dz) = dz {
        if a.dev_offset == 0 {
            d.put("detect-zeroes", dz);
        }
    }
    let flags = OpenFlags { nocache: false, no_flush: true, ..a.flags };
    match graph().blk_new_open(None, d, flags) {
        Ok(b) => Ok((tmp, b)),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The fork of `main()` for `--fork` and for `--connect` without `--verbose`: the parent
/// copies what the child writes to standard error until the child lets go of it, then exits
/// with 1 if there was anything. Returns the duplicate of standard error to restore later
/// with `--verbose`.
#[cfg(unix)]
fn fork_with_pipe(verbose: bool) -> Flow<Option<std::os::fd::OwnedFd>> {
    use std::io::Read;

    let strerror = ruvm_base::error::strerror;
    let (r, w) = match os::cloexec_pipe() {
        Ok(p) => p,
        Err(e) => {
            return Err(fail(&format!("Error setting up communication pipe: {}", strerror(&e))));
        }
    };
    match os::fork() {
        Err(e) => Err(fail(&format!("Failed to fork: {}", strerror(&e)))),
        Ok(Some(_)) => {
            drop(w);
            let mut r = std::fs::File::from(r);
            let mut buf = [0u8; 1024];
            let mut errors = false;
            loop {
                match r.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        errors = true;
                        if std::io::stderr().write_all(&buf[..n]).is_err() {
                            return Err(Exit(1));
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(e) => {
                        return Err(fail(&format!("Cannot read from daemon: {}", strerror(&e))));
                    }
                }
            }
            Err(Exit(u8::from(errors)))
        }
        Ok(None) => {
            drop(r);
            let old = if verbose {
                match os::dup_stderr() {
                    Ok(fd) => Some(fd),
                    Err(e) => {
                        return Err(fail(&format!(
                            "Could not dup original stderr: {}",
                            strerror(&e)
                        )));
                    }
                }
            } else {
                None
            };
            let daemon = os::qemu_daemon();
            if let Err(e) = os::dup2_stderr(&w) {
                let msg = format!(
                    "{}: Failed to link stderr to the pipe: {}\n",
                    ruvm_base::report::program_name(),
                    strerror(&e)
                );
                let _ = std::fs::File::from(w).write_all(msg.as_bytes());
                return Err(Exit(1));
            }
            if let Err(e) = daemon {
                return Err(fail(&format!("Failed to daemonize: {}", strerror(&e))));
            }
            drop(w);
            Ok(old)
        }
    }
}

/// `nbd_client_release_pipe()`: point standard error back at the original one (or at
/// standard output), which lets the parent exit.
#[cfg(unix)]
fn release_pipe(old_stderr: Option<std::os::fd::OwnedFd>) -> Flow<()> {
    let r = match &old_stderr {
        Some(fd) => os::dup2_stderr(fd),
        None => os::stdout_to_stderr(),
    };
    if let Err(e) = r {
        return Err(fail(&format!(
            "Could not release pipe to parent: {}",
            ruvm_base::error::strerror(&e)
        )));
    }
    Ok(())
}

/// Accepts connections on the socket systemd passed in and hands them to `server`, as the
/// `QIONetListener` of `main()` does with socket activation. Like `nbd_update_server_watch()`
/// it stops accepting while `limit` clients (0 for no limit) are connected.
#[cfg(unix)]
fn accept_activated(fd: std::os::fd::OwnedFd, server: &NbdServer, limit: u32) -> Result<(), Error> {
    use ruvm_block::nbd::NbdStream;

    let unix = rustix::net::getsockname(&fd)
        .map(|a| a.address_family() == rustix::net::AddressFamily::UNIX)
        .unwrap_or(false);
    let s = server.clone();
    let wait_for_room = move |s: &NbdServer| {
        while limit != 0 && s.connections() >= limit as usize {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    };
    let accept = move || {
        if unix {
            let l = std::os::unix::net::UnixListener::from(fd);
            loop {
                wait_for_room(&s);
                if let Ok((c, _)) = l.accept() {
                    s.serve_stream(NbdStream::Unix(c));
                }
            }
        } else {
            let l = std::net::TcpListener::from(fd);
            loop {
                wait_for_room(&s);
                if let Ok((c, _)) = l.accept() {
                    let _ = c.set_nodelay(true);
                    s.serve_stream(NbdStream::Tcp(c));
                }
            }
        }
    };
    std::thread::Builder::new()
        .name("nbd-listener".into())
        .spawn(accept)
        .map_err(|e| Error::from_io("Failed to use socket activation", e))?;
    Ok(())
}

/// `nbd_client_thread()`: connects `/dev/nbdN` to the server through the kernel.
#[cfg(target_os = "linux")]
struct KernelClient {
    device: String,
    srcpath: String,
    saddr: SocketAddress,
    old_stderr: Option<std::os::fd::OwnedFd>,
    /// Whether to let the parent go once the device is connected, rather than print that.
    release: bool,
    shared: Arc<Shared>,
}

#[cfg(target_os = "linux")]
impl KernelClient {
    fn run(self) -> u8 {
        let r = self.connect();
        // kill(getpid(), SIGTERM)
        self.shared.terminate();
        r
    }

    fn connect(&self) -> u8 {
        use std::os::fd::AsRawFd;

        use ruvm_block::nbd::{NbdExportInfo, NbdMode, NbdStream};

        let mut sock = match ruvm_block::nbd::nbd_socket_connect(&self.saddr) {
            Ok(s) => s,
            Err(e) => {
                report_error(&e);
                return 1;
            }
        };
        let mut info = NbdExportInfo { mode: NbdMode::Simple, ..NbdExportInfo::default() };
        if let Err(e) = ruvm_block::nbd::nbd_receive_negotiate(&mut sock, &mut info) {
            report_error(&e);
            return 1;
        }
        let dev = match std::fs::OpenOptions::new().read(true).write(true).open(&self.device) {
            Ok(f) => f,
            Err(e) => {
                error_report(&format!(
                    "Failed to open {}: {}",
                    self.device,
                    ruvm_base::error::strerror(&e)
                ));
                return 1;
            }
        };
        let raw = match &sock {
            NbdStream::Tcp(s) => s.as_raw_fd(),
            NbdStream::Unix(s) => s.as_raw_fd(),
        };
        if let Err(e) = kernel::nbd_init(&dev, raw, &info) {
            report_error(&e);
            return 1;
        }
        // Linux only needs an open() to read the partition table again.
        let device = self.device.clone();
        let _ = std::thread::Builder::new().name("show-parts".into()).spawn(move || {
            let _ = std::fs::OpenOptions::new().read(true).write(true).open(device);
        });
        if !self.release {
            eprintln!("NBD device {} is now connected to {}", self.device, self.srcpath);
        } else {
            let old = self.old_stderr.as_ref().and_then(|f| f.try_clone().ok());
            if release_pipe(old).is_err() {
                std::process::exit(1);
            }
        }
        let r = kernel::nbd_client(&dev);
        drop(sock);
        if r.is_err() { 1 } else { 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_has_the_kernel_part_only_on_linux() {
        let t = usage_text("qemu-nbd");
        assert!(t.starts_with("Usage: qemu-nbd [OPTIONS] FILE\n  or:  qemu-nbd -L [OPTIONS]\n"));
        assert_eq!(t.contains("Kernel NBD client support"), cfg!(target_os = "linux"));
        assert!(t.ends_with("<https://qemu.org>.\n"));
    }

    #[test]
    fn strtoi_is_int_sized() {
        assert_eq!(strtoi("0x10"), Some(16));
        assert_eq!(strtoi("4294967296"), None);
        assert_eq!(strtoi("1x"), None);
    }

    #[test]
    fn socket_addresses() {
        let a = build_socket_address(None, "0.0.0.0", None);
        let SocketAddressU::Inet(i) = a.u else { panic!() };
        assert_eq!((i.host.as_str(), i.port.as_str()), ("0.0.0.0", "10809"));
        let a = build_socket_address(Some("/tmp/s"), "x", Some("1"));
        let SocketAddressU::Unix(u) = a.u else { panic!() };
        assert_eq!(u.path, "/tmp/s");
    }
}
