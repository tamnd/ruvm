// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-storage-daemon: storage-daemon/qemu-storage-daemon.c and the block export code of
//! block/export/.
//!
//! [`main`] takes the arguments after `argv[0]` and returns the exit status. The options are
//! processed in two passes as in QEMU: `--help`, `--version`, `--daemonize` and `--pidfile`
//! first, then everything else in command line order. The daemon then serves its QMP monitors
//! until `quit` or a termination signal, and closes every export on the way out.
//!
//! The exports are in [`export`]: NBD through the block layer's NBD server, and on Linux
//! vhost-user-blk, FUSE and VDUSE. The QMP commands are in [`qmp`].
//!
//! Differences from QEMU:
//!
//! - `-T`/`--trace` is accepted and ignored: ruvm has no trace events.
//! - The QMP commands are the subset listed in [`qmp`].
//! - Monitors, exports and the NBD server run on threads of their own instead of one main
//!   loop.
//! - On Windows there is no `--daemonize` ("--daemonize not supported in this build", as in
//!   QEMU) and the PID file is written without a lock.

use std::io::Write;
use std::sync::Arc;

use ruvm_base::Error;
use ruvm_base::report::{Location, error_report, push_location, report_error, set_program_name};
use ruvm_chardev::Chardevs;
use ruvm_img::getopt::{Getopt, HasArg, LongOpt, Opt, lopt};
use ruvm_img::shared::{object_add, register_object_types};
use ruvm_img::{Exit, Flow};
use ruvm_monitor::Qmp;
use ruvm_monitor::object::monitor_new;
use ruvm_qapi::types::{BlockdevOptions, MonitorMode, MonitorOptions, NbdServerOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QValue, json, keyval::keyval_parse};
use ruvm_qom::Registry;

pub mod export;
pub mod qmp;

use export::ExportOptions;
use qmp::State;

const OPTION_BLOCKDEV: i32 = 256;
const OPTION_CHARDEV: i32 = 257;
const OPTION_DAEMONIZE: i32 = 258;
const OPTION_EXPORT: i32 = 259;
const OPTION_MONITOR: i32 = 260;
const OPTION_NBD_SERVER: i32 = 261;
const OPTION_OBJECT: i32 = 262;
const OPTION_PIDFILE: i32 = 263;

const LONGS: &[LongOpt] = &[
    lopt("blockdev", HasArg::Required, OPTION_BLOCKDEV),
    lopt("chardev", HasArg::Required, OPTION_CHARDEV),
    lopt("daemonize", HasArg::No, OPTION_DAEMONIZE),
    lopt("export", HasArg::Required, OPTION_EXPORT),
    lopt("help", HasArg::No, b'h' as i32),
    lopt("monitor", HasArg::Required, OPTION_MONITOR),
    lopt("nbd-server", HasArg::Required, OPTION_NBD_SERVER),
    lopt("object", HasArg::Required, OPTION_OBJECT),
    lopt("pidfile", HasArg::Required, OPTION_PIDFILE),
    lopt("trace", HasArg::Required, b'T' as i32),
    lopt("version", HasArg::No, b'V' as i32),
];

/// `help()`, up to the Linux only exports.
const HELP_HEAD: &[&str] = &[
    "QEMU storage daemon",
    "",
    "  -h, --help             display this help and exit",
    "  -T, --trace [[enable=]<pattern>][,events=<file>][,file=<file>]",
    "                         specify tracing options",
    "  -V, --version          output version information and exit",
    "",
    "  --blockdev [driver=]<driver>[,node-name=<N>][,discard=ignore|unmap]",
    "             [,cache.direct=on|off][,cache.no-flush=on|off]",
    "             [,read-only=on|off][,auto-read-only=on|off]",
    "             [,force-share=on|off][,detect-zeroes=on|off|unmap]",
    "             [,driver specific parameters...]",
    "                         configure a block backend",
    "",
    "  --chardev <options>    configure a character device backend",
    "                         (see the qemu(1) man page for possible options)",
    "",
    "  --daemonize            daemonize the process, and have the parent exit",
    "                         once startup is complete",
    "",
    "  --export [type=]nbd,id=<id>,node-name=<node-name>[,name=<export-name>]",
    "           [,writable=on|off][,bitmap=<name>]",
    "                         export the specified block node over NBD",
    "                         (requires --nbd-server)",
    "",
];

/// The Linux only export types, under `CONFIG_FUSE`, `CONFIG_VHOST_USER_BLK_SERVER` and
/// `CONFIG_VDUSE_BLK_EXPORT`.
const HELP_LINUX: &[&str] = &[
    "  --export [type=]fuse,id=<id>,node-name=<node-name>,mountpoint=<file>",
    "           [,growable=on|off][,writable=on|off][,allow-other=on|off|auto]",
    "                         export the specified block node over FUSE",
    "",
    "  --export [type=]vhost-user-blk,id=<id>,node-name=<node-name>,",
    "           addr.type=unix,addr.path=<socket-path>[,writable=on|off]",
    "           [,logical-block-size=<block-size>][,num-queues=<num-queues>]",
    "                         export the specified block node as a",
    "                         vhost-user-blk device over UNIX domain socket",
    "  --export [type=]vhost-user-blk,id=<id>,node-name=<node-name>,",
    "           addr.type=fd,addr.str=<fd>[,writable=on|off]",
    "           [,logical-block-size=<block-size>][,num-queues=<num-queues>]",
    "                         export the specified block node as a",
    "                         vhost-user-blk device over file descriptor",
    "",
    "  --export [type=]vduse-blk,id=<id>,node-name=<node-name>",
    "           ,name=<vduse-name>[,writable=on|off]",
    "           [,num-queues=<num-queues>][,queue-size=<queue-size>]",
    "           [,logical-block-size=<logical-block-size>]",
    "           [,serial=<serial-number>]",
    "                         export the specified block node as a",
    "                         vduse-blk device",
    "",
];

const HELP_TAIL: &[&str] = &[
    "  --monitor [chardev=]name[,mode=control][,pretty[=on|off]]",
    "                         configure a QMP monitor",
    "",
    "  --nbd-server addr.type=inet,addr.host=<host>,addr.port=<port>",
    "               [,tls-creds=<id>][,tls-authz=<id>][,max-connections=<n>]",
    "  --nbd-server addr.type=unix,addr.path=<path>",
    "               [,tls-creds=<id>][,tls-authz=<id>][,max-connections=<n>]",
    "                         start an NBD server for exporting block nodes",
    "",
    "  --object help          list object types that can be added",
    "  --object <type>,help   list properties for the given object type",
    "  --object <type>[,<property>=<value>...]",
    "                         create a new object of type <type>, setting",
    "                         properties in the order they are specified. Note",
    "                         that the 'id' property must be set.",
    "                         See the qemu(1) man page for documentation of the",
    "                         objects that can be added.",
    "",
    "  --pidfile <path>       write process ID to a file after startup",
    "",
    "See <https://qemu.org/contribute/report-a-bug> for how to report bugs.",
    "More information on the QEMU project at <https://qemu.org>.",
];

/// The text of `--help`.
pub fn help_text(prgname: &str) -> String {
    let mut lines = vec![format!("Usage: {prgname} [options]")];
    lines.extend(HELP_HEAD.iter().map(|l| l.to_string()));
    if cfg!(target_os = "linux") {
        lines.extend(HELP_LINUX.iter().map(|l| l.to_string()));
    }
    lines.extend(HELP_TAIL.iter().map(|l| l.to_string()));
    let mut s = lines.join("\n");
    s.push('\n');
    s
}

/// What the first pass found.
#[derive(Debug, Default)]
struct PreInit {
    daemonize: bool,
    pid_file: Option<String>,
}

fn fatal(e: &Error) -> Exit {
    report_error(e);
    Exit(1)
}

/// `qobject_input_visitor_new_str()`.
fn visitor_new_str(s: &str, implied_key: Option<&str>) -> ruvm_base::Result<QObjectInputVisitor> {
    if s.starts_with('{') {
        Ok(QObjectInputVisitor::new(json::from_str(s)?))
    } else {
        let args = keyval_parse(s, implied_key, None)?;
        Ok(QObjectInputVisitor::new_keyval(QValue::Dict(args)))
    }
}

/// Visits a whole `T` from an option argument.
fn visit_str<T: Visit>(s: &str, implied_key: Option<&str>) -> Flow<T> {
    let mut v = visitor_new_str(s, implied_key).map_err(|e| fatal(&e))?;
    let mut obj = T::default();
    T::visit(&mut v, None, &mut obj).map_err(|e| fatal(&e))?;
    Ok(obj)
}

/// One pass of `process_options()`. `handle` gets each option that belongs to this pass.
fn process_options(
    argv0: &str,
    args: &[String],
    pre_init_pass: bool,
    mut handle: impl FnMut(i32, &str) -> Flow<()>,
) -> Flow<()> {
    let mut argv = vec![argv0.to_string()];
    argv.extend_from_slice(args);
    let mut g = Getopt::new(argv.clone(), "-hT:V", LONGS);
    loop {
        let save_index = g.optind;
        let Some(o) = g.next() else { break };
        let (c, optarg) = match o {
            Opt::Opt(c, a) => (c, a),
            Opt::NonOpt(a) => (1, Some(a)),
            Opt::Err => (i32::from(b'?'), None),
        };
        let pre = c == i32::from(b'?')
            || c == i32::from(b'h')
            || c == i32::from(b'V')
            || c == OPTION_DAEMONIZE
            || c == OPTION_PIDFILE;
        if pre_init_pass != pre {
            continue;
        }
        // getopt_set_loc(): loc_set_cmdline(argv, save_index, MAX(1, optind - save_index))
        let _loc = optarg.as_ref().map(|_| {
            let option = argv.get(save_index).cloned().unwrap_or_default();
            let arg =
                if g.optind >= save_index + 2 { argv.get(save_index + 1).cloned() } else { None };
            push_location(Location::CmdLine { option, arg })
        });
        handle(c, optarg.as_deref().unwrap_or(""))?;
    }
    Ok(())
}

/// qemu-storage-daemon's `main()`. `argv0` is how the program was called, `args` what came
/// after it and `version` the text of `-V`. Returns the exit status.
pub fn main(argv0: &str, args: &[String], version: &str) -> u8 {
    let prgname = argv0.rsplit(['/', '\\']).next().unwrap_or(argv0).to_string();
    set_program_name(&prgname);
    let r = run(argv0, &prgname, args, version);
    let _ = std::io::stdout().flush();
    match r {
        Ok(()) => 0,
        Err(Exit(code)) => code,
    }
}

fn run(argv0: &str, prgname: &str, args: &[String], version: &str) -> Flow<()> {
    let mut pre = PreInit::default();
    process_options(argv0, args, true, |c, arg| {
        match c {
            OPTION_DAEMONIZE => {
                if cfg!(not(unix)) {
                    eprintln!("--daemonize not supported in this build");
                    return Err(Exit(1));
                }
                pre.daemonize = true;
            }
            OPTION_PIDFILE => pre.pid_file = Some(arg.to_string()),
            _ if c == i32::from(b'h') => {
                print!("{}", help_text(prgname));
                return Err(Exit(0));
            }
            _ if c == i32::from(b'V') => {
                print!("{version}");
                return Err(Exit(0));
            }
            _ => return Err(Exit(1)),
        }
        Ok(())
    })?;

    #[cfg(unix)]
    let daemon = pre.daemonize.then(ruvm_nbd::os::daemonize);

    // module_call_init(MODULE_INIT_QOM), monitor_init_globals() and init_qmp_commands().
    register_object_types();
    let registry = Registry::global().clone();
    let qmp = Qmp::new();
    let chardevs = Arc::new(Chardevs::new());
    ruvm_monitor::object::register_types(&registry, &qmp, &chardevs);
    ruvm_chardev::qom::register_types(&registry);
    chardevs.set_registry(&registry);
    chardevs.hold();
    let st = State::new(qmp.clone(), registry.clone(), chardevs.clone());
    let q = qmp.clone();
    let on_quit: Arc<dyn Fn() + Send + Sync> = Arc::new(move || q.shutdown());
    qmp.register(|cmds| qmp::register(&st, cmds, on_quit));
    let q = qmp.clone();
    ruvm_block::set_event_hook(Some(Arc::new(move |ev| qmp::forward_block_event(&q, ev))));

    #[cfg(unix)]
    {
        let q = qmp.clone();
        ruvm_sys::signal::on_termination(move |_| q.shutdown())
            .map_err(|e| fatal(&Error::from_io("cannot set up signal handling", e)))?;
    }

    let r = process_options(argv0, args, false, |c, arg| {
        second_pass(&st, &registry, &chardevs, c, arg)
    });
    if let Err(e) = r {
        st.cleanup();
        return Err(e);
    }

    // Write the PID file after creating chardevs, exports and NBD servers but before
    // accepting connections.
    let pid_file = match &pre.pid_file {
        Some(p) => match pid_file_init(p) {
            Ok(f) => Some(f),
            Err(e) => {
                st.cleanup();
                return Err(e);
            }
        },
        None => None,
    };
    #[cfg(unix)]
    if let Some(d) = daemon {
        d.setup_post();
    }

    chardevs.release();
    qmp.run_dispatcher();

    st.cleanup();
    ruvm_block::set_event_hook(None);
    if let Some((_, real)) = pid_file {
        let _ = std::fs::remove_file(real);
    }
    Ok(())
}

fn second_pass(
    st: &Arc<State>,
    registry: &Registry,
    chardevs: &Arc<Chardevs>,
    c: i32,
    arg: &str,
) -> Flow<()> {
    match c {
        // trace_opt_parse(): no trace events to enable.
        _ if c == i32::from(b'T') => {}
        OPTION_BLOCKDEV => {
            let opts: BlockdevOptions = visit_str(arg, Some("driver"))?;
            st.graph.blockdev_add(opts).map_err(|e| fatal(&e))?;
        }
        OPTION_CHARDEV => {
            let mut list = ruvm_chardev::opts::chardev_opts();
            let Some(opts) = list.parse_noisily(arg, true) else {
                return Err(Exit(1));
            };
            let opts = opts.clone();
            match chardevs.new_from_opts(&opts) {
                Err(e) => return Err(fatal(&e)),
                // No error, but no chardev means help was printed.
                Ok(None) => return Err(Exit(0)),
                Ok(Some(_)) => {}
            }
        }
        OPTION_EXPORT => {
            let mut v = visitor_new_str(arg, Some("type")).map_err(|e| fatal(&e))?;
            let opts = ExportOptions::visit(&mut v).map_err(|e| fatal(&e))?;
            st.block_export_add(&opts).map_err(|e| fatal(&e))?;
        }
        OPTION_MONITOR => {
            let o: MonitorOptions = visit_str(arg, Some("chardev"))?;
            let qmp_mode = o.mode.unwrap_or(MonitorMode::Control) == MonitorMode::Control;
            let pretty = o.pretty.unwrap_or(false);
            if !qmp_mode {
                return Err(fatal(&Error::generic("Only QMP is supported")));
            }
            monitor_new(registry, o.id.as_deref(), Some(&o.chardev), qmp_mode, pretty)
                .map_err(|e| fatal(&e))?;
        }
        OPTION_NBD_SERVER => {
            let o: NbdServerOptions = visit_str(arg, None)?;
            st.nbd_server_start(&o).map_err(|e| fatal(&e))?;
        }
        OPTION_OBJECT => object_add(arg)?,
        1 => {
            error_report("Unexpected argument");
            return Err(Exit(1));
        }
        _ => {}
    }
    Ok(())
}

/// `pid_file_init()`: the lock holder and the resolved path to remove at exit.
#[cfg(unix)]
type PidFileHandle = (ruvm_nbd::os::PidFile, std::path::PathBuf);
#[cfg(not(unix))]
type PidFileHandle = ((), std::path::PathBuf);

fn pid_file_init(path: &str) -> Flow<PidFileHandle> {
    #[cfg(unix)]
    let lock = ruvm_nbd::os::write_pidfile(path);
    #[cfg(not(unix))]
    let lock = std::fs::write(path, format!("{}\n", std::process::id()))
        .map_err(|e| Error::from_io(format!("Could not create '{path}'"), e));
    let lock = match lock {
        Ok(l) => l,
        Err(e) => {
            error_report(&format!("cannot create PID file: {}", e.message()));
            return Err(Exit(1));
        }
    };
    match std::fs::canonicalize(path) {
        Ok(real) => Ok((lock, real)),
        Err(e) => {
            error_report(&format!(
                "cannot resolve PID file path: {path}: {}",
                ruvm_base::error::strerror(&e)
            ));
            let _ = std::fs::remove_file(path);
            Err(Exit(1))
        }
    }
}
