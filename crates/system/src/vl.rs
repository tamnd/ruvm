// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu_init()` and `qemu_main_loop()` from system/vl.c and system/runstate.c: the option
//! loop, the order backends and monitors are created in, and the loop that runs until a
//! shutdown request.
//!
//! ruvm behaves like a QEMU build with only the `qtest` accelerator, the `none` machine and no
//! displays. Options for things that build would leave out fail with QEMU's own messages.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use ruvm_base::report::{
    Location, current_location, error_report, push_location, report_error, warn_report,
};
use ruvm_base::{Error, Result};
use ruvm_chardev::Chardevs;
use ruvm_chardev::opts::{chardev_opts, parse_compat};
use ruvm_hostmem::region::RegionObjects;
use ruvm_hw_core::machine::{MACHINES, machine_type_name};
use ruvm_hw_core::{Machine, create_machine};
use ruvm_mem::MemorySystem;
use ruvm_monitor::Qmp;
use ruvm_monitor::object::{TYPE_MONITOR_HMP, TYPE_MONITOR_QMP, monitor_compat_id, monitor_new};
use ruvm_qapi::keyval::{keyval_parse, keyval_parse_into};
use ruvm_qapi::opts::{OptsHandle, QemuOptDesc, QemuOptType, QemuOptsList, is_help_option};
use ruvm_qapi::types::{Audiodev, DisplayOptions, MonitorMode, MonitorOptions, ObjectOptions};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue, json};
use ruvm_qom::{Registry, type_print_class_properties, user_creatable_print_types};

use crate::options::{Opt, arch_available, help_text, lookup_opt};
use crate::qmp_cmds::{self, object_options_dict};
use crate::qtest::{self, VirtualClock};
use crate::runstate::{Killed, Runstate};

/// The accelerators this build has. qtest is left out of `-accel help`, as in QEMU.
const ACCELS: &[&str] = &["qtest"];

/// The running machine and everything QMP commands reach.
#[derive(Debug)]
pub struct Vm {
    pub registry: Registry,
    pub qmp: Arc<Qmp>,
    pub chardevs: Arc<Chardevs>,
    pub runstate: Arc<Runstate>,
    /// The memory regions and the objects that stand for them.
    pub regions: Arc<RegionObjects>,
    /// `current_machine`, once `qemu_create_machine()` has run.
    pub machine: OnceLock<Machine>,
    /// `qemu_name`, from `-name guest=...`.
    pub name: Option<String>,
    autostart: bool,
    machine_initialized: AtomicBool,
    /// `qtest_driver()`: a test drives the machine over `-qtest`.
    qtest: bool,
}

impl Vm {
    /// `qmp_x_exit_preconfig()`: finishes creating the machine and starts it unless `-S` was
    /// given.
    pub fn exit_preconfig(&self) -> Result<()> {
        if self.machine_initialized.swap(true, Ordering::AcqRel) {
            return Err(Error::generic(
                "The command is permitted only before machine initialization",
            ));
        }
        // qemu_init_board() fails with error_fatal, even under x-exit-preconfig.
        if let Some(m) = self.machine.get() {
            if let Err(e) = m.run_board_init(&self.regions) {
                report_error(&e);
                std::process::exit(1);
            }
        }
        self.qmp.set_machine_ready(true);
        if self.autostart {
            self.runstate.qmp_cont()?;
        }
        Ok(())
    }
}

/// How the process ends when startup fails or asks to stop early. The message, if any, has
/// been printed already.
#[derive(Debug)]
struct Exit(u8);

type Flow<T> = std::result::Result<T, Exit>;

fn fail(e: &Error) -> Exit {
    report_error(e);
    Exit(1)
}

fn fail_msg(msg: &str) -> Exit {
    error_report(msg);
    Exit(1)
}

/// `qemu_mon_opts`.
fn mon_opts() -> QemuOptsList {
    QemuOptsList::new(
        "mon",
        &[
            QemuOptDesc::new("mode", QemuOptType::String),
            QemuOptDesc::new("chardev", QemuOptType::String),
            QemuOptDesc::new("pretty", QemuOptType::Bool),
        ],
    )
    .with_implied_opt_name("chardev")
}

/// `qemu_name_opts`.
fn name_opts() -> QemuOptsList {
    QemuOptsList::new(
        "name",
        &[
            QemuOptDesc::new("guest", QemuOptType::String)
                .help("Sets the name of the guest.\nThis name will be displayed in the SDL window caption.\nThe name will also be used for the VNC server"),
            QemuOptDesc::new("process", QemuOptType::String)
                .help("Sets the name of the QEMU process, as shown in top etc"),
            QemuOptDesc::new("debug-threads", QemuOptType::Bool).help(
                "When enabled, name the individual threads; defaults off.\nNOTE: The thread names are for debugging and not a\nstable API.",
            ),
        ],
    )
    .with_implied_opt_name("guest")
    .with_merge_lists()
}

/// `qemu_run_with_opts`.
fn run_with_opts() -> QemuOptsList {
    let mut desc = Vec::new();
    if cfg!(target_os = "linux") {
        desc.push(QemuOptDesc::new("async-teardown", QemuOptType::Bool));
    }
    desc.extend([
        QemuOptDesc::new("chroot", QemuOptType::String),
        QemuOptDesc::new("exit-with-parent", QemuOptType::Bool),
        QemuOptDesc::new("user", QemuOptType::String),
    ]);
    QemuOptsList::new("run-with", &desc)
}

/// What the option loop gathers before anything is created.
struct Config {
    chardev: QemuOptsList,
    mon: QemuOptsList,
    accel: QemuOptsList,
    name: QemuOptsList,
    /// `QemuOpts.loc` for the chardev and monitor sets, so errors found later point at the
    /// option they came from.
    locs: HashMap<(&'static str, OptsHandle), Location>,
    objects: Vec<(ObjectOptions, Option<Location>)>,
    machine: QDict,
    /// `accelerators`, from `-machine accel=`.
    accelerators: Option<String>,
    autostart: bool,
    preconfig: bool,
    qtest: Option<String>,
    qtest_log: Option<String>,
    /// `have_custom_ram_size`: `-machine memory.size` was given.
    have_custom_ram_size: bool,
    exit_with_parent: bool,
    mon_deprecation_warned: bool,
}

impl Config {
    fn new() -> Self {
        Config {
            chardev: chardev_opts(),
            mon: mon_opts(),
            accel: QemuOptsList::new("accel", &[]).with_implied_opt_name("accel"),
            name: name_opts(),
            locs: HashMap::new(),
            objects: Vec::new(),
            machine: QDict::new(),
            accelerators: None,
            autostart: true,
            preconfig: false,
            qtest: None,
            qtest_log: None,
            have_custom_ram_size: false,
            exit_with_parent: false,
            mon_deprecation_warned: false,
        }
    }

    fn remember(&mut self, list: &'static str, handle: OptsHandle) {
        if let Some(loc) = current_location() {
            self.locs.insert((list, handle), loc);
        }
    }

    fn loc(&self, list: &'static str, handle: OptsHandle) -> Option<Location> {
        self.locs.get(&(list, handle)).cloned()
    }
}

/// What the program prints for `-help` and `-version`.
#[derive(Debug)]
pub struct Personality<'a> {
    pub target: &'a str,
    pub prgname: &'a str,
    pub version_text: &'a str,
}

/// `qemu_init()` followed by `qemu_main_loop()` and `qemu_cleanup()`. Returns the exit status.
pub fn qemu_main(p: &Personality<'_>, args: &[String]) -> u8 {
    match run(p, args) {
        Ok(()) => 0,
        Err(Exit(code)) => code,
    }
}

fn run(p: &Personality<'_>, args: &[String]) -> Flow<()> {
    // qemu_init_subsystems() registers every type before the option loop, so -object help
    // sees them.
    let registry = Registry::new();
    let qmp = Qmp::new();
    let chardevs = Arc::new(Chardevs::new());
    ruvm_monitor::object::register_types(&registry, &qmp, &chardevs);
    let regions = RegionObjects::new(Arc::new(MemorySystem::new()));
    ruvm_hostmem::region::register_types(&registry);
    ruvm_hostmem::register_types(&registry, &regions);
    ruvm_hw_core::register_types(&registry);
    qtest::register_types(&registry);
    ruvm_chardev::qom::register_types(&registry);
    chardevs.set_registry(&registry);
    chardevs.hold();
    let mut cfg = Config::new();
    parse_options(p, &registry, args, &mut cfg)?;
    let vm = start(p, Backends { registry, qmp, chardevs, regions }, cfg)?;
    main_loop(&vm.0);
    drop(vm.1);
    Ok(())
}

/// What exists before the option loop runs.
struct Backends {
    registry: Registry,
    qmp: Arc<Qmp>,
    chardevs: Arc<Chardevs>,
    regions: Arc<RegionObjects>,
}

/// Backends that live as long as the machine.
struct Keep {
    _qtest: Option<ruvm_chardev::Attachment>,
}

/// The option loop of `qemu_init()`.
fn parse_options(
    p: &Personality<'_>,
    registry: &Registry,
    args: &[String],
    cfg: &mut Config,
) -> Flow<()> {
    let mut optind = 0;
    while optind < args.len() {
        if !args[optind].starts_with('-') {
            let _loc = push_location(Location::CmdLine { option: args[optind].clone(), arg: None });
            return Err(fail_msg("disk images are not supported by ruvm yet"));
        }
        let found = lookup_opt(args, &mut optind).map_err(|e| {
            eprintln!("{}: {}", p.prgname, e.message());
            Exit(1)
        })?;
        if !arch_available(found.option.arch, p.target) {
            return Err(fail_msg("Option not supported for this target"));
        }
        let arg = found.arg.unwrap_or_default();
        match found.option.index {
            Opt::H => {
                print!("{}", help_text(p.target, p.prgname, p.version_text));
                return Err(Exit(0));
            }
            Opt::Version => {
                print!("{}", p.version_text);
                return Err(Exit(0));
            }
            Opt::Chardev => {
                let Some(opts) = cfg.chardev.parse_noisily(arg, true) else { return Err(Exit(1)) };
                let h = opts.handle();
                cfg.remember("chardev", h);
            }
            Opt::Mon => {
                if !cfg.mon_deprecation_warned {
                    cfg.mon_deprecation_warned = true;
                    warn_report(
                        "'-mon' is deprecated, use '-object' with 'monitor-hmp' or 'monitor-qmp' types instead",
                    );
                }
                let Some(opts) = cfg.mon.parse_noisily(arg, true) else { return Err(Exit(1)) };
                let h = opts.handle();
                cfg.remember("mon", h);
            }
            Opt::Monitor => {
                if !arg.starts_with("none") {
                    monitor_parse(cfg, arg, "readline", false)?;
                }
            }
            Opt::Qmp => monitor_parse(cfg, arg, "control", false)?,
            Opt::QmpPretty => monitor_parse(cfg, arg, "control", true)?,
            Opt::Object => object_option_parse(registry, cfg, arg)?,
            Opt::Machine => {
                let mut help = false;
                keyval_parse_into(&mut cfg.machine, arg, Some("type"), Some(&mut help))
                    .map_err(|e| fail(&e))?;
                if help {
                    print!("{}", machine_help(&cfg.machine));
                    return Err(Exit(0));
                }
            }
            Opt::Accel => {
                let Some(opts) = cfg.accel.parse_noisily(arg, true) else { return Err(Exit(1)) };
                if opts.get("accel").is_none_or(is_help_option) {
                    println!("Accelerators supported in QEMU binary:");
                    for a in ACCELS.iter().filter(|a| **a != "qtest") {
                        println!("{a}");
                    }
                    return Err(Exit(0));
                }
            }
            Opt::Name => {
                if cfg.name.parse_noisily(arg, true).is_none() {
                    return Err(Exit(1));
                }
            }
            Opt::S => cfg.autostart = false,
            Opt::Preconfig => cfg.preconfig = true,
            // There are no default devices to leave out.
            Opt::Nodefaults => {}
            Opt::Display => parse_display(arg)?,
            Opt::Audio => parse_audio(arg)?,
            Opt::Qtest => cfg.qtest = Some(arg.to_string()),
            Opt::QtestLog => cfg.qtest_log = Some(arg.to_string()),
            Opt::RunWith => parse_run_with(cfg, arg)?,
            _ => return Err(fail_msg("this option is not supported by ruvm yet")),
        }
    }
    Ok(())
}

/// `monitor_parse()` for `-monitor`, `-qmp` and `-qmp-pretty`.
fn monitor_parse(cfg: &mut Config, arg: &str, mode: &str, pretty: bool) -> Flow<()> {
    let label = match arg.strip_prefix("chardev:") {
        Some(label) => label.to_string(),
        None => {
            let label = monitor_compat_id();
            match parse_compat(&mut cfg.chardev, &label, arg, true) {
                Ok(h) => cfg.remember("chardev", h),
                Err(e) => {
                    if let Some(e) = e {
                        report_error(&e);
                    }
                    return Err(fail_msg(&format!("parse error: {arg}")));
                }
            }
            label
        }
    };
    let opts = cfg.mon.create(Some(&label), true).map_err(|e| fail(&e))?;
    opts.set("mode", mode).expect("mode is a string option");
    opts.set("chardev", &label).expect("chardev is a string option");
    if mode == "control" {
        opts.set_bool("pretty", pretty).expect("pretty is a bool option");
    }
    let h = opts.handle();
    cfg.remember("mon", h);
    Ok(())
}

/// `object_option_parse()`.
fn object_option_parse(registry: &Registry, cfg: &mut Config, arg: &str) -> Flow<()> {
    let root = if arg.starts_with('{') {
        QObjectInputVisitor::new(json::from_str(arg).map_err(|e| fail(&e))?)
    } else {
        let mut list = QemuOptsList::new("object", &[]).with_implied_opt_name("qom-type");
        let Some(opts) = list.parse_noisily(arg, true) else { return Err(Exit(1)) };
        let Some(ty) = opts.get("qom-type") else {
            return Err(fail_msg("Parameter 'qom-type' is missing"));
        };
        // user_creatable_print_help()
        if is_help_option(ty) {
            print!("{}", user_creatable_print_types(registry));
            return Err(Exit(0));
        }
        if opts.has_help_opt() {
            if let Some(text) = type_print_class_properties(registry, ty) {
                print!("{text}");
                return Err(Exit(0));
            }
        }
        QObjectInputVisitor::new_keyval(QValue::Dict(opts.to_qdict()))
    };
    let mut v = root;
    let mut options = ObjectOptions::default();
    ObjectOptions::visit(&mut v, None, &mut options).map_err(|e| fail(&e))?;
    cfg.objects.push((options, current_location()));
    Ok(())
}

/// `machine_help_func()`.
fn machine_help(machine: &QDict) -> String {
    let mut out = String::from("Supported machines are:\n");
    let _ = machine;
    for m in MACHINES {
        out.push_str(&format!("{:<20} {}\n", m.name, m.desc));
    }
    out
}

/// `parse_display()`. The build has no display backends.
fn parse_display(arg: &str) -> Flow<()> {
    if is_help_option(arg) {
        print!(
            "Available display backend types:\nnone\n\nSome display backends support suboptions, which can be set with\n   -display backend,option=value,option=value...\nFor a short list of the suboptions for each display, see the top-level -help output; more detail is in the documentation.\n"
        );
        return Err(Exit(0));
    }
    let dict = keyval_parse(arg, Some("type"), None).map_err(|e| fail(&e))?;
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(dict));
    let mut dpy = DisplayOptions::default();
    // Only default and none are in the schema, and with no display built in the default is
    // none as well.
    DisplayOptions::visit(&mut v, None, &mut dpy).map_err(|e| fail(&e))?;
    Ok(())
}

/// The `-audio` case of the option loop. Nothing plays sound yet, so this only checks the
/// options the way QEMU does.
fn parse_audio(arg: &str) -> Flow<()> {
    let mut help = false;
    let mut dict = keyval_parse(arg, Some("driver"), Some(&mut help)).map_err(|e| fail(&e))?;
    if help || dict.get_str("driver").is_some_and(is_help_option) {
        println!("Available audio drivers:\nnone\nwav");
        return Err(Exit(0));
    }
    if !dict.contains_key("id") {
        dict.put("id", "audiodev0");
    }
    if dict.contains_key("model") {
        return Err(fail_msg("audio models are not supported by ruvm yet"));
    }
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(dict));
    let mut dev = Audiodev::default();
    Audiodev::visit(&mut v, None, &mut dev).map_err(|e| fail(&e))?;
    Ok(())
}

/// The `-run-with` case of the option loop.
fn parse_run_with(cfg: &mut Config, arg: &str) -> Flow<()> {
    let mut list = run_with_opts();
    let Some(opts) = list.parse_noisily(arg, false) else { return Err(Exit(1)) };
    if opts.get_bool("async-teardown", false) {
        return Err(fail_msg("async-teardown is not supported by ruvm yet"));
    }
    if opts.get("chroot").is_some() {
        return Err(fail_msg("chroot is not supported by ruvm yet"));
    }
    if opts.get_bool("exit-with-parent", false) {
        if !cfg!(unix) {
            return Err(fail_msg("exit-with-parent is not available on this platform"));
        }
        cfg.exit_with_parent = true;
    }
    if opts.get("user").is_some() {
        return Err(fail_msg("user is not supported by ruvm yet"));
    }
    Ok(())
}

/// `object_create_early()`: objects whose properties name chardevs wait for them.
fn object_create_early(ty: &str) -> bool {
    !matches!(ty, "rng-egd" | "qtest" | TYPE_MONITOR_HMP | TYPE_MONITOR_QMP)
}

/// `object_option_foreach_add()` for the objects `pick` selects.
fn create_objects(vm: &Vm, cfg: &mut Config, pick: fn(&str) -> bool) -> Flow<()> {
    for (opts, loc) in cfg.objects.iter_mut().filter(|(o, _)| pick(o.u.tag().as_str())) {
        let _loc = loc.clone().map(push_location);
        let dict = object_options_dict(opts).map_err(|e| fail(&e))?;
        vm.registry.user_creatable_add(&dict, false).map_err(|e| fail(&e))?;
    }
    Ok(())
}

/// `select_machine()`: `none` is the only machine, and no target has it as its default.
/// Returns the type name.
fn select_machine(cfg: &mut Config) -> Flow<String> {
    let hint = "Use -machine help to list supported machines\n";
    let ty = match cfg.machine.get_str("type") {
        Some(ty) => ty.to_string(),
        None => {
            let e = Error::generic("No machine specified, and there is no default").hint(hint);
            return Err(fail(&e));
        }
    };
    if !MACHINES.iter().any(|m| m.name == ty) {
        let e = Error::generic(format!("unsupported machine type: \"{ty}\"")).hint(hint);
        return Err(fail(&e));
    }
    cfg.machine.remove("type");
    Ok(machine_type_name(&ty))
}

/// `qemu_apply_legacy_machine_options()` and `qemu_apply_machine_options()`. Gives the id of
/// `memory-backend`, which is looked up once the late backends exist.
fn apply_machine_options(machine: &Machine, cfg: &mut Config) -> Flow<Option<String>> {
    if let Some(accel) = cfg.machine.get_str("accel") {
        cfg.accelerators = Some(accel.to_string());
        cfg.machine.remove("accel");
    }
    let memdev = cfg.machine.get_str("memory-backend").map(str::to_string);
    cfg.machine.remove("memory-backend");
    cfg.have_custom_ram_size =
        matches!(cfg.machine.get("memory"), Some(QValue::Dict(d)) if d.get("size").is_some());
    machine.object.set_props_from_keyval(&cfg.machine, false).map_err(|e| fail(&e))?;
    Ok(memdev)
}

/// `qemu_resolve_machine_memdev()`.
fn resolve_machine_memdev(vm: &Vm, machine: &Machine, cfg: &Config, id: &str) -> Flow<()> {
    let (backend, _) = vm.registry.resolve_path_type(id, "memory-backend");
    let Some(backend) = backend else {
        return Err(fail_msg(&format!("Memory backend '{id}' not found")));
    };
    if !cfg.have_custom_ram_size {
        let size = backend.property_get_uint("size").map_err(|e| fail(&e))?;
        machine.set_ram_size(size);
    }
    machine.object.property_set_link("memory-backend", Some(&backend)).map_err(|e| fail(&e))
}

/// `configure_accelerators()`.
fn configure_accelerators(cfg: &mut Config) -> Flow<()> {
    let mut init_failed = false;
    if cfg.accel.is_empty() {
        let Some(accelerators) = cfg.accelerators.clone() else {
            return Err(fail_msg("No accelerator selected and no default accelerator available"));
        };
        for a in accelerators.split(':') {
            if ACCELS.contains(&a) {
                cfg.accel.parse_noisily(a, true);
            } else {
                init_failed = true;
                error_report(&format!("invalid accelerator {a}"));
            }
        }
    } else if cfg.accelerators.is_some() {
        return Err(fail_msg("The -accel and \"-machine accel=\" options are incompatible"));
    }

    // do_configure_accelerator() until one works.
    let mut found = false;
    for opts in cfg.accel.iter() {
        let Some(acc) = opts.get("accel") else {
            report_error(&Error::generic("Parameter 'accel' is missing"));
            return Err(Exit(1));
        };
        if !ACCELS.contains(&acc) {
            error_report(&format!("invalid accelerator {acc}"));
            init_failed = true;
            continue;
        }
        if let Some((name, _)) = opts.iter().find(|(name, _)| *name != "accel") {
            return Err(fail_msg(&format!("Property '{acc}-accel.{name}' not found")));
        }
        found = true;
        break;
    }
    if !found {
        if !init_failed {
            error_report("no accelerator found");
        }
        return Err(Exit(1));
    }
    if init_failed && cfg.qtest.is_none() {
        error_report("falling back to qtest");
    }
    Ok(())
}

/// `monitor_new_opts()` for one `mon` set.
fn monitor_new_opts(registry: &Registry, cfg: &Config, handle: OptsHandle) -> Result<()> {
    let opts = cfg.mon.get(handle).expect("handle from this list");
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(opts.to_qdict()));
    let mut o = MonitorOptions::default();
    MonitorOptions::visit(&mut v, None, &mut o)?;
    let qmp_mode = o.mode.unwrap_or(MonitorMode::Readline) == MonitorMode::Control;
    let pretty = o.pretty.unwrap_or(false);
    if !qmp_mode && pretty {
        return Err(Error::generic("'pretty' is not compatible with HMP monitors"));
    }
    monitor_new(registry, o.id.as_deref(), Some(&o.chardev), qmp_mode, pretty)?;
    Ok(())
}

/// The part of `qemu_init()` after the option loop.
fn start(p: &Personality<'_>, b: Backends, mut cfg: Config) -> Flow<(Arc<Vm>, Keep)> {
    let Backends { registry, qmp, chardevs, regions } = b;
    let runstate = Runstate::new(qmp.clone());
    let name = cfg.name.iter().next().and_then(|o| o.get("guest")).map(str::to_string);
    let vm = Arc::new(Vm {
        registry,
        qmp: qmp.clone(),
        chardevs: chardevs.clone(),
        runstate: runstate.clone(),
        regions,
        machine: OnceLock::new(),
        name,
        autostart: cfg.autostart,
        machine_initialized: AtomicBool::new(false),
        qtest: cfg.qtest.is_some(),
    });
    qmp.register(|cmds| qmp_cmds::register(&vm, cmds));

    #[cfg(unix)]
    {
        let rs = runstate.clone();
        let on_signal = move |k: ruvm_sys::signal::Killed| {
            rs.killed(Killed { signo: k.signo, pid: k.pid });
        };
        ruvm_sys::signal::on_termination(on_signal)
            .map_err(|e| fail(&Error::from_io("cannot set up signal handling", e)))?;
        if cfg.exit_with_parent {
            let rs = runstate.clone();
            let on_exit = move |k: ruvm_sys::signal::Killed| {
                rs.killed(Killed { signo: k.signo, pid: k.pid });
            };
            ruvm_sys::signal::on_parent_exit(on_exit)
                .map_err(|e| fail(&Error::from_io("cannot watch the parent process", e)))?;
        }
    }

    let typename = select_machine(&mut cfg)?;
    let machine = create_machine(&vm.registry, &typename, &vm.regions).map_err(|e| fail(&e))?;
    let machine = vm.machine.get_or_init(|| machine);

    // qemu_create_early_backends()
    create_objects(&vm, &mut cfg, object_create_early)?;
    let handles: Vec<OptsHandle> = cfg.chardev.iter().map(|o| o.handle()).collect();
    for h in handles {
        let _loc = cfg.loc("chardev", h).map(push_location);
        let opts = cfg.chardev.get(h).expect("handle from this list");
        match chardevs.new_from_opts(opts) {
            Ok(Some(_)) => {}
            Ok(None) => return Err(Exit(0)),
            Err(e) => return Err(fail(&e)),
        }
    }

    let memdev = apply_machine_options(machine, &mut cfg)?;
    configure_accelerators(&mut cfg)?;
    let clock = Arc::new(VirtualClock::default());
    if cfg.qtest.is_some() {
        // monitor_qapi_event_init() throttles events on the virtual clock under qtest.
        let c = clock.clone();
        qmp.set_event_clock(move || c.get_ns());
    }

    // qemu_create_late_backends()
    let qtest = match &cfg.qtest {
        Some(chrdev) => {
            let a = qtest::server_init(
                &chardevs,
                chrdev,
                cfg.qtest_log.as_deref(),
                p.target,
                clock,
                machine,
            )
            .map_err(|e| fail(&e))?;
            qtest::add_object(&machine.object, cfg.qtest_log.as_deref()).map_err(|e| fail(&e))?;
            Some(a)
        }
        None => None,
    };
    create_objects(&vm, &mut cfg, |ty| !object_create_early(ty))?;
    let handles: Vec<OptsHandle> = cfg.mon.iter().map(|o| o.handle()).collect();
    for h in handles {
        let _loc = cfg.loc("mon", h).map(push_location);
        monitor_new_opts(&vm.registry, &cfg, h).map_err(|e| fail(&e))?;
    }

    if let Some(id) = &memdev {
        resolve_machine_memdev(&vm, machine, &cfg, id)?;
    }

    if cfg.preconfig {
        qmp.set_machine_ready(false);
    } else {
        vm.exit_preconfig().map_err(|e| fail(&e))?;
    }
    // The main loop starts here, and with it the frontends.
    chardevs.release();
    Ok((vm, Keep { _qtest: qtest }))
}

/// `qemu_main_loop()` and `qemu_cleanup()`: serve QMP until something asks for a shutdown.
fn main_loop(vm: &Arc<Vm>) {
    vm.qmp.run_dispatcher();
    let cause = vm.runstate.take_shutdown_request();
    // qemu_kill_report()
    if let Some(k) = vm.runstate.take_killed() {
        if !vm.qtest {
            kill_report(k);
        }
    }
    vm.runstate.send_shutdown_event(cause);
    vm.runstate.vm_shutdown();
    vm.registry.user_creatable_cleanup();
}

fn kill_report(k: Killed) {
    if k.pid == 0 {
        error_report(&format!("terminating on signal {}", k.signo));
    } else {
        let name = pid_name(k.pid).unwrap_or_else(|| "<unknown process>".to_string());
        error_report(&format!("terminating on signal {} from pid {} ({name})", k.signo, k.pid));
    }
}

/// `qemu_get_pid_name()`: the program a process runs, where the system says.
fn pid_name(pid: i32) -> Option<String> {
    let cmdline = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    let first = cmdline.split(|b| *b == 0).next()?;
    Some(String::from_utf8_lossy(first).into_owned())
}
