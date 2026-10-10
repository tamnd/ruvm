// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu_init()` and `qemu_main_loop()` from system/vl.c and system/runstate.c: the option
//! loop, the order backends and monitors are created in, and the loop that runs until a
//! shutdown request.
//!
//! ruvm behaves like a QEMU build with the `qtest` accelerator, the `none` machine and no
//! displays, plus, for the x86 targets, the `microvm` and `q35` boards on `tcg` and, on Linux
//! x86_64 hosts, `kvm`, and for aarch64 the `virt` board on `tcg`. Options for things that
//! build would leave out fail with QEMU's own messages.
//!
//! Deliberate differences from QEMU:
//!
//! - Without `-accel` or `-machine accel=`, a build with both KVM and TCG tries `kvm:tcg`, so
//!   KVM is used where it works and TCG otherwise (after QEMU's "falling back to tcg"). QEMU
//!   picks `tcg:kvm` unless its program name ends in `kvm`.
//! - virt does not run under qtest yet. The x86 boards take `-qtest` on kvm or tcg, but do
//!   not run on the qtest accelerator.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use ruvm_base::report::{
    Location, current_location, error_report, push_location, report_error, warn_report,
};
use ruvm_base::{ClockType, Error, Result};
use ruvm_block::BlockGraph;
use ruvm_chardev::opts::{chardev_opts, parse_compat};
use ruvm_chardev::{Chardev, Chardevs};
use ruvm_hostmem::region::RegionObjects;
use ruvm_hw_core::machine::{MACHINES, machine_type_name};
use ruvm_hw_core::timer::TimeSource;
use ruvm_hw_core::{Clock, Machine, create_machine};
use ruvm_machine_x86::{BoardKind, canonical_machine_name};
use ruvm_mem::MemorySystem;
use ruvm_migration::Migration;
use ruvm_monitor::Qmp;
use ruvm_monitor::object::{TYPE_MONITOR_HMP, TYPE_MONITOR_QMP, monitor_compat_id, monitor_new};
use ruvm_qapi::keyval::{keyval_merge, keyval_parse, keyval_parse_into};
use ruvm_qapi::opts::{OptsHandle, QemuOptDesc, QemuOptType, QemuOptsList, is_help_option};
use ruvm_qapi::types::{
    Accelerator, DisplayOptions, MonitorMode, MonitorOptions, ObjectOptions, RunState,
    ShutdownCause,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue, json};
use ruvm_qom::{
    Registry, TYPE_OBJECT, TypeInfo, type_print_class_properties, user_creatable_print_types,
};

use crate::arm;
use crate::net::{Network, qmp_events};
use crate::options::{Opt, arch_available, help_text, lookup_opt};
use crate::qmp_cmds::{self, object_options_dict};
use crate::qtest::{self, VirtualClock};
use crate::riscv;
use crate::runstate::{Killed, Runstate};
use crate::x86::{self, Accel, AccelInitError, Located};

/// Whether KVM is built in for `target`: the host is Linux on x86_64 and so is the target.
fn have_kvm(target: &str) -> bool {
    cfg!(all(target_os = "linux", target_arch = "x86_64")) && x86::is_x86(target)
}

/// Whether TCG is built in for `target`: the x86 targets, aarch64 and riscv64, whose front
/// ends exist.
fn have_tcg(target: &str) -> bool {
    x86::is_x86(target) || arm::is_arm(target) || riscv::is_riscv(target)
}

/// The accelerators this build has for `target`. qtest is left out of `-accel help`, as in
/// QEMU.
pub(crate) fn accels(target: &str) -> &'static [&'static str] {
    match (have_kvm(target), have_tcg(target)) {
        (true, _) => &["kvm", "tcg", "qtest"],
        (false, true) => &["tcg", "qtest"],
        (false, false) => &["qtest"],
    }
}

/// The accelerator classes this build has for `target`, under the abstract `accel`, as the
/// accel type_init() functions register them. `qtest_has_accel()` looks for them with
/// `qom-list-types`.
fn register_accel_types(registry: &Registry, target: &str) {
    registry.register(TypeInfo::new("accel").parent(TYPE_OBJECT).abstract_());
    for a in accels(target) {
        registry.register(TypeInfo::new(format!("{a}-accel")).parent("accel"));
    }
}

/// The running machine and everything QMP commands reach.
#[derive(Debug)]
pub struct Vm {
    pub registry: Registry,
    pub qmp: Arc<Qmp>,
    pub chardevs: Arc<Chardevs>,
    pub runstate: Arc<Runstate>,
    /// The memory regions and the objects that stand for them.
    pub regions: Arc<RegionObjects>,
    /// The block graph `blockdev-add` builds.
    pub block: BlockGraph,
    /// `current_machine`, once `qemu_create_machine()` has run.
    pub machine: OnceLock<Machine>,
    /// `qemu_name`, from `-name guest=...`.
    pub name: Option<String>,
    /// `autostart`: `-S` clears it, and `cont` while waiting for an incoming migration sets it.
    pub(crate) autostart: Arc<AtomicBool>,
    /// `incoming`: the main channel of `-incoming`, or `defer`.
    incoming: Option<String>,
    /// The migration state, on a machine ruvm can migrate.
    pub(crate) migration: OnceLock<Migration>,
    /// The accelerator `configure_accelerators()` picked.
    pub(crate) accel: OnceLock<Accelerator>,
    /// The netdevs `net_init_clients()` made, on a machine that has them.
    pub(crate) network: OnceLock<Arc<Network>>,
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
                ruvm_chardev::stdio::term_exit();
                std::process::exit(1);
            }
            // qemu_machine_creation_done()
            if let Some(net) = self.network.get() {
                net.check_clients();
            }
        }
        self.qmp.set_machine_ready(true);
        if let Some(uri) = &self.incoming {
            if uri != "defer" {
                let res = match self.migration.get() {
                    Some(m) => crate::migration::start_incoming(m, uri),
                    None => Err(Error::generic(
                        "migration is not supported with this machine by ruvm yet",
                    )),
                };
                if let Err(e) = res {
                    report_error(&e.prepend(format!("-incoming {uri}: ")));
                    ruvm_chardev::stdio::term_exit();
                    std::process::exit(1);
                }
            }
        } else if self.autostart.load(Ordering::Acquire) {
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
pub(crate) fn mon_opts() -> QemuOptsList {
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

/// `qemu_mem_opts`.
pub(crate) fn memory_opts() -> QemuOptsList {
    QemuOptsList::new(
        "memory",
        &[
            QemuOptDesc::new("size", QemuOptType::Size),
            QemuOptDesc::new("slots", QemuOptType::Number),
            QemuOptDesc::new("maxmem", QemuOptType::Size),
        ],
    )
    .with_implied_opt_name("size")
    .with_merge_lists()
}

/// `qemu_name_opts`.
pub(crate) fn name_opts() -> QemuOptsList {
    QemuOptsList::new(
        "name",
        &[
            QemuOptDesc::new("guest", QemuOptType::String)
                .help("Sets the name of the guest.\nThis name will be displayed in the SDL window caption.\nThe name will also be used for the VNC server"),
            QemuOptDesc::new("process", QemuOptType::String)
                .help("Sets the name of the QEMU process, as shown in top etc"),
            QemuOptDesc::new("debug-threads", QemuOptType::Bool)
                .help("Enable thread names(deprecated, always enabled where supported)"),
        ],
    )
    .with_implied_opt_name("guest")
    .with_merge_lists()
}

/// `qemu_run_with_opts`.
#[cfg(unix)]
pub(crate) fn run_with_opts() -> QemuOptsList {
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
    /// `-incoming`: the main channel, or `defer`.
    incoming: Option<String>,
    /// `-incoming` with the `cpr` channel.
    incoming_cpr: Option<ruvm_qapi::types::MigrationChannel>,
    preconfig: bool,
    qtest: Option<String>,
    qtest_log: Option<String>,
    /// `have_custom_ram_size`: `-machine memory.size` was given.
    have_custom_ram_size: bool,
    /// `-run-with exit-with-parent=on`, which Windows builds never set.
    #[cfg_attr(not(unix), allow(dead_code))]
    exit_with_parent: bool,
    mon_deprecation_warned: bool,
    /// `-m`, `qemu_mem_opts`.
    memory: QemuOptsList,
    /// The x86 board options: `-L`, `-cpu`, `-serial`, `-drive`, `-device` and friends.
    x86: x86::Cmdline,
    /// `-semihosting` and `-semihosting-config`.
    semihosting: arm::Semihosting,
    /// `-machine kernel-irqchip=`, the sugar for the kvm property.
    kernel_irqchip: Option<String>,
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
            incoming: None,
            incoming_cpr: None,
            preconfig: false,
            qtest: None,
            qtest_log: None,
            have_custom_ram_size: false,
            exit_with_parent: false,
            mon_deprecation_warned: false,
            memory: memory_opts(),
            x86: x86::Cmdline::default(),
            semihosting: arm::Semihosting::default(),
            kernel_irqchip: None,
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

/// Puts the terminal back when `qemu_main()` is left, however that happens: QEMU does it
/// from `atexit()`. Whoever calls `std::process::exit()` has to call
/// [`ruvm_chardev::stdio::term_exit`] first, since no destructor runs then.
struct TermGuard;

impl Drop for TermGuard {
    fn drop(&mut self) {
        ruvm_chardev::stdio::term_exit();
    }
}

/// `qemu_init()` followed by `qemu_main_loop()` and `qemu_cleanup()`. Returns the exit status.
pub fn qemu_main(p: &Personality<'_>, args: &[String]) -> u8 {
    let _term = TermGuard;
    match run(p, args) {
        Ok(code) => code,
        Err(Exit(code)) => code,
    }
}

fn run(p: &Personality<'_>, args: &[String]) -> Flow<u8> {
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
    register_accel_types(&registry, p.target);
    crate::audio::register_models(p.target);
    chardevs.set_registry(&registry);
    chardevs.hold();
    let mut cfg = Config::new();
    parse_options(p, &registry, args, &mut cfg)?;
    let vm = start(p, Backends { registry, qmp, chardevs, regions }, cfg)?;
    if let Some(ui_main) = crate::display::take_ui_main() {
        // qemu_main of a display that needs the main thread, which is Cocoa: the main loop
        // moves to a thread of that name and the display never gives the main thread back.
        let spawned = std::thread::Builder::new().name("qemu_main".into()).spawn(move || {
            let status = main_loop(&vm.0, &vm.1);
            drop(vm.1);
            ruvm_chardev::stdio::term_exit();
            std::process::exit(i32::from(status));
        });
        if let Err(e) = spawned {
            error_report(&format!("failed to create qemu_main thread: {e}"));
            return Err(Exit(1));
        }
        ui_main();
    }
    let status = main_loop(&vm.0, &vm.1);
    drop(vm.1);
    Ok(status)
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
    /// The accelerator, when no board took it over.
    _accel: Option<Accel>,
    /// The x86 board on its accelerator.
    board: Option<x86::Running>,
    /// The virt board on TCG.
    arm_board: Option<arm::Running>,
    /// The RISC-V virt board on TCG.
    riscv_board: Option<riscv::Running>,
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
                cfg.x86.default_monitor = false;
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
                cfg.x86.default_monitor = false;
            }
            Opt::Qmp => {
                monitor_parse(cfg, arg, "control", false)?;
                cfg.x86.default_monitor = false;
            }
            Opt::QmpPretty => {
                monitor_parse(cfg, arg, "control", true)?;
                cfg.x86.default_monitor = false;
            }
            Opt::Object => object_option_parse(registry, cfg, arg)?,
            Opt::Machine | Opt::M => {
                let mut help = false;
                keyval_parse_into(&mut cfg.machine, arg, Some("type"), Some(&mut help))
                    .map_err(|e| fail(&e))?;
                if help {
                    print!("{}", machine_help(p.target));
                    return Err(Exit(0));
                }
            }
            Opt::Accel => {
                let Some(opts) = cfg.accel.parse_noisily(arg, true) else { return Err(Exit(1)) };
                if opts.get("accel").is_none_or(is_help_option) {
                    println!("Accelerators supported in QEMU binary:");
                    for a in accels(p.target).iter().filter(|a| **a != "qtest") {
                        println!("{a}");
                    }
                    return Err(Exit(0));
                }
                let h = opts.handle();
                cfg.remember("accel", h);
            }
            Opt::EnableKvm => cfg.machine.put("accel", "kvm"),
            Opt::LowerM => {
                if cfg.memory.parse_noisily(arg, true).is_none() {
                    return Err(Exit(1));
                }
            }
            Opt::Smp => machine_parse_property_opt(cfg, "smp", "cpus", arg)?,
            Opt::Kernel => cfg.machine.put("kernel", arg),
            Opt::Initrd => cfg.machine.put("initrd", arg),
            Opt::Append => cfg.machine.put("append", arg),
            Opt::Dtb => cfg.machine.put("dtb", arg),
            Opt::Semihosting => cfg.semihosting.enable(),
            Opt::SemihostingConfig => {
                cfg.semihosting.config_options(arg).map_err(|e| fail_msg(&e))?;
            }
            Opt::Bios => cfg.machine.put("firmware", arg),
            Opt::Cpu => {
                if is_help_option(arg) {
                    return Err(fail_msg("-cpu help is not supported by ruvm yet"));
                }
                cfg.x86.cpu = Some(arg.to_string());
            }
            Opt::LowerK => ruvm_ui::keymaps::set_keyboard_layout(Some(arg.to_string())),
            Opt::L => {
                if is_help_option(arg) {
                    cfg.x86.list_data_dirs = true;
                } else {
                    cfg.x86.data_dirs.push(arg.into());
                }
            }
            Opt::Serial => {
                cfg.x86.serials.push((arg.to_string(), current_location()));
                cfg.x86.default_serial = false;
                if arg.starts_with("mon:") {
                    cfg.x86.default_monitor = false;
                }
            }
            Opt::Nographic => {
                cfg.machine.put("graphics", "off");
                cfg.x86.nographic = true;
                crate::display::set_display_none();
            }
            Opt::Drive => cfg.x86.drives.push((arg.to_string(), current_location())),
            Opt::Device => {
                let driver = arg.split(',').next().unwrap_or_default();
                if is_help_option(driver) || arg.split(',').skip(1).any(is_help_option) {
                    return Err(fail_msg("-device help is not supported by ruvm yet"));
                }
                cfg.x86.devices.push((arg.to_string(), current_location()));
            }
            Opt::Netdev => cfg.x86.netdevs.push((arg.to_string(), current_location())),
            Opt::NoReboot => cfg.x86.no_reboot = true,
            Opt::Uuid => cfg.x86.set_uuid(arg).map_err(|e| fail_msg(&e))?,
            Opt::Smbios => cfg.x86.add_smbios(arg).map_err(|e| fail_msg(&e))?,
            Opt::Vga => cfg.x86.vga_model = Some(arg.to_string()),
            Opt::Name => {
                if cfg.name.parse_noisily(arg, true).is_none() {
                    return Err(Exit(1));
                }
            }
            Opt::S => cfg.autostart = false,
            Opt::Incoming => incoming_option_parse(cfg, arg)?,
            Opt::Preconfig => cfg.preconfig = true,
            Opt::Nodefaults => {
                cfg.x86.has_defaults = false;
                cfg.x86.default_serial = false;
                cfg.x86.default_monitor = false;
            }
            Opt::FullScreen => crate::display::set_full_screen(),
            Opt::Display => parse_display(arg)?,
            Opt::Vnc => crate::vnc::parse(arg).map_err(Exit)?,
            Opt::Audiodev => crate::audio::parse_audiodev(arg).map_err(Exit)?,
            Opt::Audio => crate::audio::parse_audio(arg).map_err(Exit)?,
            Opt::Qtest => cfg.qtest = Some(arg.to_string()),
            Opt::QtestLog => cfg.qtest_log = Some(arg.to_string()),
            #[cfg(unix)]
            Opt::RunWith => parse_run_with(cfg, arg)?,
            _ => return Err(fail_msg("this option is not supported by ruvm yet")),
        }
    }
    validate_options(cfg)?;
    if let Some(model) = &cfg.x86.vga_model {
        match crate::display::select_vgahw(model) {
            Ok(Some(vga)) => cfg.x86.vga = Some(vga),
            Ok(None) => return Err(Exit(0)),
            Err(e) => return Err(fail_msg(&e)),
        }
    }
    parse_memory_options(cfg)?;
    if cfg.x86.list_data_dirs {
        for dir in cfg.x86.firmware().dirs() {
            println!("{}", dir.display());
        }
        return Err(Exit(0));
    }
    Ok(())
}

/// `qemu_validate_options()`.
/// `incoming_option_parse()`: `defer`, a URI or a channel, each replacing the one of its type
/// given before.
fn incoming_option_parse(cfg: &mut Config, arg: &str) -> Flow<()> {
    match crate::migration::incoming_channel(arg).map_err(|e| fail(&e))? {
        Some(c) if c.channel_type == ruvm_qapi::types::MigrationChannelType::Cpr => {
            cfg.incoming_cpr = Some(c);
        }
        _ => cfg.incoming = Some(arg.to_string()),
    }
    Ok(())
}

fn validate_options(cfg: &Config) -> Flow<()> {
    if cfg.incoming.as_deref().is_some_and(|i| i != "defer") && cfg.preconfig {
        return Err(fail_msg("'preconfig' supports '-incoming defer' only"));
    }
    if cfg.machine.get("kernel").is_none() {
        if cfg.machine.get("append").is_some() {
            return Err(fail_msg("-append only allowed with -kernel option"));
        }
        if cfg.machine.get("initrd").is_some() {
            return Err(fail_msg("-initrd only allowed with -kernel option"));
        }
    }
    Ok(())
}

/// `machine_parse_property_opt()`: `-smp` and the like, as `-machine prop.key=value`.
fn machine_parse_property_opt(cfg: &mut Config, prop: &str, implied: &str, arg: &str) -> Flow<()> {
    let mut help = false;
    let dict = keyval_parse(arg, Some(implied), Some(&mut help)).map_err(|e| fail(&e))?;
    if help {
        return Err(fail_msg(&format!("-{prop} help is not supported by ruvm yet")));
    }
    let mut opts = QDict::new();
    opts.put(prop, QValue::Dict(dict));
    keyval_merge(&mut cfg.machine, &opts).map_err(|e| fail(&e))
}

/// `parse_memory_options()`: `-m` as `-machine memory.size=...`, where a size without a
/// suffix is in megabytes.
fn parse_memory_options(cfg: &mut Config) -> Flow<()> {
    let Some(opts) = cfg.memory.iter().next() else { return Ok(()) };
    let mut dict = QDict::new();
    if let Some(size) = opts.get("size") {
        if size.is_empty() {
            return Err(fail_msg("missing 'size' option value"));
        }
        let mut size = size.to_string();
        if size.ends_with(|c: char| c.is_ascii_digit()) {
            size.push('M');
        }
        dict.put("size", size);
    }
    if let Some(v) = opts.get("maxmem") {
        dict.put("max-size", v);
    }
    if let Some(v) = opts.get("slots") {
        dict.put("slots", v);
    }
    let mut opts = QDict::new();
    opts.put("memory", QValue::Dict(dict));
    keyval_merge(&mut cfg.machine, &opts).map_err(|e| fail(&e))
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

/// `machine_help_func()`: the machines sorted by name, an alias on the line before its
/// machine.
fn machine_help(target: &str) -> String {
    let mut lines: Vec<(String, String)> = MACHINES
        .iter()
        .map(|m| (m.name.to_string(), format!("{:<20} {}\n", m.name, m.desc)))
        .collect();
    if x86::is_x86(target) {
        lines.extend(x86::machine_help_lines());
    }
    if arm::is_arm(target) {
        lines.extend(arm::machine_help_lines());
    }
    if riscv::is_riscv(target) {
        lines.extend(riscv::machine_help_lines());
    }
    lines.sort();
    let mut out = String::from("Supported machines are:\n");
    for (_, text) in lines {
        out.push_str(&text);
    }
    out
}

/// `parse_display()`. The build has no display backends.
fn parse_display(arg: &str) -> Flow<()> {
    if is_help_option(arg) {
        print!("{}", crate::display::display_help());
        return Err(Exit(0));
    }
    if let Some(rest) = arg.strip_prefix("vnc") {
        return match rest.strip_prefix('=') {
            Some(opts) => crate::vnc::parse(opts).map_err(Exit),
            None => Err(fail_msg("VNC requires a display argument vnc=<display>")),
        };
    }
    let dict = keyval_parse(arg, Some("type"), None).map_err(|e| fail(&e))?;
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(dict));
    let mut dpy = DisplayOptions::default();
    DisplayOptions::visit(&mut v, None, &mut dpy).map_err(|e| fail(&e))?;
    crate::display::set_display(dpy);
    Ok(())
}

/// The `-run-with` case of the option loop. QEMU only has the option on POSIX hosts.
#[cfg(unix)]
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

/// The machine `-machine type=` picked.
enum MachineChoice {
    /// A machine type in the QOM registry, by type name.
    Qom(String),
    /// One of the x86 boards, and its machine type name with an alias resolved.
    X86(BoardKind, &'static str),
    /// One of the Arm boards.
    ArmVirt(arm::ArmBoard),
    /// The RISC-V virt board.
    RiscvVirt,
}

/// `select_machine()`: no target has a default machine.
fn select_machine(target: &str, cfg: &mut Config) -> Flow<MachineChoice> {
    let hint = "Use -machine help to list supported machines\n";
    let ty = match cfg.machine.get_str("type") {
        Some(ty) => ty.to_string(),
        None => {
            let e = Error::generic("No machine specified, and there is no default").hint(hint);
            return Err(fail(&e));
        }
    };
    cfg.machine.remove("type");
    if x86::is_x86(target) {
        if let Some(name) = canonical_machine_name(&ty) {
            let kind = BoardKind::from_name(name).expect("an x86 board name");
            return Ok(MachineChoice::X86(kind, name));
        }
    }
    if let Some(board) = arm::board_by_name(&ty).filter(|_| arm::is_arm(target)) {
        return Ok(MachineChoice::ArmVirt(board));
    }
    if riscv::is_riscv(target) && riscv::is_virt(&ty) {
        return Ok(MachineChoice::RiscvVirt);
    }
    if !MACHINES.iter().any(|m| m.name == ty) {
        let e = Error::generic(format!("unsupported machine type: \"{ty}\"")).hint(hint);
        return Err(fail(&e));
    }
    Ok(MachineChoice::Qom(machine_type_name(&ty)))
}

/// `qemu_apply_legacy_machine_options()`: takes out `accel`, `kernel-irqchip` and
/// `memory-backend`. Gives the id of `memory-backend`, which is looked up once the late
/// backends exist.
fn apply_legacy_machine_options(cfg: &mut Config) -> Flow<Option<String>> {
    if let Some(accel) = cfg.machine.get_str("accel") {
        cfg.accelerators = Some(accel.to_string());
        cfg.machine.remove("accel");
    }
    if let Some(v) = cfg.machine.remove("kernel-irqchip") {
        match v {
            QValue::Str(s) => cfg.kernel_irqchip = Some(s),
            _ => return Err(fail_msg("Parameter 'kernel-irqchip' expects a string")),
        }
    }
    let memdev = cfg.machine.get_str("memory-backend").map(str::to_string);
    cfg.machine.remove("memory-backend");
    cfg.have_custom_ram_size =
        matches!(cfg.machine.get("memory"), Some(QValue::Dict(d)) if d.get("size").is_some());
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

/// `configure_accelerators()`. `kind` is the x86 board, if that is the machine; it decides
/// the default of `kernel-irqchip`.
fn configure_accelerators(target: &str, kind: Option<BoardKind>, cfg: &mut Config) -> Flow<Accel> {
    let accels = accels(target);
    let mut init_failed = false;
    if cfg.accel.is_empty() {
        let accelerators = match cfg.accelerators.clone() {
            Some(a) => a,
            None if have_kvm(target) && have_tcg(target) => "kvm:tcg".to_string(),
            None if have_kvm(target) => "kvm".to_string(),
            None if have_tcg(target) => "tcg".to_string(),
            None => {
                return Err(fail_msg(
                    "No accelerator selected and no default accelerator available",
                ));
            }
        };
        for a in accelerators.split(':') {
            if accels.contains(&a) {
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
    let split = kind.is_some_and(BoardKind::default_kernel_irqchip_split);
    let handles: Vec<OptsHandle> = cfg.accel.iter().map(|o| o.handle()).collect();
    let mut chosen = None;
    for h in handles {
        let _loc = cfg.loc("accel", h).map(push_location);
        let opts = cfg.accel.get(h).expect("handle from this list");
        let Some(acc) = opts.get("accel") else {
            report_error(&Error::generic("Parameter 'accel' is missing"));
            return Err(Exit(1));
        };
        if !accels.contains(&acc) {
            error_report(&format!("invalid accelerator {acc}"));
            init_failed = true;
            continue;
        }
        let props: Vec<(String, String)> = opts
            .iter()
            .filter(|(name, _)| *name != "accel")
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        match init_accel(acc, &props, cfg.kernel_irqchip.as_deref(), split) {
            Ok(a) => {
                chosen = Some((acc.to_string(), a));
                break;
            }
            Err(AccelInitError::Fatal(e)) => return Err(fail(&e)),
            Err(AccelInitError::Failed(lines)) => {
                for line in &lines {
                    error_report(line);
                }
                init_failed = true;
            }
        }
    }
    let Some((name, accel)) = chosen else {
        if !init_failed {
            error_report("no accelerator found");
        }
        return Err(Exit(1));
    };
    if init_failed && cfg.qtest.is_none() {
        error_report(&format!("falling back to {name}"));
    }
    Ok(accel)
}

/// `accel_init_machine()` for `acc`, given its `-accel` properties and the
/// `-machine kernel-irqchip=` sugar.
fn init_accel(
    acc: &str,
    props: &[(String, String)],
    kernel_irqchip: Option<&str>,
    default_split: bool,
) -> std::result::Result<Accel, AccelInitError> {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    if acc == "kvm" {
        return x86::kvm_init(props, kernel_irqchip, default_split)
            .map(|a| Accel::Kvm(Box::new(a)));
    }
    let _ = (kernel_irqchip, default_split);
    if acc == "tcg" {
        return x86::tcg_init(props).map(Accel::Tcg);
    }
    match props.first() {
        Some((name, _)) => Err(AccelInitError::Fatal(Error::generic(format!(
            "Property '{acc}-accel.{name}' not found"
        )))),
        None => Ok(Accel::Qtest),
    }
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
        block: BlockGraph::new(),
        machine: OnceLock::new(),
        name,
        autostart: Arc::new(AtomicBool::new(cfg.autostart)),
        incoming: cfg.incoming.clone(),
        migration: OnceLock::new(),
        accel: OnceLock::new(),
        network: OnceLock::new(),
        machine_initialized: AtomicBool::new(false),
        qtest: cfg.qtest.is_some(),
    });
    qmp.register(|cmds| qmp_cmds::register(&vm, p.target, cmds));
    if cfg.incoming.is_some() {
        runstate.set(RunState::Inmigrate);
    }

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

    let choice = select_machine(p.target, &mut cfg)?;
    let rv_virt = matches!(choice, MachineChoice::RiscvVirt);
    let arm_board = match choice {
        MachineChoice::ArmVirt(board) => Some(board),
        _ => None,
    };
    let virt = arm_board.is_some() || rv_virt;
    let mut machine_type = "";
    let (kind, machine) = match choice {
        MachineChoice::X86(kind, name) => {
            machine_type = name;
            (Some(kind), None)
        }
        MachineChoice::ArmVirt(_) | MachineChoice::RiscvVirt => (None, None),
        MachineChoice::Qom(typename) => {
            let machine =
                create_machine(&vm.registry, &typename, &vm.regions).map_err(|e| fail(&e))?;
            (None, Some(vm.machine.get_or_init(|| machine)))
        }
    };
    // cpr_state_load(): before any device, which may need the descriptors.
    if let Some(c) = &cfg.incoming_cpr {
        ruvm_migration::cpr::state_load(&c.addr).map_err(|e| fail(&e))?;
    }

    // C-a x on a mux.
    chardevs.set_mux_quit_handler(mux_quit_hook(&runstate));
    create_default_devices(&mut cfg, kind.is_none() && !virt)?;

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
    // configure_blockdev(): the -drive options, which need to know the machine.
    let drives = if virt {
        let parsed = match arm_board {
            Some(board) => arm::parse_drives(board, &cfg.x86.drives),
            None => riscv::parse_drives(&cfg.x86.drives),
        };
        parsed.map_err(|e| {
            e.report();
            Exit(1)
        })?
    } else {
        parse_drives(kind, &cfg.x86.drives)?
    };
    crate::audio::create_early_backends(cfg.x86.has_defaults, vm.name.as_deref())
        .map_err(|e| fail(&e))?;

    // qemu_apply_legacy_machine_options() and qemu_apply_machine_options()
    let memdev = apply_legacy_machine_options(&mut cfg)?;
    let mut virt_opts = None;
    let mut rv_virt_opts = None;
    let mut board_opts = match (kind, machine) {
        (None, None) if virt => {
            if memdev.is_some() {
                return Err(fail_msg("memory-backend is not supported by ruvm yet"));
            }
            if let Some(board) = arm_board {
                let opts = arm::take_board_options(board, &cfg.machine).map_err(|e| fail(&e))?;
                virt_opts = Some(opts);
            } else {
                let opts = riscv::take_board_options(&cfg.machine).map_err(|e| fail(&e))?;
                rv_virt_opts = Some(opts);
            }
            None
        }
        (Some(kind), _) => {
            let mut opts = x86::take_board_options(kind, &cfg.machine).map_err(|e| fail(&e))?;
            opts.machine_type = machine_type;
            Some(opts)
        }
        (None, Some(machine)) => {
            machine.object.set_props_from_keyval(&cfg.machine, false).map_err(|e| fail(&e))?;
            None
        }
        (None, None) => unreachable!("a QOM machine was created"),
    };
    let accel = configure_accelerators(p.target, kind, &mut cfg)?;
    let _ = vm.accel.set(match &accel {
        Accel::Qtest => Accelerator::Qtest,
        Accel::Tcg(_) => Accelerator::Tcg,
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        Accel::Kvm(_) => Accelerator::Kvm,
    });
    let on_qtest = matches!(accel, Accel::Qtest);
    if (kind.is_some() && on_qtest) || (virt && (on_qtest || cfg.qtest.is_some())) {
        return Err(fail_msg(
            "this machine type is only supported with -accel kvm or tcg by ruvm yet",
        ));
    }
    let clock = VirtualClock::manual(ClockType::Virtual);
    if on_qtest {
        // monitor_qapi_event_init() throttles events on the virtual clock under qtest.
        let c = clock.clone();
        qmp.set_event_clock(move || c.get_ns());
    }

    // qemu_create_late_backends()
    let qtest = match (&cfg.qtest, machine) {
        (Some(chrdev), Some(machine)) => {
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
        _ => None,
    };
    // net_init_clients(). The boards make their netdevs with their NICs.
    if machine.is_some() {
        let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
        let net = Network::new(&cfg.x86.netdevs, &clock, Some(qmp_events(&vm.qmp))).map_err(
            |Located(loc, e)| {
                let _loc = loc.map(push_location);
                fail(&e)
            },
        )?;
        let net = Arc::new(net);
        let n = Arc::clone(&net);
        vm.runstate.set_cpu_hook(Some(Arc::new(move |run| n.vm_state_change(run))));
        let _ = vm.network.set(net);
    }
    create_objects(&vm, &mut cfg, |ty| !object_create_early(ty))?;
    let handles: Vec<OptsHandle> = cfg.mon.iter().map(|o| o.handle()).collect();
    for h in handles {
        let _loc = cfg.loc("mon", h).map(push_location);
        monitor_new_opts(&vm.registry, &cfg, h).map_err(|e| fail(&e))?;
    }
    let serial_hds = create_serials(&chardevs, &vm.registry, &mut cfg)?;

    if let (Some(id), Some(machine)) = (&memdev, machine) {
        resolve_machine_memdev(&vm, machine, &cfg, id)?;
    }
    if let (Some(id), Some(opts)) = (&memdev, board_opts.as_mut()) {
        opts.memdev = Some(x86::consume_memdev(&vm, id).map_err(|e| fail(&e))?);
    }

    let mut keep =
        Keep { _qtest: qtest, _accel: None, board: None, arm_board: None, riscv_board: None };
    if let Some(opts) = rv_virt_opts {
        if cfg.preconfig {
            return Err(fail_msg("-preconfig is not supported with this machine by ruvm yet"));
        }
        let Accel::Tcg(tcg) = accel else { unreachable!("checked above") };
        let args = riscv::RiscvArgs {
            cpu: cfg.x86.cpu.as_deref(),
            no_reboot: cfg.x86.no_reboot,
            semihosting: &cfg.semihosting,
            devices: &cfg.x86.devices,
            drives: &drives,
            firmware: cfg.x86.firmware(),
        };
        let running =
            riscv::start_board_tcg(&vm, tcg, opts, &args, &serial_hds).map_err(|errors| {
                for e in &errors {
                    e.report();
                }
                Exit(1)
            })?;
        keep.riscv_board = Some(running);
    } else if let Some(opts) = virt_opts {
        if cfg.preconfig {
            return Err(fail_msg("-preconfig is not supported with this machine by ruvm yet"));
        }
        let Accel::Tcg(tcg) = accel else { unreachable!("checked above") };
        let args = arm::ArmArgs {
            cpu: cfg.x86.cpu.as_deref(),
            no_reboot: cfg.x86.no_reboot,
            semihosting: &cfg.semihosting,
            devices: &cfg.x86.devices,
            drives: &drives,
        };
        let running =
            arm::start_board_tcg(&vm, tcg, opts, &args, &serial_hds).map_err(|errors| {
                for e in &errors {
                    e.report();
                }
                Exit(1)
            })?;
        keep.arm_board = Some(running);
    } else {
        match (kind, board_opts) {
            (Some(kind), Some(opts)) => {
                if cfg.preconfig {
                    return Err(fail_msg(
                        "-preconfig is not supported with this machine by ruvm yet",
                    ));
                }
                keep = start_x86(&vm, &cfg, accel, kind, opts, &drives, &serial_hds, keep)?;
                // qtest_server_init() on a board a real accelerator runs.
                if let (Some(chrdev), Some(board)) = (&cfg.qtest, &keep.board) {
                    let (memory, io) = board.address_spaces();
                    let log = cfg.qtest_log.as_deref();
                    let a = qtest::server_init_board(&chardevs, chrdev, log, p.target, memory, io)
                        .map_err(|e| fail(&e))?;
                    keep._qtest = Some(a);
                }
            }
            _ => keep._accel = Some(accel),
        }
    }
    crate::display::check_vga_created(cfg.x86.vga);

    if cfg.preconfig {
        qmp.set_machine_ready(false);
    } else {
        vm.exit_preconfig().map_err(|e| fail(&e))?;
    }
    crate::display::set_qemu_uuid(cfg.x86.uuid);
    crate::display::init_displays(&vm, p.version_text).map_err(Exit)?;
    crate::vnc::init(&vm).map_err(Exit)?;
    // The main loop starts here, and with it the frontends.
    chardevs.release();
    Ok((vm, keep))
}

/// `drive_new()` for every `-drive`, in order.
fn parse_drives(
    kind: Option<BoardKind>,
    args: &[(String, Option<Location>)],
) -> Flow<Vec<x86::Drive>> {
    let mut list = x86::drive_opts();
    let mut drives = Vec::new();
    for (arg, loc) in args {
        let _loc = loc.clone().map(push_location);
        let d =
            x86::parse_drive(&mut list, arg, kind, &drives, loc.clone()).map_err(|e| fail(&e))?;
        drives.push(d);
    }
    Ok(drives)
}

/// Builds the x86 board and puts it on its vCPUs: `qemu_init_board()` and
/// `qemu_create_cli_devices()`. `serial_hds` are the chardevs of the `-serial` options.
#[allow(clippy::too_many_arguments)]
fn start_x86(
    vm: &Arc<Vm>,
    cfg: &Config,
    accel: Accel,
    kind: BoardKind,
    opts: x86::BoardOptions,
    drives: &[x86::Drive],
    serial_hds: &[Option<Arc<Chardev>>],
    mut keep: Keep,
) -> Flow<Keep> {
    let running = match accel {
        Accel::Tcg(tcg) => x86::start_board_tcg(vm, tcg, kind, opts, &cfg.x86, drives, serial_hds),
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        Accel::Kvm(accel) => x86::start_board(vm, *accel, kind, opts, &cfg.x86, drives, serial_hds),
        Accel::Qtest => unreachable!("checked by the caller"),
    };
    let running = running.map_err(|errors| {
        for e in &errors {
            e.report();
        }
        Exit(1)
    })?;
    keep.board = Some(running);
    Ok(keep)
}

/// `qemu_create_default_devices()` for the serial port and the monitor. `no_serial` is set
/// for a machine without serial ports.
fn create_default_devices(cfg: &mut Config, no_serial: bool) -> Flow<()> {
    let d = cfg.x86.default_devices(no_serial);
    if let Some(dev) = d.serial {
        cfg.x86.serials.push((dev.to_string(), None));
    }
    if let Some(dev) = d.monitor {
        monitor_parse(cfg, dev, "readline", false)?;
    }
    Ok(())
}

/// `serial_parse()`: the chardev `-serial devname` makes as the `index`th serial port, or
/// `None` for `none`. The chardev is called `serialN`, and a `mon:` string also gets an HMP
/// monitor on its mux, as `qemu_chr_new_mux_mon()` does. The error is the one QEMU reports
/// last; the chardev layer's own error has been printed before it.
fn serial_parse(
    chardevs: &Chardevs,
    registry: &Registry,
    list: &mut QemuOptsList,
    index: usize,
    devname: &str,
) -> Result<Option<Arc<Chardev>>> {
    if devname == "none" {
        return Ok(None);
    }
    let not_connected = || {
        Error::generic(format!("could not connect serial device to character backend '{devname}'"))
    };
    let label = format!("serial{index}");
    match chardevs.new_from_name(list, &label, devname, true) {
        Ok((chr, mux)) => {
            if mux {
                if let Err(e) = monitor_new(registry, None, Some(&label), false, false) {
                    report_error(&e);
                    let _ = chardevs.remove(&label);
                    return Err(not_connected());
                }
            }
            Ok(Some(chr))
        }
        Err(e) => {
            if let Some(e) = e {
                report_error(&e);
            }
            Err(not_connected())
        }
    }
}

/// `foreach_device_config_or_exit(DEV_SERIAL, serial_parse)`: `serial_hd(i)` for every
/// `-serial`, in order.
fn create_serials(
    chardevs: &Chardevs,
    registry: &Registry,
    cfg: &mut Config,
) -> Flow<Vec<Option<Arc<Chardev>>>> {
    let serials = cfg.x86.serials.clone();
    let mut hds = Vec::with_capacity(serials.len());
    for (index, (dev, loc)) in serials.iter().enumerate() {
        let _loc = loc.clone().map(push_location);
        let chr =
            serial_parse(chardevs, registry, &mut cfg.chardev, index, dev).map_err(|e| fail(&e))?;
        hds.push(chr);
    }
    Ok(hds)
}

/// What `C-a x` on a mux runs after printing `QEMU: Terminated`: `qmp_quit()`, the same
/// shutdown request as the QMP `quit` command.
fn mux_quit_hook(runstate: &Arc<Runstate>) -> ruvm_chardev::Hook {
    let rs = Arc::downgrade(runstate);
    Arc::new(move || {
        if let Some(rs) = rs.upgrade() {
            rs.shutdown_request(ShutdownCause::HostQmpQuit);
        }
    })
}

/// `qemu_main_loop()` and `qemu_cleanup()`: serve QMP until something asks for a shutdown.
/// Gives the exit status.
fn main_loop(vm: &Arc<Vm>, keep: &Keep) -> u8 {
    vm.qmp.run_dispatcher();
    let cause = vm.runstate.take_shutdown_request();
    // qemu_kill_report()
    if let Some(k) = vm.runstate.take_killed() {
        if !vm.qtest {
            kill_report(k);
        }
    }
    vm.runstate.send_shutdown_event(cause);
    if let Some(board) = &keep.arm_board {
        // A vCPU waiting for semihosting console input would never stop otherwise.
        board.wake_console();
    }
    if let Some(board) = &keep.riscv_board {
        board.wake_console();
    }
    vm.runstate.vm_shutdown();
    if let Some(board) = &keep.board {
        vm.runstate.set_cpu_hook(None);
        board.quit();
    }
    if let Some(board) = &keep.arm_board {
        vm.runstate.set_cpu_hook(None);
        board.quit();
    }
    if let Some(board) = &keep.riscv_board {
        vm.runstate.set_cpu_hook(None);
        board.quit();
    }
    ruvm_audio::cleanup();
    vm.registry.user_creatable_cleanup();
    // exit() keeps the low eight bits of the status.
    vm.runstate.exit_code() as u8
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Env {
        registry: Registry,
        qmp: Arc<Qmp>,
        chardevs: Arc<Chardevs>,
    }

    fn env() -> Env {
        let registry = Registry::new();
        let qmp = Qmp::new();
        let chardevs = Arc::new(Chardevs::new());
        ruvm_monitor::object::register_types(&registry, &qmp, &chardevs);
        ruvm_chardev::qom::register_types(&registry);
        chardevs.set_registry(&registry);
        Env { registry, qmp, chardevs }
    }

    impl Env {
        fn serial(&self, index: usize, devname: &str) -> Result<Option<Arc<Chardev>>> {
            let mut list = chardev_opts();
            serial_parse(&self.chardevs, &self.registry, &mut list, index, devname)
        }
    }

    #[test]
    fn serial_parse_names_chardevs_after_the_index() {
        let e = env();
        let mut list = chardev_opts();
        let opts = list.parse("null,id=c0", true).unwrap();
        e.chardevs.new_from_opts(opts).unwrap().unwrap();

        let null = e.serial(0, "null").unwrap().unwrap();
        assert_eq!(null.label(), "serial0");
        assert!(!null.is_mux());
        // `none` still takes an index.
        assert!(e.serial(1, "none").unwrap().is_none());
        assert!(e.chardevs.find("serial1").is_none());
        let c0 = e.serial(2, "chardev:c0").unwrap().unwrap();
        assert_eq!(c0.label(), "c0");
        assert!(e.chardevs.find("serial2").is_none());
        assert!(e.qmp.monitors().is_empty());
    }

    #[test]
    fn serial_parse_puts_a_monitor_on_a_mux() {
        let e = env();
        let chr = e.serial(0, "mon:null").unwrap().unwrap();
        assert_eq!(chr.label(), "serial0");
        assert!(chr.is_mux());
        assert!(chr.is_busy());
        // The monitor is HMP, so the QMP list stays empty.
        assert!(e.qmp.monitors().is_empty());
    }

    #[test]
    fn serial_parse_reports_bad_backends() {
        let e = env();
        let err = e.serial(0, "nosuchbackend").unwrap_err();
        assert_eq!(
            err.to_string(),
            "could not connect serial device to character backend 'nosuchbackend'"
        );
        let err = e.serial(1, "chardev:missing").unwrap_err();
        assert_eq!(
            err.to_string(),
            "could not connect serial device to character backend 'chardev:missing'"
        );
        assert!(e.chardevs.find("serial0").is_none());
    }

    #[test]
    fn mux_quit_requests_a_shutdown() {
        let runstate = Runstate::new(Qmp::new());
        let hook = mux_quit_hook(&runstate);
        hook();
        assert_eq!(runstate.take_shutdown_request(), ShutdownCause::HostQmpQuit);
        // The hook does not keep the runstate alive.
        drop(runstate);
        hook();
    }
}
