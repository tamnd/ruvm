// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V `virt` board from the command line: what `-machine virt`, `-m`, `-smp`, `-cpu`,
//! `-kernel`, `-initrd`, `-append`, `-dtb`, `-bios`, `-serial`, `-device loader`,
//! `-semihosting` and `-semihosting-config` turn into, and the board running on TCG.
//!
//! `-serial` (or `-nographic`) connects `serial_hd(0)` to the 16550 UART. `-bios` names the
//! M-mode firmware: `default` (or no `-bios`) is OpenSBI's `fw_dynamic` build, looked up in
//! the firmware directories as `qemu_find_file()` does, and `none` runs without one.
//! `-device loader` puts a file or a value into guest memory and can set a hart's PC.
//! Semihosting writes its console to the `chardev` of `-semihosting-config`, or to standard
//! error without one, as semihosting/console.c does, and SYS_EXIT ends ruvm with the
//! guest's status. The SiFive test device's pass and fail finishers do the same with their
//! exit codes, and its reset finisher resets the machine.
//!
//! Deliberate differences from QEMU:
//!
//! - The only CPU model is `rv64`, QEMU's default, with one property, `xlrbr`. The other
//!   models fail with "... is not supported by ruvm yet", and so do the other CPU
//!   properties.
//! - The machine properties are taken only where their value describes the board that
//!   exists: `aclint=off`, `aia=none`, `aia-guests=0`, `acpi=off` or `auto` (there are no
//!   ACPI tables either way) and `iommu-sys=off` or `auto`. Other values fail with "... is not
//!   supported by ruvm yet".
//! - Every hart is in one socket: `-smp sockets=` above 1 fails.
//! - `-device` knows only `loader`, and `-drive` does not exist for virt yet.
//! - SYS_EXIT and the SiFive test finishers ask the main loop to quit with the guest's
//!   status (`shutdown_request` with the code) rather than calling `exit()` on the vCPU
//!   thread, so QMP clients see a SHUTDOWN event first.
//! - A vCPU waiting in SYS_READC or SYS_READ from the console blocks its thread rather than
//!   halting, so `stop` on the monitor waits until the console has input.
//! - `-semihosting-config target=gdb` fails, since there is no gdbstub.

use std::collections::VecDeque;
use std::io::Write as _;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

use ruvm_accel::tcg::TcgOptions;
use ruvm_base::report::{Location, warn_report};
use ruvm_base::{ClockType, Error, Result};
use ruvm_chardev::{Attachment, Chardev, Chardevs, Connection, Frontend};
use ruvm_hw_char::serial::{Serial, SerialBackend};
use ruvm_hw_core::Clock;
use ruvm_hw_core::timer::TimeSource;
use ruvm_machine_riscv::tcg_run::{
    ShutdownReason, VirtEvent, VirtEventHandler, VirtRunConfig, VirtTcgMachine,
};
use ruvm_machine_riscv::virt::{
    GenericLoader, VIRT_CPUS_MAX, VirtConfig, VirtMachine, riscv_find_firmware,
};
use ruvm_machine_x86::FirmwareSearch;
use ruvm_qapi::events::event_reset;
use ruvm_qapi::opts::{QemuOptsList, is_help_option};
use ruvm_qapi::types::{
    MemorySizeConfiguration, ResetArg, RunState, SMPConfiguration, ShutdownCause,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue};
use ruvm_target_riscv::tcg::SemihostingHost;

use crate::arm::{Semihosting, SemihostingTarget};
use crate::runstate::Runstate;
use crate::vl::Vm;
use crate::x86::Located;

/// Whether `target` is one the RISC-V boards exist for.
pub(crate) fn is_riscv(target: &str) -> bool {
    target == "riscv64"
}

/// `mc->desc` of virt.
const VIRT_DESC: &str = "RISC-V VirtIO board";

/// Whether `-machine type=` names the virt board.
pub(crate) fn is_virt(name: &str) -> bool {
    name == "virt"
}

/// The `-machine help` lines of the RISC-V boards, as (sort key, line) pairs.
pub(crate) fn machine_help_lines() -> Vec<(String, String)> {
    vec![("virt".to_string(), format!("{:<20} {VIRT_DESC}\n", "virt"))]
}

/// The generic machine properties and the virt properties, pulled out of the `-machine`
/// options.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct BoardOptions {
    /// `memory.size`, rounded up to 8 KiB like `machine_set_mem()` does.
    pub ram_size: Option<u64>,
    pub cpus: u32,
    pub max_cpus: u32,
    pub kernel: Option<String>,
    pub initrd: Option<String>,
    pub append: Option<String>,
    pub dtb: Option<String>,
    /// `firmware`, from `-bios`.
    pub firmware: Option<String>,
    /// `dumpdtb`: write the device tree there and exit.
    pub dumpdtb: Option<String>,
}

/// Visits `name` of `machine` as a `T`, the way the machine property setter does.
fn visit_member<T: Visit>(machine: &QDict, name: &str) -> Result<Option<T>> {
    let Some(value) = machine.get(name) else { return Ok(None) };
    let mut root = QDict::new();
    root.put(name, value.clone());
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(root));
    v.start_struct(None)?;
    let mut t = T::default();
    let r = T::visit(&mut v, Some(name), &mut t).and_then(|()| v.check_struct());
    v.end_struct();
    r.map(|()| Some(t))
}

/// The keyval value of a `-machine` property as a string.
fn prop_string(name: &str, value: &QValue) -> Result<String> {
    match value {
        QValue::Str(s) => Ok(s.clone()),
        _ => Err(Error::generic(format!("Parameter '{name}' is missing"))),
    }
}

/// One scalar property from its command line string, through the keyval input visitor as
/// qdev and the machine properties parse it, with QEMU's errors.
fn keyval_scalar<T: Default>(
    name: &str,
    value: &str,
    visit: impl FnOnce(&mut QObjectInputVisitor, Option<&str>, &mut T) -> Result<()>,
) -> Result<T> {
    let mut d = QDict::new();
    d.put(name, QValue::Str(value.to_string()));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
    v.start_struct(None)?;
    let mut out = T::default();
    let r = visit(&mut v, Some(name), &mut out);
    v.end_struct();
    r.map(|()| out)
}

fn prop_bool(name: &str, value: &str) -> Result<bool> {
    keyval_scalar(name, value, |v, n, out| v.type_bool(n, out))
}

fn not_supported(name: &str, value: &str) -> Error {
    Error::generic(format!("{name}={value} is not supported by ruvm yet"))
}

/// `visit_type_OnOffAuto()` of the keyval input visitor.
fn on_off_auto<'a>(name: &str, value: &'a str) -> Result<&'a str> {
    match value {
        "on" | "off" | "auto" => Ok(value),
        _ => Err(Error::generic(format!("Parameter '{name}' does not accept value '{value}'"))),
    }
}

/// Checks one virt property against the board that exists.
fn check_virt_prop(name: &str, value: &str) -> Result<()> {
    match name {
        "aclint" => {
            if prop_bool(name, value)? {
                Err(not_supported(name, value))
            } else {
                Ok(())
            }
        }
        "aia" => match value {
            "none" => Ok(()),
            "aplic" | "aplic-imsic" => Err(not_supported(name, value)),
            _ => Err(Error::generic("Invalid AIA interrupt controller type")
                .hint("Valid values are none, aplic, and aplic-imsic.\n")),
        },
        "aia-guests" => {
            // atoi()
            let digits: String = value
                .trim_start()
                .chars()
                .enumerate()
                .take_while(|(i, c)| c.is_ascii_digit() || (*i == 0 && (*c == '-' || *c == '+')))
                .map(|(_, c)| c)
                .collect();
            let n: i64 = digits.parse().unwrap_or(0);
            if !(0..=7).contains(&n) {
                return Err(Error::generic("Invalid number of AIA IMSIC guests")
                    .hint("Valid values be between 0 and 7.\n"));
            }
            if n == 0 { Ok(()) } else { Err(not_supported(name, value)) }
        }
        "acpi" => match on_off_auto(name, value)? {
            "on" => Err(not_supported(name, value)),
            _ => Ok(()),
        },
        "iommu-sys" => match on_off_auto(name, value)? {
            "on" => Err(not_supported(name, value)),
            _ => Ok(()),
        },
        _ => Err(Error::generic(format!("Property 'virt-machine.{name}' not found"))),
    }
}

/// `machine_parse_smp_config()` for virt, which knows none of drawers, books, dies,
/// clusters and modules. Gives (cpus, maxcpus).
pub(crate) fn parse_smp(config: &SMPConfiguration) -> Result<(u32, u32)> {
    let explicit = [
        config.cpus,
        config.drawers,
        config.books,
        config.sockets,
        config.dies,
        config.clusters,
        config.modules,
        config.cores,
        config.threads,
        config.maxcpus,
    ];
    if explicit.iter().any(|v| matches!(v, Some(n) if *n <= 0)) {
        return Err(Error::generic(
            "Invalid CPU topology: CPU topology parameters must be greater than zero",
        ));
    }
    let get = |v: Option<i64>| v.map_or(0, |n| u64::try_from(n).unwrap_or(u64::MAX));
    for (value, name) in [
        (config.modules, "modules"),
        (config.clusters, "clusters"),
        (config.dies, "dies"),
        (config.books, "books"),
        (config.drawers, "drawers"),
    ] {
        if get(value) > 1 {
            return Err(Error::generic(format!(
                "{name} > 1 not supported by this machine's CPU topology"
            )));
        }
    }
    let cpus = get(config.cpus);
    let mut sockets = get(config.sockets);
    let mut cores = get(config.cores);
    let mut threads = get(config.threads);
    let mut maxcpus = get(config.maxcpus);
    let div = |a: u64, b: u64| a.checked_div(b).unwrap_or(0);

    if cpus == 0 && maxcpus == 0 {
        sockets = sockets.max(1);
        cores = cores.max(1);
        threads = threads.max(1);
    } else {
        if maxcpus == 0 {
            maxcpus = cpus;
        }
        // Cores are preferred over sockets since 6.2.
        if cores == 0 {
            sockets = sockets.max(1);
            threads = threads.max(1);
            cores = div(maxcpus, sockets * threads);
        } else if sockets == 0 {
            threads = threads.max(1);
            sockets = div(maxcpus, cores * threads);
        }
        if threads == 0 {
            threads = div(maxcpus, sockets * cores);
        }
    }
    let total = sockets * cores * threads;
    if maxcpus == 0 {
        maxcpus = total;
    }
    let cpus = if cpus == 0 { maxcpus } else { cpus };
    let topo = format!("sockets ({sockets}) * cores ({cores}) * threads ({threads})");
    if total != maxcpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: product of the hierarchy must match maxcpus: {topo} != \
             maxcpus ({maxcpus})"
        )));
    }
    if maxcpus < cpus {
        return Err(Error::generic(format!(
            "Invalid CPU topology: maxcpus must be equal to or greater than smp: {topo} == \
             maxcpus ({maxcpus}) < smp_cpus ({cpus})"
        )));
    }
    if maxcpus > VIRT_CPUS_MAX as u64 {
        return Err(Error::generic(format!(
            "Invalid SMP CPUs {maxcpus}. The max CPUs supported by machine 'virt' is \
             {VIRT_CPUS_MAX}"
        )));
    }
    if sockets > 1 {
        return Err(Error::generic(format!("-smp sockets={sockets} is not supported by ruvm yet")));
    }
    // Both are at most 512 now.
    Ok((cpus as u32, maxcpus as u32))
}

/// `qemu_apply_machine_options()` for virt: takes the generic properties out of `machine`
/// (whose `type`, `accel` and `kernel-irqchip` are gone already) and checks the rest
/// against the board.
pub(crate) fn take_board_options(machine: &QDict) -> Result<BoardOptions> {
    let mut o = BoardOptions::default();
    if let Some(mem) = visit_member::<MemorySizeConfiguration>(machine, "memory")? {
        if mem.slots.is_some_and(|s| s != 0) || mem.max_size.is_some_and(|m| Some(m) != mem.size) {
            return Err(Error::generic("memory hotplug is not supported by ruvm yet"));
        }
        o.ram_size = mem.size.map(|s| s.next_multiple_of(8192));
    }
    let smp = visit_member::<SMPConfiguration>(machine, "smp")?.unwrap_or_default();
    (o.cpus, o.max_cpus) = parse_smp(&smp)?;
    for (name, value) in machine.iter_inserted() {
        match name {
            "memory" | "smp" => {}
            "kernel" => o.kernel = Some(prop_string(name, value)?),
            "initrd" => o.initrd = Some(prop_string(name, value)?),
            "append" => o.append = Some(prop_string(name, value)?),
            "dtb" => o.dtb = Some(prop_string(name, value)?),
            "dumpdtb" => o.dumpdtb = Some(prop_string(name, value)?),
            "firmware" => o.firmware = Some(prop_string(name, value)?),
            // Generic machine properties that change nothing here.
            "dump-guest-core" | "mem-merge" | "graphics" | "suppress-vmdesc" => {}
            _ => check_virt_prop(name, &prop_string(name, value)?)?,
        }
    }
    o.kernel = o.kernel.filter(|k| !k.is_empty());
    o.initrd = o.initrd.filter(|i| !i.is_empty());
    o.dtb = o.dtb.filter(|d| !d.is_empty());
    o.firmware = o.firmware.filter(|f| !f.is_empty());
    Ok(o)
}

/// The CPU models qemu-system-riscv64 has that are not modelled here.
const OTHER_RISCV_CPUS: &[&str] = &[
    "max",
    "max32",
    "rv32",
    "x-rv128",
    "rv32i",
    "rv32e",
    "rv64i",
    "rv64e",
    "rva22u64",
    "rva22s64",
    "rva23u64",
    "rva23s64",
    "lowrisc-ibex",
    "shakti-c",
    "sifive-e31",
    "sifive-e34",
    "sifive-e51",
    "sifive-u34",
    "sifive-u54",
    "thead-c906",
    "thead-c908",
    "thead-c908v",
    "veyron-v1",
    "tt-ascalon",
    "xiangshan-nanhu",
    "xiangshan-kunminghu",
    "mips-p8700",
    "host",
];

/// `-cpu model,prop=value,...` for virt (`rv64` without one). Gives whether the `xlrbr`
/// extension is on.
pub(crate) fn parse_cpu(arg: Option<&str>) -> Result<bool> {
    let arg = arg.unwrap_or("rv64");
    let mut parts = arg.split(',');
    let name = parts.next().unwrap_or_default();
    if name != "rv64" {
        if OTHER_RISCV_CPUS.contains(&name) {
            return Err(Error::generic(format!("CPU model '{name}' is not supported by ruvm yet")));
        }
        return Err(Error::generic(format!("unable to find CPU model '{name}'")));
    }
    let mut xlrbr = false;
    for feat in parts.filter(|f| !f.is_empty()) {
        let Some((prop, value)) = feat.split_once('=') else {
            return Err(Error::generic(format!("Expected key=value format, found {feat}")));
        };
        match prop {
            "xlrbr" => xlrbr = prop_bool(prop, value)?,
            _ => {
                return Err(Error::generic(format!(
                    "CPU property {prop}={value} is not supported by ruvm yet"
                )));
            }
        }
    }
    Ok(xlrbr)
}

/// `-device loader,...`, the properties of hw/core/generic-loader.c. The checks that need
/// the machine are made when the board realizes it.
pub(crate) fn parse_loader(
    opts: &ruvm_qapi::opts::QemuOpts,
    loc: &Option<Location>,
) -> std::result::Result<GenericLoader, Located> {
    let fail = |e: Error| Located(loc.clone(), e);
    let mut l = GenericLoader::default();
    for (k, v) in opts.iter() {
        match k {
            "driver" => {}
            "file" => l.file = Some(v.to_string()),
            "addr" => {
                l.addr = keyval_scalar(k, v, |vis, n, o| vis.type_uint64(n, o)).map_err(fail)?
            }
            "data" => {
                l.data = keyval_scalar(k, v, |vis, n, o| vis.type_uint64(n, o)).map_err(fail)?
            }
            "data-len" => {
                l.data_len = keyval_scalar(k, v, |vis, n, o| vis.type_uint8(n, o)).map_err(fail)?;
            }
            "data-be" => l.data_be = prop_bool(k, v).map_err(fail)?,
            "cpu-num" => {
                let n: u32 =
                    keyval_scalar(k, v, |vis, n, o| vis.type_uint32(n, o)).map_err(fail)?;
                // CPU_NONE is the default.
                l.cpu_num = (n != u32::MAX).then_some(n);
            }
            "force-raw" => l.force_raw = prop_bool(k, v).map_err(fail)?,
            "bus" => {
                return Err(fail(Error::generic(format!("Bus '{v}' not found"))));
            }
            _ => {
                return Err(fail(Error::generic(format!("Property 'loader.{k}' not found"))));
            }
        }
    }
    Ok(l)
}

/// The `-device` options for virt: only `loader` exists.
pub(crate) fn parse_devices(
    devices: &[(String, Option<Location>)],
) -> std::result::Result<Vec<GenericLoader>, Located> {
    let mut loaders = Vec::new();
    for (arg, loc) in devices {
        let mut list = QemuOptsList::new("device", &[]).with_implied_opt_name("driver");
        let opts = list.parse(arg, true).map_err(|e| Located(loc.clone(), e))?;
        let Some(driver) = opts.get("driver") else {
            return Err(Located(loc.clone(), Error::generic("Parameter 'driver' is missing")));
        };
        if is_help_option(driver) || opts.has_help_opt() {
            let e = Error::generic("-device help is not supported by ruvm yet");
            return Err(Located(loc.clone(), e));
        }
        if driver != "loader" {
            let e = Error::generic(format!(
                "-device {driver} is not supported with this machine by ruvm yet"
            ));
            return Err(Located(loc.clone(), e));
        }
        loaders.push(parse_loader(opts, loc)?);
    }
    Ok(loaders)
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The UART on its chardev, `serial_mm_init(..., serial_hd(0), ...)`: what the guest writes
/// goes to the chardev, what the chardev reads goes to the UART.
struct ChardevSerial {
    uart: Arc<Serial>,
    chr: Arc<Chardev>,
}

impl SerialBackend for ChardevSerial {
    fn write(&self, bytes: &[u8]) -> usize {
        // Output nobody can take is dropped.
        let _ = self.chr.write_all(bytes);
        bytes.len()
    }
}

impl Frontend for ChardevSerial {
    fn serve(&self, conn: &mut Connection) -> std::io::Result<()> {
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            let mut input = &buf[..n];
            while !input.is_empty() {
                let room = self.uart.can_receive();
                if room == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                    continue;
                }
                let k = room.min(input.len());
                self.uart.receive(&input[..k]);
                input = &input[k..];
            }
        }
    }
}

/// The size of the console input FIFO, `FIFO_SIZE`.
const SEMI_FIFO_SIZE: usize = 1024;

#[derive(Debug, Default)]
struct Fifo {
    bytes: VecDeque<u8>,
    /// The machine is going away: wake the readers.
    closed: bool,
}

/// The semihosting console, semihosting/console.c, and the exit of SYS_EXIT.
struct SemiConsole {
    chr: Option<Arc<Chardev>>,
    /// `semihosting.argv` from `arg=`; the board falls back to `-kernel` and `-append`.
    args: Vec<String>,
    fifo: Mutex<Fifo>,
    cv: Condvar,
    runstate: Weak<Runstate>,
}

impl std::fmt::Debug for SemiConsole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemiConsole")
            .field("chr", &self.chr.as_ref().map(|c| c.label().to_string()))
            .field("args", &self.args)
            .finish_non_exhaustive()
    }
}

impl SemiConsole {
    /// Wakes a vCPU waiting for console input, for good.
    fn close(&self) {
        lock(&self.fifo).closed = true;
        self.cv.notify_all();
    }
}

impl SemihostingHost for SemiConsole {
    /// `qemu_semihosting_console_write()`.
    fn console_write(&self, buf: &[u8]) -> usize {
        match &self.chr {
            Some(chr) => chr.write_all(buf).unwrap_or(0),
            None => {
                let mut err = std::io::stderr();
                match err.write_all(buf) {
                    Ok(()) => buf.len(),
                    Err(_) => 0,
                }
            }
        }
    }

    /// `qemu_semihosting_console_read()` for one byte.
    fn console_read(&self) -> u8 {
        let mut f = lock(&self.fifo);
        loop {
            if let Some(b) = f.bytes.pop_front() {
                self.cv.notify_all();
                return b;
            }
            if f.closed {
                return 0;
            }
            f = self.cv.wait(f).unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn exit(&self, code: u32) {
        if let Some(rs) = self.runstate.upgrade() {
            rs.shutdown_request_with_code(ShutdownCause::GuestShutdown, code as i32);
        }
    }

    /// `semihosting_get_cmdline()`.
    fn cmdline(&self) -> Option<String> {
        if self.args.is_empty() { None } else { Some(self.args.join(" ")) }
    }

    /// The board answers this one.
    fn heap_info(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// The chardev side of the semihosting console: `console_can_read()` and `console_read()`.
struct SemiConsoleFrontend(Arc<SemiConsole>);

impl Frontend for SemiConsoleFrontend {
    fn serve(&self, conn: &mut Connection) -> std::io::Result<()> {
        let c = &self.0;
        let mut buf = [0u8; 256];
        loop {
            let n = conn.recv(&mut buf)?;
            if n == 0 {
                return Ok(());
            }
            let mut input = &buf[..n];
            let mut f = lock(&c.fifo);
            while !input.is_empty() {
                if f.closed {
                    return Ok(());
                }
                let room = SEMI_FIFO_SIZE - f.bytes.len();
                if room == 0 {
                    f = c.cv.wait(f).unwrap_or_else(PoisonError::into_inner);
                    continue;
                }
                let k = room.min(input.len());
                f.bytes.extend(&input[..k]);
                input = &input[k..];
                c.cv.notify_all();
            }
        }
    }
}

/// The virt board running on its vCPUs, and the chardevs its devices are attached to.
#[derive(Debug)]
pub(crate) struct Running {
    machine: Arc<VirtTcgMachine>,
    console: Option<Arc<SemiConsole>>,
    _attachments: Vec<Attachment>,
}

impl Running {
    /// Lets a vCPU waiting for semihosting console input go, so that the vCPUs can stop.
    pub(crate) fn wake_console(&self) {
        if let Some(c) = &self.console {
            c.close();
        }
    }

    /// Stops the vCPU and timer threads.
    pub(crate) fn quit(&self) {
        self.wake_console();
        self.machine.quit();
    }
}

/// What the run loop reports, turned into runstate changes and QMP events.
fn event_handler(vm: &Arc<Vm>) -> VirtEventHandler {
    let rs = Arc::clone(&vm.runstate);
    let qmp = Arc::clone(&vm.qmp);
    Arc::new(move |e: VirtEvent| match e {
        VirtEvent::Shutdown(ShutdownReason::GuestShutdown(code)) => {
            rs.shutdown_request_with_code(ShutdownCause::GuestShutdown, i32::from(code));
        }
        VirtEvent::Shutdown(ShutdownReason::GuestPanic(code)) => {
            rs.shutdown_request_with_code(ShutdownCause::GuestPanic, i32::from(code));
        }
        VirtEvent::Shutdown(ShutdownReason::GuestReset) => {
            rs.shutdown_request(ShutdownCause::GuestReset);
        }
        VirtEvent::Reset => {
            let arg = ResetArg { guest: true, reason: ShutdownCause::GuestReset };
            if let Some(ev) = event_reset(&qmp.policy(), arg) {
                qmp.emit_event(ev);
            }
        }
        VirtEvent::InternalError(msg) => {
            eprintln!("{msg}");
            rs.vm_stop(RunState::InternalError);
        }
    })
}

/// Lets the runstate start and stop the vCPUs.
fn set_cpu_hook(vm: &Arc<Vm>, machine: &Arc<VirtTcgMachine>) {
    let weak = Arc::downgrade(machine);
    vm.runstate.set_cpu_hook(Some(Arc::new(move |run| {
        if let Some(m) = weak.upgrade() {
            if run {
                m.start();
            } else {
                m.pause();
            }
        }
    })));
}

/// What `start_board_tcg` needs from the command line besides the machine options.
#[derive(Debug)]
pub(crate) struct RiscvArgs<'a> {
    /// `-cpu`.
    pub cpu: Option<&'a str>,
    /// `-no-reboot`.
    pub no_reboot: bool,
    pub semihosting: &'a Semihosting,
    /// `-device`.
    pub devices: &'a [(String, Option<Location>)],
    /// Where `qemu_find_file(QEMU_FILE_TYPE_BIOS, ...)` looks.
    pub firmware: FirmwareSearch,
}

/// `qemu_semihosting_chardev_init()`: the chardev of `-semihosting-config chardev=`.
fn semihosting_chardev(
    chardevs: &Chardevs,
    semi: &Semihosting,
) -> std::result::Result<Option<Arc<Chardev>>, Located> {
    let Some(id) = &semi.chardev else { return Ok(None) };
    match chardevs.find(id) {
        Some(c) => Ok(Some(c)),
        None => Err(Located(None, Error::generic(format!("semihosting chardev '{id}' not found")))),
    }
}

/// `qemu_init_board()`, `qemu_create_cli_devices()` and `qemu_machine_creation_done()` for
/// virt on TCG: builds the board, connects the UART to `serial_hds[0]` and semihosting to
/// its chardev, realizes the `-device loader`s and puts it all on the vCPU threads, stopped
/// until `vm_start()`. A `dumpdtb` ends the process here, as in QEMU.
pub(crate) fn start_board_tcg(
    vm: &Arc<Vm>,
    tcg: TcgOptions,
    opts: BoardOptions,
    args: &RiscvArgs<'_>,
    serial_hds: &[Option<Arc<Chardev>>],
) -> std::result::Result<Running, Vec<Located>> {
    let one = |e: Error| vec![Located(None, e)];
    let semi = args.semihosting;
    let console_chr = semihosting_chardev(&vm.chardevs, semi).map_err(|e| vec![e])?;
    if semi.enabled && semi.target == SemihostingTarget::Gdb {
        return Err(one(Error::generic(
            "semihosting-config target=gdb is not supported by ruvm yet",
        )));
    }
    let xlrbr = parse_cpu(args.cpu).map_err(one)?;
    let loaders = parse_devices(args.devices).map_err(|e| vec![e])?;
    let find = |name: &str| args.firmware.find(name).map(|p| p.to_string_lossy().into_owned());
    let firmware =
        riscv_find_firmware(opts.firmware.as_deref(), find).map_err(|e| one(Error::generic(e)))?;
    let clock = Clock::new(ClockType::Virtual, TimeSource::Monotonic(Instant::now()));
    let rtc_clock = Clock::new(ClockType::Host, TimeSource::Wall);

    let console = semi.enabled.then(|| {
        Arc::new(SemiConsole {
            chr: console_chr.clone(),
            args: semi.args.clone(),
            fifo: Mutex::new(Fifo::default()),
            cv: Condvar::new(),
            runstate: Arc::downgrade(&vm.runstate),
        })
    });
    let mut cfg = VirtConfig { smp: opts.cpus as usize, ..VirtConfig::default() };
    if let Some(size) = opts.ram_size {
        cfg.ram_size = size;
    }
    cfg.kernel = opts.kernel;
    cfg.initrd = opts.initrd;
    cfg.append = opts.append;
    cfg.dtb = opts.dtb;
    cfg.firmware = firmware;
    cfg.xlrbr = xlrbr;
    cfg.loaders = loaders;
    cfg.semihosting = console.clone().map(|c| c as Arc<dyn SemihostingHost>);
    cfg.semihosting_userspace = semi.userspace;
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let mut board = VirtMachine::new(cfg).map_err(|e| one(Error::generic(e)))?;

    let mut attachments = Vec::new();
    if let Some(Some(chr)) = serial_hds.first() {
        let fe = Arc::new(ChardevSerial { uart: Arc::clone(board.uart()), chr: Arc::clone(chr) });
        board.set_serial_backend(Some(fe.clone()));
        attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
    }
    if let (Some(console), Some(chr)) = (&console, &console_chr) {
        let fe = Arc::new(SemiConsoleFrontend(Arc::clone(console)));
        attachments.push(chr.attach(fe).map_err(|e| vec![Located(None, e)])?);
    }

    board.machine_done().map_err(|e| one(Error::generic(e)))?;
    if let Some(path) = &opts.dumpdtb {
        // handle_machine_dumpdtb()
        if let Err(e) = std::fs::write(path, board.fdt().as_bytes()) {
            return Err(one(Error::generic(format!("Error saving FDT to file {path}: {e}"))));
        }
        ruvm_chardev::stdio::term_exit();
        std::process::exit(0);
    }

    let cfg = VirtRunConfig { no_reboot: args.no_reboot, tcg, backend: None };
    let (machine, warnings) =
        VirtTcgMachine::new(board, vec![clock, rtc_clock], &cfg, event_handler(vm))
            .map_err(|e| one(Error::generic(e)))?;
    for w in &warnings {
        warn_report(w);
    }
    let machine = Arc::new(machine);
    set_cpu_hook(vm, &machine);
    Ok(Running { machine, console, _attachments: attachments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smp(cpus: Option<i64>, sockets: Option<i64>, cores: Option<i64>) -> SMPConfiguration {
        SMPConfiguration { cpus, sockets, cores, ..SMPConfiguration::default() }
    }

    #[test]
    fn smp_topologies() {
        assert_eq!(parse_smp(&SMPConfiguration::default()).unwrap(), (1, 1));
        assert_eq!(parse_smp(&smp(Some(4), None, None)).unwrap(), (4, 4));
        let e = parse_smp(&smp(Some(3), None, Some(2))).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid CPU topology: product of the hierarchy must match maxcpus: sockets (1) * \
             cores (2) * threads (1) != maxcpus (3)"
        );
        let clusters = SMPConfiguration { clusters: Some(2), ..SMPConfiguration::default() };
        assert_eq!(
            parse_smp(&clusters).unwrap_err().message(),
            "clusters > 1 not supported by this machine's CPU topology"
        );
        let e = parse_smp(&smp(Some(513), None, None)).unwrap_err();
        assert_eq!(
            e.message(),
            "Invalid SMP CPUs 513. The max CPUs supported by machine 'virt' is 512"
        );
        let e = parse_smp(&smp(Some(4), Some(2), None)).unwrap_err();
        assert_eq!(e.message(), "-smp sockets=2 is not supported by ruvm yet");
    }

    #[test]
    fn cpu_models() {
        assert!(!parse_cpu(None).unwrap());
        assert!(parse_cpu(Some("rv64,xlrbr=true")).unwrap());
        assert!(!parse_cpu(Some("rv64,xlrbr=off")).unwrap());
        assert_eq!(
            parse_cpu(Some("sifive-u54")).unwrap_err().message(),
            "CPU model 'sifive-u54' is not supported by ruvm yet"
        );
        assert_eq!(parse_cpu(Some("foo")).unwrap_err().message(), "unable to find CPU model 'foo'");
        assert_eq!(
            parse_cpu(Some("rv64,h=true")).unwrap_err().message(),
            "CPU property h=true is not supported by ruvm yet"
        );
    }

    #[test]
    fn machine_properties() {
        let mut m = QDict::new();
        m.put("aclint", "off");
        m.put("aia", "none");
        m.put("acpi", "off");
        m.put("kernel", "k");
        m.put("firmware", "none");
        let o = take_board_options(&m).unwrap();
        assert_eq!(o.kernel.as_deref(), Some("k"));
        assert_eq!(o.firmware.as_deref(), Some("none"));
        let mut m = QDict::new();
        m.put("aia", "aplic");
        assert_eq!(
            take_board_options(&m).unwrap_err().message(),
            "aia=aplic is not supported by ruvm yet"
        );
        let mut m = QDict::new();
        m.put("aia", "foo");
        assert_eq!(
            take_board_options(&m).unwrap_err().message(),
            "Invalid AIA interrupt controller type"
        );
        let mut m = QDict::new();
        m.put("foo", "on");
        assert_eq!(
            take_board_options(&m).unwrap_err().message(),
            "Property 'virt-machine.foo' not found"
        );
    }

    #[test]
    fn loader_devices() {
        let devs = vec![
            ("loader,file=/tmp/x.elf".to_string(), None),
            ("loader,addr=0x80000000,data=0x1234,data-len=4,cpu-num=0".to_string(), None),
        ];
        let l = parse_devices(&devs).unwrap();
        assert_eq!(l[0].file.as_deref(), Some("/tmp/x.elf"));
        assert_eq!(l[1].addr, 0x8000_0000);
        assert_eq!(l[1].data, 0x1234);
        assert_eq!(l[1].data_len, 4);
        assert_eq!(l[1].cpu_num, Some(0));
        let devs = vec![("loader,foo=1".to_string(), None)];
        assert_eq!(
            parse_devices(&devs).unwrap_err().1.message(),
            "Property 'loader.foo' not found"
        );
        let devs = vec![("virtio-net-device".to_string(), None)];
        assert_eq!(
            parse_devices(&devs).unwrap_err().1.message(),
            "-device virtio-net-device is not supported with this machine by ruvm yet"
        );
    }
}
