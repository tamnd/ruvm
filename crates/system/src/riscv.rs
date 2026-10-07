// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V `virt` board from the command line: what `-machine virt`, `-m`, `-smp`, `-cpu`,
//! `-kernel`, `-initrd`, `-append`, `-dtb`, `-bios`, `-serial`, `-drive`, `-device`,
//! `-semihosting` and `-semihosting-config` turn into, and the board running on TCG.
//!
//! `-serial` (or `-nographic`) connects `serial_hd(0)` to the 16550 UART. `-bios` names the
//! M-mode firmware: `default` (or no `-bios`) is OpenSBI's `fw_dynamic` build, looked up in
//! the firmware directories as `qemu_find_file()` does, and `none` runs without one.
//! `-device loader` puts a file or a value into guest memory and can set a hart's PC, and
//! the virtio devices of `-device` and `-drive` go on the PCIe root bus or a virtio-mmio
//! transport (see the `devices` module).
//! Semihosting writes its console to the `chardev` of `-semihosting-config`, or to standard
//! error without one, as semihosting/console.c does, and SYS_EXIT ends ruvm with the
//! guest's status. The SiFive test device's pass and fail finishers do the same with their
//! exit codes, and its reset finisher resets the machine.
//!
//! Deliberate differences from QEMU:
//!
//! - The CPU models are `rv64` (the default), `max`, `rv64i`, `rv64e`, the RVA22 and RVA23
//!   profile CPUs, `sifive-e51`, `sifive-u54` and `shakti-c`, with QEMU's properties (see
//!   `ruvm_target_riscv::cfg`). `max` leaves out the extensions this port does not have.
//!   The other models, a profile CPU that needs such an extension, and a property that turns
//!   one on fail with "... is not supported by ruvm yet".
//! - The machine properties are taken only where their value describes the board that
//!   exists: `aclint=off`, `aia=none`, `aia-guests=0`, `acpi=off` or `auto` (there are no
//!   ACPI tables either way) and `iommu-sys=off` or `auto`. Other values fail with "... is not
//!   supported by ruvm yet".
//! - Every hart is in one socket: `-smp sockets=` above 1 fails.
//! - `-device` knows `loader` and the virtio block, RNG and serial devices only, and
//!   `-drive if=pflash` is not wired to the flash yet.
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
use ruvm_qapi::types::{
    MemorySizeConfiguration, ResetArg, RunState, SMPConfiguration, ShutdownCause,
};
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue};
use ruvm_target_riscv::cfg::{
    CPU_MODELS, CpuBuilder, OTHER_CPU_MODELS, PropError, RiscvCfg, model_missing,
};
use ruvm_target_riscv::tcg::SemihostingHost;

use crate::arm::{Semihosting, SemihostingTarget};
use crate::runstate::Runstate;
use crate::vl::Vm;
use crate::x86::{Drive, Located};

mod devices;

pub(crate) use devices::parse_drives;

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

/// `-cpu model,prop=value,...` for virt (`rv64` without one): the model with its properties
/// set, as `cpu_parse_cpu_model()` and the global properties of the CPU type leave it. The
/// harts finalize it with [`finalize_cpu`].
pub(crate) fn parse_cpu(arg: Option<&str>) -> Result<CpuBuilder> {
    let arg = arg.unwrap_or("rv64");
    let mut parts = arg.split(',');
    let name = parts.next().unwrap_or_default();
    let builder = if model_missing(name).is_empty() { CpuBuilder::new(name) } else { None };
    let Some(mut builder) = builder else {
        if CPU_MODELS.contains(&name) || OTHER_CPU_MODELS.contains(&name) {
            return Err(Error::generic(format!("CPU model '{name}' is not supported by ruvm yet")));
        }
        return Err(Error::generic(format!("unable to find CPU model '{name}'")));
    };
    for feat in parts.filter(|f| !f.is_empty()) {
        let Some((prop, value)) = feat.split_once('=') else {
            return Err(Error::generic(format!("Expected key=value format, found {feat}.")));
        };
        let global = format!("{name}-riscv-cpu.{prop}");
        let err = match builder.set(prop, value) {
            Ok(()) => continue,
            Err(PropError::NotFound) => Error::generic(format!(
                "can't apply global {global}={value}: Property '{global}' not found"
            )),
            Err(PropError::Invalid(msg)) => {
                Error::generic(format!("can't apply global {global}={value}: {msg}"))
            }
            Err(PropError::Hinted(msg, hint)) => {
                Error::generic(format!("can't apply global {global}={value}: {msg}")).hint(hint)
            }
            Err(PropError::Unsupported) => {
                Error::generic(format!("CPU property {prop}={value} is not supported by ruvm yet"))
            }
        };
        // The first hart prints the warnings of the properties set before this one.
        for w in builder.prop_warnings() {
            warn_report(w);
        }
        return Err(err);
    }
    Ok(builder)
}

/// `riscv_cpu_finalize_features()` for each of the `smp` harts: the configuration they run
/// with (the same for every hart). The warnings go to `warn` in the order QEMU prints them,
/// the ones before a failure included.
pub(crate) fn finalize_cpu(
    cpu: &CpuBuilder,
    smp: usize,
    warn: &mut Vec<String>,
) -> Result<RiscvCfg> {
    let mut first = None;
    for hart in 0..smp.max(1) {
        let cfg = cpu.clone().finalize(hart as u64, warn).map_err(Error::generic)?;
        first.get_or_insert(cfg);
    }
    Ok(first.unwrap_or_default())
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
    /// `-drive`, from [`parse_drives`].
    pub drives: &'a [Drive],
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
    let cpu = parse_cpu(args.cpu).map_err(one)?;
    let mut cpu_warnings = Vec::new();
    let cpu = finalize_cpu(&cpu, opts.cpus as usize, &mut cpu_warnings);
    for w in &cpu_warnings {
        warn_report(w);
    }
    let cpu = cpu.map_err(one)?;
    let plan = devices::plan(args.drives, args.devices)?;
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
    cfg.cpu = cpu;
    cfg.loaders = plan.loaders;
    cfg.semihosting = console.clone().map(|c| c as Arc<dyn SemihostingHost>);
    cfg.semihosting_userspace = semi.userspace;
    cfg.clock = Some(Arc::clone(&clock));
    cfg.rtc_clock = Some(Arc::clone(&rtc_clock));
    let mut board = VirtMachine::new(cfg).map_err(|e| one(Error::generic(e)))?;
    devices::plug(&board, &plan.virtio, args.drives).map_err(|e| vec![e])?;

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

    fn cpu(arg: &str) -> RiscvCfg {
        finalize_cpu(&parse_cpu(Some(arg)).unwrap(), 1, &mut Vec::new()).unwrap()
    }

    fn cpu_err(arg: &str) -> String {
        match parse_cpu(Some(arg)) {
            Ok(b) => finalize_cpu(&b, 1, &mut Vec::new()).unwrap_err().message().to_string(),
            Err(e) => e.message().to_string(),
        }
    }

    #[test]
    fn cpu_models() {
        let rv64 = finalize_cpu(&parse_cpu(None).unwrap(), 1, &mut Vec::new()).unwrap();
        assert_eq!(rv64, RiscvCfg::default());
        assert!(!rv64.ext_xlrbr && rv64.ext_h() && !rv64.ext_v());
        assert!(cpu("rv64,xlrbr=true").ext_xlrbr);
        assert!(!cpu("rv64,xlrbr=off").ext_xlrbr);
        let c = cpu("rv64,v=true");
        assert!(c.ext_v() && c.ext_zve64d && c.ext_zve32x);
        assert!(!cpu("rv64,h=false").ext_h());
        assert!(cpu("max").ext_v());
        assert_eq!(cpu("sifive-u54").mmu_type().as_deref(), Some("riscv,sv39"));
        assert_eq!(cpu_err("sifive-e31"), "CPU model 'sifive-e31' is not supported by ruvm yet");
        assert_eq!(cpu_err("foo"), "unable to find CPU model 'foo'");
        assert_eq!(
            cpu_err("rv64,zvfoo=true"),
            "can't apply global rv64-riscv-cpu.zvfoo=true: Property 'rv64-riscv-cpu.zvfoo' not \
             found"
        );
        assert_eq!(cpu_err("rv64,v"), "Expected key=value format, found v.");
        assert_eq!(cpu_err("rv64,pmu-mask=7"), "\"pmu-mask\" contains invalid bits (0-2) set");
        // QEMU takes this set too.
        assert!(cpu("rv64,v=on,zve64d=off").ext_v());
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
}
