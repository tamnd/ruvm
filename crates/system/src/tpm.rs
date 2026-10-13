// SPDX-License-Identifier: GPL-2.0-or-later

//! `-tpmdev`, `system/tpm.c`: the TPM backends. They are made with the late backends and wait
//! for the TPM device whose `tpmdev` property names them.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use ruvm_base::report::{Location, error_report, push_location, report_error};
use ruvm_chardev::Chardevs;
use ruvm_hw_tpm::TpmBackend;
use ruvm_qapi::opts::{
    OptsHandle, QemuOptDesc, QemuOptType, QemuOpts, QemuOptsList, is_help_option,
};

/// A backend and its id, `TPMBackend`. The device that takes it leaves `None`.
struct Backend {
    id: String,
    be: Option<Box<dyn TpmBackend>>,
}

/// `tpm_backends`.
static BACKENDS: Mutex<Vec<Backend>> = Mutex::new(Vec::new());

/// The TPM devices realized so far, for `tpm_find()`.
static DEVICES: AtomicU32 = AtomicU32::new(0);

fn backends() -> MutexGuard<'static, Vec<Backend>> {
    BACKENDS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The backend types with `TPMBackendClass.desc`. QEMU also has `passthrough`. Like QEMU,
/// ruvm has `-tpmdev` only on POSIX hosts.
const TYPES: &[(&str, &str)] = &[("emulator", "TPM emulator backend driver")];

/// `tpm_emulator_cmdline_opts`.
const EMULATOR_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("type", QemuOptType::String),
    QemuOptDesc::new("chardev", QemuOptType::String),
];

/// `qemu_tpmdev_opts`: the backends define the options.
pub(crate) fn tpmdev_opts() -> QemuOptsList {
    QemuOptsList::new("tpmdev", &[]).with_implied_opt_name("type")
}

/// `tpm_display_backend_drivers()`.
fn display_backend_drivers() {
    eprintln!("Supported TPM types (choose only one):");
    for (name, desc) in TYPES {
        eprintln!("{name:>12}   {desc}");
    }
}

/// `tpm_config_parse()`: `-tpmdev help` lists the types and exits.
#[cfg(unix)]
pub(crate) fn config_parse(list: &mut QemuOptsList, arg: &str) -> Result<OptsHandle, u8> {
    if is_help_option(arg) {
        display_backend_drivers();
        return Err(0);
    }
    list.parse_noisily(arg, true).map(|o| o.handle()).ok_or(1)
}

/// `tpm_init()`: makes the backend of each `-tpmdev`, given with where it came from. The
/// first failure is reported and ends the loop.
pub(crate) fn init(chardevs: &Chardevs, tpmdevs: &mut [(QemuOpts, Option<Location>)]) -> bool {
    for (opts, loc) in tpmdevs {
        let _loc = loc.clone().map(push_location);
        if let Err(e) = init_tpmdev(chardevs, opts) {
            if !e.is_empty() {
                error_report(&e);
            }
            return false;
        }
    }
    true
}

/// `tpm_init_tpmdev()`. An empty error has been reported already.
fn init_tpmdev(chardevs: &Chardevs, opts: &mut QemuOpts) -> Result<(), String> {
    if !backends().is_empty() {
        return Err("Only one TPM is allowed.".to_string());
    }
    let Some(id) = opts.id().map(str::to_string) else {
        return Err("Parameter 'id' is missing".to_string());
    };
    let Some(ty) = opts.get("type") else {
        error_report("Parameter 'type' is missing");
        display_backend_drivers();
        return Err(String::new());
    };
    if ty == "passthrough" {
        return Err("-tpmdev passthrough is not supported by ruvm yet".to_string());
    }
    if ty != "emulator" {
        error_report("Parameter 'type' expects a TPM backend type");
        display_backend_drivers();
        return Err(String::new());
    }
    if let Err(e) = opts.validate(EMULATOR_OPTS) {
        report_error(&e);
        return Err(String::new());
    }
    let be = emulator_create(chardevs, opts)?;
    backends().push(Backend { id, be: Some(be) });
    Ok(())
}

/// `tpm_emulator_handle_device_opts()`.
#[cfg(unix)]
fn emulator_create(chardevs: &Chardevs, opts: &QemuOpts) -> Result<Box<dyn TpmBackend>, String> {
    let Some(name) = opts.get("chardev") else {
        return Err("tpm-emulator: parameter 'chardev' is missing".to_string());
    };
    let Some(chr) = chardevs.find(name) else {
        return Err(format!("tpm-emulator: tpm chardev '{name}' not found"));
    };
    let ctrl = chr
        .claim_unix_socket()
        .map_err(|e| format!("tpm-emulator: No valid chardev found at '{name}':{e}"))?;
    // Only a connected Unix socket carries the data socket over to the emulator, and a
    // chardev that is not one cannot answer the shutdown QEMU sends as the backend goes away.
    let Some(ctrl) = ctrl else {
        error_report("tpm-emulator: Failed to send CMD_SET_DATAFD");
        return Err("tpm-emulator: Could not cleanly shutdown the TPM".to_string());
    };
    let mut keep = ctrl.try_clone().map_err(|e| e.to_string())?;
    match ruvm_hw_tpm::emulator::TpmEmulator::new(ctrl, name) {
        Ok(emu) => Ok(Box::new(emu)),
        Err(e) => {
            error_report(&e);
            Err(ruvm_hw_tpm::emulator::shutdown(&mut keep).err().unwrap_or_default())
        }
    }
}

#[cfg(not(unix))]
fn emulator_create(_: &Chardevs, _: &QemuOpts) -> Result<Box<dyn TpmBackend>, String> {
    Err("-tpmdev emulator is not supported on this host by ruvm yet".to_string())
}

/// What the `tpmdev` property of a TPM device does with the id it is given, `set_tpm()` and
/// `tpm_backend_init()`: takes the backend. `prop` is `<type>.tpmdev` for the error.
pub(crate) fn take(prop: &str, id: &str) -> Result<Box<dyn TpmBackend>, String> {
    let mut all = backends();
    let Some(b) = all.iter_mut().find(|b| b.id == id) else {
        return Err(format!("Property '{prop}' can't find value '{id}'"));
    };
    b.be.take().ok_or_else(|| format!("TPM backend '{id}' is already initialized"))
}

/// The `tpm_find()` check of a TPM device's realize: counts the device, and says whether it is
/// the only one.
pub(crate) fn add_device() -> bool {
    DEVICES.fetch_add(1, Ordering::AcqRel) == 0
}
