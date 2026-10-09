// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio parts of system/vl.c: `-audiodev`, `-audio`, creating the backends before the
//! machine, and `query-audiodevs`. It also plans and realizes the sound cards of `-device`
//! and `-audio model=`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use ruvm_audio::{model, registry};
use ruvm_base::report::{Location, error_report, push_location, report_error};
use ruvm_base::{Error, Result};
use ruvm_hw_audio::{Ac97, TYPE_AC97};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_machine_x86::X86Board;
use ruvm_mem::AddressSpace;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::register_query_audiodevs;
use ruvm_qapi::keyval::keyval_parse;
use ruvm_qapi::opts::{QemuOpts, is_help_option};
use ruvm_qapi::types::Audiodev;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QValue, json};

use crate::x86::Located;

/// `default_audio`: no `-audiodev`, `-audio` or `-nodefaults` was given, so devices get the
/// default audiodevs.
static DEFAULT_AUDIO: AtomicBool = AtomicBool::new(true);

fn fatal(e: &Error) -> u8 {
    report_error(e);
    1
}

/// `audio_parse_option()`, the `-audiodev` case of the option loop. The error is the exit
/// status.
pub(crate) fn parse_audiodev(arg: &str) -> std::result::Result<(), u8> {
    DEFAULT_AUDIO.store(false, Ordering::Relaxed);
    if is_help_option(arg) {
        print!("{}", registry::help_text());
        return Err(0);
    }
    // qobject_input_visitor_new_str()
    let mut v = if arg.starts_with('{') {
        QObjectInputVisitor::new(json::from_str(arg).map_err(|e| fatal(&e))?)
    } else {
        let dict = keyval_parse(arg, Some("driver"), None).map_err(|e| fatal(&e))?;
        QObjectInputVisitor::new_keyval(QValue::Dict(dict))
    };
    let mut dev = Audiodev::default();
    Audiodev::visit(&mut v, None, &mut dev).map_err(|e| fatal(&e))?;
    registry::add_audiodev(dev).map_err(|e| fatal(&e))
}

/// The `-audio` case of the option loop.
pub(crate) fn parse_audio(arg: &str) -> std::result::Result<(), u8> {
    let mut help = false;
    let mut dict = keyval_parse(arg, Some("driver"), Some(&mut help)).map_err(|e| fatal(&e))?;
    DEFAULT_AUDIO.store(false, Ordering::Relaxed);
    if help || dict.get_str("driver").is_some_and(is_help_option) {
        print!("{}", registry::help_text());
        return Err(0);
    }
    if !dict.contains_key("id") {
        dict.put("id", "audiodev0");
    }
    let model = match dict.get_str("model") {
        Some(m) => {
            let m = m.to_string();
            dict.remove("model");
            if is_help_option(&m) {
                print!("{}", model::available_models_text());
                return Err(0);
            }
            Some(m)
        }
        None => None,
    };
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(dict));
    let mut dev = Audiodev::default();
    Audiodev::visit(&mut v, None, &mut dev).map_err(|e| fatal(&e))?;
    let Some(m) = model else {
        return registry::add_default_audiodev(dev).map_err(|e| fatal(&e));
    };
    let id = dev.id.clone();
    registry::add_audiodev(dev).map_err(|e| fatal(&e))?;
    match model::set_model(&m, &id) {
        Ok(()) => Ok(()),
        Err(model::SetModelError::Twice) => {
            error_report("only one -audio option is allowed");
            Err(1)
        }
        Err(model::SetModelError::Unknown(list)) => {
            error_report(&format!("Unknown audio device model `{m}'"));
            print!("{list}");
            Err(1)
        }
    }
}

/// The audio half of `qemu_create_early_backends()`: a backend for every `-audiodev`, then the
/// default audiodevs unless the command line set up audio itself or gave `-nodefaults`.
pub(crate) fn create_early_backends(has_defaults: bool) -> Result<()> {
    registry::init_audiodevs()?;
    if has_defaults && DEFAULT_AUDIO.load(Ordering::Relaxed) {
        registry::create_default_audiodevs();
    }
    Ok(())
}

/// Registers `query-audiodevs`.
pub(crate) fn register(cmds: &mut Commands) {
    register_query_audiodevs(cmds, |_: &MonitorQmp| Ok(registry::query_audiodevs()));
}

/// Registers the cards `-audio model=` offers on `target`, as each card's `type_init()` does.
pub(crate) fn register_models(target: &str) {
    if matches!(target, "x86_64" | "i386") {
        ruvm_hw_audio::register_pc_models();
    }
}

/// A sound card of `-device` or `-audio model=`, planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AudioPlug {
    pub typename: &'static str,
    pub id: Option<String>,
    pub devfn: Option<u8>,
    /// The `audiodev` property. Without it the card takes the default backend.
    pub audiodev: Option<String>,
    pub loc: Option<Location>,
}

/// The sound cards `-device` takes.
const AUDIO_TYPES: &[&str] = &[TYPE_AC97];

/// `qdev_device_add()` up to realize for a sound card: the bus, then the properties. `None`
/// when `driver` is not a sound card. `pci` says whether the machine has the root bus
/// `pcie.0`.
pub(crate) fn plan_device(
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
) -> Option<std::result::Result<AudioPlug, Located>> {
    let &typename = AUDIO_TYPES.iter().find(|t| **t == driver)?;
    Some(plan(typename, opts, loc, pci))
}

fn plan(
    typename: &'static str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
) -> std::result::Result<AudioPlug, Located> {
    if let Some(b) = opts.get("bus") {
        let bus_type = match b {
            "pcie.0" if pci => "PCIE",
            "main-system-bus" => "System",
            _ => return Err(Located::new(loc, format!("Bus '{b}' not found"))),
        };
        if bus_type != "PCIE" {
            return Err(Located::new(
                loc,
                format!("Device '{typename}' can't go on {bus_type} bus"),
            ));
        }
    }
    if !pci {
        return Err(Located::new(loc, format!("No 'PCI' bus found for device '{typename}'")));
    }
    let mut plug = AudioPlug { typename, id: None, devfn: None, audiodev: None, loc: loc.clone() };
    for (k, v) in opts.iter() {
        match k {
            "driver" | "bus" => {}
            "id" => plug.id = Some(v.to_string()),
            "addr" => {
                plug.devfn = Some(crate::display::parse_devfn(v).ok_or_else(|| {
                    Located::new(
                        loc,
                        format!("Property '{typename}.addr' doesn't take value '{v}'"),
                    )
                })?);
            }
            // set_audiodev() looks the backend up as the property is set.
            "audiodev" => {
                registry::be_by_name(v).map_err(|e| Located(loc.clone(), e))?;
                plug.audiodev = Some(v.to_string());
            }
            _ => return Err(Located::new(loc, format!("Property '{typename}.{k}' not found"))),
        }
    }
    Ok(plug)
}

/// The card `-audio model=` picked, for `audio_model_init()`. It goes on the default bus with
/// the next free slot.
pub(crate) fn selected_model() -> Option<AudioPlug> {
    let (m, audiodev) = model::selected()?;
    let typename = AUDIO_TYPES.iter().copied().find(|t| *t == m.typename)?;
    Some(AudioPlug { typename, id: None, devfn: None, audiodev: Some(audiodev), loc: None })
}

/// Guest memory for bus master DMA that does not keep the machine alive.
struct WeakDma(Weak<AddressSpace>);

impl DmaMemory for WeakDma {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::read(&*a, addr, buf))
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        self.0.upgrade().is_some_and(|a| DmaMemory::write(&*a, addr, buf))
    }
}

/// Realizes a planned sound card on an x86 board.
pub(crate) fn realize_x86(board: &X86Board, plug: &AudioPlug) -> std::result::Result<(), Located> {
    let at = |e: Error| Located(plug.loc.clone(), e);
    // What realize reports, such as a voice the backend cannot open, carries the location.
    let _loc = plug.loc.clone().map(push_location);
    let X86Board::Q35(m, _) = board else {
        return Err(Located::new(
            &plug.loc,
            format!("No 'PCI' bus found for device '{}'", plug.typename),
        ));
    };
    let be = match &plug.audiodev {
        Some(name) => registry::be_by_name(name),
        None => registry::be_check(None),
    }
    .map_err(at)?;
    let dma: Arc<dyn DmaMemory> = Arc::new(WeakDma(Arc::downgrade(board.memory_as())));
    Ac97::realize(m.pci_bus(), plug.devfn, plug.id.clone(), be, dma).map_err(at)?;
    Ok(())
}
