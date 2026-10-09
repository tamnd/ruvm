// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio parts of system/vl.c: `-audiodev`, `-audio`, creating the backends before the
//! machine, and `query-audiodevs`. It also plans and realizes the sound cards of `-device`
//! and `-audio model=`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_audio::{AudioBackend, model, registry};
use ruvm_base::report::{Location, error_report, push_location, report_error};
use ruvm_base::{Error, Result};
use ruvm_hw_audio::{
    Ac97, HdaCodecKind, IntelHda, TYPE_AC97, TYPE_HDA_DUPLEX, TYPE_HDA_MICRO, TYPE_HDA_OUTPUT,
    TYPE_ICH9_INTEL_HDA, TYPE_INTEL_HDA,
};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_machine_x86::X86Board;
use ruvm_mem::AddressSpace;
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::register_query_audiodevs;
use ruvm_qapi::keyval::keyval_parse;
use ruvm_qapi::opts::{QemuOpts, is_help_option};
use ruvm_qapi::types::Audiodev;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit, Visitor, VisitorExt};
use ruvm_qapi::{QDict, QValue, json};

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

/// A sound card or HDA codec of `-device` or `-audio model=`, planned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AudioPlug {
    pub typename: &'static str,
    pub id: Option<String>,
    pub devfn: Option<u8>,
    /// The `audiodev` property. Without it the card takes the default backend.
    pub audiodev: Option<String>,
    pub loc: Option<Location>,
    /// The `bus` property. [`add_plug`] resolves it to the HDA bus a codec goes on.
    pub bus: Option<String>,
    /// The name of the HDA bus an HDA controller provides, once [`add_plug`] has named it.
    pub hda_bus: Option<String>,
    /// A codec's `cad` property, `u32::MAX` for the next free address.
    pub cad: u32,
    /// A codec's `mixer` property.
    pub mixer: bool,
    /// An HDA controller's `msi` property, `None` for `auto`.
    pub msi: Option<bool>,
    /// An HDA controller's `old_msi_addr` property.
    pub old_msi_addr: bool,
    /// The codec `-audio model=hda` puts on its controller.
    pub codec: Option<Box<AudioPlug>>,
    /// Whether the machine has the root bus `pcie.0`.
    pci: bool,
    /// A property error, its message and hint. QEMU only gets to the properties once the
    /// bus is found.
    err: Option<(String, Option<String>)>,
}

impl AudioPlug {
    fn new(typename: &'static str, loc: Option<Location>, pci: bool) -> AudioPlug {
        AudioPlug {
            typename,
            id: None,
            devfn: None,
            audiodev: None,
            loc,
            bus: None,
            hda_bus: None,
            cad: u32::MAX,
            mixer: true,
            msi: None,
            old_msi_addr: false,
            codec: None,
            pci,
            err: None,
        }
    }
}

/// The sound cards and codecs `-device` takes.
const AUDIO_TYPES: &[&str] = &[
    TYPE_AC97,
    TYPE_INTEL_HDA,
    TYPE_ICH9_INTEL_HDA,
    TYPE_HDA_OUTPUT,
    TYPE_HDA_DUPLEX,
    TYPE_HDA_MICRO,
];

fn is_hda_controller(typename: &str) -> bool {
    matches!(typename, TYPE_INTEL_HDA | TYPE_ICH9_INTEL_HDA)
}

/// `qdev_device_add()` up to realize for a sound card or codec: the properties. `None` when
/// `driver` is not one. `pci` says whether the machine has the root bus `pcie.0`. The bus
/// is checked by [`add_plug`], which knows the HDA buses planned so far.
pub(crate) fn plan_device(
    driver: &str,
    opts: &QemuOpts,
    loc: &Option<Location>,
    pci: bool,
) -> Option<std::result::Result<AudioPlug, Located>> {
    let &typename = AUDIO_TYPES.iter().find(|t| **t == driver)?;
    let mut plug = AudioPlug::new(typename, loc.clone(), pci);
    plug.id = opts.id().map(str::to_string);
    plug.bus = opts.get("bus").map(str::to_string);
    if let Err(Located(_, e)) = plan(&mut plug, opts) {
        plug.err = Some((e.message().to_string(), e.hint_text().map(str::to_string)));
    }
    Some(Ok(plug))
}

/// A qdev property of the sound cards, parsed the way the keyval input visitor does.
fn prop<T: Default>(
    name: &str,
    value: &str,
    visit: impl FnOnce(&mut QObjectInputVisitor, &str, &mut T) -> Result<()>,
) -> Result<T> {
    let mut d = QDict::new();
    d.put(name, QValue::Str(value.to_string()));
    let mut v = QObjectInputVisitor::new_keyval(QValue::Dict(d));
    v.start_struct(None)?;
    let mut out = T::default();
    visit(&mut v, name, &mut out)?;
    v.end_struct();
    Ok(out)
}

fn prop_u32(name: &str, value: &str) -> Result<u32> {
    prop(name, value, |v, n, out| v.type_uint32(Some(n), out))
}

fn prop_bool(name: &str, value: &str) -> Result<bool> {
    prop(name, value, |v, n, out| v.type_bool(Some(n), out))
}

fn plan(plug: &mut AudioPlug, opts: &QemuOpts) -> std::result::Result<(), Located> {
    let typename = plug.typename;
    let loc = plug.loc.clone();
    let at = |e: Error| Located(loc.clone(), e);
    let codec = HdaCodecKind::from_type(typename).is_some();
    let ctrl = is_hda_controller(typename);
    for (k, v) in opts.iter() {
        match k {
            "driver" | "bus" => {}
            "addr" if !codec => {
                plug.devfn = Some(crate::display::parse_devfn(v).ok_or_else(|| {
                    Located::new(
                        &loc,
                        format!("Property '{typename}.addr' doesn't take value '{v}'"),
                    )
                })?);
            }
            // set_audiodev() looks the backend up as the property is set.
            "audiodev" if !ctrl => {
                registry::be_by_name(v).map_err(at)?;
                plug.audiodev = Some(v.to_string());
            }
            // Only the debug output depends on it.
            "debug" if ctrl || codec => {
                prop_u32(k, v).map_err(at)?;
            }
            "cad" if codec => plug.cad = prop_u32(k, v).map_err(at)?,
            "mixer" if codec => plug.mixer = prop_bool(k, v).map_err(at)?,
            // An OnOffAuto.
            "msi" if ctrl => {
                plug.msi = match v {
                    "on" => Some(true),
                    "off" => Some(false),
                    "auto" => None,
                    _ => {
                        let msg = format!("Parameter '{k}' does not accept value '{v}'");
                        return Err(Located::new(&loc, msg));
                    }
                };
            }
            "old_msi_addr" if ctrl => plug.old_msi_addr = prop_bool(k, v).map_err(at)?,
            _ => return Err(Located::new(&loc, format!("Property '{typename}.{k}' not found"))),
        }
    }
    Ok(())
}

/// The name `-audio model=hda` gives its controller's bus: it is created first and has no
/// id.
const MODEL_HDA_BUS: &str = "hda.0";

fn model_is_hda() -> bool {
    model::selected().is_some_and(|(m, _)| is_hda_controller(m.typename))
}

/// Adds a planned `-device` to the sound cards, after the checks of `qdev_device_add()` that
/// depend on what came before: the bus, which for a codec is an HDA bus `qbus_find()` or
/// `qbus_find_recursive()` finds, and then the properties. An HDA controller gets the name
/// of its bus here, `ID.0`, or `hda.N` with `N` counting the controllers without an id.
pub(crate) fn add_plug(
    audio: &mut Vec<(usize, usize, AudioPlug)>,
    vi: usize,
    dn: usize,
    mut plug: AudioPlug,
) -> std::result::Result<(), Located> {
    let model_hda = model_is_hda();
    // The HDA buses so far, oldest first.
    let mut buses: Vec<String> = Vec::new();
    if model_hda {
        buses.push(MODEL_HDA_BUS.to_string());
    }
    buses.extend(audio.iter().filter_map(|(_, _, a)| a.hda_bus.clone()));
    let t = plug.typename;
    let codec = HdaCodecKind::from_type(t).is_some();
    let fail = |msg: String| Err(Located::new(&plug.loc, msg));
    match plug.bus.as_deref() {
        Some(b) if buses.iter().any(|h| h == b) => {
            if !codec {
                return fail(format!("Device '{t}' can't go on HDA bus"));
            }
        }
        Some("pcie.0") if plug.pci => {
            if codec {
                return fail(format!("Device '{t}' can't go on PCIE bus"));
            }
        }
        Some("main-system-bus") => return fail(format!("Device '{t}' can't go on System bus")),
        Some(b) => return fail(format!("Bus '{b}' not found")),
        // qbus_find_recursive() walks the newest devices first.
        None if codec => match buses.last() {
            Some(b) => plug.bus = Some(b.clone()),
            None => return fail(format!("No 'HDA' bus found for device '{t}'")),
        },
        None => {}
    }
    if !codec && !plug.pci {
        return fail(format!("No 'PCI' bus found for device '{t}'"));
    }
    if let Some((msg, hint)) = plug.err.take() {
        let e = Error::generic(msg);
        return Err(Located(
            plug.loc,
            match hint {
                Some(h) => e.hint(h),
                None => e,
            },
        ));
    }
    if is_hda_controller(t) {
        plug.hda_bus = Some(match &plug.id {
            Some(id) => format!("{id}.0"),
            None => {
                let n = usize::from(model_hda)
                    + audio
                        .iter()
                        .filter(|(_, _, a)| a.hda_bus.is_some() && a.id.is_none())
                        .count();
                format!("hda.{n}")
            }
        });
    }
    audio.push((vi, dn, plug));
    Ok(())
}

/// The card `-audio model=` picked, for `audio_model_init()`. It goes on the default bus with
/// the next free slot. For `hda` that is `intel_hda_and_codec_init()`: the controller and an
/// `hda-duplex` codec on its bus, the codec taking the audiodev.
pub(crate) fn selected_model() -> Option<AudioPlug> {
    let (m, audiodev) = model::selected()?;
    let typename = AUDIO_TYPES.iter().copied().find(|t| *t == m.typename)?;
    let mut plug = AudioPlug::new(typename, None, true);
    if is_hda_controller(typename) {
        plug.hda_bus = Some(MODEL_HDA_BUS.to_string());
        let mut codec = AudioPlug::new(TYPE_HDA_DUPLEX, None, true);
        codec.bus = Some(MODEL_HDA_BUS.to_string());
        codec.audiodev = Some(audiodev);
        plug.codec = Some(Box::new(codec));
    } else {
        plug.audiodev = Some(audiodev);
    }
    Some(plug)
}

/// The HDA controllers realized so far and the names of their buses, for the codecs.
static HDA: Mutex<Vec<(String, IntelHda)>> = Mutex::new(Vec::new());

fn hda_list() -> MutexGuard<'static, Vec<(String, IntelHda)>> {
    HDA.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `audio_be_check()` on the card's `audiodev`.
fn backend(plug: &AudioPlug) -> Result<Arc<AudioBackend>> {
    match &plug.audiodev {
        Some(name) => registry::be_by_name(name),
        None => registry::be_check(None),
    }
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

/// Realizes a planned sound card or codec on an x86 board.
pub(crate) fn realize_x86(board: &X86Board, plug: &AudioPlug) -> std::result::Result<(), Located> {
    let at = |e: Error| Located(plug.loc.clone(), e);
    // What realize reports, such as a voice the backend cannot open, carries the location.
    let _loc = plug.loc.clone().map(push_location);
    if let Some(kind) = HdaCodecKind::from_type(plug.typename) {
        let bus = plug.bus.as_deref().unwrap_or_default();
        let list = hda_list();
        let Some((_, ctrl)) = list.iter().rev().find(|(b, _)| b == bus) else {
            return Err(Located::new(&plug.loc, format!("Bus '{bus}' not found")));
        };
        ctrl.add_codec(kind, plug.cad, plug.mixer, || backend(plug)).map_err(at)?;
        return Ok(());
    }
    let X86Board::Q35(m, _) = board else {
        return Err(Located::new(
            &plug.loc,
            format!("No 'PCI' bus found for device '{}'", plug.typename),
        ));
    };
    let dma: Arc<dyn DmaMemory> = Arc::new(WeakDma(Arc::downgrade(board.memory_as())));
    if plug.typename == TYPE_AC97 {
        let be = backend(plug).map_err(at)?;
        Ac97::realize(m.pci_bus(), plug.devfn, plug.id.clone(), be, dma).map_err(at)?;
        return Ok(());
    }
    let clock = registry::clock().ok_or_else(|| at(Error::generic("no virtual clock yet")))?;
    let hda = IntelHda::realize(
        m.pci_bus(),
        plug.devfn,
        plug.id.clone(),
        plug.typename == TYPE_ICH9_INTEL_HDA,
        plug.msi,
        plug.old_msi_addr,
        clock,
        dma,
    )
    .map_err(at)?;
    hda_list().push((plug.hda_bus.clone().unwrap_or_default(), hda));
    if let Some(codec) = &plug.codec {
        realize_x86(board, codec)?;
    }
    Ok(())
}
