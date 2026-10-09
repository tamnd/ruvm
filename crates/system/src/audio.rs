// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio parts of system/vl.c: `-audiodev`, `-audio`, creating the backends before the
//! machine, and `query-audiodevs`.

use std::sync::atomic::{AtomicBool, Ordering};

use ruvm_audio::{model, registry};
use ruvm_base::report::{error_report, report_error};
use ruvm_base::{Error, Result};
use ruvm_monitor::{Commands, MonitorQmp};
use ruvm_qapi::commands::register_query_audiodevs;
use ruvm_qapi::keyval::keyval_parse;
use ruvm_qapi::opts::is_help_option;
use ruvm_qapi::types::Audiodev;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QValue, json};

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
