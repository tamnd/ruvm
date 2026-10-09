// SPDX-License-Identifier: GPL-2.0-or-later

//! The audiodev list and the backends made from it, QEMU's `audio/audio.c`.
//!
//! `-audiodev` and `-audio` add audiodevs while the command line is parsed. When the machine is
//! created every listed audiodev becomes a backend, and a device without an `audiodev` property
//! gets the first default audiodev that opens. QEMU keeps the backends as children of the
//! `/audiodevs` container; here they sit in a list with the same lookup by id.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_hw_core::timer::Clock;
use ruvm_qapi::types::{
    AudioFormat, Audiodev, AudiodevDriver, AudiodevPerDirectionOptions, AudiodevU,
};

#[cfg(feature = "audio-alsa")]
use crate::alsa::AlsaDriver;
use crate::engine::AudioBackend;
use crate::none::NoneDriver;
use crate::pcm::{Driver, Pdo};
use crate::wav::WavDriver;

/// `audio_prio_list`: the drivers a default audiodev is tried with, in order. `none` always
/// works, so it comes last.
const PRIO_LIST: &[AudiodevDriver] = &[AudiodevDriver::None];

struct Reg {
    audiodevs: Vec<Audiodev>,
    default_audiodevs: VecDeque<Audiodev>,
    backends: Vec<Arc<AudioBackend>>,
    default_be: Option<Arc<AudioBackend>>,
    clock: Option<Arc<Clock>>,
    running: bool,
}

static REG: Mutex<Reg> = Mutex::new(Reg {
    audiodevs: Vec::new(),
    default_audiodevs: VecDeque::new(),
    backends: Vec::new(),
    default_be: None,
    clock: None,
    running: false,
});

fn reg() -> MutexGuard<'static, Reg> {
    REG.lock().unwrap_or_else(|e| e.into_inner())
}

/// The driver behind `driver`, if this build has it, like `audio_be_class_by_name()`.
fn driver_for(driver: AudiodevDriver) -> Option<Box<dyn Driver>> {
    match driver {
        AudiodevDriver::None => Some(Box::new(NoneDriver)),
        AudiodevDriver::Wav => Some(Box::new(WavDriver)),
        // The audio of the D-Bus display is not there, so dbus is not a driver of this build.
        #[cfg(feature = "ui-dbus")]
        AudiodevDriver::Dbus => None,
        #[cfg(feature = "audio-alsa")]
        AudiodevDriver::Alsa => Some(Box::new(AlsaDriver)),
    }
}

/// The per-direction options of a driver. Some drivers extend `AudiodevPerDirectionOptions`
/// with members of their own, and qapi-gen flattens the base into those types, so the base
/// members are copied out and back where QEMU would take a pointer to the base.
trait PerDirection: Default {
    fn base(&self) -> AudiodevPerDirectionOptions;
    fn set_base(&mut self, b: AudiodevPerDirectionOptions);
}

impl PerDirection for AudiodevPerDirectionOptions {
    fn base(&self) -> AudiodevPerDirectionOptions {
        self.clone()
    }

    fn set_base(&mut self, b: AudiodevPerDirectionOptions) {
        *self = b;
    }
}

// Only the drivers a build has use it.
#[allow(unused_macros)]
macro_rules! per_direction {
    ($t:ty) => {
        impl PerDirection for $t {
            fn base(&self) -> AudiodevPerDirectionOptions {
                AudiodevPerDirectionOptions {
                    mixing_engine: self.mixing_engine,
                    fixed_settings: self.fixed_settings,
                    frequency: self.frequency,
                    channels: self.channels,
                    voices: self.voices,
                    format: self.format,
                    buffer_length: self.buffer_length,
                }
            }

            fn set_base(&mut self, b: AudiodevPerDirectionOptions) {
                self.mixing_engine = b.mixing_engine;
                self.fixed_settings = b.fixed_settings;
                self.frequency = b.frequency;
                self.channels = b.channels;
                self.voices = b.voices;
                self.format = b.format;
                self.buffer_length = b.buffer_length;
            }
        }
    };
}

#[cfg(feature = "audio-alsa")]
per_direction!(ruvm_qapi::types::AudiodevAlsaPerDirectionOptions);

/// The base `in` and `out` options of an audiodev, whatever its driver.
fn pdos(
    u: &AudiodevU,
) -> (Option<AudiodevPerDirectionOptions>, Option<AudiodevPerDirectionOptions>) {
    fn base<P: PerDirection>(
        i: &Option<P>,
        o: &Option<P>,
    ) -> (Option<AudiodevPerDirectionOptions>, Option<AudiodevPerDirectionOptions>) {
        (i.as_ref().map(P::base), o.as_ref().map(P::base))
    }
    match u {
        AudiodevU::None(o) => base(&o.in_, &o.out),
        AudiodevU::Wav(o) => base(&o.in_, &o.out),
        #[cfg(feature = "ui-dbus")]
        AudiodevU::Dbus(o) => base(&o.in_, &o.out),
        #[cfg(feature = "audio-alsa")]
        AudiodevU::Alsa(o) => base(&o.in_, &o.out),
    }
}

/// `audio_validate_per_direction_opts()`.
fn validate_per_direction_opts(pdo: &mut AudiodevPerDirectionOptions) -> Result<()> {
    let mixing_engine = *pdo.mixing_engine.get_or_insert(true);
    let fixed_settings = *pdo.fixed_settings.get_or_insert(mixing_engine);
    if !fixed_settings
        && (pdo.frequency.is_some() || pdo.channels.is_some() || pdo.format.is_some())
    {
        return Err(Error::generic(
            "You can't use frequency, channels or format with fixed-settings=off",
        ));
    }
    if !mixing_engine && fixed_settings {
        return Err(Error::generic("You can't use fixed-settings without mixeng"));
    }
    pdo.frequency.get_or_insert(44100);
    pdo.channels.get_or_insert(2);
    pdo.voices.get_or_insert(if mixing_engine { 1 } else { i32::MAX as u32 });
    pdo.format.get_or_insert(AudioFormat::S16);
    Ok(())
}

/// `audio_validate_opts()`: fills in every default, the way `query-audiodevs` shows them.
pub fn validate_opts(dev: &mut Audiodev) -> Result<()> {
    fn validate<P: PerDirection>(in_: &mut Option<P>, out: &mut Option<P>) -> Result<()> {
        for p in [in_, out] {
            let p = p.get_or_insert_with(Default::default);
            let mut b = p.base();
            validate_per_direction_opts(&mut b)?;
            p.set_base(b);
        }
        Ok(())
    }
    match &mut dev.u {
        AudiodevU::None(o) => validate(&mut o.in_, &mut o.out)?,
        AudiodevU::Wav(o) => validate(&mut o.in_, &mut o.out)?,
        #[cfg(feature = "ui-dbus")]
        AudiodevU::Dbus(o) => validate(&mut o.in_, &mut o.out)?,
        #[cfg(feature = "audio-alsa")]
        AudiodevU::Alsa(o) => validate(&mut o.in_, &mut o.out)?,
    }
    dev.timer_period.get_or_insert(10000);
    Ok(())
}

/// `audio_add_audiodev()`, with the error QEMU makes fatal returned instead.
pub fn add_audiodev(mut dev: Audiodev) -> Result<()> {
    validate_opts(&mut dev)?;
    reg().audiodevs.push(dev);
    Ok(())
}

/// `audio_add_default_audiodev()`: an audiodev from `-audio` without a model, used for devices
/// that do not name one.
pub fn add_default_audiodev(mut dev: Audiodev) -> Result<()> {
    validate_opts(&mut dev)?;
    reg().default_audiodevs.push_back(dev);
    Ok(())
}

/// `audio_create_default_audiodevs()`: one `#default` audiodev per driver in the priority list.
pub fn create_default_audiodevs() {
    for &drv in PRIO_LIST {
        if driver_for(drv).is_none() {
            continue;
        }
        let dev =
            Audiodev { id: "#default".to_string(), timer_period: None, u: AudiodevU::for_tag(drv) };
        // QEMU passes &error_abort: the defaults always validate.
        if let Err(e) = add_default_audiodev(dev) {
            panic!("default audiodev: {e}");
        }
    }
}

/// `audio_be_new()`.
fn be_new(dev: Audiodev, clock: Option<&Arc<Clock>>, running: bool) -> Result<Arc<AudioBackend>> {
    let drv = dev.u.tag();
    let Some(driver) = driver_for(drv) else {
        return Err(Error::generic(format!("Unknown audio driver `{}'", drv.as_str())));
    };
    let (pin, pout) = pdos(&dev.u);
    let pdo_in = Pdo::from_qapi(&pin.unwrap_or_default());
    let pdo_out = Pdo::from_qapi(&pout.unwrap_or_default());
    let be = AudioBackend::new(dev, pdo_in, pdo_out, driver, running);
    if let Some(c) = clock {
        be.attach_clock(c);
    }
    Ok(be)
}

/// `audio_init()`: a backend for `dev`, or for the first default audiodev that opens.
fn audio_init(r: &mut Reg, dev: Option<Audiodev>) -> Result<Arc<AudioBackend>> {
    let be = match dev {
        Some(dev) => be_new(dev, r.clock.as_ref(), r.running)?,
        None => loop {
            let Some(dev) = r.default_audiodevs.pop_front() else {
                return Err(Error::generic("no default audio driver available"));
            };
            if let Ok(be) = be_new(dev, r.clock.as_ref(), r.running) {
                break be;
            }
        },
    };
    // object_property_try_add_child() on the /audiodevs container.
    if r.backends.iter().any(|b| b.id() == be.id()) {
        be.shutdown();
        return Err(Error::generic(format!(
            "attempt to add duplicate property '{}' to object (type 'container')",
            be.id()
        )));
    }
    r.backends.push(be.clone());
    Ok(be)
}

/// `audio_init_audiodevs()`: a backend for every `-audiodev`. QEMU exits on the first error.
pub fn init_audiodevs() -> Result<()> {
    let mut r = reg();
    for dev in r.audiodevs.clone() {
        audio_init(&mut r, Some(dev))?;
    }
    Ok(())
}

/// `audio_get_default_audio_be()`.
pub fn default_audio_be() -> Result<Arc<AudioBackend>> {
    let mut r = reg();
    if let Some(be) = &r.default_be {
        return Ok(be.clone());
    }
    match audio_init(&mut r, None) {
        Ok(be) => {
            r.default_be = Some(be.clone());
            Ok(be)
        }
        Err(e) => match r.audiodevs.first() {
            Some(dev) => Err(
                e.hint(format!("Perhaps you wanted to use -audio or set audiodev={}?\n", dev.id))
            ),
            None => Err(e),
        },
    }
}

/// `audio_be_check()`: `be`, or the default backend when a device names none.
pub fn be_check(be: Option<Arc<AudioBackend>>) -> Result<Arc<AudioBackend>> {
    match be {
        Some(be) => Ok(be),
        None => default_audio_be(),
    }
}

/// `audio_be_by_name()`: the backend of the audiodev with id `name`.
pub fn be_by_name(name: &str) -> Result<Arc<AudioBackend>> {
    reg()
        .backends
        .iter()
        .find(|b| b.id() == name)
        .cloned()
        .ok_or_else(|| Error::generic(format!("audiodev '{name}' not found")))
}

/// The drivers this build has, in QAPI order.
pub fn driver_names() -> Vec<&'static str> {
    AudiodevDriver::ALL.iter().filter(|d| driver_for(**d).is_some()).map(|d| d.as_str()).collect()
}

/// What `audio_help()` prints.
pub fn help_text() -> String {
    let mut s = String::from("Available audio drivers:\n");
    for name in driver_names() {
        s.push_str(name);
        s.push('\n');
    }
    s
}

/// `qmp_query_audiodevs()`. QEMU prepends each entry, so the list comes out last first.
pub fn query_audiodevs() -> Vec<Audiodev> {
    reg().audiodevs.iter().rev().cloned().collect()
}

/// The machine's virtual clock, once [`attach_clock`] has run. Devices that pace themselves,
/// such as the HDA codecs, put their timers on it.
pub fn clock() -> Option<Arc<Clock>> {
    reg().clock.clone()
}

/// Paces every backend, now and later, by the machine's virtual clock.
pub fn attach_clock(clock: &Arc<Clock>) {
    let backends = {
        let mut r = reg();
        r.clock = Some(clock.clone());
        r.backends.clone()
    };
    for be in backends {
        be.attach_clock(clock);
    }
}

/// Tells every backend the VM started or stopped, the run state handler each QEMU backend
/// registers.
pub fn vm_state_change(running: bool) {
    let backends = {
        let mut r = reg();
        r.running = running;
        r.backends.clone()
    };
    for be in backends {
        be.vm_state_change(running);
    }
}

/// `audio_cleanup()`: closes every backend, which finishes files such as the WAVE header.
pub fn cleanup() {
    let backends = {
        let mut r = reg();
        r.default_be = None;
        std::mem::take(&mut r.backends)
    };
    for be in backends {
        be.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_fills_defaults() {
        let mut dev = Audiodev {
            id: "a".into(),
            timer_period: None,
            u: AudiodevU::for_tag(AudiodevDriver::Wav),
        };
        validate_opts(&mut dev).unwrap();
        assert_eq!(dev.timer_period, Some(10000));
        let (pin, pout) = pdos(&dev.u);
        let out = pout.unwrap();
        assert_eq!(out.mixing_engine, Some(true));
        assert_eq!(out.fixed_settings, Some(true));
        assert_eq!(out.frequency, Some(44100));
        assert_eq!(out.voices, Some(1));
        assert_eq!(pin.unwrap().format, Some(AudioFormat::S16));
    }

    #[test]
    fn validate_rejects_conflicts() {
        let mut p = AudiodevPerDirectionOptions {
            fixed_settings: Some(false),
            frequency: Some(8000),
            ..Default::default()
        };
        let e = validate_per_direction_opts(&mut p).unwrap_err();
        assert_eq!(
            e.to_string(),
            "You can't use frequency, channels or format with fixed-settings=off"
        );
        let mut p = AudiodevPerDirectionOptions {
            mixing_engine: Some(false),
            fixed_settings: Some(true),
            ..Default::default()
        };
        let e = validate_per_direction_opts(&mut p).unwrap_err();
        assert_eq!(e.to_string(), "You can't use fixed-settings without mixeng");
        let mut p =
            AudiodevPerDirectionOptions { mixing_engine: Some(false), ..Default::default() };
        validate_per_direction_opts(&mut p).unwrap();
        assert_eq!(p.voices, Some(i32::MAX as u32));
        assert_eq!(p.fixed_settings, Some(false));
    }
}
