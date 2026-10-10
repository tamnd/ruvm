// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio core, mixing engine and audio backends.
//!
//! This is QEMU's `audio/` directory: the audiodev list `-audiodev` and `-audio` fill
//! ([`registry`]), the backend devices open voices on ([`engine`]), the integer mixing engine
//! and resampler ([`mixeng`]), the layer host drivers plug into ([`pcm`]) and the drivers
//! themselves. The `none` driver plays nothing and records silence, and the `wav` driver writes
//! what the guest plays to a file, which is how the sound cards are checked against QEMU. The
//! host drivers sit behind cargo features: `audio-alsa` adds `alsa`, `audio-pa` adds `pa` and
//! `audio-pipewire` adds `pipewire`. On macOS `audio-coreaudio` adds `coreaudio`, which is the one
//! driver with unsafe code of its own, its calls into CoreAudio.

#![cfg_attr(not(all(feature = "audio-coreaudio", target_os = "macos")), forbid(unsafe_code))]
#![cfg_attr(all(feature = "audio-coreaudio", target_os = "macos"), deny(unsafe_code))]

#[cfg(feature = "audio-alsa")]
pub mod alsa;
#[cfg(all(feature = "audio-coreaudio", target_os = "macos"))]
pub mod coreaudio;
pub mod engine;
pub mod mixeng;
pub mod model;
pub mod none;
#[cfg(feature = "audio-pa")]
pub mod pa;
pub mod pcm;
#[cfg(feature = "audio-pipewire")]
pub mod pipewire;
pub mod registry;
pub mod wav;

pub use engine::{
    AudioBackend, AudioCallback, CaptureHandle, CaptureNotify, CaptureOps, SwVoiceIn, SwVoiceOut,
};
pub use pcm::{AudSettings, Volume, set_application_name};
pub use registry::{
    attach_clock, be_by_name, be_check, cleanup, default_audio_be, help_text, query_audiodevs,
    vm_state_change,
};
