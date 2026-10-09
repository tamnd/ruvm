// SPDX-License-Identifier: GPL-2.0-or-later

//! The audio core, mixing engine and audio backends.
//!
//! This is QEMU's `audio/` directory: the audiodev list `-audiodev` and `-audio` fill
//! ([`registry`]), the backend devices open voices on ([`engine`]), the integer mixing engine
//! and resampler ([`mixeng`]), the layer host drivers plug into ([`pcm`]) and the drivers
//! themselves. The `none` driver plays nothing and records silence, and the `wav` driver writes
//! what the guest plays to a file, which is how the sound cards are checked against QEMU.

#![forbid(unsafe_code)]

pub mod engine;
pub mod mixeng;
pub mod model;
pub mod none;
pub mod pcm;
pub mod registry;
pub mod wav;

pub use engine::{
    AudioBackend, AudioCallback, CaptureHandle, CaptureNotify, CaptureOps, SwVoiceIn, SwVoiceOut,
};
pub use pcm::{AudSettings, Volume};
pub use registry::{
    attach_clock, be_by_name, be_check, cleanup, default_audio_be, help_text, query_audiodevs,
    vm_state_change,
};
