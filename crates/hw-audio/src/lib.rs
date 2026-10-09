// SPDX-License-Identifier: GPL-2.0-or-later

//! HDA, AC97, SB16, ES1370, virtio-snd and board audio devices.
//!
//! The models talk to the host only through [`ruvm_audio::AudioBackend`]: each opens voices
//! on the backend its `audiodev` property names and moves samples in the voice callbacks the
//! backend's timer drives. So far there is the AC97 controller.

#![forbid(unsafe_code)]

pub mod ac97;

pub use ac97::{Ac97, TYPE_AC97};

/// `PCI_VENDOR_ID_INTEL`.
pub(crate) const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// `PCI_CLASS_MULTIMEDIA_AUDIO`.
pub(crate) const PCI_CLASS_MULTIMEDIA_AUDIO: u16 = 0x0401;
/// `PCI_STATUS_DEVSEL_MEDIUM`.
pub(crate) const PCI_STATUS_DEVSEL_MEDIUM: u16 = 0x200;

/// The cards `-audio model=` can name on a PC, registered in QEMU's order.
pub fn register_pc_models() {
    ruvm_audio::model::register("ac97", "Intel 82801AA AC97 Audio", TYPE_AC97);
}
