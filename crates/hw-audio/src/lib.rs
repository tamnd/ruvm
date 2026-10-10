// SPDX-License-Identifier: GPL-2.0-or-later

//! HDA, AC97, SB16, ES1370, virtio-snd and board audio devices.
//!
//! The models talk to the host only through [`ruvm_audio::AudioBackend`]: each opens voices
//! on the backend its `audiodev` property names and moves samples in the voice callbacks the
//! backend's timer drives. So far there are the AC97 controller and the Intel HDA controllers
//! with QEMU's three HDA codecs, the Sound Blaster 16 with the ISA DMA controllers it reads
//! from, the sound of the PC speaker, and virtio-sound, a device model for the transports of
//! `ruvm-hw-virtio`.

#![forbid(unsafe_code)]

pub mod ac97;
pub mod hda_codec;
pub mod i8257;
pub mod intel_hda;
pub mod pcspk;
mod portio;
pub mod sb16;
pub mod virtio_snd;

pub use ac97::{Ac97, TYPE_AC97};
pub use hda_codec::{HdaCodecKind, TYPE_HDA_DUPLEX, TYPE_HDA_MICRO, TYPE_HDA_OUTPUT};
pub use i8257::{I8257, IsaDma};
pub use intel_hda::{IntelHda, TYPE_ICH9_INTEL_HDA, TYPE_INTEL_HDA};
pub use pcspk::PcSpkAudio;
pub use sb16::{Sb16, Sb16Config, TYPE_SB16};
pub use virtio_snd::{
    DeviceAccess, TYPE_VIRTIO_SND, TYPE_VIRTIO_SND_PCI, VirtioSnd, VirtioSndConf,
};

/// `PCI_VENDOR_ID_INTEL`.
pub(crate) const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// `PCI_CLASS_MULTIMEDIA_AUDIO`.
pub(crate) const PCI_CLASS_MULTIMEDIA_AUDIO: u16 = 0x0401;
/// `PCI_CLASS_MULTIMEDIA_HD_AUDIO`.
pub(crate) const PCI_CLASS_MULTIMEDIA_HD_AUDIO: u16 = 0x0403;
/// `PCI_STATUS_DEVSEL_MEDIUM`.
pub(crate) const PCI_STATUS_DEVSEL_MEDIUM: u16 = 0x200;

/// The cards `-audio model=` can name on a PC, registered in QEMU's order.
pub fn register_pc_models() {
    ruvm_audio::model::register("ac97", "Intel 82801AA AC97 Audio", TYPE_AC97);
    // intel_hda_and_codec_init(): the controller plus an hda-duplex codec.
    ruvm_audio::model::register("hda", "Intel HD Audio", TYPE_INTEL_HDA);
    ruvm_audio::model::register("sb16", "Creative Sound Blaster 16", TYPE_SB16);
    ruvm_audio::model::register("virtio", "Virtio Sound", TYPE_VIRTIO_SND_PCI);
}
