// SPDX-License-Identifier: GPL-2.0-or-later

//! The `i8254` section of the PIT, from hw/timer/i8254_common.c, and the `pcspk` section of the
//! PC speaker, from hw/audio/pcspk.c.
//!
//! The PIT's times are on the virtual clock, whose value travels in the `timer` section.

use std::sync::{Arc, LazyLock};

use ruvm_hw_timer::i8254::{I8254, I8254VmState, PitChannelState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use crate::pc::{PcSpeaker, PcSpkVmState};

/// `vmstate_pit_channel`. `irq_disabled` is not in it: only channel 0's goes out, in front of
/// the channels.
static VMSTATE_PIT_CHANNEL: LazyLock<VmStateDescription<PitChannelState>> = LazyLock::new(|| {
    type S = PitChannelState;
    VmStateDescription::new("pit channel").version_id(2).minimum_version_id(2).fields([
        VmStateField::scalar("count", |s: &mut S| &mut s.count),
        VmStateField::scalar("latched_count", |s: &mut S| &mut s.latched_count),
        VmStateField::scalar("count_latched", |s: &mut S| &mut s.count_latched),
        VmStateField::scalar("status_latched", |s: &mut S| &mut s.status_latched),
        VmStateField::scalar("status", |s: &mut S| &mut s.status),
        VmStateField::scalar("read_state", |s: &mut S| &mut s.read_state),
        VmStateField::scalar("write_state", |s: &mut S| &mut s.write_state),
        VmStateField::scalar("write_latch", |s: &mut S| &mut s.write_latch),
        VmStateField::scalar("rw_mode", |s: &mut S| &mut s.rw_mode),
        VmStateField::scalar("mode", |s: &mut S| &mut s.mode),
        VmStateField::scalar("bcd", |s: &mut S| &mut s.bcd),
        VmStateField::scalar("gate", |s: &mut S| &mut s.gate),
        VmStateField::scalar("count_load_time", |s: &mut S| &mut s.count_load_time),
        VmStateField::scalar("next_transition_time", |s: &mut S| &mut s.next_transition_time),
    ])
});

/// `vmstate_pit_common`. The last field used to be the IRQ timer and is
/// `channels[0].next_transition_time` again; the post_load hook is [`I8254::vmstate_load`].
pub(crate) static VMSTATE_PIT: LazyLock<VmStateDescription<I8254VmState>> = LazyLock::new(|| {
    type S = I8254VmState;
    VmStateDescription::new("i8254").version_id(3).minimum_version_id(2).fields([
        VmStateField::scalar("channels[0].irq_disabled", |s: &mut S| {
            &mut s.channels[0].irq_disabled
        })
        .version(3),
        VmStateField::struct_array("channels", &VMSTATE_PIT_CHANNEL, |s: &mut S| &mut s.channels)
            .version(2),
        VmStateField::scalar("channels[0].next_transition_time", |s: &mut S| {
            &mut s.channels[0].next_transition_time
        }),
    ])
});

/// `vmstate_spk`.
pub(crate) static VMSTATE_PCSPK: LazyLock<VmStateDescription<PcSpkVmState>> = LazyLock::new(|| {
    type S = PcSpkVmState;
    VmStateDescription::new("pcspk").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("data_on", |s: &mut S| &mut s.data_on),
        VmStateField::scalar("dummy_refresh_clock", |s: &mut S| &mut s.dummy_refresh_clock),
    ])
});

/// Registers the `i8254` section of `pit`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, pit: &Arc<I8254>) {
    let (get, put) = (Arc::clone(pit), Arc::clone(pit));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PIT,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// Registers the `pcspk` section of `spk`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register_pcspk(savevm: &mut SaveVm, spk: &Arc<PcSpeaker>) {
    let (get, put) = (Arc::clone(spk), Arc::clone(spk));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PCSPK,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn i8254_layout_matches_qemu() {
        let mut s = I8254VmState::default();
        s.channels[0].irq_disabled = 1;
        s.channels[0].count = 0x10000;
        s.channels[0].next_transition_time = 0x1122_3344_5566_7788;
        s.channels[2].gate = 1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PIT, &mut s).unwrap();
        let b = f.into_inner();
        // irq_disabled, 3 channels of 32 bytes, next_transition_time again.
        assert_eq!(b.len(), 4 + 3 * 32 + 8);
        assert_eq!(&b[..8], &[0, 0, 0, 1, 0, 1, 0, 0]);
        assert_eq!(&b[4 + 24..4 + 32], &0x1122_3344_5566_7788i64.to_be_bytes());
        assert_eq!(&b[4 + 2 * 32 + 15], &1);
        assert_eq!(&b[4 + 3 * 32..], &0x1122_3344_5566_7788i64.to_be_bytes());

        let mut back = I8254VmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PIT, &mut back, 3).unwrap();
        assert_eq!(back, s);

        // Version 2 has no irq_disabled in front.
        let mut v2 = I8254VmState::default();
        vmstate_load_state(&mut StreamReader::new(&b[4..]), &VMSTATE_PIT, &mut v2, 2).unwrap();
        assert_eq!(v2.channels[0].irq_disabled, 0);
        assert_eq!(v2.channels[2].gate, 1);
    }

    #[test]
    fn pcspk_layout_matches_qemu() {
        let mut s = PcSpkVmState { data_on: 1, dummy_refresh_clock: 0x10 };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PCSPK, &mut s).unwrap();
        assert_eq!(f.as_bytes(), &[1, 0x10]);
    }
}
