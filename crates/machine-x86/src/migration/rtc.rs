// SPDX-License-Identifier: GPL-2.0-or-later

//! The `mc146818rtc` section, from hw/rtc/mc146818rtc.c.
//!
//! The timers and times are on the RTC clock, the host wall clock by default on both QEMU and
//! ruvm, so they mean the same on either side without help from the `timer` section.

use std::sync::{Arc, LazyLock};

use ruvm_hw_timer::mc146818::{Mc146818Rtc, Mc146818VmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// `vmstate_rtc_irq_reinject_on_ack_count`.
static VMSTATE_RTC_IRQ_REINJECT_ON_ACK_COUNT: LazyLock<VmStateDescription<Mc146818VmState>> =
    LazyLock::new(|| {
        type S = Mc146818VmState;
        VmStateDescription::new("mc146818rtc/irq_reinject_on_ack_count")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &S| s.irq_reinject_on_ack_count != 0)
            .field(VmStateField::scalar("irq_reinject_on_ack_count", |s: &mut S| {
                &mut s.irq_reinject_on_ack_count
            }))
    });

/// `vmstate_rtc`. `rtc_pre_save()` and `rtc_post_load()` are in [`Mc146818Rtc::vmstate_save`]
/// and [`Mc146818Rtc::vmstate_load`].
pub(crate) static VMSTATE_RTC: LazyLock<VmStateDescription<Mc146818VmState>> =
    LazyLock::new(|| {
        type S = Mc146818VmState;
        VmStateDescription::new("mc146818rtc")
            .version_id(3)
            .minimum_version_id(3)
            .fields([
                VmStateField::buffer("cmos_data", |s: &mut S| &mut s.cmos_data),
                VmStateField::scalar("cmos_index", |s: &mut S| &mut s.cmos_index),
                VmStateField::unused(7 * 4),
                VmStateField::single("periodic_timer", &Timer, |s: &mut S| &mut s.periodic_timer),
                VmStateField::scalar("next_periodic_time", |s: &mut S| &mut s.next_periodic_time),
                VmStateField::unused(3 * 8),
                VmStateField::scalar("irq_coalesced", |s: &mut S| &mut s.irq_coalesced),
                VmStateField::scalar("period", |s: &mut S| &mut s.period),
                VmStateField::scalar("base_rtc", |s: &mut S| &mut s.base_rtc),
                VmStateField::scalar("last_update", |s: &mut S| &mut s.last_update),
                VmStateField::scalar("offset", |s: &mut S| &mut s.offset),
                VmStateField::single("update_timer", &Timer, |s: &mut S| &mut s.update_timer),
                VmStateField::scalar("next_alarm_time", |s: &mut S| &mut s.next_alarm_time),
            ])
            .subsection(&VMSTATE_RTC_IRQ_REINJECT_ON_ACK_COUNT)
    });

/// Registers the `mc146818rtc` section of `rtc`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, rtc: &Arc<Mc146818Rtc>) {
    let (get, put) = (Arc::clone(rtc), Arc::clone(rtc));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_RTC,
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
    fn rtc_layout_matches_qemu() {
        let mut s = Mc146818VmState { cmos_index: 0x0b, update_timer: 77, ..Default::default() };
        s.cmos_data[0x0a] = 0x26;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_RTC, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 245);
        assert_eq!((b[0x0a], b[128]), (0x26, 0x0b));
        // An idle periodic timer goes out as -1.
        assert_eq!(&b[157..165], &(-1i64).to_be_bytes());
        assert_eq!(&b[229..237], &77i64.to_be_bytes());

        let mut back = Mc146818VmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_RTC, &mut back, 3).unwrap();
        assert_eq!(back, s);

        // The reinjection count is a subsection, only sent when it is not 0.
        s.irq_reinject_on_ack_count = 5;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_RTC, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 245 + 2 + "mc146818rtc/irq_reinject_on_ack_count".len() + 4 + 2);
        assert_eq!(&b[b.len() - 2..], &[0, 5]);
    }
}
