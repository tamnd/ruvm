// SPDX-License-Identifier: GPL-2.0-or-later

//! The `hpet` section, from hw/timer/hpet.c.
//!
//! The comparator timers and `hpet_offset` are on the virtual clock, whose value travels in the
//! `timer` section.

use std::sync::{Arc, LazyLock};

use ruvm_base::err;
use ruvm_hw_timer::hpet::{HPET_CFG_ENABLE, Hpet, HpetTimerVmState, HpetVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Timer;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// `vmstate_hpet_timer`.
static VMSTATE_HPET_TIMER: LazyLock<VmStateDescription<HpetTimerVmState>> = LazyLock::new(|| {
    type S = HpetTimerVmState;
    VmStateDescription::new("hpet_timer").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("tn", |s: &mut S| &mut s.tn),
        VmStateField::scalar("config", |s: &mut S| &mut s.config),
        VmStateField::scalar("cmp", |s: &mut S| &mut s.cmp),
        VmStateField::scalar("fsb", |s: &mut S| &mut s.fsb),
        VmStateField::scalar("period", |s: &mut S| &mut s.period),
        VmStateField::scalar("wrap_flag", |s: &mut S| &mut s.wrap_flag),
        VmStateField::single("qemu_timer", &Timer, |s: &mut S| &mut s.qemu_timer),
    ])
});

/// `vmstate_hpet_rtc_irq_level`.
static VMSTATE_HPET_RTC_IRQ_LEVEL: LazyLock<VmStateDescription<HpetVmState>> =
    LazyLock::new(|| {
        VmStateDescription::new("hpet/rtc_irq_level")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|s: &HpetVmState| s.rtc_irq_level != 0)
            .field(VmStateField::scalar("rtc_irq_level", |s: &mut HpetVmState| {
                &mut s.rtc_irq_level
            }))
    });

/// `vmstate_hpet_offset`, sent while the HPET is enabled.
static VMSTATE_HPET_OFFSET: LazyLock<VmStateDescription<HpetVmState>> = LazyLock::new(|| {
    VmStateDescription::new("hpet/offset")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &HpetVmState| s.config & HPET_CFG_ENABLE != 0)
        .field(VmStateField::scalar("hpet_offset", |s: &mut HpetVmState| &mut s.hpet_offset))
});

/// `vmstate_hpet`. `hpet_pre_save()` and `hpet_post_load()` are in [`Hpet::vmstate_save`] and
/// [`Hpet::vmstate_load`].
pub(crate) static VMSTATE_HPET: LazyLock<VmStateDescription<HpetVmState>> = LazyLock::new(|| {
    type S = HpetVmState;
    VmStateDescription::new("hpet")
        .version_id(2)
        .minimum_version_id(2)
        .fields([
            VmStateField::scalar("config", |s: &mut S| &mut s.config),
            VmStateField::scalar("isr", |s: &mut S| &mut s.isr),
            VmStateField::scalar("hpet_counter", |s: &mut S| &mut s.hpet_counter),
            VmStateField::scalar("num_timers_save", |s: &mut S| &mut s.num_timers_save),
            VmStateField::validate("num_timers must match", |s: &S, _| {
                s.num_timers == s.num_timers_save
            }),
            VmStateField::struct_varray_alloc(
                "timer",
                |s: &S| usize::from(s.num_timers_save),
                &VMSTATE_HPET_TIMER,
                |s: &mut S| &mut s.timer,
            ),
        ])
        .subsection(&VMSTATE_HPET_RTC_IRQ_LEVEL)
        .subsection(&VMSTATE_HPET_OFFSET)
});

/// Registers the `hpet` section of `hpet`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register(savevm: &mut SaveVm, hpet: &Arc<Hpet>) {
    let (get, put) = (Arc::clone(hpet), Arc::clone(hpet));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_HPET,
        move || Ok(get.vmstate_save()),
        move |s| put.vmstate_load(&s).map_err(|e| err!("{e}")),
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    fn state() -> HpetVmState {
        HpetVmState {
            config: 1,
            isr: 0,
            hpet_counter: 400,
            num_timers_save: 3,
            timer: (0..3)
                .map(|tn| HpetTimerVmState { tn, qemu_timer: -1, ..HpetTimerVmState::default() })
                .collect(),
            rtc_irq_level: 0,
            hpet_offset: 0x1234,
            num_timers: 3,
        }
    }

    #[test]
    fn hpet_layout_matches_qemu() {
        let mut s = state();
        s.timer[2].qemu_timer = 5_010_000;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_HPET, &mut s).unwrap();
        let b = f.into_inner();
        // 3 * 8 + 1, three timers of 42 bytes, and the offset subsection while enabled.
        let sub = 1 + 1 + "hpet/offset".len() + 4 + 8;
        assert_eq!(b.len(), 25 + 3 * 42 + sub);
        assert_eq!(b[24], 3);
        assert_eq!(&b[25 + 2 * 42 + 34..25 + 3 * 42], &5_010_000i64.to_be_bytes());
        assert_eq!(&b[b.len() - 8..], &0x1234u64.to_be_bytes());

        let mut back = state();
        back.timer.clear();
        back.hpet_offset = 0;
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_HPET, &mut back, 2).unwrap();
        assert_eq!(back, s);

        // A destination with another number of timers refuses the stream.
        let mut other = state();
        other.num_timers = 4;
        let r = vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_HPET, &mut other, 2);
        assert!(r.is_err());
    }
}
