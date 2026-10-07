// SPDX-License-Identifier: GPL-2.0-or-later

//! The `i8259` sections of the master and slave 8259, `vmstate_pic_common` from
//! hw/intc/i8259_common.c.

use std::sync::{Arc, LazyLock};

use ruvm_hw_intc::i8259::{I8259, I8259Pair, PicCommonState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

type Vmsd = VmStateDescription<PicCommonState>;

/// `vmstate_pic_ltim`: only sent when the chip is level triggered as a whole.
static PIC_LTIM: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("i8259/ltim")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|s: &PicCommonState| s.ltim != 0)
        .field(VmStateField::scalar("ltim", |s: &mut PicCommonState| &mut s.ltim))
});

/// `vmstate_pic_common`.
pub(crate) static VMSTATE_PIC: LazyLock<Vmsd> = LazyLock::new(|| {
    VmStateDescription::new("i8259")
        .version_id(1)
        .minimum_version_id(1)
        .fields([
            VmStateField::scalar("last_irr", |s: &mut PicCommonState| &mut s.last_irr),
            VmStateField::scalar("irr", |s: &mut PicCommonState| &mut s.irr),
            VmStateField::scalar("imr", |s: &mut PicCommonState| &mut s.imr),
            VmStateField::scalar("isr", |s: &mut PicCommonState| &mut s.isr),
            VmStateField::scalar("priority_add", |s: &mut PicCommonState| &mut s.priority_add),
            VmStateField::scalar("irq_base", |s: &mut PicCommonState| &mut s.irq_base),
            VmStateField::scalar("read_reg_select", |s: &mut PicCommonState| {
                &mut s.read_reg_select
            }),
            VmStateField::scalar("poll", |s: &mut PicCommonState| &mut s.poll),
            VmStateField::scalar("special_mask", |s: &mut PicCommonState| &mut s.special_mask),
            VmStateField::scalar("init_state", |s: &mut PicCommonState| &mut s.init_state),
            VmStateField::scalar("auto_eoi", |s: &mut PicCommonState| &mut s.auto_eoi),
            VmStateField::scalar("rotate_on_auto_eoi", |s: &mut PicCommonState| {
                &mut s.rotate_on_auto_eoi
            }),
            VmStateField::scalar("special_fully_nested_mode", |s: &mut PicCommonState| {
                &mut s.special_fully_nested_mode
            }),
            VmStateField::scalar("init4", |s: &mut PicCommonState| &mut s.init4),
            VmStateField::scalar("single_mode", |s: &mut PicCommonState| &mut s.single_mode),
            VmStateField::scalar("elcr", |s: &mut PicCommonState| &mut s.elcr),
        ])
        .subsection(&PIC_LTIM)
});

fn register_chip(savevm: &mut SaveVm, instance_id: u32, pic: &Arc<I8259>) {
    let (get, put) = (Arc::clone(pic), Arc::clone(pic));
    savevm.register_vmsd(
        "",
        Some(instance_id),
        &VMSTATE_PIC,
        move || Ok(get.state()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// Registers the two `i8259` sections: instance 0 is the master, 1 the slave, in the order
/// `i8259_init()` creates them.
pub(crate) fn register(savevm: &mut SaveVm, pair: &I8259Pair) {
    register_chip(savevm, 0, &pair.master);
    register_chip(savevm, 1, &pair.slave);
}
