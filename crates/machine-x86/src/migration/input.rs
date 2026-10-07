// SPDX-License-Identifier: GPL-2.0-or-later

//! The `ps2kbd`, `ps2mouse` and `pckbd` sections of the i8042, from hw/input/ps2.c and
//! hw/input/pckbd.c, and the `vmmouse` section of hw/i386/vmmouse.c, which is only parsed.
//!
//! The i8042 throttle timer is on the virtual clock, but only whether it is pending goes on the
//! wire, as a flag in `pckbd/extended_state`.

use std::sync::{Arc, LazyLock};

use ruvm_hw_input::pckbd::{I8042, I8042VmState};
use ruvm_hw_input::ps2::{Ps2CommonVmState, Ps2KbdVmState, Ps2MouseVmState};
use ruvm_migration::SaveVm;
use ruvm_vmstate::info::Int32Equal;
use ruvm_vmstate::{VmStateDescription, VmStateField};

/// `vmstate_ps2_common`. `queue.cwptr` is not in it: the keyboard sends its own in a subsection.
static VMSTATE_PS2_COMMON: LazyLock<VmStateDescription<Ps2CommonVmState>> = LazyLock::new(|| {
    type S = Ps2CommonVmState;
    VmStateDescription::new("PS2 Common State").version_id(3).minimum_version_id(2).fields([
        VmStateField::scalar("write_cmd", |s: &mut S| &mut s.write_cmd),
        VmStateField::scalar("queue.rptr", |s: &mut S| &mut s.rptr),
        VmStateField::scalar("queue.wptr", |s: &mut S| &mut s.wptr),
        VmStateField::scalar("queue.count", |s: &mut S| &mut s.count),
        VmStateField::buffer("queue.data", |s: &mut S| &mut s.data),
    ])
});

/// `vmstate_ps2_keyboard_ledstate`. Its post_load is in [`I8042::kbd_vmstate_load`].
static VMSTATE_PS2_KEYBOARD_LEDSTATE: LazyLock<VmStateDescription<Ps2KbdVmState>> =
    LazyLock::new(|| {
        type S = Ps2KbdVmState;
        VmStateDescription::new("ps2kbd/ledstate")
            .version_id(3)
            .minimum_version_id(2)
            .needed(S::ledstate_needed)
            .field(VmStateField::scalar("ledstate", |s: &mut S| &mut s.ledstate))
    });

/// `vmstate_ps2_keyboard_need_high_bit`.
static VMSTATE_PS2_KEYBOARD_NEED_HIGH_BIT: LazyLock<VmStateDescription<Ps2KbdVmState>> =
    LazyLock::new(|| {
        type S = Ps2KbdVmState;
        VmStateDescription::new("ps2kbd/need_high_bit")
            .version_id(1)
            .minimum_version_id(1)
            .needed(S::need_high_bit_needed)
            .field(VmStateField::scalar("need_high_bit", |s: &mut S| &mut s.need_high_bit))
    });

/// `vmstate_ps2_keyboard_cqueue`.
static VMSTATE_PS2_KEYBOARD_CQUEUE: LazyLock<VmStateDescription<Ps2KbdVmState>> =
    LazyLock::new(|| {
        type S = Ps2KbdVmState;
        VmStateDescription::new("ps2kbd/command_reply_queue").needed(S::cqueue_needed).field(
            VmStateField::scalar("parent_obj.queue.cwptr", |s: &mut S| &mut s.parent_obj.cwptr),
        )
    });

/// `vmstate_ps2_keyboard`. The version part of `ps2_kbd_post_load()` is here, the rest is
/// [`I8042::kbd_vmstate_load`].
pub(crate) static VMSTATE_PS2_KEYBOARD: LazyLock<VmStateDescription<Ps2KbdVmState>> =
    LazyLock::new(|| {
        type S = Ps2KbdVmState;
        VmStateDescription::new("ps2kbd")
            .version_id(3)
            .minimum_version_id(2)
            .post_load(|s: &mut S, version_id| {
                if version_id == 2 {
                    s.scancode_set = 2;
                }
                0
            })
            .fields([
                VmStateField::structure("parent_obj", &VMSTATE_PS2_COMMON, |s: &mut S| {
                    &mut s.parent_obj
                }),
                VmStateField::scalar("scan_enabled", |s: &mut S| &mut s.scan_enabled),
                VmStateField::scalar("translate", |s: &mut S| &mut s.translate),
                VmStateField::scalar("scancode_set", |s: &mut S| &mut s.scancode_set).version(3),
            ])
            .subsection(&VMSTATE_PS2_KEYBOARD_LEDSTATE)
            .subsection(&VMSTATE_PS2_KEYBOARD_NEED_HIGH_BIT)
            .subsection(&VMSTATE_PS2_KEYBOARD_CQUEUE)
    });

/// `vmstate_ps2_mouse`. `ps2_mouse_post_load()` is [`I8042::mouse_vmstate_load`].
pub(crate) static VMSTATE_PS2_MOUSE: LazyLock<VmStateDescription<Ps2MouseVmState>> =
    LazyLock::new(|| {
        type S = Ps2MouseVmState;
        VmStateDescription::new("ps2mouse").version_id(2).minimum_version_id(2).fields([
            VmStateField::structure("parent_obj", &VMSTATE_PS2_COMMON, |s: &mut S| {
                &mut s.parent_obj
            }),
            VmStateField::scalar("mouse_status", |s: &mut S| &mut s.mouse_status),
            VmStateField::scalar("mouse_resolution", |s: &mut S| &mut s.mouse_resolution),
            VmStateField::scalar("mouse_sample_rate", |s: &mut S| &mut s.mouse_sample_rate),
            VmStateField::scalar("mouse_wrap", |s: &mut S| &mut s.mouse_wrap),
            VmStateField::scalar("mouse_type", |s: &mut S| &mut s.mouse_type),
            VmStateField::scalar("mouse_detect_state", |s: &mut S| &mut s.mouse_detect_state),
            VmStateField::scalar("mouse_dx", |s: &mut S| &mut s.mouse_dx),
            VmStateField::scalar("mouse_dy", |s: &mut S| &mut s.mouse_dy),
            VmStateField::scalar("mouse_dz", |s: &mut S| &mut s.mouse_dz),
            VmStateField::scalar("mouse_buttons", |s: &mut S| &mut s.mouse_buttons),
        ])
    });

type S = I8042VmState;

/// `vmstate_kbd_outport`.
static VMSTATE_KBD_OUTPORT: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("pckbd_outport")
        .version_id(1)
        .minimum_version_id(1)
        .post_load(|s: &mut S, _| {
            s.outport_present = true;
            0
        })
        .needed(S::outport_needed)
        .field(VmStateField::scalar("outport", |s: &mut S| &mut s.outport))
});

/// `vmstate_kbd_extended_state`. The pre_save is in [`I8042::vmstate_save`] and the throttle
/// part of the post_load in [`I8042::vmstate_load`].
static VMSTATE_KBD_EXTENDED_STATE: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("pckbd/extended_state")
        .post_load(|s: &mut S, _| {
            s.extended_state_loaded = true;
            0
        })
        .needed(S::extended_state_needed)
        .fields([
            VmStateField::scalar("migration_flags", |s: &mut S| &mut s.migration_flags),
            VmStateField::scalar("obsrc", |s: &mut S| &mut s.obsrc),
            VmStateField::scalar("obdata", |s: &mut S| &mut s.obdata),
            VmStateField::scalar("cbdata", |s: &mut S| &mut s.cbdata),
        ])
});

/// `vmstate_kbd`. `kbd_pre_load()` is here; `kbd_pre_save()` is [`I8042::vmstate_save`] and
/// `kbd_post_load()` is [`I8042::vmstate_load`].
static VMSTATE_KBD: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("pckbd")
        .version_id(3)
        .minimum_version_id(3)
        .pre_load(|s: &mut S| {
            s.outport_present = false;
            s.extended_state_loaded = false;
            0
        })
        .fields([
            VmStateField::scalar("write_cmd", |s: &mut S| &mut s.write_cmd),
            VmStateField::scalar("status", |s: &mut S| &mut s.status),
            VmStateField::scalar("mode", |s: &mut S| &mut s.mode),
            VmStateField::scalar("pending_tmp", |s: &mut S| &mut s.pending_tmp),
        ])
        .subsection(&VMSTATE_KBD_OUTPORT)
        .subsection(&VMSTATE_KBD_EXTENDED_STATE)
});

/// `vmstate_kbd_isa`: the `KBDState` as the struct field `kbd`.
pub(crate) static VMSTATE_KBD_ISA: LazyLock<VmStateDescription<S>> = LazyLock::new(|| {
    VmStateDescription::new("pckbd")
        .version_id(3)
        .minimum_version_id(3)
        .field(VmStateField::structure("kbd", &VMSTATE_KBD, |s: &mut S| s))
});

/// `VMMOUSE_QUEUE_SIZE`.
const VMMOUSE_QUEUE_SIZE: usize = 1024;

/// The fields of `VMMouseState`. There is no vmmouse model, so the section is only parsed.
#[derive(Clone, Debug, PartialEq, Eq)]
struct VmMouseVmState {
    queue_size: i32,
    queue: [u32; VMMOUSE_QUEUE_SIZE],
    nb_queue: u16,
    status: u16,
    absolute: u8,
}

impl Default for VmMouseVmState {
    fn default() -> Self {
        VmMouseVmState {
            queue_size: VMMOUSE_QUEUE_SIZE as i32,
            queue: [0; VMMOUSE_QUEUE_SIZE],
            nb_queue: 0,
            status: 0,
            absolute: 0,
        }
    }
}

/// `vmstate_vmmouse`.
static VMSTATE_VMMOUSE: LazyLock<VmStateDescription<VmMouseVmState>> = LazyLock::new(|| {
    type S = VmMouseVmState;
    VmStateDescription::new("vmmouse").fields([
        VmStateField::single("queue_size", &Int32Equal, |s: &mut S| &mut s.queue_size),
        VmStateField::array("queue", |s: &mut S| &mut s.queue),
        VmStateField::scalar("nb_queue", |s: &mut S| &mut s.nb_queue),
        VmStateField::scalar("status", |s: &mut S| &mut s.status),
        VmStateField::scalar("absolute", |s: &mut S| &mut s.absolute),
    ])
});

/// Registers the `ps2kbd`, `ps2mouse` and `pckbd` sections of `i8042`, then `vmmouse` when the
/// machine has a VMware port, all instance 0 with no path prefix and in QEMU's order.
pub(crate) fn register(savevm: &mut SaveVm, i8042: &Arc<I8042>, vmport: bool) {
    let (get, put) = (Arc::clone(i8042), Arc::clone(i8042));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PS2_KEYBOARD,
        move || Ok(get.kbd_vmstate_save()),
        move |s| {
            put.kbd_vmstate_load(&s);
            Ok(())
        },
    );
    let (get, put) = (Arc::clone(i8042), Arc::clone(i8042));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PS2_MOUSE,
        move || Ok(get.mouse_vmstate_save()),
        move |s| {
            put.mouse_vmstate_load(&s);
            Ok(())
        },
    );
    let (get, put) = (Arc::clone(i8042), Arc::clone(i8042));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_KBD_ISA,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
    if vmport {
        savevm.register_discard_with("", Some(0), &VMSTATE_VMMOUSE, VmMouseVmState::default);
    }
}

#[cfg(test)]
mod tests {
    use ruvm_base::ClockType;
    use ruvm_hw_core::Clock;
    use ruvm_hw_input::pckbd::I8042Props;
    use ruvm_hw_input::ps2::{Ps2Kbd, Ps2Mouse};
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn ps2kbd_layout_matches_qemu() {
        let mut s = Ps2Kbd::new().vmstate_save();
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PS2_KEYBOARD, &mut s).unwrap();
        // write_cmd, rptr, wptr, count, 256 bytes of queue, scan_enabled, translate,
        // scancode_set and no subsections.
        assert_eq!(f.as_bytes().len(), 4 * 4 + 256 + 3 * 4);
        assert_eq!(&f.as_bytes()[..4], &(-1i32).to_be_bytes());

        s.ledstate = 2;
        s.need_high_bit = true;
        s.parent_obj.cwptr = 3;
        s.parent_obj.count = 3;
        s.parent_obj.wptr = 3;
        s.parent_obj.data[..3].copy_from_slice(&[0xfa, 0xab, 0x83]);
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PS2_KEYBOARD, &mut s).unwrap();
        let b = f.into_inner();
        let subs = (2 + "ps2kbd/ledstate".len() + 4 + 4)
            + (2 + "ps2kbd/need_high_bit".len() + 4 + 1)
            + (2 + "ps2kbd/command_reply_queue".len() + 4 + 4);
        assert_eq!(b.len(), 4 * 4 + 256 + 3 * 4 + subs);
        let mut back = Ps2Kbd::new().vmstate_save();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PS2_KEYBOARD, &mut back, 3)
            .unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn ps2kbd_version_2_uses_scancode_set_2() {
        let mut s = Ps2Kbd::new().vmstate_save();
        s.scancode_set = 1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PS2_KEYBOARD, &mut s).unwrap();
        // A version 2 stream stops before scancode_set.
        let b = f.into_inner();
        let mut back = s.clone();
        vmstate_load_state(
            &mut StreamReader::new(&b[..b.len() - 4]),
            &VMSTATE_PS2_KEYBOARD,
            &mut back,
            2,
        )
        .unwrap();
        assert_eq!(back.scancode_set, 2);
    }

    #[test]
    fn ps2mouse_layout_matches_qemu() {
        let mut s = Ps2Mouse::new().vmstate_save();
        s.mouse_dx = -3;
        s.mouse_buttons = 1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PS2_MOUSE, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 4 * 4 + 256 + 6 + 3 * 4 + 1);
        assert_eq!(&b[4 * 4 + 256 + 6..][..4], &(-3i32).to_be_bytes());
        let mut back = Ps2Mouse::new().vmstate_save();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_PS2_MOUSE, &mut back, 2).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn pckbd_layout_matches_qemu() {
        let i8042 = I8042::new(Clock::manual(ClockType::Virtual), I8042Props::default()).unwrap();
        let mut s = i8042.vmstate_save();
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_KBD_ISA, &mut s).unwrap();
        // write_cmd, status, mode, pending_tmp, then the extended state, which is on by default
        // as QEMU's "extended-state" property is; the outport is the default.
        let ext = 2 + "pckbd/extended_state".len() + 4 + 10;
        assert_eq!(f.as_bytes().len(), 4 + ext);

        s.extended_state = false;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_KBD_ISA, &mut s).unwrap();
        assert_eq!(f.as_bytes().len(), 4);

        s.extended_state = true;
        s.outport ^= 0x01;
        s.obsrc = 1;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_KBD_ISA, &mut s).unwrap();
        let b = f.into_inner();
        let subs = (2 + "pckbd_outport".len() + 4 + 1) + ext;
        assert_eq!(b.len(), 4 + subs);

        let mut back =
            I8042VmState { outport_present: true, extended_state_loaded: true, ..s.clone() };
        back.obsrc = 0;
        // What follows the section in a stream: the field at the end peeks past it.
        let base = [&b[..4], &[0]].concat();
        vmstate_load_state(&mut StreamReader::new(&base), &VMSTATE_KBD_ISA, &mut back, 3).unwrap();
        assert!(!back.outport_present);
        assert!(!back.extended_state_loaded);
        let b = [&b[..], &[0]].concat();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_KBD_ISA, &mut back, 3).unwrap();
        assert!(back.outport_present);
        assert!(back.extended_state_loaded);
        assert_eq!((back.outport, back.obsrc), (s.outport, 1));
    }

    #[test]
    fn vmmouse_is_parsed() {
        let mut s = VmMouseVmState { nb_queue: 2, absolute: 1, ..VmMouseVmState::default() };
        s.queue[1] = 0x1234;
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_VMMOUSE, &mut s).unwrap();
        let b = f.into_inner();
        assert_eq!(b.len(), 4 + 4 * VMMOUSE_QUEUE_SIZE + 2 + 2 + 1);
        let mut back = VmMouseVmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_VMMOUSE, &mut back, 0).unwrap();
        assert_eq!(back, s);

        // QEMU refuses a queue of another size.
        let mut bad = b.clone();
        bad[2] = 0;
        let mut back = VmMouseVmState::default();
        assert!(
            vmstate_load_state(&mut StreamReader::new(&bad), &VMSTATE_VMMOUSE, &mut back, 0)
                .is_err()
        );
    }
}
