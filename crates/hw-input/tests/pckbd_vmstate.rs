// SPDX-License-Identifier: GPL-2.0-or-later

//! Save and load of the state the `pckbd`, `ps2kbd` and `ps2mouse` VMStates carry.

use std::sync::Arc;

use ruvm_base::ClockType;
use ruvm_hw_core::Clock;
use ruvm_hw_input::pckbd::*;
use ruvm_hw_input::ps2::*;

const KEY_A: u16 = 30;
const KEY_B: u16 = 48;

fn kbc(props: I8042Props) -> Arc<I8042> {
    I8042::new(Clock::manual(ClockType::Virtual), props).unwrap()
}

/// What the subsection hooks would set for a stream that carried everything `v` needs.
fn as_loaded(mut v: I8042VmState) -> I8042VmState {
    v.outport_present = v.outport_needed();
    v.extended_state_loaded = v.extended_state_needed();
    v
}

fn migrate(src: &I8042, dst: &I8042) {
    dst.kbd_vmstate_load(&src.kbd_vmstate_save());
    dst.mouse_vmstate_load(&src.mouse_vmstate_save());
    dst.vmstate_load(&as_loaded(src.vmstate_save()));
}

fn drain(k: &I8042) -> Vec<u8> {
    let mut v = Vec::new();
    while k.read_status() & KBD_STAT_OBF != 0 {
        v.push(k.read_data());
    }
    v
}

#[test]
fn reset_state_sends_no_subsections() {
    let k = kbc(I8042Props::default());
    let v = k.vmstate_save();
    assert!(!v.outport_needed());
    assert!(v.extended_state_needed());
    assert_eq!(v.migration_flags, 0);
    let kv = k.kbd_vmstate_save();
    assert!(!kv.ledstate_needed());
    assert!(!kv.need_high_bit_needed());
    assert!(!kv.cqueue_needed());
    assert_eq!(kv, Ps2KbdVmState::default());
    assert_eq!(k.mouse_vmstate_save(), Ps2MouseVmState::default());
}

#[test]
fn queued_keys_survive() {
    for extended_state in [true, false] {
        let props = I8042Props { extended_state, ..I8042Props::default() };
        let a = kbc(props);
        let b = kbc(props);
        a.key_event(KEY_A, true);
        a.key_event(KEY_A, false);
        a.key_event(KEY_B, true);
        assert_ne!(a.read_status() & KBD_STAT_OBF, 0);

        migrate(&a, &b);
        assert_eq!(b.regs(), a.regs());
        assert_eq!(b.vmstate_save(), a.vmstate_save());
        assert_eq!(b.kbd_vmstate_save(), a.kbd_vmstate_save());
        let out = drain(&a);
        assert!(!out.is_empty());
        assert_eq!(drain(&b), out);
    }
}

#[test]
fn outport_comes_from_status_when_absent() {
    let k = kbc(I8042Props::default());
    let mut v = k.vmstate_save();
    v.status |= KBD_STAT_OBF | KBD_STAT_MOUSE_OBF;
    v.outport = 0;
    v.outport_present = false;
    k.vmstate_load(&v);
    assert_eq!(
        k.regs().outport,
        KBD_OUT_RESET | KBD_OUT_A20 | KBD_OUT_ONES | KBD_OUT_OBF | KBD_OUT_MOUSE_OBF
    );

    v.outport = KBD_OUT_RESET | KBD_OUT_ONES;
    v.outport_present = true;
    k.vmstate_load(&v);
    assert_eq!(k.regs().outport, KBD_OUT_RESET | KBD_OUT_ONES);
    assert!(k.vmstate_save().outport_needed());
}

#[test]
fn compat_pending_without_extended_state() {
    let k = kbc(I8042Props { extended_state: false, ..I8042Props::default() });
    k.key_event(KEY_A, true);
    let v = k.vmstate_save();
    assert!(!v.extended_state_needed());
    // KBD_PENDING_KBD went out as KBD_PENDING_KBD_COMPAT.
    assert_eq!(v.pending_tmp, 0x01);

    let b = kbc(I8042Props { extended_state: false, ..I8042Props::default() });
    migrate(&k, &b);
    assert_eq!(b.vmstate_save(), v);
}

#[test]
fn ps2_queue_is_bounded_on_load() {
    let k = kbc(I8042Props::default());
    let mut v = k.kbd_vmstate_save();
    v.parent_obj.rptr = 250 + 256;
    v.parent_obj.count = 100;
    v.parent_obj.cwptr = (250 + 20) & 255;
    v.ledstate = 0x104;
    v.need_high_bit = true;
    k.kbd_vmstate_load(&v);
    let got = k.kbd_vmstate_save();
    assert_eq!(got.parent_obj.rptr, 250);
    assert_eq!(got.parent_obj.count, PS2_QUEUE_HEADROOM + PS2_QUEUE_SIZE);
    assert_eq!(got.parent_obj.wptr, (250 + 24) & 255);
    assert_eq!(got.parent_obj.cwptr, (250 + 8) & 255);
    assert_eq!(got.ledstate, 4);
    assert!(got.need_high_bit_needed());
    assert!(got.cqueue_needed());

    let mut m = k.mouse_vmstate_save();
    m.parent_obj.count = -5;
    m.mouse_wrap = 1;
    k.mouse_vmstate_load(&m);
    let got = k.mouse_vmstate_save();
    assert_eq!(got.parent_obj.count, 0);
    assert_eq!(got.parent_obj.cwptr, -1);
    assert_eq!(got.mouse_wrap, 1);
}
