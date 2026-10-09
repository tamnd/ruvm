// SPDX-License-Identifier: GPL-2.0-or-later

//! The keycode conversions of QEMU's ui/input-keymap.c.

use ruvm_qapi::types::{KeyValue, KeyValueU, QKeyCode};

use super::tables::{LINUX_TO_QCODE, LINUX_TO_QNUM, QCODE_TO_LINUX, QNUM_TO_LINUX};

/// `KEY_PAUSE`.
const KEY_PAUSE: u32 = 119;

/// `SCANCODE_GREY`: the key needs an `0xe0` prefix.
pub(crate) const SCANCODE_GREY: u32 = 0x80;
/// `SCANCODE_EMUL0`.
pub(crate) const SCANCODE_EMUL0: u32 = 0xe0;
/// `SCANCODE_UP`.
pub(crate) const SCANCODE_UP: u32 = 0x80;

/// `qemu_input_linux_to_qcode()`.
pub fn linux_to_qcode(lnx: u32) -> QKeyCode {
    LINUX_TO_QCODE.get(lnx as usize).map_or(QKeyCode::Unmapped, |&q| QKeyCode::ALL[usize::from(q)])
}

/// `qemu_input_map_qcode_to_linux[]`.
pub fn qcode_to_linux(qcode: QKeyCode) -> u32 {
    u32::from(QCODE_TO_LINUX[qcode as usize])
}

/// `qemu_input_key_number_to_linux()`. The number is a QEMU key number, an XT scancode with
/// `0x80` for the keys that take an `0xe0` prefix.
pub fn key_number_to_linux(nr: i64) -> u32 {
    // The C code takes the number as unsigned, so a negative one is out of range too.
    usize::try_from(nr).ok().and_then(|i| QNUM_TO_LINUX.get(i)).map_or(0, |&l| u32::from(l))
}

/// `qemu_input_key_number_to_qcode()`.
pub fn key_number_to_qcode(nr: i64) -> QKeyCode {
    linux_to_qcode(key_number_to_linux(nr))
}

/// `qemu_input_key_value_to_linux()`.
pub fn key_value_to_linux(value: &KeyValue) -> u32 {
    match &value.u {
        KeyValueU::Number(n) => key_number_to_linux(n.data),
        KeyValueU::Qcode(q) => qcode_to_linux(q.data),
    }
}

/// `qemu_input_linux_to_scancode()`: the XT scancode bytes of a key going down or up.
pub fn linux_to_scancode(lnx: u32, down: bool) -> Vec<u32> {
    let mut keycode = LINUX_TO_QNUM.get(lnx as usize).map_or(0, |&k| u32::from(k));
    if lnx == KEY_PAUSE {
        let v = if down { 0 } else { 0x80 };
        return vec![0xe1, 0x1d | v, 0x45 | v];
    }
    let mut codes = Vec::with_capacity(2);
    if (keycode & SCANCODE_GREY) != 0 {
        codes.push(SCANCODE_EMUL0);
        keycode &= !SCANCODE_GREY;
    }
    if !down {
        keycode |= SCANCODE_UP;
    }
    codes.push(keycode);
    codes
}
