// SPDX-License-Identifier: GPL-2.0-or-later

//! The keyboard state of a front end, QEMU's ui/kbd-state.c: which keys are down and the
//! modifier and lock state they add up to.
//!
//! Unlike `QKbdState` it does not send the keys itself. A front end updates it under its own
//! lock and gets the keys to send back as [`KbdOut`] items, which [`send`] hands to the input
//! layer once that lock is dropped.

use crate::console::QemuConsole;
use crate::input::InputState;

/// `KEY_CNT`.
const KEY_CNT: u32 = 0x300;

const KEY_LEFTCTRL: u32 = 29;
const KEY_LEFTSHIFT: u32 = 42;
const KEY_RIGHTSHIFT: u32 = 54;
const KEY_LEFTALT: u32 = 56;
const KEY_CAPSLOCK: u32 = 58;
const KEY_NUMLOCK: u32 = 69;
const KEY_RIGHTCTRL: u32 = 97;
const KEY_RIGHTALT: u32 = 100;

/// `QKbdModifier`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QKbdModifier {
    Shift = 1,
    Ctrl,
    Alt,
    AltGr,
    NumLock,
    CapsLock,
}

/// A key the state wants sent to the guest, in order.
#[derive(Clone, Debug)]
pub enum KbdOut {
    /// `qemu_input_event_send_key_linux()`.
    Key { con: Option<QemuConsole>, lnx: u32, down: bool },
    /// `qemu_input_event_send_key_delay()`.
    Delay(u32),
}

/// `QKbdState`.
#[derive(Debug)]
pub struct KbdState {
    con: Option<QemuConsole>,
    key_delay_ms: u32,
    keys: [u64; (KEY_CNT / 64) as usize],
    mods: u8,
}

impl KbdState {
    /// `qkbd_state_init()`.
    pub fn new(con: Option<QemuConsole>) -> KbdState {
        KbdState { con, key_delay_ms: 0, keys: [0; (KEY_CNT / 64) as usize], mods: 0 }
    }

    /// `qkbd_state_set_delay()`.
    pub fn set_delay(&mut self, delay_ms: u32) {
        self.key_delay_ms = delay_ms;
    }

    /// `qkbd_state_key_get()`.
    pub fn key_get(&self, lnx: u32) -> bool {
        lnx < KEY_CNT && (self.keys[(lnx / 64) as usize] >> (lnx % 64)) & 1 != 0
    }

    fn set_key(&mut self, lnx: u32, down: bool) {
        let w = &mut self.keys[(lnx / 64) as usize];
        if down {
            *w |= 1 << (lnx % 64);
        } else {
            *w &= !(1 << (lnx % 64));
        }
    }

    /// `qkbd_state_modifier_get()`.
    pub fn modifier_get(&self, m: QKbdModifier) -> bool {
        (self.mods >> m as u8) & 1 != 0
    }

    fn set_modifier(&mut self, m: QKbdModifier, on: bool) {
        if on {
            self.mods |= 1 << m as u8;
        } else {
            self.mods &= !(1 << m as u8);
        }
    }

    /// `qkbd_state_modifier_update()`.
    fn modifier_update(&mut self, lnx1: u32, lnx2: u32, m: QKbdModifier) {
        let on = self.key_get(lnx1) || self.key_get(lnx2);
        self.set_modifier(m, on);
    }

    /// `qkbd_state_key_event()`. A key going up that is not down is dropped, so a front end can
    /// pass on every release, including those of keys it kept for itself.
    pub fn key_event(&mut self, lnx: u32, down: bool, out: &mut Vec<KbdOut>) {
        if lnx >= KEY_CNT {
            return;
        }
        if !down && !self.key_get(lnx) {
            return;
        }
        self.set_key(lnx, down);
        match lnx {
            KEY_LEFTSHIFT | KEY_RIGHTSHIFT => {
                self.modifier_update(KEY_LEFTSHIFT, KEY_RIGHTSHIFT, QKbdModifier::Shift)
            }
            KEY_LEFTCTRL | KEY_RIGHTCTRL => {
                self.modifier_update(KEY_LEFTCTRL, KEY_RIGHTCTRL, QKbdModifier::Ctrl)
            }
            KEY_LEFTALT => self.modifier_update(KEY_LEFTALT, KEY_LEFTALT, QKbdModifier::Alt),
            KEY_RIGHTALT => self.modifier_update(KEY_RIGHTALT, KEY_RIGHTALT, QKbdModifier::AltGr),
            KEY_CAPSLOCK if down => {
                let on = !self.modifier_get(QKbdModifier::CapsLock);
                self.set_modifier(QKbdModifier::CapsLock, on);
            }
            KEY_NUMLOCK if down => {
                let on = !self.modifier_get(QKbdModifier::NumLock);
                self.set_modifier(QKbdModifier::NumLock, on);
            }
            _ => {}
        }
        // Every console is graphic, so the key always goes to the guest.
        out.push(KbdOut::Key { con: self.con.clone(), lnx, down });
        if self.key_delay_ms != 0 {
            out.push(KbdOut::Delay(self.key_delay_ms));
        }
    }

    /// `qkbd_state_lift_all_keys()`.
    pub fn lift_all_keys(&mut self, out: &mut Vec<KbdOut>) {
        for lnx in 0..KEY_CNT {
            if self.key_get(lnx) {
                self.key_event(lnx, false, out);
            }
        }
    }

    /// `qkbd_state_switch_console()`.
    pub fn switch_console(&mut self, con: Option<QemuConsole>, out: &mut Vec<KbdOut>) {
        self.lift_all_keys(out);
        self.con = con;
    }
}

/// Sends what [`KbdState`] queued.
pub fn send(input: &InputState, out: Vec<KbdOut>) {
    for o in out {
        match o {
            KbdOut::Key { con, lnx, down } => input.send_key_linux(con.as_ref(), lnx, down),
            KbdOut::Delay(ms) => input.send_key_delay(ms),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(out: &[KbdOut]) -> Vec<(u32, bool)> {
        out.iter()
            .filter_map(|o| match o {
                KbdOut::Key { lnx, down, .. } => Some((*lnx, *down)),
                KbdOut::Delay(_) => None,
            })
            .collect()
    }

    #[test]
    fn stray_releases_are_dropped() {
        let mut k = KbdState::new(None);
        let mut out = Vec::new();
        k.key_event(30, false, &mut out);
        assert!(out.is_empty());
        k.key_event(30, true, &mut out);
        k.key_event(30, true, &mut out);
        k.key_event(30, false, &mut out);
        assert_eq!(keys(&out), [(30, true), (30, true), (30, false)]);
    }

    #[test]
    fn modifiers_and_locks() {
        let mut k = KbdState::new(None);
        let mut out = Vec::new();
        k.key_event(KEY_RIGHTSHIFT, true, &mut out);
        assert!(k.modifier_get(QKbdModifier::Shift));
        k.key_event(KEY_LEFTSHIFT, true, &mut out);
        k.key_event(KEY_RIGHTSHIFT, false, &mut out);
        assert!(k.modifier_get(QKbdModifier::Shift));
        k.key_event(KEY_LEFTSHIFT, false, &mut out);
        assert!(!k.modifier_get(QKbdModifier::Shift));
        k.key_event(KEY_RIGHTALT, true, &mut out);
        assert!(k.modifier_get(QKbdModifier::AltGr) && !k.modifier_get(QKbdModifier::Alt));
        k.key_event(KEY_CAPSLOCK, true, &mut out);
        k.key_event(KEY_CAPSLOCK, false, &mut out);
        assert!(k.modifier_get(QKbdModifier::CapsLock));
        k.key_event(KEY_CAPSLOCK, true, &mut out);
        assert!(!k.modifier_get(QKbdModifier::CapsLock));
    }

    #[test]
    fn lift_all_and_delay() {
        let mut k = KbdState::new(None);
        k.set_delay(10);
        let mut out = Vec::new();
        k.key_event(30, true, &mut out);
        k.key_event(2, true, &mut out);
        out.clear();
        k.lift_all_keys(&mut out);
        assert_eq!(keys(&out), [(2, false), (30, false)]);
        assert!(matches!(out[1], KbdOut::Delay(10)));
    }
}
