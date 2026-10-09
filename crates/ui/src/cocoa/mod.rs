// SPDX-License-Identifier: GPL-2.0-or-later

//! `-display cocoa`, QEMU's ui/cocoa.m, over the objc2 crates and the system's AppKit.
//!
//! AppKit wants the main thread. `init` builds the application, the window and the menus there,
//! registers the listener and leaves a function behind for vl.rs to run once the machine is up,
//! as `qemu_main` is in QEMU: the machine's main loop moves to a `qemu_main` thread and the main
//! thread runs the application, which never returns. The listener posts to the main queue with
//! Grand Central Dispatch, as QEMU's does.
//!
//! The keyboard rules of `handleEventLocked` are in [`Keys`], with the other arithmetic the
//! window does, so that they build and are tested on every host. The AppKit side only builds on
//! macOS.
//!
//! Where this differs from QEMU:
//! - There is no clipboard, because ruvm has no clipboard core yet.
//! - There is no `Speed` menu, because ruvm cannot throttle the vCPUs.
//! - `Reset` and `Power Down` are disabled when the machine has nothing behind them, as in the
//!   GTK window. QEMU leaves both enabled.
//! - No display device defines a cursor sprite yet, so there is no cursor layer and
//!   `dpy_mouse_set` has nothing to move.
//! - The refresh rate comes from `-[NSScreen maximumFramesPerSecond]` rather than a CVDisplayLink
//!   of the screen, which is the same number on a fixed rate display.
//! - The window shows a copy of the console's surface, which the listener updates, rather than
//!   the surface itself. A redraw draws the whole copy and leaves the clipping to AppKit, which
//!   puts the same pixels on the screen as QEMU's loop over the dirty rectangles.
//! - The `Removable Media` heading is plain text rather than underlined bold italic Helvetica.
//!   No machine here has removable drives yet, so nothing is listed under it, as in QEMU without
//!   them.
//! - The About panel has no icon and `QEMU Documentation` finds no manual, because ruvm does not
//!   install QEMU's icons and documentation. The item beeps and shows QEMU's alert.
//! - There are no text consoles, so keys typed on a console that is not graphic go nowhere.

#[cfg(target_os = "macos")]
mod appkit;

#[cfg(target_os = "macos")]
pub use appkit::{init, run};

use ruvm_base::report::error_report;
use ruvm_qapi::types::InputButton;

use crate::input::osx_to_linux;
use crate::kbd_state::{KbdOut, KbdState, QKbdModifier};

/// What the window needs from the machine.
pub trait Hooks: Send + Sync {
    /// `qmp_stop()`.
    fn stop(&self);

    /// `qmp_cont()`, whose error the menu drops.
    fn cont(&self);

    /// Whether [`Hooks::reset`] does anything.
    fn can_reset(&self) -> bool;

    /// `qmp_system_reset()`.
    fn reset(&self);

    /// Whether [`Hooks::powerdown`] does anything.
    fn can_powerdown(&self) -> bool;

    /// `qmp_system_powerdown()`.
    fn powerdown(&self);

    /// `qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI)` with the shutdown action forced to
    /// `poweroff`.
    fn quit(&self);
}

/// `NSEventModifierFlagCapsLock`.
pub const FLAG_CAPS_LOCK: u64 = 1 << 16;
/// `NSEventModifierFlagShift`.
pub const FLAG_SHIFT: u64 = 1 << 17;
/// `NSEventModifierFlagControl`.
pub const FLAG_CONTROL: u64 = 1 << 18;
/// `NSEventModifierFlagOption`.
pub const FLAG_OPTION: u64 = 1 << 19;
/// `NSEventModifierFlagCommand`.
pub const FLAG_COMMAND: u64 = 1 << 20;

/// The virtual keycodes of the modifier keys, from Carbon's Events.h.
const KVK_COMMAND: u16 = 0x37;
const KVK_SHIFT: u16 = 0x38;
const KVK_OPTION: u16 = 0x3a;
const KVK_CONTROL: u16 = 0x3b;
const KVK_RIGHT_COMMAND: u16 = 0x36;
const KVK_RIGHT_SHIFT: u16 = 0x3c;
const KVK_RIGHT_OPTION: u16 = 0x3d;
const KVK_RIGHT_CONTROL: u16 = 0x3e;

const KEY_LEFTCTRL: u32 = 29;
const KEY_LEFTSHIFT: u32 = 42;
const KEY_RIGHTSHIFT: u32 = 54;
const KEY_LEFTALT: u32 = 56;
const KEY_CAPSLOCK: u32 = 58;
const KEY_RIGHTCTRL: u32 = 97;
const KEY_RIGHTALT: u32 = 100;
const KEY_LEFTMETA: u32 = 125;
const KEY_RIGHTMETA: u32 = 126;

/// The part of an `NSEvent` that `handleEventLocked` looks at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// `NSEventTypeFlagsChanged`, with `keyCode`.
    FlagsChanged { keycode: u16 },
    /// `NSEventTypeKeyDown`, with `keyCode` and the one UTF-16 unit of
    /// `charactersIgnoringModifiers` when there is exactly one.
    KeyDown { keycode: u16, unit: Option<u16> },
    /// `NSEventTypeKeyUp`, with `keyCode`.
    KeyUp { keycode: u16 },
    /// `NSEventTypeScrollWheel`, with `deltaX` and `deltaY`.
    ScrollWheel { dx: f64, dy: f64 },
    /// Anything else.
    Other,
}

/// What `handleEventLocked` decided.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handled {
    /// The event goes on to AppKit.
    No,
    /// The event was taken.
    Yes,
    /// Taken, and the window switches to the console of this index: `selectConsoleLocked:`.
    SelectConsole(u32),
    /// Taken, and the window lets go of the mouse: `ungrabMouse`.
    Ungrab,
    /// Taken, and the wheel button is pressed and released, each followed by a sync.
    Wheel(InputButton),
}

/// `cocoa_keycode_to_linux()`.
fn keycode_to_linux(keycode: u16) -> u32 {
    match osx_to_linux(u32::from(keycode)) {
        Some(lnx) => lnx,
        None => {
            error_report(&format!("(cocoa) warning unknown keycode 0x{keycode:x}"));
            0
        }
    }
}

/// The keyboard state of the window and the options that change how keys map.
#[derive(Debug)]
pub struct Keys {
    pub kbd: KbdState,
    /// `swap-opt-cmd`.
    pub swap_opt_cmd: bool,
    /// `left-command-key`, on unless the option turns it off.
    pub left_command_key: bool,
}

impl Keys {
    pub fn new(kbd: KbdState) -> Keys {
        Keys { kbd, swap_opt_cmd: false, left_command_key: true }
    }

    /// `toggleKey:`.
    fn toggle(&mut self, lnx: u32, out: &mut Vec<KbdOut>) {
        let down = !self.kbd.key_get(lnx);
        self.kbd.key_event(lnx, down, out);
    }

    /// The alt and meta keys, which `swap-opt-cmd` exchanges.
    fn alt_meta(&self) -> ([u32; 2], [u32; 2]) {
        let alt = [KEY_LEFTALT, KEY_RIGHTALT];
        let meta = [KEY_LEFTMETA, KEY_RIGHTMETA];
        if self.swap_opt_cmd { (meta, alt) } else { (alt, meta) }
    }

    /// `handleEventLocked:` with `modifierFlags`, whether the mouse is grabbed and whether the
    /// console is graphic. The keys to send go to `out`.
    pub fn handle(
        &mut self,
        ev: Event,
        modifiers: u64,
        grabbed: bool,
        graphic: bool,
        out: &mut Vec<KbdOut>,
    ) -> Handled {
        // The modifier state can change while the application is inactive, so every event
        // brings the guest's in line with the flags first.
        if (modifiers & FLAG_CAPS_LOCK != 0) != self.kbd.modifier_get(QKbdModifier::CapsLock) {
            self.kbd.key_event(KEY_CAPSLOCK, true, out);
            self.kbd.key_event(KEY_CAPSLOCK, false, out);
        }
        let (option, command) = self.alt_meta();
        let releases = [
            (FLAG_SHIFT, [KEY_LEFTSHIFT, KEY_RIGHTSHIFT]),
            (FLAG_CONTROL, [KEY_LEFTCTRL, KEY_RIGHTCTRL]),
            (FLAG_OPTION, option),
            (FLAG_COMMAND, command),
        ];
        for (flag, keys) in releases {
            if modifiers & flag == 0 {
                for lnx in keys {
                    self.kbd.key_event(lnx, false, out);
                }
            }
        }

        match ev {
            Event::FlagsChanged { keycode } => {
                let (flag, lnx) = match keycode {
                    KVK_SHIFT => (FLAG_SHIFT, KEY_LEFTSHIFT),
                    KVK_RIGHT_SHIFT => (FLAG_SHIFT, KEY_RIGHTSHIFT),
                    KVK_CONTROL => (FLAG_CONTROL, KEY_LEFTCTRL),
                    KVK_RIGHT_CONTROL => (FLAG_CONTROL, KEY_RIGHTCTRL),
                    KVK_OPTION => (FLAG_OPTION, option[0]),
                    KVK_RIGHT_OPTION => (FLAG_OPTION, option[1]),
                    // The command keys go to the guest only while the mouse is grabbed.
                    KVK_COMMAND if grabbed && self.left_command_key => (FLAG_COMMAND, command[0]),
                    KVK_RIGHT_COMMAND if grabbed => (FLAG_COMMAND, command[1]),
                    _ => return Handled::Yes,
                };
                if modifiers & flag != 0 {
                    self.toggle(lnx, out);
                }
                Handled::Yes
            }
            Event::KeyDown { keycode, unit } => {
                let lnx = keycode_to_linux(keycode);
                // Command combinations are the host's unless the mouse is grabbed.
                if !grabbed && modifiers & FLAG_COMMAND != 0 {
                    return Handled::No;
                }
                // Control, option and 1 to 9 or g are the window's. The character is a C char,
                // so only the low byte of the UTF-16 unit counts.
                if modifiers & FLAG_CONTROL != 0 && modifiers & FLAG_OPTION != 0 {
                    match unit.map(|u| u as u8) {
                        Some(key @ b'1'..=b'9') => {
                            return Handled::SelectConsole(u32::from(key - b'0' - 1));
                        }
                        Some(b'g') => return Handled::Ungrab,
                        _ => {}
                    }
                }
                if graphic {
                    self.kbd.key_event(lnx, true, out);
                }
                Handled::Yes
            }
            Event::KeyUp { keycode } => {
                let lnx = keycode_to_linux(keycode);
                // The host took the key down, so the guest gets no key up either.
                if !grabbed && modifiers & FLAG_COMMAND != 0 {
                    return Handled::Yes;
                }
                if graphic {
                    self.kbd.key_event(lnx, false, out);
                }
                Handled::Yes
            }
            Event::ScrollWheel { dx, dy } => {
                // The wheel goes to the guest whether or not the window has the focus, as is
                // usual on macOS.
                if dy != 0.0 {
                    Handled::Wheel(if dy > 0.0 {
                        InputButton::WheelUp
                    } else {
                        InputButton::WheelDown
                    })
                } else if dx != 0.0 {
                    Handled::Wheel(if dx > 0.0 {
                        InputButton::WheelLeft
                    } else {
                        InputButton::WheelRight
                    })
                } else {
                    Handled::Yes
                }
            }
            Event::Other => Handled::No,
        }
    }
}

/// `fixAspectRatio:`: the largest size within `max` with the shape of the guest's `screen`.
pub fn fix_aspect_ratio(screen: (i32, i32), max: (f64, f64)) -> (f64, f64) {
    let (sw, sh) = (f64::from(screen.0), f64::from(screen.1));
    // Comparing the two scale factors times the screen's area saves a division.
    let scaled = (sw * max.1, sh * max.0);
    if scaled.0 < scaled.1 { (scaled.0 / sh, max.1) } else { (max.0, scaled.1 / sw) }
}

/// The absolute position `handleMouseEvent:` sends for a point in the window, whose origin is
/// bottom left, and a view `frame_height` points high. The C code converts to int, which
/// truncates.
pub fn abs_position(p: (f64, f64), frame_height: f64, screen: (i32, i32)) -> (i32, i32) {
    let d = f64::from(screen.1) / frame_height;
    ((p.0 * d) as i32, (f64::from(screen.1) - p.1 * d) as i32)
}

/// The window's title, with the grab hint while the mouse is grabbed.
pub fn title(name: Option<&str>, grabbed: bool) -> String {
    let hint = if grabbed { " - (Press  \u{2303} \u{2325} G  to release Mouse)" } else { "" };
    match name {
        Some(name) => format!("QEMU {name}{hint}"),
        None => format!("QEMU{hint}"),
    }
}

/// The refresh interval in ms and the refresh rate in mHz that `updateUIInfoLocked` gets from a
/// screen of `fps` frames a second, None when the screen does not say.
pub fn refresh_from_fps(fps: i64) -> Option<(u64, u32)> {
    let fps = u64::try_from(fps).ok().filter(|&f| f > 0)?;
    Some((1000 / fps, u32::try_from(1000 * fps).unwrap_or(u32::MAX)))
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

    fn run(k: &mut Keys, ev: Event, mods: u64, grabbed: bool) -> (Handled, Vec<(u32, bool)>) {
        let mut out = Vec::new();
        let h = k.handle(ev, mods, grabbed, true, &mut out);
        (h, keys(&out))
    }

    #[test]
    fn keymap_matches_qemu() {
        // A sample of input-keymap-osx-to-linux.c.inc of QEMU 11.1.
        assert_eq!((0..256).filter(|&k| osx_to_linux(k) != Some(0)).count(), 120);
        for (osx, lnx) in [
            (0x00, 30),
            (0x0a, 86),
            (0x24, 28),
            (0x31, 57),
            (0x35, 1),
            (0x36, 126),
            (0x37, 125),
            (0x3f, 0x1d0),
            (0x34, 0),
            (0x66, 123),
            (0x7e, 103),
            (0xff, 0),
        ] {
            assert_eq!(osx_to_linux(osx), Some(lnx), "osx 0x{osx:x}");
        }
        assert_eq!(osx_to_linux(0x100), None);
    }

    #[test]
    fn keys_go_down_and_up() {
        let mut k = Keys::new(KbdState::new(None));
        assert_eq!(
            run(&mut k, Event::KeyDown { keycode: 0, unit: Some(0x61) }, 0, false),
            (Handled::Yes, vec![(30, true)])
        );
        assert_eq!(
            run(&mut k, Event::KeyUp { keycode: 0 }, 0, false),
            (Handled::Yes, vec![(30, false)])
        );
    }

    #[test]
    fn unknown_keycodes_send_key_zero() {
        let mut k = Keys::new(KbdState::new(None));
        assert_eq!(
            run(&mut k, Event::KeyDown { keycode: 0x100, unit: None }, 0, false),
            (Handled::Yes, vec![(0, true)])
        );
    }

    #[test]
    fn command_combinations_are_the_hosts_until_grabbed() {
        let mut k = Keys::new(KbdState::new(None));
        let q = Event::KeyDown { keycode: 0x0c, unit: Some(u16::from(b'q')) };
        assert_eq!(run(&mut k, q, FLAG_COMMAND, false), (Handled::No, vec![]));
        assert_eq!(
            run(&mut k, Event::KeyUp { keycode: 0x0c }, FLAG_COMMAND, false),
            (Handled::Yes, vec![])
        );
        assert_eq!(run(&mut k, q, FLAG_COMMAND, true), (Handled::Yes, vec![(16, true)]));
    }

    #[test]
    fn control_option_digits_and_g_are_the_windows() {
        let mut k = Keys::new(KbdState::new(None));
        let mods = FLAG_CONTROL | FLAG_OPTION;
        let two = Event::KeyDown { keycode: 0x13, unit: Some(u16::from(b'2')) };
        assert_eq!(run(&mut k, two, mods, true).0, Handled::SelectConsole(1));
        let g = Event::KeyDown { keycode: 0x05, unit: Some(u16::from(b'g')) };
        assert_eq!(run(&mut k, g, mods, true).0, Handled::Ungrab);
        // QEMU keeps the low byte of the character, so U+0131 is '1'.
        let wide = Event::KeyDown { keycode: 0x22, unit: Some(0x131) };
        assert_eq!(run(&mut k, wide, mods, true).0, Handled::SelectConsole(0));
        let zero = Event::KeyDown { keycode: 0x1d, unit: Some(u16::from(b'0')) };
        assert_eq!(run(&mut k, zero, mods, true), (Handled::Yes, vec![(11, true)]));
    }

    #[test]
    fn flags_changed_toggles_the_modifier() {
        let mut k = Keys::new(KbdState::new(None));
        let shift = Event::FlagsChanged { keycode: KVK_SHIFT };
        assert_eq!(run(&mut k, shift, FLAG_SHIFT, false).1, vec![(KEY_LEFTSHIFT, true)]);
        // Both shift keys down, then the left one comes up and the flag stays.
        let rshift = Event::FlagsChanged { keycode: KVK_RIGHT_SHIFT };
        assert_eq!(run(&mut k, rshift, FLAG_SHIFT, false).1, vec![(KEY_RIGHTSHIFT, true)]);
        assert_eq!(run(&mut k, shift, FLAG_SHIFT, false).1, vec![(KEY_LEFTSHIFT, false)]);
        // The flag going away lifts what is left.
        assert_eq!(run(&mut k, rshift, 0, false).1, vec![(KEY_RIGHTSHIFT, false)]);
    }

    #[test]
    fn command_keys_need_the_grab() {
        let mut k = Keys::new(KbdState::new(None));
        let cmd = Event::FlagsChanged { keycode: KVK_COMMAND };
        assert_eq!(run(&mut k, cmd, FLAG_COMMAND, false).1, vec![]);
        assert_eq!(run(&mut k, cmd, FLAG_COMMAND, true).1, vec![(KEY_LEFTMETA, true)]);
        assert_eq!(run(&mut k, cmd, 0, true).1, vec![(KEY_LEFTMETA, false)]);
        k.left_command_key = false;
        assert_eq!(run(&mut k, cmd, FLAG_COMMAND, true).1, vec![]);
        let rcmd = Event::FlagsChanged { keycode: KVK_RIGHT_COMMAND };
        assert_eq!(run(&mut k, rcmd, FLAG_COMMAND, true).1, vec![(KEY_RIGHTMETA, true)]);
    }

    #[test]
    fn swap_opt_cmd_exchanges_alt_and_meta() {
        let mut k = Keys::new(KbdState::new(None));
        k.swap_opt_cmd = true;
        let opt = Event::FlagsChanged { keycode: KVK_OPTION };
        assert_eq!(run(&mut k, opt, FLAG_OPTION, false).1, vec![(KEY_LEFTMETA, true)]);
        // Without the option flag the meta keys are what gets lifted.
        assert_eq!(run(&mut k, Event::Other, 0, false), (Handled::No, vec![(KEY_LEFTMETA, false)]));
        let ropt = Event::FlagsChanged { keycode: KVK_RIGHT_OPTION };
        assert_eq!(run(&mut k, ropt, FLAG_OPTION, false).1, vec![(KEY_RIGHTMETA, true)]);
        let cmd = Event::FlagsChanged { keycode: KVK_COMMAND };
        assert_eq!(run(&mut k, cmd, FLAG_OPTION | FLAG_COMMAND, true).1, vec![(KEY_LEFTALT, true)]);
    }

    #[test]
    fn caps_lock_follows_the_flag() {
        let mut k = Keys::new(KbdState::new(None));
        let caps = [(KEY_CAPSLOCK, true), (KEY_CAPSLOCK, false)];
        assert_eq!(run(&mut k, Event::Other, FLAG_CAPS_LOCK, false).1, caps);
        assert_eq!(run(&mut k, Event::Other, FLAG_CAPS_LOCK, false).1, vec![]);
        assert_eq!(run(&mut k, Event::Other, 0, false).1, caps);
    }

    #[test]
    fn keys_on_a_text_console_go_nowhere() {
        let mut k = Keys::new(KbdState::new(None));
        let mut out = Vec::new();
        let ev = Event::KeyDown { keycode: 0, unit: Some(0x61) };
        assert_eq!(k.handle(ev, 0, false, false, &mut out), Handled::Yes);
        assert!(out.is_empty());
    }

    #[test]
    fn scroll_wheel_buttons() {
        let mut k = Keys::new(KbdState::new(None));
        let wheel = |k: &mut Keys, dx, dy| run(k, Event::ScrollWheel { dx, dy }, 0, false).0;
        assert_eq!(wheel(&mut k, 0.0, 1.5), Handled::Wheel(InputButton::WheelUp));
        assert_eq!(wheel(&mut k, 3.0, -0.1), Handled::Wheel(InputButton::WheelDown));
        assert_eq!(wheel(&mut k, 2.0, 0.0), Handled::Wheel(InputButton::WheelLeft));
        assert_eq!(wheel(&mut k, -2.0, 0.0), Handled::Wheel(InputButton::WheelRight));
        assert_eq!(wheel(&mut k, 0.0, 0.0), Handled::Yes);
    }

    #[test]
    fn aspect_ratio_fits_inside() {
        assert_eq!(fix_aspect_ratio((640, 480), (1000.0, 1000.0)), (1000.0, 750.0));
        assert_eq!(fix_aspect_ratio((640, 480), (2000.0, 900.0)), (1200.0, 900.0));
    }

    #[test]
    fn absolute_position_truncates() {
        // A 640x480 guest in a view 960 points high, so 0.5 pixels a point.
        assert_eq!(abs_position((101.9, 0.0), 960.0, (640, 480)), (50, 480));
        assert_eq!(abs_position((0.0, 959.0), 960.0, (640, 480)), (0, 0));
    }

    #[test]
    fn titles() {
        assert_eq!(title(None, false), "QEMU");
        assert_eq!(title(Some("vm"), false), "QEMU vm");
        assert_eq!(
            title(Some("vm"), true),
            "QEMU vm - (Press  \u{2303} \u{2325} G  to release Mouse)"
        );
        assert_eq!(title(None, true), "QEMU - (Press  \u{2303} \u{2325} G  to release Mouse)");
    }

    #[test]
    fn refresh_rates() {
        assert_eq!(refresh_from_fps(60), Some((16, 60_000)));
        assert_eq!(refresh_from_fps(120), Some((8, 120_000)));
        assert_eq!(refresh_from_fps(0), None);
    }
}
