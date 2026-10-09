// SPDX-License-Identifier: GPL-2.0-or-later

//! Keyboard layouts, QEMU's ui/keymaps.c: the `pc-bios/keymaps` files that map X keysyms to
//! QEMU key numbers, for a VNC client that sends keysyms.
//!
//! A layout is looked up as `keymaps/<language>` in the data directories, after the name as a
//! path of its own, like `qemu_find_file()` does. Where this differs from QEMU: when no
//! `en-us` file is found the copy of it built into ruvm is used, so a VNC server works without
//! QEMU's data files.

mod keysym;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use ruvm_base::report::warn_report;
use ruvm_base::{Error, Result};

use crate::input::key_number_to_linux;
use crate::kbd_state::{KbdState, QKbdModifier};

/// `SCANCODE_KEYMASK`: the scancode without the modifier bits.
pub const SCANCODE_KEYMASK: u32 = 0xff;
/// `SCANCODE_SHIFT` and the other modifier bits a layout entry can carry.
pub const SCANCODE_SHIFT: u32 = 0x100;
pub const SCANCODE_CTRL: u32 = 0x200;
pub const SCANCODE_ALT: u32 = 0x400;
pub const SCANCODE_ALTGR: u32 = 0x800;

const XK_ISO_LEFT_TAB: u32 = 0xfe20;
const XK_TAB: u32 = 0xff09;

/// The `en-us` layout of QEMU's pc-bios/keymaps.
const EN_US: &str = include_str!("en-us");

/// `keyboard_layout`, from `-k`.
static KEYBOARD_LAYOUT: Mutex<Option<String>> = Mutex::new(None);
/// The data directories `qemu_find_file()` searches.
static DATA_DIRS: Mutex<Vec<PathBuf>> = Mutex::new(Vec::new());

static NAMES: LazyLock<HashMap<&'static str, u32>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for &(name, sym) in keysym::NAME2KEYSYM {
        m.entry(name).or_insert(sym);
    }
    m
});

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Sets `keyboard_layout`, the `-k` option.
pub fn set_keyboard_layout(layout: Option<String>) {
    *lock(&KEYBOARD_LAYOUT) = layout;
}

/// The `-k` layout, if one was given.
pub fn keyboard_layout() -> Option<String> {
    lock(&KEYBOARD_LAYOUT).clone()
}

/// Sets the data directories a layout is looked for in, in search order.
pub fn set_data_dirs(dirs: Vec<PathBuf>) {
    *lock(&DATA_DIRS) = dirs;
}

/// `get_keysym()`: a keysym by name, or a `Uxxxx` Unicode name. Zero when there is none.
fn get_keysym(name: &str) -> u32 {
    if let Some(&sym) = NAMES.get(name) {
        return sym;
    }
    if let Some(hex) = name.strip_prefix('U').filter(|h| h.len() == 4) {
        if let Ok(v) = u32::from_str_radix(hex, 16) {
            return v;
        }
    }
    0
}

/// `strtol(s, NULL, 0)`, which stops at the first character that does not fit.
fn strtol(s: &str) -> i64 {
    let s = s.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']);
    let (neg, s) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let (radix, digits) = if let Some(h) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        if h.chars().next().is_some_and(|c| c.is_ascii_hexdigit()) { (16, h) } else { (8, s) }
    } else if s.starts_with('0') {
        (8, s)
    } else {
        (10, s)
    };
    let mut v: i64 = 0;
    for c in digits.chars() {
        let Some(d) = c.to_digit(radix) else { break };
        v = v.saturating_mul(i64::from(radix)).saturating_add(i64::from(d));
    }
    if neg { -v } else { v }
}

/// `kbd_layout_t`: each keysym with up to four key numbers, each with its modifier bits.
#[derive(Debug, Default)]
pub struct KbdLayout {
    map: HashMap<u32, Vec<u32>>,
}

impl KbdLayout {
    /// `kbd_layout_new()`: reads the layout file of `language`.
    pub fn new(language: &str) -> Result<KbdLayout> {
        let dirs = lock(&DATA_DIRS).clone();
        let found = find_file(&dirs, language);
        let text = match found {
            Some(path) => std::fs::read(&path)
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .map_err(|e| Error::generic(format!("Could not open '{}': {e}", path.display())))?,
            None if language == "en-us" => EN_US.to_string(),
            None => {
                return Err(Error::generic(format!(
                    "could not find keymap file for language '{language}'"
                )));
            }
        };
        Self::parse(&text)
    }

    /// `parse_keyboard_layout()` on the text of a layout file.
    pub fn parse(text: &str) -> Result<KbdLayout> {
        let mut k = KbdLayout::default();
        for line in text.split_inclusive('\n') {
            let line = line.strip_suffix('\n').unwrap_or(line);
            if line.starts_with('#') || line.starts_with("map ") {
                continue;
            }
            if line.starts_with("include ") {
                return Err(Error::generic("keymap include files are not supported any more"));
            }
            let mut end = line.find(' ').unwrap_or(line.len()).min(63);
            while !line.is_char_boundary(end) {
                end -= 1;
            }
            let mut keyname = line[..end].to_string();
            if keyname.is_empty() {
                continue;
            }
            let keysym = get_keysym(&keyname);
            if keysym == 0 {
                continue;
            }
            let rest = line.get(end + 1..).unwrap_or("");
            let mut keycode = strtol(rest) as i32 as u32;
            if rest.contains("shift") {
                keycode |= SCANCODE_SHIFT;
            }
            if rest.contains("altgr") {
                keycode |= SCANCODE_ALTGR;
            }
            if rest.contains("ctrl") {
                keycode |= SCANCODE_CTRL;
            }
            k.add_keysym(keysym, keycode);
            if rest.contains("addupper") {
                keyname.make_ascii_uppercase();
                let upper = get_keysym(&keyname);
                if upper != 0 {
                    k.add_keysym(upper, keycode | SCANCODE_SHIFT);
                }
            }
        }
        Ok(k)
    }

    /// `add_keysym()`.
    fn add_keysym(&mut self, keysym: u32, keycode: u32) {
        let codes = self.map.entry(keysym).or_default();
        if codes.len() < 4 {
            // The C code keeps a uint16_t.
            codes.push(keycode & 0xffff);
        } else {
            warn_report(&format!("more than 4 keycodes for keysym {keysym}"));
        }
    }

    /// `keysym2scancode()`: the key number with its modifier bits for `keysym`, zero for none.
    /// With more than one mapping, a key going down takes the one whose modifiers match those
    /// held now, and a key going up the one that is down.
    pub fn keysym2scancode(&self, keysym: u32, kbd: Option<&KbdState>, down: bool) -> u32 {
        const MASK: u32 = SCANCODE_SHIFT | SCANCODE_ALTGR | SCANCODE_CTRL;
        let keysym = if keysym == XK_ISO_LEFT_TAB { XK_TAB } else { keysym };
        let Some(codes) = self.map.get(&keysym) else {
            warn_report(&format!("no scancode found for keysym {keysym}"));
            return 0;
        };
        if codes.len() == 1 {
            return codes[0];
        }
        let held = |m| kbd.is_some_and(|k| k.modifier_get(m));
        if down {
            let mut mods = 0;
            if held(QKbdModifier::Shift) {
                mods |= SCANCODE_SHIFT;
            }
            if held(QKbdModifier::AltGr) {
                mods |= SCANCODE_ALTGR;
            }
            if held(QKbdModifier::Ctrl) {
                mods |= SCANCODE_CTRL;
            }
            if let Some(&c) = codes.iter().find(|&&c| (c & MASK) == mods) {
                return c;
            }
        } else if let Some(&c) = codes
            .iter()
            .find(|&&c| kbd.is_some_and(|k| k.key_get(key_number_to_linux(i64::from(c)))))
        {
            return c;
        }
        codes[0]
    }
}

/// `qemu_find_file(QEMU_FILE_TYPE_KEYMAP, name)`.
fn find_file(dirs: &[PathBuf], name: &str) -> Option<PathBuf> {
    let direct = Path::new(name);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    dirs.iter().map(|d| d.join("keymaps").join(name)).find(|p| p.is_file())
}

/// `keycode_is_keypad()`.
pub fn keycode_is_keypad(keycode: u32) -> bool {
    (0x47..=0x53).contains(&keycode)
}

/// `keysym_is_numlock()`: the keypad digits, separator and decimal point.
pub fn keysym_is_numlock(keysym: u32) -> bool {
    matches!(keysym, 0xffb0..=0xffb9 | 0xffac | 0xffae)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kbd_state::KbdOut;

    #[test]
    fn en_us() {
        let k = KbdLayout::parse(EN_US).unwrap();
        // "a 0x1e" and "A 0x1e shift".
        assert_eq!(k.keysym2scancode(u32::from(b'a'), None, true), 0x1e);
        assert_eq!(k.keysym2scancode(u32::from(b'A'), None, true), 0x1e | SCANCODE_SHIFT);
        assert_eq!(k.keysym2scancode(XK_ISO_LEFT_TAB, None, true), 0x0f);
        // Return, the grey keypad Enter and an unknown keysym.
        assert_eq!(k.keysym2scancode(0xff0d, None, true), 0x1c);
        assert_eq!(k.keysym2scancode(0xff8d, None, true), 0x9c);
        assert_eq!(k.keysym2scancode(0x12345, None, true), 0);
    }

    #[test]
    fn several_mappings_follow_the_modifiers() {
        let k = KbdLayout::parse("plus 0x0d shift\nplus 0x4e\n").unwrap();
        let mut kbd = KbdState::new(None);
        let mut out: Vec<KbdOut> = Vec::new();
        assert_eq!(k.keysym2scancode(0x2b, Some(&kbd), true), 0x4e);
        kbd.key_event(42, true, &mut out);
        assert_eq!(k.keysym2scancode(0x2b, Some(&kbd), true), 0x0d | SCANCODE_SHIFT);
        // On release the one that is down wins.
        kbd.key_event(78, true, &mut out);
        assert_eq!(k.keysym2scancode(0x2b, Some(&kbd), false), 0x4e);
    }

    #[test]
    fn parse_errors_and_numbers() {
        let e = KbdLayout::parse("include common\n").unwrap_err();
        assert_eq!(e.to_string(), "keymap include files are not supported any more");
        assert_eq!(strtol(" 0x1e"), 0x1e);
        assert_eq!(strtol("010"), 8);
        assert_eq!(strtol("57 shift"), 57);
        assert_eq!(strtol("junk"), 0);
        assert_eq!(get_keysym("U20ac"), 0x20ac);
        assert_eq!(get_keysym("space"), 0x20);
        assert_eq!(get_keysym("nosuchkey"), 0);
    }
}
