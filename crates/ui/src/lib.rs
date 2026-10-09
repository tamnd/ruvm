// SPDX-License-Identifier: GPL-2.0-or-later

//! Consoles, VNC, SPICE, GTK, SDL, Cocoa, the D-Bus display and keymaps.
//!
//! So far this holds the console core of QEMU's `ui/`: pixman style pixel formats and images
//! ([`pixman`]), display surfaces ([`surface`]), consoles with their device and listener hooks
//! ([`console`]), the VGA font ([`vgafont`]), the QMP `screendump` command ([`screendump`]) and
//! the VNC server ([`vnc`]). The input layer of `ui/input.c` is in [`input`], with the keyboard
//! state of `ui/kbd-state.c` in [`kbd_state`] and the keysym layouts of `ui/keymaps.c` in
//! [`keymaps`]. With the `ui-sdl` feature, `sdl` is the window of `-display sdl` over the
//! system's SDL2 library, and with `ui-gtk`, `gtk` is the window of `-display gtk` over GTK 4.
//! With `ui-dbus`, `dbus` exports the consoles on D-Bus for `-display dbus`. With `ui-cocoa`,
//! `cocoa` is the window of `-display cocoa`, whose AppKit half builds only for macOS. The other
//! front ends come later. The plan for this crate is in `spec/24-workspace-layout.md`.

#[cfg(feature = "ui-cocoa")]
pub mod cocoa;
pub mod console;
#[cfg(all(feature = "ui-dbus", unix))]
pub mod dbus;
#[cfg(feature = "ui-gtk")]
pub mod gtk;
pub mod input;
pub mod kbd_state;
pub mod keymaps;
pub mod pixman;
pub mod screendump;
#[cfg(feature = "ui-sdl")]
pub mod sdl;
pub mod surface;
pub mod vgafont;
pub mod vnc;
