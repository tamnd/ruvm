// SPDX-License-Identifier: GPL-2.0-or-later

//! Consoles, VNC, SPICE, GTK, SDL, Cocoa, the D-Bus display and keymaps.
//!
//! So far this holds the console core of QEMU's `ui/`: pixman style pixel formats and images
//! ([`pixman`]), display surfaces ([`surface`]), consoles with their device and listener hooks
//! ([`console`]), the VGA font ([`vgafont`]) and the QMP `screendump` command ([`screendump`]).
//! The remote and local front ends come later. The plan for this crate is in
//! `spec/24-workspace-layout.md`.

pub mod console;
pub mod pixman;
pub mod screendump;
pub mod surface;
pub mod vgafont;
