// SPDX-License-Identifier: GPL-2.0-or-later

//! PS/2, i8042, virtio and board input devices.
//!
//! So far this is the i8042 keyboard controller with the PS/2 keyboard and mouse behind it, and
//! the virtio keyboard, mouse, tablet and multitouch devices. The rest of the plan is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod keymap;
pub mod pckbd;
pub mod ps2;
pub mod virtio_input;
