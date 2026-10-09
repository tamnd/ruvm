// SPDX-License-Identifier: GPL-2.0-or-later

//! VGA, cirrus, bochs-display, ramfb, QXL, virtio-gpu and board framebuffers.
//!
//! Ported so far: the standard VGA core with the Bochs VBE extensions ([`vga`]), the PCI "VGA"
//! device ([`vga_pci`]), `bochs-display` ([`bochs_display`]), `ramfb` ([`ramfb`]) and the EDID
//! blob they hand out ([`edid`]). The rest of the plan for this crate is in
//! `spec/24-workspace-layout.md`.

#![forbid(unsafe_code)]

pub mod bochs_display;
pub mod edid;
pub mod ramfb;
pub mod vga;
pub mod vga_pci;
