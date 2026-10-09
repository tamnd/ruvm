// SPDX-License-Identifier: GPL-2.0-or-later

//! The colour palette of the tight encoder, from QEMU's ui/vnc-palette.c.
//!
//! A set of up to `max` colours that remembers the order they were first seen in, since that
//! order is the index each colour gets on the wire.

use std::collections::HashMap;

/// `VNC_PALETTE_MAX_SIZE`.
pub const MAX_SIZE: usize = 256;

#[derive(Debug)]
pub struct Palette {
    max: usize,
    colors: Vec<u32>,
    index: HashMap<u32, u8>,
}

impl Palette {
    /// `palette_new()`.
    pub fn new(max: usize) -> Palette {
        Palette { max: max.min(MAX_SIZE), colors: Vec::new(), index: HashMap::new() }
    }

    /// `palette_put()`: adds `color` and returns the palette size, or 0 when it is full and the
    /// colour is new.
    pub fn put(&mut self, color: u32) -> usize {
        if self.index.contains_key(&color) {
            return self.colors.len();
        }
        if self.colors.len() >= self.max {
            return 0;
        }
        self.index.insert(color, self.colors.len() as u8);
        self.colors.push(color);
        self.colors.len()
    }

    /// `palette_idx()`.
    pub fn idx(&self, color: u32) -> Option<u8> {
        self.index.get(&color).copied()
    }

    /// `palette_size()`.
    pub fn size(&self) -> usize {
        self.colors.len()
    }

    /// The colours in index order, what `palette_iter()` visits.
    pub fn colors(&self) -> &[u32] {
        &self.colors
    }
}
