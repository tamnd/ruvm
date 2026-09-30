// SPDX-License-Identifier: GPL-2.0-or-later

//! The e820 memory map that x86 machines hand to the firmware, ported from
//! hw/i386/e820_memory_layout.c.
//!
//! QEMU keeps the table in file scope globals and asserts that nothing is added once the table
//! has been read. Here it is a plain value that a machine owns, with the same rule: after
//! [`E820Table::get_table`] has been called, [`E820Table::add_entry`] panics.
//!
//! The firmware sees the table as the fw_cfg file `etc/e820`, which is the entries back to back
//! in the packed little endian layout of `struct e820_entry` (20 bytes each, no padding).
//! [`E820Table::to_blob`] builds those bytes.

/// Usable RAM.
pub const E820_RAM: u32 = 1;
/// Reserved, not usable by the OS.
pub const E820_RESERVED: u32 = 2;
/// ACPI reclaimable memory.
pub const E820_ACPI: u32 = 3;
/// ACPI NVS memory.
pub const E820_NVS: u32 = 4;
/// Memory with errors in it.
pub const E820_UNUSABLE: u32 = 5;
/// Soft reserved (specific purpose) memory.
pub const E820_SOFT_RESERVED: u32 = 0xefff_ffff;

/// The fw_cfg file name QEMU publishes the table under.
pub const E820_FILE: &str = "etc/e820";

/// Size in bytes of one packed `struct e820_entry`.
pub const E820_ENTRY_SIZE: usize = 20;

/// One e820 entry: a physical range and its type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct E820Entry {
    /// Start of the range.
    pub address: u64,
    /// Length of the range in bytes.
    pub length: u64,
    /// One of the `E820_*` types.
    pub kind: u32,
}

impl E820Entry {
    /// The packed little endian encoding used in `etc/e820`.
    pub fn to_bytes(&self) -> [u8; E820_ENTRY_SIZE] {
        let mut out = [0u8; E820_ENTRY_SIZE];
        out[0..8].copy_from_slice(&self.address.to_le_bytes());
        out[8..16].copy_from_slice(&self.length.to_le_bytes());
        out[16..20].copy_from_slice(&self.kind.to_le_bytes());
        out
    }

    /// Decodes one packed entry.
    pub fn from_bytes(bytes: &[u8; E820_ENTRY_SIZE]) -> Self {
        let mut a = [0u8; 8];
        let mut l = [0u8; 8];
        let mut t = [0u8; 4];
        a.copy_from_slice(&bytes[0..8]);
        l.copy_from_slice(&bytes[8..16]);
        t.copy_from_slice(&bytes[16..20]);
        Self {
            address: u64::from_le_bytes(a),
            length: u64::from_le_bytes(l),
            kind: u32::from_le_bytes(t),
        }
    }
}

/// The e820 table of one machine.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct E820Table {
    entries: Vec<E820Entry>,
    done: bool,
}

impl E820Table {
    /// An empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry. Entries are neither sorted nor merged, exactly as in QEMU.
    ///
    /// # Panics
    ///
    /// Panics if [`E820Table::get_table`] has already been called, which is the
    /// `assert(!e820_done)` in QEMU.
    pub fn add_entry(&mut self, address: u64, length: u64, kind: u32) {
        assert!(!self.done, "e820_add_entry after e820_get_table");
        self.entries.push(E820Entry { address, length, kind });
    }

    /// Returns the entries and marks the table as final, like `e820_get_table`.
    pub fn get_table(&mut self) -> &[E820Entry] {
        self.done = true;
        &self.entries
    }

    /// The entries so far, without marking the table as final.
    pub fn entries(&self) -> &[E820Entry] {
        &self.entries
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// True once [`E820Table::get_table`] has been called.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Returns `(address, length)` of entry `idx` if it exists and has type `kind`, like
    /// `e820_get_entry`.
    pub fn get_entry(&self, idx: usize, kind: u32) -> Option<(u64, u64)> {
        self.entries.get(idx).filter(|e| e.kind == kind).map(|e| (e.address, e.length))
    }

    /// The contents of the `etc/e820` fw_cfg file.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.entries.len() * E820_ENTRY_SIZE);
        for e in &self.entries {
            out.extend_from_slice(&e.to_bytes());
        }
        out
    }
}
