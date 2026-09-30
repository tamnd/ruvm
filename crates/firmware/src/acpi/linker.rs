// SPDX-License-Identifier: GPL-2.0-or-later

//! The BIOS linker and loader script from hw/acpi/bios-linker-loader.c.
//!
//! QEMU does not place ACPI tables in guest memory itself. It hands the firmware a few fw_cfg
//! files plus `etc/table-loader`, a list of fixed size commands that say which files to allocate,
//! where pointers between them go and which checksums to compute. SeaBIOS, OVMF and the guest
//! side of every other firmware QEMU boots run that script, so the command layout below is an ABI.

use std::fmt;

/// `FW_CFG_MAX_FILE_PATH`, the size of every file name field in a command.
pub const FILESZ: usize = 56;

/// Every command is padded to this many bytes.
pub const ENTRY_SIZE: usize = 128;

const COMMAND_ALLOCATE: u32 = 0x1;
const COMMAND_ADD_POINTER: u32 = 0x2;
const COMMAND_ADD_CHECKSUM: u32 = 0x3;
const COMMAND_WRITE_POINTER: u32 = 0x4;

const ALLOC_ZONE_HIGH: u8 = 0x1;
const ALLOC_ZONE_FSEG: u8 = 0x2;

/// One decoded loader command, handy for tests and for a firmware that runs the script itself.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    /// Load `file` into memory aligned to `align`, in the F segment when `fseg` is set.
    Allocate { file: String, align: u32, fseg: bool },
    /// Add the address of `src_file` to the `size` byte little endian value at `offset` in
    /// `dest_file`. The value starts out as the offset within `src_file`.
    AddPointer { dest_file: String, src_file: String, offset: u32, size: u8 },
    /// Store the checksum of `length` bytes from `start` in `file` at `offset`.
    AddChecksum { file: String, offset: u32, start: u32, length: u32 },
    /// Write the address of `src_file` plus `src_offset` back to QEMU through `dest_file`.
    WritePointer { dest_file: String, src_file: String, dst_offset: u32, src_offset: u32, size: u8 },
}

/// `BIOSLinker`. The file blobs themselves stay with the caller. The linker only remembers
/// the names it has been told about and how long each blob was at the time.
#[derive(Default)]
pub struct BiosLinker {
    cmd_blob: Vec<u8>,
    files: Vec<String>,
}

impl fmt::Debug for BiosLinker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BiosLinker")
            .field("files", &self.files)
            .field("commands", &self.commands())
            .finish()
    }
}

fn put_name(entry: &mut [u8], at: usize, name: &str) {
    // strncpy() into a zeroed field, keeping room for the terminating NUL.
    let n = name.len().min(FILESZ - 1);
    entry[at..at + n].copy_from_slice(&name.as_bytes()[..n]);
}

fn get_name(entry: &[u8], at: usize) -> String {
    let field = &entry[at..at + FILESZ];
    let end = field.iter().position(|&b| b == 0).unwrap_or(FILESZ);
    String::from_utf8_lossy(&field[..end]).into_owned()
}

fn put_u32(entry: &mut [u8], at: usize, v: u32) {
    entry[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(entry: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(entry[at..at + 4].try_into().expect("4 bytes"))
}

impl BiosLinker {
    /// `bios_linker_loader_init()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// The `etc/table-loader` contents.
    pub fn cmd_blob(&self) -> &[u8] {
        &self.cmd_blob
    }

    fn has_file(&self, name: &str) -> bool {
        self.files.iter().any(|f| f == name)
    }

    /// `bios_linker_loader_alloc()`. Allocation commands go to the front of the script so the
    /// firmware has every file in memory before it patches anything.
    pub fn alloc(&mut self, file: &str, align: u32, fseg: bool) {
        assert!(align.is_power_of_two(), "alignment {align} is not a power of two");
        assert!(!self.has_file(file), "{file} allocated twice");
        self.files.push(file.to_string());
        let mut entry = [0u8; ENTRY_SIZE];
        put_u32(&mut entry, 0, COMMAND_ALLOCATE);
        put_name(&mut entry, 4, file);
        put_u32(&mut entry, 4 + FILESZ, align);
        entry[8 + FILESZ] = if fseg { ALLOC_ZONE_FSEG } else { ALLOC_ZONE_HIGH };
        self.cmd_blob.splice(0..0, entry);
    }

    /// `bios_linker_loader_add_checksum()`.
    pub fn add_checksum(&mut self, file: &str, start: u32, size: u32, checksum_offset: u32) {
        assert!(self.has_file(file), "{file} was never allocated");
        assert!(checksum_offset >= start && checksum_offset < start + size);
        let mut entry = [0u8; ENTRY_SIZE];
        put_u32(&mut entry, 0, COMMAND_ADD_CHECKSUM);
        put_name(&mut entry, 4, file);
        put_u32(&mut entry, 4 + FILESZ, checksum_offset);
        put_u32(&mut entry, 8 + FILESZ, start);
        put_u32(&mut entry, 12 + FILESZ, size);
        self.cmd_blob.extend_from_slice(&entry);
    }

    /// `bios_linker_loader_add_pointer()`. `dest` is the blob of `dest_file`, which gets the
    /// source offset written into the patched field the same way QEMU does.
    #[allow(clippy::too_many_arguments)]
    pub fn add_pointer(
        &mut self,
        dest_file: &str,
        dest: &mut [u8],
        dst_offset: u32,
        size: u8,
        src_file: &str,
        src_offset: u32,
    ) {
        assert!(self.has_file(dest_file), "{dest_file} was never allocated");
        assert!(self.has_file(src_file), "{src_file} was never allocated");
        assert!(matches!(size, 1 | 2 | 4 | 8), "pointer size {size}");
        let at = dst_offset as usize;
        dest[at..at + usize::from(size)]
            .copy_from_slice(&u64::from(src_offset).to_le_bytes()[..usize::from(size)]);
        let mut entry = [0u8; ENTRY_SIZE];
        put_u32(&mut entry, 0, COMMAND_ADD_POINTER);
        put_name(&mut entry, 4, dest_file);
        put_name(&mut entry, 4 + FILESZ, src_file);
        put_u32(&mut entry, 4 + 2 * FILESZ, dst_offset);
        entry[8 + 2 * FILESZ] = size;
        self.cmd_blob.extend_from_slice(&entry);
    }

    /// `bios_linker_loader_write_pointer()`.
    pub fn write_pointer(
        &mut self,
        dest_file: &str,
        dst_offset: u32,
        size: u8,
        src_file: &str,
        src_offset: u32,
    ) {
        assert!(self.has_file(src_file), "{src_file} was never allocated");
        assert!(matches!(size, 1 | 2 | 4 | 8), "pointer size {size}");
        let mut entry = [0u8; ENTRY_SIZE];
        put_u32(&mut entry, 0, COMMAND_WRITE_POINTER);
        put_name(&mut entry, 4, dest_file);
        put_name(&mut entry, 4 + FILESZ, src_file);
        put_u32(&mut entry, 4 + 2 * FILESZ, dst_offset);
        put_u32(&mut entry, 8 + 2 * FILESZ, src_offset);
        entry[12 + 2 * FILESZ] = size;
        self.cmd_blob.extend_from_slice(&entry);
    }

    /// Decodes the script.
    pub fn commands(&self) -> Vec<Command> {
        self.cmd_blob
            .chunks(ENTRY_SIZE)
            .map(|e| match get_u32(e, 0) {
                COMMAND_ALLOCATE => Command::Allocate {
                    file: get_name(e, 4),
                    align: get_u32(e, 4 + FILESZ),
                    fseg: e[8 + FILESZ] == ALLOC_ZONE_FSEG,
                },
                COMMAND_ADD_POINTER => Command::AddPointer {
                    dest_file: get_name(e, 4),
                    src_file: get_name(e, 4 + FILESZ),
                    offset: get_u32(e, 4 + 2 * FILESZ),
                    size: e[8 + 2 * FILESZ],
                },
                COMMAND_ADD_CHECKSUM => Command::AddChecksum {
                    file: get_name(e, 4),
                    offset: get_u32(e, 4 + FILESZ),
                    start: get_u32(e, 8 + FILESZ),
                    length: get_u32(e, 12 + FILESZ),
                },
                COMMAND_WRITE_POINTER => Command::WritePointer {
                    dest_file: get_name(e, 4),
                    src_file: get_name(e, 4 + FILESZ),
                    dst_offset: get_u32(e, 4 + 2 * FILESZ),
                    src_offset: get_u32(e, 8 + 2 * FILESZ),
                    size: e[12 + 2 * FILESZ],
                },
                other => panic!("unknown loader command {other}"),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocations_go_first() {
        let mut l = BiosLinker::new();
        l.alloc("etc/acpi/tables", 64, false);
        l.add_checksum("etc/acpi/tables", 0, 36, 9);
        l.alloc("etc/acpi/rsdp", 16, true);
        let mut rsdp = vec![0u8; 20];
        l.add_pointer("etc/acpi/rsdp", &mut rsdp, 16, 4, "etc/acpi/tables", 0x40);
        assert_eq!(&rsdp[16..20], [0x40, 0, 0, 0]);
        assert_eq!(l.cmd_blob().len(), 4 * ENTRY_SIZE);
        assert_eq!(
            l.commands(),
            [
                Command::Allocate { file: "etc/acpi/rsdp".into(), align: 16, fseg: true },
                Command::Allocate { file: "etc/acpi/tables".into(), align: 64, fseg: false },
                Command::AddChecksum {
                    file: "etc/acpi/tables".into(),
                    offset: 9,
                    start: 0,
                    length: 36
                },
                Command::AddPointer {
                    dest_file: "etc/acpi/rsdp".into(),
                    src_file: "etc/acpi/tables".into(),
                    offset: 16,
                    size: 4
                },
            ]
        );
    }

    #[test]
    fn entry_layout_matches_the_packed_c_struct() {
        let mut l = BiosLinker::new();
        l.alloc("etc/acpi/tables", 64, false);
        let b = l.cmd_blob();
        assert_eq!(&b[..4], [1, 0, 0, 0]);
        assert_eq!(&b[4..19], b"etc/acpi/tables");
        assert_eq!(&b[60..64], [64, 0, 0, 0]);
        assert_eq!(b[64], ALLOC_ZONE_HIGH);
    }
}
