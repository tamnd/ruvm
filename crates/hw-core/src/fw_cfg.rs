// SPDX-License-Identifier: GPL-2.0-or-later

//! The firmware configuration device, hw/nvram/fw_cfg.c, with the definitions from
//! include/hw/nvram/fw_cfg.h and include/standard-headers/linux/qemu_fw_cfg.h.
//!
//! [`FwCfgState`] holds the entry table, the file directory and the guest visible state
//! (current entry, offset and DMA address). Guests reach it through one of two flavors:
//! [`FwCfgIo`], the combined selector and data port plus the DMA port used on x86, and
//! [`FwCfgMem`], separate selector, data and DMA MMIO regions used on Arm, RISC-V and others.
//! Each region is an [`MmioOps`] implementation mirroring the C `MemoryRegionOps`.
//!
//! DMA goes through the small [`DmaMemory`] trait, which [`AddressSpace`] implements.
//!
//! Select and write callbacks run with the device lock held, so they must not call back into
//! the same device.
//!
//! VMState, trace points, QOM registration, the ACPI MR sizes kept for migration and the machine
//! wiring are left out; the `fw_cfg_init_*` constructors only build the device and its ops.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result, error_report};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, Endian, MemResult, MemTxAttrs, MmioOps,
};

/// QOM type names, kept for the machine code.
pub const TYPE_FW_CFG: &str = "fw_cfg";
/// `TYPE_FW_CFG_IO`.
pub const TYPE_FW_CFG_IO: &str = "fw_cfg_io";
/// `TYPE_FW_CFG_MEM`.
pub const TYPE_FW_CFG_MEM: &str = "fw_cfg_mem";
/// `TYPE_FW_CFG_DATA_GENERATOR_INTERFACE`.
pub const TYPE_FW_CFG_DATA_GENERATOR_INTERFACE: &str = "fw_cfg-data-generator";

/// The ACPI `_HID` of the device.
pub const FW_CFG_ACPI_DEVICE_ID: &str = "QEMU0002";

// Selector key values for "well-known" fw_cfg entries.
/// `FW_CFG_SIGNATURE`.
pub const FW_CFG_SIGNATURE: u16 = 0x00;
/// `FW_CFG_ID`.
pub const FW_CFG_ID: u16 = 0x01;
/// `FW_CFG_UUID`.
pub const FW_CFG_UUID: u16 = 0x02;
/// `FW_CFG_RAM_SIZE`.
pub const FW_CFG_RAM_SIZE: u16 = 0x03;
/// `FW_CFG_NOGRAPHIC`.
pub const FW_CFG_NOGRAPHIC: u16 = 0x04;
/// `FW_CFG_NB_CPUS`.
pub const FW_CFG_NB_CPUS: u16 = 0x05;
/// `FW_CFG_MACHINE_ID`.
pub const FW_CFG_MACHINE_ID: u16 = 0x06;
/// `FW_CFG_KERNEL_ADDR`.
pub const FW_CFG_KERNEL_ADDR: u16 = 0x07;
/// `FW_CFG_KERNEL_SIZE`.
pub const FW_CFG_KERNEL_SIZE: u16 = 0x08;
/// `FW_CFG_KERNEL_CMDLINE`.
pub const FW_CFG_KERNEL_CMDLINE: u16 = 0x09;
/// `FW_CFG_INITRD_ADDR`.
pub const FW_CFG_INITRD_ADDR: u16 = 0x0a;
/// `FW_CFG_INITRD_SIZE`.
pub const FW_CFG_INITRD_SIZE: u16 = 0x0b;
/// `FW_CFG_BOOT_DEVICE`.
pub const FW_CFG_BOOT_DEVICE: u16 = 0x0c;
/// `FW_CFG_NUMA`.
pub const FW_CFG_NUMA: u16 = 0x0d;
/// `FW_CFG_BOOT_MENU`.
pub const FW_CFG_BOOT_MENU: u16 = 0x0e;
/// `FW_CFG_MAX_CPUS`.
pub const FW_CFG_MAX_CPUS: u16 = 0x0f;
/// `FW_CFG_KERNEL_ENTRY`.
pub const FW_CFG_KERNEL_ENTRY: u16 = 0x10;
/// `FW_CFG_KERNEL_DATA`.
pub const FW_CFG_KERNEL_DATA: u16 = 0x11;
/// `FW_CFG_INITRD_DATA`.
pub const FW_CFG_INITRD_DATA: u16 = 0x12;
/// `FW_CFG_CMDLINE_ADDR`.
pub const FW_CFG_CMDLINE_ADDR: u16 = 0x13;
/// `FW_CFG_CMDLINE_SIZE`.
pub const FW_CFG_CMDLINE_SIZE: u16 = 0x14;
/// `FW_CFG_CMDLINE_DATA`.
pub const FW_CFG_CMDLINE_DATA: u16 = 0x15;
/// `FW_CFG_SETUP_ADDR`.
pub const FW_CFG_SETUP_ADDR: u16 = 0x16;
/// `FW_CFG_SETUP_SIZE`.
pub const FW_CFG_SETUP_SIZE: u16 = 0x17;
/// `FW_CFG_SETUP_DATA`.
pub const FW_CFG_SETUP_DATA: u16 = 0x18;
/// `FW_CFG_FILE_DIR`.
pub const FW_CFG_FILE_DIR: u16 = 0x19;

/// `FW_CFG_FILE_FIRST`: the key of the first file slot.
pub const FW_CFG_FILE_FIRST: u16 = 0x20;
/// `FW_CFG_FILE_SLOTS_MIN`.
pub const FW_CFG_FILE_SLOTS_MIN: u16 = 0x10;
/// `FW_CFG_FILE_SLOTS_DFLT`: the default of the `x-file-slots` property.
pub const FW_CFG_FILE_SLOTS_DFLT: u16 = 0x20;

/// `FW_CFG_WRITE_CHANNEL`.
pub const FW_CFG_WRITE_CHANNEL: u16 = 0x4000;
/// `FW_CFG_ARCH_LOCAL`.
pub const FW_CFG_ARCH_LOCAL: u16 = 0x8000;
/// `FW_CFG_ENTRY_MASK`.
pub const FW_CFG_ENTRY_MASK: u16 = !(FW_CFG_WRITE_CHANNEL | FW_CFG_ARCH_LOCAL);

/// `FW_CFG_INVALID`.
pub const FW_CFG_INVALID: u16 = 0xffff;

/// Width in bytes of the fw_cfg control register, `FW_CFG_CTL_SIZE`.
pub const FW_CFG_CTL_SIZE: u64 = 0x02;

/// A file name is up to 56 characters, terminating nul included, `FW_CFG_MAX_FILE_PATH`.
pub const FW_CFG_MAX_FILE_PATH: usize = 56;

/// Size in bytes of the signature, `FW_CFG_SIG_SIZE`.
pub const FW_CFG_SIG_SIZE: usize = 4;

// FW_CFG_ID bits.
/// `FW_CFG_VERSION`: the traditional interface.
pub const FW_CFG_VERSION: u32 = 0x01;
/// `FW_CFG_VERSION_DMA`: the DMA interface.
pub const FW_CFG_VERSION_DMA: u32 = 0x02;

// FW_CFG_DMA_CONTROL bits.
/// `FW_CFG_DMA_CTL_ERROR`.
pub const FW_CFG_DMA_CTL_ERROR: u32 = 0x01;
/// `FW_CFG_DMA_CTL_READ`.
pub const FW_CFG_DMA_CTL_READ: u32 = 0x02;
/// `FW_CFG_DMA_CTL_SKIP`.
pub const FW_CFG_DMA_CTL_SKIP: u32 = 0x04;
/// `FW_CFG_DMA_CTL_SELECT`.
pub const FW_CFG_DMA_CTL_SELECT: u32 = 0x08;
/// `FW_CFG_DMA_CTL_WRITE`.
pub const FW_CFG_DMA_CTL_WRITE: u32 = 0x10;

/// "QEMU CFG", what the DMA register reads as, `FW_CFG_DMA_SIGNATURE`.
pub const FW_CFG_DMA_SIGNATURE: u64 = 0x51454d5520434647;

/// `FW_CFG_VMCOREINFO_FILENAME`.
pub const FW_CFG_VMCOREINFO_FILENAME: &str = "etc/vmcoreinfo";
/// `FW_CFG_VMCOREINFO_FORMAT_NONE`.
pub const FW_CFG_VMCOREINFO_FORMAT_NONE: u16 = 0x0;
/// `FW_CFG_VMCOREINFO_FORMAT_ELF`.
pub const FW_CFG_VMCOREINFO_FORMAT_ELF: u16 = 0x1;

/// The PC port base, `FW_CFG_IO_BASE` from hw/i386/fw_cfg.h.
pub const FW_CFG_IO_BASE: u32 = 0x510;

/// Size of one `struct fw_cfg_file` in the file directory: size, select, reserved and name.
pub const FW_CFG_FILE_SIZE: usize = 8 + FW_CFG_MAX_FILE_PATH;

/// Size of `struct fw_cfg_dma_access`: control, length and address, all big endian.
pub const FW_CFG_DMA_ACCESS_SIZE: usize = 16;

/// Size of the DMA register region, `sizeof(dma_addr_t)`.
pub const FW_CFG_DMA_SIZE: u64 = 8;

/// Guest memory as seen by the DMA interface. Both calls return `false` if any part of the
/// access failed, like a `MemTxResult` other than `MEMTX_OK`.
pub trait DmaMemory: Send + Sync {
    /// `dma_memory_read()`.
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool;

    /// `dma_memory_write()`.
    fn write(&self, addr: u64, buf: &[u8]) -> bool;
}

impl DmaMemory for AddressSpace {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        AddressSpace::read(self, addr, MemTxAttrs::UNSPECIFIED, buf).is_ok()
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        AddressSpace::write(self, addr, MemTxAttrs::UNSPECIFIED, buf).is_ok()
    }
}

/// `FWCfgCallback`: runs when the entry is selected. It gets the entry data and may update it
/// in place, which is how lazily built tables are filled in.
pub type FwCfgCallback = Box<dyn FnMut(&mut [u8]) + Send>;

/// `FWCfgWriteCallback`: runs after a DMA write to the entry. It gets the entry data, the
/// offset of the write and its length.
pub type FwCfgWriteCallback = Box<dyn FnMut(&[u8], u32, u32) + Send>;

/// `FWCfgDataGeneratorClass`: an object that produces the contents of a file on demand.
pub trait FwCfgDataGenerator {
    /// `get_data()`: the data to add, or `None` if no data is required.
    fn get_data(&self) -> Result<Option<Vec<u8>>>;
}

/// `key_name()`: the name of a well known key, for messages. Arch local keys have their names
/// in the target code, so they are `None` here.
pub fn key_name(key: u16) -> Option<&'static str> {
    const NAMES: [&str; FW_CFG_FILE_DIR as usize + 1] = [
        "signature",
        "id",
        "uuid",
        "ram_size",
        "nographic",
        "nb_cpus",
        "machine_id",
        "kernel_addr",
        "kernel_size",
        "kernel_cmdline",
        "initrd_addr",
        "initdr_size",
        "boot_device",
        "numa",
        "boot_menu",
        "max_cpus",
        "kernel_entry",
        "kernel_data",
        "initrd_data",
        "cmdline_addr",
        "cmdline_size",
        "cmdline_data",
        "setup_addr",
        "setup_size",
        "setup_data",
        "file_dir",
    ];
    if key & FW_CFG_ARCH_LOCAL != 0 {
        return None;
    }
    NAMES.get(usize::from(key)).copied()
}

/// The device properties shared by both flavors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FwCfgProps {
    /// `dma_enabled`, default true.
    pub dma_enabled: bool,
    /// `x-file-slots`, default [`FW_CFG_FILE_SLOTS_DFLT`].
    pub file_slots: u16,
}

impl Default for FwCfgProps {
    fn default() -> Self {
        FwCfgProps { dma_enabled: true, file_slots: FW_CFG_FILE_SLOTS_DFLT }
    }
}

/// What `fw_cfg_common_realize()` takes from the machine: `qemu_uuid`, `enable_graphics` and
/// the `-boot` options.
#[derive(Clone, Debug)]
pub struct FwCfgMachineConfig {
    /// `qemu_uuid`.
    pub uuid: [u8; 16],
    /// `machine->enable_graphics`.
    pub enable_graphics: bool,
    /// `-boot menu=`.
    pub boot_menu: Option<bool>,
    /// `-boot splash-time=`.
    pub splash_time: Option<i64>,
    /// `-boot splash=`, already resolved to a path by `qemu_find_file()`.
    pub splash: Option<String>,
    /// `-boot reboot-timeout=`.
    pub reboot_timeout: Option<i64>,
}

impl Default for FwCfgMachineConfig {
    fn default() -> Self {
        FwCfgMachineConfig {
            uuid: [0; 16],
            enable_graphics: true,
            boot_menu: None,
            splash_time: None,
            splash: None,
            reboot_timeout: None,
        }
    }
}

/// `FWCfgEntry`.
#[derive(Default)]
struct FwCfgEntry {
    data: Option<Vec<u8>>,
    allow_write: bool,
    select_cb: Option<FwCfgCallback>,
    write_cb: Option<FwCfgWriteCallback>,
}

impl FwCfgEntry {
    fn len(&self) -> usize {
        self.data.as_ref().map_or(0, Vec::len)
    }
}

/// The mutable part of `FWCfgState`.
struct FwCfgInner {
    file_slots: u16,
    entries: [Vec<FwCfgEntry>; 2],
    // Whether `s->files` exists. The directory itself is the data of FW_CFG_FILE_DIR.
    has_files: bool,
    cur_entry: u16,
    cur_offset: u32,
    dma_addr: u64,
}

fn arch(key: u16) -> usize {
    usize::from(key & FW_CFG_ARCH_LOCAL != 0)
}

fn file_off(i: usize) -> usize {
    4 + i * FW_CFG_FILE_SIZE
}

// The part of a C string before the nul.
fn cstr(b: &[u8]) -> &[u8] {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    &b[..end]
}

impl FwCfgInner {
    /// `fw_cfg_max_entry()`: an exclusive limit.
    fn max_entry(&self) -> u32 {
        u32::from(FW_CFG_FILE_FIRST) + u32::from(self.file_slots)
    }

    /// The slot of the current entry, or `None` for `FW_CFG_INVALID`.
    fn cur_slot(&self) -> Option<(usize, usize)> {
        if self.cur_entry == FW_CFG_INVALID {
            return None;
        }
        Some((arch(self.cur_entry), usize::from(self.cur_entry & FW_CFG_ENTRY_MASK)))
    }

    /// `fw_cfg_select()`.
    fn select(&mut self, key: u16) -> bool {
        self.cur_offset = 0;
        if u32::from(key & FW_CFG_ENTRY_MASK) >= self.max_entry() {
            self.cur_entry = FW_CFG_INVALID;
            return false;
        }
        self.cur_entry = key;
        // Entry successfully selected, now run the callback if present.
        let e = &mut self.entries[arch(key)][usize::from(key & FW_CFG_ENTRY_MASK)];
        let FwCfgEntry { data, select_cb, .. } = e;
        if let Some(cb) = select_cb.as_mut() {
            cb(data.as_deref_mut().unwrap_or_default());
        }
        true
    }

    /// `fw_cfg_data_read()`.
    fn data_read(&mut self, size: u32) -> u64 {
        assert!(size > 0 && size <= 8);
        let mut value = 0u64;
        let Some((a, k)) = self.cur_slot() else {
            return 0;
        };
        let Some(data) = self.entries[a][k].data.as_deref() else {
            return 0;
        };
        let mut off = self.cur_offset as usize;
        if off < data.len() {
            // The least significant `size` bytes of the return value hold a string
            // preserving part of the item, padded with zeros on the right if we run out
            // early: the host endian form of the big endian reading of the data.
            let mut size = size;
            loop {
                value = (value << 8) | u64::from(data[off]);
                off += 1;
                size -= 1;
                if size == 0 || off >= data.len() {
                    break;
                }
            }
            // If size is still not zero we did run out early, so shift in the padding.
            value <<= 8 * size;
            self.cur_offset = off as u32;
        }
        value
    }

    /// `fw_cfg_add_bytes_callback()`.
    fn add_bytes_callback(
        &mut self,
        key: u16,
        select_cb: Option<FwCfgCallback>,
        write_cb: Option<FwCfgWriteCallback>,
        data: Vec<u8>,
        read_only: bool,
    ) {
        let a = arch(key);
        let key = key & FW_CFG_ENTRY_MASK;
        assert!(u32::from(key) < self.max_entry() && data.len() < u32::MAX as usize);
        let e = &mut self.entries[a][usize::from(key)];
        // Avoid key conflict.
        assert!(e.data.is_none(), "fw_cfg key {key:#x} added twice");
        *e = FwCfgEntry { data: Some(data), allow_write: !read_only, select_cb, write_cb };
    }

    /// `fw_cfg_modify_bytes_read()`: replaces the data and returns the old one.
    fn modify_bytes_read(&mut self, key: u16, data: Vec<u8>) -> Option<Vec<u8>> {
        let a = arch(key);
        let key = key & FW_CFG_ENTRY_MASK;
        assert!(u32::from(key) < self.max_entry() && data.len() < u32::MAX as usize);
        let e = &mut self.entries[a][usize::from(key)];
        e.allow_write = false;
        e.data.replace(data)
    }

    fn files(&self) -> &[u8] {
        self.entries[0][usize::from(FW_CFG_FILE_DIR)].data.as_deref().unwrap_or_default()
    }

    fn files_mut(&mut self) -> &mut [u8] {
        self.entries[0][usize::from(FW_CFG_FILE_DIR)].data.as_deref_mut().unwrap_or_default()
    }

    fn files_count(&self) -> usize {
        let f = self.files();
        u32::from_be_bytes([f[0], f[1], f[2], f[3]]) as usize
    }

    fn file_name(&self, i: usize) -> &[u8] {
        let o = file_off(i) + 8;
        cstr(&self.files()[o..o + FW_CFG_MAX_FILE_PATH])
    }

    /// `fw_cfg_add_file_callback()`.
    fn add_file_callback(
        &mut self,
        filename: &str,
        select_cb: Option<FwCfgCallback>,
        write_cb: Option<FwCfgWriteCallback>,
        data: Vec<u8>,
        read_only: bool,
    ) -> Result<()> {
        let slots = usize::from(self.file_slots);
        if !self.has_files {
            let dsize = 4 + FW_CFG_FILE_SIZE * slots;
            self.add_bytes_callback(FW_CFG_FILE_DIR, None, None, vec![0; dsize], true);
            self.has_files = true;
        }

        let count = self.files_count();
        assert!(count < slots, "fw_cfg file directory is full");

        // pstrcpy() into the directory truncates the name, sorting uses the full one.
        let full = cstr(filename.as_bytes());
        let name = &full[..full.len().min(FW_CFG_MAX_FILE_PATH - 1)];

        // Checked before anything moves; QEMU exits here after the move.
        if (0..count).any(|i| self.file_name(i) == name) {
            let name = String::from_utf8_lossy(name);
            return Err(Error::generic(format!("duplicate fw_cfg file name: {name}")));
        }

        // Find the insertion point, sorting by file name.
        let mut index = count;
        while index > 0 && full < self.file_name(index - 1) {
            index -= 1;
        }

        // Move the entries from the index point and after down one to make room.
        let first = usize::from(FW_CFG_FILE_FIRST);
        let files = self.files_mut();
        files.copy_within(file_off(index)..file_off(count), file_off(index + 1));
        for i in index + 1..=count {
            let o = file_off(i) + 4;
            files[o..o + 2].copy_from_slice(&(FW_CFG_FILE_FIRST + i as u16).to_be_bytes());
        }
        self.entries[0][first + index..=first + count].rotate_right(1);

        let files = self.files_mut();
        let o = file_off(index);
        files[o..o + FW_CFG_FILE_SIZE].fill(0);
        files[o + 8..o + 8 + name.len()].copy_from_slice(name);
        self.entries[0][first + index] = FwCfgEntry::default();

        let len = data.len() as u32;
        let key = FW_CFG_FILE_FIRST + index as u16;
        self.add_bytes_callback(key, select_cb, write_cb, data, read_only);

        let files = self.files_mut();
        files[o..o + 4].copy_from_slice(&len.to_be_bytes());
        files[o + 4..o + 6].copy_from_slice(&key.to_be_bytes());
        files[0..4].copy_from_slice(&(count as u32 + 1).to_be_bytes());
        Ok(())
    }
}

/// `FWCfgState`: the entry table and the guest visible state.
pub struct FwCfgState {
    inner: Mutex<FwCfgInner>,
    dma_enabled: bool,
    dma_as: Option<Arc<dyn DmaMemory>>,
}

impl fmt::Debug for FwCfgState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.lock();
        f.debug_struct("FwCfgState")
            .field("file_slots", &s.file_slots)
            .field("cur_entry", &s.cur_entry)
            .field("cur_offset", &s.cur_offset)
            .field("dma_enabled", &self.dma_enabled)
            .field("dma_addr", &s.dma_addr)
            .finish_non_exhaustive()
    }
}

/// `read_splashfile()`: the contents and whether the file is a JPEG, or `None` after an error
/// message.
fn read_splashfile(filename: &str) -> Option<(Vec<u8>, bool)> {
    let content = match std::fs::read(filename) {
        Ok(c) => c,
        Err(e) => {
            error_report(&format!(
                "failed to read splash file '{filename}': {}",
                ruvm_base::error::strerror(&e)
            ));
            return None;
        }
    };
    let bad = || {
        error_report(&format!(
            "splash file '{filename}' format not recognized; must be JPEG or 24 bit BMP"
        ));
        None
    };
    // Check the file size.
    if content.len() < 30 {
        return bad();
    }
    // Check the magic ID, and the bpp for a BMP.
    let is_jpg = match u16::from_le_bytes([content[0], content[1]]) {
        0xd8ff => true,
        0x4d42 => {
            if u16::from_le_bytes([content[28], content[29]]) != 24 {
                return bad();
            }
            false
        }
        _ => return bad(),
    };
    Some((content, is_jpg))
}

impl FwCfgState {
    /// Builds an empty device, `fw_cfg_file_slots_allocate()` included. `dma_as` is only kept
    /// if `props.dma_enabled` is set.
    pub fn new(props: FwCfgProps, dma_as: Option<Arc<dyn DmaMemory>>) -> Result<Self> {
        if props.file_slots < FW_CFG_FILE_SLOTS_MIN {
            return Err(Error::generic(format!(
                "\"file_slots\" must be at least 0x{FW_CFG_FILE_SLOTS_MIN:x}"
            )));
        }
        // (UINT16_MAX & FW_CFG_ENTRY_MASK) is the highest inclusive selector value we permit.
        let file_slots_max = FW_CFG_ENTRY_MASK - FW_CFG_FILE_FIRST + 1;
        if props.file_slots > file_slots_max {
            return Err(Error::generic(format!(
                "\"file_slots\" must not exceed 0x{file_slots_max:x}"
            )));
        }
        let max = usize::from(FW_CFG_FILE_FIRST) + usize::from(props.file_slots);
        let table = || (0..max).map(|_| FwCfgEntry::default()).collect::<Vec<_>>();
        Ok(FwCfgState {
            inner: Mutex::new(FwCfgInner {
                file_slots: props.file_slots,
                entries: [table(), table()],
                has_files: false,
                cur_entry: 0,
                cur_offset: 0,
                dma_addr: 0,
            }),
            dma_enabled: props.dma_enabled,
            dma_as: if props.dma_enabled { dma_as } else { None },
        })
    }

    fn lock(&self) -> MutexGuard<'_, FwCfgInner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// `fw_cfg_dma_enabled()`.
    pub fn dma_enabled(&self) -> bool {
        self.dma_enabled
    }

    /// `fw_cfg_file_slots()`.
    pub fn file_slots(&self) -> u16 {
        self.lock().file_slots
    }

    /// The selected key, `FW_CFG_INVALID` if the last select failed.
    pub fn cur_entry(&self) -> u16 {
        self.lock().cur_entry
    }

    /// The read offset into the selected entry.
    pub fn cur_offset(&self) -> u32 {
        self.lock().cur_offset
    }

    /// `fw_cfg_common_realize()`: the signature, UUID, graphics and boot entries and `FW_CFG_ID`.
    pub fn common_realize(&self, cfg: &FwCfgMachineConfig) -> Result<()> {
        self.add_bytes(FW_CFG_SIGNATURE, b"QEMU".to_vec());
        self.add_bytes(FW_CFG_UUID, cfg.uuid.to_vec());
        self.add_i16(FW_CFG_NOGRAPHIC, u16::from(!cfg.enable_graphics));
        self.add_i16(FW_CFG_BOOT_MENU, u16::from(cfg.boot_menu == Some(true)));
        self.bootsplash(cfg)?;
        self.reboot(cfg)?;

        let mut version = FW_CFG_VERSION;
        if self.dma_enabled {
            version |= FW_CFG_VERSION_DMA;
        }
        self.add_i32(FW_CFG_ID, version);
        Ok(())
    }

    /// `fw_cfg_bootsplash()`.
    fn bootsplash(&self, cfg: &FwCfgMachineConfig) -> Result<()> {
        // Insert the splash time if the user configured it.
        if let Some(bst) = cfg.splash_time {
            if !(0..=0xffff).contains(&bst) {
                return Err(Error::generic(
                    "splash-time is invalid,it should be a value between 0 and 65535",
                ));
            }
            self.add_file("etc/boot-menu-wait", (bst as u16).to_le_bytes().to_vec())?;
        }

        // Insert the splash file if the user configured it.
        if let Some(filename) = cfg.splash.as_deref() {
            if let Some((data, is_jpg)) = read_splashfile(filename) {
                let name = if is_jpg { "bootsplash.jpg" } else { "bootsplash.bmp" };
                self.add_file(name, data)?;
            }
        }
        Ok(())
    }

    /// `fw_cfg_reboot()`.
    fn reboot(&self, cfg: &FwCfgMachineConfig) -> Result<()> {
        let mut rt_val = u64::MAX;
        if let Some(rt) = cfg.reboot_timeout {
            rt_val = rt as u64;
            if rt_val > 0xffff && rt_val != u64::MAX {
                return Err(Error::generic(
                    "reboot timeout is invalid,it should be a value between -1 and 65535",
                ));
            }
        }
        self.add_file("etc/boot-fail-wait", (rt_val as u32).to_le_bytes().to_vec())
    }

    /// `fw_cfg_reset()`: selects the signature, which never has a callback.
    pub fn reset(&self) {
        self.select(FW_CFG_SIGNATURE);
    }

    /// `fw_cfg_machine_reset()`: refreshes `bootorder` and `bios-geometry` from the lists
    /// `get_boot_devices_list()` and `get_boot_devices_lchs_list()` built.
    pub fn machine_reset(&self, bootorder: Vec<u8>, bios_geometry: Vec<u8>) -> Result<()> {
        self.modify_file("bootorder", bootorder)?;
        self.modify_file("bios-geometry", bios_geometry)?;
        Ok(())
    }

    /// `fw_cfg_select()`: returns whether the key names a slot.
    pub fn select(&self, key: u16) -> bool {
        self.lock().select(key)
    }

    /// `fw_cfg_data_read()`: the next `size` bytes of the selected entry, first byte most
    /// significant, zero padded past the end.
    pub fn data_read(&self, size: u32) -> u64 {
        self.lock().data_read(size)
    }

    /// `fw_cfg_data_mem_write()`: ignored, write support was removed in QEMU 2.4.
    pub fn data_write(&self, _value: u64, _size: u32) {}

    /// `fw_cfg_dma_mem_read()`: the DMA register reads as [`FW_CFG_DMA_SIGNATURE`].
    pub fn dma_mem_read(&self, addr: u64, size: u32) -> u64 {
        if size == 0 || addr + u64::from(size) > 8 {
            return 0;
        }
        let v = FW_CFG_DMA_SIGNATURE >> ((8 - addr - u64::from(size)) * 8);
        if size == 8 { v } else { v & ((1u64 << (size * 8)) - 1) }
    }

    /// `fw_cfg_dma_mem_write()`: a write of the low half, or of all 8 bytes, starts a transfer.
    pub fn dma_mem_write(&self, addr: u64, value: u64, size: u32) {
        let mut s = self.lock();
        if size == 4 {
            if addr == 0 {
                // FWCfgDmaAccess high address.
                s.dma_addr = value << 32;
            } else if addr == 4 {
                // FWCfgDmaAccess low address.
                s.dma_addr |= value;
                self.dma_transfer(&mut s);
            }
        } else if size == 8 && addr == 0 {
            s.dma_addr = value;
            self.dma_transfer(&mut s);
        }
    }

    /// `fw_cfg_dma_transfer()`.
    fn dma_transfer(&self, s: &mut FwCfgInner) {
        // Reset the address before the next access.
        let dma_addr = s.dma_addr;
        s.dma_addr = 0;
        let Some(mem) = self.dma_as.as_deref() else {
            return;
        };

        // The control word is the first field of FWCfgDmaAccess.
        let mut raw = [0u8; FW_CFG_DMA_ACCESS_SIZE];
        if !mem.read(dma_addr, &mut raw) {
            mem.write(dma_addr, &FW_CFG_DMA_CTL_ERROR.to_be_bytes());
            return;
        }
        let mut control = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
        let mut length = u32::from_be_bytes([raw[4], raw[5], raw[6], raw[7]]);
        let mut address = u64::from_be_bytes([
            raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
        ]);

        if control & FW_CFG_DMA_CTL_SELECT != 0 {
            s.select((control >> 16) as u16);
        }

        let slot = s.cur_slot();

        let (read, write) = if control & FW_CFG_DMA_CTL_READ != 0 {
            (true, false)
        } else if control & FW_CFG_DMA_CTL_WRITE != 0 {
            (false, true)
        } else if control & FW_CFG_DMA_CTL_SKIP != 0 {
            (false, false)
        } else {
            length = 0;
            (false, false)
        };

        control = 0;

        while length > 0 && control & FW_CFG_DMA_CTL_ERROR == 0 {
            let off = s.cur_offset as usize;
            let entry = slot.map(|(a, k)| &mut s.entries[a][k]);
            let len = match entry {
                Some(e) if off < e.len() => {
                    let len = length.min((e.len() - off) as u32);
                    let FwCfgEntry { data, allow_write, write_cb, .. } = e;
                    let data = data.as_deref_mut().unwrap_or_default();
                    let range = off..off + len as usize;

                    // Not a read or a write means a skip, tested above.
                    if read && !mem.write(address, &data[range.clone()]) {
                        control |= FW_CFG_DMA_CTL_ERROR;
                    }
                    if write {
                        if !*allow_write || len != length || !mem.read(address, &mut data[range]) {
                            control |= FW_CFG_DMA_CTL_ERROR;
                        } else if let Some(cb) = write_cb.as_mut() {
                            cb(data, off as u32, len);
                        }
                    }

                    s.cur_offset += len;
                    len
                }
                _ => {
                    // Nothing left to read: reads fill with zeros, writes fail.
                    if read && !dma_memory_zero(mem, address, u64::from(length)) {
                        control |= FW_CFG_DMA_CTL_ERROR;
                    }
                    if write {
                        control |= FW_CFG_DMA_CTL_ERROR;
                    }
                    length
                }
            };

            address = address.wrapping_add(u64::from(len));
            length -= len;
        }

        mem.write(dma_addr, &control.to_be_bytes());
    }

    /// `fw_cfg_add_bytes()`. Panics if the key is out of range or already has data.
    pub fn add_bytes(&self, key: u16, data: Vec<u8>) {
        self.lock().add_bytes_callback(key, None, None, data, true);
    }

    /// `fw_cfg_add_string()`: the string with its nul.
    pub fn add_string(&self, key: u16, value: &str) {
        self.add_bytes(key, nul_terminated(value));
    }

    /// `fw_cfg_modify_string()`.
    pub fn modify_string(&self, key: u16, value: &str) {
        self.lock().modify_bytes_read(key, nul_terminated(value));
    }

    /// `fw_cfg_add_i16()`: stored little endian.
    pub fn add_i16(&self, key: u16, value: u16) {
        self.add_bytes(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_modify_i16()`.
    pub fn modify_i16(&self, key: u16, value: u16) {
        self.lock().modify_bytes_read(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_add_i32()`: stored little endian.
    pub fn add_i32(&self, key: u16, value: u32) {
        self.add_bytes(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_modify_i32()`.
    pub fn modify_i32(&self, key: u16, value: u32) {
        self.lock().modify_bytes_read(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_add_i64()`: stored little endian.
    pub fn add_i64(&self, key: u16, value: u64) {
        self.add_bytes(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_modify_i64()`.
    pub fn modify_i64(&self, key: u16, value: u64) {
        self.lock().modify_bytes_read(key, value.to_le_bytes().to_vec());
    }

    /// `fw_cfg_add_file_callback()`: adds a file, keeping the directory sorted by name. Entries
    /// already added after the new one move up a slot and get a new selector. A file that is not
    /// `read_only` can be written through DMA; `write_cb` then runs after each write. Panics if
    /// the directory is full; a duplicate name is an error.
    pub fn add_file_callback(
        &self,
        filename: &str,
        select_cb: Option<FwCfgCallback>,
        write_cb: Option<FwCfgWriteCallback>,
        data: Vec<u8>,
        read_only: bool,
    ) -> Result<()> {
        self.lock().add_file_callback(filename, select_cb, write_cb, data, read_only)
    }

    /// `fw_cfg_add_file()`.
    pub fn add_file(&self, filename: &str, data: Vec<u8>) -> Result<()> {
        self.add_file_callback(filename, None, None, data, true)
    }

    /// `fw_cfg_modify_file()`: replaces the data of a file and returns the old data, or adds
    /// the file and returns `None`. Panics if no file was ever added.
    pub fn modify_file(&self, filename: &str, data: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let mut s = self.lock();
        assert!(s.has_files, "fw_cfg_modify_file() before any file was added");

        let count = s.files_count();
        let name = filename.as_bytes();
        if let Some(i) = (0..count).find(|&i| s.file_name(i) == cstr(name)) {
            let len = data.len() as u32;
            let old = s.modify_bytes_read(FW_CFG_FILE_FIRST + i as u16, data);
            let o = file_off(i);
            s.files_mut()[o..o + 4].copy_from_slice(&len.to_be_bytes());
            return Ok(old);
        }

        assert!(count < usize::from(s.file_slots));

        // Add a new one.
        s.add_file_callback(filename, None, None, data, true)?;
        Ok(None)
    }

    /// `fw_cfg_add_file_from_generator()`: adds the data `generator` produces as `filename`.
    /// Returns `Ok(false)` if the generator had nothing to add. Resolving the object by ID is
    /// the caller's job.
    pub fn add_file_from_generator(
        &self,
        generator: &dyn FwCfgDataGenerator,
        filename: &str,
    ) -> Result<bool> {
        let Some(data) = generator.get_data()? else {
            return Ok(false);
        };
        self.add_file(filename, data)?;
        Ok(true)
    }

    /// The directory as `(name, select, size)` in slot order, for tests and debugging.
    pub fn files(&self) -> Vec<(String, u16, u32)> {
        let s = self.lock();
        if !s.has_files {
            return Vec::new();
        }
        (0..s.files_count())
            .map(|i| {
                let f = &s.files()[file_off(i)..file_off(i + 1)];
                (
                    String::from_utf8_lossy(s.file_name(i)).into_owned(),
                    u16::from_be_bytes([f[4], f[5]]),
                    u32::from_be_bytes([f[0], f[1], f[2], f[3]]),
                )
            })
            .collect()
    }

    /// A copy of the data of `key`, `None` if the entry is empty or out of range.
    pub fn entry_data(&self, key: u16) -> Option<Vec<u8>> {
        let s = self.lock();
        let k = usize::from(key & FW_CFG_ENTRY_MASK);
        s.entries[arch(key)].get(k).and_then(|e| e.data.clone())
    }
}

fn nul_terminated(value: &str) -> Vec<u8> {
    let mut v = cstr(value.as_bytes()).to_vec();
    v.push(0);
    v
}

/// `dma_memory_set()` with zero.
fn dma_memory_zero(mem: &dyn DmaMemory, addr: u64, len: u64) -> bool {
    const CHUNK: u64 = 4096;
    let zeros = [0u8; CHUNK as usize];
    let mut done = 0;
    let mut ok = true;
    while done < len {
        let n = (len - done).min(CHUNK);
        ok &= mem.write(addr.wrapping_add(done), &zeros[..n as usize]);
        done += n;
    }
    ok
}

/// `fw_cfg_comb_mem_ops`: the I/O port flavor's selector and data register. A 2 byte write
/// selects, a 1 byte read returns data, a 1 byte write is ignored.
#[derive(Debug)]
pub struct FwCfgCombOps(pub Arc<FwCfgState>);

impl MmioOps for FwCfgCombOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.data_read(size.bytes()))
    }

    /// `fw_cfg_comb_write()`.
    fn write(&self, _cx: &AccessCtx, _offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        match size.bytes() {
            1 => self.0.data_write(value, 1),
            2 => {
                self.0.select(value as u16);
            }
            _ => {}
        }
        Ok(())
    }

    /// `fw_cfg_comb_valid()`.
    fn accepts(&self, _offset: u64, size: u32, is_write: bool, _attrs: MemTxAttrs) -> bool {
        size == 1 || (is_write && size == 2)
    }

    fn endianness(&self) -> Endian {
        Endian::Little
    }
}

/// `fw_cfg_ctl_mem_ops`: the MMIO selector register, 2 byte writes only.
#[derive(Debug)]
pub struct FwCfgCtlOps(pub Arc<FwCfgState>);

impl MmioOps for FwCfgCtlOps {
    /// `fw_cfg_ctl_mem_read()`.
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    /// `fw_cfg_ctl_mem_write()`.
    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.select(value as u16);
        Ok(())
    }

    /// `fw_cfg_ctl_mem_valid()`.
    fn accepts(&self, _offset: u64, size: u32, is_write: bool, _attrs: MemTxAttrs) -> bool {
        is_write && size == 2
    }

    fn endianness(&self) -> Endian {
        Endian::Big
    }
}

/// `fw_cfg_data_mem_ops`, widened to `data_width` bytes when that is more than one.
#[derive(Debug)]
pub struct FwCfgDataOps {
    state: Arc<FwCfgState>,
    data_width: u32,
}

impl FwCfgDataOps {
    /// Ops for a data register `data_width` bytes wide.
    pub fn new(state: Arc<FwCfgState>, data_width: u32) -> Self {
        FwCfgDataOps { state, data_width }
    }

    fn wide(&self) -> bool {
        self.data_width > 1
    }

    /// The size of the data region, `data_ops->valid.max_access_size`.
    pub fn region_size(&self) -> u64 {
        if self.wide() { u64::from(self.data_width) } else { 1 }
    }
}

impl MmioOps for FwCfgDataOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.state.data_read(size.bytes()))
    }

    /// `fw_cfg_data_mem_write()`.
    fn write(&self, _cx: &AccessCtx, _offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.state.data_write(value, size.bytes());
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, self.region_size() as u32)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        if self.wide() {
            AccessConstraints::any_size(0, self.data_width)
        } else {
            AccessConstraints::default()
        }
    }

    /// `fw_cfg_data_mem_valid()`.
    fn accepts(&self, offset: u64, _size: u32, _is_write: bool, _attrs: MemTxAttrs) -> bool {
        offset == 0
    }

    fn endianness(&self) -> Endian {
        Endian::Big
    }
}

/// `fw_cfg_dma_mem_ops`: the DMA address register, big endian, reading as the DMA signature.
#[derive(Debug)]
pub struct FwCfgDmaOps(pub Arc<FwCfgState>);

impl MmioOps for FwCfgDmaOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.dma_mem_read(offset, size.bytes()))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.0.dma_mem_write(offset, value, size.bytes());
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(0, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(0, 8)
    }

    /// `fw_cfg_dma_mem_valid()`.
    fn accepts(&self, offset: u64, size: u32, is_write: bool, _attrs: MemTxAttrs) -> bool {
        !is_write || (size == 4 && (offset == 0 || offset == 4)) || (size == 8 && offset == 0)
    }

    fn endianness(&self) -> Endian {
        Endian::Big
    }
}

/// `FWCfgIoState`: the "fwcfg" port pair at `iobase` and, with DMA, "fwcfg.dma" at
/// `iobase + 4`.
#[derive(Debug)]
pub struct FwCfgIo {
    state: Arc<FwCfgState>,
    iobase: u32,
    comb: Arc<FwCfgCombOps>,
    dma: Option<Arc<FwCfgDmaOps>>,
}

impl FwCfgIo {
    /// `fw_cfg_io_realize()`.
    pub fn new(
        iobase: u32,
        props: FwCfgProps,
        dma_as: Option<Arc<dyn DmaMemory>>,
        cfg: &FwCfgMachineConfig,
    ) -> Result<Self> {
        let state = Arc::new(FwCfgState::new(props, dma_as)?);
        // With port I/O the 8 bit data register always overlaps half of the 16 bit control
        // register, so the region is FW_CFG_CTL_SIZE bytes.
        let comb = Arc::new(FwCfgCombOps(state.clone()));
        let dma = state.dma_enabled.then(|| Arc::new(FwCfgDmaOps(state.clone())));
        state.common_realize(cfg)?;
        Ok(FwCfgIo { state, iobase, comb, dma })
    }

    /// The device state.
    pub fn state(&self) -> &Arc<FwCfgState> {
        &self.state
    }

    /// Where the port pair goes.
    pub fn iobase(&self) -> u32 {
        self.iobase
    }

    /// The "fwcfg" region, [`FW_CFG_CTL_SIZE`] bytes at `iobase`.
    pub fn comb_ops(&self) -> &Arc<FwCfgCombOps> {
        &self.comb
    }

    /// The "fwcfg.dma" region, [`FW_CFG_DMA_SIZE`] bytes at `iobase + 4`, if DMA is enabled.
    pub fn dma_ops(&self) -> Option<&Arc<FwCfgDmaOps>> {
        self.dma.as_ref()
    }
}

/// `fw_cfg_init_io_dma()`: the PC flavor with DMA enabled and default properties.
pub fn fw_cfg_init_io_dma(
    iobase: u32,
    dma_as: Arc<dyn DmaMemory>,
    cfg: &FwCfgMachineConfig,
) -> Result<FwCfgIo> {
    FwCfgIo::new(iobase, FwCfgProps::default(), Some(dma_as), cfg)
}

/// `FWCfgMemState`: "fwcfg.ctl", "fwcfg.data" and, with DMA, "fwcfg.dma".
#[derive(Debug)]
pub struct FwCfgMem {
    state: Arc<FwCfgState>,
    ctl: Arc<FwCfgCtlOps>,
    data: Arc<FwCfgDataOps>,
    dma: Option<Arc<FwCfgDmaOps>>,
    ctl_addr: u64,
    data_addr: u64,
    dma_addr: u64,
}

impl FwCfgMem {
    /// `fw_cfg_mem_realize()`. `data_width` must be 1, 2, 4 or 8.
    pub fn new(
        data_width: u32,
        props: FwCfgProps,
        dma_as: Option<Arc<dyn DmaMemory>>,
        cfg: &FwCfgMachineConfig,
    ) -> Result<Self> {
        if AccessSize::new(data_width).is_none() {
            return Err(Error::generic("\"data_width\" must be 1, 2, 4 or 8"));
        }
        let state = Arc::new(FwCfgState::new(props, dma_as)?);
        let ctl = Arc::new(FwCfgCtlOps(state.clone()));
        let data = Arc::new(FwCfgDataOps::new(state.clone(), data_width));
        let dma = state.dma_enabled.then(|| Arc::new(FwCfgDmaOps(state.clone())));
        state.common_realize(cfg)?;
        Ok(FwCfgMem { state, ctl, data, dma, ctl_addr: 0, data_addr: 0, dma_addr: 0 })
    }

    /// The device state.
    pub fn state(&self) -> &Arc<FwCfgState> {
        &self.state
    }

    /// The "fwcfg.ctl" region, [`FW_CFG_CTL_SIZE`] bytes.
    pub fn ctl_ops(&self) -> &Arc<FwCfgCtlOps> {
        &self.ctl
    }

    /// The "fwcfg.data" region, [`FwCfgDataOps::region_size`] bytes.
    pub fn data_ops(&self) -> &Arc<FwCfgDataOps> {
        &self.data
    }

    /// The "fwcfg.dma" region, [`FW_CFG_DMA_SIZE`] bytes, if DMA is enabled.
    pub fn dma_ops(&self) -> Option<&Arc<FwCfgDmaOps>> {
        self.dma.as_ref()
    }

    /// Where the selector, data and DMA regions go.
    pub fn addrs(&self) -> (u64, u64, u64) {
        (self.ctl_addr, self.data_addr, self.dma_addr)
    }
}

/// `fw_cfg_init_mem_internal()`: DMA is enabled only if both `dma_addr` and `dma_as` are given.
fn fw_cfg_init_mem_internal(
    ctl_addr: u64,
    data_addr: u64,
    data_width: u32,
    dma_addr: u64,
    dma_as: Option<Arc<dyn DmaMemory>>,
    cfg: &FwCfgMachineConfig,
) -> Result<FwCfgMem> {
    let dma_requested = dma_addr != 0 && dma_as.is_some();
    let props = FwCfgProps { dma_enabled: dma_requested, ..FwCfgProps::default() };
    let mut m = FwCfgMem::new(data_width, props, dma_as, cfg)?;
    m.ctl_addr = ctl_addr;
    m.data_addr = data_addr;
    m.dma_addr = if m.dma.is_some() { dma_addr } else { 0 };
    Ok(m)
}

/// `fw_cfg_init_mem_dma()`: data at `base_addr`, 8 bytes wide, selector at +8, DMA at +16.
pub fn fw_cfg_init_mem_dma(
    base_addr: u64,
    dma_as: Arc<dyn DmaMemory>,
    cfg: &FwCfgMachineConfig,
) -> Result<FwCfgMem> {
    fw_cfg_init_mem_internal(base_addr + 8, base_addr, 8, base_addr + 16, Some(dma_as), cfg)
}

/// `fw_cfg_init_mem_nodma()`.
pub fn fw_cfg_init_mem_nodma(
    ctl_addr: u64,
    data_addr: u64,
    data_width: u32,
    cfg: &FwCfgMachineConfig,
) -> Result<FwCfgMem> {
    fw_cfg_init_mem_internal(ctl_addr, data_addr, data_width, 0, None, cfg)
}
