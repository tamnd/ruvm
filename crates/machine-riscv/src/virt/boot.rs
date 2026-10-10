// SPDX-License-Identifier: GPL-2.0-or-later

//! Booting the virt board: hw/riscv/boot.c (`riscv_find_firmware()`, `riscv_load_firmware()`,
//! `riscv_load_kernel()`, `riscv_load_initrd()`, `riscv_compute_fdt_addr()`,
//! `riscv_load_fdt()`, `riscv_setup_rom_reset_vec()` and `riscv_rom_copy_firmware_info()`),
//! hw/core/generic-loader.c (`-device loader`) and the parts of hw/core/loader.c they need:
//! `load_elf_ram_sym()` for ELF32 and ELF64 of either byte order, `load_image_targphys_as()`
//! and the ROM list with its address spaces, overlap check, reset copy and
//! `rom_find_largest_gap_between()`.
//!
//! Messages QEMU prints and carries on after go to standard error and into the board's
//! message list. Fatal ones are returned as errors without the `qemu-system-riscv64: `
//! prefix.

use std::fmt;
use std::io::Read;
use std::sync::Arc;

use flate2::read::GzDecoder;
use ruvm_base::error::strerror;
use ruvm_base::warn_report;
use ruvm_hw_core::fw_cfg::{
    FW_CFG_CMDLINE_DATA, FW_CFG_CMDLINE_SIZE, FW_CFG_INITRD_DATA, FW_CFG_INITRD_SIZE,
    FW_CFG_KERNEL_DATA, FW_CFG_KERNEL_SIZE, FwCfgState,
};
use ruvm_machine_arm::fdt::Fdt;
use ruvm_mem::{AddressSpace, MemTxAttrs, RamBlock};
use ruvm_qapi::cutils::size_to_str;

/// The name of `address_space_memory`, where the board's own blobs go.
pub const AS_MEMORY: &str = "memory";
/// The name of the first CPU's address space, where `-device loader` puts its blobs.
pub const AS_CPU0: &str = "cpu-memory-0";

/// `RISCV64_BIOS_BIN`, the default `-bios`.
pub const RISCV64_BIOS_BIN: &str = "opensbi-riscv64-generic-fw_dynamic.bin";

/// `EM_RISCV`.
pub(crate) const EM_RISCV: u16 = 243;
/// `PT_LOAD`.
const PT_LOAD: u32 = 1;
/// `PF_X`.
const PF_X: u32 = 1;
/// `ELFDATA2LSB`.
const ELFDATA2LSB: u8 = 1;
/// `ELFCLASS64`.
const ELFCLASS64: u8 = 2;
const MIB: u64 = 1 << 20;
/// `LOAD_IMAGE_MAX_DECOMPRESSED_BYTES`.
const LOAD_IMAGE_MAX_DECOMPRESSED_BYTES: u64 = 256 << 20;

// `ELF_LOAD_*`.
const ELF_LOAD_FAILED: i64 = -1;
const ELF_LOAD_NOT_ELF: i64 = -2;
const ELF_LOAD_WRONG_ARCH: i64 = -3;
const ELF_LOAD_WRONG_ENDIAN: i64 = -4;
const ELF_LOAD_TOO_BIG: i64 = -5;

/// `FW_DYNAMIC_INFO_MAGIC_VALUE`, "OSBI".
const FW_DYNAMIC_INFO_MAGIC_VALUE: u64 = 0x4942_534f;
/// `FW_DYNAMIC_INFO_VERSION`.
const FW_DYNAMIC_INFO_VERSION: u64 = 0x2;
/// `FW_DYNAMIC_INFO_NEXT_MODE_S`.
const FW_DYNAMIC_INFO_NEXT_MODE_S: u64 = 0x1;
/// The size of the reset vector: 6 instructions and two dwords, `sizeof(reset_vec)`.
pub(crate) const RESET_VEC_SIZE: u64 = 40;

/// One blob of the ROM list, `Rom` in hw/core/loader.c: `data` goes to `addr` in the address
/// space `as_name` at every reset and the rest of `romsize` is zeroed.
#[derive(Clone, PartialEq, Eq)]
pub struct Rom {
    /// The name the overlap check prints.
    pub name: String,
    /// The guest physical address.
    pub addr: u64,
    /// The contents, `datasize` bytes.
    pub data: Vec<u8>,
    /// How many bytes the blob covers, at least `data.len()`.
    pub romsize: u64,
    /// The address space, [`AS_MEMORY`] or [`AS_CPU0`].
    pub as_name: &'static str,
}

impl fmt::Debug for Rom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rom")
            .field("name", &self.name)
            .field("addr", &format_args!("{:#x}", self.addr))
            .field("datasize", &self.data.len())
            .field("romsize", &self.romsize)
            .field("as", &self.as_name)
            .finish()
    }
}

impl Rom {
    /// A blob whose `romsize` is its length in `address_space_memory`,
    /// `rom_add_blob_fixed_as()`.
    pub(crate) fn blob(name: &str, addr: u64, data: Vec<u8>) -> Rom {
        Rom { name: name.to_string(), addr, romsize: data.len() as u64, data, as_name: AS_MEMORY }
    }

    fn end(&self) -> u64 {
        self.addr.wrapping_add(self.romsize)
    }
}

/// Where an address space sorts in the ROM list. QEMU compares the `AddressSpace` pointers:
/// `address_space_memory` is a static and comes before the CPU address spaces, which are
/// allocated.
fn as_rank(name: &str) -> u8 {
    u8::from(name != AS_MEMORY)
}

/// `rom_order_compare()`: whether `rom` goes after `item`.
fn rom_order_compare(rom: &Rom, item: &Rom) -> bool {
    let (a, b) = (as_rank(rom.as_name), as_rank(item.as_name));
    a > b || (a == b && rom.addr >= item.addr)
}

/// `rom_insert()`: keep the list sorted by address space and address, a new blob going
/// after older ones at the same address.
pub(crate) fn rom_insert(roms: &mut Vec<Rom>, rom: Rom) {
    let pos = roms.iter().position(|item| !rom_order_compare(&rom, item)).unwrap_or(roms.len());
    roms.insert(pos, rom);
}

/// `rom_check_and_register_reset()`: the overlap report, or `None` if no two neighbours in
/// the same address space overlap.
pub(crate) fn rom_check(roms: &[Rom]) -> Option<String> {
    let mut out = String::new();
    for w in roms.windows(2) {
        let (last, rom) = (&w[0], &w[1]);
        if last.as_name == rom.as_name && last.addr.wrapping_add(last.romsize) > rom.addr {
            if out.is_empty() {
                out.push_str(
                    "Some ROM regions are overlapping\n\
                     These ROM regions might have been loaded by direct user request or by \
                     default.\n\
                     They could be BIOS/firmware images, a guest kernel, initrd or some other \
                     file loaded into guest memory.\n\
                     Check whether you intended to load all this guest code, and whether it \
                     has been built to load to the correct addresses.\n",
                );
            }
            out.push_str(&format!(
                "\nThe following two regions overlap (in the {} address space):\n",
                rom.as_name
            ));
            for r in [last, rom] {
                out.push_str(&format!(
                    "  {} (addresses 0x{:016x} - 0x{:016x})\n",
                    r.name,
                    r.addr,
                    r.end()
                ));
            }
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// A RAM or ROM range of the system address space.
#[derive(Clone, Debug)]
pub struct RamRange {
    /// The guest physical address.
    pub addr: u64,
    /// The size in bytes.
    pub size: u64,
    /// Whether the guest cannot write it (the boot ROM).
    pub readonly: bool,
    /// The backing block.
    pub block: Arc<RamBlock>,
    /// Where `addr` is in `block`.
    pub offset: u64,
    /// A ROM device, a flash: the ROMs are written into it, but it is not RAM.
    pub rom_device: bool,
}

/// Copy `data` to `addr` and zero the following `zeros` bytes, in RAM and ROM only.
fn write_ram(ram: &[RamRange], addr: u64, data: &[u8], zeros: u64) {
    let end = addr.saturating_add(data.len() as u64).saturating_add(zeros);
    let data_end = addr.saturating_add(data.len() as u64);
    for r in ram {
        let r_end = r.addr.saturating_add(r.size);
        let lo = addr.max(r.addr);
        let hi = end.min(r_end);
        if lo >= hi {
            continue;
        }
        let dlo = lo.min(data_end);
        let dhi = hi.min(data_end);
        if dlo < dhi {
            let src = &data[(dlo - addr) as usize..(dhi - addr) as usize];
            let _ = r.block.write(r.offset + (dlo - r.addr), src);
        }
        let zlo = lo.max(data_end);
        if zlo < hi {
            let _ = r.block.fill(r.offset + (zlo - r.addr), hi - zlo, 0);
        }
    }
}

/// `rom_reset()`, with the writes clipped to RAM and ROM. Every address space here is a
/// view of the same memory, so the CPU's blobs land after (on top of) the board's.
pub(crate) fn rom_reset(roms: &[Rom], ram: &[RamRange]) {
    for rom in roms {
        let zeros = rom.romsize.saturating_sub(rom.data.len() as u64);
        write_ram(ram, rom.addr, &rom.data, zeros);
    }
}

/// `rom_find_largest_gap_between(base, size)`: the start and end of the largest stretch of
/// `[base, base + size)` no blob covers.
pub(crate) fn largest_gap(roms: &[Rom], base: u64, size: u64) -> (u64, u64) {
    let top = base.wrapping_add(size);
    let mut secs: Vec<(u64, i32)> = Vec::new();
    for rom in roms {
        // Ignore anything finishing below base or starting above the region.
        if rom.end() <= base || rom.addr >= top {
            continue;
        }
        secs.push((rom.addr, 1));
        if rom.end() < top {
            secs.push((rom.end(), -1));
        }
    }
    // A sentinel at the top of the region.
    secs.push((top, 1));
    secs.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let (mut best_base, mut best_size) = (0u64, 0u64);
    let mut gapstart = base;
    let mut count = 0;
    for (b, se) in secs {
        if count == 0 && count + se == 1 {
            // A blob that starts below `base` makes this wrap, as the `size_t` does in QEMU.
            let gap = b.wrapping_sub(gapstart);
            if gap > best_size {
                best_base = gapstart;
                best_size = gap;
            }
        } else if count == 1 && count + se == 0 {
            gapstart = b;
        }
        count += se;
    }
    (best_base, best_base.wrapping_add(best_size))
}

/// `RISCVBootInfo` after `virt_machine_done()`, and the addresses the reset vector uses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BootInfo {
    /// `start_addr`: where the reset vector jumps, the firmware entry or the base of RAM.
    pub start_addr: u64,
    /// The end of the firmware, the base of RAM without one.
    pub firmware_end: u64,
    /// `kernel_size`, zero without `-kernel`.
    pub kernel_size: u64,
    /// `image_low_addr`.
    pub image_low_addr: u64,
    /// `image_high_addr`.
    pub image_high_addr: u64,
    /// `kernel_entry`: the `next_addr` of the firmware's dynamic info, zero without a kernel.
    pub kernel_entry: u64,
    /// `initrd_start`.
    pub initrd_start: u64,
    /// `initrd_size`.
    pub initrd_size: u64,
    /// Where the device tree went, `fdt_load_addr`.
    pub fdt_addr: u64,
}

/// The loader's output: the ROM list and the messages printed along the way.
#[derive(Debug, Default)]
pub(crate) struct Loader {
    pub(crate) roms: Vec<Rom>,
    pub(crate) messages: Vec<String>,
}

impl Loader {
    /// A message QEMU prints as it is and goes on, such as `perror()`'s.
    fn note(&mut self, msg: String) {
        eprintln!("{msg}");
        self.messages.push(msg);
    }

    /// A `warn_report()`.
    fn warn(&mut self, msg: String) {
        warn_report(&msg);
        self.messages.push(format!("warning: {msg}"));
    }

    pub(crate) fn add(&mut self, rom: Rom) {
        rom_insert(&mut self.roms, rom);
    }
}

/// `load_elf_strerror()`.
pub(crate) fn load_elf_strerror(code: i64) -> &'static str {
    match code {
        0 => "No error",
        -1 => "Failed to load ELF",
        -2 => "The image is not ELF",
        -3 => "The image is from incompatible architecture",
        -4 => "The image has incorrect endianness",
        -5 => "The image segments are too big to load",
        _ => "Unknown error",
    }
}

/// Where the loadable segments of an ELF file go.
pub(crate) enum ElfDest<'a> {
    /// Blobs in the ROM list, in the address space with this name (`load_rom`).
    Rom(&'static str),
    /// Written straight into memory once, with the rest of each segment zeroed.
    Direct(&'a AddressSpace),
}

/// The checks `load_elf_ram_sym()` makes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ElfWant {
    /// `elf_data_order`: `Some(true)` for big-endian only, `Some(false)` for little-endian
    /// only, `None` for either (`ELFDATANONE`).
    pub(crate) big_endian: Option<bool>,
    /// `elf_machine`, 0 for any.
    pub(crate) machine: u16,
}

/// What `load_elf_ram_sym()` loaded: the total size, the entry, the lowest and highest
/// address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ElfLoaded {
    pub(crate) size: i64,
    pub(crate) entry: u64,
    pub(crate) low: u64,
    pub(crate) high: u64,
}

/// An ELF program header.
#[derive(Clone, Copy, Debug)]
struct Phdr {
    p_type: u32,
    p_flags: u32,
    p_offset: u64,
    p_vaddr: u64,
    p_paddr: u64,
    p_filesz: u64,
    p_memsz: u64,
}

/// Reads fields of the file in its byte order.
struct ElfReader<'a> {
    f: &'a [u8],
    big: bool,
}

impl ElfReader<'_> {
    fn u16(&self, o: usize) -> Option<u16> {
        let b: [u8; 2] = self.f.get(o..o.checked_add(2)?)?.try_into().ok()?;
        Some(if self.big { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }

    fn u32(&self, o: usize) -> Option<u32> {
        let b: [u8; 4] = self.f.get(o..o.checked_add(4)?)?.try_into().ok()?;
        Some(if self.big { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
    }

    fn u64(&self, o: usize) -> Option<u64> {
        let b: [u8; 8] = self.f.get(o..o.checked_add(8)?)?.try_into().ok()?;
        Some(if self.big { u64::from_be_bytes(b) } else { u64::from_le_bytes(b) })
    }
}

/// `load_elf_ram_sym()`: the file `name` with contents `file` (`None` if it could not be
/// opened, which `perror()`s). Returns the size or an `ELF_LOAD_*` code.
pub(crate) fn load_elf(
    ld: &mut Loader,
    name: &str,
    file: Result<&[u8], &std::io::Error>,
    want: ElfWant,
    dest: ElfDest<'_>,
) -> Result<ElfLoaded, i64> {
    let f = match file {
        Ok(f) => f,
        Err(e) => {
            ld.note(format!("{name}: {}", strerror(e)));
            return Err(ELF_LOAD_FAILED);
        }
    };
    if f.len() < 16 {
        return Err(ELF_LOAD_FAILED);
    }
    if &f[..4] != b"\x7fELF" {
        return Err(ELF_LOAD_NOT_ELF);
    }
    // The host is little-endian: anything else is swapped as big-endian.
    let big = f[5] != ELFDATA2LSB;
    if let Some(b) = want.big_endian {
        let order = if b { 2 } else { ELFDATA2LSB };
        if f[5] != order {
            return Err(ELF_LOAD_WRONG_ENDIAN);
        }
    }
    let is64 = f[4] == ELFCLASS64;
    let r = ElfReader { f, big };
    let fail = ELF_LOAD_FAILED;
    // Read the header; a short one fails.
    let ehdr_size = if is64 { 64 } else { 52 };
    if f.len() < ehdr_size {
        return Err(fail);
    }
    let e_machine = r.u16(18).ok_or(fail)?;
    let machine = if want.machine == 0 { e_machine } else { want.machine };
    if machine != e_machine {
        return Err(ELF_LOAD_WRONG_ARCH);
    }
    let (e_entry, e_phoff, e_phnum, phentsize) = if is64 {
        (r.u64(24).ok_or(fail)?, r.u64(32).ok_or(fail)?, r.u16(56).ok_or(fail)?, 56usize)
    } else {
        (
            u64::from(r.u32(24).ok_or(fail)?),
            u64::from(r.u32(28).ok_or(fail)?),
            r.u16(44).ok_or(fail)?,
            32usize,
        )
    };
    // g_malloc0(0) gives NULL, which fails.
    if e_phnum == 0 {
        return Err(fail);
    }
    let mut ph = Vec::with_capacity(usize::from(e_phnum));
    for i in 0..usize::from(e_phnum) {
        let o =
            usize::try_from(e_phoff).ok().and_then(|p| p.checked_add(i * phentsize)).ok_or(fail)?;
        let p = if is64 {
            Phdr {
                p_type: r.u32(o).ok_or(fail)?,
                p_flags: r.u32(o + 4).ok_or(fail)?,
                p_offset: r.u64(o + 8).ok_or(fail)?,
                p_vaddr: r.u64(o + 16).ok_or(fail)?,
                p_paddr: r.u64(o + 24).ok_or(fail)?,
                p_filesz: r.u64(o + 32).ok_or(fail)?,
                p_memsz: r.u64(o + 40).ok_or(fail)?,
            }
        } else {
            let w = |d: usize| r.u32(o + d).map(u64::from).ok_or(fail);
            Phdr {
                p_type: r.u32(o).ok_or(fail)?,
                p_offset: w(4)?,
                p_vaddr: w(8)?,
                p_paddr: w(12)?,
                p_filesz: w(16)?,
                p_memsz: w(20)?,
                p_flags: r.u32(o + 24).ok_or(fail)?,
            }
        };
        ph.push(p);
    }
    // `elf_word` arithmetic: 32 bits wide in an ELF32 file.
    let word = |v: u64| if is64 { v } else { v & 0xffff_ffff };
    let mut entry = e_entry;
    let mut total: i64 = 0;
    let (mut low, mut high) = (u64::MAX, 0u64);
    for (i, p) in ph.iter().enumerate() {
        if p.p_type != PT_LOAD {
            continue;
        }
        let file_size = p.p_filesz;
        let mut mem_size = p.p_memsz;
        let mut data: &[u8] = &[];
        if file_size > 0 {
            let end = file_size.checked_add(p.p_offset).ok_or(fail)?;
            if (f.len() as u64) < end {
                return Err(fail);
            }
            data = &f[p.p_offset as usize..end as usize];
        }
        // A segment whose zero-initialised part overlaps another segment is loaded with only
        // its file size: its memsz is the runtime size.
        if mem_size > file_size {
            let zero_start = word(p.p_paddr.wrapping_add(file_size));
            let zero_end = word(p.p_paddr.wrapping_add(mem_size));
            let overlaps = ph.iter().enumerate().any(|(j, q)| {
                let other_start = q.p_paddr;
                let other_end = word(q.p_paddr.wrapping_add(q.p_memsz));
                i != j
                    && q.p_type == PT_LOAD
                    && !(other_start >= zero_end || zero_start >= other_end)
            });
            if overlaps {
                mem_size = file_size;
            }
        }
        if mem_size > (i64::MAX - total) as u64 {
            return Err(ELF_LOAD_TOO_BIG);
        }
        let addr = p.p_paddr;
        // The entry point in the header is a virtual address.
        if p.p_vaddr != p.p_paddr
            && e_entry >= p.p_vaddr
            && e_entry < word(p.p_vaddr.wrapping_add(p.p_filesz))
            && p.p_flags & PF_X != 0
        {
            entry = word(e_entry.wrapping_sub(p.p_vaddr).wrapping_add(p.p_paddr));
        }
        // Zero sized segments make no blob, which could trip the overlap check.
        if mem_size != 0 {
            match &dest {
                ElfDest::Rom(as_name) => ld.add(Rom {
                    name: format!("{name} ELF program header segment {i}"),
                    addr,
                    data: data.to_vec(),
                    romsize: mem_size,
                    as_name,
                }),
                ElfDest::Direct(a) => {
                    if !a.write(addr, MemTxAttrs::UNSPECIFIED, data).is_ok() {
                        return Err(fail);
                    }
                    if file_size < mem_size {
                        let zeros = vec![0u8; usize::try_from(mem_size - file_size).or(Err(fail))?];
                        let at = addr.wrapping_add(file_size);
                        if !a.write(at, MemTxAttrs::UNSPECIFIED, &zeros).is_ok() {
                            return Err(fail);
                        }
                    }
                }
            }
        }
        total += mem_size as i64;
        low = low.min(addr);
        high = high.max(addr.wrapping_add(mem_size));
    }
    Ok(ElfLoaded { size: total, entry, low, high })
}

/// `load_image_targphys_as()`: `file` as one blob at `addr` in the address space `as_name`.
/// Returns the size, or the message QEMU sets in `errp`.
pub(crate) fn load_image_targphys_as(
    ld: &mut Loader,
    filename: &str,
    file: Result<&[u8], &std::io::Error>,
    addr: u64,
    max_sz: u64,
    as_name: &'static str,
) -> Result<u64, String> {
    let data = file.map_err(|e| format!("Could not open '{filename}': {}", strerror(e)))?;
    if data.is_empty() {
        return Err(format!("empty file: {filename}"));
    }
    let size = data.len() as u64;
    if size > max_sz {
        return Err(format!("{filename} exceeds maximum image size ({})", size_to_str(max_sz)));
    }
    ld.add(Rom { name: filename.to_string(), addr, data: data.to_vec(), romsize: size, as_name });
    Ok(size)
}

/// `load_image_gzipped_buffer()` on the contents `data` of `filename`: the inflated image, or
/// `None` when it is not gzip or does not inflate to at most 256 MiB.
fn load_gzipped(ld: &mut Loader, filename: &str, data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return None;
    }
    let mut out = Vec::new();
    let max = LOAD_IMAGE_MAX_DECOMPRESSED_BYTES;
    match GzDecoder::new(data).take(max + 1).read_to_end(&mut out) {
        Ok(n) if n as u64 <= max => Some(out),
        _ => {
            ld.note(format!("{filename}: unable to decompress gzipped kernel file"));
            None
        }
    }
}

/// `load_image_to_fw_cfg()`: `filename` as the fw_cfg items `size_key` and `data_key`,
/// inflated first if `try_decompress` is set and it is gzip.
fn load_image_to_fw_cfg(
    ld: &mut Loader,
    fw_cfg: &FwCfgState,
    keys: (u16, u16),
    filename: Option<&str>,
    try_decompress: bool,
) -> Result<(), String> {
    let Some(filename) = filename else {
        return Ok(());
    };
    let raw = std::fs::read(filename).map_err(|_| format!("failed to load \"{filename}\""))?;
    let data = if try_decompress { load_gzipped(ld, filename, &raw) } else { None };
    let data = data.unwrap_or(raw);
    fw_cfg.add_i32(keys.0, data.len() as u32);
    fw_cfg.add_bytes(keys.1, data);
    Ok(())
}

/// `riscv_setup_firmware_boot()`: hand the kernel, the initrd and the command line to the
/// firmware in the flash through fw_cfg, untouched apart from inflating a gzip kernel.
pub(crate) fn riscv_setup_firmware_boot(
    ld: &mut Loader,
    fw_cfg: &FwCfgState,
    files: KernelFiles<'_>,
) -> Result<(), String> {
    let kernel_keys = (FW_CFG_KERNEL_SIZE, FW_CFG_KERNEL_DATA);
    load_image_to_fw_cfg(ld, fw_cfg, kernel_keys, Some(files.kernel), true)?;
    let initrd_keys = (FW_CFG_INITRD_SIZE, FW_CFG_INITRD_DATA);
    load_image_to_fw_cfg(ld, fw_cfg, initrd_keys, files.initrd, false)?;
    if let Some(cmdline) = files.cmdline {
        fw_cfg.add_i32(FW_CFG_CMDLINE_SIZE, cmdline.len() as u32 + 1);
        fw_cfg.add_string(FW_CFG_CMDLINE_DATA, cmdline);
    }
    Ok(())
}

/// `riscv_find_firmware()`: the file `-bios` names, through `find` (`qemu_find_file()` with
/// `QEMU_FILE_TYPE_BIOS`). `None` or `default` is the OpenSBI build QEMU ships and `none`
/// is no firmware. Under the qtest accelerator a missing file means no firmware, as in
/// `riscv_find_bios()`.
pub fn riscv_find_firmware(
    bios: Option<&str>,
    qtest_enabled: bool,
    find: impl Fn(&str) -> Option<String>,
) -> Result<Option<String>, String> {
    let name = match bios {
        None | Some("default") => RISCV64_BIOS_BIN,
        Some("none") => return Ok(None),
        Some(n) => n,
    };
    match find(name) {
        Some(p) => Ok(Some(p)),
        None if qtest_enabled => Ok(None),
        None => Err(format!("Unable to find the RISC-V BIOS \"{name}\"")),
    }
}

/// `riscv_load_firmware()`: an ELF (written straight into memory) or a raw image at
/// `*load_addr`. Returns the end of the firmware and sets `*load_addr` to the ELF entry.
pub(crate) fn riscv_load_firmware(
    ld: &mut Loader,
    memory: &AddressSpace,
    filename: &str,
    load_addr: &mut u64,
    mem_size: u64,
) -> Result<u64, String> {
    let file = std::fs::read(filename);
    let want = ElfWant { big_endian: None, machine: EM_RISCV };
    let size = match load_elf(ld, filename, file.as_deref(), want, ElfDest::Direct(memory)) {
        Ok(l) if l.size > 0 => {
            *load_addr = l.entry;
            return Ok(l.high);
        }
        Ok(l) => l.size,
        Err(code) => code,
    };
    if size != ELF_LOAD_NOT_ELF {
        // If the user specified an ELF format firmware that could not be loaded as an
        // ELF, it's possible that loading it as a binary is not what was intended.
        ld.warn(format!(
            "could not load ELF format firmware '{filename}' ({}). Attempting to load as \
             binary.",
            load_elf_strerror(size)
        ));
    }
    match load_image_targphys_as(
        ld,
        filename,
        file.as_ref().map(Vec::as_slice),
        *load_addr,
        mem_size,
        AS_MEMORY,
    ) {
        Ok(n) => Ok(*load_addr + n),
        Err(_) => Err(format!(
            "could not load firmware '{filename}': {}",
            load_elf_strerror(ELF_LOAD_FAILED)
        )),
    }
}

/// `riscv_calc_kernel_start_addr()` for RV64: the firmware end rounded up to 2 MiB.
pub(crate) fn riscv_calc_kernel_start_addr(firmware_end: u64) -> u64 {
    firmware_end.div_ceil(2 * MIB).wrapping_mul(2 * MIB)
}

/// The `-kernel`, `-initrd` and `-append` options.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct KernelFiles<'a> {
    pub(crate) kernel: &'a str,
    pub(crate) initrd: Option<&'a str>,
    pub(crate) cmdline: Option<&'a str>,
}

/// `riscv_load_kernel()` with `riscv_load_initrd()`: an RV64 little-endian ELF or a raw
/// image at `kernel_start`, then the initrd, then `/chosen` in `fdt` (when there is one).
pub(crate) fn riscv_load_kernel(
    ld: &mut Loader,
    info: &mut BootInfo,
    mut fdt: Option<&mut Fdt>,
    files: KernelFiles<'_>,
    kernel_start: u64,
    mem_size: u64,
) -> Result<(), String> {
    let name = files.kernel;
    let file = std::fs::read(name);
    let want = ElfWant { big_endian: Some(false), machine: EM_RISCV };
    let mut loaded = false;
    if let Ok(l) = load_elf(ld, name, file.as_deref(), want, ElfDest::Rom(AS_MEMORY)) {
        if l.size > 0 {
            info.kernel_size = l.size as u64;
            info.image_low_addr = l.low;
            info.image_high_addr = l.high;
            loaded = true;
        }
    }
    // uImage is not modelled; then the raw image.
    if !loaded {
        let res = load_image_targphys_as(
            ld,
            name,
            file.as_ref().map(Vec::as_slice),
            kernel_start,
            mem_size,
            AS_MEMORY,
        );
        match res {
            Ok(n) => {
                info.kernel_size = n;
                info.image_low_addr = kernel_start;
                info.image_high_addr = kernel_start.wrapping_add(n);
            }
            Err(_) => return Err(format!("could not load kernel '{name}'")),
        }
    }

    if let Some(initrd) = files.initrd {
        // Far enough into RAM that the kernel does not clobber it when it uncompresses:
        // halfway into RAM below 1 GiB of RAM, else at 512 MiB.
        let start = info.image_low_addr.wrapping_add((mem_size / 2).min(512 * MIB));
        // load_ramdisk() (a u-boot ramdisk) is not modelled.
        let data = std::fs::read(initrd);
        let size = load_image_targphys_as(
            ld,
            initrd,
            data.as_ref().map(Vec::as_slice),
            start,
            mem_size.wrapping_sub(start),
            AS_MEMORY,
        )
        .map_err(|_| format!("could not load ramdisk '{initrd}'"))?;
        info.initrd_start = start;
        info.initrd_size = size;
        if let Some(fdt) = fdt.as_deref_mut() {
            fdt.setprop_u64("/chosen", "linux,initrd-start", start)?;
            fdt.setprop_u64("/chosen", "linux,initrd-end", start + size)?;
        }
    }

    if let (Some(fdt), Some(cmdline)) = (fdt, files.cmdline) {
        if !cmdline.is_empty() {
            fdt.setprop_string("/chosen", "bootargs", cmdline)?;
        }
    }
    Ok(())
}

/// `riscv_compute_fdt_addr()` for RV64 with the RAM at `dram_base`: the packed tree goes
/// as high in RAM as it fits, on a 2 MiB boundary, above the kernel and initrd.
pub(crate) fn riscv_compute_fdt_addr(
    dram_base: u64,
    ram_size: u64,
    fdtsize: u64,
    info: &BootInfo,
) -> Result<u64, String> {
    if fdtsize == 0 {
        return Err("invalid device-tree".to_string());
    }
    let limit = if info.initrd_size != 0 {
        // If initrd is successfully loaded, place DTB after it.
        info.initrd_start + info.initrd_size
    } else if info.kernel_size != 0 {
        // If only kernel is successfully loaded, place DTB after it.
        info.image_high_addr
    } else {
        // Otherwise, do not check DTB overlapping.
        0
    };
    let dram_end = dram_base.wrapping_add(ram_size);
    let dtb_start = dram_end.wrapping_sub(fdtsize) & !(2 * MIB - 1);
    if limit != 0 && dtb_start < limit {
        return Err("Not enough memory to place DTB after kernel/initrd".to_string());
    }
    Ok(dtb_start)
}

/// `riscv_setup_rom_reset_vec()` for a little-endian RV64 hart with Zicsr: the code that
/// loads `a0` with `mhartid`, `a1` with the device tree and `a2` with the dynamic info and
/// jumps to `start_addr`.
pub(crate) fn reset_vec(start_addr: u64, fdt_addr: u64) -> Vec<u8> {
    let code: [u32; 6] = [
        0x0000_0297, // 1:  auipc  t0, %pcrel_hi(fw_dyn)
        0x0282_8613, //     addi   a2, t0, %pcrel_lo(1b)
        0xf140_2573, //     csrr   a0, mhartid
        0x0202_b583, //     ld     a1, 32(t0)
        0x0182_b283, //     ld     t0, 24(t0)
        0x0002_8067, //     jr     t0
    ];
    let mut v: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    v.extend_from_slice(&start_addr.to_le_bytes()); // start: .dword
    v.extend_from_slice(&fdt_addr.to_le_bytes()); // fdt_laddr: .dword
    v
}

/// `riscv_rom_copy_firmware_info()` for RV64: `struct fw_dynamic_info64`.
pub(crate) fn firmware_info(
    rom_size: u64,
    reset_vec_size: u64,
    kernel_entry: u64,
) -> Result<Vec<u8>, String> {
    let words = [
        FW_DYNAMIC_INFO_MAGIC_VALUE,
        FW_DYNAMIC_INFO_VERSION,
        kernel_entry,
        FW_DYNAMIC_INFO_NEXT_MODE_S,
        0,
        0,
    ];
    let v: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    // The copy of the dynamic firmware info must fit in the boot ROM.
    if v.len() as u64 > rom_size - reset_vec_size {
        return Err("not enough space to store dynamic firmware info".to_string());
    }
    Ok(v)
}

/// `load_device_tree()`: the blob in a buffer twice its size plus 10000.
pub(crate) fn load_device_tree(ld: &mut Loader, filename: &str) -> Option<Fdt> {
    let Ok(data) = std::fs::read(filename) else {
        ld.note(format!("Unable to get size of device tree file '{filename}'"));
        return None;
    };
    if data.len() > (i32::MAX as usize) / 2 - 10000 {
        ld.note(format!("Device tree file '{filename}' is too large"));
        return None;
    }
    let dt_size = (data.len() + 10000) * 2;
    let fdt = match Fdt::open_into(&data, dt_size) {
        Ok(f) => f,
        Err(_) => {
            ld.note(format!("load_device_tree: Unable to open device tree file '{filename}'"));
            return None;
        }
    };
    if fdt.check_header().is_err() {
        ld.note(format!("Device tree file loaded into memory is invalid: {filename}"));
        return None;
    }
    Some(fdt)
}

/// `CPU_NONE`, the default `cpu-num` of `-device loader`.
const CPU_NONE: u32 = 0xffff_ffff;

/// The properties of a `-device loader`, hw/core/generic-loader.c.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenericLoader {
    /// `file`.
    pub file: Option<String>,
    /// `addr`.
    pub addr: u64,
    /// `data`.
    pub data: u64,
    /// `data-len`.
    pub data_len: u8,
    /// `data-be`.
    pub data_be: bool,
    /// `cpu-num`, `None` for the default `CPU_NONE`.
    pub cpu_num: Option<u32>,
    /// `force-raw`.
    pub force_raw: bool,
}

/// A realized `-device loader`: what it does at every reset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LoaderReset {
    /// The CPU it acts on, `first_cpu` without `cpu-num`.
    pub(crate) cpu: usize,
    /// `set_pc`: reset the CPU and point it at `addr`.
    pub(crate) set_pc: bool,
    /// `addr`, the ELF entry once an ELF is loaded.
    pub(crate) addr: u64,
    /// The bytes to write at `addr`, the first `data-len` bytes of `data` as it sits in
    /// memory after the byte swap.
    pub(crate) data: Vec<u8>,
}

/// `generic_loader_realize()`: check the options and load the file into the CPU's
/// address space. `smp` is the number of CPUs and `ram_size` limits a raw image.
pub(crate) fn generic_loader_realize(
    ld: &mut Loader,
    s: &GenericLoader,
    smp: usize,
    ram_size: u64,
) -> Result<LoaderReset, String> {
    let cpu_num = s.cpu_num.unwrap_or(CPU_NONE);
    let mut set_pc = false;
    // Perform some error checking on the user's options.
    if s.data != 0 || s.data_len != 0 || s.data_be {
        // User is loading memory values.
        if s.file.is_some() {
            return Err("Specifying a file is not supported when loading memory values".to_string());
        } else if s.force_raw {
            return Err(
                "Specifying force-raw is not supported when loading memory values".to_string()
            );
        } else if s.data_len == 0 {
            // We can't check for !data here as a value of 0 is still valid.
            return Err("Both data and data-len must be specified".to_string());
        } else if s.data_len > 8 {
            return Err("data-len cannot be greater then 8 bytes".to_string());
        }
    } else if s.file.is_some() || s.force_raw {
        // User is loading an image: only set the PC if they also specified a CPU to use.
        set_pc = cpu_num != CPU_NONE;
    } else if s.addr != 0 {
        // User is setting the PC.
        if cpu_num == CPU_NONE {
            return Err("cpu_num must be specified when setting a program counter".to_string());
        }
        set_pc = true;
    } else {
        // Did the user specify anything?
        return Err("please include valid arguments".to_string());
    }

    let cpu = if cpu_num != CPU_NONE {
        if cpu_num as usize >= smp {
            return Err(format!("Specified boot CPU#{} is nonexistent", cpu_num as i32));
        }
        cpu_num as usize
    } else {
        0
    };

    let mut addr = s.addr;
    if let Some(name) = &s.file {
        let file = std::fs::read(name);
        let mut size: i64 = -1;
        if !s.force_raw {
            let want = ElfWant { big_endian: None, machine: 0 };
            match load_elf(ld, name, file.as_deref(), want, ElfDest::Rom(AS_CPU0)) {
                Ok(l) => {
                    size = l.size;
                    addr = l.entry;
                }
                Err(code) => size = code,
            }
            // uImage and Intel HEX are not modelled.
        }
        if size < 0 || s.force_raw {
            // Default to the maximum size being the machine's ram size.
            load_image_targphys_as(
                ld,
                name,
                file.as_ref().map(Vec::as_slice),
                s.addr,
                ram_size,
                AS_CPU0,
            )
            .map_err(|e| format!("Cannot load specified image {name}: {e}"))?;
            addr = s.addr;
        }
    }

    // Convert the data endianness.
    let bytes = if s.data_be { s.data.to_be_bytes() } else { s.data.to_le_bytes() };
    let data = bytes[..usize::from(s.data_len.min(8))].to_vec();
    Ok(LoaderReset { cpu, set_pc, addr, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rom(name: &str, addr: u64, romsize: u64, as_name: &'static str) -> Rom {
        Rom { name: name.to_string(), addr, data: Vec::new(), romsize, as_name }
    }

    #[test]
    fn roms_sort_by_address_space_then_address() {
        let mut roms = Vec::new();
        rom_insert(&mut roms, rom("elf", 0x8000_0000, 0x100, AS_CPU0));
        rom_insert(&mut roms, rom("fw", 0x8000_0000, 0x4_0000, AS_MEMORY));
        rom_insert(&mut roms, rom("mrom", 0x1000, 0x28, AS_MEMORY));
        rom_insert(&mut roms, rom("fw2", 0x8000_0000, 1, AS_MEMORY));
        let names: Vec<_> = roms.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["mrom", "fw", "fw2", "elf"]);
    }

    #[test]
    fn overlap_only_within_an_address_space() {
        let roms =
            [rom("fw", 0x8000_0000, 0x4_0000, AS_MEMORY), rom("elf", 0x8000_0000, 0x10, AS_CPU0)];
        assert!(rom_check(&roms).is_none());
        let roms = [rom("a", 0x1000, 0x100, AS_MEMORY), rom("b", 0x1080, 0x10, AS_MEMORY)];
        let msg = rom_check(&roms).unwrap();
        assert!(msg.ends_with(
            "\nThe following two regions overlap (in the memory address space):\n  \
             a (addresses 0x0000000000001000 - 0x0000000000001100)\n  \
             b (addresses 0x0000000000001080 - 0x0000000000001090)\n"
        ));
    }

    #[test]
    fn largest_gap_with_a_blob_below_the_base() {
        let roms = [rom("elf", 0x7fff_f000, 0x2000, AS_MEMORY)];
        // QEMU's size_t gap wraps, so the whole space from the base up wins.
        let (lo, hi) = largest_gap(&roms, 0x8000_0000, 0x800_0000);
        assert_eq!((lo, hi), (0x8000_0000, 0x7fff_f000));
        let roms = [rom("fw", 0x8000_0000, 0x4_0000, AS_MEMORY)];
        assert_eq!(largest_gap(&roms, 0x8000_0000, 0x800_0000), (0x8004_0000, 0x8800_0000));
    }

    #[test]
    fn reset_vector_words() {
        let v = reset_vec(0x8000_0000, 0x87e0_0000);
        assert_eq!(v.len() as u64, RESET_VEC_SIZE);
        assert_eq!(&v[..4], &0x0000_0297u32.to_le_bytes());
        assert_eq!(&v[24..32], &0x8000_0000u64.to_le_bytes());
        assert_eq!(&v[32..40], &0x87e0_0000u64.to_le_bytes());
        let fi = firmware_info(0xf000, RESET_VEC_SIZE, 0x8020_0000).unwrap();
        assert_eq!(fi.len(), 48);
        assert_eq!(&fi[..8], &0x4942_534fu64.to_le_bytes());
        assert_eq!(&fi[16..24], &0x8020_0000u64.to_le_bytes());
        assert_eq!(&fi[24..32], &1u64.to_le_bytes());
        assert_eq!(
            firmware_info(0x40, RESET_VEC_SIZE, 0).unwrap_err(),
            "not enough space to store dynamic firmware info"
        );
    }

    #[test]
    fn fdt_placement() {
        let none = BootInfo::default();
        assert_eq!(riscv_compute_fdt_addr(0x8000_0000, 128 << 20, 0x1000, &none), Ok(0x87e0_0000));
        let kernel =
            BootInfo { kernel_size: 1, image_high_addr: 0x87f0_0000, ..BootInfo::default() };
        assert_eq!(
            riscv_compute_fdt_addr(0x8000_0000, 128 << 20, 0x1000, &kernel).unwrap_err(),
            "Not enough memory to place DTB after kernel/initrd"
        );
        assert_eq!(riscv_calc_kernel_start_addr(0x8004_2000), 0x8020_0000);
        assert_eq!(riscv_calc_kernel_start_addr(0x8000_0000), 0x8000_0000);
    }

    #[test]
    fn find_firmware() {
        let find = |n: &str| (n == RISCV64_BIOS_BIN).then(|| format!("/fw/{n}"));
        assert_eq!(
            riscv_find_firmware(None, false, find).unwrap().unwrap(),
            format!("/fw/{RISCV64_BIOS_BIN}")
        );
        assert_eq!(riscv_find_firmware(Some("none"), false, find), Ok(None));
        assert_eq!(
            riscv_find_firmware(Some("x.bin"), false, find).unwrap_err(),
            "Unable to find the RISC-V BIOS \"x.bin\""
        );
        assert_eq!(riscv_find_firmware(Some("x.bin"), true, find), Ok(None));
    }

    /// A little ELF64 with one PT_LOAD of `code` at `paddr`.
    fn elf64(paddr: u64, entry: u64, code: &[u8], memsz: u64) -> Vec<u8> {
        let mut f = vec![0u8; 64 + 56];
        f[..4].copy_from_slice(b"\x7fELF");
        f[4] = 2;
        f[5] = 1;
        f[6] = 1;
        f[18..20].copy_from_slice(&EM_RISCV.to_le_bytes());
        f[24..32].copy_from_slice(&entry.to_le_bytes());
        f[32..40].copy_from_slice(&64u64.to_le_bytes());
        f[54..56].copy_from_slice(&56u16.to_le_bytes());
        f[56..58].copy_from_slice(&1u16.to_le_bytes());
        let p = 64;
        f[p..p + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        f[p + 4..p + 8].copy_from_slice(&5u32.to_le_bytes());
        f[p + 8..p + 16].copy_from_slice(&120u64.to_le_bytes());
        f[p + 16..p + 24].copy_from_slice(&paddr.to_le_bytes());
        f[p + 24..p + 32].copy_from_slice(&paddr.to_le_bytes());
        f[p + 32..p + 40].copy_from_slice(&(code.len() as u64).to_le_bytes());
        f[p + 40..p + 48].copy_from_slice(&memsz.to_le_bytes());
        f.extend_from_slice(code);
        f
    }

    #[test]
    fn elf_loading() {
        let f = elf64(0x8000_0000, 0x8000_0004, &[1, 2, 3, 4, 5, 6, 7, 8], 16);
        let mut ld = Loader::default();
        let want = ElfWant { big_endian: Some(false), machine: EM_RISCV };
        let l = load_elf(&mut ld, "t", Ok(&f), want, ElfDest::Rom(AS_CPU0)).unwrap();
        assert_eq!(
            l,
            ElfLoaded { size: 16, entry: 0x8000_0004, low: 0x8000_0000, high: 0x8000_0010 }
        );
        assert_eq!(ld.roms[0].name, "t ELF program header segment 0");
        assert_eq!(ld.roms[0].as_name, AS_CPU0);
        assert_eq!(ld.roms[0].romsize, 16);
        let other = ElfWant { big_endian: None, machine: 183 };
        assert_eq!(load_elf(&mut ld, "t", Ok(&f), other, ElfDest::Rom(AS_MEMORY)), Err(-3));
        let be = ElfWant { big_endian: Some(true), machine: 0 };
        assert_eq!(load_elf(&mut ld, "t", Ok(&f), be, ElfDest::Rom(AS_MEMORY)), Err(-4));
        assert_eq!(
            load_elf(&mut ld, "t", Ok(b"not an elf file!"), want, ElfDest::Rom(AS_MEMORY)),
            Err(-2)
        );
        assert_eq!(load_elf(&mut ld, "t", Ok(b"short"), want, ElfDest::Rom(AS_MEMORY)), Err(-1));
    }

    #[test]
    fn generic_loader_checks() {
        let mut ld = Loader::default();
        let mut s = GenericLoader { data: 1, ..GenericLoader::default() };
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap_err(),
            "Both data and data-len must be specified"
        );
        s.data_len = 9;
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap_err(),
            "data-len cannot be greater then 8 bytes"
        );
        s.file = Some("x".to_string());
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap_err(),
            "Specifying a file is not supported when loading memory values"
        );
        let s = GenericLoader { addr: 0x8000_0000, ..GenericLoader::default() };
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap_err(),
            "cpu_num must be specified when setting a program counter"
        );
        let s = GenericLoader { addr: 0x8000_0000, cpu_num: Some(2), ..GenericLoader::default() };
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 2, 1 << 20).unwrap_err(),
            "Specified boot CPU#2 is nonexistent"
        );
        assert_eq!(
            generic_loader_realize(&mut ld, &GenericLoader::default(), 1, 1 << 20).unwrap_err(),
            "please include valid arguments"
        );
        let s = GenericLoader {
            addr: 0x8000_0000,
            data: 0x1122_3344,
            data_len: 2,
            data_be: true,
            ..GenericLoader::default()
        };
        let r = generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap();
        assert_eq!(r.data, [0, 0]);
        let s = GenericLoader { data_be: false, ..s };
        assert_eq!(generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap().data, [0x44, 0x33]);
        let s = GenericLoader {
            file: Some("/nonexistent/ruvm".to_string()),
            ..GenericLoader::default()
        };
        assert_eq!(
            generic_loader_realize(&mut ld, &s, 1, 1 << 20).unwrap_err(),
            "Cannot load specified image /nonexistent/ruvm: Could not open '/nonexistent/ruvm': No \
             such file or directory"
        );
    }
}
