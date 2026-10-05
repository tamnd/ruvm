// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct kernel boot for the virt board: hw/arm/boot.c (`arm_load_kernel()`,
//! `arm_load_elf()`, `load_aarch64_image()`, `arm_load_dtb()`) and the parts of
//! hw/core/loader.c it needs (`gunzip()`, `unpack_efi_zboot_image()`, the ELF64 loader and the
//! ROM list with its overlap check, reset copy and `rom_find_largest_gap_between()`).
//!
//! Messages QEMU prints and carries on after (a gzip error, then a fallback to the raw file)
//! go to standard error and into the board's message list. Fatal ones are returned as errors
//! without the `qemu-system-aarch64: ` prefix.

use std::fmt;
use std::sync::Arc;

use flate2::{Decompress, FlushDecompress, Status};
use ruvm_mem::RamBlock;
use ruvm_target_arm::tcg::PsciConduit;

use crate::fdt::{Fdt, sized_cells};

/// `KERNEL64_LOAD_ADDR`: where an arm64 Image without a header goes, from the base of RAM.
pub(crate) const KERNEL64_LOAD_ADDR: u64 = 0x0008_0000;
/// `BOOTLOADER_MAX_SIZE`.
const BOOTLOADER_MAX_SIZE: u64 = 4 * 1024;
/// `LOAD_IMAGE_MAX_DECOMPRESSED_BYTES`.
const LOAD_IMAGE_MAX_DECOMPRESSED_BYTES: usize = 256 << 20;
/// `TARGET_PAGE_SIZE` for Arm.
const TARGET_PAGE_SIZE: u64 = 4096;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// `ARM64_MAGIC_OFFSET`.
const ARM64_MAGIC_OFFSET: usize = 56;
/// `ARM64_TEXT_OFFSET_OFFSET`.
const ARM64_TEXT_OFFSET_OFFSET: usize = 8;
/// `EM_AARCH64`.
const EM_AARCH64: u16 = 183;
/// `PT_LOAD`.
const PT_LOAD: u32 = 1;
/// `PF_X`.
const PF_X: u32 = 1;

// `ELF_LOAD_*`.
const ELF_LOAD_FAILED: i64 = -1;
const ELF_LOAD_WRONG_ARCH: i64 = -3;
const ELF_LOAD_TOO_BIG: i64 = -5;

/// The PSCI function IDs `fdt_add_psci_node()` puts in the tree for an AArch64 CPU.
const QEMU_PSCI_0_2_FN64_CPU_SUSPEND: u32 = 0xc400_0001;
const QEMU_PSCI_0_2_FN_CPU_OFF: u32 = 0x8400_0002;
const QEMU_PSCI_0_2_FN64_CPU_ON: u32 = 0xc400_0003;
const QEMU_PSCI_0_2_FN64_MIGRATE: u32 = 0xc400_0005;

/// One blob of the ROM list, `Rom` in hw/core/loader.c: `data` goes to `addr` at every reset
/// and the rest of `romsize` is zeroed.
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
}

impl fmt::Debug for Rom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rom")
            .field("name", &self.name)
            .field("addr", &format_args!("{:#x}", self.addr))
            .field("datasize", &self.data.len())
            .field("romsize", &self.romsize)
            .finish()
    }
}

impl Rom {
    /// A blob whose `romsize` is its length, `rom_add_blob_fixed_as()`.
    pub(crate) fn blob(name: &str, addr: u64, data: Vec<u8>) -> Rom {
        Rom { name: name.to_string(), addr, romsize: data.len() as u64, data }
    }

    fn end(&self) -> u64 {
        self.addr.wrapping_add(self.romsize)
    }
}

/// `rom_insert()`: keep the list sorted by address, a new blob going after older ones at the
/// same address.
pub(crate) fn rom_insert(roms: &mut Vec<Rom>, rom: Rom) {
    let pos = roms.iter().position(|r| rom.addr < r.addr).unwrap_or(roms.len());
    roms.insert(pos, rom);
}

/// `rom_check_and_register_reset()`: the overlap report, or `None` if no two neighbours
/// overlap. The address space is the CPU's, `cpu-memory-0`.
pub(crate) fn rom_check(roms: &[Rom]) -> Option<String> {
    let mut out = String::new();
    for w in roms.windows(2) {
        let (last, rom) = (&w[0], &w[1]);
        if last.addr.wrapping_add(last.romsize) > rom.addr {
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
            out.push_str(
                "\nThe following two regions overlap (in the cpu-memory-0 address space):\n",
            );
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

/// A RAM range of the system address space.
#[derive(Clone, Debug)]
pub struct RamRange {
    /// The guest physical address.
    pub addr: u64,
    /// The size in bytes.
    pub size: u64,
    /// Whether the guest cannot write it.
    pub readonly: bool,
    /// The backing block.
    pub block: Arc<RamBlock>,
    /// Where `addr` is in `block`.
    pub offset: u64,
    /// A ROM device, a flash: the ROMs are written into it, but it is not RAM.
    pub rom_device: bool,
}

/// Copy `data` to `addr` and zero the following `zeros` bytes, in RAM only.
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

/// `rom_reset()`, with the writes clipped to RAM.
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
            let gap = b - gapstart;
            if gap > best_size {
                best_base = gapstart;
                best_size = gap;
            }
        } else if count == 1 && count + se == 0 {
            gapstart = b;
        }
        count += se;
    }
    (best_base, best_base + best_size)
}

/// `struct arm_boot_info` after `arm_load_kernel()`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BootInfo {
    /// `loader_start`, the base of RAM.
    pub loader_start: u64,
    /// `ram_size`.
    pub ram_size: u64,
    /// Whether the CPUs reset into the kernel (`env->boot_info` set): a direct kernel boot.
    /// Without it the PC stays at its reset value.
    pub direct: bool,
    /// `is_linux`: the CPUs enter the boot stub at `loader_start` in EL1.
    pub is_linux: bool,
    /// `entry`, where the kernel starts.
    pub entry: u64,
    /// `initrd_start`.
    pub initrd_start: u64,
    /// `initrd_size`.
    pub initrd_size: u64,
    /// `dtb_start`, where the device tree goes.
    pub dtb_start: u64,
    /// `dtb_limit`, zero for none.
    pub dtb_limit: u64,
}

/// The loader's output: the ROM list and the messages printed along the way.
#[derive(Debug, Default)]
pub(crate) struct Loader {
    pub(crate) roms: Vec<Rom>,
    pub(crate) messages: Vec<String>,
}

impl Loader {
    /// A message QEMU prints and goes on.
    fn note(&mut self, msg: String) {
        eprintln!("{msg}");
        self.messages.push(msg);
    }

    fn add(&mut self, rom: Rom) {
        rom_insert(&mut self.roms, rom);
    }
}

/// `gunzip()`: inflate the gzip stream `src` into at most `max` bytes.
fn gunzip(src: &[u8], max: usize) -> Result<Vec<u8>, String> {
    const FHCRC: u8 = 0x02;
    const FEXTRA: u8 = 0x04;
    const FNAME: u8 = 0x08;
    const FCOMMENT: u8 = 0x10;
    const RESERVED: u8 = 0xe0;
    const TOOSMALL: &str = "Error: gunzip out of data in header\n";
    let srclen = src.len();
    if srclen < 4 {
        return Err(TOOSMALL.to_string());
    }
    let flags = src[3];
    if src[2] != 8 || flags & RESERVED != 0 {
        return Err("Error: Bad gzipped data\n".to_string());
    }
    let mut i = 10usize;
    if flags & FEXTRA != 0 {
        if srclen < 12 {
            return Err(TOOSMALL.to_string());
        }
        i = 12 + usize::from(src[10]) + (usize::from(src[11]) << 8);
    }
    for f in [FNAME, FCOMMENT] {
        if flags & f != 0 {
            while i < srclen {
                i += 1;
                if src[i - 1] == 0 {
                    break;
                }
            }
        }
    }
    if flags & FHCRC != 0 {
        i += 2;
    }
    if i >= srclen {
        return Err(TOOSMALL.to_string());
    }
    let mut d = Decompress::new(false);
    let mut out = Vec::with_capacity(max.min(srclen.saturating_mul(8)));
    loop {
        if out.len() == out.capacity() {
            if out.len() >= max {
                // Z_BUF_ERROR: the output buffer is full.
                return Err("Error: inflate() returned -5\n".to_string());
            }
            out.reserve((out.len().max(1 << 16)).min(max - out.len()));
        }
        let before_in = d.total_in();
        let before_out = d.total_out();
        let consumed = d.total_in() as usize;
        match d.decompress_vec(&src[i + consumed..], &mut out, FlushDecompress::Finish) {
            Ok(Status::StreamEnd) => return Ok(out),
            Ok(_) => {
                if d.total_in() == before_in && d.total_out() == before_out {
                    // No progress: the input ran out before the end of the stream.
                    return Err("Error: inflate() returned -5\n".to_string());
                }
            }
            // Z_DATA_ERROR.
            Err(_) => return Err("Error: inflate() returned -3\n".to_string()),
        }
    }
}

/// `load_image_gzipped_buffer()`: `None` if `data` is not gzip or does not inflate.
fn load_gzipped(ld: &mut Loader, filename: &str, data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 2 || data[0] != 0x1f || data[1] != 0x8b {
        return None;
    }
    match gunzip(data, LOAD_IMAGE_MAX_DECOMPRESSED_BYTES) {
        Ok(v) => Some(v),
        Err(e) => {
            ld.note(e);
            ld.note(format!("{filename}: unable to decompress gzipped kernel file"));
            None
        }
    }
}

/// `unpack_efi_zboot_image()`: the payload of a Linux EFI zboot image, or `data` itself if it
/// is not one.
fn unpack_efi_zboot(ld: &mut Loader, data: Vec<u8>) -> Result<Vec<u8>, ()> {
    if data.len() < 64
        || &data[0..2] != b"MZ"
        || &data[4..8] != b"zimg"
        || data[56..60] != [0xcd, 0x23, 0x82, 0x81]
    {
        return Ok(data);
    }
    let ploff = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    let plsize = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    if ploff + plsize > data.len() {
        ld.note("unable to handle corrupt EFI zboot image".to_string());
        return Err(());
    }
    let ctype = &data[24..56];
    let ctype_len = ctype.iter().position(|&b| b == 0).unwrap_or(ctype.len());
    if &ctype[..ctype_len] != b"gzip" {
        let shown = &ctype[..ctype_len.min(31)];
        ld.note(format!(
            "unable to handle EFI zboot image with \"{}\" compression",
            String::from_utf8_lossy(shown)
        ));
        return Err(());
    }
    match gunzip(&data[ploff..ploff + plsize], LOAD_IMAGE_MAX_DECOMPRESSED_BYTES) {
        Ok(v) => Ok(v),
        Err(e) => {
            ld.note(e);
            ld.note("failed to decompress EFI zboot image".to_string());
            Err(())
        }
    }
}

/// `load_aarch64_image()`: returns the kernel size and the entry point.
fn load_aarch64_image(
    ld: &mut Loader,
    filename: &str,
    file: Option<&[u8]>,
    mem_base: u64,
) -> Option<(u64, u64)> {
    let file = file?;
    // On aarch64, it's the bootloader's job to uncompress the kernel.
    let buffer = match load_gzipped(ld, filename, file) {
        Some(b) => b,
        // Load as raw file otherwise, unpacking an EFI zboot image.
        None => unpack_efi_zboot(ld, file.to_vec()).ok()?,
    };
    let size = buffer.len();
    let mut kernel_load_offset = KERNEL64_LOAD_ADDR;
    let mut kernel_size = 0u64;
    // Check the arm64 magic header value; very old kernels may not have it.
    if size > ARM64_MAGIC_OFFSET + 4
        && &buffer[ARM64_MAGIC_OFFSET..ARM64_MAGIC_OFFSET + 4] == b"ARM\x64"
    {
        let le64 = |o: usize| u64::from_le_bytes(buffer[o..o + 8].try_into().unwrap_or([0; 8]));
        // text_offset and image_size; text_offset is only valid if image_size is nonzero.
        kernel_size = le64(ARM64_TEXT_OFFSET_OFFSET + 8);
        if kernel_size != 0 {
            kernel_load_offset = le64(ARM64_TEXT_OFFSET_OFFSET);
            // The boot stub sits at the very bottom of RAM, so an image asking for an offset
            // that may overlap it goes 2 MiB further up.
            if kernel_load_offset < BOOTLOADER_MAX_SIZE {
                kernel_load_offset += 2 * MIB;
            }
        }
    }
    // Kernels before v3.17 don't populate image_size, and raw images have no header.
    if kernel_size == 0 {
        kernel_size = size as u64;
    }
    let entry = mem_base.wrapping_add(kernel_load_offset);
    ld.add(Rom::blob(filename, entry, buffer));
    Some((kernel_size, entry))
}

/// `load_elf_strerror()`.
fn load_elf_strerror(code: i64) -> &'static str {
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

/// What `load_elf64()` loaded: size, entry, lowest and highest address.
type ElfLoaded = (i64, u64, u64, u64);

/// `load_elf64()` for a little-endian AArch64 ELF, with the segments added as ROMs. Returns
/// the size or an `ELF_LOAD_*` code.
fn load_elf64(ld: &mut Loader, name: &str, f: &[u8]) -> Result<ElfLoaded, i64> {
    let u16_at = |o: usize| f.get(o..o + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |o: usize| f.get(o..o + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()));
    let u64_at = |o: usize| f.get(o..o + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap()));
    let e_machine = u16_at(18).ok_or(ELF_LOAD_FAILED)?;
    if e_machine != EM_AARCH64 {
        return Err(ELF_LOAD_WRONG_ARCH);
    }
    let e_entry = u64_at(24).ok_or(ELF_LOAD_FAILED)?;
    let e_phoff = u64_at(32).ok_or(ELF_LOAD_FAILED)?;
    let e_phnum = usize::from(u16_at(56).ok_or(ELF_LOAD_FAILED)?);
    if e_phnum == 0 {
        return Err(ELF_LOAD_FAILED);
    }
    let mut ph = Vec::with_capacity(e_phnum);
    for i in 0..e_phnum {
        let o = usize::try_from(e_phoff)
            .ok()
            .and_then(|p| p.checked_add(i * 56))
            .ok_or(ELF_LOAD_FAILED)?;
        let get32 = |d| u32_at(o + d).ok_or(ELF_LOAD_FAILED);
        let get64 = |d| u64_at(o + d).ok_or(ELF_LOAD_FAILED);
        ph.push(Phdr {
            p_type: get32(0)?,
            p_flags: get32(4)?,
            p_offset: get64(8)?,
            p_vaddr: get64(16)?,
            p_paddr: get64(24)?,
            p_filesz: get64(32)?,
            p_memsz: get64(40)?,
        });
    }
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
            let end = file_size.checked_add(p.p_offset).ok_or(ELF_LOAD_FAILED)?;
            if (f.len() as u64) < end {
                return Err(ELF_LOAD_FAILED);
            }
            data = &f[p.p_offset as usize..end as usize];
        }
        // A segment whose zero-initialised part overlaps another segment is loaded with only
        // its file size: its memsz is the runtime size.
        if mem_size > file_size {
            let zero_start = p.p_paddr.wrapping_add(file_size);
            let zero_end = p.p_paddr.wrapping_add(mem_size);
            let overlaps = ph.iter().enumerate().any(|(j, q)| {
                let other_start = q.p_paddr;
                let other_end = q.p_paddr.wrapping_add(q.p_memsz);
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
            && e_entry < p.p_vaddr.wrapping_add(p.p_filesz)
            && p.p_flags & PF_X != 0
        {
            entry = e_entry - p.p_vaddr + p.p_paddr;
        }
        // Zero sized segments make no blob, which could trip the overlap check.
        if mem_size != 0 {
            ld.add(Rom {
                name: format!("{name} ELF program header segment {i}"),
                addr,
                data: data.to_vec(),
                romsize: mem_size,
            });
        }
        total += mem_size as i64;
        low = low.min(addr);
        high = high.max(addr.wrapping_add(mem_size));
    }
    Ok((total, entry, low, high))
}

/// `arm_load_elf()`: `Ok(None)` if the file is not ELF.
fn arm_load_elf(ld: &mut Loader, name: &str, f: &[u8]) -> Result<Option<ElfLoaded>, String> {
    // load_elf_hdr(): anything without the magic and a full header is not ELF, silently.
    if f.len() < 16 || &f[..4] != b"\x7fELF" {
        return Ok(None);
    }
    let is64 = f[4] == 2;
    if f.len() < if is64 { 64 } else { 52 } {
        return Ok(None);
    }
    let big = f[5] == 2;
    let fail = |code: i64| format!("Couldn't load elf '{name}': {}", load_elf_strerror(code));
    if !is64 || big {
        // The machine number decides first, as in load_elf32() and load_elf64().
        let m = if big {
            u16::from_be_bytes([f[18], f[19]])
        } else {
            u16::from_le_bytes([f[18], f[19]])
        };
        if m != EM_AARCH64 {
            return Err(fail(ELF_LOAD_WRONG_ARCH));
        }
        return Err(format!(
            "Couldn't load elf '{name}': big-endian and ELF32 AArch64 images are not supported"
        ));
    }
    match load_elf64(ld, name, f) {
        Ok((size, entry, low, high)) if size > 0 => Ok(Some((size, entry, low, high))),
        Ok((size, ..)) => Err(fail(size)),
        Err(code) => Err(fail(code)),
    }
}

/// The 10 words of `bootloader_aarch64` with the fixups filled in.
fn bootloader_aarch64(dtb: u64, entry: u64) -> Vec<u8> {
    let words: [u32; 10] = [
        0x5800_00c0, // ldr x0, arg ; Load the lower 32-bits of DTB
        0xaa1f_03e1, // mov x1, xzr
        0xaa1f_03e2, // mov x2, xzr
        0xaa1f_03e3, // mov x3, xzr
        0x5800_0084, // ldr x4, entry ; Load the lower 32-bits of kernel entry
        0xd61f_0080, // br x4      ; Jump to the kernel entry point
        dtb as u32,
        (dtb >> 32) as u32,
        entry as u32,
        (entry >> 32) as u32,
    ];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// The `-kernel`, `-initrd` and `-dtb` file names.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct BootFiles<'a> {
    pub(crate) kernel: Option<&'a str>,
    pub(crate) initrd: Option<&'a str>,
}

/// `arm_load_kernel()` without the DTB, which the board loads at machine_done.
pub(crate) fn arm_load_kernel(
    ld: &mut Loader,
    info: &mut BootInfo,
    files: BootFiles<'_>,
) -> Result<(), String> {
    info.dtb_limit = 0;
    match files.kernel {
        None => {
            // arm_setup_firmware_boot(): the DTB goes to the base of RAM for the firmware.
            info.dtb_start = info.loader_start;
            Ok(())
        }
        Some(k) => arm_setup_direct_kernel_boot(ld, info, k, files.initrd),
    }
}

/// `arm_setup_direct_kernel_boot()` for an AArch64 CPU.
fn arm_setup_direct_kernel_boot(
    ld: &mut Loader,
    info: &mut BootInfo,
    kernel: &str,
    initrd: Option<&str>,
) -> Result<(), String> {
    let ram_end = info.loader_start + info.ram_size;
    let file = std::fs::read(kernel).ok();
    let mut is_linux = false;
    let mut kernel_size: i64 = -1;
    let mut entry = 0u64;
    let (mut low, mut high) = (0u64, 0u64);

    // Assume that raw images are linux kernels, and ELF images are not.
    if let Some(f) = &file {
        if let Some((size, e, l, h)) = arm_load_elf(ld, kernel, f)? {
            kernel_size = size;
            entry = e;
            low = l;
            high = h;
            // If there is still some room at the base of RAM, try to put the DTB there.
            if low > info.loader_start || high < info.loader_start {
                // Use low as the limit if it may point into RAM, else no limit.
                if low < info.loader_start {
                    low = 0;
                }
                info.dtb_start = info.loader_start;
                info.dtb_limit = low;
            }
        }
    }
    // uImage is not modelled.
    if kernel_size < 0 {
        if let Some((size, e)) = load_aarch64_image(ld, kernel, file.as_deref(), info.loader_start)
        {
            kernel_size = size as i64;
            entry = e;
            low = entry;
            high = low.wrapping_add(size);
        }
        is_linux = true;
    }
    if kernel_size < 0 {
        return Err(format!("could not load kernel '{kernel}'"));
    }
    if kernel_size as u64 > info.ram_size {
        return Err(format!(
            "kernel '{kernel}' is too large to fit in RAM (kernel size {kernel_size}, RAM size {})",
            info.ram_size
        ));
    }
    let _ = low;
    info.entry = entry;

    // Put the initrd far enough into RAM that the kernel does not clobber it when it
    // uncompresses, but low enough to stay in lowmem.
    info.initrd_start = info.loader_start + (info.ram_size / 2).min(128 * MIB);
    if high != 0 {
        info.initrd_start = info.initrd_start.max(high);
    }
    info.initrd_start =
        info.initrd_start.wrapping_add(TARGET_PAGE_SIZE - 1) & !(TARGET_PAGE_SIZE - 1);

    if is_linux {
        info.initrd_size = 0;
        if let Some(f) = initrd {
            if info.initrd_start >= ram_end {
                return Err("not enough space after kernel to load initrd".to_string());
            }
            // load_ramdisk_as() (u-boot ramdisks) is not modelled; load_image_targphys_as().
            let max = ram_end - info.initrd_start;
            let data = std::fs::read(f)
                .ok()
                .filter(|d| !d.is_empty() && d.len() as u64 <= max)
                .ok_or_else(|| format!("could not load initrd '{f}'"))?;
            info.initrd_size = data.len() as u64;
            ld.add(Rom::blob(f, info.initrd_start, data));
        }
        // Some AArch64 kernels map the 2 MiB around the DTB early, so it is 2 MiB aligned.
        let align = 2 * MIB;
        info.dtb_start = (info.initrd_start + info.initrd_size).div_ceil(align) * align;
        if info.dtb_start >= ram_end {
            return Err("Not enough space for DTB after kernel/initrd".to_string());
        }
        ld.add(Rom::blob(
            "bootloader",
            info.loader_start,
            bootloader_aarch64(info.dtb_start, entry),
        ));
    }
    info.is_linux = is_linux;
    info.direct = true;
    Ok(())
}

/// `load_device_tree()`: the blob in a buffer twice its size plus 10000.
fn load_device_tree(ld: &mut Loader, filename: &str) -> Option<Fdt> {
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

/// `fdt_add_psci_node()` for PSCI 1.0 on an AArch64 CPU: nothing when the conduit is
/// disabled.
fn add_psci_node(fdt: &mut Fdt, conduit: PsciConduit) -> Result<(), String> {
    let method = match conduit {
        PsciConduit::Disabled => return Ok(()),
        PsciConduit::Hvc => "hvc",
        PsciConduit::Smc => "smc",
    };
    // A /psci node already there may have other function IDs: replace it.
    if fdt.exists("/psci") {
        fdt.nop_node("/psci")?;
    }
    fdt.add_subnode("/psci")?;
    fdt.setprop("/psci", "compatible", b"arm,psci-1.0\0arm,psci-0.2\0arm,psci\0")?;
    fdt.setprop_string("/psci", "method", method)?;
    fdt.setprop_cell("/psci", "cpu_suspend", QEMU_PSCI_0_2_FN64_CPU_SUSPEND)?;
    fdt.setprop_cell("/psci", "cpu_off", QEMU_PSCI_0_2_FN_CPU_OFF)?;
    fdt.setprop_cell("/psci", "cpu_on", QEMU_PSCI_0_2_FN64_CPU_ON)?;
    fdt.setprop_cell("/psci", "migrate", QEMU_PSCI_0_2_FN64_MIGRATE)
}

/// `arm_load_dtb()`: finish the tree (the board's, or the user's `-dtb`) and add it to the
/// ROM list at `info.dtb_start`. Returns the tree, or `None` if it does not fit below
/// `info.dtb_limit`, which QEMU does not treat as an error. `conduit` is the CPUs'
/// `psci-conduit`.
pub(crate) fn arm_load_dtb(
    ld: &mut Loader,
    info: &BootInfo,
    board_fdt: &Fdt,
    dtb_filename: Option<&str>,
    cmdline: &str,
    conduit: PsciConduit,
) -> Result<Option<Fdt>, String> {
    let addr = info.dtb_start;
    let addr_limit = info.dtb_limit;
    let mut fdt = match dtb_filename {
        Some(f) => match load_device_tree(ld, f) {
            Some(fdt) => fdt,
            None => return Err(format!("Couldn't open dtb file {f}")),
        },
        None => board_fdt.clone(),
    };
    let size = fdt.as_bytes().len() as u64;
    if addr_limit > addr && size > addr_limit - addr {
        // Installing the blob at addr would cross addr_limit: the caller decides.
        return Ok(None);
    }

    let acells = fdt.getprop_cell("/", "#address-cells")?;
    let scells = fdt.getprop_cell("/", "#size-cells")?;
    if acells == 0 || scells == 0 {
        return Err("dtb file invalid (#address-cells or #size-cells 0)".to_string());
    }
    if scells < 2 && info.ram_size >= 4 * GIB {
        return Err("qemu: dtb file not compatible with RAM size > 4GB".to_string());
    }

    // Nop all root nodes matching /memory or /memory@unit-address.
    for path in fdt.node_unit_path("memory")? {
        if path.starts_with("/memory") {
            fdt.nop_node(&path)?;
        }
    }

    // fdt_add_memory_node() without NUMA.
    let nodename = format!("/memory@{:x}", info.loader_start);
    fdt.add_subnode(&nodename)?;
    fdt.setprop_string(&nodename, "device_type", "memory")?;
    let reg = sized_cells(&[(acells, info.loader_start), (scells, info.ram_size)])
        .ok_or_else(|| format!("couldn't add /memory@{:x} node", info.loader_start))?;
    fdt.setprop(&nodename, "reg", &reg)?;

    if !fdt.exists("/chosen") {
        fdt.add_subnode("/chosen")?;
    }
    if !cmdline.is_empty() {
        fdt.setprop_string("/chosen", "bootargs", cmdline)?;
    }
    if info.initrd_size != 0 {
        let start = sized_cells(&[(acells, info.initrd_start)])
            .ok_or_else(|| "couldn't set /chosen/linux,initrd-start".to_string())?;
        fdt.setprop("/chosen", "linux,initrd-start", &start)?;
        let end = sized_cells(&[(acells, info.initrd_start + info.initrd_size)])
            .ok_or_else(|| "couldn't set /chosen/linux,initrd-end".to_string())?;
        fdt.setprop("/chosen", "linux,initrd-end", &end)?;
    }

    add_psci_node(&mut fdt, conduit)?;

    // The blob is a ROM so that it is copied again at every reset.
    ld.add(Rom::blob("dtb", addr, fdt.as_bytes().to_vec()));
    Ok(Some(fdt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rom(name: &str, addr: u64, romsize: u64) -> Rom {
        Rom { name: name.to_string(), addr, data: Vec::new(), romsize }
    }

    #[test]
    fn roms_stay_sorted() {
        let mut roms = Vec::new();
        rom_insert(&mut roms, rom("b", 0x2000, 1));
        rom_insert(&mut roms, rom("a", 0x1000, 1));
        rom_insert(&mut roms, rom("c", 0x2000, 1));
        let names: Vec<_> = roms.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn gap_between_blobs() {
        let roms = [rom("a", 0x100, 0x100), rom("b", 0x800, 0x100)];
        assert_eq!(largest_gap(&roms, 0, 0x1000), (0x900, 0x1000));
        let roms = [rom("a", 0x100, 0x100), rom("b", 0xe00, 0x100)];
        assert_eq!(largest_gap(&roms, 0, 0x1000), (0x200, 0xe00));
        // A blob running past the top ends the region.
        let roms = [rom("a", 0x0, 0x100), rom("b", 0xf00, 0x200)];
        assert_eq!(largest_gap(&roms, 0, 0x1000), (0x100, 0xf00));
        assert_eq!(largest_gap(&[], 0x4000, 0x1000), (0x4000, 0x5000));
        // Overlapping blobs count as one.
        let roms = [rom("a", 0x100, 0x400), rom("b", 0x200, 0x100)];
        assert_eq!(largest_gap(&roms, 0, 0x1000), (0x500, 0x1000));
    }

    #[test]
    fn overlap_report() {
        let roms = [rom("a", 0x1000, 0x100), rom("b", 0x1080, 0x10), rom("c", 0x2000, 1)];
        let msg = rom_check(&roms).unwrap();
        assert!(msg.starts_with("Some ROM regions are overlapping\n"));
        assert!(msg.ends_with(
            "\nThe following two regions overlap (in the cpu-memory-0 address space):\n  \
             a (addresses 0x0000000000001000 - 0x0000000000001100)\n  \
             b (addresses 0x0000000000001080 - 0x0000000000001090)\n"
        ));
        assert!(rom_check(&roms[1..]).is_none());
    }

    fn gzip(data: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn gunzip_errors() {
        let data: Vec<u8> = (0..100_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let z = gzip(&data);
        assert_eq!(gunzip(&z, 1 << 20).unwrap(), data);
        assert_eq!(gunzip(&z[..3], 1 << 20).unwrap_err(), "Error: gunzip out of data in header\n");
        let mut bad = z.clone();
        bad[2] = 7;
        assert_eq!(gunzip(&bad, 1 << 20).unwrap_err(), "Error: Bad gzipped data\n");
        assert_eq!(gunzip(&z[..10], 1 << 20).unwrap_err(), "Error: gunzip out of data in header\n");
        assert_eq!(
            gunzip(&z[..z.len() / 2], 1 << 20).unwrap_err(),
            "Error: inflate() returned -5\n"
        );
        assert_eq!(gunzip(&z, 1000).unwrap_err(), "Error: inflate() returned -5\n");
        let mut corrupt = z.clone();
        corrupt[10] = 0xff;
        assert_eq!(gunzip(&corrupt, 1 << 20).unwrap_err(), "Error: inflate() returned -3\n");
    }

    #[test]
    fn zboot_images() {
        let payload = b"payload!".repeat(100);
        let z = gzip(&payload);
        let mut img = vec![0u8; 64];
        img[0..2].copy_from_slice(b"MZ");
        img[4..8].copy_from_slice(b"zimg");
        img[8..12].copy_from_slice(&64u32.to_le_bytes());
        img[12..16].copy_from_slice(&(z.len() as u32).to_le_bytes());
        img[24..28].copy_from_slice(b"gzip");
        img[56..60].copy_from_slice(&[0xcd, 0x23, 0x82, 0x81]);
        img.extend_from_slice(&z);
        let mut ld = Loader::default();
        assert_eq!(unpack_efi_zboot(&mut ld, img.clone()).unwrap(), payload);
        let mut short = img.clone();
        short.truncate(100);
        assert!(unpack_efi_zboot(&mut ld, short).is_err());
        assert_eq!(ld.messages.last().unwrap(), "unable to handle corrupt EFI zboot image");
        let mut zstd = img.clone();
        zstd[24..28].copy_from_slice(b"zstd");
        assert!(unpack_efi_zboot(&mut ld, zstd).is_err());
        assert_eq!(
            ld.messages.last().unwrap(),
            "unable to handle EFI zboot image with \"zstd\" compression"
        );
        // Not a zboot image: unchanged.
        assert_eq!(unpack_efi_zboot(&mut ld, b"plain".to_vec()).unwrap(), b"plain");
    }
}
