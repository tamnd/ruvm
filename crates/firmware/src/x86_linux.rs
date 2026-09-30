// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct kernel boot on x86, ported from `x86_load_linux()` in hw/i386/x86-common.c.
//!
//! QEMU reads the kernel, initrd and DTB files, patches the Linux setup header and pushes
//! everything into fw_cfg, where the linuxboot_dma.bin or pvh.bin option ROM picks it up. This
//! module does the same computation over byte buffers and returns the fw_cfg items instead of
//! adding them, so a machine crate can feed them to its fw_cfg device later. Reading files is the
//! caller's job, and so are the error messages for files that cannot be read.
//!
//! Three kinds of kernel are recognised, in the order QEMU tries them:
//!
//! - A Linux bzImage or zImage, found by the `HdrS` magic at offset 0x202. This is the main path
//!   and is ported completely.
//! - A multiboot kernel, found by the multiboot magic in the first 8 KiB. Only detection is
//!   ported; the caller gets [`X86KernelBoot::Multiboot`] and has to hand the image to a
//!   multiboot loader.
//! - An uncompressed ELF `vmlinux` with a `XEN_ELFNOTE_PHYS32_ENTRY` note (PVH). The ELF is
//!   parsed with a small reader here, and the loadable segments are returned so the caller can
//!   place them in guest memory the way QEMU's ROM blobs would.
//!
//! Anything else is treated as a very old kernel with boot protocol 0, as QEMU does.

use std::fmt;

/// fw_cfg selector for the protected mode kernel load address.
pub const FW_CFG_KERNEL_ADDR: u16 = 0x07;
/// fw_cfg selector for the protected mode kernel size.
pub const FW_CFG_KERNEL_SIZE: u16 = 0x08;
/// fw_cfg selector for the initrd load address.
pub const FW_CFG_INITRD_ADDR: u16 = 0x0a;
/// fw_cfg selector for the initrd size.
pub const FW_CFG_INITRD_SIZE: u16 = 0x0b;
/// fw_cfg selector for the PVH entry point.
pub const FW_CFG_KERNEL_ENTRY: u16 = 0x10;
/// fw_cfg selector for the protected mode kernel bytes.
pub const FW_CFG_KERNEL_DATA: u16 = 0x11;
/// fw_cfg selector for the initrd bytes.
pub const FW_CFG_INITRD_DATA: u16 = 0x12;
/// fw_cfg selector for the command line address.
pub const FW_CFG_CMDLINE_ADDR: u16 = 0x13;
/// fw_cfg selector for the command line size, including the terminating NUL.
pub const FW_CFG_CMDLINE_SIZE: u16 = 0x14;
/// fw_cfg selector for the command line bytes, including the terminating NUL.
pub const FW_CFG_CMDLINE_DATA: u16 = 0x15;
/// fw_cfg selector for the real mode setup load address.
pub const FW_CFG_SETUP_ADDR: u16 = 0x16;
/// fw_cfg selector for the real mode setup size.
pub const FW_CFG_SETUP_SIZE: u16 = 0x17;
/// fw_cfg selector for the real mode setup bytes, with the patched header.
pub const FW_CFG_SETUP_DATA: u16 = 0x18;

/// fw_cfg file holding the kernel image without the header patches, for OVMF's own loader.
pub const BOOT_KERNEL_FILE: &str = "etc/boot/kernel";

/// Option ROM QEMU registers for a Linux kernel.
pub const LINUXBOOT_DMA_ROM: &str = "linuxboot_dma.bin";
/// Option ROM QEMU registers for a PVH kernel.
pub const PVH_ROM: &str = "pvh.bin";

/// Size of the header buffer QEMU reads from the start of the kernel.
pub const HEADER_SIZE: usize = 8192;

/// "HdrS", the setup header signature at offset 0x202.
pub const HDRS_MAGIC: u32 = 0x5372_6448;

/// `loadflags` bit: the protected mode code is loaded at 0x100000.
pub const LOADED_HIGH: u8 = 1 << 0;
/// `loadflags` bit: `heap_end_ptr` is valid.
pub const CAN_USE_HEAP: u8 = 1 << 7;
/// `xloadflags` bit: the kernel accepts an initrd above 4 GiB.
pub const XLF_CAN_BE_LOADED_ABOVE_4G: u16 = 1 << 1;

/// `setup_data` type for a flattened device tree.
pub const SETUP_DTB: u32 = 2;
/// `setup_data` type for a random seed.
pub const SETUP_RNG_SEED: u32 = 9;
/// Size of `struct setup_data` without the payload.
pub const SETUP_DATA_HEADER_SIZE: usize = 16;

/// The Xen ELF note type carrying the 32-bit PVH entry point.
pub const XEN_ELFNOTE_PHYS32_ENTRY: u32 = 18;

/// The multiboot header magic.
pub const MULTIBOOT_MAGIC: u32 = 0x1BAD_B002;

/// Value QEMU writes to `type_of_loader`: high nybble 0xB is QEMU, low nybble the revision.
pub const QEMU_LOADER_TYPE: u8 = 0xB0;

/// Setup header field offsets used by the loader.
pub mod hdr {
    /// `setup_sects`, number of 512 byte setup sectors after the boot sector.
    pub const SETUP_SECTS: usize = 0x1f1;
    /// `vid_mode`.
    pub const VID_MODE: usize = 0x1fa;
    /// `header`, the "HdrS" magic.
    pub const MAGIC: usize = 0x202;
    /// `version`, the boot protocol version.
    pub const VERSION: usize = 0x206;
    /// `type_of_loader`.
    pub const TYPE_OF_LOADER: usize = 0x210;
    /// `loadflags`.
    pub const LOADFLAGS: usize = 0x211;
    /// `ramdisk_image`.
    pub const RAMDISK_IMAGE: usize = 0x218;
    /// `ramdisk_size`.
    pub const RAMDISK_SIZE: usize = 0x21c;
    /// `heap_end_ptr`.
    pub const HEAP_END_PTR: usize = 0x224;
    /// `cmd_line_ptr`, protocol 2.02 and later.
    pub const CMD_LINE_PTR: usize = 0x228;
    /// `initrd_addr_max`, protocol 2.03 and later.
    pub const INITRD_ADDR_MAX: usize = 0x22c;
    /// `xloadflags`, protocol 2.12 and later.
    pub const XLOADFLAGS: usize = 0x236;
    /// `setup_data`, protocol 2.09 and later.
    pub const SETUP_DATA: usize = 0x250;
    /// `cl_magic` in the zero page, used by protocols before 2.02.
    pub const CL_MAGIC: usize = 0x20;
    /// `cl_offset` in the zero page, used by protocols before 2.02.
    pub const CL_OFFSET: usize = 0x22;
}

/// One fixed-selector fw_cfg item with its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FwCfgItem {
    /// The selector, one of the `FW_CFG_*` constants.
    pub key: u16,
    /// The item contents. Items QEMU adds with `fw_cfg_add_i32` are 4 bytes little endian.
    pub data: Vec<u8>,
}

/// One named fw_cfg file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FwCfgFile {
    /// File name as it appears in the fw_cfg directory.
    pub name: String,
    /// File contents.
    pub data: Vec<u8>,
}

/// What `x86_load_linux` needs from the machine and the command line.
#[derive(Debug, Clone, Copy, Default)]
pub struct X86LinuxInput<'a> {
    /// `-kernel` path, used only in error messages.
    pub kernel_filename: &'a str,
    /// The whole kernel file.
    pub kernel: &'a [u8],
    /// `-append`. QEMU works on a C string, so anything after a NUL byte is ignored.
    pub cmdline: &'a str,
    /// The whole `-initrd` file, if given.
    pub initrd: Option<&'a [u8]>,
    /// `-dtb` path, used only in error messages.
    pub dtb_filename: &'a str,
    /// The whole `-dtb` file, if given.
    pub dtb: Option<&'a [u8]>,
    /// A random seed to pass as a `SETUP_RNG_SEED` setup_data entry. QEMU 11.1 does not do this,
    /// so leave it `None` for QEMU compatible behaviour. See [`x86_load_linux`].
    pub rng_seed: Option<&'a [u8]>,
    /// RAM below 4 GiB.
    pub below_4g_mem_size: u64,
    /// Space reserved at the top of low RAM for firmware data (`PC_FW_DATA` on pc and q35, 0 on
    /// microvm).
    pub acpi_data_size: u64,
    /// True for a confidential guest (`machine->cgs`), where the setup header is left unpatched.
    pub confidential_guest: bool,
}

/// Failures of the direct kernel boot, with QEMU's messages as their `Display` text.
///
/// The `qemu: ...` messages are printed by QEMU with `fprintf(stderr, ...)` followed by a
/// newline. The other three go through `error_report`, which prefixes the program name; that
/// prefix is left to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X86LinuxError {
    /// The kernel file is empty.
    CouldNotLoadKernel {
        /// The `-kernel` path.
        filename: String,
    },
    /// The ELF header flags have `LOAD_ELF_HEADER_HAS_ADDR` bits set.
    ElfbootUnsupportedFlags(u32),
    /// The ELF image could not be parsed or loaded.
    ElfLoad,
    /// An ELF kernel without a PVH entry note.
    NoPvhNote,
    /// The initrd does not fit below `initrd_max`.
    InitrdTooLarge {
        /// The highest usable initrd address.
        max: u32,
        /// The initrd size.
        need: u64,
    },
    /// The value after `vga=` is not a mode name or a number.
    InvalidVga,
    /// An initrd was given for a kernel with boot protocol older than 2.00.
    TooOldForInitrd,
    /// The setup sectors extend past the end of the file.
    InvalidKernelHeader,
    /// A DTB was given for a kernel with boot protocol older than 2.09.
    TooOldForDtb,
    /// The DTB file is empty.
    DtbRead {
        /// The `-dtb` path.
        filename: String,
    },
}

impl fmt::Display for X86LinuxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // QEMU appends strerror(errno) to two of these. Neither case sets errno, so on Linux the
        // text is "Success".
        match self {
            Self::CouldNotLoadKernel { filename } => {
                write!(f, "qemu: could not load kernel '{filename}': Success")
            }
            Self::ElfbootUnsupportedFlags(flags) => {
                write!(f, "elfboot unsupported flags = {flags:x}")
            }
            Self::ElfLoad => f.write_str("Error while loading elf kernel"),
            Self::NoPvhNote => {
                f.write_str("Error loading uncompressed kernel without PVH ELF Note")
            }
            Self::InitrdTooLarge { max, need } => {
                write!(f, "qemu: initrd is too large, cannot support.(max: {max}, need {need})")
            }
            Self::InvalidVga => f.write_str("qemu: invalid 'vga=' kernel parameter."),
            Self::TooOldForInitrd => f.write_str("qemu: linux kernel too old to load a ram disk"),
            Self::InvalidKernelHeader => f.write_str("qemu: invalid kernel header"),
            Self::TooOldForDtb => f.write_str("qemu: Linux kernel too old to load a dtb"),
            Self::DtbRead { filename } => write!(f, "qemu: error reading dtb {filename}: Success"),
        }
    }
}

impl std::error::Error for X86LinuxError {}

/// Where the pieces of a Linux kernel go, from the boot protocol and `loadflags`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadAddresses {
    /// Real mode setup code and the zero page.
    pub real_addr: u64,
    /// Kernel command line.
    pub cmdline_addr: u64,
    /// Protected mode kernel.
    pub prot_addr: u64,
}

/// A Linux kernel ready for linuxboot_dma.bin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxBoot {
    /// Boot protocol version, 0 for a kernel without a setup header.
    pub protocol: u16,
    /// The chosen load addresses.
    pub addresses: LoadAddresses,
    /// Highest address the initrd may end below.
    pub initrd_max: u32,
    /// Where the initrd goes, if there is one.
    pub initrd_addr: Option<u64>,
    /// Size of the real mode setup, boot sector included.
    pub setup_size: usize,
    /// The fixed-selector items, in the order QEMU adds them.
    pub fw_cfg: Vec<FwCfgItem>,
    /// The named files (`etc/boot/kernel`).
    pub files: Vec<FwCfgFile>,
    /// Option ROM to register with boot index 0.
    pub option_rom: &'static str,
}

/// A loadable ELF segment, the equivalent of one ROM blob QEMU adds for the image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElfSegment {
    /// Guest physical load address (`p_paddr`).
    pub addr: u64,
    /// The bytes from the file.
    pub data: Vec<u8>,
    /// Size in memory; anything past `data` is zero.
    pub mem_size: u64,
}

/// An uncompressed kernel booted through its PVH entry point by pvh.bin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PvhBoot {
    /// The value read from the `XEN_ELFNOTE_PHYS32_ENTRY` note. fw_cfg gets the low 32 bits.
    pub entry: u64,
    /// Lowest load address of the image.
    pub load_addr: u32,
    /// Span of the image from the lowest to the highest loaded byte.
    pub kernel_size: u32,
    /// Where the initrd goes, if there is one.
    pub initrd_addr: Option<u64>,
    /// The segments to place in guest memory.
    pub segments: Vec<ElfSegment>,
    /// The fixed-selector items, in the order QEMU adds them.
    pub fw_cfg: Vec<FwCfgItem>,
    /// Option ROM to register with boot index 0.
    pub option_rom: &'static str,
}

/// A multiboot header found in the first 8 KiB of the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultibootHeader {
    /// Offset of the magic in the file.
    pub offset: usize,
    /// The header flags.
    pub flags: u32,
}

/// The outcome of [`x86_load_linux`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X86KernelBoot {
    /// A Linux bzImage or zImage.
    Linux(LinuxBoot),
    /// A PVH ELF kernel.
    Pvh(PvhBoot),
    /// A multiboot kernel. QEMU hands these to `load_multiboot`, which is not ported here.
    Multiboot(MultibootHeader),
}

impl LinuxBoot {
    /// The contents of fixed-selector item `key`, if it was added.
    pub fn item(&self, key: u16) -> Option<&[u8]> {
        find_item(&self.fw_cfg, key)
    }
}

impl PvhBoot {
    /// The contents of fixed-selector item `key`, if it was added.
    pub fn item(&self, key: u16) -> Option<&[u8]> {
        find_item(&self.fw_cfg, key)
    }
}

fn find_item(items: &[FwCfgItem], key: u16) -> Option<&[u8]> {
    items.iter().find(|i| i.key == key).map(|i| i.data.as_slice())
}

fn ld16(b: &[u8], off: usize) -> u16 {
    let mut v = [0u8; 2];
    v.copy_from_slice(&b[off..off + 2]);
    u16::from_le_bytes(v)
}

fn ld32(b: &[u8], off: usize) -> u32 {
    let mut v = [0u8; 4];
    v.copy_from_slice(&b[off..off + 4]);
    u32::from_le_bytes(v)
}

fn ld64(b: &[u8], off: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(v)
}

fn st16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}

fn st32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn st64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

fn item_u32(key: u16, v: u64) -> FwCfgItem {
    // fw_cfg_add_i32 takes a uint32_t, so wider values are truncated.
    FwCfgItem { key, data: (v as u32).to_le_bytes().to_vec() }
}

/// The first [`HEADER_SIZE`] bytes of the kernel, zero padded when the file is shorter.
///
/// QEMU leaves the tail of its stack buffer uninitialised in that case; zeroes make the result
/// deterministic.
pub fn kernel_header(kernel: &[u8]) -> [u8; HEADER_SIZE] {
    let mut header = [0u8; HEADER_SIZE];
    let n = kernel.len().min(HEADER_SIZE);
    header[..n].copy_from_slice(&kernel[..n]);
    header
}

/// The boot protocol version if the header carries the `HdrS` signature.
pub fn boot_protocol(header: &[u8; HEADER_SIZE]) -> Option<u16> {
    if ld32(header, hdr::MAGIC) == HDRS_MAGIC { Some(ld16(header, hdr::VERSION)) } else { None }
}

/// The command line as QEMU sees it: the bytes up to the first NUL.
fn c_cmdline(cmdline: &str) -> &[u8] {
    let b = cmdline.as_bytes();
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    &b[..end]
}

/// The command line length rounded up to 16 bytes with room for the NUL, as QEMU computes it.
pub fn cmdline_size(cmdline: &str) -> u64 {
    (c_cmdline(cmdline).len() as u64 + 16) & !15
}

/// Load addresses for a kernel with boot protocol `protocol` and the given `loadflags`.
///
/// Kernels before 2.00, and zImages without `LOADED_HIGH`, put the setup at 0x90000, the
/// command line just below 0x9a000 and the kernel at 0x10000. Protocols 2.00 and 2.01 with
/// `LOADED_HIGH` move the kernel to 1 MiB. From 2.02 the setup goes to 0x10000 and the command
/// line to 0x20000.
pub fn load_addresses(protocol: u16, loadflags: u8, cmdline_size: u64) -> LoadAddresses {
    if protocol < 0x200 || loadflags & LOADED_HIGH == 0 {
        LoadAddresses {
            real_addr: 0x90000,
            cmdline_addr: 0x9a000 - cmdline_size,
            prot_addr: 0x10000,
        }
    } else if protocol < 0x202 {
        LoadAddresses {
            real_addr: 0x90000,
            cmdline_addr: 0x9a000 - cmdline_size,
            prot_addr: 0x100000,
        }
    } else {
        LoadAddresses { real_addr: 0x10000, cmdline_addr: 0x20000, prot_addr: 0x100000 }
    }
}

/// The highest address an initrd may occupy.
///
/// This is `initrd_addr_max` from the header for protocol 2.03 and later, 0x37ffffff before that,
/// and `UINT32_MAX` for a 2.12 or later kernel with `XLF_CAN_BE_LOADED_ABOVE_4G`. The result is
/// then capped just below the firmware data at the top of low RAM.
pub fn initrd_max(
    protocol: u16,
    header: &[u8; HEADER_SIZE],
    below_4g_mem_size: u64,
    acpi_data_size: u64,
) -> u32 {
    let mut max =
        if protocol >= 0x20c && ld16(header, hdr::XLOADFLAGS) & XLF_CAN_BE_LOADED_ABOVE_4G != 0 {
            u32::MAX
        } else if protocol >= 0x203 {
            ld32(header, hdr::INITRD_ADDR_MAX)
        } else {
            0x37ff_ffff
        };
    let top = below_4g_mem_size.wrapping_sub(acpi_data_size);
    if u64::from(max) >= top {
        max = top.wrapping_sub(1) as u32;
    }
    max
}

/// Places an initrd of `size` bytes page aligned just below `initrd_max`.
pub fn place_initrd(initrd_max: u32, size: u64) -> Result<u64, X86LinuxError> {
    if size >= u64::from(initrd_max) {
        return Err(X86LinuxError::InitrdTooLarge { max: initrd_max, need: size });
    }
    Ok((u64::from(initrd_max) - size) & !4095)
}

/// Size of the real mode setup from `setup_sects`, where 0 means 4.
pub fn setup_size(header: &[u8; HEADER_SIZE]) -> usize {
    let mut sects = usize::from(header[hdr::SETUP_SECTS]);
    if sects == 0 {
        sects = 4;
    }
    (sects + 1) * 512
}

/// The video mode from a `vga=` option on the command line, or `None` if there is none.
///
/// Like QEMU this looks at the first `vga=` substring anywhere in the command line, accepts
/// `normal`, `ext` and `ask`, and otherwise wants a C style number (decimal, 0x hex or 0 octal)
/// followed by a space or the end of the line.
pub fn vga_mode(cmdline: &str) -> Result<Option<u32>, X86LinuxError> {
    let b = c_cmdline(cmdline);
    let Some(pos) = b.windows(4).position(|w| w == b"vga=") else {
        return Ok(None);
    };
    let v = &b[pos + 4..];
    if v.starts_with(b"normal") {
        return Ok(Some(0xffff));
    }
    if v.starts_with(b"ext") {
        return Ok(Some(0xfffe));
    }
    if v.starts_with(b"ask") {
        return Ok(Some(0xfffd));
    }
    match strtoui(v) {
        Some((mode, end)) if end == v.len() || v[end] == b' ' => Ok(Some(mode)),
        _ => Err(X86LinuxError::InvalidVga),
    }
}

/// `qemu_strtoui(s, &end, 0, &val)` on glibc: returns the value and the end offset, or `None`
/// for no conversion or overflow.
fn strtoui(s: &[u8]) -> Option<(u32, usize)> {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    let base: u64 = if s.get(i) == Some(&b'0')
        && matches!(s.get(i + 1), Some(b'x' | b'X'))
        && s.get(i + 2).is_some_and(|c| c.is_ascii_hexdigit())
    {
        i += 2;
        16
    } else if s.get(i) == Some(&b'0') {
        8
    } else {
        10
    };
    let start = i;
    let mut value: u64 = 0;
    let mut overflow = false;
    while i < s.len() {
        let d = match (s[i] as char).to_digit(16) {
            Some(d) if u64::from(d) < base => u64::from(d),
            _ => break,
        };
        match value.checked_mul(base).and_then(|v| v.checked_add(d)) {
            Some(v) => value = v,
            None => overflow = true,
        }
        i += 1;
    }
    if i == start || overflow {
        return None;
    }
    // strtoull negates, qemu_strtoui undoes that for the range check and negates again.
    if value > u64::from(u32::MAX) {
        return None;
    }
    let v = value as u32;
    Some((if neg { v.wrapping_neg() } else { v }, i))
}

/// Looks for a valid multiboot header in the first 8 KiB, as `load_multiboot` does before it
/// decides whether the image is multiboot.
pub fn find_multiboot_header(header: &[u8; HEADER_SIZE]) -> Option<MultibootHeader> {
    (0..HEADER_SIZE - 48).step_by(4).find_map(|i| {
        if ld32(header, i) != MULTIBOOT_MAGIC {
            return None;
        }
        let flags = ld32(header, i + 4);
        let checksum = ld32(header, i + 8).wrapping_add(flags).wrapping_add(MULTIBOOT_MAGIC);
        (checksum == 0).then_some(MultibootHeader { offset: i, flags })
    })
}

/// Runs QEMU's direct kernel boot logic over `input`.
///
/// The result lists what QEMU would add to fw_cfg, byte for byte, apart from where QEMU's own
/// output depends on uninitialised memory (the tail of an image shorter than 8 KiB and the
/// padding in front of an appended DTB), which is zero here.
///
/// `rng_seed` is not something QEMU 11.1 does. When it is set, and the kernel speaks protocol
/// 2.09 or later, a `SETUP_RNG_SEED` entry is appended after the DTB and chained in front of it,
/// following the layout QEMU 7.1 and 7.2 used before the feature was dropped.
pub fn x86_load_linux(input: &X86LinuxInput<'_>) -> Result<X86KernelBoot, X86LinuxError> {
    let kernel = input.kernel;
    let cmdline = c_cmdline(input.cmdline);
    let cmdline_size = cmdline_size(input.cmdline);
    let mut cmdline_data = cmdline.to_vec();
    cmdline_data.push(0);
    let cmdline_len = cmdline_data.len() as u64;

    if kernel.is_empty() {
        return Err(X86LinuxError::CouldNotLoadKernel {
            filename: input.kernel_filename.to_string(),
        });
    }
    let mut header = kernel_header(kernel);

    let protocol = match boot_protocol(&header) {
        Some(p) => p,
        None => {
            // Multiboot goes first because a multiboot image can also be an ELF.
            if let Some(mb) = find_multiboot_header(&header) {
                return Ok(X86KernelBoot::Multiboot(mb));
            }
            if ld32(&header, 0) == ELF_MAGIC {
                return load_pvh(input, &header, cmdline_data).map(X86KernelBoot::Pvh);
            }
            0
        }
    };

    let addrs = load_addresses(protocol, header[hdr::LOADFLAGS], cmdline_size);
    let LoadAddresses { real_addr, cmdline_addr, prot_addr } = addrs;
    let initrd_max = initrd_max(protocol, &header, input.below_4g_mem_size, input.acpi_data_size);

    let mut fw_cfg = vec![
        item_u32(FW_CFG_CMDLINE_ADDR, cmdline_addr),
        item_u32(FW_CFG_CMDLINE_SIZE, cmdline_len),
        FwCfgItem { key: FW_CFG_CMDLINE_DATA, data: cmdline_data },
    ];

    if protocol >= 0x202 {
        st32(&mut header, hdr::CMD_LINE_PTR, cmdline_addr as u32);
    } else {
        st16(&mut header, hdr::CL_MAGIC, 0xA33F);
        st16(&mut header, hdr::CL_OFFSET, (cmdline_addr - real_addr) as u16);
    }

    if let Some(mode) = vga_mode(input.cmdline)? {
        st16(&mut header, hdr::VID_MODE, mode as u16);
    }

    if protocol >= 0x200 {
        header[hdr::TYPE_OF_LOADER] = QEMU_LOADER_TYPE;
    }
    if protocol >= 0x201 {
        header[hdr::LOADFLAGS] |= CAN_USE_HEAP;
        st16(&mut header, hdr::HEAP_END_PTR, (cmdline_addr - real_addr - 0x200) as u16);
    }

    let mut initrd_addr = None;
    if let Some(initrd) = input.initrd {
        if protocol < 0x200 {
            return Err(X86LinuxError::TooOldForInitrd);
        }
        let size = initrd.len() as u64;
        let addr = place_initrd(initrd_max, size)?;
        fw_cfg.push(item_u32(FW_CFG_INITRD_ADDR, addr));
        fw_cfg.push(item_u32(FW_CFG_INITRD_SIZE, size));
        fw_cfg.push(FwCfgItem { key: FW_CFG_INITRD_DATA, data: initrd.to_vec() });
        st32(&mut header, hdr::RAMDISK_IMAGE, addr as u32);
        st32(&mut header, hdr::RAMDISK_SIZE, size as u32);
        initrd_addr = Some(addr);
    }

    let setup_size = setup_size(&header);
    if setup_size > kernel.len() {
        return Err(X86LinuxError::InvalidKernelHeader);
    }
    let mut setup = kernel[..setup_size].to_vec();
    let mut image = kernel.to_vec();

    let mut first_setup_data = 0u64;
    if let Some(dtb) = input.dtb {
        if protocol < 0x209 {
            return Err(X86LinuxError::TooOldForDtb);
        }
        if dtb.is_empty() {
            return Err(X86LinuxError::DtbRead { filename: input.dtb_filename.to_string() });
        }
        first_setup_data =
            append_setup_data(&mut image, prot_addr, first_setup_data, SETUP_DTB, dtb);
        st64(&mut header, hdr::SETUP_DATA, first_setup_data);
    }
    if let Some(seed) = input.rng_seed {
        if protocol >= 0x209 {
            first_setup_data =
                append_setup_data(&mut image, prot_addr, first_setup_data, SETUP_RNG_SEED, seed);
            st64(&mut header, hdr::SETUP_DATA, first_setup_data);
        }
    }

    if !input.confidential_guest && protocol > 0 {
        let n = HEADER_SIZE.min(setup_size);
        setup[..n].copy_from_slice(&header[..n]);
    }

    fw_cfg.push(item_u32(FW_CFG_KERNEL_ADDR, prot_addr));
    fw_cfg.push(item_u32(FW_CFG_KERNEL_SIZE, (image.len() - setup_size) as u64));
    fw_cfg.push(FwCfgItem { key: FW_CFG_KERNEL_DATA, data: image[setup_size..].to_vec() });
    fw_cfg.push(item_u32(FW_CFG_SETUP_ADDR, real_addr));
    fw_cfg.push(item_u32(FW_CFG_SETUP_SIZE, setup_size as u64));
    fw_cfg.push(FwCfgItem { key: FW_CFG_SETUP_DATA, data: setup });

    let files = vec![FwCfgFile { name: BOOT_KERNEL_FILE.to_string(), data: image }];

    Ok(X86KernelBoot::Linux(LinuxBoot {
        protocol,
        addresses: addrs,
        initrd_max,
        initrd_addr,
        setup_size,
        fw_cfg,
        files,
        option_rom: LINUXBOOT_DMA_ROM,
    }))
}

/// Appends one `struct setup_data` at the next 16 byte boundary of `image`, linked to `next`,
/// and returns its guest address.
fn append_setup_data(
    image: &mut Vec<u8>,
    prot_addr: u64,
    next: u64,
    kind: u32,
    payload: &[u8],
) -> u64 {
    let offset = image.len().next_multiple_of(16);
    image.resize(offset + SETUP_DATA_HEADER_SIZE, 0);
    st64(image, offset, next);
    st32(image, offset + 8, kind);
    st32(image, offset + 12, payload.len() as u32);
    image.extend_from_slice(payload);
    prot_addr + offset as u64
}

/// The PVH branch of `x86_load_linux` together with `load_elfboot`.
fn load_pvh(
    input: &X86LinuxInput<'_>,
    header: &[u8; HEADER_SIZE],
    cmdline_data: Vec<u8>,
) -> Result<PvhBoot, X86LinuxError> {
    let is64 = header[EI_CLASS] == ELFCLASS64;
    let flags = if is64 { ld32(header, 48) } else { ld32(header, 36) };
    if flags & 0x0001_0004 != 0 {
        return Err(X86LinuxError::ElfbootUnsupportedFlags(flags));
    }

    let elf = load_elf(input.kernel, XEN_ELFNOTE_PHYS32_ENTRY).ok_or(X86LinuxError::ElfLoad)?;
    let entry = elf.note.unwrap_or(0);
    if entry == 0 {
        return Err(X86LinuxError::NoPvhNote);
    }
    let load_addr = elf.low as u32;
    let kernel_size = elf.high.wrapping_sub(elf.low) as u32;

    let mut fw_cfg = vec![
        item_u32(FW_CFG_KERNEL_ENTRY, entry),
        item_u32(FW_CFG_KERNEL_ADDR, u64::from(load_addr)),
        item_u32(FW_CFG_KERNEL_SIZE, u64::from(kernel_size)),
        item_u32(FW_CFG_CMDLINE_SIZE, cmdline_data.len() as u64),
        FwCfgItem { key: FW_CFG_CMDLINE_DATA, data: cmdline_data },
        item_u32(FW_CFG_SETUP_SIZE, HEADER_SIZE as u64),
        FwCfgItem { key: FW_CFG_SETUP_DATA, data: header.to_vec() },
    ];

    let mut initrd_addr = None;
    if let Some(initrd) = input.initrd {
        let max = input.below_4g_mem_size.wrapping_sub(input.acpi_data_size).wrapping_sub(1) as u32;
        let size = initrd.len() as u64;
        let addr = place_initrd(max, size)?;
        fw_cfg.push(item_u32(FW_CFG_INITRD_ADDR, addr));
        fw_cfg.push(item_u32(FW_CFG_INITRD_SIZE, size));
        fw_cfg.push(FwCfgItem { key: FW_CFG_INITRD_DATA, data: initrd.to_vec() });
        initrd_addr = Some(addr);
    }

    Ok(PvhBoot {
        entry,
        load_addr,
        kernel_size,
        initrd_addr,
        segments: elf.segments,
        fw_cfg,
        option_rom: PVH_ROM,
    })
}

const ELF_MAGIC: u32 = 0x464c_457f;
const EI_CLASS: usize = 4;
const EI_DATA: usize = 5;
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EM_386: u16 = 3;
const EM_X86_64: u16 = 62;
const PT_LOAD: u32 = 1;
const PT_NOTE: u32 = 4;

struct Phdr {
    p_type: u32,
    offset: u64,
    paddr: u64,
    filesz: u64,
    memsz: u64,
    align: u64,
}

struct LoadedElf {
    low: u64,
    high: u64,
    note: Option<u64>,
    segments: Vec<ElfSegment>,
}

/// Returns the PVH entry point of an ELF kernel, or `None` if the image is not a little endian
/// x86 ELF or has no `XEN_ELFNOTE_PHYS32_ENTRY` note.
pub fn pvh_entry(kernel: &[u8]) -> Option<u64> {
    load_elf(kernel, XEN_ELFNOTE_PHYS32_ENTRY)?.note
}

/// The subset of QEMU's `load_elf` that x86 direct boot relies on: header checks, the PT_LOAD
/// segments with their overlap trimming, and the note callback `read_pvh_start_addr`.
fn load_elf(file: &[u8], note_type: u32) -> Option<LoadedElf> {
    if file.len() < 16 || ld32(file, 0) != ELF_MAGIC || file[EI_DATA] != ELFDATA2LSB {
        return None;
    }
    let is64 = file[EI_CLASS] == ELFCLASS64;
    let (ehdr_size, phdr_size) = if is64 { (64, 56) } else { (52, 32) };
    if file.len() < ehdr_size {
        return None;
    }
    let machine = ld16(file, 18);
    if machine != EM_X86_64 && machine != EM_386 {
        return None;
    }
    let (phoff, phnum) = if is64 {
        (ld64(file, 32), usize::from(ld16(file, 56)))
    } else {
        (u64::from(ld32(file, 28)), usize::from(ld16(file, 44)))
    };
    let phoff = usize::try_from(phoff).ok()?;
    let table_end = phoff.checked_add(phnum * phdr_size)?;
    if table_end > file.len() {
        return None;
    }
    let phdrs: Vec<Phdr> = (0..phnum)
        .map(|i| {
            let p = &file[phoff + i * phdr_size..];
            if is64 {
                Phdr {
                    p_type: ld32(p, 0),
                    offset: ld64(p, 8),
                    paddr: ld64(p, 24),
                    filesz: ld64(p, 32),
                    memsz: ld64(p, 40),
                    align: ld64(p, 48),
                }
            } else {
                Phdr {
                    p_type: ld32(p, 0),
                    offset: u64::from(ld32(p, 4)),
                    paddr: u64::from(ld32(p, 12)),
                    filesz: u64::from(ld32(p, 16)),
                    memsz: u64::from(ld32(p, 20)),
                    align: u64::from(ld32(p, 28)),
                }
            }
        })
        .collect();

    // elf_word is 32 bits wide for ELFCLASS32, so the overlap arithmetic wraps there.
    let mask = if is64 { u64::MAX } else { u64::from(u32::MAX) };
    let mut low = u64::MAX;
    let mut high = 0u64;
    let mut note = None;
    let mut segments = Vec::new();
    let mut total: u64 = 0;

    for (i, ph) in phdrs.iter().enumerate() {
        if ph.p_type == PT_LOAD {
            let data = segment_bytes(file, ph.offset, ph.filesz)?;
            let mut mem_size = ph.memsz;
            if mem_size > ph.filesz {
                let zero_start = ph.paddr.wrapping_add(ph.filesz) & mask;
                let zero_end = ph.paddr.wrapping_add(mem_size) & mask;
                let overlaps = phdrs.iter().enumerate().any(|(j, o)| {
                    let other_start = o.paddr;
                    let other_end = o.paddr.wrapping_add(o.memsz) & mask;
                    i != j
                        && o.p_type == PT_LOAD
                        && !(other_start >= zero_end || zero_start >= other_end)
                });
                if overlaps {
                    mem_size = ph.filesz;
                }
            }
            if mem_size > (isize::MAX as u64) - total {
                return None;
            }
            total += mem_size;
            let addr = ph.paddr;
            low = low.min(addr);
            high = high.max(addr.wrapping_add(mem_size));
            if mem_size != 0 {
                segments.push(ElfSegment { addr, data: data.to_vec(), mem_size });
            }
        } else if ph.p_type == PT_NOTE {
            let data = segment_bytes(file, ph.offset, ph.filesz)?;
            if let Some(v) = find_note(data, ph.align, note_type, is64) {
                note = Some(v);
            }
        }
    }
    Some(LoadedElf { low, high, note, segments })
}

fn segment_bytes(file: &[u8], offset: u64, size: u64) -> Option<&[u8]> {
    if size == 0 {
        return Some(&[]);
    }
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(size).ok()?)?;
    file.get(start..end)
}

fn align_up(n: u64, align: u64) -> u64 {
    // QEMU_ALIGN_UP with an alignment of 0 divides by zero; ELF says 0 means no alignment.
    let a = align.max(1);
    n.div_ceil(a).wrapping_mul(a)
}

/// `get_elf_note_type` followed by `read_pvh_start_addr`.
///
/// Like QEMU, the walk stops when the size of a single note entry exceeds the size of the whole
/// note segment, and the descriptor is read as a `size_t` for 64-bit images and a `uint32_t` for
/// 32-bit ones. Unlike QEMU, a walk that runs past the segment stops instead of reading beyond it.
fn find_note(notes: &[u8], align: u64, note_type: u32, is64: bool) -> Option<u64> {
    let size = notes.len() as u64;
    let mut pos: usize = 0;
    loop {
        if notes.len() < pos.checked_add(12)? {
            return None;
        }
        let namesz = u64::from(ld32(notes, pos));
        let descsz = u64::from(ld32(notes, pos + 4));
        let kind = ld32(notes, pos + 8);
        if kind == note_type {
            let desc = usize::try_from(12 + align_up(namesz, align)).ok()?.checked_add(pos)?;
            let rest = notes.get(desc..)?;
            return if is64 && rest.len() >= 8 {
                Some(ld64(rest, 0))
            } else if rest.len() >= 4 {
                Some(u64::from(ld32(rest, 0)))
            } else {
                None
            };
        }
        let entry = 12 + align_up(namesz, align) + align_up(descsz, align);
        if entry > size {
            return None;
        }
        pos = pos.checked_add(usize::try_from(entry).ok()?)?;
    }
}
