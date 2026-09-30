// SPDX-License-Identifier: GPL-2.0-or-later

//! Direct kernel boot against hand built bzImage headers and ELF images, with the expected
//! values worked out from x86_load_linux() in hw/i386/x86-common.c.

use ruvm_firmware::x86_linux::*;

const MIB: u64 = 1024 * 1024;
/// PC_FW_DATA on pc and q35.
const ACPI: u64 = 0x28000;

fn rd16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn rd32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

fn rd64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

fn u32item(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// A synthetic bzImage: `setup_sects` setup sectors, then a recognisable payload.
struct Img {
    protocol: u16,
    setup_sects: u8,
    loadflags: u8,
    initrd_addr_max: u32,
    xloadflags: u16,
    len: usize,
}

impl Img {
    fn new(protocol: u16) -> Self {
        Img {
            protocol,
            setup_sects: 4,
            loadflags: LOADED_HIGH,
            initrd_addr_max: 0x7fff_ffff,
            xloadflags: 0,
            len: 16384,
        }
    }

    fn build(&self) -> Vec<u8> {
        let mut k: Vec<u8> = (0..self.len).map(|i| (i * 7 % 251) as u8).collect();
        // Clear the fields the loader writes so the tests see its values only.
        for off in [0x20usize, 0x22, 0x1fa, 0x210, 0x218, 0x21c, 0x224, 0x228, 0x250] {
            k[off..off + 4].fill(0);
        }
        k[0x250..0x258].fill(0);
        k[0x1f1] = self.setup_sects;
        k[0x202..0x206].copy_from_slice(b"HdrS");
        k[0x206..0x208].copy_from_slice(&self.protocol.to_le_bytes());
        k[0x211] = self.loadflags;
        k[0x22c..0x230].copy_from_slice(&self.initrd_addr_max.to_le_bytes());
        k[0x236..0x238].copy_from_slice(&self.xloadflags.to_le_bytes());
        k
    }
}

fn input<'a>(kernel: &'a [u8], cmdline: &'a str) -> X86LinuxInput<'a> {
    X86LinuxInput {
        kernel_filename: "bzImage",
        kernel,
        cmdline,
        dtb_filename: "board.dtb",
        below_4g_mem_size: 128 * MIB,
        acpi_data_size: ACPI,
        ..Default::default()
    }
}

fn linux(i: &X86LinuxInput<'_>) -> LinuxBoot {
    match x86_load_linux(i).unwrap() {
        X86KernelBoot::Linux(l) => l,
        other => panic!("expected a Linux kernel, got {other:?}"),
    }
}

fn err(i: &X86LinuxInput<'_>) -> String {
    x86_load_linux(i).unwrap_err().to_string()
}

#[test]
fn protocol_2_00() {
    let k = Img::new(0x200).build();
    let mut i = input(&k, "console=ttyS0");
    i.below_4g_mem_size = 2048 * MIB;
    let l = linux(&i);
    // 13 bytes plus NUL rounded to 16.
    assert_eq!(
        l.addresses,
        LoadAddresses { real_addr: 0x90000, cmdline_addr: 0x99ff0, prot_addr: 0x100000 }
    );
    assert_eq!(l.initrd_max, 0x37ff_ffff);
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(setup.len(), 5 * 512);
    assert_eq!(rd16(setup, 0x20), 0xA33F);
    assert_eq!(rd16(setup, 0x22), 0x9ff0);
    assert_eq!(rd32(setup, 0x228), 0);
    assert_eq!(setup[0x210], 0xB0);
    // No heap before 2.01.
    assert_eq!(setup[0x211], LOADED_HIGH);
    assert_eq!(rd16(setup, 0x224), 0);
    assert_eq!(l.item(FW_CFG_SETUP_ADDR).unwrap(), u32item(0x90000));
    assert_eq!(l.item(FW_CFG_CMDLINE_ADDR).unwrap(), u32item(0x99ff0));
    assert_eq!(l.item(FW_CFG_CMDLINE_SIZE).unwrap(), u32item(14));
    assert_eq!(l.item(FW_CFG_CMDLINE_DATA).unwrap(), b"console=ttyS0\0");
}

#[test]
fn protocol_2_00_zimage_loads_low() {
    let mut img = Img::new(0x200);
    img.loadflags = 0;
    let k = img.build();
    let l = linux(&input(&k, ""));
    assert_eq!(
        l.addresses,
        LoadAddresses { real_addr: 0x90000, cmdline_addr: 0x99ff0, prot_addr: 0x10000 }
    );
    assert_eq!(l.item(FW_CFG_KERNEL_ADDR).unwrap(), u32item(0x10000));
}

#[test]
fn protocol_2_01_uses_heap_with_old_cmdline() {
    let k = Img::new(0x201).build();
    let l = linux(&input(&k, "a"));
    assert_eq!(l.addresses.cmdline_addr, 0x99ff0);
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(setup[0x211], LOADED_HIGH | CAN_USE_HEAP);
    assert_eq!(rd16(setup, 0x224), 0x9ff0 - 0x200);
    assert_eq!(rd16(setup, 0x20), 0xA33F);
}

#[test]
fn protocol_2_02() {
    let k = Img::new(0x202).build();
    let mut i = input(&k, "root=/dev/vda");
    i.below_4g_mem_size = 2048 * MIB;
    let l = linux(&i);
    assert_eq!(
        l.addresses,
        LoadAddresses { real_addr: 0x10000, cmdline_addr: 0x20000, prot_addr: 0x100000 }
    );
    assert_eq!(l.initrd_max, 0x37ff_ffff);
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(rd32(setup, 0x228), 0x20000);
    assert_eq!(rd16(setup, 0x20), 0);
    assert_eq!(setup[0x210], 0xB0);
    assert_eq!(setup[0x211], LOADED_HIGH | CAN_USE_HEAP);
    assert_eq!(rd16(setup, 0x224), 0xfe00);
    assert_eq!(l.item(FW_CFG_SETUP_ADDR).unwrap(), u32item(0x10000));
}

#[test]
fn protocol_2_06_initrd_near_top() {
    let k = Img::new(0x206).build();
    let initrd = vec![0x5a; 0x1234];
    let mut i = input(&k, "");
    i.initrd = Some(&initrd);
    let l = linux(&i);
    // initrd_addr_max 0x7fffffff is above 128 MiB minus PC_FW_DATA, so it is capped.
    assert_eq!(l.initrd_max, 0x7fd_7fff);
    assert_eq!(l.initrd_addr, Some(0x7fd_6000));
    assert_eq!(l.item(FW_CFG_INITRD_ADDR).unwrap(), u32item(0x7fd_6000));
    assert_eq!(l.item(FW_CFG_INITRD_SIZE).unwrap(), u32item(0x1234));
    assert_eq!(l.item(FW_CFG_INITRD_DATA).unwrap(), initrd.as_slice());
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(rd32(setup, 0x218), 0x7fd_6000);
    assert_eq!(rd32(setup, 0x21c), 0x1234);
}

#[test]
fn protocol_2_06_initrd_addr_max_from_header() {
    let mut img = Img::new(0x206);
    img.initrd_addr_max = 0x37ff_ffff;
    let k = img.build();
    let initrd = vec![1u8; 4096];
    let mut i = input(&k, "");
    i.initrd = Some(&initrd);
    i.below_4g_mem_size = 2048 * MIB;
    let l = linux(&i);
    assert_eq!(l.initrd_max, 0x37ff_ffff);
    assert_eq!(l.initrd_addr, Some(0x37ff_e000));
}

#[test]
fn initrd_too_large() {
    let k = Img::new(0x206).build();
    let initrd = vec![0u8; 0x7fd_7fff];
    let mut i = input(&k, "");
    i.initrd = Some(&initrd);
    assert_eq!(
        err(&i),
        "qemu: initrd is too large, cannot support.(max: 134053887, need 134053887)"
    );
    let initrd = vec![0u8; 0x7fd_7ffe];
    i.initrd = Some(&initrd);
    let l = linux(&i);
    assert_eq!(l.initrd_addr, Some(0));
}

#[test]
fn initrd_needs_protocol_2_00() {
    let k = Img::new(0x105).build();
    let initrd = [0u8; 16];
    let mut i = input(&k, "");
    i.initrd = Some(&initrd);
    assert_eq!(err(&i), "qemu: linux kernel too old to load a ram disk");
}

#[test]
fn protocol_2_10_dtb() {
    let mut img = Img::new(0x20a);
    img.setup_sects = 1;
    img.len = 4101;
    let k = img.build();
    let dtb = [0xd0, 0x0d, 0xfe, 0xed, 1, 2, 3];
    let mut i = input(&k, "");
    i.dtb = Some(&dtb);
    let l = linux(&i);
    assert_eq!(l.setup_size, 1024);
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(rd64(setup, 0x250), 0x100000 + 4112);
    let data = l.item(FW_CFG_KERNEL_DATA).unwrap();
    assert_eq!(l.item(FW_CFG_KERNEL_SIZE).unwrap(), u32item((4112 + 16 + 7 - 1024) as u32));
    let sd = &data[4112 - 1024..];
    assert_eq!(rd64(sd, 0), 0);
    assert_eq!(rd32(sd, 8), SETUP_DTB);
    assert_eq!(rd32(sd, 12), 7);
    assert_eq!(&sd[16..], &dtb);
    // etc/boot/kernel carries the setup_data but not the header patches.
    let file = &l.files[0];
    assert_eq!(file.name, "etc/boot/kernel");
    assert_eq!(file.data.len(), 4112 + 16 + 7);
    assert_eq!(&file.data[..4101], k.as_slice());
    assert_eq!(l.option_rom, "linuxboot_dma.bin");
}

#[test]
fn dtb_needs_protocol_2_09_and_data() {
    let k = Img::new(0x208).build();
    let dtb = [1u8; 4];
    let mut i = input(&k, "");
    i.dtb = Some(&dtb);
    assert_eq!(err(&i), "qemu: Linux kernel too old to load a dtb");
    let k = Img::new(0x209).build();
    let mut i = input(&k, "");
    i.dtb = Some(&[]);
    assert_eq!(err(&i), "qemu: error reading dtb board.dtb: Success");
}

#[test]
fn rng_seed_chains_after_dtb() {
    let mut img = Img::new(0x20f);
    img.len = 4096;
    let k = img.build();
    let dtb = [9u8; 3];
    let seed = [0xaa; 32];
    let mut i = input(&k, "");
    i.dtb = Some(&dtb);
    i.rng_seed = Some(&seed);
    let l = linux(&i);
    let image = &l.files[0].data;
    // dtb at 4096, seed at align16(4096 + 16 + 3) = 4128.
    assert_eq!(rd64(image, 4096), 0);
    assert_eq!(rd64(image, 4128), 0x100000 + 4096);
    assert_eq!(rd32(image, 4128 + 8), SETUP_RNG_SEED);
    assert_eq!(rd32(image, 4128 + 12), 32);
    assert_eq!(image.len(), 4128 + 16 + 32);
    assert_eq!(rd64(l.item(FW_CFG_SETUP_DATA).unwrap(), 0x250), 0x100000 + 4128);
}

#[test]
fn protocol_2_12_above_4g() {
    let mut img = Img::new(0x20c);
    img.xloadflags = XLF_CAN_BE_LOADED_ABOVE_4G;
    img.initrd_addr_max = 0x37ff_ffff;
    let k = img.build();
    let mut i = input(&k, "");
    i.below_4g_mem_size = 3072 * MIB;
    let l = linux(&i);
    assert_eq!(l.initrd_max, (3072 * MIB - ACPI - 1) as u32);

    // Without the flag the header limit applies.
    img.xloadflags = 0;
    let k = img.build();
    let mut i = input(&k, "");
    i.below_4g_mem_size = 3072 * MIB;
    assert_eq!(linux(&i).initrd_max, 0x37ff_ffff);
}

#[test]
fn protocol_2_15_full() {
    let mut img = Img::new(0x20f);
    img.xloadflags = XLF_CAN_BE_LOADED_ABOVE_4G;
    img.setup_sects = 0;
    let k = img.build();
    let initrd = vec![7u8; 3 * 4096 + 5];
    let mut i = input(&k, "console=ttyS0 vga=0x317 quiet");
    i.initrd = Some(&initrd);
    i.acpi_data_size = 0;
    i.below_4g_mem_size = 4096 * MIB - 1;
    let l = linux(&i);
    assert_eq!(l.protocol, 0x20f);
    assert_eq!(l.setup_size, 2560);
    assert_eq!(l.initrd_max, 0xffff_fffe);
    assert_eq!(l.initrd_addr, Some(0xffff_c000));
    let keys: Vec<u16> = l.fw_cfg.iter().map(|i| i.key).collect();
    assert_eq!(
        keys,
        [
            FW_CFG_CMDLINE_ADDR,
            FW_CFG_CMDLINE_SIZE,
            FW_CFG_CMDLINE_DATA,
            FW_CFG_INITRD_ADDR,
            FW_CFG_INITRD_SIZE,
            FW_CFG_INITRD_DATA,
            FW_CFG_KERNEL_ADDR,
            FW_CFG_KERNEL_SIZE,
            FW_CFG_KERNEL_DATA,
            FW_CFG_SETUP_ADDR,
            FW_CFG_SETUP_SIZE,
            FW_CFG_SETUP_DATA,
        ]
    );
    let setup = l.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(rd16(setup, 0x1fa), 0x317);
    assert_eq!(l.item(FW_CFG_SETUP_SIZE).unwrap(), u32item(2560));
    assert_eq!(l.item(FW_CFG_KERNEL_SIZE).unwrap(), u32item(16384 - 2560));
    assert_eq!(l.item(FW_CFG_KERNEL_DATA).unwrap(), &k[2560..]);
    // Everything past the patched header matches the file.
    assert_eq!(&setup[0x260..], &k[0x260..2560]);
}

#[test]
fn invalid_kernel_header() {
    let mut img = Img::new(0x20f);
    img.setup_sects = 0;
    img.len = 2559;
    let k = img.build();
    assert_eq!(err(&input(&k, "")), "qemu: invalid kernel header");
}

#[test]
fn confidential_guest_keeps_header() {
    let k = Img::new(0x20f).build();
    let mut i = input(&k, "vga=ask");
    i.confidential_guest = true;
    let l = linux(&i);
    assert_eq!(l.item(FW_CFG_SETUP_DATA).unwrap(), &k[..2560]);
}

#[test]
fn no_setup_header_is_protocol_0() {
    let mut k: Vec<u8> = vec![0x11; 8192];
    k[0x1f1] = 2;
    let l = linux(&input(&k, "x"));
    assert_eq!(l.protocol, 0);
    assert_eq!(
        l.addresses,
        LoadAddresses { real_addr: 0x90000, cmdline_addr: 0x99ff0, prot_addr: 0x10000 }
    );
    // The patched header is not copied for protocol 0.
    assert_eq!(l.item(FW_CFG_SETUP_DATA).unwrap(), &k[..1536]);
}

#[test]
fn empty_kernel() {
    assert_eq!(err(&input(&[], "")), "qemu: could not load kernel 'bzImage': Success");
}

#[test]
fn cmdline_sizes_and_nul() {
    assert_eq!(cmdline_size(""), 16);
    assert_eq!(cmdline_size("0123456789abcde"), 16);
    assert_eq!(cmdline_size("0123456789abcdef"), 32);
    assert_eq!(cmdline_size("ab\0cdef"), 16);
    let k = Img::new(0x20f).build();
    let l = linux(&input(&k, "ab\0cd"));
    assert_eq!(l.item(FW_CFG_CMDLINE_DATA).unwrap(), b"ab\0");
    assert_eq!(l.item(FW_CFG_CMDLINE_SIZE).unwrap(), u32item(3));
}

#[test]
fn vga_parameter() {
    assert_eq!(vga_mode("quiet"), Ok(None));
    assert_eq!(vga_mode("vga=normal"), Ok(Some(0xffff)));
    assert_eq!(vga_mode("vga=extended"), Ok(Some(0xfffe)));
    assert_eq!(vga_mode("vga=ask"), Ok(Some(0xfffd)));
    assert_eq!(vga_mode("vga=791 quiet"), Ok(Some(791)));
    assert_eq!(vga_mode("vga=0x317"), Ok(Some(0x317)));
    assert_eq!(vga_mode("vga=010"), Ok(Some(8)));
    assert_eq!(vga_mode("vga= 5"), Ok(Some(5)));
    assert_eq!(vga_mode("vga=-1"), Ok(Some(u32::MAX)));
    assert_eq!(vga_mode("xvga=3"), Ok(Some(3)));
    assert_eq!(vga_mode("vga=foo"), Err(X86LinuxError::InvalidVga));
    assert_eq!(vga_mode("vga=12,"), Err(X86LinuxError::InvalidVga));
    assert_eq!(vga_mode("vga=0x"), Err(X86LinuxError::InvalidVga));
    assert_eq!(vga_mode("vga=08"), Err(X86LinuxError::InvalidVga));
    assert_eq!(vga_mode("vga=4294967296"), Err(X86LinuxError::InvalidVga));
    assert_eq!(vga_mode("vga="), Err(X86LinuxError::InvalidVga));
    let k = Img::new(0x20f).build();
    assert_eq!(err(&input(&k, "vga=bad")), "qemu: invalid 'vga=' kernel parameter.");
}

#[test]
fn multiboot_detection() {
    let mut k = vec![0u8; 4096];
    let flags: u32 = 0x0001_0003;
    k[0x40..0x44].copy_from_slice(&MULTIBOOT_MAGIC.to_le_bytes());
    k[0x44..0x48].copy_from_slice(&flags.to_le_bytes());
    let sum = 0u32.wrapping_sub(MULTIBOOT_MAGIC).wrapping_sub(flags);
    k[0x48..0x4c].copy_from_slice(&sum.to_le_bytes());
    assert_eq!(
        x86_load_linux(&input(&k, "")).unwrap(),
        X86KernelBoot::Multiboot(MultibootHeader { offset: 0x40, flags })
    );
    // A bad checksum is not multiboot.
    k[0x48] ^= 1;
    assert!(matches!(x86_load_linux(&input(&k, "")).unwrap(), X86KernelBoot::Linux(_)));
}

fn note(namesz_name: &[u8], kind: u32, desc: &[u8]) -> Vec<u8> {
    let mut n = Vec::new();
    n.extend_from_slice(&(namesz_name.len() as u32).to_le_bytes());
    n.extend_from_slice(&(desc.len() as u32).to_le_bytes());
    n.extend_from_slice(&kind.to_le_bytes());
    n.extend_from_slice(namesz_name);
    while n.len() % 4 != 0 {
        n.push(0);
    }
    n.extend_from_slice(desc);
    while n.len() % 4 != 0 {
        n.push(0);
    }
    n
}

/// A tiny ELF64 with one PT_LOAD at 16 MiB and one PT_NOTE holding two Xen notes.
fn elf64(pvh: Option<u64>, e_flags: u32) -> Vec<u8> {
    let mut notes = note(b"Xen\0", 1, &0xdead_beef_u64.to_le_bytes());
    if let Some(entry) = pvh {
        notes.extend(note(b"Xen\0", XEN_ELFNOTE_PHYS32_ENTRY, &entry.to_le_bytes()));
    }
    let code = [0x90u8; 0x10];
    let phoff = 64usize;
    let load_off = phoff + 2 * 56;
    let note_off = load_off + code.len();
    let mut f = vec![0u8; note_off];
    f[0..4].copy_from_slice(b"\x7fELF");
    f[4] = 2;
    f[5] = 1;
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes());
    f[18..20].copy_from_slice(&62u16.to_le_bytes());
    f[24..32].copy_from_slice(&0x0100_0000u64.to_le_bytes());
    f[32..40].copy_from_slice(&(phoff as u64).to_le_bytes());
    f[48..52].copy_from_slice(&e_flags.to_le_bytes());
    f[52..54].copy_from_slice(&64u16.to_le_bytes());
    f[54..56].copy_from_slice(&56u16.to_le_bytes());
    f[56..58].copy_from_slice(&2u16.to_le_bytes());
    let ph = |f: &mut Vec<u8>,
              at: usize,
              ty: u32,
              off: usize,
              paddr: u64,
              filesz: usize,
              memsz: usize,
              align: u64| {
        f[at..at + 4].copy_from_slice(&ty.to_le_bytes());
        f[at + 8..at + 16].copy_from_slice(&(off as u64).to_le_bytes());
        f[at + 16..at + 24].copy_from_slice(&(paddr | 0xffff_ffff_8000_0000).to_le_bytes());
        f[at + 24..at + 32].copy_from_slice(&paddr.to_le_bytes());
        f[at + 32..at + 40].copy_from_slice(&(filesz as u64).to_le_bytes());
        f[at + 40..at + 48].copy_from_slice(&(memsz as u64).to_le_bytes());
        f[at + 48..at + 56].copy_from_slice(&align.to_le_bytes());
    };
    ph(&mut f, phoff, 1, load_off, 0x0100_0000, code.len(), 0x20, 0x20_0000);
    ph(&mut f, phoff + 56, 4, note_off, 0, notes.len(), notes.len(), 4);
    f[load_off..note_off].copy_from_slice(&code);
    f.extend_from_slice(&notes);
    f
}

/// The same idea as an ELF32 with only the PVH note.
fn elf32(entry: u32) -> Vec<u8> {
    let notes = note(b"Xen\0", XEN_ELFNOTE_PHYS32_ENTRY, &entry.to_le_bytes());
    let phoff = 52usize;
    let load_off = phoff + 2 * 32;
    let note_off = load_off + 8;
    let mut f = vec![0u8; note_off];
    f[0..4].copy_from_slice(b"\x7fELF");
    f[4] = 1;
    f[5] = 1;
    f[18..20].copy_from_slice(&3u16.to_le_bytes());
    f[28..32].copy_from_slice(&(phoff as u32).to_le_bytes());
    f[44..46].copy_from_slice(&2u16.to_le_bytes());
    let ph = |f: &mut Vec<u8>, at: usize, ty: u32, off: usize, paddr: u32, size: usize| {
        f[at..at + 4].copy_from_slice(&ty.to_le_bytes());
        f[at + 4..at + 8].copy_from_slice(&(off as u32).to_le_bytes());
        f[at + 12..at + 16].copy_from_slice(&paddr.to_le_bytes());
        f[at + 16..at + 20].copy_from_slice(&(size as u32).to_le_bytes());
        f[at + 20..at + 24].copy_from_slice(&(size as u32).to_le_bytes());
        f[at + 28..at + 32].copy_from_slice(&4u32.to_le_bytes());
    };
    ph(&mut f, phoff, 1, load_off, 0x20_0000, 8);
    ph(&mut f, phoff + 32, 4, note_off, 0, notes.len());
    f.extend_from_slice(&notes);
    f
}

#[test]
fn pvh_elf64() {
    let k = elf64(Some(0x0100_0100), 0);
    assert_eq!(pvh_entry(&k), Some(0x0100_0100));
    let initrd = vec![3u8; 4096];
    let mut i = input(&k, "console=hvc0");
    i.initrd = Some(&initrd);
    i.acpi_data_size = 0;
    let X86KernelBoot::Pvh(p) = x86_load_linux(&i).unwrap() else { panic!("expected PVH") };
    assert_eq!(p.entry, 0x0100_0100);
    assert_eq!(p.load_addr, 0x0100_0000);
    assert_eq!(p.kernel_size, 0x20);
    assert_eq!(p.option_rom, "pvh.bin");
    assert_eq!(
        p.segments,
        [ElfSegment { addr: 0x0100_0000, data: vec![0x90; 0x10], mem_size: 0x20 }]
    );
    let keys: Vec<u16> = p.fw_cfg.iter().map(|i| i.key).collect();
    assert_eq!(
        keys,
        [
            FW_CFG_KERNEL_ENTRY,
            FW_CFG_KERNEL_ADDR,
            FW_CFG_KERNEL_SIZE,
            FW_CFG_CMDLINE_SIZE,
            FW_CFG_CMDLINE_DATA,
            FW_CFG_SETUP_SIZE,
            FW_CFG_SETUP_DATA,
            FW_CFG_INITRD_ADDR,
            FW_CFG_INITRD_SIZE,
            FW_CFG_INITRD_DATA,
        ]
    );
    assert_eq!(p.item(FW_CFG_KERNEL_ENTRY).unwrap(), u32item(0x0100_0100));
    assert_eq!(p.item(FW_CFG_CMDLINE_DATA).unwrap(), b"console=hvc0\0");
    assert_eq!(p.item(FW_CFG_SETUP_SIZE).unwrap(), u32item(8192));
    let setup = p.item(FW_CFG_SETUP_DATA).unwrap();
    assert_eq!(setup.len(), 8192);
    assert_eq!(&setup[..k.len()], k.as_slice());
    assert_eq!(p.initrd_addr, Some(0x7ffe000));
    assert_eq!(p.item(FW_CFG_INITRD_ADDR).unwrap(), u32item(0x7ffe000));
}

#[test]
fn pvh_elf32() {
    let k = elf32(0x20_0040);
    let X86KernelBoot::Pvh(p) = x86_load_linux(&input(&k, "")).unwrap() else {
        panic!("expected PVH")
    };
    assert_eq!(p.entry, 0x20_0040);
    assert_eq!(p.load_addr, 0x20_0000);
    assert_eq!(p.kernel_size, 8);
}

#[test]
fn pvh_errors() {
    assert_eq!(pvh_entry(&elf64(None, 0)), None);
    assert_eq!(
        err(&input(&elf64(None, 0), "")),
        "Error loading uncompressed kernel without PVH ELF Note"
    );
    assert_eq!(err(&input(&elf64(Some(1), 4), "")), "elfboot unsupported flags = 4");
    let mut k = elf64(Some(1), 0);
    k[18] = 40;
    assert_eq!(err(&input(&k, "")), "Error while loading elf kernel");
    let initrd = vec![0u8; 128 * MIB as usize];
    let k = elf64(Some(1), 0);
    let mut i = input(&k, "");
    i.acpi_data_size = 0;
    i.initrd = Some(&initrd);
    assert_eq!(
        err(&i),
        "qemu: initrd is too large, cannot support.(max: 134217727, need 134217728)"
    );
}
