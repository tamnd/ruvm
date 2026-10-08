// SPDX-License-Identifier: GPL-2.0-or-later

//! The virt board: its device trees against `qemu-system-aarch64 -M virt,dumpdtb=` (see
//! tests/data/gen.sh), its memory map, kernel loading, PSCI between two vCPUs and the
//! tests/tcg/aarch64/system hello test on the interpreter.

use std::io::Read;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ruvm_hw_pci::MsiMessage;
use ruvm_hw_virtio::VirtioPciProps;
use ruvm_hw_virtio::rng::{RandomFile, VirtioRng, VirtioRngConf};
use ruvm_jit::cpu_exec::cpu_exec;
use ruvm_jit::{Vcpu, excp};
use ruvm_machine_arm::virt::{
    VIRT_FW_CFG, VIRT_GIC_DIST, VIRT_GIC_ITS, VIRT_GIC_REDIST, VIRT_MEM, VIRT_MMIO, VIRT_PCIE_MMIO,
    VIRT_PCIE_PIO, VIRT_RTC, VIRT_UART, VirtConfig, VirtMachine, VirtMsi, VirtRequest,
};
use ruvm_mem::{Endian, MemTxAttrs};
use ruvm_target_arm::cpu::ArmCpuModel;
use ruvm_target_arm::tcg::{PSCI_OFF, PSCI_ON, SemihostingHost, new_jit, save_vcpu};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MIB: u64 = 1 << 20;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

fn gunzip_file(name: &str) -> Vec<u8> {
    let f = std::fs::File::open(data(name)).unwrap();
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(f).read_to_end(&mut out).unwrap();
    out
}

/// A scratch file, removed when dropped.
struct TmpFile(PathBuf);

impl TmpFile {
    fn new(tag: &str, name: &str, contents: &[u8]) -> TmpFile {
        let dir = std::env::temp_dir().join(format!("ruvm-virt-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, contents).unwrap();
        TmpFile(p)
    }

    fn path(&self) -> String {
        self.0.to_str().unwrap().to_string()
    }
}

impl Drop for TmpFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        if let Some(d) = self.0.parent() {
            let _ = std::fs::remove_dir(d);
        }
    }
}

/// The fake arm64 Image of gen.sh: text_offset 0, image_size 0x20000, 4 KiB long.
fn fake_image(image_size: u64) -> Vec<u8> {
    let mut v = vec![0u8; 4096];
    v[0..4].copy_from_slice(&0x1400_0000u32.to_le_bytes());
    v[16..24].copy_from_slice(&image_size.to_le_bytes());
    v[56..60].copy_from_slice(b"ARM\x64");
    v
}

fn gzip(data: &[u8]) -> Vec<u8> {
    use std::io::Write;
    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    e.write_all(data).unwrap();
    e.finish().unwrap()
}

/// A little-endian ELF64 executable with one PT_LOAD per `(paddr, vaddr, data, memsz, flags)`.
fn elf(machine: u16, entry: u64, segs: &[(u64, u64, Vec<u8>, u64, u32)]) -> Vec<u8> {
    let mut f = vec![0u8; 64 + 56 * segs.len()];
    f[0..4].copy_from_slice(b"\x7fELF");
    f[4] = 2;
    f[5] = 1;
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes());
    f[18..20].copy_from_slice(&machine.to_le_bytes());
    f[20..24].copy_from_slice(&1u32.to_le_bytes());
    f[24..32].copy_from_slice(&entry.to_le_bytes());
    f[32..40].copy_from_slice(&64u64.to_le_bytes());
    f[52..54].copy_from_slice(&64u16.to_le_bytes());
    f[54..56].copy_from_slice(&56u16.to_le_bytes());
    f[56..58].copy_from_slice(&(segs.len() as u16).to_le_bytes());
    for (i, (paddr, vaddr, d, memsz, flags)) in segs.iter().enumerate() {
        let off = f.len() as u64;
        let h = 64 + 56 * i;
        f[h..h + 4].copy_from_slice(&1u32.to_le_bytes());
        f[h + 4..h + 8].copy_from_slice(&flags.to_le_bytes());
        f[h + 8..h + 16].copy_from_slice(&off.to_le_bytes());
        f[h + 16..h + 24].copy_from_slice(&vaddr.to_le_bytes());
        f[h + 24..h + 32].copy_from_slice(&paddr.to_le_bytes());
        f[h + 32..h + 40].copy_from_slice(&(d.len() as u64).to_le_bytes());
        f[h + 40..h + 48].copy_from_slice(&memsz.to_le_bytes());
        f[h + 48..h + 56].copy_from_slice(&0x1000u64.to_le_bytes());
        f.extend_from_slice(d);
    }
    f
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn model(name: &str) -> ArmCpuModel {
    ArmCpuModel::by_name(name).unwrap()
}

/// The default board: one cortex-a57 and 128 MiB.
fn a57() -> VirtConfig {
    VirtConfig::new(model("cortex-a57"))
}

fn read(m: &VirtMachine, addr: u64, len: usize) -> Vec<u8> {
    let mut b = vec![0; len];
    assert!(m.memory_as().read(addr, U, &mut b).is_ok(), "read at {addr:#x}");
    b
}

fn r32(m: &VirtMachine, addr: u64) -> u32 {
    u32::from_le_bytes(read(m, addr, 4).try_into().unwrap())
}

// ---- A small DTB reader, independent of crate::fdt.

fn be32(b: &[u8], o: usize) -> u32 {
    u32::from_be_bytes(b[o..o + 4].try_into().unwrap())
}

fn cstr(b: &[u8], o: usize) -> String {
    let end = b[o..].iter().position(|&c| c == 0).unwrap() + o;
    String::from_utf8(b[o..end].to_vec()).unwrap()
}

#[derive(Debug, PartialEq, Eq)]
struct Node {
    path: String,
    props: Vec<(String, Vec<u8>)>,
}

/// The nodes of `blob` in order with their properties, without NOPs.
fn parse(blob: &[u8]) -> Vec<Node> {
    assert_eq!(be32(blob, 0), 0xd00d_feed);
    let off_struct = be32(blob, 8) as usize;
    let off_strings = be32(blob, 12) as usize;
    let mut p = off_struct;
    let mut stack: Vec<String> = Vec::new();
    let mut out: Vec<Node> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    loop {
        let tok = be32(blob, p);
        p += 4;
        match tok {
            1 => {
                let name = cstr(blob, p);
                p += (name.len() + 1).div_ceil(4) * 4;
                let path = match stack.last() {
                    None => "/".to_string(),
                    Some(parent) if parent == "/" => format!("/{name}"),
                    Some(parent) => format!("{parent}/{name}"),
                };
                stack.push(path.clone());
                current.push(out.len());
                out.push(Node { path, props: Vec::new() });
            }
            2 => {
                stack.pop();
                current.pop();
            }
            3 => {
                let len = be32(blob, p) as usize;
                let nameoff = be32(blob, p + 4) as usize;
                p += 8;
                let val = blob[p..p + len].to_vec();
                p += len.div_ceil(4) * 4;
                let name = cstr(blob, off_strings + nameoff);
                out[*current.last().unwrap()].props.push((name, val));
            }
            4 => {}
            9 => break,
            t => panic!("bad token {t:#x} at {p:#x}"),
        }
    }
    out
}

/// The nodes QEMU's dumps have for devices the board does not model: the PL061 with its key,
/// and the secure PL061 of `secure=on` with its poweroff and restart lines.
const MISSING: [&str; 5] =
    ["/pl061@9030000", "/gpio-keys", "/pl061@90b0000", "/gpio-poweroff", "/gpio-restart"];

fn strip(nodes: Vec<Node>) -> Vec<Node> {
    nodes
        .into_iter()
        .filter(|n| !MISSING.iter().any(|m| n.path == *m || n.path.starts_with(&format!("{m}/"))))
        .collect()
}

fn header(blob: &[u8]) -> (u32, u32, u32) {
    // totalsize, version, boot_cpuid_phys.
    (be32(blob, 4), be32(blob, 20), be32(blob, 28))
}

/// Compares the board's tree with a dump of gen.sh's `$M`, which has `its=off`.
fn compare_with_qemu(mut cfg: VirtConfig, dump: &str) {
    cfg.msi = VirtMsi::Off;
    compare(cfg, dump);
}

fn compare(cfg: VirtConfig, dump: &str) {
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    let ours = m.fdt().as_bytes().to_vec();
    let qemu = gunzip_file(dump);
    assert_eq!(header(&ours), header(&qemu), "{dump}");
    let (a, b) = (parse(&ours), strip(parse(&qemu)));
    for (x, y) in a.iter().zip(b.iter()) {
        assert_eq!(x, y, "{dump}");
    }
    assert_eq!(a.len(), b.len(), "{dump}");
}

#[test]
fn dtb_a57_matches_qemu() {
    compare_with_qemu(VirtConfig::new(model("cortex-a57")), "virt-a57.dtb.gz");
}

#[test]
fn dtb_a57_smp2_linux_matches_qemu() {
    let image = TmpFile::new("smp2", "Image", &fake_image(0x20000));
    let initrd = TmpFile::new("smp2", "initrd", &[b'r'; 1000]);
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.smp = 2;
    cfg.ram_size = 512 * MIB;
    cfg.kernel = Some(image.path());
    cfg.initrd = Some(initrd.path());
    cfg.append = Some("console=ttyAMA0 root=/dev/vda".to_string());
    compare_with_qemu(cfg, "virt-a57-smp2-linux.dtb.gz");
}

#[test]
fn dtb_max_smp3_matches_qemu() {
    let mut cfg = VirtConfig::new(model("max"));
    cfg.smp = 3;
    cfg.ram_size = 1024 * MIB;
    compare_with_qemu(cfg, "virt-max-smp3.dtb.gz");
}

#[test]
fn dtb_a76_smp20_matches_qemu() {
    let mut cfg = VirtConfig::new(model("cortex-a76"));
    cfg.smp = 20;
    cfg.ram_size = 256 * MIB;
    compare_with_qemu(cfg, "virt-a76-smp20.dtb.gz");
}

#[test]
fn dtb_a57_its_matches_qemu() {
    // The default msi=auto: the ITS node under the GIC and the msi-map of the PCIe node.
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.smp = 2;
    compare(cfg, "virt-a57-smp2-its.dtb.gz");
}

#[test]
fn dtb_a57_smp130_matches_qemu() {
    // 123 redistributors fill the low region, so the other 7 go to the high one.
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.smp = 130;
    compare(cfg, "virt-a57-smp130-its.dtb.gz");
}

/// A chardev that drops what it is given.
struct NullSerial;

impl ruvm_hw_char::serial::SerialBackend for NullSerial {
    fn write(&self, bytes: &[u8]) -> usize {
        bytes.len()
    }
}

#[test]
fn dtb_max_el2_el3_matches_qemu() {
    // EL3 without firmware: the CPUs start in EL3, so the SMC conduit is disabled and there
    // is no /psci, but the CPU nodes still say psci. The secure UART, RAM and flash appear.
    let mut cfg = VirtConfig::new(model("max"));
    cfg.smp = 2;
    cfg.virtualization = true;
    cfg.secure = true;
    compare_with_qemu(cfg, "virt-max-el2-el3.dtb.gz");
}

#[test]
fn dtb_a57_el2_serial2_matches_qemu() {
    // EL2: PSCI through SMC, the GIC maintenance interrupt, and a second -serial adds the
    // non-secure UART1 before UART0.
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.smp = 2;
    cfg.virtualization = true;
    cfg.serial1 = Some(Arc::new(NullSerial));
    compare_with_qemu(cfg, "virt-a57-el2-serial2.dtb.gz");
}

#[test]
fn dtb_max_secure_bios_matches_qemu() {
    // secure=on with firmware: PSCI is the firmware's, so no enable-method either.
    let bios = TmpFile::new("secbios", "bios.fd", &[0; 4096]);
    let mut cfg = VirtConfig::new(model("max"));
    cfg.smp = 2;
    cfg.secure = true;
    cfg.firmware = Some(bios.path());
    compare_with_qemu(cfg, "virt-max-secure-bios.dtb.gz");
}

#[test]
fn bios_goes_into_flash0() {
    let image: Vec<u8> = (0..8192u32).map(|i| (i * 7) as u8).collect();
    let bios = TmpFile::new("bios", "bios.fd", &image);
    let mut cfg = a57();
    cfg.firmware = Some(bios.path());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_eq!(read(&m, 0, image.len()), image);
    // The rest of the flash reads as erased, zeros for the virt flash, and so does flash1.
    assert_eq!(read(&m, image.len() as u64, 16), [0; 16]);
    assert_eq!(read(&m, 0x0400_0000, 16), [0; 16]);
    // Firmware boot: the vCPUs start at 0.
    assert!(!m.boot_info().direct);

    let mut cfg = a57();
    cfg.firmware = Some("/nonexistent/ruvm.fd".to_string());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "Could not find ROM image '/nonexistent/ruvm.fd'"
    );
    let big = TmpFile::new("bigbios", "bios.fd", &vec![0; 64 * MIB as usize + 1]);
    let mut cfg = a57();
    cfg.firmware = Some(big.path());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!("Could not load ROM image '{}'", big.path())
    );
}

#[test]
fn user_dtb_matches_qemu_byte_for_byte() {
    let image = TmpFile::new("userdtb", "Image", &fake_image(0x20000));
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.ram_size = 256 * MIB;
    cfg.kernel = Some(image.path());
    cfg.append = Some("quiet".to_string());
    cfg.dtb = Some(data("user.dtb").to_str().unwrap().to_string());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    let qemu = gunzip_file("virt-user-dtb.dtb.gz");
    assert_eq!(m.fdt().as_bytes(), &qemu[..]);
    // The blob is in RAM, 2 MiB aligned after the kernel.
    let dtb = m.boot_info().dtb_start;
    assert_eq!(read(&m, dtb, qemu.len()), qemu);
}

#[test]
fn dtb_errors() {
    let mut cfg = VirtConfig::new(model("cortex-a57"));
    cfg.dtb = Some("/nonexistent/ruvm.dtb".to_string());
    let mut m = VirtMachine::new(cfg).unwrap();
    assert_eq!(m.machine_done().unwrap_err(), "Couldn't open dtb file /nonexistent/ruvm.dtb");
}

#[test]
fn memory_map() {
    let mut m = VirtMachine::new(VirtConfig::default()).unwrap();
    m.machine_done().unwrap();
    // RAM.
    let ram = m.ram_ranges();
    let names: Vec<_> =
        ram.iter().map(|r| (r.addr, r.size, r.block.name().to_string(), r.rom_device)).collect();
    assert_eq!(
        names,
        [
            (0, 64 * MIB, "virt.flash0".to_string(), true),
            (0x0400_0000, 64 * MIB, "virt.flash1".to_string(), true),
            (VIRT_MEM, 128 * MIB, "mach-virt.ram".to_string(), false),
        ]
    );
    // GICD_PIDR2: architecture revision 3.
    assert_eq!(r32(&m, VIRT_GIC_DIST + 0xffe8), 0x3b);
    // The PrimeCell IDs of the PL011 and the PL031.
    assert_eq!(r32(&m, VIRT_UART + 0xfe0), 0x11);
    assert_eq!(r32(&m, VIRT_RTC + 0xfe0), 0x31);
    // RTCDR counts the seconds since the epoch: the host date, not twice it.
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    assert!(u64::from(r32(&m, VIRT_RTC)).abs_diff(now.as_secs()) < 60, "RTCDR is the date");
    // Every virtio-mmio transport says "virt".
    for i in 0..32 {
        assert_eq!(r32(&m, VIRT_MMIO + i * 0x200), 0x7472_6976);
    }
    // fw_cfg: select the signature, read it from the data register, and check the DMA
    // register's signature.
    assert!(m.memory_as().write(VIRT_FW_CFG + 8, U, &[0, 0]).is_ok());
    assert_eq!(read(&m, VIRT_FW_CFG, 4), b"QEMU");
    assert_eq!(read(&m, VIRT_FW_CFG + 16, 8), b"QEMU CFG");
    // No kernel: firmware boot, the device tree at the base of RAM.
    assert!(!m.boot_info().direct);
    assert_eq!(m.boot_info().dtb_start, VIRT_MEM);
    assert_eq!(read(&m, VIRT_MEM, 4), [0xd0, 0x0d, 0xfe, 0xed]);
    assert_eq!(m.heap_info(), (VIRT_MEM + MIB, VIRT_MEM + 128 * MIB));
}

#[test]
fn config_errors() {
    let mut cfg = a57();
    cfg.smp = 124;
    cfg.highmem.redists = false;
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "Number of SMP CPUs requested (124) exceeds max CPUs supported by machine 'mach-virt' \
         (123)\nTry 'highmem-redists=on' for more CPUs"
    );
    // The high region has room for 512 more.
    let mut cfg = a57();
    cfg.max_cpus = Some(636);
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        "Number of SMP CPUs requested (636) exceeds max CPUs supported by machine 'mach-virt' \
         (635)"
    );
    let mut cfg = a57();
    cfg.msi = VirtMsi::Gicv2m;
    assert_eq!(VirtMachine::new(cfg).unwrap_err(), "msi=gicv2m is not supported by ruvm yet");
    let mut cfg = a57();
    let bits = cfg.cpu.pamax();
    cfg.ram_size = 1 << bits;
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!("Addressing limited to {bits} bits, but memory exceeds it by 1073741824 bytes")
    );
}

#[test]
fn el2_and_el3_are_taken_away() {
    let m = VirtMachine::new(VirtConfig::new(model("max").with_el2().with_el3())).unwrap();
    assert!(!m.cpu_model().features.el2);
    assert!(!m.cpu_model().features.el3);
    assert_eq!(m.cpu_model().id_aa64pfr0 & 0xff00, 0);
}

#[test]
fn kernel_errors() {
    let mut cfg = a57();
    cfg.kernel = Some("/nonexistent/Image".to_string());
    assert_eq!(VirtMachine::new(cfg).unwrap_err(), "could not load kernel '/nonexistent/Image'");

    let big = TmpFile::new("kerr", "big", &fake_image(256 * MIB));
    let mut cfg = a57();
    cfg.kernel = Some(big.path());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!(
            "kernel '{}' is too large to fit in RAM (kernel size 268435456, RAM size 134217728)",
            big.path()
        )
    );

    let image = TmpFile::new("kerr", "Image", &fake_image(0x20000));
    let mut cfg = a57();
    cfg.kernel = Some(image.path());
    cfg.initrd = Some("/nonexistent/initrd".to_string());
    assert_eq!(VirtMachine::new(cfg).unwrap_err(), "could not load initrd '/nonexistent/initrd'");

    let x86 = TmpFile::new("kerr", "x86.elf", &elf(62, 0x4020_0000, &[]));
    let mut cfg = a57();
    cfg.kernel = Some(x86.path());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!("Couldn't load elf '{}': The image is from incompatible architecture", x86.path())
    );

    let empty = TmpFile::new("kerr", "empty.elf", &elf(183, 0x4020_0000, &[]));
    let mut cfg = a57();
    cfg.kernel = Some(empty.path());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!("Couldn't load elf '{}': Failed to load ELF", empty.path())
    );

    let zero = elf(183, 0x4020_0000, &[(0x4020_0000, 0x4020_0000, Vec::new(), 0, 5)]);
    let zero = TmpFile::new("kerr", "zero.elf", &zero);
    let mut cfg = a57();
    cfg.kernel = Some(zero.path());
    assert_eq!(
        VirtMachine::new(cfg).unwrap_err(),
        format!("Couldn't load elf '{}': No error", zero.path())
    );
}

#[test]
fn image_boot() {
    for gz in [false, true] {
        let raw = fake_image(0x20000);
        let image = TmpFile::new("image", "Image", &if gz { gzip(&raw) } else { raw.clone() });
        let mut cfg = a57();
        cfg.kernel = Some(image.path());
        let mut m = VirtMachine::new(cfg).unwrap();
        m.machine_done().unwrap();
        let info = m.boot_info().clone();
        assert!(info.direct && info.is_linux);
        // text_offset 0 is below the boot stub, so the kernel goes 2 MiB up.
        assert_eq!(info.entry, VIRT_MEM + 2 * MIB);
        assert_eq!(info.initrd_start, VIRT_MEM + 64 * MIB);
        assert_eq!(info.dtb_start, VIRT_MEM + 64 * MIB);
        assert_eq!(read(&m, info.entry, 4096), raw);
        let stub = read(&m, VIRT_MEM, 40);
        let w: Vec<u32> =
            stub.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(
            w,
            [
                0x5800_00c0,
                0xaa1f_03e1,
                0xaa1f_03e2,
                0xaa1f_03e3,
                0x5800_0084,
                0xd61f_0080,
                0x4400_0000,
                0,
                0x4020_0000,
                0
            ]
        );
        assert_eq!(read(&m, info.dtb_start, 4), [0xd0, 0x0d, 0xfe, 0xed]);
        let names: Vec<_> = m.roms().iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["bootloader", &image.path(), "dtb"]);
        assert!(m.messages().is_empty());
    }
}

#[test]
fn bad_gzip_falls_back_to_the_raw_file() {
    let mut z = gzip(&fake_image(0x20000));
    z[2] = 7;
    let image = TmpFile::new("badgz", "Image", &z);
    let mut cfg = a57();
    cfg.kernel = Some(image.path());
    let m = VirtMachine::new(cfg).unwrap();
    assert_eq!(
        m.messages(),
        [
            "Error: Bad gzipped data\n".to_string(),
            format!("{}: unable to decompress gzipped kernel file", image.path())
        ]
    );
    // No arm64 header: the default offset and the file size.
    assert_eq!(m.boot_info().entry, VIRT_MEM + 0x80000);
}

#[test]
fn elf_boot() {
    let code = words(&[0xd503_207f; 4]);
    let k = elf(
        183,
        0xffff_0000_0000_0008,
        &[
            (0x4020_0000, 0xffff_0000_0000_0000, code.clone(), 0x1000, 5),
            (0x4030_0000, 0x4030_0000, vec![0xaa; 16], 0x100, 6),
        ],
    );
    let k = TmpFile::new("elf", "kernel.elf", &k);
    let mut cfg = a57();
    cfg.kernel = Some(k.path());
    let mut m = VirtMachine::new(cfg).unwrap();
    // Make the BSS dirty, so the reset must clear it.
    assert!(m.memory_as().write(0x4030_0010, U, &[0x55; 16]).is_ok());
    m.machine_done().unwrap();
    let info = m.boot_info().clone();
    assert!(info.direct && !info.is_linux);
    // The entry point is translated from the virtual address of the executable segment.
    assert_eq!(info.entry, 0x4020_0008);
    // There is room below the kernel, so the device tree goes to the base of RAM.
    assert_eq!((info.dtb_start, info.dtb_limit), (VIRT_MEM, 0x4020_0000));
    assert_eq!(read(&m, VIRT_MEM, 4), [0xd0, 0x0d, 0xfe, 0xed]);
    assert_eq!(read(&m, 0x4020_0000, 16), code);
    assert_eq!(read(&m, 0x4030_0000, 32), [[0xaa; 16], [0; 16]].concat());
    assert_eq!(m.heap_info(), (0x4030_0100, VIRT_MEM + 128 * MIB));
    let names: Vec<_> = m.roms().iter().map(|r| r.name.clone()).collect();
    assert_eq!(
        names,
        [
            "dtb".to_string(),
            format!("{} ELF program header segment 0", k.path()),
            format!("{} ELF program header segment 1", k.path()),
        ]
    );

    // No room for the 1 MiB board blob below a kernel at 0x40080000: no device tree, and no
    // error either.
    let k2 = elf(183, 0x4008_0000, &[(0x4008_0000, 0x4008_0000, code.clone(), 0x10, 5)]);
    let k2 = TmpFile::new("elf", "low.elf", &k2);
    let mut cfg = a57();
    cfg.kernel = Some(k2.path());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_eq!(m.roms().len(), 1);
}

#[test]
fn overlapping_roms() {
    let k = elf(
        183,
        0x4020_0000,
        &[
            (0x4020_0000, 0x4020_0000, vec![1; 0x100], 0x100, 5),
            (0x4020_0080, 0x4020_0080, vec![2; 0x100], 0x100, 5),
        ],
    );
    let k = TmpFile::new("overlap", "k.elf", &k);
    let mut cfg = a57();
    cfg.kernel = Some(k.path());
    let mut m = VirtMachine::new(cfg).unwrap();
    let e = m.machine_done().unwrap_err();
    assert!(e.starts_with("Some ROM regions are overlapping\n"), "{e}");
    assert!(e.ends_with(&format!(
        "The following two regions overlap (in the cpu-memory-0 address space):\n  \
         {p} ELF program header segment 0 (addresses 0x0000000040200000 - 0x0000000040200100)\n  \
         {p} ELF program header segment 1 (addresses 0x0000000040200080 - 0x0000000040200180)",
        p = k.path()
    )));
}

#[test]
fn virtio_plugging() {
    let rng =
        || Box::new(VirtioRng::new(Box::new(RandomFile::default()), VirtioRngConf::default()));
    let mut m = VirtMachine::new(VirtConfig::default()).unwrap();
    assert_eq!(m.attach_virtio(rng()).unwrap(), 31);
    assert_eq!(m.attach_virtio(rng()).unwrap(), 30);
    assert_eq!(
        m.attach_virtio_at(31, rng(), true).unwrap_err(),
        "Bus 'virtio-mmio-bus.31' does not support hotplugging"
    );
    assert_eq!(
        m.attach_virtio_at(32, rng(), true).unwrap_err(),
        "Bus 'virtio-mmio-bus.32' not found"
    );
    m.machine_done().unwrap();
    // Device ID 4 is the entropy device; the empty transports report 0.
    assert_eq!(r32(&m, VIRT_MMIO + 31 * 0x200 + 8), 4);
    assert_eq!(r32(&m, VIRT_MMIO + 30 * 0x200 + 8), 4);
    assert_eq!(r32(&m, VIRT_MMIO + 29 * 0x200 + 8), 0);
    assert_eq!(
        m.attach_virtio(rng()).unwrap_err(),
        "virtio-mmio devices must be plugged before machine_done"
    );
}

#[test]
fn pcie_host() {
    let rng =
        || Box::new(VirtioRng::new(Box::new(RandomFile::default()), VirtioRngConf::default()));
    let mut m = VirtMachine::new(VirtConfig::default()).unwrap();
    let ecam = m.memmap().ecam;
    assert_eq!((ecam.base, ecam.size), (0x40_1000_0000, 256 * MIB));
    let props = VirtioPciProps::default();
    m.attach_virtio_pci(rng(), Some(2 << 3), &props).unwrap();
    m.attach_virtio_pci(rng(), None, &props).unwrap();
    m.machine_done().unwrap();
    // The root function at 00:00.0 is gpex-root, 1b36:0008, and the two rng functions are
    // the transitional virtio entropy device, 1af4:1005, at 00:02.0 and the first free slot.
    assert_eq!(r32(&m, ecam.base), 0x0008_1b36);
    assert_eq!(r32(&m, ecam.base + (2 << 15)), 0x1005_1af4);
    assert_eq!(r32(&m, ecam.base + (1 << 15)), 0x1005_1af4);
    assert_eq!(r32(&m, ecam.base + (3 << 15)), u32::MAX);
    // The config space of 00:01.0 as QEMU 11.1 shows it with `xp /44wx 0x4010008000` before
    // the guest runs: the BARs, the virtio vendor capabilities from 0x40 and the MSI-X
    // capability at 0x98, which is there without an MSI controller too.
    let want: [u32; 44] = [
        0x10051af4, 0x00100000, 0x00ff0000, 0x00000000, 0x00000001, 0x00000000, 0x00000000,
        0x00000000, 0x0000000c, 0x00000000, 0x00000000, 0x00041af4, 0x00000000, 0x00000098,
        0x00000000, 0x00000100, 0x01100009, 0x00000004, 0x00000000, 0x00001000, 0x03104009,
        0x00000004, 0x00001000, 0x00001000, 0x04105009, 0x00000004, 0x00002000, 0x00001000,
        0x02146009, 0x00000004, 0x00003000, 0x00001000, 0x00000004, 0x05147009, 0x00000000,
        0x00000000, 0x00000000, 0x00000000, 0x00018411, 0x00000001, 0x00000801, 0x00000000,
        0x00000000, 0x00000000,
    ];
    let got: Vec<u32> = (0..44).map(|i| r32(&m, ecam.base + (1 << 15) + 4 * i)).collect();
    assert_eq!(got, want);
    // Nothing is mapped in the windows yet, which read as all ones.
    assert_eq!(r32(&m, VIRT_PCIE_MMIO), u32::MAX);
    assert_eq!(r32(&m, VIRT_PCIE_PIO), u32::MAX);
    assert_eq!(r32(&m, 0x80_0000_0000), u32::MAX);
    // EDK2 puts the I/O BAR of the first function at port 0, which the board allows. Legacy
    // register 12 is then the size of the selected queue, 8 for virtio-rng.
    w(&m, ecam.base + (1 << 15) + 0x10, 4, 0);
    w(&m, ecam.base + (1 << 15) + 4, 2, 0x1);
    assert_eq!(read(&m, VIRT_PCIE_PIO + 12, 2), [8, 0]);

    // highmem-ecam=off: the 16 bus ECAM below 4 GiB.
    let mut cfg = a57();
    cfg.highmem.ecam = false;
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_eq!(m.memmap().ecam.base, 0x3f00_0000);
    assert_eq!(r32(&m, 0x3f00_0000), 0x0008_1b36);
    let pcie = parse(m.fdt().as_bytes()).into_iter().find(|n| n.path == "/pcie@10000000");
    let pcie = pcie.unwrap();
    let bus_range = pcie.props.iter().find(|(n, _)| n == "bus-range").unwrap();
    assert_eq!(bus_range.1, [0, 0, 0, 0, 0, 0, 0, 15]);
}

fn w(m: &VirtMachine, addr: u64, size: u32, v: u64) {
    let r = m.memory_as().store(addr, size, v, Endian::Little, U);
    assert!(r.is_ok(), "write of {addr:#x} failed");
}

#[test]
fn msi_through_the_its() {
    let rng =
        || Box::new(VirtioRng::new(Box::new(RandomFile::default()), VirtioRngConf::default()));
    let mut m = VirtMachine::new(VirtConfig::default()).unwrap();
    let props = VirtioPciProps::default();
    let dev = m.attach_virtio_pci(rng(), Some(2 << 3), &props).unwrap();
    let other = m.attach_virtio_pci(rng(), Some(3 << 3), &props).unwrap();
    m.machine_done().unwrap();
    // GITS_TYPER: physical LPIs, 16 bits of device and event ID.
    let typer = u64::from(r32(&m, VIRT_GIC_ITS + 8)) | u64::from(r32(&m, VIRT_GIC_ITS + 12)) << 32;
    assert_eq!(typer, (1 << 36) | (0xf << 32) | (0xf << 13) | (0xf << 8) | 0xb1);

    // LPI 8192 enabled in the property table, and the LPIs of CPU 0 on.
    let (propbase, pendbase) = (VIRT_MEM + 0x100_0000, VIRT_MEM + 0x101_0000);
    let (dt, ct, cmdq, itt) = (
        VIRT_MEM + 0x110_0000,
        VIRT_MEM + 0x120_0000,
        VIRT_MEM + 0x130_0000,
        VIRT_MEM + 0x140_0000,
    );
    w(&m, propbase, 1, 0xa1);
    w(&m, VIRT_GIC_REDIST + 0x70, 8, propbase | 0xf);
    w(&m, VIRT_GIC_REDIST + 0x78, 8, pendbase);
    w(&m, VIRT_GIC_REDIST, 4, 1);
    // The device and collection tables, the command queue, and the ITS on.
    for (reg, base) in [(0x100, dt), (0x108, ct)] {
        let baser = u64::from(r32(&m, VIRT_GIC_ITS + reg + 4)) << 32;
        w(&m, VIRT_GIC_ITS + reg, 8, baser | (1 << 63) | base);
    }
    w(&m, VIRT_GIC_ITS + 0x80, 8, (1 << 63) | cmdq);
    w(&m, VIRT_GIC_ITS, 4, 1);
    // MAPD of 00:02.0 with five bits of event ID, MAPC of collection 0 to CPU 0, and MAPTI of
    // event 0 to LPI 8192.
    let devid = u64::from(dev.pci_dev().requester_id());
    assert_eq!(devid, 0x10);
    let cmds: [[u64; 4]; 3] = [
        [0x08 | (devid << 32), 4, (1 << 63) | itt, 0],
        [0x09, 0, 1 << 63, 0],
        [0x0a | (devid << 32), 8192 << 32, 0, 0],
    ];
    for (i, c) in cmds.iter().enumerate() {
        for (j, word) in c.iter().enumerate() {
            w(&m, cmdq + 32 * i as u64 + 8 * j as u64, 8, *word);
        }
    }
    w(&m, VIRT_GIC_ITS + 0x88, 8, 32 * cmds.len() as u64);
    assert_eq!(r32(&m, VIRT_GIC_ITS + 0x90), 32 * cmds.len() as u32, "GITS_CREADR");

    let ecam = m.memmap().ecam.base;
    let msg = MsiMessage { address: VIRT_GIC_ITS + 0x1_0040, data: 0 };
    let lpi_pending = |m: &VirtMachine| read(m, pendbase + 8192 / 8, 1)[0] & 1 != 0;
    // Not a bus master yet, so nothing is sent.
    dev.pci_dev().msi_send_message(msg);
    assert!(!lpi_pending(&m));
    // The other function's requester ID has no device table entry.
    w(&m, ecam + (3 << 15) + 4, 2, 0x6);
    other.pci_dev().msi_send_message(msg);
    assert!(!lpi_pending(&m));
    w(&m, ecam + (2 << 15) + 4, 2, 0x6);
    dev.pci_dev().msi_send_message(msg);
    assert!(lpi_pending(&m));

    // msi=off: no ITS, but the GIC keeps its LPIs (GICD_TYPER.LPIS) as in QEMU.
    let mut cfg = a57();
    cfg.msi = VirtMsi::Off;
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_ne!(r32(&m, VIRT_GIC_DIST + 4) & (1 << 17), 0);
    let mut b = [0; 4];
    assert!(!m.memory_as().read(VIRT_GIC_ITS + 8, U, &mut b).is_ok());
    let nodes = parse(m.fdt().as_bytes());
    assert!(!nodes.iter().any(|n| n.path.contains("/its@")));
    let pcie = nodes.iter().find(|n| n.path == "/pcie@10000000").unwrap();
    assert!(!pcie.props.iter().any(|(n, _)| n == "msi-map"));
}

fn movz(rd: u32, imm: u32, hw: u32) -> u32 {
    0xd280_0000 | (hw << 21) | ((imm & 0xffff) << 5) | rd
}

fn movk(rd: u32, imm: u32, hw: u32) -> u32 {
    0xf280_0000 | (hw << 21) | ((imm & 0xffff) << 5) | rd
}

fn mov64(rd: u32, v: u64) -> Vec<u32> {
    let mut out = vec![movz(rd, v as u32, 0)];
    for hw in 1..4 {
        let part = (v >> (16 * hw)) as u32 & 0xffff;
        if part != 0 {
            out.push(movk(rd, part, hw));
        }
    }
    out
}

fn mov(rd: u32, rm: u32) -> u32 {
    0xaa00_03e0 | (rm << 16) | rd
}

const HVC0: u32 = 0xd400_0002;
const WFI: u32 = 0xd503_207f;

/// A PSCI call with function `f` and arguments in X1 to X3, the result copied to `save`.
fn psci(f: u64, args: &[u64], save: u32) -> Vec<u32> {
    let mut c = mov64(0, f);
    for (i, &a) in args.iter().enumerate() {
        c.extend(mov64(i as u32 + 1, a));
    }
    c.push(HVC0);
    c.push(mov(save, 0));
    c
}

fn run(v: &mut Vcpu, m: &VirtMachine) -> Option<VirtRequest> {
    for _ in 0..1000 {
        let r = cpu_exec(&mut v.cpu());
        if let Some(req) = m.take_request() {
            return Some(req);
        }
        if r == excp::HLT {
            return None;
        }
        assert_ne!(r, excp::DEBUG);
    }
    panic!("the vCPU did not stop");
}

#[test]
fn psci_cpu_on_between_two_vcpus() {
    const SECONDARY: u64 = 0x4020_1000;
    let mut cpu0 = Vec::new();
    cpu0.extend(psci(0x8400_0004, &[1, 0], 10));
    cpu0.extend(psci(0xc400_0003, &[1, SECONDARY, 0x1234], 11));
    cpu0.extend(psci(0x8400_0004, &[1, 0], 12));
    cpu0.extend(psci(0xc400_0003, &[5, SECONDARY, 0], 13));
    cpu0.extend(psci(0xc400_0003, &[0, SECONDARY, 0], 14));
    cpu0.push(WFI);
    let mut cpu1 = vec![mov(20, 0)];
    cpu1.extend(psci(0x8400_0008, &[], 21));
    cpu1.push(WFI);
    let k = elf(
        183,
        0x4020_0000,
        &[
            (0x4020_0000, 0x4020_0000, words(&cpu0), 0x1000, 5),
            (SECONDARY, SECONDARY, words(&cpu1), 0x1000, 5),
        ],
    );
    let k = TmpFile::new("psci", "psci.elf", &k);
    let mut cfg = a57();
    cfg.smp = 2;
    cfg.kernel = Some(k.path());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    let jit = new_jit();
    let mut v = m.create_vcpus(&jit).unwrap();
    assert_eq!(m.arm().power_state(0), PSCI_ON);
    assert_eq!(m.arm().power_state(1), PSCI_OFF);
    assert_eq!(m.arm().mp_affinity(1), 1);

    assert_eq!(run(&mut v[0], &m), None);
    let st = save_vcpu(&v[0]);
    // AFFINITY_INFO says off, CPU_ON succeeds, then it is pending; an unknown MPIDR is
    // refused and the running CPU is already on.
    assert_eq!(st.xregs[10], 1);
    assert_eq!(st.xregs[11], 0);
    assert_eq!(st.xregs[12], 2);
    assert_eq!(st.xregs[13] as i64, -2);
    assert_eq!(st.xregs[14] as i64, -4);

    v[1].cpu().process_queued_cpu_work();
    assert_eq!(m.arm().power_state(1), PSCI_ON);
    assert_eq!(run(&mut v[1], &m), Some(VirtRequest::Shutdown));
    assert_eq!(save_vcpu(&v[1]).xregs[20], 0x1234);
}

/// Collects the semihosting console and the exit status.
#[derive(Default)]
struct Host {
    console: Mutex<Vec<u8>>,
    exit: Mutex<Option<u32>>,
}

impl SemihostingHost for Host {
    fn console_write(&self, buf: &[u8]) -> usize {
        self.console.lock().unwrap().extend_from_slice(buf);
        buf.len()
    }

    fn console_read(&self) -> u8 {
        0
    }

    fn exit(&self, code: u32) {
        *self.exit.lock().unwrap() = Some(code);
    }

    fn heap_info(&self) -> (u64, u64) {
        (0, 0)
    }
}

/// tests/tcg/multiarch/system/hello.c with the tests/tcg/aarch64/system boot code, as
/// `qemu-system-aarch64 -M virt -cpu max -semihosting -kernel hello` runs it.
#[test]
fn tcg_system_hello() {
    let hello = TmpFile::new("hello", "hello", &gunzip_file("hello.gz"));
    let host = Arc::new(Host::default());
    let mut cfg = VirtConfig::new(model("max"));
    cfg.kernel = Some(hello.path());
    cfg.semihosting = Some(host.clone());
    let mut m = VirtMachine::new(cfg).unwrap();
    m.machine_done().unwrap();
    assert_eq!(m.boot_info().entry, 0x4000_17b0);
    // The device tree would go to 0, which is not RAM here.
    assert_eq!(m.boot_info().dtb_start, 0);
    assert_eq!(m.heap_info(), (0x402b_1000, 0x4800_0000));
    let jit = new_jit();
    let mut v = m.create_vcpus(&jit).unwrap();
    for _ in 0..100_000 {
        if host.exit.lock().unwrap().is_some() {
            break;
        }
        let r = cpu_exec(&mut v[0].cpu());
        assert_ne!(r, excp::DEBUG);
    }
    let out = String::from_utf8(host.console.lock().unwrap().clone()).unwrap();
    assert_eq!(out, "Hello World\n");
    assert_eq!(*host.exit.lock().unwrap(), Some(0));
}
