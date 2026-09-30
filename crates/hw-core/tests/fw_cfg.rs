// SPDX-License-Identifier: GPL-2.0-or-later

//! fw_cfg tests: tests/qtest/fw_cfg-test.c driven the way tests/qtest/libqos/fw_cfg.c drives
//! the port flavor, plus the file directory, DMA and the MMIO flavor.

use std::sync::{Arc, Mutex};

use ruvm_hw_core::fw_cfg::*;
use ruvm_mem::{Endian, MemTxAttrs, MmioOps, dispatch_read, dispatch_write};

const RAM_SIZE: u64 = 128 << 20;
const NB_CPUS: u16 = 1;
const MAX_CPUS: u16 = 1;
const NB_NODES: u64 = 0;
const BOOT_MENU: u16 = 0;

/// Flat guest RAM starting at 0.
struct Ram(Mutex<Vec<u8>>);

impl Ram {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Ram(Mutex::new(vec![0; size])))
    }

    fn bufwrite(&self, addr: u64, buf: &[u8]) {
        let a = addr as usize;
        self.0.lock().unwrap()[a..a + buf.len()].copy_from_slice(buf);
    }

    fn bufread(&self, addr: u64, len: usize) -> Vec<u8> {
        let a = addr as usize;
        self.0.lock().unwrap()[a..a + len].to_vec()
    }
}

impl DmaMemory for Ram {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        match m.get(a..a + buf.len()) {
            Some(src) => {
                buf.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        match m.get_mut(a..a + buf.len()) {
            Some(dst) => {
                dst.copy_from_slice(buf);
                true
            }
            None => false,
        }
    }
}

/// A PC: fw_cfg at 0x510 plus what `fw_cfg_arch_create()` adds.
struct Pc {
    io: FwCfgIo,
    ram: Arc<Ram>,
}

impl Pc {
    fn new(cfg: FwCfgMachineConfig) -> Self {
        let ram = Ram::new(0x10000);
        let io = fw_cfg_init_io_dma(FW_CFG_IO_BASE, ram.clone(), &cfg).unwrap();
        let s = io.state();
        s.add_i16(FW_CFG_NB_CPUS, NB_CPUS);
        s.add_i16(FW_CFG_MAX_CPUS, MAX_CPUS);
        s.add_i64(FW_CFG_RAM_SIZE, RAM_SIZE);
        let mut numa = vec![0u8; 8 * (1 + usize::from(MAX_CPUS) + NB_NODES as usize)];
        numa[..8].copy_from_slice(&NB_NODES.to_le_bytes());
        s.add_bytes(FW_CFG_NUMA, numa);
        s.reset();
        Pc { io, ram }
    }

    fn state(&self) -> &FwCfgState {
        self.io.state()
    }

    fn port(&self, port: u16) -> (&dyn MmioOps, u64) {
        let base = FW_CFG_IO_BASE as u16;
        if (base..base + 2).contains(&port) {
            return (&**self.io.comb_ops(), u64::from(port - base));
        }
        if (base + 4..base + 12).contains(&port) {
            return (&**self.io.dma_ops().unwrap(), u64::from(port - base - 4));
        }
        panic!("no fw_cfg port at {port:#x}");
    }

    fn out(&self, port: u16, size: u32, value: u64) -> bool {
        let (ops, off) = self.port(port);
        dispatch_write(ops, "fwcfg", off, size, value, Endian::Little, MemTxAttrs::UNSPECIFIED)
            .is_ok()
    }

    fn inp(&self, port: u16, size: u32) -> Option<u64> {
        let (ops, off) = self.port(port);
        let (v, r) =
            dispatch_read(ops, "fwcfg", off, size, Endian::Little, MemTxAttrs::UNSPECIFIED);
        r.is_ok().then_some(v)
    }

    fn outw(&self, port: u16, value: u16) {
        assert!(self.out(port, 2, u64::from(value)));
    }

    fn outl(&self, port: u16, value: u32) {
        assert!(self.out(port, 4, u64::from(value)));
    }

    fn inb(&self, port: u16) -> u8 {
        self.inp(port, 1).unwrap() as u8
    }

    /// `io_fw_cfg_select()`.
    fn select(&self, key: u16) {
        self.outw(FW_CFG_IO_BASE as u16, key);
    }

    /// `io_fw_cfg_read()`.
    fn read_data(&self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.inb(FW_CFG_IO_BASE as u16 + 1)).collect()
    }

    /// `qfw_cfg_get()`.
    fn get(&self, key: u16, len: usize) -> Vec<u8> {
        self.select(key);
        self.read_data(len)
    }

    fn get_u16(&self, key: u16) -> u16 {
        u16::from_le_bytes(self.get(key, 2).try_into().unwrap())
    }

    fn get_u32(&self, key: u16) -> u32 {
        u32::from_le_bytes(self.get(key, 4).try_into().unwrap())
    }

    fn get_u64(&self, key: u16) -> u64 {
        u64::from_le_bytes(self.get(key, 8).try_into().unwrap())
    }

    /// `find_pdir_entry()`: the selector and size of a file.
    fn find_file(&self, filename: &str) -> Option<(u16, u32)> {
        let count = u32::from_be_bytes(self.get(FW_CFG_FILE_DIR, 4).try_into().unwrap());
        let dir = self.get(FW_CFG_FILE_DIR, 4 + count as usize * FW_CFG_FILE_SIZE);
        dir[4..].chunks(FW_CFG_FILE_SIZE).find_map(|f| {
            let name = &f[8..];
            let end = name.iter().position(|&c| c == 0).unwrap();
            (&name[..end] == filename.as_bytes()).then(|| {
                let size = u32::from_be_bytes(f[0..4].try_into().unwrap());
                (u16::from_be_bytes([f[4], f[5]]), size)
            })
        })
    }

    /// `qfw_cfg_get_file()`: the file size and up to `buflen` bytes of it.
    fn get_file(&self, filename: &str, buflen: usize) -> (usize, Vec<u8>) {
        match self.find_file(filename) {
            Some((sel, len)) => (len as usize, self.get(sel, buflen.min(len as usize))),
            None => (0, Vec::new()),
        }
    }

    /// Points the DMA register at an access at `access_addr`, high half first, and returns the
    /// control word written back. The values are byte swapped like `cpu_to_be32()` in libqos,
    /// since the register is big endian and the port access little endian.
    fn dma(&self, access_addr: u64, control: u32, length: u32, address: u64) -> u32 {
        let mut access = Vec::new();
        access.extend_from_slice(&control.to_be_bytes());
        access.extend_from_slice(&length.to_be_bytes());
        access.extend_from_slice(&address.to_be_bytes());
        self.ram.bufwrite(access_addr, &access);
        let base = FW_CFG_IO_BASE as u16;
        self.outl(base + 4, ((access_addr >> 32) as u32).swap_bytes());
        self.outl(base + 8, (access_addr as u32).swap_bytes());
        u32::from_be_bytes(self.ram.bufread(access_addr, 4).try_into().unwrap())
    }
}

fn pc() -> Pc {
    Pc::new(FwCfgMachineConfig::default())
}

#[test]
fn signature() {
    assert_eq!(pc().get(FW_CFG_SIGNATURE, 4), b"QEMU");
}

#[test]
fn id() {
    assert_eq!(pc().get_u32(FW_CFG_ID), FW_CFG_VERSION | FW_CFG_VERSION_DMA);

    let props = FwCfgProps { dma_enabled: false, ..FwCfgProps::default() };
    let io = FwCfgIo::new(FW_CFG_IO_BASE, props, None, &FwCfgMachineConfig::default()).unwrap();
    assert!(io.dma_ops().is_none());
    assert_eq!(io.state().entry_data(FW_CFG_ID).unwrap(), FW_CFG_VERSION.to_le_bytes());
}

#[test]
fn uuid() {
    let uuid = [
        0x46, 0x00, 0xcb, 0x32, 0x38, 0xec, 0x4b, 0x2f, 0x8a, 0xcb, 0x81, 0xc6, 0xea, 0x54, 0xf2,
        0xd8,
    ];
    let pc = Pc::new(FwCfgMachineConfig { uuid, ..FwCfgMachineConfig::default() });
    assert_eq!(pc.get(FW_CFG_UUID, 16), uuid);
}

#[test]
fn ram_size() {
    assert_eq!(pc().get_u64(FW_CFG_RAM_SIZE), RAM_SIZE);
}

#[test]
fn nographic() {
    assert_eq!(pc().get_u16(FW_CFG_NOGRAPHIC), 0);
    let pc = Pc::new(FwCfgMachineConfig { enable_graphics: false, ..Default::default() });
    assert_eq!(pc.get_u16(FW_CFG_NOGRAPHIC), 1);
}

#[test]
fn nb_cpus() {
    assert_eq!(pc().get_u16(FW_CFG_NB_CPUS), NB_CPUS);
}

#[test]
fn max_cpus() {
    assert_eq!(pc().get_u16(FW_CFG_MAX_CPUS), MAX_CPUS);
}

#[test]
fn numa() {
    let pc = pc();
    assert_eq!(pc.get_u64(FW_CFG_NUMA), NB_NODES);
    let cpu_mask = pc.read_data(8 * usize::from(MAX_CPUS));
    let node_mask = pc.read_data(8 * NB_NODES as usize);
    assert!(cpu_mask.iter().all(|&b| b == 0));
    assert!(node_mask.is_empty());
    // Past the end, reads return zero.
    assert_eq!(pc.read_data(4), [0; 4]);
}

#[test]
fn boot_menu() {
    assert_eq!(pc().get_u16(FW_CFG_BOOT_MENU), BOOT_MENU);
    let pc = Pc::new(FwCfgMachineConfig { boot_menu: Some(true), ..Default::default() });
    assert_eq!(pc.get_u16(FW_CFG_BOOT_MENU), 1);
}

#[test]
fn reboot_timeout() {
    let pc = Pc::new(FwCfgMachineConfig { reboot_timeout: Some(15), ..Default::default() });
    let (size, data) = pc.get_file("etc/boot-fail-wait", 4);
    assert_eq!(size, 4);
    assert_eq!(u32::from_le_bytes(data.try_into().unwrap()), 15);
}

#[test]
fn no_reboot_timeout() {
    // The special value -1 means "don't reboot", and is also the default.
    for rt in [Some(-1), None] {
        let pc = Pc::new(FwCfgMachineConfig { reboot_timeout: rt, ..Default::default() });
        let (size, data) = pc.get_file("etc/boot-fail-wait", 4);
        assert_eq!(size, 4);
        assert_eq!(u32::from_le_bytes(data.try_into().unwrap()), u32::MAX);
    }
}

#[test]
fn bad_boot_options() {
    let ram: Arc<dyn DmaMemory> = Ram::new(16);
    for cfg in [
        FwCfgMachineConfig { reboot_timeout: Some(0x10000), ..Default::default() },
        FwCfgMachineConfig { reboot_timeout: Some(-2), ..Default::default() },
        FwCfgMachineConfig { splash_time: Some(-1), ..Default::default() },
        FwCfgMachineConfig { splash_time: Some(0x10000), ..Default::default() },
    ] {
        assert!(fw_cfg_init_io_dma(FW_CFG_IO_BASE, ram.clone(), &cfg).is_err());
    }
}

#[test]
fn splash_time() {
    let pc = Pc::new(FwCfgMachineConfig { splash_time: Some(12), ..Default::default() });
    let (size, data) = pc.get_file("etc/boot-menu-wait", 2);
    assert_eq!(size, 2);
    assert_eq!(u16::from_le_bytes(data.try_into().unwrap()), 12);
}

#[test]
fn file_dir_sorted() {
    let pc = pc();
    let s = pc.state();
    for name in ["z", "etc/zz", "a", "etc/acpi/tables", "etc/a", "bootorder", "etc/boot"] {
        s.add_file(name, name.as_bytes().to_vec()).unwrap();
    }
    let files = s.files();
    let names: Vec<&str> = files.iter().map(|f| f.0.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
    assert!(names.contains(&"etc/boot-fail-wait"));
    for (i, (name, sel, size)) in files.iter().enumerate() {
        // Selectors follow the slot, and each entry moved along with its directory slot.
        assert_eq!(*sel, FW_CFG_FILE_FIRST + i as u16);
        assert_eq!(pc.find_file(name), Some((*sel, *size)));
        if name != "etc/boot-fail-wait" {
            assert_eq!(pc.get(*sel, *size as usize), name.as_bytes());
        }
    }

    assert!(s.add_file("a", vec![1]).is_err());
    assert_eq!(s.files().len(), files.len());

    // Names are cut to 55 bytes in the directory.
    let long = "x".repeat(70);
    s.add_file(&long, vec![1]).unwrap();
    assert!(s.files().iter().any(|f| f.0 == "x".repeat(55)));
}

#[test]
fn file_dir_full() {
    let props = FwCfgProps { file_slots: FW_CFG_FILE_SLOTS_MIN, ..Default::default() };
    let s = FwCfgState::new(props, None).unwrap();
    for i in 0..FW_CFG_FILE_SLOTS_MIN {
        s.add_file(&format!("f{i:02}"), vec![]).unwrap();
    }
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.add_file("g", vec![])));
    assert!(r.is_err());

    let small = FwCfgProps { file_slots: FW_CFG_FILE_SLOTS_MIN - 1, ..Default::default() };
    assert!(FwCfgState::new(small, None).is_err());
    let big = FwCfgProps { file_slots: 0x3fe1, ..Default::default() };
    assert!(FwCfgState::new(big, None).is_err());
}

#[test]
fn modify() {
    let pc = pc();
    let s = pc.state();
    s.add_file("etc/x", vec![1, 2, 3]).unwrap();
    assert_eq!(s.modify_file("etc/x", vec![4, 5]).unwrap(), Some(vec![1, 2, 3]));
    assert_eq!(pc.get_file("etc/x", 8), (2, vec![4, 5]));
    assert_eq!(s.modify_file("etc/new", vec![6]).unwrap(), None);
    assert_eq!(pc.get_file("etc/new", 8), (1, vec![6]));

    s.modify_i16(FW_CFG_NB_CPUS, 4);
    assert_eq!(pc.get_u16(FW_CFG_NB_CPUS), 4);
    s.modify_i32(FW_CFG_ID, 7);
    assert_eq!(pc.get_u32(FW_CFG_ID), 7);
    s.modify_i64(FW_CFG_RAM_SIZE, 1 << 40);
    assert_eq!(pc.get_u64(FW_CFG_RAM_SIZE), 1 << 40);
    s.add_string(FW_CFG_KERNEL_CMDLINE, "quiet");
    assert_eq!(pc.get(FW_CFG_KERNEL_CMDLINE, 6), b"quiet\0");
    s.modify_string(FW_CFG_KERNEL_CMDLINE, "ro");
    assert_eq!(pc.get(FW_CFG_KERNEL_CMDLINE, 4), b"ro\0\0");

    s.machine_reset(b"/pci@i0cf8/ide@1,1\0".to_vec(), vec![]).unwrap();
    assert_eq!(pc.get_file("bootorder", 64).0, 19);
    assert_eq!(pc.get_file("bios-geometry", 64).0, 0);
}

#[test]
fn invalid_selector() {
    let pc = pc();
    let max = FW_CFG_FILE_FIRST + FW_CFG_FILE_SLOTS_DFLT;
    pc.select(max);
    assert_eq!(pc.state().cur_entry(), FW_CFG_INVALID);
    assert_eq!(pc.read_data(4), [0; 4]);
    // The write channel bit is masked for the lookup but kept in the current entry.
    pc.select(FW_CFG_SIGNATURE | FW_CFG_WRITE_CHANNEL);
    assert_eq!(pc.state().cur_entry(), FW_CFG_WRITE_CHANNEL);
    assert_eq!(pc.read_data(4), b"QEMU");
}

#[test]
fn arch_local() {
    let pc = pc();
    let key = FW_CFG_ARCH_LOCAL;
    pc.state().add_i32(key, 0x11223344);
    assert_eq!(pc.get_u32(key), 0x11223344);
    assert_eq!(pc.get(FW_CFG_SIGNATURE, 4), b"QEMU");
    assert_eq!(key_name(key), None);
    assert_eq!(key_name(FW_CFG_FILE_DIR), Some("file_dir"));
    assert_eq!(key_name(FW_CFG_FILE_FIRST), None);
}

#[test]
#[should_panic]
fn key_conflict() {
    pc().state().add_i16(FW_CFG_NB_CPUS, 2);
}

#[test]
fn port_access_rules() {
    let pc = pc();
    let base = FW_CFG_IO_BASE as u16;
    // Data writes are ignored.
    pc.select(FW_CFG_SIGNATURE);
    assert!(pc.out(base + 1, 1, 0xff));
    assert!(pc.out(base, 1, 0xff));
    assert_eq!(pc.read_data(4), b"QEMU");
    assert_eq!(pc.state().entry_data(FW_CFG_SIGNATURE).unwrap(), b"QEMU");
    // Only byte reads and word writes.
    assert!(pc.inp(base, 2).is_none());
    assert!(!pc.out(base, 4, 0));
    // The DMA register reads as "QEMU CFG".
    let hi = pc.inp(base + 4, 4).unwrap() as u32;
    let lo = pc.inp(base + 8, 4).unwrap() as u32;
    assert_eq!(hi.to_le_bytes(), *b"QEMU");
    assert_eq!(lo.to_le_bytes(), *b" CFG");
    assert_eq!(pc.inp(base + 5, 1), Some(u64::from(b'E')));
    // DMA writes must be 4 bytes at 0 or 4, or 8 bytes at 0.
    assert!(!pc.out(base + 4, 2, 0));
    assert!(!pc.out(base + 6, 1, 0));
}

#[test]
fn dma_read_skip_select() {
    let pc = pc();
    let s = pc.state();
    let data: Vec<u8> = (0..32).collect();
    s.add_file("etc/data", data.clone()).unwrap();
    let (sel, _) = pc.find_file("etc/data").unwrap();

    // Select and read.
    let ctl = pc.dma(
        0x100,
        (u32::from(sel) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_READ,
        8,
        0x1000,
    );
    assert_eq!(ctl, 0);
    assert_eq!(pc.ram.bufread(0x1000, 8), &data[..8]);

    // Skip 8 bytes, then read the next 4.
    assert_eq!(pc.dma(0x100, FW_CFG_DMA_CTL_SKIP, 8, 0), 0);
    assert_eq!(pc.dma(0x100, FW_CFG_DMA_CTL_READ, 4, 0x1100), 0);
    assert_eq!(pc.ram.bufread(0x1100, 4), &data[16..20]);

    // Reading past the end zero fills.
    pc.ram.bufwrite(0x1200, &[0xaa; 32]);
    assert_eq!(pc.dma(0x100, FW_CFG_DMA_CTL_READ, 32, 0x1200), 0);
    let got = pc.ram.bufread(0x1200, 32);
    assert_eq!(&got[..12], &data[20..]);
    assert!(got[12..].iter().all(|&b| b == 0));

    // No operation bit: nothing moves.
    let before = s.cur_offset();
    assert_eq!(pc.dma(0x100, 0, 4, 0x1300), 0);
    assert_eq!(s.cur_offset(), before);

    // The port and DMA share the offset.
    pc.dma(
        0x100,
        (u32::from(FW_CFG_SIGNATURE) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_SKIP,
        2,
        0,
    );
    assert_eq!(pc.read_data(2), b"MU");

    // A read going outside guest memory fails.
    let ctl = pc.dma(
        0x100,
        (u32::from(sel) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_READ,
        8,
        0xfffff,
    );
    assert_eq!(ctl, FW_CFG_DMA_CTL_ERROR);

    // The DMA address is reset after each transfer, and the 8 byte form works too.
    pc.ram.bufwrite(0x200, &{
        let mut a = Vec::new();
        a.extend_from_slice(
            &((u32::from(sel) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_READ).to_be_bytes(),
        );
        a.extend_from_slice(&4u32.to_be_bytes());
        a.extend_from_slice(&0x1400u64.to_be_bytes());
        a
    });
    assert!(pc.out(FW_CFG_IO_BASE as u16 + 4, 8, 0x200u64.swap_bytes()));
    assert_eq!(pc.ram.bufread(0x1400, 4), &data[..4]);
    assert_eq!(pc.ram.bufread(0x200, 4), [0; 4]);
}

#[test]
fn dma_write() {
    let pc = pc();
    let s = pc.state();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen2 = seen.clone();
    s.add_file_callback(
        "etc/rw",
        None,
        Some(Box::new(move |data: &[u8], off: u32, len: u32| {
            seen2.lock().unwrap().push((data.to_vec(), off, len));
        })),
        vec![0; 8],
        false,
    )
    .unwrap();
    s.add_file("etc/ro", vec![0; 8]).unwrap();
    let (rw, _) = pc.find_file("etc/rw").unwrap();
    let (ro, _) = pc.find_file("etc/ro").unwrap();

    pc.ram.bufwrite(0x1000, &[1, 2, 3, 4]);
    let ctl =
        pc.dma(0x100, (u32::from(rw) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_SKIP, 2, 0);
    assert_eq!(ctl, 0);
    assert_eq!(pc.dma(0x100, FW_CFG_DMA_CTL_WRITE, 4, 0x1000), 0);
    assert_eq!(s.entry_data(rw).unwrap(), [0, 0, 1, 2, 3, 4, 0, 0]);
    assert_eq!(*seen.lock().unwrap(), [(vec![0, 0, 1, 2, 3, 4, 0, 0], 2, 4)]);

    // A write running past the end fails as a whole.
    assert_eq!(pc.dma(0x100, FW_CFG_DMA_CTL_WRITE, 4, 0x1000), FW_CFG_DMA_CTL_ERROR);
    assert_eq!(s.entry_data(rw).unwrap(), [0, 0, 1, 2, 3, 4, 0, 0]);

    // Read only files cannot be written, and nothing is written without a write callback
    // through the data port either.
    let ctl = pc.dma(
        0x100,
        (u32::from(ro) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_WRITE,
        4,
        0x1000,
    );
    assert_eq!(ctl, FW_CFG_DMA_CTL_ERROR);
    assert_eq!(s.entry_data(ro).unwrap(), [0; 8]);
    pc.select(ro);
    assert!(pc.out(FW_CFG_IO_BASE as u16 + 1, 1, 0x55));
    assert_eq!(s.entry_data(ro).unwrap(), [0; 8]);

    // Writing to an invalid selector fails.
    let ctl =
        pc.dma(0x100, (0x3000 << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_WRITE, 4, 0x1000);
    assert_eq!(ctl, FW_CFG_DMA_CTL_ERROR);
    assert_eq!(seen.lock().unwrap().len(), 1);

    // Modifying a file makes it read only.
    s.modify_file("etc/rw", vec![0; 8]).unwrap();
    let ctl = pc.dma(
        0x100,
        (u32::from(rw) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_WRITE,
        4,
        0x1000,
    );
    assert_eq!(ctl, FW_CFG_DMA_CTL_ERROR);
}

#[test]
fn select_callback() {
    let pc = pc();
    let s = pc.state();
    let count = Arc::new(Mutex::new(0u8));
    let c2 = count.clone();
    let cb: FwCfgCallback = Box::new(move |data: &mut [u8]| {
        let mut c = c2.lock().unwrap();
        *c += 1;
        data[0] = *c;
    });
    s.add_file_callback("etc/lazy", Some(cb), None, vec![0; 2], true).unwrap();
    let (sel, _) = pc.find_file("etc/lazy").unwrap();
    assert_eq!(pc.get(sel, 2), [1, 0]);
    assert_eq!(pc.get(sel, 2), [2, 0]);
    // Moving the file to a new slot keeps its callback.
    s.add_file("etc/a", vec![]).unwrap();
    let (sel2, _) = pc.find_file("etc/lazy").unwrap();
    assert_eq!(sel2, sel + 1);
    assert_eq!(pc.get(sel2, 1), [3]);
}

struct Gen(Option<Vec<u8>>);

impl FwCfgDataGenerator for Gen {
    fn get_data(&self) -> ruvm_base::Result<Option<Vec<u8>>> {
        Ok(self.0.clone())
    }
}

#[test]
fn generator() {
    let pc = pc();
    let s = pc.state();
    assert!(s.add_file_from_generator(&Gen(Some(vec![9, 8])), "etc/gen").unwrap());
    assert!(!s.add_file_from_generator(&Gen(None), "etc/none").unwrap());
    assert_eq!(pc.get_file("etc/gen", 8), (2, vec![9, 8]));
    assert!(pc.find_file("etc/none").is_none());
}

#[test]
fn mmio_flavor() {
    let ram = Ram::new(0x10000);
    let m = fw_cfg_init_mem_dma(0x0902_0000, ram.clone(), &FwCfgMachineConfig::default()).unwrap();
    assert_eq!(m.addrs(), (0x0902_0008, 0x0902_0000, 0x0902_0010));
    assert_eq!(m.data_ops().region_size(), 8);
    let a = MemTxAttrs::UNSPECIFIED;
    let ctl: &dyn MmioOps = &**m.ctl_ops();
    let data: &dyn MmioOps = &**m.data_ops();

    // A big endian guest selects with a 2 byte write and reads 8 bytes at once.
    assert!(
        dispatch_write(ctl, "fwcfg.ctl", 0, 2, u64::from(FW_CFG_SIGNATURE), Endian::Big, a).is_ok()
    );
    let (v, r) = dispatch_read(data, "fwcfg.data", 0, 8, Endian::Big, a);
    assert!(r.is_ok());
    assert_eq!(v.to_be_bytes(), *b"QEMU\0\0\0\0");

    // A little endian guest sees the bytes in order too.
    assert!(
        dispatch_write(
            ctl,
            "fwcfg.ctl",
            0,
            2,
            u64::from(FW_CFG_ID).swap_bytes() >> 48,
            Endian::Little,
            a
        )
        .is_ok()
    );
    let (v, _) = dispatch_read(data, "fwcfg.data", 0, 4, Endian::Little, a);
    assert_eq!((v as u32).to_le_bytes(), (FW_CFG_VERSION | FW_CFG_VERSION_DMA).to_le_bytes());

    // The selector is write only and 2 bytes, the data register only at offset 0.
    assert!(!dispatch_read(ctl, "fwcfg.ctl", 0, 2, Endian::Big, a).1.is_ok());
    assert!(!dispatch_write(ctl, "fwcfg.ctl", 0, 1, 0, Endian::Big, a).is_ok());
    assert!(!dispatch_read(data, "fwcfg.data", 4, 4, Endian::Big, a).1.is_ok());

    // DMA through the MMIO register, one 8 byte big endian write.
    m.state().add_file("etc/x", vec![7; 4]).unwrap();
    let (_, sel, _) = m.state().files().into_iter().find(|f| f.0 == "etc/x").unwrap();
    let mut access = Vec::new();
    access.extend_from_slice(
        &((u32::from(sel) << 16) | FW_CFG_DMA_CTL_SELECT | FW_CFG_DMA_CTL_READ).to_be_bytes(),
    );
    access.extend_from_slice(&4u32.to_be_bytes());
    access.extend_from_slice(&0x2000u64.to_be_bytes());
    ram.bufwrite(0x100, &access);
    let dma: &dyn MmioOps = &**m.dma_ops().unwrap();
    assert!(dispatch_write(dma, "fwcfg.dma", 0, 8, 0x100, Endian::Big, a).is_ok());
    assert_eq!(ram.bufread(0x2000, 4), [7; 4]);
    assert_eq!(ram.bufread(0x100, 4), [0; 4]);

    // Without DMA there is no DMA region and FW_CFG_ID says so.
    let m = fw_cfg_init_mem_nodma(0x510, 0x511, 1, &FwCfgMachineConfig::default()).unwrap();
    assert!(m.dma_ops().is_none());
    assert_eq!(m.data_ops().region_size(), 1);
    assert_eq!(m.state().entry_data(FW_CFG_ID).unwrap(), FW_CFG_VERSION.to_le_bytes());
    assert!(!dispatch_read(&**m.data_ops(), "fwcfg.data", 0, 2, Endian::Big, a).1.is_ok());
    assert!(fw_cfg_init_mem_nodma(0, 8, 3, &FwCfgMachineConfig::default()).is_err());
}

#[test]
fn reset_selects_signature() {
    let pc = pc();
    pc.select(FW_CFG_ID);
    pc.read_data(2);
    pc.state().reset();
    assert_eq!(pc.state().cur_entry(), FW_CFG_SIGNATURE);
    assert_eq!(pc.state().cur_offset(), 0);
    assert_eq!(pc.read_data(4), b"QEMU");
}
