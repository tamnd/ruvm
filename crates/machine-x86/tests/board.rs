// SPDX-License-Identifier: GPL-2.0-or-later

//! The pieces the command line uses to build an x86 board: the firmware search order, the
//! raw file disk backend and the board builder. None of this needs KVM.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use ruvm_base::ClockType;
use ruvm_hw_core::Clock;
use ruvm_machine_x86::board::{
    BoardKind, BoardSpec, KernelFiles, build_board, canonical_machine_name, load_kernel,
};
use ruvm_machine_x86::firmware::{DEFAULT_FIRMWARE_DIRS, QEMU_BINARY, qemu_data_dir};
use ruvm_machine_x86::{FileBackend, FirmwareSearch};

/// A fresh directory under the system temp directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> TempDir {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p =
            std::env::temp_dir().join(format!("ruvm-machine-x86-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn file(&self, name: &str, data: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d).unwrap();
        }
        std::fs::write(&p, data).unwrap();
        p
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn search_order_is_l_dirs_then_distro_dirs_then_qemu() {
    let l = vec![PathBuf::from("/a"), PathBuf::from("/b")];
    let s = FirmwareSearch::with_fallback(&l, Some(PathBuf::from("/opt/qemu/share/qemu")));
    let want: Vec<PathBuf> = ["/a", "/b"]
        .iter()
        .chain(DEFAULT_FIRMWARE_DIRS)
        .chain(&["/opt/qemu/share/qemu"])
        .map(PathBuf::from)
        .collect();
    assert_eq!(s.dirs(), want.as_slice());
    assert_eq!(
        DEFAULT_FIRMWARE_DIRS,
        &["/usr/share/qemu", "/usr/share/seabios", "/usr/local/share/qemu"]
    );
}

#[test]
fn first_directory_with_the_file_wins() {
    let a = TempDir::new("fw-a");
    let b = TempDir::new("fw-b");
    b.file("bios-256k.bin", b"from b");
    b.file("linuxboot_dma.bin", b"rom b");
    a.file("linuxboot_dma.bin", b"rom a");
    let s = FirmwareSearch::from_dirs(vec![a.path().to_path_buf(), b.path().to_path_buf()]);
    assert_eq!(s.load("bios-256k.bin").unwrap(), b"from b");
    assert_eq!(s.load("linuxboot_dma.bin").unwrap(), b"rom a");
    assert!(s.find("missing.bin").is_none());
    // A path that exists is taken as it is, like qemu_find_file() does.
    let direct = a.file("my.bin", b"direct");
    assert_eq!(s.find(direct.to_str().unwrap()).unwrap(), direct);
}

#[test]
fn qemu_on_path_gives_its_share_dir() {
    let t = TempDir::new("qemu-prefix");
    t.file(&format!("bin/{QEMU_BINARY}"), b"#!/bin/sh\n");
    std::fs::create_dir_all(t.path().join("share/qemu")).unwrap();
    let path = std::env::join_paths([t.path().join("nothing"), t.path().join("bin")]).unwrap();
    let got = qemu_data_dir(Some(&path)).unwrap();
    assert_eq!(got.canonicalize().unwrap(), t.path().join("share/qemu").canonicalize().unwrap());
    assert!(qemu_data_dir(None).is_none());
    let empty = std::env::join_paths([t.path().join("nothing")]).unwrap();
    assert!(qemu_data_dir(Some(&empty)).is_none());
}

#[test]
fn file_backend_reads_and_writes() {
    let t = TempDir::new("disk");
    let p = t.file("disk.img", &[0u8; 4096]);
    let f = FileBackend::open(&p, false).unwrap();
    assert_eq!(f.size(), 4096);
    assert!(f.is_writable());
    f.write_at(512, b"hello").unwrap();
    let mut buf = [0u8; 5];
    f.read_at(512, &mut buf).unwrap();
    assert_eq!(&buf, b"hello");
    f.flush().unwrap();
    assert!(f.read_at(4094, &mut buf).is_err());
    drop(f);
    assert_eq!(&std::fs::read(&p).unwrap()[512..517], b"hello");

    // Through both device traits.
    let mut f = FileBackend::open(&p, true).unwrap();
    assert!(!ruvm_hw_virtio::BlockBackend::is_writable(&f));
    assert!(ruvm_hw_virtio::BlockBackend::write_at(&mut f, 0, b"x").is_err());
    assert_eq!(ruvm_hw_storage::BlockBackend::len(&f), 4096);
    ruvm_hw_storage::BlockBackend::read_at(&f, 512, &mut buf).unwrap();
    assert_eq!(&buf, b"hello");
}

#[test]
fn file_backend_open_error_is_qemus() {
    let e = FileBackend::open("/nonexistent/disk.img", false).unwrap_err();
    assert_eq!(e, "Could not open '/nonexistent/disk.img': No such file or directory");
}

#[test]
fn board_names() {
    assert_eq!(BoardKind::from_name("microvm"), Some(BoardKind::Microvm));
    assert_eq!(BoardKind::from_name("q35"), Some(BoardKind::Q35));
    assert_eq!(BoardKind::from_name("pc-q35-11.1"), Some(BoardKind::Q35));
    assert_eq!(BoardKind::from_name("pc-q35-11.0"), Some(BoardKind::Q35));
    assert_eq!(BoardKind::from_name("pc-q35-10.2"), Some(BoardKind::Q35));
    assert_eq!(BoardKind::from_name("pc-q35-10.1"), None);
    assert_eq!(BoardKind::from_name("pc"), None);
    assert_eq!(canonical_machine_name("q35"), Some("pc-q35-11.1"));
    assert_eq!(canonical_machine_name("pc-q35-10.2"), Some("pc-q35-10.2"));
    assert_eq!(canonical_machine_name("microvm"), Some("microvm"));
    assert_eq!(canonical_machine_name("pc"), None);
    assert!(BoardKind::Microvm.default_kernel_irqchip_split());
    assert!(!BoardKind::Q35.default_kernel_irqchip_split());
}

#[test]
fn kernel_errors_are_qemus() {
    let files = KernelFiles { kernel: "/nonexistent/bzImage".into(), ..KernelFiles::default() };
    assert_eq!(
        load_kernel(&files).unwrap_err(),
        "qemu: could not open kernel file '/nonexistent/bzImage': No such file or directory"
    );
    let dir = TempDir::new("kernel-errors");
    let kernel = dir.file("bzImage", b"kernel").to_str().unwrap().to_string();
    let files = KernelFiles {
        kernel,
        initrd: Some("/nonexistent/initrd".into()),
        ..KernelFiles::default()
    };
    assert_eq!(
        load_kernel(&files).unwrap_err(),
        "qemu: error reading initrd /nonexistent/initrd: Failed to open file \u{201c}/nonexistent/initrd\u{201d}: open() failed: No such file or directory"
    );
    #[cfg(target_os = "linux")]
    {
        let d = dir.path().to_str().unwrap().to_string();
        let files = KernelFiles { kernel: d.clone(), ..KernelFiles::default() };
        assert_eq!(
            load_kernel(&files).unwrap_err(),
            format!("qemu: could not load kernel '{d}': Is a directory")
        );
        let files =
            KernelFiles { kernel: format!("{d}/bzImage"), initrd: Some(d.clone()), ..files };
        assert_eq!(
            load_kernel(&files).unwrap_err(),
            format!(
                "qemu: error reading initrd {d}: Failed to map {d}' {d}': mmap() failed: No such device"
            )
        );
    }
}

fn spec(kind: BoardKind, firmware: FirmwareSearch) -> BoardSpec {
    BoardSpec {
        kind,
        machine_type: if kind == BoardKind::Q35 { "pc-q35-11.1" } else { "microvm" },
        props: Vec::new(),
        ram_size: Some(256 << 20),
        memdev: None,
        aux_ram_share: false,
        cpus: 1,
        max_cpus: 0,
        kvm: true,
        pit_in_kernel: true,
        smm_available: false,
        phys_bits: 40,
        cpu: Default::default(),
        bios: None,
        pflash: [None, None],
        uuid: None,
        smbios: Default::default(),
        topology: None,
        kernel: None,
        firmware,
        serial_hds: vec![true],
        clock: Clock::manual(ClockType::Virtual),
        rtc_clock: Clock::manual(ClockType::Host),
    }
}

#[test]
fn builds_boards_with_firmware_from_the_search_path() {
    let t = TempDir::new("boards");
    t.file("bios-256k.bin", &[0xf4; 256 * 1024]);
    t.file("bios-microvm.bin", &[0xf4; 128 * 1024]);
    let fw = FirmwareSearch::from_dirs(vec![t.path().to_path_buf()]);

    let (mut m, _) = build_board(spec(BoardKind::Microvm, fw.clone())).unwrap();
    assert_eq!(m.name(), "microvm");
    assert_eq!(m.apic_ids(), vec![0]);
    assert!(m.serial(0).is_some() && m.serial(1).is_none());
    m.machine_done().unwrap();

    let (mut q, _) = build_board(spec(BoardKind::Q35, fw.clone())).unwrap();
    assert_eq!(q.name(), "pc-q35-11.1");
    q.machine_done().unwrap();
    let mut s = spec(BoardKind::Q35, fw.clone());
    s.machine_type = "pc-q35-11.0";
    assert_eq!(build_board(s).unwrap().0.name(), "pc-q35-11.0");

    let mut s = spec(BoardKind::Q35, FirmwareSearch::from_dirs(Vec::new()));
    s.bios = Some("nope.bin".into());
    assert_eq!(build_board(s).unwrap_err(), "qemu: could not load PC BIOS 'nope.bin'");

    let mut s = spec(BoardKind::Microvm, fw);
    s.props = vec![("bogus".into(), "on".into())];
    assert_eq!(build_board(s).unwrap_err(), "Property 'microvm-machine.bogus' not found");
}

#[test]
fn boards_take_the_memory_backend_as_their_ram() {
    let t = TempDir::new("memdev");
    t.file("bios-256k.bin", &[0xf4; 256 * 1024]);
    t.file("bios-microvm.bin", &[0xf4; 128 * 1024]);
    let fw = FirmwareSearch::from_dirs(vec![t.path().to_path_buf()]);
    for kind in [BoardKind::Q35, BoardKind::Microvm] {
        let block = std::sync::Arc::new(ruvm_mem::RamBlock::new("mem0", 64 << 20, 12).unwrap());
        let mut s = spec(kind, fw.clone());
        s.ram_size = None;
        s.memdev = Some(block.clone());
        let (m, _) = build_board(s).unwrap();
        // Without -m the RAM is the backend's size, and the board's RAM is the backend's.
        assert_eq!(m.ram_size(), 64 << 20);
        let mut b = [0];
        block.write(0x1000, &[0x5a]).unwrap();
        let r = m.memory_as().read(0x1000, ruvm_mem::MemTxAttrs::default(), &mut b);
        assert_eq!((r, b), (ruvm_mem::MemTxResult::OK, [0x5a]));

        let mut s = spec(kind, fw.clone());
        s.memdev = Some(block);
        assert_eq!(
            build_board(s).unwrap_err(),
            "Machine memory size does not match the size of the memory backend"
        );
    }
}
