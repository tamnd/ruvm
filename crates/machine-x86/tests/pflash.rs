// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the CFI01 flash and the PC system flash mapping.
//!
//! The access traces were captured from QEMU 11.1.2 via qtest (`qemu-system-x86_64 -accel qtest
//! -qtest stdio -display none -nodefaults -M q35` with two `-drive if=pflash,format=raw`
//! images the size of the edk2 x86_64 CODE and VARS files, filled with the pattern in
//! [`pattern`]). The tests replay them against the model; QEMU is not needed to run them.

use std::path::PathBuf;
use std::sync::Arc;

use ruvm_machine_x86::pflash::{
    FLASH_SECTOR_SIZE, FlashDrive, FlashMapError, IsaBiosAlias, PC_FLASH_NAMES, Pflash,
    PflashBacking, PflashProps, SystemFlashMap, pc_system_flash_map, raw_block_length,
};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemTxResult, MemorySystem, RegionId};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
/// edk2-x86_64-code.fd from QEMU 11.1.2.
const CODE_SIZE: u64 = 3_653_632;
/// edk2-i386-vars.fd from QEMU 11.1.2.
const VARS_SIZE: u64 = 540_672;
const CODE_BASE: u64 = 0x1_0000_0000 - CODE_SIZE;
const VARS_BASE: u64 = CODE_BASE - VARS_SIZE;
const R: u8 = b'r';
const W: u8 = b'w';

/// The image contents used for the QEMU captures.
fn pattern(size: u64, mul: u64) -> Vec<u8> {
    (0..size).map(|i| ((i * mul + (i >> 8)) & 0xff) as u8).collect()
}

struct Machine {
    mem: Arc<MemorySystem>,
    space: Arc<AddressSpace>,
    code: Arc<Pflash>,
    vars: Arc<Pflash>,
}

/// Maps the two flashes and the isa-bios alias the way the q35 board does.
fn machine(code: PflashBacking, vars: PflashBacking) -> Machine {
    let mem = Arc::new(MemorySystem::new());
    let root = mem.new_container("system", 1 << 64).unwrap();
    let space = mem.address_space_init(root, "memory").unwrap();
    let drives = [
        Some(FlashDrive { name: "pflash0".into(), size: CODE_SIZE }),
        Some(FlashDrive { name: "pflash1".into(), size: VARS_SIZE }),
    ];
    let map = pc_system_flash_map(&drives, 8 << 20).unwrap();
    let mut devs = Vec::new();
    for (f, backing) in map.flashes.iter().zip([code, vars]) {
        let dev = Pflash::new(&mem, f.name, f.props, backing).unwrap();
        mem.add_subregion(root, f.base, dev.region()).unwrap();
        devs.push(dev);
    }
    let isa = map.isa_bios.unwrap();
    let alias =
        mem.new_alias(isa.name, devs[0].region(), isa.flash_offset, u128::from(isa.size)).unwrap();
    mem.set_readonly(alias, isa.readonly).unwrap();
    mem.add_subregion_overlap(root, isa.addr, alias, isa.priority).unwrap();
    let vars = devs.pop().unwrap();
    let code = devs.pop().unwrap();
    Machine { mem, space, code, vars }
}

fn pattern_machine() -> Machine {
    machine(
        PflashBacking::Bytes(pattern(CODE_SIZE, 7)),
        PflashBacking::Bytes(pattern(VARS_SIZE, 13)),
    )
}

/// Replays a qtest trace and checks every read.
fn replay(m: &Machine, trace: &[(u8, u32, u64, u64)]) {
    for (step, &(op, size, addr, value)) in trace.iter().enumerate() {
        if op == W {
            assert_eq!(m.space.store(addr, size, value, Endian::Little, U), MemTxResult::OK);
        } else {
            let (got, r) = m.space.load(addr, size, Endian::Little, U);
            assert_eq!(r, MemTxResult::OK);
            assert_eq!(got, value, "step {step}: read{size} {addr:#x}");
        }
    }
}

fn romd(mem: &MemorySystem, id: RegionId) -> bool {
    mem.region(id).unwrap().romd_mode
}

fn tmp_file(name: &str, contents: &[u8]) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pflash-{name}.fd"));
    std::fs::write(&path, contents).unwrap();
    path
}

/// The main trace: reads in read array mode, CFI query, device ID, status, program, erase,
/// (un)lock, unknown commands, write to buffer (good, crossing a buffer boundary, with 32 bit
/// data, with a zero count) and the isa-bios alias while pflash0 is in command mode.
#[rustfmt::skip]
const TRACE: &[(u8, u32, u64, u64)] = &[
    (R, 1, 0xffc00000, 0x0),
    (R, 4, 0xffc00000, 0x271a0d00),
    (R, 1, 0xffc84000, 0x0),
    (R, 1, 0xffff0, 0x4f),
    (R, 4, 0xe0000, 0xd5cec7c0),
    (W, 1, 0xffc00000, 0x98),
    (R, 1, 0xffc00000, 0x0),
    (R, 1, 0xffc00001, 0x0),
    (R, 1, 0xffc00002, 0x0),
    (R, 1, 0xffc00003, 0x0),
    (R, 1, 0xffc00004, 0x0),
    (R, 1, 0xffc00005, 0x0),
    (R, 1, 0xffc00006, 0x0),
    (R, 1, 0xffc00007, 0x0),
    (R, 1, 0xffc00008, 0x0),
    (R, 1, 0xffc00009, 0x0),
    (R, 1, 0xffc0000a, 0x0),
    (R, 1, 0xffc0000b, 0x0),
    (R, 1, 0xffc0000c, 0x0),
    (R, 1, 0xffc0000d, 0x0),
    (R, 1, 0xffc0000e, 0x0),
    (R, 1, 0xffc0000f, 0x0),
    (R, 1, 0xffc00010, 0x51),
    (R, 1, 0xffc00011, 0x52),
    (R, 1, 0xffc00012, 0x59),
    (R, 1, 0xffc00013, 0x1),
    (R, 1, 0xffc00014, 0x0),
    (R, 1, 0xffc00015, 0x31),
    (R, 1, 0xffc00016, 0x0),
    (R, 1, 0xffc00017, 0x0),
    (R, 1, 0xffc00018, 0x0),
    (R, 1, 0xffc00019, 0x0),
    (R, 1, 0xffc0001a, 0x0),
    (R, 1, 0xffc0001b, 0x45),
    (R, 1, 0xffc0001c, 0x55),
    (R, 1, 0xffc0001d, 0x0),
    (R, 1, 0xffc0001e, 0x0),
    (R, 1, 0xffc0001f, 0x7),
    (R, 1, 0xffc00020, 0x7),
    (R, 1, 0xffc00021, 0xa),
    (R, 1, 0xffc00022, 0x0),
    (R, 1, 0xffc00023, 0x4),
    (R, 1, 0xffc00024, 0x4),
    (R, 1, 0xffc00025, 0x4),
    (R, 1, 0xffc00026, 0x0),
    (R, 1, 0xffc00027, 0xe),
    (R, 1, 0xffc00028, 0x2),
    (R, 1, 0xffc00029, 0x0),
    (R, 1, 0xffc0002a, 0x8),
    (R, 1, 0xffc0002b, 0x0),
    (R, 1, 0xffc0002c, 0x1),
    (R, 1, 0xffc0002d, 0x83),
    (R, 1, 0xffc0002e, 0x0),
    (R, 1, 0xffc0002f, 0x10),
    (R, 1, 0xffc00030, 0x0),
    (R, 1, 0xffc00031, 0x50),
    (R, 1, 0xffc00032, 0x52),
    (R, 1, 0xffc00033, 0x49),
    (R, 1, 0xffc00034, 0x31),
    (R, 1, 0xffc00035, 0x30),
    (R, 1, 0xffc00036, 0x0),
    (R, 1, 0xffc00037, 0x0),
    (R, 1, 0xffc00038, 0x0),
    (R, 1, 0xffc00039, 0x0),
    (R, 1, 0xffc0003a, 0x0),
    (R, 1, 0xffc0003b, 0x0),
    (R, 1, 0xffc0003c, 0x0),
    (R, 1, 0xffc0003d, 0x0),
    (R, 1, 0xffc0003e, 0x0),
    (R, 1, 0xffc0003f, 0x1),
    (R, 1, 0xffc00040, 0x0),
    (R, 1, 0xffc00041, 0x0),
    (R, 1, 0xffc00042, 0x0),
    (R, 1, 0xffc00043, 0x0),
    (R, 1, 0xffc00044, 0x0),
    (R, 1, 0xffc00045, 0x0),
    (R, 1, 0xffc00046, 0x0),
    (R, 1, 0xffc00047, 0x0),
    (R, 1, 0xffc00048, 0x0),
    (R, 1, 0xffc00049, 0x0),
    (R, 1, 0xffc0004a, 0x0),
    (R, 1, 0xffc0004b, 0x0),
    (R, 1, 0xffc0004c, 0x0),
    (R, 1, 0xffc0004d, 0x0),
    (R, 1, 0xffc0004e, 0x0),
    (R, 1, 0xffc0004f, 0x0),
    (R, 1, 0xffc00050, 0x0),
    (R, 1, 0xffc00051, 0x0),
    (R, 1, 0xffc00052, 0x0),
    (R, 1, 0xffc00053, 0x0),
    (R, 2, 0xffc00010, 0x51),
    (R, 4, 0xffc00010, 0x51),
    (R, 1, 0xffc00110, 0x51),
    (W, 1, 0xffc00005, 0x55),
    (R, 1, 0xffc00010, 0x51),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc00000, 0x90),
    (R, 1, 0xffc00000, 0x0),
    (R, 1, 0xffc00001, 0x0),
    (R, 1, 0xffc00002, 0x0),
    (R, 1, 0xffc00003, 0x0),
    (R, 4, 0xffc00000, 0x0),
    (R, 1, 0xffc00100, 0x0),
    (W, 1, 0xffc00000, 0x70),
    (R, 1, 0xffc00000, 0x80),
    (R, 2, 0xffc00000, 0x80),
    (R, 4, 0xffc00000, 0x800080),
    (R, 1, 0xffc01234, 0x80),
    (W, 1, 0xffc00000, 0x50),
    (R, 1, 0xffc00000, 0x0),
    (R, 4, 0xffc00000, 0x271a0d00),
    (W, 1, 0xffc01234, 0x10),
    (R, 1, 0xffc01234, 0x0),
    (W, 1, 0xffc01234, 0x5a),
    (R, 1, 0xffc01234, 0x80),
    (R, 4, 0xffc01234, 0x800080),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc01234, 0x5a),
    (R, 4, 0xffc01230, 0xa99c8f82),
    (W, 4, 0xffc01238, 0x40),
    (R, 1, 0xffc00000, 0x80),
    (W, 4, 0xffc01238, 0x11223344),
    (R, 4, 0xffc00000, 0x800080),
    (W, 1, 0xffc00000, 0xff),
    (R, 4, 0xffc01238, 0x11223344),
    (W, 1, 0xffc02345, 0x20),
    (R, 1, 0xffc00000, 0x80),
    (R, 1, 0xffc02345, 0x80),
    (W, 1, 0xffc02345, 0xd0),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc01fff, 0x12),
    (R, 1, 0xffc02000, 0xff),
    (R, 1, 0xffc02fff, 0xff),
    (R, 1, 0xffc03000, 0x30),
    (W, 1, 0xffc05000, 0x20),
    (W, 1, 0xffc05000, 0xff),
    (R, 1, 0xffc05000, 0xff),
    (W, 1, 0xffc06000, 0x20),
    (W, 1, 0xffc06000, 0x33),
    (R, 1, 0xffc06000, 0xff),
    (R, 1, 0xffc06001, 0xff),
    (W, 1, 0xffc00000, 0x60),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc00000, 0x1),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc00000, 0xff),
    (W, 1, 0xffc00000, 0x60),
    (W, 1, 0xffc00000, 0xd0),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc00000, 0xff),
    (W, 1, 0xffc00000, 0x60),
    (W, 1, 0xffc00000, 0x42),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc00000, 0x33),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc00000, 0xf0),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc00000, 0x0),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc03010, 0xe8),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc03010, 0x3),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc03010, 0xa0),
    (W, 1, 0xffc03011, 0xa1),
    (W, 1, 0xffc03012, 0xa2),
    (W, 1, 0xffc03013, 0xa3),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc03010, 0xd0),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc00000, 0xff),
    (R, 4, 0xffc03010, 0xa3a2a1a0),
    (R, 1, 0xffc03000, 0x30),
    (W, 1, 0xffc030ff, 0xe8),
    (W, 1, 0xffc030ff, 0x1),
    (W, 1, 0xffc030ff, 0x77),
    (R, 1, 0xffc00000, 0x80),
    (W, 1, 0xffc03100, 0x66),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc00000, 0xd0),
    (R, 1, 0xffc00000, 0x0),
    (R, 1, 0xffc030ff, 0x23),
    (R, 1, 0xffc03100, 0x31),
    (W, 1, 0xffc00000, 0x70),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc00000, 0x50),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc04000, 0xe8),
    (W, 1, 0xffc04000, 0x1),
    (W, 4, 0xffc04000, 0xdeadbeef),
    (W, 4, 0xffc04004, 0xcafebabe),
    (R, 4, 0xffc00000, 0x800080),
    (W, 1, 0xffc00000, 0xd0),
    (R, 4, 0xffc00000, 0x800080),
    (W, 1, 0xffc00000, 0xff),
    (R, 8, 0xffc04000, 0xcafebabedeadbeef),
    (W, 1, 0xffc04100, 0xe8),
    (W, 1, 0xffc04100, 0x0),
    (W, 1, 0xffc04100, 0x12),
    (W, 1, 0xffc04100, 0x99),
    (R, 1, 0xffc04100, 0x41),
    (W, 1, 0xffc84000, 0x70),
    (R, 1, 0xe0000, 0x80),
    (R, 1, 0xffff0, 0x80),
    (R, 1, 0xfffffff0, 0x80),
    (W, 1, 0xffc84000, 0x98),
    (R, 1, 0xe0010, 0x51),
    (R, 1, 0xffc84010, 0x51),
    (W, 1, 0xffc84000, 0xff),
    (R, 1, 0xe0010, 0x30),
    (W, 1, 0xffc00000, 0x70),
    (R, 8, 0xffc00000, 0x90009000900090),
    (W, 1, 0xffc00000, 0xff),
];

/// The same kind of commands with both drives `readonly=on`.
#[rustfmt::skip]
const TRACE_READ_ONLY: &[(u8, u32, u64, u64)] = &[
    (W, 1, 0xffc01234, 0x10),
    (W, 1, 0xffc01234, 0x5a),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc01234, 0xb6),
    (W, 1, 0xffc00000, 0x50),
    (W, 1, 0xffc02000, 0x20),
    (R, 1, 0xffc00000, 0xa0),
    (W, 1, 0xffc00000, 0xd0),
    (R, 1, 0xffc00000, 0xa0),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc02000, 0x20),
    (W, 1, 0xffc00000, 0x50),
    (W, 1, 0xffc03000, 0xe8),
    (W, 1, 0xffc00000, 0x1),
    (W, 1, 0xffc03000, 0x11),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc03001, 0x22),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc00000, 0xd0),
    (R, 1, 0xffc00000, 0x0),
    (W, 1, 0xffc00000, 0x70),
    (R, 1, 0xffc00000, 0x90),
    (W, 1, 0xffc00000, 0xff),
    (R, 1, 0xffc03000, 0x30),
];

/// The vars image after [`TRACE`], worked out from the commands in it and checked against the
/// file QEMU left behind.
fn vars_after_trace() -> Vec<u8> {
    let mut v = pattern(VARS_SIZE, 13);
    v[0x1234] = 0x5a;
    v[0x1238..0x123c].copy_from_slice(&0x1122_3344u32.to_le_bytes());
    for sector in [0x2000, 0x5000, 0x6000] {
        v[sector..sector + 0x1000].fill(0xff);
    }
    v[0x3010..0x3014].copy_from_slice(&[0xa0, 0xa1, 0xa2, 0xa3]);
    v[0x4000..0x4008].copy_from_slice(&0xcafe_babe_dead_beefu64.to_le_bytes());
    v
}

#[test]
fn qemu_trace_matches() {
    let m = pattern_machine();
    replay(&m, TRACE);
    assert_eq!(m.vars.contents(), vars_after_trace());
    assert_eq!(m.code.contents(), pattern(CODE_SIZE, 7));
    assert!(romd(&m.mem, m.vars.region()));
    assert!(romd(&m.mem, m.code.region()));
}

#[test]
fn trace_is_written_back_to_the_file() {
    let code = tmp_file("trace-code", &pattern(CODE_SIZE, 7));
    let vars = tmp_file("trace-vars", &pattern(VARS_SIZE, 13));
    let m = machine(
        PflashBacking::File { path: code.clone(), read_only: false },
        PflashBacking::File { path: vars.clone(), read_only: false },
    );
    assert!(!m.vars.read_only());
    replay(&m, TRACE);
    assert_eq!(std::fs::read(&vars).unwrap(), vars_after_trace());
    assert_eq!(std::fs::read(&code).unwrap(), pattern(CODE_SIZE, 7));
    drop(m);
    let _ = std::fs::remove_file(code);
    let _ = std::fs::remove_file(vars);
}

#[test]
fn read_only_drive_ignores_programs_and_erases() {
    let code = tmp_file("ro-code", &pattern(CODE_SIZE, 7));
    let vars = tmp_file("ro-vars", &pattern(VARS_SIZE, 13));
    let m = machine(
        PflashBacking::File { path: code.clone(), read_only: true },
        PflashBacking::File { path: vars.clone(), read_only: true },
    );
    assert!(m.vars.read_only());
    replay(&m, TRACE_READ_ONLY);
    assert_eq!(m.vars.contents(), pattern(VARS_SIZE, 13));
    assert_eq!(std::fs::read(&vars).unwrap(), pattern(VARS_SIZE, 13));
    drop(m);
    let _ = std::fs::remove_file(code);
    let _ = std::fs::remove_file(vars);
}

#[test]
fn write_back_covers_whole_512_byte_sectors() {
    let image = pattern(VARS_SIZE, 13);
    let path = tmp_file("extent", &image);
    let mem = Arc::new(MemorySystem::new());
    let props = PflashProps::pc_system_flash(VARS_SIZE);
    let dev = Pflash::new(
        &mem,
        PC_FLASH_NAMES[1],
        props,
        PflashBacking::File { path: path.clone(), read_only: false },
    )
    .unwrap();
    let root = mem.new_container("system", 1 << 64).unwrap();
    let space = mem.address_space_init(root, "memory").unwrap();
    mem.add_subregion(root, 0, dev.region()).unwrap();

    // Scribble over the file behind the device's back, then see what a write puts back.
    std::fs::write(&path, vec![0x11u8; VARS_SIZE as usize]).unwrap();
    let file = || std::fs::read(&path).unwrap();

    // A single byte program rewrites the 512 byte sector around it.
    space.store(0x1234, 1, 0x10, Endian::Little, U);
    space.store(0x1234, 1, 0x5a, Endian::Little, U);
    let f = file();
    assert_eq!(f[0x1234], 0x5a);
    assert_eq!(&f[0x1200..0x1234], &image[0x1200..0x1234]);
    assert_eq!(&f[0x1235..0x1400], &image[0x1235..0x1400]);
    assert!(f[0x11ff] == 0x11 && f[0x1400] == 0x11);

    // A block erase rewrites the 4 KiB sector.
    space.store(0x2345, 1, 0x20, Endian::Little, U);
    space.store(0x2345, 1, 0xd0, Endian::Little, U);
    let f = file();
    assert!(f[0x2000..0x3000].iter().all(|&b| b == 0xff));
    assert!(f[0x1fff] == 0x11 && f[0x3000] == 0x11);

    // A buffered write flushes the 256 byte buffer, widened to 512 bytes.
    space.store(0x3100, 1, 0xe8, Endian::Little, U);
    space.store(0x3100, 1, 1, Endian::Little, U);
    space.store(0x3100, 1, 0x42, Endian::Little, U);
    space.store(0x3101, 1, 0x43, Endian::Little, U);
    // Not written back before the confirm.
    assert_eq!(file()[0x3100], 0x11);
    space.store(0x3100, 1, 0xd0, Endian::Little, U);
    let f = file();
    assert_eq!(&f[0x3100..0x3102], &[0x42, 0x43]);
    assert_eq!(&f[0x3000..0x3100], &image[0x3000..0x3100]);
    assert_eq!(&f[0x3102..0x3200], &image[0x3102..0x3200]);
    assert!(f[0x2fff] == 0xff && f[0x3200] == 0x11);
    assert_eq!(dev.status(), 0x80);
    drop(space);
    drop(dev);
    let _ = std::fs::remove_file(path);
}

#[test]
fn cfi_table_of_the_ovmf_flashes() {
    let m = pattern_machine();
    let mut expected = [0u8; 0x52];
    expected[0x10..0x31].copy_from_slice(&[
        b'Q', b'R', b'Y', 0x01, 0x00, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x45, 0x55, 0x00, 0x00,
        0x07, 0x07, 0x0a, 0x00, 0x04, 0x04, 0x04, 0x00, 0x0e, 0x02, 0x00, 0x08, 0x00, 0x01, 0x83,
        0x00, 0x10, 0x00,
    ]);
    expected[0x31..0x36].copy_from_slice(b"PRI10");
    expected[0x3f] = 0x01;
    assert_eq!(m.vars.cfi_table(), expected);
    // 892 blocks of 4 KiB for CODE.
    let code = m.code.cfi_table();
    assert_eq!((code[0x27], code[0x2d], code[0x2e], code[0x2f]), (0x0e, 0x7b, 0x03, 0x10));
    assert_eq!(m.vars.writeblock_size(), 256);
}

#[test]
fn romd_mode_follows_the_command_state() {
    let m = pattern_machine();
    let id = m.vars.region();
    let first = u64::from(pattern(VARS_SIZE, 13)[0x10]);
    assert!(romd(&m.mem, id));
    assert_eq!(m.space.load(VARS_BASE + 0x10, 1, Endian::Little, U).0, first);

    // Any command write leaves romd mode; status and query commands read through the device.
    m.space.store(VARS_BASE, 1, 0x70, Endian::Little, U);
    assert!(!romd(&m.mem, id));
    assert_eq!(m.space.load(VARS_BASE + 0x10, 1, Endian::Little, U).0, 0x80);
    m.space.store(VARS_BASE, 1, 0x98, Endian::Little, U);
    assert!(!romd(&m.mem, id));
    assert_eq!(m.space.load(VARS_BASE + 0x10, 1, Endian::Little, U).0, u64::from(b'Q'));
    m.space.store(VARS_BASE, 1, 0xff, Endian::Little, U);
    assert!(romd(&m.mem, id));
    assert_eq!(m.space.load(VARS_BASE + 0x10, 1, Endian::Little, U).0, first);

    // Reset goes back to read array mode with romd on and the status ready.
    m.space.store(VARS_BASE, 1, 0xe8, Endian::Little, U);
    m.space.store(VARS_BASE, 1, 0, Endian::Little, U);
    m.space.store(VARS_BASE, 1, 0x12, Endian::Little, U);
    assert_eq!(m.vars.status(), 0x90);
    assert!(!romd(&m.mem, id));
    m.vars.reset();
    assert!(romd(&m.mem, id));
    assert_eq!(m.vars.status(), 0x80);
    assert_eq!(m.space.load(VARS_BASE + 0x10, 1, Endian::Little, U).0, first);
}

#[test]
fn isa_bios_alias_shows_the_top_of_pflash0() {
    let m = pattern_machine();
    let code = pattern(CODE_SIZE, 7);
    let top = (CODE_SIZE - 0x2_0000) as usize;
    let mut b = [0u8; 16];
    m.space.read(0xe0000, U, &mut b);
    assert_eq!(&b, &code[top..top + 16]);
    m.space.read(0xffff0, U, &mut b);
    assert_eq!(&b, &code[code.len() - 16..]);
    // Read only only keeps direct RAM writes out. As in QEMU 11.1.2 (checked via qtest), a
    // write through the alias still reaches the flash's command interface.
    m.space.store(0xe0000, 1, 0x70, Endian::Little, U);
    assert!(!romd(&m.mem, m.code.region()));
    assert_eq!(m.space.load(0xe0000, 1, Endian::Little, U).0, 0x80);
    assert_eq!(m.space.load(CODE_BASE, 1, Endian::Little, U).0, 0x80);
    m.space.store(0xe0000, 1, 0xff, Endian::Little, U);
    assert_eq!(m.space.load(0xe0000, 1, Endian::Little, U).0, u64::from(code[top]));
}

#[test]
fn flash_map_of_one_and_two_images() {
    let one = [Some(FlashDrive { name: "pflash0".into(), size: CODE_SIZE }), None];
    let map = pc_system_flash_map(&one, 8 << 20).unwrap();
    assert_eq!(map.flashes.len(), 1);
    let f = map.flashes[0];
    assert_eq!((f.index, f.name, f.base, f.size), (0, "system.flash0", CODE_BASE, CODE_SIZE));
    assert_eq!(f.base, 0xffc8_4000);
    let p = f.props;
    assert_eq!((p.num_blocks, p.sector_len, p.width, p.device_width), (892, 4096, 1, 0));
    assert_eq!(
        (p.max_device_width, p.big_endian, p.id0, p.id1, p.id2, p.id3),
        (0, false, 0, 0, 0, 0)
    );
    assert_eq!(
        map.isa_bios,
        Some(IsaBiosAlias {
            name: "isa-bios",
            flash_offset: CODE_SIZE - 0x2_0000,
            size: 0x2_0000,
            addr: 0xe0000,
            priority: 1,
            readonly: true,
        })
    );

    let two = [
        Some(FlashDrive { name: "pflash0".into(), size: CODE_SIZE }),
        Some(FlashDrive { name: "pflash1".into(), size: VARS_SIZE }),
    ];
    let map = pc_system_flash_map(&two, 8 << 20).unwrap();
    let placed: Vec<_> = map.flashes.iter().map(|f| (f.index, f.name, f.base, f.size)).collect();
    assert_eq!(
        placed,
        vec![
            (0, "system.flash0", CODE_BASE, CODE_SIZE),
            (1, "system.flash1", VARS_BASE, VARS_SIZE)
        ]
    );
    assert_eq!(VARS_BASE, 0xffc0_0000);
    assert_eq!(map.flashes[1].props.num_blocks, 132);

    // A flash smaller than 128 KiB is aliased whole.
    let small = [Some(FlashDrive { name: "pflash0".into(), size: 0x1_0000 }), None];
    let isa = pc_system_flash_map(&small, 8 << 20).unwrap().isa_bios.unwrap();
    assert_eq!((isa.flash_offset, isa.size, isa.addr), (0, 0x1_0000, 0xf_0000));

    assert_eq!(pc_system_flash_map(&[None, None], 8 << 20).unwrap(), SystemFlashMap::default());
    assert_eq!(FLASH_SECTOR_SIZE, 0x1000);
}

#[test]
fn flash_map_errors_are_qemus() {
    // The texts QEMU 11.1.2 prints for the same configurations.
    let gap = [None, Some(FlashDrive { name: "pflash1".into(), size: VARS_SIZE })];
    let e = pc_system_flash_map(&gap, 8 << 20).unwrap_err();
    assert_eq!(e, FlashMapError::Gap { index: 1 });
    assert_eq!(e.to_string(), "pflash1 requires pflash0");
    assert_eq!(e.info(), None);

    // A 5000 byte file is 5120 bytes to the raw driver.
    assert_eq!(raw_block_length(5000), 5120);
    let bad = [Some(FlashDrive { name: "pflash0".into(), size: raw_block_length(5000) }), None];
    let e = pc_system_flash_map(&bad, 8 << 20).unwrap_err();
    assert_eq!(e.to_string(), "system firmware block device pflash0 has invalid size 5120");
    assert_eq!(e.info().unwrap(), "its size must be a non-zero multiple of 0x1000");
    let empty = [Some(FlashDrive { name: "pflash0".into(), size: 0 }), None];
    assert!(matches!(
        pc_system_flash_map(&empty, 8 << 20),
        Err(FlashMapError::InvalidSize { size: 0, .. })
    ));

    let big = [
        Some(FlashDrive { name: "pflash0".into(), size: CODE_SIZE }),
        Some(FlashDrive { name: "pflash1".into(), size: 8_392_704 }),
    ];
    let e = pc_system_flash_map(&big, 8 << 20).unwrap_err();
    assert_eq!(e.to_string(), "combined size of system firmware exceeds 8388608 bytes");
    // Exactly max-fw-size is fine.
    let exact = [Some(FlashDrive { name: "pflash0".into(), size: 8 << 20 }), None];
    assert_eq!(pc_system_flash_map(&exact, 8 << 20).unwrap().flashes[0].base, 0xff80_0000);
}

#[test]
fn creation_errors() {
    let mem = Arc::new(MemorySystem::new());
    let props = PflashProps::pc_system_flash(0x2000);
    let e = Pflash::new(&mem, "f", props, PflashBacking::Bytes(vec![0; 0x1000])).unwrap_err();
    assert_eq!(e, "cfi.pflash01 device 'f' requires 8192 bytes, block backend provides 4096 bytes");
    let e = Pflash::new(&mem, "f", PflashProps { num_blocks: 0, ..props }, PflashBacking::None)
        .unwrap_err();
    assert_eq!(e, "attribute \"num-blocks\" not specified or zero.");
    let e = Pflash::new(&mem, "f", PflashProps { sector_len: 0, ..props }, PflashBacking::None)
        .unwrap_err();
    assert_eq!(e, "attribute \"sector-length\" not specified or zero.");
    let missing = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("pflash-missing.fd");
    let e = Pflash::new(&mem, "f", props, PflashBacking::File { path: missing, read_only: true })
        .unwrap_err();
    assert!(
        e.starts_with("Could not open '") && e.ends_with("': No such file or directory"),
        "{e}"
    );

    // Without a drive the flash is zeroed and writable.
    let dev = Pflash::new(&mem, "f", props, PflashBacking::None).unwrap();
    assert_eq!(dev.contents(), vec![0; 0x2000]);
    assert!(!dev.read_only());
    assert_eq!(dev.size(), 0x2000);
}

/// A 32 bit bank of two x16 chips, to cover the device-width paths the PC flashes do not use.
#[test]
fn device_width_queries() {
    let mem = Arc::new(MemorySystem::new());
    let props = PflashProps {
        num_blocks: 4,
        sector_len: 0x2_0000,
        width: 4,
        device_width: 2,
        id0: 0x89,
        id1: 0x18,
        ..PflashProps::default()
    };
    let dev = Pflash::new(&mem, "f", props, PflashBacking::None).unwrap();
    let root = mem.new_container("system", 1 << 64).unwrap();
    let space = mem.address_space_init(root, "memory").unwrap();
    mem.add_subregion(root, 0, dev.region()).unwrap();
    let t = dev.cfi_table();
    // Two devices: each has half the sector and the write buffer doubles.
    assert_eq!((t[0x2a], t[0x2f], t[0x30], dev.writeblock_size()), (0x0b, 0x00, 0x01, 4096));
    assert_eq!(t[0x27], 18);

    space.store(0, 4, 0x0098_0098, Endian::Little, U);
    assert_eq!(space.load(0x10 * 4, 4, Endian::Little, U).0, 0x0051_0051);
    // Query mode is only left with read array.
    space.store(0, 4, 0x0090_0090, Endian::Little, U);
    assert_eq!(space.load(0x11 * 4, 4, Endian::Little, U).0, 0x0052_0052);
    space.store(0, 4, 0x00ff_00ff, Endian::Little, U);
    space.store(0, 4, 0x0090_0090, Endian::Little, U);
    assert_eq!(space.load(0, 4, Endian::Little, U).0, 0x0089_0089);
    assert_eq!(space.load(4, 4, Endian::Little, U).0, 0x0018_0018);
    space.store(0, 4, 0x0070_0070, Endian::Little, U);
    assert_eq!(space.load(0, 4, Endian::Little, U).0, 0x0080_0080);
}
