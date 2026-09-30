// SPDX-License-Identifier: MIT OR Apache-2.0

//! End to end tests of the memory API: rendering rules, dispatch through address spaces,
//! listener ordering and dirty tracking.

use std::sync::{Arc, Mutex};

use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, DirtyClient, DirtyMask, Endian,
    FlatRange, GLOBAL_DIRTY_MIGRATION, IommuAccessFlags, IommuOps, IommuTlbEntry, MemError,
    MemResult, MemTxAttrs, MemTxResult, MemoryConfig, MemoryListener, MemorySystem, MmioOps,
    RegionId,
};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const SPACE: u128 = 1 << 64;

/// A device that logs its calls and returns `0x40 + offset` from every read.
#[derive(Default)]
struct Dev {
    valid: AccessConstraints,
    imp: AccessConstraints,
    log: Mutex<Vec<(char, u64, u32, u64)>>,
}

impl Dev {
    fn take(&self) -> Vec<(char, u64, u32, u64)> {
        std::mem::take(&mut self.log.lock().unwrap())
    }
}

impl MmioOps for Dev {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        self.log.lock().unwrap().push(('r', offset, size.bytes(), 0));
        Ok(0x40 + offset)
    }
    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        self.log.lock().unwrap().push(('w', offset, size.bytes(), value));
        Ok(())
    }
    fn valid(&self) -> AccessConstraints {
        self.valid
    }
    fn impl_constraints(&self) -> AccessConstraints {
        self.imp
    }
}

fn names(space: &AddressSpace) -> Vec<(u64, u128, String, u64)> {
    space
        .flatview()
        .ranges()
        .iter()
        .map(|r| (r.addr(), r.size(), r.name().to_string(), r.offset_in_region()))
        .collect()
}

fn setup() -> (MemorySystem, RegionId, Arc<AddressSpace>) {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", SPACE).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    (sys, root, space)
}

#[test]
fn higher_priority_punches_through() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x10000).unwrap();
    let dev = sys.new_io("dev", 0x1000, Arc::new(Dev::default())).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.add_subregion_overlap(root, 0x8000, dev, 1).unwrap();
    assert_eq!(
        names(&space),
        vec![
            (0, 0x8000, "ram".into(), 0),
            (0x8000, 0x1000, "dev".into(), 0),
            (0x9000, 0x7000, "ram".into(), 0x9000),
        ]
    );
    sys.set_enabled(dev, false).unwrap();
    assert_eq!(names(&space), vec![(0, 0x10000, "ram".into(), 0)]);
}

#[test]
fn equal_priority_newest_wins_and_moving_counts_as_new() {
    let (sys, root, space) = setup();
    let a = sys.new_reservation("a", 0x1000).unwrap();
    let b = sys.new_reservation("b", 0x1000).unwrap();
    sys.add_subregion(root, 0, a).unwrap();
    sys.add_subregion(root, 0x800, b).unwrap();
    assert_eq!(space.flatview().lookup(0x900).unwrap().name(), "b");
    // Moving `a` re-adds it, so it now sits above `b`.
    sys.set_address(a, 0x100).unwrap();
    assert_eq!(space.flatview().lookup(0x900).unwrap().name(), "a");
    assert_eq!(sys.region(root).unwrap().children, vec![a, b]);
}

#[test]
fn alias_windows_and_containers_offset_correctly() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x4000).unwrap();
    let low = sys.new_alias("ram-low", ram, 0x1000, 0x1000).unwrap();
    let sub = sys.new_container("bus", 0x10000).unwrap();
    sys.add_subregion(sub, 0x2000, low).unwrap();
    sys.add_subregion(root, 0x10_0000, sub).unwrap();
    assert_eq!(names(&space), vec![(0x10_2000, 0x1000, "ram".into(), 0x1000)]);
    sys.ram_block(ram).unwrap().write(0x1004, &[1, 2, 3, 4]).unwrap();
    assert_eq!(space.read_u32(0x10_2004, U), (0x0403_0201, MemTxResult::OK));
    sys.set_alias_offset(low, 0).unwrap();
    assert_eq!(space.read_u32(0x10_2004, U).0, 0);
    assert!(matches!(sys.set_alias_offset(ram, 0), Err(MemError::WrongKind(_))));
}

#[test]
fn transactions_publish_once() {
    let (sys, root, space) = setup();
    let a = sys.new_reservation("a", 0x1000).unwrap();
    let before = space.flatview();
    {
        let _t = sys.transaction();
        sys.add_subregion(root, 0, a).unwrap();
        sys.set_address(a, 0x5000).unwrap();
        assert!(space.flatview().is_empty());
    }
    assert_eq!(names(&space), vec![(0x5000, 0x1000, "a".into(), 0)]);
    assert!(before.is_empty(), "old views never change");
}

#[test]
fn tree_errors_are_reported() {
    let (sys, root, _space) = setup();
    let c = sys.new_container("c", 0x1000).unwrap();
    let d = sys.new_container("d", 0x1000).unwrap();
    sys.add_subregion(root, 0, c).unwrap();
    assert!(matches!(sys.add_subregion(root, 0, c), Err(MemError::AlreadyMapped(_))));
    sys.add_subregion(c, 0, d).unwrap();
    let e = sys.new_container("e", 0x1000).unwrap();
    assert!(matches!(sys.add_subregion(e, 0, e), Err(MemError::Cycle(_))));
    let alias = sys.new_alias("loop", c, 0, 0x100).unwrap();
    assert!(matches!(sys.add_subregion(d, 0, alias), Err(MemError::Cycle(_))));
    assert!(matches!(sys.del_subregion(root, d), Err(MemError::NotASubregion(_))));
    assert!(matches!(sys.destroy_region(c), Err(MemError::InUse(_))));
    assert!(matches!(sys.destroy_region(root), Err(MemError::InUse(_))));
    sys.destroy_region(alias).unwrap();
    sys.del_subregion(root, c).unwrap();
    sys.destroy_region(c).unwrap();
    assert_eq!(sys.region(d).unwrap().parent, None);
    assert_eq!(sys.set_enabled(c, true), Err(MemError::NoSuchRegion));
    assert!(matches!(sys.new_container("big", SPACE + 1), Err(MemError::TooLarge(_))));
}

#[test]
fn ram_and_device_accesses_split_at_the_boundary() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x1000).unwrap();
    let dev = Arc::new(Dev { valid: AccessConstraints::any_size(1, 4), ..Dev::default() });
    let io = sys.new_io("dev", 0x1000, dev.clone()).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.add_subregion(root, 0x1000, io).unwrap();
    let buf = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa];
    assert_eq!(space.write(0xffc, U, &buf), MemTxResult::OK);
    let mut back = [0; 4];
    sys.ram_block(ram).unwrap().read(0xffc, &mut back).unwrap();
    assert_eq!(back, [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(dev.take(), vec![('w', 0, 4, 0x8877_6655), ('w', 4, 2, 0xaa99)]);

    let mut out = [0u8; 7];
    assert_eq!(space.read(0x1001, U, &mut out), MemTxResult::OK);
    assert_eq!(dev.take(), vec![('r', 1, 1, 0), ('r', 2, 2, 0), ('r', 4, 4, 0)]);
    assert_eq!(out, [0x41, 0x42, 0, 0x44, 0, 0, 0]);
}

#[test]
fn holes_rom_and_memory_attribute() {
    let (sys, root, space) = setup();
    let rom = sys.new_rom("bios", 0x1000).unwrap();
    let io = sys.new_io("dev", 0x100, Arc::new(Dev::default())).unwrap();
    sys.add_subregion(root, 0x1000, rom).unwrap();
    sys.add_subregion(root, 0x3000, io).unwrap();

    let mut b = [0xffu8; 4];
    assert_eq!(space.read(0x8000, U, &mut b), MemTxResult::DECODE_ERROR);
    assert_eq!(b, [0; 4]);

    assert_eq!(space.write_u32(0x1000, U, 0xdead_beef), MemTxResult::DECODE_ERROR);
    assert_eq!(space.read_u32(0x1000, U), (0, MemTxResult::OK));
    let debug = MemTxAttrs::new().with_debug(true);
    assert_eq!(space.write_u32(0x1000, debug, 0xdead_beef), MemTxResult::OK);
    assert_eq!(space.read_u32(0x1000, U), (0xdead_beef, MemTxResult::OK));

    let mem = MemTxAttrs::new().with_memory(true);
    assert_eq!(space.read_u32(0x1000, mem).1, MemTxResult::OK);
    assert_eq!(space.read_u32(0x3000, mem).1, MemTxResult::ACCESS_ERROR);
}

#[test]
fn load_and_store_do_not_split() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x1000).unwrap();
    let dev = Arc::new(Dev { imp: AccessConstraints::any_size(1, 8), ..Dev::default() });
    let io = sys.new_io("dev", 0x1000, dev.clone()).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.add_subregion(root, 0x2000, io).unwrap();
    assert_eq!(space.store(0x10, 4, 0x1122_3344, Endian::Big, U), MemTxResult::OK);
    let mut b = [0; 4];
    sys.ram_block(ram).unwrap().read(0x10, &mut b).unwrap();
    assert_eq!(b, [0x11, 0x22, 0x33, 0x44]);
    assert_eq!(space.load(0x10, 2, Endian::Little, U), (0x2211, MemTxResult::OK));
    assert_eq!(space.load(0x2008, 8, Endian::Little, U), (0x48, MemTxResult::OK));
    assert_eq!(dev.take(), vec![('r', 8, 8, 0)]);
    // Past the end of RAM the whole access goes to the unassigned callbacks.
    assert_eq!(space.load(0xffe, 4, Endian::Little, U), (0, MemTxResult::DECODE_ERROR));
}

#[test]
fn rom_device_switches_between_ram_and_callbacks() {
    let (sys, root, space) = setup();
    let dev = Arc::new(Dev::default());
    let flash = sys.new_rom_device("flash", 0x1000, dev.clone()).unwrap();
    sys.add_subregion(root, 0, flash).unwrap();
    sys.ram_block(flash).unwrap().write(0, &[9, 0, 0, 0]).unwrap();
    assert_eq!(space.read_u32(0, U).0, 9);
    space.write_u32(0, U, 5);
    assert_eq!(dev.take(), vec![('w', 0, 4, 5)]);
    sys.set_romd(flash, false).unwrap();
    assert_eq!(space.read_u32(4, U).0, 0x44);
    assert!(matches!(sys.set_romd(root, false), Err(MemError::WrongKind(_))));
}

struct Iommu {
    target: Arc<AddressSpace>,
}

impl IommuOps for Iommu {
    fn translate(&self, addr: u64, _flag: IommuAccessFlags, _idx: u32) -> IommuTlbEntry {
        // Page 0 maps to 0x5000, read only. Everything else faults.
        let ok = addr < 0x1000;
        IommuTlbEntry {
            target_as: ok.then(|| self.target.clone()),
            iova: addr & !0xfff,
            translated_addr: 0x5000,
            addr_mask: 0xfff,
            perm: if ok { IommuAccessFlags::RO } else { IommuAccessFlags::NONE },
        }
    }
}

#[test]
fn iommu_translates_into_the_target_space() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x10000).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.ram_block(ram).unwrap().write(0x5ffe, &[1, 2, 3, 4]).unwrap();
    let dma_root = sys.new_container("dma", SPACE).unwrap();
    let dma = sys.address_space_init(dma_root, "dma").unwrap();
    let iommu = sys.new_iommu("iommu", 0x10000, Arc::new(Iommu { target: space.clone() })).unwrap();
    sys.add_subregion(dma_root, 0, iommu).unwrap();
    let mut b = [0; 2];
    assert_eq!(dma.read(0xffe, U, &mut b), MemTxResult::OK);
    assert_eq!(b, [1, 2]);
    // The second half of this read crosses into an unmapped page.
    let mut b = [0; 4];
    assert_eq!(dma.read(0xffe, U, &mut b), MemTxResult::DECODE_ERROR);
    assert_eq!(b, [1, 2, 0, 0]);
    assert_eq!(dma.write(0x10, U, &[1]), MemTxResult::DECODE_ERROR);
}

#[test]
fn many_ranges_use_the_eytzinger_layout() {
    let (sys, root, space) = setup();
    let mut ids = Vec::new();
    for i in 0..200u64 {
        let r = sys.new_reservation(&format!("r{i}"), 0x100).unwrap();
        sys.add_subregion(root, i * 0x1000 + 0x80, r).unwrap();
        ids.push(r);
    }
    let view = space.flatview();
    assert!(view.boundaries() > ruvm_mem::EYTZINGER_THRESHOLD);
    for (i, id) in ids.iter().enumerate() {
        let base = i as u64 * 0x1000 + 0x80;
        assert_eq!(view.lookup(base).map(FlatRange::region), Some(*id));
        assert_eq!(view.lookup(base + 0xff).map(FlatRange::region), Some(*id));
        assert!(view.lookup(base - 1).is_none());
        assert!(view.lookup(base + 0x100).is_none());
    }
    assert!(view.lookup(u64::MAX).is_none());
}

/// Records listener calls as strings.
struct Rec {
    tag: &'static str,
    priority: i32,
    log: Arc<Mutex<Vec<String>>>,
    fail_global_start: bool,
}

impl Rec {
    fn push(&self, s: String) {
        self.log.lock().unwrap().push(format!("{}:{s}", self.tag));
    }
}

impl MemoryListener for Rec {
    fn priority(&self) -> i32 {
        self.priority
    }
    fn begin(&self) {
        self.push("begin".into());
    }
    fn commit(&self) {
        self.push("commit".into());
    }
    fn region_add(&self, _s: &AddressSpace, r: &FlatRange) {
        self.push(format!("add {} {:x}", r.name(), r.addr()));
    }
    fn region_del(&self, _s: &AddressSpace, r: &FlatRange) {
        self.push(format!("del {} {:x}", r.name(), r.addr()));
    }
    fn region_nop(&self, _s: &AddressSpace, r: &FlatRange) {
        self.push(format!("nop {}", r.name()));
    }
    fn log_start(&self, _s: &AddressSpace, r: &FlatRange, old: DirtyMask, new: DirtyMask) {
        self.push(format!("log_start {} {:x}->{:x}", r.name(), old.bits(), new.bits()));
    }
    fn log_stop(&self, _s: &AddressSpace, r: &FlatRange, old: DirtyMask, new: DirtyMask) {
        self.push(format!("log_stop {} {:x}->{:x}", r.name(), old.bits(), new.bits()));
    }
    fn log_sync(&self, _s: &AddressSpace, r: &FlatRange) {
        self.push(format!("sync {}", r.name()));
    }
    fn log_global_start(&self) -> Result<(), MemError> {
        self.push("global_start".into());
        if self.fail_global_start { Err(MemError::Listener("no".into())) } else { Ok(()) }
    }
    fn log_global_stop(&self) {
        self.push("global_stop".into());
    }
}

fn rec(tag: &'static str, priority: i32, log: &Arc<Mutex<Vec<String>>>) -> Arc<Rec> {
    Arc::new(Rec { tag, priority, log: log.clone(), fail_global_start: false })
}

fn drain(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    std::mem::take(&mut log.lock().unwrap())
}

#[test]
fn listeners_hear_changes_in_qemu_order() {
    let (sys, root, space) = setup();
    let a = sys.new_reservation("a", 0x1000).unwrap();
    sys.add_subregion(root, 0, a).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let hi = sys.register_listener(rec("hi", 10, &log), &space).unwrap();
    sys.register_listener(rec("lo", 0, &log), &space).unwrap();
    assert_eq!(
        drain(&log),
        ["hi:begin", "hi:add a 0", "hi:commit", "lo:begin", "lo:add a 0", "lo:commit"]
    );

    let b = sys.new_reservation("b", 0x1000).unwrap();
    sys.add_subregion(root, 0x4000, b).unwrap();
    assert_eq!(
        drain(&log),
        [
            "lo:begin",
            "hi:begin",
            "lo:nop a",
            "hi:nop a",
            "lo:add b 4000",
            "hi:add b 4000",
            "lo:commit",
            "hi:commit"
        ]
    );

    sys.set_address(a, 0x8000).unwrap();
    assert_eq!(
        drain(&log),
        [
            "lo:begin",
            "hi:begin",
            "hi:del a 0",
            "lo:del a 0",
            "lo:nop b",
            "hi:nop b",
            "lo:add a 8000",
            "hi:add a 8000",
            "lo:commit",
            "hi:commit"
        ]
    );

    sys.unregister_listener(hi).unwrap();
    assert_eq!(drain(&log), ["hi:begin", "hi:del b 4000", "hi:del a 8000", "hi:commit"]);
    assert_eq!(sys.unregister_listener(hi), Err(MemError::NoSuchListener));
}

#[test]
fn vga_logging_marks_writes_and_snapshots_clear() {
    let (sys, root, space) = setup();
    let vram = sys.new_ram("vram", 0x10000).unwrap();
    sys.add_subregion(root, 0xa0000, vram).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    sys.register_listener(rec("l", 0, &log), &space).unwrap();
    drain(&log);

    space.write(0xa0000, U, &[1]);
    assert!(!sys.get_dirty(vram, 0, 1, DirtyClient::Vga).unwrap());

    sys.set_log(vram, true, DirtyClient::Vga).unwrap();
    assert_eq!(drain(&log), ["l:begin", "l:nop vram", "l:log_start vram 0->1", "l:commit"]);
    space.write(0xa1ffc, U, &[1, 2, 3, 4, 5]);
    assert!(sys.get_dirty(vram, 0x1000, 0x1000, DirtyClient::Vga).unwrap());
    assert!(sys.get_dirty(vram, 0x2000, 1, DirtyClient::Vga).unwrap());
    assert!(!sys.get_dirty(vram, 0x3000, 0x1000, DirtyClient::Vga).unwrap());
    assert!(!sys.get_dirty(vram, 0, 0x10000, DirtyClient::Migration).unwrap());

    let snap = sys.snapshot_and_clear_dirty(vram, 0, 0x10000, DirtyClient::Vga).unwrap();
    assert_eq!(drain(&log), ["l:sync vram"]);
    assert!(snap.get_dirty(0x1000, 1) && snap.get_dirty(0x2000, 1) && !snap.get_dirty(0, 0x1000));
    assert!(!sys.get_dirty(vram, 0, 0x10000, DirtyClient::Vga).unwrap());

    sys.set_dirty(vram, 0x5000, 1).unwrap();
    assert!(sys.get_dirty(vram, 0x5000, 1, DirtyClient::Vga).unwrap());
    sys.reset_dirty(vram, 0x5000, 1, DirtyClient::Vga).unwrap();
    assert!(!sys.get_dirty(vram, 0x5000, 1, DirtyClient::Vga).unwrap());

    // Logging nests, and only the last stop turns it off.
    sys.set_log(vram, true, DirtyClient::Vga).unwrap();
    sys.set_log(vram, false, DirtyClient::Vga).unwrap();
    assert!(drain(&log).is_empty());
    sys.set_log(vram, false, DirtyClient::Vga).unwrap();
    assert_eq!(drain(&log), ["l:begin", "l:nop vram", "l:log_stop vram 1->0", "l:commit"]);
    assert_eq!(sys.set_log(vram, true, DirtyClient::Migration), Err(MemError::InvalidClient));
}

#[test]
fn global_dirty_log_covers_ram_and_rolls_back() {
    let (sys, root, space) = setup();
    let ram = sys.new_ram("ram", 0x4000).unwrap();
    let io = sys.new_io("dev", 0x1000, Arc::new(Dev::default())).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    sys.add_subregion(root, 0x8000, io).unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    sys.register_listener(rec("a", 0, &log), &space).unwrap();
    sys.register_listener(rec("b", 1, &log), &space).unwrap();
    drain(&log);

    sys.global_dirty_log_start(GLOBAL_DIRTY_MIGRATION).unwrap();
    assert_eq!(
        drain(&log),
        [
            "a:global_start",
            "b:global_start",
            "a:begin",
            "b:begin",
            "a:nop ram",
            "b:nop ram",
            "a:log_start ram 0->4",
            "b:log_start ram 0->4",
            "a:nop dev",
            "b:nop dev",
            "a:commit",
            "b:commit"
        ]
    );
    space.write(0x3000, U, &[1]);
    assert!(sys.get_dirty(ram, 0x3000, 1, DirtyClient::Migration).unwrap());
    sys.global_dirty_log_sync(false);
    assert_eq!(drain(&log), ["a:sync ram", "b:sync ram"]);

    sys.global_dirty_log_stop(GLOBAL_DIRTY_MIGRATION);
    assert_eq!(
        drain(&log),
        [
            "a:begin",
            "b:begin",
            "a:nop ram",
            "b:nop ram",
            "b:log_stop ram 4->0",
            "a:log_stop ram 4->0",
            "a:nop dev",
            "b:nop dev",
            "a:commit",
            "b:commit",
            "b:global_stop",
            "a:global_stop"
        ]
    );

    let failing =
        Arc::new(Rec { tag: "c", priority: 2, log: log.clone(), fail_global_start: true });
    sys.register_listener(failing, &space).unwrap();
    drain(&log);
    assert!(sys.global_dirty_log_start(GLOBAL_DIRTY_MIGRATION).is_err());
    assert_eq!(
        drain(&log),
        ["a:global_start", "b:global_start", "c:global_start", "b:global_stop", "a:global_stop"]
    );
    assert_eq!(sys.global_dirty_tracking(), 0);
}

#[test]
fn code_client_follows_the_config() {
    let sys = MemorySystem::with_config(MemoryConfig { page_bits: 12, code_dirty_log: true });
    let root = sys.new_container("system", SPACE).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    let ram = sys.new_ram("ram", 0x2000).unwrap();
    sys.add_subregion(root, 0, ram).unwrap();
    assert_eq!(space.flatview().ranges()[0].dirty_log_mask(), DirtyClient::Code.mask());
    space.write(0x1000, U, &[1]);
    assert!(sys.get_dirty(ram, 0x1000, 1, DirtyClient::Code).unwrap());
}

#[test]
fn listener_reentry_panics_instead_of_deadlocking() {
    struct Bad(Arc<MemorySystem>);
    impl MemoryListener for Bad {
        fn begin(&self) {
            self.0.address_spaces();
        }
    }
    let sys = Arc::new(MemorySystem::new());
    let root = sys.new_container("system", SPACE).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    let s2 = sys.clone();
    let r = std::thread::spawn(move || s2.register_listener(Arc::new(Bad(s2.clone())), &space));
    assert!(r.join().is_err());
}
