// SPDX-License-Identifier: GPL-2.0-or-later

//! Migration of the virt flash: the state `vmstate_pflash` carries, taken from one flash in the
//! middle of a command and given to another.

use std::sync::Arc;

use ruvm_machine_arm::pflash::{Pflash, PflashBacking, PflashProps, PflashVmState};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemTxResult, MemorySystem};

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const SIZE: u64 = 4 << 20;
const BLOCK: u64 = 0x40000;

struct Flash {
    mem: Arc<MemorySystem>,
    space: Arc<AddressSpace>,
    dev: Arc<Pflash>,
}

impl Flash {
    fn new() -> Flash {
        let mem = Arc::new(MemorySystem::new());
        let root = mem.new_container("system", 1 << 64).unwrap();
        let space = mem.address_space_init(root, "memory").unwrap();
        let props = PflashProps::virt_flash(SIZE);
        let dev = Pflash::new(&mem, "virt.flash1", props, PflashBacking::None).unwrap();
        mem.add_subregion(root, 0, dev.region()).unwrap();
        Flash { mem, space, dev }
    }

    fn store(&self, addr: u64, value: u32) {
        let r = self.space.store(addr, 4, u64::from(value), Endian::Little, U);
        assert_eq!(r, MemTxResult::OK);
    }

    fn load(&self, addr: u64) -> u32 {
        let (v, r) = self.space.load(addr, 4, Endian::Little, U);
        assert_eq!(r, MemTxResult::OK);
        v as u32
    }

    fn romd(&self) -> bool {
        self.mem.region(self.dev.region()).unwrap().romd_mode
    }
}

#[test]
fn write_to_buffer_finishes_on_the_destination() {
    let src = Flash::new();
    // Write to buffer of two words, one written before the migration.
    src.store(BLOCK, 0x00e8_00e8);
    src.store(BLOCK, 0x0001_0001);
    src.store(BLOCK, 0x1122_3344);
    let s = src.dev.vmstate_save();
    assert_eq!((s.wcycle, s.cmd, s.counter, s.blk_offset), (2, 0xe8, 0, BLOCK as u32));
    assert_eq!(s.blk_bytes.len(), src.dev.writeblock_size() as usize);

    let dst = Flash::new();
    assert!(dst.romd());
    dst.dev.vmstate_load(&s).unwrap();
    assert_eq!(dst.dev.vmstate_save(), s);
    assert!(!dst.romd());
    dst.store(BLOCK + 4, 0x5566_7788);
    dst.store(BLOCK, 0x00d0_00d0);
    assert_eq!(dst.dev.status(), 0x80);
    dst.store(0, 0x00ff_00ff);
    assert!(dst.romd());
    assert_eq!((dst.load(BLOCK), dst.load(BLOCK + 4)), (0x1122_3344, 0x5566_7788));
    assert_eq!(dst.dev.vmstate_save().blk_offset, u32::MAX);
}

#[test]
fn status_mode_reads_the_status_on_the_destination() {
    let src = Flash::new();
    src.store(0, 0x0070_0070);
    let dst = Flash::new();
    dst.dev.vmstate_load(&src.dev.vmstate_save()).unwrap();
    assert!(!dst.romd());
    assert_eq!(dst.load(0x10), 0x0080_0080);
}

#[test]
fn load_checks_the_write_buffer() {
    let dev = Flash::new().dev;
    let good = dev.vmstate_save();
    let past_the_end = PflashVmState { blk_offset: SIZE as u32 - 4, ..good.clone() };
    assert!(dev.vmstate_load(&past_the_end).is_err());
    let short = PflashVmState { blk_bytes: vec![0; 4], ..good.clone() };
    assert!(dev.vmstate_load(&short).is_err());
    dev.vmstate_load(&good).unwrap();
}
