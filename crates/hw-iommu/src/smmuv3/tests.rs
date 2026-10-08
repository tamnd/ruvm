// SPDX-License-Identifier: GPL-2.0-or-later

//! The SMMUv3 model driven through its registers and queues, with DMA going through an IOMMU
//! region as a PCI device behind it would.

use std::sync::atomic::{AtomicU32, Ordering};

use ruvm_mem::{Endian, MemorySystem};

use super::*;

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const BASE: u64 = 0x0900_0000;
const STRTAB: u64 = 0x10000;
const CD: u64 = 0x20000;
const TTB: u64 = 0x30000;
const CMDQ: u64 = 0x50000;
const EVTQ: u64 = 0x60000;
const VTTB: u64 = 0x70000;
const SID: u32 = 8;
const AF: u64 = 1 << 10;

struct Rig {
    _sys: MemorySystem,
    space: Arc<AddressSpace>,
    smmu: Arc<SmmuV3>,
    irqs: Arc<[AtomicU32; 4]>,
    dma: Arc<AddressSpace>,
    prod: u32,
}

impl Rig {
    fn new(stages: Stage) -> Rig {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", 0x100_0000).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let irqs: Arc<[AtomicU32; 4]> = Arc::new(Default::default());
        let lines = std::array::from_fn(|i| {
            let irqs = Arc::clone(&irqs);
            IrqLine::from_fn(move |level| {
                if level != 0 {
                    irqs[i].fetch_add(1, Ordering::SeqCst);
                }
            })
        });
        let smmu = SmmuV3::new(stages, &space, lines);
        let mmio = sys.new_io("smmuv3", u128::from(SMMU_SIZE), smmu.mmio_ops()).unwrap();
        sys.add_subregion(root, BASE, mmio).unwrap();
        let iommu = sys.new_iommu("smmuv3-iommu", 1 << 64, smmu.device_ops(SID)).unwrap();
        let dma = sys.address_space_init(iommu, "dma").unwrap();
        Rig { _sys: sys, space, smmu, irqs, dma, prod: 0 }
    }

    fn w32(&self, off: u64, v: u32) {
        assert!(self.space.store(BASE + off, 4, u64::from(v), Endian::Little, U).is_ok());
    }

    fn r32(&self, off: u64) -> u32 {
        let (v, r) = self.space.load(BASE + off, 4, Endian::Little, U);
        assert!(r.is_ok());
        v as u32
    }

    fn w64(&self, off: u64, v: u64) {
        assert!(self.space.store(BASE + off, 8, v, Endian::Little, U).is_ok());
    }

    fn wq(&self, addr: u64, v: u64) {
        assert!(self.space.write(addr, U, &v.to_le_bytes()).is_ok());
    }

    fn words(&self, addr: u64, w: &[u32]) {
        let b: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(self.space.write(addr, U, &b).is_ok());
    }

    fn rd(&self, addr: u64) -> u32 {
        let mut b = [0u8; 4];
        assert!(self.space.read(addr, U, &mut b).is_ok());
        u32::from_le_bytes(b)
    }

    fn irq(&self, i: usize) -> u32 {
        self.irqs[i].load(Ordering::SeqCst)
    }

    /// Queues a command and moves CMDQ_PROD past it.
    fn cmd(&mut self, w: [u32; 4]) {
        let idx = u64::from(self.prod & 0xf);
        self.words(CMDQ + idx * 16, &w);
        self.prod = (self.prod + 1) & 0x1f;
        self.w32(A_CMDQ_PROD, self.prod);
    }

    /// A linear stream table of 256 entries, 16 entry queues, everything enabled.
    fn enable(&self) {
        self.w64(A_STRTAB_BASE, STRTAB);
        self.w32(A_STRTAB_BASE_CFG, 8);
        self.w64(A_CMDQ_BASE, CMDQ | 4);
        self.w64(A_EVENTQ_BASE, EVTQ | 4);
        self.w32(A_IRQ_CTRL, IRQ_CTRL_GERROR_IRQEN | IRQ_CTRL_EVENTQ_IRQEN);
        self.w32(A_CR0, CR0_SMMUEN | CR0_EVENTQEN | CR0_CMDQEN);
        assert_eq!(self.r32(A_CR0ACK), CR0_SMMUEN | CR0_EVENTQEN | CR0_CMDQEN);
    }

    /// The STE of `SID` with CONFIG `config` and the CD at `CD`.
    fn ste(&self, config: u32, w: [u32; 8]) {
        let mut ste = [0u32; 16];
        ste[..8].copy_from_slice(&w);
        ste[0] = 1 | (config << 1) | CD as u32;
        self.words(STRTAB + u64::from(SID) * 64, &ste);
    }

    /// A CD with a 48 bit TTB0 region of 4K pages at `TTB`, ASID 7, faults recorded.
    fn cd(&self) {
        let w0 = 16 | (1 << 30) | (1 << 31);
        let w1 = 4 | (1 << 9) | (1 << 13) | (1 << 14) | (7 << 16);
        self.words(CD, &[w0, w1, TTB as u32, 0, 0, 0, 0, 0]);
    }

    /// Maps the 4K page `iova` to `pa` in the stage 1 tables at `TTB`.
    fn map(&self, iova: u64, pa: u64, ap: u64) {
        let idx = |l: u32| (iova >> (12 + 9 * (3 - l))) & 0x1ff;
        self.wq(TTB + idx(0) * 8, (TTB + 0x1000) | 3);
        self.wq(TTB + 0x1000 + idx(1) * 8, (TTB + 0x2000) | 3);
        self.wq(TTB + 0x2000 + idx(2) * 8, (TTB + 0x3000) | 3);
        self.wq(TTB + 0x3000 + idx(3) * 8, pa | AF | (ap << 6) | 3);
    }

    fn dma_write(&self, addr: u64, v: u32) -> bool {
        self.dma
            .write(addr, MemTxAttrs::new().with_requester_id(SID as u16), &v.to_le_bytes())
            .is_ok()
    }
}

#[test]
fn id_registers_and_reset() {
    let r = Rig::new(Stage::Nested);
    assert_eq!(r.r32(A_IDR0), 0x0d44_101b);
    assert_eq!(r.r32(0x4), 0x0273_0010);
    assert_eq!(r.r32(0xc), 0x1414);
    assert_eq!(r.r32(A_IDR5), 0x74);
    assert_eq!(r.r32(A_AIDR), 1);
    assert_eq!(r.r32(A_GBPA), GBPA_RESET_VAL);
    assert_eq!(r.r32(A_CMDQ_BASE), 19);
    assert_eq!(r.r32(A_EVENTQ_BASE), 19);
    assert_eq!(r.r32(0xfe0), 0x84);
    assert_eq!(r.r32(0xffc), 0xb1);
    // Page 1 aliases page 0.
    r.w32(0x10000 + A_CR1, 0x55);
    assert_eq!(r.r32(A_CR1), 0x55);
    r.enable();
    assert_eq!(r.r32(0x10000 + A_CR0), 13);
    r.smmu.reset();
    assert_eq!(r.r32(A_CR0), 0);
    assert_eq!(r.r32(A_CR0ACK), 0);
    // CR1 and the table bases survive a reset, as in QEMU.
    assert_eq!(r.r32(A_CR1), 0x55);
    assert_eq!(r.r32(A_STRTAB_BASE), STRTAB as u32);
    assert_eq!(r.r32(A_CMDQ_BASE), CMDQ as u32 | 19);
    // An S1 only SMMU has no S2P and no XNX.
    let r = Rig::new(Stage::S1);
    assert_eq!(r.r32(A_IDR0) & 3, 2);
    assert_eq!(r.r32(0xc), 0x1404);
}

#[test]
fn stage1_translation_and_invalidation() {
    let mut r = Rig::new(Stage::Nested);
    r.enable();
    r.ste(5, [0; 8]);
    r.cd();
    r.map(0x1000, 0x40000, 0);
    assert!(r.dma_write(0x1010, 0xdead_beef));
    assert_eq!(r.rd(0x40010), 0xdead_beef);
    // The IOTLB keeps the old mapping until the guest invalidates it.
    r.map(0x1000, 0x41000, 0);
    assert!(r.dma_write(0x1010, 1));
    assert_eq!(r.rd(0x40010), 1);
    // TLBI_NH_VA for ASID 7 at 0x1000, then a CMD_SYNC with an interrupt.
    r.cmd([u32::from(cmd::TLBI_NH_VA), 7 << 16, 0x1000, 0]);
    r.cmd([u32::from(cmd::SYNC) | (1 << 12), 0, 0, 0]);
    assert_eq!(r.irq(IRQ_CMD_SYNC), 1);
    assert_eq!(r.r32(A_CMDQ_CONS), 2);
    assert!(r.dma_write(0x1010, 2));
    assert_eq!(r.rd(0x41010), 2);
    // A read only page refuses writes.
    r.map(0x2000, 0x42000, 2);
    assert!(!r.dma_write(0x2000, 3));
    let mut b = [0u8; 4];
    assert!(r.dma.read(0x2000, U, &mut b).is_ok());
}

#[test]
fn faults_go_to_the_event_queue() {
    let r = Rig::new(Stage::Nested);
    r.enable();
    r.ste(5, [0; 8]);
    r.cd();
    assert!(!r.dma_write(0x5008, 0));
    assert_eq!(r.r32(A_EVENTQ_PROD), 1);
    assert_eq!(r.irq(IRQ_EVTQ), 1);
    // F_TRANSLATION for SID 8, a write of class IN at 0x5008.
    assert_eq!(r.rd(EVTQ), u32::from(evt::F_TRANSLATION));
    assert_eq!(r.rd(EVTQ + 4), SID);
    assert_eq!(r.rd(EVTQ + 12), u32::from(CLASS_IN) << 8);
    assert_eq!(r.rd(EVTQ + 16), 0x5008);
    // A read sets RnW.
    let mut b = [0u8; 4];
    assert!(!r.dma.read(0x6000, U, &mut b).is_ok());
    assert_eq!(r.rd(EVTQ + 32 + 12), (u32::from(CLASS_IN) << 8) | (1 << 3));
    // Without CD.R nothing is recorded.
    let mut cd1 = 4 | (1 << 9) | (1 << 14) | (7 << 16);
    r.words(CD + 4, &[cd1]);
    r.smmu.lock().configs.clear();
    assert!(!r.dma_write(0x7000, 0));
    assert_eq!(r.r32(A_EVENTQ_PROD), 2);
    // A CD without A is a bad CD, which is always recorded.
    cd1 &= !(1 << 14);
    r.words(CD + 4, &[cd1]);
    r.smmu.lock().configs.clear();
    assert!(!r.dma_write(0x7000, 0));
    assert_eq!(r.r32(A_EVENTQ_PROD), 3);
    assert_eq!(r.rd(EVTQ + 64), u32::from(evt::C_BAD_CD));
    // A stream ID beyond the table is a bad stream ID.
    r.w32(A_STRTAB_BASE_CFG, 3);
    assert!(!r.dma_write(0x7000, 0));
    assert_eq!(r.rd(EVTQ + 96), u32::from(evt::C_BAD_STREAMID));
}

#[test]
fn a_full_event_queue_raises_gerror() {
    let r = Rig::new(Stage::Nested);
    r.enable();
    r.ste(5, [0; 8]);
    r.cd();
    for i in 0..16 {
        assert!(!r.dma_write(0x5000 + i * 0x1000, 0));
    }
    assert_eq!(r.r32(A_EVENTQ_PROD), 0x10);
    assert_eq!(r.r32(A_GERROR), 0);
    assert!(!r.dma_write(0x5000, 0));
    assert_eq!(r.r32(A_GERROR), GERROR_EVENTQ_ABT_ERR);
    assert_eq!(r.irq(IRQ_GERROR), 1);
}

#[test]
fn illegal_commands_stop_the_queue() {
    let mut r = Rig::new(Stage::Nested);
    r.enable();
    // CFGI_ALL with its own opcode is not implemented.
    r.cmd([7, 0, 0, 0]);
    r.cmd([u32::from(cmd::SYNC) | (1 << 12), 0, 0, 0]);
    assert_eq!(r.r32(A_CMDQ_CONS), CERROR_ILL << 24);
    assert_eq!(r.r32(A_GERROR), GERROR_CMDQ_ERR);
    assert_eq!(r.irq(IRQ_GERROR), 1);
    assert_eq!(r.irq(IRQ_CMD_SYNC), 0);
    // The guest skips the command and acknowledges the error.
    r.w32(A_CMDQ_CONS, 1);
    r.w32(A_GERRORN, GERROR_CMDQ_ERR);
    assert_eq!(r.r32(A_CMDQ_CONS), 2);
    assert_eq!(r.irq(IRQ_CMD_SYNC), 1);
}

#[test]
fn gbpa_and_ste_bypass() {
    let mut r = Rig::new(Stage::Nested);
    // With SMMUEN clear and GBPA.ABORT clear, DMA bypasses.
    assert!(r.dma_write(0x8000, 5));
    assert_eq!(r.rd(0x8000), 5);
    // GBPA only changes with UPDATE set.
    r.w32(A_GBPA, GBPA_ABORT);
    assert_eq!(r.r32(A_GBPA), GBPA_RESET_VAL);
    r.w32(A_GBPA, GBPA_ABORT | GBPA_UPDATE);
    assert_eq!(r.r32(A_GBPA), GBPA_ABORT);
    assert!(!r.dma_write(0x8000, 6));
    // An STE with CONFIG = 4 bypasses, and with CONFIG = 0 aborts without an event.
    r.enable();
    r.ste(4, [0; 8]);
    assert!(r.dma_write(0x8000, 7));
    assert_eq!(r.rd(0x8000), 7);
    r.ste(0, [0; 8]);
    r.cmd([u32::from(cmd::CFGI_STE), SID, 0, 0]);
    assert!(!r.dma_write(0x8000, 8));
    assert_eq!(r.r32(A_EVENTQ_PROD), 0);
    // An invalid STE is a bad STE.
    r.words(STRTAB + u64::from(SID) * 64, &[0]);
    r.cmd([u32::from(cmd::CFGI_STE_RANGE), 0, 31, 0]);
    assert!(!r.dma_write(0x8000, 8));
    assert_eq!(r.rd(EVTQ), u32::from(evt::C_BAD_STE));
}

#[test]
fn two_level_stream_table() {
    let r = Rig::new(Stage::Nested);
    r.enable();
    // FMT = 2-level, SPLIT = 6, LOG2SIZE = 16: SID 8 is entry 8 of the first L2 table.
    r.w32(A_STRTAB_BASE_CFG, (1 << 16) | (6 << 6) | 16);
    let l2 = 0x80000u64;
    // SPAN = 7 covers the 64 entries of the L2 table.
    r.wq(STRTAB, l2 | 7);
    r.words(l2 + u64::from(SID) * 64, &[1 | (4 << 1)]);
    assert!(r.dma_write(0x9000, 9));
    assert_eq!(r.rd(0x9000), 9);
    // SPAN = 0 makes the L2 pointer invalid.
    r.wq(STRTAB, l2);
    r.smmu.lock().configs.clear();
    assert!(!r.dma_write(0x9000, 9));
    assert_eq!(r.rd(EVTQ), u32::from(evt::C_BAD_STREAMID));
    // An unreadable L1 descriptor is an STE fetch fault with the address in ADDR2.
    r.w64(A_STRTAB_BASE, 0x4000_0000);
    assert!(!r.dma_write(0x9000, 9));
    assert_eq!(r.rd(EVTQ + 32), u32::from(evt::F_STE_FETCH));
    assert_eq!(r.rd(EVTQ + 32 + 24), 0x4000_0000);
}

/// STE words 4 to 7 for stage 2 with VMID 1, a 40 bit IPA from level 1 with 4K pages, a 44
/// bit output, faults recorded, and the tables at `VTTB`.
fn s2_words() -> [u32; 8] {
    let w5 = 24 | (1 << 6) | (4 << 16) | (1 << 19) | (1 << 26);
    [0, 0, 0, 0, 1, w5, VTTB as u32, 0]
}

#[test]
fn stage2_translation() {
    let mut r = Rig::new(Stage::Nested);
    r.enable();
    r.ste(6, s2_words());
    // A 2M block at level 2 maps IPA 0x20_0000 to 0x60_0000, read and write.
    r.wq(VTTB, 0x90000 | 3);
    r.wq(0x90000 + 8, 0x60_0000 | AF | (3 << 6) | 1);
    assert!(r.dma_write(0x20_0040, 0x77));
    assert_eq!(r.rd(0x60_0040), 0x77);
    // Unmapped IPAs fault at stage 2 with the IPA in ADDR2.
    assert!(!r.dma_write(0x40_0000, 0));
    assert_eq!(r.rd(EVTQ), u32::from(evt::F_TRANSLATION));
    assert_eq!(r.rd(EVTQ + 12), (1 << 7) | (u32::from(CLASS_IN) << 8));
    assert_eq!(r.rd(EVTQ + 24), 0x40_0000);
    // TLBI_S2_IPA drops the block, after which the new mapping shows.
    r.wq(0x90000 + 8, 0x80_0000 | AF | (3 << 6) | 1);
    r.cmd([u32::from(cmd::TLBI_S2_IPA), 1, 0x20_0000, 0]);
    assert!(r.dma_write(0x20_0040, 0x78));
    assert_eq!(r.rd(0x80_0040), 0x78);
    // A bad S2T0SZ makes the STE bad.
    let mut w = s2_words();
    w[5] = (w[5] & !0x3f) | 10;
    r.ste(6, w);
    r.cmd([u32::from(cmd::CFGI_STE), SID, 0, 0]);
    assert!(!r.dma_write(0x20_0040, 0));
    assert_eq!(r.rd(EVTQ + 32), u32::from(evt::C_BAD_STE));
}

#[test]
fn nested_translation() {
    let mut r = Rig::new(Stage::Nested);
    r.enable();
    r.ste(7, s2_words());
    // Stage 2 maps the first 1G of IPA onto PA 0x0 with one block, so the CD and the stage 1
    // tables are where their IPAs say.
    r.wq(VTTB, AF | (3 << 6) | 1);
    r.cd();
    r.map(0x3000, 0x43000, 0);
    assert!(r.dma_write(0x3004, 0x99));
    assert_eq!(r.rd(0x43004), 0x99);
    // TLBI_NH_ALL with stage 2 drops the stage 1 entries of VMID 1 only.
    r.map(0x3000, 0x44000, 0);
    r.cmd([u32::from(cmd::TLBI_NH_ALL), 1, 0, 0]);
    assert!(r.dma_write(0x3004, 0x9a));
    assert_eq!(r.rd(0x44004), 0x9a);
    // Making stage 2 read only turns the write into a stage 2 permission fault.
    r.wq(VTTB, AF | (1 << 6) | 1);
    r.cmd([u32::from(cmd::TLBI_S12_VMALL), 1, 0, 0]);
    assert!(!r.dma_write(0x3004, 0));
    assert_eq!(r.rd(EVTQ), u32::from(evt::F_PERMISSION));
    assert_eq!(r.rd(EVTQ + 12), (1 << 7) | (u32::from(CLASS_IN) << 8));
    assert_eq!(r.rd(EVTQ + 16), 0x3004);
    assert_eq!(r.rd(EVTQ + 24), 0x44004);
}

#[test]
fn range_invalidation_splits_into_powers_of_two() {
    assert_eq!(dma_aligned_pow2_mask(0x1000, 0x4fff), 0xfff);
    assert_eq!(dma_aligned_pow2_mask(0x2000, 0x4fff), 0x1fff);
    assert_eq!(dma_aligned_pow2_mask(0, u64::MAX), u64::MAX);
    let mut s = State::new(Stage::S1);
    let mut cfg = TransCfg { asid: 3, ..TransCfg::default() };
    cfg.s2cfg.vmid = -1;
    for i in 0..8u64 {
        let e = TlbEntry {
            iova: i << 12,
            addr_mask: 0xfff,
            level: 3,
            granule: 12,
            ..TlbEntry::default()
        };
        s.iotlb.insert(&cfg, e);
    }
    // TG = 4K, NUM = 2 and SCALE = 0: three pages from 0x1000.
    s.range_inval(
        &[u32::from(cmd::TLBI_NH_VA) | (2 << 12), 3 << 16, 0x1000 | (1 << 10), 0],
        Stage::S1,
    );
    assert_eq!(s.iotlb.len(), 5);
    // Another ASID is not touched by NH_VA, but NH_VAA covers every ASID.
    s.range_inval(&[u32::from(cmd::TLBI_NH_VA), 4 << 16, 0x6000 | (1 << 10), 0], Stage::S1);
    assert_eq!(s.iotlb.len(), 5);
    s.range_inval(&[u32::from(cmd::TLBI_NH_VAA), 4 << 16, 0x6000 | (1 << 10), 0], Stage::S1);
    assert_eq!(s.iotlb.len(), 4);
}

#[test]
fn registers_of_the_engaged_smmu_fail() {
    let mut r = Rig::new(Stage::Nested);
    r.enable();
    // A command queue over the SMMU's own registers reads as an abort, not a deadlock.
    r.w64(A_CMDQ_BASE, BASE | 4);
    r.prod = 0;
    r.w32(A_CMDQ_PROD, 1);
    assert_eq!(r.r32(A_CMDQ_CONS), CERROR_ABT << 24);
}
