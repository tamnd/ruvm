// SPDX-License-Identifier: GPL-2.0-or-later

//! The VT-d model driven through its registers and invalidation queue, with DMA going through an
//! IOMMU region as a PCI device behind it would.

use std::sync::Mutex as StdMutex;

use ruvm_mem::{Endian, MemorySystem, register_iommu_notifier, unregister_iommu_notifier};

use super::*;

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const ROOT: u64 = 0x10000;
const CTX: u64 = 0x11000;
const PML4: u64 = 0x20000;
const PDPT: u64 = 0x21000;
const PD: u64 = 0x22000;
const PT: u64 = 0x23000;
const IQ: u64 = 0x30000;
const DATA: u64 = 0x40000;
/// 00:01.0.
const SID: u16 = 0x08;

struct Rig {
    _sys: MemorySystem,
    space: Arc<AddressSpace>,
    iommu: Arc<IntelIommu>,
    dma: Arc<AddressSpace>,
    msis: Arc<StdMutex<Vec<(u64, u32)>>>,
    tail: u64,
}

impl Rig {
    fn new(conf: IntelIommuConf) -> Rig {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", 0x100_0000).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let iommu = IntelIommu::new(conf, &space);
        let msis = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&msis);
        iommu.set_msi_sender(move |a, d| sink.lock().unwrap().push((a, d)));
        let mmio = sys.new_io("intel_iommu", u128::from(DMAR_REG_SIZE), iommu.mmio_ops()).unwrap();
        sys.add_subregion(root, DMAR_REG_BASE, mmio).unwrap();
        let region = sys.new_iommu("vtd", 1 << 64, iommu.device_ops(SID)).unwrap();
        let dma = sys.address_space_init(region, "dma").unwrap();
        Rig { _sys: sys, space, iommu, dma, msis, tail: 0 }
    }

    fn w32(&self, off: u64, v: u32) {
        assert!(self.space.store(DMAR_REG_BASE + off, 4, u64::from(v), Endian::Little, U).is_ok());
    }

    fn w64(&self, off: u64, v: u64) {
        assert!(self.space.store(DMAR_REG_BASE + off, 8, v, Endian::Little, U).is_ok());
    }

    fn r32(&self, off: u64) -> u32 {
        let (v, r) = self.space.load(DMAR_REG_BASE + off, 4, Endian::Little, U);
        assert!(r.is_ok());
        v as u32
    }

    fn r64(&self, off: u64) -> u64 {
        let (v, r) = self.space.load(DMAR_REG_BASE + off, 8, Endian::Little, U);
        assert!(r.is_ok());
        v
    }

    fn wq(&self, addr: u64, v: u64) {
        assert!(self.space.write(addr, U, &v.to_le_bytes()).is_ok());
    }

    fn rd32(&self, addr: u64) -> u32 {
        let mut b = [0u8; 4];
        assert!(self.space.read(addr, U, &mut b).is_ok());
        u32::from_le_bytes(b)
    }

    /// A DMA read of 4 bytes by the device, through the IOMMU.
    fn dma_read(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.dma.read(addr, U, &mut b).is_ok().then(|| u32::from_le_bytes(b))
    }

    fn dma_write(&self, addr: u64, v: u32) -> bool {
        self.dma.write(addr, U, &v.to_le_bytes()).is_ok()
    }

    /// Root and context entries for [`SID`] in domain 5 with a 4-level table, the way Linux
    /// sets them up, then SRTP and TE.
    fn enable(&self, tt: u64) {
        self.wq(ROOT, CTX | 1);
        self.wq(CTX + u64::from(SID) * 16, PML4 | tt | 1);
        self.wq(CTX + u64::from(SID) * 16 + 8, 5 << 8 | 2);
        self.w64(DMAR_RTADDR_REG, ROOT);
        self.w32(DMAR_GCMD_REG, GCMD_SRTP);
        assert_ne!(self.r32(DMAR_GSTS_REG) & GSTS_RTPS, 0);
        self.w32(DMAR_GCMD_REG, GCMD_TE);
        assert_ne!(self.r32(DMAR_GSTS_REG) & GSTS_TES, 0);
    }

    /// Maps the 4K page at `iova` to `gpa` with the access bits `rw`.
    fn map(&self, iova: u64, gpa: u64, rw: u64) {
        self.wq(PML4 + ((iova >> 39) & 0x1ff) * 8, PDPT | 3);
        self.wq(PDPT + ((iova >> 30) & 0x1ff) * 8, PD | 3);
        self.wq(PD + ((iova >> 21) & 0x1ff) * 8, PT | 3);
        self.wq(PT + ((iova >> 12) & 0x1ff) * 8, gpa | rw);
    }

    /// Turns on the invalidation queue, one 4K page of 128-bit descriptors.
    fn enable_qi(&mut self) {
        self.w64(DMAR_IQA_REG, IQ);
        self.w64(DMAR_IQT_REG, 0);
        self.w32(DMAR_GCMD_REG, GCMD_TE | GCMD_QIE);
        assert_ne!(self.r32(DMAR_GSTS_REG) & GSTS_QIES, 0);
    }

    fn submit(&mut self, lo: u64, hi: u64) {
        self.wq(IQ + self.tail * 16, lo);
        self.wq(IQ + self.tail * 16 + 8, hi);
        self.tail += 1;
        self.w64(DMAR_IQT_REG, self.tail << 4);
    }

    fn msis(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.msis.lock().unwrap())
    }
}

#[test]
fn realize_resolves_and_rejects_like_qemu() {
    // Remapping is on without KVM, and eim only follows it with a KVM irqchip.
    let c = IntelIommuConf::default().realize(false, false).unwrap();
    assert_eq!(c.intremap, OnOffAuto::On);
    assert_eq!(c.eim, OnOffAuto::Off);
    assert!(c.intr_supported());
    let err = |c: IntelIommuConf, all: bool| c.realize(all, all).unwrap_err();
    let on = IntelIommuConf { intremap: OnOffAuto::On, ..Default::default() };
    assert_eq!(
        err(on, true),
        "Interrupt Remapping cannot work with kernel-irqchip=on, please use 'split|off'."
    );
    let eim = IntelIommuConf { eim: OnOffAuto::On, ..Default::default() };
    let c = eim.realize(false, false).unwrap();
    assert_eq!((c.intremap, c.eim), (OnOffAuto::On, OnOffAuto::On));
    let c = IntelIommuConf { intremap: OnOffAuto::On, ..Default::default() }
        .realize(false, true)
        .unwrap();
    assert_eq!(c.eim, OnOffAuto::On);
    // The eim check comes after remapping is decided, so aw-bits is the first error here.
    assert_eq!(
        err(IntelIommuConf { aw_bits: 40, ..eim.clone() }, false),
        "Legacy mode: supported values for aw-bits are: 39, 48"
    );
    assert_eq!(
        err(IntelIommuConf { intremap: OnOffAuto::Off, ..eim }, false),
        "eim=on cannot be selected without intremap=on"
    );
    let c = IntelIommuConf::default().realize(true, true).unwrap();
    assert_eq!((c.intremap, c.eim), (OnOffAuto::Off, OnOffAuto::Off));
    assert_eq!(
        err(IntelIommuConf { fsts: true, ..Default::default() }, false),
        "fsts is only available in scalable mode"
    );
    assert_eq!(
        err(IntelIommuConf { aw_bits: 40, ..Default::default() }, false),
        "Legacy mode: supported values for aw-bits are: 39, 48"
    );
    let sm = IntelIommuConf { scalable_mode: true, ..Default::default() };
    assert_eq!(
        err(IntelIommuConf { aw_bits: 52, ..sm.clone() }, false),
        "Scalable mode(fsts=off): supported values for aw-bits are: 39, 48"
    );
    assert_eq!(
        err(IntelIommuConf { pasid_bits: 8, ..Default::default() }, false),
        "Need to set scalable mode for PASID"
    );
    assert_eq!(
        err(IntelIommuConf { pasid_bits: 21, ..sm.clone() }, false),
        "PASID width 21 exceeds Max PASID Width 20 allowed in PCI spec"
    );
    assert_eq!(
        err(IntelIommuConf { svm: true, ..sm.clone() }, false),
        "Need to set device IOTLB for svm"
    );
    let c = sm.realize(false, false).unwrap();
    assert!(c.scalable_mode);
    let fs = IntelIommuConf { scalable_mode: true, fsts: true, ..Default::default() };
    assert_eq!(
        err(IntelIommuConf { aw_bits: 39, ..fs.clone() }, false),
        "Scalable mode(fsts=on): supported value for aw-bits is: 48"
    );
    assert!(fs.realize(false, false).is_ok());
}

#[test]
fn capabilities_match_qemu_defaults() {
    let r = Rig::new(IntelIommuConf::default());
    assert_eq!(r.r32(DMAR_VER_REG), 0x10);
    assert_eq!(r.r64(DMAR_CAP_REG), 0x80d2_008c_222f_0606);
    assert_eq!(r.r64(DMAR_ECAP_REG), 0xf42);
    let r = Rig::new(IntelIommuConf {
        aw_bits: 39,
        device_iotlb: true,
        caching_mode: true,
        ..Default::default()
    });
    assert_eq!(r.r64(DMAR_CAP_REG), 0x80d2_008c_2226_0286);
    assert_eq!(r.r64(DMAR_ECAP_REG), 0xf46);
    assert_eq!(r.r32(DMAR_FECTL_REG), FECTL_IM);
    let ir = IntelIommuConf::default().realize(false, false).unwrap();
    assert_eq!(Rig::new(ir).r64(DMAR_ECAP_REG), 0xf0_0f4a);
    let eim = IntelIommuConf { eim: OnOffAuto::On, ..Default::default() };
    assert_eq!(Rig::new(eim.realize(false, false).unwrap()).r64(DMAR_ECAP_REG), 0xf0_0f5a);
}

#[test]
fn translation_off_is_identity() {
    let r = Rig::new(IntelIommuConf::default());
    r.wq(DATA, 0x1234_5678);
    assert_eq!(r.dma_read(DATA), Some(0x1234_5678));
}

#[test]
fn second_stage_walk_translates_and_checks_access() {
    let r = Rig::new(IntelIommuConf::default());
    r.map(0x7fff_f000, DATA, 1);
    r.wq(DATA + 0x10, 0xcafe);
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    assert!(!r.dma_write(0x7fff_f010, 1));
    let frcd_hi = r.r64(DMAR_FRCD_REG_0_2);
    assert_eq!(frcd_hi, FRCD_F | u64::from(FR_WRITE) << 32 | u64::from(SID));
    assert_eq!(r.r64(DMAR_FRCD_REG_0_0), 0x7fff_f000);
    assert_eq!(r.r32(DMAR_FSTS_REG) & FSTS_PPF, FSTS_PPF);
    // FECTL.IM is set at reset, so the event stays pending.
    assert!(r.msis().is_empty());
    assert_ne!(r.r32(DMAR_FECTL_REG) & FECTL_IP, 0);
    assert_eq!(r.dma_read(0x7fff_f010), Some(0xcafe));
    // The cached read only entry refuses the write without a second fault.
    assert!(!r.dma_write(0x7fff_f010, 1));
    assert_eq!(r.rd32(DATA + 0x10), 0xcafe);
}

#[test]
fn faults_raise_the_event_and_clear_with_frcd() {
    let r = Rig::new(IntelIommuConf::default());
    r.w32(DMAR_FEDATA_REG, 0x41);
    r.w32(DMAR_FEADDR_REG, 0xfee0_0000);
    r.w32(DMAR_FECTL_REG, 0);
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    assert_eq!(r.dma_read(0x1000), None);
    assert_eq!(r.msis(), vec![(0xfee0_0000, 0x41)]);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) >> 32 & 0xff, u64::from(FR_READ));
    // A second fault from the same source collapses into the first.
    assert_eq!(r.dma_read(0x2000), None);
    assert!(r.msis().is_empty());
    r.w64(DMAR_FRCD_REG_0_2, FRCD_F);
    assert_eq!(r.r32(DMAR_FSTS_REG) & FSTS_PPF, 0);
}

#[test]
fn missing_context_entry_faults_unless_fpd() {
    let r = Rig::new(IntelIommuConf::default());
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    r.wq(CTX + u64::from(SID) * 16, 0);
    r.w64(DMAR_CCMD_REG, CCMD_ICC | CCMD_GLOBAL_INVL);
    assert_eq!(r.r64(DMAR_CCMD_REG) & (CCMD_ICC | CCMD_CAIG_MASK), CCMD_GLOBAL_INVL_A);
    assert_eq!(r.dma_read(0x1000), None);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) >> 32 & 0xff, u64::from(FR_CONTEXT_ENTRY_P));
    r.w64(DMAR_FRCD_REG_0_2, FRCD_F);
    // FPD hides the qualified faults.
    r.wq(CTX + u64::from(SID) * 16, PML4 | CONTEXT_ENTRY_FPD | 1);
    r.w64(DMAR_CCMD_REG, CCMD_ICC | CCMD_DEVICE_INVL | u64::from(SID) << 16);
    assert_eq!(r.dma_read(0x1000), None);
    assert_eq!(
        r.r64(DMAR_FRCD_REG_0_2),
        u64::from(FR_CONTEXT_ENTRY_P) << 32 | u64::from(SID) | FRCD_T
    );
}

#[test]
fn pass_through_needs_no_tables() {
    let r = Rig::new(IntelIommuConf::default());
    r.wq(DATA, 77);
    r.enable(CONTEXT_TT_PASS_THROUGH);
    assert_eq!(r.dma_read(DATA), Some(77));
    assert!(r.dma_write(DATA, 78));
    assert_eq!(r.rd32(DATA), 78);
}

#[test]
fn device_iotlb_context_needs_the_property() {
    let r = Rig::new(IntelIommuConf::default());
    r.enable(CONTEXT_TT_DEV_IOTLB);
    assert_eq!(r.dma_read(DATA), None);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) >> 32 & 0xff, u64::from(FR_CONTEXT_ENTRY_INV));
    let r = Rig::new(IntelIommuConf { device_iotlb: true, ..Default::default() });
    r.map(0, DATA, 3);
    r.wq(DATA, 9);
    r.enable(CONTEXT_TT_DEV_IOTLB);
    assert_eq!(r.dma_read(0), Some(9));
}

#[test]
fn iotlb_is_stale_until_invalidated() {
    let r = Rig::new(IntelIommuConf::default());
    r.wq(DATA, 1);
    r.wq(DATA + 0x1000, 2);
    r.map(0x5000, DATA, 3);
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    assert_eq!(r.dma_read(0x5000), Some(1));
    r.map(0x5000, DATA + 0x1000, 3);
    assert_eq!(r.dma_read(0x5000), Some(1));
    // A page selective invalidation of another domain leaves it.
    r.w64(DMAR_IVA_REG, 0x5000);
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_PSI_FLUSH | 6 << 32);
    assert_eq!(r.r64(DMAR_IOTLB_REG) & (TLB_IVT | TLB_FLUSH_GRANU_MASK_A), TLB_PSI_FLUSH_A);
    assert_eq!(r.dma_read(0x5000), Some(1));
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_PSI_FLUSH | 5 << 32);
    assert_eq!(r.dma_read(0x5000), Some(2));
    // IVA reads as zero, it is write only.
    assert_eq!(r.r64(DMAR_IVA_REG), 0);
}

#[test]
fn queued_invalidation_and_wait() {
    let mut r = Rig::new(IntelIommuConf::default());
    r.wq(DATA, 1);
    r.wq(DATA + 0x1000, 2);
    r.map(0x5000, DATA, 3);
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    r.enable_qi();
    assert_eq!(r.dma_read(0x5000), Some(1));
    r.map(0x5000, DATA + 0x1000, 3);
    // Domain selective IOTLB invalidation of domain 5, then a wait that writes 0xabc.
    r.submit(u64::from(INV_DESC_IOTLB) | 0x20 | 5 << 16, 0);
    r.submit(u64::from(INV_DESC_WAIT) | INV_DESC_WAIT_SW | 0xabc << 32, DATA + 0x800);
    assert_eq!(r.rd32(DATA + 0x800), 0xabc);
    assert_eq!(r.r64(DMAR_IQH_REG), 0x20);
    assert_eq!(r.dma_read(0x5000), Some(2));
    // The completion interrupt.
    r.w32(DMAR_IEDATA_REG, 0x42);
    r.w32(DMAR_IEADDR_REG, 0xfee0_1000);
    r.w32(DMAR_IECTL_REG, 0);
    r.submit(u64::from(INV_DESC_WAIT) | INV_DESC_WAIT_IF, 0);
    assert_eq!(r.msis(), vec![(0xfee0_1000, 0x42)]);
    assert_eq!(r.r32(DMAR_ICS_REG), ICS_IWC);
    r.w32(DMAR_ICS_REG, ICS_IWC);
    assert_eq!(r.r32(DMAR_ICS_REG), 0);
    // QIE turns off after a wait descriptor.
    r.w32(DMAR_GCMD_REG, GCMD_TE);
    assert_eq!(r.r32(DMAR_GSTS_REG) & GSTS_QIES, 0);
    assert_eq!(r.r64(DMAR_IQH_REG), 0);
}

#[test]
fn bad_descriptors_set_iqe() {
    let mut r = Rig::new(IntelIommuConf::default());
    r.enable_qi();
    r.submit(u64::from(INV_DESC_PIOTLB) | 0x20, 0);
    assert_ne!(r.r32(DMAR_FSTS_REG) & FSTS_IQE, 0);
    assert_eq!(r.r64(DMAR_IQH_REG), 0);
    assert_ne!(r.r32(DMAR_FECTL_REG) & FECTL_IP, 0);
    // The register based invalidations are ignored while the queue is on.
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_GLOBAL_FLUSH);
    assert_ne!(r.r64(DMAR_IOTLB_REG) & TLB_IVT, 0);
}

#[test]
fn reset_clears_the_registers() {
    let r = Rig::new(IntelIommuConf::default());
    r.enable(CONTEXT_TT_PASS_THROUGH);
    r.iommu.system_reset();
    assert_eq!(r.r32(DMAR_GSTS_REG), 0);
    assert_eq!(r.r64(DMAR_RTADDR_REG), 0);
    assert_eq!(r.r32(DMAR_RTADDR_REG_HI), 0);
}

const IRT: u64 = 0x50000;

impl Rig {
    /// A rig with remapping on and a 256 entry table at [`IRT`].
    fn with_ir(eime: bool) -> Rig {
        let r = Rig::new(IntelIommuConf::default().realize(false, false).unwrap());
        r.w64(DMAR_IRTA_REG, IRT | u64::from(eime) << 11 | 7);
        r.w32(DMAR_GCMD_REG, GCMD_SIRTP);
        assert_ne!(r.r32(DMAR_GSTS_REG) & GSTS_IRTPS, 0);
        r.w32(DMAR_GCMD_REG, GCMD_IRE);
        assert_ne!(r.r32(DMAR_GSTS_REG) & GSTS_IRES, 0);
        r.w32(DMAR_FEDATA_REG, 0x41);
        r.w32(DMAR_FEADDR_REG, 0xfee0_0000);
        r.w32(DMAR_FECTL_REG, 0);
        r
    }

    fn irte(&self, index: u64, lo: u64, hi: u64) {
        self.wq(IRT + index * 16, lo);
        self.wq(IRT + index * 16 + 8, hi);
    }

    fn fault_reason(&self) -> (u64, u64) {
        let lo = self.r64(DMAR_FRCD_REG_0_0);
        let hi = self.r64(DMAR_FRCD_REG_0_2);
        self.w64(DMAR_FRCD_REG_0_2, FRCD_F);
        (hi >> 32 & 0xff, lo >> 48)
    }
}

/// The remappable format address for `index`.
fn ir_addr(index: u32, shv: bool) -> u64 {
    u64::from(0xfee0_0000 | (index & 0x7fff) << 5 | 1 << 4 | u32::from(shv) << 3)
        | u64::from(index >> 15 & 1) << 2
}

#[test]
fn remapping_off_passes_messages_through() {
    let r = Rig::new(IntelIommuConf::default().realize(false, false).unwrap());
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(3, false), 0x31), Some((ir_addr(3, false), 0x31)));
}

#[test]
fn remaps_through_the_table() {
    let r = Rig::with_ir(false);
    // Vector 0x45, fixed, edge, physical, destination 2 in the xAPIC byte.
    r.irte(3, 0x0000_0200_0045_0001, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(3, false), 0), Some((0xfee0_2000, 0x4045)));
    // Compatibility format goes through as it is.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 0x31), Some((0xfee0_1000, 0x31)));
    // Level triggered, lowest priority, logical, redirection hint, low address bits kept.
    r.irte(4, 0x0000_0300_0051_013d, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(4, false) | 1, 0), Some((0xfee0_300d, 0xc151)));
    // With the subhandle the data adds to the index.
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(1, true), 2), Some((0xfee0_2000, 0x4045)));
    assert!(r.msis().is_empty());
}

#[test]
fn extended_interrupt_mode_takes_the_whole_destination() {
    let r = Rig::with_ir(true);
    r.irte(0, 0x0001_2345_0050_0001, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(0, false), 0), Some((0x0001_2300_fee4_5000, 0x4050)));
}

#[test]
fn bad_messages_and_entries_fault() {
    let r = Rig::with_ir(false);
    assert_eq!(r.iommu.remap_msi(SID, 0x1_fee0_0010, 0), None);
    assert_eq!(r.msis(), vec![(0xfee0_0000, 0x41)]);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_REQ_RSVD), 0));
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(300, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_INDEX_OVER), 300));
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(5, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_ENTRY_P), 5));
    r.irte(5, 0x45_1001, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(5, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_IRTE_RSVD), 5));
    r.irte(5, 0x45_0001, 1 << 20);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(5, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_IRTE_RSVD), 5));
    r.irte(5, 0x45_0001, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(5, true), 0x1_0000), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_REQ_RSVD), 0));
    // Fault processing disable hides the entry faults but still drops the message.
    r.irte(6, 0x45_0002, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(6, false), 0), None);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) & FRCD_F, 0);
}

#[test]
fn source_id_checks() {
    let r = Rig::with_ir(false);
    let sid = |svt: u64, sq: u64, source: u64| svt << 18 | sq << 16 | source;
    // Exact match.
    r.irte(1, 0x45_0001, sid(1, 0, 0x08));
    assert!(r.iommu.remap_msi(0x08, ir_addr(1, false), 0).is_some());
    assert_eq!(r.iommu.remap_msi(0x09, ir_addr(1, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_SID_ERR), 1));
    // The qualifier ignores the low function bits.
    r.irte(1, 0x45_0001, sid(1, 3, 0x08));
    assert!(r.iommu.remap_msi(0x0f, ir_addr(1, false), 0).is_some());
    // A bus range.
    r.irte(1, 0x45_0001, sid(2, 0, 0x0302));
    assert!(r.iommu.remap_msi(0x0208, ir_addr(1, false), 0).is_some());
    assert_eq!(r.iommu.remap_msi(0x0408, ir_addr(1, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_SID_ERR), 1));
    // The IOAPIC has no source ID to check.
    assert!(r.iommu.remap_msi(SID_INVALID, ir_addr(1, false), 0).is_some());
    r.irte(1, 0x45_0001, sid(3, 0, 0));
    assert_eq!(r.iommu.remap_msi(0x08, ir_addr(1, false), 0), None);
    assert_eq!(r.fault_reason(), (u64::from(FR_IR_SID_ERR), 1));
}

#[test]
fn turning_remapping_off_passes_through_again() {
    let r = Rig::with_ir(false);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(9, false), 0), None);
    r.w32(DMAR_GCMD_REG, 0);
    assert_eq!(r.r32(DMAR_GSTS_REG) & GSTS_IRES, 0);
    assert_eq!(r.iommu.remap_msi(SID, ir_addr(9, false), 0), Some((ir_addr(9, false), 0)));
}

const PDIR: u64 = 0x12000;
const PTBL: u64 = 0x13000;

impl Rig {
    /// A scalable mode rig with the first stage tables when `fsts`, and fault events unmasked.
    fn scalable(fsts: bool) -> Rig {
        let conf = IntelIommuConf { scalable_mode: true, fsts, ..Default::default() };
        let r = Rig::new(conf.realize(true, true).unwrap());
        r.w32(DMAR_FEDATA_REG, 0x42);
        r.w32(DMAR_FEADDR_REG, 0xfee0_0000);
        r.w32(DMAR_FECTL_REG, 0);
        r
    }

    /// The scalable root and context entries for [`SID`], the PASID directory entry and the
    /// PASID 0 entry with `pe0` and the domain `did`, then SRTP with SMT and TE.
    fn enable_sm(&self, pe0: u64, did: u64) {
        self.wq(ROOT, CTX | 1);
        self.wq(CTX + u64::from(SID) * 32, PDIR | 1);
        self.wq(PDIR, PTBL | 1);
        self.wq(PTBL, pe0);
        self.wq(PTBL + 8, did);
        self.wq(PTBL + 16, PML4);
        self.w64(DMAR_RTADDR_REG, ROOT | RTADDR_SMT);
        self.w32(DMAR_GCMD_REG, GCMD_SRTP);
        self.w32(DMAR_GCMD_REG, GCMD_TE);
        assert_ne!(self.r32(DMAR_GSTS_REG) & GSTS_TES, 0);
    }

    /// Drops the cached context entries and IOTLB, as Linux does after changing a table.
    fn flush(&self) {
        self.w64(DMAR_CCMD_REG, CCMD_ICC | CCMD_GLOBAL_INVL);
        self.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_GLOBAL_FLUSH);
    }

    /// Turns on a queue of 256-bit descriptors.
    fn enable_qi256(&mut self) {
        self.w64(DMAR_IQA_REG, IQ | IQA_DW_MASK);
        self.w64(DMAR_IQT_REG, 0);
        self.w32(DMAR_GCMD_REG, GCMD_TE | GCMD_QIE);
        assert_ne!(self.r32(DMAR_GSTS_REG) & GSTS_QIES, 0);
    }

    fn submit256(&mut self, lo: u64, hi: u64) {
        self.wq(IQ + self.tail * 32, lo);
        self.wq(IQ + self.tail * 32 + 8, hi);
        self.wq(IQ + self.tail * 32 + 16, 0);
        self.wq(IQ + self.tail * 32 + 24, 0);
        self.tail += 1;
        self.w64(DMAR_IQT_REG, self.tail << 5);
        assert_eq!(self.r32(DMAR_FSTS_REG) & FSTS_IQE, 0);
    }

    fn rq(&self, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(self.space.read(addr, U, &mut b).is_ok());
        u64::from_le_bytes(b)
    }
}

/// A present PASID entry of type `pgtt` over a 4-level second stage table.
fn pe0(pgtt: u64) -> u64 {
    PML4 | pgtt << 6 | 2 << 2 | 1
}

#[test]
fn scalable_capabilities() {
    let r = Rig::scalable(false);
    let ecap = r.r64(DMAR_ECAP_REG);
    assert_eq!(ecap & (ECAP_SMTS | ECAP_SRS | ECAP_SSTS), ECAP_SMTS | ECAP_SRS | ECAP_SSTS);
    assert_eq!(ecap & ECAP_FSTS, 0);
    let ecap = Rig::scalable(true).r64(DMAR_ECAP_REG);
    assert_eq!(ecap & (ECAP_SMTS | ECAP_FSTS | ECAP_SSTS), ECAP_SMTS | ECAP_FSTS);
}

#[test]
fn scalable_second_stage_walk() {
    let r = Rig::scalable(false);
    r.map(0x5000, DATA, 3);
    r.wq(DATA, 0xbeef);
    r.enable_sm(pe0(2), 7);
    assert_eq!(r.dma_read(0x5000), Some(0xbeef));
    assert!(r.dma_write(0x5004, 1));
    assert_eq!(r.rd32(DATA + 4), 1);
    // A miss in the table faults with the reason of the second stage walk.
    assert_eq!(r.dma_read(0x6000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_READ));
    assert_eq!(r.msis(), vec![(0xfee0_0000, 0x42)]);
    // The FPD bit of the PASID entry hides it.
    r.wq(PTBL, pe0(2) | PASID_ENTRY_FPD);
    r.flush();
    assert_eq!(r.dma_read(0x6000), None);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) & FRCD_F, 0);
    // So does the one of the directory entry.
    r.wq(PTBL, pe0(2));
    r.wq(PDIR, PTBL | PASID_DIR_FPD | 1);
    r.flush();
    assert_eq!(r.dma_read(0x6000), None);
    assert_eq!(r.r64(DMAR_FRCD_REG_0_2) & FRCD_F, 0);
}

#[test]
fn scalable_pass_through_and_pasid_faults() {
    let r = Rig::scalable(false);
    r.wq(DATA, 0x1234);
    r.enable_sm(pe0(4), 7);
    assert_eq!(r.dma_read(DATA), Some(0x1234));
    // Each broken PASID structure gives its own reason.
    let cases = [
        (PDIR, PTBL, u64::from(FR_PASID_DIR_ENTRY_P)),
        (PTBL, pe0(2) & !1, u64::from(FR_PASID_ENTRY_P)),
        (PTBL, 0, u64::from(FR_PASID_TABLE_ENTRY_INV)),
        (PTBL, pe0(1), u64::from(FR_PASID_TABLE_ENTRY_INV)),
        (PTBL, pe0(3), u64::from(FR_PASID_TABLE_ENTRY_INV)),
        (PTBL, PML4 | 2 << 6 | 7 << 2 | 1, u64::from(FR_PASID_TABLE_ENTRY_INV)),
    ];
    for (addr, val, fr) in cases {
        r.wq(addr, val);
        r.flush();
        assert_eq!(r.dma_read(DATA), None, "{fr:#x}");
        assert_eq!(r.fault_reason().0, fr);
        r.wq(PDIR, PTBL | 1);
        r.wq(PTBL, pe0(4));
    }
    r.flush();
    assert_eq!(r.dma_read(DATA), Some(0x1234));
}

/// Maps the 4K page at `iova` to `gpa` in the first stage table with the bits `bits`, and
/// the upper levels present, writable and user.
fn map_fs(r: &Rig, iova: u64, gpa: u64, bits: u64) {
    r.wq(PML4 + ((iova >> 39) & 0x1ff) * 8, PDPT | 7);
    r.wq(PDPT + ((iova >> 30) & 0x1ff) * 8, PD | 7);
    r.wq(PD + ((iova >> 21) & 0x1ff) * 8, PT | 7);
    r.wq(PT + ((iova >> 12) & 0x1ff) * 8, gpa | bits);
}

#[test]
fn first_stage_walk_sets_accessed_and_dirty() {
    let r = Rig::scalable(true);
    map_fs(&r, 0x5000, DATA, 7);
    map_fs(&r, 0x6000, DATA + 0x1000, 7);
    r.wq(DATA, 0xbeef);
    r.enable_sm(pe0(1), 7);
    assert_eq!(r.dma_read(0x5000), Some(0xbeef));
    assert_eq!(r.rq(PML4) & (FS_A | FS_D), FS_A);
    assert_eq!(r.rq(PT + 5 * 8) & (FS_A | FS_D), FS_A);
    assert!(r.dma_write(0x6000, 1));
    assert_eq!(r.rq(PT + 6 * 8) & (FS_A | FS_D), FS_A | FS_D);
    assert_eq!(r.rd32(DATA + 0x1000), 1);
    // A 2M page at the PD level.
    r.wq(PD + 8, 0x20_0000 | PT_PAGE_SIZE_MASK | 7);
    r.wq(0x20_0000 + 0x1230, 0x77);
    assert_eq!(r.dma_read(0x20_1230), Some(0x77));
}

#[test]
fn first_stage_faults() {
    let r = Rig::scalable(true);
    map_fs(&r, 0x5000, DATA, 1 | 4);
    map_fs(&r, 0x6000, DATA, 1 | 2);
    map_fs(&r, 0x8000, DATA | 1 << 50, 7);
    r.enable_sm(pe0(1), 7);
    assert!(!r.dma_write(0x5000, 1));
    assert_eq!(r.fault_reason().0, u64::from(FR_SM_WRITE));
    assert_eq!(r.dma_read(0x6000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_FS_PAGING_ENTRY_US));
    assert_eq!(r.dma_read(0x7000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_FS_PAGING_ENTRY_P));
    assert_eq!(r.dma_read(0x8000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_FS_PAGING_ENTRY_RSVD));
    assert_eq!(r.dma_read(0x8000_0000_0000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_FS_NON_CANONICAL));
    // A first stage pointer outside of RAM.
    r.wq(PTBL + 16, 0x1_0000_0000);
    r.flush();
    assert_eq!(r.dma_read(0x5000), None);
    assert_eq!(r.fault_reason().0, u64::from(FR_PASID_ENTRY_FSPTPTR_INV));
}

#[test]
fn pasid_based_iotlb_invalidation() {
    let mut r = Rig::scalable(true);
    map_fs(&r, 0x5000, DATA, 7);
    r.wq(DATA, 1);
    r.wq(DATA + 0x1000, 2);
    r.enable_sm(pe0(1), 7);
    r.enable_qi256();
    assert_eq!(r.dma_read(0x5000), Some(1));
    map_fs(&r, 0x5000, DATA + 0x1000, 7);
    assert_eq!(r.dma_read(0x5000), Some(1));
    // Another domain or PASID leaves the entry.
    r.submit256(u64::from(INV_DESC_PIOTLB) | 0x30 | 8 << 16, 0x5000);
    r.submit256(u64::from(INV_DESC_PIOTLB) | 0x30 | 7 << 16 | 1 << 32, 0x5000);
    r.submit256(u64::from(INV_DESC_PIOTLB) | 0x20 | 7 << 16 | 1 << 32, 0);
    assert_eq!(r.dma_read(0x5000), Some(1));
    r.submit256(u64::from(INV_DESC_PIOTLB) | 0x30 | 7 << 16, 0x5000);
    assert_eq!(r.dma_read(0x5000), Some(2));
    map_fs(&r, 0x5000, DATA, 7);
    r.submit256(u64::from(INV_DESC_PIOTLB) | 0x20 | 7 << 16, 0);
    assert_eq!(r.dma_read(0x5000), Some(1));
    // A page selective IOTLB invalidation of the domain drops its first stage entries
    // whatever the address.
    map_fs(&r, 0x5000, DATA + 0x1000, 7);
    r.submit256(u64::from(INV_DESC_IOTLB) | 0x30 | 7 << 16, 0x9000);
    assert_eq!(r.dma_read(0x5000), Some(2));
}

#[test]
fn page_request_registers() {
    let conf = IntelIommuConf {
        scalable_mode: true,
        fsts: true,
        device_iotlb: true,
        pasid_bits: 8,
        svm: true,
        ..Default::default()
    };
    let r = Rig::new(conf.realize(true, true).unwrap());
    assert_ne!(r.r64(DMAR_ECAP_REG) & ECAP_PRS, 0);
    r.w64(DMAR_PQA_REG, u64::MAX);
    assert_eq!(r.r64(DMAR_PQA_REG), 0xffff_ffff_ffff_f007);
    r.w64(DMAR_PQT_REG, u64::MAX);
    assert_eq!(r.r64(DMAR_PQT_REG), 0x7ffe0);
    r.w32(DMAR_PEDATA_REG, 0x51);
    r.w32(DMAR_PEADDR_REG, 0xfee0_0003);
    r.w32(DMAR_PEUADDR_REG, 0x1);
    assert_eq!(r.r32(DMAR_PEADDR_REG), 0xfee0_0000);
    assert_eq!(r.r32(DMAR_PEUADDR_REG), 1);
    r.w32(DMAR_PECTL_REG, 0);
    assert_eq!(r.r32(DMAR_PECTL_REG), 0);
    assert!(r.msis().is_empty());
    // Without svm the registers are not there.
    let r = Rig::scalable(true);
    r.w64(DMAR_PQA_REG, u64::MAX);
    assert_eq!(r.r64(DMAR_PQA_REG), 0);
}

/// The events a notifier saw, as `(iova, translated_addr, addr_mask, perm)`.
type Seen = Arc<StdMutex<Vec<(u64, u64, u64, u8)>>>;

/// A notifier on the device of the rig that keeps every event in [`Seen`].
fn recorder(r: &Rig) -> (Arc<dyn IommuOps>, Arc<IommuNotifier>, Seen) {
    let ops = r.iommu.device_ops(SID);
    let seen = Arc::new(StdMutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    let n = IommuNotifier::new(
        move |e: &IommuTlbEntry| {
            let perm = u8::from(e.perm.allows(IommuAccessFlags::RO))
                | u8::from(e.perm.allows(IommuAccessFlags::WO)) << 1;
            sink.lock().unwrap().push((e.iova, e.translated_addr, e.addr_mask, perm));
        },
        IommuNotifierFlag::IOTLB_EVENTS,
        0,
        u64::MAX,
        0,
    );
    (ops, n, seen)
}

#[test]
fn map_notifiers_need_caching_mode() {
    let r = Rig::new(IntelIommuConf::default());
    let (ops, n, _) = recorder(&r);
    assert_eq!(
        register_iommu_notifier(ops.as_ref(), &n),
        Err("device 00.01.0 requires caching mode: Operation not supported".to_string())
    );
    assert_eq!(
        r.iommu.set_iommu_device(SID),
        Err("Device assignment is not allowed without enabling caching-mode=on for Intel IOMMU."
            .to_string())
    );
    let r = Rig::new(IntelIommuConf { caching_mode: true, ..IntelIommuConf::default() });
    assert_eq!(r.iommu.set_iommu_device(SID), Ok(()));
    assert_eq!(r.iommu.set_iommu_device(SID), Err("Host IOMMU device already exist".to_string()));
    r.iommu.unset_iommu_device(SID);
    assert_eq!(r.iommu.set_iommu_device(SID), Ok(()));
}

#[test]
fn notifiers_follow_replay_and_invalidations() {
    let r = Rig::new(IntelIommuConf { caching_mode: true, ..IntelIommuConf::default() });
    r.map(0x5000, DATA, 3);
    r.enable(CONTEXT_TT_MULTI_LEVEL);
    let (ops, n, seen) = recorder(&r);
    register_iommu_notifier(ops.as_ref(), &n).unwrap();
    ops.replay(&n);
    assert_eq!(std::mem::take(&mut *seen.lock().unwrap()), vec![(0x5000, DATA, 0xfff, 3)]);
    // A new read only page shows up with the domain invalidation, the old one is not sent
    // again.
    r.map(0x6000, DATA + 0x1000, 1);
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_DSI_FLUSH | 5 << 32);
    assert_eq!(std::mem::take(&mut *seen.lock().unwrap()), vec![(0x6000, DATA + 0x1000, 0xfff, 1)]);
    // A page taken out of the table is unmapped.
    r.wq(PT + 5 * 8, 0);
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_DSI_FLUSH | 5 << 32);
    assert_eq!(std::mem::take(&mut *seen.lock().unwrap()), vec![(0x5000, 0, 0xfff, 0)]);
    // Another domain's invalidation sends nothing.
    r.wq(PT + 6 * 8, 0);
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_DSI_FLUSH | 6 << 32);
    assert!(seen.lock().unwrap().is_empty());
    unregister_iommu_notifier(ops.as_ref(), &n);
    r.w64(DMAR_IOTLB_REG, TLB_IVT | TLB_DSI_FLUSH | 5 << 32);
    assert!(seen.lock().unwrap().is_empty());
}
