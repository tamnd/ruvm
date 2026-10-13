// SPDX-License-Identifier: GPL-2.0-or-later

//! The VT-d model driven through its registers and invalidation queue, with DMA going through an
//! IOMMU region as a PCI device behind it would.

use std::sync::Mutex as StdMutex;

use ruvm_mem::{Endian, MemorySystem};

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
    let c = IntelIommuConf::default().realize(false, false).unwrap();
    assert_eq!(c.intremap, OnOffAuto::Off);
    assert_eq!(c.eim, OnOffAuto::Off);
    let err = |c: IntelIommuConf, all: bool| c.realize(all, all).unwrap_err();
    let on = IntelIommuConf { intremap: OnOffAuto::On, ..Default::default() };
    assert_eq!(
        err(on.clone(), true),
        "Interrupt Remapping cannot work with kernel-irqchip=on, please use 'split|off'."
    );
    assert!(err(on, false).contains("no interrupt remapping yet"));
    let eim = IntelIommuConf { eim: OnOffAuto::On, ..Default::default() };
    assert!(err(eim.clone(), false).contains("no interrupt remapping yet"));
    // QEMU checks aw-bits first because remapping is on by default without KVM.
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
    assert!(err(sm, false).contains("no scalable mode yet"));
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
