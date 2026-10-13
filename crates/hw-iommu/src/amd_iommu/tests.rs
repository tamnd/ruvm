// SPDX-License-Identifier: GPL-2.0-or-later

//! The AMD IOMMU model driven through its registers and command buffer, with DMA going through
//! an IOMMU region as a PCI device behind it would.

use std::sync::Mutex as StdMutex;

use ruvm_mem::{Endian, MemorySystem};

use super::*;

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const DEVTAB: u64 = 0x10000;
const CMDBUF: u64 = 0x20000;
const EVTLOG: u64 = 0x30000;
const L3: u64 = 0x40000;
const L2: u64 = 0x41000;
const L1: u64 = 0x42000;
const DATA: u64 = 0x50000;
const IRT: u64 = 0x60000;
const CWADDR: u64 = 0x70000;
const IOVA: u64 = 0x1000_0000;
/// 00:01.0.
const SID: u16 = 0x08;
const DOMID: u64 = 7;
/// The message the fake `AMDVI-PCI` function has programmed.
const PCI_MSI: (u64, u32) = (0xfee0_0000, 0x41);
const RW: u64 = DEV_PERM_READ | DEV_PERM_WRITE;
const CONTROL_ON: u64 =
    CONTROL_AMDVIEN | CONTROL_EVENTLOGEN | CONTROL_EVENTINTEN | CONTROL_CMDBUFLEN;

#[derive(Default)]
struct FakePci {
    aborts: StdMutex<u32>,
}

impl AmdviPci for FakePci {
    fn msi_message(&self) -> Option<(u64, u32)> {
        Some(PCI_MSI)
    }

    fn signal_target_abort(&self) {
        *self.aborts.lock().unwrap() += 1;
    }
}

struct Rig {
    _sys: MemorySystem,
    space: Arc<AddressSpace>,
    iommu: Arc<AmdIommu>,
    dma: Arc<AddressSpace>,
    pci: Arc<FakePci>,
    msis: Arc<StdMutex<Vec<(u64, u32)>>>,
    tail: u64,
}

impl Rig {
    fn new(conf: AmdIommuConf) -> Rig {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", 0x100_0000).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let iommu = AmdIommu::new(conf.realize(false).unwrap(), &space);
        let msis = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&msis);
        iommu.set_msi_sender(move |a, d| sink.lock().unwrap().push((a, d)));
        let pci = Arc::new(FakePci::default());
        iommu.set_pci(pci.clone());
        let mmio = sys.new_io("amd-iommu", u128::from(AMDVI_MMIO_SIZE), iommu.mmio_ops()).unwrap();
        sys.add_subregion(root, AMDVI_BASE_ADDR, mmio).unwrap();
        let region = sys.new_iommu("amdvi", 1 << 64, iommu.device_ops(SID)).unwrap();
        let dma = sys.address_space_init(region, "dma").unwrap();
        Rig { _sys: sys, space, iommu, dma, pci, msis, tail: 0 }
    }

    fn w(&self, off: u64, size: u32, v: u64) {
        assert!(self.space.store(AMDVI_BASE_ADDR + off, size, v, Endian::Little, U).is_ok());
    }

    fn w64(&self, off: u64, v: u64) {
        self.w(off, 8, v);
    }

    fn r(&self, off: u64, size: u32) -> u64 {
        let (v, r) = self.space.load(AMDVI_BASE_ADDR + off, size, Endian::Little, U);
        assert!(r.is_ok());
        v
    }

    fn r64(&self, off: u64) -> u64 {
        self.r(off, 8)
    }

    fn wq(&self, addr: u64, v: u64) {
        assert!(self.space.write(addr, U, &v.to_le_bytes()).is_ok());
    }

    fn wd(&self, addr: u64, v: u32) {
        assert!(self.space.write(addr, U, &v.to_le_bytes()).is_ok());
    }

    fn rq(&self, addr: u64) -> u64 {
        let mut b = [0u8; 8];
        assert!(self.space.read(addr, U, &mut b).is_ok());
        u64::from_le_bytes(b)
    }

    fn dma_read(&self, addr: u64) -> Option<u32> {
        let mut b = [0u8; 4];
        self.dma.read(addr, U, &mut b).is_ok().then(|| u32::from_le_bytes(b))
    }

    fn dma_write(&self, addr: u64, v: u32) -> bool {
        self.dma.write(addr, U, &v.to_le_bytes()).is_ok()
    }

    /// The device table, a 256 entry command buffer and an event log of `2^evt` entries, the
    /// order Linux writes them in, then `control`.
    fn setup(&self, evt: u64, control: u64) {
        self.w64(MMIO_DEVICE_TABLE, DEVTAB | 1);
        self.w64(MMIO_COMMAND_BASE, CMDBUF | 8 << 56);
        self.w64(MMIO_EVENT_BASE, EVTLOG | evt << 56);
        self.w64(MMIO_CONTROL, control);
    }

    /// A device table entry for `devid` with three levels of page table at [`L3`].
    fn dte(&self, devid: u16, q0: u64, q2: u64) {
        let at = DEVTAB + u64::from(devid) * 32;
        self.wq(at, q0);
        self.wq(at + 8, DOMID);
        self.wq(at + 16, q2);
        self.wq(at + 24, 0);
    }

    /// Maps the 4K page at [`IOVA`] to `gpa` with the permissions `perm`.
    fn map(&self, gpa: u64, perm: u64) {
        self.wq(L3 + ((IOVA >> 30) & 0x1ff) * 8, L2 | 2 << 9 | PTE_PR | RW);
        self.wq(L2 + ((IOVA >> 21) & 0x1ff) * 8, L1 | 1 << 9 | PTE_PR | RW);
        self.wq(L1 + ((IOVA >> 12) & 0x1ff) * 8, gpa | PTE_PR | perm);
    }

    fn submit(&mut self, cmd0: u64, cmd1: u64) {
        self.wq(CMDBUF + self.tail * 16, cmd0);
        self.wq(CMDBUF + self.tail * 16 + 8, cmd1);
        self.tail += 1;
        self.w64(MMIO_COMMAND_TAIL, self.tail << 4);
    }

    fn inval_devtab(&mut self, devid: u16) {
        self.submit(u64::from(devid) | CMD_INVAL_DEVTAB_ENTRY << 60, 0);
    }

    /// A device with a page table and `dma-remap=on`, switched to translation.
    fn translating() -> Rig {
        let mut r = Rig::new(AmdIommuConf { dma_remap: true, ..Default::default() });
        r.setup(8, CONTROL_ON);
        r.dte(SID, L3 | 3 << 9 | DEV_VALID | DEV_TRANSLATION_VALID | RW, 0);
        r.inval_devtab(SID);
        r
    }

    fn event(&self, n: u64) -> [u64; 2] {
        [self.rq(EVTLOG + n * 16), self.rq(EVTLOG + n * 16 + 8)]
    }

    fn msis(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.msis.lock().unwrap())
    }

    fn aborts(&self) -> u32 {
        *self.pci.aborts.lock().unwrap()
    }
}

#[test]
fn realize_resolves_and_rejects_like_qemu() {
    let c = AmdIommuConf::default().realize(false).unwrap();
    assert_eq!(c.intremap, OnOffAuto::On);
    assert!(c.intr_supported());
    let c = AmdIommuConf::default().realize(true).unwrap();
    assert_eq!(c.intremap, OnOffAuto::Off);
    let on = AmdIommuConf { intremap: OnOffAuto::On, ..Default::default() };
    assert_eq!(
        on.realize(true).unwrap_err(),
        "Interrupt Remapping cannot work with kernel-irqchip=on, please use 'split|off'."
    );
    assert_eq!(AmdIommuConf::default().extended_features(), 0x29d3);
    let xt = AmdIommuConf { xtsup: true, ..Default::default() };
    assert_eq!(xt.extended_features(), 0x29d7);
    let nodma = AmdIommuConf { dma_translation: false, ..Default::default() };
    assert_eq!(nodma.extended_features(), 0x2dd3);
}

#[test]
fn registers_follow_the_qemu_masks() {
    let r = Rig::new(AmdIommuConf::default());
    assert_eq!(r.r64(MMIO_EXT_FEATURES), 0x29d3);
    assert_eq!(r.r64(MMIO_STATUS), 0);
    // QEMU has no write handler for the extended features.
    r.w64(MMIO_EXT_FEATURES, 0);
    assert_eq!(r.r64(MMIO_EXT_FEATURES), 0x29d3);
    // Byte reads give all ones and byte writes do nothing, word reads work.
    assert_eq!(r.r(MMIO_EXT_FEATURES, 1), 0xff);
    r.w(MMIO_EXT_FEATURES, 1, 0);
    assert_eq!(r.r(MMIO_EXT_FEATURES, 2), 0x29d3);
    // The run bits follow the enables, and only the low bits of status clear on write.
    r.setup(8, CONTROL_ON);
    assert_eq!(r.r64(MMIO_STATUS), STATUS_CMDBUF_RUN | STATUS_EVT_RUN);
    r.w64(MMIO_STATUS, 0xff);
    assert_eq!(r.r64(MMIO_STATUS), STATUS_CMDBUF_RUN | STATUS_EVT_RUN);
    r.w64(MMIO_CONTROL, 0);
    assert_eq!(r.r64(MMIO_STATUS), 0);
    r.w64(MMIO_COMMAND_HEAD, 0x12_3456);
    assert_eq!(r.r64(MMIO_COMMAND_HEAD), 0x12_3456);
    r.iommu.system_reset();
    assert_eq!(r.r64(MMIO_COMMAND_HEAD), 0);
    assert_eq!(r.r64(MMIO_EXT_FEATURES), 0x29d3);
}

#[test]
fn completion_wait_stores_and_interrupts() {
    let mut r = Rig::new(AmdIommuConf::default());
    r.setup(8, CONTROL_ON);
    r.submit(CWADDR | 1 | CMD_COMPLETION_WAIT << 60, 0x1122_3344_5566_7788);
    assert_eq!(r.rq(CWADDR), 0x1122_3344_5566_7788);
    assert_eq!(r.r64(MMIO_COMMAND_HEAD), 16);
    assert!(r.msis().is_empty());
    r.submit(CWADDR | 2 | CMD_COMPLETION_WAIT << 60, 0);
    assert_eq!(r.msis(), vec![PCI_MSI]);
    assert_ne!(r.r64(MMIO_STATUS) & STATUS_COMP_INT, 0);
    r.w64(MMIO_STATUS, STATUS_COMP_INT);
    assert_eq!(r.r64(MMIO_STATUS) & STATUS_COMP_INT, 0);
}

#[test]
fn bad_commands_are_logged() {
    let mut r = Rig::new(AmdIommuConf::default());
    r.setup(8, CONTROL_ON);
    r.submit(0xf << 60, 0);
    assert_eq!(r.event(0), [(0xf | u64::from(EVENT_ILLEGAL_COMMAND_ERROR)) << 48, CMDBUF]);
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 16);
    // A reserved bit in an invalidate device table entry command.
    r.submit(1 << 20 | CMD_INVAL_DEVTAB_ENTRY << 60, 0);
    assert_eq!(r.event(1), [(2 | u64::from(EVENT_ILLEGAL_COMMAND_ERROR)) << 48, CMDBUF + 16]);
    assert_eq!(r.msis(), vec![PCI_MSI, PCI_MSI]);
    assert_ne!(r.r64(MMIO_STATUS) & STATUS_EVENT_INT, 0);
    assert_eq!(r.aborts(), 0);
}

#[test]
fn devices_pass_through_without_dma_remap() {
    let mut r = Rig::new(AmdIommuConf::default());
    r.setup(8, CONTROL_ON);
    r.dte(SID, L3 | 3 << 9 | DEV_VALID | DEV_TRANSLATION_VALID | RW, 0);
    r.inval_devtab(SID);
    r.wd(DATA, 0xdead_beef);
    assert_eq!(r.dma_read(DATA), Some(0xdead_beef));
    assert_eq!(r.iommu.translate(SID, IOVA, IommuAccessFlags::RO).translated_addr, IOVA);
}

#[test]
fn dma_remap_translates_through_the_page_table() {
    let r = Rig::translating();
    r.map(DATA, DEV_PERM_READ);
    r.wd(DATA + 0x10, 0x1234_5678);
    // The page is read only.
    assert!(!r.dma_write(IOVA, 1));
    let info = u64::from(2 | EVENT_IOPF_I | EVENT_IOPF);
    assert_eq!(r.event(0), [u64::from(SID) | info << 48, IOVA]);
    assert_eq!(r.aborts(), 1);
    assert_eq!(r.msis(), vec![PCI_MSI]);
    assert_eq!(r.dma_read(IOVA + 0x10), Some(0x1234_5678));
    assert!(r.msis().is_empty());
    // Nothing is mapped at the next page.
    assert_eq!(r.dma_read(IOVA + 0x1000), None);
    let info = u64::from(1 | EVENT_IOPF_I | EVENT_IOPF);
    assert_eq!(r.event(1), [u64::from(SID) | info << 48, IOVA + 0x1000]);
    // Interrupt addresses are write only and untranslated.
    let e = r.iommu.translate(SID, 0xfee0_1000, IommuAccessFlags::WO);
    assert_eq!((e.translated_addr, e.perm), (0xfee0_1000, IommuAccessFlags::WO));
    // Turning the IOMMU off gives back the identity map.
    r.w64(MMIO_CONTROL, 0);
    assert_eq!(r.dma_read(DATA + 0x10), Some(0x1234_5678));
}

#[test]
fn large_pages_and_mode_zero() {
    let mut r = Rig::translating();
    // A 2 MiB leaf at level 2.
    r.wq(L3, L2 | 2 << 9 | PTE_PR | RW);
    r.wq(L2 + ((IOVA >> 21) & 0x1ff) * 8, 0x20_0000 | PTE_PR | RW);
    r.wd(0x20_0000 + 0x1_2340, 0xabcd);
    assert_eq!(r.dma_read(IOVA + 0x1_2340), Some(0xabcd));
    let e = r.iommu.translate(SID, IOVA + 0x1_2340, IommuAccessFlags::RO);
    assert_eq!((e.iova, e.translated_addr, e.addr_mask), (IOVA, 0x20_0000, 0x1f_ffff));
    r.dte(SID, DEV_VALID | DEV_TRANSLATION_VALID | DEV_PERM_READ, 0);
    r.submit(CMD_INVAL_AMDVI_ALL << 60, 0);
    r.inval_devtab(SID);
    // Mode 0 switches the device back to pass-through, as QEMU does.
    assert!(r.dma_write(DATA, 5));
    assert_eq!(r.iommu.translate(SID, DATA, IommuAccessFlags::WO).perm, IommuAccessFlags::RW);
}

#[test]
fn invalidation_drops_cached_translations() {
    let mut r = Rig::translating();
    r.map(DATA, RW);
    r.wd(DATA, 1);
    r.wd(DATA + 0x1000, 2);
    assert_eq!(r.dma_read(IOVA), Some(1));
    r.map(DATA + 0x1000, RW);
    assert_eq!(r.dma_read(IOVA), Some(1));
    // Another domain leaves it.
    r.submit(1 << 32 | CMD_INVAL_AMDVI_PAGES << 60, IOVA);
    assert_eq!(r.dma_read(IOVA), Some(1));
    r.submit(DOMID << 32 | CMD_INVAL_AMDVI_PAGES << 60, IOVA);
    assert_eq!(r.dma_read(IOVA), Some(2));
    r.map(DATA, RW);
    r.submit(u64::from(SID) | CMD_INVAL_IOTLB_PAGES << 60, IOVA);
    assert_eq!(r.dma_read(IOVA), Some(1));
    r.map(DATA + 0x1000, RW);
    r.submit(u64::from(SID) | CMD_INVAL_IOTLB_PAGES << 60, 1);
    assert_eq!(r.dma_read(IOVA), Some(2));
    r.map(DATA, RW);
    r.submit(CMD_INVAL_AMDVI_ALL << 60, 0);
    assert_eq!(r.dma_read(IOVA), Some(1));
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 0);
}

#[test]
fn device_table_entries_switch_the_address_space() {
    let mut r = Rig::translating();
    // V=0 passes DMA untranslated again.
    r.dte(SID, 0, 0);
    r.inval_devtab(SID);
    r.wd(DATA, 0x77);
    assert_eq!(r.dma_read(DATA), Some(0x77));
    // V=1 without TV isolates the device.
    r.dte(SID, DEV_VALID | RW, 0);
    r.inval_devtab(SID);
    assert_eq!(r.dma_read(DATA), None);
    // A reserved bit is an illegal entry.
    r.dte(SID, L3 | 3 << 9 | DEV_VALID | DEV_TRANSLATION_VALID | 1 << 63, 0);
    r.inval_devtab(SID);
    assert_eq!(r.dma_read(DATA), None);
    let at = DEVTAB + u64::from(SID) * 32;
    let info = u64::from(EVENT_ILLEGAL_DEVTAB_ENTRY);
    assert_eq!(r.event(0), [u64::from(SID) | info << 48, at]);
}

#[test]
fn the_event_log_overflows() {
    let mut r = Rig::new(AmdIommuConf::default());
    r.setup(1, CONTROL_ON);
    r.submit(0xf << 60, 0);
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 16);
    assert_eq!(r.r64(MMIO_STATUS) & STATUS_EVT_OVF, 0);
    r.submit(0xf << 60, 0);
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 16);
    assert_ne!(r.r64(MMIO_STATUS) & STATUS_EVT_OVF, 0);
    assert_eq!(r.msis().len(), 2);
    r.submit(0xf << 60, 0);
    assert!(r.msis().is_empty());
    // The base can not move while the event interrupt is pending.
    r.w64(MMIO_EVENT_BASE, (EVTLOG + 0x1000) | 8 << 56);
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 16);
    r.w64(MMIO_STATUS, STATUS_EVENT_INT | STATUS_EVT_OVF);
    r.w64(MMIO_EVENT_BASE, (EVTLOG + 0x1000) | 8 << 56);
    assert_eq!(r.r64(MMIO_EVENT_TAIL), 0);
}

#[test]
fn xt_interrupts_use_the_general_interrupt_register() {
    let mut r = Rig::new(AmdIommuConf { xtsup: true, ..Default::default() });
    r.w64(MMIO_XT_GEN_INTR, 0x12 << 56 | 1 << 40 | 0x45 << 32 | 0x34_5678 << 8);
    r.setup(8, CONTROL_ON | CONTROL_INTCAPXTEN);
    r.submit(2 | CMD_COMPLETION_WAIT << 60, 0);
    let dest: u64 = 0x1234_5678;
    let addr = 0xfee0_0000 | (dest & 0xff) << 12 | (dest & 0xffff_ff00) << 32;
    assert_eq!(r.msis(), vec![(addr, 0x45 | 1 << 8 | 1 << 14)]);
}

/// A device table entry with interrupt remapping on and `intctl`.
fn ir_dte(intctl: u64) -> u64 {
    IRT | IR_REMAP_ENABLE | intctl << IR_INTCTL_SHIFT
}

#[test]
fn legacy_interrupt_remapping() {
    let r = Rig::new(AmdIommuConf::default());
    // Before the device table is set up everything passes.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), Some((0xfee0_0000, 5)));
    r.setup(8, CONTROL_ON);
    // Without remapping in the entry it still passes.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), Some((0xfee0_0000, 5)));
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP));
    r.wd(IRT + 5 * 4, 1 | 1 << 5 | 3 << 8 | 0x45 << 16);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), Some((0xfee0_3008, 0x4045)));
    // Not valid, guest mode and a reserved type all fail.
    r.wd(IRT + 5 * 4, 0x45 << 16);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), None);
    r.wd(IRT + 5 * 4, 1 | 1 << 7);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), None);
    r.wd(IRT + 5 * 4, 1 | 2 << 2);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), None);
    // Outside the interrupt range.
    assert_eq!(r.iommu.remap_msi(SID, 0xfef0_0000, 5), None);
    // QEMU turns a passed message into an empty one.
    r.dte(SID, 0, ir_dte(IR_INTCTL_PASS));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 0x31), Some((0xfee0_0000, 0x4000)));
    r.dte(SID, 0, ir_dte(0));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 0x31), None);
    // The IOAPIC uses the entry of 00:14.0.
    r.dte(AMDVI_IOAPIC_SB_DEVID, 0, ir_dte(IR_INTCTL_REMAP));
    r.wd(IRT + 2 * 4, 1 | 9 << 8 | 0x50 << 16);
    assert_eq!(r.iommu.remap_msi(SID_INVALID, 0xfee0_0000, 2), Some((0xfee0_9000, 0x4050)));
}

#[test]
fn special_delivery_modes_need_their_pass_bits() {
    let r = Rig::new(AmdIommuConf::default());
    r.setup(8, CONTROL_ON);
    let nmi = 4 << 8;
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, nmi), None);
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP) | DEV_NMI_PASS_MASK);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, nmi), Some((0xfee0_1000, nmi)));
    // Logical destination mode is refused.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1004, nmi), None);
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP) | DEV_INT_PASS_MASK | DEV_EINT_PASS_MASK);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 5 << 8), Some((0xfee0_1000, 5 << 8)));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 7 << 8), Some((0xfee0_1000, 7 << 8)));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 2 << 8), None);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_1000, 3 << 8), None);
}

#[test]
fn intremap_off_refuses_remapping_entries() {
    let r = Rig::new(AmdIommuConf { intremap: OnOffAuto::Off, ..Default::default() });
    r.setup(8, CONTROL_ON);
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP));
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), None);
    r.dte(SID, 0, 0);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 5), Some((0xfee0_0000, 5)));
}

#[test]
fn ga_interrupt_remapping() {
    let r = Rig::new(AmdIommuConf { xtsup: true, ..Default::default() });
    r.setup(8, CONTROL_ON | CONTROL_GAEN);
    r.dte(SID, 0, ir_dte(IR_INTCTL_REMAP));
    r.wq(IRT + 3 * 16, 1 | 1 << 6 | 0x34_5678 << 8);
    r.wq(IRT + 3 * 16 + 8, 0x12 << 56 | 0x61);
    // Without XT only the low byte of the destination is used.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 3), Some((0xfee7_8004, 0x4061)));
    r.w64(MMIO_CONTROL, CONTROL_ON | CONTROL_GAEN | CONTROL_XTEN);
    let dest: u64 = 0x1234_5678;
    let addr = 0xfee0_0004 | (dest & 0xff) << 12 | (dest & 0xffff_ff00) << 32;
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 3), Some((addr, 0x4061)));
    r.wq(IRT + 3 * 16, 1 | 3 << 2);
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 3), None);
}

#[test]
fn invalidation_sizes_decode_like_qemu() {
    assert_eq!(decode_invalidation_size(0x1000, 0), 0x1000);
    assert_eq!(decode_invalidation_size(0x1000, 1), 0x4000);
    assert_eq!(decode_invalidation_size(0x7_f000, 1), 0x10_0000);
    assert_eq!(decode_invalidation_size(0x7_ffff_ffff_f000, 1), INV_ALL_PAGES);
}
