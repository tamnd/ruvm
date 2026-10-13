// SPDX-License-Identifier: GPL-2.0-or-later

//! The RISC-V IOMMU model driven through its registers and queues, with DMA going through an
//! IOMMU region as a PCI device behind it would.

use std::sync::Mutex as StdMutex;

use ruvm_base::ClockType;
use ruvm_mem::{Endian, MemorySystem};

use super::*;

const U: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const BASE: u64 = 0x0100_0000;
const DDT: u64 = 0x10000;
const CQ: u64 = 0x20000;
const FQ: u64 = 0x30000;
const S_ROOT: u64 = 0x40000;
const S_L1: u64 = 0x41000;
const S_L0: u64 = 0x42000;
const DATA: u64 = 0x50000;
const DATA2: u64 = 0x51000;
const MSIPT: u64 = 0x60000;
const MRIF: u64 = 0x61000;
const NOTICE: u64 = 0x62000;
const TARGET: u64 = 0x63000;
const PDT: u64 = 0x70000;
const FENCE: u64 = 0x71000;
const G_ROOT: u64 = 0x80000;
const G_L1: u64 = 0x84000;
const G_L0: u64 = 0x85000;
const IOVA: u64 = 0x1000_0000;
const MSI_ADDR: u64 = 0x2800_0000;
/// 00:01.0.
const SID: u16 = 0x08;
const AVAIL: u64 = 0x3333;
const LEAF: u64 = PTE_V | PTE_R | PTE_W | PTE_U | PTE_A | PTE_D;
/// QEMU wants U on inner PTEs too when there is no process ID.
const NODE: u64 = PTE_V | PTE_U;

struct Rig {
    _sys: MemorySystem,
    space: Arc<AddressSpace>,
    iommu: Arc<RiscvIommu>,
    clock: Arc<Clock>,
    dma: Arc<AddressSpace>,
    irqs: Arc<StdMutex<Vec<u32>>>,
    cq_tail: u64,
}

impl Rig {
    fn new(conf: RiscvIommuConf) -> Rig {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", 1 << 64).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let ram = sys.new_ram("ram", 0x100_0000).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let clock = Clock::manual(ClockType::Virtual);
        let iommu = RiscvIommu::new(conf, RiscvIommuIgs::Msi, AVAIL, &space, &clock);
        let irqs = Arc::new(StdMutex::new(Vec::new()));
        let sink = Arc::clone(&irqs);
        iommu.set_notify(move |v| sink.lock().unwrap().push(v));
        let mmio =
            sys.new_io("riscv-iommu", u128::from(RISCV_IOMMU_REG_SIZE), iommu.mmio_ops()).unwrap();
        sys.add_subregion(root, BASE, mmio).unwrap();
        let region = sys.new_iommu("riscv_iommu", 1 << 64, iommu.device_ops(SID)).unwrap();
        let dma = sys.address_space_init(region, "dma").unwrap();
        Rig { _sys: sys, space, iommu, clock, dma, irqs, cq_tail: 0 }
    }

    fn w(&self, off: u64, size: u32, v: u64) {
        assert!(self.space.store(BASE + off, size, v, Endian::Little, U).is_ok());
    }

    fn w64(&self, off: u64, v: u64) {
        self.w(off, 8, v);
    }

    fn w32(&self, off: u64, v: u32) {
        self.w(off, 4, u64::from(v));
    }

    fn r(&self, off: u64, size: u32) -> u64 {
        let (v, r) = self.space.load(BASE + off, size, Endian::Little, U);
        assert!(r.is_ok());
        v
    }

    fn r64(&self, off: u64) -> u64 {
        self.r(off, 8)
    }

    fn r32(&self, off: u64) -> u32 {
        self.r(off, 4) as u32
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

    fn rd(&self, addr: u64) -> u32 {
        let mut b = [0u8; 4];
        assert!(self.space.read(addr, U, &mut b).is_ok());
        u32::from_le_bytes(b)
    }

    fn dma_read_as(&self, addr: u64, attrs: MemTxAttrs) -> Option<u32> {
        let mut b = [0u8; 4];
        self.dma.read(addr, attrs, &mut b).is_ok().then(|| u32::from_le_bytes(b))
    }

    fn dma_read(&self, addr: u64) -> Option<u32> {
        self.dma_read_as(addr, U)
    }

    fn dma_write(&self, addr: u64, v: u32) -> bool {
        self.dma.write(addr, U, &v.to_le_bytes()).is_ok()
    }

    fn irqs(&self) -> Vec<u32> {
        std::mem::take(&mut *self.irqs.lock().unwrap())
    }

    /// Vectors 0 to 3 for the four interrupt sources, and the command and fault queues on
    /// with their interrupts.
    fn queues(&self) {
        self.w32(REG_ICVEC, 0x3210);
        self.w64(REG_CQB, ppn_down(CQ) << 10 | 7);
        self.w32(REG_CQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE);
        self.w64(REG_FQB, ppn_down(FQ) << 10 | 6);
        self.w32(REG_FQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE);
    }

    /// A one level device directory with an extended device context for [`SID`].
    fn dc(&self, tc: u64, iohgatp: u64, ta: u64, fsc: u64, msiptp: u64, mask: u64, pattern: u64) {
        let at = DDT + u64::from(SID) * 64;
        for (i, v) in [tc, iohgatp, ta, fsc, msiptp, mask, pattern, 0].into_iter().enumerate() {
            self.wq(at + i as u64 * 8, v);
        }
        self.w64(REG_DDTP, ppn_down(DDT) << 10 | DDTP_MODE_1LVL);
    }

    /// Maps the 4K page at `iova` to `pa` in the Sv39 tables at [`S_ROOT`].
    fn s_map(&self, iova: u64, pa: u64, leaf: u64) {
        self.wq(S_ROOT + ((iova >> 30) & 0x1ff) * 8, ppn_down(S_L1) << 10 | NODE);
        self.wq(S_L1 + ((iova >> 21) & 0x1ff) * 8, ppn_down(S_L0) << 10 | NODE);
        self.wq(S_L0 + ((iova >> 12) & 0x1ff) * 8, ppn_down(pa) << 10 | leaf);
    }

    fn sv39(root: u64) -> u64 {
        ATP_MODE_SV39 << 60 | ppn_down(root)
    }

    /// Queues one command and moves the tail past it.
    fn cmd(&mut self, dword0: u64, dword1: u64) {
        let at = CQ + self.cq_tail * 16;
        self.wq(at, dword0);
        self.wq(at + 8, dword1);
        self.cq_tail += 1;
        self.w32(REG_CQT, self.cq_tail as u32);
    }

    /// The fault record at `idx`: the header, `iotval` and `iotval2`.
    fn fault(&self, idx: u64) -> (u64, u64, u64) {
        let at = FQ + idx * 32;
        (self.rq(at), self.rq(at + 16), self.rq(at + 24))
    }
}

fn rig() -> Rig {
    Rig::new(RiscvIommuConf::default())
}

#[test]
fn reset_state() {
    let r = rig();
    let cap = r.r64(REG_CAP);
    let want = 0x10
        | CAP_SV32
        | CAP_SV39
        | CAP_SV48
        | CAP_SV57
        | CAP_SVRSW60T59B
        | CAP_SV32X4
        | CAP_SV39X4
        | CAP_SV48X4
        | CAP_SV57X4
        | CAP_MSI_FLAT
        | CAP_MSI_MRIF
        | CAP_ATS
        | CAP_HPM
        | CAP_DBG
        | CAP_PD8;
    assert_eq!(cap, want);
    assert_eq!(r.iommu.cap(), want);
    assert_eq!(r.r64(REG_DDTP), DDTP_MODE_OFF);
    assert_eq!(r.r64(REG_FCTL), 0);

    let conf = RiscvIommuConf {
        intremap: false,
        ats: false,
        off: false,
        s_stage: false,
        g_stage: false,
        hpm_counters: 0,
        pas_bits: 56,
        version: 0x123,
        ..RiscvIommuConf::default()
    };
    let r = Rig::new(conf);
    assert_eq!(r.r64(REG_CAP), 0x23 | 56 << 32 | CAP_DBG | CAP_PD8);
    assert_eq!(r.r64(REG_DDTP), DDTP_MODE_BARE);
}

#[test]
fn igs_goes_in_the_capabilities() {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", 1 << 64).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    let clock = Clock::manual(ClockType::Virtual);
    let conf = RiscvIommuConf::default();
    let iommu = RiscvIommu::new(conf, RiscvIommuIgs::Both, 0, &space, &clock);
    assert_eq!(get_field(iommu.cap(), CAP_IGS), 2);
}

#[test]
fn register_access_rules() {
    let r = rig();
    // Misaligned and past the MSI table.
    let (_, res) = r.space.load(BASE + 0x14 + 2, 4, Endian::Little, U);
    assert!(!res.is_ok());
    let (_, res) = r.space.load(BASE + 0x300, 4, Endian::Little, U);
    assert!(!res.is_ok());
    // capabilities is read-only.
    r.w64(REG_CAP, 0);
    assert_eq!(r.r64(REG_CAP), r.iommu.cap());
    // fctl keeps only WSI and GXL.
    r.w32(REG_FCTL, 0xffff_ffff);
    assert_eq!(r.r32(REG_FCTL), (FCTL_WSI | FCTL_GXL) as u32);
    assert_eq!(r.iommu.fctl(), FCTL_WSI | FCTL_GXL);
    // icvec is clipped to the vectors the front end has.
    r.w32(REG_ICVEC, 0xffff);
    assert_eq!(r.r32(REG_ICVEC), 0x3333);
    r.w32(REG_ICVEC, 0x1230);
    assert_eq!(r.r32(REG_ICVEC), 0x1230);
    // The queue base keeps LOG2SZ and PPN.
    r.w64(REG_CQB, !0);
    assert_eq!(r.r64(REG_CQB), PPN_FIELD | QUEUE_LOGSZ);
}

#[test]
fn ddtp_mode_changes() {
    let r = rig();
    r.w64(REG_DDTP, ppn_down(DDT) << 10 | DDTP_MODE_BARE | 0xf0);
    assert_eq!(r.r64(REG_DDTP), ppn_down(DDT) << 10 | DDTP_MODE_BARE);
    r.w64(REG_DDTP, ppn_down(DDT) << 10 | DDTP_MODE_2LVL);
    assert_eq!(r.r64(REG_DDTP), ppn_down(DDT) << 10 | DDTP_MODE_2LVL);
    // 2LVL to 3LVL has to go through OFF or BARE.
    r.w64(REG_DDTP, ppn_down(DDT) << 10 | DDTP_MODE_3LVL);
    assert_eq!(r.r64(REG_DDTP), ppn_down(DDT) << 10 | DDTP_MODE_2LVL);
    // A mode that does not exist is turned down too.
    r.w64(REG_DDTP, 7);
    assert_eq!(r.r64(REG_DDTP), ppn_down(DDT) << 10 | DDTP_MODE_2LVL);
    r.w64(REG_DDTP, DDTP_MODE_OFF);
    assert_eq!(r.r64(REG_DDTP), DDTP_MODE_OFF);
}

#[test]
fn off_blocks_and_bare_passes_through() {
    let r = rig();
    r.queues();
    r.wd(DATA, 0x1234_5678);
    assert_eq!(r.dma_read(DATA), None);
    assert_eq!(r.r32(REG_FQT), 1);
    let (hdr, iotval, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_DMA_DISABLED));
    assert_eq!(get_field(hdr, FQ_HDR_TTYPE), TTYPE_UADDR_RD);
    assert_eq!(get_field(hdr, FQ_HDR_DID), u64::from(SID));
    assert_eq!(iotval, DATA);
    assert_eq!(r.irqs(), vec![1]);
    assert_ne!(r.r32(REG_IPSR) & IPSR_FIP, 0);

    r.w64(REG_DDTP, DDTP_MODE_BARE);
    assert_eq!(r.dma_read(DATA), Some(0x1234_5678));
    assert!(r.dma_write(DATA + 4, 7));
    assert_eq!(r.rd(DATA + 4), 7);
}

#[test]
fn first_stage_translation_and_faults() {
    let r = rig();
    r.queues();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    r.wd(DATA + 0x10, 0xcafe_f00d);
    assert_eq!(r.dma_read(IOVA + 0x10), Some(0xcafe_f00d));
    // The IOTLB keeps the permission of the access that filled it, as QEMU does, so a write
    // to a page that was only read fails, with no fault record, until the entry goes.
    assert!(!r.dma_write(IOVA + 0x20, 0x55));
    assert_eq!(r.r32(REG_FQT), 0);
    r.s_map(IOVA + 0x4000, DATA2, LEAF);
    assert!(r.dma_write(IOVA + 0x4020, 0x55));
    assert_eq!(r.rd(DATA2 + 0x20), 0x55);

    // Nothing mapped at the next page.
    assert_eq!(r.dma_read(IOVA + 0x1008), None);
    assert_eq!(r.r32(REG_FQT), 1);
    let (hdr, iotval, iotval2) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_RD_FAULT_S));
    assert_eq!(get_field(hdr, FQ_HDR_TTYPE), TTYPE_UADDR_RD);
    assert_eq!(get_field(hdr, FQ_HDR_DID), u64::from(SID));
    assert_eq!(hdr & FQ_HDR_PV, 0);
    assert_eq!(iotval, IOVA + 0x1008);
    assert_eq!(iotval2, IOVA + 0x1008);
    assert_eq!(r.irqs(), vec![1]);

    // A read-only page turns down writes.
    r.s_map(IOVA + 0x2000, DATA2, PTE_V | PTE_R | PTE_U | PTE_A);
    assert!(!r.dma_write(IOVA + 0x2000, 1));
    let (hdr, _, _) = r.fault(1);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_WR_FAULT_S));
    assert_eq!(get_field(hdr, FQ_HDR_TTYPE), TTYPE_UADDR_WR);

    // Supervisor pages need a process ID.
    r.s_map(IOVA + 0x3000, DATA2, LEAF & !PTE_U);
    assert_eq!(r.dma_read(IOVA + 0x3000), None);
    // Addresses that are not sign extended.
    assert_eq!(r.dma_read(1 << 40), None);
    assert_eq!(r.r32(REG_FQT), 4);
}

#[test]
fn device_context_faults() {
    let r = rig();
    r.queues();
    // Not valid.
    r.dc(0, 0, 0, 0, 0, 0, 0);
    assert_eq!(r.dma_read(IOVA), None);
    let (hdr, _, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_DDT_INVALID));
    // Reserved tc bits.
    r.dc(DC_TC_V | 1 << 12, 0, 0, 0, 0, 0, 0);
    assert_eq!(r.dma_read(IOVA), None);
    let (hdr, _, _) = r.fault(1);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_DDT_MISCONFIGURED));
    // A device ID too wide for one level.
    let wide = r.iommu.device_ops(0x100);
    let t = wide.translate(IOVA, IommuAccessFlags::RO, 0);
    assert_eq!(t.perm, IommuAccessFlags::NONE);
    let (hdr, _, _) = r.fault(2);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_TTYPE_BLOCKED));
    // A process ID without a process directory.
    r.dc(DC_TC_V, 0, 0, 0, 0, 0, 0);
    assert_eq!(r.dma_read_as(IOVA, MemTxAttrs::new().with_pid(3)), None);
    let (hdr, _, _) = r.fault(3);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_TTYPE_BLOCKED));
    assert_eq!(get_field(hdr, FQ_HDR_PID), 3);
    assert_ne!(hdr & FQ_HDR_PV, 0);

    // DTF keeps page faults out of the queue but not context faults.
    r.dc(DC_TC_V | DC_TC_DTF, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.iommu.with_state(|s, _| s.ctx_inval(CtxMatch::All, 0, 0));
    assert_eq!(r.dma_read(IOVA + 0x5000), None);
    assert_eq!(r.r32(REG_FQT), 4);
}

#[test]
fn process_directory() {
    let r = rig();
    r.queues();
    let fsc = PDTP_MODE_PD8 << 60 | ppn_down(PDT);
    r.dc(DC_TC_V | DC_TC_PDTV, 0, 0, fsc, 0, 0, 0);
    // Process 5 has the Sv39 tables, with a PSCID.
    r.wq(PDT + (5 << 4), PC_TA_V | 9 << 12);
    r.wq(PDT + (5 << 4) + 8, Rig::sv39(S_ROOT));
    r.s_map(IOVA, DATA, LEAF & !PTE_U);
    r.wd(DATA, 0x0bad_cafe);
    let pid5 = MemTxAttrs::new().with_pid(5);
    assert_eq!(r.dma_read_as(IOVA, pid5), Some(0x0bad_cafe));
    // Process 6 has no valid entry.
    assert_eq!(r.dma_read_as(IOVA, MemTxAttrs::new().with_pid(6)), None);
    let (hdr, _, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_PDT_INVALID));
    // Without DPE, no process ID means no first stage.
    r.wd(IOVA & 0xff_ffff, 0x4242);
    assert_eq!(r.dma_read(IOVA & 0xff_ffff), Some(0x4242));
}

#[test]
fn g_stage_and_nested_translation() {
    let r = rig();
    r.queues();
    // G-stage: GPA IOVA goes to DATA2.
    let gatp = ATP_MODE_SV39 << 60 | 3 << 44 | ppn_down(G_ROOT);
    r.wq(G_ROOT + ((IOVA >> 30) & 0x7ff) * 8, ppn_down(G_L1) << 10 | NODE);
    r.wq(G_L1 + ((IOVA >> 21) & 0x1ff) * 8, ppn_down(G_L0) << 10 | NODE);
    r.wq(G_L0 + ((IOVA >> 12) & 0x1ff) * 8, ppn_down(DATA2) << 10 | LEAF);
    r.dc(DC_TC_V, gatp, 0, 0, 0, 0, 0);
    r.wd(DATA2 + 8, 0x600d);
    assert_eq!(r.dma_read(IOVA + 8), Some(0x600d));
    // Past the G-stage is a VS fault.
    assert_eq!(r.dma_read(IOVA + 0x1000), None);
    let (hdr, _, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_RD_FAULT_VS));

    // Nested: the first stage maps IOVA to the GPA DATA, and a 1G G-stage leaf maps the low
    // GPAs, page tables included, one to one.
    r.wq(G_ROOT, LEAF | PTE_X);
    r.s_map(IOVA, DATA, LEAF);
    r.dc(DC_TC_V, gatp, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.iommu.with_state(|s, _| s.ctx_inval(CtxMatch::All, 0, 0));
    r.wd(DATA + 8, 0xbeef);
    assert_eq!(r.dma_read(IOVA + 8), Some(0xbeef));

    // A G-stage root that is not 16K aligned is misconfigured.
    r.dc(DC_TC_V, ATP_MODE_SV39 << 60 | ppn_down(G_ROOT + 0x1000), 0, 0, 0, 0, 0);
    r.iommu.with_state(|s, _| s.ctx_inval(CtxMatch::All, 0, 0));
    assert_eq!(r.dma_read(IOVA + 8), None);
    let (hdr, _, _) = r.fault(1);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_DDT_MISCONFIGURED));
}

#[test]
fn command_queue() {
    let mut r = rig();
    r.queues();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    r.wd(DATA, 1);
    r.wd(DATA2, 2);
    assert_eq!(r.dma_read(IOVA), Some(1));

    // The IOTLB keeps the old page until IOTINVAL.VMA.
    r.s_map(IOVA, DATA2, LEAF);
    assert_eq!(r.dma_read(IOVA), Some(1));
    r.cmd(CMD_IOTINVAL_VMA, 0);
    assert_eq!(r.r32(REG_CQH), 1);
    assert_eq!(r.dma_read(IOVA), Some(2));

    // The address form only drops that page.
    r.s_map(IOVA, DATA, LEAF);
    r.cmd(CMD_IOTINVAL_VMA | CMD_IOTINVAL_AV, (IOVA + 0x1000) >> 2);
    assert_eq!(r.dma_read(IOVA), Some(2));
    r.cmd(CMD_IOTINVAL_VMA | CMD_IOTINVAL_AV, IOVA >> 2);
    assert_eq!(r.dma_read(IOVA), Some(1));

    // IODIR.INVAL_DDT drops the cached context.
    r.dc(DC_TC_V, 0, 0, 0, 0, 0, 0);
    r.wd(IOVA & 0xff_ffff, 3);
    r.cmd(CMD_IOTINVAL_VMA, 0);
    assert_eq!(r.dma_read(IOVA), Some(1));
    r.cmd(CMD_IODIR_INVAL_DDT | CMD_IODIR_DV | u64::from(SID) << 40, 0);
    assert_eq!(r.dma_read(IOVA & 0xff_ffff), Some(3));

    // IOFENCE.C with AV writes its data.
    r.cmd(CMD_IOFENCE_C | CMD_IOFENCE_AV | 0x1234_5678 << 32, FENCE >> 2);
    assert_eq!(r.rd(FENCE), 0x1234_5678);
    assert_eq!(r.r32(REG_CQH), r.cq_tail as u32);
    assert!(r.irqs().is_empty());

    // An illegal command stops the queue and interrupts.
    r.cmd(0x7f, 0);
    let head = r.r32(REG_CQH);
    assert_eq!(u64::from(head), r.cq_tail - 1);
    assert_ne!(r.r32(REG_CQCSR) & CQCSR_CMD_ILL, 0);
    assert_eq!(r.irqs(), vec![0]);
    // IODIR.INVAL_PDT needs DV, and the queue stays stopped behind it.
    r.cmd(CMD_IODIR_INVAL_PDT, 0);
    assert_eq!(r.r32(REG_CQH), head);

    // Clearing CMD_ILL runs the queue again, from the same bad command.
    r.wq(CQ + u64::from(head) * 16, CMD_IOFENCE_C);
    r.wq(CQ + u64::from(head + 1) * 16, CMD_IOFENCE_C);
    r.w32(REG_CQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE | CQCSR_CMD_ILL);
    assert_eq!(r.r32(REG_CQCSR) & CQCSR_CMD_ILL, 0);
    assert_eq!(u64::from(r.r32(REG_CQH)), r.cq_tail);

    // ATS commands are illegal without ats.
    let mut r = Rig::new(RiscvIommuConf { ats: false, ..RiscvIommuConf::default() });
    r.queues();
    r.cmd(CMD_ATS_INVAL, 0);
    assert_ne!(r.r32(REG_CQCSR) & CQCSR_CMD_ILL, 0);
}

#[test]
fn queue_overflow_and_memory_faults() {
    let r = rig();
    r.w32(REG_ICVEC, 0x3210);
    // Two entries, so one fault fills the queue.
    r.w64(REG_FQB, ppn_down(FQ) << 10);
    r.w32(REG_FQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE);
    assert_eq!(r.r32(REG_FQCSR) & QUEUE_ACTIVE, QUEUE_ACTIVE);
    assert_eq!(r.dma_read(0), None);
    assert_eq!(r.r32(REG_FQT), 1);
    assert_eq!(r.dma_read(0), None);
    assert_ne!(r.r32(REG_FQCSR) & QUEUE_OVERFLOW, 0);
    // The pending bit was already set, so only one interrupt.
    assert_eq!(r.irqs(), vec![1]);
    // The head only takes bits inside the mask.
    r.w32(REG_FQH, 0xffff_ffff);
    assert_eq!(r.r32(REG_FQH), 1);

    // Write 1 clears FQOF, and with no cause left FIP clears too.
    r.w32(REG_FQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE | QUEUE_OVERFLOW);
    assert_eq!(r.r32(REG_FQCSR) & QUEUE_OVERFLOW, 0);
    r.w32(REG_IPSR, IPSR_FIP);
    assert_eq!(r.r32(REG_IPSR), 0);

    // A queue outside memory.
    r.w32(REG_FQCSR, 0);
    assert_eq!(r.r32(REG_FQCSR) & QUEUE_ACTIVE, 0);
    r.w64(REG_FQB, ppn_down(0x4000_0000) << 10 | 3);
    r.w32(REG_FQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE);
    assert_eq!(r.dma_read(0), None);
    assert_ne!(r.r32(REG_FQCSR) & QUEUE_MEM_FAULT, 0);
    assert_eq!(r.irqs(), vec![1]);
}

#[test]
fn msi_flat_basic_and_mrif() {
    let r = rig();
    r.queues();
    let msiptp = DC_MSIPTP_MODE_FLAT << 60 | ppn_down(MSIPT);
    // Two interrupt files, picked by bit 12 of the address.
    r.dc(DC_TC_V, 0, 0, 0, msiptp, 1, ppn_down(MSI_ADDR));
    r.wq(MSIPT, ppn_down(TARGET) << 10 | MSI_PTE_M_BASIC << 1 | MSI_PTE_V);
    let nid = 0x7ff;
    r.wq(MSIPT + 16, (MRIF >> 9) << 7 | MSI_PTE_M_MRIF << 1 | MSI_PTE_V);
    r.wq(MSIPT + 24, ppn_down(NOTICE) << 10 | (nid & 0x3ff) | 1 << 60);

    // Basic: the IOMMU writes the data to the target page.
    assert_eq!(r.iommu.remap_msi(SID, MSI_ADDR + 4, 0x55), None);
    assert_eq!(r.rd(TARGET), 0x55);
    // Outside the window the message goes on untouched.
    assert_eq!(r.iommu.remap_msi(SID, 0xfee0_0000, 9), Some((0xfee0_0000, 9)));

    // MRIF: identity 70 is word 1 bit 6, and the notice goes out when it is enabled.
    r.wq(MRIF + 0x18, 1 << 6);
    assert_eq!(r.iommu.remap_msi(SID, MSI_ADDR + 0x1000, 70), None);
    assert_eq!(r.rq(MRIF + 0x10), 1 << 6);
    assert_eq!(r.rd(NOTICE), 0x7ff);
    // Identity 1 is not enabled, so only the pending bit.
    r.wd(NOTICE, 0);
    assert_eq!(r.iommu.remap_msi(SID, MSI_ADDR + 0x1000, 1), None);
    assert_eq!(r.rq(MRIF), 2);
    assert_eq!(r.rd(NOTICE), 0);
    // Identities past 2047 are misconfigured.
    assert_eq!(r.iommu.remap_msi(SID, MSI_ADDR + 0x1000, 4096), None);
    let (hdr, _, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_MSI_MISCONFIGURED));
    assert_eq!(get_field(hdr, FQ_HDR_TTYPE), TTYPE_UADDR_WR);

    // An invalid MSI PTE.
    r.wq(MSIPT, 0);
    assert_eq!(r.iommu.remap_msi(SID, MSI_ADDR, 1), None);
    let (hdr, _, _) = r.fault(1);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_MSI_INVALID));
    // A plain DMA write to the window is not an MSI.
    assert!(!r.dma_write(MSI_ADDR, 1));
}

#[test]
fn msi_after_first_stage() {
    let r = rig();
    r.queues();
    let msiptp = DC_MSIPTP_MODE_FLAT << 60 | ppn_down(MSIPT);
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), msiptp, 0, ppn_down(MSI_ADDR));
    r.wq(MSIPT, ppn_down(TARGET) << 10 | MSI_PTE_M_BASIC << 1 | MSI_PTE_V);
    // The device writes its MSI to IOVA, which the first stage maps to the MSI GPA.
    r.s_map(IOVA, MSI_ADDR, LEAF);
    assert_eq!(r.iommu.remap_msi(SID, IOVA + 8, 0x77), None);
    assert_eq!(r.rd(TARGET), 0x77);
    // Any other mapping is a plain write.
    r.s_map(IOVA + 0x1000, 0x3000, LEAF);
    assert_eq!(r.iommu.remap_msi(SID, IOVA + 0x1008, 1), Some((0x3008, 1)));
}

#[test]
fn translation_request_registers() {
    let r = rig();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    r.w64(REG_TR_REQ_IOVA, IOVA + 0x123);
    r.w64(REG_TR_REQ_CTL, TR_REQ_CTL_GO_BUSY | u64::from(SID) << 40);
    assert_eq!(r.r64(REG_TR_REQ_CTL) & TR_REQ_CTL_GO_BUSY, 0);
    assert_eq!(r.r64(REG_TR_RESPONSE), ppn_down(DATA) << 10);
    // Nothing at the next page.
    r.w64(REG_TR_REQ_IOVA, IOVA + 0x1000);
    r.w64(REG_TR_REQ_CTL, TR_REQ_CTL_GO_BUSY | u64::from(SID) << 40);
    assert_eq!(r.r64(REG_TR_RESPONSE), TR_RESPONSE_FAULT | u64::from(CAUSE_WR_FAULT_S) << 10);
    // A device with no context.
    r.w64(REG_TR_REQ_CTL, TR_REQ_CTL_GO_BUSY | 0x40 << 40);
    assert_eq!(r.r64(REG_TR_RESPONSE), TR_RESPONSE_FAULT | u64::from(CAUSE_DMA_DISABLED) << 10);
}

#[test]
fn ats_requests() {
    let r = rig();
    r.queues();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    // An ATS request has no permissions and needs EN_ATS.
    let t = r.iommu.translate(SID, 0, IOVA, IommuAccessFlags::NONE);
    assert_eq!(t.perm, IommuAccessFlags::NONE);
    let (hdr, _, _) = r.fault(0);
    assert_eq!(get_field(hdr, FQ_HDR_CAUSE), u64::from(CAUSE_TTYPE_BLOCKED));
    assert_eq!(get_field(hdr, FQ_HDR_TTYPE), TTYPE_PCIE_ATS_REQ);
    // With EN_ATS and EN_PRI a miss goes to the page request queue.
    r.w32(REG_ICVEC, 0x3210);
    r.w64(REG_PQB, ppn_down(DATA2) << 10 | 3);
    r.w32(REG_PQCSR, QUEUE_ENABLE | QUEUE_INTR_ENABLE);
    r.dc(DC_TC_V | DC_TC_EN_ATS | DC_TC_EN_PRI, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.iommu.with_state(|s, _| s.ctx_inval(CtxMatch::All, 0, 0));
    r.irqs();
    let t = r.iommu.translate(SID, 0, IOVA + 0x1000, IommuAccessFlags::NONE);
    assert_eq!(t.perm, IommuAccessFlags::NONE);
    assert_eq!(r.r32(REG_PQT), 1);
    assert_eq!(get_field(r.rq(DATA2), PREQ_HDR_DID), u64::from(SID));
    assert_eq!(r.rq(DATA2 + 8), IOVA + 0x1000 | PREQ_PAYLOAD_M);
    assert_eq!(r.irqs(), vec![3]);
}

#[test]
fn hpm_event_counters() {
    let r = rig();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    // Counter 0 counts untranslated requests, counter 1 TLB misses of device 9 only.
    r.w64(reg_iohpmevt(0), u64::from(HPMEVENT_URQ));
    let dv = IOHPMEVT_DV_GSCV | 9 << 36;
    r.w64(reg_iohpmevt(1), u64::from(HPMEVENT_TLB_MISS) | dv);
    r.w64(reg_iohpmevt(2), u64::from(HPMEVENT_TLB_MISS));
    for _ in 0..3 {
        assert!(r.dma_read(IOVA).is_some());
    }
    assert_eq!(r.r64(REG_IOHPMCTR_BASE), 3);
    assert_eq!(r.r64(REG_IOHPMCTR_BASE + 8), 0);
    assert_eq!(r.r64(REG_IOHPMCTR_BASE + 16), 1);
    // An event ID past the end reads back as invalid and stops the counter.
    r.w64(reg_iohpmevt(0), 0x7fff);
    assert_eq!(r.r64(reg_iohpmevt(0)), 0);
    assert!(r.dma_read(IOVA).is_some());
    assert_eq!(r.r64(REG_IOHPMCTR_BASE), 3);
    // Inhibited counters stand still.
    r.w64(reg_iohpmevt(0), u64::from(HPMEVENT_URQ));
    r.w32(REG_IOCOUNTINH, 2);
    assert!(r.dma_read(IOVA).is_some());
    assert_eq!(r.r64(REG_IOHPMCTR_BASE), 3);

    // Overflow sets OF and interrupts once.
    r.w32(REG_ICVEC, 0x3210);
    r.w32(REG_IOCOUNTINH, 0);
    r.w64(REG_IOHPMCTR_BASE, u64::MAX);
    assert!(r.dma_read(IOVA).is_some());
    assert_eq!(r.r64(REG_IOHPMCTR_BASE), 0);
    assert_eq!(r.r32(REG_IOCOUNTOVF), 2);
    assert_ne!(r.r64(reg_iohpmevt(0)) & IOHPMEVT_OF, 0);
    assert_eq!(r.irqs(), vec![2]);
    // Clearing OF in the selector clears it in iocountovf.
    r.w64(reg_iohpmevt(0), u64::from(HPMEVENT_URQ));
    assert_eq!(r.r32(REG_IOCOUNTOVF), 0);

    // Without HPM the registers are read-only zeros.
    let r = Rig::new(RiscvIommuConf { hpm_counters: 0, ..RiscvIommuConf::default() });
    r.w64(reg_iohpmevt(0), 1);
    assert_eq!(r.r64(reg_iohpmevt(0)), 0);
}

#[test]
fn hpm_cycle_counter() {
    let r = rig();
    r.w32(REG_ICVEC, 0x3210);
    r.clock.advance_to(1000);
    assert_eq!(r.r64(REG_IOHPMCYCLES), 1000);
    // Inhibiting freezes it.
    r.w32(REG_IOCOUNTINH, 1);
    r.clock.advance_to(1500);
    assert_eq!(r.r64(REG_IOHPMCYCLES), 1000);
    r.w32(REG_IOCOUNTINH, 0);
    r.clock.advance_to(1600);
    assert_eq!(r.r64(REG_IOHPMCYCLES), 1100);

    // Ten cycles short of the top, the timer fires when it wraps.
    r.w64(REG_IOHPMCYCLES, i64::MAX as u64 - 9);
    r.clock.advance_to(1605);
    assert_eq!(r.r64(REG_IOHPMCYCLES), i64::MAX as u64 - 4);
    assert!(r.irqs().is_empty());
    r.clock.advance_to(1610);
    assert_ne!(r.r64(REG_IOHPMCYCLES) & IOHPMCYCLES_OVF, 0);
    assert_eq!(r.r32(REG_IOCOUNTOVF) & IOCOUNTOVF_CY, IOCOUNTOVF_CY);
    assert_ne!(r.r32(REG_IPSR) & IPSR_PMIP, 0);
    assert_eq!(r.irqs(), vec![2]);
    // Writing it back without OF clears CY.
    r.w64(REG_IOHPMCYCLES, 5);
    assert_eq!(r.r32(REG_IOCOUNTOVF) & IOCOUNTOVF_CY, 0);
    assert_eq!(r.r64(REG_IOHPMCYCLES), 5);
}

#[test]
fn reset_clears_queues_and_caches() {
    let r = rig();
    r.queues();
    r.dc(DC_TC_V, 0, 0, Rig::sv39(S_ROOT), 0, 0, 0);
    r.s_map(IOVA, DATA, LEAF);
    assert!(r.dma_read(IOVA).is_some());
    r.iommu.reset();
    assert_eq!(r.r64(REG_DDTP), DDTP_MODE_OFF);
    assert_eq!(r.r32(REG_CQCSR), 0);
    assert_eq!(r.r32(REG_FQCSR), 0);
    assert_eq!(r.r32(REG_IPSR), 0);
    r.iommu.with_state(|s, _| {
        assert!(s.ctx_cache.is_empty());
        assert!(s.iot_cache.is_empty());
    });
    assert_eq!(r.dma_read(IOVA), None);
}
