// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the emulated GICv3 from hw/intc/arm_gicv3*.c, driven through the
//! distributor and redistributor MMIO and the ICC_* system register entry points.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_intc::gicv3::*;
use ruvm_mem::{AccessCtx, AccessSize, AddressSpace, Endian, MemTxAttrs, MemorySystem, MmioOps};

/// Non-secure EL1 on a CPU without EL2 or EL3.
const NS_EL1: IccCpuCtx = IccCpuCtx {
    el: 1,
    has_el2: false,
    has_el3: false,
    secure: false,
    secure_below_el3: false,
    hcr_el2: 0,
    scr_el3: 0,
};

/// Non-secure EL1 on a CPU with EL3.
const NS_EL1_EL3: IccCpuCtx = IccCpuCtx { has_el3: true, ..NS_EL1 };

const GICD_CTLR: u64 = 0x0;
const GICD_TYPER: u64 = 0x4;
const GICD_IIDR: u64 = 0x8;
const GICD_IGROUPR: u64 = 0x80;
const GICD_ISENABLER: u64 = 0x100;
const GICD_ISPENDR: u64 = 0x200;
const GICD_ISACTIVER: u64 = 0x300;
const GICD_IPRIORITYR: u64 = 0x400;
const GICD_ICFGR: u64 = 0xc00;
const GICD_IGRPMODR: u64 = 0xd00;
const GICD_NSACR: u64 = 0xe00;
const GICD_IROUTER: u64 = 0x6000;

const GICR_CTLR: u64 = 0x0;
const GICR_IIDR: u64 = 0x4;
const GICR_TYPER: u64 = 0x8;
const GICR_WAKER: u64 = 0x14;
const GICR_IGROUPR0: u64 = 0x10080;
const GICR_ISENABLER0: u64 = 0x10100;
const GICR_ISPENDR0: u64 = 0x10200;
const GICR_IPRIORITYR: u64 = 0x10400;
const GICR_ICFGR1: u64 = 0x10c04;

struct Rig {
    gic: Arc<GicV3>,
    dist: Arc<dyn MmioOps>,
    redist: Arc<dyn MmioOps>,
    irq: Vec<Arc<AtomicI32>>,
    fiq: Vec<Arc<AtomicI32>>,
}

fn props(mp_affinity: Vec<u64>, security_extn: bool) -> GicV3Props {
    GicV3Props {
        num_cpu: mp_affinity.len(),
        num_irq: 64,
        revision: 3,
        security_extn,
        redist_region_count: vec![mp_affinity.len() as u32],
        mp_affinity,
        pribits: 5,
        has_lpi: false,
    }
}

fn watch() -> (Arc<AtomicI32>, IrqLine) {
    let level = Arc::new(AtomicI32::new(0));
    let l = level.clone();
    (level, IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)))
}

fn rig_with(props: GicV3Props) -> Rig {
    let gic = GicV3::new(props).unwrap();
    let mut irq = Vec::new();
    let mut fiq = Vec::new();
    for cpu in 0..gic.num_cpu() {
        let (level, line) = watch();
        gic.cpu_irq(cpu).connect(line);
        irq.push(level);
        let (level, line) = watch();
        gic.cpu_fiq(cpu).connect(line);
        fiq.push(level);
    }
    let dist = gic.dist_ops();
    let redist = gic.redist_ops(0);
    Rig { gic, dist, redist, irq, fiq }
}

/// Two CPUs at affinity 0.0.0.0 and 0.0.0.1, no security extensions, 64 interrupts.
fn rig() -> Rig {
    rig_with(props(vec![0, 1], false))
}

fn cx(secure: bool) -> AccessCtx {
    AccessCtx::new(MemTxAttrs::new().with_secure(secure))
}

fn size(bytes: u32) -> AccessSize {
    AccessSize::new(bytes).unwrap()
}

impl Rig {
    fn dr(&self, off: u64) -> u64 {
        self.dist.read(&cx(true), off, AccessSize::B4).unwrap()
    }
    fn dw(&self, off: u64, v: u64) {
        self.dist.write(&cx(true), off, AccessSize::B4, v).unwrap();
    }
    fn dr_ns(&self, off: u64) -> u64 {
        self.dist.read(&cx(false), off, AccessSize::B4).unwrap()
    }
    fn dw_ns(&self, off: u64, v: u64) {
        self.dist.write(&cx(false), off, AccessSize::B4, v).unwrap();
    }
    fn dw_sized(&self, secure: bool, off: u64, bytes: u32, v: u64) {
        self.dist.write(&cx(secure), off, size(bytes), v).unwrap();
    }
    fn dr_sized(&self, secure: bool, off: u64, bytes: u32) -> u64 {
        self.dist.read(&cx(secure), off, size(bytes)).unwrap()
    }
    fn rr(&self, cpu: u64, off: u64) -> u64 {
        self.redist.read(&cx(true), cpu * GICV3_REDIST_SIZE + off, AccessSize::B4).unwrap()
    }
    fn rw(&self, cpu: u64, off: u64, v: u64) {
        self.redist.write(&cx(true), cpu * GICV3_REDIST_SIZE + off, AccessSize::B4, v).unwrap();
    }
    fn rr_ns(&self, cpu: u64, off: u64) -> u64 {
        self.redist.read(&cx(false), cpu * GICV3_REDIST_SIZE + off, AccessSize::B4).unwrap()
    }
    fn rw_ns(&self, cpu: u64, off: u64, v: u64) {
        let off = cpu * GICV3_REDIST_SIZE + off;
        self.redist.write(&cx(false), off, AccessSize::B4, v).unwrap();
    }
    fn icc_r(&self, cpu: usize, reg: IccReg) -> u64 {
        self.gic.icc_read(cpu, reg, &NS_EL1)
    }
    fn icc_w(&self, cpu: usize, reg: IccReg, v: u64) {
        self.gic.icc_write(cpu, reg, &NS_EL1, v);
    }
    fn irq(&self, cpu: usize) -> i32 {
        self.irq[cpu].load(Ordering::SeqCst)
    }
    fn fiq(&self, cpu: usize) -> i32 {
        self.fiq[cpu].load(Ordering::SeqCst)
    }

    /// Wake `cpu`'s redistributor and open its CPU interface for Group 1.
    fn open_cpu(&self, cpu: usize) {
        self.rw(cpu as u64, GICR_WAKER, 0);
        self.icc_w(cpu, IccReg::Pmr, 0xff);
        self.icc_w(cpu, IccReg::Igrpen1, 1);
    }

    /// Make SPI `irq` Group 1, enabled, at `prio`, routed to `cpu`.
    fn setup_spi(&self, irq: u64, prio: u64, cpu: u64) {
        let word = irq / 32 * 4;
        let bit = 1 << (irq % 32);
        self.dw(GICD_IGROUPR + word, self.dr(GICD_IGROUPR + word) | bit);
        self.dw(GICD_ISENABLER + word, bit);
        self.dw_sized(true, GICD_IPRIORITYR + irq, 1, prio);
        self.dw_sized(true, GICD_IROUTER + irq * 8, 8, cpu);
    }

    /// Make SGI or PPI `irq` of `cpu` Group 1, enabled, at `prio`.
    fn setup_internal(&self, cpu: u64, irq: u64, prio: u64) {
        let bit = 1 << irq;
        self.rw(cpu, GICR_IGROUPR0, self.rr(cpu, GICR_IGROUPR0) | bit);
        self.rw(cpu, GICR_ISENABLER0, bit);
        let off = cpu * GICV3_REDIST_SIZE + GICR_IPRIORITYR + irq;
        self.redist.write(&cx(true), off, AccessSize::B1, prio).unwrap();
    }
}

#[test]
fn realize_errors_match_qemu() {
    let err = |p: GicV3Props| GicV3::new(p).unwrap_err();
    let base = props(vec![0, 1], false);
    assert_eq!(err(GicV3Props { revision: 2, ..base.clone() }), "unsupported GIC revision 2");
    assert_eq!(err(GicV3Props { revision: 4, ..base.clone() }), "unsupported GIC revision 4");
    assert_eq!(
        err(GicV3Props { num_irq: 1024, ..base.clone() }),
        "requested 1024 interrupt lines exceeds GIC maximum 1020"
    );
    assert_eq!(
        err(GicV3Props { num_irq: 16, ..base.clone() }),
        "requested 16 interrupt lines is below GIC minimum 32"
    );
    assert_eq!(
        err(GicV3Props { num_irq: 48, ..base.clone() }),
        "48 interrupt lines unsupported: not divisible by 32"
    );
    assert_eq!(
        err(GicV3Props { redist_region_count: vec![1], ..base.clone() }),
        "Capacity of the redist regions(1) does not match the number of vcpus(2)"
    );
    assert_eq!(
        err(GicV3Props { num_cpu: 0, mp_affinity: vec![], ..base }),
        "num-cpu must be at least 1"
    );
}

#[test]
fn id_and_typer_registers() {
    let r = rig();
    // No1N, A3V, IDbits 0xf, no SecurityExtn (DS is set), ITLinesNumber 1.
    assert_eq!(r.dr(GICD_TYPER), (1 << 25) | (1 << 24) | (0xf << 19) | 1);
    assert_eq!(r.dr(GICD_IIDR), 0x43b);
    assert_eq!(r.dr(0xffe0), 0x92);
    assert_eq!(r.dr(0xffe8), 0x3b);
    assert_eq!(r.dr(0xfff0), 0x0d);
    assert_eq!(r.dr(0xfffc), 0xb1);

    assert_eq!(r.rr(0, GICR_IIDR), 0x43b);
    assert_eq!(r.rr(0, 0xffe0), 0x93);
    assert_eq!(r.rr(1, 0xffe8), 0x3b);
    assert_eq!(r.rr(0, GICR_CTLR), 0);
    assert_eq!(r.rr(0, GICR_WAKER), 6);

    // CPU 0 is not the last in the region, CPU 1 is.
    assert_eq!(r.rr(0, GICR_TYPER), 1 << 24);
    assert_eq!(r.rr(0, GICR_TYPER + 4), 0);
    assert_eq!(r.rr(1, GICR_TYPER), (1 << 24) | (1 << 8) | 0x10);
    assert_eq!(r.rr(1, GICR_TYPER + 4), 1);
    let typer = r.redist.read(&cx(true), GICV3_REDIST_SIZE + GICR_TYPER, AccessSize::B8).unwrap();
    assert_eq!(typer, (1 << 32) | (1 << 24) | (1 << 8) | 0x10);

    // Aff3 moves down to bits [63:56], and each region marks its own last redistributor.
    let mut p = props(vec![0x01_0000_0203, 0x0001_0000], false);
    p.redist_region_count = vec![1, 1];
    let gic = GicV3::new(p).unwrap();
    assert_eq!(gic.redist_region_size(0), GICV3_REDIST_SIZE);
    for (region, aff) in [(0, 0x0100_0203u64), (1, 0x0001_0000)] {
        let ops = gic.redist_ops(region);
        let typer = ops.read(&cx(true), GICR_TYPER, AccessSize::B8).unwrap();
        assert_eq!(typer, (aff << 32) | (1 << 24) | ((region as u64) << 8) | 0x10);
        // Past the end of the region there is no redistributor.
        assert_eq!(ops.read(&cx(true), GICV3_REDIST_SIZE + GICR_TYPER, AccessSize::B4).unwrap(), 0);
    }

    // WAKER: only ProcessorSleep is writable and ChildrenAsleep follows it.
    r.rw(0, GICR_WAKER, 0);
    assert_eq!(r.rr(0, GICR_WAKER), 0);
    r.rw(0, GICR_WAKER, 0xff);
    assert_eq!(r.rr(0, GICR_WAKER), 6);
}

#[test]
fn gicd_ctlr_are_ds_and_group_enables() {
    // One security state: DS and ARE are RAO/WI and only the two enables are writable.
    let r = rig();
    assert_eq!(r.dr(GICD_CTLR), 0x50);
    r.dw(GICD_CTLR, 0xffff_ffff);
    assert_eq!(r.dr(GICD_CTLR), 0x53);
    r.dw(GICD_CTLR, 0);
    assert_eq!(r.dr(GICD_CTLR), 0x50);

    // Two security states: ARE_S and ARE_NS set, DS clear.
    let r = rig_with(props(vec![0, 1], true));
    assert_eq!(r.dr(GICD_CTLR), 0x30);
    assert_eq!(r.dr(GICD_TYPER) & (1 << 10), 1 << 10);
    r.dw(GICD_CTLR, 0x7);
    assert_eq!(r.dr(GICD_CTLR), 0x37);
    // The Non-secure view only has ARE_NS (at bit 4) and EnableGrp1NS.
    assert_eq!(r.dr_ns(GICD_CTLR), 0x12);
    r.dw_ns(GICD_CTLR, 0);
    assert_eq!(r.dr(GICD_CTLR), 0x35);
    r.dw_ns(GICD_CTLR, 0x7);
    assert_eq!(r.dr(GICD_CTLR), 0x37);
    // Setting DS clears EnableGrp1S and ARE_NS for good.
    r.dw(GICD_CTLR, 0x47);
    assert_eq!(r.dr(GICD_CTLR), 0x53);
    assert_eq!(r.dr(GICD_TYPER) & (1 << 10), 0);
    r.dw(GICD_CTLR, 0x7);
    assert_eq!(r.dr(GICD_CTLR), 0x53);
}

#[test]
fn spi_routed_by_irouter_reaches_the_cpu() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(40, 0x80, 1);
    assert_eq!(r.dr_sized(true, GICD_IROUTER + 40 * 8, 8), 1);
    assert_eq!(r.dr_sized(true, GICD_IPRIORITYR + 40, 1), 0x80);
    r.open_cpu(1);

    let spi = r.gic.spi(8);
    spi.raise();
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.fiq(1), 0);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.dr(GICD_ISPENDR + 4), 1 << 8);
    assert_eq!(r.icc_r(1, IccReg::Hppir1), 40);
    assert_eq!(r.icc_r(1, IccReg::Rpr), 0xff);

    // Acknowledge: the interrupt goes active and the line drops.
    assert_eq!(r.icc_r(1, IccReg::Iar1), 40);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.dr(GICD_ISACTIVER + 4), 1 << 8);
    assert_eq!(r.icc_r(1, IccReg::Rpr), 0x80);
    assert_eq!(r.icc_r(1, IccReg::Ap1r(0)), 1 << 16);
    assert_eq!(r.icc_r(1, IccReg::Iar1), 1023);

    // EOI drops the priority and deactivates.
    spi.lower();
    r.icc_w(1, IccReg::Eoir1, 40);
    assert_eq!(r.icc_r(1, IccReg::Rpr), 0xff);
    assert_eq!(r.icc_r(1, IccReg::Ap1r(0)), 0);
    assert_eq!(r.dr(GICD_ISACTIVER + 4), 0);
    assert_eq!(r.irq(1), 0);

    // Rerouting to CPU 0 moves the pending interrupt with it.
    spi.raise();
    assert_eq!(r.irq(1), 1);
    r.open_cpu(0);
    r.dw_sized(true, GICD_IROUTER + 40 * 8, 8, 0);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.irq(0), 1);
    // A route to no CPU leaves it pending but undelivered.
    r.dw(GICD_IROUTER + 40 * 8, 7);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.dr(GICD_ISPENDR + 4), 1 << 8);
}

#[test]
fn level_interrupt_reasserts_after_eoi() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(33, 0x80, 0);
    r.open_cpu(0);
    let spi = r.gic.gpio_in(1);
    spi.raise();
    assert_eq!(r.icc_r(0, IccReg::Iar1), 33);
    assert_eq!(r.irq(0), 0);
    r.icc_w(0, IccReg::Eoir1, 33);
    // Still high, so pending again.
    assert_eq!(r.irq(0), 1);
}

#[test]
fn ppi_through_its_line() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_internal(0, 27, 0xa0);
    assert_eq!(r.rr(0, GICR_IPRIORITYR + 24), 0xa0 << 24);
    r.open_cpu(0);
    r.open_cpu(1);

    let ppi = r.gic.ppi(0, 27);
    ppi.raise();
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.rr(0, GICR_ISPENDR0), 1 << 27);
    assert_eq!(r.icc_r(0, IccReg::Iar1), 27);
    assert_eq!(r.irq(0), 0);
    ppi.lower();
    r.icc_w(0, IccReg::Eoir1, 27);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.rr(0, GICR_ISPENDR0), 0);

    // The same PPI of CPU 1 is a different line.
    r.setup_internal(1, 27, 0xa0);
    r.gic.ppi(1, 27).raise();
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.irq(0), 0);
}

#[test]
fn sgi1r_targets_by_affinity() {
    // CPU 0 at 0.0.0.0, CPU 1 at 0.0.1.0.
    let r = rig_with(props(vec![0, 0x100], false));
    r.dw(GICD_CTLR, 0x2);
    for cpu in 0..2 {
        r.setup_internal(cpu, 5, 0x40);
        r.open_cpu(cpu as usize);
    }

    // Aff1 = 0, target list bit 1: nobody is at 0.0.0.1.
    r.icc_w(0, IccReg::Sgi1r, (5 << 24) | 0b10);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(1), 0);

    // Aff1 = 1, target list bit 0: CPU 1.
    r.icc_w(0, IccReg::Sgi1r, (1 << 16) | (5 << 24) | 1);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.icc_r(1, IccReg::Iar1), 5);
    r.icc_w(1, IccReg::Eoir1, 5);
    assert_eq!(r.irq(1), 0);

    // IRM = 1 from CPU 1: everybody else.
    r.icc_w(1, IccReg::Sgi1r, (1 << 40) | (5 << 24));
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(1), 0);
}

#[test]
fn group0_goes_to_fiq_with_ds() {
    let r = rig();
    r.dw(GICD_CTLR, 0x3);
    // SPI 33 stays in Group 0.
    r.dw(GICD_ISENABLER + 4, 1 << 1);
    r.dw_sized(true, GICD_IPRIORITYR + 33, 1, 0x10);
    r.dw_sized(true, GICD_IROUTER + 33 * 8, 8, 0);
    r.icc_w(0, IccReg::Pmr, 0xff);
    r.icc_w(0, IccReg::Igrpen0, 1);

    r.gic.spi(1).raise();
    assert_eq!(r.fiq(0), 1);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.icc_r(0, IccReg::Hppir1), 1023);
    assert_eq!(r.icc_r(0, IccReg::Iar1), 1023);
    assert_eq!(r.icc_r(0, IccReg::Hppir0), 33);
    assert_eq!(r.icc_r(0, IccReg::Iar0), 33);
    assert_eq!(r.fiq(0), 0);
    assert_eq!(r.icc_r(0, IccReg::Ap0r(0)), 1 << 2);
    r.gic.spi(1).lower();
    r.icc_w(0, IccReg::Eoir0, 33);
    assert_eq!(r.icc_r(0, IccReg::Ap0r(0)), 0);

    // Disabling Group 0 at the CPU interface masks it.
    r.gic.spi(1).raise();
    assert_eq!(r.fiq(0), 1);
    r.icc_w(0, IccReg::Igrpen0, 0);
    assert_eq!(r.fiq(0), 0);
}

#[test]
fn pmr_masks_and_bpr_controls_preemption() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(32, 0x40, 0);
    r.setup_spi(33, 0x20, 0);
    r.open_cpu(0);

    // Five priority bits: the bottom three are RAZ.
    r.icc_w(0, IccReg::Pmr, 0x47);
    assert_eq!(r.icc_r(0, IccReg::Pmr), 0x40);
    r.gic.spi(0).raise();
    assert_eq!(r.irq(0), 0);
    r.icc_w(0, IccReg::Pmr, 0x48);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.icc_r(0, IccReg::Iar1), 32);
    assert_eq!(r.irq(0), 0);

    // BPR1 at its minimum of 3: 0x20 preempts the running 0x40.
    r.icc_w(0, IccReg::Pmr, 0xff);
    assert_eq!(r.icc_r(0, IccReg::Bpr1), 3);
    r.gic.spi(1).raise();
    assert_eq!(r.irq(0), 1);
    // BPR1 = 7: group priority is bit 7 only, so 0x20 and 0x40 are equal and nothing preempts.
    r.icc_w(0, IccReg::Bpr1, 7);
    assert_eq!(r.icc_r(0, IccReg::Bpr1), 7);
    assert_eq!(r.irq(0), 0);
    // Writes below the minimum clamp to it.
    r.icc_w(0, IccReg::Bpr1, 0);
    assert_eq!(r.icc_r(0, IccReg::Bpr1), 3);
    assert_eq!(r.irq(0), 1);

    // Nested acknowledge, then the running priority unwinds in order.
    assert_eq!(r.icc_r(0, IccReg::Iar1), 33);
    assert_eq!(r.icc_r(0, IccReg::Rpr), 0x20);
    r.gic.spi(1).lower();
    r.icc_w(0, IccReg::Eoir1, 33);
    assert_eq!(r.icc_r(0, IccReg::Rpr), 0x40);
    r.gic.spi(0).lower();
    r.icc_w(0, IccReg::Eoir1, 32);
    assert_eq!(r.icc_r(0, IccReg::Rpr), 0xff);
}

#[test]
fn eoimode_splits_drop_and_deactivate() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(32, 0x80, 0);
    r.open_cpu(0);
    r.icc_w(0, IccReg::CtlrEl1, 2);
    assert_eq!(r.icc_r(0, IccReg::CtlrEl1) & 3, 2);

    r.gic.spi(0).raise();
    assert_eq!(r.icc_r(0, IccReg::Iar1), 32);
    r.gic.spi(0).lower();
    r.icc_w(0, IccReg::Eoir1, 32);
    assert_eq!(r.icc_r(0, IccReg::Rpr), 0xff);
    assert_eq!(r.dr(GICD_ISACTIVER + 4), 1);
    r.icc_w(0, IccReg::Dir, 32);
    assert_eq!(r.dr(GICD_ISACTIVER + 4), 0);
}

#[test]
fn icfgr_edge_and_level() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(34, 0x80, 0);
    r.setup_spi(35, 0x80, 0);
    r.open_cpu(0);

    // SPI 34 edge triggered: field 2 of ICFGR2, bit 5.
    r.dw(GICD_ICFGR + 8, 0xffff_ffff & !0x80);
    assert_eq!(r.dr(GICD_ICFGR + 8), 0xaaaa_aaaa & !0x80);
    r.dw(GICD_ICFGR + 8, 0x20);
    assert_eq!(r.dr(GICD_ICFGR + 8), 0x20);

    // A pulse latches an edge interrupt.
    r.gic.spi(2).raise();
    r.gic.spi(2).lower();
    assert_eq!(r.dr(GICD_ISPENDR + 4), 1 << 2);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.icc_r(0, IccReg::Iar1), 34);
    assert_eq!(r.dr(GICD_ISPENDR + 4), 0);
    r.icc_w(0, IccReg::Eoir1, 34);
    assert_eq!(r.irq(0), 0);

    // A level interrupt is pending only while its line is high.
    r.gic.spi(3).raise();
    assert_eq!(r.dr(GICD_ISPENDR + 4), 1 << 3);
    assert_eq!(r.irq(0), 1);
    r.gic.spi(3).lower();
    assert_eq!(r.dr(GICD_ISPENDR + 4), 0);
    assert_eq!(r.irq(0), 0);

    // Redistributor: SGIs are edge, PPIs level until ICFGR1 says otherwise.
    assert_eq!(r.rr(0, 0x10c00), 0xaaaa_aaaa);
    assert_eq!(r.rr(0, GICR_ICFGR1), 0);
    r.rw(0, GICR_ICFGR1, 2 << 22);
    assert_eq!(r.rr(0, GICR_ICFGR1), 2 << 22);
    r.setup_internal(0, 27, 0x80);
    let ppi = r.gic.ppi(0, 27);
    ppi.raise();
    ppi.lower();
    assert_eq!(r.rr(0, GICR_ISPENDR0), 1 << 27);
}

#[test]
fn nonsecure_accesses_are_banked() {
    let r = rig_with(props(vec![0, 1], true));
    let irq = 40u64;
    let word = 4u64;
    let bit = 1u64 << 8;

    // Group 0 state is invisible to Non-secure and its writes are ignored.
    r.dw_ns(GICD_IGROUPR + word, bit);
    assert_eq!(r.dr(GICD_IGROUPR + word), 0);
    r.dw_ns(GICD_ISENABLER + word, bit);
    assert_eq!(r.dr(GICD_ISENABLER + word), 0);
    r.dw_sized(false, GICD_IPRIORITYR + irq, 1, 0xff);
    assert_eq!(r.dr_sized(true, GICD_IPRIORITYR + irq, 1), 0);
    r.dw(GICD_IPRIORITYR + 40, 0x10);
    assert_eq!(r.dr_ns(GICD_IPRIORITYR + 40), 0);
    r.dw_ns(GICD_ICFGR + 8, 0xaaaa_aaaa);
    assert_eq!(r.dr(GICD_ICFGR + 8), 0);
    r.dw(GICD_IGRPMODR + word, bit);
    assert_eq!(r.dr(GICD_IGRPMODR + word), bit);
    assert_eq!(r.dr_ns(GICD_IGRPMODR + word), 0);
    r.dw_sized(false, GICD_IROUTER + irq * 8, 8, 1);
    assert_eq!(r.dr_sized(true, GICD_IROUTER + irq * 8, 8), 0);

    // GICD_NSACR = 3 opens IROUTER to Non-secure.
    r.dw(GICD_NSACR + 8, 3 << 16);
    assert_eq!(r.dr(GICD_NSACR + 8), 3 << 16);
    assert_eq!(r.dr_ns(GICD_NSACR + 8), 0);
    r.dw_sized(false, GICD_IROUTER + irq * 8, 8, 1);
    assert_eq!(r.dr_sized(false, GICD_IROUTER + irq * 8, 8), 1);
    // And NSACR >= 1 lets Non-secure set it pending.
    r.dw_ns(GICD_ISPENDR + word, bit);
    assert_eq!(r.dr(GICD_ISPENDR + word), bit);

    // Once Non-secure Group 1, the Non-secure view of the priority is shifted.
    r.dw(GICD_IGROUPR + word, bit);
    assert_eq!(r.dr_ns(GICD_IGROUPR + word), 0);
    r.dw_sized(false, GICD_IPRIORITYR + irq, 1, 0x40);
    assert_eq!(r.dr_sized(true, GICD_IPRIORITYR + irq, 1), 0xa0);
    assert_eq!(r.dr_sized(false, GICD_IPRIORITYR + irq, 1), 0x40);
    r.dw_ns(GICD_ISENABLER + word, bit);
    assert_eq!(r.dr_ns(GICD_ISENABLER + word), bit);

    // The redistributor follows GICR_IGROUPR0.
    r.rw_ns(0, GICR_IGROUPR0, 1 << 27);
    assert_eq!(r.rr(0, GICR_IGROUPR0), 0);
    r.rw_ns(0, GICR_ISENABLER0, 1 << 27);
    assert_eq!(r.rr(0, GICR_ISENABLER0), 0);
    r.rw(0, GICR_IGROUPR0, 1 << 27);
    assert_eq!(r.rr_ns(0, GICR_IGROUPR0), 0);
    r.rw_ns(0, GICR_ISENABLER0, 1 << 27);
    assert_eq!(r.rr_ns(0, GICR_ISENABLER0), 1 << 27);
    r.rw_ns(0, GICR_IPRIORITYR + 24, 0x40 << 24);
    assert_eq!(r.rr(0, GICR_IPRIORITYR + 24), 0xa0 << 24);
    assert_eq!(r.rr_ns(0, 0x10d00), 0);
    assert_eq!(r.rr_ns(0, 0x10e00), 0);

    // Deliver the Non-secure SPI to a Non-secure EL1 with EL3 present.
    r.dw(GICD_CTLR, 0x7);
    r.rw(1, GICR_WAKER, 0);
    let ctx = NS_EL1_EL3;
    r.gic.icc_write(1, IccReg::Pmr, &ctx, 0xff);
    // Group 0 is not routed to EL3, so PMR has no Non-secure view: five bits are kept.
    assert_eq!(r.gic.icc_read(1, IccReg::Pmr, &ctx), 0xf8);
    r.gic.icc_write(1, IccReg::Igrpen1, &ctx, 1);
    r.gic.spi(8).raise();
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.fiq(1), 0);
    assert_eq!(r.gic.icc_read(1, IccReg::Iar1, &ctx), 40);
    // Non-secure writes to GICD_CTLR cannot touch the Group 0 enable.
    r.dw_ns(GICD_CTLR, 0);
    assert_eq!(r.dr(GICD_CTLR), 0x35);
}

#[test]
fn reset_clears_state_and_lines() {
    let r = rig();
    r.dw(GICD_CTLR, 0x2);
    r.setup_spi(32, 0x80, 0);
    r.open_cpu(0);
    r.gic.spi(0).raise();
    assert_eq!(r.irq(0), 1);
    r.gic.reset();
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.dr(GICD_CTLR), 0x50);
    assert_eq!(r.dr(GICD_ISENABLER + 4), 0);
    assert_eq!(r.rr(0, GICR_WAKER), 6);
    // The CPU interface is in the CPU's reset domain.
    assert_eq!(r.icc_r(0, IccReg::Igrpen1), 1);
    r.gic.cpuif_reset(0);
    assert_eq!(r.icc_r(0, IccReg::Igrpen1), 0);
    assert_eq!(r.icc_r(0, IccReg::Pmr), 0);
}

#[test]
fn access_checks() {
    let r = rig();
    let g = &r.gic;
    assert_eq!(g.icc_access(0, IccReg::Iar1, &NS_EL1, true), IccAccess::Ok);
    // Five bits of priority: only AP0R0 and AP1R0 exist.
    assert_eq!(g.icc_access(0, IccReg::Ap1r(1), &NS_EL1, true), IccAccess::Undefined);
    assert!(IccReg::Ap1r(1).exists(6));
    assert!(!IccReg::Ap1r(2).exists(6));
    assert!(IccReg::Ap1r(3).exists(8));

    let hyp = IccCpuCtx { has_el2: true, hcr_el2: 1 << 4, ..NS_EL1 };
    assert_eq!(g.icc_access(0, IccReg::Sgi1r, &hyp, false), IccAccess::TrapEl2);
    assert_eq!(g.icc_access(0, IccReg::Iar1, &hyp, true), IccAccess::Ok);

    let routed = IccCpuCtx { has_el3: true, scr_el3: 0b110, ..NS_EL1 };
    assert_eq!(g.icc_access(0, IccReg::Pmr, &routed, true), IccAccess::TrapEl3);
    assert_eq!(g.icc_access(0, IccReg::Iar0, &routed, true), IccAccess::TrapEl3);
    assert_eq!(g.icc_access(0, IccReg::SreEl1, &routed, true), IccAccess::Ok);
    let el2 = IccCpuCtx { el: 2, has_el2: true, hcr_el2: 0x18, ..routed };
    assert_eq!(g.icc_access(0, IccReg::Dir, &el2, false), IccAccess::TrapEl3);
    let el3 = IccCpuCtx { el: 3, secure: true, ..routed };
    assert_eq!(g.icc_access(0, IccReg::Pmr, &el3, true), IccAccess::Ok);

    for reg in IccReg::ALL {
        let (op0, op1, crn, crm, op2) = reg.encoding();
        assert_eq!(IccReg::from_encoding(op0, op1, crn, crm, op2), Some(reg));
        assert!(reg.name().starts_with("ICC_"));
    }
    assert_eq!(IccReg::Ap1r(2).name(), "ICC_AP1R2_EL1");
    assert_eq!(r.icc_r(0, IccReg::SreEl1), 7);
}

#[test]
fn virtual_interface() {
    let r = rig();
    let g = &r.gic;
    let (virq, line) = watch();
    g.cpu_virq(0).connect(line);
    // As the boards do, the maintenance interrupt feeds PPI 25 of the same CPU.
    g.maintenance_irq(0).connect(g.ppi(0, 25));
    let hyp = IccCpuCtx { el: 2, has_el2: true, ..NS_EL1 };
    let guest = IccCpuCtx { has_el2: true, hcr_el2: 0x18, ..NS_EL1 };

    // Four list registers, five bits of virtual priority and preemption, A3V, TDS and nV4.
    let vtr = (4 << 29) | (4 << 26) | (1 << 23) | (1 << 21) | (1 << 20) | (1 << 19) | 3;
    assert_eq!(g.icc_read(0, IccReg::IchVtr, &hyp), vtr);
    assert!(IccReg::IchLr(3).exists(5));
    assert!(!IccReg::IchLr(4).exists(5));
    assert!(!IccReg::IchAp1r(1).exists(5));
    for reg in IccReg::ICH_ALL {
        let (op0, op1, crn, crm, op2) = reg.encoding();
        assert_eq!(IccReg::from_encoding(op0, op1, crn, crm, op2), Some(reg));
        assert!(reg.name().starts_with("ICH_") && reg.is_ich());
    }
    assert_eq!(IccReg::IchLr(3).name(), "ICH_LR3_EL2");

    // Enable the interface and virtual Group 1 with an open VPMR, then queue vINTID 27 at
    // priority 0x80.
    g.icc_write(0, IccReg::IchVmcr, &hyp, (0xff << 24) | 2);
    g.icc_write(0, IccReg::IchHcr, &hyp, 1);
    assert_eq!(virq.load(Ordering::SeqCst), 0);
    let lr = (1 << 62) | (1 << 60) | (0x80 << 48) | 27;
    g.icc_write(0, IccReg::IchLr(0), &hyp, lr);
    assert_eq!(virq.load(Ordering::SeqCst), 1);
    assert_eq!(g.icc_read(0, IccReg::IchElrsr, &hyp), 0xe);

    // With HCR_EL2.IMO and FMO set, the guest's ICC accesses reach the ICV registers.
    assert_eq!(g.icc_read(0, IccReg::Hppir1, &guest), 27);
    assert_eq!(g.icc_read(0, IccReg::Iar1, &guest), 27);
    assert_eq!(virq.load(Ordering::SeqCst), 0);
    assert_eq!(g.icc_read(0, IccReg::IchLr(0), &hyp) >> 62, 2);
    assert_eq!(g.icc_read(0, IccReg::Rpr, &guest), 0x80);
    assert_eq!(g.icc_read(0, IccReg::IchAp1r(0), &hyp), 1 << 16);
    g.icc_write(0, IccReg::Eoir1, &guest, 27);
    assert_eq!(g.icc_read(0, IccReg::IchLr(0), &hyp) >> 62, 0);
    assert_eq!(g.icc_read(0, IccReg::IchAp1r(0), &hyp), 0);
    assert_eq!(g.icc_read(0, IccReg::Iar1, &guest), 1023);
    // The physical interface was not touched.
    assert_eq!(g.icc_read(0, IccReg::Rpr, &NS_EL1), 0xff);

    // Underflow: with UIE and at most one valid list register, the maintenance interrupt is
    // raised, and with it PPI 25.
    assert_eq!(r.rr(0, GICR_ISPENDR0) & (1 << 25), 0);
    g.icc_write(0, IccReg::IchHcr, &hyp, 3);
    assert_eq!(g.icc_read(0, IccReg::IchMisr, &hyp), 2);
    assert_ne!(r.rr(0, GICR_ISPENDR0) & (1 << 25), 0);
    g.icc_write(0, IccReg::IchHcr, &hyp, 1);
    assert_eq!(g.icc_read(0, IccReg::IchMisr, &hyp), 0);
    assert_eq!(r.rr(0, GICR_ISPENDR0) & (1 << 25), 0);

    // TALL1 traps the guest's Group 1 accesses to EL2.
    g.icc_write(0, IccReg::IchHcr, &hyp, 1 | (1 << 12));
    assert_eq!(g.icc_access(0, IccReg::Iar1, &guest, true), IccAccess::TrapEl2);
    assert_eq!(g.icc_access(0, IccReg::Iar0, &guest, true), IccAccess::Ok);
    g.cpuif_reset(0);
    assert_eq!(g.icc_read(0, IccReg::IchHcr, &hyp), 0);
}

// LPIs and the ITS, with the GIC, the ITS and some RAM mapped the way hw/arm/virt.c maps them.

const DIST_BASE: u64 = 0x0800_0000;
const ITS_BASE: u64 = 0x0808_0000;
const REDIST_BASE: u64 = 0x080a_0000;
const RAM: u64 = 0x4000_0000;
const PROPBASE: u64 = RAM;
/// The pending tables of the CPUs, 64K apart.
const PENDBASE: u64 = RAM + 0x1_0000;
const DT_BASE: u64 = RAM + 0x10_0000;
const CT_BASE: u64 = RAM + 0x20_0000;
const CMDQ_BASE: u64 = RAM + 0x30_0000;
const ITT_BASE: u64 = RAM + 0x40_0000;

const GICR_PROPBASER: u64 = 0x70;
const GICR_PENDBASER: u64 = 0x78;
const GITS_CTLR: u64 = 0x0;
const GITS_IIDR: u64 = 0x4;
const GITS_TYPER: u64 = 0x8;
const GITS_CBASER: u64 = 0x80;
const GITS_CWRITER: u64 = 0x88;
const GITS_CREADR: u64 = 0x90;
const GITS_BASER: u64 = 0x100;
const GITS_TRANSLATER: u64 = ITS_CONTROL_SIZE + 0x40;

struct LpiRig {
    _mem: Arc<MemorySystem>,
    space: Arc<AddressSpace>,
    gic: Arc<GicV3>,
    its: Arc<GicV3Its>,
    irq: Vec<Arc<AtomicI32>>,
    /// The commands queued so far.
    queued: u64,
}

impl LpiRig {
    fn new(ncpu: usize) -> LpiRig {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let ram = mem.new_ram("ram", 0x80_0000).unwrap();
        mem.add_subregion(sysmem, RAM, ram).unwrap();
        let space = mem.address_space_init(sysmem, "memory").unwrap();

        let mut p = props((0..ncpu as u64).collect(), false);
        p.has_lpi = true;
        let gic = GicV3::with_sysmem(p, Some(&space)).unwrap();
        let dist = mem.new_io("gicv3_dist", GICV3_DIST_SIZE.into(), gic.dist_ops()).unwrap();
        mem.add_subregion(sysmem, DIST_BASE, dist).unwrap();
        let size = gic.redist_region_size(0);
        let redist = mem.new_io("gicv3_redist_region[0]", size.into(), gic.redist_ops(0)).unwrap();
        mem.add_subregion(sysmem, REDIST_BASE, redist).unwrap();

        let its = GicV3Its::new(&gic).unwrap();
        let ctl = mem.new_io("control", ITS_CONTROL_SIZE.into(), its.control_ops()).unwrap();
        mem.add_subregion(sysmem, ITS_BASE, ctl).unwrap();
        let trans =
            mem.new_io("translation", ITS_TRANS_SIZE.into(), its.translation_ops()).unwrap();
        mem.add_subregion(sysmem, ITS_BASE + ITS_CONTROL_SIZE, trans).unwrap();

        let mut irq = Vec::new();
        for cpu in 0..ncpu {
            let (level, line) = watch();
            gic.cpu_irq(cpu).connect(line);
            irq.push(level);
        }
        LpiRig { _mem: mem, space, gic, its, irq, queued: 0 }
    }

    fn rd(&self, addr: u64, size: u32) -> u64 {
        let (v, r) = self.space.load(addr, size, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok(), "read of {addr:#x} failed");
        v
    }

    fn wr(&self, addr: u64, size: u32, v: u64) {
        let r = self.space.store(addr, size, v, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok(), "write of {addr:#x} failed");
    }

    /// An MSI from the device with requester ID `devid`.
    fn msi(&self, devid: u16, eventid: u32) {
        let attrs = MemTxAttrs::new().with_requester_id(devid);
        let r =
            self.space.store(ITS_BASE + GITS_TRANSLATER, 4, eventid.into(), Endian::Little, attrs);
        assert!(r.is_ok());
    }

    fn irq(&self, cpu: usize) -> i32 {
        self.irq[cpu].load(Ordering::SeqCst)
    }

    fn pending(&self, cpu: u64, intid: u64) -> bool {
        self.rd(PENDBASE + cpu * 0x1_0000 + intid / 8, 1) & (1 << (intid % 8)) != 0
    }

    /// Group 1 on, the CPU interfaces open, LPI 8192 + n at priority 0x80 + 8 * n for n below
    /// 8, and each CPU's LPIs enabled with 16 bits of interrupt ID.
    fn setup_lpis(&self) {
        self.wr(DIST_BASE + GICD_CTLR, 4, 0x12);
        for n in 0..8 {
            self.wr(PROPBASE + n, 1, (0x80 + 8 * n) | 1);
        }
        for cpu in 0..self.gic.num_cpu() {
            let rd = REDIST_BASE + cpu as u64 * GICV3_REDIST_SIZE;
            self.wr(rd + GICR_WAKER, 4, 0);
            self.wr(rd + GICR_PROPBASER, 8, PROPBASE | 0xf);
            self.wr(rd + GICR_PENDBASER, 8, PENDBASE + cpu as u64 * 0x1_0000);
            self.wr(rd + GICR_CTLR, 4, 1);
            self.gic.icc_write(cpu, IccReg::Pmr, &NS_EL1, 0xff);
            self.gic.icc_write(cpu, IccReg::Igrpen1, &NS_EL1, 1);
        }
    }

    /// Flat device and collection tables of one 64K page, a 4K command queue, and the ITS
    /// enabled.
    fn setup_its(&self) {
        let baser0 = self.rd(ITS_BASE + GITS_BASER, 8);
        self.wr(ITS_BASE + GITS_BASER, 8, baser0 | (1 << 63) | DT_BASE);
        let baser1 = self.rd(ITS_BASE + GITS_BASER + 8, 8);
        self.wr(ITS_BASE + GITS_BASER + 8, 8, baser1 | (1 << 63) | CT_BASE);
        self.wr(ITS_BASE + GITS_CBASER, 8, (1 << 63) | CMDQ_BASE);
        self.wr(ITS_BASE + GITS_CWRITER, 8, 0);
        self.wr(ITS_BASE + GITS_CTLR, 4, 1);
    }

    /// Queue one command and run the queue.
    fn cmd(&mut self, pkt: [u64; 4]) {
        let at = CMDQ_BASE + (self.queued % 128) * 32;
        for (i, w) in pkt.iter().enumerate() {
            self.wr(at + i as u64 * 8, 8, *w);
        }
        self.queued += 1;
        self.wr(ITS_BASE + GITS_CWRITER, 8, (self.queued % 128) * 32);
    }

    fn mapd(&mut self, devid: u64, ittaddr: u64) {
        // Five bits of event ID.
        self.cmd([0x08 | (devid << 32), 4, (1 << 63) | ittaddr, 0]);
    }

    fn mapc(&mut self, icid: u64, cpu: u64) {
        self.cmd([0x09, 0, (1 << 63) | (cpu << 16) | icid, 0]);
    }

    fn mapti(&mut self, devid: u64, eventid: u64, intid: u64, icid: u64) {
        self.cmd([0x0a | (devid << 32), eventid | (intid << 32), icid, 0]);
    }
}

#[test]
fn lpi_registers() {
    let r = LpiRig::new(2);
    // GICD_TYPER.LPIS and GICR_TYPER.PLPIS are set, and GICR_CTLR.CES with them.
    assert_eq!(
        r.rd(DIST_BASE + GICD_TYPER, 4),
        (1 << 25) | (1 << 24) | (0xf << 19) | (1 << 17) | 1
    );
    assert_eq!(r.rd(REDIST_BASE + GICR_TYPER, 4), (1 << 24) | 1);
    assert_eq!(r.rd(REDIST_BASE + GICR_CTLR, 4), 2);
    r.wr(REDIST_BASE + GICR_CTLR, 4, 1);
    assert_eq!(r.rd(REDIST_BASE + GICR_CTLR, 4), 3);
    r.wr(REDIST_BASE + GICR_CTLR, 4, 0);
    assert_eq!(r.rd(REDIST_BASE + GICR_CTLR, 4), 2);

    assert_eq!(r.rd(ITS_BASE + GITS_CTLR, 4), 1 << 31);
    assert_eq!(r.rd(ITS_BASE + GITS_IIDR, 4), 0x43b);
    assert_eq!(
        r.rd(ITS_BASE + GITS_TYPER, 8),
        (1 << 36) | (0xf << 32) | (0xf << 13) | (0xf << 8) | 0xb1
    );
    assert_eq!(r.rd(ITS_BASE + 0xffe0, 4), 0x94);
    assert_eq!(r.rd(ITS_BASE + 0xffe8, 4), 0x3b);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER, 8), 0x0107_0000_0000_0200);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER + 8, 8), 0x0407_0000_0000_0200);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER + 16, 8), 0);

    // TYPE and ENTRYSIZE are read only and unimplemented tables ignore writes.
    r.wr(ITS_BASE + GITS_BASER + 4, 4, 0xffff_ffff);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER, 8), 0xf8e0_ffff_0000_0200 | 0x0107_0000_0000_0000);
    r.wr(ITS_BASE + GITS_BASER + 16, 8, u64::MAX);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER + 16, 8), 0);
    // The translation frame reads as zero.
    assert_eq!(r.rd(ITS_BASE + GITS_TRANSLATER, 4), 0);

    // Without has-lpi there is no ITS, and has-lpi needs the memory the tables live in.
    let gic = GicV3::new(props(vec![0], false)).unwrap();
    assert_eq!(GicV3Its::new(&gic).unwrap_err(), "Physical LPI not supported by CPU 0");
    let mut p = props(vec![0], false);
    p.has_lpi = true;
    assert_eq!(GicV3::new(p).unwrap_err(), "Redist-ITS: Guest 'sysmem' reference link not set");
}

#[test]
fn msi_through_the_its_raises_an_lpi() {
    let mut r = LpiRig::new(2);
    r.setup_lpis();
    r.setup_its();
    r.mapd(0x10, ITT_BASE);
    r.mapc(0, 0);
    r.mapc(1, 1);
    r.mapti(0x10, 3, 8195, 0);
    r.mapti(0x10, 5, 8193, 1);
    r.cmd([0x05, 0, 0, 0]);
    assert_eq!(r.rd(ITS_BASE + GITS_CREADR, 8), 6 * 32);
    assert_eq!(r.irq(0), 0);

    r.msi(0x10, 3);
    assert!(r.pending(0, 8195));
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.gic.icc_read(0, IccReg::Hppir1, &NS_EL1), 8195);
    assert_eq!(r.gic.icc_read(0, IccReg::Iar1, &NS_EL1), 8195);
    // Acknowledging an LPI clears its pending bit. LPIs have no active state.
    assert!(!r.pending(0, 8195));
    assert_eq!(r.irq(0), 0);
    r.gic.icc_write(0, IccReg::Eoir1, &NS_EL1, 8195);
    assert_eq!(r.gic.icc_read(0, IccReg::Iar1, &NS_EL1), 1023);

    // Event 5 goes to CPU 1. An unmapped event or device does nothing.
    r.msi(0x10, 5);
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.gic.icc_read(1, IccReg::Iar1, &NS_EL1), 8193);
    r.gic.icc_write(1, IccReg::Eoir1, &NS_EL1, 8193);
    r.msi(0x10, 4);
    r.msi(0x11, 3);
    r.msi(0x10, 40);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(1), 0);

    // A disabled LPI stays pending without being signalled, and enabling it in the
    // configuration table only takes effect with INV or INVALL.
    r.wr(PROPBASE + 3, 1, 0x88);
    r.msi(0x10, 3);
    assert!(r.pending(0, 8195));
    assert_eq!(r.irq(0), 0);
    r.wr(PROPBASE + 3, 1, 0x89);
    assert_eq!(r.irq(0), 0);
    r.cmd([0x0c | (0x10 << 32), 3, 0, 0]);
    assert_eq!(r.irq(0), 1);
    r.wr(PROPBASE + 3, 1, 0x88);
    r.cmd([0x0d, 0, 0, 0]);
    assert_eq!(r.irq(0), 0);
    r.wr(PROPBASE + 3, 1, 0x89);
    r.cmd([0x0d, 0, 0, 0]);
    assert_eq!(r.irq(0), 1);

    // CLEAR takes the pending bit away again, and INT sets it.
    r.cmd([0x04 | (0x10 << 32), 3, 0, 0]);
    assert!(!r.pending(0, 8195));
    assert_eq!(r.irq(0), 0);
    r.cmd([0x03 | (0x10 << 32), 3, 0, 0]);
    assert_eq!(r.irq(0), 1);

    // Turning GITS_CTLR.Enabled off stops translation.
    r.cmd([0x04 | (0x10 << 32), 3, 0, 0]);
    r.wr(ITS_BASE + GITS_CTLR, 4, 0);
    r.msi(0x10, 3);
    assert_eq!(r.irq(0), 0);
}

#[test]
fn lpi_priorities_and_disable() {
    let mut r = LpiRig::new(1);
    r.setup_lpis();
    r.setup_its();
    r.mapd(1, ITT_BASE);
    r.mapc(0, 0);
    for ev in 0..4 {
        r.mapti(1, ev, 8192 + ev, 0);
    }
    // LPI 8192 + n has priority 0x80 + 8 * n, so the lowest number wins.
    r.msi(1, 2);
    r.msi(1, 1);
    r.msi(1, 3);
    assert_eq!(r.gic.icc_read(0, IccReg::Iar1, &NS_EL1), 8193);
    r.gic.icc_write(0, IccReg::Eoir1, &NS_EL1, 8193);
    assert_eq!(r.gic.icc_read(0, IccReg::Hppir1, &NS_EL1), 8194);

    // Clearing EnableLPIs hides them, setting it again rescans the pending table.
    r.wr(REDIST_BASE + GICR_CTLR, 4, 0);
    assert_eq!(r.irq(0), 0);
    r.wr(REDIST_BASE + GICR_CTLR, 4, 1);
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.gic.icc_read(0, IccReg::Hppir1, &NS_EL1), 8194);

    // DISCARD clears the pending state and the mapping.
    r.cmd([0x0f | (1 << 32), 2, 0, 0]);
    assert!(!r.pending(0, 8194));
    assert_eq!(r.gic.icc_read(0, IccReg::Hppir1, &NS_EL1), 8195);
    r.msi(1, 2);
    assert!(!r.pending(0, 8194));
}

#[test]
fn movi_and_movall_move_pending_lpis() {
    let mut r = LpiRig::new(2);
    r.setup_lpis();
    r.setup_its();
    r.mapd(7, ITT_BASE);
    r.mapc(0, 0);
    r.mapc(1, 1);
    r.mapti(7, 0, 8192, 0);
    r.mapti(7, 1, 8193, 0);
    r.msi(7, 0);
    r.msi(7, 1);
    assert_eq!(r.irq(0), 1);

    // MOVI moves the pending state of one LPI and retargets its event.
    r.cmd([0x01 | (7 << 32), 0, 1, 0]);
    assert!(!r.pending(0, 8192));
    assert!(r.pending(1, 8192));
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.gic.icc_read(0, IccReg::Hppir1, &NS_EL1), 8193);

    // MOVALL moves the rest.
    r.cmd([0x0e, 0, 0, 1 << 16]);
    assert!(!r.pending(0, 8193));
    assert!(r.pending(1, 8193));
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.gic.icc_read(1, IccReg::Iar1, &NS_EL1), 8192);
    r.gic.icc_write(1, IccReg::Eoir1, &NS_EL1, 8192);
    assert_eq!(r.gic.icc_read(1, IccReg::Iar1, &NS_EL1), 8193);
    r.gic.icc_write(1, IccReg::Eoir1, &NS_EL1, 8193);

    // The event now goes to CPU 1.
    r.msi(7, 0);
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.irq(1), 1);
}

#[test]
fn command_queue_stalls_and_resets() {
    let mut r = LpiRig::new(1);
    r.setup_lpis();
    // CBASER is read only while the ITS is enabled.
    r.setup_its();
    r.wr(ITS_BASE + GITS_CBASER, 8, 0);
    assert_eq!(r.rd(ITS_BASE + GITS_CBASER, 8), (1 << 63) | CMDQ_BASE);

    // A device table pointing at nothing stalls MAPD.
    r.wr(ITS_BASE + GITS_CTLR, 4, 0);
    let baser0 = r.rd(ITS_BASE + GITS_BASER, 8);
    r.wr(ITS_BASE + GITS_BASER, 8, (baser0 & !0xffff_ffff_f000) | 0x7f_0000_0000);
    r.wr(ITS_BASE + GITS_CTLR, 4, 1);
    r.mapd(1, ITT_BASE);
    assert_eq!(r.rd(ITS_BASE + GITS_CREADR, 8), 1);

    // Without security CREADR is writable, which is how a stall is cleared.
    r.wr(ITS_BASE + GITS_CREADR, 8, 32 | 1);
    assert_eq!(r.rd(ITS_BASE + GITS_CREADR, 8), 32);

    r.its.reset();
    assert_eq!(r.rd(ITS_BASE + GITS_CTLR, 4), 1 << 31);
    assert_eq!(r.rd(ITS_BASE + GITS_CBASER, 8), 0);
    assert_eq!(r.rd(ITS_BASE + GITS_CREADR, 8), 0);
    assert_eq!(r.rd(ITS_BASE + GITS_BASER, 8), 0x0107_0000_0000_0200);
}
