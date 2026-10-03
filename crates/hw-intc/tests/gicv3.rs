// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the emulated GICv3 from hw/intc/arm_gicv3*.c, driven through the
//! distributor and redistributor MMIO and the ICC_* system register entry points.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_intc::gicv3::*;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

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
