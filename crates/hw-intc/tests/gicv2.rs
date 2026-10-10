// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the emulated GICv2 from hw/intc/arm_gic.c and the GICv2m frame from
//! hw/intc/arm_gicv2m.c, driven through their MMIO regions.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_intc::gicv2::*;
use ruvm_hw_intc::gicv2m::GicV2m;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

const GICD_CTLR: u64 = 0x0;
const GICD_TYPER: u64 = 0x4;
const GICD_IIDR: u64 = 0x8;
const GICD_IGROUPR: u64 = 0x80;
const GICD_ISENABLER: u64 = 0x100;
const GICD_ICENABLER: u64 = 0x180;
const GICD_ISPENDR: u64 = 0x200;
const GICD_ICPENDR: u64 = 0x280;
const GICD_ISACTIVER: u64 = 0x300;
const GICD_IPRIORITYR: u64 = 0x400;
const GICD_ITARGETSR: u64 = 0x800;
const GICD_ICFGR: u64 = 0xc00;
const GICD_SGIR: u64 = 0xf00;
const GICD_SPENDSGIR: u64 = 0xf20;
const GICD_PIDR2: u64 = 0xfe8;

const GICC_CTLR: u64 = 0x0;
const GICC_PMR: u64 = 0x4;
const GICC_IAR: u64 = 0xc;
const GICC_EOIR: u64 = 0x10;
const GICC_RPR: u64 = 0x14;
const GICC_HPPIR: u64 = 0x18;
const GICC_APR0: u64 = 0xd0;
const GICC_IIDR: u64 = 0xfc;
const GICC_DIR: u64 = 0x1000;

struct Rig {
    gic: Arc<GicV2>,
    dist: Arc<dyn MmioOps>,
    cpu: Arc<dyn MmioOps>,
    /// The CPU the accesses come from.
    current: Arc<AtomicUsize>,
    irq: Vec<Arc<AtomicI32>>,
    fiq: Vec<Arc<AtomicI32>>,
}

fn watch() -> (Arc<AtomicI32>, IrqLine) {
    let level = Arc::new(AtomicI32::new(0));
    let l = level.clone();
    (level, IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)))
}

fn rig(num_cpu: usize, num_irq: u32) -> Rig {
    let gic = GicV2::new(GicV2Props { num_cpu, num_irq, ..GicV2Props::default() }).unwrap();
    let current = Arc::new(AtomicUsize::new(0));
    let c = current.clone();
    gic.set_current_cpu_fn(Some(Arc::new(move || Some(c.load(Ordering::SeqCst)))));
    let mut irq = Vec::new();
    let mut fiq = Vec::new();
    for cpu in 0..num_cpu {
        let (level, line) = watch();
        gic.cpu_irq(cpu).connect(line);
        irq.push(level);
        let (level, line) = watch();
        gic.cpu_fiq(cpu).connect(line);
        fiq.push(level);
    }
    Rig { dist: gic.dist_ops(), cpu: gic.cpu_ops(), gic, current, irq, fiq }
}

fn cx() -> AccessCtx {
    AccessCtx::new(MemTxAttrs::new())
}

impl Rig {
    fn on(&self, cpu: usize) -> &Self {
        self.current.store(cpu, Ordering::SeqCst);
        self
    }

    fn dr(&self, off: u64, size: u32) -> u64 {
        self.dist.read(&cx(), off, AccessSize::new(size).unwrap()).unwrap()
    }

    fn dw(&self, off: u64, size: u32, v: u64) {
        self.dist.write(&cx(), off, AccessSize::new(size).unwrap(), v).unwrap();
    }

    fn cr(&self, off: u64) -> u64 {
        self.cpu.read(&cx(), off, AccessSize::B4).unwrap()
    }

    fn cw(&self, off: u64, v: u64) {
        self.cpu.write(&cx(), off, AccessSize::B4, v).unwrap();
    }

    fn irq(&self, cpu: usize) -> i32 {
        self.irq[cpu].load(Ordering::SeqCst)
    }

    fn fiq(&self, cpu: usize) -> i32 {
        self.fiq[cpu].load(Ordering::SeqCst)
    }

    /// Both groups on in the distributor, and `ctlr` with the mask fully open on each CPU.
    fn enable(&self, ctlr: u64) {
        self.dw(GICD_CTLR, 4, 3);
        for cpu in 0..self.gic.num_cpu() {
            self.on(cpu).cw(GICC_CTLR, ctlr);
            self.on(cpu).cw(GICC_PMR, 0xff);
        }
        self.on(0);
    }
}

#[test]
fn realize_checks_match_qemu() {
    let err = |p: GicV2Props| GicV2::new(p).unwrap_err();
    let base = GicV2Props::default();
    assert_eq!(
        err(GicV2Props { num_cpu: 9, ..base.clone() }),
        "requested 9 CPUs exceeds GIC maximum 8"
    );
    assert_eq!(
        err(GicV2Props { num_irq: 1024, ..base.clone() }),
        "requested 1024 interrupt lines exceeds GIC maximum 1020"
    );
    assert_eq!(
        err(GicV2Props { num_irq: 48, ..base.clone() }),
        "48 interrupt lines unsupported: not divisible by 32"
    );
    assert_eq!(
        err(GicV2Props { n_prio_bits: 3, ..base.clone() }),
        "num-priority-bits cannot be greater than 8 or less than 4"
    );
    assert!(err(GicV2Props { security_extn: true, ..base.clone() }).contains("not supported"));
    assert!(err(GicV2Props { revision: 1, ..base }).contains("not supported"));
}

#[test]
fn id_registers() {
    let r = rig(4, 288);
    // ITLinesNumber 8 and CPUNumber 3.
    assert_eq!(r.dr(GICD_TYPER, 4), 8 | (3 << 5));
    assert_eq!(r.dr(GICD_IIDR, 4), 0x43b);
    assert_eq!(r.dr(GICD_PIDR2, 4), 0x2b);
    assert_eq!(r.dr(0xff0, 4), 0x0d);
    assert_eq!(r.dr(0xffc, 4), 0xb1);
    assert_eq!(r.cr(GICC_IIDR), 0x2043b);
    // Reserved distributor space reads as zero.
    assert_eq!(r.dr(0x40, 4), 0);
}

#[test]
fn level_spi_round_trip() {
    let r = rig(2, 64);
    r.enable(1);
    r.dw(GICD_ISENABLER + 4, 4, 1 << 8);
    r.dw(GICD_IPRIORITYR + 40, 1, 0x80);
    r.dw(GICD_ITARGETSR + 40, 1, 0b10);
    let line = r.gic.spi(8);
    line.raise();
    assert_eq!((r.irq(0), r.irq(1)), (0, 1));
    assert_eq!(r.on(1).cr(GICC_HPPIR), 40);
    // Not for CPU 0.
    assert_eq!(r.on(0).cr(GICC_IAR), 1023);

    assert_eq!(r.on(1).cr(GICC_IAR), 40);
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.cr(GICC_RPR), 0x80);
    assert_eq!(r.cr(GICC_APR0), 0);
    assert_eq!(r.cr(GICC_APR0 + 8), 1);
    // Still high after the EOI, so it is taken again.
    r.cw(GICC_EOIR, 40);
    assert_eq!(r.cr(GICC_RPR), 0xff);
    assert_eq!(r.irq(1), 1);
    line.lower();
    assert_eq!(r.irq(1), 0);
    assert_eq!(r.cr(GICC_IAR), 1023);
}

#[test]
fn edge_spi_latches_and_moves_with_its_target() {
    let r = rig(2, 64);
    r.enable(1);
    r.dw(GICD_ICFGR + 8, 4, 2 << 16);
    r.dw(GICD_ISENABLER + 4, 4, 1 << 8);
    r.dw(GICD_ITARGETSR + 40, 1, 0b01);
    r.gic.spi(8).pulse();
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.dr(GICD_ISPENDR + 4, 4), 1 << 8);
    // Retargeting moves the pending state.
    r.dw(GICD_ITARGETSR + 40, 1, 0b10);
    assert_eq!((r.irq(0), r.irq(1)), (0, 1));
    r.dw(GICD_ICPENDR + 4, 4, 1 << 8);
    assert_eq!(r.irq(1), 0);
    // A write to ISPENDR pends it on its targets.
    r.dw(GICD_ISPENDR + 4, 4, 1 << 8);
    assert_eq!(r.irq(1), 1);
    r.dw(GICD_ICENABLER + 4, 4, 1 << 8);
    assert_eq!(r.irq(1), 0);
}

#[test]
fn group0_goes_to_fiq_with_fiqen() {
    let r = rig(1, 64);
    // EnableGrp0, EnableGrp1, AckCtl and FIQEn.
    r.enable(0xf);
    r.dw(GICD_ISENABLER + 4, 4, 0b11);
    r.dw(GICD_IGROUPR + 4, 4, 0b10);
    r.dw(GICD_IPRIORITYR + 32, 2, 0x4060);
    r.gic.spi(0).raise();
    assert_eq!((r.irq(0), r.fiq(0)), (0, 1));
    r.gic.spi(0).lower();
    r.gic.spi(1).raise();
    assert_eq!((r.irq(0), r.fiq(0)), (1, 0));
}

#[test]
fn group1_needs_ackctl() {
    let r = rig(1, 64);
    r.enable(3);
    r.dw(GICD_ISENABLER + 4, 4, 1);
    r.dw(GICD_IGROUPR + 4, 4, 1);
    r.gic.spi(0).raise();
    assert_eq!(r.cr(GICC_HPPIR), 1022);
    assert_eq!(r.cr(GICC_IAR), 1022);
    r.cw(GICC_CTLR, 7);
    assert_eq!(r.cr(GICC_IAR), 32);
}

#[test]
fn sgis_carry_their_source() {
    let r = rig(4, 64);
    r.enable(1);
    // CPU 2 sends SGI 5 to CPUs 0 and 3.
    r.on(2).dw(GICD_SGIR, 4, 5 | (0b1001 << 16));
    assert_eq!((r.irq(0), r.irq(1), r.irq(2), r.irq(3)), (1, 0, 0, 1));
    assert_eq!(r.on(3).dr(GICD_SPENDSGIR + 5, 1), 0b100);
    assert_eq!(r.on(3).cr(GICC_IAR), 5 | (2 << 10));
    r.cw(GICC_EOIR, 5 | (2 << 10));
    assert_eq!(r.irq(3), 0);
    // "All but me" from CPU 0.
    r.on(0).dw(GICD_SGIR, 4, 1 | (1 << 24));
    assert_eq!(r.on(1).cr(GICC_IAR), 1);
    // SGIs cannot be disabled.
    r.dw(GICD_ICENABLER, 4, 0xffff);
    assert_eq!(r.dr(GICD_ISENABLER, 4) & 0xffff, 0xffff);
}

#[test]
fn ppis_are_banked() {
    let r = rig(2, 64);
    r.enable(1);
    r.on(1).dw(GICD_ISENABLER, 4, 1 << 27);
    r.gic.ppi(0, 27).raise();
    assert_eq!(r.irq(0), 0);
    r.gic.ppi(1, 27).raise();
    assert_eq!(r.irq(1), 1);
    assert_eq!(r.on(1).dr(GICD_ISPENDR, 4), 1 << 27);
    assert_eq!(r.on(0).dr(GICD_ISENABLER, 4), 0xffff);
    // The targets of private interrupts read as the reading CPU.
    assert_eq!(r.on(1).dr(GICD_ITARGETSR + 27, 1), 0b10);
}

#[test]
fn split_eoi_needs_dir() {
    let r = rig(1, 64);
    // EnableGrp0 and EOImodeNS.
    r.enable(0x201);
    r.dw(GICD_ISENABLER + 4, 4, 1);
    r.gic.spi(0).raise();
    assert_eq!(r.cr(GICC_IAR), 32);
    r.cw(GICC_EOIR, 32);
    assert_eq!(r.cr(GICC_RPR), 0xff);
    // Dropped but still active.
    assert_eq!(r.dr(GICD_ISACTIVER + 4, 4), 1);
    assert_eq!(r.irq(0), 0);
    r.cw(GICC_DIR, 32);
    assert_eq!(r.dr(GICD_ISACTIVER + 4, 4), 0);
    assert_eq!(r.irq(0), 1);
}

#[test]
fn preemption_by_a_higher_priority() {
    let r = rig(1, 64);
    r.enable(1);
    r.dw(GICD_ISENABLER + 4, 4, 0b11);
    r.dw(GICD_IPRIORITYR + 32, 2, 0x40a0);
    r.gic.spi(0).raise();
    assert_eq!(r.cr(GICC_IAR), 32);
    r.gic.spi(1).raise();
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.cr(GICC_IAR), 33);
    assert_eq!(r.cr(GICC_RPR), 0x40);
    r.cw(GICC_EOIR, 33);
    assert_eq!(r.cr(GICC_RPR), 0xa0);
}

#[test]
fn uniprocessor_targets_are_raz_wi() {
    let r = rig(1, 64);
    r.dw(GICD_ITARGETSR + 40, 1, 0xff);
    assert_eq!(r.dr(GICD_ITARGETSR + 40, 1), 0);
    // But every SPI still reaches CPU 0.
    r.enable(1);
    r.dw(GICD_ISENABLER + 4, 4, 1 << 8);
    r.gic.spi(8).raise();
    assert_eq!(r.irq(0), 1);
}

#[test]
fn priority_bits_and_access_sizes() {
    let gic =
        GicV2::new(GicV2Props { num_irq: 64, n_prio_bits: 5, ..GicV2Props::default() }).unwrap();
    let dist = gic.dist_ops();
    dist.write(&cx(), GICD_IPRIORITYR + 32, AccessSize::B4, 0xffff_ffff).unwrap();
    assert_eq!(dist.read(&cx(), GICD_IPRIORITYR + 32, AccessSize::B4).unwrap(), 0xf8f8_f8f8);
    assert_eq!(dist.read(&cx(), GICD_IPRIORITYR + 34, AccessSize::B2).unwrap(), 0xf8f8);
    // SGIs are always edge triggered.
    dist.write(&cx(), GICD_ICFGR, AccessSize::B4, 0).unwrap();
    assert_eq!(dist.read(&cx(), GICD_ICFGR, AccessSize::B4).unwrap(), 0xaaaa_aaaa);
}

#[test]
fn reset_lowers_the_pins() {
    let r = rig(1, 64);
    r.enable(1);
    r.dw(GICD_ISENABLER + 4, 4, 1);
    r.dw(GICD_ICFGR + 8, 4, 2);
    r.gic.spi(0).pulse();
    assert_eq!(r.irq(0), 1);
    r.gic.reset();
    assert_eq!(r.irq(0), 0);
    assert_eq!(r.dr(GICD_CTLR, 4), 0);
    assert_eq!(r.cr(GICC_RPR), 0xff);
}

#[test]
fn gicv2m_pulses_its_spis() {
    let r = rig(1, 288);
    r.enable(1);
    r.dw(GICD_ISENABLER + 8, 4, 0xffff_ffff);
    r.dw(GICD_ICFGR + 0x14, 4, 0xaaaa_aaaa);
    let gic = r.gic.clone();
    let v2m = GicV2m::new(48, 64, |n| gic.spi(n)).unwrap();
    let rd = |off| v2m.read(&cx(), off, AccessSize::B4).unwrap();
    assert_eq!(rd(0x8), (80 << 16) | 64);
    assert_eq!(rd(0xfcc), 0x51 << 20);
    assert_eq!(v2m.read(&cx(), 0x8, AccessSize::B2).unwrap(), 0);
    // Below the frame's range, ignored.
    v2m.write(&cx(), 0x40, AccessSize::B4, 79).unwrap();
    assert_eq!(r.irq(0), 0);
    v2m.write(&cx(), 0x40, AccessSize::B4, 80).unwrap();
    assert_eq!(r.irq(0), 1);
    assert_eq!(r.cr(GICC_IAR), 80);
    // A byte write is ignored.
    v2m.write(&cx(), 0x40, AccessSize::B1, 81).unwrap();
    assert_eq!(r.cr(GICC_HPPIR), 1023);

    let err = |base, num| GicV2m::new(base, num, |n| gic.spi(n)).unwrap_err();
    assert_eq!(err(0, 129), "requested 129 SPIs exceeds GICv2m frame maximum 128");
    assert_eq!(err(960, 64), "requested base SPI 992+64 exceeds max. number 1020");
}
