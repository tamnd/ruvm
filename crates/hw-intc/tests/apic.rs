// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the local APIC from hw/intc/apic.c: the register window, IPIs, MSIs, the timer, the
//! priority rules and the APIC base transitions, with a mock CPU that records its interrupt
//! requests.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_intc::apic::*;
use ruvm_hw_intc::ioapic::IoApics;
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

#[derive(Default)]
struct MockCpu {
    is_self: AtomicBool,
    x2apic: bool,
    apic_feature: AtomicBool,
    /// The requests raised (true) and cleared (false), in order.
    log: Mutex<Vec<(bool, CpuIrq)>>,
}

impl MockCpu {
    fn take(&self) -> Vec<(bool, CpuIrq)> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl ApicCpu for MockCpu {
    fn cpu_interrupt(&self, irq: CpuIrq) {
        self.log.lock().unwrap().push((true, irq));
    }

    fn cpu_reset_interrupt(&self, irq: CpuIrq) {
        self.log.lock().unwrap().push((false, irq));
    }

    fn is_self(&self) -> bool {
        self.is_self.load(Ordering::Relaxed)
    }

    fn has_x2apic(&self) -> bool {
        self.x2apic
    }

    fn set_apic_feature(&self, on: bool) {
        self.apic_feature.store(on, Ordering::Relaxed);
    }
}

struct Rig {
    clock: Arc<Clock>,
    bus: Arc<ApicBus>,
    mmio: Arc<ApicMmio>,
    apics: Vec<Arc<Apic>>,
    cpus: Vec<Arc<MockCpu>>,
}

fn rig_x2(n: u32, x2apic: bool) -> Rig {
    let clock = Clock::manual(ClockType::Virtual);
    let bus = ApicBus::new(n, None, IoApics::new());
    let mut apics = Vec::new();
    let mut cpus = Vec::new();
    for i in 0..n {
        let cpu = Arc::new(MockCpu { x2apic, ..MockCpu::default() });
        apics.push(Apic::realize(&bus, &clock, i, i == 0, cpu.clone()).unwrap());
        cpus.push(cpu);
    }
    let mmio = bus.mmio();
    Rig { clock, bus, mmio, apics, cpus }
}

fn rig(n: u32) -> Rig {
    rig_x2(n, false)
}

impl Rig {
    /// Runs as vCPU `i`: its APIC is the current one and it is on its own thread.
    fn on(&self, i: usize) {
        for (j, c) in self.cpus.iter().enumerate() {
            c.is_self.store(i == j, Ordering::Relaxed);
        }
        set_current_apic(Some(self.apics[i].clone()));
    }

    fn read(&self, reg: u64) -> u32 {
        let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
        self.mmio.read(&cx, reg, AccessSize::B4).unwrap() as u32
    }

    fn write(&self, reg: u64, v: u32) {
        let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
        self.mmio.write(&cx, reg, AccessSize::B4, u64::from(v)).unwrap();
    }

    /// Software enables the current APIC.
    fn enable(&self) {
        self.write(0xf0, APIC_SV_ENABLE | 0xff);
    }
}

#[test]
fn reset_registers() {
    let r = rig(2);
    r.on(1);
    assert_eq!(r.read(0x20), 1 << 24);
    assert_eq!(r.read(0x30), 0x0005_0014);
    assert_eq!(r.read(0xe0), 0xffff_ffff);
    assert_eq!(r.read(0xf0), 0xff);
    assert_eq!(r.read(0x320), APIC_LVT_MASKED);
    assert_eq!(r.apics[0].apic_base(), 0xfee0_0900);
    assert_eq!(r.apics[1].apic_base(), 0xfee0_0800);
    // Narrow reads give 0, an unknown register reads 0 and sets the illegal address error.
    let cx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);
    assert_eq!(r.mmio.read(&cx, 0x20, AccessSize::B2).unwrap(), 0);
    assert_eq!(r.read(0x40), 0);
    assert_eq!(r.read(0x280), APIC_ESR_ILLEGAL_ADDRESS);
    // No current APIC reads as -1.
    set_current_apic(None);
    assert_eq!(r.mmio.read(&cx, 0x20, AccessSize::B4).unwrap(), u64::MAX);
}

#[test]
fn fixed_ipi_and_eoi() {
    let r = rig(2);
    r.on(1);
    r.enable();
    r.on(0);
    r.enable();
    r.cpus[1].take();
    // Physical destination 1, fixed, vector 0x41.
    r.write(0x310, 1 << 24);
    r.write(0x300, 0x41);
    // The target runs elsewhere, so it is asked to poll.
    assert_eq!(r.cpus[1].take(), [(true, CpuIrq::Poll)]);
    assert_eq!(r.apics[1].get_interrupt(), 0x41);
    r.on(1);
    assert_eq!(r.read(0x200 + 0x20 * 2), 0);
    assert_eq!(r.read(0x100 + 0x10 * 2), 1 << 1);
    // EOI clears the in service bit.
    r.write(0xb0, 0);
    assert_eq!(r.read(0x100 + 0x10 * 2), 0);
    assert_eq!(r.apics[1].get_interrupt(), -1);
}

#[test]
fn poll_raises_hard() {
    let r = rig(1);
    r.on(0);
    r.enable();
    r.cpus[0].take();
    r.apics[0].set_irq(0x50, APIC_TRIGGER_EDGE);
    assert_eq!(r.cpus[0].take(), [(true, CpuIrq::Hard)]);
    assert_eq!(r.apics[0].get_interrupt(), 0x50);
    // Nothing left: the request is dropped.
    assert_eq!(r.cpus[0].take(), [(false, CpuIrq::Hard)]);
}

#[test]
fn tpr_masks_lower_classes() {
    let r = rig(1);
    r.on(0);
    r.enable();
    r.apics[0].set_tpr(4);
    assert_eq!(r.read(0x80), 0x40);
    assert_eq!(r.apics[0].tpr(), 4);
    r.apics[0].set_irq(0x31, APIC_TRIGGER_EDGE);
    // Masked by the priority: the spurious vector comes back and the IRR stays set.
    assert_eq!(r.apics[0].get_interrupt(), 0xff);
    assert_eq!(r.read(0x200 + 0x10), 1 << 17);
    r.apics[0].set_tpr(0);
    assert_eq!(r.apics[0].get_interrupt(), 0x31);
    // In service 0x31 raises the processor priority to 0x30.
    assert_eq!(r.read(0xa0), 0x30);
}

#[test]
fn software_disabled_takes_nothing() {
    let r = rig(1);
    r.on(0);
    r.apics[0].set_irq(0x50, APIC_TRIGGER_EDGE);
    assert_eq!(r.apics[0].get_interrupt(), -1);
}

#[test]
fn msi_to_destination() {
    let r = rig(2);
    r.on(1);
    r.enable();
    r.cpus[1].take();
    r.on(0);
    // Physical destination 1, vector 0x62, through the MSI window.
    r.write(0x1000, 0x62);
    assert_eq!(r.cpus[1].take(), [(true, CpuIrq::Poll)]);
    assert!(r.cpus[0].take().is_empty());
    r.on(1);
    assert_eq!(r.apics[1].get_interrupt(), 0x62);
    // The same through ApicBus::send_msi, logical flat to the APIC with LDR bit 1.
    r.write(0xb0, 0);
    r.write(0xd0, 2 << 24);
    r.bus.send_msi(0xfee0_2000 | (1 << 2), 0x63);
    assert_eq!(r.apics[1].get_interrupt(), 0x63);
}

#[test]
fn init_and_sipi() {
    let r = rig(2);
    r.on(0);
    r.cpus[1].take();
    // INIT, level assert, to all but self.
    r.write(0x300, (3 << 18) | (1 << 14) | ((APIC_DM_INIT as u32) << 8));
    assert_eq!(r.cpus[1].take(), [(true, CpuIrq::Init)]);
    assert!(r.cpus[0].take().is_empty());
    // The AP waits for a SIPI after reset, the BSP does not.
    assert_eq!(r.apics[0].sipi(), None);
    r.write(0x300, (3 << 18) | ((APIC_DM_SIPI as u32) << 8) | 0x9a);
    assert_eq!(r.cpus[1].take(), [(true, CpuIrq::Sipi)]);
    assert_eq!(r.apics[1].sipi(), Some(0x9a));
    // Taken once.
    assert_eq!(r.apics[1].sipi(), None);
    r.apics[1].init_reset();
    assert_eq!(r.apics[1].sipi(), Some(0x9a));
}

#[test]
fn nmi_and_lint() {
    let r = rig(1);
    r.on(0);
    r.cpus[0].take();
    // LINT1 is masked at reset.
    r.apics[0].deliver_nmi();
    assert!(r.cpus[0].take().is_empty());
    r.write(0x360, (APIC_DM_NMI as u32) << 8);
    r.apics[0].deliver_nmi();
    assert_eq!(r.cpus[0].take(), [(true, CpuIrq::Nmi)]);
    // Without an 8259 nothing goes through LINT0.
    assert!(!r.apics[0].accept_pic_intr());
}

#[test]
fn one_shot_timer() {
    let r = rig(1);
    r.on(0);
    r.enable();
    r.cpus[0].take();
    // Vector 0x40, one-shot, 100 ticks. Reset leaves a shift of 0, so it counts every
    // nanosecond until the divide register is written, as in QEMU.
    r.write(0x320, 0x40);
    r.write(0x380, 100);
    r.clock.advance_to(50);
    assert_eq!(r.read(0x390), 50);
    r.clock.advance_to(100);
    assert!(r.cpus[0].take().is_empty());
    r.clock.advance_to(101);
    assert_eq!(r.cpus[0].take(), [(true, CpuIrq::Hard)]);
    assert_eq!(r.read(0x390), 0);
    assert_eq!(r.apics[0].get_interrupt(), 0x40);
    r.clock.advance_to(1000);
    assert!(r.cpus[0].take().iter().all(|(on, _)| !on));
}

#[test]
fn periodic_timer() {
    let r = rig(1);
    r.on(0);
    r.enable();
    // Divide by 1, periodic.
    r.write(0x3e0, 0xb);
    r.write(0x320, APIC_LVT_TIMER_PERIODIC | 0x40);
    r.write(0x380, 9);
    r.cpus[0].take();
    for n in 1..=3 {
        r.clock.advance_to(10 * n);
        assert_eq!(r.cpus[0].take(), [(true, CpuIrq::Hard)]);
        assert_eq!(r.apics[0].get_interrupt(), 0x40);
        r.write(0xb0, 0);
        r.cpus[0].take();
    }
    // Masking stops it.
    r.write(0x320, APIC_LVT_MASKED | APIC_LVT_TIMER_PERIODIC | 0x40);
    r.clock.advance_to(100);
    assert!(r.cpus[0].take().is_empty());
}

#[test]
fn apic_base_transitions() {
    let r = rig(1);
    let a = &r.apics[0];
    // x2APIC without the CPU feature.
    assert!(!a.set_base(0xfee0_0000 | MSR_IA32_APICBASE_ENABLE | MSR_IA32_APICBASE_EXTD));
    // Disabling clears the CPUID bit and software enable.
    r.cpus[0].apic_feature.store(true, Ordering::Relaxed);
    assert!(a.set_base(0xfee0_0000));
    assert!(!a.is_enabled());
    assert!(!r.cpus[0].apic_feature.load(Ordering::Relaxed));
    r.on(0);
    assert_eq!(r.read(0x20), 0xffff_ffff);
    assert!(a.set_base(0xfee0_0000 | MSR_IA32_APICBASE_ENABLE));
    assert!(r.cpus[0].apic_feature.load(Ordering::Relaxed));
    // The BSP bit is kept.
    assert_eq!(a.apic_base(), 0xfee0_0900);
    // The x2APIC MSRs fault in xAPIC mode.
    assert_eq!(a.msr_read(0x02), None);
    assert!(!a.msr_write(0x08, 0));
}

#[test]
fn x2apic_mode() {
    let r = rig_x2(2, true);
    let a = &r.apics[1];
    let en = 0xfee0_0000 | MSR_IA32_APICBASE_ENABLE;
    assert!(a.set_base(en | MSR_IA32_APICBASE_EXTD));
    // Back to xAPIC directly is refused.
    assert!(!a.set_base(en));
    assert_eq!(a.msr_read(0x02), Some(1));
    assert_eq!(a.msr_read(0x0d), Some((1 << 1) as u64));
    assert_eq!(a.msr_read(0x0e), None);
    // The xAPIC window reads as all ones.
    r.on(1);
    assert_eq!(r.read(0x30), 0xffff_ffff);
    assert!(a.msr_write(0x0f, u64::from(APIC_SV_ENABLE | 0xff)));
    r.cpus[1].take();
    // Self IPI.
    assert!(a.msr_write(0x3f, 0x77));
    assert_eq!(r.cpus[1].take(), [(true, CpuIrq::Hard)]);
    assert_eq!(a.get_interrupt(), 0x77);
    // The xAPIC only ID register write faults.
    assert!(!a.msr_write(0x02, 5));
}

#[test]
fn realize_needs_x2apic_for_high_ids() {
    let clock = Clock::manual(ClockType::Virtual);
    let bus = ApicBus::new(300, None, IoApics::new());
    let e = Apic::realize(&bus, &clock, 260, false, Arc::new(MockCpu::default())).unwrap_err();
    assert_eq!(e.message(), "APIC ID 260 requires x2APIC feature in CPU");
    assert_eq!(e.hint_text(), Some("Try x2apic=on in -cpu.\n"));
    let cpu = Arc::new(MockCpu { x2apic: true, ..MockCpu::default() });
    assert!(Apic::realize(&bus, &clock, 260, false, cpu).is_ok());
    assert!(bus.apic(260).is_some());
}

#[test]
fn vmstate_round_trip() {
    let r = rig(2);
    r.on(0);
    r.enable();
    r.write(0x80, 0x20);
    // Divide by 1, one-shot, 100 ticks from time 10.
    r.write(0x3e0, 0xb);
    r.write(0x320, 0x40);
    r.clock.advance_to(10);
    r.write(0x380, 100);
    r.clock.advance_to(30);
    let v = r.apics[0].vmstate_save();
    assert_eq!(v.apicbase, 0xfee0_0900);
    assert_eq!((v.tpr, v.spurious_vec, v.count_shift), (0x20, APIC_SV_ENABLE | 0xff, 0));
    assert_eq!((v.initial_count, v.initial_count_load_time), (100, 10));
    assert_eq!((v.next_time, v.timer_expiry), (111, 111));
    assert_eq!(v.wait_for_sipi, 0);
    // The AP waits for its SIPI, so its apic_sipi subsection goes out.
    assert_eq!(r.apics[1].vmstate_save().wait_for_sipi, 1);

    let d = rig(2);
    d.clock.advance_to(30);
    d.apics[0].vmstate_load(&v).unwrap();
    assert_eq!(d.apics[0].vmstate_save(), v);
    // Loaded from another thread: the CPU is asked to poll its APIC.
    assert_eq!(d.cpus[0].take(), [(true, CpuIrq::Poll)]);
    d.on(0);
    assert_eq!(d.read(0x390), 80);
    d.clock.advance_to(110);
    assert!(d.cpus[0].take().is_empty());
    d.clock.advance_to(111);
    assert_eq!(d.cpus[0].take(), [(true, CpuIrq::Hard)]);
    assert_eq!(d.apics[0].get_interrupt(), 0x40);

    // A stopped timer stays stopped, and a shift divide_conf cannot give is refused.
    let mut w = v.clone();
    w.timer_expiry = -1;
    d.apics[0].vmstate_load(&w).unwrap();
    d.clock.advance_to(1000);
    assert!(d.cpus[0].take().iter().all(|(on, _)| !on));
    w.count_shift = 40;
    assert!(d.apics[0].vmstate_load(&w).is_err());
}
