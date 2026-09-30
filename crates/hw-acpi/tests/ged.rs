// SPDX-License-Identifier: GPL-2.0-or-later

//! The Generic Event Device mapped the way microvm maps it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_acpi::core::{ACPI_CPU_HOTPLUG_STATUS, ACPI_VMGENID_CHANGE_STATUS};
use ruvm_hw_acpi::ged::*;
use ruvm_hw_core::IrqLine;
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult, MemorySystem};

type Requests = Arc<Mutex<Vec<SystemRequest>>>;

struct Env {
    ged: Arc<AcpiGed>,
    space: Arc<AddressSpace>,
    pulses: Arc<AtomicUsize>,
    reqs: Requests,
    _mem: Arc<MemorySystem>,
}

impl Env {
    fn new() -> Env {
        let ged =
            AcpiGed::new(AcpiGedProps { ged_event: ACPI_GED_PWR_DOWN_EVT, ..Default::default() })
                .unwrap();
        let mem = Arc::new(MemorySystem::new());
        let sys = mem.new_container("system", 1 << 64).unwrap();
        let evt =
            mem.new_io(TYPE_ACPI_GED, u128::from(ACPI_GED_EVT_SEL_LEN), ged.evt_ops()).unwrap();
        mem.add_subregion(sys, MICROVM_GED_MMIO_BASE, evt).unwrap();
        let regs =
            mem.new_io("acpi-ged-regs", u128::from(ACPI_GED_REG_COUNT), ged.regs_ops()).unwrap();
        mem.add_subregion(sys, MICROVM_GED_MMIO_BASE_REGS, regs).unwrap();
        let space = mem.address_space_init(sys, "memory").unwrap();

        let pulses = Arc::new(AtomicUsize::new(0));
        let p = pulses.clone();
        ged.irq().connect(IrqLine::from_fn(move |level| {
            if level != 0 {
                p.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let reqs: Requests = Arc::default();
        let r = reqs.clone();
        ged.set_request_handler(Arc::new(move |req| r.lock().unwrap().push(req)));
        Env { ged, space, pulses, reqs, _mem: mem }
    }

    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        let mut b = [0u8; 8];
        let r = self.space.read(addr, MemTxAttrs::UNSPECIFIED, &mut b[..size]);
        (r == MemTxResult::OK).then(|| u64::from_le_bytes(b))
    }

    fn writeb(&self, addr: u64, v: u8) -> bool {
        self.space.write(addr, MemTxAttrs::UNSPECIFIED, &[v]) == MemTxResult::OK
    }

    fn requests(&self) -> Vec<SystemRequest> {
        std::mem::take(&mut *self.reqs.lock().unwrap())
    }
}

#[test]
fn unsupported_events_are_rejected() {
    assert!(AcpiGed::new(AcpiGedProps { ged_event: 0x40, ..Default::default() }).is_err());
    let all = GED_SUPPORTED_EVENTS.iter().fold(0, |a, e| a | e);
    assert_eq!(all, 0x3f);
    assert!(AcpiGed::new(AcpiGedProps { ged_event: all, ..Default::default() }).is_ok());
    let g = AcpiGed::new(AcpiGedProps { ged_event: 0, pci_hotplug: true }).unwrap();
    assert_eq!(g.ged_event_bitmap(), ACPI_GED_PCI_HOTPLUG_EVT);
}

#[test]
fn power_button_sets_selector_and_pulses_irq() {
    let env = Env::new();
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE, 4), Some(0));

    env.ged.power_down();
    assert_eq!(env.pulses.load(Ordering::SeqCst), 1);
    assert_eq!(env.ged.sel(), ACPI_GED_PWR_DOWN_EVT);

    // Reading the selector returns it and clears it.
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE, 4), Some(u64::from(ACPI_GED_PWR_DOWN_EVT)));
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE, 4), Some(0));

    // Events accumulate until read.
    env.ged.power_down();
    assert!(env.ged.send_event(ACPI_CPU_HOTPLUG_STATUS));
    assert_eq!(env.pulses.load(Ordering::SeqCst), 3);
    assert_eq!(
        env.read(MICROVM_GED_MMIO_BASE, 4),
        Some(u64::from(ACPI_GED_PWR_DOWN_EVT | ACPI_GED_CPU_HOTPLUG_EVT))
    );

    // Events the GED has no bit for are dropped without an interrupt.
    assert!(!env.ged.send_event(ACPI_VMGENID_CHANGE_STATUS));
    assert_eq!(env.pulses.load(Ordering::SeqCst), 3);
    assert_eq!(env.ged.sel(), 0);
}

#[test]
fn selector_takes_only_32_bit_accesses() {
    let env = Env::new();
    env.ged.power_down();
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE, 1), None);
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE, 2), None);
    // The rejected reads did not consume the event.
    assert_eq!(env.ged.sel(), ACPI_GED_PWR_DOWN_EVT);
    // Writes are ignored.
    assert_eq!(
        env.space.write_u32(MICROVM_GED_MMIO_BASE, MemTxAttrs::UNSPECIFIED, 0),
        MemTxResult::OK
    );
    assert_eq!(env.ged.sel(), ACPI_GED_PWR_DOWN_EVT);
}

#[test]
fn reset_register() {
    let env = Env::new();
    let reset = MICROVM_GED_MMIO_BASE_REGS + ACPI_GED_REG_RESET;
    assert!(env.writeb(reset, 0x41));
    assert!(env.requests().is_empty());
    assert!(env.writeb(reset, ACPI_GED_RESET_VALUE));
    assert_eq!(env.requests(), [SystemRequest::Reset]);
    // The registers read as zero. A wider access is split into bytes, as QEMU does.
    assert_eq!(env.read(reset, 1), Some(0));
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE_REGS, 2), Some(0));
}

#[test]
fn sleep_control() {
    let env = Env::new();
    let ctl = MICROVM_GED_MMIO_BASE_REGS + ACPI_GED_REG_SLEEP_CTL;
    let s5 = ACPI_GED_SLP_TYP_S5 << ACPI_GED_SLP_TYP_POS;

    // SLP_TYP 5 without SLP_EN does nothing.
    assert!(env.writeb(ctl, s5));
    assert!(env.requests().is_empty());
    // Other sleep types are ignored.
    assert!(env.writeb(ctl, (3 << ACPI_GED_SLP_TYP_POS) | ACPI_GED_SLP_EN));
    assert!(env.requests().is_empty());

    assert!(env.writeb(ctl, s5 | ACPI_GED_SLP_EN));
    assert_eq!(env.requests(), [SystemRequest::Shutdown]);

    // Sleep status writes are accepted and ignored.
    assert!(env.writeb(MICROVM_GED_MMIO_BASE_REGS + ACPI_GED_REG_SLEEP_STS, 0xff));
    assert_eq!(env.read(MICROVM_GED_MMIO_BASE_REGS + ACPI_GED_REG_SLEEP_STS, 1), Some(0));
    assert!(env.requests().is_empty());
}
