// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the ICH9 LPC bridge on a q35 root bus, driven through 0xcf8/0xcfc, the I/O ports
//! and system memory the way firmware and QEMU's lpc-ich9-test.c and tco-test.c drive it.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_acpi::SystemRequest;
use ruvm_hw_acpi::core::{ACPI_BITMASK_POWER_BUTTON_ENABLE, ACPI_BITMASK_SCI_ENABLE};
use ruvm_hw_acpi::ich9::{ICH9_PMIO_SMI_EN_APMC_EN, Ich9PmProps};
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_pci::q35::{Q35Config, Q35PciHost};
use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::{PciDevice, PciDeviceInfo};
use ruvm_machine_x86::ich9_lpc::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem, RegionId};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MIB: u64 = 1 << 20;
const LPC: u8 = pci_devfn(ICH9_LPC_DEV, ICH9_LPC_FUNC);
/// What tco-test.c uses.
const PM_IO_BASE_ADDR: u64 = 0xb000;
const RCBA_BASE_ADDR: u64 = 0xfed1_c000;

struct Machine {
    mem: Arc<MemorySystem>,
    io: RegionId,
    mem_as: Arc<AddressSpace>,
    io_as: Arc<AddressSpace>,
    clock: Arc<Clock>,
    q35: Q35PciHost,
    lpc: Ich9Lpc,
    gsi: Arc<Mutex<[i32; 24]>>,
    requests: Arc<Mutex<Vec<SystemRequest>>>,
    smis: Arc<Mutex<Vec<SmiTarget>>>,
    notifications: Arc<AtomicUsize>,
}

impl Machine {
    fn new() -> Machine {
        Machine::with(|_| {})
    }

    fn with(tweak: impl FnOnce(&mut Ich9LpcConfig)) -> Machine {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let pci = mem.new_container("pci", 1 << 64).unwrap();
        let ram = mem.new_ram("pc.ram", 128 * MIB).unwrap();
        let ram_below_4g = mem.new_alias("ram-below-4g", ram, 0, u128::from(128 * MIB)).unwrap();
        mem.add_subregion(sysmem, 0, ram_below_4g).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();

        let mut cfg = Q35Config::new(ram, pci, sysmem, io);
        cfg.below_4g_mem_size = 128 * MIB;
        let q35 = Q35PciHost::new(Arc::clone(&mem), cfg).unwrap();

        let gsi = Arc::new(Mutex::new([0i32; 24]));
        let lines: Vec<IrqLine> = (0..24)
            .map(|n| {
                let g = gsi.clone();
                IrqLine::from_fn(move |level| g.lock().unwrap()[n] = level)
            })
            .collect();
        let clock = Clock::manual(ClockType::Virtual);
        let mut lcfg = Ich9LpcConfig::new(sysmem, io);
        tweak(&mut lcfg);
        let lpc = Ich9Lpc::new(Arc::clone(&mem), q35.bus(), clock.clone(), lcfg, &lines).unwrap();

        let requests: Arc<Mutex<Vec<SystemRequest>>> = Arc::default();
        let r = requests.clone();
        lpc.set_request_handler(Arc::new(move |req| r.lock().unwrap().push(req)));
        let smis: Arc<Mutex<Vec<SmiTarget>>> = Arc::default();
        let s = smis.clone();
        lpc.set_smi_handler(Some(Arc::new(move |t| s.lock().unwrap().push(t))));
        let notifications = Arc::new(AtomicUsize::new(0));
        let n = notifications.clone();
        lpc.set_intx_routing_notifier(Some(Arc::new(move || {
            n.fetch_add(1, Ordering::SeqCst);
        })));

        q35.reset();
        Machine { mem, io, mem_as, io_as, clock, q35, lpc, gsi, requests, smis, notifications }
    }

    fn gsi(&self, n: usize) -> bool {
        self.gsi.lock().unwrap()[n] != 0
    }

    fn high_gsis(&self) -> Vec<usize> {
        let g = self.gsi.lock().unwrap();
        (0..24).filter(|&n| g[n] != 0).collect()
    }

    fn requests(&self) -> Vec<SystemRequest> {
        std::mem::take(&mut *self.requests.lock().unwrap())
    }

    fn smis(&self) -> Vec<SmiTarget> {
        std::mem::take(&mut *self.smis.lock().unwrap())
    }

    fn cfg_addr(devfn: u8, off: usize) -> u64 {
        u64::from(0x8000_0000 | (u32::from(devfn) << 8) | (off as u32 & 0xfc))
    }

    fn cfg_read(&self, devfn: u8, off: usize, len: u32) -> u32 {
        assert!(
            self.io_as.store(0xcf8, 4, Self::cfg_addr(devfn, off), Endian::Little, ATTRS).is_ok()
        );
        let (v, r) = self.io_as.load(0xcfc + (off as u64 & 3), len, Endian::Little, ATTRS);
        assert!(r.is_ok());
        v as u32
    }

    fn cfg_write(&self, devfn: u8, off: usize, len: u32, v: u32) {
        assert!(
            self.io_as.store(0xcf8, 4, Self::cfg_addr(devfn, off), Endian::Little, ATTRS).is_ok()
        );
        let r = self.io_as.store(0xcfc + (off as u64 & 3), len, v.into(), Endian::Little, ATTRS);
        assert!(r.is_ok());
    }

    fn lpc_writeb(&self, off: usize, v: u8) {
        self.cfg_write(LPC, off, 1, v.into());
    }

    fn lpc_writel(&self, off: usize, v: u32) {
        self.cfg_write(LPC, off, 4, v);
    }

    fn lpc_readb(&self, off: usize) -> u8 {
        self.cfg_read(LPC, off, 1) as u8
    }

    fn lpc_readl(&self, off: usize) -> u32 {
        self.cfg_read(LPC, off, 4)
    }

    fn inb(&self, port: u64) -> u8 {
        self.io_as.load(port, 1, Endian::Little, ATTRS).0 as u8
    }

    fn inw(&self, port: u64) -> u16 {
        self.io_as.load(port, 2, Endian::Little, ATTRS).0 as u16
    }

    fn inl(&self, port: u64) -> u32 {
        self.io_as.load(port, 4, Endian::Little, ATTRS).0 as u32
    }

    fn outb(&self, port: u64, v: u8) {
        let _ = self.io_as.store(port, 1, v.into(), Endian::Little, ATTRS);
    }

    fn outw(&self, port: u64, v: u16) {
        let _ = self.io_as.store(port, 2, v.into(), Endian::Little, ATTRS);
    }

    fn outl(&self, port: u64, v: u32) {
        let _ = self.io_as.store(port, 4, v.into(), Endian::Little, ATTRS);
    }

    fn readl(&self, addr: u64) -> u32 {
        self.mem_as.load(addr, 4, Endian::Little, ATTRS).0 as u32
    }

    fn writel(&self, addr: u64, v: u32) {
        let _ = self.mem_as.store(addr, 4, v.into(), Endian::Little, ATTRS);
    }

    /// What tco-test.c and the firmware do: PMBASE = 0xb000 | 1, ACPI_CNTL = ACPI_EN, and RCBA.
    fn enable_pm_and_rcba(&self) {
        self.lpc_writel(ICH9_LPC_PMBASE as usize, PM_IO_BASE_ADDR as u32 | 1);
        self.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x80);
        self.lpc_writel(ICH9_LPC_RCBA, RCBA_BASE_ADDR as u32 | 1);
    }

    /// A single function device in `slot` with INTA.
    fn add_device(&self, slot: u8) -> Arc<PciDevice> {
        let info = PciDeviceInfo {
            name: format!("dev{slot}"),
            vendor_id: 0x1234,
            device_id: 0x5678,
            class_id: PCI_CLASS_OTHERS,
            ..PciDeviceInfo::default()
        };
        let dev = self.q35.bus().register_device(&info, Some(pci_devfn(slot, 0))).unwrap();
        dev.with_config(|c| c.config[PCI_INTERRUPT_PIN] = 1);
        dev
    }

    /// Raises the power button status with its enable set, which asserts the SCI.
    fn raise_sci(&self, pmbase: u64) {
        self.outw(pmbase + 2, ACPI_BITMASK_POWER_BUTTON_ENABLE);
        self.lpc.pm().power_down();
        assert!(self.lpc.sci_level());
    }
}

#[test]
fn ids_and_reset_defaults() {
    let m = Machine::new();
    assert_eq!(m.cfg_read(LPC, PCI_VENDOR_ID, 2), 0x8086);
    assert_eq!(m.cfg_read(LPC, PCI_DEVICE_ID, 2), 0x2918);
    assert_eq!(m.cfg_read(LPC, PCI_REVISION_ID, 1), 0x02);
    assert_eq!(m.cfg_read(LPC, PCI_CLASS_DEVICE, 2), u32::from(PCI_CLASS_BRIDGE_ISA));
    assert_ne!(m.cfg_read(LPC, PCI_HEADER_TYPE, 1) & 0x80, 0, "multifunction");

    for off in [0x60, 0x61, 0x62, 0x63, 0x68, 0x69, 0x6a, 0x6b] {
        assert_eq!(m.lpc_readb(off), ICH9_LPC_PIRQ_ROUT_DEFAULT);
    }
    assert_eq!(m.lpc_readl(ICH9_LPC_PMBASE as usize), ICH9_LPC_PMBASE_DEFAULT);
    assert_eq!(m.lpc_readb(ICH9_LPC_ACPI_CTRL as usize), ICH9_LPC_ACPI_CTRL_DEFAULT);
    assert_eq!(m.lpc_readl(ICH9_LPC_RCBA), 0);
    assert_eq!(m.lpc.sci_gsi(), 9);
    assert_eq!(m.lpc.rst_cnt(), 0);
    assert!(m.high_gsis().is_empty());

    // PMBASE takes bits 15 to 7, plus the ACPI_CNTL bits QEMU puts in its low byte.
    let wmask = m.lpc.device().wmask_bytes();
    assert_eq!(pci_get_long(&wmask, ICH9_LPC_PMBASE as usize), 0xff87);
    assert_eq!(wmask[ICH9_LPC_ACPI_CTRL as usize], 0xff);
    assert_eq!(pci_get_long(&wmask, ICH9_LPC_RCBA), 0xffff_ffff);
    m.lpc_writel(ICH9_LPC_PMBASE as usize, 0xffff_ffff);
    assert_eq!(m.lpc_readl(ICH9_LPC_PMBASE as usize), 0xff87);
}

#[test]
fn default_intx_to_pirq_mapping() {
    let m = Machine::new();
    // D31IR and friends after reset: INTA to PIRQA and so on.
    for slot in [25, 26, 27, 28, 29, 31] {
        for pin in 0..4 {
            assert_eq!(m.lpc.map_irq(pci_devfn(slot, 0), pin), pin, "slot {slot}");
        }
    }
    // The DMI to PCI bridge: INT[A-D] to PIRQ[E-H].
    for pin in 0..4 {
        assert_eq!(m.lpc.map_irq(pci_devfn(30, 0), pin), pin + 4);
    }
    // Everything else: INT[A-D] to PIRQ[E-H], rotated by slot.
    for slot in [0u8, 1, 2, 3, 4, 24] {
        for pin in 0..4 {
            assert_eq!(m.lpc.map_irq(pci_devfn(slot, 0), pin), (usize::from(slot) + pin) % 4 + 4);
        }
    }
}

#[test]
fn disabled_pirq_goes_to_ioapic_only() {
    let m = Machine::new();
    let dev = m.add_device(2);
    // Slot 2 INTA is PIRQG, which sits on GSI 22.
    assert_eq!(m.lpc.map_irq(dev.devfn(), 0), 6);
    assert_eq!(
        m.lpc.route_intx_pin_to_irq(6),
        PciIntxRoute { mode: PciIntxMode::Enabled, irq: 22 }
    );

    dev.set_irq(1);
    assert_eq!(m.high_gsis(), [22]);
    dev.set_irq(0);
    assert!(m.high_gsis().is_empty());

    // Every PIRQ has its own I/O APIC input.
    for (slot, gsi) in [(3u8, 23usize), (4, 20), (5, 21)] {
        let d = m.add_device(slot);
        d.set_irq(1);
        assert_eq!(m.high_gsis(), [gsi], "slot {slot}");
        d.set_irq(0);
    }
    for pirq in 0..8 {
        assert_eq!(ich9_pirq_to_gsi(pirq), pirq + 16);
        assert_eq!(ich9_gsi_to_pirq(pirq + 16), pirq);
    }
}

#[test]
fn pirq_route_register_moves_the_8259_input() {
    let m = Machine::new();
    let dev = m.add_device(2);
    let before = m.notifications.load(Ordering::SeqCst);

    // PIRQG to IRQ 11.
    m.lpc_writeb(ICH9_LPC_PIRQG_ROUT, 0x0b);
    assert_eq!(m.notifications.load(Ordering::SeqCst), before + 1);
    assert_eq!(m.lpc.pic_irq(6), (11, false));
    assert_eq!(
        m.lpc.route_intx_pin_to_irq(6),
        PciIntxRoute { mode: PciIntxMode::Enabled, irq: 11 }
    );

    // The I/O APIC input follows the PIRQ whatever the routing.
    dev.set_irq(1);
    assert_eq!(m.high_gsis(), [11, 22]);
    dev.set_irq(0);
    assert!(m.high_gsis().is_empty());

    // Move it to IRQ 5 while low: the next edge goes there.
    m.lpc_writeb(ICH9_LPC_PIRQG_ROUT, 0x05);
    dev.set_irq(1);
    assert_eq!(m.high_gsis(), [5, 22]);
    dev.set_irq(0);

    // Disabling the 8259 route again leaves only the I/O APIC.
    m.lpc_writeb(ICH9_LPC_PIRQG_ROUT, 0x85);
    assert_eq!(m.lpc.pic_irq(6), (5, true));
    dev.set_irq(1);
    assert_eq!(m.high_gsis(), [22]);
    dev.set_irq(0);

    // A PIRQA to PIRQD write fires the notifier too, other registers do not.
    let n = m.notifications.load(Ordering::SeqCst);
    m.lpc_writeb(ICH9_LPC_PIRQB_ROUT, 0x0a);
    m.lpc_writeb(0x64, 0x0a);
    assert_eq!(m.notifications.load(Ordering::SeqCst), n + 1);
}

#[test]
fn shared_8259_input_is_a_wired_or() {
    let m = Machine::new();
    // Slot 2 is PIRQG, slot 3 is PIRQH. Route both to IRQ 10.
    let a = m.add_device(2);
    let b = m.add_device(3);
    m.lpc_writeb(ICH9_LPC_PIRQG_ROUT, 0x0a);
    m.lpc_writeb(ICH9_LPC_PIRQH_ROUT, 0x0a);
    a.set_irq(1);
    b.set_irq(1);
    assert_eq!(m.high_gsis(), [10, 22, 23]);
    a.set_irq(0);
    assert_eq!(m.high_gsis(), [10, 23]);
    b.set_irq(0);
    assert!(m.high_gsis().is_empty());
}

#[test]
fn pmbase_makes_the_pm_timer_visible() {
    let m = Machine::new();
    let tmr = PM_IO_BASE_ADDR + 8;
    assert!(!m.io_as.load(tmr, 4, Endian::Little, ATTRS).1.is_ok());

    m.lpc_writel(ICH9_LPC_PMBASE as usize, PM_IO_BASE_ADDR as u32 | 1);
    // Still hidden until ACPI_EN.
    assert!(!m.io_as.load(tmr, 4, Endian::Little, ATTRS).1.is_ok());
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x80);
    assert_eq!(m.lpc.pm().pm_io_base(), PM_IO_BASE_ADDR as u32);

    let t0 = m.inl(tmr);
    m.clock.advance_to(1_000_000);
    let t1 = m.inl(tmr);
    // 3.579545 MHz: about 3579 ticks per millisecond.
    assert_eq!(t1.wrapping_sub(t0) & 0xff_ffff, 3579);

    // Move the window.
    m.lpc_writel(ICH9_LPC_PMBASE as usize, 0x0601);
    assert!(!m.io_as.load(tmr, 4, Endian::Little, ATTRS).1.is_ok());
    assert!(m.io_as.load(0x608, 4, Endian::Little, ATTRS).1.is_ok());

    // ACPI_EN off hides it again.
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0);
    assert!(!m.io_as.load(0x608, 4, Endian::Little, ATTRS).1.is_ok());

    // Reset hides it too.
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x80);
    assert!(m.io_as.load(0x608, 4, Endian::Little, ATTRS).1.is_ok());
    m.q35.reset();
    assert!(!m.io_as.load(0x608, 4, Endian::Little, ATTRS).1.is_ok());
}

#[test]
fn sci_follows_acpi_cntl() {
    let m = Machine::new();
    m.enable_pm_and_rcba();
    m.raise_sci(PM_IO_BASE_ADDR);
    assert_eq!(m.high_gsis(), [9]);

    // Changing the SCI IRQ select lowers the old line and raises the new one.
    for (sel, gsi) in [(0x81u8, 10usize), (0x82, 11), (0x84, 20), (0x85, 21), (0x80, 9)] {
        m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, sel);
        assert_eq!(m.lpc.sci_gsi() as usize, gsi);
        assert_eq!(m.high_gsis(), [gsi], "ACPI_CNTL {sel:#x}");
    }

    // A reserved select leaves the SCI where it is.
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x84);
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x83);
    assert_eq!(m.lpc.sci_gsi(), 20);
    assert_eq!(m.high_gsis(), [20]);

    // Clearing the status drops it.
    m.outw(PM_IO_BASE_ADDR, 0x0100);
    assert!(!m.lpc.sci_level());
    assert!(m.high_gsis().is_empty());
}

#[test]
fn sci_is_ored_with_pirqs() {
    let m = Machine::new();
    m.enable_pm_and_rcba();
    // SCI on GSI 20, which is also PIRQE's I/O APIC input. Slot 4 INTA is PIRQE.
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x84);
    let dev = m.add_device(4);
    dev.set_irq(1);
    m.raise_sci(PM_IO_BASE_ADDR);
    dev.set_irq(0);
    assert_eq!(m.high_gsis(), [20]);
    m.outw(PM_IO_BASE_ADDR, 0x0100);
    assert!(m.high_gsis().is_empty());

    // SCI on 9 with PIRQA routed there too.
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x80);
    m.lpc_writeb(ICH9_LPC_PIRQE_ROUT, 0x09);
    dev.set_irq(1);
    m.raise_sci(PM_IO_BASE_ADDR);
    dev.set_irq(0);
    assert_eq!(m.high_gsis(), [9]);
}

/// lpc-ich9-test.c, test_lp1878642_pci_bus_get_irq_level_assert: a reserved SCI select with the
/// SCI raised must not index past the PIRQs.
#[test]
fn lp1878642_pci_bus_get_irq_level_assert() {
    let m = Machine::new();
    m.io_as.store(0xcf8, 4, 0x8000_f840, Endian::Little, ATTRS);
    m.io_as.store(0xcfc, 4, 0x5d00, Endian::Little, ATTRS);
    m.io_as.store(0xcf8, 4, 0x8000_f844, Endian::Little, ATTRS);
    m.io_as.store(0xcfc, 4, 0xeb, Endian::Little, ATTRS);
    m.outw(0x5d02, 0x205d);
    assert_eq!(m.lpc.sci_gsi(), 9);
}

#[test]
fn reset_control_register() {
    let m = Machine::new();
    assert_eq!(m.inb(0xcf9), 0);

    // SYS_RST and FULL_RST are kept, nothing happens yet.
    m.outb(0xcf9, 0x02);
    assert_eq!(m.inb(0xcf9), 0x02);
    m.outb(0xcf9, 0x0a);
    assert_eq!(m.inb(0xcf9), 0x0a);
    m.outb(0xcf9, 0xf1);
    assert_eq!(m.inb(0xcf9), 0x00);
    assert!(m.requests().is_empty());

    // RST_CPU requests a reset, soft (0x04) or hard (0x06) alike, and the register keeps its
    // old value.
    m.outb(0xcf9, 0x02);
    m.outb(0xcf9, 0x06);
    assert_eq!(m.requests(), [SystemRequest::Reset]);
    assert_eq!(m.inb(0xcf9), 0x02);
    m.outb(0xcf9, 0x04);
    assert_eq!(m.requests(), [SystemRequest::Reset]);
    m.outb(0xcf9, 0x0e);
    assert_eq!(m.requests(), [SystemRequest::Reset]);
    assert_eq!(m.lpc.rst_cnt(), 0x02);

    // CONFIG_ADDRESS at 0xcf8 still takes 32 bit accesses.
    assert_eq!(m.cfg_read(LPC, PCI_VENDOR_ID, 2), 0x8086);

    m.q35.reset();
    assert_eq!(m.lpc.rst_cnt(), 0);
}

#[test]
fn apm_ports_and_smi() {
    let m = Machine::new();
    m.outb(0xb3, 0x55);
    assert_eq!(m.inb(0xb3), 0x55);
    assert!(m.smis().is_empty());

    // Without smm-enabled, APMC_EN is set at reset, so every command raises an SMI.
    assert_ne!(m.lpc.pm().smi_en() & ICH9_PMIO_SMI_EN_APMC_EN, 0);
    m.outb(0xb2, 0x10);
    assert_eq!(m.inb(0xb2), 0x10);
    assert_eq!(m.lpc.apm(), (0x10, 0x55));
    assert_eq!(m.smis(), [SmiTarget::Current]);

    // A 16 bit write is split into the two ports.
    m.outw(0xb2, 0x6677);
    assert_eq!(m.lpc.apm(), (0x77, 0x66));
    assert_eq!(m.smis(), [SmiTarget::Current]);

    // Without SMM the PM block is ACPI only: SCI_EN stays on and the ACPI commands raise no SMI.
    m.enable_pm_and_rcba();
    m.outb(0xb2, ICH9_APM_ACPI_DISABLE);
    assert!(m.smis().is_empty());
    assert_ne!(m.inw(PM_IO_BASE_ADDR + 4) & ACPI_BITMASK_SCI_ENABLE, 0);
}

#[test]
fn apm_smi_needs_apmc_en_with_smm() {
    let m = Machine::with(|c| c.pm = Ich9PmProps { smm_enabled: true, ..Ich9PmProps::default() });
    m.enable_pm_and_rcba();
    m.outb(0xb2, 0x10);
    assert!(m.smis().is_empty());

    // ACPI enable and disable only flip SCI_EN.
    assert_eq!(m.inw(PM_IO_BASE_ADDR + 4) & ACPI_BITMASK_SCI_ENABLE, 0);
    m.outb(0xb2, ICH9_APM_ACPI_ENABLE);
    assert_ne!(m.inw(PM_IO_BASE_ADDR + 4) & ACPI_BITMASK_SCI_ENABLE, 0);
    m.outb(0xb2, ICH9_APM_ACPI_DISABLE);
    assert_eq!(m.inw(PM_IO_BASE_ADDR + 4) & ACPI_BITMASK_SCI_ENABLE, 0);

    m.outl(PM_IO_BASE_ADDR + 0x30, ICH9_PMIO_SMI_EN_APMC_EN);
    m.outb(0xb2, 0x10);
    assert_eq!(m.smis(), [SmiTarget::Current]);

    // After negotiating broadcast, SMIs go to every CPU.
    m.lpc.set_smi_guest_features(1u64.to_le_bytes());
    m.lpc.smi_features_ok_select();
    assert_eq!(m.lpc.smi_features_ok(), 1);
    m.outb(0xb2, 0x10);
    assert_eq!(m.smis(), [SmiTarget::Broadcast]);

    // Reset forgets the negotiation.
    m.q35.reset();
    assert_eq!(m.lpc.smi_features_ok(), 0);
    assert_eq!(m.lpc.smi_negotiated_features(), 0);
}

#[test]
fn smi_feature_negotiation() {
    let m = Machine::new();
    assert_eq!(m.lpc.smi_host_features(), 7);
    assert_eq!(m.lpc.smi_host_features_le(), [7, 0, 0, 0, 0, 0, 0, 0]);

    // Unknown bits, hotplug without broadcast and hot-unplug without hotplug are refused.
    for bad in [8u64, 2, 4, 5] {
        m.lpc.set_smi_guest_features(bad.to_le_bytes());
        m.lpc.smi_features_ok_select();
        assert_eq!(m.lpc.smi_features_ok(), 0, "features {bad:#x}");
    }
    m.lpc.set_smi_guest_features(3u64.to_le_bytes());
    m.lpc.smi_features_ok_select();
    assert_eq!(m.lpc.smi_features_ok(), 1);
    assert_eq!(m.lpc.smi_negotiated_features(), 3);
    // Locked once accepted.
    m.lpc.set_smi_guest_features(7u64.to_le_bytes());
    m.lpc.smi_features_ok_select();
    assert_eq!(m.lpc.smi_negotiated_features(), 3);

    // Hot-unplug without hotplug on the host side is a realize error.
    let mem = Arc::new(MemorySystem::new());
    let sys = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_container("io", 1 << 16).unwrap();
    let bus = ruvm_hw_pci::PciBus::new_root("pcie.0", mem.clone(), sys, io, 0);
    let mut cfg = Ich9LpcConfig::new(sys, io);
    cfg.smi_host_features = 1 << ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT;
    let lines = vec![IrqLine::default(); 24];
    let clock = Clock::manual(ClockType::Virtual);
    assert!(Ich9Lpc::new(mem.clone(), &bus, clock.clone(), cfg, &lines).is_err());
    // So is a GSI array of the wrong size.
    let cfg = Ich9LpcConfig::new(sys, io);
    assert!(Ich9Lpc::new(mem, &bus, clock, cfg, &lines[..16]).is_err());
}

#[test]
fn rcba_maps_the_chipset_config_registers() {
    let m = Machine::new();
    let gcs = RCBA_BASE_ADDR + ICH9_CC_GCS as u64;
    assert_ne!(m.readl(gcs), 0x3210);
    assert_ne!(m.readl(RCBA_BASE_ADDR + ICH9_CC_D31IR as u64), 0x3210);

    m.enable_pm_and_rcba();
    assert_eq!(m.readl(RCBA_BASE_ADDR + ICH9_CC_D31IR as u64), ICH9_CC_DIR_DEFAULT);
    assert_eq!(m.readl(RCBA_BASE_ADDR + ICH9_CC_D30IR as u64) & 0xffff, 0);
    assert_eq!(m.readl(gcs), 0);

    // tco-test.c flips NO_REBOOT through the RCRB.
    m.writel(gcs, ICH9_CC_GCS_NO_REBOOT);
    assert_eq!(m.readl(gcs), ICH9_CC_GCS_NO_REBOOT);
    assert_eq!(m.lpc.cc_read(ICH9_CC_GCS as u64, 4), u64::from(ICH9_CC_GCS_NO_REBOOT));

    // D31IR decides where the INTx pins of slot 31 go, and writing it fires the notifier.
    let n = m.notifications.load(Ordering::SeqCst);
    m.mem_as.store(RCBA_BASE_ADDR + ICH9_CC_D31IR as u64, 2, 0x4567, Endian::Little, ATTRS);
    assert_eq!(m.notifications.load(Ordering::SeqCst), n + 1);
    let slot31 = pci_devfn(31, 0);
    assert_eq!([0, 1, 2, 3].map(|p| m.lpc.map_irq(slot31, p)), [7, 6, 5, 4]);
    // D30IR is ignored.
    m.lpc.cc_write(ICH9_CC_D30IR as u64, 0, 2);
    assert_eq!(m.lpc.map_irq(pci_devfn(30, 0), 0), 4);

    // An access at the very end is clipped, not an overflow.
    assert_eq!(m.lpc.cc_read(ICH9_CC_SIZE as u64 - 1, 4), 0);

    // Move the block.
    m.lpc_writel(ICH9_LPC_RCBA, 0xfed2_0001);
    assert_eq!(m.readl(0xfed2_0000 + ICH9_CC_GCS as u64), ICH9_CC_GCS_NO_REBOOT);
    assert_ne!(m.readl(gcs), ICH9_CC_GCS_NO_REBOOT);
    // The low bits of the base are ignored.
    m.lpc_writel(ICH9_LPC_RCBA, 0xfed1_ffff);
    assert_eq!(m.readl(gcs), ICH9_CC_GCS_NO_REBOOT);

    // Disable it.
    m.lpc_writel(ICH9_LPC_RCBA, RCBA_BASE_ADDR as u32);
    assert!(!m.mem_as.load(gcs, 4, Endian::Little, ATTRS).1.is_ok());

    // Reset unmaps it and restores the defaults.
    m.lpc_writel(ICH9_LPC_RCBA, RCBA_BASE_ADDR as u32 | 1);
    m.q35.reset();
    assert!(!m.mem_as.load(gcs, 4, Endian::Little, ATTRS).1.is_ok());
    assert_eq!(m.lpc.cc_read(ICH9_CC_GCS as u64, 4), 0);
    assert_eq!(m.lpc.map_irq(slot31, 0), 0);
}

#[test]
fn noreboot_sets_gcs() {
    let m = Machine::with(|c| c.noreboot = true);
    m.enable_pm_and_rcba();
    assert_eq!(m.readl(RCBA_BASE_ADDR + ICH9_CC_GCS as u64), ICH9_CC_GCS_NO_REBOOT);
}

#[test]
fn smi_lock_sticks() {
    let m = Machine::new();
    m.cfg_write(LPC, ICH9_LPC_GEN_PMCON_1, 2, 0x0001);
    assert_eq!(m.cfg_read(LPC, ICH9_LPC_GEN_PMCON_1, 2), 0x0001);
    m.cfg_write(LPC, ICH9_LPC_GEN_PMCON_1, 2, 0x0010);
    m.cfg_write(LPC, ICH9_LPC_GEN_PMCON_1, 2, 0x0000);
    assert_eq!(m.cfg_read(LPC, ICH9_LPC_GEN_PMCON_1, 2), 0x0010);
}

#[test]
fn reset_restores_routing() {
    let m = Machine::new();
    m.enable_pm_and_rcba();
    m.lpc_writeb(ICH9_LPC_PIRQA_ROUT, 0x0b);
    m.lpc_writeb(ICH9_LPC_ACPI_CTRL as usize, 0x81);
    m.raise_sci(PM_IO_BASE_ADDR);
    assert_eq!(m.high_gsis(), [10]);
    m.q35.reset();
    assert_eq!(m.lpc_readb(ICH9_LPC_PIRQA_ROUT), 0x80);
    assert_eq!(m.lpc.sci_gsi(), 9);
    assert!(!m.lpc.sci_level());
    assert!(!m.gsi(10));
    assert_eq!(m.lpc.pm().pm_io_base(), 0);
}

#[test]
fn machine_ready_reports_legacy_devices() {
    let m = Machine::new();
    let com1 = m.mem.new_reservation("serial", 8).unwrap();
    m.mem.add_subregion(m.io, 0x3f8, com1).unwrap();
    let fdc = m.mem.new_reservation("fdc", 4).unwrap();
    m.mem.add_subregion(m.io, 0x3f0, fdc).unwrap();
    m.lpc.machine_ready();
    assert_eq!(m.lpc_readb(0x82), 0x09);
}
