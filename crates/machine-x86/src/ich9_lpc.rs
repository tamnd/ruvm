// SPDX-License-Identifier: GPL-2.0-or-later

//! The ICH9 LPC bridge at 00:1f.0 of the q35 machine, from hw/isa/lpc_ich9.c, plus the APM
//! control and status ports from hw/isa/apm.c.
//!
//! [`Ich9Lpc::new`] does what `ich9_lpc_initfn()`, `ich9_lpc_realize()` and `ich9_lpc_pm_init()`
//! do. It registers function 8086:2918 at devfn 0xf8 on the root bus and takes over the bus's
//! INTx routing (`pci_bus_irqs()` and `pci_bus_map_irqs()`), creates the ICH9 power management
//! block ([`Ich9Pm`]) with its SCI wired back into the bridge, and maps these regions:
//!
//! - "ich9-pm", the 128 byte PM window, hidden until PMBASE and ACPI_CNTL enable it,
//! - "apm-io" at 0xb2 and 0xb3 in the I/O space,
//! - "lpc-reset-control" at 0xcf9, over the PCI CONFIG_ADDRESS register at priority 1,
//! - "lpc-rcrb-mmio", the 16 KiB root complex register block, placed in system memory at
//!   priority 1 whenever RCBA has its enable bit set.
//!
//! Interrupts work as in QEMU. A function behind the bridge raising INTx pin `n` in slot `s` is
//! mapped to PIRQ `irr[s][n]` (the D25IR to D31IR chipset config registers, with slot 30 fixed
//! to PIRQE to PIRQH). Each PIRQ then drives two things: its fixed I/O APIC input, GSI 16 to 23,
//! and the 8259 input its PIRQx_ROUT register selects when that register's IRQEN bit is clear.
//! The SCI goes to the GSI that ACPI_CNTL selects (9, 10, 11, 20 or 21) and is ORed into that
//! line. The board hands [`Ich9Lpc::new`] the 24 GSI lines, `lpc->gsi[]` in QEMU.
//!
//! Like QEMU, nothing is decoded until the first reset: call [`Ich9Lpc::reset`], or reset the
//! root bus, before running the guest.
//!
//! Differences from QEMU:
//!
//! - Reset requests from 0xcf9 go to the [`SystemRequestHandler`] as [`SystemRequest::Reset`],
//!   and SMIs raised by an APM command go to an [`SmiHandler`] instead of `cpu_interrupt()`.
//! - The INTx routing notifier (`pci_bus_fire_intx_routing_notifier()`) is a callback set with
//!   [`Ich9Lpc::set_intx_routing_notifier`], and `ich9_route_intx_pin_to_irq()` is
//!   [`Ich9Lpc::route_intx_pin_to_irq`]; the PCI core has no `pci_route_intx_to_irq()` yet.
//! - Resetting the bridge also runs `pm_reset()`, which QEMU registers as a separate reset
//!   handler. The board should not reset the PM block on its own.
//! - Setting SMI_LOCK in GEN_PMCON_1 locks the bit in config space but cannot lock GBL_SMI_EN in
//!   SMI_EN, because [`Ich9Pm`] has no hook for its SMI_EN write mask.
//! - The SMI feature negotiation (`smi_features_ok_callback()`) is ported as methods, but the
//!   board adds the three `etc/smi/*` fw_cfg files itself: [`Ich9Lpc::smi_host_features_le`] is
//!   the content of `etc/smi/supported-features`, [`Ich9Lpc::set_smi_guest_features`] receives
//!   `etc/smi/requested-features` and [`Ich9Lpc::smi_features_ok_select`] is the select callback
//!   of `etc/smi/features-ok`.
//! - `memory_region_present()` in the machine-ready hook is answered from the rendered I/O space.
//!
//! Not ported: VMState (`ich9_lpc_post_load()` would call the three update functions), trace
//! points, QOM registration and properties (they are fields of [`Ich9LpcConfig`]), the TCO
//! watchdog and `ich9_generate_smi()`, the SWSMI and periodic SMI timers, the ISA bus with its
//! i8257 DMA and RTC children (the board creates those), hotplug handlers, ACPI device
//! interfaces and the AML builder.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, RwLock, Weak};

use ruvm_base::Error;
use ruvm_hw_acpi::ich9::{ICH9_PMIO_SMI_EN_APMC_EN, Ich9Pm, Ich9PmProps};
use ruvm_hw_acpi::{SystemRequest, SystemRequestHandler};
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_pci::regs::{
    PCI_CLASS_BRIDGE_ISA, PCI_NUM_PINS, PCI_SLOT_MAX, pci_devfn, pci_get_long, pci_get_word,
    pci_set_long, pci_set_word, pci_slot,
};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemError, MemResult, MemorySystem, MmioOps, RegionId,
};

pub use ruvm_hw_acpi::ich9::{
    ICH9_LPC_ACPI_CTRL, ICH9_LPC_ACPI_CTRL_ACPI_EN, ICH9_LPC_ACPI_CTRL_DEFAULT,
    ICH9_LPC_ACPI_CTRL_SCI_IRQ_SEL_MASK, ICH9_LPC_PMBASE, ICH9_LPC_PMBASE_BASE_ADDRESS_MASK,
    ICH9_LPC_PMBASE_DEFAULT, ICH9_LPC_PMBASE_RTE,
};

/// `TYPE_ICH9_LPC_DEVICE`.
pub const TYPE_ICH9_LPC_DEVICE: &str = "ICH9-LPC";

/// `PCI_VENDOR_ID_INTEL`.
pub const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// `PCI_DEVICE_ID_INTEL_ICH9_8`.
pub const PCI_DEVICE_ID_INTEL_ICH9_8: u16 = 0x2918;
/// `ICH9_A2_LPC_REVISION`.
pub const ICH9_A2_LPC_REVISION: u8 = 0x2;

pub const ICH9_LPC_DEV: u8 = 31;
pub const ICH9_LPC_FUNC: u8 = 0;
/// PIRQA to PIRQH.
pub const ICH9_LPC_NB_PIRQS: usize = 8;
/// Inputs 0 to 15 are shared by the 8259 pair and the I/O APIC.
pub const ICH9_LPC_PIC_NUM_PINS: usize = 16;
/// `ICH9_LPC_IOAPIC_NUM_PINS`, also `IOAPIC_NUM_PINS`: the number of GSI lines.
pub const ICH9_LPC_IOAPIC_NUM_PINS: usize = 24;

/// The size of the chipset configuration registers (the root complex register block).
pub const ICH9_CC_SIZE: usize = 16 * 1024;
pub const ICH9_CC_ADDR_MASK: u64 = ICH9_CC_SIZE as u64 - 1;
pub const ICH9_CC_D28IP: usize = 0x310c;
pub const ICH9_CC_D28IP_SHIFT: u32 = 4;
pub const ICH9_CC_D28IP_MASK: u32 = 0xf;
pub const ICH9_CC_D28IP_DEFAULT: u32 = 0x0021_4321;
pub const ICH9_CC_D31IR: usize = 0x3140;
pub const ICH9_CC_D30IR: usize = 0x3142;
pub const ICH9_CC_D29IR: usize = 0x3144;
pub const ICH9_CC_D28IR: usize = 0x3146;
pub const ICH9_CC_D27IR: usize = 0x3148;
pub const ICH9_CC_D26IR: usize = 0x314c;
pub const ICH9_CC_D25IR: usize = 0x3150;
pub const ICH9_CC_DIR_DEFAULT: u32 = 0x3210;
pub const ICH9_CC_D30IR_DEFAULT: u32 = 0x0;
pub const ICH9_CC_DIR_SHIFT: u32 = 4;
pub const ICH9_CC_DIR_MASK: u16 = 0x7;
pub const ICH9_CC_OIC: usize = 0x31ff;
pub const ICH9_CC_OIC_AEN: u8 = 0x1;
pub const ICH9_CC_GCS: usize = 0x3410;
pub const ICH9_CC_GCS_DEFAULT: u32 = 0x0000_0000;
pub const ICH9_CC_GCS_NO_REBOOT: u32 = 1 << 5;

/// The reset control register.
pub const ICH9_RST_CNT_IOPORT: u64 = 0xcf9;

pub const ICH9_LPC_PIRQA_ROUT: usize = 0x60;
pub const ICH9_LPC_PIRQB_ROUT: usize = 0x61;
pub const ICH9_LPC_PIRQC_ROUT: usize = 0x62;
pub const ICH9_LPC_PIRQD_ROUT: usize = 0x63;
pub const ICH9_LPC_PIRQE_ROUT: usize = 0x68;
pub const ICH9_LPC_PIRQF_ROUT: usize = 0x69;
pub const ICH9_LPC_PIRQG_ROUT: usize = 0x6a;
pub const ICH9_LPC_PIRQH_ROUT: usize = 0x6b;
/// Set to disconnect the PIRQ from the 8259.
pub const ICH9_LPC_PIRQ_ROUT_IRQEN: u8 = 0x80;
/// `ICH9_MASK(8, 3, 0)`.
pub const ICH9_LPC_PIRQ_ROUT_MASK: u8 = 0x0f;
pub const ICH9_LPC_PIRQ_ROUT_DEFAULT: u8 = 0x80;

pub const ICH9_LPC_GEN_PMCON_1: usize = 0xa0;
pub const ICH9_LPC_GEN_PMCON_1_SMI_LOCK: u16 = 1 << 4;
pub const ICH9_LPC_GEN_PMCON_2: usize = 0xa2;
pub const ICH9_LPC_GEN_PMCON_3: usize = 0xa4;
pub const ICH9_LPC_GEN_PMCON_LOCK: usize = 0xa6;

pub const ICH9_LPC_RCBA: usize = 0xf0;
/// `ICH9_MASK(32, 31, 14)`.
pub const ICH9_LPC_RCBA_BA_MASK: u32 = 0xffff_c000;
pub const ICH9_LPC_RCBA_EN: u32 = 0x1;
pub const ICH9_LPC_RCBA_DEFAULT: u32 = 0x0;

/// The APM commands that turn SCI_EN on and off.
pub const ICH9_APM_ACPI_ENABLE: u8 = 0x2;
pub const ICH9_APM_ACPI_DISABLE: u8 = 0x3;

/// `APM_CNT_IOPORT`, followed by the status port at 0xb3.
pub const APM_CNT_IOPORT: u64 = 0xb2;
pub const APM_STS_IOPORT: u64 = 0xb3;

pub const ICH9_LPC_SMI_F_BROADCAST_BIT: u32 = 0;
pub const ICH9_LPC_SMI_F_CPU_HOTPLUG_BIT: u32 = 1;
pub const ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT: u32 = 2;

/// Where an SMI raised by an APM command goes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SmiTarget {
    /// The CPU that wrote the APM command, `current_cpu`.
    Current,
    /// Every CPU, once the guest negotiated SMI broadcast.
    Broadcast,
}

/// Receives the SMIs the APM command port raises.
pub type SmiHandler = Arc<dyn Fn(SmiTarget) + Send + Sync>;

/// Called when the INTx routing changes, `pci_bus_fire_intx_routing_notifier()`.
pub type IntxRoutingNotifier = Arc<dyn Fn() + Send + Sync>;

/// `PCIINTxMode` as far as this bridge uses it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PciIntxMode {
    Enabled,
    Disabled,
}

/// `PCIINTxRoute`: where a PIRQ ends up, for callers that bypass the emulated routing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PciIntxRoute {
    pub mode: PciIntxMode,
    /// The GSI, -1 when disabled.
    pub irq: i32,
}

/// `ich9_pirq_to_gsi()`: PIRQA to PIRQH sit on I/O APIC inputs 16 to 23.
pub const fn ich9_pirq_to_gsi(pirq: usize) -> usize {
    pirq + ICH9_LPC_PIC_NUM_PINS
}

/// `ich9_gsi_to_pirq()`.
pub const fn ich9_gsi_to_pirq(gsi: usize) -> usize {
    gsi - ICH9_LPC_PIC_NUM_PINS
}

/// `ich9_lpc_rout()`: the 8259 input a PIRQx_ROUT value selects and whether routing to the 8259
/// is disabled.
pub const fn ich9_lpc_rout(pirq_rout: u8) -> (u8, bool) {
    (pirq_rout & ICH9_LPC_PIRQ_ROUT_MASK, pirq_rout & ICH9_LPC_PIRQ_ROUT_IRQEN != 0)
}

/// The config register holding the route of `pirq`.
const fn pirq_rout_reg(pirq: usize) -> usize {
    if pirq < 4 { ICH9_LPC_PIRQA_ROUT + pirq } else { ICH9_LPC_PIRQE_ROUT + (pirq - 4) }
}

/// The properties of the `ICH9-LPC` device and the regions it needs from the board.
#[derive(Clone, Debug)]
pub struct Ich9LpcConfig {
    /// Where the root complex register block goes when RCBA enables it.
    pub system_memory: RegionId,
    /// The I/O space, `pci_address_space_io()`: PM window, APM ports and 0xcf9 go here.
    pub address_space_io: RegionId,
    /// `noreboot`, the speaker pin strap: sets NO_REBOOT in GCS at reset.
    pub noreboot: bool,
    /// The PM properties, including `smm-enabled` and `smm-compat`.
    pub pm: Ich9PmProps,
    /// `x-smi-broadcast`, `x-smi-cpu-hotplug` and `x-smi-cpu-hotunplug`, all on by default.
    pub smi_host_features: u64,
}

impl Ich9LpcConfig {
    /// The given regions with QEMU's default properties.
    pub fn new(system_memory: RegionId, address_space_io: RegionId) -> Ich9LpcConfig {
        Ich9LpcConfig {
            system_memory,
            address_space_io,
            noreboot: false,
            pm: Ich9PmProps::default(),
            smi_host_features: (1 << ICH9_LPC_SMI_F_BROADCAST_BIT)
                | (1 << ICH9_LPC_SMI_F_CPU_HOTPLUG_BIT)
                | (1 << ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT),
        }
    }
}

/// The mutable part of `ICH9LPCState` that is not config space.
struct State {
    /// `irr[slot][intx]`: the PIRQ each INTx pin of each slot is wired to.
    irr: [[u8; PCI_NUM_PINS]; PCI_SLOT_MAX as usize],
    chip_config: Vec<u8>,
    sci_level: bool,
    sci_gsi: u8,
    rst_cnt: u8,
    apmc: u8,
    apms: u8,
    smi_guest_features_le: [u8; 8],
    smi_features_ok: u8,
    smi_negotiated_features: u64,
}

#[derive(Debug)]
struct Regions {
    rcrb: RegionId,
    rst_cnt: RegionId,
    apm: RegionId,
    pm: RegionId,
}

struct Inner {
    memory: Arc<MemorySystem>,
    system_memory: RegionId,
    io: RegionId,
    pm: Arc<Ich9Pm>,
    gsi: Vec<IrqLine>,
    noreboot: bool,
    smi_host_features: u64,
    bus: Weak<PciBus>,
    dev: OnceLock<Weak<PciDevice>>,
    regions: OnceLock<Regions>,
    state: Mutex<State>,
    request: RwLock<Option<SystemRequestHandler>>,
    smi: RwLock<Option<SmiHandler>>,
    intx_notifier: RwLock<Option<IntxRoutingNotifier>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn read<T: Clone>(l: &RwLock<Option<T>>) -> Option<T> {
    l.read().unwrap_or_else(|p| p.into_inner()).clone()
}

fn write<T>(l: &RwLock<Option<T>>, v: Option<T>) {
    *l.write().unwrap_or_else(|p| p.into_inner()) = v;
}

fn ranges_overlap(first1: u32, len1: u32, first2: usize, len2: u32) -> bool {
    let (first1, first2) = (u64::from(first1), first2 as u64);
    let last1 = first1 + u64::from(len1) - 1;
    let last2 = first2 + u64::from(len2) - 1;
    !(last2 < first1 || last1 < first2)
}

/// `ich9_cc_update_ir()`.
fn cc_update_ir(irr: &mut [u8; PCI_NUM_PINS], ir: u16) {
    for (intx, r) in irr.iter_mut().enumerate() {
        *r = ((ir >> (intx as u32 * ICH9_CC_DIR_SHIFT)) & ICH9_CC_DIR_MASK) as u8;
    }
}

impl State {
    /// `ich9_cc_update()`.
    fn cc_update(&mut self) {
        let reg_offsets = [
            ICH9_CC_D25IR,
            ICH9_CC_D26IR,
            ICH9_CC_D27IR,
            ICH9_CC_D28IR,
            ICH9_CC_D29IR,
            ICH9_CC_D30IR,
            ICH9_CC_D31IR,
        ];
        // D{25 - 31}IR, but D30IR is read only to 0.
        for (slot, off) in (25..32).zip(reg_offsets) {
            if slot == 30 {
                continue;
            }
            let ir = pci_get_word(&self.chip_config, off);
            cc_update_ir(&mut self.irr[slot], ir);
        }
        // D30 is the DMI to PCI bridge. How the INTx lines of the devices behind it reach the
        // PIRQs is an arbitrary choice: INT[A-D] go to PIRQ[E-H].
        for (intx, r) in self.irr[30].iter_mut().enumerate() {
            *r = intx as u8 + 4;
        }
    }

    /// `ich9_cc_init()`. The default routing only has to match the ACPI routing table. It
    /// differs from the PIIX one on purpose, and avoids PIRQ A-D, which the PCI Express ports
    /// use: INT[A-D] go to PIRQ[E-H].
    fn cc_init(&mut self) {
        for (slot, pins) in self.irr.iter_mut().enumerate() {
            for (intx, r) in pins.iter_mut().enumerate() {
                *r = ((slot + intx) % 4 + 4) as u8;
            }
        }
        self.cc_update();
    }

    /// `ich9_cc_reset()`.
    fn cc_reset(&mut self, noreboot: bool) {
        let mut gcs = ICH9_CC_GCS_DEFAULT;
        if noreboot {
            gcs |= ICH9_CC_GCS_NO_REBOOT;
        }
        let c = &mut self.chip_config;
        c.fill(0);
        pci_set_long(c, ICH9_CC_D31IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_D30IR, ICH9_CC_D30IR_DEFAULT);
        pci_set_long(c, ICH9_CC_D29IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_D28IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_D27IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_D26IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_D25IR, ICH9_CC_DIR_DEFAULT);
        pci_set_long(c, ICH9_CC_GCS, gcs);
        self.cc_update();
    }
}

/// `ich9_cc_addr_len()`: wraps the offset into the block and clips the length at its end.
fn cc_addr_len(addr: u64, len: u32) -> (usize, usize) {
    let addr = (addr & ICH9_CC_ADDR_MASK) as usize;
    let mut len = len as usize;
    if addr + len >= ICH9_CC_SIZE {
        len = ICH9_CC_SIZE - addr;
    }
    (addr, len)
}

impl Inner {
    fn dev(&self) -> Option<Arc<PciDevice>> {
        self.dev.get().and_then(Weak::upgrade)
    }

    fn regions(&self) -> &Regions {
        self.regions.get().expect("ICH9 LPC regions are created in new()")
    }

    fn fire_intx_routing_notifier(&self) {
        if let Some(n) = read(&self.intx_notifier) {
            n();
        }
    }

    /// `ich9_lpc_pic_irq()`.
    fn pic_irq(&self, pirq: usize) -> (u8, bool) {
        assert!(pirq < ICH9_LPC_NB_PIRQS, "PIRQ {pirq} out of range");
        let rout = self.dev().map_or(0, |d| d.with_config(|c| c.config[pirq_rout_reg(pirq)]));
        ich9_lpc_rout(rout)
    }

    fn bus_irq_level(&self, pirq: usize) -> bool {
        self.bus.upgrade().is_some_and(|b| b.irq_level(pirq))
    }

    fn sci_on(&self, gsi: usize) -> bool {
        let s = lock(&self.state);
        usize::from(s.sci_gsi) == gsi && s.sci_level
    }

    fn set_gsi(&self, gsi: usize, level: bool) {
        if let Some(line) = self.gsi.get(gsi) {
            line.set_bool(level);
        }
    }

    /// `ich9_lpc_update_pic()`: the 8259 input is the OR of every PIRQ routed to it, plus the
    /// SCI when it is routed there.
    fn update_pic(&self, gsi: usize) {
        assert!(gsi < ICH9_LPC_PIC_NUM_PINS);
        let mut level = false;
        for pirq in 0..ICH9_LPC_NB_PIRQS {
            let (irq, dis) = self.pic_irq(pirq);
            if !dis && usize::from(irq) == gsi {
                level |= self.bus_irq_level(pirq);
            }
        }
        level |= self.sci_on(gsi);
        self.set_gsi(gsi, level);
    }

    /// `ich9_lpc_update_apic()`: GSI 16 to 23 follow their PIRQ, plus the SCI.
    fn update_apic(&self, gsi: usize) {
        assert!(gsi >= ICH9_LPC_PIC_NUM_PINS);
        let level = self.bus_irq_level(ich9_gsi_to_pirq(gsi)) | self.sci_on(gsi);
        self.set_gsi(gsi, level);
    }

    /// `ich9_lpc_set_irq()`, the bus's `set_irq` handler.
    fn set_irq(&self, pirq: usize) {
        assert!(pirq < ICH9_LPC_NB_PIRQS, "PIRQ {pirq} out of range");
        self.update_apic(ich9_pirq_to_gsi(pirq));
        let (pic_irq, _) = self.pic_irq(pirq);
        self.update_pic(usize::from(pic_irq));
    }

    /// `ich9_lpc_map_irq()`.
    fn map_irq(&self, devfn: u8, intx: i32) -> i32 {
        let s = lock(&self.state);
        i32::from(s.irr[usize::from(pci_slot(devfn))][intx as usize])
    }

    /// `ich9_route_intx_pin_to_irq()`.
    fn route_intx_pin_to_irq(&self, pirq: usize) -> PciIntxRoute {
        let (pic_irq, pic_dis) = self.pic_irq(pirq);
        if !pic_dis {
            if usize::from(pic_irq) < ICH9_LPC_PIC_NUM_PINS {
                PciIntxRoute { mode: PciIntxMode::Enabled, irq: i32::from(pic_irq) }
            } else {
                PciIntxRoute { mode: PciIntxMode::Disabled, irq: -1 }
            }
        } else {
            // Strictly speaking the PIRQ should go to both the I/O APIC and the 8259, on
            // different pins. QEMU and the KVM irqchip cannot express pin numbers that differ
            // between the two, so it goes to the fixed I/O APIC input only when routing to the
            // 8259 is disabled. Linux with 'noapic' followed by a kexec into an APIC kernel still
            // works, because the new kernel explicitly disables the PIRQ routing.
            PciIntxRoute { mode: PciIntxMode::Enabled, irq: ich9_pirq_to_gsi(pirq) as i32 }
        }
    }

    /// `ich9_set_sci()`, where the PM block's SCI output lands.
    fn set_sci(&self, level: bool) {
        let gsi = {
            let mut s = lock(&self.state);
            if level == s.sci_level {
                return;
            }
            s.sci_level = level;
            usize::from(s.sci_gsi)
        };
        if gsi >= ICH9_LPC_PIC_NUM_PINS {
            self.update_apic(gsi);
        } else {
            self.update_pic(gsi);
        }
    }

    /// `ich9_lpc_pmbase_sci_update()`: moves the PM window and switches the SCI line, lowering
    /// the old GSI and raising the new one if the SCI is asserted.
    fn pmbase_sci_update(&self, dev: &PciDevice) {
        let (pmbase, acpi_cntl) = dev.with_config(|c| {
            (
                pci_get_long(c.config, ICH9_LPC_PMBASE as usize),
                c.config[ICH9_LPC_ACPI_CTRL as usize],
            )
        });
        let Some(new_gsi) = self.pm.lpc_config_update(pmbase, acpi_cntl) else {
            return;
        };
        let new_gsi = new_gsi as u8;
        let switch = {
            let mut s = lock(&self.state);
            let switch = s.sci_level && new_gsi != s.sci_gsi;
            if !switch {
                s.sci_gsi = new_gsi;
            }
            switch
        };
        if switch {
            self.set_sci(false);
            lock(&self.state).sci_gsi = new_gsi;
            self.set_sci(true);
        }
    }

    /// `ich9_lpc_rcba_update()`.
    fn rcba_update(&self, dev: &PciDevice, rcba_old: u32) {
        let rcba = dev.with_config(|c| pci_get_long(c.config, ICH9_LPC_RCBA));
        let rcrb = self.regions().rcrb;
        let m = &self.memory;
        let _t = m.transaction();
        if rcba_old & ICH9_LPC_RCBA_EN != 0 {
            // Only fails if it was not mapped, which leaves nothing to undo.
            let _ = m.del_subregion(self.system_memory, rcrb);
        }
        if rcba & ICH9_LPC_RCBA_EN != 0 {
            let base = u64::from(rcba & ICH9_LPC_RCBA_BA_MASK);
            if let Err(e) = m.add_subregion_overlap(self.system_memory, base, rcrb, 1) {
                panic!("ich9-lpc: mapping the RCRB failed: {e}");
            }
        }
    }

    /// `ich9_lpc_pmcon_update()`. Once SMI_LOCK is set it cannot be cleared.
    fn pmcon_update(&self, dev: &PciDevice) {
        dev.with_config(|c| {
            let gen_pmcon_1 = pci_get_word(c.config, ICH9_LPC_GEN_PMCON_1);
            if gen_pmcon_1 & ICH9_LPC_GEN_PMCON_1_SMI_LOCK != 0 {
                let wmask = pci_get_word(c.wmask, ICH9_LPC_GEN_PMCON_1);
                pci_set_word(c.wmask, ICH9_LPC_GEN_PMCON_1, wmask & !ICH9_LPC_GEN_PMCON_1_SMI_LOCK);
            }
        });
    }

    /// `ich9_cc_read()`: little endian, `len` bytes at `addr`.
    fn cc_read(&self, addr: u64, len: u32) -> u64 {
        let (addr, len) = cc_addr_len(addr, len);
        let s = lock(&self.state);
        let mut b = [0u8; 8];
        b[..len].copy_from_slice(&s.chip_config[addr..addr + len]);
        u64::from_le_bytes(b)
    }

    /// `ich9_cc_write()`.
    fn cc_write(&self, addr: u64, val: u64, len: u32) {
        let (addr, len) = cc_addr_len(addr, len);
        {
            let mut s = lock(&self.state);
            s.chip_config[addr..addr + len].copy_from_slice(&val.to_le_bytes()[..len]);
        }
        self.fire_intx_routing_notifier();
        lock(&self.state).cc_update();
    }

    /// `ich9_rst_cnt_write()`: bit 2 requests a reset, bits 1 (SYS_RST) and 3 (FULL_RST) are
    /// kept.
    fn rst_cnt_write(&self, val: u8) {
        if val & 4 != 0 {
            if let Some(h) = read(&self.request) {
                h(SystemRequest::Reset);
            }
            return;
        }
        lock(&self.state).rst_cnt = val & 0xa;
    }

    /// `ich9_apm_ctrl_changed()`.
    fn apm_ctrl_changed(&self, val: u8) {
        // ACPI specs 3.0, 4.7.2.5.
        self.pm.acpi().pm1_cnt_update(val == ICH9_APM_ACPI_ENABLE, val == ICH9_APM_ACPI_DISABLE);
        if val == ICH9_APM_ACPI_ENABLE || val == ICH9_APM_ACPI_DISABLE {
            return;
        }
        // SMI_EN = PMBASE + 30, the SMI control and enable register.
        if self.pm.smi_en() & ICH9_PMIO_SMI_EN_APMC_EN != 0 {
            let broadcast = lock(&self.state).smi_negotiated_features
                & (1 << ICH9_LPC_SMI_F_BROADCAST_BIT)
                != 0;
            if let Some(h) = read(&self.smi) {
                h(if broadcast { SmiTarget::Broadcast } else { SmiTarget::Current });
            }
        }
    }

    /// `apm_ioport_writeb()`.
    fn apm_write(&self, addr: u64, val: u8) {
        if addr & 1 == 0 {
            lock(&self.state).apmc = val;
            self.apm_ctrl_changed(val);
        } else {
            lock(&self.state).apms = val;
        }
    }

    /// `apm_ioport_readb()`.
    fn apm_read(&self, addr: u64) -> u8 {
        let s = lock(&self.state);
        if addr & 1 == 0 { s.apmc } else { s.apms }
    }
}

impl PciDeviceOps for Inner {
    /// `ich9_lpc_config_write()`.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        let rcba_old = dev.with_config(|c| pci_get_long(c.config, ICH9_LPC_RCBA));
        dev.default_write_config(addr, val, len);
        if ranges_overlap(addr, len, ICH9_LPC_PMBASE as usize, 4)
            || ranges_overlap(addr, len, ICH9_LPC_ACPI_CTRL as usize, 1)
        {
            self.pmbase_sci_update(dev);
        }
        if ranges_overlap(addr, len, ICH9_LPC_RCBA, 4) {
            self.rcba_update(dev, rcba_old);
        }
        if ranges_overlap(addr, len, ICH9_LPC_PIRQA_ROUT, 4) {
            self.fire_intx_routing_notifier();
        }
        if ranges_overlap(addr, len, ICH9_LPC_PIRQE_ROUT, 4) {
            self.fire_intx_routing_notifier();
        }
        if ranges_overlap(addr, len, ICH9_LPC_GEN_PMCON_1, 8) {
            self.pmcon_update(dev);
        }
    }

    /// `pm_reset()` followed by `ich9_lpc_reset()`.
    fn reset(&self, dev: &PciDevice) {
        self.pm.reset();

        let rcba_old = dev.with_config(|c| {
            let rcba_old = pci_get_long(c.config, ICH9_LPC_RCBA);
            for pirq in 0..ICH9_LPC_NB_PIRQS {
                c.config[pirq_rout_reg(pirq)] = ICH9_LPC_PIRQ_ROUT_DEFAULT;
            }
            c.config[ICH9_LPC_ACPI_CTRL as usize] = ICH9_LPC_ACPI_CTRL_DEFAULT;
            pci_set_long(c.config, ICH9_LPC_PMBASE as usize, ICH9_LPC_PMBASE_DEFAULT);
            pci_set_long(c.config, ICH9_LPC_RCBA, ICH9_LPC_RCBA_DEFAULT);
            rcba_old
        });

        lock(&self.state).cc_reset(self.noreboot);

        self.pmbase_sci_update(dev);
        self.rcba_update(dev, rcba_old);

        let mut s = lock(&self.state);
        s.sci_level = false;
        s.rst_cnt = 0;
        s.smi_guest_features_le = [0; 8];
        s.smi_features_ok = 0;
        s.smi_negotiated_features = 0;
    }
}

impl fmt::Debug for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = lock(&self.state);
        f.debug_struct("Ich9Lpc")
            .field("sci_gsi", &s.sci_gsi)
            .field("sci_level", &s.sci_level)
            .field("rst_cnt", &s.rst_cnt)
            .field("regions", &self.regions.get())
            .finish_non_exhaustive()
    }
}

/// `rcrb_mmio_ops`: the chipset config registers.
#[derive(Debug)]
struct RcrbOps(Weak<Inner>);

impl MmioOps for RcrbOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.upgrade().map_or(0, |l| l.cc_read(offset, size.bytes())))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if let Some(l) = self.0.upgrade() {
            l.cc_write(offset, value, size.bytes());
        }
        Ok(())
    }
}

/// `ich9_rst_cnt_ops`.
#[derive(Debug)]
struct RstCntOps(Weak<Inner>);

impl MmioOps for RstCntOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.upgrade().map_or(0, |l| u64::from(lock(&l.state).rst_cnt)))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        if let Some(l) = self.0.upgrade() {
            l.rst_cnt_write(value as u8);
        }
        Ok(())
    }
}

/// `apm_ops`: byte accesses only, wider ones are split.
#[derive(Debug)]
struct ApmOps(Weak<Inner>);

impl MmioOps for ApmOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.upgrade().map_or(0, |l| u64::from(l.apm_read(offset))))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        if let Some(l) = self.0.upgrade() {
            l.apm_write(offset, value as u8);
        }
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

/// The ICH9 LPC bridge, `ICH9LPCState`.
pub struct Ich9Lpc {
    inner: Arc<Inner>,
    dev: Arc<PciDevice>,
}

impl fmt::Debug for Ich9Lpc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ich9Lpc").field("inner", &self.inner).field("dev", &self.dev).finish()
    }
}

impl Ich9Lpc {
    /// Creates the bridge at 00:1f.0 on `bus`, which must be the root bus, and takes over its
    /// INTx routing. `gsi` are the 24 GSI lines of the machine. `clock` drives the PM timer.
    pub fn new(
        memory: Arc<MemorySystem>,
        bus: &Arc<PciBus>,
        clock: Arc<Clock>,
        config: Ich9LpcConfig,
        gsi: &[IrqLine],
    ) -> Result<Ich9Lpc, Error> {
        let hotplug = 1u64 << ICH9_LPC_SMI_F_CPU_HOTPLUG_BIT;
        let hot_unplug = 1u64 << ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT;
        if config.smi_host_features & hot_unplug != 0 && config.smi_host_features & hotplug == 0 {
            // smi_features_ok_callback() rejects this, so refuse to advertise it.
            return Err(Error::generic("cpu hot-unplug requires cpu hot-plug"));
        }
        if gsi.len() != ICH9_LPC_IOAPIC_NUM_PINS {
            return Err(Error::generic(format!(
                "ich9-lpc: {} GSI lines given, {ICH9_LPC_IOAPIC_NUM_PINS} needed",
                gsi.len()
            )));
        }
        let err = |e: MemError| Error::generic(format!("ich9-lpc: {e}"));

        let pm = Ich9Pm::new(clock, config.pm);
        let mut state = State {
            irr: [[0; PCI_NUM_PINS]; PCI_SLOT_MAX as usize],
            chip_config: vec![0; ICH9_CC_SIZE],
            sci_level: false,
            sci_gsi: 0,
            rst_cnt: 0,
            apmc: 0,
            apms: 0,
            smi_guest_features_le: [0; 8],
            smi_features_ok: 0,
            smi_negotiated_features: 0,
        };
        state.cc_init();
        let inner = Arc::new(Inner {
            memory: Arc::clone(&memory),
            system_memory: config.system_memory,
            io: config.address_space_io,
            pm: Arc::clone(&pm),
            gsi: gsi.to_vec(),
            noreboot: config.noreboot,
            smi_host_features: config.smi_host_features,
            bus: Arc::downgrade(bus),
            dev: OnceLock::new(),
            regions: OnceLock::new(),
            state: Mutex::new(state),
            request: RwLock::new(None),
            smi: RwLock::new(None),
            intx_notifier: RwLock::new(None),
        });
        let weak = Arc::downgrade(&inner);

        let info = PciDeviceInfo {
            name: TYPE_ICH9_LPC_DEVICE.to_string(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: PCI_DEVICE_ID_INTEL_ICH9_8,
            revision: ICH9_A2_LPC_REVISION,
            class_id: PCI_CLASS_BRIDGE_ISA,
            multifunction: true,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, Some(pci_devfn(ICH9_LPC_DEV, ICH9_LPC_FUNC)))?;
        let _ = inner.dev.set(Arc::downgrade(&dev));

        // The write mask of PMBASE. QEMU then overwrites its low byte with the ACPI_CNTL bits,
        // which leaves ACPI_CNTL fully writable and bits 0 to 2 of PMBASE writable too.
        dev.with_config(|c| {
            pci_set_long(c.wmask, ICH9_LPC_PMBASE as usize, ICH9_LPC_PMBASE_BASE_ADDRESS_MASK);
            c.wmask[ICH9_LPC_PMBASE as usize] =
                ICH9_LPC_ACPI_CTRL_ACPI_EN | ICH9_LPC_ACPI_CTRL_SCI_IRQ_SEL_MASK;
        });

        let rcrb = memory
            .new_io("lpc-rcrb-mmio", ICH9_CC_SIZE as u128, Arc::new(RcrbOps(weak.clone())))
            .map_err(err)?;

        // apm_init(): 0xb2 and 0xb3.
        let apm = memory.new_io("apm-io", 2, Arc::new(ApmOps(weak.clone()))).map_err(err)?;
        memory.add_subregion(config.address_space_io, APM_CNT_IOPORT, apm).map_err(err)?;

        let rst_cnt = memory
            .new_io("lpc-reset-control", 1, Arc::new(RstCntOps(weak.clone())))
            .map_err(err)?;
        memory
            .add_subregion_overlap(config.address_space_io, ICH9_RST_CNT_IOPORT, rst_cnt, 1)
            .map_err(err)?;

        bus.set_irqs(
            {
                let w = weak.clone();
                Arc::new(move |pirq, _level| {
                    if let Some(l) = w.upgrade() {
                        l.set_irq(pirq as usize);
                    }
                })
            },
            ICH9_LPC_NB_PIRQS,
        );
        bus.set_map_irq({
            let w = weak.clone();
            Arc::new(move |devfn, intx| w.upgrade().map_or(intx, |l| l.map_irq(devfn, intx)))
        });

        // ich9_lpc_pm_init().
        let pm_region = pm.map(&memory, config.address_space_io).map_err(err)?;
        pm.sci().connect(IrqLine::from_fn(move |level| {
            if let Some(l) = weak.upgrade() {
                l.set_sci(level != 0);
            }
        }));

        let _ = inner.regions.set(Regions { rcrb, rst_cnt, apm, pm: pm_region });
        dev.set_ops(Arc::clone(&inner) as Arc<dyn PciDeviceOps>);
        Ok(Ich9Lpc { inner, dev })
    }

    /// The PCI function, 00:1f.0.
    pub fn device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The power management block behind PMBASE.
    pub fn pm(&self) -> &Arc<Ich9Pm> {
        &self.inner.pm
    }

    /// `ich9_lpc_reset()` plus the generic PCI reset, as a bus reset does it.
    pub fn reset(&self) {
        self.dev.reset();
    }

    /// Where reset requests from 0xcf9 and shutdown and suspend requests from PM1_CNT go.
    pub fn set_request_handler(&self, handler: SystemRequestHandler) {
        self.inner.pm.set_request_handler(Arc::clone(&handler));
        write(&self.inner.request, Some(handler));
    }

    /// Where SMIs raised by an APM command go.
    pub fn set_smi_handler(&self, handler: Option<SmiHandler>) {
        write(&self.inner.smi, handler);
    }

    /// Called whenever the PIRQ routing or the chipset config registers are written.
    pub fn set_intx_routing_notifier(&self, notifier: Option<IntxRoutingNotifier>) {
        write(&self.inner.intx_notifier, notifier);
    }

    /// `ich9_lpc_map_irq()`: the PIRQ (0 is PIRQA) pin `intx` of the function at `devfn` is
    /// wired to.
    pub fn map_irq(&self, devfn: u8, intx: usize) -> usize {
        self.inner.map_irq(devfn, intx as i32) as usize
    }

    /// `ich9_lpc_pic_irq()`: the 8259 input PIRQx_ROUT selects for `pirq`, and whether routing
    /// to the 8259 is disabled.
    pub fn pic_irq(&self, pirq: usize) -> (u8, bool) {
        self.inner.pic_irq(pirq)
    }

    /// `ich9_route_intx_pin_to_irq()`.
    pub fn route_intx_pin_to_irq(&self, pirq: usize) -> PciIntxRoute {
        assert!(pirq < ICH9_LPC_NB_PIRQS, "PIRQ {pirq} out of range");
        self.inner.route_intx_pin_to_irq(pirq)
    }

    /// The GSI the SCI goes to, the `sci-int` property.
    pub fn sci_gsi(&self) -> u8 {
        lock(&self.inner.state).sci_gsi
    }

    /// The SCI level the bridge last saw.
    pub fn sci_level(&self) -> bool {
        lock(&self.inner.state).sci_level
    }

    /// The reset control register at 0xcf9.
    pub fn rst_cnt(&self) -> u8 {
        lock(&self.inner.state).rst_cnt
    }

    /// The APM control and status registers at 0xb2 and 0xb3.
    pub fn apm(&self) -> (u8, u8) {
        let s = lock(&self.inner.state);
        (s.apmc, s.apms)
    }

    /// `ich9_cc_read()`: `len` bytes of the chipset config registers at `addr`.
    pub fn cc_read(&self, addr: u64, len: u32) -> u64 {
        self.inner.cc_read(addr, len)
    }

    /// `ich9_cc_write()`.
    pub fn cc_write(&self, addr: u64, val: u64, len: u32) {
        self.inner.cc_write(addr, val, len);
    }

    /// The "lpc-rcrb-mmio" region.
    pub fn rcrb_region(&self) -> RegionId {
        self.inner.regions().rcrb
    }

    /// The "lpc-reset-control" region.
    pub fn rst_cnt_region(&self) -> RegionId {
        self.inner.regions().rst_cnt
    }

    /// The "apm-io" region.
    pub fn apm_region(&self) -> RegionId {
        self.inner.regions().apm
    }

    /// The "ich9-pm" container.
    pub fn pm_region(&self) -> RegionId {
        self.inner.regions().pm
    }

    /// `ich9_lpc_machine_ready()`: flags the legacy devices found in the I/O space (COM1, COM2,
    /// LPT and floppy) in config byte 0x82. Call it once the board is complete.
    pub fn machine_ready(&self) {
        let io = self.inner.io;
        let Ok(view) = self.inner.memory.render(io) else { return };
        let mut bits = 0u8;
        for (port, bit) in [(0x3f8, 0x01), (0x2f8, 0x02), (0x378, 0x04), (0x3f2, 0x08)] {
            if view.lookup(port).is_some() {
                bits |= bit;
            }
        }
        self.dev.with_config(|c| c.config[0x82] |= bits);
    }

    /// The host side SMI features, the `x-smi-*` properties.
    pub fn smi_host_features(&self) -> u64 {
        self.inner.smi_host_features
    }

    /// The contents of the `etc/smi/supported-features` fw_cfg file.
    pub fn smi_host_features_le(&self) -> [u8; 8] {
        self.inner.smi_host_features.to_le_bytes()
    }

    /// What the guest wrote to `etc/smi/requested-features`.
    pub fn smi_guest_features_le(&self) -> [u8; 8] {
        lock(&self.inner.state).smi_guest_features_le
    }

    /// A guest write to `etc/smi/requested-features`.
    pub fn set_smi_guest_features(&self, le: [u8; 8]) {
        lock(&self.inner.state).smi_guest_features_le = le;
    }

    /// `smi_features_ok_callback()`, run when the guest selects `etc/smi/features-ok`: locks
    /// the requested features in if they are a valid subset of the host's.
    pub fn smi_features_ok_select(&self) {
        let host = self.inner.smi_host_features;
        let mut s = lock(&self.inner.state);
        if s.smi_features_ok != 0 {
            // Negotiation already complete, features locked.
            return;
        }
        let guest = u64::from_le_bytes(s.smi_guest_features_le);
        if guest & !host != 0 {
            return;
        }
        let hotplug = 1u64 << ICH9_LPC_SMI_F_CPU_HOTPLUG_BIT;
        let hot_unplug = 1u64 << ICH9_LPC_SMI_F_CPU_HOT_UNPLUG_BIT;
        let guest_hotplug = guest & (hotplug | hot_unplug);
        if guest & (1 << ICH9_LPC_SMI_F_BROADCAST_BIT) == 0 && guest_hotplug != 0 {
            // CPU hot-(un)plug with SMI needs SMI broadcast.
            return;
        }
        if guest_hotplug == hot_unplug {
            // CPU hot-unplug is unsupported without CPU hotplug.
            return;
        }
        s.smi_negotiated_features = guest;
        s.smi_features_ok = 1;
    }

    /// The contents of `etc/smi/features-ok`.
    pub fn smi_features_ok(&self) -> u8 {
        lock(&self.inner.state).smi_features_ok
    }

    /// `x-smi-negotiated-features`.
    pub fn smi_negotiated_features(&self) -> u64 {
        lock(&self.inner.state).smi_negotiated_features
    }
}
