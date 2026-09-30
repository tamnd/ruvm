// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the generic PCIe root port: config space layout, the resource reserve capability,
//! native hotplug with MSI-X and INTx, and devices behind the port through ECAM on q35.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_hw_pci::pcie::*;
use ruvm_hw_pci::q35::*;
use ruvm_hw_pci::regs::*;
use ruvm_hw_pci::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const EXP: u32 = 0x54;
const AER: u32 = 0x100;
const ACS: u32 = 0x148;

type Messages = Arc<Mutex<Vec<(u64, u32)>>>;

struct Env {
    bus: Arc<PciBus>,
    levels: Arc<[AtomicI32; 4]>,
    msgs: Messages,
}

impl Env {
    fn new() -> Env {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let bus = PciBus::new_root("pcie.0", Arc::clone(&mem), sysmem, io, 0);
        bus.set_extended_config_space(true);
        let levels: Arc<[AtomicI32; 4]> = Arc::new(Default::default());
        let l = Arc::clone(&levels);
        bus.set_irqs(Arc::new(move |irq, level| l[irq as usize].store(level, Ordering::SeqCst)), 4);
        bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));
        let msgs: Messages = Arc::default();
        let m = Arc::clone(&msgs);
        bus.set_msi_handler(Some(Arc::new(move |a, d| m.lock().unwrap().push((a, d)))));
        Env { bus, levels, msgs }
    }

    fn port(&self, cfg: &PcieRootPortConfig) -> PcieRootPort {
        let port = PcieRootPort::new(&self.bus, cfg, Some(pci_devfn(2, 0))).unwrap();
        port.device().reset();
        port
    }

    fn take_msgs(&self) -> Vec<(u64, u32)> {
        std::mem::take(&mut *self.msgs.lock().unwrap())
    }
}

fn rd(dev: &PciDevice, off: u32, len: u32) -> u32 {
    dev.config_read(off, len)
}

fn wr(dev: &PciDevice, off: u32, len: u32, v: u32) {
    dev.config_write(off, v, len);
}

fn endpoint(name: &str) -> PciDeviceInfo {
    PciDeviceInfo {
        name: name.to_string(),
        vendor_id: 0x8086,
        device_id: 0x10d3,
        class_id: 0x0200,
        express: true,
        ..Default::default()
    }
}

fn caps(dev: &PciDevice) -> Vec<(u8, u8)> {
    let cfg = dev.config_bytes();
    let mut out = Vec::new();
    let mut p = cfg[PCI_CAPABILITY_LIST];
    while p != 0 {
        out.push((p, cfg[usize::from(p)]));
        p = cfg[usize::from(p) + 1];
    }
    out
}

#[test]
fn config_space_layout() {
    let env = Env::new();
    let cfg = PcieRootPortConfig { port: 0x10, slot: 3, chassis: 1, ..Default::default() };
    let port = env.port(&cfg);
    let d = port.device();

    assert_eq!(rd(d, 0, 4), 0x000c_1b36);
    assert_eq!(rd(d, PCI_CLASS_DEVICE as u32, 2), 0x0604);
    assert_eq!(rd(d, PCI_HEADER_TYPE as u32, 1), 1);
    assert_eq!(rd(d, PCI_INTERRUPT_PIN as u32, 1), 1);
    // PCIe ports do not claim 66 MHz or fast back to back; only the list bit remains.
    assert_eq!(rd(d, PCI_STATUS as u32, 2), u32::from(PCI_STATUS_CAP_LIST));
    assert_eq!(rd(d, PCI_SEC_STATUS as u32, 2), 0);
    assert_eq!(
        caps(d),
        vec![(0x54, PCI_CAP_ID_EXP), (0x48, PCI_CAP_ID_MSIX), (0x40, PCI_CAP_ID_SSVID)]
    );
    assert_eq!(port.exp_cap(), 0x54);
    assert_eq!(port.reserve_cap(), None);

    // Subsystem vendor ID capability.
    assert_eq!(rd(d, 0x44, 2), 0x1b36);
    assert_eq!(rd(d, 0x46, 2), 0);
    // One MSI-X vector in BAR 0.
    assert_eq!(rd(d, 0x4a, 2), 0);
    assert_eq!(rd(d, 0x4c, 4), 0);
    assert_eq!(rd(d, 0x50, 4), 0x800);
    assert_eq!(d.msix_exclusive_bar(), Some(0));

    // PCI Express capability, version 2, root port with a slot.
    assert_eq!(rd(d, EXP + 2, 2), 0x0142);
    assert_eq!(rd(d, EXP + 4, 4), PCI_EXP_DEVCAP_RBER | PCI_EXP_DEVCAP_EXT_TAG);
    // Port 0x10, L0s, x32, 16GT/s, bandwidth notification and DLL link active reporting.
    assert_eq!(rd(d, EXP + 0xc, 4), 0x1030_0604);
    // Nothing plugged: the link status mirrors the port's own maximum.
    assert_eq!(rd(d, EXP + 0x12, 2), 0x0204);
    assert_eq!(rd(d, EXP + 0x30, 2), 4, "target link speed 16GT/s");
    assert_eq!(rd(d, EXP + 0x2c, 4), 0x1e, "2.5 to 16GT/s supported");
    assert_eq!(rd(d, EXP + 0x14, 4), (3 << 19) | 0x2_007b);
    // After reset an empty slot has its power controller and both indicators off.
    assert_eq!(rd(d, EXP + 0x18, 2), 0x07c0);
    assert_eq!(rd(d, EXP + 0x1a, 2), 0);
    assert_eq!(rd(d, EXP + 0x24, 4), 0x30_0020, "EFF, EETLPP and ARI forwarding");

    let wmask = d.wmask_bytes();
    let w1c = d.w1cmask_bytes();
    let e = EXP as usize;
    assert_eq!(pci_get_word(&wmask, e + PCI_EXP_SLTCTL), 0x0ff9);
    assert_eq!(pci_get_word(&w1c, e + PCI_EXP_SLTSTA), 0x19);
    assert_eq!(pci_get_word(&wmask, e + PCI_EXP_DEVCTL), 0xf);
    assert_eq!(pci_get_word(&w1c, e + PCI_EXP_DEVSTA), 0xf);
    assert_eq!(pci_get_word(&wmask, e + PCI_EXP_RTCTL), 0x7);
    assert_eq!(pci_get_word(&wmask, e + PCI_EXP_DEVCTL2), 0x8020);
    assert_eq!(pci_get_word(&wmask, PCI_BRIDGE_CONTROL) & 0xfa0, 0);
    assert_ne!(pci_get_word(&wmask, PCI_BRIDGE_CONTROL) & PCI_BRIDGE_CTL_SERR, 0);
    assert_eq!(pci_get_word(&d.cmask_bytes(), e + PCI_EXP_LNKSTA), 0);

    // Extended capabilities: AER then ACS.
    assert_eq!(rd(d, AER, 4), 0x1482_0001);
    assert_eq!(pcie_find_capability(d, PCI_EXT_CAP_ID_ERR), 0x100);
    assert_eq!(pcie_find_capability(d, PCI_EXT_CAP_ID_ACS), 0x148);
    assert_eq!(pcie_find_capability(d, PCI_EXT_CAP_ID_ARI), 0);
    assert_eq!(rd(d, AER + 0x8, 4), 0x0240_0000);
    assert_eq!(rd(d, AER + 0xc, 4), 0x0046_2030);
    assert_eq!(rd(d, AER + 0x14, 4), 0xe000);
    assert_eq!(rd(d, AER + 0x18, 4), 0x2a0);
    assert_eq!(pci_get_long(&wmask, 0x100 + PCI_ERR_ROOT_COMMAND), 7);
    assert_eq!(pci_get_long(&w1c, 0x100 + PCI_ERR_ROOT_STATUS), 0x7f);
    assert_eq!(rd(d, ACS, 4), 0x0001_000d);
    assert_eq!(rd(d, ACS + 4, 2), 0x5f);
    assert_eq!(pci_get_word(&wmask, 0x148 + PCI_ACS_CTRL), 0x5f);
    // The first extended header is read-only.
    wr(d, AER, 4, 0);
    assert_eq!(rd(d, AER, 4), 0x1482_0001);

    // Writable control registers are cleared by reset.
    wr(d, ACS + 6, 2, 0xffff);
    wr(d, EXP + 0x8, 2, 0xffff);
    wr(d, EXP + 0x1c, 2, 0xffff);
    wr(d, EXP + 0x28, 2, 0xffff);
    wr(d, AER + 0x2c, 4, 0xffff_ffff);
    assert_eq!(rd(d, ACS + 6, 2), 0x5f);
    assert_eq!(rd(d, EXP + 0x1c, 2), 7);
    assert_eq!(rd(d, AER + 0x2c, 4), 7);
    d.reset();
    assert_eq!(rd(d, ACS + 6, 2), 0);
    assert_eq!(rd(d, EXP + 0x8, 2) & 0xf, 0);
    assert_eq!(rd(d, EXP + 0x1c, 2), 0);
    assert_eq!(rd(d, EXP + 0x28, 2) & 0x20, 0);
    assert_eq!(rd(d, AER + 0x2c, 4), 0);
}

#[test]
fn link_speed_and_width_properties() {
    let env = Env::new();
    let cfg = PcieRootPortConfig {
        speed: PcieLinkSpeed::Gt2_5,
        width: PcieLinkWidth::X1,
        ..Default::default()
    };
    let port = env.port(&cfg);
    let d = port.device();
    // A plain 2.5GT/s x1 port has neither bandwidth notification nor DLLLARC.
    assert_eq!(rd(d, EXP + 0xc, 4), 0x0411);
    assert_eq!(rd(d, EXP + 0x30, 2), 0);
    assert_eq!(rd(d, EXP + 0x2c, 4), 0);

    let env = Env::new();
    let cfg = PcieRootPortConfig {
        speed: PcieLinkSpeed::Gt8,
        width: PcieLinkWidth::X4,
        ..Default::default()
    };
    let port = env.port(&cfg);
    let d = port.device();
    assert_eq!(rd(d, EXP + 0xc, 4), 0x0030_0443);
    assert_eq!(rd(d, EXP + 0x2c, 4), 0xe);

    // The link status follows the device in the slot, clamped to the port.
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    let cap = pcie_cap_init(&child, 0, PCI_EXP_TYPE_ENDPOINT, 0).unwrap();
    child.with_config(|c| pci_set_word(c.config, usize::from(cap) + PCI_EXP_LNKSTA, 0x0102));
    port.plug(&child, false);
    // x16 at 5GT/s is clamped to x4, and DLLLA is set by the plug.
    assert_eq!(rd(d, EXP + 0x12, 2), 0x2042);
    child.with_config(|c| pci_set_word(c.config, usize::from(cap) + PCI_EXP_LNKSTA, 0x0025));
    assert_eq!(rd(d, EXP + 0x12, 2), 0x2023, "x2 at 8GT/s, clamped from 32GT/s");
}

#[test]
fn resource_reserve_capability() {
    let env = Env::new();
    let res = PciResReserve {
        bus: Some(3),
        io: Some(0x2000),
        mem_non_pref: Some(0x10_0000),
        mem_pref_64: Some(0x1_0000_0000),
        ..Default::default()
    };
    let cfg = PcieRootPortConfig { res_reserve: res, ..Default::default() };
    let port = env.port(&cfg);
    let d = port.device();
    assert_eq!(port.reserve_cap(), Some(0x90));
    assert_eq!(caps(d)[0], (0x90, PCI_CAP_ID_VNDR));
    assert_eq!(rd(d, 0x92, 1), 32);
    assert_eq!(rd(d, 0x93, 1), 1);
    assert_eq!(rd(d, 0x94, 4), 3);
    assert_eq!(rd(d, 0x98, 4), 0x2000);
    assert_eq!(rd(d, 0x9c, 4), 0);
    assert_eq!(rd(d, 0xa0, 4), 0x10_0000);
    assert_eq!(rd(d, 0xa4, 4), 0xffff_ffff, "pref32 unset");
    assert_eq!(rd(d, 0xa8, 4), 0);
    assert_eq!(rd(d, 0xac, 4), 1);

    // A zero I/O reserve makes the I/O window read-only.
    let env = Env::new();
    let cfg = PcieRootPortConfig {
        res_reserve: PciResReserve { io: Some(0), ..Default::default() },
        ..Default::default()
    };
    let port = env.port(&cfg);
    let d = port.device();
    let wmask = d.wmask_bytes();
    assert_eq!(wmask[PCI_IO_BASE], 0);
    assert_eq!(wmask[PCI_IO_LIMIT], 0);
    assert_eq!(pci_get_word(&wmask, PCI_COMMAND) & PCI_COMMAND_IO, 0);
    assert_eq!(rd(d, 0x98, 4), 0);
    assert_eq!(rd(d, 0x94, 4), 0xffff_ffff);

    // The 6.1 compat knob hides native hotplug and reserves 4 KiB of I/O.
    let env = Env::new();
    let cfg = PcieRootPortConfig { hide_native_hotplug_cap: true, ..Default::default() };
    let port = env.port(&cfg);
    let d = port.device();
    assert_eq!(rd(d, EXP + 0x14, 4) & 0x60, 0);
    assert_eq!(rd(d, 0x98, 4), 4096);

    let bad = |res: PciResReserve| {
        let env = Env::new();
        let cfg = PcieRootPortConfig { res_reserve: res, ..Default::default() };
        let err = PcieRootPort::new(&env.bus, &cfg, None).unwrap_err().to_string();
        assert!(env.bus.devices().is_empty(), "a failed port leaves nothing behind");
        err
    };
    assert_eq!(
        bad(PciResReserve { mem_pref_32: Some(1), mem_pref_64: Some(1), ..Default::default() }),
        "PCI resource reserve cap: PREF32 and PREF64 conflict"
    );
    assert_eq!(
        bad(PciResReserve { mem_non_pref: Some(4 << 30), ..Default::default() }),
        "PCI resource reserve cap: mem-reserve must be less than 4G"
    );
    assert_eq!(
        bad(PciResReserve { mem_pref_32: Some(4 << 30), ..Default::default() }),
        "PCI resource reserve cap: pref32-reserve  must be less than 4G"
    );
}

#[test]
fn chassis_slots_are_unique() {
    let env = Env::new();
    let reg = Arc::new(PcieChassisRegistry::new());
    let cfg = PcieRootPortConfig {
        chassis: 1,
        slot: 2,
        chassis_registry: Some(Arc::clone(&reg)),
        ..Default::default()
    };
    let a = PcieRootPort::new(&env.bus, &cfg, Some(pci_devfn(1, 0))).unwrap();
    let err = PcieRootPort::new(&env.bus, &cfg, Some(pci_devfn(2, 0))).unwrap_err();
    assert_eq!(err.to_string(), "Can't add chassis slot, error -16");
    assert!(env.bus.device(pci_devfn(2, 0)).is_none());
    let other = PcieRootPortConfig { slot: 3, sec_bus_name: "pcie.2".into(), ..cfg.clone() };
    PcieRootPort::new(&env.bus, &other, Some(pci_devfn(2, 0))).unwrap();
    a.unrealize(&env.bus);
    assert!(!reg.contains(1, 2));
    PcieRootPort::new(&env.bus, &cfg, Some(pci_devfn(1, 0))).unwrap();

    // The port needs MSI-X.
    let mem = Arc::new(MemorySystem::new());
    let sysmem = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_container("io", 1 << 16).unwrap();
    let bus = PciBus::new_root("pcie.0", mem, sysmem, io, 0);
    assert!(PcieRootPort::new(&bus, &PcieRootPortConfig::default(), None).is_err());
    assert!(bus.devices().is_empty());
}

#[test]
fn bus_numbers_and_windows() {
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig::default());
    let d = port.device();
    // Reset leaves the windows closed: base above limit.
    assert_eq!(rd(d, PCI_MEMORY_BASE as u32, 2), 0xfff0);
    assert_eq!(rd(d, PCI_MEMORY_LIMIT as u32, 2), 0);
    wr(d, PCI_PRIMARY_BUS as u32, 4, 0x0005_0100);
    assert_eq!(rd(d, PCI_SECONDARY_BUS as u32, 1), 1);
    assert_eq!(rd(d, PCI_SUBORDINATE_BUS as u32, 1), 5);
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    assert!(Arc::ptr_eq(&env.bus.find_bus_nr(1).unwrap(), port.sec_bus()));
    assert!(Arc::ptr_eq(&env.bus.find_device(1, 0).unwrap(), &child));
    assert!(env.bus.find_bus_nr(2).is_none());

    wr(d, PCI_MEMORY_BASE as u32, 4, 0xfe10_fe00);
    wr(d, PCI_COMMAND as u32, 2, u32::from(PCI_COMMAND_MEMORY));
    let w = port.bridge().mem_window();
    assert_eq!((w.base, w.size), (0xfe00_0000, 0x20_0000));
}

/// Sets up the port's MSI-X vector the way a guest would.
fn enable_msix(port: &PcieRootPort) {
    let d = port.device();
    d.msix_set_message(0, MsiMessage { address: 0xfee0_0000, data: 0x41 });
    d.msix_set_mask(0, false);
    wr(d, PCI_COMMAND as u32, 2, u32::from(PCI_COMMAND_MASTER));
    wr(d, 0x4b, 1, 0x80);
    assert!(d.msix_enabled());
}

const SLTCTL: u32 = EXP + 0x18;
const SLTSTA: u32 = EXP + 0x1a;
const HP_ENABLES: u32 = 0x39;

#[test]
fn hotplug_with_msix() {
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig { id: Some("rp0".into()), ..Default::default() });
    let d = port.device();
    enable_msix(&port);
    let removed: Arc<Mutex<Vec<u8>>> = Arc::default();
    let r = Arc::clone(&removed);
    port.set_unplug_notifier(Some(Arc::new(move |dev: &Arc<PciDevice>| {
        r.lock().unwrap().push(dev.devfn());
    })));

    // The guest enables hotplug interrupts; the command completes at once, but the command
    // completion enable is what lets it interrupt.
    wr(d, SLTCTL, 2, 0x07c0 | HP_ENABLES);
    assert_eq!(rd(d, SLTSTA, 2), u32::from(PCI_EXP_SLTSTA_CC));
    assert_eq!(env.take_msgs(), vec![(0xfee0_0000, 0x41)]);
    assert!(port.hotplug_event_notified());
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_CC));
    assert_eq!(rd(d, SLTSTA, 2), 0);
    assert!(!port.hotplug_event_notified());

    // Hot add: presence detect changed plus the attention button.
    port.pre_plug(true).unwrap();
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, true);
    assert_eq!(rd(d, SLTSTA, 2), 0x49);
    assert_ne!(rd(d, EXP + 0x12, 2) & u32::from(PCI_EXP_LNKSTA_DLLLA), 0);
    assert_eq!(env.take_msgs().len(), 1);
    // The slot power is still off, so the new function is disabled.
    assert!(!child.is_enabled());

    // Clearing only some events keeps the interrupt condition; a write that clears bits that
    // were not set gets them put back.
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_ABP));
    assert_eq!(rd(d, SLTSTA, 2), 0x48);
    wr(d, SLTSTA, 2, 0x1f);
    assert_eq!(rd(d, SLTSTA, 2), 0x48, "CC was not set, so the write is undone");
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_PDC));
    assert_eq!(rd(d, SLTSTA, 2), 0x40);
    env.take_msgs();

    // Power on with the power indicator on.
    wr(d, SLTCTL, 2, 0x01c0 | HP_ENABLES);
    assert!(child.is_enabled());
    assert_eq!(env.take_msgs().len(), 1, "command completed");
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_CC));

    // Blinking power indicator: the guest is busy.
    wr(d, SLTCTL, 2, 0x02c0 | HP_ENABLES);
    assert_eq!(
        port.unplug_request(&child).unwrap_err().to_string(),
        "Hot-unplug failed: guest is busy (power indicator blinking)"
    );
    wr(d, SLTCTL, 2, 0x01c0 | HP_ENABLES);
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_CC));
    env.take_msgs();

    // Unplug request: the attention button is pushed and the guest powers the slot off.
    port.unplug_request(&child).unwrap();
    assert_eq!(rd(d, SLTSTA, 2), 0x41);
    assert_eq!(env.take_msgs().len(), 1);
    assert!(port.sec_bus().device(0).is_some());
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_ABP));
    wr(d, SLTCTL, 2, 0x07c0 | HP_ENABLES);
    assert!(port.sec_bus().device(0).is_none());
    assert_eq!(*removed.lock().unwrap(), vec![0]);
    assert_eq!(rd(d, SLTSTA, 2), u32::from(PCI_EXP_SLTSTA_PDC | PCI_EXP_SLTSTA_CC));
    assert_eq!(rd(d, EXP + 0x12, 2) & u32::from(PCI_EXP_LNKSTA_DLLLA), 0);

    // A request on a powered off slot removes the function without asking the guest.
    wr(d, SLTSTA, 2, 0x18);
    let child = port.sec_bus().register_device(&endpoint("nic2"), Some(0)).unwrap();
    port.plug(&child, true);
    port.unplug_request(&child).unwrap();
    assert!(port.sec_bus().device(0).is_none());
    assert_eq!(rd(d, SLTSTA, 2) & 0x49, u32::from(PCI_EXP_SLTSTA_PDC));
}

#[test]
fn interlock_and_hotplug_disabled() {
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig::default());
    let d = port.device();
    // Writing 1 to the interlock control toggles the interlock; the bit reads as 0.
    wr(d, SLTCTL, 2, 0x07c0 | u32::from(PCI_EXP_SLTCTL_EIC));
    assert_eq!(rd(d, SLTCTL, 2), 0x07c0);
    assert_ne!(rd(d, SLTSTA, 2) & u32::from(PCI_EXP_SLTSTA_EIS), 0);
    assert_eq!(port.pre_plug(true).unwrap_err().to_string(), "slot is electromechanically locked");
    // Reset releases the lock.
    d.reset();
    port.pre_plug(true).unwrap();

    let env = Env::new();
    let cfg = PcieRootPortConfig { id: Some("rp1".into()), hotplug: false, ..Default::default() };
    let port = env.port(&cfg);
    assert_eq!(rd(port.device(), EXP + 0x14, 4) & 0x60, 0);
    assert_eq!(
        port.pre_plug(true).unwrap_err().to_string(),
        "Hot-plug failed: unsupported by the port device 'rp1'"
    );
    port.pre_plug(false).unwrap();
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, false);
    assert_eq!(
        port.unplug_request(&child).unwrap_err().to_string(),
        "Hot-unplug failed: unsupported by the port device 'rp1'"
    );
}

#[test]
fn cold_plug_and_power_controller() {
    let env = Env::new();
    let cfg = PcieRootPortConfig::default();
    let port = PcieRootPort::new(&env.bus, &cfg, Some(pci_devfn(2, 0))).unwrap();
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, false);
    let d = port.device();
    assert_eq!(rd(d, SLTSTA, 2), u32::from(PCI_EXP_SLTSTA_PDS), "no event for cold plug");
    // Machine reset: a populated slot comes up powered. The power indicator bits stay "off"
    // since QEMU ORs the "on" pattern into them.
    d.reset();
    assert_eq!(rd(d, SLTCTL, 2), 0x03c0);
    assert!(child.is_enabled());
    // The guest turns the power off, which also removes the device.
    wr(d, SLTCTL, 2, 0x07c0);
    assert!(port.sec_bus().device(0).is_none());

    // Without a power controller the slot is always powered.
    let env = Env::new();
    let cfg = PcieRootPortConfig { power_controller_present: false, ..Default::default() };
    let port = env.port(&cfg);
    let d = port.device();
    assert_eq!(rd(d, EXP + 0x14, 4) & 2, 0);
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, false);
    wr(d, SLTCTL, 2, 0x07c0);
    assert!(child.is_enabled());
    assert_eq!(pci_get_word(&d.wmask_bytes(), EXP as usize + PCI_EXP_SLTCTL) & 0x400, 0);

    // enable_power() turns the controller on without an event.
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig::default());
    port.enable_power();
    assert_eq!(rd(port.device(), SLTCTL, 2) & 0x400, 0);
}

#[test]
fn hotplug_with_intx() {
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig::default());
    let d = port.device();
    // QEMU's AER root code drives INTx from the (always idle) AER state at the end of every
    // config write, so the command completed interrupt raised by this write is dropped at once.
    wr(d, SLTCTL, 2, 0x07c0 | HP_ENABLES);
    assert!(port.hotplug_event_notified());
    assert_eq!(env.levels[2].load(Ordering::SeqCst), 0);
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_CC));
    assert!(!port.hotplug_event_notified());

    // A hot add is not a config write, so it does raise the line. Slot 2, INTA is PIRQ C
    // with the swizzle.
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, true);
    assert_eq!(env.levels[2].load(Ordering::SeqCst), 1);
    // Clearing the events lowers it.
    wr(d, SLTSTA, 2, u32::from(PCI_EXP_SLTSTA_PDC | PCI_EXP_SLTSTA_ABP));
    assert_eq!(env.levels[2].load(Ordering::SeqCst), 0);
    assert!(!port.hotplug_event_notified());
    // Any other write drops the line too while the event is still pending.
    port.push_attention_button();
    assert_eq!(env.levels[2].load(Ordering::SeqCst), 1);
    wr(d, PCI_COMMAND as u32, 2, 0);
    assert_eq!(env.levels[2].load(Ordering::SeqCst), 0);
    assert!(port.hotplug_event_notified());
    // A multi-function add with function 0 missing: no event until function 0 is there.
    let env = Env::new();
    let port = env.port(&PcieRootPortConfig::default());
    let d = port.device();
    wr(d, SLTCTL, 2, 0x07c0 | HP_ENABLES);
    wr(d, SLTSTA, 2, 0x10);
    let mf = PciDeviceInfo { multifunction: true, ..endpoint("f1") };
    let f1 = port.sec_bus().register_device(&mf, Some(1)).unwrap();
    port.plug(&f1, true);
    assert_eq!(rd(d, SLTSTA, 2), 0);
    // Cancelling the add removes the lone function directly.
    port.unplug_request(&f1).unwrap();
    assert!(port.sec_bus().device(1).is_none());
}

/// A q35 machine with ECAM at 0xb0000000 and a root port at 00:1c.0.
struct Q35 {
    io_as: Arc<AddressSpace>,
    mem_as: Arc<AddressSpace>,
    q35: Q35PciHost,
}

impl Q35 {
    const ECAM: u64 = 0xb000_0000;

    fn new() -> Q35 {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let pci = mem.new_container("pci", 1 << 64).unwrap();
        let ram = mem.new_ram("pc.ram", 128 << 20).unwrap();
        let low = mem.new_alias("ram-below-4g", ram, 0, 128 << 20).unwrap();
        mem.add_subregion(sysmem, 0, low).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        let mut cfg = Q35Config::new(ram, pci, sysmem, io);
        cfg.below_4g_mem_size = 128 << 20;
        let q35 = Q35PciHost::new(Arc::clone(&mem), cfg).unwrap();
        q35.bus().set_msi_nonbroken(true);
        q35.reset();
        let m = Q35 { io_as, mem_as, q35 };
        m.cfg_write(0, MCH_HOST_BRIDGE_PCIEXBAR as u32 + 4, 0);
        m.cfg_write(
            0,
            MCH_HOST_BRIDGE_PCIEXBAR as u32,
            (Q35::ECAM | MCH_HOST_BRIDGE_PCIEXBAREN) as u32,
        );
        assert_eq!(m.q35.mcfg_base(), Q35::ECAM);
        m
    }

    fn cfg_write(&self, devfn: u8, off: u32, v: u32) {
        let addr = 0x8000_0000 | (u32::from(devfn) << 8) | off;
        assert!(self.io_as.store(0xcf8, 4, addr.into(), Endian::Little, ATTRS).is_ok());
        assert!(self.io_as.store(0xcfc, 4, v.into(), Endian::Little, ATTRS).is_ok());
    }

    fn ecam(bus: u8, devfn: u8, off: u32) -> u64 {
        Q35::ECAM | (u64::from(bus) << 20) | (u64::from(devfn) << 12) | u64::from(off)
    }

    fn ecam_read(&self, bus: u8, devfn: u8, off: u32) -> u32 {
        self.mem_as.load(Q35::ecam(bus, devfn, off), 4, Endian::Little, ATTRS).0 as u32
    }

    fn ecam_write(&self, bus: u8, devfn: u8, off: u32, v: u32) {
        let r = self.mem_as.store(Q35::ecam(bus, devfn, off), 4, v.into(), Endian::Little, ATTRS);
        assert!(r.is_ok());
    }
}

#[test]
fn secondary_bus_through_q35_ecam() {
    let m = Q35::new();
    let rp_devfn = pci_devfn(0x1c, 0);
    let cfg = PcieRootPortConfig { chassis: 0, slot: 1, ..Default::default() };
    let port = PcieRootPort::new(m.q35.bus(), &cfg, Some(rp_devfn)).unwrap();
    let child = port.sec_bus().register_device(&endpoint("nic"), Some(0)).unwrap();
    port.plug(&child, false);
    // Machine reset comes after cold plug and powers the populated slot on.
    port.device().reset();
    assert!(child.is_enabled());

    // The port itself, including its extended space.
    assert_eq!(m.ecam_read(0, rp_devfn, 0), 0x000c_1b36);
    assert_eq!(m.ecam_read(0, rp_devfn, 0x100), 0x1482_0001);
    assert_eq!(m.ecam_read(0, rp_devfn, 0x148), 0x0001_000d);

    // Nothing answers on bus 1 until the bus numbers are programmed.
    assert_eq!(m.ecam_read(1, 0, 0), 0xffff_ffff);
    m.ecam_write(0, rp_devfn, PCI_PRIMARY_BUS as u32, 0x0001_0100);
    assert_eq!(m.ecam_read(0, rp_devfn, PCI_PRIMARY_BUS as u32) & 0xff_ffff, 0x01_0100);
    assert_eq!(m.ecam_read(1, 0, 0), 0x10d3_8086);
    assert_eq!(m.ecam_read(1, pci_devfn(1, 0), 0), 0xffff_ffff);
    assert_eq!(m.ecam_read(2, 0, 0), 0xffff_ffff);

    // Config writes reach the device behind the port, extended space included.
    child.with_config(|c| c.wmask[0x200..0x204].fill(0xff));
    m.ecam_write(1, 0, 0x200, 0xdead_beef);
    assert_eq!(m.ecam_read(1, 0, 0x200), 0xdead_beef);
    assert_eq!(child.config_read(0x200, 4), 0xdead_beef);

    // Slot status through ECAM: present, no events.
    assert_eq!(m.ecam_read(0, rp_devfn, EXP + 0x18) >> 16, u32::from(PCI_EXP_SLTSTA_PDS));
}
