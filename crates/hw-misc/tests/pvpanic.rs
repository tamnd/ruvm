// SPDX-License-Identifier: GPL-2.0-or-later

//! Ports of QEMU's pvpanic-test.c and pvpanic-pci-test.c. Where the qtests wait for a QMP event,
//! these check what the device handed to its event handler.

use std::sync::{Arc, Mutex};

use ruvm_hw_core::fw_cfg::{FwCfgProps, FwCfgState};
use ruvm_hw_misc::pvpanic::{
    PCI_CLASS_SYSTEM_OTHER, PCI_DEVICE_ID_REDHAT_PVPANIC, PVPANIC_FW_CFG_FILE,
    PVPANIC_ISA_DEFAULT_IOPORT,
};
use ruvm_hw_misc::*;
use ruvm_hw_pci::PciBus;
use ruvm_hw_pci::regs::*;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem, RegionId};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;

type Events = Arc<Mutex<Vec<PvPanicEvent>>>;

fn recorder() -> (Events, PvPanicHandler) {
    let events: Events = Arc::default();
    let e = Arc::clone(&events);
    (events, Arc::new(move |ev| e.lock().unwrap().push(ev)))
}

struct Isa {
    mem: Arc<MemorySystem>,
    io: RegionId,
    io_as: Arc<AddressSpace>,
}

impl Isa {
    fn new() -> Isa {
        let mem = Arc::new(MemorySystem::new());
        let io = mem.new_container("io", 1 << 16).unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        Isa { mem, io, io_as }
    }

    fn inb(&self, port: u64) -> u8 {
        let (v, r) = self.io_as.load(port, 1, Endian::Little, ATTRS);
        assert!(r.is_ok());
        v as u8
    }

    fn outb(&self, port: u64, v: u8) {
        assert!(self.io_as.store(port, 1, v.into(), Endian::Little, ATTRS).is_ok());
    }
}

fn fw_cfg() -> FwCfgState {
    FwCfgState::new(FwCfgProps::default(), None).unwrap()
}

/// Reads a whole fw_cfg file through the selector and data registers.
fn read_file(fw: &FwCfgState, name: &str) -> Vec<u8> {
    let (_, key, size) = fw.files().into_iter().find(|(n, _, _)| n == name).unwrap();
    assert!(fw.select(key));
    (0..size).map(|_| fw.data_read(1) as u8).collect()
}

#[test]
fn default_mask_includes_shutdown() {
    assert_eq!(PVPANIC_EVENTS, 0x7);
    assert_eq!(PvPanicIsaConfig::default().events, PVPANIC_EVENTS);
    assert_eq!(PvPanicIsaConfig::default().ioport, 0x505);
}

/// `test_panic` and `test_panic_nopause`: the port reads back the event mask and a write of 1
/// reports a panic.
#[test]
fn isa_panic() {
    let m = Isa::new();
    let fw = fw_cfg();
    let (events, handler) = recorder();
    let dev =
        PvPanicIsa::realize(&m.mem, m.io, Some(&fw), PvPanicIsaConfig::default(), handler).unwrap();
    assert!(dev.is_mapped());
    assert_eq!(dev.ioport(), PVPANIC_ISA_DEFAULT_IOPORT);

    assert_eq!(m.inb(0x505), PVPANIC_EVENTS);
    m.outb(0x505, 0x1);
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::Panicked]);
}

/// `test_pvshutdown`.
#[test]
fn isa_pvshutdown() {
    let m = Isa::new();
    let fw = fw_cfg();
    let (events, handler) = recorder();
    PvPanicIsa::realize(&m.mem, m.io, Some(&fw), PvPanicIsaConfig::default(), handler).unwrap();

    assert_eq!(m.inb(0x505), PVPANIC_EVENTS);
    m.outb(0x505, PVPANIC_SHUTDOWN);
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::Shutdown]);
}

#[test]
fn isa_crash_loaded_and_priority() {
    let m = Isa::new();
    let fw = fw_cfg();
    let (events, handler) = recorder();
    PvPanicIsa::realize(&m.mem, m.io, Some(&fw), PvPanicIsaConfig::default(), handler).unwrap();

    m.outb(0x505, PVPANIC_CRASH_LOADED);
    // Panicked wins over the other bits, crash loaded over shutdown.
    m.outb(0x505, PVPANIC_EVENTS);
    m.outb(0x505, PVPANIC_CRASH_LOADED | PVPANIC_SHUTDOWN);
    // Nothing known, nothing reported.
    m.outb(0x505, 0);
    m.outb(0x505, 0xf8);
    assert_eq!(
        *events.lock().unwrap(),
        [PvPanicEvent::CrashLoaded, PvPanicEvent::Panicked, PvPanicEvent::CrashLoaded]
    );
}

#[test]
fn isa_events_property_and_ioport() {
    let m = Isa::new();
    let fw = fw_cfg();
    let (events, handler) = recorder();
    let config = PvPanicIsaConfig { ioport: 0x600, events: PVPANIC_PANICKED };
    PvPanicIsa::realize(&m.mem, m.io, Some(&fw), config, handler).unwrap();

    assert_eq!(m.inb(0x600), PVPANIC_PANICKED);
    // The mask only changes what reads return, writes are not filtered by it.
    m.outb(0x600, PVPANIC_SHUTDOWN);
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::Shutdown]);
    // The old port is not decoded.
    assert_eq!(m.io_as.load(0x505, 1, Endian::Little, ATTRS).0, 0);

    assert_eq!(read_file(&fw, PVPANIC_FW_CFG_FILE), [0x00, 0x06]);
}

#[test]
fn isa_fw_cfg_port_file() {
    let m = Isa::new();
    let fw = fw_cfg();
    let (_, handler) = recorder();
    PvPanicIsa::realize(&m.mem, m.io, Some(&fw), PvPanicIsaConfig::default(), handler).unwrap();
    assert_eq!(read_file(&fw, PVPANIC_FW_CFG_FILE), [0x05, 0x05]);
    assert_eq!(pvpanic_port_file(0x1234), [0x34, 0x12]);
}

/// QEMU only maps the port when the machine has fw_cfg.
#[test]
fn isa_without_fw_cfg_is_not_mapped() {
    let m = Isa::new();
    let (events, handler) = recorder();
    let dev =
        PvPanicIsa::realize(&m.mem, m.io, None, PvPanicIsaConfig::default(), handler).unwrap();
    assert!(!dev.is_mapped());
    m.io_as.store(0x505, 1, 1, Endian::Little, ATTRS);
    assert!(events.lock().unwrap().is_empty());
}

struct Pci {
    mem_as: Arc<AddressSpace>,
    bus: Arc<PciBus>,
}

impl Pci {
    fn new() -> Pci {
        let mem = Arc::new(MemorySystem::new());
        let sysmem = mem.new_container("system", 1 << 64).unwrap();
        let io = mem.new_container("io", 1 << 16).unwrap();
        let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
        let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
        Pci { mem_as, bus }
    }
}

const BAR_ADDR: u64 = 0xfebf_0000;

/// `-device pvpanic-pci,addr=04.0`, then `qpci_device_enable()` and `qpci_iomap()` on BAR 0.
fn pci_device(env: &Pci, handler: PvPanicHandler) -> PvPanicPci {
    let dev =
        PvPanicPci::realize(&env.bus, Some(pci_devfn(4, 0)), PVPANIC_EVENTS, handler).unwrap();
    let pci = dev.pci_device();
    assert_eq!(pci.config_read(PCI_VENDOR_ID as u32, 2), 0x1b36);
    assert_eq!(pci.config_read(PCI_DEVICE_ID as u32, 2), 0x0011);
    assert_eq!(pci.config_read(PCI_REVISION_ID as u32, 1), 1);
    assert_eq!(pci.config_read(PCI_CLASS_DEVICE as u32, 2), u32::from(PCI_CLASS_SYSTEM_OTHER));

    // Sizing: a two byte memory BAR. Like QEMU, the size is not rounded up to 16 bytes.
    pci.config_write(PCI_BASE_ADDRESS_0 as u32, 0xffff_ffff, 4);
    assert_eq!(pci.config_read(PCI_BASE_ADDRESS_0 as u32, 4), 0xffff_fffe);

    pci.config_write(PCI_BASE_ADDRESS_0 as u32, BAR_ADDR as u32, 4);
    pci.config_write(PCI_COMMAND as u32, PCI_COMMAND_MEMORY.into(), 2);
    assert_eq!(pci.bar_addr(0), BAR_ADDR);
    dev
}

/// `test_panic` and `test_panic_nopause` of pvpanic-pci-test.c.
#[test]
fn pci_panic() {
    let env = Pci::new();
    let (events, handler) = recorder();
    let dev = pci_device(&env, handler);
    assert_eq!(PCI_DEVICE_ID_REDHAT_PVPANIC, 0x0011);
    assert_eq!(dev.events(), PVPANIC_EVENTS);

    let (v, r) = env.mem_as.load(BAR_ADDR, 1, Endian::Little, ATTRS);
    assert!(r.is_ok());
    assert_eq!(v, u64::from(PVPANIC_EVENTS));

    assert!(env.mem_as.store(BAR_ADDR, 1, 1, Endian::Little, ATTRS).is_ok());
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::Panicked]);
}

/// `test_pvshutdown` of pvpanic-pci-test.c.
#[test]
fn pci_pvshutdown() {
    let env = Pci::new();
    let (events, handler) = recorder();
    pci_device(&env, handler);

    assert_eq!(env.mem_as.load(BAR_ADDR, 1, Endian::Little, ATTRS).0, u64::from(PVPANIC_EVENTS));
    assert!(env.mem_as.store(BAR_ADDR, 1, PVPANIC_SHUTDOWN.into(), Endian::Little, ATTRS).is_ok());
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::Shutdown]);
}

/// The callbacks are one byte wide, so a word access is split into two byte accesses. Both
/// bytes read as the mask, and a write only reports an event for the byte that carries one.
#[test]
fn pci_word_access_is_split() {
    let env = Pci::new();
    let (events, handler) = recorder();
    pci_device(&env, handler);

    assert_eq!(env.mem_as.load(BAR_ADDR, 2, Endian::Little, ATTRS).0, 0x0707);
    assert!(env.mem_as.store(BAR_ADDR, 2, 0x0200, Endian::Little, ATTRS).is_ok());
    assert_eq!(*events.lock().unwrap(), [PvPanicEvent::CrashLoaded]);
}

#[test]
fn pci_bar_disabled_until_memory_decode() {
    let env = Pci::new();
    let (events, handler) = recorder();
    let dev = PvPanicPci::realize(&env.bus, None, PVPANIC_EVENTS, handler).unwrap();
    dev.pci_device().config_write(PCI_BASE_ADDRESS_0 as u32, BAR_ADDR as u32, 4);
    env.mem_as.store(BAR_ADDR, 1, 1, Endian::Little, ATTRS);
    assert!(events.lock().unwrap().is_empty());
}
