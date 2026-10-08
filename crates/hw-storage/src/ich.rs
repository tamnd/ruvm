// SPDX-License-Identifier: GPL-2.0-or-later

//! The ICH9 AHCI PCI function, ported from QEMU's `hw/ide/ich.c`.

use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::thread::{self, ThreadId};

use ruvm_base::Error;
use ruvm_hw_pci::regs::{
    PCI_BASE_ADDRESS_SPACE_IO, PCI_BASE_ADDRESS_SPACE_MEMORY, PCI_CAP_ID_SATA, PCI_CLASS_PROG,
    PCI_INTERRUPT_PIN, PCI_LATENCY_TIMER, pci_set_long, pci_set_word,
};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps, PciDeviceVmState};
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

use crate::ahci::{AHCI_MEM_BAR_SIZE, AhciState, AhciVmState, DmaMemory};
use crate::block::BlockBackend;
use crate::ide::{DriveConfig, DriveKind, IdeDrive};

/// The Intel vendor ID.
pub const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// The ICH9 6-port SATA controller in AHCI mode.
pub const PCI_DEVICE_ID_INTEL_82801IR: u16 = 0x2922;
/// Mass storage, SATA.
pub const PCI_CLASS_STORAGE_SATA: u16 = 0x0106;
/// How many ports the ICH9 controller has.
pub const ICH9_AHCI_PORTS: usize = 6;

const ICH9_MSI_CAP_OFFSET: u8 = 0x80;
const ICH9_SATA_CAP_OFFSET: u8 = 0xa8;
const ICH9_IDP_BAR: usize = 4;
const ICH9_MEM_BAR: usize = 5;
const ICH9_IDP_INDEX: u64 = 0x10;
const ICH9_IDP_SIZE: u128 = 0x20;
const SATA_CAP_SIZE: u8 = 0x08;
const SATA_CAP_REV: usize = 0x02;
const SATA_CAP_BAR: usize = 0x04;
const AHCI_PROGMODE_MAJOR_REV_1: u8 = 1;

/// QEMU numbers drives from one global counter, two per IDE bus. Each AHCI port is a bus, on
/// this controller and on `sysbus-ahci`.
pub(crate) static DRIVE_SERIAL: AtomicU32 = AtomicU32::new(1);

/// DMA that only reaches memory while the function is a bus master.
struct GatedDma {
    inner: Arc<dyn DmaMemory>,
    dev: OnceLock<Weak<PciDevice>>,
}

impl GatedDma {
    fn allowed(&self) -> bool {
        self.dev.get().and_then(Weak::upgrade).is_some_and(|d| d.is_bus_master())
    }
}

impl DmaMemory for GatedDma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.allowed() && self.inner.dma_read(addr, buf)
    }

    fn dma_write(&self, addr: u64, buf: &[u8]) -> bool {
        self.allowed() && self.inner.dma_write(addr, buf)
    }
}

struct Shared {
    state: Mutex<AhciState>,
    /// The thread inside a register access, so that a DMA transfer that lands on the
    /// controller's own registers is dropped instead of deadlocking.
    owner: Mutex<Option<ThreadId>>,
    dev: OnceLock<Weak<PciDevice>>,
    serial_base: u32,
}

impl fmt::Debug for Shared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Shared").field("serial_base", &self.serial_base).finish_non_exhaustive()
    }
}

impl Shared {
    fn lock_state(&self) -> MutexGuard<'_, AhciState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Runs `f` on the state, then delivers the interrupt changes it caused. Returns `None` if
    /// this thread is already inside the controller.
    fn access<R>(&self, f: impl FnOnce(&mut AhciState) -> R) -> Option<R> {
        let me = thread::current().id();
        if *self.owner.lock().unwrap_or_else(|p| p.into_inner()) == Some(me) {
            return None;
        }
        let mut st = self.lock_state();
        *self.owner.lock().unwrap_or_else(|p| p.into_inner()) = Some(me);
        let r = f(&mut st);
        let events = std::mem::take(&mut st.irq_events);
        *self.owner.lock().unwrap_or_else(|p| p.into_inner()) = None;
        drop(st);
        self.deliver(&events);
        Some(r)
    }

    /// `ahci_irq_raise()` and `ahci_irq_lower()`.
    fn deliver(&self, events: &[bool]) {
        let Some(dev) = self.dev.get().and_then(Weak::upgrade) else {
            return;
        };
        for &level in events {
            if dev.msi_enabled() {
                if level {
                    dev.msi_notify(0);
                }
            } else {
                dev.set_irq(i32::from(level));
            }
        }
    }
}

struct AbarOps(Arc<Shared>);

impl MmioOps for AbarOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.access(|st| st.mem_read(offset, size.bytes())).unwrap_or(0))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.access(|st| st.mem_write(offset, value));
        Ok(())
    }
}

struct IdpOps(Arc<Shared>);

impl MmioOps for IdpOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.access(|st| st.idp_read(offset, size.bytes())).unwrap_or(0))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.access(|st| st.idp_write(offset, value));
        Ok(())
    }
}

struct ResetOps(Arc<Shared>);

impl PciDeviceOps for ResetOps {
    /// `pci_ich9_reset()`.
    fn reset(&self, _dev: &PciDevice) {
        self.0.access(AhciState::reset);
    }
}

/// `vmstate_ich9_ahci` (version 1).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ich9AhciVmState {
    pub parent_obj: PciDeviceVmState,
    pub ahci: AhciVmState,
}

/// The ICH9 AHCI controller, QEMU's `ich9-ahci` device, with six SATA ports.
pub struct Ich9Ahci {
    dev: Arc<PciDevice>,
    shared: Arc<Shared>,
}

impl fmt::Debug for Ich9Ahci {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ich9Ahci").field("devfn", &self.dev.devfn()).finish_non_exhaustive()
    }
}

impl Ich9Ahci {
    /// Creates the controller on `bus` at `devfn` (or the first free slot), `pci_ich9_ahci_realize()`.
    ///
    /// `dma` is the memory the controller reads command lists and data buffers from and writes
    /// FISes and data to, normally the system memory address space. MSI is offered only when the
    /// bus supports it, as in QEMU.
    pub fn new(
        bus: &Arc<PciBus>,
        dma: Arc<dyn DmaMemory>,
        devfn: Option<u8>,
    ) -> Result<Arc<Self>, Error> {
        Self::create(bus, dma, devfn, false)
    }

    /// [`Ich9Ahci::new`] as a function of a multifunction device, the way q35 creates the
    /// built-in controller at 00:1f.2 with `pci_create_simple_multifunction()`.
    pub fn new_multifunction(
        bus: &Arc<PciBus>,
        dma: Arc<dyn DmaMemory>,
        devfn: Option<u8>,
    ) -> Result<Arc<Self>, Error> {
        Self::create(bus, dma, devfn, true)
    }

    fn create(
        bus: &Arc<PciBus>,
        dma: Arc<dyn DmaMemory>,
        devfn: Option<u8>,
        multifunction: bool,
    ) -> Result<Arc<Self>, Error> {
        let info = PciDeviceInfo {
            name: "ich9-ahci".to_string(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: PCI_DEVICE_ID_INTEL_82801IR,
            revision: 0x02,
            class_id: PCI_CLASS_STORAGE_SATA,
            multifunction,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;

        let gated = Arc::new(GatedDma { inner: dma, dev: OnceLock::new() });
        let _ = gated.dev.set(Arc::downgrade(&dev));
        let state = AhciState::new(ICH9_AHCI_PORTS, ICH9_IDP_INDEX, gated);
        let shared = Arc::new(Shared {
            state: Mutex::new(state),
            owner: Mutex::new(None),
            dev: OnceLock::new(),
            serial_base: DRIVE_SERIAL.fetch_add(2 * ICH9_AHCI_PORTS as u32, Ordering::Relaxed),
        });
        let _ = shared.dev.set(Arc::downgrade(&dev));
        dev.set_ops(Arc::new(ResetOps(Arc::clone(&shared))));

        dev.with_config(|c| {
            c.config[PCI_CLASS_PROG] = AHCI_PROGMODE_MAJOR_REV_1;
            // Cache line size of 32 bytes, in dwords.
            c.config[0x0c] = 0x08;
            c.config[PCI_LATENCY_TIMER] = 0;
            c.config[PCI_INTERRUPT_PIN] = 1;
            // The ICH9 manual, 14.1.29: "Address Map".
            c.config[0x90] = 1 << 6;
        });

        let mem = bus.memory();
        let idp = mem
            .new_io("ahci-idp", ICH9_IDP_SIZE, Arc::new(IdpOps(Arc::clone(&shared))))
            .map_err(|e| Error::generic(format!("ahci-idp: {e}")))?;
        let abar = mem
            .new_io("ahci", u128::from(AHCI_MEM_BAR_SIZE), Arc::new(AbarOps(Arc::clone(&shared))))
            .map_err(|e| Error::generic(format!("ahci: {e}")))?;
        dev.register_bar(ICH9_IDP_BAR, PCI_BASE_ADDRESS_SPACE_IO, idp);
        dev.register_bar(ICH9_MEM_BAR, PCI_BASE_ADDRESS_SPACE_MEMORY, abar);

        let sata = dev.add_capability(PCI_CAP_ID_SATA, ICH9_SATA_CAP_OFFSET, SATA_CAP_SIZE)?;
        let sata = usize::from(sata);
        dev.with_config(|c| {
            // Revision 1.0. The BAR field is the config offset of BAR4 in dwords (4 + 4) and the
            // offset field is log2 of the IDP offset 0x10 within it.
            pci_set_word(c.config, sata + SATA_CAP_REV, 0x10);
            pci_set_long(
                c.config,
                sata + SATA_CAP_BAR,
                (ICH9_IDP_BAR as u32 + 4) | (ICH9_IDP_INDEX.trailing_zeros() << 4),
            );
        });

        // An error here only means the platform has no MSI; the device works without it.
        let _ = dev.msi_init(ICH9_MSI_CAP_OFFSET, 1, true, false);

        Ok(Arc::new(Ich9Ahci { dev, shared }))
    }

    /// The PCI function.
    pub fn pci_device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The device's section.
    pub fn vmstate_save(&self) -> Ich9AhciVmState {
        Ich9AhciVmState {
            parent_obj: self.dev.vmstate_save(),
            ahci: self.shared.lock_state().vmstate_save(),
        }
    }

    /// Loads the device's section: config space first, so that bus mastering is back before
    /// the engines restart, then the controller. Only an idle controller loads; see
    /// [`AhciVmState`]. The interrupt line is not recomputed, as in QEMU: the PCI part of the
    /// stream carries its level.
    pub fn vmstate_load(&self, v: &Ich9AhciVmState) -> Result<(), String> {
        self.dev.vmstate_load(&v.parent_obj)?;
        self.shared
            .access(|st| st.vmstate_load(&v.ahci))
            .unwrap_or_else(|| Err("ahci: busy".to_string()))
    }

    /// Plugs a drive into `port` (0 to 5), the `ide-hd` or `ide-cd` device with `bus=ahci.N`.
    ///
    /// A hard disk needs a backend; a CD-ROM drive without one has no medium. The port goes
    /// through a reset so the guest sees the device signature.
    pub fn attach_drive(
        &self,
        port: usize,
        config: DriveConfig,
        blk: Option<Arc<dyn BlockBackend>>,
    ) -> Result<(), Error> {
        if port >= ICH9_AHCI_PORTS {
            return Err(Error::generic(format!("ahci: no port {port}")));
        }
        if config.kind == DriveKind::Hd && blk.is_none() {
            return Err(Error::generic("ide-hd: a hard disk needs a block backend"));
        }
        let serial = format!("QM{:05}", self.shared.serial_base + 2 * port as u32);
        let drive = IdeDrive::new(&config, blk, serial);
        self.shared
            .access(|st| {
                if st.ports[port].attached() {
                    return Err(Error::generic(format!("ahci: port {port} is in use")));
                }
                st.attach(port, drive);
                Ok(())
            })
            .unwrap_or_else(|| Err(Error::generic("ahci: busy")))
    }
}
