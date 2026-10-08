// SPDX-License-Identifier: GPL-2.0-or-later

//! The memory mapped AHCI controller, ported from QEMU's `hw/ide/ahci-sysbus.c`.
//!
//! [`SysbusAhci`] is `SysbusAHCIState`, the `sysbus-ahci` device: the AHCI register block of
//! [`AhciState`] as one 0x1000 byte MMIO region and one interrupt line, with no PCI function
//! around it. The `sbsa-ref` board maps it with six ports.
//!
//! Differences from QEMU:
//!
//! - Not ported: QOM registration and VMState (`vmstate_sysbus_ahci`).
//! - DMA always reaches memory; there is no bus master bit to gate it.
//! - The board calls [`SysbusAhci::reset`] for the legacy reset handler.

use std::fmt;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, ThreadId};

use ruvm_base::Error;
use ruvm_hw_core::irq::IrqPin;
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

use crate::ahci::{AHCI_MEM_BAR_SIZE, AhciState, DmaMemory};
use crate::block::BlockBackend;
use crate::ich::DRIVE_SERIAL;
use crate::ide::{DriveConfig, DriveKind, IdeDrive};

/// `TYPE_SYSBUS_AHCI`.
pub const TYPE_SYSBUS_AHCI: &str = "sysbus-ahci";

/// The size of the register block, `AHCI_MEM_BAR_SIZE`.
pub const SYSBUS_AHCI_MMIO_SIZE: u64 = AHCI_MEM_BAR_SIZE;

struct Shared {
    state: Mutex<AhciState>,
    /// The thread inside a register access, so that a DMA transfer that lands on the
    /// controller's own registers is dropped instead of deadlocking.
    owner: Mutex<Option<ThreadId>>,
    irq: IrqPin,
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
        // ahci_irq_raise() and ahci_irq_lower() without a PCI device: qemu_irq_raise() and
        // qemu_irq_lower() on the sysbus line.
        for level in events {
            self.irq.set_bool(level);
        }
        Some(r)
    }
}

/// `ahci_mem_ops` on the `sysbus-ahci` register block.
struct MemOps(Arc<Shared>);

impl MmioOps for MemOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        Ok(self.0.access(|st| st.mem_read(offset, size.bytes())).unwrap_or(0))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.access(|st| st.mem_write(offset, value));
        Ok(())
    }
}

/// `SysbusAHCIState`, the `sysbus-ahci` device.
pub struct SysbusAhci {
    shared: Arc<Shared>,
    nports: usize,
    serial_base: u32,
}

impl fmt::Debug for SysbusAhci {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SysbusAhci").field("nports", &self.nports).finish_non_exhaustive()
    }
}

impl SysbusAhci {
    /// `sysbus_ahci_init()` and `sysbus_ahci_realize()` with `num-ports` set to `nports`.
    ///
    /// `dma` is the memory the controller reads command lists and data buffers from and writes
    /// FISes and data to, `address_space_memory` in QEMU.
    pub fn new(nports: usize, dma: Arc<dyn DmaMemory>) -> Arc<SysbusAhci> {
        let shared = Arc::new(Shared {
            state: Mutex::new(AhciState::new(nports, 0, dma)),
            owner: Mutex::new(None),
            irq: IrqPin::new(),
        });
        let serial_base = DRIVE_SERIAL.fetch_add(2 * nports as u32, Ordering::Relaxed);
        Arc::new(SysbusAhci { shared, nports, serial_base })
    }

    /// The number of ports, `num-ports`.
    pub fn nports(&self) -> usize {
        self.nports
    }

    /// The interrupt output, sysbus IRQ 0.
    pub fn irq(&self) -> &IrqPin {
        &self.shared.irq
    }

    /// The register block, sysbus MMIO region 0, [`SYSBUS_AHCI_MMIO_SIZE`] bytes.
    pub fn mmio_ops(&self) -> Arc<dyn MmioOps> {
        Arc::new(MemOps(Arc::clone(&self.shared)))
    }

    /// `sysbus_ahci_reset()`.
    pub fn reset(&self) {
        self.shared.access(AhciState::reset);
    }

    /// Plugs a drive into `port`, as `ahci_ide_create_devs()` does for the drive with that
    /// index.
    ///
    /// A hard disk needs a backend; a CD-ROM drive without one has no medium. The port goes
    /// through a reset so the guest sees the device signature.
    pub fn attach_drive(
        &self,
        port: usize,
        config: DriveConfig,
        blk: Option<Arc<dyn BlockBackend>>,
    ) -> Result<(), Error> {
        if port >= self.nports {
            return Err(Error::generic(format!("ahci: no port {port}")));
        }
        if config.kind == DriveKind::Hd && blk.is_none() {
            return Err(Error::generic("ide-hd: a hard disk needs a block backend"));
        }
        let serial = format!("QM{:05}", self.serial_base + 2 * port as u32);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::VecBackend;
    use ruvm_hw_core::irq::IrqLine;
    use std::sync::atomic::AtomicI32;

    struct NoDma;

    impl DmaMemory for NoDma {
        fn dma_read(&self, _addr: u64, _buf: &mut [u8]) -> bool {
            false
        }

        fn dma_write(&self, _addr: u64, _buf: &[u8]) -> bool {
            false
        }
    }

    fn read32(ahci: &SysbusAhci, off: u64) -> u64 {
        ahci.mmio_ops().read(&AccessCtx::default(), off, AccessSize::new(4).unwrap()).unwrap()
    }

    fn write32(ahci: &SysbusAhci, off: u64, v: u64) {
        ahci.mmio_ops().write(&AccessCtx::default(), off, AccessSize::new(4).unwrap(), v).unwrap();
    }

    #[test]
    fn six_ports_and_drives() {
        let ahci = SysbusAhci::new(6, Arc::new(NoDma));
        // HOST_CAP.NP is the number of ports less one, HOST_PORTS_IMPL has a bit per port.
        assert_eq!(read32(&ahci, 0x00) & 0x1f, 5);
        assert_eq!(read32(&ahci, 0x0c), 0x3f);
        let blk: Arc<dyn BlockBackend> = Arc::new(VecBackend::new(1 << 20));
        ahci.attach_drive(2, DriveConfig::hd(), Some(blk.clone())).unwrap();
        assert!(ahci.attach_drive(2, DriveConfig::hd(), Some(blk)).is_err());
        assert!(ahci.attach_drive(6, DriveConfig::cdrom(), None).is_err());
        assert!(ahci.attach_drive(0, DriveConfig::hd(), None).is_err());
    }

    #[test]
    fn interrupt_follows_host_is() {
        let ahci = SysbusAhci::new(1, Arc::new(NoDma));
        let level = Arc::new(AtomicI32::new(-1));
        let l = level.clone();
        ahci.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        // GHC.AE and GHC.IE, then port 0 PxIE for D2H register FIS and a write to PxIS
        // that has no bit set leaves the line low.
        write32(&ahci, 0x04, 0x8000_0002);
        write32(&ahci, 0x100 + 0x14, 1);
        assert_ne!(level.load(Ordering::SeqCst), 1);
        ahci.reset();
        assert_eq!(read32(&ahci, 0x04) & 2, 0);
    }
}
