// SPDX-License-Identifier: GPL-2.0-or-later

//! The ICH9 SMBus controller, QEMU's `hw/i2c/smbus_ich9.c`.
//!
//! PCI function 00:1f.3 on Q35, 8086:2930. The host registers are the [`PmSmbus`] block behind
//! BAR 4, an I/O BAR. HOSTC at config offset 0x40 turns the block on and off, switches block
//! transfers to I2C mode and soft resets the controller. Interrupts go out on INTA.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use ruvm_base::Error;
use ruvm_hw_pci::regs::{PCI_BASE_ADDRESS_SPACE_IO, PCI_INTERRUPT_PIN};
use ruvm_hw_pci::{PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{MemorySystem, RegionId};

use crate::i2c::I2cBus;
use crate::pm_smbus::{PM_SMBUS_IO_SIZE, PmSmbus};
use crate::smbus_eeprom::{SmbusEepromSlave, smbus_eeprom_init};

/// `TYPE_ICH9_SMB_DEVICE`.
pub const TYPE_ICH9_SMB_DEVICE: &str = "ICH9-SMB";

pub const PCI_VENDOR_ID_INTEL: u16 = 0x8086;
/// `PCI_DEVICE_ID_INTEL_ICH9_6`.
pub const PCI_DEVICE_ID_INTEL_ICH9_6: u16 = 0x2930;
/// `PCI_CLASS_SERIAL_SMBUS`.
pub const PCI_CLASS_SERIAL_SMBUS: u16 = 0x0c05;
pub const ICH9_A2_SMB_REVISION: u8 = 0x02;

pub const ICH9_SMB_DEV: u8 = 31;
pub const ICH9_SMB_FUNC: u8 = 3;
/// 00:1f.3.
pub const ICH9_SMB_DEVFN: u8 = ICH9_SMB_DEV << 3 | ICH9_SMB_FUNC;

pub const ICH9_SMB_SMB_BASE: usize = 0x20;
pub const ICH9_SMB_SMB_BASE_BAR: usize = 4;
/// The size the datasheet gives SMB_BASE. QEMU registers the 64 byte `pm-smbus` region
/// instead, so the BAR really sizes as 64 bytes.
pub const ICH9_SMB_SMB_BASE_SIZE: u64 = 1 << 5;
pub const ICH9_SMB_HOSTC: usize = 0x40;
pub const ICH9_SMB_HOSTC_SSRESET: u8 = 1 << 3;
pub const ICH9_SMB_HOSTC_I2C_EN: u8 = 1 << 2;
pub const ICH9_SMB_HOSTC_SMB_SMI_EN: u8 = 1 << 1;
pub const ICH9_SMB_HOSTC_HST_EN: u8 = 1 << 0;

/// `ICH9SMBState`.
pub struct Ich9Smbus {
    dev: Arc<PciDevice>,
    smb: Arc<PmSmbus>,
    io: RegionId,
    memory: Arc<MemorySystem>,
    irq_enabled: AtomicBool,
}

impl fmt::Debug for Ich9Smbus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ich9Smbus")
            .field("devfn", &self.dev.devfn())
            .field("smb", &self.smb)
            .field("irq_enabled", &self.irq_enabled.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

struct Ich9SmbusOps(Weak<Ich9Smbus>);

impl PciDeviceOps for Ich9SmbusOps {
    /// `ich9_smbus_write_config()`.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        dev.default_write_config(addr, val, len);
        let hostc_addr = ICH9_SMB_HOSTC as u32;
        if !(addr <= hostc_addr && hostc_addr < addr + len) {
            return;
        }
        let Some(s) = self.0.upgrade() else {
            return;
        };
        let hostc = dev.config_read(hostc_addr, 1) as u8;
        let _ = s.memory.set_enabled(s.io, hostc & ICH9_SMB_HOSTC_HST_EN != 0);
        s.smb.set_i2c_enable(hostc & ICH9_SMB_HOSTC_I2C_EN != 0);
        if hostc & ICH9_SMB_HOSTC_SSRESET != 0 {
            s.smb.reset();
            dev.with_config(|c| c.config[ICH9_SMB_HOSTC] &= !ICH9_SMB_HOSTC_SSRESET);
        }
    }
}

impl Ich9Smbus {
    /// `ich9_smbus_realize()`: registers the function on `bus`. Q35 puts it at
    /// [`ICH9_SMB_DEVFN`] as a multifunction device, see [`ich9_smbus_q35_init`].
    pub fn new(bus: &Arc<PciBus>, devfn: Option<u8>) -> Result<Arc<Ich9Smbus>, Error> {
        let info = PciDeviceInfo {
            name: TYPE_ICH9_SMB_DEVICE.to_owned(),
            vendor_id: PCI_VENDOR_ID_INTEL,
            device_id: PCI_DEVICE_ID_INTEL_ICH9_6,
            revision: ICH9_A2_SMB_REVISION,
            class_id: PCI_CLASS_SERIAL_SMBUS,
            multifunction: true,
            ..PciDeviceInfo::default()
        };
        let dev = bus.register_device(&info, devfn)?;
        dev.with_config(|c| {
            c.config[PCI_INTERRUPT_PIN] = 0x01;
            c.config[ICH9_SMB_HOSTC] = 0;
        });

        let smb = PmSmbus::new(false);
        let memory = Arc::clone(bus.memory());
        let io = memory
            .new_io("pm-smbus", u128::from(PM_SMBUS_IO_SIZE), smb.io_ops())
            .map_err(|e| Error::generic(format!("ICH9-SMB: {e}")))?;
        dev.register_bar(ICH9_SMB_SMB_BASE_BAR, PCI_BASE_ADDRESS_SPACE_IO, io);

        let s = Arc::new(Ich9Smbus { dev, smb, io, memory, irq_enabled: AtomicBool::new(false) });
        let weak = Arc::downgrade(&s);
        s.smb.set_irq_handler(Some(Arc::new(move |level| {
            if let Some(s) = weak.upgrade() {
                s.set_irq(level);
            }
        })));
        s.dev.set_ops(Arc::new(Ich9SmbusOps(Arc::downgrade(&s))));
        Ok(s)
    }

    /// `ich9_smb_set_irq()`: INTA follows the controller, driven only on a change.
    fn set_irq(&self, enabled: bool) {
        if self.irq_enabled.swap(enabled, Ordering::SeqCst) == enabled {
            return;
        }
        self.dev.set_irq(i32::from(enabled));
    }

    /// The PCI function.
    pub fn device(&self) -> &Arc<PciDevice> {
        &self.dev
    }

    /// The host controller.
    pub fn pm(&self) -> &Arc<PmSmbus> {
        &self.smb
    }

    /// The SMBus the controller masters, the `"i2c"` child bus.
    pub fn smbus(&self) -> &Arc<I2cBus> {
        self.smb.bus()
    }

    /// The `pm-smbus` I/O region behind BAR 4.
    pub fn io_region(&self) -> RegionId {
        self.io
    }

    /// The level last driven on INTA.
    pub fn irq_enabled(&self) -> bool {
        self.irq_enabled.load(Ordering::SeqCst)
    }
}

/// What `pc_q35_init()` does when the machine has SMBus: the controller at 00:1f.3 and 8 blank
/// EEPROMs at 0x50 to 0x57. QEMU 11.1 still leaves the SPD data empty.
pub fn ich9_smbus_q35_init(
    bus: &Arc<PciBus>,
) -> Result<(Arc<Ich9Smbus>, Vec<Arc<SmbusEepromSlave>>), Error> {
    let smb = Ich9Smbus::new(bus, Some(ICH9_SMB_DEVFN))?;
    let eeproms = smbus_eeprom_init(smb.smbus(), 8, &[])?;
    Ok((smb, eeproms))
}
