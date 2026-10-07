// SPDX-License-Identifier: GPL-2.0-or-later

//! I2C and SMBus controllers and devices.
//!
//! - [`i2c`] is the bus core from `hw/i2c/core.c`: slaves, addressing and transfers.
//! - [`smbus`] layers SMBus on top, the slave state machine from `smbus_slave.c` and the master
//!   helpers from `smbus_master.c`.
//! - [`pm_smbus`] is the Intel style host controller register block from `pm_smbus.c`.
//! - [`smbus_ich9`] is the Q35 SMBus function at 00:1f.3 from `smbus_ich9.c`.
//! - [`smbus_eeprom`] is the 256 byte EEPROM and SPD generator from `smbus_eeprom.c`.

#![forbid(unsafe_code)]

pub mod i2c;
pub mod pm_smbus;
pub mod smbus;
pub mod smbus_eeprom;
pub mod smbus_ich9;

pub use i2c::{I2C_BROADCAST, I2cBus, I2cEvent, I2cNak, I2cSlave};
pub use pm_smbus::{PmSmbus, PmSmbusIrqFn, PmSmbusRegs, PmSmbusVmState};
pub use smbus::{SmbusDevice, SmbusMode, SmbusSlave};
pub use smbus_eeprom::{
    SdramType, SmbusEeprom, SmbusEepromSlave, smbus_eeprom_init, smbus_eeprom_init_one,
    spd_data_generate,
};
pub use smbus_ich9::{Ich9Smbus, Ich9SmbusVmState, ich9_smbus_q35_init};
