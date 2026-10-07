// SPDX-License-Identifier: GPL-2.0-or-later

//! The `saved_address` of the `i2c_bus` section.

use std::sync::Arc;

use ruvm_hw_i2c::i2c::I2cBusVmState;
use ruvm_hw_i2c::*;

#[derive(Debug)]
struct Quiet;

impl I2cSlave for Quiet {}

#[test]
fn saved_address_follows_the_transfer() {
    let bus = I2cBus::new("i2c");
    bus.attach(0x50, Arc::new(Quiet));
    bus.attach(0x51, Arc::new(Quiet));
    assert_eq!(bus.vmstate_save(), I2cBusVmState { saved_address: 0xff });

    bus.start_recv(0x51).unwrap();
    assert_eq!(bus.vmstate_save().saved_address, 0x51);
    bus.end_transfer();
    assert_eq!(bus.vmstate_save().saved_address, 0xff);

    bus.start_send(I2C_BROADCAST).unwrap();
    assert_eq!(bus.vmstate_save().saved_address, I2C_BROADCAST);
    bus.end_transfer();

    let dst = I2cBus::new("i2c");
    assert_eq!(dst.saved_address(), 0);
    dst.vmstate_load(&I2cBusVmState { saved_address: 0x50 });
    assert_eq!(dst.saved_address(), 0x50);
    assert!(!dst.busy());
}
