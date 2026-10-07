// SPDX-License-Identifier: GPL-2.0-or-later

//! The `port92` section of hw/i386/port92.c, the `dma` sections of the two i8257s of
//! hw/dma/i8257.c, which are only parsed, and the `i2c_bus` section of hw/i2c/core.c.

use std::sync::{Arc, LazyLock};

use ruvm_hw_i2c::I2cBus;
use ruvm_hw_i2c::i2c::I2cBusVmState;
use ruvm_migration::SaveVm;
use ruvm_vmstate::{VmStateDescription, VmStateField};

use crate::pc::{Port92, Port92VmState};

/// `vmstate_port92_isa`.
pub(crate) static VMSTATE_PORT92: LazyLock<VmStateDescription<Port92VmState>> =
    LazyLock::new(|| {
        type S = Port92VmState;
        VmStateDescription::new("port92")
            .version_id(1)
            .minimum_version_id(1)
            .field(VmStateField::scalar("outport", |s: &mut S| &mut s.outport))
    });

/// Registers the `port92` section of `port92`, instance 0 with no path prefix as in QEMU.
pub(crate) fn register_port92(savevm: &mut SaveVm, port92: &Arc<Port92>) {
    let (get, put) = (Arc::clone(port92), Arc::clone(port92));
    savevm.register_vmsd(
        "",
        Some(0),
        &VMSTATE_PORT92,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

/// The fields of `I8257Regs`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct I8257RegsVmState {
    now: [i32; 2],
    base: [u16; 2],
    mode: u8,
    page: u8,
    pageh: u8,
    dack: u8,
    eop: u8,
}

/// The fields of `I8257State`. There is no i8257 model, so the section is only parsed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct I8257VmState {
    command: u8,
    mask: u8,
    flip_flop: u8,
    dshift: i32,
    regs: [I8257RegsVmState; 4],
}

/// `vmstate_i8257_regs`.
static VMSTATE_I8257_REGS: LazyLock<VmStateDescription<I8257RegsVmState>> = LazyLock::new(|| {
    type S = I8257RegsVmState;
    VmStateDescription::new("dma_regs").version_id(1).minimum_version_id(1).fields([
        VmStateField::array("now", |s: &mut S| &mut s.now),
        VmStateField::array("base", |s: &mut S| &mut s.base),
        VmStateField::scalar("mode", |s: &mut S| &mut s.mode),
        VmStateField::scalar("page", |s: &mut S| &mut s.page),
        VmStateField::scalar("pageh", |s: &mut S| &mut s.pageh),
        VmStateField::scalar("dack", |s: &mut S| &mut s.dack),
        VmStateField::scalar("eop", |s: &mut S| &mut s.eop),
    ])
});

/// `vmstate_i8257`.
static VMSTATE_I8257: LazyLock<VmStateDescription<I8257VmState>> = LazyLock::new(|| {
    type S = I8257VmState;
    VmStateDescription::new("dma").version_id(1).minimum_version_id(1).fields([
        VmStateField::scalar("command", |s: &mut S| &mut s.command),
        VmStateField::scalar("mask", |s: &mut S| &mut s.mask),
        VmStateField::scalar("flip_flop", |s: &mut S| &mut s.flip_flop),
        VmStateField::scalar("dshift", |s: &mut S| &mut s.dshift),
        VmStateField::struct_array("regs", &VMSTATE_I8257_REGS, |s: &mut S| &mut s.regs).version(1),
    ])
});

/// Registers the `dma` sections of the slave (instance 0) and master (instance 1) i8257, which
/// are parsed and dropped.
pub(crate) fn register_dma(savevm: &mut SaveVm) {
    savevm.register_discard_with("", Some(0), &VMSTATE_I8257, I8257VmState::default);
    savevm.register_discard_with("", Some(1), &VMSTATE_I8257, I8257VmState::default);
}

/// `vmstate_i2c_bus`. `i2c_bus_pre_save()` is [`I2cBus::vmstate_save`].
pub(crate) static VMSTATE_I2C_BUS: LazyLock<VmStateDescription<I2cBusVmState>> =
    LazyLock::new(|| {
        type S = I2cBusVmState;
        VmStateDescription::new("i2c_bus")
            .version_id(1)
            .minimum_version_id(1)
            .field(VmStateField::scalar("saved_address", |s: &mut S| &mut s.saved_address))
    });

/// Registers the `i2c_bus` section of `bus` at the next free instance, as
/// `vmstate_register_any()` does: 0 for the ICH9 SMBus, the only bus on q35.
pub(crate) fn register_i2c_bus(savevm: &mut SaveVm, bus: &Arc<I2cBus>) {
    let (get, put) = (Arc::clone(bus), Arc::clone(bus));
    savevm.register_vmsd(
        "",
        None,
        &VMSTATE_I2C_BUS,
        move || Ok(get.vmstate_save()),
        move |s| {
            put.vmstate_load(&s);
            Ok(())
        },
    );
}

#[cfg(test)]
mod tests {
    use ruvm_vmstate::{StreamReader, StreamWriter, vmstate_load_state, vmstate_save_state};

    use super::*;

    #[test]
    fn port92_layout_matches_qemu() {
        let mut s = Port92VmState { outport: 0x02 };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_PORT92, &mut s).unwrap();
        assert_eq!(f.as_bytes(), &[0x02]);
    }

    #[test]
    fn dma_is_75_bytes() {
        let mut s = I8257VmState { command: 1, dshift: 1, ..I8257VmState::default() };
        s.regs[3] = I8257RegsVmState {
            now: [1, -1],
            base: [0xffff, 2],
            eop: 1,
            ..I8257RegsVmState::default()
        };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_I8257, &mut s).unwrap();
        let mut b = f.into_inner();
        assert_eq!(b.len(), 75);
        assert_eq!(&b[..7], &[1, 0, 0, 0, 0, 0, 1]);
        // The last channel: now[], base[], mode, page, pageh, dack and eop.
        let last = [0, 0, 0, 1, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 2, 0, 0, 0, 0, 1];
        assert_eq!(&b[75 - 17..], &last[..]);
        // What follows the section in a stream: the nested structures at the end peek past it.
        b.push(0);
        let mut back = I8257VmState::default();
        vmstate_load_state(&mut StreamReader::new(&b), &VMSTATE_I8257, &mut back, 1).unwrap();
        assert_eq!(back, s);
    }

    #[test]
    fn i2c_bus_layout_matches_qemu() {
        let mut s = I2cBusVmState { saved_address: 0xff };
        let mut f = StreamWriter::new();
        vmstate_save_state(&mut f, &VMSTATE_I2C_BUS, &mut s).unwrap();
        assert_eq!(f.as_bytes(), &[0xff]);
    }
}
