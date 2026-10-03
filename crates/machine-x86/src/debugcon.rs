// SPDX-License-Identifier: GPL-2.0-or-later

//! `-device isa-debugcon`, hw/char/debugcon.c: the Bochs style debug port.
//!
//! Every byte the guest writes to the port goes to the device's chardev, and a read returns
//! the `readback` property (0xe9 by default, so a guest can probe for the port). The port is
//! one byte wide and takes byte accesses only.
//!
//! Deliberate differences from QEMU: the device lives with the x86 machines rather than in
//! `ruvm-hw-char`, and it writes to a [`DebugconSink`] instead of holding a chardev frontend,
//! so the caller decides where the bytes go. The "empty char device" check of
//! `debugcon_realize_core()` is left to the caller, which is the one that knows the chardev.

use std::fmt;
use std::sync::Arc;

use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RegionId,
};

/// `TYPE_ISA_DEBUGCON_DEVICE`.
pub const TYPE_ISA_DEBUGCON: &str = "isa-debugcon";
/// The default of the `iobase` property.
pub const DEBUGCON_DEFAULT_IOBASE: u32 = 0xe9;
/// The default of the `readback` property.
pub const DEBUGCON_DEFAULT_READBACK: u32 = 0xe9;

/// Where the bytes the guest writes go, `qemu_chr_fe_write_all()`.
pub type DebugconSink = Arc<dyn Fn(&[u8]) + Send + Sync>;

/// The properties of `-device isa-debugcon`, less the chardev.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DebugconConfig {
    /// `iobase`, 0xe9 by default.
    pub iobase: u32,
    /// `readback`, 0xe9 by default.
    pub readback: u32,
}

impl Default for DebugconConfig {
    fn default() -> Self {
        DebugconConfig { iobase: DEBUGCON_DEFAULT_IOBASE, readback: DEBUGCON_DEFAULT_READBACK }
    }
}

/// `debugcon_ops`.
struct DebugconOps {
    readback: u32,
    sink: DebugconSink,
}

impl fmt::Debug for DebugconOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugconOps").field("readback", &self.readback).finish_non_exhaustive()
    }
}

impl MmioOps for DebugconOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.readback))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        (self.sink)(&[value as u8]);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(1)
    }
}

/// `ISADebugconState`.
#[derive(Debug, Clone, Copy)]
pub struct IsaDebugcon {
    config: DebugconConfig,
    region: RegionId,
}

impl IsaDebugcon {
    /// `debugcon_isa_realizefn()`: maps the one byte port at `iobase` in `io_space`.
    pub fn realize(
        mem: &MemorySystem,
        io_space: RegionId,
        config: DebugconConfig,
        sink: DebugconSink,
    ) -> Result<IsaDebugcon, String> {
        let ops: Arc<dyn MmioOps> = Arc::new(DebugconOps { readback: config.readback, sink });
        let region = mem.new_io(TYPE_ISA_DEBUGCON, 1, ops).map_err(|e| e.to_string())?;
        mem.add_subregion(io_space, config.iobase.into(), region).map_err(|e| e.to_string())?;
        Ok(IsaDebugcon { config, region })
    }

    /// The `iobase` property.
    pub fn iobase(&self) -> u32 {
        self.config.iobase
    }

    /// The I/O region.
    pub fn region(&self) -> RegionId {
        self.region
    }
}
