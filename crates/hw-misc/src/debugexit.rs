// SPDX-License-Identifier: GPL-2.0-or-later

//! `-device isa-debug-exit`, hw/misc/debugexit.c.
//!
//! Test harnesses use this port to end a run with a status of their choosing. A write of `val`
//! of any width from 1 to 4 bytes asks for a guest shutdown with exit code `(val << 1) | 1`, so
//! the code is always odd and a guest can never fake a clean exit. Reads return 0.

use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemResult, MemorySystem, MmioOps, RegionId,
};

/// The default of the `iobase` property.
pub const DEBUG_EXIT_DEFAULT_IOBASE: u32 = 0x501;
/// The default of the `iosize` property.
pub const DEBUG_EXIT_DEFAULT_IOSIZE: u32 = 0x02;

/// Called with the exit code on every guest write. QEMU passes it to
/// `qemu_system_shutdown_request_with_code()` with `SHUTDOWN_CAUSE_GUEST_SHUTDOWN`.
pub type DebugExitHandler = Arc<dyn Fn(u64) + Send + Sync>;

/// The exit code for a guest write of `value`.
pub const fn debug_exit_code(value: u64) -> u64 {
    (value << 1) | 1
}

/// The properties of `-device isa-debug-exit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsaDebugExitConfig {
    /// `iobase`, 0x501 by default.
    pub iobase: u32,
    /// `iosize`, 2 by default.
    pub iosize: u32,
}

impl Default for IsaDebugExitConfig {
    fn default() -> Self {
        IsaDebugExitConfig { iobase: DEBUG_EXIT_DEFAULT_IOBASE, iosize: DEBUG_EXIT_DEFAULT_IOSIZE }
    }
}

/// `debug_exit_ops`.
struct DebugExitOps {
    handler: DebugExitHandler,
}

impl fmt::Debug for DebugExitOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DebugExitOps").finish_non_exhaustive()
    }
}

impl MmioOps for DebugExitOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(0)
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        (self.handler)(debug_exit_code(value));
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}

/// `ISADebugExitState`.
#[derive(Debug, Clone, Copy)]
pub struct IsaDebugExit {
    config: IsaDebugExitConfig,
    region: RegionId,
}

impl IsaDebugExit {
    /// `debug_exit_realizefn()`: maps an `iosize` byte region at `iobase` in `io_space`.
    pub fn realize(
        mem: &MemorySystem,
        io_space: RegionId,
        config: IsaDebugExitConfig,
        handler: DebugExitHandler,
    ) -> Result<IsaDebugExit> {
        let ops: Arc<dyn MmioOps> = Arc::new(DebugExitOps { handler });
        let region = mem
            .new_io("isa-debug-exit", config.iosize.into(), ops)
            .map_err(|e| Error::generic(e.to_string()))?;
        mem.add_subregion(io_space, config.iobase.into(), region)
            .map_err(|e| Error::generic(e.to_string()))?;
        Ok(IsaDebugExit { config, region })
    }

    /// The `iobase` property.
    pub fn iobase(&self) -> u32 {
        self.config.iobase
    }

    /// The `iosize` property.
    pub fn iosize(&self) -> u32 {
        self.config.iosize
    }

    /// The I/O region.
    pub fn region(&self) -> RegionId {
        self.region
    }
}
