// SPDX-License-Identifier: GPL-2.0-or-later

//! The GICv2m MSI frame, from hw/intc/arm_gicv2m.c.
//!
//! A write of an interrupt ID to MSI_SETSPI_NS pulses the matching SPI. The frame owns
//! `num_spi` lines starting at SPI `base_spi`, so interrupt ID `base_spi + 32 + n` pulses line
//! `n`. Where QEMU logs `LOG_GUEST_ERROR` for a bad size or offset the frame stays silent.

use std::fmt;

use ruvm_hw_core::irq::IrqLine;
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

/// The size of the frame's MMIO region.
pub const GICV2M_SIZE: u64 = 0x1000;
/// `GICV2M_NUM_SPI_MAX`.
pub const GICV2M_NUM_SPI_MAX: u32 = 128;

const V2M_MSI_TYPER: u64 = 0x008;
const V2M_MSI_SETSPI_NS: u64 = 0x040;
const V2M_MSI_IIDR: u64 = 0xfcc;
const V2M_IIDR0: u64 = 0xfd0;
const V2M_IIDR11: u64 = 0xffc;
/// `PRODUCT_ID_QEMU`, the ASCII code of Q.
const PRODUCT_ID_QEMU: u32 = 0x51;

/// The GICv2m frame. Map [`GicV2m`] itself as the MMIO region.
pub struct GicV2m {
    base_spi: u32,
    spi: Vec<IrqLine>,
}

impl fmt::Debug for GicV2m {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GicV2m")
            .field("base_spi", &self.base_spi)
            .field("num_spi", &self.spi.len())
            .finish()
    }
}

impl GicV2m {
    /// Check the props like `gicv2m_realize()` and build the frame. `spi(n)` gives the GIC input
    /// for the frame's line `n`, which is SPI `base_spi + n`.
    pub fn new(
        base_spi: u32,
        num_spi: u32,
        spi: impl Fn(u32) -> IrqLine,
    ) -> Result<GicV2m, String> {
        if num_spi > GICV2M_NUM_SPI_MAX {
            return Err(format!(
                "requested {num_spi} SPIs exceeds GICv2m frame maximum {GICV2M_NUM_SPI_MAX}"
            ));
        }
        if base_spi + 32 > 1020 - num_spi {
            return Err(format!(
                "requested base SPI {}+{} exceeds max. number 1020",
                base_spi + 32,
                num_spi
            ));
        }
        Ok(GicV2m { base_spi, spi: (0..num_spi).map(|n| spi(base_spi + n)).collect() })
    }

    /// `base-spi`.
    pub fn base_spi(&self) -> u32 {
        self.base_spi
    }

    /// `num-spi`.
    pub fn num_spi(&self) -> u32 {
        self.spi.len() as u32
    }
}

impl MmioOps for GicV2m {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if size.bytes() != 4 {
            return Ok(0);
        }
        let val = match offset {
            V2M_MSI_TYPER => ((self.base_spi + 32) << 16) | self.num_spi(),
            // No implementer and architecture revision 0, as the spec asks.
            V2M_MSI_IIDR => PRODUCT_ID_QEMU << 20,
            // The optional ID registers are not there and MSI_PIDR2 reads as 0.
            V2M_IIDR0..=V2M_IIDR11 => 0,
            _ => 0,
        };
        Ok(u64::from(val))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if !matches!(size.bytes(), 2 | 4) || offset != V2M_MSI_SETSPI_NS {
            return Ok(());
        }
        let spi = i64::from((value & 0x3ff) as u32) - i64::from(self.base_spi + 32);
        if (0..i64::from(self.num_spi())).contains(&spi) {
            self.spi[spi as usize].pulse();
        }
        Ok(())
    }
}
