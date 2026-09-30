// SPDX-License-Identifier: GPL-2.0-or-later

//! The ICH9 LPC power management block from hw/acpi/ich9.c and the PMBASE and ACPI_CNTL
//! decoding from hw/isa/lpc_ich9.c.
//!
//! The 128 byte `ich9-pm` window holds PM1 status and enable at +0, PM1_CNT at +4, the PM timer
//! at +8, GPE0 (16 bytes, status then enable) at +0x20 and SMI_EN / SMI_STS at +0x30. It starts
//! disabled and moves when the LPC's PMBASE or ACPI_CNTL change: the LPC calls
//! [`Ich9Pm::lpc_config_update`] (or [`Ich9Pm::iospace_update`] directly), and once
//! [`Ich9Pm::map`] has put the window in an I/O space the region is moved and shown or hidden
//! there. The SCI is [`Ich9Pm::sci`]; the LPC routes it to the GSI [`ich9_lpc_sci_irq`] picks.
//!
//! Not ported: the TCO watchdog (`enable_tco`, TCO_EN locking in SMI_EN), the SWSMI and periodic
//! SMI timers from hw/acpi/ich9_timer.c, ACPI PCI hotplug, CPU hotplug and memory hotplug.
//! VMState, trace points and QOM properties are not ported either.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_hw_core::{Clock, IrqPin};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, MemError, MemResult, MemorySystem, MmioOps, RegionId,
};

use crate::core::{
    ACPI_DEFAULT_S4_VAL, ACPI_PM_TMR_LEN, ACPI_PM1_CNT_LEN, ACPI_PM1_EVT_LEN, AcpiPm, AcpiPmConfig,
    PmTimerWidth, SystemRequestHandler, system_states,
};

/// PMBASE in the LPC's config space.
pub const ICH9_LPC_PMBASE: u32 = 0x40;
/// `ICH9_MASK(32, 15, 7)`.
pub const ICH9_LPC_PMBASE_BASE_ADDRESS_MASK: u32 = 0xff80;
pub const ICH9_LPC_PMBASE_RTE: u32 = 0x1;
pub const ICH9_LPC_PMBASE_DEFAULT: u32 = 0x1;

/// ACPI_CNTL in the LPC's config space.
pub const ICH9_LPC_ACPI_CTRL: u32 = 0x44;
pub const ICH9_LPC_ACPI_CTRL_ACPI_EN: u8 = 0x80;
/// `ICH9_MASK(8, 2, 0)`.
pub const ICH9_LPC_ACPI_CTRL_SCI_IRQ_SEL_MASK: u8 = 0x07;
pub const ICH9_LPC_ACPI_CTRL_9: u8 = 0x0;
pub const ICH9_LPC_ACPI_CTRL_10: u8 = 0x1;
pub const ICH9_LPC_ACPI_CTRL_11: u8 = 0x2;
pub const ICH9_LPC_ACPI_CTRL_20: u8 = 0x4;
pub const ICH9_LPC_ACPI_CTRL_21: u8 = 0x5;
pub const ICH9_LPC_ACPI_CTRL_DEFAULT: u8 = 0x0;

pub const ICH9_PMIO_SIZE: u32 = 128;
pub const ICH9_PMIO_MASK: u32 = ICH9_PMIO_SIZE - 1;

pub const ICH9_PMIO_PM1_STS: u64 = 0x00;
pub const ICH9_PMIO_PM1_EN: u64 = 0x02;
pub const ICH9_PMIO_PM1_CNT: u64 = 0x04;
pub const ICH9_PMIO_PM1_TMR: u64 = 0x08;
pub const ICH9_PMIO_GPE0_STS: u64 = 0x20;
pub const ICH9_PMIO_GPE0_EN: u64 = 0x28;
pub const ICH9_PMIO_GPE0_LEN: u8 = 16;
pub const ICH9_PMIO_SMI_EN: u64 = 0x30;
pub const ICH9_PMIO_SMI_EN_APMC_EN: u32 = 1 << 5;
pub const ICH9_PMIO_SMI_EN_SWSMI_EN: u32 = 1 << 6;
pub const ICH9_PMIO_SMI_EN_TCO_EN: u32 = 1 << 13;
pub const ICH9_PMIO_SMI_EN_PERIODIC_EN: u32 = 1 << 14;
pub const ICH9_PMIO_SMI_STS: u64 = 0x34;
pub const ICH9_PMIO_SMI_STS_SWSMI_STS: u32 = 1 << 6;
pub const ICH9_PMIO_SMI_STS_PERIODIC_STS: u32 = 1 << 14;
pub const ICH9_PMIO_TCO_RLD: u64 = 0x60;
pub const ICH9_PMIO_TCO_LEN: u64 = 32;

/// Length of the SMI register block.
pub const ICH9_PMIO_SMI_LEN: u64 = 8;

/// `ich9_lpc_sci_irq()`: the GSI ACPI_CNTL routes the SCI to, or `None` for a reserved value.
pub fn ich9_lpc_sci_irq(acpi_cntl: u8) -> Option<u32> {
    match acpi_cntl & ICH9_LPC_ACPI_CTRL_SCI_IRQ_SEL_MASK {
        ICH9_LPC_ACPI_CTRL_9 => Some(9),
        ICH9_LPC_ACPI_CTRL_10 => Some(10),
        ICH9_LPC_ACPI_CTRL_11 => Some(11),
        ICH9_LPC_ACPI_CTRL_20 => Some(20),
        ICH9_LPC_ACPI_CTRL_21 => Some(21),
        _ => None,
    }
}

/// The PM I/O base the LPC config gives: PMBASE masked, or 0 when ACPI_EN is clear.
pub fn ich9_lpc_pmbase(pmbase: u32, acpi_cntl: u8) -> u32 {
    if acpi_cntl & ICH9_LPC_ACPI_CTRL_ACPI_EN != 0 {
        pmbase & ICH9_LPC_PMBASE_BASE_ADDRESS_MASK
    } else {
        0
    }
}

/// The ICH9 PM properties the LPC exposes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ich9PmProps {
    /// `disable_s3`.
    pub disable_s3: bool,
    /// `disable_s4`.
    pub disable_s4: bool,
    /// `s4_val`.
    pub s4_val: u8,
    /// The LPC's `smm-enabled`.
    pub smm_enabled: bool,
    /// The LPC's `smm-compat`.
    pub smm_compat: bool,
}

impl Default for Ich9PmProps {
    /// `ich9_pm_add_properties()` defaults.
    fn default() -> Self {
        Ich9PmProps {
            disable_s3: false,
            disable_s4: false,
            s4_val: ACPI_DEFAULT_S4_VAL,
            smm_enabled: false,
            smm_compat: false,
        }
    }
}

/// The part of `ICH9LPCPMRegs` outside `ACPIREGS`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
struct SmiState {
    smi_en: u32,
    smi_en_wmask: u32,
    smi_sts: u32,
    smi_sts_wmask: u32,
    pm_io_base: u32,
}

/// Where [`Ich9Pm::map`] put the window.
struct Mapping {
    mem: Weak<MemorySystem>,
    region: RegionId,
}

/// `ICH9LPCPMRegs`.
pub struct Ich9Pm {
    acpi: Arc<AcpiPm>,
    props: Ich9PmProps,
    state: Mutex<SmiState>,
    mapping: Mutex<Option<Mapping>>,
}

impl fmt::Debug for Ich9Pm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ich9Pm")
            .field("acpi", &self.acpi)
            .field("props", &self.props)
            .field("state", &*lock(&self.state))
            .finish_non_exhaustive()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Ich9Pm {
    /// `ich9_pm_init()` without the regions (see [`Ich9Pm::map`]), then `pm_reset()`.
    pub fn new(clock: Arc<Clock>, props: Ich9PmProps) -> Arc<Self> {
        let cfg = AcpiPmConfig {
            gpe_len: ICH9_PMIO_GPE0_LEN,
            tmr_width: PmTimerWidth::Bits24,
            s4_val: props.s4_val,
            acpi_only: !props.smm_compat && !props.smm_enabled,
        };
        let pm = Arc::new(Ich9Pm {
            acpi: AcpiPm::new(clock, cfg),
            props,
            state: Mutex::new(SmiState::default()),
            mapping: Mutex::new(None),
        });
        pm.reset();
        pm
    }

    pub fn props(&self) -> &Ich9PmProps {
        &self.props
    }

    /// The ACPI registers, for the APM handler (`acpi_pm1_cnt_update()`) and the power button.
    pub fn acpi(&self) -> &Arc<AcpiPm> {
        &self.acpi
    }

    /// The SCI output, which the LPC routes to the GSI ACPI_CNTL selects.
    pub fn sci(&self) -> &IrqPin {
        self.acpi.sci()
    }

    /// Where shutdown and suspend requests from PM1_CNT go.
    pub fn set_request_handler(&self, handler: SystemRequestHandler) {
        self.acpi.set_request_handler(handler);
    }

    /// The `etc/system-states` fw_cfg file for these properties.
    pub fn system_states(&self) -> [u8; 6] {
        system_states(self.props.disable_s3, self.props.disable_s4, self.props.s4_val)
    }

    /// Builds the `ich9-pm` container with its subregions and adds it, disabled, to `io` at
    /// offset 0. Later PMBASE changes move it within `io`. Returns the container.
    pub fn map(
        self: &Arc<Self>,
        mem: &Arc<MemorySystem>,
        io: RegionId,
    ) -> Result<RegionId, MemError> {
        let c = mem.new_container("ich9-pm", u128::from(ICH9_PMIO_SIZE))?;
        let tmr = mem.new_io("acpi-tmr", u128::from(ACPI_PM_TMR_LEN), self.acpi.tmr_ops())?;
        mem.add_subregion(c, ICH9_PMIO_PM1_TMR, tmr)?;
        let evt = mem.new_io("acpi-evt", u128::from(ACPI_PM1_EVT_LEN), self.acpi.evt_ops())?;
        mem.add_subregion(c, ICH9_PMIO_PM1_STS, evt)?;
        let cnt = mem.new_io("acpi-cnt", u128::from(ACPI_PM1_CNT_LEN), self.acpi.cnt_ops())?;
        mem.add_subregion(c, ICH9_PMIO_PM1_CNT, cnt)?;
        let gpe = mem.new_io("acpi-gpe0", u128::from(ICH9_PMIO_GPE0_LEN), self.acpi.gpe_ops())?;
        mem.add_subregion(c, ICH9_PMIO_GPE0_STS, gpe)?;
        let smi = mem.new_io("acpi-smi", u128::from(ICH9_PMIO_SMI_LEN), self.smi_ops())?;
        mem.add_subregion(c, ICH9_PMIO_SMI_EN, smi)?;

        let base = lock(&self.state).pm_io_base;
        mem.set_enabled(c, base != 0)?;
        mem.add_subregion(io, u64::from(base), c)?;
        *lock(&self.mapping) = Some(Mapping { mem: Arc::downgrade(mem), region: c });
        Ok(c)
    }

    /// `ich9_pm_iospace_update()`: moves the window to `pm_io_base`, hiding it for 0.
    ///
    /// # Panics
    ///
    /// If `pm_io_base` is not aligned to the window size, like the C assertion.
    pub fn iospace_update(&self, pm_io_base: u32) {
        assert_eq!(pm_io_base & ICH9_PMIO_MASK, 0, "PM I/O base must be 128 byte aligned");
        lock(&self.state).pm_io_base = pm_io_base;
        let mapping = lock(&self.mapping);
        if let Some(m) = mapping.as_ref() {
            if let Some(mem) = m.mem.upgrade() {
                let _t = mem.transaction();
                // The region was mapped by map(), so these cannot fail.
                let _ = mem.set_enabled(m.region, pm_io_base != 0);
                let _ = mem.set_address(m.region, u64::from(pm_io_base));
            }
        }
    }

    /// `ich9_lpc_pmbase_sci_update()` less the GSI switch: applies PMBASE and ACPI_CNTL and
    /// returns the SCI GSI they select, `None` if the selection is reserved. When the GSI
    /// changes while the SCI is high the LPC lowers the old one and raises the new one.
    pub fn lpc_config_update(&self, pmbase: u32, acpi_cntl: u8) -> Option<u32> {
        self.iospace_update(ich9_lpc_pmbase(pmbase, acpi_cntl));
        ich9_lpc_sci_irq(acpi_cntl)
    }

    /// The `pm_io_base` property.
    pub fn pm_io_base(&self) -> u32 {
        lock(&self.state).pm_io_base
    }

    /// The `gpe0_blk` property.
    pub fn gpe0_blk(&self) -> u32 {
        self.pm_io_base() + ICH9_PMIO_GPE0_STS as u32
    }

    /// The `gpe0_blk_len` property.
    pub fn gpe0_blk_len(&self) -> u32 {
        u32::from(ICH9_PMIO_GPE0_LEN)
    }

    /// SMI_EN.
    pub fn smi_en(&self) -> u32 {
        lock(&self.state).smi_en
    }

    /// SMI_STS.
    pub fn smi_sts(&self) -> u32 {
        lock(&self.state).smi_sts
    }

    /// `ich9_smi_readl()`.
    pub fn smi_read(&self, addr: u64) -> u32 {
        let s = lock(&self.state);
        match addr {
            0 => s.smi_en,
            4 => s.smi_sts,
            _ => 0,
        }
    }

    /// `ich9_smi_writel()`, without the TCO lock and the SMI timers.
    pub fn smi_write(&self, addr: u64, val: u32) {
        let mut s = lock(&self.state);
        match addr {
            0 => {
                s.smi_en &= !s.smi_en_wmask;
                s.smi_en |= val & s.smi_en_wmask;
            }
            4 => {
                s.smi_sts &= !s.smi_sts_wmask;
                s.smi_sts |= val & s.smi_sts_wmask;
            }
            _ => {}
        }
    }

    /// `pm_reset()`.
    pub fn reset(&self) {
        self.iospace_update(0);
        self.acpi.reset();
        {
            let mut s = lock(&self.state);
            s.smi_en = 0;
            if !self.props.smm_enabled {
                // Mark SMM as already set up so that SMM never runs.
                s.smi_en |= ICH9_PMIO_SMI_EN_APMC_EN;
            }
            s.smi_en_wmask = !0;
        }
        self.acpi.update_sci();
    }

    /// `pm_powerdown_req()`.
    pub fn power_down(&self) {
        self.acpi.power_down();
    }

    /// The `acpi-smi` region.
    pub fn smi_ops(self: &Arc<Self>) -> Arc<Ich9SmiOps> {
        Arc::new(Ich9SmiOps(self.clone()))
    }
}

/// `ich9_smi_ops`: SMI_EN and SMI_STS, 32 bit accesses only.
#[derive(Debug)]
pub struct Ich9SmiOps(pub Arc<Ich9Pm>);

impl MmioOps for Ich9SmiOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.smi_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.smi_write(offset, value as u32);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}
