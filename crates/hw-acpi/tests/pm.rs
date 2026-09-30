// SPDX-License-Identifier: GPL-2.0-or-later

//! The PM1 blocks, the PM timer and GPE from hw/acpi/core.c, and the ICH9 PM window and its
//! PMBASE decoding, driven through an I/O address space the way a guest sees them.

use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_acpi::core::*;
use ruvm_hw_acpi::ich9::*;
use ruvm_hw_core::timer::muldiv64;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_mem::{AddressSpace, MemTxAttrs, MemTxResult, MemorySystem, RegionId};

type Requests = Arc<Mutex<Vec<SystemRequest>>>;

struct Env {
    clock: Arc<Clock>,
    pm: Arc<Ich9Pm>,
    io_as: Arc<AddressSpace>,
    sci: Arc<AtomicI32>,
    sci_edges: Arc<AtomicUsize>,
    reqs: Requests,
    _mem: Arc<MemorySystem>,
    _io: RegionId,
}

impl Env {
    fn new(props: Ich9PmProps) -> Env {
        let clock = Clock::manual(ClockType::Virtual);
        let pm = Ich9Pm::new(clock.clone(), props);
        let mem = Arc::new(MemorySystem::new());
        let io = mem.new_container("io", 1 << 16).unwrap();
        let io_as = mem.address_space_init(io, "I/O").unwrap();
        pm.map(&mem, io).unwrap();

        let sci = Arc::new(AtomicI32::new(0));
        let sci_edges = Arc::new(AtomicUsize::new(0));
        let (s, e) = (sci.clone(), sci_edges.clone());
        pm.sci().connect(IrqLine::from_fn(move |level| {
            if s.swap(level, Ordering::SeqCst) == 0 && level != 0 {
                e.fetch_add(1, Ordering::SeqCst);
            }
        }));
        let reqs: Requests = Arc::default();
        let r = reqs.clone();
        pm.set_request_handler(Arc::new(move |req| r.lock().unwrap().push(req)));
        Env { clock, pm, io_as, sci, sci_edges, reqs, _mem: mem, _io: io }
    }

    fn sci(&self) -> bool {
        self.sci.load(Ordering::SeqCst) != 0
    }

    fn requests(&self) -> Vec<SystemRequest> {
        std::mem::take(&mut *self.reqs.lock().unwrap())
    }

    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        let mut b = [0u8; 8];
        let r = self.io_as.read(addr, MemTxAttrs::UNSPECIFIED, &mut b[..size]);
        (r == MemTxResult::OK).then(|| u64::from_le_bytes(b))
    }

    fn write(&self, addr: u64, size: usize, value: u64) -> bool {
        let b = value.to_le_bytes();
        self.io_as.write(addr, MemTxAttrs::UNSPECIFIED, &b[..size]) == MemTxResult::OK
    }

    fn inb(&self, addr: u64) -> u64 {
        self.read(addr, 1).unwrap()
    }
    fn inw(&self, addr: u64) -> u64 {
        self.read(addr, 2).unwrap()
    }
    fn inl(&self, addr: u64) -> u64 {
        self.read(addr, 4).unwrap()
    }
    fn outb(&self, addr: u64, v: u64) {
        assert!(self.write(addr, 1, v));
    }
    fn outw(&self, addr: u64, v: u64) {
        assert!(self.write(addr, 2, v));
    }
    fn outl(&self, addr: u64, v: u64) {
        assert!(self.write(addr, 4, v));
    }
}

const BASE: u64 = 0xb000;

/// What tco-test.c and the firmware do: PMBASE = 0xb000 | 1, ACPI_CNTL = ACPI_EN.
fn enabled_env() -> Env {
    let env = Env::new(Ich9PmProps::default());
    assert_eq!(env.pm.lpc_config_update(BASE as u32 | ICH9_LPC_PMBASE_RTE, 0x80), Some(9));
    env
}

/// The first nanosecond at which the PM timer reads `ticks`.
fn ticks_ns(ticks: u64) -> i64 {
    muldiv64(ticks, 1_000_000_000, PM_TIMER_FREQUENCY) as i64 + 1
}

/// Where the overflow timer is armed for `ticks`, rounded down like `acpi_pm_tmr_update()`.
fn deadline_ns(ticks: u64) -> i64 {
    muldiv64(ticks, 1_000_000_000, PM_TIMER_FREQUENCY) as i64
}

#[test]
fn window_is_hidden_until_acpi_en() {
    let env = Env::new(Ich9PmProps::default());
    assert_eq!(env.pm.pm_io_base(), 0);
    // PMBASE set but ACPI_EN clear: still hidden.
    assert_eq!(env.pm.lpc_config_update(0xb001, 0), Some(9));
    assert_eq!(env.pm.pm_io_base(), 0);
    assert_eq!(env.read(BASE + ICH9_PMIO_PM1_CNT, 2), None);

    assert_eq!(env.pm.lpc_config_update(0xb001, ICH9_LPC_ACPI_CTRL_ACPI_EN), Some(9));
    assert_eq!(env.pm.pm_io_base(), 0xb000);
    assert_eq!(env.pm.gpe0_blk(), 0xb020);
    assert_eq!(env.pm.gpe0_blk_len(), 16);
    assert!(env.read(BASE + ICH9_PMIO_PM1_CNT, 2).is_some());
}

#[test]
fn register_offsets() {
    let env = enabled_env();
    // No SMM: acpi_only, so SCI_EN is set from reset.
    assert_eq!(env.inw(BASE + ICH9_PMIO_PM1_CNT), u64::from(ACPI_BITMASK_SCI_ENABLE));
    assert_eq!(env.inw(BASE + ICH9_PMIO_PM1_EN), 0);
    // TMR_STS reads set right after reset: the overflow point is 0.
    assert_eq!(env.inw(BASE + ICH9_PMIO_PM1_STS), u64::from(ACPI_BITMASK_TIMER_STATUS));
    // SMI_EN has APMC_EN so SMM never runs.
    assert_eq!(env.inl(BASE + ICH9_PMIO_SMI_EN), u64::from(ICH9_PMIO_SMI_EN_APMC_EN));
    assert_eq!(env.inl(BASE + ICH9_PMIO_SMI_STS), 0);

    // GPE0: status at +0x20, enable at +0x28.
    env.outb(BASE + ICH9_PMIO_GPE0_EN, 0x5a);
    assert_eq!(env.inb(BASE + ICH9_PMIO_GPE0_EN), 0x5a);
    assert_eq!(env.inb(BASE + ICH9_PMIO_GPE0_STS), 0);
    // A 32 bit GPE access is split into bytes.
    env.outl(BASE + ICH9_PMIO_GPE0_EN, 0x0403_0201);
    assert_eq!(env.inl(BASE + ICH9_PMIO_GPE0_EN), 0x0403_0201);

    // The timer at +8.
    env.clock.advance_to(ticks_ns(1000));
    assert_eq!(env.inl(BASE + ICH9_PMIO_PM1_TMR), 1000);
    // The callback ignores the offset, so a byte read anywhere returns the low byte, as in QEMU.
    assert_eq!(env.inb(BASE + ICH9_PMIO_PM1_TMR + 1), 1000 & 0xff);
}

#[test]
fn smi_registers() {
    let env = enabled_env();
    env.outl(BASE + ICH9_PMIO_SMI_EN, 0xffff_ffff);
    assert_eq!(env.inl(BASE + ICH9_PMIO_SMI_EN), 0xffff_ffff);
    env.outl(BASE + ICH9_PMIO_SMI_EN, 0);
    assert_eq!(env.inl(BASE + ICH9_PMIO_SMI_EN), 0);
    // Without the SMI timers no SMI_STS bit is writable.
    env.outl(BASE + ICH9_PMIO_SMI_STS, 0xffff_ffff);
    assert_eq!(env.inl(BASE + ICH9_PMIO_SMI_STS), 0);
    // SMI registers only take 32 bit accesses.
    assert_eq!(env.read(BASE + ICH9_PMIO_SMI_EN, 1), None);

    // With SMM enabled APMC_EN starts clear.
    let smm = Env::new(Ich9PmProps { smm_enabled: true, ..Ich9PmProps::default() });
    assert_eq!(smm.pm.smi_en(), 0);
    assert_eq!(smm.pm.acpi().regs().pm1_cnt, 0);
    smm.pm.acpi().pm1_cnt_update(true, false);
    assert_eq!(smm.pm.acpi().regs().pm1_cnt, ACPI_BITMASK_SCI_ENABLE);
    smm.pm.acpi().pm1_cnt_update(false, true);
    assert_eq!(smm.pm.acpi().regs().pm1_cnt, 0);
}

#[test]
fn pmbase_moves_and_reset_hides() {
    let env = enabled_env();
    env.outw(BASE + ICH9_PMIO_PM1_EN, u64::from(ACPI_BITMASK_POWER_BUTTON_ENABLE));

    env.pm.lpc_config_update(0x0601, 0x80);
    assert_eq!(env.pm.pm_io_base(), 0x600);
    assert_eq!(env.read(BASE + ICH9_PMIO_PM1_EN, 2), None);
    assert_eq!(env.inw(0x600 + ICH9_PMIO_PM1_EN), u64::from(ACPI_BITMASK_POWER_BUTTON_ENABLE));
    // The low bits of PMBASE are masked off.
    env.pm.lpc_config_update(0x067f, 0x80);
    assert_eq!(env.pm.pm_io_base(), 0x600);

    // Clearing ACPI_EN hides it again.
    env.pm.lpc_config_update(0x0601, 0);
    assert_eq!(env.read(0x600 + ICH9_PMIO_PM1_EN, 2), None);

    // pm_reset() hides the window and clears the registers.
    env.pm.lpc_config_update(0x0601, 0x80);
    env.pm.reset();
    assert_eq!(env.pm.pm_io_base(), 0);
    assert_eq!(env.read(0x600 + ICH9_PMIO_PM1_EN, 2), None);
    assert_eq!(env.pm.acpi().regs().pm1_en, 0);
}

#[test]
fn sci_irq_select() {
    assert_eq!(ich9_lpc_sci_irq(ICH9_LPC_ACPI_CTRL_9), Some(9));
    assert_eq!(ich9_lpc_sci_irq(ICH9_LPC_ACPI_CTRL_10 | 0x80), Some(10));
    assert_eq!(ich9_lpc_sci_irq(ICH9_LPC_ACPI_CTRL_11), Some(11));
    assert_eq!(ich9_lpc_sci_irq(ICH9_LPC_ACPI_CTRL_20), Some(20));
    assert_eq!(ich9_lpc_sci_irq(ICH9_LPC_ACPI_CTRL_21), Some(21));
    assert_eq!(ich9_lpc_sci_irq(3), None);
    assert_eq!(ich9_lpc_sci_irq(6), None);
    assert_eq!(ich9_lpc_sci_irq(7), None);
    assert_eq!(ich9_lpc_pmbase(0xffff_ffff, 0x80), 0xff80);
}

/// lpc-ich9-test.c, test_lp1878642_pci_bus_get_irq_level_assert: a reserved SCI selection and
/// then an SCI must not trip anything.
#[test]
fn lp1878642_reserved_sci_select() {
    let env = Env::new(Ich9PmProps::default());
    assert_eq!(env.pm.lpc_config_update(0x5d00, 0xeb), None);
    assert_eq!(env.pm.pm_io_base(), 0x5d00);
    env.outw(0x5d02, 0x205d);
    // TMR_EN with TMR_STS pending raises the SCI; routing it is the LPC's business.
    assert!(env.sci());
}

#[test]
fn pm_timer_counts_and_wraps_at_24_bits() {
    let env = enabled_env();
    assert_eq!(env.inl(BASE + 8), 0);
    env.clock.advance_to(1_000_000_000);
    assert_eq!(env.inl(BASE + 8), u64::from(PM_TIMER_FREQUENCY));
    env.clock.advance_to(5_000_000_000);
    assert_eq!(env.inl(BASE + 8), (5 * u64::from(PM_TIMER_FREQUENCY)) & 0xff_ffff);
    // Writes are ignored.
    env.outl(BASE + 8, 0);
    assert_eq!(env.inl(BASE + 8), (5 * u64::from(PM_TIMER_FREQUENCY)) & 0xff_ffff);
}

#[test]
fn pm_timer_overflow_status_and_interrupt() {
    let env = enabled_env();
    let sts = BASE + ICH9_PMIO_PM1_STS;
    let en = BASE + ICH9_PMIO_PM1_EN;
    let tmr = u64::from(ACPI_BITMASK_TIMER_STATUS);

    // Clearing TMR_STS computes the next overflow, when bit 23 flips.
    env.outw(sts, tmr);
    assert_eq!(env.inw(sts) & tmr, 0);
    assert_eq!(env.pm.acpi().regs().overflow_time, 0x80_0000);

    env.outw(en, u64::from(ACPI_BITMASK_TIMER_ENABLE));
    assert!(!env.sci());
    assert_eq!(env.pm.acpi().overflow_timer_deadline(), Some(deadline_ns(0x80_0000)));

    env.clock.advance_to(deadline_ns(0x80_0000) - 1);
    assert!(!env.sci());
    assert_eq!(env.inw(sts) & tmr, 0);

    // The timer fires, latches TMR_STS and raises the SCI. It also asks for a wakeup.
    env.clock.advance_to(ticks_ns(0x80_0000));
    assert!(env.sci());
    assert_eq!(env.inw(sts) & tmr, tmr);
    assert_eq!(env.requests(), [SystemRequest::Wakeup(WakeupReason::PmTimer)]);
    assert_eq!(env.pm.acpi().overflow_timer_deadline(), None);

    // Acknowledge: the SCI drops and the next overflow is 2^23 ticks later.
    env.outw(sts, tmr);
    assert!(!env.sci());
    assert_eq!(env.pm.acpi().regs().overflow_time, 0x100_0000);
    env.clock.advance_to(ticks_ns(0x100_0000));
    assert!(env.sci());

    // With TMR_EN clear the status still latches but no SCI and no timer.
    env.outw(en, 0);
    assert!(!env.sci());
    env.outw(sts, tmr);
    assert_eq!(env.pm.acpi().overflow_timer_deadline(), None);
    env.clock.advance_to(ticks_ns(0x180_0000));
    assert_eq!(env.inw(sts) & tmr, tmr);
    assert!(!env.sci());
}

#[test]
fn pm_timer_32_bit() {
    let clock = Clock::manual(ClockType::Virtual);
    let cfg = AcpiPmConfig { tmr_width: PmTimerWidth::Bits32, ..AcpiPmConfig::default() };
    let pm = AcpiPm::new(clock.clone(), cfg);
    clock.advance_to(5_000_000_000);
    assert_eq!(pm.pm_tmr_read(), 5 * PM_TIMER_FREQUENCY);
    pm.pm1_evt_write(0, ACPI_BITMASK_TIMER_STATUS);
    assert_eq!(pm.regs().overflow_time, 0x8000_0000);
    assert_eq!(pm.pm1_evt_read(0) & ACPI_BITMASK_TIMER_STATUS, 0);
}

#[test]
fn pm1_status_is_write_one_to_clear_and_drives_sci() {
    let env = enabled_env();
    let sts = BASE + ICH9_PMIO_PM1_STS;
    let en = BASE + ICH9_PMIO_PM1_EN;
    let pwr = u64::from(ACPI_BITMASK_POWER_BUTTON_STATUS);
    env.outw(sts, 0xffff);
    assert_eq!(env.inw(sts), 0);

    // Power button with PWRBTN_EN clear: nothing happens.
    env.pm.power_down();
    assert_eq!(env.inw(sts), 0);
    assert!(!env.sci());

    env.outw(en, u64::from(ACPI_BITMASK_POWER_BUTTON_ENABLE));
    env.pm.power_down();
    assert_eq!(env.inw(sts), pwr);
    assert!(env.sci());
    assert_eq!(env.sci_edges.load(Ordering::SeqCst), 1);

    // Writing zeros or other bits does not clear it.
    env.outw(sts, 0);
    env.outw(sts, 0x0200);
    assert_eq!(env.inw(sts), pwr);
    assert!(env.sci());

    // Masking the enable drops the SCI but the status stays.
    env.outw(en, 0);
    assert!(!env.sci());
    env.outw(en, u64::from(ACPI_BITMASK_POWER_BUTTON_ENABLE));
    assert!(env.sci());

    env.outw(sts, pwr);
    assert_eq!(env.inw(sts), 0);
    assert!(!env.sci());

    // With impl.min_access_size 2 a byte write to the high half of PM1_EN reaches the callback
    // as a 16 bit write at offset 3, which it ignores. QEMU behaves the same way.
    env.outb(en + 1, 0x02);
    assert_eq!(env.inw(en), u64::from(ACPI_BITMASK_POWER_BUTTON_ENABLE));
    assert_eq!(env.inb(en + 1), 0);
    assert!(env.pm.acpi().wakeup_enabled(WakeupReason::Other));
    assert!(!env.pm.acpi().wakeup_enabled(WakeupReason::PmTimer));
}

#[test]
fn pm1_cnt_sleep_requests() {
    let env = enabled_env();
    let cnt = BASE + ICH9_PMIO_PM1_CNT;
    let slp = |typ: u64| (typ << 10) | u64::from(ACPI_BITMASK_SLEEP_ENABLE) | 1;

    // SLP_TYP without SLP_EN only records the type.
    env.outw(cnt, (5 << 10) | 1);
    assert_eq!(env.inw(cnt), (5 << 10) | 1);
    assert!(env.requests().is_empty());

    // S5 is SLP_TYP 0 in QEMU's DSDT.
    env.outw(cnt, slp(0));
    assert_eq!(env.requests(), [SystemRequest::Shutdown]);
    // SLP_EN is write only.
    assert_eq!(env.inw(cnt), 1);

    env.outw(cnt, slp(1));
    assert_eq!(env.requests(), [SystemRequest::Suspend]);

    env.outw(cnt, slp(u64::from(ACPI_DEFAULT_S4_VAL)));
    assert_eq!(env.requests(), [SystemRequest::SuspendDisk]);

    env.outw(cnt, slp(5));
    assert!(env.requests().is_empty());

    // A byte write to the high half keeps the low byte.
    env.outb(cnt + 1, u64::from(ACPI_BITMASK_SLEEP_ENABLE) >> 8);
    assert_eq!(env.requests(), [SystemRequest::Shutdown]);
    assert_eq!(env.inw(cnt), 1);

    // After a resume the runstate code sets the wake status.
    env.outw(BASE + ICH9_PMIO_PM1_STS, 0xffff);
    env.pm.acpi().notify_wakeup(WakeupReason::Other);
    assert_eq!(
        env.inw(BASE + ICH9_PMIO_PM1_STS),
        u64::from(ACPI_BITMASK_WAKE_STATUS | ACPI_BITMASK_POWER_BUTTON_STATUS)
    );
}

#[test]
fn gpe_status_and_enable() {
    let env = enabled_env();
    let sts = BASE + ICH9_PMIO_GPE0_STS;
    let en = BASE + ICH9_PMIO_GPE0_EN;
    // Keep the timer status out of the way.
    env.outw(BASE + ICH9_PMIO_PM1_STS, 0xffff);

    env.pm.acpi().send_gpe_event(0x02);
    assert_eq!(env.inb(sts), 0x02);
    assert!(!env.sci());

    env.outb(en, 0x02);
    assert!(env.sci());

    // Status is write one to clear.
    env.outb(sts, 0x01);
    assert_eq!(env.inb(sts), 0x02);
    assert!(env.sci());
    env.outb(sts, 0x02);
    assert_eq!(env.inb(sts), 0);
    assert!(!env.sci());

    // Only GPE0 byte 0 feeds the SCI, like acpi_update_sci().
    env.pm.acpi().with_regs(|r| r.gpe_sts[1] = 0xff);
    env.outb(en + 1, 0xff);
    assert_eq!(env.inb(sts + 1), 0xff);
    assert!(!env.sci());

    // Reset clears the block.
    env.pm.reset();
    let r = env.pm.acpi().regs();
    assert_eq!(&r.gpe_sts[..8], &[0; 8]);
    assert_eq!(&r.gpe_en[..8], &[0; 8]);
}

#[test]
fn system_states_blob() {
    let env = Env::new(Ich9PmProps::default());
    assert_eq!(env.pm.system_states(), [128, 0, 0, 129, 130, 128]);
    assert_eq!(system_states(true, true, 3), [128, 0, 0, 1, 3, 128]);
}
