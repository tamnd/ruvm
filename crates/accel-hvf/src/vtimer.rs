// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtual timer arithmetic `hvf_sync_vtimer()` and `hvf_arm_wfi_timer()` do when the GIC
//! is not the framework's.
//!
//! The framework runs CNTV for the guest and exits with `HV_EXIT_REASON_VTIMER_ACTIVATED` when
//! it fires, masking it until the host says otherwise. It only exits from inside
//! `hv_vcpu_run()`, so a vCPU halted in WFI needs a host timer for the deadline instead.

/// `TMR_CTL_ENABLE`.
pub const CTL_ENABLE: u64 = 1;
/// `TMR_CTL_IMASK`.
pub const CTL_IMASK: u64 = 2;
/// `TMR_CTL_ISTATUS`.
pub const CTL_ISTATUS: u64 = 4;

/// The level of the vtimer interrupt line for a CNTV_CTL_EL0 value: enabled, not masked and
/// firing.
pub fn irq_level(ctl: u64) -> bool {
    ctl & (CTL_ENABLE | CTL_IMASK | CTL_ISTATUS) == CTL_ENABLE | CTL_ISTATUS
}

/// What a WFI should do about the vtimer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum WfiTimer {
    /// The timer is off or masked: halt until something else wakes the vCPU.
    Halt,
    /// The deadline has passed: do not halt.
    Expired,
    /// Halt, and wake after this many nanoseconds.
    Sleep(u64),
}

/// `hvf_arm_wfi_timer()`: the deadline for CNTV_CTL_EL0 `ctl` and CNTV_CVAL_EL0 `cval` when
/// the guest's counter reads `now` and ticks at `freq_hz`.
pub fn wfi_timer(ctl: u64, cval: u64, now: u64, freq_hz: u64) -> WfiTimer {
    if ctl & CTL_ENABLE == 0 || ctl & CTL_IMASK != 0 {
        return WfiTimer::Halt;
    }
    if cval <= now {
        return WfiTimer::Expired;
    }
    WfiTimer::Sleep(muldiv64(cval - now, 1_000_000_000, freq_hz))
}

/// `muldiv64()`: `a * b / c` with a 128 bit intermediate, saturated to 64 bits.
fn muldiv64(a: u64, b: u64, c: u64) -> u64 {
    if c == 0 {
        return u64::MAX;
    }
    let r = u128::from(a) * u128::from(b) / u128::from(c);
    u64::try_from(r).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_needs_enable_and_status_without_mask() {
        assert!(irq_level(CTL_ENABLE | CTL_ISTATUS));
        assert!(!irq_level(CTL_ENABLE | CTL_ISTATUS | CTL_IMASK));
        assert!(!irq_level(CTL_ISTATUS));
        assert!(!irq_level(CTL_ENABLE));
    }

    #[test]
    fn wfi_deadlines() {
        // Apple Silicon counters run at 24 MHz.
        let f = 24_000_000;
        assert_eq!(wfi_timer(0, 100, 0, f), WfiTimer::Halt);
        assert_eq!(wfi_timer(CTL_ENABLE | CTL_IMASK, 100, 0, f), WfiTimer::Halt);
        assert_eq!(wfi_timer(CTL_ENABLE, 100, 100, f), WfiTimer::Expired);
        assert_eq!(wfi_timer(CTL_ENABLE, 1100, 100, f), WfiTimer::Sleep(41_666));
        assert_eq!(wfi_timer(CTL_ENABLE, 24_000_100, 100, f), WfiTimer::Sleep(1_000_000_000));
        assert_eq!(wfi_timer(CTL_ENABLE, u64::MAX, 0, 1), WfiTimer::Sleep(u64::MAX));
    }
}
