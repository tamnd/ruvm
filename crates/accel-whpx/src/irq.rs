// SPDX-License-Identifier: GPL-2.0-or-later

//! Interrupt injection around each run, `whpx_vcpu_pre_run()`, `whpx_vcpu_post_run()`,
//! `whpx_vcpu_process_async_events()` and `whpx_handle_halt()`, plus the register encodings
//! they write. With the userspace irqchip every interrupt goes through the pending interruption
//! register. With the Hyper-V LAPIC only PIC interrupts do, as ExtInt events.

/// `CPU_INTERRUPT_HARD`.
pub const INTERRUPT_HARD: u32 = 0x2;
/// `CPU_INTERRUPT_POLL`.
pub const INTERRUPT_POLL: u32 = 0x10;
/// `CPU_INTERRUPT_SMI`.
pub const INTERRUPT_SMI: u32 = 0x40;
/// `CPU_INTERRUPT_NMI`.
pub const INTERRUPT_NMI: u32 = 0x200;
/// `CPU_INTERRUPT_INIT`.
pub const INTERRUPT_INIT: u32 = 0x400;
/// `CPU_INTERRUPT_SIPI`.
pub const INTERRUPT_SIPI: u32 = 0x800;
/// `CPU_INTERRUPT_TPR`.
pub const INTERRUPT_TPR: u32 = 0x2000;

/// `WHvX64PendingInterrupt`.
pub const PENDING_INTERRUPT: u64 = 0;
/// `WHvX64PendingNmi`.
pub const PENDING_NMI: u64 = 2;
/// `WHvX64PendingEventExtInt`.
pub const EVENT_EXT_INT: u128 = 5;

/// `WHV_INTERNAL_ACTIVITY_REGISTER.HaltSuspend`.
pub const HALT_SUSPEND: u64 = 1 << 1;
/// `VpContext.ExecutionState.InterruptionPending`.
pub const EXEC_INTERRUPTION_PENDING: u16 = 1 << 6;
/// `VpContext.ExecutionState.InterruptShadow`.
pub const EXEC_INTERRUPT_SHADOW: u16 = 1 << 12;

/// `WHV_X64_PENDING_INTERRUPTION_REGISTER` for an interrupt or NMI.
pub fn pending_interruption(kind: u64, vector: u8) -> u64 {
    1 | kind << 1 | u64::from(vector) << 16
}

/// `WHV_X64_PENDING_EXT_INT_EVENT`: a PIC interrupt next to the Hyper-V LAPIC.
pub fn ext_int_event(vector: u8) -> u128 {
    1 | EVENT_EXT_INT << 1 | u128::from(vector) << 8
}

/// `WHV_X64_PENDING_EXCEPTION_EVENT`, what `whpx_inject_exceptions()` writes.
pub fn exception_event(vector: u8, error_code: Option<u32>, parameter: u64) -> u128 {
    let deliver = u128::from(error_code.is_some());
    let code = u128::from(error_code.unwrap_or(0));
    1 | deliver << 8 | u128::from(vector) << 16 | code << 32 | u128::from(parameter) << 64
}

/// `WHV_X64_DELIVERABILITY_NOTIFICATIONS_REGISTER` asking for an exit once an interrupt of
/// `priority` can be taken.
pub fn interrupt_window(priority: u8) -> u64 {
    1 << 1 | u64::from(priority & 0xf) << 2
}

/// What QEMU keeps in `AccelCPUState` between runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct InjectState {
    /// The last exit had an event in flight.
    pub interruption_pending: bool,
    /// No interrupt shadow at the last exit.
    pub interruptable: bool,
    /// The interrupt window exit fired, so the PIC can deliver.
    pub ready_for_pic_interrupt: bool,
    /// An interrupt window is requested.
    pub window_registered: bool,
    /// The priority it was requested for.
    pub window_priority: i32,
    /// CR8 as last synced.
    pub tpr: u8,
}

impl Default for InjectState {
    fn default() -> InjectState {
        InjectState {
            interruption_pending: false,
            interruptable: true,
            ready_for_pic_interrupt: false,
            window_registered: false,
            window_priority: 0,
            tpr: 0,
        }
    }
}

/// The vCPU state `whpx_vcpu_pre_run()` reads.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PreRunInput {
    /// `cpu->interrupt_request`.
    pub pending: u32,
    /// `apic_get_highest_priority_irr()`, -1 for none.
    pub irr: i32,
    /// `pic_get_output()` of the ISA PIC.
    pub pic_output: bool,
    /// EFLAGS.IF.
    pub interrupts_enabled: bool,
    /// The vCPU is in SMM.
    pub smm: bool,
    /// The Hyper-V LAPIC is in use.
    pub irqchip_in_kernel: bool,
    /// `cpu_get_apic_tpr()`.
    pub apic_tpr: u8,
}

/// The registers to write before the run and what else to do.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct PreRun {
    /// `WHvRegisterPendingInterruption`.
    pub pending_interruption: Option<u64>,
    /// `WHvRegisterPendingEvent`.
    pub pending_event: Option<u128>,
    /// `WHvX64RegisterCr8`.
    pub cr8: Option<u64>,
    /// `WHvX64RegisterDeliverabilityNotifications`.
    pub deliverability: Option<u64>,
    /// Interrupt request bits to reset.
    pub clear: u32,
    /// Leave the run loop after this run.
    pub exit_request: bool,
    /// Clear `HaltSuspend`, since an ExtInt event does not wake a halted vCPU by itself.
    pub kick_out_of_hlt: bool,
}

/// `whpx_vcpu_pre_run()`. `pic_interrupt` is `cpu_get_pic_interrupt()`, which acknowledges
/// the interrupt, so it is only called when one is injected.
pub fn pre_run(
    st: &mut InjectState,
    input: &PreRunInput,
    mut pic_interrupt: impl FnMut() -> i32,
) -> PreRun {
    let mut out = PreRun::default();
    let pending = input.pending;
    let hard = pending & INTERRUPT_HARD != 0;

    if !st.interruption_pending && pending & (INTERRUPT_NMI | INTERRUPT_SMI) != 0 {
        if pending & INTERRUPT_NMI != 0 {
            out.clear |= INTERRUPT_NMI;
            st.interruptable = false;
            out.pending_interruption = Some(pending_interruption(PENDING_NMI, 2));
        }
        if pending & INTERRUPT_SMI != 0 {
            out.clear |= INTERRUPT_SMI;
        }
    }

    // Leave the inner loop to handle INIT or a TPR access report.
    if (pending & INTERRUPT_INIT != 0 && !input.smm) || pending & INTERRUPT_TPR != 0 {
        out.exit_request = true;
    }

    // QEMU aborts when HARD is set with neither an APIC nor a PIC interrupt. Here the request
    // stays pending and nothing is injected.
    let mut irr = input.irr;
    if irr == -1 && input.pic_output {
        irr = 0;
    }

    if !input.irqchip_in_kernel {
        if !st.interruption_pending
            && st.interruptable
            && input.interrupts_enabled
            && irr >= 0
            && (i32::from(st.tpr) < irr || irr == 0)
            && hard
            && out.pending_interruption.is_none()
        {
            out.clear |= INTERRUPT_HARD;
            let irq = pic_interrupt();
            if irq >= 0 {
                out.pending_interruption = Some(pending_interruption(PENDING_INTERRUPT, irq as u8));
            }
        }
    } else if st.ready_for_pic_interrupt && hard {
        out.clear |= INTERRUPT_HARD;
        let irq = pic_interrupt();
        if irq >= 0 {
            out.pending_event = Some(ext_int_event(irq as u8));
            out.kick_out_of_hlt = true;
        }
    }

    if !input.irqchip_in_kernel && input.apic_tpr != st.tpr {
        st.tpr = input.apic_tpr;
        out.cr8 = Some(u64::from(input.apic_tpr));
        out.exit_request = true;
    }

    let wp = st.window_priority;
    if irr >= 0 && hard && (!st.window_registered || (wp < irr && wp != 0) || (irr == 0 && wp != 0))
    {
        out.deliverability = Some(interrupt_window((irr >> 4) as u8));
        st.window_registered = true;
        st.window_priority = irr;
    }

    st.ready_for_pic_interrupt = false;
    out
}

/// `whpx_vcpu_post_run()`. Returns the new TPR when the guest changed CR8 and the APIC device
/// needs `cpu_set_apic_tpr()`.
pub fn post_run(
    st: &mut InjectState,
    execution_state: u16,
    cr8: u64,
    irqchip_in_kernel: bool,
) -> Option<u8> {
    let mut tpr = None;
    if !irqchip_in_kernel && u64::from(st.tpr) != cr8 {
        st.tpr = cr8 as u8;
        tpr = Some(st.tpr);
    }
    st.interruption_pending = execution_state & EXEC_INTERRUPTION_PENDING != 0;
    st.interruptable = execution_state & EXEC_INTERRUPT_SHADOW == 0;
    tpr
}

/// The interrupt window exit: the window is used up and the PIC may deliver next time.
pub fn window_exit(st: &mut InjectState) {
    st.window_registered = false;
    st.window_priority = 0;
    st.ready_for_pic_interrupt = true;
}

/// What `whpx_vcpu_process_async_events()` does before a run.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct AsyncEvents {
    /// `do_cpu_init()`, after which the vCPU is interruptable.
    pub init: bool,
    /// `apic_poll_irq()`.
    pub poll: bool,
    /// The vCPU leaves HLT, and the Hyper-V idle flag clears.
    pub wake: bool,
    /// `do_cpu_sipi()`.
    pub sipi: bool,
    /// `apic_handle_tpr_access_report()`.
    pub tpr_report: bool,
    /// Interrupt request bits to reset.
    pub clear: u32,
}

/// `whpx_vcpu_process_async_events()`. `hyperv_hlt` is `HF2_HYPERV_HLT_MASK`, set while the
/// vCPU idles through the guest idle MSR, which wakes on an interrupt even with IF clear.
pub fn async_events(
    pending: u32,
    interrupts_enabled: bool,
    hyperv_hlt: bool,
    smm: bool,
) -> AsyncEvents {
    let mut ev =
        AsyncEvents { init: pending & INTERRUPT_INIT != 0 && !smm, ..AsyncEvents::default() };
    if pending & INTERRUPT_POLL != 0 {
        ev.poll = true;
        ev.clear |= INTERRUPT_POLL;
    }
    ev.wake = (pending & INTERRUPT_HARD != 0 && (interrupts_enabled || hyperv_hlt))
        || pending & INTERRUPT_NMI != 0;
    if pending & INTERRUPT_SIPI != 0 {
        ev.sipi = true;
        ev.clear |= INTERRUPT_SIPI;
    }
    if pending & INTERRUPT_TPR != 0 {
        ev.tpr_report = true;
        ev.clear |= INTERRUPT_TPR;
    }
    ev
}

/// `whpx_handle_halt()`: true when the vCPU really halts, false when an interrupt it can take
/// is already pending.
pub fn halts(pending: u32, interrupts_enabled: bool) -> bool {
    !(pending & INTERRUPT_HARD != 0 && interrupts_enabled) && pending & INTERRUPT_NMI == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodings() {
        assert_eq!(pending_interruption(PENDING_NMI, 2), 0x2_0005);
        assert_eq!(pending_interruption(PENDING_INTERRUPT, 0x30), 0x30_0001);
        assert_eq!(ext_int_event(0x20), 0x200b);
        let e = exception_event(13, Some(0), 0);
        assert_eq!(e, 1 | 1 << 8 | 13 << 16);
        assert_eq!(exception_event(14, Some(2), 0xdead) >> 32, 2 | 0xdead << 32);
        assert_eq!(interrupt_window(3), 0xe);
    }

    fn input(pending: u32, irr: i32) -> PreRunInput {
        PreRunInput { pending, irr, interrupts_enabled: true, ..PreRunInput::default() }
    }

    #[test]
    fn nmi_wins() {
        let mut st = InjectState::default();
        let out = pre_run(&mut st, &input(INTERRUPT_NMI | INTERRUPT_HARD, 0x30), || 0x30);
        assert_eq!(out.pending_interruption, Some(pending_interruption(PENDING_NMI, 2)));
        assert_eq!(out.clear, INTERRUPT_NMI);
        assert!(!st.interruptable);
        assert_eq!(out.deliverability, Some(interrupt_window(3)));
    }

    #[test]
    fn userspace_injects_hard() {
        let mut st = InjectState::default();
        let out = pre_run(&mut st, &input(INTERRUPT_HARD, 0x41), || 0x41);
        assert_eq!(out.pending_interruption, Some(pending_interruption(PENDING_INTERRUPT, 0x41)));
        assert_eq!(out.clear, INTERRUPT_HARD);
        let mut blocked = InjectState::default();
        let inp = PreRunInput { interrupts_enabled: false, ..input(INTERRUPT_HARD, 0x41) };
        assert_eq!(pre_run(&mut blocked, &inp, || 0x41).pending_interruption, None);
        let mut pend = InjectState { interruption_pending: true, ..InjectState::default() };
        assert_eq!(pre_run(&mut pend, &input(INTERRUPT_HARD, 0x41), || 0x41).clear, 0);
    }

    #[test]
    fn kernel_irqchip_ext_int() {
        let mut st = InjectState { ready_for_pic_interrupt: true, ..InjectState::default() };
        let inp =
            PreRunInput { irqchip_in_kernel: true, pic_output: true, ..input(INTERRUPT_HARD, -1) };
        let out = pre_run(&mut st, &inp, || 0x20);
        assert_eq!(out.pending_event, Some(ext_int_event(0x20)));
        assert!(out.kick_out_of_hlt && out.cr8.is_none());
        assert!(!st.ready_for_pic_interrupt);
    }

    #[test]
    fn tpr_sync_and_post_run() {
        let mut st = InjectState::default();
        let out = pre_run(&mut st, &PreRunInput { apic_tpr: 4, ..input(0, -1) }, || -1);
        assert_eq!((out.cr8, out.exit_request), (Some(4), true));
        assert_eq!(post_run(&mut st, EXEC_INTERRUPT_SHADOW, 6, false), Some(6));
        assert!(!st.interruptable && !st.interruption_pending);
        assert_eq!(post_run(&mut st, EXEC_INTERRUPTION_PENDING, 6, false), None);
        assert!(st.interruptable && st.interruption_pending);
        window_exit(&mut st);
        assert!(st.ready_for_pic_interrupt && !st.window_registered);
    }

    #[test]
    fn async_and_halt() {
        let ev =
            async_events(INTERRUPT_INIT | INTERRUPT_SIPI | INTERRUPT_POLL, false, false, false);
        assert!(ev.init && ev.sipi && ev.poll && !ev.wake);
        assert_eq!(ev.clear, INTERRUPT_SIPI | INTERRUPT_POLL);
        assert!(async_events(INTERRUPT_HARD, false, true, false).wake);
        assert!(!async_events(INTERRUPT_INIT, true, false, true).init);
        assert!(halts(INTERRUPT_HARD, false));
        assert!(!halts(INTERRUPT_HARD, true));
        assert!(!halts(INTERRUPT_NMI, false));
    }
}
