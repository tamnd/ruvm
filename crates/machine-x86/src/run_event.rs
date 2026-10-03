// SPDX-License-Identifier: GPL-2.0-or-later

//! What a running x86 machine reports to its owner, whatever accelerator runs its vCPUs.
//!
//! Both run loops, [`crate::tcg_run`] and (on Linux x86 hosts) `kvm_run`, hand guest power
//! events and vCPU failures to an [`EventHandler`]; the system emulator turns them into
//! runstate changes and QMP events, as `qemu_system_shutdown_request()`,
//! `qemu_system_reset()` and `qemu_system_guest_panicked()` do.

use std::sync::Arc;

/// Why the guest asked to stop.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ShutdownReason {
    /// The guest powered off, `SHUTDOWN_CAUSE_GUEST_SHUTDOWN`.
    GuestShutdown,
    /// The guest reset with `-no-reboot` in effect, `SHUTDOWN_CAUSE_GUEST_RESET`.
    GuestReset,
}

/// What a running machine tells its owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GuestEvent {
    /// The guest wants the machine to stop.
    Shutdown(ShutdownReason),
    /// The guest reset the machine and it has been reset.
    Reset,
    /// The accelerator reported a guest crash, `KVM_SYSTEM_EVENT_CRASH`. The vCPU that saw it
    /// stops.
    Panicked,
    /// A vCPU failed. The message is QEMU's; the vCPU stops.
    InternalError(String),
}

/// Receives the [`GuestEvent`]s, on whatever thread they happen.
pub type EventHandler = Arc<dyn Fn(GuestEvent) + Send + Sync>;
