// SPDX-License-Identifier: GPL-2.0-or-later

//! PSCI firmware emulated by the CPU: the port of QEMU's `target/arm/tcg/psci.c`, which
//! the virt board turns on (through the `psci-conduit` property, here
//! [`Arm::with_psci`](super::Arm::with_psci)) when no guest firmware runs at EL3 or EL2.
//!
//! An HVC or SMC on the conduit with a PSCI function ID in X0 is handled here instead of
//! being taken as an exception; the result goes to X0 and execution continues after the
//! instruction. The calls that reach the rest of the machine go through
//! [`ArmBoard`](super::ArmBoard), standing in for QEMU's `arm_set_cpu_on()`,
//! `arm_get_cpu_by_id()` and the system reset and shutdown requests.
//!
//! Differences from QEMU: CPU_SUSPEND halts the CPU until it has work without running the
//! WFI trap checks of `helper_wfi()`, and only AArch64 callers exist.

use ruvm_jit::{Cpu, interrupt};

use super::{Arm, PsciConduit};
use crate::cpu::{CpuArmState, EXCP_HVC, EXCP_SMC};

/// `QEMU_PSCI_0_2_64BIT`.
const PSCI_64BIT: u64 = 0x4000_0000;
/// `QEMU_PSCI_0_1_FN_BASE`.
const FN01_BASE: u64 = 0x95c1_ba5e;
const FN01_CPU_SUSPEND: u64 = FN01_BASE;
const FN01_CPU_OFF: u64 = FN01_BASE + 1;
const FN01_CPU_ON: u64 = FN01_BASE + 2;
const FN01_MIGRATE: u64 = FN01_BASE + 3;
/// `QEMU_PSCI_0_2_FN_BASE`.
const FN_BASE: u64 = 0x8400_0000;
const FN_PSCI_VERSION: u64 = FN_BASE;
const FN_CPU_SUSPEND: u64 = FN_BASE + 1;
const FN_CPU_OFF: u64 = FN_BASE + 2;
const FN_CPU_ON: u64 = FN_BASE + 3;
const FN_AFFINITY_INFO: u64 = FN_BASE + 4;
const FN_MIGRATE: u64 = FN_BASE + 5;
const FN_MIGRATE_INFO_TYPE: u64 = FN_BASE + 6;
const FN_MIGRATE_INFO_UP_CPU: u64 = FN_BASE + 7;
const FN_SYSTEM_OFF: u64 = FN_BASE + 8;
const FN_SYSTEM_RESET: u64 = FN_BASE + 9;
const FN_PSCI_FEATURES: u64 = FN_BASE + 10;
const FN64_CPU_SUSPEND: u64 = FN_CPU_SUSPEND | PSCI_64BIT;
const FN64_CPU_ON: u64 = FN_CPU_ON | PSCI_64BIT;
const FN64_AFFINITY_INFO: u64 = FN_AFFINITY_INFO | PSCI_64BIT;
const FN64_MIGRATE: u64 = FN_MIGRATE | PSCI_64BIT;
const FN64_MIGRATE_INFO_UP_CPU: u64 = FN_MIGRATE_INFO_UP_CPU | PSCI_64BIT;

/// `QEMU_PSCI_VERSION_1_1`.
const PSCI_VERSION_1_1: i64 = 0x10001;
/// `QEMU_PSCI_0_2_RET_TOS_MIGRATION_NOT_REQUIRED`.
const RET_TOS_MIGRATION_NOT_REQUIRED: i64 = 2;

/// `QEMU_PSCI_RET_SUCCESS`.
pub const PSCI_RET_SUCCESS: i64 = 0;
/// `QEMU_PSCI_RET_NOT_SUPPORTED`.
pub const PSCI_RET_NOT_SUPPORTED: i64 = -1;
/// `QEMU_PSCI_RET_INVALID_PARAMS`.
pub const PSCI_RET_INVALID_PARAMS: i64 = -2;
/// `QEMU_PSCI_RET_DENIED`.
pub const PSCI_RET_DENIED: i64 = -3;
/// `QEMU_PSCI_RET_ALREADY_ON`.
pub const PSCI_RET_ALREADY_ON: i64 = -4;
/// `QEMU_PSCI_RET_ON_PENDING`.
pub const PSCI_RET_ON_PENDING: i64 = -5;
/// `QEMU_PSCI_RET_INTERNAL_FAILURE`.
pub const PSCI_RET_INTERNAL_FAILURE: i64 = -6;

/// `PSCI_ON`: the CPU is running.
pub const PSCI_ON: u32 = 0;
/// `PSCI_OFF`: the CPU is powered off.
pub const PSCI_OFF: u32 = 1;
/// `PSCI_ON_PENDING`: a CPU_ON for the CPU is in progress.
pub const PSCI_ON_PENDING: u32 = 2;

/// `arm_is_psci_call()`: whether the exception `excp` is a PSCI call on the conduit.
pub(crate) fn is_psci_call(arm: &Arm, cpu: &Cpu<'_>, excp: i32) -> bool {
    match (arm.psci_conduit(), excp) {
        (PsciConduit::Hvc, EXCP_HVC) | (PsciConduit::Smc, EXCP_SMC) => {}
        _ => return false,
    }
    let param = CpuArmState::load(cpu.env).xregs[0];
    matches!(
        param,
        FN_PSCI_VERSION
            | FN_MIGRATE_INFO_TYPE
            | FN_PSCI_FEATURES
            | FN_SYSTEM_RESET
            | FN_SYSTEM_OFF
            | FN_CPU_ON
            | FN64_CPU_ON
            | FN_CPU_OFF
            | FN_CPU_SUSPEND
            | FN64_CPU_SUSPEND
            | FN_AFFINITY_INFO
            | FN64_AFFINITY_INFO
            | FN_MIGRATE
            | FN64_MIGRATE
            | FN_MIGRATE_INFO_UP_CPU
            | FN64_MIGRATE_INFO_UP_CPU
            | FN01_CPU_SUSPEND
            | FN01_CPU_OFF
            | FN01_CPU_ON
            | FN01_MIGRATE
    )
}

/// `arm_handle_psci_call()`.
pub(crate) fn handle_psci_call(arm: &Arm, cpu: &mut Cpu<'_>) {
    let mut st = CpuArmState::load(cpu.env);
    // All PSCI functions take explicit 32-bit or native int sized arguments so we can
    // simply zero-extend all arguments regardless of which exact function we are about to
    // call.
    let param = [st.xregs[0], st.xregs[1], st.xregs[2], st.xregs[3]];
    let board = arm.board.clone();

    let ret: i64 = match param[0] {
        FN_PSCI_VERSION => PSCI_VERSION_1_1,
        FN_MIGRATE_INFO_TYPE => RET_TOS_MIGRATION_NOT_REQUIRED,
        FN_AFFINITY_INFO | FN64_AFFINITY_INFO => {
            let mpidr = param[1];
            match param[2] {
                0 => match board.as_ref().and_then(|b| b.psci_power_state(mpidr)) {
                    Some(state) => i64::from(state),
                    None => PSCI_RET_INVALID_PARAMS,
                },
                // Everything above affinity level 0 is always on.
                _ => 0,
            }
        }
        FN_SYSTEM_RESET => {
            if let Some(b) = &board {
                b.psci_system_reset();
            }
            // QEMU reset and shutdown are async requests, but PSCI mandates that we never
            // return from the reset/shutdown call, so power the CPU off now so it doesn't
            // execute anything further.
            arm.cpu_off(cpu);
            return;
        }
        FN_SYSTEM_OFF => {
            if let Some(b) = &board {
                b.psci_system_off();
            }
            arm.cpu_off(cpu);
            return;
        }
        FN01_CPU_ON | FN_CPU_ON | FN64_CPU_ON => {
            // The PSCI spec mandates that newly brought up CPUs start in the highest
            // exception level which exists and is enabled on the calling CPU. Since the
            // QEMU PSCI implementation is acting as a "fake EL3" or "fake EL2" firmware,
            // this for us means that we want to start at the highest NS exception level
            // that we are providing to the guest.
            let target_el = if arm.features().el2 { 2 } else { 1 };
            match &board {
                Some(b) => b.psci_cpu_on(param[1], param[2], param[3], target_el),
                None => PSCI_RET_INVALID_PARAMS,
            }
        }
        FN01_CPU_OFF | FN_CPU_OFF => {
            arm.cpu_off(cpu);
            return;
        }
        FN01_CPU_SUSPEND | FN_CPU_SUSPEND | FN64_CPU_SUSPEND => {
            // Affinity levels are not supported in QEMU.
            if param[1] & 0xfffe_0000 != 0 {
                PSCI_RET_INVALID_PARAMS
            } else {
                // Powerdown is not supported, we always go into WFI.
                st.xregs[0] = 0;
                st.store(cpu.env);
                if !cpu.has_work() {
                    cpu.core.shared().set_interrupt(interrupt::HALT);
                }
                return;
            }
        }
        FN_PSCI_FEATURES => match param[1] {
            FN_PSCI_VERSION | FN_MIGRATE_INFO_TYPE | FN_AFFINITY_INFO | FN64_AFFINITY_INFO
            | FN_SYSTEM_RESET | FN_SYSTEM_OFF | FN_CPU_ON | FN64_CPU_ON | FN_CPU_OFF
            | FN_CPU_SUSPEND | FN64_CPU_SUSPEND | FN_PSCI_FEATURES => 0,
            _ => PSCI_RET_NOT_SUPPORTED,
        },
        _ => PSCI_RET_NOT_SUPPORTED,
    };
    st.xregs[0] = ret as u64;
    st.store(cpu.env);
}
