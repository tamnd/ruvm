// SPDX-License-Identifier: GPL-2.0-or-later

//! PSCI calls on the HVC or SMC conduit, `hvf_handle_psci_call()`.
//!
//! The vCPU decodes the call and answers the ones that need nothing outside it. The rest
//! (CPU_ON, AFFINITY_INFO at level 0, the power calls) go back to the caller as a
//! [`PsciCall`], since they reach other vCPUs or the machine.

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
const FN_SYSTEM_OFF: u64 = FN_BASE + 8;
const FN_SYSTEM_RESET: u64 = FN_BASE + 9;
const FN_PSCI_FEATURES: u64 = FN_BASE + 10;
const FN64_CPU_SUSPEND: u64 = FN_CPU_SUSPEND | PSCI_64BIT;
const FN64_CPU_ON: u64 = FN_CPU_ON | PSCI_64BIT;
const FN64_AFFINITY_INFO: u64 = FN_AFFINITY_INFO | PSCI_64BIT;

/// `QEMU_PSCI_VERSION_1_1`.
pub const VERSION_1_1: i32 = 0x10001;
/// `QEMU_PSCI_0_2_RET_TOS_MIGRATION_NOT_REQUIRED`.
pub const RET_TOS_MIGRATION_NOT_REQUIRED: i32 = 2;
/// `QEMU_PSCI_RET_SUCCESS`.
pub const RET_SUCCESS: i32 = 0;
/// `QEMU_PSCI_RET_NOT_SUPPORTED`.
pub const RET_NOT_SUPPORTED: i32 = -1;
/// `QEMU_PSCI_RET_INVALID_PARAMS`.
pub const RET_INVALID_PARAMS: i32 = -2;

/// A PSCI call QEMU knows.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PsciCall {
    /// PSCI_VERSION.
    Version,
    /// MIGRATE_INFO_TYPE.
    MigrateInfoType,
    /// AFFINITY_INFO for the CPU with this MPIDR at this affinity level.
    AffinityInfo { mpidr: u64, level: u64 },
    /// SYSTEM_RESET: request a reset and turn this CPU off.
    SystemReset,
    /// SYSTEM_OFF: request a shutdown and turn this CPU off.
    SystemOff,
    /// CPU_ON, any version: start the CPU with this MPIDR at `entry` with `context` in X0.
    CpuOn { mpidr: u64, entry: u64, context: u64 },
    /// CPU_OFF.
    CpuOff,
    /// CPU_SUSPEND with its power state.
    CpuSuspend { power_state: u64 },
    /// MIGRATE.
    Migrate,
    /// PSCI_FEATURES for a function id.
    Features(u64),
}

impl PsciCall {
    /// Decodes the call from X0 to X3. `None` is a function QEMU does not know, which the
    /// caller answers with NOT_SUPPORTED in X0.
    pub fn decode(x: [u64; 4]) -> Option<PsciCall> {
        Some(match x[0] {
            FN_PSCI_VERSION => PsciCall::Version,
            FN_MIGRATE_INFO_TYPE => PsciCall::MigrateInfoType,
            FN_AFFINITY_INFO | FN64_AFFINITY_INFO => {
                PsciCall::AffinityInfo { mpidr: x[1], level: x[2] }
            }
            FN_SYSTEM_RESET => PsciCall::SystemReset,
            FN_SYSTEM_OFF => PsciCall::SystemOff,
            FN01_CPU_ON | FN_CPU_ON | FN64_CPU_ON => {
                PsciCall::CpuOn { mpidr: x[1], entry: x[2], context: x[3] }
            }
            FN01_CPU_OFF | FN_CPU_OFF => PsciCall::CpuOff,
            FN01_CPU_SUSPEND | FN_CPU_SUSPEND | FN64_CPU_SUSPEND => {
                PsciCall::CpuSuspend { power_state: x[1] }
            }
            FN01_MIGRATE | FN_MIGRATE => PsciCall::Migrate,
            FN_PSCI_FEATURES => PsciCall::Features(x[1]),
            _ => return None,
        })
    }

    /// The X0 result of a call that needs nothing outside the vCPU, or `None` when the
    /// caller has to act. CPU_SUSPEND with a valid power state is the caller's too: X0 goes
    /// to zero and the vCPU halts.
    pub fn local_result(&self) -> Option<i32> {
        match *self {
            PsciCall::Version => Some(VERSION_1_1),
            PsciCall::MigrateInfoType => Some(RET_TOS_MIGRATION_NOT_REQUIRED),
            // Everything above affinity level 0 is always on.
            PsciCall::AffinityInfo { level, .. } if level != 0 => Some(0),
            // Affinity levels are not supported in QEMU.
            PsciCall::CpuSuspend { power_state } if power_state & 0xfffe_0000 != 0 => {
                Some(RET_INVALID_PARAMS)
            }
            PsciCall::Migrate => Some(RET_NOT_SUPPORTED),
            PsciCall::Features(fid) => Some(features(fid)),
            _ => None,
        }
    }
}

/// PSCI_FEATURES: zero for the functions QEMU implements, NOT_SUPPORTED for the rest.
pub fn features(fid: u64) -> i32 {
    match fid {
        FN_PSCI_VERSION | FN_MIGRATE_INFO_TYPE | FN_AFFINITY_INFO | FN64_AFFINITY_INFO
        | FN_SYSTEM_RESET | FN_SYSTEM_OFF | FN01_CPU_ON | FN_CPU_ON | FN64_CPU_ON
        | FN01_CPU_OFF | FN_CPU_OFF | FN01_CPU_SUSPEND | FN_CPU_SUSPEND | FN64_CPU_SUSPEND
        | FN_PSCI_FEATURES => 0,
        _ => RET_NOT_SUPPORTED,
    }
}

/// The X0 value for a result: QEMU stores the `int32_t` into the 64 bit register, so
/// negative results are sign extended.
pub fn x0(ret: i32) -> u64 {
    i64::from(ret) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_each_version() {
        assert_eq!(
            PsciCall::decode([FN01_CPU_ON, 1, 0x4000_0000, 7]),
            Some(PsciCall::CpuOn { mpidr: 1, entry: 0x4000_0000, context: 7 })
        );
        assert_eq!(
            PsciCall::decode([0xc400_0003, 2, 0, 0]),
            PsciCall::decode([FN_CPU_ON, 2, 0, 0])
        );
        assert_eq!(PsciCall::decode([0x8400_0002, 0, 0, 0]), Some(PsciCall::CpuOff));
        assert_eq!(PsciCall::decode([0x8400_0008, 0, 0, 0]), Some(PsciCall::SystemOff));
        assert_eq!(PsciCall::decode([0x8400_0009, 0, 0, 0]), Some(PsciCall::SystemReset));
        // SMCCC_VERSION is not a PSCI call.
        assert_eq!(PsciCall::decode([0x8000_0000, 0, 0, 0]), None);
        // The 0.1 calls have no 64 bit forms.
        assert_eq!(PsciCall::decode([FN01_CPU_ON | PSCI_64BIT, 0, 0, 0]), None);
    }

    #[test]
    fn local_results() {
        let r = |x0| PsciCall::decode([x0, 0, 0, 0]).and_then(|c| c.local_result());
        assert_eq!(r(FN_PSCI_VERSION), Some(0x10001));
        assert_eq!(r(FN_MIGRATE_INFO_TYPE), Some(2));
        assert_eq!(r(FN_MIGRATE), Some(-1));
        assert_eq!(r(FN_CPU_OFF), None);
        let aff = |level| PsciCall::AffinityInfo { mpidr: 0, level }.local_result();
        assert_eq!((aff(0), aff(1)), (None, Some(0)));
        let sus = |power_state| PsciCall::CpuSuspend { power_state }.local_result();
        assert_eq!((sus(0x1_0000), sus(0x2_0000)), (None, Some(-2)));
    }

    #[test]
    fn features_list() {
        assert_eq!(features(FN64_CPU_SUSPEND), 0);
        assert_eq!(features(FN_PSCI_FEATURES), 0);
        assert_eq!(features(FN_MIGRATE), -1);
        assert_eq!(features(FN01_MIGRATE), -1);
        assert_eq!(features(0x8400_0012), -1);
        assert_eq!(x0(-1), u64::MAX);
        assert_eq!(x0(0x10001), 0x10001);
    }
}
