// SPDX-License-Identifier: GPL-2.0-or-later

//! The guest architecture: what the target-independent code asks the target for, the parts of
//! `linux-user/<target>/` that differ from one target to the next.

use std::sync::{Arc, OnceLock};

use ruvm_jit::Cpu;
use ruvm_user_common::GuestSpace;

use crate::signal::{Info, Sigaction, Task};
use crate::syscall::Proc;

/// One target of the emulator.
pub(crate) struct Guest {
    /// `UNAME_MACHINE`, what `uname()` says the machine is.
    pub(crate) machine: &'static str,
    /// `TARGET_MINSIGSTKSZ`.
    pub(crate) minsigstksz: u64,
    /// The size of the vCPU's `env`.
    pub(crate) env_size: usize,
    /// Whether the target has the system call numbers and structures of `asm-generic`
    /// (`struct stat`, `struct epoll_event`), which the host's differ from.
    pub(crate) generic_abi: bool,
    /// The `O_` flags whose target value differs from the host's, as `(target, host)`.
    pub(crate) open_flags: &'static [(u64, u64)],
    /// `/proc/cpuinfo`, when the target has its own, `open_cpuinfo()`.
    pub(crate) cpuinfo: Option<fn() -> String>,
    /// The stack pointer.
    pub(crate) sp: fn(&Cpu<'_>) -> u64,
    /// `cpu_clone_regs_child()`: the registers of the new thread or process, which returns 0
    /// from `clone()` and runs on `newsp` when it is not 0.
    pub(crate) clone_regs: fn(&mut Cpu<'_>, u64),
    /// `cpu_set_tls()`.
    pub(crate) set_tls: fn(&mut Cpu<'_>, u64),
    /// `cpu_loop()`, until the thread calls `exit` with others left.
    pub(crate) cpu_loop: fn(&Arc<Proc>, &mut Task, &mut Cpu<'_>),
    /// `setup_rt_frame()`: the frame for the handler of a signal on the guest stack, and the
    /// registers that enter it. The last argument is the guest mask to return to.
    pub(crate) setup_rt_frame:
        fn(&GuestSpace, &mut Task, &mut Cpu<'_>, i32, &Sigaction, &Info, u64),
    /// `do_rt_sigreturn()`.
    pub(crate) rt_sigreturn: fn(&GuestSpace, &mut Task, &mut Cpu<'_>) -> i64,
}

static GUEST: OnceLock<&'static Guest> = OnceLock::new();

/// Makes `g` the target, once, before the guest runs.
pub(crate) fn set(g: &'static Guest) {
    let _ = GUEST.set(g);
}

/// The target.
pub(crate) fn guest() -> &'static Guest {
    GUEST.get().expect("the guest is set before it runs")
}

/// `target_to_host_bitmask()` of the `O_` flags.
pub(crate) fn open_flags_to_host(f: u64) -> u64 {
    let mut h = f;
    for &(t, _) in guest().open_flags {
        h &= !t;
    }
    for &(t, host) in guest().open_flags {
        if f & t != 0 {
            h |= host;
        }
    }
    h
}

/// `host_to_target_bitmask()` of the `O_` flags.
pub(crate) fn open_flags_to_target(h: u64) -> u64 {
    let mut f = h;
    for &(_, host) in guest().open_flags {
        f &= !host;
    }
    for &(t, host) in guest().open_flags {
        if h & host != 0 {
            f |= t;
        }
    }
    f
}
