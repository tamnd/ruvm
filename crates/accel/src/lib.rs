// SPDX-License-Identifier: GPL-2.0-or-later

//! The accelerator and vCPU traits, vCPU threads, kicks and lazy register sync.
//!
//! What is here so far is the part a machine needs to run its vCPUs on TCG:
//!
//! - [`VcpuControl`], the run control half of QEMU's `AccelOpsClass`: resume and pause all
//!   vCPUs, kick one, and ask whether they are all paused.
//! - [`tcg`], the `tcg-accel` object: its properties (`thread`, `tb-size`, `split-wx`,
//!   `one-insn-per-tb`) with QEMU's parsing and errors, `tcg_init_machine()`'s choice between
//!   MTTCG and round robin, and [`tcg::TcgVcpus`], the vCPU threads of
//!   `tcg-accel-ops-mttcg.c` and `tcg-accel-ops-rr.c` (from `ruvm-jit`) under that control.
//!
//! The rest of document 06's `Accel` and `Vcpu` traits (`init_machine()`, memory slots,
//! register get and put, the accelerator registry) is not here yet. KVM keeps its own run
//! loop in `ruvm-machine-x86`.

#![forbid(unsafe_code)]

pub mod tcg;

/// Starting, stopping and kicking the vCPUs of a machine, `resume_all_vcpus()`,
/// `pause_all_vcpus()` and `qemu_cpu_kick()`.
pub trait VcpuControl: Send + Sync {
    /// The accelerator's name, as `-accel` takes it.
    fn accel_name(&self) -> &'static str;

    /// How many vCPUs there are.
    fn vcpu_count(&self) -> usize;

    /// `resume_all_vcpus()`: let every vCPU run.
    fn resume_all(&self);

    /// `pause_all_vcpus()`: stop every vCPU and wait until they are all out of guest code.
    /// From a vCPU thread it only asks, since waiting there would never end.
    fn pause_all(&self);

    /// `all_vcpus_paused()`.
    fn all_paused(&self) -> bool;

    /// `qemu_cpu_kick()` for vCPU `index`: wake it and make it leave guest code.
    fn kick(&self, index: usize);
}
