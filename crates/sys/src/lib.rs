// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin host bindings ruvm needs that no existing crate covers: newer KVM ioctls, Hypervisor.framework, WHPX, MSHV, iommufd, guest_memfd and MAP_JIT.
//!
//! So far this holds the termination signal handler in [`signal`], host memory for guest RAM
//! in [`hostmem`], host atomic operations on it in [`hostatomic`], the direct I/O flag in
//! [`directio`] and, on Linux, the userfaultfd postcopy migration uses in `userfaultfd` and the
//! KVM dirty ring in `kvm`. The plan for the rest is in `spec/24-workspace-layout.md`.

pub mod directio;
pub mod hostatomic;
pub mod hostmem;
#[cfg(target_os = "linux")]
pub mod kvm;
#[cfg(unix)]
pub mod signal;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod userfaultfd;

pub use hostmem::HostMemory;
