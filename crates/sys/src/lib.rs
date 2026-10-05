// SPDX-License-Identifier: MIT OR Apache-2.0

//! Thin host bindings ruvm needs that no existing crate covers: newer KVM ioctls, Hypervisor.framework, WHPX, MSHV, iommufd, guest_memfd and MAP_JIT.
//!
//! So far this holds the termination signal handler in [`signal`], host memory for guest RAM
//! in [`hostmem`] and host atomic operations on it in [`hostatomic`]. The plan for the rest is in
//! `spec/24-workspace-layout.md`.

pub mod hostatomic;
pub mod hostmem;
#[cfg(unix)]
pub mod signal;

pub use hostmem::HostMemory;
