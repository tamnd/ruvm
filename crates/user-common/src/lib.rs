// SPDX-License-Identifier: GPL-2.0-or-later

//! Code shared by linux-user and bsd-user: the guest address space and its mmap engine, the
//! part of QEMU's `accel/tcg/user-exec.c` and `linux-user/mmap.c` that does not depend on the
//! guest's system call ABI.

#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod space;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub use space::{GuestSpace, MapKind, PAGE_SIZE, page, page_align};
