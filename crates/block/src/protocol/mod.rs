// SPDX-License-Identifier: GPL-2.0-or-later

//! Protocol drivers other than `file`: `host_device` and `host_cdrom`, and the Linux asynchronous I/O back ends.

#[cfg(unix)]
pub(crate) mod aio;
#[cfg(any(target_os = "linux", target_os = "macos"))]
pub(crate) mod host;
