// SPDX-License-Identifier: GPL-2.0-or-later

//! TPM TIS, CRB and the SPAPR vTPM, with the swtpm and passthrough backends.
//!
//! What is here so far: the TIS FIFO interface ([`tis`]), the PPI memory ([`ppi`]) and the
//! swtpm backend ([`emulator`]).

pub mod backend;
#[cfg(unix)]
pub mod emulator;
pub mod ppi;
pub mod tis;
pub mod util;

pub use backend::{TpmBackend, TpmCmd};
pub use util::TpmVersion;
