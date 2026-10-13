// SPDX-License-Identifier: GPL-2.0-or-later

//! The backend interface, `TPMBackendClass` in include/system/tpm_backend.h.

use crate::util::TpmVersion;

/// `TPMBackendCmd`: a request for the TPM and the room for its response.
#[derive(Debug)]
pub struct TpmCmd<'a> {
    /// The locality the request was sent from.
    pub locty: u8,
    /// The request.
    pub input: &'a [u8],
    /// Where the response goes, as big as the negotiated buffer.
    pub output: &'a mut [u8],
    /// Set when the request was a `TPM_ContinueSelfTest` that succeeded.
    pub selftest_done: bool,
}

/// A TPM backend: `-tpmdev emulator` or `-tpmdev passthrough`.
///
/// QEMU runs `handle_request` in a worker thread and completes the request in a bottom half.
/// Here the front end calls it directly, so the vCPU that starts a command waits for the
/// response.
pub trait TpmBackend: Send {
    /// `get_tpm_version`.
    fn version(&self) -> TpmVersion;

    /// `get_buffer_size`: the size of the TPM's command buffer.
    fn buffer_size(&mut self) -> usize;

    /// `startup_tpm`: sets the buffer size, unless it is 0, and initializes the TPM.
    fn startup(&mut self, buffer_size: usize) -> Result<(), String>;

    /// `handle_request`. On failure the response is a `TPM_FAIL` and the error is returned
    /// for the caller to report.
    fn handle_request(&mut self, cmd: &mut TpmCmd<'_>) -> Result<(), String>;

    /// `cancel_cmd`.
    fn cancel(&mut self);

    /// `get_tpm_established_flag`.
    fn established_flag(&mut self) -> bool {
        false
    }

    /// `reset_tpm_established_flag`.
    fn reset_established_flag(&mut self, locty: u8) -> Result<(), String> {
        let _ = locty;
        Ok(())
    }
}
