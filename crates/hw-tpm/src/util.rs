// SPDX-License-Identifier: GPL-2.0-or-later

//! The TPM command header helpers of include/system/tpm_util.h and backends/tpm/tpm_util.c.

/// `TPM_TAG_RQU_COMMAND`.
pub const TPM_TAG_RQU_COMMAND: u16 = 0xc1;
/// `TPM_TAG_RSP_COMMAND`.
pub const TPM_TAG_RSP_COMMAND: u16 = 0xc4;
/// `TPM_FAIL`.
pub const TPM_FAIL: u32 = 9;
/// `TPM_ORD_ContinueSelfTest`.
pub const TPM_ORD_CONTINUE_SELF_TEST: u32 = 0x53;
/// `TPM_ORD_GetTicks`.
pub const TPM_ORD_GET_TICKS: u32 = 0xf1;
/// `TPM2_ST_NO_SESSIONS`.
pub const TPM2_ST_NO_SESSIONS: u16 = 0x8001;
/// `TPM2_CC_ReadClock`.
pub const TPM2_CC_READ_CLOCK: u32 = 0x181;

/// The size of `struct tpm_req_hdr` and `struct tpm_resp_hdr`.
pub const TPM_HDR_SIZE: usize = 10;

/// The TPM family a backend emulates, `TPMVersion` without `TPM_VERSION_UNSPEC`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TpmVersion {
    /// `TPM_VERSION_1_2`.
    V1_2,
    /// `TPM_VERSION_2_0`.
    V2_0,
}

fn be16(b: &[u8], at: usize) -> u16 {
    b.get(at..at + 2).map_or(0, |s| u16::from_be_bytes([s[0], s[1]]))
}

fn be32(b: &[u8], at: usize) -> u32 {
    b.get(at..at + 4).map_or(0, |s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// `tpm_cmd_get_tag()`.
pub fn cmd_tag(b: &[u8]) -> u16 {
    be16(b, 0)
}

/// `tpm_cmd_get_size()`, the length field of a request or a response.
pub fn cmd_size(b: &[u8]) -> u32 {
    be32(b, 2)
}

/// `tpm_cmd_get_ordinal()`.
pub fn cmd_ordinal(b: &[u8]) -> u32 {
    be32(b, 6)
}

/// `tpm_cmd_get_errcode()`.
pub fn cmd_errcode(b: &[u8]) -> u32 {
    be32(b, 6)
}

/// `tpm_util_write_fatal_error_response()`: a bare `TPM_FAIL` response in `out`.
pub fn write_fatal_error_response(out: &mut [u8]) {
    if out.len() >= TPM_HDR_SIZE {
        out[0..2].copy_from_slice(&TPM_TAG_RSP_COMMAND.to_be_bytes());
        out[2..6].copy_from_slice(&(TPM_HDR_SIZE as u32).to_be_bytes());
        out[6..10].copy_from_slice(&TPM_FAIL.to_be_bytes());
    }
}

/// `tpm_util_is_selftest()`: whether `cmd` is a TPM 1.2 `TPM_ContinueSelfTest`.
pub fn is_selftest(cmd: &[u8]) -> bool {
    cmd.len() >= TPM_HDR_SIZE && cmd_ordinal(cmd) == TPM_ORD_CONTINUE_SELF_TEST
}

/// A bare request header with `tag` and `ordinal`, the probes `tpm_util_test_tpmdev()` sends.
pub fn request_header(tag: u16, ordinal: u32) -> [u8; TPM_HDR_SIZE] {
    let mut h = [0; TPM_HDR_SIZE];
    h[0..2].copy_from_slice(&tag.to_be_bytes());
    h[2..6].copy_from_slice(&(TPM_HDR_SIZE as u32).to_be_bytes());
    h[6..10].copy_from_slice(&ordinal.to_be_bytes());
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_fields() {
        let h = request_header(TPM2_ST_NO_SESSIONS, TPM2_CC_READ_CLOCK);
        assert_eq!(h, [0x80, 0x01, 0, 0, 0, 10, 0, 0, 0x01, 0x81]);
        assert_eq!(cmd_tag(&h), TPM2_ST_NO_SESSIONS);
        assert_eq!(cmd_size(&h), 10);
        assert_eq!(cmd_ordinal(&h), TPM2_CC_READ_CLOCK);
        assert!(!is_selftest(&h));
        assert!(is_selftest(&request_header(TPM_TAG_RQU_COMMAND, TPM_ORD_CONTINUE_SELF_TEST)));
        // Short buffers read as zero instead of panicking.
        assert_eq!(cmd_size(&h[..4]), 0);
    }

    #[test]
    fn fatal_error() {
        let mut out = [0xaa; 12];
        write_fatal_error_response(&mut out);
        assert_eq!(out, [0, 0xc4, 0, 0, 0, 10, 0, 0, 0, 9, 0xaa, 0xaa]);
        let mut short = [0xaa; 9];
        write_fatal_error_response(&mut short);
        assert_eq!(short, [0xaa; 9]);
    }
}
