// SPDX-License-Identifier: GPL-2.0-or-later

//! Drives a TIS the way the Linux tpm_tis driver does.

#![allow(dead_code)]

use ruvm_hw_tpm::tis::TpmTis;

pub(crate) const ACCESS: u64 = 0x00;
pub(crate) const INT_ENABLE: u64 = 0x08;
pub(crate) const INT_STATUS: u64 = 0x10;
pub(crate) const INTF_CAPABILITY: u64 = 0x14;
pub(crate) const STS: u64 = 0x18;
pub(crate) const FIFO: u64 = 0x24;
pub(crate) const INTERFACE_ID: u64 = 0x30;
pub(crate) const DID_VID: u64 = 0xf00;
pub(crate) const RID: u64 = 0xf04;

pub(crate) const STS_VALID: u32 = 1 << 7;
pub(crate) const STS_COMMAND_READY: u32 = 1 << 6;
pub(crate) const STS_GO: u32 = 1 << 5;
pub(crate) const STS_DATA_AVAILABLE: u32 = 1 << 4;
pub(crate) const STS_EXPECT: u32 = 1 << 3;

pub(crate) const ACCESS_REQUEST_USE: u32 = 1 << 1;
pub(crate) const ACCESS_ACTIVE_LOCALITY: u32 = 1 << 5;

pub(crate) fn reg(locty: u8, r: u64) -> u64 {
    (u64::from(locty) << 12) | r
}

pub(crate) fn burst(tis: &TpmTis, locty: u8) -> u32 {
    (tis.read_data(reg(locty, STS), 4) >> 8) & 0xffff
}

/// Claims `locty`, sends `cmd` and reads the response.
pub(crate) fn transact(tis: &TpmTis, locty: u8, cmd: &[u8]) -> Vec<u8> {
    tis.write_data(reg(locty, ACCESS), ACCESS_REQUEST_USE.into(), 1);
    assert_ne!(tis.read_data(reg(locty, ACCESS), 1) & ACCESS_ACTIVE_LOCALITY, 0);
    tis.write_data(reg(locty, STS), STS_COMMAND_READY.into(), 1);
    assert_ne!(tis.read_data(reg(locty, STS), 1) & STS_COMMAND_READY, 0);
    for (i, &b) in cmd.iter().enumerate() {
        tis.write_data(reg(locty, FIFO), b.into(), 1);
        let sts = tis.read_data(reg(locty, STS), 1);
        assert_ne!(sts & STS_VALID, 0);
        // The TIS wants more until the length in the header is reached.
        assert_eq!(sts & STS_EXPECT != 0, i + 1 < cmd.len(), "byte {i}");
    }
    tis.write_data(reg(locty, STS), STS_GO.into(), 1);
    let sts = tis.read_data(reg(locty, STS), 1);
    assert_eq!(sts & (STS_VALID | STS_DATA_AVAILABLE), STS_VALID | STS_DATA_AVAILABLE);
    let mut out = Vec::new();
    while tis.read_data(reg(locty, STS), 1) & STS_DATA_AVAILABLE != 0 {
        out.push(tis.read_data(reg(locty, FIFO), 1) as u8);
    }
    tis.write_data(reg(locty, STS), STS_COMMAND_READY.into(), 1);
    tis.write_data(reg(locty, ACCESS), ACCESS_ACTIVE_LOCALITY.into(), 1);
    out
}
