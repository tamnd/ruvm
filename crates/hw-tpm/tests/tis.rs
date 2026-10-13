// SPDX-License-Identifier: GPL-2.0-or-later

//! The TIS state machine over a backend that answers every command with a fixed response.

mod common;

use std::sync::{Arc, Mutex};

use common::*;
use ruvm_hw_tpm::tis::TpmTis;
use ruvm_hw_tpm::{TpmBackend, TpmCmd, TpmVersion};

#[derive(Default)]
struct Log {
    startups: Vec<usize>,
    requests: Vec<(u8, Vec<u8>)>,
}

struct Fake {
    version: TpmVersion,
    log: Arc<Mutex<Log>>,
}

impl TpmBackend for Fake {
    fn version(&self) -> TpmVersion {
        self.version
    }

    fn buffer_size(&mut self) -> usize {
        4096
    }

    fn startup(&mut self, buffer_size: usize) -> Result<(), String> {
        self.log.lock().unwrap().startups.push(buffer_size);
        Ok(())
    }

    fn handle_request(&mut self, cmd: &mut TpmCmd<'_>) -> Result<(), String> {
        self.log.lock().unwrap().requests.push((cmd.locty, cmd.input.to_vec()));
        let resp = [0x80, 0x01, 0, 0, 0, 0x0c, 0, 0, 0, 0, 0xab, 0xcd];
        cmd.output[..resp.len()].copy_from_slice(&resp);
        Ok(())
    }

    fn cancel(&mut self) {}
}

fn tis(version: TpmVersion) -> (Arc<TpmTis>, Arc<Mutex<Log>>) {
    let log = Arc::new(Mutex::new(Log::default()));
    let t = TpmTis::new(Box::new(Fake { version, log: log.clone() }), 5);
    t.reset().unwrap();
    (t, log)
}

#[test]
fn identity_registers() {
    let (t, log) = tis(TpmVersion::V2_0);
    assert_eq!(log.lock().unwrap().startups, [4096]);
    assert_eq!(t.read_data(DID_VID, 4), 0x0001_1014);
    assert_eq!(t.read_data(RID, 4), 1);
    assert_eq!(t.read_data(INTF_CAPABILITY, 4), 0x3000_0697);
    assert_eq!(t.read_data(INTERFACE_ID, 4), 0x2100);
    // Reset leaves no locality active, the access register valid and the established bit
    // clear, which reads as 1.
    assert_eq!(t.read_data(ACCESS, 1), 0x81);
    assert_eq!(t.read_data(INT_ENABLE, 4), 1 << 3);
    assert_eq!(t.read_data(reg(4, 0x0c), 4), 5);
    // STS reads all ones from a locality that is not active.
    assert_eq!(t.read_data(STS, 4), 0xffff_ffff);

    let (t, _) = tis(TpmVersion::V1_2);
    assert_eq!(t.read_data(INTF_CAPABILITY, 4), 0x2000_0697);
    assert_eq!(t.read_data(INTERFACE_ID, 4), 0xffff_ffff);
}

#[test]
fn command_round_trip() {
    let (t, log) = tis(TpmVersion::V2_0);
    let cmd = [0x80, 0x01, 0, 0, 0, 0x0c, 0, 0, 0x01, 0x44, 0, 0];
    let resp = transact(&t, 0, &cmd);
    assert_eq!(resp, [0x80, 0x01, 0, 0, 0, 0x0c, 0, 0, 0, 0, 0xab, 0xcd]);
    assert_eq!(log.lock().unwrap().requests, [(0, cmd.to_vec())]);
    let resp = transact(&t, 2, &cmd);
    assert_eq!(resp.len(), 12);
    assert_eq!(log.lock().unwrap().requests[1].0, 2);
}

#[test]
fn burst_count_and_wide_fifo_access() {
    let (t, _) = tis(TpmVersion::V2_0);
    t.write_data(ACCESS, ACCESS_REQUEST_USE.into(), 1);
    t.write_data(STS, STS_COMMAND_READY.into(), 1);
    // A byte read caps the burst count at 0xff, a wider one shows the whole buffer.
    assert_eq!(t.read_data(STS + 1, 1) & 0xff, 0xff);
    assert_eq!(burst(&t, 0), 4096);
    // Four bytes at a time through the FIFO, then the rest one by one.
    t.write_data(FIFO, 0x0000_0180, 4);
    assert_eq!(burst(&t, 0), 4092);
    t.write_data(FIFO, 0x0000_0c00, 4);
    t.write_data(FIFO, 0x0000_4401, 4);
    assert_eq!(t.read_data(STS, 1) & STS_EXPECT, 0);
    t.write_data(STS, STS_GO.into(), 1);
    assert_eq!(burst(&t, 0), 12);
    assert_eq!(t.read_data(FIFO, 4), 0x0000_0180);
    assert_eq!(burst(&t, 0), 8);
    // Response retry starts the response over.
    t.write_data(STS, (1u32 << 1).into(), 1);
    assert_eq!(burst(&t, 0), 12);
}

#[test]
fn locality_handover_and_seize() {
    let (t, _) = tis(TpmVersion::V2_0);
    t.write_data(reg(0, ACCESS), ACCESS_REQUEST_USE.into(), 1);
    assert_eq!(t.read_data(reg(0, ACCESS), 1), 0xa1);
    // Locality 2 asks too; 0 sees the pending request.
    t.write_data(reg(2, ACCESS), ACCESS_REQUEST_USE.into(), 1);
    assert_eq!(t.read_data(reg(2, ACCESS), 1), 0x83);
    assert_eq!(t.read_data(reg(0, ACCESS), 1), 0xa5);
    // 0 gives up, 2 gets it.
    t.write_data(reg(0, ACCESS), ACCESS_ACTIVE_LOCALITY.into(), 1);
    assert_eq!(t.read_data(reg(0, ACCESS), 1), 0x81);
    assert_eq!(t.read_data(reg(2, ACCESS), 1), 0xa1);
    // 3 seizes it, and 2 is told it was seized.
    t.write_data(reg(3, ACCESS), 1 << 3, 1);
    assert_eq!(t.read_data(reg(3, ACCESS), 1), 0xa1);
    assert_eq!(t.read_data(reg(2, ACCESS), 1), 0x91);
    t.write_data(reg(2, ACCESS), 1 << 4, 1);
    assert_eq!(t.read_data(reg(2, ACCESS), 1), 0x81);
    // A lower locality cannot seize.
    t.write_data(reg(1, ACCESS), 1 << 3, 1);
    assert_eq!(t.read_data(reg(3, ACCESS), 1), 0xa1);
    // Writes to locality 4 are dropped.
    t.write_data(reg(4, ACCESS), ACCESS_REQUEST_USE.into(), 1);
    assert_eq!(t.read_data(reg(4, ACCESS), 1), 0x81);
}

#[test]
fn interrupts() {
    let (t, _) = tis(TpmVersion::V2_0);
    let level = Arc::new(Mutex::new(Vec::new()));
    let l = level.clone();
    t.irq().connect(ruvm_hw_core::irq::IrqLine::from_fn(move |v| l.lock().unwrap().push(v)));
    t.write_data(INT_ENABLE, 0x8000_0000 | 1 << 2 | 1 << 7, 4);
    t.write_data(ACCESS, ACCESS_REQUEST_USE.into(), 1);
    assert_eq!(t.read_data(INT_STATUS, 4), 1 << 2);
    t.write_data(STS, STS_COMMAND_READY.into(), 1);
    assert_eq!(t.read_data(INT_STATUS, 4), 1 << 2 | 1 << 7);
    t.write_data(INT_STATUS, 1 << 2 | 1 << 7, 4);
    assert_eq!(t.read_data(INT_STATUS, 4), 0);
    assert_eq!(*level.lock().unwrap(), [1, 1, 0]);
}
