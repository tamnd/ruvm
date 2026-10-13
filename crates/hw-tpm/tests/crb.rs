// SPDX-License-Identifier: GPL-2.0-or-later

//! The CRB registers and command buffer, driven the way the Linux tpm_crb driver does, over a
//! backend that answers every command with a response of a set length.

use std::sync::{Arc, Mutex};

use ruvm_hw_tpm::crb::{TPM_CRB_CMD_SIZE, TpmCrb};
use ruvm_hw_tpm::{TpmBackend, TpmCmd, TpmVersion};
use ruvm_mem::{AccessCtx, AccessSize, MemTxAttrs, MmioOps};

const CX: AccessCtx = AccessCtx::new(MemTxAttrs::UNSPECIFIED);

const LOC_STATE: u64 = 0x00;
const LOC_CTRL: u64 = 0x08;
const LOC_STS: u64 = 0x0c;
const INTF_ID: u64 = 0x30;
const INTF_ID2: u64 = 0x34;
const CTRL_REQ: u64 = 0x40;
const CTRL_STS: u64 = 0x44;
const CTRL_START: u64 = 0x4c;
const CTRL_CMD_SIZE: u64 = 0x58;
const CTRL_CMD_LADDR: u64 = 0x5c;
const CTRL_RSP_SIZE: u64 = 0x64;
const CTRL_RSP_ADDR: u64 = 0x68;

#[derive(Default)]
struct Log {
    startups: Vec<usize>,
    requests: Vec<Vec<u8>>,
}

struct Fake {
    buffer_size: usize,
    resp_len: usize,
    log: Arc<Mutex<Log>>,
}

impl TpmBackend for Fake {
    fn version(&self) -> TpmVersion {
        TpmVersion::V2_0
    }

    fn buffer_size(&mut self) -> usize {
        self.buffer_size
    }

    fn startup(&mut self, buffer_size: usize) -> Result<(), String> {
        self.log.lock().unwrap().startups.push(buffer_size);
        Ok(())
    }

    fn handle_request(&mut self, cmd: &mut TpmCmd<'_>) -> Result<(), String> {
        self.log.lock().unwrap().requests.push(cmd.input.to_vec());
        let out = &mut cmd.output[..self.resp_len];
        for (i, b) in out.iter_mut().enumerate() {
            *b = i as u8;
        }
        out[..6].copy_from_slice(&[0x80, 0x01, 0, 0, 0, 0]);
        out[2..6].copy_from_slice(&(self.resp_len as u32).to_be_bytes());
        Ok(())
    }

    fn cancel(&mut self) {}
}

struct Dev {
    crb: Arc<TpmCrb>,
    cmd: Arc<dyn MmioOps>,
    log: Arc<Mutex<Log>>,
}

impl Dev {
    fn new(buffer_size: usize, resp_len: usize, cap_chunk: bool) -> Dev {
        let log = Arc::new(Mutex::new(Log::default()));
        let crb =
            TpmCrb::new(Box::new(Fake { buffer_size, resp_len, log: log.clone() }), cap_chunk);
        crb.reset().unwrap();
        let cmd = crb.cmd_region();
        Dev { crb, cmd, log }
    }

    fn read(&self, addr: u64) -> u32 {
        self.crb.read(&CX, addr, AccessSize::B4).unwrap() as u32
    }

    fn write(&self, addr: u64, val: u32) {
        self.crb.write(&CX, addr, AccessSize::B4, val.into()).unwrap();
    }

    fn put(&self, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            self.cmd.write(&CX, i as u64, AccessSize::B1, b.into()).unwrap();
        }
    }

    fn get(&self, len: usize) -> Vec<u8> {
        (0..len).map(|i| self.cmd.read(&CX, i as u64, AccessSize::B1).unwrap() as u8).collect()
    }

    fn claim(&self) {
        self.write(LOC_CTRL, 1);
        self.write(CTRL_REQ, 1);
    }
}

/// A command of `len` bytes with its size in the header.
fn command(len: usize) -> Vec<u8> {
    let mut c: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
    c[..2].copy_from_slice(&[0x80, 0x01]);
    c[2..6].copy_from_slice(&(len as u32).to_be_bytes());
    c
}

#[test]
fn registers_after_reset() {
    let d = Dev::new(4096, 12, true);
    assert_eq!(d.log.lock().unwrap().startups, [4096]);
    // tpmRegValidSts, and tpmEstablished from a backend that reports the flag clear.
    assert_eq!(d.read(LOC_STATE), 0x81);
    assert_eq!(d.read(CTRL_STS), 2);
    assert_eq!(d.read(INTF_ID), 0x0002_5c11);
    assert_eq!(d.read(INTF_ID2), 0x1014);
    assert_eq!(d.read(CTRL_CMD_SIZE), 0xf80);
    assert_eq!(d.read(CTRL_CMD_LADDR), 0xfed4_0080);
    assert_eq!(d.read(CTRL_RSP_SIZE), 0xf80);
    assert_eq!(d.read(CTRL_RSP_ADDR), 0xfed4_0080);
    // Narrow reads see the bytes of the register.
    let byte = |addr| d.crb.read(&CX, addr, AccessSize::B1).unwrap();
    assert_eq!(byte(INTF_ID + 1), 0x5c);
    assert_eq!(byte(CTRL_CMD_LADDR + 3), 0xfe);

    let d = Dev::new(4096, 12, false);
    assert_eq!(d.read(INTF_ID), 0x0002_5811);
}

#[test]
fn locality_and_idle() {
    let d = Dev::new(4096, 12, true);
    d.write(LOC_CTRL, 1);
    assert_eq!(d.read(LOC_STATE), 0x83);
    assert_eq!(d.read(LOC_STS), 1);
    d.write(CTRL_REQ, 1);
    assert_eq!(d.read(CTRL_STS), 0);
    d.write(CTRL_REQ, 2);
    assert_eq!(d.read(CTRL_STS), 2);
    d.write(LOC_CTRL, 2);
    assert_eq!(d.read(LOC_STATE), 0x81);
    assert_eq!(d.read(LOC_STS), 0);
}

#[test]
fn command_round_trip() {
    let d = Dev::new(4096, 12, true);
    let cmd = command(22);
    d.put(&cmd);
    // Start without the locality does nothing.
    d.write(CTRL_START, 1);
    assert!(d.log.lock().unwrap().requests.is_empty());

    d.claim();
    d.write(CTRL_START, 1);
    assert_eq!(d.read(CTRL_START), 0);
    assert_eq!(d.log.lock().unwrap().requests, [cmd]);
    let resp = d.get(16);
    assert_eq!(resp[..12], [0x80, 0x01, 0, 0, 0, 12, 6, 7, 8, 9, 10, 11]);
    // The rest of the buffer is cleared.
    assert_eq!(resp[12..], [0; 4]);
    assert_eq!(d.read(CTRL_STS) & 1, 0);
}

#[test]
fn short_command_is_an_error() {
    let d = Dev::new(4096, 12, true);
    d.claim();
    d.put(&[0x80, 0x01, 0, 0, 0, 4]);
    d.write(CTRL_START, 1);
    assert!(d.log.lock().unwrap().requests.is_empty());
    assert_eq!(d.read(CTRL_STS) & 1, 1);
}

#[test]
fn chunks_both_ways() {
    let n = TPM_CRB_CMD_SIZE;
    let d = Dev::new(8192, n + 100, true);
    d.claim();
    let cmd = command(n + 50);
    d.put(&cmd[..n]);
    d.write(CTRL_START, 4);
    assert_eq!(d.read(CTRL_START), 0);
    d.put(&cmd[n..]);
    d.write(CTRL_START, 1);
    assert_eq!(d.log.lock().unwrap().requests, [cmd]);

    let first = d.get(n);
    assert_eq!(first[..6], [0x80, 0x01, 0, 0, ((n + 100) >> 8) as u8, (n + 100) as u8]);
    assert_eq!(first[n - 1], (n - 1) as u8);
    d.write(CTRL_START, 4);
    let second = d.get(n);
    let want: Vec<u8> = (n..n + 100).map(|i| i as u8).collect();
    assert_eq!(second[..100], want[..]);
    assert!(second[100..].iter().all(|&b| b == 0));
    // A retry starts the response over.
    d.write(CTRL_START, 2);
    assert_eq!(d.get(n), first);
}

#[test]
fn no_chunks_without_the_capability() {
    let n = TPM_CRB_CMD_SIZE;
    let d = Dev::new(8192, n + 100, false);
    d.claim();
    let cmd = command(n + 50);
    d.put(&cmd[..n]);
    d.write(CTRL_START, 4);
    // nextChunk is ignored, so Start sends what fits.
    d.write(CTRL_START, 1);
    assert_eq!(d.log.lock().unwrap().requests, [cmd[..n].to_vec()]);
}
