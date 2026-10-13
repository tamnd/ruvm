// SPDX-License-Identifier: GPL-2.0-or-later

//! The emulator backend and the TIS against a real swtpm, when one is installed.

#![cfg(unix)]

mod common;

use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::transact;
use ruvm_hw_tpm::emulator::TpmEmulator;
use ruvm_hw_tpm::tis::TpmTis;
use ruvm_hw_tpm::{TpmBackend, TpmVersion};

struct Swtpm {
    child: Child,
    dir: PathBuf,
}

impl Drop for Swtpm {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Starts `swtpm socket` the way QEMU's docs/specs/tpm.rst does and connects to its control
/// socket, or gives `None` when there is no swtpm.
fn swtpm(tpm2: bool) -> Option<(Swtpm, UnixStream)> {
    let dir = std::env::temp_dir().join(format!("ruvm-swtpm-{}-{tpm2}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ctrl = dir.join("sock");
    let mut cmd = Command::new("swtpm");
    cmd.arg("socket")
        .arg("--tpmstate")
        .arg(format!("dir={}", dir.display()))
        .arg("--ctrl")
        .arg(format!("type=unixio,path={}", ctrl.display()))
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if tpm2 {
        cmd.arg("--tpm2");
    }
    let Ok(child) = cmd.spawn() else {
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("no swtpm, skipping");
        return None;
    };
    let s = Swtpm { child, dir };
    Some((s, connect(&ctrl)))
}

fn connect(path: &Path) -> UnixStream {
    for _ in 0..500 {
        if let Ok(c) = UnixStream::connect(path) {
            return c;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("swtpm did not come up at {}", path.display());
}

fn rc(resp: &[u8]) -> u32 {
    u32::from_be_bytes(resp[6..10].try_into().unwrap())
}

/// `TPM2_Startup(TPM_SU_CLEAR)`, which firmware sends first.
const TPM2_STARTUP: [u8; 12] = [0x80, 0x01, 0, 0, 0, 12, 0, 0, 0x01, 0x44, 0, 0];

/// `TPM2_PCR_Read` of SHA-256 PCRs 0 and 1.
const TPM2_PCR_READ: [u8; 20] =
    [0x80, 0x01, 0, 0, 0, 20, 0, 0, 0x01, 0x7e, 0, 0, 0, 1, 0, 0x0b, 3, 0x03, 0, 0];

/// `TPM2_PCR_Extend` of SHA-256 PCR 1 with a digest of 0x11 bytes, no session.
fn pcr_extend() -> Vec<u8> {
    let mut c = vec![0x80, 0x02, 0, 0, 0, 0, 0, 0, 0x01, 0x82];
    c.extend_from_slice(&1u32.to_be_bytes());
    // An empty password session for the PCR's authorization.
    c.extend_from_slice(&9u32.to_be_bytes());
    c.extend_from_slice(&[0x40, 0, 0, 0x09, 0, 0, 0x01, 0, 0]);
    c.extend_from_slice(&1u32.to_be_bytes());
    c.extend_from_slice(&[0, 0x0b]);
    c.extend_from_slice(&[0x11; 32]);
    let len = c.len() as u32;
    c[2..6].copy_from_slice(&len.to_be_bytes());
    c
}

#[test]
fn tpm2_through_tis() {
    let Some((_s, ctrl)) = swtpm(true) else { return };
    let mut emu = TpmEmulator::new(ctrl, "chrtpm").unwrap();
    assert_eq!(emu.version(), TpmVersion::V2_0);
    assert_eq!(emu.buffer_size(), 4096);
    let tis = TpmTis::new(Box::new(emu), 5);
    tis.reset().unwrap();
    assert_eq!(tis.tpm_version(), Some(TpmVersion::V2_0));

    assert_eq!(rc(&transact(&tis, 0, &TPM2_STARTUP)), 0);
    let r = transact(&tis, 0, &TPM2_PCR_READ);
    assert_eq!(rc(&r), 0);
    // Header, update counter, the selection, a count of 2, then two sized digests of zeros.
    assert_eq!(r.len(), 10 + 4 + 10 + 4 + 2 * 34);
    assert_eq!(&r[24..28], &2u32.to_be_bytes());
    assert!(r[28..].chunks(34).all(|d| d[..2] == [0, 32] && d[2..].iter().all(|&b| b == 0)));

    let r = transact(&tis, 0, &pcr_extend());
    assert_eq!(rc(&r), 0, "{r:02x?}");
    let r = transact(&tis, 0, &TPM2_PCR_READ);
    // PCR 0 stays zero and PCR 1 changed.
    assert!(r[30..62].iter().all(|&b| b == 0));
    assert!(r[64..96].iter().any(|&b| b != 0));
}

#[test]
fn tpm12_version_probe() {
    let Some((_s, ctrl)) = swtpm(false) else { return };
    let emu = TpmEmulator::new(ctrl, "chrtpm").unwrap();
    assert_eq!(emu.version(), TpmVersion::V1_2);
    let tis = TpmTis::new(Box::new(emu), 5);
    tis.reset().unwrap();
    assert_eq!(tis.read_data(0x14, 4), 0x2000_0697);
}

#[test]
fn not_a_tpm() {
    // A peer that takes the data socket and closes everything fails the probe the way QEMU
    // reports it.
    let (ctrl, mut peer) = UnixStream::pair().unwrap();
    let t = std::thread::spawn(move || {
        use std::io::{Read, Write};
        let mut cmd = [0; 4];
        peer.read_exact(&mut cmd).unwrap();
        assert_eq!(cmd, 0x10u32.to_be_bytes());
        peer.write_all(&[0; 4]).unwrap();
    });
    let e = TpmEmulator::new(ctrl, "chrtpm").unwrap_err();
    t.join().unwrap();
    assert_eq!(e, "'chrtpm' is not emulating TPM device.");
}
