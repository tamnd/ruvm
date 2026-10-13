// SPDX-License-Identifier: GPL-2.0-or-later

//! `-tpmdev emulator`, backends/tpm/tpm_emulator.c: a TPM in another process, normally swtpm.
//!
//! The control channel is the socket of the `chardev` property, which speaks the protocol of
//! backends/tpm/tpm_ioctl.h: a big endian command number, its request, then a response that
//! starts with a big endian result code. The TPM commands themselves go over a second socket,
//! one end of a socket pair whose other end is handed to the emulator with `CMD_SET_DATAFD`.
//!
//! State blobs and migration are not supported yet.

use std::io::{IoSlice, Read, Write};
use std::mem::MaybeUninit;
use std::os::fd::AsFd;
use std::os::unix::net::UnixStream;
use std::time::Duration;

use rustix::net::{SendAncillaryBuffer, SendAncillaryMessage, SendFlags};

use crate::backend::{TpmBackend, TpmCmd};
use crate::util::{self, TPM_HDR_SIZE, TpmVersion};

/// The control commands, `CMD_*` in tpm_ioctl.h.
const CMD_GET_CAPABILITY: u32 = 0x01;
const CMD_INIT: u32 = 0x02;
const CMD_SHUTDOWN: u32 = 0x03;
const CMD_GET_TPMESTABLISHED: u32 = 0x04;
const CMD_SET_LOCALITY: u32 = 0x05;
const CMD_CANCEL_TPM_CMD: u32 = 0x09;
const CMD_RESET_TPMESTABLISHED: u32 = 0x0b;
const CMD_STOP: u32 = 0x0e;
const CMD_SET_DATAFD: u32 = 0x10;
const CMD_SET_BUFFERSIZE: u32 = 0x11;

/// The capability bits of `CMD_GET_CAPABILITY`, `PTM_CAP_*`.
pub const PTM_CAP_INIT: u32 = 1;
pub const PTM_CAP_SHUTDOWN: u32 = 1 << 1;
pub const PTM_CAP_GET_TPMESTABLISHED: u32 = 1 << 2;
pub const PTM_CAP_SET_LOCALITY: u32 = 1 << 3;
pub const PTM_CAP_CANCEL_TPM_CMD: u32 = 1 << 5;
pub const PTM_CAP_RESET_TPMESTABLISHED: u32 = 1 << 7;
pub const PTM_CAP_STOP: u32 = 1 << 10;
pub const PTM_CAP_SET_DATAFD: u32 = 1 << 12;
pub const PTM_CAP_SET_BUFFERSIZE: u32 = 1 << 13;

/// The minimum set `tpm_emulator_check_caps()` wants for a TPM 1.2.
const CAPS_1_2: u32 = PTM_CAP_INIT
    | PTM_CAP_SHUTDOWN
    | PTM_CAP_GET_TPMESTABLISHED
    | PTM_CAP_SET_LOCALITY
    | PTM_CAP_SET_DATAFD
    | PTM_CAP_STOP
    | PTM_CAP_SET_BUFFERSIZE;
/// And for a TPM 2.
const CAPS_2_0: u32 = CAPS_1_2 | PTM_CAP_RESET_TPMESTABLISHED;

/// `PTM_INIT_FLAG_DELETE_VOLATILE`.
const PTM_INIT_FLAG_DELETE_VOLATILE: u32 = 1;

/// `tpm_emulator_strerror()`.
fn strerror(res: u32) -> &'static str {
    match res {
        3 => "a parameter is bad",
        9 => "operation failed",
        13 => "key could not be found",
        25 => "bad parameter size",
        32 => "encryption error",
        33 => "decryption error",
        40 => "bad key property",
        44 => "bad (encryption) mode",
        46 => "bad version identifier",
        61 => "bad locality",
        0x101 => "operation failed",
        0x907 => "bad locality",
        0x9a => "insufficient amount of data",
        _ => "",
    }
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The `TPMEmulator` state.
#[derive(Debug)]
pub struct TpmEmulator {
    ctrl: UnixStream,
    data: UnixStream,
    version: TpmVersion,
    caps: u32,
    /// `cur_locty_number`, starting out as `~0`.
    cur_locty: u8,
    /// `established_flag` once `established_flag_cached` is set.
    established: Option<bool>,
}

/// One control command, `tpm_emulator_ctrlcmd()`: `msg` holds the request and gets the
/// response. `out_err` bytes are read first; the rest up to `out_total` only when the result
/// code in them is 0.
fn ctrlcmd(
    ctrl: &mut UnixStream,
    cmd: u32,
    msg: &mut [u8],
    len_in: usize,
    out_err: usize,
    out_total: usize,
) -> std::io::Result<()> {
    let mut buf = cmd.to_be_bytes().to_vec();
    buf.extend_from_slice(&msg[..len_in]);
    ctrl.write_all(&buf)?;
    read_response(ctrl, msg, out_err, out_total)
}

fn read_response(
    ctrl: &mut UnixStream,
    msg: &mut [u8],
    out_err: usize,
    out_total: usize,
) -> std::io::Result<()> {
    if out_total == 0 {
        return Ok(());
    }
    ctrl.read_exact(&mut msg[..out_err])?;
    if out_err == out_total || be32(msg, 0) != 0 {
        return Ok(());
    }
    ctrl.read_exact(&mut msg[out_err..out_total])
}

/// `tpm_util_request()`: one TPM command on `fd`, with a second for the response.
fn util_request(data: &mut UnixStream, request: &[u8], response: &mut [u8]) -> Result<usize, ()> {
    data.write_all(request).map_err(|_| ())?;
    data.set_read_timeout(Some(Duration::from_secs(1))).map_err(|_| ())?;
    let n = data.read(response);
    data.set_read_timeout(None).map_err(|_| ())?;
    let n = n.map_err(|_| ())?;
    if n < TPM_HDR_SIZE || util::cmd_size(response) as usize != n {
        return Err(());
    }
    Ok(n)
}

/// `tpm_util_test_tpmdev()`: which TPM answers on `data`.
fn test_tpmdev(data: &mut UnixStream) -> Option<TpmVersion> {
    let mut buf = [0; 1024];
    let tpm2 = util::request_header(util::TPM2_ST_NO_SESSIONS, util::TPM2_CC_READ_CLOCK);
    if util_request(data, &tpm2, &mut buf).is_ok()
        && util::cmd_tag(&buf) == util::TPM2_ST_NO_SESSIONS
    {
        return Some(TpmVersion::V2_0);
    }
    let tpm12 = util::request_header(util::TPM_TAG_RQU_COMMAND, util::TPM_ORD_GET_TICKS);
    if util_request(data, &tpm12, &mut buf).is_ok()
        && util::cmd_tag(&buf) == util::TPM_TAG_RSP_COMMAND
    {
        return Some(TpmVersion::V1_2);
    }
    None
}

impl TpmEmulator {
    /// `tpm_emulator_handle_device_opts()` once the chardev is connected: hands the emulator
    /// its data socket, probes the TPM version and checks the capabilities. `chardev` is the
    /// id for the error messages.
    pub fn new(mut ctrl: UnixStream, chardev: &str) -> Result<TpmEmulator, String> {
        let datafd_err = || "tpm-emulator: Failed to send CMD_SET_DATAFD".to_string();
        let (mut data, peer) = UnixStream::pair()
            .map_err(|_| "tpm-emulator: Failed to create socketpair".to_string())?;
        let fds = [peer.as_fd()];
        let mut space = [MaybeUninit::<u8>::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        if !control.push(SendAncillaryMessage::ScmRights(&fds)) {
            return Err(datafd_err());
        }
        let cmd = CMD_SET_DATAFD.to_be_bytes();
        let sent =
            rustix::net::sendmsg(&ctrl, &[IoSlice::new(&cmd)], &mut control, SendFlags::empty());
        if sent != Ok(cmd.len()) {
            return Err(datafd_err());
        }
        let mut res = [0; 4];
        if read_response(&mut ctrl, &mut res, 4, 4).is_err() || res != [0; 4] {
            return Err(datafd_err());
        }
        drop(peer);

        let Some(version) = test_tpmdev(&mut data) else {
            return Err(format!("'{chardev}' is not emulating TPM device."));
        };
        let mut emu =
            TpmEmulator { ctrl, data, version, caps: 0, cur_locty: !0, established: None };
        emu.probe_caps()?;
        let (want, name) = match version {
            TpmVersion::V1_2 => (CAPS_1_2, "1.2"),
            TpmVersion::V2_0 => (CAPS_2_0, "2"),
        };
        if emu.caps & want != want {
            return Err(format!(
                "tpm-emulator: TPM does not implement minimum set of required capabilities for TPM {name} (0x{want:x})"
            ));
        }
        Ok(emu)
    }

    /// The capabilities the emulator reported.
    pub fn caps(&self) -> u32 {
        self.caps
    }

    fn probe_caps(&mut self) -> Result<(), String> {
        let mut msg = [0; 8];
        ctrlcmd(&mut self.ctrl, CMD_GET_CAPABILITY, &mut msg, 0, 4, 8)
            .map_err(|_| "tpm-emulator: probing failed".to_string())?;
        self.caps = be32(&msg, 4);
        Ok(())
    }

    /// A command whose request and response are a bare result code.
    fn simple(&mut self, cmd: u32) -> std::io::Result<u32> {
        let mut res = [0; 4];
        ctrlcmd(&mut self.ctrl, cmd, &mut res, 0, 4, 4)?;
        Ok(u32::from_be_bytes(res))
    }

    fn set_locality(&mut self, locty: u8) -> Result<(), String> {
        if self.cur_locty == locty {
            return Ok(());
        }
        let mut msg = [locty, 0, 0, 0];
        ctrlcmd(&mut self.ctrl, CMD_SET_LOCALITY, &mut msg, 4, 4, 4)
            .map_err(|_| "tpm-emulator: could not set locality".to_string())?;
        let res = be32(&msg, 0);
        if res != 0 {
            return Err(format!("tpm-emulator: TPM result for set locality : 0x{res:x}"));
        }
        self.cur_locty = locty;
        Ok(())
    }

    /// `tpm_emulator_unix_tx_bufs()`.
    fn tx_bufs(&mut self, cmd: &mut TpmCmd<'_>) -> Result<(), String> {
        let is_selftest = util::is_selftest(cmd.input);
        cmd.selftest_done = false;
        let io = |e: std::io::Error| e.to_string();
        self.data.write_all(cmd.input).map_err(io)?;
        let out = &mut *cmd.output;
        self.data.read_exact(&mut out[..TPM_HDR_SIZE]).map_err(io)?;
        let to_read = util::cmd_size(out) as usize;
        if to_read > out.len() {
            let _ = self.data.shutdown(std::net::Shutdown::Both);
            return Err(format!(
                "tpm-emulator: Disconnected after receiving unacceptable large response ({to_read} > {})",
                out.len()
            ));
        }
        if to_read > TPM_HDR_SIZE {
            self.data.read_exact(&mut out[TPM_HDR_SIZE..to_read]).map_err(io)?;
        }
        if is_selftest {
            cmd.selftest_done = util::cmd_errcode(out) == 0;
        }
        Ok(())
    }

    fn stop(&mut self) -> Result<(), String> {
        let res = self.simple(CMD_STOP).map_err(|_| "tpm-emulator: Could not stop TPM")?;
        if res != 0 {
            return Err(format!(
                "tpm-emulator: TPM result for CMD_STOP: 0x{res:x} {}",
                strerror(res)
            ));
        }
        Ok(())
    }

    /// `tpm_emulator_set_buffer_size()`: 0 only asks for the current size.
    pub fn set_buffer_size(&mut self, wanted: usize) -> Result<usize, String> {
        self.stop()?;
        let mut msg = [0; 16];
        msg[..4].copy_from_slice(&(wanted as u32).to_be_bytes());
        ctrlcmd(&mut self.ctrl, CMD_SET_BUFFERSIZE, &mut msg, 4, 4, 16)
            .map_err(|_| "tpm-emulator: Could not set buffer size".to_string())?;
        let res = be32(&msg, 0);
        if res != 0 {
            return Err(format!(
                "tpm-emulator: TPM result for set buffer size : 0x{res:x} {}",
                strerror(res)
            ));
        }
        let actual = be32(&msg, 4) as usize;
        if wanted != 0 && wanted != actual {
            return Err(format!(
                "tpm-emulator: TPM did not accept the requested buffer size of {wanted} bytes but adjusted it to {actual} bytes"
            ));
        }
        Ok(actual)
    }

    /// `tpm_emulator_startup_tpm_resume()`.
    fn startup_resume(&mut self, buffer_size: usize, is_resume: bool) -> Result<(), String> {
        if buffer_size != 0 {
            self.set_buffer_size(buffer_size)?;
        }
        let flags = if is_resume { PTM_INIT_FLAG_DELETE_VOLATILE } else { 0 };
        let mut msg = flags.to_be_bytes();
        ctrlcmd(&mut self.ctrl, CMD_INIT, &mut msg, 4, 4, 4)
            .map_err(|_| "tpm-emulator: could not send INIT".to_string())?;
        let res = u32::from_be_bytes(msg);
        if res != 0 {
            return Err(format!(
                "tpm-emulator: TPM result for CMD_INIT: 0x{res:x} {}",
                strerror(res)
            ));
        }
        Ok(())
    }

    /// `tpm_emulator_shutdown()`, which QEMU does when the backend goes away.
    pub fn shutdown(&mut self) -> Result<(), String> {
        let res = self
            .simple(CMD_SHUTDOWN)
            .map_err(|_| "tpm-emulator: Could not cleanly shutdown the TPM")?;
        if res != 0 {
            return Err(format!(
                "tpm-emulator: TPM result for shutdown: 0x{res:x} {}",
                strerror(res)
            ));
        }
        Ok(())
    }
}

impl TpmBackend for TpmEmulator {
    fn version(&self) -> TpmVersion {
        self.version
    }

    fn buffer_size(&mut self) -> usize {
        // tpm_emulator_get_buffer_size() reports the error and goes on with 4 KiB.
        self.set_buffer_size(0).unwrap_or(4096)
    }

    fn startup(&mut self, buffer_size: usize) -> Result<(), String> {
        self.startup_resume(buffer_size, false)
    }

    fn handle_request(&mut self, cmd: &mut TpmCmd<'_>) -> Result<(), String> {
        let r = self.set_locality(cmd.locty).and_then(|()| self.tx_bufs(cmd));
        if r.is_err() {
            util::write_fatal_error_response(cmd.output);
        }
        r
    }

    fn cancel(&mut self) {
        if self.caps & PTM_CAP_CANCEL_TPM_CMD == 0 {
            return;
        }
        // Errors only get reported in QEMU.
        let _ = self.simple(CMD_CANCEL_TPM_CMD);
    }

    fn established_flag(&mut self) -> bool {
        if let Some(f) = self.established {
            return f;
        }
        let mut msg = [0; 8];
        if ctrlcmd(&mut self.ctrl, CMD_GET_TPMESTABLISHED, &mut msg, 0, 8, 8).is_err() {
            return false;
        }
        let f = msg[4] != 0;
        self.established = Some(f);
        f
    }

    fn reset_established_flag(&mut self, _locty: u8) -> Result<(), String> {
        // Only a TPM 2 supports this, and QEMU sends the current locality, not the one asked.
        if self.version != TpmVersion::V2_0 {
            return Ok(());
        }
        let mut msg = [self.cur_locty, 0, 0, 0];
        ctrlcmd(&mut self.ctrl, CMD_RESET_TPMESTABLISHED, &mut msg, 4, 4, 4)
            .map_err(|_| "tpm-emulator: Could not reset the establishment bit".to_string())?;
        let res = be32(&msg, 0);
        if res != 0 {
            return Err(format!(
                "tpm-emulator: TPM result for rest established flag: 0x{res:x} {}",
                strerror(res)
            ));
        }
        self.established = None;
        Ok(())
    }
}
