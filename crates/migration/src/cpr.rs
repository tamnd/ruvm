// SPDX-License-Identifier: GPL-2.0-or-later

//! The CPR state of migration/cpr.c and migration/cpr-transfer.c: what `cpr-transfer` hands to
//! the next process before the migration proper starts.
//!
//! With `cpr-transfer` the guest's memory is not copied. The next process maps the very same
//! memory, so before anything else the source connects to the `cpr` channel, a UNIX socket, and
//! sends its list of saved descriptors (the memfds of shared RAM among them) as the `CprState`
//! section, each descriptor riding along with a single space byte. The source then waits for
//! the destination to close that socket, which it does once it built its machine on those
//! descriptors and listens on the main channel, and only then connects the main channel.
//!
//! The stream is QEMU's: the magic `QCPR`, version 1, the `cpr fd` entries as a QLIST and the
//! `CprState/vfio devices` subsection, which is empty here since ruvm has no VFIO.

use ruvm_base::{Result, bail};
use ruvm_qapi::types::{MigrationAddressU, SocketAddressU};

use crate::channel::MigrationAddr;

/// `QEMU_CPR_FILE_MAGIC`, "QCPR".
pub const QEMU_CPR_FILE_MAGIC: u32 = 0x5143_5052;
/// `QEMU_CPR_FILE_VERSION`.
pub const QEMU_CPR_FILE_VERSION: u32 = 1;

const VMSTATE_SUBSECTION: u8 = 5;
const VFIO_SUBSECTION: &str = "CprState/vfio devices";

/// One descriptor in the stream: its name and id, with the descriptor in the order it came.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdEntry {
    pub name: String,
    pub id: i32,
}

/// Something to send: bytes, or the space byte that carries a descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Piece {
    Bytes(Vec<u8>),
    Fd(usize),
}

/// The pieces of the `CprState` stream for `entries`, where `Piece::Fd(i)` is the descriptor
/// of `entries[i]`.
pub fn encode(entries: &[FdEntry]) -> Vec<Piece> {
    let mut out = Vec::new();
    let mut b = Vec::new();
    b.extend_from_slice(&QEMU_CPR_FILE_MAGIC.to_be_bytes());
    b.extend_from_slice(&QEMU_CPR_FILE_VERSION.to_be_bytes());
    for (i, e) in entries.iter().enumerate() {
        // VMSTATE_QLIST_V: a byte 1 before each element.
        b.push(1);
        // namelen counts the terminating NUL, which goes out too.
        b.extend_from_slice(&(e.name.len() as u32 + 1).to_be_bytes());
        b.extend_from_slice(e.name.as_bytes());
        b.push(0);
        b.extend_from_slice(&e.id.to_be_bytes());
        // VMSTATE_FD: qemu_file_put_fd() flushes and sends the descriptor with a space.
        out.push(Piece::Bytes(std::mem::take(&mut b)));
        out.push(Piece::Fd(i));
    }
    b.push(0);
    // The subsection of VFIO devices, always there, with an empty list.
    b.push(VMSTATE_SUBSECTION);
    b.push(VFIO_SUBSECTION.len() as u8);
    b.extend_from_slice(VFIO_SUBSECTION.as_bytes());
    b.extend_from_slice(&1u32.to_be_bytes());
    b.push(0);
    out.push(Piece::Bytes(b));
    out
}

/// Reads the stream like a `QEMUFile` does: past the end come zeros and an error.
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    eof: bool,
}

impl Reader<'_> {
    fn byte(&mut self) -> u8 {
        match self.data.get(self.pos) {
            Some(&b) => {
                self.pos += 1;
                b
            }
            None => {
                self.eof = true;
                0
            }
        }
    }

    fn be32(&mut self) -> u32 {
        let mut v = 0;
        for _ in 0..4 {
            v = v << 8 | u32::from(self.byte());
        }
        v
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.byte()).collect()
    }

    fn check(&self) -> Result<()> {
        if self.eof {
            bail!("Failed to load CprState: Input/output error");
        }
        Ok(())
    }
}

/// `cpr_state_load()` once the stream is in: the entries in the order they came, each of
/// which took the next of the descriptors that came with the stream.
pub fn decode(data: &[u8], fds: usize) -> Result<Vec<FdEntry>> {
    let mut r = Reader { data, pos: 0, eof: false };
    let magic = r.be32();
    if magic != QEMU_CPR_FILE_MAGIC {
        bail!("Not a migration stream (bad magic {:x})", magic);
    }
    let version = r.be32();
    if version != QEMU_CPR_FILE_VERSION {
        bail!("Unsupported migration stream version {}", version as i32);
    }
    let mut out = Vec::new();
    while r.byte() != 0 {
        let len = r.be32() as usize;
        if len > data.len() {
            bail!("Failed to load CprState: Input/output error");
        }
        let mut name = r.bytes(len);
        if name.last() == Some(&0) {
            name.pop();
        }
        let id = r.be32() as i32;
        r.check()?;
        // qemu_file_get_fd()
        let service = r.byte();
        if service != b' ' {
            bail!("cpr-in unexpected service byte: {}({})", service, service as char);
        }
        if out.len() >= fds {
            bail!("cpr-in no FD come with service byte");
        }
        out.push(FdEntry { name: String::from_utf8_lossy(&name).into_owned(), id });
    }
    r.check()?;
    // vmstate_subsection_load(): subsections follow for as long as the next byte says so.
    while r.data.get(r.pos) == Some(&VMSTATE_SUBSECTION) {
        r.pos += 1;
        let len = usize::from(r.byte());
        let name = String::from_utf8_lossy(&r.bytes(len)).into_owned();
        let _version = r.be32();
        r.check()?;
        if name != VFIO_SUBSECTION {
            bail!("VM subsection '{}' in 'CprState' does not exist", name);
        }
        if r.byte() != 0 {
            bail!("CPR state of VFIO devices is not supported by ruvm");
        }
        r.check()?;
    }
    Ok(out)
}

/// The UNIX socket path of a `cpr` channel, `cpr_transfer_output()`.
fn output_path(addr: &MigrationAddr) -> Result<&str> {
    match &addr.u {
        MigrationAddressU::Socket(s) => match &s.u {
            SocketAddressU::Unix(u) => Ok(&u.path),
            _ => bail!("bad cpr channel address; must be unix"),
        },
        _ => bail!("bad cpr channel address; must be unix"),
    }
}

/// The UNIX socket path of a `cpr` channel, `cpr_transfer_input()`. QEMU also listens on a
/// socket passed as `fd`; ruvm cannot look one up that early.
fn input_path(addr: &MigrationAddr) -> Result<&str> {
    match &addr.u {
        MigrationAddressU::Socket(s) => match &s.u {
            SocketAddressU::Unix(u) => Ok(&u.path),
            SocketAddressU::Fd(_) => bail!("a cpr channel of type fd is not supported by ruvm yet"),
            _ => bail!("bad cpr channel socket type; must be unix"),
        },
        _ => bail!("bad cpr channel socket type; must be unix"),
    }
}

#[cfg(unix)]
mod imp {
    use std::io::{self, IoSlice, Read, Write};
    use std::mem::MaybeUninit;
    use std::os::fd::{AsFd, OwnedFd};
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::time::Duration;

    use rustix::net::{
        RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SendAncillaryBuffer,
        SendAncillaryMessage, SendFlags, recvmsg, sendmsg,
    };
    use ruvm_base::{Error, Result};
    use ruvm_mem::cpr::{self as fds, CprFd};

    use super::{FdEntry, Piece, decode, encode, input_path, output_path};
    use crate::channel::MigrationAddr;

    /// `cpr_state_file`: the socket of the CPR state, kept open until the other side may
    /// know that this one is done with it.
    static STATE_FILE: Mutex<Option<UnixStream>> = Mutex::new(None);

    fn state_file() -> MutexGuard<'static, Option<UnixStream>> {
        STATE_FILE.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn send_fd(sock: &UnixStream, fd: &OwnedFd) -> io::Result<()> {
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = SendAncillaryBuffer::new(&mut space);
        let fds = [fd.as_fd()];
        if !control.push(SendAncillaryMessage::ScmRights(&fds)) {
            return Err(io::ErrorKind::OutOfMemory.into());
        }
        let n = sendmsg(sock, &[IoSlice::new(b" ")], &mut control, SendFlags::empty())?;
        if n != 1 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        Ok(())
    }

    /// `cpr_state_save()` for `cpr-transfer`: connects to `addr`, sends every saved descriptor
    /// and shuts the socket down for writing. The socket stays open for [`wait_hup`].
    pub fn state_save(addr: &MigrationAddr) -> Result<()> {
        let path = output_path(addr)?;
        let sock = UnixStream::connect(path)
            .map_err(|e| Error::from_io(format!("Failed to connect to '{path}'"), e))?;
        let list =
            fds::saved_fds().map_err(|e| Error::from_io("Failed to save the CPR state", e))?;
        let entries: Vec<FdEntry> =
            list.iter().map(|e| FdEntry { name: e.name.clone(), id: e.id }).collect();
        let io_err = |e| Error::from_io("Failed to save the CPR state", e);
        for p in encode(&entries) {
            match p {
                Piece::Bytes(b) => (&sock).write_all(&b).map_err(io_err)?,
                Piece::Fd(i) => send_fd(&sock, &list[i].fd).map_err(io_err)?,
            }
        }
        // Close only the writing half, so that the other side closing shows as a hangup.
        let _ = sock.shutdown(std::net::Shutdown::Write);
        *state_file() = Some(sock);
        Ok(())
    }

    /// The `G_IO_HUP` watch of `cpr_transfer_add_hup_watch()`: waits until the destination
    /// closes the CPR socket. Gives false when `cancel` was set first.
    pub fn wait_hup(cancel: &AtomicBool) -> bool {
        let sock = match state_file().as_ref().map(UnixStream::try_clone) {
            Some(Ok(s)) => s,
            _ => return true,
        };
        let _ = sock.set_read_timeout(Some(Duration::from_millis(100)));
        let mut buf = [0u8; 64];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return false;
            }
            match (&sock).read(&mut buf) {
                Ok(0) => return true,
                Ok(_) => {}
                Err(e)
                    if matches!(e.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut) => {}
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(_) => return true,
            }
        }
    }

    fn recv_all(sock: &UnixStream) -> io::Result<(Vec<u8>, Vec<OwnedFd>)> {
        let mut data = Vec::new();
        let mut fds = Vec::new();
        let mut buf = [0u8; 4096];
        loop {
            let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(64))];
            let mut control = RecvAncillaryBuffer::new(&mut space);
            #[cfg(not(target_vendor = "apple"))]
            let flags = RecvFlags::CMSG_CLOEXEC;
            #[cfg(target_vendor = "apple")]
            let flags = RecvFlags::empty();
            let msg = match recvmsg(sock, &mut [io::IoSliceMut::new(&mut buf)], &mut control, flags)
            {
                Ok(m) => m,
                Err(rustix::io::Errno::INTR) => continue,
                Err(e) => return Err(e.into()),
            };
            for m in control.drain() {
                if let RecvAncillaryMessage::ScmRights(received) = m {
                    fds.extend(received);
                }
            }
            if msg.bytes == 0 {
                return Ok((data, fds));
            }
            data.extend_from_slice(&buf[..msg.bytes]);
        }
    }

    /// `cpr_state_load()` for `cpr-transfer`: listens on `addr`, takes the state of the one
    /// process that connects and keeps the descriptors for the RAM blocks to find. The socket
    /// stays open until [`state_close`].
    pub fn state_load(addr: &MigrationAddr) -> Result<()> {
        let path = input_path(addr)?;
        fds::set_incoming(true);
        let _ = std::fs::remove_file(path);
        let listener = UnixListener::bind(path)
            .map_err(|e| Error::from_io(format!("Failed to bind socket to {path}"), e))?;
        let (sock, _) = listener.accept().map_err(|e| Error::from_io("Failed to accept", e))?;
        drop(listener);
        let _ = std::fs::remove_file(path);
        let (data, received) =
            recv_all(&sock).map_err(|e| Error::from_io("Failed to load CprState", e))?;
        let entries = decode(&data, received.len())?;
        let list = entries
            .into_iter()
            .zip(received)
            .map(|(e, fd)| CprFd { name: e.name, id: e.id, fd })
            .collect();
        fds::load_fds(list);
        *state_file() = Some(sock);
        Ok(())
    }

    /// `cpr_state_close()`: on the destination this tells the source that it listens now.
    pub fn state_close() {
        state_file().take();
    }
}

#[cfg(not(unix))]
mod imp {
    use std::sync::atomic::AtomicBool;

    use ruvm_base::{Result, bail};

    use super::{input_path, output_path};
    use crate::channel::MigrationAddr;

    /// `cpr_state_save()`: there are no UNIX sockets to pass descriptors over.
    pub fn state_save(addr: &MigrationAddr) -> Result<()> {
        output_path(addr)?;
        bail!("cpr-transfer is not supported on this host");
    }

    pub fn wait_hup(_cancel: &AtomicBool) -> bool {
        true
    }

    /// `cpr_state_load()`.
    pub fn state_load(addr: &MigrationAddr) -> Result<()> {
        input_path(addr)?;
        bail!("cpr-transfer is not supported on this host");
    }

    pub fn state_close() {}
}

pub use imp::{state_close, state_load, state_save, wait_hup};

#[cfg(test)]
mod tests {
    use super::*;

    /// The state QEMU 11.1 sent for a q35 guest with its RAM in a memfd, without the
    /// descriptors.
    const QEMU_STATE: &str = concat!(
        "514350520000000101000000132f726f6d406574632f616370692f727364700000000000200100000015",
        "2f726f6d406574632f616370692f7461626c6573000000000020010000000870632e62696f7300000000",
        "002001000000162f726f6d406574632f7461626c652d6c6f61646572000000000020010000000770632e",
        "726f6d000000000020010000000770632e72616d00000000002000051543707253746174652f7666696f",
        "20646576696365730000000100",
    );

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
    }

    fn flatten(p: &[Piece]) -> Vec<u8> {
        p.iter()
            .flat_map(|p| match p {
                Piece::Bytes(b) => b.clone(),
                Piece::Fd(_) => vec![b' '],
            })
            .collect()
    }

    #[test]
    fn qemu_state_round_trips() {
        let data = hex(QEMU_STATE);
        let entries = decode(&data, 6).unwrap();
        let names: Vec<_> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "/rom@etc/acpi/rsdp",
                "/rom@etc/acpi/tables",
                "pc.bios",
                "/rom@etc/table-loader",
                "pc.rom",
                "pc.ram"
            ]
        );
        assert!(entries.iter().all(|e| e.id == 0));
        // What ruvm sends for the same list is what QEMU sent.
        assert_eq!(flatten(&encode(&entries)), data);
    }

    #[test]
    fn bad_states() {
        assert_eq!(decode(&[], 0).unwrap_err().message(), "Not a migration stream (bad magic 0)");
        let mut v = QEMU_CPR_FILE_MAGIC.to_be_bytes().to_vec();
        v.extend_from_slice(&2u32.to_be_bytes());
        assert_eq!(decode(&v, 0).unwrap_err().message(), "Unsupported migration stream version 2");
        let data = hex(QEMU_STATE);
        assert_eq!(decode(&data, 5).unwrap_err().message(), "cpr-in no FD come with service byte");
        let empty = flatten(&encode(&[]));
        assert_eq!(decode(&empty, 0).unwrap(), []);
        assert_eq!(
            decode(&empty[..empty.len() - 1], 0).unwrap_err().message(),
            "Failed to load CprState: Input/output error"
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn state_goes_over_a_socket() {
        use std::sync::atomic::AtomicBool;

        use ruvm_qapi::types::{MigrationAddressU, SocketAddress, UnixSocketAddress};

        let dir = std::env::temp_dir().join(format!("ruvm-cpr-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cpr.sock").to_string_lossy().into_owned();
        let addr = MigrationAddr {
            u: MigrationAddressU::Socket(SocketAddress {
                u: SocketAddressU::Unix(UnixSocketAddress {
                    path: path.clone(),
                    ..Default::default()
                }),
            }),
        };
        let file = std::fs::File::open("/dev/null").unwrap();
        ruvm_mem::cpr::save_fd("cpr-socket-test", 3, file.into());
        let a = addr.clone();
        let dest = std::thread::spawn(move || {
            state_load(&a).unwrap();
            assert!(ruvm_mem::cpr::find_fd("cpr-socket-test", 3).is_some());
            state_close();
        });
        // The destination may not listen yet.
        let mut tries = 0;
        while let Err(e) = state_save(&addr) {
            tries += 1;
            assert!(tries < 500, "{}", e.message());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(wait_hup(&AtomicBool::new(false)));
        dest.join().unwrap();
        state_close();
        ruvm_mem::cpr::delete_fd("cpr-socket-test", 3);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
