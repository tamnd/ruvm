// SPDX-License-Identifier: GPL-2.0-or-later

//! Network Block Device: the protocol from nbd/common.c, nbd/client.c and nbd/server.c, the
//! `nbd` block driver from block/nbd.c and nbd/client-connection.c, and NBD exports of block
//! backends from blockdev-nbd.c. docs/interop/nbd.txt lists what QEMU speaks, and so does this.
//!
//! - [`NbdClient`] connects to a server (TCP or a Unix socket), negotiates, and sends requests.
//!   It is what the `nbd` block driver runs on, and can be used on its own.
//! - [`NbdServer`] listens, and serves exports of [`crate::BlockBackend`]s to every client on a
//!   thread of its own. [`nbd_server_start`], [`nbd_server_add`] and [`nbd_server_remove`] are
//!   the QMP commands of the same names.
//! - [`nbd_receive_negotiate`] and [`nbd_receive_export_list`] are the client handshake by
//!   itself, generic over the channel, for tools like `qemu-nbd --list`.
//! - [`nbd_parse_filename`] and [`nbd_options_from_qdict`] turn `nbd://`, `nbd+unix://` and
//!   `nbd:` file names into options.
//!
//! The client asks for extended headers, falls back to structured replies and then to simple
//! ones, sets the `base:allocation` context (or `x-dirty-bitmap`), and understands oldstyle
//! servers. The server offers fixed newstyle only, like QEMU, with all the options, meta
//! contexts `base:allocation`, `qemu:allocation-depth` and `qemu:dirty-bitmap:<name>`, and all
//! commands including `NBD_CMD_WRITE_ZEROES` with `NO_HOLE` and `FAST_ZERO`,
//! `NBD_CMD_BLOCK_STATUS` and `NBD_CMD_CACHE`.
//!
//! Differences from QEMU:
//!
//! - There is no TLS. The client fails with `No TLS credentials with id '<id>'` when given
//!   `tls-creds`, as QEMU does for an id that names no object, and never sends
//!   `NBD_OPT_STARTTLS`. The server answers `NBD_OPT_STARTTLS` with `NBD_REP_ERR_POLICY` and
//!   "TLS not configured", which is QEMU's answer when it has no credentials, and fails to
//!   start with `No TLS credentials with id '<id>'` when given some. `tls-hostname` and
//!   `tls-authz` are accepted and ignored.
//! - The client has one request in flight at a time, always with cookie 1. QEMU's driver
//!   runs up to 16 in parallel on coroutines. The bytes on the wire for any one request are
//!   the same.
//! - The client splits reads, writes, zero writes and discards bigger than the limits it
//!   reports itself. In QEMU the generic block layer does that; for [`NbdClient`] used on its
//!   own there is nobody else to do it.
//! - The reconnect delay runs as a deadline on the connection attempt rather than as a timer
//!   that wakes waiting coroutines; the states and what requests see are the same.
//! - The `fd` address type is refused: a descriptor number cannot be adopted without unsafe
//!   code, and there is no monitor to look names up in. `vsock` addresses fail with
//!   `socket family AF_VSOCK unsupported`, as in a QEMU built without vsock.
//! - A Unix socket with an empty path listens in the temporary directory under a
//!   `qemu-socket-XXXXXX` name like QEMU's, picked without `mkstemp()`. On Windows the
//!   `keep-alive` options are accepted and not applied, since std cannot set them.
//! - Errors that QEMU only traces (why a request failed on the channel) are kept in
//!   [`NbdClient::last_error`] instead.
//! - The server takes dirty bitmaps from its creator ([`NbdExportExtras`]), since the block
//!   layer has none yet. `bitmaps` naming a bitmap nobody passed in fails with QEMU's
//!   message for a missing bitmap. Requests on one connection are handled one after the
//!   other, not by up to 16 coroutines.
//! - `NBD_CMD_CACHE` on the client is only sent by [`NbdClient::cache`]; QEMU's driver never
//!   sends it.
//! - URI parsing follows GLib's `g_uri_parse()` for the parts NBD uses, but does not remove
//!   `.` and `..` path segments.

mod client;
mod connection;
mod driver;
mod proto;
mod server;
mod sock;
mod uri;

pub use client::{NbdExportInfo, nbd_receive_export_list, nbd_receive_negotiate};
pub use connection::NbdClientConnection;
pub use driver::NbdClient;
pub(crate) use driver::{NBD, NBD_TCP, NBD_UNIX};
pub use proto::{
    NBD_CMD_BLOCK_STATUS, NBD_CMD_CACHE, NBD_CMD_DISC, NBD_CMD_FLAG_DF, NBD_CMD_FLAG_FAST_ZERO,
    NBD_CMD_FLAG_FUA, NBD_CMD_FLAG_NO_HOLE, NBD_CMD_FLAG_PAYLOAD_LEN, NBD_CMD_FLAG_REQ_ONE,
    NBD_CMD_FLUSH, NBD_CMD_READ, NBD_CMD_TRIM, NBD_CMD_WRITE, NBD_CMD_WRITE_ZEROES,
    NBD_DEFAULT_HANDSHAKE_MAX_SECS, NBD_DEFAULT_MAX_CONNECTIONS, NBD_DEFAULT_PORT,
    NBD_FLAG_BLOCK_STAT_PAYLOAD, NBD_FLAG_CAN_MULTI_CONN, NBD_FLAG_HAS_FLAGS, NBD_FLAG_READ_ONLY,
    NBD_FLAG_ROTATIONAL, NBD_FLAG_SEND_CACHE, NBD_FLAG_SEND_DF, NBD_FLAG_SEND_FAST_ZERO,
    NBD_FLAG_SEND_FLUSH, NBD_FLAG_SEND_FUA, NBD_FLAG_SEND_RESIZE, NBD_FLAG_SEND_TRIM,
    NBD_FLAG_SEND_WRITE_ZEROES, NBD_MAX_BUFFER_SIZE, NBD_MAX_STRING_SIZE, NBD_STATE_DIRTY,
    NBD_STATE_HOLE, NBD_STATE_ZERO, NbdMode, NbdRequest, NbdStream, nbd_cmd_lookup, nbd_err_lookup,
    nbd_errno_to_system_errno, nbd_info_lookup, nbd_opt_lookup, nbd_rep_lookup,
    nbd_reply_type_lookup, system_errno_to_nbd_errno,
};
pub use server::{
    NbdBlockStatus, NbdDirtyBitmap, NbdExportExtras, NbdServer, NbdStatusSource, block_export_add,
    nbd_server, nbd_server_add, nbd_server_is_running, nbd_server_remove, nbd_server_start,
    nbd_server_stop,
};
pub use uri::{inet_parse, nbd_options_from_qdict, nbd_parse_filename};
