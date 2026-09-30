// SPDX-License-Identifier: GPL-2.0-or-later

//! `NBDClientConnection` from nbd/client-connection.c: connecting and negotiating on a thread
//! of its own, so a caller can give up waiting while the attempt goes on, and retrying with a
//! growing pause (1, 2, 4, 8 and then 16 seconds) once retries are enabled.

use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::SocketAddress;

use super::client::{NbdExportInfo, nbd_receive_negotiate};
use super::proto::{NbdMode, NbdStream};
use super::sock::socket_connect;

#[derive(Debug, Default)]
struct State {
    do_retry: bool,
    /// The thread is running.
    running: bool,
    /// Nobody waits for the thread any more; it should stop.
    detached: bool,
    /// The error of the last attempt.
    err: Option<Error>,
    /// A finished connection nobody has taken yet, with what negotiation found.
    result: Option<(NbdStream, NbdExportInfo)>,
    /// The socket of the attempt in progress, so that release can shut it down.
    sioc: Option<NbdStream>,
}

#[derive(Debug)]
struct Shared {
    saddr: SocketAddress,
    do_negotiation: bool,
    initial_info: NbdExportInfo,
    state: Mutex<State>,
    cond: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `NBDClientConnection`.
#[derive(Debug)]
pub struct NbdClientConnection {
    shared: Arc<Shared>,
}

/// `error_copy()`: [`Error`] is not `Clone`, but its message and hint are all that matters.
fn error_copy(e: &Error) -> Error {
    let mut c = Error::new(e.class(), e.message().to_string());
    if let Some(h) = e.hint_text() {
        c = c.hint(h.to_string());
    }
    c
}

/// `nbd_connect()`: connect, and negotiate when `info` is given.
fn nbd_connect(
    saddr: &SocketAddress,
    info: Option<&mut NbdExportInfo>,
    sioc_slot: &Shared,
) -> Result<NbdStream> {
    let mut s = socket_connect(saddr)?;
    if let Ok(c) = s.try_clone() {
        sioc_slot.lock().sioc = Some(c);
    }
    let Some(info) = info else {
        return Ok(s);
    };
    match nbd_receive_negotiate(&mut s, info) {
        Ok(()) => Ok(s),
        Err(e) => {
            s.shutdown();
            Err(e)
        }
    }
}

fn connect_thread(sh: &Arc<Shared>) {
    let mut timeout = 1u64;
    let max_timeout = 16u64;
    let mut st = sh.lock();
    while !st.detached {
        drop(st);
        let mut info = sh.initial_info.clone();
        let r = nbd_connect(&sh.saddr, if sh.do_negotiation { Some(&mut info) } else { None }, sh);
        // The caller does not care for the input fields.
        info.x_dirty_bitmap = None;
        info.name = String::new();
        st = sh.lock();
        st.sioc = None;
        match r {
            Ok(s) => {
                st.err = None;
                st.result = Some((s, info));
                break;
            }
            Err(e) => {
                st.err = Some(e);
                if st.do_retry && !st.detached {
                    // Waiting on the condition variable rather than sleeping lets release
                    // end the thread early.
                    let d = Duration::from_secs(timeout);
                    let (g, _) = sh
                        .cond
                        .wait_timeout_while(st, d, |s| !s.detached)
                        .unwrap_or_else(|e| e.into_inner());
                    st = g;
                    if timeout < max_timeout {
                        timeout *= 2;
                    }
                    continue;
                }
            }
        }
        break;
    }
    st.running = false;
    if st.detached {
        if let Some((s, _)) = st.result.take() {
            s.shutdown();
        }
    }
    sh.cond.notify_all();
}

impl NbdClientConnection {
    /// `nbd_client_connection_new()`. With `do_negotiation` the handshake runs too, asking for
    /// extended headers, block sizes and `base:allocation` (or `x_dirty_bitmap`).
    pub fn new(
        saddr: &SocketAddress,
        do_negotiation: bool,
        export_name: Option<&str>,
        x_dirty_bitmap: Option<&str>,
    ) -> NbdClientConnection {
        let initial_info = NbdExportInfo {
            request_sizes: true,
            mode: NbdMode::Extended,
            base_allocation: true,
            x_dirty_bitmap: x_dirty_bitmap.map(str::to_string),
            name: export_name.unwrap_or("").to_string(),
            ..NbdExportInfo::default()
        };
        NbdClientConnection {
            shared: Arc::new(Shared {
                saddr: saddr.clone(),
                do_negotiation,
                initial_info,
                state: Mutex::new(State::default()),
                cond: Condvar::new(),
            }),
        }
    }

    /// `nbd_client_connection_enable_retry()`.
    pub fn enable_retry(&self) {
        self.shared.lock().do_retry = true;
    }

    fn take_result(st: &mut State) -> Option<(NbdStream, NbdExportInfo)> {
        st.result.take()
    }

    /// `nbd_co_establish_connection()`: a connection, from an attempt that already finished,
    /// or from a new or running one. Without `blocking` this only starts an attempt. With a
    /// `deadline` the wait ends there, as when QEMU's open or reconnect timer cancels it; the
    /// attempt then goes on in the background.
    pub fn establish(
        &self,
        blocking: bool,
        deadline: Option<Instant>,
    ) -> Result<(NbdStream, NbdExportInfo)> {
        let sh = &self.shared;
        let mut st = sh.lock();
        if !st.running {
            if let Some(r) = Self::take_result(&mut st) {
                return Ok(r);
            }
            st.running = true;
            let sh2 = sh.clone();
            let spawned = thread::Builder::new()
                .name("nbd-connect".into())
                .spawn(move || connect_thread(&sh2));
            if let Err(e) = spawned {
                st.running = false;
                return Err(Error::from_io("Failed to start the NBD connect thread", e));
            }
        }
        if !blocking {
            return Err(match &st.err {
                Some(e) => error_copy(e),
                None => Error::generic("No connection at the moment"),
            });
        }
        loop {
            if !st.running {
                break;
            }
            match deadline {
                None => st = sh.cond.wait(st).unwrap_or_else(|e| e.into_inner()),
                Some(d) => {
                    let now = Instant::now();
                    if now >= d {
                        break;
                    }
                    st = sh.cond.wait_timeout(st, d - now).unwrap_or_else(|e| e.into_inner()).0;
                }
            }
        }
        if st.running {
            // Cancelled: leave the thread running for the next attempt.
            return Err(match &st.err {
                Some(e) => error_copy(e),
                None => Error::generic("Connection attempt cancelled by timeout"),
            });
        }
        if let Some(r) = Self::take_result(&mut st) {
            return Ok(r);
        }
        Err(match st.err.take() {
            Some(e) => e,
            None => Error::generic("No connection at the moment"),
        })
    }
}

impl Drop for NbdClientConnection {
    /// `nbd_client_connection_release()`.
    fn drop(&mut self) {
        let mut st = self.shared.lock();
        st.detached = true;
        if let Some(s) = &st.sioc {
            s.shutdown();
        }
        if let Some((s, _)) = st.result.take() {
            s.shutdown();
        }
        self.shared.cond.notify_all();
    }
}
