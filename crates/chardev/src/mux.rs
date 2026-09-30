// SPDX-License-Identifier: GPL-2.0-or-later

//! The `mux` chardev, chardev/char-mux.c: up to four frontends on one backend chardev, such as
//! a serial port and the monitor on stdio. Input goes to the frontend that has the focus, and
//! the escape character, Ctrl-A unless `-echr` says otherwise, starts a command:
//!
//! - `h` or `?` prints the help,
//! - `x` exits,
//! - `s` saves disk data back (for `-snapshot`),
//! - `t` turns timestamps on the output on and off,
//! - `b` sends a break to the frontend with the focus,
//! - `c` gives the focus to the next frontend,
//! - the escape character itself sends one escape character.
//!
//! The mux is the frontend of its backend chardev. Each time the backend opens, as when a client
//! connects to a socket, every frontend of the mux gets a connection of its own, which ends when
//! the backend closes again.

use std::collections::VecDeque;
use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Instant;

use ruvm_base::{Error, Result};

use crate::conn::{Connection, POLL_INTERVAL, Port, Stream};
use crate::local::timed_out;
use crate::{Attachment, Chardev, ChrEvent, FeRef, Frontend};

/// `MAX_MUX`.
pub const MAX_MUX: usize = 4;
/// `MUX_BUFFER_SIZE`: how many bytes wait for a frontend that is not reading.
pub const MUX_BUFFER_SIZE: usize = 32;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A callback the mux commands run.
pub type Hook = Arc<dyn Fn() + Send + Sync>;

/// What the mux commands do outside the chardev layer, shared by every mux of a
/// [`crate::Chardevs`].
pub(crate) struct Hooks {
    /// `term_escape_char`, `-echr`.
    pub(crate) escape: AtomicI32,
    /// `qmp_quit()` for `C-a x`.
    pub(crate) quit: Mutex<Option<Hook>>,
    /// `blk_commit_all()` for `C-a s`.
    pub(crate) commit: Mutex<Option<Hook>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Hooks { escape: AtomicI32::new(0x01), quit: Mutex::new(None), commit: Mutex::new(None) }
    }
}

impl fmt::Debug for Hooks {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hooks").field("escape", &self.escape).finish_non_exhaustive()
    }
}

const HELP: &[&str] = &[
    "% h    print this help\n\r",
    "% x    exit emulator\n\r",
    "% s    save disk data back to file (if -snapshot)\n\r",
    "% t    toggle console timestamps\n\r",
    "% b    send break (magic sysrq)\n\r",
    "% c    switch between console and monitor\n\r",
    "% %  sends %\n\r",
];

/// `mux_print_help()`: the text `C-a h` prints for escape character `escape`.
pub fn help_text(escape: i32) -> String {
    let (cbuf, ebuf) = if escape > 0 && escape < 26 {
        ("\n\r".to_string(), format!("C-{}", char::from(b'a' + (escape - 1) as u8)))
    } else {
        (
            format!("\n\rEscape-Char set to Ascii: 0x{:02x}\n\r\n\r", escape),
            "Escape-Char".to_string(),
        )
    };
    let mut s = cbuf;
    for line in HELP {
        s.push_str(&line.replace('%', &ebuf));
    }
    s
}

/// `[hh:mm:ss.mmm] `, the time since timestamps were turned on.
fn timestamp(ms: u128) -> String {
    let secs = ms / 1000;
    format!("[{:02}:{:02}:{:02}.{:03}] ", secs / 3600, (secs / 60) % 60, secs % 60, ms % 1000)
}

#[derive(Debug, Default)]
struct State {
    fes: [Option<FeRef>; MAX_MUX],
    /// `buffer`, `prod` and `cons`: what each frontend has not read yet.
    buf: [VecDeque<u8>; MAX_MUX],
    /// Whether the frontend is in a connection and so reading its buffer.
    serving: [bool; MAX_MUX],
    focus: Option<usize>,
    got_escape: bool,
    /// Whether the backend is open, and which time this is.
    open: bool,
    session: u64,
}

#[derive(Debug)]
struct Stamps {
    on: bool,
    start: Option<Instant>,
    linestart: bool,
}

/// `MuxChardev`.
#[derive(Debug)]
pub(crate) struct Mux {
    pub(crate) base: Arc<Chardev>,
    base_fe: Mutex<Option<Attachment>>,
    st: Mutex<State>,
    changed: Condvar,
    stamps: Mutex<Stamps>,
    hooks: Arc<Hooks>,
}

/// What a byte from the backend asks for, done once the state lock is let go.
enum Action {
    None,
    Help,
    Quit,
    Commit,
    Break,
    Focus(usize),
}

impl Mux {
    pub(crate) fn new(base: Arc<Chardev>, hooks: Arc<Hooks>) -> Mux {
        Mux {
            base,
            base_fe: Mutex::new(None),
            st: Mutex::new(State::default()),
            changed: Condvar::new(),
            stamps: Mutex::new(Stamps { on: false, start: None, linestart: false }),
            hooks,
        }
    }

    /// `mux_chr_open()`, once the chardev around the mux exists: the mux becomes the frontend
    /// of its backend.
    pub(crate) fn start(chr: &Arc<Chardev>) -> Result<()> {
        let m = chr.mux().expect("a mux");
        let fe = Arc::new(MuxFrontend(Arc::downgrade(chr)));
        let a = m.base.attach(fe)?;
        *lock(&m.base_fe) = Some(a);
        Ok(())
    }

    /// Lets go of the backend, `char_mux_finalize()`.
    pub(crate) fn close(&self) {
        drop(lock(&self.base_fe).take());
    }

    /// `qemu_chr_is_busy()` of a mux: whether any frontend is attached.
    pub(crate) fn is_busy(&self) -> bool {
        lock(&self.st).fes.iter().any(Option::is_some)
    }

    /// Whether the frontend with the focus is there, `chr->fe`.
    pub(crate) fn focus_attached(&self) -> bool {
        let st = lock(&self.st);
        st.focus.is_some_and(|f| st.fes[f].is_some())
    }

    fn send_event(fe: &Option<FeRef>, ev: ChrEvent) {
        if let Some(fe) = fe {
            fe.0.event(ev);
        }
    }

    /// `mux_chr_be_event()`: an event of the mux goes to the frontend with the focus.
    pub(crate) fn be_event(&self, ev: ChrEvent) {
        let fe = {
            let st = lock(&self.st);
            st.focus.and_then(|f| st.fes[f].clone())
        };
        Self::send_event(&fe, ev);
    }

    /// `mux_chr_send_all_event()`: an event of the backend goes to every frontend.
    fn send_all_event(&self, ev: ChrEvent) {
        let fes: Vec<FeRef> = lock(&self.st).fes.iter().flatten().cloned().collect();
        for fe in fes {
            fe.0.event(ev);
        }
    }

    /// `mux_set_focus()`.
    pub(crate) fn set_focus(&self, focus: usize) {
        let (old, new) = {
            let mut st = lock(&self.st);
            let old = st.focus.and_then(|f| st.fes[f].clone());
            st.focus = Some(focus);
            (old, st.fes[focus].clone())
        };
        self.changed.notify_all();
        Self::send_event(&old, ChrEvent::MuxOut);
        Self::send_event(&new, ChrEvent::MuxIn);
    }

    /// `mux_chr_attach_frontend()` and the focus a frontend takes when it sets its handlers.
    pub(crate) fn attach(chr: &Arc<Chardev>, fe: Arc<dyn Frontend>) -> Result<Attachment> {
        let m = chr.mux().expect("a mux");
        let tag = {
            let mut st = lock(&m.st);
            let Some(tag) = st.fes.iter().position(Option::is_none) else {
                return Err(Error::generic(format!(
                    "too many uses of multiplexed chardev '{}' (maximum is {MAX_MUX})",
                    chr.label()
                )));
            };
            st.fes[tag] = Some(FeRef(fe.clone()));
            tag
        };
        m.set_focus(tag);

        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let chr = chr.clone();
            let stop = stop.clone();
            std::thread::Builder::new().name(format!("chardev-{}-{tag}", chr.label())).spawn(
                move || {
                    chr.gate.wait();
                    let m = chr.mux().expect("a mux");
                    let mut last = 0;
                    while let Some(session) = m.wait_open(last, &stop) {
                        last = session;
                        let port = Port { tag, session };
                        let Ok(mut conn) =
                            Connection::new(Stream::Local(chr.clone(), port), stop.clone())
                        else {
                            break;
                        };
                        lock(&m.st).serving[tag] = true;
                        let _ = fe.serve(&mut conn);
                        lock(&m.st).serving[tag] = false;
                    }
                },
            )
        };
        match thread {
            Ok(t) => Ok(Attachment { chr: chr.clone(), stop, thread: Some(t), tag: Some(tag) }),
            Err(e) => {
                m.detach(tag);
                Err(Error::from_io("Failed to start the chardev thread", e))
            }
        }
    }

    /// `mux_chr_detach_frontend()`.
    pub(crate) fn detach(&self, tag: usize) {
        let mut st = lock(&self.st);
        st.fes[tag] = None;
        st.buf[tag].clear();
        drop(st);
        self.changed.notify_all();
    }

    /// Waits for the backend to open after session `last`. `None` when `stop` is set first.
    fn wait_open(&self, last: u64, stop: &AtomicBool) -> Option<u64> {
        let mut st = lock(&self.st);
        loop {
            if stop.load(Ordering::Acquire) {
                return None;
            }
            if st.open && st.session != last {
                return Some(st.session);
            }
            st = self.changed.wait_timeout(st, POLL_INTERVAL).unwrap_or_else(|e| e.into_inner()).0;
        }
    }

    /// What the frontend on `port` gets to read, `mux_chr_accept_input()`: only the frontend
    /// with the focus reads, and its connection ends when the backend closes.
    pub(crate) fn read(&self, port: Port, buf: &mut [u8]) -> io::Result<usize> {
        let mut st = lock(&self.st);
        for _ in 0..2 {
            if !st.open || st.session != port.session {
                return Ok(0);
            }
            let focused = st.focus == Some(port.tag);
            let q = &mut st.buf[port.tag];
            if focused && !q.is_empty() {
                let n = q.len().min(buf.len());
                for (d, s) in buf.iter_mut().zip(q.drain(..n)) {
                    *d = s;
                }
                return Ok(n);
            }
            st = self.changed.wait_timeout(st, POLL_INTERVAL).unwrap_or_else(|e| e.into_inner()).0;
        }
        Err(timed_out())
    }

    /// `qemu_chr_be_write()` on the mux itself: straight to the frontend with the focus.
    pub(crate) fn be_write(&self, buf: &[u8]) {
        let mut st = lock(&self.st);
        if let Some(f) = st.focus {
            Self::queue(&mut st, f, buf);
        }
        drop(st);
        self.changed.notify_all();
    }

    /// Puts bytes in the buffer of frontend `m`. A frontend that is not reading gets at most
    /// [`MUX_BUFFER_SIZE`] of them, and bytes for a slot without a frontend are dropped.
    fn queue(st: &mut State, m: usize, buf: &[u8]) {
        if st.fes[m].is_none() {
            return;
        }
        for &b in buf {
            if !st.serving[m] && st.buf[m].len() >= MUX_BUFFER_SIZE {
                break;
            }
            st.buf[m].push_back(b);
        }
    }

    /// `mux_proc_byte()` for one byte from the backend.
    fn proc_byte(&self, ch: u8) -> Action {
        let escape = self.hooks.escape.load(Ordering::Relaxed);
        let mut st = lock(&self.st);
        if st.got_escape {
            st.got_escape = false;
            if i32::from(ch) != escape {
                return match ch {
                    b'?' | b'h' => Action::Help,
                    b'x' => Action::Quit,
                    b's' => Action::Commit,
                    b'b' => Action::Break,
                    b'c' => {
                        let used: Vec<usize> =
                            (0..MAX_MUX).filter(|&i| st.fes[i].is_some()).collect();
                        let from = st.focus.map_or(0, |f| f + 1);
                        match used.iter().find(|&&i| i >= from).or(used.first()) {
                            Some(&next) => Action::Focus(next),
                            None => Action::None,
                        }
                    }
                    b't' => {
                        let mut t = lock(&self.stamps);
                        t.on = !t.on;
                        t.start = None;
                        t.linestart = false;
                        Action::None
                    }
                    _ => Action::None,
                };
            }
        } else if i32::from(ch) == escape {
            st.got_escape = true;
            return Action::None;
        }
        if let Some(f) = st.focus {
            Self::queue(&mut st, f, &[ch]);
        }
        drop(st);
        self.changed.notify_all();
        Action::None
    }

    /// `mux_chr_read()`: bytes the backend read.
    fn receive(&self, chr: &Chardev, buf: &[u8]) {
        for &ch in buf {
            match self.proc_byte(ch) {
                Action::None => {}
                Action::Help => {
                    let _ = chr
                        .write_all(help_text(self.hooks.escape.load(Ordering::Relaxed)).as_bytes());
                }
                Action::Quit => {
                    let _ = chr.write_all(b"QEMU: Terminated\n\r");
                    let hook = lock(&self.hooks.quit).clone();
                    match hook {
                        Some(h) => h(),
                        None => {
                            crate::stdio::term_exit();
                            std::process::exit(0);
                        }
                    }
                }
                Action::Commit => {
                    let hook = lock(&self.hooks.commit).clone();
                    if let Some(h) = hook {
                        h();
                    }
                }
                Action::Break => self.be_event(ChrEvent::Break),
                Action::Focus(next) => self.set_focus(next),
            }
        }
    }

    /// `mux_chr_write()`: output of any frontend, with timestamps when they are on.
    pub(crate) fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let mut t = lock(&self.stamps);
        if !t.on {
            drop(t);
            return self.base.write(buf);
        }
        let mut out = Vec::with_capacity(buf.len() + 16);
        for &b in buf {
            if t.linestart {
                let now = Instant::now();
                let start = *t.start.get_or_insert(now);
                out.extend(timestamp(now.duration_since(start).as_millis()).bytes());
                t.linestart = false;
            }
            out.push(b);
            if b == b'\n' {
                t.linestart = true;
            }
        }
        drop(t);
        self.base.write_all(&out)?;
        Ok(buf.len())
    }

    fn set_open(&self, open: bool) {
        let mut st = lock(&self.st);
        st.open = open;
        if open {
            st.session += 1;
        }
        drop(st);
        self.changed.notify_all();
    }
}

/// The mux as the frontend of its backend, `mux_chr_can_read()`, `mux_chr_read()` and
/// `mux_chr_event()`.
struct MuxFrontend(Weak<Chardev>);

impl Frontend for MuxFrontend {
    fn serve(&self, conn: &mut Connection) -> io::Result<()> {
        let Some(chr) = self.0.upgrade() else { return Ok(()) };
        let m = chr.mux().expect("a mux");
        m.set_open(true);
        let mut buf = [0u8; 4096];
        let r = loop {
            match conn.recv(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => m.receive(&chr, &buf[..n]),
                Err(e) => break Err(e),
            }
        };
        m.set_open(false);
        r
    }

    fn event(&self, event: ChrEvent) {
        if let Some(chr) = self.0.upgrade() {
            chr.mux().expect("a mux").send_all_event(event);
        }
    }
}
