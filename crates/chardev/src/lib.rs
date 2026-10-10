// SPDX-License-Identifier: GPL-2.0-or-later

//! Character device backends and the mux, chardev/.
//!
//! [`Chardevs`] is the `/chardevs` container: every chardev by id, what `chardev-add`,
//! `chardev-remove` and `query-chardev` work on. A frontend such as a monitor attaches to one
//! chardev with [`Chardev::attach`] and then gets each client connection on a thread of its
//! own, until it detaches. A chardev takes one frontend at a time, except for a mux, which takes
//! up to four.
//!
//! The backends are `null`, `socket` on Unix and TCP sockets and on a socket passed in by
//! number, `file`, `pipe`, `stdio`, `pty` on Unix, `ringbuf` (also called `memory`) and `mux`.
//! For a backend that is not a socket the frontend gets one connection for as long as the
//! backend is open. What a frontend writes goes through [`Attachment::write_all`] or the
//! connection's writer. [`opts`] turns `-chardev` and the old compat strings into backends.

#![deny(unsafe_code)]

use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{QemuOpts, QemuOptsList, is_help_option};
use ruvm_qapi::types::{
    ChardevBackend, ChardevBackendU, ChardevInfo, ChardevMux, ChardevMuxWrapper, ChardevReturn,
    DataFormat,
};

pub mod conn;
#[cfg(unix)]
mod fd;
mod file;
mod local;
pub mod mux;
pub mod opts;
#[cfg(unix)]
pub mod pty;
pub mod qom;
mod ringbuf;
pub mod socket;
pub mod stdio;

pub use conn::Connection;
pub use mux::Hook;
pub use socket::SocketChardev;

use conn::{Port, Stream};
use local::{ByteQueue, Local};
use mux::Mux;
use ringbuf::Ringbuf;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The events a frontend hears of besides its connections, `QEMUChrEvent`. The opening and
/// closing of the backend are the start and end of [`Frontend::serve`]; a frontend of a mux
/// also hears of them here when they happen on the mux's backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChrEvent {
    Opened,
    Closed,
    /// A serial break, from `C-a b` on a mux or `chardev-send-break`.
    Break,
    /// The frontend got the focus of its mux.
    MuxIn,
    /// The frontend lost the focus of its mux.
    MuxOut,
}

/// The frontend side, what `qemu_chr_fe_set_handlers()` installs. `serve` runs on the
/// chardev's thread for each client from `CHR_EVENT_OPENED` to `CHR_EVENT_CLOSED` and returns
/// when [`Connection::recv`] gives 0.
pub trait Frontend: Send + Sync {
    fn serve(&self, conn: &mut Connection) -> io::Result<()>;

    /// The other events. This runs on whatever thread raised the event, so it should not wait
    /// for long.
    fn event(&self, event: ChrEvent) {
        let _ = event;
    }
}

#[derive(Clone)]
struct FeRef(Arc<dyn Frontend>);

impl fmt::Debug for FeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Frontend")
    }
}

/// Holds frontends back until the main loop runs. QEMU only hands chardev input to a
/// frontend from its main loop, so nothing a client sends is seen before startup is over.
#[derive(Debug, Default)]
struct Gate {
    held: Mutex<bool>,
    released: Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut held = lock(&self.held);
        while *held {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn set(&self, hold: bool) {
        *lock(&self.held) = hold;
        if !hold {
            self.released.notify_all();
        }
    }
}

#[derive(Debug)]
enum Backend {
    Null,
    Socket(SocketChardev),
    /// `file`, `pipe` and `stdio`, the last with its claim on the terminal.
    Local(Local, Option<stdio::Claim>),
    #[cfg(unix)]
    Pty(pty::Pty),
    Ringbuf(Ringbuf),
    Mux(Box<Mux>),
}

/// The backend names `query-chardev-backends` lists and `-chardev help` prints.
pub const BACKENDS: &[&str] = &[
    "file",
    "memory",
    "mux",
    "null",
    #[cfg(unix)]
    "pipe",
    #[cfg(unix)]
    "pty",
    "ringbuf",
    "socket",
    "stdio",
];

/// One chardev, `Chardev`.
#[derive(Debug)]
pub struct Chardev {
    label: String,
    backend: Backend,
    /// Set while a frontend is attached, `chr->fe`.
    busy: AtomicBool,
    /// The attached frontend, for events.
    fe: Mutex<Option<FeRef>>,
    /// Held by the thread serving a frontend. A frontend attached right after another one
    /// detached waits here until the old thread has let go of the connection.
    serving: Arc<Mutex<()>>,
    gate: Arc<Gate>,
    /// Input handed in with [`Chardev::be_write`].
    input: ByteQueue,
    /// `chr_write_lock`.
    write_lock: Mutex<()>,
}

impl Chardev {
    /// `chardev_new()` and the backend's open. Sockets connect, or listen and maybe wait for a
    /// client, before this returns. A mux needs its backend to be in a [`Chardevs`], so it is
    /// made with [`Chardevs::add`].
    pub fn open(label: &str, backend: &ChardevBackend) -> Result<Arc<Chardev>> {
        Self::open_in(label, backend, Arc::default(), None, false)
    }

    fn open_in(
        label: &str,
        backend: &ChardevBackend,
        gate: Arc<Gate>,
        chardevs: Option<&Chardevs>,
        announce: bool,
    ) -> Result<Arc<Chardev>> {
        let backend = match &backend.u {
            ChardevBackendU::Null(_) => Backend::Null,
            ChardevBackendU::Socket(s) => Backend::Socket(SocketChardev::open(&s.data)?),
            ChardevBackendU::File(f) => Backend::Local(file::open_file(&f.data)?, None),
            ChardevBackendU::Pipe(p) => Backend::Local(file::open_pipe(&p.data)?, None),
            ChardevBackendU::Stdio(s) => {
                let (local, claim) = stdio::open(&s.data)?;
                Backend::Local(local, Some(claim))
            }
            #[cfg(unix)]
            ChardevBackendU::Pty(p) => Backend::Pty(pty::Pty::open(label, &p.data, announce)?),
            ChardevBackendU::Ringbuf(r) => Backend::Ringbuf(Ringbuf::open(&r.data, false)?),
            ChardevBackendU::Memory(r) => Backend::Ringbuf(Ringbuf::open(&r.data, true)?),
            ChardevBackendU::Mux(m) => {
                let name = &m.data.chardev;
                let base = chardevs
                    .and_then(|c| c.find(name))
                    .ok_or_else(|| Error::generic(format!("mux: base chardev {name} not found")))?;
                let hooks = chardevs.map(|c| c.hooks.clone()).unwrap_or_default();
                Backend::Mux(Box::new(Mux::new(base, hooks)))
            }
            _ => {
                let _ = announce;
                let kind = backend.u.tag().as_str();
                return Err(Error::generic(format!(
                    "chardev backend '{kind}' is not supported by ruvm yet"
                )));
            }
        };
        let chr = Arc::new(Chardev {
            label: label.to_string(),
            backend,
            busy: AtomicBool::new(false),
            fe: Mutex::new(None),
            serving: Arc::new(Mutex::new(())),
            gate,
            input: ByteQueue::default(),
            write_lock: Mutex::new(()),
        });
        if chr.mux().is_some() {
            Mux::start(&chr)?;
        }
        Ok(chr)
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// The QOM type name, `chardev-` and the backend.
    pub fn typename(&self) -> &'static str {
        match &self.backend {
            Backend::Null => qom::TYPE_CHARDEV_NULL,
            Backend::Socket(_) => qom::TYPE_CHARDEV_SOCKET,
            Backend::Local(l, _) => match l.kind {
                local::LocalKind::File => qom::TYPE_CHARDEV_FILE,
                local::LocalKind::Pipe => qom::TYPE_CHARDEV_PIPE,
                local::LocalKind::Stdio => qom::TYPE_CHARDEV_STDIO,
            },
            #[cfg(unix)]
            Backend::Pty(_) => qom::TYPE_CHARDEV_PTY,
            Backend::Ringbuf(r) if r.memory => qom::TYPE_CHARDEV_MEMORY,
            Backend::Ringbuf(_) => qom::TYPE_CHARDEV_RINGBUF,
            Backend::Mux(_) => qom::TYPE_CHARDEV_MUX,
        }
    }

    /// `qemu_chr_get_filename()`: the backend's own text, or the type name after `chardev-`.
    pub fn filename(&self) -> String {
        match &self.backend {
            Backend::Socket(s) => s.filename(),
            #[cfg(unix)]
            Backend::Pty(p) => format!("pty:{}", p.name()),
            _ => self.typename()["chardev-".len()..].to_string(),
        }
    }

    /// `qemu_chr_get_pty_name()`: the slave a `pty` chardev made.
    pub fn pty_name(&self) -> Option<String> {
        match &self.backend {
            #[cfg(unix)]
            Backend::Pty(p) => Some(p.name().to_string()),
            _ => None,
        }
    }

    /// What `chardev-add` returns for this chardev.
    pub fn chardev_return(&self) -> ChardevReturn {
        ChardevReturn { pty: self.pty_name() }
    }

    /// `qemu_chr_is_busy()`.
    pub fn is_busy(&self) -> bool {
        match &self.backend {
            Backend::Mux(m) => m.is_busy(),
            _ => self.busy.load(Ordering::Acquire),
        }
    }

    /// Whether a frontend is there to take input, what `query-chardev` calls `frontend-open`.
    pub fn frontend_open(&self) -> bool {
        match &self.backend {
            Backend::Mux(m) => m.focus_attached(),
            _ => self.busy.load(Ordering::Acquire),
        }
    }

    /// The socket backend, when this is a socket chardev.
    pub fn socket(&self) -> Option<&SocketChardev> {
        match &self.backend {
            Backend::Socket(s) => Some(s),
            _ => None,
        }
    }

    fn mux(&self) -> Option<&Mux> {
        match &self.backend {
            Backend::Mux(m) => Some(m),
            _ => None,
        }
    }

    fn ringbuf(&self) -> Option<&Ringbuf> {
        match &self.backend {
            Backend::Ringbuf(r) => Some(r),
            _ => None,
        }
    }

    /// Whether this is a mux, `CHARDEV_IS_MUX()`.
    pub fn is_mux(&self) -> bool {
        self.mux().is_some()
    }

    /// The chardev a mux sits on.
    pub fn mux_base(&self) -> Option<&Arc<Chardev>> {
        self.mux().map(|m| &m.base)
    }

    /// `qemu_chr_write()` without `write_all`: one write to the backend, which may take only
    /// part of `buf`. Output nobody can see is dropped and counts as written.
    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let _g = lock(&self.write_lock);
        match &self.backend {
            Backend::Null => Ok(buf.len()),
            Backend::Socket(s) => s.write(buf),
            Backend::Local(l, _) => l.write(buf),
            #[cfg(unix)]
            Backend::Pty(p) => p.write(buf),
            Backend::Ringbuf(r) => Ok(r.write(buf)),
            Backend::Mux(m) => m.write(buf),
        }
    }

    /// `qemu_chr_write_all()`: writes until everything is out, waiting while the backend is
    /// full. Gives how much was written, which is less than `buf.len()` only when the backend
    /// stopped taking bytes.
    pub fn write_all(&self, buf: &[u8]) -> io::Result<usize> {
        let mut off = 0;
        while off < buf.len() {
            match self.write(&buf[off..]) {
                Ok(0) => break,
                Ok(n) => off += n,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_micros(100));
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if off == 0 => return Err(e),
                Err(_) => break,
            }
        }
        Ok(off)
    }

    /// `qemu_chr_be_write()`: hands `buf` to the frontend as if the backend had read it. A mux
    /// gives it to the frontend with the focus, without looking for escapes. Sockets read only
    /// from their peer and ignore this.
    pub fn be_write(&self, buf: &[u8]) {
        match &self.backend {
            Backend::Mux(m) => m.be_write(buf),
            _ => self.input.push(buf),
        }
    }

    /// `qemu_chr_be_event()`. On a mux the event goes to the frontend with the focus, on the
    /// backend of a mux to every frontend of the mux.
    pub fn be_event(&self, event: ChrEvent) {
        match &self.backend {
            Backend::Mux(m) => m.be_event(event),
            _ => {
                let fe = lock(&self.fe).clone();
                if let Some(fe) = fe {
                    fe.0.event(event);
                }
            }
        }
    }

    /// `qemu_chr_fe_set_echo()`: only `stdio` does anything with it.
    pub fn set_echo(&self, echo: bool) {
        match &self.backend {
            Backend::Local(l, Some(_)) if l.kind == local::LocalKind::Stdio => {
                stdio::set_echo(echo)
            }
            Backend::Mux(m) => m.base.set_echo(echo),
            _ => {}
        }
    }

    /// Whether a `pty` chardev's slave is open.
    pub fn pty_connected(&self) -> bool {
        match &self.backend {
            #[cfg(unix)]
            Backend::Pty(p) => p.is_connected(),
            _ => false,
        }
    }

    /// What a local connection reads, see [`Stream::Local`].
    pub(crate) fn read_input(&self, port: Port, buf: &mut [u8]) -> io::Result<usize> {
        if let Backend::Mux(m) = &self.backend {
            return m.read(port, buf);
        }
        if let Some(n) = self.input.pop(buf) {
            return Ok(n);
        }
        match &self.backend {
            Backend::Local(l, _) => l.read(&self.input, buf),
            #[cfg(unix)]
            Backend::Pty(p) => p.read(buf),
            _ => self.input.read(buf),
        }
    }

    /// Lets go of what the chardev holds outside itself once it is removed: the base of a
    /// mux, the symlink of a pty and the terminal of stdio.
    fn close(&self) {
        match &self.backend {
            Backend::Mux(m) => m.close(),
            #[cfg(unix)]
            Backend::Pty(p) => p.close(),
            Backend::Local(_, Some(_)) => stdio::term_exit(),
            _ => {}
        }
    }

    /// Serves `fe` on a backend that is not a socket: one connection for as long as the
    /// backend is open, and for a pty one each time the slave is opened.
    fn run_local(self: &Arc<Self>, stop: &Arc<AtomicBool>, fe: &Arc<dyn Frontend>) {
        let conn = || Connection::new(Stream::Local(self.clone(), Port::default()), stop.clone());
        #[cfg(unix)]
        if let Backend::Pty(p) = &self.backend {
            while !stop.load(Ordering::Acquire) {
                if !p.is_connected() && !p.poll_connected() {
                    std::thread::sleep(conn::POLL_INTERVAL);
                    continue;
                }
                let Ok(mut c) = conn() else { return };
                let _ = fe.serve(&mut c);
            }
            return;
        }
        if let Ok(mut c) = conn() {
            let _ = fe.serve(&mut c);
        }
    }

    /// `qemu_chr_fe_init()` and `qemu_chr_fe_set_handlers()`. Fails when another frontend has
    /// the chardev, or a mux has four already. A frontend of a mux takes the focus.
    pub fn attach(self: &Arc<Self>, fe: Arc<dyn Frontend>) -> Result<Attachment> {
        if self.mux().is_some() {
            return Mux::attach(self, fe);
        }
        if self.busy.swap(true, Ordering::AcqRel) {
            return Err(Error::generic(format!("chardev '{}' is already in use", self.label)));
        }
        *lock(&self.fe) = Some(FeRef(fe.clone()));
        let stop = Arc::new(AtomicBool::new(false));
        let thread = if matches!(self.backend, Backend::Null) {
            None
        } else {
            let chr = self.clone();
            let stop = stop.clone();
            let name = format!("chardev-{}", self.label);
            let spawned = std::thread::Builder::new().name(name).spawn(move || {
                chr.gate.wait();
                let serving = chr.serving.clone();
                let _g = lock(&serving);
                match &chr.backend {
                    Backend::Socket(s) => s.run(&stop, |conn| fe.serve(conn)),
                    _ => chr.run_local(&stop, &fe),
                }
            });
            match spawned {
                Ok(t) => Some(t),
                Err(e) => {
                    *lock(&self.fe) = None;
                    self.busy.store(false, Ordering::Release);
                    return Err(Error::from_io("Failed to start the chardev thread", e));
                }
            }
        };
        Ok(Attachment { chr: self.clone(), stop, thread, tag: None })
    }
}

/// A frontend attached to a chardev. Dropping it detaches the frontend, as
/// `qemu_chr_fe_deinit()` does, and leaves any connection open for the next frontend.
#[derive(Debug)]
pub struct Attachment {
    chr: Arc<Chardev>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// The frontend's slot on a mux.
    tag: Option<usize>,
}

impl Attachment {
    pub fn chardev(&self) -> &Arc<Chardev> {
        &self.chr
    }

    /// `qemu_chr_fe_write()`.
    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        self.chr.write(buf)
    }

    /// `qemu_chr_fe_write_all()`.
    pub fn write_all(&self, buf: &[u8]) -> io::Result<usize> {
        self.chr.write_all(buf)
    }

    /// `qemu_chr_fe_take_focus()`: on a mux, input goes to this frontend from now on.
    pub fn take_focus(&self) {
        if let (Some(m), Some(tag)) = (self.chr.mux(), self.tag) {
            m.set_focus(tag);
        }
    }

    /// `qemu_chr_fe_set_echo()`.
    pub fn set_echo(&self, echo: bool) {
        self.chr.set_echo(echo);
    }

    /// Detaches and waits for the frontend's thread to end. Only call this from a thread that
    /// the frontend does not wait on.
    pub fn join(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for Attachment {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        match (self.chr.mux(), self.tag) {
            (Some(m), Some(tag)) => m.detach(tag),
            _ => {
                *lock(&self.chr.fe) = None;
                self.chr.busy.store(false, Ordering::Release);
            }
        }
    }
}

/// Every chardev, the `/chardevs` container.
#[derive(Debug, Default)]
pub struct Chardevs {
    list: Mutex<Vec<Arc<Chardev>>>,
    /// Where the objects for each chardev go, once there is a QOM tree.
    registry: OnceLock<ruvm_qom::Registry>,
    gate: Arc<Gate>,
    hooks: Arc<mux::Hooks>,
}

impl Chardevs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps frontends of every chardev from seeing clients until [`Chardevs::release`], as
    /// `suspend_mux_open()` and the main loop not running yet do in QEMU.
    pub fn hold(&self) {
        self.gate.set(true);
    }

    /// Lets frontends serve their clients, once the main loop runs.
    pub fn release(&self) {
        self.gate.set(false);
    }

    /// Puts an object for each chardev under `/chardevs` of `registry`, from now on. The
    /// chardev types have to be registered there.
    pub fn set_registry(&self, registry: &ruvm_qom::Registry) {
        let _ = self.registry.set(registry.clone());
    }

    /// What `C-a x` on a mux runs after printing `QEMU: Terminated`, `qmp_quit()`. Without
    /// one the terminal is put back and the process exits.
    pub fn set_mux_quit_handler(&self, hook: Hook) {
        *lock(&self.hooks.quit) = Some(hook);
    }

    /// What `C-a s` on a mux runs, `blk_commit_all()`.
    pub fn set_mux_commit_handler(&self, hook: Hook) {
        *lock(&self.hooks.commit) = Some(hook);
    }

    /// `term_escape_char`, which `-echr` sets.
    pub fn set_escape_char(&self, ch: i32) {
        self.hooks.escape.store(ch, Ordering::Relaxed);
    }

    pub fn escape_char(&self) -> i32 {
        self.hooks.escape.load(Ordering::Relaxed)
    }

    /// `qemu_chr_find()`.
    pub fn find(&self, id: &str) -> Option<Arc<Chardev>> {
        lock(&self.list).iter().find(|c| c.label == id).cloned()
    }

    /// `qmp_chardev_add()` without the QMP reply, for which see [`Chardev::chardev_return`].
    pub fn add(&self, id: &str, backend: &ChardevBackend) -> Result<Arc<Chardev>> {
        self.add_inner(id, backend, false)
            .map_err(|e| e.prepend(format!("Failed to add chardev '{id}': ")))
    }

    fn add_inner(
        &self,
        id: &str,
        backend: &ChardevBackend,
        announce: bool,
    ) -> Result<Arc<Chardev>> {
        if self.find(id).is_some() {
            return Err(Error::generic(format!("Chardev with id '{id}' already exists")));
        }
        let chr = Chardev::open_in(id, backend, self.gate.clone(), Some(self), announce)?;
        let mut list = lock(&self.list);
        // Opening a waiting server can take a while, and the id may be taken by now.
        if list.iter().any(|c| c.label == id) {
            chr.close();
            return Err(Error::generic(format!(
                "attempt to add duplicate property '{id}' to object (type 'container')"
            )));
        }
        if let Some(registry) = self.registry.get() {
            if let Err(e) = qom::add_object(registry, &chr) {
                chr.close();
                return Err(e);
            }
        }
        list.push(chr.clone());
        Ok(chr)
    }

    /// `qemu_chr_new_from_opts()` for a `-chardev` set. Gives `None` after printing the list of
    /// backends for `-chardev help`. With `mux=on` the backend is called `<id>-base` and the
    /// mux on top of it gets the id.
    pub fn new_from_opts(&self, opts: &QemuOpts) -> Result<Option<Arc<Chardev>>> {
        let name = opts.get("backend");
        if name.is_some_and(is_help_option) {
            println!("{}", opts::backend_help());
            return Ok(None);
        }
        let Some(id) = opts.id() else {
            return Err(Error::generic("chardev: no id specified"));
        };
        let backend = opts::parse_opts(opts)?;
        if !opts.get_bool("mux", false) {
            return self.add_inner(id, &backend, true).map(Some);
        }
        let bid = format!("{id}-base");
        self.add_inner(&bid, &backend, true)?;
        let mux = ChardevBackend {
            u: ChardevBackendU::Mux(ChardevMuxWrapper {
                data: ChardevMux { chardev: bid.clone(), ..Default::default() },
            }),
        };
        match self.add_inner(id, &mux, true) {
            Ok(chr) => Ok(Some(chr)),
            Err(e) => {
                let _ = self.remove(&bid);
                Err(e)
            }
        }
    }

    /// `qemu_chr_new_from_name()` without the monitor: the chardev for an old style string
    /// such as `-serial mon:stdio`, made from a set in `list` that is gone again afterwards.
    /// `chardev:<id>` names a chardev that exists. The flag says whether the chardev is a mux
    /// the caller should put an HMP monitor on, as `mon:` asks. The error is `None` when
    /// QEMU says nothing of its own, and the caller's message is all the user gets.
    pub fn new_from_name(
        &self,
        list: &mut QemuOptsList,
        label: &str,
        filename: &str,
        permit_mux_mon: bool,
    ) -> std::result::Result<(Arc<Chardev>, bool), Option<Error>> {
        if let Some(id) = filename.strip_prefix("chardev:") {
            return self.find(id).map(|c| (c, false)).ok_or(None);
        }
        let h = opts::parse_compat(list, label, filename, permit_mux_mon)?;
        let opts = list.get(h).expect("just made");
        let mux = opts.get_bool("mux", false);
        let r = self.new_from_opts(opts);
        list.del(h);
        match r {
            Ok(Some(chr)) => Ok((chr, mux)),
            Ok(None) => Err(None),
            Err(e) => Err(Some(e)),
        }
    }

    /// `qmp_chardev_remove()`.
    pub fn remove(&self, id: &str) -> Result<()> {
        let mut list = lock(&self.list);
        let Some(pos) = list.iter().position(|c| c.label == id) else {
            return Err(Error::generic(format!("Chardev '{id}' not found")));
        };
        if list[pos].is_busy() {
            return Err(Error::generic(format!("Chardev '{id}' is busy")));
        }
        let chr = list.remove(pos);
        drop(list);
        chr.close();
        if let Some(registry) = self.registry.get() {
            qom::remove_object(registry, id);
        }
        Ok(())
    }

    /// The chardevs with a yank instance, which a socket chardev registers as it opens, oldest
    /// first.
    pub fn yank_instances(&self) -> Vec<String> {
        lock(&self.list)
            .iter()
            .filter(|c| matches!(c.backend, Backend::Socket(_)))
            .map(|c| c.label.clone())
            .collect()
    }

    /// `qmp_query_chardev()`. QEMU prepends each chardev to the list, so the newest comes
    /// first.
    pub fn query(&self) -> Vec<ChardevInfo> {
        lock(&self.list)
            .iter()
            .rev()
            .map(|c| ChardevInfo {
                label: c.label.clone(),
                filename: c.filename(),
                frontend_open: c.frontend_open(),
            })
            .collect()
    }

    /// `qmp_chardev_send_break()`.
    pub fn send_break(&self, id: &str) -> Result<()> {
        let chr =
            self.find(id).ok_or_else(|| Error::generic(format!("Chardev '{id}' not found")))?;
        chr.be_event(ChrEvent::Break);
        Ok(())
    }

    fn find_ringbuf(&self, device: &str) -> Result<Arc<Chardev>> {
        let chr = self
            .find(device)
            .ok_or_else(|| Error::generic(format!("Device '{device}' not found")))?;
        if chr.ringbuf().is_none() {
            return Err(Error::generic(format!("{device} is not a ringbuf device")));
        }
        Ok(chr)
    }

    /// `qmp_ringbuf_write()`.
    pub fn ringbuf_write(
        &self,
        device: &str,
        data: &str,
        format: Option<DataFormat>,
    ) -> Result<()> {
        let chr = self.find_ringbuf(device)?;
        chr.ringbuf().expect("checked").qmp_write(data, format)
    }

    /// `qmp_ringbuf_read()`.
    pub fn ringbuf_read(
        &self,
        device: &str,
        size: i64,
        format: Option<DataFormat>,
    ) -> Result<String> {
        let chr = self.find_ringbuf(device)?;
        chr.ringbuf().expect("checked").qmp_read(size, format)
    }
}
