// SPDX-License-Identifier: GPL-2.0-or-later

//! Character device backends and the mux, chardev/.
//!
//! [`Chardevs`] is the `/chardevs` container: every chardev by id, what `chardev-add`,
//! `chardev-remove` and `query-chardev` work on. A frontend such as a monitor attaches to one
//! chardev with [`Chardev::attach`] and then gets each client connection on a thread of its
//! own, until it detaches. A chardev takes one frontend at a time.
//!
//! Backends so far are `null` and `socket` on Unix and TCP sockets. [`opts`] turns `-chardev`
//! and the old compat strings into backends.

#![forbid(unsafe_code)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

use ruvm_base::{Error, Result};
use ruvm_qapi::opts::{QemuOpts, is_help_option};
use ruvm_qapi::types::{ChardevBackend, ChardevBackendU, ChardevInfo};

pub mod conn;
pub mod opts;
pub mod qom;
pub mod socket;

pub use conn::Connection;
pub use socket::SocketChardev;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The frontend side, what `qemu_chr_fe_set_handlers()` installs. `serve` runs on the
/// chardev's thread for each client from `CHR_EVENT_OPENED` to `CHR_EVENT_CLOSED` and returns
/// when [`Connection::recv`] gives 0.
pub trait Frontend: Send + Sync {
    fn serve(&self, conn: &mut Connection) -> std::io::Result<()>;
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
}

/// The backend names `query-chardev-backends` lists.
pub const BACKENDS: &[&str] = &["null", "socket"];

/// One chardev, `Chardev`.
#[derive(Debug)]
pub struct Chardev {
    label: String,
    backend: Backend,
    /// Set while a frontend is attached, `chr->fe`.
    busy: AtomicBool,
    /// Held by the thread serving a frontend. A frontend attached right after another one
    /// detached waits here until the old thread has let go of the connection.
    serving: Arc<Mutex<()>>,
    gate: Arc<Gate>,
}

impl Chardev {
    /// `chardev_new()` and the backend's open. Sockets connect, or listen and maybe wait for a
    /// client, before this returns.
    pub fn open(label: &str, backend: &ChardevBackend) -> Result<Arc<Chardev>> {
        Self::open_gated(label, backend, Arc::default())
    }

    fn open_gated(label: &str, backend: &ChardevBackend, gate: Arc<Gate>) -> Result<Arc<Chardev>> {
        let backend = match &backend.u {
            ChardevBackendU::Null(_) => Backend::Null,
            ChardevBackendU::Socket(s) => Backend::Socket(SocketChardev::open(&s.data)?),
            _ => {
                let kind = backend.u.tag().as_str();
                return Err(Error::generic(format!(
                    "chardev backend '{kind}' is not supported by ruvm yet"
                )));
            }
        };
        Ok(Arc::new(Chardev {
            label: label.to_string(),
            backend,
            busy: AtomicBool::new(false),
            serving: Arc::new(Mutex::new(())),
            gate,
        }))
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    /// `qemu_chr_get_filename()`: the backend's own text, or the type name after `chardev-`.
    pub fn filename(&self) -> String {
        match &self.backend {
            Backend::Null => "null".to_string(),
            Backend::Socket(s) => s.filename(),
        }
    }

    /// `qemu_chr_is_busy()`.
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    /// The socket backend, when this is a socket chardev.
    pub fn socket(&self) -> Option<&SocketChardev> {
        match &self.backend {
            Backend::Socket(s) => Some(s),
            Backend::Null => None,
        }
    }

    /// `qemu_chr_fe_init()` and `qemu_chr_fe_set_handlers()`. Fails when another frontend has
    /// the chardev.
    pub fn attach(self: &Arc<Self>, fe: Arc<dyn Frontend>) -> Result<Attachment> {
        if self.busy.swap(true, Ordering::AcqRel) {
            return Err(Error::generic(format!("chardev '{}' is already in use", self.label)));
        }
        let stop = Arc::new(AtomicBool::new(false));
        let thread = match &self.backend {
            Backend::Null => None,
            Backend::Socket(_) => {
                let chr = self.clone();
                let stop = stop.clone();
                let name = format!("chardev-{}", self.label);
                let spawned = std::thread::Builder::new().name(name).spawn(move || {
                    chr.gate.wait();
                    let serving = chr.serving.clone();
                    let _g = lock(&serving);
                    let Backend::Socket(s) = &chr.backend else { unreachable!("checked above") };
                    s.run(&stop, |conn| fe.serve(conn));
                });
                match spawned {
                    Ok(t) => Some(t),
                    Err(e) => {
                        self.busy.store(false, Ordering::Release);
                        return Err(Error::from_io("Failed to start the chardev thread", e));
                    }
                }
            }
        };
        Ok(Attachment { chr: self.clone(), stop, thread })
    }
}

/// A frontend attached to a chardev. Dropping it detaches the frontend, as
/// `qemu_chr_fe_deinit()` does, and leaves any connection open for the next frontend.
#[derive(Debug)]
pub struct Attachment {
    chr: Arc<Chardev>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Attachment {
    pub fn chardev(&self) -> &Arc<Chardev> {
        &self.chr
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
        self.chr.busy.store(false, Ordering::Release);
    }
}

/// Every chardev, the `/chardevs` container.
#[derive(Debug, Default)]
pub struct Chardevs {
    list: Mutex<Vec<Arc<Chardev>>>,
    /// Where the objects for each chardev go, once there is a QOM tree.
    registry: OnceLock<ruvm_qom::Registry>,
    gate: Arc<Gate>,
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

    /// `qemu_chr_find()`.
    pub fn find(&self, id: &str) -> Option<Arc<Chardev>> {
        lock(&self.list).iter().find(|c| c.label == id).cloned()
    }

    /// `qmp_chardev_add()` without the QMP reply.
    pub fn add(&self, id: &str, backend: &ChardevBackend) -> Result<Arc<Chardev>> {
        self.add_inner(id, backend)
            .map_err(|e| e.prepend(format!("Failed to add chardev '{id}': ")))
    }

    fn add_inner(&self, id: &str, backend: &ChardevBackend) -> Result<Arc<Chardev>> {
        if self.find(id).is_some() {
            return Err(Error::generic(format!("Chardev with id '{id}' already exists")));
        }
        let chr = Chardev::open_gated(id, backend, self.gate.clone())?;
        let mut list = lock(&self.list);
        // Opening a waiting server can take a while, and the id may be taken by now.
        if list.iter().any(|c| c.label == id) {
            return Err(Error::generic(format!(
                "attempt to add duplicate property '{id}' to object (type 'container')"
            )));
        }
        if let Some(registry) = self.registry.get() {
            qom::add_object(registry, &chr)?;
        }
        list.push(chr.clone());
        Ok(chr)
    }

    /// `qemu_chr_new_from_opts()` for a `-chardev` set. Gives `None` after printing the list of
    /// backends for `-chardev help`.
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
        if opts.get_bool("mux", false) {
            return Err(Error::generic("chardev backend 'mux' is not supported by ruvm yet"));
        }
        self.add_inner(id, &backend).map(Some)
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
        list.remove(pos);
        if let Some(registry) = self.registry.get() {
            qom::remove_object(registry, id);
        }
        Ok(())
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
                frontend_open: c.is_busy(),
            })
            .collect()
    }
}
