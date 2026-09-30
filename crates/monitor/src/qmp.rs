// SPDX-License-Identifier: GPL-2.0-or-later

//! The QMP server, monitor/qmp.c.
//!
//! [`Qmp`] is the state QEMU keeps in globals: the two command lists, the list of monitors, the
//! in-band dispatcher and the event throttle. [`MonitorQmp`] is one monitor. Bytes read from its
//! client go to [`MonitorQmp::feed`], which parses them, runs out of band commands at once and
//! queues the rest for the dispatcher. Replies and events go to the monitor's writer.
//!
//! The dispatcher runs on whatever thread calls [`Qmp::dispatch_pending`] or
//! [`Qmp::run_dispatcher`], which is the main loop in the emulator, and out of band commands run
//! on the thread that feeds the bytes, the monitor I/O thread. The queue and the suspend and
//! resume rules between the two are QEMU's, because clients that send several requests at once
//! can see them.

use std::collections::VecDeque;
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, RwLock, Weak};
use std::time::{Duration, Instant};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_chardev::{Connection, Frontend};
use ruvm_qapi::dispatch::{
    DispatchEnv, QmpCommandList, qmp_dispatch, qmp_error_response, qmp_is_oob,
};
use ruvm_qapi::events::QapiEvent;
use ruvm_qapi::json::{self, Streamer};
use ruvm_qapi::visit::CompatPolicy;
use ruvm_qapi::{QDict, QValue};

use crate::control;
use crate::event::EventThrottle;
#[cfg(unix)]
use crate::fds::{FdSets, NamedFds};

/// `QMP_REQ_QUEUE_LEN_MAX`: how many in-band requests a monitor with OOB enabled may have
/// queued before it stops reading.
pub const QMP_REQ_QUEUE_LEN_MAX: usize = 8;

/// The command table type every QMP handler is registered in. The handler gets the monitor the
/// request came from, what `monitor_cur()` returns in QEMU.
pub type Commands = QmpCommandList<MonitorQmp>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A parsed request, or the parse error that stands in for one, `QMPRequest`.
type Request = Result<QValue>;

struct Session {
    /// `mon->commands == &qmp_commands`: capabilities negotiation is over.
    negotiated: bool,
    oob_offered: bool,
    oob: bool,
    requests: VecDeque<Request>,
    suspend_cnt: u32,
}

/// One QMP monitor, `MonitorQMP`.
pub struct MonitorQmp {
    qmp: Weak<Qmp>,
    id: String,
    pretty: bool,
    /// `monitor_requires_iothread()`: whether the chardev can be served from the I/O thread,
    /// which is what makes OOB possible.
    requires_iothread: bool,
    out: Mutex<Option<Box<dyn Write + Send>>>,
    parser: Mutex<Streamer>,
    session: Mutex<Session>,
    resumed: Condvar,
    /// The descriptors that came with the last read that had any, what the socket chardev
    /// keeps in `read_msgfds` until a command takes one.
    #[cfg(unix)]
    msgfds: Mutex<Vec<OwnedFd>>,
    #[cfg(unix)]
    named_fds: Mutex<NamedFds>,
}

impl std::fmt::Debug for MonitorQmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonitorQmp").field("id", &self.id).finish_non_exhaustive()
    }
}

impl MonitorQmp {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn qmp(&self) -> Option<Arc<Qmp>> {
        self.qmp.upgrade()
    }

    /// Whether capabilities negotiation is over.
    pub fn negotiated(&self) -> bool {
        lock(&self.session).negotiated
    }

    /// `qmp_oob_enabled()`.
    pub fn oob_enabled(&self) -> bool {
        lock(&self.session).oob
    }

    /// Replaces the writer replies and events go to. `None` drops output, as writing to a
    /// chardev without a client does.
    pub fn set_output(&self, out: Option<Box<dyn Write + Send>>) {
        *lock(&self.out) = out;
    }

    /// `qmp_send_response()`.
    pub fn send(&self, rsp: &QDict) {
        let mut text = json::to_string(&QValue::Dict(rsp.clone()), self.pretty);
        text.push('\n');
        let mut out = lock(&self.out);
        if let Some(w) = out.as_mut() {
            // A client that went away is noticed by the reader, which closes the session.
            let _ = w.write_all(text.as_bytes()).and_then(|()| w.flush());
        }
    }

    /// `monitor_qmp_caps_reset()` and the greeting, what `CHR_EVENT_OPENED` does.
    pub fn open(&self) {
        {
            let mut s = lock(&self.session);
            s.negotiated = false;
            s.oob_offered = self.requires_iothread;
            s.oob = false;
        }
        let greeting = self.greeting();
        self.send(&greeting);
    }

    /// `qmp_greeting()`.
    fn greeting(&self) -> QDict {
        let mut caps = Vec::new();
        if lock(&self.session).oob_offered {
            caps.push(QValue::str("oob"));
        }
        let version = control::version_value();
        let qmp = QDict::new().with("version", version).with("capabilities", QValue::List(caps));
        QDict::new().with("QMP", qmp)
    }

    /// What `CHR_EVENT_CLOSED` does: drop queued requests, resume a suspended monitor and
    /// start the parser over.
    pub fn close(&self) {
        {
            let mut s = lock(&self.session);
            let need_resume =
                (!s.oob || s.requests.len() == QMP_REQ_QUEUE_LEN_MAX) && !s.requests.is_empty();
            s.requests.clear();
            if need_resume {
                Self::resume_locked(&mut s);
                self.resumed.notify_all();
            }
        }
        *lock(&self.parser) = Streamer::new();
        #[cfg(unix)]
        {
            lock(&self.msgfds).clear();
            if let Some(qmp) = self.qmp() {
                qmp.fdsets().cleanup();
            }
        }
    }

    /// Stores descriptors that arrived with a read, `tcp_chr_recv()`. A read that brings some
    /// closes the ones still waiting from before, and a read that brings none leaves them.
    #[cfg(unix)]
    pub fn set_msgfds(&self, fds: Vec<OwnedFd>) {
        if !fds.is_empty() {
            *lock(&self.msgfds) = fds;
        }
    }

    /// `qemu_chr_fe_get_msgfd()`: the first waiting descriptor. Any others are closed, since
    /// every command that takes one takes exactly one.
    #[cfg(unix)]
    pub fn take_msgfd(&self) -> Option<OwnedFd> {
        let mut fds = std::mem::take(&mut *lock(&self.msgfds));
        if fds.is_empty() { None } else { Some(fds.swap_remove(0)) }
    }

    /// The descriptors `getfd` stored in this monitor.
    #[cfg(unix)]
    pub fn named_fds(&self) -> MutexGuard<'_, NamedFds> {
        lock(&self.named_fds)
    }

    /// `monitor_can_read()`: a suspended monitor reads nothing.
    pub fn can_read(&self) -> bool {
        lock(&self.session).suspend_cnt == 0
    }

    /// Blocks until the monitor is no longer suspended.
    pub fn wait_readable(&self) {
        let mut s = lock(&self.session);
        while s.suspend_cnt > 0 {
            s = self.resumed.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn resume_locked(s: &mut Session) {
        s.suspend_cnt = s.suspend_cnt.saturating_sub(1);
    }

    fn resume(&self) {
        let mut s = lock(&self.session);
        Self::resume_locked(&mut s);
        if s.suspend_cnt == 0 {
            self.resumed.notify_all();
        }
    }

    /// Feeds bytes from the client. QEMU's chardev hands the monitor one byte at a time while
    /// it can read, so a request that suspends the monitor stops the bytes after it. This
    /// returns how many bytes were taken, and the caller keeps the rest until
    /// [`MonitorQmp::can_read`] says yes again.
    pub fn feed(&self, bytes: &[u8]) -> usize {
        let mut taken = 0;
        for b in bytes {
            if !self.can_read() {
                break;
            }
            let reqs: Vec<Request> = {
                let mut p = lock(&self.parser);
                p.feed(std::slice::from_ref(b));
                std::iter::from_fn(|| p.next()).collect()
            };
            taken += 1;
            for req in reqs {
                self.handle_command(req);
            }
        }
        taken
    }

    /// Ends the input, `json_message_parser_flush()`, which turns an unfinished value into an
    /// error.
    pub fn flush(&self) {
        let reqs: Vec<Request> = {
            let mut p = lock(&self.parser);
            p.flush();
            std::iter::from_fn(|| p.next()).collect()
        };
        for req in reqs {
            self.handle_command(req);
        }
    }

    /// `handle_qmp_command()`.
    fn handle_command(&self, req: Request) {
        let Some(qmp) = self.qmp.upgrade() else { return };
        if let Ok(QValue::Dict(d)) = &req {
            if qmp_is_oob(d) {
                // OOB commands run at once, on this thread.
                self.dispatch(&qmp, req.as_ref().expect("checked above"));
                return;
            }
        }
        {
            let mut s = lock(&self.session);
            // With OOB off the monitor takes one request at a time, for compatibility.
            if !s.oob || s.requests.len() == QMP_REQ_QUEUE_LEN_MAX - 1 {
                s.suspend_cnt += 1;
            }
            assert!(s.requests.len() < QMP_REQ_QUEUE_LEN_MAX);
            s.requests.push_back(req);
        }
        qmp.wake();
    }

    /// `monitor_qmp_dispatch()`.
    fn dispatch(&self, qmp: &Qmp, req: &QValue) {
        let (negotiated, oob) = {
            let s = lock(&self.session);
            (s.negotiated, s.oob)
        };
        let cmds = if negotiated { qmp.commands() } else { qmp.cap_commands.clone() };
        let policy = qmp.policy();
        let env = DispatchEnv {
            policy: &policy,
            allow_oob: oob,
            machine_ready: qmp.machine_ready.load(Ordering::Acquire),
        };
        let Some(mut rsp) = qmp_dispatch(&cmds, req, &env, self) else { return };
        if !negotiated {
            if let Some(QValue::Dict(error)) = rsp.get("error") {
                if error.get_str("class") == Some(ErrorClass::CommandNotFound.as_str()) {
                    let mut error = error.clone();
                    error.remove("desc");
                    error.put("desc", "Expecting capabilities negotiation with 'qmp_capabilities'");
                    rsp.put("error", error);
                }
            }
        }
        self.send(&rsp);
    }

    /// `qmp_caps_accept()` and the switch to command mode, what `qmp_capabilities` does.
    pub(crate) fn accept_capabilities(&self, enable: &[&str]) -> Result<()> {
        let mut s = lock(&self.session);
        if s.negotiated {
            return Err(Error::new(
                ErrorClass::CommandNotFound,
                "Capabilities negotiation is already complete, command ignored",
            ));
        }
        let unavailable: Vec<&str> =
            enable.iter().copied().filter(|c| !(*c == "oob" && s.oob_offered)).collect();
        if !unavailable.is_empty() {
            return Err(Error::generic(format!(
                "Capability {} not available",
                unavailable.join(", ")
            )));
        }
        s.oob = enable.contains(&"oob");
        s.negotiated = true;
        Ok(())
    }

    /// Reads from `reader` until it ends, feeding the monitor and waiting while it is
    /// suspended. This is the monitor side of one client connection, from `CHR_EVENT_OPENED`
    /// to `CHR_EVENT_CLOSED`.
    pub fn serve(&self, mut reader: impl Read, writer: Box<dyn Write + Send>) -> io::Result<()> {
        self.serve_with(writer, |buf| reader.read(buf))
    }

    /// Like [`MonitorQmp::serve`] on a Unix socket, where the client can pass descriptors with
    /// SCM_RIGHTS for `getfd` and `add-fd`.
    #[cfg(unix)]
    pub fn serve_unix(&self, stream: UnixStream) -> io::Result<()> {
        let writer = Box::new(stream.try_clone()?);
        self.serve_with(writer, |buf| {
            let (n, fds) = ruvm_chardev::conn::recv_with_fds(&stream, buf)?;
            self.set_msgfds(fds);
            Ok(n)
        })
    }

    /// Serves one client of a chardev the monitor is attached to.
    pub fn serve_conn(&self, conn: &mut Connection) -> io::Result<()> {
        let writer = conn.writer()?;
        self.serve_with(writer, |buf| {
            let n = conn.recv(buf)?;
            #[cfg(unix)]
            self.set_msgfds(conn.take_fds());
            Ok(n)
        })
    }

    fn serve_with(
        &self,
        writer: Box<dyn Write + Send>,
        mut recv: impl FnMut(&mut [u8]) -> io::Result<usize>,
    ) -> io::Result<()> {
        self.set_output(Some(writer));
        self.open();
        let mut buf = [0u8; 4096];
        let result = loop {
            let n = match recv(&mut buf) {
                Ok(0) => break Ok(()),
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => break Err(e),
            };
            let mut rest = &buf[..n];
            while !rest.is_empty() {
                self.wait_readable();
                let taken = self.feed(rest);
                rest = &rest[taken..];
            }
        };
        self.close();
        self.set_output(None);
        result
    }
}

impl Frontend for MonitorQmp {
    fn serve(&self, conn: &mut Connection) -> io::Result<()> {
        self.serve_conn(conn)
    }
}

struct Monitors {
    /// In the order the dispatcher looks at them. A monitor whose request was just taken goes
    /// to the back, so no client can starve another.
    list: VecDeque<Arc<MonitorQmp>>,
    shutdown: bool,
}

/// The QMP state of the process.
pub struct Qmp {
    commands: RwLock<Arc<Commands>>,
    cap_commands: Arc<Commands>,
    monitors: Mutex<Monitors>,
    work: Condvar,
    throttle: Mutex<EventThrottle>,
    event_clock: RwLock<Arc<dyn Fn() -> i64 + Send + Sync>>,
    policy: RwLock<CompatPolicy>,
    machine_ready: AtomicBool,
    /// The monitor whose in-band request the dispatcher is running,
    /// `qmp_dispatcher_current_mon`.
    current: Mutex<Option<Weak<MonitorQmp>>>,
    #[cfg(unix)]
    fdsets: Mutex<FdSets>,
}

impl std::fmt::Debug for Qmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qmp").finish_non_exhaustive()
    }
}

impl Qmp {
    /// Sets up both command lists with the commands the monitor implements itself, as
    /// `monitor_init_qmp_commands()` does. Other subsystems add theirs with
    /// [`Qmp::register`].
    pub fn new() -> Arc<Qmp> {
        let mut commands = Commands::new();
        control::register(&mut commands);
        let mut cap_commands = Commands::new();
        control::register_negotiation(&mut cap_commands);
        let start = Instant::now();
        Arc::new(Qmp {
            commands: RwLock::new(Arc::new(commands)),
            cap_commands: Arc::new(cap_commands),
            monitors: Mutex::new(Monitors { list: VecDeque::new(), shutdown: false }),
            work: Condvar::new(),
            throttle: Mutex::new(EventThrottle::new()),
            event_clock: RwLock::new(Arc::new(move || start.elapsed().as_nanos() as i64)),
            policy: RwLock::new(CompatPolicy::default()),
            machine_ready: AtomicBool::new(true),
            current: Mutex::new(None),
            #[cfg(unix)]
            fdsets: Mutex::new(FdSets::new()),
        })
    }

    /// The command table monitors in command mode use.
    pub fn commands(&self) -> Arc<Commands> {
        self.commands.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Changes the command table, for registering, enabling and disabling commands. Requests
    /// already running keep the table they started with.
    pub fn update_commands(&self, f: impl FnOnce(&mut Commands)) {
        let mut w = self.commands.write().unwrap_or_else(|e| e.into_inner());
        let mut table = (**w).clone();
        f(&mut table);
        *w = Arc::new(table);
    }

    /// Registers commands through one of the generated `register_*` functions.
    pub fn register(&self, f: impl FnOnce(&mut Commands)) {
        self.update_commands(f);
    }

    /// The fd sets `add-fd` fills, shared by every monitor.
    #[cfg(unix)]
    pub fn fdsets(&self) -> MutexGuard<'_, FdSets> {
        lock(&self.fdsets)
    }

    pub fn policy(&self) -> CompatPolicy {
        *self.policy.read().unwrap_or_else(|e| e.into_inner())
    }

    /// Sets the `-compat` policy.
    pub fn set_policy(&self, policy: CompatPolicy) {
        *self.policy.write().unwrap_or_else(|e| e.into_inner()) = policy;
    }

    /// Whether commands without `allow-preconfig` may run. It is false while `-preconfig`
    /// holds the machine back.
    pub fn set_machine_ready(&self, ready: bool) {
        self.machine_ready.store(ready, Ordering::Release);
    }

    /// Sets the clock event throttling runs on, in nanoseconds. qtest switches it to the
    /// virtual clock.
    pub fn set_event_clock(&self, clock: impl Fn() -> i64 + Send + Sync + 'static) {
        *self.event_clock.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(clock);
    }

    fn event_now(&self) -> i64 {
        let clock = self.event_clock.read().unwrap_or_else(|e| e.into_inner()).clone();
        clock()
    }

    /// Creates a monitor, `monitor_new_qmp()`. `requires_iothread` is true for chardevs that
    /// can run in the I/O thread, sockets among them, and decides whether OOB is offered.
    pub fn add_monitor(
        self: &Arc<Self>,
        id: &str,
        pretty: bool,
        requires_iothread: bool,
    ) -> Arc<MonitorQmp> {
        let mon = Arc::new(MonitorQmp {
            qmp: Arc::downgrade(self),
            id: id.to_string(),
            pretty,
            requires_iothread,
            out: Mutex::new(None),
            parser: Mutex::new(Streamer::new()),
            session: Mutex::new(Session {
                negotiated: false,
                oob_offered: requires_iothread,
                oob: false,
                requests: VecDeque::new(),
                suspend_cnt: 0,
            }),
            resumed: Condvar::new(),
            #[cfg(unix)]
            msgfds: Mutex::new(Vec::new()),
            #[cfg(unix)]
            named_fds: Mutex::new(NamedFds::default()),
        });
        lock(&self.monitors).list.push_back(mon.clone());
        mon
    }

    /// Removes a monitor. Its queued requests are dropped.
    pub fn remove_monitor(&self, mon: &Arc<MonitorQmp>) {
        lock(&self.monitors).list.retain(|m| !Arc::ptr_eq(m, mon));
        mon.close();
    }

    /// Whether the dispatcher is running a request from `mon` right now,
    /// `monitor_qmp_dispatcher_is_servicing()`.
    pub fn is_servicing(&self, mon: &Arc<MonitorQmp>) -> bool {
        lock(&self.current).as_ref().is_some_and(|w| std::ptr::eq(w.as_ptr(), Arc::as_ptr(mon)))
    }

    pub fn monitors(&self) -> Vec<Arc<MonitorQmp>> {
        lock(&self.monitors).list.iter().cloned().collect()
    }

    fn wake(&self) {
        let _g = lock(&self.monitors);
        self.work.notify_all();
    }

    /// `monitor_qmp_requests_pop_any_with_lock()`. Returns the monitor and its request, and
    /// resumes the monitor if the queue has room again and OOB is on.
    fn pop_any(monitors: &mut Monitors) -> Option<(Arc<MonitorQmp>, Request, bool)> {
        let idx = monitors.list.iter().position(|m| !lock(&m.session).requests.is_empty())?;
        let mon = monitors.list.remove(idx).expect("index from position");
        monitors.list.push_back(mon.clone());
        let mut s = lock(&mon.session);
        let req = s.requests.pop_front().expect("checked non-empty");
        let oob = s.oob;
        if oob && s.requests.len() == QMP_REQ_QUEUE_LEN_MAX - 1 {
            MonitorQmp::resume_locked(&mut s);
            mon.resumed.notify_all();
        }
        drop(s);
        Some((mon, req, oob))
    }

    fn process(self: &Arc<Self>, mon: &Arc<MonitorQmp>, req: Request, oob: bool) {
        *lock(&self.current) = Some(Arc::downgrade(mon));
        match req {
            Ok(req) => mon.dispatch(self, &req),
            Err(e) => mon.send(&qmp_error_response(&e)),
        }
        *lock(&self.current) = None;
        if !oob {
            mon.resume();
        }
    }

    /// Runs every queued in-band request and every event whose throttle period ended, then
    /// returns. This is one pass of `monitor_qmp_dispatcher_co()`.
    pub fn dispatch_pending(self: &Arc<Self>) {
        self.run_timers();
        loop {
            let next = Self::pop_any(&mut lock(&self.monitors));
            let Some((mon, req, oob)) = next else { break };
            self.process(&mon, req, oob);
        }
    }

    /// Runs the dispatcher on this thread until [`Qmp::shutdown`].
    pub fn run_dispatcher(self: &Arc<Self>) {
        loop {
            let next = {
                let mut m = lock(&self.monitors);
                loop {
                    if m.shutdown {
                        return;
                    }
                    if let Some(next) = Self::pop_any(&mut m) {
                        break Some(next);
                    }
                    let deadline = lock(&self.throttle).next_deadline();
                    let now = self.event_now();
                    match deadline {
                        Some(d) if d <= now => break None,
                        Some(d) => {
                            let wait = Duration::from_nanos((d - now) as u64);
                            m = self
                                .work
                                .wait_timeout(m, wait)
                                .unwrap_or_else(|e| e.into_inner())
                                .0;
                        }
                        None => m = self.work.wait(m).unwrap_or_else(|e| e.into_inner()),
                    }
                }
            };
            match next {
                Some((mon, req, oob)) => self.process(&mon, req, oob),
                None => self.run_timers(),
            }
        }
    }

    /// Stops [`Qmp::run_dispatcher`], `qmp_dispatcher_co_shutdown`.
    pub fn shutdown(&self) {
        lock(&self.monitors).shutdown = true;
        self.work.notify_all();
    }

    /// `monitor_qapi_event_emit()`: sends an event to every monitor in command mode.
    fn broadcast(&self, qdict: &QDict) {
        for mon in self.monitors() {
            if mon.negotiated() {
                mon.send(qdict);
            }
        }
    }

    /// `qapi_event_emit()`. `qdict` is what one of the generated `event_*` functions built.
    /// Throttled events may go out later, from the dispatcher.
    pub fn emit_event(&self, qdict: QDict) {
        let event = qdict.get_str("event").and_then(QapiEvent::from_name);
        let Some(event) = event else {
            self.broadcast(&qdict);
            return;
        };
        let now = self.event_now();
        let out = lock(&self.throttle).queue(event, qdict, now);
        match out {
            Some(qdict) => self.broadcast(&qdict),
            // Make sure a waiting dispatcher knows about the new deadline.
            None => self.wake(),
        }
    }

    /// Sends the throttled events whose period ended, `monitor_qapi_event_handler()`.
    pub fn run_timers(&self) {
        let now = self.event_now();
        let out = lock(&self.throttle).expire(now);
        for qdict in out {
            self.broadcast(&qdict);
        }
    }
}
