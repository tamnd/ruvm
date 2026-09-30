// SPDX-License-Identifier: MIT OR Apache-2.0

//! The reactor and its handle.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use mio::event::Source;
use mio::{Events, Interest, Poll as MioPoll, Token, Waker as MioWaker};
use ruvm_base::{TimerId, TimerList};

use crate::bh::{Bh, BhInner, EventNotifier, EventNotifierState};
use crate::clock::{Clock, ClockType};

const WAKE_TOKEN: Token = Token(usize::MAX);
const CLOCKS: [ClockType; 4] =
    [ClockType::Realtime, ClockType::Virtual, ClockType::Host, ClockType::VirtualRt];

type Posted = Box<dyn FnOnce(&mut Reactor) + Send>;
type TimerCallback = Box<dyn FnMut(&mut Reactor)>;

/// Readiness reported to an fd handler.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ready {
    pub readable: bool,
    pub writable: bool,
    /// The peer hung up or the fd is in an error state. A read will say which.
    pub closed: bool,
}

/// Names an fd handler registered with [`Reactor::add_source`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FdToken(usize);

/// Names a task spawned on the reactor's executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskId(usize);

/// A timer on one of the reactor's clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Timer {
    clock: ClockType,
    id: TimerId,
}

impl Timer {
    pub fn clock(&self) -> ClockType {
        self.clock
    }
}

/// The part of a reactor that other threads can touch.
pub(crate) struct Shared {
    id: u64,
    waker: MioWaker,
    // Set while the reactor is blocked or about to block, so a wake is only sent when needed. This
    // is `notify_me` in util/async.c.
    notified: AtomicBool,
    queue: Mutex<Inbox>,
    stop: AtomicBool,
}

#[derive(Default)]
struct Inbox {
    bhs: VecDeque<Arc<BhInner>>,
    posted: VecDeque<Posted>,
    ready_tasks: Vec<usize>,
    sleeps: Vec<(ClockType, i64, Waker)>,
}

impl Shared {
    pub(crate) fn push_bh(&self, bh: Arc<BhInner>) {
        self.inbox().bhs.push_back(bh);
        self.wake();
    }

    fn inbox(&self) -> std::sync::MutexGuard<'_, Inbox> {
        self.queue.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn wake(&self) {
        if !self.notified.swap(true, Ordering::AcqRel) {
            // A failed wake means the poller is gone, which only happens while the reactor is
            // being dropped. Nothing is waiting for the wake in that case.
            let _ = self.waker.wake();
        }
    }
}

/// A `Send + Sync` handle to a reactor.
#[derive(Clone)]
pub struct ReactorHandle {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for ReactorHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReactorHandle").field("id", &self.shared.id).finish()
    }
}

impl ReactorHandle {
    /// Runs `f` on the reactor's thread at its next iteration.
    pub fn post(&self, f: impl FnOnce(&mut Reactor) + Send + 'static) {
        self.shared.inbox().posted.push_back(Box::new(f));
        self.shared.wake();
    }

    /// Makes a bottom half that runs on this reactor. It is not scheduled yet.
    pub fn bh(&self, f: impl FnMut() + Send + 'static) -> Bh {
        Bh::new(Arc::clone(&self.shared), f)
    }

    /// Makes an event notifier whose handler runs on this reactor.
    pub fn event_notifier(
        &self,
        f: impl FnMut(&EventNotifierState) + Send + 'static,
    ) -> EventNotifier {
        EventNotifier::new(Arc::clone(&self.shared), f)
    }

    /// `aio_notify()`: makes the reactor go round its loop once more.
    pub fn wake(&self) {
        self.shared.wake();
    }

    /// Asks [`Reactor::run`] to return after the current iteration.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.wake();
    }

    pub(crate) fn same(&self, other: &ReactorHandle) -> bool {
        Arc::ptr_eq(&self.shared, &other.shared)
    }
}

trait FdHandler {
    fn source(&mut self) -> &mut dyn Source;
    fn call(&mut self, reactor: &mut Reactor, ready: Ready);
}

struct Entry<S, F> {
    source: S,
    handler: F,
}

impl<S: Source, F: FnMut(&mut Reactor, &mut S, Ready)> FdHandler for Entry<S, F> {
    fn source(&mut self) -> &mut dyn Source {
        &mut self.source
    }

    fn call(&mut self, reactor: &mut Reactor, ready: Ready) {
        (self.handler)(reactor, &mut self.source, ready)
    }
}

enum Slot {
    Free,
    // The handler is out of the slab because it is running.
    Running { removed: bool },
    Live(Box<dyn FdHandler>),
}

enum TaskSlot {
    Free,
    Running,
    Live(Pin<Box<dyn Future<Output = ()>>>),
}

struct TaskWaker {
    id: usize,
    shared: Arc<Shared>,
}

impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.shared.inbox().ready_tasks.push(self.id);
        self.shared.wake();
    }
}

/// The equivalent of `aio_poll()` for one polling function, see [`Reactor::set_polling`].
type PollFn = Box<dyn FnMut(&mut Reactor) -> bool>;

/// Chooses how long to busy poll next time from how long the reactor blocked. The QEMU policy for
/// iothreads lives with the iothread object, this crate only runs it.
pub type PollTuning = Box<dyn FnMut(Duration, Duration) -> Duration>;

thread_local! {
    static CURRENT: RefCell<Vec<ReactorHandle>> = const { RefCell::new(Vec::new()) };
}

/// The handle of the reactor that is running on this thread, if one is.
pub fn current() -> Option<ReactorHandle> {
    CURRENT.with(|c| c.borrow().last().cloned())
}

/// A future that resolves once `clock` reaches `deadline`. It must be polled from a task on a
/// running reactor.
pub fn sleep(clock: ClockType, deadline: i64) -> impl Future<Output = ()> {
    Sleep { clock, deadline }
}

struct Sleep {
    clock: ClockType,
    deadline: i64,
}

impl Future for Sleep {
    type Output = ();

    // Every pending poll arms a one shot timer. A spurious wake costs one extra timer, which is
    // cheaper than tracking whether the last one is still armed.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let handle = current().expect("sleep polled outside a running reactor");
        let now = CURRENT_CLOCK.with(|c| c.borrow().as_ref().map(|c| c.now(self.clock)));
        if now.is_some_and(|n| n >= self.deadline) {
            return Poll::Ready(());
        }
        handle.shared.inbox().sleeps.push((self.clock, self.deadline, cx.waker().clone()));
        Poll::Pending
    }
}

thread_local! {
    static CURRENT_CLOCK: RefCell<Option<Clock>> = const { RefCell::new(None) };
}

/// One thread's event loop.
pub struct Reactor {
    poll: MioPoll,
    events: Events,
    shared: Arc<Shared>,
    clock: Clock,
    timers: [TimerList<Option<TimerCallback>>; 4],
    fds: Vec<Slot>,
    free_fds: Vec<usize>,
    tasks: Vec<TaskSlot>,
    free_tasks: Vec<usize>,
    poll_fns: Vec<Option<PollFn>>,
    poll_max: Duration,
    poll_now: Duration,
    tuning: Option<PollTuning>,
    // Keeps the reactor `!Send`: fd handlers and tasks are not required to be `Send`.
    _not_send: std::marker::PhantomData<*const ()>,
}

impl std::fmt::Debug for Reactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor")
            .field("id", &self.shared.id)
            .field("fds", &(self.fds.len() - self.free_fds.len()))
            .field("tasks", &(self.tasks.len() - self.free_tasks.len()))
            .finish()
    }
}

static NEXT_ID: AtomicU64 = AtomicU64::new(0);

impl Reactor {
    pub fn new(clock: Clock) -> io::Result<Self> {
        let poll = MioPoll::new()?;
        let waker = MioWaker::new(poll.registry(), WAKE_TOKEN)?;
        let shared = Arc::new(Shared {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            waker,
            notified: AtomicBool::new(false),
            queue: Mutex::new(Inbox::default()),
            stop: AtomicBool::new(false),
        });
        let reactor = Reactor {
            poll,
            events: Events::with_capacity(256),
            shared,
            clock,
            timers: Default::default(),
            fds: Vec::new(),
            free_fds: Vec::new(),
            tasks: Vec::new(),
            free_tasks: Vec::new(),
            poll_fns: Vec::new(),
            poll_max: Duration::ZERO,
            poll_now: Duration::ZERO,
            tuning: None,
            _not_send: std::marker::PhantomData,
        };
        reactor.clock.attach(reactor.handle());
        Ok(reactor)
    }

    pub fn handle(&self) -> ReactorHandle {
        ReactorHandle { shared: Arc::clone(&self.shared) }
    }

    pub fn clock(&self) -> &Clock {
        &self.clock
    }

    pub fn now(&self, clock: ClockType) -> i64 {
        self.clock.now(clock)
    }

    /// Makes a bottom half on this reactor.
    pub fn bh(&self, f: impl FnMut() + Send + 'static) -> Bh {
        self.handle().bh(f)
    }

    /// `timer_new_ns()`: a timer that is not armed yet.
    pub fn timer_new(&mut self, clock: ClockType, f: impl FnMut(&mut Reactor) + 'static) -> Timer {
        let id = self.list(clock).insert(Some(Box::new(f)));
        Timer { clock, id }
    }

    /// `timer_mod()`, with the deadline in nanoseconds on the timer's clock.
    pub fn timer_mod(&mut self, t: Timer, deadline: i64) {
        self.list(t.clock).arm(t.id, deadline);
    }

    /// `timer_mod_anticipate()`.
    pub fn timer_mod_earlier(&mut self, t: Timer, deadline: i64) {
        self.list(t.clock).arm_earlier(t.id, deadline);
    }

    /// `timer_del()`.
    pub fn timer_del(&mut self, t: Timer) {
        self.list(t.clock).disarm(t.id);
    }

    /// `timer_free()`.
    pub fn timer_free(&mut self, t: Timer) {
        self.list(t.clock).remove(t.id);
    }

    pub fn timer_pending(&self, t: Timer) -> bool {
        self.timers[clock_index(t.clock)].is_armed(t.id)
    }

    pub fn timer_expire_time(&self, t: Timer) -> Option<i64> {
        self.timers[clock_index(t.clock)].deadline_of(t.id)
    }

    /// Registers `source` and calls `handler` when it becomes ready. The reactor owns the source
    /// until [`Reactor::remove_source`] gives it back.
    ///
    /// Readiness is edge triggered on every host, so a handler has to read or write until the
    /// operation would block before it returns, or it will not be called again.
    pub fn add_source<S, F>(
        &mut self,
        mut source: S,
        interest: Interest,
        handler: F,
    ) -> io::Result<FdToken>
    where
        S: Source + 'static,
        F: FnMut(&mut Reactor, &mut S, Ready) + 'static,
    {
        let idx = self.free_fds.pop().unwrap_or_else(|| {
            self.fds.push(Slot::Free);
            self.fds.len() - 1
        });
        if let Err(e) = self.poll.registry().register(&mut source, Token(idx), interest) {
            self.free_fds.push(idx);
            return Err(e);
        }
        self.fds[idx] = Slot::Live(Box::new(Entry { source, handler }));
        Ok(FdToken(idx))
    }

    /// Changes what a registered source is watched for.
    pub fn reregister(&mut self, token: FdToken, interest: Interest) -> io::Result<()> {
        match self.fds.get_mut(token.0) {
            Some(Slot::Live(h)) => {
                self.poll.registry().reregister(h.source(), Token(token.0), interest)
            }
            Some(Slot::Running { .. }) => {
                Err(io::Error::other("the handler is running, reregister from inside it"))
            }
            _ => Err(io::Error::from(io::ErrorKind::NotFound)),
        }
    }

    /// Unregisters a source and drops it. A handler may remove itself, and the removal happens
    /// when it returns.
    pub fn remove_source(&mut self, token: FdToken) -> bool {
        match self.fds.get_mut(token.0) {
            Some(Slot::Running { removed }) => {
                *removed = true;
                true
            }
            Some(slot @ Slot::Live(_)) => {
                if let Slot::Live(mut h) = std::mem::replace(slot, Slot::Free) {
                    let _ = self.poll.registry().deregister(h.source());
                }
                self.free_fds.push(token.0);
                true
            }
            _ => false,
        }
    }

    /// Spawns a task on this reactor's executor. Tasks never move to another thread.
    pub fn spawn(&mut self, fut: impl Future<Output = ()> + 'static) -> TaskId {
        let idx = self.free_tasks.pop().unwrap_or_else(|| {
            self.tasks.push(TaskSlot::Free);
            self.tasks.len() - 1
        });
        self.tasks[idx] = TaskSlot::Live(Box::pin(fut));
        self.shared.inbox().ready_tasks.push(idx);
        TaskId(idx)
    }

    /// Adds a function that the reactor calls while busy polling. It returns true when it found
    /// work, which ends the polling phase early. Busy polling only happens after
    /// [`Reactor::set_polling`] turns it on.
    pub fn add_poll_fn(&mut self, f: impl FnMut(&mut Reactor) -> bool + 'static) -> usize {
        self.poll_fns.push(Some(Box::new(f)));
        self.poll_fns.len() - 1
    }

    pub fn remove_poll_fn(&mut self, idx: usize) {
        if let Some(f) = self.poll_fns.get_mut(idx) {
            *f = None;
        }
    }

    /// Turns busy polling on with a limit of `max` per iteration, or off with a limit of zero.
    /// `tuning` gets the current polling time and how long the reactor last blocked, and returns
    /// the polling time for the next iteration.
    pub fn set_polling(&mut self, max: Duration, tuning: Option<PollTuning>) {
        self.poll_max = max;
        self.poll_now = Duration::ZERO;
        self.tuning = tuning;
    }

    /// How long the reactor will busy poll on its next iteration.
    pub fn polling_time(&self) -> Duration {
        self.poll_now
    }

    /// Runs until [`ReactorHandle::stop`] is called.
    pub fn run(&mut self) -> io::Result<()> {
        self.shared.stop.store(false, Ordering::Release);
        while !self.shared.stop.load(Ordering::Acquire) {
            self.run_once(true)?;
        }
        Ok(())
    }

    /// Runs until `done` returns true, checking it after every iteration.
    pub fn run_until(&mut self, mut done: impl FnMut(&mut Reactor) -> bool) -> io::Result<()> {
        while !done(self) {
            self.run_once(true)?;
        }
        Ok(())
    }

    /// Runs the reactor until `fut` completes and returns its output. This is how a command line
    /// tool like qemu-img drives async code from `main`.
    pub fn block_on<T: 'static>(
        &mut self,
        fut: impl Future<Output = T> + 'static,
    ) -> io::Result<T> {
        let out = std::rc::Rc::new(RefCell::new(None));
        let slot = std::rc::Rc::clone(&out);
        self.spawn(async move {
            *slot.borrow_mut() = Some(fut.await);
        });
        self.run_until(|_| out.borrow().is_some())?;
        Ok(out.borrow_mut().take().expect("checked by run_until"))
    }

    /// One iteration of the loop, `aio_poll()`. With `blocking` false it never waits. Returns
    /// whether anything ran.
    pub fn run_once(&mut self, blocking: bool) -> io::Result<bool> {
        let _cur = self.enter();
        let mut progress = self.dispatch_timers();
        progress |= self.dispatch_inbox();

        if !self.poll_now.is_zero() && !self.inbox_busy() {
            let start = std::time::Instant::now();
            while start.elapsed() < self.poll_now {
                if self.run_poll_fns() {
                    progress = true;
                    break;
                }
            }
        }

        let timeout = if !blocking || progress || self.inbox_busy() {
            Some(Duration::ZERO)
        } else {
            self.timeout()
        };
        let started = std::time::Instant::now();
        self.poll_events(timeout)?;
        let blocked = started.elapsed();
        self.shared.notified.store(false, Ordering::Release);

        progress |= self.dispatch_events();
        progress |= self.dispatch_timers();
        progress |= self.dispatch_inbox();

        if !self.poll_max.is_zero() {
            if let Some(tuning) = self.tuning.as_mut() {
                self.poll_now = tuning(self.poll_now, blocked).min(self.poll_max);
            }
        }
        Ok(progress)
    }

    fn enter(&self) -> CurrentGuard {
        CURRENT.with(|c| c.borrow_mut().push(self.handle()));
        let prev = CURRENT_CLOCK.with(|c| c.borrow_mut().replace(self.clock.clone()));
        CurrentGuard { prev_clock: prev }
    }

    fn list(&mut self, clock: ClockType) -> &mut TimerList<Option<TimerCallback>> {
        &mut self.timers[clock_index(clock)]
    }

    fn inbox_busy(&self) -> bool {
        let inbox = self.shared.inbox();
        !inbox.bhs.is_empty() || !inbox.posted.is_empty() || !inbox.ready_tasks.is_empty()
    }

    fn poll_events(&mut self, timeout: Option<Duration>) -> io::Result<()> {
        loop {
            match self.poll.poll(&mut self.events, timeout) {
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                r => return r,
            }
        }
    }

    // The time until the earliest timer on any clock that is running. Virtual timers do not count
    // while the VM is stopped, and a change to that wakes the reactor.
    fn timeout(&mut self) -> Option<Duration> {
        let mut best: Option<i64> = None;
        for clock in CLOCKS {
            if matches!(clock, ClockType::Virtual | ClockType::VirtualRt)
                && !self.clock.is_running()
            {
                continue;
            }
            if let Some(d) = self.list(clock).next_deadline() {
                let wait = (d - self.clock.now(clock)).max(0);
                best = Some(best.map_or(wait, |b| b.min(wait)));
            }
        }
        best.map(|ns| Duration::from_nanos(ns as u64))
    }

    fn dispatch_timers(&mut self) -> bool {
        let mut progress = false;
        for clock in CLOCKS {
            loop {
                let now = self.clock.now(clock);
                let mut due = Vec::new();
                self.list(clock).expire(now, |_, id| due.push(id));
                if due.is_empty() {
                    break;
                }
                for id in due {
                    let Some(mut cb) = self.list(clock).callback_mut(id).and_then(Option::take)
                    else {
                        continue;
                    };
                    cb(self);
                    progress = true;
                    // Put the callback back unless the timer was freed while it ran.
                    if let Some(slot @ None) = self.list(clock).callback_mut(id) {
                        *slot = Some(cb);
                    }
                }
            }
        }
        progress
    }

    fn dispatch_inbox(&mut self) -> bool {
        let mut progress = false;
        loop {
            let (bhs, posted, ready, sleeps) = {
                let mut inbox = self.shared.inbox();
                (
                    std::mem::take(&mut inbox.bhs),
                    std::mem::take(&mut inbox.posted),
                    std::mem::take(&mut inbox.ready_tasks),
                    std::mem::take(&mut inbox.sleeps),
                )
            };
            if bhs.is_empty() && posted.is_empty() && ready.is_empty() && sleeps.is_empty() {
                return progress;
            }
            for bh in bhs {
                progress |= bh.run();
            }
            for f in posted {
                f(self);
                progress = true;
            }
            for (clock, deadline, waker) in sleeps {
                let t = self.timer_new(clock, move |_| waker.wake_by_ref());
                self.timer_mod(t, deadline);
                self.one_shot(t);
            }
            for idx in ready {
                progress |= self.poll_task(idx);
            }
        }
    }

    // Sleep timers free themselves after they fire. They are rare enough that a small wrapper is
    // fine, and it keeps the timer slab from growing without bound.
    fn one_shot(&mut self, t: Timer) {
        let list = self.list(t.clock);
        if let Some(Some(cb)) = list.callback_mut(t.id).map(Option::take) {
            let mut cb = Some(cb);
            *list.callback_mut(t.id).expect("just taken") =
                Some(Box::new(move |r: &mut Reactor| {
                    if let Some(mut f) = cb.take() {
                        f(r);
                    }
                    r.timer_free(t);
                }));
        }
    }

    fn poll_task(&mut self, idx: usize) -> bool {
        let Some(slot) = self.tasks.get_mut(idx) else { return false };
        // A task that wakes itself while it runs is already back in the inbox, so it gets polled
        // again on the next pass.
        let TaskSlot::Live(mut fut) = std::mem::replace(slot, TaskSlot::Running) else {
            return false;
        };
        let waker = Waker::from(Arc::new(TaskWaker { id: idx, shared: Arc::clone(&self.shared) }));
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                self.tasks[idx] = TaskSlot::Free;
                self.free_tasks.push(idx);
            }
            Poll::Pending => self.tasks[idx] = TaskSlot::Live(fut),
        }
        true
    }

    fn dispatch_events(&mut self) -> bool {
        let mut ready = Vec::with_capacity(self.events.iter().count());
        for ev in self.events.iter() {
            if ev.token() == WAKE_TOKEN {
                continue;
            }
            ready.push((
                ev.token().0,
                Ready {
                    readable: ev.is_readable(),
                    writable: ev.is_writable(),
                    closed: ev.is_read_closed() || ev.is_write_closed() || ev.is_error(),
                },
            ));
        }
        let progress = !ready.is_empty();
        for (idx, r) in ready {
            let Some(slot) = self.fds.get_mut(idx) else { continue };
            let Slot::Live(mut h) = std::mem::replace(slot, Slot::Running { removed: false })
            else {
                // Freed by an earlier handler in this batch.
                if matches!(slot, Slot::Running { .. }) {
                    *slot = Slot::Free;
                }
                continue;
            };
            h.call(self, r);
            match &self.fds[idx] {
                Slot::Running { removed: true } => {
                    let _ = self.poll.registry().deregister(h.source());
                    self.fds[idx] = Slot::Free;
                    self.free_fds.push(idx);
                }
                _ => self.fds[idx] = Slot::Live(h),
            }
        }
        progress
    }

    fn run_poll_fns(&mut self) -> bool {
        let mut found = false;
        for i in 0..self.poll_fns.len() {
            if let Some(mut f) = self.poll_fns[i].take() {
                found |= f(self);
                if self.poll_fns[i].is_none() {
                    self.poll_fns[i] = Some(f);
                }
            }
        }
        found
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        self.clock.detach(&self.handle());
    }
}

struct CurrentGuard {
    prev_clock: Option<Clock>,
}

impl Drop for CurrentGuard {
    fn drop(&mut self) {
        CURRENT.with(|c| c.borrow_mut().pop());
        let prev = self.prev_clock.take();
        CURRENT_CLOCK.with(|c| *c.borrow_mut() = prev);
    }
}

fn clock_index(c: ClockType) -> usize {
    match c {
        ClockType::Realtime => 0,
        ClockType::Virtual => 1,
        ClockType::Host => 2,
        ClockType::VirtualRt => 3,
    }
}

/// A single value sent from any thread to a task, used by [`crate::ThreadPool::run`].
pub(crate) fn oneshot<T>() -> (OneshotTx<T>, OneshotRx<T>) {
    let inner = Arc::new(Mutex::new((None, None::<Waker>)));
    (OneshotTx { inner: Arc::clone(&inner) }, OneshotRx { inner })
}

type OneshotInner<T> = Arc<Mutex<(Option<T>, Option<Waker>)>>;

pub(crate) struct OneshotTx<T> {
    inner: OneshotInner<T>,
}

impl<T> OneshotTx<T> {
    pub(crate) fn send(self, v: T) {
        let waker = {
            let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
            g.0 = Some(v);
            g.1.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

pub(crate) struct OneshotRx<T> {
    inner: OneshotInner<T>,
}

impl<T> Future for OneshotRx<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        match g.0.take() {
            Some(v) => Poll::Ready(v),
            None => {
                g.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::io::{Read, Write};
    use std::rc::Rc;
    use std::sync::atomic::AtomicUsize;

    fn reactor() -> Reactor {
        Reactor::new(Clock::new()).unwrap()
    }

    #[test]
    fn timers_fire_in_order_and_can_rearm() {
        let mut r = reactor();
        let log = Rc::new(RefCell::new(Vec::new()));
        let now = r.now(ClockType::Realtime);
        for (name, delay) in [("b", 2_000_000), ("a", 1_000_000), ("c", 3_000_000)] {
            let log = Rc::clone(&log);
            let t = r.timer_new(ClockType::Realtime, move |_| log.borrow_mut().push(name));
            r.timer_mod(t, now + delay);
        }
        let count = Rc::new(Cell::new(0));
        let c = Rc::clone(&count);
        let rearm = r.timer_new(ClockType::Realtime, move |r| {
            c.set(c.get() + 1);
            if c.get() < 3 {
                let at = r.now(ClockType::Realtime) + 100_000;
                r.timer_mod(rearm_of(r), at);
            }
        });
        REARM.with(|x| x.set(Some(rearm)));
        r.timer_mod(rearm, now);
        r.run_until(|_| log.borrow().len() == 3 && count.get() == 3).unwrap();
        assert_eq!(*log.borrow(), ["a", "b", "c"]);
    }

    thread_local! {
        static REARM: Cell<Option<Timer>> = const { Cell::new(None) };
    }

    fn rearm_of(_: &Reactor) -> Timer {
        REARM.with(|x| x.get().unwrap())
    }

    #[test]
    fn virtual_timers_wait_for_the_vm_to_run() {
        let mut r = reactor();
        let fired = Rc::new(Cell::new(false));
        let f = Rc::clone(&fired);
        let t = r.timer_new(ClockType::Virtual, move |_| f.set(true));
        r.timer_mod(t, 1);
        r.run_once(false).unwrap();
        assert!(!fired.get());
        let clock = r.clock().clone();
        std::thread::spawn(move || clock.start());
        r.run_until(|_| fired.get()).unwrap();
    }

    #[test]
    fn bottom_halves_coalesce_and_run_from_other_threads() {
        let mut r = reactor();
        let runs = Arc::new(AtomicUsize::new(0));
        let n = Arc::clone(&runs);
        let bh = r.bh(move || {
            n.fetch_add(1, Ordering::SeqCst);
        });
        bh.schedule();
        bh.schedule();
        r.run_once(false).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 1);

        let remote = bh.clone();
        std::thread::spawn(move || remote.schedule()).join().unwrap();
        r.run_until(|_| runs.load(Ordering::SeqCst) == 2).unwrap();

        bh.schedule();
        bh.cancel();
        r.run_once(false).unwrap();
        bh.delete();
        bh.schedule();
        r.run_once(false).unwrap();
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn event_notifiers_run_once_per_set() {
        let mut r = reactor();
        let seen = Arc::new(AtomicUsize::new(0));
        let s = Arc::clone(&seen);
        let en = r.handle().event_notifier(move |st| {
            if st.test_and_clear() {
                s.fetch_add(1, Ordering::SeqCst);
            }
        });
        en.set();
        en.set();
        r.run_once(false).unwrap();
        assert_eq!(seen.load(Ordering::SeqCst), 1);
        assert!(!en.is_set());
    }

    #[test]
    fn posted_closures_and_stop() {
        let mut r = reactor();
        let h = r.handle();
        let t = std::thread::spawn(move || {
            h.post(|r| r.handle().stop());
        });
        r.run().unwrap();
        t.join().unwrap();
    }

    #[test]
    fn sockets_are_dispatched_to_their_handler() {
        let mut r = reactor();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        listener.set_nonblocking(true).unwrap();
        let listener = mio::net::TcpListener::from_std(listener);
        let got = Rc::new(RefCell::new(Vec::new()));
        let g = Rc::clone(&got);
        r.add_source(listener, Interest::READABLE, move |r, l, _| {
            while let Ok((stream, _)) = l.accept() {
                let g = Rc::clone(&g);
                r.add_source(stream, Interest::READABLE, move |r, s, _| {
                    let mut buf = [0u8; 64];
                    loop {
                        match s.read(&mut buf) {
                            Ok(0) => {
                                r.handle().stop();
                                return;
                            }
                            Ok(n) => g.borrow_mut().extend_from_slice(&buf[..n]),
                            Err(_) => return,
                        }
                    }
                })
                .unwrap();
            }
        })
        .unwrap();
        let client = std::thread::spawn(move || {
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            s.write_all(b"{\"execute\": \"qmp_capabilities\"}").unwrap();
        });
        r.run().unwrap();
        client.join().unwrap();
        assert_eq!(got.borrow().as_slice(), b"{\"execute\": \"qmp_capabilities\"}");
    }

    #[test]
    fn tasks_sleep_and_use_the_pool() {
        let mut r = reactor();
        let pool = crate::ThreadPool::new(0, 4);
        let start = r.now(ClockType::Realtime);
        let out = r
            .block_on(async move {
                sleep(ClockType::Realtime, start + 2_000_000).await;
                pool.run(|| 6 * 7).await
            })
            .unwrap();
        assert_eq!(out, 42);
        assert!(r.now(ClockType::Realtime) >= start + 2_000_000);
    }

    #[test]
    fn the_pool_reports_back_on_the_reactor() {
        let mut r = reactor();
        let pool = crate::ThreadPool::new(0, 2);
        let done = Rc::new(Cell::new(0));
        for i in 0..10 {
            pool.submit(&r.handle(), move || i * 2, |_, v| DONE.with(|d| d.set(d.get() + v)));
        }
        let d = Rc::clone(&done);
        r.run_until(|_| {
            d.set(DONE.with(|x| x.get()));
            d.get() == 90
        })
        .unwrap();
        assert!(pool.threads() <= 2);
    }

    thread_local! {
        static DONE: Cell<i32> = const { Cell::new(0) };
    }

    #[test]
    fn busy_polling_finds_work_before_blocking() {
        let mut r = reactor();
        let hits = Rc::new(Cell::new(0));
        let h = Rc::clone(&hits);
        r.add_poll_fn(move |_| {
            h.set(h.get() + 1);
            true
        });
        r.set_polling(Duration::from_micros(50), Some(Box::new(|_, _| Duration::from_micros(20))));
        r.run_once(false).unwrap();
        assert_eq!(r.polling_time(), Duration::from_micros(20));
        r.run_once(false).unwrap();
        assert_eq!(hits.get(), 1);
    }
}
