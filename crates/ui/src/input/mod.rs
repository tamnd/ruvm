// SPDX-License-Identifier: GPL-2.0-or-later

//! The input layer, QEMU's ui/input.c: the devices that take keyboard and pointer input
//! register a handler, and the front ends and the QMP commands send events that go to the
//! first handler that takes their kind.
//!
//! An [`InputState`] holds the handler list, the notifiers for mouse mode and LED changes and
//! the keyboard queue that paces `send-key` and the VNC key delay on the virtual clock. There is
//! one for the process, [`InputState::global`]. Handlers, notifiers and the queue timer are
//! always called with the state's lock dropped, so a handler can call back in.
//!
//! Where this differs from QEMU:
//! - There is no record and replay, so events go straight to the handler.
//! - Without a virtual clock (see [`InputState::set_clock`]) the key delays are skipped and
//!   every key goes out at once.
//! - Text consoles do not exist, so no key goes to one.

mod keymap;
mod tables;

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Weak};

use ruvm_base::{Error, Result};
use ruvm_hw_core::{Clock, Timer};
use ruvm_qapi::types::{
    InputAxis, InputBtnEvent, InputButton, InputEventKind, InputEventU, InputMoveEvent,
    InputMultiTouchEvent, InputSendEventArg, MouseInfo, SendKeyArg,
};

use crate::console::{DisplayState, QemuConsole};

pub use keymap::{
    key_number_to_linux, key_number_to_qcode, key_value_to_linux, linux_to_qcode,
    linux_to_scancode, osx_to_linux, qcode_to_linux, usb_to_linux,
};

/// `INPUT_EVENT_MASK_KEY` and the other event masks, `1 << kind`.
pub const INPUT_EVENT_MASK_KEY: u32 = 1 << InputEventKind::Key as u32;
pub const INPUT_EVENT_MASK_BTN: u32 = 1 << InputEventKind::Btn as u32;
pub const INPUT_EVENT_MASK_REL: u32 = 1 << InputEventKind::Rel as u32;
pub const INPUT_EVENT_MASK_ABS: u32 = 1 << InputEventKind::Abs as u32;
pub const INPUT_EVENT_MASK_MTT: u32 = 1 << InputEventKind::Mtt as u32;

/// The range absolute events are scaled to.
pub const INPUT_EVENT_ABS_MIN: i64 = 0;
pub const INPUT_EVENT_ABS_MAX: i64 = 0x7FFF;

/// The LED bits of `leds_mask`.
pub const QEMU_SCROLL_LOCK_LED: u32 = 1 << 0;
pub const QEMU_NUM_LOCK_LED: u32 = 1 << 1;
pub const QEMU_CAPS_LOCK_LED: u32 = 1 << 2;

/// `kbd_default_delay_ms`.
const KBD_DEFAULT_DELAY_MS: u32 = 10;
/// `queue_limit`.
const QUEUE_LIMIT: usize = 1024;

/// `InputEvent` the way the input layer passes it on, `QemuInputEvent`: keys are Linux
/// keycodes.
#[derive(Clone, Debug, PartialEq)]
pub enum QemuInputEvent {
    Key { key: u32, down: bool },
    Btn(InputBtnEvent),
    Rel(InputMoveEvent),
    Abs(InputMoveEvent),
    Mtt(InputMultiTouchEvent),
}

impl QemuInputEvent {
    pub fn kind(&self) -> InputEventKind {
        match self {
            QemuInputEvent::Key { .. } => InputEventKind::Key,
            QemuInputEvent::Btn(_) => InputEventKind::Btn,
            QemuInputEvent::Rel(_) => InputEventKind::Rel,
            QemuInputEvent::Abs(_) => InputEventKind::Abs,
            QemuInputEvent::Mtt(_) => InputEventKind::Mtt,
        }
    }

    fn mask(&self) -> u32 {
        1 << self.kind() as u32
    }
}

/// `QemuInputHandler`: a device that takes input.
pub trait InputHandler: Send + Sync {
    /// The name `query-mice` shows.
    fn name(&self) -> &str;

    /// The `INPUT_EVENT_MASK_*` bits of the events it takes.
    fn mask(&self) -> u32;

    /// One event. `src` is the console it came from, if any.
    fn event(&self, src: Option<&QemuConsole>, evt: &QemuInputEvent);

    /// The end of a batch of events.
    fn sync(&self) {}
}

/// A registered handler, `QemuInputHandlerState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandlerId(i32);

/// A registered notifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NotifierId(u64);

type Notifier = Arc<dyn Fn() + Send + Sync>;

struct Entry {
    id: i32,
    handler: Arc<dyn InputHandler>,
    events: u32,
    con: Option<QemuConsole>,
    leds_mask: u32,
}

/// `QemuInputEventQueue`.
enum QueueItem {
    Delay(u32),
    Event(Option<QemuConsole>, QemuInputEvent),
    Sync,
}

struct Inner {
    handlers: Vec<Entry>,
    next_id: i32,
    next_notifier: u64,
    mouse_mode_notifiers: Vec<(u64, Notifier)>,
    leds_notifiers: Vec<(u64, Notifier)>,
    kbd_queue: VecDeque<QueueItem>,
    clock: Option<Arc<Clock>>,
    kbd_timer: Option<Timer>,
}

impl Inner {
    /// `qemu_input_find_handler()`: a handler bound to `con` first, then the first unbound one.
    fn find_handler(&self, mask: u32, con: Option<&QemuConsole>) -> Option<usize> {
        let bound = self.handlers.iter().position(|s| {
            matches!((&s.con, con), (Some(a), Some(b)) if a.ptr_eq(b))
                && (mask & s.handler.mask()) != 0
        });
        bound.or_else(|| {
            self.handlers.iter().position(|s| s.con.is_none() && (mask & s.handler.mask()) != 0)
        })
    }

    fn position(&self, id: HandlerId) -> Option<usize> {
        self.handlers.iter().position(|s| s.id == id.0)
    }

    /// The notifiers `notify_input_changed()` calls.
    fn changed_notifiers(&self, mask: u32) -> Vec<Notifier> {
        let mut v: Vec<Notifier> =
            self.mouse_mode_notifiers.iter().map(|(_, n)| Arc::clone(n)).collect();
        if (mask & INPUT_EVENT_MASK_KEY) != 0 {
            v.extend(self.leds_notifiers.iter().map(|(_, n)| Arc::clone(n)));
        }
        v
    }

    fn leds(&self) -> Vec<Notifier> {
        self.leds_notifiers.iter().map(|(_, n)| Arc::clone(n)).collect()
    }

    fn arm_kbd_timer(&self, delay_ms: u32) {
        if let (Some(clock), Some(timer)) = (&self.clock, &self.kbd_timer) {
            timer.modify((clock.get_ms() + i64::from(delay_ms)) * 1_000_000);
        }
    }
}

type RunCheck = Arc<dyn Fn() -> bool + Send + Sync>;

/// The handler list and the keyboard queue of ui/input.c.
pub struct InputState {
    me: Weak<InputState>,
    inner: Mutex<Inner>,
    running: Mutex<Option<RunCheck>>,
}

impl std::fmt::Debug for InputState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InputState").finish_non_exhaustive()
    }
}

static GLOBAL: LazyLock<Arc<InputState>> = LazyLock::new(InputState::new);

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn notify(list: Vec<Notifier>) {
    for n in list {
        n();
    }
}

impl InputState {
    /// A fresh state with no handlers, for a test.
    pub fn new() -> Arc<InputState> {
        Arc::new_cyclic(|me| InputState {
            me: me.clone(),
            inner: Mutex::new(Inner {
                handlers: Vec::new(),
                next_id: 1,
                next_notifier: 1,
                mouse_mode_notifiers: Vec::new(),
                leds_notifiers: Vec::new(),
                kbd_queue: VecDeque::new(),
                clock: None,
                kbd_timer: None,
            }),
            running: Mutex::new(None),
        })
    }

    /// The state of the process.
    pub fn global() -> Arc<InputState> {
        Arc::clone(&GLOBAL)
    }

    /// Sets the check for `runstate_is_running() || runstate_check(RUN_STATE_SUSPENDED)`. Until
    /// it is set the machine counts as running.
    pub fn set_runstate_check(&self, check: impl Fn() -> bool + Send + Sync + 'static) {
        *lock(&self.running) = Some(Arc::new(check));
    }

    /// Sets `QEMU_CLOCK_VIRTUAL`, which the keyboard queue runs on.
    pub fn set_clock(&self, clock: &Arc<Clock>) {
        let mut inner = lock(&self.inner);
        if let Some(t) = inner.kbd_timer.take() {
            t.del();
        }
        let me = self.me.clone();
        inner.kbd_timer = Some(clock.new_timer(move || {
            if let Some(s) = me.upgrade() {
                s.queue_process();
            }
        }));
        inner.clock = Some(Arc::clone(clock));
    }

    fn running(&self) -> bool {
        let check = lock(&self.running).clone();
        check.is_none_or(|f| f())
    }

    /// `qemu_input_handler_register()`.
    pub fn register(&self, handler: Arc<dyn InputHandler>) -> HandlerId {
        let mask = handler.mask();
        let (id, list) = {
            let mut inner = lock(&self.inner);
            let id = inner.next_id;
            inner.next_id += 1;
            inner.handlers.push(Entry { id, handler, events: 0, con: None, leds_mask: 0 });
            (id, inner.changed_notifiers(mask))
        };
        notify(list);
        HandlerId(id)
    }

    /// Moves the handler to the head or the tail of the list.
    fn move_handler(&self, id: HandlerId, head: bool) {
        let list = {
            let mut inner = lock(&self.inner);
            let Some(i) = inner.position(id) else { return };
            let e = inner.handlers.remove(i);
            let mask = e.handler.mask();
            if head {
                inner.handlers.insert(0, e);
            } else {
                inner.handlers.push(e);
            }
            inner.changed_notifiers(mask)
        };
        notify(list);
    }

    /// `qemu_input_handler_activate()`.
    pub fn activate(&self, id: HandlerId) {
        self.move_handler(id, true);
    }

    /// `qemu_input_handler_deactivate()`.
    pub fn deactivate(&self, id: HandlerId) {
        self.move_handler(id, false);
    }

    /// `qemu_input_handler_unregister()`.
    pub fn unregister(&self, id: HandlerId) {
        let list = {
            let mut inner = lock(&self.inner);
            let Some(i) = inner.position(id) else { return };
            let e = inner.handlers.remove(i);
            inner.changed_notifiers(e.handler.mask())
        };
        notify(list);
    }

    /// `qemu_input_handler_bind()`.
    pub fn bind(&self, id: HandlerId, ds: &DisplayState, device_id: &str, head: u32) -> Result<()> {
        let con = ds.lookup_by_device_name(device_id, head)?;
        let mut inner = lock(&self.inner);
        if let Some(i) = inner.position(id) {
            inner.handlers[i].con = Some(con);
        }
        Ok(())
    }

    /// `qemu_input_handler_set_leds_mask()`.
    pub fn set_leds_mask(&self, id: HandlerId, leds_mask: u32) {
        let list = {
            let mut inner = lock(&self.inner);
            let Some(i) = inner.position(id) else { return };
            assert!((inner.handlers[i].handler.mask() & INPUT_EVENT_MASK_KEY) != 0);
            inner.handlers[i].leds_mask = leds_mask;
            inner.leds()
        };
        notify(list);
    }

    fn add_notifier(&self, leds: bool, f: impl Fn() + Send + Sync + 'static) -> NotifierId {
        let mut inner = lock(&self.inner);
        let id = inner.next_notifier;
        inner.next_notifier += 1;
        let list = if leds { &mut inner.leds_notifiers } else { &mut inner.mouse_mode_notifiers };
        list.push((id, Arc::new(f)));
        NotifierId(id)
    }

    /// `qemu_input_led_notifier_add()`.
    pub fn add_led_notifier(&self, f: impl Fn() + Send + Sync + 'static) -> NotifierId {
        self.add_notifier(true, f)
    }

    /// `qemu_add_mouse_mode_change_notifier()`.
    pub fn add_mouse_mode_notifier(&self, f: impl Fn() + Send + Sync + 'static) -> NotifierId {
        self.add_notifier(false, f)
    }

    /// `qemu_input_led_notifier_remove()` and `qemu_remove_mouse_mode_change_notifier()`.
    pub fn remove_notifier(&self, id: NotifierId) {
        let mut inner = lock(&self.inner);
        inner.leds_notifiers.retain(|(n, _)| *n != id.0);
        inner.mouse_mode_notifiers.retain(|(n, _)| *n != id.0);
    }

    /// `qemu_input_event_send_impl()`.
    fn send_impl(&self, src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        let handler = {
            let mut inner = lock(&self.inner);
            let Some(i) = inner.find_handler(evt.mask(), src) else { return };
            inner.handlers[i].events += 1;
            Arc::clone(&inner.handlers[i].handler)
        };
        handler.event(src, evt);
    }

    /// `qemu_input_event_send()`.
    pub fn event_send(&self, src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        if matches!(evt, QemuInputEvent::Key { key: 0, .. }) {
            return;
        }
        if !self.running() {
            return;
        }
        self.send_impl(src, evt);
    }

    /// `qemu_input_event_sync()`: the handlers that got events since the last sync get theirs.
    pub fn event_sync(&self) {
        if !self.running() {
            return;
        }
        let list: Vec<Arc<dyn InputHandler>> = {
            let mut inner = lock(&self.inner);
            inner
                .handlers
                .iter_mut()
                .filter(|s| s.events != 0)
                .map(|s| {
                    s.events = 0;
                    Arc::clone(&s.handler)
                })
                .collect()
        };
        for h in list {
            h.sync();
        }
    }

    /// `qemu_input_event_send_key_linux()`: at once when the keyboard queue is empty, otherwise
    /// behind what is queued.
    pub fn send_key_linux(&self, src: Option<&QemuConsole>, lnx: u32, down: bool) {
        let evt = QemuInputEvent::Key { key: lnx, down };
        {
            let mut inner = lock(&self.inner);
            if !inner.kbd_queue.is_empty() {
                if inner.kbd_queue.len() < QUEUE_LIMIT {
                    inner.kbd_queue.push_back(QueueItem::Event(src.cloned(), evt));
                    inner.kbd_queue.push_back(QueueItem::Sync);
                }
                return;
            }
        }
        self.event_send(src, &evt);
        self.event_sync();
    }

    /// `qemu_input_event_send_key_number()`.
    pub fn send_key_number(&self, src: Option<&QemuConsole>, num: i64, down: bool) {
        self.send_key_linux(src, key_number_to_linux(num), down);
    }

    /// `qemu_input_event_send_key_delay()`: the keys after this wait `delay_ms`, or the default
    /// 10 ms for zero.
    pub fn send_key_delay(&self, delay_ms: u32) {
        if !self.running() {
            return;
        }
        let mut inner = lock(&self.inner);
        if inner.kbd_timer.is_none() || inner.kbd_queue.len() >= QUEUE_LIMIT {
            return;
        }
        let delay = if delay_ms != 0 { delay_ms } else { KBD_DEFAULT_DELAY_MS };
        let start = inner.kbd_queue.is_empty();
        inner.kbd_queue.push_back(QueueItem::Delay(delay));
        if start {
            inner.arm_kbd_timer(delay);
        }
    }

    /// `qemu_input_queue_process()`: the delay at the head is over, so send what follows it up
    /// to the next delay. An item stays queued while it is sent, so a key sent meanwhile waits.
    fn queue_process(&self) {
        {
            let mut inner = lock(&self.inner);
            if !matches!(inner.kbd_queue.front(), Some(QueueItem::Delay(_))) {
                return;
            }
            inner.kbd_queue.pop_front();
        }
        loop {
            let item = {
                let inner = lock(&self.inner);
                match inner.kbd_queue.front() {
                    None => return,
                    Some(QueueItem::Delay(ms)) => {
                        inner.arm_kbd_timer(*ms);
                        return;
                    }
                    Some(QueueItem::Event(src, evt)) => Some((src.clone(), evt.clone())),
                    Some(QueueItem::Sync) => None,
                }
            };
            match item {
                Some((src, evt)) => self.event_send(src.as_ref(), &evt),
                None => self.event_sync(),
            }
            lock(&self.inner).kbd_queue.pop_front();
        }
    }

    /// `qemu_input_queue_btn()`.
    pub fn queue_btn(&self, src: Option<&QemuConsole>, button: InputButton, down: bool) {
        self.event_send(src, &QemuInputEvent::Btn(InputBtnEvent { button, down }));
    }

    /// `qemu_input_update_buttons()`: a button event for each button of `button_map` whose bit
    /// differs between `old` and `new`.
    pub fn update_buttons(
        &self,
        src: Option<&QemuConsole>,
        button_map: &[(InputButton, u32)],
        old: u32,
        new: u32,
    ) {
        for &(btn, mask) in button_map {
            if (old & mask) != (new & mask) {
                self.queue_btn(src, btn, (new & mask) != 0);
            }
        }
    }

    /// `qemu_input_queue_rel()`.
    pub fn queue_rel(&self, src: Option<&QemuConsole>, axis: InputAxis, value: i64) {
        self.event_send(src, &QemuInputEvent::Rel(InputMoveEvent { axis, value }));
    }

    /// `qemu_input_queue_abs()`: `value` in `min_in..=max_in` scaled to the absolute range.
    pub fn queue_abs(
        &self,
        src: Option<&QemuConsole>,
        axis: InputAxis,
        value: i32,
        min_in: i32,
        max_in: i32,
    ) {
        let value = i64::from(scale_axis(
            value,
            min_in,
            max_in,
            INPUT_EVENT_ABS_MIN as i32,
            INPUT_EVENT_ABS_MAX as i32,
        ));
        self.event_send(src, &QemuInputEvent::Abs(InputMoveEvent { axis, value }));
    }

    /// `qemu_input_is_absolute()`.
    pub fn is_absolute(&self, con: Option<&QemuConsole>) -> bool {
        let inner = lock(&self.inner);
        inner
            .find_handler(INPUT_EVENT_MASK_REL | INPUT_EVENT_MASK_ABS, con)
            .is_some_and(|i| (inner.handlers[i].handler.mask() & INPUT_EVENT_MASK_ABS) != 0)
    }

    /// `qemu_input_get_leds_mask()`.
    pub fn get_leds_mask(&self, con: Option<&QemuConsole>) -> u32 {
        let inner = lock(&self.inner);
        inner.find_handler(INPUT_EVENT_MASK_KEY, con).map_or(0, |i| inner.handlers[i].leds_mask)
    }

    /// `qmp_query_mice()`. QEMU prepends each mouse, so the last handler comes first.
    pub fn query_mice(&self) -> Vec<MouseInfo> {
        let inner = lock(&self.inner);
        let mut list = Vec::new();
        let mut current = true;
        for s in &inner.handlers {
            let mask = s.handler.mask();
            if (mask & (INPUT_EVENT_MASK_REL | INPUT_EVENT_MASK_ABS)) == 0 {
                continue;
            }
            list.push(MouseInfo {
                name: s.handler.name().to_string(),
                index: i64::from(s.id),
                current,
                absolute: (mask & INPUT_EVENT_MASK_ABS) != 0,
            });
            current = false;
        }
        list.reverse();
        list
    }

    /// `qemu_mouse_set()`, the HMP `mouse_set` command.
    pub fn mouse_set(&self, index: i32) -> Result<()> {
        let id = {
            let inner = lock(&self.inner);
            let Some(s) = inner.handlers.iter().find(|s| s.id == index) else {
                return Err(Error::generic(format!("Mouse at index '{index}' not found")));
            };
            if (s.handler.mask() & (INPUT_EVENT_MASK_REL | INPUT_EVENT_MASK_ABS)) == 0 {
                return Err(Error::generic(format!(
                    "Input device '{}' is not a mouse",
                    s.handler.name()
                )));
            }
            HandlerId(s.id)
        };
        self.activate(id);
        Ok(())
    }

    /// `qmp_input_send_event()`.
    pub fn qmp_input_send_event(&self, ds: &DisplayState, arg: InputSendEventArg) -> Result<()> {
        let con = match &arg.device {
            Some(device) => Some(ds.lookup_by_device_name(device, arg.head.unwrap_or(0) as u32)?),
            None => None,
        };
        if !self.running() {
            return Err(Error::generic("VM not running"));
        }
        {
            let inner = lock(&self.inner);
            for e in &arg.events {
                let kind = e.u.tag();
                if inner.find_handler(1 << kind as u32, con.as_ref()).is_none() {
                    return Err(Error::generic(format!(
                        "Input handler not found for event type {}",
                        kind.as_str()
                    )));
                }
            }
        }
        for e in arg.events {
            let evt = match e.u {
                InputEventU::Key(k) => {
                    QemuInputEvent::Key { key: key_value_to_linux(&k.data.key), down: k.data.down }
                }
                InputEventU::Btn(b) => QemuInputEvent::Btn(b.data),
                InputEventU::Rel(m) => QemuInputEvent::Rel(m.data),
                InputEventU::Abs(m) => QemuInputEvent::Abs(m.data),
                InputEventU::Mtt(m) => QemuInputEvent::Mtt(m.data),
            };
            self.event_send(con.as_ref(), &evt);
        }
        self.event_sync();
        Ok(())
    }

    /// `qmp_send_key()`: each key down with a `hold-time` delay after it, then each key up in
    /// the reverse order, also with a delay.
    pub fn qmp_send_key(&self, arg: SendKeyArg) -> Result<()> {
        // The C code passes the int64 on as a uint32_t.
        let hold_time = arg.hold_time.unwrap_or(0) as u32;
        let mut up = Vec::with_capacity(arg.keys.len());
        for k in &arg.keys {
            let lnx = key_value_to_linux(k);
            up.push(lnx);
            self.send_key_linux(None, lnx, true);
            self.send_key_delay(hold_time);
        }
        for &lnx in up.iter().rev() {
            self.send_key_linux(None, lnx, false);
            self.send_key_delay(hold_time);
        }
        Ok(())
    }
}

/// `qemu_input_scale_axis()`.
pub fn scale_axis(value: i32, min_in: i32, max_in: i32, min_out: i32, max_out: i32) -> i32 {
    let range_in = i64::from(max_in) - i64::from(min_in);
    let range_out = i64::from(max_out) - i64::from(min_out);
    if range_in < 1 {
        return (i64::from(min_out) + range_out / 2) as i32;
    }
    ((i64::from(value) - i64::from(min_in)) * range_out / range_in + i64::from(min_out)) as i32
}

#[cfg(test)]
mod tests;
