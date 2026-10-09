// SPDX-License-Identifier: GPL-2.0-or-later

//! The console core, from QEMU's `ui/console.c`: `QemuConsole`, `DisplayState`,
//! `GraphicHwOps` and `DisplayChangeListener`.
//!
//! A graphic device creates a console with [`DisplayState::graphic_console_create`] and hands it
//! a [`GraphicHwOps`]. The device then reports what it shows through the console:
//! [`QemuConsole::set_surface`] when the mode changes, [`QemuConsole::update`] for the rectangles
//! that changed, and [`QemuConsole::resize`] for the common case of a fresh `x8r8g8b8` surface.
//! The UIs register a [`DisplayChangeListener`] on a console and get those calls passed on.
//!
//! [`DisplayState::global`] is the process wide state QEMU keeps in static variables. Tests
//! build their own with [`DisplayState::new`].
//!
//! Differences from QEMU:
//! - There are no text consoles yet (`console-vc.c`), so the "graphic consoles first" ordering of
//!   `qemu_console_register()` has nothing to reorder, and a console's label falls back to
//!   `vcN` only for a graphic console without a device, which QEMU labels "VGA" as here.
//! - Consoles are not QOM objects. A console names its device by the device's `id` and type
//!   name, and [`DisplayState::lookup_by_device_name`] searches the consoles rather than the
//!   qdev tree, so an `id` that exists but has no console is reported as not found.
//! - There is no OpenGL scanout, so a console's scanout is always its surface.
//! - Listener callbacks run on the thread that made the change, without a big lock around them.
//!   A listener must not call back into the device from `gfx_update` or `gfx_switch`.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use ruvm_base::{Error, ErrorClass, Result};

use crate::pixman::{PixelFormat, default_pixman_format};
use crate::surface::DisplaySurface;

/// `GUI_REFRESH_INTERVAL_DEFAULT`, in milliseconds.
pub const GUI_REFRESH_INTERVAL_DEFAULT: u64 = 30;
/// `GUI_REFRESH_INTERVAL_IDLE`, in milliseconds.
pub const GUI_REFRESH_INTERVAL_IDLE: u64 = 3000;

const NOINIT: &str = "Guest has not initialized the display (yet).";
const NOT_ACTIVE: &str = "Display output is not active.";
const UNPLUGGED: &str = "Guest display has been unplugged";

/// `GraphicHwOps`: what the console asks of the device behind it.
pub trait GraphicHwOps: Send + Sync {
    /// `invalidate`: redraw everything on the next update.
    fn invalidate(&self) {}

    /// `gfx_update`: bring the surface up to date. Returns false when the update finishes later,
    /// in which case the device calls [`QemuConsole::hw_update_done`] once it has.
    fn gfx_update(&self, con: &QemuConsole) -> bool {
        let _ = con;
        true
    }

    /// Whether the device has a `gfx_update` callback at all.
    fn has_gfx_update(&self) -> bool {
        true
    }

    /// `text_update`: fills `chardata` with the text screen, for devices in a text mode.
    fn text_update(&self, chardata: &mut [u32]) {
        let _ = chardata;
    }

    /// `ui_info`: the UI's window size and position changed.
    fn ui_info(&self, head: u32, info: &QemuUiInfo) {
        let _ = (head, info);
    }

    /// Whether the device has a `ui_info` callback.
    fn has_ui_info(&self) -> bool {
        false
    }
}

/// `QemuUIInfo`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QemuUiInfo {
    pub width_mm: u16,
    pub height_mm: u16,
    pub xoff: i32,
    pub yoff: i32,
    pub width: u32,
    pub height: u32,
    pub refresh_rate: u32,
}

/// `DisplayChangeListenerOps`, the half a UI implements.
pub trait DisplayChangeListener: Send + Sync {
    /// `dpy_name`.
    fn name(&self) -> &str;

    /// Whether the listener has a `dpy_refresh` callback, which keeps the refresh timer running.
    fn has_refresh(&self) -> bool {
        false
    }

    /// `dpy_refresh`: called every refresh interval.
    fn refresh(&self, con: &QemuConsole) {
        let _ = con;
    }

    /// `dpy_gfx_update`: the rectangle changed. Read the pixels with [`QemuConsole::with_surface`].
    fn gfx_update(&self, con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        let _ = (con, x, y, w, h);
    }

    /// `dpy_gfx_switch`: the console has a new surface.
    fn gfx_switch(&self, con: &QemuConsole) {
        let _ = con;
    }

    /// `dpy_gfx_check_format`: None when the listener has no such callback, in which case only
    /// native 32 bpp is accepted.
    fn gfx_check_format(&self, format: PixelFormat) -> Option<bool> {
        let _ = format;
        None
    }

    /// `dpy_mouse_set`.
    fn mouse_set(&self, x: i32, y: i32, on: bool) {
        let _ = (x, y, on);
    }
}

/// The device a graphic console belongs to: its `id` and its type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsoleDevice {
    pub id: Option<String>,
    pub typename: String,
}

/// A registered listener, as returned by [`DisplayState::register_listener`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ListenerId(u64);

struct Listener {
    id: ListenerId,
    con: QemuConsole,
    ops: Arc<dyn DisplayChangeListener>,
    update_interval: u64,
}

struct DsInner {
    consoles: Vec<QemuConsole>,
    listeners: Vec<Listener>,
    update_interval: u64,
    last_update: Option<Instant>,
    refresh: Option<Arc<RefreshThread>>,
    /// `DisplayState.refreshing`: a refresh pass is running, so a listener changing its
    /// interval does not rearm the timer, the end of the pass picks the change up.
    refreshing: bool,
}

/// `DisplayState` together with the console list, which QEMU keeps in file scope statics.
pub struct DisplayState {
    me: Weak<DisplayState>,
    inner: Mutex<DsInner>,
    next_listener: AtomicU64,
}

impl fmt::Debug for DisplayState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = lock(&self.inner);
        f.debug_struct("DisplayState")
            .field("consoles", &inner.consoles.len())
            .field("listeners", &inner.listeners.len())
            .finish()
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl DisplayState {
    /// A display state of its own, with no consoles.
    pub fn new() -> Arc<DisplayState> {
        Arc::new_cyclic(|me| DisplayState {
            me: me.clone(),
            inner: Mutex::new(DsInner {
                consoles: Vec::new(),
                listeners: Vec::new(),
                update_interval: 0,
                last_update: None,
                refresh: None,
                refreshing: false,
            }),
            next_listener: AtomicU64::new(1),
        })
    }

    /// `get_alloc_displaystate()`: the process wide display state.
    pub fn global() -> Arc<DisplayState> {
        static GLOBAL: OnceLock<Arc<DisplayState>> = OnceLock::new();
        Arc::clone(GLOBAL.get_or_init(DisplayState::new))
    }

    /// The consoles in index order.
    pub fn consoles(&self) -> Vec<QemuConsole> {
        lock(&self.inner).consoles.clone()
    }

    /// `qemu_graphic_console_create()`: a graphic console for head `head` of `dev`, showing the
    /// "not initialized" placeholder until the device sets a surface. A console left behind by an
    /// unplugged device is reused, keeping its index and size.
    pub fn graphic_console_create(
        &self,
        dev: Option<ConsoleDevice>,
        head: u32,
        hw_ops: Arc<dyn GraphicHwOps>,
    ) -> QemuConsole {
        let (con, width, height) = match self.lookup_unused() {
            Some(con) => {
                let w = con.width(0);
                let h = con.height(0);
                (con, w, h)
            }
            None => (self.register(), 640, 480),
        };
        {
            let mut st = lock(&con.inner.state);
            st.head = head;
            st.hw_ops = Some(hw_ops);
            st.device = dev;
            st.device_address.clear();
        }
        con.set_surface(Some(DisplaySurface::placeholder(width as usize, height as usize, NOINIT)));
        con
    }

    /// `qemu_console_register()` for a new graphic console: appended at the end, which is where
    /// QEMU puts it while there are no text consoles.
    fn register(&self) -> QemuConsole {
        let mut inner = lock(&self.inner);
        let index = inner.consoles.last().map_or(0, |c| c.index() + 1);
        let con = QemuConsole {
            inner: Arc::new(ConsoleInner {
                ds: self.me.clone(),
                state: Mutex::new(ConsoleState {
                    index,
                    head: 0,
                    device: None,
                    device_address: String::new(),
                    hw_ops: None,
                    surface: None,
                    ui_info: QemuUiInfo::default(),
                    cursor: (0, 0, false),
                }),
                dump: DumpQueue::default(),
            }),
        };
        inner.consoles.push(con.clone());
        con
    }

    /// `qemu_graphic_console_lookup_unused()`.
    fn lookup_unused(&self) -> Option<QemuConsole> {
        let inner = lock(&self.inner);
        inner
            .consoles
            .iter()
            .find(|c| {
                let st = lock(&c.inner.state);
                st.hw_ops.is_none() && st.device.is_none()
            })
            .cloned()
    }

    /// `qemu_console_lookup_default()`: the first graphic console.
    pub fn lookup_default(&self) -> Option<QemuConsole> {
        lock(&self.inner).consoles.first().cloned()
    }

    /// `qemu_console_lookup_by_index()`.
    pub fn lookup_by_index(&self, index: u32) -> Option<QemuConsole> {
        lock(&self.inner).consoles.iter().find(|c| c.index() == index).cloned()
    }

    /// `qemu_console_lookup_by_device_name()`.
    pub fn lookup_by_device_name(&self, device_id: &str, head: u32) -> Result<QemuConsole> {
        let consoles = self.consoles();
        let mut found = false;
        for con in consoles {
            let st = lock(&con.inner.state);
            if st.device.as_ref().and_then(|d| d.id.as_deref()) != Some(device_id) {
                continue;
            }
            found = true;
            if st.head == head {
                drop(st);
                return Ok(con);
            }
        }
        if !found {
            return Err(Error::new(
                ErrorClass::DeviceNotFound,
                format!("Device '{device_id}' not found"),
            ));
        }
        Err(Error::generic(format!(
            "Device {device_id} (head {head}) is not bound to a QemuConsole"
        )))
    }

    /// `qemu_console_register_listener()`: attaches `ops` to `con`, shows it the current surface
    /// and starts the refresh timer if the listener wants one.
    pub fn register_listener(
        &self,
        con: &QemuConsole,
        ops: Arc<dyn DisplayChangeListener>,
    ) -> ListenerId {
        let id = ListenerId(self.next_listener.fetch_add(1, Ordering::Relaxed));
        {
            let mut inner = lock(&self.inner);
            // QLIST_INSERT_HEAD
            inner.listeners.insert(
                0,
                Listener { id, con: con.clone(), ops: Arc::clone(&ops), update_interval: 0 },
            );
        }
        self.setup_refresh();
        // displaychangelistener_display_console(): switch to the console's surface and draw it.
        if con.surface_size().is_some() {
            ops.gfx_switch(con);
            let (w, h) = con.surface_size().unwrap_or((0, 0));
            ops.gfx_update(con, 0, 0, w as i32, h as i32);
        }
        let (x, y, on) = lock(&con.inner.state).cursor;
        ops.mouse_set(x, y, on);
        id
    }

    /// `qemu_console_unregister_listener()`.
    pub fn unregister_listener(&self, id: ListenerId) {
        lock(&self.inner).listeners.retain(|l| l.id != id);
        self.setup_refresh();
    }

    /// `qemu_console_listener_set_refresh()`: the listener's own refresh interval, in ms.
    pub fn listener_set_refresh(&self, id: ListenerId, interval: u64) {
        let refresh = {
            let mut inner = lock(&self.inner);
            if let Some(l) = inner.listeners.iter_mut().find(|l| l.id == id) {
                l.update_interval = interval;
            }
            if !inner.refreshing && inner.update_interval > interval {
                inner.refresh.clone()
            } else {
                None
            }
        };
        if let Some(r) = refresh {
            r.kick();
        }
    }

    fn listeners_of(&self, con: &QemuConsole) -> Vec<Arc<dyn DisplayChangeListener>> {
        lock(&self.inner)
            .listeners
            .iter()
            .filter(|l| l.con.ptr_eq(con))
            .map(|l| Arc::clone(&l.ops))
            .collect()
    }

    /// `gui_update()`: one refresh of every listener. Returns the interval until the next one.
    pub fn gui_update(&self) -> Duration {
        let listeners: Vec<(QemuConsole, Arc<dyn DisplayChangeListener>)> = {
            let mut inner = lock(&self.inner);
            inner.refreshing = true;
            inner.listeners.iter().map(|l| (l.con.clone(), Arc::clone(&l.ops))).collect()
        };
        for (con, ops) in &listeners {
            if ops.has_refresh() {
                ops.refresh(con);
            }
        }
        // The intervals as the refresh callbacks left them.
        let mut inner = lock(&self.inner);
        inner.refreshing = false;
        let mut interval = GUI_REFRESH_INTERVAL_IDLE;
        for l in &inner.listeners {
            let i = if l.update_interval != 0 {
                l.update_interval
            } else {
                GUI_REFRESH_INTERVAL_DEFAULT
            };
            interval = interval.min(i);
        }
        inner.update_interval = interval;
        inner.last_update = Some(Instant::now());
        Duration::from_millis(interval)
    }

    /// `gui_setup_refresh()`: runs the refresh timer while some listener has a refresh callback.
    fn setup_refresh(&self) {
        let mut inner = lock(&self.inner);
        let need = inner.listeners.iter().any(|l| l.ops.has_refresh());
        if need && inner.refresh.is_none() {
            inner.refresh = Some(RefreshThread::start(self.me.clone()));
        } else if !need {
            if let Some(r) = inner.refresh.take() {
                r.stop();
            }
        }
    }

    /// `qemu_console_check_format()` over the listeners of `con`.
    fn check_format(&self, con: &QemuConsole, format: PixelFormat) -> bool {
        for ops in self.listeners_of(con) {
            match ops.gfx_check_format(format) {
                Some(false) => return false,
                Some(true) => {}
                None => {
                    if Some(format) != default_pixman_format(32, true) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// The realtime clock timer behind `gui_update()`, a thread that sleeps between refreshes.
struct RefreshThread {
    state: Mutex<(bool, bool)>,
    cv: Condvar,
}

impl RefreshThread {
    fn start(ds: Weak<DisplayState>) -> Arc<RefreshThread> {
        let t = Arc::new(RefreshThread { state: Mutex::new((false, false)), cv: Condvar::new() });
        let me = Arc::clone(&t);
        let spawned = std::thread::Builder::new().name("gui-refresh".into()).spawn(move || {
            let mut wait = Duration::ZERO;
            loop {
                {
                    let mut st = lock(&me.state);
                    let deadline = Instant::now() + wait;
                    while !st.0 && !st.1 {
                        let now = Instant::now();
                        if now >= deadline {
                            break;
                        }
                        st = me
                            .cv
                            .wait_timeout(st, deadline - now)
                            .map(|(g, _)| g)
                            .unwrap_or_else(|e| e.into_inner().0);
                    }
                    if st.0 {
                        return;
                    }
                    st.1 = false;
                }
                let Some(ds) = ds.upgrade() else { return };
                wait = ds.gui_update();
            }
        });
        // Without a thread the UIs only update when something else asks, which is what QEMU does
        // too if it cannot arm the timer.
        drop(spawned);
        t
    }

    fn kick(&self) {
        lock(&self.state).1 = true;
        self.cv.notify_all();
    }

    fn stop(&self) {
        lock(&self.state).0 = true;
        self.cv.notify_all();
    }
}

#[derive(Default)]
struct DumpQueue {
    generation: Mutex<(u64, bool)>,
    cv: Condvar,
}

struct ConsoleState {
    index: u32,
    head: u32,
    device: Option<ConsoleDevice>,
    /// What `qemu_console_fill_device_address()` gives for the device, or empty.
    device_address: String,
    hw_ops: Option<Arc<dyn GraphicHwOps>>,
    surface: Option<DisplaySurface>,
    ui_info: QemuUiInfo,
    cursor: (i32, i32, bool),
}

struct ConsoleInner {
    ds: Weak<DisplayState>,
    state: Mutex<ConsoleState>,
    dump: DumpQueue,
}

/// `QemuConsole`, a graphic one. Cloning gives another handle to the same console.
#[derive(Clone)]
pub struct QemuConsole {
    inner: Arc<ConsoleInner>,
}

impl fmt::Debug for QemuConsole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QemuConsole")
            .field("index", &self.index())
            .field("label", &self.label())
            .finish_non_exhaustive()
    }
}

impl QemuConsole {
    /// Whether both are the same console.
    pub fn ptr_eq(&self, other: &QemuConsole) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn ds(&self) -> Option<Arc<DisplayState>> {
        self.inner.ds.upgrade()
    }

    /// `qemu_console_get_index()`.
    pub fn index(&self) -> u32 {
        lock(&self.inner.state).index
    }

    /// `qemu_console_get_head()`.
    pub fn head(&self) -> u32 {
        lock(&self.inner.state).head
    }

    /// The device behind the console.
    pub fn device(&self) -> Option<ConsoleDevice> {
        lock(&self.inner.state).device.clone()
    }

    /// `qemu_console_fill_device_address()`: `pci/0000/SS.F` for a PCI display function, with
    /// the bridges above it in front, or empty for any other device.
    pub fn device_address(&self) -> String {
        lock(&self.inner.state).device_address.clone()
    }

    /// Sets the address [`QemuConsole::device_address`] gives. The board does this once the
    /// device that took the console is plugged, since the console does not know the bus.
    pub fn set_device_address(&self, address: String) {
        lock(&self.inner.state).device_address = address;
    }

    /// `qemu_console_is_graphic()`: every console here is.
    pub fn is_graphic(&self) -> bool {
        true
    }

    /// `qemu_console_get_label()`: the device id or type name, with `.head` when the device has
    /// more than one console, or "VGA" without a device.
    pub fn label(&self) -> String {
        let (device, head) = {
            let st = lock(&self.inner.state);
            (st.device.clone(), st.head)
        };
        let Some(dev) = device else {
            return "VGA".to_string();
        };
        let name = dev.id.clone().unwrap_or_else(|| dev.typename.clone());
        let multihead = self.ds().is_some_and(|ds| {
            ds.consoles().iter().any(|c| {
                let st = lock(&c.inner.state);
                st.device.as_ref() == Some(&dev) && st.head != head
            })
        });
        if multihead { format!("{name}.{head}") } else { name }
    }

    /// `qemu_console_get_width()`.
    pub fn width(&self, fallback: i32) -> i32 {
        self.surface_size().map_or(fallback, |(w, _)| w as i32)
    }

    /// `qemu_console_get_height()`.
    pub fn height(&self, fallback: i32) -> i32 {
        self.surface_size().map_or(fallback, |(_, h)| h as i32)
    }

    fn surface_size(&self) -> Option<(usize, usize)> {
        lock(&self.inner.state).surface.as_ref().map(|s| (s.width(), s.height()))
    }

    /// `qemu_console_surface()` for reading: runs `f` on the current surface under the console's
    /// lock.
    pub fn with_surface<R>(&self, f: impl FnOnce(Option<&DisplaySurface>) -> R) -> R {
        f(lock(&self.inner.state).surface.as_ref())
    }

    /// `qemu_console_surface()` for the device to draw into.
    pub fn with_surface_mut<R>(&self, f: impl FnOnce(Option<&mut DisplaySurface>) -> R) -> R {
        f(lock(&self.inner.state).surface.as_mut())
    }

    /// `qemu_console_update()`: tells the listeners that the rectangle changed, clipped to the
    /// surface.
    pub fn update(&self, x: i32, y: i32, w: i32, h: i32) {
        let width = self.width(x + w);
        let height = self.height(y + h);
        let x = x.max(0).min(width);
        let y = y.max(0).min(height);
        let w = w.min(width - x);
        let h = h.min(height - y);
        let Some(ds) = self.ds() else { return };
        for ops in ds.listeners_of(self) {
            ops.gfx_update(self, x, y, w, h);
        }
    }

    /// `qemu_console_update_full()`.
    pub fn update_full(&self) {
        let w = self.width(0);
        let h = self.height(0);
        self.update(0, 0, w, h);
    }

    /// `qemu_console_set_surface()`. None puts up the "not active" placeholder at the old size.
    pub fn set_surface(&self, surface: Option<DisplaySurface>) {
        let placeholder = surface.is_none();
        let new = match surface {
            Some(s) => s,
            None => {
                let (w, h) = self.surface_size().unwrap_or((640, 480));
                DisplaySurface::placeholder(w, h, NOT_ACTIVE)
            }
        };
        let (w, h) = (new.width() as i32, new.height() as i32);
        lock(&self.inner.state).surface = Some(new);
        let Some(ds) = self.ds() else { return };
        for ops in ds.listeners_of(self) {
            ops.gfx_switch(self);
            if placeholder {
                ops.gfx_update(self, 0, 0, w, h);
            }
        }
    }

    /// `qemu_console_resize()`: a fresh `x8r8g8b8` surface of the new size, unless the console
    /// already has an allocated one of that size.
    pub fn resize(&self, width: usize, height: usize) {
        let keep = lock(&self.inner.state).surface.as_ref().is_some_and(|s| {
            s.is_allocated() && !s.is_placeholder() && s.width() == width && s.height() == height
        });
        if keep {
            return;
        }
        self.set_surface(Some(DisplaySurface::new(width, height)));
    }

    /// `qemu_console_check_format()`: whether every listener of the console takes `format`.
    pub fn check_format(&self, format: PixelFormat) -> bool {
        self.ds().is_none_or(|ds| ds.check_format(self, format))
    }

    fn hw_ops(&self) -> Option<Arc<dyn GraphicHwOps>> {
        lock(&self.inner.state).hw_ops.clone()
    }

    /// `qemu_console_hw_invalidate()`.
    pub fn hw_invalidate(&self) {
        if let Some(ops) = self.hw_ops() {
            ops.invalidate();
        }
    }

    /// `qemu_console_hw_text_update()`.
    pub fn hw_text_update(&self, chardata: &mut [u32]) {
        if let Some(ops) = self.hw_ops() {
            ops.text_update(chardata);
        }
    }

    /// `qemu_console_hw_update()` followed by the wait of `qemu_console_co_wait_update()`: asks
    /// the device for a fresh frame and returns once it is there.
    pub fn hw_update(&self) {
        let Some(ops) = self.hw_ops() else {
            self.hw_update_done();
            return;
        };
        let start = lock(&self.inner.dump.generation).0;
        if !ops.has_gfx_update() || ops.gfx_update(self) {
            self.hw_update_done();
            return;
        }
        let mut g = lock(&self.inner.dump.generation);
        while g.0 == start {
            g = self.inner.dump.cv.wait(g).unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    /// `graphic_hw_update()`: asks the device for a fresh frame without waiting for it.
    pub fn hw_update_nowait(&self) {
        let Some(ops) = self.hw_ops() else {
            self.hw_update_done();
            return;
        };
        if !ops.has_gfx_update() || ops.gfx_update(self) {
            self.hw_update_done();
        }
    }

    /// `qemu_console_hw_update_done()`: wakes everyone waiting in [`QemuConsole::hw_update`].
    pub fn hw_update_done(&self) {
        let mut g = lock(&self.inner.dump.generation);
        g.0 = g.0.wrapping_add(1);
        self.inner.dump.cv.notify_all();
    }

    /// `qemu_console_set_mouse()`.
    pub fn set_mouse(&self, x: i32, y: i32, on: bool) {
        lock(&self.inner.state).cursor = (x, y, on);
        let Some(ds) = self.ds() else { return };
        for ops in ds.listeners_of(self) {
            ops.mouse_set(x, y, on);
        }
    }

    /// `qemu_console_ui_info_supported()`.
    pub fn ui_info_supported(&self) -> bool {
        self.hw_ops().is_some_and(|o| o.has_ui_info())
    }

    /// `qemu_console_get_ui_info()`.
    pub fn ui_info(&self) -> QemuUiInfo {
        lock(&self.inner.state).ui_info
    }

    /// `qemu_console_set_ui_info()` without the delay timer: stores `info` and passes it to the
    /// device if it changed. Returns false, like QEMU's -1, when the device does not take UI info.
    pub fn set_ui_info(&self, info: QemuUiInfo) -> bool {
        if !self.ui_info_supported() {
            return false;
        }
        let changed = {
            let mut st = lock(&self.inner.state);
            let changed = st.ui_info != info;
            st.ui_info = info;
            changed
        };
        if changed {
            if let Some(ops) = self.hw_ops() {
                ops.ui_info(self.head(), &info);
            }
        }
        true
    }

    /// `qemu_graphic_console_close()`: the device is gone. The console stays, with the
    /// "unplugged" placeholder, and a later device may take it over.
    pub fn close(&self) {
        let w = self.width(640);
        let h = self.height(480);
        {
            let mut st = lock(&self.inner.state);
            st.device = None;
            st.device_address.clear();
            st.hw_ops = None;
        }
        self.set_surface(Some(DisplaySurface::placeholder(w as usize, h as usize, UNPLUGGED)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixman::{R5G6B5, X8R8G8B8};
    use std::sync::atomic::AtomicUsize;

    struct Hw(AtomicUsize);

    impl GraphicHwOps for Hw {
        fn gfx_update(&self, con: &QemuConsole) -> bool {
            self.0.fetch_add(1, Ordering::Relaxed);
            con.resize(320, 200);
            true
        }
    }

    #[derive(Default)]
    struct Recorder {
        log: Mutex<Vec<String>>,
        formats: Option<bool>,
    }

    impl DisplayChangeListener for Recorder {
        fn name(&self) -> &str {
            "recorder"
        }
        fn gfx_update(&self, _: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
            lock(&self.log).push(format!("update {x} {y} {w} {h}"));
        }
        fn gfx_switch(&self, con: &QemuConsole) {
            lock(&self.log).push(format!("switch {}x{}", con.width(0), con.height(0)));
        }
        fn gfx_check_format(&self, _: PixelFormat) -> Option<bool> {
            self.formats
        }
    }

    fn dev(id: Option<&str>) -> Option<ConsoleDevice> {
        Some(ConsoleDevice { id: id.map(String::from), typename: "VGA".into() })
    }

    #[test]
    fn a_new_console_shows_the_noinit_placeholder() {
        let ds = DisplayState::new();
        let con = ds.graphic_console_create(dev(None), 0, Arc::new(Hw(AtomicUsize::new(0))));
        assert_eq!(con.index(), 0);
        assert_eq!((con.width(0), con.height(0)), (640, 480));
        assert!(con.with_surface(|s| s.unwrap().is_placeholder()));
        assert_eq!(con.label(), "VGA");
    }

    #[test]
    fn hw_update_runs_gfx_update() {
        let ds = DisplayState::new();
        let hw = Arc::new(Hw(AtomicUsize::new(0)));
        let con = ds.graphic_console_create(dev(Some("vga0")), 0, hw.clone());
        con.hw_update();
        assert_eq!(hw.0.load(Ordering::Relaxed), 1);
        assert_eq!((con.width(0), con.height(0)), (320, 200));
        // A second resize to the same size keeps the surface.
        con.hw_update();
        assert!(con.with_surface(|s| s.unwrap().is_allocated()));
    }

    #[test]
    fn lookup_by_device_name_errors() {
        let ds = DisplayState::new();
        let _c = ds.graphic_console_create(dev(Some("vga0")), 0, Arc::new(Hw(AtomicUsize::new(0))));
        assert!(ds.lookup_by_device_name("vga0", 0).is_ok());
        let e = ds.lookup_by_device_name("vga0", 1).unwrap_err();
        assert_eq!(e.message(), "Device vga0 (head 1) is not bound to a QemuConsole");
        let e = ds.lookup_by_device_name("nope", 0).unwrap_err();
        assert_eq!(e.class(), ErrorClass::DeviceNotFound);
        assert_eq!(e.message(), "Device 'nope' not found");
    }

    #[test]
    fn listeners_see_switches_and_clipped_updates() {
        let ds = DisplayState::new();
        let con = ds.graphic_console_create(dev(None), 0, Arc::new(Hw(AtomicUsize::new(0))));
        let rec = Arc::new(Recorder::default());
        let id = ds.register_listener(&con, rec.clone());
        con.resize(100, 50);
        con.update(90, 40, 20, 20);
        con.set_surface(None);
        ds.unregister_listener(id);
        con.update_full();
        assert_eq!(
            *lock(&rec.log),
            [
                "switch 640x480",
                "update 0 0 640 480",
                "switch 100x50",
                "update 90 40 10 10",
                "switch 100x50",
                "update 0 0 100 50",
            ]
        );
    }

    #[test]
    fn check_format_defaults_to_native_32bpp() {
        let ds = DisplayState::new();
        let con = ds.graphic_console_create(None, 0, Arc::new(Hw(AtomicUsize::new(0))));
        assert!(con.check_format(R5G6B5));
        ds.register_listener(&con, Arc::new(Recorder::default()));
        assert!(con.check_format(X8R8G8B8));
        assert!(!con.check_format(R5G6B5));
        let all = Recorder { formats: Some(true), ..Recorder::default() };
        let ds2 = DisplayState::new();
        let con2 = ds2.graphic_console_create(None, 0, Arc::new(Hw(AtomicUsize::new(0))));
        ds2.register_listener(&con2, Arc::new(all));
        assert!(con2.check_format(R5G6B5));
    }

    #[test]
    fn a_closed_console_is_reused() {
        let ds = DisplayState::new();
        let a = ds.graphic_console_create(dev(Some("a")), 0, Arc::new(Hw(AtomicUsize::new(0))));
        let _b = ds.graphic_console_create(dev(Some("b")), 0, Arc::new(Hw(AtomicUsize::new(0))));
        a.resize(800, 600);
        a.close();
        let c = ds.graphic_console_create(dev(Some("c")), 0, Arc::new(Hw(AtomicUsize::new(0))));
        assert_eq!(c.index(), 0);
        assert_eq!((c.width(0), c.height(0)), (800, 600));
    }

    #[test]
    fn multihead_labels() {
        let ds = DisplayState::new();
        let d = dev(Some("gpu"));
        let a = ds.graphic_console_create(d.clone(), 0, Arc::new(Hw(AtomicUsize::new(0))));
        let b = ds.graphic_console_create(d, 1, Arc::new(Hw(AtomicUsize::new(0))));
        assert_eq!(a.label(), "gpu.0");
        assert_eq!(b.label(), "gpu.1");
        let c = ds.graphic_console_create(
            Some(ConsoleDevice { id: None, typename: "ramfb".into() }),
            0,
            Arc::new(Hw(AtomicUsize::new(0))),
        );
        assert_eq!(c.label(), "ramfb");
    }
}
