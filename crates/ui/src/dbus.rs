// SPDX-License-Identifier: GPL-2.0-or-later

//! `-display dbus`, QEMU's ui/dbus.c, dbus-console.c and dbus-listener.c, over the `zbus` crate.
//!
//! [`init`] connects to the session bus, or to the bus `addr=` names, and exports the objects of
//! QEMU's `org.qemu.Display1` interfaces under `/org/qemu/Display1`: an object manager, `VM`
//! and a `Console_N` for each console, with the `Console`, `Keyboard`, `Mouse` and `MultiTouch`
//! interfaces. Then it asks for the name `org.qemu`. A client calls `RegisterListener` with one
//! end of a socket pair and serves `org.qemu.Display1.Listener` on it, and the display sends it
//! the frames, as `Scanout` for the whole surface and `Update` for a part, and the cursor
//! position.
//!
//! The method calls run on the connection's own thread, one at a time. Property changes go
//! through a thread of their own, which compares the values and sends one `PropertiesChanged`
//! when something changed, as GDBus does. Each listener has a thread that does the peer to peer
//! handshake and then waits for the client to go away, and a thread that sends its frames. A
//! `Scanout` drops the `Scanout` and `Update` calls that are still queued, which is what QEMU's
//! message filter does.
//!
//! Where this differs from QEMU:
//! - There is no OpenGL, so there are no DMABUF scanouts and `gl` fails as in a QEMU built
//!   without OpenGL.
//! - Frames are always sent in the message. The shared memory of `Listener.Unix.Map` is not
//!   offered, even to a listener that lists it.
//! - No display device defines a cursor sprite yet, so `CursorDefine` is never called.
//! - `p2p=yes` exports nothing, as in QEMU before a client is added, and there is no QMP
//!   `add_client` to add one.
//! - The `Clipboard` object, the `dbus` chardevs and the `Audio` object are not exported, and
//!   `audiodev` has no effect. `-audiodev dbus` fails as an unknown audio driver.
//! - Consoles added after start are not exported.
//! - A mouse button number past the last QAPI button is ignored. QEMU passes it on.
//! - The introspection data is the one `zbus` generates. So is the text of the errors for an
//!   unknown method or property, and for a failed connection that is not a socket error.
//! - The listener proxy does not fetch the listener's properties when it is set up.
//! - This is Unix only. On Windows `RegisterListener` takes the socket in a byte array, which is
//!   not done here.

use std::collections::{HashMap, VecDeque};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

use ruvm_qapi::types::{
    DisplayOptions, DisplayOptionsU, InputAxis, InputButton, InputMultiTouchEvent,
    InputMultiTouchType,
};
use zbus::blocking::Connection;
use zbus::blocking::connection::Builder;
use zbus::message::Message;
use zbus::zvariant::{OwnedFd, Value};

use crate::console::{DisplayChangeListener, DisplayState, QemuConsole, QemuUiInfo};
use crate::input::{
    INPUT_EVENT_ABS_MAX, INPUT_EVENT_ABS_MIN, InputState, QemuInputEvent, key_number_to_linux,
    scale_axis,
};
use crate::kbd_state::{self, KbdState};

/// `DBUS_DISPLAY1_ROOT`.
const ROOT: &str = "/org/qemu/Display1";
/// The listener's object path on its own connection.
const LISTENER_PATH: &str = "/org/qemu/Display1/Listener";
const LISTENER_IFACE: &str = "org.qemu.Display1.Listener";
/// `INPUT_EVENT_SLOTS_MAX`.
const SLOTS_MAX: usize = 10;

/// The errors of `dbus_display_error_quark()`.
#[derive(zbus::DBusError, Debug)]
#[zbus(prefix = "org.qemu.Display1.Error")]
enum DisplayError {
    #[zbus(error)]
    ZBus(zbus::Error),
    Failed(String),
    Invalid(String),
    Unsupported(String),
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `dbus_init()` and `dbus_display_complete()`. `name` is `qemu_name` or "QEMU" and the version,
/// and `uuid` is `qemu_uuid`. The error is the exit status, after the message.
pub fn init(
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    opts: &DisplayOptions,
    name: String,
    uuid: [u8; 16],
) -> Result<(), u8> {
    let DisplayOptionsU::Dbus(dbus) = &opts.u else { return Ok(()) };
    if dbus.addr.is_some() && dbus.p2p == Some(true) {
        ruvm_base::error_report("dbus: can't accept both addr=X and p2p=yes options");
        return Err(1);
    }
    if dbus.p2p == Some(true) {
        // Waits for dbus_display_add_client(), which nothing calls.
        return Ok(());
    }
    let built = match dbus.addr.as_deref().filter(|a| !a.is_empty()) {
        Some(addr) => Builder::address(addr).and_then(Builder::build),
        None => Builder::session().and_then(Builder::build),
    };
    let conn = match built {
        Ok(c) => c,
        Err(e) => {
            // GIO's text for a socket that does not connect.
            let msg = match &e {
                zbus::Error::Connection(e, _) => {
                    format!("Could not connect: {}", ruvm_base::error::strerror(e))
                }
                e => e.to_string(),
            };
            ruvm_base::error_report(&format!("failed to connect to DBus: {msg}"));
            return Err(1);
        }
    };
    match export(&conn, &ds, &input, name, uuid) {
        Ok(()) => {
            // g_bus_own_name_on_connection() with no flags, whose outcome nobody looks at.
            let _ = conn.call_method(
                Some("org.freedesktop.DBus"),
                "/org/freedesktop/DBus",
                Some("org.freedesktop.DBus"),
                "RequestName",
                &("org.qemu", 0u32),
            );
            Ok(())
        }
        Err(e) => {
            ruvm_base::error_report(&format!("failed to export the D-Bus display: {e}"));
            Err(1)
        }
    }
}

/// Exports the consoles, `VM` and the object manager, and starts the property thread.
fn export(
    conn: &Connection,
    ds: &Arc<DisplayState>,
    input: &Arc<InputState>,
    name: String,
    uuid: [u8; 16],
) -> zbus::Result<()> {
    let (tx, rx) = mpsc::channel();
    let slots = Arc::new(Mutex::new([TouchSlot::default(); SLOTS_MAX]));
    let mut consoles = Vec::new();
    for con in ds.consoles() {
        consoles.push(export_console(conn, ds, input, &con, &tx, &slots)?);
    }
    // The LED and mouse mode notifiers of each console, which look at all of them.
    let leds = tx.clone();
    input.add_led_notifier(move || {
        let _ = leds.send(Change::Leds);
    });
    input.add_mouse_mode_notifier(move || {
        let _ = tx.send(Change::MouseMode);
    });
    let console_ids = consoles.iter().map(|c| c.con.index()).collect();
    let vm = Vm { name, uuid: unparse_uuid(&uuid), console_ids };
    conn.object_server().at(format!("{ROOT}/VM"), vm)?;
    conn.object_server().at(ROOT, zbus::fdo::ObjectManager)?;
    let updater = Updater { conn: conn.clone(), input: Arc::clone(input), consoles };
    std::thread::Builder::new()
        .name("dbus-props".into())
        .spawn(move || updater.run(&rx))
        .map_err(|e| zbus::Error::Failure(e.to_string()))?;
    Ok(())
}

/// `qemu_uuid_unparse_strdup()`.
fn unparse_uuid(u: &[u8; 16]) -> String {
    let hex: Vec<String> = u.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        hex[0..4].concat(),
        hex[4..6].concat(),
        hex[6..8].concat(),
        hex[8..10].concat(),
        hex[10..16].concat()
    )
}

/// A property change, for the property thread.
enum Change {
    /// The `dbus-console` listener saw a new surface on the console at this position.
    Size(usize, u32, u32),
    /// The keyboard LEDs changed.
    Leds,
    /// The mouse mode changed.
    MouseMode,
}

/// The properties of a console that change, as the property thread last sent them.
#[derive(Default)]
struct LiveProps {
    width: u32,
    height: u32,
    modifiers: u32,
    is_absolute: bool,
}

/// What the property thread knows of one exported console.
struct Exported {
    con: QemuConsole,
    path: String,
    props: Arc<Mutex<LiveProps>>,
}

/// `dbus_display_console_new()`.
fn export_console(
    conn: &Connection,
    ds: &Arc<DisplayState>,
    input: &Arc<InputState>,
    con: &QemuConsole,
    tx: &Sender<Change>,
    slots: &Arc<Mutex<[TouchSlot; SLOTS_MAX]>>,
) -> zbus::Result<Exported> {
    let path = format!("{ROOT}/Console_{}", con.index());
    let props = Arc::new(Mutex::new(LiveProps {
        width: con.width(0) as u32,
        height: con.height(0) as u32,
        modifiers: 0,
        is_absolute: false,
    }));
    let kbd = Arc::new(Mutex::new(KbdState::new(Some(con.clone()))));
    let console = Console {
        con: con.clone(),
        ds: Arc::clone(ds),
        input: Arc::clone(input),
        kbd: Arc::clone(&kbd),
        label: con.label(),
        head: con.head(),
        device_address: con.device_address(),
        props: Arc::clone(&props),
    };
    let keyboard = Keyboard { input: Arc::clone(input), kbd, props: Arc::clone(&props) };
    let mouse = Mouse { con: con.clone(), input: Arc::clone(input), props: Arc::clone(&props) };
    let touch = MultiTouch { con: con.clone(), input: Arc::clone(input), slots: Arc::clone(slots) };
    let server = conn.object_server();
    server.at(path.as_str(), console)?;
    server.at(path.as_str(), keyboard)?;
    server.at(path.as_str(), mouse)?;
    server.at(path.as_str(), touch)?;
    drop(server);
    for slot in lock(slots).iter_mut() {
        slot.tracking_id = -1;
    }
    let index = con.index() as usize;
    ds.register_listener(con, Arc::new(ConsoleSize { index, tx: tx.clone() }));
    // dbus_mouse_update_is_absolute()
    lock(&props).is_absolute = input.is_absolute(Some(con));
    Ok(Exported { con: con.clone(), path, props })
}

/// The `dbus-console` listener, which keeps `Width` and `Height` up to date.
struct ConsoleSize {
    index: usize,
    tx: Sender<Change>,
}

impl DisplayChangeListener for ConsoleSize {
    fn name(&self) -> &str {
        "dbus-console"
    }

    fn gfx_switch(&self, con: &QemuConsole) {
        let size = con.with_surface(|s| s.map(|s| (s.width() as u32, s.height() as u32)));
        if let Some((w, h)) = size {
            let _ = self.tx.send(Change::Size(self.index, w, h));
        }
    }
}

/// The property thread: sets the properties and sends `PropertiesChanged` for what changed.
struct Updater {
    conn: Connection,
    input: Arc<InputState>,
    consoles: Vec<Exported>,
}

impl Updater {
    fn run(&self, rx: &Receiver<Change>) {
        while let Ok(change) = rx.recv() {
            match change {
                Change::Size(index, w, h) => {
                    let Some(c) = self.consoles.iter().find(|c| c.con.index() as usize == index)
                    else {
                        continue;
                    };
                    let mut changed = HashMap::new();
                    {
                        let mut p = lock(&c.props);
                        if p.width != w {
                            p.width = w;
                            changed.insert("Width", Value::from(w));
                        }
                        if p.height != h {
                            p.height = h;
                            changed.insert("Height", Value::from(h));
                        }
                    }
                    self.emit(&c.path, "org.qemu.Display1.Console", changed);
                }
                Change::Leds => {
                    for c in &self.consoles {
                        let mask = self.input.get_leds_mask(Some(&c.con));
                        let mut changed = HashMap::new();
                        {
                            let mut p = lock(&c.props);
                            if p.modifiers != mask {
                                p.modifiers = mask;
                                changed.insert("Modifiers", Value::from(mask));
                            }
                        }
                        self.emit(&c.path, "org.qemu.Display1.Keyboard", changed);
                    }
                }
                Change::MouseMode => {
                    for c in &self.consoles {
                        let abs = self.input.is_absolute(Some(&c.con));
                        let mut changed = HashMap::new();
                        {
                            let mut p = lock(&c.props);
                            if p.is_absolute != abs {
                                p.is_absolute = abs;
                                changed.insert("IsAbsolute", Value::from(abs));
                            }
                        }
                        self.emit(&c.path, "org.qemu.Display1.Mouse", changed);
                    }
                }
            }
        }
    }

    fn emit(&self, path: &str, iface: &str, changed: HashMap<&str, Value<'_>>) {
        if changed.is_empty() {
            return;
        }
        let _ = self.conn.emit_signal(
            None::<&str>,
            path,
            "org.freedesktop.DBus.Properties",
            "PropertiesChanged",
            &(iface, changed, Vec::<&str>::new()),
        );
    }
}

/// `org.qemu.Display1.VM`.
struct Vm {
    name: String,
    uuid: String,
    console_ids: Vec<u32>,
}

#[zbus::interface(name = "org.qemu.Display1.VM", spawn = false)]
impl Vm {
    #[zbus(property)]
    fn name(&self) -> &str {
        &self.name
    }

    #[zbus(property, name = "UUID")]
    fn uuid(&self) -> &str {
        &self.uuid
    }

    #[zbus(property, name = "ConsoleIDs")]
    fn console_ids(&self) -> Vec<u32> {
        self.console_ids.clone()
    }

    #[zbus(property)]
    fn interfaces(&self) -> Vec<String> {
        Vec::new()
    }
}

/// `org.qemu.Display1.Console`.
struct Console {
    con: QemuConsole,
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    kbd: Arc<Mutex<KbdState>>,
    label: String,
    head: u32,
    device_address: String,
    props: Arc<Mutex<LiveProps>>,
}

#[zbus::interface(name = "org.qemu.Display1.Console", spawn = false)]
impl Console {
    /// `dbus_console_register_listener()`: replies, then does the handshake as the server of
    /// the peer to peer connection on its own thread.
    fn register_listener(&self, listener: OwnedFd) {
        let stream = UnixStream::from(std::os::fd::OwnedFd::from(listener));
        let con = self.con.clone();
        let ds = Arc::clone(&self.ds);
        let input = Arc::clone(&self.input);
        let kbd = Arc::clone(&self.kbd);
        let spawned = std::thread::Builder::new()
            .name("dbus-listener".into())
            .spawn(move || run_listener(stream, &con, &ds, &input, &kbd));
        if let Err(e) = spawned {
            ruvm_base::error_report(&format!("Failed to setup peer connection: {e}"));
        }
    }

    /// `dbus_console_set_ui_info()`.
    #[zbus(name = "SetUIInfo")]
    fn set_ui_info(
        &self,
        width_mm: u16,
        height_mm: u16,
        xoff: i32,
        yoff: i32,
        width: u32,
        height: u32,
    ) -> Result<(), DisplayError> {
        if !self.con.ui_info_supported() {
            return Err(DisplayError::Unsupported("SetUIInfo is not supported".into()));
        }
        let info = QemuUiInfo { width_mm, height_mm, xoff, yoff, width, height, refresh_rate: 0 };
        self.con.set_ui_info(info);
        Ok(())
    }

    #[zbus(property)]
    fn label(&self) -> &str {
        &self.label
    }

    #[zbus(property)]
    fn head(&self) -> u32 {
        self.head
    }

    #[zbus(property, name = "Type")]
    fn type_(&self) -> &str {
        if self.con.is_graphic() { "Graphic" } else { "Text" }
    }

    #[zbus(property)]
    fn width(&self) -> u32 {
        lock(&self.props).width
    }

    #[zbus(property)]
    fn height(&self) -> u32 {
        lock(&self.props).height
    }

    #[zbus(property)]
    fn device_address(&self) -> &str {
        &self.device_address
    }

    #[zbus(property)]
    fn interfaces(&self) -> Vec<&str> {
        vec![
            "org.qemu.Display1.Keyboard",
            "org.qemu.Display1.Mouse",
            "org.qemu.Display1.MultiTouch",
        ]
    }
}

/// `org.qemu.Display1.Keyboard`.
struct Keyboard {
    input: Arc<InputState>,
    kbd: Arc<Mutex<KbdState>>,
    props: Arc<Mutex<LiveProps>>,
}

impl Keyboard {
    fn key(&self, keycode: u32, down: bool) {
        let lnx = key_number_to_linux(i64::from(keycode));
        let mut out = Vec::new();
        lock(&self.kbd).key_event(lnx, down, &mut out);
        kbd_state::send(&self.input, out);
    }
}

#[zbus::interface(name = "org.qemu.Display1.Keyboard", spawn = false)]
impl Keyboard {
    /// `dbus_kbd_press()`.
    fn press(&self, keycode: u32) {
        self.key(keycode, true);
    }

    /// `dbus_kbd_release()`.
    fn release(&self, keycode: u32) {
        self.key(keycode, false);
    }

    #[zbus(property)]
    fn modifiers(&self) -> u32 {
        lock(&self.props).modifiers
    }
}

/// `org.qemu.Display1.Mouse`.
struct Mouse {
    con: QemuConsole,
    input: Arc<InputState>,
    props: Arc<Mutex<LiveProps>>,
}

impl Mouse {
    fn button(&self, button: u32, down: bool) {
        if let Some(&b) = InputButton::ALL.get(button as usize) {
            self.input.queue_btn(Some(&self.con), b, down);
            self.input.event_sync();
        }
    }
}

#[zbus::interface(name = "org.qemu.Display1.Mouse", spawn = false)]
impl Mouse {
    /// `dbus_mouse_press()`.
    fn press(&self, button: u32) {
        self.button(button, true);
    }

    /// `dbus_mouse_release()`.
    fn release(&self, button: u32) {
        self.button(button, false);
    }

    /// `dbus_mouse_set_pos()`.
    fn set_abs_position(&self, x: u32, y: u32) -> Result<(), DisplayError> {
        let con = Some(&self.con);
        if !self.input.is_absolute(con) {
            return Err(DisplayError::Invalid("Mouse is not absolute".into()));
        }
        let width = self.con.width(0);
        let height = self.con.height(0);
        // The comparison is unsigned, as in QEMU.
        if x >= width as u32 || y >= height as u32 {
            return Err(DisplayError::Invalid("Invalid mouse position".into()));
        }
        self.input.queue_abs(con, InputAxis::X, x as i32, 0, width);
        self.input.queue_abs(con, InputAxis::Y, y as i32, 0, height);
        self.input.event_sync();
        Ok(())
    }

    /// `dbus_mouse_rel_motion()`.
    fn rel_motion(&self, dx: i32, dy: i32) -> Result<(), DisplayError> {
        let con = Some(&self.con);
        if self.input.is_absolute(con) {
            return Err(DisplayError::Invalid("Mouse is not relative".into()));
        }
        self.input.queue_rel(con, InputAxis::X, i64::from(dx));
        self.input.queue_rel(con, InputAxis::Y, i64::from(dy));
        self.input.event_sync();
        Ok(())
    }

    #[zbus(property)]
    fn is_absolute(&self) -> bool {
        lock(&self.props).is_absolute
    }
}

/// `struct touch_slot`.
#[derive(Clone, Copy, Default)]
struct TouchSlot {
    x: f64,
    y: f64,
    tracking_id: i64,
}

/// `org.qemu.Display1.MultiTouch`. The slots are shared by all consoles, as QEMU's static
/// `touch_slots` are.
struct MultiTouch {
    con: QemuConsole,
    input: Arc<InputState>,
    slots: Arc<Mutex<[TouchSlot; SLOTS_MAX]>>,
}

#[zbus::interface(name = "org.qemu.Display1.MultiTouch", spawn = false)]
impl MultiTouch {
    /// `dbus_touch_send_event()`.
    fn send_event(&self, kind: u32, num_slot: u64, x: f64, y: f64) -> Result<(), DisplayError> {
        let kind = match kind {
            0 => InputMultiTouchType::Begin,
            1 => InputMultiTouchType::Update,
            2 => InputMultiTouchType::End,
            3 => InputMultiTouchType::Cancel,
            _ => return Err(DisplayError::Invalid("Invalid touch event kind".into())),
        };
        let width = self.con.width(0);
        let height = self.con.height(0);
        touch_event(&self.input, &self.con, &self.slots, num_slot, width, height, x, y, kind)
            .map_err(DisplayError::Invalid)
    }

    #[zbus(property)]
    fn max_slots(&self) -> i32 {
        SLOTS_MAX as i32
    }
}

/// `qemu_input_touch_event()`.
#[allow(clippy::too_many_arguments)]
fn touch_event(
    input: &InputState,
    con: &QemuConsole,
    slots: &Mutex<[TouchSlot; SLOTS_MAX]>,
    num_slot: u64,
    width: i32,
    height: i32,
    x: f64,
    y: f64,
    kind: InputMultiTouchType,
) -> Result<(), String> {
    if num_slot >= SLOTS_MAX as u64 {
        // QEMU's format is "% " PRId64, which puts a space before a positive number.
        let n = num_slot as i64;
        let n = if n >= 0 { format!(" {n}") } else { n.to_string() };
        return Err(format!("Unexpected touch slot number: {n} >= {SLOTS_MAX}"));
    }
    let src = Some(con);
    let mtt = |type_, slot: usize, tracking_id, axis, value| {
        let evt = InputMultiTouchEvent { type_, slot: slot as i64, tracking_id, axis, value };
        input.event_send(src, &QemuInputEvent::Mtt(evt));
    };
    let abs = |value: f64, max| {
        let v = scale_axis(
            value as i32,
            0,
            max,
            INPUT_EVENT_ABS_MIN as i32,
            INPUT_EVENT_ABS_MAX as i32,
        );
        i64::from(v)
    };
    let mut needs_sync = false;
    {
        let mut slots = lock(slots);
        let n = num_slot as usize;
        slots[n].x = x;
        slots[n].y = y;
        if kind == InputMultiTouchType::Begin {
            slots[n].tracking_id = num_slot as i64;
        }
        for (i, slot) in slots.iter_mut().enumerate() {
            let update = if i == n { kind } else { InputMultiTouchType::Update };
            if slot.tracking_id == -1 {
                continue;
            }
            if update == InputMultiTouchType::End {
                slot.tracking_id = -1;
                mtt(update, i, -1, InputAxis::X, 0);
            } else {
                let tid = slot.tracking_id;
                mtt(update, i, tid, InputAxis::X, 0);
                input.queue_btn(src, InputButton::Touch, true);
                mtt(InputMultiTouchType::Data, i, tid, InputAxis::X, abs(slot.x, width));
                mtt(InputMultiTouchType::Data, i, tid, InputAxis::Y, abs(slot.y, height));
            }
            needs_sync = true;
        }
    }
    if needs_sync {
        input.event_sync();
    }
    Ok(())
}

/// The listener's thread: the handshake, then `dbus_display_listener_new()`, then the wait for
/// the client to go away and `listener_vanished_cb()`.
fn run_listener(
    stream: UnixStream,
    con: &QemuConsole,
    ds: &DisplayState,
    input: &InputState,
    kbd: &Mutex<KbdState>,
) {
    let conn = Builder::async_io_unix_stream(stream)
        .server(zbus::Guid::generate())
        .map(|b| b.p2p())
        .and_then(Builder::build);
    let conn = match conn {
        Ok(c) => c,
        Err(e) => {
            ruvm_base::error_report(&format!("Failed to setup peer connection: {e}"));
            return;
        }
    };
    let outbox = Arc::new(Outbox::default());
    let sender = {
        let (conn, outbox) = (conn.clone(), Arc::clone(&outbox));
        std::thread::Builder::new().name("dbus-send".into()).spawn(move || outbox.run(&conn))
    };
    if sender.is_err() {
        return;
    }
    let id = ds.register_listener(con, Arc::new(Listener { outbox: Arc::clone(&outbox) }));
    conn.closed();
    ds.unregister_listener(id);
    outbox.close();
    let mut out = Vec::new();
    lock(kbd).lift_all_keys(&mut out);
    kbd_state::send(input, out);
}

/// A call to the listener.
enum Call {
    Scanout { width: u32, height: u32, stride: u32, format: u32, data: Vec<u8> },
    Update { x: i32, y: i32, w: i32, h: i32, stride: u32, format: u32, data: Vec<u8> },
    MouseSet { x: i32, y: i32, on: i32 },
}

/// The calls waiting for the listener's sender thread.
#[derive(Default)]
struct Outbox {
    queue: Mutex<(VecDeque<Call>, bool)>,
    cv: Condvar,
}

impl Outbox {
    fn push(&self, call: Call) {
        let mut q = lock(&self.queue);
        if matches!(call, Call::Scanout { .. }) {
            // ddl_discard_display_messages()
            q.0.retain(|c| matches!(c, Call::MouseSet { .. }));
        }
        q.0.push_back(call);
        self.cv.notify_one();
    }

    fn close(&self) {
        lock(&self.queue).1 = true;
        self.cv.notify_one();
    }

    fn run(&self, conn: &Connection) {
        loop {
            let call = {
                let mut q = lock(&self.queue);
                loop {
                    if q.1 {
                        return;
                    }
                    if let Some(c) = q.0.pop_front() {
                        break c;
                    }
                    q = self.cv.wait(q).unwrap_or_else(PoisonError::into_inner);
                }
            };
            // A method call on the listener, which nobody waits for the reply of.
            let method = |member| {
                Message::method_call(LISTENER_PATH, member)
                    .and_then(|b| b.interface(LISTENER_IFACE))
            };
            let msg = match &call {
                Call::Scanout { width, height, stride, format, data } => method("Scanout")
                    .and_then(|b| b.build(&(*width, *height, *stride, *format, data.as_slice()))),
                Call::Update { x, y, w, h, stride, format, data } => method("Update")
                    .and_then(|b| b.build(&(*x, *y, *w, *h, *stride, *format, data.as_slice()))),
                Call::MouseSet { x, y, on } => {
                    method("MouseSet").and_then(|b| b.build(&(*x, *y, *on)))
                }
            };
            if let Ok(msg) = msg {
                let _ = conn.send(&msg);
            }
        }
    }
}

/// The `dbus` listener, `dbus_dcl_ops`.
struct Listener {
    outbox: Arc<Outbox>,
}

impl DisplayChangeListener for Listener {
    fn name(&self) -> &str {
        "dbus"
    }

    fn has_refresh(&self) -> bool {
        true
    }

    /// `dbus_refresh()`.
    fn refresh(&self, con: &QemuConsole) {
        con.hw_update_nowait();
    }

    /// `dbus_gfx_update()`: the whole surface goes out as `Scanout`, a part as `Update` with
    /// the rows packed.
    fn gfx_update(&self, con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        let call = con.with_surface(|s| {
            let s = s?;
            let format = s.format().0;
            let (sw, sh, stride) = (s.width(), s.height(), s.stride());
            if x == 0 && y == 0 && w as usize == sw && h as usize == sh {
                let data = s.data().get(..stride * sh)?.to_vec();
                let (width, height, stride) = (sw as u32, sh as u32, stride as u32);
                return Some(Call::Scanout { width, height, stride, format, data });
            }
            if x < 0 || y < 0 || w <= 0 || h <= 0 {
                return None;
            }
            let bpp = s.bytes_per_pixel();
            let row = w as usize * bpp;
            let mut data = Vec::with_capacity(row * h as usize);
            for r in 0..h as usize {
                let start = (y as usize + r) * stride + x as usize * bpp;
                data.extend_from_slice(s.data().get(start..start + row)?);
            }
            Some(Call::Update { x, y, w, h, stride: row as u32, format, data })
        });
        if let Some(call) = call {
            self.outbox.push(call);
        }
    }

    /// `dbus_mouse_set()`.
    fn mouse_set(&self, x: i32, y: i32, on: bool) {
        self.outbox.push(Call::MouseSet { x, y, on: i32::from(on) });
    }
}
