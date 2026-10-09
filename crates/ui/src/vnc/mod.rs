// SPDX-License-Identifier: GPL-2.0-or-later

//! The VNC server, from QEMU's ui/vnc.c, ui/vnc-jobs.c and the encoders next to them.
//!
//! A display is a `-vnc` option: its listening sockets, its clients and a console it watches as
//! a [`DisplayChangeListener`]. Like QEMU it keeps three pictures of the screen. The guest
//! surface belongs to the console. The server surface is the display's own `x8r8g8b8` copy,
//! refreshed from the guest every refresh interval in 16 pixel chunks, and every chunk that
//! really changed is marked in each client's dirty map. A client that asked for an update gets
//! the dirty rectangles of its map, encoded with the encoding it picked.
//!
//! The server speaks RFB 3.3, 3.7 and 3.8 with no security or VNC password authentication,
//! and sends raw, hextile, zlib and tight (without JPEG) rectangles, desktop size changes in
//! both forms and WMVi pixel format changes. Where it differs from QEMU:
//!
//! - Cut text messages are read and dropped, as ruvm has no clipboard yet.
//! - The Ctrl+Alt+1 to 9 keys move the keyboard to that console, but the picture stays on the
//!   display's own console. With the one graphic console ruvm has, the two are the same.
//! - ZRLE, ZYWRLE, tight PNG, the extended clipboard and audio are not built in, so the
//!   client's choice falls to its next encoding, as with a QEMU built without them.
//! - The XVP shutdown and reset actions answer with an XVP failure, as ruvm has no powerdown
//!   or reset request to make yet.
//! - Updates are encoded on the thread that asks for them rather than on a worker thread, which
//!   only changes when the bytes are written.
//! - zlib streams are opened with flate2's memory level rather than zlib's largest, so the
//!   compressed bytes can differ. They inflate to the same pixels. A level change that zlib-rs
//!   cannot make on a used stream keeps the old level.
//! - Without a display option ruvm does not start a VNC server of its own on localhost:0.
//!
//! Each client has a reader and a writer thread. The reader feeds the protocol handlers under
//! the display's lock, and the writer drains what they queued. The device side is only called
//! once the lock is dropped, see [`Deferred`]. Key and pointer input are worked out under the
//! lock too, keeping the keyboard state of `ui/kbd-state.c` there, and go to the input layer
//! afterwards.

mod client;
mod hextile;
mod net;
mod opts;
mod raw;
mod tight;
mod zlib;

pub mod auth;
pub mod palette;
pub mod pixels;

#[cfg(test)]
mod tests;

use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::types::{
    ChangeVncPasswordArg, ExpirePasswordOptions, ExpirePasswordOptionsU, InputAxis, InputButton,
    SetPasswordAction, SetPasswordOptions, SetPasswordOptionsU, VncClientInfo, VncConnectedArg,
    VncDisconnectedArg, VncInfo, VncInfo2, VncInitializedArg, VncPrimaryAuth, VncServerInfo,
    VncServerInfo2,
};

use crate::console::{DisplayChangeListener, DisplayState, ListenerId, QemuConsole, QemuUiInfo};
use crate::input::InputState;
use crate::kbd_state::{self, KbdOut, KbdState};
use crate::keymaps::KbdLayout;
use crate::pixman::{self, A8R8G8B8, PixelFormat, X8R8G8B8};
use crate::surface::DisplaySurface;

use client::{Client, ShareMode, Update};
use net::{AddrInfo, Listener};

pub use opts::{configured, init, parse};

pub(crate) const ENCODING_RAW: i32 = 0;
pub(crate) const ENCODING_HEXTILE: i32 = 5;
pub(crate) const ENCODING_ZLIB: i32 = 6;
pub(crate) const ENCODING_TIGHT: i32 = 7;
pub(crate) const ENCODING_DESKTOPRESIZE: i32 = -223;
pub(crate) const ENCODING_RICH_CURSOR: i32 = -239;
pub(crate) const ENCODING_POINTER_TYPE_CHANGE: i32 = -257;
pub(crate) const ENCODING_EXT_KEY_EVENT: i32 = -258;
pub(crate) const ENCODING_AUDIO: i32 = -259;
pub(crate) const ENCODING_LED_STATE: i32 = -261;
pub(crate) const ENCODING_DESKTOP_RESIZE_EXT: i32 = -308;
pub(crate) const ENCODING_XVP: i32 = -309;
pub(crate) const ENCODING_ALPHA_CURSOR: i32 = -314;
pub(crate) const ENCODING_WMVI: i32 = 0x574D_5669;
pub(crate) const ENCODING_COMPRESSLEVEL0: i32 = -256;
pub(crate) const ENCODING_QUALITYLEVEL0: i32 = -32;

/// `VNC_DIRTY_PIXELS_PER_BIT`.
pub(crate) const DIRTY_PIXELS_PER_BIT: usize = 16;
/// `VNC_MAX_WIDTH`.
pub(crate) const MAX_WIDTH: usize = 5120;
/// `VNC_MAX_HEIGHT`.
pub(crate) const MAX_HEIGHT: usize = 2160;
/// `VNC_DIRTY_BPL()`: the bits of one dirty map row.
pub(crate) const DIRTY_BPL: usize = MAX_WIDTH / DIRTY_PIXELS_PER_BIT;
const DIRTY_WORDS: usize = DIRTY_BPL / 64;

/// `VNC_REFRESH_INTERVAL_BASE`, `_INC` and `_MAX`, in milliseconds.
pub(crate) const REFRESH_INTERVAL_BASE: u64 = 30;
const REFRESH_INTERVAL_INC: u64 = 50;
pub(crate) const REFRESH_INTERVAL_MAX: u64 = 3000;

const NODEV_MSG: &str = "This VM has no graphic display device.";
const NOT_ACTIVE_MSG: &str = "Display output is not active.";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// `time(NULL)`.
pub(crate) fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64)
}

/// The server surface as the encoders read it: host order `x8r8g8b8` pixels, `stride` a row.
pub(crate) struct Fb<'a> {
    data: &'a [u32],
    stride: usize,
}

impl<'a> Fb<'a> {
    pub(crate) fn new(data: &'a [u32], stride: usize) -> Fb<'a> {
        Fb { data, stride }
    }

    pub(crate) fn row(&self, x: usize, y: usize, w: usize) -> &'a [u32] {
        let start = y * self.stride + x;
        &self.data[start..start + w]
    }

    pub(crate) fn pixel(&self, x: usize, y: usize) -> u32 {
        self.data[y * self.stride + x]
    }
}

/// `vnc_framebuffer_update()`: a rectangle header.
pub(crate) fn framebuffer_update(
    out: &mut Vec<u8>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    encoding: i32,
) {
    out.extend_from_slice(&(x as u16).to_be_bytes());
    out.extend_from_slice(&(y as u16).to_be_bytes());
    out.extend_from_slice(&(w as u16).to_be_bytes());
    out.extend_from_slice(&(h as u16).to_be_bytes());
    out.extend_from_slice(&encoding.to_be_bytes());
}

/// A dirty bitmap of `VNC_MAX_HEIGHT` rows of `VNC_DIRTY_BPL` bits, one bit for 16 pixels.
#[derive(Clone)]
pub(crate) struct DirtyMap {
    words: Vec<u64>,
}

impl DirtyMap {
    pub(crate) fn new() -> DirtyMap {
        DirtyMap { words: vec![0; MAX_HEIGHT * DIRTY_WORDS] }
    }

    pub(crate) fn clear_all(&mut self) {
        self.words.fill(0);
    }

    fn at(y: usize, bit: usize) -> (usize, u64) {
        (y * DIRTY_WORDS + bit / 64, 1u64 << (bit % 64))
    }

    pub(crate) fn set(&mut self, y: usize, bit: usize) {
        let (i, m) = DirtyMap::at(y, bit);
        self.words[i] |= m;
    }

    pub(crate) fn test(&self, y: usize, bit: usize) -> bool {
        let (i, m) = DirtyMap::at(y, bit);
        self.words[i] & m != 0
    }

    pub(crate) fn test_and_clear(&mut self, y: usize, bit: usize) -> bool {
        let (i, m) = DirtyMap::at(y, bit);
        let was = self.words[i] & m != 0;
        self.words[i] &= !m;
        was
    }

    /// `bitmap_set()` on row `y`.
    pub(crate) fn set_bits(&mut self, y: usize, start: usize, n: usize) {
        for bit in start..(start + n).min(DIRTY_BPL) {
            self.set(y, bit);
        }
    }

    /// `bitmap_clear()` on row `y`.
    pub(crate) fn clear_bits(&mut self, y: usize, start: usize, n: usize) {
        for bit in start..(start + n).min(DIRTY_BPL) {
            self.test_and_clear(y, bit);
        }
    }

    /// `find_next_bit()` over the rows as one bitmap of `size` bits.
    pub(crate) fn find_next_bit(&self, size: usize, offset: usize) -> usize {
        let mut i = offset;
        while i < size {
            let w = self.words[i / 64] >> (i % 64);
            if w != 0 {
                return (i + w.trailing_zeros() as usize).min(size);
            }
            i = (i / 64 + 1) * 64;
        }
        size
    }

    /// `find_next_zero_bit()` in row `y`.
    pub(crate) fn find_next_zero_bit(&self, y: usize, offset: usize) -> usize {
        let mut i = offset;
        while i < DIRTY_BPL {
            let w = !self.words[y * DIRTY_WORDS + i / 64] >> (i % 64);
            if w != 0 {
                return (i + w.trailing_zeros() as usize).min(DIRTY_BPL);
            }
            i = (i / 64 + 1) * 64;
        }
        DIRTY_BPL
    }
}

/// `vnc_set_area_dirty()`, against a guest surface of `width` by `height` pixels.
pub(crate) fn set_area_dirty(
    dirty: &mut DirtyMap,
    (gw, gh): (usize, usize),
    x: i64,
    y: i64,
    w: i64,
    h: i64,
) {
    let width = vnc_width(gw) as i64;
    let height = vnc_height(gh) as i64;
    let px = DIRTY_PIXELS_PER_BIT as i64;
    let w = w + x % px;
    let x = x - x % px;
    let x = x.min(width);
    let mut y = y.min(height);
    let w = (x + w).min(width) - x;
    let h = (y + h).min(height);
    while y < h {
        if x >= 0 && w > 0 && y >= 0 {
            dirty.set_bits(
                y as usize,
                x as usize / DIRTY_PIXELS_PER_BIT,
                (w as usize).div_ceil(16),
            );
        }
        y += 1;
    }
}

/// `vnc_width()`.
pub(crate) fn vnc_width(guest_width: usize) -> usize {
    MAX_WIDTH.min(guest_width.next_multiple_of(DIRTY_PIXELS_PER_BIT))
}

/// `vnc_height()`.
pub(crate) fn vnc_height(guest_height: usize) -> usize {
    MAX_HEIGHT.min(guest_height)
}

/// `vnc_true_width()`.
fn vnc_true_width(guest_width: usize) -> usize {
    MAX_WIDTH.min(guest_width)
}

/// `VncDisplay.auth`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Auth {
    None = 1,
    Vnc = 2,
}

impl Auth {
    /// `vnc_auth_name()`.
    fn name(self) -> &'static str {
        match self {
            Auth::None => "none",
            Auth::Vnc => "vnc",
        }
    }

    fn primary(self) -> VncPrimaryAuth {
        match self {
            Auth::None => VncPrimaryAuth::None,
            Auth::Vnc => VncPrimaryAuth::Vnc,
        }
    }
}

/// `VncSharePolicy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharePolicy {
    Ignore,
    AllowExclusive,
    ForceShared,
}

/// What `vnc_display_open()` takes from the options.
pub(crate) struct Config {
    pub(crate) auth: Auth,
    pub(crate) password: Option<String>,
    pub(crate) share_policy: SharePolicy,
    pub(crate) connections_limit: u64,
    pub(crate) power_control: bool,
    pub(crate) lossy: bool,
    pub(crate) lock_key_sync: bool,
    pub(crate) key_delay_ms: u32,
}

/// A QMP event a display wants sent.
#[derive(Debug)]
pub enum VncEvent {
    Connected(VncConnectedArg),
    Initialized(VncInitializedArg),
    Disconnected(VncDisconnectedArg),
}

/// What the VNC server needs from the rest of the emulator.
pub trait Hooks: Send + Sync {
    /// Sends a VNC QMP event.
    fn event(&self, event: VncEvent);

    /// `qemu_system_powerdown_request()`, for the XVP shutdown action. False when the machine
    /// cannot take it, which the client hears as an XVP failure.
    fn powerdown(&self) -> bool {
        false
    }

    /// `qemu_system_reset_request()`, for the XVP reset action.
    fn reset(&self) -> bool {
        false
    }
}

/// Work that calls into the console or the rest of the emulator, done after the display's lock
/// is dropped. The device behind a console takes its own locks there, and a vCPU thread holding
/// them may be waiting for the display's lock to report a dirty rectangle.
pub(crate) enum Deferred {
    Event(VncEvent),
    SetRefresh(u64),
    HwInvalidate,
    HwUpdate,
    SetUiInfo(u32, u32),
    XvpPowerdown(u64),
    XvpReset(u64),
    /// Keys from the keyboard state, in order.
    Keys(Vec<KbdOut>),
    /// `pointer_event()`: the buttons that changed, the motion and a sync.
    Pointer {
        old: u8,
        new: u8,
        motion: Motion,
    },
}

/// The pointer motion of `pointer_event()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Motion {
    None,
    /// A position on a server surface of `width` by `height`.
    Abs {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
    },
    Rel {
        dx: i64,
        dy: i64,
    },
}

/// The `bmap` of `pointer_event()`.
const POINTER_BMAP: [(InputButton, u32); 5] = [
    (InputButton::Left, 0x01),
    (InputButton::Middle, 0x02),
    (InputButton::Right, 0x04),
    (InputButton::WheelUp, 0x08),
    (InputButton::WheelDown, 0x10),
];

/// The guest surface as the display last saw it in `dpy_gfx_switch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Guest {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) format: PixelFormat,
}

/// `VncDisplay.server`.
pub(crate) struct ServerFb {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) data: Vec<u32>,
}

/// The half of `VncDisplay` behind its lock.
pub(crate) struct VdState {
    pub(crate) password: Option<String>,
    pub(crate) expires: i64,
    pub(crate) guest: Option<Guest>,
    pub(crate) guest_dirty: DirtyMap,
    pub(crate) server: Option<ServerFb>,
    pub(crate) true_width: usize,
    pub(crate) clients: Vec<Client>,
    pub(crate) interval: u64,
    pub(crate) ledstate: u8,
    /// `vd->kbd`.
    pub(crate) kbd: KbdState,
    pub(crate) deferred: Vec<Deferred>,
}

impl VdState {
    pub(crate) fn client_index(&self, id: u64) -> Option<usize> {
        self.clients.iter().position(|c| c.id == id)
    }

    /// `update_displaychangelistener()`.
    pub(crate) fn set_refresh(&mut self, interval: u64) {
        self.interval = interval;
        self.deferred.push(Deferred::SetRefresh(interval));
    }

    pub(crate) fn guest_dims(&self) -> (usize, usize) {
        self.guest.map_or((0, 0), |g| (g.width, g.height))
    }

    pub(crate) fn server_dims(&self) -> (usize, usize) {
        self.server.as_ref().map_or((0, 0), |s| (s.width, s.height))
    }

    pub(crate) fn num_mode(&self, mode: ShareMode) -> u64 {
        self.clients.iter().filter(|c| c.share_mode == mode).count() as u64
    }

    /// `vnc_update_server_surface()`.
    fn update_server_surface(&mut self) {
        self.server = None;
        if self.clients.is_empty() {
            return;
        }
        let (gw, gh) = self.guest_dims();
        let width = vnc_width(gw);
        let height = vnc_height(gh);
        self.true_width = vnc_true_width(gw);
        self.server = Some(ServerFb { width, height, data: vec![0; width * height] });
        self.guest_dirty.clear_all();
        set_area_dirty(&mut self.guest_dirty, (gw, gh), 0, 0, width as i64, height as i64);
    }

    /// `vnc_abort_display_jobs()`. Jobs run to the end before the lock is dropped, so only a
    /// job without rectangles, which QEMU drops without finishing, is left to undo.
    fn abort_display_jobs(&mut self) {
        for c in &mut self.clients {
            if c.update == Update::None && c.job_update != Update::None {
                c.update = c.job_update;
                c.job_update = Update::None;
            }
        }
    }

    /// `vnc_refresh_server_surface()`: copies the guest chunks that changed into the server
    /// surface and marks them dirty for every client. Returns the number of chunks.
    fn refresh_server_surface(&mut self, surface: Option<&DisplaySurface>) -> i64 {
        let (Some(server), Some(guest), Some(surface)) =
            (self.server.as_mut(), self.guest, surface)
        else {
            return 0;
        };
        // A surface the display was not told about yet waits for its switch.
        if surface.width() != guest.width
            || surface.height() != guest.height
            || surface.format() != guest.format
        {
            return 0;
        }
        let width = guest.width.min(server.width);
        let height = guest.height.min(server.height);
        let size = height * DIRTY_BPL;
        let mut offset = self.guest_dirty.find_next_bit(size, 0);
        if offset == size {
            return 0;
        }
        let image = surface.image();
        let same = guest.format == X8R8G8B8;
        let cmp_px = DIRTY_PIXELS_PER_BIT.min(server.width);
        let line_px = if same { server.width.min(guest.width) } else { server.width };
        let fill = if same { line_px } else { width };
        let mut line = vec![0u32; server.width];
        let mut has_dirty = 0;
        loop {
            let y = offset / DIRTY_BPL;
            let mut x = offset % DIRTY_BPL;
            if same {
                for (i, px) in image.row(y)[..fill * 4].chunks_exact(4).enumerate() {
                    line[i] = u32::from_ne_bytes([px[0], px[1], px[2], px[3]]);
                }
            } else if guest.format == A8R8G8B8 {
                // pixman copies this one as it is, alpha and all.
                for (i, p) in line[..fill].iter_mut().enumerate() {
                    *p = image.pixel(i, y);
                }
            } else {
                // pixman's conversions into x8r8g8b8 keep the source alpha in the unused byte,
                // 0xff for a format without one, and a client with the server's format gets that
                // byte too.
                for (i, p) in line[..fill].iter_mut().enumerate() {
                    *p = pixman::pack(A8R8G8B8, pixman::unpack(guest.format, image.pixel(i, y)));
                }
            }
            while x < width.div_ceil(DIRTY_PIXELS_PER_BIT) {
                if self.guest_dirty.test_and_clear(y, x) {
                    let start = x * cmp_px;
                    let n = if (x + 1) * cmp_px > line_px { line_px - start } else { cmp_px };
                    let at = y * server.width + start;
                    let dst = &mut server.data[at..at + n];
                    let src = &line[start..start + n];
                    if dst != src {
                        dst.copy_from_slice(src);
                        for c in &mut self.clients {
                            c.dirty.set(y, x);
                        }
                        has_dirty += 1;
                    }
                }
                x += 1;
            }
            offset = self.guest_dirty.find_next_bit(size, (y + 1) * DIRTY_BPL);
            if offset == size {
                break;
            }
        }
        has_dirty
    }
}

/// Wakes the refresh thread of a display that has no console to ride on.
struct Ticker {
    kicked: Mutex<bool>,
    cv: Condvar,
}

/// `VncDisplay`.
pub struct VncDisplay {
    id: String,
    me: Weak<VncDisplay>,
    name: Option<String>,
    ds: Arc<DisplayState>,
    con: Option<QemuConsole>,
    /// The surface shown when the console has none, or when there is no console.
    fallback: DisplaySurface,
    listener_id: OnceLock<ListenerId>,
    listeners: OnceLock<Vec<AddrInfo>>,
    ticker: Ticker,
    pub(crate) cfg: Config,
    pub(crate) layout: KbdLayout,
    input: Arc<InputState>,
    hooks: Arc<dyn Hooks>,
    next_client: std::sync::atomic::AtomicU64,
    state: Mutex<VdState>,
}

impl std::fmt::Debug for VncDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VncDisplay").field("id", &self.id).finish()
    }
}

/// `vnc_displays`.
static DISPLAYS: Mutex<Vec<Arc<VncDisplay>>> = Mutex::new(Vec::new());

/// `vnc_display_find()`: by id, or the first display.
fn display_find(id: Option<&str>) -> Option<Arc<VncDisplay>> {
    let displays = lock(&DISPLAYS);
    match id {
        None => displays.first().cloned(),
        Some(id) => displays.iter().find(|d| d.id == id).cloned(),
    }
}

impl VncDisplay {
    /// The display half of `vnc_display_new()` and `vnc_display_open()`, after the options
    /// were read: picks up the console, starts watching it and hooks into the input layer. The
    /// caller listens.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: &str,
        cfg: Config,
        layout: KbdLayout,
        name: Option<&str>,
        ds: Arc<DisplayState>,
        con: Option<QemuConsole>,
        input: Arc<InputState>,
        hooks: Arc<dyn Hooks>,
    ) -> Arc<VncDisplay> {
        let fallback = DisplaySurface::placeholder(
            640,
            480,
            if con.is_some() { NOT_ACTIVE_MSG } else { NODEV_MSG },
        );
        let password = cfg.password.clone();
        let mut kbd = KbdState::new(con.clone());
        kbd.set_delay(cfg.key_delay_ms);
        let lock_key_sync = cfg.lock_key_sync;
        let vd = Arc::new_cyclic(|me| VncDisplay {
            id: id.to_string(),
            me: me.clone(),
            name: name.map(str::to_string),
            ds,
            con,
            fallback,
            listener_id: OnceLock::new(),
            listeners: OnceLock::new(),
            ticker: Ticker { kicked: Mutex::new(false), cv: Condvar::new() },
            cfg,
            layout,
            input,
            hooks,
            next_client: std::sync::atomic::AtomicU64::new(1),
            state: Mutex::new(VdState {
                password,
                expires: i64::MAX,
                guest: None,
                guest_dirty: DirtyMap::new(),
                server: None,
                true_width: 0,
                clients: Vec::new(),
                interval: 0,
                ledstate: 0,
                kbd,
                deferred: Vec::new(),
            }),
        });
        if lock_key_sync {
            let me = Arc::downgrade(&vd);
            vd.input.add_led_notifier(move || {
                if let Some(vd) = me.upgrade() {
                    vd.kbd_leds();
                }
            });
        }
        // Each client's `mouse_mode_notifier`, one for all of them.
        let me = Arc::downgrade(&vd);
        vd.input.add_mouse_mode_notifier(move || {
            if let Some(vd) = me.upgrade() {
                vd.check_pointer_type_change();
            }
        });
        match &vd.con {
            Some(con) => {
                let id =
                    vd.ds.register_listener(con, Arc::clone(&vd) as Arc<dyn DisplayChangeListener>);
                let _ = vd.listener_id.set(id);
                // A console without a surface yet still has the display show something.
                if con.with_surface(|s| s.is_none()) {
                    vd.switch();
                }
            }
            None => {
                vd.switch();
                vd.start_refresh_thread();
            }
        }
        vd
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub(crate) fn console(&self) -> Option<&QemuConsole> {
        self.con.as_ref()
    }

    /// Runs `f` under the display's lock, queues what the clients were sent and then does the
    /// deferred work.
    pub(crate) fn with_state<R>(&self, f: impl FnOnce(&mut VdState) -> R) -> R {
        let (r, deferred) = {
            let mut st = lock(&self.state);
            let r = f(&mut st);
            for c in &mut st.clients {
                c.flush();
            }
            (r, std::mem::take(&mut st.deferred))
        };
        self.run_deferred(deferred);
        r
    }

    fn run_deferred(&self, deferred: Vec<Deferred>) {
        for d in deferred {
            match d {
                Deferred::Event(ev) => self.hooks.event(ev),
                Deferred::SetRefresh(ms) => match self.listener_id.get() {
                    Some(id) => self.ds.listener_set_refresh(*id, ms),
                    None => {
                        *lock(&self.ticker.kicked) = true;
                        self.ticker.cv.notify_all();
                    }
                },
                Deferred::HwInvalidate => {
                    if let Some(con) = &self.con {
                        con.hw_invalidate();
                    }
                }
                Deferred::HwUpdate => {
                    if let Some(con) = &self.con {
                        con.hw_update_nowait();
                    }
                }
                Deferred::SetUiInfo(width, height) => {
                    if let Some(con) = &self.con {
                        con.set_ui_info(QemuUiInfo { width, height, ..Default::default() });
                    }
                }
                Deferred::XvpPowerdown(id) => {
                    if !self.hooks.powerdown() {
                        self.with_state(|st| client::xvp_fail(st, id));
                    }
                }
                Deferred::XvpReset(id) => {
                    if !self.hooks.reset() {
                        self.with_state(|st| client::xvp_fail(st, id));
                    }
                }
                Deferred::Keys(out) => kbd_state::send(&self.input, out),
                Deferred::Pointer { old, new, motion } => {
                    let con = self.con.as_ref();
                    self.input.update_buttons(con, &POINTER_BMAP, u32::from(old), u32::from(new));
                    match motion {
                        Motion::None => {}
                        Motion::Abs { x, y, width, height } => {
                            self.input.queue_abs(con, InputAxis::X, x, 0, width);
                            self.input.queue_abs(con, InputAxis::Y, y, 0, height);
                        }
                        Motion::Rel { dx, dy } => {
                            self.input.queue_rel(con, InputAxis::X, dx);
                            self.input.queue_rel(con, InputAxis::Y, dy);
                        }
                    }
                    self.input.event_sync();
                }
            }
        }
    }

    /// `qemu_input_is_absolute()` for the display's console, as `vs->absolute` holds it.
    pub(crate) fn is_absolute(&self) -> i32 {
        i32::from(self.input.is_absolute(self.con.as_ref()))
    }

    /// `check_pointer_type_change()` for every client, on a mouse mode change.
    fn check_pointer_type_change(&self) {
        let absolute = self.is_absolute();
        self.with_state(|st| {
            for i in 0..st.clients.len() {
                client::check_pointer_type_change(st, i, absolute);
            }
        });
    }

    /// `kbd_leds()`: the guest's keyboard LEDs changed.
    fn kbd_leds(&self) {
        let ledstate = self.input.get_leds_mask(self.con.as_ref()) as u8;
        self.with_state(|st| {
            if ledstate == st.ledstate {
                return;
            }
            st.ledstate = ledstate;
            for i in 0..st.clients.len() {
                client::led_state_change(st, i);
            }
        });
    }

    /// Runs `f` on the guest surface.
    fn with_guest<R>(&self, f: impl FnOnce(Option<&DisplaySurface>) -> R) -> R {
        match &self.con {
            Some(con) => con.with_surface(|s| f(Some(s.unwrap_or(&self.fallback)))),
            None => f(Some(&self.fallback)),
        }
    }

    /// `vnc_dpy_switch()`.
    fn switch(&self) {
        self.with_state(|st| {
            let new = self.with_guest(|s| {
                s.map(|s| Guest { width: s.width(), height: s.height(), format: s.format() })
            });
            // vnc_check_pageflip()
            let pageflip = st.guest.is_some() && st.guest == new;
            st.abort_display_jobs();
            st.guest = new;
            let dims = st.guest_dims();
            if pageflip {
                set_area_dirty(&mut st.guest_dirty, dims, 0, 0, dims.0 as i64, dims.1 as i64);
                return;
            }
            st.update_server_surface();
            for i in 0..st.clients.len() {
                client::colordepth(st, i);
                client::desktop_resize(st, i);
                let c = &mut st.clients[i];
                c.dirty.clear_all();
                set_area_dirty(
                    &mut c.dirty,
                    dims,
                    0,
                    0,
                    vnc_width(dims.0) as i64,
                    vnc_height(dims.1) as i64,
                );
                c.update_throttle_offset();
            }
        });
    }

    /// `vnc_refresh()`.
    fn refresh(&self) {
        let idle = self.with_state(|st| {
            if st.clients.is_empty() {
                st.set_refresh(REFRESH_INTERVAL_MAX);
                return true;
            }
            false
        });
        if idle {
            return;
        }
        if let Some(con) = &self.con {
            con.hw_update_nowait();
        }
        self.with_state(|st| {
            let has_dirty = self.with_guest(|s| st.refresh_server_surface(s));
            let ids: Vec<u64> = st.clients.iter().map(|c| c.id).collect();
            let mut rects = 0;
            for id in ids {
                if let Some(i) = st.client_index(id) {
                    rects += client::update_client(self, st, i, has_dirty);
                }
            }
            let mut interval = st.interval;
            if has_dirty != 0 && rects != 0 {
                interval = (interval / 2).max(REFRESH_INTERVAL_BASE);
            } else {
                interval = (interval + REFRESH_INTERVAL_INC).min(REFRESH_INTERVAL_MAX);
            }
            st.set_refresh(interval);
        });
    }

    /// The refresh timer of a display without a console, where the console's own timer would
    /// otherwise call [`VncDisplay::refresh`].
    fn start_refresh_thread(&self) {
        let me = self.me.clone();
        let spawned = std::thread::Builder::new().name("vnc-refresh".into()).spawn(move || {
            loop {
                let Some(vd) = me.upgrade() else { return };
                let interval = lock(&vd.state).interval.max(1);
                {
                    let kicked = lock(&vd.ticker.kicked);
                    let (mut kicked, _) = vd
                        .ticker
                        .cv
                        .wait_timeout_while(kicked, Duration::from_millis(interval), |k| !*k)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    *kicked = false;
                }
                vd.refresh();
            }
        });
        if let Err(e) = spawned {
            ruvm_base::report::error_report(&format!("vnc: cannot start the refresh thread: {e}"));
        }
    }

    /// `vnc_server_info_get()`: the first listening socket.
    pub(crate) fn server_info(&self) -> Option<VncServerInfo> {
        let a = self.listeners.get()?.first()?;
        Some(VncServerInfo {
            host: a.host.clone(),
            service: a.service.clone(),
            family: a.family,
            websocket: false,
            auth: Some(self.cfg.auth.name().to_string()),
        })
    }

    /// Starts accepting clients on `listeners`.
    pub(crate) fn listen(&self, listeners: Vec<Listener>) {
        let infos: Vec<AddrInfo> = listeners.iter().map(Listener::info).collect();
        let _ = self.listeners.set(infos);
        for l in listeners {
            net::spawn_acceptor(self.me.clone(), l);
        }
    }

    /// `vnc_connect()`.
    pub(crate) fn connect(&self, stream: net::Stream, peer: AddrInfo) {
        let id = self.next_client.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Some(io) = net::ClientIo::start(stream) else { return };
        self.with_state(|st| client::connect(self, st, id, Arc::clone(&io), peer));
        net::spawn_reader(self.me.clone(), id, io);
    }

    /// Bytes from a client, `vnc_client_read()`. False once the client is gone.
    pub(crate) fn client_input(&self, id: u64, data: &[u8]) -> bool {
        self.with_state(|st| client::input(self, st, id, data))
    }

    /// The client's socket closed or failed.
    pub(crate) fn client_gone(&self, id: u64) {
        self.with_state(|st| {
            if let Some(i) = st.client_index(id) {
                client::disconnect_start(st, i);
                client::disconnect_finish(self, st, i);
            }
        });
    }

    /// The `VncClientInfo` list of `qmp_query_client_list()`, newest client first.
    fn client_list(&self) -> Vec<VncClientInfo> {
        let st = lock(&self.state);
        st.clients.iter().rev().map(|c| c.info.client_info()).collect()
    }

    fn query_info(&self) -> VncInfo {
        let Some(a) = self.listeners.get().and_then(|l| l.first()) else {
            return VncInfo {
                enabled: false,
                host: None,
                family: None,
                service: None,
                auth: None,
                clients: None,
            };
        };
        VncInfo {
            enabled: true,
            host: Some(a.host.clone()),
            family: Some(a.family),
            service: Some(a.service.clone()),
            auth: Some(self.cfg.auth.name().to_string()),
            clients: Some(self.client_list()),
        }
    }

    fn query_info2(&self) -> VncInfo2 {
        let server = self
            .listeners
            .get()
            .map(|l| {
                l.iter()
                    .rev()
                    .map(|a| VncServerInfo2 {
                        host: a.host.clone(),
                        service: a.service.clone(),
                        family: a.family,
                        websocket: false,
                        auth: self.cfg.auth.primary(),
                        vencrypt: None,
                    })
                    .collect()
            })
            .unwrap_or_default();
        VncInfo2 {
            id: self.id.clone(),
            server,
            clients: self.client_list(),
            auth: self.cfg.auth.primary(),
            vencrypt: None,
            display: self.con.as_ref().and_then(|c| c.device()).and_then(|d| d.id),
        }
    }
}

impl DisplayChangeListener for VncDisplay {
    fn name(&self) -> &str {
        "vnc"
    }

    fn has_refresh(&self) -> bool {
        true
    }

    fn refresh(&self, _con: &QemuConsole) {
        VncDisplay::refresh(self);
    }

    /// `vnc_dpy_update()`.
    fn gfx_update(&self, _con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        let mut st = lock(&self.state);
        let dims = st.guest_dims();
        set_area_dirty(&mut st.guest_dirty, dims, x.into(), y.into(), w.into(), h.into());
    }

    fn gfx_switch(&self, _con: &QemuConsole) {
        self.switch();
    }

    fn gfx_check_format(&self, format: PixelFormat) -> Option<bool> {
        Some(pixman::pixman_check_format(format))
    }
}

/// Adds a display to `vnc_displays`.
pub(crate) fn register_display(vd: Arc<VncDisplay>) {
    lock(&DISPLAYS).push(vd);
}

/// `qmp_query_vnc()`.
pub fn query_vnc() -> Result<VncInfo> {
    Ok(match display_find(None) {
        Some(vd) => vd.query_info(),
        None => VncInfo {
            enabled: false,
            host: None,
            family: None,
            service: None,
            auth: None,
            clients: None,
        },
    })
}

/// `qmp_query_vnc_servers()`: the displays, last first.
pub fn query_vnc_servers() -> Result<Vec<VncInfo2>> {
    let displays: Vec<Arc<VncDisplay>> = lock(&DISPLAYS).clone();
    Ok(displays.iter().rev().map(|d| d.query_info2()).collect())
}

/// `vnc_display_password()`.
pub fn display_password(id: Option<&str>, password: &str) -> Result<()> {
    let Some(vd) = display_find(id) else {
        return Err(
            Error::generic("No VNC display is present").hint("To enable it, use '-vnc ...'")
        );
    };
    if vd.cfg.auth == Auth::None {
        return Err(Error::generic("VNC password authentication is disabled")
            .hint("To enable it, use '-vnc ...,password-secret=ID'"));
    }
    lock(&vd.state).password = Some(password.to_string());
    Ok(())
}

/// `vnc_display_pw_expire()`. False when there is no such display.
pub fn display_pw_expire(id: Option<&str>, expires: i64) -> bool {
    match display_find(id) {
        Some(vd) => {
            lock(&vd.state).expires = expires;
            true
        }
        None => false,
    }
}

/// `qemu_using_spice()` in a build without SPICE.
fn spice_not_in_use() -> Error {
    Error::new(ErrorClass::DeviceNotActive, "SPICE is not in use")
}

/// `qmp_set_password()`.
pub fn qmp_set_password(opts: SetPasswordOptions) -> Result<()> {
    match opts.u {
        SetPasswordOptionsU::Spice => Err(spice_not_in_use()),
        SetPasswordOptionsU::Vnc(vnc) => {
            if opts.connected.unwrap_or(SetPasswordAction::Keep) != SetPasswordAction::Keep {
                return Err(Error::generic(
                    "parameter 'connected' must be 'keep' when 'protocol' is 'vnc'",
                ));
            }
            display_password(vnc.display.as_deref(), &opts.password)
        }
    }
}

/// `qmp_expire_password()`.
pub fn qmp_expire_password(opts: ExpirePasswordOptions) -> Result<()> {
    let whenstr = opts.time.as_str();
    let (mut when, numstr) = match whenstr {
        "now" => (0i64, None),
        "never" => (i64::MAX, None),
        s => match s.strip_prefix('+') {
            Some(rest) => (now(), Some(rest)),
            None => (0, Some(s)),
        },
    };
    if let Some(numstr) = numstr {
        match ruvm_qapi::cutils::strtou64(numstr, 10, true) {
            Ok((num, _)) => when = when.wrapping_add(num as i64),
            Err(_) => {
                return Err(Error::generic(format!(
                    "Parameter 'time' doesn't take value '{whenstr}'"
                )));
            }
        }
    }
    match opts.u {
        ExpirePasswordOptionsU::Spice => Err(spice_not_in_use()),
        ExpirePasswordOptionsU::Vnc(vnc) => {
            if display_pw_expire(vnc.display.as_deref(), when) {
                Ok(())
            } else {
                Err(Error::generic("Could not set password expire time"))
            }
        }
    }
}

/// `qmp_change_vnc_password()`.
pub fn qmp_change_vnc_password(arg: ChangeVncPasswordArg) -> Result<()> {
    display_password(None, &arg.password)
}
