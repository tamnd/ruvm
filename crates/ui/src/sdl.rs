// SPDX-License-Identifier: GPL-2.0-or-later

//! `-display sdl`, QEMU's ui/sdl2.c, sdl2-2d.c and sdl2-input.c, over the `sdl2` crate and the
//! system's SDL2 library.
//!
//! SDL wants all of its calls on one thread, so [`init`] starts an `sdl` thread that owns the
//! library, the windows and the keyboard state of each console. The console listeners never
//! call SDL. They post to that thread, and on each refresh the timer asks the device for a frame
//! and wakes the thread, which draws what changed and then polls the SDL events, as
//! `sdl2_2d_refresh()` does on QEMU's main loop. Keys and pointer events go from the thread to
//! the input layer.
//!
//! Where this differs from QEMU:
//! - There is no OpenGL, so `gl` is accepted and ignored, as in a QEMU built without OpenGL.
//! - No display device defines a cursor sprite yet, so SDL never draws the guest cursor.
//! - There is no window icon, because ruvm does not install QEMU's icons.
//! - Text consoles do not exist, so every window shows a graphic console.
//! - The SDL thread is not the main thread, which SDL needs on macOS.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

use ruvm_qapi::types::{DisplayOptions, DisplayOptionsU, HotKeyMod, InputAxis, InputButton};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::{KeyboardUtil, Mod, Scancode};
use sdl2::mouse::{MouseButton, MouseUtil};
use sdl2::pixels::PixelFormatEnum;
use sdl2::render::{BlendMode, TextureCreator, WindowCanvas};
use sdl2::video::{FullscreenType, WindowContext};
use sdl2::{EventPump, Sdl, VideoSubsystem};

use crate::console::{
    DisplayChangeListener, DisplayState, GUI_REFRESH_INTERVAL_DEFAULT, ListenerId, QemuConsole,
    QemuUiInfo,
};
use crate::input::{InputState, usb_to_linux};
use crate::kbd_state::{self, KbdState};
use crate::pixman::{
    A8B8G8R8, A8R8G8B8, B8G8R8A8, B8G8R8X8, PixelFormat, R5G6B5, R8G8B8A8, R8G8B8X8, X1R5G5B5,
    X8B8G8R8, X8R8G8B8,
};

/// `SDL2_REFRESH_INTERVAL_BUSY`, in milliseconds.
const REFRESH_INTERVAL_BUSY: u64 = 10;
/// `SDL2_MAX_IDLE_COUNT`.
const MAX_IDLE_COUNT: u64 = 2 * GUI_REFRESH_INTERVAL_DEFAULT / REFRESH_INTERVAL_BUSY + 1;
/// The refresh interval of a minimized window, in milliseconds.
const REFRESH_INTERVAL_MINIMIZED: u64 = 500;

/// `SDL_WINDOW_INPUT_FOCUS`.
const WINDOW_INPUT_FOCUS: u32 = 0x200;

/// The `bmap` of `sdl_send_mouse_event()`: `SDL_BUTTON()` of each button.
const BUTTON_MAP: [(InputButton, u32); 5] = [
    (InputButton::Left, 1 << 0),
    (InputButton::Middle, 1 << 1),
    (InputButton::Right, 1 << 2),
    (InputButton::Side, 1 << 3),
    (InputButton::Extra, 1 << 4),
];

/// What the machine does for the SDL frontend.
pub trait Hooks: Send + Sync {
    /// `runstate_is_running()`, for the window title.
    fn is_running(&self) -> bool;

    /// `qemu_system_shutdown_request(SHUTDOWN_CAUSE_HOST_UI)`: the window was closed.
    fn close(&self);
}

/// What the listeners and the input layer post to the SDL thread.
enum Msg {
    Switch(usize),
    Update(usize, i32, i32, i32, i32),
    Refresh(usize),
    MouseSet(usize, i32, i32, bool),
    MouseMode,
}

/// The listener on one console, `dcl_2d_ops`.
struct SdlListener {
    idx: usize,
    tx: Sender<Msg>,
}

impl DisplayChangeListener for SdlListener {
    fn name(&self) -> &str {
        "sdl2-2d"
    }

    fn has_refresh(&self) -> bool {
        true
    }

    /// `sdl2_2d_refresh()`: the frame is asked for here, on the timer's thread, and the events
    /// are polled on SDL's.
    fn refresh(&self, con: &QemuConsole) {
        con.hw_update_nowait();
        let _ = self.tx.send(Msg::Refresh(self.idx));
    }

    fn gfx_update(&self, _con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        let _ = self.tx.send(Msg::Update(self.idx, x, y, w, h));
    }

    fn gfx_switch(&self, _con: &QemuConsole) {
        let _ = self.tx.send(Msg::Switch(self.idx));
    }

    /// `sdl2_2d_check_format()`.
    fn gfx_check_format(&self, format: PixelFormat) -> Option<bool> {
        Some(sdl_format(format).is_some())
    }

    fn mouse_set(&self, x: i32, y: i32, on: bool) {
        let _ = self.tx.send(Msg::MouseSet(self.idx, x, y, on));
    }
}

/// The SDL texture format of a surface format, from `sdl2_2d_switch()`.
fn sdl_format(format: PixelFormat) -> Option<PixelFormatEnum> {
    Some(match format {
        X1R5G5B5 => PixelFormatEnum::ARGB1555,
        R5G6B5 => PixelFormatEnum::RGB565,
        A8R8G8B8 | X8R8G8B8 => PixelFormatEnum::ARGB8888,
        A8B8G8R8 | X8B8G8R8 => PixelFormatEnum::ABGR8888,
        R8G8B8A8 | R8G8B8X8 => PixelFormatEnum::RGBA8888,
        B8G8R8X8 => PixelFormatEnum::BGRX8888,
        B8G8R8A8 => PixelFormatEnum::BGRA8888,
        _ => return None,
    })
}

/// The SDL thread's copy of a console's surface, which the texture is filled from.
struct Frame {
    width: u32,
    height: u32,
    stride: usize,
    format: PixelFormatEnum,
    data: Vec<u8>,
}

/// `struct sdl2_console`.
struct Output {
    con: QemuConsole,
    id: Option<ListenerId>,
    kbd: KbdState,
    canvas: Option<WindowCanvas>,
    creator: Option<TextureCreator<WindowContext>>,
    frame: Option<Frame>,
    dirty: bool,
    hidden: bool,
    last_vm_running: Option<bool>,
    idle_counter: u64,
    interval: u64,
    ignore_hotkeys: bool,
    gui_keysym: bool,
}

/// Everything the SDL thread owns: the statics of ui/sdl2.c and the consoles.
struct Ui {
    _sdl: Sdl,
    video: VideoSubsystem,
    pump: EventPump,
    mouse: MouseUtil,
    keyboard: KeyboardUtil,
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    hooks: Arc<dyn Hooks>,
    name: Option<String>,
    opts: DisplayOptions,
    outputs: Vec<Output>,
    gui_grab: bool,
    alt_grab: bool,
    ctrl_grab: bool,
    gui_saved_grab: bool,
    gui_fullscreen: bool,
    absolute_enabled: bool,
    guest_cursor: bool,
    guest_x: i32,
    guest_y: i32,
    prev_state: u32,
}

/// `sdl2_display_init()`: starts the SDL thread and returns once it has opened SDL, made a
/// window for each console that has a surface and registered its listeners. `name` is `-name`,
/// for the window title. The error is the exit status, after the message.
pub fn init(
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    opts: &DisplayOptions,
    name: Option<&str>,
    hooks: Arc<dyn Hooks>,
) -> Result<(), u8> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let opts = opts.clone();
    let name = name.map(str::to_string);
    let spawned = std::thread::Builder::new().name("sdl".into()).spawn(move || {
        let (tx, rx) = mpsc::channel();
        match Ui::open(ds, input, opts, name, hooks, &tx, &rx) {
            Ok(ui) => {
                let _ = ready_tx.send(Ok(()));
                ui.run(&rx);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    });
    let result = match spawned {
        Ok(_) => ready_rx.recv().unwrap_or_else(|_| Err("the SDL thread exited".into())),
        Err(e) => Err(e.to_string()),
    };
    result.map_err(|e| {
        eprintln!("Could not initialize SDL({e}) - exiting");
        1
    })
}

impl Ui {
    fn open(
        ds: Arc<DisplayState>,
        input: Arc<InputState>,
        opts: DisplayOptions,
        name: Option<String>,
        hooks: Arc<dyn Hooks>,
        tx: &Sender<Msg>,
        rx: &Receiver<Msg>,
    ) -> Result<Ui, String> {
        let sdl = sdl2::init()?;
        let video = sdl.video()?;
        sdl2::hint::set("SDL_VIDEO_X11_NET_WM_BYPASS_COMPOSITOR", "0");
        sdl2::hint::set("SDL_GRAB_KEYBOARD", "1");
        sdl2::hint::set("SDL_ALLOW_ALT_TAB_WHILE_GRABBED", "0");
        sdl2::hint::set("SDL_WINDOWS_NO_CLOSE_ON_ALT_F4", "1");
        video.enable_screen_saver();
        let pump = sdl.event_pump()?;
        let (alt_grab, ctrl_grab) = match &opts.u {
            DisplayOptionsU::Sdl(s) => match s.grab_mod {
                Some(HotKeyMod::LshiftLctrlLalt) => (true, false),
                Some(HotKeyMod::Rctrl) => (false, true),
                _ => (false, false),
            },
            _ => (false, false),
        };
        let mut ui = Ui {
            mouse: sdl.mouse(),
            keyboard: sdl.keyboard(),
            _sdl: sdl,
            video,
            pump,
            ds: Arc::clone(&ds),
            input,
            hooks,
            name,
            gui_fullscreen: opts.full_screen == Some(true),
            opts,
            outputs: Vec::new(),
            gui_grab: false,
            alt_grab,
            ctrl_grab,
            gui_saved_grab: false,
            absolute_enabled: false,
            guest_cursor: false,
            guest_x: 0,
            guest_y: 0,
            prev_state: 0,
        };
        let mut index = 0;
        while let Some(con) = ds.lookup_by_index(index) {
            ui.outputs.push(Output {
                hidden: !con.is_graphic() && index != 0,
                kbd: KbdState::new(Some(con.clone())),
                con,
                id: None,
                canvas: None,
                creator: None,
                frame: None,
                dirty: false,
                last_vm_running: None,
                idle_counter: 0,
                interval: 0,
                ignore_hotkeys: false,
                gui_keysym: false,
            });
            index += 1;
        }
        if ui.outputs.is_empty() {
            return Ok(ui);
        }
        for i in 0..ui.outputs.len() {
            let ops = Arc::new(SdlListener { idx: i, tx: tx.clone() });
            let id = ds.register_listener(&ui.outputs[i].con, ops);
            ui.outputs[i].id = Some(id);
            // qemu_console_register_listener() makes the window at once in QEMU.
            ui.drain(rx);
        }
        let mode_tx = tx.clone();
        ui.input.add_mouse_mode_notifier(move || {
            let _ = mode_tx.send(Msg::MouseMode);
        });
        if ui.gui_fullscreen {
            ui.grab_start(0);
        }
        Ok(ui)
    }

    /// The thread's loop: every message, then the drawing it asked for, then the events.
    fn run(mut self, rx: &Receiver<Msg>) {
        while let Ok(msg) = rx.recv() {
            let mut refresh = Vec::new();
            for msg in std::iter::once(msg).chain(std::iter::from_fn(|| rx.try_recv().ok())) {
                match msg {
                    Msg::Refresh(i) => {
                        if !refresh.contains(&i) {
                            refresh.push(i);
                        }
                    }
                    msg => self.handle(msg),
                }
            }
            self.draw_dirty();
            for i in refresh {
                self.poll_events(i);
            }
        }
    }

    /// The messages that are already queued, with the drawing they ask for.
    fn drain(&mut self, rx: &Receiver<Msg>) {
        while let Ok(msg) = rx.try_recv() {
            if !matches!(msg, Msg::Refresh(_)) {
                self.handle(msg);
            }
        }
        self.draw_dirty();
    }

    fn handle(&mut self, msg: Msg) {
        match msg {
            Msg::Switch(i) => self.switch(i),
            Msg::Update(i, x, y, w, h) => self.update(i, x, y, w, h),
            Msg::Refresh(_) => {}
            Msg::MouseSet(i, x, y, on) => self.mouse_warp(i, x, y, on),
            Msg::MouseMode => self.mouse_mode_change(),
        }
    }

    fn draw_dirty(&mut self) {
        for i in 0..self.outputs.len() {
            if std::mem::take(&mut self.outputs[i].dirty) {
                self.draw(i);
            }
        }
    }

    /// `sdl2_window_create()`.
    fn window_create(&mut self, i: usize, width: u32, height: u32) {
        let o = &mut self.outputs[i];
        let mut builder = self.video.window("", width, height);
        if self.gui_fullscreen {
            builder.fullscreen_desktop();
        } else {
            builder.resizable();
        }
        if o.hidden {
            builder.hidden();
        }
        let canvas = match builder
            .build()
            .map_err(|e| e.to_string())
            .and_then(|w| w.into_canvas().build().map_err(|e| e.to_string()))
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("sdl: cannot create the window of console {i}: {e}");
                return;
            }
        };
        o.creator = Some(canvas.texture_creator());
        o.canvas = Some(canvas);
        self.update_caption(i);
    }

    /// `sdl2_window_destroy()`.
    fn window_destroy(&mut self, i: usize) {
        let o = &mut self.outputs[i];
        o.creator = None;
        o.canvas = None;
    }

    /// `sdl2_2d_switch()`: a new surface, which may make, drop or resize the window.
    fn switch(&mut self, i: usize) {
        let con = self.outputs[i].con.clone();
        let Some((width, height, format, placeholder)) = con.with_surface(|s| {
            s.map(|s| (s.width() as u32, s.height() as u32, s.format(), s.is_placeholder()))
        }) else {
            return;
        };
        let old = self.outputs[i].frame.take();
        if placeholder && con.index() != 0 {
            self.window_destroy(i);
            return;
        }
        if self.outputs[i].canvas.is_none() {
            self.window_create(i, width, height);
        } else if old.is_some_and(|f| (f.width, f.height) != (width, height)) {
            self.window_resize(i, width, height);
        }
        let o = &mut self.outputs[i];
        let Some(canvas) = o.canvas.as_mut() else { return };
        let _ = canvas.set_logical_size(width, height);
        let Some(format) = sdl_format(format) else { return };
        let mut frame = Frame { width, height, stride: 0, format, data: Vec::new() };
        con.with_surface(|s| {
            if let Some(s) = s {
                frame.stride = s.stride();
                frame.data.extend_from_slice(s.data());
            }
        });
        o.frame = Some(frame);
        o.dirty = true;
    }

    /// `sdl2_window_resize()`.
    fn window_resize(&mut self, i: usize, width: u32, height: u32) {
        if let Some(canvas) = self.outputs[i].canvas.as_mut() {
            let _ = canvas.window_mut().set_size(width, height);
        }
    }

    /// `sdl2_2d_update()`, first half: the changed rectangle is copied from the surface.
    fn update(&mut self, i: usize, x: i32, y: i32, w: i32, h: i32) {
        let o = &mut self.outputs[i];
        let Some(frame) = o.frame.as_mut() else { return };
        let copied = o.con.with_surface(|s| {
            let Some(s) = s else { return false };
            if (s.width() as u32, s.height() as u32) != (frame.width, frame.height)
                || s.stride() != frame.stride
                || s.data().len() != frame.data.len()
            {
                // A switch is on its way.
                return false;
            }
            let bpp = s.bytes_per_pixel();
            let (x, y) = (x.max(0) as usize, y.max(0) as usize);
            let (w, h) = (w.max(0) as usize, h.max(0) as usize);
            let w = w.min(s.width().saturating_sub(x));
            let h = h.min(s.height().saturating_sub(y));
            for row in y..y + h {
                let start = row * frame.stride + x * bpp;
                let end = start + w * bpp;
                frame.data[start..end].copy_from_slice(&s.data()[start..end]);
            }
            true
        });
        o.dirty |= copied;
    }

    /// `sdl2_2d_update()`, second half, for the whole surface: `sdl2_2d_redraw()`.
    fn draw(&mut self, i: usize) {
        let o = &mut self.outputs[i];
        let (Some(canvas), Some(creator), Some(frame)) =
            (o.canvas.as_mut(), o.creator.as_ref(), o.frame.as_ref())
        else {
            return;
        };
        let Ok(mut texture) =
            creator.create_texture_streaming(frame.format, frame.width, frame.height)
        else {
            return;
        };
        // The surfaces have no alpha worth blending, whatever their format says.
        texture.set_blend_mode(BlendMode::None);
        if texture.update(None, &frame.data, frame.stride).is_err() {
            return;
        }
        canvas.clear();
        let _ = canvas.copy(&texture, None, None);
        canvas.present();
    }

    /// `sdl_update_caption()`.
    fn update_caption(&mut self, i: usize) {
        let status = if !self.hooks.is_running() {
            " [Stopped]"
        } else if self.gui_grab {
            if self.alt_grab {
                if cfg!(target_os = "macos") {
                    " - Press ⌃⌥⇧G to exit grab"
                } else {
                    " - Press Ctrl-Alt-Shift-G to exit grab"
                }
            } else if self.ctrl_grab {
                " - Press Right-Ctrl-G to exit grab"
            } else if cfg!(target_os = "macos") {
                " - Press ⌃⌥G to exit grab"
            } else {
                " - Press Ctrl-Alt-G to exit grab"
            }
        } else {
            ""
        };
        let title = match &self.name {
            Some(name) => format!("QEMU ({name}-{i}){status}"),
            None => format!("QEMU{status}"),
        };
        if let Some(canvas) = self.outputs[i].canvas.as_mut() {
            let _ = canvas.window_mut().set_title(&title);
        }
    }

    fn is_absolute(&self, i: usize) -> bool {
        self.input.is_absolute(Some(&self.outputs[i].con))
    }

    fn show_cursor_opt(&self) -> bool {
        self.opts.show_cursor == Some(true)
    }

    /// `sdl_hide_cursor()`.
    fn hide_cursor(&mut self, i: usize) {
        if self.show_cursor_opt() {
            return;
        }
        self.mouse.show_cursor(false);
        if !self.is_absolute(i) {
            self.mouse.set_relative_mouse_mode(true);
        }
    }

    /// `sdl_show_cursor()`.
    fn show_cursor(&mut self, i: usize) {
        if self.show_cursor_opt() {
            return;
        }
        if !self.is_absolute(i) {
            self.mouse.set_relative_mouse_mode(false);
        }
        self.mouse.show_cursor(true);
    }

    /// `sdl_grab_start()`.
    fn grab_start(&mut self, i: usize) {
        let Some(o) = self.outputs.get(i) else { return };
        if !o.con.is_graphic() {
            return;
        }
        // An inactive window does not take the grab, which would block SDL.
        let Some(canvas) = o.canvas.as_ref() else { return };
        if canvas.window().window_flags() & WINDOW_INPUT_FOCUS == 0 {
            return;
        }
        if self.guest_cursor {
            if !self.is_absolute(i) && !self.absolute_enabled {
                self.mouse.warp_mouse_in_window(canvas.window(), self.guest_x, self.guest_y);
            }
        } else {
            self.hide_cursor(i);
        }
        if let Some(canvas) = self.outputs[i].canvas.as_mut() {
            canvas.window_mut().set_grab(true);
        }
        self.gui_grab = true;
        self.update_caption(i);
    }

    /// `sdl_grab_end()`.
    fn grab_end(&mut self, i: usize) {
        if let Some(canvas) = self.outputs[i].canvas.as_mut() {
            canvas.window_mut().set_grab(false);
        }
        self.gui_grab = false;
        self.show_cursor(i);
        self.update_caption(i);
    }

    fn window_size(&self, i: usize) -> (i32, i32) {
        let (w, h) = self.outputs[i].canvas.as_ref().map_or((1, 1), |c| c.window().size());
        (w.max(1) as i32, h.max(1) as i32)
    }

    fn surface_size(&self, i: usize) -> (i32, i32) {
        let con = &self.outputs[i].con;
        (con.width(0), con.height(0))
    }

    /// `absolute_mouse_grab()`.
    fn absolute_mouse_grab(&mut self, i: usize) {
        let state = self.pump.mouse_state();
        let (scr_w, scr_h) = self.window_size(i);
        let (x, y) = (state.x(), state.y());
        if x > 0 && x < scr_w - 1 && y > 0 && y < scr_h - 1 {
            self.grab_start(i);
        }
    }

    /// `sdl_mouse_mode_change()`.
    fn mouse_mode_change(&mut self) {
        if self.outputs.is_empty() {
            return;
        }
        if self.is_absolute(0) {
            if !self.absolute_enabled {
                self.absolute_enabled = true;
                self.mouse.set_relative_mouse_mode(false);
                self.absolute_mouse_grab(0);
            }
        } else if self.absolute_enabled {
            if !self.gui_fullscreen {
                self.grab_end(0);
            }
            self.absolute_enabled = false;
        }
    }

    /// `sdl_send_mouse_event()`.
    fn send_mouse_event(&mut self, i: usize, dx: i32, dy: i32, x: i32, y: i32, state: u32) {
        let con = self.outputs[i].con.clone();
        if self.prev_state != state {
            self.input.update_buttons(Some(&con), &BUTTON_MAP, self.prev_state, state);
            self.prev_state = state;
        }
        if self.is_absolute(i) {
            let (w, h) = self.surface_size(i);
            self.input.queue_abs(Some(&con), InputAxis::X, x, 0, w);
            self.input.queue_abs(Some(&con), InputAxis::Y, y, 0, h);
        } else {
            let (mut dx, mut dy) = (dx, dy);
            if self.guest_cursor {
                let (x, y) = (x - self.guest_x, y - self.guest_y);
                self.guest_x += x;
                self.guest_y += y;
                dx = x;
                dy = y;
            }
            self.input.queue_rel(Some(&con), InputAxis::X, dx.into());
            self.input.queue_rel(Some(&con), InputAxis::Y, dy.into());
        }
        self.input.event_sync();
    }

    /// `toggle_full_screen()`.
    fn toggle_full_screen(&mut self, i: usize) {
        self.gui_fullscreen = !self.gui_fullscreen;
        if self.gui_fullscreen {
            if let Some(canvas) = self.outputs[i].canvas.as_mut() {
                let _ = canvas.window_mut().set_fullscreen(FullscreenType::Desktop);
            }
            self.gui_saved_grab = self.gui_grab;
            self.grab_start(i);
        } else {
            if !self.gui_saved_grab {
                self.grab_end(i);
            }
            if let Some(canvas) = self.outputs[i].canvas.as_mut() {
                let _ = canvas.window_mut().set_fullscreen(FullscreenType::Off);
            }
        }
        self.draw(i);
    }

    /// `get_mod_state()`: whether the grab modifiers are held.
    fn get_mod_state(&self) -> bool {
        let m = self.keyboard.mod_state();
        let code = Mod::LALTMOD | Mod::LCTRLMOD;
        if self.alt_grab {
            m & (code | Mod::LSHIFTMOD) == code | Mod::LSHIFTMOD
        } else if self.ctrl_grab {
            m & Mod::RCTRLMOD == Mod::RCTRLMOD
        } else {
            m & code == code
        }
    }

    fn find(&self, window_id: u32) -> Option<usize> {
        self.outputs
            .iter()
            .position(|o| o.canvas.as_ref().is_some_and(|c| c.window().id() == window_id))
    }

    /// `sdl2_process_key()`.
    fn process_key(&mut self, i: usize, scancode: Scancode, down: bool) {
        let Some(lnx) = usb_to_linux(scancode as i32 as u32) else { return };
        let mut out = Vec::new();
        self.outputs[i].kbd.key_event(lnx, down, &mut out);
        kbd_state::send(&self.input, out);
    }

    /// `sdl2_release_modifiers()`.
    fn release_modifiers(&mut self, i: usize) {
        let mut out = Vec::new();
        self.outputs[i].kbd.lift_all_keys(&mut out);
        kbd_state::send(&self.input, out);
    }

    /// `handle_keydown()`.
    fn handle_keydown(&mut self, window_id: u32, scancode: Scancode, repeat: bool) {
        let Some(i) = self.find(window_id) else { return };
        let modifier = self.get_mod_state();
        self.outputs[i].gui_keysym = false;
        if !self.outputs[i].ignore_hotkeys && modifier && !repeat {
            match scancode {
                Scancode::Num2
                | Scancode::Num3
                | Scancode::Num4
                | Scancode::Num5
                | Scancode::Num6
                | Scancode::Num7
                | Scancode::Num8
                | Scancode::Num9 => {
                    if self.gui_grab {
                        self.grab_end(i);
                    }
                    let win = (scancode as i32 - Scancode::Num1 as i32) as usize;
                    if win < self.outputs.len() {
                        let o = &mut self.outputs[win];
                        o.hidden = !o.hidden;
                        let hidden = o.hidden;
                        if let Some(canvas) = o.canvas.as_mut() {
                            if hidden {
                                canvas.window_mut().hide();
                            } else {
                                canvas.window_mut().show();
                            }
                        }
                        self.release_modifiers(i);
                        self.outputs[i].gui_keysym = true;
                    }
                }
                Scancode::F => {
                    self.toggle_full_screen(i);
                    self.outputs[i].gui_keysym = true;
                }
                Scancode::G => {
                    self.outputs[i].gui_keysym = true;
                    if !self.gui_grab {
                        self.grab_start(i);
                    } else if !self.gui_fullscreen {
                        self.grab_end(i);
                    }
                }
                Scancode::Num0 => {
                    let (w, h) = self.surface_size(i);
                    self.window_resize(i, w.max(0) as u32, h.max(0) as u32);
                    // Makes the texture again.
                    self.switch(i);
                    self.draw_dirty();
                    self.outputs[i].gui_keysym = true;
                }
                _ => {}
            }
        }
        if !self.outputs[i].gui_keysym {
            self.process_key(i, scancode, true);
        }
    }

    /// `handle_keyup()`.
    fn handle_keyup(&mut self, window_id: u32, scancode: Scancode) {
        let Some(i) = self.find(window_id) else { return };
        self.outputs[i].ignore_hotkeys = false;
        self.process_key(i, scancode, false);
    }

    /// `handle_mousemotion()`.
    fn handle_mousemotion(&mut self, window_id: u32, state: u32, ev: (i32, i32, i32, i32)) {
        let Some(i) = self.find(window_id) else { return };
        if !self.outputs[i].con.is_graphic() {
            return;
        }
        let (mx, my, xrel, yrel) = ev;
        let (scr_w, scr_h) = self.window_size(i);
        if self.is_absolute(i) || self.absolute_enabled {
            let (max_x, max_y) = (scr_w - 1, scr_h - 1);
            if self.gui_grab
                && !self.gui_fullscreen
                && (mx == 0 || my == 0 || mx == max_x || my == max_y)
            {
                self.grab_end(i);
            }
            if !self.gui_grab && mx > 0 && mx < max_x && my > 0 && my < max_y {
                self.grab_start(i);
            }
        }
        let (surf_w, surf_h) = self.surface_size(i);
        let scale =
            |v: i32, surf: i32, scr: i32| (i64::from(v) * i64::from(surf) / i64::from(scr)) as i32;
        let (x, y) = (scale(mx, surf_w, scr_w), scale(my, surf_h, scr_h));
        let (dx, dy) = (scale(xrel, surf_w, scr_w), scale(yrel, surf_h, scr_h));
        if self.gui_grab || self.is_absolute(i) || self.absolute_enabled {
            self.send_mouse_event(i, dx, dy, x, y, state);
        }
    }

    /// `handle_mousebutton()`.
    fn handle_mousebutton(
        &mut self,
        window_id: u32,
        button: MouseButton,
        pos: (i32, i32),
        down: bool,
    ) {
        let mut state = self.pump.mouse_state().to_sdl_state();
        let Some(i) = self.find(window_id) else { return };
        if !self.outputs[i].con.is_graphic() {
            return;
        }
        let (scr_w, scr_h) = self.window_size(i);
        let (surf_w, surf_h) = self.surface_size(i);
        let x = (i64::from(pos.0) * i64::from(surf_w) / i64::from(scr_w)) as i32;
        let y = (i64::from(pos.1) * i64::from(surf_h) / i64::from(scr_h)) as i32;
        if !self.gui_grab && !self.is_absolute(i) {
            if !down && button == MouseButton::Left {
                // Start grabbing all events.
                self.grab_start(i);
            }
        } else {
            let bit = match button as u8 {
                0 => 0,
                b => 1u32 << (b - 1),
            };
            if down {
                state |= bit;
            } else {
                state &= !bit;
            }
            self.send_mouse_event(i, 0, 0, x, y, state);
        }
    }

    /// `handle_mousewheel()`.
    fn handle_mousewheel(&mut self, window_id: u32, x: i32, y: i32) {
        let Some(i) = self.find(window_id) else { return };
        if !self.outputs[i].con.is_graphic() {
            return;
        }
        let btn = if y > 0 {
            InputButton::WheelUp
        } else if y < 0 {
            InputButton::WheelDown
        } else if x < 0 {
            InputButton::WheelRight
        } else if x > 0 {
            InputButton::WheelLeft
        } else {
            return;
        };
        let con = self.outputs[i].con.clone();
        self.input.queue_btn(Some(&con), btn, true);
        self.input.event_sync();
        self.input.queue_btn(Some(&con), btn, false);
        self.input.event_sync();
    }

    fn set_refresh(&mut self, i: usize, interval: u64) {
        let o = &mut self.outputs[i];
        if o.interval != interval {
            o.interval = interval;
            if let Some(id) = o.id {
                self.ds.listener_set_refresh(id, interval);
            }
        }
    }

    /// The `window-close` option and the poweroff of `SDL_QUIT` and `SDL_WINDOWEVENT_CLOSE`.
    fn close(&self) {
        if self.opts.window_close != Some(false) {
            self.hooks.close();
        }
    }

    /// `handle_windowevent()`.
    fn handle_windowevent(&mut self, window_id: u32, ev: WindowEvent) {
        let Some(i) = self.find(window_id) else { return };
        match ev {
            WindowEvent::Resized(w, h) => {
                let info = QemuUiInfo {
                    width: w.max(0) as u32,
                    height: h.max(0) as u32,
                    ..Default::default()
                };
                self.outputs[i].con.set_ui_info(info);
                self.draw(i);
            }
            WindowEvent::Exposed => self.draw(i),
            WindowEvent::FocusGained | WindowEvent::Enter => {
                if !self.gui_grab && (self.is_absolute(i) || self.absolute_enabled) {
                    self.absolute_mouse_grab(i);
                }
                // A console window a hotkey opened gets the key down again when it takes the
                // focus, which would close it at once, so hotkeys wait for a key release.
                self.outputs[i].ignore_hotkeys = self.get_mod_state();
            }
            WindowEvent::FocusLost => {
                if self.gui_grab && !self.gui_fullscreen {
                    self.grab_end(i);
                }
            }
            WindowEvent::Restored => self.set_refresh(i, GUI_REFRESH_INTERVAL_DEFAULT),
            WindowEvent::Minimized => self.set_refresh(i, REFRESH_INTERVAL_MINIMIZED),
            WindowEvent::Close => {
                if self.outputs[i].con.is_graphic() {
                    self.close();
                } else {
                    let o = &mut self.outputs[i];
                    if let Some(canvas) = o.canvas.as_mut() {
                        canvas.window_mut().hide();
                    }
                    o.hidden = true;
                }
            }
            WindowEvent::Shown => self.outputs[i].hidden = false,
            WindowEvent::Hidden => self.outputs[i].hidden = true,
            _ => {}
        }
    }

    /// `sdl2_poll_events()`.
    fn poll_events(&mut self, i: usize) {
        if i >= self.outputs.len() {
            return;
        }
        let running = self.hooks.is_running();
        if self.outputs[i].last_vm_running != Some(running) {
            self.outputs[i].last_vm_running = Some(running);
            self.update_caption(i);
        }
        let mut idle = true;
        while let Some(ev) = self.pump.poll_event() {
            match ev {
                Event::KeyDown { window_id, scancode: Some(sc), repeat, .. } => {
                    idle = false;
                    self.handle_keydown(window_id, sc, repeat);
                }
                Event::KeyUp { window_id, scancode: Some(sc), .. } => {
                    idle = false;
                    self.handle_keyup(window_id, sc);
                }
                Event::TextInput { .. } => idle = false,
                Event::Quit { .. } => self.close(),
                Event::MouseMotion { window_id, mousestate, x, y, xrel, yrel, .. } => {
                    idle = false;
                    self.handle_mousemotion(
                        window_id,
                        mousestate.to_sdl_state(),
                        (x, y, xrel, yrel),
                    );
                }
                Event::MouseButtonDown { window_id, mouse_btn, x, y, .. } => {
                    idle = false;
                    self.handle_mousebutton(window_id, mouse_btn, (x, y), true);
                }
                Event::MouseButtonUp { window_id, mouse_btn, x, y, .. } => {
                    idle = false;
                    self.handle_mousebutton(window_id, mouse_btn, (x, y), false);
                }
                Event::MouseWheel { window_id, x, y, .. } => {
                    idle = false;
                    self.handle_mousewheel(window_id, x, y);
                }
                Event::Window { window_id, win_event, .. } => {
                    self.handle_windowevent(window_id, win_event);
                }
                _ => {}
            }
        }
        if idle {
            let o = &mut self.outputs[i];
            if o.idle_counter < MAX_IDLE_COUNT {
                o.idle_counter += 1;
                if o.idle_counter >= MAX_IDLE_COUNT {
                    self.set_refresh(i, GUI_REFRESH_INTERVAL_DEFAULT);
                }
            }
        } else {
            self.outputs[i].idle_counter = 0;
            self.set_refresh(i, REFRESH_INTERVAL_BUSY);
        }
    }

    /// `sdl_mouse_warp()`, the `dpy_mouse_set` callback.
    fn mouse_warp(&mut self, i: usize, x: i32, y: i32, on: bool) {
        if !self.outputs[i].con.is_graphic() {
            return;
        }
        if on {
            if !self.guest_cursor {
                self.show_cursor(i);
            }
            if (self.gui_grab || self.is_absolute(i) || self.absolute_enabled)
                && !self.is_absolute(i)
                && !self.absolute_enabled
            {
                if let Some(canvas) = self.outputs[i].canvas.as_ref() {
                    self.mouse.warp_mouse_in_window(canvas.window(), x, y);
                }
            }
        } else if self.gui_grab {
            self.hide_cursor(i);
        }
        self.guest_cursor = on;
        self.guest_x = x;
        self.guest_y = y;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_are_the_ones_sdl2_2d_takes() {
        assert_eq!(sdl_format(X8R8G8B8), Some(PixelFormatEnum::ARGB8888));
        assert_eq!(sdl_format(R5G6B5), Some(PixelFormatEnum::RGB565));
        assert_eq!(sdl_format(B8G8R8X8), Some(PixelFormatEnum::BGRX8888));
        assert_eq!(sdl_format(crate::pixman::R8G8B8), None);
        assert_eq!(MAX_IDLE_COUNT, 7);
    }
}
