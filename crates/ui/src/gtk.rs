// SPDX-License-Identifier: GPL-2.0-or-later

//! `-display gtk`, QEMU's ui/gtk.c, over the `gtk4` crate and the system's GTK 4 library.
//!
//! GTK wants all of its calls on one thread, so [`init`] starts a `gtk` thread that opens GTK,
//! builds the window and runs the GLib main loop. The console listeners never call GTK. They
//! post to that thread and wake its main loop, and on each refresh the timer asks the device for
//! a frame. Keys and pointer events go from the thread to the input layer.
//!
//! Where this differs from QEMU:
//! - QEMU's window is GTK 3. GTK 4 has no menu widgets, so the menus are a menu model with the
//!   same items, mnemonics and shortcuts, and separators are menu sections.
//! - GTK 4 can neither grab nor warp the pointer. `Grab Input` takes the keyboard by inhibiting
//!   the system shortcuts and hides the cursor, but the pointer is not confined to the window, it
//!   is not moved back when the grab ends, and `dpy_mouse_set` does not move it. A pointer in
//!   relative mode still only moves the guest's while it is over the window.
//! - There is no OpenGL, so `gl` is accepted and ignored, as in a QEMU built without OpenGL.
//! - There are no text consoles, so no VTE tabs and no `Copy` item, and `clipboard` is accepted
//!   and ignored, as in a QEMU built without the GTK clipboard.
//! - `Power Down` is insensitive, because no machine here has a powerdown request yet.
//! - Keycodes are taken to be evdev ones, as on Wayland and X servers with evdev keycodes.
//!   QEMU's tables for the older XFree86 keycodes are not here.
//! - No display device defines a cursor sprite yet, so the guest cursor is never drawn, and
//!   consoles are not added or removed after the window opens.
//! - The window has no icon and the strings are not translated, because ruvm does not install
//!   QEMU's icons and message catalogs.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use gtk4::prelude::*;
use gtk4::{cairo, gdk, gio, glib};
use ruvm_qapi::types::{DisplayOptions, DisplayOptionsU, InputAxis, InputButton};

use crate::console::{
    DisplayChangeListener, DisplayState, GUI_REFRESH_INTERVAL_DEFAULT, ListenerId, QemuConsole,
};
use crate::input::InputState;
use crate::kbd_state::{self, KbdState};
use crate::pixman::{Image, PixelFormat, X8R8G8B8, pixman_check_format};

/// `VC_WINDOW_X_MIN` and `VC_WINDOW_Y_MIN`.
const WINDOW_X_MIN: i32 = 320;
const WINDOW_Y_MIN: i32 = 240;
/// `VC_SCALE_MIN`, `VC_SCALE_MAX` and `VC_SCALE_STEP`.
const SCALE_MIN: f64 = 0.25;
const SCALE_MAX: f64 = 4.0;
const SCALE_STEP: f64 = 0.25;
/// `HOTKEY_MODIFIERS`, as a shortcut trigger prefix.
const HOTKEY: &str = "<Control><Alt>";
/// `KEY_PAUSE`.
const KEY_PAUSE: u32 = 119;
/// `KEY_RESERVED`.
const KEY_RESERVED: u32 = 0;

/// What the machine does for the GTK frontend, the QMP commands of the `Machine` menu.
pub trait Hooks: Send + Sync {
    /// `runstate_is_running()`.
    fn is_running(&self) -> bool;

    /// `qmp_stop()`.
    fn stop(&self);

    /// `qmp_cont()`, whose error the menu drops.
    fn cont(&self);

    /// Whether [`Hooks::reset`] does anything.
    fn can_reset(&self) -> bool;

    /// `qmp_system_reset()`.
    fn reset(&self);

    /// Whether [`Hooks::powerdown`] does anything.
    fn can_powerdown(&self) -> bool;

    /// `qmp_system_powerdown()`.
    fn powerdown(&self);

    /// `qmp_quit()`.
    fn quit(&self);
}

/// What the listeners and the input layer post to the GTK thread.
enum Msg {
    Switch(usize),
    Update(usize, i32, i32, i32, i32),
    Refresh(usize),
    MouseSet(usize, i32, i32),
    MouseMode,
}

/// The way into the GTK thread from the others: the message goes on the channel, and one idle
/// callback at a time drains it.
#[derive(Clone)]
struct Waker {
    tx: Sender<Msg>,
    pending: Arc<AtomicBool>,
}

impl Waker {
    fn send(&self, msg: Msg) {
        if self.tx.send(msg).is_err() {
            return;
        }
        if !self.pending.swap(true, Ordering::AcqRel) {
            let pending = Arc::clone(&self.pending);
            glib::idle_add_once(move || {
                pending.store(false, Ordering::Release);
                with_ui(Ui::drain);
            });
        }
    }
}

/// The listener on one console, `dcl_ops`.
struct GtkListener {
    idx: usize,
    waker: Waker,
}

impl DisplayChangeListener for GtkListener {
    fn name(&self) -> &str {
        "gtk"
    }

    fn has_refresh(&self) -> bool {
        true
    }

    /// `gd_refresh()`: the frame is asked for here, on the timer's thread.
    fn refresh(&self, con: &QemuConsole) {
        con.hw_update_nowait();
        self.waker.send(Msg::Refresh(self.idx));
    }

    fn gfx_update(&self, _con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        self.waker.send(Msg::Update(self.idx, x, y, w, h));
    }

    fn gfx_switch(&self, _con: &QemuConsole) {
        self.waker.send(Msg::Switch(self.idx));
    }

    /// `qemu_pixman_check_format()`.
    fn gfx_check_format(&self, format: PixelFormat) -> Option<bool> {
        Some(pixman_check_format(format))
    }

    fn mouse_set(&self, x: i32, y: i32, _on: bool) {
        self.waker.send(Msg::MouseSet(self.idx, x, y));
    }
}

/// The scaling state the draw function shares with the rest of the window.
#[derive(Default)]
struct Scaling {
    free_scale: Cell<bool>,
    full_screen: Cell<bool>,
    keep_aspect_ratio: Cell<bool>,
}

/// What a console's draw function reads: `vc->gfx` of ui/gtk.c without the widgets.
struct View {
    /// `vc->gfx.convert`, the surface as `CAIRO_FORMAT_RGB24`.
    image: Option<Image>,
    /// `vc->gfx.surface`, made again from `image` when that changed.
    surface: Option<cairo::ImageSurface>,
    dirty: bool,
    scale_x: f64,
    scale_y: f64,
    preferred_scale: f64,
}

impl View {
    fn size(&self) -> Option<(i32, i32)> {
        self.image.as_ref().map(|i| (i.width() as i32, i.height() as i32))
    }

    /// `gd_update_scale()`.
    fn update_scale(&mut self, scaling: &Scaling, ww: i32, wh: i32, fbw: i32, fbh: i32) {
        if scaling.full_screen.get() {
            self.scale_x = f64::from(ww) / f64::from(fbw);
            self.scale_y = f64::from(wh) / f64::from(fbh);
        } else if scaling.free_scale.get() {
            let sx = f64::from(ww) / f64::from(fbw);
            let sy = f64::from(wh) / f64::from(fbh);
            if scaling.keep_aspect_ratio.get() {
                self.scale_x = sx.min(sy);
                self.scale_y = self.scale_x;
            } else {
                self.scale_x = sx;
                self.scale_y = sy;
            }
        }
    }

    /// The cairo surface of `image`, copied again if the image changed since the last draw.
    fn cairo_surface(&mut self) -> Option<cairo::ImageSurface> {
        let image = self.image.as_ref()?;
        if !self.dirty {
            if let Some(s) = &self.surface {
                return Some(s.clone());
            }
        }
        let (w, h) = (image.width() as i32, image.height() as i32);
        let mut surface = match self.surface.take() {
            Some(s) if s.width() == w && s.height() == h => s,
            _ => cairo::ImageSurface::create(cairo::Format::Rgb24, w, h).ok()?,
        };
        if !copy_image(image, &mut surface) {
            // The old surface is still referenced by a pattern.
            surface = cairo::ImageSurface::create(cairo::Format::Rgb24, w, h).ok()?;
            if !copy_image(image, &mut surface) {
                return None;
            }
        }
        self.dirty = false;
        self.surface = Some(surface.clone());
        Some(surface)
    }
}

/// Copies the rows of `image` into `surface`, which has its size.
fn copy_image(image: &Image, surface: &mut cairo::ImageSurface) -> bool {
    let stride = surface.stride() as usize;
    let row_len = image.width() * 4;
    let Ok(mut data) = surface.data() else { return false };
    for y in 0..image.height() {
        data[y * stride..y * stride + row_len].copy_from_slice(&image.row(y)[..row_len]);
    }
    true
}

/// `gd_draw_event()`.
fn draw(view: &mut View, scaling: &Scaling, cr: &cairo::Context, ww: i32, wh: i32) {
    let Some((fbw, fbh)) = view.size() else { return };
    if fbw <= 0 || fbh <= 0 {
        return;
    }
    let Some(surface) = view.cairo_surface() else { return };
    view.update_scale(scaling, ww, wh, fbw, fbh);
    let ww_surface = (f64::from(fbw) * view.scale_x) as i32;
    let wh_surface = (f64::from(fbh) * view.scale_y) as i32;
    let wx_offset = if ww > ww_surface { (ww - ww_surface) / 2 } else { 0 };
    let wy_offset = if wh > wh_surface { (wh - wh_surface) / 2 } else { 0 };
    cr.rectangle(0.0, 0.0, f64::from(ww), f64::from(wh));
    // Drawn from right to left, the inner rectangle cuts a hole for the surface.
    cr.rectangle(
        f64::from(wx_offset + ww_surface),
        f64::from(wy_offset),
        -f64::from(ww_surface),
        f64::from(wh_surface),
    );
    let _ = cr.fill();
    cr.scale(view.scale_x, view.scale_y);
    let _ = cr.set_source_surface(
        &surface,
        f64::from(wx_offset) / view.scale_x,
        f64::from(wy_offset) / view.scale_y,
    );
    let _ = cr.paint();
}

/// The check items of the menus, as stateful boolean actions.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Check {
    Pause,
    ZoomFit,
    GrabOnHover,
    Grab,
    ShowTabs,
    ShowMenubar,
}

impl Check {
    fn name(self) -> &'static str {
        match self {
            Check::Pause => "pause",
            Check::ZoomFit => "zoom-fit",
            Check::GrabOnHover => "grab-on-hover",
            Check::Grab => "grab",
            Check::ShowTabs => "show-tabs",
            Check::ShowMenubar => "show-menubar",
        }
    }
}

/// The plain items of the menus.
#[derive(Clone, Copy)]
enum Item {
    Reset,
    Powerdown,
    Quit,
    FullScreen,
    ZoomIn,
    ZoomOut,
    ZoomFixed,
    Detach,
}

impl Item {
    fn name(self) -> &'static str {
        match self {
            Item::Reset => "reset",
            Item::Powerdown => "powerdown",
            Item::Quit => "quit",
            Item::FullScreen => "full-screen",
            Item::ZoomIn => "zoom-in",
            Item::ZoomOut => "zoom-out",
            Item::ZoomFixed => "zoom-fixed",
            Item::Detach => "untabify",
        }
    }
}

/// `VirtualConsole` of a graphic console.
struct Vc {
    con: QemuConsole,
    label: String,
    id: Option<ListenerId>,
    kbd: KbdState,
    area: gtk4::DrawingArea,
    view: Rc<RefCell<View>>,
    /// The window of a detached tab.
    window: Option<gtk4::Window>,
    interval: u64,
}

/// `GtkDisplayState`.
struct Ui {
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    hooks: Arc<dyn Hooks>,
    name: Option<String>,
    rx: Receiver<Msg>,
    window: gtk4::Window,
    menu_bar: gtk4::PopoverMenuBar,
    notebook: gtk4::Notebook,
    actions: gio::SimpleActionGroup,
    vc_menu: gio::Menu,
    vc_keys: Option<gtk4::ShortcutController>,
    vcs: Vec<Vc>,
    scaling: Rc<Scaling>,
    kbd_owner: Option<usize>,
    ptr_owner: Option<usize>,
    last_x: i32,
    last_y: i32,
    last_set: bool,
    last_running: Option<bool>,
    null_cursor: Option<gdk::Cursor>,
}

/// Work for the window, see [`with_ui`].
type Deferred = Box<dyn FnOnce(&mut Ui)>;

thread_local! {
    /// The window, which only the GTK thread has.
    static UI: RefCell<Option<Ui>> = const { RefCell::new(None) };
    /// The work that waits for the window to be free.
    static DEFERRED: RefCell<VecDeque<Deferred>> = const { RefCell::new(VecDeque::new()) };
}

/// Runs `f` on the window. GTK calls some handlers from inside the calls the window makes, as
/// QEMU's handlers do, and those run once the outer call is done.
fn with_ui(f: impl FnOnce(&mut Ui) + 'static) {
    DEFERRED.with(|d| d.borrow_mut().push_back(Box::new(f)));
    UI.with(|ui| {
        let Ok(mut guard) = ui.try_borrow_mut() else { return };
        let Some(ui) = guard.as_mut() else { return };
        while let Some(f) = DEFERRED.with(|d| d.borrow_mut().pop_front()) {
            f(ui);
        }
    });
}

/// `gtk_display_init()`, after `early_gtk_display_init()`: starts the GTK thread and returns
/// once it has opened GTK, built the window and registered the listeners. `name` is `-name`,
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
    let spawned = std::thread::Builder::new().name("gtk".into()).spawn(move || {
        // QEMU keeps the C locale.
        gtk4::disable_setlocale();
        if gtk4::init().is_err() {
            let _ = ready_tx.send(false);
            return;
        }
        Ui::open(ds, input, opts, name, hooks);
        let _ = ready_tx.send(true);
        glib::MainLoop::new(None, false).run();
    });
    let ready = spawned.is_ok() && ready_rx.recv().unwrap_or(false);
    if ready {
        Ok(())
    } else {
        eprintln!("gtk initialization failed");
        Err(1)
    }
}

/// A shortcut of `trigger` that runs `f` on the window, which says whether it took the key.
fn shortcut(trigger: &str, f: impl Fn(&mut Ui) -> bool + 'static) -> gtk4::Shortcut {
    let f = Rc::new(f);
    let action = gtk4::CallbackAction::new(move |_, _| {
        let taken = Rc::new(Cell::new(true));
        let (f, t) = (Rc::clone(&f), Rc::clone(&taken));
        with_ui(move |ui| t.set(f(ui)));
        if taken.get() { glib::Propagation::Stop } else { glib::Propagation::Proceed }
    });
    gtk4::Shortcut::new(gtk4::ShortcutTrigger::parse_string(trigger), Some(action))
}

/// A menu item with its action and the shortcut it shows.
fn menu_item(label: &str, action: &str, accel: Option<&str>) -> gio::MenuItem {
    let item = gio::MenuItem::new(Some(label), Some(&format!("win.{action}")));
    if let Some(accel) = accel {
        item.set_attribute_value("accel", Some(&accel.to_variant()));
    }
    item
}

impl Ui {
    fn open(
        ds: Arc<DisplayState>,
        input: Arc<InputState>,
        opts: DisplayOptions,
        name: Option<String>,
        hooks: Arc<dyn Hooks>,
    ) {
        let gtk_opts = match &opts.u {
            DisplayOptionsU::Gtk(g) => g.clone(),
            _ => Default::default(),
        };
        glib::set_prgname(Some("qemu"));
        let window = gtk4::Window::new();
        let vbox = gtk4::Box::new(gtk4::Orientation::Vertical, 0);
        let notebook = gtk4::Notebook::new();
        let null_cursor = if opts.show_cursor == Some(true) {
            None
        } else {
            gdk::Cursor::from_name("none", None)
        };
        let (tx, rx) = mpsc::channel();
        let waker = Waker { tx, pending: Arc::new(AtomicBool::new(false)) };
        window.set_icon_name(Some("qemu"));

        let actions = gio::SimpleActionGroup::new();
        let menu = gio::Menu::new();
        let machine = gio::Menu::new();
        let section = gio::Menu::new();
        section.append_item(&menu_item("_Pause", Check::Pause.name(), None));
        machine.append_section(None, &section);
        let section = gio::Menu::new();
        section.append_item(&menu_item("_Reset", Item::Reset.name(), None));
        section.append_item(&menu_item("Power _Down", Item::Powerdown.name(), None));
        machine.append_section(None, &section);
        let section = gio::Menu::new();
        section.append_item(&menu_item("_Quit", Item::Quit.name(), Some(&format!("{HOTKEY}q"))));
        machine.append_section(None, &section);
        menu.append_submenu(Some("_Machine"), &machine);

        let view = gio::Menu::new();
        let section = gio::Menu::new();
        section.append_item(&menu_item(
            "_Fullscreen",
            Item::FullScreen.name(),
            Some(&format!("{HOTKEY}f")),
        ));
        view.append_section(None, &section);
        let section = gio::Menu::new();
        section.append_item(&menu_item(
            "Zoom _In",
            Item::ZoomIn.name(),
            Some(&format!("{HOTKEY}plus")),
        ));
        section.append_item(&menu_item(
            "Zoom _Out",
            Item::ZoomOut.name(),
            Some(&format!("{HOTKEY}minus")),
        ));
        section.append_item(&menu_item(
            "Best _Fit",
            Item::ZoomFixed.name(),
            Some(&format!("{HOTKEY}0")),
        ));
        section.append_item(&menu_item("Zoom To _Fit", Check::ZoomFit.name(), None));
        view.append_section(None, &section);
        let section = gio::Menu::new();
        section.append_item(&menu_item("Grab On _Hover", Check::GrabOnHover.name(), None));
        section.append_item(&menu_item(
            "_Grab Input",
            Check::Grab.name(),
            Some(&format!("{HOTKEY}g")),
        ));
        view.append_section(None, &section);
        let vc_menu = gio::Menu::new();
        view.append_section(None, &vc_menu);
        let section = gio::Menu::new();
        section.append_item(&menu_item("Show _Tabs", Check::ShowTabs.name(), None));
        section.append_item(&menu_item("Detach Tab", Item::Detach.name(), None));
        section.append_item(&menu_item(
            "Show Menubar",
            Check::ShowMenubar.name(),
            Some(&format!("{HOTKEY}m")),
        ));
        view.append_section(None, &section);
        menu.append_submenu(Some("_View"), &view);
        let menu_bar = gtk4::PopoverMenuBar::from_model(Some(&menu));
        // QEMU turns off gtk-menu-bar-accel, the F10 that opens the menus.
        let controllers: Vec<_> = menu_bar
            .observe_controllers()
            .iter::<glib::Object>()
            .filter_map(|c| c.ok()?.downcast::<gtk4::ShortcutController>().ok())
            .collect();
        for c in controllers {
            menu_bar.remove_controller(&c);
        }

        let scaling = Rc::new(Scaling::default());
        let mut ui = Ui {
            ds: Arc::clone(&ds),
            input,
            hooks,
            name,
            rx,
            window,
            menu_bar,
            notebook,
            actions,
            vc_menu,
            vc_keys: None,
            vcs: Vec::new(),
            scaling,
            kbd_owner: None,
            ptr_owner: None,
            last_x: 0,
            last_y: 0,
            last_set: false,
            last_running: None,
            null_cursor,
        };
        ui.add_actions();

        // gd_create_menu_view(): the consoles.
        let mut preferred_scale = 1.0;
        if let Some(scale) = gtk_opts.scale {
            if (SCALE_MIN..=SCALE_MAX).contains(&scale) {
                preferred_scale = scale;
            } else {
                ruvm_base::error_report(&format!(
                    "Invalid scale value {scale:.6} given, being ignored"
                ));
            }
        }
        let mut zoom_to_fit = false;
        let mut index = 0;
        while let Some(con) = ds.lookup_by_index(index) {
            if con.ui_info_supported() {
                zoom_to_fit = true;
            }
            ui.add_gfx_console(con, preferred_scale);
            index += 1;
        }
        if let Some(z) = gtk_opts.zoom_to_fit {
            zoom_to_fit = z;
        }
        if zoom_to_fit {
            ui.set_state(Check::ZoomFit, true);
            ui.scaling.free_scale.set(true);
        }
        ui.scaling.keep_aspect_ratio.set(gtk_opts.keep_aspect_ratio != Some(false));
        ui.rebuild_vc_menu();
        ui.set_state(Check::ShowMenubar, gtk_opts.show_menubar != Some(false));

        ui.connect_signals(&opts);
        ui.notebook.set_show_tabs(false);
        ui.notebook.set_show_border(false);
        ui.update_caption();
        vbox.append(&ui.menu_bar);
        ui.notebook.set_vexpand(true);
        vbox.append(&ui.notebook);
        ui.window.set_child(Some(&vbox));

        // qemu_console_register_listener() switches to the surface at once in QEMU.
        for i in 0..ui.vcs.len() {
            let ops = Arc::new(GtkListener { idx: i, waker: waker.clone() });
            let id = ds.register_listener(&ui.vcs[i].con, ops);
            ui.vcs[i].id = Some(id);
            ui.vcs[i].interval = GUI_REFRESH_INTERVAL_DEFAULT;
        }
        ui.drain();
        let mode = waker.clone();
        ui.input.add_mouse_mode_notifier(move || mode.send(Msg::MouseMode));

        ui.window.present();
        if gtk_opts.show_menubar == Some(false) {
            ui.menu_bar.set_visible(false);
        }
        if ui.vcs.is_empty() {
            // gtk_widget_set_sensitive(s->view_menu, false).
            for name in ["full-screen", "zoom-in", "zoom-out", "zoom-fixed", "untabify"] {
                ui.enable(name, false);
            }
            for c in [Check::ZoomFit, Check::GrabOnHover, Check::Grab, Check::ShowTabs] {
                ui.enable(c.name(), false);
            }
            ui.enable(Check::ShowMenubar.name(), false);
        }
        if let Some(i) = ui.current() {
            ui.vcs[i].area.grab_focus();
        }
        if opts.full_screen == Some(true) {
            ui.menu_full_screen();
        }
        if gtk_opts.grab_on_hover == Some(true) {
            ui.activate(Check::GrabOnHover);
        }
        if gtk_opts.show_tabs == Some(true) {
            ui.activate(Check::ShowTabs);
        }
        UI.with(|u| *u.borrow_mut() = Some(ui));
        with_ui(|_| {});
    }

    fn action(&self, name: &str) -> Option<gio::SimpleAction> {
        self.actions.lookup_action(name)?.downcast::<gio::SimpleAction>().ok()
    }

    fn enable(&self, name: &str, on: bool) {
        if let Some(a) = self.action(name) {
            a.set_enabled(on);
        }
    }

    fn get(&self, c: Check) -> bool {
        self.action(c.name()).and_then(|a| a.state()).and_then(|s| s.get::<bool>()) == Some(true)
    }

    /// Sets a check item without its handler, as `external_pause_update` does for `Pause`.
    fn set_state(&self, c: Check, on: bool) {
        if let Some(a) = self.action(c.name()) {
            a.set_state(&on.to_variant());
        }
    }

    /// `gtk_check_menu_item_set_active()`: the handler runs if the item changes.
    fn set_active(&mut self, c: Check, on: bool) {
        if self.get(c) != on {
            self.set_state(c, on);
            self.toggled(c);
        }
    }

    /// `gtk_menu_item_activate()` of a check item.
    fn activate(&mut self, c: Check) {
        let on = !self.get(c);
        self.set_state(c, on);
        self.toggled(c);
    }

    /// The `activate` handler of a check item, `gd_connect_signals()`.
    fn toggled(&mut self, c: Check) {
        match c {
            Check::Pause => self.menu_pause(),
            Check::ZoomFit => self.menu_zoom_fit(),
            Check::GrabOnHover => {}
            Check::Grab => self.menu_grab_input(),
            Check::ShowTabs => self.menu_show_tabs(),
            Check::ShowMenubar => self.menu_show_menubar(),
        }
    }

    fn clicked(&mut self, item: Item) {
        match item {
            Item::Reset => self.hooks.reset(),
            Item::Powerdown => self.hooks.powerdown(),
            Item::Quit => self.hooks.quit(),
            Item::FullScreen => self.menu_full_screen(),
            Item::ZoomIn => self.menu_zoom_in(),
            Item::ZoomOut => self.menu_zoom_out(),
            Item::ZoomFixed => self.menu_zoom_fixed(),
            Item::Detach => self.menu_untabify(),
        }
    }

    fn add_actions(&mut self) {
        for c in [
            Check::Pause,
            Check::ZoomFit,
            Check::GrabOnHover,
            Check::Grab,
            Check::ShowTabs,
            Check::ShowMenubar,
        ] {
            let a = gio::SimpleAction::new_stateful(c.name(), None, &false.to_variant());
            a.connect_change_state(move |a, v| {
                let Some(on) = v.and_then(|v| v.get::<bool>()) else { return };
                a.set_state(&on.to_variant());
                with_ui(move |ui| ui.toggled(c));
            });
            self.actions.add_action(&a);
        }
        for item in [
            Item::Reset,
            Item::Powerdown,
            Item::Quit,
            Item::FullScreen,
            Item::ZoomIn,
            Item::ZoomOut,
            Item::ZoomFixed,
            Item::Detach,
        ] {
            let a = gio::SimpleAction::new(item.name(), None);
            a.connect_activate(move |_, _| with_ui(move |ui| ui.clicked(item)));
            self.actions.add_action(&a);
        }
        self.enable(Item::Reset.name(), self.hooks.can_reset());
        self.enable(Item::Powerdown.name(), self.hooks.can_powerdown());
        self.window.insert_action_group("win", Some(&self.actions));
    }

    /// The shortcuts of `accel_group` and the window's signals, `gd_connect_signals()`.
    fn connect_signals(&mut self, opts: &DisplayOptions) {
        let keys = gtk4::ShortcutController::new();
        keys.set_propagation_phase(gtk4::PropagationPhase::Capture);
        // The accel paths of the menu items only work while the menu bar is on screen.
        let path = |ui: &Ui| ui.menu_bar.is_drawable();
        keys.add_shortcut(shortcut(&format!("{HOTKEY}q"), move |ui| {
            path(ui) && {
                ui.clicked(Item::Quit);
                true
            }
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}f"), |ui| {
            ui.clicked(Item::FullScreen);
            true
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}plus"), move |ui| {
            path(ui) && {
                ui.clicked(Item::ZoomIn);
                true
            }
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}equal"), |ui| {
            ui.clicked(Item::ZoomIn);
            true
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}minus"), move |ui| {
            path(ui) && {
                ui.clicked(Item::ZoomOut);
                true
            }
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}0"), move |ui| {
            path(ui) && {
                ui.clicked(Item::ZoomFixed);
                true
            }
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}g"), move |ui| {
            let on = ui.action(Check::Grab.name()).is_some_and(|a| a.is_enabled());
            path(ui) && on && {
                ui.activate(Check::Grab);
                true
            }
        }));
        keys.add_shortcut(shortcut(&format!("{HOTKEY}m"), |ui| {
            ui.activate(Check::ShowMenubar);
            true
        }));
        self.window.add_controller(keys);

        let allow_close = opts.window_close != Some(false);
        self.window.connect_close_request(move |_| {
            if allow_close {
                with_ui(|ui| ui.hooks.quit());
            }
            glib::Propagation::Stop
        });
        self.notebook.connect_switch_page(|nb, _, page| {
            if nb.is_realized() {
                with_ui(move |ui| ui.change_page(page));
            }
        });
    }

    /// `add_gfx_console()`.
    fn add_gfx_console(&mut self, con: QemuConsole, preferred_scale: f64) {
        let i = self.vcs.len();
        let label = con.label();
        let area = gtk4::DrawingArea::new();
        area.set_focusable(true);
        area.set_hexpand(true);
        area.set_vexpand(true);
        let view = Rc::new(RefCell::new(View {
            image: None,
            surface: None,
            dirty: false,
            scale_x: preferred_scale,
            scale_y: preferred_scale,
            preferred_scale,
        }));
        let (v, scaling) = (Rc::clone(&view), Rc::clone(&self.scaling));
        area.set_draw_func(move |_, cr, w, h| draw(&mut v.borrow_mut(), &scaling, cr, w, h));
        self.notebook.append_page(&area, Some(&gtk4::Label::new(Some(&label))));
        self.vcs.push(Vc {
            kbd: KbdState::new(Some(con.clone())),
            con,
            label,
            id: None,
            area,
            view,
            window: None,
            interval: 0,
        });
        self.connect_vc_gfx_signals(i);
    }

    /// `gd_connect_vc_gfx_signals()`.
    fn connect_vc_gfx_signals(&self, i: usize) {
        let area = &self.vcs[i].area;
        let legacy = gtk4::EventControllerLegacy::new();
        legacy.connect_event(move |_, event| {
            let down = match event.event_type() {
                gdk::EventType::ButtonPress => true,
                gdk::EventType::ButtonRelease => false,
                gdk::EventType::Scroll => {
                    let Some(s) = event.downcast_ref::<gdk::ScrollEvent>() else {
                        return glib::Propagation::Proceed;
                    };
                    let (dir, deltas) = (s.direction(), s.deltas());
                    with_ui(move |ui| ui.scroll_event(i, dir, deltas));
                    return glib::Propagation::Stop;
                }
                _ => return glib::Propagation::Proceed,
            };
            let Some(b) = event.downcast_ref::<gdk::ButtonEvent>() else {
                return glib::Propagation::Proceed;
            };
            let button = b.button();
            with_ui(move |ui| ui.button_event(i, button, down));
            glib::Propagation::Stop
        });
        area.add_controller(legacy);

        let keys = gtk4::EventControllerKey::new();
        keys.connect_key_pressed(move |_, keyval, keycode, _| {
            with_ui(move |ui| ui.key_event(i, keyval, keycode, true));
            glib::Propagation::Stop
        });
        keys.connect_key_released(move |_, keyval, keycode, _| {
            with_ui(move |ui| ui.key_event(i, keyval, keycode, false));
        });
        area.add_controller(keys);

        let motion = gtk4::EventControllerMotion::new();
        motion.connect_motion(move |_, x, y| with_ui(move |ui| ui.motion_event(i, x, y)));
        motion.connect_enter(move |_, _, _| with_ui(move |ui| ui.enter_event(i)));
        motion.connect_leave(|_| with_ui(Ui::leave_event));
        area.add_controller(motion);

        let focus = gtk4::EventControllerFocus::new();
        focus.connect_leave(|_| with_ui(Ui::release_modifiers));
        area.add_controller(focus);

        area.connect_resize(move |_, w, h| with_ui(move |ui| ui.configure(i, w, h)));
    }

    /// The messages that are queued.
    fn drain(&mut self) {
        while let Ok(msg) = self.rx.try_recv() {
            match msg {
                Msg::Switch(i) => self.switch(i),
                Msg::Update(i, x, y, w, h) => self.update(i, x, y, w, h),
                Msg::Refresh(i) => self.refresh(i),
                Msg::MouseSet(i, x, y) => self.mouse_set(i, x, y),
                Msg::MouseMode => self.mouse_mode_change(),
            }
        }
    }

    /// `gd_switch()`.
    fn switch(&mut self, i: usize) {
        let vc = &self.vcs[i];
        let Some(image) = vc.con.with_surface(|s| {
            let s = s?;
            let mut image = Image::new(X8R8G8B8, s.width(), s.height(), 0);
            let (w, h) = (s.width() as i64, s.height() as i64);
            image.composite_src(s.image(), 0, 0, 0, 0, w, h);
            Some(image)
        }) else {
            return;
        };
        let resized = {
            let mut view = vc.view.borrow_mut();
            let old = view.size();
            let new = (image.width() as i32, image.height() as i32);
            view.image = Some(image);
            view.dirty = true;
            old != Some(new)
        };
        if resized {
            self.update_windowsize(i);
        } else {
            self.vcs[i].area.queue_draw();
        }
    }

    /// `gd_update()`.
    fn update(&mut self, i: usize, x: i32, y: i32, w: i32, h: i32) {
        let vc = &self.vcs[i];
        let mut view = vc.view.borrow_mut();
        let Some(image) = view.image.as_mut() else { return };
        vc.con.with_surface(|s| {
            if let Some(s) = s {
                let (x, y) = (i64::from(x), i64::from(y));
                image.composite_src(s.image(), x, y, x, y, i64::from(w), i64::from(h));
            }
        });
        view.dirty = true;
        vc.area.queue_draw();
    }

    /// The rest of `gd_refresh()`: the runstate the caption shows, which QEMU follows with a
    /// change state handler, and `gd_update_monitor_refresh_rate()`.
    fn refresh(&mut self, i: usize) {
        let running = self.hooks.is_running();
        if self.last_running != Some(running) {
            self.last_running = Some(running);
            self.update_caption();
        }
        let vc = &self.vcs[i];
        let top = vc.window.as_ref().unwrap_or(&self.window);
        let refresh_rate = top
            .surface()
            .and_then(|s| WidgetExt::display(top).monitor_at_surface(&s))
            .map_or(0, |m| m.refresh_rate());
        if vc.con.ui_info_supported() {
            let mut info = vc.con.ui_info();
            info.refresh_rate = refresh_rate.max(0) as u32;
            vc.con.set_ui_info(info);
        }
        let interval = if refresh_rate > 0 {
            (1_000_000 / refresh_rate as u64).min(GUI_REFRESH_INTERVAL_DEFAULT)
        } else {
            GUI_REFRESH_INTERVAL_DEFAULT
        };
        if vc.interval != interval {
            self.vcs[i].interval = interval;
            if let Some(id) = self.vcs[i].id {
                self.ds.listener_set_refresh(id, interval);
            }
        }
    }

    /// `gd_mouse_set()`. GTK 4 cannot move the pointer, so only the position is kept.
    fn mouse_set(&mut self, i: usize, x: i32, y: i32) {
        if self.is_absolute(i) {
            return;
        }
        self.last_x = x;
        self.last_y = y;
    }

    fn is_absolute(&self, i: usize) -> bool {
        self.input.is_absolute(Some(&self.vcs[i].con))
    }

    /// `gd_vc_find_current()`.
    fn current(&self) -> Option<usize> {
        let page = self.notebook.current_page()?;
        self.by_page(page)
    }

    /// `gd_vc_find_by_page()`.
    fn by_page(&self, page: u32) -> Option<usize> {
        let child = self.notebook.nth_page(Some(page))?;
        self.vcs.iter().position(|vc| vc.area.upcast_ref::<gtk4::Widget>() == &child)
    }

    /// The console of the checked `vc` radio item, `gd_vc_find_by_menu()`.
    fn by_menu(&self) -> Option<usize> {
        let a = self.action("vc-0")?;
        let i = a.state()?.get::<i32>()?;
        usize::try_from(i).ok().filter(|&i| i < self.vcs.len())
    }

    fn set_vc_state(&self, i: usize) {
        for n in 0..self.vcs.len() {
            if let Some(a) = self.action(&format!("vc-{n}")) {
                a.set_state(&(i as i32).to_variant());
            }
        }
    }

    /// `gd_update_caption()`.
    fn update_caption(&mut self) {
        let prefix = match &self.name {
            Some(name) => format!("QEMU ({name})"),
            None => "QEMU".to_string(),
        };
        let grab = match self.ptr_owner {
            Some(i) if self.vcs[i].window.is_none() => " - Press Ctrl+Alt+G to release grab",
            _ => "",
        };
        let paused = !self.hooks.is_running();
        let status = if paused { " [Paused]" } else { "" };
        self.set_state(Check::Pause, paused);
        self.window.set_title(Some(&format!("{prefix}{status}{grab}")));
        for (i, vc) in self.vcs.iter().enumerate() {
            let Some(w) = &vc.window else { continue };
            let kbd = if self.kbd_owner == Some(i) { " +kbd" } else { "" };
            let ptr = if self.ptr_owner == Some(i) { " +ptr" } else { "" };
            w.set_title(Some(&format!("{prefix}: {}{kbd}{ptr}", vc.label)));
        }
    }

    /// `gd_update_cursor()`.
    fn update_cursor(&self, i: usize) {
        let hide =
            self.scaling.full_screen.get() || self.is_absolute(i) || self.ptr_owner == Some(i);
        let area = &self.vcs[i].area;
        if hide {
            area.set_cursor(self.null_cursor.as_ref());
        } else {
            area.set_cursor(None);
        }
    }

    /// `gd_update_geometry_hints()`: the window cannot be smaller than the scaled surface, or
    /// than a quarter of it with zoom to fit.
    fn update_geometry_hints(&self, i: usize) {
        let vc = &self.vcs[i];
        let view = vc.view.borrow();
        let Some((fbw, fbh)) = view.size() else { return };
        let free = self.scaling.free_scale.get();
        let sx = if free { SCALE_MIN } else { view.scale_x };
        let sy = if free { SCALE_MIN } else { view.scale_y };
        vc.area.set_size_request((f64::from(fbw) * sx) as i32, (f64::from(fbh) * sy) as i32);
    }

    /// `gd_update_windowsize()`: without zoom to fit the window shrinks to the surface.
    fn update_windowsize(&mut self, i: usize) {
        self.update_geometry_hints(i);
        if !self.scaling.full_screen.get() && !self.scaling.free_scale.get() {
            let w = self.vcs[i].window.as_ref().unwrap_or(&self.window);
            w.set_default_size(WINDOW_X_MIN, WINDOW_Y_MIN);
        }
    }

    /// `gd_update_full_redraw()`.
    fn update_full_redraw(&self, i: usize) {
        self.vcs[i].area.queue_draw();
    }

    /// `gtk_release_modifiers()`.
    fn release_modifiers(&mut self) {
        let Some(i) = self.current() else { return };
        let mut out = Vec::new();
        self.vcs[i].kbd.lift_all_keys(&mut out);
        kbd_state::send(&self.input, out);
    }

    fn toplevel(&self, i: usize) -> Option<gdk::Toplevel> {
        let w = self.vcs[i].window.as_ref().unwrap_or(&self.window);
        w.surface()?.downcast::<gdk::Toplevel>().ok()
    }

    /// `gd_grab_update()`: GTK 4 takes the keyboard by inhibiting the system shortcuts, and has
    /// no pointer grab.
    fn grab_update(&self, i: usize, kbd: bool) {
        let Some(top) = self.toplevel(i) else { return };
        if kbd {
            top.inhibit_system_shortcuts(None::<&gdk::Event>);
        } else {
            top.restore_system_shortcuts();
        }
    }

    /// `gd_grab_keyboard()`.
    fn grab_keyboard(&mut self, i: usize) {
        if let Some(k) = self.kbd_owner {
            if k == i {
                return;
            }
            self.ungrab_keyboard();
        }
        self.grab_update(i, true);
        self.kbd_owner = Some(i);
        self.update_caption();
    }

    /// `gd_ungrab_keyboard()`.
    fn ungrab_keyboard(&mut self) {
        let Some(i) = self.kbd_owner.take() else { return };
        self.grab_update(i, false);
        self.update_caption();
    }

    /// `gd_grab_pointer()`.
    fn grab_pointer(&mut self, i: usize) {
        if let Some(p) = self.ptr_owner {
            if p == i {
                return;
            }
            self.ungrab_pointer();
        }
        self.ptr_owner = Some(i);
        self.update_caption();
    }

    /// `gd_ungrab_pointer()`.
    fn ungrab_pointer(&mut self) {
        if self.ptr_owner.take().is_none() {
            return;
        }
        self.update_caption();
    }

    /// `gd_menu_pause()`.
    fn menu_pause(&mut self) {
        if self.hooks.is_running() {
            self.hooks.stop();
        } else {
            self.hooks.cont();
        }
        self.last_running = Some(self.hooks.is_running());
        self.update_caption();
    }

    /// `gd_menu_switch_vc()`.
    fn menu_switch_vc(&mut self) {
        self.release_modifiers();
        let Some(i) = self.by_menu() else { return };
        if let Some(page) = self.notebook.page_num(&self.vcs[i].area) {
            self.notebook.set_current_page(Some(page));
        }
        self.vcs[i].area.grab_focus();
    }

    /// `gd_accel_switch_vc()`.
    fn accel_switch_vc(&mut self, i: usize) {
        if self.by_menu() != Some(i) {
            self.set_vc_state(i);
            self.menu_switch_vc();
        }
    }

    /// `gd_menu_show_tabs()`.
    fn menu_show_tabs(&mut self) {
        self.notebook.set_show_tabs(self.get(Check::ShowTabs));
        if let Some(i) = self.current() {
            self.update_windowsize(i);
        }
    }

    /// `gd_menu_show_menubar()`.
    fn menu_show_menubar(&mut self) {
        if self.scaling.full_screen.get() {
            return;
        }
        self.menu_bar.set_visible(self.get(Check::ShowMenubar));
        if let Some(i) = self.current() {
            self.update_windowsize(i);
        }
    }

    /// `gd_menu_full_screen()`.
    fn menu_full_screen(&mut self) {
        let Some(i) = self.current() else { return };
        if !self.scaling.full_screen.get() {
            self.notebook.set_show_tabs(false);
            self.menu_bar.set_visible(false);
            self.vcs[i].area.set_size_request(-1, -1);
            self.window.fullscreen();
            self.scaling.full_screen.set(true);
        } else {
            self.window.unfullscreen();
            self.menu_show_tabs();
            if self.get(Check::ShowMenubar) {
                self.menu_bar.set_visible(true);
            }
            self.scaling.full_screen.set(false);
            {
                let mut view = self.vcs[i].view.borrow_mut();
                view.scale_x = view.preferred_scale;
                view.scale_y = view.preferred_scale;
            }
            self.update_windowsize(i);
        }
        self.update_cursor(i);
    }

    /// `gd_menu_zoom_in()`.
    fn menu_zoom_in(&mut self) {
        let Some(i) = self.current() else { return };
        self.set_active(Check::ZoomFit, false);
        {
            let mut view = self.vcs[i].view.borrow_mut();
            view.scale_x += SCALE_STEP;
            view.scale_y += SCALE_STEP;
        }
        self.update_windowsize(i);
    }

    /// `gd_menu_zoom_out()`.
    fn menu_zoom_out(&mut self) {
        let Some(i) = self.current() else { return };
        self.set_active(Check::ZoomFit, false);
        {
            let mut view = self.vcs[i].view.borrow_mut();
            view.scale_x = (view.scale_x - SCALE_STEP).max(SCALE_MIN);
            view.scale_y = (view.scale_y - SCALE_STEP).max(SCALE_MIN);
        }
        self.update_windowsize(i);
    }

    /// `gd_menu_zoom_fixed()`.
    fn menu_zoom_fixed(&mut self) {
        let Some(i) = self.current() else { return };
        {
            let mut view = self.vcs[i].view.borrow_mut();
            view.scale_x = view.preferred_scale;
            view.scale_y = view.preferred_scale;
        }
        self.update_windowsize(i);
    }

    /// `gd_menu_zoom_fit()`.
    fn menu_zoom_fit(&mut self) {
        let Some(i) = self.current() else { return };
        if self.get(Check::ZoomFit) {
            self.scaling.free_scale.set(true);
        } else {
            self.scaling.free_scale.set(false);
            let mut view = self.vcs[i].view.borrow_mut();
            view.scale_x = view.preferred_scale;
            view.scale_y = view.preferred_scale;
        }
        self.update_windowsize(i);
        self.update_full_redraw(i);
    }

    /// `gd_menu_grab_input()`.
    fn menu_grab_input(&mut self) {
        let Some(i) = self.current() else { return };
        if self.get(Check::Grab) {
            self.grab_keyboard(i);
            self.grab_pointer(i);
        } else {
            self.ungrab_keyboard();
            self.ungrab_pointer();
        }
        self.update_cursor(i);
    }

    /// `gd_change_page()`.
    fn change_page(&mut self, page: u32) {
        let Some(i) = self.by_page(page) else { return };
        self.set_vc_state(i);
        // Every console is graphic, so the grab item stays sensitive.
        if self.scaling.full_screen.get() {
            self.set_active(Check::Grab, true);
        }
        self.enable(Check::Grab.name(), true);
        self.update_windowsize(i);
        self.update_cursor(i);
    }

    /// `gd_vc_notebook_pos()`: where a detached console goes back in the notebook.
    fn notebook_pos(&self, i: usize) -> u32 {
        self.vcs[..i].iter().filter(|vc| vc.window.is_none()).count() as u32
    }

    /// `gd_rebuild_vc_menu()`.
    fn rebuild_vc_menu(&mut self) {
        self.vc_menu.remove_all();
        if let Some(c) = self.vc_keys.take() {
            self.window.remove_controller(&c);
        }
        let keys = gtk4::ShortcutController::new();
        keys.set_propagation_phase(gtk4::PropagationPhase::Capture);
        let current = self.current().unwrap_or(0) as i32;
        let mut shortcut_idx = 0;
        for i in 0..self.vcs.len() {
            let name = format!("vc-{i}");
            if self.action(&name).is_none() {
                let a = gio::SimpleAction::new_stateful(
                    &name,
                    Some(glib::VariantTy::INT32),
                    &current.to_variant(),
                );
                a.connect_change_state(|a, v| {
                    let Some(n) = v.and_then(|v| v.get::<i32>()) else { return };
                    a.set_state(&n.to_variant());
                    with_ui(move |ui| {
                        ui.set_vc_state(n as usize);
                        ui.menu_switch_vc();
                    });
                });
                self.actions.add_action(&a);
            }
            let item = gio::MenuItem::new(Some(&self.vcs[i].label), None);
            item.set_action_and_target_value(
                Some(&format!("win.{name}")),
                Some(&(i as i32).to_variant()),
            );
            let detached = self.vcs[i].window.is_some();
            self.enable(&name, !detached);
            if !detached && shortcut_idx < 9 {
                let trigger = format!("{HOTKEY}{}", shortcut_idx + 1);
                item.set_attribute_value("accel", Some(&trigger.to_variant()));
                keys.add_shortcut(shortcut(&trigger, move |ui| {
                    ui.accel_switch_vc(i);
                    true
                }));
                shortcut_idx += 1;
            }
            self.vc_menu.append_item(&item);
        }
        self.window.add_controller(keys.clone());
        self.vc_keys = Some(keys);
        if let Some(i) = self.current() {
            self.set_vc_state(i);
        }
    }

    /// `gd_menu_untabify()`.
    fn menu_untabify(&mut self) {
        let Some(i) = self.current() else { return };
        self.set_active(Check::Grab, false);
        if self.vcs[i].window.is_some() {
            return;
        }
        let area = self.vcs[i].area.clone();
        let window = gtk4::Window::new();
        self.notebook.detach_tab(&area);
        window.set_child(Some(&area));
        window.connect_close_request(move |_| {
            with_ui(move |ui| ui.tab_window_close(i));
            glib::Propagation::Stop
        });
        let keys = gtk4::ShortcutController::new();
        keys.set_propagation_phase(gtk4::PropagationPhase::Capture);
        keys.add_shortcut(shortcut(&format!("{HOTKEY}g"), move |ui| {
            ui.win_grab(i);
            true
        }));
        window.add_controller(keys);
        window.present();
        self.vcs[i].window = Some(window);
        self.rebuild_vc_menu();
        self.update_geometry_hints(i);
        self.update_caption();
    }

    /// `gd_tab_window_close()`.
    fn tab_window_close(&mut self, i: usize) {
        let Some(window) = self.vcs[i].window.take() else { return };
        let area = self.vcs[i].area.clone();
        window.set_child(None::<&gtk4::Widget>);
        let page = self.notebook_pos(i);
        let label = gtk4::Label::new(Some(&self.vcs[i].label));
        self.notebook.insert_page(&area, Some(&label), Some(page));
        window.destroy();
        self.rebuild_vc_menu();
        if self.by_menu() == Some(i) {
            area.grab_focus();
        }
    }

    /// `gd_win_grab()`, the grab key of a detached tab.
    fn win_grab(&mut self, i: usize) {
        eprintln!("gd_win_grab: {}", self.vcs[i].label);
        if self.ptr_owner.is_some() {
            self.ungrab_pointer();
        } else {
            self.grab_pointer(i);
        }
    }

    /// `gd_mouse_mode_change()`.
    fn mouse_mode_change(&mut self) {
        if let Some(p) = self.ptr_owner {
            if self.is_absolute(p) {
                if self.vcs[p].window.is_none() {
                    self.set_active(Check::Grab, false);
                } else {
                    self.ungrab_pointer();
                }
            }
        }
        for i in 0..self.vcs.len() {
            self.update_cursor(i);
        }
    }

    /// `gd_configure()` and `gd_set_ui_size()`.
    fn configure(&mut self, i: usize, width: i32, height: i32) {
        let vc = &self.vcs[i];
        let (mut w, mut h) = (f64::from(width), f64::from(height));
        if !self.scaling.free_scale.get() && !self.scaling.full_screen.get() {
            let view = vc.view.borrow();
            w /= view.scale_x;
            h /= view.scale_y;
        }
        if !vc.con.ui_info_supported() {
            return;
        }
        let mut info = vc.con.ui_info();
        info.width = w as u32;
        info.height = h as u32;
        vc.con.set_ui_info(info);
    }

    /// `gd_motion_event()`, without the warp to the middle of the monitor that keeps a grabbed
    /// pointer from stopping at its edges.
    fn motion_event(&mut self, i: usize, x: f64, y: f64) {
        let vc = &self.vcs[i];
        let (fbw, fbh, sx, sy) = {
            let view = vc.view.borrow();
            let Some((w, h)) = view.size() else { return };
            (w, h, view.scale_x, view.scale_y)
        };
        let ww_surface = (f64::from(fbw) * sx) as i32;
        let wh_surface = (f64::from(fbh) * sy) as i32;
        let ww_widget = vc.area.width();
        let wh_widget = vc.area.height();
        let wx_offset = if ww_widget > ww_surface { (ww_widget - ww_surface) / 2 } else { 0 };
        let wy_offset = if wh_widget > wh_surface { (wh_widget - wh_surface) / 2 } else { 0 };
        let fbx = ((x - f64::from(wx_offset)) / sx) as i32;
        let fby = ((y - f64::from(wy_offset)) / sy) as i32;
        let con = vc.con.clone();
        if self.is_absolute(i) {
            if fbx < 0 || fby < 0 || fbx >= fbw || fby >= fbh {
                return;
            }
            self.input.queue_abs(Some(&con), InputAxis::X, fbx, 0, fbw);
            self.input.queue_abs(Some(&con), InputAxis::Y, fby, 0, fbh);
            self.input.event_sync();
        } else if self.last_set && self.ptr_owner == Some(i) {
            self.input.queue_rel(Some(&con), InputAxis::X, (fbx - self.last_x).into());
            self.input.queue_rel(Some(&con), InputAxis::Y, (fby - self.last_y).into());
            self.input.event_sync();
        }
        self.last_x = fbx;
        self.last_y = fby;
        self.last_set = true;
    }

    /// `gd_button_event()`.
    fn button_event(&mut self, i: usize, button: u32, down: bool) {
        // A click takes the keyboard focus back from the menus.
        if down {
            self.vcs[i].area.grab_focus();
        }
        // The first click in relative mode grabs the input.
        if button == 1 && down && !self.is_absolute(i) && self.ptr_owner != Some(i) {
            if self.vcs[i].window.is_none() {
                self.set_active(Check::Grab, true);
            } else {
                self.grab_pointer(i);
            }
            return;
        }
        let btn = match button {
            1 => InputButton::Left,
            2 => InputButton::Middle,
            3 => InputButton::Right,
            8 => InputButton::Side,
            9 => InputButton::Extra,
            _ => return,
        };
        let con = self.vcs[i].con.clone();
        self.input.queue_btn(Some(&con), btn, down);
        self.input.event_sync();
    }

    /// `gd_scroll_event()`.
    fn scroll_event(&mut self, i: usize, dir: gdk::ScrollDirection, deltas: (f64, f64)) {
        let btn = match dir {
            gdk::ScrollDirection::Up => InputButton::WheelUp,
            gdk::ScrollDirection::Down => InputButton::WheelDown,
            gdk::ScrollDirection::Left => InputButton::WheelLeft,
            gdk::ScrollDirection::Right => InputButton::WheelRight,
            gdk::ScrollDirection::Smooth => {
                let (dx, dy) = deltas;
                if dy > 0.0 {
                    InputButton::WheelDown
                } else if dy < 0.0 {
                    InputButton::WheelUp
                } else if dx > 0.0 {
                    InputButton::WheelRight
                } else if dx < 0.0 {
                    InputButton::WheelLeft
                } else {
                    return;
                }
            }
            _ => return,
        };
        let con = self.vcs[i].con.clone();
        self.input.queue_btn(Some(&con), btn, true);
        self.input.event_sync();
        self.input.queue_btn(Some(&con), btn, false);
        self.input.event_sync();
    }

    /// `gd_key_event()`.
    fn key_event(&mut self, i: usize, keyval: gdk::Key, keycode: u32, down: bool) {
        let lnx = if keyval == gdk::Key::Pause { KEY_PAUSE } else { map_keycode(keycode) };
        let mut out = Vec::new();
        self.vcs[i].kbd.key_event(lnx, down, &mut out);
        kbd_state::send(&self.input, out);
    }

    /// `gd_enter_event()`.
    fn enter_event(&mut self, i: usize) {
        if self.get(Check::GrabOnHover) {
            self.grab_keyboard(i);
        }
    }

    /// `gd_leave_event()`.
    fn leave_event(&mut self) {
        if self.get(Check::GrabOnHover) {
            self.ungrab_keyboard();
        }
    }
}

/// `gd_map_keycode()` with evdev keycodes, which are Linux key codes plus 8.
fn map_keycode(keycode: u32) -> u32 {
    keycode.checked_sub(8).unwrap_or(KEY_RESERVED)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keycodes_are_evdev() {
        // KEY_A is 30, which X and Wayland report as 38.
        assert_eq!(map_keycode(38), 30);
        assert_eq!(map_keycode(7), KEY_RESERVED);
        assert_eq!(map_keycode(9), 1);
    }

    #[test]
    fn scale_follows_the_window() {
        let scaling = Scaling::default();
        let mut view = View {
            image: None,
            surface: None,
            dirty: false,
            scale_x: 1.0,
            scale_y: 1.0,
            preferred_scale: 1.0,
        };
        // A fixed scale stays.
        view.update_scale(&scaling, 1280, 800, 640, 480);
        assert_eq!((view.scale_x, view.scale_y), (1.0, 1.0));
        scaling.free_scale.set(true);
        scaling.keep_aspect_ratio.set(true);
        view.update_scale(&scaling, 1280, 800, 640, 480);
        assert_eq!((view.scale_x, view.scale_y), (800.0 / 480.0, 800.0 / 480.0));
        scaling.keep_aspect_ratio.set(false);
        view.update_scale(&scaling, 1280, 800, 640, 480);
        assert_eq!((view.scale_x, view.scale_y), (2.0, 800.0 / 480.0));
        scaling.full_screen.set(true);
        view.update_scale(&scaling, 1920, 1080, 640, 480);
        assert_eq!((view.scale_x, view.scale_y), (3.0, 2.25));
    }
}
