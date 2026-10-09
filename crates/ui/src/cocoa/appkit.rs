// SPDX-License-Identifier: GPL-2.0-or-later

//! The AppKit half of `-display cocoa`: the application, the view, the window, the menus and the
//! listener, as `QemuApplication`, `QemuCocoaView`, `QemuCocoaAppController` and
//! `cocoa_display_init()` are in QEMU.
//!
//! Everything but the listener lives on the main thread, in [`Ui`], which the classes reach
//! through a thread local rather than through globals as in QEMU. The listener runs on the
//! machine's threads and posts to the main queue.

use std::cell::{Cell, OnceCell, RefCell};
use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use dispatch2::DispatchQueue;
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, ProtocolObject, Sel};
use objc2::{AllocAnyThread, ClassType, MainThreadOnly, define_class, msg_send, sel};
use objc2_app_kit::{
    NSAlert, NSAlertSecondButtonReturn, NSApplication, NSApplicationActivationPolicy,
    NSApplicationDelegate, NSApplicationPresentationOptions, NSApplicationTerminateReply,
    NSBackingStoreType, NSBeep, NSColor, NSColorSpace, NSControlStateValueOff,
    NSControlStateValueOn, NSCursor, NSEvent, NSEventModifierFlags, NSEventType, NSFont,
    NSGraphicsContext, NSMenu, NSMenuItem, NSResponder, NSTextField, NSTrackingArea,
    NSTrackingAreaOptions, NSView, NSWindow, NSWindowCollectionBehavior, NSWindowDelegate,
    NSWindowStyleMask, NSWorkspace,
};
use objc2_core_foundation::{CFData, CFMachPort, CFRetained, CFRunLoop, kCFRunLoopDefaultMode};
use objc2_core_graphics::{
    CGAssociateMouseAndMouseCursorPosition, CGBitmapInfo, CGColorRenderingIntent, CGContext,
    CGDataProvider, CGDisplayScreenSize, CGEvent, CGEventTapLocation, CGEventTapOptions,
    CGEventTapPlacement, CGEventTapProxy, CGEventType, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, CGInterpolationQuality,
};
use objc2_foundation::{
    MainThreadMarker, NSDictionary, NSNotification, NSNumber, NSObject, NSObjectProtocol, NSPoint,
    NSRect, NSSize, NSString, NSURL, ns_string,
};
use ruvm_base::report::{error_report, warn_report};
use ruvm_qapi::types::{DisplayOptions, DisplayOptionsU, InputAxis, InputButton};

use super::{Event, Handled, Hooks, Keys, abs_position, fix_aspect_ratio, refresh_from_fps, title};
use crate::console::{DisplayChangeListener, DisplayState, ListenerId, QemuConsole, QemuUiInfo};
use crate::input::InputState;
use crate::kbd_state::{self, KbdState};
use crate::pixman::{Image, X8R8G8B8};

thread_local! {
    /// The window's state, which only the main thread touches.
    static UI: OnceCell<Rc<Ui>> = const { OnceCell::new() };
}

fn ui() -> Option<Rc<Ui>> {
    UI.with(|u| u.get().cloned())
}

/// Runs `f` on the main thread's [`Ui`], later.
fn on_main(f: impl FnOnce(&Ui) + Send + 'static) {
    DispatchQueue::main().exec_async(move || {
        if let Some(ui) = ui() {
            f(&ui);
        }
    });
}

/// The window's copy of the console's surface.
type Frame = Arc<Mutex<Option<Image>>>;

fn lock(frame: &Frame) -> MutexGuard<'_, Option<Image>> {
    frame.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `dcl_ops` of cocoa.m.
#[derive(Debug)]
struct Listener {
    frame: Frame,
}

impl DisplayChangeListener for Listener {
    fn name(&self) -> &str {
        "cocoa"
    }

    fn has_refresh(&self) -> bool {
        true
    }

    /// `cocoa_refresh()`.
    fn refresh(&self, con: &QemuConsole) {
        con.hw_update_nowait();
    }

    /// `cocoa_update()`: the pixels go to the copy here and the view redraws them later.
    fn gfx_update(&self, con: &QemuConsole, x: i32, y: i32, w: i32, h: i32) {
        con.with_surface(|s| {
            if let (Some(s), Some(img)) = (s, lock(&self.frame).as_mut()) {
                let (x, y, w, h) = (i64::from(x), i64::from(y), i64::from(w), i64::from(h));
                img.composite_src(s.image(), x, y, x, y, w, h);
            }
        });
        on_main(move |ui| {
            let sh = ui.screen.get().1;
            let rect = NSRect::new(
                NSPoint::new(f64::from(x), f64::from(sh - y - h)),
                NSSize::new(f64::from(w), f64::from(h)),
            );
            ui.view().setNeedsDisplayInRect(rect);
        });
    }

    /// `cocoa_switch()`. The copy changes now, so that no update after the switch is lost, and
    /// the window changes size on the main thread as `switchSurface:` does.
    fn gfx_switch(&self, con: &QemuConsole) {
        let size = con.with_surface(|s| {
            let s = s?;
            let (w, h) = (s.width(), s.height());
            let mut img = Image::new(X8R8G8B8, w, h, 0);
            img.composite_src(s.image(), 0, 0, 0, 0, w as i64, h as i64);
            *lock(&self.frame) = Some(img);
            Some((i32::try_from(w).unwrap_or(i32::MAX), i32::try_from(h).unwrap_or(i32::MAX)))
        });
        if let Some(size) = size {
            on_main(move |ui| ui.switch_surface(size));
        }
    }
}

/// The state of `QemuCocoaView` and `QemuCocoaAppController`, and the globals of cocoa.m.
struct Ui {
    mtm: MainThreadMarker,
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    hooks: Arc<dyn Hooks>,
    /// `qemu_name`.
    name: Option<String>,
    /// The version and copyright lines of the About panel.
    about: (String, String),
    frame: Frame,
    listener: Arc<Listener>,
    /// `dcl.con`.
    con: RefCell<Option<QemuConsole>>,
    id: Cell<Option<ListenerId>>,
    keys: RefCell<Keys>,
    /// `screen`: the size of the guest's display.
    screen: Cell<(i32, i32)>,
    /// `isMouseGrabbed`.
    grabbed: Cell<bool>,
    /// `isAbsoluteEnabled`.
    absolute: Cell<bool>,
    cursor_hide: Cell<bool>,
    allow_events: Cell<bool>,
    zoom_interpolation: Cell<CGInterpolationQuality>,
    view: OnceCell<Retained<QemuCocoaView>>,
    pause_label: OnceCell<Retained<NSTextField>>,
    pause_item: OnceCell<Retained<NSMenuItem>>,
    resume_item: OnceCell<Retained<NSMenuItem>>,
    /// The application keeps a weak reference to its delegate.
    controller: OnceCell<Retained<QemuCocoaAppController>>,
    events_tap: RefCell<Option<CFRetained<CFMachPort>>>,
}

define_class!(
    // SAFETY: NSApplication may be subclassed, and QemuApplication has no Drop.
    #[unsafe(super(NSApplication, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    struct QemuApplication;

    impl QemuApplication {
        // SAFETY: the signature is that of -[NSApplication sendEvent:].
        #[unsafe(method(sendEvent:))]
        fn send_event(&self, event: &NSEvent) {
            if !ui().is_some_and(|ui| ui.handle_event(event)) {
                // SAFETY: the superclass has sendEvent: with this signature.
                let () = unsafe { msg_send![super(self), sendEvent: event] };
            }
        }
    }
);

define_class!(
    // SAFETY: NSView may be subclassed, and QemuCocoaView has no Drop.
    #[unsafe(super(NSView, NSResponder, NSObject))]
    #[thread_kind = MainThreadOnly]
    struct QemuCocoaView;

    impl QemuCocoaView {
        // SAFETY: the signatures below are those of NSView and NSResponder.
        #[unsafe(method(isOpaque))]
        fn is_opaque(&self) -> bool {
            true
        }

        #[unsafe(method(viewDidMoveToWindow))]
        fn view_did_move_to_window(&self) {
            with_ui(Ui::resize_window);
        }

        #[unsafe(method(drawRect:))]
        fn draw_rect(&self, rect: NSRect) {
            with_ui(|ui| ui.draw(rect));
        }

        #[unsafe(method(mouseMoved:))]
        fn mouse_moved(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse(event));
        }

        #[unsafe(method(mouseDown:))]
        fn mouse_down(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse_button(event, InputButton::Left, true));
        }

        #[unsafe(method(rightMouseDown:))]
        fn right_mouse_down(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse_button(event, InputButton::Right, true));
        }

        #[unsafe(method(otherMouseDown:))]
        fn other_mouse_down(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse_button(event, InputButton::Middle, true));
        }

        #[unsafe(method(mouseDragged:))]
        fn mouse_dragged(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse(event));
        }

        #[unsafe(method(rightMouseDragged:))]
        fn right_mouse_dragged(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse(event));
        }

        #[unsafe(method(otherMouseDragged:))]
        fn other_mouse_dragged(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse(event));
        }

        #[unsafe(method(mouseUp:))]
        fn mouse_up(&self, event: &NSEvent) {
            with_ui(|ui| {
                if !ui.grabbed.get() {
                    ui.grab_mouse();
                }
                ui.handle_mouse_button(event, InputButton::Left, false);
            });
        }

        #[unsafe(method(rightMouseUp:))]
        fn right_mouse_up(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse_button(event, InputButton::Right, false));
        }

        #[unsafe(method(otherMouseUp:))]
        fn other_mouse_up(&self, event: &NSEvent) {
            with_ui(|ui| ui.handle_mouse_button(event, InputButton::Middle, false));
        }

        #[unsafe(method(mouseExited:))]
        fn mouse_exited(&self, _event: &NSEvent) {
            with_ui(|ui| {
                if ui.absolute.get() && ui.grabbed.get() {
                    ui.ungrab_mouse();
                }
            });
        }

        #[unsafe(method(mouseEntered:))]
        fn mouse_entered(&self, _event: &NSEvent) {
            with_ui(|ui| {
                if ui.absolute.get() && !ui.grabbed.get() {
                    ui.grab_mouse();
                }
            });
        }
    }
);

define_class!(
    // SAFETY: NSObject may be subclassed, and QemuCocoaAppController has no Drop.
    #[unsafe(super = NSObject)]
    #[thread_kind = MainThreadOnly]
    struct QemuCocoaAppController;

    // SAFETY: NSObjectProtocol has no requirements.
    unsafe impl NSObjectProtocol for QemuCocoaAppController {}

    // SAFETY: the signatures below are those of NSApplicationDelegate.
    unsafe impl NSApplicationDelegate for QemuCocoaAppController {
        #[unsafe(method(applicationDidFinishLaunching:))]
        fn did_finish_launching(&self, _note: &NSNotification) {
            with_ui(|ui| ui.allow_events.set(true));
        }

        #[unsafe(method(applicationWillTerminate:))]
        fn will_terminate(&self, _note: &NSNotification) {
            with_ui(|ui| ui.hooks.quit());
            // Returning lets the system kill the process now. The machine's main loop handles
            // the shutdown request and ends it instead.
            loop {
                std::thread::park();
            }
        }

        #[unsafe(method(applicationShouldTerminateAfterLastWindowClosed:))]
        fn should_terminate_after_last_window_closed(&self, _app: &NSApplication) -> bool {
            true
        }

        #[unsafe(method(applicationShouldTerminate:))]
        fn should_terminate(&self, _app: &NSApplication) -> NSApplicationTerminateReply {
            if verify_quit(self.mtm()) {
                NSApplicationTerminateReply::TerminateNow
            } else {
                NSApplicationTerminateReply::TerminateCancel
            }
        }
    }

    // SAFETY: the signatures below are those of NSWindowDelegate.
    unsafe impl NSWindowDelegate for QemuCocoaAppController {
        #[unsafe(method(windowDidChangeScreen:))]
        fn window_did_change_screen(&self, _note: &NSNotification) {
            with_ui(Ui::update_ui_info);
        }

        #[unsafe(method(windowDidEnterFullScreen:))]
        fn window_did_enter_full_screen(&self, _note: &NSNotification) {
            with_ui(Ui::grab_mouse);
        }

        #[unsafe(method(windowDidExitFullScreen:))]
        fn window_did_exit_full_screen(&self, _note: &NSNotification) {
            with_ui(|ui| {
                ui.resize_window();
                ui.ungrab_mouse();
            });
        }

        #[unsafe(method(windowDidResize:))]
        fn window_did_resize(&self, _note: &NSNotification) {
            with_ui(|ui| {
                ui.update_bounds();
                ui.update_ui_info();
            });
        }

        /// The close button asks the application to quit, which asks the user first. If the
        /// application is still here, the user said no and the window stays.
        #[unsafe(method(windowShouldClose:))]
        fn window_should_close(&self, sender: &NSWindow) -> bool {
            NSApplication::sharedApplication(self.mtm()).terminate(Some(sender));
            false
        }

        #[unsafe(method(window:willUseFullScreenPresentationOptions:))]
        fn will_use_full_screen_presentation_options(
            &self,
            _window: &NSWindow,
            proposed: NSApplicationPresentationOptions,
        ) -> NSApplicationPresentationOptions {
            (proposed
                & !(NSApplicationPresentationOptions::AutoHideDock
                    | NSApplicationPresentationOptions::AutoHideMenuBar))
                | NSApplicationPresentationOptions::HideDock
                | NSApplicationPresentationOptions::HideMenuBar
        }

        /// QEMU going into the background. `windowDidResignKey:` also sees the Dock being
        /// clicked, which `applicationWillResignActive:` does not.
        #[unsafe(method(windowDidResignKey:))]
        fn window_did_resign_key(&self, _note: &NSNotification) {
            with_ui(|ui| {
                ui.ungrab_mouse();
                ui.raise_all_keys();
            });
        }
    }

    // SAFETY: the menu items below send these with a menu item.
    impl QemuCocoaAppController {
        #[unsafe(method(doToggleFullScreen:))]
        fn do_toggle_full_screen(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                if let Some(w) = ui.view().window() {
                    w.toggleFullScreen(None);
                }
            });
        }

        #[unsafe(method(showQEMUDoc:))]
        fn show_qemu_doc(&self, _sender: Option<&NSMenuItem>) {
            open_documentation(self.mtm(), "index.html");
        }

        /// Stretches the guest's display to the window.
        #[unsafe(method(zoomToFit:))]
        fn zoom_to_fit(&self, sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                let Some(w) = ui.view().window() else {
                    return;
                };
                let mask = w.styleMask() ^ NSWindowStyleMask::Resizable;
                w.setStyleMask(mask);
                if let Some(sender) = sender {
                    sender.setState(state(mask.contains(NSWindowStyleMask::Resizable)));
                }
                ui.resize_window();
            });
        }

        #[unsafe(method(toggleZoomInterpolation:))]
        fn toggle_zoom_interpolation(&self, sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                let on = ui.zoom_interpolation.get() == CGInterpolationQuality::None;
                ui.zoom_interpolation.set(if on {
                    CGInterpolationQuality::Low
                } else {
                    CGInterpolationQuality::None
                });
                if let Some(sender) = sender {
                    sender.setState(state(on));
                }
            });
        }

        #[unsafe(method(displayConsole:))]
        fn display_console(&self, sender: Option<&NSMenuItem>) {
            if let Some(index) = sender.and_then(|s| u32::try_from(s.tag()).ok()) {
                with_ui(|ui| ui.select_console(index));
            }
        }

        #[unsafe(method(pauseQEMU:))]
        fn pause_qemu(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                ui.hooks.stop();
                set_enabled(&ui.pause_item, false);
                set_enabled(&ui.resume_item, true);
                ui.display_pause();
            });
        }

        #[unsafe(method(resumeQEMU:))]
        fn resume_qemu(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                ui.hooks.cont();
                set_enabled(&ui.resume_item, false);
                set_enabled(&ui.pause_item, true);
                if let Some(label) = ui.pause_label.get() {
                    label.removeFromSuperview();
                }
            });
        }

        #[unsafe(method(restartQEMU:))]
        fn restart_qemu(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| ui.hooks.reset());
        }

        #[unsafe(method(powerDownQEMU:))]
        fn power_down_qemu(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| ui.hooks.powerdown());
        }

        #[unsafe(method(do_about_menu_item:))]
        fn do_about_menu_item(&self, _sender: Option<&NSMenuItem>) {
            with_ui(|ui| {
                let (version, copyright) = (NSString::from_str(&ui.about.0), NSString::from_str(&ui.about.1));
                // The values of NSAboutPanelOptionApplicationVersion and the copyright key.
                let options = NSDictionary::<NSString, AnyObject>::from_slices(
                    &[ns_string!("ApplicationVersion"), ns_string!("Copyright")],
                    &[&version, &copyright],
                );
                let app = NSApplication::sharedApplication(ui.mtm);
                // SAFETY: both values are strings, as the two keys want.
                unsafe { app.orderFrontStandardAboutPanelWithOptions(&options) };
            });
        }
    }
);

fn with_ui(f: impl FnOnce(&Ui)) {
    if let Some(ui) = ui() {
        f(&ui);
    }
}

fn state(on: bool) -> isize {
    if on { NSControlStateValueOn } else { NSControlStateValueOff }
}

fn set_enabled(item: &OnceCell<Retained<NSMenuItem>>, enabled: bool) {
    if let Some(item) = item.get() {
        item.setEnabled(enabled);
    }
}

/// `QEMU_Alert()`.
fn alert(mtm: MainThreadMarker, message: &str) {
    let alert = NSAlert::new(mtm);
    alert.setMessageText(&NSString::from_str(message));
    alert.runModal();
}

/// `verifyQuit`.
fn verify_quit(mtm: MainThreadMarker) -> bool {
    let alert = NSAlert::new(mtm);
    alert.setMessageText(ns_string!("Are you sure you want to quit QEMU?"));
    alert.addButtonWithTitle(ns_string!("Cancel"));
    alert.addButtonWithTitle(ns_string!("Quit"));
    alert.runModal() == NSAlertSecondButtonReturn
}

/// `openDocumentation:`: the file in one of the places QEMU installs its manual, relative to the
/// executable.
fn open_documentation(mtm: MainThreadMarker, filename: &str) {
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(ToOwned::to_owned))
    {
        let workspace = NSWorkspace::sharedWorkspace();
        for prefix in ["../share/doc/qemu/", "../doc/qemu/", "docs/"] {
            let path = format!("{}/{prefix}{filename}", dir.display());
            let url = NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&path), false);
            if workspace.openURL(&url) {
                return;
            }
        }
    }
    NSBeep();
    alert(mtm, "Failed to open file");
}

/// A menu item that sends `action` up the responder chain.
fn item(
    mtm: MainThreadMarker,
    title: &str,
    action: Option<Sel>,
    key: &str,
) -> Retained<NSMenuItem> {
    // SAFETY: each action is a method of the controller, the window or the application, which
    // take the item as the sender.
    unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            action,
            &NSString::from_str(key),
        )
    }
}

/// Adds `menu` to the menu bar under an item called `title`.
fn add_menu(app: &NSApplication, mtm: MainThreadMarker, title: &str, menu: &NSMenu) {
    let parent = item(mtm, title, None, "");
    parent.setSubmenu(Some(menu));
    if let Some(main) = app.mainMenu() {
        main.addItem(&parent);
    }
}

/// `handleTapEvent()`: with `full-grab`, the keys the system would take go to the guest while the
/// mouse is grabbed.
extern "C-unwind" fn handle_tap_event(
    _proxy: CGEventTapProxy,
    _type: CGEventType,
    cg_event: NonNull<CGEvent>,
    _user_info: *mut c_void,
) -> *mut CGEvent {
    // SAFETY: the tap passes an event that lives until the callback returns.
    let cg = unsafe { cg_event.as_ref() };
    let taken = ui().is_some_and(|ui| {
        ui.grabbed.get() && NSEvent::eventWithCGEvent(cg).is_some_and(|e| ui.handle_event(&e))
    });
    if taken { ptr::null_mut() } else { cg_event.as_ptr() }
}

/// The part of an `NSEvent` that [`Keys::handle`] looks at. Only key events have a keycode.
fn event_of(e: &NSEvent) -> Event {
    let ty = e.r#type();
    if ty == NSEventType::FlagsChanged {
        Event::FlagsChanged { keycode: e.keyCode() }
    } else if ty == NSEventType::KeyDown {
        let unit = e.charactersIgnoringModifiers().and_then(|s| {
            let units: Vec<u16> = s.to_string().encode_utf16().collect();
            if units.len() == 1 { Some(units[0]) } else { None }
        });
        Event::KeyDown { keycode: e.keyCode(), unit }
    } else if ty == NSEventType::KeyUp {
        Event::KeyUp { keycode: e.keyCode() }
    } else if ty == NSEventType::ScrollWheel {
        Event::ScrollWheel { dx: e.deltaX(), dy: e.deltaY() }
    } else {
        Event::Other
    }
}

impl Ui {
    fn view(&self) -> &QemuCocoaView {
        self.view.get().expect("the view is made before anything uses it")
    }

    fn con(&self) -> Option<QemuConsole> {
        self.con.borrow().clone()
    }

    /// `handleEvent:`: whether the event was taken.
    fn handle_event(&self, event: &NSEvent) -> bool {
        let ev = event_of(event);
        let modifiers = event.modifierFlags().0 as u64;
        let graphic = self.con().is_some_and(|c| c.is_graphic());
        let mut out = Vec::new();
        let handled =
            self.keys.borrow_mut().handle(ev, modifiers, self.grabbed.get(), graphic, &mut out);
        kbd_state::send(&self.input, out);
        match handled {
            Handled::No => false,
            Handled::Yes => true,
            Handled::SelectConsole(index) => {
                self.select_console(index);
                true
            }
            Handled::Ungrab => {
                self.ungrab_mouse();
                true
            }
            Handled::Wheel(button) => {
                let con = self.con();
                self.input.queue_btn(con.as_ref(), button, true);
                self.input.event_sync();
                self.input.queue_btn(con.as_ref(), button, false);
                self.input.event_sync();
                true
            }
        }
    }

    /// `handleMouseEvent:button:down:`.
    fn handle_mouse_button(&self, event: &NSEvent, button: InputButton, down: bool) {
        if !self.grabbed.get() {
            return;
        }
        self.input.queue_btn(self.con().as_ref(), button, down);
        self.handle_mouse(event);
    }

    /// `handleMouseEvent:`.
    fn handle_mouse(&self, event: &NSEvent) {
        if !self.grabbed.get() {
            return;
        }
        let con = self.con();
        if self.absolute.get() {
            let screen = self.screen.get();
            let p = event.locationInWindow();
            let (x, y) = abs_position((p.x, p.y), self.view().frame().size.height, screen);
            self.input.queue_abs(con.as_ref(), InputAxis::X, x, 0, screen.0);
            self.input.queue_abs(con.as_ref(), InputAxis::Y, y, 0, screen.1);
        } else {
            // The C code converts the deltas to int.
            self.input.queue_rel(con.as_ref(), InputAxis::X, event.deltaX() as i64);
            self.input.queue_rel(con.as_ref(), InputAxis::Y, event.deltaY() as i64);
        }
        self.input.event_sync();
    }

    fn set_title(&self, grabbed: bool) {
        if let Some(w) = self.view().window() {
            w.setTitle(&NSString::from_str(&title(self.name.as_deref(), grabbed)));
        }
    }

    /// `grabMouse`.
    fn grab_mouse(&self) {
        self.set_title(true);
        if self.cursor_hide.get() {
            NSCursor::hide();
        }
        CGAssociateMouseAndMouseCursorPosition(self.absolute.get());
        self.grabbed.set(true);
    }

    /// `ungrabMouse`.
    fn ungrab_mouse(&self) {
        self.set_title(false);
        if self.cursor_hide.get() {
            NSCursor::unhide();
        }
        CGAssociateMouseAndMouseCursorPosition(true);
        self.grabbed.set(false);
        self.raise_all_buttons();
    }

    /// `notifyMouseModeChange`.
    fn notify_mouse_mode_change(&self) {
        let absolute = self.input.is_absolute(self.con().as_ref());
        if absolute == self.absolute.get() {
            return;
        }
        self.absolute.set(absolute);
        if self.grabbed.get() {
            if absolute {
                self.ungrab_mouse();
            } else {
                CGAssociateMouseAndMouseCursorPosition(false);
            }
        }
    }

    /// `raiseAllKeys`: the keys still down go up, since their key up events go to whatever has
    /// the focus now.
    fn raise_all_keys(&self) {
        let mut out = Vec::new();
        self.keys.borrow_mut().kbd.lift_all_keys(&mut out);
        kbd_state::send(&self.input, out);
    }

    /// `raiseAllButtons`.
    fn raise_all_buttons(&self) {
        let con = self.con();
        for button in [InputButton::Left, InputButton::Right, InputButton::Middle] {
            self.input.queue_btn(con.as_ref(), button, false);
        }
    }

    /// `selectConsoleLocked:`.
    fn select_console(&self, index: u32) {
        let Some(con) = self.ds.lookup_by_index(index) else {
            return;
        };
        if let Some(id) = self.id.take() {
            self.ds.unregister_listener(id);
        }
        let mut out = Vec::new();
        self.keys.borrow_mut().kbd.switch_console(Some(con.clone()), &mut out);
        kbd_state::send(&self.input, out);
        self.con.replace(Some(con.clone()));
        self.register(&con);
        self.notify_mouse_mode_change();
        self.update_ui_info();
    }

    fn register(&self, con: &QemuConsole) {
        let listener: Arc<dyn DisplayChangeListener> = self.listener.clone();
        self.id.set(Some(self.ds.register_listener(con, listener)));
    }

    /// `screenSafeAreaSize`.
    fn screen_safe_area_size(window: &NSWindow) -> (f64, f64) {
        let Some(screen) = window.screen() else {
            return (0.0, 0.0);
        };
        let size = screen.frame().size;
        let insets = screen.safeAreaInsets();
        (size.width - insets.left - insets.right, size.height - insets.top - insets.bottom)
    }

    /// `resizeWindow`.
    fn resize_window(&self) {
        let Some(w) = self.view().window() else {
            return;
        };
        let screen = self.screen.get();
        w.setContentAspectRatio(NSSize::new(f64::from(screen.0), f64::from(screen.1)));
        let mask = w.styleMask();
        if !mask.contains(NSWindowStyleMask::Resizable) {
            let scale = w.backingScaleFactor();
            w.setContentSize(NSSize::new(f64::from(screen.0) / scale, f64::from(screen.1) / scale));
            w.center();
        } else if mask.contains(NSWindowStyleMask::FullScreen) {
            let (fw, fh) = fix_aspect_ratio(screen, Self::screen_safe_area_size(&w));
            w.setContentSize(NSSize::new(fw, fh));
            w.center();
        } else {
            let size = self.view().frame().size;
            let (fw, fh) = fix_aspect_ratio(screen, (size.width, size.height));
            w.setContentSize(NSSize::new(fw, fh));
        }
    }

    /// `updateBounds`.
    fn update_bounds(&self) {
        let (sw, sh) = self.screen.get();
        self.view().setBoundsSize(NSSize::new(f64::from(sw), f64::from(sh)));
    }

    /// `updateUIInfo`: nothing goes to the console until the application has finished
    /// launching. The listener is registered by then and its switch tells the window the size.
    fn update_ui_info(&self) {
        if self.allow_events.get() {
            self.update_ui_info_locked();
        }
    }

    /// `updateUIInfoLocked`.
    fn update_ui_info_locked(&self) {
        let Some(con) = self.con() else {
            return;
        };
        if !con.is_graphic() {
            return;
        }
        let view = self.view();
        let mut info = QemuUiInfo::default();
        let (frame, scale) = match view.window() {
            Some(w) => {
                let full_screen = w.styleMask().contains(NSWindowStyleMask::FullScreen);
                let frame = if full_screen {
                    Self::screen_safe_area_size(&w)
                } else {
                    let size = view.frame().size;
                    (size.width, size.height)
                };
                if let Some(screen) = w.screen() {
                    let display = screen
                        .deviceDescription()
                        .objectForKey(ns_string!("NSScreenNumber"))
                        .and_then(|n| n.downcast::<NSNumber>().ok())
                        .map_or(0, |n| n.unsignedIntValue());
                    let screen_size = screen.frame().size;
                    let physical = CGDisplayScreenSize(display);
                    if let Some((interval, rate)) =
                        refresh_from_fps(screen.maximumFramesPerSecond() as i64)
                    {
                        if let Some(id) = self.id.get() {
                            self.ds.listener_set_refresh(id, interval);
                        }
                        info.refresh_rate = rate;
                    }
                    info.width_mm = (frame.0 / screen_size.width * physical.width) as u16;
                    info.height_mm = (frame.1 / screen_size.height * physical.height) as u16;
                }
                (frame, w.backingScaleFactor())
            }
            // A message to a nil window gives a scale of 0.
            None => {
                let size = view.frame().size;
                ((size.width, size.height), 0.0)
            }
        };
        info.width = (frame.0 * scale) as u32;
        info.height = (frame.1 * scale) as u32;
        con.set_ui_info(info);
    }

    /// `switchSurface:`.
    fn switch_surface(&self, size: (i32, i32)) {
        if size != self.screen.get() {
            // Resize before the redraw, or it draws at the old size.
            self.screen.set(size);
            self.resize_window();
            self.update_bounds();
        }
    }

    /// `drawRect:`: the copy of the surface fills the view, whose bounds are the guest's
    /// display.
    fn draw(&self, rect: NSRect) {
        let Some(ctx) = NSGraphicsContext::currentContext() else {
            return;
        };
        let cg = ctx.CGContext();
        CGContext::set_interpolation_quality(Some(&cg), self.zoom_interpolation.get());
        CGContext::set_should_antialias(Some(&cg), false);
        let copy = lock(&self.frame).as_ref().map(|img| {
            let len = img.stride() * img.height();
            (img.width(), img.height(), img.stride(), CFData::from_bytes(&img.data()[..len]))
        });
        let Some((w, h, stride, data)) = copy else {
            // Nothing to show before a device sets up a framebuffer.
            CGContext::set_rgb_fill_color(Some(&cg), 0.0, 0.0, 0.0, 1.0);
            CGContext::fill_rect(Some(&cg), rect);
            return;
        };
        let provider = CGDataProvider::with_cf_data(Some(&data));
        let space = NSColorSpace::sRGBColorSpace().CGColorSpace();
        let info =
            CGBitmapInfo(CGImageByteOrderInfo::Order32Little.0 | CGImageAlphaInfo::NoneSkipFirst.0);
        // SAFETY: the provider holds stride * h bytes of 32 bit pixels and there is no decode
        // array.
        let image = unsafe {
            CGImage::new(
                w,
                h,
                8,
                32,
                stride,
                space.as_deref(),
                info,
                provider.as_deref(),
                ptr::null(),
                false,
                CGColorRenderingIntent::RenderingIntentDefault,
            )
        };
        let bounds = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(w as f64, h as f64));
        CGContext::draw_image(Some(&cg), bounds, image.as_deref());
    }

    /// `displayPause`: the label sits in the middle, near the top. The coordinates are ints in
    /// the C code.
    fn display_pause(&self) {
        let Some(label) = self.pause_label.get() else {
            return;
        };
        let view = self.view().frame().size;
        let size = label.frame().size;
        let x = ((view.width - size.width) / 2.0) as i32;
        let y = (view.height - size.height - size.height * 0.5) as i32;
        let (w, h) = (size.width as i32, size.height as i32);
        label.setFrame(NSRect::new(
            NSPoint::new(f64::from(x), f64::from(y)),
            NSSize::new(f64::from(w), f64::from(h)),
        ));
        self.view().addSubview(label);
    }

    /// `setFullGrab:`.
    fn set_full_grab(&self) {
        let mask = (1u64 << CGEventType::KeyDown.0)
            | (1u64 << CGEventType::KeyUp.0)
            | (1u64 << CGEventType::FlagsChanged.0);
        // SAFETY: the callback has the signature of CGEventTapCallBack and takes no user data.
        let tap = unsafe {
            CGEvent::tap_create(
                CGEventTapLocation::HIDEventTap,
                CGEventTapPlacement::HeadInsertEventTap,
                CGEventTapOptions::Default,
                mask,
                Some(handle_tap_event),
                ptr::null_mut(),
            )
        };
        let Some(tap) = tap else {
            warn_report("Could not create event tap, system key combos will not be captured.\n");
            return;
        };
        let source = CFMachPort::new_run_loop_source(None, Some(&tap), 0);
        self.events_tap.replace(Some(tap));
        let (Some(run_loop), Some(source)) = (CFRunLoop::current(), source) else {
            warn_report(
                "Could not obtain current CF RunLoop, system key combos will not be captured.\n",
            );
            return;
        };
        // SAFETY: kCFRunLoopDefaultMode is a constant CoreFoundation sets up before main.
        run_loop.add_source(Some(&source), unsafe { kCFRunLoopDefaultMode });
    }

    /// The view and the window of `-[QemuCocoaAppController init]`.
    fn make_window(&self, controller: &QemuCocoaAppController) {
        let mtm = self.mtm;
        let frame = NSRect::new(NSPoint::new(0.0, 0.0), NSSize::new(640.0, 480.0));
        let view = QemuCocoaView::alloc(mtm).set_ivars(());
        // SAFETY: initWithFrame: is NSView's designated initializer.
        let view: Retained<QemuCocoaView> = unsafe { msg_send![super(view), initWithFrame: frame] };
        let options = NSTrackingAreaOptions::ActiveInKeyWindow
            | NSTrackingAreaOptions::MouseEnteredAndExited
            | NSTrackingAreaOptions::MouseMoved
            | NSTrackingAreaOptions::InVisibleRect;
        // SAFETY: the view owns the area and gets its mouse events, and there is no user info.
        let area = unsafe {
            NSTrackingArea::initWithRect_options_owner_userInfo(
                NSTrackingArea::alloc(),
                NSRect::ZERO,
                options,
                Some(&view),
                None,
            )
        };
        view.addTrackingArea(&area);
        if view.respondsToSelector(sel!(setClipsToBounds:)) {
            view.setClipsToBounds(true);
        }
        view.setWantsLayer(true);
        let _ = self.view.set(view);
        let view = self.view();

        // SAFETY: the window is not released when it closes, which it never does.
        let window = unsafe {
            NSWindow::initWithContentRect_styleMask_backing_defer(
                NSWindow::alloc(mtm),
                view.frame(),
                NSWindowStyleMask::Titled
                    | NSWindowStyleMask::Miniaturizable
                    | NSWindowStyleMask::Closable,
                NSBackingStoreType::Buffered,
                false,
            )
        };
        // SAFETY: the window is kept by the view, which the thread local keeps.
        unsafe { window.setReleasedWhenClosed(false) };
        window.setAcceptsMouseMovedEvents(true);
        window.setCollectionBehavior(NSWindowCollectionBehavior::FullScreenPrimary);
        window.setTitle(&NSString::from_str(&title(self.name.as_deref(), false)));
        window.setContentView(Some(view));
        window.makeKeyAndOrderFront(None);
        window.center();
        window.setDelegate(Some(ProtocolObject::from_ref(controller)));

        // Shown on the screen while the machine is paused.
        let label = NSTextField::new(mtm);
        label.setBezeled(true);
        label.setDrawsBackground(true);
        label.setBackgroundColor(Some(&NSColor::whiteColor()));
        label.setEditable(false);
        label.setSelectable(false);
        label.setStringValue(ns_string!("Paused"));
        label.setFont(NSFont::fontWithName_size(ns_string!("Helvetica"), 90.0).as_deref());
        label.setTextColor(Some(&NSColor::blackColor()));
        label.sizeToFit();
        let _ = self.pause_label.set(label);
    }

    /// `create_initial_menus()`, without the `Speed` menu.
    fn create_initial_menus(&self, app: &NSApplication) {
        let mtm = self.mtm;
        app.setMainMenu(Some(&NSMenu::new(mtm)));
        app.setServicesMenu(Some(&NSMenu::initWithTitle(
            NSMenu::alloc(mtm),
            ns_string!("Services"),
        )));

        // The application menu.
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!(""));
        menu.addItem(&item(mtm, "About QEMU", Some(sel!(do_about_menu_item:)), ""));
        menu.addItem(&NSMenuItem::separatorItem(mtm));
        let services = item(mtm, "Services", None, "");
        services.setSubmenu(app.servicesMenu().as_deref());
        menu.addItem(&services);
        menu.addItem(&NSMenuItem::separatorItem(mtm));
        menu.addItem(&item(mtm, "Hide QEMU", Some(sel!(hide:)), "h"));
        let others = item(mtm, "Hide Others", Some(sel!(hideOtherApplications:)), "h");
        others.setKeyEquivalentModifierMask(
            NSEventModifierFlags::Option | NSEventModifierFlags::Command,
        );
        menu.addItem(&others);
        menu.addItem(&item(mtm, "Show All", Some(sel!(unhideAllApplications:)), ""));
        menu.addItem(&NSMenuItem::separatorItem(mtm));
        menu.addItem(&item(mtm, "Quit QEMU", Some(sel!(terminate:)), "q"));
        add_menu(app, mtm, "Apple", &menu);
        // setAppleMenu: is private, so it has no binding.
        // SAFETY: NSApplication has setAppleMenu:, which takes a menu.
        let () = unsafe { msg_send![app, setAppleMenu: &*menu] };

        // The Machine menu. Reset and Power Down work only when the machine has them.
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("Machine"));
        menu.setAutoenablesItems(false);
        let pause = item(mtm, "Pause", Some(sel!(pauseQEMU:)), "");
        menu.addItem(&pause);
        let resume = item(mtm, "Resume", Some(sel!(resumeQEMU:)), "");
        menu.addItem(&resume);
        resume.setEnabled(false);
        let _ = self.pause_item.set(pause);
        let _ = self.resume_item.set(resume);
        menu.addItem(&NSMenuItem::separatorItem(mtm));
        let reset = item(mtm, "Reset", Some(sel!(restartQEMU:)), "");
        reset.setEnabled(self.hooks.can_reset());
        menu.addItem(&reset);
        let powerdown = item(mtm, "Power Down", Some(sel!(powerDownQEMU:)), "");
        powerdown.setEnabled(self.hooks.can_powerdown());
        menu.addItem(&powerdown);
        add_menu(app, mtm, "Machine", &menu);

        // The View menu.
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("View"));
        menu.addItem(&item(mtm, "Enter Fullscreen", Some(sel!(doToggleFullScreen:)), "f"));
        let zoom = item(mtm, "Zoom To Fit", Some(sel!(zoomToFit:)), "");
        let resizable = self
            .view()
            .window()
            .is_some_and(|w| w.styleMask().contains(NSWindowStyleMask::Resizable));
        zoom.setState(state(resizable));
        menu.addItem(&zoom);
        let interpolation =
            item(mtm, "Zoom Interpolation", Some(sel!(toggleZoomInterpolation:)), "");
        interpolation.setState(state(self.zoom_interpolation.get() == CGInterpolationQuality::Low));
        menu.addItem(&interpolation);
        add_menu(app, mtm, "View", &menu);

        // The Window menu.
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("Window"));
        menu.addItem(&item(mtm, "Minimize", Some(sel!(performMiniaturize:)), "m"));
        add_menu(app, mtm, "Window", &menu);
        app.setWindowsMenu(Some(&menu));

        // The Help menu, whose item QEMU also calls Window.
        let menu = NSMenu::initWithTitle(NSMenu::alloc(mtm), ns_string!("Help"));
        menu.addItem(&item(mtm, "QEMU Documentation", Some(sel!(showQEMUDoc:)), "?"));
        add_menu(app, mtm, "Window", &menu);
    }

    /// `add_console_menu_entries()` and the heading of `addRemovableDevicesMenuItems()`, which
    /// has no drives to list.
    fn add_menu_entries(&self, app: &NSApplication) {
        let mtm = self.mtm;
        let Some(main) = app.mainMenu() else {
            return;
        };
        if let Some(menu) = main.itemWithTitle(ns_string!("View")).and_then(|i| i.submenu()) {
            menu.addItem(&NSMenuItem::separatorItem(mtm));
            for con in self.ds.consoles() {
                let entry = item(mtm, &con.label(), Some(sel!(displayConsole:)), "");
                entry.setTag(con.index() as isize);
                menu.addItem(&entry);
            }
        }
        if let Some(menu) = main.itemWithTitle(ns_string!("Machine")).and_then(|i| i.submenu()) {
            menu.addItem(&NSMenuItem::separatorItem(mtm));
            let heading = item(mtm, "Removable Media", None, "");
            heading.setEnabled(false);
            menu.addItem(&heading);
        }
    }
}

/// `cocoa_display_init()`: builds the application on the main thread and registers the listener.
/// [`run`] then runs the application. `about` is what `-version` prints, whose two lines go in
/// the About panel.
pub fn init(
    ds: Arc<DisplayState>,
    input: Arc<InputState>,
    opts: &DisplayOptions,
    name: Option<&str>,
    about: &str,
    hooks: Arc<dyn Hooks>,
) -> Result<(), u8> {
    let Some(mtm) = MainThreadMarker::new() else {
        error_report("cocoa: the display must be set up on the main thread");
        return Err(1);
    };
    let cocoa = match &opts.u {
        DisplayOptionsU::Cocoa(c) => c.clone(),
        _ => Default::default(),
    };
    // Before anything else asks for the shared application, so that it is a QemuApplication.
    // SAFETY: sharedApplication takes no arguments and returns the application.
    let app: Retained<NSApplication> =
        unsafe { msg_send![QemuApplication::class(), sharedApplication] };
    // A process started from a terminal gets a menu bar and a Dock icon.
    app.setActivationPolicy(NSApplicationActivationPolicy::Regular);

    let mut lines = about.lines();
    let about =
        (lines.next().unwrap_or_default().to_owned(), lines.next().unwrap_or_default().to_owned());
    let frame = Frame::default();
    let ui = Rc::new(Ui {
        mtm,
        ds: Arc::clone(&ds),
        input: Arc::clone(&input),
        hooks,
        name: name.map(ToOwned::to_owned),
        about,
        frame: Arc::clone(&frame),
        listener: Arc::new(Listener { frame }),
        con: RefCell::new(None),
        id: Cell::new(None),
        keys: RefCell::new(Keys::new(KbdState::new(None))),
        screen: Cell::new((640, 480)),
        grabbed: Cell::new(false),
        absolute: Cell::new(false),
        cursor_hide: Cell::new(true),
        allow_events: Cell::new(false),
        zoom_interpolation: Cell::new(CGInterpolationQuality::None),
        view: OnceCell::new(),
        pause_label: OnceCell::new(),
        pause_item: OnceCell::new(),
        resume_item: OnceCell::new(),
        controller: OnceCell::new(),
        events_tap: RefCell::new(None),
    });
    if UI.with(|u| u.set(Rc::clone(&ui))).is_err() {
        error_report("cocoa: the display is already set up");
        return Err(1);
    }

    let controller = QemuCocoaAppController::alloc(mtm).set_ivars(());
    // SAFETY: init is NSObject's designated initializer.
    let controller: Retained<QemuCocoaAppController> =
        unsafe { msg_send![super(controller), init] };
    ui.make_window(&controller);
    app.setDelegate(Some(ProtocolObject::from_ref(&*controller)));
    let _ = ui.controller.set(controller);

    let window = ui.view().window();
    if opts.full_screen == Some(true) {
        if let Some(w) = &window {
            w.toggleFullScreen(None);
        }
    }
    if cocoa.full_grab == Some(true) {
        ui.set_full_grab();
    }
    if opts.show_cursor == Some(true) {
        ui.cursor_hide.set(false);
    }
    {
        let mut keys = ui.keys.borrow_mut();
        if let Some(swap) = cocoa.swap_opt_cmd {
            keys.swap_opt_cmd = swap;
        }
        if cocoa.left_command_key == Some(false) {
            keys.left_command_key = false;
        }
    }
    if cocoa.zoom_to_fit == Some(true) {
        if let Some(w) = &window {
            w.setStyleMask(w.styleMask() | NSWindowStyleMask::Resizable);
        }
    }
    if cocoa.zoom_interpolation == Some(true) {
        ui.zoom_interpolation.set(CGInterpolationQuality::Low);
    }

    ui.create_initial_menus(&app);
    ui.add_menu_entries(&app);

    if let Some(con) = ds.lookup_default() {
        ui.con.replace(Some(con.clone()));
        ui.register(&con);
        ui.keys.borrow_mut().kbd = KbdState::new(Some(con));
    }
    input.add_mouse_mode_notifier(|| on_main(Ui::notify_mouse_mode_change));
    ui.notify_mouse_mode_change();
    ui.update_ui_info();
    Ok(())
}

/// `cocoa_main()`: runs the application, which never returns.
pub fn run() -> ! {
    if let Some(mtm) = MainThreadMarker::new() {
        NSApplication::sharedApplication(mtm).run();
    }
    std::process::abort()
}
