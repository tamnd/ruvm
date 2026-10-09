// SPDX-License-Identifier: GPL-2.0-or-later

//! The PS/2 keyboard and mouse, hw/input/ps2.c.
//!
//! [`Ps2Kbd`] is `PS2KbdState` and [`Ps2Mouse`] is `PS2MouseState`. Both are plain state with
//! no lock of their own, because the controller that owns them (the i8042 in
//! [`crate::pckbd`]) has to see every change of their IRQ output in order while it holds its
//! own lock. So every method that can move the IRQ output takes an `irq` callback, which is
//! the `PS2_DEVICE_IRQ` output line: it gets `true` for `qemu_set_irq(irq, 1)` and `false` for
//! `qemu_set_irq(irq, 0)`, one call per call QEMU makes.
//!
//! Input comes in through methods: [`Ps2Kbd::keyboard_event`] takes a Linux keycode,
//! [`Ps2Kbd::put_keycode`] takes a raw scancode, and the mouse has [`Ps2Mouse::rel_event`],
//! [`Ps2Mouse::button_event`] and [`Ps2Mouse::sync`]. The controller registers the
//! `QemuInputHandler`s that call them, and passes the LED state on to the input layer when
//! [`Ps2Kbd::take_leds_update`] has one.
//!
//! `vmstate_save` and `vmstate_load` on both devices move what the `ps2kbd` and `ps2mouse`
//! VMStates carry, as [`Ps2KbdVmState`] and [`Ps2MouseVmState`].
//!
//! Not ported: trace points, QOM registration and the wakeup requests.

pub use ruvm_qapi::types::{InputAxis, InputButton};

use crate::keymap::{LINUX_TO_ATSET1, LINUX_TO_ATSET2, LINUX_TO_ATSET3};

// Keyboard commands.

/// Set keyboard LEDs.
pub const KBD_CMD_SET_LEDS: u8 = 0xED;
pub const KBD_CMD_ECHO: u8 = 0xEE;
/// Get or set the scancode set.
pub const KBD_CMD_SCANCODE: u8 = 0xF0;
/// Get keyboard ID.
pub const KBD_CMD_GET_ID: u8 = 0xF2;
/// Set typematic rate.
pub const KBD_CMD_SET_RATE: u8 = 0xF3;
/// Enable scanning.
pub const KBD_CMD_ENABLE: u8 = 0xF4;
/// Reset and disable scanning.
pub const KBD_CMD_RESET_DISABLE: u8 = 0xF5;
/// Reset and enable scanning.
pub const KBD_CMD_RESET_ENABLE: u8 = 0xF6;
/// Reset.
pub const KBD_CMD_RESET: u8 = 0xFF;
/// Set make and break mode.
pub const KBD_CMD_SET_MAKE_BREAK: u8 = 0xFC;
/// Set typematic make and break mode.
pub const KBD_CMD_SET_TYPEMATIC: u8 = 0xFA;

// Keyboard replies.

/// Power on reset.
pub const KBD_REPLY_POR: u8 = 0xAA;
/// Keyboard ID.
pub const KBD_REPLY_ID: u8 = 0xAB;
/// Command ACK.
pub const KBD_REPLY_ACK: u8 = 0xFA;
/// Command NACK, send the command again.
pub const KBD_REPLY_RESEND: u8 = 0xFE;

// Mouse commands.

/// Set 1:1 scaling.
pub const AUX_SET_SCALE11: u8 = 0xE6;
/// Set 2:1 scaling.
pub const AUX_SET_SCALE21: u8 = 0xE7;
/// Set resolution.
pub const AUX_SET_RES: u8 = 0xE8;
/// Get scaling factor.
pub const AUX_GET_SCALE: u8 = 0xE9;
/// Set stream mode.
pub const AUX_SET_STREAM: u8 = 0xEA;
/// Poll.
pub const AUX_POLL: u8 = 0xEB;
/// Reset wrap mode.
pub const AUX_RESET_WRAP: u8 = 0xEC;
/// Set wrap mode.
pub const AUX_SET_WRAP: u8 = 0xEE;
/// Set remote mode.
pub const AUX_SET_REMOTE: u8 = 0xF0;
/// Get type.
pub const AUX_GET_TYPE: u8 = 0xF2;
/// Set sample rate.
pub const AUX_SET_SAMPLE: u8 = 0xF3;
/// Enable aux device.
pub const AUX_ENABLE_DEV: u8 = 0xF4;
/// Disable aux device.
pub const AUX_DISABLE_DEV: u8 = 0xF5;
pub const AUX_SET_DEFAULT: u8 = 0xF6;
/// Reset aux device.
pub const AUX_RESET: u8 = 0xFF;
/// Command byte ACK.
pub const AUX_ACK: u8 = 0xFA;

pub const MOUSE_STATUS_REMOTE: u8 = 0x40;
pub const MOUSE_STATUS_ENABLED: u8 = 0x20;
pub const MOUSE_STATUS_SCALE21: u8 = 0x10;

/// Queue size required by the PS/2 protocol.
pub const PS2_QUEUE_SIZE: i32 = 16;
/// Room for keyboard command replies on top of [`PS2_QUEUE_SIZE`].
pub const PS2_QUEUE_HEADROOM: i32 = 8;
/// Size of the ring buffer. Only [`PS2_QUEUE_SIZE`] scancodes plus [`PS2_QUEUE_HEADROOM`]
/// command replies are ever in it, the rest is there for migration compatibility in QEMU.
pub const PS2_BUFFER_SIZE: i32 = 256;

pub const PS2_MOUSE_BUTTON_LEFT: u8 = 0x01;
pub const PS2_MOUSE_BUTTON_RIGHT: u8 = 0x02;
pub const PS2_MOUSE_BUTTON_MIDDLE: u8 = 0x04;
pub const PS2_MOUSE_BUTTON_SIDE: u8 = 0x08;
pub const PS2_MOUSE_BUTTON_EXTRA: u8 = 0x10;

// Bits of the `modifiers` field of the keyboard.
const MOD_CTRL_L: u32 = 1 << 0;
const MOD_SHIFT_L: u32 = 1 << 1;
const MOD_ALT_L: u32 = 1 << 2;
const MOD_CTRL_R: u32 = 1 << 3;
const MOD_SHIFT_R: u32 = 1 << 4;
const MOD_ALT_R: u32 = 1 << 5;

// The Linux keycodes ps2.c looks at, from linux/input-event-codes.h.

pub const KEY_LEFTCTRL: u16 = 29;
pub const KEY_LEFTSHIFT: u16 = 42;
pub const KEY_RIGHTSHIFT: u16 = 54;
pub const KEY_LEFTALT: u16 = 56;
pub const KEY_RIGHTCTRL: u16 = 97;
pub const KEY_SYSRQ: u16 = 99;
pub const KEY_RIGHTALT: u16 = 100;
pub const KEY_PAUSE: u16 = 119;
pub const KEY_HANGEUL: u16 = 122;
pub const KEY_HANJA: u16 = 123;

/// Scancode set 2 to set 1 translation, what the i8042 does when bit 6 of its mode byte is
/// set. `translate_table`.
#[rustfmt::skip]
pub static TRANSLATE_TABLE: [u8; 256] = [
    0xff, 0x43, 0x41, 0x3f, 0x3d, 0x3b, 0x3c, 0x58,
    0x64, 0x44, 0x42, 0x40, 0x3e, 0x0f, 0x29, 0x59,
    0x65, 0x38, 0x2a, 0x70, 0x1d, 0x10, 0x02, 0x5a,
    0x66, 0x71, 0x2c, 0x1f, 0x1e, 0x11, 0x03, 0x5b,
    0x67, 0x2e, 0x2d, 0x20, 0x12, 0x05, 0x04, 0x5c,
    0x68, 0x39, 0x2f, 0x21, 0x14, 0x13, 0x06, 0x5d,
    0x69, 0x31, 0x30, 0x23, 0x22, 0x15, 0x07, 0x5e,
    0x6a, 0x72, 0x32, 0x24, 0x16, 0x08, 0x09, 0x5f,
    0x6b, 0x33, 0x25, 0x17, 0x18, 0x0b, 0x0a, 0x60,
    0x6c, 0x34, 0x35, 0x26, 0x27, 0x19, 0x0c, 0x61,
    0x6d, 0x73, 0x28, 0x74, 0x1a, 0x0d, 0x62, 0x6e,
    0x3a, 0x36, 0x1c, 0x1b, 0x75, 0x2b, 0x63, 0x76,
    0x55, 0x56, 0x77, 0x78, 0x79, 0x7a, 0x0e, 0x7b,
    0x7c, 0x4f, 0x7d, 0x4b, 0x47, 0x7e, 0x7f, 0x6f,
    0x52, 0x53, 0x50, 0x4c, 0x4d, 0x48, 0x01, 0x45,
    0x57, 0x4e, 0x51, 0x4a, 0x37, 0x49, 0x46, 0x54,
    0x80, 0x81, 0x82, 0x41, 0x54, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
    0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
    0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
    0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
    0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
    0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
    0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
    0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
    0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
    0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

/// `ps2_modifier_bit()`.
fn modifier_bit(key: u16) -> u32 {
    match key {
        KEY_LEFTCTRL => MOD_CTRL_L,
        KEY_RIGHTCTRL => MOD_CTRL_R,
        KEY_LEFTSHIFT => MOD_SHIFT_L,
        KEY_RIGHTSHIFT => MOD_SHIFT_R,
        KEY_LEFTALT => MOD_ALT_L,
        KEY_RIGHTALT => MOD_ALT_R,
        _ => 0,
    }
}

/// `PS2Queue`: a 256 byte ring holding scancodes and, in front of them, command replies.
///
/// Command replies are written in front of the read pointer, between `rptr` and `cwptr`, so
/// the reply to a command comes out before any scancode that was already queued. `cwptr` is
/// -1 when there are none.
#[derive(Clone, Debug)]
pub struct Ps2Queue {
    data: [u8; PS2_BUFFER_SIZE as usize],
    rptr: i32,
    wptr: i32,
    cwptr: i32,
    count: i32,
}

impl Default for Ps2Queue {
    fn default() -> Self {
        Ps2Queue { data: [0; PS2_BUFFER_SIZE as usize], rptr: 0, wptr: 0, cwptr: -1, count: 0 }
    }
}

impl Ps2Queue {
    /// Bytes waiting to be read.
    pub fn count(&self) -> usize {
        self.count as usize
    }
}

/// `vmstate_ps2_common` (version 3): the fields of `PS2State`, named as in QEMU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ps2CommonVmState {
    pub write_cmd: i32,
    /// `queue.rptr`.
    pub rptr: i32,
    /// `queue.wptr`.
    pub wptr: i32,
    /// `queue.count`.
    pub count: i32,
    /// `queue.data`.
    pub data: [u8; PS2_BUFFER_SIZE as usize],
    /// `queue.cwptr`. Only the keyboard sends it, in `ps2kbd/command_reply_queue`; the mouse
    /// keeps the destination's.
    pub cwptr: i32,
}

impl Default for Ps2CommonVmState {
    fn default() -> Self {
        Ps2CommonVmState {
            write_cmd: -1,
            rptr: 0,
            wptr: 0,
            count: 0,
            data: [0; PS2_BUFFER_SIZE as usize],
            cwptr: -1,
        }
    }
}

/// `vmstate_ps2_keyboard` (version 3) and its subsections, named as in QEMU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ps2KbdVmState {
    pub parent_obj: Ps2CommonVmState,
    pub scan_enabled: i32,
    pub translate: i32,
    pub scancode_set: i32,
    /// `ps2kbd/ledstate`.
    pub ledstate: i32,
    /// `ps2kbd/need_high_bit`.
    pub need_high_bit: bool,
}

impl Default for Ps2KbdVmState {
    fn default() -> Self {
        Ps2Kbd::new().vmstate_save()
    }
}

impl Ps2KbdVmState {
    /// `ps2_keyboard_ledstate_needed()`.
    pub fn ledstate_needed(&self) -> bool {
        self.ledstate != 0
    }

    /// `ps2_keyboard_need_high_bit_needed()`.
    pub fn need_high_bit_needed(&self) -> bool {
        self.need_high_bit
    }

    /// `ps2_keyboard_cqueue_needed()`.
    pub fn cqueue_needed(&self) -> bool {
        self.parent_obj.cwptr != -1
    }
}

/// `vmstate_ps2_mouse` (version 2), named as in QEMU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ps2MouseVmState {
    pub parent_obj: Ps2CommonVmState,
    pub mouse_status: u8,
    pub mouse_resolution: u8,
    pub mouse_sample_rate: u8,
    pub mouse_wrap: u8,
    pub mouse_type: u8,
    pub mouse_detect_state: u8,
    pub mouse_dx: i32,
    pub mouse_dy: i32,
    pub mouse_dz: i32,
    pub mouse_buttons: u8,
}

impl Default for Ps2MouseVmState {
    fn default() -> Self {
        Ps2Mouse::new().vmstate_save()
    }
}

/// `PS2State`, the part the keyboard and the mouse share: the queue and the argument byte
/// the device is waiting for.
#[derive(Clone, Debug)]
struct Ps2State {
    queue: Ps2Queue,
    /// The command whose argument byte comes next, or -1.
    write_cmd: i32,
}

impl Default for Ps2State {
    fn default() -> Self {
        Ps2State { queue: Ps2Queue::default(), write_cmd: -1 }
    }
}

impl Ps2State {
    fn vmstate_save(&self) -> Ps2CommonVmState {
        let q = &self.queue;
        Ps2CommonVmState {
            write_cmd: self.write_cmd,
            rptr: q.rptr,
            wptr: q.wptr,
            count: q.count,
            data: q.data,
            cwptr: q.cwptr,
        }
    }

    /// Loads the common fields, then `ps2_common_post_load()` bounds the queue.
    fn vmstate_load(&mut self, v: &Ps2CommonVmState) {
        self.write_cmd = v.write_cmd;
        let q = &mut self.queue;
        q.data = v.data;
        q.rptr = v.rptr;
        q.count = v.count;

        // Limit the number of queued command replies to PS2_QUEUE_HEADROOM.
        let mut ccount = 0;
        if v.cwptr != -1 {
            ccount = (v.cwptr.wrapping_sub(q.rptr) & (PS2_BUFFER_SIZE - 1)).min(PS2_QUEUE_HEADROOM);
        }

        // Limit the scancode queue size to PS2_QUEUE_SIZE.
        if q.count < ccount {
            q.count = ccount;
        } else if q.count > ccount + PS2_QUEUE_SIZE {
            q.count = ccount + PS2_QUEUE_SIZE;
        }

        // Sanitize rptr and recalculate wptr and cwptr.
        q.rptr &= PS2_BUFFER_SIZE - 1;
        q.wptr = (q.rptr + q.count) & (PS2_BUFFER_SIZE - 1);
        q.cwptr = if ccount != 0 { (q.rptr + ccount) & (PS2_BUFFER_SIZE - 1) } else { -1 };
    }

    /// `ps2_reset_queue()`.
    fn reset_queue(&mut self) {
        let q = &mut self.queue;
        q.rptr = 0;
        q.wptr = 0;
        q.cwptr = -1;
        q.count = 0;
    }

    /// `ps2_queue_empty()`.
    fn queue_empty(&self) -> bool {
        self.queue.count == 0
    }

    /// `ps2_queue_noirq()`.
    fn queue_noirq(&mut self, b: u8) {
        let q = &mut self.queue;
        if q.count >= PS2_QUEUE_SIZE {
            return;
        }
        q.data[q.wptr as usize] = b;
        q.wptr += 1;
        if q.wptr == PS2_BUFFER_SIZE {
            q.wptr = 0;
        }
        q.count += 1;
    }

    /// `ps2_queue()`, `ps2_queue_2()`, `ps2_queue_3()` and `ps2_queue_4()`: queues all of
    /// `bytes` or, when they do not fit, none of them.
    fn queue(&mut self, bytes: &[u8], irq: &mut dyn FnMut(bool)) {
        if PS2_QUEUE_SIZE - self.queue.count < bytes.len() as i32 {
            return;
        }
        for &b in bytes {
            self.queue_noirq(b);
        }
        irq(true);
    }

    /// `ps2_cqueue_data()`.
    fn cqueue_data(q: &mut Ps2Queue, b: u8) {
        q.data[q.cwptr as usize] = b;
        q.cwptr += 1;
        if q.cwptr >= PS2_BUFFER_SIZE {
            q.cwptr = 0;
        }
        q.count += 1;
    }

    /// `ps2_cqueue_1()`, `ps2_cqueue_2()` and `ps2_cqueue_3()`: puts a command reply in front
    /// of the queued scancodes.
    fn cqueue(&mut self, bytes: &[u8], irq: &mut dyn FnMut(bool)) {
        let q = &mut self.queue;
        q.rptr = (q.rptr - bytes.len() as i32) & (PS2_BUFFER_SIZE - 1);
        q.cwptr = q.rptr;
        for &b in bytes {
            Self::cqueue_data(q, b);
        }
        irq(true);
    }

    /// `ps2_cqueue_reset()`: drops command replies nobody read.
    fn cqueue_reset(&mut self) {
        let q = &mut self.queue;
        if q.cwptr == -1 {
            return;
        }
        let ccount = (q.cwptr - q.rptr) & (PS2_BUFFER_SIZE - 1);
        q.count -= ccount;
        q.rptr = q.cwptr;
        q.cwptr = -1;
    }

    /// `ps2_read_data()`.
    fn read_data(&mut self, irq: &mut dyn FnMut(bool)) -> u8 {
        let q = &mut self.queue;
        if q.count == 0 {
            // With nothing left, return the last byte again. EMM386 needs this. QEMU notes
            // that a timer would be needed to do it properly.
            let mut index = q.rptr - 1;
            if index < 0 {
                index = PS2_BUFFER_SIZE - 1;
            }
            q.data[index as usize]
        } else {
            let val = q.data[q.rptr as usize];
            q.rptr += 1;
            if q.rptr == PS2_BUFFER_SIZE {
                q.rptr = 0;
            }
            q.count -= 1;
            if q.rptr == q.cwptr {
                // The command reply queue is empty.
                q.cwptr = -1;
            }
            // Reading deasserts the IRQ, and it comes back if there is data left.
            let left = q.count != 0;
            irq(false);
            if left {
                irq(true);
            }
            val
        }
    }

    /// `ps2_reset_hold()`.
    fn reset_hold(&mut self) {
        self.write_cmd = -1;
        self.reset_queue();
    }
}

/// `PS2KbdState`, the `ps2-kbd` device: an MF2 AT keyboard.
#[derive(Clone, Debug)]
pub struct Ps2Kbd {
    common: Ps2State,
    scan_enabled: bool,
    translate: bool,
    /// 1 is XT, 2 is AT, 3 is PS/2.
    scancode_set: i32,
    ledstate: u8,
    need_high_bit: bool,
    /// `MOD_*` bits.
    modifiers: u32,
    /// Set where QEMU calls `qemu_input_handler_set_leds_mask()`.
    leds_update: bool,
}

impl Default for Ps2Kbd {
    fn default() -> Self {
        Self::new()
    }
}

impl Ps2Kbd {
    /// A keyboard in its reset state.
    pub fn new() -> Self {
        let mut s = Ps2Kbd {
            common: Ps2State::default(),
            scan_enabled: false,
            translate: false,
            scancode_set: 0,
            ledstate: 0,
            need_high_bit: false,
            modifiers: 0,
            leds_update: false,
        };
        s.reset_hold();
        s
    }

    /// `ps2_kbd_reset_hold()`. The LED state and a half received translated break code are
    /// left alone, as in QEMU.
    fn reset_hold(&mut self) {
        self.common.reset_hold();
        self.scan_enabled = true;
        self.translate = false;
        self.scancode_set = 2;
        self.modifiers = 0;
    }

    /// Device reset: the hold phase, then `ps2_reset_exit()` lowers the IRQ.
    pub fn reset(&mut self, irq: &mut dyn FnMut(bool)) {
        self.reset_hold();
        irq(false);
    }

    /// The state `vmstate_ps2_keyboard` sends.
    pub fn vmstate_save(&self) -> Ps2KbdVmState {
        Ps2KbdVmState {
            parent_obj: self.common.vmstate_save(),
            scan_enabled: i32::from(self.scan_enabled),
            translate: i32::from(self.translate),
            scancode_set: self.scancode_set,
            ledstate: i32::from(self.ledstate),
            need_high_bit: self.need_high_bit,
        }
    }

    /// Loads what `vmstate_ps2_keyboard` carried, then the queue part of
    /// `ps2_kbd_post_load()`. The caller has already set `scancode_set` to 2 for version 2
    /// streams. The IRQ output is left alone; the controller's pending bits carry it.
    pub fn vmstate_load(&mut self, v: &Ps2KbdVmState) {
        self.common.vmstate_load(&v.parent_obj);
        self.scan_enabled = v.scan_enabled != 0;
        self.translate = v.translate != 0;
        self.scancode_set = v.scancode_set;
        self.ledstate = v.ledstate as u8;
        self.need_high_bit = v.need_high_bit;
        // The post_load of `ps2kbd/ledstate`, which is only sent when the state is not zero.
        if v.ledstate != 0 {
            self.leds_update = true;
        }
    }

    /// Whether the keyboard sends scancodes, cleared by [`KBD_CMD_RESET_DISABLE`].
    pub fn scan_enabled(&self) -> bool {
        self.scan_enabled
    }

    /// Whether scancodes are translated to set 1 on the way out.
    pub fn translate(&self) -> bool {
        self.translate
    }

    /// The scancode set in use, 1 to 3.
    pub fn scancode_set(&self) -> i32 {
        self.scancode_set
    }

    /// The LED byte last written with [`KBD_CMD_SET_LEDS`]: bit 0 scroll lock, bit 1 num lock,
    /// bit 2 caps lock.
    pub fn ledstate(&self) -> u8 {
        self.ledstate
    }

    /// The LED state, if it was set since the last call. The owner passes it on to
    /// `qemu_input_handler_set_leds_mask()` once its lock is dropped.
    pub fn take_leds_update(&mut self) -> Option<u8> {
        std::mem::take(&mut self.leds_update).then_some(self.ledstate)
    }

    /// The output queue.
    pub fn queue(&self) -> &Ps2Queue {
        &self.common.queue
    }

    /// `ps2_queue_empty()`.
    pub fn queue_empty(&self) -> bool {
        self.common.queue_empty()
    }

    /// `ps2_queue()`: queues one byte as if the keyboard had sent it.
    pub fn queue_byte(&mut self, b: u8, irq: &mut dyn FnMut(bool)) {
        self.common.queue(&[b], irq);
    }

    /// `ps2_read_data()`.
    pub fn read_data(&mut self, irq: &mut dyn FnMut(bool)) -> u8 {
        self.common.read_data(irq)
    }

    /// `ps2_keyboard_set_translation()`: `true` translates to set 1, which is what the i8042
    /// asks for when bit 6 of its mode byte is set.
    pub fn set_translation(&mut self, mode: bool) {
        self.translate = mode;
    }

    /// `ps2_put_keycode()`: queues `keycode`, an untranslated scancode in the current set,
    /// translating it to set 1 when that is on.
    pub fn put_keycode(&mut self, keycode: u8, irq: &mut dyn FnMut(bool)) {
        if self.translate {
            if keycode == 0xf0 {
                self.need_high_bit = true;
            } else if self.need_high_bit {
                self.common.queue(&[TRANSLATE_TABLE[usize::from(keycode)] | 0x80], irq);
                self.need_high_bit = false;
            } else {
                self.common.queue(&[TRANSLATE_TABLE[usize::from(keycode)]], irq);
            }
        } else {
            self.common.queue(&[keycode], irq);
        }
    }

    fn put_keycodes(&mut self, codes: &[u8], irq: &mut dyn FnMut(bool)) {
        for &c in codes {
            self.put_keycode(c, irq);
        }
    }

    /// `ps2_keyboard_event()`: a key identified by its Linux keycode (`KEY_*` from
    /// linux/input-event-codes.h, which is what QEMU's input layer hands the device) went up
    /// or down. Ignored while scanning is disabled. Keys with no code in the current set are
    /// dropped.
    pub fn keyboard_event(&mut self, key: u16, down: bool, irq: &mut dyn FnMut(bool)) {
        // Do not process events while disabled, to prevent stream corruption.
        if !self.scan_enabled {
            return;
        }

        let m = modifier_bit(key);
        if down {
            self.modifiers |= m;
        } else {
            self.modifiers &= !m;
        }
        let mods = self.modifiers;

        if self.scancode_set == 1 {
            if key == KEY_PAUSE {
                if (mods & (MOD_CTRL_L | MOD_CTRL_R)) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0x46, 0xe0, 0xc6], irq);
                    }
                } else if down {
                    self.put_keycodes(&[0xe1, 0x1d, 0x45, 0xe1, 0x9d, 0xc5], irq);
                }
            } else if key == KEY_SYSRQ {
                if (mods & MOD_ALT_L) != 0 {
                    if down {
                        self.put_keycodes(&[0xb8, 0x38, 0x54], irq);
                    } else {
                        self.put_keycodes(&[0xd4, 0xb8, 0x38], irq);
                    }
                } else if (mods & MOD_ALT_R) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0xb8, 0xe0, 0x38, 0x54], irq);
                    } else {
                        self.put_keycodes(&[0xd4, 0xe0, 0xb8, 0xe0, 0x38], irq);
                    }
                } else if (mods & (MOD_SHIFT_L | MOD_CTRL_L | MOD_SHIFT_R | MOD_CTRL_R)) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0x37], irq);
                    } else {
                        self.put_keycodes(&[0xe0, 0xb7], irq);
                    }
                } else if down {
                    self.put_keycodes(&[0xe0, 0x2a, 0xe0, 0x37], irq);
                } else {
                    self.put_keycodes(&[0xe0, 0xb7, 0xe0, 0xaa], irq);
                }
            } else if (key == KEY_HANGEUL || key == KEY_HANJA) && !down {
                // Ignore release for these keys.
            } else {
                let mut keycode = LINUX_TO_ATSET1.get(usize::from(key)).copied().unwrap_or(0);
                if keycode != 0 {
                    if (keycode & 0xff00) != 0 {
                        self.put_keycode((keycode >> 8) as u8, irq);
                    }
                    if !down {
                        keycode |= 0x80;
                    }
                    self.put_keycode(keycode as u8, irq);
                }
            }
        } else if self.scancode_set == 2 {
            if key == KEY_PAUSE {
                if (mods & (MOD_CTRL_L | MOD_CTRL_R)) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0x7e, 0xe0, 0xf0, 0x7e], irq);
                    }
                } else if down {
                    self.put_keycodes(&[0xe1, 0x14, 0x77, 0xe1, 0xf0, 0x14, 0xf0, 0x77], irq);
                }
            } else if key == KEY_SYSRQ {
                if (mods & MOD_ALT_L) != 0 {
                    if down {
                        self.put_keycodes(&[0xf0, 0x11, 0x11, 0x84], irq);
                    } else {
                        self.put_keycodes(&[0xf0, 0x84, 0xf0, 0x11, 0x11], irq);
                    }
                } else if (mods & MOD_ALT_R) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0xf0, 0x11, 0xe0, 0x11, 0x84], irq);
                    } else {
                        self.put_keycodes(&[0xf0, 0x84, 0xe0, 0xf0, 0x11, 0xe0, 0x11], irq);
                    }
                } else if (mods & (MOD_SHIFT_L | MOD_CTRL_L | MOD_SHIFT_R | MOD_CTRL_R)) != 0 {
                    if down {
                        self.put_keycodes(&[0xe0, 0x7c], irq);
                    } else {
                        self.put_keycodes(&[0xe0, 0xf0, 0x7c], irq);
                    }
                } else if down {
                    self.put_keycodes(&[0xe0, 0x12, 0xe0, 0x7c], irq);
                } else {
                    self.put_keycodes(&[0xe0, 0xf0, 0x7c, 0xe0, 0xf0, 0x12], irq);
                }
            } else if (key == KEY_HANGEUL || key == KEY_HANJA) && !down {
                // Ignore release for these keys.
            } else {
                let keycode = LINUX_TO_ATSET2.get(usize::from(key)).copied().unwrap_or(0);
                if keycode != 0 {
                    if (keycode & 0xff00) != 0 {
                        self.put_keycode((keycode >> 8) as u8, irq);
                    }
                    if !down {
                        self.put_keycode(0xf0, irq);
                    }
                    self.put_keycode(keycode as u8, irq);
                }
            }
        } else if self.scancode_set == 3 {
            let keycode = LINUX_TO_ATSET3.get(usize::from(key)).copied().unwrap_or(0);
            if keycode != 0 {
                // QEMU has a FIXME here: the break code should be configurable per key.
                if !down {
                    self.put_keycode(0xf0, irq);
                }
                // Every set 3 code fits in a byte.
                self.put_keycode(keycode as u8, irq);
            }
        }
    }

    /// `ps2_set_ledstate()`.
    fn set_ledstate(&mut self, ledstate: u8) {
        self.ledstate = ledstate;
        self.leds_update = true;
    }

    /// `ps2_reset_keyboard()`, what the reset commands do. Unlike a device reset it keeps the
    /// translation mode and the modifiers.
    fn reset_keyboard(&mut self) {
        self.scan_enabled = true;
        self.scancode_set = 2;
        self.common.reset_queue();
        self.set_ledstate(0);
    }

    /// `ps2_write_keyboard()`: a byte from the host.
    pub fn write(&mut self, val: u8, irq: &mut dyn FnMut(bool)) {
        let ps2 = &mut self.common;
        ps2.cqueue_reset();
        match ps2.write_cmd {
            c if c == i32::from(KBD_CMD_SET_MAKE_BREAK) => {
                ps2.cqueue(&[KBD_REPLY_ACK], irq);
                ps2.write_cmd = -1;
            }
            c if c == i32::from(KBD_CMD_SCANCODE) => {
                if val == 0 {
                    let set = self.scancode_set as u8;
                    let reply =
                        if self.translate { TRANSLATE_TABLE[usize::from(set)] } else { set };
                    ps2.cqueue(&[KBD_REPLY_ACK, reply], irq);
                } else if (1..=3).contains(&val) {
                    self.scancode_set = i32::from(val);
                    ps2.cqueue(&[KBD_REPLY_ACK], irq);
                } else {
                    ps2.cqueue(&[KBD_REPLY_RESEND], irq);
                }
                ps2.write_cmd = -1;
            }
            c if c == i32::from(KBD_CMD_SET_LEDS) => {
                ps2.cqueue(&[KBD_REPLY_ACK], irq);
                ps2.write_cmd = -1;
                self.set_ledstate(val);
            }
            c if c == i32::from(KBD_CMD_SET_RATE) => {
                ps2.cqueue(&[KBD_REPLY_ACK], irq);
                ps2.write_cmd = -1;
            }
            _ => self.command(val, irq),
        }
    }

    /// The `write_cmd == -1` half of `ps2_write_keyboard()`: a command byte.
    fn command(&mut self, val: u8, irq: &mut dyn FnMut(bool)) {
        match val {
            0x00 => self.common.cqueue(&[KBD_REPLY_ACK], irq),
            0x05 => self.common.cqueue(&[KBD_REPLY_RESEND], irq),
            KBD_CMD_GET_ID => {
                // An MF2 AT keyboard.
                let id = if self.translate { 0x41 } else { 0x83 };
                self.common.cqueue(&[KBD_REPLY_ACK, KBD_REPLY_ID, id], irq);
            }
            KBD_CMD_ECHO => self.common.cqueue(&[KBD_CMD_ECHO], irq),
            KBD_CMD_ENABLE => {
                self.scan_enabled = true;
                self.common.cqueue(&[KBD_REPLY_ACK], irq);
            }
            KBD_CMD_SCANCODE | KBD_CMD_SET_LEDS | KBD_CMD_SET_RATE | KBD_CMD_SET_MAKE_BREAK => {
                self.common.write_cmd = i32::from(val);
                self.common.cqueue(&[KBD_REPLY_ACK], irq);
            }
            KBD_CMD_RESET_DISABLE => {
                self.reset_keyboard();
                self.scan_enabled = false;
                self.common.cqueue(&[KBD_REPLY_ACK], irq);
            }
            KBD_CMD_RESET_ENABLE => {
                self.reset_keyboard();
                self.scan_enabled = true;
                self.common.cqueue(&[KBD_REPLY_ACK], irq);
            }
            KBD_CMD_RESET => {
                self.reset_keyboard();
                self.common.cqueue(&[KBD_REPLY_ACK, KBD_REPLY_POR], irq);
            }
            KBD_CMD_SET_TYPEMATIC => self.common.cqueue(&[KBD_REPLY_ACK], irq),
            _ => self.common.cqueue(&[KBD_REPLY_RESEND], irq),
        }
    }
}

/// The `bmap` table of `ps2_mouse_event()`. Wheels and touch have no button bit.
fn ps2_bit(button: InputButton) -> u8 {
    match button {
        InputButton::Left => PS2_MOUSE_BUTTON_LEFT,
        InputButton::Middle => PS2_MOUSE_BUTTON_MIDDLE,
        InputButton::Right => PS2_MOUSE_BUTTON_RIGHT,
        InputButton::Side => PS2_MOUSE_BUTTON_SIDE,
        InputButton::Extra => PS2_MOUSE_BUTTON_EXTRA,
        _ => 0,
    }
}

/// `PS2MouseState`, the `ps2-mouse` device. It speaks plain PS/2 and turns into an
/// IntelliMouse (type 3) or IntelliMouse Explorer (type 4) after the usual sample rate knocks.
#[derive(Clone, Debug)]
pub struct Ps2Mouse {
    common: Ps2State,
    mouse_status: u8,
    mouse_resolution: u8,
    mouse_sample_rate: u8,
    mouse_wrap: bool,
    /// 0 is PS/2, 3 is IMPS/2, 4 is IMEX.
    mouse_type: u8,
    mouse_detect_state: u8,
    // Current movement, needed for remote mode.
    mouse_dx: i32,
    mouse_dy: i32,
    mouse_dz: i32,
    mouse_dw: i32,
    mouse_buttons: u8,
}

impl Default for Ps2Mouse {
    fn default() -> Self {
        Self::new()
    }
}

impl Ps2Mouse {
    /// A mouse in its reset state.
    pub fn new() -> Self {
        let mut s = Ps2Mouse {
            common: Ps2State::default(),
            mouse_status: 0,
            mouse_resolution: 0,
            mouse_sample_rate: 0,
            mouse_wrap: false,
            mouse_type: 0,
            mouse_detect_state: 0,
            mouse_dx: 0,
            mouse_dy: 0,
            mouse_dz: 0,
            mouse_dw: 0,
            mouse_buttons: 0,
        };
        s.reset_hold();
        s
    }

    /// `ps2_mouse_reset_hold()`.
    fn reset_hold(&mut self) {
        self.common.reset_hold();
        self.mouse_status = 0;
        self.mouse_resolution = 0;
        self.mouse_sample_rate = 0;
        self.mouse_wrap = false;
        self.mouse_type = 0;
        self.mouse_detect_state = 0;
        self.mouse_dx = 0;
        self.mouse_dy = 0;
        self.mouse_dz = 0;
        self.mouse_dw = 0;
        self.mouse_buttons = 0;
    }

    /// Device reset: the hold phase, then `ps2_reset_exit()` lowers the IRQ.
    pub fn reset(&mut self, irq: &mut dyn FnMut(bool)) {
        self.reset_hold();
        irq(false);
    }

    /// The state `vmstate_ps2_mouse` sends.
    pub fn vmstate_save(&self) -> Ps2MouseVmState {
        Ps2MouseVmState {
            parent_obj: self.common.vmstate_save(),
            mouse_status: self.mouse_status,
            mouse_resolution: self.mouse_resolution,
            mouse_sample_rate: self.mouse_sample_rate,
            mouse_wrap: u8::from(self.mouse_wrap),
            mouse_type: self.mouse_type,
            mouse_detect_state: self.mouse_detect_state,
            mouse_dx: self.mouse_dx,
            mouse_dy: self.mouse_dy,
            mouse_dz: self.mouse_dz,
            mouse_buttons: self.mouse_buttons,
        }
    }

    /// Loads what `vmstate_ps2_mouse` carried, then `ps2_mouse_post_load()`. The IRQ output is
    /// left alone; the controller's pending bits carry it.
    pub fn vmstate_load(&mut self, v: &Ps2MouseVmState) {
        self.common.vmstate_load(&v.parent_obj);
        self.mouse_status = v.mouse_status;
        self.mouse_resolution = v.mouse_resolution;
        self.mouse_sample_rate = v.mouse_sample_rate;
        self.mouse_wrap = v.mouse_wrap != 0;
        self.mouse_type = v.mouse_type;
        self.mouse_detect_state = v.mouse_detect_state;
        self.mouse_dx = v.mouse_dx;
        self.mouse_dy = v.mouse_dy;
        self.mouse_dz = v.mouse_dz;
        self.mouse_buttons = v.mouse_buttons;
    }

    /// The status byte: `MOUSE_STATUS_*` plus the button bits.
    pub fn status(&self) -> u8 {
        self.mouse_status
    }

    pub fn resolution(&self) -> u8 {
        self.mouse_resolution
    }

    pub fn sample_rate(&self) -> u8 {
        self.mouse_sample_rate
    }

    /// Whether wrap (echo) mode is on.
    pub fn wrap(&self) -> bool {
        self.mouse_wrap
    }

    /// The device ID: 0 for a PS/2 mouse, 3 for IMPS/2, 4 for IMEX.
    pub fn mouse_type(&self) -> u8 {
        self.mouse_type
    }

    /// The output queue.
    pub fn queue(&self) -> &Ps2Queue {
        &self.common.queue
    }

    /// `ps2_queue_empty()`.
    pub fn queue_empty(&self) -> bool {
        self.common.queue_empty()
    }

    /// `ps2_queue()`: queues one byte as if the mouse had sent it.
    pub fn queue_byte(&mut self, b: u8, irq: &mut dyn FnMut(bool)) {
        self.common.queue(&[b], irq);
    }

    /// `ps2_read_data()`.
    pub fn read_data(&mut self, irq: &mut dyn FnMut(bool)) -> u8 {
        self.common.read_data(irq)
    }

    /// `ps2_mouse_send_packet()`: queues one movement packet if it fits. Returns whether it
    /// did.
    fn send_packet(&mut self, irq: &mut dyn FnMut(bool)) -> bool {
        // IMPS/2 and IMEX send 4 bytes, PS/2 sends 3.
        let needed = if self.mouse_type != 0 { 4 } else { 3 };

        if PS2_QUEUE_SIZE - self.common.queue.count < needed {
            return false;
        }

        // QEMU wonders whether the range should be 8 bits.
        let dx1 = self.mouse_dx.clamp(-127, 127);
        let dy1 = self.mouse_dy.clamp(-127, 127);
        let mut dz1 = self.mouse_dz;
        let mut dw1 = self.mouse_dw;
        let b = 0x08
            | (u8::from(dx1 < 0) << 4)
            | (u8::from(dy1 < 0) << 5)
            | (self.mouse_buttons & 0x07);
        self.common.queue_noirq(b);
        self.common.queue_noirq(dx1 as u8);
        self.common.queue_noirq(dy1 as u8);
        // The extra byte of IMPS/2 and IMEX.
        match self.mouse_type {
            3 => {
                dz1 = dz1.clamp(-127, 127);
                self.common.queue_noirq(dz1 as u8);
                self.mouse_dz -= dz1;
                self.mouse_dw = 0;
            }
            4 => {
                // This matches what Linux expects for exps/2 in
                // drivers/input/mouse/psmouse-base.c. Pressing or releasing the 4th or 5th
                // button at the same moment as a horizontal scroll loses the button change,
                // since at this point there is no telling whether the buttons changed.
                let b = if dw1 != 0 {
                    dw1 = dw1.clamp(-31, 31);
                    // Linux expects the horizontal scroll value in the low 6 bits.
                    self.mouse_dw -= dw1;
                    (dw1 as u8 & 0x3f) | 0x40
                } else {
                    dz1 = dz1.clamp(-7, 7);
                    self.mouse_dz -= dz1;
                    (dz1 as u8 & 0x0f) | ((self.mouse_buttons & 0x18) << 1)
                };
                self.common.queue_noirq(b);
            }
            _ => {
                // Ignore the wheels when they are not supported.
                self.mouse_dz = 0;
                self.mouse_dw = 0;
            }
        }

        irq(true);

        // Update the deltas.
        self.mouse_dx -= dx1;
        self.mouse_dy -= dy1;

        true
    }

    /// The `INPUT_EVENT_KIND_REL` half of `ps2_mouse_event()`: relative motion as the input
    /// layer reports it, so a positive `Y` is down the screen. Ignored while the mouse is
    /// disabled. Nothing is sent until [`Ps2Mouse::sync`].
    pub fn rel_event(&mut self, axis: InputAxis, value: i32) {
        // Movement is not recorded while disabled.
        if (self.mouse_status & MOUSE_STATUS_ENABLED) == 0 {
            return;
        }
        match axis {
            InputAxis::X => self.mouse_dx = self.mouse_dx.wrapping_add(value),
            InputAxis::Y => self.mouse_dy = self.mouse_dy.wrapping_sub(value),
        }
    }

    /// The `INPUT_EVENT_KIND_BTN` half of `ps2_mouse_event()`. Wheel buttons count one step
    /// each time they go down. Ignored while the mouse is disabled.
    pub fn button_event(&mut self, button: InputButton, down: bool) {
        if (self.mouse_status & MOUSE_STATUS_ENABLED) == 0 {
            return;
        }
        if down {
            self.mouse_buttons |= ps2_bit(button);
            if button == InputButton::WheelUp {
                self.mouse_dz -= 1;
            } else if button == InputButton::WheelDown {
                self.mouse_dz += 1;
            }

            if button == InputButton::WheelRight {
                self.mouse_dw -= 1;
            } else if button == InputButton::WheelLeft {
                self.mouse_dw += 1;
            }
        } else {
            self.mouse_buttons &= !ps2_bit(button);
        }
    }

    /// `ps2_mouse_sync()`: the end of a batch of events. In stream mode this sends packets
    /// until the movement is used up or the queue is full.
    pub fn sync(&mut self, irq: &mut dyn FnMut(bool)) {
        // Do not sync while disabled, to prevent stream corruption.
        if (self.mouse_status & MOUSE_STATUS_ENABLED) == 0 {
            return;
        }
        if (self.mouse_status & MOUSE_STATUS_REMOTE) == 0 {
            // Not remote, so send the event. Big movements take several packets.
            while self.send_packet(irq) {
                if self.mouse_dx == 0
                    && self.mouse_dy == 0
                    && self.mouse_dz == 0
                    && self.mouse_dw == 0
                {
                    break;
                }
            }
        }
    }

    /// `ps2_mouse_fake_event()`: one step right, used by vmmouse to get the guest to poll.
    pub fn fake_event(&mut self, irq: &mut dyn FnMut(bool)) {
        self.mouse_dx += 1;
        self.sync(irq);
    }

    /// `ps2_write_mouse()`: a byte from the host.
    pub fn write(&mut self, val: u8, irq: &mut dyn FnMut(bool)) {
        match self.common.write_cmd {
            c if c == i32::from(AUX_SET_SAMPLE) => {
                self.mouse_sample_rate = val;
                // Detect IMPS/2 (200, 100, 80) or IMEX (200, 200, 80).
                match self.mouse_detect_state {
                    1 => {
                        if val == 100 {
                            self.mouse_detect_state = 2;
                        } else if val == 200 {
                            self.mouse_detect_state = 3;
                        } else {
                            self.mouse_detect_state = 0;
                        }
                    }
                    2 => {
                        if val == 80 {
                            // IMPS/2.
                            self.mouse_type = 3;
                        }
                        self.mouse_detect_state = 0;
                    }
                    3 => {
                        if val == 80 {
                            // IMEX.
                            self.mouse_type = 4;
                        }
                        self.mouse_detect_state = 0;
                    }
                    _ => {
                        if val == 200 {
                            self.mouse_detect_state = 1;
                        }
                    }
                }
                self.common.queue(&[AUX_ACK], irq);
                self.common.write_cmd = -1;
            }
            c if c == i32::from(AUX_SET_RES) => {
                self.mouse_resolution = val;
                self.common.queue(&[AUX_ACK], irq);
                self.common.write_cmd = -1;
            }
            _ => self.command(val, irq),
        }
    }

    /// The `write_cmd == -1` half of `ps2_write_mouse()`: a command byte.
    fn command(&mut self, val: u8, irq: &mut dyn FnMut(bool)) {
        if self.mouse_wrap {
            if val == AUX_RESET_WRAP {
                self.mouse_wrap = false;
                self.common.queue(&[AUX_ACK], irq);
                return;
            } else if val != AUX_RESET {
                self.common.queue(&[val], irq);
                return;
            }
        }
        match val {
            AUX_SET_SCALE11 => {
                self.mouse_status &= !MOUSE_STATUS_SCALE21;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_SET_SCALE21 => {
                self.mouse_status |= MOUSE_STATUS_SCALE21;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_SET_STREAM => {
                self.mouse_status &= !MOUSE_STATUS_REMOTE;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_SET_WRAP => {
                self.mouse_wrap = true;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_SET_REMOTE => {
                self.mouse_status |= MOUSE_STATUS_REMOTE;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_GET_TYPE => {
                self.common.queue(&[AUX_ACK, self.mouse_type], irq);
            }
            AUX_SET_RES | AUX_SET_SAMPLE => {
                self.common.write_cmd = i32::from(val);
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_GET_SCALE => {
                let reply =
                    [AUX_ACK, self.mouse_status, self.mouse_resolution, self.mouse_sample_rate];
                self.common.queue(&reply, irq);
            }
            AUX_POLL => {
                self.common.queue(&[AUX_ACK], irq);
                self.send_packet(irq);
            }
            AUX_ENABLE_DEV => {
                self.mouse_status |= MOUSE_STATUS_ENABLED;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_DISABLE_DEV => {
                self.mouse_status &= !MOUSE_STATUS_ENABLED;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_SET_DEFAULT => {
                self.mouse_sample_rate = 100;
                self.mouse_resolution = 2;
                self.mouse_status = 0;
                self.common.queue(&[AUX_ACK], irq);
            }
            AUX_RESET => {
                self.mouse_sample_rate = 100;
                self.mouse_resolution = 2;
                self.mouse_status = 0;
                self.mouse_type = 0;
                self.common.reset_queue();
                self.common.queue(&[AUX_ACK, 0xaa, self.mouse_type], irq);
            }
            _ => {}
        }
    }
}
