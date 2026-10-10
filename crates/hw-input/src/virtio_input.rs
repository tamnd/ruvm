// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-keyboard, virtio-mouse, virtio-tablet and virtio-multitouch, a port of
//! `hw/input/virtio-input.c` and `hw/input/virtio-input-hid.c`.
//!
//! The device has an event queue and a status queue of 64 entries each. Input events from the
//! UI become `virtio_input_event` records (type, code, value) and are held until the
//! `EV_SYN`/`SYN_REPORT` that ends the batch. Then the batch goes to the guest in one go, one
//! event per buffer, or is dropped whole when the guest has not made enough buffers available.
//! The status queue carries LED changes the other way.
//!
//! The config space is one `virtio_input_config` entry, picked by the `select` and `subsel`
//! bytes the guest writes: the name, the serial, the bus, vendor and product ids, the property
//! and event bitmaps and the absolute axis ranges. Its size is the largest entry plus the
//! 8 byte header, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - QEMU writes a batch into the event queue from the input handler itself. Here the handler
//!   finishes the batch under its own lock and then calls the kick set with
//!   [`VirtioInput::set_kick`], which takes the transport lock and runs
//!   [`VirtioInput::flush`]. Without a kick, batches wait for the guest's next kick of the event
//!   queue.
//! - An `EV_LED` status from the guest on a device whose handler takes no keys (the mouse,
//!   tablet and multitouch) is ignored. QEMU fails an assertion in
//!   `qemu_input_handler_set_leds_mask()` there.
//! - A serial of 129 to 255 bytes is a realize error where QEMU fails an assertion.
//! - The guest errors QEMU logs with `qemu_log_mask(LOG_GUEST_ERROR)` are not printed.
//!
//! The `virtio-input-host` passthrough device and vhost-user-input are not ported.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ruvm_base::{Error, Result};
use ruvm_hw_virtio::virtio::VIRTIO_CONFIG_S_DRIVER_OK;
use ruvm_hw_virtio::{VirtIODevice, VirtioDeviceClass};
use ruvm_qapi::types::{InputAxis, InputButton, InputMultiTouchType};
use ruvm_ui::console::{DisplayState, QemuConsole};
use ruvm_ui::input::{
    HandlerId, INPUT_EVENT_ABS_MAX, INPUT_EVENT_ABS_MIN, INPUT_EVENT_MASK_ABS,
    INPUT_EVENT_MASK_BTN, INPUT_EVENT_MASK_KEY, INPUT_EVENT_MASK_MTT, INPUT_EVENT_MASK_REL,
    InputHandler, InputState, QEMU_CAPS_LOCK_LED, QEMU_NUM_LOCK_LED, QEMU_SCROLL_LOCK_LED,
    QemuInputEvent,
};

/// `TYPE_VIRTIO_KEYBOARD`.
pub const TYPE_VIRTIO_KEYBOARD: &str = "virtio-keyboard-device";
/// `TYPE_VIRTIO_MOUSE`.
pub const TYPE_VIRTIO_MOUSE: &str = "virtio-mouse-device";
/// `TYPE_VIRTIO_TABLET`.
pub const TYPE_VIRTIO_TABLET: &str = "virtio-tablet-device";
/// `TYPE_VIRTIO_MULTITOUCH`.
pub const TYPE_VIRTIO_MULTITOUCH: &str = "virtio-multitouch-device";
/// `TYPE_VIRTIO_KEYBOARD_PCI`.
pub const TYPE_VIRTIO_KEYBOARD_PCI: &str = "virtio-keyboard-pci";
/// `TYPE_VIRTIO_MOUSE_PCI`.
pub const TYPE_VIRTIO_MOUSE_PCI: &str = "virtio-mouse-pci";
/// `TYPE_VIRTIO_TABLET_PCI`.
pub const TYPE_VIRTIO_TABLET_PCI: &str = "virtio-tablet-pci";
/// `TYPE_VIRTIO_MULTITOUCH_PCI`.
pub const TYPE_VIRTIO_MULTITOUCH_PCI: &str = "virtio-multitouch-pci";

/// `VIRTIO_ID_INPUT`.
pub const VIRTIO_ID_INPUT: u16 = 18;
/// `PCI_CLASS_INPUT_KEYBOARD`.
pub const PCI_CLASS_INPUT_KEYBOARD: u16 = 0x0900;
/// `PCI_CLASS_INPUT_MOUSE`.
pub const PCI_CLASS_INPUT_MOUSE: u16 = 0x0902;
/// `PCI_CLASS_INPUT_OTHER`.
pub const PCI_CLASS_INPUT_OTHER: u16 = 0x0980;

/// The size of the event and status queues.
pub const VIRTIO_INPUT_QUEUE_SIZE: u16 = 64;

pub const VIRTIO_INPUT_CFG_UNSET: u8 = 0x00;
pub const VIRTIO_INPUT_CFG_ID_NAME: u8 = 0x01;
pub const VIRTIO_INPUT_CFG_ID_SERIAL: u8 = 0x02;
pub const VIRTIO_INPUT_CFG_ID_DEVIDS: u8 = 0x03;
pub const VIRTIO_INPUT_CFG_PROP_BITS: u8 = 0x10;
pub const VIRTIO_INPUT_CFG_EV_BITS: u8 = 0x11;
pub const VIRTIO_INPUT_CFG_ABS_INFO: u8 = 0x12;

/// `sizeof(virtio_input_config)`: select, subsel, size, five reserved bytes and a 128 byte
/// union.
pub const VIRTIO_INPUT_CONFIG_SIZE: usize = 136;
/// `sizeof(virtio_input_event)`.
pub const VIRTIO_INPUT_EVENT_SIZE: usize = 8;

// The Linux event codes from `standard-headers/linux/input-event-codes.h`.
pub const EV_SYN: u16 = 0x00;
pub const EV_KEY: u16 = 0x01;
pub const EV_REL: u16 = 0x02;
pub const EV_ABS: u16 = 0x03;
pub const EV_LED: u16 = 0x11;
pub const EV_REP: u16 = 0x14;
pub const SYN_REPORT: u16 = 0;
pub const REL_X: u16 = 0x00;
pub const REL_Y: u16 = 0x01;
pub const REL_WHEEL: u16 = 0x08;
pub const ABS_X: u16 = 0x00;
pub const ABS_Y: u16 = 0x01;
pub const ABS_MT_SLOT: u16 = 0x2f;
pub const ABS_MT_POSITION_X: u16 = 0x35;
pub const ABS_MT_POSITION_Y: u16 = 0x36;
pub const ABS_MT_TRACKING_ID: u16 = 0x39;
pub const BTN_LEFT: u16 = 0x110;
pub const BTN_RIGHT: u16 = 0x111;
pub const BTN_MIDDLE: u16 = 0x112;
pub const BTN_SIDE: u16 = 0x113;
pub const BTN_EXTRA: u16 = 0x114;
pub const BTN_TOUCH: u16 = 0x14a;
pub const BTN_GEAR_DOWN: u16 = 0x150;
pub const BTN_GEAR_UP: u16 = 0x151;
pub const KEY_ESC: u16 = 1;
pub const KEY_REPLY: u16 = 232;
pub const LED_NUML: u16 = 0x00;
pub const LED_CAPSL: u16 = 0x01;
pub const LED_SCROLLL: u16 = 0x02;
pub const BUS_VIRTUAL: u16 = 0x06;
pub const INPUT_PROP_DIRECT: u16 = 0x01;

/// `INPUT_EVENT_SLOTS_MIN` and `INPUT_EVENT_SLOTS_MAX`.
const INPUT_EVENT_SLOTS_MIN: u32 = 0;
const INPUT_EVENT_SLOTS_MAX: u32 = 10;

/// The vendor id QEMU also gives its USB HID devices.
const VENDOR_QEMU: u16 = 0x0627;

/// Which of the four HID devices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtioInputKind {
    Keyboard,
    Mouse,
    Tablet,
    MultiTouch,
}

impl VirtioInputKind {
    /// The `-device` name of the bare device on a virtio-mmio bus.
    pub fn typename(self) -> &'static str {
        match self {
            VirtioInputKind::Keyboard => TYPE_VIRTIO_KEYBOARD,
            VirtioInputKind::Mouse => TYPE_VIRTIO_MOUSE,
            VirtioInputKind::Tablet => TYPE_VIRTIO_TABLET,
            VirtioInputKind::MultiTouch => TYPE_VIRTIO_MULTITOUCH,
        }
    }

    /// The `-device` name of the PCI function.
    pub fn pci_typename(self) -> &'static str {
        match self {
            VirtioInputKind::Keyboard => TYPE_VIRTIO_KEYBOARD_PCI,
            VirtioInputKind::Mouse => TYPE_VIRTIO_MOUSE_PCI,
            VirtioInputKind::Tablet => TYPE_VIRTIO_TABLET_PCI,
            VirtioInputKind::MultiTouch => TYPE_VIRTIO_MULTITOUCH_PCI,
        }
    }

    /// The name the device gives the virtio core, which the PCI transport turns into the
    /// `-pci` name.
    pub fn stem(self) -> &'static str {
        match self {
            VirtioInputKind::Keyboard => "virtio-keyboard",
            VirtioInputKind::Mouse => "virtio-mouse",
            VirtioInputKind::Tablet => "virtio-tablet",
            VirtioInputKind::MultiTouch => "virtio-multitouch",
        }
    }

    /// `VIRTIO_ID_NAME_*`, also the handler name `query-mice` shows.
    pub fn name(self) -> &'static str {
        match self {
            VirtioInputKind::Keyboard => "QEMU Virtio Keyboard",
            VirtioInputKind::Mouse => "QEMU Virtio Mouse",
            VirtioInputKind::Tablet => "QEMU Virtio Tablet",
            VirtioInputKind::MultiTouch => "QEMU Virtio MultiTouch",
        }
    }

    /// The `INPUT_EVENT_MASK_*` bits of the handler.
    pub fn mask(self) -> u32 {
        match self {
            VirtioInputKind::Keyboard => INPUT_EVENT_MASK_KEY,
            VirtioInputKind::Mouse => INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_REL,
            VirtioInputKind::Tablet => INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_ABS,
            VirtioInputKind::MultiTouch => INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_MTT,
        }
    }

    /// The PCI class `virtio_input_pci_realize()` and the instance inits pick.
    pub fn pci_class(self) -> u16 {
        match self {
            VirtioInputKind::Keyboard => PCI_CLASS_INPUT_KEYBOARD,
            VirtioInputKind::Mouse => PCI_CLASS_INPUT_MOUSE,
            VirtioInputKind::Tablet | VirtioInputKind::MultiTouch => PCI_CLASS_INPUT_OTHER,
        }
    }

    /// The config entries in the order the instance init adds them.
    pub fn configs(self) -> Vec<[u8; VIRTIO_INPUT_CONFIG_SIZE]> {
        let abs = |subsel: u16, min: u32, max: u32| {
            let mut c = entry(VIRTIO_INPUT_CFG_ABS_INFO, subsel as u8, 20);
            c[8..12].copy_from_slice(&min.to_le_bytes());
            c[12..16].copy_from_slice(&max.to_le_bytes());
            c
        };
        let (abs_min, abs_max) = (INPUT_EVENT_ABS_MIN as u32, INPUT_EVENT_ABS_MAX as u32);
        let mut v = vec![name_config(self.name()), devids_config(self)];
        match self {
            VirtioInputKind::Keyboard => {
                v.push(entry(VIRTIO_INPUT_CFG_EV_BITS, EV_REP as u8, 1));
                let mut led = entry(VIRTIO_INPUT_CFG_EV_BITS, EV_LED as u8, 1);
                led[8] = (1 << LED_NUML) | (1 << LED_CAPSL) | (1 << LED_SCROLLL);
                v.push(led);
                // Every key up to KEY_REPLY, as Linux's xen-kbdfront does.
                let mut keys =
                    entry(VIRTIO_INPUT_CFG_EV_BITS, EV_KEY as u8, KEY_REPLY.div_ceil(8) as u8);
                for i in KEY_ESC..KEY_REPLY {
                    keys[8 + usize::from(i / 8)] |= 1 << (i % 8);
                }
                v.push(keys);
            }
            VirtioInputKind::Mouse => {
                let mut rel = entry(VIRTIO_INPUT_CFG_EV_BITS, EV_REL as u8, 2);
                rel[8] = (1 << REL_X) | (1 << REL_Y);
                rel[9] = 1 << (REL_WHEEL - 8);
                v.push(rel);
                v.push(extend_config(&KEYMAP_BUTTON, VIRTIO_INPUT_CFG_EV_BITS, EV_KEY as u8));
            }
            VirtioInputKind::Tablet => {
                let mut ev = entry(VIRTIO_INPUT_CFG_EV_BITS, EV_ABS as u8, 1);
                ev[8] = (1 << ABS_X) | (1 << ABS_Y);
                v.push(ev);
                let mut rel = entry(VIRTIO_INPUT_CFG_EV_BITS, EV_REL as u8, 2);
                rel[9] = 1 << (REL_WHEEL - 8);
                v.push(rel);
                v.push(abs(ABS_X, abs_min, abs_max));
                v.push(abs(ABS_Y, abs_min, abs_max));
                v.push(extend_config(&KEYMAP_BUTTON, VIRTIO_INPUT_CFG_EV_BITS, EV_KEY as u8));
            }
            VirtioInputKind::MultiTouch => {
                v.push(abs(ABS_MT_SLOT, INPUT_EVENT_SLOTS_MIN, INPUT_EVENT_SLOTS_MAX));
                v.push(abs(ABS_MT_TRACKING_ID, INPUT_EVENT_SLOTS_MIN, INPUT_EVENT_SLOTS_MAX));
                v.push(abs(ABS_MT_POSITION_X, abs_min, abs_max));
                v.push(abs(ABS_MT_POSITION_Y, abs_min, abs_max));
                v.push(extend_config(&KEYMAP_BUTTON, VIRTIO_INPUT_CFG_EV_BITS, EV_KEY as u8));
                v.push(extend_config(&[INPUT_PROP_DIRECT], VIRTIO_INPUT_CFG_PROP_BITS, 0));
                v.push(extend_config(
                    &[ABS_MT_SLOT, ABS_MT_TRACKING_ID, ABS_MT_POSITION_X, ABS_MT_POSITION_Y],
                    VIRTIO_INPUT_CFG_EV_BITS,
                    EV_ABS as u8,
                ));
            }
        }
        v
    }
}

/// The codes of `keymap_button`, the buttons that have one.
const KEYMAP_BUTTON: [u16; 8] =
    [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE, BTN_GEAR_UP, BTN_GEAR_DOWN, BTN_SIDE, BTN_EXTRA, BTN_TOUCH];

/// `keymap_button[button]`, 0 for the horizontal wheel.
fn button_code(button: InputButton) -> u16 {
    match button {
        InputButton::Left => BTN_LEFT,
        InputButton::Right => BTN_RIGHT,
        InputButton::Middle => BTN_MIDDLE,
        InputButton::WheelUp => BTN_GEAR_UP,
        InputButton::WheelDown => BTN_GEAR_DOWN,
        InputButton::Side => BTN_SIDE,
        InputButton::Extra => BTN_EXTRA,
        InputButton::Touch => BTN_TOUCH,
        InputButton::WheelLeft | InputButton::WheelRight => 0,
    }
}

fn entry(select: u8, subsel: u8, size: u8) -> [u8; VIRTIO_INPUT_CONFIG_SIZE] {
    let mut c = [0; VIRTIO_INPUT_CONFIG_SIZE];
    c[0] = select;
    c[1] = subsel;
    c[2] = size;
    c
}

/// The `VIRTIO_INPUT_CFG_ID_NAME` entry, whose size counts the terminating NUL as the
/// `sizeof` of the string literal does.
fn name_config(name: &str) -> [u8; VIRTIO_INPUT_CONFIG_SIZE] {
    let mut c = entry(VIRTIO_INPUT_CFG_ID_NAME, 0, (name.len() + 1) as u8);
    c[8..8 + name.len()].copy_from_slice(name.as_bytes());
    c
}

/// The `VIRTIO_INPUT_CFG_ID_DEVIDS` entry: bus, vendor, product and version.
fn devids_config(kind: VirtioInputKind) -> [u8; VIRTIO_INPUT_CONFIG_SIZE] {
    let (product, version) = match kind {
        VirtioInputKind::Keyboard => (1, 1),
        VirtioInputKind::Mouse => (2, 2),
        VirtioInputKind::Tablet => (3, 2),
        VirtioInputKind::MultiTouch => (3, 1),
    };
    let mut c = entry(VIRTIO_INPUT_CFG_ID_DEVIDS, 0, 8);
    for (i, v) in [BUS_VIRTUAL, VENDOR_QEMU, product, version].into_iter().enumerate() {
        c[8 + 2 * i..10 + 2 * i].copy_from_slice(&v.to_le_bytes());
    }
    c
}

/// `virtio_input_extend_config()`: a bitmap entry with a bit for each code in `map`.
fn extend_config(map: &[u16], select: u8, subsel: u8) -> [u8; VIRTIO_INPUT_CONFIG_SIZE] {
    let mut c = entry(select, subsel, 0);
    let mut bmax = 0;
    for &bit in map.iter().filter(|&&b| b != 0) {
        let byte = usize::from(bit / 8);
        c[8 + byte] |= 1 << (bit % 8);
        bmax = bmax.max(byte + 1);
    }
    c[2] = bmax as u8;
    c
}

/// A `virtio_input_event`: type, code and value, little endian.
fn event(type_: u16, code: u16, value: u32) -> [u8; VIRTIO_INPUT_EVENT_SIZE] {
    let mut e = [0; VIRTIO_INPUT_EVENT_SIZE];
    e[0..2].copy_from_slice(&type_.to_le_bytes());
    e[2..4].copy_from_slice(&code.to_le_bytes());
    e[4..8].copy_from_slice(&value.to_le_bytes());
    e
}

/// The properties of the HID devices: `serial` from `virtio_input_properties` and `display`
/// and `head` from `virtio_input_hid_properties`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtioInputConf {
    pub serial: Option<String>,
    pub display: Option<String>,
    pub head: u32,
}

/// What the input handler shares with the device.
#[derive(Default)]
struct Hid {
    /// `VirtIOInput.active`.
    active: bool,
    /// `VirtIOInput.queue` up to `qindex`: the batch so far.
    batch: Vec<[u8; VIRTIO_INPUT_EVENT_SIZE]>,
    /// Finished batches the event queue has not taken yet.
    ready: VecDeque<Vec<[u8; VIRTIO_INPUT_EVENT_SIZE]>>,
}

type Kick = Arc<dyn Fn() + Send + Sync>;

#[derive(Default)]
struct Shared {
    hid: Mutex<Hid>,
    kick: Mutex<Option<Kick>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    /// `virtio_input_send()` up to the point where it writes into the queue: the event joins
    /// the batch, and a `SYN_REPORT` finishes the batch and kicks the device.
    fn send(&self, e: [u8; VIRTIO_INPUT_EVENT_SIZE]) {
        {
            let mut hid = lock(&self.hid);
            if !hid.active {
                return;
            }
            hid.batch.push(e);
            if e[0..4] != event(EV_SYN, SYN_REPORT, 0)[0..4] {
                return;
            }
            let batch = std::mem::take(&mut hid.batch);
            hid.ready.push_back(batch);
        }
        let kick = lock(&self.kick).clone();
        if let Some(kick) = kick {
            kick();
        }
    }
}

/// The `QemuInputHandler` of the device, `virtio_input_handle_event()` and
/// `virtio_input_handle_sync()`.
struct HidHandler {
    kind: VirtioInputKind,
    shared: Arc<Shared>,
}

impl InputHandler for HidHandler {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn mask(&self) -> u32 {
        self.kind.mask()
    }

    fn event(&self, _src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        let axis = |a: InputAxis, x: u16, y: u16| if a == InputAxis::X { x } else { y };
        match evt {
            QemuInputEvent::Key { key, down } => {
                self.shared.send(event(EV_KEY, *key as u16, u32::from(*down)));
            }
            QemuInputEvent::Btn(b) => {
                let wheel = matches!(b.button, InputButton::WheelUp | InputButton::WheelDown);
                if wheel && b.down {
                    let value = if b.button == InputButton::WheelUp { 1 } else { -1i32 as u32 };
                    self.shared.send(event(EV_REL, REL_WHEEL, value));
                } else {
                    let code = button_code(b.button);
                    if code != 0 {
                        self.shared.send(event(EV_KEY, code, u32::from(b.down)));
                    }
                }
            }
            QemuInputEvent::Rel(m) => {
                self.shared.send(event(EV_REL, axis(m.axis, REL_X, REL_Y), m.value as u32));
            }
            QemuInputEvent::Abs(m) => {
                self.shared.send(event(EV_ABS, axis(m.axis, ABS_X, ABS_Y), m.value as u32));
            }
            QemuInputEvent::Mtt(m) => {
                if m.type_ == InputMultiTouchType::Data {
                    let code = axis(m.axis, ABS_MT_POSITION_X, ABS_MT_POSITION_Y);
                    self.shared.send(event(EV_ABS, code, m.value as u32));
                } else {
                    self.shared.send(event(EV_ABS, ABS_MT_SLOT, m.slot as u32));
                    self.shared.send(event(EV_ABS, ABS_MT_TRACKING_ID, m.tracking_id as u32));
                }
            }
        }
    }

    fn sync(&self) {
        self.shared.send(event(EV_SYN, SYN_REPORT, 0));
    }
}

/// `VirtIOInputHID`: one of the four HID devices.
pub struct VirtioInput {
    kind: VirtioInputKind,
    conf: VirtioInputConf,
    input: Arc<InputState>,
    ds: Arc<DisplayState>,
    /// `cfg_list`.
    configs: Vec<[u8; VIRTIO_INPUT_CONFIG_SIZE]>,
    cfg_select: u8,
    cfg_subsel: u8,
    cfg_size: usize,
    shared: Arc<Shared>,
    /// `VirtIOInputHID.hs`, from realize on.
    hs: Option<HandlerId>,
    ledstate: u32,
}

impl fmt::Debug for VirtioInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtioInput")
            .field("kind", &self.kind)
            .field("conf", &self.conf)
            .field("cfg_select", &self.cfg_select)
            .field("cfg_subsel", &self.cfg_subsel)
            .field("cfg_size", &self.cfg_size)
            .field("ledstate", &self.ledstate)
            .finish_non_exhaustive()
    }
}

impl VirtioInput {
    /// The instance init of the kind's type. The handler is registered with `input` at
    /// realize and bound to the console `conf.display` names in `ds`.
    pub fn new(
        kind: VirtioInputKind,
        conf: VirtioInputConf,
        input: Arc<InputState>,
        ds: Arc<DisplayState>,
    ) -> Self {
        VirtioInput {
            kind,
            conf,
            input,
            ds,
            configs: kind.configs(),
            cfg_select: 0,
            cfg_subsel: 0,
            cfg_size: 0,
            shared: Arc::default(),
            hs: None,
            ledstate: 0,
        }
    }

    pub fn kind(&self) -> VirtioInputKind {
        self.kind
    }

    /// The input handler, once realized.
    pub fn handler(&self) -> Option<HandlerId> {
        self.hs
    }

    /// Sets what runs when a batch of events is ready, normally a call to
    /// [`VirtioInput::flush`] through the transport.
    pub fn set_kick(&self, kick: Option<Box<dyn Fn() + Send + Sync>>) {
        *lock(&self.shared.kick) = kick.map(Kick::from);
    }

    /// The rest of `virtio_input_send()` for every finished batch: each goes to the guest whole,
    /// one event per buffer of the event queue, or is dropped when there are not enough buffers.
    pub fn flush(&mut self, vdev: &mut VirtIODevice) {
        let ready = std::mem::take(&mut lock(&self.shared.hid).ready);
        for batch in ready {
            Self::deliver(vdev, &batch);
        }
    }

    fn deliver(vdev: &mut VirtIODevice, batch: &[[u8; VIRTIO_INPUT_EVENT_SIZE]]) {
        // virtqueue_unpop() is putting the index back and forgetting the chains.
        let start = vdev.last_avail_idx(0);
        let mut elems = Vec::with_capacity(batch.len());
        for _ in batch {
            match vdev.pop(0) {
                Some(chain) => elems.push(chain),
                None => {
                    for chain in &elems {
                        vdev.detach(0, chain);
                    }
                    if !elems.is_empty() {
                        vdev.set_last_avail_idx(0, start);
                    }
                    return;
                }
            }
        }
        let mem = Arc::clone(vdev.mem());
        for (chain, e) in elems.iter().zip(batch) {
            let len = chain.writer(&*mem).write(e).unwrap_or(0);
            vdev.push(0, chain, len as u32);
        }
        vdev.notify(0);
    }

    /// `virtio_input_handle_sts()`.
    fn handle_sts(&mut self, vdev: &mut VirtIODevice) {
        let mem = Arc::clone(vdev.mem());
        while let Some(chain) = vdev.pop(1) {
            let mut e = [0; VIRTIO_INPUT_EVENT_SIZE];
            let len = chain.reader(&*mem).read(&mut e).unwrap_or(0);
            self.handle_status(&e);
            vdev.push(1, &chain, len as u32);
        }
        vdev.notify(1);
    }

    /// `virtio_input_hid_handle_status()`.
    fn handle_status(&mut self, e: &[u8; VIRTIO_INPUT_EVENT_SIZE]) {
        let type_ = u16::from_le_bytes([e[0], e[1]]);
        let code = u16::from_le_bytes([e[2], e[3]]);
        let value = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
        if type_ != EV_LED {
            return;
        }
        let ledbit = match code {
            LED_NUML => QEMU_NUM_LOCK_LED,
            LED_CAPSL => QEMU_CAPS_LOCK_LED,
            LED_SCROLLL => QEMU_SCROLL_LOCK_LED,
            _ => 0,
        };
        if value != 0 {
            self.ledstate |= ledbit;
        } else {
            self.ledstate &= !ledbit;
        }
        match self.hs {
            Some(hs) if self.kind.mask() & INPUT_EVENT_MASK_KEY != 0 => {
                self.input.set_leds_mask(hs, self.ledstate);
            }
            _ => {}
        }
    }

    /// `virtio_input_hid_change_active()`.
    fn change_active(&self, active: bool) {
        if let Some(hs) = self.hs {
            if active {
                self.input.activate(hs);
            } else {
                self.input.deactivate(hs);
            }
        }
    }
}

impl Drop for VirtioInput {
    /// `virtio_input_hid_unrealize()`.
    fn drop(&mut self) {
        if let Some(hs) = self.hs.take() {
            self.input.unregister(hs);
        }
    }
}

impl VirtioDeviceClass for VirtioInput {
    /// `virtio_input_device_realize()` with `virtio_input_hid_realize()`.
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let handler = Arc::new(HidHandler { kind: self.kind, shared: Arc::clone(&self.shared) });
        let hs = self.input.register(handler);
        self.hs = Some(hs);
        if let Some(display) = &self.conf.display {
            // QEMU passes no Error here, so a console that is not there is not an error.
            let _ = self.input.bind(hs, &self.ds, display, self.conf.head);
        }

        // virtio_input_idstr_config(): the size is what snprintf() returns, kept to a byte.
        if let Some(serial) = &self.conf.serial {
            let size = serial.len() as u8;
            if usize::from(size) + 8 > VIRTIO_INPUT_CONFIG_SIZE {
                return Err(Error::generic(format!(
                    "serial '{serial}' is too long, the most is 128 bytes"
                )));
            }
            let mut c = entry(VIRTIO_INPUT_CFG_ID_SERIAL, 0, size);
            let n = serial.len().min(127);
            c[8..8 + n].copy_from_slice(&serial.as_bytes()[..n]);
            self.configs.push(c);
        }

        self.cfg_size = self.configs.iter().map(|c| usize::from(c[2])).max().unwrap_or(0) + 8;
        vdev.init(self.kind.stem(), VIRTIO_ID_INPUT, self.cfg_size);
        vdev.add_queue(VIRTIO_INPUT_QUEUE_SIZE)?;
        vdev.add_queue(VIRTIO_INPUT_QUEUE_SIZE)?;
        Ok(())
    }

    /// `virtio_input_get_features()`.
    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        Ok(features)
    }

    /// `virtio_input_get_config()`: the selected entry, or zeros.
    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let n = self.cfg_size.min(config.len());
        let found =
            self.configs.iter().find(|c| c[0] == self.cfg_select && c[1] == self.cfg_subsel);
        match found {
            Some(c) => config[..n].copy_from_slice(&c[..n]),
            None => config[..n].fill(0),
        }
    }

    /// `virtio_input_set_config()`.
    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        self.cfg_select = config.first().copied().unwrap_or(0);
        self.cfg_subsel = config.get(1).copied().unwrap_or(0);
        vdev.notify_config();
    }

    /// `virtio_input_set_status()`.
    fn set_status(&mut self, _vdev: &mut VirtIODevice, status: u8) -> Result<()> {
        if status & VIRTIO_CONFIG_S_DRIVER_OK != 0 {
            let was = std::mem::replace(&mut lock(&self.shared.hid).active, true);
            if !was {
                self.change_active(true);
            }
        }
        Ok(())
    }

    /// `virtio_input_reset()`. A batch that was not finished stays, as in QEMU.
    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        let was = {
            let mut hid = lock(&self.shared.hid);
            hid.ready.clear();
            std::mem::replace(&mut hid.active, false)
        };
        if was {
            self.change_active(false);
        }
    }

    /// `virtio_input_handle_evt()` does nothing in QEMU. Batches waiting for a kick go now.
    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        match queue {
            0 => self.flush(vdev),
            1 => self.handle_sts(vdev),
            _ => {}
        }
    }

    /// `virtio_input_post_load()`.
    fn post_load(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        let active = vdev.status() & VIRTIO_CONFIG_S_DRIVER_OK != 0;
        lock(&self.shared.hid).active = active;
        self.change_active(active);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bitmap(c: &[u8; VIRTIO_INPUT_CONFIG_SIZE]) -> Vec<u16> {
        let size = usize::from(c[2]);
        (0..size * 8).filter(|&b| c[8 + b / 8] & (1 << (b % 8)) != 0).map(|b| b as u16).collect()
    }

    fn find(kind: VirtioInputKind, select: u8, subsel: u16) -> [u8; VIRTIO_INPUT_CONFIG_SIZE] {
        *kind.configs().iter().find(|c| c[0] == select && c[1] == subsel as u8).unwrap()
    }

    #[test]
    fn names_and_ids() {
        let c = find(VirtioInputKind::Keyboard, VIRTIO_INPUT_CFG_ID_NAME, 0);
        assert_eq!(c[2], 21);
        assert_eq!(&c[8..29], b"QEMU Virtio Keyboard\0");
        let c = find(VirtioInputKind::Tablet, VIRTIO_INPUT_CFG_ID_DEVIDS, 0);
        assert_eq!(c[2], 8);
        assert_eq!(&c[8..16], &[6, 0, 0x27, 0x06, 3, 0, 2, 0]);
        let c = find(VirtioInputKind::MultiTouch, VIRTIO_INPUT_CFG_ID_DEVIDS, 0);
        assert_eq!(&c[8..16], &[6, 0, 0x27, 0x06, 3, 0, 1, 0]);
    }

    #[test]
    fn keyboard_bitmaps() {
        let keys = find(VirtioInputKind::Keyboard, VIRTIO_INPUT_CFG_EV_BITS, EV_KEY);
        assert_eq!(keys[2], 29);
        assert_eq!(bitmap(&keys), (1..232).collect::<Vec<_>>());
        let led = find(VirtioInputKind::Keyboard, VIRTIO_INPUT_CFG_EV_BITS, EV_LED);
        assert_eq!(bitmap(&led), vec![0, 1, 2]);
        let rep = find(VirtioInputKind::Keyboard, VIRTIO_INPUT_CFG_EV_BITS, EV_REP);
        assert_eq!((rep[2], bitmap(&rep)), (1, vec![]));
    }

    #[test]
    fn button_bitmap() {
        for kind in [VirtioInputKind::Mouse, VirtioInputKind::Tablet, VirtioInputKind::MultiTouch] {
            let c = find(kind, VIRTIO_INPUT_CFG_EV_BITS, EV_KEY);
            assert_eq!(c[2], 43);
            assert_eq!(bitmap(&c), vec![0x110, 0x111, 0x112, 0x113, 0x114, 0x14a, 0x150, 0x151]);
        }
        let rel = find(VirtioInputKind::Mouse, VIRTIO_INPUT_CFG_EV_BITS, EV_REL);
        assert_eq!(bitmap(&rel), vec![0, 1, 8]);
        let rel = find(VirtioInputKind::Tablet, VIRTIO_INPUT_CFG_EV_BITS, EV_REL);
        assert_eq!(bitmap(&rel), vec![8]);
    }

    #[test]
    fn multitouch_config() {
        let p = find(VirtioInputKind::MultiTouch, VIRTIO_INPUT_CFG_PROP_BITS, 0);
        assert_eq!(bitmap(&p), vec![1]);
        let ev = find(VirtioInputKind::MultiTouch, VIRTIO_INPUT_CFG_EV_BITS, EV_ABS);
        assert_eq!(bitmap(&ev), vec![0x2f, 0x35, 0x36, 0x39]);
        let slot = find(VirtioInputKind::MultiTouch, VIRTIO_INPUT_CFG_ABS_INFO, ABS_MT_SLOT);
        assert_eq!(&slot[2..3], &[20]);
        assert_eq!(&slot[8..16], &[0, 0, 0, 0, 10, 0, 0, 0]);
        let x = find(VirtioInputKind::Tablet, VIRTIO_INPUT_CFG_ABS_INFO, ABS_X);
        assert_eq!(&x[8..16], &[0, 0, 0, 0, 0xff, 0x7f, 0, 0]);
    }

    #[test]
    fn handler_translation() {
        let shared = Arc::new(Shared::default());
        lock(&shared.hid).active = true;
        let h = HidHandler { kind: VirtioInputKind::Tablet, shared: Arc::clone(&shared) };
        let btn =
            |button, down| QemuInputEvent::Btn(ruvm_qapi::types::InputBtnEvent { button, down });
        h.event(None, &btn(InputButton::WheelUp, true));
        h.event(None, &btn(InputButton::WheelUp, false));
        h.event(None, &btn(InputButton::WheelDown, true));
        h.event(None, &btn(InputButton::WheelLeft, true));
        h.event(None, &btn(InputButton::Left, true));
        let mv = ruvm_qapi::types::InputMoveEvent { axis: InputAxis::Y, value: 0x1234 };
        h.event(None, &QemuInputEvent::Abs(mv));
        assert!(lock(&shared.hid).ready.is_empty());
        h.sync();
        let hid = lock(&shared.hid);
        assert!(hid.batch.is_empty());
        assert_eq!(
            hid.ready[0],
            vec![
                event(EV_REL, REL_WHEEL, 1),
                event(EV_KEY, BTN_GEAR_UP, 0),
                event(EV_REL, REL_WHEEL, u32::MAX),
                event(EV_KEY, BTN_LEFT, 1),
                event(EV_ABS, ABS_Y, 0x1234),
                event(EV_SYN, SYN_REPORT, 0),
            ]
        );
    }

    #[test]
    fn inactive_drops() {
        let shared = Arc::new(Shared::default());
        let h = HidHandler { kind: VirtioInputKind::Keyboard, shared: Arc::clone(&shared) };
        h.event(None, &QemuInputEvent::Key { key: 30, down: true });
        h.sync();
        let hid = lock(&shared.hid);
        assert!(hid.batch.is_empty() && hid.ready.is_empty());
    }
}
