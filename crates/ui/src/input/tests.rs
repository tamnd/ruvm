// SPDX-License-Identifier: GPL-2.0-or-later

use std::sync::Mutex;

use ruvm_base::ClockType;
use ruvm_qapi::types::{
    InputEvent, InputKeyEvent, InputKeyEventWrapper, InputMoveEventWrapper, IntWrapper, KeyValue,
    KeyValueU, QKeyCode, QKeyCodeWrapper,
};

use super::*;

/// A handler that writes down what it gets.
struct Rec {
    name: &'static str,
    mask: u32,
    log: Mutex<Vec<String>>,
}

impl Rec {
    fn new(name: &'static str, mask: u32) -> Arc<Rec> {
        Arc::new(Rec { name, mask, log: Mutex::new(Vec::new()) })
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

impl InputHandler for Rec {
    fn name(&self) -> &str {
        self.name
    }

    fn mask(&self) -> u32 {
        self.mask
    }

    fn event(&self, _src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        let s = match evt {
            QemuInputEvent::Key { key, down } => format!("key {key} {down}"),
            QemuInputEvent::Btn(b) => format!("btn {} {}", b.button.as_str(), b.down),
            QemuInputEvent::Rel(m) => format!("rel {} {}", m.axis.as_str(), m.value),
            QemuInputEvent::Abs(m) => format!("abs {} {}", m.axis.as_str(), m.value),
            QemuInputEvent::Mtt(_) => "mtt".to_string(),
        };
        self.log.lock().unwrap().push(s);
    }

    fn sync(&self) {
        self.log.lock().unwrap().push("sync".to_string());
    }
}

fn qcode(q: QKeyCode) -> KeyValue {
    KeyValue { u: KeyValueU::Qcode(QKeyCodeWrapper { data: q }) }
}

fn number(n: i64) -> KeyValue {
    KeyValue { u: KeyValueU::Number(IntWrapper { data: n }) }
}

fn key_event(key: KeyValue, down: bool) -> InputEvent {
    InputEvent { u: InputEventU::Key(InputKeyEventWrapper { data: InputKeyEvent { key, down } }) }
}

#[test]
fn keycode_maps() {
    assert_eq!(qcode_to_linux(QKeyCode::A), 30);
    assert_eq!(linux_to_qcode(30), QKeyCode::A);
    assert_eq!(linux_to_qcode(100_000), QKeyCode::Unmapped);
    // Key number 0x9c is the keypad Enter, 0xc8 the up arrow.
    assert_eq!(key_number_to_linux(0x9c), 96);
    assert_eq!(key_number_to_linux(0xc8), 103);
    assert_eq!(key_number_to_linux(-1), 0);
    assert_eq!(key_number_to_qcode(0x1e), QKeyCode::A);
    assert_eq!(linux_to_scancode(30, true), [0x1e]);
    assert_eq!(linux_to_scancode(30, false), [0x9e]);
    assert_eq!(linux_to_scancode(103, false), [0xe0, 0xc8]);
    assert_eq!(linux_to_scancode(119, true), [0xe1, 0x1d, 0x45]);
    assert_eq!(linux_to_scancode(119, false), [0xe1, 0x9d, 0xc5]);
    // USB usage 4 is A, 0x28 Return and 0xe0 the left Control. The table stops at 252.
    assert_eq!(usb_to_linux(4), Some(30));
    assert_eq!(usb_to_linux(0x28), Some(28));
    assert_eq!(usb_to_linux(0xe0), Some(29));
    assert_eq!(usb_to_linux(252), None);
}

#[test]
fn handlers_and_mice() {
    let s = InputState::new();
    let kbd = Rec::new("kbd", INPUT_EVENT_MASK_KEY);
    let mouse = Rec::new("mouse", INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_REL);
    let tablet = Rec::new("tablet", INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_ABS);
    let changes = Arc::new(Mutex::new(0));
    let c = Arc::clone(&changes);
    s.add_mouse_mode_notifier(move || *c.lock().unwrap() += 1);
    s.register(kbd.clone());
    let m = s.register(mouse.clone());
    let t = s.register(tablet.clone());
    assert_eq!(*changes.lock().unwrap(), 3);
    assert!(!s.is_absolute(None));

    let mice = s.query_mice();
    let names: Vec<_> = mice.iter().map(|m| (m.name.as_str(), m.index, m.current)).collect();
    assert_eq!(names, [("tablet", 3, false), ("mouse", 2, true)]);

    s.activate(t);
    assert!(s.is_absolute(None));
    s.deactivate(t);
    assert!(!s.is_absolute(None));
    s.mouse_set(3).unwrap();
    assert!(s.is_absolute(None));
    assert_eq!(s.mouse_set(9).unwrap_err().to_string(), "Mouse at index '9' not found");
    assert_eq!(s.mouse_set(1).unwrap_err().to_string(), "Input device 'kbd' is not a mouse");

    s.queue_btn(None, InputButton::Left, true);
    s.queue_rel(None, InputAxis::X, 5);
    s.queue_abs(None, InputAxis::Y, 50, 0, 100);
    s.event_sync();
    assert_eq!(tablet.take(), ["btn left true", "abs y 16383", "sync"]);
    assert_eq!(mouse.take(), ["rel x 5", "sync"]);
    assert!(kbd.take().is_empty());

    s.unregister(t);
    s.unregister(m);
    assert!(s.query_mice().is_empty());
}

#[test]
fn update_buttons_and_scale() {
    let s = InputState::new();
    let mouse = Rec::new("mouse", INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_REL);
    s.register(mouse.clone());
    let map = [(InputButton::Left, 1), (InputButton::Middle, 2), (InputButton::Right, 4)];
    s.update_buttons(None, &map, 0b001, 0b100);
    assert_eq!(mouse.take(), ["btn left false", "btn right true"]);
    assert_eq!(scale_axis(640, 0, 640, 0, 0x7fff), 0x7fff);
    assert_eq!(scale_axis(3, 0, 0, 0, 0x7fff), 0x3fff);
}

#[test]
fn input_send_event() {
    let s = InputState::new();
    let ds = DisplayState::new();
    let kbd = Rec::new("kbd", INPUT_EVENT_MASK_KEY);
    s.register(kbd.clone());
    let rel = InputEvent {
        u: InputEventU::Rel(InputMoveEventWrapper {
            data: InputMoveEvent { axis: InputAxis::X, value: 1 },
        }),
    };
    let arg = |events| InputSendEventArg { device: None, head: None, events };
    let e = s.qmp_input_send_event(&ds, arg(vec![key_event(number(0x1e), true), rel])).unwrap_err();
    assert_eq!(e.to_string(), "Input handler not found for event type rel");
    assert!(kbd.take().is_empty());

    let events = vec![
        key_event(qcode(QKeyCode::ShiftR), true),
        key_event(number(0x1e), true),
        key_event(number(0), true),
    ];
    s.qmp_input_send_event(&ds, arg(events.clone())).unwrap();
    assert_eq!(kbd.take(), ["key 54 true", "key 30 true", "sync"]);

    let e = s
        .qmp_input_send_event(
            &ds,
            InputSendEventArg { device: Some("nope".into()), head: None, events: events.clone() },
        )
        .unwrap_err();
    assert_eq!(e.to_string(), "Device 'nope' not found");

    s.set_runstate_check(|| false);
    let e = s.qmp_input_send_event(&ds, arg(events)).unwrap_err();
    assert_eq!(e.to_string(), "VM not running");
}

#[test]
fn send_key_is_paced_on_the_virtual_clock() {
    let s = InputState::new();
    let clock = Clock::manual(ClockType::Virtual);
    s.set_clock(&clock);
    let kbd = Rec::new("kbd", INPUT_EVENT_MASK_KEY);
    s.register(kbd.clone());
    let arg = SendKeyArg {
        keys: vec![qcode(QKeyCode::Ctrl), qcode(QKeyCode::Alt), qcode(QKeyCode::Delete)],
        hold_time: Some(100),
    };
    s.qmp_send_key(arg).unwrap();
    assert_eq!(kbd.take(), ["key 29 true", "sync"]);
    clock.advance_to(99_000_000);
    assert!(kbd.take().is_empty());
    clock.advance_to(100_000_000);
    assert_eq!(kbd.take(), ["key 56 true", "sync"]);
    clock.advance_to(1_000_000_000);
    assert_eq!(
        kbd.take(),
        [
            "key 111 true",
            "sync",
            "key 111 false",
            "sync",
            "key 56 false",
            "sync",
            "key 29 false",
            "sync"
        ]
    );
    // The queue ran dry, so the next key goes out at once.
    s.send_key_linux(None, 30, true);
    assert_eq!(kbd.take(), ["key 30 true", "sync"]);
}

#[test]
fn leds_follow_the_first_keyboard() {
    let s = InputState::new();
    let k1 = s.register(Rec::new("k1", INPUT_EVENT_MASK_KEY));
    let k2 = s.register(Rec::new("k2", INPUT_EVENT_MASK_KEY));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (s2, seen2) = (Arc::clone(&s), Arc::clone(&seen));
    s.add_led_notifier(move || seen2.lock().unwrap().push(s2.get_leds_mask(None)));
    s.set_leds_mask(k2, QEMU_CAPS_LOCK_LED);
    s.set_leds_mask(k1, QEMU_NUM_LOCK_LED);
    s.activate(k2);
    assert_eq!(*seen.lock().unwrap(), [0, QEMU_NUM_LOCK_LED, QEMU_CAPS_LOCK_LED]);
}
