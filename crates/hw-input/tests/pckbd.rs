// SPDX-License-Identifier: GPL-2.0-or-later

//! Register level tests of the i8042 and the PS/2 keyboard and mouse behind it, plus the port
//! 0x60 and 0x64 sequence of QEMU's tests/qtest/dbus-display-test.c.

use std::sync::{Arc, Mutex};

use ruvm_base::ClockType;
use ruvm_hw_core::{Clock, IrqLine};
use ruvm_hw_input::pckbd::*;
use ruvm_hw_input::ps2::*;
use ruvm_mem::{AccessCtx, AccessSize, MmioOps};

type Log = Arc<Mutex<Vec<i32>>>;

fn recorder() -> (IrqLine, Log) {
    let log: Log = Arc::new(Mutex::new(Vec::new()));
    let l = log.clone();
    (IrqLine::from_fn(move |level| l.lock().unwrap().push(level)), log)
}

fn last(log: &Log) -> i32 {
    log.lock().unwrap().last().copied().unwrap_or(0)
}

struct Rig {
    clock: Arc<Clock>,
    kbc: Arc<I8042>,
    irq1: Log,
    irq12: Log,
    a20: Log,
    reset: Log,
}

impl Rig {
    fn new() -> Self {
        Self::with_props(I8042Props::default())
    }

    fn with_props(props: I8042Props) -> Self {
        let clock = Clock::manual(ClockType::Virtual);
        let kbc = I8042::new(clock.clone(), props).unwrap();
        let (l1, irq1) = recorder();
        let (l12, irq12) = recorder();
        let (la, a20) = recorder();
        let (lr, reset) = recorder();
        kbc.kbd_irq().connect(l1);
        kbc.mouse_irq().connect(l12);
        kbc.a20_out().connect(la);
        kbc.reset_out().connect(lr);
        Rig { clock, kbc, irq1, irq12, a20, reset }
    }

    fn cmd(&self, v: u8) {
        self.kbc.write_command(v);
    }

    fn outb(&self, v: u8) {
        self.kbc.write_data(v);
    }

    fn inb(&self) -> u8 {
        self.kbc.read_data()
    }

    fn status(&self) -> u8 {
        self.kbc.read_status()
    }

    /// Reads bytes while the output buffer is full.
    fn drain(&self) -> Vec<u8> {
        let mut v = Vec::new();
        while self.status() & KBD_STAT_OBF != 0 {
            v.push(self.inb());
        }
        v
    }

    /// Sends bytes to the keyboard, reading the replies after each one the way a driver
    /// does. Replies left unread are dropped by the next byte.
    fn kbd_send(&self, bytes: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        for &b in bytes {
            self.outb(b);
            v.extend(self.drain());
        }
        v
    }

    fn irq1(&self) -> i32 {
        last(&self.irq1)
    }

    fn irq12(&self) -> i32 {
        last(&self.irq12)
    }

    /// Enables the mouse and turns on reporting.
    fn enable_mouse(&self) {
        self.cmd(KBD_CCMD_MOUSE_ENABLE);
        self.aux(AUX_ENABLE_DEV);
        assert_eq!(self.drain(), [AUX_ACK]);
    }

    fn aux(&self, v: u8) {
        self.cmd(KBD_CCMD_WRITE_MOUSE);
        self.outb(v);
    }
}

#[test]
fn reset_state() {
    let r = Rig::new();
    let regs = r.kbc.regs();
    assert_eq!(regs.status, KBD_STAT_CMD | KBD_STAT_UNLOCKED);
    assert_eq!(regs.mode, KBD_MODE_KBD_INT | KBD_MODE_MOUSE_INT);
    assert_eq!(regs.outport, KBD_OUT_RESET | KBD_OUT_A20 | KBD_OUT_ONES);
    assert_eq!(r.irq1(), 0);
    assert_eq!(r.irq12(), 0);
    // Reset does not drive the A20 line, as in QEMU.
    assert!(r.a20.lock().unwrap().is_empty());
    let k = r.kbc.kbd();
    assert!(k.scan_enabled());
    assert!(!k.translate());
    assert_eq!(k.scancode_set(), 2);
}

#[test]
fn bad_irq_properties_are_rejected() {
    let clock = Clock::manual(ClockType::Virtual);
    let p = I8042Props { kbd_irq: 16, ..I8042Props::default() };
    assert!(I8042::new(clock.clone(), p).is_err());
    let p = I8042Props { mouse_irq: 200, ..I8042Props::default() };
    assert!(I8042::new(clock.clone(), p).is_err());
    // Throttling without extended state is turned off with a warning.
    let p = I8042Props { kbd_throttle: true, extended_state: false, ..I8042Props::default() };
    assert!(!I8042::new(clock, p).unwrap().props().kbd_throttle);
}

/// `test_dbus_display_keyboard()`: Enter pressed and released, read back in set 2.
#[test]
fn qtest_dbus_display_keyboard() {
    let r = Rig::new();
    assert_eq!(r.status() & 0x1, 0);
    assert_eq!(r.inb(), 0);

    // qnum 0x1c is Enter, Linux KEY_ENTER.
    r.kbc.key_event(28, true);
    assert_eq!(r.status() & 0x1, 1);
    assert_eq!(r.inb(), 0x5a);

    r.kbc.key_event(28, false);
    assert_eq!(r.status() & 0x1, 1);
    assert_eq!(r.inb(), 0xf0);
    assert_eq!(r.inb(), 0x5a);
}

#[test]
fn self_test_returns_0x55() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_SELF_TEST);
    assert_eq!(r.status() & (KBD_STAT_OBF | KBD_STAT_SELFTEST | KBD_STAT_MOUSE_OBF), 0x05);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0x55);
    assert_eq!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 0);
    // The self test bit stays set.
    assert_ne!(r.status() & KBD_STAT_SELFTEST, 0);
    // With nothing new, the data port returns the last byte again.
    assert_eq!(r.inb(), 0x55);
}

#[test]
fn interface_tests_and_ports() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_KBD_TEST);
    assert_eq!(r.drain(), [0x00]);
    r.cmd(KBD_CCMD_TEST_MOUSE);
    assert_eq!(r.drain(), [0x00]);
    r.cmd(KBD_CCMD_READ_INPORT);
    assert_eq!(r.drain(), [0x80]);
    r.cmd(KBD_CCMD_READ_OUTPORT);
    assert_eq!(r.drain(), [0xcf]);
    // Unknown and pulse-nothing commands do nothing.
    r.cmd(KBD_CCMD_GET_VERSION);
    r.cmd(0xF1);
    r.cmd(KBD_CCMD_NO_OP);
    assert!(r.drain().is_empty());
    assert!(r.reset.lock().unwrap().is_empty());
}

#[test]
fn output_port_shows_buffer_state() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_WRITE_OBUF);
    r.outb(0x12);
    assert_ne!(r.kbc.regs().outport & KBD_OUT_OBF, 0);
    assert_eq!(r.inb(), 0x12);
    assert_eq!(r.kbc.regs().outport & (KBD_OUT_OBF | KBD_OUT_MOUSE_OBF), 0);
    r.cmd(KBD_CCMD_WRITE_AUX_OBUF);
    r.outb(0x34);
    assert_eq!(r.kbc.regs().outport & (KBD_OUT_OBF | KBD_OUT_MOUSE_OBF), 0x30);
}

#[test]
fn command_byte_read_and_write() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_READ_MODE);
    assert_eq!(r.drain(), [0x03]);

    r.cmd(KBD_CCMD_WRITE_MODE);
    assert_eq!(r.kbc.regs().write_cmd, KBD_CCMD_WRITE_MODE);
    r.outb(0x47);
    assert_eq!(r.kbc.regs().write_cmd, 0);
    assert_eq!(r.kbc.regs().mode, 0x47);
    assert!(r.kbc.kbd().translate());
    r.cmd(KBD_CCMD_READ_MODE);
    assert_eq!(r.drain(), [0x47]);

    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(0x03);
    assert!(!r.kbc.kbd().translate());

    // The interface disable commands set their mode bits.
    r.cmd(KBD_CCMD_KBD_DISABLE);
    r.cmd(KBD_CCMD_MOUSE_DISABLE);
    assert_eq!(r.kbc.regs().mode, 0x33);
    r.cmd(KBD_CCMD_KBD_ENABLE);
    r.cmd(KBD_CCMD_MOUSE_ENABLE);
    assert_eq!(r.kbc.regs().mode, 0x03);
}

#[test]
fn keyboard_reset_returns_ack_and_por() {
    let r = Rig::new();
    r.outb(KBD_CMD_RESET);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), KBD_REPLY_ACK);
    // The second byte refills the buffer and raises IRQ1 again.
    assert_eq!(r.irq1(), 1);
    assert_ne!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.inb(), KBD_REPLY_POR);
    assert_eq!(r.irq1(), 0);
    assert_eq!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1.lock().unwrap().as_slice(), [1, 0, 1, 0]);
}

#[test]
fn keyboard_commands() {
    let r = Rig::new();
    assert_eq!(r.kbd_send(&[KBD_CMD_SET_LEDS, 0x07]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);
    assert_eq!(r.kbc.kbd().ledstate(), 7);

    r.outb(KBD_CMD_ECHO);
    assert_eq!(r.drain(), [KBD_CMD_ECHO]);

    r.outb(KBD_CMD_GET_ID);
    assert_eq!(r.drain(), [KBD_REPLY_ACK, KBD_REPLY_ID, 0x83]);

    assert_eq!(r.kbd_send(&[KBD_CMD_SET_RATE, 0x20]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);

    r.outb(KBD_CMD_SET_TYPEMATIC);
    assert_eq!(r.drain(), [KBD_REPLY_ACK]);
    assert_eq!(r.kbd_send(&[KBD_CMD_SET_MAKE_BREAK, 0x1c]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);

    r.outb(0x00);
    assert_eq!(r.drain(), [KBD_REPLY_ACK]);
    r.outb(0x05);
    assert_eq!(r.drain(), [KBD_REPLY_RESEND]);
    r.outb(0x42);
    assert_eq!(r.drain(), [KBD_REPLY_RESEND]);

    // Scancode set: query, set, bad value.
    assert_eq!(r.kbd_send(&[KBD_CMD_SCANCODE, 0]), [KBD_REPLY_ACK, KBD_REPLY_ACK, 2]);
    assert_eq!(r.kbd_send(&[KBD_CMD_SCANCODE, 3]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);
    assert_eq!(r.kbc.kbd().scancode_set(), 3);
    assert_eq!(r.kbd_send(&[KBD_CMD_SCANCODE, 4]), [KBD_REPLY_ACK, KBD_REPLY_RESEND]);
    assert_eq!(r.kbc.kbd().scancode_set(), 3);

    // Reset and disable: back to set 2, LEDs off, no scanning.
    r.outb(KBD_CMD_RESET_DISABLE);
    assert_eq!(r.drain(), [KBD_REPLY_ACK]);
    let k = r.kbc.kbd();
    assert_eq!(k.scancode_set(), 2);
    assert_eq!(k.ledstate(), 0);
    assert!(!k.scan_enabled());
    r.kbc.key_event(30, true);
    assert!(r.drain().is_empty());

    r.outb(KBD_CMD_ENABLE);
    assert_eq!(r.drain(), [KBD_REPLY_ACK]);
    r.kbc.key_event(30, true);
    assert_eq!(r.drain(), [0x1c]);

    assert_eq!(r.kbd_send(&[KBD_CMD_RESET_DISABLE, KBD_CMD_RESET_ENABLE]), [KBD_REPLY_ACK; 2]);
    assert!(r.kbc.kbd().scan_enabled());
}

#[test]
fn unread_replies_are_dropped_by_the_next_command() {
    let r = Rig::new();
    r.outb(KBD_CMD_GET_ID);
    assert_eq!(r.inb(), KBD_REPLY_ACK);
    // AB 83 are still queued; the next command throws them away.
    r.outb(KBD_CMD_ECHO);
    assert_eq!(r.drain(), [KBD_CMD_ECHO]);
}

#[test]
fn replies_go_ahead_of_queued_scancodes() {
    let r = Rig::new();
    // Hold the output buffer with a controller byte so the keys stay in the queue.
    r.cmd(KBD_CCMD_WRITE_OBUF);
    r.outb(0x99);
    r.kbc.put_keycode(0x1c);
    r.kbc.put_keycode(0x32);
    // Sending to the keyboard does not disturb the full buffer.
    r.outb(KBD_CMD_ECHO);
    assert_eq!(r.drain(), [0x99, KBD_CMD_ECHO, 0x1c, 0x32]);
}

#[test]
fn scancode_translation() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(KBD_MODE_KCC | KBD_MODE_SYS | KBD_MODE_KBD_INT | KBD_MODE_MOUSE_INT);

    // KEY_A: set 2 1c, translated to set 1 1e and 9e.
    r.kbc.key_event(30, true);
    r.kbc.key_event(30, false);
    assert_eq!(r.drain(), [0x1e, 0x9e]);

    // KEY_UP: e0 75 and e0 f0 75 in set 2.
    r.kbc.key_event(103, true);
    r.kbc.key_event(103, false);
    assert_eq!(r.drain(), [0xe0, 0x48, 0xe0, 0xc8]);

    // Raw set 2 codes go through the same table.
    r.kbc.put_keycode(0x5a);
    r.kbc.put_keycode(0xf0);
    r.kbc.put_keycode(0x5a);
    assert_eq!(r.drain(), [0x1c, 0x9c]);

    // The scancode set query and the ID are translated too.
    assert_eq!(r.kbd_send(&[KBD_CMD_SCANCODE, 0]), [KBD_REPLY_ACK, KBD_REPLY_ACK, 0x41]);
    r.outb(KBD_CMD_GET_ID);
    assert_eq!(r.drain(), [KBD_REPLY_ACK, KBD_REPLY_ID, 0x41]);

    // The translate table itself.
    assert_eq!(TRANSLATE_TABLE[0x1c], 0x1e);
    assert_eq!(TRANSLATE_TABLE[0x76], 0x01);
    assert_eq!(TRANSLATE_TABLE[0x80], 0x80);
}

#[test]
fn scancode_sets_1_and_3() {
    let r = Rig::new();
    assert_eq!(r.kbd_send(&[KBD_CMD_SCANCODE, 1]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);
    r.kbc.key_event(30, true);
    r.kbc.key_event(30, false);
    r.kbc.key_event(103, true);
    r.kbc.key_event(103, false);
    assert_eq!(r.drain(), [0x1e, 0x9e, 0xe0, 0x48, 0xe0, 0xc8]);

    // Pause in set 1, and ignored on release.
    r.kbc.key_event(KEY_PAUSE, true);
    r.kbc.key_event(KEY_PAUSE, false);
    assert_eq!(r.drain(), [0xe1, 0x1d, 0x45, 0xe1, 0x9d, 0xc5]);

    r.kbd_send(&[KBD_CMD_SCANCODE, 3]);
    r.kbc.key_event(30, true);
    r.kbc.key_event(30, false);
    assert_eq!(r.drain(), [0x1c, 0xf0, 0x1c]);
}

#[test]
fn special_keys_in_set_2() {
    let r = Rig::new();
    r.kbc.key_event(KEY_PAUSE, true);
    assert_eq!(r.drain(), [0xe1, 0x14, 0x77, 0xe1, 0xf0, 0x14, 0xf0, 0x77]);

    // Ctrl+Pause is Break.
    r.kbc.key_event(KEY_LEFTCTRL, true);
    r.kbc.key_event(KEY_PAUSE, true);
    r.kbc.key_event(KEY_LEFTCTRL, false);
    assert_eq!(r.drain(), [0x14, 0xe0, 0x7e, 0xe0, 0xf0, 0x7e, 0xf0, 0x14]);

    // Print Screen alone.
    r.kbc.key_event(KEY_SYSRQ, true);
    r.kbc.key_event(KEY_SYSRQ, false);
    assert_eq!(r.drain(), [0xe0, 0x12, 0xe0, 0x7c, 0xe0, 0xf0, 0x7c, 0xe0, 0xf0, 0x12]);

    // Hangeul sends nothing on release.
    r.kbc.key_event(KEY_HANGEUL, false);
    assert!(r.drain().is_empty());
}

#[test]
fn keyboard_queue_holds_16_bytes() {
    let r = Rig::new();
    for i in 0..20u8 {
        r.kbc.put_keycode(i + 1);
    }
    assert_eq!(r.kbc.kbd().queue().count(), 16);
    let got = r.drain();
    assert_eq!(got, (1..=16).collect::<Vec<u8>>());
    // An empty queue repeats its last byte.
    assert_eq!(r.inb(), 16);
}

#[test]
fn irq_follows_the_output_buffer() {
    let r = Rig::new();
    r.kbc.put_keycode(0x1c);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.irq12(), 0);
    r.inb();
    assert_eq!(r.irq1(), 0);

    // With IRQs off in the mode byte the buffer fills but the line stays low.
    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(0x00);
    r.kbc.put_keycode(0x1c);
    assert_ne!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 0);
    // Turning them back on raises the line at once.
    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(KBD_MODE_KBD_INT);
    // The mode write itself was answered by nothing, the key is still there.
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0x1c);
    assert_eq!(r.irq1(), 0);
}

#[test]
fn disabled_keyboard_interface_holds_data() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_KBD_DISABLE);
    r.kbc.put_keycode(0x1c);
    assert_eq!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 0);
    r.cmd(KBD_CCMD_KBD_ENABLE);
    assert_ne!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0x1c);

    // Writing to the keyboard enables the interface again.
    r.cmd(KBD_CCMD_KBD_DISABLE);
    r.outb(KBD_CMD_ECHO);
    assert_eq!(r.kbc.regs().mode & KBD_MODE_DISABLE_KBD, 0);
    assert_eq!(r.drain(), [KBD_CMD_ECHO]);
}

#[test]
fn controller_bytes_come_first_then_keyboard_and_mouse_alternate() {
    let r = Rig::new();
    r.enable_mouse();
    r.cmd(KBD_CCMD_WRITE_MODE);
    // Hold everything back while the queues fill.
    r.outb(0x33);
    r.kbc.put_keycode(0x11);
    r.kbc.put_keycode(0x12);
    r.kbc.mouse_event(1, 0, &[]);
    r.cmd(KBD_CCMD_SELF_TEST);
    // The controller reply goes out despite the disabled interfaces.
    assert_eq!(r.inb(), 0x55);
    assert_eq!(r.status() & KBD_STAT_OBF, 0);

    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(0x03);
    // The keyboard has priority, and after each keyboard byte the mouse gets a turn.
    assert_eq!(r.status() & KBD_STAT_MOUSE_OBF, 0);
    assert_eq!(r.inb(), 0x11);
    assert_ne!(r.status() & KBD_STAT_MOUSE_OBF, 0);
    assert_eq!(r.irq12(), 1);
    assert_eq!(r.inb(), 0x08);
    assert_eq!(r.status() & KBD_STAT_MOUSE_OBF, 0);
    assert_eq!(r.irq12(), 0);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0x12);
    assert_eq!(r.drain(), [0x01, 0x00]);
}

#[test]
fn write_obuf_commands() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_WRITE_OBUF);
    r.outb(0xab);
    assert_eq!(r.status() & (KBD_STAT_OBF | KBD_STAT_MOUSE_OBF), KBD_STAT_OBF);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0xab);

    r.cmd(KBD_CCMD_WRITE_AUX_OBUF);
    r.outb(0xcd);
    assert_eq!(r.status() & (KBD_STAT_OBF | KBD_STAT_MOUSE_OBF), 0x21);
    assert_eq!(r.irq12(), 1);
    assert_eq!(r.irq1(), 0);
    assert_eq!(r.inb(), 0xcd);
    assert_eq!(r.irq12(), 0);
}

#[test]
fn mouse_enable_and_packets() {
    let r = Rig::new();
    // Disabled mouse: movement is not recorded.
    r.kbc.mouse_event(5, 5, &[]);
    assert!(r.drain().is_empty());

    r.enable_mouse();
    assert_ne!(r.kbc.mouse().status() & MOUSE_STATUS_ENABLED, 0);

    r.kbc.mouse_event(5, -3, &[(InputButton::Left, true)]);
    assert_ne!(r.status() & KBD_STAT_MOUSE_OBF, 0);
    assert_eq!(r.irq12(), 1);
    assert_eq!(r.drain(), [0x09, 5, 3]);
    assert_eq!(r.irq12(), 0);

    // Negative X, positive screen Y is negative PS/2 Y, button released.
    r.kbc.mouse_event(-2, 4, &[(InputButton::Left, false), (InputButton::Right, true)]);
    assert_eq!(r.drain(), [0x08 | 0x10 | 0x20 | 0x02, 0xfe, 0xfc]);

    // Big moves are split into several packets.
    r.kbc.mouse_event(200, 0, &[(InputButton::Right, false)]);
    assert_eq!(r.drain(), [0x08, 127, 0, 0x08, 73, 0]);

    // The status query.
    r.aux(AUX_SET_SCALE21);
    r.aux(AUX_GET_SCALE);
    assert_eq!(r.drain(), [AUX_ACK, AUX_ACK, 0x30, 0, 0]);
    r.aux(AUX_SET_RES);
    r.aux(3);
    r.aux(AUX_SET_SAMPLE);
    r.aux(40);
    r.aux(AUX_SET_SCALE11);
    r.aux(AUX_GET_SCALE);
    assert_eq!(r.drain(), [AUX_ACK, AUX_ACK, AUX_ACK, AUX_ACK, AUX_ACK, AUX_ACK, 0x20, 3, 40]);

    r.aux(AUX_DISABLE_DEV);
    r.aux(AUX_SET_DEFAULT);
    r.aux(AUX_GET_SCALE);
    assert_eq!(r.drain(), [AUX_ACK, AUX_ACK, AUX_ACK, 0, 2, 100]);

    r.aux(AUX_RESET);
    assert_eq!(r.drain(), [AUX_ACK, 0xaa, 0x00]);
    // Unknown mouse commands are ignored.
    r.aux(0x42);
    assert!(r.drain().is_empty());
}

#[test]
fn mouse_remote_mode() {
    let r = Rig::new();
    r.enable_mouse();
    r.aux(AUX_SET_REMOTE);
    assert_eq!(r.drain(), [AUX_ACK]);
    r.kbc.mouse_event(3, 0, &[]);
    assert!(r.drain().is_empty());
    r.aux(AUX_POLL);
    assert_eq!(r.drain(), [AUX_ACK, 0x08, 3, 0]);
    // The movement was used up.
    r.aux(AUX_POLL);
    assert_eq!(r.drain(), [AUX_ACK, 0x08, 0, 0]);
    r.aux(AUX_SET_STREAM);
    assert_eq!(r.drain(), [AUX_ACK]);
    r.kbc.mouse_event(1, 0, &[]);
    assert_eq!(r.drain(), [0x08, 1, 0]);
}

#[test]
fn mouse_wrap_mode() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_MOUSE_ENABLE);
    r.aux(AUX_SET_WRAP);
    assert_eq!(r.drain(), [AUX_ACK]);
    assert!(r.kbc.mouse().wrap());
    r.aux(0x12);
    r.aux(AUX_GET_TYPE);
    assert_eq!(r.drain(), [0x12, AUX_GET_TYPE]);
    r.aux(AUX_RESET_WRAP);
    assert_eq!(r.drain(), [AUX_ACK]);
    assert!(!r.kbc.mouse().wrap());
    r.aux(AUX_GET_TYPE);
    assert_eq!(r.drain(), [AUX_ACK, 0]);

    // AUX_RESET gets through in wrap mode, and leaves wrap mode on as QEMU does.
    r.aux(AUX_SET_WRAP);
    r.aux(AUX_RESET);
    assert_eq!(r.drain(), [AUX_ACK, 0xaa, 0]);
    assert!(r.kbc.mouse().wrap());
}

fn knock(r: &Rig, rates: &[u8]) {
    for &rate in rates {
        r.aux(AUX_SET_SAMPLE);
        r.aux(rate);
    }
    r.aux(AUX_GET_TYPE);
}

#[test]
fn intellimouse_detection() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_MOUSE_ENABLE);
    knock(&r, &[200, 100, 80]);
    let mut v = r.drain();
    assert_eq!(v.split_off(6), [AUX_ACK, 3]);
    assert_eq!(r.kbc.mouse().mouse_type(), 3);
    assert_eq!(r.kbc.mouse().sample_rate(), 80);

    r.aux(AUX_ENABLE_DEV);
    r.drain();
    r.kbc.mouse_event(0, 0, &[(InputButton::WheelDown, true), (InputButton::WheelDown, false)]);
    assert_eq!(r.drain(), [0x08, 0, 0, 1]);
    r.kbc.mouse_event(0, 0, &[(InputButton::WheelUp, true), (InputButton::WheelUp, false)]);
    assert_eq!(r.drain(), [0x08, 0, 0, 0xff]);

    // IMEX.
    knock(&r, &[200, 200, 80]);
    let mut v = r.drain();
    assert_eq!(v.split_off(6), [AUX_ACK, 4]);
    r.kbc.mouse_event(0, 0, &[(InputButton::Side, true), (InputButton::WheelDown, true)]);
    assert_eq!(r.drain(), [0x08, 0, 0, 0x10 | 0x01]);
    r.kbc.mouse_event(0, 0, &[(InputButton::WheelLeft, true)]);
    assert_eq!(r.drain(), [0x08, 0, 0, 0x40 | 0x01]);

    // A broken sequence does not change the type, and a reset goes back to PS/2.
    r.aux(AUX_RESET);
    r.drain();
    knock(&r, &[200, 60, 80]);
    let mut v = r.drain();
    assert_eq!(v.split_off(6), [AUX_ACK, 0]);
}

#[test]
fn mouse_queue_full_drops_whole_packets() {
    let r = Rig::new();
    r.enable_mouse();
    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(0x23);
    for _ in 0..7 {
        r.kbc.mouse_event(1, 0, &[]);
    }
    // Five 3 byte packets fit in 16 bytes.
    assert_eq!(r.kbc.mouse().queue().count(), 15);
    r.cmd(KBD_CCMD_MOUSE_ENABLE);
    assert_eq!(r.drain().len(), 15);
}

#[test]
fn fake_event_nudges_the_mouse() {
    let r = Rig::new();
    r.enable_mouse();
    r.kbc.mouse_fake_event();
    assert_eq!(r.drain(), [0x08, 1, 0]);
}

#[test]
fn reset_command_pulses_reset_output() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_RESET);
    assert_eq!(r.reset.lock().unwrap().as_slice(), [1, 0]);
    // 0xF0 to 0xFF pulse bits 3 to 0; only bit 0 matters.
    r.cmd(0xF0);
    assert_eq!(r.reset.lock().unwrap().len(), 4);
    r.cmd(0xF1);
    r.cmd(0xFD);
    assert_eq!(r.reset.lock().unwrap().len(), 4);
    assert!(r.a20.lock().unwrap().is_empty());
}

#[test]
fn reset_handler_can_reset_the_controller() {
    let clock = Clock::manual(ClockType::Virtual);
    let kbc = I8042::new(clock, I8042Props::default()).unwrap();
    let w = Arc::downgrade(&kbc);
    kbc.reset_out().connect(IrqLine::from_fn(move |level| {
        if level != 0 {
            if let Some(k) = w.upgrade() {
                k.reset();
            }
        }
    }));
    kbc.write_command(KBD_CCMD_KBD_DISABLE);
    kbc.write_command(KBD_CCMD_RESET);
    assert_eq!(kbc.regs().mode, 0x03);
}

#[test]
fn a20_through_the_output_port() {
    let r = Rig::new();
    r.cmd(KBD_CCMD_WRITE_OUTPORT);
    r.outb(0xdd);
    assert_eq!(r.a20.lock().unwrap().as_slice(), [0]);
    assert_eq!(r.kbc.regs().outport, 0xdd);
    r.cmd(KBD_CCMD_READ_OUTPORT);
    assert_eq!(r.drain(), [0xdd]);

    r.cmd(KBD_CCMD_WRITE_OUTPORT);
    r.outb(0xdf);
    assert_eq!(r.a20.lock().unwrap().as_slice(), [0, 1]);
    assert!(r.reset.lock().unwrap().is_empty());

    // Bit 0 clear asks for a reset.
    r.cmd(KBD_CCMD_WRITE_OUTPORT);
    r.outb(0xde);
    assert_eq!(r.a20.lock().unwrap().as_slice(), [0, 1, 1]);
    assert_eq!(r.reset.lock().unwrap().as_slice(), [1, 0]);

    // The HP Vectra commands.
    r.cmd(KBD_CCMD_DISABLE_A20);
    assert_eq!(r.kbc.regs().outport & KBD_OUT_A20, 0);
    assert_eq!(last(&r.a20), 0);
    r.cmd(KBD_CCMD_ENABLE_A20);
    assert_ne!(r.kbc.regs().outport & KBD_OUT_A20, 0);
    assert_eq!(last(&r.a20), 1);
}

#[test]
fn device_reset_clears_the_buffer() {
    let r = Rig::new();
    r.kbc.put_keycode(0x1c);
    r.cmd(KBD_CCMD_WRITE_MODE);
    r.outb(0x47);
    assert_eq!(r.irq1(), 1);
    r.kbc.reset();
    assert_eq!(r.irq1(), 0);
    assert_eq!(r.status(), KBD_STAT_CMD | KBD_STAT_UNLOCKED);
    assert!(r.kbc.kbd().queue_empty());
    assert!(!r.kbc.kbd().translate());
    assert_eq!(r.kbc.regs().mode, 0x03);
}

#[test]
fn legacy_state_queues_replies_in_ps2_queues() {
    let r = Rig::with_props(I8042Props { extended_state: false, ..I8042Props::default() });
    r.cmd(KBD_CCMD_SELF_TEST);
    assert_eq!(r.kbc.kbd().queue().count(), 1);
    assert_eq!(r.drain(), [0x55]);
    r.cmd(KBD_CCMD_WRITE_AUX_OBUF);
    r.outb(0x77);
    assert_eq!(r.kbc.mouse().queue().count(), 1);
    assert_ne!(r.status() & KBD_STAT_MOUSE_OBF, 0);
    assert_eq!(r.inb(), 0x77);

    // Without extended state, a disabled interface does not hold data back.
    r.cmd(KBD_CCMD_KBD_DISABLE);
    r.kbc.put_keycode(0x1c);
    assert_ne!(r.status() & KBD_STAT_OBF, 0);
    // But IRQ1 is masked by the disable bit.
    assert_eq!(r.irq1(), 0);
}

#[test]
fn keyboard_throttle() {
    let r = Rig::with_props(I8042Props { kbd_throttle: true, ..I8042Props::default() });
    r.kbc.put_keycode(0x1c);
    r.kbc.put_keycode(0x32);
    assert_eq!(r.inb(), 0x1c);
    // The second byte waits for the timer.
    assert_eq!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 0);
    r.clock.advance_to(999_000);
    assert_eq!(r.status() & KBD_STAT_OBF, 0);
    r.clock.advance_to(1_000_000);
    assert_ne!(r.status() & KBD_STAT_OBF, 0);
    assert_eq!(r.irq1(), 1);
    assert_eq!(r.inb(), 0x32);
}

#[test]
fn port_regions() {
    let r = Rig::new();
    let data = r.kbc.data_io();
    let cmd = r.kbc.cmd_io();
    let cx = AccessCtx::default();
    cmd.write(&cx, 0, AccessSize::B1, u64::from(KBD_CCMD_SELF_TEST)).unwrap();
    assert_eq!(cmd.read(&cx, 0, AccessSize::B1).unwrap() & 1, 1);
    assert_eq!(data.read(&cx, 0, AccessSize::B1).unwrap(), 0x55);
    data.write(&cx, 0, AccessSize::B1, u64::from(KBD_CMD_ECHO)).unwrap();
    assert_eq!(data.read(&cx, 0, AccessSize::B1).unwrap(), u64::from(KBD_CMD_ECHO));
    assert_eq!(I8042_DATA_PORT, 0x60);
    assert_eq!(I8042_CMD_PORT, 0x64);
}

#[test]
fn input_layer_handlers_and_leds() {
    use ruvm_ui::input::{InputState, QEMU_CAPS_LOCK_LED, QEMU_NUM_LOCK_LED};

    let r = Rig::new();
    let input = InputState::new();
    r.kbc.register_input(&input);
    r.kbc.register_input(&input);
    let mice = input.query_mice();
    assert_eq!(mice.len(), 1);
    assert_eq!(
        (mice[0].name.as_str(), mice[0].current, mice[0].absolute),
        ("QEMU PS/2 Mouse", true, false)
    );

    // A key through the input layer comes out in set 2, untranslated in this mode.
    input.send_key_linux(None, 30, true);
    assert_eq!(r.drain(), [0x1c]);

    r.enable_mouse();
    input.queue_btn(None, InputButton::Left, true);
    input.queue_rel(None, InputAxis::X, 5);
    input.queue_rel(None, InputAxis::Y, -3);
    input.event_sync();
    assert_eq!(r.drain(), [0x09, 5, 3]);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let (i2, s2) = (Arc::clone(&input), Arc::clone(&seen));
    input.add_led_notifier(move || s2.lock().unwrap().push(i2.get_leds_mask(None)));
    assert_eq!(r.kbd_send(&[KBD_CMD_SET_LEDS, 0x06]), [KBD_REPLY_ACK, KBD_REPLY_ACK]);
    assert_eq!(input.get_leds_mask(None), QEMU_NUM_LOCK_LED | QEMU_CAPS_LOCK_LED);
    r.outb(KBD_CMD_RESET);
    r.drain();
    assert_eq!(*seen.lock().unwrap(), [6, 0]);
}
