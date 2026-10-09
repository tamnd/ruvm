// SPDX-License-Identifier: GPL-2.0-or-later

//! The i8042 keyboard controller, hw/input/pckbd.c, as the ISA `i8042` device.
//!
//! [`I8042`] is `ISAKBDState` with its `KBDState`, and owns the [`Ps2Kbd`] and [`Ps2Mouse`]
//! behind it. The data port (0x60) and the command and status port (0x64) are two one byte
//! regions, [`I8042::data_io`] and [`I8042::cmd_io`], or the port handlers can be called
//! directly.
//!
//! Outputs are [`IrqPin`]s the board wires up: the keyboard IRQ (ISA IRQ 1), the mouse IRQ
//! (ISA IRQ 12), the `a20` gate line, and a reset request line that the controller pulses
//! where QEMU calls `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`. The A20 and reset
//! lines are driven after the device lock is dropped, so their handlers may call back into
//! the device, for example to reset it.
//!
//! [`I8042::register_input`] registers the keyboard and the mouse with the input layer, as
//! "QEMU PS/2 Keyboard" and "QEMU PS/2 Mouse", the way `ps2_kbd_realize()` and
//! `ps2_mouse_realize()` do. LED changes go to the input layer after the device lock is
//! dropped too.
//!
//! [`I8042::vmstate_save`] and [`I8042::vmstate_load`] move what the `pckbd` VMState carries,
//! as an [`I8042VmState`]; the `kbd_` and `mouse_` variants do the same for the PS/2 devices.
//!
//! Not ported: trace points, QOM registration, the ACPI description, the `LOG_GUEST_ERROR`
//! message for unknown commands, and the `i8042-mmio` variant.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_base::{Result, bail, warn_report};
use ruvm_hw_core::{Clock, IrqPin, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};
use ruvm_ui::console::QemuConsole;
use ruvm_ui::input::{
    HandlerId, INPUT_EVENT_MASK_BTN, INPUT_EVENT_MASK_KEY, INPUT_EVENT_MASK_REL, InputHandler,
    InputState, QemuInputEvent,
};

use crate::ps2::{InputAxis, InputButton, Ps2Kbd, Ps2KbdVmState, Ps2Mouse, Ps2MouseVmState};

// Controller commands, written to port 0x64.

/// Read the mode byte.
pub const KBD_CCMD_READ_MODE: u8 = 0x20;
/// Write the mode byte.
pub const KBD_CCMD_WRITE_MODE: u8 = 0x60;
/// Get the controller firmware version.
pub const KBD_CCMD_GET_VERSION: u8 = 0xA1;
/// Disable the mouse interface.
pub const KBD_CCMD_MOUSE_DISABLE: u8 = 0xA7;
/// Enable the mouse interface.
pub const KBD_CCMD_MOUSE_ENABLE: u8 = 0xA8;
/// Mouse interface test.
pub const KBD_CCMD_TEST_MOUSE: u8 = 0xA9;
/// Controller self test.
pub const KBD_CCMD_SELF_TEST: u8 = 0xAA;
/// Keyboard interface test.
pub const KBD_CCMD_KBD_TEST: u8 = 0xAB;
/// Keyboard interface disable.
pub const KBD_CCMD_KBD_DISABLE: u8 = 0xAD;
/// Keyboard interface enable.
pub const KBD_CCMD_KBD_ENABLE: u8 = 0xAE;
/// Read the input port.
pub const KBD_CCMD_READ_INPORT: u8 = 0xC0;
/// Read the output port.
pub const KBD_CCMD_READ_OUTPORT: u8 = 0xD0;
/// Write the output port.
pub const KBD_CCMD_WRITE_OUTPORT: u8 = 0xD1;
/// Write the next byte to the output buffer as if the keyboard sent it.
pub const KBD_CCMD_WRITE_OBUF: u8 = 0xD2;
/// Write the next byte to the output buffer as if the mouse sent it.
pub const KBD_CCMD_WRITE_AUX_OBUF: u8 = 0xD3;
/// Send the next byte to the mouse.
pub const KBD_CCMD_WRITE_MOUSE: u8 = 0xD4;
/// HP vectra only.
pub const KBD_CCMD_DISABLE_A20: u8 = 0xDD;
/// HP vectra only.
pub const KBD_CCMD_ENABLE_A20: u8 = 0xDF;
/// Pulse bits 3 to 0 of the output port P2.
pub const KBD_CCMD_PULSE_BITS_3_0: u8 = 0xF0;
/// Pulse bit 0 of the output port P2, the system reset.
pub const KBD_CCMD_RESET: u8 = 0xFE;
/// Pulse no bits of the output port P2.
pub const KBD_CCMD_NO_OP: u8 = 0xFF;

// Status register bits, read from port 0x64.

/// Keyboard output buffer full.
pub const KBD_STAT_OBF: u8 = 0x01;
/// Keyboard input buffer full.
pub const KBD_STAT_IBF: u8 = 0x02;
/// Self test successful.
pub const KBD_STAT_SELFTEST: u8 = 0x04;
/// Last write was a command write (0 means data).
pub const KBD_STAT_CMD: u8 = 0x08;
/// Zero if the keyboard is locked.
pub const KBD_STAT_UNLOCKED: u8 = 0x10;
/// Mouse output buffer full.
pub const KBD_STAT_MOUSE_OBF: u8 = 0x20;
/// General receive or transmit timeout.
pub const KBD_STAT_GTO: u8 = 0x40;
/// Parity error.
pub const KBD_STAT_PERR: u8 = 0x80;

// Controller mode register bits.

/// Keyboard data generates IRQ1.
pub const KBD_MODE_KBD_INT: u8 = 0x01;
/// Mouse data generates IRQ12.
pub const KBD_MODE_MOUSE_INT: u8 = 0x02;
/// The system flag.
pub const KBD_MODE_SYS: u8 = 0x04;
/// The keylock does not affect the keyboard.
pub const KBD_MODE_NO_KEYLOCK: u8 = 0x08;
/// Disable the keyboard interface.
pub const KBD_MODE_DISABLE_KBD: u8 = 0x10;
/// Disable the mouse interface.
pub const KBD_MODE_DISABLE_MOUSE: u8 = 0x20;
/// Scan code conversion to PC format.
pub const KBD_MODE_KCC: u8 = 0x40;
pub const KBD_MODE_RFU: u8 = 0x80;

// Output port bits.

/// 1 is normal mode, 0 is reset.
pub const KBD_OUT_RESET: u8 = 0x01;
/// The A20 gate, x86 only.
pub const KBD_OUT_A20: u8 = 0x02;
/// Keyboard output buffer full.
pub const KBD_OUT_OBF: u8 = 0x10;
/// Mouse output buffer full.
pub const KBD_OUT_MOUSE_OBF: u8 = 0x20;

/// OSes typically write 0xdd or 0xdf to turn the A20 line off and on. We make the default
/// value of the output port have bits 2, 3, 6 and 7 set, so that 0xdd and 0xdf writes do not
/// look like changes to the other bits.
pub const KBD_OUT_ONES: u8 = 0xcc;

const KBD_PENDING_CTRL_KBD: u8 = 0x04;
const KBD_PENDING_CTRL_AUX: u8 = 0x08;
const KBD_PENDING_KBD: u8 = KBD_MODE_DISABLE_KBD;
const KBD_PENDING_AUX: u8 = KBD_MODE_DISABLE_MOUSE;

// The `pending` bits a controller without extended state sends.
const KBD_PENDING_KBD_COMPAT: u8 = 0x01;
const KBD_PENDING_AUX_COMPAT: u8 = 0x02;

/// `KBD_MIGR_TIMER_PENDING` in `migration_flags`: the throttle timer was armed.
pub const KBD_MIGR_TIMER_PENDING: u32 = 0x1;

const KBD_OBSRC_KBD: u8 = 0x01;
const KBD_OBSRC_MOUSE: u8 = 0x02;
const KBD_OBSRC_CTRL: u8 = 0x04;

/// `TYPE_I8042`.
pub const TYPE_I8042: &str = "i8042";
/// `I8042_A20_LINE`, the name of the A20 GPIO output.
pub const I8042_A20_LINE: &str = "a20";
/// `ISA_NUM_IRQS`.
const ISA_NUM_IRQS: u8 = 16;

/// The i8042 data port.
pub const I8042_DATA_PORT: u16 = 0x60;
/// The i8042 command and status port.
pub const I8042_CMD_PORT: u16 = 0x64;

/// The properties of the `i8042` device.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct I8042Props {
    /// `extended-state`: keep controller replies in the controller rather than pushing them
    /// into the PS/2 queues. On by default, off only for old machine types.
    pub extended_state: bool,
    /// `kbd-throttle`: after the guest reads a keyboard byte, hold the next one back for 1ms.
    /// Needs `extended_state`.
    pub kbd_throttle: bool,
    /// `kbd-irq`.
    pub kbd_irq: u8,
    /// `mouse-irq`.
    pub mouse_irq: u8,
}

impl Default for I8042Props {
    fn default() -> Self {
        I8042Props { extended_state: true, kbd_throttle: false, kbd_irq: 1, mouse_irq: 12 }
    }
}

/// The controller part of `KBDState`.
#[derive(Clone, Debug)]
struct KbdCtrl {
    /// When not zero, the next write to port 0x60 is the argument of this command.
    write_cmd: u8,
    status: u8,
    mode: u8,
    outport: u8,
    obsrc: u8,
    extended_state: bool,
    /// Bitmask of devices with data available.
    pending: u8,
    obdata: u8,
    cbdata: u8,
}

/// `KBDState`: the controller plus the two PS/2 devices behind it, all under one lock.
#[derive(Clone, Debug)]
struct KbdState {
    ctrl: KbdCtrl,
    kbd: Ps2Kbd,
    mouse: Ps2Mouse,
}

/// Output line changes that are made after the lock is released.
#[derive(Default)]
struct Deferred {
    a20: Option<bool>,
    reset: bool,
    /// A new keyboard LED state for the input layer.
    leds: Option<u8>,
}

/// A snapshot of the controller registers, for tests and the monitor.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct I8042Regs {
    pub status: u8,
    pub mode: u8,
    pub outport: u8,
    pub write_cmd: u8,
}

/// The fields of `vmstate_kbd` (version 3) and its subsections, named as in QEMU, plus the
/// flags its hooks keep in `KBDState`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct I8042VmState {
    pub write_cmd: u8,
    pub status: u8,
    pub mode: u8,
    /// `pending`, in the compat encoding without extended state, from `kbd_pre_save()`.
    pub pending_tmp: u8,
    /// `pckbd_outport`.
    pub outport: u8,
    /// `pckbd/extended_state`.
    pub migration_flags: u32,
    pub obsrc: u32,
    pub obdata: u8,
    pub cbdata: u8,
    /// The `extended-state` property, not on the wire.
    pub extended_state: bool,
    /// Set by the `pckbd_outport` post_load, cleared by `kbd_pre_load()`.
    pub outport_present: bool,
    /// Set by the `pckbd/extended_state` post_load, cleared by `kbd_pre_load()`.
    pub extended_state_loaded: bool,
}

impl I8042VmState {
    /// `kbd_outport_needed()`.
    pub fn outport_needed(&self) -> bool {
        self.outport != outport_default(self.status)
    }

    /// `kbd_extended_state_needed()`.
    pub fn extended_state_needed(&self) -> bool {
        self.extended_state
    }
}

/// `kbd_outport_default()`.
fn outport_default(status: u8) -> u8 {
    let mut v = KBD_OUT_RESET | KBD_OUT_A20 | KBD_OUT_ONES;
    if (status & KBD_STAT_OBF) != 0 {
        v |= KBD_OUT_OBF;
    }
    if (status & KBD_STAT_MOUSE_OBF) != 0 {
        v |= KBD_OUT_MOUSE_OBF;
    }
    v
}

/// `ISAKBDState`, the `i8042` device.
pub struct I8042 {
    state: Mutex<KbdState>,
    props: I8042Props,
    kbd_irq: IrqPin,
    mouse_irq: IrqPin,
    a20_out: IrqPin,
    reset_out: IrqPin,
    clock: Arc<Clock>,
    throttle_timer: Option<Timer>,
    /// The input layer and the keyboard's handler in it, once registered.
    input: OnceLock<(Arc<InputState>, HandlerId)>,
}

impl fmt::Debug for I8042 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("I8042")
            .field("state", &self.lock().ctrl)
            .field("props", &self.props)
            .field("kbd_irq", &self.kbd_irq)
            .field("mouse_irq", &self.mouse_irq)
            .field("a20_out", &self.a20_out)
            .field("reset_out", &self.reset_out)
            .finish()
    }
}

impl I8042 {
    /// `i8042_initfn()`, `i8042_realizefn()` and a reset. `clock` is the virtual clock, used
    /// by the keyboard throttle.
    pub fn new(clock: Arc<Clock>, props: I8042Props) -> Result<Arc<I8042>> {
        if props.kbd_irq >= ISA_NUM_IRQS {
            bail!("Maximum value for \"kbd-irq\" is: {}", ISA_NUM_IRQS - 1);
        }
        if props.mouse_irq >= ISA_NUM_IRQS {
            bail!("Maximum value for \"mouse-irq\" is: {}", ISA_NUM_IRQS - 1);
        }
        let mut props = props;
        if props.kbd_throttle && !props.extended_state {
            warn_report(&format!(
                "{TYPE_I8042}: can't enable kbd-throttle without extended-state, disabling \
                 kbd-throttle"
            ));
            props.kbd_throttle = false;
        }

        let s = Arc::new_cyclic(|weak: &Weak<I8042>| {
            let throttle_timer = props.kbd_throttle.then(|| {
                let w = weak.clone();
                clock.new_timer(move || {
                    if let Some(s) = w.upgrade() {
                        s.throttle_timeout();
                    }
                })
            });
            I8042 {
                state: Mutex::new(KbdState {
                    ctrl: KbdCtrl {
                        write_cmd: 0,
                        status: 0,
                        mode: 0,
                        outport: 0,
                        obsrc: 0,
                        extended_state: props.extended_state,
                        pending: 0,
                        obdata: 0,
                        cbdata: 0,
                    },
                    kbd: Ps2Kbd::new(),
                    mouse: Ps2Mouse::new(),
                }),
                props,
                kbd_irq: IrqPin::new(),
                mouse_irq: IrqPin::new(),
                a20_out: IrqPin::new(),
                reset_out: IrqPin::new(),
                clock: clock.clone(),
                throttle_timer,
                input: OnceLock::new(),
            }
        });
        s.reset();
        Ok(s)
    }

    fn lock(&self) -> MutexGuard<'_, KbdState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// The device properties, after realize adjusted them.
    pub fn props(&self) -> &I8042Props {
        &self.props
    }

    /// Output `I8042_KBD_IRQ`, to ISA IRQ `kbd_irq`.
    pub fn kbd_irq(&self) -> &IrqPin {
        &self.kbd_irq
    }

    /// Output `I8042_MOUSE_IRQ`, to ISA IRQ `mouse_irq`.
    pub fn mouse_irq(&self) -> &IrqPin {
        &self.mouse_irq
    }

    /// The `a20` output: bit 1 of the output port.
    pub fn a20_out(&self) -> &IrqPin {
        &self.a20_out
    }

    /// The system reset request, pulsed high then low when the guest resets the machine
    /// through the controller.
    pub fn reset_out(&self) -> &IrqPin {
        &self.reset_out
    }

    /// The controller registers.
    pub fn regs(&self) -> I8042Regs {
        let s = self.lock();
        I8042Regs {
            status: s.ctrl.status,
            mode: s.ctrl.mode,
            outport: s.ctrl.outport,
            write_cmd: s.ctrl.write_cmd,
        }
    }

    /// A copy of the keyboard state.
    pub fn kbd(&self) -> Ps2Kbd {
        self.lock().kbd.clone()
    }

    /// A copy of the mouse state.
    pub fn mouse(&self) -> Ps2Mouse {
        self.lock().mouse.clone()
    }

    /// The state `vmstate_kbd` sends, with `kbd_pre_save()` and the `pckbd/extended_state`
    /// pre_save applied.
    pub fn vmstate_save(&self) -> I8042VmState {
        let st = self.lock();
        let s = &st.ctrl;
        let pending_tmp = if s.extended_state {
            s.pending
        } else {
            let mut p = 0;
            if (s.pending & KBD_PENDING_KBD) != 0 {
                p |= KBD_PENDING_KBD_COMPAT;
            }
            if (s.pending & KBD_PENDING_AUX) != 0 {
                p |= KBD_PENDING_AUX_COMPAT;
            }
            p
        };
        let timer_pending = self.throttle_timer.as_ref().is_some_and(Timer::pending);
        I8042VmState {
            write_cmd: s.write_cmd,
            status: s.status,
            mode: s.mode,
            pending_tmp,
            outport: s.outport,
            migration_flags: if timer_pending { KBD_MIGR_TIMER_PENDING } else { 0 },
            obsrc: u32::from(s.obsrc),
            obdata: s.obdata,
            cbdata: s.cbdata,
            extended_state: s.extended_state,
            outport_present: false,
            extended_state_loaded: false,
        }
    }

    /// Loads what `vmstate_kbd` carried: the `pckbd/extended_state` post_load, then
    /// `kbd_post_load()`.
    ///
    /// As in QEMU, a set `KBD_MIGR_TIMER_PENDING` runs `kbd_throttle_timeout()` before
    /// `pending` is loaded, and the throttle timer is not armed again. The IRQ lines are not
    /// driven otherwise; the interrupt controllers carry their levels.
    pub fn vmstate_load(&self, v: &I8042VmState) {
        let mut st = self.lock();
        let s = &mut st.ctrl;
        s.write_cmd = v.write_cmd;
        s.status = v.status;
        s.mode = v.mode;
        if v.outport_present {
            s.outport = v.outport;
        }
        if v.extended_state_loaded {
            s.obsrc = v.obsrc as u8;
            s.obdata = v.obdata;
            s.cbdata = v.cbdata;
            if (v.migration_flags & KBD_MIGR_TIMER_PENDING) != 0 && Self::pending(s) != 0 {
                self.update_irq(s);
            }
        }

        // kbd_post_load().
        if !v.outport_present {
            s.outport = outport_default(s.status);
        }
        s.pending = v.pending_tmp;
        if !v.extended_state_loaded {
            s.obsrc = if (s.status & KBD_STAT_OBF) != 0 {
                if (s.status & KBD_STAT_MOUSE_OBF) != 0 { KBD_OBSRC_MOUSE } else { KBD_OBSRC_KBD }
            } else {
                0
            };
            if (s.pending & KBD_PENDING_KBD_COMPAT) != 0 {
                s.pending |= KBD_PENDING_KBD;
            }
            if (s.pending & KBD_PENDING_AUX_COMPAT) != 0 {
                s.pending |= KBD_PENDING_AUX;
            }
        }
        // Clear all unused flags.
        s.pending &=
            KBD_PENDING_CTRL_KBD | KBD_PENDING_CTRL_AUX | KBD_PENDING_KBD | KBD_PENDING_AUX;
    }

    /// The keyboard state `vmstate_ps2_keyboard` sends.
    pub fn kbd_vmstate_save(&self) -> Ps2KbdVmState {
        self.lock().kbd.vmstate_save()
    }

    /// See [`Ps2Kbd::vmstate_load`].
    pub fn kbd_vmstate_load(&self, v: &Ps2KbdVmState) {
        let leds = {
            let mut st = self.lock();
            st.kbd.vmstate_load(v);
            st.kbd.take_leds_update()
        };
        self.fire(Deferred { leds, ..Deferred::default() });
    }

    /// The mouse state `vmstate_ps2_mouse` sends.
    pub fn mouse_vmstate_save(&self) -> Ps2MouseVmState {
        self.lock().mouse.vmstate_save()
    }

    /// See [`Ps2Mouse::vmstate_load`].
    pub fn mouse_vmstate_load(&self, v: &Ps2MouseVmState) {
        self.lock().mouse.vmstate_load(v);
    }

    /// `kbd_update_irq_lines()`.
    fn update_irq_lines(&self, s: &KbdCtrl) {
        let mut irq_kbd_level = false;
        let mut irq_mouse_level = false;

        if (s.status & KBD_STAT_OBF) != 0 {
            if (s.status & KBD_STAT_MOUSE_OBF) != 0 {
                if (s.mode & KBD_MODE_MOUSE_INT) != 0 {
                    irq_mouse_level = true;
                }
            } else if (s.mode & KBD_MODE_KBD_INT) != 0 && (s.mode & KBD_MODE_DISABLE_KBD) == 0 {
                irq_kbd_level = true;
            }
        }
        self.kbd_irq.set_bool(irq_kbd_level);
        self.mouse_irq.set_bool(irq_mouse_level);
    }

    /// `kbd_deassert_irq()`.
    fn deassert_irq(&self, s: &mut KbdCtrl) {
        s.status &= !(KBD_STAT_OBF | KBD_STAT_MOUSE_OBF);
        s.outport &= !(KBD_OUT_OBF | KBD_OUT_MOUSE_OBF);
        self.update_irq_lines(s);
    }

    /// `kbd_pending()`: with extended state, a device whose interface is disabled in the mode
    /// byte does not count.
    fn pending(s: &KbdCtrl) -> u8 {
        if s.extended_state {
            s.pending & (!s.mode | !(KBD_PENDING_KBD | KBD_PENDING_AUX))
        } else {
            s.pending
        }
    }

    /// `kbd_update_irq()`: fills the output buffer from the highest priority source. Replies
    /// of the controller come first, then the keyboard, then the mouse.
    fn update_irq(&self, s: &mut KbdCtrl) {
        let pending = Self::pending(s);

        s.status &= !(KBD_STAT_OBF | KBD_STAT_MOUSE_OBF);
        s.outport &= !(KBD_OUT_OBF | KBD_OUT_MOUSE_OBF);
        if pending != 0 {
            s.status |= KBD_STAT_OBF;
            s.outport |= KBD_OUT_OBF;
            if (pending & KBD_PENDING_CTRL_KBD) != 0 {
                s.obsrc = KBD_OBSRC_CTRL;
            } else if (pending & KBD_PENDING_CTRL_AUX) != 0 {
                s.status |= KBD_STAT_MOUSE_OBF;
                s.outport |= KBD_OUT_MOUSE_OBF;
                s.obsrc = KBD_OBSRC_CTRL;
            } else if (pending & KBD_PENDING_KBD) != 0 {
                s.obsrc = KBD_OBSRC_KBD;
            } else {
                s.status |= KBD_STAT_MOUSE_OBF;
                s.outport |= KBD_OUT_MOUSE_OBF;
                s.obsrc = KBD_OBSRC_MOUSE;
            }
        }
        self.update_irq_lines(s);
    }

    /// `kbd_safe_update_irq()`: like [`I8042::update_irq`], but leaves a full output buffer
    /// alone. The next read of the data port refills it.
    fn safe_update_irq(&self, s: &mut KbdCtrl) {
        // With KBD_STAT_OBF set, a read of the data port will eventually update the IRQ.
        if (s.status & KBD_STAT_OBF) != 0 {
            return;
        }
        // The throttle timer is pending and will update the IRQ.
        if self.throttle_timer.as_ref().is_some_and(Timer::pending) {
            return;
        }
        if Self::pending(s) != 0 {
            self.update_irq(s);
        }
    }

    /// `kbd_update_kbd_irq()`, the input the keyboard's IRQ output is wired to.
    fn update_kbd_irq(&self, s: &mut KbdCtrl, level: bool) {
        if level {
            s.pending |= KBD_PENDING_KBD;
        } else {
            s.pending &= !KBD_PENDING_KBD;
        }
        self.safe_update_irq(s);
    }

    /// `kbd_update_aux_irq()`, the input the mouse's IRQ output is wired to.
    fn update_aux_irq(&self, s: &mut KbdCtrl, level: bool) {
        if level {
            s.pending |= KBD_PENDING_AUX;
        } else {
            s.pending &= !KBD_PENDING_AUX;
        }
        self.safe_update_irq(s);
    }

    /// `kbd_throttle_timeout()`.
    fn throttle_timeout(&self) {
        let mut st = self.lock();
        let s = &mut st.ctrl;
        if Self::pending(s) != 0 {
            self.update_irq(s);
        }
    }

    /// `kbd_read_status()`: port 0x64 reads.
    pub fn read_status(&self) -> u8 {
        self.lock().ctrl.status
    }

    /// `kbd_queue()`: a byte from the controller itself for the output buffer. With `aux`
    /// it looks like it came from the mouse.
    fn queue(&self, st: &mut KbdState, b: u8, aux: bool) {
        let KbdState { ctrl, kbd, mouse } = st;
        if ctrl.extended_state {
            ctrl.cbdata = b;
            ctrl.pending &= !KBD_PENDING_CTRL_KBD & !KBD_PENDING_CTRL_AUX;
            ctrl.pending |= if aux { KBD_PENDING_CTRL_AUX } else { KBD_PENDING_CTRL_KBD };
            self.safe_update_irq(ctrl);
        } else if aux {
            mouse.queue_byte(b, &mut |l| self.update_aux_irq(ctrl, l));
        } else {
            kbd.queue_byte(b, &mut |l| self.update_kbd_irq(ctrl, l));
        }
    }

    /// `kbd_dequeue()`.
    fn dequeue(&self, s: &mut KbdCtrl) -> u8 {
        let b = s.cbdata;
        s.pending &= !KBD_PENDING_CTRL_KBD & !KBD_PENDING_CTRL_AUX;
        if Self::pending(s) != 0 {
            self.update_irq(s);
        }
        b
    }

    /// `outport_write()`.
    fn outport_write(s: &mut KbdCtrl, val: u8, out: &mut Deferred) {
        s.outport = val;
        out.a20 = Some(((val >> 1) & 1) != 0);
        if (val & 1) == 0 {
            out.reset = true;
        }
    }

    /// Drives the output lines [`Deferred`] collected, with the lock released.
    fn fire(&self, out: Deferred) {
        if let Some(level) = out.a20 {
            self.a20_out.set_bool(level);
        }
        if out.reset {
            self.reset_out.pulse();
        }
        if let (Some(leds), Some((input, id))) = (out.leds, self.input.get()) {
            input.set_leds_mask(*id, u32::from(leds));
        }
    }

    /// `kbd_write_command()`: port 0x64 writes.
    pub fn write_command(&self, val: u8) {
        let mut out = Deferred::default();
        {
            let mut st = self.lock();
            self.write_command_locked(&mut st, val, &mut out);
        }
        self.fire(out);
    }

    fn write_command_locked(&self, st: &mut KbdState, val: u8, out: &mut Deferred) {
        // Bits 3 to 0 of the output port P2 can be pulsed low for about 6 microseconds. Bits 3
        // to 0 of the command say which: 0 pulses the bit, 1 leaves it alone. The only useful
        // version pulses bit 0, which resets the CPU.
        let mut val = val;
        if (val & KBD_CCMD_PULSE_BITS_3_0) == KBD_CCMD_PULSE_BITS_3_0 {
            val = if (val & 1) == 0 { KBD_CCMD_RESET } else { KBD_CCMD_NO_OP };
        }

        match val {
            KBD_CCMD_READ_MODE => {
                let mode = st.ctrl.mode;
                self.queue(st, mode, false);
            }
            KBD_CCMD_WRITE_MODE
            | KBD_CCMD_WRITE_OBUF
            | KBD_CCMD_WRITE_AUX_OBUF
            | KBD_CCMD_WRITE_MOUSE
            | KBD_CCMD_WRITE_OUTPORT => {
                st.ctrl.write_cmd = val;
            }
            KBD_CCMD_MOUSE_DISABLE => {
                st.ctrl.mode |= KBD_MODE_DISABLE_MOUSE;
            }
            KBD_CCMD_MOUSE_ENABLE => {
                st.ctrl.mode &= !KBD_MODE_DISABLE_MOUSE;
                self.safe_update_irq(&mut st.ctrl);
            }
            KBD_CCMD_TEST_MOUSE => self.queue(st, 0x00, false),
            KBD_CCMD_SELF_TEST => {
                st.ctrl.status |= KBD_STAT_SELFTEST;
                self.queue(st, 0x55, false);
            }
            KBD_CCMD_KBD_TEST => self.queue(st, 0x00, false),
            KBD_CCMD_KBD_DISABLE => {
                st.ctrl.mode |= KBD_MODE_DISABLE_KBD;
            }
            KBD_CCMD_KBD_ENABLE => {
                st.ctrl.mode &= !KBD_MODE_DISABLE_KBD;
                self.safe_update_irq(&mut st.ctrl);
            }
            KBD_CCMD_READ_INPORT => self.queue(st, 0x80, false),
            KBD_CCMD_READ_OUTPORT => {
                let outport = st.ctrl.outport;
                self.queue(st, outport, false);
            }
            KBD_CCMD_ENABLE_A20 => {
                out.a20 = Some(true);
                st.ctrl.outport |= KBD_OUT_A20;
            }
            KBD_CCMD_DISABLE_A20 => {
                out.a20 = Some(false);
                st.ctrl.outport &= !KBD_OUT_A20;
            }
            KBD_CCMD_RESET => out.reset = true,
            KBD_CCMD_NO_OP => {}
            // QEMU logs "unsupported keyboard cmd" as a guest error.
            _ => {}
        }
    }

    /// `kbd_read_data()`: port 0x60 reads.
    pub fn read_data(&self) -> u8 {
        let mut st = self.lock();
        let KbdState { ctrl, kbd, mouse } = &mut *st;

        if (ctrl.status & KBD_STAT_OBF) != 0 {
            self.deassert_irq(ctrl);
            if (ctrl.obsrc & KBD_OBSRC_KBD) != 0 {
                if let Some(t) = &self.throttle_timer {
                    t.modify((self.clock.get_ns() / 1000 + 1000) * 1000);
                }
                ctrl.obdata = kbd.read_data(&mut |l| self.update_kbd_irq(ctrl, l));
            } else if (ctrl.obsrc & KBD_OBSRC_MOUSE) != 0 {
                ctrl.obdata = mouse.read_data(&mut |l| self.update_aux_irq(ctrl, l));
            } else if (ctrl.obsrc & KBD_OBSRC_CTRL) != 0 {
                ctrl.obdata = self.dequeue(ctrl);
            }
        }

        ctrl.obdata
    }

    /// `kbd_write_data()`: port 0x60 writes.
    pub fn write_data(&self, val: u8) {
        let mut out = Deferred::default();
        {
            let mut st = self.lock();
            let st = &mut *st;
            match st.ctrl.write_cmd {
                0 => {
                    let KbdState { ctrl, kbd, .. } = &mut *st;
                    kbd.write(val, &mut |l| self.update_kbd_irq(ctrl, l));
                    // Sending data to the keyboard enables PS/2 communication again.
                    ctrl.mode &= !KBD_MODE_DISABLE_KBD;
                    self.safe_update_irq(ctrl);
                }
                KBD_CCMD_WRITE_MODE => {
                    st.ctrl.mode = val;
                    st.kbd.set_translation((st.ctrl.mode & KBD_MODE_KCC) != 0);
                    // A write to the interrupt enable bits of the mode byte updates the IRQ
                    // lines directly.
                    self.update_irq_lines(&st.ctrl);
                    // A write to the interface disable bits may raise an IRQ if there is data
                    // waiting in the PS/2 queues.
                    self.safe_update_irq(&mut st.ctrl);
                }
                KBD_CCMD_WRITE_OBUF => self.queue(st, val, false),
                KBD_CCMD_WRITE_AUX_OBUF => self.queue(st, val, true),
                KBD_CCMD_WRITE_OUTPORT => Self::outport_write(&mut st.ctrl, val, &mut out),
                KBD_CCMD_WRITE_MOUSE => {
                    let KbdState { ctrl, mouse, .. } = &mut *st;
                    mouse.write(val, &mut |l| self.update_aux_irq(ctrl, l));
                    // Sending data to the mouse enables PS/2 communication again.
                    ctrl.mode &= !KBD_MODE_DISABLE_MOUSE;
                    self.safe_update_irq(ctrl);
                }
                _ => {}
            }
            st.ctrl.write_cmd = 0;
            out.leds = st.kbd.take_leds_update();
        }
        self.fire(out);
    }

    /// Device reset: the PS/2 devices' hold phase, `kbd_reset()`, then the PS/2 devices'
    /// exit phase, which lowers their IRQs.
    pub fn reset(&self) {
        let mut st = self.lock();
        let KbdState { ctrl, kbd, mouse } = &mut *st;
        // The hold phases, which do not touch the IRQ.
        kbd.reset(&mut |_| {});
        mouse.reset(&mut |_| {});

        ctrl.mode = KBD_MODE_KBD_INT | KBD_MODE_MOUSE_INT;
        ctrl.status = KBD_STAT_CMD | KBD_STAT_UNLOCKED;
        ctrl.outport = KBD_OUT_RESET | KBD_OUT_A20 | KBD_OUT_ONES;
        ctrl.pending = 0;
        self.deassert_irq(ctrl);
        if let Some(t) = &self.throttle_timer {
            t.del();
        }

        // The exit phases.
        self.update_kbd_irq(ctrl, false);
        self.update_aux_irq(ctrl, false);
    }

    /// The virtual clock the controller was made with.
    pub fn clock(&self) -> &Arc<Clock> {
        &self.clock
    }

    /// Registers the keyboard and the mouse with `input`, `ps2_kbd_realize()` and
    /// `ps2_mouse_realize()`. Only the first call does anything.
    pub fn register_input(self: &Arc<Self>, input: &Arc<InputState>) {
        if self.input.get().is_some() {
            return;
        }
        let kbd = input.register(Arc::new(Ps2KbdHandler(Arc::downgrade(self))));
        input.register(Arc::new(Ps2MouseHandler(Arc::downgrade(self))));
        let _ = self.input.set((Arc::clone(input), kbd));
    }

    /// A key went up or down, by Linux keycode. See [`Ps2Kbd::keyboard_event`].
    pub fn key_event(&self, key: u16, down: bool) {
        let mut st = self.lock();
        let KbdState { ctrl, kbd, .. } = &mut *st;
        kbd.keyboard_event(key, down, &mut |l| self.update_kbd_irq(ctrl, l));
    }

    /// Queues a raw scancode in the keyboard's current set, translated to set 1 when the
    /// mode byte asks for it. See [`Ps2Kbd::put_keycode`].
    pub fn put_keycode(&self, keycode: u8) {
        let mut st = self.lock();
        let KbdState { ctrl, kbd, .. } = &mut *st;
        kbd.put_keycode(keycode, &mut |l| self.update_kbd_irq(ctrl, l));
    }

    /// One batch of mouse input, the way the UI reports it: relative movement `dx` and
    /// `dy` (positive `dy` is down the screen), then button changes, then a sync.
    pub fn mouse_event(&self, dx: i32, dy: i32, buttons: &[(InputButton, bool)]) {
        let mut st = self.lock();
        let KbdState { ctrl, mouse, .. } = &mut *st;
        if dx != 0 {
            mouse.rel_event(InputAxis::X, dx);
        }
        if dy != 0 {
            mouse.rel_event(InputAxis::Y, dy);
        }
        for &(b, down) in buttons {
            mouse.button_event(b, down);
        }
        mouse.sync(&mut |l| self.update_aux_irq(ctrl, l));
    }

    /// A mouse button went up or down. Nothing is sent until [`I8042::mouse_sync`].
    pub fn mouse_button(&self, button: InputButton, down: bool) {
        self.lock().mouse.button_event(button, down);
    }

    /// Relative mouse movement. Nothing is sent until [`I8042::mouse_sync`].
    pub fn mouse_rel(&self, axis: InputAxis, value: i32) {
        self.lock().mouse.rel_event(axis, value);
    }

    /// `ps2_mouse_sync()`.
    pub fn mouse_sync(&self) {
        let mut st = self.lock();
        let KbdState { ctrl, mouse, .. } = &mut *st;
        mouse.sync(&mut |l| self.update_aux_irq(ctrl, l));
    }

    /// `i8042_isa_mouse_fake_event()`.
    pub fn mouse_fake_event(&self) {
        let mut st = self.lock();
        let KbdState { ctrl, mouse, .. } = &mut *st;
        mouse.fake_event(&mut |l| self.update_aux_irq(ctrl, l));
    }

    /// The data port as a region, `i8042_data_ops`, for port 0x60.
    pub fn data_io(self: &Arc<Self>) -> Arc<I8042Data> {
        Arc::new(I8042Data(self.clone()))
    }

    /// The command port as a region, `i8042_cmd_ops`, for port 0x64.
    pub fn cmd_io(self: &Arc<Self>) -> Arc<I8042Cmd> {
        Arc::new(I8042Cmd(self.clone()))
    }
}

/// `ps2_keyboard_handler`.
struct Ps2KbdHandler(Weak<I8042>);

impl InputHandler for Ps2KbdHandler {
    fn name(&self) -> &str {
        "QEMU PS/2 Keyboard"
    }

    fn mask(&self) -> u32 {
        INPUT_EVENT_MASK_KEY
    }

    fn event(&self, _src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        if let (Some(s), QemuInputEvent::Key { key, down }) = (self.0.upgrade(), evt) {
            // Linux keycodes stop below `KEY_CNT`, 0x300.
            s.key_event(*key as u16, *down);
        }
    }
}

/// `ps2_mouse_handler`.
struct Ps2MouseHandler(Weak<I8042>);

impl InputHandler for Ps2MouseHandler {
    fn name(&self) -> &str {
        "QEMU PS/2 Mouse"
    }

    fn mask(&self) -> u32 {
        INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_REL
    }

    fn event(&self, _src: Option<&QemuConsole>, evt: &QemuInputEvent) {
        let Some(s) = self.0.upgrade() else { return };
        match evt {
            QemuInputEvent::Btn(b) => s.mouse_button(b.button, b.down),
            // `mouse_dx` is an int in QEMU too.
            QemuInputEvent::Rel(m) => s.mouse_rel(m.axis, m.value as i32),
            _ => {}
        }
    }

    fn sync(&self) {
        if let Some(s) = self.0.upgrade() {
            s.mouse_sync();
        }
    }
}

/// `i8042_data_ops`: the one byte data port.
#[derive(Debug)]
pub struct I8042Data(pub Arc<I8042>);

impl MmioOps for I8042Data {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.read_data()))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.write_data(value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

/// `i8042_cmd_ops`: the one byte command and status port.
#[derive(Debug)]
pub struct I8042Cmd(pub Arc<I8042>);

impl MmioOps for I8042Cmd {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.read_status()))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.write_command(value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}
