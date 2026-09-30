// SPDX-License-Identifier: GPL-2.0-or-later

//! The 8259 interrupt controller, `isa-i8259`, from hw/intc/i8259.c and hw/intc/i8259_common.c.
//!
//! A PC has two, cascaded by [`i8259_init`]: the slave's output goes to IRQ2 of the master and
//! the master's output goes to the CPU, which acknowledges with [`I8259::pic_read_irq`]. Each
//! chip also has a PIIX style ELCR port that switches single lines to level triggering.
//!
//! VMState, trace points, QOM registration, the interrupt statistics and the KVM in-kernel
//! `kvm-i8259` are not ported.
//!
//! Each chip keeps its registers behind a mutex. The output line is set after the lock is
//! dropped, so the slave can drive the master and the master can read the slave while
//! acknowledging. The lock order is master before slave.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use ruvm_hw_core::irq::{IrqLine, IrqPin};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_I8259`.
pub const TYPE_I8259: &str = "isa-i8259";

/// `ISA_NUM_IRQS`.
pub const ISA_NUM_IRQS: usize = 16;

/// `PICCommonState` minus the QOM parts and the properties.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PicCommonState {
    /// Edge detection.
    pub last_irr: u8,
    /// Interrupt request register.
    pub irr: u8,
    /// Interrupt mask register.
    pub imr: u8,
    /// Interrupt service register.
    pub isr: u8,
    /// Highest irq priority.
    pub priority_add: u8,
    pub irq_base: u8,
    pub read_reg_select: u8,
    pub poll: u8,
    pub special_mask: u8,
    pub init_state: u8,
    pub auto_eoi: u8,
    pub rotate_on_auto_eoi: u8,
    pub special_fully_nested_mode: u8,
    /// True if 4 byte init.
    pub init4: u8,
    /// True if slave pic is not initialized.
    pub single_mode: u8,
    /// PIIX edge/trigger selection.
    pub elcr: u8,
    /// Edge/Level Bank Select (pre-PIIX, chip-wide).
    pub ltim: u8,
    /// Reflects the /SP input pin. A copy of [`I8259::master`].
    pub master: bool,
    /// Set by `pic_update_irq()`: the output has to be recomputed once the lock is dropped.
    update: bool,
}

/// `get_priority()`: the highest priority found in `mask` (highest = smallest number), or 8
/// if there is no irq.
fn get_priority(s: &PicCommonState, mask: u8) -> u8 {
    if mask == 0 {
        return 8;
    }
    let mut priority = 0;
    while mask & (1 << ((priority + s.priority_add) & 7)) == 0 {
        priority += 1;
    }
    priority
}

/// `pic_get_irq()`: the interrupt the chip wants to raise, or -1 if none.
fn pic_get_irq(s: &PicCommonState) -> i32 {
    let priority = get_priority(s, s.irr & !s.imr);
    if priority == 8 {
        return -1;
    }
    // Compute current priority. In special fully nested mode on the master, the IRQ coming
    // from the slave is not taken into account for the priority computation.
    let mut mask = s.isr;
    if s.special_mask != 0 {
        mask &= !s.imr;
    }
    if s.special_fully_nested_mode != 0 && s.master {
        mask &= !(1 << 2);
    }
    let cur_priority = get_priority(s, mask);
    if priority < cur_priority {
        // Higher priority found: an irq should be generated.
        i32::from((priority + s.priority_add) & 7)
    } else {
        -1
    }
}

/// `pic_update_irq()`. The output itself is set by the caller after unlocking.
fn pic_update_irq(s: &mut PicCommonState) {
    s.update = true;
}

/// `pic_set_irq()`: sets the level of input `irq`. An edge sets the bit in IRR.
fn pic_set_irq(s: &mut PicCommonState, irq: u32, level: i32) {
    let mask = 1u8 << irq;
    if s.ltim != 0 || s.elcr & mask != 0 {
        // Level triggered.
        if level != 0 {
            s.irr |= mask;
            s.last_irr |= mask;
        } else {
            s.irr &= !mask;
            s.last_irr &= !mask;
        }
    } else if level != 0 {
        // Edge triggered.
        if s.last_irr & mask == 0 {
            s.irr |= mask;
        }
        s.last_irr |= mask;
    } else {
        s.last_irr &= !mask;
    }
    pic_update_irq(s);
}

/// `pic_intack()`: acknowledges interrupt `irq`.
fn pic_intack(s: &mut PicCommonState, irq: i32) {
    let bit = 1u8 << irq;
    if s.auto_eoi != 0 {
        if s.rotate_on_auto_eoi != 0 {
            s.priority_add = ((irq + 1) & 7) as u8;
        }
    } else {
        s.isr |= bit;
    }
    // We don't clear a level sensitive interrupt here.
    if s.ltim == 0 && s.elcr & bit == 0 {
        s.irr &= !bit;
    }
    pic_update_irq(s);
}

/// `pic_reset_common()`. ELCR and LTIM are not reset.
pub fn pic_reset_common(s: &mut PicCommonState) {
    s.last_irr = 0;
    s.irr &= s.elcr;
    s.imr = 0;
    s.isr = 0;
    s.priority_add = 0;
    s.irq_base = 0;
    s.read_reg_select = 0;
    s.poll = 0;
    s.special_mask = 0;
    s.init_state = 0;
    s.auto_eoi = 0;
    s.rotate_on_auto_eoi = 0;
    s.special_fully_nested_mode = 0;
    s.init4 = 0;
    s.single_mode = 0;
}

/// `pic_init_reset()`.
fn pic_init_reset(s: &mut PicCommonState) {
    pic_reset_common(s);
    pic_update_irq(s);
}

/// `pic_ioport_write()`.
fn pic_ioport_write(s: &mut PicCommonState, addr: u64, val: u8) {
    if addr == 0 {
        if val & 0x10 != 0 {
            // ICW1
            pic_init_reset(s);
            s.init_state = 1;
            s.init4 = val & 1;
            s.single_mode = val & 2;
            s.ltim = val & 8;
        } else if val & 0x08 != 0 {
            // OCW3
            if val & 0x04 != 0 {
                s.poll = 1;
            }
            if val & 0x02 != 0 {
                s.read_reg_select = val & 1;
            }
            if val & 0x40 != 0 {
                s.special_mask = (val >> 5) & 1;
            }
        } else {
            // OCW2
            let cmd = val >> 5;
            match cmd {
                0 | 4 => s.rotate_on_auto_eoi = cmd >> 2,
                // End of interrupt.
                1 | 5 => {
                    let priority = get_priority(s, s.isr);
                    if priority != 8 {
                        let irq = (priority + s.priority_add) & 7;
                        s.isr &= !(1 << irq);
                        if cmd == 5 {
                            s.priority_add = (irq + 1) & 7;
                        }
                        pic_update_irq(s);
                    }
                }
                3 => {
                    let irq = val & 7;
                    s.isr &= !(1 << irq);
                    pic_update_irq(s);
                }
                6 => {
                    s.priority_add = (val + 1) & 7;
                    pic_update_irq(s);
                }
                7 => {
                    let irq = val & 7;
                    s.isr &= !(1 << irq);
                    s.priority_add = (irq + 1) & 7;
                    pic_update_irq(s);
                }
                // No operation.
                _ => {}
            }
        }
    } else {
        match s.init_state {
            0 => {
                // Normal mode.
                s.imr = val;
                pic_update_irq(s);
            }
            1 => {
                s.irq_base = val & 0xf8;
                s.init_state =
                    if s.single_mode != 0 { if s.init4 != 0 { 3 } else { 0 } } else { 2 };
            }
            2 => {
                s.init_state = if s.init4 != 0 { 3 } else { 0 };
            }
            3 => {
                s.special_fully_nested_mode = (val >> 4) & 1;
                s.auto_eoi = (val >> 1) & 1;
                s.init_state = 0;
            }
            _ => {}
        }
    }
}

/// `pic_ioport_read()`.
fn pic_ioport_read(s: &mut PicCommonState, addr: u64) -> u8 {
    if s.poll != 0 {
        let mut ret = pic_get_irq(s);
        if ret >= 0 {
            pic_intack(s, ret);
            ret |= 0x80;
        } else {
            ret = 0;
        }
        s.poll = 0;
        ret as u8
    } else if addr == 0 {
        if s.read_reg_select != 0 { s.isr } else { s.irr }
    } else {
        s.imr
    }
}

/// The `isa-i8259` device.
#[derive(Debug)]
pub struct I8259 {
    /// The `iobase` property: 0x20 for the master, 0xa0 for the slave.
    pub iobase: u32,
    /// The `elcr_addr` property: 0x4d0 or 0x4d1, or -1 for no ELCR port.
    pub elcr_addr: u32,
    /// The `elcr_mask` property: the ELCR bits that can be set.
    pub elcr_mask: u8,
    /// The `master` property.
    pub master: bool,
    state: Mutex<PicCommonState>,
    /// `int_out[0]`: INT, to the CPU or to IRQ2 of the master.
    pub int_out: IrqPin,
    /// The slave, for the master to acknowledge through. `slave_pic` in the C.
    slave: OnceLock<Arc<I8259>>,
}

impl I8259 {
    /// Creates a chip in its reset state.
    pub fn new(iobase: u32, elcr_addr: u32, elcr_mask: u8, master: bool) -> Arc<Self> {
        let s = PicCommonState { master, ..Default::default() };
        Arc::new(I8259 {
            iobase,
            elcr_addr,
            elcr_mask,
            master,
            state: Mutex::new(s),
            int_out: IrqPin::new(),
            slave: OnceLock::new(),
        })
    }

    /// `i8259_init_chip()`: a chip with the PC's addresses and ELCR mask.
    pub fn new_pc(master: bool) -> Arc<Self> {
        if master {
            Self::new(0x20, 0x4d0, 0xf8, true)
        } else {
            Self::new(0xa0, 0x4d1, 0xde, false)
        }
    }

    fn lock(&self) -> MutexGuard<'_, PicCommonState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Takes the pending `pic_update_irq()` and returns the output level to set.
    fn take_update(s: &mut PicCommonState) -> Option<bool> {
        if s.update {
            s.update = false;
            Some(pic_get_irq(s) >= 0)
        } else {
            None
        }
    }

    fn set_output(&self, level: Option<bool>) {
        if let Some(level) = level {
            self.int_out.set_bool(level);
        }
    }

    /// Runs `f` on the registers and then updates the output if `f` asked for it.
    fn with_state<R>(&self, f: impl FnOnce(&mut PicCommonState) -> R) -> R {
        let (r, level) = {
            let mut s = self.lock();
            let r = f(&mut s);
            (r, Self::take_update(&mut s))
        };
        self.set_output(level);
        r
    }

    /// A copy of the registers.
    pub fn state(&self) -> PicCommonState {
        let mut s = self.lock().clone();
        s.update = false;
        s
    }

    /// The slave, once [`i8259_init`] has wired it.
    pub fn slave(&self) -> Option<&Arc<I8259>> {
        self.slave.get()
    }

    /// `pic_set_irq()`: gpio in `irq`.
    pub fn set_irq(&self, irq: u32, level: i32) {
        assert!(irq < 8, "i8259 has 8 inputs");
        self.with_state(|s| pic_set_irq(s, irq, level));
    }

    /// gpio in `n`, `qdev_get_gpio_in(dev, n)`.
    pub fn irq_in(self: &Arc<Self>, n: u32) -> IrqLine {
        assert!(n < 8, "i8259 has 8 inputs");
        let w = Arc::downgrade(self);
        IrqLine::new(
            Arc::new(move |irq, level| {
                if let Some(pic) = w.upgrade() {
                    pic.set_irq(irq, level);
                }
            }),
            n,
        )
    }

    /// `pic_ioport_write()` on the two base ports.
    pub fn ioport_write(&self, addr: u64, val: u8) {
        self.with_state(|s| pic_ioport_write(s, addr & 1, val));
    }

    /// `pic_ioport_read()` on the two base ports.
    pub fn ioport_read(&self, addr: u64) -> u8 {
        self.with_state(|s| pic_ioport_read(s, addr & 1))
    }

    /// `elcr_ioport_write()`.
    pub fn elcr_ioport_write(&self, val: u8) {
        let mask = self.elcr_mask;
        self.with_state(|s| s.elcr = val & mask);
    }

    /// `elcr_ioport_read()`.
    pub fn elcr_ioport_read(&self) -> u8 {
        self.lock().elcr
    }

    /// `pic_get_output()`.
    pub fn pic_get_output(&self) -> bool {
        pic_get_irq(&self.lock()) >= 0
    }

    /// `pic_read_irq()`: the interrupt acknowledge cycle of the CPU. Called on the master, it
    /// returns the vector and acknowledges the interrupt, going through the slave for IRQ2.
    /// With nothing pending it returns the spurious IRQ7 vector.
    pub fn pic_read_irq(&self) -> u8 {
        let mut slave_level = None;
        let (intno, level) = {
            let mut s = self.lock();
            let irq = pic_get_irq(&s);
            let intno = if irq < 0 {
                // Spurious IRQ on host controller.
                s.irq_base + 7
            } else {
                match self.slave.get() {
                    Some(slave) if irq == 2 => {
                        let mut ss = slave.lock();
                        let mut irq2 = pic_get_irq(&ss);
                        if irq2 >= 0 {
                            pic_intack(&mut ss, irq2);
                        } else {
                            // Spurious IRQ on slave controller.
                            irq2 = 7;
                        }
                        slave_level = Self::take_update(&mut ss);
                        let intno = ss.irq_base + irq2 as u8;
                        drop(ss);
                        pic_intack(&mut s, irq);
                        intno
                    }
                    // Without a slave IRQ2 is a plain input.
                    _ => {
                        pic_intack(&mut s, irq);
                        s.irq_base + irq as u8
                    }
                }
            };
            (intno, Self::take_update(&mut s))
        };
        // The master first: the slave's output goes into the master and updates it again.
        self.set_output(level);
        if let Some(slave) = self.slave.get() {
            slave.set_output(slave_level);
        }
        intno
    }

    /// `pic_reset()`.
    pub fn reset(&self) {
        self.with_state(|s| {
            s.elcr = 0;
            s.ltim = 0;
            pic_init_reset(s);
        });
    }

    /// `pic_print_info()`, the `info pic` line.
    pub fn print_info(&self) -> String {
        let s = self.lock();
        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "pic{}: irr={:02x} imr={:02x} isr={:02x} hprio={} irq_base={:02x} rr_sel={} \
             elcr={:02x} fnm={}",
            if self.master { 0 } else { 1 },
            s.irr,
            s.imr,
            s.isr,
            s.priority_add,
            s.irq_base,
            s.read_reg_select,
            s.elcr,
            s.special_fully_nested_mode
        );
        buf
    }

    /// The ELCR port as its own region, `pic_elcr_ioport_ops`.
    pub fn elcr_io(self: &Arc<Self>) -> Arc<I8259Elcr> {
        Arc::new(I8259Elcr(self.clone()))
    }
}

/// `pic_base_ioport_ops`: the two command and data ports.
impl MmioOps for I8259 {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.ioport_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.ioport_write(offset, value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

/// `pic_elcr_ioport_ops`: the one byte ELCR port of a chip.
#[derive(Debug)]
pub struct I8259Elcr(pub Arc<I8259>);

impl MmioOps for I8259Elcr {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.elcr_ioport_read()))
    }

    fn write(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.elcr_ioport_write(value as u8);
        Ok(())
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}

/// What [`i8259_init`] builds.
#[derive(Debug)]
pub struct I8259Pair {
    /// The master, `isa_pic`. The CPU acknowledges through it.
    pub master: Arc<I8259>,
    pub slave: Arc<I8259>,
    /// The 16 ISA IRQ lines, 0 to 7 on the master and 8 to 15 on the slave.
    pub irq_set: Vec<IrqLine>,
}

/// `i8259_init()`: creates the master and slave, wires the master's output to
/// `parent_irq_in` and the slave's output to IRQ2 of the master.
pub fn i8259_init(parent_irq_in: IrqLine) -> I8259Pair {
    let master = I8259::new_pc(true);
    master.int_out.connect(parent_irq_in);
    let mut irq_set: Vec<IrqLine> = (0..8).map(|i| master.irq_in(i)).collect();

    let slave = I8259::new_pc(false);
    slave.int_out.connect(irq_set[2].clone());
    irq_set.extend((0..8).map(|i| slave.irq_in(i)));

    let _ = master.slave.set(slave.clone());
    I8259Pair { master, slave, irq_set }
}
