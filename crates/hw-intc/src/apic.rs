// SPDX-License-Identifier: GPL-2.0-or-later

//! The emulated local APIC, from hw/intc/apic.c, hw/intc/apic_common.c,
//! include/hw/i386/apic_internal.h and include/hw/i386/apic-msidef.h.
//!
//! Each vCPU has one [`Apic`]. They all sit on an [`ApicBus`], the `local_apics[]` array of
//! QEMU, indexed by initial APIC ID, which delivers MSIs ([`ApicBus::send_msi`]), IPIs and
//! the EOIs of level triggered vectors back to the IOAPICs. The CPU side is the [`ApicCpu`]
//! trait: raising and clearing the CPU's interrupt request bits, telling whether the caller
//! runs on the CPU's own thread (`qemu_cpu_is_self()`), and the CPUID bits the APIC reads or
//! changes. The register window at 0xfee00000 is [`ApicMmio`], which, like QEMU, finds the
//! APIC of the vCPU making the access through a per thread "current APIC"
//! ([`set_current_apic`]) and turns writes outside the registers into MSIs.
//!
//! The timer runs on the virtual clock given to [`Apic::realize`], with the divide
//! configuration, one-shot and periodic modes of QEMU. The TSC deadline mode is not offered,
//! as in QEMU's TCG, which leaves `tsc-deadline` out of CPUID.
//!
//! QEMU runs all of this under the big lock. Here each APIC has its own lock. Work that
//! reaches another APIC, the IOAPICs or the CPU (IPIs, EOI broadcasts, interrupt requests) is
//! collected while the lock is held and done once it is dropped, so an APIC never holds its
//! lock while taking another APIC's. The 8259 is asked for its output with an APIC lock held;
//! the 8259 never calls out with its own lock held, so the order is safe.
//!
//! Deliberate differences from QEMU:
//!
//! - The VAPIC (kvmvapic option ROM support, `apic_sync_vapic()`) is not ported, so TPR
//!   access reporting does nothing and CR8 updates always reach the TPR.
//! - VMState, trace points, QOM registration and the KVM, Xen, WHPX and MSHV APICs are not
//!   ported.

use std::cell::RefCell;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, RwLock, Weak};

use ruvm_base::Error;
use ruvm_hw_core::timer::{Clock, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

use crate::i8259::I8259;
use crate::ioapic::IoApics;

/// `APIC_DEFAULT_ADDRESS`.
pub const APIC_DEFAULT_ADDRESS: u64 = 0xfee0_0000;
/// `APIC_SPACE_SIZE`.
pub const APIC_SPACE_SIZE: u64 = 0x10_0000;

pub const APIC_LVT_TIMER: usize = 0;
pub const APIC_LVT_THERMAL: usize = 1;
pub const APIC_LVT_PERFORM: usize = 2;
pub const APIC_LVT_LINT0: usize = 3;
pub const APIC_LVT_LINT1: usize = 4;
pub const APIC_LVT_ERROR: usize = 5;
pub const APIC_LVT_NB: usize = 6;

pub const APIC_DM_FIXED: u8 = 0;
pub const APIC_DM_LOWPRI: u8 = 1;
pub const APIC_DM_SMI: u8 = 2;
pub const APIC_DM_NMI: u8 = 4;
pub const APIC_DM_INIT: u8 = 5;
pub const APIC_DM_SIPI: u8 = 6;
pub const APIC_DM_EXTINT: u8 = 7;

pub const APIC_DESTMODE_PHYSICAL: u8 = 0;
pub const APIC_DESTMODE_LOGICAL: u8 = 1;
pub const APIC_DESTMODE_LOGICAL_FLAT: u8 = 0xf;
pub const APIC_DESTMODE_LOGICAL_CLUSTER: u8 = 0;

pub const APIC_TRIGGER_EDGE: u8 = 0;
pub const APIC_TRIGGER_LEVEL: u8 = 1;

pub const APIC_LVT_TIMER_PERIODIC: u32 = 1 << 17;
pub const APIC_LVT_MASKED: u32 = 1 << 16;
pub const APIC_LVT_LEVEL_TRIGGER: u32 = 1 << 15;

pub const APIC_ESR_ILLEGAL_ADDRESS: u32 = 1 << 7;

pub const APIC_SV_DIRECTED_IO: u32 = 1 << 12;
pub const APIC_SV_ENABLE: u32 = 1 << 8;

pub const MSR_IA32_APICBASE_BSP: u64 = 1 << 8;
pub const MSR_IA32_APICBASE_EXTD: u64 = 1 << 10;
pub const MSR_IA32_APICBASE_ENABLE: u64 = 1 << 11;
pub const MSR_IA32_APICBASE_BASE: u64 = 0xfffff << 12;

/// The APIC version register, `s->version`.
const APIC_VERSION: u32 = 0x14;

/// The CPU interrupt requests an APIC raises or clears, the `CPU_INTERRUPT_*` bits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CpuIrq {
    /// `CPU_INTERRUPT_HARD`.
    Hard,
    /// `CPU_INTERRUPT_POLL`.
    Poll,
    /// `CPU_INTERRUPT_SMI`.
    Smi,
    /// `CPU_INTERRUPT_NMI`.
    Nmi,
    /// `CPU_INTERRUPT_INIT`.
    Init,
    /// `CPU_INTERRUPT_SIPI`.
    Sipi,
}

/// The CPU an APIC belongs to, `s->cpu`.
pub trait ApicCpu: Send + Sync {
    /// `cpu_interrupt()`.
    fn cpu_interrupt(&self, irq: CpuIrq);
    /// `cpu_reset_interrupt()`.
    fn cpu_reset_interrupt(&self, irq: CpuIrq);
    /// `qemu_cpu_is_self()`: whether the caller runs on the CPU's thread.
    fn is_self(&self) -> bool;
    /// `cpu_has_x2apic_feature()`.
    fn has_x2apic(&self) -> bool;
    /// `cpu_set_apic_feature()` with true, `cpu_clear_apic_feature()` with false.
    fn set_apic_feature(&self, on: bool);
}

/// The registers of `APICCommonState` that change at run time.
#[derive(Clone, Debug, Default)]
struct ApicState {
    apicbase: u64,
    id: u8,
    arb_id: u8,
    tpr: u8,
    spurious_vec: u32,
    log_dest: u8,
    dest_mode: u8,
    isr: [u32; 8],
    tmr: [u32; 8],
    irr: [u32; 8],
    lvt: [u32; APIC_LVT_NB],
    esr: u32,
    icr: [u32; 2],
    divide_conf: u32,
    count_shift: u32,
    initial_count: u32,
    initial_count_load_time: i64,
    next_time: i64,
    timer_expiry: i64,
    sipi_vector: u8,
    wait_for_sipi: bool,
    extended_log_dest: u32,
}

impl ApicState {
    fn is_x2apic_mode(&self) -> bool {
        self.apicbase & MSR_IA32_APICBASE_EXTD != 0
    }

    /// `apic_get_ppr()`.
    fn get_ppr(&self) -> u32 {
        let tpr = u32::from(self.tpr >> 4);
        let isrv = (get_highest_priority_int(&self.isr).max(0) as u32) >> 4;
        if tpr >= isrv { u32::from(self.tpr) } else { isrv << 4 }
    }

    /// `apic_irq_pending()`: below zero for an interrupt masked by the priority, zero for
    /// none, otherwise the vector.
    fn irq_pending(&self) -> i32 {
        if self.spurious_vec & APIC_SV_ENABLE == 0 {
            return 0;
        }
        let irrv = get_highest_priority_int(&self.irr);
        if irrv < 0 {
            return 0;
        }
        let ppr = self.get_ppr() as i32;
        if ppr != 0 && (irrv & 0xf0) <= (ppr & 0xf0) {
            return -1;
        }
        irrv
    }

    /// `apic_next_timer()`.
    fn next_timer(&mut self, current_time: i64) -> bool {
        self.timer_expiry = -1;
        let lvt = self.lvt[APIC_LVT_TIMER];
        if lvt & APIC_LVT_MASKED != 0 {
            return false;
        }
        let mut d = (current_time.wrapping_sub(self.initial_count_load_time)) >> self.count_shift;
        let ic = u64::from(self.initial_count);
        if lvt & APIC_LVT_TIMER_PERIODIC != 0 {
            if ic == 0 {
                return false;
            }
            d = (((d as u64) / (ic + 1) + 1) * (ic + 1)) as i64;
        } else {
            if d >= ic as i64 {
                return false;
            }
            d = (ic + 1) as i64;
        }
        self.next_time = self.initial_count_load_time.wrapping_add(d << self.count_shift);
        self.timer_expiry = self.next_time;
        true
    }

    /// `apic_get_current_count()`.
    fn current_count(&self, now: i64) -> u32 {
        let d = (now.wrapping_sub(self.initial_count_load_time)) >> self.count_shift;
        let ic = u64::from(self.initial_count);
        if self.lvt[APIC_LVT_TIMER] & APIC_LVT_TIMER_PERIODIC != 0 {
            (ic - (d as u64) % (ic + 1)) as u32
        } else if d >= ic as i64 {
            0
        } else {
            (ic as i64 - d) as u32
        }
    }
}

fn get_bit(tab: &[u32], index: usize) -> bool {
    tab[index >> 5] & (1 << (index & 0x1f)) != 0
}

fn set_bit(tab: &mut [u32], index: usize) {
    tab[index >> 5] |= 1 << (index & 0x1f);
}

fn reset_bit(tab: &mut [u32], index: usize) {
    tab[index >> 5] &= !(1 << (index & 0x1f));
}

/// `get_highest_priority_int()`: the highest set bit, or -1.
fn get_highest_priority_int(tab: &[u32; 8]) -> i32 {
    for i in (0..8).rev() {
        if tab[i] != 0 {
            return (i as i32) * 32 + 31 - tab[i].leading_zeros() as i32;
        }
    }
    -1
}

/// An interrupt command, what `apic_deliver()` gets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Ipi {
    dest: u32,
    dest_mode: u8,
    delivery_mode: u8,
    vector: u8,
    trigger_mode: u8,
    shorthand: u8,
    /// `s->icr[0]` when the command was written.
    icr0: u32,
}

/// Work found with the APIC lock held and done after it is dropped.
#[derive(Debug)]
enum Act {
    Irq(CpuIrq),
    ResetIrq(CpuIrq),
    /// `ioapic_eoi_broadcast()`.
    Eoi(i32),
    /// `apic_update_irq()` again, after the work before it.
    UpdateIrq,
    Ipi(Ipi),
}

type Acts = Vec<Act>;

/// What delivery looks at: the APIC base, ID, LDR, DFR model and x2APIC LDR.
type DestInfo = (u64, u8, u8, u8, u32);

/// One local APIC, `APICCommonState` of type `apic`.
pub struct Apic {
    initial_apic_id: u32,
    cpu: Arc<dyn ApicCpu>,
    bus: Weak<ApicBus>,
    clock: Arc<Clock>,
    timer: Timer,
    state: Mutex<ApicState>,
}

impl fmt::Debug for Apic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Apic")
            .field("initial_apic_id", &self.initial_apic_id)
            .field("state", &*self.lock())
            .finish_non_exhaustive()
    }
}

impl Apic {
    /// `apic_common_realize()` and `apic_realize()`, then the reset of `apic_reset_common()`
    /// with the BSP bit for `bsp`. The new APIC takes slot `initial_apic_id` of `bus`, and its
    /// timer runs on `clock`, the virtual clock.
    pub fn realize(
        bus: &Arc<ApicBus>,
        clock: &Arc<Clock>,
        initial_apic_id: u32,
        bsp: bool,
        cpu: Arc<dyn ApicCpu>,
    ) -> Result<Arc<Apic>, Error> {
        if initial_apic_id >= 255 && !cpu.has_x2apic() {
            return Err(Error::generic(format!(
                "APIC ID {initial_apic_id} requires x2APIC feature in CPU"
            ))
            .hint("Try x2apic=on in -cpu.\n"));
        }
        let s = Arc::new_cyclic(|weak: &Weak<Apic>| {
            let w = weak.clone();
            let timer = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    s.timer_cb();
                }
            });
            Apic {
                initial_apic_id,
                cpu,
                bus: Arc::downgrade(bus),
                clock: Arc::clone(clock),
                timer,
                state: Mutex::new(ApicState {
                    // APIC LDR in x2APIC mode.
                    extended_log_dest: ((initial_apic_id >> 4) << 16)
                        | (1 << (initial_apic_id & 0xf)),
                    ..ApicState::default()
                }),
            }
        });
        bus.insert(&s)?;
        s.reset(bsp);
        Ok(s)
    }

    fn lock(&self) -> MutexGuard<'_, ApicState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `s->initial_apic_id`.
    pub fn initial_apic_id(&self) -> u32 {
        self.initial_apic_id
    }

    /// Runs `f` on the state, then does the work it asked for.
    fn with<R>(&self, f: impl FnOnce(&Apic, &mut ApicState, &mut Acts) -> R) -> R {
        let mut acts = Vec::new();
        let r = {
            let mut s = self.lock();
            f(self, &mut s, &mut acts)
        };
        self.run(acts);
        r
    }

    fn run(&self, acts: Acts) {
        for a in acts {
            match a {
                Act::Irq(k) => self.cpu.cpu_interrupt(k),
                Act::ResetIrq(k) => self.cpu.cpu_reset_interrupt(k),
                Act::Eoi(v) => {
                    if let Some(bus) = self.bus.upgrade() {
                        bus.ioapics.eoi_broadcast(v);
                    }
                }
                Act::UpdateIrq => self.with(|a, s, acts| a.update_irq(s, acts)),
                Act::Ipi(ipi) => {
                    if let Some(bus) = self.bus.upgrade() {
                        bus.deliver(self.initial_apic_id, ipi);
                    }
                }
            }
        }
    }

    fn pic(&self) -> Option<Arc<I8259>> {
        self.bus.upgrade().and_then(|b| b.pic.clone())
    }

    /// `apic_accept_pic_intr()` on locked state.
    fn accept_pic_intr_locked(&self, s: &ApicState) -> bool {
        let lvt0 = s.lvt[APIC_LVT_LINT0];
        if s.apicbase & MSR_IA32_APICBASE_ENABLE == 0 || lvt0 & APIC_LVT_MASKED == 0 {
            return self.pic().is_some();
        }
        false
    }

    fn pic_output(&self) -> bool {
        self.pic().is_some_and(|p| p.pic_get_output())
    }

    /// `apic_update_irq()`: signal the CPU if an interrupt is pending.
    fn update_irq(&self, s: &ApicState, acts: &mut Acts) {
        if !self.cpu.is_self() {
            acts.push(Act::Irq(CpuIrq::Poll));
        } else if s.irq_pending() > 0 {
            acts.push(Act::Irq(CpuIrq::Hard));
        } else if !self.accept_pic_intr_locked(s) || !self.pic_output() {
            acts.push(Act::ResetIrq(CpuIrq::Hard));
        }
    }

    /// `apic_set_irq()`.
    fn set_irq_locked(&self, s: &mut ApicState, vector: u8, trigger_mode: u8, acts: &mut Acts) {
        let v = usize::from(vector);
        set_bit(&mut s.irr, v);
        if trigger_mode != 0 {
            set_bit(&mut s.tmr, v);
        } else {
            reset_bit(&mut s.tmr, v);
        }
        self.update_irq(s, acts);
    }

    /// `apic_set_irq()` from outside: an interrupt for `vector` arrives.
    pub fn set_irq(&self, vector: u8, trigger_mode: u8) {
        self.with(|a, s, acts| a.set_irq_locked(s, vector, trigger_mode, acts));
    }

    /// `apic_local_deliver()`.
    fn local_deliver(&self, s: &mut ApicState, vector: usize, acts: &mut Acts) {
        let lvt = s.lvt[vector];
        if lvt & APIC_LVT_MASKED != 0 {
            return;
        }
        match ((lvt >> 8) & 7) as u8 {
            APIC_DM_SMI => acts.push(Act::Irq(CpuIrq::Smi)),
            APIC_DM_NMI => acts.push(Act::Irq(CpuIrq::Nmi)),
            APIC_DM_EXTINT => acts.push(Act::Irq(CpuIrq::Hard)),
            APIC_DM_FIXED => {
                let mut trigger_mode = APIC_TRIGGER_EDGE;
                if (vector == APIC_LVT_LINT0 || vector == APIC_LVT_LINT1)
                    && lvt & APIC_LVT_LEVEL_TRIGGER != 0
                {
                    trigger_mode = APIC_TRIGGER_LEVEL;
                }
                self.set_irq_locked(s, lvt as u8, trigger_mode, acts);
            }
            _ => {}
        }
    }

    fn deliver_pic_intr_locked(&self, s: &mut ApicState, level: bool, acts: &mut Acts) {
        if level {
            self.local_deliver(s, APIC_LVT_LINT0, acts);
        } else {
            let lvt = s.lvt[APIC_LVT_LINT0];
            match ((lvt >> 8) & 7) as u8 {
                APIC_DM_FIXED => {
                    if lvt & APIC_LVT_LEVEL_TRIGGER != 0 {
                        reset_bit(&mut s.irr, (lvt & 0xff) as usize);
                        self.update_irq(s, acts);
                    }
                }
                APIC_DM_EXTINT => self.update_irq(s, acts),
                _ => {}
            }
        }
    }

    /// `apic_deliver_pic_intr()`: the 8259's INTR line, wired to LINT0, changed to `level`.
    pub fn deliver_pic_intr(&self, level: bool) {
        self.with(|a, s, acts| a.deliver_pic_intr_locked(s, level, acts));
    }

    /// `apic_deliver_nmi()`: an NMI from outside, through LINT1.
    pub fn deliver_nmi(&self) {
        self.with(|a, s, acts| a.local_deliver(s, APIC_LVT_LINT1, acts));
    }

    /// `apic_accept_pic_intr()`: whether interrupts of the 8259 go through this APIC.
    pub fn accept_pic_intr(&self) -> bool {
        let s = self.lock();
        self.accept_pic_intr_locked(&s)
    }

    /// `apic_poll_irq()`.
    pub fn poll_irq(&self) {
        self.with(|a, s, acts| a.update_irq(s, acts));
    }

    /// `apic_get_interrupt()`: acknowledge the highest priority interrupt and return its
    /// vector, the spurious vector when the pending ones are masked by the priority, or -1
    /// when there is nothing to take or the 8259 goes first.
    pub fn get_interrupt(&self) -> i32 {
        self.with(|a, s, acts| {
            if s.spurious_vec & APIC_SV_ENABLE == 0 {
                return -1;
            }
            let intno = s.irq_pending();
            // If there is an interrupt from the 8259, let the caller handle that first
            // since ExtINT interrupts ignore the priority.
            if intno == 0 || a.check_pic(s, acts) {
                return -1;
            } else if intno < 0 {
                return (s.spurious_vec & 0xff) as i32;
            }
            reset_bit(&mut s.irr, intno as usize);
            set_bit(&mut s.isr, intno as usize);
            a.update_irq(s, acts);
            intno
        })
    }

    /// `apic_check_pic()`.
    fn check_pic(&self, s: &mut ApicState, acts: &mut Acts) -> bool {
        if !self.accept_pic_intr_locked(s) || !self.pic_output() {
            return false;
        }
        self.deliver_pic_intr_locked(s, true, acts);
        true
    }

    /// `apic_eoi()`.
    fn eoi(&self, s: &mut ApicState, acts: &mut Acts) {
        let isrv = get_highest_priority_int(&s.isr);
        if isrv < 0 {
            return;
        }
        reset_bit(&mut s.isr, isrv as usize);
        if s.spurious_vec & APIC_SV_DIRECTED_IO == 0 && get_bit(&s.tmr, isrv as usize) {
            acts.push(Act::Eoi(isrv));
        }
        acts.push(Act::UpdateIrq);
    }

    /// `apic_sipi()`: the startup vector, if the APIC waits for a SIPI. QEMU loads CS from it
    /// here with `cpu_x86_load_seg_cache_sipi()`; the caller does that.
    pub fn sipi(&self) -> Option<u8> {
        let mut s = self.lock();
        if !s.wait_for_sipi {
            return None;
        }
        s.wait_for_sipi = false;
        Some(s.sipi_vector)
    }

    /// `apic_init_reset()`.
    pub fn init_reset(&self) {
        let mut s = self.lock();
        Self::init_reset_locked(&mut s);
        self.timer.del();
    }

    fn init_reset_locked(s: &mut ApicState) {
        s.tpr = 0;
        s.spurious_vec = 0xff;
        s.log_dest = 0;
        s.dest_mode = 0xf;
        s.isr = [0; 8];
        s.tmr = [0; 8];
        s.irr = [0; 8];
        s.lvt = [APIC_LVT_MASKED; APIC_LVT_NB];
        s.esr = 0;
        s.icr = [0; 2];
        s.divide_conf = 0;
        s.count_shift = 0;
        s.initial_count = 0;
        s.initial_count_load_time = 0;
        s.next_time = 0;
        s.wait_for_sipi = s.apicbase & MSR_IA32_APICBASE_BSP == 0;
        s.timer_expiry = -1;
    }

    /// `apic_designate_bsp()` and then `apic_reset_common()`, the cold reset of the device.
    pub fn reset(&self, bsp: bool) {
        let mut s = self.lock();
        s.apicbase = APIC_DEFAULT_ADDRESS
            | if bsp { MSR_IA32_APICBASE_BSP } else { 0 }
            | MSR_IA32_APICBASE_ENABLE;
        s.id = self.initial_apic_id as u8;
        Self::init_reset_locked(&mut s);
        self.timer.del();
    }

    /// `cpu_get_apic_base()`.
    pub fn apic_base(&self) -> u64 {
        self.lock().apicbase
    }

    /// `cpu_is_apic_enabled()`.
    pub fn is_enabled(&self) -> bool {
        self.lock().apicbase & MSR_IA32_APICBASE_ENABLE != 0
    }

    /// `cpu_set_apic_base()`: false for a transition the APIC refuses.
    pub fn set_base(&self, val: u64) -> bool {
        let mut s = self.lock();
        // Reset a possibly modified xAPIC ID.
        s.id = self.initial_apic_id as u8;
        self.set_base_locked(&mut s, val)
    }

    /// `apic_set_base_check()`.
    fn set_base_check(&self, s: &ApicState, val: u64) -> bool {
        let x2 = self.cpu.has_x2apic();
        let cur_en = s.apicbase & MSR_IA32_APICBASE_ENABLE != 0;
        let cur_ext = s.apicbase & MSR_IA32_APICBASE_EXTD != 0;
        let en = val & MSR_IA32_APICBASE_ENABLE != 0;
        let ext = val & MSR_IA32_APICBASE_EXTD != 0;
        // Enabling x2APIC when the CPU does not have it.
        if !x2 && ext {
            return false;
        }
        // Into the invalid state: disabled with x2APIC set.
        if !en && ext {
            return false;
        }
        // From disabled straight to x2APIC.
        if !cur_en && !cur_ext && en && ext {
            return false;
        }
        // From x2APIC back to xAPIC.
        if cur_en && cur_ext && en && !ext {
            return false;
        }
        true
    }

    /// `apic_set_base()`.
    fn set_base_locked(&self, s: &mut ApicState, val: u64) -> bool {
        if !self.set_base_check(s, val) {
            return false;
        }
        s.apicbase = (val & MSR_IA32_APICBASE_BASE)
            | (s.apicbase & (MSR_IA32_APICBASE_BSP | MSR_IA32_APICBASE_ENABLE));
        if val & MSR_IA32_APICBASE_ENABLE == 0 {
            s.apicbase &= !MSR_IA32_APICBASE_ENABLE;
            self.cpu.set_apic_feature(false);
            s.spurious_vec &= !APIC_SV_ENABLE;
        }
        // From disabled to xAPIC.
        if s.apicbase & MSR_IA32_APICBASE_ENABLE == 0 && val & MSR_IA32_APICBASE_ENABLE != 0 {
            s.apicbase |= MSR_IA32_APICBASE_ENABLE;
            self.cpu.set_apic_feature(true);
        }
        // From xAPIC to x2APIC.
        if self.cpu.has_x2apic()
            && s.apicbase & MSR_IA32_APICBASE_EXTD == 0
            && val & MSR_IA32_APICBASE_EXTD != 0
        {
            s.apicbase |= MSR_IA32_APICBASE_EXTD;
            let id = self.initial_apic_id;
            s.log_dest = ((((id & 0xffff0) << 16) | (1 << (id & 0xf))) & 0xff) as u8;
        }
        true
    }

    /// `cpu_get_apic_tpr()`: the TPR as CR8 sees it.
    pub fn tpr(&self) -> u8 {
        self.lock().tpr >> 4
    }

    /// `cpu_set_apic_tpr()`, from a CR8 write.
    pub fn set_tpr(&self, val: u8) {
        self.with(|a, s, acts| {
            s.tpr = val << 4;
            a.update_irq(s, acts);
        });
    }

    /// `apic_timer_update()`.
    fn timer_update(&self, s: &mut ApicState, current_time: i64) {
        if s.next_timer(current_time) {
            self.timer.modify(s.next_time);
        } else {
            self.timer.del();
        }
    }

    /// `apic_timer()`.
    fn timer_cb(&self) {
        self.with(|a, s, acts| {
            a.local_deliver(s, APIC_LVT_TIMER, acts);
            let next = s.next_time;
            a.timer_update(s, next);
        });
    }

    /// `apic_register_read()`: `None` where QEMU returns -1.
    fn register_read(&self, s: &mut ApicState, index: u32) -> Option<u64> {
        let val: u32 = match index {
            0x02 => {
                if s.is_x2apic_mode() {
                    self.initial_apic_id
                } else {
                    u32::from(s.id) << 24
                }
            }
            0x03 => APIC_VERSION | ((APIC_LVT_NB as u32 - 1) << 16),
            0x08 => u32::from(s.tpr),
            // Arbitration priority: not modelled, as in QEMU.
            0x09 => 0,
            0x0a => s.get_ppr(),
            0x0b => 0,
            0x0d => {
                if s.is_x2apic_mode() {
                    s.extended_log_dest
                } else {
                    u32::from(s.log_dest) << 24
                }
            }
            0x0e => {
                if s.is_x2apic_mode() {
                    return None;
                }
                (u32::from(s.dest_mode) << 28) | 0x0fff_ffff
            }
            0x0f => s.spurious_vec,
            0x10..=0x17 => s.isr[(index & 7) as usize],
            0x18..=0x1f => s.tmr[(index & 7) as usize],
            0x20..=0x27 => s.irr[(index & 7) as usize],
            0x28 => s.esr,
            0x30 | 0x31 => s.icr[(index & 1) as usize],
            0x32..=0x37 => s.lvt[(index - 0x32) as usize],
            0x38 => s.initial_count,
            0x39 => s.current_count(self.clock.get_ns()),
            0x3e => s.divide_conf,
            _ => {
                s.esr |= APIC_ESR_ILLEGAL_ADDRESS;
                return None;
            }
        };
        Some(u64::from(val))
    }

    /// `apic_register_write()`: false where QEMU returns -1.
    fn register_write(&self, s: &mut ApicState, index: u32, val: u64, acts: &mut Acts) -> bool {
        match index {
            0x02 => {
                if s.is_x2apic_mode() {
                    return false;
                }
                s.id = (val >> 24) as u8;
            }
            0x03 | 0x09 | 0x0a | 0x10..=0x28 | 0x39 => {}
            0x08 => {
                s.tpr = val as u8;
                self.update_irq(s, acts);
            }
            0x0b => self.eoi(s, acts),
            0x0d => {
                if s.is_x2apic_mode() {
                    return false;
                }
                s.log_dest = (val >> 24) as u8;
            }
            0x0e => {
                if s.is_x2apic_mode() {
                    return false;
                }
                s.dest_mode = ((val as u32) >> 28) as u8;
            }
            0x0f => {
                s.spurious_vec = (val & 0x1ff) as u32;
                self.update_irq(s, acts);
            }
            0x30 => {
                s.icr[0] = val as u32;
                let dest = if s.is_x2apic_mode() {
                    s.icr[1] = (val >> 32) as u32;
                    s.icr[1]
                } else {
                    (s.icr[1] >> 24) & 0xff
                };
                let icr0 = s.icr[0];
                acts.push(Act::Ipi(Ipi {
                    dest,
                    dest_mode: ((icr0 >> 11) & 1) as u8,
                    delivery_mode: ((icr0 >> 8) & 7) as u8,
                    vector: icr0 as u8,
                    trigger_mode: ((icr0 >> 15) & 1) as u8,
                    shorthand: ((icr0 >> 18) & 3) as u8,
                    icr0,
                }));
            }
            0x31 => {
                if s.is_x2apic_mode() {
                    return false;
                }
                s.icr[1] = val as u32;
            }
            0x32..=0x37 => {
                let n = (index - 0x32) as usize;
                s.lvt[n] = val as u32;
                if n == APIC_LVT_TIMER {
                    self.timer_update(s, self.clock.get_ns());
                } else if n == APIC_LVT_LINT0 && self.check_pic(s, acts) {
                    self.update_irq(s, acts);
                }
            }
            0x38 => {
                s.initial_count = val as u32;
                s.initial_count_load_time = self.clock.get_ns();
                let t = s.initial_count_load_time;
                self.timer_update(s, t);
            }
            0x3e => {
                s.divide_conf = (val as u32) & 0xb;
                let v = (s.divide_conf & 3) | ((s.divide_conf >> 1) & 4);
                s.count_shift = (v + 1) & 7;
            }
            0x3f => {
                if !s.is_x2apic_mode() {
                    return false;
                }
                // A self IPI is an IPI with the self shorthand, edge triggered and fixed.
                acts.push(Act::Ipi(Ipi {
                    dest: 0,
                    dest_mode: 0,
                    delivery_mode: APIC_DM_FIXED,
                    vector: val as u8,
                    trigger_mode: 0,
                    shorthand: 1,
                    icr0: s.icr[0],
                }));
            }
            _ => {
                s.esr |= APIC_ESR_ILLEGAL_ADDRESS;
                return false;
            }
        }
        true
    }

    /// `apic_msr_read()`: register `index` of the x2APIC MSR range, `None` for #GP.
    pub fn msr_read(&self, index: u32) -> Option<u64> {
        let mut s = self.lock();
        if !s.is_x2apic_mode() {
            return None;
        }
        self.register_read(&mut s, index)
    }

    /// `apic_msr_write()`: false for #GP.
    pub fn msr_write(&self, index: u32, val: u64) -> bool {
        self.with(|a, s, acts| {
            if !s.is_x2apic_mode() {
                return false;
            }
            a.register_write(s, index, val, acts)
        })
    }

    /// The xAPIC MMIO read of `apic_mem_read()` once the APIC is known.
    fn mem_read(&self, addr: u64) -> u64 {
        let mut s = self.lock();
        // A disabled xAPIC, or one in x2APIC mode, reads as all ones.
        if s.apicbase & MSR_IA32_APICBASE_ENABLE == 0 || s.is_x2apic_mode() {
            return 0xffff_ffff;
        }
        let index = ((addr >> 4) & 0xff) as u32;
        self.register_read(&mut s, index).unwrap_or(0)
    }

    /// The register write of `apic_mem_write()`.
    fn mem_write(&self, addr: u64, val: u64) {
        let index = ((addr >> 4) & 0xff) as u32;
        self.with(|a, s, acts| {
            a.register_write(s, index, val, acts);
        });
    }

    /// The delivery fields that `apic_get_delivery_bitmask()` looks at.
    fn dest_info(&self) -> DestInfo {
        let s = self.lock();
        (s.apicbase, s.id, s.log_dest, s.dest_mode, s.extended_log_dest)
    }
}

/// The local APICs of a machine, `local_apics[]`, with what they reach: the 8259
/// (`isa_pic`) and the IOAPICs that get EOIs.
pub struct ApicBus {
    apics: RwLock<Vec<Option<Arc<Apic>>>>,
    words: usize,
    pic: Option<Arc<I8259>>,
    ioapics: IoApics,
}

impl fmt::Debug for ApicBus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApicBus")
            .field("max_apics", &(self.words * 32))
            .field("pic", &self.pic.is_some())
            .finish_non_exhaustive()
    }
}

impl ApicBus {
    /// `apic_set_max_apic_id()`: room for APIC IDs up to `max_apic_id`, rounded up to a
    /// multiple of 32. `pic` is `isa_pic`, `ioapics` the IOAPICs that EOIs are broadcast to.
    pub fn new(max_apic_id: u32, pic: Option<Arc<I8259>>, ioapics: IoApics) -> Arc<ApicBus> {
        let max_apics = (max_apic_id.max(1) as usize).div_ceil(32) * 32;
        Arc::new(ApicBus {
            apics: RwLock::new(vec![None; max_apics]),
            words: max_apics / 32,
            pic,
            ioapics,
        })
    }

    fn insert(&self, apic: &Arc<Apic>) -> Result<(), Error> {
        let mut l = self.apics.write().unwrap_or_else(PoisonError::into_inner);
        let i = apic.initial_apic_id as usize;
        match l.get_mut(i) {
            Some(slot) if slot.is_none() => {
                *slot = Some(Arc::clone(apic));
                Ok(())
            }
            _ => Err(Error::generic(format!("APIC ID {i} is already taken or out of range"))),
        }
    }

    /// The APIC with initial ID `id`.
    pub fn apic(&self, id: u32) -> Option<Arc<Apic>> {
        let l = self.apics.read().unwrap_or_else(PoisonError::into_inner);
        l.get(id as usize).and_then(Clone::clone)
    }

    /// Every APIC, in initial ID order.
    pub fn apics(&self) -> Vec<Arc<Apic>> {
        let l = self.apics.read().unwrap_or_else(PoisonError::into_inner);
        l.iter().flatten().cloned().collect()
    }

    fn slots(&self) -> Vec<Option<Arc<Apic>>> {
        self.apics.read().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// The 8259, `isa_pic`.
    pub fn pic(&self) -> Option<&Arc<I8259>> {
        self.pic.as_ref()
    }

    /// `apic_get_delivery_bitmask()`.
    fn delivery_bitmask(&self, slots: &[Option<Arc<Apic>>], dest: u32, dest_mode: u8) -> Vec<u32> {
        let mut mask = vec![0u32; self.words];
        let infos: Vec<Option<DestInfo>> =
            slots.iter().map(|a| a.as_ref().map(|a| a.dest_info())).collect();
        let x2 = |base: u64| base & MSR_IA32_APICBASE_EXTD != 0;
        let broadcast = |mask: &mut Vec<u32>, x2apic: bool| {
            for (i, info) in infos.iter().enumerate() {
                if let Some(info) = info {
                    if x2(info.0) == x2apic {
                        set_bit(mask, i);
                    }
                }
            }
        };
        // An x2APIC broadcast goes to every x2APIC CPU whatever the destination mode, and in
        // physical mode to the xAPIC CPUs too.
        if dest == 0xffff_ffff {
            if dest_mode == APIC_DESTMODE_PHYSICAL {
                mask.fill(u32::MAX);
                return mask;
            }
            broadcast(&mut mask, true);
        }
        if dest_mode == APIC_DESTMODE_PHYSICAL {
            // apic_find_dest()
            for (i, info) in infos.iter().enumerate() {
                if let Some(&(base, id, ..)) = info.as_ref() {
                    let hit = if x2(base) {
                        slots[i].as_ref().is_some_and(|a| a.initial_apic_id == dest)
                    } else {
                        u32::from(id) == dest & 0xff
                    };
                    if hit {
                        set_bit(&mut mask, i);
                    }
                }
            }
            // Any APIC in xAPIC mode takes 0xff as a broadcast.
            if dest == 0xff {
                broadcast(&mut mask, false);
            }
        } else {
            let mut dest = dest;
            for (i, info) in infos.iter().enumerate() {
                let Some(&(base, _, log_dest, dmode, ext_log)) = info.as_ref() else { continue };
                if x2(base) {
                    // x2APIC logical mode.
                    if (dest >> 16) == (ext_log >> 16) && dest & ext_log & 0xffff != 0 {
                        set_bit(&mut mask, i);
                    }
                    continue;
                }
                // xAPIC logical mode.
                dest &= 0xff;
                let log_dest = u32::from(log_dest);
                if dmode == APIC_DESTMODE_LOGICAL_FLAT {
                    if dest & log_dest != 0 {
                        set_bit(&mut mask, i);
                    }
                } else if dmode == APIC_DESTMODE_LOGICAL_CLUSTER
                    // The high 4 bits are the cluster (0xf for all of them), the low 4 bits
                    // pick APICs in the cluster.
                    && ((dest & 0xf0) == 0xf0 || (dest & 0xf0) == (log_dest & 0xf0))
                    && dest & log_dest & 0x0f != 0
                {
                    set_bit(&mut mask, i);
                }
            }
        }
        mask
    }

    /// `apic_bus_deliver()`.
    fn bus_deliver(
        &self,
        slots: &[Option<Arc<Apic>>],
        mask: &[u32],
        delivery_mode: u8,
        vector: u8,
        trigger_mode: u8,
    ) {
        let each = |f: &dyn Fn(&Arc<Apic>)| {
            for (i, a) in slots.iter().enumerate() {
                if let Some(a) = a {
                    if get_bit(mask, i) {
                        f(a);
                    }
                }
            }
        };
        match delivery_mode {
            APIC_DM_LOWPRI => {
                // No search for the focus processor or arbitration, as in QEMU: the first
                // APIC in the mask takes it.
                let first = mask
                    .iter()
                    .enumerate()
                    .find(|(_, w)| **w != 0)
                    .map(|(i, w)| i * 32 + w.trailing_zeros() as usize);
                if let Some(Some(a)) = first.and_then(|d| slots.get(d)) {
                    a.set_irq(vector, trigger_mode);
                }
            }
            APIC_DM_FIXED | APIC_DM_EXTINT => each(&|a| a.set_irq(vector, trigger_mode)),
            APIC_DM_SMI => each(&|a| a.cpu.cpu_interrupt(CpuIrq::Smi)),
            APIC_DM_NMI => each(&|a| a.cpu.cpu_interrupt(CpuIrq::Nmi)),
            APIC_DM_INIT => each(&|a| a.cpu.cpu_interrupt(CpuIrq::Init)),
            _ => {}
        }
    }

    /// `apic_deliver_irq()`.
    pub fn deliver_irq(
        &self,
        dest: u32,
        dest_mode: u8,
        delivery_mode: u8,
        vector: u8,
        trigger_mode: u8,
    ) {
        let slots = self.slots();
        let mask = self.delivery_bitmask(&slots, dest, dest_mode);
        self.bus_deliver(&slots, &mask, delivery_mode, vector, trigger_mode);
    }

    /// `apic_send_msi()`: an MSI write of `data` to `addr`, the redirection hint ignored.
    pub fn send_msi(&self, addr: u64, data: u32) {
        // The high bits of the destination ID are in the high word of the address.
        let dest = (((addr & 0x000f_f000) >> 12) | (addr >> 32)) as u32;
        let vector = data as u8;
        let dest_mode = ((addr >> 2) & 1) as u8;
        let trigger_mode = ((data >> 15) & 1) as u8;
        let delivery = ((data >> 8) & 7) as u8;
        self.deliver_irq(dest, dest_mode, delivery, vector, trigger_mode);
    }

    /// `apic_deliver()`: an IPI written to the ICR of APIC `from`.
    fn deliver(&self, from: u32, ipi: Ipi) {
        let slots = self.slots();
        let mut mask = match ipi.shorthand {
            0 => self.delivery_bitmask(&slots, ipi.dest, ipi.dest_mode),
            1 => {
                // Self and all-but-self index the slots by initial APIC ID, which never
                // changes, rather than matching the xAPIC ID.
                let mut m = vec![0; self.words];
                set_bit(&mut m, from as usize);
                m
            }
            _ => vec![u32::MAX; self.words],
        };
        if ipi.shorthand == 3 {
            reset_bit(&mut mask, from as usize);
        }
        let each = |f: &dyn Fn(&Arc<Apic>)| {
            for (i, a) in slots.iter().enumerate() {
                if let Some(a) = a {
                    if get_bit(&mask, i) {
                        f(a);
                    }
                }
            }
        };
        match ipi.delivery_mode {
            APIC_DM_INIT => {
                let trig_mode = (ipi.icr0 >> 15) & 1;
                let level = (ipi.icr0 >> 14) & 1;
                if level == 0 && trig_mode == 1 {
                    each(&|a| {
                        let mut s = a.lock();
                        s.arb_id = s.id;
                    });
                    return;
                }
            }
            APIC_DM_SIPI => {
                // apic_startup()
                each(&|a| {
                    a.lock().sipi_vector = ipi.vector;
                    a.cpu.cpu_interrupt(CpuIrq::Sipi);
                });
                return;
            }
            _ => {}
        }
        self.bus_deliver(&slots, &mask, ipi.delivery_mode, ipi.vector, ipi.trigger_mode);
    }

    /// The `apic-msi` region, `apic_io_ops`, to map at [`APIC_DEFAULT_ADDRESS`].
    pub fn mmio(self: &Arc<Self>) -> Arc<ApicMmio> {
        Arc::new(ApicMmio { bus: Arc::downgrade(self) })
    }
}

thread_local! {
    static CURRENT_APIC: RefCell<Option<Arc<Apic>>> = const { RefCell::new(None) };
}

/// Make `apic` the APIC of the vCPU running on this thread, what `cpu_get_current_apic()`
/// finds through `current_cpu`. The vCPU loop sets it when it starts running a vCPU and
/// clears it when it stops.
pub fn set_current_apic(apic: Option<Arc<Apic>>) {
    CURRENT_APIC.with(|c| *c.borrow_mut() = apic);
}

/// `cpu_get_current_apic()`.
pub fn current_apic() -> Option<Arc<Apic>> {
    CURRENT_APIC.with(|c| c.borrow().clone())
}

/// The `apic-msi` region: the registers of the current vCPU's APIC, with writes outside of
/// them taken as MSIs.
#[derive(Debug)]
pub struct ApicMmio {
    bus: Weak<ApicBus>,
}

impl MmioOps for ApicMmio {
    /// `apic_mem_read()`.
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        if size.bytes() < 4 {
            return Ok(0);
        }
        Ok(match current_apic() {
            None => u64::MAX,
            Some(s) => s.mem_read(offset),
        })
    }

    /// `apic_mem_write()`.
    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let index = (offset >> 4) & 0xff;
        if size.bytes() < 4 {
            return Ok(());
        }
        if offset > 0xfff || index == 0 {
            // MSIs and the APIC registers share the addresses, though MSIs are on the PCI
            // bus and the APIC is wired to the CPU; the MSI registers are reserved in the APIC
            // window and the other way round, so mapping both here works.
            if let Some(bus) = self.bus.upgrade() {
                bus.send_msi(offset, value as u32);
            }
            return Ok(());
        }
        if let Some(s) = current_apic() {
            s.mem_write(offset, value);
        }
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }
}
