// SPDX-License-Identifier: GPL-2.0-or-later

//! The SBSA generic watchdog, hw/watchdog/sbsa_gwdt.c and include/hw/watchdog/sbsa_gwdt.h.
//!
//! [`SbsaGwdt`] is `SBSA_GWDTState`. It has two 0x1000 byte frames: the refresh frame
//! ([`SbsaGwdt::refresh_ops`]), where any write to WRR refreshes the watchdog, and the control
//! frame ([`SbsaGwdt::control_ops`]) with the enable bit, the 48 bit offset register and the
//! compare value. Both take aligned 4 byte accesses only and are little endian.
//!
//! The timeout runs on the virtual clock at the `clock-frequency` rate. When it first expires
//! the device sets WS0, raises its interrupt and starts a second period. If that one expires
//! too, it sets WS1 and performs the watchdog action, which by default resets the machine.
//!
//! Differences from QEMU:
//!
//! - Not ported: VMState, trace points and QOM registration.
//! - The `qemu_log_mask()` guest error messages are not printed, since the workspace has no
//!   `-d` log yet.
//! - `get_watchdog_action()` and `watchdog_perform_action()` are a [`WatchdogAction`] the board
//!   sets and a handler it gives [`SbsaGwdt::new`], since the workspace has no global
//!   `-action watchdog=` state yet. The default action is a reset, as in QEMU.
//! - The board calls [`SbsaGwdt::reset`] for the legacy reset handler.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};

use ruvm_hw_core::irq::IrqPin;
use ruvm_hw_core::timer::{Clock, NANOSECONDS_PER_SECOND, Timer, muldiv64};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `TYPE_WDT_SBSA`.
pub const TYPE_WDT_SBSA: &str = "sbsa-gwdt";

/// `SBSA_GWDT_WRR`: the watchdog refresh register in the refresh frame.
pub const SBSA_GWDT_WRR: u64 = 0x000;
/// `SBSA_GWDT_WCS`: the control and status register.
pub const SBSA_GWDT_WCS: u64 = 0x000;
/// `SBSA_GWDT_WOR`: the low 32 bits of the offset register.
pub const SBSA_GWDT_WOR: u64 = 0x008;
/// `SBSA_GWDT_WORU`: the high 16 bits of the offset register.
pub const SBSA_GWDT_WORU: u64 = 0x00c;
/// `SBSA_GWDT_WCV`: the low 32 bits of the compare value.
pub const SBSA_GWDT_WCV: u64 = 0x010;
/// `SBSA_GWDT_WCVU`: the high 32 bits of the compare value.
pub const SBSA_GWDT_WCVU: u64 = 0x014;
/// `SBSA_GWDT_W_IIDR`: the interface identification register, in both frames.
pub const SBSA_GWDT_W_IIDR: u64 = 0xfcc;

/// `SBSA_GWDT_WCS_EN`.
pub const SBSA_GWDT_WCS_EN: u32 = 1 << 0;
/// `SBSA_GWDT_WCS_WS0`.
pub const SBSA_GWDT_WCS_WS0: u32 = 1 << 1;
/// `SBSA_GWDT_WCS_WS1`.
pub const SBSA_GWDT_WCS_WS1: u32 = 1 << 2;
/// `SBSA_GWDT_WOR_MASK`.
pub const SBSA_GWDT_WOR_MASK: u32 = 0x0000_ffff;
/// `SBSA_GWDT_ID`: Arm as the implementer, architecture version 1.
pub const SBSA_GWDT_ID: u32 = 0x1043b;

/// `SBSA_GWDT_RMMIO_SIZE`.
pub const SBSA_GWDT_RMMIO_SIZE: u64 = 0x1000;
/// `SBSA_GWDT_CMMIO_SIZE`.
pub const SBSA_GWDT_CMMIO_SIZE: u64 = 0x1000;

/// `WatchdogAction` from qapi/run-state.json: what happens when a watchdog expires.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum WatchdogAction {
    /// `reset`, the default.
    #[default]
    Reset,
    /// `shutdown`.
    Shutdown,
    /// `poweroff`.
    Poweroff,
    /// `pause`.
    Pause,
    /// `debug`.
    Debug,
    /// `none`.
    None,
    /// `inject-nmi`.
    InjectNmi,
}

/// The `sbsa-gwdt` properties.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SbsaGwdtProps {
    /// `clock-frequency`: the timer rate in Hz, which must match the generic timer's.
    pub clock_frequency: u64,
    /// `wdat`: run at the 1 kHz the ACPI WDAT table needs, overriding `clock-frequency`.
    pub wdat: bool,
}

impl Default for SbsaGwdtProps {
    fn default() -> Self {
        SbsaGwdtProps { clock_frequency: NANOSECONDS_PER_SECOND as u64, wdat: false }
    }
}

/// `watchdog_perform_action()`, called with the action when the second period expires.
pub type WatchdogHandler = Arc<dyn Fn(WatchdogAction) + Send + Sync>;

/// `WdtRefreshType`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RefreshType {
    Explicit,
    Timeout,
}

#[derive(Debug, Default)]
struct GwdtState {
    wcs: u32,
    worl: u32,
    woru: u32,
    wcvl: u32,
    wcvu: u32,
    id: u32,
    action: WatchdogAction,
}

/// `SBSA_GWDTState`, the `sbsa-gwdt` device.
pub struct SbsaGwdt {
    state: Mutex<GwdtState>,
    freq: u64,
    irq: IrqPin,
    clock: Arc<Clock>,
    timer: Timer,
    handler: WatchdogHandler,
}

impl fmt::Debug for SbsaGwdt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SbsaGwdt")
            .field("state", &*self.lock())
            .field("freq", &self.freq)
            .field("irq", &self.irq)
            .finish_non_exhaustive()
    }
}

impl SbsaGwdt {
    /// `wdt_sbsa_gwdt_realize()` and a reset. `clock` is the virtual clock, and `handler` is
    /// what `watchdog_perform_action()` does.
    pub fn new(clock: Arc<Clock>, props: SbsaGwdtProps, handler: WatchdogHandler) -> Arc<Self> {
        // WDAT spec: "The clock interval that the WDT uses must be greater than or equal to
        // 1 millisecond."
        let freq = if props.wdat { 1000 } else { props.clock_frequency };
        let dev = Arc::new_cyclic(|weak: &Weak<SbsaGwdt>| {
            let w = weak.clone();
            let timer = clock.new_timer(move || {
                if let Some(s) = w.upgrade() {
                    s.timer_sysinterrupt();
                }
            });
            SbsaGwdt {
                state: Mutex::new(GwdtState::default()),
                freq,
                irq: IrqPin::new(),
                clock,
                timer,
                handler,
            }
        });
        dev.reset();
        dev
    }

    fn lock(&self) -> MutexGuard<'_, GwdtState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The WS0 interrupt, `s->irq`.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    /// Sets the action taken when the watchdog expires, `-action watchdog=`.
    pub fn set_action(&self, action: WatchdogAction) {
        self.lock().action = action;
    }

    /// The refresh frame, `sbsa_gwdt_rops`.
    pub fn refresh_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(RefreshFrame(Arc::clone(self)))
    }

    /// The control frame, `sbsa_gwdt_ops`.
    pub fn control_ops(self: &Arc<Self>) -> Arc<dyn MmioOps> {
        Arc::new(ControlFrame(Arc::clone(self)))
    }

    /// `sbsa_gwdt_rread()`.
    pub fn refresh_read(&self, addr: u64) -> u32 {
        match addr {
            // A read of the refresh register has no effect and returns 0.
            SBSA_GWDT_WRR => 0,
            SBSA_GWDT_W_IIDR => self.lock().id,
            // QEMU logs "bad address in refresh frame read".
            _ => 0,
        }
    }

    /// `sbsa_gwdt_read()`.
    pub fn control_read(&self, addr: u64) -> u32 {
        let s = self.lock();
        match addr {
            SBSA_GWDT_WCS => s.wcs,
            SBSA_GWDT_WOR => s.worl,
            SBSA_GWDT_WORU => s.woru,
            SBSA_GWDT_WCV => s.wcvl,
            SBSA_GWDT_WCVU => s.wcvu,
            SBSA_GWDT_W_IIDR => s.id,
            // QEMU logs "bad address in control frame read".
            _ => 0,
        }
    }

    /// `sbsa_gwdt_update_timer()`.
    fn update_timer(&self, s: &mut GwdtState, rtype: RefreshType) {
        self.timer.del();
        if s.wcs & SBSA_GWDT_WCS_EN != 0 {
            // The 48 bit offset is the low 16 bits of WORU over the 32 bits of WOR.
            let offset = (u64::from(s.woru) << 32) | u64::from(s.worl);
            // QEMU passes the 64 bit frequency to muldiv64(), which takes 32 bits.
            let timeout = muldiv64(offset, NANOSECONDS_PER_SECOND as u32, self.freq as u32)
                .wrapping_add(self.clock.get_ns() as u64);
            if rtype == RefreshType::Explicit || s.wcs & SBSA_GWDT_WCS_WS0 == 0 {
                // Store the current timeout value in the compare registers.
                s.wcvu = (timeout >> 32) as u32;
                s.wcvl = timeout as u32;
            }
            self.timer.modify(timeout as i64);
        }
    }

    /// `sbsa_gwdt_rwrite()`.
    pub fn refresh_write(&self, addr: u64, _data: u64) {
        if addr == SBSA_GWDT_WRR {
            let mut s = self.lock();
            s.wcs &= !(SBSA_GWDT_WCS_WS0 | SBSA_GWDT_WCS_WS1);
            self.update_timer(&mut s, RefreshType::Explicit);
        }
        // QEMU logs "bad address in refresh frame write" for anything else.
    }

    /// `sbsa_gwdt_write()`.
    pub fn control_write(&self, addr: u64, data: u64) {
        let mut guard = self.lock();
        let s = &mut *guard;
        match addr {
            SBSA_GWDT_WCS => {
                s.wcs = data as u32 & SBSA_GWDT_WCS_EN;
                self.irq.lower();
                self.update_timer(s, RefreshType::Explicit);
            }
            SBSA_GWDT_WOR => {
                s.worl = data as u32;
                s.wcs &= !(SBSA_GWDT_WCS_WS0 | SBSA_GWDT_WCS_WS1);
                self.irq.lower();
                self.update_timer(s, RefreshType::Explicit);
            }
            SBSA_GWDT_WORU => {
                s.woru = data as u32 & SBSA_GWDT_WOR_MASK;
                s.wcs &= !(SBSA_GWDT_WCS_WS0 | SBSA_GWDT_WCS_WS1);
                self.irq.lower();
                self.update_timer(s, RefreshType::Explicit);
            }
            SBSA_GWDT_WCV => s.wcvl = data as u32,
            SBSA_GWDT_WCVU => s.wcvu = data as u32,
            // QEMU logs "bad address in control frame write".
            _ => {}
        }
    }

    /// `wdt_sbsa_gwdt_reset()`.
    pub fn reset(&self) {
        let mut s = self.lock();
        self.reset_locked(&mut s);
    }

    fn reset_locked(&self, s: &mut GwdtState) {
        self.timer.del();
        s.wcs = 0;
        s.wcvl = 0;
        s.wcvu = 0;
        s.worl = 0;
        s.woru = 0;
        s.id = SBSA_GWDT_ID;
    }

    /// `sbsa_gwdt_timer_sysinterrupt()`.
    fn timer_sysinterrupt(&self) {
        let mut s = self.lock();
        if s.wcs & SBSA_GWDT_WCS_WS0 == 0 {
            s.wcs |= SBSA_GWDT_WCS_WS0;
            self.update_timer(&mut s, RefreshType::Timeout);
            self.irq.raise();
        } else {
            s.wcs |= SBSA_GWDT_WCS_WS1;
            let action = s.action;
            // Reset the watchdog only if the guest gets told about the expiry, and before the
            // action runs.
            match action {
                WatchdogAction::Debug | WatchdogAction::None | WatchdogAction::Pause => {}
                _ => self.reset_locked(&mut s),
            }
            drop(s);
            (self.handler)(action);
        }
    }
}

/// `sbsa_gwdt_rops`.
struct RefreshFrame(Arc<SbsaGwdt>);

impl MmioOps for RefreshFrame {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.refresh_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.refresh_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

/// `sbsa_gwdt_ops`.
struct ControlFrame(Arc<SbsaGwdt>);

impl MmioOps for ControlFrame {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.control_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.control_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::exact(4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_base::ClockType;
    use ruvm_hw_core::irq::IrqLine;
    use std::sync::atomic::{AtomicI32, Ordering};

    struct Rig {
        clock: Arc<Clock>,
        wdt: Arc<SbsaGwdt>,
        irq: Arc<AtomicI32>,
        actions: Arc<Mutex<Vec<WatchdogAction>>>,
    }

    fn rig(props: SbsaGwdtProps) -> Rig {
        let clock = Clock::manual(ClockType::Virtual);
        let actions = Arc::new(Mutex::new(Vec::new()));
        let a = actions.clone();
        let wdt = SbsaGwdt::new(clock.clone(), props, Arc::new(move |x| a.lock().unwrap().push(x)));
        let irq = Arc::new(AtomicI32::new(-1));
        let l = irq.clone();
        wdt.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        Rig { clock, wdt, irq, actions }
    }

    #[test]
    fn identification() {
        let r = rig(SbsaGwdtProps::default());
        assert_eq!(r.wdt.refresh_read(SBSA_GWDT_W_IIDR), SBSA_GWDT_ID);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_W_IIDR), SBSA_GWDT_ID);
        assert_eq!(r.wdt.refresh_read(SBSA_GWDT_WRR), 0);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), 0);
    }

    #[test]
    fn two_stage_expiry_resets() {
        let r = rig(SbsaGwdtProps::default());
        r.wdt.control_write(SBSA_GWDT_WOR, 1000);
        r.wdt.control_write(SBSA_GWDT_WORU, 0x1_0000);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WORU), 0);
        r.wdt.control_write(SBSA_GWDT_WCS, 0xff);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), SBSA_GWDT_WCS_EN);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCV), 1000);
        r.clock.advance_to(999);
        assert_eq!(r.irq.load(Ordering::SeqCst), 0);
        r.clock.advance_to(1000);
        assert_eq!(r.irq.load(Ordering::SeqCst), 1);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), SBSA_GWDT_WCS_EN | SBSA_GWDT_WCS_WS0);
        // A timeout refresh with WS0 set keeps the compare value.
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCV), 1000);
        r.clock.advance_to(2000);
        assert_eq!(*r.actions.lock().unwrap(), vec![WatchdogAction::Reset]);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), 0);
    }

    #[test]
    fn refresh_clears_status() {
        let r = rig(SbsaGwdtProps { clock_frequency: 1000, wdat: false });
        r.wdt.control_write(SBSA_GWDT_WOR, 5);
        r.wdt.control_write(SBSA_GWDT_WCS, 1);
        r.clock.advance_to(5_000_000);
        assert_eq!(r.irq.load(Ordering::SeqCst), 1);
        r.wdt.refresh_write(SBSA_GWDT_WRR, 0);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), SBSA_GWDT_WCS_EN);
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCV), 10_000_000);
        // Writing WCS lowers the interrupt.
        r.wdt.control_write(SBSA_GWDT_WCS, 1);
        assert_eq!(r.irq.load(Ordering::SeqCst), 0);
        assert!(r.actions.lock().unwrap().is_empty());
    }

    #[test]
    fn pause_action_keeps_state() {
        let r = rig(SbsaGwdtProps { clock_frequency: 0, wdat: true });
        r.wdt.set_action(WatchdogAction::Pause);
        r.wdt.control_write(SBSA_GWDT_WOR, 1);
        r.wdt.control_write(SBSA_GWDT_WCS, 1);
        r.clock.advance_to(2_000_000);
        assert_eq!(*r.actions.lock().unwrap(), vec![WatchdogAction::Pause]);
        let ws = SBSA_GWDT_WCS_EN | SBSA_GWDT_WCS_WS0 | SBSA_GWDT_WCS_WS1;
        assert_eq!(r.wdt.control_read(SBSA_GWDT_WCS), ws);
    }
}
