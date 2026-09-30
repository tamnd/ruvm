// SPDX-License-Identifier: GPL-2.0-or-later

//! The fixed ACPI hardware from hw/acpi/core.c and include/hw/acpi/acpi.h: the PM1 event and
//! control blocks, the PM timer and a GPE block.
//!
//! [`AcpiRegs`] is `ACPIREGS`, plain data that takes the current virtual time as an argument
//! where the C code reads the clock. [`AcpiPm`] wraps it with the virtual [`Clock`], the PM
//! timer's [`Timer`], the SCI output and the handler for shutdown, suspend and wakeup requests.
//! It is what a chipset (ICH9 here, PIIX4 later) embeds. Its `update_sci` is `acpi_update_sci()`,
//! which is what both ICH9 and PIIX4 plug into `ar->tmr.update_sci` and `ar->pm1.evt.update_sci`.
//!
//! Where QEMU calls `qemu_system_shutdown_request()`, `qemu_system_suspend_request()` or
//! `qemu_system_wakeup_request()`, the device sends a [`SystemRequest`] to the handler set with
//! [`AcpiPm::set_request_handler`]. The handler runs with no device lock held.
//!
//! QEMU's PM timer is always 24 bits wide. [`PmTimerWidth::Bits32`] is an extension for boards
//! that set `TMR_VAL_EXT` in the FADT; the overflow status then follows bit 31.
//!
//! The global wakeup enable mask (`qemu_system_wakeup_enable()`) is not kept here. The runstate
//! code asks [`AcpiPm::wakeup_enabled`] instead, which reads the same bits of PM1_EN. The
//! `etc/system-states` fw_cfg file is built by [`system_states`] for the board to add.
//!
//! VMState, trace points and QOM registration are not ported.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use ruvm_hw_core::timer::{NANOSECONDS_PER_SECOND, muldiv64};
use ruvm_hw_core::{Clock, IrqPin, Timer};
use ruvm_mem::{AccessConstraints, AccessCtx, AccessSize, MemResult, MmioOps};

/// `PM_TIMER_FREQUENCY`, in Hz.
pub const PM_TIMER_FREQUENCY: u32 = 3_579_545;

/// `ACPI_GPE_REGISTER_WIDTH`.
pub const ACPI_GPE_REGISTER_WIDTH: u32 = 8;
/// `ACPI_PM1_REGISTER_WIDTH`.
pub const ACPI_PM1_REGISTER_WIDTH: u32 = 16;
/// `ACPI_PM_TIMER_WIDTH`.
pub const ACPI_PM_TIMER_WIDTH: u32 = 32;

/// Size of the PM1 event block (status then enable).
pub const ACPI_PM1_EVT_LEN: u64 = 4;
/// Size of the PM1 control block.
pub const ACPI_PM1_CNT_LEN: u64 = 2;
/// Size of the PM timer block.
pub const ACPI_PM_TMR_LEN: u64 = 4;
/// Where `acpi_pm1_evt_init()` puts the event block in the PM I/O window.
pub const ACPI_PM1_EVT_OFFSET: u64 = 0;
/// Where `acpi_pm1_cnt_init()` puts the control block.
pub const ACPI_PM1_CNT_OFFSET: u64 = 4;
/// Where `acpi_pm_tmr_init()` puts the timer.
pub const ACPI_PM_TMR_OFFSET: u64 = 8;

// PM1x_STS.
pub const ACPI_BITMASK_TIMER_STATUS: u16 = 0x0001;
pub const ACPI_BITMASK_BUS_MASTER_STATUS: u16 = 0x0010;
pub const ACPI_BITMASK_GLOBAL_LOCK_STATUS: u16 = 0x0020;
pub const ACPI_BITMASK_POWER_BUTTON_STATUS: u16 = 0x0100;
pub const ACPI_BITMASK_SLEEP_BUTTON_STATUS: u16 = 0x0200;
pub const ACPI_BITMASK_RT_CLOCK_STATUS: u16 = 0x0400;
pub const ACPI_BITMASK_PCIEXP_WAKE_STATUS: u16 = 0x4000;
pub const ACPI_BITMASK_WAKE_STATUS: u16 = 0x8000;

pub const ACPI_BITMASK_ALL_FIXED_STATUS: u16 = ACPI_BITMASK_TIMER_STATUS
    | ACPI_BITMASK_BUS_MASTER_STATUS
    | ACPI_BITMASK_GLOBAL_LOCK_STATUS
    | ACPI_BITMASK_POWER_BUTTON_STATUS
    | ACPI_BITMASK_SLEEP_BUTTON_STATUS
    | ACPI_BITMASK_RT_CLOCK_STATUS
    | ACPI_BITMASK_WAKE_STATUS;

// PM1x_EN.
pub const ACPI_BITMASK_TIMER_ENABLE: u16 = 0x0001;
pub const ACPI_BITMASK_GLOBAL_LOCK_ENABLE: u16 = 0x0020;
pub const ACPI_BITMASK_POWER_BUTTON_ENABLE: u16 = 0x0100;
pub const ACPI_BITMASK_SLEEP_BUTTON_ENABLE: u16 = 0x0200;
pub const ACPI_BITMASK_RT_CLOCK_ENABLE: u16 = 0x0400;
pub const ACPI_BITMASK_PCIEXP_WAKE_DISABLE: u16 = 0x4000;

/// The enable bits that raise an SCI.
pub const ACPI_BITMASK_PM1_COMMON_ENABLED: u16 = ACPI_BITMASK_RT_CLOCK_ENABLE
    | ACPI_BITMASK_POWER_BUTTON_ENABLE
    | ACPI_BITMASK_GLOBAL_LOCK_ENABLE
    | ACPI_BITMASK_TIMER_ENABLE;

// PM1x_CNT.
pub const ACPI_BITMASK_SCI_ENABLE: u16 = 0x0001;
pub const ACPI_BITMASK_BUS_MASTER_RLD: u16 = 0x0002;
pub const ACPI_BITMASK_GLOBAL_LOCK_RELEASE: u16 = 0x0004;
pub const ACPI_BITMASK_SLEEP_TYPE: u16 = 0x1c00;
pub const ACPI_BITMASK_SLEEP_ENABLE: u16 = 0x2000;

/// `AcpiEventStatusBits`, what `acpi_send_event()` takes.
pub const ACPI_PCI_HOTPLUG_STATUS: u32 = 2;
pub const ACPI_CPU_HOTPLUG_STATUS: u32 = 4;
pub const ACPI_MEMORY_HOTPLUG_STATUS: u32 = 8;
pub const ACPI_NVDIMM_HOTPLUG_STATUS: u32 = 16;
pub const ACPI_VMGENID_CHANGE_STATUS: u32 = 32;
pub const ACPI_POWER_DOWN_STATUS: u32 = 64;
pub const ACPI_GENERIC_ERROR: u32 = 128;

/// The default `s4_val` property.
pub const ACPI_DEFAULT_S4_VAL: u8 = 2;

/// `WakeupReason`, less `QEMU_WAKEUP_REASON_NONE`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WakeupReason {
    Rtc,
    PmTimer,
    Other,
}

/// What a device asks of the machine. The handler maps these onto the runstate calls.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SystemRequest {
    /// `qemu_system_shutdown_request(SHUTDOWN_CAUSE_GUEST_SHUTDOWN)`, S5.
    Shutdown,
    /// `qapi_event_send_suspend_disk()` followed by a guest shutdown, S4.
    SuspendDisk,
    /// `qemu_system_suspend_request()`, S3.
    Suspend,
    /// `qemu_system_reset_request(SHUTDOWN_CAUSE_GUEST_RESET)`.
    Reset,
    /// `qemu_system_wakeup_request()`. The handler ignores it unless the machine is suspended
    /// and the reason is enabled.
    Wakeup(WakeupReason),
}

/// Receives [`SystemRequest`]s.
pub type SystemRequestHandler = Arc<dyn Fn(SystemRequest) + Send + Sync>;

/// How many bits of the PM timer the guest sees.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum PmTimerWidth {
    /// What QEMU implements.
    #[default]
    Bits24,
    Bits32,
}

impl PmTimerWidth {
    pub const fn bits(self) -> u32 {
        match self {
            PmTimerWidth::Bits24 => 24,
            PmTimerWidth::Bits32 => 32,
        }
    }
}

/// Setup for [`AcpiPm::new`], the arguments of `acpi_pm1_cnt_init()` and `acpi_gpe_init()`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AcpiPmConfig {
    /// Total GPE block length in bytes: the first half is status, the second enable.
    pub gpe_len: u8,
    pub tmr_width: PmTimerWidth,
    /// The SLP_TYP value that means S4.
    pub s4_val: u8,
    /// No SMM to switch modes: SCI_EN is always set and cannot be changed.
    pub acpi_only: bool,
}

impl Default for AcpiPmConfig {
    fn default() -> Self {
        AcpiPmConfig {
            gpe_len: 4,
            tmr_width: PmTimerWidth::Bits24,
            s4_val: ACPI_DEFAULT_S4_VAL,
            acpi_only: false,
        }
    }
}

/// The `etc/system-states` fw_cfg file `acpi_pm1_cnt_init()` adds: which sleep states exist and
/// their SLP_TYP values, bit 7 meaning enabled.
pub fn system_states(disable_s3: bool, disable_s4: bool, s4_val: u8) -> [u8; 6] {
    let mut suspend = [128, 0, 0, 129, 128, 128];
    suspend[3] = 1 | (u8::from(!disable_s3) << 7);
    suspend[4] = s4_val | (u8::from(!disable_s4) << 7);
    suspend
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// `acpi_pm_tmr_get_clock()`: the PM timer tick count at `now_ns`.
pub fn pm_tmr_ticks(now_ns: i64) -> i64 {
    muldiv64(now_ns.max(0) as u64, PM_TIMER_FREQUENCY, NANOSECONDS_PER_SECOND as u32) as i64
}

/// Tick `ticks` in virtual nanoseconds.
fn ticks_to_ns(ticks: i64) -> i64 {
    muldiv64(ticks.max(0) as u64, NANOSECONDS_PER_SECOND as u32, PM_TIMER_FREQUENCY) as i64
}

/// `ACPIREGS`: PM1 event and control, the timer overflow point and the GPE block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcpiRegs {
    pub pm1_sts: u16,
    pub pm1_en: u16,
    pub pm1_cnt: u16,
    pub s4_val: u8,
    pub acpi_only: bool,
    /// In PM timer ticks: when the timer's top bit next flips.
    pub overflow_time: i64,
    pub tmr_width: PmTimerWidth,
    pub gpe_len: u8,
    pub gpe_sts: Vec<u8>,
    pub gpe_en: Vec<u8>,
}

impl AcpiRegs {
    pub fn new(cfg: &AcpiPmConfig) -> Self {
        // Like acpi_gpe_init() the arrays are the full length, only the first half is used.
        let len = usize::from(cfg.gpe_len);
        AcpiRegs {
            pm1_sts: 0,
            pm1_en: 0,
            pm1_cnt: 0,
            s4_val: cfg.s4_val,
            acpi_only: cfg.acpi_only,
            overflow_time: 0,
            tmr_width: cfg.tmr_width,
            gpe_len: cfg.gpe_len,
            gpe_sts: vec![0; len],
            gpe_en: vec![0; len],
        }
    }

    /// `acpi_pm1_evt_get_sts()`. Latches TMR_STS once the virtual clock passed the overflow.
    pub fn pm1_evt_get_sts(&mut self, now_ns: i64) -> u16 {
        // Compare nanoseconds, not ticks, because the overflow timer is armed in nanoseconds.
        if now_ns >= ticks_to_ns(self.overflow_time) {
            self.pm1_sts |= ACPI_BITMASK_TIMER_STATUS;
        }
        self.pm1_sts
    }

    /// `acpi_pm1_evt_write_sts()`: write one to clear.
    pub fn pm1_evt_write_sts(&mut self, now_ns: i64, val: u16) {
        let sts = self.pm1_evt_get_sts(now_ns);
        if sts & val & ACPI_BITMASK_TIMER_STATUS != 0 {
            // Clearing TMR_STS starts a new overflow period.
            self.pm_tmr_calc_overflow_time(now_ns);
        }
        self.pm1_sts &= !val;
    }

    /// `acpi_pm1_evt_write_en()`.
    pub fn pm1_evt_write_en(&mut self, val: u16) {
        self.pm1_en = val;
    }

    /// `acpi_pm1_evt_reset()`.
    pub fn pm1_evt_reset(&mut self) {
        self.pm1_sts = 0;
        self.pm1_en = 0;
    }

    /// `acpi_pm_tmr_calc_overflow_time()`: the next point the timer's top bit flips.
    pub fn pm_tmr_calc_overflow_time(&mut self, now_ns: i64) {
        let d = pm_tmr_ticks(now_ns);
        let half = 1i64 << (self.tmr_width.bits() - 1);
        self.overflow_time = (d + half) & !(half - 1);
    }

    /// When the overflow timer should fire, in virtual nanoseconds.
    pub fn overflow_time_ns(&self) -> i64 {
        ticks_to_ns(self.overflow_time)
    }

    /// `acpi_pm_tmr_get()`.
    pub fn pm_tmr_get(&self, now_ns: i64) -> u32 {
        let d = pm_tmr_ticks(now_ns) as u32;
        match self.tmr_width {
            PmTimerWidth::Bits24 => d & 0xff_ffff,
            PmTimerWidth::Bits32 => d,
        }
    }

    /// `acpi_pm_tmr_reset()`, less the `timer_del()`.
    pub fn pm_tmr_reset(&mut self) {
        self.overflow_time = 0;
    }

    /// `acpi_pm_cnt_read()`.
    pub fn pm1_cnt_read(&self, addr: u64) -> u64 {
        u64::from(self.pm1_cnt) >> (addr * 8)
    }

    /// `acpi_pm_cnt_write()`: returns the request SLP_EN makes, if any.
    pub fn pm1_cnt_write(&mut self, addr: u64, val: u64) -> Option<SystemRequest> {
        let mut val = val as u16;
        if addr == 1 {
            val = (val << 8) | (self.pm1_cnt & 0xff);
        }
        self.pm1_cnt = val & !ACPI_BITMASK_SLEEP_ENABLE;

        if val & ACPI_BITMASK_SLEEP_ENABLE == 0 {
            return None;
        }
        let sus_typ = ((val >> 10) & 7) as u8;
        match sus_typ {
            // Soft power off.
            0 => Some(SystemRequest::Shutdown),
            1 => Some(SystemRequest::Suspend),
            t if t == self.s4_val => Some(SystemRequest::SuspendDisk),
            _ => None,
        }
    }

    /// `acpi_pm1_cnt_update()`: the APM enable and disable commands (ACPI 3.0, 4.7.2.5).
    pub fn pm1_cnt_update(&mut self, sci_enable: bool, sci_disable: bool) {
        if self.acpi_only {
            return;
        }
        if sci_enable {
            self.pm1_cnt |= ACPI_BITMASK_SCI_ENABLE;
        } else if sci_disable {
            self.pm1_cnt &= !ACPI_BITMASK_SCI_ENABLE;
        }
    }

    /// `acpi_pm1_cnt_reset()`.
    pub fn pm1_cnt_reset(&mut self) {
        self.pm1_cnt = 0;
        if self.acpi_only {
            self.pm1_cnt |= ACPI_BITMASK_SCI_ENABLE;
        }
    }

    /// `acpi_gpe_reset()`.
    pub fn gpe_reset(&mut self) {
        let half = usize::from(self.gpe_len / 2);
        self.gpe_sts[..half].fill(0);
        self.gpe_en[..half].fill(0);
    }

    /// `acpi_gpe_ioport_readb()`. QEMU aborts past the block; this reads 0.
    pub fn gpe_readb(&self, addr: u32) -> u8 {
        let (half, len) = (u32::from(self.gpe_len / 2), u32::from(self.gpe_len));
        if addr < half {
            self.gpe_sts[addr as usize]
        } else if addr < len {
            self.gpe_en[(addr - half) as usize]
        } else {
            0
        }
    }

    /// `acpi_gpe_ioport_writeb()`: status is write one to clear, enable is a plain write.
    pub fn gpe_writeb(&mut self, addr: u32, val: u8) {
        let (half, len) = (u32::from(self.gpe_len / 2), u32::from(self.gpe_len));
        if addr < half {
            self.gpe_sts[addr as usize] &= !val;
        } else if addr < len {
            self.gpe_en[(addr - half) as usize] = val;
        }
    }

    /// The level `acpi_update_sci()` puts on the SCI, and whether the overflow timer should be
    /// armed afterwards.
    pub fn sci_level(&mut self, now_ns: i64) -> (bool, bool) {
        let sts = self.pm1_evt_get_sts(now_ns);
        let gpe = match (self.gpe_sts.first(), self.gpe_en.first()) {
            (Some(s), Some(e)) => s & e != 0,
            _ => false,
        };
        let level = (sts & self.pm1_en & ACPI_BITMASK_PM1_COMMON_ENABLED) != 0 || gpe;
        let arm =
            self.pm1_en & ACPI_BITMASK_TIMER_ENABLE != 0 && sts & ACPI_BITMASK_TIMER_STATUS == 0;
        (level, arm)
    }

    /// `acpi_notify_wakeup()`: the status bits a resume sets.
    pub fn notify_wakeup(&mut self, reason: WakeupReason) {
        self.pm1_sts |= ACPI_BITMASK_WAKE_STATUS
            | match reason {
                WakeupReason::Rtc => ACPI_BITMASK_RT_CLOCK_STATUS,
                WakeupReason::PmTimer => ACPI_BITMASK_TIMER_STATUS,
                // Pretend the power button woke us.
                WakeupReason::Other => ACPI_BITMASK_POWER_BUTTON_STATUS,
            };
    }
}

/// The ACPI PM registers of one chipset with their clock, overflow timer and SCI.
pub struct AcpiPm {
    clock: Arc<Clock>,
    regs: Mutex<AcpiRegs>,
    timer: Timer,
    sci: IrqPin,
    handler: Mutex<Option<SystemRequestHandler>>,
}

impl fmt::Debug for AcpiPm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AcpiPm").field("regs", &*lock(&self.regs)).finish_non_exhaustive()
    }
}

impl AcpiPm {
    /// `acpi_pm_tmr_init()`, `acpi_pm1_evt_init()`, `acpi_pm1_cnt_init()` and `acpi_gpe_init()`.
    /// `clock` is the virtual clock. The registers start in their reset state.
    pub fn new(clock: Arc<Clock>, cfg: AcpiPmConfig) -> Arc<Self> {
        let pm = Arc::new_cyclic(|weak: &Weak<AcpiPm>| {
            let w = weak.clone();
            let timer = clock.new_timer(move || {
                if let Some(pm) = w.upgrade() {
                    pm.tmr_timer();
                }
            });
            AcpiPm {
                clock: clock.clone(),
                regs: Mutex::new(AcpiRegs::new(&cfg)),
                timer,
                sci: IrqPin::new(),
                handler: Mutex::new(None),
            }
        });
        pm.reset();
        pm
    }

    /// The SCI output.
    pub fn sci(&self) -> &IrqPin {
        &self.sci
    }

    /// Where shutdown, suspend and wakeup requests go.
    pub fn set_request_handler(&self, handler: SystemRequestHandler) {
        *lock(&self.handler) = Some(handler);
    }

    fn request(&self, req: SystemRequest) {
        let h = lock(&self.handler).clone();
        if let Some(h) = h {
            h(req);
        }
    }

    fn now(&self) -> i64 {
        self.clock.get_ns()
    }

    /// A copy of the registers.
    pub fn regs(&self) -> AcpiRegs {
        lock(&self.regs).clone()
    }

    /// Runs `f` on the registers, then updates the SCI.
    pub fn with_regs<R>(&self, f: impl FnOnce(&mut AcpiRegs) -> R) -> R {
        let r = f(&mut lock(&self.regs));
        self.update_sci();
        r
    }

    /// The overflow timer's deadline, if armed.
    pub fn overflow_timer_deadline(&self) -> Option<i64> {
        self.timer.expire_time()
    }

    /// `acpi_update_sci()`.
    pub fn update_sci(&self) {
        let now = self.now();
        let level = {
            let mut r = lock(&self.regs);
            let (level, arm) = r.sci_level(now);
            // acpi_pm_tmr_update(): schedule a timer interrupt if needed.
            if arm {
                self.timer.modify(r.overflow_time_ns());
            } else {
                self.timer.del();
            }
            level
        };
        self.sci.set_bool(level);
    }

    /// `acpi_pm_tmr_timer()`.
    fn tmr_timer(&self) {
        self.request(SystemRequest::Wakeup(WakeupReason::PmTimer));
        self.update_sci();
    }

    /// `acpi_pm_evt_read()`.
    pub fn pm1_evt_read(&self, addr: u64) -> u16 {
        let now = self.now();
        let mut r = lock(&self.regs);
        match addr {
            0 => r.pm1_evt_get_sts(now),
            2 => r.pm1_en,
            _ => 0,
        }
    }

    /// `acpi_pm_evt_write()`.
    pub fn pm1_evt_write(&self, addr: u64, val: u16) {
        let now = self.now();
        {
            let mut r = lock(&self.regs);
            match addr {
                0 => r.pm1_evt_write_sts(now, val),
                2 => r.pm1_evt_write_en(val),
                _ => return,
            }
        }
        self.update_sci();
    }

    /// `acpi_pm_cnt_read()`.
    pub fn pm1_cnt_read(&self, addr: u64) -> u64 {
        lock(&self.regs).pm1_cnt_read(addr)
    }

    /// `acpi_pm_cnt_write()`. For S4 QEMU sends SUSPEND_DISK and then shuts down; here the
    /// handler gets [`SystemRequest::SuspendDisk`] and does both.
    pub fn pm1_cnt_write(&self, addr: u64, val: u64) {
        let req = lock(&self.regs).pm1_cnt_write(addr, val);
        if let Some(req) = req {
            self.request(req);
        }
    }

    /// `acpi_pm1_cnt_update()`.
    pub fn pm1_cnt_update(&self, sci_enable: bool, sci_disable: bool) {
        lock(&self.regs).pm1_cnt_update(sci_enable, sci_disable);
    }

    /// `acpi_pm_tmr_read()`.
    pub fn pm_tmr_read(&self) -> u32 {
        let now = self.now();
        lock(&self.regs).pm_tmr_get(now)
    }

    /// `acpi_gpe_ioport_readb()`.
    pub fn gpe_readb(&self, addr: u32) -> u8 {
        lock(&self.regs).gpe_readb(addr)
    }

    /// `acpi_gpe_ioport_writeb()` followed by `acpi_update_sci()`, as the chipsets do it.
    pub fn gpe_writeb(&self, addr: u32, val: u8) {
        lock(&self.regs).gpe_writeb(addr, val);
        self.update_sci();
    }

    /// `acpi_send_gpe_event()`: sets bits in the first GPE status byte.
    pub fn send_gpe_event(&self, status: u8) {
        {
            let mut r = lock(&self.regs);
            if let Some(s) = r.gpe_sts.first_mut() {
                *s |= status;
            }
        }
        self.update_sci();
    }

    /// `acpi_pm1_evt_power_down()`: the power button, if the guest enabled it.
    pub fn power_down(&self) {
        let raise = {
            let mut r = lock(&self.regs);
            let en = r.pm1_en & ACPI_BITMASK_POWER_BUTTON_ENABLE != 0;
            if en {
                r.pm1_sts |= ACPI_BITMASK_POWER_BUTTON_STATUS;
            }
            en
        };
        if raise {
            self.update_sci();
        }
    }

    /// `acpi_notify_wakeup()`, called by the runstate code on resume.
    pub fn notify_wakeup(&self, reason: WakeupReason) {
        lock(&self.regs).notify_wakeup(reason);
    }

    /// Whether PM1_EN lets `reason` wake the machine, what `acpi_pm1_evt_write_en()` feeds to
    /// `qemu_system_wakeup_enable()`.
    pub fn wakeup_enabled(&self, reason: WakeupReason) -> bool {
        let en = lock(&self.regs).pm1_en;
        match reason {
            WakeupReason::Rtc => en & ACPI_BITMASK_RT_CLOCK_ENABLE != 0,
            WakeupReason::PmTimer => en & ACPI_BITMASK_TIMER_ENABLE != 0,
            WakeupReason::Other => true,
        }
    }

    /// `acpi_pm1_evt_reset()`, `acpi_pm1_cnt_reset()`, `acpi_pm_tmr_reset()` and
    /// `acpi_gpe_reset()`. The SCI is left alone; the chipset updates it.
    pub fn reset(&self) {
        {
            let mut r = lock(&self.regs);
            r.pm1_evt_reset();
            r.pm1_cnt_reset();
            r.pm_tmr_reset();
            r.gpe_reset();
        }
        self.timer.del();
    }

    /// The `acpi-evt` region, [`ACPI_PM1_EVT_LEN`] bytes.
    pub fn evt_ops(self: &Arc<Self>) -> Arc<AcpiPm1EvtOps> {
        Arc::new(AcpiPm1EvtOps(self.clone()))
    }

    /// The `acpi-cnt` region, [`ACPI_PM1_CNT_LEN`] bytes.
    pub fn cnt_ops(self: &Arc<Self>) -> Arc<AcpiPm1CntOps> {
        Arc::new(AcpiPm1CntOps(self.clone()))
    }

    /// The `acpi-tmr` region, [`ACPI_PM_TMR_LEN`] bytes.
    pub fn tmr_ops(self: &Arc<Self>) -> Arc<AcpiPmTmrOps> {
        Arc::new(AcpiPmTmrOps(self.clone()))
    }

    /// A GPE block region the size of the configured GPE length, with byte callbacks.
    pub fn gpe_ops(self: &Arc<Self>) -> Arc<AcpiGpeOps> {
        Arc::new(AcpiGpeOps(self.clone()))
    }

    /// The configured GPE block length.
    pub fn gpe_len(&self) -> u8 {
        lock(&self.regs).gpe_len
    }
}

/// `acpi_pm_evt_ops`.
#[derive(Debug)]
pub struct AcpiPm1EvtOps(pub Arc<AcpiPm>);

impl MmioOps for AcpiPm1EvtOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.pm1_evt_read(offset)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.pm1_evt_write(offset, value as u16);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 2)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(2, 4)
    }
}

/// `acpi_pm_cnt_ops`.
#[derive(Debug)]
pub struct AcpiPm1CntOps(pub Arc<AcpiPm>);

impl MmioOps for AcpiPm1CntOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(self.0.pm1_cnt_read(offset))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.pm1_cnt_write(offset, value);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 2)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(2, 4)
    }
}

/// `acpi_pm_tmr_ops`. Writes are ignored.
#[derive(Debug)]
pub struct AcpiPmTmrOps(pub Arc<AcpiPm>);

impl MmioOps for AcpiPmTmrOps {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.pm_tmr_read()))
    }

    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(4, 4)
    }
}

/// `ich9_gpe_ops`, which any chipset with a byte wide GPE block can use.
#[derive(Debug)]
pub struct AcpiGpeOps(pub Arc<AcpiPm>);

impl MmioOps for AcpiGpeOps {
    fn read(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::from(self.0.gpe_readb(offset as u32)))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, _size: AccessSize, value: u64) -> MemResult<()> {
        self.0.gpe_writeb(offset as u32, value as u8);
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 1)
    }
}
