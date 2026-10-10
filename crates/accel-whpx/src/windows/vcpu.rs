// SPDX-License-Identifier: GPL-2.0-or-later

//! One vCPU: `whpx_init_vcpu()`, `whpx_vcpu_run()` with the exits it handles, and the
//! callbacks of the instruction emulator in WinHvEmulation.dll.
//!
//! A WHPX vCPU is not tied to a thread, but only one thread runs it at a time, which `&mut
//! self` gives. Other threads reach it through [`WhpxKick`], which also holds QEMU's
//! `interrupt_request` bits.

use std::ffi::c_void;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use ruvm_mem::{AddressSpace, MemTxAttrs};
use windows_sys::Win32::System::Hypervisor::{
    WHV_EMULATOR_CALLBACKS, WHV_EMULATOR_IO_ACCESS_INFO, WHV_EMULATOR_MEMORY_ACCESS_INFO,
    WHV_EMULATOR_STATUS, WHV_REGISTER_VALUE, WHV_RUN_VP_EXIT_CONTEXT, WHV_TRANSLATE_GVA_RESULT,
    WHvCapabilityCodeInterruptClockFrequency, WHvCapabilityCodeProcessorClockFrequency,
    WHvRegisterInternalActivityState, WHvRegisterPendingEvent, WHvRegisterPendingInterruption,
    WHvRunVpExitReasonCanceled, WHvRunVpExitReasonException, WHvRunVpExitReasonMemoryAccess,
    WHvRunVpExitReasonX64ApicEoi, WHvRunVpExitReasonX64Cpuid, WHvRunVpExitReasonX64Halt,
    WHvRunVpExitReasonX64InterruptWindow, WHvRunVpExitReasonX64IoPortAccess,
    WHvRunVpExitReasonX64MsrAccess, WHvX64RegisterApicBase, WHvX64RegisterCr8,
    WHvX64RegisterDeliverabilityNotifications, WHvX64RegisterInitialApicId, WHvX64RegisterRax,
    WHvX64RegisterRbx, WHvX64RegisterRcx, WHvX64RegisterRdx, WHvX64RegisterRflags,
    WHvX64RegisterRip,
};

use super::dispatch::{Dispatch, Hr, Reg};
use super::{Partition, WhpxAccel, frequency};
use crate::cpuid::{self, CpuidExit};
use crate::exit::{self, ExceptionInfo, IoInfo};
use crate::irq::{self, InjectState, PreRunInput};
use crate::msr::{self, MsrAction, MsrContext, MsrExit};
use crate::{
    HYPERV_APIC_BUS_FREQUENCY, USERSPACE_APIC_BUS_FREQUENCY, VcpuStop, WhpxError, apic, features,
};

/// EFLAGS.IF.
const IF_MASK: u64 = 1 << 9;

/// The parts of the machine a vCPU exit reaches: the CPU model, the userspace APIC and PIC,
/// and the IOAPIC.
pub trait WhpxExits {
    /// `cpu_x86_cpuid()`: the CPU model's answer, before the WHPX fixups.
    fn cpuid(&mut self, leaf: u32, subleaf: u32) -> cpuid::Regs;

    /// `cpu_set_apic_base()`. `Ok` carries the change to CPUID's APIC bit, as
    /// [`apic::LapicState::set_base`] reports it. An error makes the write #GP.
    fn set_apic_base(&mut self, val: u64) -> Result<Option<bool>, apic::BadApicBase>;

    /// `apic_msr_read()` of an x2APIC register of the userspace APIC. `None` is #GP.
    fn apic_msr_read(&mut self, index: u32) -> Option<u64> {
        let _ = index;
        None
    }

    /// `apic_msr_write()`. `false` is #GP.
    fn apic_msr_write(&mut self, index: u32, val: u64) -> bool {
        let _ = (index, val);
        false
    }

    /// `apic_get_highest_priority_irr()`, -1 for none.
    fn apic_irr(&mut self) -> i32 {
        -1
    }

    /// `pic_get_output()` of the ISA PIC.
    fn pic_output(&mut self) -> bool {
        false
    }

    /// `cpu_get_pic_interrupt()`, which acknowledges the interrupt. -1 for none.
    fn pic_interrupt(&mut self) -> i32 {
        -1
    }

    /// `cpu_get_apic_tpr()` of the userspace APIC.
    fn apic_tpr(&mut self) -> u8 {
        0
    }

    /// `cpu_set_apic_tpr()`: the guest wrote CR8.
    fn set_apic_tpr(&mut self, tpr: u8) {
        let _ = tpr;
    }

    /// `apic_poll_irq()`.
    fn apic_poll(&mut self) {}

    /// `apic_handle_tpr_access_report()`.
    fn tpr_access_report(&mut self, rip: u64) {
        let _ = rip;
    }

    /// `ioapic_eoi_broadcast()`: the Hyper-V LAPIC finished a level triggered interrupt.
    fn apic_eoi(&mut self, vector: u8) {
        let _ = vector;
    }
}

#[derive(Debug)]
struct Shared {
    part: Arc<Partition>,
    index: u32,
    interrupt_request: AtomicU32,
    exit_request: AtomicBool,
}

impl Shared {
    fn pending(&self) -> u32 {
        self.interrupt_request.load(Ordering::Acquire)
    }

    fn reset(&self, bits: u32) {
        if bits != 0 {
            self.interrupt_request.fetch_and(!bits, Ordering::AcqRel);
        }
    }
}

/// A handle that reaches a vCPU from any thread.
#[derive(Clone, Debug)]
pub struct WhpxKick(Arc<Shared>);

impl WhpxKick {
    /// `cpu_exit()`: the vCPU leaves [`WhpxVcpu::run`] with [`VcpuStop::Kicked`] as soon as it
    /// can.
    pub fn kick(&self) {
        self.0.exit_request.store(true, Ordering::Release);
        self.0.part.cancel_run(self.0.index);
    }

    /// `cpu_interrupt()`: raises `CPU_INTERRUPT_*` bits, [`irq::INTERRUPT_HARD`] and the
    /// others, and stops the current run so they are seen.
    pub fn interrupt(&self, bits: u32) {
        self.0.interrupt_request.fetch_or(bits, Ordering::AcqRel);
        self.0.part.cancel_run(self.0.index);
    }

    /// `cpu_reset_interrupt()`.
    pub fn reset_interrupt(&self, bits: u32) {
        self.0.reset(bits);
    }

    /// `cpu->interrupt_request`.
    pub fn interrupt_request(&self) -> u32 {
        self.0.pending()
    }
}

/// A WinHvEmulation.dll emulator. It holds no thread affinity.
#[derive(Debug)]
struct Emulator {
    handle: *mut c_void,
    d: &'static Dispatch,
}

// SAFETY: the emulator handle is an opaque object that any thread may use, one at a time,
// which the owning vCPU's `&mut self` ensures.
unsafe impl Send for Emulator {}

impl Drop for Emulator {
    fn drop(&mut self) {
        // SAFETY: the handle came from WHvEmulatorCreateEmulator() and is not used again.
        unsafe { (self.d.emulator_destroy)(self.handle) };
    }
}

/// What the emulator callbacks see: the vCPU and the address spaces.
struct EmuCtx<'a> {
    part: &'a Partition,
    index: u32,
    io: &'a AddressSpace,
    mem: &'a AddressSpace,
}

/// The `WHvEmulatorIoPortCallback`.
unsafe extern "system" fn emu_io(ctx: *const c_void, acc: *mut WHV_EMULATOR_IO_ACCESS_INFO) -> Hr {
    // SAFETY: `ctx` is the EmuCtx emulate() passed, which outlives the emulator call that
    // makes this callback, and the emulator hands over a valid access record.
    let (ctx, acc) = unsafe { (&*ctx.cast::<EmuCtx<'_>>(), &mut *acc) };
    let size = usize::from(acc.AccessSize).min(4);
    let port = u64::from(acc.Port);
    let mut buf = acc.Data.to_le_bytes();
    if acc.Direction == 0 {
        let _ = ctx.io.read(port, MemTxAttrs::UNSPECIFIED, &mut buf[..size]);
        acc.Data = u32::from_le_bytes(buf);
    } else {
        let _ = ctx.io.write(port, MemTxAttrs::UNSPECIFIED, &buf[..size]);
    }
    0
}

/// The `WHvEmulatorMemoryCallback`.
unsafe extern "system" fn emu_mem(
    ctx: *const c_void,
    acc: *mut WHV_EMULATOR_MEMORY_ACCESS_INFO,
) -> Hr {
    // SAFETY: as in emu_io().
    let (ctx, acc) = unsafe { (&*ctx.cast::<EmuCtx<'_>>(), &mut *acc) };
    let size = usize::from(acc.AccessSize).min(8);
    if acc.Direction == 0 {
        let _ = ctx.mem.read(acc.GpaAddress, MemTxAttrs::UNSPECIFIED, &mut acc.Data[..size]);
    } else {
        let _ = ctx.mem.write(acc.GpaAddress, MemTxAttrs::UNSPECIFIED, &acc.Data[..size]);
    }
    0
}

/// The `WHvEmulatorGetVirtualProcessorRegisters` callback.
unsafe extern "system" fn emu_get_regs(
    ctx: *const c_void,
    names: *const i32,
    count: u32,
    vals: *mut WHV_REGISTER_VALUE,
) -> Hr {
    // SAFETY: as in emu_io(); the arrays are the emulator's and `count` long, and `Reg` has
    // the layout of `WHV_REGISTER_VALUE`.
    unsafe {
        let ctx = &*ctx.cast::<EmuCtx<'_>>();
        (ctx.part.d().get_vp_registers)(ctx.part.handle(), ctx.index, names, count, vals.cast())
    }
}

/// The `WHvEmulatorSetVirtualProcessorRegisters` callback.
unsafe extern "system" fn emu_set_regs(
    ctx: *const c_void,
    names: *const i32,
    count: u32,
    vals: *const WHV_REGISTER_VALUE,
) -> Hr {
    // SAFETY: as in emu_get_regs().
    unsafe {
        let ctx = &*ctx.cast::<EmuCtx<'_>>();
        (ctx.part.d().set_vp_registers)(ctx.part.handle(), ctx.index, names, count, vals.cast())
    }
}

/// The `WHvEmulatorTranslateGvaPage` callback.
unsafe extern "system" fn emu_translate(
    ctx: *const c_void,
    gva: u64,
    flags: i32,
    result: *mut i32,
    gpa: *mut u64,
) -> Hr {
    // SAFETY: as in emu_io(); both out pointers are the emulator's.
    unsafe {
        let ctx = &*ctx.cast::<EmuCtx<'_>>();
        let mut res = WHV_TRANSLATE_GVA_RESULT::default();
        let hr =
            (ctx.part.d().translate_gva)(ctx.part.handle(), ctx.index, gva, flags, &mut res, gpa);
        if hr >= 0 {
            *result = res.ResultCode;
        }
        hr
    }
}

static CALLBACKS: WHV_EMULATOR_CALLBACKS = WHV_EMULATOR_CALLBACKS {
    Size: size_of::<WHV_EMULATOR_CALLBACKS>() as u32,
    Reserved: 0,
    WHvEmulatorIoPortCallback: Some(emu_io),
    WHvEmulatorMemoryCallback: Some(emu_mem),
    WHvEmulatorGetVirtualProcessorRegisters: Some(emu_get_regs),
    WHvEmulatorSetVirtualProcessorRegisters: Some(emu_set_regs),
    WHvEmulatorTranslateGvaPage: Some(emu_translate),
};

/// The LAPIC state page, aligned like QEMU's `struct whpx_lapic_state`.
#[repr(C, align(16))]
struct LapicPage([u8; apic::PAGE_SIZE]);

/// A WHPX vCPU.
pub struct WhpxVcpu {
    shared: Arc<Shared>,
    irqchip: bool,
    msr: MsrContext,
    model: cpuid::Model,
    inject: InjectState,
    halted: bool,
    hyperv_hlt: bool,
    exception: Option<u128>,
    exit: Box<WHV_RUN_VP_EXIT_CONTEXT>,
    emulator: Emulator,
}

impl fmt::Debug for WhpxVcpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WhpxVcpu")
            .field("index", &self.shared.index)
            .field("irqchip", &self.irqchip)
            .field("inject", &self.inject)
            .field("halted", &self.halted)
            .finish_non_exhaustive()
    }
}

impl Drop for WhpxVcpu {
    fn drop(&mut self) {
        let part = &self.shared.part;
        // SAFETY: the vCPU was created in new() and nothing runs it any more.
        unsafe { (part.d().delete_vp)(part.handle(), self.shared.index) };
    }
}

impl WhpxVcpu {
    /// Creates vCPU `index`, `whpx_init_vcpu()`. `initial_apic_id` goes to the hypervisor
    /// when the APIC is ruvm's. `tsc_khz` is the CPU's `tsc-frequency`; without one the host
    /// frequency is used, and [`WhpxVcpu::tsc_khz`] reports it so it can migrate.
    pub fn new(
        accel: &WhpxAccel,
        index: u32,
        initial_apic_id: u32,
        tsc_khz: Option<u32>,
    ) -> Result<WhpxVcpu, WhpxError> {
        let part = Arc::clone(accel.partition());
        let d = part.d();
        let mut handle = std::ptr::null_mut();
        // SAFETY: the callbacks are 'static and the call writes one handle.
        let hr = unsafe { (d.emulator_create)(&CALLBACKS, &mut handle) };
        WhpxError::check("Failed to create instruction emulator", hr)?;
        let emulator = Emulator { handle, d };

        // SAFETY: the partition is set up and the index is the caller's.
        let hr = unsafe { (d.create_vp)(part.handle(), index, 0) };
        WhpxError::check("Failed to create a virtual processor", hr)?;

        let irqchip = accel.irqchip_in_kernel();
        let tsc_khz = tsc_khz
            .or_else(|| {
                frequency(d, WHvCapabilityCodeProcessorClockFrequency).map(|f| (f / 1000) as u32)
            })
            .unwrap_or(0);
        // The userspace APIC runs at 1 GHz.
        let apic_bus_freq = if irqchip {
            frequency(d, WHvCapabilityCodeInterruptClockFrequency)
                .unwrap_or(HYPERV_APIC_BUS_FREQUENCY)
        } else {
            USERSPACE_APIC_BUS_FREQUENCY
        };
        let vcpu = WhpxVcpu {
            shared: Arc::new(Shared {
                part,
                index,
                interrupt_request: AtomicU32::new(0),
                exit_request: AtomicBool::new(false),
            }),
            irqchip,
            msr: MsrContext {
                irqchip_in_kernel: irqchip,
                hyperv: accel.hyperv_enabled(),
                ignore_unknown_msr: accel.ignore_unknown_msr(),
                apic_bus_freq,
            },
            model: cpuid::Model {
                hyperv: accel.hyperv_enabled(),
                vmware_cpuid_freq: false,
                tsc_khz,
                apic_bus_freq,
                x2apic: false,
                apic: true,
            },
            inject: InjectState::default(),
            halted: false,
            hyperv_hlt: false,
            exception: None,
            exit: Box::default(),
            emulator,
        };
        if !irqchip {
            vcpu.set(&[WHvX64RegisterInitialApicId], &[Reg::u64(u64::from(initial_apic_id))])?;
        }
        Ok(vcpu)
    }

    /// The vCPU index.
    pub fn index(&self) -> u32 {
        self.shared.index
    }

    /// A handle for other threads.
    pub fn kick_handle(&self) -> WhpxKick {
        WhpxKick(Arc::clone(&self.shared))
    }

    /// `env->tsc_khz` after creation.
    pub fn tsc_khz(&self) -> u32 {
        self.model.tsc_khz
    }

    /// `env->apic_bus_freq`.
    pub fn apic_bus_freq(&self) -> u64 {
        self.model.apic_bus_freq
    }

    /// The CPU model properties the CPUID fixups look at: `vmware-cpuid-freq` and whether the
    /// model has x2APIC.
    pub fn set_cpu_model(&mut self, vmware_cpuid_freq: bool, x2apic: bool) {
        self.model.vmware_cpuid_freq = vmware_cpuid_freq;
        self.model.x2apic = x2apic;
    }

    /// `cpu->halted`.
    pub fn halted(&self) -> bool {
        self.halted
    }

    /// Sets `cpu->halted`, which `do_cpu_init()` does for an AP and `do_cpu_sipi()` clears.
    pub fn set_halted(&mut self, halted: bool) {
        self.halted = halted;
    }

    fn set(&self, names: &[i32], vals: &[Reg]) -> Result<(), WhpxError> {
        self.shared.part.set_regs(self.shared.index, names, vals)
    }

    fn get(&self, names: &[i32], vals: &mut [Reg]) -> Result<(), WhpxError> {
        self.shared.part.get_regs(self.shared.index, names, vals)
    }

    /// Reads registers by `WHV_REGISTER_NAME`, each as its low and high 64 bits.
    pub fn registers(&self, names: &[i32]) -> Result<Vec<[u64; 2]>, WhpxError> {
        let mut vals = vec![Reg::default(); names.len()];
        self.get(names, &mut vals)?;
        Ok(vals.iter().map(|r| [r.lo, r.hi]).collect())
    }

    /// Writes registers by `WHV_REGISTER_NAME`, the other half of [`WhpxVcpu::registers`].
    pub fn set_registers(&mut self, names: &[i32], vals: &[[u64; 2]]) -> Result<(), WhpxError> {
        let regs: Vec<Reg> = vals.iter().map(|v| Reg { lo: v[0], hi: v[1] }).collect();
        self.set(names, &regs)?;
        // The interrupt code reads IF from the last exit, so keep it current.
        if let Some(i) = names.iter().position(|&n| n == WHvX64RegisterRflags) {
            self.exit.VpContext.Rflags = vals[i][0];
        }
        Ok(())
    }

    /// The Hyper-V LAPIC state page, `whpx_apic_get()`, for [`apic::LapicState::from_page`].
    pub fn lapic_state(&self) -> Result<[u8; apic::PAGE_SIZE], WhpxError> {
        let part = &self.shared.part;
        let f = part
            .d()
            .get_lapic_state2
            .ok_or(WhpxError::Function("WHvGetVirtualProcessorInterruptControllerState2"))?;
        let mut page = LapicPage([0; apic::PAGE_SIZE]);
        let mut written = 0;
        // SAFETY: the page is PAGE_SIZE writable bytes.
        let hr = unsafe {
            f(
                part.handle(),
                self.shared.index,
                page.0.as_mut_ptr().cast(),
                apic::PAGE_SIZE as u32,
                &mut written,
            )
        };
        WhpxError::check("Failed to get interrupt controller state", hr)?;
        Ok(page.0)
    }

    /// Loads the Hyper-V LAPIC, `whpx_apic_put()`.
    pub fn set_lapic_state(&self, state: &[u8; apic::PAGE_SIZE]) -> Result<(), WhpxError> {
        let part = &self.shared.part;
        let f = part
            .d()
            .set_lapic_state2
            .ok_or(WhpxError::Function("WHvSetVirtualProcessorInterruptControllerState2"))?;
        let page = LapicPage(*state);
        // SAFETY: the page is PAGE_SIZE readable bytes.
        let hr = unsafe {
            f(part.handle(), self.shared.index, page.0.as_ptr().cast(), apic::PAGE_SIZE as u32)
        };
        WhpxError::check("Failed to set interrupt controller state", hr)
    }

    fn interrupts_enabled(&self) -> bool {
        self.exit.VpContext.Rflags & IF_MASK != 0
    }

    /// Runs the vCPU until something needs the caller, `whpx_vcpu_run()`. `io` is the port
    /// I/O address space and `mem` the system memory one.
    pub fn run(
        &mut self,
        io: &AddressSpace,
        mem: &AddressSpace,
        exits: &mut dyn WhpxExits,
    ) -> Result<VcpuStop, WhpxError> {
        let ev = irq::async_events(
            self.shared.pending(),
            self.interrupts_enabled(),
            self.hyperv_hlt,
            false,
        );
        self.shared.reset(ev.clear);
        if ev.init {
            // do_cpu_init() keeps only a pending SIPI.
            self.shared.reset(!irq::INTERRUPT_SIPI);
            self.inject.interruptable = true;
            return Ok(VcpuStop::Init);
        }
        if ev.poll {
            exits.apic_poll();
        }
        if ev.wake {
            self.halted = false;
            self.hyperv_hlt = false;
        }
        if ev.tpr_report {
            exits.tpr_access_report(self.exit.VpContext.Rip);
        }
        if ev.sipi {
            return Ok(VcpuStop::Sipi);
        }
        if self.halted && !self.irqchip {
            self.shared.exit_request.store(false, Ordering::Release);
            return Ok(VcpuStop::Halted);
        }

        let part = Arc::clone(&self.shared.part);
        loop {
            self.pre_run(exits)?;
            if self.shared.exit_request.load(Ordering::Acquire) {
                part.cancel_run(self.shared.index);
            }
            if let Some(event) = self.exception.take() {
                self.set(&[WHvRegisterPendingEvent], &[Reg::u128(event)])?;
            }
            let ctx = (&raw mut *self.exit).cast::<c_void>();
            // SAFETY: the exit context is a live buffer of the size passed and this thread
            // holds the vCPU.
            let hr = unsafe {
                (part.d().run_vp)(
                    part.handle(),
                    self.shared.index,
                    ctx,
                    size_of::<WHV_RUN_VP_EXIT_CONTEXT>() as u32,
                )
            };
            WhpxError::check("Failed to exec a virtual processor", hr)?;
            self.post_run(exits);
            if let Some(stop) = self.handle_exit(io, mem, exits)? {
                return Ok(stop);
            }
        }
    }

    /// `whpx_vcpu_pre_run()`.
    fn pre_run(&mut self, exits: &mut dyn WhpxExits) -> Result<(), WhpxError> {
        let input = PreRunInput {
            pending: self.shared.pending(),
            irr: exits.apic_irr(),
            pic_output: exits.pic_output(),
            interrupts_enabled: self.interrupts_enabled(),
            smm: false,
            irqchip_in_kernel: self.irqchip,
            apic_tpr: if self.irqchip { 0 } else { exits.apic_tpr() },
        };
        let out = irq::pre_run(&mut self.inject, &input, || exits.pic_interrupt());
        self.shared.reset(out.clear);

        let mut names = Vec::with_capacity(4);
        let mut vals = Vec::with_capacity(4);
        if let Some(v) = out.pending_interruption {
            names.push(WHvRegisterPendingInterruption);
            vals.push(Reg::u64(v));
        }
        if let Some(v) = out.pending_event {
            names.push(WHvRegisterPendingEvent);
            vals.push(Reg::u128(v));
        }
        if let Some(v) = out.cr8 {
            names.push(WHvX64RegisterCr8);
            vals.push(Reg::u64(v));
        }
        if let Some(v) = out.deliverability {
            names.push(WHvX64RegisterDeliverabilityNotifications);
            vals.push(Reg::u64(v));
        }
        if !names.is_empty() {
            self.set(&names, &vals)?;
        }
        // whpx_vcpu_kick_out_of_hlt().
        if out.kick_out_of_hlt {
            let mut v = [Reg::default()];
            self.get(&[WHvRegisterInternalActivityState], &mut v)?;
            if v[0].lo & irq::HALT_SUSPEND != 0 {
                v[0].lo &= !irq::HALT_SUSPEND;
                self.set(&[WHvRegisterInternalActivityState], &v)?;
            }
        }
        if out.exit_request {
            self.shared.exit_request.store(true, Ordering::Release);
        }
        Ok(())
    }

    /// `whpx_vcpu_post_run()`.
    fn post_run(&mut self, exits: &mut dyn WhpxExits) {
        let vp = &self.exit.VpContext;
        // SAFETY: both members of the union are the same 16 bits.
        let state = unsafe { vp.ExecutionState.AsUINT16 };
        if let Some(tpr) =
            irq::post_run(&mut self.inject, state, exit::cr8(vp._bitfield), self.irqchip)
        {
            exits.set_apic_tpr(tpr);
        }
    }

    // The exit reasons are windows-sys constants with their C names.
    #[allow(non_upper_case_globals)]
    fn handle_exit(
        &mut self,
        io: &AddressSpace,
        mem: &AddressSpace,
        exits: &mut dyn WhpxExits,
    ) -> Result<Option<VcpuStop>, WhpxError> {
        let rip = self.exit.VpContext.Rip;
        let next =
            rip.wrapping_add(u64::from(exit::instruction_length(self.exit.VpContext._bitfield)));
        match self.exit.ExitReason {
            WHvRunVpExitReasonMemoryAccess => self.emulate(io, mem, false)?,
            WHvRunVpExitReasonX64IoPortAccess => self.port_io(io, mem, next)?,
            WHvRunVpExitReasonX64InterruptWindow => irq::window_exit(&mut self.inject),
            WHvRunVpExitReasonX64ApicEoi => {
                // SAFETY: the exit reason says which member of the union is filled.
                let vector = unsafe { self.exit.Anonymous.ApicEoi.InterruptVector };
                exits.apic_eoi(vector as u8);
            }
            WHvRunVpExitReasonX64Halt => {
                if irq::halts(self.shared.pending(), self.interrupts_enabled()) {
                    self.halted = true;
                    return Ok(Some(VcpuStop::Halted));
                }
            }
            WHvRunVpExitReasonCanceled => {
                self.shared.exit_request.store(false, Ordering::Release);
                return Ok(Some(VcpuStop::Kicked));
            }
            WHvRunVpExitReasonX64MsrAccess => return self.msr_exit(rip, next, exits),
            WHvRunVpExitReasonX64Cpuid => self.cpuid_exit(next, exits)?,
            WHvRunVpExitReasonException => return Ok(self.exception_exit()),
            other => return Ok(Some(VcpuStop::Unexpected(other))),
        }
        Ok(None)
    }

    /// MMIO, or string port I/O when `port` is set, through the instruction emulator.
    fn emulate(
        &mut self,
        io: &AddressSpace,
        mem: &AddressSpace,
        port: bool,
    ) -> Result<(), WhpxError> {
        let part = &self.shared.part;
        let ctx = EmuCtx { part, index: self.shared.index, io, mem };
        let ctx = (&raw const ctx).cast::<c_void>();
        let d = part.d();
        let mut status = WHV_EMULATOR_STATUS::default();
        let e = &*self.exit;
        // SAFETY: the emulator is live, the exit reason says which member of the union is
        // filled, and `ctx` outlives the call that runs the callbacks.
        let (hr, status) = unsafe {
            let hr = if port {
                (d.emulator_try_io)(
                    self.emulator.handle,
                    ctx,
                    &e.VpContext,
                    &raw const e.Anonymous.IoPortAccess,
                    &mut status,
                )
            } else {
                (d.emulator_try_mmio)(
                    self.emulator.handle,
                    ctx,
                    &e.VpContext,
                    &raw const e.Anonymous.MemoryAccess,
                    &mut status,
                )
            };
            (hr, status.AsUINT32)
        };
        let what = if port { "PortIO access" } else { "MMIO access" };
        WhpxError::check(
            if port { "Failed to parse PortIO access" } else { "Failed to parse MMIO access" },
            hr,
        )?;
        if !exit::emulation_ok(status) {
            return Err(WhpxError::Emulation(what, status));
        }
        Ok(())
    }

    /// `whpx_handle_portio()`: IN and OUT directly, INS and OUTS through the emulator.
    fn port_io(
        &mut self,
        io: &AddressSpace,
        mem: &AddressSpace,
        next: u64,
    ) -> Result<(), WhpxError> {
        // SAFETY: the exit reason says which member of the union is filled.
        let (bits, port, rax) = unsafe {
            let c = &self.exit.Anonymous.IoPortAccess;
            (c.AccessInfo.AsUINT32, c.PortNumber, c.Rax)
        };
        let info = IoInfo::new(bits);
        if info.string {
            return self.emulate(io, mem, true);
        }
        let size = info.size.min(4);
        let port = u64::from(port);
        if info.write {
            let _ = io.write(port, MemTxAttrs::UNSPECIFIED, &rax.to_le_bytes()[..size]);
            self.set(&[WHvX64RegisterRip], &[Reg::u64(next)])
        } else {
            let mut buf = [0; 8];
            let _ = io.read(port, MemTxAttrs::UNSPECIFIED, &mut buf[..size]);
            let rax = exit::port_read_rax(rax, size, u64::from_le_bytes(buf));
            self.set(&[WHvX64RegisterRip, WHvX64RegisterRax], &[Reg::u64(next), Reg::u64(rax)])
        }
    }

    /// The `WHvRunVpExitReasonX64MsrAccess` case.
    fn msr_exit(
        &mut self,
        rip: u64,
        next: u64,
        exits: &mut dyn WhpxExits,
    ) -> Result<Option<VcpuStop>, WhpxError> {
        // SAFETY: the exit reason says which member of the union is filled.
        let (bits, number, rax, rdx) = unsafe {
            let c = &self.exit.Anonymous.MsrAccess;
            (c.AccessInfo.AsUINT32, c.MsrNumber, c.Rax, c.Rdx)
        };
        let e = MsrExit::new(number, bits & 1 != 0, rax, rdx);
        let mut value = 0;
        let mut gpf = false;
        match msr::classify(&e, &self.msr) {
            MsrAction::Read(v) => value = v,
            MsrAction::Ignore | MsrAction::Unknown => {}
            MsrAction::Gpf => gpf = true,
            MsrAction::SetApicBase(v) => match exits.set_apic_base(v) {
                Ok(apic) => {
                    if let Some(on) = apic {
                        self.model.apic = on;
                    }
                    self.set(&[WHvX64RegisterApicBase], &[Reg::u64(v)])?;
                }
                Err(apic::BadApicBase) => gpf = true,
            },
            MsrAction::ApicRead(i) => match exits.apic_msr_read(i) {
                Some(v) => value = v,
                None => gpf = true,
            },
            MsrAction::ApicWrite(i, v) => gpf = !exits.apic_msr_write(i, v),
            MsrAction::GuestIdle => {
                // whpx_handle_hyperv_guestidle(): step over and halt with the Hyper-V flag,
                // which lets an interrupt wake the vCPU even with IF clear.
                self.set(&[WHvX64RegisterRip], &[Reg::u64(next)])?;
                self.hyperv_hlt = true;
                if irq::halts(self.shared.pending(), self.interrupts_enabled()) {
                    self.halted = true;
                    return Ok(Some(VcpuStop::Halted));
                }
                return Ok(None);
            }
        }
        let len = next.wrapping_sub(rip) as u8;
        let r = msr::finish(&e, rip, len, value, gpf);
        match r.rax_rdx {
            Some((a, d)) => self.set(
                &[WHvX64RegisterRip, WHvX64RegisterRax, WHvX64RegisterRdx],
                &[Reg::u64(r.rip), Reg::u64(a), Reg::u64(d)],
            )?,
            None => self.set(&[WHvX64RegisterRip], &[Reg::u64(r.rip)])?,
        }
        if gpf {
            self.exception = Some(irq::exception_event(features::EXCEPTION_GP as u8, Some(0), 0));
        }
        Ok(None)
    }

    /// The `WHvRunVpExitReasonX64Cpuid` case.
    fn cpuid_exit(&mut self, next: u64, exits: &mut dyn WhpxExits) -> Result<(), WhpxError> {
        // SAFETY: the exit reason says which member of the union is filled.
        let e = unsafe {
            let c = &self.exit.Anonymous.CpuidAccess;
            CpuidExit {
                leaf: c.Rax as u32,
                subleaf: c.Rcx as u32,
                default: cpuid::Regs {
                    eax: c.DefaultResultRax as u32,
                    ebx: c.DefaultResultRbx as u32,
                    ecx: c.DefaultResultRcx as u32,
                    edx: c.DefaultResultRdx as u32,
                },
            }
        };
        let r = cpuid::answer(&e, exits.cpuid(e.leaf, e.subleaf), &self.model);
        self.set(
            &[
                WHvX64RegisterRip,
                WHvX64RegisterRax,
                WHvX64RegisterRcx,
                WHvX64RegisterRdx,
                WHvX64RegisterRbx,
            ],
            &[
                Reg::u64(next),
                Reg::u64(u64::from(r.eax)),
                Reg::u64(u64::from(r.ecx)),
                Reg::u64(u64::from(r.edx)),
                Reg::u64(u64::from(r.ebx)),
            ],
        )
    }

    /// The `WHvRunVpExitReasonException` case. Only #GP is intercepted outside the gdbstub,
    /// for `intercept-msr-gp`. QEMU 11.1 decodes the instruction there, and for RDMSR and
    /// WRMSR it knows no MSR and raises #GP(0) again, so either way the guest gets its #GP
    /// back. QEMU only warns about a software exception and drops it.
    fn exception_exit(&mut self) -> Option<VcpuStop> {
        // SAFETY: the exit reason says which member of the union is filled.
        let (kind, info, code, param) = unsafe {
            let c = &self.exit.Anonymous.VpException;
            (c.ExceptionType, c.ExceptionInfo.AsUINT32, c.ErrorCode, c.ExceptionParameter)
        };
        if u32::from(kind) != features::EXCEPTION_GP {
            return Some(VcpuStop::Exception(kind));
        }
        let info = ExceptionInfo::new(info);
        if !info.software {
            self.exception =
                Some(irq::exception_event(kind, info.error_code_valid.then_some(code), param));
        }
        None
    }
}
