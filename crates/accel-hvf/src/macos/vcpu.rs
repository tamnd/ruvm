// SPDX-License-Identifier: GPL-2.0-or-later

//! One vCPU: `hvf_arch_init_vcpu()`, the register sync, and the run loop of
//! `hvf_arch_vcpu_exec()` with `hvf_handle_exception()`.
//!
//! The framework ties a vCPU to the thread that created it, so [`HvfVcpu`] is not `Send` and
//! is made, run and dropped on its own thread. Other threads reach it through [`HvfKick`].

use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use ruvm_mem::{AddressSpace, MemTxAttrs};

use super::HvfAccel;
use super::ffi::{self, HvVcpu};
use super::slots::{SlotListener, WriteFault};
use crate::cpu::{self, HostIdRegs};
use crate::esr::{self, DataAbort, Exception};
use crate::psci::{self, PsciCall};
use crate::regs::ArmRegs;
use crate::sysreg::{self, TrapOwner, TrapReg, id, trap};
use crate::vtimer::{self, WfiTimer};
use crate::{Conduit, HvfError, VcpuStop, hv};

/// The exits the vCPU hands to the rest of the machine.
pub trait HvfExits {
    /// A trapped system register read the vCPU does not answer itself: the PMU, the physical
    /// timer and the GICv3 CPU interface when the GIC is ruvm's, `hvf_sysreg_read_cp()`.
    /// `None` makes the access UNDEF.
    fn sysreg_read(&mut self, reg: TrapReg) -> Option<u64> {
        let _ = reg;
        None
    }

    /// The write side of [`HvfExits::sysreg_read`]. `false` makes the access UNDEF.
    fn sysreg_write(&mut self, reg: TrapReg, val: u64) -> bool {
        let _ = (reg, val);
        false
    }

    /// The vtimer interrupt line, the CPU's `gt_timer_outputs[GTIMER_VIRT]`, when the GIC is
    /// ruvm's.
    fn vtimer_irq(&mut self, level: bool) {
        let _ = level;
    }
}

/// What other threads share with a vCPU.
#[derive(Debug)]
struct Shared {
    fd: HvVcpu,
    exit_request: AtomicBool,
    irq: AtomicBool,
    fiq: AtomicBool,
}

/// A handle that reaches a running vCPU from any thread, `hvf_kick_vcpu_thread()`.
#[derive(Clone, Debug)]
pub struct HvfKick(Arc<Shared>);

impl HvfKick {
    /// Makes the vCPU leave [`HvfVcpu::run`] with [`VcpuStop::Kicked`] as soon as it can.
    pub fn kick(&self) {
        self.0.exit_request.store(true, Ordering::Release);
        // SAFETY: hv_vcpus_exit() takes any thread's request to stop a list of vCPUs; the
        // list is one id read from a valid place.
        let r = unsafe { ffi::hv_vcpus_exit(&self.0.fd, 1) };
        if let Err(e) = HvfError::check("hv_vcpus_exit", r) {
            panic!("{e}");
        }
    }

    /// Sets the IRQ line, `CPU_INTERRUPT_HARD`. The framework forgets a pending interrupt
    /// after each run, so the vCPU injects it again before every run while the line is up.
    pub fn set_irq(&self, level: bool) {
        if self.0.irq.swap(level, Ordering::AcqRel) != level && level {
            self.kick_run();
        }
    }

    /// Sets the FIQ line, `CPU_INTERRUPT_FIQ`.
    pub fn set_fiq(&self, level: bool) {
        if self.0.fiq.swap(level, Ordering::AcqRel) != level && level {
            self.kick_run();
        }
    }

    /// Whether an interrupt line is up, `cpu_has_work()`.
    pub fn has_work(&self) -> bool {
        self.0.irq.load(Ordering::Acquire) || self.0.fiq.load(Ordering::Acquire)
    }

    /// Stops the current run so the new line state is seen, without asking the caller to
    /// look at anything.
    fn kick_run(&self) {
        // SAFETY: as in kick().
        unsafe { ffi::hv_vcpus_exit(&self.0.fd, 1) };
    }
}

/// `get_cntfrq_el0()`: the counter frequency the guest's vtimer runs at.
fn cntfrq() -> u64 {
    let f: u64;
    // SAFETY: CNTFRQ_EL0 is readable at EL0 on macOS and the read has no side effects.
    unsafe { core::arch::asm!("mrs {}, cntfrq_el0", out(reg) f, options(nomem, nostack)) };
    f
}

/// An HVF vCPU, owned by the thread that created it.
#[derive(Debug)]
pub struct HvfVcpu {
    shared: Arc<Shared>,
    exit: *const ffi::HvVcpuExit,
    index: u32,
    irqchip: bool,
    conduit: Conduit,
    sync_list: Vec<u16>,
    slots: Arc<SlotListener>,
    vtimer_offset: Arc<AtomicU64>,
    vtimer_masked: bool,
    cntfrq: u64,
    oslsr: u64,
    _not_send: PhantomData<*const ()>,
}

impl HvfVcpu {
    /// Creates vCPU `index` on the calling thread, `hvf_init_vcpu()`, and sets the registers
    /// `hvf_arch_init_vcpu()` sets once: MIDR, MPIDR and the ID registers QEMU never reads
    /// back. `host` is [`HvfAccel::host_id_regs`], `gicv3` says whether the guest has a GICv3
    /// CPU interface.
    pub fn new(
        accel: &HvfAccel,
        index: u32,
        mpidr: u64,
        host: &HostIdRegs,
        gicv3: bool,
        conduit: Conduit,
    ) -> Result<HvfVcpu, HvfError> {
        let mut fd: HvVcpu = 0;
        let mut exit: *const ffi::HvVcpuExit = std::ptr::null();
        // SAFETY: both out pointers are valid, and a null config asks for the defaults.
        let r = unsafe { ffi::hv_vcpu_create(&mut fd, &mut exit, std::ptr::null_mut()) };
        HvfError::check("hv_vcpu_create", r)?;
        let vcpu = HvfVcpu {
            shared: Arc::new(Shared {
                fd,
                exit_request: AtomicBool::new(false),
                irq: AtomicBool::new(false),
                fiq: AtomicBool::new(false),
            }),
            exit,
            index,
            irqchip: accel.irqchip_in_kernel(),
            conduit,
            sync_list: sysreg::sync_list(accel.irqchip_in_kernel(), accel.el2()),
            slots: accel.slot_listener(),
            vtimer_offset: accel.vtimer_offset(),
            vtimer_masked: false,
            cntfrq: cntfrq(),
            oslsr: 0,
            _not_send: PhantomData,
        };
        vcpu.set_sysreg(id::MIDR_EL1, cpu::MIDR)?;
        vcpu.set_sysreg(id::MPIDR_EL1, mpidr)?;
        vcpu.set_sysreg(id::ID_AA64PFR0_EL1, host.pfr0_for_vcpu(gicv3))?;
        vcpu.set_sysreg(id::ID_AA64ISAR0_EL1, host.isar0)?;
        vcpu.set_sysreg(id::ID_AA64MMFR0_EL1, cpu::clamp_mmfr0(host.mmfr0, accel.ipa_bits()))?;
        Ok(vcpu)
    }

    /// The vCPU index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// A handle for other threads.
    pub fn kick_handle(&self) -> HvfKick {
        HvfKick(Arc::clone(&self.shared))
    }

    /// The counter frequency in Hz.
    pub fn cntfrq(&self) -> u64 {
        self.cntfrq
    }

    /// The system registers [`HvfVcpu::get_regs`] and [`HvfVcpu::put_regs`] move.
    pub fn sync_list(&self) -> &[u16] {
        &self.sync_list
    }

    fn fd(&self) -> HvVcpu {
        self.shared.fd
    }

    /// Reads a general purpose or special register, `HV_REG_*`.
    pub fn reg(&self, reg: u32) -> Result<u64, HvfError> {
        let mut v = 0;
        // SAFETY: this thread owns the vCPU and the call writes one u64.
        let r = unsafe { ffi::hv_vcpu_get_reg(self.fd(), reg, &mut v) };
        HvfError::check("hv_vcpu_get_reg", r).map(|()| v)
    }

    /// Writes a general purpose or special register.
    pub fn set_reg(&self, reg: u32, val: u64) -> Result<(), HvfError> {
        // SAFETY: this thread owns the vCPU.
        let r = unsafe { ffi::hv_vcpu_set_reg(self.fd(), reg, val) };
        HvfError::check("hv_vcpu_set_reg", r)
    }

    /// Reads general register `rt` as an instruction names it, `hvf_get_reg()`: 31 is XZR.
    pub fn x(&self, rt: u32) -> Result<u64, HvfError> {
        if rt < 31 { self.reg(rt) } else { Ok(0) }
    }

    /// Writes general register `rt`, `hvf_set_reg()`. Writes to XZR go nowhere.
    pub fn set_x(&self, rt: u32, val: u64) -> Result<(), HvfError> {
        if rt < 31 { self.set_reg(rt, val) } else { Ok(()) }
    }

    /// Reads a system register by `hv_sys_reg_t` id.
    pub fn sysreg(&self, reg: u16) -> Result<u64, HvfError> {
        let mut v = 0;
        // SAFETY: this thread owns the vCPU and the call writes one u64.
        let r = unsafe { ffi::hv_vcpu_get_sys_reg(self.fd(), reg, &mut v) };
        HvfError::check("hv_vcpu_get_sys_reg", r).map(|()| v)
    }

    /// Writes a system register by `hv_sys_reg_t` id.
    pub fn set_sysreg(&self, reg: u16, val: u64) -> Result<(), HvfError> {
        // SAFETY: this thread owns the vCPU.
        let r = unsafe { ffi::hv_vcpu_set_sys_reg(self.fd(), reg, val) };
        HvfError::check("hv_vcpu_set_sys_reg", r)
    }

    fn q(&self, n: u32) -> Result<u128, HvfError> {
        let mut v = 0u128;
        // SAFETY: this thread owns the vCPU and the call writes 16 bytes to an aligned u128.
        let r = unsafe { ffi::hv_vcpu_get_simd_fp_reg(self.fd(), n, &mut v) };
        HvfError::check("hv_vcpu_get_simd_fp_reg", r).map(|()| v)
    }

    fn set_q(&self, n: u32, val: u128) -> Result<(), HvfError> {
        // SAFETY: this thread owns the vCPU and the shim reads 16 bytes from an aligned u128.
        let r = unsafe { ffi::ruvm_hv_vcpu_set_simd_fp_reg(self.fd(), n, &val) };
        HvfError::check("hv_vcpu_set_simd_fp_reg", r)
    }

    /// Reads every register QEMU syncs, `hvf_arch_get_registers()`.
    pub fn get_regs(&self) -> Result<ArmRegs, HvfError> {
        let mut regs = ArmRegs::default();
        for (i, x) in regs.x.iter_mut().enumerate() {
            *x = self.reg(i as u32)?;
        }
        regs.pc = self.reg(ffi::HV_REG_PC)?;
        regs.cpsr = self.reg(ffi::HV_REG_CPSR)?;
        regs.fpcr = self.reg(ffi::HV_REG_FPCR)?;
        regs.fpsr = self.reg(ffi::HV_REG_FPSR)?;
        for (i, q) in regs.q.iter_mut().enumerate() {
            *q = self.q(i as u32)?;
        }
        regs.sysregs = self
            .sync_list
            .iter()
            .map(|&r| self.sysreg(r).map(|v| (r, v)))
            .collect::<Result<_, _>>()?;
        Ok(regs)
    }

    /// Writes every register back, `hvf_arch_put_registers()`, and the vtimer offset with
    /// them. System registers outside the sync list are written too.
    pub fn put_regs(&self, regs: &ArmRegs) -> Result<(), HvfError> {
        for (i, &x) in regs.x.iter().enumerate() {
            self.set_reg(i as u32, x)?;
        }
        self.set_reg(ffi::HV_REG_PC, regs.pc)?;
        self.set_reg(ffi::HV_REG_CPSR, regs.cpsr)?;
        self.set_reg(ffi::HV_REG_FPCR, regs.fpcr)?;
        self.set_reg(ffi::HV_REG_FPSR, regs.fpsr)?;
        for (i, &q) in regs.q.iter().enumerate() {
            self.set_q(i as u32, q)?;
        }
        for &(r, v) in &regs.sysregs {
            self.set_sysreg(r, v)?;
        }
        let offset = self.vtimer_offset.load(Ordering::Relaxed);
        // SAFETY: this thread owns the vCPU.
        let r = unsafe { ffi::hv_vcpu_set_vtimer_offset(self.fd(), offset) };
        HvfError::check("hv_vcpu_set_vtimer_offset", r)
    }

    fn advance_pc(&self) -> Result<(), HvfError> {
        let pc = self.reg(ffi::HV_REG_PC)?;
        self.set_reg(ffi::HV_REG_PC, pc.wrapping_add(4))
    }

    /// `hvf_raise_exception(cpu, EXCP_UDEF, syn_uncategorized(), 1)`.
    fn raise_undef(&self) -> Result<(), HvfError> {
        let pc = self.reg(ffi::HV_REG_PC)?;
        let cpsr = self.reg(ffi::HV_REG_CPSR)?;
        let vbar = self.sysreg(id::VBAR_EL1)?;
        let sctlr = self.sysreg(id::SCTLR_EL1)?;
        let e = esr::take_to_el1(pc, cpsr, vbar, sctlr, esr::syn_uncategorized());
        self.set_sysreg(id::ESR_EL1, e.esr)?;
        self.set_sysreg(id::ELR_EL1, e.elr)?;
        self.set_sysreg(id::SPSR_EL1, e.spsr)?;
        self.set_reg(ffi::HV_REG_CPSR, e.cpsr)?;
        self.set_reg(ffi::HV_REG_PC, e.pc)
    }

    /// The guest's virtual counter now.
    fn vtimer_now(&self) -> u64 {
        // SAFETY: no arguments.
        let now = unsafe { ffi::mach_absolute_time() };
        now.wrapping_sub(self.vtimer_offset.load(Ordering::Relaxed))
    }

    /// `hvf_sync_vtimer()`: after the framework masked the vtimer, keep the line up while it
    /// fires and unmask it once the guest has dealt with it.
    fn sync_vtimer(&mut self, exits: &mut dyn HvfExits) -> Result<(), HvfError> {
        if !self.vtimer_masked {
            return Ok(());
        }
        let level = vtimer::irq_level(self.sysreg(id::CNTV_CTL_EL0)?);
        exits.vtimer_irq(level);
        if !level {
            // SAFETY: this thread owns the vCPU.
            let r = unsafe { ffi::hv_vcpu_set_vtimer_mask(self.fd(), false) };
            HvfError::check("hv_vcpu_set_vtimer_mask", r)?;
            self.vtimer_masked = false;
        }
        Ok(())
    }

    /// `hvf_sysreg_read()` for the registers the vCPU answers itself.
    fn sysreg_read(&self, reg: TrapReg, exits: &mut dyn HvfExits) -> Result<Option<u64>, HvfError> {
        if !self.irqchip && (reg == trap::PMCEID0_EL0 || reg == trap::PMCEID1_EL0) {
            // We can't really count anything yet, declare all events invalid.
            return Ok(Some(0));
        }
        let v = match reg {
            trap::OSLSR_EL1 => Some(self.oslsr),
            trap::OSDLR_EL1 => Some(0),
            trap::CNTHCTL_EL2 => {
                Some(if ffi::macos15().is_some() { self.sysreg(id::CNTHCTL_EL2)? } else { 0 })
            }
            trap::MDCCINT_EL1 => Some(self.sysreg(id::MDCCINT_EL1)?),
            r if r.debug_reg().is_some() => Some(self.sysreg(r.hv_id())?),
            r if TrapOwner::of(r).is_some() => exits.sysreg_read(r),
            r if r.is_id_space() => Some(0),
            r => exits.sysreg_read(r),
        };
        Ok(v)
    }

    /// `hvf_sysreg_write()` for the registers the vCPU answers itself.
    fn sysreg_write(
        &mut self,
        reg: TrapReg,
        val: u64,
        exits: &mut dyn HvfExits,
    ) -> Result<bool, HvfError> {
        match reg {
            trap::OSLAR_EL1 => self.oslsr = val & 1,
            trap::OSDLR_EL1 | trap::LORC_EL1 => {}
            trap::CNTHCTL_EL2 => {
                if ffi::macos15().is_some() {
                    self.set_sysreg(id::CNTHCTL_EL2, val)?;
                }
            }
            trap::MDCCINT_EL1 | trap::MDSCR_EL1 => self.set_sysreg(reg.hv_id(), val)?,
            r if r.debug_reg().is_some() => self.set_sysreg(r.hv_id(), val)?,
            r => return Ok(exits.sysreg_write(r, val)),
        }
        Ok(true)
    }

    /// `hvf_wfi()`. `None` means do not halt.
    fn wfi(&self) -> Result<Option<VcpuStop>, HvfError> {
        if self.kick_handle().has_work() {
            return Ok(None);
        }
        let ctl = self.sysreg(id::CNTV_CTL_EL0)?;
        let cval = self.sysreg(id::CNTV_CVAL_EL0)?;
        Ok(match vtimer::wfi_timer(ctl, cval, self.vtimer_now(), self.cntfrq) {
            WfiTimer::Expired => None,
            WfiTimer::Halt => Some(VcpuStop::Wfi(None)),
            WfiTimer::Sleep(ns) => Some(VcpuStop::Wfi(Some(Duration::from_nanos(ns)))),
        })
    }

    /// A PSCI call on the conduit, `hvf_handle_psci_call()`, after the PC moved past an SMC.
    fn psci(&self) -> Result<Option<VcpuStop>, HvfError> {
        let args = [self.reg(0)?, self.reg(1)?, self.reg(2)?, self.reg(3)?];
        let Some(call) = PsciCall::decode(args) else {
            // SMCCC 1.3 section 5.2 says every unknown SMCCC call returns -1.
            self.set_reg(0, psci::x0(psci::RET_NOT_SUPPORTED))?;
            return Ok(None);
        };
        if let Some(ret) = call.local_result() {
            self.set_reg(0, psci::x0(ret))?;
            return Ok(None);
        }
        if let PsciCall::CpuSuspend { .. } = call {
            self.set_reg(0, 0)?;
        }
        Ok(Some(VcpuStop::Psci(call)))
    }

    /// `hvf_handle_exception()`. `None` means run on.
    fn exception(
        &mut self,
        exc: ffi::HvExitException,
        mem: &AddressSpace,
        exits: &mut dyn HvfExits,
    ) -> Result<Option<VcpuStop>, HvfError> {
        let syndrome = exc.syndrome;
        let unhandled = |pc| VcpuStop::Unhandled {
            ec: esr::ec(syndrome),
            syndrome,
            pc,
            far: exc.virtual_address,
        };
        match Exception::decode(syndrome) {
            Exception::DataAbort(d) => self.data_abort(d, exc.physical_address, mem).map(|ok| {
                if ok { None } else { Some(unhandled(self.reg(ffi::HV_REG_PC).unwrap_or(0))) }
            }),
            Exception::SysReg { reg, rt, read } => {
                let handled = if read {
                    match self.sysreg_read(reg, exits)? {
                        Some(v) => {
                            self.set_x(rt, v)?;
                            true
                        }
                        None => false,
                    }
                } else {
                    let v = self.x(rt)?;
                    self.sysreg_write(reg, v, exits)?
                };
                if handled {
                    self.advance_pc()?
                } else {
                    self.raise_undef()?
                }
                Ok(None)
            }
            Exception::Wfx { wfe } => {
                self.advance_pc()?;
                if wfe { Ok(None) } else { self.wfi() }
            }
            Exception::Hvc(_) => {
                // Do not advance the PC for HVC.
                if self.conduit == Conduit::Hvc {
                    self.psci()
                } else {
                    self.raise_undef()?;
                    Ok(None)
                }
            }
            Exception::Smc(_) => {
                if self.conduit == Conduit::Smc {
                    self.advance_pc()?;
                    self.psci()
                } else {
                    self.raise_undef()?;
                    Ok(None)
                }
            }
            Exception::Debug(ec) => Ok(Some(VcpuStop::Debug(ec))),
            Exception::Other(_) => Ok(Some(unhandled(self.reg(ffi::HV_REG_PC)?))),
        }
    }

    /// The `EC_DATAABORT` case. `false` is an abort QEMU asserts on: a stage 1 walk fault or
    /// an access without a valid syndrome.
    fn data_abort(&self, d: DataAbort, ipa: u64, mem: &AddressSpace) -> Result<bool, HvfError> {
        // Cache maintenance on MMIO: nothing to do.
        if d.cm {
            self.advance_pc()?;
            return Ok(true);
        }
        // A write to dirty logged RAM: mark the page and retry with write access.
        if d.write && self.slots.write_fault(ipa) == WriteFault::Retry {
            return Ok(true);
        }
        if d.s1ptw || !d.isv {
            return Ok(false);
        }
        let len = d.len as usize;
        if d.write {
            let val = self.x(d.srt)?;
            // QEMU does not inject a fault for a failed access either.
            let _ = mem.write(ipa, MemTxAttrs::UNSPECIFIED, &val.to_le_bytes()[..len]);
        } else {
            let mut buf = [0u8; 8];
            let _ = mem.read(ipa, MemTxAttrs::UNSPECIFIED, &mut buf[..len]);
            self.set_x(d.srt, d.load_value(u64::from_le_bytes(buf)))?;
        }
        self.advance_pc()?;
        Ok(true)
    }

    /// Runs the vCPU until something needs the caller, `hvf_arch_vcpu_exec()`. MMIO goes to
    /// `mem`, the system address space.
    pub fn run(
        &mut self,
        mem: &AddressSpace,
        exits: &mut dyn HvfExits,
    ) -> Result<VcpuStop, HvfError> {
        loop {
            if self.shared.exit_request.swap(false, Ordering::AcqRel) {
                return Ok(VcpuStop::Kicked);
            }
            // The framework clears a pending interrupt after each run, so set it every time.
            for (line, kind) in [
                (&self.shared.fiq, ffi::HV_INTERRUPT_TYPE_FIQ),
                (&self.shared.irq, ffi::HV_INTERRUPT_TYPE_IRQ),
            ] {
                if line.load(Ordering::Acquire) {
                    // SAFETY: this thread owns the vCPU.
                    let r = unsafe { ffi::hv_vcpu_set_pending_interrupt(self.fd(), kind, true) };
                    HvfError::check("hv_vcpu_set_pending_interrupt", r)?;
                }
            }
            // SAFETY: this thread owns the vCPU, and the guest memory it can reach is mapped
            // from RAM blocks the slot listener keeps alive.
            let r = unsafe { ffi::hv_vcpu_run(self.fd()) };
            if r == hv::ILLEGAL_GUEST_STATE {
                return Err(HvfError::Guest("HV_ILLEGAL_GUEST_STATE".to_string()));
            }
            HvfError::check("hv_vcpu_run", r)?;
            // SAFETY: the framework owns the exit record for the vCPU's lifetime and has just
            // filled it; nothing else writes it on this thread.
            let exit = unsafe { *self.exit };
            match exit.reason {
                ffi::HV_EXIT_REASON_CANCELED => return Ok(VcpuStop::Kicked),
                ffi::HV_EXIT_REASON_VTIMER_ACTIVATED => {
                    exits.vtimer_irq(true);
                    self.vtimer_masked = true;
                }
                ffi::HV_EXIT_REASON_EXCEPTION => {
                    if !self.irqchip {
                        self.sync_vtimer(exits)?;
                    }
                    if let Some(stop) = self.exception(exit.exception, mem, exits)? {
                        return Ok(stop);
                    }
                }
                other => {
                    return Err(HvfError::Guest(format!("unknown HVF exit reason {other}")));
                }
            }
        }
    }
}

impl Drop for HvfVcpu {
    fn drop(&mut self) {
        // SAFETY: the vCPU is not Send, so this is the thread that created it.
        unsafe { ffi::hv_vcpu_destroy(self.fd()) };
    }
}
