// SPDX-License-Identifier: GPL-2.0-or-later

//! x86 vCPU glue for KVM, the port of `target/i386/kvm/kvm.c`.
//!
//! [`host_cpuid`] probes the host once per VM and gives the CPU model code
//! the [`HostCpuid`] it filters features against. [`X86KvmVcpu::init`] is
//! `kvm_arch_init_vcpu()`: it programs CPUID, the TSC rate and machine
//! checks, and works out which MSRs the vCPU has. After that
//! [`X86KvmVcpu::put_registers`] and [`X86KvmVcpu::get_registers`] move an
//! [`X86CpuState`] in and out of the vCPU, like `kvm_arch_put_registers()`
//! and `kvm_arch_get_registers()`. [`setup_vcpu`] does the whole reset
//! sequence in one call.
//!
//! The decisions about what goes into each structure live in
//! [`crate::kvm_convert`], which builds on every host; this file only copies
//! between those mirrors and the `kvm-bindings` types and issues ioctls.
//!
//! Not ported: Hyper-V and Xen enlightenments, nested VMX state, the PMU,
//! `KVM_GET/SET_SREGS2`, AMX permission requests, SGX, and the PVH and
//! 64-bit Linux boot entry states.

use std::fmt;
use std::io;

use kvm_bindings::{
    CpuId, KVM_MAX_CPUID_ENTRIES, Msrs, Xsave, kvm_cpuid_entry2, kvm_debugregs, kvm_dtable,
    kvm_lapic_state, kvm_mp_state, kvm_msr_entry, kvm_regs, kvm_segment, kvm_sregs,
    kvm_vcpu_events, kvm_xcrs, kvm_xsave2,
};
use kvm_ioctls::{Cap, Kvm, VcpuFd};
use ruvm_accel_kvm::{KernelIrqchip, KvmAccel, KvmError};

use crate::cpuid::X86Cpu;
use crate::cpuid::host::{CpuidEntry, HostCpuid};
use crate::kvm_convert::{
    EventCaps, KvmCpuidEntry, KvmDtable, KvmFpu, KvmSegment, KvmSregs, MP_STATE_HALTED, MsrSupport,
    PutLevel, VcpuEvents, debugregs_from_state, events_from_state, fpu_from_state,
    freq_within_bounds, get_msr_indices, init_msrs, lapic_reset_regs, negotiate_mcg_cap,
    pkru_offset, put_msrs, set_msr_value, sregs_from_state, state_from_debugregs,
    state_from_events, state_from_fpu, state_from_sregs, state_from_xsave, vcpu_cpuid, wants_mce,
    xsave_from_state,
};
use crate::msr::{
    MCG_LMCE_P, MSR_IA32_APICBASE, MSR_IA32_ARCH_CAPABILITIES, MSR_IA32_FEATURE_CONTROL,
    MSR_IA32_SMBASE, MSR_IA32_TSCDEADLINE, MSR_IA32_VMX_PROCBASED_CTLS2,
};
use crate::state::{
    IrqchipMode, R_EAX, R_EBP, R_EBX, R_ECX, R_EDI, R_EDX, R_ESI, R_ESP, X86CpuState,
};

/// `KVM_X86_SETUP_MCE`, `_IOW(KVMIO, 0x9c, __u64)`.
const KVM_X86_SETUP_MCE: std::os::raw::c_ulong = 0x4008_ae9c;
/// `KVM_X86_GET_MCE_CAP_SUPPORTED`, `_IOR(KVMIO, 0x9d, __u64)`.
const KVM_X86_GET_MCE_CAP_SUPPORTED: std::os::raw::c_ulong = 0x8008_ae9d;

/// Size of the legacy `struct kvm_xsave`.
const KVM_XSAVE_SIZE: usize = 4096;

/// Something went wrong setting up or syncing an x86 vCPU.
#[derive(Debug)]
pub enum X86KvmError {
    /// The accelerator itself failed.
    Kvm(KvmError),
    /// A vCPU or system ioctl failed.
    Ioctl(&'static str, io::Error),
    /// The CPU configuration does not fit what KVM offers.
    Config(String),
}

impl fmt::Display for X86KvmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            X86KvmError::Kvm(e) => e.fmt(f),
            X86KvmError::Ioctl(what, e) => write!(f, "{what}: {e}"),
            X86KvmError::Config(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for X86KvmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            X86KvmError::Kvm(e) => Some(e),
            X86KvmError::Ioctl(_, e) => Some(e),
            X86KvmError::Config(_) => None,
        }
    }
}

impl From<KvmError> for X86KvmError {
    fn from(e: KvmError) -> Self {
        X86KvmError::Kvm(e)
    }
}

fn ioctl_err(what: &'static str) -> impl FnOnce(vmm_sys_util::errno::Error) -> X86KvmError {
    move |e| X86KvmError::Ioctl(what, io::Error::from_raw_os_error(e.errno()))
}

// Host probing.

/// The host CPUID instruction.
fn host_cpuid_insn(function: u32, index: u32) -> [u32; 4] {
    // Older toolchains declare the intrinsic unsafe, newer ones safe.
    #[allow(unsafe_code, unused_unsafe)]
    // SAFETY: CPUID exists on every x86-64 CPU and only reads registers.
    let r = unsafe { core::arch::x86_64::__cpuid_count(function, index) };
    [r.eax, r.ebx, r.ecx, r.edx]
}

/// The host CPUID leaves the CPU model code looks at: vendor and version,
/// structured features, the XSAVE layout and the extended range.
fn host_leaves() -> Vec<CpuidEntry> {
    let mut v = Vec::new();
    let mut add = |f: u32, i: u32| v.push(CpuidEntry::new(f, i, host_cpuid_insn(f, i)));
    let max = host_cpuid_insn(0, 0)[0];
    add(0, 0);
    for f in 1..=max.min(0x24) {
        match f {
            7 => {
                for i in 0..=2 {
                    add(7, i);
                }
            }
            0xd => {
                for i in 0..64 {
                    add(0xd, i);
                }
            }
            _ => add(f, 0),
        }
    }
    let xmax = host_cpuid_insn(0x8000_0000, 0)[0];
    for f in 0x8000_0000..=xmax.min(0x8000_0021) {
        add(f, 0);
    }
    v
}

/// `kvm_arch_get_supported_msr_feature()` for every feature MSR: the raw
/// values `KVM_GET_MSRS` on the system fd returns.
fn feature_msrs(kvm: &Kvm) -> Result<Vec<(u32, u64)>, X86KvmError> {
    let list = match kvm.get_msr_feature_index_list() {
        Ok(l) => l,
        // Kernels before 4.17 have no feature MSRs at all.
        Err(_) => return Ok(Vec::new()),
    };
    let mut out = Vec::new();
    for &index in list.as_slice() {
        let entry = kvm_msr_entry { index, ..Default::default() };
        let mut msrs = Msrs::from_entries(&[entry]).map_err(|_| {
            X86KvmError::Config("KVM_GET_MSRS: cannot build the MSR list".to_string())
        })?;
        if kvm.get_msrs(&mut msrs).map_err(ioctl_err("KVM_GET_MSRS"))? == 1 {
            out.push((index, msrs.as_slice()[0].data));
        }
    }
    Ok(out)
}

/// `KVM_X86_GET_MCE_CAP_SUPPORTED`, or 0 without `KVM_CAP_MCE`.
fn mce_cap_supported(kvm: &Kvm) -> Result<u64, X86KvmError> {
    if !kvm.check_extension(Cap::Mce) {
        return Ok(0);
    }
    let mut supported = 0u64;
    #[allow(unsafe_code)]
    // SAFETY: the system fd is open for the life of `kvm`, and the ioctl
    // writes one u64 into `supported`, which is that size.
    let ret = unsafe {
        vmm_sys_util::ioctl::ioctl_with_mut_ref(kvm, KVM_X86_GET_MCE_CAP_SUPPORTED, &mut supported)
    };
    if ret < 0 {
        return Err(X86KvmError::Ioctl(
            "KVM_X86_GET_MCE_CAP_SUPPORTED",
            io::Error::last_os_error(),
        ));
    }
    Ok(supported)
}

fn irqchip_mode(k: KernelIrqchip) -> IrqchipMode {
    match k {
        KernelIrqchip::On => IrqchipMode::Full,
        KernelIrqchip::Split => IrqchipMode::Split,
        KernelIrqchip::Off => IrqchipMode::Off,
    }
}

/// Everything the CPU model code needs to know about the host:
/// `KVM_GET_SUPPORTED_CPUID`, the raw host CPUID, the feature MSRs and a
/// few capabilities. The quirks QEMU applies on top of the supported CPUID
/// (`kvm_arch_get_supported_cpuid()`) are applied when [`HostCpuid`] is
/// queried, so the lists here are the kernel's answers as is.
pub fn host_cpuid(accel: &KvmAccel) -> Result<HostCpuid, X86KvmError> {
    let kvm = accel.kvm();
    let cpuid = kvm
        .get_supported_cpuid(KVM_MAX_CPUID_ENTRIES)
        .map_err(ioctl_err("KVM_GET_SUPPORTED_CPUID"))?;
    let supported = cpuid
        .as_slice()
        .iter()
        .map(|e| CpuidEntry::new(e.function, e.index, [e.eax, e.ebx, e.ecx, e.edx]))
        .collect();
    let index_list = kvm.get_msr_index_list().map_err(ioctl_err("KVM_GET_MSR_INDEX_LIST"))?;
    let has = |m: u32| index_list.as_slice().contains(&m);
    Ok(HostCpuid {
        supported,
        host: host_leaves(),
        feature_msrs: feature_msrs(kvm)?,
        irqchip: irqchip_mode(accel.kernel_irqchip()),
        has_tsc_deadline: kvm.check_extension(Cap::TscDeadlineTimer),
        has_msr_arch_capabs: has(MSR_IA32_ARCH_CAPABILITIES),
        has_msr_vmx_procbased_ctls2: has(MSR_IA32_VMX_PROCBASED_CTLS2),
        xcomp_guest_supp: None,
        lmce_supported: mce_cap_supported(kvm)? & MCG_LMCE_P != 0,
    })
}

// Structure copies.

fn to_kvm_seg(s: &KvmSegment) -> kvm_segment {
    kvm_segment {
        base: s.base,
        limit: s.limit,
        selector: s.selector,
        type_: s.type_,
        present: s.present,
        dpl: s.dpl,
        db: s.db,
        s: s.s,
        l: s.l,
        g: s.g,
        avl: s.avl,
        unusable: s.unusable,
        padding: 0,
    }
}

fn from_kvm_seg(s: &kvm_segment) -> KvmSegment {
    KvmSegment {
        base: s.base,
        limit: s.limit,
        selector: s.selector,
        type_: s.type_,
        present: s.present,
        dpl: s.dpl,
        db: s.db,
        s: s.s,
        l: s.l,
        g: s.g,
        avl: s.avl,
        unusable: s.unusable,
    }
}

fn to_kvm_sregs(s: &KvmSregs) -> kvm_sregs {
    let dt = |d: &KvmDtable| kvm_dtable { base: d.base, limit: d.limit, padding: [0; 3] };
    kvm_sregs {
        cs: to_kvm_seg(&s.cs),
        ds: to_kvm_seg(&s.ds),
        es: to_kvm_seg(&s.es),
        fs: to_kvm_seg(&s.fs),
        gs: to_kvm_seg(&s.gs),
        ss: to_kvm_seg(&s.ss),
        tr: to_kvm_seg(&s.tr),
        ldt: to_kvm_seg(&s.ldt),
        gdt: dt(&s.gdt),
        idt: dt(&s.idt),
        cr0: s.cr0,
        cr2: s.cr2,
        cr3: s.cr3,
        cr4: s.cr4,
        cr8: s.cr8,
        efer: s.efer,
        apic_base: s.apic_base,
        interrupt_bitmap: [0; 4],
    }
}

fn from_kvm_sregs(s: &kvm_sregs) -> KvmSregs {
    let dt = |d: &kvm_dtable| KvmDtable { base: d.base, limit: d.limit };
    KvmSregs {
        cs: from_kvm_seg(&s.cs),
        ds: from_kvm_seg(&s.ds),
        es: from_kvm_seg(&s.es),
        fs: from_kvm_seg(&s.fs),
        gs: from_kvm_seg(&s.gs),
        ss: from_kvm_seg(&s.ss),
        tr: from_kvm_seg(&s.tr),
        ldt: from_kvm_seg(&s.ldt),
        gdt: dt(&s.gdt),
        idt: dt(&s.idt),
        cr0: s.cr0,
        cr2: s.cr2,
        cr3: s.cr3,
        cr4: s.cr4,
        cr8: s.cr8,
        efer: s.efer,
        apic_base: s.apic_base,
    }
}

fn to_kvm_events(e: &VcpuEvents) -> kvm_vcpu_events {
    let mut k = kvm_vcpu_events::default();
    k.exception.injected = e.exception_injected;
    k.exception.nr = e.exception_nr;
    k.exception.has_error_code = e.exception_has_error_code;
    k.exception.pending = e.exception_pending;
    k.exception.error_code = e.exception_error_code;
    k.exception_has_payload = e.exception_has_payload;
    k.exception_payload = e.exception_payload;
    k.interrupt.injected = e.interrupt_injected;
    k.interrupt.nr = e.interrupt_nr;
    k.interrupt.soft = e.interrupt_soft;
    k.nmi.injected = e.nmi_injected;
    k.nmi.pending = e.nmi_pending;
    k.nmi.masked = e.nmi_masked;
    k.sipi_vector = e.sipi_vector;
    k.flags = e.flags;
    k.smi.smm = e.smi_smm;
    k.smi.smm_inside_nmi = e.smi_smm_inside_nmi;
    k.triple_fault.pending = e.triple_fault_pending;
    k
}

fn from_kvm_events(k: &kvm_vcpu_events) -> VcpuEvents {
    VcpuEvents {
        flags: k.flags,
        exception_injected: k.exception.injected,
        exception_nr: k.exception.nr,
        exception_has_error_code: k.exception.has_error_code,
        exception_pending: k.exception.pending,
        exception_error_code: k.exception.error_code,
        exception_has_payload: k.exception_has_payload,
        exception_payload: k.exception_payload,
        interrupt_injected: k.interrupt.injected,
        interrupt_nr: k.interrupt.nr,
        interrupt_soft: k.interrupt.soft,
        nmi_injected: k.nmi.injected,
        nmi_pending: k.nmi.pending,
        nmi_masked: k.nmi.masked,
        sipi_vector: k.sipi_vector,
        smi_smm: k.smi.smm,
        smi_smm_inside_nmi: k.smi.smm_inside_nmi,
        triple_fault_pending: k.triple_fault.pending,
    }
}

fn to_kvm_cpuid(entries: &[KvmCpuidEntry]) -> Result<CpuId, X86KvmError> {
    let v: Vec<kvm_cpuid_entry2> = entries
        .iter()
        .map(|e| kvm_cpuid_entry2 {
            function: e.function,
            index: e.index,
            flags: e.flags,
            eax: e.regs[0],
            ebx: e.regs[1],
            ecx: e.regs[2],
            edx: e.regs[3],
            padding: [0; 3],
        })
        .collect();
    CpuId::from_entries(&v)
        .map_err(|_| X86KvmError::Config("KVM_SET_CPUID2: too many CPUID entries".to_string()))
}

fn msr_list(entries: &[(u32, u64)]) -> Result<Msrs, X86KvmError> {
    let v: Vec<kvm_msr_entry> =
        entries.iter().map(|&(index, data)| kvm_msr_entry { index, reserved: 0, data }).collect();
    Msrs::from_entries(&v)
        .map_err(|_| X86KvmError::Config("KVM_SET_MSRS: too many MSRs".to_string()))
}

/// `KVM_SET_MSRS` that must set every entry, as QEMU asserts.
fn set_msrs(vcpu: &VcpuFd, entries: &[(u32, u64)]) -> Result<(), X86KvmError> {
    if entries.is_empty() {
        return Ok(());
    }
    let msrs = msr_list(entries)?;
    let n = vcpu.set_msrs(&msrs).map_err(ioctl_err("KVM_SET_MSRS"))?;
    if n < entries.len() {
        let (index, value) = entries[n];
        return Err(X86KvmError::Config(format!(
            "error: failed to set MSR 0x{index:x} to 0x{value:x}"
        )));
    }
    Ok(())
}

// The vCPU.

/// Per-vCPU facts `kvm_arch_init_vcpu()` works out once and the register
/// sync needs afterwards.
#[derive(Debug, Clone)]
pub struct X86KvmVcpu {
    cpuid: Vec<KvmCpuidEntry>,
    msrs: MsrSupport,
    features: crate::cpuid::words::FeatureWordArray,
    mcg_cap: u64,
    tsc_khz: u32,
    user_tsc_khz: u32,
    has_xsave: bool,
    xsave_size: usize,
    has_xcrs: bool,
    has_tsc_control: bool,
    has_get_tsc_khz: bool,
    irqchip_in_kernel: bool,
    phys_bits: u32,
    pkru_offset: Option<usize>,
    apic_id: u32,
}

impl X86KvmVcpu {
    /// `kvm_arch_init_vcpu()`: set the TSC rate if the user asked for one,
    /// hand the CPUID table to KVM, set up machine checks, and find out
    /// which optional MSRs the vCPU has. `cpu` must be realized with
    /// [`crate::cpuid::Accel::Kvm`].
    pub fn init(accel: &KvmAccel, vcpu: &VcpuFd, cpu: &X86Cpu) -> Result<Self, X86KvmError> {
        let kvm = accel.kvm();
        let vm = accel.vm();

        let xsave2 = vm.check_extension_int(Cap::Xsave2);
        let has_xsave = kvm.check_extension(Cap::Xsave);
        let xsave_size =
            if xsave2 > 0 { (xsave2 as usize).max(KVM_XSAVE_SIZE) } else { KVM_XSAVE_SIZE };

        let user_tsc_khz = u32::try_from(cpu.tsc_khz().max(0)).unwrap_or(u32::MAX);
        let mut ctx = X86KvmVcpu {
            cpuid: Vec::new(),
            msrs: MsrSupport::default(),
            features: *cpu.features(),
            mcg_cap: cpu.mcg_cap(),
            tsc_khz: user_tsc_khz,
            user_tsc_khz,
            has_xsave,
            xsave_size,
            has_xcrs: kvm.check_extension(Cap::Xcrs),
            has_tsc_control: vm.check_extension(Cap::TscControl),
            has_get_tsc_khz: vm.check_extension(Cap::GetTscKhz),
            irqchip_in_kernel: accel.kernel_irqchip() != KernelIrqchip::Off,
            phys_bits: cpu.phys_bits(),
            pkru_offset: pkru_offset(cpu),
            apic_id: cpu.apic_id(),
        };

        ctx.set_tsc_khz(vcpu)?;
        if ctx.tsc_khz == 0 && ctx.has_get_tsc_khz {
            if let Ok(khz) = vcpu.get_tsc_khz() {
                ctx.tsc_khz = khz;
            }
        }

        ctx.cpuid = vcpu_cpuid(cpu, ctx.tsc_khz).map_err(|e| X86KvmError::Config(e.to_string()))?;

        if ctx.mcg_cap != 0 && wants_mce(cpu) {
            let banks = kvm.check_extension_int(Cap::Mce).max(0) as u64;
            let supported = mce_cap_supported(kvm)?;
            ctx.mcg_cap = negotiate_mcg_cap(ctx.mcg_cap, supported, banks)
                .map_err(|e| X86KvmError::Config(e.to_string()))?;
            let cap = ctx.mcg_cap;
            #[allow(unsafe_code)]
            // SAFETY: the vCPU fd is open for the life of `vcpu`, and the
            // ioctl reads one u64 from `cap`, which is that size.
            let ret = unsafe { vmm_sys_util::ioctl::ioctl_with_ref(vcpu, KVM_X86_SETUP_MCE, &cap) };
            if ret < 0 {
                return Err(X86KvmError::Ioctl("KVM_X86_SETUP_MCE", io::Error::last_os_error()));
            }
        }

        vcpu.set_cpuid2(&to_kvm_cpuid(&ctx.cpuid)?).map_err(ioctl_err("KVM_SET_CPUID2"))?;

        let index_list = kvm.get_msr_index_list().map_err(ioctl_err("KVM_GET_MSR_INDEX_LIST"))?;
        ctx.msrs = MsrSupport::from_index_list(index_list.as_slice()).for_cpu(
            cpu,
            &ctx.cpuid,
            ctx.mcg_cap,
        );

        set_msrs(vcpu, &init_msrs(cpu, &ctx.msrs))?;
        Ok(ctx)
    }

    /// `kvm_arch_set_tsc_khz()`. Does nothing without a user set rate. A
    /// failure is only an error when the vCPU is not already running at
    /// the requested rate.
    fn set_tsc_khz(&self, vcpu: &VcpuFd) -> Result<(), X86KvmError> {
        if self.user_tsc_khz == 0 {
            return Ok(());
        }
        let cur = if self.has_get_tsc_khz { vcpu.get_tsc_khz().ok() } else { None };
        let within = cur.is_some_and(|c| {
            c > 0 && freq_within_bounds(i64::from(c), i64::from(self.user_tsc_khz))
        });
        let r = if self.has_tsc_control || within {
            vcpu.set_tsc_khz(self.user_tsc_khz).map_err(ioctl_err("KVM_SET_TSC_KHZ"))
        } else {
            Err(X86KvmError::Ioctl("KVM_SET_TSC_KHZ", io::Error::from(io::ErrorKind::Unsupported)))
        };
        match r {
            Err(_) if cur == Some(self.user_tsc_khz) => Ok(()),
            Err(_) => Err(X86KvmError::Config(format!(
                "TSC frequency mismatch between VM ({} kHz) and host ({} kHz), and TSC scaling unavailable",
                self.user_tsc_khz,
                cur.unwrap_or(0)
            ))),
            Ok(()) => Ok(()),
        }
    }

    /// The CPUID table handed to `KVM_SET_CPUID2`.
    pub fn cpuid(&self) -> &[KvmCpuidEntry] {
        &self.cpuid
    }

    /// The optional MSRs this vCPU has.
    pub fn msr_support(&self) -> &MsrSupport {
        &self.msrs
    }

    /// `MCG_CAP` after matching it to what KVM supports.
    pub fn mcg_cap(&self) -> u64 {
        self.mcg_cap
    }

    /// The TSC rate in kHz, as set or as read back from KVM; 0 if unknown.
    pub fn tsc_khz(&self) -> u32 {
        self.tsc_khz
    }

    /// Size of the XSAVE area in bytes, `KVM_CAP_XSAVE2` or 4096.
    pub fn xsave_size(&self) -> usize {
        self.xsave_size
    }

    fn event_caps(&self) -> EventCaps {
        EventCaps {
            exception_payload: false,
            triple_fault_event: false,
            smm: self.msrs.has(MSR_IA32_SMBASE),
        }
    }

    /// `kvm_arch_put_registers()`: write `state` into the vCPU. `level`
    /// picks how much, as in QEMU: [`PutLevel::Runtime`] after an exit,
    /// [`PutLevel::Reset`] after reset, [`PutLevel::Full`] after loading a
    /// snapshot.
    pub fn put_registers(
        &self,
        vcpu: &VcpuFd,
        state: &X86CpuState,
        level: PutLevel,
    ) -> Result<(), X86KvmError> {
        if level >= PutLevel::Reset && self.msrs.feature_control {
            set_msrs(vcpu, &[(MSR_IA32_FEATURE_CONTROL, state.msr_ia32_feature_control)])?;
        }
        vcpu.set_sregs(&to_kvm_sregs(&sregs_from_state(state)))
            .map_err(ioctl_err("KVM_SET_SREGS"))?;
        if level == PutLevel::Full {
            // QEMU ignores errors here; migration checks the rate itself.
            let _ = self.set_tsc_khz(vcpu);
        }
        self.put_regs(vcpu, state)?;
        self.put_xsave(vcpu, state)?;
        if self.has_xcrs {
            let mut xcrs = kvm_xcrs { nr_xcrs: 1, ..Default::default() };
            xcrs.xcrs[0].xcr = 0;
            xcrs.xcrs[0].value = state.xcr0;
            vcpu.set_xcrs(&xcrs).map_err(ioctl_err("KVM_SET_XCRS"))?;
        }
        set_msrs(vcpu, &put_msrs(&self.msrs, &self.features, state, level, self.phys_bits))?;
        let events = events_from_state(state, level, self.event_caps());
        vcpu.set_vcpu_events(&to_kvm_events(&events)).map_err(ioctl_err("KVM_SET_VCPU_EVENTS"))?;
        if level >= PutLevel::Reset {
            vcpu.set_mp_state(kvm_mp_state { mp_state: state.mp_state })
                .map_err(ioctl_err("KVM_SET_MP_STATE"))?;
        }
        if self.msrs.has(MSR_IA32_TSCDEADLINE) {
            set_msrs(vcpu, &[(MSR_IA32_TSCDEADLINE, state.tsc_deadline)])?;
        }
        let (db, dr6, dr7) = debugregs_from_state(state);
        let dbg = kvm_debugregs { db, dr6, dr7, ..Default::default() };
        vcpu.set_debug_regs(&dbg).map_err(ioctl_err("KVM_SET_DEBUGREGS"))?;
        Ok(())
    }

    fn put_regs(&self, vcpu: &VcpuFd, state: &X86CpuState) -> Result<(), X86KvmError> {
        let r = &state.regs;
        let regs = kvm_regs {
            rax: r[R_EAX],
            rbx: r[R_EBX],
            rcx: r[R_ECX],
            rdx: r[R_EDX],
            rsi: r[R_ESI],
            rdi: r[R_EDI],
            rsp: r[R_ESP],
            rbp: r[R_EBP],
            r8: r[8],
            r9: r[9],
            r10: r[10],
            r11: r[11],
            r12: r[12],
            r13: r[13],
            r14: r[14],
            r15: r[15],
            rip: state.rip,
            rflags: state.rflags,
        };
        vcpu.set_regs(&regs).map_err(ioctl_err("KVM_SET_REGS"))
    }

    /// `kvm_put_xsave()`, or `kvm_put_fpu()` without `KVM_CAP_XSAVE`.
    ///
    /// The state does not hold the x87, SSE or AVX register contents, so the
    /// current area is read back first and only the fields the state models
    /// are replaced. Components past the first 4096 bytes are written as
    /// zero.
    fn put_xsave(&self, vcpu: &VcpuFd, state: &X86CpuState) -> Result<(), X86KvmError> {
        if !self.has_xsave {
            let mut fpu = vcpu.get_fpu().map_err(ioctl_err("KVM_GET_FPU"))?;
            let f: KvmFpu = fpu_from_state(state);
            fpu.fcw = f.fcw;
            fpu.fsw = f.fsw;
            fpu.ftwx = f.ftwx;
            fpu.mxcsr = f.mxcsr;
            return vcpu.set_fpu(&fpu).map_err(ioctl_err("KVM_SET_FPU"));
        }
        let mut kx = vcpu.get_xsave().map_err(ioctl_err("KVM_GET_XSAVE"))?;
        let ours = xsave_from_state(state, self.pkru_offset);
        let mut current = region_bytes(&kx.region);
        overlay_xsave(&mut current, &ours, self.pkru_offset);
        for (w, c) in kx.region.iter_mut().zip(current.chunks_exact(4)) {
            *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
        let mut xsave = Xsave::from_header(kvm_xsave2::from(kx))
            .map_err(|_| X86KvmError::Config("KVM_SET_XSAVE: bad buffer".to_string()))?;
        for _ in 0..(self.xsave_size - KVM_XSAVE_SIZE).div_ceil(4) {
            xsave
                .push(0)
                .map_err(|_| X86KvmError::Config("KVM_SET_XSAVE: bad buffer".to_string()))?;
        }
        #[allow(unsafe_code)]
        // SAFETY: `xsave` holds `xsave_size` bytes, the size KVM_CAP_XSAVE2
        // reported (or 4096 without it), which is what the kernel reads.
        let r = unsafe { vcpu.set_xsave2(&xsave) };
        r.map_err(ioctl_err("KVM_SET_XSAVE"))
    }

    /// `kvm_arch_get_registers()`: read the vCPU back into `state`.
    ///
    /// CR8 and the APIC base come from the special registers here; QEMU
    /// picks them up from `kvm_run` after each exit instead.
    pub fn get_registers(&self, vcpu: &VcpuFd, state: &mut X86CpuState) -> Result<(), X86KvmError> {
        let events = vcpu.get_vcpu_events().map_err(ioctl_err("KVM_GET_VCPU_EVENTS"))?;
        state_from_events(state, &from_kvm_events(&events));

        let mp = vcpu.get_mp_state().map_err(ioctl_err("KVM_GET_MP_STATE"))?;
        state.mp_state = mp.mp_state;
        if self.irqchip_in_kernel {
            state.halted = mp.mp_state == MP_STATE_HALTED;
        }

        let regs = vcpu.get_regs().map_err(ioctl_err("KVM_GET_REGS"))?;
        let r = &mut state.regs;
        r[R_EAX] = regs.rax;
        r[R_EBX] = regs.rbx;
        r[R_ECX] = regs.rcx;
        r[R_EDX] = regs.rdx;
        r[R_ESI] = regs.rsi;
        r[R_EDI] = regs.rdi;
        r[R_ESP] = regs.rsp;
        r[R_EBP] = regs.rbp;
        r[8] = regs.r8;
        r[9] = regs.r9;
        r[10] = regs.r10;
        r[11] = regs.r11;
        r[12] = regs.r12;
        r[13] = regs.r13;
        r[14] = regs.r14;
        r[15] = regs.r15;
        state.rip = regs.rip;
        state.rflags = regs.rflags;

        if self.has_xsave {
            let kx = vcpu.get_xsave().map_err(ioctl_err("KVM_GET_XSAVE"))?;
            state_from_xsave(state, &region_bytes(&kx.region), self.pkru_offset);
        } else {
            let fpu = vcpu.get_fpu().map_err(ioctl_err("KVM_GET_FPU"))?;
            let f = KvmFpu { fcw: fpu.fcw, fsw: fpu.fsw, ftwx: fpu.ftwx, mxcsr: fpu.mxcsr };
            state_from_fpu(state, &f);
        }

        if self.has_xcrs {
            let xcrs = vcpu.get_xcrs().map_err(ioctl_err("KVM_GET_XCRS"))?;
            let n = (xcrs.nr_xcrs as usize).min(xcrs.xcrs.len());
            for x in &xcrs.xcrs[..n] {
                if x.xcr == 0 {
                    state.xcr0 = x.value;
                }
            }
        }

        let sregs = vcpu.get_sregs().map_err(ioctl_err("KVM_GET_SREGS"))?;
        state_from_sregs(state, &from_kvm_sregs(&sregs));

        self.get_msrs(vcpu, state)?;

        let dbg = vcpu.get_debug_regs().map_err(ioctl_err("KVM_GET_DEBUGREGS"))?;
        state_from_debugregs(state, dbg.db, dbg.dr6, dbg.dr7);
        Ok(())
    }

    fn get_msrs(&self, vcpu: &VcpuFd, state: &mut X86CpuState) -> Result<(), X86KvmError> {
        let indices = get_msr_indices(&self.msrs, &self.features, state);
        let entries: Vec<(u32, u64)> = indices.iter().map(|&i| (i, 0)).collect();
        let mut msrs = msr_list(&entries)?;
        let n = vcpu.get_msrs(&mut msrs).map_err(ioctl_err("KVM_GET_MSRS"))?;
        if n < indices.len() {
            return Err(X86KvmError::Config(format!(
                "error: failed to get MSR 0x{:x}",
                indices[n]
            )));
        }
        for e in msrs.as_slice() {
            set_msr_value(state, e.index, e.data, self.phys_bits);
        }
        Ok(())
    }

    /// The local APIC half of a reset, `kvm_apic_put()` after
    /// `apic_reset_common()`: `IA32_APIC_BASE` from `state`, then the reset
    /// register page. Does nothing when the APIC is not in the kernel.
    pub fn put_lapic_reset(&self, vcpu: &VcpuFd, state: &X86CpuState) -> Result<(), X86KvmError> {
        if !self.irqchip_in_kernel {
            return Ok(());
        }
        set_msrs(vcpu, &[(MSR_IA32_APICBASE, state.apic_base)])?;
        let regs = lapic_reset_regs(self.apic_id);
        let mut lapic = kvm_lapic_state::default();
        for (d, s) in lapic.regs.iter_mut().zip(regs.iter()) {
            *d = *s as std::os::raw::c_char;
        }
        vcpu.set_lapic(&lapic).map_err(ioctl_err("KVM_SET_LAPIC"))
    }
}

fn region_bytes(region: &[u32; 1024]) -> Vec<u8> {
    region.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Copy the fields [`xsave_from_state`] fills in from `ours` over `current`:
/// FCW, FSW, the abridged FTW, MXCSR, `XSTATE_BV` and PKRU.
fn overlay_xsave(current: &mut [u8], ours: &[u8], pkru: Option<usize>) {
    for (off, len) in [(0, 6), (24, 4), (512, 8)] {
        current[off..off + len].copy_from_slice(&ours[off..off + len]);
    }
    if let Some(off) = pkru {
        if off + 4 <= current.len() {
            current[off..off + 4].copy_from_slice(&ours[off..off + 4]);
        }
    }
}

/// `kvm_arch_init_vcpu()` followed by the reset time register write: the
/// vCPU comes out holding `state` at [`PutLevel::Reset`], with a freshly
/// reset local APIC when the irqchip is in the kernel.
pub fn setup_vcpu(
    accel: &KvmAccel,
    vcpu: &VcpuFd,
    cpu: &X86Cpu,
    state: &X86CpuState,
) -> Result<X86KvmVcpu, X86KvmError> {
    let ctx = X86KvmVcpu::init(accel, vcpu, cpu)?;
    ctx.put_registers(vcpu, state, PutLevel::Reset)?;
    ctx.put_lapic_reset(vcpu, state)?;
    Ok(ctx)
}
