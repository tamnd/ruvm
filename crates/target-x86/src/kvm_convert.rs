// SPDX-License-Identifier: GPL-2.0-or-later

//! The host independent half of the KVM vCPU glue.
//!
//! `target/i386/kvm/kvm.c` mixes two kinds of work: deciding what goes into
//! a KVM structure, and the ioctl that hands it over. This module has the
//! first kind, written against small mirror structs instead of the
//! `kvm-bindings` types, so it builds and is tested on any host. The Linux
//! only `kvm` module copies these mirrors field by field into the real
//! structures and makes the calls.
//!
//! What lives here:
//!
//! - segment conversion ([`set_seg`], [`set_v8086_seg`], [`get_seg`]) and
//!   the special register block ([`KvmSregs`]), plus [`update_hflags`];
//! - the CPUID table for `KVM_SET_CPUID2` ([`vcpu_cpuid`]);
//! - which MSRs exist on the host ([`MsrSupport`]) and the MSR lists for
//!   `KVM_SET_MSRS` and `KVM_GET_MSRS`;
//! - the local APIC register page after reset ([`lapic_reset_regs`]);
//! - the legacy XSAVE region and the `kvm_fpu` fallback;
//! - `kvm_vcpu_events` in both directions.

use crate::cpuid::X86Cpu;
use crate::cpuid::cache::Regs;
use crate::cpuid::words::{
    CPUID_7_0_EBX_SGX, CPUID_7_0_ECX_CET_SHSTK, CPUID_7_0_ECX_SGX_LC, CPUID_7_0_EDX_CET_IBT,
    CPUID_APM_INVTSC, CPUID_EXT_VMX, CPUID_EXT2_RDTSCP, CPUID_MCA, CPUID_MCE, CPUID_MTRR,
    FEAT_1_EDX, FEAT_7_0_ECX, FEAT_7_0_EDX, FEAT_7_1_EAX, FEAT_8000_0001_EDX, FEAT_8000_0007_EDX,
    FEAT_KVM, FEAT_KVM_HINTS, FEAT_XSAVE, FeatureWordArray, KVM_CPUID_FEATURES,
};
use crate::msr::{
    MCG_LMCE_P, MSR_AMD64_TSC_RATIO, MSR_CSTAR, MSR_FMASK, MSR_IA32_ARCH_CAPABILITIES,
    MSR_IA32_BNDCFGS, MSR_IA32_CORE_CAPABILITY, MSR_IA32_FEATURE_CONTROL, MSR_IA32_FRED_CONFIG,
    MSR_IA32_FRED_RSP0, MSR_IA32_FRED_RSP1, MSR_IA32_FRED_RSP2, MSR_IA32_FRED_RSP3,
    MSR_IA32_FRED_SSP1, MSR_IA32_FRED_SSP2, MSR_IA32_FRED_SSP3, MSR_IA32_FRED_STKLVLS,
    MSR_IA32_INT_SSP_TAB, MSR_IA32_MISC_ENABLE, MSR_IA32_PERF_CAPABILITIES, MSR_IA32_PKRS,
    MSR_IA32_PL0_SSP, MSR_IA32_PL1_SSP, MSR_IA32_PL2_SSP, MSR_IA32_PL3_SSP, MSR_IA32_S_CET,
    MSR_IA32_SGXLEPUBKEYHASH0, MSR_IA32_SGXLEPUBKEYHASH1, MSR_IA32_SGXLEPUBKEYHASH2,
    MSR_IA32_SGXLEPUBKEYHASH3, MSR_IA32_SMBASE, MSR_IA32_SPEC_CTRL, MSR_IA32_SYSENTER_CS,
    MSR_IA32_SYSENTER_EIP, MSR_IA32_SYSENTER_ESP, MSR_IA32_TSC, MSR_IA32_TSCDEADLINE,
    MSR_IA32_TSX_CTRL, MSR_IA32_U_CET, MSR_IA32_UCODE_REV, MSR_IA32_UMWAIT_CONTROL,
    MSR_IA32_VMX_PROCBASED_CTLS2, MSR_IA32_VMX_VMFUNC, MSR_IA32_XFD, MSR_IA32_XFD_ERR,
    MSR_IA32_XSS, MSR_K7_HWCR, MSR_KERNELGSBASE, MSR_KVM_ASYNC_PF_EN, MSR_KVM_ASYNC_PF_INT,
    MSR_KVM_POLL_CONTROL, MSR_KVM_PV_EOI_EN, MSR_KVM_STEAL_TIME, MSR_KVM_SYSTEM_TIME,
    MSR_KVM_WALL_CLOCK, MSR_LSTAR, MSR_MC0_CTL, MSR_MCG_CTL, MSR_MCG_EXT_CTL, MSR_MCG_STATUS,
    MSR_MTRRDEFTYPE, MSR_PAT, MSR_SMI_COUNT, MSR_STAR, MSR_TSC_ADJUST, MSR_TSC_AUX, MSR_VIRT_SSBD,
    MSR_VM_HSAVE_PA, MTRR_FIXED_MSRS, MsrGate, RESET_MSRS, msr_mtrr_phys_base, msr_mtrr_phys_mask,
};
use crate::state::{
    CR0_PE_MASK, CR4_OSFXSR_MASK, DESC_AVL_MASK, DESC_B_MASK, DESC_B_SHIFT, DESC_DPL_SHIFT,
    DESC_G_MASK, DESC_L_MASK, DESC_P_MASK, DESC_S_MASK, DESC_TYPE_SHIFT, HF_ADDSEG_MASK,
    HF_CPL_MASK, HF_CS32_MASK, HF_CS64_MASK, HF_EM_MASK, HF_LMA_MASK, HF_MP_MASK, HF_PE_MASK,
    HF_SMM_MASK, HF_SS32_MASK, HF_TS_MASK, HF2_NMI_MASK, MSR_EFER_LMA, MTRR_VAR_COUNT, R_CS, R_DS,
    R_ES, R_FS, R_GS, R_SS, SegmentCache, X86CpuState,
};

/// How much state a register write carries, as `KvmPutState` in QEMU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PutLevel {
    /// Registers that change while the guest runs.
    Runtime = 1,
    /// Runtime state plus what a reset sets up.
    Reset = 2,
    /// Everything, as after an incoming migration.
    Full = 3,
}

/// `HF_TF_MASK`: the trap flag, copied from RFLAGS.
const HF_TF_MASK: u32 = 1 << 8;
/// `HF_IOPL_MASK`: the I/O privilege level, copied from RFLAGS.
const HF_IOPL_MASK: u32 = 3 << 12;
/// `HF_VM_MASK`: virtual 8086 mode, copied from RFLAGS.
const HF_VM_MASK: u32 = 1 << 17;
/// `HF_OSFXSR_MASK`: CR4.OSFXSR.
const HF_OSFXSR_MASK: u32 = 1 << 22;
/// `HF2_SMM_INSIDE_NMI_MASK`.
pub const HF2_SMM_INSIDE_NMI_MASK: u32 = 1 << 4;
/// `VM_MASK`: the virtual 8086 bit of RFLAGS.
pub const RFLAGS_VM_MASK: u64 = 1 << 17;

/// `CPUID_EXT_SMX`, `CPUID[1].ECX` bit 6.
const CPUID_EXT_SMX: u64 = 1 << 6;
/// `CPUID_7_1_EAX_FRED`.
const CPUID_7_1_EAX_FRED: u64 = 1 << 17;
/// `CPUID_D_1_EAX_XFD`.
const CPUID_D_1_EAX_XFD: u64 = 1 << 4;
/// `CPUID_KVM_CLOCK`.
pub const CPUID_KVM_CLOCK: u64 = 1 << 0;
/// `CPUID_KVM_CLOCK2`.
pub const CPUID_KVM_CLOCK2: u64 = 1 << 3;
/// `CPUID_KVM_ASYNCPF`.
pub const CPUID_KVM_ASYNCPF: u64 = 1 << 4;
/// `CPUID_KVM_STEAL_TIME`.
pub const CPUID_KVM_STEAL_TIME: u64 = 1 << 5;
/// `CPUID_KVM_PV_EOI`.
pub const CPUID_KVM_PV_EOI: u64 = 1 << 6;
/// `CPUID_KVM_POLL_CONTROL`.
pub const CPUID_KVM_POLL_CONTROL: u64 = 1 << 12;
/// `CPUID_KVM_ASYNCPF_INT`.
pub const CPUID_KVM_ASYNCPF_INT: u64 = 1 << 14;

/// `MCG_CAP_BANKS_MASK`: the bank count field of `MCG_CAP`.
pub const MCG_CAP_BANKS_MASK: u64 = 0xff;

// Segments.

/// `struct kvm_segment` with the same field widths.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvmSegment {
    /// Linear base address.
    pub base: u64,
    /// Limit in bytes.
    pub limit: u32,
    /// Selector.
    pub selector: u16,
    /// Descriptor type, four bits.
    pub type_: u8,
    /// Present bit.
    pub present: u8,
    /// Descriptor privilege level.
    pub dpl: u8,
    /// Default operation size (the B/D bit).
    pub db: u8,
    /// Code or data segment rather than a system one.
    pub s: u8,
    /// 64-bit code segment.
    pub l: u8,
    /// Granularity.
    pub g: u8,
    /// Available for software.
    pub avl: u8,
    /// KVM's "this segment cannot be used" bit.
    pub unusable: u8,
}

/// `set_seg()`: a cached descriptor as KVM wants it.
pub fn set_seg(rhs: &SegmentCache) -> KvmSegment {
    let flags = rhs.flags;
    let present = u8::from(flags & DESC_P_MASK != 0);
    KvmSegment {
        base: rhs.base,
        limit: rhs.limit,
        selector: rhs.selector as u16,
        type_: ((flags >> DESC_TYPE_SHIFT) & 15) as u8,
        present,
        dpl: ((flags >> DESC_DPL_SHIFT) & 3) as u8,
        db: ((flags >> DESC_B_SHIFT) & 1) as u8,
        s: u8::from(flags & DESC_S_MASK != 0),
        l: ((flags >> 21) & 1) as u8,
        g: u8::from(flags & DESC_G_MASK != 0),
        avl: u8::from(flags & DESC_AVL_MASK != 0),
        unusable: u8::from(present == 0),
    }
}

/// `set_v8086_seg()`: in virtual 8086 mode every data and code segment is
/// a present, writable, ring 3, 16-bit segment whatever the cache says.
pub fn set_v8086_seg(rhs: &SegmentCache) -> KvmSegment {
    KvmSegment {
        base: rhs.base,
        limit: rhs.limit,
        selector: rhs.selector as u16,
        type_: 3,
        present: 1,
        dpl: 3,
        db: 0,
        s: 1,
        l: 0,
        g: 0,
        avl: 0,
        unusable: 0,
    }
}

/// `get_seg()`: back from KVM into a cached descriptor. An unusable segment
/// comes back without the present bit.
pub fn get_seg(rhs: &KvmSegment) -> SegmentCache {
    let present = rhs.present != 0 && rhs.unusable == 0;
    let flags = (u32::from(rhs.type_) << DESC_TYPE_SHIFT)
        | if present { DESC_P_MASK } else { 0 }
        | (u32::from(rhs.dpl) << DESC_DPL_SHIFT)
        | (u32::from(rhs.db) << DESC_B_SHIFT)
        | if rhs.s != 0 { DESC_S_MASK } else { 0 }
        | (u32::from(rhs.l) << 21)
        | if rhs.g != 0 { DESC_G_MASK } else { 0 }
        | if rhs.avl != 0 { DESC_AVL_MASK } else { 0 };
    SegmentCache { selector: u32::from(rhs.selector), base: rhs.base, limit: rhs.limit, flags }
}

/// `struct kvm_dtable`, for the GDT and IDT.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvmDtable {
    /// Linear base address.
    pub base: u64,
    /// Limit.
    pub limit: u16,
}

/// `struct kvm_sregs` without the interrupt bitmap, which QEMU always
/// writes as zero because pending interrupts go through the vCPU events.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvmSregs {
    /// CS.
    pub cs: KvmSegment,
    /// DS.
    pub ds: KvmSegment,
    /// ES.
    pub es: KvmSegment,
    /// FS.
    pub fs: KvmSegment,
    /// GS.
    pub gs: KvmSegment,
    /// SS.
    pub ss: KvmSegment,
    /// Task register.
    pub tr: KvmSegment,
    /// Local descriptor table.
    pub ldt: KvmSegment,
    /// Global descriptor table.
    pub gdt: KvmDtable,
    /// Interrupt descriptor table.
    pub idt: KvmDtable,
    /// CR0.
    pub cr0: u64,
    /// CR2.
    pub cr2: u64,
    /// CR3.
    pub cr3: u64,
    /// CR4.
    pub cr4: u64,
    /// CR8, the task priority.
    pub cr8: u64,
    /// EFER.
    pub efer: u64,
    /// `IA32_APIC_BASE`.
    pub apic_base: u64,
}

/// The special registers `kvm_put_sregs()` writes for `state`.
///
/// In virtual 8086 mode the six segment registers go through
/// [`set_v8086_seg`]; TR and LDT always use [`set_seg`]. CR8 and the APIC
/// base come from the state, where QEMU reads them from its APIC model.
pub fn sregs_from_state(state: &X86CpuState) -> KvmSregs {
    let seg = if state.rflags & RFLAGS_VM_MASK != 0 { set_v8086_seg } else { set_seg };
    let dt = |c: &SegmentCache| KvmDtable { base: c.base, limit: c.limit as u16 };
    KvmSregs {
        cs: seg(&state.segs[R_CS]),
        ds: seg(&state.segs[R_DS]),
        es: seg(&state.segs[R_ES]),
        fs: seg(&state.segs[R_FS]),
        gs: seg(&state.segs[R_GS]),
        ss: seg(&state.segs[R_SS]),
        tr: set_seg(&state.tr),
        ldt: set_seg(&state.ldt),
        gdt: dt(&state.gdt),
        idt: dt(&state.idt),
        cr0: state.cr0,
        cr2: state.cr2,
        cr3: state.cr3,
        cr4: state.cr4,
        cr8: state.cr8,
        efer: state.efer,
        apic_base: state.apic_base,
    }
}

/// `kvm_get_sregs()`: load the special registers into `state` and
/// recompute `hflags`.
///
/// QEMU picks up CR8 and the APIC base after each exit from `kvm_run`
/// instead; here they come from the same block.
pub fn state_from_sregs(state: &mut X86CpuState, sregs: &KvmSregs) {
    state.segs[R_CS] = get_seg(&sregs.cs);
    state.segs[R_DS] = get_seg(&sregs.ds);
    state.segs[R_ES] = get_seg(&sregs.es);
    state.segs[R_FS] = get_seg(&sregs.fs);
    state.segs[R_GS] = get_seg(&sregs.gs);
    state.segs[R_SS] = get_seg(&sregs.ss);
    state.tr = get_seg(&sregs.tr);
    state.ldt = get_seg(&sregs.ldt);
    state.idt.limit = u32::from(sregs.idt.limit);
    state.idt.base = sregs.idt.base;
    state.gdt.limit = u32::from(sregs.gdt.limit);
    state.gdt.base = sregs.gdt.base;
    state.cr0 = sregs.cr0;
    state.cr2 = sregs.cr2;
    state.cr3 = sregs.cr3;
    state.cr4 = sregs.cr4;
    state.cr8 = sregs.cr8;
    state.apic_base = sregs.apic_base;
    state.efer = sregs.efer;
    update_hflags(state);
}

/// `x86_update_hflags()`: rebuild the mode bits of `hflags` from the
/// control registers, RFLAGS, EFER and the CS and SS descriptors.
pub fn update_hflags(state: &mut X86CpuState) {
    let copy_mask = !(HF_CPL_MASK
        | HF_PE_MASK
        | HF_MP_MASK
        | HF_EM_MASK
        | HF_TS_MASK
        | HF_TF_MASK
        | HF_VM_MASK
        | HF_IOPL_MASK
        | HF_OSFXSR_MASK
        | HF_LMA_MASK
        | HF_CS32_MASK
        | HF_SS32_MASK
        | HF_CS64_MASK
        | HF_ADDSEG_MASK);
    let cr0 = state.cr0 as u32;
    let eflags = state.rflags as u32;
    let mut hflags = state.hflags & copy_mask;
    hflags |= (state.segs[R_SS].flags >> DESC_DPL_SHIFT) & HF_CPL_MASK;
    hflags |= (cr0 & CR0_PE_MASK as u32) << 7;
    hflags |= (cr0 << 8) & (HF_MP_MASK | HF_EM_MASK | HF_TS_MASK);
    hflags |= eflags & (HF_TF_MASK | HF_VM_MASK | HF_IOPL_MASK);
    if state.cr4 & CR4_OSFXSR_MASK != 0 {
        hflags |= HF_OSFXSR_MASK;
    }
    if state.efer & MSR_EFER_LMA != 0 {
        hflags |= HF_LMA_MASK;
    }
    if hflags & HF_LMA_MASK != 0 && state.segs[R_CS].flags & DESC_L_MASK != 0 {
        hflags |= HF_CS32_MASK | HF_SS32_MASK | HF_CS64_MASK;
    } else {
        hflags |= (state.segs[R_CS].flags & DESC_B_MASK) >> (DESC_B_SHIFT - 4);
        hflags |= (state.segs[R_SS].flags & DESC_B_MASK) >> (DESC_B_SHIFT - 5);
        let flat = state.cr0 & CR0_PE_MASK != 0
            && state.rflags & RFLAGS_VM_MASK == 0
            && hflags & HF_CS32_MASK != 0
            && (state.segs[R_DS].base | state.segs[R_ES].base | state.segs[R_SS].base) == 0;
        if !flat {
            hflags |= HF_ADDSEG_MASK;
        }
    }
    state.hflags = hflags;
}

// CPUID.

/// `KVM_CPUID_FLAG_SIGNIFCANT_INDEX`: the entry is selected by ECX too.
pub const KVM_CPUID_FLAG_SIGNIFCANT_INDEX: u32 = 1 << 0;
/// `KVM_CPUID_FLAG_STATEFUL_FUNC`: repeated reads return different values.
pub const KVM_CPUID_FLAG_STATEFUL_FUNC: u32 = 1 << 1;
/// `KVM_CPUID_FLAG_STATE_READ_NEXT`: the next entry of a stateful leaf.
pub const KVM_CPUID_FLAG_STATE_READ_NEXT: u32 = 1 << 2;

/// Size of QEMU's CPUID table (`KVM_MAX_CPUID_ENTRIES` in `kvm_i386.h`).
pub const MAX_CPUID_ENTRIES: usize = 100;

/// `KVM_CPUID_SIGNATURE`.
pub const KVM_CPUID_SIGNATURE: u32 = 0x4000_0000;

/// `KVM_APIC_BUS_FREQUENCY`: KVM's APIC timer runs at 1 GHz.
pub const KVM_APIC_BUS_FREQUENCY: u64 = 1_000_000_000;

/// `struct kvm_cpuid_entry2`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvmCpuidEntry {
    /// Leaf.
    pub function: u32,
    /// Subleaf.
    pub index: u32,
    /// `KVM_CPUID_FLAG_*`.
    pub flags: u32,
    /// EAX, EBX, ECX, EDX.
    pub regs: Regs,
}

impl KvmCpuidEntry {
    fn new(function: u32, index: u32, flags: u32, regs: Regs) -> Self {
        Self { function, index, flags, regs }
    }
}

/// The table ran out of room. QEMU aborts here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuidTableFull {
    /// Leaf that did not fit.
    pub function: u32,
    /// Subleaf that did not fit.
    pub index: u32,
}

impl std::fmt::Display for CpuidTableFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cpuid_data is full, no space for cpuid(eax:0x{:x},ecx:0x{:x})",
            self.function, self.index
        )
    }
}

impl std::error::Error for CpuidTableFull {}

struct Table {
    entries: Vec<KvmCpuidEntry>,
}

impl Table {
    fn push(&mut self, e: KvmCpuidEntry) -> Result<(), CpuidTableFull> {
        if self.entries.len() == MAX_CPUID_ENTRIES {
            return Err(CpuidTableFull { function: e.function, index: e.index });
        }
        self.entries.push(e);
        Ok(())
    }
}

/// Inputs to [`build_cpuid`] beyond the CPUID function itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuidWalk {
    /// `x86_has_cpuid_0x1f()`.
    pub has_0x1f: bool,
    /// `env->cpuid_xlevel2`; the Centaur range is walked only when nonzero.
    pub xlevel2: u32,
}

/// `kvm_x86_build_cpuid()`: walk the basic, extended and Centaur ranges and
/// append one entry per leaf and subleaf to `entries`.
///
/// `cpuid` answers `cpu_x86_cpuid()` for a leaf and subleaf. Plain leaves
/// that are all zero are dropped, since KVM already answers zero for a leaf
/// it does not know.
pub fn build_cpuid(
    entries: Vec<KvmCpuidEntry>,
    cpuid: impl Fn(u32, u32) -> Regs,
    walk: CpuidWalk,
) -> Result<Vec<KvmCpuidEntry>, CpuidTableFull> {
    let mut t = Table { entries };
    let sig = KVM_CPUID_FLAG_SIGNIFCANT_INDEX;

    let limit = cpuid(0, 0)[0];
    for i in 0..=limit {
        match i {
            2 => {
                let r = cpuid(2, 0);
                let times = r[0] & 0xff;
                let flags = if times > 1 {
                    KVM_CPUID_FLAG_STATEFUL_FUNC | KVM_CPUID_FLAG_STATE_READ_NEXT
                } else {
                    0
                };
                t.push(KvmCpuidEntry::new(2, 0, flags, r))?;
                for _ in 1..times {
                    t.push(KvmCpuidEntry::new(2, 0, KVM_CPUID_FLAG_STATEFUL_FUNC, cpuid(2, 0)))?;
                }
            }
            0x1f if !walk.has_0x1f => {}
            4 | 0xb | 0xd | 0x1f => {
                let mut j = 0u32;
                loop {
                    let r = cpuid(i, j);
                    let e = KvmCpuidEntry::new(i, j, sig, r);
                    if i == 4 && r[0] == 0 {
                        t.push(e)?;
                        break;
                    }
                    if (i == 0xb || i == 0x1f) && r[2] & 0xff00 == 0 {
                        t.push(e)?;
                        break;
                    }
                    if i == 0xd && r[0] == 0 {
                        if j < 63 {
                            j += 1;
                            continue;
                        }
                        break;
                    }
                    t.push(e)?;
                    if i == 0xd && j == 63 {
                        break;
                    }
                    j += 1;
                }
            }
            0x12 => {
                let mut j = 0u32;
                loop {
                    let r = cpuid(i, j);
                    t.push(KvmCpuidEntry::new(i, j, sig, r))?;
                    if j > 1 && r[0] & 0xf != 1 {
                        break;
                    }
                    j += 1;
                }
            }
            7 | 0x14 | 0x1d | 0x1e | 0x24 => {
                let r = cpuid(i, 0);
                t.push(KvmCpuidEntry::new(i, 0, sig, r))?;
                for j in 1..=r[0] {
                    t.push(KvmCpuidEntry::new(i, j, sig, cpuid(i, j)))?;
                }
            }
            _ => {
                let r = cpuid(i, 0);
                if r != [0; 4] {
                    t.push(KvmCpuidEntry::new(i, 0, 0, r))?;
                }
            }
        }
    }

    let limit = cpuid(0x8000_0000, 0)[0];
    for i in 0x8000_0000..=limit {
        if i == 0x8000_001d {
            let mut j = 0u32;
            loop {
                let r = cpuid(i, j);
                t.push(KvmCpuidEntry::new(i, j, sig, r))?;
                if r[0] == 0 {
                    break;
                }
                j += 1;
            }
        } else {
            let r = cpuid(i, 0);
            if r != [0; 4] {
                t.push(KvmCpuidEntry::new(i, 0, 0, r))?;
            }
        }
    }

    if walk.xlevel2 > 0 {
        let limit = cpuid(0xC000_0000, 0)[0];
        for i in 0xC000_0000..=limit {
            t.push(KvmCpuidEntry::new(i, 0, 0, cpuid(i, 0)))?;
        }
    }
    Ok(t.entries)
}

/// `tsc_is_stable_and_known()`.
fn tsc_is_stable_and_known(cpu: &X86Cpu, tsc_khz: u32) -> bool {
    tsc_khz != 0
        && (cpu.features()[FEAT_8000_0007_EDX] & CPUID_APM_INVTSC != 0 || cpu.user_tsc_khz() != 0)
}

/// The whole table `kvm_arch_init_vcpu()` hands to `KVM_SET_CPUID2`.
///
/// With the `kvm` property on it starts with the KVM signature leaf and
/// the paravirtual feature leaf, then the walk of [`build_cpuid`] over
/// [`X86Cpu::cpuid`], then the VMware style frequency leaf 0x40000010 when
/// the TSC rate is stable and known. `tsc_khz` is the rate after
/// `KVM_SET_TSC_KHZ` or `KVM_GET_TSC_KHZ`. Hyper-V and Xen leaves are not
/// built.
pub fn vcpu_cpuid(cpu: &X86Cpu, tsc_khz: u32) -> Result<Vec<KvmCpuidEntry>, CpuidTableFull> {
    let mut entries = Vec::new();
    let expose_kvm = cpu.expose_kvm();
    if expose_kvm {
        let sig = *b"KVMKVMKVM\0\0\0";
        let word = |i: usize| u32::from_le_bytes([sig[i], sig[i + 1], sig[i + 2], sig[i + 3]]);
        entries.push(KvmCpuidEntry::new(
            KVM_CPUID_SIGNATURE,
            0,
            0,
            [KVM_CPUID_FEATURES, word(0), word(4), word(8)],
        ));
        let f = cpu.features();
        entries.push(KvmCpuidEntry::new(
            KVM_CPUID_FEATURES,
            0,
            0,
            [f[FEAT_KVM] as u32, 0, 0, f[FEAT_KVM_HINTS] as u32],
        ));
    }
    let walk = CpuidWalk { has_0x1f: cpu.has_cpuid_0x1f(), xlevel2: cpu.levels().2 };
    let mut entries = build_cpuid(entries, |i, j| cpu.cpuid(i, j), walk)?;

    if expose_kvm && tsc_is_stable_and_known(cpu, tsc_khz) {
        let e = KvmCpuidEntry::new(
            KVM_CPUID_SIGNATURE | 0x10,
            0,
            0,
            [tsc_khz, (KVM_APIC_BUS_FREQUENCY / 1000) as u32, 0, 0],
        );
        let mut t = Table { entries };
        t.push(e)?;
        entries = t.entries;
        if let Some(s) = entries.iter_mut().find(|e| e.function == KVM_CPUID_SIGNATURE) {
            s.regs[0] = s.regs[0].max(KVM_CPUID_SIGNATURE | 0x10);
        }
    }
    Ok(entries)
}

/// `freq_within_bounds()`: `target` is within 250 ppm of `cur`, the range
/// NTP can correct.
pub fn freq_within_bounds(cur: i64, target: i64) -> bool {
    let max = cur * (1_000_000 + 250) / 1_000_000;
    let min = cur * (1_000_000 - 250) / 1_000_000;
    target <= max && target >= min
}

// Machine checks.

/// The machine check setup could not be matched to what KVM offers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MceError {
    /// KVM has fewer banks than the CPU wants.
    Banks {
        /// Banks in the CPU's `MCG_CAP`.
        want: u64,
        /// Banks KVM offers.
        have: u64,
    },
    /// The CPU wants local machine checks and KVM has none.
    Lmce,
}

impl std::fmt::Display for MceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MceError::Banks { want, have } => {
                write!(f, "kvm: Unsupported MCE bank count (QEMU = {want}, KVM = {have})")
            }
            MceError::Lmce => f.write_str("kvm: LMCE not supported"),
        }
    }
}

impl std::error::Error for MceError {}

/// Whether `kvm_arch_init_vcpu()` sets up machine checks for this CPU:
/// family 6 or later with both MCE and MCA.
pub fn wants_mce(cpu: &X86Cpu) -> bool {
    let both = CPUID_MCE | CPUID_MCA;
    cpu.family() >= 6 && cpu.features()[FEAT_1_EDX] & both == both
}

/// The `MCG_CAP` to hand to `KVM_X86_SETUP_MCE`, given the CPU's value,
/// what `KVM_X86_GET_MCE_CAP_SUPPORTED` returned and the bank count from
/// `KVM_CAP_MCE`.
pub fn negotiate_mcg_cap(mcg_cap: u64, supported: u64, banks: u64) -> Result<u64, MceError> {
    if banks < mcg_cap & MCG_CAP_BANKS_MASK {
        return Err(MceError::Banks { want: mcg_cap & MCG_CAP_BANKS_MASK, have: banks });
    }
    let unsupported = mcg_cap & !(supported | MCG_CAP_BANKS_MASK);
    if unsupported & MCG_LMCE_P != 0 {
        return Err(MceError::Lmce);
    }
    Ok(mcg_cap & (supported | MCG_CAP_BANKS_MASK))
}

// MSRs.

/// The MSRs QEMU looks for in `KVM_GET_MSR_INDEX_LIST`.
const PROBED_MSRS: [u32; 23] = [
    MSR_STAR,
    MSR_VM_HSAVE_PA,
    MSR_TSC_AUX,
    MSR_TSC_ADJUST,
    MSR_IA32_TSCDEADLINE,
    MSR_IA32_SMBASE,
    MSR_SMI_COUNT,
    MSR_IA32_MISC_ENABLE,
    MSR_IA32_BNDCFGS,
    MSR_IA32_XSS,
    MSR_IA32_UMWAIT_CONTROL,
    MSR_IA32_SPEC_CTRL,
    MSR_AMD64_TSC_RATIO,
    MSR_IA32_TSX_CTRL,
    MSR_VIRT_SSBD,
    MSR_IA32_ARCH_CAPABILITIES,
    MSR_IA32_CORE_CAPABILITY,
    MSR_IA32_PERF_CAPABILITIES,
    MSR_IA32_VMX_VMFUNC,
    MSR_IA32_UCODE_REV,
    MSR_IA32_VMX_PROCBASED_CTLS2,
    MSR_IA32_PKRS,
    MSR_K7_HWCR,
];

/// Which optional MSRs a vCPU has, the `has_msr_*` globals of QEMU.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MsrSupport {
    /// The probed MSRs found in `KVM_GET_MSR_INDEX_LIST`, sorted.
    host: Vec<u32>,
    /// `has_msr_tsc_aux` after the RDTSCP check.
    pub tsc_aux: bool,
    /// `has_msr_feature_control`.
    pub feature_control: bool,
    /// `has_msr_mcg_ext_ctl`.
    pub mcg_ext_ctl: bool,
    /// `lm_capable_kernel`: the host kernel runs 64-bit guests.
    pub lm_capable_kernel: bool,
}

impl MsrSupport {
    /// `kvm_get_supported_msrs()`: keep the MSRs of interest from the
    /// host's index list. The per-CPU flags start out clear; see
    /// [`MsrSupport::for_cpu`].
    pub fn from_index_list(list: &[u32]) -> Self {
        let mut host: Vec<u32> = PROBED_MSRS.iter().copied().filter(|m| list.contains(m)).collect();
        host.sort_unstable();
        let tsc_aux = host.binary_search(&MSR_TSC_AUX).is_ok();
        Self { host, tsc_aux, feature_control: false, mcg_ext_ctl: false, lm_capable_kernel: true }
    }

    /// The flags `kvm_arch_init_vcpu()` derives for one CPU: feature
    /// control with VMX, SMX or SGX, the machine check extension control
    /// with LMCE, and no `TSC_AUX` without RDTSCP. `mcg_cap` is the value
    /// after [`negotiate_mcg_cap`], `cpuid` the table from [`vcpu_cpuid`].
    pub fn for_cpu(&self, cpu: &X86Cpu, cpuid: &[KvmCpuidEntry], mcg_cap: u64) -> Self {
        let mut s = self.clone();
        let find = |f: u32| cpuid.iter().find(|e| e.function == f && e.index == 0);
        if let Some(e) = find(1) {
            s.feature_control = u64::from(e.regs[2]) & (CPUID_EXT_VMX | CPUID_EXT_SMX) != 0;
        }
        if find(7).is_some_and(|e| u64::from(e.regs[1]) & CPUID_7_0_EBX_SGX != 0) {
            s.feature_control = true;
        }
        if mcg_cap & MCG_LMCE_P != 0 {
            s.mcg_ext_ctl = true;
            s.feature_control = true;
        }
        if cpu.features()[FEAT_8000_0001_EDX] & CPUID_EXT2_RDTSCP == 0 {
            s.tsc_aux = false;
        }
        s
    }

    /// Whether the host listed `index` (one of the probed MSRs).
    pub fn has(&self, index: u32) -> bool {
        if index == MSR_TSC_AUX {
            return self.tsc_aux;
        }
        self.host.binary_search(&index).is_ok()
    }
}

/// `kvm_init_msrs()`: the feature MSRs written once per vCPU.
///
/// The VMX capability MSRs and `PERF_CAPABILITIES` are not included; the
/// CPU models here do not expose VMX or a PMU.
pub fn init_msrs(cpu: &X86Cpu, support: &MsrSupport) -> Vec<(u32, u64)> {
    let f = cpu.features();
    let mut v = Vec::new();
    if support.has(MSR_IA32_ARCH_CAPABILITIES) {
        v.push((MSR_IA32_ARCH_CAPABILITIES, f[crate::cpuid::words::FEAT_ARCH_CAPABILITIES]));
    }
    if support.has(MSR_IA32_CORE_CAPABILITY) {
        v.push((MSR_IA32_CORE_CAPABILITY, f[crate::cpuid::words::FEAT_CORE_CAPABILITY]));
    }
    if support.has(MSR_IA32_UCODE_REV) {
        v.push((MSR_IA32_UCODE_REV, cpu.ucode_rev()));
    }
    v
}

fn gate_open(gate: MsrGate, support: &MsrSupport, f: &FeatureWordArray, mcg_cap: u64) -> bool {
    let shstk = f[FEAT_7_0_ECX] & CPUID_7_0_ECX_CET_SHSTK != 0;
    let fred = support.lm_capable_kernel && f[FEAT_7_1_EAX] & CPUID_7_1_EAX_FRED != 0;
    let kvm = f[FEAT_KVM];
    match gate {
        MsrGate::Always => true,
        MsrGate::HostHas(index) => support.has(index),
        MsrGate::LongModeKernel => support.lm_capable_kernel,
        MsrGate::Fred => fred,
        MsrGate::FredWithoutShadowStack => fred && !shstk,
        MsrGate::KvmClock => kvm & (CPUID_KVM_CLOCK | CPUID_KVM_CLOCK2) != 0,
        MsrGate::KvmAsyncPfInt => kvm & CPUID_KVM_ASYNCPF_INT != 0,
        MsrGate::KvmAsyncPf => kvm & CPUID_KVM_ASYNCPF != 0,
        MsrGate::KvmPvEoi => kvm & CPUID_KVM_PV_EOI != 0,
        MsrGate::KvmStealTime => kvm & CPUID_KVM_STEAL_TIME != 0,
        MsrGate::KvmPollControl => kvm & CPUID_KVM_POLL_CONTROL != 0,
        MsrGate::Mtrr => f[FEAT_1_EDX] & CPUID_MTRR != 0,
        MsrGate::SgxLc => f[FEAT_7_0_ECX] & CPUID_7_0_ECX_SGX_LC != 0,
        MsrGate::Xfd => f[FEAT_XSAVE] & CPUID_D_1_EAX_XFD != 0,
        MsrGate::Mce => mcg_cap != 0,
        MsrGate::MceExtCtl => mcg_cap != 0 && support.mcg_ext_ctl,
        MsrGate::Cet => shstk || f[FEAT_7_0_EDX] & CPUID_7_0_EDX_CET_IBT != 0,
        MsrGate::CetShadowStack => shstk,
        MsrGate::CetShadowStackLongMode => shstk && support.lm_capable_kernel,
    }
}

/// The MSR indices of `kvm_put_msrs()` and `kvm_get_msrs()`, in QEMU's put
/// order, with the machine check banks after `MCG_EXT_CTL`. `reset_only`
/// entries are included when `with_reset` is set.
fn msr_indices(
    support: &MsrSupport,
    f: &FeatureWordArray,
    mcg_cap: u64,
    with_reset: bool,
) -> Vec<u32> {
    let mut v = Vec::with_capacity(RESET_MSRS.len() + 40);
    let banks = |v: &mut Vec<u32>| {
        if mcg_cap != 0 {
            for i in 0..(mcg_cap & MCG_CAP_BANKS_MASK) as u32 * 4 {
                v.push(MSR_MC0_CTL + i);
            }
        }
    };
    let mut banks_done = false;
    for e in RESET_MSRS {
        if !banks_done && matches!(e.gate, MsrGate::Cet) {
            banks(&mut v);
            banks_done = true;
        }
        if e.reset_only && !with_reset {
            continue;
        }
        if gate_open(e.gate, support, f, mcg_cap) {
            v.push(e.index);
        }
    }
    if !banks_done {
        banks(&mut v);
    }
    v
}

/// `(1 << bits) - 1` for `bits` up to 64.
fn low_mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

/// `kvm_put_msrs()`: index and value pairs for `KVM_SET_MSRS`.
///
/// The TSC, the paravirtual KVM MSRs, the MTRRs, the SGX hashes and XFD go
/// out only at [`PutLevel::Reset`] and above. Variable MTRR masks are cut
/// to `phys_bits`, since the CPU faults on a bit above its address width.
pub fn put_msrs(
    support: &MsrSupport,
    features: &FeatureWordArray,
    state: &X86CpuState,
    level: PutLevel,
    phys_bits: u32,
) -> Vec<(u32, u64)> {
    let with_reset = level >= PutLevel::Reset;
    msr_indices(support, features, state.mcg_cap, with_reset)
        .into_iter()
        .map(|index| {
            let mut value = msr_value(state, index).unwrap_or(0);
            if is_mtrr_mask(index) {
                value &= low_mask(phys_bits);
            }
            (index, value)
        })
        .collect()
}

/// `kvm_get_msrs()`: the indices to read back. Adds the TSC deadline and
/// feature control MSRs, which QEMU writes on their own.
pub fn get_msr_indices(
    support: &MsrSupport,
    features: &FeatureWordArray,
    state: &X86CpuState,
) -> Vec<u32> {
    let mut v = msr_indices(support, features, state.mcg_cap, true);
    if support.has(MSR_IA32_TSCDEADLINE) {
        v.push(MSR_IA32_TSCDEADLINE);
    }
    if support.feature_control {
        v.push(MSR_IA32_FEATURE_CONTROL);
    }
    v
}

fn is_mtrr_mask(index: u32) -> bool {
    (msr_mtrr_phys_base(0)..=msr_mtrr_phys_mask(MTRR_VAR_COUNT as u32 - 1)).contains(&index)
        && index & 1 == 1
}

fn mtrr_var_slot(index: u32) -> Option<(usize, bool)> {
    let first = msr_mtrr_phys_base(0);
    let last = msr_mtrr_phys_mask(MTRR_VAR_COUNT as u32 - 1);
    if (first..=last).contains(&index) {
        Some((((index - first) / 2) as usize, index & 1 == 1))
    } else {
        None
    }
}

fn mce_bank_slot(state: &X86CpuState, index: u32) -> Option<usize> {
    let n = (state.mcg_cap & MCG_CAP_BANKS_MASK) as u32 * 4;
    if index >= MSR_MC0_CTL && index < MSR_MC0_CTL + n {
        let i = (index - MSR_MC0_CTL) as usize;
        (i < state.mce_banks.len()).then_some(i)
    } else {
        None
    }
}

/// The value of MSR `index` held in `state`, if the state tracks it.
pub fn msr_value(state: &X86CpuState, index: u32) -> Option<u64> {
    if let Some(i) = MTRR_FIXED_MSRS.iter().position(|&m| m == index) {
        return Some(state.mtrr_fixed[i]);
    }
    if let Some((i, mask)) = mtrr_var_slot(index) {
        let v = state.mtrr_var[i];
        return Some(if mask { v.mask } else { v.base });
    }
    if let Some(i) = mce_bank_slot(state, index) {
        return Some(state.mce_banks[i]);
    }
    let v = match index {
        MSR_IA32_SYSENTER_CS => u64::from(state.sysenter_cs),
        MSR_IA32_SYSENTER_ESP => state.sysenter_esp,
        MSR_IA32_SYSENTER_EIP => state.sysenter_eip,
        MSR_PAT => state.pat,
        MSR_STAR => state.star,
        MSR_VM_HSAVE_PA => state.vm_hsave,
        MSR_TSC_AUX => state.tsc_aux,
        MSR_TSC_ADJUST => state.tsc_adjust,
        MSR_IA32_TSCDEADLINE => state.tsc_deadline,
        MSR_IA32_MISC_ENABLE => state.msr_ia32_misc_enable,
        MSR_IA32_SMBASE => u64::from(state.smbase),
        MSR_SMI_COUNT => state.msr_smi_count,
        MSR_IA32_FEATURE_CONTROL => state.msr_ia32_feature_control,
        MSR_IA32_PKRS => u64::from(state.pkrs),
        MSR_IA32_BNDCFGS => state.msr_bndcfgs,
        MSR_IA32_XSS => state.xss,
        MSR_IA32_UMWAIT_CONTROL => u64::from(state.umwait),
        MSR_IA32_SPEC_CTRL => state.spec_ctrl,
        MSR_AMD64_TSC_RATIO => state.amd_tsc_scale_msr,
        MSR_IA32_TSX_CTRL => u64::from(state.tsx_ctrl),
        MSR_VIRT_SSBD => state.virt_ssbd,
        MSR_K7_HWCR => state.msr_hwcr,
        MSR_CSTAR => state.cstar,
        MSR_KERNELGSBASE => state.kernelgsbase,
        MSR_FMASK => state.fmask,
        MSR_LSTAR => state.lstar,
        MSR_IA32_FRED_RSP0 => state.fred_rsp[0],
        MSR_IA32_FRED_RSP1 => state.fred_rsp[1],
        MSR_IA32_FRED_RSP2 => state.fred_rsp[2],
        MSR_IA32_FRED_RSP3 => state.fred_rsp[3],
        MSR_IA32_FRED_STKLVLS => state.fred_stklvls,
        MSR_IA32_FRED_SSP1 => state.fred_ssp[0],
        MSR_IA32_FRED_SSP2 => state.fred_ssp[1],
        MSR_IA32_FRED_SSP3 => state.fred_ssp[2],
        MSR_IA32_FRED_CONFIG => state.fred_config,
        MSR_IA32_PL0_SSP => state.pl_ssp[0],
        MSR_IA32_PL1_SSP => state.pl_ssp[1],
        MSR_IA32_PL2_SSP => state.pl_ssp[2],
        MSR_IA32_PL3_SSP => state.pl_ssp[3],
        MSR_IA32_INT_SSP_TAB => state.int_ssp_table,
        MSR_IA32_U_CET => state.u_cet,
        MSR_IA32_S_CET => state.s_cet,
        MSR_IA32_TSC => state.tsc,
        MSR_KVM_SYSTEM_TIME => state.system_time_msr,
        MSR_KVM_WALL_CLOCK => state.wall_clock_msr,
        MSR_KVM_ASYNC_PF_INT => state.async_pf_int_msr,
        MSR_KVM_ASYNC_PF_EN => state.async_pf_en_msr,
        MSR_KVM_PV_EOI_EN => state.pv_eoi_en_msr,
        MSR_KVM_STEAL_TIME => state.steal_time_msr,
        MSR_KVM_POLL_CONTROL => state.poll_control_msr,
        MSR_MTRRDEFTYPE => state.mtrr_deftype,
        MSR_IA32_SGXLEPUBKEYHASH0 => state.msr_ia32_sgxlepubkeyhash[0],
        MSR_IA32_SGXLEPUBKEYHASH1 => state.msr_ia32_sgxlepubkeyhash[1],
        MSR_IA32_SGXLEPUBKEYHASH2 => state.msr_ia32_sgxlepubkeyhash[2],
        MSR_IA32_SGXLEPUBKEYHASH3 => state.msr_ia32_sgxlepubkeyhash[3],
        MSR_IA32_XFD => state.msr_xfd,
        MSR_IA32_XFD_ERR => state.msr_xfd_err,
        MSR_MCG_STATUS => state.mcg_status,
        MSR_MCG_CTL => state.mcg_ctl,
        MSR_MCG_EXT_CTL => state.mcg_ext_ctl,
        _ => return None,
    };
    Some(v)
}

/// The switch at the end of `kvm_get_msrs()`: store what `KVM_GET_MSRS`
/// returned. Variable MTRR masks get the bits from `phys_bits` up to bit 51
/// filled in, as QEMU does with `fill-mtrr-mask` on (the default). Returns
/// false for an MSR the state does not track.
pub fn set_msr_value(state: &mut X86CpuState, index: u32, data: u64, phys_bits: u32) -> bool {
    if let Some(i) = MTRR_FIXED_MSRS.iter().position(|&m| m == index) {
        state.mtrr_fixed[i] = data;
        return true;
    }
    if let Some((i, mask)) = mtrr_var_slot(index) {
        if mask {
            let bits = phys_bits.min(52);
            let top = low_mask(52) & !low_mask(bits);
            state.mtrr_var[i].mask = data | top;
        } else {
            state.mtrr_var[i].base = data;
        }
        return true;
    }
    if let Some(i) = mce_bank_slot(state, index) {
        state.mce_banks[i] = data;
        return true;
    }
    match index {
        MSR_IA32_SYSENTER_CS => state.sysenter_cs = data as u32,
        MSR_IA32_SYSENTER_ESP => state.sysenter_esp = data,
        MSR_IA32_SYSENTER_EIP => state.sysenter_eip = data,
        MSR_PAT => state.pat = data,
        MSR_STAR => state.star = data,
        MSR_VM_HSAVE_PA => state.vm_hsave = data,
        MSR_TSC_AUX => state.tsc_aux = data,
        MSR_TSC_ADJUST => state.tsc_adjust = data,
        MSR_IA32_TSCDEADLINE => state.tsc_deadline = data,
        MSR_IA32_MISC_ENABLE => state.msr_ia32_misc_enable = data,
        MSR_IA32_SMBASE => state.smbase = data as u32,
        MSR_SMI_COUNT => state.msr_smi_count = data,
        MSR_IA32_FEATURE_CONTROL => state.msr_ia32_feature_control = data,
        MSR_IA32_PKRS => state.pkrs = data as u32,
        MSR_IA32_BNDCFGS => state.msr_bndcfgs = data,
        MSR_IA32_XSS => state.xss = data,
        MSR_IA32_UMWAIT_CONTROL => state.umwait = data as u32,
        MSR_IA32_SPEC_CTRL => state.spec_ctrl = data,
        MSR_AMD64_TSC_RATIO => state.amd_tsc_scale_msr = data,
        MSR_IA32_TSX_CTRL => state.tsx_ctrl = data as u32,
        MSR_VIRT_SSBD => state.virt_ssbd = data,
        MSR_K7_HWCR => state.msr_hwcr = data,
        MSR_CSTAR => state.cstar = data,
        MSR_KERNELGSBASE => state.kernelgsbase = data,
        MSR_FMASK => state.fmask = data,
        MSR_LSTAR => state.lstar = data,
        MSR_IA32_FRED_RSP0 => state.fred_rsp[0] = data,
        MSR_IA32_FRED_RSP1 => state.fred_rsp[1] = data,
        MSR_IA32_FRED_RSP2 => state.fred_rsp[2] = data,
        MSR_IA32_FRED_RSP3 => state.fred_rsp[3] = data,
        MSR_IA32_FRED_STKLVLS => state.fred_stklvls = data,
        MSR_IA32_FRED_SSP1 => state.fred_ssp[0] = data,
        MSR_IA32_FRED_SSP2 => state.fred_ssp[1] = data,
        MSR_IA32_FRED_SSP3 => state.fred_ssp[2] = data,
        MSR_IA32_FRED_CONFIG => state.fred_config = data,
        MSR_IA32_PL0_SSP => state.pl_ssp[0] = data,
        MSR_IA32_PL1_SSP => state.pl_ssp[1] = data,
        MSR_IA32_PL2_SSP => state.pl_ssp[2] = data,
        MSR_IA32_PL3_SSP => state.pl_ssp[3] = data,
        MSR_IA32_INT_SSP_TAB => state.int_ssp_table = data,
        MSR_IA32_U_CET => state.u_cet = data,
        MSR_IA32_S_CET => state.s_cet = data,
        MSR_IA32_TSC => state.tsc = data,
        MSR_KVM_SYSTEM_TIME => state.system_time_msr = data,
        MSR_KVM_WALL_CLOCK => state.wall_clock_msr = data,
        MSR_KVM_ASYNC_PF_INT => state.async_pf_int_msr = data,
        MSR_KVM_ASYNC_PF_EN => state.async_pf_en_msr = data,
        MSR_KVM_PV_EOI_EN => state.pv_eoi_en_msr = data,
        MSR_KVM_STEAL_TIME => state.steal_time_msr = data,
        MSR_KVM_POLL_CONTROL => state.poll_control_msr = data,
        MSR_MTRRDEFTYPE => state.mtrr_deftype = data,
        MSR_IA32_SGXLEPUBKEYHASH0 => state.msr_ia32_sgxlepubkeyhash[0] = data,
        MSR_IA32_SGXLEPUBKEYHASH1 => state.msr_ia32_sgxlepubkeyhash[1] = data,
        MSR_IA32_SGXLEPUBKEYHASH2 => state.msr_ia32_sgxlepubkeyhash[2] = data,
        MSR_IA32_SGXLEPUBKEYHASH3 => state.msr_ia32_sgxlepubkeyhash[3] = data,
        MSR_IA32_XFD => state.msr_xfd = data,
        MSR_IA32_XFD_ERR => state.msr_xfd_err = data,
        MSR_MCG_STATUS => state.mcg_status = data,
        MSR_MCG_CTL => state.mcg_ctl = data,
        MSR_MCG_EXT_CTL => state.mcg_ext_ctl = data,
        _ => return false,
    }
    true
}

// Local APIC.

/// Size of `kvm_lapic_state.regs`.
pub const KVM_APIC_REG_SIZE: usize = 1024;

/// `APIC_LVT_NB`: timer, thermal, performance, LINT0, LINT1, error.
pub const APIC_LVT_NB: usize = 6;

/// `APIC_LVT_MASKED`.
pub const APIC_LVT_MASKED: u32 = 1 << 16;

/// `kvm_apic_set_reg()`: register `reg` lives at byte `reg * 16`.
pub fn lapic_set_reg(regs: &mut [u8; KVM_APIC_REG_SIZE], reg: usize, value: u32) {
    regs[reg << 4..(reg << 4) + 4].copy_from_slice(&value.to_le_bytes());
}

/// `kvm_apic_get_reg()`.
pub fn lapic_reg(regs: &[u8; KVM_APIC_REG_SIZE], reg: usize) -> u32 {
    let b = &regs[reg << 4..(reg << 4) + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The register page `kvm_put_apic_state()` builds after
/// `apic_init_reset()`, in xAPIC format.
///
/// The ID sits in the top byte, TPR, LDR, ESR, ICR, ISR, TMR, IRR and the
/// timer are zero, DFR is flat mode (all ones), the spurious vector is 0xff
/// with the APIC software disabled, and every LVT entry is masked.
pub fn lapic_reset_regs(apic_id: u32) -> [u8; KVM_APIC_REG_SIZE] {
    let mut r = [0u8; KVM_APIC_REG_SIZE];
    lapic_set_reg(&mut r, 0x2, apic_id << 24);
    lapic_set_reg(&mut r, 0x8, 0);
    lapic_set_reg(&mut r, 0xd, 0);
    lapic_set_reg(&mut r, 0xe, (0xf << 28) | 0x0fff_ffff);
    lapic_set_reg(&mut r, 0xf, 0xff);
    for i in 0..APIC_LVT_NB {
        lapic_set_reg(&mut r, 0x32 + i, APIC_LVT_MASKED);
    }
    lapic_set_reg(&mut r, 0x38, 0);
    lapic_set_reg(&mut r, 0x3e, 0);
    r
}

// FPU and XSAVE.

/// Number of 32-bit words in the legacy `kvm_xsave` region.
pub const KVM_XSAVE_WORDS: usize = 1024;

/// Index of the PKRU component in [`X86Cpu::ext_save_areas`].
const XSTATE_PKRU_BIT: usize = 9;

/// Offset of PKRU in the XSAVE area, when the CPU has the component.
pub fn pkru_offset(cpu: &X86Cpu) -> Option<usize> {
    let e = &cpu.ext_save_areas()[XSTATE_PKRU_BIT];
    (e.size != 0 && e.offset != 0).then_some(e.offset as usize)
}

fn fsw(state: &X86CpuState) -> u16 {
    (state.fpus & !(7 << 11)) | (((state.fpstt & 7) as u16) << 11)
}

/// The abridged tag word: bit i set when register i is in use.
fn abridged_ftw(state: &X86CpuState) -> u8 {
    let mut twd = 0u8;
    for (i, &tag) in state.fptags.iter().enumerate() {
        if tag == 0 {
            twd |= 1 << i;
        }
    }
    twd
}

fn put_bytes(buf: &mut [u8], off: usize, b: &[u8]) {
    buf[off..off + b.len()].copy_from_slice(b);
}

/// `x86_cpu_xsave_all_areas()` for the state this crate models: the x87
/// control, status and tag words, MXCSR, the XSAVE header's `XSTATE_BV`,
/// and PKRU. Register contents (x87 stack, XMM and above) are not modelled
/// and stay zero.
pub fn xsave_from_state(state: &X86CpuState, pkru_offset: Option<usize>) -> Vec<u8> {
    let mut buf = vec![0u8; KVM_XSAVE_WORDS * 4];
    put_bytes(&mut buf, 0, &state.fpuc.to_le_bytes());
    put_bytes(&mut buf, 2, &fsw(state).to_le_bytes());
    // The legacy area holds the abridged tag in a 16-bit slot.
    put_bytes(&mut buf, 4, &u16::from(abridged_ftw(state)).to_le_bytes());
    put_bytes(&mut buf, 24, &state.mxcsr.to_le_bytes());
    put_bytes(&mut buf, 512, &state.xstate_bv.to_le_bytes());
    if let Some(off) = pkru_offset {
        if off + 4 <= buf.len() {
            put_bytes(&mut buf, off, &state.pkru.to_le_bytes());
        }
    }
    buf
}

/// `x86_cpu_xrstor_all_areas()` for the modelled fields.
pub fn state_from_xsave(state: &mut X86CpuState, buf: &[u8], pkru_offset: Option<usize>) {
    let u16_at = |o: usize| u16::from_le_bytes([buf[o], buf[o + 1]]);
    let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    let cwd = u16_at(0);
    let swd = u16_at(2);
    let twd = u16_at(4);
    state.fpuc = cwd;
    state.fpstt = u32::from((swd >> 11) & 7);
    state.fpus = swd;
    for i in 0..8 {
        state.fptags[i] = u8::from((twd >> i) & 1 == 0);
    }
    state.mxcsr = u32_at(24);
    state.xstate_bv = u64::from(u32_at(512)) | (u64::from(u32_at(516)) << 32);
    if let Some(off) = pkru_offset {
        if off + 4 <= buf.len() {
            state.pkru = u32_at(off);
        }
    }
}

/// The parts of `struct kvm_fpu` that the state models, for hosts without
/// `KVM_CAP_XSAVE`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KvmFpu {
    /// Control word.
    pub fcw: u16,
    /// Status word with the top of stack.
    pub fsw: u16,
    /// Abridged tag word.
    pub ftwx: u8,
    /// MXCSR.
    pub mxcsr: u32,
}

/// `kvm_put_fpu()`.
pub fn fpu_from_state(state: &X86CpuState) -> KvmFpu {
    KvmFpu { fcw: state.fpuc, fsw: fsw(state), ftwx: abridged_ftw(state), mxcsr: state.mxcsr }
}

/// `kvm_get_fpu()`.
pub fn state_from_fpu(state: &mut X86CpuState, fpu: &KvmFpu) {
    state.fpstt = u32::from((fpu.fsw >> 11) & 7);
    state.fpus = fpu.fsw;
    state.fpuc = fpu.fcw;
    for i in 0..8 {
        state.fptags[i] = u8::from((fpu.ftwx >> i) & 1 == 0);
    }
    state.mxcsr = fpu.mxcsr;
}

// vCPU events.

/// `KVM_VCPUEVENT_VALID_NMI_PENDING`.
pub const KVM_VCPUEVENT_VALID_NMI_PENDING: u32 = 1 << 0;
/// `KVM_VCPUEVENT_VALID_SIPI_VECTOR`.
pub const KVM_VCPUEVENT_VALID_SIPI_VECTOR: u32 = 1 << 1;
/// `KVM_VCPUEVENT_VALID_SMM`.
pub const KVM_VCPUEVENT_VALID_SMM: u32 = 1 << 3;
/// `KVM_VCPUEVENT_VALID_PAYLOAD`.
pub const KVM_VCPUEVENT_VALID_PAYLOAD: u32 = 1 << 4;
/// `KVM_VCPUEVENT_VALID_TRIPLE_FAULT`.
pub const KVM_VCPUEVENT_VALID_TRIPLE_FAULT: u32 = 1 << 5;

/// `KVM_MP_STATE_SIPI_RECEIVED`.
pub const MP_STATE_SIPI_RECEIVED: u32 = 4;
/// `KVM_MP_STATE_HALTED`.
pub const MP_STATE_HALTED: u32 = 3;

/// The fields of `struct kvm_vcpu_events` QEMU fills in.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VcpuEvents {
    /// `KVM_VCPUEVENT_VALID_*`.
    pub flags: u32,
    /// `exception.injected`.
    pub exception_injected: u8,
    /// `exception.nr`.
    pub exception_nr: u8,
    /// `exception.has_error_code`.
    pub exception_has_error_code: u8,
    /// `exception.pending`, only meaningful with the payload flag.
    pub exception_pending: u8,
    /// `exception.error_code`.
    pub exception_error_code: u32,
    /// `exception_has_payload`.
    pub exception_has_payload: u8,
    /// `exception_payload`.
    pub exception_payload: u64,
    /// `interrupt.injected`.
    pub interrupt_injected: u8,
    /// `interrupt.nr`.
    pub interrupt_nr: u8,
    /// `interrupt.soft`.
    pub interrupt_soft: u8,
    /// `nmi.injected`.
    pub nmi_injected: u8,
    /// `nmi.pending`.
    pub nmi_pending: u8,
    /// `nmi.masked`.
    pub nmi_masked: u8,
    /// `sipi_vector`.
    pub sipi_vector: u32,
    /// `smi.smm`.
    pub smi_smm: u8,
    /// `smi.smm_inside_nmi`.
    pub smi_smm_inside_nmi: u8,
    /// `triple_fault.pending`.
    pub triple_fault_pending: u8,
}

/// Optional parts of the vCPU event block that depend on the host.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EventCaps {
    /// `KVM_CAP_EXCEPTION_PAYLOAD` is enabled on the VM.
    pub exception_payload: bool,
    /// `KVM_CAP_X86_TRIPLE_FAULT_EVENT` is enabled on the VM.
    pub triple_fault_event: bool,
    /// `has_msr_smbase`: the host has SMM, so SMM state is exchanged.
    pub smm: bool,
}

/// `kvm_put_vcpu_events()`. SMIs and INITs latched in user space are not
/// modelled, so `smi.pending` and `smi.latched_init` stay clear.
pub fn events_from_state(state: &X86CpuState, level: PutLevel, caps: EventCaps) -> VcpuEvents {
    let mut ev = VcpuEvents::default();
    if caps.exception_payload {
        ev.flags |= KVM_VCPUEVENT_VALID_PAYLOAD;
        ev.exception_pending = u8::from(state.exception_pending);
        ev.exception_has_payload = u8::from(state.exception_has_payload);
        ev.exception_payload = state.exception_payload;
    }
    ev.exception_nr = state.exception_nr as u8;
    ev.exception_injected = u8::from(state.exception_injected);
    ev.exception_has_error_code = u8::from(state.has_error_code);
    ev.exception_error_code = state.error_code;

    ev.interrupt_injected = u8::from(state.interrupt_injected >= 0);
    ev.interrupt_nr = state.interrupt_injected as u8;
    ev.interrupt_soft = u8::from(state.soft_interrupt);

    ev.nmi_injected = u8::from(state.nmi_injected);
    ev.nmi_pending = u8::from(state.nmi_pending);
    ev.nmi_masked = u8::from(state.hflags2 & HF2_NMI_MASK != 0);

    ev.sipi_vector = state.sipi_vector;

    if caps.smm {
        ev.flags |= KVM_VCPUEVENT_VALID_SMM;
        ev.smi_smm = u8::from(state.hflags & HF_SMM_MASK != 0);
        ev.smi_smm_inside_nmi = u8::from(state.hflags2 & HF2_SMM_INSIDE_NMI_MASK != 0);
    }

    if level >= PutLevel::Reset {
        ev.flags |= KVM_VCPUEVENT_VALID_NMI_PENDING;
        if state.mp_state == MP_STATE_SIPI_RECEIVED {
            ev.flags |= KVM_VCPUEVENT_VALID_SIPI_VECTOR;
        }
    }

    if caps.triple_fault_event {
        ev.flags |= KVM_VCPUEVENT_VALID_TRIPLE_FAULT;
        ev.triple_fault_pending = u8::from(state.triple_fault_pending);
    }
    ev
}

/// `kvm_get_vcpu_events()`.
pub fn state_from_events(state: &mut X86CpuState, ev: &VcpuEvents) {
    if ev.flags & KVM_VCPUEVENT_VALID_PAYLOAD != 0 {
        state.exception_pending = ev.exception_pending != 0;
        state.exception_has_payload = ev.exception_has_payload != 0;
        state.exception_payload = ev.exception_payload;
    } else {
        state.exception_pending = false;
        state.exception_has_payload = false;
    }
    state.exception_injected = ev.exception_injected != 0;
    state.exception_nr = if state.exception_pending || state.exception_injected {
        i32::from(ev.exception_nr)
    } else {
        -1
    };
    state.has_error_code = ev.exception_has_error_code != 0;
    state.error_code = ev.exception_error_code;

    state.interrupt_injected =
        if ev.interrupt_injected != 0 { i32::from(ev.interrupt_nr) } else { -1 };
    state.soft_interrupt = ev.interrupt_soft != 0;

    state.nmi_injected = ev.nmi_injected != 0;
    state.nmi_pending = ev.nmi_pending != 0;
    if ev.nmi_masked != 0 {
        state.hflags2 |= HF2_NMI_MASK;
    } else {
        state.hflags2 &= !HF2_NMI_MASK;
    }

    if ev.flags & KVM_VCPUEVENT_VALID_SMM != 0 {
        if ev.smi_smm != 0 {
            state.hflags |= HF_SMM_MASK;
        } else {
            state.hflags &= !HF_SMM_MASK;
        }
        if ev.smi_smm_inside_nmi != 0 {
            state.hflags2 |= HF2_SMM_INSIDE_NMI_MASK;
        } else {
            state.hflags2 &= !HF2_SMM_INSIDE_NMI_MASK;
        }
    }

    if ev.flags & KVM_VCPUEVENT_VALID_TRIPLE_FAULT != 0 {
        state.triple_fault_pending = ev.triple_fault_pending != 0;
    }
    state.sipi_vector = ev.sipi_vector;
}

/// The debug registers `kvm_put_debugregs()` writes: DR0 to DR3, DR6, DR7.
pub fn debugregs_from_state(state: &X86CpuState) -> ([u64; 4], u64, u64) {
    ([state.dr[0], state.dr[1], state.dr[2], state.dr[3]], state.dr[6], state.dr[7])
}

/// `kvm_get_debugregs()`: DR4 and DR5 alias DR6 and DR7.
pub fn state_from_debugregs(state: &mut X86CpuState, db: [u64; 4], dr6: u64, dr7: u64) {
    state.dr[..4].copy_from_slice(&db);
    state.dr[4] = dr6;
    state.dr[6] = dr6;
    state.dr[5] = dr7;
    state.dr[7] = dr7;
}

#[cfg(test)]
mod tests;
