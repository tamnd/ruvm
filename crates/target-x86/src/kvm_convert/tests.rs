// SPDX-License-Identifier: GPL-2.0-or-later

//! Host independent tests for the KVM conversions. Expected values come
//! from reading `target/i386/kvm/kvm.c`, `target/i386/helper.c` and
//! `hw/intc/apic_common.c` in QEMU 11.1.

use super::*;
use crate::cpuid::Accel;
use crate::cpuid::host::{CpuidEntry, HostCpuid};
use crate::msr::{MSR_IA32_APICBASE_BSP, MSR_IA32_APICBASE_ENABLE, MSR_MC0_STATUS};
use crate::state::{HF_SMM_MASK, IrqchipMode, MP_STATE_RUNNABLE};

fn tcg_cpu(model: &str) -> X86Cpu {
    let mut cpu = X86Cpu::new(model, Accel::Tcg).unwrap();
    cpu.realize().unwrap();
    cpu
}

/// A GenuineIntel host where KVM allows every bit of the leaves the CPU
/// models use.
fn permissive_host() -> HostCpuid {
    let all = [u32::MAX; 4];
    let mut supported = Vec::new();
    for (f, i) in [
        (1, 0),
        (6, 0),
        (7, 0),
        (7, 1),
        (7, 2),
        (0xd, 0),
        (0xd, 1),
        (0x8000_0001, 0),
        (0x8000_0007, 0),
        (0x8000_0008, 0),
        (0x8000_000a, 0),
        (KVM_CPUID_FEATURES, 0),
    ] {
        supported.push(CpuidEntry::new(f, i, all));
    }
    let host = vec![
        CpuidEntry::new(0, 0, [0x16, 0x756e_6547, 0x6c65_746e, 0x4965_6e69]),
        CpuidEntry::new(1, 0, [0x0005_06e3, 0, 0, 0]),
        CpuidEntry::new(0x8000_0000, 0, [0x8000_0008, 0, 0, 0]),
        CpuidEntry::new(0x8000_0008, 0, [0x3027, 0, 0, 0]),
    ];
    HostCpuid {
        supported,
        host,
        irqchip: IrqchipMode::Full,
        has_tsc_deadline: true,
        ..Default::default()
    }
}

fn kvm_cpu(model: &str) -> X86Cpu {
    let mut cpu = X86Cpu::new(model, Accel::Kvm(permissive_host())).unwrap();
    cpu.realize().unwrap();
    cpu
}

// Segments and special registers.

#[test]
fn set_seg_splits_flags() {
    // A flat 32-bit ring 0 code segment: type 0xb, S, P, DB, G.
    let cs = SegmentCache {
        selector: 0x10,
        base: 0,
        limit: 0xffff_ffff,
        flags: (0xb << DESC_TYPE_SHIFT) | DESC_S_MASK | DESC_P_MASK | DESC_B_MASK | DESC_G_MASK,
    };
    let k = set_seg(&cs);
    assert_eq!(k.selector, 0x10);
    assert_eq!(k.limit, 0xffff_ffff);
    assert_eq!((k.type_, k.present, k.dpl, k.db, k.s, k.l, k.g, k.avl), (0xb, 1, 0, 1, 1, 0, 1, 0));
    assert_eq!(k.unusable, 0);
    assert_eq!(get_seg(&k), cs);
}

#[test]
fn not_present_segment_is_unusable() {
    let ds = SegmentCache { selector: 0, base: 0, limit: 0, flags: 3 << DESC_TYPE_SHIFT };
    let k = set_seg(&ds);
    assert_eq!(k.present, 0);
    assert_eq!(k.unusable, 1);
    assert_eq!(get_seg(&k).flags & DESC_P_MASK, 0);
    // A present segment that KVM reports unusable loses the P bit.
    let k = KvmSegment { present: 1, unusable: 1, type_: 3, s: 1, ..Default::default() };
    assert_eq!(get_seg(&k).flags, (3 << DESC_TYPE_SHIFT) | DESC_S_MASK);
}

#[test]
fn long_mode_and_dpl_round_trip() {
    let cs = SegmentCache {
        selector: 0x33,
        base: 0,
        limit: 0xffff_ffff,
        flags: (0xb << DESC_TYPE_SHIFT)
            | DESC_S_MASK
            | DESC_P_MASK
            | DESC_L_MASK
            | DESC_G_MASK
            | DESC_AVL_MASK
            | (3 << DESC_DPL_SHIFT),
    };
    let k = set_seg(&cs);
    assert_eq!((k.l, k.dpl, k.avl, k.db), (1, 3, 1, 0));
    assert_eq!(get_seg(&k), cs);
}

#[test]
fn v8086_segments_are_ring3_data() {
    let mut state = tcg_cpu("qemu64").new_state(true);
    state.rflags |= RFLAGS_VM_MASK;
    state.segs[R_DS].selector = 0x1234;
    state.segs[R_DS].base = 0x12340;
    let s = sregs_from_state(&state);
    for seg in [s.cs, s.ds, s.es, s.fs, s.gs, s.ss] {
        assert_eq!((seg.type_, seg.present, seg.dpl, seg.s, seg.db, seg.g), (3, 1, 3, 1, 0, 0));
        assert_eq!(seg.unusable, 0);
    }
    assert_eq!(s.ds.selector, 0x1234);
    assert_eq!(s.ds.base, 0x12340);
    // TR keeps its real descriptor.
    assert_eq!(s.tr, set_seg(&state.tr));
}

#[test]
fn reset_sregs() {
    let state = tcg_cpu("qemu64").new_state(true);
    let s = sregs_from_state(&state);
    assert_eq!(s.cs.selector, 0xf000);
    assert_eq!(s.cs.base, 0xffff_0000);
    assert_eq!(s.cs.limit, 0xffff);
    assert_eq!(s.cs.present, 1);
    assert_eq!(s.cs.type_, 0xb);
    assert_eq!(s.ds.base, 0);
    assert_eq!(s.ds.type_, 3);
    assert_eq!(s.tr.type_, 0xb);
    assert_eq!(s.ldt.type_, 2);
    assert_eq!(s.gdt.limit, 0xffff);
    assert_eq!(s.idt.limit, 0xffff);
    assert_eq!(s.cr0, 0x6000_0010);
    assert_eq!(s.apic_base & !0xfff, 0xfee0_0000);
    assert_eq!(s.apic_base & MSR_IA32_APICBASE_BSP, MSR_IA32_APICBASE_BSP);
    assert_eq!(s.apic_base & MSR_IA32_APICBASE_ENABLE, MSR_IA32_APICBASE_ENABLE);
    assert_eq!(state.rip, 0xfff0);
}

#[test]
fn sregs_round_trip_keeps_hflags() {
    let state = tcg_cpu("qemu64").new_state(true);
    let s = sregs_from_state(&state);
    let mut back = X86CpuState { rflags: state.rflags, ..Default::default() };
    state_from_sregs(&mut back, &s);
    assert_eq!(back.segs, state.segs);
    assert_eq!(back.tr, state.tr);
    assert_eq!(back.ldt, state.ldt);
    assert_eq!(back.gdt.limit, state.gdt.limit);
    assert_eq!(back.cr0, state.cr0);
    assert_eq!(back.apic_base, state.apic_base);
    assert_eq!(back.hflags, state.hflags);
}

#[test]
fn hflags_for_real_mode() {
    let mut state = tcg_cpu("qemu64").new_state(true);
    state.hflags = 0;
    update_hflags(&mut state);
    // Real mode: no PE, 16-bit segments, so ADDSEG. CR0.ET is not an hflag.
    assert_eq!(state.hflags, HF_ADDSEG_MASK);
}

#[test]
fn hflags_for_long_mode() {
    let mut state = X86CpuState::default();
    let code = (0xb << DESC_TYPE_SHIFT) | DESC_S_MASK | DESC_P_MASK | DESC_L_MASK;
    let data = (3 << DESC_TYPE_SHIFT) | DESC_S_MASK | DESC_P_MASK | DESC_B_MASK;
    state.segs[R_CS].flags = code;
    state.segs[R_SS].flags = data;
    state.cr0 = CR0_PE_MASK | 0x8000_0000 | (1 << 3);
    state.cr4 = CR4_OSFXSR_MASK;
    state.efer = MSR_EFER_LMA;
    state.rflags = 0x2 | (3 << 12);
    state.hflags = HF_SMM_MASK;
    update_hflags(&mut state);
    let want = HF_SMM_MASK
        | HF_PE_MASK
        | HF_TS_MASK
        | HF_OSFXSR_MASK
        | HF_LMA_MASK
        | HF_CS32_MASK
        | HF_SS32_MASK
        | HF_CS64_MASK
        | HF_IOPL_MASK;
    assert_eq!(state.hflags, want);
}

#[test]
fn hflags_protected_mode_addseg() {
    let mut state = X86CpuState::default();
    let seg32 = (3 << DESC_TYPE_SHIFT) | DESC_S_MASK | DESC_P_MASK | DESC_B_MASK;
    state.segs[R_CS].flags = seg32 | (8 << DESC_TYPE_SHIFT);
    state.segs[R_SS].flags = seg32 | (3 << DESC_DPL_SHIFT);
    state.cr0 = CR0_PE_MASK;
    update_hflags(&mut state);
    assert_eq!(state.hflags, 3 | HF_PE_MASK | HF_CS32_MASK | HF_SS32_MASK);
    state.segs[R_DS].base = 0x1000;
    update_hflags(&mut state);
    assert_eq!(state.hflags & HF_ADDSEG_MASK, HF_ADDSEG_MASK);
}

// CPUID.

/// A fake CPUID with interesting subleaf structure.
fn fake_cpuid(i: u32, j: u32) -> Regs {
    match (i, j) {
        (0, _) => [0x1f, 1, 2, 3],
        (2, _) => [0x0000_0002, 0x11, 0, 0],
        (4, 0..=2) => [0x121 + j, 1, 2, 3],
        (4, _) => [0; 4],
        (7, 0) => [1, 0xff, 0, 0],
        (7, 1) => [0x10, 0, 0, 0],
        (0xb, 0) => [1, 2, 0x100, 0],
        (0xb, 1) => [4, 8, 0x201, 0],
        (0xb, _) => [0, 0, j, 0],
        (0xd, 0) => [0x240, 0x240, 0x240, 0],
        (0xd, 1) => [0xf, 0, 0, 0],
        (0xd, 2) => [0x100, 0x240, 0, 0],
        (0xd, _) => [0; 4],
        (0x12, 0) => [1, 0, 0, 0],
        (0x12, 1) => [2, 0, 0, 0],
        (0x12, 2) => [1, 0, 0, 0],
        (0x12, _) => [0, 0, 0, 0],
        (0x14, 0) => [0, 1, 0, 0],
        (5, _) | (6, _) => [0; 4],
        (0x8000_0000, _) => [0x8000_001d, 0, 0, 0],
        (0x8000_0001, _) => [0, 0, 1, 1],
        (0x8000_001d, 0) => [0x121, 0, 0, 0],
        (0x8000_001d, 1) => [0x122, 0, 0, 0],
        (0x8000_001d, _) => [0; 4],
        (0xC000_0000, _) => [0xC000_0001, 0, 0, 0],
        (0xC000_0001, _) => [0; 4],
        (0x1f, 0) => [1, 1, 0x100, 0],
        (0x1f, _) => [0, 0, j, 0],
        (i, _) if i < 0x20 => [0, i, 0, 0],
        _ => [0; 4],
    }
}

fn find(e: &[KvmCpuidEntry], f: u32, i: u32) -> Option<KvmCpuidEntry> {
    e.iter().copied().find(|x| x.function == f && x.index == i)
}

#[test]
fn build_cpuid_walk() {
    let walk = CpuidWalk { has_0x1f: false, xlevel2: 0 };
    let e = build_cpuid(Vec::new(), fake_cpuid, walk).unwrap();

    // Leaf 2 repeats twice, stateful.
    let l2: Vec<_> = e.iter().filter(|x| x.function == 2).collect();
    assert_eq!(l2.len(), 2);
    assert_eq!(l2[0].flags, KVM_CPUID_FLAG_STATEFUL_FUNC | KVM_CPUID_FLAG_STATE_READ_NEXT);
    assert_eq!(l2[1].flags, KVM_CPUID_FLAG_STATEFUL_FUNC);

    // Leaf 4 keeps the terminating all zero subleaf.
    let l4: Vec<_> = e.iter().filter(|x| x.function == 4).map(|x| x.index).collect();
    assert_eq!(l4, [0, 1, 2, 3]);
    assert_eq!(find(&e, 4, 3).unwrap().regs, [0; 4]);
    assert!(e.iter().filter(|x| x.function == 4).all(|x| x.flags == 1));

    // Leaf 7 lists subleaves up to EAX of subleaf 0.
    let l7: Vec<_> = e.iter().filter(|x| x.function == 7).map(|x| x.index).collect();
    assert_eq!(l7, [0, 1]);

    // Leaf 0xb stops at the first subleaf with a zero level type, kept.
    let lb: Vec<_> = e.iter().filter(|x| x.function == 0xb).map(|x| x.index).collect();
    assert_eq!(lb, [0, 1, 2]);

    // Leaf 0xd skips empty subleaves.
    let ld: Vec<_> = e.iter().filter(|x| x.function == 0xd).map(|x| x.index).collect();
    assert_eq!(ld, [0, 1, 2]);

    // Leaf 0x12 walks until a subleaf past 1 is not an EPC section.
    let l12: Vec<_> = e.iter().filter(|x| x.function == 0x12).map(|x| x.index).collect();
    assert_eq!(l12, [0, 1, 2, 3]);

    // Leaf 0x14 has only subleaf 0.
    assert!(find(&e, 0x14, 0).is_some());
    assert!(find(&e, 0x14, 1).is_none());

    // Plain zero leaves are dropped, others kept with flags 0.
    assert!(find(&e, 5, 0).is_none());
    assert_eq!(find(&e, 0x10, 0).unwrap().flags, 0);

    // 0x1f is skipped without has_0x1f.
    assert!(find(&e, 0x1f, 0).is_none());

    // Extended range: zero leaves dropped, 0x8000001d walked until EAX 0.
    assert!(find(&e, 0x8000_0002, 0).is_none());
    let lx: Vec<_> = e.iter().filter(|x| x.function == 0x8000_001d).map(|x| x.index).collect();
    assert_eq!(lx, [0, 1, 2]);

    // No Centaur leaves without xlevel2.
    assert!(e.iter().all(|x| x.function < 0xC000_0000));

    let e = build_cpuid(Vec::new(), fake_cpuid, CpuidWalk { has_0x1f: true, xlevel2: 1 }).unwrap();
    assert_eq!(find(&e, 0x1f, 0).unwrap().flags, KVM_CPUID_FLAG_SIGNIFCANT_INDEX);
    assert!(find(&e, 0x1f, 1).is_some());
    assert!(find(&e, 0x1f, 2).is_none());
    // Centaur leaves are kept even when zero.
    assert_eq!(find(&e, 0xC000_0001, 0).unwrap().regs, [0; 4]);
}

#[test]
fn build_cpuid_reports_full_table() {
    let big = |i: u32, _j: u32| if i == 0 { [0x200, 0, 0, 0] } else { [1, 0, 0, 0] };
    let err = build_cpuid(Vec::new(), big, CpuidWalk { has_0x1f: false, xlevel2: 0 }).unwrap_err();
    // Leaves 0 to 3 take four slots and leaf 4, which never ends, the rest.
    assert_eq!(err, CpuidTableFull { function: 4, index: (MAX_CPUID_ENTRIES - 4) as u32 });
    assert!(err.to_string().contains("cpuid(eax:0x4,ecx:0x60)"));
}

#[test]
fn vcpu_cpuid_signature_and_vmware_leaf() {
    let cpu = kvm_cpu("qemu64");
    assert!(cpu.expose_kvm());
    let e = vcpu_cpuid(&cpu, 0).unwrap();
    assert_eq!(e[0].function, KVM_CPUID_SIGNATURE);
    assert_eq!(e[0].regs, [KVM_CPUID_FEATURES, 0x4b4d_564b, 0x564b_4d56, 0x4d]);
    assert_eq!(e[1].function, KVM_CPUID_FEATURES);
    assert_eq!(e[1].regs[0] as u64, cpu.features()[FEAT_KVM]);
    assert!(find(&e, 0x4000_0010, 0).is_none());
    // The rest matches cpu.cpuid().
    assert_eq!(find(&e, 1, 0).unwrap().regs, cpu.cpuid(1, 0));
    assert_eq!(find(&e, 0, 0).unwrap().regs, cpu.cpuid(0, 0));

    // With a user set TSC rate the frequency leaf appears.
    let mut cpu = X86Cpu::new("qemu64", Accel::Kvm(permissive_host())).unwrap();
    cpu.set_property("tsc-frequency", "2000000000").unwrap();
    cpu.realize().unwrap();
    let e = vcpu_cpuid(&cpu, 2_000_000).unwrap();
    assert_eq!(e[0].regs[0], 0x4000_0010);
    assert_eq!(find(&e, 0x4000_0010, 0).unwrap().regs, [2_000_000, 1_000_000, 0, 0]);
}

#[test]
fn tsc_bounds() {
    assert!(freq_within_bounds(2_000_000, 2_000_000));
    assert!(freq_within_bounds(2_000_000, 2_000_500));
    assert!(freq_within_bounds(2_000_000, 1_999_500));
    assert!(!freq_within_bounds(2_000_000, 2_000_501));
    assert!(!freq_within_bounds(2_000_000, 1_999_499));
}

#[test]
fn mce_negotiation() {
    let cap = 0x0100_0000 | MCG_LMCE_P | 10;
    assert_eq!(negotiate_mcg_cap(cap, u64::MAX, 32), Ok(cap));
    assert_eq!(negotiate_mcg_cap(cap, 0, 32), Err(MceError::Lmce));
    assert_eq!(negotiate_mcg_cap(10, 0, 4), Err(MceError::Banks { want: 10, have: 4 }));
    // Unsupported bits other than LMCE are dropped silently.
    assert_eq!(negotiate_mcg_cap(0x0100_0000 | 10, 0, 32), Ok(10));
}

#[test]
fn qemu64_wants_mce() {
    // qemu64 is family 15 with MCE and MCA.
    assert!(wants_mce(&tcg_cpu("qemu64")));
    let mut cpu = X86Cpu::new("qemu64", Accel::Tcg).unwrap();
    cpu.parse_features("-mca").unwrap();
    cpu.realize().unwrap();
    assert!(!wants_mce(&cpu));
}

// MSRs.

fn full_support() -> MsrSupport {
    MsrSupport::from_index_list(&PROBED_MSRS)
}

#[test]
fn msr_support_flags() {
    let s = MsrSupport::from_index_list(&[MSR_STAR, MSR_TSC_AUX, 0x1234]);
    assert!(s.has(MSR_STAR));
    assert!(s.has(MSR_TSC_AUX));
    assert!(!s.has(MSR_IA32_SMBASE));
    assert!(!s.has(0x1234));

    // qemu64 has no RDTSCP, so TSC_AUX goes away; it has no VMX either.
    let cpu = tcg_cpu("qemu64");
    let c = s.for_cpu(&cpu, &[], 0);
    assert!(!c.tsc_aux);
    assert!(!c.feature_control);
    let entries = [KvmCpuidEntry::new(1, 0, 0, [0, 0, 1 << 5, 0])];
    assert!(s.for_cpu(&cpu, &entries, 0).feature_control);
    let c = s.for_cpu(&cpu, &[], MCG_LMCE_P);
    assert!(c.mcg_ext_ctl && c.feature_control);
}

#[test]
fn put_msrs_levels() {
    let cpu = tcg_cpu("qemu64");
    let state = cpu.new_state(true);
    let s = full_support().for_cpu(&cpu, &[], state.mcg_cap);
    let f = cpu.features();

    let rt = put_msrs(&s, f, &state, PutLevel::Runtime, 40);
    let idx: Vec<u32> = rt.iter().map(|x| x.0).collect();
    assert!(!idx.contains(&MSR_IA32_TSC));
    assert!(!idx.contains(&MSR_MTRRDEFTYPE));
    assert!(idx.contains(&MSR_IA32_SYSENTER_CS));
    assert!(idx.contains(&MSR_LSTAR));
    assert!(idx.contains(&MSR_MCG_STATUS));

    let reset = put_msrs(&s, f, &state, PutLevel::Reset, 40);
    let idx: Vec<u32> = reset.iter().map(|x| x.0).collect();
    assert!(idx.contains(&MSR_IA32_TSC));
    assert!(idx.contains(&MSR_MTRRDEFTYPE));
    assert!(!idx.contains(&MSR_TSC_AUX));
    assert!(idx.contains(&MSR_IA32_SMBASE));
    // qemu64 has no CET, FRED, SGX LC or XFD.
    assert!(!idx.contains(&MSR_IA32_U_CET));
    assert!(!idx.contains(&MSR_IA32_FRED_RSP0));
    assert!(!idx.contains(&MSR_IA32_XFD));

    // Banks follow MCG_EXT_CTL (or MCG_CTL without it), with their values.
    let banks = (state.mcg_cap & 0xff) as usize * 4;
    assert!(banks > 0);
    let pos = idx.iter().position(|&m| m == MSR_MC0_CTL).unwrap();
    assert_eq!(idx[pos - 1], MSR_MCG_CTL);
    assert_eq!(idx[pos + 1], MSR_MC0_STATUS);
    assert_eq!(idx.len(), pos + banks);
    assert_eq!(reset[pos].1, state.mce_banks[0]);

    // Values come from the state.
    let pat = reset.iter().find(|x| x.0 == MSR_PAT).unwrap().1;
    assert_eq!(pat, state.pat);
    let smbase = reset.iter().find(|x| x.0 == MSR_IA32_SMBASE).unwrap().1;
    assert_eq!(smbase, 0x30000);
}

#[test]
fn mtrr_masks_trimmed_and_filled() {
    let cpu = tcg_cpu("qemu64");
    let mut state = cpu.new_state(true);
    state.mtrr_var[1].mask = u64::MAX;
    let s = full_support();
    let v = put_msrs(&s, cpu.features(), &state, PutLevel::Reset, 40);
    let m = v.iter().find(|x| x.0 == msr_mtrr_phys_mask(1)).unwrap().1;
    assert_eq!(m, (1 << 40) - 1);

    assert!(set_msr_value(&mut state, msr_mtrr_phys_mask(2), 0x800, 40));
    assert_eq!(state.mtrr_var[2].mask, 0x800 | (((1u64 << 12) - 1) << 40));
    assert!(set_msr_value(&mut state, msr_mtrr_phys_base(2), 0x6, 40));
    assert_eq!(state.mtrr_var[2].base, 6);
}

#[test]
fn get_msr_list_adds_deadline_and_feature_control() {
    let cpu = tcg_cpu("qemu64");
    let state = cpu.new_state(true);
    let mut s = full_support();
    s.feature_control = true;
    let idx = get_msr_indices(&s, cpu.features(), &state);
    assert!(idx.contains(&MSR_IA32_TSC));
    assert_eq!(idx[idx.len() - 2], MSR_IA32_TSCDEADLINE);
    assert_eq!(idx[idx.len() - 1], MSR_IA32_FEATURE_CONTROL);
}

#[test]
fn msr_values_round_trip() {
    let cpu = tcg_cpu("qemu64");
    let state = cpu.new_state(true);
    let mut s = full_support();
    s.feature_control = true;
    s.mcg_ext_ctl = true;
    let idx = get_msr_indices(&s, cpu.features(), &state);
    let mut other = X86CpuState { mcg_cap: state.mcg_cap, ..Default::default() };
    for (n, &i) in idx.iter().enumerate() {
        if !is_mtrr_mask(i) {
            assert!(set_msr_value(&mut other, i, n as u64 + 7, 52), "msr {i:#x}");
            assert_eq!(msr_value(&other, i), Some(n as u64 + 7), "msr {i:#x}");
        }
    }
    assert!(!set_msr_value(&mut other, 0xdead, 1, 40));
    assert_eq!(msr_value(&other, 0xdead), None);
    // Banks past MCG_CAP are not tracked.
    let past = MSR_MC0_CTL + (state.mcg_cap & 0xff) as u32 * 4;
    assert!(!set_msr_value(&mut other, past, 1, 40));
}

#[test]
fn init_msrs_follow_host_list() {
    let cpu = tcg_cpu("qemu64");
    assert!(init_msrs(&cpu, &MsrSupport::from_index_list(&[])).is_empty());
    let v = init_msrs(&cpu, &full_support());
    let idx: Vec<u32> = v.iter().map(|x| x.0).collect();
    assert_eq!(idx, [MSR_IA32_ARCH_CAPABILITIES, MSR_IA32_CORE_CAPABILITY, MSR_IA32_UCODE_REV]);
    assert_eq!(v[2].1, cpu.ucode_rev());
}

// Local APIC.

#[test]
fn lapic_reset_page() {
    let r = lapic_reset_regs(3);
    assert_eq!(lapic_reg(&r, 0x2), 3 << 24);
    assert_eq!(lapic_reg(&r, 0x8), 0);
    assert_eq!(lapic_reg(&r, 0xe), 0xffff_ffff);
    assert_eq!(lapic_reg(&r, 0xf), 0xff);
    for i in 0x32..=0x37 {
        assert_eq!(lapic_reg(&r, i), APIC_LVT_MASKED);
    }
    assert_eq!(lapic_reg(&r, 0x3), 0);
    // Everything else is zero.
    let nonzero = (0..64).filter(|&i| lapic_reg(&r, i) != 0).count();
    assert_eq!(nonzero, 3 + 6);
    // The ID byte lands in the top byte of the register at 0x20.
    assert_eq!(r[0x23], 3);
}

// FPU and XSAVE.

#[test]
fn xsave_round_trip() {
    let mut state = tcg_cpu("qemu64").new_state(true);
    state.fpstt = 5;
    state.fpus = 0x3800 | 0x41;
    state.fptags = [1, 1, 0, 1, 1, 0, 1, 1];
    state.mxcsr = 0x1fa0;
    state.xstate_bv = 0x3;
    state.pkru = 0x5555_0000;
    let buf = xsave_from_state(&state, Some(0xa80));
    assert_eq!(buf.len(), 4096);
    assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), 0x37f);
    // TOP replaces bits 11..13 of the status word.
    assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), (5 << 11) | 0x41);
    assert_eq!(buf[4], 0b0010_0100);
    assert_eq!(u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]), 0x1fa0);
    assert_eq!(buf[512], 3);

    let mut back = X86CpuState::default();
    state_from_xsave(&mut back, &buf, Some(0xa80));
    assert_eq!(back.fpuc, 0x37f);
    assert_eq!(back.fpstt, 5);
    assert_eq!(back.fpus, (5 << 11) | 0x41);
    assert_eq!(back.fptags, state.fptags);
    assert_eq!(back.mxcsr, 0x1fa0);
    assert_eq!(back.xstate_bv, 3);
    assert_eq!(back.pkru, 0x5555_0000);

    let fpu = fpu_from_state(&state);
    assert_eq!(fpu.ftwx, 0b0010_0100);
    let mut back = X86CpuState::default();
    state_from_fpu(&mut back, &fpu);
    assert_eq!(back.fptags, state.fptags);
    assert_eq!(back.fpstt, 5);
    assert_eq!(back.mxcsr, 0x1fa0);
}

#[test]
fn reset_xsave_is_empty_fpu() {
    let state = tcg_cpu("qemu64").new_state(true);
    let buf = xsave_from_state(&state, None);
    // After reset every tag is "empty", so the abridged tag word is zero.
    assert_eq!(buf[4], 0);
    assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), 0x37f);
}

// Events and debug registers.

#[test]
fn events_round_trip() {
    let mut state = tcg_cpu("qemu64").new_state(true);
    state.exception_nr = 14;
    state.exception_injected = true;
    state.has_error_code = true;
    state.error_code = 0x2;
    state.interrupt_injected = 0x30;
    state.soft_interrupt = true;
    state.nmi_pending = true;
    state.hflags2 |= HF2_NMI_MASK;
    state.hflags |= HF_SMM_MASK;
    state.sipi_vector = 0x9;
    state.mp_state = MP_STATE_SIPI_RECEIVED;

    let caps = EventCaps { smm: true, ..Default::default() };
    let ev = events_from_state(&state, PutLevel::Runtime, caps);
    assert_eq!(ev.flags, KVM_VCPUEVENT_VALID_SMM);
    assert_eq!((ev.exception_nr, ev.exception_injected, ev.exception_error_code), (14, 1, 2));
    assert_eq!((ev.interrupt_injected, ev.interrupt_nr, ev.interrupt_soft), (1, 0x30, 1));
    assert_eq!((ev.nmi_pending, ev.nmi_masked, ev.smi_smm), (1, 1, 1));

    let ev = events_from_state(&state, PutLevel::Reset, caps);
    assert_eq!(
        ev.flags,
        KVM_VCPUEVENT_VALID_SMM | KVM_VCPUEVENT_VALID_NMI_PENDING | KVM_VCPUEVENT_VALID_SIPI_VECTOR
    );

    let mut back = X86CpuState::default();
    state_from_events(&mut back, &ev);
    assert_eq!(back.exception_nr, 14);
    assert!(back.exception_injected && back.has_error_code);
    assert_eq!(back.error_code, 2);
    assert_eq!(back.interrupt_injected, 0x30);
    assert!(back.soft_interrupt && back.nmi_pending && !back.nmi_injected);
    assert_eq!(back.hflags2 & HF2_NMI_MASK, HF2_NMI_MASK);
    assert_eq!(back.hflags & HF_SMM_MASK, HF_SMM_MASK);
    assert_eq!(back.sipi_vector, 9);
}

#[test]
fn reset_events_are_idle() {
    let mut state = tcg_cpu("qemu64").new_state(true);
    state.mp_state = MP_STATE_RUNNABLE;
    let ev = events_from_state(&state, PutLevel::Reset, EventCaps::default());
    assert_eq!(ev.flags, KVM_VCPUEVENT_VALID_NMI_PENDING);
    assert_eq!(ev.interrupt_injected, 0);
    assert_eq!(ev.exception_injected, 0);

    let mut back = state.clone();
    back.nmi_pending = true;
    back.hflags2 |= HF2_NMI_MASK;
    state_from_events(&mut back, &ev);
    assert_eq!(back.exception_nr, -1);
    assert_eq!(back.interrupt_injected, -1);
    assert!(!back.nmi_pending);
    assert_eq!(back.hflags2 & HF2_NMI_MASK, 0);
}

#[test]
fn debugregs_alias() {
    let mut state = X86CpuState::default();
    state_from_debugregs(&mut state, [1, 2, 3, 4], 0xffff_0ff0, 0x400);
    assert_eq!(state.dr, [1, 2, 3, 4, 0xffff_0ff0, 0x400, 0xffff_0ff0, 0x400]);
    assert_eq!(debugregs_from_state(&state), ([1, 2, 3, 4], 0xffff_0ff0, 0x400));
}
