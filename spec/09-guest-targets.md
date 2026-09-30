# 09. Guest targets

This document covers the guest instruction set architectures ruvm emulates, how each one plugs into the JIT described in documents 07 and 08, and the rules for matching QEMU's CPU models, feature properties, identification registers and gdb register descriptions exactly. The reference is the QEMU 11.1.0 tree (tag v11.1.0). Where the master branch after 11.1 (VERSION 11.1.50 at the end of September 2026) differs, the text says so, because ruvm tracks master for the next release but the compatibility contract in document 02 is pinned to a released version.

## Target inventory

QEMU 11.1.0 builds system emulators and user-mode emulators for 19 target directories under target/: alpha, arm, avr, hexagon, hppa, i386, loongarch, m68k, microblaze, mips, or1k (OpenRISC), ppc, riscv, rx, s390x, sh4, sparc, tricore, xtensa. Two architectures that older ruvm planning notes listed are gone and are out of scope: Nios II was removed in QEMU 9.1 (the architecture was orphaned and Intel ended the IP) and CRIS was removed in 9.2 (Linux dropped it in 4.17 and no distribution packaged a compiler). The iwMMXt extension and the pxa CPUs were removed in 10.2, and 32-bit host operating systems were dropped entirely in 11.0 (docs/about/removed-features.rst). ruvm follows all of these removals and does not resurrect anything QEMU has deleted, since a feature that exists only in ruvm is not a compatibility feature.

Two changes in the 11.x cycle matter for this document. First, Hexagon is no longer user-mode only: v11.1.0 ships configs/targets/hexagon-softmmu.mak and hw/hexagon, so qemu-system-hexagon exists alongside qemu-hexagon. Second, RISC-V big-endian support landed in 11.1 as a CPU property, not as a new binary. target/riscv/cpu.c defines `DEFINE_PROP_BOOL("big-endian", RISCVCPU, cfg.big_endian, false)`, and at reset the CPU hardwires mstatus.MBE, SBE and UBE to the property value. Instruction fetch stays little-endian as the RISC-V spec requires; data accesses become big-endian. There is no riscv64be-linux-user configuration in 11.1.0, so big-endian RISC-V is a system-mode feature only, and ruvm mirrors that.

The binaries QEMU 11.1.0 builds, which become argv[0] names for the ruvm multi-call binary (document 03), are listed below by target directory.

| target/ dir | ruvm crate | qemu-system-* | qemu-* (linux-user) | bsd-user |
|---|---|---|---|---|
| alpha | ruvm-target-alpha | alpha | alpha | no |
| arm | ruvm-target-arm | arm, aarch64 | arm, armeb, aarch64, aarch64_be | arm, aarch64 |
| avr | ruvm-target-avr | avr | no | no |
| hexagon | ruvm-target-hexagon | hexagon | hexagon | no |
| hppa | ruvm-target-hppa | hppa | hppa | no |
| i386 | ruvm-target-x86 | i386, x86_64 | i386, x86_64 | i386, x86_64 |
| loongarch | ruvm-target-loongarch | loongarch64 | loongarch64 | no |
| m68k | ruvm-target-m68k | m68k | m68k | no |
| microblaze | ruvm-target-microblaze | microblaze | microblaze, microblazeel | no |
| mips | ruvm-target-mips | mips, mipsel, mips64, mips64el | mips, mipsel, mips64, mips64el, mipsn32, mipsn32el | no |
| or1k | ruvm-target-openrisc | or1k | or1k | no |
| ppc | ruvm-target-ppc | ppc, ppc64 | ppc, ppc64, ppc64le | no |
| riscv | ruvm-target-riscv | riscv32, riscv64 | riscv32, riscv64 | riscv64 |
| rx | ruvm-target-rx | rx | no | no |
| s390x | ruvm-target-s390x | s390x | s390x | no |
| sh4 | ruvm-target-sh4 | sh4, sh4eb | sh4, sh4eb | no |
| sparc | ruvm-target-sparc | sparc, sparc64 | sparc, sparc32plus, sparc64 | no |
| tricore | ruvm-target-tricore | tricore | no | no |
| xtensa | ruvm-target-xtensa | xtensa, xtensaeb | xtensa, xtensaeb | no |

The 19 crates map one to one onto the 19 directories. The OpenRISC crate is named ruvm-target-openrisc, after the architecture, while the binary names keep QEMU's or1k.

HPPA deserves a note because "64-bit HPPA" is often misreported. Since QEMU 8.2 the hppa target is built with TARGET_LONG_BITS=64 (configs/targets/hppa-softmmu.mak) and models both PA-RISC 1.1 and PA-RISC 2.0. The CPU types are pa-7300lc (is_pa20 false), pa-8500 and pa-8700 (is_pa20 true), defined in target/hppa/cpu.c. There is one qemu-system-hppa binary; the machine (B160L versus C3700 and later boards) picks the CPU. ruvm keeps the single binary and the same three CPU types. The 11.1 release notes mention HP-UX related TLB fixes and new SeaBIOS-hppa firmware; document 11 covers the firmware side.

## The GuestArch trait

Every ruvm-target crate implements `GuestArch`. The trait is the only interface between a target and the rest of the system: the JIT (documents 07 and 08), the accelerators (document 06, for register sync on KVM, HVF, WHPX), the gdbstub, the monitor, migration (document 17) and user mode (document 10). It replaces what QEMU spreads across CPUClass, TCGCPUOps, SysemuCPUOps, the TranslatorOps table in accel/tcg/translator.c and a set of per-target globals.

```rust
pub trait GuestArch: Send + Sync + 'static {
    /// Per-vCPU architectural state. Must be #[repr(C)] because JIT code
    /// addresses fields by offset, and plugins read registers through it.
    type State: ArchStateLayout + Default + Send;

    const NAME: &'static str;            // "aarch64", matches QEMU target name
    const PAGE_BITS: PageBits;           // Fixed(12) or Vary { min, legacy }
    const VIRT_ADDR_BITS: u8;            // TARGET_VIRT_ADDR_SPACE_BITS
    const PHYS_ADDR_BITS: u8;            // TARGET_PHYS_ADDR_SPACE_BITS
    const ENDIAN: Endian;                // default data endianness
    const INSN_START_WORDS: usize;       // like TARGET_INSN_START_WORDS

    fn cpu_models() -> &'static [CpuModelDef];
    fn default_cpu(machine: &str) -> &'static str;
    fn properties(model: &CpuModelDef) -> PropertySet;   // feature bits as QOM props
    fn realize(cfg: &CpuConfig, accel: AccelKind) -> Result<Self::State, CpuError>;
    fn reset(state: &mut Self::State, kind: ResetType);

    /// Translate one guest basic block into ruvm-jit IR.
    fn translate(ctx: &mut TranslateCtx<'_, Self>, pc: GuestAddr, flags: TbFlags) -> TbEnd;
    fn tb_flags(state: &Self::State) -> (GuestAddr, TbFlags, u64 /* cs_base */);
    fn restore_state(state: &mut Self::State, words: &[u64]);  // from side table

    fn mmu_indexes() -> &'static [MmuIndexDesc];
    fn tlb_fill(state: &mut Self::State, req: &TlbFillReq) -> Result<TlbEntry, GuestFault>;
    fn raise(state: &mut Self::State, fault: GuestFault) -> !;
    fn do_interrupt(state: &mut Self::State, pending: IrqLines) -> bool;

    fn gdb_features(cfg: &CpuConfig) -> Vec<GdbFeature>;
    fn gdb_read_reg(state: &Self::State, feature: usize, n: usize, out: &mut Vec<u8>) -> usize;
    fn gdb_write_reg(state: &mut Self::State, feature: usize, n: usize, data: &[u8]) -> usize;

    fn vmstate() -> &'static VmStateDescription;       // must match target/*/machine.c
    fn to_accel_regs(state: &Self::State, out: &mut dyn ArchRegSink);
    fn from_accel_regs(state: &mut Self::State, src: &dyn ArchRegSource);
    fn disas(bytes: &[u8], pc: GuestAddr, cfg: &CpuConfig, out: &mut dyn fmt::Write) -> usize;
    fn user_mode() -> Option<&'static dyn UserArch>;   // syscalls, signals, ELF (doc 10)
}
```

Several design decisions sit in this sketch.

The state struct is a concrete #[repr(C)] type with field offsets exported at build time (`ArchStateLayout`), not a trait object. JIT code loads and stores guest registers at fixed offsets from a base register, as TCG does with env, and the tier 2 optimizer in document 08 needs to know which offsets alias. The offsets are also what the plugin register API (qemu_plugin_read_register) exposes indirectly through gdb feature numbering, so they must be stable within a build.

`translate` produces ruvm-jit IR, not host code, and each target is monomorphized over its own `TranslateCtx<Self>`. The translate loop itself (instruction budget, page crossing, plugin instrumentation points, single step, icount) lives in ruvm-jit and is shared, as accel/tcg/translator.c is shared in QEMU. The target only decodes and emits.

`restore_state` receives the per-instruction words recorded in the side table (document 08 describes the encoding). This is the equivalent of TCGCPUOps.restore_state_to_opc and uses the same number of words per target as QEMU's TARGET_INSN_START_WORDS (for example Arm records pc, condexec bits and a syndrome word). We keep the same word count so that the side table can be checked against QEMU's -d op output during differential testing.

`tlb_fill` returns a result rather than longjmp'ing out of the helper. The JIT runtime converts a `GuestFault` into an unwind to the dispatcher using the side table, with no setjmp/longjmp and no sigsetjmp in the hot path. This is the single largest structural difference from QEMU's cpu_loop_exit, and document 08 has the unwinding protocol.

`gdb_features` is a function of the realized configuration, because on several targets (Arm, RISC-V, PowerPC, LoongArch, x86 with APX) the register set depends on enabled features. It returns ordered feature descriptions; the gdbstub assigns register numbers by concatenation exactly as gdbstub/gdbstub.c does with gdb_register_coprocessor.

`user_mode` returns the linux-user and bsd-user hooks (syscall table, signal frame layout, ELF hwcaps, cpu_loop exit reasons). It returns None for avr, rx and tricore, which QEMU has no user mode for.

Accelerators other than the JIT see a target only through `to_accel_regs`/`from_accel_regs` and the CPU model machinery. That is why x86, arm, riscv, ppc, s390x, loongarch and mips targets must build without the JIT feature enabled: a KVM-only ruvm build for a cloud host links ruvm-target-x86 with the `jit` cargo feature off, the same way QEMU builds with --disable-tcg.

## CPU models and feature exposure

CPU model compatibility is the part of target work that management software notices first. libvirt calls query-cpu-definitions and query-cpu-model-expansion at startup, caches the result per QEMU binary, and generates -cpu strings from it. If ruvm returns a different list, a different expansion, or a different set of property names, libvirt either refuses the domain or silently builds a different guest. The contract from document 02 therefore applies at the level of individual feature bits.

### Model definitions are data, generated from QEMU

For every target, the model table (name, aliases, versions, base identification values, default feature set) is kept as data in the ruvm-target crate: a TOML file per target under `models/`, turned into Rust statics by a build script. The first version of each file is produced by a one-off extraction tool (`cargo xtask import-cpu-models`) that compiles QEMU's own tables and dumps them, rather than by hand transcription. For x86 that means builtin_x86_defs and the X86CPUVersionDefinition chains in target/i386/cpu.c; for Arm the per-model init functions in target/arm/cpu64.c and target/arm/tcg/cpu32.c and tcg/cpu64.c, which set MIDR, REVIDR, CTR, reset SCTLR and every ID register via SET_IDREG; for PowerPC the 215 POWERPC_DEF entries and the alias table in target/ppc/cpu-models.c; for s390x the CPUDEF_INIT table in target/s390x/cpu_models.c plus the generated feature groups from target/s390x/gen-features.c. The extraction is repeated for every QEMU release and the diff is reviewed by a human, because a changed default in a versioned model is a guest ABI change.

The model lists in 11.1.0 that ruvm must reproduce exactly include:

- x86: qemu64, qemu32, kvm64, kvm32, 486, pentium, pentium2, pentium3, athlon, phenom, coreduo, core2duo, n270, Conroe, Penryn, Nehalem, Westmere, SandyBridge, IvyBridge, Haswell, Broadwell, Skylake-Client, Skylake-Server, Cascadelake-Server, Cooperlake, Icelake-Server, SapphireRapids, GraniteRapids, DiamondRapids, SierraForest, ClearwaterForest, Denverton, Snowridge, KnightsMill, Opteron_G1 to Opteron_G5, EPYC, EPYC-Rome, EPYC-Milan, EPYC-Genoa, EPYC-Turin, Dhyana, YongFeng, each with its versioned variants (Name-v1, Name-v2, and so on), plus host, max and base.
- Arm (A-profile 64-bit): cortex-a35, cortex-a53, cortex-a55, cortex-a57, cortex-a72, cortex-a76, cortex-a78ae, cortex-a710, neoverse-n1, neoverse-n2, neoverse-v1, a64fx, host, max. 32-bit and M/R-profile: arm926, arm946, arm1026, arm1136, arm1136-r2, arm1176, arm11mpcore, cortex-a7, cortex-a8, cortex-a9, cortex-a15, cortex-r5, cortex-r5f, cortex-r52, ti925t, sa1100, sa1110, and the cortex-m0/m3/m4/m7/m33/m55 family from target/arm/tcg/cpu-v7m.c. Master after 11.1 adds max-v8 and max-v9; these are not in the 11.1 contract.
- RISC-V: rv32, rv64, rv32i, rv32e, rv64i, rv64e, x-rv128, max, max32, the profile CPUs rva22u64, rva22s64, rva23u64, rva23s64, and vendor models lowrisc-ibex, shakti-c, sifive-e, sifive-e31, sifive-e34, sifive-e51, sifive-u, sifive-u34, sifive-u54, thead-c906, thead-c908, thead-c908v, veyron-v1, tt-ascalon, xiangshan-nanhu, xiangshan-kunminghu, mips-p8700, host (KVM).
- PowerPC: the e300, e500, e600, 4xx, 6xx, 7xx, 74xx families and power5+ through power11 (power11_v2.0 is the newest; POWER8E and POWER8NVL were removed in 11.1).
- s390x: z900 through z14ZR1, gen15a/b, gen16a/b, gen17a/b, plus qemu, max and host; the qemu model is versioned by machine type through qemu_V6_0, qemu_V6_2, qemu_V7_0, qemu_V7_1 feature lists in gen-features.c.
- LoongArch: la464, la132 (32-bit), max, host. Master adds la664.
- Hexagon: v5, v55, v60, v61, v62, v65, v66, v67, v68, v69, v71, v73 (master adds v75, v79, v81).
- MIPS: 4Kc through Octeon68XX, 33 definitions in target/mips/cpu-defs.c.inc including Loongson-2E/2F/3A1000/3A4000, I6400, I6500, P5600, mips32r6-generic and MIPS64R2-generic.
- SPARC: 30 definitions from Fujitsu-Sparc64 through LEON3 in target/sparc/cpu.c.
- m68k: m68000, m68010, m68020, m68030, m68040, m68060, m5206, m5208, cfv4e, any.
- Alpha: ev4, ev5, ev56, ev6, ev67, ev68 and the 21064, 21164, 21264 aliases.
- Others: sh7750r, sh7751r, sh7785; or1200 and any; tc1796, tc1797, tc27x, tc37x; rx62n; avr5, avr51, avr6; xtensa cores dc232b, dc233c, de212, de233_fpu, dsp3400, fsf, lx106, sample_controller, test_kc705_be, test_mmuhifi_c3; microblaze configured through properties (use-fpu, use-mmu, use-barrel and so on) on a single CPU type.

### Features are QOM properties with QEMU's names

Every feature bit is a boolean QOM property with QEMU's exact spelling, including legacy aliases. On x86 that is feature_word_info[], which names each CPUID bit ("avx2", "avx10", "apxf", "cmpccxadd") and each MSR-based feature word (the VMX capability words, ARCH_CAPABILITIES). Aliases such as "sse4_1" versus "sse4.1" and underscore versus dash spellings go through the same compatibility path QEMU uses (x86_cpu_register_feature_bit_props and the legacy name table). On Arm the properties are sve, sve128 through sve2048, sme, sme128 through sme2048, pauth, pauth-impdef, pauth-qarma3, pmu, aarch64 (to disable AArch64 under KVM), lpa2 and a handful of others; most Arm features are not individual properties because QEMU derives them from ID register fields, and ruvm does the same. On RISC-V each extension is a property ("zba", "zicbom", "v", "zvfbfa") with the MULTI_EXT_CFG_BOOL tables of target/riscv/cpu.c (388 entries in 11.1.0), plus value properties such as vlen, elen, cbom_blocksize, priv_spec, mvendorid, marchid, mimpid and the new big-endian. Capitalized "Z" spellings, deprecated since 8.2, still parse and warn.

The implementation is table driven: a feature table maps property name to (register or word, bit range, default per model, dependency list, accelerator availability). Property setting happens before realize, then one expansion function computes the final identification values. This mirrors x86_cpu_expand_features followed by x86_cpu_filter_features, and arm_cpu_realizefn's cross-checks (for example, SVE vector lengths must form a valid set, and disabling FP forces off AdvSIMD).

### Filtering and the host/max models

The rule for what the guest sees is identical to QEMU's and has three inputs: the model's defaults, the user's properties, and what the accelerator can support.

For TCG, each feature word has a mask of what the emulator implements. QEMU calls these tcg_features in feature_word_info; ruvm stores the same masks in the model data and they must match the JIT's actual capability. When a named model asks for a bit the JIT does not implement, the bit is removed, the name is added to the unavailable-features property returned by query-cpu-definitions, and a warning is printed with the same text QEMU prints ("TCG doesn't support requested feature: CPUID.07H:EBX.avx512f [bit 16]"). With `enforce` set, realize fails. With `check`, the warning is printed. The effect matters for x86: in 11.1.0 the TCG masks stop at AVX2 and CMPCCXADD; TCG_7_1_EDX_FEATURES is 0, so AVX10 and APX are never exposed under TCG, and Skylake-Server and later models lose their AVX-512 bits under TCG in both QEMU and ruvm.

For KVM, HVF and WHPX, supported bits come from the kernel or hypervisor (KVM_GET_SUPPORTED_CPUID on x86, KVM_GET_ONE_REG on ID registers for Arm, the HVF feature registers on macOS). The `host` model passes through everything supported and is only valid with a hardware accelerator. The `max` model means "everything this accelerator supports": host passthrough under KVM, and the full TCG mask under TCG. On x86 under KVM, host and max also honour migratable=on (the default), which removes bits that cannot migrate; that list is in QEMU's feature_word_info unmigratable_flags and ruvm copies it verbatim.

Exposure of identification registers must be byte-identical, which is testable. The conformance harness boots a tiny guest stub on both QEMU and ruvm for every (model, accelerator, machine version) triple and dumps: on x86 every CPUID leaf and subleaf up to the max basic and extended leaves plus the MSRs that expose features; on Arm every ID_AA64*_EL1 and ID_*_EL1 register, MIDR, REVIDR, CTR, DCZID and CLIDR; on RISC-V misa, mvendorid, marchid, mimpid and the riscv,isa device tree string; on s390x STFLE bits and STIDP; on PowerPC the PVR and the ibm,pa-features device tree property. Any difference is a test failure. This is the same idea as QEMU's own tests/functional CPU model tests but broader, and it runs in CI for every change to a models file.

### Machine versions change CPU defaults

Versioned machine types (pc-q35-11.1, virt-11.1, pseries-11.1, s390-ccw-virtio-11.1) carry compat properties that change CPU defaults. Examples from hw/i386/pc.c in 11.1.0 are x-l1-cache-per-thread, x-amd-topoext-features-only, x-vendor-cpuid-only-v2 and x-arch-cap-always-on on TYPE_X86_CPU for older machine versions; on s390x the qemu model feature list itself changes per machine version. These are part of the `Machine` trait's compat property arrays in document 11, not part of the target crate. The target only exposes the properties; the machine sets them. Document 04 describes how global and compat properties are applied in QEMU's order.

### QMP commands

ruvm implements the target-specific QMP commands with the same availability per binary: query-cpu-definitions and query-cpu-model-expansion (x86, arm, riscv, s390x, loongarch, ppc for definitions), query-cpu-model-comparison and query-cpu-model-baseline (s390x only), query-sev and friends (x86, document 19), query-gic-capabilities (arm). Availability is driven by the QAPI 'if' conditions compiled per target, so query-qmp-schema introspection matches QEMU for each qemu-system-<arch> name (document 18).

### Proposed extension: opt-in ISA beyond QEMU's TCG

ruvm's JIT will eventually implement some instructions QEMU's TCG does not, AVX-512 being the obvious case since x86-64 guests on Arm hosts increasingly assume it. To keep default behaviour identical, these are never enabled by -cpu max or by a named model. They require the accelerator property `-accel tcg,x-ruvm-extra-isa=on`, which widens the TCG masks. With the property off, CPUID is the same as QEMU's bit for bit. This is a new decision; it has been added to document 25 as an open question on naming and on whether it should be allowed on migratable machines. The current answer is no: a VM with the property on refuses migration to a QEMU destination, because a QEMU destination cannot run the instructions the guest may be using.

## Privileged architecture and vector state per target

This section lists, for each target, what the JIT and the target crate have to model beyond user-level instructions. It is the scope checklist used in document 23 to decide when a target is done. "Matches QEMU" always means: the same set of implemented features under TCG, not the full architecture. Where QEMU does not implement something (x86 VMX under TCG, for instance), ruvm does not either unless it is behind the opt-in described above.

### x86 (i386, x86_64)

Modes: real, virtual-8086, 16/32-bit protected, long mode with compatibility submode, and SMM (target/i386/tcg/system/smm_helper.c). The page walker is mmu_translate in target/i386/tcg/system/excp_helper.c: 2-level 32-bit, PAE, 4-level and 5-level (LA57 is in TCG_7_0_ECX_FEATURES), with PSE, NX, SMEP, SMAP, UMIP, PKU and PKS, and A/D bit updates done with atomic compare-and-swap on guest page table entries. Nested virtualization under TCG is SVM only (CPUID_EXT3_SVM is in TCG_EXT3_FEATURES, implemented in svm_helper.c, including nested paging); VMX is KVM-only. ruvm keeps that split. Under KVM, nested VMX, SEV-SNP, TDX and CET virtualization are accelerator features covered in documents 06 and 19; the target crate supplies only the CPUID and MSR plumbing and the XSAVE layouts.

Vector and FP state: x87 with 80-bit extended precision (floatx80 in softfloat), MMX, SSE through SSE4.2, AES-NI, PCLMULQDQ, SHA-NI, VAES and VPCLMULQDQ, F16C, FMA, AVX and AVX2, BMI1/2, ADX. Since QEMU 7.2 these go through the table-driven decoder in target/i386/tcg/decode-new.c.inc and emit.c.inc, and ruvm ports that decoder structure directly, because it already separates operand decoding from emission in a way that maps cleanly to IR generation. AVX-512, AMX, AVX10 and APX are exposed only under KVM (APX's extended GPRs get their own gdb feature, i386-64bit-apx.xml, and an XSAVE component; target/i386/cpu.c refuses APX together with MPX because APX reuses MPX's XSAVE area). XSAVE, XSAVEOPT and XGETBV1 are in the TCG XSAVE mask; XSAVES and XSAVEC are not.

x86 is the target where the JIT's quality matters most (headline SPEC target in the canon), so its lazy flags scheme is special cased: the target emits CC_OP-style deferred flag records as QEMU does with cc_op, and the tier 2 optimizer eliminates dead records across blocks. The "ARMing x86 Games" flag speculation work (MobiSys 2025) is evaluated in document 08 as a tier 2 option for x86-on-Arm.

### Arm (arm, aarch64)

Exception levels EL0 to EL3 in AArch64 and AArch32, Secure and Non-secure states, Secure EL2 (FEAT_SEL2), and Realm and Root states from FEAT_RME with the granule protection check (FEAT_RME_GPC2 in 11.1). Nested virtualization via FEAT_NV and FEAT_NV2. The table walker is target/arm/ptw.c, which handles VMSAv7 short descriptors, LPAE, AArch64 stage 1 and stage 2 with 4K, 16K and 64K granules, FEAT_LPA and FEAT_LPA2 (52-bit), hardware access and dirty flags (FEAT_HAFDBS), permission indirection (FEAT_S1PIE, FEAT_S2PIE), and for M and R profiles the PMSAv7 and PMSAv8 MPUs. The MMU index scheme (target/arm/mmuidx.h) has a large number of indexes because regime, stage, privilege and PAN state are all folded in; QEMU sets NB_MMU_MODES to 22 in include/hw/core/cpu.h, and ruvm's softmmu TLB (document 08) supports the same 22 indexes per vCPU so that index numbering, which appears in -d mmu logs and plugin hwaddr queries, is identical. M-profile adds the NVIC exception model, TrustZone-M with the IDAU/SAU and lazy FP stacking (target/arm/tcg/m_helper.c).

Other privileged features in the 11.1.0 emulation list (docs/system/arm/emulation.rst) that carry real work: MTE up to FEAT_MTE4 with tag storage in a separate address space, pointer authentication with QARMA5, QARMA3 and the impdef algorithm, BTI, GCS (guarded control stack, a new cpregs-gcs.c), FEAT_NMI, FEAT_MOPS memcpy/memset instructions, the PMU (PMUv3p5), the generic timer, and FEAT_ECV. The GICv3 CPU interface system registers live in hw/intc/arm_gicv3_cpuif.c in QEMU; in ruvm they are registered by ruvm-hw-intc into the CPU's system register table through a callback interface, because they belong in the same lock domain as the distributor (document 12). The experimental GICv5 in 11.1 follows the same pattern.

Vectors: AdvSIMD and VFP, SVE, SVE2, SVE2p1 with vector lengths 128 to 2048 bits, SME, SME2, SME2p1 with streaming mode and the ZA array (plus ZT0 for SME2), FP8 conversions and dot products (FEAT_FP8, FEAT_FP8DOT2, FEAT_FP8DOT4, FEAT_FP8FMA), BF16, I8MM, and on M-profile MVE (Helium). The system register surface is large: QEMU's cpreg hashtable has hundreds of entries per CPU, each with access functions, reset values, and trap checks for FGT (fine-grained traps). ruvm's Arm crate represents system registers as a generated table keyed by the (op0, op1, CRn, CRm, op2) encoding, with the access checks expressed as data where possible; that table is what the MRS BSD JSON cross-check described below validates.

### RISC-V (riscv32, riscv64)

Privilege modes M, S, U, with the H extension adding HS, VS and VU. Translation schemes Sv32, Sv39, Sv48, Sv57 and the G-stage variants (Sv39x4 and so on) for two-stage translation, with Svnapot, Svpbmt, Svade and Svadu, Svinval and Svvptc. Memory protection through PMP and Smepmp. 11.1.0 also carries pointer masking (Smmpm, Smnpm, Ssnpm, Sspm, Supm), control-flow integrity (Zicfilp landing pads and Zicfiss shadow stacks), double trap (Smdbltrp, Ssdbltrp), resumable NMI (Smrnmi), AIA CSRs (Smaia, Ssaia), state enable (Smstateen), counter delegation, control transfer records (Smctr, Ssctr), and the Sdtrig debug triggers. The platform-level parts (IMSIC, APLIC, ACLINT, IOMMU) are devices in document 12 and 16.

Vectors: RVV 1.0 with vlen up to 1024 bits (RV_VLEN_MAX in target/riscv/cpu.h), the Zve32x/Zve32f/Zve64x/Zve64f/Zve64d embedded subsets, vector FP16 and BF16 (Zvfh, Zvfhmin, Zvfbfmin, Zvfbfwma, and Zvfbfa added in 11.1), and vector crypto (Zvbb, Zvbc, Zvkg, Zvkned, Zvknha, Zvknhb, Zvksed, Zvksh and the shorthand groups). RVV is expensive to emulate because vl, vtype and LMUL change the meaning of every vector instruction. QEMU folds vl==vlmax, LMUL, SEW, and vta/vma into tb flags so that translation can specialize; ruvm does the same, and the tier 2 optimizer (document 08) additionally specializes loops on a stable vl.

Endianness: with big-endian=on, the target sets MSTATUS_MBE/SBE/UBE at reset and uses the MO_BE memory operation flag for all data accesses while keeping MO_LE on instruction fetch. ruvm routes endianness through the memory op in the IR, so this costs nothing on the fast path.

### PowerPC (ppc, ppc64)

MMU models: 32-bit hash (6xx, 7xx, 74xx), 64-bit hash (POWER5 through POWER11), radix (POWER9 and later), software-loaded TLBs for 4xx and 6xx variants, and the BookE MMU with the e500 MAS registers. Privilege levels are problem state, supervisor, and hypervisor (MSR[HV]), with the PowerNV machine running in hypervisor mode and pseries in LPAR mode with hcalls implemented in hw/ppc/spapr_hcall.c, including the nested KVM-HV API (v1 and v2, hw/ppc/spapr_nested.c). The 11.1 PowerNV nest MMU is a device (document 12). Vectors: AltiVec/VMX, VSX, the POWER10 prefixed instructions (insn64.decode), and MMA (the 30 xv*ger rank-k update patterns in insn64.decode, which update 512-bit accumulators). SPE on e500 has its own register file and gdb XML (power-spe.xml).

### s390x

DAT with up to five table levels (region-first, region-second, region-third, segment, page), EDAT-1 and EDAT-2 large frames, storage keys, low-address and fetch protection, PER, and the channel subsystem (document 12). SIE, the interpretive execution instruction used for nested KVM, is not implemented by QEMU's TCG, so nested virtualization on s390x requires KVM in both QEMU and ruvm. Vectors: the vector facility with enhancements 1 and 2 (S390_FEAT_VECTOR_ENH2 in qemu_V7_1), the vector packed decimal facilities are not in the qemu or max model feature lists, so TCG does not implement them. CPACF message security assist functions exposed in qemu_MAX (MSA_EXT_5, KIMD SHA-512 and others) are implemented as helpers.

### LoongArch (loongarch64)

Privilege levels PLV0 to PLV3, software-refilled TLB with separate STLB and MTLB, the LDDIR/LDPTE page walk helper instructions, and direct mapped windows (DMW). Vectors: LSX (128-bit) and LASX (256-bit). The LVZ virtualization extension is available under KVM only. The la132 32-bit CPU runs in qemu-system-loongarch64.

### Remaining targets

MIPS: R4000-style software-managed TLB with KSEG segments, MIPS32/64 releases 1 to 6, microMIPS and nanoMIPS (I7200), MSA (128-bit SIMD, msa.decode), the DSP ASE, Loongson MMI and extensions (loong-ext.decode, godson2.decode), Octeon (octeon.decode), TX79 (tx79.decode) and the VR54xx extensions. SPARC: SRMMU for sun4m, the sun4u and sun4v MMU with TSBs and the UltraSPARC T1/T2 hypervisor mode, register windows, and VIS1 through VIS4 plus FMAF and IMA as CPU features. HPPA: PA-RISC 1.1 and 2.0 with space registers, the software TLB (including the PA-2.0 TLB format with page sizes), and protection IDs. m68k: 68030 and 68040/68060 MMUs, 68881/68882 FPU with 80-bit extended precision, and ColdFire EMAC and ISA revisions. SH4: UTLB and ITLB, store queues, and the SH4 FPU with its paired-single mode switch (FPSCR.PR and SZ in tb flags). Alpha: PALcode-based system mode with QEMU's own palcode-clipper, a software TLB walked by PALcode, and 8K pages. Xtensa: configuration-defined cores (see below), region protection or MMU v2/v3, windowed registers, and optional DSP coprocessors. Hexagon: VLIW packets of up to four instructions with packet-level commit semantics, HVX vectors (128-byte registers, a separate gdb feature hexagon-hvx.xml), and in system mode the new TLB and interrupt model (target/hexagon/hex_mmu.c, hex_interrupts.c). OpenRISC: software TLB refill. MicroBlaze: optional MMU and FPU chosen by properties. TriCore, RX and AVR have no MMU; AVR has separate code and data address spaces, which ruvm models as two MMU indexes into two address spaces.

## How a target is ported

### Decoders

ruvm-decode (document 07) reads QEMU's decodetree syntax unchanged: the same %fields, &argument sets, @formats, patterns, and {} / [] overlap groups, and generates a Rust decoder that calls `trans_<name>(&mut ctx, &args) -> bool` exactly as decodetree.py generates C calls. Using QEMU's .decode files verbatim is deliberate. They are the most precise machine-readable statement of what QEMU decodes, and when QEMU fixes a decode bug the fix arrives as a small diff to a .decode file that applies to ruvm's copy mechanically. The files are vendored under each target crate's `decode/` directory with a provenance header naming the QEMU commit, and `cargo xtask sync-decode` reports drift against a QEMU checkout.

In 11.1.0 the decodetree coverage (non-comment lines) is: Arm a64.decode 1,399, sve.decode 1,260, sme.decode 840, t32.decode 574, mve.decode 572, neon-dp.decode 386, a32.decode 371, plus smaller files; LoongArch insns.decode 1,868; PowerPC insn32.decode 953 and insn64.decode 218; RISC-V insn32.decode 900 and insn16.decode 183 plus the vendor files xthead.decode, xmips.decode, xlrbr.decode and XVentanaCondOps.decode; SPARC 613; HPPA 439; RX 281; MIPS msa.decode 219, octeon.decode 269 and five small ones; MicroBlaze 169; OpenRISC 150; AVR 123. These targets are ported by writing Rust trans_ functions that emit ruvm-jit IR; the decode layer comes for free.

The targets without decodetree need a different approach each:

- x86 uses the table-driven decoder in target/i386/tcg/decode-new.c.inc (opcode maps with operand type and size codes, VEX and EVEX prefixes, CPUID gating per entry) for most of the instruction set and the older translate.c switch for the rest. ruvm ports the table format into a Rust table plus a small interpreter of operand codes, and does not port the legacy switch; instead every legacy opcode gets a table entry. This is the largest single piece of target work and the reason x86 has its own milestone in M4.
- s390x describes its 920 instruction entries in target/s390x/tcg/insn-data.h.inc as C macro records (opcode, format, input and output operand helpers, operation, condition code handler). ruvm converts the file with a build script into a Rust table and keeps the same "in1/in2/prep/op/wout/cout" pipeline, since it maps one to one onto IR generation.
- Hexagon generates everything from imported Qualcomm semantics files (target/hexagon/imported/*.idef): gen_decodetree.py produces decodetree input, and idef-parser turns semantic definitions into TCG generation code, with hand-written overrides in gen_tcg.h and gen_tcg_hvx.h. ruvm reuses the Python generators up to the decodetree step, then has its own idef-to-IR backend replacing idef-parser's TCG output. Packet semantics (all reads before all writes, predicate and new-value forwarding, store ordering within a packet) are implemented in the shared translate loop as a packet commit phase.
- Xtensa decodes through the libisa interface generated per core (target/xtensa/core-*/xtensa-modules.c.inc from the Tensilica overlay), and the translator maps opcode names to generation functions (translate.c's XtensaOpcodeOps tables). ruvm links the same generated core tables, compiled into Rust statics by a build script, and implements the opcode ops table in Rust. Supporting a new core is still "drop in the overlay", as documented in QEMU.
- m68k, SH4, Alpha, TriCore and the non-decodetree part of MIPS use hand-written switch decoders in QEMU. For these ruvm writes new decodetree files (the language handles fixed 16-bit and 32-bit words and m68k's extension words through multi-word patterns plus custom field loaders) because a declarative decoder is easier to fuzz against a reference. These files are ruvm-original and are offered upstream.

### Helpers

Helpers are Rust functions called from JIT code with a fixed ABI (document 08). The rules: a helper that only computes a value is `#[ruvm_helper(pure)]` so the optimizer can CSE and dead-code eliminate it; a helper that may fault takes the vCPU state and returns through the side table unwinding path; a helper that reads or writes guest memory must use the softmmu access functions, never a raw pointer, so that watchpoints, MMIO, plugins and record/replay see the access. These correspond to QEMU's DEF_HELPER_FLAGS with TCG_CALL_NO_RWG_SE and friends, and the flags are mechanically translated from helper.h during porting. Vector helpers (SVE, SME, RVV, AltiVec, MSA, LSX) are the bulk of helper code in QEMU (target/arm/tcg/sve_helper.c alone is several thousand lines); ruvm writes them as generic Rust over element type and lane count, generated with macros, and then lets the JIT inline the small ones (a predicate test, a single-element move) as IR instead of calls.

### Softfloat rules

ruvm-softfloat is bit exact with QEMU's fpu/softfloat.c and fpu/softfloat-parts.c.inc, including every target's special cases, which are configured through float_status fields in QEMU and through a `FloatStatus` struct with the same fields in ruvm. The per-target differences that must be preserved include:

- NaN propagation order when several operands are NaN (Float2NaNPropRule in include/fpu/softfloat-types.h: Arm's vfp_helper.c selects float_2nan_prop_s_ab for its standard status and float_2nan_prop_ab for another, x86 uses float_2nan_prop_x87 for x87, MMX and SSE status alike, which returns the NaN with the larger significand, and RISC-V runs in default NaN mode so no rule applies), and the separate 3-operand rule for fused multiply-add (Float3NaNPropRule) plus the infinity times zero plus NaN case (FloatInfZeroNaNRule), which differs between Arm, PowerPC, x86 and others.
- Signalling NaN encoding (FloatSNaNRule): the most significant fraction bit means "signalling" on HPPA, SH4 and pre-2008 MIPS, the opposite of IEEE 754-2008, and some modes never detect signalling NaNs.
- Default NaN bit patterns (positive quiet NaN on Arm and RISC-V, the negative "real indefinite" on x86).
- Flush-to-zero and denormals-are-zero, input versus output flushing, and whether tininess and flush-to-zero are detected before or after rounding (tininess_before_rounding and float_ftz_before_rounding versus float_ftz_after_rounding differ between Arm and x86).
- Arm's FEAT_AFP and FPCR.AH alternate handling, which changes NaN selection, flush behavior and exception reporting for a subset of instructions, and FPCR.NEP for scalar merging.
- x87 80-bit extended precision with precision control (FloatX80RoundPrec), and the floatx80_behaviour flags that make x86 and m68k differ on the explicit integer bit, pseudo-infinities and pseudo-NaNs.
- PowerPC FPSCR exception bits (VXSNAN, VXISI, VXIDI, VXZDZ and so on) that record the cause of an invalid operation, and non-IEEE mode.
- Hexagon's own fused multiply-add emulation in target/hexagon/fma_emu.c.
- BF16 and FP8 (E4M3, E5M2) formats with Arm's and RISC-V's differing rounding and saturation rules.

The verification strategy is to compile QEMU's softfloat as a C test oracle (it is GPL, and so is the test crate) and run TestFloat-style generated vectors plus random vectors through both for every operation, rounding mode and FloatStatus configuration used by any target. A failing vector is a release blocker. The JIT may use host FP instructions directly only in the cases document 08 lists as proven equivalent (for example, round-to-nearest double add with no pending exception checks and a status mode where the host produces the same NaN), and each such fast path has a differential test that forces the slow path and compares.

### Porting order within a target

Each target is ported in the same order, and each step has a test gate: (1) state struct, reset and VMState with a round-trip test against QEMU's migration stream for that CPU (document 17); (2) user-level integer instructions, gated by the tests/tcg/<arch> user-mode tests under ruvm linux-user (document 10); (3) FP and vector, gated by the softfloat oracle and the tcg tests for those features; (4) system mode MMU and exceptions, gated by booting the functional-test kernels in tests/functional for that target; (5) gdb XML and register access, gated by the gdbstub tests in tests/tcg/multiarch/gdbstub; (6) CPU models and properties, gated by the identification register dump comparison above.

## Tiering of effort and priority

Targets are grouped by who uses them and how much work they are. The tier decides the milestone (document 23) and the depth of testing, not whether the target ships: all 19 ship by 1.0 because the compatibility contract covers every qemu-system and qemu-user binary.

| Tier | Targets | Milestone | Why | Relative effort |
|---|---|---|---|---|
| 1 | x86_64/i386, aarch64/arm | M4 (JIT), M2 and M6 (KVM, HVF) | Cloud, CI, cross builds, Apple and Arm laptops, the SPEC headline numbers | Very high. x86 decoder and flags; Arm SVE/SME, system registers, ptw.c |
| 2 | riscv64/riscv32, ppc64, s390x, loongarch64 | M6 to M7 | Cross builds, distribution ports and CI; KVM hosts exist for all four | High. RVV and H extension; PowerPC hash and radix MMUs; s390x instruction count |
| 3 | mips family, sparc/sparc64, m68k, hppa, alpha, sh4 | M10 | Retro OS work, firmware, Debian ports, HP-UX and NeXTSTEP style hobby use | Medium. Mostly scalar, old MMUs, FP quirks |
| 4 | hexagon, xtensa, microblaze, or1k, tricore, rx, avr | M10 | Embedded and DSP firmware, toolchain CI (LLVM, GCC, Zephyr) | Low to medium, except Hexagon which is high because of its generators and new system mode |

Within tier 1 the order is aarch64 guest first on both hosts, then x86_64 guest, because the Arm decoder is cleaner to port and flushes out JIT bugs before the x86 decoder arrives. The canon SPEC targets (x86-64 on x86-64, aarch64 on x86-64, x86-64 on aarch64) all need both by M4.

Tier 3 and 4 ports can be done by contributors who are not JIT experts, since the IR, translate loop, softfloat and decoder generator exist by then (document 20).

## Formal specifications: differential testing and generated code

Three families of machine-readable ISA specifications are usable, and each has licensing and coverage limits that decide how ruvm uses it.

### Sail RISC-V

The Sail RISC-V model (github.com/riscv/sail-riscv) is a formal specification of the RISC-V architecture written in Sail and adopted by RISC-V International; it is licensed BSD-2-Clause, so it can be used freely inside a GPL project. It covers RV32 and RV64 with M, S and U modes, virtual memory and a large and growing set of extensions. ruvm uses it three ways. First, lockstep differential testing: a harness runs a random or directed instruction stream on the Sail-generated C emulator and on ruvm's JIT (and interpreter backend), comparing architectural state after every instruction through ruvm's plugin register API; this catches both decoder and semantic bugs, and RISC-V corner cases such as misaligned accesses, vector tail and mask agnostic behaviour and CSR side effects are exactly where JITs go wrong. Second, the test generator produces trap and CSR sequences that QEMU's tcg tests do not cover. Third, Pydrofoil (Bolz-Tereick et al., ECOOP 2025, https://drops.dagstuhl.de/storage/00lipics/lipics-vol333-ecoop2025/LIPIcs.ECOOP.2025.3/LIPIcs.ECOOP.2025.3.pdf) makes the reference fast enough to boot Linux in the loop: the paper reports 233x and 255x speedups over the Sail-generated simulator on its Linux boot and SPEC benchmarks, while still being 26.7x slower than QEMU. That is fast enough for nightly full-boot lockstep comparisons at checkpoints, which pure Sail is not.

Differences between Sail and QEMU are not automatically bugs in ruvm. Where QEMU and Sail disagree (usually a WARL CSR field or an implementation-defined choice), ruvm matches QEMU, records the difference in a known-divergence list in the test crate, and reports it upstream to whichever side is wrong.

### Arm ASL and the machine-readable specification

Arm publishes its A-profile architecture as the AARCHMRS machine-readable specification in two packages. The full XML package contains the ASL pseudocode and descriptions under a proprietary Arm licence that is not suitable for inclusion in an open source project. A second package, released under BSD-3-Clause, contains JSON descriptions of instructions, system registers and features but no shared pseudocode and no descriptive text; its latest release notes (2026-03) describe it as approximately equivalent to the 2026-03 XML release (https://developer.arm.com/documentation/110173/). Arm is also moving its pseudocode from ASL version 0 to ASL1, with the language reference (DDI 0626) and the ASLRef reference implementation public.

ruvm uses the BSD JSON to generate and cross-check data, not semantics: system register encodings, field layouts, reset values where specified, feature dependencies (which FEAT_ implies which), and instruction encodings. The Arm crate's system register table is checked in CI against the JSON: every register QEMU implements must have the encoding and field widths the JSON gives, and the feature dependency graph is used to reject invalid property combinations the same way arm_cpu_realizefn does, with the JSON as a second opinion. For semantics, the Sail Arm models (github.com/rems-project/sail-arm) are the usable route: Armv8.5-A, Armv9.3-A and Armv9.4-A models machine-translated from Arm's internal ASL by asl_to_sail and released under BSD-3-Clause-Clear by agreement with Arm. ruvm runs them as a lockstep reference for AArch64 user and system instructions in the same harness as RISC-V. Generated Sail emulators are slow, and Pydrofoil's Arm support is described by its authors as experimental, so the Arm models are used for directed tests and short random streams, not boots. Features newer than Armv9.4 (SME2p1, FP8, GCS details) have no Sail model yet and rely on QEMU differential testing only.

### x86

There is no complete formal x86 model comparable to Sail RISC-V. The best open option is the Sail x86 model translated from the ACL2 X86isa model of Goel, Hunt and Kaufmann (github.com/rems-project/sail-x86-from-acl2), which the Sail project itself describes as a core user-mode fragment. ruvm uses it for integer and flags semantics only (the area where lazy flag optimization bugs hide) through isla-testgen style directed tests. Everything else on x86 is tested differentially against QEMU and, on x86 hosts, against real hardware via KVM: the same instruction stream runs natively in a KVM guest and under the JIT, and user-visible state is compared. Hardware is the reference for x86 in practice.

### QEMU itself as the reference

For every target, QEMU 11.1 is the primary oracle. The lockstep harness (document 22) runs QEMU with a small TCG plugin that exports register state after each instruction, and ruvm with the same plugin (the plugin ABI is identical), and compares. This is the only oracle that covers every target, including all the tier 3 and 4 targets that have no formal model. The formal specs catch the cases where QEMU and ruvm share a bug because ruvm ported QEMU's code.

### Generated decoders from specs

Generating decoders from Sail or the Arm JSON was considered and rejected for the main path. The decodetree files encode QEMU's choices about which encodings are UNDEFINED versus CONSTRAINED UNPREDICTABLE and how overlapping patterns are grouped, and matching QEMU requires matching those choices. Generated decoders are used as test oracles instead: a decoder generated from the Arm JSON classifies every 32-bit AArch64 encoding (all 2^32 of them), and CI compares the set of encodings ruvm accepts against the set the spec defines for the enabled features. Every difference must either be listed as a deliberate QEMU behaviour or fixed. The same exhaustive sweep is done for RISC-V 32-bit and 16-bit encodings against sail-riscv's decoder.

## gdb register descriptions

The gdbstub (ruvm-gdbstub, layer L5 in document 03) sends target.xml assembled from per-feature XML. QEMU keeps static feature files in gdbstub/gdb-xml/ (in 11.1.0: aarch64-core, aarch64-fpu, aarch64-mte, aarch64-pauth, aarch64-sme2, arm-core, arm-neon, arm-vfp, arm-vfp3, arm-vfp-sysregs, arm-m-profile, arm-m-profile-mve, i386-32bit, i386-64bit, i386-64bit-apx, the -linux variants, power-core, power64-core, power-fpu, power-altivec, power-vsx, power-spe, riscv-32bit/64bit cpu, fpu and virtual, s390x-core64 and the s390 acr, cr, fpr, gs, vx and virt files, loongarch base32, base64, fpu, lsx and lasx, sparc32 and sparc64 files, hexagon-core and hexagon-hvx, m68k-core, m68k-fp, cf-core, cf-fp, microblaze-core, microblaze-stack-protect, or1k-core, rx-core, avr-cpu, alpha-core) and generates others at runtime with gdb_feature_builder: Arm's system-registers.xml, sve-registers.xml, sme-registers.xml and tls-registers.xml, the M-profile system and security extension features, and RISC-V's org.gnu.gdb.riscv.csr and org.gnu.gdb.riscv.vector features whose contents depend on enabled extensions and vlen.

ruvm vendors the static files byte for byte (they are part of the compatibility surface: IDEs and scripts depend on register names and numbers) and ports each dynamic generator so it emits the same XML text for the same configuration, including register order, because gdb numbers registers in document order and register numbers leak into user scripts and into plugin register handles. A golden-file test per target and per representative CPU configuration compares ruvm's full target.xml, and the concatenated register numbering, against QEMU's output captured through qXfer:features:read.
