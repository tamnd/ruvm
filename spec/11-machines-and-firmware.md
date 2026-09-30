# 11. Machines and firmware

This document specifies how ruvm builds a machine: the `Machine` trait, versioned machine types and their compat property chains, CPU topology and NUMA, the catalog of boards per architecture and the order in which we implement them, and everything between power-on and the first guest kernel instruction (firmware blobs, pflash, fw_cfg, ACPI, SMBIOS, device trees, direct kernel boot, IGVM, boot order). The device model the machines are assembled from is in document 12 (non-virtio devices) and document 13 (virtio). The object model and property system underneath are in document 04. Migration stream compatibility, which is the main reason versioned machines exist, is in document 17.

All counts in this document were taken from a build of the QEMU v11.1.0 tag (all system targets except hexagon, macOS arm64 host, default configure options) by running `qemu-system-<arch> -machine help`, `-device help` and QMP `qom-list-types`, and cross-checked against the source tree. Machines that only build on Linux or with Xen (listed below) were counted from source.

## Reference numbers

| Item | Value in QEMU 11.1.0 |
| --- | --- |
| System emulator binaries | 29 (28 built here plus qemu-system-hexagon) |
| Distinct board families across all targets | 180 counted from `-machine help` plus source for hexagon, before Linux-only and Xen-only machines |
| Machine type names including every versioned variant | 281 (same basis) |
| Versioned families | pc-i440fx, pc-q35, arm virt, pseries and s390-ccw-virtio each register 18 versions (5.2 to 11.1); m68k virt registers 17 (6.0 to 11.1); loongarch virt has only 11.1 |
| hw_compat_X_Y arrays in hw/core/machine.c | 4.1 through 11.0 (older arrays still exist because older version code paths chain through them, but machine types older than 5.2 are no longer registered) |
| Expected ACPI blobs in tests/data/acpi | 188 files: x86/q35 82, x86/pc 33, x86/microvm 10, aarch64/virt 39, loongarch64/virt 17, riscv64/virt 7 |
| C lines under hw/ | about 811,000 |

The machine version lifecycle is encoded in include/hw/core/boards.h: `MACHINE_VER_DEPRECATION_MAJOR 3` and `MACHINE_VER_DELETION_MAJOR 6`. A versioned machine is marked deprecated once it is more than three years old and stops being registered after six. The `MACHINE_VER_DELETION` check in the registration macro is why machine types older than 5.2 no longer appear in `-machine help`. ruvm implements the same arithmetic in `ruvm_machine_core::lifecycle` with the reported QEMU-compatible version (11.1.0) as input, so `-machine help` lists the same set with the same "(deprecated)" suffixes.

## The Machine trait

A machine in QEMU is a QOM class derived from `TYPE_MACHINE` whose `MachineClass` carries a large set of fields (`init`, `default_cpu_type`, `max_cpus`, `default_ram_id`, `smp_props`, `possible_cpu_arch_ids`, `get_hotplug_handler`, `compat_props`, `no_floppy`, `default_nic`, and many more) and whose instance `MachineState` holds the parsed `-m`, `-smp`, `-numa`, kernel paths, the RAM backend and the fw_cfg handle. The board init function in, for example, `pc_q35_init` in hw/i386/pc_q35.c or `machvirt_init` in hw/arm/virt.c creates CPUs, the memory map, chipset devices and firmware in an order that the guest can observe (PCI slot numbers, IRQ routing, fw_cfg file order, ACPI table contents all depend on it).

ruvm splits this into static class data and a trait for behaviour. The trait lives in ruvm-hw-core (L2) so that devices can call back into the machine through narrow interfaces (hotplug handler, IRQ routing queries) without depending on L4 crates. Implementations live in the ruvm-machine-<family> crates (L4).

```rust
pub struct MachineClassInfo {
    pub name: &'static str,              // "pc-q35-11.1"
    pub alias: Option<&'static str>,     // Some("q35") on the latest version only
    pub family: &'static str,            // "pc_q35", used for -machine help grouping
    pub desc: &'static str,
    pub version: Option<MachineVersion>, // Some(11.1) for versioned machines
    pub is_default: bool,
    pub deprecation: Option<&'static str>,
    pub max_cpus: u32,
    pub min_cpus: u32,
    pub default_cpu_type: Option<&'static str>,
    pub valid_cpu_types: &'static [&'static str],
    pub default_ram_id: Option<&'static str>, // "pc.ram", "mach-virt.ram"
    pub default_ram_size: u64,
    pub default_display: Option<&'static str>,
    pub default_nic: Option<&'static str>,
    pub block_default_type: BlockInterfaceType,
    pub smp: SmpProps,                   // which topology levels are accepted
    pub numa: NumaProps,                 // mem_is_supported, auto_enable, node_mem_align
    pub dynamic_sysbus: &'static [&'static str],
    pub compat: &'static CompatChain,    // see next section
    pub knobs: MachineKnobs,             // typed non-property compat flags
}

pub trait Machine: Object + Send + Sync {
    fn class_info(&self) -> &'static MachineClassInfo;
    /// Create CPUs, memory map, devices. Equivalent of MachineClass::init.
    fn init(&mut self, cx: &mut BoardCx) -> Result<()>;
    /// Called after all -device options are realized (QEMU's machine_done notifier).
    fn machine_done(&mut self, cx: &mut BoardCx) -> Result<()>;
    fn possible_cpus(&self) -> &[PossibleCpu];
    fn cpu_index_to_props(&self, idx: u32) -> CpuInstanceProperties;
    fn hotplug_handler(&self, dev: &dyn Device) -> Option<&dyn HotplugHandler>;
    fn reset(&mut self, kind: ShutdownCause);
    fn firmware(&self) -> &FirmwarePlan;
    fn fw_cfg(&self) -> Option<&FwCfg>;
    fn acpi(&self) -> Option<&dyn AcpiBuilder> { None }
    fn fdt(&self) -> Option<&Fdt> { None }
    fn boot_order_changed(&mut self, order: &str) -> Result<()> { Ok(()) }
}
```

`BoardCx` owns the root `MemoryRegion` (document 05), the accelerator (document 06), the fw_cfg builder, the bootindex registry and the control lock. Board init runs with the control lock held on one thread, so running board code in the same order as the C code it was ported from reproduces the guest-observable ordering exactly.

We keep QEMU's init phases because the command line semantics depend on them. `qemu_init` in system/vl.c creates the machine object, applies `-machine` properties, creates the accelerator, calls `machine_run_board_init` (hw/core/machine.c), processes `-device` in order, then calls `qdev_machine_creation_done`, which fires the machine done notifiers. ACPI tables, the `bootorder` file and SMBIOS are built in machine done because they must see `-device` additions. ruvm-system walks the same phases with the same failure points, so an invalid `-smp` or `-numa` combination fails at the same stage with the same message text (document 02 treats error strings as part of the libvirt contract).

## Versioned machine types and compat properties

### How QEMU does it

Each versioned machine has an options function that calls the next newer version's options function and then appends compat arrays. From hw/i386/pc_q35.c in 11.1:

```c
static void pc_q35_machine_11_0_options(MachineClass *m)
{
    pc_q35_machine_11_1_options(m);
    compat_props_add(m->compat_props, hw_compat_11_0, hw_compat_11_0_len);
    compat_props_add(m->compat_props, pc_compat_11_0, pc_compat_11_0_len);
}
```

`hw_compat_X_Y` (hw/core/machine.c) holds property overrides that apply to every architecture, `pc_compat_X_Y` (hw/i386/pc.c) holds x86 ones, and other families have their own (`virt_machine_X_Y_options` in hw/arm/virt.c sets fields and calls the shared arrays, spapr and s390 do the same). Each entry is a `GlobalProperty { driver, property, value }`. At machine creation, `compat_props` are registered as global properties (`object_register_sugar_prop` style, see `machine_register_compat_props`), and every object whose type is, or inherits from, `driver` gets `property` set to `value` before realize. User `-global` options are applied after compat props, so the user wins. Some entries name an abstract parent type, such as `{ TYPE_PCI_DEVICE, "x-pcie-ext-tag", "false" }` in hw_compat_9_1, which reaches every PCI device.

The 11.0 compat array shows the kind of drift these arrays capture in one release: `virtio-mmio` `x-override-queue-size` back to 1024, `chardev-vc` encoding back to cp437, `tpm-crb` chunked command capability off, `tpm-tis-device` PPI off, four SMMUv3 defaults (`ats=off`, `ril=on`, `ssidsize=0`, `oas=44`), and a migration property. The x86 side (`pc_compat_10_1`) has `{ "mch", "extended-tseg-mbytes", "16" }` and `{ TYPE_X86_CPU, "x-migrate-error-code", "false" }`. Not every compat difference is a property: older versions also flip `MachineClass` and `PCMachineClass` fields directly, for example `m->smp_props.prefer_sockets = true` for pc-q35 6.1 and older (hw/i386/pc_q35.c line 542) and for arm virt 6.1 and older.

### How ruvm represents it

We treat compat data as data, not code. Each array is a `static` slice of `CompatProp` in a generated Rust module, named identically to the C array so a grep across both trees lines up.

```rust
pub struct CompatProp { pub driver: &'static str, pub property: &'static str, pub value: &'static str }

pub static HW_COMPAT_11_0: &[CompatProp] = &[
    CompatProp { driver: "virtio-mmio", property: "x-override-queue-size", value: "1024" },
    CompatProp { driver: "chardev-vc", property: "encoding", value: "cp437" },
    // ...
];

pub struct CompatChain {
    pub version: MachineVersion,
    pub arrays: &'static [&'static [CompatProp]], // newest first, as QEMU appends them
    pub newer: Option<&'static CompatChain>,
}
```

The module is produced by `cargo xtask compat-import --qemu-tag v11.1.0`, which parses hw/core/machine.c, hw/i386/pc.c, hw/arm/virt.c, hw/ppc/spapr.c, hw/s390x/s390-virtio-ccw.c, hw/m68k/virt.c and hw/loongarch/virt.c with a small C initializer parser (the arrays are plain brace initializers of string literals and a few macros such as `TYPE_ARM_SMMUV3`, which the tool resolves from the headers). The generated file is checked in and reviewed. CI reruns the import against the pinned tag and fails if the checked-in file differs, which catches both hand edits and a forgotten import after a QEMU rebase. When QEMU 11.2 ships, the import adds `HW_COMPAT_11_1`.

Typed knobs that are not QOM properties (`prefer_sockets`, `pcmc->pci_root_uid`, `default_cpu_version` and similar) go into a `MachineKnobs` struct, written by hand per version as a function of the newer version's knobs, mirroring the C options function. Each is covered by a fingerprint test.

Application order is identical to QEMU: class defaults, then compat props from the chain (oldest array last so it wins, matching the append order and the "last registered global wins" rule in qom/object.c `object_apply_global_props`), then accelerator compat props (`AccelClass::compat_props`), then `-global` and `-set` from the user. A compat prop whose driver type does not exist in this build is ignored silently, as in QEMU, but a compat prop whose property does not exist on an existing type is a hard error at startup in debug builds and a warning in release builds. QEMU treats the second case as an error only when the object is created; we check eagerly because a typo in a compat array silently changes guest ABI, which is the bug class this whole mechanism exists to prevent.

### The guest ABI guarantee

A versioned machine promises that a guest started on `pc-q35-10.1` sees the same hardware on every later ruvm release and on QEMU 10.1 through 11.1, and that migration between them works. ruvm enforces this with a machine fingerprint. For every registered versioned machine and a fixed set of 14 representative command lines per family (default, with NUMA, with a vIOMMU, with each disk bus, with TPM, with maximal SMP, and so on), `cargo xtask abi-fingerprint` records:

- the QOM composition tree with every property value after realize (document 04 describes `qom-list` walking),
- the flattened guest physical and I/O address space with region names and sizes,
- the fw_cfg file directory (names, sizes and SHA-256 of contents),
- the ACPI tables after linker resolution and the SMBIOS blob,
- the device tree blob for FDT machines, canonicalized by `dtc -I dtb -O dts`,
- the list of RAMBlock ids and sizes (these names are part of the migration stream, for example `pc.ram`, `pc.bios`, `pc.rom`, `/rom@etc/acpi/tables`, `0000:00:01.0/vga.vram`),
- the VMState section list with names, instance ids and version ids (document 17).

The same script runs against QEMU 11.1 (via QMP and qtest) to produce the reference, and differences fail CI with a structured diff. Once ruvm has shipped a machine version, its fingerprint is frozen and a change requires an `abi-break` label and a written justification. Migration interop tests in document 17 check that the fingerprint covers enough.

## CPU topology

`-smp` accepts `cpus`, `maxcpus`, `drawers`, `books`, `sockets`, `dies`, `clusters`, `modules`, `cores`, `threads`, and QEMU stores them in `CpuTopology` (include/hw/core/boards.h). Which levels a machine accepts is declared in `SMPCompatProps`: pc machines set `dies_supported` and `modules_supported` (hw/i386/pc.c), arm virt and sbsa-ref set `clusters_supported`, s390-ccw-virtio sets `books_supported` and `drawers_supported`. A level a machine does not support must be 1 or omitted, otherwise `machine_parse_smp_config` in hw/core/machine-smp.c errors out with a message naming the level. ruvm ports `machine_parse_smp_config` line for line into `ruvm_machine_core::smp::parse`, including the defaulting rules that bite users: when only `cpus` is given, older machines with `prefer_sockets` fill sockets first and newer ones fill cores first; `maxcpus` defaults to `cpus`; the product of all levels must equal `maxcpus`. QEMU's tests/unit/test-smp-parse.c tables become Rust tests.

Topology is guest visible through several channels, all of which we generate from one `Topology` value:

- x86: APIC IDs are built by `x86_topo_ids_from_idx` and `x86_apicid_from_topo_ids` (include/hw/i386/topology.h), with each level taking `ceil(log2(count))` bits, so `-smp 6,cores=3` produces non-contiguous APIC IDs. CPUID leaves 0x4, 0xB, 0x1F and AMD 0x8000001E report the same fields. ruvm-target-x86 (document 09) consumes the topology struct and the same bit-width function.
- Arm and RISC-V: PPTT (built by `build_pptt` in hw/acpi/aml-build.c) and the `cpu-map` node in the device tree. MPIDR affinity values on arm virt are assigned by `virt_cpu_mp_affinity`, which packs 16 CPUs per Aff0 value for GICv3 and GICv5 (the ICC_SGIxR target list limit) and 8 for GICv2, independent of the `-smp` levels, and we copy that rule.
- s390x: the STSI 15.1.x topology list and the CPU topology facility (`-cpu ...,ctop=on`) with drawers and books, plus the dedicated and polarization properties on each CPU.
- ppc spapr: `ibm,chip-id` and threads per core. The vCPU id spacing depends on `vsmt` (`spapr_vcpu_id` in hw/ppc/spapr.c), threads are capped by `ppc_compat_max_vthreads` for the CPU compat mode, and under KVM the host core's SMT mode constrains `vsmt`.

`-machine smp-cache` (the `SmpCache` array) lets a user set the topology level at which each cache level is shared (for example `l3=die`); it is supported on x86 and on arm virt. ruvm implements it as a field of `Topology` and feeds it to CPUID leaf 4 and to PPTT cache nodes.

`possible_cpu_arch_ids` gives each potential CPU slot a stable arch id and props (socket-id, core-id, thread-id, node-id). `query-hotpluggable-cpus` and `device_add` of a CPU are driven from it. We implement it per family and test it by comparing `query-hotpluggable-cpus` output with QEMU across a matrix of topologies.

## NUMA and HMAT

`-numa node,nodeid=,memdev=,cpus=,initiator=`, `-numa dist`, `-numa cpu`, `-numa hmat-lb` and `-numa hmat-cache` are parsed into `NumaState` by hw/core/numa.c. Memory for a node comes from a `memory-backend-*` object (document 05). The legacy `-numa node,mem=` form is only accepted on machines with `numa_mem_supported = true`; pc, q35 and arm virt set it only in their 5.0 and older options functions, so no machine version registered in 11.1 accepts it, and we reject it identically.

Guest visibility is SRAT and SLIT on ACPI machines, the `numa-node-id` and `distance-map` properties in device trees, and `ibm,associativity` on spapr (with `FORM1` or `FORM2` affinity chosen by what the guest negotiates in client architecture support, see the `OV5_FORM2_AFFINITY` checks in hw/ppc/spapr_numa.c). HMAT (hw/acpi/hmat.c) emits three structure types: Memory Proximity Domain Attributes, System Locality Latency and Bandwidth Information, and Memory Side Cache Information. The expected-blob tests `HMAT.acpihmat`, `HMAT.acpihmat-noinitiator` and `HMAT.acpihmat-generic-x` in tests/data/acpi/x86/q35 exercise the three shapes QEMU cares about: normal initiators, memory-only nodes without an initiator, and nodes whose initiator is an `acpi-generic-initiator` object (a PCI device acting as initiator) or `acpi-generic-port` (a CXL host bridge). ruvm-hw-acpi ports these objects and the HMAT builder directly.

NUMA node assignment of CPUs follows `numa_complete_configuration` and the machine's `get_default_cpu_node_id` hook. When the user specifies nodes but no CPU mapping, the per-machine hook (`x86_get_default_cpu_node_id` in hw/i386/x86.c, `virt_get_default_cpu_node_id` in hw/arm/virt.c) decides. QEMU warns when some CPUs up to maxcpus are not in any node, and `validate_cpu_cluster_to_numa_boundary` warns when CPUs of one socket and cluster land in different nodes. We port the hooks and both warnings with identical text.

## Machine families per architecture

Counts below are distinct board families from the 11.1.0 build (a versioned family counts once). Linux-only and Xen-only machines are listed separately at the end of each row where they exist.

| Guest arch | Boards | Notable members |
| --- | --- | --- |
| x86 (i386, x86_64) | 4 built on macOS: pc (i440FX), q35, microvm, isapc. Plus on Linux or Xen: nitro-enclave, nitro (target independent, needs the nitro accelerator), xenfv-4.2, xenpv, xenpvh | 18 versions each of pc-i440fx and pc-q35, 5.2 to 11.1 |
| Arm (arm, aarch64) | 89 (76 in qemu-system-arm, all 76 also in aarch64), plus vmapple (HVF only) and xenpvh | virt (18 versions, 5.2 to 11.1), sbsa-ref, raspi0 to raspi4b, 22 Aspeed BMC boards, mps2/mps3, imx6/7/8mm/8mp, versal, zynqmp, npcm7xx/8xx, orangepi-pc, stm32 |
| RISC-V (riscv32, riscv64) | 12 | virt, spike, sifive_e, sifive_u, microchip-icicle-kit, opentitan (rv32 only), k230, tt-atlantis, xiangshan-kunminghu, amd-microblaze-v-generic, boston-aia, shakti_c (deprecated) |
| PowerPC (ppc, ppc64) | 18 | pseries (18 versions), powernv8/9/10/10-rainier/11, mac99, g3beige, ppce500, mpc8544ds, 40p, sam460ex, bamboo, pegasos1/2, amigaone, virtex-ml507, ppe42_machine |
| s390x | 1 | s390-ccw-virtio (18 versions, 5.2 to 11.1) |
| MIPS (4 binaries) | 6 | malta, boston, fuloong2e, loongson3-virt, magnum, pica61 |
| SPARC (sparc, sparc64) | 13 | SS-5, SS-10, SS-20, SS-4, SS-600MP, SPARCClassic, SPARCbook, LX, Voyager, leon3_generic, sun4u, sun4v, niagara |
| LoongArch | 1 | virt (versioned, only 11.1 so far) |
| m68k | 5 | virt (17 versions), q800, next-cube, an5206, mcf5208evb |
| Xtensa | 10 | sim, lx60, lx200, kc705, ml605 and their nommu variants |
| HPPA | 4 | B160L, C3700, 715, A400 |
| AVR | 4 | arduino-uno, arduino-mega, arduino-mega-2560-v3, arduino-duemilanove |
| MicroBlaze | 3 | petalogix-s3adsp1800, petalogix-ml605, xlnx-zynqmp-pmu |
| Hexagon | 2 (from source) | virt, V66G_1024 |
| OpenRISC | 2 | or1k-sim, virt |
| TriCore | 2 | tricore_testboard, KIT_AURIX_TC277_TRB |
| RX | 2 | gdbsim-r5f562n7, gdbsim-r5f562n8 |
| Alpha | 1 | clipper |
| SH4 | 1 | r2d |

Arm dominates: 89 boards and about 830 to 890 concrete QOM device types reachable from qemu-system-aarch64, most of them SoC-internal. A rewrite that tries to port that long tail at once will not finish.

### Priority tiers

Tier 0 (M2): x86 microvm and pc-q35 latest version, KVM only. These carry the performance targets and give us a real guest for everything else.

Tier 1 (M2 to M6): all pc-q35 and pc-i440fx versions 5.2 to 11.1 with migration interop (M5), isapc, arm virt all versions, riscv virt, s390-ccw-virtio all versions, pseries all versions, loongarch virt. These are what libvirt users run. Tier 1 is gated on bit-identical ACPI blobs and fingerprints.

Tier 2 (M8 to M10): sbsa-ref, nitro-enclave and nitro, xenfv and xenpvh, powernv8 to 11, m68k virt, ppce500, raspi3b and raspi4b, the most used Aspeed boards (ast2600-evb, ast2700a1-evb and the BMC boards with functional tests), imx8mp-evk and imx8mm-evk, versal and zcu102, sifive_u, microchip-icicle-kit, spike, malta, q800, mac99 and g3beige, sun4u, clipper, the HPPA machines. Criterion: a functional test in tests/functional boots a real OS on it.

Tier 3 (M10 and after, contributions welcome): everything else, ported with the long-tail process in document 12 (mechanical translation of the board file plus its SoC and device crates, gated on the board's qtest and functional tests passing). A board without a test gets one written during the port, with QEMU as the oracle.

## Firmware

### Blob discovery and identity

ruvm does not build firmware. It uses the same blobs QEMU 11.1 ships in pc-bios/ (bios-256k.bin, bios-microvm.bin, qboot.rom, the edk2-*.fd.bz2 images, opensbi-riscv{32,64}-generic-fw_dynamic.bin, slof.bin, vof.bin, skiboot.lid, s390-ccw.img, openbios-*, u-boot.e500, u-boot-sam460.bin, hppa-firmware*.img, palcode-clipper, the vgabios-*.bin family, pxe-*.rom and efi-*.rom, linuxboot_dma.bin, pvh.bin, multiboot_dma.bin, kvmvapic.bin), found through the same search path: `-L` directories first, then the configured data directory, then the directory relative to the executable (`qemu_add_data_dir` and `qemu_find_file` in util/datadir.c). A distribution that already installs QEMU's firmware packages (seabios, edk2-ovmf, ipxe-qemu, opensbi) serves ruvm unchanged.

Identical blobs are a requirement. Firmware lives in RAMBlocks (`pc.bios`, `pc.rom`, `/rom@genroms/linuxboot_dma.bin`) and in option ROM BARs whose sizes are guest visible and migrated, so a SeaBIOS build of a different size would make a QEMU to ruvm migration fail at the RAMBlock size check. ruvm reuses QEMU's `romfile` and `romsize` semantics and power-of-two ROM padding.

### Firmware per family

- x86 BIOS: SeaBIOS (bios-256k.bin for pc and q35, bios.bin 128 KiB for isapc and legacy configs, bios-microvm.bin for microvm) or qboot.rom for microvm. The BIOS is mapped at the top of the 4 GiB space and its last 128 KiB aliased at 0xE0000 to 0xFFFFF (`x86_bios_rom_init` in hw/i386/x86-common.c). PAM registers in i440FX and MCH control shadowing (document 12 covers PAM as part of the host bridges).
- x86 UEFI: OVMF from EDK2 via two pflash devices (`pc.flash0` code, read only, and `pc.flash1` vars), or via `-bios` for the stateless build. Secure boot builds require SMM (`-machine smm=on`) and `-global driver=cfi.pflash01,property=secure,value=on`. There is also the `uefi-vars-x64` and `uefi-vars-sysbus` device in hw/uefi (docs/devel/uefi-vars.rst), which implements the UEFI variable service on the host side (the `host-uefi-vars` firmware feature) so that vars no longer need SMM isolation in pflash. roms/edk2-version in the 11.1.0 tree records edk2-stable202408 as the bundled build.
- Arm: EDK2 (edk2-aarch64-code.fd, edk2-arm-code.fd) in two 64 MiB cfi01 flash banks at 0x0 and 0x04000000 on virt; with `-kernel` and no firmware, QEMU's built-in boot stub. sbsa-ref needs its own TF-A plus EDK2 build, not shipped by QEMU. U-Boot for several boards via `-bios`. Aspeed and NPCM boards use the vbootrom (ast27x0_bootrom.bin, npcm7xx_bootrom.bin, npcm8xx_bootrom.bin).
- RISC-V: OpenSBI fw_dynamic is the default `-bios` on virt, spike and sifive_u. It is loaded at the start of DRAM (0x80000000 on virt) and the next stage is found through the `fw_dynamic_info` structure QEMU writes (hw/riscv/boot.c, `riscv_setup_rom_reset_vec`). EDK2 for riscv64 via pflash on virt.
- ppc64 pseries: SLOF (slof.bin) by default, or VOF (vof.bin, the minimal Virtual Open Firmware, `-machine x-vof=on`) which skips SLOF and boots much faster. powernv: skiboot.lid plus a PNOR image. mac99 and g3beige: OpenBIOS. ppce500: U-Boot (u-boot.e500). sam460ex: u-boot-sam460.bin.
- s390x: s390-ccw.img is the IPL BIOS, loaded into guest memory and run as the first program; it understands virtio-blk-ccw, virtio-scsi-ccw, virtio-net-ccw and DASD passthrough, and reads `loadparm` and the boot device list from the IPL parameter block QEMU fills (hw/s390x/ipl.c).
- Alpha: palcode-clipper. HPPA: hppa-firmware.img (a SeaBIOS port) and hppa-firmware64.img. SPARC: OpenBIOS. m68k q800: its own ROM image supplied by the user or direct kernel boot.

### Firmware descriptor JSON

docs/interop/firmware.json defines the descriptor schema that libvirt uses to pick firmware. It lists interface types (`bios`, `openfirmware`, `svsm`, `uboot`, `uefi`), mapping devices (`flash`, `kernel`, `memory`, `igvm`), features (`acpi-s3`, `acpi-s4`, `amd-sev`, `amd-sev-es`, `amd-sev-snp`, `intel-tdx`, `enrolled-keys`, `requires-smm`, `secure-boot`, `host-uefi-vars`, `verbose-dynamic`, `verbose-static`) and targets with machine globs such as `pc-q35-*`. QEMU installs files like 60-edk2-x86_64.json into `/usr/share/qemu/firmware`, and consumers scan `/usr/share/qemu/firmware`, `/etc/qemu/firmware` and `$XDG_CONFIG_HOME/qemu/firmware` with filename-based priority and override.

QEMU does not read these descriptors; libvirt does. ruvm keeps that split, installs the same descriptors, and generates a parser from firmware.json (a QAPI schema) with ruvm-qapi so the native `ruvm run` CLI can resolve `--firmware uefi,secure-boot` the way libvirt does. The legacy CLI never selects firmware implicitly.

### pflash

`cfi.pflash01` (Intel command set, hw/block/pflash_cfi01.c) and `cfi.pflash02` (AMD command set) are ported as ruvm-hw-storage devices. Two details matter for performance and correctness. First, reads of flash in array mode must be RAM speed, so the device maps the backing as a ROM device region (`memory_region_init_rom_device`) that switches to MMIO dispatch only while a command is in progress. KVM cannot execute from MMIO, and OVMF executes in place from flash during SEC and PEI, so the flip must be exact. Second, writes are persisted through the block layer (document 14) with the same sector granularity QEMU uses (`pflash_update` writes back the dirty 512-byte range). We port this code rather than reimplementing it.

## fw_cfg

fw_cfg (hw/nvram/fw_cfg.c, 1,322 lines) is the channel between the machine and the firmware. It has a 16-bit selector register, a data register and a DMA interface (docs/specs/fw_cfg.rst; since QEMU 2.9 guest writes are only possible through DMA).

| Machine | Selector | Data | DMA address register |
| --- | --- | --- | --- |
| x86 (pc, q35, microvm) | I/O 0x510 (16 bit) | I/O 0x511 (8 bit) | I/O 0x514, 64-bit big endian: one 64-bit write, or the high half at offset 0 then the low half at offset 4, which triggers the transfer (`fw_cfg_dma_mem_write`) |
| arm virt | MMIO 0x09020008 | MMIO 0x09020000 (up to 64 bit) | MMIO 0x09020010 |
| riscv virt | MMIO 0x10100008 | MMIO 0x10100000 | MMIO 0x10100010 |
| loongarch virt, others | MMIO at a board-specific base, same layout | | |

Well-known selectors are defined in include/standard-headers/linux/qemu_fw_cfg.h: 0x00 signature ("QEMU"), 0x01 ID (bit 0 traditional interface, bit 1 DMA), 0x02 UUID, 0x03 RAM size, 0x05 CPU count, 0x07 to 0x18 kernel, initrd, command line and setup blobs, 0x0c boot device, 0x0d NUMA, 0x0e boot menu, 0x0f max CPUs, 0x19 the file directory, 0x20 onward files. Bit 14 (0x4000) is the legacy write channel and bit 15 (0x8000) selects arch-local keys (x86 uses 0x8000 ACPI tables legacy, 0x8001 SMBIOS entries legacy, 0x8002 IRQ0 override, 0x8004 HPET). The file directory is a big-endian count followed by `{ u32 size; u16 select; u16 reserved; char name[56] }` entries sorted by name, as the insertion loop in `fw_cfg_add_file_callback` guarantees. The number of slots defaults to 0x20 (`FW_CFG_FILE_SLOTS_DFLT`) and can be changed with the `x-file-slots` property.

The DMA interface: the guest writes the guest physical address of a `FWCfgDmaAccess { be32 control; be32 length; be64 address; }` to the DMA register. Control bits are error 0x01, read 0x02, skip 0x04, select 0x08 (with the selector in the upper 16 bits) and write 0x10. The device performs the transfer synchronously in `fw_cfg_dma_transfer` and clears control to 0 on success or sets the error bit. Reading the DMA register returns the signature "QEMU CFG" (0x51454d5520434647). The synchronous transfer runs on the vCPU thread in QEMU under the BQL. In ruvm it runs on the vCPU thread holding only the fw_cfg device lock, and because the transfer reads guest memory through the normal DMA address space (document 05) it respects vIOMMU mappings for machines that put fw_cfg behind one. Large blobs (a 30 MB kernel, a 200 MB initrd) are copied with a single memcpy from a `Bytes` handle rather than QEMU's per-entry buffer path, which is one of the places the microvm boot budget in document 21 is won.

The files QEMU 11.1 can expose, gathered from all `fw_cfg_add_file` call sites in hw/, are: `bootorder`, `bios-geometry`, `etc/acpi/tables`, `etc/acpi/rsdp`, `etc/table-loader`, `etc/tpm/log`, `etc/tpm/config`, `etc/smbios/smbios-anchor`, `etc/smbios/smbios-tables`, `etc/e820`, `etc/boot-fail-wait`, `etc/boot-menu-wait`, `etc/max-cpus`, `etc/reserved-memory-end`, `etc/msr_feature_control`, `etc/pvpanic-port`, `etc/system-states`, `etc/extra-pci-roots`, `etc/smi/supported-features`, `etc/smi/requested-features`, `etc/smi/features-ok`, `etc/igd-opregion`, `etc/igd-bdsm-size`, `etc/ramfb`, `etc/fdt`, `etc/memmap`, `etc/boot/kernel`, `etc/boot/shim`, `etc/hardware_errors`, `etc/hardware_errors_addr`, `etc/acpi_table_hest_addr`, `etc/hardware-info`, `etc/vmcoreinfo`, `etc/vmgenid_guid` and `etc/vmgenid_addr`, `etc/qemu-version`, `etc/firmware-min-version`, the hppa `/etc/hppa/*` and `/etc/cpu/*` files, `bootsplash.bmp` or `bootsplash.jpg`, `ndrv/qemu_vga.ndrv`, and every option ROM under `genroms/` plus user `-fw_cfg name=opt/...` files. The `opt/` prefix is reserved for users, and QEMU warns when a user file name does not start with `opt/`; we keep the warning text.

ruvm-hw-core provides `FwCfg` with the same add and modify API (`add_bytes`, `add_file`, `add_file_callback`, `modify_file`, `add_i16/i32/i64`, `add_string`), plus a `select_cb` and `write_cb` model for the few files that are generated lazily (ACPI tables are regenerated on select after hotplug via `acpi_build_update`, the SMI feature negotiation files use write callbacks). Contents are `Arc<[u8]>`, so snapshots are pointer copies.

## ACPI

### The linker and loader

QEMU does not place ACPI tables in guest memory itself. It builds them into fw_cfg blobs (`etc/acpi/tables`, `etc/acpi/rsdp`, `etc/tpm/log`, plus device-specific ones such as the vmgenid and GHES blobs) and a command script `etc/table-loader` (hw/acpi/bios-linker-loader.c) that firmware executes. The commands are ALLOCATE (0x1, allocate a file in a zone with alignment), ADD_POINTER (0x2, patch a pointer in one file with the address of another), ADD_CHECKSUM (0x3, compute a byte checksum over a range) and WRITE_POINTER (0x4, write the allocated address of a file back to QEMU through another fw_cfg file, used by vmgenid and GHES). SeaBIOS and OVMF both implement this interface. microvm, arm virt, riscv virt and loongarch virt use the same scheme with EDK2; direct boot on arm virt without firmware does not get ACPI.

ruvm implements the same linker and emits byte-identical scripts. For tests, ruvm-hw-acpi also contains a Rust implementation of the loader (`TableLoader::execute` with a fake allocator that assigns addresses the way SeaBIOS would) so tests can resolve tables without booting firmware.

### Tables

The x86 builder (hw/i386/acpi-build.c) emits RSDP, RSDT or XSDT, FADT (FACP), FACS, DSDT, MADT (APIC), HPET, SRAT, SLIT, HMAT, MCFG, DMAR or IVRS, TPM2 or TCPA, NFIT and its SSDT, VIOT, CEDT, ERST, WAET, WDAT, SLIC (when user supplied) and `-acpitable` user tables. arm virt (hw/arm/virt-acpi-build.c) emits FADT rev 6, DSDT, MADT with GICC, GICD, GIC ITS and GICR entries, GTDT, MCFG, SPCR, DBG2, IORT, PPTT, SRAT, SLIT, HMAT, TPM2, HEST when RAS is on, and the watchdog table. riscv virt emits FADT, DSDT, MADT with RINTC, IMSIC, APLIC and PLIC entries, RHCT, RIMT, MCFG, SPCR, SRAT and SLIT. loongarch virt has its own set in hw/loongarch/virt-acpi-build.c. The shared builders (`build_fadt`, `build_madt` pieces, `build_srat_memory`, `build_slit`, `build_pptt`, `build_spcr`, `build_mcfg`, `build_tpm2`, `build_hmat`) live in hw/acpi/aml-build.c and hw/acpi/*.c, and we keep the same split: ruvm-hw-acpi has the shared builders, each machine crate has its own table set.

The OEM ID and OEM table ID default to "BOCHS " and "BXPC    " (`ACPI_BUILD_APPNAME6` and `ACPI_BUILD_APPNAME8`) and are settable per machine with `x-oem-id` and `x-oem-table-id`. Creator ID and revision values are copied exactly because they are in the golden blobs.

### AML builder in Rust

QEMU builds AML with a C API of constructors in include/hw/acpi/aml-build.h (`aml_device`, `aml_name_decl`, `aml_method`, `aml_if`, `aml_store`, `aml_operation_region`, `aml_field`, `aml_resource_template`, `aml_interrupt`, and so on) that return `Aml *` nodes appended into a tree. The design is fine; what hurts in C is manual ownership and appending to the wrong parent.

ruvm-hw-acpi keeps the same vocabulary, deliberately, so that porting DSDT code is transliteration:

```rust
pub enum Aml {
    Bytes(SmallVec<[u8; 8]>),                 // opcodes and integer encodings
    Block { op: BlockOp, name: Option<NameSeg>, children: Vec<Aml> }, // Device, Scope, Method, If, Else, Package...
    ResourceTemplate(Vec<ResourceDesc>),
}

pub fn device(name: &str) -> Block;
pub fn name_decl(name: &str, value: impl Into<Aml>) -> Aml;
pub fn method(name: &str, args: u8, serialize: Serialize) -> Block;
pub fn if_(cond: Aml) -> Block;

let mut dev = aml::device("COM1");
dev.push(aml::name_decl("_HID", aml::eisaid("PNP0501")));
dev.push(aml::name_decl("_UID", 1u8));
dev.push(aml::name_decl("_CRS", aml::resource_template([
    aml::io(Decode::Decode16, 0x3f8, 0x3f8, 0x00, 0x8),
    aml::irq_no_flags(4),
])));
scope.push(dev);
```

Blocks own their children, so the wrong-parent bug cannot compile. PkgLength is computed bottom-up with the shortest of the 1 to 4 byte forms, as `build_prepend_package_length` does, and integers use the smallest of ZeroOp, OneOp, OnesOp and the Byte to QWord prefixes, as `aml_int` does, because a different valid encoding still breaks the golden test.

We evaluated the rust-vmm acpi_tables crate (https://github.com/rust-vmm/acpi_tables), which Cloud Hypervisor uses. Its encodings and layouts are its own, and our acceptance test is byte identity with QEMU. We use it as a reference and for fuzzing (we decode our output with an independent AML parser), not as a dependency. ruvm-hw-acpi is GPL-2.0-or-later because its table code is ported from QEMU.

### Testing against QEMU's expected blobs

tests/qtest/bios-tables-test.c boots each configuration under qtest with SeaBIOS or EDK2, finds the RSDP in guest memory, walks the tables, and compares each against tests/data/acpi/<arch>/<machine>/<SIG>[.variant]. When iasl is installed it disassembles both and prints an ASL diff.

ruvm runs that test binary unmodified against ruvm through the qtest protocol (document 22), which proves the full path including firmware. It also runs a faster in-process variant, `ruvm-hw-acpi/tests/golden.rs`, that builds each of the 188 configurations, executes the linker script with the Rust loader, and compares every table byte for byte against the same expected files, pinned from the v11.1.0 tag. That variant runs on every commit; the full qtest run is nightly. The rule is absolute for tier 1 machines: no ACPI difference against QEMU for the same machine version and command line. Hotplug paths (CPU hotplug `DSDT.cphp`, memory hotplug `DSDT.memhp`, PCI bridge hotplug `DSDT.bridge`, `DSDT.noacpihp`) have their own golden files and are part of the set.

Runtime ACPI (the parts of DSDT that call into QEMU through I/O ports: CPU hotplug registers at 0x0cd8 on q35 and 0xaf00 on pc, ACPI PCI hotplug at 0x0cc0 on q35 and 0xae00 on pc, memory hotplug at 0x0a00, and the Generic Event Device on microvm and arm virt) is specified with the devices in document 12, and those register interfaces are tested with qtest scripts that play the AML's sequence of port accesses.

## SMBIOS

hw/smbios/smbios.c builds types 0 (BIOS), 1 (system), 2 (baseboard), 3 (chassis), 4 (processor, one per socket), 8 (port connector, user supplied), 9 (system slots), 11 (OEM strings), 16 (physical memory array), 17 (memory device), 19 (memory array mapped address), 32 (system boot), 41 (onboard devices) and 127 (end of table). `-smbios type=N,field=value` overrides, `-smbios file=` adds raw tables. The entry point is either the 32-bit `_SM_` form or the 64-bit `_SM3_` form, selected with `-machine smbios-entry-point-type=32|64|auto`; pc-q35 and pc-i440fx 9.0 and newer default to `auto` (64-bit only when the tables do not fit the 32-bit entry point limits), 8.1 and 8.2 default to 64, and 8.0 and older to 32 (`default_smbios_ep_type` in the versioned options functions). QEMU emits them into `etc/smbios/smbios-tables` and `etc/smbios/smbios-anchor` for SeaBIOS and EDK2, and on arm virt and riscv virt the same files are consumed by EDK2.

No machine registered in 11.1 reaches `smbios_legacy_mode`, so ruvm does not implement it. Type 4 generation depends on topology (core count, thread count, `core_count2` above 255) and has three golden ACPI companion tests (`core-count`, `core-count2`, `thread-count2`); SMBIOS itself is checked by bios-tables-test, which locates the entry point in guest memory and walks the structures, and by our fingerprint.

## Device tree generation

Arm virt, riscv virt, loongarch virt, ppc e500, spapr (through SLOF or VOF), microblaze, openrisc and several others generate a device tree at runtime with libfdt calls (`qemu_fdt_add_subnode`, `qemu_fdt_setprop_cell` and friends in system/device_tree.c). Node order and phandle numbers are guest visible, and `-machine dumpdtb=file` lets users and tests compare.

ruvm-firmware provides an `Fdt` builder that produces the same structure block byte for byte as libfdt for the same sequence of calls: nodes in insertion order, properties in insertion order, the strings block deduplicated in first-use order the way `fdt_find_add_string_` does, and `fdt_pack` at the end. We do not use a sorted or hashed representation. Phandles are allocated from `phandle-start` (a machine property, default as in QEMU) in call order with the same `qemu_fdt_alloc_phandle` rule. `-dtb` (user blob) is merged as in QEMU: the user blob replaces the generated one, and the machine then patches `/chosen` (bootargs, initrd addresses, `kaslr-seed` and `rng-seed` unless `dtb-kaslr-seed=off` or `dtb-randomness=off`) and `/memory` nodes.

Golden testing: for each FDT machine and each representative command line, `-machine dumpdtb` output is compared against QEMU 11.1 output. Random seeds are disabled for these runs. spapr generates its tree again at CAS time (client architecture support negotiation, `h_client_architecture_support` in hw/ppc/spapr_hcall.c), so for spapr we also compare the post-CAS tree through qtest-driven hcalls.

## Direct kernel boot

`-kernel`, `-initrd`, `-append` and `-dtb` behave per architecture, and each path is ported from its QEMU file.

- x86 (hw/i386/x86-common.c `x86_load_linux`): for a bzImage with boot protocol 2.02 or newer the real-mode setup goes to 0x10000, the command line to 0x20000 and the protected mode kernel to 0x100000. Older protocols use 0x90000 and 0x9a000 minus the command line size. The initrd is placed below `initrd_addr_max` from the header (or 0x37ffffff for old kernels, or below 4 GiB minus ACPI data when the kernel sets XLF_CAN_BE_LOADED_ABOVE_4G) and the blobs are passed to the firmware through fw_cfg keys 0x07 to 0x18. SeaBIOS or OVMF then runs linuxboot_dma.bin (an option ROM that pulls them over fw_cfg DMA), and OVMF can instead use `etc/boot/kernel` and the shim files with its own loader. A `-dtb` on x86 is appended as a `setup_data` of type SETUP_DTB. Multiboot kernels go through multiboot_dma.bin (hw/i386/multiboot.c).
- x86 PVH: when the ELF contains a `XEN_ELFNOTE_PHYS32_ENTRY` note, `read_pvh_start_addr` records the 32-bit entry and QEMU loads the ELF directly, sets FW_CFG_KERNEL_ENTRY, and uses pvh.bin to jump there. microvm with qboot or bios-microvm and PVH is the fastest path to a running Linux, and it is the configuration the 15 ms and 110 ms targets in document 21 are measured on. With `ruvm run --direct-boot` (native CLI only), microvm under KVM skips the firmware and sets up the PVH start_info and registers directly, as Firecracker does. The QEMU-compatible CLI always runs the firmware.
- Arm (hw/arm/boot.c `arm_load_kernel`): AArch64 Image files are placed at RAM base plus `text_offset` from the header (2 MiB aligned for modern kernels), the DTB at the start of RAM, and a small bootloader stub from `arm_setup_direct_kernel_boot` sets x0 to the DTB address and jumps. 32-bit zImage uses KERNEL_LOAD_ADDR 0x10000 above loader start. For a Linux kernel without secure boot, the CPU is put in Non-secure state and the kernel starts at EL2 when the CPU has EL2 (`virtualization=on`), otherwise EL1 (the `boot_el` selection in `arm_load_kernel`); secondary CPUs are started via PSCI (`psci-conduit`). With EDK2, `-kernel` goes through fw_cfg to the firmware instead.
- RISC-V (hw/riscv/boot.c): OpenSBI at DRAM base, the kernel at the first 2 MiB (RV32: 4 MiB) aligned address after the firmware end, the DTB placed near the end of DRAM (clamped to 3 GiB on RV32) aligned down to 2 MiB and required to sit above the kernel and initrd, and `fw_dynamic_info` pointing OpenSBI at the kernel.
- ppc64 pseries: QEMU loads the kernel and initrd into guest RAM itself and tells SLOF where they are through `/chosen` properties in the device tree (`qemu,boot-kernel` and the initrd start and end); VOF boots the ELF directly. powernv: skiboot takes the kernel as a payload.
- s390x: `-kernel` loads the image at `KERN_IMAGE_START` 0x10000 with the command line at `KERN_PARM_AREA` 0x10480 (hw/s390x/ipl.c), or the s390-ccw BIOS boots from a device.
- LoongArch, MIPS malta, m68k virt, and the embedded boards each have their own small loaders, ported as-is.

ELF, uImage, raw and gzip-compressed images are recognized by the same probing order QEMU uses in hw/core/loader.c (`load_elf_ram_sym`, `load_uimage_as`, `load_image_gzipped_buffer`), and ruvm-firmware implements that loader as a plain Rust module. We do not use rust-vmm linux-loader (https://github.com/rust-vmm/linux-loader) because QEMU's placement rules differ from its defaults.

## IGVM

IGVM (Independent Guest Virtual Machine format, specified by Microsoft at https://github.com/microsoft/igvm) is a file format describing the initial guest state for confidential and non-confidential VMs: memory pages with their measurement type, VP context, parameter areas that the host fills (memory map, VP count, command line), and per-platform directives for SEV-SNP, TDX and VBS. QEMU supports it through `-object igvm-cfg,id=igvm0,file=path` and `-machine ...,igvm-cfg=igvm0` on pc, q35 and microvm (backends/igvm.c and backends/igvm-cfg.c, linked from hw/i386/pc.c and hw/i386/microvm.c); the firmware descriptor schema has an `igvm` mapping device and an `svsm` interface type because COCONUT-SVSM ships as IGVM.

ruvm implements IGVM loading in ruvm-firmware as a Rust parser of the IGVM header, platform table and directive stream, feeding a `ConfidentialLaunch` interface in ruvm-accel (document 19 owns the SEV-SNP and TDX launch flows). Pages go into guest memory through the same `rom_add_blob`-equivalent registry that other firmware uses so that reset re-applies them, except for confidential guests where reset follows the SEV-SNP and TDX reset support added in QEMU 11.0. The platform selected is the first one in the file compatible with the configured `confidential-guest-support` object, which is QEMU's rule. We use Microsoft's MIT-licensed `igvm` crate (https://crates.io/crates/igvm) for parsing, since it contains no QEMU-derived logic and a shared parser reduces disagreement with other loaders.

## Boot order and bootindex

Boot order has two mechanisms. The legacy `-boot order=cdn,once=d,menu=on,splash=,splash-time=,reboot-timeout=,strict=on` is passed to firmware via fw_cfg key 0x0c and `etc/boot-menu-wait`, and `qemu_boot_set` lets `once=` switch the order at the first reset (system/bootdevice.c `restore_boot_order`). The modern mechanism is the `bootindex` property on any bootable device, registered through `device_add_bootindex_property`. At machine done, `get_boot_devices_list` builds the `bootorder` fw_cfg file: one Open Firmware device path per line, sorted by bootindex, each path built by walking the qdev tree and asking each bus for `get_fw_dev_path` (for example `/pci@i0cf8/ide@1,1/drive@0/disk@0` or `/pci@i0cf8/ethernet@3/ethernet-phy@0`). SeaBIOS and OVMF match these strings against their own device discovery, so the path format is an ABI and we port every bus's `get_fw_dev_path` and `fw_name` verbatim. `bios-geometry` carries LCHS overrides in the same path format.

Duplicate bootindex values fail with "The bootindex %d has already been used" (`check_boot_index`). Bootindex can change at runtime via `qom-set` on the device, which updates the file for the next reset. s390x uses bootindex differently: the IPL device is the lowest bootindex, and `loadparm` per device selects an entry, so the s390 machine consumes the list directly rather than via fw_cfg. spapr passes it through the device tree `qemu,boot-list`, and the embedded boards largely ignore it.

ruvm stores bootindex in a `BootRegistry` in ruvm-hw-core keyed by device id and suffix, and produces the file with the same sort (stable by insertion order for devices without bootindex, which are excluded from the file). Tests: tests/qtest/boot-order-test.c and boot-serial-test.c run as-is, plus fingerprint comparison of the `bootorder` file.

## Crate placement

- ruvm-hw-core: `Machine` trait, `MachineClassInfo`, `CompatProp` and `CompatChain` types, `FwCfg`, `BootRegistry`, SMP parsing, NUMA state.
- ruvm-hw-acpi: AML builder, linker and loader, shared table builders, GED, runtime hotplug register blocks, the Rust table loader used in tests.
- ruvm-firmware: blob discovery and data dirs, firmware descriptor parser, FDT builder, SMBIOS builder, kernel loaders (ELF, uImage, bzImage, Image, PVH), IGVM loader.
- ruvm-machine-<family>: board code and that family's generated compat modules. The shared `hw_compat_*` arrays are generated into ruvm-hw-core.

## Decisions made in this document

1. The `Machine` trait and `MachineClassInfo` live in ruvm-hw-core (L2); machine implementations are in L4 crates.
2. Compat property arrays are generated from QEMU source by `cargo xtask compat-import`, checked in, and verified in CI against the pinned QEMU tag. Non-property compat knobs are hand-written typed fields.
3. A compat prop naming an existing type but a nonexistent property is a startup error in debug builds and a warning in release builds (QEMU only fails when the object is created).
4. Guest ABI stability is enforced by a machine fingerprint (QOM tree, address map, fw_cfg, ACPI, SMBIOS, FDT, RAMBlocks, VMState sections) compared against QEMU 11.1 and frozen per released machine version.
5. ruvm ships no firmware builds of its own and uses QEMU's pc-bios blobs and search path, because ROM sizes and RAMBlock names are migrated.
6. The AML builder mirrors QEMU's `aml_*` vocabulary with owned blocks; rust-vmm acpi_tables is a reference and fuzzing oracle, not a dependency.
7. ACPI golden tests run in two modes: QEMU's bios-tables-test unmodified over qtest (nightly) and an in-process Rust table loader against the same 188 expected blobs (every commit).
8. The FDT builder reproduces libfdt byte layout and phandle allocation; DTB output is compared against QEMU `dumpdtb`.
9. The native `ruvm run --direct-boot` may bypass firmware for microvm PVH under KVM; the QEMU-compatible CLI never does.
10. IGVM parsing uses Microsoft's MIT-licensed `igvm` crate.
