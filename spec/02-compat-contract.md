# 02. The compatibility contract

"100% compatible with QEMU" is meaningless until it says compatible with which QEMU, on which interface, to what precision, measured how. This document is the contract. It lists every surface on which ruvm promises QEMU's behavior, assigns each one a tier, names the QEMU file that is the source of truth, and says how the claim is tested. Everything else in the spec is judged against it: a design in documents 03 to 21 that cannot meet this contract is wrong, and a behavior not listed here is not promised.

## 1. The reference

The reference is QEMU 11.1.0 (tag `v11.1.0`, released 11 August 2026). Every "QEMU does X" in this document means "the v11.1.0 tree does X", read from source, not from documentation. When the documentation and the code disagree, the code wins, and the disagreement is filed upstream.

ruvm ships as one multi-call binary. The name it is invoked by selects a personality:

| Invoked as | Personality | Contract applies |
| --- | --- | --- |
| `qemu-system-<arch>` | QEMU system emulator for that target | Sections 3.1 to 3.6, 3.8 to 3.13 |
| `qemu-<arch>` | QEMU linux-user for that target | Sections 3.1, 3.7 to 3.10 |
| `qemu-img`, `qemu-io`, `qemu-nbd`, `qemu-storage-daemon` | QEMU tools | Sections 3.1, 3.2 (storage daemon), 3.5 |
| `qemu-ga` | Guest agent | Section 3.12 |
| `ruvm` | Native CLI | Not bound by this contract except where it drives the same core |

The `ruvm` personality is free to have its own options, defaults and extensions, as long as the VM it builds is one the qemu-system personality could also build (document 18). Everything below is about the `qemu-*` personalities.

Version reporting is part of the contract, because management software gates features on it. In `qemu-*` personalities, `query-version` returns `{"qemu": {"major": 11, "minor": 1, "micro": 0}, "package": " (ruvm X.Y.Z)"}` and `-version` prints `QEMU emulator version 11.1.0 (ruvm X.Y.Z)` followed by QEMU's copyright line. The numeric version is the reference QEMU version, not ruvm's own, because libvirt refuses anything below its minimum (7.2.0 in current libvirt, `QEMU_MIN_MAJOR`/`QEMU_MIN_MINOR` in `src/qemu/qemu_capabilities.c`) and enables features by version as well as by probing. The `package` field is exactly where QEMU puts distribution suffixes, so no parser breaks. (New decision.)

## 2. Tiers

Each item in the conformance matrix has one of three tiers. The tier is part of the contract, not an implementation status: an item at tier 1 that is not bit-exact is a bug.

### Tier 1: bit-exact

The observable bytes are identical to QEMU's for the same inputs. Examples: the output of `query-qmp-schema`; ACPI tables for a given machine type and command line; PCI configuration space reset values; a VMState section's field layout on the wire; qemu-img exit codes; the target description XML sent by the gdbstub; the list of trace event names. Tier 1 is measured by byte comparison against a QEMU 11.1.0 run of the same test, with only declared nondeterministic fields masked (timestamps, PIDs, randomly generated UUIDs, host paths). The mask list is part of each test and is reviewed like code.

### Tier 2: behaviorally equivalent

The observable result is the same, but the bytes on the way there may differ. Examples: a migration stream from ruvm is accepted by QEMU and produces the same guest state, even if pages are sent in a different order; `qemu-img convert` produces an image that `qemu-img compare` reports identical and `qemu-img check` reports clean, even if clusters were allocated in a different order; a QMP error has the same `class` but a differently worded `desc`. Tier 2 is measured by an oracle that defines "same": guest state checksums, compare and check results, or a structural diff with a documented equivalence relation.

### Tier 3: documented divergence

ruvm deliberately behaves differently, the difference is written down in `conformance/divergences.toml` with a reason, and the reason is one of four:

1. QEMU behavior is a crash, abort, assertion failure or memory corruption reachable from any input. Nobody can depend on a use-after-free. ruvm returns an error or ignores the access instead, choosing whatever QEMU's closest non-crashing path does (section 6.3).
2. QEMU behavior depends on host C library or compiler details (for example `%p` formatting in a message, or the order of `g_hash_table` iteration where the order leaks out).
3. The feature is inherently tied to QEMU's implementation and has no meaning in ruvm (section 8 lists these).
4. The feature is on QEMU's deprecation list and ruvm has not implemented it, which is allowed only for items QEMU already marks deprecated in 11.1 (see section 6.4).

A divergence entry names the item, the QEMU behavior, the ruvm behavior, the reason number, and the first ruvm release that carries it. The file is published with every release. A divergence that is not in the file is a bug.

## 3. The axes

Each axis below says what the surface is, where QEMU defines it, the default tier, the tier exceptions, and how conformance is measured. Counts are from the v11.1.0 tree and are the denominators of the matrix.

### 3.1 Command line

Surface: every option defined by `DEF(` in `qemu-options.hx` (115 entries at v11.1.0), with every sub-option each one accepts; the tool command lines defined in `qemu-img-cmds.hx`, `docs/tools/*.rst` and the corresponding `main()` functions; and environment variables read by each personality.

Both syntaxes are in scope. Legacy forms such as `-hda`, `-cdrom`, `-drive`, `-net nic -net user`, `-serial stdio`, `-m 4G`, `-smp 4`, and `-boot order=dc` must produce the same VM as they do in QEMU, including the implicit devices and default IDs they create (`ide0-hd0`, `net0`, `serial0` and so on). Modern forms `-blockdev`, `-netdev`, `-nic`, `-audiodev`, `-object`, `-device` with keyval or JSON syntax (`-device '{"driver":"virtio-blk-pci",...}'`), `-machine` keyval properties, and `-compat` must parse identically, including QEMU's keyval quirks (comma escaping with `,,`, implied keys such as `-device virtio-net-pci` meaning `driver=virtio-net-pci`, dotted keys for nested objects). `-set group.id.arg=value`, `-readconfig` with QEMU's INI-style config file syntax, `-no-user-config`, `-nodefaults` and `-S` are included.

`-mon` is deprecated since 11.1 in favor of `-object monitor-hmp` and `-object monitor-qmp`; ruvm implements both the deprecated and the new form, because a deprecated option is still a supported option until QEMU removes it.

Environment variables are part of the command line axis. For linux-user the set is fixed by `linux-user/main.c`: `QEMU_ARGV0`, `QEMU_CPU`, `QEMU_DFILTER`, `QEMU_GDB`, `QEMU_GUEST_BASE`, `QEMU_JITDUMP`, `QEMU_LD_PREFIX`, `QEMU_LOG`, `QEMU_LOG_FILENAME`, `QEMU_ONE_INSN_PER_TB`, `QEMU_PERFMAP`, `QEMU_PLUGIN`, `QEMU_RAND_SEED`, `QEMU_RESERVED_VA`, `QEMU_RTSIG_MAP`, `QEMU_SET_ENV`, `QEMU_STACK_SIZE`, `QEMU_STRACE`, `QEMU_TB_SIZE`, `QEMU_TRACE`, `QEMU_UNAME`, `QEMU_UNSET_ENV`, `QEMU_VERSION` and `QEMU_XTENSA_ABI_CALL0`. For the system emulator and tools the set is found mechanically by `cargo xtask upstream-diff` (section 7), which lists every `getenv` and `g_getenv` call in the tree.

Tier: 1 for acceptance (every command line QEMU accepts, ruvm accepts; every command line QEMU rejects, ruvm rejects with the same exit status). Tier 1 for `-help`, `-machine help`, `-cpu help`, `-device help`, `-device <driver>,help` and `-object <type>,help` output, because libvirt and virt-manager era tools and many scripts parse it. Tier 2 for error message text on rejection (the first line must name the same option and the same problem; exact wording is compared but a wording difference is a tier 2 finding, not a tier 1 failure).

Measurement: the "resulting VM" oracle. For each command line in the corpus, both QEMU and ruvm are started with `-S` and a QMP socket, and a fixed script dumps the full QOM composition tree (`qom-list` recursively with `qom-get` on every readable property), `query-block`, `query-named-block-nodes`, `query-chardev`, `query-netdev`-equivalent QOM data, `query-machines`, `query-cpus-fast`, and the fw_cfg file directory (read over qtest). The dumps must be identical after masking. The corpus is: every command line in QEMU's `tests/qtest` and `tests/functional`; every `.args` file in libvirt's `tests/qemuxmlconfdata`, which is libvirt's generated argv for thousands of domain XML variants; and command lines harvested from Proxmox VE's `qemu-server` test fixtures and OpenStack Nova's libvirt driver tests. Rejection tests come from the same sources plus a grammar fuzzer over `qemu-options.hx`.

### 3.2 QMP and QAPI

Surface: the QMP protocol (`docs/interop/qmp-spec.rst`), every command, event, type and feature in the QAPI schema (`qapi/*.json`, about 50 files at v11.1.0, generated by `scripts/qapi/`), and the `-compat` policy knobs.

Protocol, tier 1: the greeting `{"QMP": {"version": {...}, "capabilities": [...]}}` with the same capability list (`oob` where QEMU offers it); capabilities negotiation must come first via `qmp_capabilities`; the same response to commands before negotiation; `id` echoed verbatim with any JSON type; out-of-band execution with `exec-oob`; error responses of the form `{"error": {"class": ..., "desc": ...}, "id": ...}`. The error classes are exactly the five in `qapi/error.json`: `GenericError`, `CommandNotFound`, `DeviceNotActive`, `DeviceNotFound`, `KVMMissingCap`. ruvm never invents a class.

Schema, tier 1: `query-qmp-schema` output is byte-identical to QEMU's for the same target and the same enabled features. This is stronger than it sounds, because the QAPI generator (`scripts/qapi/introspect.py`) replaces most type names with numbered names (the documented rule is that type names are not part of the wire ABI) and the numbering depends on traversal order. ruvm's `ruvm-qapi` generator consumes the same `qapi/*.json` files and reproduces QEMU's traversal and numbering exactly; a Rust port of the generator is simpler than trying to match a hand-maintained schema. Schema conditionals (`'if': 'CONFIG_SPICE'`, target conditionals such as `TARGET_S390X`) are evaluated against the feature set of the ruvm build. The comparison reference is QEMU 11.1.0 configured with the same feature set, which `cargo xtask conform` derives from ruvm's Cargo features and passes to QEMU's `configure` (new decision: the reference QEMU for schema comparison is not a distribution build but one configured to match the ruvm build under test).

Commands and events: every command's argument handling and return value follows its schema, and semantic behavior follows QEMU's implementation. Tier 1 for return values of `query-*` commands on a VM built by the same command line (after masking). Tier 1 for event names and data members; timestamps are masked. Tier 1 for the order in which a command's side-effect events are emitted relative to its return (for example `DEVICE_DELETED` after `device_del` returns, `JOB_STATUS_CHANGE` sequences for block jobs), because management software is written against that order. Tier 2 for `desc` text, with one exception: a `desc` string that libvirt, Nova, Proxmox or `qemu.qmp` matches in its source is tier 1. `cargo xtask upstream-diff` keeps the list by grepping those projects for string literals that appear in QEMU `error_setg` calls.

Feature flags, tier 1: `deprecated` and `unstable` special features in the schema are reproduced, and `-compat deprecated-input=accept|reject|crash`, `deprecated-output=accept|hide`, `unstable-input=...` and `unstable-output=...` behave as in QEMU, including the documented limitation that they cover only syntactic aspects of QMP. Deprecated commands QEMU still ships (for example `query-kvm`, deprecated since 11.0 in favor of `query-accelerators`) are implemented.

Extensions: ruvm-only QMP commands and events use the downstream prefix `__io.github.tamnd.ruvm_` defined by the QAPI naming rules (document 25, Q2), so no future upstream name can collide. They are absent from the schema and rejected as `CommandNotFound` in `qemu-*` personalities, so the schema stays byte-identical. They exist only in the `ruvm` personality or when `RUVM_EXTENSIONS=1` is set in the environment (new decision).

Measurement: `tests/qtest/qmp-cmd-test.c`, which runs every query command on many machine types, is run against both implementations and the outputs diffed. `query-qmp-schema` is diffed for every target and every machine accelerator combination. A QMP fuzzer generates schema-valid and schema-invalid commands against both and compares response class and success. libvirt's `tests/qemumonitorjsondata` replies are checked as expected-output fixtures. The QAPI-level semantic suite is the union of iotests, functional tests and qtests that issue QMP.

### 3.3 HMP

Surface: 114 commands in `hmp-commands.hx` and 71 `info` subcommands in `hmp-commands-info.hx`, reachable through `-monitor`, `-object monitor-hmp`, and QMP `human-monitor-command`.

QEMU does not promise HMP stability, but people script it and libvirt passes some commands through `human-monitor-command`. Tier 1 for command names, argument syntax (the `args_type` mini-language in the `.hx` files) and acceptance. Tier 1 for the output of a fixed list of commands known to be scraped: `info version`, `info status`, `info name`, `info uuid`, `info kvm`, `info cpus`, `info block`, `info network`, `info chardev`, `info migrate`, `info snapshots`, `info registers`, `info mtree`, `info qtree` and `info pci`, plus `savevm`, `loadvm`, `delvm`, `drive_add` and `device_add` result text. Tier 2 for the rest (the same information, formatting may differ). Deprecated HMP commands in 11.1 (`wavcapture`, `stopcapture`, `info capture`) are implemented.

Measurement: every HMP command with every documented argument form is issued to both implementations on the reference machine set; outputs are diffed with the tier 1 list compared exactly.

### 3.4 Guest-visible hardware

This is the axis that matters most and costs most. A guest must not be able to tell ruvm from QEMU with the same machine type, the same command line and the same accelerator. The contract is per versioned machine type: `pc-q35-11.1` in ruvm matches `pc-q35-11.1` in QEMU, and `pc-q35-9.2` in ruvm matches `pc-q35-9.2` in QEMU 11.1.0 (not QEMU 9.2.0, because QEMU itself only promises that the 11.1 binary's 9.2 machine is compatible with 9.2's, and bugs in that promise are part of what we copy).

Machine versions supported are exactly those QEMU 11.1.0 defines. QEMU's versioned machine policy (deprecation after 3 years, removal after 6 years, in `docs/about/deprecated.rst`) is inherited: ruvm never keeps a machine version QEMU removed and never adds one QEMU does not have. A new versioned machine type appears in ruvm only once every compat property in its `hw_compat_*` and per-architecture compat arrays is implemented (new decision). At v11.1.0 the newest generic array is `hw_compat_11_0` in `hw/core/machine.c`, with ten entries including `arm-smmuv3` `ats`, `ril`, `ssidsize` and `oas` defaults, `chardev-vc` `encoding=cp437`, TPM CRB chunking and `migration` `switchover-ack-legacy`; the oldest is `hw_compat_4_1`. On x86, `pc_compat_11_0` and `pc_compat_10_2` in `hw/i386/pc.c` are empty and `pc_compat_10_1` sets `mch` `extended-tseg-mbytes=16` and the x86 CPU's `x-migrate-error-code=false`.

The surface, all tier 1:

- Register level. For every device, the reset value of every register, the result of every read after every legal write sequence, the handling of illegal accesses (access size, alignment, unimplemented offsets: QEMU usually returns 0 and logs `LOG_UNIMP` or `LOG_GUEST_ERROR`, sometimes raises a bus error through `MemTxResult`), and interrupt behavior.
- PCI. Vendor, device, subsystem IDs, revision, class code, BAR sizes and types, capability list order and contents, bus and slot placement of default and implicit devices, MSI and MSI-X vector counts, config space write masks.
- ACPI. Every table in `etc/acpi/tables`, byte for byte, including OEM IDs, table revisions, AML byte code and checksums, for each machine type and each command line in the ACPI corpus.
- fw_cfg. The file directory (names, sizes, selector numbers), the contents of every file (`etc/table-loader`, `etc/acpi/rsdp`, `bootorder`, `etc/e820`, `etc/smbios/smbios-tables`, `genroms/*`, `opt/*` supplied by the user), the DMA interface and the legacy selector interface.
- SMBIOS. Every structure, including the strings QEMU derives from the machine (the machine type name such as `pc-q35-11.1` in the type 1 version field and the machine description in the product field).
- CPU identity. CPUID leaves and values for every x86 CPU model and every version (`Skylake-Server-v4` and so on) including feature words, cache descriptors, topology leaves and the hypervisor leaves (KVM, Hyper-V enlightenments); Arm ID registers and `-cpu max` feature sets for TCG and KVM; RISC-V `misa` and extension strings; the same for other targets. With KVM and `-cpu host`, the contract is "what QEMU would pass to KVM on this host", tested on the same host.
- Device trees. For machines that generate one (`virt` on Arm, RISC-V and LoongArch, `ppce500`, `spapr`, and others), the DTB QEMU would produce, compared after `fdtdump` normalization. Both expose it through the `dumpdtb` machine property.
- Timing-visible behavior with icount. With `-icount shift=N` and a single vCPU under TCG, the guest-visible clock values after N instructions match QEMU's. Without icount, timing is not part of the contract (section 8).
- Reentrancy. When a device DMA reaches its own MMIO, QEMU's guard in `system/memory.c` returns `MEMTX_ACCESS_ERROR` and logs "Blocked re-entrant IO on MemoryRegion". ruvm does the same wherever QEMU does. Where QEMU has no guard (timer and ioeventfd paths, as in CVE-2026-17588) and the result is a use-after-free, ruvm returns the same `MEMTX_ACCESS_ERROR` result (tier 3, reason 1; see document 01, section 2.3).

Measurement: several oracles, all run by `cargo xtask conform hw`.

- ACPI: QEMU's `tests/qtest/bios-tables-test.c` with its expected blobs in `tests/data/acpi/{x86,aarch64,riscv64,loongarch64}`, run against ruvm unmodified. Plus a larger corpus: for every machine type and a matrix of `-smp`, `-m`, NUMA, memory hotplug, CXL, IOMMU and device options, dump tables from both and diff; `iasl` disassembly is attached to failures.
- Registers: a register walker over the qtest protocol reads and writes every register of every device instantiated by the corpus command lines and diffs traces. The Morphuzz-style differential fuzzer (document 01, section 8, and document 22) drives both implementations with the same input and compares every MMIO and PIO read result and every DMA write.
- PCI, fw_cfg, SMBIOS, CPUID and device trees: dumped over qtest and by a small guest-side probe (a Linux initramfs that prints `lspci -xxxx`, `/sys/firmware/acpi/tables`, `dmidecode --dump-bin`, `cpuid -r` and `/sys/firmware/fdt`) under both, then diffed.
- Firmware boot: SeaBIOS, OVMF and the Arm, RISC-V, PowerPC and s390x firmware shipped in `pc-bios/` boot the same guests to the same serial console output.

### 3.5 Disk formats and qemu-img

Surface: the on-disk formats QEMU reads and writes (raw, qcow2 including external data files, compression types, extended L2 and bitmaps, qcow, vmdk, vdi, vhdx, vpc, qed, parallels, dmg read-only, cloop, bochs, luks), the protocol drivers (file, host_device, nbd, iscsi, ssh, http and https through curl, nfs, blkio), the block export formats (NBD server, vhost-user-blk, FUSE, VDUSE), and the output and exit codes of `qemu-img`, `qemu-io`, `qemu-nbd` and `qemu-storage-daemon`.

Tiers:

- Images written by ruvm open in QEMU, and images written by QEMU open in ruvm, for every format QEMU can write: tier 1 for acceptance.
- `qemu-img create` with the same options produces a byte-identical image header and metadata (tier 1), because tests and tools compare created images directly.
- Images modified by guest writes or by `convert`, `commit`, `rebase`, `amend` and block jobs are tier 2: `qemu-img compare` reports identical content (exit 0) and `qemu-img check` reports a consistent image (exit 0) under both implementations. Cluster allocation order may differ.
- `qemu-img info`, `check`, `map`, `measure`, `snapshot -l` and `bitmap` output in both `--output=human` and `--output=json` are tier 1, with host paths and timestamps masked.
- Exit codes are tier 1. For `qemu-img check`: 0 consistent, 1 check not completed because of an internal error, 2 corruptions found, 3 leaked clusters found, 63 checks not supported by the format; with `-r` the code reflects the state after repair. For `qemu-img compare`: 0 identical, 1 different, 2 error opening an image, 3 error checking sector allocation, 4 error reading data. These come from `docs/tools/qemu-img.rst` and are asserted against `qemu-img.c`.
- NBD protocol behavior, including structured replies, block status contexts (`base:allocation`, `qemu:dirty-bitmap:*`, `qemu:allocation-depth`) and the export names, is tier 1 on the wire as seen by a client; the server's choice of reply chunking is tier 2.

Measurement: QEMU's iotests (`tests/qemu-iotests`, with 112 named tests under `tests/qemu-iotests/tests` at v11.1.0 plus the numbered tests) run against ruvm's `qemu-img`, `qemu-io`, `qemu-nbd` and system emulator with QEMU's reference output files unmodified. Iotest reference outputs are exact text; they are the single largest tier 1 oracle in this contract. Cross-implementation images: a generator creates images with QEMU and ruvm in every format and option combination, and each implementation reads the other's.

### 3.6 Migration stream

Surface: the migration stream format (`migration/savevm.c`: the `QEVM` magic `QEMU_VM_FILE_MAGIC`, `QEMU_VM_FILE_VERSION`, section types `QEMU_VM_SECTION_START`, `PART`, `END`, `FULL`, `QEMU_VM_SUBSECTION`, `QEMU_VM_VMDESCRIPTION`, `QEMU_VM_CONFIGURATION`, `QEMU_VM_COMMAND`, `QEMU_VM_EOF` and section footers), every `VMStateDescription` for every device, CPU and machine (`migration/vmstate.c`: `vmstate_save_state`, `vmstate_load_state`, `vmstate_subsection_load`), RAM transfer (precopy, postcopy, multifd, `mapped-ram`, XBZRLE, zero page handling, dirty rate), the return path, migration capabilities and parameters, `savevm`/`loadvm` internal snapshots in qcow2, and `migrate` to `file:`, `fd:`, `exec:`, `tcp:`, `unix:` and `rdma:` URIs (where the host supports RDMA).

The promise is interoperability in both directions for every machine type both implementations share: QEMU 11.1.0 to ruvm and ruvm to QEMU 11.1.0, live and to file, precopy and postcopy, with and without multifd. Older QEMU sources are covered to the extent QEMU 11.1.0 covers them: a stream from QEMU 10.0 with `pc-q35-10.0` that QEMU 11.1.0 accepts, ruvm accepts.

Tiers:

- VMState section layout: tier 1. For every device and machine type, the fields, their order, sizes, versions, subsections and the conditions under which subsections are sent (`needed` functions) are identical, so a stream section produced by ruvm is bytewise what QEMU would produce from the same device state. `-dump-vmstate` output is tier 1, and the check is QEMU's own `scripts/vmstate-static-checker.py` run on both dumps in both directions.
- Stream as a whole: tier 2. RAM page order, multifd channel assignment, and iteration counts may differ. The oracle is "the destination accepts the stream and the guest state after migration is identical", checked by comparing a `-dump-vmstate`-driven state hash and guest memory checksums, and by running the guest workload to completion.
- Capability and parameter negotiation, `query-migrate` fields and the order of `MIGRATION` status events: tier 1.
- `mapped-ram` files: tier 1 for layout, since the whole point of the format is fixed offsets.

Measurement: a migration matrix run nightly: every machine type times a device set of the most common libvirt and OpenStack configurations times {QEMU to ruvm, ruvm to QEMU, ruvm to ruvm} times {precopy, postcopy, multifd, file, mapped-ram}, with a guest running a memory-dirtying and I/O workload (document 22). QEMU's `tests/qtest/migration/` suite runs with ruvm as source, destination, or both. Per-device state equivalence is checked by the static checker plus a dynamic check that loads each section into both implementations and dumps it back.

### 3.7 User mode

Surface: for each linux-user target, the syscall ABI (numbers, argument conversion, struct layouts, errno values, signal numbers and frames, `sigreturn` behavior, `AT_*` auxv entries), the initial process state (stack layout, `AT_HWCAP` and `AT_HWCAP2`, `AT_PLATFORM`, reserved VA and guest base handling), `-strace` output, the emulated `/proc` files, `binfmt_misc` integration and the environment variables in section 3.1. BSD user mode is in scope for the targets QEMU 11.1.0 supports on BSD hosts.

The emulated `/proc` files are those QEMU fakes in `maybe_do_fake_open()` in `linux-user/syscall.c`: for the process itself `maps`, `smaps`, `stat`, `auxv` and `cmdline`, and globally `/proc/net/route`, `/proc/cpuinfo` and `/proc/hardware` where the target defines them, plus the special handling of `/proc/self/exe`. Tier 1 for their contents given the same guest process state.

`binfmt_misc`: `scripts/qemu-binfmt-conf.sh` registers `:qemu-<cpu>:M::<magic>:<mask>:<interpreter>:<flags>` with flags from `P` (preserve argv0), `O` and `C` (credentials) and `F` (persistent). ruvm ships the same magic and mask table and the same script behavior, tier 1, so distribution packaging can switch interpreters by changing the path only.

Tiers: tier 1 for syscall results and guest-visible memory effects; tier 1 for `-strace` line format (people diff strace logs); tier 2 for guest address space layout of mappings the guest did not request at fixed addresses (QEMU's layout depends on host mmap results; ruvm follows the same algorithm in `linux-user/mmap.c` but host differences are allowed).

Measurement: QEMU's `tests/tcg` suites for each target, the Linux Test Project syscall tests run under both, and a differential harness that runs the same static binaries (busybox, coreutils test suites, Go and Rust toolchains, Python test suite) under both and compares stdout, exit status and `-strace` logs.

### 3.8 TCG plugin ABI

Surface: `include/plugins/qemu-plugin.h` at v11.1.0, `QEMU_PLUGIN_VERSION` 7, the exported `qemu_plugin_version` and `qemu_plugin_install` symbols that a plugin must provide, `-plugin file=...,arg=...` and `QEMU_PLUGIN`. Version 7 passes userdata to all callbacks; version 6 added `qemu_plugin_set_pc`, the discontinuity callback and the syscall filter; version 5 added memory read and write by virtual and physical address, `qemu_plugin_write_register` and `qemu_plugin_translate_vaddr`.

Tier 1: a plugin compiled against QEMU 11.1.0's header loads into ruvm without recompilation and sees the same callbacks with the same arguments in the same order for the same guest execution. That includes the translation-time view: the instructions in each `qemu_plugin_tb` handed to the translation callback must be the same as QEMU's, which means that when any plugin is loaded, ruvm-jit tier 1 forms blocks using QEMU's rules (maximum instruction count per TB, page-crossing rules, the instructions that end a TB per target, `-one-insn-per-tb`). Tier 2 of ruvm-jit may still optimize a region, but it must deliver plugin callbacks as if the tier 1 blocks were executed one by one, and it skips any region containing an instrumented instruction whose callbacks it cannot preserve (new decision; document 08 gives the mechanism).

Measurement: every plugin in `tests/tcg/plugins` (such as `insn.c`) and `contrib/plugins` is built against the QEMU header and run on the same guest programs under both, in user mode and in system mode with `-icount` for determinism; outputs are diffed. Multi-vCPU runs compare per-vCPU aggregates rather than interleavings.

### 3.9 gdbstub

Surface: the GDB remote serial protocol as implemented in `gdbstub/`, `-gdb dev`, `-s`, `QEMU_GDB` and `-g` in user mode, the target description XML QEMU sends via `qXfer:features:read` (generated from `gdb-xml/*.xml` plus dynamic system register descriptions on Arm and others), register numbering, `vCont` behavior, thread IDs for vCPUs, `qOffsets`, `qXfer:auxv:read` and `qXfer:exec-file:read` in user mode, and the QEMU-specific packets `qqemu.Supported`, `Qqemu.PhyMemMode`, `qqemu.sstep`, `Qqemu.sstep` and `qqemu.sstepbits`.

Tier 1 for the target description XML (byte comparison), the packet set advertised in `qSupported`, register numbering and the responses to every packet. Tier 2 for the timing of stop replies relative to other vCPUs in all-stop mode.

Measurement: QEMU's `tests/guest-debug` scripts and `tests/tcg/*/gdbstub` tests under both; a packet-level replay tool that records a GDB session against QEMU and replays it against ruvm, diffing responses.

### 3.10 Trace events

Surface: every event declared in the `trace-events` files across the tree, `-trace enable=pattern,events=file,file=file`, the QMP commands `trace-event-get-state` and `trace-event-set-state`, the HMP `trace-event` and `info trace-events` commands, and the trace backends selectable at configure time (`nop`, `log`, `simple`, `ftrace`, `syslog`, `dtrace` for SystemTap and DTrace, `ust` for LTTng).

Tier 1 for the set of event names, their argument names and types, and which events exist per target. Tier 1 for the `log` backend line format (it is a debug printf format string per event, and people grep it). Tier 1 for the `simple` backend binary format, because `scripts/simpletrace.py` parses it. Tier 2 for when an event fires relative to others: ruvm fires each event at the point in the operation where QEMU does, but ruvm's internal structure may add events of its own (under a `ruvm_` prefix, excluded in `qemu-*` personalities unless enabled explicitly) and may not be able to fire a QEMU event whose trigger is an implementation artifact (for example TCG internals). Such events exist, are listed, and never fire; each is a tier 3 entry with reason 3.

Measurement: name and signature lists diffed by `cargo xtask upstream-diff`; the functional test suite run with `-trace` for a set of device events and the logs diffed after masking timestamps and PIDs.

### 3.11 Firmware interfaces

Surface: the fw_cfg device ABI (`docs/specs/fw_cfg.rst`, selectors, DMA interface) as consumed by SeaBIOS, OVMF, and Linux; the ACPI table loader protocol (`etc/table-loader` commands `ALLOCATE`, `ADD_POINTER`, `ADD_CHECKSUM`, `WRITE_POINTER`); `-bios`, `-pflash`, `-drive if=pflash`, `-kernel`, `-initrd`, `-append`, `-dtb` loading rules and the resulting memory layout; the firmware descriptor files (`docs/interop/firmware.json`) that libvirt uses to pick firmware; the vhost-user and vhost-user-blk backend descriptors (`docs/interop/vhost-user.json`); the firmware binaries shipped in `pc-bios/`; and the Linux boot protocol details QEMU implements (x86 setup header handling in `hw/i386/x86-common.c`, Arm boot in `hw/arm/boot.c`, the RISC-V boot in `hw/riscv/boot.c`).

Tier 1 for everything here. ruvm ships the same firmware binaries QEMU 11.1.0 ships, built from the same submodule commits (`roms/`), with ruvm-specific firmware builds only where QEMU's blob has to change and never under the same file name (new decision; document 11 details the build). The firmware descriptor JSON files ruvm installs are those QEMU's distributions install, pointing at the same binaries.

Measurement: covered by the hardware oracles in section 3.4 plus a boot matrix of distributions and firmware (SeaBIOS legacy, OVMF with and without Secure Boot and SMM, AAVMF, OpenSBI with U-Boot, SLOF, s390-ccw) comparing serial console transcripts up to the first line whose content depends on timing.

### 3.12 Guest agent

Surface: `qemu-ga` and its protocol (`qga/qapi-schema.json`, 44 commands at v11.1.0), transport over virtio-serial (port `org.qemu.guest_agent.0`), isa-serial, and vsock, the `guest-sync` and `guest-sync-delimited` handshakes (the latter prefixes the response with the 0xFF sentinel byte), the command allow and block lists, fsfreeze hooks, and the configuration file.

Tier 1 for the protocol and command results on the same guest. ruvm's `qemu-ga` personality is a Rust implementation that runs inside the guest, so it is also tested on Windows guests where QEMU's agent uses VSS. The host side (the virtio-serial device and chardev) is covered by section 3.4.

Measurement: libvirt's `tests/qemuagentdata` fixtures and a guest image matrix running both agents and diffing JSON replies with masking of timestamps and guest-dependent values.

### 3.13 libvirt capability probing

libvirt is the largest consumer of QEMU and its probe is the most concentrated compatibility test there is. At startup and whenever a binary changes, libvirt runs `qemu-system-<arch> -S -no-user-config -nodefaults -nographic -machine none,accel=<accel> -qmp <socket> -pidfile <file> -daemonize` (from `qemuProcessQMPLaunch` in `src/qemu/qemu_process.c`) and issues a fixed sequence of QMP queries. The recorded x86_64 probe for 11.1.0 contains `query-version`, `query-target`, `query-qmp-schema`, `query-accelerators`, `qom-list-types`, `query-command-line-options`, `query-machines` and `query-cpu-definitions` twice (once per accelerator), 7 `query-cpu-model-expansion` calls, 5 `qom-list-properties` calls, 32 `device-list-properties` calls, `query-sev-capabilities` and `query-sgx-capabilities`. libvirt reads most of the command, event and type surface from `query-qmp-schema` rather than probing each one. The replies decide which features libvirt will use.

libvirt keeps recorded probe conversations in `tests/qemucapabilitiesdata/caps_<version>_<arch>.replies` with the resulting capabilities in `caps_<version>_<arch>.xml`. Files for `caps_11.1.0_x86_64`, `caps_11.1.0_aarch64` and `caps_11.1.0_s390x` exist in libvirt master as of September 2026, alongside variant files such as `caps_10.1.0_x86_64+inteltdx` and `caps_10.2.0_x86_64+mshv`.

Tier 1: replaying the command sequence of each `caps_11.1.0_<arch>.replies` file against ruvm yields the recorded replies after masking host-dependent values (CPU model expansion under KVM on a different host is compared on the same host instead). libvirt's `qemucapabilitiestest` and `qemucaps2xmltest` then produce the same `.xml` file from ruvm's replies, which is the real acceptance criterion: libvirt believes ruvm is QEMU 11.1.0 with exactly the same features.

Beyond probing, libvirt certification (milestone M11, document 23) runs libvirt's TCK and the `virt-install` and `virsh` scenarios used by Fedora and Debian QA against ruvm installed as the QEMU binary.

## 4. The conformance matrix

The matrix is a directory of TOML files in the ruvm repository under `conformance/`, one file per axis, one table per item. It is the single source of truth for what is claimed and what is tested. The rendered HTML is published per release.

```toml
[[item]]
axis = "qmp"
id = "qmp.command.query-block"
tier = 1
status = "pass"            # pass | fail | partial | not-started | waived
tests = ["qmp-cmd-test/query-block", "iotests/030", "conform/qmp/query-block-dump"]
qemu_ref = "v11.1.0"       # tag or commit the expectation was taken from
since = "0.4.0"            # first ruvm release where status became pass
notes = ""

[[item]]
axis = "hw"
id = "hw.pc-q35-11.1.acpi.DSDT"
tier = 1
status = "pass"
tests = ["bios-tables-test/q35/DSDT", "conform/acpi/q35-matrix"]
qemu_ref = "v11.1.0"
since = "0.6.0"
notes = "includes -smp 1..288 and NUMA variants"
```

Item IDs are generated from QEMU sources where possible so that coverage is checkable: one item per `DEF(` in `qemu-options.hx`, per QAPI command and event, per HMP command, per device type (`TYPE_*` registered with `type_register_static` or `DEFINE_TYPES`), per versioned machine type, per `VMStateDescription`, per qemu-img subcommand, per linux-user target, per plugin API function, per gdbstub packet, per trace event file and per qga command. `cargo xtask conform coverage` fails the build if a generated ID has no item. An item with status `waived` must have a matching tier 3 entry in `divergences.toml`.

A test is only allowed to back a tier 1 item if it compares against a QEMU 11.1.0 run or a checked-in expected output generated by QEMU 11.1.0. Tests that only check ruvm against itself back no conformance claim.

The QEMU reference runs use a pinned container image with QEMU 11.1.0 built from the tag by `cargo xtask qemu-ref build`, with the configure options derived from the ruvm features under test (section 3.2). Reference outputs are cached by (test id, QEMU commit, configure hash) so the nightly run does not rebuild QEMU.

## 5. What counts as a bug

A compatibility bug is any of:

1. A tier 1 item whose output differs from QEMU's under the test's mask.
2. A tier 2 item whose oracle fails.
3. A behavior difference on any listed surface that is not in `divergences.toml`, whether or not a test caught it. A user report with a reproducer counts.
4. A surface that QEMU 11.1.0 exposes and ruvm does not, once the release claims the axis. Before an axis is claimed (the matrix shows `not-started`), missing items are gaps, not bugs.
5. A crash, hang, or memory growth in ruvm on any input QEMU handles without crashing.
6. A performance regression beyond the thresholds in document 21. This is not a compatibility bug but is tracked with the same severity.

Compatibility bugs in tier 1 items on the reference machine types (`pc-q35-*`, `pc-i440fx-*`, `microvm`, Arm and RISC-V `virt`, `pseries`, `s390-ccw-virtio`) block a release. Others are triaged by the matrix owner of the axis.

## 6. QEMU bugs

ruvm will find QEMU bugs, because two implementations diffing each other is the best bug finder there is. The policy:

### 6.1 Bug for bug by default

If QEMU 11.1.0 has a guest-visible or management-visible bug, ruvm reproduces it. A guest driver may depend on it, a migration stream certainly encodes it, and libvirt may work around it in a way that breaks if it is fixed. The bug gets an entry in `conformance/qemu-bugs.toml` with a link to the upstream report (ruvm files one if none exists) and the ruvm code that reproduces it carries a comment naming the entry.

### 6.2 Fixes behind machine versions

When QEMU fixes a guest-visible bug, it normally keeps the old behavior for old machine types with a compat property in the `hw_compat_*` or target-specific compat array. ruvm follows the same mechanism: the fix is a device property whose default is the fixed behavior, and the compat entry for older machine versions sets the old one. ruvm adopts the fix when QEMU merges it and releases the matching machine version, not earlier, because shipping a fixed `pc-q35-11.1` before QEMU does would break migration to QEMU's `pc-q35-11.1`.

### 6.3 Exceptions

Four kinds of QEMU bug are not reproduced:

- Memory safety bugs and crashes (tier 3 reason 1). ruvm takes the nearest non-crashing behavior. If QEMU's upstream fix picks a different behavior later, ruvm switches to it.
- Security bugs with a CVE. ruvm ships the fix as soon as it is public, unconditionally, even if QEMU's fix is not yet released, unless the fix changes the migration stream; in that case ruvm follows QEMU's compat property approach and matches QEMU's released fix.
- Bugs that are not observable on any listed surface (internal inefficiencies, lock ordering, leaks).
- Bugs QEMU has fixed on master for a release that ruvm is tracking (section 7). The reference moves; the old bug goes with it.

### 6.4 Deprecated and removed features

A feature deprecated in QEMU 11.1.0 is still part of the contract, because QEMU still supports it; the matrix marks it and the deprecation warning text is tier 1. When QEMU removes it, ruvm removes it in the release that tracks that QEMU version. ruvm may choose not to implement a deprecated feature before its first claimed release of an axis only with a tier 3 reason 4 entry; this is intended for things like obsolete audio and display backends, not for anything libvirt still generates.

## 7. Tracking upstream

ruvm tracks QEMU releases, not QEMU master. Each ruvm release series declares one reference QEMU version. A new reference is adopted within about 60 days of a QEMU release (new decision): QEMU releases three times a year, so the reference is never more than one release behind for longer than two months.

The input to each tracking cycle is `cargo xtask upstream-diff <old-tag> <new-tag>`, which reads the two QEMU trees and writes a report with one section per surface:

| Surface | Source in QEMU | What the diff produces |
| --- | --- | --- |
| QMP/QAPI | `qapi/*.json`, `qga/qapi-schema.json` | Added, removed and changed commands, events, types, members, features, conditionals; regenerated `query-qmp-schema` diff per target |
| Command line | `qemu-options.hx`, `qemu-img-cmds.hx`, `getenv` calls | Added, removed and changed options and sub-options, help text changes |
| HMP | `hmp-commands.hx`, `hmp-commands-info.hx` | Command and argument changes |
| Machine compat | `hw/core/machine.c` `hw_compat_*`, `hw/i386/pc.c` `pc_compat_*`, `hw/arm/virt.c`, `hw/s390x/s390-virtio-ccw.c`, `hw/ppc/spapr.c` machine version functions | New machine versions and every new compat property with its old value |
| Devices | `type_register_static`, `DEFINE_TYPES`, `device_class_set_props` and `DEFINE_PROP_*` | New device types and property changes |
| Migration | `-dump-vmstate` output of both versions for every target and machine | `vmstate-static-checker.py` report in both directions |
| Trace events | `trace-events` files | Added, removed and changed events |
| Plugins | `include/plugins/qemu-plugin.h` | API additions, `QEMU_PLUGIN_VERSION` change |
| ACPI | `tests/data/acpi/` | Expected blob changes, each with the commit that changed it |
| linux-user | `linux-user/syscall.c`, `*/syscall_nr.h`, `syscall.tbl` imports | New syscalls, changed fake `/proc` entries |
| Deprecations | `docs/about/deprecated.rst`, `removed-features.rst` | New deprecations and removals |

Each line of the report becomes a matrix item or an item change, assigned to the owner of the axis. The cycle is done when every item is `pass` or has a divergence entry, the migration matrix passes against the new QEMU release in both directions, and the libvirt probe replay for the new version (once libvirt has recorded it) passes. Until then the previous ruvm release series remains the supported one.

Between releases, a weekly job runs `upstream-diff` from the current reference tag to QEMU master. It does not change the contract; it gives early warning of large incoming changes and feeds the milestone plan in document 23.

Security fixes are the exception to release tracking: CVE fixes in QEMU's stable branches (`stable-11.1`) are reviewed within a week and ported when ruvm shares the affected behavior (section 6.3).

## 8. Explicitly outside the contract

These are not promised and are not bugs if they differ:

- Performance and timing without icount: wall-clock speed, latency, interrupt timing, scheduling of vCPU threads, and anything a guest can only observe through timing. Performance has its own targets in document 21.
- The record/replay log file format written by `-icount rr=record,rrfile=...`. The CLI and QMP for record/replay are in the contract; the file is ruvm's own (new decision, see document 01, section 7). QEMU does not accept replay logs across versions either.
- `-d` debug log output (`-d in_asm,op,out_asm,exec,cpu` and so on) beyond the categories and their names. The contents of `op` and `out_asm` describe QEMU's TCG internals, which ruvm does not have. `-d cpu` and `in_asm` output is tier 2 (same information, formatting may differ); `-d guest_errors` and `unimp` messages are tier 2.
- `-accel tcg` tuning options that describe TCG internals (`tb-size`, `split-wx`, `one-insn-per-tb` is the exception and is honored exactly, see section 3.8) are accepted and validated as QEMU does, but their effect is ruvm-jit's nearest equivalent. `thread=single|multi` is accepted; ruvm always runs MTTCG-style and `single` is emulated by serializing vCPUs.
- QEMU's loadable module ABI (`--enable-modules` `.so` files). ruvm does not load QEMU modules; its own extension mechanism is in document 20. The TCG plugin ABI (section 3.8) is the only binary ABI ruvm implements.
- Internal QOM paths of objects QEMU creates for implementation reasons and hides from users (for example `/machine/unattached/device[N]` numbering) are tier 2: they exist with the same types, and ruvm matches numbering where it can, but management software must not rely on the index.
- Host resource usage: file descriptors, threads, memory layout of the process, names of threads.

## 9. Where each surface is designed

| Axis | Contract section | Design documents |
| --- | --- | --- |
| Command line and config | 3.1 | 04, 18 |
| QMP and QAPI | 3.2 | 04, 18 |
| HMP | 3.3 | 18 |
| Guest-visible hardware | 3.4 | 05, 09, 11, 12, 13, 15, 16 |
| Disk formats and tools | 3.5 | 14 |
| Migration | 3.6 | 17 |
| User mode | 3.7 | 10 |
| Plugin ABI | 3.8 | 07, 08, 20 |
| gdbstub | 3.9 | 18 |
| Trace events | 3.10 | 18, 21 |
| Firmware interfaces | 3.11 | 11 |
| Guest agent | 3.12 | 18 |
| libvirt probing | 3.13 | 18, 23 |
| Matrix and harness | 4 | 22 |
| Upstream tracking | 7 | 23, 24 |
