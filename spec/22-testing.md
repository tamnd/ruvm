# 22. Testing: proving compatibility and correctness

ruvm's promise is that a user can replace `qemu-system-x86_64` with a symlink to `ruvm` and nothing changes except speed and memory use. That promise is only as good as the evidence behind it. This document defines where that evidence comes from: QEMU's own test suites run unmodified against ruvm, differential testing against a real QEMU 11.1 binary, ISA conformance suites and formal models as oracles for the JIT, a guest OS boot matrix, management stack suites from libvirt and Avocado-vt, fuzzing, deterministic concurrency testing, and the CI infrastructure and dashboard that tie them together. Performance testing is in document 21; the compatibility contract that these tests enforce is in document 02.

## Principles

The first principle is that QEMU's tests are the specification wherever they exist, since QEMU has no formal specification of its guest-visible behavior, QMP error strings, or image format corner cases. We run them as written and treat a failure as a ruvm bug unless the test depends on a QEMU internal outside the compatibility surface; every such exclusion is written down with a reason.

The second principle is that when QEMU's tests are silent, QEMU itself is the oracle. Differential testing compares ruvm to a real QEMU 11.1.0 binary on the same inputs and treats any observable difference as a bug in ruvm unless document 02 lists it as an intentional divergence. Where QEMU is known to be wrong (a published bug, a Risotto style fence error, a softfloat flag bug fixed upstream after 11.1), the difference is recorded in a divergence registry with a link to the upstream report, and ruvm matches the architecture specification rather than the bug, unless matching the bug is required for migration compatibility.

The third principle is that tests pin versions. The upstream tree is a git submodule at `tests/upstream/qemu` pinned to `v11.1.0`; moving the reference to a newer release is one pull request that also updates the skip lists.

## Test tiers

| Tier | Where it runs | Budget | Contents |
|---|---|---|---|
| T0 unit | every push, hosted runners (Linux x86-64, Linux aarch64, macOS arm64, Windows x86-64) | 15 min | `cargo nextest` for all crates, doctests, loom models, Miri on unsafe-heavy crates, clippy, `cargo xtask layers` and `provenance` |
| T1 pull request | every pull request, self-hosted bare metal with `/dev/kvm` plus hosted runners | 45 min | T0, qtest subset for touched crates, iotests `-quick` groups for qcow2 and raw, tcg multiarch tests for touched targets, QAPI schema tests, short differential runs, fuzz regression corpus replay |
| T2 merge queue | before landing on main | 90 min | full qtest and iotests for x86_64 and aarch64 targets, functional tests at `SPEED=quick`, bios-tables-test, migration round trips for q35 and virt |
| T3 nightly | main, all hardware classes | 10 hours | all targets' qtest, all iotests formats, functional tests at `SPEED=thorough`, tcg tests all targets, kvm-unit-tests on KVM and TCG, risu, ACT, guest boot matrix core set, 8 hours of fuzzing per target |
| T4 weekly | main | 48 hours | full guest boot matrix, libvirt TCK, Avocado-vt tp-qemu and tp-libvirt subsets, Arm ACS, Sail lockstep campaigns, long fuzzing, TestFloat level 2 |
| T5 release | release candidates | as weekly plus manual | everything, plus Windows and macOS guest runs on licensed hardware, sign-off per area |

A test that fails intermittently is quarantined within one working day by the area owner, which moves it to a quarantine list that still runs and reports but does not block merges. Quarantined tests have a two week limit: after that they either get fixed or become a tracked known failure with a milestone.

## Running QEMU's own test suites against ruvm

QEMU's testing documentation (`docs/devel/testing/main.rst`) lists the suites we use: unit tests in `tests/unit`, qtest, the QAPI schema tests in `tests/qapi-schema`, iotests in `tests/qemu-iotests`, TCG tests via `make check-tcg`, and functional tests in `tests/functional`. Each runs against ruvm in a different way.

### qtest

qtest is QEMU's device test framework ([QEMU docs: QTest](https://www.qemu.org/docs/master/devel/testing/qtest.html)). A test binary (C, linked against `tests/qtest/libqtest.c` and libqos) starts the emulator named by the `QTEST_QEMU_BINARY` environment variable, passing `-qtest` with a socket path and `-accel qtest`, plus a QMP socket. The test then drives the machine over a line-based text protocol: `outb`, `outw`, `outl` and `inb`, `inw`, `inl` for port I/O; `writeb` through `writeq` and `readb` through `readq` for memory; `read`, `write`, `b64read`, `b64write`, and `memset` for bulk memory; `clock_step` and `clock_set` for the virtual clock; `irq_intercept_in`, `irq_intercept_out`, and `set_irq_in` for GPIO lines, with asynchronous `IRQ raise` and `IRQ lower` notifications. There are a few additional commands in `system/qtest.c` (for example `endianness`, `module_load`, and the RTAS and PPC `rtas` call used by pSeries tests) that are not in the documentation but are used by tests, so we implement the protocol from `system/qtest.c` rather than from the docs.

ruvm implements the server side in ruvm-accel-qtest (L2). The qtest accelerator creates no vCPU execution: vCPUs exist as objects (so `query-cpus-fast` and CPU reset work) but never run. The virtual clock is advanced only by `clock_step` and `clock_set`, which run expired timers in deadline order on the main loop, with the same ordering rules as QEMU's `system/qtest.c` implementation. Memory commands go through the normal AddressSpace dispatch (`address_space_memory` of the first CPU, or `address_space_io` for port commands), which means qtest exercises the same ruvm-mem dispatch path as a real guest, including MemTxAttrs defaults. IRQ interception replaces the named GPIO inputs or outputs of a QOM object with ruvm-hw-core IRQ lines that report level changes on the qtest socket; the protocol requires that interception can only be attached once per test process, and ruvm returns the same error text as QEMU if a test tries twice.

The wire format is byte-for-byte QEMU's: responses start with `OK` or `FAIL`, values are printed as `0x%016` style hex exactly as `system/qtest.c` prints them, and base64 uses the same alphabet and padding. The qtest log (`-qtest-log`) is also produced in QEMU's format because some fuzzing reproducers are qtest logs replayed with `-qtest stdio`.

libqtest does more than speak the protocol. It passes `-display none`, `-accel qtest`, and a set of `-chardev` and `-mon` options; it waits for the QMP greeting; it reads `QTEST_QEMU_IMG` and `QTEST_QEMU_STORAGE_DAEMON_BINARY` for tests that need those tools; and for migration tests it spawns two instances. All of these must work. We build QEMU's qtest binaries from the pinned tree once per target (they do not depend on QEMU's emulator code, only on libqtest, libqos, and glib) and run them with `QTEST_QEMU_BINARY=/path/to/ruvm-symlink/qemu-system-<arch>`. The build uses QEMU's own meson setup with `--disable-system --disable-user --enable-tools=off` plus the qtest targets, so we never compile QEMU's emulator.

The qos graph (`tests/qtest/libqos/qgraph.c`) generates test cases from a graph of machines, buses, drivers, and tests, and composes command lines with `-device` options for every reachable path. This is valuable to us because it produces device combinations nobody would write by hand, for example virtio devices on every transport each machine supports. The `qos-test` binary is run in full for every target in T3.

Some qtests check things that are observable but only incidentally part of the contract, such as `qom-test` walking every QOM property of every machine and `device-introspect-test` listing every device type's properties. Those are part of our contract (libvirt reads `device-list-properties` and `qom-list-properties`), so they run and must pass. A small number of tests depend on QEMU internals that are not guest or management visible; the skip list `tests/upstream/skip/qtest.toml` holds each with a reason, a link to an issue, and an expiry milestone. The target is zero skipped qtests by M12 for tier 1 targets (x86_64, aarch64, riscv64, s390x, ppc64).

### iotests

iotests (`tests/qemu-iotests`) are the block layer's regression suite: numbered bash tests and named Python tests under `tests/qemu-iotests/tests`, each with a golden `.out` file. The `check` runner sets up the environment in `testenv.py`, which exports `QEMU_PROG`, `QEMU_IMG_PROG`, `QEMU_IO_PROG`, `QEMU_NBD_PROG`, and `QSD_PROG` among others and derives default paths from the build directory. We run `check` from the pinned tree with those variables pointing at `ruvm` multi-call symlinks (`qemu-system-x86_64`, `qemu-img`, `qemu-io`, `qemu-nbd`, `qemu-storage-daemon`).

iotests are strict. Golden output files contain exact `qemu-img info` and `qemu-img check` text, exact error messages, exact QMP event sequences and job status transitions, and exact `qemu-io` output including its timing line format (filtered by the test's filter functions). This is why document 14 requires ruvm-img and ruvm-io to reproduce QEMU's messages verbatim. We run the suite for each format and protocol combination QEMU's CI uses: `-qcow2`, `-raw`, `-qed`, `-vmdk`, `-vpc`, `-vhdx`, `-luks`, `-parallels`, `-qcow` and `-nbd`, `-file`, `-fuse` where FUSE is available, and `-nfs` on a runner with an NFS server. The `quick` group runs in T1; the full set in T3.

Python iotests import `iotests.py`, which imports QEMU's `qemu.machine` and `qemu.qmp` Python packages. Those talk to ruvm over QMP like any client, so they work without changes as long as the QMP transcript is identical, which the QMP diffing below checks independently.

### Functional tests

The functional tests in `tests/functional` are Python unittest based tests that boot real guests (kernels, firmware, disk images, and sometimes full distributions) and interact with them over serial console and QMP ([QEMU docs: functional testing](https://www.qemu.org/docs/master/devel/testing/functional.html)). They replaced QEMU's older Avocado based tests: the new framework was merged in September 2024, and Thomas Huth's April 2025 series converted the remaining Avocado tests and removed `tests/avocado` ([patch series](https://www.mail-archive.com/qemu-devel@nongnu.org/msg1107498.html)). Each test file can be run directly with `QEMU_TEST_QEMU_BINARY` set to the emulator; downloaded assets are cached in `~/.cache/qemu/download` or `QEMU_TEST_CACHE_DIR`; the `SPEED=thorough` mode includes the asset-downloading tests. Optional classes of tests are gated by `QEMU_TEST_ALLOW_LARGE_STORAGE`, `QEMU_TEST_ALLOW_UNTRUSTED_CODE`, `QEMU_TEST_FLAKY_TESTS`, and `QEMU_TEST_ALLOW_SLOW`.

These tests matter more to ruvm than to QEMU because they encode the long tail: boards like `raspi4b`, `aspeed` BMC machines, `sbsa-ref`, `imx8mp-evk`, MIPS Malta, SPARC sun4m, m68k `q800`, HPPA, and dozens of others, each with a known-good kernel and an expected console string. They are the acceptance tests for M10 (long tail targets and boards). We run them from the pinned tree with the binary variable pointing at ruvm, share the asset cache across CI runners through a read-only NFS mirror (prepopulated by `make precache-functional` from the pinned tree), and never download assets during a gated run, so an upstream mirror outage cannot fail our CI.

Some functional tests depend on QEMU build details, for example checking that a specific firmware file exists under `pc-bios`. ruvm installs firmware blobs in the same relative layout (document 11), so those pass. Tests that call `qemu-system-*` with `-d` debug options and parse the log text (for example some TCG plugin and replay tests) require ruvm's `-d` output to follow QEMU's format for the specific log items they parse (`in_asm`, `exec`, `cpu`); document 07 commits to that for those items only.

### TCG tests

`tests/tcg` holds per-target test programs (C and assembly) built with cross compilers and run under the linux-user binary or, for system mode tests, as bare-metal images under `qemu-system-<arch>` with a semihosting or debug-exit harness. `make check-tcg` runs them; the multiarch directory holds tests shared across targets (sha1, sha512, float conversions, signal handling, threading, `linux-test`, gdbstub tests driven by GDB scripts), and the per-target directories cover ISA specifics such as x86 `test-i386` instruction tests, aarch64 SVE, MTE, PAuth, and BTI tests, and s390x vector tests. The plugin tests in `tests/tcg/plugins` build the example plugins and run each test with each plugin loaded, which is a direct check of ruvm-plugin's C ABI compatibility.

We build the tests with QEMU's container cross toolchains (`tests/docker/dockerfiles`) and run them with a thin runner that reproduces the Makefile run rules, expected outputs, and timeouts but invokes ruvm; its test list is regenerated from the Makefiles on each submodule bump.

### Unit tests

`tests/unit` contains C tests that link directly against QEMU objects (for example `test-hbitmap`, `test-bdrv-drain`, `test-qobject-input-visitor`, `test-crypto-*`, `test-aio`). They cannot run against ruvm because they test C functions, not behavior at an external interface. We handle them in two ways. Where the unit test checks an algorithm whose output is externally visible (hbitmap serialization used in dirty bitmap migration, crypto cipher and hash outputs, QObject JSON formatting, the qcow2 refcount rebuild logic), we port the test cases to Rust tests in the corresponding ruvm crate with the same inputs and expected outputs, and keep a mapping file `tests/upstream/unit-map.toml` from each upstream test to its port. Where the unit test checks internal mechanics (coroutine scheduling, `aio_poll` details, drain semantics as a C API), we do not port it, but the behavior it protects is covered by iotests and our own ruvm-aio and ruvm-block tests. The mapping file is reviewed on each submodule bump for new upstream unit tests.

### QAPI schema tests

`tests/qapi-schema` contains QAPI parser test cases, each with a `.json` input and expected `.out`, `.err`, and `.exit` files. ruvm-qapi has its own schema parser (a build-time Rust tool, document 18), and it runs this whole directory with identical expected output, including error messages and line and column positions. Any divergence in how it resolves includes, conditionals, features, or union discriminators would silently change the generated QMP interface.

On top of the parser tests, the generated schema is compared end to end: we start QEMU 11.1 and ruvm with the same machine and target and diff the output of `query-qmp-schema`. The comparison is exact on the JSON after canonical key ordering; entity names in the introspection output are QEMU's generated names (QEMU masks type names with numbers in introspection), and ruvm reproduces those numbers because it generates the introspection data with the same algorithm as QEMU's `scripts/qapi/introspect.py`. This diff runs for every system target and every accelerator, because the schema varies by build configuration (target specific commands, KVM only commands). When extensions are enabled (the `ruvm` personality or `RUVM_EXTENSIONS=1`), ruvm adds downstream `__io.github.tamnd.ruvm_*` entries (document 21 introduces the first one); the diff test runs in the `qemu-*` personalities with extensions off and must be empty.

### Skip lists and ratchets

Each upstream suite has a skip list in `tests/upstream/skip/`, one TOML file per suite, and every entry has four fields: test identifier, reason category (`internal`, `not-yet-implemented`, `host-feature`, `upstream-bug`), issue link, and the milestone by which it must be removed. CI enforces a ratchet: the number of skips per suite and target can only go down on main, and a pull request that adds a skip needs approval from the testing area owner. `upstream-bug` entries are for tests that are broken in QEMU 11.1 itself (they fail against QEMU too); those are verified weekly by running the same test against the reference QEMU binary.

## Differential testing against QEMU

The reference QEMU binary is 11.1.0 built from the same pinned tree with the build rules of document 21, plus a debug build with `--enable-debug` for investigation. The harness is `ruvm-difftest`, a Rust program that starts both emulators with the same arguments, feeds them the same inputs, and compares defined observation streams. It runs in T1 (short runs on touched areas), T3 (full), and T4 (long campaigns).

### Lockstep execution for the JIT

To compare CPU emulation we need both emulators to execute the same guest instruction stream with the same external inputs, then compare architectural state. The design:

- Both run under TCG with `-icount shift=0,align=off,sleep=off` and the same `-rtc clock=vm,base=...` so that time is a function of instructions executed, not host time. Both use `-smp 1` for lockstep runs; multi-vCPU differential testing uses record and replay (below).
- QEMU exports state through a TCG plugin, `difftest-probe.so`, built against `qemu-plugin.h`. It registers a callback on each translated block execution (or per instruction in fine mode) and reads registers with the plugin register API (`qemu_plugin_get_registers` and `qemu_plugin_read_register`), streaming PC, general purpose registers, flags, and a rolling hash of vector registers over a shared memory ring.
- ruvm loads the same plugin binary. This is deliberate: since ruvm-plugin implements the QEMU plugin C ABI bit for bit, the identical `.so` producing identical streams on both sides is itself a test of ruvm-plugin, and it guarantees the probe observes both emulators the same way.
- The comparator reads both rings and compares state at each block boundary. Block boundaries can differ legitimately (ruvm's tier 1 may form different blocks than TCG), so comparison is keyed on retired guest instruction count, which icount makes identical: state is compared whenever both sides report the same instruction count, which happens at every boundary the two share, and at worst every few instructions. In fine mode, both sides use per-instruction callbacks and compare every instruction.
- On divergence, the harness records the last matching instruction count, restarts both emulators from a snapshot taken before that point (both support `-loadvm` of a qcow2 internal snapshot), reruns in fine mode to find the first differing instruction, disassembles it with both disassemblers, and writes a reproducer consisting of the snapshot, the instruction count, and the differing registers.

Workloads for lockstep: Linux boot to init for each tier 1 target, the tcg tests, SPEC CPU2017 test-size runs, and random programs from the fuzzers below. Tier 2 runs lockstep with tier 2 forced on at a low threshold, so that every hot loop in a boot is optimized, and with the comparison mode that only checks state at tier 2 region exits (tier 2 is allowed to keep guest state in host registers inside a region, so intermediate state is not visible; precise exceptions via side tables must still produce identical state at any fault, which the fault injection mode checks by making random pages unmapped mid-run on both sides).

Memory ordering bugs do not show up in single vCPU lockstep. For those we use litmus tests (below in ISA conformance) and differential runs of multithreaded programs that check final outcomes, not interleavings.

### Record and replay for multi-vCPU and devices

QEMU's record and replay (`-icount shift=auto,rr=record,rrfile=...`) captures nondeterministic inputs (device I/O, clock reads, interrupts) into a log. ruvm implements the same replay log format (document 17). This gives a second differential mode: record a run in QEMU, replay it in ruvm, and compare the observation streams. It also gives a way to bring a hard-to-reproduce ruvm bug back to QEMU for comparison. Replay format compatibility is itself tested by recording in each emulator and replaying in the other.

### Device register trace diffing

For device models the observation stream is the sequence of register accesses and their results. QEMU's trace events `memory_region_ops_read` and `memory_region_ops_write` (in `system/trace-events`) log every dispatched MMIO and PIO access with the region name, address, value, and size. ruvm-trace implements the same events with the same arguments. The harness runs the same guest workload in both under TCG with icount (so that the guest takes the same path), enables those events plus device specific events, and diffs the streams after normalizing pointer arguments.

A divergence in a read value is almost always a device model bug. A divergence in the access sequence itself means the guest took a different path, which means an earlier read differed or an interrupt arrived at a different instruction; the harness reports the first difference and discards the rest. Interrupt timing is compared through the trace events for interrupt controllers (for example `ioapic_set_irq`, `gicv3_*` events), which must fire at the same instruction count. The workloads are the qtests (as drivers of device behavior), the guest boot matrix core set, and fuzzer outputs.

### ACPI, SMBIOS, and device tree comparison

Guest-visible firmware tables must be identical, because Windows activation, Linux device naming, and firmware behavior depend on them. Three checks:

- `bios-tables-test` from `tests/qtest` runs against ruvm unchanged. It boots with a small test disk, locates the RSDP, reads all ACPI tables through qtest memory commands, and compares them to the expected blobs in `tests/data/acpi/<arch>/<machine>/`, with `iasl` disassembly diffs on mismatch. It covers q35, pc, microvm, arm virt, and riscv virt variants with many options (NUMA, memory hotplug, CXL, TPM, IOMMU variants).
- Beyond the test's configurations, `ruvm-difftest acpi` generates random valid machine configurations (vCPU count and topology, NUMA layout, memory size, hotplug slots, PCI expander bridges, IOMMU choice, TPM, CXL windows) and compares the table blobs from both emulators byte for byte, reading them through fw_cfg (`etc/acpi/tables`, `etc/acpi/rsdp`, `etc/table-loader`) via qtest. It also compares the SMBIOS blobs (`etc/smbios/smbios-tables`, `etc/smbios/smbios-anchor`) and the full fw_cfg file directory.
- For device tree machines, `-machine dumpdtb=file.dtb` on both sides, compiled back with `dtc -I dtb -O dts` and diffed. Node order and phandle numbers must match, because some guests and bootloaders depend on them.

The AML cache that document 21 uses for boot speed is tested here by running every configuration twice (cold and warm cache) and requiring identical output.

### QMP response diffing

`ruvm-difftest qmp` drives both emulators with the same sequence of QMP commands and compares responses and events. Command sequences come from three sources: transcripts captured from libvirt test runs (libvirt keeps capability probing replies for many QEMU versions in `tests/qemucapabilitiesdata/*.replies`, and the 11.1 replies file is a direct target for ruvm), transcripts from the iotests and qtests, and generated sequences from the QMP fuzzer. Normalization removes values that legitimately differ (timestamps in events, PIDs, thread ids, host paths in `query-block` for temporary files, and memory addresses in `query-memory-devices` only where QEMU itself makes them host dependent). Everything else must match, including error `class` and `desc` strings, the order of array elements, and the order of events.

The libvirt capabilities comparison is particularly strict. libvirt probes a QEMU binary with a long QMP script (`query-version`, `query-commands`, `query-qmp-schema`, `qom-list-types`, `device-list-properties` for many devices, `query-cpu-definitions`, `query-cpu-model-expansion`, `query-machines`, `query-sev-capabilities` and more) and caches the result; a single missing property or reordered CPU model changes libvirt's behavior. We run libvirt's own `qemucapsprobe` tool against ruvm for each target and require the output to be identical to the same tool run against QEMU 11.1 on the same host, with the host dependent CPU sections compared separately.

### Migration stream round trips

Migration compatibility is tested at three levels:

- Static: QEMU's `-dump-vmstate` option writes a JSON description of every VMState for a machine type, and `scripts/vmstate-static-checker.py` compares two such dumps for compatibility. ruvm implements `-dump-vmstate` and we compare ruvm's dump against QEMU's for every versioned machine type we support. The checker is run in both directions (QEMU as source, ruvm as destination, and the reverse) and must report no incompatibilities.
- Stream: `scripts/analyze-migration.py` parses a migration stream saved to a file. We save the state of the same paused VM from both emulators with `migrate` to a `file:` URI and compare the parsed streams section by section. RAM contents are compared by hash; device sections must be byte identical given identical guest state, which holds after a deterministic icount run to the same instruction count.
- Live: the round trip QEMU to ruvm to QEMU. A guest boots in QEMU, runs a workload (the guestperf `stress` program, a fio job with data verification, and a network ping stream), migrates live to ruvm, keeps running, migrates back to QEMU, and then verifies data integrity (fio verify, stress checksum) and that no guest error appeared in dmesg. The same runs in the other direction, ruvm to QEMU to ruvm. This matrix runs for every machine type version from the oldest QEMU still supports down to 11.1 (`pc-q35-*`, `pc-i440fx-*`, `virt-*`, `s390-ccw-virtio-*`, `pseries-*`), with each device class that supports migration (virtio devices on each transport, e1000e, AHCI, USB xHCI with a tablet and storage, VGA and virtio-gpu, TPM, vIOMMU), with post-copy, multifd, and compression variants. QEMU's own `tests/qtest/migration-test` also runs against ruvm, with both source and destination set to ruvm and then, using its support for different source and destination binaries (`QTEST_QEMU_BINARY_SRC` and `QTEST_QEMU_BINARY_DST`), with one side set to QEMU 11.1.

### Command line and help output

Management tools parse `-help`, `-machine help`, `-cpu help`, `-device help`, and `-device <name>,help` output, and some scripts parse `qemu-img --help`. The harness diffs these for every target and tool. Differences are permitted only in the version banner line, which reports ruvm's version alongside the QEMU version it is compatible with (document 02).

## ISA conformance

Differential testing against QEMU finds places where ruvm differs from QEMU. It does not find places where both are wrong, and ruvm's tier 2 optimizer creates new ways to be wrong that QEMU never had. So the JIT is also checked against hardware and against formal models.

- risu ([Peter Maydell's repository](https://git.linaro.org/people/peter.maydell/risu.git/about/)): `risugen` generates random instruction sequences from pattern files, and `risu` runs them on real hardware (the master) and under the emulator (the apprentice), comparing register state after every instruction. risu supports aarch64, arm, ppc64, s390x, and others; we use its record mode so that traces captured once on A-NEO and on an IBM Power and z host (rented time, weekly) are replayed against ruvm linux-user in T3 without hardware. For x86 we use the same approach with a ruvm-maintained x86 pattern set, checked against X-AMD and X-INTEL hardware, because Intel and AMD differ on some undefined flag results; the divergence registry records which vendor each `-cpu` model follows.
- RISC-V Architectural Certification Tests ([riscv-arch-test](https://github.com/riscv/riscv-arch-test)) with the ACT4 framework: tests are compiled into self-checking ELFs whose expected values come from the Sail RISC-V reference model configured to match the device under test. We maintain the UDB configuration and `rvmodel_macros.h` for ruvm's `-M virt` with each `-cpu` model we claim, and run all applicable tests in T3. A failure against Sail is a ruvm bug unless the spec allows the behavior and QEMU chose differently, in which case we follow QEMU and record it.
- Arm system architecture compliance ([sysarch-acs](https://github.com/ARM-software/sysarch-acs), the successor of bsa-acs and sbsa-acs): the BSA and SBSA suites run as UEFI applications on `-M sbsa-ref` with the edk2 firmware QEMU uses, weekly, and the result must match QEMU 11.1's result on the same image test for test. These test the platform (GIC, SMMU, timers, PCIe, watchdog) more than the ISA.
- Formal models as oracles: the Sail RISC-V model ([sail-riscv](https://github.com/riscv/sail-riscv)) and the Sail Armv9.4-A model translated from Arm's ASL ([sail-arm](https://github.com/rems-project/sail-arm)) are compiled to C emulators and wrapped in a lockstep adapter speaking the same probe protocol as `difftest-probe.so`. Random instruction streams from the JIT fuzzer run in ruvm and in the model, comparing architectural state per instruction. The Arm model is slow (Pydrofoil shows that faster Sail derived emulators are possible for RISC-V), so Arm model runs are limited to short random programs at user and EL1 level. Where the model and QEMU disagree we investigate before choosing; we expect most such cases to be IMPLEMENTATION DEFINED behavior, which is pinned to QEMU's choice.
- Softfloat: QEMU tests `fpu/softfloat.c` with `tests/fp/fp-test`, which uses Berkeley TestFloat and SoftFloat release 3 as submodules. ruvm-softfloat is tested with the same TestFloat generators at levels 1 and 2 for every operation, rounding mode, and tininess mode, and additionally against QEMU's softfloat itself: a test-only crate links QEMU's `fpu/softfloat.c` (GPL, test only, never shipped) and compares results and exception flags for every target's `float_status` configuration, including default NaN, NaN propagation rules, flush-to-zero, and x87 80-bit behavior. Bit exactness with QEMU is the requirement (canon), so the QEMU comparison is the gate and TestFloat is the sanity check on both.
- kvm-unit-tests: the suite (`x86`, `arm`, `riscv`, `s390x`, `powerpc`) runs with its `run_tests.sh` and the `QEMU` environment variable pointing at ruvm, under KVM on each host class and under TCG (`ACCEL=tcg`). It tests CPU and platform behavior from bare-metal guests: APIC and x2APIC, PMU, debug registers, SVM and VMX nested virtualization under KVM, GICv3 and ITS, PSCI, timers, s390x SIE and interruption handling. The pass and skip set must equal QEMU 11.1's on the same host.
- Memory model litmus tests: for strong-on-weak translation (x86 guests on Arm and RISC-V hosts), we run x86-TSO litmus tests generated with herdtools7 `diy` and run with `litmus7` in linux-user, millions of iterations each, checking that no outcome forbidden by x86-TSO appears. This complements the formal verification of the fence mappings borrowed from Risotto and Arancini (document 08): proofs cover the mapping, litmus tests cover our implementation of it.
- LTP (Linux Test Project) syscall tests run under ruvm linux-user for x86_64, aarch64, riscv64, s390x, and ppc64le guests on each host, compared to QEMU linux-user's pass set on the same host. This is the main conformance suite for document 10's syscall translation.

## Guest OS boot matrix

A guest "passes" when it boots to a defined point, runs a smoke script over the serial console or QEMU guest agent (network up, disk read and write with checksum, clean shutdown), and produces no kernel warnings that it does not produce under QEMU 11.1 with the same configuration. Each entry runs under KVM (or HVF on A-APPLE) where the host architecture matches, and under TCG otherwise.

| Family | Guests | Architectures and machines |
|---|---|---|
| Linux distributions | Debian 13, Ubuntu 24.04 and 26.04, Fedora (current two releases), AlmaLinux 9 and 10, openSUSE Leap, Alpine, Arch Linux | x86_64 on q35, pc, microvm; aarch64 on virt and sbsa-ref; riscv64 on virt; ppc64le on pseries; s390x on s390-ccw-virtio |
| Linux long tail | kernels and rootfs images from QEMU's functional test assets | mips, loongarch64, sparc64, m68k, alpha, hppa, sh4, and the embedded boards |
| Windows | Windows 11 and Windows Server 2025 (evaluation media), with OVMF Secure Boot, swtpm TPM 2.0, virtio-win drivers; Windows 11 on Arm on arm virt | x86_64 q35, aarch64 virt |
| BSDs | FreeBSD (current and previous release), OpenBSD, NetBSD, DragonFly BSD | x86_64; FreeBSD and NetBSD also aarch64 and riscv64 |
| Others | Haiku, 9front (Plan 9), illumos (OpenIndiana), ReactOS, FreeDOS, MS-DOS 4.0 built from Microsoft's MIT licensed source release, Windows 3.x and 9x from owned media on private runners | x86 pc and isapc |
| macOS | macOS 12 guests on the `vmapple` machine with HVF | A-APPLE only |

macOS guests are tested only on Apple hardware, because Apple's license permits macOS virtualization only on Apple-branded hardware, and only on `vmapple`, which QEMU documents as supporting macOS 12 guests on Apple silicon ([QEMU vmapple docs](https://www.qemu.org/docs/master/system/arm/vmapple.html)). Guests whose licenses do not permit redistribution (Windows 9x, retail DOS) live on private runners with the media, and their results are published without the images.

The core set (one Linux per tier 1 architecture, Windows 11, FreeBSD, and Haiku) runs nightly; the full matrix runs weekly and in T5.

## Management stack suites

- libvirt TCK ([libvirt-tck](https://libvirt.org/testtck.html)): Perl based functional tests of libvirt driving a real hypervisor (domain lifecycle, save and restore, snapshots, hotplug, storage pools, networks). It runs weekly on a clean dedicated host (the TCK requires an empty libvirt state), with libvirt's QEMU driver configured to use ruvm's `qemu-system-*` symlinks.
- Avocado-vt with its test providers [tp-qemu](https://github.com/autotest/tp-qemu) and tp-libvirt: large suites used by distribution QA for QEMU and libvirt. We run curated subsets weekly (block, migration, hotplug, virtio, CPU model, and guest agent tests for Linux and Windows guests) and grow them toward the full set for M11.
- libvirt's own unit tests are not run against ruvm, but their QEMU capability and command line fixtures are used as described under QMP response diffing: libvirt generates a command line for each test domain XML from the 11.1 capabilities, and we check that ruvm accepts every generated command line and produces the same `query-*` view of the resulting VM as QEMU.
- OpenStack Tempest (compute and volume API tests with the libvirt driver) and Proxmox VE's QEMU integration are M11 certification targets (document 23); they run on a dedicated two-node lab before each release.

## Fuzzing

QEMU fuzzes its device models with libFuzzer targets in `tests/qtest/fuzz`, most importantly `generic-fuzz`, which fuzzes the PIO, MMIO, and DMA spaces of the memory regions named by `QEMU_FUZZ_OBJECTS` in a machine configured by `QEMU_FUZZ_ARGS`, with predefined configurations in `generic_fuzz_configs.h`, running continuously on OSS-Fuzz ([QEMU docs: fuzzing](https://www.qemu.org/docs/master/devel/testing/fuzzing.html)). ruvm adopts that design and extends it.

- Device fuzzing: `ruvm-fuzz-device` is a cargo-fuzz (libFuzzer) target that links ruvm in process with the qtest accelerator and interprets the fuzz input with the same opcode encoding as QEMU's generic-fuzz (so crash inputs can be converted to qtest command sequences with the same tooling, and QEMU's reproducer format works). It reads the same `QEMU_FUZZ_ARGS` and `QEMU_FUZZ_OBJECTS` variables and ships the upstream configuration list. Between inputs, state is reset by a snapshot restore of the machine's device state (fork-based reset like QEMU's is also supported on Linux). Coverage comes from SanitizerCoverage instrumentation as cargo-fuzz configures it, on ruvm crates only. Beyond crashes and sanitizer reports, the fuzzer checks invariants: no Rust panic in device code, no access outside guest RAM through DMA helpers, device state serializable by VMState after every input (a save and load round trip must reproduce identical state), and no reentrant MMIO dispatch into a device that is not marked reentrancy safe (QEMU added its `MemReentrancyGuard` after a series of such bugs; ruvm enforces it in the type system but fuzzes it anyway).
- Differential device fuzzing: crash-free inputs are periodically replayed against QEMU 11.1 as qtest scripts and the qtest transcripts compared, which turns the fuzzer into a behavior comparison tool as well.
- JIT fuzzing: a generator produces random but valid guest instruction sequences per target (from the decodetree patterns, so every encoding the decoder accepts is reachable), wraps them in a harness that sets random initial state, and runs them in ruvm linux-user and QEMU linux-user in lockstep, comparing register state per block, memory side effects by hash, and signals raised. Separate campaigns run with tier 2 forced and with random self-modifying stores into the sequence. Divergences are minimized automatically by delta debugging on the instruction list. A second mode uses the Sail models as oracle instead of QEMU for RISC-V and Arm.
- Block format fuzzing: structure aware fuzzers for qcow2, VMDK, VHDX, VDI, VPC, QED, parallels, DMG, and LUKS headers mutate valid images (headers, L1 and L2 tables, refcount blocks, snapshot tables, bitmap extensions) and run `qemu-img info`, `check`, `check -r all`, `convert`, and random guest style read and write sequences through ruvm-io. The oracles are no crash, no out of bounds host file access (the image file is opened through a wrapper that records all offsets), bounded memory allocation, and agreement with QEMU's `qemu-img check` verdict on the same image. NBD and the storage daemon's export protocols get protocol fuzzers too.
- QMP fuzzing: the fuzzer derives a grammar from `query-qmp-schema` and generates well typed command sequences with random arguments, plus malformed JSON and out of range values. Oracles: no crash or hang, every response is either a success matching the schema's return type or an error with a class QEMU uses, and responses match QEMU's when the same sequence is replayed there (after normalization). Sequences that change the device tree (`device_add`, `device_del`, `blockdev-add`, `object-add`) are weighted heavily because topology changes are where the control lock and lock domains (document 03) interact.
- Guest facing parsers not reached by the above (fw_cfg DMA descriptors, virtio ring layouts, USB descriptors from passthrough devices, VNC and SPICE protocol input, the gdbstub protocol, the guest agent's JSON channel) have dedicated targets.

ruvm will apply to OSS-Fuzz for continuous fuzzing of all targets once M3 lands. Our own fleet runs every target for at least 8 hours per night and keeps corpora in a bucket; T1 replays the regression corpus (every input that ever found a bug) in under 5 minutes.

## Concurrency testing

Without the BQL, ruvm relies on per-device locks, lock domains, RCU FlatViews, and lock-free fast paths (document 03). Those need testing that ordinary tests cannot give.

- loom (exhaustive interleaving exploration under the C11 memory model) for small core primitives: the epoch RCU in ruvm-base, the virtqueue notification fast path, the TB cache lookup and invalidation protocol, the dirty bitmap harvest, the per-vCPU exit request flags, and the io_uring submission queue wrapper. These are written as loom models with at most three or four threads so exploration finishes.
- shuttle ([awslabs/shuttle](https://github.com/awslabs/shuttle)) for larger components where exhaustive search does not scale: randomized and PCT schedulers over the real code with shuttle's drop-in sync primitives, run for hundreds of thousands of iterations nightly. Targets include hotplug concurrent with MMIO from multiple vCPUs, block job completion concurrent with drain and `blockdev-del`, migration stop concurrent with device interrupts, and QMP commands concurrent with guest reset. Failing schedules are saved and replayed as regression tests.
- Deterministic whole-VM simulation: a ruvm build feature `sim` replaces thread spawning, the reactors, and host clocks with a single threaded deterministic scheduler driven by a seed, while vCPUs run under the qtest or TCG icount accelerator. The simulator runs full machines (q35 with virtio devices and an iothread per device) with random scheduling decisions and injected faults (I/O errors, short reads, delayed completions, backend disconnects for vhost-user), checking invariants after every step. A failing seed reproduces exactly. This is the FoundationDB style of testing applied to a VMM; it finds ordering bugs between iothreads and vCPUs that loom and shuttle models are too small to contain.
- ThreadSanitizer builds of ruvm run the qtest and iotests suites nightly on x86-64, and Miri runs the unit tests of crates with unsafe code (ruvm-base, ruvm-mem, ruvm-aio, ruvm-jit runtime helpers) with Stacked Borrows checking.

## CI infrastructure

Hosted runners (GitHub Actions) run T0 on Linux x86-64 and aarch64, macOS arm64, and Windows x86-64, which covers builds, unit tests, TCG only tests, and WHPX compile checks. Everything that needs hardware virtualization or stable timing runs on self-hosted bare metal, registered as ephemeral runners that reimage between jobs.

| Pool | Machines | Purpose |
|---|---|---|
| kvm-x86 | 6x AMD EPYC, 4x Intel Xeon (including one with TDX and one with SEV-SNP enabled) | KVM tests, kvm-unit-tests, confidential computing tests, boot matrix |
| kvm-arm | 4x Neoverse V2 bare metal | KVM arm64, sbsa-ref ACS, arm guests |
| hvf | 4x Apple M4 Mac mini | HVF, macOS host builds, vmapple |
| whpx | 2x x86 Windows Server hosts with Hyper-V platform | WHPX tests |
| exotic | rented time on IBM Power and z hosts, one RISC-V board with the H extension when available | risu traces, native KVM on those hosts |
| fuzz | 4x high core count x86 hosts | nightly fuzzing |
| bench | the document 21 classes | performance only, never shared with functional tests |

Nested virtualization is used only for smoke tests. Cloud VMs with nested KVM work for booting guests, but exit costs are much higher, timers are less precise, and some features (nested VMX inside nested, PMU passthrough, SEV and TDX) are unavailable, so kvm-unit-tests and anything timing sensitive run only on bare metal. A test marked `requires = "bare-metal"` fails loudly if scheduled on a nested runner rather than silently skipping.

Cost is controlled by tiering, not by skipping tests. The rough budget is 10 hours of wall time per night across the pools and 48 hours per week, which the pool sizes above are chosen to fit; weekly Windows, ACS, and Sail campaigns run on the kvm pools when nightly jobs are finished. Asset caches (guest images, functional test assets, cross toolchain containers) are mirrored locally so that no gated run downloads from the internet.

## The conformance dashboard

The dashboard (a static site regenerated after every T3 and T4 run and published with each release) answers one question per row: how close is ruvm to being a drop-in QEMU 11.1 for this target, machine, or suite. It shows:

- Upstream suite pass rates per target: qtest, iotests per format, functional (quick and thorough), tcg tests, QAPI schema tests, with counts of pass, fail, skip by reason category, and quarantine. Each number links to the skip list entry or failure log.
- Differential status: QMP schema diff (must be empty), libvirt capabilities diff per target, ACPI and DTB configurations compared and mismatches, `-dump-vmstate` checker status per machine type version, and the live round trip matrix (source emulator, destination emulator, machine type, device class) as a colored grid.
- ISA conformance: ACT pass rate per `-cpu` model, risu instructions compared per pattern file, lockstep instructions executed without divergence in the last week per target (this number grows over time and a divergence resets nothing but opens an issue), TestFloat and QEMU softfloat comparison status, kvm-unit-tests pass and skip set compared to QEMU's, LTP pass set compared to QEMU linux-user.
- Guest boot matrix: one cell per guest and machine and accelerator, with the date of the last pass.
- Fuzzing: executions per day per target, open crash count, median age of open fuzz bugs.
- Concurrency: shuttle iterations per night, open issues from simulation seeds.
- Divergence registry: every intentional difference from QEMU with its justification, so users can see exactly where ruvm is not a drop-in.

A single "drop-in score" per tier 1 target is the product of the qtest, iotests, functional, and tcg pass fractions for that target, times 1 if every differential check is clean and 0 otherwise. We publish it, but the release criterion for M12 in document 23 is stated on the underlying numbers (no skipped qtests or iotests for tier 1 targets, empty schema and capability diffs, full live round trip grid green), not on the score.

## Decisions made in this document

- Upstream QEMU tests run from a submodule pinned at `v11.1.0` under `tests/upstream/qemu`, with per-suite skip lists that carry a reason category and expiry milestone and can only shrink on main.
- ruvm implements the qtest protocol from `system/qtest.c` (including undocumented commands) in ruvm-accel-qtest, and implements `-dump-vmstate`, `-machine dumpdtb`, `-qtest-log`, and the record and replay log format, because the tests depend on them.
- `-d` log output follows QEMU's format for `in_asm`, `exec`, and `cpu` only, since functional tests parse those.
- Lockstep JIT comparison uses a single QEMU TCG plugin loaded into both emulators, keyed on icount instruction counts.
- A test-only crate links QEMU's `fpu/softfloat.c` as the bit exactness oracle for ruvm-softfloat; it is never shipped.
- A deterministic simulation build feature `sim` exists for whole-VM seeded testing.
- Where QEMU is known to be wrong, ruvm follows the architecture unless migration compatibility requires the QEMU behavior, and records the difference in the divergence registry.
