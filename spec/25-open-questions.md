# 25. Open questions and decision log

This document has two parts. The first lists the questions that are still open, each with the options, what the answer depends on, the default the project follows until the question is settled, and the milestone by which it has to be answered. The second is the log of decisions the other documents made that go beyond the canon. It is kept in one place so a reviewer can see every commitment without reading 25 documents, and so that a later change to one of them is visible as a change to this list.

An open question is closed by a pull request that edits this file and the documents it affects. The question moves to the decision log with the date and the reason.

## Open questions

### Q1. The license split

The canon makes the project GPL-2.0-or-later, with permissive leaf crates (ruvm-base, ruvm-aio, ruvm-sys, ruvm-mem, ruvm-mem-vmm, ruvm-jit-core, ruvm-decode, the host backends where they are written fresh). The open part is whether the line holds up in practice. Document 05 keeps ruvm-mem permissive by writing QEMU's access splitting rules from the documented contract instead of porting `access_with_adjusted_size()`, and checking them by differential test. That argument is sound for one function. It gets weaker as more behavior is matched this way, because a careful clean room process costs time on every function.

Options: (a) keep the split and require a written provenance note for every permissive module that matches QEMU behavior, (b) make everything GPL and drop the rust-vmm reuse story, (c) narrow the permissive set to ruvm-base, ruvm-aio and ruvm-jit-core. Default: (a). Decide by M2, when ruvm-mem and the first host backend are real code and the cost of the clean room rule can be measured instead of guessed.

### Q2. Name of the downstream QMP extension namespace

The documents used four different spellings for ruvm-only QMP commands and events. This file settles it: the prefix is `__io.github.tamnd.ruvm_`, following QAPI's downstream extension rule (`__RFQDN_`), and documents 02, 04, 18 and 21 use it. Trace events added by ruvm use the `ruvm_` prefix. Command line and property extensions use `x-ruvm-` (for example `-accel tcg,x-ruvm-extra-isa=on`), following QEMU's `x-` convention for unstable interfaces. Extensions are visible when the binary is invoked as `ruvm`, or with `RUVM_EXTENSIONS=1` in the environment. They are never visible in the `qemu-*` personalities by default, so `query-qmp-schema` from those personalities stays byte-identical to QEMU's. Configuration for ruvm-only features (metrics, stricter sandboxing, extra tracing) goes through user creatable objects (`-object ruvm-metrics,...`, `-object ruvm-sandbox,...`) and not new top level options, so that the option table stays QEMU's. Document 21's `-ruvm-guest-symbols` and `-ruvm-trace-file` become properties of a `ruvm-trace` object.

The open part is only whether the RFQDN should be a domain the project controls instead of a GitHub namespace. If the project gets its own domain before 1.0, the prefix changes once, before 1.0, and never again.

### Q3. How far bug-for-bug compatibility goes

Document 02 makes bug-for-bug compatibility the default and lists four allowed reasons to diverge. Several documents found cases at the edge. Document 22 says that where QEMU is wrong against the architecture specification (a TCG instruction that sets a flag incorrectly, say), ruvm follows the architecture and records it in the divergence registry, unless migration compatibility needs QEMU's behavior. Document 19 refuses `json:` filenames that come from inside image files (backing file and data file names), which QEMU accepts. Document 05 drops QEMU's subpage machinery, so `info mtree -d` output differs.

The question is where the line sits for guest-visible CPU behavior. Following the architecture fixes real guest software and costs nothing on a fresh boot. But a guest that migrates from QEMU mid-computation can observe the change. Default: follow the architecture for TCG correctness bugs, gated by the machine version like every other guest-visible change, and record each one in `conformance/divergences.toml`. Revisit at M4, with the list of real cases from differential testing in hand.

### Q4. Lock domains versus QEMU's ordering in rare cases

Document 03 replaces the BQL with computed lock domains, and it argues that each of the six ordering guarantees the BQL provides (G1 to G6) is preserved. The argument covers the known cross-device paths: PCI config, MSI, memory map changes, peer-to-peer DMA, migration and reset, and I2C/SSI. The risk is a device on the long tail that relies on BQL ordering in a way nobody has written down. An example is a board device that pokes another device's state directly from a timer.

Options: (a) domains everywhere, with a per-device `serialize_with_control` flag as an escape hatch, (b) a board-level switch that puts every device on a board in one domain, which gives BQL semantics for that board, (c) both. Default: (c). The M10 long tail work uses (b) for boards that have no differential test coverage yet, and moves a board to (a) once its tests pass. Measure at M10 how many boards stay on (b).

### Q5. Guest call and return mapping in the JIT

Document 08 predicts indirect branches with an inline jump cache probe and a 16-entry software return address stack. Mapping guest calls and returns directly onto host `call` and `ret` (or `bl` and `ret`) would use the host's return predictor, which is faster, but it breaks when guests manipulate the return address. Longjmp, coroutine libraries and kernel context switches all do this. FEX-Emu and Box64 have both experimented with it. Default: the software stack. Decide at M9, based on how much of the profile indirect branches take after tier 2.

### Q6. Memory model shortcuts for x86 guests on Arm hosts

Document 07 implements the verified fence mappings from Risotto and Arancini as the default. It adds three optional shortcuts behind flags. The first uses LRCPC (`LDAPR`) instead of `LDAR` for acquire loads. The second drops fences on stack accesses, on the argument that stack memory is thread-private. The third uses Apple's hardware TSO mode where it is available. The first and third are sound. The second is unsound for programs that share stack addresses between threads, which does happen, although rarely.

The open part is the policy for exposing unsound but fast options. Options: (a) never expose them, (b) expose them only in the `ruvm` personality with `x-ruvm-` names and a warning on use, (c) turn them on per binary through a profile file, as FEX does. Default: (b). Hardware TSO is not a shortcut: on hosts that have it, ruvm uses it by default. Decide before M9 ships tier 2.

### Q7. LL/SC emulation precision

Document 08 keeps QEMU's approach of emulating load-linked and store-conditional with compare-and-swap. That approach has the ABA problem: a store conditional succeeds when another thread wrote the same value back in between, which real hardware would fail. Almost no software depends on the difference, but lock-free algorithms that use LL/SC to detect ABA are the exception. An exact mode is possible, using a per-reservation-granule version counter or page protection, but it costs throughput. Default: QEMU's behavior. Add an exact mode as `x-ruvm-exact-llsc` if a real workload is found that needs it.

### Q8. RISC-V user mode signal frames with vector state

QEMU's linux-user omits vector state from the RISC-V signal frame, and it does not set `AT_MINSIGSTKSZ` to account for vector registers. Document 10 matches QEMU. Linux does include the vector context. A program that inspects its own signal frame for vector state works on hardware and fails under QEMU and ruvm. Default: match QEMU until QEMU changes, then follow it. If ruvm's differential runs turn up a real failure first, send the fix upstream.

### Q9. guest_memfd in-place conversion

Document 05 uses guest_memfd for confidential guests with separate shared and private backing, which is QEMU's approach in 11.1. Upstream Linux work on in-place conversion (one guest_memfd that holds both shared and private pages, with conversion by attribute) would remove the double allocation and the copy on conversion. Default: follow QEMU, and adopt in-place conversion in the release after both Linux and QEMU merge it.

### Q10. Arm CCA

Arm CCA realm support is not upstream in either QEMU or Linux as of September 2026 (documents 01 and 19). ruvm does not implement it until both the KVM interface and QEMU's object and property names are merged. The object names matter because libvirt will encode them. The open part is whether ruvm should prototype against the posted series to shorten the lag. Default: no. The API surface moves too much, and M8 has enough work already.

### Q11. Tier 2 when plugins or icount are active

Document 08 turns tier 2 off for blocks instrumented by TCG plugins, and in icount and record/replay modes. Document 02 lets tier 2 run with plugins only if it preserves every callback. The simple rule loses the 2x gain for plugin users, and people doing performance analysis with plugins are exactly the users who would notice. Default: tier 2 off with plugins for M9. Revisit after M9 with a design for tier 2 that treats plugin callbacks as barriers.

### Q12. Cranelift as an alternative tier 2

Document 07 rejects Cranelift and LLVM as the main tiers because of compile latency. It allows a Cranelift tier 2 as an experiment behind a feature flag. The question is whether that experiment deserves staffing at all before 1.0. Default: no staffing. The flag exists so an outside contributor can try it without forking.

### Q13. Out-of-process devices by default

Document 01 proposes an optional mode that runs high-risk devices (USB, legacy storage controllers, audio) out of process through vfio-user. Document 20 does not include `x-pci-proxy-dev` (QEMU's multi-process device proxy) before 1.0. Document 19 treats any backend process ruvm ships as untrusted. These three agree, but they leave open whether the out-of-process mode should become the default for some devices once it exists. Default: off, and revisit after 1.0 with measured overhead.

### Q14. A WebAssembly host

QEMU 10.1 added an experimental Emscripten build. ruvm could target wasm64 with the interpreter backend, which would allow running guests in a browser. Document 20 puts WASM device sandboxing out of scope. A WASM host is a separate question. Default: out of scope for 1.0, since no milestone user asked for it. Nothing in the architecture prevents it, as long as ruvm-aio grows a backend for it.

### Q15. Captive-style host MMU use in system mode

Captive (VEE 2019) runs guest code under a host hypervisor so that guest virtual memory maps onto host page tables. That removes the softmmu TLB lookup entirely for same-width guests. It would raise the ceiling on TCG system mode performance well past 2x. It needs KVM on the host and a real change to the JIT runtime. Default: after 1.0 (document 01). Keep the softmmu interface narrow enough that a second implementation can sit behind it.

### Q16. Native library thunking in user mode

Document 10 adds an optional library forwarding mechanism in the style of FEX and Box64, as `-x-ruvm-thunks=<config>` on the `ruvm-<arch>` binaries only. It starts with GL, EGL, Vulkan and X11 for x86-64 guests on aarch64 hosts. It lies outside QEMU compatibility by design. The open part is priority. It is the biggest single speedup for gaming and graphics workloads in user mode, and it serves none of the compatibility goals. Default: after M9, staffed only if contributors show up for it.

### Q17. Security scope of TCG

Document 19 keeps TCG out of the 1.0 security scope. A guest escaping through a JIT bug is treated as a bug, not a security issue, which matches QEMU's stated position that TCG is not a security boundary. Rust and the verified mappings make ruvm's JIT easier to harden than QEMU's. The question is whether ruvm should aim to change that position after 1.0, which would need JIT fuzzing and hardening to reach the level of the device models. Default: revisit after 1.0.

### Q18. Passthrough loose ends

Document 16 leaves four items open. The first is whether CPR from a QEMU process into ruvm can cover every device type or only VFIO and RAM. The second is getting repeatable CI access to Tegra241 CMDQV and Arm nested translation hardware, which cannot be tested properly under emulation. The third is whether VFIO device data fd reads for multifd should go through io_uring once kernel drivers are known to handle it. The fourth is whether HVF or WHPX hosts ever get device assignment; neither has a VFIO equivalent. Current defaults: VFIO and RAM only, lab hardware for CMDQV and nesting once it can be bought, plain reads on the data fd until a profile shows the copy matters, and assignment on Linux only for 1.0.

## Decision log

These are decisions made in documents 01 through 22 that go beyond the canon. The document number says where the reasoning is.

### Compatibility and versioning

- In the `qemu-*` personalities, `query-version` and `-version` report QEMU 11.1.0, with the ruvm version in the `package` field (02, 18). libvirt keys its capability handling on the version number.
- The QEMU build used for schema comparison is configured to match the feature set of the ruvm build under test (02).
- A new versioned machine type ships only when all of its compat properties are implemented (02). `cargo xtask compat-sync` keeps the arrays in sync with upstream (04).
- ruvm follows each new QEMU release within about 60 days (02).
- The record and replay log format is not part of the contract (02), but ruvm uses QEMU's format anyway (17).
- The firmware blobs are QEMU's. Any ruvm-specific build of one gets a different file name (02).
- HMP has a fixed list of commands that must match QEMU byte for byte. QMP error `desc` strings are bit-exact only where management software is known to match on them (02). Error strings from the core must match QEMU word for word, and CI checks this (03).
- `-d` log output matches QEMU's format only for `in_asm`, `exec` and `cpu` (22).
- The version banner is the only allowed difference in help output (22).

### Architecture and concurrency

- Lock domains are computed automatically, and ranked in bands: control 0, platform 100 to 999, device 1000 to 8999, interrupt 9000 to 9499, memory map 9500, leaf 9600 and up (03).
- IRQ lines are an atomic plus flat combining. Peer-to-peer DMA re-locks and uses a posted-write FIFO (03).
- icount, record/replay and qtest run in a serialized mode (03). In icount and replay, vCPUs keep a thread each but pass a serial token, instead of QEMU's round-robin loop (08).
- There is one timer heap per clock. The GLib context runs only when a GLib UI is linked (03).
- Release builds use `panic = "abort"`. Device and parser crates also build with `overflow-checks = true` in release (03, 19, 24). A device panic stops the VM with a diagnostic naming the device, and it does not unwind or recover.
- On panic, a per-device ring buffer of recent register accesses is printed (19).
- The reentrancy guard is a per-thread stack of active devices, with a depth limit of 16. It covers every entry point (MMIO, PIO, timers, bottom halves, ioeventfd, net queues) and returns `MEMTX_ACCESS_ERROR` like QEMU's guard (01, 19).

### Object model and QAPI

- The QAPI generator is a Rust port of `scripts/qapi`. Every build condition in the schema must map to a cargo feature, or the build fails (04).
- One introspection table serves all targets (04).
- Visitors are generated code. serde is used only at the edges, because its error messages and field ordering cannot match QEMU's (04).
- A small shim reproduces the property order that QEMU's `qom-list` returns (04).
- `object_create_early` is ported exactly (04).
- Every type must declare the `secure` flag that QEMU master is adding for 11.2. Types QEMU has not classified report false (19).

### Machines and devices

- The `Machine` trait lives in ruvm-hw-core (L2). Compat arrays are generated by `cargo xtask compat-import`, checked in, and verified in CI (11).
- A compat property that names a property missing from an existing type is a startup error in debug builds and a warning in release builds (11).
- Guest ABI is pinned by a machine fingerprint (`cargo xtask abi-fingerprint`), frozen per released machine version. Changing it needs an `abi-break` label (11).
- ruvm builds no firmware and uses QEMU's pc-bios blobs and search path (11).
- The AML builder mirrors QEMU's `aml_*` vocabulary. The rust-vmm `acpi_tables` crate is only a reference and a fuzzing oracle. ACPI golden tests run through QEMU's bios-tables-test nightly and through an in-process loader on every commit (11).
- The FDT builder produces the same bytes as libfdt (11).
- `ruvm run --direct-boot` can skip firmware for microvm with PVH under KVM. It exists on the native CLI only (11).
- Each device declares its lock domain at realize. Per-vCPU interrupt controller state (LAPIC, GIC redistributor and CPU interface, IMSIC file) is accessed from its own vCPU without a lock (12).
- ruvm registers the same KVM coalesced MMIO and PIO ranges as QEMU (12).
- NVMe `ioeventfd` defaults to false in the QEMU-compatible CLI, as in QEMU, and to true in `ruvm run` (12).
- Memory trace events use a stable region id in place of the region pointer, so QEMU and ruvm logs can be diffed (12).
- Device models are ported by hand against a seven-point checklist, with no automatic C to Rust translation. Devices are tiered A, B and C by which machines use them, and `cargo xtask device-coverage` tracks progress against the 2,501 device types recorded from QEMU 11.1 (12).

### Virtio, network, UI, passthrough

- New crates: ruvm-virtio-queue, ruvm-vhost and ruvm-vfio-user (L0, permissive), and ruvm-vhost-backends (GPL). virtio-gpu lives in ruvm-hw-display and virtio-snd in ruvm-hw-audio (13, 16).
- ruvm ships its own vhost-user backends for blk, scsi, gpu, input, vsock, rng, snd and rtc, and points users at rust-vmm's vhost-device for gpio, i2c, spi, scmi and CAN, which CI runs against the ruvm frontend (13).
- The admin virtqueue, iothread to virtqueue mapping on virtio-net, io_uring fixed buffers, busy polling and inline notification are opt-in through `x-admin-vq`, `x-iothread-vq-mapping`, `x-uring-fixed-bufs`, `x-poll-mode=busy` and `x-notify-inline`, and no machine type turns them on (13).
- libslirp stays a C library behind a thin `-sys` crate for 1.0. `ruvm run` defaults to passt with vhost-user when passt is installed, while the `qemu-*` personalities keep `user` networking as QEMU does (15).
- VNC is a Rust rewrite with byte-identical encoders. SPICE links libspice-server through FFI. `qemu-vnc`, new in QEMU 11.1, is another multi-call symlink (15).
- The GTK frontend uses GTK 4 through gtk4-rs, because gtk3-rs is archived (RUSTSEC-2024-0415). The D-Bus display uses zbus, generated from ui/dbus-display1.xml. TLS uses rustls with QEMU's PEM directory layout (15).
- The audio mixer is ported bit for bit, including the 32.32 rate converter. A sinc resampler exists as `x-resampler=sinc`, off by default (15).
- VFIO backend selection follows QEMU: the legacy container unless an `iommufd` object is linked. `ruvm run` defaults to iommufd when /dev/iommu and the device cdev exist, and expands that into an explicit `-object iommufd` (16).
- vfio and iommufd bindings are generated once from a pinned kernel header snapshot and checked in (16).
- Everything that must survive cpr-transfer registers with a `CprFdRegistry` in ruvm-migration when it is created. Passing vhost fds across CPR waits until after 1.0 (13, 16).
- The vfio-user server is native Rust in ruvm-vfio-user, not libvfio-user through FFI (16).
- CXL keeps QEMU's decoding but caches the decoded route per 256 MiB window, dropped whenever an HDM decoder commits or resets (16).

### Memory and accelerators

- Dispatch uses a sorted boundary array with a two-entry per-thread cache, with no subpage machinery (05).
- FlatViews are regenerated only for changed subtrees, and identical views are shared. `x-verify-incremental` checks this against a full rebuild (05).
- RCU uses membarrier on Linux, FlushProcessWriteBuffers on Windows, and a reader fence on macOS and the BSDs (05).
- Dirty bitmaps are per RamBlock and allocated only while logging is on (05).
- Guest memory is accessed only through `GuestPtr` and `GuestSlice`, tied to an RCU guard. Scalars use volatile access, bulk copies go through opaque copies, and there are never Rust references into guest memory (05).
- KVM requires `KVM_CAP_IMMEDIATE_EXIT` and `KVM_CAP_IRQFD`, with no signal mask fallback (06).
- The x86 instruction emulator lives in ruvm-target-x86 and is shared by HVF on x86, WHPX and MSHV (06).
- Register sync tracks validity and dirtiness per register class. `x-verify-sync` checks the subsets (06).
- SEV-SNP and TDX reset is done by `Accel::rebuild_vm()` inside the reset hold phase (06).
- `x-vcpu-affinity` and `x-vcpu-thread-context` provide pinning as extensions (06).
- MSHV, NVMM, real Xen, nitro and the non-x86, non-Arm KVM ports land in M10 (06).

### JIT

- The JIT is two crates: ruvm-jit-core (permissive) and ruvm-jit (GPL) (07).
- The tier 2 promotion threshold is 4,000 executions. Regions are capped at 64 blocks or 4,096 guest instructions, and check for exit requests at entry and on back edges (07, 08).
- Tier 1 translation must stay within 1.2x of TCG's translation time (07).
- Split W^X mappings are the default on Linux. Side tables for precise exceptions live in a separate metadata region (08).
- Helpers carry the flags `MAY_FAULT`, `READS_MEM` and `WRITES_MEM`. Fences carry a `FenceOrigin` (07).
- When a TCG plugin is loaded, tier 1 forms blocks by QEMU's rules (02).
- There is no sparc64 host backend (08).
- The M4 target is at least 1.25x QEMU TCG on SPEC CPU2017 intrate with tier 1 only, and the M9 target is 2x (21, 23).
- `-accel tcg,x-ruvm-extra-isa=on` exposes ISA features QEMU's TCG does not implement, such as AVX-512. A guest started with it cannot migrate to QEMU (09).
- CPU models are imported as data with `cargo xtask import-cpu-models`, and decoders are synced with `cargo xtask sync-decode`. ruvm writes its own decodetree files for m68k, sh4, alpha, tricore and legacy MIPS (09).
- Sail and ASL are test oracles, not code generators (02, 09).

### User mode

- linux-user and bsd-user share the new crate ruvm-user-common (10).
- User mode ships as a static musl multi-call binary with a `ruvm binfmt` subcommand (10, 24).
- A vDSO `clock_gettime` fast path is on by default and turns off under `-strace` (10).
- `x-ruvm-mmap-shared-emul=on` emulates shared file mappings at misaligned offsets with userfaultfd (10).
- The nightly job builds about 200 chroot packages per tier 1 guest. FreeBSD's kyua tests take the place of LTP for bsd-user (10).

### Block, migration, snapshots

- The new crate ruvm-crypto uses RustCrypto and rustls, with no OpenSSL, gnutls or nettle (14).
- The block graph lock is an RCU snapshot, with per-node in-flight counters for drain (14).
- System zlib and zstd are linked by default so that compressed output is byte-identical to QEMU's. Pure-Rust versions sit behind a feature (14).
- io_uring is the internal default for file I/O on Linux, but QMP still reports QEMU's default `aio` value (14).
- There is no gluster driver, because QEMU 11.1 removed it (14).
- The FUSE export talks to `/dev/fuse` directly, with no libfuse (14).
- Device state uses one table-driven codec, not a serializer generated per struct. Incoming migration streams are untrusted input, and the loader is fuzzed (17).
- Fast restore uses QEMU's mapped-ram file format. Direct mapping and working-set recording are ruvm options (`x-restore-mode=map`, `x-record-working-set`), and lazy restore is off by default (17, 21).
- There is an optional Firecracker-compatible userfaultfd page handler protocol, off by default (17).
- cpr-transfer from QEMU to ruvm on the same host is an M5 deliverable (17).
- COLO comes after M5 and supports only the negotiation used by QEMU 10.2 and later (17).

### Security

- Backend processes that ruvm ships (vhost-user, vfio-user) are treated as untrusted, which is stricter than QEMU (19).
- Device crates forbid `unsafe` entirely, except ruvm-hw-vfio (19, 24).
- Supply chain uses cargo vet, reproducible builds and Sigstore signing (19).

### Testing and performance

- QEMU's test suites run unmodified from a copy pinned at v11.1.0. Each suite has a skip list that may only shrink (22).
- A test-only crate links QEMU's `fpu/softfloat.c` as an oracle and is never shipped (22).
- A `sim` build feature provides deterministic whole-VM testing (22).
- A target is met only when the lower bound of the 95% bootstrap confidence interval clears it. For "equal or better" targets, the rule is that the upper bound is at least 1.0 and the point estimate at least 0.98 (21).
- qemu-img targets are parity for uncompressed convert, at least 1.2x for compressed convert, and at least 1.5x for check (21).
- Release builds keep frame pointers. The cost is measured again at M4 (21).

## Claims to verify before 1.0

The documents mark some numbers as design targets or estimates rather than measurements. They are listed here so nobody quotes them as results:

- The unsafe block budgets per crate, the reentrancy depth limit of 16, and the 1% cost budget for overflow checks (19).
- The share of the 2x JIT gain that each source contributes, and the cycle counts of the TLB fast path on each host (08).
- The CI pool sizes and the exit throughput and memory overhead targets (21, 22).
- The dispatch structure benchmarks. The size comparison in document 05 was measured, but its timings were not published, because the benchmark host was overloaded (05).
- Whether HVF on Arm supports migration in QEMU 11.1, and in which release WHPX on Arm was merged (06).
