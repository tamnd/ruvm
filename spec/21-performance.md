# 21. Performance: targets, method, and regression gates

This document turns the headline performance goals in the canon into claims that a skeptical reader can reproduce and that a CI system can fail. Every target here has five parts: a workload, an exact configuration, a metric, a statistical rule for deciding whether the target is met, and the design techniques (with the document that specifies them) that are supposed to deliver the gain. If a technique does not show up in the numbers, the technique is wrong or the measurement is wrong, and we find out which before we ship. Correctness always wins over speed: a benchmark result from a build that fails any conformance gate in document 22 is discarded, not reported.

## What counts as a performance claim

A ruvm performance claim is a ratio against QEMU 11.1.0 on the same host, kernel, guest image, and command line, with both binaries built under the rules in the next section. Absolute numbers are reported but are not the claim, except for the canon's two absolute KVM latency targets (15 ms to first guest instruction, 110 ms Linux boot to init for a microvm config).

A claim is "met" only when the lower bound of the 95% confidence interval of the ratio clears the target. A point estimate that clears the target while the interval straddles it is "not yet met". This is deliberately harsh: noisy benchmarks cannot produce a win by luck, and it pushes us to reduce noise rather than rerun until the numbers look good.

We publish every run, including the ones that go against us, with raw per-iteration data, the host description from `ruvm-bench hostinfo`, and both build manifests.

## The QEMU 11.1 baseline and build rules

Distribution QEMU packages differ in compiler, hardening flags, and module sets, so the baseline is built by us from the QEMU 11.1.0 release tarball under these rules, applied symmetrically to ruvm.

| Rule | QEMU 11.1 baseline | ruvm |
|---|---|---|
| Compiler | clang from the LLVM release that matches the rustc LLVM major version used for ruvm | rustc stable pinned in rust-toolchain.toml |
| Optimization | `--enable-lto`, `-O2` (meson buildtype=release) | `opt-level=2`, `lto="fat"`, `codegen-units=1` |
| Target CPU | `-march=x86-64-v3` on x86-64 hosts, `-march=armv8.2-a` on aarch64 hosts | `-C target-cpu` equivalent (`x86-64-v3`, `generic` plus `+v8.2a`) |
| Allocator | glibc malloc | glibc malloc via the system allocator (no jemalloc or mimalloc in the headline numbers) |
| Hardening | `-fstack-protector-strong`, `_FORTIFY_SOURCE=3`, CFI off | Rust defaults, overflow checks off in release, no extra hardening |
| Debug info | `-g` split out, not stripped from the measured binary's symbol table | `debug=1` split out |
| Coroutine backend | ucontext (the default on Linux) | not applicable |
| Modules | `--disable-modules`, static device set equivalent to ruvm's default feature set | default features |
| Tracing | `--enable-trace-backends=nop` for headline runs, `log` for profiling runs | trace compiled in, disabled at runtime (see profiling section) |

The allocator is pinned because a ruvm win from mimalloc would be a win QEMU could get with one configure flag; an informational mimalloc run of both binaries shows the size of that effect. The optimization level rule is symmetric: if ruvm moves to `opt-level=3`, the baseline moves to `-O3`. An informational run against the Debian 13 distribution QEMU package shows what users actually get, and never gates.

## Host hardware classes

Claims are made per host class. A class is a CPU microarchitecture family and memory configuration, not a SKU, so a dead machine can be replaced without invalidating history. Each class has at least two identical machines.

| Class | CPU family | Sockets, memory | Storage, network | Used for |
|---|---|---|---|---|
| X-AMD | AMD EPYC 9005 (Zen 5) | 1 socket, 12 channels DDR5, 384 GiB | 2x PCIe Gen5 NVMe, 100 GbE dual port | KVM, TCG x86 host, virtio, migration source |
| X-INTEL | Intel Xeon 6 P-core (Granite Rapids) | 1 socket, DDR5, 256 GiB | 2x PCIe Gen5 NVMe, 100 GbE | KVM, TCG x86 host, TDX when available |
| A-NEO | Arm Neoverse V2 (for example AWS Graviton4 bare metal or NVIDIA Grace) | 1 socket, 256 GiB | NVMe, 100 GbE | KVM arm64, TCG aarch64 host |
| A-APPLE | Apple M4 Pro Mac mini | 64 GiB unified | internal SSD, 10 GbE | HVF, TCG aarch64 host on macOS, MAP_JIT costs |
| MIG-PAIR | Two X-AMD machines, direct 100 GbE link | as X-AMD | as X-AMD | migration and remote vhost-user |

Machines are dedicated to benchmarking. Cloud bare-metal instances are acceptable only if we control frequency and SMT; otherwise they are informational and do not gate.

## Host software configuration

Host OS is Debian 13 with Linux 6.18 LTS, the current longterm kernel ([kernel.org releases](https://www.kernel.org/category/releases.html)). Linux 6.12 LTS runs the weekly suite so we notice results that depend on newer kernel features. macOS hosts run the current release.

Noise control on Linux hosts is applied by `ruvm-bench host-prep`, which refuses to start if any check fails:

- CPU frequency: turbo and boost disabled (`/sys/devices/system/cpu/intel_pstate/no_turbo=1` on Intel, `/sys/devices/system/cpu/cpufreq/boost=0` on AMD and Arm where exposed), governor `performance`, and minimum frequency set equal to maximum nominal.
- SMT disabled (`/sys/devices/system/cpu/smt/control=off`) for all latency benchmarks. Throughput benchmarks run once with SMT off (the gate) and once with SMT on (informational).
- Isolation: kernel command line `isolcpus=managed_irq,domain,<bench cpus> nohz_full=<bench cpus> rcu_nocbs=<bench cpus>`, IRQ affinity of all devices except the benchmark NIC and NVMe moved to housekeeping CPUs, `irqbalance` stopped.
- NUMA: benchmark CPUs, guest RAM (`memory-backend-memfd` with `host-nodes` and `policy=bind`), NVMe, and NIC on one node.
- THP `always` for TCG and boot suites, `madvise` for the memory suite; page cache dropped and memory compacted before each boot, memory, or qemu-img iteration.
- C-states deeper than C1 disabled via `/dev/cpu_dma_latency` during latency runs.
- ASLR stays on and the environment block size is randomized per iteration, so that layout effects average out across fresh processes instead of biasing one binary, following Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009).

vCPU and iothread pinning uses thread ids from `query-cpus-fast` and `query-iothreads` for both VMMs, never a ruvm-only option.

## Guest images

Guest images are built reproducibly (`xtask bench-images`) and published with SHA-256 hashes.

| Image | Contents | Used by |
|---|---|---|
| `bench-debian13-x86_64.qcow2` | Debian 13 cloud image, systemd, GCC 14, SPEC harness prerequisites, fio 3.x, iperf3, netperf, CoreMark | TCG system, virtio, migration |
| `bench-debian13-arm64.qcow2` | same for arm64 | TCG system (aarch64 guest), KVM on A-NEO |
| `bench-min-vmlinux-<arch>` | Linux 6.18 with a Firecracker-style minimal config, uncompressed, PVH note on x86 | microvm boot latency |
| `bench-init-static` | a static init that writes boot markers to I/O port 0x3f0 (x86) or a fixed MMIO word (arm64) and then powers off | boot latency |
| `bench-rootfs-sysroot-<arch>` | Debian 13 chroot with a native toolchain for linux-user builds | linux-user |
| `bench-ovmf` and `bench-seabios` | edk2 and SeaBIOS built from the exact versions QEMU 11.1 ships in pc-bios | q35 firmware boot |

SPEC CPU2017 binaries are built once per guest architecture with GCC 14 at `-O2` (x86-64: `-march=x86-64-v3`; aarch64: `-march=armv8.2-a`), stored in the private results store, and never rebuilt between runs. SPEC CPU2017 is licensed and cannot be redistributed; our published numbers are estimates under SPEC's fair use rules and are labelled "SPEC CPU2017 estimated" since they are not reviewed submissions.

## Statistical method

We follow the approach of Georges, Buytaert, and Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007) and Kalibera and Jones, "Rigorous Benchmarking in Reasonable Time" (ISMM 2013): repeat at the level where the variance lives, and report confidence intervals rather than means.

- The unit of repetition is a fresh VMM process and guest boot, because code cache, allocator state, and host page placement vary per process.
- Default repetition counts: 10 fresh processes for long benchmarks (SPEC, kernel builds, migration), 30 for medium benchmarks (fio, iperf, qemu-img), 200 for boot and startup latency, and 10 million samples per process across 10 processes for exit latency.
- For JIT benchmarks the first in-guest iteration is excluded from steady state and reported separately as warmup.
- QEMU and ruvm runs are interleaved in ABBA order (ruvm, QEMU, QEMU, ruvm, ...) so that slow drift (thermal, background kernel work) affects both equally.
- Ratios of central tendency are computed with a percentile bootstrap (10,000 resamples) over the per-process values, giving a 95% interval for the ratio of means (throughput) or ratio of medians (latency). For suites (SPEC, the fio matrix) the headline is the geometric mean of per-benchmark ratios, and its interval comes from bootstrapping each benchmark independently and recomputing the geometric mean.
- Tail latency (p99, p99.9) is reported per process and then summarized by the median across processes with its bootstrap interval.
- Outliers are never dropped. Invalid iterations (guest crash, host check failure) are discarded with a logged reason; more than 5% discards invalidates the run.
- A coefficient of variation above 3% on any gating metric flags the host for investigation.

For trends we run E-divisive means change point detection over the nightly series, as described by Daly et al. (ICPE 2020, MongoDB). A change point opens an issue with the commit range; it does not fail a build.

## Area 1: JIT (TCG replacement)

### Targets

| Metric | Configuration | Target at M4 (tier 1 only) | Target at M9 (tier 1 plus tier 2) |
|---|---|---|---|
| SPEC CPU2017 intrate, 1 copy, geometric mean of time ratio | x86-64 guest on X-AMD, `-accel tcg`, `-cpu max`, 4 vCPU, 8 GiB | at least 1.25x QEMU | at least 2.0x QEMU |
| same | aarch64 guest (`-M virt -cpu neoverse-n1`) on X-AMD | at least 1.25x | at least 2.0x |
| same | x86-64 guest on A-NEO | at least 1.2x | at least 2.0x |
| linux-user SPEC CPU2017 intrate, geometric mean | `qemu-x86_64` on A-NEO, `qemu-aarch64` on X-AMD | at least 1.3x | at least 2.0x |
| CoreMark, iterations per second | same three guest and host pairs, system mode, 1 thread and N threads | at least 1.3x | at least 2.5x |
| Boot to shell | Debian 13 aarch64 on X-AMD, `-M virt` with edk2, until `login:` on the serial console | at most 0.85x QEMU wall time | at most 0.7x |
| Linux kernel build in guest | `make -j8 defconfig vmlinux` of Linux 6.18, 8 vCPU, x86-64 guest on X-AMD | at least 1.2x | at least 1.8x |
| Kernel build under linux-user | arm64 native GCC in a chroot via binfmt_misc on X-AMD, `make -j16 defconfig Image` | at least 1.3x | at least 2.0x |
| MTTCG scaling | kernel build, 1 to 32 vCPU, parallel efficiency at 32 | at least QEMU's efficiency | at least QEMU's efficiency plus 10 points |
| Translation cost | guest instructions translated per host second, tier 1, measured on boot-to-shell | at least 1.0x QEMU | at least 1.0x |

The M4 and M9 split is a decision of this document: the canon's 2x is a tier 2 goal. Tier 1 compiles about as fast as TCG, so its gain comes from cheaper softmmu fast paths, fewer helper calls, and lazy flags. Published results bound what is available. HQEMU ([CGO 2012](https://dl.acm.org/doi/10.1145/2259016.2259030)) reported 2.4x (integer) and 4x (floating point) over QEMU on SPEC CPU2006 for x86 to x86-64 with an LLVM trace optimizer; Instrew ([VEE 2021](https://dl.acm.org/doi/10.1145/3453933.3454022)) reported 53% emulation overhead against QEMU's 648% on SPEC CPU2017 x86-64 to x86-64 in user mode; Risotto ([ASPLOS 2023](https://dl.acm.org/doi/10.1145/3567955.3567962)) gained 6.7% on average and up to 19.7% over QEMU from verified fence placement alone for x86 on Arm; Arancini ([ASPLOS 2026](https://dl.acm.org/doi/10.1145/3779212.3790127)) reports up to 5x over QEMU based translators on multithreaded workloads. Those are mostly user mode results with expensive LLVM compilation; system mode pays the softmmu TLB on every access, which is why the target is 2x.

### Workload definitions

SPEC CPU2017 intrate means the ten intrate benchmarks (500.perlbench_r through 557.xz_r) at `--size ref`, 1 copy, 3 iterations with the median reported, run with `runcpu` inside a freshly booted guest. A ref pass under TCG takes many hours, so it runs weekly; nightly uses `--size train`, which is not claimed. An fprate subset (503.bwaves_r, 519.lbm_r, 538.imagick_r, 544.nab_r) runs as informational to expose softfloat regressions.

CoreMark is EEMBC CoreMark built with GCC 14 `-O2`, iteration count set so one run takes about 30 seconds on QEMU; the multithread variant uses the `MULTITHREAD` build with pthreads and one thread per vCPU.

Boot to shell is timed from `execve` to `login:` on the serial chardev (a pipe timestamped with `CLOCK_MONOTONIC`). It covers firmware, GRUB, and systemd, and is the benchmark most sensitive to translation cost.

The in-guest kernel build uses a Linux 6.18 tree on virtio-blk (`cache=none,aio=io_uring`), `ccache` off. The linux-user build runs an arm64 Debian 13 chroot with `binfmt_misc` pointing at `qemu-aarch64` (either binary), so make, sh, cc1, as, and ld are all emulated; it stresses translation of thousands of short processes and the `execve` and `mmap` paths from document 10.

### Techniques that deliver the gain

- Tier 1 local register allocation with the softmmu TLB fast path inlined (document 08). QEMU's `accel/tcg/cputlb.c` fast path is already short; the gain is avoiding guest state spills at every block boundary.
- Lazy flags with liveness (document 07): x86 guests compute EFLAGS lazily in QEMU too (`CC_OP` in `target/i386/tcg`), but tier 2 removes the flag computation entirely when liveness shows no consumer across the trace. Yen et al., "ARMing x86 Games" ([MobiSys 2025](https://doi.org/10.1145/3711875.3729163)), report up to 18% speedup on compute tasks from replacing software flag emulation with host flags where data flow analysis proves it safe; tier 2 applies the same idea for x86 guests on Arm hosts.
- Block chaining plus a per-vCPU jump cache.
- Tier 2 region formation from hot counters, promoting guest registers across blocks and forwarding loads and stores within the guest memory model rules (document 07).
- Fence placement using the verified mapping schemes from Risotto and Arancini when the guest is stronger than the host (x86 on Arm), instead of QEMU's conservative per-access barriers (document 08).
- MTTCG with lock-free TB lookup and no global lock on lookup or exception paths (documents 03 and 08).
- Host FPU fast paths where the result is provably bit-identical, extending QEMU's hardfloat approach in `fpu/softfloat.c` (document 08).
- linux-user: syscall dispatch without a global mmap lock on the fast path (document 10).

### Regression gates

- Per pull request touching ruvm-jit or any ruvm-target crate: CoreMark (1 thread, 3 guest and host pairs) and a 5 minute "SPEC-like" open source proxy set (the `tests/tcg` multiarch benchmarks plus sha512, zstd compression of a fixed corpus, and a Lua interpreter running a fixed script). Fail if the geometric mean of time ratios against the main branch baseline regresses by more than 2% with the lower bound of the 99% interval also worse than 0%. Warn at 1%.
- Nightly: SPEC CPU2017 intrate train, boot to shell, in-guest kernel build. Fail the nightly (and block the next release candidate) if any headline ratio against QEMU drops below its milestone target, or if any single SPEC benchmark regresses by more than 5% against the previous nightly.
- Weekly: SPEC CPU2017 intrate ref, linux-user kernel build, MTTCG scaling curve. Results feed the published dashboard.
- Translation cost gate: tier 1 guest instructions translated per host second must stay at or above QEMU's rate on boot to shell, so tier 1 never becomes a slow optimizing compiler.

## Area 2: startup and boot latency

### Configurations

| Config | Machine | Firmware | Kernel | Devices |
|---|---|---|---|---|
| U1 microvm direct | `-M microvm,x-option-roms=off,isa-serial=off,rtc=off` | none (PVH entry) | `bench-min-vmlinux-x86_64` | 1 vCPU, 128 MiB, virtio-mmio blk (rootfs), no serial, `-nodefaults -no-user-config` |
| U2 microvm qboot | as U1 with `-bios qboot.rom` | qboot | same | same |
| U3 q35 direct | `-M q35` | SeaBIOS with `-kernel` fw_cfg boot | `bench-min-vmlinux-x86_64` built with PCI and ACPI | 1 vCPU, 512 MiB, virtio-blk-pci |
| U4 q35 OVMF | `-M q35` | edk2 OVMF with `-kernel` | same as U3 | same |
| U5 arm virt direct | `-M virt,gic-version=3` on A-NEO | none (`-kernel` Image) | arm64 minimal config | 1 vCPU, 128 MiB, virtio-mmio |
| U6 arm virt UEFI | as U5 with edk2 ArmVirtQemu | edk2 | Debian 13 kernel | virtio-blk-pci |

All run under KVM on X-AMD and X-INTEL (x86) and A-NEO (arm64), with the same kernel for both VMMs. U1 and U5 carry the canon's absolute targets; the others are relative.

### Measurement

Phases come from host kernel tracepoints so neither VMM is instrumented. The harness runs `perf record -e kvm:kvm_entry -e kvm:kvm_pio -e kvm:kvm_mmio -e sched:sched_process_exec` system-wide on the benchmark CPUs. The guest kernel and init write markers to an I/O port (x86) or a reserved MMIO word (arm64) that no device claims, which produces a `kvm_pio` or `kvm_mmio` exit with a recognizable value. This is the method used by Stefano Garzarella's [qemu-boot-time](https://github.com/stefano-garzarella/qemu-boot-time) scripts, which measured QEMU 4.0 q35 with qboot and a PVH vmlinux at about 52 ms to end of QEMU init, 57 ms to kernel start, and 134 ms to user space ([blog post](https://stefano-garzarella.github.io/posts/2019-08-23-qemu-linux-kernel-pvh/)).

Phases reported:

- T0 `execve` of the VMM binary.
- T1 first `kvm:kvm_entry` for the VM (first guest instruction). Canon target: T1 minus T0 at most 15 ms on U1.
- T2 kernel `start_kernel` marker.
- T3 init marker (first instruction of `/sbin/init`). Canon target: T3 minus T0 at most 110 ms on U1.
- T4 VM exit (init writes the power off marker and calls `reboot(RB_POWER_OFF)`), plus T5 process exit, which measures teardown. Teardown matters for density and serverless workloads and is often ignored.

For context, the [Firecracker specification](https://github.com/firecracker-microvm/firecracker/blob/main/SPECIFICATION.md) states VMM startup to API socket within 8 CPU ms (wall clock typically about 12 ms, spread 6 to 60 ms) and advertises under 125 ms to user space. We report Firecracker on the same host as context only.

### Targets

| Config | Metric | Target |
|---|---|---|
| U1 | T1 minus T0, median | at most 15 ms (canon), and at most 0.5x QEMU |
| U1 | T1 minus T0, p99 | at most 25 ms |
| U1 | T3 minus T0, median | at most 110 ms (canon) |
| U2 | T3 minus T0 | at most 0.8x QEMU |
| U3, U4 | T3 minus T0 | at most 0.9x QEMU; VMM attributable time (total minus guest time between markers where the vCPU was in guest mode) at most 0.5x QEMU |
| U5 | T1 minus T0, T3 minus T0 | at most 15 ms and at most 110 ms on A-NEO |
| All | T5 minus T4 | at most 0.5x QEMU |
| All | CPU time (user plus sys) of the VMM process from T0 to T3, excluding vCPU guest time | at most 0.5x QEMU |

### Techniques

- Static type registration via link-time distributed slices (document 04) instead of thousands of `type_init` constructors.
- Compat props for versioned machines resolved at build time into static tables instead of string property round trips (document 11).
- Guest RAM on `memfd_create` without pre-touch, KVM slots registered in one batch (document 06).
- q35 AML cached per machine version and configuration hash (document 11); bios-tables-test (document 22) runs with the cache cold and warm.
- Kernel and initrd served over fw_cfg DMA from a shared file mapping rather than a heap copy.
- The boot vCPU starts as soon as its state is set; secondary vCPU threads are created in parallel.
- Dynamic linking cost: on the microvm feature profile the `ruvm` binary links only libc, libm, and libgcc_s at startup; GTK, SDL, spice, and audio libraries are loaded with `dlopen` only when a UI or audio backend is configured (document 15). Relocation and constructor work for dozens of shared libraries is a large part of a distro QEMU's time before `main`, and it is why the canon target is framed on a microvm-class config.
- Teardown: after the final QMP event is flushed the process exits without unmapping guest RAM or destroying devices one by one, and leaves the address space teardown to the kernel.

### Regression gates

Per pull request touching ruvm-system, ruvm-cli, ruvm-machine-x86, ruvm-firmware, or ruvm-accel-kvm: 200 runs of U1 and 50 runs of U4 on X-AMD. Fail if U1 median T1 minus T0 exceeds 15 ms or T3 minus T0 exceeds 110 ms, or if either regresses by more than 3% (99% interval) against main. Nightly runs every configuration on every host class.

## Area 3: memory overhead

### Definitions

Guest RAM pages are not overhead. We define overhead as the VMM process's resident memory minus the resident pages of the guest RAM mappings, measured from `/proc/<pid>/smaps_rollup` and per mapping `/proc/<pid>/smaps` (guest RAM is always a named memfd mapping, `memory-backend-memfd`, in both VMMs for this suite so that it is identifiable). We report:

- Overhead RSS and overhead PSS (PSS divides shared library pages among processes, which matters for density).
- Page table memory (`VmPTE` from `/proc/<pid>/status`), which grows with guest RAM size when RAM is mapped with 4 KiB pages and is not visible in RSS.
- Per-vCPU overhead: the slope of overhead RSS against vCPU count from 1 to 64, fitted by least squares over 5 points (1, 8, 16, 32, 64).
- Anonymous heap after 10 minutes of guest idle, to catch slow leaks.
- For TCG: code cache resident size and TLB memory per vCPU, reported separately, with `tb-size` set identically on both VMMs.

### Targets

| Config | Metric | Target |
|---|---|---|
| U1 microvm (1 vCPU, 128 MiB), KVM | overhead RSS after boot to init | at most 0.6x QEMU (canon) |
| q35, 4 vCPU, 4 GiB, virtio-blk, virtio-net, virtio-rng, virtio-balloon, KVM | overhead RSS | at most 0.6x QEMU |
| q35, 1 to 64 vCPU | per-vCPU slope | at most 0.5x QEMU |
| q35, 256 GiB guest RAM fully touched, 4 KiB pages | `VmPTE` | at most 1.0x QEMU (parity; the page tables are the kernel's) |
| TCG, x86-64 guest, 4 vCPU | overhead RSS excluding code cache | at most 0.7x QEMU |

### Techniques

- vCPU threads do not allocate on the hot path (document 03), so glibc does not grow an arena per vCPU thread.
- No coroutine stacks: ruvm-aio uses completion based state machines, while QEMU allocates 1 MiB coroutine stacks from a pool.
- Static property tables from `#[derive(Device)]` instead of per-instance property hash tables (document 04).
- FlatViews shared between address spaces with identical views (document 05), and dirty bitmaps allocated only when logging starts.

### Regression gates

Per pull request: U1 and the q35 4 vCPU config on X-AMD, 30 runs each. Fail if overhead RSS rises by more than 256 KiB or 2% (whichever is larger) against main, or if the ratio against QEMU crosses 0.6. Nightly: the full vCPU sweep and the 10 minute idle leak check (fail if anonymous heap grows by more than 64 KiB between minute 1 and minute 10).

## Area 4: MMIO and PIO exit latency

### Workload

We use kvm-unit-tests `x86/vmexit.c` unchanged. `inl_from_qemu` reads unassigned port 0x1234 and so measures the userspace exit round trip, `inl_from_kernel` reads a kernel handled port (the difference is roughly the userspace exit cost, as Paolo Bonzini's commit introducing them explains), and `inl_from_pmtimer` adds device model cost. `-device pci-testdev` provides MMIO and PIO BARs for vmexit's PCI cases. On arm64 we use `arm/micro-bench.c` and its MMIO read cases.

We add one test because vmexit only covers trivial handlers: a guest loop reading virtio-pci common config (`queue_size`) and writing the notify register with ioeventfd disabled, which exercises the full dispatch path real guests hit.

### Metrics and targets

Cycles per exit (TSC based, as vmexit reports) summarized as median and p99 per process over 10 processes on X-AMD, X-INTEL, and A-NEO.

| Case | Target |
|---|---|
| `inl_from_qemu` (userspace PIO exit, no device) | median at most 0.85x QEMU, p99 at most 1.0x QEMU |
| `inl_from_pmtimer` | median at most 0.8x QEMU |
| pci-testdev MMIO without ioeventfd | median at most 0.85x QEMU |
| virtio common config read | median at most 0.8x QEMU, p99 at most 0.8x QEMU |
| same with 16 vCPUs concurrently hammering different devices | aggregate exits per second at least 4x QEMU |

The last row is where the absence of the BQL shows. In QEMU, `kvm_cpu_exec` dispatches MMIO without the BQL, but `prepare_mmio_access` in `system/physmem.c` takes the BQL for every region that has not called `memory_region_clear_global_locking`, which is almost all devices, so 16 vCPUs exiting to different devices serialize. In ruvm each device's lock domain is independent (document 03), so the only shared state is the RCU protected FlatView.

### Techniques

- RCU FlatView lookup with a per-vCPU last-hit cache in front of the sorted boundary array dispatch (document 05).
- Per-device lock domains instead of a global lock (document 03), and lock-free notify paths for virtio (document 13).
- `KVM_RUN` loop in Rust with the `kvm_run` shared page accessed directly, no intermediate exit structure copy (document 06).

### Regression gates

Per pull request touching ruvm-mem, ruvm-accel-kvm, ruvm-hw-core, or ruvm-hw-virtio: vmexit `inl_from_qemu` and the virtio config read, 10 processes of 1 million samples. Fail if median rises by more than 3% or 100 cycles (whichever is larger) against main, or crosses the QEMU ratio target.

## Area 5: virtio-blk, virtio-net, and virtio-fs

The canon target is parity or better against QEMU's best configuration (iothreads, vhost), not its default.

### virtio-blk

QEMU baseline config: `-object iothread,id=io0..ioN`, `-device virtio-blk-pci,iothread-vq-mapping=...,num-queues=N`, `-blockdev driver=host_device,cache.direct=on,aio=io_uring`, guest RAM on a memfd. ruvm uses the same command line. Two backends: a real PCIe Gen5 NVMe namespace (whole device, raw), and the `null-co` driver with `read-zeroes=on` to measure VMM overhead without a disk in the way.

fio runs inside the guest against the raw virtio disk with `direct=1`, `ioengine=io_uring`, `time_based=1`, `runtime=60`, `ramp_time=10`, and `--output-format=json+` so that latency histograms are captured. Job matrix:

| Job | bs | rw | iodepth | numjobs | Primary metric |
|---|---|---|---|---|---|
| lat-rr | 4k | randread | 1 | 1 | completion latency p50, p99, p99.9 |
| lat-rw | 4k | randwrite | 1 | 1 | same |
| iops-rr | 4k | randread | 32 | 4 (one per queue) | IOPS |
| iops-rw | 4k | randwrite | 32 | 4 | IOPS |
| bw-sr | 128k | read | 16 | 1 | MiB/s |
| bw-sw | 128k | write | 16 | 1 | MiB/s |
| sync-w | 4k | randwrite with `fsync=1` | 1 | 1 | ops/s, exercises flush path |
| mixed | 4k | randrw 70/30 | 64 | 8 | IOPS and p99 |

Guests have 4, 8, and 16 vCPUs with num-queues matching and 1, 2, and 4 iothreads. We report VMM host CPU time per I/O, because matching IOPS by burning twice the CPU is not parity.

Targets: every fio job with the null-co backend at least 1.0x QEMU IOPS or bandwidth and at most 1.0x QEMU p99 latency; with NVMe the same; and host CPU per I/O at most 0.8x QEMU on iops-rr. Parity targets ("at least 1.0x") are met when the upper bound of the 95% interval is at least 1.0 and the point estimate is at least 0.98, meaning ruvm is not measurably slower; this is the one relaxation of the lower-bound rule.

### virtio-net

QEMU baseline: `-netdev tap,vhost=on,queues=N` with `-device virtio-net-pci,mq=on,vectors=2N+2`, the tap on a host bridge with no physical port so the path is guest to host kernel. A second topology runs guest to guest across MIG-PAIR over 100 GbE to catch effects that only appear with a real NIC (interrupt coalescing, GRO). Client and server are pinned, runs are long enough for TCP steady state, and latency comes from request/response tests, not ping.

| Test | Tool and options | Metric |
|---|---|---|
| TCP stream, 1 flow, guest to host | `iperf3 -c <host> -t 60 -O 5` | Gbit/s |
| TCP stream, 8 flows | `iperf3 -P 8` | Gbit/s aggregate |
| TCP_RR 1 byte | `netperf -t TCP_RR -l 60 -- -r 1,1 -o P50_LATENCY,P99_LATENCY,THROUGHPUT` | transactions/s, p50, p99 |
| UDP_RR 1 byte | `netperf -t UDP_RR` same selectors | same |
| TCP_CRR | `netperf -t TCP_CRR` | connections/s |
| UDP 64 byte packet rate | `iperf3 -u -l 64 -b 0` | packets/s |

Backends: tap with vhost-net (the canon comparison), tap without vhost (userspace datapath, where ruvm's own virtio-net code is on the path), and passt (document 15). vhost-user-net is covered in the next area.

Targets: with vhost-net, parity on all throughput tests and TCP_RR p99 at most 1.0x QEMU. Without vhost, at least 1.2x QEMU throughput and at most 0.85x TCP_RR p50, where ruvm's io_uring tap backend and lock-free virtqueue processing (document 13) are on the path. Host CPU per gigabit at most 0.9x QEMU in all modes.

### virtio-fs

Both VMMs use the external virtiofsd over vhost-user, so the VMM only does setup and notification forwarding. Workloads: fio `randread` 4k QD1 and QD32 on a 4 GiB file in the shared directory, a metadata test (untar the Linux source tree, then `find . -type f | wc -l`, then `rm -rf`), and `git status` on a large repository. Target: parity.

### Techniques

- ruvm-aio completion based reactors on io_uring with registered buffers and fixed files for the block and tap backends (document 03 and 14); one reactor per iothread with no cross-thread wakeups on the completion path.
- virtio queue processing with batching of used ring updates and event index suppression honored (document 13), and notifications via ioeventfd and irqfd so vCPUs do not exit to userspace on the data path.
- Multi-queue with iothread to virtqueue mapping compatible with QEMU's `iothread-vq-mapping` property.
- Block graph fast path: when the graph is a single format-less node with no filters, requests bypass the generic graph machinery (document 14), with no per-request coroutine or allocation.

### Regression gates

Per pull request touching ruvm-hw-virtio, ruvm-block, ruvm-net, or ruvm-aio: fio lat-rr and iops-rr on null-co, and netperf TCP_RR plus iperf3 1 flow on tap without vhost, 10 runs each on X-AMD. Fail on more than 3% throughput loss or 5% p99 latency increase (99% interval) against main. Nightly: the full matrix on X-AMD, X-INTEL, and A-NEO, and the NVMe backed runs.

## Area 6: vhost-user

With vhost-user the data path lives in another process, so the VMM shows up in control plane latency (`VHOST_USER_SET_MEM_TABLE`, device start and stop), memory hotplug, and notification forwarding.

Configurations: vhost-user-blk served by `qemu-storage-daemon` 11.1 and by `ruvm-storage-daemon`, in all four frontend and backend pairings (QEMU with QSD, QEMU with ruvm-storage-daemon, ruvm with QSD, ruvm with ruvm-storage-daemon); vhost-user-net served by DPDK testpmd in io forwarding mode; vhost-user-fs via virtiofsd.

Metrics and targets:

- Data path: same fio and netperf matrix as area 5. Target parity when the backend is the same binary, and ruvm-storage-daemon at least 1.0x QSD IOPS on iops-rr with at most 0.8x CPU per I/O.
- Device start latency: time from `device_add` of a vhost-user-blk device to the guest seeing it ready. Target at most 0.7x QEMU.
- Memory table update latency during memory hotplug of a 1 GiB DIMM with 4 vhost-user devices attached: target at most 0.7x QEMU (ruvm sends the updates to all backends concurrently rather than sequentially).

Gate: nightly. Cross-pairing failures are compatibility bugs tracked in document 22.

## Area 7: live migration

### Workloads and configuration

Source and destination are the two MIG-PAIR machines connected back to back. Guests: q35, KVM, 8 vCPU, 16 GiB and 64 GiB. Workloads:

- W-idle: booted Debian 13 with nothing running.
- W-stress: the guestperf `stress` program from QEMU's `tests/migration-stress/guestperf` (one thread per vCPU xoring memory with random data), configured to dirty 4 GiB at a controlled rate of 1 GiB/s, which makes pre-copy convergence depend on bandwidth.
- W-kv: Redis with a 12 GiB dataset under a memtier or YCSB workload at 50% of the guest's saturation throughput.
- W-io: fio randwrite on virtio-blk with shared storage (NFS over the second port) so block job interaction is covered.

Modes: pre-copy with default parameters; pre-copy with multifd at 4 and 8 channels; post-copy after one pre-copy pass; pre-copy with `max-bandwidth` capped at 10 Gbit/s to emulate a constrained link. `downtime-limit` is 300 ms for all runs. The same guestperf parameters drive both VMMs; ruvm runs guestperf unmodified since it speaks QMP.

### Metrics

- Downtime from `query-migrate` and as observed externally: a host process sends UDP requests to the guest at 1 kHz over a second port, and the longest gap in answers is the observed downtime. Guest clocks cannot be used because kvmclock is adjusted across migration.
- Total migration time (`total-time`) and setup time (`setup-time`).
- Bandwidth efficiency: bytes on the wire (`ram.transferred` plus device state) divided by guest RAM size, and achieved throughput as a fraction of link capacity.
- Redis throughput and p99 during migration, and VMM CPU time on both sides.

### Targets

| Metric | Target |
|---|---|
| Downtime (observed), each workload and mode | at most 1.0x QEMU median, at most 1.0x QEMU p99 across runs (canon) |
| Total time, W-stress, multifd 8 | at most 0.9x QEMU |
| Bytes transferred / RAM, W-idle | at most 1.0x QEMU (zero page detection must be at least as good) |
| Link utilization, multifd 8 at 100 GbE | at least 1.1x QEMU |
| Redis p99 during migration | at most 1.0x QEMU |
| Interop: QEMU source to ruvm destination and the reverse | same downtime targets measured against QEMU to QEMU |

### Techniques

- KVM dirty ring where available, harvested per vCPU thread (document 17).
- Multifd threads doing zero page detection and sending from guest memory with `MSG_ZEROCOPY` where supported.
- Device state saved in parallel per lock domain in the stop phase; QEMU saves it sequentially under the BQL, which adds downtime with many devices.

### Regression gates

Nightly on MIG-PAIR: W-idle and W-stress, pre-copy and multifd 8, 10 runs each, both homogeneous and interop pairs. Fail if observed downtime median exceeds QEMU's or regresses more than 10% against the previous nightly. Weekly: all workloads and modes.

## Area 8: snapshot and restore

Workloads: savevm and loadvm of internal qcow2 snapshots (`savevm`/`loadvm` HMP and `snapshot-save`/`snapshot-load` QMP jobs), and file migration to a local file with the `mapped-ram` capability followed by `-incoming file:...` restore. Guests: q35, 4 vCPU, 4 GiB and 16 GiB, W-idle and W-kv. Storage: local NVMe, page cache dropped before each restore.

Metrics: save time; restore time from `execve` to the guest running (first `kvm_entry` after resume); time to first useful work (a Redis GET answered); and total bytes read during the first 10 seconds after resume.

Targets: save and restore time at most 0.8x QEMU for `mapped-ram` file restore; time to first useful work at most 0.5x QEMU when ruvm's lazy restore is enabled (guest memory faulted in from the snapshot file via userfaultfd with a working set prefetch list recorded at save time, in the line of Firecracker snapshots, REAP, and FaaSnap, document 17). Lazy restore is an opt-in machine property, off by default and off in all compatibility tests.

Gate: nightly, 30 runs each. Fail on more than 5% regression against main or crossing the QEMU ratio.

## Area 9: qemu-img convert and check

ruvm-img is invoked as `qemu-img` and must accept the same options (document 14). Benchmarks run on X-AMD with source and destination on separate NVMe devices, page cache dropped, both binaries using the same `-m` (coroutines, or parallel requests for ruvm) and `-W` settings.

| Operation | Input | Command |
|---|---|---|
| convert raw to qcow2 | 64 GiB raw, 60% allocated with a fixed file system image | `qemu-img convert -O qcow2 -m 16 -W` |
| convert qcow2 to raw | the result above | `qemu-img convert -O raw -m 16 -W` |
| convert with compression | same input | `-c -o compression_type=zstd` and `compression_type=zlib` |
| convert across a backing chain | 5 layer qcow2 chain with 10% of clusters changed per layer | `qemu-img convert -O qcow2` |
| check | 1 TiB qcow2 with 64 KiB clusters, fully allocated, and a second one with 10,000 internal snapshots' worth of refcount churn | `qemu-img check` |
| map | the backing chain | `qemu-img map --output=json` |
| bench | the built-in `qemu-img bench` read and write modes | `qemu-img bench -c 1000000 -d 64` |

Metrics: wall time, CPU time, and peak RSS. Output files must be bit-identical to QEMU's output for uncompressed conversions and identical in guest-visible content for compressed conversions (compressed cluster layout may differ only if document 14 allows it; today it does not, so they must be identical too).

Targets: uncompressed convert at least 1.0x QEMU (it is I/O bound, so the realistic outcome is parity); compressed convert at least 1.2x QEMU from better pipelining of compression and write; `check` at least 1.5x QEMU on the 1 TiB image from parallel refcount scanning across L2 tables with a single consolidation pass (document 14); `map` at least 1.0x. These are decisions of this document, since the canon does not set image tool targets.

Gate: per pull request touching ruvm-block or ruvm-img: convert raw to qcow2 at 8 GiB and check at 64 GiB, 10 runs. Fail on more than 3% regression. Output bit-identity is checked in every run and is a hard failure regardless of performance.

## Area 10: many-VM density

Workload: launch U1-style microVMs (1 vCPU, 128 MiB, Debian minimal rootfs running a tiny HTTP responder) on X-AMD until the host has 32 GiB of free memory left, recording for each VM the launch latency (T0 to first HTTP response) and the aggregate host memory used, with KSM off and THP `madvise`. A second run uses q35 guests (1 vCPU, 512 MiB) to represent conventional VMs.

Metrics: number of VMs at the memory limit; launch rate sustained over 60 seconds with 32 concurrent launchers (VMs per second); p99 launch latency at 50% and 90% of the final VM count; per-VM host overhead (host memory used minus guest touched memory, divided by VM count). For context only, Firecracker advertises up to 150 microVMs per second per host and under 5 MiB overhead for a 1 vCPU, 128 MiB microVM ([Firecracker](https://firecracker-microvm.github.io/)).

Targets: VM count at least 1.4x QEMU (a consequence of the 0.6x overhead target once guest memory is included), launch rate at least 2x QEMU, p99 launch latency at 90% load at most 0.5x QEMU. Techniques are those of areas 2 and 3, plus file-backed shared mappings for firmware and ROM blobs instead of per-VM copies.

Gate: weekly. Nightly runs a 64 VM version to catch launch path regressions.

## CI regression gate summary

| Tier | When | Hardware | Duration budget | Suites |
|---|---|---|---|---|
| perf-smoke | every pull request that touches a listed crate | one X-AMD, one A-NEO | 20 minutes wall | CoreMark and proxy set, U1 boot, overhead RSS, vmexit, fio null-co, netperf TCP_RR, qemu-img small |
| perf-nightly | every night on main | all classes | 8 hours | SPEC train, boot matrix, full fio and net matrix, migration W-idle and W-stress, snapshot, vCPU sweep |
| perf-weekly | Sunday | all classes plus MIG-PAIR | 48 hours | SPEC ref, linux-user builds, all migration workloads, density, 6.12 kernel repeat |
| perf-release | each release candidate | all classes | as weekly | everything, results published with the release |

perf-smoke interleaves the pull request binary with a cached main binary in ABBA order. A pull request can override a failing perf-smoke gate only with a label applied by a maintainer of the affected area plus a written justification in the pull request, and the override is visible on the dashboard. The QEMU ratio gates never have overrides: if we fall below a canon target, the release does not ship until it is fixed or the target is formally changed in document 25.

## Profiling toolkit

The toolkit ships in release builds, not a separate debug build.

### perf with jitdump and perfmap

ruvm implements QEMU's `-perfmap` and `-jitdump` options with the same semantics as documented in QEMU's `docs/devel/tcg.rst`: `-perfmap` writes `/tmp/perf-<pid>.map` entries for each translated block so `perf report` can attribute samples, and `-jitdump` writes a jitdump file (the format defined by the Linux perf tools) with code bytes and guest debug information, which `perf inject -j` merges into `perf.data`. Symbol names follow QEMU's format, with the tier (`t1` or `t2`) appended. QEMU resolves system mode guest symbols only from an ELF `-kernel`; ruvm also accepts `-object ruvm-trace,guest-symbols=<file>` (System.map or ELF) for kernels booted from disk. Tier 2 regions list their constituent guest blocks in the jitdump debug info.

On macOS the perf-map file is written for samply, and translations are marked with `os_signpost` for Instruments.

### Tracing spans

ruvm-trace (document 03) compiles in every QEMU trace event with the same names and arguments as the `trace-events` files in QEMU, so `-trace enable=...` and the `trace-event-set-state` QMP command behave identically. Separately, ruvm records hierarchical spans (vCPU exit, MMIO dispatch, block request from virtqueue pop to completion, migration iteration, JIT translation) tagged with vCPU index, QOM path, and node name, into per-thread ring buffers exported as Perfetto traces (`-object ruvm-trace,file=<path>`). The timeline shows, for example, a vCPU waiting on a device lock held by an iothread. Spans are off by default and enabled per category at runtime.

### Flamegraphs

`ruvm-bench profile <benchmark>` runs a benchmark under `perf record -g` (release builds keep frame pointers; we re-measure the cost at M4), merges jitdump, and renders flamegraphs with inferno: a host view, a guest view (host samples in translated code mapped to guest PCs and symbols), and a differential ruvm versus QEMU view, which is the fastest way to see where a ratio comes from.

### Internal counters via QMP

ruvm keeps per-vCPU and per-device counters in thread-local cache-line aligned structures that are summed on read: exits by reason, MMIO and PIO counts per device, TB translations and invalidations, tier 2 region count and code size, TLB fills and flushes, jump cache hit rate, helper call counts by helper, virtqueue notifications and interrupts per queue, iothread reactor submissions and completions per wakeup, block requests per node, dirty pages harvested per iteration.

Because `query-qmp-schema` must match QEMU 11.1 by default (document 02), counters are exposed through a command with QAPI's downstream extension naming, `__io.github.tamnd.ruvm_query-counters`, present only when extensions are enabled (the `ruvm` personality or `RUVM_EXTENSIONS=1`, document 02). With extensions off (the default, and the mode all compatibility tests use) the schema is byte-identical to QEMU's. The command takes optional group names and a `reset` flag and returns flat `{name, labels, value}` records that map directly to Prometheus. KVM statistics stay on the standard `query-stats`, and HMP `info jit` keeps QEMU's format.

`ruvm top` attaches to such a socket and shows counters as rates, enough to spot an exit storm or a device on a slow path.

### ruvm-bench

`ruvm-bench` (document 24) provides `host-prep`, `hostinfo`, `run <suite>` with interleaving, `stats` for the bootstrap, and `profile`, writing JSON lines with raw samples. It is tested weekly with an A/A run (same binary on both sides): intervals must contain 1.0 for at least 95% of metrics, or the harness or host is biased.

## Decisions made in this document

- Performance claims are ratios against a self-built QEMU 11.1.0 with clang, LTO, `-O2`, the same target CPU level, and glibc malloc, and are "met" only when the lower bound of the 95% bootstrap interval clears the target.
- Host classes X-AMD, X-INTEL, A-NEO, A-APPLE, MIG-PAIR as above; host kernel Linux 6.18 LTS with 6.12 LTS as a weekly secondary.
- JIT targets are split by milestone: at least 1.25x SPEC CPU2017 intrate geometric mean at M4 with tier 1, and the canon's 2x at M9 with tier 2.
- qemu-img targets: uncompressed convert parity, compressed convert at least 1.2x, check at least 1.5x on large images.
- Internal counters are exposed via `__io.github.tamnd.ruvm_query-counters`, only when extensions are enabled, so default QMP introspection stays identical to QEMU.
- System mode JIT symbolization and Perfetto span output are configured as properties of a `ruvm-trace` object (`-object ruvm-trace,guest-symbols=...,file=...`), so the top level option table stays QEMU's (document 25, Q2).
- Lazy snapshot restore via userfaultfd is an opt-in machine property, off by default.
- Release builds keep frame pointers, subject to re-measurement at M4.
