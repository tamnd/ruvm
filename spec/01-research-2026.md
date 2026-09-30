# 01. Research and prior art, September 2026

This document records what the rest of the ruvm spec is standing on. For each piece of prior work it says what the work actually shows (with the numbers the authors report, not our paraphrase of them) and exactly what ruvm takes from it, whether that is an algorithm, a test oracle, a data format, or a lesson about what not to do. Every citation here was checked against the primary source or the publisher page in September 2026. Where we could not verify a claim we left it out. The reference QEMU for the whole spec is 11.1.0, released 11 August 2026 (see document 02 for what "compatible with 11.1.0" means).

## 1. QEMU in 2026

### 1.1 QEMU 11.0 (22 April 2026)

QEMU 11.0.0 was [announced on 22 April 2026](https://www.qemu.org/2026/04/22/qemu-11-0-0/) with 2,500+ commits from 237 authors. The items that matter for ruvm:

- A new `nitro` accelerator and machine type that runs AWS Nitro Enclave images natively under QEMU. This is a new accelerator class, not a device, which is why the canon has a `ruvm-accel-nitro` crate and a `nitro` machine in ruvm-machine-x86.
- KVM: CET (shadow stack and IBT) virtualization, and reset support for SEV-SNP and TDX guests. Reset of a confidential guest means tearing down and rebuilding the encrypted launch context, which touches `system/runstate.c`, the `ConfidentialGuestSupport` class and the KVM VM fd lifecycle. ruvm has to model "VM fd may be replaced on reset" in `trait Accel` from the start (document 06).
- TCG: in-tree support for C++ plugins. The plugin ABI is still the C header `include/plugins/qemu-plugin.h`; C++ support is about headers and build glue, so ruvm-plugin gets it for free as long as the C ABI is exact.
- Arm: TCG emulation of SME, and HVF acceleration of SME2 on macOS hosts that have it.
- A new CPU model for Intel Xeon Diamond Rapids.
- Removal of support for 32-bit hosts. ruvm never supported them; this removes one reason anyone would ask.
- Removed machine types: `pc-i440fx-2.6`, `pc-q35-2.6`, `pc-i440fx-2.7`, `pc-q35-2.7`, plus the Arm `highbank` and `midway` boards. The versioned machine policy in `docs/about/deprecated.rst` is now mechanical: a versioned machine is deprecated automatically after 3 years (9 releases) and deleted after 6 years (18 releases). In the v11.1.0 tree the oldest generic compat array in `hw/core/machine.c` is `hw_compat_4_1`. ruvm follows the same window and never resurrects a machine version QEMU has deleted (document 02, section 5).

### 1.2 QEMU 11.1 (11 August 2026)

The [11.1.0 announcement](https://lists.nongnu.org/archive/html/qemu-devel/2026-08/msg01674.html) reports 3,200+ commits from 285 authors, the largest release by commit count in several years. Highlights: UFS emulation updated to UFS 4.1 with Write Booster and host-initiated defragmentation; a vhost-user backend for virtio-rtc; GTK and VNC console work including a character `encoding` property on `chardev-vc`; Arm CPU features, the `imx8mp-evk` board, cache topology on `virt`, and nested virtualization plus vGIC under HVF; HPPA moves to SeaBIOS-hppa v25; PowerPC MPIPL and nest MMU emulation; RISC-V big-endian support, Zbr and Zvfbfa and other extensions, the K230 board; s390x ASTFLE facility 2 under KVM.

Two facts from the tree itself are more useful to us than the release notes. First, `hw_compat_11_0` in `hw/core/machine.c` at the v11.1.0 tag contains ten properties, among them `virtio-mmio` queue size override 1024, `chardev-vc` `encoding=cp437`, four `arm-smmuv3` defaults (`ats=off`, `ril=on`, `ssidsize=0`, `oas=44`), TPM CRB chunking off, and `migration` `switchover-ack-legacy=on`. Every one of these is a guest-visible or migration-visible behavior change between the 11.0 and 11.1 machine types, and every one is something ruvm has to reproduce behind the machine version (document 11). Second, `qemu-options.hx` at v11.1.0 has 115 `DEF(` entries, `hmp-commands.hx` defines 114 commands and `hmp-commands-info.hx` 71, and `qga/qapi-schema.json` defines 44 guest agent commands. These counts are the denominators in the conformance matrix of document 02.

### 1.3 Rust inside QEMU

Rust in QEMU is still marked experimental. Paolo Bonzini's [January 2026 status update](https://ratatoskr.run/qemu-devel/2026/01/14421554) lists what exists: two device models in Rust (the Arm PL011 UART and the x86 HPET), initial QAPI bindings, dtrace probes via the probe crate, Rust enabled in CI on more hosts, and an MSRV fixed at 1.83 (which the v11.1.0 `rust/Cargo.toml` confirms as `rust-version = "1.83.0"`). Plans listed for later releases are QMP commands in Rust, a GStreamer audio backend, integrating the `vm-memory` crate, and a discussion about enabling Rust by default. At the v11.1.0 tag, `rust/hw/` contains exactly `char/pl011` and `timer/hpet`; a Rust I2C GPIO expander was mentioned as converted in January but is not in the tag. The 11.1 cycle mostly finished Meson and Cargo integration (for example propagating `qemu_ldflags` to the Rust link step).

The interesting detail is the shape of the bindings. The v11.1.0 tree has a `rust/bql` crate: QEMU's Rust code encodes "this data is protected by the Big QEMU Lock" in the type system (cells that can only be borrowed while the BQL is held). That is the correct way to add Rust to a BQL-based program, and it also shows the ceiling of the approach: the Rust devices inherit the global lock, the C memory API, the C object model and the C migration code. Rust at the leaves cannot remove the BQL or change the TCG IR.

What ruvm takes: nothing in code (the bindings wrap C APIs we do not have), but two lessons. The QEMU Rust `qom` and `migration` crates show which QOM and VMState concepts map cleanly to Rust traits and which need escape hatches; ruvm-qom and ruvm-vmstate (documents 04 and 17) start from the same concept list. And the QEMU community has chosen a 1.83 MSRV to cover Debian and Ubuntu LTS. ruvm uses edition 2024, which needs Rust 1.85 or later; we set ruvm's MSRV to 1.85 and revisit it once per year (new decision, recorded in document 25).

## 2. Why QEMU's architecture motivates a rewrite

QEMU is a successful program and most of this spec is about matching it exactly. The reasons for a rewrite are structural, not cosmetic. There are three.

### 2.1 The Big QEMU Lock

The BQL (renamed from "iothread lock" to `bql_lock()`/`bql_unlock()` by Stefan Hajnoczi's series merged in January 2024) is a single global mutex protecting most of QEMU's state. KVM vCPU threads drop it while running guest code, but reacquire it for every MMIO or PIO exit that is not on a region marked lockless, and all timers, bottom halves and QMP commands in the main loop run under it. Under MTTCG the same applies to every IO access from translated code.

The cost is visible in QEMU's own patches. Igor Mammedov's 2025 series [reintroducing BQL-free MMIO](https://www.mail-archive.com/qemu-devel@nongnu.org/msg1132741.html) (`memory_region_enable_lockless_io()`) gives the motivation directly: make ACPI PM timer and HPET reads cheaper and "prevent BQL contention in case of workload that heavily uses the timers with a lot of vCPUs". The same patch has to disable the per-device reentrancy guard for lockless regions, with a TODO saying it should become per transaction. That is the core problem: in QEMU, device state consistency, reentrancy protection and scalability are all tied to one lock, so loosening one loosens the others.

What ruvm takes: the concurrency model in the canon (per-device lock chosen by the device, device domains for synchronously coupled devices, one control lock for topology). The HPET and ACPI PM timer are the first devices we benchmark against QEMU for lock contention, because QEMU's own maintainers identified them (document 21). The reentrancy guard becomes per transaction from day one (section 7).

### 2.2 TCG IR limits

TCG translates one guest translation block (a basic block, or an extended basic block since the EBB work) into its IR, optimizes it locally, allocates host registers with a simple local allocator and emits host code. Guest registers live in `CPUArchState` in memory; TCG caches them in host registers only within a block and must write them back before helper calls and at block exits. Cross-block optimization does not exist. Condition flags are handled per frontend (x86 uses `cc_op` lazy flags), so dead flag work is only eliminated when it is dead inside the same block. Memory ordering is enforced by fences that the Risotto authors showed are both too conservative and, for some cases, incorrect (section 4.3).

This design is the reason TCG translates quickly and supports 20 guest ISAs on 8 or so hosts. It is also the ceiling on steady-state speed. The papers in section 4 all beat TCG by doing one of three things TCG cannot: cross-block register allocation, flag liveness across blocks, or better memory ordering mappings.

What ruvm takes: ruvm-jit keeps a TCG-like tier 1 (fast, block-local, same granularity so TB-based plugin callbacks and precise exceptions stay simple) and adds a tier 2 over hot regions, as the canon says (documents 07 and 08).

### 2.3 Memory safety in device emulation

QEMU's own security policy (`docs/system/security.rst`) treats the guest, user-facing interfaces, network protocols, user-supplied files and passthrough devices as untrusted, but only for the "virtualization use case": KVM or HVF plus one of a listed set of machine types (`virt` on aarch64, loongarch64 and riscv; `microvm`, `xenfv`, `xenpv`, `xenpvh`, `pc`, `q35` on x86; `s390-ccw-virtio`; `pseries`). Everything else is out of scope for security fixes. The history of guest-to-host bugs in the in-scope devices is long, and most are the kind that a memory-safe language removes or turns into a controlled failure:

- [CVE-2015-3456 (VENOM)](https://access.redhat.com/articles/1444903): out-of-bounds write in the floppy controller FIFO in `hw/block/fdc.c`, reachable through commands like `FD_CMD_READ_ID`. The FDC was instantiated on every x86 PC machine whether or not a floppy was configured. The bug had been present since 2004.
- [CVE-2019-14378](https://www.openwall.com/lists/oss-security/2019/08/01/2): heap overflow in SLiRP `ip_reass()` when the first fragment is larger than `m->m_dat[]`, with a public [exploit write-up](https://blog.bi0s.in/2019/08/13/Pwn/VM-Escape/2019-07-29-qemu-vm-escape-cve-2019-14378/) that escapes the VM.
- [CVE-2020-14364](https://access.redhat.com/security/cve/cve-2020-14364): out-of-bounds read and write in `hw/usb/core.c` when `setup_len` from the guest exceeds `data_buf[4096]` in `do_token_in`/`do_token_out`. libvirt adds a USB controller and tablet by default, so the reachable population was large.
- [CVE-2021-3929](https://ubuntu.com/security/CVE-2021-3929): use-after-free in the NVMe controller. The guest points a DMA transfer at the controller's own MMIO registers, the write triggers `nvme_ctrl_reset()`, and the reset frees structures still in use by the outer call. The fix in 7.0 denied DMA to the device's own iomem locally.
- CVE-2023-0330 (lsi53c895a) and the general DMA reentrancy class were addressed by Alexander Bulekov's [reentrancy guard](https://gitlab.com/qemu-project/qemu/-/commit/a2e1753b8054), a `MemReentrancyGuard` in `DeviceState` checked in the memory dispatch path, which returns `MEMTX_ACCESS_ERROR` and logs "Blocked re-entrant IO on MemoryRegion". It closed a group of long-standing QEMU issues, including GitLab #556.
- CVE-2024-3446: double free in virtio-gpu, virtio-serial-bus and virtio-crypto where the guard did not cover bottom halves, fixed by `virtio_bh_new_guarded()`.
- [CVE-2026-17588](http://www.mail-archive.com/qemu-devel@nongnu.org/msg1226232.html): heap use-after-free in xHCI, fixed in September 2026. `xhci_mfwrap_timer` and `xhci_ep_kick_timer` process events from timer callbacks, outside any MMIO handler, so the guard is not engaged. The guest points the event ring at the doorbell registers, the DMA write of a completion event rings doorbell 0, a prepared `CR_DISABLE_SLOT` frees endpoint and transfer objects, and the outer stack keeps using them.

The first three are plain bounds errors, which Rust turns into a panic. ruvm builds with `panic = "abort"`, so the VM stops with the device named and its recent register accesses printed, instead of letting the guest write outside the buffer (document 19). A denial of service against your own VM is a much smaller problem than code execution on the host. The last four are a subtler class: reentrancy through DMA into the device's own MMIO while the device is mid-operation, which in C becomes a use-after-free. Rust does not make that class disappear by itself; it makes it impossible to express the bug as silent aliasing. If device state is behind a lock and the DMA path tries to re-enter the same device, the code either deadlocks or the reentrant access is detected with a `try_lock` failure. ruvm chooses detection: the device domain executor marks the domain busy for every entry point (MMIO, PIO, timer, bottom half, ioeventfd, netqueue), and a reentrant access into a busy domain returns `MEMTX_ACCESS_ERROR` exactly as QEMU's guard does, so the guest-visible behavior matches QEMU where QEMU guards and is safe where QEMU does not. Bulekov himself pointed out that ioeventfd and timers were still unguarded in QEMU after 2023; CVE-2026-17588 is the timer case arriving three years later.

What ruvm takes: the CVE list above becomes regression tests in ruvm's device test suite (document 22), each reproducer ported to the qtest protocol so it runs against both QEMU and ruvm. The "reentrancy returns `MEMTX_ACCESS_ERROR`" behavior is a documented part of the compat contract (document 02, section 3.4). And ruvm treats every guest-reachable device as a security boundary under any accelerator, which is broader than QEMU's policy (new decision; document 19 lists the consequences for triage).

## 3. The rust-vmm ecosystem and Rust VMMs

Six years of Rust VMMs have proven that Rust is the right language for a device model and a VMM control plane. None of them tries to be QEMU, and each one says what it leaves out on purpose. Those choices explain why a Rust QEMU replacement did not already exist.

### 3.1 rust-vmm crates

rust-vmm is a set of shared crates: `kvm-ioctls` and `kvm-bindings`, `vm-memory`, `virtio-queue`, `vhost` and `vhost-user-backend`, `vfio-ioctls`, `linux-loader`, `vmm-sys-util`, `event-manager`, `seccompiler`, `vm-superio`, `acpi_tables`. They are mostly dual-licensed Apache-2.0 plus BSD-3-Clause or MIT, which is compatible with GPL-2.0-or-later when the permissive option is taken. QEMU itself now plans to use `vm-memory`, and the January 2026 update notes that IOMMU support in rust-vmm is complete.

What ruvm takes: `kvm-bindings` and `vfio-bindings` for generated kernel structures (behind ruvm-sys, so the rest of ruvm never sees them), and `virtio-bindings` for constants. The virtqueue and vhost-user code is ruvm's own (ruvm-virtio-queue and ruvm-vhost, document 13), because it has to reproduce QEMU's VMState layouts and legacy transport quirks, and rust-vmm's `virtio-queue` and `vhost` crates serve as differential test oracles for it. ruvm does not use `vm-memory` as its core memory model, because QEMU semantics need a MemoryRegion tree with priorities, aliases, IOMMU regions and per-access attributes, which `GuestMemory` does not model. Instead a small crate, ruvm-mem-vmm, provides an adapter that implements `vm_memory::GuestMemory` over a FlatView snapshot, so existing `vhost-user-backend` daemons and virtio device crates can run inside ruvm unchanged (document 05). ruvm does not use `linux-loader`, because QEMU's kernel loading (`hw/i386/x86-common.c`, `hw/arm/boot.c`) has specific placement and setup header rules that `linux-loader` does not reproduce byte for byte. ruvm consumes rust-vmm crates under their MIT or BSD-3-Clause option (new decision, recorded for document 24's provenance check).

### 3.2 Firecracker

[Firecracker (Agache et al., NSDI 2020)](https://www.usenix.org/system/files/nsdi20-paper-agache.pdf) reported under 5 MB memory overhead per microVM, boot to application code in under 125 ms, and up to 150 microVMs per second per host. It exposes five devices (virtio-net, virtio-block, virtio-vsock, serial, and an i8042 used only to stop the VM), boots a Linux kernel directly with no firmware, and has a jailer and per-thread seccomp filters. PCI arrived only in [1.13.0 behind `--enable-pci`](https://github.com/firecracker-microvm/firecracker/blob/main/CHANGELOG.md), and a virtio-pci transport bug followed soon after: [CVE-2026-5747](https://aws.amazon.com/security/security-bulletins/2026-015-aws/), an out-of-bounds write affecting 1.13.0 to 1.14.3 and 1.15.0. That is a useful reminder that Rust moves the bug class, it does not end it: an index computed from guest data still needs a check, and the check has to be right.

What Firecracker deliberately does not do: TCG, firmware boot, legacy devices, GPU or device passthrough, QMP, QEMU CLI compatibility, migration interop with anything. What ruvm takes: the startup budget. Firecracker proves a Rust VMM can go from process start to guest in tens of milliseconds, which anchors the canon target of 15 ms to first guest instruction for a microvm-class config. ruvm's `microvm` machine is QEMU's (`hw/i386/microvm.c`), not Firecracker's, but the startup path design (no device enumeration at startup that is not needed, lazy QOM type init, prefaulted guest memory only when asked) is borrowed from Firecracker's approach (document 21). The snapshot format lessons are in section 6.

### 3.3 Cloud Hypervisor

[Cloud Hypervisor](https://github.com/cloud-hypervisor/cloud-hypervisor) runs on KVM and MSHV, targets "modern cloud workloads" with virtio I/O, lists "64-bit support only", "minimal emulation" and "small attack surface" as goals, and supports CPU, memory and PCI hotplug and machine-to-machine migration. It deliberately does not do TCG, 32-bit guests, legacy devices beyond a serial port, or QEMU compatibility.

What ruvm takes: Cloud Hypervisor's MSHV support is the only production Rust code for the Microsoft hypervisor, and ruvm-accel-mshv reads it for the ioctl sequences (QEMU 11.0 improved its own MSHV accelerator, which is the behavior we match). Its vhost-user and VFIO plumbing informs document 16.

### 3.4 crosvm

[crosvm](https://crosvm.dev/book/architecture/overview.html) runs each virtual device in its own forked process, jailed with minijail (PID, mount, user and network namespaces, `PR_SET_NO_NEW_PRIVS`, per-device seccomp policy files), talking to the main process only through proxied `BusDevice` calls, shared guest memory, and explicitly allowed file descriptors. It supports several hypervisors. It deliberately does not do TCG or QEMU compatibility, and the process-per-device design costs a context switch on every trapped MMIO access to a jailed device.

What ruvm takes: an optional out-of-process mode for high-risk devices (USB host controllers, audio, display, SLiRP) built on vhost-user where a vhost-user protocol exists and on a crosvm-style proxied `MmioOps` otherwise. It is off by default because it changes latency, and it is never used for devices on the hot path such as virtio-net with vhost (document 19).

### 3.5 libkrun

[libkrun](https://github.com/containers/libkrun) is a library, not a program: an application links it and gets a VM with a C API. It supports KVM on Linux and HVF on macOS arm64, variants for SEV and TDX, and a minimal device set. Its Transparent Socket Impersonation gives the guest network access without a network interface by forwarding AF_INET, AF_INET6 and AF_UNIX stream and datagram sockets over vsock, with a custom guest kernel. Its security model states that guest and VMM share one security context; virtio-fs gives no protection against the guest reaching outside the shared directory.

What ruvm takes: the "VMM as a library" shape. ruvm's crates are libraries first and the `ruvm` binary is a thin shell, so an embedder can build a VM from `ruvm-system` without the CLI (document 20). ruvm does not take TSI, because it requires a modified guest kernel and has no QEMU equivalent.

### 3.6 Dragonball

[Dragonball](https://github.com/kata-containers/kata-containers/blob/main/src/dragonball/README.md) is the Rust VMM built into the Kata Containers runtime (runtime-rs) since Kata 3.0 and the default built-in VMM in Kata 4.0. It removes the separate VMM process entirely, and its "upcall" channel lets the VMM hotplug vCPUs, memory and virtio-mmio devices by talking directly to a guest kernel driver over vsock instead of going through ACPI. It deliberately does not target anything but Kata container sandboxes.

What ruvm takes: the evidence that an in-process VMM with a direct control channel beats ACPI hotplug for containers. ruvm does not implement upcall (it needs guest kernel patches and QEMU has no equivalent) but keeps `ruvm-system` embeddable so a Kata shim could link ruvm the same way.

### 3.7 What none of them do

None of the Rust VMMs has a binary translator, none speaks QMP, none accepts QEMU's command line, none can migrate to or from QEMU, and none runs the pc, q35 or Arm boards that libvirt, OpenStack and Proxmox deploy. That set is exactly the scope of ruvm.

## 4. Binary translation research

### 4.1 HQEMU (CGO 2012)

[HQEMU](https://dl.acm.org/doi/10.1145/2259016.2259030) (Hong et al.) keeps QEMU's TCG as a fast first translator and runs LLVM on a separate core to optimize hot traces. On SPEC CPU2006, x86 to x86-64, it reports 2.4x (integer) and 4x (floating point) speedups over QEMU. What ruvm takes: the two-tier structure with the optimizing tier on background threads so vCPU threads never wait for tier 2 (document 08). ruvm does not use LLVM for tier 2: compile latency and binary size are too high for a system emulator that may hold hundreds of thousands of hot regions, and LLVM's memory model does not express the guest orderings we need (see Instrew's LR/SC limitation below).

### 4.2 Instrew (VEE 2021)

[Instrew](https://dl.acm.org/doi/10.1145/3453933.3454022) (Engelke, Okwieka, Schulz) lifts guest machine code to LLVM IR and lets LLVM optimize and allocate registers over regions larger than a basic block, which is where its advantage over TCG comes from. The VEE 2021 paper covers a RISC-V guest and an AArch64 host and reports SPEC CPU2017 improvements over QEMU. It also documents a limit: RISC-V LR/SC loops cannot be represented in LLVM and are treated as non-atomic.

What ruvm takes: region-level register allocation in tier 2, with hot guest registers kept in host registers across block boundaries. Separately, and as ruvm's own design rather than something the paper evaluates, tier 2 translates guest call/return pairs to host call/return with a shadow stack check that falls back to the jump cache on mismatch, so the host return stack buffer predicts guest returns. The LR/SC limit is a warning: ruvm's IR has first-class exclusive monitor operations (document 07).

### 4.3 Risotto (ASPLOS 2023) and Arancini (ASPLOS 2026)

[Risotto](https://dl.acm.org/doi/10.1145/3567955.3567962) (Gouicem et al.) formalized the memory model of TCG's IR, proved mapping schemes from x86 through the IR to Arm correct, and showed that QEMU's mappings both forbid reorderings x86 allows and in some cases are incorrect for strong-on-weak translation. Built on QEMU, it improves performance by up to 19.7% and 6.7% on average over stock QEMU while being correct, also using Arm's CAS instructions for x86 `cmpxchg` and a cross-architecture dynamic linker for native host libraries.

[Arancini](https://dl.acm.org/doi/10.1145/3779212.3790127) (Reimers, Sprokholt et al., ASPLOS 2026) is a hybrid static plus dynamic translator built from scratch around ArancinIR, a single low-level IR used by both translators, with a formal memory model and formally verified mappings from x86-64 to Arm and to RISC-V. It is the first to prove mappings that handle mixed-size accesses on Armv8. Evaluated on Phoenix with both backends, it is up to 5x faster than Risotto while enforcing x86 ordering correctly. The proofs are in Agda in the [binary-translation](https://github.com/binary-translation) organization.

What ruvm takes: the fence placement tables. ruvm-jit's IR memory model is defined to be the same as ArancinIR's for the operations they share, so the verified x86-64 to Arm and x86-64 to RISC-V mappings apply directly; the tables are transcribed into `ruvm-jit` with a reference to the lemma each row comes from, and document 08 lists the rows. Mixed-size access handling on Arm hosts follows Arancini's scheme. ruvm does not take the static half: system emulation cannot know code ahead of time, and linux-user AOT caching is a later optimization (document 10).

### 4.4 Rosetta 2, FEX-Emu, Box64, LATX

These are production translators and each one shows what matters in practice for x86 guests on other hosts.

- Rosetta 2 relies on hardware: Apple cores have a TSO memory ordering mode (the `ACTLR_EL1.TSOEN` bit, known from public reverse engineering rather than Apple documentation), FEAT_FlagM and FlagM2, and extra flag bits (26 and 27, next to NZCV) for x86 parity and adjust flags. Apple exposes the Rosetta runtime to Linux guests through the [Virtualization framework](https://developer.apple.com/documentation/virtualization/accelerating-the-performance-of-rosetta). What ruvm takes: on macOS hosts with HVF, a Linux guest running under ruvm's own TCG replacement cannot flip `ACTLR.TSOEN` from user space, but ruvm-jit-aarch64 detects FEAT_LRCPC, LRCPC2, LSE2 and FlagM at startup and picks cheaper mappings where the host supports them.
- [FEX-Emu](https://fex-emu.com/) is an IR-based x86 and x86-64 user-mode translator for arm64 Linux with library thunks for GL and Vulkan, JIT and IR caches, and configurable TSO emulation that can be relaxed per application. It uses LRCPC and LRCPC2 acquire/release instructions where the host has them, and it backpatches unaligned atomic accesses that fault into a barrier plus a plain access instead of taking the fault every time. What ruvm takes: the backpatching approach for unaligned atomics on Arm hosts, and the lesson that the tier 1 fence mapping is where x86-on-Arm performance is won or lost (document 08).
- [Box64](https://github.com/ptitseb/box64) targets arm64, RISC-V 64 and LoongArch 64 with a dynarec reported at 5 to 10x over its interpreter, and forwards calls to native host libraries (libc, SDL, GL) through wrappers. It exposes a `STRONGMEM` knob for ordering. What ruvm takes: nothing for system mode. For linux-user, native library forwarding is incompatible with QEMU's behavior (qemu-user never does it), so it is out of scope for the compatible personality and considered only as an opt-in `ruvm run` extension (document 10, open question in document 25).
- [LATX](https://github.com/lat-opensource/lat) is a Loongson x86 translator built on QEMU 6 using LoongArch's binary translation extensions (LBT) and borrowing Box64's library passthrough. The Loongson ISA paper reports x86 to LoongArch running 3.6x (int) and 47.0x (fp) faster than QEMU with hardware support. What ruvm takes: ruvm-jit-loongarch64 uses LBT flag instructions when present.

### 4.5 Captive (USENIX ATC 2019)

[Captive](https://www.usenix.org/system/files/atc19-spink.pdf) (Spink, Wagstaff, Franke, best paper) generates a system-level translator from a high-level architecture description and runs it inside a KVM virtual machine, so the translator can use the host MMU and privilege levels for the guest's page tables instead of a software TLB. From an 8,100 line ARMv8-A model it beat QEMU by 2.21x on SPEC CPU2006 integer and up to 6.49x on floating point. What ruvm takes: this is the most important idea for ruvm's system-mode performance that ruvm does not adopt in version 1. Using the host MMU for guest memory translation removes the softmmu TLB lookup from every load and store, but it requires running translated code inside a KVM guest with its own mini-kernel, which conflicts with running on macOS and Windows hosts and with the TCG plugin API's memory callbacks. It is recorded as a post-1.0 research item in document 25.

### 4.6 Learned translation rules (CGO 2024)

[Jiang et al., CGO 2024](https://arxiv.org/abs/2402.09688) apply automatically learned translation rules to system-level emulation and report 1.36x average speedup over QEMU 6.1 on SPEC CINT2006 and 1.15x on real applications. What ruvm takes: learned rules are a way of producing better instruction selection patterns offline. ruvm's backend pattern tables are hand-written, but the idea of verifying each rule by differential execution is used in ruvm's backend test harness (document 22).

### 4.7 Flag speculation (MobiSys 2025)

["ARMing x86 Games"](https://doi.org/10.1145/3711875.3729163) (Yen et al., MobiSys 2025) uses data flow analysis to find where Arm's hardware NZCV flags can stand in for x86 flags, with software validation, and reports up to 18% on compute tasks and 7% to over 12% FPS on Steam titles. What ruvm takes: tier 2's flag pass. Tier 1 keeps lazy flags (operands plus op kind, as in TCG's x86 `cc_op`); tier 2 runs liveness over the region and, where only Z, N, C or V are consumed and the Arm semantics match (carry inversion for subtraction is handled explicitly), keeps flags in NZCV instead of materializing them.

### 4.8 Direct translation without an IR (arXiv 2501.03427)

[Parker, arXiv 2501.03427](https://arxiv.org/abs/2501.03427) (January 2025, a 6 page conference paper) argues that TCG's IR adds pipeline steps and reports a proof-of-concept emulator up to 35x faster than QEMU TCG. The scope is narrow: RISC-V base integer instructions, and the prototype executes through a register array and memory interface rather than translating. The paper proposes a middle tier of direct translators for common pairs. What ruvm takes: the measurement that TCG's per-instruction overhead on simple integer code is large enough to matter, which supports ruvm's decision to make tier 1 IR small (SSA-lite, few passes). ruvm does not build per-pair direct translators: with 19 guests and 6 hosts that is up to 114 translators, and the IR is what makes the plugin API, precise exceptions and the verified memory mappings apply uniformly.

### 4.9 Other translators

Unicorn is QEMU's TCG extracted as a CPU emulation library; it shows the demand for embedding a translator without a machine, which ruvm-jit plus a `GuestArch` crate serve without QEMU's global state (document 20). We looked for 2025 and 2026 CGO and VEE papers on system-level DBT beyond those cited here and could not verify one with results that bear on ruvm, so none is cited. Document 25 keeps a standing item to recheck each year.

## 5. Formal ISA specifications

- Sail is the language of the official RISC-V golden model, and Sail models of Armv8/v9 are generated from Arm's ASL. [Pydrofoil (Bolz-Tereick et al., ECOOP 2025)](https://drops.dagstuhl.de/storage/00lipics/lipics-vol333-ecoop2025/LIPIcs.ECOOP.2025.3/LIPIcs.ECOOP.2025.3.pdf) compiles Sail with an AOT compiler plus a PyPy meta-tracing JIT and reports over 230x speedup over the Sail-generated C simulator on the RISC-V model, while still being 26.7x slower than QEMU.
- Arm publishes the A-profile Architecture Reference Manual (DDI 0487, M.c at the time of writing) with machine-readable descriptions of features, registers and instructions up to Armv9.6, written in ASL1, whose [ASL Reference (DDI 0626)](https://developer.arm.com/Architectures/Architecture%20Specification%20Language) defines the sequential language at EAC quality (the concurrent semantics is still DEV quality). ASLRef in herdtools7 is the reference interpreter.

What ruvm takes: formal models are test oracles, not code generators. Pydrofoil's own numbers show that even an aggressively optimized spec-derived simulator is an order of magnitude behind TCG, and ruvm has to be faster than TCG. So ruvm-target-riscv and ruvm-target-arm are hand-written (ported from QEMU's decodetree files and translators), and the Sail RISC-V model and ASLRef-executed Arm pseudocode are run in lockstep against ruvm-jit on random and directed instruction streams (document 22). Arm's register XML is used to generate system register tables and field layouts for ruvm-target-arm, checked against QEMU's `target/arm/cpregs` definitions, because register metadata has no speed cost.

## 6. Snapshot and restore

- Firecracker snapshots save guest memory as a file and device state separately, and restore by `mmap` of the memory file, so pages fault in on demand.
- [REAP (Ustiugov et al., ASPLOS 2021)](https://dl.acm.org/doi/10.1145/3445814.3446714) found that functions restored from a Firecracker snapshot are slowed mainly by one-page-at-a-time faults, that the working set is stable across invocations, and that recording and prefetching it removes 97% of page faults and speeds cold starts 3.7x.
- [FaaSnap (Ao, Porter, Voelker, EuroSys 2022)](https://dl.acm.org/doi/10.1145/3492321.3524270) improves the working set estimate ("loading sets"), handles page types differently per region, prefetches asynchronously, and reports up to 3.5x faster cold starts than prior snapshot approaches.
- [Catalyzer (Du et al., ASPLOS 2020)](https://dl.acm.org/doi/10.1145/3373376.3378512) restores from checkpoint images with on-demand recovery of memory and system state and adds `sfork` to clone a running sandbox, reaching sub-millisecond startup in the best case.
- QEMU itself added the `mapped-ram` migration capability in 9.0: each RAM page has a fixed offset in the file, so multifd threads can write in parallel and `O_DIRECT` works. A 2026 series ("migration: fast snapshot load", v4 in August 2026) combines mapped-ram with postcopy and userfaultfd to load pages on demand; it is not merged at the time of writing.

What ruvm takes: ruvm implements QEMU's mapped-ram format exactly (it is part of the migration compat surface in document 02). On top, ruvm adds lazy restore from a mapped-ram file through userfaultfd on Linux and a REAP-style working set record: the first restore records faulting page offsets into a sidecar file next to the snapshot, and later restores prefetch those ranges with io_uring before resuming vCPUs. The sidecar is a ruvm extension that QEMU ignores (it is a separate file), so the snapshot stays loadable by QEMU. `sfork`-style cloning is out of scope because it needs host kernel support. Document 17 has the details.

## 7. Record and replay

QEMU's [record/replay](https://www.qemu.org/docs/master/system/replay.html) builds on icount: `-icount shift=auto,rr=record,rrfile=replay.bin` logs every non-deterministic event (input, clocks, interrupts, block and network I/O through `blkreplay` and `filter-replay`), and `rr=replay` reproduces it; `rrsnapshot` takes a VM snapshot at the start, and reverse debugging works through the gdbstub. Because it rides on icount it works only with single-vCPU TCG, not KVM or MTTCG. [rr (O'Callahan et al., USENIX ATC 2017)](https://www.usenix.org/system/files/conference/atc17/atc17-o_callahan.pdf) records user-space process groups on stock Linux and hardware with low overhead for low-parallelism workloads, using hardware performance counters to count retired branches.

What ruvm takes: QEMU's command line and QMP (`replay-break`, `replay-seek`, `query-replay`) for record/replay, with the same single-vCPU TCG restriction in the compatible personality. The replay log file is not a compatibility surface: QEMU versions it and rejects mismatches, and ruvm uses its own format with the same event model (new decision, recorded in document 02, section 9). ruvm-jit's precise instruction counting is done with per-block instruction counts and a decrementer checked at block entry, as TCG does, so icount results match QEMU within the same block boundaries (document 17).

## 8. Device emulation fuzzing

- [Nyx (Schumilo et al., USENIX Security 2021)](https://www.usenix.org/system/files/sec21-schumilo.pdf) is a coverage-guided hypervisor fuzzer built on KVM-PT with fast whole-VM snapshot reset and an affine-typed bytecode for inputs.
- [Morphuzz (Bulekov et al., USENIX Security 2022)](https://www.usenix.org/system/files/sec22-bulekov.pdf) needs no device-specific seeds or grammars: it reshapes the input space so any byte stream becomes a sequence of MMIO, PIO and DMA-backed interactions, runs inside QEMU through the qtest accelerator, and detects double fetches. It fuzzed 33 devices in QEMU and bhyve, found 110 crashes against Nyx's 44 on the shared device set, reported 61 new QEMU bugs and 5 in bhyve (9 CVEs), and has run on OSS-Fuzz against QEMU continuously since 2020.
- [ViDeZZo (Liu, Toffalini, Zhou, Payer, IEEE S&P 2023)](https://ieeexplore.ieee.org/document/10179354/) adds a lightweight grammar for intra-message dependencies and learned inter-message dependencies, finding 28 new bugs.
- [HyperPill (Bulekov et al., USENIX Security 2024)](https://www.usenix.org/system/files/usenixsecurity24-bulekov.pdf) fuzzes hypervisors from hardware-level snapshots and found 26 new bugs: 11 in QEMU, 9 in Hyper-V and 6 in macOS's Hypervisor.framework.

What ruvm takes: Morphuzz's design directly, because it only needs the qtest protocol, which ruvm implements (ruvm-accel-qtest). The same generic fuzzer runs against ruvm and against QEMU; any crash in ruvm is a bug, and any divergence in the MMIO read trace between the two for the same input is a compatibility bug (differential fuzzing). ViDeZZo-style dependency models are added for virtio queues, xHCI rings and NVMe queues, where dependencies between messages make blind fuzzing slow. Document 22 gives the harness; every CVE in section 2.3 is a seed.

## 9. Confidential computing

- AMD SEV-SNP guests are supported in QEMU since [9.1 (September 2024)](https://www.qemu.org/2024/09/03/qemu-9-1-0/) through `-object sev-snp-guest`, backed by KVM's `guest_memfd`.
- Intel TDX guests are supported since [QEMU 10.1 (August 2025)](https://www.qemu.org/2025/08/26/qemu-10-1-0/) (`-object tdx-guest`), and 10.1 also added VFIO for guest_memfd-backed confidential guests.
- IGVM (Independent Guest Virtual Machine format) loading arrived in 10.1 through `-object igvm-cfg,file=...` referenced from `-machine igvm-cfg=`, for SEV, SEV-ES, SEV-SNP and non-confidential guests. QEMU links the C API of Microsoft's Rust `igvm` crate.
- QEMU 11.0 added reset for SEV-SNP and TDX guests and the Nitro Enclave accelerator.
- Arm CCA host support is not upstream anywhere. The KVM series reached [v17 in September 2026](https://ratatoskr.run/kvmarm/2026/09/17532460/t) tracking the RMM v2.0 beta specification, and the QEMU side is an [RFC (v3, August 2026)](https://ratatoskr.run/kvm/2026/08/17461766/t).

What ruvm takes: SEV-SNP, TDX, IGVM and reset semantics in M8, matching QEMU's objects and properties exactly (`sev-snp-guest`, `tdx-guest`, `igvm-cfg`), so libvirt launch XML works unchanged. ruvm uses the `igvm` Rust crate natively instead of its C bindings. Arm CCA is implemented against the KVM uAPI only after it is merged in Linux, and the QMP and CLI surface only after QEMU merges its series, because inventing an interface QEMU later does differently would break the compat contract (new decision; document 19).

## 10. Host I/O building blocks

These are not research results, but they are the substrates the performance targets in the canon depend on, and each has a QEMU equivalent ruvm must match in behavior: io_uring (QEMU's `aio=io_uring`; ruvm-aio's native completion model on Linux), [passt](https://passt.top/) (unprivileged user-mode networking in a separate process, exposed in QEMU 11.1 as `-netdev passt,id=...,path=...,quiet=...,vhost-user=...` in `qemu-options.hx` and implemented in `net/passt.c`; ruvm implements the same netdev, and `-netdev user` keeps libslirp semantics because the guest-visible behavior differs), SPDK and VDUSE (vhost-user-blk backends and kernel-visible vDPA block devices, both reached through ruvm's vhost-user frontend), and iommufd (QEMU's `-object iommufd` backend for VFIO, which ruvm-hw-vfio supports from M8). Document 13 and document 16 cover them.

## 11. Research to document map

| Research or system | What ruvm takes | Documents |
| --- | --- | --- |
| QEMU 11.0 and 11.1 releases | Reference feature set, machine compat arrays, option and command counts | 02, 11, 23 |
| Rust in QEMU (pl011, HPET, `rust/bql`) | Concept map for QOM and VMState in Rust; MSRV policy | 04, 17, 24, 25 |
| BQL and lockless MMIO patches | Device domains, per-transaction reentrancy guard, timer benchmarks | 03, 05, 12, 21 |
| QEMU device CVEs (VENOM to CVE-2026-17588) | Regression seeds, reentrancy semantics, device-boundary panic handling | 02, 12, 19, 22 |
| rust-vmm crates | Bindings, vhost-user definitions, `GuestMemory` adapter | 05, 06, 13, 24 |
| Firecracker (NSDI 2020) | Startup budget and startup path design | 06, 11, 21 |
| Cloud Hypervisor | MSHV ioctl sequences, VFIO and vhost-user plumbing | 06, 13, 16 |
| crosvm | Optional out-of-process device mode | 19, 20 |
| libkrun | VMM as a library | 20 |
| Dragonball | Embeddable system crate | 20 |
| HQEMU (CGO 2012) | Background tier 2 compilation | 08 |
| Instrew (VEE 2021) | Region register allocation, host call/return for guest calls, first-class exclusives | 07, 08 |
| Risotto (ASPLOS 2023), Arancini (ASPLOS 2026) | IR memory model, verified fence mappings, mixed-size handling | 07, 08 |
| Rosetta 2, FEX-Emu, Box64, LATX | Host feature detection, unaligned atomic backpatching, LBT use | 08, 10 |
| Captive (ATC 2019) | Host-MMU guest translation, deferred past 1.0 | 25 |
| Learned rules (CGO 2024) | Differential validation of backend patterns | 22 |
| ARMing x86 Games (MobiSys 2025) | Tier 2 flag liveness and NZCV reuse | 08 |
| arXiv 2501.03427 | Evidence for a small tier 1 IR | 07, 21 |
| Sail, Pydrofoil, Arm ASL and MRS | Lockstep test oracles, generated system register tables | 09, 22 |
| Firecracker snapshots, REAP, FaaSnap, Catalyzer, mapped-ram | Exact mapped-ram, lazy restore, working set prefetch sidecar | 17, 21 |
| QEMU record/replay, rr | Compatible CLI and QMP, private log format | 02, 17 |
| Nyx, Morphuzz, ViDeZZo, HyperPill | qtest-based generic and differential fuzzing | 22 |
| SEV-SNP, TDX, IGVM, Nitro, Arm CCA | Exact confidential guest objects; CCA gated on upstream | 06, 19 |
| io_uring, passt, SPDK, VDUSE, iommufd | I/O substrates behind QEMU-identical options | 13, 14, 15, 16 |
