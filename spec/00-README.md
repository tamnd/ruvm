# Spec 2147: ruvm

A machine emulator and virtualizer written in Rust that replaces QEMU without anyone downstream noticing. Same binaries by name, same command line, same QMP schema byte for byte, same machine types, same guest-visible hardware, same disk image formats, same migration stream in both directions, same TCG plugin ABI. Underneath it is a different program: no Big QEMU Lock, a completion-based event loop, a two-tier JIT with verified memory ordering, and a crate graph where every layer can be built, tested and reused on its own.

Repo: `github.com/tamnd/ruvm`. Crate: `ruvm`, free on crates.io when checked on 30 September 2026 and not yet reserved. Binary: `ruvm`, installed with `qemu-*` symlinks. Reference: QEMU 11.1.0. Written 30 September 2026.

## Where QEMU is this year

QEMU 11.0 shipped on 22 April 2026 with a new `nitro` accelerator for AWS Nitro Enclave images, CET virtualization under KVM, reset for SEV-SNP and TDX guests, C++ TCG plugins, SME under TCG, and the removal of 32-bit host support. QEMU 11.1 shipped on 11 August 2026 with more than 3,200 commits from 285 authors, the largest release by commit count in years. The project is healthy and moving fast, which is the first thing any replacement has to plan around: the target moves twice a year.

Rust inside QEMU is still marked experimental. At the v11.1.0 tag, `rust/hw/` holds exactly two devices, the PL011 UART and the HPET, plus QOM, VMState, QAPI and BQL bindings, with an MSRV of 1.83. That work is careful and good, and it also shows the ceiling of the approach. The bindings encode "protected by the BQL" in the type system because the program around them is built on the BQL. Rust at the edges of a C program inherits the C program's concurrency model, its memory dispatch, its TCG, and its object lifetimes. Converting leaf devices one at a time will never change those.

On the other side, the Rust VMMs (Firecracker, Cloud Hypervisor, crosvm, and the rust-vmm crates under them) proved that Rust works for virtualization, and each one got there by dropping most of what QEMU does. None of them has TCG, none of them runs a q35 guest installed under QEMU, none of them speaks QMP to libvirt. Document 01 walks through all of this with sources.

The gap ruvm fills is the obvious one that nobody has taken on: all of QEMU, compatible at the byte level where it matters, rebuilt on a design that would not be possible as an incremental conversion.

## The goal, stated so it can be falsified

Four axes. Each has a number and a test that can fail.

**Compatibility.** A libvirt domain XML, a command line, or a QMP script that works with QEMU 11.1.0 works with ruvm, and every surface in the conformance matrix of document 02 is either bit-exact, behaviorally equivalent under a stated oracle, or listed as a documented divergence with one of four allowed reasons. QEMU's own qtest, iotests, functional tests and tcg tests run against ruvm unmodified. Live migration works from QEMU 11.1 to ruvm and back for every device with a VMState. The conformance dashboard in document 22 publishes the pass rate per surface with every release, and "100%" means 100% of that matrix, not a feeling.

**Performance.** Measured under the rules of document 21, against QEMU 11.1.0 built with its release flags on the same host. JIT: at least 1.25x QEMU TCG on SPEC CPU2017 intrate with tier 1 at M4, and at least 2x with tier 2 at M9. KVM microvm: process start to first guest instruction at most 15 ms, boot to init at most 110 ms. Overhead RSS at most 0.6x QEMU. With 16 vCPUs hitting different devices, at least 4x QEMU's aggregate MMIO exits per second, which is where removing the BQL shows up.

**Safety.** No `unsafe` outside crates with a declared budget, each block carrying a `SAFETY` comment that CI checks. Every guest-controlled memory access goes through `GuestPtr` and `GuestSlice`. Every device model runs under the QEMU device CVE regression corpus and continuous fuzzing from the milestone it lands in. Document 19 maps QEMU's CVE history to the classes Rust removes and the classes it does not.

**Modularity.** A new device, board, CPU model, block driver, netdev backend or accelerator is added in one crate, registered at link time, with no edits to a central list. The worked example in document 20 adds the PL031 RTC and touches nothing outside its own files plus one line in a machine. The foundation crates (event loop, virtqueue, vhost, vfio-user, JIT core) are permissively licensed and usable outside ruvm.

## Why Rust, concretely

QEMU's security advisories are dominated by memory safety bugs in device emulation: out-of-bounds indexing from guest-supplied values, use-after-free when a device is unplugged or reset in the middle of DMA, and reentrancy where a device's DMA write lands on its own MMIO region. Rust removes the first two classes by construction, and the ownership model makes the third something we can check in the type system (the device reentrancy guard in document 19). It does not remove logic bugs, and the spec never claims it does.

The ecosystem carries the boring parts now. kvm-ioctls and kvm-bindings, vfio-bindings, io-uring, rustls, zbus, gtk4-rs, gimli, object and the rest are maintained and used in production. rust-vmm has already worked out how to wrap KVM and VFIO safely.

The concurrency argument is the one that matters most for performance. Replacing the BQL with per-device lock domains in C would be a decade of subtle races. In Rust the compiler checks which state each domain can touch, and the lock ordering rules in document 03 are enforced by the types rather than by review.

## Settled decisions

**GPL-2.0-or-later for the emulator, permissive for the leaf crates.** Most device models are ports of GPL code and the result has to be GPL. The crates that contain no QEMU-derived code (ruvm-base, ruvm-aio, ruvm-sys, ruvm-virtio-queue, ruvm-vhost, ruvm-vfio-user, ruvm-jit-core and a few others) are MIT OR Apache-2.0, and `cargo xtask provenance` fails the build if a permissive crate depends on a GPL one. Document 24 has the full table.

**One binary, many names.** `ruvm` is a multi-call binary that dispatches on argv[0], so `qemu-system-x86_64`, `qemu-aarch64`, `qemu-img`, `qemu-nbd`, `qemu-storage-daemon`, `qemu-vnc` and the helpers are symlinks. The native `ruvm run` front end is sugar that expands into an ordinary QEMU configuration. Documents 18 and 24.

**Six layers, enforced.** L0 foundation, L1 core model, L2 subsystems, L3 devices, L4 machines and accelerators, L5 binaries. `cargo xtask layers` fails on any upward edge. Document 03.

**No BQL. Device lock domains instead.** Each device or tightly coupled group of devices owns a lock domain. MMIO dispatch takes the domain of the target region only, and a small control lock covers the rare global operations (hotplug, reset, machine state changes). Document 03 explains the ordering rules and document 25 records the cases where QEMU's order still has to be reproduced exactly.

**One completion-based event loop.** ruvm-aio runs on io_uring on Linux, kqueue on macOS and the BSDs, and IOCP on Windows, one reactor per thread. No tokio in the data path. Document 03.

**Memory dispatch through a sorted boundary array.** The flattened view of an address space is an array of region boundaries searched with a branch-free binary search and published through RCU, not a radix tree. Guest RAM access goes through `GuestPtr` and `GuestSlice`. Document 05.

**A two-tier JIT with verified fences.** Tier 1 is a fast template translator that compiles about as quickly as TCG and wins on cheaper softmmu paths, fewer helper calls and lazy flags. Tier 2 is an optimizing compiler for hot regions. Memory ordering for strong guests on weak hosts follows the mappings proved in Risotto (ASPLOS 2023) and Arancini (ASPLOS 2026), with each row of the table citing its lemma. The TCG plugin ABI v7 is kept exactly. Documents 07 and 08.

**Migration is QEMU's wire format.** VMState sections, field layouts, subsections and version numbers match QEMU 11.1 so that a VM moves between the two in either direction. Document 17.

**The QAPI generator is ported, not replaced.** The same schema files produce the same `query-qmp-schema` output byte for byte, plus Rust types. Document 04.

**Extensions are namespaced and off by default.** QMP extensions use the `__io.github.tamnd.ruvm_` prefix and exist only in the `ruvm` personality or with `RUVM_EXTENSIONS=1`. Properties and options use `x-ruvm-`. No machine type turns on anything QEMU does not have.

**panic=abort.** A panic in a device is a bug that ends the process with a diagnostic, the same way an assertion does in QEMU. Catching panics at the device boundary would leave state half-updated. Document 03.

**Firmware blobs are QEMU's, unmodified.** SeaBIOS, edk2, OpenSBI, SLOF and the rest ship as built by QEMU so guest-visible firmware behavior is identical. Document 11.

## The documents

| | | |
|---|---|---|
| 00 | this file | the pitch, the goal, the settled decisions |
| 01 | `01-research-2026.md` | QEMU in 2026, Rust VMMs, binary translation research, and what each one forces |
| 02 | `02-compat-contract.md` | the three tiers, every compatibility surface, the conformance matrix, upstream tracking |
| 03 | `03-architecture.md` | layers, threads, lock domains, ruvm-aio, errors, lifecycle, hot paths, unsafe policy |
| 04 | `04-object-model-and-config.md` | ruvm-qom and ruvm-qapi, properties, the ported QAPI generator |
| 05 | `05-memory.md` | address spaces, flat views, dispatch, guest memory access, IOMMU regions, hostmem |
| 06 | `06-accelerators.md` | KVM, HVF, WHPX, MSHV, NVMM, nitro, qtest, xen, and the accelerator trait |
| 07 | `07-jit-frontend-and-ir.md` | the JIT IR, guest decoders, lazy flags, the memory model |
| 08 | `08-jit-backend-and-runtime.md` | host backends, register allocation, softmmu TLB, block chaining, tier 2, plugins |
| 09 | `09-guest-targets.md` | every guest ISA, CPU models, feature bits, and how each is tested |
| 10 | `10-user-mode.md` | linux-user and bsd-user, syscalls, signals, binfmt_misc |
| 11 | `11-machines-and-firmware.md` | machine types, versioned compat properties, ACPI, device trees, firmware |
| 12 | `12-core-devices.md` | PCI, USB, storage controllers, interrupt controllers, timers, reentrancy guard |
| 13 | `13-virtio-and-vhost.md` | virtqueues, transports, every virtio device, vhost, vhost-user, vDPA, VDUSE |
| 14 | `14-block-layer-and-tools.md` | the block graph, formats, jobs, NBD, qemu-img, qemu-io, the storage daemon |
| 15 | `15-net-chardev-ui-audio.md` | netdevs, chardevs, VNC, SPICE, GTK, Cocoa, D-Bus display, audio |
| 16 | `16-vfio-iommu-cxl.md` | VFIO, iommufd, vfio-user, vIOMMUs, CXL |
| 17 | `17-migration-snapshots-replay.md` | VMState, live migration, CPR, snapshots, record and replay |
| 18 | `18-management-plane.md` | QMP, HMP, the command line, libvirt, gdbstub, tracing, guest agent |
| 19 | `19-security-and-confidential.md` | threat model, CVE classes, sandboxing, SEV-SNP, TDX, CCA, measured boot |
| 20 | `20-extensibility.md` | registries, features, out-of-process devices, stability policy, a worked example |
| 21 | `21-performance.md` | targets, hardware, statistics, and the CI regression gates |
| 22 | `22-testing.md` | QEMU's suites against ruvm, differential testing, fuzzing, the dashboard |
| 23 | `23-milestones.md` | M0 to M12, exit criteria, estimates, schedule risks |
| 24 | `24-workspace-layout.md` | the crate catalog, binaries, features, dependencies, lints, platforms |
| 25 | `25-open-questions.md` | open questions with defaults, the decision log, claims to verify |

Read 02 first, then 03. Document 02 defines what "100% compatible" means precisely enough to fail, and document 03 is the design every other document assumes.

## Milestones in one table

| Milestone | What it delivers | Engineer-months |
|---|---|---|
| M0 | Skeleton, CI, layer checks, QEMU tree vendored for tests | 3 to 5 |
| M1 | QOM, QAPI, QMP, memory core, main loop | 12 to 18 |
| M2 | x86 KVM: microvm and q35 boot Linux with virtio | 18 to 26 |
| M3 | Block layer, qcow2, qemu-img parity | 14 to 20 |
| M4 | JIT tier 1 for x86-64 and aarch64 guests and hosts, 1.25x QEMU | 24 to 36 |
| M5 | Migration interop with QEMU in both directions | 10 to 16 |
| M6 | HVF, WHPX, arm virt, riscv virt | 18 to 28 |
| M7 | linux-user and bsd-user | 12 to 18 |
| M8 | VFIO, iommufd, vIOMMUs, confidential computing | 20 to 30 |
| M9 | JIT tier 2 and the 2x target | 18 to 30 |
| M10 | Long tail of targets, boards and devices | 60 to 110 |
| M11 | libvirt, OpenStack, Proxmox, distro certification | 12 to 20 |
| M12 | 1.0 | 6 to 10 |

A real guest boots from M2 onward and every later milestone is judged by running guests and QEMU's own tests. The order follows users: KVM on x86 for cloud and CI first, then Apple silicon and Windows developers, then embedded boards and the long tail of guest ISAs.

## What this is not

Not a fork of QEMU and not a gradual conversion of it. No QEMU C code is linked into ruvm except through the few libraries QEMU itself links (libslirp, libspice-server, virglrenderer), and each of those has a note in document 24 saying why.

Not a new VMM with its own ideas about configuration. ruvm has a native front end, but everything it does can be written as a QEMU command line, and `ruvm run --print-qemu-cmdline` does exactly that.

Not better than QEMU at being QEMU where QEMU is wrong in a way guests depend on. Bug-for-bug compatibility is kept wherever a guest, a management tool or a migration stream can see the difference. Document 02 section 6 says where the line is.

Not a promise to support every board forever. ruvm follows QEMU's deprecations and removals as they happen upstream and does not revive what QEMU drops.

## Honesty about scope

227 to 367 engineer-months to 1.0, which is 19 to 31 person-years. With a core team of seven, that is roughly three years of calendar time with outside help on the long tail. The estimate is only believable because QEMU already specifies the behavior in running code, test suites and schema files. We port behavior, we do not discover it.

M10 is half the effort by itself. It is the long tail: dozens of guest ISAs, hundreds of boards, and thousands of device variants that each have a few users. The fallback there is scope, never quality. A board nobody asks for can be marked unsupported under the divergence policy, which is the same call QEMU makes when it deprecates one.

The claim with the least precedent is 2x over TCG. HQEMU and Instrew got large wins with LLVM's compile latency, and Arancini shows up to 5x on its own suite with a narrower scope. If tier 2 slips, 1.0 ships with tier 1 and publishes the measured ratio, and nothing else in the plan depends on it.

The riskiest assumption is not technical. It is that QEMU upstream keeps moving at its current pace without changing something structural, such as the migration format or the QAPI generator's output. Each upstream release costs about one engineer-month of sync work, and document 02 section 7 describes how each release becomes a work list.

## On the name

`ruvm` is Rust plus VM, four letters, easy to type. The crate name was free on crates.io on 30 September 2026. The binary installs as `ruvm` and answers to every QEMU binary name through its symlinks.
