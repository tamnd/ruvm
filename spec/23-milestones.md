# 23. Milestones

Thirteen milestones, M0 through M12. Each has an exit criterion that a script can check, and an effort estimate in engineer-months with the uncertainty stated. The estimates assume a core team of six to eight engineers who know virtualization and compilers, with outside contributors picking up long tail boards and devices from M6 on.

The ordering rests on one rule: **ruvm boots a real guest from M2 onward, and every later milestone is judged by running real guests and QEMU's own test suites.** No milestone is a large body of code that has never executed a guest instruction. The second rule is that the order follows users rather than subsystems. Cloud and CI users of KVM on x86 are the largest group whose needs are well defined and whose configurations are narrow, so they come first. Developers on Apple silicon and Windows come next. Embedded board users and the long tail of guest ISAs come last. This is not because they matter less. Their surface is the widest, and by then the core that everything depends on has stopped moving.

## Summary

| Milestone | Content | Estimate (engineer-months) | Calendar with 7 engineers |
|---|---|---|---|
| M0 | Skeleton, CI, layers, vendor-qemu | 3 to 5 | month 1 |
| M1 | QOM, QAPI, QMP, memory core, main loop | 12 to 18 | months 2 to 4 |
| M2 | x86 KVM: microvm and q35 boot Linux with virtio | 18 to 26 | months 4 to 8 |
| M3 | Block layer, qcow2, qemu-img parity | 14 to 20 | months 6 to 10 (overlaps M2) |
| M4 | JIT tier 1, x86-64 and aarch64 guests on x86-64 and aarch64 hosts | 24 to 36 | months 8 to 14 |
| M5 | Migration interop with QEMU | 10 to 16 | months 10 to 14 |
| M6 | HVF, WHPX, arm virt, riscv virt | 18 to 28 | months 13 to 18 |
| M7 | linux-user and bsd-user | 12 to 18 | months 15 to 19 |
| M8 | VFIO, iommufd, vIOMMUs, confidential computing | 20 to 30 | months 17 to 23 |
| M9 | JIT tier 2 and the 2x target | 18 to 30 | months 19 to 26 |
| M10 | Long tail targets, boards, devices | 60 to 110 | months 20 to 36 |
| M11 | libvirt, OpenStack, Proxmox, distro certification | 12 to 20 | months 30 to 36 |
| M12 | 1.0 | 6 to 10 | months 34 to 38 |

The total is 227 to 367 engineer-months. For scale: QEMU has had more than 2,000 contributors, and a single release now carries more than 3,000 commits. The estimate is only credible because ruvm ports behavior QEMU already specifies in running code, tests, and schema files, rather than discovering that behavior. It also assumes the long tail in M10 is shared with outside contributors. M10's range is the widest because its size depends on how many board users show up to help, and nobody can know that in advance. Document 25 lists the scope decisions that would shrink it.

## M0: Skeleton

The workspace from document 24, with every L0 through L5 crate created as an empty shell so the layer table is real from the first commit. `cargo xtask layers`, `provenance`, and `unsafe-audit` run in CI. `vendor-qemu/` is populated from the QEMU v11.1.0 tag, and `cargo xtask upstream-sync` works end to end, including the delta report. CI runs build, clippy, rustfmt, and unit tests on Linux x86-64, Linux AArch64, macOS AArch64, and Windows x86-64. The argv[0] dispatcher exists, and `qemu-system-x86_64 --version` prints a version string in QEMU's format, because libvirt parses it.

**Exit:** CI is green on all four hosts. `qemu-system-x86_64 --version` output passes libvirt's version parser. `cargo xtask upstream-sync v11.1.0` runs on a clean checkout with an empty delta.

## M1: Object model, QAPI, QMP, memory core

`ruvm-qom` with types, interfaces, properties, composition tree, realize, and compat props (document 04). `ruvm-qapi` generating Rust types from the full vendored schema, including QMP dispatch and introspection. `ruvm-monitor` QMP server with capabilities negotiation, OOB, events, and fd passing. `ruvm-aio` with io_uring, kqueue, and IOCP reactors, timers, and bottom halves. `ruvm-mem` with regions, FlatView, RCU publication, dispatch, and dirty bitmaps (document 05). `ruvm-vmstate` encoder and decoder working on synthetic descriptions. `ruvm-accel-qtest` with the qtest protocol server, so QEMU's qtest binaries can connect.

QMP ships this early on purpose. Every later milestone is tested through QMP and qtest, so they must exist before there is anything worth testing.

**Exit:** `query-qmp-schema` output from `qemu-system-x86_64 -machine none` is byte-identical to QEMU 11.1's for the same configure options, after the documented normalization of commands whose backing subsystem is not built yet. That normalization list is checked in and must shrink to zero by M11. QEMU's `tests/qtest/qmp-test` and `qom-test` pass with `-machine none`. The FlatView property tests (random region trees compared against a reference model) pass for 10^6 cases.

## M2: x86 KVM, microvm and q35

`ruvm-accel-kvm` on x86-64 with in-kernel irqchip and split irqchip. `ruvm-target-x86` CPU models and CPUID/MSR handling for KVM (document 09), with no JIT yet. Machines `microvm` and `q35` at the 11.1 machine version (document 11). The devices a modern Linux and Windows guest needs on q35: LAPIC, IOAPIC, HPET, RTC, PIT, the ICH9 LPC and SMBus, AHCI, 16550, i8042, fw_cfg with DMA, PCIe root ports, pvpanic, and the ACPI hardware. Virtio over PCI and MMIO: net, blk, scsi, rng, balloon, console, vsock, fs (vhost-user), and the vhost kernel and vhost-user frontends (document 13). Netdevs: tap, user via passt and libslirp, vhost-user. Chardevs: stdio, socket, pty, file, mux. SeaBIOS and OVMF boot through ruvm-firmware, with ACPI tables built in Rust. Direct kernel boot including PVH. Block is the minimum needed to boot, meaning raw files on the file protocol. The rest is M3.

The first performance gates go into CI here: microvm start latency, boot to init, and RSS.

**Exit:** Fedora, Debian, Ubuntu, and Alpine current releases, plus Windows Server 2025, install and boot on q35 with OVMF and SeaBIOS. A microvm config boots a Linux kernel to init. The ACPI tables for q35 and microvm match QEMU's `tests/data/acpi` expected blobs for every variant that `bios-tables-test` covers with M2 devices. The x86 part of `kvm-unit-tests` passes at the same count as QEMU on the same host. Document 21 targets are measured and recorded, and CI gates on regressions, although the absolute targets are only required at M12.

## M3: Block layer and tools

`ruvm-block` in full for local use: the graph, permissions, drain, and graph lock. Formats: raw, qcow2 with every feature including subclusters, external data files, compression, LUKS, persistent bitmaps, and internal snapshots, plus luks, vmdk, vdi, vhdx, vpc, qed, parallels, dmg, cloop, bochs, and vvfat. Filters, throttling groups, and the file and host_device protocols on io_uring, linux-aio, and threads. NBD client and server. Block jobs: stream, commit, mirror, backup, create, and amend. Tools: qemu-img, qemu-io, qemu-nbd, and qemu-storage-daemon, with NBD, vhost-user-blk, FUSE, and VDUSE exports. The remote protocols with C dependencies (rbd, iscsi, nfs, curl, ssh, blkio; gluster is gone because QEMU 11.1 removed it) wait for M10, because their risk is in integration rather than design.

**Exit:** QEMU's iotests in the `quick` and `auto` groups for the qcow2, raw, and nbd formats pass against ruvm-img and ruvm at a rate of 100% of the tests that pass against QEMU 11.1 on the same host. Every exclusion is listed with a reason. qemu-img output and exit codes match QEMU for every subcommand on the document 14 corpus of images. Images created by either implementation pass `qemu-img check` from the other.

## M4: JIT tier 1

`ruvm-jit` IR, tier 1 translator, register allocator, TB cache, chaining, jump cache, softmmu TLB, MTTCG, and exclusive sections (documents 07 and 08). Host backends for x86-64 and aarch64, plus the interpreter. `ruvm-softfloat` passes TestFloat for every rounding mode and every target NaN rule used by x86 and Arm. Guests: x86-64 including SSE through AVX2 and the system instructions needed to boot Linux and Windows, and aarch64 including SVE2 and the EL2 and EL3 features that the virt machine and firmware use. TCG plugin ABI with the example plugins from QEMU's `contrib/plugins`. Memory model mappings for x86 guest on Arm host follow the verified schemes (document 07).

**Exit:** QEMU's `tests/tcg` for x86_64 and aarch64 pass. risu comparison against real hardware passes for the implemented instruction groups. Linux boots to a shell under TCG on q35 (x86-64 guest) and arm virt (aarch64 guest), on both host architectures. Tier 1 performance is at least 1.25x QEMU 11.1 TCG on SPEC CPU2017 intrate, as a geometric mean, in all three canonical guest and host pairs, measured with the method in document 21. The 2x target belongs to M9. The 1.25x at M4 comes from things tier 1 does without any region optimization: the inline TLB fast path, cheaper indirect branch lookup, and flag liveness within a block.

## M5: Migration interop

`ruvm-migration` with precopy, postcopy, multifd, and compression, and the tcp, unix, fd, exec, and file (mapped-ram) channels. Also savevm and loadvm to qcow2, CPR, background snapshot, and the dirty ring (document 17). VMState descriptions for every device shipped in M2 are audited field by field against QEMU's.

Interop is a milestone of its own, not a line item, because it is the property operators test first. "Can I live-migrate my existing fleet onto it and back off if it misbehaves" decides whether anyone tries ruvm in production.

**Exit:** For q35 and microvm at machine versions 10.2, 11.0, and 11.1, with the M2 device set, live migration QEMU to ruvm and ruvm to QEMU succeeds under a memory-dirtying workload and a disk and network I/O workload, with the guest's checksummed state verified after each hop. The migration functional tests from QEMU's suite that cover M2 devices pass. Downtime and total migration time are measured against QEMU on the same link and recorded.

## M6: macOS, Windows, Arm, RISC-V

`ruvm-accel-hvf` on Apple silicon including vGIC and nested virtualization (added in QEMU 11.1), and `ruvm-accel-whpx` on Windows. Machines: arm `virt` with GICv3, ITS, SMMUv3, PCIe, and ACPI, plus `sbsa-ref`, and riscv `virt` with AIA, PCIe, and ACPI. The Arm GIC in userspace and in kernel under KVM on AArch64 Linux. RISC-V guest in the JIT, including RVV and the H extension. The riscv64 host backend. UI arrives here, because desktop users arrive here: Cocoa, GTK, SDL, VNC, and D-Bus display, plus the audio backends for those hosts (document 15).

**Exit:** On macOS AArch64, a Linux aarch64 guest and Windows 11 on Arm boot under HVF with virtio-gpu and graphical output. On Windows x86-64, a Linux guest boots under WHPX. A RISC-V Linux guest boots under TCG on all tier 1 hosts. QEMU's qtest suites for aarch64 and riscv64 pass at the same rate as QEMU, restricted to the machines above. `tests/tcg` for riscv64 passes.

## M7: User mode

`ruvm-linux-user` for x86-64, aarch64, riscv64, arm, i386, ppc64le, and s390x guests first, then every other target QEMU supports in user mode. `ruvm-bsd-user` for FreeBSD hosts. binfmt_misc integration and the static per-target binaries. /proc emulation, signals, threads, the page size mismatch handling, the strace mode, and the gdbstub in user mode (document 10).

**Exit:** LTP syscall tests under user mode pass at QEMU's pass rate or better for each target in the first group, and every regression against QEMU is triaged. A Debian arm64 and riscv64 chroot on an x86-64 host builds the Debian packages from the document 10 list. linux-user runs SPEC CPU2017 intrate at no worse than QEMU linux-user speed with tier 1.

## M8: Passthrough, vIOMMU, confidential

`ruvm-hw-vfio` with both the legacy container and iommufd backends, the GPU and device quirk tables, display passthrough, vfio migration, and vfio-user. vIOMMUs: intel-iommu with scalable mode, amd-iommu, smmuv3 including accelerated nested mode, virtio-iommu, and riscv-iommu (document 16). Confidential computing: SEV, SEV-ES, and SEV-SNP with IGVM and reset, TDX with guest_memfd, and Arm CCA realms as far as Linux and QEMU support them at that point (document 19). CET virtualization. The swtpm backend with measured boot.

**Exit:** A physical NIC and an NVMe device work under vfio-pci on both backends, a GPU passes through to a Windows guest with the driver loading and a display, and vfio migration of a device that supports it works QEMU to ruvm and back. SEV-SNP and TDX guests boot with attestation reports verified by the vendor tools on the lab hardware in document 22. The QEMU functional tests for intel-iommu, smmuv3, and virtio-iommu pass.

## M9: JIT tier 2 and the 2x target

The region optimizer, guest register promotion, flag liveness across blocks, load and store forwarding within memory model rules, and the linear scan allocator over regions (documents 07 and 08). The ppc64, s390x, and loongarch64 host backends start here, and each is tier 1 quality before it replaces the interpreter on its host.

**Exit:** The document 21 JIT targets are met: a geometric mean of at least 2x QEMU 11.1 TCG on SPEC CPU2017 intrate for the three canonical pairs, in both system mode and linux-user, with the methodology from document 21. Tier 2 passes the same correctness suites as tier 1 (`tests/tcg`, risu, and the differential lockstep runner in document 22) with tier-up forced to trigger on every block.

## M10: The long tail

Every remaining guest ISA, every remaining machine and board, and every remaining device, backend, and protocol from QEMU 11.1, plus whatever QEMU added between 11.1 and the time this milestone runs (documents 09, 11, 12). This milestone is run as a queue, not a plan. The conformance matrix from document 22 lists every machine type, device type, and target, and each has a state: absent, builds, passes smoke test, passes QEMU's tests for it, passes differential testing. The milestone is done when every row is in the last state or has a documented exception accepted under the policy in document 02.

Priority inside the queue comes from signal, not taste: bug reports and download counts for the corresponding QEMU binaries in distribution popcon data, board popularity in embedded CI systems (Zephyr, U-Boot, Linux kernel CI), and requests on the tracker. The first items are the remaining x86 machines (pc, isapc, xen), ppc64 pseries and powernv, s390x, mips malta, loongarch virt, and the Arm boards that the Linux kernel CI boots (raspi, aspeed, imx, versal, and the npcm boards).

**Exit:** The conformance matrix has no row outside the final state without an accepted exception. `-device help`, `-machine help`, and `-cpu help` output for every system binary matches QEMU 11.1 plus the documented upstream syncs.

## M11: Ecosystem certification

libvirt's test suite and TCK run against ruvm binaries. The libvirt capabilities XML generated from probing ruvm matches what libvirt generates for QEMU, after removing features ruvm deliberately lacks. OpenStack Nova's libvirt driver tempest runs, Proxmox VE's qemu-server test suite, GNOME Boxes and virt-manager smoke tests, Kata Containers with the ruvm binary as its QEMU, and the Firecracker-style microVM users (document 18). Distribution packaging for Fedora, Debian, Arch, and Homebrew, with the symlink farm and firmware placement matching each distribution's QEMU packaging so that switching is a package swap.

**Exit:** libvirt's test suite passes, and its capability probing produces no unexpected differences. Tempest compute API tests pass with ruvm on the compute nodes. Kata's CI passes with ruvm substituted. The QMP schema normalization list from M1 is empty.

## M12: 1.0

Stabilization. A feature freeze of six to eight weeks with only fixes. The document 21 performance targets are verified on the reference hardware and published with raw data. A third-party security review of the device models reachable from a default q35 guest and a default virt guest (document 19). The long-term support policy is announced: ruvm 1.x tracks QEMU releases with a lag of at most one release cycle.

**Exit:** Every M exit criterion above still passes on the release candidate. Every document 21 target is met, or is published as missed with the numbers. A missed target is reported as missed, not left out. There are zero open issues labeled `miscompile`, `guest-abi`, or `migration-compat`.

## Risks to the schedule

**Tier 2 JIT slips.** The 2x target is the claim with the least precedent. HQEMU and Instrew show large gains on SPEC-like code, but with LLVM's compile latency. Arancini shows up to 5x over QEMU-based translators on its multithreaded suite, but it is a research system with a narrower scope. If M9 slips, 1.0 ships with tier 1 and publishes the measured ratio. Nothing else in the plan depends on tier 2.

**Guest-visible ABI drift.** Machine types are where "100% compatible" is most expensive, because any difference in a register, a table, or a PCI ID breaks a guest that was installed on QEMU. The mitigation is that differential testing (document 22) runs from M2 on, not at the end.

**Upstream moves.** QEMU ships two releases a year, and each one adds devices, options, and QMP commands. The sync process in document 02 turns each release into a work list. The schedule assumes about one engineer-month per upstream release for sync work, which is already inside the estimates above. If upstream changes something structural, such as the migration format or the QAPI generator's output, that estimate is wrong, and the release notes will say so.

**Staffing on the long tail.** M10 is half the project by effort, and its end date depends on outside contributors. The fallback is scope, not quality. Boards with no identifiable users can be marked unsupported under the exception policy. That is the same decision QEMU makes when it deprecates and removes boards, and ruvm follows those removals as they happen upstream.
