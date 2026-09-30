# ruvm

[![ci](https://github.com/tamnd/ruvm/actions/workflows/ci.yml/badge.svg)](https://github.com/tamnd/ruvm/actions/workflows/ci.yml)

A machine emulator and virtualizer written in Rust that aims to replace QEMU without anyone downstream noticing.

Same binary names, same command line, same QMP schema byte for byte, same machine types, same guest-visible hardware, same disk image formats, same migration stream in both directions, same TCG plugin ABI. Underneath it is a different program: no Big QEMU Lock, one completion-based event loop, a two-tier JIT with verified memory ordering, and a crate graph where every layer can be built, tested and reused on its own.

This is early. The full technical design is written down in [`spec/`](spec/) before it is built, and the milestones that build it are tracked as [issues](https://github.com/tamnd/ruvm/issues?q=label%3Akind%2Fmilestone). What exists today is listed under Status below and nothing more is claimed.

## Why

QEMU is healthy and moving fast. 11.0 shipped in April 2026 with a new `nitro` accelerator, CET under KVM, reset for SEV-SNP and TDX guests and C++ TCG plugins, and 11.1 shipped in August with more than 3,200 commits from 285 authors. Any replacement has to plan around a target that moves twice a year.

Rust inside QEMU is still marked experimental. At the v11.1.0 tag the Rust device models are the PL011 UART and the HPET, plus bindings for QOM, VMState, QAPI and the BQL. That work is careful and good, and it also shows the ceiling of the approach: Rust at the edges of a C program inherits the C program's concurrency model, its memory dispatch, its TCG and its object lifetimes. Converting leaf devices one at a time will never change those.

The Rust VMMs (Firecracker, Cloud Hypervisor, crosvm and the rust-vmm crates under them) proved that Rust works for virtualization, and each one got there by dropping most of what QEMU does. None of them has TCG, none of them runs a q35 guest installed under QEMU, and none of them speaks QMP to libvirt. ruvm is the obvious project nobody has taken on: all of QEMU, compatible at the byte level where it matters, rebuilt on a design that an incremental conversion cannot reach.

## The goal, stated so it can fail

**Compatibility.** A libvirt domain, a command line or a QMP script that works with QEMU 11.1.0 works with ruvm. Every surface in the conformance matrix of [`spec/02-compat-contract.md`](spec/02-compat-contract.md) is bit-exact, behaviorally equivalent under a stated oracle, or listed as a documented divergence with one of four allowed reasons. QEMU's own qtest, iotests, functional and tcg suites run against ruvm unmodified, and live migration works from QEMU to ruvm and back.

**Performance.** Against QEMU 11.1.0 built with its release flags on the same host, under the rules of [`spec/21-performance.md`](spec/21-performance.md). At least 1.25x QEMU TCG on SPEC CPU2017 intrate with the tier 1 JIT and 2x with tier 2. A KVM microvm that reaches its first guest instruction within 15 ms and init within 110 ms, with overhead RSS at most 0.6x QEMU's.

**Safety.** No `unsafe` outside crates with a declared budget, each block carrying a `SAFETY` comment that CI checks. Guest memory is only touched through `GuestPtr` and `GuestSlice`. Every device model runs under QEMU's CVE regression corpus and continuous fuzzing from the milestone it lands in.

**Modularity.** A new device, board, CPU model, block driver, netdev or accelerator is one crate registered at link time, with no edits to a central list. The foundation crates are permissively licensed and usable outside ruvm.

## Design in one page

**Six layers, checked.** L0 foundation, L1 core model, L2 execution, L3 targets and devices, L4 machines, L5 binaries. `cargo xtask layers` fails the build on any upward edge.

**No BQL.** Each device or tightly coupled group of devices owns a lock domain. MMIO dispatch takes only the domain of the target region, and a small control lock covers the rare global operations such as hotplug and reset.

**One event loop.** `ruvm-aio` runs on io_uring on Linux, kqueue on macOS and the BSDs, and IOCP on Windows, one reactor per thread. No tokio in the data path.

**A two-tier JIT.** Tier 1 translates about as fast as TCG and wins on cheaper softmmu paths, fewer helper calls and lazy flags. Tier 2 optimizes hot regions. Fences for strong guests on weak hosts follow the mappings proved in Risotto (ASPLOS 2023) and Arancini (ASPLOS 2026).

**QEMU's wire formats.** The QAPI generator is ported so `query-qmp-schema` matches byte for byte. VMState sections match QEMU 11.1 so a VM moves between the two in either direction.

**One binary, many names.** `ruvm` dispatches on argv[0], so `qemu-system-x86_64`, `qemu-aarch64`, `qemu-img` and the rest are symlinks.

## Status

Nothing runs a guest yet. M0 is finished and released as 0.1.0: the workspace, the checks, CI, the vendored QEMU inputs and a `ruvm` binary that answers `--version` under every QEMU name. M1 is in progress. `qemu-system-*` can start the empty `none` machine with the `qtest` accelerator, the way QEMU's own tests start it, and serve QMP and the qtest protocol on sockets:

```sh
ruvm qemu-system-x86_64 -machine none -accel qtest -qmp unix:/tmp/qmp.sock,server=on,wait=off
```

The milestones, in order:

| Milestone | What it delivers |
|---|---|
| M0 | Workspace, CI, layer checks, vendored QEMU inputs, the argv[0] dispatcher |
| M1 | QOM, QAPI, QMP, the memory core, the main loop |
| M2 | x86 KVM: microvm and q35 boot Linux with virtio |
| M3 | Block layer, qcow2, qemu-img parity |
| M4 | JIT tier 1 for x86-64 and aarch64 guests and hosts |
| M5 | Live migration with QEMU in both directions |
| M6 | HVF, WHPX, arm virt, riscv virt, the UIs |
| M7 | linux-user and bsd-user |
| M8 | VFIO, iommufd, vIOMMUs, confidential computing |
| M9 | JIT tier 2 and the 2x target |
| M10 | The long tail of targets, boards and devices |
| M11 | libvirt, OpenStack, Proxmox and distribution packaging |
| M12 | 1.0 |

The minor version counts finished milestones: 0.1.0 is the release where M0 closes, 0.2.0 where M1 closes, and so on. Patch releases come whenever enough has landed to be worth a tag. [`CHANGELOG.md`](CHANGELOG.md) says what each one contains.

## The specification

| | |
|---|---|
| [00](spec/00-README.md) | the pitch, the goal, the settled decisions |
| [01](spec/01-research-2026.md) | QEMU in 2026, Rust VMMs, binary translation research |
| [02](spec/02-compat-contract.md) | what "100% compatible" means, precisely enough to fail |
| [03](spec/03-architecture.md) | layers, threads, lock domains, the event loop, errors |
| [04](spec/04-object-model-and-config.md) | QOM and QAPI in Rust |
| [05](spec/05-memory.md) | address spaces, flat views, dispatch, guest memory access |
| [06](spec/06-accelerators.md) | KVM, HVF, WHPX, MSHV, NVMM, nitro, Xen, qtest |
| [07](spec/07-jit-frontend-and-ir.md) | the JIT IR, guest decoders, the memory model |
| [08](spec/08-jit-backend-and-runtime.md) | host backends, the softmmu TLB, tier 2, plugins |
| [09](spec/09-guest-targets.md) | every guest ISA and CPU model |
| [10](spec/10-user-mode.md) | linux-user and bsd-user |
| [11](spec/11-machines-and-firmware.md) | machine types, ACPI, device trees, firmware |
| [12](spec/12-core-devices.md) | PCI, USB, storage controllers, interrupt controllers, timers |
| [13](spec/13-virtio-and-vhost.md) | virtio, vhost, vhost-user, vDPA, VDUSE |
| [14](spec/14-block-layer-and-tools.md) | the block graph, formats, jobs, NBD, qemu-img |
| [15](spec/15-net-chardev-ui-audio.md) | netdevs, chardevs, VNC, SPICE, GTK, audio |
| [16](spec/16-vfio-iommu-cxl.md) | VFIO, iommufd, vfio-user, vIOMMUs, CXL |
| [17](spec/17-migration-snapshots-replay.md) | migration, CPR, snapshots, record and replay |
| [18](spec/18-management-plane.md) | QMP, HMP, the command line, libvirt, gdbstub |
| [19](spec/19-security-and-confidential.md) | threat model, sandboxing, SEV-SNP, TDX, CCA |
| [20](spec/20-extensibility.md) | registries, features, out-of-process devices |
| [21](spec/21-performance.md) | targets, method, regression gates |
| [22](spec/22-testing.md) | QEMU's suites against ruvm, differential testing, fuzzing |
| [23](spec/23-milestones.md) | M0 to M12 with exit criteria and estimates |
| [24](spec/24-workspace-layout.md) | the crate catalog, binaries, features, lints |
| [25](spec/25-open-questions.md) | open questions and the decision log |

Read 02 first, then 03.

## License

The emulator is GPL-2.0-or-later, because most device models and the machine types are ports of QEMU's GPL code and the result has to be GPL. See [`LICENSE-GPL-2.0`](LICENSE-GPL-2.0).

The crates that contain no QEMU-derived code are MIT OR Apache-2.0, at your option, so that rust-vmm, Cloud Hypervisor, Firecracker and anyone else can use them. See [`LICENSE-MIT`](LICENSE-MIT) and [`LICENSE-APACHE`](LICENSE-APACHE). Each crate's `Cargo.toml` says which one it is, and `cargo xtask provenance` fails the build if a permissive crate ever depends on a GPL one.

QEMU is a trademark of Fabrice Bellard. ruvm is not affiliated with the QEMU project.
