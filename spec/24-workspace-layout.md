# 24. Workspace layout

This document is the map of the repository: every crate, what layer it sits in, what license it carries, what it may depend on, and how the binaries are assembled from it. It is the document to read before adding a crate, and the document `cargo xtask layers` and `cargo xtask provenance` enforce. When this document and the code disagree, the code is wrong until someone changes this document in the same pull request.

## Why so many crates

QEMU is one Meson project with one global namespace. Any `.c` file can include any header and call any non-static function, and in practice many do. The consequence is that the dependency graph of QEMU is whatever the linker accepts, which is almost everything. You cannot build "the block layer" without dragging in the main loop, QOM, QAPI, the monitor, and enough of the system emulator to satisfy link errors, which is why `qemu-img` links against code it never runs.

ruvm splits along the seams that already exist conceptually in QEMU and makes them real. The rules are enforced by the compiler (a crate cannot use a crate it does not depend on) and by `xtask` (a crate cannot depend on a crate in a higher layer). The cost is some boilerplate and some care in trait design at crate boundaries. The benefits are concrete. `ruvm-img` builds without any accelerator, JIT, or device crate, so it compiles in a fraction of the time and its binary contains nothing it does not use. A microvm-only build has no USB, no VGA, and no JIT, which shrinks both the binary and the attack surface. A device crate can be tested against a mock memory and interrupt layer without booting a machine. Incremental build times stay tolerable because a change in `ruvm-hw-usb` does not rebuild `ruvm-block`.

The granularity rule is: one crate per family of things that share code, not one crate per device. `ruvm-hw-usb` holds every USB host controller and every emulated USB device, because they share the USB core. Splitting further buys nothing and costs compile time in link and codegen unit overhead.

## Layers

There are six layers, L0 through L5. A crate in layer N may depend on crates in layers 0 through N, including other crates in its own layer, as long as there is no cycle. It may never depend on a crate in a higher layer. Dependencies that need to go upward (a device needs to tell the machine to reset, a block job needs to emit a QMP event) go through traits defined in the lower layer and implemented in the higher one, or through the event and notifier mechanisms in `ruvm-base` and `ruvm-qom`.

```
L5  system       ruvm-system  ruvm-monitor  ruvm-gdbstub  ruvm-cli  tools
L4  machines     ruvm-machine-*  ruvm-firmware
L3  targets,     ruvm-target-*  ruvm-hw-*  ruvm-block  ruvm-net  ruvm-chardev
    devices,     ruvm-ui  ruvm-audio  ruvm-migration  ruvm-linux-user
    backends     ruvm-bsd-user
L2  execution    ruvm-hw-core  ruvm-accel  ruvm-accel-*  ruvm-jit-core  ruvm-jit  ruvm-jit-*
                 ruvm-softfloat  ruvm-decode  ruvm-plugin
L1  core model   ruvm-qom  ruvm-qapi  ruvm-mem  ruvm-hostmem  ruvm-vmstate  ruvm-trace
                 ruvm-crypto
L0  foundation   ruvm-base  ruvm-aio  ruvm-sys  ruvm-virtio-queue  ruvm-vhost
                 ruvm-vfio-user
```

`cargo xtask layers` reads a table in `xtask/layers.toml` that assigns each crate a layer and fails if any edge in `cargo metadata` goes upward. It also fails if a crate is missing from the table, so nobody can add a crate without deciding where it lives.

## Crate catalog

The tables below list every crate planned for 1.0. "Lic" is the license: G is GPL-2.0-or-later, P is MIT OR Apache-2.0. The reason for the split is in document 00 and document 02: behavioral compatibility requires porting QEMU logic, and QEMU is GPL-2.0, so anything carrying ported logic is GPL. Crates with no QEMU-derived logic stay permissive so that rust-vmm, Cloud Hypervisor, Firecracker and others can use them.

### L0 foundation

| Crate | Lic | Contents |
|---|---|---|
| ruvm-base | P | Error type and error classes, bit and field helpers, intrusive lists, the epoch based RCU used for FlatView and TB cache publication, `Notifier` lists, `Timer` wheel data structure, tracing macros that compile to nothing when disabled, the `SAFETY` lint helpers. |
| ruvm-aio | P | The event loop: one reactor per thread, completion based on io_uring (Linux), kqueue (macOS, FreeBSD, NetBSD, OpenBSD), IOCP (Windows). Timers, bottom halves, event notifiers, fd handlers, a small future executor for block and network code. No tokio in the data path. |
| ruvm-virtio-queue | P | Split and packed virtqueue processing, descriptor chain walking, event index and notification suppression, independent of the device and transport (document 13). |
| ruvm-vhost | P | vhost kernel, vhost-user frontend protocol, vhost-vdpa and VDUSE plumbing (document 13). |
| ruvm-vfio-user | P | The vfio-user protocol, client and server side, written in Rust so libvfio-user is not linked (document 16). |
| ruvm-sys | P | Thin host bindings not covered by existing crates: KVM additions newer than kvm-bindings, HVF and Virtualization framework FFI, WHPX and MSHV FFI, iommufd, userfaultfd, memfd and guest_memfd, vfio extras, macOS MAP_JIT helpers. Everything here is `unsafe` at the boundary and safe above it. |

### L1 core model

| Crate | Lic | Contents |
|---|---|---|
| ruvm-qom | G | The object model: `Object`, `TypeInfo`, interfaces, properties, `child<>` and `link<>`, the composition tree rooted at `/`, realize and unrealize, user creatable objects, class and global properties, compat property arrays. Link time type registration through `linkme` distributed slices. Derive macros live in `ruvm-qom-macros`. |
| ruvm-qapi | G | Build time QAPI schema parser and code generator reading a vendored copy of `qapi/*.json`, generated Rust types with serde, the keyval and QemuOpts visitors for command line input, QMP command dispatch tables, introspection identical to `query-qmp-schema`. Codegen lives in `ruvm-qapi-gen`. |
| ruvm-mem | P | `MemoryRegion`, `AddressSpace`, `FlatView`, sorted boundary array dispatch, RCU publication, listeners, dirty bitmaps, RamBlock bookkeeping, IOMMU regions, `GuestPtr` and `GuestSlice`, DMA helpers. No dependency on ruvm-qom. The access splitting rules are written from the documented contract and checked by differential tests, so no QEMU code is ported and the crate stays permissive (document 05). |
| ruvm-hostmem | G | The QOM memory backend types (memory-backend-ram, -file, -memfd, -shm, -epc) and thread-context, with QEMU's property names and error messages. |
| ruvm-mem-vmm | P | Adapter implementing rust-vmm's vm-memory `GuestMemory` and `Bytes` traits over ruvm address spaces, so rust-vmm's virtio-queue and vhost-user-backend can run against ruvm memory as test oracles. |
| ruvm-crypto | G | The equivalent of QEMU's `crypto/`: cipher modes, IV generators, hashes, PBKDF2, `secret` objects, TLS credentials, LUKS header handling. Primitives come from RustCrypto and rustls; no OpenSSL, gnutls or nettle (document 14). |
| ruvm-vmstate | G | VMState descriptions, the derive macro, the QEMU compatible stream encoder and decoder, subsection and version handling. |
| ruvm-trace | G | Parser for QEMU `trace-events` files, generated trace points with the same names and arguments, backends mapping onto the `tracing` crate, simple trace binary format, dtrace and ftrace glue. |

### L2 execution

| Crate | Lic | Contents |
|---|---|---|
| ruvm-hw-core | G | The qdev equivalent: `Device`, buses, qdev properties, GPIO and IRQ lines, clocks, three phase `Resettable`, hotplug handlers, fw_cfg core, the device lock domain machinery. |
| ruvm-accel | G | `Accel` and `Vcpu` traits, vCPU thread management, kick and exit request, lazy register sync, the accelerator registry. |
| ruvm-accel-kvm | G | KVM on x86, Arm, RISC-V, s390x, PowerPC and LoongArch (QEMU has removed MIPS KVM support). Uses kvm-ioctls and kvm-bindings from rust-vmm plus ruvm-sys for newer ioctls. |
| ruvm-accel-hvf | G | Hypervisor.framework on macOS, x86 and Arm. |
| ruvm-accel-whpx | G | Windows Hypervisor Platform. |
| ruvm-accel-mshv | G | Microsoft Hypervisor on Linux (`/dev/mshv`). |
| ruvm-accel-nvmm | G | NetBSD NVMM. |
| ruvm-accel-xen | G | Xen HVM and the device model side of Xen, plus the KVM based Xen emulation mode. |
| ruvm-accel-nitro | G | AWS Nitro Enclaves accelerator added in QEMU 11.0. |
| ruvm-accel-qtest | G | The qtest accelerator and the qtest protocol server. |
| ruvm-jit-core | P | IR types, builder, verifier, tier 1 optimizer, liveness, tier 2 region SSA and passes, fence placement engine, register allocators (documents 07 and 08). |
| ruvm-jit | G | The runtime ported from `accel/tcg`: TB cache, jump cache, block chaining, softmmu TLB runtime, the cpu_exec loop, exclusive sections, shared helpers. |
| ruvm-jit-x86_64 | G/P | Host backend for x86-64. |
| ruvm-jit-aarch64 | G/P | Host backend for AArch64. |
| ruvm-jit-riscv64 | G/P | Host backend for RV64GC plus Zba, Zbb, Zbs, Zicond, V where present. |
| ruvm-jit-ppc64 | G/P | Host backend for ppc64le. Post M4. |
| ruvm-jit-s390x | G/P | Host backend for s390x. Post M4. |
| ruvm-jit-loongarch64 | G/P | Host backend for LoongArch64. Post M4. |
| ruvm-jit-interp | P | Portable IR interpreter, the TCI equivalent, used on hosts without a native backend and as a reference oracle in tests. |
| ruvm-softfloat | G | Bit exact port of `fpu/softfloat*.c`, including every target's NaN propagation, default NaN, flush to zero, and exception flag rules. |
| ruvm-decode | P | Build time decoder generator that reads QEMU `decodetree` `.decode` files and emits Rust match trees. Ships as a library used from `build.rs`; the generated code takes the license of the target crate that owns the `.decode` file. |
| ruvm-plugin | G | Host for TCG plugins through the `qemu-plugin.h` C ABI, loading `.so`, `.dylib` and `.dll` plugins unmodified. |

Host backends marked G/P are GPL where the instruction encoding helpers and patch sequences are ported from `tcg/<host>/`, and dual licensed where written fresh; `cargo xtask provenance` tracks this per module (document 07).

### L3 targets, devices, backends

Guest ISA crates, one per QEMU target directory. Each implements `GuestArch` and depends on `ruvm-jit-core`, `ruvm-jit`, `ruvm-softfloat`, `ruvm-decode` (build), `ruvm-hw-core`, and `ruvm-vmstate`.

| Crate | QEMU source it tracks |
|---|---|
| ruvm-target-x86 | target/i386 (i386, x86_64) |
| ruvm-target-arm | target/arm (arm, aarch64) |
| ruvm-target-riscv | target/riscv (riscv32, riscv64, both endiannesses) |
| ruvm-target-ppc | target/ppc (ppc, ppc64) |
| ruvm-target-s390x | target/s390x |
| ruvm-target-mips | target/mips (mips, mipsel, mips64, mips64el, and the n32 ABIs in user mode) |
| ruvm-target-loongarch | target/loongarch |
| ruvm-target-sparc | target/sparc (sparc, sparc64) |
| ruvm-target-m68k | target/m68k |
| ruvm-target-alpha | target/alpha |
| ruvm-target-hppa | target/hppa (32 and 64 bit) |
| ruvm-target-sh4 | target/sh4 |
| ruvm-target-microblaze | target/microblaze |
| ruvm-target-openrisc | target/openrisc |
| ruvm-target-xtensa | target/xtensa, including the core configuration overlays |
| ruvm-target-tricore | target/tricore |
| ruvm-target-rx | target/rx |
| ruvm-target-avr | target/avr |
| ruvm-target-hexagon | target/hexagon, including the idef-parser generated semantics |

The exact list tracks what QEMU 11.1 ships. Document 09 records the verified list and any targets QEMU has removed in recent releases; a removed target is not implemented.

Device crates. Each holds a family and depends on `ruvm-hw-core`, `ruvm-mem`, `ruvm-vmstate`, and where needed on backend crates through traits.

| Crate | Contents |
|---|---|
| ruvm-hw-intc | 8259, IOAPIC, LAPIC and x2APIC, GICv2, GICv3, GICv4 and ITS, PLIC, APLIC, IMSIC, XICS, XIVE and XIVE2, s390 FLIC, and the board specific controllers. |
| ruvm-hw-timer | PIT, HPET, MC146818 RTC, Arm generic timer glue, ACLINT, and board timers. |
| ruvm-hw-pci | PCI and PCIe core, host bridges, root ports, switches, bridges, SHPC, PCIe native hotplug, AER, SR-IOV, ARI, ATS, PASID capability plumbing. |
| ruvm-hw-usb | UHCI, OHCI, EHCI, xHCI, USB core, emulated USB devices, usbredir and host passthrough glue. |
| ruvm-hw-char | 16550, pl011, board UARTs, parallel port, virtio independent console glue. |
| ruvm-hw-display | VGA family, cirrus, bochs-display, ramfb, QXL, board framebuffers, EDID generation. |
| ruvm-hw-audio | HDA codec and controller, AC97, SB16, ES1370, Gravis, board audio. |
| ruvm-hw-input | PS/2, i8042, virtio independent HID glue, board keypads and touch controllers. |
| ruvm-hw-net | e1000, e1000e, igb, rtl8139, pcnet, ne2000, vmxnet3, board NICs. |
| ruvm-hw-storage | IDE and AHCI, LSI53C895A, megasas, mptsas, NVMe including ZNS and SR-IOV, SD and eMMC, UFS, floppy, SCSI core and SCSI disk and cd. |
| ruvm-hw-virtio | Every virtio device and transport except virtio-gpu (in ruvm-hw-display) and virtio-snd (in ruvm-hw-audio), built on ruvm-virtio-queue and ruvm-vhost. |
| ruvm-vhost-backends | vhost-user backends ruvm ships as separate processes (blk, scsi, gpu, input, a net bridge for tests, vsock, rng, snd, rtc), reusing the in-process device models. gpio, i2c, spi, scmi and CAN backends are left to rust-vmm's vhost-device project (document 13). |
| ruvm-hw-vfio | VFIO PCI, platform, AP and CCW, legacy container and iommufd backends, the quirk tables, vfio-user client. |
| ruvm-hw-iommu | intel-iommu, amd-iommu, smmuv3, virtio-iommu, riscv-iommu. |
| ruvm-hw-tpm | TPM TIS (ISA and sysbus), TPM CRB, the SPAPR vTPM, and the swtpm and passthrough backends. |
| ruvm-hw-cxl | CXL type 3 devices, switches, host bridges, DCD, mailbox. |
| ruvm-hw-acpi | ACPI hardware (PM timer, GPE, CPU and memory hotplug registers, GED). Table building lives in ruvm-firmware. |
| ruvm-hw-i2c | I2C and SMBus controllers and devices. |
| ruvm-hw-ssi | SPI and SSI controllers and flash devices. |
| ruvm-hw-misc | IPMI, watchdogs, pvpanic, pvpanic-pci, vmcoreinfo, ivshmem, edu, testdev, fw_cfg devices per arch, and everything that does not fit elsewhere. |

Board specific SoC devices (Aspeed, NXP i.MX, Raspberry Pi, Xilinx, Allwinner, Nuvoton, STM32 and so on) live in their machine crate in L4 when they are only used by one family, and move down to an `ruvm-hw-*` crate the first time a second family needs them.

Backend and subsystem crates in L3.

| Crate | Contents |
|---|---|
| ruvm-block | The block graph, permissions, drain, graph lock, every format and protocol driver, filters, jobs, dirty bitmaps, throttling, exports. Formats with heavy optional dependencies (rbd, iscsi, nfs, curl, ssh, blkio) are behind cargo features and build as modules (document 20). There is no gluster driver because QEMU 11.1 removed it. |
| ruvm-net | Netdev backends: user (libslirp compatible and passt), tap, bridge helper, socket, stream, dgram, vde, l2tpv3, vhost-user, vhost-vdpa, af-xdp, vmnet, netmap, hubs, and net filters. |
| ruvm-chardev | Every chardev backend and the mux. |
| ruvm-ui | Console core, VNC, SPICE glue, GTK, SDL2, Cocoa, D-Bus display, curses, egl-headless, keymaps, clipboard. |
| ruvm-audio | Audio core, mixing engine, and every audio backend. |
| ruvm-migration | Migration state machine, channels, precopy, postcopy, multifd, compression, CPR, savevm and loadvm, background snapshot, fast local snapshot and restore, COLO. |
| ruvm-replay | Record and replay, icount integration, reverse debugging support. |
| ruvm-user-common | Shared by linux-user and bsd-user: guest address space and page flags, the mmap engine including page size mismatch, safe_syscall as naked functions per host, host signal entry, user mode JIT glue (document 10). |
| ruvm-linux-user | Linux user mode: loader, syscall translation, signals, threads, /proc emulation. |
| ruvm-bsd-user | BSD user mode for FreeBSD hosts. |

### L4 machines

| Crate | Contents |
|---|---|
| ruvm-machine-x86 | pc (i440FX), q35, microvm, isapc, nitro, xenfv and xenpv. |
| ruvm-machine-arm | virt, sbsa-ref, and every Arm board QEMU 11.1 ships, grouped in modules by vendor. |
| ruvm-machine-riscv | virt, spike, sifive_u, sifive_e, microchip-icicle-kit, shakti_c, opentitan, K230 and the rest. |
| ruvm-machine-ppc | pseries, powernv, mac99, g3beige, ppce500, the 40x and 44x boards, pegasos2, amigaone. |
| ruvm-machine-s390x | s390-ccw-virtio. |
| ruvm-machine-mips | malta, the Loongson boards, boston, and the rest. |
| ruvm-machine-loongarch | virt. |
| ruvm-machine-other | Boards for the remaining targets, one module per target. |
| ruvm-firmware | Firmware discovery and descriptor JSON, the bundled blob manifest, fw_cfg file builders, the ACPI AML builder and tables, SMBIOS, device tree generation, IGVM loading, direct kernel boot helpers. |

### L5 system and tools

| Crate | Contents |
|---|---|
| ruvm-system | The `vl.c` equivalent: option table, parsing, config assembly, machine creation, main loop wiring, runstate machine, shutdown and reset. |
| ruvm-monitor | QMP server, HMP, fd passing, event rate limiting, OOB. |
| ruvm-gdbstub | GDB remote protocol server for system and user mode. |
| ruvm-cli | The multi-call binary `ruvm`, argv[0] dispatch, the native `ruvm run` and `ruvm img` front ends. |
| ruvm-img | qemu-img. |
| ruvm-io | qemu-io. |
| ruvm-nbd | qemu-nbd. |
| ruvm-storage-daemon | qemu-storage-daemon. |
| ruvm-ga | qemu-ga, built as a separate small binary with its own minimal dependency set because it runs inside guests. |
| ruvm-helpers | qemu-bridge-helper, qemu-pr-helper, qemu-vmsr-helper, qemu-edid, qemu-keymap, elf2dmp, and qemu-vnc, the standalone VNC server over the D-Bus display that is new in QEMU 11.1 (document 15). |

## Repository tree

```
ruvm/
  Cargo.toml              workspace, shared lints, profiles
  rust-toolchain.toml     pinned stable toolchain
  deny.toml               cargo-deny: licenses, advisories, duplicate versions
  supply-chain/           cargo-vet audits and imports
  xtask/                  layers, provenance, upstream sync, conformance runner
  vendor-qemu/            pinned upstream QEMU inputs used at build time
    qapi/                 qapi/*.json
    decode/               target/*/*.decode
    trace-events/         every trace-events file, path preserved
    hx/                   qemu-options.hx, hmp-commands.hx, hmp-commands-info.hx
    acpi-expected/        tests/data/acpi blobs
    targets/              configs/targets/*.mak, the list of system and user mode targets
    MANIFEST              the SHA-256 of every file above, checked by cargo xtask vendor-check
    UPSTREAM              the QEMU commit these files come from
  crates/
    base/ aio/ sys/ virtio-queue/ vhost/ vfio-user/
    qom/ qom-macros/ qapi/ qapi-gen/ mem/ hostmem/ mem-vmm/ crypto/ vmstate/ vmstate-macros/ trace/
    hw-core/ accel/ accel-kvm/ ... jit-core/ jit/ jit-x86_64/ ... softfloat/ decode/ plugin/
    target-x86/ target-arm/ ...
    hw-intc/ hw-timer/ ... vhost-backends/
    block/ net/ chardev/ ui/ audio/ migration/ replay/ user-common/ linux-user/ bsd-user/
    machine-x86/ machine-arm/ ... firmware/
    system/ monitor/ gdbstub/ cli/ img/ io/ nbd/ storage-daemon/ ga/ helpers/
  pc-bios/                QEMU's firmware blobs from the pinned tag, unmodified
  tests/
    qtest-compat/         harness running QEMU's tests/qtest binaries against ruvm
    iotests-compat/       harness running QEMU's iotests against ruvm-img and ruvm
    functional/           ruvm's own functional tests plus QEMU's, adapted
    diff/                 differential runners against a pinned QEMU build
    guests/               guest image manifests (downloaded, hashed, not stored)
  fuzz/                   cargo-fuzz targets
  bench/                  benchmark harness and baselines (document 21)
  docs/
```

`vendor-qemu/` deserves a note. ruvm consumes several QEMU source files as data at build time: the QAPI schema, decodetree files, trace-events, the `.hx` option and command tables, the expected ACPI blobs, and the target configurations. These are copied, not submoduled, so a build never needs network access and so the exact upstream commit is one file (`UPSTREAM`) that code review can see. `cargo xtask upstream-sync <tag>` refreshes them from a QEMU checkout and produces a report of what changed (new QMP commands, new options, new decode patterns, new trace points) that becomes the work list for that sync. Document 02 describes the policy around those syncs.

## Binaries and argv[0] dispatch

The release artifact is one binary, `ruvm`, plus `ruvm-ga`. The installer creates symlinks: `qemu-system-x86_64`, `qemu-system-aarch64` and every other system target, `qemu-x86_64`, `qemu-aarch64` and every other user mode target, `qemu-img`, `qemu-io`, `qemu-nbd`, `qemu-storage-daemon`, `qemu-pr-helper`, `qemu-bridge-helper`, `qemu-vmsr-helper`, `qemu-vnc`, `qemu-edid`, `qemu-keymap`, `elf2dmp`. `ruvm-cli` looks at the basename of argv[0], strips a known suffix list (`.exe` on Windows), and dispatches.

A single binary is chosen over one binary per target for three reasons. Distributions ship dozens of `qemu-system-*` binaries today and most of their bytes are duplicated device and core code; one binary with dispatch makes that shared. Startup cost does not change measurably, because the dispatch is a string compare and every target's code is only paged in when used. And libvirt and other tools that search `PATH` for `qemu-system-*` names work unchanged.

There are two exceptions. `qemu-<arch>` user mode binaries used with binfmt_misc are frequently copied into containers and chroots, where a 60 MB system emulator is unwelcome, so the build also produces a second multi-call binary containing only the user mode personalities, statically linked against musl, with a size target under 25 MB for all 33 linux-user personalities together (document 10). `ruvm binfmt` registers it with binfmt_misc using QEMU's flags. QEMU ships one static binary per target instead; this is a packaging difference, not a behavioral one. And `ruvm-ga` stays separate because it runs inside guests, often on minimal images, and must stay small.

## Cargo features and build profiles

Features exist to cut binaries down, not to change behavior. A feature that is enabled must behave exactly like the corresponding QEMU build option; a disabled feature makes the corresponding device, backend, or target absent, and `-device help`, `query-qmp-schema` and friends report it absent the same way a QEMU build without that option would.

Top level features on `ruvm-cli`:

| Feature | Meaning |
|---|---|
| `target-<arch>` | One per guest ISA, system and user mode both. |
| `system`, `user` | System emulation and user mode emulation. |
| `accel-kvm`, `accel-hvf`, `accel-whpx`, `accel-mshv`, `accel-nvmm`, `accel-xen`, `accel-nitro`, `accel-tcg` | Accelerators. `accel-tcg` pulls in `ruvm-jit` and the host backend for the build target. |
| `block-<proto>` | Optional block protocols with external dependencies: rbd, iscsi, nfs, curl, ssh, blkio. |
| `ui-<name>` | gtk, sdl, cocoa, spice, dbus, curses, opengl. |
| `audio-<name>` | pa, pipewire, alsa, oss, coreaudio, dsound, jack, sdl, spice, dbus. |
| `net-<name>` | slirp, passt, vde, netmap, af-xdp, vmnet. |
| `modules` | Build block, UI and audio backends as loadable modules instead of statically (document 20). |
| `plugins` | TCG plugin support. |
| `profile-full` | Everything the host supports. The default for distribution builds. |
| `profile-microvm` | x86 and Arm, KVM and HVF only, microvm and virt machines, virtio devices, no legacy devices, no JIT, no UI. |
| `profile-tcg-dev` | All targets, TCG only, no UI, for JIT development. |
| `profile-tools` | Only the tools. |

Cargo profiles: `dev` with `opt-level = 1` for the JIT and softfloat crates (debug builds of an emulator at `opt-level = 0` are too slow to boot anything, and a test suite that takes an hour in debug does not get run), `release` with thin LTO and `codegen-units = 16`, and `dist` with fat LTO, `codegen-units = 1`, and `panic = "abort"`. The `dist` profile is what benchmarks in document 21 measure. Profile guided optimization with a boot and SPEC training run is applied to `dist` on Linux x86-64 and AArch64 from M9 on, and the benchmark report states whether PGO was used.

`panic = "abort"` in release is deliberate. A panic inside a device model means an invariant broke; unwinding across a vCPU thread that holds a device lock and guest state halfway through an update does not produce a usable VM. The VM exits with an error that names the device, the panic message, and a backtrace, which is what QEMU does on `abort()` and what management software already handles.

## Dependency policy

External crates are welcome where they are mature, small in surface, and license compatible. The approved set at the start:

| Area | Crates |
|---|---|
| Virtualization | kvm-ioctls, kvm-bindings, vfio-bindings, vfio-ioctls, virtio-bindings, vm-memory (through the ruvm-mem-vmm adapter, see below); rust-vmm's vhost, vhost-user-backend and virtio-queue are dev-dependencies only, used as differential test oracles, linux-loader, vmm-sys-util |
| Serialization | serde, serde_json |
| Async IO | io-uring (the crate), windows-sys, libc, rustix |
| Compression and crypto | zstd, flate2 (zlib backend), lz4_flex, aes, xts-mode, sha2, hmac, pbkdf2, argon2 for LUKS |
| Codegen | proc-macro2, quote, syn |
| Concurrency | crossbeam-utils, parking_lot, arc-swap |
| Registration | linkme |
| Testing | proptest, loom, shuttle, criterion |
| Tracing | tracing, tracing-subscriber |
| D-Bus | zbus, behind the `ui-dbus` feature of ruvm-ui, for `-display dbus` |
| macOS UI | objc2, objc2-foundation, objc2-app-kit, objc2-core-graphics, objc2-core-foundation, dispatch2, behind the `ui-cocoa` feature of ruvm-ui, for `-display cocoa` |

vm-memory appears only through `ruvm-mem-vmm`, the adapter that implements its traits over ruvm address spaces. ruvm ships its own ring and vhost crates (ruvm-virtio-queue and ruvm-vhost, document 13) because they need QEMU's exact VMState layouts and legacy transport quirks, but the adapter lets rust-vmm code run against ruvm memory in tests and lets our permissive crates interoperate with rust-vmm projects. The VMM itself does not use vm-memory as its guest memory model; QEMU's `MemoryRegion` semantics (priorities, aliases, IOMMU regions, per-device address spaces, MMIO dispatch) are a superset of what vm-memory models, and adapting one to the other in the hot path costs more than writing `ruvm-mem`. Document 05 has the detail.

Rules: every dependency goes through `cargo deny` for license and advisory checks, `cargo vet` for audit status, and a size check that fails CI if the `profile-microvm` binary grows by more than 2% in one pull request without a label acknowledging it. C library dependencies (libslirp, spice-server, gtk, libiscsi, librbd, libnfs, libssh, libcurl, libblkio, virglrenderer, libusb, usbredir, pipewire, pulseaudio) are allowed only behind features and only where QEMU uses the same library, because reimplementing SPICE or Ceph's client is not in scope. Each one has a `ruvm-sys`-style binding crate with its own unsafe budget.

## Lints and unsafe

Workspace level lints: `unsafe_op_in_unsafe_fn = "deny"`, `clippy::undocumented_unsafe_blocks = "deny"`, `missing_safety_doc = "deny"`, `rust_2024_compatibility = "warn"`. Each crate declares an unsafe budget in its `Cargo.toml` metadata: the number of `unsafe` blocks it is allowed. `cargo xtask unsafe-audit` counts them with a syn walk and fails if a crate exceeds its budget. Raising a budget needs review from someone on the security list in document 19. The expected shape is that `ruvm-sys`, `ruvm-jit-*` backends, `ruvm-mem` guest access, `ruvm-aio`, and `ruvm-plugin` hold nearly all of the unsafe code, and device crates hold almost none, with a target of zero for every `ruvm-hw-*` crate except `ruvm-hw-vfio`.

## xtask commands

| Command | What it does |
|---|---|
| `cargo xtask layers` | Checks the layer rule against `cargo metadata`. |
| `cargo xtask provenance` | Checks that no permissive crate depends on a GPL crate, and that every file in a permissive crate carries the permissive SPDX header. |
| `cargo xtask unsafe-audit` | Counts unsafe blocks per crate against budgets. |
| `cargo xtask upstream-sync <tag>` | Refreshes `vendor-qemu/` and prints the delta report. |
| `cargo xtask conformance [suite]` | Runs QEMU's qtest, iotests, functional, tcg and QAPI suites against the current build and updates the conformance matrix (document 22). |
| `cargo xtask diff <suite>` | Runs differential tests against the pinned reference QEMU build. |
| `cargo xtask bench [area]` | Runs the benchmark suites from document 21 and compares against stored baselines. |
| `cargo xtask firmware` | Copies `pc-bios/` from the pinned QEMU tag and checks every blob's hash against it. ruvm builds no firmware of its own (document 11). |
| `cargo xtask dist` | Produces release artifacts, symlink farm, man pages generated from the option tables. |

## Platform matrix

| Host | Accelerators | JIT backend | Status |
|---|---|---|---|
| Linux x86-64 | KVM, MSHV, Xen, nitro, TCG | x86_64 | Tier 1 from M2 |
| Linux AArch64 | KVM, Xen, TCG | aarch64 | Tier 1 from M4 |
| macOS AArch64 | HVF, TCG | aarch64 | Tier 1 from M6 |
| Windows x86-64 | WHPX, TCG | x86_64 | Tier 1 from M6 |
| Linux RISC-V 64 | KVM, TCG | riscv64 | Tier 2 from M6 |
| macOS x86-64 | HVF, TCG | x86_64 | Tier 2 (Apple has stopped shipping x86 Macs, the platform is maintained while QEMU maintains it) |
| FreeBSD x86-64 and AArch64 | TCG, bsd-user | x86_64, aarch64 | Tier 2 from M7 |
| NetBSD x86-64 | NVMM, TCG | x86_64 | Tier 3 |
| OpenBSD x86-64 | TCG | x86_64 | Tier 3 |
| Linux ppc64le, s390x, LoongArch64 | KVM, TCG | native from M10, interpreter before | Tier 2 from M10 |
| Windows AArch64 | WHPX, TCG | aarch64 | Tier 3 |

Tier 1 means every CI run builds and tests on that host with real hardware acceleration. Tier 2 means every CI run builds and runs the TCG and tools tests, with accelerated runs nightly. Tier 3 means it builds in CI and bugs are accepted. 32-bit hosts are out of scope: QEMU deprecated them in 10.0 and removed them in 11.0. A WebAssembly host (QEMU 10.1 added an experimental Emscripten build) is an open question in document 25; if it happens it will be wasm64 with the interpreter backend.

## MSRV and toolchain

The toolchain is pinned in `rust-toolchain.toml` to a stable release and moved forward deliberately, at most once per ruvm minor release. The minimum supported Rust version for building from source starts at 1.85, the first release with edition 2024, and after that follows the pinned version minus two releases, reviewed once a year and checked in CI (document 02). ruvm does not track QEMU's MSRV of 1.83 because QEMU's constraint comes from Debian stable packaging of a C project that happens to contain Rust, while ruvm is built by cargo and is packaged like other Rust applications. Nightly features are not used, with one exception allowed behind a feature: `core::arch` intrinsics that are nightly only on some hosts may be used in `ruvm-jit-*` backends if a stable fallback exists.
