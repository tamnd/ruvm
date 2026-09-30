# 20. Extensibility

This document describes how people add things to ruvm: devices, machines, guest ISAs, host JIT backends, accelerators, block drivers, network backends and UIs. It covers three mechanisms, in the order we expect them to be used. The first and by far the most common is in-tree extension: a new crate, or a new module in an existing crate, that registers itself through link-time registries and is selected with cargo features. The second is out-of-process extension through protocols QEMU already speaks: vhost-user, vfio-user, and QEMU's multi-process device emulation. The third is dynamic loading: TCG plugins through QEMU's plugin C ABI, and loadable modules for the block, UI, audio, chardev and display pieces that QEMU also builds as modules. The document ends with the stability policy for each of these interfaces and a worked example that adds the PL031 real-time clock to ruvm, with tests.

The guiding rule is that an extension should touch one crate. QEMU's C tree needs a new device to be listed in a `meson.build`, a `Kconfig` file, sometimes a `configs/devices/*.mak` file, and a board file; the type is registered by a `type_init()` constructor. ruvm replaces all of that with registration statements in the crate itself and a cargo feature that turns the crate on.

## Extension points at a glance

| What you add | Trait you implement (crate) | Registered through | Selected by |
|---|---|---|---|
| Device | `Device`, `MmioOps`, optionally `VirtioDevice`, `PciDevice` (ruvm-hw-core, ruvm-hw-pci, ruvm-hw-virtio) | `register_type!` into `QOM_TYPES` | cargo feature on the device crate, pulled in by boards or by the user preset |
| Machine | `Machine` (ruvm-hw-core) | `register_type!` via `versioned_machine!` | feature on the machine crate |
| Guest ISA | `GuestArch` (ruvm-jit, ruvm-hw-core for CPU objects) | `GUEST_ARCHES` slice plus CPU model types in `QOM_TYPES` | `target-<arch>` feature |
| Host JIT backend | `HostBackend` (ruvm-jit) | `HOST_BACKENDS` slice | `cfg(target_arch)`, always one per host, plus ruvm-jit-interp |
| Accelerator | `Accel`, `Vcpu` (ruvm-accel) | `register_type!` (accelerators are QOM types named `<name>-accel`) | `accel-<name>` feature |
| Block driver | `BlockDriver` (ruvm-block) | `BLOCK_DRIVERS` slice | `block-<name>` feature, or a module |
| Netdev backend | `NetBackend` (ruvm-net) | `NET_BACKENDS` slice | `net-<name>` feature |
| Chardev backend | `CharBackend` (ruvm-chardev) | `register_type!` (chardevs are QOM types `chardev-<name>`) | `chardev-<name>` feature, or a module |
| UI | `DisplayListener`, `DisplayBackend` (ruvm-ui) | `DISPLAY_BACKENDS` slice | `ui-<name>` feature, or a module |
| Audio backend | `AudioBackend` (ruvm-audio) | `register_type!` (in the reference tree audio backends are QOM subtypes of `audio-backend`) | `audio-<name>` feature, or a module |
| QMP commands for a subsystem | generated handler trait (ruvm-qapi) | `QMP_HANDLERS` slice | follows the subsystem's feature |
| Trace events | `trace_events!` macro (ruvm-trace) | `TRACE_EVENTS` slice | always on |

Several of QEMU's extension points are QOM types (accelerators, chardevs, audio backends, machines, devices, `-object` types), and ruvm keeps them as QOM types, because `qom-list-types` exposes them and libvirt probes them. The others (block drivers, netdev types, display backends) are registered in QEMU through their own tables (`bdrv_register()` in block.c, the `net_client_init_fun` array in net/net.c, `qemu_display_register()` in ui/console.c), and ruvm gives each its own distributed slice.

## Link-time registries

Every registry is a `linkme` distributed slice. A crate adds an element with a static item; the linker collects all elements from all linked object files into one contiguous array; at start-up ruvm-system reads the array. `linkme` supports Linux, macOS, Windows, FreeBSD, OpenBSD and illumos, which covers every host ruvm targets, and it runs no code before `main`, unlike C constructors or the `ctor` crate. That property matters: QEMU's `module_init()` runs before `main()` and cannot fail cleanly, while a ruvm registration is plain data and the code that consumes it runs inside `main` with error reporting available.

```rust
// ruvm-block/src/registry.rs
#[linkme::distributed_slice]
pub static BLOCK_DRIVERS: [&'static BlockDriverDesc] = [..];

pub struct BlockDriverDesc {
    pub format_name: &'static str,     // "qcow2", "raw", "nbd", "file", "host_device"
    pub protocol_name: Option<&'static str>, // URL prefix for legacy filename syntax: "nbd", "http"
    pub kind: DriverKind,              // Format, Protocol, Filter
    pub create_opts: Option<&'static QapiTypeDesc>, // BlockdevCreateOptions branch
    pub open: fn(&mut OpenCtx) -> Result<Box<dyn BlockDriver>>,
    pub probe: Option<fn(&[u8], &str) -> i32>,       // same scores as bdrv_probe
    pub probe_device: Option<fn(&str) -> i32>,
    pub supports_backing: bool,
    pub is_whitelisted_ro: bool,       // --block-drv-ro-whitelist behavior
}
```

There is one trap with link-time registration in Rust, and every contributor hits it once. The linker only includes an object file from a static library if something references a symbol in it. A crate whose only purpose is to register types is never referenced by name, so its registrations silently vanish. `inventory` and `linkme` both document this. ruvm solves it centrally: ruvm-cli's build script reads the enabled cargo features, maps each to the crates it enables, and generates a `linked.rs` file with one `extern crate ruvm_hw_char as _;` style line and one reference to a `#[used]` anchor symbol per crate. Each ruvm crate that registers anything defines `pub static LINK_ANCHOR: u8 = 0;` through the `ruvm_base::link_anchor!()` macro, and the generated code takes its address. A CI test builds each preset and compares the output of `-device help`, `-machine help`, `-accel help`, `-netdev help` and `qemu-img --help` (which lists formats) against the expected list for that preset, so a missing anchor is caught immediately.

Registries are validated once at start-up: duplicate names are fatal with both source locations printed (each entry records `file!()` and `line!()`), QOM parents must exist, and a device that declares a bus type must name a bus type that exists in the build. Validation takes well under a millisecond for the full build (a few thousand entries), which fits in the 15 ms start-up budget from the canon.

## Adding each kind of thing

### A device

A device is a struct with `#[derive(Object, Device)]`, an `impl MmioOps` (or `PioOps`, or `PciDevice`, or `VirtioDevice` for the transport-independent virtio model), an `impl Device`, a `#[derive(VmState)]` state struct, and a `register_type!` line. Document 04 shows the PL011 in full and the worked example at the end of this document shows the PL031. The device goes into the ruvm-hw-<family> crate that matches QEMU's hw/<family> directory, behind its own cargo feature named after the QEMU type (`pl031`, `virtio-rng`, `e1000e`), so that a slim build can include exactly the devices its boards need.

Rules a new device must follow, all checked in review and most checked by tooling:

1. QOM type name, property names, property type strings, defaults, and descriptions are QEMU's. `cargo xtask qom-diff` checks this for every device both emulators know.
2. VMState name, version, field order and subsections are QEMU's. The `vmstate-diff` job compares `-dump-vmstate` JSON output from both emulators (QEMU's `dump_vmstate_json_to_file()` format).
3. No `unsafe`, no threads of its own, no blocking except where the QEMU model blocks (document 04 discusses the PL011 chardev write).
4. Guest errors go to `guest_error!`, unimplemented features to `unimp!`, never to `Err` or panic.
5. Cross-device calls are declared with `#[sync_peer]` so the lock domain graph (document 03) knows about them.
6. Timers, bottom halves and backend handlers are created through the `RealizeCtx`, which binds them to the device's domain.

### A machine

A machine is a QOM type deriving from `machine` with an `init` function that creates CPUs, RAM, and onboard devices, and a set of versioned compat property arrays. The `versioned_machine!` macro from document 04 generates one QOM type per version. Board code is mostly wiring, and ruvm provides builders for the patterns that recur in QEMU boards: `sysbus_create_simple` equivalents, IRQ wiring, flattened device tree nodes (ruvm-firmware's FDT builder), and ACPI table construction. A board lives in the ruvm-machine-<family> crate and has a cargo feature that enables the device features it needs, which is ruvm's replacement for Kconfig's `select`. QEMU's hw/arm/Kconfig has `config ARM_VIRT` select `PL011`, `PL031`, `PL061`, `GPIO_KEY`, `ARM_GIC`, `ARM_SMMUV3` and more, and imply `PCI_DEVICES`, `TEST_DEVICES` and others; ruvm-machine-arm's Cargo.toml says `virt = ["ruvm-hw-char/pl011", "ruvm-hw-timer/pl031", ...]`. Kconfig's `imply` (a default that the user can turn off) becomes a separate `virt-default-devices` feature that `default` turns on, and `--without-default-devices` corresponds to building with the board feature but without its `-default-devices` feature.

The board must reproduce QEMU's creation and realize order so that `/machine/unattached/device[N]` numbering matches (document 04), and its FDT and ACPI output must match QEMU's byte for byte for the same configuration. The ACPI part is checked with QEMU's own `tests/qtest/bios-tables-test` expected blobs in tests/data/acpi/, which ruvm runs unmodified.

### A guest ISA

A guest ISA is a ruvm-target-<arch> crate implementing `GuestArch` (document 09): CPU models as QOM types (`<model>-<arch>-cpu`, as QEMU names them), the decoder (usually generated by ruvm-decode from QEMU's `.decode` files), the translation of each instruction into ruvm-jit IR, runtime helpers, the exception and interrupt model, gdbstub XML (reusing QEMU's gdb-xml/ files), the CPU VMState, and the disassembler hook. The minimum viable target is interpreter-grade: implement `GuestArch` against ruvm-jit's IR, and all host backends including ruvm-jit-interp can run it. linux-user support (document 10) is a second step: syscall numbers, signal frame layout, ELF loader details and `cpu_loop` behavior. The target registers itself in `GUEST_ARCHES` with its name as it appears in the binary names (`qemu-system-riscv64`, `qemu-riscv64`), and ruvm-cli's multi-call dispatcher finds it by that name.

### A host JIT backend

A host backend is a ruvm-jit-<host> crate implementing `HostBackend` (document 08): register file description, instruction selection from the IR, the softmmu TLB fast path sequence, block chaining patch sites, call ABI for helpers, and the memory barrier mapping for the guest memory model. It registers in `HOST_BACKENDS` with a predicate on the host CPU. Backends are selected at start-up by host detection, and `ruvm-jit-interp` is always linked as the fallback, which is how ruvm supports hosts without a native backend (as QEMU does with TCI). A new backend must pass the `tcg` test suite (tests/tcg in QEMU) for every guest that is enabled on it, and the differential JIT test harness (document 22) that runs random instruction streams on both the new backend and the interpreter and compares architectural state.

### An accelerator

An accelerator is a QOM type named `<name>-accel` with parent `accel`, implementing `Accel` and `Vcpu` (document 06). It registers with `register_type!` like any QOM type, since `-accel help` and `qom-list-types` list accelerators by QOM type. An accelerator must provide memory slot mapping from ruvm-mem's memory listener, the vCPU run loop and exit decoding, register access through `ArchState`, dirty logging for migration, and irqchip integration if the hypervisor has an in-kernel interrupt controller. Accelerators often need host bindings, which go into ruvm-sys behind a feature so that the accelerator crate itself contains no unsafe code outside its ioctl wrappers. QEMU 11 added `nitro` for AWS Nitro Enclaves; ruvm-accel-nitro is the in-tree example of an accelerator with an unusual vCPU model (the enclave's vCPUs are not driven by a user-space run loop), and it shows how an accelerator reports capabilities that restrict which machines and devices are allowed.

### A block driver

A block driver implements `BlockDriver` (document 14): open with options from `BlockdevOptions` (the generated QAPI type, whose branch for the driver must exist in qapi/block-core.json since ruvm does not extend the schema), async read, write, flush, discard, and write-zeroes, block status, and optionally create, amend, snapshot and bitmap hooks. Drivers are async functions that run on the node's reactor. A driver that wraps a C library with blocking calls (libiscsi, librbd, libnfs, libssh are the QEMU examples, all built as modules in QEMU) runs those calls on the reactor's thread pool. A new format driver must pass the iotests for that format; qemu-img conversion round trips against QEMU's qemu-img are part of document 22's suite.

### A netdev backend

A network backend implements `NetBackend` (document 15): receive a batch of packets from the frontend, deliver packets to the frontend through its queue, report offload capabilities (vnet header, TSO, UFO, USO) and link state. The QAPI branch for its options must exist in `Netdev` in qapi/net.json. `-netdev` accepts both legacy QemuOpts and keyval syntax for backends, and the backend gets its already-validated options struct either way. Backends that use a kernel data path (tap with vhost-net, vhost-vdpa, AF_XDP) implement the optional `VhostBackend` trait so the virtio-net frontend can hand the virtqueues to the kernel.

### A UI

A UI implements `DisplayBackend` (created from a `-display` option, whose `DisplayOptions` branch must exist in qapi/ui.json) and one or more `DisplayListener`s (document 15), the equivalent of QEMU's `DisplayChangeListener`: surface switch, dirty rectangle update, cursor define and move, and optionally GL scanout. Input goes back through ruvm-ui's input router with QEMU's `InputEvent` QAPI types, so `input-send-event` works regardless of UI. UIs that need their own event loop (GTK, SDL, Cocoa) run on the thread that toolkit demands, and communicate with the main reactor through `ReactorHandle` (document 03).

## Cargo features and slim builds

QEMU decides what goes into a binary with configure options, Kconfig, and the `configs/devices/<target>/*.mak` files. ruvm uses cargo features only, arranged in three levels so that the common cases are one flag.

1. Leaf features on each crate name one thing: a device (`ruvm-hw-timer/pl031`), a board (`ruvm-machine-arm/virt`), a block driver (`ruvm-block/qcow2`), an accelerator (`ruvm-cli/accel-kvm`), a UI (`ruvm-cli/ui-gtk`), a guest (`ruvm-cli/target-aarch64`).
2. Board features enable the leaf features a board needs, as described above, replacing Kconfig `select` and `imply`.
3. Presets on ruvm-cli enable a consistent product: `full` (the default: everything that builds on the host), `microvm`, `tcg-only`, `tools` (qemu-img, qemu-io, qemu-nbd, qemu-storage-daemon only), `user` (linux-user or bsd-user only).

Two presets are required to exist and are built in CI on every change, because they are the ones that stress the modular structure.

The `microvm` preset is for serverless and sandbox use in the Firecracker class. It builds `qemu-system-x86_64` and `qemu-system-aarch64` personalities with `accel-kvm` only (plus `accel-hvf` on macOS), the x86 `microvm` machine and the arm `virt` machine with a trimmed device set, virtio-mmio and virtio-pci transports, virtio-blk, virtio-net, virtio-console, virtio-rng, virtio-vsock (vhost), the serial and RTC devices those boards need, the `raw` and `file` block drivers (qcow2 as an option), tap and vhost-user network backends, chardevs for socket, file, stdio and pty, QMP, and no JIT, no UI, no audio, no USB, no legacy ISA devices beyond what microvm needs. The target for this preset is the canon's 15 ms process start to first guest instruction and 110 ms Linux boot to init. A small binary helps the first number because fewer pages are touched and relocated at start-up, and the registry validation walks fewer entries.

The `tcg-only` preset builds every system and user target with the JIT and interpreter, no hardware accelerators, no ruvm-sys hypervisor bindings. It is the build for hosts without virtualization support, for CI of guest software, and for the Windows and macOS builds used for cross-architecture development. It also checks that no device crate has grown an accidental dependency on an accelerator crate, which the layer check alone would not catch because accelerators and devices are both L2 and L3 crates that could depend on each other through ruvm-hw-core.

Features are additive, as cargo requires. That rules out "negative" features, so anything that would be `--disable-foo` in QEMU is instead the absence of a positive feature. Mutually exclusive choices (the allocator, the JIT's default tier policy) are runtime options, not features. The QAPI schema conditions from document 04 (`CONFIG_VNC`, `CONFIG_SPICE`, and so on) are derived from features, so a `microvm` build's `query-qmp-schema` differs from a `full` build's in exactly the way a QEMU built with `--disable-vnc --disable-spice` differs from a default one.

A few numbers make the reasons for this concrete. In the reference tree, hw/ alone has about 1,500 C source files; a board like `microvm` uses a few dozen devices. Linking everything and relying on the linker's dead code elimination does not work with link-time registries, because every registered type is referenced from the registry and so nothing is dead. Leaving out the crates is the only way to get a small binary.

## Out-of-process extension

Some extensions should not be linked into the emulator: device models from third parties, devices that need a different license, devices that should run with fewer privileges, and device implementations that already exist in another program. QEMU supports three protocols for this, and ruvm speaks all of them in the same way, because they are part of QEMU's compatibility surface (libvirt configures them, and existing backends such as virtiofsd, the vhost-user-blk export in qemu-storage-daemon, and SPDK depend on them).

### vhost-user

vhost-user (docs/interop/vhost-user.rst) moves a virtio device's data path into another process over a Unix socket, with guest memory shared through file descriptors and notifications through eventfds. The front end stays in the emulator (virtio transport, feature negotiation, config space), the back end processes virtqueues. ruvm implements the front end in ruvm-hw-virtio for every vhost-user device type QEMU has (`vhost-user-blk`, `vhost-user-scsi`, `vhost-user-fs`, `vhost-user-gpu`, `vhost-user-input`, `vhost-user-vsock`, `vhost-user-i2c`, `vhost-user-spi`, `vhost-user-gpio`, `vhost-user-rng`, `vhost-user-snd`, `vhost-user-scmi`, the generic `vhost-user-test-device`, and `vhost-user-rtc` added in QEMU 11.1, each with its PCI and MMIO variants where QEMU has them), plus the netdev `vhost-user`. Protocol feature negotiation, inflight I/O tracking (`VHOST_USER_GET_INFLIGHT_FD`), reconnect, and the `VHOST_USER_PROTOCOL_F_DEVICE_STATE` migration path are all required (document 13).

For people writing back ends, ruvm ships `ruvm-vhost-user-backend`, a thin layer over the rust-vmm `vhost` and `vhost-user-backend` crates that adds ruvm-aio integration and the block and network helpers ruvm's own daemons use. `qemu-storage-daemon`'s `--export type=vhost-user-blk` is built on it. This is the recommended route for a third-party virtio device: write a vhost-user back end, and the device works with both QEMU and ruvm without any change to either.

### vfio-user

vfio-user moves a whole PCI device into another process, using a protocol modeled on the VFIO kernel interface over a Unix socket. QEMU has both sides: the client device `vfio-user-pci` (hw/vfio-user/), which attaches a remote device as if it were a VFIO device, and the server `x-vfio-user-server` object (hw/remote/vfio-user-obj.c), which exports a QEMU device model to another VMM and must be used with the `x-remote` machine. ruvm implements the client in ruvm-hw-vfio with the same properties and the same migration behavior, and the server as the `x-vfio-user-server` object with the same restrictions.

vfio-user is the recommended route for non-virtio devices (an emulated NIC, a storage controller, an accelerator card) that a third party wants to ship separately. It is also how ruvm's own device models can be run out of process for isolation: `ruvm --personality device-server` starts a process with the `x-remote` machine and one device, and the main VMM attaches it with `vfio-user-pci`. The cost is one extra context switch per MMIO access that is not handled by shared memory, which is acceptable for devices whose hot path is DMA and doorbells, and not for devices polled by the guest through MMIO registers.

### Multi-process QEMU

QEMU's older multi-process mode (docs/system/multi-process.rst) uses the `x-remote` machine on the remote side and the `x-pci-proxy-dev` device on the local side, with a QEMU-specific protocol (hw/remote/mpqemu-link.c). ruvm implements `x-remote` because the vfio-user server needs it. It does not implement `x-pci-proxy-dev` and the mpqemu protocol before 1.0: both are marked experimental (`x-` prefix), vfio-user covers the same use case with a protocol that is documented and has other implementations (libvfio-user, SPDK), and we know of no management tool that configures the proxy device. If a user shows up with a need, it is a bounded piece of work in ruvm-hw-misc. This is listed for document 25.

## Dynamic loading

### TCG plugins

TCG plugins are shared objects loaded with `-plugin file=<path>,<args>` that instrument guest execution through the API in include/plugins/qemu-plugin.h. The API is a C ABI: the plugin exports `qemu_plugin_version` and `qemu_plugin_install(id, info, argc, argv)` and calls back into functions exported by the emulator binary (the `qemu_plugin_*` functions marked `QEMU_PLUGIN_API`). The loader in plugins/loader.c checks the plugin's version against `QEMU_PLUGIN_MIN_VERSION` and `QEMU_PLUGIN_VERSION` (7 in QEMU's development tree as of this writing). The canon requires this ABI bit for bit so that existing plugins such as the ones in contrib/plugins (cache, execlog, hotblocks, howvec, lockstep, drcov, uftrace and others) load unmodified, and document 08 covers how the JIT inserts callbacks and inline operations.

The ABI requirement has consequences for how the ruvm binary is linked, and these are the extensibility-relevant parts:

1. The binary must export every `qemu_plugin_*` symbol, with C linkage and QEMU's names. ruvm-plugin defines them as `#[no_mangle] pub extern "C" fn`, and the link step uses the same symbol lists QEMU generates with scripts/qemu-plugin-symbols.py: `--dynamic-list` on ELF hosts and `-exported_symbols_list` on macOS.
2. On Windows, QEMU plugins link against an import library whose DLL name is the placeholder `qemu.exe`, delay-loaded, with a hook (the `win32_linker.c` file shipped with plugins) that resolves the symbols from the running program. ruvm's Windows executable exports the same symbols, so plugins built for QEMU on Windows resolve against ruvm through the same hook without being rebuilt.
3. Plugins are written in C, and since QEMU 11.0 in C++ as well. ruvm also offers `ruvm-plugin-sdk`, a safe Rust wrapper crate over the same C ABI for people writing plugins in Rust. It is not a separate ABI: a Rust plugin built with it is a normal QEMU plugin and also loads in QEMU.
4. The multi-call binary means one ruvm executable serves every target. The `qemu_info_t` passed to `qemu_plugin_install` reports the target name of the personality selected by argv[0], so a plugin that checks `info->target_name` sees the same value it would in QEMU.

### Loadable modules

QEMU can be built with `--enable-modules`, in which case some drivers are shared objects loaded on demand by util/module.c: block drivers with heavy library dependencies (`blkio`, `curl`, `iscsi`, `nfs`, `ssh`, `rbd` in block/meson.build), UI front ends (`curses`, `gtk`, `sdl`, `spice-core`, `spice-app`, `dbus`, `opengl`, `egl-headless`), audio drivers, some chardevs (`baum`, `spice`), display devices (`qxl`, the virtio-gpu family), USB redirection and host passthrough, and the qtest accelerator. Distributions use this to split packages so that a minimal install does not pull in GTK, Ceph, or libiscsi. The loader only accepts modules from the same build: each module must contain a build-specific stamp symbol, and a module from a different build fails with `Only modules from the same build can be loaded`.

ruvm needs the same packaging property, so it has loadable modules for the same categories. The question is the ABI at the boundary, and the options were: Rust's native ABI with `dylib` crates, `abi_stable`, or a hand-written C ABI.

Rust's native ABI is out. Rust makes no layout or calling convention promise across separately compiled artifacts, a `dylib` crate brings its own copy of std unless everything links dynamically against the same `libstd-<hash>.so` (which distributions do not ship for applications), and link-time registries in a dynamically loaded image produce a second, separate distributed slice that the main binary does not see.

`abi_stable` solves the layout problem well: it checks type layouts at load time, supports prefix types that can grow in minor versions, and lets a module be built with a different compiler version. We decided against it for three reasons. First, it makes the module interface Rust-only, and part of the value of modules for distributions is that the modules wrap C libraries (libcurl, librbd, libiscsi, GTK) where the natural implementation language of a third-party module may be C. Second, it adds its own type system (`RString`, `RVec`, `#[sabi_trait]`) to every type crossing the boundary, which in our case would mean re-expressing `BlockDriver` and `DisplayListener` in its vocabulary, a large and permanent tax on the interfaces we most want to keep simple. Third, its main advantage, loading modules built by a different compiler against a different version of the host, is something we do not want to promise in 1.0 anyway, for the same reason QEMU does not: module interfaces follow internal interfaces, and internal interfaces change.

So the boundary is a hand-written C ABI, versioned, with a same-build requirement in 1.0 that can be relaxed later per category. It is defined in `ruvm-module-abi`, a small crate with `#[repr(C)]` types only.

```rust
// ruvm-module-abi/src/lib.rs, GPL-2.0-or-later like the interfaces it carries
#[repr(C)]
pub struct RuvmModuleHeader {
    pub magic: [u8; 8],                 // b"RUVMMOD\0"
    pub abi_major: u16,                 // 1
    pub abi_minor: u16,                 // grows when tables grow at the end
    pub build_id: [u8; 32],             // must equal the host's build id in 1.0
    pub name: *const c_char,            // "block-curl", "ui-gtk", same names as QEMU modules
    pub kind: RuvmModuleKind,           // Block, Ui, Audio, Chardev, HwDisplay, HwUsb, Accel
    pub register: unsafe extern "C" fn(host: *const RuvmHostApi, out: *mut RuvmRegistrations) -> i32,
}

#[repr(C)]
pub struct RuvmRegistrations {
    pub qom_types: *const RuvmTypeInfoC, pub n_qom_types: usize,
    pub block_drivers: *const RuvmBlockDriverC, pub n_block_drivers: usize,
    pub display_backends: *const RuvmDisplayBackendC, pub n_display_backends: usize,
}

#[no_mangle]
pub static RUVM_MODULE: RuvmModuleHeader = /* generated by ruvm_module!() */;
```

Each table entry is a C struct of function pointers plus an opaque context pointer, and the host wraps it in an adapter that implements the corresponding Rust trait. The host gives the module a `RuvmHostApi` table of function pointers for the services modules need (allocation is not one of them: each side frees what it allocates; error creation; logging and tracing; reactor submission; guest memory access for display and USB modules; QOM property access). A Rust module never touches that table directly: the `ruvm_module!` macro in `ruvm-module-sdk` generates the C tables from normal trait implementations and gives the module code the same Rust API it would have in-tree. So a module is the same source as the in-tree driver, built as a `cdylib` with the `module` feature on, which is how QEMU builds its modules from the same source files.

The module loader follows util/module.c's behavior where it is visible: the module directory search order (the `QEMU_MODULE_DIR` environment variable, then the configured module directory relocated relative to the executable, then `/var/run/qemu/<version>` when module upgrades are enabled), the naming (`block-curl`, `ui-gtk`, `hw-display-qxl`, and so on, so that a distribution's package split can reuse its QEMU package lists), loading on demand when a QOM type, block driver or display type is requested but not present (the `modinfo` table that maps type names to modules is generated at build time, as QEMU generates its modinfo), and `-display help` and `-device help` listing types provided by modules that are installed.

Load-time checks: magic, `abi_major` equal to the host's, `abi_minor` at most the host's (a newer module against an older host is rejected), and build id equal. The build id check is what makes the same-build rule; relaxing it later for a category means freezing that category's tables and guaranteeing `abi_major` stability for it, which we will consider after 1.0 for block drivers only, since they are the only category where out-of-tree drivers have a real history.

### WASM device sandbox: out of scope

We considered letting device models be WebAssembly components run in an embedded runtime (for example Wasmtime), as a way to load untrusted third-party devices into the emulator process with memory isolation. We decided it is out of scope for 1.0, and it is recorded as an open question for later.

The reasons: a device's MMIO handler is on the vCPU exit path, and every access would cross into the WASM runtime and back, adding a call boundary and a copy of the access context on the path the canon requires to be allocation-free and single-lock; we have not measured this cost and will not guess it. DMA is the bigger problem: a device model reads and writes guest memory, and a WASM module can only address its own linear memory, so every DMA would be a host call with a copy, or guest RAM would have to be mapped into the module's address space, which current runtimes do not support for arbitrary host mappings and which would give up much of the isolation. And the isolation goal is already met by vhost-user and vfio-user: a separate process with its own seccomp policy, its own user, and its own crash domain, with protocols that other VMMs also speak. If WASM devices come back, the most plausible shape is a vfio-user or vhost-user server that hosts WASM components, outside the emulator, so that the emulator's hot path is unaffected.

## Versioning and stability policy

ruvm has interfaces at very different levels of stability, and extension authors need to know which is which. There are four tiers.

| Tier | Interfaces | Stability promise |
|---|---|---|
| QEMU-compatible | Command line, QMP and QAPI schema, QOM type and property names, migration stream, disk formats, TCG plugin C ABI, vhost-user, vfio-user, qtest protocol, gdbstub protocol | Whatever QEMU promises, following QEMU's deprecation policy (docs/about/deprecated.rst: two releases of deprecation before removal). ruvm tracks QEMU; it does not make these more or less stable than QEMU does |
| Extension API | `Device`, `MmioOps`, `PioOps`, `PciDevice`, `VirtioDevice`, `Machine`, `BlockDriver`, `NetBackend`, `CharBackend`, `DisplayListener`, `AudioBackend`, `UserCreatable`, the derive macros and registries, the test harness in ruvm-hw-test | Semver on the ruvm crate version. After 1.0, breaking changes only in a new major version. Before 1.0, breaking changes at minor versions, with a changelog entry and a migration note |
| Module ABI | `ruvm-module-abi` C tables | Same build only in 1.0. `abi_major` changes when a table changes incompatibly; `abi_minor` grows when tables grow at the end |
| Internal | Everything else: `GuestArch` and `HostBackend` details, ruvm-mem internals, JIT IR, lock domain internals, ruvm-aio internals | No promise. Marked `#[doc(hidden)]` or in modules named `internal` |

`GuestArch` and `HostBackend` are deliberately internal. They are large, they change as the JIT improves (document 07 and 08 expect the IR to evolve through M9), and guest ISAs and host backends are expected to live in the tree. An out-of-tree ISA is possible but has to track ruvm's main branch.

Three techniques keep the extension API stable without freezing it. Traits have default methods for everything that is optional, so adding a method is not breaking. Structs that cross the boundary (`AccessCtx`, `RealizeCtx`, `OpenCtx`, `BlockDriverDesc`) are `#[non_exhaustive]` and built with constructors or builders, so adding fields is not breaking. And the derive macros generate the glue that would otherwise be written by hand, so changes to glue are absorbed by the macro. `cargo semver-checks` runs in CI on every extension API crate against the last release.

Deprecation inside ruvm's own APIs follows QEMU's shape: `#[deprecated]` for at least one minor release before removal before 1.0, and one major release after.

## Worked example: adding the PL031 RTC

This section walks through adding a device from nothing to merged: the ARM PrimeCell PL031 real-time clock, which QEMU implements in hw/rtc/pl031.c and which the arm `virt` board instantiates at 0x09010000 with SPI 2 (`create_rtc()` in hw/arm/virt.c). It is small but touches most of the machinery: an MMIO region, an interrupt line, a timer on the RTC clock, a QMP event, a VMState with a subsection and load hooks, and board wiring with a device tree node.

### Step 1: find the observable surface

Read the C model and write down what must match. For the PL031: QOM type `pl031`, parent `sys-bus-device`, no properties in the reference tree (the old `migrate-tick-offset` compat property went away with the machine versions that used it). One 4 KiB MMIO region named `pl031`, one IRQ output. Registers: `DR` at 0x00 (current count, read-only), `MR` at 0x04 (match), `LR` at 0x08 (load), `CR` at 0x0c (reads as 1, writes ignored), `IMSC` at 0x10 (only bit 0 kept), `RIS` at 0x14, `MIS` at 0x18, `ICR` at 0x1c (write-only), and ID bytes at 0xfe0 to 0xffc: `0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1`. Guest error messages for bad offsets and read-only or write-only accesses. A write to `LR` emits the QMP event `RTC_CHANGE` (qapi/misc.json, data `offset` and `qom-path`), which is rate limited to one per second by the monitor. VMState `pl031` version 1 with fields `tick_offset_vmstate`, `mr`, `lr`, `cr`, `im`, `is`, and an always-sent subsection `pl031/tick-offset` with the real `tick_offset`, plus pre-save, pre-load and post-load logic that converts between the RTC clock and the virtual clock for streams from old QEMU versions.

### Step 2: write the device

The device goes into ruvm-hw-timer (QEMU keeps it in hw/rtc, and ruvm folds rtc into the timer family crate) behind the feature `pl031`.

```rust
// crates/ruvm-hw-timer/src/pl031.rs
use ruvm_hw_core::prelude::*;
use ruvm_qapi::events::misc::rtc_change;

static ID: [u8; 8] = [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1];
const NS_PER_SEC: i64 = 1_000_000_000;

#[derive(Default, VmState)]
#[vmstate(name = "pl031", version = 1, minimum_version = 1,
          pre_save = Self::pre_save, pre_load = Self::pre_load,
          post_load = Self::post_load, subsections = [TICK_OFFSET])]
pub struct Pl031Regs {
    tick_offset_vmstate: u32,
    mr: u32,
    lr: u32,
    cr: u32,
    im: u32,
    is: u32,
    #[vmstate(skip)] tick_offset: u32,
    #[vmstate(skip)] tick_offset_migrated: bool,
}

#[derive(VmState)]
#[vmstate(name = "pl031/tick-offset", version = 1, minimum_version = 1,
          post_load = |r: &mut Pl031Regs| r.tick_offset_migrated = true)]
struct TickOffset(#[vmstate(field = "tick_offset")] u32);

#[derive(Object, Device)]
#[qom(name = "pl031", parent = "sys-bus-device")]
pub struct Pl031 {
    #[parent] parent: SysBusDevice,
    regs: DomainCell<Pl031Regs>,
    #[mmio(name = "pl031", size = 0x1000, ops = Self)] iomem: MmioRegion,
    #[sysbus_irqs] irq: [IrqOut; 1],
    #[timer(clock = Rtc, callback = Self::alarm)] timer: Timer,
}

impl Pl031 {
    fn count(&self, cx: &impl ClockCtx, r: &Pl031Regs) -> u32 {
        r.tick_offset.wrapping_add((cx.now(Clock::Rtc) / NS_PER_SEC) as u32)
    }

    fn update(&self, r: &Pl031Regs) {
        self.irq[0].set(r.is & r.im != 0);
    }

    fn set_alarm(&self, cx: &impl ClockCtx, r: &mut Pl031Regs) {
        // Wrapping subtraction, as in pl031_set_alarm(): correct when mr < now.
        let ticks = r.mr.wrapping_sub(self.count(cx, r));
        if ticks == 0 {
            self.timer.cancel();
            r.is = 1;
            self.update(r);
        } else {
            self.timer.arm_ns(cx.now(Clock::Rtc) + ticks as i64 * NS_PER_SEC);
        }
    }

    fn alarm(&self, cx: &mut TimerCtx) {
        let mut r = cx.state(&self.regs);
        r.is = 1;
        self.update(&r);
    }
}

impl MmioOps for Pl031 {
    fn read(&self, cx: &AccessCtx, off: u64, _s: AccessSize) -> MemResult<u64> {
        let r = cx.state(&self.regs);
        Ok(match off {
            0x00 => self.count(cx, &r) as u64,
            0x04 => r.mr as u64,
            0x08 => r.lr as u64,
            0x0c => 1, // RTC is permanently enabled
            0x10 => r.im as u64,
            0x14 => r.is as u64,
            0x18 => (r.is & r.im) as u64,
            0x1c => { guest_error!("pl031: read of write-only register at offset 0x{:x}", off); 0 }
            0xfe0..=0xfff => ID[((off - 0xfe0) >> 2) as usize] as u64,
            _ => { guest_error!("pl031_read: Bad offset 0x{:x}", off); 0 }
        })
    }

    fn write(&self, cx: &AccessCtx, off: u64, _s: AccessSize, v: u64) -> MemResult<()> {
        let mut r = cx.state(&self.regs);
        let v = v as u32;
        match off {
            0x08 => {
                r.lr = v;
                let now = self.count(cx, &r);
                r.tick_offset = r.tick_offset.wrapping_add(v.wrapping_sub(now));
                let tm = cx.clock().guest_time_from_offset(r.tick_offset);
                rtc_change(cx.events(), tm.diff_from_host(), self.canonical_path());
                self.set_alarm(cx, &mut r);
            }
            0x04 => { r.mr = v; self.set_alarm(cx, &mut r); }
            0x10 => { r.im = v & 1; self.update(&r); }
            0x1c => { r.is &= !v; self.update(&r); }
            0x0c => {} // written value is ignored
            0x00 | 0x14 | 0x18 => guest_error!(
                "pl031: write to read-only register at offset 0x{:x}", off),
            _ => guest_error!("pl031_write: Bad offset 0x{:x}", off),
        }
        Ok(())
    }

    fn valid() -> AccessConstraints { AccessConstraints::default() } // as MemoryRegionOps defaults
}

impl Device for Pl031 {
    fn realize(&self, cx: &mut RealizeCtx) -> Result<()> {
        // pl031_init(): tick_offset = host time now - rtc_clock seconds
        let host = cx.clock().host_time_seconds();
        self.regs.with(|r| r.tick_offset =
            (host - cx.now(Clock::Rtc) / NS_PER_SEC) as u32);
        Ok(())
    }
    fn vmstate(&self) -> &'static VmStateDescription { Pl031Regs::VMSTATE }
}

register_type!(Pl031::TYPE_INFO);
```

The pre-save, pre-load and post-load functions are straight ports of `pl031_pre_save()`, `pl031_pre_load()` and `pl031_post_load()`, and are omitted here. Two details are easy to get wrong. The `tick_offset` initialization happens in `instance_init` in C, not in realize; since it only reads clocks, doing it in realize is not observable, but if a board reads `DR` through a debug path before realize the values would differ, so the review checklist says to keep C's placement unless there is a reason. The second detail is the event: `RTC_CHANGE` is emitted from the vCPU thread while holding the device domain; ruvm-monitor's event emission only enqueues to the monitor's reactor and takes no domain, so this is a legal call in rank order (leaf band).

### Step 3: register the feature and wire the board

```toml
# crates/ruvm-hw-timer/Cargo.toml
[features]
pl031 = []

# crates/ruvm-machine-arm/Cargo.toml
[features]
virt = ["ruvm-hw-char/pl011", "ruvm-hw-timer/pl031", "ruvm-hw-gpio/pl061", "..."]
```

In the board, the wiring mirrors `create_rtc()`: create the device with `sysbus_create_simple("pl031", base, gic.gpio_in(irq))`, and add the FDT node `/pl031@9010000` with `compatible = "arm,pl031\0arm,primecell"`, `reg`, `interrupts` (SPI, level high), `clocks` and `clock-names = "apb_pclk"`. The FDT builder emits properties in the order they are set, and the order must match QEMU's for the dumped DTB (`-machine dumpdtb=`) to be byte-identical.

### Step 4: tests

Every new device lands with four kinds of tests. None of them is optional for merge.

Unit tests run the device without a machine using ruvm-hw-test, which provides a fake clock set, an IRQ probe, an event sink and a direct MMIO accessor that goes through the same dispatcher as a real vCPU:

```rust
#[test]
fn alarm_fires_and_clears() {
    let mut t = DeviceHarness::<Pl031>::realize_default();
    t.clocks.set(Clock::Rtc, 1_000 * NS_PER_SEC);
    let now = t.read32(0x00);
    t.write32(0x10, 1);            // IMSC: enable
    t.write32(0x04, now + 5);      // MR: alarm in 5 s
    assert!(!t.irq(0).level());
    t.clocks.advance(Clock::Rtc, 5 * NS_PER_SEC);
    t.run_timers();
    assert!(t.irq(0).level());
    assert_eq!(t.read32(0x18), 1); // MIS
    t.write32(0x1c, 1);            // ICR
    assert!(!t.irq(0).level());
}

#[test]
fn load_register_emits_rtc_change() {
    let mut t = DeviceHarness::<Pl031>::realize_default();
    t.write32(0x08, 0);
    let ev = t.events.take_one("RTC_CHANGE");
    assert_eq!(ev["qom-path"], t.path());
}
```

The harness also has a mode that runs every test twice, once in normal lock domain mode and once in serialized mode (document 03), so a device that accidentally depends on the global order shows up as a difference.

Protocol-level tests run the full binary with `-machine virt -accel qtest` and drive it with the qtest protocol, the same protocol QEMU's tests/qtest uses. QEMU has no dedicated PL031 qtest in the reference tree; the new test is written in Rust against ruvm's qtest client and is also run against QEMU in CI, which is how we check that the test encodes QEMU's behavior and not ruvm's:

```rust
#[qtest(machine = "virt", both_emulators)]
fn pl031_id_and_cr(qt: &mut QTest) {
    let base = 0x0901_0000;
    let id: Vec<u32> = (0..8).map(|i| qt.readl(base + 0xfe0 + 4 * i)).collect();
    assert_eq!(id, [0x31, 0x10, 0x14, 0x00, 0x0d, 0xf0, 0x05, 0xb1]);
    assert_eq!(qt.readl(base + 0x0c), 1);
}
```

Compatibility tests check the surfaces from step 1 against QEMU automatically, and adding the device to the build is enough to include it: `qom-diff` compares type, properties and `qom-list` output; `vmstate-diff` compares `-dump-vmstate` output; the migration interop job (document 17) migrates a `virt` guest from QEMU to ruvm and back with an alarm pending, and checks that the alarm fires at the same guest time on the destination; the DTB comparison dumps both device trees and diffs them.

A fuzz target is generated for every MMIO device by the `#[mmio]` attribute: random sequences of reads and writes at random offsets and sizes, with the clock advanced between them, checking that the device never panics and that the IRQ line level always equals the value computed from the state (`is & im != 0`), an invariant the author states in one line in the fuzz config. For devices with DMA, the fuzzer also mutates guest memory, in the style of QEMU's generic-fuzz target.

### Step 5: review checklist

The reviewer checks the observable surface table from step 1 against the code, confirms the guest error strings match QEMU's, confirms no `unsafe`, confirms the device declares no synchronous peers (the PL031 has none: its only outbound effect is an interrupt line and an event), and confirms the board wiring order. The whole change is one new file in ruvm-hw-timer, two lines of Cargo.toml, about twenty lines in the board, and the tests.

## Decisions made in this document

1. Link-time registries are anchored centrally by a build script in ruvm-cli that references each enabled crate's `LINK_ANCHOR`, and CI compares the `help` listings of each preset against expected lists.
2. Kconfig `select` and `imply` map to board features and separate `-default-devices` features.
3. The `microvm` and `tcg-only` presets are required and built in CI on every change.
4. Loadable modules use a hand-written, versioned C ABI (`ruvm-module-abi`) and require the same build id in 1.0; `abi_stable` was considered and rejected. A stable cross-build ABI may be offered after 1.0 for block drivers only.
5. Module names and the module directory search order follow QEMU's so that distribution packaging can be reused.
6. `x-pci-proxy-dev` and the mpqemu protocol are not implemented before 1.0; `x-remote` and `x-vfio-user-server` are, and vfio-user is the recommended out-of-process route for non-virtio devices.
7. A WASM device sandbox is out of scope for 1.0; if revisited, it would live in an out-of-process vfio-user or vhost-user server.
8. `ruvm-plugin-sdk` provides a Rust wrapper over the unchanged TCG plugin C ABI; there is no separate Rust plugin ABI.
9. `GuestArch` and `HostBackend` are internal interfaces with no stability promise; the device, machine, and backend traits are the stable extension API under semver.
