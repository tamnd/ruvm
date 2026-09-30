# Changelog

Notable changes, newest first. This project is pre-1.0 and makes no compatibility promise about its own APIs until it has one. The compatibility it does promise is with QEMU 11.1, and each release says how much of that is real.

The minor version is the number of milestones finished. 0.1.0 is the release where M0 closes, 0.2.0 where M1 closes, and so on up to M12, which is 1.0. Patch releases come whenever enough has landed to be worth a tag. The milestones are the issues labeled `kind/milestone` at https://github.com/tamnd/ruvm/issues.

## Unreleased

## 0.2.2

This patch fills in most of the hardware the M2 boards need. Both boards now exist as objects with their devices wired up, but ruvm does not run vCPUs against them from the command line yet. That wiring is the next piece of work.

Virtqueues come first. `ruvm-virtio-queue` has split and packed rings with descriptor chains, readers and writers, and event index handling (#65). It is MIT or Apache-2.0 licensed and holds no QEMU code. On top of it, `ruvm-hw-virtio` has the virtio device core, the virtio-mmio transport, and the rng, console and blk devices (#70), plus the virtio-pci transport (#77).

The PC side got the i8042 controller with PS/2 keyboard and mouse (#66), and a PCI core with config access, bridges, MSI and MSI-X (#67). The q35 host bridge, PAM and PCIe MMCONFIG followed (#69). Also added: ACPI PM registers, ICH9 PM and the generic event device (#68), the ICH9 LPC bridge with APM (#71), and the ICH9 AHCI controller with ATA and ATAPI disks (#73).

`ruvm-target-x86` has the CPU reset state, the named CPU models, CPUID and MSRs (#72). It can also push that state into a KVM vCPU and read it back (#75). The KVM hlt tests now keep the irqchip in userspace, because a guest `hlt` never exits to userspace when the in-kernel APIC is on (#76).

`ruvm-machine-x86` assembles the microvm board (#74) and the q35 board (#78). Each one has its memory map, interrupt routing, fw_cfg, device tree or ACPI inputs, and reset.

## 0.2.1

This is the first patch on the way to M2, and most of it is the PC chipset. Nothing boots yet, but the devices a microvm or q35 guest touches first are ported and tested against QEMU's behavior at the register level.

Guest RAM is now backed by an anonymous host mapping through `ruvm-sys` (#56), which is what KVM needs to map it into a guest. `ruvm-hw-core` has IRQ lines and device timers (#55) and fw_cfg with the DMA interface (#60).

The legacy devices are in: the 16550 UART (#57), the MC146818 RTC (#58), the i8254 PIT and the i8259 PIC pair (#59), and the HPET and the IOAPIC (#61).

`ruvm-accel-kvm` opens `/dev/kvm` with QEMU's checks and messages, sets up the in-kernel or split irqchip, keeps KVM memory slots in step with the guest memory map and runs vCPUs, sending port and MMIO exits into the address spaces (#62). CI now gives the Linux runner access to `/dev/kvm` so those tests run for real there.

`ruvm-firmware` can work out a direct kernel boot: the bzImage setup header for every boot protocol, initrd placement, PVH kernels and the e820 table, all producing the fw_cfg items QEMU would add (#63).

## 0.2.0

M1 is done. The core runtime pieces that every machine sits on are ported and checked against QEMU: QOM, QAPI and QMP, the monitor, chardevs, the option parser, vmstate, the memory API and the block layer core. ruvm still cannot run a guest, that is what M2 is for.

`ruvm-mem` has memory regions, flat views, address spaces and dirty tracking, ported from system/memory.c and physmem.c (#47). Flat views are rendered the same way QEMU renders them, and a property test compares them against a straight port of the reference algorithm. It passed a million cases for both the narrow and the wide address space variants before this release.

The `none` machine has sysbus and shows its memory regions in the QOM tree (#48), so `qom-list` and `info mtree` style queries line up with QEMU.

`ruvm-block` has the node graph with the `null-co`, `null-aio` and `blkdebug` drivers, plus `blockdev-add` and `blockdev-del` with QEMU's errors (#49). The `query-qmp-schema` comparison now uses a checked-in normalization list, so a schema drift shows up as a diff in review (#50).

M2 work has started. `ruvm-firmware` has an AML builder, the BIOS linker and loader, and the full ACPI table sets for microvm (#51) and q35 (#52, #53). Every table in the default bios-tables-test runs for both machines matches QEMU byte for byte, including the q35 DSDT with PCI and CPU hotplug.

## 0.1.5

ruvm starts a machine now. It is only the empty `none` machine with the `qtest` accelerator, but that is how QEMU's own tests launch QEMU, so a libqtest style command line comes up. It serves QMP and the qtest protocol on its sockets and shuts down cleanly on `quit` or a signal.

The startup path from system/vl.c is ported to `ruvm-system` (#45). That covers the option loop for the monitor, chardev, object, machine, accel, name, display, audio and qtest options. It also covers the order `qemu_init()` creates things in, the run state with `stop` and `cont` and their events, and the main loop with the SHUTDOWN event. Machines, accelerators and displays a QEMU build could leave out fail with QEMU's own messages. Options ruvm does not handle yet say so and name the option.

`ruvm-accel-qtest` is the qtest protocol server from system/qtest.c, with every command, reply and log line matching QEMU (#44).

## 0.1.4

The pieces a QMP monitor needs are in place, though nothing starts one from the command line yet. That is the next step.

`ruvm-monitor` is a QMP server ported from monitor/qmp.c (#34). It does capability negotiation, out of band commands, the request queue with its limit and the event rate limiting. File descriptors can be passed with `getfd`, `add-fd` and fd sets (#35). Monitors are QOM objects now, so `object-add` and `object-del` create and remove them with QEMU's error messages (#41).

`ruvm-chardev` has the `null` and `socket` backends, Unix and TCP, client and server, and serves QMP through them (#38). `-chardev` options and the old compat strings like `tcp:host:port,server=on` parse the same way as in QEMU (#39).

The option table is generated from qemu-options.hx (#36), and `QemuOpts` and the keyval parser are ported with their error messages (#37).

`ruvm-sys` turns SIGINT, SIGHUP and SIGTERM into a callback on a normal thread, keeping the sender's pid for the log line (#40).

`ruvm-vmstate` can save and load state from VMState descriptions, checked against the byte streams in test-vmstate.c (#42).

## 0.1.3

QMP commands can be marshalled and dispatched now. There is no socket to send them over yet, that comes with the monitor.

`ruvm-qapi-gen` generates a Rust type for every enum, struct, union and alternate in the schema, with visitors that make the same calls in the same order as QEMU's generated C, so bad arguments get the same errors (#31). Unions keep the tag inside the branch value, so the two cannot disagree.

`ruvm-qapi` has `qmp_dispatch` and the command table, ported from qmp-dispatch.c and qmp-registry.c, and the build generates a `register_*` function for each of the 227 generated commands and an `event_*` builder for each event (#32). Request errors, the preconfig check, disabled commands, out of band execution and the `-compat` policy all behave as in QEMU.

## 0.1.2

The object model is in, which is the first M1 piece that everything above it depends on.

`ruvm-qapi` has the visitors: QObject input and output, string input and output, keyval input and the forwarding visitor, with QEMU's error texts and its C number parsing (#28). It also has a hash table that iterates in the same order as GLib's, because `qom-list` and friends print properties in hash order and management tools see that order.

`ruvm-qom` is a port of qom/ (#29). It has type registration, interfaces, the class and instance hooks in QEMU's order, every property kind including child, link and alias, the composition tree with canonical paths and partial path resolution, user creatable objects, global and compat properties, and the QOM QMP commands from `qom-list` to `object-del`. The tests port check-qom-proplist.c and check-qom-interface.c.

## 0.1.1

The first pieces of M1. Nothing runs a guest yet, but the foundation crates and the QAPI layer are real code now.

`ruvm-base` has the error type with QEMU's error classes and message formats, `error_report` and `warn_report` with their location prefixes, bit and bitmap helpers, notifier lists, an RCU cell and a timer list (#22). The RCU cell is lock based for now and keeps the interface the epoch based one will have.

`ruvm-aio` has the event loop: one reactor per thread on mio, bottom halves, event notifiers, fd handlers, timers on four clocks, adaptive polling hooks, a thread pool and a small executor for async tasks (#24). io_uring comes with the block layer in M3.

`ruvm-qapi` has QEMU's QObject values and its JSON dialect (#25). The parser and the writer match QEMU 11.1 byte for byte, including key order in objects, the way doubles are printed, the error texts with their positions and the recovery after bad input.

`ruvm-qapi-gen` reads the vendored QAPI schema and generates introspection, and the `query-qmp-schema` reply built from it is identical to what QEMU 11.1 returns for the same build configuration (#26).

## 0.1.0

M0 is done. There is no emulator yet, but everything the emulator will be built inside is in place and checked on every pull request.

The workspace has all 107 crates from the catalog in `spec/24-workspace-layout.md`, each with its license, its layer and an unsafe budget of zero. `cargo xtask ci` runs the layer rule, the license provenance rule, the unsafe audit, the prose rules, the vendored input check, rustfmt, clippy, the tests and the docs, and CI runs the same thing on Linux x86_64 and aarch64, macOS and Windows, along with an MSRV build on 1.85, cargo-deny and actionlint with zizmor for the workflows (#14, #15).

`ruvm` dispatches on argv[0] to all 29 system emulators, the 38 user mode emulators and the 11 tools, and prints the same version text QEMU 11.1.0 prints for each one, so libvirt's version probe parses it (#16). Everything past `--version` exits with an error that says it is not implemented yet.

`vendor-qemu/` holds the QAPI schemas, the decodetree files, the trace-events files, the hx files, the ACPI expected tables and the target list from QEMU v11.1.0, with a manifest of their hashes. `cargo xtask upstream-sync <tag>` refreshes them and prints what changed (#17).

Tags now produce a release with attested archives for Linux, macOS and Windows, and `cargo xtask version` sets the version everywhere at once (#20).

The specification, the README and the license files came before all of this and are unchanged.
