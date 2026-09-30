# Changelog

Notable changes, newest first. This project is pre-1.0 and makes no compatibility promise about its own APIs until it has one. The compatibility it does promise is with QEMU 11.1, and each release says how much of that is real.

The minor version is the number of milestones finished. 0.1.0 is the release where M0 closes, 0.2.0 where M1 closes, and so on up to M12, which is 1.0. Patch releases come whenever enough has landed to be worth a tag. The milestones are the issues labeled `kind/milestone` at https://github.com/tamnd/ruvm/issues.

## Unreleased

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
