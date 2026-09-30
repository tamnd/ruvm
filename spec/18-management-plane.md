# 18. Management plane: QMP, HMP, command line, libvirt, gdbstub, tracing, guest agent

This document specifies everything that sits between an operator (a human, libvirt, a test harness, a debugger) and a running ruvm process. The crates involved are ruvm-monitor (QMP server and HMP), ruvm-system (command line parsing and the startup state machine, shared with document 04), ruvm-gdbstub, ruvm-trace, and the ruvm-ga guest agent binary. The reference is QEMU 11.1.0. Every wire format in this document is a compatibility surface in the sense of document 02: tools already written against QEMU must work against ruvm without a code change, and that includes tools that scrape human-readable output.

The management plane is where "drop-in replacement" is tested hardest. A guest never sees the monitor, but libvirt, OpenStack Nova, Proxmox, Kubevirt, virt-manager, the QEMU iotests, and countless shell scripts do. They depend on exact command names, exact error classes, exact event ordering, and in the HMP case on exact column layouts. The design rule is simple: the schema and the bytes on the wire come from QEMU, the implementation behind them is ours.

## Crate structure and threading

ruvm-monitor is an L5 crate. It depends on ruvm-qapi for generated types and the dispatch table, and on every subsystem crate only through the command traits that ruvm-qapi generates. It does not reach into device structs.

| Piece | QEMU equivalent | Thread |
| --- | --- | --- |
| Transport (chardev socket, stdio, fd) | chardev/ plus monitor/qmp.c I/O callbacks | monitor I/O thread (one reactor, ruvm-aio) |
| JSON streamer and parser | qobject/json-streamer.c, json-parser.c | monitor I/O thread |
| OOB dispatcher | monitor_qmp_dispatch in I/O thread | monitor I/O thread |
| In-band dispatcher | monitor_qmp_dispatcher_co | main thread |
| Event emitter and throttle | monitor/monitor.c monitor_qapi_event_queue | any thread, lock-free queue to I/O thread |
| HMP line editor and command table | monitor/hmp.c, readline.c, hmp-commands*.hx | main thread |

QEMU runs the monitor I/O in a dedicated "mon_iothread" whenever any monitor is QMP, and runs in-band commands in a coroutine on the main loop under the BQL. ruvm keeps the same two-level split because OOB semantics depend on it, but the in-band dispatcher does not take a global lock. Section "Dispatch without the BQL" below covers how each command acquires exactly the state it needs.

## QMP server

### Wire protocol

The protocol is the one in QEMU's docs/interop/qmp-spec.rst, implemented byte for byte. On connect the server writes the greeting:

```json
{"QMP": {"version": {"qemu": {"micro": 0, "minor": 1, "major": 11}, "package": "ruvm 1.0.0"}, "capabilities": ["oob"]}}
```

The `version.qemu` triple reports the QEMU release ruvm is compatible with (11.1.0 for the first release series), never ruvm's own version, because libvirt gates features on it (libvirt's qemu_capabilities.c refuses anything older than QEMU 7.2.0 today). The `package` string is free-form in QEMU and is where ruvm identifies itself. `query-version` returns the same structure. The capabilities array lists `oob` only when the monitor requires an I/O thread (`monitor_requires_iothread` in monitor/qmp.c); ruvm uses the same predicate.

The session starts in capabilities negotiation mode. Only `qmp_capabilities` is accepted; any other command returns `CommandNotFound` with QEMU's exact message ("Expecting capabilities negotiation with 'qmp_capabilities'"). `qmp_capabilities` takes an optional `enable` list; unknown capabilities produce `GenericError` with the QEMU message text. After negotiation the session is in command mode and `qmp_capabilities` itself becomes an error, again with QEMU's wording. Error text matters because some management code matches on `desc` strings even though the spec says not to. ruvm keeps a table of QEMU error strings generated from QEMU's source by a build-time extraction script and fails CI when a string drifts (document 22).

Requests are JSON objects with `execute` or `exec-oob`, optional `arguments`, optional `id` of any JSON type. Responses are `{"return": ..., "id": ...}` or `{"error": {"class": ..., "desc": ...}, "id": ...}`. Error classes are the fixed QAPI set (`GenericError`, `CommandNotFound`, `DeviceNotActive`, `DeviceNotFound`, `KVMMissingCap`) from qapi/error.json. Framing is by JSON value boundary, not by newline, so the streamer implements the same brace and bracket counting as json-streamer.c, including QEMU's limits from qobject/json-streamer.c: `MAX_TOKEN_SIZE` 64 MiB, `MAX_TOKEN_COUNT` 2 Mi tokens, `MAX_NESTING` 1024; exceeding any of them is an error that resets the parser. Output is compact by default and pretty-printed with `-qmp-pretty`, where pretty printing must use QEMU's exact indentation (four spaces, newline after every element) because iotests compare it textually.

### OOB execution

A command whose schema says `'allow-oob': true` may be sent with `exec-oob` once the client enabled `oob`. In QEMU 11.1 the OOB-capable commands are `migrate-recover`, `migrate-pause`, `yank`, and `query-yank`. They exist for one reason: to break a stuck main loop, for example when a postcopy migration stalls because the network died and the main thread is blocked on a page fault. ruvm keeps this list exactly and adds nothing to it without a QEMU counterpart, because `allow-oob` is visible in `query-qmp-schema`.

OOB commands execute directly in the monitor I/O thread and must never block on anything that an in-band command could hold. The ruvm type system enforces this: the codegen in ruvm-qapi emits OOB handlers against a `OobContext` that does not expose the control lock or any device domain lock, only the migration state machine's atomic control word, the yank registry (an RCU list of yank instances with their callbacks), and the monitor itself. A handler that tries to reach anything else fails to compile.

In-band requests go to a per-monitor queue. QEMU bounds it with `QMP_REQ_QUEUE_LEN_MAX = 8` in monitor/monitor-internal.h: when a monitor with OOB enabled has 8 pending in-band requests, the I/O thread stops reading from that client until the queue drains, and without OOB the monitor is suspended after every request so requests are strictly serialized. ruvm uses the same constant and the same suspend and resume rules, because clients that pipeline requests observe the backpressure. An OOB request can overtake queued in-band requests, which is the point; responses therefore can arrive out of order and the client must use `id`. When OOB is not enabled, `exec-oob` is rejected with QEMU's error.

### Events

Events are `{"event": NAME, "data": {...}, "timestamp": {"seconds": s, "microseconds": us}}`, emitted to every monitor in command mode. The QAPI schema of 11.1 defines about 60 events (SHUTDOWN, RESET, STOP, RESUME, DEVICE_DELETED, BLOCK_JOB_READY, JOB_STATUS_CHANGE, MIGRATION, MIGRATION_PASS, GUEST_PANICKED, NETDEV_STREAM_CONNECTED and so on). Two properties are part of the contract and easy to break in a multithreaded design.

The first is rate limiting. QEMU throttles guest-triggerable events to one per second per event type, in the `monitor_qapi_event_conf` table in monitor/monitor.c: RTC_CHANGE, BLOCK_IO_ERROR, WATCHDOG, BALLOON_CHANGE, QUORUM_REPORT_BAD, QUORUM_FAILURE, VSERPORT_CHANGE, MEMORY_DEVICE_SIZE_CHANGE, HV_BALLOON_STATUS_REPORT, each with a 1000 ms period. The algorithm is not a token bucket: the first event in a quiet period is sent immediately, later events within the period overwrite a single pending slot, and when the timer fires the pending event (the most recent one) is sent and a new period starts. For some events the throttle key includes a data member (`qapi_event_throttle_hash` in monitor.c: VSERPORT_CHANGE per `id`, QUORUM_REPORT_BAD per `node-name`, MEMORY_DEVICE_SIZE_CHANGE and BLOCK_IO_ERROR per `qom-path`), so that one noisy port does not hide another. BLOCK_IO_ERROR with `action` equal to `stop` bypasses the throttle entirely, because the VM stops and the management tool must see every such event. ruvm reproduces the table and the key functions exactly. The clock is realtime, except under qtest where QEMU uses the virtual clock so tests can step through rate limits; ruvm-accel-qtest provides the same switch.

The second is ordering. An event emitted by a subsystem must reach the wire before the response of any QMP command that was dispatched after the state change it reports. QEMU gets this for free because both happen under the BQL on one thread. ruvm emits events into a per-process multi-producer queue with a global sequence number taken at the moment of emission, and in-band responses carry the sequence number current at completion; the I/O thread writes events with a lower sequence number before the response. This is what makes "send `device_del`, wait for DEVICE_DELETED" and "send `stop`, see STOP before the return" behave as in QEMU.

### Async jobs

Long operations (block jobs, `blockdev-create`, `snapshot-save`, `snapshot-load`, `snapshot-delete`, `dump-guest-memory` with `detach`) follow QEMU's job model from qapi/job.json: the command returns immediately and progress is reported through JOB_STATUS_CHANGE with the state machine `created, running, paused, ready, standby, waiting, pending, aborting, concluded, null`, plus the legacy BLOCK_JOB_* events for block jobs. `query-jobs` and `query-block-jobs` return the same fields QEMU returns, including `current-progress` and `total-progress`. The job state machine lives in ruvm-block (document 14) and in ruvm-migration for snapshot jobs (document 17); the monitor only relays. The transition table is generated from the same data as QEMU's `JobVerbTable` and `JobSTT` in job.c so illegal verbs produce the same errors.

### File descriptor passing

QEMU accepts file descriptors from the management layer over a UNIX socket using SCM_RIGHTS ancillary data attached to a QMP message, and ruvm does the same. Received descriptors are held in a per-monitor list until a command consumes them. Two consuming APIs exist and both are required by libvirt.

`getfd fdname=NAME` takes the most recently received descriptor and binds it to a name in the monitor's named-fd table, and `closefd` removes it. Consumers such as `netdev_add ... fd=NAME`, `-chardev socket,fd=`, and `migrate fd:NAME` look names up with QEMU's `monitor_fd_param` semantics: a string that parses as a decimal integer is treated as a raw descriptor number only on the command line, while through QMP it is a name.

`add-fd fdset-id=N opaque=STR` puts the received descriptor into fd set N. Paths of the form `/dev/fdset/N` anywhere QEMU opens a file (block protocol `file`, `-drive file=`, chardev file backend, `-add-fd` at startup) are resolved by `qemu_open` semantics: pick a descriptor from the set whose access mode matches the requested `O_ACCMODE`, `dup` it, and apply other flags with `fcntl`. `remove-fd` and `query-fdsets` complete the API. The subtle rule that ruvm must reproduce is lifetime: an fd set whose descriptors were all removed is freed only when no dup'ed descriptor is still in use, and fd sets added by a monitor are cleaned up when that monitor disconnects unless they are in use. libvirt relies on this for `-add-fd` passed disks and for migration to and from files.

On Windows the equivalent is `get-win32-socket`, which takes a WSAPROTOCOL_INFOW blob in base64. ruvm implements it in ruvm-sys and exposes it only when built for Windows, following QEMU's `'if': 'CONFIG_WIN32'`.

## QAPI schema and command groups

ruvm vendors QEMU's qapi/*.json files unchanged (45 modules in 11.1, from error.json to uefi.json, included by qapi/qapi-schema.json) and generates Rust from them at build time with ruvm-qapi (document 04). `query-qmp-schema` returns the same introspection data QEMU returns for the same build configuration, including the masked type names (`"1"`, `"2"`, ...) that QEMU's introspect.py assigns, because libvirt's capability code walks that tree by path (for example `blockdev-add/arg-type/+file/aio` style queries in qemu_qapi.c). Conditional members (`'if': 'CONFIG_...'`) are resolved from ruvm's cargo features so that a build without SPICE reports no SPICE commands, exactly like a QEMU built without SPICE.

The 11.1 schema defines about 250 commands. The table groups them by the subsystem that owns them and the locking class each group needs in ruvm.

| Schema module | Representative commands | Owner crate | Lock class |
| --- | --- | --- | --- |
| control.json, introspect.json | qmp_capabilities, query-version, query-commands, query-qmp-schema, quit | ruvm-monitor | none |
| misc.json | stop, cont, x-exit-preconfig, human-monitor-command, getfd, add-fd, query-fdsets, query-iothreads, query-command-line-options | ruvm-system, ruvm-monitor | runstate |
| run-state.json | query-status, set-action, watchdog-set-action | ruvm-system | runstate |
| machine.json | query-cpus-fast, query-machines, query-hotpluggable-cpus, system_reset, system_powerdown, memsave, pmemsave, query-cpu-model-expansion, query-cpu-definitions, balloon, x-query-jit | ruvm-machine-*, ruvm-accel | snapshot or control |
| qom.json, qdev.json | qom-list, qom-get, qom-set, qom-list-get, qom-list-types, object-add, object-del, device_add, device_del, device-list-properties | ruvm-qom, ruvm-hw-core | control (writes), snapshot (reads) |
| block-core.json, block.json, block-export.json, transaction.json, job.json | blockdev-add, blockdev-reopen, blockdev-mirror, block-commit, block-stream, query-block, query-named-block-nodes, block-dirty-bitmap-*, nbd-server-*, block-export-add, transaction, job-* | ruvm-block | block graph lock |
| migration.json | migrate, migrate-incoming, migrate-set-capabilities, migrate-set-parameters, query-migrate, migrate-recover, migrate-pause, snapshot-save, calc-dirty-rate | ruvm-migration | migration actor |
| net.json | netdev_add, netdev_del, set_link, query-rx-filter, announce-self | ruvm-net | control |
| char.json | chardev-add, chardev-change, chardev-remove, ringbuf-read, ringbuf-write | ruvm-chardev | control |
| ui.json, audio.json | screendump, send-key, input-send-event, query-vnc, set_password, display-reload, query-audiodevs | ruvm-ui, ruvm-audio | ui actor |
| dump.json | dump-guest-memory, query-dump | ruvm-system | runstate plus job |
| misc-i386.json | query-sev, query-sev-launch-measure, sev-inject-launch-secret, query-sev-attestation-report, query-sgx, rtc-reset-reinjection | ruvm-target-x86, ruvm-accel-kvm | snapshot |
| misc-arm.json, machine-s390x.json | query-gic-capabilities, set-cpu-topology, query-s390x-cpu-polarization | target crates | snapshot or control |
| stats.json, accelerator.json | query-stats, query-stats-schemas, query-kvm, query-accelerators, x-accel-stats | ruvm-accel | snapshot |
| virtio.json, pci.json, cxl.json, acpi*.json, tpm.json, cryptodev.json, rocker.json | x-query-virtio*, query-pci, cxl-inject-*, query-acpi-ospm-status, query-tpm | device crates | device domain |
| replay.json, trace.json, yank.json | replay-break, replay-seek, trace-event-set-state, yank | ruvm-migration, ruvm-trace | varies |

### Dispatch without the BQL

QEMU runs every in-band QMP command on the main thread holding the BQL (commands marked `'coroutine': true` may yield but still hold it when running). ruvm has no BQL (document 03), so every command handler declares what it touches, and the generated dispatcher acquires it. The declaration is a Rust attribute on the handler, checked at compile time against the argument types the handler is given.

```rust
#[qmp_handler(command = "qom-get", class = Snapshot)]
fn qom_get(ctx: &SnapshotCtx, path: &str, property: &str) -> QmpResult<serde_json::Value>;

#[qmp_handler(command = "device_add", class = Control)]
fn device_add(ctx: &mut ControlCtx, opts: DeviceAddOpts) -> QmpResult<()>;

#[qmp_handler(command = "blockdev-mirror", class = Actor(BlockGraph))]
async fn blockdev_mirror(ctx: BlockCtx, args: BlockdevMirrorArgs) -> QmpResult<()>;
```

The classes are:

- `None`: pure monitor state (capabilities, schema, version). Runs inline.
- `Snapshot`: read-only queries. The handler receives an RCU read guard on the composition tree (document 04) and reads properties through getters that take each object's own lock briefly. A query never holds more than one device lock at a time, so it cannot deadlock against vCPU threads. The result is consistent per object, not globally, which matches what QEMU clients can actually observe: QEMU's `query-cpus-fast` is also a set of per-CPU reads that race with running vCPUs.
- `Runstate`: stop, cont, system_reset, dump, set-action. Takes the runstate mutex, which serializes transitions of the `RunState` machine from qapi/run-state.json and the "pause all vCPUs" barrier (document 06).
- `Control`: tree mutations (device_add, device_del, object-add, netdev_add, qom-set on a writable property, chardev-add). Takes the single control lock from the canon concurrency model. `qom-set` is `Control` because arbitrary properties can have side effects on realized devices; a property may opt into a finer class with `#[property(set_class = Device)]` when its setter only touches the device's own state (for example `guest-stats-polling-interval` on virtio-balloon).
- `Actor(X)`: subsystems with their own event loop (block graph, migration, UI) receive the command as a message and answer through a oneshot channel. The in-band dispatcher awaits the reply without holding any lock. The block graph actor maps to QEMU's graph lock and AioContext rules (document 14): commands run in the main context and drain affected nodes as QEMU does.
- `DeviceDomain(D)`: device-specific commands (`cxl-inject-poison`, `x-query-virtio-queue-element`, `rtc-reset-reinjection`) take the lock of the device's domain only.

The in-band dispatcher still executes one command at a time per monitor, and the process keeps a global in-band order across monitors for commands in classes `Runstate` and `Control`, because QEMU's single-threaded execution is observable: two management clients racing `device_add` and `stop` see a total order in QEMU and must see one in ruvm. Snapshot queries from different monitors run in parallel. This is where ruvm wins on large hosts: a Kubevirt style agent polling `query-stats` and `query-blockstats` every second on 200 VMs does not contend with anything on the vCPU path.

`human-monitor-command` runs the HMP line through the HMP dispatcher (below) and inherits the class of the underlying HMP command. `transaction` takes the block graph actor and, for actions that are not block actions (none today), would escalate to `Control`.

### Policy handling: -compat and feature flags

QAPI marks members with features `deprecated` and `unstable` (the `x-` prefix is the naming convention for the latter). `-compat deprecated-input=accept|reject|crash,deprecated-output=accept|hide,unstable-input=...,unstable-output=...` changes dispatch: `reject` returns an error naming the deprecated member, `crash` aborts (for testing management code), `hide` drops deprecated members from output and suppresses deprecated events. ruvm implements this in the generated visitors, not in handlers, the same place QEMU does (qapi/qapi-visit-core.c `visit_deprecated_accept` and friends). QEMU master has a pending third axis, `-compat insecure-types=accept|warn|reject`, tied to a per-type `secure` flag; document 19 describes how ruvm adopts it.

## HMP

HMP is the human monitor. It is officially not a stable interface, but in practice it is scraped: `virsh qemu-monitor-command --hmp`, OpenStack debugging runbooks, Proxmox's `qm monitor`, CI scripts that grep `info status` or `info block`, and the iotests and functional tests that parse `info` output. ruvm treats HMP output text as a compatibility surface with golden files.

### Command set

The command table is generated from QEMU's hmp-commands.hx and hmp-commands-info.hx, vendored like the QAPI schema. The generator reads the `.name`, `.args_type`, `.params`, `.help` fields and the SRST documentation blocks, so `help` and `help info` print exactly QEMU's text, and each entry binds to a Rust function by name. In 11.1 the top-level table has these commands:

help (?), commit, quit (q), exit_preconfig, block_resize, block_stream, block_job_set_speed, block_job_cancel, block_job_complete, block_job_pause, block_job_resume, eject, drive_del, change, screendump, logfile, trace-event, trace-file, log, savevm, loadvm, delvm, one-insn-per-tb, stop (s), cont (c), system_wakeup, gdbserver, x, xp, gpa2hva, gpa2hpa, gva2gpa, print (p), i, o, sendkey, sync-profile, system_reset, system_powerdown, sum, device_add, device_del, cpu, mouse_move, mouse_button, mouse_set, wavcapture, stopcapture, memsave, pmemsave, boot_set, nmi, ringbuf_write, ringbuf_read, announce_self, migrate, migrate_cancel, migrate_continue, migrate_incoming, migrate_recover, migrate_pause, migrate_set_capability, migrate_set_parameter, migrate_start_postcopy, x_colo_lost_heartbeat, client_migrate_info, dump-guest-memory, dump-skeys, migration_mode, snapshot_blkdev, snapshot_blkdev_internal, snapshot_delete_blkdev_internal, drive_mirror, drive_backup, drive_add, pcie_aer_inject_error, netdev_add, netdev_del, object_add, object_del, hostfwd_add, hostfwd_remove, balloon, set_link, watchdog_action, nbd_server_start, nbd_server_add, nbd_server_remove, nbd_server_stop, mce, getfd, closefd, block_set_io_throttle, set_password, expire_password, chardev-add, chardev-change, chardev-remove, chardev-send-break, qemu-io, qom-list, qom-get, qom-set, replay_break, replay_delete_break, replay_seek, calc_dirty_rate, set_vcpu_dirty_limit, cancel_vcpu_dirty_limit, dumpdtb, xen-event-inject, xen-event-list, info.

The `info` subcommands are: version, network, chardev, block, blockstats, block-jobs, registers, lapic, cpus, history, irq, pic, pci, tlb, mem, mtree, jit, sync-profile, accel, kvm, accelerators, numa, usb, usbhost, capture, snapshots, status, mice, vnc, spice, name, uuid, usernet, migrate, migrate_capabilities, migrate_parameters, balloon, qtree, qdm, qom-tree, roms, trace-events, tpm, memdev, memory-devices, iothreads, rocker, rocker-ports, rocker-of-dpa-flows, rocker-of-dpa-groups, skeys, cmma, dump, ramblock, hotpluggable-cpus, vm-generation-id, memory_size_summary, sev, replay, dirty_rate, vcpu_dirty_limit, sgx, via, stats, virtio, virtio-status, virtio-queue-status, virtio-vhost-queue-status, virtio-queue-element, cryptodev, firmware-log.

Some entries only exist for particular targets or build options (`info lapic` on x86, `info skeys` and `info cmma` on s390x, `info sev` and `info sgx` on x86, `info spice` with SPICE). The generator honours the `#if defined(TARGET_...)` and `CONFIG_*` conditionals in the .hx files by mapping them to cargo features and target crates.

### Argument parsing and output fidelity

HMP argument parsing uses the `args_type` mini-language from monitor/hmp.c (`s` string, `F` filename, `B` block device, `i` 32-bit int, `l` target-sized int, `M` size defaulting to MiB with suffixes, `o` size with suffixes, `T` double with optional `ms`/`us`/`ns`, `/` format for x and xp, `b` bool on/off, `-x` flags, `O` QemuOpts, `S` rest of line, `?` optional). ruvm ports `monitor_parse_arguments` and the expression evaluator used by `x`, `xp`, `print`, `i`, `o`, and `sum` (which accepts registers like `$pc` and `$eax` resolved through `GuestArch`'s monitor register table, document 09). Tab completion (readline.c's completion callbacks for device names, block nodes, chardev ids, migrate capabilities) is ported so interactive users on `-monitor stdio` get the same experience.

Output formatting is where fidelity costs real effort. Most HMP commands in QEMU are thin wrappers that call the QMP handler and format its result (hmp_info_block in block/monitor/block-hmp-cmds.c, hmp_info_migrate in migration/migration-hmp-cmds.c), so ruvm does the same: an HMP formatter is a pure function from the QAPI result type to text, written by porting QEMU's `monitor_printf` sequences literally, including spacing, capitalization, and the odd cases (for example `info status` printing `VM status: paused (prelaunch)`, `info migrate` printing size values with `qemu_strtosz`-inverse units, `info block` printing `Removable device: not locked, tray closed`). Commands whose output is not derived from QMP (`info registers`, `info tlb`, `info mem`, `info mtree`, `info qtree`, `info jit`, `info pic`, `info lapic`) print through per-target or per-device dump hooks that must match QEMU's format for the same architectural state; the `info registers` format in particular comes from each target's `cpu_dump_state` (for x86, target/i386/cpu-dump.c) and is used by gdb scripts and CI log checks.

The fidelity test is differential (document 22): for each HMP command in a corpus of about 400 invocations across machine types, run QEMU 11.1 and ruvm with the same configuration, freeze the guest at the same point via qtest, and compare output byte for byte after masking a documented list of volatile fields (host addresses, timings, pids). `info jit` is exempt because ruvm's JIT statistics differ by design; it prints the same header lines and then ruvm's own counters.

HMP runs in the main thread for `-monitor` instances and never has OOB. The `readline` history and the `cpu` command's per-monitor "current CPU" state are per monitor, like QEMU.

## Command line

ruvm-system owns command line parsing. The object model side (how parsed options become QOM objects, compat properties, and machine versions) is in document 04; this section covers the parsing contract and the management-visible behaviours.

### The option table

QEMU's option list is qemu-options.hx: 115 `DEF(...)` entries in 11.1, each with a name, an argument flag (`HAS_ARG` or 0), an enum id, help text, and an architecture mask (`QEMU_ARCH_ALL`, `QEMU_ARCH_I386`, and so on). Current options include -machine, -cpu, -accel, -smp, -numa, -add-fd, -set, -global, -boot, -m, -mem-path, -mem-prealloc, -audiodev, -device, -blockdev, -drive, -netdev, -nic, -chardev, -tpmdev, -object, -bios, -pflash, -kernel, -shim, -append, -initrd, -dtb, -compat, -fw_cfg, -serial, -monitor, -qmp, -qmp-pretty, -mon, -pidfile, -preconfig, -S, -overcommit, -gdb, -s, -d, -D, -dfilter, -icount, -incoming, -only-migratable, -nodefaults, -sandbox, -readconfig, -no-user-config, -trace, -plugin, -qtest, -run-with, -msg, -dump-vmstate, -perfmap, -jitdump, and the legacy shorthands (-hda, -cdrom, -fda, -net, -usbdevice, -vga, -nographic, -enable-kvm).

ruvm vendors qemu-options.hx and generates three things from it: the option enum and lookup table used by the parser, the `-help` text (byte-identical, because humans and some older scripts read it), and the data behind `query-command-line-options`. The architecture mask is applied per binary name, so `qemu-system-aarch64 -help` does not list x86-only options, as in QEMU. Option lookup follows `lookup_opt` in system/vl.c: a leading `--` is accepted as `-`, options are matched exactly (no prefix abbreviation), and `HAS_ARG` options consume the next argv element even if it starts with `-`.

Parsing is two-pass like QEMU. The first pass handles options that must be known before anything else (`-nodefaults`, `-no-user-config`, `-readconfig`, `-trace`, `-d`, `-D`, `-sandbox` is recorded but applied late, `-run-with`). The second pass processes the rest in order, because order is significant for `-set`, `-global` precedence, and legacy options that create devices.

### QemuOpts, keyval, and JSON

QEMU has two option syntaxes and both are part of the contract.

QemuOpts (util/qemu-option.c) is the old `key=value,key=value` syntax with an implied first key for some groups (`-drive file=...`, `-netdev user,id=n0`, where `user` fills the `type` key), `,,` as an escaped comma, `help` and `?` as special values, and duplicate keys where the last one wins. Each group (`drive`, `netdev`, `chardev`, `machine`, `accel`, `smp-opts`, and so on) has a descriptor list that may be empty (accept anything, validate later) or fixed. ruvm ports the parser and the group descriptors as data.

keyval (util/keyval.c) is the newer syntax used by `-blockdev`, `-audiodev`, `-object`, `-device` in its JSON-capable form, `-display`, and `-compat`. It supports dotted keys that build nested objects (`file.driver=file,file.filename=x.img`), list indexing (`server.0.host=...`), and an implied key. keyval output is a QDict of strings that is then visited by the QAPI input visitor with string-to-type conversion (`qobject_input_visitor_new_keyval`). ruvm ports keyval exactly, including its error messages.

JSON syntax: when the argument of `-blockdev`, `-device`, `-object`, `-audiodev`, `-netdev`, or `-compat` starts with `{`, it is parsed as JSON and visited with the strict QAPI input visitor, giving the same typing as QMP. Current libvirt generates JSON for `-blockdev`, `-device`, `-object`, and `-netdev` whenever the probed QEMU supports it. For `-device` the JSON path also enforces that property values have the right JSON type, which is different from the string path where everything is a string parsed by the property's setter. ruvm keeps that difference: a JSON `-device` with `"bootindex": "1"` (a string) is rejected by QEMU's strict visitor and must be rejected by ruvm too, while the same value in `bootindex=1` form is accepted.

### Config files, -set, -global

`-readconfig file` reads an INI-style file with `[group "id"]` sections (the format of `qemu_config_parse` in util/qemu-config.c), and is still supported in 11.1. `-writeconfig` was removed in QEMU 7.1 as "a failed experiment" and ruvm does not implement it; the binary rejects it with QEMU's unknown-option message. `-nodefconfig` is long gone; `-no-user-config` disables loading the default config files from sysconfdir, which in ruvm are looked up under the same paths (`/etc/qemu/` and the build prefix) so distribution packaging keeps working.

`-set group.id.key=value` modifies an option set created earlier by id (it errors if the id does not exist). `-global driver.property=value` (or the old `driver.property=value` without `-global`) registers a global property applied when a device of that type (or a subtype) is created, with the same precedence as QEMU: machine compat props first, then accelerator compat props, then `-global` in command line order, then the device's own properties. `-global` also affects devices added later with `device_add`. The `used` flag on globals is tracked so that QEMU's warning "Warning: global DRIVER.PROP has invalid class name" and the "not used" check (`qdev_prop_check_globals`) produce the same diagnostics.

Removed options stay removed. ruvm tracks QEMU's docs/about/removed-features.rst as data (for example `-chroot` removed in 9.0 in favour of `-run-with chroot=`, `-runas` removed in 10.0 in favour of `-run-with user=`, `-singlestep` replaced by `-accel tcg,one-insn-per-tb=on` in 9.0, `-no-hpet` removed in 9.0) and emits QEMU's error for each. Deprecated options (docs/about/deprecated.rst) are accepted with QEMU's deprecation warning text.

### Preconfig

`-preconfig` stops startup after the machine object exists but before it is initialized (QEMU's `PHASE_MACHINE_CREATED`), and runs the main loop with only QMP commands marked `'allow-preconfig': true` available. This is used to configure NUMA with `set-numa-node` after querying `query-hotpluggable-cpus`, which needs the machine type's CPU topology but must happen before CPUs are created. `x-exit-preconfig` (HMP `exit_preconfig`) continues startup. ruvm implements the phases from hw/core/machine.c's `MachineInitPhase` (`PHASE_NO_MACHINE`, `PHASE_MACHINE_CREATED`, `PHASE_ACCEL_CREATED`, `PHASE_LATE_BACKENDS_CREATED`, `PHASE_MACHINE_INITIALIZED`, `PHASE_MACHINE_READY`) as a state enum in ruvm-system, and each QMP handler's allowed phases are checked by the dispatcher from the schema's `allow-preconfig` flag. QEMU has for years discussed a fully QMP-driven startup (`-M none` plus configuration commands); ruvm's phase machine is written so that such a mode can be added without restructuring, but ruvm does not add non-QEMU commands for it.

### Native ruvm CLI

`ruvm run` and friends (canon) are sugar: they build the same internal config model and can print the equivalent QEMU command line with `ruvm run --print-qemu-cmdline`. They never accept anything that cannot be expressed as a QEMU command line plus QMP, so a VM started with the native CLI can always be reproduced with qemu-system-*.

## libvirt interaction

libvirt is the most important management client. Its QEMU driver (src/qemu/) probes capabilities per binary, caches them, builds a command line, and then drives the VM entirely over QMP. ruvm's M11 milestone is to pass libvirt's own test suite and the TCK style functional tests against ruvm binaries.

### Capability probing

libvirt stopped parsing `-help` output for QEMU 1.2.0 and later in 2014 (it now reports "too new for help parsing" if QMP probing fails), and its current minimum is QEMU 7.2.0 (`QEMU_MIN_MAJOR 7`, `QEMU_MIN_MINOR 2` in qemu_capabilities.c). Probing is QMP-only. `qemuProcessQMPLaunch` in qemu_process.c starts the binary as:

```
qemu-system-x86_64 -S -no-user-config -nodefaults -nographic -machine none,accel=kvm:tcg -qmp unix:/var/lib/libvirt/qemu/qmp-XXXXXX/qmp.monitor,server=on,wait=off -pidfile .../qmp.pid -daemonize
```

and for x86 repeats the probe with `accel=tcg` to learn TCG CPU models. So ruvm must support `-machine none` (a machine with no devices and no CPUs, where CPU model queries still work), `-daemonize` with the exact synchronization libvirt expects (the parent exits only after the monitor socket exists), and a `-pidfile` written before daemonize returns.

In the probe session libvirt calls, in roughly this order: `qmp_capabilities`, `query-version`, `query-target`, `query-qmp-schema` (the main feature detector, walked by `virQEMUQAPISchemaPathGet`), `query-machines` (with `compat-props` and the `default-ram-id` and `acpi` fields), `query-cpu-definitions`, `query-cpu-model-expansion` with `type=static` and `full` on `host` and `max`, `qom-list-types` (for device and object availability, including `abstract=true` queries), `qom-list-properties` and `device-list-properties` on a fixed list of types (virtio-blk-pci, virtio-net-pci, scsi-disk, usb-host, memory-backend-file, max-x86_64-cpu, and dozens more), `query-command-line-options` (for `-machine`, `-spice`, `-sandbox`, `-object` sub-options not visible in the schema), `query-kvm`, `query-accelerators`, `query-migrate-capabilities`, `query-tpm-models`, `query-tpm-types`, `query-gic-capabilities` on Arm, `query-sev-capabilities` and `query-sgx-capabilities` on x86, and `query-stats-schemas`. Every one of these must return what QEMU 11.1 returns for an equivalent build, and the probe must finish quickly: libvirt caches results keyed on the binary's ctime and the libvirt version, so a slow probe only hurts once per upgrade, but ruvm still targets under 100 ms for the whole session.

### Commands libvirt depends on at runtime

From libvirt's qemu_monitor_json.c (master, September 2026), the QMP commands libvirt issues are:

add_client, add-fd, announce-self, balloon, block_resize, block_set_io_throttle, block-commit, block-dirty-bitmap-remove, block-export-add, block-job-cancel, block-job-set-speed, block-latency-histogram-set, block-set-write-threshold, block-stream, blockdev-add, blockdev-close-tray, blockdev-create, blockdev-del, blockdev-insert-medium, blockdev-mirror, blockdev-open-tray, blockdev-remove-medium, blockdev-reopen, blockdev-set-active, calc-dirty-rate, chardev-add, chardev-remove, client_migrate_info, closefd, cont, device_add, device_del, device-list-properties, display-reload, dump-guest-memory, expire_password, getfd, human-monitor-command, inject-nmi, job-complete, job-dismiss, job-finalize, migrate, migrate_cancel, migrate-continue, migrate-incoming, migrate-pause, migrate-recover, migrate-set-capabilities, migrate-set-parameters, migrate-start-postcopy, nbd-server-start, nbd-server-stop, netdev_add, netdev_del, object-add, object-del, qmp_capabilities, qom-get, qom-list, qom-list-get, qom-list-properties, qom-list-types, qom-set, query-accelerators, query-balloon, query-block, query-block-jobs, query-blockstats, query-chardev, query-command-line-options, query-cpu-definitions, query-cpu-model-baseline, query-cpu-model-comparison, query-cpu-model-expansion, query-cpus-fast, query-current-machine, query-dirty-rate, query-dump, query-dump-guest-memory-capability, query-fdsets, query-gic-capabilities, query-hotpluggable-cpus, query-iothreads, query-jobs, query-kvm, query-machines, query-memory-devices, query-migrate, query-migrate-capabilities, query-migrate-parameters, query-named-block-nodes, query-pr-managers, query-qmp-schema, query-rx-filter, query-sev, query-sev-capabilities, query-sev-launch-measure, query-sgx-capabilities, query-stats, query-stats-schemas, query-status, query-target, query-version, remove-fd, rtc-reset-reinjection, screendump, send-key, set_link, set_password, set-action, sev-inject-launch-secret, snapshot-delete, snapshot-load, snapshot-save, stop, system_powerdown, system_reset, system_wakeup, transaction.

This list is the priority order for M1 through M11: a command on it is implemented before any command not on it, and each has a libvirt-driven test in ruvm's CI. libvirt also depends on events (SHUTDOWN with `guest` and `reason`, STOP, RESUME, RESET, DEVICE_DELETED, DEVICE_UNPLUG_GUEST_ERROR, BLOCK_JOB_*, JOB_STATUS_CHANGE, MIGRATION, MIGRATION_PASS, BLOCK_IO_ERROR, BLOCK_WRITE_THRESHOLD, GUEST_PANICKED, GUEST_CRASHLOADED, NIC_RX_FILTER_CHANGED, RTC_CHANGE, WATCHDOG, BALLOON_CHANGE, VSERPORT_CHANGE, MEMORY_DEVICE_SIZE_CHANGE, MEMORY_FAILURE, PR_MANAGER_STATUS_CHANGED, DUMP_COMPLETED, SPICE_* and VNC_*, NETDEV_STREAM_* and NETDEV_VHOST_USER_*), on `-S` plus `cont` for every start, on fd passing for tap devices, disks and migration, on `-sandbox on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny`, on `-msg timestamp=on`, on `-overcommit mem-lock=`, and on `-run-with` fields. `human-monitor-command` is used by `virsh qemu-monitor-command --hmp` passthrough and a small number of legacy paths.

libvirt also parses QEMU's stderr log in /var/log/libvirt/qemu/NAME.log for startup errors, and shows the last lines to the user on failure. ruvm's error messages for startup failures must therefore match QEMU's wording where libvirt or its tests match on them (for example "Could not access KVM kernel module", "cannot set up guest memory"). This list is maintained as part of the error string table mentioned in the QMP section.

## gdbstub

ruvm-gdbstub implements the GDB remote serial protocol with the same packet set as QEMU's gdbstub/gdbstub.c, gdbstub/system.c, and gdbstub/user.c, so gdb, lldb (in gdb-remote mode), IDA, Ghidra's debugger, and scripts using `target remote` work unchanged.

Transport and options: `-gdb dev` accepts any chardev spec (`tcp::1234`, `unix:path,server=on`), `-s` is shorthand for `-gdb tcp::1234`, and HMP `gdbserver` starts or stops it at runtime. In linux-user and bsd-user the stub listens on a port or UNIX socket given by `-g` (or `QEMU_GDB`) and handles guest `fork` the way QEMU's gdbstub/user.c does. The packet buffer size is QEMU's `MAX_PACKET_LENGTH`, advertised as `PacketSize` in the `qSupported` reply, because some clients size their reads from it.

Packets implemented: `?`, `c`, `C`, `s`, `D`, `g`, `G`, `p`, `P`, `m`, `M`, `H`, `T`, `z`/`Z` types 0 to 4, `vCont` and `vCont?`, `vAttach`, `vKill`, `qSupported`, `qAttached`, `qfThreadInfo`/`qsThreadInfo`, `qThreadExtraInfo`, `qOffsets` (user mode), `qRcmd` (monitor passthrough), `qGDBServerVersion`, `qXfer:features:read`, `qXfer:auxv:read` and `qXfer:exec-file:read` and `qXfer:siginfo:read` (user mode), `QCatchSyscalls` (user mode), `vFile:open`, `vFile:pread`, `vFile:close`, `vFile:readlink` (user mode host I/O), and QEMU's private extensions `qqemu.sstepbits`, `qqemu.sstep`, `Qqemu.sstep=`, `qqemu.PhyMemMode`, `Qqemu.PhyMemMode:`, and `qqemu.Supported`. Single-step flags (`SSTEP_ENABLE`, `SSTEP_NOIRQ`, `SSTEP_NOTIMER`) default to QEMU's values, so stepping over an instruction does not land in the timer interrupt handler.

Multiprocess: with `multiprocess+` negotiated, each CPU cluster is a gdb inferior (QEMU maps each `cpu-cluster` object to a process id, so a machine with an Arm Cortex-A cluster and a Cortex-R cluster exposes two processes with different target descriptions). Thread ids are `pPID.TID` with TID being the CPU index plus one. ruvm derives clusters from the same QOM objects, so board definitions that group CPUs in QEMU group them identically.

Target descriptions: `qXfer:features:read:target.xml` returns an XML document composed from the target's core feature files plus dynamically generated ones. QEMU keeps the static XML in gdb-xml/ (for example `aarch64-core.xml`, `i386-64bit.xml`, `riscv-64bit-cpu.xml`) and generates feature XML at runtime for system registers and vector registers (Arm `org.qemu.gdb.arm.sys.regs`, SVE `org.gnu.gdb.aarch64.sve` with the vector length of the current CPU, RISC-V CSRs). ruvm vendors gdb-xml/ and implements `GuestArch::gdb_features()` to produce the dynamic parts with identical register numbering, since gdb addresses registers by number in `p`/`P` packets and reordering breaks existing scripts.

Breakpoints and watchpoints: under the JIT (ruvm-jit), software breakpoints are implemented by invalidating translated blocks that contain the address and inserting a debug exception check at translation time, like QEMU's `cpu_breakpoint_insert` and `tb_invalidate_phys_addr`; the guest's memory is never modified. Watchpoints (`Z2` write, `Z3` read, `Z4` access) use the softmmu TLB: pages with a watchpoint get the `TLB_WATCHPOINT` flag and fall off the inline fast path into a slow path that checks ranges and raises a debug exit before the access completes (document 08). Under KVM, breakpoints and watchpoints go through `KVM_SET_GUEST_DEBUG` with hardware debug registers where available (four on x86, as many as the Arm host has), and software breakpoints use the architecture's breakpoint instruction patched into guest memory, as QEMU's `kvm_insert_breakpoint` does. HVF and WHPX follow their QEMU counterparts' capabilities, and the stub reports an error for watchpoints an accelerator cannot support instead of silently ignoring them.

Reverse debugging: when record/replay is active (`-icount shift=auto,rr=replay,rrfile=...`, document 17), the stub advertises `ReverseStep+` and `ReverseContinue+` and implements `bs` and `bc`. Reverse step replays from the nearest snapshot to the instruction before the current one; reverse continue replays forward from the previous snapshot while recording the last breakpoint or watchpoint hit, then replays again to stop there, which is QEMU's `replay_reverse_step` and `replay_reverse_continue` algorithm in replay/replay-debugging.c. The HMP commands `replay_break`, `replay_delete_break`, `replay_seek` and their QMP forms share the same machinery.

The stub runs in its own thread with a small reactor. It stops the VM through the runstate machinery (the same path as QMP `stop`) and reads registers through `Vcpu` get-register calls once all vCPUs are parked, so it never touches vCPU state while it runs.

## Tracing

### Trace events

QEMU declares trace points in `trace-events` files in each source directory, one per line: `name(type arg, type arg) "format"`, with optional properties (`disable`, `tcg`, `vcpu` in older trees). The tracetool Python script turns them into C inline functions per backend. The event names are a user-facing interface: `-trace enable=virtio_blk_*`, `-trace events=file`, HMP `trace-event NAME on`, QMP `trace-event-set-state` and `trace-event-get-state`, and `info trace-events` all address events by name and glob.

ruvm keeps QEMU's event names for every trace point that corresponds to QEMU behaviour. Each ruvm crate carries a `trace-events` file in QEMU syntax, and ruvm-trace's build-time generator reads them and emits a Rust macro per event. Where a ruvm component is a port of a QEMU component, its trace-events file starts as a copy of QEMU's and keeps the names and argument lists, so traces from QEMU and ruvm are comparable line for line in differential debugging. ruvm-only events use the prefix `ruvm_` so they cannot collide with future QEMU names.

Each generated event has a static `AtomicU16` dstate (the per-event enable count, the same concept as QEMU's `_TRACE_*_DSTATE`), checked with a relaxed load before argument evaluation. Disabled events cost one load and a predictable branch. The event registry is a linkme distributed slice, so `info trace-events` and QMP enumerate all events linked into the binary.

### Backends

QEMU selects backends at configure time (`--enable-trace-backends=`) from nop, log, simple, syslog, ftrace, dtrace, and ust, with log as the default. ruvm builds all portable backends in and selects at runtime, with the same names accepted wherever QEMU accepts them.

| QEMU backend | Output | ruvm implementation |
| --- | --- | --- |
| nop | none | events compiled out with the `trace-nop` cargo feature |
| log | stderr or `-D` file, only when `-d trace:` or `-trace` enables | writes QEMU's exact line format (`pid@sec.usec:name args`) via the ruvm-base logger |
| simple | binary trace file, `-trace file=` | writer thread with a lock-free ring, byte-compatible with QEMU's simpletrace format so scripts/simpletrace.py parses it |
| syslog | POSIX syslog at LOG_INFO | same |
| ftrace | writes to the tracefs `trace_marker` | same, Linux only |
| dtrace | USDT probes, also used by SystemTap | USDT probes via the `usdt` crate on Linux, macOS, illumos; probe provider name `qemu` and probe names equal to event names so existing .stp and .d scripts work |
| ust | LTTng-UST tracepoints | LTTng-UST through its C ABI, provider names matching QEMU's |

On top of these, the Rust `tracing` ecosystem is available as an opt-in extra: with the `tracing=on` property of the ruvm-metrics object (see observability below) every enabled event is also emitted as a `tracing` event with target `qemu::<subdir>` and structured fields named after the trace-events arguments, so any `tracing` subscriber (JSON logs, OpenTelemetry, tokio-console style tools) can consume it. The QEMU-named backends remain the default because distribution tooling and documentation assume them.

`-d` log items (`-d in_asm,op,out_asm,int,exec,cpu,mmu,guest_errors,unimp,page,nochain,plugin,strace,tid`, and `-dfilter` ranges) are part of the same logging path. ruvm keeps the item names and formats; `op` and `op_opt` print ruvm-jit IR instead of TCG ops because the IR differs by design, which is documented as an intended difference in document 02.

## Guest agent (qemu-ga)

qemu-ga runs inside the guest and talks QMP-like JSON over a virtio-serial port (`org.qemu.guest_agent.0`), an isa-serial port, or vsock. ruvm-ga is a separate small binary (canon), statically linked on Linux, with Windows builds using the same VSS provider interface as QEMU's qga/vss-win32 for filesystem freeze.

### Protocol

The protocol has no greeting and no capabilities negotiation. Because the channel is a raw serial stream that survives agent restarts and host reconnects, synchronization is explicit: the host sends `guest-sync` or `guest-sync-delimited` with a random `id`, discards input until it sees the echoed id, and only then trusts the stream. `guest-sync-delimited` prefixes the response with a 0xFF byte, and the host may send 0xFF before the request to reset the agent's JSON parser (0xFF is never valid UTF-8, so it cannot appear in JSON). ruvm-ga reproduces this exactly, including the parser reset on 0xFF. Commands can be disabled with `--block-rpcs` or restricted with `--allow-rpcs`, and the agent can be frozen: while filesystems are frozen only a fixed allow-list of commands (guest-ping, guest-sync, guest-fsfreeze-status, guest-fsfreeze-thaw and a few others as in qga/main.c) is accepted, to avoid deadlocks on frozen filesystems.

### Commands

ruvm-ga implements every command in QEMU 11.1's qga/qapi-schema.json: guest-sync-delimited, guest-sync, guest-ping, guest-get-time, guest-set-time, guest-info, guest-shutdown, guest-file-open, guest-file-close, guest-file-read, guest-file-write, guest-file-seek, guest-file-flush, guest-fsfreeze-status, guest-fsfreeze-freeze, guest-fsfreeze-freeze-list, guest-fsfreeze-thaw, guest-fstrim, guest-suspend-disk, guest-suspend-ram, guest-suspend-hybrid, guest-network-get-interfaces, guest-get-vcpus, guest-set-vcpus, guest-get-disks, guest-get-fsinfo, guest-set-user-password, guest-get-memory-blocks, guest-set-memory-blocks, guest-get-memory-block-info, guest-exec, guest-exec-status, guest-get-host-name, guest-get-users, guest-get-timezone, guest-get-osinfo, guest-get-devices, guest-ssh-get-authorized-keys, guest-ssh-add-authorized-keys, guest-ssh-remove-authorized-keys, guest-get-diskstats, guest-get-cpustats, guest-get-load, guest-network-get-route.

Per-OS availability matches QEMU (for example guest-get-devices is Windows only, guest-get-diskstats and guest-get-cpustats Linux only). The returned data must match: libvirt's `virsh domfsinfo`, `domifaddr --source agent`, and `guestinfo` parse these structures, and OpenStack uses `guest-set-user-password` and `guest-fsfreeze-*` for snapshots. On the host side, ruvm needs no special code: the agent channel is an ordinary chardev that libvirt connects to.

ruvm-ga is a clean Rust implementation, not a port, because its logic (reading /proc, calling fsfreeze ioctls, spawning processes) is plain systems code. It reuses ruvm-qapi for parsing and dispatch with the qga schema, so its JSON handling is identical to the QMP server's.

## Observability beyond QEMU

These features have no QEMU counterpart, are off by default, and never change any QEMU-defined output. They are configured through a ruvm-specific QOM object, and exposed in QMP only under the downstream-extension naming rule from qmp-spec.rst ("Downstream extension of QMP"): new names must start with `__RFQDN_`. ruvm uses `__io.github.tamnd.ruvm_` as its prefix (new decision, recorded in document 25), so for example `__io.github.tamnd.ruvm_query-metrics`. These names exist only when extensions are enabled, meaning the binary runs as the `ruvm` personality or `RUVM_EXTENSIONS=1` is set (document 02). When they exist, `query-qmp-schema` lists them, which QEMU's own rules permit. In the `qemu-*` personalities they are absent, and a management tool that does not know about ruvm never sees them.

- Metrics endpoint: `-object ruvm-metrics,id=m0,listen=unix:/run/ruvm/vm1.metrics` (or `tcp:` with TLS credentials from a `tls-creds-x509` object, the same authz model as the VNC and migration listeners) serves OpenMetrics text: per-vCPU exit counts by reason, JIT counters (blocks translated, tier-2 promotions, TLB misses, code cache occupancy), per-queue virtio counters, block node latency histograms (the same data as `block-latency-histogram-set` but always on), migration progress, and ruvm-aio reactor loop stats. Counters are per-thread and summed on scrape so the fast path does no shared writes. `query-stats` remains the QEMU-compatible way to get KVM stats; the metrics object is for fleets that already scrape Prometheus.
- Structured logs: the ruvm-metrics object's `log-format=json` property (ruvm does not add values to QEMU's own `-msg` option, since they could clash with future QEMU values) writes each error_report, warning, and enabled trace event as a JSON line with timestamp, severity, subsystem, and QOM path where known. Plain-text stderr output stays the default so libvirt's log parsing is unaffected.
- `tracing` bridge: described in the tracing section; enabled by the metrics object's `tracing=on` or `otlp-endpoint` property.

These extras follow the same rule as everything else in this document: they are additive, isolated behind names QEMU will never use, and removable without affecting compatibility.
