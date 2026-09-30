# 03. System architecture

This document describes how ruvm is put together: the crate layers, the threads that run in a ruvm process, how device state is protected without a big lock, the event loop in ruvm-aio, error handling, the path from argv to a running guest, runstates, reset and shutdown, the hot paths, and the rules for unsafe code. Document 04 covers the object model and configuration in detail, document 05 the memory core, document 06 the accelerators, documents 07 and 08 the JIT. Where this document names a QEMU function or file, it refers to the QEMU 11.1 tree that ruvm treats as its reference (see document 02 for the compatibility contract).

## Principles that shape the architecture

Four constraints drive the decisions below. The guest and the management stack must not be able to tell ruvm from QEMU (registers, QMP replies, error classes, QOM paths, runstates, event order, migration stream). Hot paths (MMIO exits, virtqueue notifications, softmmu TLB misses, interrupt injection) must not allocate and take at most one lock. Adding a device, board, block driver or guest ISA touches one crate (document 20). Unsafe code lives in a few crates with a stated reason for each block.

## Layers and crates

The workspace is split into six layers. A crate may depend only on crates in its own layer or a lower one. `cargo xtask layers` reads `[package.metadata.ruvm] layer = N` from every Cargo.toml and fails CI if an edge points upward. The same xtask checks license direction: a crate licensed MIT OR Apache-2.0 may not depend on a GPL-2.0-or-later crate (`cargo xtask provenance`, see document 24).

```
 L5 system     ruvm-cli  ruvm-system  ruvm-monitor  ruvm-gdbstub
               tools: ruvm-img ruvm-nbd ruvm-io ruvm-storage-daemon ruvm-ga
        |
 L4 machines   ruvm-machine-x86  ruvm-machine-arm  ruvm-machine-riscv ...
               ruvm-firmware (ACPI, SMBIOS, FDT, fw_cfg files, descriptors)
        |
 L3 targets    ruvm-target-{x86,arm,riscv,ppc,s390x,mips,loongarch,...}
    devices    ruvm-hw-{intc,timer,pci,usb,char,display,audio,input,net,
               storage,virtio,vfio,iommu,tpm,misc,i2c,ssi,cxl,acpi}
    backends   ruvm-block ruvm-net ruvm-chardev ruvm-ui ruvm-audio
               ruvm-migration ruvm-linux-user ruvm-bsd-user
        |
 L2 execution  ruvm-hw-core  ruvm-accel  ruvm-accel-{kvm,hvf,whpx,mshv,
               nvmm,xen,nitro,qtest}  ruvm-jit  ruvm-jit-{x86_64,aarch64,
               riscv64,ppc64,s390x,loongarch64,interp}  ruvm-softfloat
               ruvm-decode (build tool)  ruvm-plugin
        |
 L1 core       ruvm-qom  ruvm-qapi  ruvm-mem  ruvm-vmstate  ruvm-trace
        |
 L0 found.     ruvm-base  ruvm-aio  ruvm-sys
```

The table gives the job of each group and the QEMU code it replaces.

| Crate group | Responsibility | QEMU counterpart |
|---|---|---|
| ruvm-base | Error type, bit ops, intrusive lists, epoch RCU, trace macros, lock ranks | util/, include/qemu/, util/rcu.c, qapi/error.c |
| ruvm-aio | Reactors, timers, bottom halves, async file and socket IO, thread pool | util/main-loop.c, util/aio-posix.c, util/async.c, util/qemu-timer.c, util/thread-pool.c, util/fdmon-*.c |
| ruvm-sys | Raw host bindings: KVM, HVF, WHPX, vfio, iommufd, userfaultfd, memfd, guest_memfd | linux-headers/, scattered ioctl code |
| ruvm-qom | Types, classes, interfaces, properties, composition tree | qom/object.c, qom/object_interfaces.c |
| ruvm-qapi | Schema parser, codegen, visitors, QMP dispatch, introspection | scripts/qapi/, qapi/*.c, qobject/ |
| ruvm-mem | MemoryRegion, AddressSpace, FlatView, dispatch, dirty tracking, RAM backends | system/memory.c, system/physmem.c, backends/hostmem*.c |
| ruvm-vmstate | VMState descriptions and the stream codec | migration/vmstate*.c, migration/qemu-file.c |
| ruvm-hw-core | Device trait, buses, IRQ lines, clocks, resettable, fw_cfg core, lock domains | hw/core/qdev.c, hw/core/resettable.c, hw/core/irq.c, hw/core/clock.c |
| ruvm-accel* | Accel and Vcpu traits and one crate per hypervisor | accel/, target/*/kvm, target/*/hvf |
| ruvm-jit* | TCG replacement | tcg/, accel/tcg/ |
| ruvm-target-* | Guest ISA definitions | target/<arch>/ |
| ruvm-hw-* | Device models | hw/<family>/ |
| ruvm-machine-* | Boards | hw/<arch>/ board files |
| ruvm-system | Command line, machine creation, main loop, runstate | system/vl.c, system/runstate.c, system/cpus.c |
| ruvm-monitor | QMP server, HMP | monitor/ |

Device crates never depend on accelerator crates: a device that needs an irqfd, an ioeventfd or a KVM-routed MSI asks through a trait in ruvm-hw-core (`IrqRouting`, `NotifierBinding`) that the accelerator implements. ruvm-system does not name machines or devices either; it iterates the link-time registries (documents 04 and 20).

## Process and thread model

A running `qemu-system-*` process in ruvm has the same thread roles as QEMU, with one addition (a dedicated RCU reclaim thread already exists in QEMU as the `call_rcu` thread in util/rcu.c) and one change (no BQL shared between them).

```
+---------------------------------------------------------------------+
|  ruvm process                                                        |
|                                                                      |
|  main thread                     vCPU threads (1 per vCPU)           |
|  +--------------------------+    +-------------------------------+   |
|  | main reactor (ruvm-aio)  |    | accel loop: Vcpu::run()       |   |
|  |  QMP/HMP monitors        |    |  exits -> MMIO/PIO dispatch   |   |
|  |  chardev frontends       |    |  JIT: TB exec, TLB fill       |   |
|  |  realtime/host timers    |    |  run_on_cpu work queue        |   |
|  |  runstate machine        |    +-------------------------------+   |
|  |  default block/net IO    |                                        |
|  +--------------------------+    iothreads (-object iothread)       |
|                                  +-------------------------------+   |
|  monitor iothread (OOB)          | reactor per thread            |   |
|  +--------------------------+    |  virtqueue handlers           |   |
|  | reactor, QMP parsing,    |    |  block and net IO             |   |
|  | allow-oob commands       |    |  ioeventfd / irqfd            |   |
|  +--------------------------+    +-------------------------------+   |
|                                                                      |
|  helper threads: RCU reclaim, migration, thread pool workers,        |
|  UI thread on macOS (Cocoa needs the process main thread)            |
+---------------------------------------------------------------------+
```

The main thread runs the main reactor. It owns monitor sockets, chardev frontends that are not assigned to an iothread, timers on the realtime and host clocks, the runstate machine, and the default block and network IO when no iothread is configured. This is the same set of work that `main_loop_wait()` in util/main-loop.c drives in QEMU. On macOS with the Cocoa UI, AppKit must own the process's first thread, so, as QEMU does with its `qemu_main` hook in ui/cocoa.m, ruvm starts the main reactor on a second thread and gives the first thread to the UI.

Each vCPU has one thread. Under KVM, HVF, WHPX, MSHV and NVMM it loops on `Vcpu::run()`. Under the JIT it executes translated blocks. MTTCG is always on (canon), so there is no round-robin mode with one thread for all vCPUs, except in serialized mode, described below, which is used for icount and record/replay.

Iothreads are created by `-object iothread,id=...` exactly as in QEMU, with the same properties (`poll-max-ns`, `poll-grow`, `poll-shrink` and `poll-weight` from iothread.c, plus `aio-max-batch`, `thread-pool-min` and `thread-pool-max` inherited from event-loop-base.c). Each runs one ruvm-aio reactor. Devices that accept an `iothread` or `iothread-vq-mapping` property run their virtqueue and IO handlers there.

A monitor with `"allow-oob"` negotiated runs its parser on a dedicated monitor iothread, matching `mon_iothread` in monitor/monitor.c, so that out-of-band commands such as `migrate-recover` and `migrate-pause` are handled while the main thread is blocked.

### No shared big lock

QEMU serializes almost everything that touches device state with the BQL (`bql_lock()` and `bql_unlock()` in include/qemu/main-loop.h, previously named `qemu_mutex_lock_iothread`). vCPU threads take it on every MMIO exit that lands on a normal region. The main loop drops it only around the blocking poll, see `os_host_main_loop_wait()`. QEMU has been moving away from this for years: KVM vCPUs touch guest RAM without the lock, iothreads run virtio-blk and virtio-scsi without it, and `memory_region_enable_lockless_io()` lets a region opt out entirely (in the reference tree the HPET in hw/timer/hpet.c and the ACPI code in hw/acpi/core.c use it). The QEMU Rust devices wrap their state in `BqlRefCell`, which is a runtime check that the BQL is held, so even the new Rust code in QEMU is BQL-bound. ruvm finishes the move: there is no BQL at all.

## Lock domains and the control lock

### What the BQL gives the guest

Before replacing a lock one has to write down what it guarantees. From the guest's and the management stack's point of view the BQL provides these properties.

1. G1, access atomicity. A single MMIO or PIO access to a device runs to completion before any other access, timer callback, bottom half or backend callback touches that device's state.
2. G2, causal visibility. If a vCPU's access to device A returns, and the guest later (in its own memory order) accesses device B, B's handler sees every state change that A's handler made to shared state.
3. G3, synchronous side effects. When an access has a side effect on another device through a direct call (raising an interrupt line, writing a PCI config register through the host bridge, changing the memory map), that side effect is complete when the vCPU's access returns. A guest that writes an interrupt-clear register and then reads the interrupt controller's pending register depends on this.
4. G4, callback atomicity. Timers, bottom halves and chardev or network callbacks for a device are atomic with respect to MMIO on that device.
5. G5, stable topology. While a QMP command that changes the composition tree runs, no guest access can observe a half-plugged or half-unplugged device.
6. G6, quiescence. `vm_stop()` followed by device state save sees a consistent snapshot of all devices, and `qemu_system_reset()` runs the three-phase reset of the whole tree with nothing else moving.

The BQL also gives a global total order over all device operations. That order is not guest-visible, and the argument for that is the core of this section.

### Domains

A lock domain is a mutex plus a rank plus the set of devices whose state it protects. Every device instance belongs to exactly one domain. Every entry point into a device (MMIO and PIO handlers, timer callbacks, bottom halves, backend callbacks, QOM property accessors after realize, reset phases, vmstate save and load) runs with that domain's lock held. The lock is taken by the dispatch code in ruvm-hw-core and ruvm-mem, never by the device, so a device author cannot forget it.

```rust
pub struct LockDomain {
    lock: RawDomainMutex,      // parking_lot-style word lock, adaptive spin
    rank: DomainRank,          // u32, see ranking below
    id: DomainId,              // stable for the life of the domain
    deferred: DeferQueue,      // work posted by holders of other domains
    home: ReactorHandle,       // reactor that runs this domain's BHs and timers
}

pub struct DomainCell<T> {
    domain: Arc<LockDomain>,
    value: UnsafeCell<T>,
}

impl<T> DomainCell<T> {
    /// Only dispatch code calls this. Device code receives the guard.
    pub(crate) fn lock(&self) -> DomainGuard<'_, T>;
}
```

A device's mutable state lives in a `DomainCell<State>`. The `MmioOps` methods in the canon take `&self`; the device struct holds the `DomainCell`, and the dispatcher passes a `DomainGuard` in through the `AccessCtx`, so `ctx.state()` returns `&mut State` with the borrow checker tying the reference to the guard's lifetime. Devices that want a lock-free fast path for a particular register (the virtio notify register is the canonical case) put the fields that register touches in atomics outside the `DomainCell` and mark the subregion `lockless`, which is the same opt-in as `memory_region_enable_lockless_io()`.

Domain assignment is computed, not hand-written. Each device declares its synchronous peers: a PCI device has its host bridge as a peer because config space accesses go host bridge to device; the i8259 slave has the master as a peer because cascade is a direct call; the PC speaker has the PIT as a peer because `pcspk` reads PIT channel 2 state directly. Declarations come from bus relationships (automatic) and from `#[sync_peer]` on link properties (document 04). At `PHASE_MACHINE_READY` ruvm-system builds a directed graph with an edge A to B for each "A calls into B synchronously", computes strongly connected components, and makes each component one domain. The condensed graph is acyclic, and ranks are assigned by topological order, so that every synchronous call goes from a lower rank to a higher rank. Interrupt lines are not edges (they are lock-free, below), which keeps the graph sparse: in a q35 machine with a dozen devices most devices end up alone in their own domain.

Rank bands make the order readable in lockdep output and leave room for hotplug.

| Band | Ranks | Holders |
|---|---|---|
| Control | 0 | the control lock |
| Platform | 100 to 999 | PCI host bridges and config space, chipset (ICH9 LPC, PIIX), fw_cfg, machine-level registers |
| Device | 1000 to 8999 | ordinary devices, assigned in topological then creation order |
| Interrupt | 9000 to 9499 | IOAPIC, i8259 pair, GIC distributor and redistributors, PLIC, APLIC, other irqchips |
| Memory map | 9500 | the memory topology lock in ruvm-mem (document 05) |
| Leaf | 9600 and up | vCPU work queues, reactor internals, allocator-free queues; never call out |

A hotplugged device is placed in a new domain in the device band with a rank above its bus's domain. It only adds edges from existing domains into the new one, so it cannot create a cycle. If a hotplugged device declares a synchronous edge into an existing domain of lower rank (rare, and only for boards that wire devices together at plug time), it is merged into that domain instead.

### The control lock

The control lock is a single mutex that serializes topology and lifecycle changes: `device_add`, `device_del` completion, `object-add` and `object-del`, `blockdev-add` and its relatives when they touch the graph, `system_reset`, `stop` and `cont`, migration start and completion, snapshot save and load, and machine creation itself. It has rank 0, so it must be taken before any domain. vCPU threads never take it. QMP commands that only read state (`query-*`, `qom-get`, `qom-list`) do not take it: they read through RCU-protected views or take the specific domain whose state they read.

Topology changes that must be visible to guest accesses are published through RCU. Unplugging a device is: take the control lock, unmap its regions (new FlatView, published with an RCU swap), wait for a grace period so no vCPU is still inside the old FlatView, run unrealize under the device's domain lock, drop the domain. This is `memory_region_del_subregion()` followed by `object_unparent()`, with the grace period doing the fencing the BQL did.

### Why QEMU-visible ordering is preserved

Take each guarantee in turn.

G1 and G4 hold because every entry point into a device runs under that device's domain lock. A single mutex gives linearizability of the operations it protects, which is exactly the "one at a time" property the BQL gave, restricted to the operations that can touch this device's state.

G3 holds because of how domains are built. A synchronous side effect in QEMU is a direct C function call from one device into another. In ruvm, such a call is an edge in the domain graph. Either both ends ended up in the same domain (if there is a cycle), or the callee has a higher rank than the caller, in which case the caller acquires the callee's domain while still holding its own and makes the call directly. Either way the effect is complete before the vCPU's access returns, as in QEMU. Interrupt lines are handled separately below and also satisfy G3.

G2 holds because lock release and acquire form a happens-before edge, and because a vCPU performs its MMIO accesses synchronously in program order. Suppose a vCPU writes device A, then (after its access returns) accesses device B. If A's handler touched B's state, it did so while holding B's domain (G3), and released it before returning. The vCPU's later access to B acquires the same lock, so it sees the change. If a different vCPU accesses B, the guest must have ordered the two vCPUs through guest memory (a store-release and load-acquire pair, or a stronger barrier), since there is no other channel between vCPUs. Guest RAM in QEMU is not under the BQL either (KVM and MTTCG vCPUs access it concurrently), so there too the guest can only rely on its own memory synchronization, and ruvm preserves happens-before along that chain because every link in it is a synchronizing operation.

What ruvm drops is the total order between operations on devices in different domains that have no synchronous connection. Two such operations commute: neither reads state the other writes. The guest observes device state only by accessing the device, which goes through the paths above. The one other channel is DMA into guest RAM, and concurrent DMA from two devices already races in QEMU whenever they run in different iothreads (virtio-blk and virtio-scsi dataplane), so ruvm adds no new race.

G5 holds because topology changes take the control lock and publish through RCU with a grace period, so a guest access either sees the old topology entirely or the new one entirely.

G6 holds because `vm_stop()` in ruvm does the same thing as in QEMU: pause every vCPU (they exit their run loop and park, releasing any domain they held), drain iothreads (`bdrv_drain_all_begin()` equivalent, plus a quiesce on every reactor that runs device handlers), and only then walk devices. With no thread able to enter a device, taking each domain in turn gives the same snapshot the BQL gave.

Three places in QEMU rely on the BQL for something stronger. The first is `run_on_cpu()`, which in QEMU drops the BQL while waiting (`do_run_on_cpu()` in cpu-common.c waits on a condition variable with the BQL as its mutex). In ruvm, synchronous `run_on_cpu` may only be called while holding no domain; debug builds assert this, and device code uses `async_run_on_cpu`. The second is `vm_stop` requested from a device (for example `werror=stop` on an IO error). Like QEMU (`qemu_system_vmstop_request()`), ruvm turns it into a main loop request, so a device never waits for vCPUs while holding a domain. The third is the MMIO re-entrancy guard (`mem_reentrancy_guard` on `DeviceState`), which rejects a device DMA-ing into its own MMIO. In ruvm the lock's owner field detects re-entry into a held domain and returns `MemTxResult::AccessError` with the same guest-error log line.

### Interrupt lines without a second lock

Interrupt lines are the most frequent cross-device call, and routing them through the rank order would force a second lock on every interrupt. Instead, an `IrqLine` is an atomic level word in the receiving interrupt controller plus a flat combining lock on the controller's domain.

```rust
pub struct IrqLine {
    sink: &'static IrqSinkShared,  // lives as long as the machine
    pin: u32,
}

impl IrqLine {
    pub fn set(&self, level: bool) {
        // 1. publish the input level with release ordering
        if !self.sink.inputs.set(self.pin, level) { return; } // no edge
        // 2. try to run the controller update ourselves
        match self.sink.domain.try_lock() {
            Some(guard) => self.sink.update(guard),       // processes all dirty pins
            None => self.sink.dirty.store(true, Release), // current holder will run it
        }
    }
}
```

The controller's unlock path checks `dirty` and reruns the update before releasing. That preserves G3 in the form the guest can observe: after the device's access returns, the input bit is published, and any later access to the interrupt controller acquires its lock after the holder has processed the bit. Delivery to a vCPU (setting `interrupt_request` bits and kicking the thread) happens inside the update. With an in-kernel irqchip under KVM, `IrqLine::set` becomes an irqfd write or `KVM_IRQ_LINE`, with no user-space lock at all.

### Deadlock avoidance

The rule is simple: a thread may acquire a domain only if its rank is strictly greater than every rank it already holds. The control lock is rank 0. Debug builds keep a small per-thread stack of held ranks (eight entries is enough; the deepest chain we know of is PCI config, device, interrupt controller, memory map) and panic with both lock names on a violation, which is the ruvm equivalent of Linux lockdep. CI runs the full qtest suite with this checking enabled. The lock primitives themselves are model-checked with loom.

Waiting operations have their own rule. A thread holding any domain may not block on something another thread must do while holding a domain: synchronous `run_on_cpu`, waiting for a vCPU to pause, waiting for an iothread to quiesce, or a synchronous QMP round trip. These are all turned into posted requests. The lock order plus this rule is sufficient: a cycle in the wait-for graph would need either a lock edge against rank order or a wait edge from a lock holder.

### Devices that need more than one domain

Most devices only ever hold their own domain. The cases that need more are listed here so reviewers can recognize them.

1. PCI config space. The host bridge's `config_data` handler (platform band) locates the target function and calls its config read or write while holding the host bridge's domain. The target device is in the device band, so this is a legal upward-rank acquisition.
2. MSI and MSI-X. An MSI is a DMA write to the interrupt controller's address window. It takes the interrupt band domain (or goes to KVM as `KVM_SIGNAL_MSI` or an irqfd), which is always above the device.
3. Memory map changes triggered by a register write: PCI BAR programming, PAM and SMRAM registers on the i440FX and q35 host bridges, x86 A20. The device holds its domain and takes the memory map lock (rank 9500) to commit the new topology. This is not a hot path.
4. Peer-to-peer DMA that lands on another device's MMIO region. The DMA helper resolves the target through the FlatView. If the target domain's rank is higher, it is locked directly. If lower, the helper first tries `try_lock`. If that fails, the helper uses relock: it releases the source domain, locks both in rank order, and returns. Because this can release the source's lock, the DMA functions that can hit MMIO (`dma_memory_read`, `dma_memory_write`, `pci_dma_read`, `pci_dma_write`) take the `DomainGuard` by `&mut` and invalidate outstanding borrows of device state. The borrow checker then forces device code to re-read state after the call. Posted writes (the common case for p2p) use a per-source FIFO on the target's `DeferQueue` instead of relock, which is permitted by PCIe ordering rules for posted requests and preserves per-source order.
5. Migration, snapshot and reset. These run with the control lock held and vCPUs paused, and take one domain at a time. Three-phase reset (enter across the whole tree, then hold, then exit, see resettable.c) does not need more than one domain at once because nothing else is running.
6. Boards that model shared buses as one piece of hardware (an I2C controller calling into its slaves on every transfer, an SSI controller and its flash). Here the domain graph produces either one merged domain or a strictly ordered pair, and no special code is needed.

### Serialized mode

Record/replay (`-icount ...,rr=record|replay`) and icount without `sleep=off` require a deterministic global order of device events, which is exactly what the BQL provided. ruvm has a serialized mode in which the domain assignment function maps every device to a single domain and all vCPUs run on one thread in round-robin, as QEMU's single-threaded TCG does. qtest (`-accel qtest`) also uses serialized mode, since qtest scripts assume that an `outl` completes all side effects before the next command. The cost is only paid when those features are on.

## Event loop: ruvm-aio

ruvm-aio provides the equivalent of QEMU's `AioContext` (util/async.c, util/aio-posix.c, util/aio-win32.c), the main loop (util/main-loop.c), timer lists (util/qemu-timer.c), and the thread pool (util/thread-pool.c). It does not use tokio or any general-purpose runtime in the data path: those runtimes are work-stealing and multi-threaded by default, which is wrong for code that must run on a specific iothread chosen by the user, and they add allocation per task.

### Reactor

A `Reactor` is owned by one thread and is `!Send`. Other threads interact with it through a `ReactorHandle`, which is `Send + Sync` and can schedule bottom halves, post closures, and wake the reactor. The reactor's loop is the equivalent of `aio_poll()`:

```
loop {
    run expired timers for each clock this reactor owns
    run scheduled bottom halves (in scheduling order)
    if adaptive polling is enabled and poll_ns > 0:
        spin on registered poll handlers (virtqueue avail idx, NVMe SQ tail)
        for up to poll_ns; if one fires, adjust poll_ns and restart
    compute timeout = min(next timer deadline, 0 if BHs pending)
    submit pending IO and wait for completions with that timeout
    dispatch completions to their handlers
    adjust poll_ns using poll-grow, poll-shrink, poll-weight
}
```

The adaptive polling algorithm is copied from `aio_poll()` and the per-handler `poll.ns` logic in util/aio-posix.c, because users tune the `iothread` polling properties and expect the same response.

### Host backends

| Host | Backend | Wakeup | Notes |
|---|---|---|---|
| Linux | io_uring with `IORING_SETUP_SINGLE_ISSUER` and `IORING_SETUP_DEFER_TASKRUN` when available | eventfd registered as a polled fd, or a `MSG_RING` from another ring | Fallback to epoll when io_uring is missing or disabled by seccomp or sysctl, mirroring QEMU's util/fdmon-io_uring.c, fdmon-epoll.c and fdmon-poll.c |
| macOS, FreeBSD, NetBSD, OpenBSD | kqueue | `EVFILT_USER` | File IO goes to the thread pool (no completion-based file IO on these hosts) |
| Windows | IOCP | `PostQueuedCompletionStatus` | Sockets through AFD polling, files through overlapped IO |

The public API is completion-based: a caller submits an operation (read, write, readv, writev, fsync, poll-add, accept, connect, timeout) with a buffer it owns until completion, and gets a completion callback or a future. On kqueue and epoll this is emulated by readiness plus a nonblocking syscall. On Linux the block layer's hot path (document 14) submits reads and writes directly to io_uring with registered buffers and fixed files. Since the reactor is single-issuer, submission needs no lock.

### Timers

Timers are per reactor and per clock. The four QEMU clock types are kept with their exact semantics because device models depend on them: `Realtime` (monotonic host time, runs when the VM is stopped), `Virtual` (guest time, stops when the VM is stopped, driven by icount if enabled), `Host` (wall clock, may jump), and `VirtualRt` (realtime used for icount warp, equal to `Virtual` outside icount). Each clock on each reactor keeps a binary heap of deadlines rather than QEMU's sorted linked list; insertion is O(log n) instead of O(n), which matters for a busy virtio-net device with per-queue coalescing timers. Timer handles are preallocated in the device's state and rearming does not allocate.

The virtual clock is shared across reactors. When it is stopped (`vm_stop`) or its offset changes (migration, `-rtc clock=vm`), every reactor that owns virtual timers is woken to recompute its deadline, as `qemu_clock_notify()` does.

### Bottom halves

A bottom half is a preallocated callback with an atomic flag word, like `QEMUBH` in util/async.c with its `BH_PENDING`, `BH_SCHEDULED`, `BH_ONESHOT`, `BH_IDLE` and `BH_DELETED` flags. Scheduling a BH from any thread pushes it onto the reactor's lock-free intrusive list and wakes the reactor if the flag transitioned from idle. One-shot BHs (`aio_bh_schedule_oneshot` equivalents) use a small per-reactor slab so they also avoid the general allocator. BHs run with their device's domain lock taken by the reactor, which is how G4 is enforced for them.

### The AioContext equivalent and async code

QEMU's block layer uses coroutines (`qemu_coroutine_create`, `qemu_co_*` primitives) running inside an `AioContext`. ruvm uses Rust futures on a per-reactor local executor instead. The executor is deliberately small: tasks are `!Send`, spawned into a slab owned by the reactor, woken by the reactor's own wakers, and never migrate threads. Moving a block node to another iothread (`x-blockdev-set-iothread`, or a device with an `iothread` property being realized) drains the node, detaches its tasks, and re-spawns them on the new reactor, which matches the semantics of `bdrv_try_change_aio_context()`.

`AioContext` in QEMU is also a GLib `GSource`, which is how it composes with GLib-based UIs and chardevs. ruvm's main reactor can own a `GMainContext` when the GTK or D-Bus UI is linked in: the reactor adds the GLib context's fds to its poll set and runs `g_main_context_dispatch()` when they fire. Builds without a GLib-based UI do not link GLib at all.

## Error handling

QEMU has one error type, `Error`, carrying a class, a message, an optional hint, and the source location. The class shows up on the wire in QMP error replies as `{"error": {"class": "...", "desc": "..."}}`. The classes are defined by `QapiErrorClass` in qapi/error.json: `GenericError`, `CommandNotFound`, `DeviceNotActive`, `DeviceNotFound`, `KVMMissingCap`. Almost all errors are `GenericError`, and management tools match on the `desc` text more often than anyone would like. So the Rust type is shaped to make QEMU's messages easy to reproduce, not to be idiomatic for its own sake.

```rust
#[derive(Debug)]
pub struct Error {
    class: ErrorClass,               // generated from qapi/error.json
    msg: Box<str>,
    hint: Option<Box<str>>,          // error_append_hint()
    src: &'static Location<'static>, // #[track_caller]
    cause: Option<Box<dyn std::error::Error + Send + Sync>>,
}

pub type Result<T, E = Error> = core::result::Result<T, E>;

#[macro_export]
macro_rules! err { ($($t:tt)*) => { $crate::Error::generic(format!($($t)*)) } }
```

The conventions:

1. Message strings that QEMU test suites match (iotests reference output, qtest assertions, libvirt's error parsing) are copied verbatim, including capitalization and quotes, for example `Invalid parameter type for 'x', expected: integer` from qapi/qobject-input-visitor.c. A CI job greps the iotests reference files for `qemu-img:` and `qemu-system-` prefixed lines and compares against ruvm's output.
2. `error_prepend()` and `error_append_hint()` have direct equivalents (`Error::prepend`, `Error::hint`). Hints are printed on the command line and dropped from QMP replies, as in QEMU.
3. `&error_fatal`, `&error_abort` and `&error_warn` become methods on `Result`: `.or_fatal()` prints and exits with status 1, `.or_abort()` panics, `.or_warn()` reports and continues. `.or_fatal()` uses the current `CliLocation` (the equivalent of `loc_push_restore()`), so an error while applying `-device` prints `qemu-system-x86_64: -device virtio-blk-pci,drive=nope: ...` in QEMU's format.
4. Guest errors are not `Error`. A guest writing a read-only register is logged under `-d guest_errors` (`LOG_GUEST_ERROR`) and otherwise ignored, exactly as QEMU does, and unimplemented features go under `-d unimp`.
5. Memory transactions return `MemResult<T> = Result<T, MemTxError>`, where `MemTxError` is a `Copy` bitflags value matching `MEMTX_ERROR`, `MEMTX_DECODE_ERROR` and `MEMTX_ACCESS_ERROR`. It never allocates.
6. Panics are bugs. Release builds use `panic = "abort"`, which matches `abort()` on a failed `assert` in QEMU, avoids unwinding across the plugin and module C ABIs, and avoids poisoned locks. `hw_error()` becomes `ruvm_base::fatal!`, which prints CPU state like QEMU's `hw_error` and aborts.

## Lifecycle from argv to a running VM

The start-up sequence mirrors `qemu_init()` in system/vl.c step by step, because the order of object creation is visible through QOM paths (`/machine/unattached/device[N]` numbering), through which command-line options can refer to which objects, and through error messages.

```
argv[0] dispatch (ruvm-cli)
  |  "qemu-system-aarch64" -> system emulator, target = aarch64
  v
parse options (qemu-options table, same names, same arg kinds)
  |  -device {...} kept as JSON, -device a,b=c kept as QemuOpts
  v
validate, process sugar (-m, -smp, -boot, -accel shortcuts), early options,
help options (-device help, -machine help), daemonize
  v
trace init, main reactor init, clocks, register -global user props,
replay config, -rtc
  v
create machine object (PHASE_NO_MACHINE -> PHASE_MACHINE_CREATED)
  |  compat props for the versioned machine type are registered here
  v
early backends: chardevs, early -object (secrets, memory backends that
the machine needs, etc.), displays chosen, default devices decided
  v
apply machine options; configure accelerators
  |  PHASE_ACCEL_CREATED
  v
late backends: netdevs, remaining -object, audiodev, blockdev
  |  PHASE_LATE_BACKENDS_CREATED
  v
migration object, cpu type (-cpu), memdev, NUMA
  |
  +-- if -preconfig: run main loop until QMP x-exit-preconfig
  v
qemu_init_board: machine init (CPUs, RAM, onboard devices)
  |  PHASE_MACHINE_INITIALIZED
  v
create CLI devices: fw_cfg entries, USB, then every QemuOpts -device,
then every JSON -device
  v
machine creation done: machine_done notifiers, fw_cfg and ACPI finalize,
lock domain graph built, ranks assigned
  |  PHASE_MACHINE_READY
  v
-loadvm, or -incoming, or autostart (cont) -> RUNNING
  v
displays init, accelerator post-setup, main loop
```

QEMU creates all QemuOpts-style `-device` options before all JSON-style ones, regardless of their order on the command line (see `qemu_create_cli_devices()`), so mixing the two forms changes device numbering and PCI slot auto-assignment. ruvm reproduces this. The `phase_check()` values from include/hw/core/qdev.h (`PHASE_NO_MACHINE`, `PHASE_MACHINE_CREATED`, `PHASE_ACCEL_CREATED`, `PHASE_LATE_BACKENDS_CREATED`, `PHASE_MACHINE_INITIALIZED`, `PHASE_MACHINE_READY`) are exposed as `ruvm_hw_core::Phase` and gate the same things: `device_add` before `PHASE_MACHINE_READY` is only allowed through `-device` or in preconfig for types that allow it, and QMP commands marked `allow-preconfig` in the schema are the only ones accepted before `x-exit-preconfig`.

## Runstates

The runstate enum is generated from `RunState` in qapi/run-state.json: `debug`, `inmigrate`, `internal-error`, `io-error`, `paused`, `postmigrate`, `prelaunch`, `finish-migrate`, `restore-vm`, `running`, `save-vm`, `shutdown`, `suspended`, `watchdog`, `guest-panicked`, `colo`. Allowed transitions are the table `runstate_transitions_def[]` in system/runstate.c, copied as a static array. `runstate_set()` to a transition not in the table aborts in QEMU (it is a programming error), and ruvm does the same, because a silent transition would produce a `STOP` or `RESUME` event sequence libvirt has never seen.

```
            -S or -preconfig
 start --> prelaunch --cont--> running <--cont-- paused <--stop-- running
              |                 |   |                               ^
              | -incoming       |   +--guest shutdown--> shutdown   |
              v                 |   +--io error--------> io-error --+ (cont)
          inmigrate --done--> running / paused (per autostart)
                                |   +--migrate--> finish-migrate --> postmigrate
                                |   +--savevm---> save-vm --------> running
                                |   +--panic----> guest-panicked
                                +------suspend--> suspended --wakeup--> running
```

The diagram is a reading aid; the table in system/runstate.c is the source of truth, and a unit test enumerates every pair and checks ruvm's table against a copy extracted from the QEMU tree.

Runstate changes happen only on the main thread with the control lock held. `vm_stop(state)` pauses vCPUs, drains IO, flushes block devices (`bdrv_flush_all()`), sets the runstate, and emits the `STOP` event; `vm_start()` does the reverse and emits `RESUME`. The ordering of events relative to command replies on the QMP socket follows QEMU: for `stop`, the `STOP` event is emitted before the `{"return": {}}`, because QEMU's `qmp_stop()` calls `vm_stop()` synchronously.

## Shutdown and reset

Guest-initiated and host-initiated requests are posted to the main thread as flags, as in QEMU (`qemu_system_shutdown_request()`, `qemu_system_reset_request()`, `qemu_system_powerdown_request()`, `qemu_system_wakeup_request()`, `qemu_system_vmstop_request()`, `qemu_system_debug_request()`). A vCPU that triggers a reset (a write to the x86 reset control register 0xcf9, a PSCI `SYSTEM_RESET` call, a watchdog expiry) only sets the flag with a `ShutdownCause` and kicks the main reactor. The main reactor then runs the equivalent of `main_loop_should_exit()` in system/runstate.c, in the same order: debug request, suspend request, shutdown request, reset request, wakeup request, powerdown request, vmstop request.

System reset is:

1. Take the control lock. `pause_all_vcpus()`.
2. `cpu_synchronize_all_states()`: each accelerator pulls vCPU state into ruvm.
3. Choose `ResetType`: `SnapshotLoad` for `SHUTDOWN_CAUSE_SNAPSHOT_LOAD`, `Cold` otherwise. The full set from include/hw/core/resettable.h is `Cold`, `SnapshotLoad`, `Wakeup`, `S390CpuInitial`, `S390CpuNormal`.
4. For confidential guests whose vCPUs cannot be reset in place, call the accelerator's `rebuild_guest` hook (new VM file descriptor), as QEMU 11 does for SEV-SNP and TDX (document 19).
5. Call `Machine::reset(type)` if the board overrides it, otherwise run three-phase reset over the root reset container: `enter` on every object in the tree, then `hold` on every object, then `exit` on every object. The traversal order, including the handling of reset counts for objects reset by more than one parent, follows `resettable_assert_reset()` and `resettable_release_reset()` in hw/core/resettable.c.
6. Emit the `RESET` event unless the cause is none, subsystem reset, or snapshot load.
7. `cpu_synchronize_all_post_reset()`, clear the suspended flag, `resume_all_vcpus()`. If the VM was not running, move to `prelaunch`.

Shutdown depends on `-action shutdown=poweroff|pause` and `-action panic=...`. For `poweroff`, `main_loop_should_exit()` returns true, the main loop exits, and ruvm-system tears down in QEMU's order: stop vCPUs, run exit notifiers, flush and close block devices (`bdrv_close_all()`), close chardevs, flush trace output, and exit with the configured status (1 for `panic=exit-failure`). With `-no-shutdown` (shutdown=pause) the VM goes to `shutdown` and stays alive for QMP.

Teardown order matters for correctness in two places. Block devices must be flushed before the process exits even on SIGTERM, which QEMU handles by turning signals into a shutdown request with `SHUTDOWN_CAUSE_HOST_SIGNAL`. And vhost-user backends expect `GET_VRING_BASE` on stop so they can save ring state, which happens in the device's stop path before the socket closes (document 13).

## Hot paths

The requirement for every hot path is: no heap allocation, no lock other than the target device's domain (and none at all where the device opted into lockless access), and no reference count traffic on shared objects. RCU readers are free: ruvm-base's epoch scheme gives each vCPU and reactor thread a per-thread epoch counter that it bumps on entry and exit of a read-side section, with no atomic read-modify-write on shared cache lines.

### MMIO exit under KVM

```
vCPU thread
  ioctl(KVM_RUN) returns, exit_reason = KVM_EXIT_MMIO
  ruvm-accel-kvm: decode kvm_run.mmio {phys_addr, data, len, is_write}
  ruvm-mem: rcu read section begins (thread-local epoch, no RMW)
     FlatView lookup: per-AddressSpace sorted boundary array, cached
     last-hit section per vCPU checked first
     -> MemoryRegionSection { region, offset_within_region }
  split/combine access per region's AccessConstraints (valid/impl sizes)
  region is lockless?  yes -> call MmioOps directly
                       no  -> lock the region owner's domain (the one lock)
  MmioOps::write(ctx, offset, size, value)
     device updates state; may call IrqLine::set (atomic + try_lock,
     or irqfd write with in-kernel irqchip)
  unlock domain; rcu read section ends
  back to KVM_RUN
```

The `AccessCtx` is built on the vCPU's stack. Access size handling (`access_with_adjusted_size()` in system/memory.c) is done with fixed-size arrays. The per-vCPU "last section" cache avoids the binary search for devices that are hit repeatedly (a UART being polled, a timer's counter register).

### virtio notify

With KVM and ioeventfd (the default for virtio-pci and virtio-mmio), the guest's write to the notify address never exits to user space as an MMIO exit. KVM signals an eventfd that is registered with the iothread's reactor (as an io_uring poll or a multishot poll). The reactor dispatches the queue's handler, which runs under the virtio device's domain, pops descriptors from the vring in guest RAM through bounded volatile accessors, and submits IO. Completion is signaled back with an irqfd write. No lock beyond the device domain, no allocation (request structures are preallocated per queue depth).

When ioeventfd is unavailable (TCG, HVF today, or `ioeventfd=off`), the notify register is a lockless subregion: the handler sets a per-queue atomic "kicked" bit and wakes the owning reactor through its `ReactorHandle`. The vCPU never touches the device domain. With adaptive polling the reactor often sees the new avail index before the kick arrives.

### Softmmu TLB miss under the JIT

```
vCPU thread (JIT code)
  inline fast path: TLB entry compare misses
  call helper: tlb_fill(vaddr, access, mmu_idx)
    GuestArch::tlb_fill -> page table walk reading guest RAM
      (loads through the same TLB-less physical accessors, RCU FlatView)
    resolve physical address -> MemoryRegionSection
    RAM: fill entry {vaddr page, host addend, flags}
    MMIO: fill entry with IO flag and section index
  retry the access
```

The TLB is per-vCPU and only its own thread writes it. Cross-vCPU flushes (`tlb_flush_page_all_cpus_synced()` equivalents) are posted as async work plus a flag the target checks at the next TB boundary, as in accel/tcg/cputlb.c. The page walk can fault; the fault is raised through the precise exception side tables (document 08). If a walk touches an MMIO page (rare, some embedded boards), the access goes through the MMIO path and takes that device's domain, which is still one lock.

### Interrupt injection

Under KVM with an in-kernel irqchip, a device raising an interrupt is an irqfd `write(2)` of 8 bytes (or a `KVM_SIGNAL_MSI` ioctl for MSI without an irqfd route). Under HVF, WHPX and the JIT the irqchip is in user space: `IrqLine::set` publishes the input level atomically, the controller update runs under the interrupt band domain (via the combining scheme above, so the device thread never waits for it), and delivery sets bits in the target vCPU's `interrupt_request` atomic, sets the vCPU's exit flag (the equivalent of writing `icount_decr.u16.high = -1` in QEMU so that translated code leaves at the next TB entry), and wakes the vCPU if it is halted. No allocation, and the device thread takes no lock it did not already hold.

## Unsafe code policy

Unsafe code is allowed only in crates that exist to wrap something unsafe. Every other crate has `#![forbid(unsafe_code)]`, which `cargo xtask layers` also checks.

| Crate | Why unsafe is needed |
|---|---|
| ruvm-sys | ioctls, mmap, raw fds, OS APIs |
| ruvm-base | epoch RCU, intrusive lists, `DomainCell` |
| ruvm-aio | io_uring and IOCP buffer ownership, kqueue |
| ruvm-mem | guest memory mapping and the volatile accessors |
| ruvm-jit and ruvm-jit-* | executable memory, calling generated code, W^X toggling (MAP_JIT and `pthread_jit_write_protect_np` on macOS) |
| ruvm-accel-* | shared `kvm_run` page, hypervisor frameworks |
| ruvm-plugin, module loader in ruvm-system | C ABI |
| ruvm-hw-vfio | mapping device BARs, DMA mapping |
| ruvm-linux-user, ruvm-bsd-user | raw syscalls on behalf of the guest |

Device crates, machine crates, target crates (except for helpers the JIT calls, which live in a small `unsafe` module per target, reviewed like JIT code), the block layer, networking, chardevs, QOM, QAPI, and the monitor contain no unsafe code.

The rules for crates that do contain it:

1. Every `unsafe` block has a `// SAFETY:` comment stating the invariant it relies on; `clippy::undocumented_unsafe_blocks` and `clippy::multiple_unsafe_ops_per_block` are denied.
2. Guest memory is never exposed as `&[u8]` or `&mut [u8]`. The guest (or another vCPU, or a device doing DMA) can change it at any time, which makes a Rust reference to it undefined behavior. Access goes through `GuestSlice` and `GuestPtr<T>` types that use volatile or relaxed atomic copies, the same model as rust-vmm's `vm-memory` `VolatileSlice`. Descriptor rings are read through these types too.
3. Unsafe public functions document their preconditions in a `# Safety` section; CI tracks their count per crate, and an increase needs a ruvm-base owner's review.
4. Miri runs on ruvm-base, the ruvm-mem core, and ruvm-vmstate in CI. Loom runs on the RCU, `DomainCell`, `IrqLine`, BH list and reactor wakeup code.
5. Fuzzing (cargo-fuzz with libFuzzer) covers every parser that takes untrusted input: the QMP JSON parser, the migration stream decoder, qcow2 and other image format metadata, virtio descriptor chains, and vhost-user and vfio-user message decoding (document 22).

## Decisions made in this document

These are decisions not already in the canon; they are listed for document 25.

1. Lock domains are computed at `PHASE_MACHINE_READY` from declared synchronous peer edges by strongly connected components, and ranks come from topological order within fixed bands (platform, device, interrupt, memory map, leaf).
2. Interrupt lines use an atomic input word plus a flat combining lock on the controller's domain, so raising an interrupt never blocks and never takes a second lock.
3. Peer-to-peer DMA to a lower-ranked domain uses try-lock, then relock with borrow invalidation for reads, and a per-source posted FIFO for writes.
4. Serialized mode (all devices in one domain, one vCPU thread) is used for icount, record/replay and qtest.
5. Timers use a per-reactor, per-clock binary heap instead of a sorted list.
6. The main reactor hosts a GLib main context only when a GLib-based UI or chardev is linked.
7. Release builds use `panic = "abort"`.
8. QEMU error message strings that appear in QEMU and libvirt test expectations are copied verbatim and checked in CI.
