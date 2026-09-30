# 19. Security and confidential computing

A VMM is a parser of hostile input with host privileges. The guest controls every register write to every emulated device, disk images come from users, migration streams and NBD exports arrive over networks, and the monitor accepts commands that open files and spawn processes. QEMU's security history is a history of those inputs reaching C code that trusted them. ruvm is written in Rust largely because of that history, but Rust removes only some bug classes. This document states the threat model, what the language removes, what it does not, the engineering rules that cover the remainder (unsafe budgets, verification, fuzzing, reentrancy guards, overflow policy), process-level containment, confidential computing support (SEV, SEV-ES, SEV-SNP, TDX, Arm CCA), CET, secure and measured boot, supply chain, and the security support policy.

## Threat model

ruvm adopts QEMU's model from docs/system/security.rst and makes it more precise where Rust lets us enforce something QEMU can only recommend.

Untrusted, may be buggy or malicious:

- The guest, including guest kernel and guest firmware. Anything a guest can write to MMIO, PIO, config space, virtqueues, DMA buffers, hypercalls, MSRs, or fw_cfg is attacker input.
- User-facing interfaces: VNC, SPICE, the WebSocket VNC proxy, D-Bus display clients.
- Network protocols: NBD client and server, live migration streams, vhost-user and vfio-user peers to the extent listed below, the user-mode network stack, iSCSI, and any TLS peer before authentication.
- User-supplied files: disk images in every format, kernels, initrds, device trees, firmware files, ACPI tables passed with `-acpitable`, IGVM files, snapshots.
- Passthrough devices: a VFIO device is hardware controlled by the guest, and can DMA anywhere its IOMMU mapping allows.

Trusted:

- The management layer (libvirt or equivalent), the QMP and HMP monitors, and the command line. security.rst is explicit that "the monitor console should be considered to have privileges equivalent to those of the user account QEMU is running under", because `blockdev-add` opens arbitrary files and `migrate` can spawn processes. ruvm keeps that position. Monitor bugs are ordinary bugs unless reachable from an untrusted input (for example a malformed image whose metadata is echoed through `query-block`).
- The host kernel, the hardware, and in non-confidential mode the host administrator.

QEMU's policy has two scopes. The "virtualization use case" covers hardware accelerators with a listed set of machine types (aarch64 `virt`; x86 `microvm`, `xenfv`, `xenpv`, `xenpvh`, `pc`, `q35`; s390x `s390-ccw-virtio`; loongarch64 `virt`; ppc64 `pseries`; riscv `virt`). The "non-virtualization use case", meaning TCG and other machine types, is explicitly not a security boundary today. security.rst also scopes out several classes even within the virtualization case: asserts reachable only from guest kernel privilege, flaws between QEMU and vhost-user or vfio-user backends (no security boundary is claimed between them), unbounded memory allocation, degraded behaviour reachable only from guest root, L2-to-L0 nested issues, migration and snapshot load failures, and low-severity issues.

ruvm starts from the same scope and makes two changes, both recorded as decisions in document 25.

1. Every QOM type carries a `secure` flag, matching the "Encode object type security status in code" series by Daniel Berrangé that is merging into QEMU master for 11.2 ([cover letter](http://www.mail-archive.com/qemu-devel@nongnu.org/msg1224524.html)). QEMU's flag is a bool in `TypeInfo`, default false, and `-compat insecure-types=accept|warn|reject` controls whether unmarked types may be instantiated, with status reported by `qom-list-types` and `query-machines`. ruvm implements the flag as a required field of `register_type!` (the macro does not compile without `secure = true` or `secure = false`), so no type is ever unclassified in ruvm even though QEMU's default is implicit. The QMP-visible values match QEMU's for every type QEMU has classified; for types QEMU has not classified yet, ruvm reports `false` so that `-compat insecure-types=reject` behaves identically.
2. ruvm treats vhost-user and vfio-user backends it ships (ruvm-storage-daemon exports, ruvm's own virtiofsd-compatible and GPU daemons) as separate trust domains, and the VMM side validates everything they write into shared memory as if it came from the guest. This is stricter than QEMU's stated scope, costs little because the same virtqueue parsers are used, and is what makes process decomposition (below) a real security feature rather than only an operational one.

Out of scope for ruvm's security policy, same as QEMU: TCG as an isolation mechanism in 1.0 (tracked for post-1.0; see the policy section), host denial of service by memory exhaustion (deploy cgroups), and side channels in the CPU that the host kernel and firmware must mitigate. ruvm does implement the guest-visible side of CPU mitigations (CPUID bits, ARCH_CAPABILITIES MSR passthrough, SPEC_CTRL save and restore) identically to QEMU, as described in document 09.

## Which QEMU CVE classes Rust eliminates

The table uses real QEMU CVEs, each verified against the linked advisory, and classifies them by root cause.

| CVE | Component | Root cause | Status in ruvm |
| --- | --- | --- | --- |
| [CVE-2015-3456](https://access.redhat.com/security/cve/cve-2015-3456) (VENOM) | hw/block/fdc.c floppy controller | FIFO index not bounded, guest writes past a fixed buffer | Eliminated as memory corruption: indexing a `[u8; 512]` out of range is a checked panic. Still a guest-triggerable crash if the logic is ported wrong, so the port must bound the index like the QEMU fix. |
| [CVE-2020-14364](https://access.redhat.com/security/cve/cve-2020-14364) | hw/usb/core.c | `setup_len` stored before validation, later IN/OUT copies overflow the 4096-byte `data_buf` | Memory corruption eliminated (slice bounds). The state machine bug (length accepted then rejected) is a logic error that fuzzing must find. |
| [CVE-2019-14378](https://www.openwall.com/lists/oss-security/2019/08/01/2) | libslirp ip_reass | pointer miscalculation when reassembling a large first fragment | Eliminated in Rust code. ruvm's user-mode networking prefers passt (document 15); if libslirp is linked for compatibility it keeps this class of risk and runs in a sandboxed helper. |
| [CVE-2021-3929](https://gitlab.com/qemu-project/qemu/-/issues/782) | hw/nvme | DMA reentrancy: device DMA targets its own MMIO, handler frees state in use | Use-after-free eliminated by ownership, but reentrancy itself is a logic hazard; see the reentrancy section. |
| [CVE-2024-3446](https://access.redhat.com/security/cve/cve-2024-3446) | virtio-gpu, virtio-serial-bus, virtio-crypto | DMA reentrancy through bottom halves not covered by the 2023 guard, double free | Double free eliminated. Reentrancy handled by the uniform guard. |
| [CVE-2021-3416](https://www.openwall.com/lists/oss-security/2021/02/26/1) | many NICs in loopback mode | reentrant delivery, unbounded recursion (CWE-835) | Not eliminated by the language. Stack overflow becomes a guaranteed abort (guard page) instead of possible corruption; the fix is the same design rule as QEMU's `qemu_receive_packet`. |
| [CVE-2020-11869](https://www.openwall.com/lists/oss-security/2020/04/24/2) | hw/display/ati_2d.c | integer overflow in `dst_x + (dst_y + dst_height) * dst_stride` feeding a bounds check | Not eliminated by default Rust (release builds wrap). Covered by ruvm's overflow policy below. |
| [CVE-2024-4467](https://access.redhat.com/security/cve/cve-2024-4467) | qemu-img info, qcow2 data-file | a `json:{}` string in image metadata is interpreted as a block graph description, opening arbitrary host files | Not eliminated. Pure logic bug about trusting image contents. ruvm's rule: image metadata never reaches the option parser (see below). |

The pattern is consistent. Spatial memory errors (out-of-bounds read or write) and temporal ones (use-after-free, double free) are the classes that turn a guest bug into host code execution, and they are the ones safe Rust removes. What remains is: logic errors in state machines, trusting data that should not be trusted, arithmetic on guest-controlled sizes, unbounded loops and recursion, and panics. Those produce denial of service far more often than escape, but CVE-2024-4467 shows logic bugs can also be confidentiality and integrity bugs.

Research results support the same split. Morphuzz (Bulekov et al., [USENIX Security 2022](https://www.usenix.org/conference/usenixsecurity22/presentation/bulekov)) reported 61 bugs to QEMU from fuzzing 33 devices, with DMA data races and double fetches among them. ViDeZZo (Liu et al., [IEEE S&P 2023](https://nebelwelt.net/files/23Oakland4.pdf)) found 28 new bugs across QEMU and VirtualBox by modelling intra- and inter-message dependencies. HyperPill (Bulekov et al., [USENIX Security 2024](https://www.usenix.org/system/files/usenixsecurity24-bulekov.pdf)) fuzzes hypervisors through the hardware virtualization interface using snapshots of a running hypervisor. Many of the bugs these tools found are assertion failures and hangs, which Rust turns into panics and hangs, not into safety. Fuzzing therefore stays mandatory in ruvm.

### Arithmetic policy

Every crate that parses guest or file input (all ruvm-hw-*, ruvm-block format drivers, ruvm-net, ruvm-chardev, ruvm-ui protocol code, ruvm-migration load path, ruvm-firmware loaders) builds with `overflow-checks = true` in the release profile. A wrap becomes a panic, which is a DoS, not a bounds-check bypass. Hot paths that need wrapping arithmetic (checksums, ring indices modulo queue size, guest register emulation where wrapping is architectural) use `wrapping_*` or `Wrapping<T>` explicitly, which makes intent reviewable. Size math on guest values uses a small `GuestSize` newtype whose operations return `Option` and forces a decision at every call site. The JIT, the JIT backends, and ruvm-softfloat are exempt: they operate on guest register values that are architecturally defined to wrap, and the checks would cost real time on the translation and execution paths. Document 21 tracks the cost of the checks in the device crates; the budget is under 1% on the virtio-blk and virtio-net benchmarks, and a crate that exceeds it must justify the exemption in review rather than turn checks off silently.

### Panics

A panic in a device is a denial of service against that VM. The process is built with `panic = "abort"`, because unwinding through device code with partially updated state and then continuing would be worse than stopping. This matches QEMU's treatment of `abort()` and `assert()` failures, and security.rst's triage rule applies unchanged: a panic reachable from guest root only is a hardening bug, reachable from an unprivileged guest user it may be a security bug. Panics print the device's QOM path and the last MMIO access from a small per-device ring buffer of recent accesses (a new decision, to be reflected in document 12) before aborting, which makes fuzz findings and field crashes actionable.

### Trusting image contents

CVE-2024-4467 happened because a string stored in a qcow2 header (the external data file name) could be `json:{...}`, which QEMU's `bdrv_open` parses as a full block graph specification. The general rule in ruvm-block: strings from image files (backing file names, data file names, bitmap names, snapshot names) are typed as `UntrustedImageString` and there is no conversion from it to a block options object. Opening a backing file from metadata goes through a function that accepts only a plain filename or a whitelisted protocol URL, resolves it relative to the image directory like QEMU, and refuses `json:`. When QEMU's behaviour accepts something ruvm refuses (for example old images deliberately using `json:` backing strings created by a trusted admin), ruvm requires the explicit `-blockdev` graph, as QEMU itself now recommends. This is an intended difference listed in document 02.

## Memory safety strategy

Rust's guarantee holds only outside `unsafe`. A VMM needs `unsafe` in real quantities: ioctls, mmap of guest RAM, lock-free rings shared with the guest, JIT code buffers, FFI to the TCG plugin ABI, vhost shared memory. The strategy is to concentrate it in few crates, budget it, document every block, and verify the concentrated parts harder than the rest.

### Unsafe budget per crate

`cargo xtask unsafe-audit` counts `unsafe` blocks, `unsafe fn`, `unsafe impl`, and `extern` blocks per crate using a syn-based scanner, and compares against a checked-in budget file. CI fails if a count rises without the budget file changing in the same commit, and changes to the budget file require approval from a member of the security review group (CODEOWNERS). Initial budgets:

| Crates | Budget | Rationale |
| --- | --- | --- |
| ruvm-hw-* (all device crates), ruvm-machine-*, ruvm-block format drivers, ruvm-net protocol code, ruvm-chardev, ruvm-ui protocol code, ruvm-monitor, ruvm-qapi generated code, ruvm-ga | 0, enforced with `#![forbid(unsafe_code)]` | These parse attacker input. They reach guest memory only through ruvm-mem's safe `GuestMemory` API (copy in, copy out, bounds-checked `VolatileSlice`) |
| ruvm-mem | about 60 blocks | mmap of RAM blocks, volatile access, RCU FlatView pointer swap, dirty bitmap atomics |
| ruvm-sys | unbounded, but every ioctl wrapper is one small function | KVM, HVF, WHPX, vfio, iommufd, userfaultfd bindings |
| ruvm-aio | about 40 | io_uring SQE and CQE rings, buffer registration, kqueue and IOCP |
| ruvm-jit, ruvm-jit-<host> | about 30 per backend | executable memory, W^X toggling, calling generated code, patching chained jumps |
| ruvm-plugin | about 50 | C ABI for qemu-plugin.h |
| ruvm-base | about 20 | epoch RCU, intrusive lists |
| ruvm-hw-virtio split and packed ring core | about 10 | guest-shared ring access with explicit ordering |

The device crates having zero unsafe is the single most important rule in this document. Nearly every historical QEMU escape was in device emulation (floppy, USB, NICs, display, SCSI, NVMe). In ruvm that code cannot express an out-of-bounds write.

### SAFETY comments and review

Every `unsafe` block carries a `// SAFETY:` comment naming the invariant that makes it sound and where that invariant is established, enforced by `clippy::undocumented_unsafe_blocks` at deny level. Every `unsafe fn` has a `# Safety` doc section (`clippy::missing_safety_doc`). Guest memory access is the place where soundness is subtle: guest RAM can change underneath the host at any time (other vCPUs, DMA from passthrough devices), so ruvm-mem never produces a `&[u8]` or `&mut [u8]` into guest RAM. It exposes `VolatileSlice` and copy functions, the same design as rust-vmm's vm-memory crate, because a Rust reference to memory another agent mutates concurrently is undefined behaviour even if the code "only reads". Double fetch bugs, where a device validates a guest field and then reads it again, are prevented by construction: descriptor and request headers are copied into host structs once, and the parsed copy is used thereafter.

### Miri and Kani

Miri runs in CI on every crate's unit tests that do not issue syscalls, which covers ruvm-base, the RCU scheme, the pure parts of ruvm-mem (FlatView construction, dispatch lookup, dirty bitmap operations with a mocked backing), the virtqueue ring logic against a heap-backed fake guest memory, ruvm-softfloat, the JIT IR and register allocator, the qcow2 and VMDK metadata parsers, and ruvm-vmstate codecs. Tree Borrows mode is used as the primary check with Stacked Borrows as a second pass, since the two catch different aliasing mistakes in pointer-heavy code. Miri is slow, so the suite runs nightly in full and per-PR on changed crates.

Kani (bounded model checking for Rust) is used for small, high-value properties where testing cannot give confidence:

- ruvm-mem: address arithmetic in `FlatView::translate` never yields a host pointer outside the mapped block for any `(addr, len)` input; region splitting and merging preserves coverage.
- ruvm-hw-virtio: descriptor chain walking terminates and never visits more than queue size descriptors for any ring contents, and indirect descriptor tables are bounded (QEMU has had loops here).
- ruvm-block qcow2: L1/L2 index computation and refcount block addressing cannot overflow for any header satisfying the checks in `qcow2_do_open`.
- ruvm-vmstate: decoding any byte string either fails or produces field values within declared bounds.
- ruvm-mem `GuestSize` operations and the SAFETY invariants of a curated set of about 30 unsafe functions in ruvm-mem and ruvm-aio, expressed as Kani harnesses with contracts.

Harnesses live next to the code under `#[cfg(kani)]` and run in a nightly CI job with a time budget per harness. Kani proofs are bounded, so each harness documents its bound (for example queue sizes up to 16 for exhaustive descriptor checking, with the loop argument extended by an inductive invariant written in the comment and checked for the general case by review).

### Fuzzing

Two layers, both continuous.

Unit-level: cargo-fuzz (libFuzzer) targets for every parser that consumes untrusted bytes, including every image format header and metadata walker, the NBD client and server protocol, VNC and SPICE message parsers, the migration stream loader, the QMP JSON parser (monitor is trusted, but the parser is also used by the guest agent client path and it is cheap to fuzz), the IGVM parser, the FDT and ACPI table loaders used for user-supplied files, the slirp replacement if any, and USB descriptor parsing. Targets are also built for AFL++ and run in OSS-Fuzz, the same infrastructure QEMU uses.

Device-level: a Morphuzz-style generic device fuzzer driven through qtest. ruvm implements QEMU's qtest protocol (canon: ruvm-accel-qtest) so that the same fuzzer design works: the fuzzer interprets its input as a sequence of operations (MMIO and PIO reads and writes to the device's regions, clock steps, and DMA pattern fills), and a DMA hook intercepts every device-initiated read of guest memory and fills the accessed range from the fuzz input on first access, which is the core Morphuzz idea that lets a generic fuzzer reach deep device states without per-device grammars. On top of that, ViDeZZo's dependency annotations are added as optional per-device grammars for the devices where they measurably raise coverage (virtio, xHCI, NVMe, e1000e, SDHCI). The harness runs in-process with a fork server or a snapshot reset (the device is reset with the three-phase reset and guest RAM restored from a copy-on-write snapshot) so iterations are independent, which is the reproducibility property that HyperPill's authors pointed out ViDeZZo lacked. Each device crate must ship its fuzz configuration (which machine, which device options, which regions) before it can be marked `secure = true`. Crashes are reproduced as qtest scripts, which the same tooling can replay against QEMU; a crash that reproduces in both is reported to QEMU's security list as well.

Differential fuzzing is a third, cheaper layer: the same qtest operation sequence is replayed against QEMU 11.1 and ruvm, and device-visible outputs (MMIO read values, DMA writes, interrupts raised) are compared. Divergence is a compatibility bug, and in practice it also finds bugs in QEMU.

## Device reentrancy guard

DMA reentrancy is the class that Rust does not solve alone: a device's MMIO handler starts DMA, the DMA target address is the same device's MMIO region, and the device handler runs again while its state is half updated. In QEMU the consequences were use-after-free and double free (CVE-2021-3929, CVE-2024-3446) and infinite recursion (CVE-2021-3416). QEMU's fix, by Alexander Bulekov ([patch series](https://patchew.org/QEMU/20230427211013.2994127-1-alxndr@bu.edu/)), added `MemReentrancyGuard { bool engaged_in_io; }` to `DeviceState`, checked in `access_with_adjusted_size()` in system/memory.c: if the owning device is already engaged in I/O, the access is dropped with `MEMTX_ERROR` and a trace event; regions may opt out with `disable_reentrancy_guard`. Bottom halves were covered by `qemu_bh_new_guarded`, network delivery by passing the guard to `qemu_new_nic()`, and later timers were patched one by one as holes were found.

In ruvm the problem shows up differently. A device's state is behind its lock (canon concurrency model). A reentrant MMIO access from the device's own DMA would try to take the same lock on the same thread, which with a plain mutex is a deadlock and with a reentrant mutex is exactly QEMU's bug. ruvm makes reentrancy detection structural:

- Every entry into device code goes through `DeviceCell::enter(&self, ctx) -> Result<DeviceGuard, Reentrant>`, whether the entry is MMIO or PIO dispatch, a bottom half, a timer callback, an ioeventfd or irqfd handler, a network backend delivery, a chardev read callback, a block completion, or a migration hook. There is no other way to get `&mut DeviceState`, so a new entry point cannot forget the guard; QEMU's hole-by-hole history (MMIO first, then bottom halves, then NICs, then timers) cannot repeat.
- `enter` records the device id in a thread-local stack of active devices. If the id is already on the stack, the access fails. For MMIO the failure is `MemResult::Err(MemTxError)` with a `memory_region_reentrant_io` trace event under QEMU's name, which is guest-visible behaviour identical to QEMU's guard. Devices that QEMU marks `disable_reentrancy_guard` declare the same opt-out on the region and get a scoped reentrant entry with the state already borrowed, so they must structure the code to allow it.
- Cross-device reentrancy (device A's DMA hits device B, whose handler DMAs into A) is caught the same way because the stack holds all active devices on this thread. Across threads it cannot occur synchronously, since DMA in ruvm is a memory access by the calling thread.
- Recursion depth is also bounded: `enter` fails if the active stack exceeds 16 devices, turning pathological chains (the loopback NIC case) into a dropped access instead of a stack overflow.

The guard is cheap: a thread-local push and pop and a comparison over a stack that is almost always of depth one.

## Sandboxing

Memory safety lowers the probability of a guest getting code execution in the VMM; sandboxing limits what it gets if it does. ruvm follows security.rst's principle of least privilege ("the QEMU process should not have access to any resources that are inaccessible to the guest") and applies the in-process mechanisms itself, while leaving SELinux, AppArmor, namespaces and cgroups to the management layer as QEMU does.

### seccomp and -sandbox

`-sandbox on[,obsolete=allow|deny][,elevateprivileges=allow|deny|children][,spawn=allow|deny][,resourcecontrol=allow|deny]` is implemented with the same semantics as system/qemu-seccomp.c. QEMU's filter is a deny-list with sets: `QEMU_SECCOMP_SET_DEFAULT` (always denied when on: reboot, swapon, swapoff, syslog, mount, umount, kexec_load, and legacy unused calls such as afs_syscall, break, ftime, getpmsg, gtty, lock, mpx, prof, profil, putpmsg, security, stty, tuxcall, ulimit, vserver), `QEMU_SECCOMP_SET_OBSOLETE` (readdir, _sysctl, bdflush, create_module, get_kernel_syms, query_module, sgetmask, ssetmask, sysfs, uselib, ustat), `QEMU_SECCOMP_SET_PRIVILEGED` (setuid, setgid, setpgid, setsid, setre*id, setres*id, setfs*id), `QEMU_SECCOMP_SET_SPAWN` (fork, vfork, execve, execveat, setns, unshare, and clone and clone3 with argument filtering: clone must have the thread flags CLONE_VM, CLONE_FS, CLONE_FILES, CLONE_SIGHAND, CLONE_THREAD, CLONE_SYSVSEM, CLONE_SETTLS, CLONE_PARENT_SETTID, CLONE_CHILD_CLEARTID set, and must not have CLONE_PIDFD, CLONE_PTRACE, CLONE_VFORK, CLONE_PARENT, or any CLONE_NEW* namespace flag), and `QEMU_SECCOMP_SET_RESOURCECTL` (setpriority, sched_setparam, sched_setscheduler except to SCHED_IDLE, sched_setaffinity). Actions follow QEMU exactly: most entries use `SCMP_ACT_TRAP`, which `qemu_seccomp_update_action` upgrades to `SECCOMP_RET_KILL_PROCESS` when the kernel supports it; the clone flag rules and the resource control set return `EPERM`, and clone3 returns `ENOSYS` so that the C library falls back to clone, where the flag rules apply. QEMU installs its filter with TSYNC so it covers every thread, and ruvm does the same for `-sandbox`. ruvm generates the BPF program with the `seccompiler` crate from rust-vmm (used by Firecracker and Cloud Hypervisor), avoiding a libseccomp dependency, and is tested to produce the same allow and deny decisions as QEMU's filter on the same syscall and argument vectors using a table-driven test that runs both filters under `SECCOMP_RET_TRACE`.

The deny-list is QEMU's contract: libvirt passes `-sandbox on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny` and expects nothing else to break. ruvm adds a second, much stronger mode as an opt-in: `-object ruvm-sandbox,id=sb0,profile=strict`, an allow-list filter generated per thread role (vCPU threads, iothreads, main thread, monitor thread) from the syscalls those roles are known to need, in the Firecracker style. A vCPU thread under KVM needs about a dozen syscalls (ioctl on the vCPU fd with a fixed set of request numbers, futex, a few for signals and timers). Because ruvm's threads have well-defined roles (canon concurrency model) this is practical in ruvm where it is not in QEMU, whose threads can run nearly any code under the BQL. Unlike `-sandbox`, these filters are installed per thread without TSYNC, at thread start, after the thread's resources are opened, and they stack on top of the QEMU-compatible filter when both are enabled. The strict profile is a new ruvm decision, recorded in document 25; it is incompatible with features that spawn processes (the `exec:` migration transport, bridge helper, `-netdev tap` with script) and ruvm refuses the combination at startup with a clear error instead of failing at runtime with SIGSYS.

### Landlock

On Linux 5.13 and later, ruvm can restrict its own filesystem access with Landlock, opt-in through the same `ruvm-sandbox` object (`landlock=on`). After startup, the set of paths the configuration references (image files and their backing chains, firmware, sockets, the monitor socket directory, `/dev/kvm`, `/dev/vfio/*`, `/dev/net/tun`, `/dev/vhost-*`) is collected, and a ruleset allowing only those paths with only the needed access rights is applied. Hotplug of a new file after the ruleset is applied must go through fd passing (`add-fd`, `getfd`), which is what libvirt does anyway. Network restriction (Landlock ABI v4 and later) limits TCP bind and connect to the ports named in the configuration (migration, NBD, VNC). If the kernel's Landlock ABI is older than what is requested, ruvm applies what is available and reports the effective level through `__io.github.tamnd.ruvm_query-sandbox`.

### macOS and Windows

On macOS, Hypervisor.framework requires the `com.apple.security.hypervisor` entitlement and ruvm is signed with it plus `com.apple.security.cs.allow-jit` for MAP_JIT (document 08). ruvm can run inside the App Sandbox with the `com.apple.security.app-sandbox` entitlement when launched by a GUI front end; in that case file access comes from security-scoped bookmarks the front end passes, the same fd passing model as libvirt. The deprecated `sandbox_init` API and `sandbox-exec` profiles are not used. On Windows, ruvm supports running in an AppContainer with WHPX; the details are platform plumbing in ruvm-sys.

### Privilege dropping

`-run-with user=username|uid:gid` and `-run-with chroot=dir` behave as in QEMU: after all resources are opened and before guest execution starts, ruvm calls setgroups, setgid, setuid (for `username` it applies the user's supplementary groups; for `uid:gid` only the gid), and chroot happens immediately before. ruvm additionally drops all capabilities from the bounding set and sets `PR_SET_NO_NEW_PRIVS` whenever `-sandbox on` or the ruvm-sandbox object is used, and on Linux clears `PR_SET_DUMPABLE` when a confidential guest is configured so that other processes of the same user cannot ptrace the VMM (which for confidential guests would not expose private memory, but would expose the VMM's own secrets such as TLS keys and launch secrets).

### Process decomposition

QEMU supports moving device backends out of process: vhost-user (virtio devices implemented in another process that maps guest RAM through shared memory fds), vfio-user (any PCI device in another process, protocol from libvfio-user), and qemu-storage-daemon for block exports. ruvm supports all three as client and server (documents 13, 14, 16), and uses them for a decomposed deployment mode:

- ruvm-storage-daemon serves block devices over vhost-user-blk, so image format parsing (the source of CVE-2024-4467 style bugs) runs in a process with access to the images but no access to KVM or the network.
- Network backends that need host privileges or complex parsing (user-mode networking, passt) run as separate processes, which passt already does.
- Display protocol servers (VNC, SPICE) can run in a ruvm-display helper over the D-Bus display interface QEMU already defines (`-display dbus`), which is where remote clients connect, so a VNC parsing bug does not land in the process that holds `/dev/kvm`.
- Selected emulated devices with large attack surface (USB host controllers, GPU with virgl or native context) can run as vfio-user servers built from the same ruvm-hw crates.

The main VMM process in this mode holds the KVM fds, the vCPU threads, the memory map, and the interrupt controllers, and runs under the strict seccomp profile. As noted in the threat model, ruvm treats each helper as untrusted from the VMM's side, which QEMU does not claim. The cost is latency on paths that cross processes; vhost-user data paths avoid it through shared memory and eventfds, so throughput is unchanged in the measurements that QEMU and SPDK users already rely on, and document 21 carries ruvm's own numbers.

## Confidential computing

In a confidential VM the host, including ruvm, is outside the trust boundary of the guest. The hardware encrypts guest private memory and register state, and the guest verifies its launch state by remote attestation. ruvm's job is to set up the VM correctly, report accurate launch parameters, provide shared-memory I/O, and otherwise stay out of the way. All confidential computing support lands in M8 (canon milestones) and is implemented in ruvm-accel-kvm, ruvm-target-x86 (and ruvm-target-arm for CCA), and ruvm-machine-x86, behind QEMU's `confidential-guest-support` machine property and the same QOM objects.

### Common model

QEMU's `ConfidentialGuestSupport` class (include/system/confidential-guest-support.h) has hooks for KVM init, reset, launch data measurement (`set_guest_state`), memory map queries for IGVM, and since 11.0 a `can_rebuild_guest_state` flag. ruvm maps it to a trait:

```rust
pub trait ConfidentialGuest: Object {
    fn kvm_init(&self, vm: &KvmVm) -> Result<()>;
    fn launch_update(&self, gpa: u64, data: &[u8], kind: PageKind) -> Result<()>;
    fn set_vcpu_state(&self, vcpu: &KvmVcpu, state: &VmsaOrTdVp) -> Result<()>;
    fn launch_finish(&self) -> Result<()>;
    fn can_rebuild_guest_state(&self) -> bool;
    fn private_memory(&self) -> PrivateMemoryModel; // GuestMemfd or InPlaceEncrypted (legacy SEV)
}
```

Rules that apply to every technology:

- Private memory is allocated with guest_memfd (`KVM_CREATE_GUEST_MEMFD`) for SEV-SNP and TDX, as QEMU does through the RAMBlock `guest_memfd` field set when the machine has `require_guest_memfd`. ruvm-mem never maps guest_memfd pages into the VMM, so there is no host virtual address for private memory at all, and a device model bug cannot read it. Shared pages (bounce buffers, virtqueues after the guest converts them) live in ordinary memory. Private to shared conversion requests from the guest (`KVM_EXIT_MEMORY_FAULT`, or the hypercall exits for SNP page state changes and TDX MapGPA) are handled by `kvm_convert_memory` equivalent logic in ruvm-accel-kvm that punches holes in whichever side is being released.
- The VMM must not touch: private memory contents, encrypted register state (VMSA for SEV-ES and SNP, TD VP state for TDX) after launch, debug registers when debug is disabled by policy, and the guest's secure TSC. Code paths that read guest memory for convenience (HMP `x`, `memsave`, `dump-guest-memory`, the gdbstub, `info registers`, `info tlb`) check `MemTxAttrs` and the confidential state and return QEMU's errors instead of garbage, matching QEMU's behaviour on these machines.
- Device choices are constrained. Emulated devices still work over shared memory, but everything must use DMA through shared buffers (virtio with `iommu_platform=on` and restricted DMA in the guest). ruvm refuses configurations QEMU refuses (for example TDX requires a split irqchip and has no SMM).
- Migration of confidential guests is not supported in 1.0, as in QEMU. Snapshots are refused.

### SEV and SEV-ES

`-object sev-guest,id=sev0,cbitpos=51,reduced-phys-bits=1,policy=0x...` with properties `sev-device`, `cbitpos`, `reduced-phys-bits`, `kernel-hashes`, `debug-swap`, `dh-cert-file`, `session-file`, `policy`, `handle`, `legacy-vm-type` (from `SevCommonProperties` and `SevGuestProperties` in qapi/qom.json). SEV-ES is selected by the ES bit in `policy`. Launch uses the `KVM_SEV_LAUNCH_START`, `LAUNCH_UPDATE_DATA`, `LAUNCH_UPDATE_VMSA` (ES), `LAUNCH_MEASURE`, `LAUNCH_SECRET`, `LAUNCH_FINISH` sequence from target/i386/sev.c. With `kernel-hashes=on`, ruvm places the SHA-256 hashes of kernel, initrd, and command line in the OVMF-defined table in the firmware so the measurement covers direct kernel boot, byte-compatible with QEMU's `sev_add_kernel_loader_hashes`. The attestation plumbing is QMP: `query-sev`, `query-sev-capabilities`, `query-sev-launch-measure` (the launch digest for the guest owner), `sev-inject-launch-secret` (inject a secret wrapped by the guest owner before launch finishes), and `query-sev-attestation-report`. The VM is started with `-S`, measured, the secret injected, and then `cont`, the flow libvirt implements for `<launchSecurity type='sev'>`. SEV-ES termination requests are reported as GUEST_PANICKED, as in QEMU 11.0.

### SEV-SNP

`-object sev-snp-guest,id=sev0,policy=...` with `guest-visible-workarounds`, `id-block`, `id-auth`, `author-key-enabled`, `host-data`, `vcek-disabled`, `secure-tsc`, `tsc-frequency`. Launch uses `KVM_SEV_SNP_LAUNCH_START`, `SNP_LAUNCH_UPDATE` with page types (normal, zero, secrets, CPUID, VMSA), and `SNP_LAUNCH_FINISH` with the ID block and host data. The CPUID page is built from the vCPU model. If the secure processor rejects it, the firmware writes back the values it would accept; QEMU's sev.c reports the mismatching leaves and fails the launch, and ruvm does the same with the same messages, so a bad `-cpu` choice is diagnosable. Attestation for SNP happens between the guest and the AMD secure processor through `SNP_GUEST_REQUEST`, handled by KVM; the VMM's role is to supply `host-data` and the ID block at launch, and ruvm adds nothing to that path.

IGVM: `-object igvm-cfg,id=igvm0,file=path` plus `-machine igvm-cfg=igvm0` loads an IGVM file (the format from Microsoft's igvm project, used by COCONUT-SVSM and OpenHCL) that describes the initial memory image, VMSA, and policy. QEMU gained IGVM support in 10.1 for SEV, SEV-ES, and SEV-SNP (and for non-confidential guests); per QEMU's own July 2026 documentation patches, TDX guests do not yet support IGVM. ruvm implements the same directive processing (page data, parameter areas, memory map, VP context, policy and ID block from the file) in ruvm-firmware using the Rust `igvm` crate, which is the reference parser maintained by the IGVM authors, with ruvm's fuzz targets added. IGVM is the recommended path because it lets the guest owner compute the expected measurement from one file without knowing anything about ruvm.

### Reset of confidential guests (QEMU 11.0)

Before 11.0, a reboot inside an SEV-ES, SEV-SNP, or TDX guest terminated QEMU, because encrypted CPU state and locked private memory cannot be reset in place. Ani Sinha's series merged for 11.0 makes them resettable by throwing the VM away at the KVM level and building a new one inside the same QEMU process: `kvm_reset_vmfd` unregisters memory listeners, notifies a `VmfdChangeNotifier` list (with a `pre` flag for subsystems that must act before the switch), closes the old VM fd, creates a new one, calls `kvm_arch_on_vmfd_change` (which reruns `kvm_arch_init` so SEV or TDX context is created again, and re-registers SMM listeners), recreates guest_memfd for every RAMBlock with `RAM_GUEST_MEMFD` (`ram_block_rebind`), rebinds existing vCPUs to new vCPU fds, and re-executes launch. Firmware is reloaded from the original file when IGVM is not used, because the old copy was in encrypted memory; with IGVM the directives are re-executed. The confidential class declares `can_rebuild_guest_state = true` for SNP and TDX. Devices that hold VM-fd-derived resources (irqfds, ioeventfds, Hyper-V VMBus event fds) re-associate through the notifier. A test-only machine property, `x-change-vmfd-on-reset`, exercises the same path on non-confidential guests.

ruvm implements this with its accelerator abstraction rather than notifiers: `Accel::rebuild_vm()` returns a new `KvmVm`, and every object that holds a VM-scoped handle holds it through a `VmScoped<T>` wrapper that the accelerator re-creates during rebuild, so the compiler finds every holder. The sequence is otherwise identical, and the guest observes the same thing it observes under QEMU: a cold boot with a fresh launch measurement. ruvm supports `x-change-vmfd-on-reset` with the same name so QEMU's functional test runs unchanged.

### TDX

`-object tdx-guest,id=tdx0` with `attributes`, `sept-ve-disable`, `mrconfigid`, `mrowner`, `mrownerconfig`, `quote-generation-socket`, `features` (TdxGuestProperties), merged in QEMU 10.1 and requiring Linux 6.16 or newer on the host. The launch flow is `KVM_TDX_INIT_VM` with the CPUID configuration filtered through what the TDX module reports as configurable, `KVM_TDX_INIT_VCPU`, `KVM_TDX_INIT_MEM_REGION` for the TDVF (TDX-capable OVMF) sections described by its metadata table, and `KVM_TDX_FINALIZE_VM`. Private memory is guest_memfd only. Attestation: the guest asks for a quote with the `GetQuote` TDVMCALL; KVM exits to the VMM, and QEMU forwards the request to the Quote Generation Service over the socket named by `quote-generation-socket` and writes the reply into the shared buffer the guest provided. ruvm implements the same exit handling and the same QGS message format. The VMM sees only the TD report and the quote, which are designed to be public. `mrconfigid`, `mrowner`, and `mrownerconfig` are passed through to the TD attributes so the guest owner can bind the quote to a configuration.

### Arm CCA

Arm's Confidential Compute Architecture runs guests as Realms managed by the RMM firmware. As of September 2026 neither side is upstream: the KVM host series by Steven Price reached v17 (September 2026, cut down to basic plumbing) and has not been merged into mainline Linux, and QEMU Realm support is an RFC from Linaro (v3, August 2026) that adds an `rme-guest` object used as `-machine confidential-guest-support=rme0 -object rme-guest,id=rme0` with a `memory-backend-guest-memfd` backend. Guest-side support has been in Linux since 6.14. ruvm's decision: design the `ConfidentialGuest` trait so a CCA implementation fits (Realm memory split into private lower half and shared upper half of the IPA space, Realm Initial Measurement from populated pages, RIPAS changes as conversion exits), track the RFC's QOM names, and ship CCA only after both the KVM uAPI and QEMU's object names are merged, because shipping against an unmerged uAPI would lock ruvm into an interface that is still changing between versions. This is recorded as an open question in document 25.

### What the VMM must not rely on

For all technologies, ruvm assumes that anything the confidential guest writes to shared memory is hostile, including in ways a normal guest would not do (a confidential guest has no reason to trust the host and may be deliberately odd, and a host bug that trusts it is still a host bug). The same zero-unsafe device crates and fuzzing apply unchanged.

## CET virtualization

QEMU 11.0 added KVM CET virtualization. The host side landed in Linux 6.18: shadow stacks (SHSTK) on Intel and AMD, and indirect branch tracking (IBT) on Intel only, since KVM explicitly hides IBT on SVM. For the VMM the work is CPUID and state: expose CPUID.(EAX=7,ECX=0):ECX.CET_SS and EDX.CET_IBT when the host and the CPU model allow, handle the XSAVE components for CET user and supervisor state (XFEATURE 11 and 12), and save and restore the CET MSRs (IA32_U_CET, IA32_S_CET, IA32_PL0_SSP through IA32_PL3_SSP, IA32_INT_SSP_TAB) in the CPU vmstate so migration carries them, with the section layout matching QEMU's so migration interoperates (document 17). CPU models in ruvm-target-x86 carry the same feature names (`shstk`, `ibt`) and the same version-dependent defaults as QEMU's target/i386/cpu.c. ruvm does not emulate CET under its JIT in 1.0; that is a post-1.0 item. After the 6.18 merge, host hangs were reported on some hosts when guests enabled shadow stacks, which is a KVM issue; ruvm does not work around it but documents that the feature can be masked with `-cpu ...,-shstk`.

## Secure boot and measured boot

Secure boot is a firmware feature: OVMF or AAVMF built with Secure Boot, with its variable store in a pflash device whose enrolled keys come from the distribution. ruvm's contribution is correctness of the pieces the firmware relies on: SMM emulation on q35 (`-machine smm=on`), so that the variable store is writable only from SMM, the `cfi.pflash01` write-protection semantics, and firmware descriptor JSON (docs/interop/firmware.json) so libvirt picks a Secure Boot build when asked. QEMU's `uefi-vars` device (host-side UEFI variable service, qapi/uefi.json) is implemented too, since it removes the need for SMM on Arm and is used by confidential and microVM setups.

Measured boot uses a vTPM. ruvm implements `tpm-tis`, `tpm-crb`, `tpm-tis-device` (Arm), and `tpm-spapr` with the `emulator` backend talking to swtpm over its control channel and data socket, and the `passthrough` backend. TPM state migration uses swtpm's state blobs exactly as QEMU does, so libvirt's handling of `--tpmstate` works. The firmware measures into PCRs and ruvm provides the ACPI TPM2 table and the TCG event log area. For confidential guests the vTPM must live inside the trust boundary (for example in COCONUT-SVSM loaded via IGVM), not in swtpm on the host; ruvm's role there is only to load the IGVM file.

## Supply chain

- Dependencies are reviewed with `cargo vet`: every crate in Cargo.lock must be audited by a ruvm reviewer or covered by an imported audit set from organisations whose criteria are published (Mozilla, Google, Bytecode Alliance, Embark, the rust-vmm project). New dependencies need an audit entry in the same PR. `cargo deny` enforces license compatibility with GPL-2.0-or-later (and the permissive-only rule for the MIT OR Apache-2.0 crates, canon) and bans known-vulnerable versions via the RustSec advisory database.
- Proc macros and build scripts are the highest-risk dependencies because they run at build time. The workspace allows a short, named list of proc-macro crates (syn, quote, proc-macro2, serde_derive, thiserror, linkme) plus ruvm's own, enforced by a `cargo xtask deps` check.
- Vendored inputs from QEMU (qapi/*.json, .hx files, trace-events, gdb-xml, decodetree files, firmware descriptors) are pinned to a QEMU git tag and the tag's commit hash is recorded; the import script refuses files that differ from that tag.
- Reproducible builds: release binaries are built with a pinned toolchain from rust-toolchain.toml, `--locked`, `SOURCE_DATE_EPOCH`, `-C remap-path-prefix`, and no build-time timestamps or host paths embedded. Two independent CI builders must produce bit-identical binaries for the Linux x86-64 and aarch64 release artifacts before a release is published. Firmware blobs shipped with ruvm are the upstream projects' own builds with their checksums recorded, not rebuilt by ruvm.
- Releases are signed with Sigstore (keyless, tied to the CI workflow identity) and ship an SBOM in SPDX format generated from Cargo.lock.

## Security support policy

ruvm's policy mirrors QEMU's so that operators do not have to learn two sets of rules.

- Scope: the virtualization use case (KVM, HVF, WHPX, MSHV, and Xen), with machine types listed as secure (QEMU's list above, reflected in each machine's `secure` flag) and devices whose types have `secure = true`. TCG, other machines, and insecure device types are not a security boundary in 1.0. Because ruvm device crates are memory safe, fuzzed, and differential-tested regardless of accelerator, ruvm intends to extend scope to TCG after 1.0 once the JIT has its own fuzzing coverage; until then TCG bugs are handled as ordinary bugs.
- Triage follows security.rst's boundary rules (guest root only asserts and panics are hardening bugs, memory exhaustion is not a security bug, migration load failures at the destination are not security bugs) plus ruvm's stricter rule that a vhost-user or vfio-user backend shipped by ruvm corrupting the VMM is a security bug.
- Reporting: a private address (security@ on the project domain) and GitHub private vulnerability reporting. Issues that also affect QEMU are coordinated with qemu-security under QEMU's process; ruvm does not publish before QEMU's embargo ends.
- Fixes: supported branches are the latest release and the previous minor release, plus any release designated long-term. Fixes ship as point releases with a RustSec advisory where the bug is in a published crate, and a CVE through GitHub's CNA.
- Every security fix adds a regression test: a qtest reproducer for device bugs, a fuzz corpus entry for parser bugs, and where the bug class could recur, a lint or a Kani harness that prevents the class rather than the instance.
