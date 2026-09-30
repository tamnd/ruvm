# 10. User-mode emulation: linux-user and bsd-user

User-mode emulation runs a single guest process on a host kernel: guest instructions go through the JIT (documents 07 and 08), and guest system calls are translated into host system calls. QEMU implements this in linux-user/ and bsd-user/, with shared code in common-user/. ruvm implements it in two L3 crates, ruvm-linux-user and ruvm-bsd-user, and must match QEMU 11.1.0's behaviour closely enough that the same binfmt_misc registrations, the same container images and the same cross-build pipelines work unchanged when /usr/bin/qemu-aarch64 is a symlink to ruvm. This document is scheduled for milestone M7 (document 23), and its performance target from the canon is at least 2x QEMU linux-user on SPEC CPU2017 intrate for the three host and guest pairs.

## Scope and binaries

QEMU 11.1.0 builds linux-user emulators for 33 configurations (configs/targets/*-linux-user.mak): aarch64, aarch64_be, alpha, arm, armeb, hexagon, hppa, i386, loongarch64, m68k, microblaze, microblazeel, mips, mipsel, mips64, mips64el, mipsn32, mipsn32el, or1k, ppc, ppc64, ppc64le, riscv32, riscv64, s390x, sh4, sh4eb, sparc, sparc32plus, sparc64, x86_64, xtensa, xtensaeb. There is no linux-user for avr, rx or tricore, and none for big-endian RISC-V (that is a system-mode property in 11.1, see document 09). bsd-user in 11.1.0 supports FreeBSD hosts only, with guest configurations aarch64, arm, i386, x86_64 and riscv64 (configs/targets/*-bsd-user.mak). The NetBSD and OpenBSD code paths that existed in older trees are not built.

Each name becomes an argv[0] alias for the ruvm multi-call binary (document 03). In practice distributions install user-mode emulators statically linked (qemu-user-static in Debian, qemu-user-static-* in Fedora) so they work inside foreign-architecture chroots and containers where the host's dynamic loader and libraries are absent. ruvm therefore produces a second build of the multi-call binary with only the user-mode personalities, statically linked against musl, target size under 25 MB for all 33 linux-user personalities together (the FreeBSD build carries the 5 bsd-user ones). That is a ruvm packaging decision; QEMU builds one static binary per target.

Code layout. QEMU's linux-user is about 14,700 lines in syscall.c alone, plus per-target directories for signal frames, cpu_loop, ELF details and syscall numbers. ruvm splits it as follows:

| ruvm module | QEMU equivalent | Content |
|---|---|---|
| ruvm-user-common (new crate, L3) | common-user/, accel/tcg/user-exec.c, parts of linux-user/mmap.c | guest address space, page flags, mmap engine, safe syscall, host signal entry, TB invalidation on writes |
| ruvm-linux-user::load | linux-user/elfload.c, linuxload.c, flatload.c, */elfload.c | ELF, FDPIC, bFLT, auxv, vDSO, commpage |
| ruvm-linux-user::sys | linux-user/syscall.c, syscall_defs.h, syscall_types.h, ioctls.h, fd-trans.c, uname.c | syscall dispatch, struct conversion, ioctls, fd translators |
| ruvm-linux-user::signal | linux-user/signal.c, */signal.c | signal queues, frame setup, sigreturn |
| ruvm-linux-user::proc | fake_open in syscall.c, */target_proc.h | /proc emulation |
| ruvm-linux-user::strace | strace.c, strace.list | -strace output |
| ruvm-bsd-user | bsd-user/, bsd-user/freebsd/ | FreeBSD equivalent of the above |

The new ruvm-user-common crate is a decision made here: QEMU shares only a small common-user directory between linux-user and bsd-user, and a lot of the FreeBSD code is copied from linux-user. ruvm puts the address space manager, mmap engine, signal plumbing and the user-mode JIT glue in one crate used by both, so bsd-user benefits from the page size mismatch work below without a second implementation. This crate is GPL like the rest of the user-mode code (it is derived from QEMU's mmap.c) and has been added to document 25 for inclusion in the canon crate list.

Per-target user-mode knowledge (syscall numbers, struct layouts, signal frames, ELF hwcaps, cpu_loop exit handling) lives in the ruvm-target crates behind the `UserArch` interface returned by `GuestArch::user_mode()` (document 09), so adding a target does not touch ruvm-linux-user's generic code.

## Process startup and command line

The command line and environment variables are compatibility surface, because binfmt registrations and container runtimes pass them. ruvm accepts exactly QEMU's option table from linux-user/main.c, each with its environment variable: -g (QEMU_GDB), -L (QEMU_LD_PREFIX), -s (QEMU_STACK_SIZE), -cpu (QEMU_CPU), -E and -U (QEMU_SET_ENV, QEMU_UNSET_ENV), -0 (QEMU_ARGV0), -r (QEMU_UNAME), -B (QEMU_GUEST_BASE), -R (QEMU_RESERVED_VA), -t (QEMU_RTSIG_MAP), -d, -dfilter, -D (QEMU_LOG, QEMU_DFILTER, QEMU_LOG_FILENAME), -tb-size, -strace (QEMU_STRACE), -seed (QEMU_RAND_SEED), -trace, -plugin (QEMU_PLUGIN), -one-insn-per-tb, -perfmap and -jitdump (QEMU_PERFMAP, QEMU_JITDUMP), -xtensa-abi-call0, and -version. The -p page size option was removed in 10.2 and ruvm does not accept it either. ruvm-specific options are only accepted with an x-ruvm- prefix and are never needed for normal operation.

binfmt_misc integration follows scripts/qemu-binfmt-conf.sh. The magic and mask strings per architecture are copied from that script (ruvm ships the script's data as a generated table and a `ruvm binfmt` subcommand that prints or installs registrations, including the systemd binfmt.d format and the Debian update-binfmts format the script supports). The flags matter:

- F (fix binary): the kernel opens the interpreter at registration time, which is what makes static user-mode emulators work in containers without the interpreter present inside. Requires the static build above.
- P (preserve argv0): the kernel passes the original argv[0] as an extra argument. The kernel reports that it did so through the AT_FLAGS_PRESERVE_ARGV0 bit of AT_FLAGS in the emulator's own auxiliary vector, which main.c reads with qemu_getauxval(AT_FLAGS); the script enables the flag with --preserve-argv0. ruvm reads the same bit and applies the same argument shuffling.
- O (open binary): the kernel passes an fd in AT_EXECFD, which QEMU uses in place of opening the path, so unreadable (execute-only) binaries work. ruvm does the same and uses the fd for /proc/self/exe emulation.
- C (credentials): setuid and setgid binaries get their credentials computed from the guest binary; implies O.

The script's --ignore-family option (skip registering an emulator for the host's own architecture family, for example i386 on x86_64) is honoured by `ruvm binfmt` using the same family table.

## Loading guest binaries

### ELF

The loader follows linux-user/elfload.c, which since the 11.x refactor keeps generic code in elfload.c and moves per-target pieces (hwcap computation, register initialization, core dump register layout, commpage setup) into linux-user/<arch>/elfload.c. ruvm mirrors that split: the generic loader in ruvm-linux-user::load, the target pieces in `UserArch`. The sequence is the one load_elf_binary and load_elf_image implement:

1. Read the ELF header, check class, data encoding and e_machine against the target (with the per-target elf_check_arch rules, for example MIPS n32 versus o32 via EF_MIPS_ABI2 and the ABI flags, or ppc64 ELFv1 versus ELFv2 via e_flags).
2. Compute the image's address range from PT_LOAD headers; for ET_DYN pick a load bias at ELF_ET_DYN_BASE (adjusted to two thirds of reserved_va when a reservation exists, as main.c does), for ET_EXEC map at the fixed addresses.
3. Before mapping anything, choose guest_base (see the next section) so the whole image range, and any commpage, fits.
4. Map each PT_LOAD with the mmap engine, zero the bss tail, and record PT_GNU_STACK for stack executability. PT_GNU_RELRO is left to the guest's dynamic loader, as the kernel does.
5. If PT_INTERP is present, load the interpreter the same way, resolving its path under the -L prefix (QEMU_LD_PREFIX) first, and falling back to the host path.
6. Parse PT_GNU_PROPERTY notes for targets where QEMU uses them: on aarch64 the BTI property (GNU_PROPERTY_AARCH64_FEATURE_1_BTI) marks executable mappings as guarded pages so BTI checks apply.
7. Build the initial stack: argv, envp, the auxiliary vector, AT_RANDOM bytes (seeded by -seed for reproducibility), AT_PLATFORM and AT_BASE_PLATFORM strings, and AT_HWCAP/AT_HWCAP2 computed from the CPU model's feature bits (for example GET_FEATURE_ID(aa64_gcs, ARM_HWCAP_A64_GCS) in linux-user/aarch64/elfload.c), so a guest glibc's ifunc resolution sees the features the -cpu model provides, not the host's. QEMU 11.1 does not emit AT_HWCAP3, AT_HWCAP4 or AT_MINSIGSTKSZ, and ruvm emits the same auxv entries in the same order. Whether to add AT_MINSIGSTKSZ for guests with large SVE or SME signal frames is listed in document 25; the answer will follow upstream.

The stack size default is TARGET_DEFAULT_STACK_SIZE, 8 MiB (80 MiB on hppa), raised to the host RLIMIT_STACK when that is larger and finite, and overridden by -s, exactly as main.c computes guest_stack_size.

### FDPIC and bFLT

FDPIC ELF loading is supported for exactly the targets QEMU supports: Arm (ELFOSABI_ARM_FDPIC) and Xtensa (ELFOSABI_XTENSA_FDPIC), detected by elf_is_fdpic on e_ident[EI_OSABI]. FDPIC places each segment independently and passes load maps to the program in registers, using the PER_LINUX_FDPIC personality (FDPIC_FUNCPTRS). The bFLT flat binary format for no-MMU systems (flatload.c) is built for targets that define TARGET_HAS_BFLT. Both are niche but part of the contract, and both are covered by small test binaries in ruvm's conformance suite since QEMU's tests do not exercise them much.

### Commpage, vsyscall and vDSO

Several targets need a page the kernel would provide:

- Arm: the kuser helpers page near 0xffff0000 (__kuser_cmpxchg, __kuser_get_tls, __kuser_memory_barrier), installed by linux-user/arm/elfload.c. On Arm the address falls inside the 32-bit guest space and the commpage must be aligned to the host page size, which is why QEMU computes `COMMPAGE & -host_page_size`.
- HPPA: the gateway page at address 0 with the light-weight syscall and atomic helpers, marked with page_set_flags in linux-user/hppa/elfload.c.
- x86_64: the legacy vsyscall page at 0xffffffffff600000, which is outside any possible guest_base-relative mapping and is handled by trapping execution there and emulating the three vsyscall entry points.

vDSOs are prebuilt guest shared objects checked into QEMU for aarch64 (little and big endian), arm (le, be8 and be32), hppa, i386, x86_64, loongarch64, ppc (32-bit, 64-bit big endian, 64-bit little endian), riscv (32 and 64), s390x, and sh4 (both endians, new in 11.1). They exist mainly to provide the signal return trampoline (so unwinders recognize signal frames) and fast clock_gettime/gettimeofday entry points that simply execute a syscall under emulation. The gen-vdso tool (linux-user/gen-vdso.c) turns each .so into a C array with relocation information. ruvm uses the same .so files, byte for byte, embedded with include_bytes!, and ports gen-vdso's relocation processing, because debuggers and libunwind identify the sigreturn trampoline by symbol name and exact instruction bytes.

An improvement ruvm makes here: clock_gettime in the vDSO can be emulated without a host syscall. The JIT recognizes the vDSO's syscall instruction at a known offset and replaces it with a direct call into a helper that reads the host's clock through the host vDSO. This preserves guest-visible behaviour (same results, same errno on bad clock IDs) and removes a host syscall from a path that some workloads call millions of times per second. It is enabled by default because it is not observable except in strace output, and -strace disables it so traces match QEMU.

## Guest address space

### guest_base, reserved_va and 32-on-64

A guest virtual address g maps to host address g + guest_base (g2h in QEMU's include/user/guest-host.h). The JIT emits this as a base register plus offset on the fast path, with guest_base held in a reserved host register when it is non-zero, or folded away when it is zero. Choosing guest_base well is what makes user mode fast: there is no softmmu TLB in user mode, every guest load is a host load at guest_base + g.

reserved_va is the size of guest address space reserved up front. QEMU's rule (linux-user/main.c, MAX_RESERVED_VA) is: when the host has more address bits than the guest, a 32-bit guest (or a 64-bit CPU running a 32-bit ABI such as mipsn32 or sparc32plus) reserves the full 4 GiB (0xffffffff), and other guests reserve up to their TARGET_VIRT_ADDR_SPACE_BITS but only when the user asks with -R, because reserving huge ranges causes rlimit and OOM accounting trouble. When a reservation exists, TASK_UNMAPPED_BASE becomes a third of it and ELF_ET_DYN_BASE two thirds, again as in main.c. ruvm uses the same values because they are guest-visible: programs that print their mapping addresses, and sanitizer runtimes that assume shadow memory layouts, see the same numbers under both emulators.

guest_base selection follows common-user/probe-guest-base.c and linux_probe_guest_base in elfload.c: if -B is given use it; otherwise try guest_base 0 when the image range and commpage are free in the host address space; otherwise search for a hole large enough for the reservation (or for the image range plus commpage when there is no reservation), checking against /proc/self/maps of the emulator itself. ruvm keeps this algorithm but changes one detail: its own heap, JIT code buffer and thread stacks are placed above the chosen guest window by allocating them after the probe, using an arena reserved at startup, so a later large guest mmap never collides with emulator allocations. QEMU mitigates the same collision by reserving the whole guest range when reserved_va is set; ruvm does that too and adds the arena for the 64-bit guest case where no reservation exists.

64-bit guests on 64-bit hosts with no reservation and guest_base 0 is the common case for x86_64 on aarch64 and the reverse. Address space differences still exist: x86_64 user space ends at 2^47, aarch64 Linux kernels are configured with 39, 42, 47, 48 or 52 bit user address spaces, and Apple Silicon Linux kernels (Asahi) use 16K pages. When the host VA is smaller than the guest expects, mmap hint addresses beyond the host range fail and ruvm returns the same ENOMEM QEMU returns.

### Page flags

User mode tracks per-guest-page flags (PAGE_READ, PAGE_WRITE, PAGE_EXEC, PAGE_VALID, PAGE_WRITE_ORG, PAGE_ANON, PAGE_PASSTHROUGH and target bits such as PAGE_TARGET_1 for Arm BTI guarded pages or MTE) with page_set_flags. QEMU stores these in an interval tree. PAGE_WRITE_ORG records that a page was writable before the JIT write-protected it to detect self-modifying code; a host SIGSEGV on such a page invalidates the translations for that page and restores write permission.

ruvm keeps the same flag semantics in an interval map (a B-tree keyed by start address, with range split and merge), protected by the mmap lock, with an RCU-published snapshot for lock-free lookup by the JIT on translation (the JIT needs PAGE_EXEC and target bits when translating, and should not take a lock for it). This is the user-mode counterpart of the FlatView RCU scheme in document 05.

## mmap emulation and page size mismatch

The guest believes in its page size; the host enforces its own. linux-user/mmap.c handles three cases, and since the 9.x rewrite it names them explicitly: mmap_h_eq_g (host page size equals guest), mmap_h_lt_g (host smaller than guest) and mmap_h_gt_g (host larger than guest). The third is the case that matters now, because the two most interesting Arm hosts run large pages: Apple Silicon Linux with 16K pages and many Arm servers (and distribution kernels for POWER) with 64K pages, running x86_64 guests that assume 4K.

### Choosing the guest page size

Some targets allow a variable page size in user mode. In 11.1.0 target/*/cpu-param.h defines TARGET_PAGE_BITS_VARY under CONFIG_USER_ONLY for aarch64 on Linux, alpha, loongarch and ppc, and main.c calls set_preferred_target_page_bits(ctz32(host_page_size)) before creating the CPU, so for these guests the guest page size simply becomes the host's and there is no mismatch. That is legitimate because those architectures allow several page sizes and their ABIs tell programs to ask (getpagesize, AT_PAGESZ). x86, 32-bit Arm, RISC-V, s390x and the rest have a fixed 4K (or 8K for sparc64 and alpha system mode) guest page, and the mismatch must be emulated. ruvm uses the identical rule, including the choice of which targets vary, because AT_PAGESZ is guest visible.

### Host smaller than guest

Rare on modern hosts (for example an 8K-page guest on a 4K host). mmap_h_lt_g over-allocates anonymous memory, aligns it to the guest page, and maps the file on top. The subtle part is a file mapping that ends mid guest page: the guest expects the rest of that guest page to be accessible (zero), the host would deliver SIGBUS beyond EOF, so QEMU truncates the file mapping at EOF and backs the remainder with anonymous memory. The comment in mmap.c spells out the limitation: if the file later grows, the anonymous tail does not reflect the new file contents. ruvm keeps this behaviour and the limitation.

### Host larger than guest

mmap_h_gt_g is the hard case. Consider a 4K x86_64 guest on a 16K host:

- A mapping whose start and length are 4K aligned but not 16K aligned covers partial host pages at one or both ends. For each partial host page, mmap_frag either finds it unused and maps a fresh anonymous host page, or finds other guest pages already in it and must merge. For file mappings the partial page is filled by reading the file (pread) into anonymous memory, because a host file mapping cannot start at a non-16K-aligned file offset or cover only part of a host page.
- A file mapping whose file offset and address are misaligned relative to each other modulo the host page size cannot be represented by a host file mapping at all. For MAP_PRIVATE QEMU falls back to an anonymous mapping filled by pread; for MAP_SHARED it fails with EINVAL, since a copy cannot keep shared semantics. This is a real incompatibility visible to guests (a MAP_SHARED of a file at offset 4096 fails), and ruvm has the same limitation unless the improvement below applies.
- mprotect on a guest page range that shares a host page with other guest pages must set the host protection to the union of the guest protections in that host page (target_mprotect computes this per host page), so a guest PROT_NONE guard page inside a 16K host page is not actually inaccessible. Accesses to it succeed on the host instead of faulting. QEMU checks page flags on some paths (for example when the JIT translates code, and in access_ok for syscall buffers), but plain guest loads and stores to such a guard page are not caught. Stack guard pages and sanitizer redzones are the typical victims.
- munmap of part of a host page must keep the host page if other guest pages in it are still mapped.

### What ruvm changes

ruvm implements the same three paths with the same guest-visible results, then adds two optional mechanisms, both off by default for strict compatibility and on by default in the `ruvm` native CLI:

1. Guard page precision. For a host page that holds guest pages with different protections (a "mixed" page), ruvm sets the host protection to the intersection instead of the union, so every access to the host page faults. The SIGSEGV handler looks up the guest page flags: if the guest page really forbids the access, the guest gets its SIGSEGV with the correct si_addr; if it permits it, the handler retranslates the faulting block in checked mode, where accesses to that host page go through a slow path that checks the guest flags and then performs the access through a second host mapping of the same memory (anonymous memory for user mode is memfd-backed for exactly this reason). Only blocks that touched a mixed page pay the cost. Programs that rely on guard pages (Go and Java runtimes with stack guards, ASan) then get their faults.
2. Shared misaligned file mappings. When the host supports it (Linux with memfd and userfaultfd, see ruvm-sys), a MAP_SHARED mapping at a misaligned offset can be served by a userfaultfd-backed region that copies file contents on fault and writes back on msync and munmap. This is not the same as real shared semantics with other processes, so it is only enabled with `-x-ruvm-mmap-shared-emul=on` and never under the qemu-* names.

These are recorded as new decisions for document 25.

### Locking

All mmap, munmap, mprotect, mremap, brk and shmat operations take the process-wide mmap lock (mmap_lock in QEMU), which also protects the page flags tree. The JIT's translation path takes it for reading only via the RCU snapshot described above, and SIGSEGV handling for self-modifying code takes it briefly for writing. ruvm's lock is a reader-writer lock with writer preference; QEMU's is a recursive mutex, and recursion is replaced by explicit "already held" parameters.

## System call translation

### Tables generated per target

Syscall numbers come from the Linux kernel's syscall.tbl files, imported into linux-user/<arch>/ (syscall_64.tbl for aarch64 and x86_64, syscall_32.tbl for i386, syscall_o32.tbl, syscall_n32.tbl and syscall_n64.tbl for MIPS, syscall.tbl for the rest) and turned into syscall_nr.h by syscallhdr.sh. scripts/update-syscalltbl.sh refreshes them from a kernel tree. ruvm uses the same .tbl files and a build script that emits, per target, a dense Rust array indexed by guest syscall number with an entry naming a generic handler plus a per-target ABI adapter. do_syscall1 in QEMU is a 388-case switch with #ifdef TARGET_NR_ guards that decide which syscalls exist for which target; the generated table replaces the #ifdefs with data, and the "does this target have syscall X" question becomes a table lookup that tests can enumerate. Unknown numbers return ENOSYS and log "Unsupported syscall: N" under -d unimp, as QEMU does.

Architecture-specific entry conventions are handled in the adapter: 64-bit arguments split across register pairs with alignment padding on 32-bit ABIs (arm EABI, mips o32, ppc32; regpairs_aligned in QEMU), the mips o32 convention of passing arguments 5 to 8 on the stack, the s390x and x86 old_mmap argument block, the socketcall and ipc multiplexers on targets that have them, and the separate error return conventions (MIPS a3 flag, sparc carry flag, ppc CR0.SO, alpha a3).

### Struct conversion

Guest structs differ from host structs in size, alignment, field order and endianness. QEMU describes many of them in linux-user/syscall_types.h with STRUCT() macros consumed by the thunk machinery (thunk.c, thunk_convert) and handles the rest with hand-written target_ struct definitions in syscall_defs.h and per-target headers (target_structs.h, target_fcntl.h, termbits.h, sockbits.h). ruvm replaces both with one mechanism: a derive macro on paired struct definitions.

```rust
#[derive(GuestStruct)]
#[guest(abi = "target")]          // field types resolved per target ABI at build time
pub struct TargetStat64 {
    pub st_dev: AbiU64,
    pub st_ino: AbiU64,
    pub st_mode: AbiU32,
    #[guest(pad = 4, only(target = "arm"))]
    pub _pad0: (),
    // ...
}
```

The derive generates to_host and from_host conversions that byte-swap when the guest and host endianness differ, and a layout assertion test per target that compares size and field offsets against numbers extracted from the real target C headers (compiled once with cross compilers in CI, stored as JSON). Generic ABI types (AbiLong, AbiUlong, AbiPtr) take their width from the target's ABI, which handles 32-bit ABIs on 64-bit CPUs such as mipsn32 and sparc32plus.

Guest pointers are never dereferenced raw. Handlers use `GuestSlice` and `GuestRef` types obtained from lock_user checks (access_ok against page flags, as QEMU's lock_user and lock_user_struct do), which return EFAULT for unmapped or protected ranges before the host syscall runs. On a host with no endianness difference and matching layout, the conversion compiles to a pointer cast and no copy, which is the common case for x86_64 guests on aarch64 hosts for most structs.

### ioctls

linux-user/ioctls.h lists 524 IOCTL entries in 11.1.0, each mapping a target ioctl number to a host number and a thunk type describing the argument (terminal, block, filesystem, sound, fb, dm, rtc, usbdevfs, drm, kd, vt, loop, fiemap, and others), with IOCTL_SPECIAL entries for those needing custom code. ruvm imports the same table with a script into a generated Rust table, with target ioctl numbers recomputed per target from the _IOC direction and size encoding (which differs between architectures: mips, ppc, sparc and alpha use a different _IOC layout). An ioctl number not in the table is logged as "Unsupported ioctl: cmd=0x..." under -d unimp and returns ENOTTY, as in QEMU's do_ioctl; ruvm returns the same, and a test walks the table to check every entry's number and argument conversion per target.

### File descriptor translators

Some file descriptors carry data whose layout depends on the architecture: netlink sockets (route and audit), packet sockets, signalfd, eventfd, timerfd and inotify. QEMU registers a TargetFdTrans per fd in fd-trans.c that converts data on read, write, recvmsg and sendmsg. ruvm has the same set of translators attached to its fd table, and, like QEMU, does not implement io_uring for guests: syscall.c has no io_uring handling in 11.1.0, so io_uring_setup falls through to the "Unsupported syscall" path and returns ENOSYS. Shared ring memory holding guest-layout submission entries cannot be passed to the host kernel without a translating proxy, so ruvm does not add one.

### Interruptible syscalls

A blocking host syscall must not lose a guest signal that arrives between the pending-signal check and entering the kernel. QEMU solves this with safe_syscall (common-user/host/<host>/safe-syscall.inc.S, provided for aarch64, loongarch64, ppc64, riscv64, s390x, sparc64 and x86_64 hosts): a small assembly routine that checks the pending flag and issues the syscall, with its start and end addresses exported. If a host signal lands with the PC between safe_syscall_start and the syscall instruction, the host signal handler rewinds the PC to safe_syscall_start (rewind_if_in_safe_syscall in linux-user/signal.c), so the check runs again and the syscall returns TARGET_ERESTARTSYS. ruvm keeps this design exactly, with the assembly written as naked Rust functions per host in ruvm-user-common. It is the only correct approach without kernel help, and the same file serves bsd-user.

## Signals

### Host side

The emulator installs host handlers for every signal. A host signal is either synchronous and caused by guest code (SIGSEGV or SIGBUS from a guest memory access, SIGILL or SIGTRAP from JIT-generated traps), synchronous and caused by the emulator (a write to a page the JIT write-protected for self-modifying code detection, which is handled internally and never reaches the guest), or asynchronous (kill, timers, SIGCHLD). For synchronous faults in JIT code, QEMU uses the host PC to find the translation block, restores guest state with cpu_restore_state and raises the guest exception. ruvm does the same through the side tables described in document 08, which map a host PC to the guest PC and the per-instruction state words, with no retranslation.

Asynchronous signals are queued per thread and the vCPU is kicked out of the JIT (QEMU sets cpu->exit_request and uses the icount decrementer trick; ruvm uses the same exit flag checked at block entry). Delivery happens at the next block boundary, in process_pending_signals, which builds the guest signal frame.

### Signal numbering

Host and guest signal numbers differ across architectures (MIPS, SPARC, Alpha and HPPA use their own numbering), and the real-time range is the hard part. glibc on the host reserves the lowest real-time signals for its own use, so the host SIGRTMIN is usually 34, not 32. QEMU's default mapping (linux-user/signal.c) starts at host SIGRTMIN + 2, reserving two host real-time signals for internal use, and maps guest TARGET_SIGRTMIN upward until host real-time signals run out; guest signals that cannot be mapped are silently ignored in sigaction. It also remaps guest SIGABRT to a host real-time signal so that a guest abort can be distinguished from a host abort, mapping back to a real SIGABRT for the core dump. The -t option (QEMU_RTSIG_MAP) lets users specify the mapping as tsig hsig count triples. ruvm implements the same default and the same option syntax. The comment in QEMU says the proper fix would be manual delivery multiplexed over one host signal; ruvm does not do that in the compatible mode, since the resulting numbering would differ from QEMU's for programs that probe their signal limits.

### Frames and sigreturn per architecture

Each target defines its signal frame layout in linux-user/<arch>/signal.c (setup_frame, setup_rt_frame, do_sigreturn, do_rt_sigreturn). The frame must be byte-exact with the Linux kernel's layout for that architecture because guest programs read it: unwinders, debuggers, garbage collectors that scan ucontext, and programs that modify the saved context and return (user-level threading, JITs, and fault-handling runtimes such as the JVM and Wine). Notable per-architecture content:

- x86_64 and i386: sigcontext plus the FXSAVE area, with the XSAVE extension marked by FP_XSTATE_MAGIC1 and FP_XSTATE_MAGIC2 so that AVX state is saved and restored; i386 has both the legacy frame and rt frame, and vm86 support in linux-user/vm86.c.
- aarch64: a list of tagged records after the general registers: FPSIMD, ESR, SVE, ZA, ZT, TPIDR2, FPMR, GCS and the EXTRA record for frames too big for the fixed area (TARGET_*_MAGIC values in linux-user/aarch64/signal.c). Record order and sizes depend on the configured SVE and SME vector lengths.
- arm: the VFP coprocessor record (TARGET_VFP_MAGIC) in the rt frame, and the sigreturn trampoline in the vDSO or, without it, on the stack.
- ppc and ppc64: the ELFv1 and ELFv2 frames with AltiVec and VSX state, and the 32-bit signal frame with its trampoline.
- riscv: general and FP registers with fcsr. linux-user/riscv/signal.c in 11.1.0 has no vector state record, so RVV state is not saved across guest signal handlers. ruvm matches QEMU here (adding the kernel's vector context record changes the frame size) and lists the gap in document 25 so both projects can close it together.
- s390x, mips (o32, n32, n64), sparc (32 and 64 with register window flush), alpha, hppa, m68k, sh4, microblaze, or1k, xtensa, loongarch64 (with LSX and LASX extended contexts) and hexagon, each with their own layouts.

ruvm writes these frames with the GuestStruct derive, and each frame type has a layout test against the kernel's struct definitions for that architecture (extracted from kernel headers by the same CI job that extracts syscall struct layouts). sigaltstack, SA_ONSTACK, SA_RESTART with the ERESTARTSYS path, SA_SIGINFO siginfo conversion (including si_code values that differ per architecture), and the QEMU_ESIGRETURN convention (sigreturn must not clobber the return register) are handled in the generic layer.

## Threads, fork, exec and futex

A guest thread is a host thread. clone with the CLONE_THREAD flag set (CLONE_VM, CLONE_FS, CLONE_FILES, CLONE_SIGHAND, CLONE_THREAD, CLONE_SYSVSEM) creates a new host pthread with its own vCPU state copied from the parent; the new thread starts in the JIT loop with the child's return value set. clone without CLONE_VM is a host fork. CLONE_VFORK is special cased in do_fork: QEMU clears CLONE_VFORK and CLONE_VM and performs a normal fork, since the emulator cannot share its own state with a vfork child. ruvm does the same. clone3 is not implemented by QEMU 11.1.0 (it falls through to ENOSYS and glibc falls back to clone), and ruvm matches that until upstream adds it.

After fork the child must reset emulator state that is not fork-safe: other threads' vCPUs are gone, the TB cache lock and the mmap lock must be reinitialized, and RCU state must be reset. QEMU does this in fork_start and fork_end. ruvm uses pthread_atfork style hooks in each subsystem that owns a lock, registered through a linkme distributed slice so no central list is needed. execve of another binary goes through the host execve. If the target binary is a foreign-architecture binary, the host kernel invokes the binfmt_misc interpreter again, with the P flag behaviour preserved. QEMU also intercepts execve and execveat of /proc/self/exe and substitutes real_exec_path, and ruvm does too.

futex calls pass guest addresses through g2h to the host futex (do_futex in syscall.c), which works because the guest's futex word is a host word at a translated address; values are compared in host byte order, so cross-endian guests byte-swap the expected value (tswap32) before the host call, and FUTEX_WAKE_OP and FUTEX_CMP_REQUEUE convert their operands. futex_time64 on 32-bit guests converts the 64-bit timespec. The robust futex list (set_robust_list, get_robust_list) is not supported: QEMU returns ENOSYS because the kernel would walk a list in guest layout at thread death, and guest glibc falls back to non-robust mutexes. ruvm matches, and could emulate robust lists by walking the list itself at guest thread exit, but that changes behaviour visible to glibc's feature detection, so it is recorded as a possible future extension only. rseq registration likewise is not implemented by QEMU; glibc handles the ENOSYS.

Memory ordering between guest threads is the JIT's job (document 08): for x86 guests on Arm and RISC-V hosts, the strong-on-weak fence placement uses the Risotto and Arancini mappings from the canon; guest atomic instructions become host atomics on the translated address, and LL/SC guests (Arm, RISC-V, PowerPC, MIPS) use the same compare-and-swap emulation as QEMU's exclusive monitor, with the same known ABA weakness.

## /proc emulation

Some /proc files describe the emulator, not the guest, and QEMU fakes them. The list in syscall.c's fake_open table in 11.1.0 is: /proc/self/maps, /proc/self/smaps, /proc/self/stat, /proc/self/auxv, /proc/self/cmdline (each also matched as /proc/<own pid>/...), /proc/net/route (only when host and guest endianness differ, to byte-swap the addresses), /proc/cpuinfo for alpha, arm, hppa, loongarch64, m68k, ppc, riscv, s390x and sparc (per-target open_cpuinfo in target_proc.h; the ppc, loongarch and m68k versions are new in 11.1), and /proc/hardware for m68k. /proc/self/exe readlink and open return the guest binary (or the AT_EXECFD file). The maps output lists guest mappings with guest addresses, hides emulator mappings, and labels the stack and vDSO as [stack] and [vdso]; smaps adds the per-mapping fields with plausible values.

ruvm implements the same set with the same output formats, generated from the page flags tree for maps and smaps. The /proc/cpuinfo contents follow each target_proc.h exactly (for example, the riscv version prints the ISA string from the CPU model and the mmu type); programs such as build systems and runtime CPU detection code parse these, so differences break things. QEMU calls realpath on the opened name before matching (do_guest_openat), so /proc/self, /proc/<pid> and paths with redundant components all hit the same fake; ruvm keeps this order of operations, including the quirk that the realpath runs on the host view of the path, which matters when -L redirects are in play.

## Debugging and observability

### strace

-strace (or QEMU_STRACE) prints each guest syscall with decoded arguments and the result. QEMU drives it from linux-user/strace.list (1,141 TARGET_NR_ references in 11.1.0), where each entry names a print function for arguments and one for the return value, and from linux-user/strace.c, which holds the flag tables for open, mmap, clone and so on. The output format is scripted against by users, and the QEMU tcg tests compare some of it, so ruvm reproduces the format character for character. In ruvm the strace printers are generated from the same syscall table source as the dispatcher (the table from the section on system call translation carries an optional printer attribute), which removes the class of bug where strace.list and syscall.c disagree about an argument's type. When -strace is active, the vDSO fast paths are disabled so that clock_gettime and friends appear as syscalls, which matches what QEMU shows because QEMU's vDSO images always make the real syscall.

### gdbstub

-g port (QEMU_GDB) starts the guest stopped and waits for gdb on a TCP port or, with a path, a Unix socket. The user-mode gdbstub (gdbstub/user.c and gdbstub/user-target.c) adds features that system mode lacks: qXfer:auxv:read, qXfer:siginfo:read, qXfer:exec-file:read, the vFile host I/O packets so gdb can read the guest's binary and libraries through the stub, catch syscall support (QCatchSyscalls), fork following, and the /proc/<pid>/maps view that gdb's info proc mappings reads. The multiarch gdbstub tests in tests/tcg/multiarch/gdbstub/ exercise these: catch-syscalls.py, follow-fork-mode-child.py, follow-fork-mode-parent.py, interrupt.py, late-attach.py, memory.py, prot-none.py, registers.py, sha1.py, test-proc-mappings.py, test-qxfer-auxv-read.py, test-qxfer-siginfo-read.py and test-thread-breakpoint.py. ruvm runs all of them. The register descriptions come from each target's GuestArch::gdb_features, using the same XML files as QEMU (document 09). late-attach.py covers attaching to an already running process, which QEMU supports in user mode through the -g option's suspend=n form; ruvm implements the same syntax.

### Plugins, perf and logging

TCG plugins work in user mode with the same API as system mode (document 20): ruvm-plugin exposes the qemu-plugin.h interface at QEMU_PLUGIN_VERSION 7. The syscall callbacks (qemu_plugin_register_vcpu_syscall_cb and the _ret variant) matter more here than in system mode, and version 6 added qemu_plugin_register_vcpu_syscall_filter_cb, which lets a plugin skip a syscall and supply its return value. ruvm calls the filter before the dispatcher and after argument capture, exactly where QEMU calls it, so a plugin that fakes syscalls sees the same argument values. The tests/tcg/plugins directory and the contrib plugins run against ruvm in CI.

-perfmap writes /tmp/perf-<pid>.map and -jitdump writes a jitdump file for perf inject (tcg/perf.c). Both map host code addresses back to guest symbols, which is how people profile guest code under emulation. ruvm writes both formats, and because its JIT tiers replace code more often than TCG does, it emits jitdump JIT_CODE_MOVE records when code is relocated and a fresh JIT_CODE_LOAD record each time a region is retranslated. -d, -D and -dfilter follow document 22's logging model; the log item names (in_asm, op, out_asm, exec, cpu, page, strace and so on) are the same as QEMU's.

## bsd-user

bsd-user in 11.1.0 supports FreeBSD hosts running FreeBSD guests of the same OS, for five guest architectures (aarch64, arm, i386, x86_64 and riscv64). The layout splits BSD-generic code (bsd-user/bsd-mem.c, bsd-proc.c, bsd-misc.c, bsd-ioctl.c, bsdload.c, elfload.c, mmap.c, signal.c, strace.c) from FreeBSD specifics in bsd-user/freebsd/ (os-syscall.c as the dispatcher, os-proc.c, os-stat.c, os-sys.c, the os-ioctl-*.h tables, and target_os_*.h headers for signal, ucontext, stack, thread and vmparam layouts), with per-architecture directories under bsd-user/<arch>/. Much of the FreeBSD support lives in the out-of-tree qemu-bsd-user repository and is upstreamed in stages; ruvm tracks what is in the 11.1 tree and treats the out-of-tree code as a reference for layouts only, since the canon's compatibility target is the release.

ruvm-bsd-user shares ruvm-user-common with ruvm-linux-user: the loader core, guest address space and page flags, mmap engine, safe_syscall, the JIT loop, gdbstub and plugins. What differs is the syscall table (generated from FreeBSD's sys/kern/syscalls.master rather than Linux's syscall.tbl files), the signal frame layouts (ucontext and mcontext from target_os_ucontext.h and the arch headers), the sysctl emulation in os-sys.c (kern.proc, hw.pagesize, hw.machine and friends, which FreeBSD programs use the way Linux programs use /proc), and thread creation via thr_new. FreeBSD's umtx operations (_umtx_op) take the place of futex and are mapped onto host umtx with address translation, the same approach as futex. Porting bsd-user to NetBSD or OpenBSD hosts is out of scope, as it is for QEMU 11.1.

## Performance and the other translators

The canon's target for user mode is at least twice QEMU's throughput on SPEC CPU2017 intrate (document 21). User mode is where that is easiest to reach, because there is no softmmu TLB: a guest load is a host load at guest_base plus the address, and ruvm can use a register pinned to guest_base or, when guest_base is zero, no offset at all. The gains come from the JIT tiers in documents 07 and 08 (region formation, register allocation across blocks, flag liveness), not from anything user-mode specific. The syscall path itself is rarely hot; the exceptions are programs that make many small read and write calls and clock_gettime-heavy loops, which the vDSO fast path covers.

Other user-mode translators set a higher bar on specific pairs. FEX translates x86 and x86-64 Linux binaries on Arm64 Linux. Box64 translates x86-64 Linux binaries on Arm64, RISC-V and LoongArch hosts. Rosetta 2 translates x86-64 macOS binaries on Apple silicon and relies on the hardware TSO mode of those cores, with a Linux variant for virtual machines. All three are narrower than ruvm: one or two guest ISAs, one host ISA family and in Rosetta's case one vendor's hardware. That lets them specialize, and part of their speed comes from something ruvm's compatible mode will not do, which is running host-native copies of guest libraries. FEX calls this library forwarding, using thunks that marshal calls from guest code into host builds of libraries such as OpenGL, Vulkan and the X11 client libraries (host-side libraries installed under /usr/lib/fex-emu/HostThunks). The Box64 README describes the same idea: calls into native host builds of libc, libm, SDL and OpenGL through wrapped library definitions. For graphics and games this matters more than JIT quality, because the frame time is spent in the driver stack.

### Native library thunking as an optional extension

ruvm will offer thunking as an extension outside QEMU compatibility. The rules are the canon's rules for extensions (document 20): off by default, named with the x-ruvm prefix, never enabled under the qemu-<arch> compatible binary names, and documented as changing behaviour. It is enabled with -x-ruvm-thunks=<config> on the ruvm-<arch> binaries only. The config lists guest library sonames to intercept and the thunk bundle to use for each. A thunk bundle has a guest-side shim library, loaded by the guest dynamic linker in place of the real library, whose exported functions trap into the emulator with a thunk number, and a host-side library that unpacks guest arguments, calls the host library and packs results. Bundles are generated from annotated C headers by a generator in the ruvm-thunkgen crate, the same model FEX uses, with manual code for callbacks (guest function pointers passed to host code, which need a reverse trampoline back into the JIT) and for structures whose layout differs between guest and host ABI.

The initial scope is x86-64 guests on aarch64 hosts with libGL, libEGL, libvulkan and libX11-family libraries, which is the pairing where users most want it and where FEX's experience shows the hard cases (callbacks, varargs, and 32-bit guests whose pointers the host library cannot hold). 32-bit guests are excluded at first. Thunking breaks some guarantees: plugins do not see instructions executed in host libraries, gdb cannot step into them, record and replay loses determinism for those calls, and -strace shows the host library's syscalls without guest translation. These limits are printed when the extension is enabled.

## Conformance plan

User-mode conformance has four layers, run in CI per guest architecture as described in document 22.

1. QEMU's own tests: the tests/tcg tree built with cross compilers for every guest that has a directory there (aarch64, aarch64_be, alpha, arm, hexagon, hppa, i386, loongarch64, m68k, mips, mips64, mips64el, or1k, ppc64, ppc64le, riscv64, s390x, sh4, tricore, x86_64, xtensa, xtensaeb) plus multiarch (signals.c, sigreturn-sigmask.c, test-mmap.c, vma-pthread.c, munmap-pthread.c and the rest), the gdbstub scripts and the plugin tests. Tricore has no linux-user binary; its tests are system-mode and run in document 11's machines. Pass state is recorded per test and any test that passes on QEMU 11.1.0 and fails on ruvm blocks a release.
2. The Linux Test Project syscalls suite run inside a guest root filesystem (Debian for most targets, Alpine for musl coverage) under ruvm and under QEMU 11.1.0, on the same host. The comparison is QEMU versus ruvm, not ruvm versus a native kernel, because many LTP tests fail under QEMU for known reasons (for example the robust futex and ptrace tests). The gate is that ruvm passes every test QEMU passes; tests only ruvm passes are logged, not gated, to keep behaviour aligned with QEMU.
3. Differential syscall traces: a set of workloads (package builds in chroots, test suites of Python, Perl and Go, a busybox shell session) run under both emulators with -strace, and the traces are compared after normalizing addresses, pids and timestamps. Differences in argument decoding, errno or call order are bugs unless a known list says otherwise. This catches mistakes in struct conversion that return plausible results.
4. Real-world builds: Debian and Alpine package builds in foreign-architecture chroots via binfmt_misc, the way distributions and container builders use qemu-user-static. A nightly job builds a fixed list of about 200 packages per tier 1 guest and compares build success and test suite results with QEMU runs.

For bsd-user the first two layers apply with FreeBSD's kyua test suite in place of LTP, running on a FreeBSD host in CI.
