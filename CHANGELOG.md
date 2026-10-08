# Changelog

Notable changes, newest first. This project is pre-1.0 and makes no compatibility promise about its own APIs until it has one. The compatibility it does promise is with QEMU 11.1, and each release says how much of that is real.

The minor version is the number of milestones finished. 0.1.0 is the release where M0 closes, 0.2.0 where M1 closes, and so on up to M12, which is 1.0. Patch releases come whenever enough has landed to be worth a tag. The milestones are the issues labeled `kind/milestone` at https://github.com/tamnd/ruvm/issues.

## Unreleased

## 0.4.4

This patch is all M6. The riscv virt board gets AIA and ACPI, the riscv Linux boot is about 30% faster, and the arm virt board gets its PCIe host bridge.

For M6: riscv virt has the APLIC and IMSIC interrupt controllers with `-M virt,aia=aplic|aplic-imsic,aia-guests=N`, the Smaia and Ssaia CSRs, and guest external interrupts for the H extension (#167). The device tree matches QEMU 11.1 apart from rng-seed in every setup checked, and Linux boots on both APLIC modes. riscv virt also builds QEMU's ACPI tables, so EDK2 can boot Linux with `-drive if=pflash` and `acpi=on` (#168). Every table the guest sees has the same md5sum as under QEMU, with the PLIC and with `aia=aplic-imsic`. There is no SMBIOS or `-numa` yet. The TCG code buffer is now sized as in QEMU, and register allocation caches its constraint sets, which takes the riscv boot to a shell from about 9 to 10 s down to about 6 to 7 s, against 5.3 to 5.9 s for QEMU (#169). An x86 TCG boot is about 20% faster too. arm virt has QEMU's generic PCIe host bridge at QEMU 11.1's addresses, with all six `highmem*` properties, `-device virtio-{blk,rng,serial}-pci` and `-drive if=virtio` (#171). The guest sees the same config space as under QEMU for every function. The ITS, the PL061 and virtio-net-pci on arm are not in yet.

## 0.4.3

This patch finishes device state for migration and fills in more of the riscv virt board.

For M5: virtio devices now migrate in QEMU 11.1's layout, covering blk, net, scsi with its disks, rng, balloon and serial with the console port, over both the pci and mmio transports (#164). Dirty logging stays on for the whole migration so that device writes to guest RAM reach the other side, and option ROMs migrate as RAM blocks. The `cpu/msr_smi_count` subsection that QEMU sends on q35 is handled now. A guest that keeps virtio disks and NICs busy hopped QEMU to ruvm to QEMU on q35 and microvm, and ruvm to QEMU to ruvm on microvm, with no bad output. With this, every M2 device's VMState is matched to QEMU field by field. `-netdev` with hubs is wired up too.

For M6: riscv virt has QEMU's generic PCIe host bridge at the same addresses as QEMU, and virtio blk, rng and serial work behind it as pci devices (#163). The config space of every function is byte for byte the same as QEMU's. `-cpu max` and eleven named riscv64 CPUs, including the rva22 and rva23 profiles, now come from a port of QEMU's CPU config, and their device trees match QEMU's (#165). That brought in scalar crypto, the PMU with Sscofpmf, the indirect CSRs, counter delegation and pointer masking. `-cpu max` still lacks AIA and seven other extensions.

## 0.4.2

This patch moves M5 and M6 forward. A q35 or microvm guest now carries its device state across a migration hop, and the RISC-V guest has the vector extension and the H extension.

For M5: the q35 and microvm devices now save and load their real state instead of being read and dropped, matched field by field to QEMU 11.1 (#160). On q35 that is the APIC, PIC, IOAPIC, RTC, PIT, HPET, the ICH9 LPC, PM, SMBus and SATA devices, fw_cfg, the serial ports, the TSC, mch, the PCI host and bus, the keyboard controller, port92 and the rest of QEMU's sections, in QEMU's order. microvm also gained the acpi-ged section. A guest driven by timer interrupts survived every hop chain between QEMU and ruvm on q35 and microvm with no bad output. The pc-q35-11.0 and pc-q35-10.2 machine types are in too. Virtio devices don't migrate yet.

For M6: the riscv guest has V, the Zve subsets, Zvfh and the bf16 vector extensions, and the full vector crypto set (#161). V is off by default as in QEMU 11.1 and `-cpu rv64,v=on` turns it on. The H extension is on by default, with two-stage translation, the hypervisor CSRs, HLV, HLVX and HSV, virtual instruction faults and VS interrupt routing. A bare metal H test prints the same 251 lines on ruvm and QEMU, and Linux reports `rv64imafdch`. Guest external interrupts and AIA are not in yet.

## 0.4.1

This patch starts M5 and M6. ruvm can live migrate a guest to and from QEMU 11.1, and it has a RISC-V guest that boots Linux.

For M5: ruvm reads and writes the QEMU 11.1 migration stream and does live precopy over the tcp, unix, fd, exec and file channels, with `migrate`, `migrate-incoming`, `query-migrate` and the other migration QMP commands, plus `-incoming` (#156). A q35 TCG guest that keeps rewriting its memory migrated QEMU to ruvm, ruvm to QEMU and ruvm to ruvm with its checksum intact. Only the vCPUs and RAM move for real so far. The other device sections are read and dropped, so a guest that depends on interrupt controller state won't survive a hop yet.

For M6: there is a RISC-V guest on TCG with RV64GC, M, S and U modes, Sv39/48/57 paging, PMP and the Zb* extensions, and a riscv `-M virt` board with OpenSBI loading and a device tree (#157). QEMU's five riscv64 tests/tcg system tests pass. A Debian 6.12 riscv64 kernel boots to a shell at 1, 2 and 4 vCPUs on both x86-64 and aarch64 hosts, after a fix for a lock order deadlock with more than one vCPU (#158). RVV and the H extension are not in yet.

## 0.4.0

M4 is done, so this is the first 0.4 release. The JIT runs real guests, and on CoreMark it is faster than QEMU 11.1's TCG on all three host and guest pairs we measure.

For M4: QEMU 11.1's tests/tcg system tests for x86_64 and aarch64 print exactly what QEMU prints. risu matches hardware for every aarch64 group, and for the x86 groups except where QEMU itself differs from hardware (#149, #151). Linux boots to a shell under TCG on q35 and arm virt, on an x86-64 host and on an Apple M4. CoreMark runs at 1.32x QEMU's speed for an x86-64 guest on an x86-64 host, 1.37x to 1.46x for an aarch64 guest on an x86-64 host, and 1.43x for an x86-64 guest on an Apple M4. Since 0.3.6, a DIV side exit, quotient reuse for remainders, and loop back edge chaining bring bench.c g_s4 on the M4 from 1.24x to 1.30x (#153).

The M4 exit criterion asks for SPEC CPU2017 intrate, which we have not run, because we don't have a license yet. Those numbers are tracked in #154. The x86-64 guest has what Windows checks for in `-cpu max`, but no Windows guest has booted under TCG yet. Windows on q35 is still open under M2.

M5 has started, with the QEMU migration stream and precopy.

Known issues: `sigterm_from_another_process` in sys was flaky and is fixed in #153. tests/tcg user mode tests wait on linux-user in M7.

## 0.3.6

This patch makes the JIT faster than QEMU 11.1's TCG on bare metal CoreMark for all three host and guest pairs we measure, and finishes the x86-64 guest for M4.

CoreMark now runs at about 1.32x QEMU's speed for an x86-64 guest on an x86-64 host, 1.37x to 1.46x for an aarch64 guest on an x86-64 host, and 1.44x for an x86-64 guest on an Apple M4. The arm front end keeps guest registers in TCG globals and builds superblocks across short conditional branches (#147), and global stores are delayed to block exits (#148). The aarch64 backend gained inline caches, direct helper calls and out of line slow paths, and the register allocator's liveness pass is shared by both backends (#150). A leak that kept every finished vCPU alive is fixed, which brings the peak memory of the a64_simd tests from 3.5 GB to 54 MB (#148).

risu harnesses for both guests are in (#149). The aarch64 groups match hardware with no differences, and the x86 groups match except where QEMU 11.1 itself differs from hardware.

The x86-64 guest now has AES-NI, SHA, the SSE4.2 string compares, the AVX2 gathers, and protection keys (PKU and PKS). `-cpu max` CPUID matches QEMU 11.1 TCG exactly, and CMPXCHG and BSF/BSR edge cases now match QEMU. DIV and IDIV take an inline fast path when one host divide is enough (#151).

Linux boots to a shell under TCG on q35 and arm virt on an aarch64 host as well as an x86-64 one.

## 0.3.5

This patch is mostly about JIT speed. An x86-64 guest on an x86-64 host now runs bare metal CoreMark faster than QEMU 11.1's TCG, at about 1.32x its speed, where 0.3.4 was below 0.1x.

Guest atomics now run as host atomics on guest RAM, so MTTCG no longer loses a spinlock release to a plain store from another vCPU, and both boards are back on MTTCG by default like QEMU (#138, #142, #143). Both native backends chain blocks and look up indirect jumps without leaving generated code, with an inline softmmu TLB fast path (#138). The x86 front end builds superblocks across short conditional jumps, indirect jumps, calls and returns use per call site inline caches, and side effect free helpers are called directly (#144). LDP and STP take the TLB fast path, which brings the aarch64 guest on an x86-64 host from 0.49x to 0.72x QEMU's speed (#145).

Linux on q35 now boots to a shell in about 20 s of CPU time with one vCPU and 25 s with two (#143). An x86 memory map change now flushes every vCPU's TLB like QEMU, which fixes a SeaBIOS hang (#144).

On arm, every test in QEMU's tests/tcg/aarch64/system runs and matches QEMU, EDK2 boots to the UEFI shell on virt, and `-cpu max` has FEAT_RNG (#142).

## 0.3.4

This patch is the first one where Linux boots under TCG, on both q35 and the arm virt board.

On q35, Linux boots to a busybox shell with one or two vCPUs (#136). That needed a local APIC ported from QEMU, with IPIs, MSIs, INIT and SIPI and the APIC timer, plus QEMU's interrupt order and triple fault reset in the x86 guest. The RTC on q35 and microvm no longer reports a date decades ahead, and q35 now stores the right century in CMOS.

On arm, `-M virt` is wired into the system emulator (#135), and the Debian bookworm 6.1 kernel boots to a busybox shell on two vCPUs. QEMU's tests/tcg aarch64 system tests hello, memory, memory-sve, interrupt, asid2, feat-xs and semiheap print what QEMU prints. The guest gained FEAT_TLBIOS, FEAT_XS, FEAT_TCR2 and FEAT_ASID2.

The x86 guest now runs x87, FXSAVE and XSAVE (#134).

Both boards run vCPUs round robin by default for now, because the atomic helpers are not yet safe against plain stores from other vCPUs. `thread=multi` still gives MTTCG. Boots are slow, about 15 to 20 minutes on a loaded server, since the native backends still return to Rust after every block. Both are being worked on next.

## 0.3.3

This patch is the first one where machines run guest code on the JIT. Linux does not boot under TCG yet, but QEMU's own system tests do.

`-accel tcg` now works (#131). The native aarch64 and x86-64 backends are wired into the runtime, vCPUs run on MTTCG or round robin threads, and q35 and microvm run on the x86 guest with `isa-debugcon` and `isa-debug-exit`. QEMU 11.1's tests/tcg x86_64 system tests (hello, interrupt and memory) print exactly what QEMU prints, both on an Apple M4 and on an EPYC without KVM.

There is a first arm `virt` board (#132) with GICv3, PL011, PL031, A64 semihosting, virtio-mmio, fw_cfg, PSCI and kernel loading. Its device tree matches QEMU's for four setups, and the tests/tcg aarch64 hello test runs on it.

The x86 guest now runs MMX, SSE through SSE4.2, AVX, AVX2, FMA and F16C through a table driven decoder generated from QEMU's sources, checked against 2107 native cases (#130). The aarch64 guest has the rest of SVE2 short of SVE2.1, plus BF16 and I8MM (#128).

Two more pieces of M4 are done. Strong guests on Arm hosts can use the fence mappings from Risotto and Arancini, with a litmus test on an Apple M4 that sees no forbidden x86-TSO outcome (#127). Plugins built against QEMU 11.1's `qemu-plugin.h` load without changes, and QEMU's own example plugins give the expected counts (#129).

M2 has not changed, since none of our machines has `/dev/kvm` right now.

## 0.3.2

This patch fills in most of the AArch64 SVE and SVE2 instruction set and the x86 general purpose extensions. Nothing boots on the JIT yet.

For AArch64 (`ruvm-target-arm`): SVE and SVE2 integer instructions, predicates, first fault and non fault loads, and the gather and scatter forms (#123). Then all SVE floating point, the SVE2 floating point pairwise and conversion forms, the SVE2 crypto instructions and FMMLA (#125). The `max` CPU now advertises SVE2 with AES, BitPerm, SHA3, SM4, F32MM and F64MM, and only those, because each of them has its instructions in. The SVE test data is generated by running the same code under QEMU and holds 5934 cases across three vector lengths.

For x86 (`ruvm-target-x86`): VEX prefix decoding, BMI1 and BMI2, ADCX and ADOX, MOVBE, CRC32, RDRAND and RDSEED, RDPID, XGETBV and XSETBV, the FS and GS base instructions, MOVNTI, and LDMXCSR and STMXCSR (#124). 494 cases match native runs on an AMD EPYC. x87, SSE and AVX still raise #UD.

M2 has not changed, since none of our machines has `/dev/kvm` right now.

## 0.3.1

This patch brings the first guest front ends for M4. ruvm can now translate and run x86 and AArch64 guest code on its JIT, although nothing boots on it yet.

For AArch64 (`ruvm-target-arm`): every A64 base integer instruction, the LSE atomics, the EL0 and EL1 system registers, the 4K page walk and exception entry through VBAR_EL1 (#118). Scalar FP and AdvSIMD go through `ruvm-softfloat`, along with AES, SHA1, SHA256 and PMULL. About 2150 test cases match an Apple M4 bit for bit (#119). EL2 and EL3 came after that, with VHE, stage 2 translation in all three granules, HVC and SMC routing, PSCI through a board hook, the generic timers and broadcast TLBI (#120).

For x86 (`ruvm-target-x86`): every integer instruction with lazy flags, string ops with REP, LOCK atomics, the 4 and 5 level page walk, exceptions through the IDT, SYSCALL and SYSRET, and the switch from real mode to long mode (#121). x87, SSE and AVX still raise #UD.

Fixes: two block tests that failed now and then turned out to be a real race. Deleting a node while another thread drained every node freed it late, on the wrong thread. `blockdev_del` now keeps drain-all sections out, and the job transaction tests hold the BQL like QEMU's do (#117).

## 0.3.0

M3 is done, so this is the first 0.3 release. The block layer and its tools now pass QEMU's own tests.

For M3: the last formats landed with the qcow2 driver, which has subclusters, external data files, compression, LUKS, persistent bitmaps and internal snapshots (#104). `qemu-nbd` and `qemu-storage-daemon` work, with NBD, vhost-user-blk, FUSE and VDUSE exports (#102, #103). `cargo xtask iotests` runs QEMU 11.1's tests/qemu-iotests suite against the ruvm tools. On Linux the quick and auto groups for qcow2, raw and nbd show no regressions against QEMU, and every test we still skip is listed with a reason (#112). Getting there fixed a long tail of output and behavior differences across qcow2, qed, raw, the filters and the tools. A new test makes each implementation run `qemu-img check` on images the other one wrote, for every format (#113). What is still skipped is tracked in #114.

M4 has started. `ruvm-decode` turns QEMU's decodetree files into Rust decoders (#105). `ruvm-jit-core` has the TCG IR, the optimizer and liveness, and `ruvm-jit-interp` is a portable interpreter that the tests use as a reference (#106). `ruvm-softfloat` is a bit exact port of QEMU's fpu (#107). There are two native backends, aarch64 (#108) and x86-64 with SSE2 through AVX2 (#111). They share a port of the TCG register allocator (#110), and both are checked against the interpreter on random blocks. `ruvm-jit` is the runtime: the TB cache, chaining, the softmmu TLB, `cpu_exec`, MTTCG and exclusive sections (#109). No guest front end runs on it yet, so this is all plumbing for now.

M2 has not changed. The boards still have not booted on real KVM, because none of our machines has `/dev/kvm` right now.

Known issues: two block unit tests are flaky when the suite runs in parallel (#115).

## 0.2.4

This patch is mostly storage. The M2 boards now have firmware, and most of the M3 block layer is in, along with qemu-img and qemu-io.

For M2: virtio-scsi with scsi-disk and scsi-cd, vhost-user and vhost kernel glue for devices, vhost-vsock and vhost-user-fs (#91). fw_cfg now produces the same bytes QEMU 11.1 does for the same command line, and we check that against dumps taken from QEMU. SMBIOS tables, pflash, `-smbios`, `-uuid` and chardev mux wiring came with it (#92). `cargo xtask boot-smoke` boots a kernel on each board and checks for the login banner, and CI has a job that runs it (#93). None of this has booted a guest on real KVM yet, because none of the machines we have access to right now has `/dev/kvm` and CI runners are backed up.

For M3: `ruvm-crypto` holds the ciphers, hashes and key derivation that LUKS needs, built on RustCrypto (#95). The block core has the node graph, permissions, drain, filters, throttle groups, the file and host_device protocols on io_uring, linux-aio or a thread pool, and the luks driver (#96). There is an NBD client, an NBD server and the nbd driver (#97). Block jobs, dirty bitmaps, fleecing backup and the vmdk, vdi, vhdx, vpc, qed, parallels, dmg, cloop, bochs, vvfat and old qcow formats came next (#99). qemu-img and qemu-io work through the ruvm binary, with golden tests against QEMU 11.1 output (#100).

Fixes: a race in the vhost-user test backend that showed up on Linux (#94), and block io tests that reopened a file while the old handle still held its OFD lock (#98).

## 0.2.3

With this patch, the command line can pick the microvm and q35 boards and run their vCPUs on KVM (#87). The boot tests need a guest kernel in `RUVM_TEST_KERNEL` and have not yet run on a host with KVM, so the boards should still be called unproven. Firmware, ACPI and a boot smoke test are next.

More devices: virtio-net and virtio-balloon (#81), PCIe root ports (#82), pvpanic, isa-debug-exit and the ICH9 SMBus with its EEPROMs (#83).

On the backend side, `ruvm-block` has the file protocol, the raw format, `BlockBackend` and `-drive` (#84). `ruvm-vhost` has the vhost-user frontend and the vhost kernel backend (#85), and it is MIT or Apache-2.0 licensed like the virtqueue crate. `ruvm-net` has the netdev core, hubs, tap, socket, stream and dgram (#86). User networking over libslirp, passt and vhost-user came later (#89). libslirp is loaded when the netdev is created, so building ruvm does not need it. `ruvm-chardev` gained the file, pipe, stdio, pty, ringbuf and mux backends (#88).

CI had a bug where every run on main shared one concurrency group, so a stuck run cancelled all the ones after it, and the KVM tests never ran. Each push to main now gets its own group. The Linux x86-64 job fails if `/dev/kvm` is missing, rather than quietly skipping the KVM tests (#80). The Windows build is fixed too.

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
