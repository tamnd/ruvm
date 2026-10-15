# qtest and tests/tcg status against QEMU

This note records how ruvm does on QEMU's own qtest suites for aarch64 and riscv64 and on QEMU's riscv64 tests/tcg, next to QEMU 11.1 running the same tests on the same host. It covers the third and fourth exit criteria of M6 (macOS, Windows, Arm, RISC-V), and it is updated as the failures below get fixed.

## Method

The tests come from the QEMU 11.1.0 source tree. QEMU itself was built from that tree with `--target-list=aarch64-softmmu,riscv64-softmmu --without-default-features --enable-fdt=internal --enable-tcg`, which also builds the qtest binaries. The host is server2, an x86_64 Linux machine.

Each qtest binary was run on its own, once with `QTEST_QEMU_BINARY` pointing at the QEMU build and once pointing at ruvm. libqtest takes the target from the binary name, so ruvm was reached through symlinks named `qemu-system-aarch64` and `qemu-system-riscv64`. The test list, environment and timeouts are the ones meson writes into `tests.json` for the `qtest-aarch64` and `qtest-riscv64` suites, and the counts below are read from the TAP output of each test.

The ruvm binary was a release build of main at #224 plus the human monitor change that follows it, except for netdev-socket and the riscv64 linux-user tests, which were run again with a release build that has #239. The qtest-riscv64 suite was run again with the change that runs riscv64 `virt` under the qtest accelerator, and the qtest-aarch64 suite with the change that does the same for aarch64 `virt`. Both suites were run again with the change that adds `system_reset` and the QOM device types, and again with the virtio-blk device types. qom-test, device-introspect-test and qos-test were run again with the virtio-net device types. Both suites were run again with `query-block` and HMP `info block`, and bios-tables-test and cdrom-test again with `snapshot=on`. numa-test was run again with `-numa` on Arm `virt`, and arm-cpu-features with `gic-version=max`.

A count is the number of TAP cases that passed. A test that registers no cases passes trivially with 0, and that is shown as 0 rather than as a pass, because it means the test found nothing in ruvm to run against.

## Criterion 3: a RISC-V Linux guest under TCG

On the x86_64 Linux host (server2), ruvm boots a Linux kernel and initramfs on the riscv64 `virt` machine under TCG. OpenSBI prints its banner at 0.93 s, the kernel banner appears at 1.33 s, the kernel runs `/init` at 12.36 s and the shell prompt is up at 13.78 s.

The riscv64 host was tested in #176 by running ruvm, built for riscv64gc Linux, under qemu-riscv64 8.2.2 on server3. There the riscv64 `virt` guest reached the kernel banner at 1080 s and a shell at 2446 s, and an x86 q35 guest reached a shell in about 570 s. That run has not been repeated for this note, because server3 has been too loaded to give a fair time and its old artifacts are gone. The numbers above are the ones from #176.

## qtest-aarch64

| Test | QEMU | ruvm | Why ruvm is short |
|---|---|---|---|
| bios-tables-test | 17 | 0 | `ras=on` is refused, and `virt` has no SMBIOS tables yet |
| qom-test | 90 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| device-introspect-test | 6 | 6 | |
| cdrom-test | 9 | 1 | the other 8 cases are 32-bit Arm boards ruvm does not have |
| migration-test | 12 | 6 | Counted with a release build. analyze-script, multifd/tcp/plain/zlib, multifd/tcp/plain/cancel, multifd/tcp/uri/plain/none, multifd/file/mapped-ram/dio and postcopy/plain pass. postcopy/recovery/plain fails because ruvm has no postcopy recovery, so `migrate-pause` is refused, and the test bails out there, so the cases after it do not run |
| boot-serial-test | 1 | 1 | |
| test-hmp | 90 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| qmp-cmd-test | 61 | 64 | ruvm also has the three TPM queries, see below |
| qos-test | 90 (5 skipped) | 16 (3 skipped) | runs the virtio-rng, virtio-blk, virtio-balloon and virtio-net cases only, see below |
| xlnx-canfd-test | 3 | 0 | no `xlnx-zcu102` board |
| xlnx-versal-trng-test | 5 | 0 | no `xlnx-versal-virt` board |
| bcm2835-dma-test | 1 | 0 | no Raspberry Pi boards |
| bcm2835-i2c-test | 3 | 0 | no Raspberry Pi boards |
| ast2700-gpio-test | 2 | 0 | no Aspeed boards |
| ast2700-hace-test | 11 | 0 | no Aspeed boards |
| ast2700-sgpio-test | 3 | 0 | no Aspeed boards |
| ast2700-smc-test | 8 | 0 | no Aspeed boards |
| npcm_gmac-test | 4 | 0 | no Nuvoton boards |
| iommu-smmuv3-test | 3 | 0 (3 skipped) | skipped: needs `arm-gicv3` among the device types |
| cxl-test | 1 | 0 | no `memory-backend-file` and no CXL |
| arm-cpu-features | 3 | 3 | |
| numa-test | 5 | 5 | |
| machine-none-test | 1 | 1 | |
| qmp-test | 9 | 9 | |
| readconfig-test | 1 | 1 | |
| netdev-socket | 10 | 10 | |
| Total | 449 | 129 | |

## qtest-riscv64

| Test | QEMU | ruvm | Why ruvm is short |
|---|---|---|---|
| bios-tables-test | 3 | 1 | numamem stops at `-numa` |
| qom-test | 13 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| device-introspect-test | 6 | 6 | |
| cdrom-test | 1 | 1 | |
| test-hmp | 13 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| qmp-cmd-test | 61 | 64 | ruvm also has the three TPM queries, see below |
| qos-test | 113 (5 skipped) | 27 (3 skipped) | runs the virtio-rng, virtio-blk, virtio-balloon and virtio-net cases only, see below |
| riscv-csr-test | 1 | 0 | no `veyron-v1` CPU model |
| iommu-riscv-test | 4 | 0 | no `riscv-iommu-pci` or `iommu-testdev` |
| k230-wdt-test | 7 | 0 | no `k230` board |
| machine-none-test | 1 | 1 | |
| qmp-test | 9 | 9 | |
| readconfig-test | 1 | 1 | |
| netdev-socket | 10 | 10 | |
| Total | 243 | 126 | |

## qmp-cmd-test and VNC

The QEMU build used for the other tests has no VNC, so its `qmp-cmd-test` is compiled to expect `query-vnc` to fail, and ruvm, which has VNC and answers it, stopped at the 18th case. The qmp-cmd-test rows now come from a second QEMU build from the same tree with `--enable-vnc --enable-pixman` added and `--disable-docs --disable-werror --disable-rust`. The stock `qmp-cmd-test` of that build passes 61 of 61 with its own QEMU and 64 of 64 with ruvm on both targets. The three extra cases are `query-tpm`, `query-tpm-models` and `query-tpm-types`, which ruvm answers and which a QEMU built with `--without-default-features` leaves out of its schema. ruvm was a debug build of main at #307 for these rows.

## tests/tcg for riscv64

| Suite | Reference | Reference passes | ruvm passes | Notes |
|---|---|---|---|---|
| riscv64-softmmu | QEMU 11.1 | 3 of 5 | 3 of 5 | the other two need TCG plugins, which neither build has |
| riscv64-linux-user | qemu-riscv64 8.2.2 | 31 of 34 | 26 of 34 | ruvm measured with #239 |

The QEMU 11.1 build here is softmmu only, so the linux-user reference is the distribution's qemu-riscv64 8.2.2. It fails `tb-link`, `test-mmap` and `linux-sigrtminmax`. The 12 gdbstub tests were skipped for both, because the host has no gdb with riscv64 support. ruvm passes `tb-link`, which 8.2.2 fails, and fails eight: `test-mmap` and `linux-sigrtminmax` like 8.2.2 (the second because ruvm does not take `-t`), `semihosting` (the semihosting call stops with SIGTRAP, and ruvm's linux-user does not take `-semihosting`), `signals` (`timer_create` returns ENOSYS), `linux-test` (`shmget` returns ENOSYS), `linux-madvise` (a file mapping reads back the wrong byte after `madvise`), and `linux-shmat-maps` and `linux-shmat-null` (no System V shared memory).

## What is left

On riscv64, `virt` now runs under the qtest accelerator with no harts and with its timers on the qtest clock, has a `/machine` object, and is listed in `query-machines` with the values QEMU 11.1 gives. qom-test and test-hmp pass on it. The machine takes `aclint=on` with QEMU's devices and device tree, which every qos-test command line on riscv64 asks for. riscv-csr-test and iommu-riscv-test now start and fail on the CPU model and the devices they ask for.

On aarch64, `virt` now runs under the qtest accelerator the same way, and is listed in `query-machines` as `virt-11.1` with the alias `virt` and the values QEMU 11.1 gives. qom-test and test-hmp pass on it. sbsa-ref is not listed yet, because it does not run under qtest. `query-cpu-model-expansion` now answers for every Arm CPU type of QEMU 11.1, with the properties, errors and SVE vector lengths QEMU gives under TCG and under the qtest accelerator, and `-cpu` takes `sve`, the `sve<N>` lengths and `aarch64=on` the way QEMU does, so a guest of `max` gets only the vector lengths it asks for. With `gic-version=max` on `virt`, arm-cpu-features passes all 3 cases. Under TCG and qtest it gives a GICv3, as QEMU does when the CPUs have no EL2, and `gic-version=host` fails with QEMU's error. With `virtualization=on` QEMU would pick a GICv4, which ruvm does not have, so ruvm refuses that combination. numa-test passes since `virt` takes `-numa` with `node`, `dist` and `cpu`. The nodes take their memory from `memdev`, and the CPUs go to the nodes the way QEMU puts them, by socket when `-numa` gives no CPUs. `info numa`, `x-query-numa` and the `node-id` of `query-cpus-fast` give what QEMU gives, and the device tree has the `numa-node-id` properties and the distance map QEMU writes into it. `virt` takes `-machine memory-backend=` now, so migration-test starts both guests under TCG. `virt` on TCG now migrates with QEMU's `timer`, `ram` and `globalstate` sections and QEMU's configuration section, and QEMU's analyze-migration.py reads the stream. ruvm takes the `x-ignore-shared` capability and writes the address of each block into the block list the way QEMU 11.1 does, so the block list and the configuration section match QEMU byte for byte apart from the three `/rom@etc` ACPI blocks QEMU adds and ruvm does not have yet, and the analyze-script case of migration-test passes. The vCPUs now migrate too, in QEMU's `cpu_common` and `cpu` sections with the `vfp`, `sve`, `irq-line-state`, `wfxt-timer` and `fpmr` subsections QEMU 11.1 sends for `max`. The register list has the same 362 indexes as QEMU's. A guest of `max` gives a different stream from QEMU's in the reset value of CPSR, SCTLR_EL1 and CNTFRQ_EL0 and in the ID registers, because ruvm's `max` has fewer features and reads the AArch32 ID registers as 0, so QEMU would refuse a stream from ruvm on these registers. The multifd case of migration-test now moves the a-b guest to the destination, which goes on printing `B`. Under a debug build each pass of that guest takes about a minute on server2, too slow for the 480 s timeout of the test, so the migration-test count comes from a release build. The GICv3 and its ITS now migrate too, in QEMU's `arm_gicv3` and `arm_gicv3_its` sections, and for `max` with `-S` both match QEMU 11.1 field for field apart from the `gicd_nmi` and `nmi` subsections, which QEMU sends because its `max` has FEAT_NMI and ruvm's does not. The generic timers and the PMU cycle counter now count the board's virtual clock, as QEMU's do, so they stop while the VM is stopped and the `timer` section carries them, and a Linux guest moved to the destination goes on with its uptime from where it left off, as it does under QEMU 11.1. ruvm takes the `postcopy-blocktime` capability, and the destination of a postcopy migration reports how long its vCPUs waited for pages in `query-migrate` the way QEMU 11.1 does, so the postcopy/plain case of migration-test passes. The UARTs, the RTC, the GPIO controllers and the power key of `virt` now migrate in QEMU 11.1's `pl011`, `pl031`, `pl061` and `gpio-key` sections, with the instance and section ids QEMU gives them, also with a second `-serial` and with `secure=on`, and for `max` with `-S` they match QEMU field for field apart from the time the RTC holds. The two flashes now migrate in QEMU's `pflash_cfi01` sections, with the `pflash_cfi01_blk_write` subsection while a write to buffer is in progress, and a guest that leaves a flash in status mode or in the middle of a write to buffer gives the same sections as under QEMU 11.1. The PCI host now migrates in QEMU 11.1's `0000:00:00.0/gpex_root`, `PCIHost` and `PCIBUS` sections, which match QEMU's byte for byte for `max` with `-S`. `fw_cfg` and the state of the ACPI tables now migrate in QEMU's `fw_cfg` and `virt_acpi_build` sections, and `fw_cfg` sends the sizes of the three ACPI files in its `fw_cfg/acpi_mr` subsection as QEMU does, also on q35. Both match QEMU byte for byte, also with `acpi=off`, so without firmware the stream now has every section QEMU 11.1 sends for `virt`. With firmware the ACPI GED now migrates in QEMU's `acpi-ged` section with its `acpi-ged/memhp` subsection, which matches QEMU's byte for byte, so with firmware too the stream has every section QEMU 11.1 sends for `virt`. Devices plugged into the PCI bus and the three `/rom@etc` ACPI blocks come next. boot-serial-test passes since ruvm takes `-no-shutdown`, which stops the machine in the `shutdown` state on a guest shutdown, as QEMU does, while `quit` and a signal still end the process.

On both targets `system_reset` now resets `virt` under the qtest accelerator and sends the RESET event QEMU sends, which qos-test needs between cases that share a command line. qos-test, bios-tables-test, cdrom-test and iommu-smmuv3-test decide what to run from `qom-list-types` with `implements=device`. ruvm now lists the device types with the parents QEMU gives them, but only for the devices whose qos-test cases pass, because a qos-test case that fails stops the whole test. So far that is virtio-rng, virtio-blk, virtio-balloon and virtio-net, on PCI and on virtio-mmio. Their hotplug cases are skipped on both sides, because the generic PCI host does not take hotplug. The virtio-blk cases needed `-drive` to take `auto-read-only` and `file=null-co://` with `file.read-zeroes`, and the QMP `block_resize` command to resize a drive and raise the configuration interrupt of its device. virtio-balloon only needed the boards to plug it, with the same backend as on x86, which leaves the RAM the guest gives back alone. virtio-net needed the boards to take `-netdev`, the QMP `announce-self` command, which sends the RARP frames from a thread of its own because ruvm has no realtime timer list yet, and the NIC timers to run on their own thread under qtest, since the qtest clock only moves when a test steps it and QEMU 11.1's virtio tests no longer do.

With virtio-blk listed, bios-tables-test and cdrom-test now register their cases. ruvm now takes `-cdrom`, `-hda` to `-hdd` and a bare image name as the `-drive` QEMU makes of them. QMP `query-block` now lists the drives of these boards and the devices they are on as QEMU does, with the empty `floppy0` drive QEMU adds when there is no `-nodefaults`, and HMP `info block` prints them the same way, so cdrom-test passes on `virt` and `none`. `info block -v` is not there yet. `-drive` now takes `snapshot=on`, which puts a temporary qcow2 overlay on the image as QEMU does, and the pflash devices read and write their drives through the block layer, so bios-tables-test boots its UEFI image with the variable store it gives with `snapshot=on`. The RISC-V `virt` case of bios-tables-test passes. The Arm cases stop at `ras=on`, which needs the HEST table and the hardware error memory, and then at the SMBIOS tables, which ruvm does not build for `virt` yet. On the Arm and RISC-V boards and on `none` a `-drive` image is now opened by the block layer, as `drive_new()` does in QEMU, so a probed qcow2 image such as the CD image of bios-tables-test is read as qcow2. The next devices are virtio-scsi and virtio-serial, which needs `virtconsole` and `virtserialport` on the serial bus. The rest of the qos-test devices (the Intel and other PCI network cards, the SCSI, NVMe and UFS controllers, the sound cards and the IOMMUs) are not built on these boards by ruvm.

readconfig-test passes since ruvm takes `-readconfig`, which reads the file the way QEMU's `sscanf()` loop does and handles each group as the command line option it stands for, and has the `rng-builtin` and `rng-random` objects. virtio-rng does not take their `rng` link yet. bios-tables-test also boots the guest from a qcow2 CD image with UEFI firmware. netdev-socket passes since HMP `info network` came in #234 and the stream netdev events in #239.

The Xilinx, Raspberry Pi, Aspeed, Nuvoton and Kendryte K230 boards are not in ruvm. Those tests fail because the machine type is unknown. Each board is its own piece of work and none of them is planned for M6.

The two softmmu tcg tests that use TCG plugins fail on both sides, because neither build has plugin support.
