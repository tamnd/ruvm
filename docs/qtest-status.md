# qtest and tests/tcg status against QEMU

This note records how ruvm does on QEMU's own qtest suites for aarch64 and riscv64 and on QEMU's riscv64 tests/tcg, next to QEMU 11.1 running the same tests on the same host. It covers the third and fourth exit criteria of M6 (macOS, Windows, Arm, RISC-V), and it is updated as the failures below get fixed.

## Method

The tests come from the QEMU 11.1.0 source tree. QEMU itself was built from that tree with `--target-list=aarch64-softmmu,riscv64-softmmu --without-default-features --enable-fdt=internal --enable-tcg`, which also builds the qtest binaries. The host is server2, an x86_64 Linux machine.

Each qtest binary was run on its own, once with `QTEST_QEMU_BINARY` pointing at the QEMU build and once pointing at ruvm. libqtest takes the target from the binary name, so ruvm was reached through symlinks named `qemu-system-aarch64` and `qemu-system-riscv64`. The test list, environment and timeouts are the ones meson writes into `tests.json` for the `qtest-aarch64` and `qtest-riscv64` suites, and the counts below are read from the TAP output of each test.

The ruvm binary was a release build of main at #224 plus the human monitor change that follows it, except for netdev-socket and the riscv64 linux-user tests, which were run again with a release build that has #239. The qtest-riscv64 suite was run again with the change that runs riscv64 `virt` under the qtest accelerator.

A count is the number of TAP cases that passed. A test that registers no cases passes trivially with 0, and that is shown as 0 rather than as a pass, because it means the test found nothing in ruvm to run against.

## Criterion 3: a RISC-V Linux guest under TCG

On the x86_64 Linux host (server2), ruvm boots a Linux kernel and initramfs on the riscv64 `virt` machine under TCG. OpenSBI prints its banner at 0.93 s, the kernel banner appears at 1.33 s, the kernel runs `/init` at 12.36 s and the shell prompt is up at 13.78 s.

The riscv64 host was tested in #176 by running ruvm, built for riscv64gc Linux, under qemu-riscv64 8.2.2 on server3. There the riscv64 `virt` guest reached the kernel banner at 1080 s and a shell at 2446 s, and an x86 q35 guest reached a shell in about 570 s. That run has not been repeated for this note, because server3 has been too loaded to give a fair time and its old artifacts are gone. The numbers above are the ones from #176.

## qtest-aarch64

| Test | QEMU | ruvm | Why ruvm is short |
|---|---|---|---|
| bios-tables-test | 17 | 0 | registers no cases: ruvm does not list `virt` in `query-machines` |
| qom-test | 90 | 2 | one case per machine, and ruvm lists only `none` |
| device-introspect-test | 6 | 6 | |
| cdrom-test | 9 | 0 | registers no cases: needs `virt` |
| migration-test | 12 | 0 (12 skipped) | skipped: "machine virt not supported" |
| boot-serial-test | 1 | 0 | registers no cases: needs a board from its list |
| test-hmp | 90 | 2 | one case per machine |
| qmp-cmd-test | 59 | 17 | stops at `query-vnc`, see below |
| qos-test | 90 (5 skipped) | 0 | registers no cases: needs `virt` |
| xlnx-canfd-test | 3 | 0 | no `xlnx-zcu102` board |
| xlnx-versal-trng-test | 5 | 0 | no `xlnx-versal-virt` board |
| bcm2835-dma-test | 1 | 0 | no Raspberry Pi boards |
| bcm2835-i2c-test | 3 | 0 | no Raspberry Pi boards |
| ast2700-gpio-test | 2 | 0 | no Aspeed boards |
| ast2700-hace-test | 11 | 0 | no Aspeed boards |
| ast2700-sgpio-test | 3 | 0 | no Aspeed boards |
| ast2700-smc-test | 8 | 0 | no Aspeed boards |
| npcm_gmac-test | 4 | 0 | no Nuvoton boards |
| iommu-smmuv3-test | 3 | 0 (3 skipped) | skipped: needs `virt` |
| cxl-test | 1 | 0 | needs `virt` with CXL |
| arm-cpu-features | 3 | 0 | registers no cases: needs `virt` |
| numa-test | 5 | 0 | registers no cases: needs `virt` |
| machine-none-test | 1 | 1 | |
| qmp-test | 9 | 9 | |
| readconfig-test | 1 | 0 | `-readconfig` is not supported |
| netdev-socket | 10 | 10 | |
| Total | 447 | 47 | |

## qtest-riscv64

| Test | QEMU | ruvm | Why ruvm is short |
|---|---|---|---|
| bios-tables-test | 3 | 0 | registers no cases, see below |
| qom-test | 13 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| device-introspect-test | 6 | 6 | |
| cdrom-test | 1 | 0 | `-cdrom` is not supported |
| test-hmp | 13 | 3 | one case per machine: ruvm has `none` and `virt`, and passes both |
| qmp-cmd-test | 59 | 17 | stops at `query-vnc`, see below |
| qos-test | 113 (5 skipped) | 0 | registers no cases, see below |
| riscv-csr-test | 1 | 0 | no `veyron-v1` CPU model |
| iommu-riscv-test | 4 | 0 | `-net none` is not supported, and there is no `riscv-iommu-pci` or `iommu-testdev` |
| k230-wdt-test | 7 | 0 | no `k230` board |
| machine-none-test | 1 | 1 | |
| qmp-test | 9 | 9 | |
| readconfig-test | 1 | 0 | `-readconfig` is not supported |
| netdev-socket | 10 | 10 | |
| Total | 241 | 49 | |

## qmp-cmd-test and VNC

The QEMU build used for the tests has no VNC, so `qmp-cmd-test` is compiled to expect `query-vnc` to fail. ruvm has VNC and answers it, so the stock test stops at its 18th case. To compare the rest, the test was rebuilt from a copy of its source with the `#ifndef CONFIG_VNC` block turned off, because `config-host.h` undefines `CONFIG_VNC` and a `-D` on the command line does not reach it. With that variant ruvm passes 61 of 61 on both targets and QEMU passes 59 of 59. The two extra cases are `query-vnc` and `query-vnc-servers`, which QEMU without VNC does not have in its schema.

## tests/tcg for riscv64

| Suite | Reference | Reference passes | ruvm passes | Notes |
|---|---|---|---|---|
| riscv64-softmmu | QEMU 11.1 | 3 of 5 | 3 of 5 | the other two need TCG plugins, which neither build has |
| riscv64-linux-user | qemu-riscv64 8.2.2 | 31 of 34 | 26 of 34 | ruvm measured with #239 |

The QEMU 11.1 build here is softmmu only, so the linux-user reference is the distribution's qemu-riscv64 8.2.2. It fails `tb-link`, `test-mmap` and `linux-sigrtminmax`. The 12 gdbstub tests were skipped for both, because the host has no gdb with riscv64 support. ruvm passes `tb-link`, which 8.2.2 fails, and fails eight: `test-mmap` and `linux-sigrtminmax` like 8.2.2 (the second because ruvm does not take `-t`), `semihosting` (the semihosting call stops with SIGTRAP, and ruvm's linux-user does not take `-semihosting`), `signals` (`timer_create` returns ENOSYS), `linux-test` (`shmget` returns ENOSYS), `linux-madvise` (a file mapping reads back the wrong byte after `madvise`), and `linux-shmat-maps` and `linux-shmat-null` (no System V shared memory).

## What is left

On riscv64, `virt` now runs under the qtest accelerator with no harts and with its timers on the qtest clock, has a `/machine` object, and is listed in `query-machines` with the values QEMU 11.1 gives. qom-test and test-hmp pass on it. qos-test and bios-tables-test still register no cases, and riscv-csr-test and iommu-riscv-test now start and fail on the CPU model and the devices they ask for.

On aarch64, `query-machines` still lists only `none`, although ruvm runs the `virt` machine on TCG. Most qtests choose their machines from that list, so bios-tables-test, cdrom-test, qos-test, arm-cpu-features, numa-test and migration-test register nothing or skip, and qom-test and test-hmp cover one machine instead of many. Running aarch64 `virt` under the qtest accelerator is the next change.

`-cdrom` and `-readconfig` are command line options ruvm does not take yet. netdev-socket passes since HMP `info network` came in #234 and the stream netdev events in #239.

The Xilinx, Raspberry Pi, Aspeed, Nuvoton and Kendryte K230 boards are not in ruvm. Those tests fail because the machine type is unknown. Each board is its own piece of work and none of them is planned for M6.

The two softmmu tcg tests that use TCG plugins fail on both sides, because neither build has plugin support.
