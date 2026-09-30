# 12. Core devices

This document specifies the non-virtio device models in ruvm: how a device is written against the `Device` and `MmioOps` traits, the catalog of device classes QEMU 11.1 has and how many of them there are, and for each class the modelling approach, the fast paths that matter for performance, and the way we prove register-level equivalence with QEMU. virtio devices, including virtio-gpu, are in document 13. Network card models are in document 15 together with the network backends. VFIO, vIOMMUs and CXL are in document 16. The VMState format that every device here must reproduce is in document 17, and the test infrastructure (qtest server, functional test runner, differential harness) is in document 22. Machines that assemble these devices are in document 11.

## How big the job is

Counts were taken from a build of the QEMU v11.1.0 tag on a macOS arm64 host with default configure options, covering all 28 system emulators except hexagon, by running `-device help` and QMP `qom-list-types` with `implements=device` and `abstract=false` on every binary and taking the union.

| Measure | Count |
| --- | --- |
| Concrete QOM device types, union of all binaries | 2501 |
| of which CPU models (type names ending in `-cpu`) | 1217 |
| Non-CPU concrete device types | 1284 |
| `-device help` entries, union | 930 |
| of which CPU entries | 461 |
| User-creatable non-CPU devices | 469 |
| qemu-system-x86_64 `-device help`, non-CPU | 219 of 407 |
| virtio device types in the union | 63 |
| Concrete device types per binary | aarch64 893, arm 835, ppc64 824, ppc 572, x86_64 454, riscv64 409 |

A Linux build adds types this macOS build does not have, among them `qxl` and `qxl-vga` (SPICE), `usb-redir` (usbredir), `vfio-pci`, the `vhost-user-*` family, `virtio-gpu-gl` and `ivshmem-plain` and `ivshmem-doorbell`. We did not count the Linux delta separately; the numbers above are a lower bound.

The source behind these types is large. hw/ holds about 811,000 lines of C. The directories that matter most for this document: hw/intc 54.3k, hw/usb 36.3k, hw/display 36.3k, hw/scsi 24.9k, hw/audio 14.7k, hw/nvme 13.0k, hw/pci 10.5k, hw/ide 9.8k. Board and SoC directories add hw/arm 59.5k, hw/ppc 46.2k, hw/i386 32.7k, hw/s390x 15.5k and hw/riscv 13.2k, and hw/net is 67.8k. The 469 user-creatable devices are the compatibility surface libvirt and users see. The other roughly 800 non-CPU types are SoC internals (UARTs, timers, GPIO blocks, clock controllers) that only exist inside a board, and they are ported board by board under the long-tail process at the end of this document.

QEMU 11.1 itself contains two Rust device models, `pl011` in rust/hw/char/pl011 and `hpet` in rust/hw/timer/hpet. hw/char/Kconfig and hw/timer/Kconfig select the Rust version when the build has Rust (`X_PL011_RUST`, `X_HPET_RUST`) and the C version otherwise. They are written against QEMU's own bindings and cannot be reused in ruvm.

## The Device trait in practice

The canon fixes the shape: `trait Device: Object` with `realize`, `unrealize`, `reset(ResetType, ResetPhase)` and `vmstate()`, properties declared with `#[derive(Device)]`, and `trait MmioOps` whose `read` and `write` receive an `AccessCtx` carrying `MemTxAttrs`. This section shows what a device written against it looks like and which rules every device in this document follows.

```rust
#[derive(Device)]
#[device(type_name = "isa-serial", parent = "isa-device", user_creatable = true)]
pub struct IsaSerial {
    #[property(name = "index", default = -1)]
    index: i32,
    #[property(name = "iobase", default = -1)]
    iobase: i32,
    #[property(name = "irq", default = -1)]
    isairq: i32,
    #[property(name = "chardev")]
    chr: CharBackend,
    #[property(name = "wakeup", default = 0)]
    wakeup: u8,
    state: DeviceLock<Serial16550>,
    irq: IrqLine,
}

impl MmioOps for IsaSerial {
    fn read(&self, ctx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let mut s = self.state.lock(ctx);
        Ok(s.read(offset, &self.irq) as u64)
    }
    fn write(&self, ctx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let mut s = self.state.lock(ctx);
        s.write(offset, value as u8, &self.chr, &self.irq);
        Ok(())
    }
    fn valid() -> AccessConstraints { AccessConstraints { min: 1, max: 1, unaligned: false } }
}
```

The rules:

- Property names, types, defaults and descriptions match QEMU exactly, including `x-` experimental ones, because `device-list-properties` output and `-device foo,help` text are part of the compatibility contract (document 02). tests/qtest/device-introspect-test.c walks every device type and calls `device-list-properties` and `device_add ...,help`; it runs against ruvm unmodified.
- The `vmstate()` description has the same section name, version id, minimum version id, field order, subsections and `needed` predicates as the C `VMStateDescription`. Document 17 describes how this is checked against QEMU's `-dump-vmstate` JSON.
- Trace points keep QEMU's names and argument lists. Every `trace_foo(...)` call in the C file becomes a `trace!(foo, ...)` call in the same place, generated from a copy of the directory's trace-events file. There are about 4,323 trace events across hw/. Keeping them is what makes trace diffing (below) possible, and users' tracing scripts keep working.
- Access size and alignment rules come from the C `MemoryRegionOps.valid` and `.impl` blocks. When `.impl` is narrower than `.valid`, QEMU's memory core splits the access (`access_with_adjusted_size` in system/memory.c), and ruvm-mem does the same split so the device sees the same sequence of calls.
- Reset uses the three phases with the rules in docs/devel/reset.rst: `enter` resets only local state and must not raise or lower an IRQ line or touch guest memory; effects on other objects happen in `hold` or `exit`. ruvm enforces this by not giving `enter` access to IRQ lines or the DMA address space.

### Locking and device domains

QEMU runs almost every device callback under the BQL. ruvm has no global lock on the hot path, so each device declares its lock domain at realize time. There are three cases.

- A standalone device (a UART, a watchdog, a USB controller) owns a `DeviceLock<T>`, a mutex that records the domain in debug builds so that lock order violations fail tests.
- Devices that call into each other synchronously share a domain. The canon examples are an interrupt controller and its CPUs, and a PCI host bridge and its configuration space. Other shared domains in this document: the i8259 pair, the IOAPIC and the LAPICs it targets (only when the irqchip is emulated in user space), the ISA bridge and its PIT, RTC and PIC children on pc machines, and an AHCI controller with its ports.
- Devices on the performance path where a mutex would be visible use lock-free state for the hot register and a lock for the rest. Examples are the NVMe doorbells, the MSI-X pending bit array and the HPET main counter read.

The lock a device takes is recorded in `AccessCtx`, so a device calling another device through an IRQ line or a DMA read either stays in its domain or is required by the type system to release its lock first. IRQ lines (`IrqLine`, the equivalent of `qemu_irq`) are `Send + Sync` handles whose `set` either calls into the target domain directly when the domain is the same, or posts to the target's mailbox and returns. The target domain handles the post before the vCPU that raised it returns to the guest, which is the ordering guarantee QEMU gets from the BQL and that guests depend on (for example, a write to a UART's interrupt enable register is followed by the interrupt being visible on the next instruction).

### Buses, GPIO and clocks

ruvm-hw-core provides the qdev bus types with QEMU's names and addressing rules (`System`, `ISA`, `PCI`, `PCIE`, `usb-bus`, `SCSI`, `IDE`, `i2c-bus`, `SSI`, `SD`, `ccw`, `spapr-vio`). Bus names such as `pcie.0` and `ide.0` resolve as in QEMU's `qbus_find` because libvirt XML uses them. Named GPIO arrays (`qdev_init_gpio_in_named`) and clocks (`qdev_init_clock_in`, hw/core/clock.c) are ported with the same names because board code and `qom-get` refer to them.

## Interrupt controllers

### x86: 8259, IOAPIC, LAPIC and x2APIC

QEMU has three implementations of each x86 interrupt controller: the emulated one (hw/intc/i8259.c, hw/intc/ioapic.c, hw/intc/apic.c), the KVM in-kernel proxy (hw/i386/kvm/i8259.c, hw/i386/kvm/ioapic.c, hw/i386/kvm/apic.c) that only moves state in and out of the kernel for migration, and for Xen and WHPX their own variants. `-machine kernel-irqchip=on|split|off` selects which one: `on` puts all three in KVM, `split` keeps LAPICs in KVM and the PIC and IOAPIC in user space, `off` emulates everything. microvm defaults to split (`default_kernel_irqchip_split = true` in hw/i386/microvm.c) and q35 to on.

ruvm follows the same model. Under KVM the in-kernel objects are thin state carriers whose VMState matches the C `kvm-apic`, `kvm-ioapic` and `kvm-i8259` sections. When split or off, the user-space IOAPIC is ported from ioapic.c, including `ioapic_service` and the version 0x20 default, and the user-space LAPIC from apic.c. Under TCG, HVF and WHPX without an in-kernel APIC, the LAPIC state is owned by the vCPU thread it belongs to: LAPIC register accesses from that vCPU are plain function calls with no lock, and cross-CPU delivery (IPIs, IOAPIC to LAPIC messages) goes through the target vCPU's mailbox and kick, the same thing `cpu_interrupt` does in QEMU. x2APIC MSR access is handled in ruvm-target-x86 (document 09) and ends in the same LAPIC code. The x2APIC rule from apic_common.c, that APIC IDs above 255 require the `x2apic` CPU feature and fail realize with the hint "Try x2apic=on in -cpu", is kept with its text.

Fast path: under KVM, MSI delivery from any device uses irqfd with a route added by `kvm_irqchip_add_msi_route`, and level-triggered INTx uses irqfd with a resample fd, so no device interrupt crosses a user-space lock at all. QEMU does the same; in ruvm the device raising the irqfd holds no global lock.

### Arm: GICv2, GICv3, GICv4 and ITS, GICv5

hw/intc/arm_gic.c (GICv2), arm_gicv3*.c (GICv3 and the TCG CPU interface), arm_gicv3_its.c (ITS), arm_gicv2m.c (MSI frame for GICv2) and the accelerator-specific arm_gic_kvm.c, arm_gicv3_kvm.c, arm_gicv3_its_kvm.c, arm_gicv3_hvf.c and arm_gicv3_whpx.c make up the Arm side. The GICv3 common code has a `revision` property (3 by default); `-machine virt,gic-version=4` sets revision 4, which enables GICv4 virtual LPI support in the emulated redistributor and ITS so that nested guests under TCG get direct injection. QEMU 11.1 also has a GICv5 model under hw/intc (the IRS and related pieces), reachable on virt with `gic-version=x-5`, which virt.c lists as "Valid values are 2, 3, 4, x-5, host and max".

The redistributor and CPU interface for each vCPU live in that vCPU's domain, and the distributor and ITS are a shared domain. The GICv3 CPU interface is accessed through system registers from TCG, so `ICC_IAR1_EL1` and `ICC_EOIR1_EL1` reads are direct calls into the per-CPU state without a lock, which is where most GIC time goes in a TCG guest. The ITS command queue is processed when `GITS_CWRITER` is written, as in `process_cmdq` in arm_gicv3_its.c, reading the queue from guest memory through the ITS's DMA address space.

Under KVM and HVF the GIC is in the host kernel or hypervisor framework, and ruvm only moves state for migration. `arm_gicv3_hvf.c` (added in 2025) uses the GIC that Hypervisor.framework provides on macOS releases that have one; ruvm-accel-hvf (document 06) supports both it and the user-space GIC.

### RISC-V: PLIC, ACLINT, APLIC and IMSIC

riscv virt takes `aia=none|aplic|aplic-imsic` and `aclint=on|off` machine properties (hw/riscv/virt.c). With `aia=none` the interrupt controller is the SiFive PLIC (hw/intc/sifive_plic.c) and the CLINT or, with `aclint=on`, the ACLINT MTIMER, MSWI and SSWI blocks (hw/intc/riscv_aclint.c). With `aplic-imsic`, wired interrupts go to the APLIC (riscv_aplic.c), which forwards them as MSIs to the per-hart IMSIC files (riscv_imsic.c), and PCI MSIs go directly to the IMSIC. `aia-guests` sets the number of guest interrupt files per hart for the H extension.

The IMSIC file of each hart is in that hart's domain and is accessed both as MMIO (the `seteipnum` doorbell) and through the `*ireg` indirect CSRs. The MMIO doorbell write is the MSI fast path: it sets a bit in an atomic bitmap and kicks the hart, with no lock taken on the sender side. Under KVM the AIA is in the kernel and the user-space model only transfers state.

### POWER: XICS and XIVE

pseries supports both XICS (hw/intc/xics.c, xics_spapr.c, xics_kvm.c) and XIVE (xive.c, spapr_xive.c, spapr_xive_kvm.c), and the guest picks one at CAS time when `ic-mode=dual`, which is the default for recent pseries versions. powernv9 and later use the XIVE and XIVE2 models (pnv_xive.c, pnv_xive2.c) that follow the hardware more closely, including the thread interrupt management area. The switch at CAS time is a control lock operation in ruvm. XIVE's ESB pages (event state buffer, MMIO loads and stores that trigger or query an interrupt) are the fast path and are implemented as an MMIO region per source whose handler touches only that source's state.

### s390x and the rest

s390x has the floating interrupt controller `s390-flic` (hw/intc/s390_flic.c) and its KVM variant `s390-flic-kvm` (s390_flic_kvm.c), which handle I/O, service and machine check interrupts and adapter interrupts for virtio-ccw. The flic has no MMIO; it is driven by instructions and by the channel subsystem, so it is a domain shared with the channel subsystem.

The remaining controllers are ported board by board: openpic (ppce500 and mac99), armv7m_nvic for Cortex-M boards, the LoongArch extioi, pch_pic, pch_msi and IPI blocks and their KVM variants, and the SoC controllers under hw/intc.

### Testing interrupt controllers

tests/qtest has direct tests for some (pnv-xive2-test.c, stm32l4x5_exti-test.c) and the rest are exercised by booting. We add qtest scripts that use `irq_intercept_in` and `set_irq_in` to drive input lines and observe outputs, run each script against QEMU and ruvm, and require identical output. For the x86 APIC, kvm-unit-tests (the apic, ioapic and x2apic tests) run under TCG and KVM in the nightly suite. For the GIC, the `arm/gic` tests from kvm-unit-tests and the tests/tcg/aarch64 system tests are used. VMState equivalence for the in-kernel proxies is checked by migrating between QEMU and ruvm under each `kernel-irqchip` mode (document 17).

## Timers

### PIT, HPET, RTC

The i8254 PIT (hw/timer/i8254.c, and hw/i386/kvm/i8254.c for the in-kernel one), HPET (hw/timer/hpet.c and the Rust version) and the MC146818 RTC (hw/rtc/mc146818rtc.c) all use QEMU's virtual clock through `QEMUTimer`. ruvm has an equivalent timer list per clock type (document 04 and 06 cover the main loop and clocks), and each timer callback runs in the owning device's domain on the main loop thread or on a vCPU thread when the vCPU is the one that armed it with an overdue deadline.

Guest-visible time behaviour is ported exactly, including the lost tick policy of the RTC: with `-global mc146818rtc.lost_tick_policy=slew`, `periodic_timer_update` and the `irq_coalesced` counter in mc146818rtc.c reinject missed periodic interrupts at a faster rate, which old Windows guests rely on for timekeeping. The RTC's index port 0x70 is registered as coalesced PIO (`memory_region_add_coalescing` on the `coalesced_io` subregion), so under KVM a write to the index register is buffered in the kernel's coalesced ring and not handled until the data port access flushes it. ruvm supports KVM's coalesced MMIO and PIO rings (document 06) and registers the same ranges: the RTC index port, the q35 and i440FX 0xcf8 config address port, and the legacy VGA window.

The HPET main counter is read often by guests that choose it as clocksource. The counter read is computed from the virtual clock without taking the device lock (an atomic load of the offset plus a clock read), and only comparator writes take the lock. The comparator timers use `hpet_timer` logic from hw/timer/hpet.c, including the 32-bit mode wraparound rules.

### ARM generic timer and RISC-V ACLINT

The ARM generic timer is part of the CPU in QEMU (target/arm/helper.c, `gt_recalc_timer`), not a device. It is in ruvm-target-arm (document 09); this document only notes that its outputs are wired to GIC PPIs by the board, which is a device concern. Under KVM and HVF the timer is virtualized by the host.

The RISC-V ACLINT MTIMER (hw/intc/riscv_aclint.c) is an MMIO device holding `mtime` and per-hart `mtimecmp`. With the Sstc extension, the supervisor timer is a CSR (`stimecmp`) in the CPU model and needs no MMIO exit. Reads of `mtime` are computed from the virtual clock and the `timebase-freq` without a lock.

### SoC timers

pl031, goldfish_rtc, arm_mptimer, a9gtimer, sse-timer and the many vendor timers under hw/timer are long-tail devices. The CMSDK, NPCM7xx and SSE timers have qtests (cmsdk-apb-timer-test.c, cmsdk-apb-dualtimer-test.c, npcm7xx_timer-test.c, sse-timer-test.c) that drive time with `clock_step`, which is the model for tests we write for the others.

## PCI and PCI Express

### Host bridges and configuration space

The host bridges are ported from hw/pci-host: q35 (MCH), i440fx, gpex (arm, riscv and loongarch virt, and the generic ECAM host), the spapr PHB, the pnv PHB3, PHB4 and PHB5, xilinx-pcie, designware and the smaller SoC hosts. Configuration space access goes through `pci_host_config_write_common` and `pci_host_config_read_common` semantics: for the 0xcf8/0xcfc mechanism the address latch is per host bridge, and for ECAM the address is decoded from the MMIO offset. q35 puts the MMCONFIG window at 0xb0000000 by default (`MCH_HOST_BRIDGE_PCIEXBAR_DEFAULT`) with a 256 MiB size.

Configuration space writes that change BARs, the command register or bridge windows trigger `pci_update_mappings`, which rebuilds the address space's FlatView (document 05). This is hot during boot, when firmware sizes and assigns every BAR. ruvm batches FlatView rebuilds per config write, as QEMU does with its memory transaction, and the rebuild itself is RCU-published so vCPUs never wait on it.

Each device's configuration space is a 256-byte or 4 KiB array with the `wmask`, `w1cmask` and `cmask` arrays from hw/pci/pci.c, so writability is data and the incoming migration check in `get_pci_config_device` is unchanged.

### BARs, MSI and MSI-X

BAR sizing, 64-bit BARs, prefetchable windows and the option ROM BAR are ported from hw/pci/pci.c. MSI (hw/pci/msi.c) is a config space capability; `msi_notify` builds the message and delivers it through the device's DMA address space so that a vIOMMU with interrupt remapping sees it. MSI-X (hw/pci/msix.c) has an MMIO table and pending bit array in a BAR.

The MSI-X fast path is under KVM: every unmasked vector gets a KVM MSI route and an irqfd when the device is backed by an iothread or an external process (vhost, VFIO), and device models inside ruvm signal the irqfd directly from the thread that completed the I/O. Vector masking and unmasking writes to the table are handled in the MSI-X region's lock without touching the device's main lock, and pending bits are atomics. `msix_vector_use` and `msix_vector_unuse` reference counting and the `msix-exclusive-bar` layouts are ported because the BAR layout is guest visible.

### Root ports, switches and expanders

`pcie-root-port` (gen_pcie_root_port.c), `ioh3420`, `x3130-upstream` and `xio3130-downstream`, `pcie-pci-bridge`, `pci-bridge` (pci_bridge_dev.c), and the expander bridges `pxb` and `pxb-pcie` are all user-creatable and used by libvirt, which builds deep topologies of them. Bridge windows, bus numbering, `chassis` and `slot` properties and the `x-` compat properties on them are ported exactly. The CXL root, upstream and downstream ports live in the same directory and are specified in document 16.

### Hotplug

PCI has three hotplug mechanisms in QEMU and all three are used.

- Native PCIe hotplug: the slot capability on root and downstream ports (`hotplug` property, default true, in hw/pci/pcie_port.c), driven by `pcie_cap_slot_plug_cb` and the attention button and presence detect state machine in hw/pci/pcie.c.
- SHPC on conventional PCI bridges (hw/pci/shpc.c, `shpc_device_plug_cb`).
- ACPI PCI hotplug (hw/acpi/pcihp.c, `acpi_pcihp_device_plug_cb`) with its register block at 0xae00 on pc and 0x0cc0 on q35. On q35, ACPI hotplug for bridges and root ports is on from 6.1; pc_compat_6_0 sets `{ "ICH9-LPC", "acpi-pci-hotplug-with-bridge-support", "off" }` and the `x-keep-pci-slot-hpc` property controls whether the native slot capability stays visible when ACPI hotplug is active.

Hotplug changes the device tree, so it runs under the control lock. The guest sequence (write to the eject register, the `_EJ0` method, the attention button, the presence detect change) is part of the ABI, and the ACPI side is covered by the `DSDT.bridge`, `DSDT.noacpihp` and related expected blobs described in document 11. tests/qtest/device-plug-test.c and ioh3420-test.c run against ruvm, and we add scripts that replay the Linux pciehp and acpiphp register sequences captured from QEMU.

### SR-IOV, ARI, AER, ATS, PASID, PRI

These are PCIe extended capabilities. hw/pci/pcie_sriov.c implements the SR-IOV capability and VF creation; its users in 11.1 include nvme, igb and virtio-pci. VFs are real PCI functions in the QOM tree, created and destroyed when the guest writes `NumVFs` and the VF Enable bit, which makes VF enable a topology change under the control lock. hw/pci/pcie.c has `pcie_ari_init`, `pcie_ats_init`, `pcie_pasid_init` and `pcie_pri_init`, and hw/pci/pcie_aer.c has AER with the error injection path used by HMP `pcie_aer_inject_error`. ATS, PASID and PRI matter mostly with a vIOMMU; document 16 covers the IOMMU side. hw/pci/pcie_doe.c implements Data Object Exchange, which CXL and SPDM use.

Capability layout is ABI, as the compat arrays show: hw_compat_7_2 has `{ TYPE_PCI_DEVICE, "x-pcie-err-unc-mask", "off" }`, hw_compat_8_0 has `{ TYPE_PCI_DEVICE, "x-pcie-ari-nextfn-1", "on" }` and hw_compat_9_1 has `{ TYPE_PCI_DEVICE, "x-pcie-ext-tag", "false" }`.

### Testing PCI

pci-test.c, i440fx-test.c, q35-test.c and the libqos PCI driver (tests/qtest/libqos/pci.c and pci-pc.c, pci-spapr.c, generic-pcihost.c) run against ruvm. The strongest check is the config space dump: for each of the fingerprint command lines in document 11, we read the full config space of every function through qtest and compare with QEMU after firmware has run, which catches wrong masks, capability offsets and compat values in one pass.

## USB

Host controllers: `piix3-usb-uhci` and the other UHCI variants (hcd-uhci.c), `pci-ohci` and `sysbus-ohci` (hcd-ohci.c), `usb-ehci` and `ich9-usb-ehci1` (hcd-ehci.c), and the XHCI models `qemu-xhci`, `nec-usb-xhci` and `sysbus-xhci` (hcd-xhci.c with its PCI, NEC and sysbus front ends), plus the SoC controllers dwc2, dwc3 and chipidea. The XHCI `p2` and `p3` properties (4 USB 2 and 4 USB 3 ports each by default) and `streams` are ported with their defaults.

Devices: `usb-kbd`, `usb-mouse`, `usb-tablet` (dev-hid.c), `usb-hub`, `usb-storage` and `usb-bot` (dev-storage.c), `usb-uas`, `usb-mtp`, `usb-net`, `usb-serial` and `usb-braille`, `usb-ccid` with its smartcard backends, `usb-audio`, `usb-wacom-tablet`, and, when built with their libraries, `u2f-emulated`, `u2f-passthru` and `canokey`. `usb-host` (host-libusb.c) passes a host device through with libusb and exists on Linux and macOS, and `usb-redir` (redirect.c) speaks the usbredir protocol over a chardev and is present when the build has usbredirparser.

Modelling: hw/usb/core.c's packet model (`USBPacket`, `usb_handle_packet`, async completion) is the interface between controllers and devices, and ruvm keeps it because every device and controller is written against it. The controllers do their schedule processing from frame timers (UHCI, OHCI, EHCI periodic lists) or from doorbells (XHCI, `xhci_doorbell_write`). The XHCI doorbell is the only USB register on a hot path; it kicks the endpoint's transfer ring processing, which runs in the controller's domain on the vCPU thread for control and interrupt endpoints and hands bulk transfers to the backing device's iothread (usb-storage and uas complete through the block layer).

Testing: usb-hcd-uhci-test.c, usb-hcd-ohci-test.c, usb-hcd-ehci-test.c and usb-hcd-xhci-test.c, and the libqos USB helpers. They mostly cover hotplug and enumeration, so we add trace diffs of Linux boots with a hub, a tablet and a storage device on each controller.

## Storage controllers

### IDE and AHCI

hw/ide has the ATA core (core.c, atapi.c), the PIIX3 and PIIX4 IDE functions (piix.c), ICH9 AHCI (ich.c and ahci.c, six ports, `ahci.ports = 6` in `pci_ich9_ahci_realize`), cmd646, via, sii3112, the macio and ISA and MMIO variants. Modelling follows QEMU's split between the ATA register state machine and the DMA engines (BMDMA for PIIX, the AHCI command list and FIS receive area). The AHCI port registers are per-port state in the controller's domain; issuing a command (`PxCI` write, `ahci_port_write`) starts DMA processing on the vCPU thread and submits the I/O to the block layer (document 14), completing on the iothread. PIO data transfers for IDE are byte or word accesses, one exit each under KVM, and are only used by firmware and old guests; we keep them correct and do not optimize them.

Testing: ide-test.c, ahci-test.c (one of the largest qtests), cdrom-test.c and hd-geo-test.c.

### SCSI HBAs

`lsi53c895a` and `lsi53c810` (lsi53c895a.c, which interprets the SCRIPTS program in the guest's memory in `lsi_execute_script`), `megasas` and `megasas-gen2` (megasas.c, `megasas_handle_frame`), `mptsas1068` (mptsas.c), `am53c974` and `dc390` (esp-pci.c), `pvscsi` (vmw_pvscsi.c) and `spapr-vscsi`. The SCSI bus and the disk and CD devices (`scsi-hd`, `scsi-cd`, `scsi-generic`, `scsi-block`) are in hw/scsi/scsi-bus.c and scsi-disk.c and are shared with virtio-scsi (document 13). These HBAs have a long history of guest-triggered memory safety bugs; QEMU's fuzz tests fuzz-lsi53c895a-test.c and fuzz-megasas-test.c are regression tests for them. We port the logic but take all DMA through the bounds-checked `DmaBuf` API from ruvm-mem, and the port runs the same fuzz inputs.

### NVMe

hw/nvme is the most complete emulated storage controller in QEMU: ctrl.c, ns.c, subsys.c, dif.c for end-to-end protection, nguid.c. The controller reports NVMe 1.4 (`NVME_SPEC_VER` 0x00010400) and supports multiple namespaces (`nvme-ns` devices), NVM subsystems shared between controllers (`nvme-subsys`) for multipath, zoned namespaces (`zoned=on` on `nvme-ns` with `zoned.zone_size`, `zoned.zone_capacity`, `zoned.max_open`, `zoned.max_active`, `zoned.numzrwa` and the zone random write area properties), SR-IOV with flexible resources (`sriov_max_vfs`, `sriov_vq_flexible`, `sriov_vi_flexible`), a controller memory buffer (`cmb_size_mb`), a persistent memory region (`pmrdev`), atomic write parameters, OCP extensions and SPDM through `spdm_port`.

Fast paths, in order of importance:

- Shadow doorbells (the Doorbell Buffer Config admin command, `nvme_dbbuf_config`). A Linux guest that finds the feature writes doorbells to a buffer in guest memory and only does an MMIO write when the event index says the controller needs one. The `dbcs` property enables the feature and defaults to true. This removes most doorbell exits without any host-side trick.
- ioeventfd for submission queue doorbells (`ioeventfd` property, default false in QEMU). With it, a doorbell write under KVM signals an eventfd and the vCPU continues immediately; the queue is processed on the iothread. ruvm honours the property and the QEMU default on the compatible CLI, and the native `ruvm run` enables it by default because the property has no guest-visible effect.
- Completion by irqfd from the iothread, as for MSI-X in general.

`nvme_process_sq` and the command handlers are ported to run on the iothread with the controller's lock held only while touching queue state, not across block layer submission.

Testing: nvme-test.c and the libqos nvme driver cover little. We add the NVMe tests from the Linux kernel's blktests suite, run in a guest nightly, and a qtest script library that issues admin and I/O commands directly (identify, create queues, zone management send and receive, SR-IOV virtualization management) and diffs completions against QEMU.

### SD, eMMC, UFS and floppy

hw/sd/sd.c implements the SD card (`sd-card`, and the SPI mode variant) and eMMC (`emmc`, `TYPE_EMMC` in include/hw/sd/sd.h), with `sd_do_command` as the command state machine. Host controllers are the SDHCI family (sdhci.c, sdhci-pci for `-device sdhci-pci`), pl181, ssi-sd and SoC controllers. sdhci-test.c, npcm7xx_sdhci-test.c and fuzz-sdcard-test.c are the qtests.

hw/ufs implements a UFS host controller (`ufs`) and logical units (`ufs-lu`) at UFS 4.1 (`UFS_SPEC_VER` 0x0410), including multi-circular queue mode (`mcq`, `mcq-maxq`) and Write Booster (`wb-max-size`, `wb-min-size` properties, and the exception event handling in `ufs_wb_update_ee_status`). ufs-test.c is thorough and runs against ruvm unchanged.

The floppy controller (hw/block/fdc.c, fdc-isa.c, fdc-sysbus.c; `isa-fdc`) is still used by some installers and by DOS-era guests on isapc. fdc-test.c covers it. It is ported directly with no performance work.

Parallel flash (`cfi.pflash01`, `cfi.pflash02`) is covered in document 11, and SPI NOR flash (hw/block/m25p80.c) with the SSI devices below.

## Display

Devices in this document: `VGA` (hw/display/vga.c and vga-pci.c), `isa-vga`, `secondary-vga`, `cirrus-vga` and `isa-cirrus-vga`, `bochs-display`, `ramfb` (a framebuffer configured through the `etc/ramfb` fw_cfg file), `qxl` and `qxl-vga` (Linux only, with SPICE), `ati-vga`, `vmware-svga`, and the SoC and legacy framebuffers (tcx, cg3, sm501, pl110, and so on). The EDID generator (hw/display/edid-generate.c) is shared by bochs-display, VGA and virtio-gpu. virtio-gpu, vhost-user-gpu and the display backends (VNC, SPICE, GTK, D-Bus, Cocoa) are in documents 13 and 15.

Modelling: VGA keeps QEMU's split between the register state machine (sequencer, graphics controller, attribute controller, CRTC), the legacy memory window at 0xa0000 with planar and chain-4 modes, and the linear framebuffer in a VRAM RAMBlock. Display updates are driven by the UI refresh timer calling the device's update function, which uses `memory_region_snapshot_and_clear_dirty` on the VRAM with the `DIRTY_MEMORY_VGA` log to redraw only changed lines. The cirrus BitBLT engine (`cirrus_bitblt_start`) has had several guest-to-host CVEs; the port keeps every QEMU bounds check and uses slices, so a range error panics instead of overflowing.

Fast paths: the linear framebuffer is plain guest RAM mapped into the guest, so drawing in any mode with a linear framebuffer costs no exits. The legacy window is MMIO and each access exits; QEMU registers it as coalesced MMIO, and so do we. ruvm's dirty tracking uses the per-RAMBlock bitmap of document 05, synchronized with KVM's dirty log (or dirty ring) only for the VRAM block and only on display refresh.

Testing: display-vga-test.c covers realize and hotplug. For output, a harness boots the same guest on QEMU and ruvm, stops at fixed points, and compares `screendump` output pixel by pixel.

## Audio

`intel-hda` and `ich9-intel-hda` with the codecs `hda-duplex`, `hda-micro` and `hda-output` (hw/audio/intel-hda.c and hda-codec.c), `AC97`, `sb16`, `ES1370`, `adlib`, `gus`, `cs4231a` and the PC speaker. The audio backends (CoreAudio, PipeWire, PulseAudio, ALSA, SDL, the wav capture, none) are in document 15, behind the same `audiodev` interface.

The HDA controller processes its CORB and RIRB command rings (`intel_hda_corb_run`) on register writes and moves audio through buffer descriptor lists by DMA. Stream DMA is paced by the audio backend's callbacks, as in QEMU. ac97-test.c, intel-hda-test.c, es1370-test.c and fuzz-sb16-test.c are the qtests; we add trace diffs for a Linux guest playing a fixed PCM file.

## Input, serial and parallel

PS/2 keyboard and mouse sit behind the i8042 controller (hw/input/pckbd.c, `i8042`, and the MMIO variant for some boards), with the PS/2 devices in hw/input/ps2.c. USB HID devices are above. `virtio-input` is in document 13. Input events from the UI arrive through the QEMU input layer's event model (ui/input.c in QEMU, ported in document 15), which routes to the active handler.

Serial: the 16550 core (hw/char/serial.c) with front ends `isa-serial`, `pci-serial`, `pci-serial-2x`, `pci-serial-4x` and the MMIO `serial-mm` used by boards; `pl011` for Arm; and many SoC UARTs. `serial_ioport_write` and the FIFO timing (the character transmit timer based on the configured baud rate, and the timeout interrupt) are ported exactly, because Linux's 8250 driver probes FIFO depth and interrupt behaviour at boot. The serial console is on the critical path of the boot time targets (15 ms to first instruction and 110 ms to init on microvm): every THR write is a PIO exit on x86. We keep each exit cheap (no global lock, buffered chardev write) and leave guest-visible behaviour alone; virtio-console (document 13) is the fast console. parallel.c (`isa-parallel`) is ported unchanged.

Testing: boot-serial-test.c checks the first bytes printed by firmware on many boards and runs against ruvm. For the 16550 we add a qtest script that walks the FIFO and interrupt identification states and diffs against QEMU.

## TPM

Front ends: `tpm-crb` (hw/tpm/tpm_crb.c) at 0xFED40000 on x86 (`TPM_CRB_ADDR_BASE`), `tpm-tis` on ISA at the same base (`TPM_TIS_ADDR_BASE`), `tpm-tis-device` on sysbus for arm virt, `tpm-tis-i2c` for BMC boards, and `tpm-spapr` on pseries. The physical presence interface (tpm_ppi.c) adds a RAM region that firmware and ACPI methods use. Back ends: the `emulator` type talking to swtpm over a control channel and a data socket, and `passthrough` to a host `/dev/tpmN`.

11.1 adds command chunking to the CRB interface (`cap-chunk` property, default true), and hw_compat_11_0 turns it off for older machines, along with `ppi=off` on `tpm-tis-device`. Both are ported with those defaults.

TPM traffic is small and latency-insensitive, so there are no fast paths. The swtpm protocol is ported from backends/tpm/tpm_emulator.c, including the state blob transfer used for migration (the TPM state travels in the migration stream as blobs fetched from swtpm). tpm-crb-test.c, tpm-tis-test.c, tpm-tis-device-test.c and tpm-tis-i2c-test.c and their `-swtpm` variants run against ruvm; the swtpm variants need swtpm installed on the CI host.

## IPMI, watchdogs and pvpanic

IPMI: `ipmi-bmc-sim` (an in-process BMC simulator, ipmi_bmc_sim.c), `ipmi-bmc-extern` (an external BMC over a chardev), and the system interfaces `isa-ipmi-kcs`, `isa-ipmi-bt`, `pci-ipmi-kcs`, `pci-ipmi-bt` and `smbus-ipmi`. ipmi-kcs-test.c and ipmi-bt-test.c exercise the simulator through both interfaces.

Watchdogs: `i6300esb` (PCI, wdt_i6300esb.c), `ib700` (ISA), `diag288` (s390x, driven by the DIAG 288 instruction), `sbsa_gwdt` (arm), the spapr watchdog and the SoC ones (aspeed, imx2, k230, cmsdk, npcm7xx). Expiry actions follow `-action watchdog=reset|shutdown|poweroff|pause|debug|none|inject-nmi` and emit the `WATCHDOG` QMP event. The ICH9 TCO watchdog is part of the LPC bridge (tco-test.c), and q35's `wdat` table in document 11 exposes it through ACPI. wdt_ib700-test.c, tco-test.c, k230-wdt-test.c, cmsdk-apb-watchdog-test.c and npcm7xx_watchdog_timer-test.c run against ruvm.

pvpanic: `pvpanic` on ISA with I/O port 0x505 by default (the `ioport` property in pvpanic-isa.c), `pvpanic-pci`, and the MMIO variant on some boards. A guest write reports a panic or crash load event, which becomes the `GUEST_PANICKED` or `GUEST_CRASHLOADED` QMP event and triggers the `-action panic=` policy. The port number is also published to firmware in the `etc/pvpanic-port` fw_cfg file. pvpanic-test.c and pvpanic-pci-test.c run against ruvm.

## I2C, SPI and GPIO

The bus frameworks are hw/i2c/core.c with SMBus on top (smbus_slave.c, smbus_master.c), hw/ssi/ssi.c for SPI, and GPIO lines through the qdev GPIO API. Bus controllers are mostly SoC devices: the ICH9 SMBus controller on q35, Aspeed, NPCM, i.MX, BCM2835, STM32 and many others. Devices on the buses include SPD EEPROMs on the PC SMBus (smbus_eeprom.c), sensors under hw/sensor (tmp105, tmp421, emc141x, adm1272, adm1266, isl_pmbus_vr, max34451, lsm303dlhc), RTCs (ds1338, rs5c372, m41t80), GPIO expanders (pca9552, pca9554), and SPI flash (m25p80.c). OpenBMC firmware on the Aspeed and NPCM boards probes them at boot.

They are modelled as state machines driven by bus events (`start_transfer`, `send`, `recv`, `event` for I2C; `transfer` for SSI), which ruvm keeps because board code wires them by these interfaces. There are no fast paths. The qtest coverage here is good: tmp105-test.c, ds1338-test.c, rs5c372-test.c, pca9552-test.c, emc141x-test.c, adm1272-test.c, adm1266-test.c, isl_pmbus_vr-test.c, max34451-test.c, lsm303dlhc-mag-test.c, the aspeed and npcm and stm32 gpio tests, and bcm2835-i2c-test.c. They all run against ruvm unchanged through libqos's i2c driver.

## Proving register-level equivalence

Every device port must show that it behaves like the C device at the register level. Porter-written unit tests are not enough, because the porter's understanding is what is under test. We use four mechanisms, in increasing order of coverage.

### QEMU's qtests, unmodified

tests/qtest has 164 `*-test.c` files (189 C files including helpers) and tests/qtest/libqos has 56 C files. They talk to the emulator over the qtest protocol implemented in system/qtest.c: `readb`/`readw`/`readl`/`readq`, `writeb` through `writeq`, `inb`/`inw`/`inl` and `outb`/`outw`/`outl`, `memread`, `memwrite`, `b64read`, `b64write`, `memset`, `clock_step`, `clock_set`, `irq_intercept_in`, `irq_intercept_out` and `set_irq_in`. ruvm-accel-qtest implements the same protocol, so the test binaries built from the QEMU tree run against `qemu-system-*` symlinks to ruvm with no changes (document 22 has the runner). A device class is not considered ported until every qtest that exercises it passes.

### qtest scripts diffed against QEMU

For devices whose qtests are thin, we write qtest scripts: plain text files of protocol commands, generated by small Rust or Python programs that walk the register map (every register, every access size, reserved bits, write-then-read patterns, reset in the middle) and the documented command sequences. The harness runs each script against QEMU 11.1 and against ruvm, both with `-accel qtest` and the same command line, and requires identical responses line by line. Interrupt line state is included in the output through `irq_intercept_out`. Scripts and recorded QEMU responses are checked in under `tests/qtest-scripts/<device>/` so the check runs without QEMU; a nightly job re-records them.

### Trace diffing

MMIO and PIO accesses from any source go through the memory core, which in QEMU emits `memory_region_ops_read` and `memory_region_ops_write` (system/trace-events: cpu index, region pointer, address, value, size, region name). ruvm emits the same events with the same format, except that the region pointer field is replaced by a stable region id so the logs can be compared. Device trace events keep QEMU's names and arguments, as required above.

The trace diff harness boots the same guest on QEMU and ruvm under TCG with `-icount shift=0,sleep=off` (document 17 covers icount determinism) and `-trace` enabled for the memory core and the device's own events, then aligns the two logs and reports the first divergence with context. Under icount both runs execute the same instruction stream until a device returns a different value, so the first divergence points at the register access where behaviour differs. This is how we test devices that only a full guest exercises well: USB schedules, the SCSI HBAs, HDA, VGA mode setting, the IDE state machines during Linux probing, and the SoC devices on the Tier 2 boards of document 11. When a timer or host backend makes timing nondeterministic, the harness compares per-register access multisets instead of exact order.

### Fuzzing both implementations

QEMU's generic fuzzer (tests/qtest/fuzz/generic_fuzz.c, configured by `QEMU_FUZZ_ARGS` and `QEMU_FUZZ_OBJECTS` and the device configurations in generic_fuzz_configs.h) produces qtest command streams. We run it in differential mode: each input is replayed on QEMU and ruvm, and any difference in responses, interrupt state or DMA writes to guest memory is a failure, as is a panic in ruvm. QEMU's `fuzz-*-test.c` reproducers run in the normal suite. Differential fuzzing runs continuously on the Tier A devices below and on demand for a device under port.

## Porting strategy and the long tail

### Tiers

Devices are ported in the same order as machines in document 11, and a device inherits the highest tier of any machine that uses it.

- Tier A (M2): the devices of microvm and q35: 8259, IOAPIC, LAPIC and the KVM proxies, PIT, HPET, RTC, the q35 MCH and ICH9 LPC with its ACPI, TCO and SMBus, PCIe root ports and switches, MSI and MSI-X, ACPI and native PCIe hotplug, 16550, i8042 and PS/2, AHCI, `VGA`, `bochs-display`, `ramfb`, `qemu-xhci` with HID and storage devices, pvpanic, fw_cfg, `tpm-crb` and `tpm-tis`, NVMe. These are what libvirt's default q35 domain uses.
- Tier B (M5 to M8): i440fx, PIIX3 and PIIX4 with IDE and UHCI, cirrus, the remaining USB controllers and devices, `usb-host`, `usb-redir`, LSI, megasas, mptsas, pvscsi, HDA, AC97, `i6300esb`, `ib700`, IPMI, `sdhci-pci`, UFS, the floppy controller, `pxb` and `pxb-pcie`, SR-IOV, AER, and the GIC, PL011, PL031 and gpex of arm virt; the PLIC, ACLINT, APLIC and IMSIC of riscv virt; XICS, XIVE and spapr devices; s390 flic and channel subsystem devices.
- Tier C (M10 and later): every device only reachable through a Tier 2 or Tier 3 board, which is most of the roughly 800 SoC-internal types.

### How a device is ported

We do not use automatic C-to-Rust translation for device models. c2rust-style output keeps C's aliasing and would need a second rewrite to fit the lock domains and `MmioOps`. A device port is a manual translation that follows a fixed checklist, and a reviewer checks the checklist, not the prose:

1. Properties: `device-list-properties` output identical to QEMU, including descriptions and defaults.
2. VMState: the JSON from `-dump-vmstate` identical for the device's sections (document 17).
3. Trace events: the device's trace-events entries copied, and every call site kept.
4. Registers: the C `MemoryRegionOps` access constraints copied, and a qtest script covering every register checked in.
5. Tests: every QEMU qtest touching the device passes, and the trace diff passes on at least one guest boot that uses it.
6. Compat: every compat property in any `hw_compat_*` or family array naming this device's type or a parent type has a matching property.
7. Security: all DMA through `DmaBuf`, no `unsafe` in the device crate, and the device's `fuzz-*` reproducers pass.

Most SoC devices are small (a register file, a timer, an IRQ output), so the checklist is most of the work and contributors can do it in parallel. The board crates of document 11 list their devices, so a board port is complete when its device list is.

### Tracking coverage

`cargo xtask device-coverage` compares ruvm's type registry against the QOM type list recorded from QEMU 11.1 (the 2501 types above, per binary), and prints per-directory and per-tier counts of ported, in progress and missing types. The same recorded list backs `-device help` parity: ruvm must not list a device QEMU does not have, and the per-binary difference must shrink to zero for each binary before that binary's targets are declared complete in document 23.

## Crate placement

- ruvm-hw-core: `Device` and `MmioOps` traits, `DeviceLock`, `IrqLine`, GPIO and clocks, the qdev bus types, the USB packet model, the PCI bus core (configuration space, BARs, MSI, MSI-X, bridges, hotplug handlers, the PCIe capability helpers including SR-IOV, ARI, AER, ATS, PASID, PRI and DOE).
- ruvm-hw-intc: all interrupt controllers and their accelerator proxies.
- ruvm-hw-timer: PIT, HPET, RTC and SoC timers.
- ruvm-hw-pci: host bridges, root ports, switches, expanders.
- ruvm-hw-usb: host controllers and USB devices, usbredir and libusb passthrough.
- ruvm-hw-storage: IDE, AHCI, SCSI HBAs and the SCSI bus devices, NVMe, SD and eMMC, UFS, floppy, pflash, SPI flash.
- ruvm-hw-display, ruvm-hw-audio, ruvm-hw-input, ruvm-hw-char: their device classes.
- ruvm-hw-tpm: TPM front ends and the emulator and passthrough back ends.
- ruvm-hw-misc: IPMI, watchdogs, pvpanic, and miscellaneous devices.
- ruvm-hw-i2c and ruvm-hw-ssi: the bus frameworks and their devices, with GPIO devices in ruvm-hw-misc.

## Decisions made in this document

1. Every device keeps QEMU's property names and defaults, VMState layout, trace event names and formats, and `MemoryRegionOps` access constraints; these four are checked mechanically for each port.
2. Each device declares a lock domain at realize; IRQ delivery across domains is posted to the target's mailbox and handled before the raising vCPU re-enters the guest.
3. Per-vCPU interrupt controller state (LAPIC, GIC redistributor and CPU interface, IMSIC file) lives in the vCPU's domain and is accessed without a lock from that vCPU.
4. ruvm registers the same KVM coalesced MMIO and PIO ranges as QEMU (RTC index port, 0xcf8, legacy VGA window).
5. NVMe `ioeventfd` keeps QEMU's default (false) on the QEMU-compatible CLI and defaults to true in the native `ruvm run` CLI.
6. Memory core trace events replace the region pointer with a stable region id so QEMU and ruvm logs can be diffed.
7. Register-level equivalence is proved by unmodified QEMU qtests, recorded qtest scripts diffed against QEMU, icount-aligned trace diffing of guest boots, and differential fuzzing with QEMU's generic fuzzer.
8. No automatic C-to-Rust translation for device models; ports follow a seven-point checklist.
9. Devices are tiered A, B and C by the machines that use them, and `cargo xtask device-coverage` tracks ported types against the 2501 QOM types recorded from QEMU 11.1.
