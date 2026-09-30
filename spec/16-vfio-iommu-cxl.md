# 16. VFIO, IOMMUs and CXL: ruvm-hw-vfio, ruvm-hw-iommu, ruvm-hw-cxl

This document covers the three areas where a VM stops being self-contained: VFIO device assignment (real hardware handed to the guest), the virtual IOMMUs that sit between guest devices and guest memory, and CXL memory devices. The reference is QEMU 11.1.0, specifically hw/vfio/, hw/vfio-user/, hw/remote/, backends/iommufd.c, backends/host_iommu_device.c, hw/i386/intel_iommu.c, hw/i386/amd_iommu.c, hw/i386/x86-iommu.c, hw/arm/smmu-common.c, hw/arm/smmuv3.c, hw/arm/smmuv3-accel.c, hw/virtio/virtio-iommu.c, hw/riscv/riscv-iommu*.c, hw/cxl/, hw/mem/cxl_type3.c, hw/pci-bridge/cxl_*.c and hw/acpi/cxl.c, plus the docs cited in each section. This work is milestone M8 (document 23). The compatibility rules from document 02 apply unchanged: every device name, property name, default, QMP command, QMP event and migration section below must match QEMU 11.1.0 for the same machine type.

## Scope and crates

Three L3 crates hold the device models, and one L0 crate holds the kernel interfaces.

| Crate | Layer | License | Contents |
|---|---|---|---|
| ruvm-sys (vfio, iommufd modules) | L0 | MIT OR Apache-2.0 | ioctl numbers and structs generated from linux/vfio.h and linux/iommufd.h, thin safe wrappers, no policy |
| ruvm-vfio-user | L0 | MIT OR Apache-2.0 | vfio-user protocol codec, client transport and server state machine |
| ruvm-hw-vfio | L3 | GPL-2.0-or-later | containers, vfio-pci and its quirks, IGD, display, vfio-ap, vfio-ccw, vfio-user-pci, migration, CPR |
| ruvm-hw-iommu | L3 | GPL-2.0-or-later | intel-iommu, amd-iommu, arm-smmuv3 (with accel), virtio-iommu, riscv-iommu, host IOMMU device objects, iommufd backend object |
| ruvm-hw-cxl | L3 | GPL-2.0-or-later | CXL host bridge glue, fixed memory windows, root, upstream and downstream ports, cxl-type3, switch mailbox CCI, mailbox command set, events |

The split between ruvm-sys and ruvm-hw-vfio follows the canon rule that permissive leaf crates never depend on GPL crates (checked by `cargo xtask provenance`, document 24). The uapi bindings are generated once from a pinned kernel header snapshot and checked in, so a build does not need kernel headers on the host. ruvm-vfio-user is permissive because the protocol codec is useful outside ruvm. The ACPI pieces (DMAR, IVRS, IORT, VIOT, CEDT tables) live in ruvm-hw-acpi next to the other table builders (document 11), with the IOMMU and CXL crates supplying the data.

Two hooks from ruvm-mem (document 05) carry most of the weight: the memory listener, which a VFIO container uses to turn region add and delete events into host DMA map and unmap calls, and the IOMMU memory region, whose `translate` method returns an IOTLB entry for an IOVA, requester id and access type and which fires map and unmap notifiers when the guest invalidates.

## Locking model

QEMU runs nearly all of this under the BQL: interrupt eventfds in the main loop, container map and unmap from memory listener callbacks, and vIOMMU invalidations on the vCPU that wrote the register. ruvm has no BQL (document 03), so the assignments below are explicit.

Each VFIO device is its own device domain. Its lock covers emulated config space, the interrupt routing table and quirk state. Direct BAR access does not take the lock at all, because mmapped BARs are mapped straight into the guest through a KVM memory slot. VFIO exists only on Linux hosts, so KVM is the only accelerator involved. Trapped BAR accesses (quirk windows, the MSI-X table, regions that cannot be mmapped) go through `MmioOps` on the vCPU thread and take the device lock.

A VFIO container is shared by every device attached to it and gets its own lock. Map and unmap calls are made by the memory listener, which in ruvm runs on whichever thread commits the FlatView change, under the control lock for topology changes and under the region owner's lock for RAM discard changes from virtio-mem. The container lock is a leaf lock: nothing else is acquired while it is held, so there is no ordering problem between a vIOMMU invalidation on a vCPU thread and a hotplug on the main thread.

Each vIOMMU instance is one device domain that also contains its register file, its invalidation queue state and its IOTLB. Translation lookups from emulated devices hit a per-vIOMMU IOTLB that is read under RCU and written under the vIOMMU lock, so the common case (a cached translation) does not take a lock.

## VFIO container backends

QEMU 11.1 supports two ways of talking to the kernel. The legacy path opens /dev/vfio/vfio, attaches one or more /dev/vfio/$group files to a container, picks an IOMMU model with VFIO_SET_IOMMU (Type1v2 on x86 and Arm, the sPAPR TCE model on ppc, handled in hw/vfio/spapr.c), and then obtains device fds from the group. The iommufd path, added to QEMU in 9.0, opens /dev/iommu and a per-device character device under /dev/vfio/devices/, binds the device to the iommufd, allocates an IO address space (IOAS) and attaches the device to it. iommufd was merged in Linux 6.2 and the VFIO device character device (cdev) arrived in Linux 6.6. The legacy container is group-centric; iommufd is device-centric, which is what makes nesting, PASID and per-device dirty tracking tractable.

QEMU splits this as a base container (hw/vfio/container.c) with the memory listener and address space handling (hw/vfio/listener.c), a legacy implementation (hw/vfio/container-legacy.c) and an iommufd implementation (hw/vfio/iommufd.c). ruvm mirrors that with a trait.

```rust
pub trait VfioContainer: Send + Sync {
    fn dma_map(&self, iova: u64, size: u64, host: HostPtr, readonly: bool, mr: &MemoryRegion) -> Result<()>;
    fn dma_unmap(&self, iova: u64, size: u64, flags: UnmapFlags) -> Result<()>;
    fn set_dirty_tracking(&self, start: bool) -> Result<()>;
    fn query_dirty_bitmap(&self, iova: u64, size: u64, out: &mut DirtyBitmap) -> Result<()>;
    fn attach_device(&self, dev: &VfioDeviceHandle) -> Result<()>;
    fn detach_device(&self, dev: &VfioDeviceHandle);
    fn iova_ranges(&self) -> &[IovaRange];
    fn pgsizes(&self) -> u64;
}
```

`LegacyContainer` and `IommufdContainer` implement it. The listener logic above the trait is shared: skip non-RAM regions except BARs of other assigned devices (for peer to peer), honor `RamDiscardManager` so only populated virtio-mem blocks are mapped, split mappings to the host page size set reported by the container, and respect the container's usable IOVA ranges. One behavior copied deliberately: with iommufd, mapping another device's BAR fails with EFAULT because the kernel does not yet map MMIO through IOMMU_IOAS_MAP, and QEMU prints a warning (`IOMMU_IOAS_MAP failed: Bad address, PCI BAR?`) rather than failing. ruvm prints the same warning text.

### Selecting a backend

The QEMU rule, which ruvm follows for the QEMU-compatible binaries: a VFIO device uses iommufd only if its `iommufd` link property names an `-object iommufd` instance. Otherwise it uses the legacy container. The iommufd object has one property, `fd`, for a management-opened /dev/iommu descriptor (QAPI `IOMMUFDProperties`, since 9.0). vfio-pci has a matching `fd` property for a pre-opened cdev, so libvirt can run QEMU without access to /dev/vfio at all.

```
-object iommufd,id=iommufd0,fd=22
-device vfio-pci,iommufd=iommufd0,fd=23
```

Decision: `ruvm run` (the native CLI, document 18) defaults to iommufd when /dev/iommu exists and the device has a cdev node, and falls back to the legacy container with a one-line notice otherwise. This is sugar that expands to an explicit `-object iommufd` in the resolved configuration, so the underlying model and the QMP-visible tree are the same as if the user had typed it. The QEMU-compatible binaries never change backend on their own.

One quirk of fd passing carries over: vfio-pci decides whether a device is an mdev by looking at the `sysfsdev` path, and with `fd=` there is no path, so an mdev passed by fd is treated as a physical device and `x-balloon-allowed=on` is rejected. ruvm keeps that behavior, because guessing wrong about RAM discard safety corrupts guest memory.

### Host IOMMU device objects

Since 9.1, QEMU creates a host IOMMU device object for each assigned device (backends/host_iommu_device.c, with legacy and iommufd subclasses). A vIOMMU queries it for the host's capabilities (address width, nesting support, PASID width, errata) before it agrees to put the device behind itself. This is how intel-iommu refuses `fsts=on` with a legacy container device (`Need IOMMUFD backend when fsts=on`) and how arm-smmuv3 with `accel=on` derives its ID register values from the host SMMU. ruvm models it as a `HostIommuDevice` trait object handed to the vIOMMU through the PCI IOMMU ops (`set_iommu_device` and `unset_iommu_device` in QEMU's PCIIOMMUOps), with the error strings copied verbatim.

## The vfio-pci device

vfio-pci (hw/vfio/pci.c, device.c and region.c) makes a real device look like a sane PCI function inside a VM.

Config space is split three ways. Some fields pass straight through to the device fd. Some are fully emulated (BAR registers, the MSI and MSI-X capabilities, the expansion ROM BAR, parts of the PCIe capability) and kept in an emulated copy. Some are virtualized by writing to both. vfio-pci also hides capabilities that do not make sense in a guest and can place virtual ones: `x-vpasid-cap-offset` picks where the emulated PASID capability goes, `skip-vsc-check` (default on since 9.1, off on older machine types) relaxes the vendor-specific capability check, and `x-pci-vendor-id`, `x-pci-device-id`, the sub-ids and `x-pci-class-code` (10.1) override identity. ruvm keeps the same masks, the same emulated bits and the same capability layout, because Windows drivers in particular notice when a capability moves.

BARs are mmapped when the kernel says a region supports it, and those mappings go into the guest as RAM device regions so accesses never exit. `x-no-mmap` forces trapping for debugging. Sparse mmap capabilities are honored so the MSI-X table page can be trapped while the rest of the BAR is mapped. `x-msix-relocation` moves the MSI-X table to a new BAR or a larger region of an existing one, for devices whose table shares a page with registers the guest touches often. Since 9.2, QEMU aligns large BAR mappings so the host can use PMD or PUD pages, and ruvm does the same.

Interrupts: INTx uses an eventfd from the kernel and, when KVM supports it, an irqfd with a resample eventfd, so the level-triggered line never visits userspace in the common case. The awkward part is that while INTx is asserted, the BAR mmaps are disabled so that a guest register access can be trapped to unmask the interrupt. `x-intx-mmap-timeout-ms` (default 1100) controls how long QEMU waits before re-enabling them, and the comment above `vfio_intx_mmap_enable` in pci.c explains the choice. `x-no-kvm-intx` and `intx-interrupt` control this path. MSI and MSI-X vectors each get an eventfd wired to a KVM irqfd (`x-no-kvm-msi`, `x-no-kvm-msix` turn that off). The MSI-X table and PBA are emulated. `x-req` enables the device request interrupt that the host kernel uses to ask for a device back, which QEMU turns into a hot-unplug request.

ioeventfds: vfio-pci can register KVM ioeventfds on quirk registers so writes go straight to the kernel VFIO driver (`x-no-vfio-ioeventfd`, `x-no-kvm-ioeventfd`). This matters for the NVIDIA BAR0 mirror described below, which some drivers hammer.

ruvm-specific design: the eventfd to irqfd plumbing is set up by ruvm-hw-vfio and the accelerator (document 06) together, and when irqfd is not available the eventfds are registered with the ruvm-aio reactor of the device's home iothread rather than a global main loop. The home iothread defaults to the main thread, matching QEMU's placement, so interrupt latency on the fallback path does not change unless the user assigns one. Reset follows QEMU's order: device-specific resets first, then VFIO_DEVICE_RESET when the kernel reports FLR or PM reset, then bus or slot hot reset when every affected device is owned by this VM. Hot reset across multiple devices takes the control lock.

`vf-token` passes the shared secret needed to open a VF whose PF is owned by a userspace driver. `x-balloon-allowed` permits RAM discard for devices that are known not to pin all of guest memory. `ats` (OnOffAuto) exposes the host device's ATS capability to the guest; see the PASID, PRI and ATS section.

## Quirks

hw/vfio/pci-quirks.c exists because drivers, and often the device's own option ROM, reach registers through side doors that bypass the BARs QEMU knows about. ruvm ports every quirk with the same trigger conditions (vendor, device, BAR size) and the same emulated behavior, table-driven where the original is table-driven.

| Quirk | Hardware | What it handles |
|---|---|---|
| ATI 3c3 | AMD/ATI GPUs with VGA | I/O port 0x3c3 returns the upper byte of BAR4's address, which the VBIOS uses to find the I/O BAR |
| ATI BAR4 window | AMD/ATI | an address and data register pair in the I/O BAR that reaches config space and MMIO |
| ATI BAR2 mirror | AMD/ATI | a copy of PCI config space inside the MMIO BAR |
| NVIDIA 3d4/3d0 | NVIDIA GPUs with VGA | a VGA register backdoor into PCI config space and MMIO |
| NVIDIA BAR5 window | NVIDIA | an address and data window in BAR5 that reaches config space |
| NVIDIA BAR0 mirror | NVIDIA | config space mirrored at offsets 0x88000 and 0x1800 in BAR0 |
| RTL8168 | Realtek RTL8168 NICs | a backdoor in BAR2 that reaches the MSI-X table |
| Radeon reset | older AMD GPUs | a device-specific reset sequence where the generic reset leaves the GPU unusable |
| GPUDirect clique | NVIDIA | a vendor capability advertising a peer to peer clique id, set with `x-nv-gpudirect-clique` |
| VMD shadow | Intel VMD | a vendor capability exposing shadow registers of the VMD bus offsets |
| ROM quirk | assorted | devices that report a ROM but must not have it read or need it hidden |

`x-no-geforce-quirks` disables the NVIDIA quirks for GeForce parts. QEMU 10.0 added support for old ATI GPUs (x550), and ruvm includes it. The mirror quirks are the performance-sensitive ones: they trap one page of an otherwise mmapped BAR, and the NVIDIA mirror is also where the ioeventfd registration above pays off.

### Intel IGD

hw/vfio/igd.c handles Intel integrated graphics. IGD is not a clean PCI device: it uses an OpRegion (a host memory table holding the Video BIOS Table) and Data Stolen Memory, and legacy boot code expects it at 00:02.0 with a matching LPC bridge at 00:1f.0. docs/igd-assign.txt lists the conditions each guest software stack needs. The properties are `x-igd-opregion` (copy the host OpRegion to the guest through fw_cfg), `x-igd-lpc` (copy host LPC and host bridge ids, i440fx only), `x-igd-gms` (stolen memory size), `x-igd-legacy-mode` (OnOffAuto) and `x-vga`. Legacy mode turns on OpRegion, LPC ids and VGA ranges automatically for generations 6 to 9 on i440fx at 00:02.0 with a ROM present; `x-igd-legacy-mode=on` makes the checks fatal. QEMU 9.2 added support for generation 11 and later, 10.0 improved generations 11 and 12, and 10.1 enables the OpRegion automatically where it is needed and fixed OpRegion detection. ruvm matches the 11.1 behavior; the table mapping device ids to generations is data regenerated from igd.c on each rebase.

## Display passthrough and mdev

Mediated devices (mdev) are virtual functions created by a host driver, selected with `sysfsdev=/sys/bus/mdev/devices/<uuid>`. Intel GVT-g, NVIDIA vGPU and the s390 AP and CCW drivers use this. From vfio-pci's point of view an mdev is a device fd like any other, with two differences: it can expose extra region types, and RAM discard is safer because mdev drivers pin only what they use.

`display=` (OnOffAuto, default off) turns on hw/vfio/display.c, which queries the device for graphics planes. Two kernel mechanisms exist: a dma-buf per plane (VFIO_DEVICE_QUERY_GFX_PLANE plus VFIO_DEVICE_GET_GFX_DMABUF, used by GVT-g), which feeds straight into an OpenGL-capable UI, and a region-based framebuffer for simpler drivers. The EDID region lets QEMU set the virtual monitor's modes; `xres` and `yres` set the initial ones. In ruvm these become console surfaces in the console model of document 15.

`ramfb=on` adds a ramfb boot display next to the vGPU so firmware and early boot have something to draw on before the guest driver loads. It is a property of the non-hotpluggable variant, vfio-pci-nohotplug, together with `x-ramfb-migrate` and `use-legacy-x86-rom` (10.1; forced to true on q35 and on older machine types through compat properties).

## vfio-ap and vfio-ccw

These are the s390x members of the family. vfio-ap (hw/vfio/ap.c, docs/system/s390x/vfio-ap.rst) passes an AP matrix (adapters, domains and control domains of the crypto express cards) to the guest. It is an mdev with no DMA mappings of its own; the guest issues AP instructions that KVM handles with the configuration the vfio_ap driver installs. QEMU wires two interrupts: the request interrupt (the host wants the device back) and, since 10.1, the configuration change interrupt, which QEMU queues and reports to the guest so it can rescan.

vfio-ccw (hw/vfio/ccw.c, docs/system/s390x/vfio-ccw.rst) passes a channel subsystem device, typically a DASD. The guest's channel programs are handed to the kernel, which translates and runs them. Properties are `sysfsdev`, `iommufd`, `force-orb-pfch` and the inherited `loadparm` for IPL. It has no BARs or config space but still needs a container, which is why the container code cannot assume PCI.

For s390x PCI passthrough, zPCI adds its own layer (hw/s390x/s390-pci-vfio.c, docs/system/s390x/pcidevices.rst) that reads the host's function characteristics through VFIO capability chains. That lives in ruvm-target-s390x's machine code and uses ruvm-hw-vfio as a library.

The removed platform devices are not ported. vfio-platform, vfio-calxeda-xgmac and vfio-amd-xgbe were deprecated in 10.0 and removed in 10.2, so they do not exist in 11.1 and ruvm rejects them with QEMU's "not a valid device model name" error.

## VFIO migration

VFIO migration moves the internal state of an assigned device (its hidden registers, queues and on-device memory) alongside guest RAM. The kernel side is vendor-specific (mlx5, hisi_acc, pds and others), and QEMU drives it through a generic state machine in hw/vfio/migration.c and hw/vfio/migration-multifd.c.

### Protocol history and the state machine

The v1 protocol was removed in QEMU 8.0 in favor of the v2 protocol, which uses VFIO_DEVICE_FEATURE with VFIO_DEVICE_FEATURE_MIGRATION and VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE to move the device between states and returns a data fd for reading or writing state. VFIO migration stopped being experimental in 8.1, which also added precopy. 8.2 added peer to peer support, 9.1 added the `VFIO_MIGRATION` QMP event, 10.0 added multifd transfer and included VFIO state in `calc-dirty-rate`, 10.1 extended multifd to aarch64, and 11.1 made `query-migrate` count VFIO state in its downtime estimate.

The states (from the kernel uapi and docs/devel/migration/vfio.rst) are STOP, RUNNING, STOP_COPY, RESUMING, RUNNING_P2P, PRE_COPY and PRE_COPY_P2P. RUNNING_P2P is a quiescent state in which the device still accepts peer to peer accesses but initiates no new DMA; QEMU moves every device there before any is stopped, so one device's final state does not depend on another device's in-flight DMA. PRE_COPY lets the device stream most of its state while running, and VFIO_DEVICE_FEATURE_MIG_DATA_SIZE and the VFIO_MIG_GET_PRECOPY_INFO ioctl report how much is left. The `VFIO_PRECOPY_INFO_REINIT` flag tells QEMU that the device's initial data has changed and needs to be sent again.

The QAPI enum `QapiVfioMigrationState` (qapi/vfio.json) exposes `stop`, `running`, `stop-copy`, `resuming`, `running-p2p`, `pre-copy`, `pre-copy-p2p`, and since 11.0 `pre-copy-p2p-prepare`. The prepare variants mean the device is about to enter a state, not that it will. The `VFIO_MIGRATION` event carries `device-id`, `qom-path` and `device-state` and is only emitted when `migration-events=on`. ruvm emits the same events in the same order for the same transitions, which the migration interop tests in document 22 check against QEMU event logs.

### Properties

`enable-migration` (OnOffAuto) decides whether a device without dirty tracking support may still migrate. `x-pre-copy-dirty-page-tracking` controls dirty tracking during precopy, `x-device-dirty-page-tracking` controls whether device dirty tracking is used when available, `x-migration-multifd-transfer` sends device state over multifd channels, `x-migration-load-config-after-iter` loads the device config space after iterative data on load, and `x-migration-max-queued-buffers-size` bounds the receive side buffering for multifd. Devices without migration support add a migration blocker, as in QEMU.

### Dirty tracking

Device DMA dirties guest memory behind the CPU's back, so the migration dirty bitmap (document 17) needs another source. QEMU tries, in order:

1. Device dirty tracking: the device itself logs the pages it writes, through VFIO_DEVICE_FEATURE_DMA_LOGGING_START, DMA_LOGGING_REPORT and DMA_LOGGING_STOP. Used when every device in the container supports it and `x-device-dirty-page-tracking` is on. QEMU passes the ranges to track, merging them into a bounded number of ranges when the device can only track a few.
2. IOMMU dirty tracking: the host IOMMU marks written pages in its page table entries (the dirty bit in Intel second-stage, AMD and SMMUv3 tables). With the legacy container this is VFIO_IOMMU_DIRTY_PAGES; with iommufd, QEMU 9.1 added support by allocating the hardware page table with IOMMU_HWPT_ALLOC_DIRTY_TRACKING and then calling IOMMU_HWPT_SET_DIRTY_TRACKING and IOMMU_HWPT_GET_DIRTY_BITMAP (see `iommufd_set_dirty_page_tracking` in hw/vfio/iommufd.c).
3. Neither: every mapped page is treated as dirty on every iteration. Migration still works but cannot converge until the device stops, so precopy is only useful for the device's internal state.

Unmaps during migration need care: when the guest unmaps an IOVA range under a vIOMMU while tracking is on, QEMU reads the dirty bits for that range first so they are not lost (the `IOMMU_HWPT_GET_DIRTY_BITMAP_NO_CLEAR` path in iommufd.c). ruvm keeps that order.

ruvm design: the migration core in ruvm-migration pulls dirty bitmaps from registered sources, and a VFIO container is one source. The bitmap query runs on a migration worker thread, not a vCPU, and takes only the container lock. The section names and stream layout written by `vfio_save_*` are byte-compatible with QEMU, so migration between QEMU and ruvm works with assigned devices whenever the kernel drivers on both hosts accept the device data.

### Multifd and the stream

With `x-migration-multifd-transfer`, device state is split into buffers that travel on the multifd channels and are reassembled in order on the destination, while config space still goes through the main channel. In ruvm the receive side is a small reorder buffer per device bounded by `x-migration-max-queued-buffers-size`, drained by a dedicated load thread as QEMU does. The load thread uses blocking writes to the device data fd; io_uring for that fd is an open item.

## Live update (CPR)

CPR, for checkpoint and restart, is how you update the QEMU binary under a running VM without a full migration. It went through several stages: `cpr-reboot` mode first, `cpr-transfer` mode in 10.0, VFIO support in 10.1 (hw/vfio/cpr.c, cpr-legacy.c, cpr-iommufd.c), and `cpr-exec` mode in 10.2. The `MigMode` values are `normal`, `cpr-reboot`, `cpr-transfer` and `cpr-exec`. docs/devel/migration/CPR.rst is the reference.

The idea is that the new process inherits file descriptors from the old one, so guest RAM stays in place (it must be shared memory, `share=on` on memory backends and `-machine aux-ram-share=on` so implicit RAM blocks such as VGA and ROMs are memfd-backed) and VFIO devices keep their kernel state, including pinned pages, without the device being reset or needing migration support. In cpr-transfer the fds travel over a second `-incoming` channel of type `cpr`, which must be a UNIX socket for SCM_RIGHTS. In cpr-exec the old process execs the new one with the command in the `cpr-exec-command` migration parameter, and the CPR state is serialized to a memfd whose number is passed in the `QEMU_CPR_EXEC_STATE` environment variable. cpr-reboot supports VFIO only if the guest is suspended to RAM first.

The VFIO specifics: with the legacy container, QEMU invalidates the host virtual addresses of all DMA mappings with VFIO_DMA_UNMAP_FLAG_VADDR before handing over, and the new process supplies new addresses with VFIO_DMA_MAP_FLAG_VADDR, which needs the VFIO_UPDATE_VADDR extension. With iommufd, the new process takes ownership of the IOAS with IOMMU_IOAS_CHANGE_PROCESS, and the saved state records the device id, IOAS id and hardware page table id for each device. MSI and MSI-X eventfds and the INTx eventfds are also preserved and rewired, so the new process does not have to renegotiate vectors with the kernel.

ruvm design: CPR state is a set of named fds plus a small VMState-encoded record, both defined so that QEMU 11.1 to ruvm and ruvm to QEMU 11.1 CPR transfers work. That is harder than normal migration interop, since the receiver must understand the sender's fd layout exactly, so cpr-transfer from QEMU to ruvm is the first target. Every place ruvm holds a kernel object that should survive CPR (VFIO device fds, container or iommufd fds, KVM irqfds, the memfds behind RAM) registers it with a `CprFdRegistry` in ruvm-migration at creation time. vhost fd transfer stays post-1.0 as decided in document 13.

## vfio-user

vfio-user is the VFIO device model spoken over a UNIX socket instead of ioctls on a kernel fd, specified in docs/interop/vfio-user.rst. It lets a separate process emulate a PCI device (SPDK's NVMe controller is the usual example) with the guest seeing an ordinary PCI function. Messages mirror the VFIO ioctls: version negotiation, device info, region info and access, interrupt setup, DMA map and unmap (with fds for shared memory so the server can access guest RAM directly), and migration.

### Client

QEMU gained a vfio-user client in 10.1, in hw/vfio-user/ (container.c, device.c, pci.c, proxy.c, protocol.h). The device is `vfio-user-pci` with a `socket` property:

```
--device '{"driver": "vfio-user-pci", "socket": {"path": "/tmp/vfio-user.sock", "type": "unix"}}'
```

It reuses vfio-pci's config space and interrupt code on top of a container that forwards map and unmap to the server rather than to the kernel. Extra properties are `x-send-queued`, `x-msg-timeout` (default 5000 ms) and `x-no-posted-writes`, plus the same identity overrides as vfio-pci. tests/functional/x86_64/test_vfio_user_client.py covers it.

In ruvm the client transport is in ruvm-vfio-user and the device is in ruvm-hw-vfio as a second implementation of the device fd abstraction. The socket belongs to the device's home iothread reactor. Posted writes (BAR writes sent without waiting for a reply, the default) stay ordered against later reads on the same socket, as the protocol guarantees.

### Server

QEMU's server side is `-object x-vfio-user-server` (7.1, hw/remote/vfio-user-obj.c), which only works with the `x-remote` machine: you start a second QEMU with `-machine x-remote,vfio-user=on`, put a PCI device in it, and point the server object at it with `type=unix,path=...,device=<id>`. It is built on libvfio-user, which QEMU pulls in as a meson subproject. The rest of the multiprocess code is in hw/remote/.

Decision: ruvm implements the server natively in ruvm-vfio-user rather than linking libvfio-user through FFI. The protocol is small and this removes a C dependency that only this feature needed. The `x-remote` machine and `x-vfio-user-server` object keep their names and properties, and the server maps the client's shared memory fds as libvfio-user does.

## Virtual IOMMUs

A vIOMMU gives the guest an IOMMU to program. That is needed for guest kernel DMA protection, for nested device assignment (a guest assigning a device to its own guest or to userspace with VFIO), for guest shared virtual addressing, and for large VMs where x2APIC needs interrupt remapping. Every vIOMMU in ruvm implements two things: the IOMMU memory region interface (translate plus notifiers) for emulated devices, and the PCI IOMMU ops for assigned devices, which either shadow the guest tables into the host through map and unmap (the classic path) or hand the guest's first-stage tables to the host IOMMU for nested translation (the accelerated path).

The shadow path works like this. For each assigned device behind a vIOMMU, the VFIO container listens on the device's IOMMU address space rather than on system memory. When the guest maps or unmaps in its IOMMU tables and invalidates, the vIOMMU sends map and unmap notifications, and the container replays them into the host. This requires the guest to invalidate on map as well as unmap, which real hardware does not require; Intel calls it caching mode, and QEMU needs `caching-mode=on` on intel-iommu for assigned devices behind it. The cost is a VM exit per invalidation, which is why nested translation matters for DMA-heavy guests.

### intel-iommu

hw/i386/intel_iommu.c emulates Intel VT-d on q35. Properties (11.1):

| Property | Default | Meaning |
|---|---|---|
| `version` | 0 | reported VT-d version |
| `eim` | auto | extended interrupt mode, needed for x2APIC above 255 CPUs |
| `aw-bits` | 48 | guest address width; 39 on pc machine types 9.1 and older |
| `caching-mode` | off | required for assigned devices on the shadow path |
| `scalable-mode` | off | scalable mode (PASID-granular translation) |
| `fsts` | off | first-stage translation nested in the host through iommufd |
| `snoop-control` | off | report snoop control |
| `pasid-bits` | 0 | PASID width exposed to the guest |
| `svm` | off | shared virtual memory (needs `dma-translation` and device IOTLB) |
| `stale-tm` | off | stale translation mark handling for device IOTLB invalidation; on for pc machine types 9.1 and older |
| `fs1gp` | on | 1 GiB pages in first-stage tables |

The shared x86 properties in hw/i386/x86-iommu.c are `intremap` (OnOffAuto), `device-iotlb` (off; needed for ATS) and `dma-translation` (on). Interrupt remapping works with the split irqchip only, as in QEMU.

Scalable mode is the modern VT-d layout, with a PASID directory and PASID table per device instead of a single context entry, and it is what a guest needs for PASID. `fsts=on` (first-stage translation, implemented with hw/i386/intel_iommu_accel.c) is the nesting path: QEMU allocates a nested hardware page table in the host iommufd with the guest's first-stage table pointer, so the host IOMMU walks the guest tables directly and QEMU only handles invalidations. It requires the iommufd backend, rejects devices behind a conventional PCI bridge (`Host device downstream to a PCI bridge is unsupported when fsts=on`), and has an erratum rule for hosts with ERRATA_772415_SPR17. ruvm matches those checks and messages.

### amd-iommu

hw/i386/amd_iommu.c emulates the AMD IOMMU, which appears as a PCI function plus the IVRS ACPI table. Properties are `xtsup` (x2APIC interrupt remapping support), `pci-id` and `dma-remap` (default off). With `dma-remap=on` the emulated IOMMU does DMA translation for assigned devices through the shadow path; otherwise it is used for interrupt remapping and for emulated devices. ruvm follows QEMU's feature set and does not add AMD nesting until QEMU has it.

### arm-smmuv3

hw/arm/smmuv3.c and smmu-common.c emulate the Arm SMMUv3 on the virt and sbsa-ref machines. History: stage 2 in 8.1, nested two-stage translation in 9.1, two-stage on virt and sbsa-ref in 9.2, multiple user-created instances on virt in 10.2 (`-device arm-smmuv3,primary-bus=pcie.N`, not allowed together with the machine-wide `iommu=smmuv3`), accelerated nesting with `accel=on` in 11.0, and Tegra241 CMDQV with `cmdqv=on` in 11.1.

Properties: `stage` (default stage 1), `identifier` (for IORT and, for CMDQV, DSDT generation), `accel`, `msi-gpa` (guest physical address of the MSI doorbell for accel), and the OnOffAuto or auto-mode properties `ril`, `ats`, `oas`, `ssidsize` and `cmdqv`. With `accel=off`, auto resolves to RIL on, ATS off, OAS 44 and SSID size 0. With `accel=on`, QEMU derives them from the host SMMU's capabilities through the host IOMMU device object.

Accelerated mode (hw/arm/smmuv3-accel.c) makes the guest's stage-1 tables walked by the physical SMMU, with the host providing stage 2. QEMU creates an iommufd vIOMMU object of type IOMMU_VIOMMU_TYPE_ARM_SMMUV3, allocates nested hardware page tables from guest stream table entries, forwards guest invalidations through the vIOMMU object, and reads host fault events through a vEVENTQ to inject into the guest event queue. It requires the iommufd backend and ACPI boot (not device tree). With `cmdqv=on` on NVIDIA Tegra241 hosts, each accelerated SMMU gets hardware command queues the guest writes directly, so invalidation commands stop trapping. ruvm ports both paths; the CMDQV path depends on host kernel support that we test only on hardware we have access to, which document 22 notes.

As in QEMU, the software table walker (for emulated devices and the non-accelerated path) is separate from the accel backend, and its IOTLB keying (ASID, VMID, IOVA, granule and level) follows QEMU so invalidation semantics match.

### virtio-iommu

hw/virtio/virtio-iommu.c and virtio-iommu-pci.c implement the paravirtual IOMMU from the VIRTIO specification. Properties: `primary-bus`, `boot-bypass` (default on), `granule` and `aw-bits` (64 in the device, with machine defaults of 39 on q35 and 48 on arm virt), plus `reserved-regions` on the PCI proxy. On arm virt the topology is described by the VIOT ACPI table (since 7.0) or device tree; on x86 by VIOT. It is the one vIOMMU that works on every architecture, which makes it the natural choice on HVF and WHPX hosts. Queues follow document 13; translation lives in ruvm-hw-iommu. For assigned devices it uses the shadow path, and QEMU's host IOMMU device checks for page size and IOVA range compatibility are copied.

### riscv-iommu

hw/riscv/riscv-iommu.c implements the RISC-V IOMMU specification, as `riscv-iommu-pci` (9.2, with virt machine support) and the platform `riscv-iommu-sys` (10.0, enabled on virt with `iommu-sys=on`). Properties: `version`, `pas-bits`, `bus`, `ioatc-limit`, `intremap`, `ats`, `off`, `s-stage`, `g-stage`, `hpm-counters` and `downstream-mr`. It supports single and two-stage translation (S-stage and G-stage), MSI remapping through MSI page tables, a command queue, fault queue and page request queue, and hardware performance monitoring counters (riscv-iommu-hpm.c). 10.0 added a translation tag for the page table cache and bypass for PCI devices, 10.2 fixed the nested walk through the process directory table, and 11.1 brought a series of compliance fixes (IPSR.PMIP as write-one-to-clear, the FSC SV32 check, CMD_ILL requeue behavior, interrupt overflow accounting). ruvm ports the 11.1 behavior. QEMU 11.1 has no accelerated nesting for RISC-V and ruvm does not add one.

## PASID, PRI and ATS

These three PCIe features let a device participate in address translation. PASID tags a transaction with a process address space id so one function can DMA into several address spaces. ATS lets a device ask the IOMMU for a translation and cache it in a device IOTLB. PRI lets a device ask the OS to fault in a page it could not translate. Shared virtual addressing needs all three.

QEMU provides the capabilities in hw/pci/pcie.c (`pcie_pasid_init`, `pcie_pri_init`, and the ATS capability) and the device-side request API in hw/pci/pci.c: `pci_ats_request_translation` and `pci_pri_request_page`, both taking a PASID, plus notifier registration so a device learns about invalidations of its cached translations. `MemTxAttrs` carries the PASID alongside the requester id, and in ruvm that is the `AccessCtx` of document 05, so a PASID-tagged DMA from an emulated device flows through the same translate path with the PASID selecting the table.

On the emulated side, intel-iommu implements device IOTLB invalidation (with `device-iotlb=on`) and page request handling for `svm=on` in scalable mode. On the passthrough side the host does the real work: vfio-pci's `ats` property and `x-vpasid-cap-offset` expose the host device's capabilities to the guest, and nested translation (`fsts=on` or SMMUv3 `accel=on`) lets the host IOMMU use the guest's PASID tables. Page requests from a physical device under nesting are delivered by the host as fault events and relayed to the guest's page request queue. riscv-iommu has ATS and a page request queue in the emulated model.

ruvm keeps the capability layouts and emulated behavior exactly. Device IOTLB entries held by emulated devices are pinned by IOTLB references (the rule document 13 adopted for virtio), so an invalidation completes to the guest only after in-flight users of the translation finish. QEMU gets this from the BQL.

## CXL

Compute Express Link is PCIe with added coherent memory semantics. QEMU emulates a CXL 2.0 and later topology so that the Linux CXL stack (and CXL-aware firmware) can be developed and tested without hardware. ruvm treats it as a development tool: functional parity, exact guest-visible registers, no performance targets.

### Topology

A CXL setup in QEMU has these pieces:

- CXL Fixed Memory Windows (CFMWs, type `cxl-fmw`), host physical address ranges that the platform promises to route to CXL host bridges, with an interleave across one or more bridges. They are described to the guest in the CEDT ACPI table (hw/acpi/cxl.c). Machine properties are `cxl=on` and `cxl-fmw.N.targets.M`, `cxl-fmw.N.size` and `cxl-fmw.N.interleave-granularity`.
- CXL host bridges, created with `pxb-cxl` on a PCIe host, whose component registers live in a machine region (`cxl_host_reg` on arm virt; see `cxl_hook_up_pxb_registers` in hw/i386/pc.c for q35).
- Root ports (`cxl-rp`, hw/pci-bridge/cxl_root_port.c).
- Switches built from one `cxl-upstream` port and several `cxl-downstream` ports (hw/pci-bridge/cxl_upstream.c and cxl_downstream.c), with a single virtual hierarchy as docs/system/devices/cxl.rst describes.
- Type 3 memory devices (`cxl-type3`, hw/mem/cxl_type3.c).
- A switch mailbox CCI (`cxl-switch-mailbox-cci`, hw/cxl/switch-mailbox-cci.c), a PCI function in the switch that exposes the switch's command interface.

HDM (host-managed device memory) decoders at each level route and interleave accesses; guest software programs them to build regions inside the CFMWs. hw/cxl/cxl-component-utils.c emulates the component register blocks (HDM decoders, RAS, link, security) and cxl-device-utils.c the device register blocks (device status, mailbox, memory device status). CXL on arm virt arrived in 10.1, with a fixed MMIO region for host bridge registers and the fixed memory windows placed above RAM.

### cxl-type3

Properties at 11.1: `memdev` (deprecated alias for persistent), `persistent-memdev`, `volatile-memdev`, `lsa` (label storage area backend), `sn` (serial number), `cdat` (a file to load as the CDAT table instead of the generated one, served by hw/cxl/cxl-cdat.c through DOE), `num-dc-regions` and `volatile-dc-memdev` for dynamic capacity, and link properties `x-speed`, `x-width`, `x-256b-flit`, plus `hdm-db` (back-invalidate HDM decoders, which requires 256 byte flit mode). cxl_type3.c is about 2,500 lines.

Guest accesses to CXL memory go through the CFMW region, which decodes host bridge, root port, switch and device HDM decoders in turn to find the device physical address, then read or write the backend. QEMU decodes on every access, which is why CXL memory is slow under emulation. ruvm caches the decoded route per 256 MiB window and drops the cache whenever any HDM decoder commits or resets, which the guest cannot observe.

### Mailbox and commands

hw/cxl/cxl-mailbox-utils.c implements the CCI command set. The groups ruvm must match: event records (get, clear, interrupt policy), firmware update and activation, timestamp, logs (supported logs, get log, CEL), features (get supported, get and set feature, including the patrol scrub and ECS maintenance features), maintenance operations, identify, partition info and set partition, LSA get and set, sanitize and secure erase, media operations, the poison list with inject, clear and scan media, dynamic capacity configuration, extent list, add response and release, and on switches identify switch device, physical port state, port statistics and tunnel management. Commands can arrive over the PCI mailbox, the switch mailbox CCI, or MCTP. Background commands (sanitize, scan media) run on a timer and report completion through the background command status register; ruvm runs them on the device's iothread with the same observable timing.

ruvm generates the command effects log and the opcode dispatch from one table, so the CEL always matches the implemented commands.

### Events and error injection

QMP commands in qapi/cxl.json: `cxl-inject-general-media-event`, `cxl-inject-dram-event`, `cxl-inject-memory-module-event` and `cxl-inject-poison` (all 8.1), `cxl-inject-uncorrectable-errors` and `cxl-inject-correctable-error` (8.0), and `cxl-add-dynamic-capacity` and `cxl-release-dynamic-capacity` (9.1). Injected events land in the device's event logs (hw/cxl/cxl-events.c) and raise the configured MSI or MSI-X interrupt; RAS errors set the component RAS registers and raise AER. ruvm matches their argument checking and error messages because test scripts written against QEMU depend on them.

### Dynamic capacity devices

A DCD exposes regions of device memory whose backing extents come and go at runtime under a fabric manager's control. In QEMU the fabric manager's side is the QMP pair above: `cxl-add-dynamic-capacity` offers extents (with a selection policy and a tag) to the host, and the guest accepts them through the mailbox; `cxl-release-dynamic-capacity` asks for them back, and the guest releases them. The device tracks pending and accepted extent lists and a bitmap of backed blocks, and an access to an unbacked DPA range fails as a transaction error (`MEMTX_ERROR` in `cxl_type3_read`). `num-dc-regions` sets the region count and `volatile-dc-memdev` backs all of them. ruvm keeps the same extent list limits and ordering, since the guest driver sees them.

## ACPI and firmware

The vIOMMU and CXL crates provide table content to ruvm-hw-acpi: DMAR for intel-iommu, IVRS for amd-iommu, IORT for SMMUv3 (with per-instance entries and, for CMDQV, DSDT devices), VIOT for virtio-iommu, and CEDT plus the _OSC and host bridge objects for CXL. QEMU's tests/data/acpi blobs (for example IORT.smmuv3-dev, IORT.smmuv3-legacy and CEDT.cxl) are byte-level acceptance tests, as document 11 requires.

## Security

The container or iommufd isolates DMA, but the ruvm process holds device fds and can reprogram the device, so the sandboxing rules of document 19 matter more here. The rules:

- ruvm never opens /dev/vfio or /dev/iommu in the QEMU-compatible binaries when fds are passed, and the seccomp profile for a VM with only fd-passed devices does not allow open on those paths.
- Quirk handlers validate every guest-controlled offset against the window size before touching device config space, and are fuzzed (document 22) with a mock device fd.
- VFIO on confidential guests (TDX and SEV-SNP, supported in QEMU since 10.1) needs guest_memfd-backed RAM to be mapped for DMA only while shared. The container follows private and shared conversions from ruvm-mem, as QEMU does through the RAM discard manager (document 19).
- vfio-user servers are untrusted. The client bounds every reply length and every region access against the region info the server advertised at connect time, and a malformed reply fails the device, not the VM.

## Testing

The QEMU tests we run against ruvm unchanged: tests/qtest/intel-iommu-test.c, iommu-intel-test.c, iommu-intel-inv-test.c, amd-iommu-test.c, iommu-smmuv3-test.c, riscv-iommu-test.c, iommu-riscv-test.c, virtio-iommu-test.c and cxl-test.c, plus the libqos drivers they use, and the functional tests tests/functional/x86_64/test_intel_iommu.py, tests/functional/aarch64/test_smmu.py and tests/functional/x86_64/test_vfio_user_client.py. QEMU 11.1 also ships `iommu-testdev` (hw/misc/iommu-testdev.c, docs/specs/iommu-testdev.rst), a test-only PCI device that issues DMA through an IOMMU without any guest; ruvm implements it because it lets the vIOMMU walkers be tested without booting anything.

VFIO needs hardware. The CI plan (document 22) has three tiers. Tier one uses the kernel's sample mdev drivers (mtty, mdpy, mbochs) in a nested VM, which covers the container code, config space, interrupts, the region display path and CPR without real devices. Tier two runs on a small hardware pool with an SR-IOV NIC that supports VFIO migration (mlx5 class), an NVIDIA GPU, an AMD GPU and an Intel IGD host, and exercises quirks, both dirty tracking methods, and migration between QEMU and ruvm in both directions. Tier three is best-effort: Arm hosts with SMMUv3 nesting, Grace hosts for CMDQV, and s390x for vfio-ap and vfio-ccw through a partner.

CXL is tested entirely in emulation: tests/qtest/cxl-test.c, the ACPI table comparisons above, a Linux guest running the ndctl `cxl` tool against the emulated topology, and a differential test that replays the same mailbox command sequences against QEMU and ruvm and compares outputs byte for byte.

## Open items

These are tracked in document 25.

- Whether cross-implementation CPR (QEMU process handing fds to ruvm) can be supported for all device types or only VFIO and RAM.
- How to get repeatable CI access to Tegra241 CMDQV and Arm nesting hardware.
- Whether to put VFIO device data fd reads on io_uring for multifd once kernel drivers are known to handle it.
- Whether HVF or WHPX hosts should get any form of device assignment. Neither has a VFIO equivalent, so assignment is Linux-only in 1.0.
