# 13. Virtio and vhost: ruvm-hw-virtio

This document specifies how ruvm implements VIRTIO devices, their transports, the vhost family of offload protocols, and the queue processing fast path. The reference is QEMU 11.1.0 (hw/virtio/, hw/net/virtio-net.c, hw/block/virtio-blk.c, hw/scsi/virtio-scsi.c, hw/display/virtio-gpu*.c, hw/s390x/virtio-ccw*.c, docs/interop/vhost-user.rst) and the OASIS VIRTIO specification. Virtio is where most real workloads spend their I/O time under KVM, HVF and WHPX, so it is also where the canon performance target "virtio-blk and virtio-net equal or better than QEMU with iothreads and vhost" (document 21) is won or lost. Guest-visible behavior, feature bit defaults per machine version, and the migration stream must match QEMU exactly; everything behind that line is ours to redesign.

## Spec baseline

QEMU does not implement a specification version, it implements feature bits, and so does ruvm. The spec documents we track are VIRTIO 1.2 (Committee Specification 01, July 2022), the VIRTIO 1.3 draft (CSD01, October 2023, which never became a Committee Specification), and VIRTIO 1.4, which was published as Committee Specification 01 on 8 April 2026 and supersedes 1.2 ([OASIS VIRTIO 1.4 CS01](https://docs.oasis-open.org/virtio/virtio/v1.4/cs01/virtio-v1.4-cs01.html), [1.3 CSD01](https://docs.oasis-open.org/virtio/virtio/v1.3/csd01/virtio-v1.3-csd01.html)).

The transport-level feature bits we care about, with their bit numbers from the 1.4 text:

| Bit | Name | QEMU 11.1 status | ruvm |
|---|---|---|---|
| 28 | VIRTIO_F_INDIRECT_DESC | default on (`indirect_desc`) | same |
| 29 | VIRTIO_F_EVENT_IDX | default on (`event_idx`) | same |
| 32 | VIRTIO_F_VERSION_1 | modern transports | same |
| 33 | VIRTIO_F_ACCESS_PLATFORM | `iommu_platform`, forced for confidential guests | same |
| 34 | VIRTIO_F_RING_PACKED | `packed`, default off | same default, fully supported |
| 35 | VIRTIO_F_IN_ORDER | `in_order`, default off | same |
| 36 | VIRTIO_F_ORDER_PLATFORM | set with vhost-vdpa hardware | same |
| 37 | VIRTIO_F_SR_IOV | not offered by emulated devices | not offered |
| 38 | VIRTIO_F_NOTIFICATION_DATA | `notification_data` since 9.1, needs ioeventfd off | same, plus a fast path (below) |
| 39 | VIRTIO_F_NOTIF_CONFIG_DATA | not implemented | not offered by default |
| 40 | VIRTIO_F_RING_RESET | `queue_reset` | same |
| 41 | VIRTIO_F_ADMIN_VQ | not implemented | opt-in, see below |
| 43 | VIRTIO_F_SUSPEND | not implemented | not offered by default |

Rule: every feature bit ruvm offers by default must be one QEMU 11.1 offers by default for the same machine type and device properties, because Linux and Windows drivers negotiate features at probe time and the negotiated set is part of the migration stream (`guest_features` in `vmstate_virtio`). Features QEMU lacks are available only behind `x-` properties and are never enabled by a versioned machine type.

## The VirtioDevice trait

QEMU splits a virtio device into a `VirtIODevice` (hw/virtio/virtio.c) that knows features, config space and queues, a `VirtioBusState` (hw/virtio/virtio-bus.c) that connects it to a proxy, and a transport proxy (`VirtIOPCIProxy`, `VirtIOMMIOProxy`, `VirtioCcwDevice`) that owns the registers. The device and the proxy are two QOM objects; `-device virtio-net-pci` creates the proxy which creates the device as a child. ruvm keeps that composition because it is visible through QOM paths (`/machine/peripheral/net0/virtio-backend`) that libvirt and management scripts read via `qom-get`, and because the QOM tree layout is part of the compatibility contract in document 02. What changes is the Rust shape of the device side.

```rust
pub trait VirtioDevice: Device {
    fn device_id(&self) -> u16;                       // VIRTIO_ID_*
    fn device_features(&self) -> u64;                 // offered, before transport bits
    fn config_len(&self) -> usize;
    fn read_config(&self, offset: u32, data: &mut [u8]);
    fn write_config(&self, offset: u32, data: &[u8]);
    fn queue_layout(&self) -> &[QueueSpec];           // count, max size, per-queue flags
    fn set_features(&self, acked: u64) -> Result<(), VirtioError>;
    fn activate(&self, ctx: ActivateCtx) -> Result<(), VirtioError>;   // DRIVER_OK
    fn reset(&self);                                  // status write 0 or transport reset
    fn queue_reset(&self, idx: u16) -> Result<(), VirtioError>;        // VIRTIO_F_RING_RESET
    fn queue_enable(&self, idx: u16) -> Result<(), VirtioError>;
    fn status_changed(&self, old: u8, new: u8) {}
    fn backend_kind(&self) -> BackendKind;            // InProcess, VhostKernel, VhostUser, VhostVdpa
    fn vmstate_device(&self) -> Option<&'static VmStateDescription>;
    fn save_extra(&self, w: &mut VmStateWriter) -> Result<(), MigError> { Ok(()) }
    fn load_extra(&self, r: &mut VmStateReader, version: u32) -> Result<(), MigError> { Ok(()) }
}

pub struct ActivateCtx<'a> {
    pub mem: GuestMemoryView<'a>,        // RCU snapshot of the device's DMA address space
    pub queues: Vec<QueueHandle>,        // ready queues with notifier and interrupt objects
    pub features: u64,                   // negotiated
    pub home: ReactorId,                 // iothread or main loop
}
```

`QueueHandle` bundles the ring (split or packed), an `Interrupt` object (MSI-X vector, MMIO interrupt status bit, or CCW adapter indicator), and a `Notifier` (eventfd, kqueue user event, or direct callback for TCG). The device never sees the transport; the transport never parses descriptors. rust-vmm and crosvm draw the same line; ruvm additionally needs QEMU's exact VMState layouts and three transports with legacy quirks.

Feature negotiation follows `virtio_set_features()` in hw/virtio/virtio.c closely, including when the device's `set_features` callback runs relative to status changes and the unchecked path used on migration load. Host feature bits come from device properties (`DEFINE_VIRTIO_COMMON_FEATURES` in include/hw/virtio/virtio.h: `indirect_desc`, `event_idx`, `notify_on_empty`, `any_layout`, `iommu_platform`, `packed`, `queue_reset`, `in_order`) plus per-device properties, filtered by machine compat properties (document 11). In QEMU the 128-bit feature arrays introduced for virtio-net's tunnel offloads (the `host_tunnel`, `guest_tunnel` properties in 11.1) are already beyond 64 bits in places; ruvm uses a `Features([u64; 2])` type from the start so we do not repeat that migration.

### Crate split

- `ruvm-virtio-queue` (new, L0, MIT OR Apache-2.0): split and packed ring parsing, descriptor chain iteration, used-ring and event suppression logic, all generic over a `GuestMemory` trait. Contains no QEMU-derived code; it is written from the OASIS text and is meant to be usable by rust-vmm users.
- `ruvm-vhost` (new, L0, MIT OR Apache-2.0): vhost-user message codec, frontend and backend state machines, vhost-vdpa and vhost kernel ioctl wrappers on top of ruvm-sys. Written from docs/interop/vhost-user.rst and the Linux uAPI headers.
- `ruvm-hw-virtio` (L3, GPL-2.0-or-later): `VirtioDevice`, the three transports, VMState for every device, vhost glue, and most device models. virtio-gpu lives in `ruvm-hw-display` and virtio-snd in `ruvm-hw-audio` because they need the display and audio stacks; both depend on ruvm-hw-virtio.
- `ruvm-vhost-backends` (new, L3/L5, GPL-2.0-or-later where they reuse device models): vhost-user backend daemons built from the same device code, see "vhost-user backends shipped by ruvm".

These four crate names are new decisions and are reported for the workspace map in document 24.

## Transports

### PCI: modern, transitional, legacy

hw/virtio/virtio-pci.c is the reference. A virtio-pci proxy exposes capability structures in config space: `VIRTIO_PCI_CAP_COMMON_CFG`, `NOTIFY_CFG`, `ISR_CFG`, `DEVICE_CFG`, `PCI_CFG`, plus the optional `SHARED_MEMORY_CFG` used by virtio-gpu blob resources and virtiofs DAX, and for transitional devices the legacy I/O BAR. QEMU's default BAR layout (the comment in `virtio_pci_realize()`) is BAR 0 for the legacy I/O region, BAR 1 for MSI-X, BAR 2 for the optional modern I/O notify region, and BARs 4 and 5 for the 64-bit modern memory region. Inside that memory BAR the offsets are fixed: common at 0x0, ISR at 0x1000, device config at 0x2000 (each 0x1000 long), and notify at 0x3000 with size equal to the notify multiplier times `VIRTIO_QUEUE_MAX` (1024). The multiplier is 4 bytes, or `QEMU_VIRTIO_PCI_QUEUE_MEM_MULT` (0x1000) when `page-per-vq=on`, and the BAR is rounded up to a power of two. ruvm reproduces the exact layout, BAR indices and capability order, since firmware and our differential tests compare them.

Device identity: modern devices use device ID 0x1040 plus the virtio device ID, revision 1; transitional devices use the legacy ID range 0x1000 to 0x103f, revision 0, and subsystem device ID equal to the virtio ID. `disable-legacy` defaults to `auto`, which `virtio_pci_realize()` resolves to on when the device sits behind a PCIe port (`pci_bus_is_express()` and not `pci_bus_is_root()`) and off otherwise; `disable-modern` defaults off, with older machine types changing both through compat props. ruvm matches this, including the consequence that virtio devices plugged directly into the q35 root bus are integrated endpoints and remain transitional. QEMU 11.1 also disabled legacy virtio-pci on s390x because it never worked there; we do the same.

Notification paths, fastest first:

1. ioeventfd on the notify region, registered with KVM (`KVM_IOEVENTFD` with datamatch off, length 2 for modern, since the driver writes the 16-bit queue index). The vCPU never exits to userspace. ruvm's KVM accel (document 06) exposes `register_ioeventfd(addr, len, datamatch, Notifier)`. On HVF and WHPX there is no in-kernel ioeventfd, so the vCPU thread takes the MMIO exit and the notify handler signals the queue's reactor directly with a lock-free `Notifier::notify()` (a futex wake or a kqueue `EVFILT_USER` trigger) instead of going through the device lock.
2. `modern-pio-notify` (an I/O port notify capability) for guests that want PIO doorbells; ioeventfd with PIO works the same way.
3. With `VIRTIO_F_NOTIFICATION_DATA`, the driver writes a 32-bit value containing the queue index plus the next available index (split) or next offset and wrap counter (packed). QEMU requires `ioeventfd=off` in that case because a plain eventfd drops the data. ruvm keeps the requirement for command-line compatibility but on KVM registers a separate `KVM_IOEVENTFD` per queue with datamatch disabled on the 4-byte access and reads the value from the ring instead; the data is an optimization hint, not a correctness requirement, so the device can ignore it.

Interrupts: MSI-X with `vectors` defaulting per device (virtio-net-pci uses 2 times the netdev queue count plus 2, virtio-blk-pci `num-queues + 1`), with irqfd routing through KVM (`KVM_IRQFD` bound to an MSI route) so that completions raised from an iothread never touch a vCPU or the control lock. INTx fallback goes through the ISR register with read-to-clear semantics. The `vector` fields in common config and legacy `VIRTIO_MSI_QUEUE_VECTOR` must return `VIRTIO_NO_VECTOR` (0xffff) when the vector is out of range, exactly as `virtio_pci_common_write` does; Windows drivers probe this.

`VIRTIO_PCI_CAP_PCI_CFG` is the window through config space for firmware that cannot map BARs. It is slow and rarely used, but SeaBIOS and some OVMF paths use it, so it is implemented with the same 1, 2 and 4 byte access rules.

PCIe features on the proxy: `ats`, `x-ats-page-aligned`, `aer`, `x-pcie-flr-init`, `x-pcie-pm-no-soft-reset`, and SR-IOV are proxy properties handled by ruvm-hw-pci (document 12). ATS on a virtio device matters with vIOMMUs (document 16).

### MMIO v1 and v2

hw/virtio/virtio-mmio.c. The register map starts with magic 0x74726976 ("virt"), version 1 (legacy) or version 2 (modern). Version 1 uses `QueuePFN` and a guest page size register; version 2 uses split 64-bit descriptor, driver and device addresses and `QueueReady`. The `force-legacy` property defaults to true in hw/virtio/virtio-mmio.c, so arm virt and x86 microvm present version 1 unless the user passes `-global virtio-mmio.force-legacy=false`; the m68k virt, or1k virt and nubus-virtio-mmio boards set it false themselves. ruvm reproduces each board's choice rather than picking a "better" default, since Linux drivers for version 1 and version 2 negotiate differently and the version is visible in the migration stream.

MMIO transports are allocated by the machine (arm virt reserves 32 of them; QEMU 11.0 added the `virtio-mmio-transports` machine property on virt to trim unused ones), and the order of creation determines FDT node order and therefore Linux enumeration order. That ordering bug is a classic source of "disk became vdb" regressions, so ruvm's machine builders create transports and their FDT nodes in the same two loops hw/arm/virt.c uses (transports created in ascending address order, FDT nodes emitted in descending order). ioeventfd works as for PCI (4-byte write of the queue index at offset 0x50). Interrupt status (0x60) and acknowledge (0x64) are updated with atomics, so completions do not take the device lock.

### CCW

hw/s390x/virtio-ccw.c. Virtio over channel I/O uses CCW commands (`CCW_CMD_SET_VQ`, `CCW_CMD_WRITE_FEAT`, `CCW_CMD_READ_FEAT`, `CCW_CMD_WRITE_CONF`, `CCW_CMD_READ_CONF`, `CCW_CMD_WRITE_STATUS`, `CCW_CMD_SET_IND`, `CCW_CMD_SET_CONF_IND`, `CCW_CMD_SET_IND_ADAPTER`, `CCW_CMD_READ_VQ_CONF`, `CCW_CMD_SET_VIRTIO_REV`, `CCW_CMD_READ_STATUS`) with revisions 0 to 2 negotiated by `SET_VIRTIO_REV`. Notifications are diagnose 0x500 subcode 3 hypercalls, which KVM can turn into ioeventfds (`KVM_IOEVENTFD_FLAG_VIRTIO_CCW_NOTIFY`); interrupts are either classic I/O interrupts with indicator bits or adapter interrupts with summary indicators, routed through the s390 floating interrupt controller and irqfd adapter routes. CCW is only reachable from the s390x target (document 09) and the s390-ccw-virtio machine (document 11), so the transport lives in ruvm-hw-virtio behind a `ccw` feature that ruvm-machine-s390x enables. The VMState layout of `VirtioCcwDevice` includes subchannel state from the css code; we port it field for field.

Transports share one `VirtioTransport` trait internally (`notify_queue`, `raise_config_interrupt`, `raise_queue_interrupt`, `queue_address`, `legacy_endianness`), and each implements `MmioOps` or a CCW handler for guest access.

## Virtqueues

### Split ring

The split ring is three areas: descriptor table (16 bytes per entry), available ring (`flags`, `idx`, `ring[N]`, `used_event`), used ring (`flags`, `idx`, `ring[N]` of 8-byte elements, `avail_event`). QEMU's hot functions are `virtqueue_pop()`, `virtqueue_split_read_next_desc()`, `virtqueue_fill()`, `virtqueue_flush()` and `virtio_notify()` in hw/virtio/virtio.c, with `VRingMemoryRegionCaches` holding mapped caches of the three areas so each access does not walk the FlatView. ruvm's equivalent is a `SplitRing` in ruvm-virtio-queue holding three `GuestSlice` handles resolved once at `activate` and revalidated when the FlatView generation changes (document 05 describes the RCU FlatView swap and the generation counter). A memory hotplug or BAR remap that moves guest RAM forces re-resolution on the next pop, which costs one RCU read and a comparison per batch.

Correctness rules we test explicitly, because QEMU has fixed CVEs in each: descriptor chains longer than the queue size are an error (`virtio_error()`, which sets `VIRTIO_CONFIG_S_NEEDS_RESET` and stops the device, not a crash); indirect descriptors inside indirect tables are rejected; an indirect table length that is not a multiple of 16 is rejected; `avail->idx` moving more than `num` entries ahead is an error; reading `avail->ring` must use the value of `avail->idx` loaded before, with an acquire fence (`smp_rmb` in QEMU). Every guest-controlled length is checked against `VIRTQUEUE_MAX_SIZE` (1024) and the iovec capacity before allocation.

### Packed ring

`VIRTIO_F_RING_PACKED` uses a single descriptor ring with AVAIL and USED flag bits and a wrap counter per side, plus driver and device event suppression structures. QEMU's implementation is `virtqueue_packed_pop()` and friends; it defaults off (`packed=off`) and ruvm keeps that default. Packed rings matter mainly for vDPA hardware that implements only packed rings. Edge cases: descriptor chains are identified by buffer ID, so in-flight tracking is by ID not by head index; used elements may be written in any order unless IN_ORDER is negotiated; the wrap counter must flip exactly when `last_avail_idx` wraps, and migration saves it as the high bit of `last_avail_idx` (QEMU encodes `last_avail_wrap_counter` into bit 15 in the vhost `GET_VRING_BASE` value, and saves `last_avail_wrap_counter`, `used_wrap_counter` in the `virtio/packed_virtqueues` subsection).

### Event suppression and event idx

With `VIRTIO_F_EVENT_IDX` the device suppresses notifications using `used_event` (in the avail ring) and publishes `avail_event` (in the used ring). The rule is the `vring_need_event(event, new, old)` wraparound comparison. The subtle part is the ordering: after publishing `avail_event`, the device must re-check `avail->idx` after a full memory barrier or it can sleep while a buffer is pending. QEMU does `virtio_queue_set_notification(vq, 1)` then `smp_mb()` then re-checks `virtio_queue_empty()`. ruvm's poll loop does the same; the barrier is `fence(SeqCst)` and the re-check is mandatory, enforced by the `QueuePoller` API returning a `MustRecheck` token that the caller has to consume.

### Indirect descriptors

Indirect tables are read with a single bulk copy when they fit in one host-contiguous guest RAM region, which is the common case; otherwise the chain walker reads per descriptor. QEMU maps indirect tables via `address_space_cache_init()` per chain; we do the same, keeping a small per-queue cache keyed by table GPA because Linux reuses indirect tables from a slab and the same GPAs recur.

### In-order

`VIRTIO_F_IN_ORDER` lets the device use buffers in the order they were made available and write a single used element for a batch. QEMU added device-side in-order support to hw/virtio/virtio.c for split and packed rings (the `in_order` property), with per-queue `used_elems` tracking so out-of-order completions are held back until the head completes. ruvm implements the same holding buffer as a fixed ring sized to the queue size, allocated once. For virtio-net RX, in-order is natural; for virtio-blk with multiple outstanding requests completing out of order from the host, the hold-back adds latency, which is why QEMU leaves it off by default and so do we.

### Queue reset

`VIRTIO_F_RING_RESET` (`queue_reset=on` default) allows per-queue reset through `queue_reset` in common config, used by Linux to resize virtio-net rings via ethtool and by AF_XDP zero-copy in the guest. The device must stop processing, discard in-flight state for that queue, and allow re-enable with new addresses. For vhost backends this maps to stopping and restarting one vring (`VHOST_USER_SET_VRING_ENABLE` or a full vring stop and start, as `vhost_net_virtqueue_reset()` does).

### Admin virtqueue (new decision)

VIRTIO 1.2 introduced the admin virtqueue (`VIRTIO_F_ADMIN_VQ`, bit 41) and admin command set, used today mainly by an owner PF to manage member VFs (legacy register access commands, and in 1.4 device parts get and set for migration). QEMU 11.1 does not emulate an admin virtqueue on any device. ruvm implements the admin queue machinery in ruvm-virtio-queue and exposes it on virtio-net and virtio-blk only behind `x-admin-vq=on`, never enabled by a machine type. It exists to test guest admin and virtio vfio migration code. Commands implemented are `VIRTIO_ADMIN_CMD_LIST_QUERY`, `LIST_USE`, the legacy common and device config read and write commands, and the 1.4 `DEV_PARTS_METADATA_GET`, `DEV_PARTS_GET`, `DEV_PARTS_SET` backed by the same state serializer as migration.

## Device inventory

The table lists every virtio device type QEMU 11.1 knows and where ruvm runs its data path. "In-process" means a ruvm device model processes the queues itself (in an iothread or the main loop); "vhost-user only" means QEMU only has a vhost-user frontend stub for that device type and the device logic lives in an external daemon. QEMU type names are the `-device` names ruvm accepts.

| Device (virtio ID) | QEMU 11.1 frontends | In-process in ruvm | vhost | Notes |
|---|---|---|---|---|
| net (1) | virtio-net-{pci,device,ccw} | yes | kernel vhost-net, vhost-user, vhost-vdpa | RSS, eBPF RSS, failover, tunnel offloads |
| blk (2) | virtio-blk-*, vhost-user-blk-* | yes | vhost-user, vhost-vdpa-device | iothread-vq-mapping, zoned |
| console (3) | virtio-serial-*, virtconsole, virtserialport | yes | no | chardev backed (document 15) |
| rng (4) | virtio-rng-*, vhost-user-rng-* | yes | vhost-user | rng-random, rng-builtin, rng-egd backends |
| balloon (5) | virtio-balloon-* | yes | no | free page hinting and reporting, stats |
| scsi (8) | virtio-scsi-*, vhost-scsi-*, vhost-user-scsi-* | yes | kernel vhost-scsi, vhost-user | iothread-vq-mapping since 10.0 |
| 9p (9) | virtio-9p-* | yes | no | local and synth fsdrivers; proxy removed in 9.2 |
| gpu (16) | virtio-gpu-*, -gl, -rutabaga, virtio-vga*, vhost-user-gpu, vhost-user-vga | yes | vhost-user | virgl, venus, native context, rutabaga |
| rtc (17, "clock") | virtio-rtc-*, vhost-user-rtc-* | yes | vhost-user | vhost-user-rtc new in 11.1 |
| input (18) | virtio-{keyboard,mouse,tablet,multitouch}-*, virtio-input-host-*, vhost-user-input-* | yes | vhost-user | evdev passthrough on Linux |
| vsock (19) | vhost-vsock-*, vhost-user-vsock-* | no (see below) | kernel vhost-vsock, vhost-user | |
| crypto (20) | virtio-crypto-* | yes | via cryptodev-vhost-user | cryptodev backends |
| iommu (23) | virtio-iommu-* | yes | no | document 16 |
| mem (24) | virtio-mem-* | yes | no | dynamic memslots |
| sound (25) | virtio-sound-*, vhost-user-snd-* | yes | vhost-user | audiodev backed (document 15) |
| fs (26) | vhost-user-fs-* | no | vhost-user only | virtiofsd (Rust, external) |
| pmem (27) | virtio-pmem-* | yes | no | memory-backend |
| scmi (32) | vhost-user-scmi-* | no | vhost-user only | |
| nsm (33) | virtio-nsm-* | yes | no | Nitro Secure Module, nitro machine |
| i2c (34) | vhost-user-i2c-* | no | vhost-user only | |
| gpio (41) | vhost-user-gpio-* | no | vhost-user only | |
| spi (45) | vhost-user-spi-* | no | vhost-user only | present in the 11.1 tree |
| any | vhost-user-test-device (`virtio-id`, `num_vqs`, `vq_size`, `config_size`) | no | vhost-user | generic frontend |

Bluetooth (ID 40), CAN (36), watchdog (35), media (48), video encoder and decoder (30, 31) have IDs in include/standard-headers/linux/virtio_ids.h but no QEMU device. ruvm does not add in-process models for them before 1.0. They are reachable through the generic `vhost-user-test-device` frontend (a deliberately odd name that QEMU uses for its generic vhost-user device) if someone has a backend. We do not invent `-device virtio-bt` names, since a name QEMU does not have would be a compatibility liability the day QEMU picks a different one.

### Per-device notes

virtio-net (hw/net/virtio-net.c) features we match exactly: mergeable RX buffers, control virtqueue (`ctrl_rx`, `ctrl_vlan`, `ctrl_mac_addr`, `ctrl_guest_offloads`, `ctrl_rx_extra`), multiqueue (`mq`, up to `VIRTIO_NET_CTRL_MQ_VQ_PAIRS_MAX` 0x8000 in the header, limited in practice by `VIRTIO_QUEUE_MAX` 1024 queues), RSS and hash reporting (`rss`, `hash`, the nine `hash-*` type properties, indirection table up to `VIRTIO_NET_RSS_MAX_TABLE_LEN` 128), eBPF RSS steering for tap (`ebpf-rss-fds`, the program built from tools/ebpf/rss.bpf.c), USO (`host_uso`, `guest_uso4`, `guest_uso6`), RSC for Windows (`guest_rsc_ext`, `rsc_interval`), guest announce, standby failover (`failover`, docs/system/virtio-net-failover.rst), and in 11.1 the UDP tunnel offloads (`host_tunnel`, `host_tunnel_csum`, `guest_tunnel`, `guest_tunnel_csum`). TX is bottom half by default (`x-txburst` 256) or timer driven at 150 us with `tx=timer`; we keep both because they change guest-visible interrupt coalescing. The ruvm improvement is on the backend side (document 15 covers netdevs): the RX path uses the backend's `recv_batch()` to fill several guest buffers per wakeup instead of QEMU's per-packet `qemu_send_packet` through the net queue.

virtio-blk (hw/block/virtio-blk.c) supports `num-queues`, `queue-size` (default 256), `seg-max-adjust`, discard and write-zeroes with the `max-discard-sectors` and `max-write-zeroes-sectors` limits, zoned block commands (8.1), and `iothread-vq-mapping` (9.0). The legacy `scsi` property was removed in 9.1; we do not accept it. Request merging (`request-merging=on`) coalesces adjacent requests within one batch before submission to ruvm-block; we reproduce QEMU's merge limits (`BDRV_REQUEST_MAX_BYTES`, the iovec limit `IOV_MAX`) so that I/O patterns seen by the host match.

virtio-scsi (hw/scsi/virtio-scsi.c, virtio-scsi-dataplane.c) is a SCSI HBA whose LUNs are scsi-hd, scsi-cd, scsi-block and scsi-generic devices from ruvm-hw-storage (document 14). Hotplug events go through the event queue; the control queue carries TMF and AN requests. Since 10.0 each command queue can be mapped to a different iothread.

virtio-gpu covers several QEMU devices: plain 2D `virtio-gpu-pci` (host-side pixman rendering), `virtio-gpu-gl-pci` with virglrenderer for OpenGL (`virgl`), Vulkan through Venus (`venus=on`, 9.2, requires `blob=on` and `hostmem`), DRM native context (`drm_native_context=on`, new in 11.0, where the guest runs the real Mesa driver for the host GPU, for example freedreno or amdgpu, and virglrenderer passes command buffers to the host kernel driver), `virtio-gpu-rutabaga` (8.2, gfxstream and cross-domain via the crosvm rutabaga_gfx library), and `vhost-user-gpu` which offloads rendering to an external process speaking the vhost-user-gpu protocol on a side socket (`VHOST_USER_GPU_SET_SOCKET`). ruvm links virglrenderer through FFI and rutabaga_gfx as a Rust dependency. Blob resources map host memory into the guest through the PCI shared memory capability; ruvm maps them as `MemoryRegion` subregions of the hostmem BAR (document 05) and, when the backing is a dmabuf, passes the dmabuf through to the display listener (document 15) without copying.

virtio-balloon supports inflate and deflate, `deflate-on-oom`, stats polling (`guest-stats-polling-interval`), free page hinting (`free-page-hint`, which cooperates with migration to skip free pages) and free page reporting (`free-page-reporting`). Balloon inflate calls `madvise(MADV_DONTNEED)` or `fallocate(PUNCH_HOLE)` on the RAM block; this is refused when VFIO devices are present unless `x-balloon-allowed` is set on vfio-pci (document 16), as in QEMU.

virtio-mem (hw/virtio/virtio-mem.c) plugs and unplugs memory in `block-size` units from a memory backend, with `dynamic-memslots` (8.2) to use multiple KVM memslots; it is also wired for confidential guests (11.1 fixed virtio-mem inside CoCo VMs). Dynamic memslots respect the vhost-user backend limit from `VHOST_USER_GET_MAX_MEM_SLOTS`.

virtio-iommu is described in document 16 alongside the other vIOMMUs.

virtio-rtc is new on both sides in 11.1: the in-process `virtio-rtc` device answers configuration, clock capability and read requests with a UTC clock sourced from `QEMU_CLOCK_HOST`, and `vhost-user-rtc` connects to an external daemon offering a fuller implementation (alarms, cross-timestamping). ruvm implements the in-process device with the same request coverage and a monotonic plus UTC clock pair, and offers `vhost-user-rtc` as a frontend.

vsock uses the kernel vhost-vsock (`vhost-vsock-pci`, needs /dev/vhost-vsock and a `guest-cid`) or vhost-user-vsock (for example the rust-vmm `vhost-device-vsock` backend that maps to Unix sockets). QEMU has no in-process vsock, and neither does ruvm, because the kernel implementation is what host applications connect to (`AF_VSOCK` sockets on the host). On macOS, where there is no vhost-vsock, ruvm supports vhost-user-vsock with its own backend (see below).

virtio-fs is vhost-user only. The C virtiofsd was removed from QEMU in 8.0 in favor of the Rust [virtiofsd](https://gitlab.com/virtio-fs/virtiofsd). ruvm does not ship its own virtiofsd; it interoperates with the upstream one, including DAX via the shared memory capability and `VHOST_USER_BACKEND_SHMEM_MAP` where available, and device state migration via `VHOST_USER_PROTOCOL_F_DEVICE_STATE`.

virtio-9p is in-process and a steady CVE source (the 11.1 changelog lists four 9pfs CVEs and a new `max_xattr` limit of 1024 open xattr fids). ruvm ports hw/9pfs with Rust ownership replacing manual fid refcounting, and resolves paths with openat2 `RESOLVE_BENEATH` on Linux.

## vhost

### vhost kernel

hw/virtio/vhost.c and vhost-kernel.c drive /dev/vhost-net, /dev/vhost-scsi and /dev/vhost-vsock. The sequence: `VHOST_SET_OWNER`, `VHOST_GET_FEATURES`, `VHOST_SET_FEATURES`, `VHOST_SET_MEM_TABLE` (a region list derived from the device's DMA address space, merged the way `vhost_region_add_section()` merges adjacent sections, respecting the backend's memslot limit), per vring `VHOST_SET_VRING_NUM`, `SET_VRING_BASE`, `SET_VRING_ADDR`, `SET_VRING_KICK`, `SET_VRING_CALL`, and for net `VHOST_NET_SET_BACKEND` with the tap fd. With a vIOMMU, `VHOST_SET_BACKEND_FEATURES` negotiates the IOTLB message API and ruvm answers `VHOST_IOTLB_MISS` from the vIOMMU translation (document 16), exactly as `vhost_device_iotlb_miss()` does. Dirty logging for migration uses `VHOST_SET_LOG_BASE` with a shared bitmap, which ruvm-mem merges into the global dirty bitmap on each sync (document 17).

The kick eventfd is the queue's ioeventfd and the call eventfd is an irqfd, so no ruvm thread touches the steady-state data path. ruvm handles setup, teardown and the stop sequence: on vm stop, `VHOST_GET_VRING_BASE` returns the last available index which becomes the device's `last_avail_idx` for migration, and the used index is read from guest memory.

### vhost-user frontend

The frontend (QEMU calls it the front-end, formerly master) speaks the protocol in docs/interop/vhost-user.rst over a Unix socket chardev. ruvm implements every message and protocol feature listed in the 11.1 copy of that document. The 11.1 copy defines protocol feature bits 0 to 22, from `MQ` (0) through `DEVICE_STATE` (19) and `GET_VRING_BASE_INFLIGHT` (20) to `GPA_ADDRESSES` (21) and `SHMEM_MAP` (22); the last two are new in 11.1 (the 11.0 document stops at bit 20). ruvm tracks the 11.1 numbering and adds later bits when a QEMU release carries them.

Points where implementations commonly get vhost-user wrong, and what ruvm does:

- Memory table: `SET_MEM_TABLE` sends one fd per region (up to 8 without `CONFIGURE_MEM_SLOTS`); with `CONFIGURE_MEM_SLOTS` the frontend uses `ADD_MEM_REG` and `REM_MEM_REG` incrementally. ruvm requires all guest RAM that a vhost-user device can DMA to be fd-backed (memfd, hugetlbfs, or `memory-backend-file,share=on`), and fails realize with QEMU's error text if not. With `GPA_ADDRESSES` (new in 11.1) vring addresses are guest physical rather than frontend virtual; ruvm prefers it when offered because it removes a class of bugs where the frontend's mapping changes.
- REPLY_ACK: ruvm always negotiates it and waits for acks on messages that change memory layout, so that a memory hot-unplug cannot race a backend still using the old table.
- Backend channel: with `BACKEND_REQ` the backend can send IOTLB misses, config change notifications, host notifier setup (mapping a backend-owned doorbell page directly into the guest's notify region, which lets hardware-assisted backends avoid eventfds) and, with `SHMEM_MAP`, shared memory mappings used by virtiofs DAX and virtio-gpu blobs. ruvm services this channel on the device's home reactor rather than the main thread, so a backend blocked on an IOTLB reply can never wait behind a QMP command holding the control lock.
- Reconnect: with `reconnect-ms` on the socket chardev and `INFLIGHT_SHMFD`, a restarted backend recovers in-flight descriptors from the shared inflight buffer (`GET_INFLIGHT_FD`, `SET_INFLIGHT_FD`). ruvm allocates the inflight region per device and keeps it across reconnects, matching vhost-user-blk behavior in hw/block/vhost-user-blk.c.
- Device state: with `DEVICE_STATE`, migration uses `SET_DEVICE_STATE_FD` to open a pipe over which the backend streams opaque internal state, then `CHECK_DEVICE_STATE`. This is how virtiofsd migrates its open file handles. ruvm stores the blob exactly where QEMU's `vhost-user-fs-backend` VMState (field "back-end", in hw/virtio/vhost-user-fs.c) puts it, so a ruvm source can migrate to a QEMU destination with the same backend.

### vhost-user backends shipped by ruvm

ruvm ships backends as crates (library plus a thin binary each) in `ruvm-vhost-backends`, built on `ruvm-vhost`'s backend state machine and ruvm-aio reactors. QEMU ships contrib backends and a qemu-storage-daemon export, so a drop-in replacement needs equivalents. The ruvm backends reuse the in-process device models unchanged, which is the main argument for the transport independent `VirtioDevice` trait: the same `VirtioBlk` code runs in-process, behind vhost-user, or behind VDUSE.

| Backend | QEMU counterpart | Notes |
|---|---|---|
| vhost-user-blk | qemu-storage-daemon `--export vhost-user-blk`, contrib/vhost-user-blk | full ruvm-block graph (document 14); also exported via VDUSE |
| vhost-user-scsi | contrib/vhost-user-scsi | ruvm-hw-storage SCSI disks |
| vhost-user-gpu | contrib/vhost-user-gpu | virglrenderer, runs as a separate sandboxed process |
| vhost-user-input | contrib/vhost-user-input | evdev passthrough |
| vhost-user-net bridge | contrib/vhost-user-bridge | test and example only |
| vhost-user-vsock | none in QEMU | Unix socket mapping, used on macOS |
| vhost-user-rng, -snd, -rtc | none in QEMU | small, share device code with in-process models |

For gpio, i2c, spi, scmi and CAN we point users at rust-vmm's vhost-device project rather than duplicating it; our CI runs those backends against the ruvm frontend to guarantee interop. The backends that depend on ruvm-block or device models are GPL-2.0-or-later; the `ruvm-vhost` protocol library underneath is permissive, per the canon licensing rule, and `cargo xtask provenance` enforces the direction.

### vhost-vdpa

hw/virtio/vhost-vdpa.c and net/vhost-vdpa.c talk to /dev/vhost-vdpa-N, which fronts either a hardware vDPA device (for example a SmartNIC VF) or a software one (VDUSE, or the vdpa_sim simulator). The ioctl set extends vhost kernel with device ID, status, config, vring enable, `GET_IOVA_RANGE`, `SUSPEND`, `RESUME` and ASIDs for isolating the control virtqueue; DMA mapping uses batched vhost IOTLB messages.

Two frontends exist in QEMU and ruvm: `-netdev vhost-vdpa` feeding a virtio-net device, and `vhost-vdpa-device-pci` (generic, any device type; used with VDUSE block exports, fixed in 9.0).

Shadow virtqueue (hw/virtio/vhost-shadow-virtqueue.c, `x-svq=on`) is the migration trick for vDPA: the frontend interposes its own vring between guest and device, so it can see used buffers and mark dirty pages, and so it can intercept the control virtqueue to replay MAC, MQ, offload and RSS state on the destination. Since QEMU 8.0 vhost-vdpa net devices without a CVQ migrate without SVQ, and 8.1 added SVQ offload support. ruvm implements SVQ with an IOVA tree allocator equivalent to hw/virtio/vhost-iova-tree.c, allocating shadow rings from the device's `GET_IOVA_RANGE` window. When the device supports `VHOST_BACKEND_F_SUSPEND` and virtio 1.4 device parts, a future path would let hardware report its own state; we leave that for after 1.0.

### VDUSE

VDUSE (vDPA Device in Userspace, Linux 5.15, [kernel doc](https://docs.kernel.org/userspace-api/vduse.html)) lets a userspace process implement a vDPA device which the host kernel then exposes either to host applications through virtio-vdpa or to VMs through vhost-vdpa. QEMU uses it for block exports (`--export vduse-blk` in qemu-storage-daemon). The kernel initially limited VDUSE to block devices for security reasons. ruvm's storage daemon supports the same export, reusing `VirtioBlk` via the `ruvm-vhost` VDUSE adaptor (`/dev/vduse/control`, `VDUSE_CREATE_DEV`, per-device char device with message-based control plane and shared memory data plane).

## Queue processing fast path

This section is where ruvm departs from QEMU's implementation while keeping its behavior. The design goal is zero heap allocation, zero syscalls per request in the steady state where the backend allows it, and bounded latency when idle.

### Execution contexts

Each queue has a home reactor: the main loop by default, an iothread when `iothread=` or `iothread-vq-mapping=` is set, or the vCPU thread for synchronous TCG notify where configured (below). Notifications from the guest are delivered to the home reactor through the queue's `Notifier` (an eventfd registered in the io_uring as a multishot poll, or a kqueue event). The reactor calls the device's `process_queue(idx, budget)`. Completions from the backend (io_uring CQEs for block, socket readiness or AF_XDP ring updates for net) arrive on the same reactor, so a request's whole lifecycle runs on one thread without locks.

### Batching

`process_queue` pops up to `budget` chains (default 64, a starting value to be tuned in document 21) into a per-queue `ArrayVec` of prepared requests. Each chain is converted to an iovec slice pointing directly at host virtual addresses of guest RAM (zero copy), with the iovec storage borrowed from a per-queue arena sized at activate time to `queue_size * max_seg`. Requests are submitted as a batch: for virtio-blk that is one `io_uring_enter` with N SQEs, or none at all if SQPOLL is enabled; for virtio-net TX it is one `sendmmsg`, one `writev` burst to tap, or one AF_XDP TX ring update.

Completions are coalesced: the device writes used elements as completions arrive, but publishes `used->idx` and evaluates event suppression once per reactor iteration, not per request. That means at most one interrupt per queue per iteration. QEMU does this for virtio-blk and virtio-scsi with `defer_call_begin()` and `defer_call_end()` (util/defer-call.c) around queue processing, which also defers irqfd notifications; ruvm makes it the default for every device via a `CompletionBatch` guard dropped at the end of the iteration.

### Zero copy rules

Guest buffers are passed to the host kernel directly whenever the memory is ordinary RAM and not subject to a vIOMMU mapping that could change under the I/O. With a vIOMMU, ruvm pins the translation for the duration of the request by holding an IOTLB entry reference; an unmap from the guest waits until the reference drops (QEMU instead relies on the BQL plus `dma_memory_map` bounce buffers for non-RAM). Buffers that land on MMIO or ROM regions fall back to a bounce buffer, exactly like `dma_memory_map()` does via `address_space_map()` with its single bounce buffer (QEMU 9.x made the bounce buffer size configurable per address space with `x-max-bounce-buffer-size` on PCI devices; we keep that property).

For confidential guests with shared and private memory (document 19), virtio buffers live in shared memory by definition (the guest uses swiotlb), so zero copy works unchanged.

### io_uring integration

On Linux, the reactor is io_uring based (canon: ruvm-aio). For virtio-blk with a raw or qcow2 file on O_DIRECT, the block layer submits `IORING_OP_READV` or `WRITEV` with the guest iovecs; registered buffers are not used because guest RAM is large and fixed buffer registration of whole RAM slots (`IORING_REGISTER_BUFFERS` of each RAM block, up to the kernel's per-buffer size limit) is an optional mode (`x-uring-fixed-bufs=on`) we will measure before defaulting. For virtio-net over tap, TX submits one `IORING_OP_WRITEV` per packet, all in one submission, pointing at guest memory; RX uses provided buffer rings so the kernel picks buffers, then the device copies into guest buffers. Tap RX cannot be zero copy into guest memory without vhost-net, because the kernel does not know the guest buffer before the packet arrives. For RX-heavy workloads the right answer remains kernel vhost-net or AF_XDP, and ruvm selects vhost-net by default when `/dev/vhost-net` is accessible and the user did not say `vhost=off`, which is also QEMU's behavior with libvirt.

Eventfd kicks are consumed through a multishot `IORING_OP_POLL_ADD` on the ioeventfd; MSI delivery is a write to the irqfd, issued as `IORING_OP_WRITE` in the same submission batch as I/O so the reactor does one syscall per iteration in the busy case.

### Polling with adaptive backoff

QEMU iothreads poll before blocking, with `poll-max-ns` defaulting to 32768 ns on POSIX hosts (`IOTHREAD_POLL_MAX_NS_DEFAULT` in include/system/iothread.h) and `poll-grow` and `poll-shrink` tuning an adaptive window in util/aio-posix.c. ruvm keeps the same QOM properties with the same defaults and semantics, so libvirt's `<iothread>` tuning maps unchanged. The poller checks avail indices in guest memory and the io_uring CQ head, with guest notifications disabled while polling, which removes vmexits under load. The adaptive rule is QEMU's `adjust_polling_time()`: after each blocking wait, if the time spent blocked exceeded the polling window, the window grows (multiplied by `poll-grow`, or set straight to the blocked time if that is larger) up to `poll-max-ns`; if the blocked time was well under the window divided by `poll-shrink`, the window shrinks by that factor. Before blocking, notifications are re-enabled, followed by the mandatory re-check described under event idx.

An additional mode, `x-poll-mode=busy`, pins a reactor to a core and never blocks, for the SPDK-style deployments where a core per iothread is budgeted. It is off by default.

### TCG path

Under the JIT (documents 07, 08) there is no ioeventfd. A notify write from a vCPU thread lands in the transport's `MmioOps::write`, which does an atomic store of the pending queue bit and signals the home reactor. With `x-notify-inline=on`, small devices (rng, console) process the queue synchronously on the vCPU thread, which saves a thread hop and is what QEMU effectively does under the BQL for non-iothread devices. Default is off to keep ordering identical to the KVM path.

### Multiqueue and iothread-vq-mapping

`iothread-vq-mapping` (virtio-blk since 9.0, virtio-scsi since 10.0, implemented in hw/virtio/iothread-vq-mapping.c) is a list of `{iothread, vqs}` entries; if `vqs` is omitted, queues are assigned round robin across the listed iothreads. It is mutually exclusive with `iothread=`. The ruvm implementation is the same assignment algorithm with the same validation errors (every queue assigned exactly once, iothreads exist, no mixing with `iothread=`), because libvirt generates these configurations and compares error strings in its tests.

For virtio-net, multiqueue maps queue pairs to tap queues (`queues=N` on the netdev, one fd per queue with `IFF_MULTI_QUEUE`) and to vhost-net worker threads in the kernel. When not using vhost, ruvm can spread queue pairs across iothreads with an `x-iothread-vq-mapping` property on virtio-net; QEMU 11.1 has no equivalent, so this is off by default and is an extension, reported for document 25.

Multiqueue and MSI-X vectors: a device with N queues and fewer than N+1 vectors shares vectors among queues; the guest driver decides. ruvm must raise the same vector the guest programmed, so the interrupt object is looked up per queue at `queue_enable` time and cached.

### Locking

Per canon, there is no big lock on the data path. Each in-process virtio device has: an atomic status and feature snapshot readable without locks by the fast path, a per-queue state owned by the home reactor (no lock), and a small `Mutex` for config space and control operations (reset, feature set, config writes, queue enable). The transport's register handlers take that mutex; notify handlers do not. Reset is the tricky case: a guest writing status 0 must stop all queue processing and wait for in-flight requests (QEMU's `virtio_reset` plus `blk_drain`). ruvm posts a `Quiesce` message to each home reactor and waits for acks before completing the MMIO write; the vCPU is blocked in the MMIO handler for the drain duration, which matches QEMU's guest-visible behavior (reset is synchronous from the guest's point of view).

## Migration of virtio state

Virtio migration is where compatibility breaks most easily, because QEMU's format evolved through subsections added over a decade. ruvm-vmstate (document 17) ports each description, and this crate supplies:

- The top-level `virtio` VMState (`vmstate_virtio` in hw/virtio/virtio.c) with its subsections: `virtio/device_endian`, `virtio/64bit_features`, `virtio/virtqueues` (vring addresses for used and avail when not derived from the legacy PFN), `virtio/ringsize`, `virtio/broken`, `virtio/extra_state` (transport-specific data such as the PCI modern queue state saved by `virtio_pci_save_extra_state`), `virtio/started`, `virtio/disabled`, `virtio/packed_virtqueues`, and `virtio/128bit_features`. Subsections are emitted only when their `needed` predicate is true, exactly as in QEMU, because a destination QEMU rejects unknown subsections.
- The legacy "put" function layout: `virtio_save()` writes the transport config (`save_config`), feature words, config space blob, `queue_sel`, `guest_features` low 32 bits, `config_len`, the config bytes, the number of queues in use, and per queue `vring.num`, `vring.desc` (or PFN for legacy), `last_avail_idx`, and the transport's `save_queue`, then the device-specific `vmsd` or `save` output. ruvm writes the same bytes; there is no "ruvm native" virtio migration format.
- Per device state: virtio-net saves MAC, `status`, promiscuous and multicast state, the MAC table, VLAN bitmap, `curr_queue_pairs`, `curr_guest_offloads`, RSS configuration (`virtio-net-device/rss`), announce timer; virtio-blk saves the list of in-flight requests (`virtio_blk_save_device` writes each pending `VirtIOBlockReq` element so the destination resubmits them); virtio-scsi does the same through the SCSI bus; virtio-gpu saves resources and blob mappings.
- In-flight element encoding: `qemu_put_virtqueue_element()` writes a fixed-size legacy struct (`VirtQueueElementOld`) including in and out addresses and sg lengths. ruvm serializes in-flight requests into that exact struct, including its padding, and on load maps the addresses again.

Before saving, every queue is quiesced: in-process queues stop at a batch boundary, vhost kernel and vhost-user rings are stopped with `GET_VRING_BASE`, and vhost-vdpa uses `SUSPEND` if available. The `last_avail_idx` saved for a vhost device is the backend's, and `used_idx` is re-read from guest memory on load (`virtio_queue_restore_last_avail_idx`). On load, ruvm validates `last_avail_idx - used_idx <= vring.num` (QEMU's "VQ %d size 0x%x < last_avail_idx 0x%x - used_idx 0x%x" error) and fails migration rather than letting a corrupted stream cause a guest-memory scan.

Dirty tracking: in-process devices mark written guest pages dirty via ruvm-mem's dirty bitmap as part of writing used elements and data (the `dma_memory_write` path does this automatically); vhost kernel and vhost-user devices use the vhost log (`VHOST_F_LOG_ALL`, `SET_LOG_BASE`), which ruvm merges on each `log_sync`. vDPA without device dirty logging requires SVQ.

For CPR modes (document 17), QEMU 11.1 preserves guest RAM and VFIO and iommufd descriptors across the exec, and its CPR documentation lists vhost and chardev descriptor transfer as future work. ruvm matches that scope for interop and treats vhost fd transfer as a post-1.0 extension, since a vhost-net fd handed to the new process could keep the data path running through the switchover.

## Testing

QEMU's libqos virtio qtests run unmodified against ruvm through the qtest accelerator (document 22); descriptor chain parsing is fuzzed; each device type is migrated QEMU 11.1 to ruvm and back in the middle of fio or iperf3 runs; and the vhost-user frontend is tested against virtiofsd, rust-vmm vhost-device backends, DPDK's vhost library, passt and SPDK.

## Open items

- Admin virtqueue exposure (`x-admin-vq`) and whether to implement virtio 1.4 `VIRTIO_F_SUSPEND` for in-process devices ahead of QEMU.
- `x-iothread-vq-mapping` for virtio-net without vhost.
- io_uring fixed buffer registration of guest RAM (`x-uring-fixed-bufs`) default, pending document 21 measurements.
- Whether ruvm should provide in-process virtio-fs for macOS and Windows hosts where the upstream virtiofsd does not run. Leaning no.
