# 17. Migration, snapshots and record/replay (ruvm-vmstate, ruvm-migration)

This document covers two crates. ruvm-vmstate (L1) defines how device and CPU state is described and encoded, and must produce exactly the bytes QEMU produces. ruvm-migration (L3) owns everything that moves a running VM: the migration stream, channels, precopy, postcopy, multifd, dirty tracking policy, snapshots in qcow2 and in files, CheckPoint and Restart (CPR), the fast local snapshot path for serverless use, record/replay, and COLO. The reference is QEMU 11.1.0 (`migration/*.c`, `replay/*.c`, `include/migration/vmstate.h`, `qapi/migration.json`). The canon sets the bar: migration must interoperate with QEMU in both directions for every supported machine type, and downtime must be the same or lower than QEMU for the same workload and bandwidth. Migration interop is milestone M5 (document 23).

## Why wire compatibility dominates the design

Interop with QEMU is what lets an operator put ruvm into an existing fleet: live migrate a VM from QEMU to ruvm to try it, and back when something goes wrong. That rules out any design where ruvm has its own state format and a converter. The stream QEMU writes is not self-describing enough for that anyway: a VMState section is a sequence of fields with no tags, so the only way to read it is to know the exact field list, order, sizes, versions and subsection predicates for that device on that machine version. So ruvm-vmstate reproduces QEMU's field lists for every device, and every device crate in ruvm carries a VMState description that is checked against QEMU's by tooling (below). The price is real: device structs in ruvm are free to be laid out however we want in memory, but their migration description is not free at all.

## ruvm-vmstate: describing state

### The model

QEMU describes state with `VMStateDescription` (name, `version_id`, `minimum_version_id`, `priority`, `early_setup`, `unmigratable`, pre/post load/save hooks in plain and `_errp` flavors, `needed`, `dev_unplug_pending`, a field array, and a subsection array) and `VMStateField` (name, offset, size, `VMStateInfo` for leaf types, flags such as `VMS_SINGLE`, `VMS_POINTER`, `VMS_ARRAY`, `VMS_STRUCT`, `VMS_VARRAY_*`, `VMS_BUFFER`, `VMS_VBUFFER`, `VMS_MULTIPLY`, `VMS_ALLOC`, `VMS_MUST_EXIST`, `VMS_VSTRUCT`, `VMS_ARRAY_OF_POINTER`, `VMS_ARRAY_OF_POINTER_AUTO_ALLOC`, the per-field `version_id`, `struct_version_id`, and `field_exists`). The encoder in `migration/vmstate.c` interprets this table.

ruvm keeps the table-driven model rather than generating a bespoke serializer per struct. Three reasons: the same table produces the vmdesc JSON appended to the stream and the `-dump-vmstate` output that QEMU's `scripts/vmstate-static-checker.py` consumes; device state is small next to RAM (kilobytes against gigabytes), so interpretation cost does not matter; and a table can be diffed against QEMU's table automatically, where generated code cannot.

```rust
pub struct VmStateDescription {
    pub name: &'static str,                 // "virtio-blk", "cpu", "timer"...
    pub version_id: u32,
    pub minimum_version_id: u32,
    pub priority: MigPriority,              // MIG_PRI_DEFAULT, IOMMU, PCI_BUS, VIRTIO_MEM, APIC, GICV3_ITS, GICV3...
    pub early_setup: bool,
    pub unmigratable: bool,
    pub fields: &'static [VmStateField],
    pub subsections: &'static [&'static VmStateDescription],
    pub hooks: &'static VmStateHooks,       // pre_save, post_save, pre_load, post_load, needed, dev_unplug_pending
}

pub struct VmStateField {
    pub name: &'static str,
    pub offset: usize,                      // core::mem::offset_of!
    pub size: usize,
    pub start: usize,
    pub num: u32,
    pub num_offset: usize,                  // for VARRAY: offset of the length field
    pub size_offset: usize,                 // for VBUFFER
    pub info: Option<&'static dyn VmStateInfo>,
    pub vmsd: Option<&'static VmStateDescription>,
    pub flags: VmsFlags,
    pub version_id: u32,                    // field present since this version
    pub struct_version_id: u32,
    pub field_exists: Option<fn(*const (), u32) -> bool>,
}

pub trait VmStateInfo: Sync {
    fn name(&self) -> &'static str;         // "uint32", "int64", "bool", "buffer", "timer", "qtailq"...
    fn get(&self, r: &mut StreamReader, p: *mut u8, size: usize, f: &VmStateField) -> Result<()>;
    fn put(&self, w: &mut StreamWriter, p: *const u8, size: usize, f: &VmStateField, d: &mut VmDesc) -> Result<()>;
}
```

Raw pointers appear only inside ruvm-vmstate's codec, which is the one place that needs type-erased field access. Device code never sees them.

### The derive macro

Devices declare state with `#[derive(VmState)]`. The macro emits a `static VMSTATE_<NAME>: VmStateDescription` and implements `HasVmState`. It is a direct transliteration of the `VMSTATE_*` macros, so a porter reading QEMU's `vmstate_pl011` writes the same list in the same order:

```rust
#[derive(VmState)]
#[vmstate(name = "pl011", version = 2, minimum_version = 2,
          post_load = Self::post_load, subsections(vmstate_pl011_clock))]
pub struct Pl011Regs {
    #[vmstate(unused = 4)]                        // VMSTATE_UNUSED(sizeof(uint32_t))
    _pad: (),
    flags: u32,
    lcr: u32,
    rsr: u32,
    cr: u32,
    dmacr: u32,
    int_enabled: u32,
    int_level: u32,
    #[vmstate(array = 16)]                        // VMSTATE_UINT32_ARRAY(read_fifo, PL011State, 16)
    read_fifo: [u32; 16],
    ilpr: u32,
    ibrd: u32,
    fbrd: u32,
    ifl: u32,
    read_pos: i32,
    read_count: i32,
    read_trigger: i32,
}
```

Attributes cover the full macro family: `since = N` (field version), `array`, `varray(len = field, type = u32)`, `vbuffer(len = field)`, `buffer`, `struct(vmsd = ...)`, `pointer`, `alloc`, `timer`, `ptimer`, `qtailq`, `gtree`, `qlist`, `with_tmp(T)` for `VMSTATE_WITH_TMP`, `validate(fn)` for `VMSTATE_VALIDATE`, `exists = fn`, `unused = N`, and `info = custom` for a hand-written `VmStateInfo`. Rust types map to QEMU infos by width and signedness (u8 to `uint8`, i64 to `int64`, bool to `bool`, and so on); a type with no obvious mapping is a compile error rather than a guess. Floats use `float64`, `CPU_DoubleU` style unions use their QEMU encoding. The macro also asserts at compile time that `size_of` of each field matches the info's size, which catches the classic bug where a field grows from u32 to u64 in the Rust struct and silently changes the stream.

Subsections are separate descriptions with a `needed` predicate. The predicate is the compatibility valve QEMU relies on: new state is sent only when it differs from what an old destination assumes, so a new source can still migrate to an old destination in the common case. ruvm device authors port the predicates exactly, including machine compat property checks such as "only send if `x-foo` compat flag is on".

Hooks are methods on the device: `pre_save(&mut self) -> Result<()>`, `post_save(&mut self)`, `pre_load(&mut self) -> Result<()>`, `post_load(&mut self, version_id: u32) -> Result<()>`. They take `&mut self` because the codec runs while the VM is stopped and the device's lock domain (document 03) is held. `post_load` is where devices recompute derived state (IRQ levels, timers, memory region enables); it must be written so that loading QEMU-produced state yields the same derived state QEMU computes, which is tested by the interop suite rather than argued.

### Registration, naming and ordering

A section is identified by `idstr` and `instance_id`. For qdev devices QEMU builds the idstr from the device's path on its bus (`qdev_get_dev_path`, for example `0000:00:03.0/virtio-blk`) plus the vmsd name, and the instance id disambiguates duplicates. For non-qdev state (`ram`, `timer`, `slirp`, `cpu_common`, `dirty-bitmap`, `block`, and so on) the name is registered directly. ruvm-hw-core computes the same dev paths from its own bus model, and the vmstate registry (`SaveStateEntry` list in `migration/savevm.c`) is ordered the way QEMU orders it: sorted by priority descending at insertion (`savevm_state_handler_insert`), then by registration order. Registration order in QEMU depends on device creation order, which depends on the machine init code and command line order. ruvm's machine crates (document 11) create devices in QEMU's order for this reason, and the interop suite checks the resulting section order against QEMU's vmdesc JSON.

The destination matches sections by idstr and instance id, not by position (`find_se`), so small order differences in full sections do not break loading. They do change which device's `post_load` runs first, and that matters for pairs like an interrupt controller and its CPUs, which is exactly why priorities exist. Priority order is therefore a hard compatibility item and ordering within a priority is a soft one.

## The migration stream, byte by byte

All integers are big-endian. The stream is produced by `QEMUFile` in QEMU (32 KiB buffer, `IO_BUF_SIZE`) and by `StreamWriter` in ruvm with the same flush points, since some channels (fd, exec) are pipes where the reader blocks on exact boundaries.

### Header and configuration

```
be32  0x5145564d            QEMU_VM_FILE_MAGIC ("QEVM")
be32  0x00000003            QEMU_VM_FILE_VERSION
u8    0x07                  QEMU_VM_CONFIGURATION   (if migration property send-configuration=on)
      vmstate "configuration" v1:
        be32 len; u8[len] machine type name (e.g. "pc-q35-11.1")
        subsection "configuration/target-page-bits"  (if target page bits > legacy)
        subsection "configuration/capabilities"      (count + list of validated capability names)
        subsection "configuration/uuid"              (if validate-uuid)
```

The configuration section is loaded on the destination by `configuration_post_load`, which fails the migration on a machine type mismatch (with the exact error text) and validates capabilities that must match on both sides. `send-configuration`, `send-section-footer` and `store-global-state` are properties of the migration object that old machine types turn off through compat props, so ruvm's machine compat tables must include them.

### Sections

```
u8    type                  0x01 START, 0x02 PART, 0x03 END, 0x04 FULL
be32  section_id
      if START or FULL:
u8    idstr_len; u8[idstr_len] idstr
be32  instance_id
be32  version_id
      payload
u8    0x7e                  QEMU_VM_SECTION_FOOTER (if send-section-footer)
be32  section_id            must match; catches misparsed payloads
```

START, PART and END carry iterative state (RAM, dirty bitmaps, VFIO precopy data, and the `block` dirty bitmap section). FULL carries a whole VMState description. Within a VMState payload, fields are written in order with no framing, then each needed subsection:

```
u8    0x05                  QEMU_VM_SUBSECTION
u8    name_len; u8[name_len] name   (must start with parent name + "/")
be32  version_id
      payload (recursively, with its own subsections)
```

The loader peeks for `0x05` after the fields; a subsection it does not know is an error, a subsection it knows but the source did not send simply keeps defaults. That asymmetry is the whole forward compatibility story, and ruvm's codec implements the peek logic exactly as `vmstate_subsection_load` does, including the rule that a subsection name must start with the parent's name and a slash.

### Commands

```
u8    0x08                  QEMU_VM_COMMAND
be16  cmd                   MIG_CMD_*
be16  len
u8[len] data
```

Commands in order of the enum: 1 `OPEN_RETURN_PATH`, 2 `PING` (be32), 3 `POSTCOPY_ADVISE` (two be64: host page size summary and target page size, or empty), 4 `POSTCOPY_LISTEN`, 5 `POSTCOPY_RUN`, 6 `POSTCOPY_RAM_DISCARD` (u8 version 0, u8 name length, name, NUL, then pairs of be64 start and length), 7 `PACKAGED` (be32 length of an embedded stream that follows), 8 deprecated (was `ENABLE_COLO` before QEMU 10.2), 9 `POSTCOPY_RESUME`, 10 `RECV_BITMAP` (name), 11 `SWITCHOVER_START`. A ruvm destination must accept command 8 from older QEMU sources on machine types that allow COLO, and a ruvm source never sends it.

### End of stream

```
u8    0x00                  QEMU_VM_EOF
u8    0x06                  QEMU_VM_VMDESCRIPTION (unless suppressed)
be32  json_len
u8[json_len] JSON           {"page_size":..., "devices":[{"name":..,"instance_id":..,"vmsd_name":..,"version":..,"fields":[...],"subsections":[...]}]}
```

The trailing JSON exists for `scripts/analyze-migration.py` and is ignored by the loader. It is suppressed for machine types with `suppress-vmdesc` set. ruvm writes it byte identical, which makes it a free oracle: a diff of vmdesc between a QEMU run and a ruvm run with the same config pinpoints any field list mismatch.

### RAM section payload

RAM is the `ram` section, iterative, version 4. Every record starts with a be64 whose low bits are flags and whose high bits are the page offset within the RAM block (flags live below 0x400, so migration works for targets with pages of 1 KiB or larger):

| Flag | Value | Meaning |
|---|---|---|
| ZERO | 0x002 | page is zero; one following byte, always 0 (historically the fill byte) |
| MEM_SIZE | 0x004 | setup record: total RAM size in the high bits, then the block list |
| PAGE | 0x008 | raw page of target page size follows |
| EOS | 0x010 | end of this section's RAM data |
| CONTINUE | 0x020 | same block as the previous record; idstr omitted |
| XBZRLE | 0x040 | u8 0x01 encoding flag, be16 length, XBZRLE delta follows |
| HOOK | 0x080 | RDMA control, only on rdma channels |
| MULTIFD_FLUSH | 0x200 | sync point for multifd channels |

Removed values stay reserved: 0x001 (FULL, obsolete since 2009) and 0x100 (COMPRESS_PAGE, removed in QEMU 9.1 along with the old compress threads). Without CONTINUE, the record has `u8 len, idstr` after the be64. The setup record lists each migratable RAM block as `u8 len, idstr, be64 used_length`, plus `be64 page_size` if postcopy is on and the block's page size differs from the host page size, plus `be64 mr->addr` with `x-ignore-shared`, plus the mapped-ram header when mapped-ram is on. XBZRLE encodes the XOR of the old and new page as alternating zero-run and nonzero-run lengths in ULEB128 followed by nonzero bytes (`migration/xbzrle.c`), with a page cache sized by `xbzrle-cache-size` (64 MiB default). ruvm implements XBZRLE with SIMD scanning; format and cache replacement policy are identical because the destination has no cache and a mismatch would corrupt pages, so the only freedom is speed.

Zero page detection (`zero-page-detection=none|legacy|multifd`) decides who checks for zeros: the main thread (`legacy`), the multifd threads (`multifd`, default since 9.0), or nobody. ruvm's buffer-is-zero uses the same early exit strategy as `util/bufferiszero.c` with AVX2/AVX-512/NEON paths.

## Channels

Addresses come from `-incoming` and the `migrate` command's `uri` or `channels` list (`MigrationChannel` with `channel-type` main or cpr and a `MigrationAddress` of type socket, exec, rdma, file). All of these are supported: `tcp:host:port` (with `ipv4`, `ipv6`, and multiple addresses), `unix:path`, `vsock:cid:port`, `fd:name` (an fd passed with `getfd` or `add-fd`), `exec:cmd` (a shell pipeline, still used by some tools), `file:path,offset=N` (seekable, needed for mapped-ram), and `rdma:host:port` (via rdma-core FFI, `rdma-pin-all`, IPv6 since QEMU 10.1, `x-rdma-chunk-size` since 11.1). TLS uses `tls-creds`, `tls-hostname` and `tls-authz` over rustls in ruvm-crypto (document 14 introduces that crate).

The destination identifies extra connections the way `migration_channel_identify` does: if the channel supports peeking, a first be32 of `QEMU_VM_FILE_MAGIC` means main, `MULTIFD_MAGIC` (0x11223344) means multifd, and once main and multifd are established, the next connection is the postcopy preempt channel (which sends no magic). Non-peekable channels (TLS, file) fall back to order. ruvm must behave identically because a QEMU source opens channels concurrently and relies on this.

The return path (source to destination reverse channel, `return-path` capability, implied by postcopy) carries `MIG_RP_MSG_*` messages: SHUT, PONG, REQ_PAGES (be64 start, be32 len), REQ_PAGES_ID (plus RAM block name), RECV_BITMAP, RESUME_ACK, SWITCHOVER_ACK.

## Precopy

The algorithm is QEMU's, with implementation choices aimed at downtime:

1. Setup: start dirty logging on all RAM (document 05 for TCG and document 06 for KVM), send the MEM_SIZE record and each iterative handler's setup data, and run `save_prepare` for devices like VFIO.
2. Iterate: walk the dirty bitmap, send pages, periodically resync the bitmap from the accelerator (`migration_bitmap_sync`), estimate bandwidth and remaining dirty bytes.
3. Switchover when estimated downtime is under `downtime-limit` (default 300 ms), counting device state as well as RAM (QEMU 11.1 extended `query-migrate` to report expected downtime including VFIO device state; ruvm computes the same numbers and uses them for the decision).
4. Stop the VM, sync the bitmap one last time, send remaining pages and all FULL sections, EOF.

Bandwidth: `max-bandwidth` (default 128 MiB/s, `MAX_THROTTLE`), `avail-switchover-bandwidth` to override the estimate at switchover, `downtime-limit`. `pause-before-switchover` inserts a `pre-switchover` state where the management layer can finish block handover (with storage daemon backends, `blockdev-set-active` from document 14).

Where ruvm expects to win on downtime, all within the wire format:

- Parallel final sync. QEMU's final bitmap sync and last page send are mostly serial on the migration thread unless multifd is on. ruvm splits the final dirty set across multifd channels whenever multifd is negotiated, and prepares device state in parallel per lock domain while RAM drains.
- Clear-log batching. With KVM's manual dirty log protect, ruvm clears in large aligned chunks while pages are being sent (QEMU 10.1 also cut unnecessary `LOG_CLEAR` work; we start from that behavior).
- No BQL at stop. Device state save runs per lock domain concurrently, and only the serialization into the single main channel is ordered.

### Dirty tracking: bitmap and ring

Two KVM mechanisms: the per-slot dirty bitmap (`KVM_GET_DIRTY_LOG`, optionally `KVM_CLEAR_DIRTY_LOG` with manual protect) and the per-vCPU dirty ring (`dirty-ring-size` accelerator property, `KVM_CAP_DIRTY_LOG_RING`), where the kernel pushes GFNs into a ring mapped into userspace and a reaper collects them. The ring gives per-vCPU dirty rates, which is what the dirty limit feature needs, and costs a vmexit when a ring fills. The bitmap is cheaper for workloads that dirty huge regions. ruvm supports both with the same property names (document 06); the reaper runs on a dedicated thread as in QEMU's `kvm-reaper`. For TCG, ruvm-mem's dirty tracking (document 05) marks pages from the softmmu slow path and the JIT's store fast path through TLB_NOTDIRTY, as QEMU does.

### Convergence: auto-converge and dirty limit

Auto-converge (`auto-converge` capability) throttles all vCPUs by forcing them to sleep a percentage of each 10 ms period: `cpu-throttle-initial` 20, `cpu-throttle-increment` 10, `max-cpu-throttle` 99, `throttle-trigger-threshold` 50 (percent of dirtied bytes to transferred bytes that triggers throttling), `cpu-throttle-tailslow`. QEMU 9.2 made the dirty sync inside auto-converge more frequent on large hosts (every 5 seconds by default). ruvm matches these parameters and the throttle formula, because users tune them.

Dirty limit (`dirty-limit` capability, QEMU 8.1) needs the dirty ring and throttles only the vCPUs that exceed `vcpu-dirty-limit` (default 1 MB/s) as measured over `x-vcpu-dirty-limit-period` (default 1000 ms), by adjusting a per-vCPU sleep on each ring-full exit. It hurts read-mostly vCPUs far less than auto-converge. `calc-dirty-rate` and `query-dirty-rate` (modes page-sampling, dirty-bitmap, dirty-ring) are implemented for users who measure before migrating.

## Multifd

Multifd (`multifd` capability, `multifd-channels` default 2) sends RAM over N extra connections with page payload handled by sender threads. Each channel starts with an init packet:

```
be32 magic 0x11223344; be32 version 1; u8 uuid[16]; u8 channel id; u8 unused1[7]; be64 unused2[4]
```

then repeats packet header plus page data:

```
MultiFDPacket_t (packed, big-endian):
  be32 magic; be32 version; be32 flags        // SYNC bit 0, compression bits 1..5, DEVICE_STATE (32 << 1)
  be32 pages_alloc; be32 normal_pages; be32 next_packet_size; be64 packet_num
  be32 zero_pages; be32 unused32[1]; be64 unused64[3]
  char ramblock[256]
  be64 offset[normal_pages + zero_pages]
followed by normal page data (raw or compressed, next_packet_size bytes)
```

Compression values in flags: none 0, zlib 1 << 1, zstd 2 << 1, qpl 4 << 1, uadk 8 << 1, qatzip 16 << 1. Packet size is 512 KiB (`MULTIFD_PACKET_SIZE`). Device state packets (`MULTIFD_FLAG_DEVICE_STATE`, used by VFIO since QEMU 10.0) carry `idstr[256]`, `instance_id` and a length instead of page offsets.

Compression methods and ruvm's implementation:

| Method | QEMU since | ruvm | Notes |
|---|---|---|---|
| zlib | 5.0 | system zlib (FFI) | `multifd-zlib-level` default 1; per-channel deflate stream, flushed per packet |
| zstd | 5.0 | libzstd (FFI) | `multifd-zstd-level` default 1; per-channel stream |
| qpl | 9.1 | libqpl (FFI), feature gated | Intel IAA offload, deflate-compatible, falls back to software per page |
| uadk | 9.1 | libwd (FFI), feature gated | HiSilicon and other UADK accelerators |
| qatzip | 9.2 | QATzip (FFI), feature gated | Intel QAT, `multifd-qatzip-level` default 1 |

Streams for zlib and zstd are stateful per channel, so both sides must use the same library semantics; byte identity of compressed output is not required because only the decompressed pages matter, but the framing is.

Sync: the source sends `RAM_SAVE_FLAG_MULTIFD_FLUSH` in the main stream and `MULTIFD_FLAG_SYNC` packets on every channel at the points QEMU does (per dirty bitmap sync in current QEMU, per section on old machine types via the `multifd-flush-after-each-section` compat behavior). The destination waits for all channels to reach the sync before processing later main-stream data. Getting this exactly right is what the pre-9.0 to post-9.1 regression fixed in QEMU 10.0 was about, and ruvm's interop suite includes that pair explicitly.

`zero-copy-send` uses `MSG_ZEROCOPY` on Linux for uncompressed multifd, with locked memory requirements identical to QEMU. With TLS it is refused, as in QEMU.

## Postcopy

Postcopy (`postcopy-ram`) switches execution to the destination before all RAM has arrived. Missing pages fault on the destination, which requests them over the return path.

Sequence (source side commands): `POSTCOPY_ADVISE` during setup; precopy runs for a while; on `migrate-start-postcopy`, the source stops the VM, sends `POSTCOPY_RAM_DISCARD` lists for pages dirtied since they were sent (the destination drops them with `MADV_DONTNEED` or fallocate punch hole so they fault), then a `PACKAGED` blob containing device state, `POSTCOPY_LISTEN` and `POSTCOPY_RUN`. The destination loads device state while pages keep streaming, enters `postcopy-device` state (QEMU 10.2) until devices are loaded, then runs.

Destination fault handling uses userfaultfd (ruvm-sys): all guest RAM is registered with `UFFDIO_REGISTER_MODE_MISSING`; a fault thread reads fault events, sends `REQ_PAGES`, and the page is placed atomically with `UFFDIO_COPY` (or `UFFDIO_ZEROPAGE`), which wakes the vCPU. Hugepage-backed RAM is placed a whole huge page at a time, which is why the setup record carries the block page size. For shared memory backends a second mapping is used for placement, as QEMU does. `postcopy-blocktime` measures vCPU stall time per fault.

Postcopy preempt (`postcopy-preempt`, QEMU 7.1) adds a separate channel for urgent pages so a faulting vCPU does not wait behind a large precopy backlog in the main channel. The source services page requests from the return path on this channel first. QEMU 10.1 added an optimization for sequential access patterns; ruvm adds read-ahead of the next pages in the faulting region on the preempt channel, bounded to a small window so it does not starve requested pages. Postcopy can combine with multifd since 10.1, with multifd used during precopy only.

Recovery: if the network fails during postcopy, both sides enter `postcopy-paused` instead of dying; `migrate-recover` on the destination and `migrate` with `resume=true` on the source reconnect, exchange received bitmaps (`RECV_BITMAP`), and continue (`postcopy-recover-setup`, `postcopy-recover`). ruvm supports the same flows and states.

Failure policy is inherited from postcopy's nature: once the destination runs, the source no longer has a consistent copy, so loss of either side loses the VM. Documentation and QMP behavior say so exactly as QEMU does.

## Switchover acknowledgment

With `switchover-ack` (QEMU 8.1, requires `return-path`) the source will not stop the VM until the destination sends `MIG_RP_MSG_SWITCHOVER_ACK`. Destination devices that need preparation before switchover (VFIO devices that must receive their precopy initial data and set up hardware state) register as needing ack; when all have acked, the destination sends it. ruvm uses the same hook in the device-facing migration trait so VFIO in document 16 can opt in.

## Background snapshot

`background-snapshot` (QEMU 6.0) saves RAM as of the moment the snapshot starts while the VM keeps running, using userfaultfd write protect (`UFFDIO_WRITEPROTECT`): all RAM is write protected, the migration thread saves pages in order, and a vCPU write to a not-yet-saved page traps, the page is saved first, then unprotected. Device state is saved at the start with the VM briefly stopped. The output is a normal migration stream, so it loads with `-incoming`. Requires anonymous or shmem-backed RAM that supports uffd-wp; ruvm checks the same kernel features QEMU checks and rejects the capability with the same message otherwise.

## Mapped-ram and file migration

`mapped-ram` (QEMU 9.0) writes each RAM page at a fixed offset in a seekable file instead of appending records. For each RAM block the setup record is followed by a packed header:

```
be32 version (1); be64 page_size; be64 bitmap_offset; be64 pages_offset
```

`pages_offset` is aligned to 1 MiB (`MAPPED_RAM_FILE_OFFSET_ALIGNMENT`). The block's pages live at `pages_offset + page_offset`, and a bitmap at `bitmap_offset` (written at the end) says which pages are present. Zero pages are not written; the file keeps a hole. Device state follows at the end of the file. With multifd, each channel `pwrite`s its pages directly to their offsets, and `direct-io` (QEMU 9.1) opens the file with `O_DIRECT`. QEMU 10.2 extended mapped-ram to `snapshot-save` and `snapshot-load`.

This layout is the basis for ruvm's fast local restore below, because the guest RAM region of the file is directly mmappable.

## savevm and loadvm into qcow2

Internal snapshots store the whole VM (RAM plus devices) inside a qcow2 image. HMP `savevm`, `loadvm`, `delvm`, `info snapshots`, and the job based QMP commands `snapshot-save`, `snapshot-load`, `snapshot-delete` (with `tag`, `vmstate` node, and `devices` list) are all supported. The mechanics, matching `migration/savevm.c` `save_snapshot`:

1. Stop the VM, drain all block devices.
2. Write the migration stream (without the configuration section framing that live migration needs, same as QEMU) through `bdrv_save_vmstate` into the vmstate area of the chosen qcow2 node, which lives past the end of the virtual disk (document 14).
3. Create a qcow2 internal snapshot with the same tag on every writable, snapshot-capable node, recording `vm_state_size`, `date`, `vm_clock_nsec`, and `icount` (-1 when icount is off).
4. Resume if the VM was running.

`loadvm` reverses it: `bdrv_snapshot_goto` on all nodes then load the stream with `qemu_loadvm_state`. Every node must have the snapshot, and the error messages for missing ones must match. ruvm's snapshots are readable by QEMU and the reverse, and `qemu-img snapshot -l` prints them identically (document 14 gives the format).

## CheckPoint and Restart (CPR)

CPR is the family of migration modes where the VM moves to a new process on the same host. `migrate-set-parameters mode=...`:

- `cpr-reboot` (QEMU 8.2): stop, save to a file, quit; optionally update and reboot the host; start new QEMU with `-incoming`. Guest RAM is preserved only if the memory backend is shared and persists (for example a dax device) with `x-ignore-shared`; otherwise RAM is written to the file. VFIO requires the guest to be suspended first.
- `cpr-transfer` (QEMU 10.0, VFIO and IOMMUFD since 10.1): new process started with the main `-incoming` plus a second `-incoming` of channel type `cpr` on a UNIX socket. The old process sends file descriptors (memfd-backed RAM, VFIO device fds, IOMMUFD, KVM-independent device fds) with `SCM_RIGHTS` on the cpr channel, then the normal stream on the main channel with RAM skipped. Requires `share=on` memory backends and `-machine aux-ram-share=on`. QEMU 11.1 reduced downtime with a hash table for fd lookups.
- `cpr-exec` (QEMU 10.2): the old process `exec`s the new binary given by `cpr-exec-command`, keeping its PID, and passes state through a file channel.

ruvm implements all three with the exact same parameters, channel types and restrictions, including cross-implementation CPR: a QEMU 11.1 process can cpr-transfer to ruvm and the reverse, since the fd transfer protocol is the `cpr-state` vmstate described in `migration/cpr.c` and the stream is the normal one. This is the fastest path for migrating a fleet from QEMU to ruvm without moving VMs between hosts, and it is a listed M5 deliverable. The `cpr` fd list must name fds the way QEMU does (`cpr_save_fd(name, id, fd)`), which means ruvm's memory backends and VFIO code use QEMU's names for them.

## VFIO device migration integration

VFIO devices (document 16) migrate through the kernel's VFIO migration v2 uAPI: device states RUNNING, STOP, STOP_COPY, RESUMING, and optionally PRE_COPY and RUNNING_P2P, set with `VFIO_DEVICE_FEATURE_MIG_DEVICE_STATE`, with data read from and written to a data fd. The migration integration:

- VFIO registers an iterative handler (`vfio` idstr per device) so PRE_COPY data flows during precopy and STOP_COPY data at switchover, in the same section format as QEMU's `hw/vfio/migration.c`.
- With multifd (QEMU 10.0), device state goes over multifd channels as device state packets, loaded on the destination by per-device load threads; ruvm supports the same.
- Dirty tracking for DMA uses the IOMMUFD dirty tracking or VFIO device dirty tracking, feeding the same RAM dirty bitmap.
- `switchover-ack` gates switchover on the destination having consumed initial precopy data.
- P2P quiescing: when multiple VFIO devices exist, all move to RUNNING_P2P before any moves to STOP, as QEMU orders it.

## Fast local snapshot and restore for serverless

The serverless use case (Firecracker snapshot restore and systems built on it) wants a VM restored from a snapshot on local disk in a few milliseconds and paying only for pages it actually touches. The research on why naive restore is slow is clear. REAP (Ustiugov et al., [ASPLOS 2021](https://dl.acm.org/doi/10.1145/3445814.3446714)) measured that cold invocations from a snapshot were 95% slower on average than memory-resident ones, traced it to page faults bringing guest memory in one page at a time, observed that a function touches a stable working set across invocations, and by recording that set and prefetching it from disk in one read eliminated 97% of page faults and cut cold start time 3.7x on average. FaaSnap (Ao, Porter, Voelker, [EuroSys 2022](https://dl.acm.org/doi/10.1145/3492321.3524270)) added compact loading set files, per-region memory mapping based on region content (zero regions mapped anonymous, others to the file), hierarchical overlapping mappings, and concurrent paging so the guest starts running while the working set loads. Catalyzer (Du et al., [ASPLOS 2020](https://dl.acm.org/doi/10.1145/3373376.3378512)) restores from a checkpoint image with on-demand recovery of both memory and system state and adds `sfork` to clone running sandboxes, reporting sub-millisecond startup in its best case. Firecracker exposes the mechanism rather than the policy: its snapshot load API maps the memory file privately or hands a userfaultfd to an external page fault handler over a UNIX socket ([docs](https://github.com/firecracker-microvm/firecracker/blob/main/docs/snapshotting/handling-page-faults-on-snapshot-resume.md)).

ruvm's design takes these results and keeps QEMU compatibility:

1. Snapshot format. A ruvm local snapshot is a mapped-ram file produced by the standard `migrate` to `file:` with `mapped-ram` on (optionally multifd and `direct-io` for fast writes). QEMU can load it and ruvm can load QEMU's. No new format.
2. Direct mapping restore. Because each block's pages sit at a 1 MiB aligned offset, ruvm maps guest RAM for the restored VM as `mmap(MAP_PRIVATE)` of the file range `[pages_offset, pages_offset + used_length)`. Absent pages are file holes and read as zero; present pages come from the page cache on first touch. Writes are private copy-on-write, so the snapshot file is never modified and many VMs can restore from one file and share clean pages through the page cache. The device state at the end of the file is loaded normally. This requires the incoming path to accept a RAM block backed by a file mapping instead of copying into anonymous memory, which is new code in ruvm (`-incoming file:...` plus the ruvm-only property `x-restore-mode=map`). With it off, loading is QEMU's normal read path. With KVM the mapped range is used as the memslot directly.
3. Working set recording. With `x-record-working-set=path`, the restored VM runs with userfaultfd in missing mode (or `mincore` sampling for the mmap path) and logs the order of first touches per RAM block for a configurable window (default until the first `x-ws-mark` QMP command or 2 seconds). The result is a sidecar file: a list of `(block, page)` extents in first-touch order plus a compacted copy of those pages laid out contiguously, which is REAP's working set file.
4. Prefetch on restore. On the next restore, ruvm reads the compacted working set file with one large read (io_uring, `O_DIRECT`), and installs pages with `UFFDIO_COPY` into the guest mapping (userfaultfd mode) or populates the private mapping with `MADV_POPULATE_WRITE` over the extents (mmap mode). Following FaaSnap, vCPUs start immediately; prefetch runs concurrently, and faults for pages not yet installed are served on demand from the main snapshot file, jumping the prefetch queue.
5. Region policy. RAM blocks or ranges that the bitmap says are entirely zero are mapped anonymous (FaaSnap's per-region mapping). ROM and firmware blocks, which are identical across snapshots, can be served from a shared read-only mapping.
6. Page fault handler options. For users who already run Firecracker-style external handlers, ruvm can hand the userfaultfd to an external process over a UNIX socket using a protocol compatible with Firecracker's (fd plus a JSON list of regions with base address, size, offset and page size). This is a ruvm-only extension, off by default.

Limits: confidential VMs with guest_memfd (document 19) cannot be restored this way because private memory cannot be mapped from a file; restore of those goes through the full stream and whatever the platform allows. Snapshots taken on one host CPU model restore on another only within the CPU model rules of document 06. Clock and entropy state after restore is the guest's problem in QEMU too; ruvm offers the same `vmgenid` device so guests can reseed.

Targets for this path are defined in document 21. The mechanisms are the ones the papers measured; we will publish our own numbers rather than borrow theirs.

## Interop testing plan with QEMU

Compatibility is tested, not reasoned about. Document 22 owns the harness; this section defines the matrix.

- Static layer. For every machine type in both implementations, `-dump-vmstate` from QEMU 11.1 and from ruvm are compared with `scripts/vmstate-static-checker.py` (which checks field names, sizes, versions, subsections and flags) and with a stricter exact diff. Run on every pull request touching a device crate. This catches most bugs before any VM boots.
- Stream layer. For each machine type and a set of device configurations (minimal, typical libvirt q35 and virt guests, device-heavy configs with every migratable device in ruvm), boot a guest to a checkpoint in both QEMU and ruvm, migrate to file, and compare the vmdesc JSON and the parsed sections with `scripts/analyze-migration.py`. Payload differences are expected where state differs (timers, counters), so the comparison is per field with a small allowlist per device of fields that legitimately differ.
- Live layer, both directions. For each machine type version that QEMU 11.1 ships (the older i440FX and Q35 versions removed in QEMU 11.0 are out of scope) and every arm `virt`, s390x `s390-ccw-virtio`, ppc `pseries`, and riscv `virt` version: QEMU to ruvm and ruvm to QEMU, repeated three times (A to B to A to B), with a guest running a workload that checksums its memory and disk continuously and reports mismatch. Variants: plain precopy, multifd with each compression method available on the host, xbzrle, postcopy, postcopy with preempt, postcopy recovery with an injected network cut, auto-converge, dirty limit, TLS, mapped-ram file, background snapshot, savevm and loadvm in qcow2, cpr-transfer, and storage migration via NBD mirror (document 14).
- Old sources. Streams produced by QEMU releases still supported for incoming migration on those machine types (stored as fixtures from real QEMU builds of each release in the support window) must load into ruvm. This catches the per-release compat property mistakes that QEMU itself has shipped fixes for.
- Fuzzing. The loader is fuzzed with cargo-fuzz using recorded streams as seeds; a malformed stream must fail cleanly, never crash. QEMU treats the stream as trusted; ruvm treats it as untrusted because it is cheap to do so in Rust, and because file based snapshots cross trust boundaries in serverless setups.

M5 exit criteria: the static layer is clean for q35, pc, microvm and arm virt; the live layer passes for those at the latest machine version with the plain, multifd, postcopy and file variants in both directions; downtime is equal or lower than QEMU in the benchmark defined in document 21.

## Record/replay

QEMU's record/replay (`replay/`) makes an execution deterministic by running with instruction counting (icount) and logging every non-deterministic input. ruvm implements it in the JIT (document 08) because it needs precise instruction counts; hardware accelerators are excluded, as in QEMU.

Configuration is identical: `-icount shift=N|auto,rr=record|replay,rrfile=FILE,rrsnapshot=NAME`, plus `blkreplay` filter nodes over every disk (document 14) and `filter-replay` on netdevs. The log format is QEMU's, so a log recorded by one can be replayed by the other when the guest, machine and devices are identical, which is how we test determinism of the JIT against QEMU TCG:

- Header: 12 bytes, a big-endian 32-bit version `0xe0200e` then 8 reserved bytes, written at the end of recording.
- Events: one byte kind followed by kind-specific data. `EVENT_INSTRUCTION` carries a 32-bit count of instructions executed since the last event. Then `EVENT_INTERRUPT`, `EVENT_EXCEPTION`, `EVENT_ASYNC` plus a sub-kind (BH, BH_ONESHOT, INPUT, INPUT_SYNC, CHAR_READ, BLOCK, NET), `EVENT_SHUTDOWN` plus cause, `EVENT_CHAR_WRITE`, `EVENT_CHAR_READ_ALL`, `EVENT_CHAR_READ_ALL_ERROR`, `EVENT_AUDIO_OUT`, `EVENT_AUDIO_IN`, `EVENT_RANDOM`, `EVENT_CLOCK` plus clock kind (host, virtual_rt), `EVENT_CHECKPOINT` plus checkpoint kind (clock warp start and account, reset requested, suspend requested, clock virtual, host, virtual_rt, init, reset), and `EVENT_END`. Event numbering is by enum position, so ruvm uses QEMU's enum values exactly.
- Checkpoints order asynchronous events relative to timers and the main loop; async events (bottom halves from block completions, input, char reads, network packets) are queued at record time and replayed at the same checkpoint and instruction count.

The part that matters for ruvm's design is that determinism constrains the JIT and device model. vCPUs run in round robin on one thread in replay mode (no MTTCG), block completions are delivered only through `blkreplay`, and every device timer must use virtual clock ticks derived from icount. ruvm-jit's icount support counts instructions per TB with exits at the exact boundary where an event is due (document 08).

Reverse debugging: with `rrsnapshot`, the recorder takes periodic snapshots (the initial one named by `rrsnapshot`, stored as qcow2 internal snapshots via the savevm path above), and the gdbstub (ruvm-gdbstub) supports reverse step (`bs`) and reverse continue (`bc`) by loading the nearest earlier snapshot and replaying forward to the target icount, as `replay/replay-debugging.c` does. QMP `query-replay`, `replay-break`, `replay-delete-break`, and `replay-seek` are implemented. Because ruvm's snapshot restore is faster than QEMU's for large RAM (mapped-ram direct mapping), reverse steps over long recordings should be cheaper; we will measure that rather than claim it.

Prior art for comparison is rr (O'Callahan et al., [USENIX ATC 2017](https://www.usenix.org/conference/atc17/technical-sessions/presentation/ocallahan)), which records user-space processes with low overhead using hardware performance counters. ruvm's replay is whole-system and runs on the JIT; the two are complementary.

## COLO

COLO (COarse-grained LOck-stepping, Dong et al., [SoCC 2013](https://dl.acm.org/doi/10.1145/2523616.2523630)) runs a primary and a secondary VM in parallel, compares their network output, and only checkpoints the primary into the secondary when outputs diverge or a period expires. QEMU's implementation (`migration/colo.c`, `net/colo-compare.c`, `net/filter-mirror.c`, `net/filter-rewriter.c`, `block/replication.c`) is marked experimental (`x-colo` capability), and its state machine is:

1. Primary migrates to the secondary normally, then both enter COLO mode.
2. Primary's network traffic is mirrored to the secondary with `filter-mirror` and `filter-redirector`; `colo-compare` on the primary compares TCP, UDP and ICMP output packet streams from both and releases the primary's packets only when they match.
3. On mismatch or after `x-checkpoint-delay` (default 20 seconds, 200 * 100 ms), a checkpoint: stop both, send dirty RAM and device state from primary to secondary through the migration channel (with multifd since QEMU 11.0), the secondary loads it into a RAM cache first and flushes after the whole checkpoint is received, resume both.
4. Disks: primary writes are replicated to the secondary through NBD and the `replication` filter keeps a hidden and active disk so the secondary can roll back to the last checkpoint.
5. Failover: on heartbeat loss (`x-colo-lost-heartbeat`), the survivor continues. `query-colo-status` reports mode and reason.

ruvm implements COLO after M5, with the same objects, options and QMP commands, and it is tested only ruvm to ruvm and QEMU to ruvm in lockstep pairs at the latest machine version. Since QEMU 10.2 the stream no longer carries the `ENABLE_COLO` command (COLO state is negotiated through capabilities), and ruvm follows the new behavior only.

## Decisions recorded in this document

- Device state descriptions are tables interpreted by one codec, produced by `#[derive(VmState)]`, not generated per-struct serializers.
- ruvm treats incoming migration streams as untrusted and fuzzes the loader.
- ruvm's local snapshot format for fast restore is QEMU's mapped-ram file; restore can map guest RAM directly from the file (`x-restore-mode=map`), with REAP-style working set recording and FaaSnap-style concurrent prefetch as ruvm-only extensions (`x-record-working-set`, `x-ws-mark`).
- An optional Firecracker-compatible external userfaultfd handler protocol is offered, off by default.
- Cross-implementation cpr-transfer (QEMU to ruvm on the same host) is an M5 deliverable.
- COLO lands after M5 and only with the post-10.2 negotiation.
