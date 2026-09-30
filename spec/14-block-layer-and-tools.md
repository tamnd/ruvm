# 14. Block layer and storage tools (ruvm-block)

This document specifies ruvm-block, the crate that owns the block graph, every image format and protocol driver, block jobs, dirty bitmaps, I/O throttling, and the storage tools built on top of it: qemu-img, qemu-io, qemu-nbd and qemu-storage-daemon, plus the NBD, vhost-user-blk, FUSE and VDUSE exports. The reference is QEMU 11.1.0. The compatibility bar is the one set in document 02: the QAPI schema for block commands is byte identical to QEMU's `qapi/block-core.json`, `qapi/block.json`, `qapi/block-export.json` and `qapi/job.json`; images written by ruvm open in QEMU and the reverse; and QEMU's own `tests/qemu-iotests` suite (608 entries in the 11.1 tree, 110 of them named tests under `tests/qemu-iotests/tests/`) passes against ruvm binaries with the same reference output files. Block is milestone M3 (document 23), right after the first KVM boot, because nothing useful ships without disks.

## Why this is the part of QEMU we most want to rewrite

QEMU's block layer is the most heavily reworked subsystem in the tree over the last decade. It moved from AIO callbacks to coroutines, from `-drive` to `-blockdev` and a node graph, from the AioContext lock to a graph lock plus per-request thread safety (the AioContext lock was removed in QEMU 8.2), and from one iothread per device to true multiqueue, where virtio-blk (QEMU 9.0) and virtio-scsi (QEMU 10.0) can spread virtqueues over several iothreads with `iothread-vq-mapping`. Each step was done incrementally, so the code carries the intermediate states: `GRAPH_RDLOCK` annotations checked by clang thread safety analysis, generated `co_wrapper` functions (`block/block-gen.h`) that bounce between coroutine and non-coroutine context, and drain semantics behind a long series of deadlock fixes.

ruvm starts from the end state. The design constraints we take from QEMU are the observable ones: the graph shape, the permission rules, the option names, the on-disk formats, the job state machine, the QMP events and the tool output. Internals are free to differ, and they do: requests are Rust futures on ruvm-aio reactors, the graph is an RCU-published immutable snapshot, and every driver is written against a single async trait.

## Crate layout

ruvm-block is an L3 crate (document 03). It depends on ruvm-base, ruvm-aio, ruvm-qom, ruvm-qapi and ruvm-trace, and on the new ruvm-crypto crate described below. It does not depend on ruvm-mem or any device crate; devices (ruvm-hw-storage, ruvm-hw-virtio) depend on it through the `BlockBackend` handle. The tools in L5 (ruvm-img, ruvm-io, ruvm-nbd, ruvm-storage-daemon) depend on ruvm-block and ruvm-monitor.

```
ruvm-block/
  graph/       node, child, permission, graph publication, drain
  backend/     BlockBackend (the device-facing handle), root perms, stats, error policy
  io/          request types, alignment, bounce buffers, tracked requests, serialising
  format/      qcow2, qcow, qed, raw, vmdk, vdi, vhdx, vpc, parallels, dmg, cloop, bochs, vvfat, luks
  protocol/    file, host_device, host_cdrom, nbd, iscsi, nfs, curl, ssh, rbd, blkio, nvme, null
  filter/      throttle, copy-on-read, compress, preallocate, blkdebug, blkverify, blklogwrites,
               quorum, replication, blkreplay, copy-before-write, snapshot-access
  job/         job core (job.c), stream, commit, mirror, backup, create, amend, snapshot jobs
  bitmap/      dirty bitmaps, hbitmap equivalent, persistence hooks
  throttle/    leaky bucket (util/throttle.c), throttle groups
  export/      nbd server, vhost-user-blk server, fuse, vduse-blk
  nbd/         NBD protocol client and server codec (nbd/*.c equivalent)
```

New decision (for document 25): cryptography used by LUKS, legacy qcow AES, and migration TLS lives in a new L1 crate `ruvm-crypto` that mirrors QEMU's `crypto/` directory (cipher modes, IV generators, hash, PBKDF2, secrets and `--object secret`, TLS credentials). It is GPL-2.0-or-later because the LUKS header handling and IV generator semantics are ported from `crypto/block-luks.c` and `crypto/ivgen*.c`. The primitives come from RustCrypto crates and rustls; there is no OpenSSL, gnutls or nettle dependency.

## The block graph

### Nodes, children and roles

A node (`BlockDriverState` in QEMU, `Node` in ruvm) is one instance of one driver with its options, its open flags, its children and its parents. A child edge (`BdrvChild`) connects a parent to a node and carries a name (`file`, `backing`, `data-file`, `image`, `target`, `children.0` and so on), a role bitmask, and the permissions the parent holds and shares. Parents are either other nodes or non-node users: a `BlockBackend` (device or export), a block job, or a snapshot or migration helper.

Roles are the same bits as `include/block/block-common.h`: `BDRV_CHILD_DATA` (1), `BDRV_CHILD_METADATA` (2), `BDRV_CHILD_FILTERED` (4), `BDRV_CHILD_COW` (8), `BDRV_CHILD_PRIMARY` (16), with `BDRV_CHILD_IMAGE` defined as data plus metadata. Roles drive default permissions (`bdrv_default_perms`), what "filtered child" and "COW child" mean for backing chain walks (`bdrv_filter_or_cow_child`, `bdrv_skip_filters`), and which child gets `BDRV_O_*` flags inherited. The role of each named child for each driver must match QEMU exactly because it is visible through `query-named-block-nodes`, through which node `block-stream` or `block-commit` will accept as base, and through `x-blockdev-reopen` and `blockdev-reopen` behavior.

```rust
pub struct Node {
    pub node_name: NodeName,             // user given or auto "#block123"
    pub driver: &'static DriverInfo,
    pub state: Box<dyn BlockDriver>,
    pub opts: QDict,                     // full options, for query and reopen
    pub explicit_opts: QDict,
    pub open_flags: OpenFlags,           // BDRV_O_RDWR, NOCACHE, NO_FLUSH, INACTIVE...
    pub limits: BlockLimits,             // request_alignment, max_transfer, opt/pdiscard...
    pub children: SmallVec<[ChildRef; 2]>,
    pub parents: SmallVec<[ChildRef; 2]>,
    pub dirty_bitmaps: BitmapList,
    pub in_flight: AtomicU32,
    pub quiesce_counter: AtomicU32,
    pub write_threshold: AtomicU64,
    pub active: bool,                    // inactivated for migration
}

pub struct Child {
    pub name: SmolStr,
    pub role: ChildRole,
    pub perm: Perm,                      // what the parent holds
    pub shared_perm: Perm,               // what the parent lets others hold
    pub parent: ParentRef,               // node, backend, job, export
    pub node: NodeRef,
}
```

### Permissions

The permission model is QEMU's, bit for bit, because it produces user-visible errors such as `Conflicts with use by ... as 'root', which does not allow 'write' on #block123` that libvirt parses and that iotests compare. Permissions are `BLK_PERM_CONSISTENT_READ` (0x01), `BLK_PERM_WRITE` (0x02), `BLK_PERM_WRITE_UNCHANGED` (0x04) and `BLK_PERM_RESIZE` (0x08); the old `BLK_PERM_GRAPH_MOD` bit 0x10 is gone in QEMU and never existed in ruvm. Every child holds `perm` and allows `shared_perm`. A graph change is valid only if for every node, the union of `perm` of all parents is a subset of the intersection of `shared_perm` of all other parents. Drivers compute what they need from their own children via a `child_perm(child, role, reopen_queue, parent_perm, parent_shared) -> (Perm, Perm)` callback, as in `.bdrv_child_perm`.

The update is transactional, as in `block.c` `bdrv_list_refresh_perms` using `Transaction` from `util/transactions.c`: compute a topologically sorted list of affected nodes, run `check_perm` on each (which for `file-posix` includes taking or releasing OFD byte-range locks at offsets 100 plus the permission bit and 200 plus the bit, the locking scheme in `raw_apply_lock_bytes`), and either commit all or abort all. ruvm keeps the same lock byte layout so a QEMU process and a ruvm process on the same image see each other's locks. A distro `qemu-img info` against a disk a ruvm VM has open must fail with `Failed to get shared "write" lock` exactly as today.

### Graph lock and publication

QEMU's graph lock (`block/graph-lock.c`) is a reader-writer lock where readers are coroutines in any AioContext, counted per context, and the single writer (`bdrv_graph_wrlock`, main loop only) waits for all reader counts to drop to zero, usually inside a drained section (`bdrv_graph_wrlock_drained`). ruvm keeps the same two-level contract but implements the read side as an epoch read on an immutable graph snapshot:

- The graph topology (nodes, edges, roles, permissions) lives in an `Arc<GraphSnapshot>` published through ruvm-base RCU. The I/O path reads child pointers from the snapshot it entered with; a request never observes a half-applied change.
- Writers run only on the main thread under the control lock (document 03). A writer builds a new snapshot, drains the affected subtree, publishes, waits one grace period, then undrains.
- A request future holds its epoch guard only while it walks the graph, not while it waits for I/O. Instead, each in-flight request increments `in_flight` on every node it touches, which is what drain waits for.

The hot path pays one atomic load and one increment per node per request, with no reader counters shared across iothreads.

### Drain

Drain is the operation that makes a subtree quiescent: no new requests start, all in-flight requests complete. QEMU's semantics (`block/io.c` `bdrv_drained_begin`, `bdrv_drained_end`, `bdrv_drain_all_begin`) are observable because jobs pause at drain points, because `BlockBackend` queues requests from devices during drain (`blk_wait_while_drained`), and because parents are notified through `drained_begin`, `drained_end` and `drained_poll` callbacks. ruvm's drain:

1. Increments `quiesce_counter` on the node and propagates to parents (not children, matching current QEMU, where `bdrv_do_drained_begin` quiesces parents through `bdrv_parent_drained_begin`).
2. Each `BlockBackend` above sees the counter and parks new requests in a per-queue wait list. Jobs see it via their child and yield at the next pause point.
3. The main thread polls `in_flight` of the subtree to zero by running its own reactor and waking iothread reactors (a cross-thread notify, not a spin).
4. `drained_end` releases parked requests in submission order per queue.

Nested drains nest. Drain from inside an iothread (for example a job completion that reopens a node) is posted to the main thread instead of polled in place; QEMU's "drain from coroutine" path (`bdrv_co_yield_to_drain`) does the same thing.

## The BlockDriver trait

All formats, protocols and filters implement one trait. Methods are async, run on whatever iothread the request came from, and must be `Send`. Optional methods have defaults that return `ENOTSUP` or pass through to the primary child, mirroring how QEMU treats a NULL callback.

```rust
#[async_trait(?Send)]  // concrete futures are Send; boxed only at dyn boundaries
pub trait BlockDriver: Send + Sync + 'static {
    fn info() -> &'static DriverInfo where Self: Sized;   // format_name, protocol_name, flags
    fn probe(buf: &[u8], filename: &str) -> u8 where Self: Sized { 0 } // 0..100 like bdrv_probe
    fn parse_filename(filename: &str, opts: &mut QDict) -> Result<()> where Self: Sized { Ok(()) }

    async fn open(ctx: OpenCtx<'_>, opts: &mut QDict, flags: OpenFlags) -> Result<Box<dyn BlockDriver>> where Self: Sized;
    async fn close(&mut self) {}
    fn refresh_limits(&self, node: &NodeView) -> Result<BlockLimits>;
    fn child_perm(&self, c: &ChildView, role: ChildRole, parent: Perm, shared: Perm) -> (Perm, Perm);

    async fn preadv(&self, n: &NodeView, off: u64, bytes: u64, qiov: &mut IoVec, f: ReqFlags) -> Result<()>;
    async fn pwritev(&self, n: &NodeView, off: u64, bytes: u64, qiov: &IoVec, f: ReqFlags) -> Result<()>;
    async fn pwrite_zeroes(&self, n: &NodeView, off: u64, bytes: u64, f: ReqFlags) -> Result<()> { Err(ENOTSUP) }
    async fn pdiscard(&self, n: &NodeView, off: u64, bytes: u64) -> Result<()> { Err(ENOTSUP) }
    async fn flush_to_os(&self, n: &NodeView) -> Result<()> { Ok(()) }
    async fn flush_to_disk(&self, n: &NodeView) -> Result<()> { Ok(()) }
    async fn block_status(&self, n: &NodeView, want_zero: bool, off: u64, bytes: u64) -> Result<Status>;
    async fn copy_range_from(&self, ...) -> Result<()> { Err(ENOTSUP) }
    async fn truncate(&self, n: &NodeView, off: u64, exact: bool, prealloc: PreallocMode, f: ReqFlags) -> Result<()>;

    fn get_info(&self) -> Option<ImageInfoSpecific> { None }   // qemu-img info "Format specific information"
    async fn check(&self, res: &mut CheckResult, fix: CheckFix) -> Result<()> { Err(ENOTSUP) }
    async fn amend(&mut self, opts: &QemuOpts, force: bool, job: &JobHandle) -> Result<()> { Err(ENOTSUP) }
    async fn snapshot_create/goto/delete/list(...)                // internal snapshots
    async fn save_vmstate/load_vmstate(&self, qiov, pos) -> Result<()>  // savevm payload area
    fn reopen_prepare/commit/abort(...)                           // blockdev-reopen transaction
    fn inactivate/activate(...)                                   // migration handover
    fn bitmap_ops(&self) -> Option<&dyn PersistentBitmaps> { None }
}
```

`DriverInfo` carries `format_name`, `protocol_name`, filter and backing flags, and `create_opts`. `create_opts` is part of the contract: `qemu-img create -o help` prints them, `qemu-img create` prints them in the `Formatting ...` line in declaration order, and QAPI `blockdev-create` has its own per-driver struct. Both paths are generated from the QAPI schema plus a table ported from each driver's `QemuOptsList`, and a unit test diffs the generated `-o help` text against QEMU's.

Drivers register with `ruvm_block::register_driver!` into a linkme distributed slice (document 04), so format and protocol support is decided by which crates and cargo features are linked. `-drive` probing uses the same probe scores as QEMU (`raw` is never probed as a winner over a real format, and probing an image as raw on a writable device prints the same warning).

### Request path in the core

Between `BlockBackend` and the driver sits the generic request layer, the equivalent of `block/io.c`:

- Alignment: if the request is not aligned to `request_alignment`, the core does read-modify-write with a bounce buffer, and marks the request serialising so overlapping writes wait (`bdrv_make_request_serialising`).
- Splitting at `max_transfer` and `max_pwrite_zeroes`, fallback from write zeroes to a zeroed buffer write when the driver says `ENOTSUP` and `BDRV_REQ_MAY_UNMAP` rules allow.
- Copy-on-read when the backend or the `copy-on-read` filter asks for it.
- Tracked requests per node in an interval tree (QEMU uses a list; we need it for `serialising` checks and `reqlist` in block-copy, and the interval tree turns O(n) scans into O(log n)).
- Before-write notifiers used by `write-threshold` and dirty bitmap updates.
- Accounting (`block/accounting.c`) feeding `query-blockstats` with the same fields.

## Formats

Every format below reads and writes images byte compatible with QEMU. "Write compatible" means a ruvm-written image passes `qemu-img check` from QEMU 11.1 with zero errors and zero leaks, and QEMU-written images pass ruvm's check, tested in CI for each format feature combination (document 22).

### qcow2

qcow2 is the format that matters most and gets the most engineering. The on-disk layout follows `docs/interop/qcow2.rst`:

- Header: magic `QFI\xfb`, version 2 or 3, backing file offset and size (max 1023 bytes), `cluster_bits` 9 to 21 (QEMU's implementation limit of 2 MiB clusters, `MAX_CLUSTER_BITS 21`), virtual size, `crypt_method` (0 none, 1 legacy AES, 2 LUKS), L1 size and offset, refcount table offset and clusters, snapshot count and offset. For v3: `incompatible_features` (bit 0 dirty, bit 1 corrupt, bit 2 external data file, bit 3 compression type, bit 4 extended L2), `compatible_features` (bit 0 lazy refcounts), `autoclear_features` (bit 0 bitmaps, bit 1 raw external data), `refcount_order`, `header_length`, and the byte at offset 104 with the compression type (0 deflate, 1 zstd) padded to a multiple of 8.
- Header extensions: end (0), backing format (0xe2792aca), feature name table (0x6803f857), bitmaps (0x23852875), full disk encryption header pointer (0x0537be77), external data file name (0x44415441). Unknown extensions are preserved on rewrite, as QEMU does.
- Two-level mapping: L1 entries (bits 9 to 55 offset, bit 63 "copied") and L2 entries (bit 62 compressed, bit 63 copied, bit 0 of the standard descriptor "reads as zero" in v3). With a 64 KiB cluster, one L2 table maps 512 MiB. QEMU caps the L1 at 32 MiB (`QCOW_MAX_L1_SIZE`).
- Refcounts with `refcount_order` 0 to 6 (1 to 64 bit refcounts, default 16).

Features that must all be supported, read and write:

- Extended L2 entries and subclusters. With `extended_l2=on` each L2 entry is 128 bits and a cluster is 32 subclusters, with a 32-bit allocation bitmap and a 32-bit "reads as zero" bitmap. This requires `cluster_bits` of at least 14. The main win is that COW on a partial write only copies the touched subcluster range, so a 4 KiB guest write into a fresh 128 KiB cluster with a backing file copies 4 KiB, not 128 KiB. ruvm's allocation path is written subcluster first, and the plain L2 path is the special case with one subcluster.
- External data files (`data_file`, `data_file_raw`). The guest offset equals the host offset in the data file, L2 entries may have offset 0 with bit 63 set, and with `data_file_raw=on` the data file is a valid raw image. Since QEMU 10.2 image creation refuses protocol prefixes in the data file name, and QEMU 11.0 added `keep_data_file` to `qemu-img create` to wrap an existing raw file. ruvm matches both.
- Compression with deflate (raw deflate with no zlib header, window bits -12) and zstd. Compressed cluster descriptors pack the host offset and the count of additional 512-byte sectors into bits 0 to 61, with the split at `62 - (cluster_bits - 8)`. Compressed writes only happen whole cluster at a time and only into unallocated clusters, as in `qcow2_co_pwritev_compressed_part`. New decision: ruvm links system zlib and libzstd by default so that compressed output is byte identical to QEMU's (iotests print host offsets and `qemu-img map` output that depend on compressed sizes). Pure Rust backends (zlib-rs and a Rust zstd) are behind a cargo feature for builds that want no C, with the known cost that a handful of iotests produce different offsets.
- Encryption: LUKS (`encrypt.format=luks`) stores a LUKS1 header in clusters pointed to by the 0x0537be77 extension, and data is encrypted per 512-byte sector with the configured cipher, mode and IV generator (`plain64` by default, `essiv` supported). Legacy AES (`crypt_method` 1) is supported for reading and for `qemu-img convert` and `amend` only, matching QEMU, which refuses to run a guest from it.
- Persistent dirty bitmaps: the bitmap directory (name, granularity, flags `in_use`, `auto`, and the bitmap table of cluster offsets), with the autoclear bit protecting against older writers. On open, a bitmap with `in_use` set is inconsistent and must be reported as such (`query-block` `inconsistent: true`) and can only be removed. On close or inactivation, bitmaps are stored and `in_use` cleared.
- Internal snapshots: the snapshot table (id, name, L1 copy, date, `vm_clock`, `vm_state_size`, extra data with 64-bit `vm_state_size`, virtual disk size, and `icount`). Creation copies the L1 and increments refcounts; goto rebuilds the copied flags. The VM state area for savevm lives past the end of the virtual disk at `qcow2_vm_state_offset`, addressed by the `save_vmstate` and `load_vmstate` driver methods (document 17 uses this).
- Lazy refcounts, the dirty bit, the corrupt bit and the `qcow2_signal_corruption` path, which marks the image corrupt and emits `BLOCK_IMAGE_CORRUPTED` with the same fields.
- Preallocation modes off, metadata, falloc, full; `refcount_bits`; `cluster_size`; `compat=0.10|1.1` and downgrade via amend; `amend` of `refcount_bits` (full refcount rebuild), encryption keyslots, `data_file_raw`, and `lazy_refcounts`.
- Discard options (`pass-discard-request`, `pass-discard-snapshot`, `pass-discard-other`, `discard-no-unref`) and `overlap-check` modes (none, constant, cached, all) with the individual `overlap-check.*` switches.

Performance design for qcow2:

- The L2 and refcount caches (`block/qcow2-cache.c`) become sharded caches keyed by table offset with per-shard locks. `l2-cache-size`, `l2-cache-entry-size`, `refcount-cache-size`, `cache-size` and `cache-clean-interval` keep QEMU's names and defaults (on Linux the default L2 cache is enough to cover the whole disk up to a 32 MiB cap, with a 600 second clean interval; elsewhere 8 MiB and no cleaning). Entry size smaller than the cluster (QEMU allows partial L2 table caching) is supported, and it is the key to making random reads over multi-terabyte images cheap.
- Allocation is serialised per image through an async mutex around the allocator only, not around whole requests. Data writes to already allocated clusters never take it. QEMU gets the same effect with `s->lock` released around data I/O; we make the scope explicit.
- In-flight allocations are tracked as `QCowL2Meta` equivalents in the interval tree, so a second write to a cluster being allocated waits on the first instead of allocating twice (the `handle_dependencies` logic).
- Compression and encryption run on a small worker pool (QEMU uses `QCOW2_MAX_THREADS` 4 threads in `qcow2-threads.c`). ruvm uses the per-iothread CPU pool described in document 21, bounded by the same default, so `qemu-img convert -c` scales with `-m` coroutines the same way.

### Other formats

| Format | Source file | Read | Write/create | Check | Notes |
|---|---|---|---|---|---|
| raw | raw-format.c | yes | yes | no | `offset` and `size` options for partition slices; probing guard |
| qcow (v1) | qcow.c | yes | yes | no | legacy AES; kept for convert |
| qed | qed*.c | yes | yes | yes | L1/L2 with 64 KiB default cluster, need-check flag, timer driven clear |
| vmdk | vmdk.c | yes | yes | yes | monolithicSparse, monolithicFlat, twoGbMaxExtentSparse/Flat, streamOptimized, seSparse (read); descriptor file parsing, extents restricted to local paths since 10.2 |
| vdi | vdi.c | yes | yes | yes | static and dynamic |
| vhdx | vhdx*.c | yes | yes | yes | log replay on open, BAT, metadata region; log replay needs write access |
| vpc | vpc.c | yes | yes | no | fixed and dynamic VHD; `force_size`; Azure fixes from 10.0 |
| parallels | parallels*.c | yes | yes | yes | format extension with dirty bitmaps |
| dmg | dmg*.c | yes | no | no | zlib, bzip2, lzfse chunks (bzip2 and lzfse via cargo features) |
| cloop | cloop.c | yes | no | no | |
| bochs | bochs.c | yes | no | no | |
| vvfat | vvfat.c | yes | limited | no | directory as FAT12/16/32 disk, `rw=on` writes back through a qcow shadow |
| luks | crypto.c | yes | yes | no | LUKS1 standalone image, same header code as qcow2 LUKS |

`check` is supported exactly for qcow2, qed, parallels, vhdx, vmdk and vdi, and `qemu-img check` on any other format exits 63, as QEMU documents.

### Filters

Filters are nodes with one primary filtered child (`BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY`) and `is_filter` set, so chain walks skip them.

- `throttle`: attaches to a throttle group (below).
- `copy-on-read`: with the `bottom` option limiting which backing nodes are copied from, as used by `block-stream`.
- `compress`: forces compressed writes on a format that supports them.
- `preallocate`: extends the underlying file in large steps ahead of writes (`prealloc-align`, `prealloc-size`), a large win on filesystems where every extending write is expensive.
- `blkdebug`: rule based error injection on named events (`l1_update`, `refblock_alloc` and the full `BlkdebugEvent` enum), plus `align`, `max-transfer` and similar limit overrides. iotests depend on it, which inject errors at exact metadata update points, so ruvm's qcow2 must emit the same events at the same points in its update sequence. This is a real constraint on the qcow2 implementation, and it is listed in document 25 as a known source of porting friction.
- `blkverify`: reads from two children and compares.
- `blklogwrites`: dm-log-writes compatible log.
- `quorum`: votes across N children with `vote-threshold`, `rewrite-corrupted`, `read-pattern` quorum or fifo, emits `QUORUM_REPORT_BAD` and `QUORUM_FAILURE`.
- `replication`: used by COLO (document 17).
- `blkreplay`: serialises block completions into the record/replay log (document 17).
- `copy-before-write`: the core of backup and image fleecing; before a guest write reaches the source, the old data is copied to the target through block-copy, with `bitmap`, `on-cbw-error` (added to QMP in QEMU 10.1) and `cbw-timeout`.
- `snapshot-access`: reads a point-in-time view from a copy-before-write node for fleecing exports.

## Protocols

### file, host_device, host_cdrom

`block/file-posix.c` is where most real I/O ends up, and it is where ruvm's performance lead has to come from. Options keep QEMU's names: `aio=threads|native|io_uring`, `aio-max-batch`, `locking=auto|on|off`, `pr-manager`, `x-check-cache-dropped`, `drop-cache`, and cache modes mapped to `O_DIRECT` and flush behavior (`cache.direct`, `cache.no-flush`).

- `aio=io_uring` is the default on Linux in ruvm when `cache.direct=on`, because ruvm-aio already runs one io_uring per iothread. This is a new decision; QEMU's default stays `threads`. Since the difference is internal and `query-named-block-nodes` reports the configured value, we report `threads` when the user did not set it, to keep management tools seeing QEMU's defaults, and document the real engine in trace output.
- `aio=native` (linux-aio) is kept for compatibility, implemented as io_uring submission underneath on kernels that support it, with a true libaio fallback for old kernels.
- `aio=threads` uses a bounded thread pool (QEMU's `thread-pool.c`, 64 threads max by default).
- FUA writes use `RWF_DSYNC` where supported, as QEMU does since 10.0, instead of write plus `fdatasync`.
- `host_device` handles block devices, zoned devices (`BLKREPORTZONE`, zone append emulation as in QEMU 8.0 and later), SCSI generic passthrough via `scsi-generic` and persistent reservation helpers.
- macOS uses kqueue plus a thread pool for file I/O, `F_FULLFSYNC` for flush-to-disk; Windows (`file-win32.c`) uses IOCP.

### Network and userspace protocols

| Protocol | QEMU file | ruvm implementation | Notes |
|---|---|---|---|
| nbd | block/nbd.c, nbd/client.c | native Rust | structured replies, extended headers (64-bit lengths, QEMU 8.2), block status meta contexts `base:allocation` and `qemu:dirty-bitmap:*`, `qemu:allocation-depth`, TLS, reconnect-delay, open-timeout |
| iscsi | block/iscsi.c | libiscsi via FFI | same URL syntax and options, iSER when libiscsi has it |
| nfs | block/nfs.c | libnfs via FFI | libnfs v6 API supported (QEMU 11.0), including zero copy reads into a single-vector qiov and `refresh_limits` for the dropped request splitting; v5 API also supported |
| http/https/ftp/ftps | block/curl.c | libcurl via FFI | readahead, cookies, sslverify, `force-range` (QEMU 11.0) |
| ssh | block/ssh.c | libssh via FFI | host key checking modes match |
| rbd | block/rbd.c | librbd via FFI | encryption formats luks and luks2, layered |
| blkio | block/blkio.c | libblkio via FFI | drivers `io_uring`, `nvme-io_uring`, `virtio-blk-vfio-pci`, `virtio-blk-vhost-user`, `virtio-blk-vhost-vdpa` |
| nvme | block/nvme.c | native Rust over vfio | userspace NVMe driver, one queue pair per iothread |
| null-co, null-aio | block/null.c | native | benchmarking |

GlusterFS is not implemented. QEMU 11.1 removed the gluster driver after deprecating it in 9.2, so the reference schema has no `gluster` member in `BlockdevDriver`, and adding one would break schema identity. This is a recorded decision.

NBD is native Rust because it sits on every live storage migration path (libvirt mirrors to an NBD target) and the protocol is small and well specified.

## Block jobs

Jobs follow `job.c` and `blockjob.c`. The state machine is `JobStatus`: created, running, paused, ready, standby, waiting, pending, aborting, concluded, null, with the exact transition table `JobSTT` and verb table `JobVerbTable` so that `job-pause`, `job-resume`, `job-complete`, `job-finalize`, `job-dismiss` and `job-cancel` fail with the same error classes. `auto-finalize` and `auto-dismiss` default to true. Events are `JOB_STATUS_CHANGE`, `BLOCK_JOB_COMPLETED`, `BLOCK_JOB_CANCELLED`, `BLOCK_JOB_READY`, `BLOCK_JOB_ERROR`, `BLOCK_JOB_PENDING`. `on-source-error` and `on-target-error` take `report|ignore|enospc|stop|auto`. The legacy `block-job-*` commands exist with their deprecation status from QEMU 10.1. Rate limiting uses `speed` in bytes per second with the same 100 ms slice logic (`ratelimit.h`).

Jobs are async tasks on the node's home iothread. A job yields at pause points (`job_pause_point`), which is where drain and `job-pause` take effect.

- stream: copies data from a range of the backing chain into the top node, then drops the backing link. Uses a `copy-on-read` filter with `bottom` set, like QEMU since 6.0. `base`, `base-node`, `bottom`, `backing-file`, `backing-mask-protocol`.
- commit: two variants. Active commit (top is the active layer) is a mirror job internally, reaching `ready` and needing `job-complete`. Intermediate commit copies from `top` into `base` through a `commit_top` filter. Non-active commit keeps sparseness (QEMU 10.1): zero ranges are written as zeroes with unmap, not as data.
- mirror (`drive-mirror`, `blockdev-mirror`): the dirty bitmap driven loop from `block/mirror.c`, with `granularity`, `buf-size`, `sync=full|top|none`, `copy-mode=background|write-blocking` (switchable at runtime via `block-job-change`), `target-is-zero` (QEMU 10.1), and the zero detection that skips work on zero blocks. `mirror_top` filter intercepts guest writes in write-blocking mode. The in-flight limit (16 operations by default via `MAX_IN_FLIGHT`) and chunk sizing match QEMU so that iotests counting operations pass.
- backup (`blockdev-backup`, `drive-backup`): implemented as a copy-before-write filter plus a background copier through block-copy, `sync=full|top|incremental|bitmap|none`, `bitmap-mode=on-success|never|always`, `x-perf` options, `compress`, and `on-cbw-error`.
- create (`blockdev-create`): runs a driver's `co_create` as a job.
- amend (`x-blockdev-amend`): runs a driver's amend as a job (qcow2 LUKS keyslot management).
- snapshot-save, snapshot-load, snapshot-delete: the job based internal snapshot commands, defined in document 17.

Block-copy (`block/block-copy.c`) is shared by backup and copy-before-write and has its own logic: it splits work into tasks sized by cluster, uses `copy_range` where both sides support it (reflinks, `copy_file_range`), and falls back to read and write. ruvm keeps its "call state" API so the async backup loop and the synchronous copy-before-write path share one in-flight set.

## Dirty bitmaps

Dirty bitmaps (`block/dirty-bitmap.c`, `util/hbitmap.c`) track changed regions with a configurable granularity (512 bytes to 2 GiB, power of two, default 64 KiB or the cluster size). QMP surface: `block-dirty-bitmap-add`, `-remove`, `-clear`, `-enable`, `-disable`, `-merge`, the `x-debug-block-dirty-bitmap-sha256` debug command, transactions over all of them, and `persistent` bitmaps stored in qcow2 or parallels.

The in-memory structure is a hierarchical bitmap like hbitmap: a leaf level of u64 words plus summary levels where each bit says "some bit set below", so iteration over sparse bitmaps skips empty regions in O(levels). ruvm's version is lock-free for setting bits (atomic `fetch_or` on leaf and summary words), which lets the write path mark dirty from any iothread without the bitmap mutex QEMU uses (`bdrv_dirty_bitmap_lock`). Readers that need a stable view (mirror iteration, migration of bitmaps, `merge`) take a short exclusive section. Bitmaps have status flags: `recording`, `busy` (in use by a job or export), `persistent`, `inconsistent`; operations on a busy bitmap fail with QEMU's exact messages.

Bitmaps migrate with the `dirty-bitmaps` migration capability through the `dirty-bitmap` section format in `migration/block-dirty-bitmap.c`, which document 17 covers.

## Throttling and throttle groups

I/O limits (`block_set_io_throttle`, the `throttling.*` options on `-drive`, and the `throttle` filter with `throttle-group` objects) use `util/throttle.c`'s leaky bucket model: for each of bps total, read, write and iops total, read, write there is an average rate, a burst max, and a burst length in seconds, plus `iops-size` to count large requests as several operations. Throttle groups share limits across several nodes and schedule them round robin (`block/throttle-groups.c`). ruvm matches the math exactly, because iotest 093 and others check request timing against a virtual clock, and because users size limits from QEMU experience. Timers run on the group's clock, which is `QEMU_CLOCK_VIRTUAL` under qtest and `QEMU_CLOCK_REALTIME` otherwise, as in QEMU.

Implementation difference: in QEMU a throttle group has a mutex and per-AioContext timers. In ruvm, a group has one atomic token bucket per direction and a scheduler task pinned to the iothread of the first member. Members on other iothreads enqueue into a lock-free queue. Round robin order is preserved and the fast path takes no mutex.

## I/O path design for performance

The target from the canon is virtio-blk throughput and latency equal to or better than QEMU with iothreads on the same host. The design choices that get there:

1. Per-iothread queues end to end. A `BlockBackend` has one submission context per iothread that uses it. With `iothread-vq-mapping` (virtio-blk since QEMU 9.0, virtio-scsi since 10.0; document 13), each virtqueue's requests are parsed, submitted and completed on its own iothread with no cross-thread handoff, and each iothread has its own io_uring. QEMU's multiqueue block layer allows the same topology; ruvm makes it the only design, so the single-queue case is just N equals 1.
2. Batching. Requests from one virtqueue notification are collected and submitted with one `io_uring_enter`, the same idea as QEMU's `blk_io_plug` and `aio-max-batch`, but the batch boundary is the end of the virtqueue pop loop rather than a plug counter.
3. Zero copy. Guest RAM is mapped in the process (document 05). A virtio-blk request's descriptors become an `IoVec` that points straight into guest memory, and for raw on `file` with `O_DIRECT` that `IoVec` is what io_uring reads into. There is no bounce unless alignment requires it. For qcow2, data I/O to allocated clusters is also zero copy; only COW, compression and encryption touch intermediate buffers. With fixed buffers registered (`IORING_REGISTER_BUFFERS`) over guest RAM regions, the kernel skips per-request page pinning; we register lazily per RAM block and fall back to normal reads if registration fails due to memlock limits.
4. Polling. Each iothread polls its io_uring completion queue and its virtqueue notifiers for an adaptive window (`poll-max-ns`, `poll-grow`, `poll-shrink`, the same iothread properties as QEMU), then sleeps on an eventfd. With `IORING_SETUP_SQPOLL` optional per iothread for users with spare cores, and `IORING_SETUP_IOPOLL` used by the nvme-io_uring blkio path.
5. Completion in place. Completions for a virtqueue are pushed to the used ring and the guest is notified with one irqfd write per batch, subject to event index suppression.
6. No per-request allocation. Request structs come from a per-iothread slab; futures for the common raw and qcow2 read and write paths are fixed-size and never boxed.

Document 21 defines the benchmark matrix (fio 4 KiB random read and write at queue depth 1 and 128, 1 to 8 iothreads, raw on NVMe and qcow2 on NVMe) and the pass criteria against QEMU 11.1 on the same machine.

## Tools

All tools are the `ruvm` multi-call binary dispatched on argv[0] (canon). Each tool's option parsing, error text, and exit codes are ported from the C source and checked against iotests reference output. Error messages are printed as `qemu-img: <message>` with the program name taken from argv[0] exactly as `error_report` does, so a symlink named `qemu-img` prints `qemu-img:`.

### qemu-img

The subcommands are the `DEF` list in `qemu-img-cmds.hx`: amend, bench, bitmap, check, commit, compare, convert, create, dd, info, map, measure, snapshot, rebase, resize. Common options: `--object`, `--image-opts`, `-f`, `-q`, `-U` (force share), `-T` source cache, `-t` cache, `--output=human|json`. JSON output is the QAPI type serialised with QEMU's pretty printer (four space indent and the same key order), which comes for free from ruvm-qapi's generated serializer since QAPI order is struct declaration order.

| Subcommand | Output (human) | Exit codes |
|---|---|---|
| create | `Formatting 'NAME', fmt=FMT <opts in create_opts order> size=N ...`, for qcow2: `fmt=qcow2 cluster_size=65536 extended_l2=off compression_type=zlib size=N lazy_refcounts=off refcount_bits=16` | 0 ok, 1 error |
| info | `image:`, `file format:`, `virtual size: 128 MiB (134217728 bytes)`, `disk size:`, `cluster_size:`, snapshot list, `Format specific information:` block, `Child node` sections with `--backing-chain`, `--limits` (QEMU 10.2) | 0, 1 |
| check | `No errors were found on the image.`, or counts: `N errors were found on the image.`, `N leaked clusters were found on the image.`, `Image end offset: N`, repair lines with `-r` | 0 consistent, 1 check not completed, 2 corrupted, 3 leaks only, 63 unsupported format |
| compare | `Images are identical.` or `Content mismatch at offset N!`, `Warning: Image size mismatch!` | 0 identical, 1 differ, 2 open error, 3 allocation check error, 4 read error |
| convert | silent unless `-p` progress `    (12.34/100%)` | 0, 1 |
| commit | `Image committed.` | 0, 1 |
| map | `Offset Length Mapped to File` table, or JSON array with `start`, `length`, `depth`, `present`, `zero`, `data`, `compressed`, `offset` | 0, 1 |
| measure | `required size: N`, `fully allocated size: N`, and for qcow2 targets `bitmaps size: N` | 0, 1 |
| snapshot | `-l` prints `Snapshot list:` then header `ID TAG VM_SIZE DATE VM_CLOCK ICOUNT` in the `%-7s %-16s %8s %19s %15s %10s` layout from `bdrv_snapshot_dump` | 0, 1 |
| rebase | silent, `-u` unsafe, `-c` compress | 0, 1 |
| resize | `Image resized.` | 0, 1 |
| amend | silent, progress with `-p` | 0, 1 |
| bitmap | silent | 0, 1 |
| dd | silent | 0, 1 |
| bench | `Sending N requests, N bytes each, N in parallel (starting at offset 0, step size N)` then `Run completed in N seconds.` | 0, 1 |

Sizes in human output use `size_to_str` semantics (binary prefixes with up to 3 significant digits after truncation, for example `128 MiB`). `convert` is the command most worth optimizing, and ruvm keeps QEMU's parallelism model (`-m` up to 16 concurrent coroutines, `-W` out-of-order writes) while running on io_uring with larger default batch sizes, `copy_file_range` offload with `-C`, and zero detection that turns zero reads into `write_zeroes` or skips them with `--target-is-zero`. `--bitmaps` and `--skip-broken-bitmaps` copy persistent bitmaps.

### qemu-io

qemu-io is the iotests workhorse. Its command set (`qemu-io-cmds.c`) is ported exactly: `read`, `readv`, `write`, `writev`, `aio_read`, `aio_write`, `aio_flush`, `flush`, `discard`, `truncate`, `length`, `info`, `map`, `alloc`, `break`, `remove_break`, `resume`, `wait_break`, `abort`, `sigraise`, `sleep`, `reopen`, `zone_*` and the rest, with the same flags (`-P pattern`, `-v`, `-q`, `-C`, `-z`, `-u`, `-f` for FUA and so on). Output lines such as `wrote 4096/4096 bytes at offset 0` and the timing line `4 KiB, 1 ops; 00.00 sec (...)` are printed in the same format, since `_filter_qemu_io` in `tests/qemu-iotests/common.filter` only masks the rates and times. qemu-io is also reachable from QMP and HMP as `human-monitor-command "qemu-io ..."`, which iotests use heavily.

### qemu-nbd

qemu-nbd exports one image over NBD or attaches it to a kernel `/dev/nbdN` with `-c`. All options are supported: `-p`, `-b`, `-k` unix socket, `-e` shared clients, `-t` persistent, `-x` export name, `-D` description, `-B` bitmap, `-A` allocation depth, `--tls-creds`, `--tls-authz`, `--tls-hostname`, `--fork`, `--pid-file`, `--handshake-limit` (QEMU 10.0), `--cache`, `--aio`, `--discard`, `--detect-zeroes`, `-r`, `-s` snapshot, `-l` internal snapshot, `-P` partition (MBR only, as QEMU). `-c` uses the kernel netlink or ioctl interface through ruvm-sys. The server is the same NBD server code used by `nbd-server-start` and the storage daemon.

### qemu-storage-daemon

qemu-storage-daemon runs the block layer without a VM: `--blockdev`, `--chardev`, `--export`, `--monitor`, `--nbd-server`, `--object`, `--pidfile`, `--daemonize`, with a QMP monitor exposing the block, job, export, object and bitmap commands only. It shares all code with the system emulator; the difference is which QAPI modules are compiled into the dispatcher, the same mechanism QEMU uses with `storage-daemon/qapi/qapi-schema.json`. Its `query-qmp-schema` must match QEMU's storage daemon schema, not the full system schema.

Since QEMU 10.0 users can control node activation explicitly (`blockdev-set-active`), which makes live migration with a storage daemon backend safe: the source daemon inactivates, the destination activates. ruvm implements the same, and document 17 uses it.

### Exports

Exports are created with `block-export-add` and `--export`, and each export type implements a small trait over a `BlockBackend`. Types are exactly the `BlockExportType` enum: `nbd`, `vhost-user-blk`, `fuse`, `vduse-blk`. The `iothread` option accepts a single iothread or, for exports that support it, a `multi` list; `fixed-iothread` must be false with multiple iothreads.

- NBD: meta contexts for allocation, dirty bitmaps and allocation depth, multiple clients, TLS with authz, `handshake-max-seconds`.
- vhost-user-blk: a vhost-user backend server (`block/export/vhost-user-blk-server.c`) with the virtio-blk request handler shared with VDUSE (`virtio-blk-handler.c`). ruvm implements it on the vhost-user code in document 13, supports `num-queues` and multiple iothreads, `logical-block-size`.
- FUSE: exposes an image as a regular file. Since QEMU 11.0 the FUSE export processes requests asynchronously and can use multiple iothreads; ruvm does the same with `/dev/fuse` read directly per iothread (one FUSE channel clone per thread via `FUSE_DEV_IOC_CLONE`), which avoids libfuse. Options `mountpoint`, `growable`, `writable`, `allow-other`, `allow-inode-change`. Linux only; on macOS FUSE exports are unsupported, as in QEMU.
- VDUSE: exposes a virtio-blk device through `/dev/vduse`, which the kernel can bind to virtio-vdpa (host block device) or vhost-vdpa (feed another VM). Options `num-queues`, `queue-size`, `logical-block-size`, `serial`. Each virtqueue is served on its own iothread.

## Management behavior that must match

- `-drive` legacy option parsing (`blockdev.c` `drive_new`), including `if=`, `index=`, `media=`, `snapshot=on` (temporary qcow2 overlay in `$TMPDIR`), `werror` and `rerror`, `copy-on-read`, `detect-zeroes`, and the automatic node names `#blockNNN`.
- `blockdev-add`, `blockdev-del`, `blockdev-reopen` (list form), `blockdev-snapshot`, `blockdev-snapshot-sync`, `blockdev-change-medium`, `eject`, `blockdev-open-tray` and friends, `block_resize`, `block-set-write-threshold`, `x-blockdev-change` for quorum, `transaction` with all action types, `query-block`, `query-blockstats`, `query-named-block-nodes` (with `flat`, and `flat` on `query-block` since 11.0), `query-block-jobs`, `query-jobs`, `x-debug-query-block-graph`.
- Events: `BLOCK_IO_ERROR` with the `reason` field, `BLOCK_WRITE_THRESHOLD`, `DEVICE_TRAY_MOVED`, `BLOCK_IMAGE_CORRUPTED`, `QUORUM_*`, `BLOCK_EXPORT_DELETED`.
- The `backing` string syntax for `json:{...}` filenames and the reconstruction of `json:` filenames for `query-block` (`bdrv_refresh_filename`) must produce the same strings, since libvirt compares them. This is small but fiddly and gets its own golden test file.

## iotests compatibility plan

QEMU's iotests (`tests/qemu-iotests/`) are shell and Python tests with reference `.out` files, run by `./check -qcow2` (and `-raw`, `-nbd`, `-luks`, `-vmdk` and so on). They drive `qemu-img`, `qemu-io`, `qemu-nbd`, the storage daemon and the system emulator through QMP via the `iotests.py` module and the `qemu.machine` Python package. They are the single best specification of the block layer that exists, and ruvm treats them as tests of ruvm, not as something to rewrite.

Plan:

1. Harness. `cargo xtask iotests` checks out the QEMU 11.1.0 tag of `tests/qemu-iotests`, `python/qemu`, and `scripts/`, and runs `check` with environment variables `QEMU_PROG`, `QEMU_IMG_PROG`, `QEMU_IO_PROG`, `QEMU_NBD_PROG`, `QSD_PROG` pointing at ruvm symlinks, the same variables `tests/qemu-iotests/testenv.py` reads. No file in the test tree is modified. Anything we need to skip goes in a separate `ruvm-iotests-expected.toml` with a reason per test, reviewed like code.
2. Formats in CI. Every pull request runs the `quick` group for qcow2, raw, and nbd. Nightly runs `auto` and full groups for qcow2, raw, qed, vmdk, vdi, vhdx, vpc, parallels, luks, nbd, and file protocol variants with `-o compat=0.10`, `-o refcount_bits=1`, `-o extended_l2=on`, `-o data_file=...` and `-c none` / `-c writeback` cache modes.
3. Output parity. Because iotests compare stdout byte for byte after filters, any divergence in messages, JSON key order, QMP event timing, or job progress numbers fails a test. We treat each failure as a bug in ruvm unless the reference output itself depends on something like a glibc error string, in which case the filter in `common.filter` already handles it or we document the skip.
4. Differential mode. The same harness can run with QEMU binaries and ruvm binaries mixed: ruvm `qemu-img` against QEMU's system emulator and the reverse. A nightly job runs `qemu-img create` and `convert` from one, `qemu-img check` and `compare` from the other, for every format and option set, plus random workload images written by fio through a ruvm VM and checked by QEMU.
5. Fuzzing. cargo-fuzz targets cover every format metadata parser (qcow2 tables, vmdk descriptors, vhdx log, vpc footer, dmg plist) and the NBD codecs.
6. blkdebug event parity. We generate a trace of blkdebug events from QEMU for each qcow2 iotest and compare the event sequence from ruvm for the same test. Mismatches mean an error injection test would hit a different metadata state, so they are caught before they show up as confusing iotest diffs.

Exit criteria for M3: 100% of `quick` group tests for qcow2, raw, nbd pass or are listed with an accepted reason; at least 95% of the full qcow2 run passes; the differential format matrix is clean. Document 22 carries the dashboard and document 23 the dates.

## Failure modes and edge cases we design for

- Power loss during qcow2 metadata update. We follow QEMU's ordering: data before L2, L2 before L1, refcount increase before use and decrease after unlink, with flushes between dependent writes when `cache.no-flush` is off. The dirty bit plus lazy refcounts relaxes this and `qemu-img check -r` repairs. We do not invent a journal because the format has none.
- ENOSPC on thin storage. `werror=enospc` pauses the VM (runstate `io-error`), and `BLOCK_IO_ERROR` fires with `nospace: true`. Retry on `cont` reissues the failed requests in order.
- Unaligned devices: 4 KiB logical sector disks with `O_DIRECT` force a `request_alignment` of 4096; probing uses `BLKSSZGET` and a trial read as `raw_probe_alignment` does.
- Backing file format: `qemu-img create` with `-b` requires `-F` as in current QEMU, and opening an old image with no backing format extension falls back to probing with the same warning text.
