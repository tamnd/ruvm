# 05. Memory: ruvm-mem

This document specifies how ruvm models guest physical memory: the MemoryRegion tree, how it is flattened and published to readers, how an access is dispatched, how accelerators and vhost learn about the map, how dirty pages are tracked, where RAM comes from on the host, and how Rust code touches guest memory without undefined behavior. The reference is QEMU 11.1 as implemented in system/memory.c, system/physmem.c, include/system/memory.h, include/system/ramlist.h, include/system/ram_addr.h, include/exec/memattrs.h, backends/hostmem*.c and hw/virtio/virtio-mem.c. Accelerator specifics are in document 06, the softmmu TLB in document 08, VFIO and vIOMMU in document 16, and migration of RAM in document 17.

## Goals and constraints

The guest must not be able to tell ruvm from QEMU by any memory access. That means the same resolution of overlapping regions, the same splitting of wide accesses into narrow device callbacks, the same values returned for unassigned reads, the same MemTxResult errors surfacing as the same guest faults, and the same RAMBlock names and sizes in the migration stream. Management must not be able to tell either: `info mtree`, `info mtree -f`, `info ramblock`, `query-memory-devices`, `query-memdev` and the memory-backend-* QOM properties produce identical output for identical configurations.

Inside those constraints we want three things QEMU does not have. First, memory map updates that cost proportional to what changed, not to the size of the machine. Second, map reads with no locks and no shared cache line writes, so 256 vCPUs doing MMIO do not contend. Third, a type-level guarantee that no Rust reference (`&T`, `&mut T`, `&[u8]`) ever points into guest RAM, because the guest and DMA engines write that memory concurrently and a Rust reference promises the compiler it does not change underneath it.

## Crates and licensing

`ruvm-mem` (L1, MIT OR Apache-2.0) contains the region tree, FlatView renderer, dispatch structure, RCU publication, AddressSpace, listener fan-out, dirty bitmaps, RamBlock bookkeeping, host mapping primitives, `GuestPtr` and the DMA helpers. It does not depend on ruvm-qom. The access-splitting rules are behavior, not code: they are written from the documented contract and verified by differential tests, so the crate stays permissive per the canon's provenance rule.

New decision: a separate `ruvm-hostmem` crate (L1, GPL-2.0-or-later, depends on ruvm-qom and ruvm-mem) implements the QOM types memory-backend-ram, memory-backend-file, memory-backend-memfd, memory-backend-shm, memory-backend-epc and thread-context, with QEMU's exact property names and error messages. Keeping QOM out of ruvm-mem lets rust-vmm style consumers use the core without the object model. MemoryRegion objects still appear in the QOM tree (QEMU parents them to their owner device), through an adapter in ruvm-hw-core that registers each region's name under its owner.

## Vocabulary

Guest physical address (GPA) is an address in an AddressSpace, 64 bits wide, like QEMU's `hwaddr`. Host virtual address (HVA) is a pointer in our process. A RamBlock is one contiguous host allocation backing one RAM-like region: a stable name (`idstr`) used by migration, `used_length` and `max_length` (resizable blocks such as the ACPI tables blob grow up to `max_length`), a host pointer, an optional fd, and flags. QEMU also places every RamBlock in one global `ram_addr_t` space and indexes dirty bitmaps by it. ruvm indexes everything per RamBlock instead, which removes the global RAM list lock from paths that touch one block; `ram_addr_t` order survives only where migration needs block order (document 17).

## MemoryRegion kinds

A MemoryRegion is a node in a tree. It has a size (up to 2^64, represented as `u128` internally so that a full 64-bit region is expressible, which QEMU does with `Int128`), a name, an owner, an enabled flag, a priority within its container, and a kind.

| Kind | QEMU constructor | Reads | Writes | Notes |
|---|---|---|---|---|
| RAM | memory_region_init_ram*, memory_region_init_ram_from_fd/file | direct host memory | direct host memory, dirty tracked | backed by a RamBlock; may be marked readonly, nonvolatile, or protected (guest_memfd) |
| ROM | memory_region_init_rom | direct host memory | ignored (discarded, MEMTX_OK) | a RAM region with readonly set; debug writes (MemTxAttrs.debug, gdbstub, firmware loaders) go through |
| ROM device | memory_region_init_rom_device | direct host memory while in romd mode, callback otherwise | always callback | flash devices (pflash_cfi01/02) toggle romd mode when entering command mode |
| MMIO | memory_region_init_io | callback | callback | MemoryRegionOps with valid and impl constraints, endianness |
| RAM device | memory_region_init_ram_device_ptr | host pointer, but accessed with exact access size | same | for VFIO BARs mmapped from the device; never touched with memcpy because PCIe BARs can have side effects per access width |
| Alias | memory_region_init_alias | through target | through target | window of another region at an offset; cannot be circular |
| Container | memory_region_init | holes fall through | holes fall through | groups subregions; a container can also be a RAM or MMIO region with subregions layered on top |
| IOMMU | memory_region_init_iommu | translated, then dispatched in target AS | same | translate() callback per access, notifiers for map/unmap, one or more iommu indexes chosen from MemTxAttrs |
| Reservation | memory_region_init_io with NULL ops | as unassigned | as unassigned | claims address space handled outside the emulator (for example by KVM in-kernel devices); KVM does not register a slot for it, so accesses exit |

Each kind maps to one Rust enum variant. The tree is stored in an arena owned by the memory core, and regions are referred to by `RegionId` (a generation-checked index), not by `Arc`. QEMU's lifetime rule ("You must not destroy a memory region as long as it may be in use by a device or CPU", docs/devel/memory.rst) is enforced by construction: a RegionId held by a published FlatView keeps the region's callbacks alive until the RCU grace period in which that FlatView is retired.

```rust
pub enum RegionKind {
    Ram { block: RamBlockId, readonly: bool, nonvolatile: bool },
    RomDevice { block: RamBlockId, ops: Arc<dyn MmioOps>, romd: AtomicBool },
    Mmio { ops: Arc<dyn MmioOps> },
    RamDevice { host: HostMapping },
    Alias { target: RegionId, offset: u64 },
    Container,
    Iommu { ops: Arc<dyn IommuOps>, target_as: AsId },
    Reservation,
}

pub struct MemoryRegion {
    name: CompactString,
    owner: OwnerRef,              // opaque handle, resolved to a QOM path by ruvm-hw-core
    size: u128,
    kind: RegionKind,
    parent: Option<RegionId>,
    addr_in_parent: u64,
    priority: i32,
    enabled: bool,
    children: SmallVec<[RegionId; 4]>, // kept in QEMU order, see below
    ioeventfds: Vec<IoEventFd>,
    coalesced: Vec<AddrRange>,
    flush_coalesced_mmio: bool,
    dirty_log_mask: u8,           // bit per DirtyClient
    lock_domain: Option<LockDomainRef>, // None means lock-free MMIO fast path
    discard_manager: Option<Arc<dyn RamDiscardManager>>,
    generation: u64,              // bumped on any change to this node
}
```

`MmioOps` is the canonical trait from the canon (read and write with `AccessCtx`, and `valid() -> AccessConstraints`). We add an associated `impl_constraints()` with a default that matches QEMU's defaults (min 1, max 4, aligned), because the valid and impl sets are different things and devices ported from QEMU set them independently.

## Priority and overlap resolution

Rendering follows `render_memory_region()` in system/memory.c exactly, because the result is guest visible. A container's children are kept in a list sorted by descending priority. QEMU's `memory_region_update_container_subregions()` inserts a new child before the first existing sibling whose priority is less than or equal to the new child's (`subregion->priority >= other->priority`), so among equal priorities the most recently added child comes first. The renderer walks children in list order and a range already claimed in the output obscures later children. The consequence, which ruvm reproduces, is that on an equal-priority overlap the most recently mapped region wins. Guests hit this: firmware that programs two BARs to the same address during enumeration sees the second one.

Priorities are local to a container: they are compared only between siblings. A high priority container that leaves holes lets lower priority siblings show through those holes. A disabled region renders as nothing. Readonly and nonvolatile are inherited downward during rendering (`readonly |= mr->readonly`). An alias is rendered by rendering its target with the base shifted by `alias_offset` and clipped to the alias's size; an alias of an alias is legal, an alias cycle is rejected at `add_subregion` time with the same error QEMU raises.

The output of rendering is a list of FlatRanges: non-overlapping, sorted, each pointing at a region, an offset into that region, and the inherited attributes. `flatview_simplify()` then merges adjacent FlatRanges that point at contiguous offsets of the same region with the same attributes, unless the region is marked unmergeable (QEMU uses that for RAM that must stay in separate KVM slots, for example virtio-mem memslots). ruvm does the same merge, because listeners see the merged ranges, and KVM slot layout is visible through migration of dirty logs and through `info mtree -f`.

## FlatView generation and incremental update

QEMU regenerates every FlatView on every committed transaction. `memory_region_transaction_commit()` calls `flatviews_reset()`, which drops the cache of views keyed by root region, and then `address_space_set_flatview()` for each AddressSpace re-renders its root from scratch with `generate_memory_topology()`, rebuilds the dispatch radix tree, and diffs old against new to call listeners. Views are shared between address spaces with the same root through the `flat_views` hash table.

To size this, we ran QEMU 11.1.2 (Homebrew build, TCG) with `-M q35 -m 4G -smp 4` plus virtio-net, virtio-blk, virtio-rng, qemu-xhci, ich9-intel-hda and VGA, let SeaBIOS enumerate PCI, and dumped `info mtree -f`. There were 10 distinct FlatViews: system memory with 59 ranges (shared by `memory`, the four `cpu-memory-N` spaces and the three bus-mastering PCI devices), I/O ports with 85 ranges, one SMM view per vCPU with 58 ranges each (x86 TCG gives every CPU its own root), three small virtio-pci config views, and one empty view shared by seven devices whose bus master enable was still off. One BAR write re-renders about 400 ranges and rebuilds 10 dispatch trees. PCI enumeration by firmware and Linux does this hundreds of times per boot, and with 64 or more vCPUs under TCG the per-CPU SMM views dominate.

ruvm makes three changes, all invisible to the guest.

First, dirty-subtree tracking. Every change bumps the `generation` of the changed node and marks its ancestors up to each root. On commit, only address spaces whose root was marked are re-rendered. A BAR move on a device behind a root port marks the PCI memory container and the system root, not the I/O port root or the virtio config views.

Second, memoized subtree rendering. For each container we cache the FlatRange list it rendered last time, keyed by (region generation, clip, inherited attributes). When re-rendering a root, unchanged subtrees splice in their cached output. The cost of a commit becomes proportional to the number of ranges under changed containers plus a linear merge.

Third, content hashing of views. Two roots that render to the same range list share one FlatView (the list is hashed on build). All per-CPU SMM roots on q35 render identically while no CPU is in SMM-specific configuration, so N vCPUs cost one SMM view, not N. When a CPU's view diverges (SMRAM open on one CPU only) it gets its own view.

The listener diff is computed between the old and new FlatView of each address space, exactly like `address_space_update_topology_pass()`: first a pass issuing `region_del` for removed ranges and `region_nop` for unchanged ones, then a pass issuing `region_add`, with `log_start` and `log_stop` when only the dirty log mask changed. Order matters for KVM (a slot must be deleted before an overlapping slot is added) and we keep QEMU's two-pass order.

A full rebuild path remains, and a debug option (`-global ruvm-mem.x-verify-incremental=on`) renders from scratch after every commit and asserts equality with the incremental result. The differential test suite runs with it on.

Transactions are explicit. `MemoryTransaction` is a guard; nested guards increment a depth counter; the commit happens when the outermost guard drops. Transactions are taken under the control lock (canon: topology changes only), never on a vCPU hot path. Changes to ioeventfds without topology changes only rerun the ioeventfd diff, like QEMU's `ioeventfd_update_pending` path.

## RCU publication and reader cost

Each AddressSpace holds an `RcuCell<FlatView>` from ruvm-base. Publishing a new view is a single release store of a pointer. Readers enter an epoch, load the pointer with acquire ordering, use the view, and leave. The old view is freed by the reclaimer thread after every thread that could have seen it has passed through a quiescent state.

QEMU's util/rcu.c uses a global grace period counter and per-thread counters; `rcu_read_lock()` copies the global counter into the thread's slot, and the writer uses `smp_mb_global()` (membarrier where available) so readers need no full fence. ruvm-base uses the same asymmetric design: `membarrier(MEMBARRIER_CMD_PRIVATE_EXPEDITED)` on Linux, `FlushProcessWriteBuffers()` on Windows, and a reader-side full fence on macOS and the BSDs, which lack an equivalent. On Linux and Windows a read-side section costs one relaxed load of the global epoch and one store to a thread-local slot, with no reference count on the FlatView and no shared cache line written by readers.

Reader sections are coarse where it helps. A TCG vCPU thread holds its epoch across a run of translated blocks and passes a quiescent state when it exits to the outer loop, as QEMU's cpu_exec does with `rcu_read_lock()`. A KVM or HVF vCPU enters the epoch only while it handles an MMIO or PIO exit. An iothread running virtio queues enters the epoch per request batch. Nothing may block on I/O inside a reader section; the debug build enforces this with a per-thread flag checked by ruvm-aio's blocking wrappers.

A FlatView pins everything it references: the MemoryRegion callbacks (through `Arc<dyn MmioOps>`), RamBlocks, and IOMMU regions. Unplugging a device removes its regions from the tree, commits, and the old FlatView, retired after the grace period, drops the last references. Device unrealize waits for that grace period before tearing down backing state, which is QEMU's behavior with `object_unparent` plus RCU.

## Dispatch data structure

Dispatch answers: for address A in this FlatView, which section, and at what offset into its region. QEMU answers it with `AddressSpaceDispatch` in system/physmem.c: a multi-level page table (PhysPageMap) with `P_L2_BITS = 9`, so 512 entries per node, over `ADDR_SPACE_BITS = 64` with `TARGET_PAGE_BITS` for the leaf, giving `P_L2_LEVELS = ((64 - 12 - 1) / 9) + 1 = 6` levels for 4 KiB target pages. Each `PhysPageEntry` is 32 bits: a 6-bit skip and a 26-bit pointer. Large aligned ranges are stored as leaves at upper levels, `phys_page_compact()` collapses single-child chains using the skip field, and there is a one-entry `mru_section` cache per dispatch. Regions smaller than a target page, or not page aligned, go through a `subpage_t` that is itself an MMIO region containing a `uint16_t sub_section[]` array with one entry per byte of the page, 8 KiB per partially covered page.

We considered four structures.

The QEMU radix tree. We built the q35 system memory view from the dump above (59 sections, 81 entries once holes are filled with the unassigned section) with a prototype of QEMU's algorithm including upper-level leaves. After the equivalent of compaction it needs 15 nodes of 2 KiB plus 9 subpages of 8 KiB, 102 KiB in total, and a lookup is up to 5 dependent loads in different cache lines. It must be rebuilt wholesale, because compaction changes its shape.

A sorted boundary array with binary search. The same 81 boundaries take 648 bytes of keys plus 324 bytes of section ids. A lookup is 7 comparisons within 11 cache lines, the first probes always hit the same lines, and sub-page ranges (a 16-byte xHCI port block) are just more boundaries, so there is no subpage machinery. Rebuild is a linear copy of the FlatRange list.

The same array in Eytzinger (BFS) order with branchless descent: same footprint, no mispredicted branches, top three levels in one cache line.

An interval B-tree: suited to tens of thousands of sections with point updates, which FlatViews never have, since they are replaced whole through RCU.

We did not publish timings from the prototype because the development host (Apple M4) was under a load average above 60 when we ran it, which made nanosecond numbers meaningless; document 21's suite measures this on quiet hosts.

Decision: ruvm uses a sorted boundary array per FlatView with branchless binary search, in Eytzinger order above 64 boundaries and plain order below, fronted by a two-entry per-thread MRU cache (last RAM hit, last MMIO hit). Reasons, by weight: no subpage special case, which removes a class of QEMU sub-page bugs and an MMIO indirection layer; about 100 times smaller for a typical machine, which matters with 10 or more views; linear-time rebuild, so incremental FlatView updates stay cheap end to end; and lookup is not the hot path anyway. Under KVM, HVF and WHPX dispatch runs only on MMIO and PIO exits, whose fixed cost is thousands of cycles. Under the JIT the softmmu TLB (document 08) caches the result per page, using the section index as QEMU's `iotlb` does. Virtio DMA almost always hits the MRU RAM entry.

```rust
pub struct Dispatch {
    keys: Box<[u64]>,          // boundary start addresses, sorted or Eytzinger
    sections: Box<[SectionIdx]>,
    eytzinger: bool,
}

pub struct Section {
    region: RegionId,
    offset_in_region: u64,     // offset of the section start within the region
    size: u128,
    readonly: bool,
    nonvolatile: bool,
    romd: bool,
    host: Option<NonNull<u8>>, // cached HVA base for RAM, ROM, romd ROM devices
}
```

Section index 0 is always the unassigned section, as QEMU's `PHYS_SECTION_UNASSIGNED` is. Unassigned reads return the value QEMU's `unassigned_mem_read()` returns (0) and the result code is MEMTX_DECODE_ERROR; whether that becomes a guest fault is decided by the target (for example Arm external aborts, x86 ignores it), in document 09.

## AddressSpaces: per CPU and per device

An AddressSpace is a root region plus a name plus its current FlatView. ruvm creates the same address spaces QEMU does, with the same names, because `info mtree` lists them. The global ones are `memory` (the system address space) and `I/O` (x86 port space). Each CPU gets one or more CPU address spaces indexed by an address space index chosen from MemTxAttrs: on x86 index 0 is `cpu-memory-N` and index 1 is `cpu-smm-N` (the SMM view with SMRAM overlaid); on Arm with EL3 or RME the secure, root and realm spaces are separate indexes. The target's `attrs_to_asidx` hook (document 09) picks the index per access, like QEMU's `cpu_asidx_from_attrs()`.

Each PCI device has a bus master address space rooted in a per-device container holding an alias of the bus's DMA address space, enabled only while the Bus Master Enable bit is set in the command register. That is why seven devices shared an empty FlatView in our dump. Behind a vIOMMU, the bus's DMA address space root is an IOMMU region, and translation happens per access (document 16). Platform devices (SysBus) that do DMA use `memory` or a per-SoC interconnect address space passed in as a link property.

Listeners subscribe per address space. KVM on x86 subscribes to `memory` and, when SMM is enabled, to a separate `kvm-smram` address space registered as KVM address space 1 (`X86ASIdx_SMM` in target/i386/kvm/kvm.c); vhost subscribes to the device's DMA address space or to `memory` when there is no vIOMMU.

## MemTxAttrs

The attributes follow include/exec/memattrs.h bit for bit because devices branch on them and because the plugin API exposes some of them. In QEMU 11.1 the struct holds `secure:1`, `space:2` (ArmSecuritySpace), `user:1`, `memory:1` (restrict to normal memory, AMBA), `debug:1`, `requester_id:16`, `pid:8` (PCI PASID), `address_type:1` (PCI address type for IOMMU), and a separate `unspecified` bool, with a build check that the struct is at most 8 bytes. `MEMTXATTRS_UNSPECIFIED` sets only `unspecified`.

```rust
#[derive(Copy, Clone, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct MemTxAttrs(u64); // same bit positions as QEMU, accessors per field

pub const MEMTX_OK: MemTxResult = MemTxResult(0);
pub const MEMTX_ERROR: MemTxResult = MemTxResult(1 << 0);
pub const MEMTX_DECODE_ERROR: MemTxResult = MemTxResult(1 << 1);
pub const MEMTX_ACCESS_ERROR: MemTxResult = MemTxResult(1 << 2);

pub struct AccessCtx {
    pub attrs: MemTxAttrs,
    pub requester: RequesterRef, // vCPU index or device, for tracing and reentrancy
}
```

Results are bit sets, OR-ed across the pieces of a split access, exactly as `access_with_adjusted_size()` does with `r |= access_fn(...)`.

## Access size, alignment and endianness

This is where compatibility is easiest to lose. MemoryRegionOps carry two constraint sets. `valid` describes what the device accepts: if an access is outside it, the access is rejected. `impl` describes what the callback implements: other sizes are emulated by the core by splitting or widening.

The exact QEMU rules, which ruvm implements and tests one by one:

1. Validation in `memory_region_access_valid()`: if `valid.accepts` exists and returns false, reject. If `valid.unaligned` is false and `addr & (size - 1)` is nonzero, reject. If `valid.max_access_size` is zero, accept everything else ("compatibility all valid"). Otherwise reject sizes above `valid.max_access_size` or below `valid.min_access_size`. A rejected read returns the unassigned value with MEMTX_DECODE_ERROR; a rejected write is dropped with MEMTX_DECODE_ERROR. Each rejection logs under the `guest_errors` log mask with the same message text (`-d guest_errors` output is compared in tests).
2. Before dispatch, physmem's `memory_access_size()` clamps a multi-byte access from a bus master or a CPU slow path: the max defaults to 4 when `valid.max_access_size` is zero, it is further bounded by the address alignment (`addr & -addr`) unless `impl.unaligned` is set, and then rounded down to a power of two. So an 8-byte `address_space_read` at an address that is 4 mod 8 becomes two 4-byte accesses, even if the device would accept 8.
3. Splitting in `access_with_adjusted_size()`: `impl.min_access_size` defaults to 1 and `impl.max_access_size` to 4. The per-call size is `max(min(size, impl_max), impl_min)`. If that exceeds the request, the device is called once at the implemented width at the same address and the result is shifted and masked. If smaller, the core loops over `addr + i` and assembles the value with a shift of `i * 8` (little-endian) or `(size - access_size - i) * 8` (big-endian). QEMU does not align the widened access down: a 1-byte read at offset 3 of a device with `impl.min_access_size = 4` becomes a 4-byte read at offset 3 (the source carries a `FIXME: support unaligned access?`). ruvm reproduces the exact callback addresses, because ported device models encode assumptions about them.
4. Endianness: the ops declare DEVICE_LITTLE_ENDIAN, DEVICE_BIG_ENDIAN or DEVICE_NATIVE_ENDIAN (target endianness, and for bi-endian targets the build's default). `adjust_endianness()` byte-swaps the value when the access's MemOp endianness differs from the device's. We represent this as an `Endian` enum resolved at region creation, so there is no per-access lookup of target endianness.
5. Reentrancy: since QEMU 8.0 a device's MMIO callbacks may not be re-entered through its own DMA (`mem_reentrancy_guard`, which blocks the nested access, returns MEMTX_ACCESS_ERROR and warns once). This closed a family of fuzzing bugs. ruvm keeps the same guard per device, with the same opt-out (`disable_reentrancy_guard`, used by a few devices that need it), and applies it to MMIO regions that are not RAM, RAM device, ROM device or readonly, exactly as QEMU does.

These rules exist in one module, `ruvm_mem::access`, with a table-driven test that enumerates every combination of valid and impl min, max and unaligned, access size 1 to 8 and address offset 0 to 7, and compares callback traces with a recording device run under QEMU (document 22 describes the differential harness). Accesses wider than 8 bytes are split by the target front end before they reach the memory core, as in QEMU.

## Locking on the dispatch path

QEMU takes the BQL in `prepare_mmio_access()` for every MMIO access unless the region is marked `lockless_io`. ruvm replaces the BQL with the region's lock domain (canon: device domains): dispatch acquires the domain lock, flushes coalesced MMIO if the region has `flush_coalesced_mmio`, calls the device, and releases. Regions with no domain are lock-free and synchronize themselves (virtio notify registers, MSI-X PBA, the HPET counter). Cross-domain effects such as raising an interrupt go through the GPIO and IRQ primitives in document 12, which are callable from any domain without lock inversion.

## Listeners and how accelerators and vhost subscribe

A MemoryListener receives the diff of an address space's FlatView. QEMU's struct has begin, commit, region_add, region_del, region_nop, log_start, log_stop, log_sync, log_sync_global, log_clear, log_global_start, log_global_stop, log_global_after_sync, eventfd_add, eventfd_del, coalesced_io_add, coalesced_io_del, and a priority (MEMORY_LISTENER_PRIORITY_MIN 0, ACCEL 10, DEV_BACKEND 10). Forward order on add, reverse on delete, so accelerators map memory before device backends reference it, and unmap after.

```rust
pub trait MemoryListener: Send + Sync {
    fn priority(&self) -> i32 { 0 }
    fn name(&self) -> &str;
    fn begin(&self) {}
    fn commit(&self) {}
    fn region_add(&self, s: &SectionView) {}
    fn region_del(&self, s: &SectionView) {}
    fn region_nop(&self, s: &SectionView) {}
    fn log_start(&self, s: &SectionView, old: DirtyMask, new: DirtyMask) {}
    fn log_stop(&self, s: &SectionView, old: DirtyMask, new: DirtyMask) {}
    fn log_sync(&self, s: &SectionView) {}
    fn log_sync_global(&self, last_stage: bool) {}
    fn log_clear(&self, s: &SectionView) {}
    fn log_global_start(&self) -> Result<(), Error> { Ok(()) }
    fn log_global_stop(&self) {}
    fn log_global_after_sync(&self) {}
    fn eventfd_add(&self, s: &SectionView, e: &IoEventFd) {}
    fn eventfd_del(&self, s: &SectionView, e: &IoEventFd) {}
    fn coalesced_io_add(&self, s: &SectionView, range: AddrRange) {}
    fn coalesced_io_del(&self, s: &SectionView, range: AddrRange) {}
}
```

Registering a listener replays the current FlatView as `region_add` calls (QEMU's `listener_add_address_space()`), so late subscribers such as a hotplugged vhost device see the full map.

Subscribers and what they do:

- KVM (document 06) turns RAM sections into memslots via KVM_SET_USER_MEMORY_REGION2 when guest_memfd is supported (QEMU checks KVM_CAP_GUEST_MEMFD, KVM_CAP_USER_MEMORY2 and the private memory attribute), else KVM_SET_USER_MEMORY_REGION. It splits sections larger than the kernel's maximum slot size, marks readonly slots with KVM_MEM_READONLY so ROM writes exit, sets KVM_MEM_LOG_DIRTY_PAGES on log_start, registers ioeventfds with KVM_IOEVENTFD, and coalesced ranges with KVM_REGISTER_COALESCED_MMIO. Slot count is bounded by KVM_CAP_NR_MEMSLOTS.
- HVF, WHPX, NVMM and MSHV map RAM sections into the partition with their own map calls and handle dirty logging by write-protecting (document 06).
- The JIT (document 08) flushes affected softmmu TLB entries on commit (QEMU's `tcg_commit()`), and on `log_global_after_sync` makes sure no vCPU still holds a TLB entry that bypasses dirty tracking.
- vhost and vhost-user (document 13) build their memory table from RAM sections (vhost-user needs an fd per region, so RAM without an fd is refused with QEMU's error), and relay dirty logging through the vhost log.
- VFIO (document 16) maps RAM sections into the IOMMU for DMA, registers IOMMU notifiers for vIOMMU regions, and uses RamDiscardManager for virtio-mem.
- Xen maps nothing (the hypervisor owns guest memory) but tracks sections for its mapcache.

A listener callback runs in the committing thread under the control lock. Callbacks must not start new transactions. QEMU asserts on that and so do we.

## ioeventfd and coalesced MMIO

`memory_region_add_eventfd()` attaches (address, size, match data, fd) to a region. On commit the core computes the sorted ioeventfd list per address space (comparator as in `memory_region_ioeventfd_before()`), diffs it, and calls eventfd_add and eventfd_del. When the accelerator cannot handle ioeventfds (TCG), dispatch signals the fd itself on write, as `memory_region_dispatch_write_eventfds()` does. Coalesced MMIO ranges work likewise.

## Dirty memory tracking

QEMU tracks dirtiness per client: DIRTY_MEMORY_VGA (0), DIRTY_MEMORY_CODE (1) and DIRTY_MEMORY_MIGRATION (2), `DIRTY_MEMORY_NUM` 3 (include/system/ram_addr.h). Bitmaps are indexed by `ram_addr_t` page number and stored in `DirtyMemoryBlocks` of `DIRTY_MEMORY_BLOCK_SIZE = 256 * 1024 * 8` bits, so each block covers 8 GiB of 4 KiB pages; the array of blocks is RCU-published so RAM hotplug can grow it. Migration additionally keeps a per-RAMBlock `bmap` and a `clear_bmap` for lazy KVM_CLEAR_DIRTY_LOG.

ruvm stores dirty state per RamBlock, one atomic bitmap per enabled client, allocated only while that client's logging is enabled for a region covering the block. A 4 GiB guest with migration running needs 128 KiB of bitmap; a guest not migrating, without VGA and under KVM needs none. Bits are set with `fetch_or` on 64-bit words (relaxed ordering, with a release fence published by the sync operation), cleared by the consumer with `swap(0)` on whole words, and read in bulk.

The three clients and their producers:

- MIGRATION: produced by the accelerator's dirty log (KVM bitmap or ring, HVF and WHPX write-protection faults), by the JIT's notdirty slow path for pages whose TLB entry is marked not-dirty, and by every DMA write through `address_space_write`, `dma_memory_write` or an unmap of a writable mapping. Consumed by ruvm-migration (document 17).
- VGA: produced the same way for regions whose owner called `memory_region_set_log(mr, true, DIRTY_MEMORY_VGA)`. Consumed by display devices via `snapshot_and_clear_dirty()`, which returns an immutable snapshot of the bits for a range, like QEMU's `memory_region_snapshot_and_clear_dirty()`.
- CODE: produced only by the JIT. A page containing translated code has its CODE bit clear; any write to it takes the slow path, invalidates translations for that page, and sets the bit. This is how self-modifying code is detected (document 08).

For KVM, ruvm supports both the bitmap log and the dirty ring. With the bitmap log it uses KVM_CAP_MANUAL_DIRTY_LOG_PROTECT2 and re-arms protection lazily with KVM_CLEAR_DIRTY_LOG in 64-page aligned chunks, matching kvm-all.c. With the dirty ring (`-accel kvm,dirty-ring-size=N`), each vCPU has an mmapped array of `struct kvm_dirty_gfn { u32 flags; u32 slot; u64 offset; }` that a reaper harvests into the MIGRATION bitmap before calling KVM_RESET_DIRTY_RINGS. On weakly ordered hosts only KVM_CAP_DIRTY_LOG_RING_ACQ_REL is valid. With KVM_CAP_DIRTY_LOG_RING_WITH_BITMAP (needed on arm64, where vGIC/ITS table saves dirty memory without a vCPU), the final sync also reads the backup bitmap. Document 06 covers ring sizing and the ring-full exit.

Bitmaps are always at target page granularity regardless of backing page size: 4 KiB on every target ruvm supports under an accelerator, and the target page size under the JIT (8 KiB on Alpha, for example). KVM logs 1 GiB-backed guests at 4 KiB too.

## RAM backends

Guest RAM is created by memory-backend objects (`-object memory-backend-*,id=...` with `-machine memory-backend=` or `-numa node,memdev=`) or implicitly by `-m`, which creates a memory-backend-ram with the machine's default RAM id (`pc.ram`, `mach-virt.ram` and so on). RamBlock names must match QEMU's exactly because the migration stream identifies RAM by name: the implicit RAM uses the machine's `default_ram_id`, backends use their object id, device RAM is `<qdev path>/<region name>` such as `0000:00:02.0/vga.vram`, and fw_cfg blobs use `/rom@etc/acpi/tables` style names.

| Backend | Host mechanism | Key properties |
|---|---|---|
| memory-backend-ram | anonymous mmap, MAP_PRIVATE unless share=on | size, share, merge, dump, prealloc, prealloc-threads, prealloc-context, host-nodes, policy, reserve |
| memory-backend-memfd | memfd_create, always shareable | hugetlb, hugetlbsize, seal (default on) |
| memory-backend-file | open plus mmap of mem-path, hugetlbfs if the path is on it | mem-path, align, offset, pmem, readonly, rom, discard-data |
| memory-backend-shm | POSIX shm_open plus shm_unlink, shareable, portable to macOS and BSD | common properties |
| memory-backend-epc | SGX EPC via /dev/sgx_vepc | common properties |

Common properties come from backends/hostmem.c and are reproduced with QEMU's names, types and defaults: `merge` (MADV_MERGEABLE), `dump` (MADV_DONTDUMP when off), `prealloc`, `prealloc-threads`, `prealloc-context` (a thread-context object that pins the preallocation threads to CPUs near the target NUMA node), `host-nodes` and `policy` (default, preferred, bind, interleave, applied with mbind before touching pages), `share`, and `reserve` (MAP_NORESERVE when off).

On macOS there is no memfd, no mbind and no hugetlbfs. Anonymous mmap and shm_open are available. HVF maps RAM with hv_vm_map, which requires page-aligned HVAs; 16 KiB pages on Apple silicon mean RAM block sizes and alignments are rounded to 16 KiB on that host, which QEMU also does via `qemu_real_host_page_size()`. On Windows, RAM is VirtualAlloc'd, and shared RAM uses CreateFileMapping.

Preallocation. QEMU's `qemu_prealloc_mem()` in util/oslib-posix.c already uses `MADV_POPULATE_WRITE`, splits the range evenly across `prealloc-threads` threads (default: the vCPU count, set in `host_memory_backend_init()`), can run asynchronously during device creation, and falls back to touching pages under a SIGBUS handler. ruvm keeps the default and the fallback but cuts the range into chunks aligned to the backing page size and has workers bound to the thread-context's CPU set pull them from a shared queue, so a slow NUMA node does not leave other threads idle at the end.

guest_memfd. For confidential guests (SEV-SNP, TDX, later Arm CCA and pKVM), private memory is not mappable by the VMM. When the confidential-guest-support object requires it, every RAM block gets QEMU's RAM_GUEST_MEMFD flag: ruvm calls KVM_CREATE_GUEST_MEMFD, keeps the ordinary mapping for the shared view, and registers slots with KVM_SET_USER_MEMORY_REGION2 carrying both `userspace_addr` and `guest_memfd`/`guest_memfd_offset`. Conversions use KVM_SET_MEMORY_ATTRIBUTES, driven by KVM_EXIT_MEMORY_FAULT (document 06) and guest hypercalls, and discard the page on the side being left (fallocate PUNCH_HOLE for the guest_memfd). Since Linux 6.18, GUEST_MEMFD_FLAG_MMAP and GUEST_MEMFD_FLAG_INIT_SHARED let the VMM map guest_memfd directly; ruvm supports that mode behind the capability check. In-place private/shared conversion (v13 on the kernel lists in September 2026, unmerged) is future work in document 25. For Rust code the rule is that a private range has no valid HVA: `GuestMemory` returns `MemTxResult::ACCESS_ERROR` for it. Document 19 has the confidential computing details.

## RAM discard, virtio-mem, balloon and hotplug

A RamDiscardManager is attached to a RAM region whose parts may be unplugged while the region stays mapped. virtio-mem is the main user. Consumers that pin or map memory (VFIO, vhost-user with postcopy, confidential guests) must not map discarded parts, so they register a RamDiscardListener and get populate and discard notifications per range, plus a replay of currently populated ranges at registration. ruvm's trait mirrors QEMU's RamDiscardManagerClass: `min_granularity`, `is_populated`, `replay_populated`, `replay_discarded`, `register_listener`, `unregister_listener`.

virtio-mem (hw/virtio/virtio-mem.c) exposes a device-managed memory area with `requested-size` (set via qom-set), `size` (plugged) and `block-size` (minimum 1 MiB, `VIRTIO_MEM_MIN_BLOCK_SIZE`; default the host THP size). Unplugging discards the backing, and guest access to unplugged blocks is refused (`unplugged-inaccessible`). With `dynamic-memslots=on` the region is split into multiple memslots mapped on demand; ruvm uses the same slot count heuristics because slot layout shows in `info mtree` and in vhost-user memory tables. Migration carries the plugged bitmap using the `early-migration` state (document 17).

virtio-balloon (hw/virtio/virtio-balloon.c) is implemented in ruvm-hw-virtio (document 13); the memory side is only: inflate calls `ram_block_discard_range()` on each reported page unless discarding is disabled (it is disabled when VFIO or confidential computing requires pinned memory, QEMU's `ram_block_discard_disable()` counter, which ruvm keeps with the same semantics); free page reporting discards reported ranges; free page hinting clears MIGRATION bits for hinted pages during the bulk stage; deflate is a no-op on the host because the next guest touch faults the page back in.

Memory hotplug covers pc-dimm, nvdimm, virtio-pmem and virtio-mem through the memory device interface (hw/mem/memory-device.c): a device-memory region reserved by the machine (sized by `-m maxmem=` and `slots=`), address assignment that reproduces QEMU's first-fit algorithm with the same alignment rules (so the same command line gives the same guest physical addresses), and a per-address-space memslot budget shared with the accelerator's KVM_CAP_NR_MEMSLOTS limit and vhost's limit (`vhost_get_free_memslots`). ACPI and device tree exposure of hotplugged memory is in document 11.

## DMA helpers and bounce buffers

Devices never compute HVAs themselves. They use the address space API:

```rust
impl AddressSpace {
    pub fn read(&self, addr: u64, attrs: MemTxAttrs, buf: &mut [u8]) -> MemTxResult;
    pub fn write(&self, addr: u64, attrs: MemTxAttrs, buf: &[u8]) -> MemTxResult;
    pub fn ld<T: GuestScalar>(&self, addr: u64, attrs: MemTxAttrs, e: Endian) -> (T, MemTxResult);
    pub fn st<T: GuestScalar>(&self, addr: u64, v: T, attrs: MemTxAttrs, e: Endian) -> MemTxResult;
    pub fn map(&self, addr: u64, len: u64, dir: DmaDir, attrs: MemTxAttrs) -> Result<DmaMapping<'_>, MapError>;
    pub fn access_valid(&self, addr: u64, len: u64, dir: DmaDir, attrs: MemTxAttrs) -> bool;
    pub fn cache(&self, addr: u64, len: u64, writable: bool) -> Result<MemoryRegionCache, MemTxResult>;
}
```

`read` and `write` walk the FlatView section by section, doing a bulk copy for RAM (with dirty marking on write), and splitting into MMIO accesses for everything else with the `memory_access_size()` rule from above. This is `flatview_read_continue()` and `flatview_write_continue()`.

`map` returns a guard exposing a `GuestSlice`. If the range is RAM in one section, it is a direct mapping. Otherwise QEMU bounces: it allocates a temporary buffer, fills it for reads, and writes it back with `address_space_write()` on unmap. Bounce space per address space is bounded by `max_bounce_buffer_size` (`DEFAULT_MAX_BOUNCE_BUFFER_SIZE`, 4096 bytes, per PCI device via `x-max-bounce-buffer-size`); a request that cannot get space returns a short or null mapping and the caller registers `address_space_register_map_client()` for a retry. ruvm reproduces the limit, short mappings and retry, since virtio and IDE depend on them. The guard's Drop unmaps: dirty marking for the explicitly set written length (as `address_space_unmap(..., access_len)`), bounce write-back, and IOMMU notifications.

`MemoryRegionCache` is QEMU's cache for virtqueue rings: a pre-translated window that stays valid until the next topology change. ruvm implements it as a FlatView generation plus a resolved section; on use, if the address space's current FlatView generation differs, the cache re-resolves. Under RCU the check is one load and one compare.

DMA from devices sits behind the device's bus master address space, so an IOMMU region on the path causes translation. The translation result is cached per device in an IOTLB owned by the IOMMU model (document 16), and invalidations arrive as IOMMU notifier events that also invalidate `MemoryRegionCache` entries covering the range.

## Safety model for guest memory in Rust

Guest RAM is shared with vCPUs (native or JIT), with DMA from vhost, vfio or hardware, and with ruvm's own threads. Rust assumes bytes behind `&[u8]` do not change and bytes behind `&mut [u8]` are not read by others; both are false for guest memory, and a data race on non-atomic memory is undefined behavior whether or not the value is used. QEMU lives with this by convention; rust-vmm's vm-memory uses `VolatileSlice`. The principled fix, byte-wise atomic memcpy (rust-lang/rfcs#3301, "AtomicPerByte"), has not been accepted as of September 2026.

ruvm's rules, enforced by the type system and by a lint in `cargo xtask provenance`:

1. No Rust reference into guest memory is ever created. There is no API that returns `&[u8]` or `&T` pointing into a RamBlock. The raw host pointer is private to ruvm-mem and ruvm-sys.
2. Guest memory is accessed through `GuestPtr<T>` and `GuestSlice`, which wrap a raw pointer and length, borrow the FlatView epoch (so the mapping cannot go away underneath), and expose only copying operations: `read() -> T`, `write(T)`, `copy_to(&mut [u8])`, `copy_from(&[u8])`, `fill(u8)`, and atomic operations for the few places that need them (virtio used ring index, Xen shared info, Hyper-V SynIC pages).
3. Scalar accesses use `read_volatile` and `write_volatile` of naturally aligned, naturally sized types, or `AtomicU16/U32/U64` with explicit orderings where the device protocol requires ordering (virtqueue avail and used indexes need acquire and release respectively, as specified by the virtio spec's memory barrier rules).
4. Bulk copies use a copy routine that is opaque to the compiler: on x86-64 an inline `rep movsb` (fast on every CPU with ERMS), on aarch64 a hand-written ldp/stp loop in inline assembly, elsewhere word-sized volatile loads and stores. Because inline assembly is outside the abstract machine, the compiler cannot assume the source is stable or the destination is unobserved, which is exactly the semantics of a DMA copy. This is also what the Linux kernel's Rust bindings are converging on for user and page memory.
5. `T: GuestScalar` is implemented only for types with no invalid bit patterns and no padding (integers, fixed-size byte arrays, and `#[repr(C)]` structs derived with `#[derive(GuestScalar)]`, which checks for padding at compile time). A `bool` or an enum is never read from guest memory directly; the device reads the integer and validates.
6. Endianness is explicit. `ld`/`st` take an `Endian`; `GuestPtr<Le<u32>>` and `GuestPtr<Be<u32>>` wrap byte order into the type so virtio (little-endian from 1.0) and legacy big-endian devices cannot mix them up.

```rust
pub struct GuestPtr<'g, T: GuestScalar> {
    ptr: NonNull<T>,
    _epoch: PhantomData<&'g RcuGuard>,
}

impl<'g, T: GuestScalar> GuestPtr<'g, T> {
    pub fn read(&self) -> T { unsafe { self.ptr.as_ptr().read_volatile() } }
    pub fn write(&self, v: T) { unsafe { self.ptr.as_ptr().write_volatile(v) } }
}

pub struct GuestSlice<'g> {
    ptr: NonNull<u8>,
    len: usize,
    _epoch: PhantomData<&'g RcuGuard>,
}

impl<'g> GuestSlice<'g> {
    pub fn copy_to(&self, dst: &mut [u8]) { /* opaque asm copy */ }
    pub fn copy_from(&self, src: &[u8]) { /* opaque asm copy, then dirty marking by caller's guard */ }
    pub fn subslice(&self, off: usize, len: usize) -> Option<GuestSlice<'g>>;
    pub fn as_iovec(&self) -> IoVec;  // for io_uring and writev: kernel does the copy
}
```

`as_iovec` is how block and network I/O avoid copies: the host kernel reads or writes guest memory directly through io_uring or a socket. Producing an iovec is safe because it hands the kernel a pointer and length and creates no Rust reference.

The lifetime `'g` ties every pointer to an RCU read guard. A device that needs a mapping across an asynchronous I/O (a block request in flight on io_uring) cannot hold an RCU guard that long, because that would stall reclamation. For that case `DmaMapping` has an owned form that takes a reference on the RamBlock (a counter, not an RCU guard). A RamBlock with outstanding owned mappings cannot be freed; unplugging its region waits for the counter to drain, the same way QEMU waits for in-flight requests before unplugging memory devices.

For interoperability, the crate `ruvm-mem-vmm` (MIT OR Apache-2.0) implements vm-memory's `GuestMemory` and `Bytes` traits on top of ruvm's address spaces, so rust-vmm crates such as virtio-queue and vhost-user-backend can run against ruvm memory and ruvm's memory model can be embedded elsewhere. New decision: this adapter exists from M2, because the virtqueue and vhost-user tests in document 13 use rust-vmm's crates as a differential oracle for ruvm's own ruvm-virtio-queue and ruvm-vhost.

## IOMMU regions

An IOMMU region's `translate(addr, flag, iommu_idx) -> IommuTlbEntry` returns a target address space, translated address, mask and permissions. Dispatch loops as `address_space_translate_iommu()` does until it reaches a non-IOMMU section, clamping the length to the page mask at each step. Notifiers subscribe with (start, end, flags, iommu_idx) for MAP, UNMAP and DEVIOTLB_UNMAP; `replay` walks current mappings; `attrs_to_index` lets an IOMMU (Arm SMMU secure streams) translate per MemTxAttrs. CPUs never sit behind device IOMMUs in QEMU machines, so the JIT does not see them. Details are in document 16.

## Diagnostics

`info mtree` prints the tree, `-f` the FlatViews, `-d` the dispatch, `-o` owners, `-D` disabled regions. The exact output format is part of the compatibility contract (document 02) because tests and tools grep it; ruvm's dispatch dump (`-d`) prints the boundary array instead of the radix tree and is the one intentional difference, noted in document 02's list of accepted deviations. Trace points use QEMU's names (`memory_region_ops_read`, `memory_region_ops_write`, `flatview_new`, `address_space_map`, and so on) so existing trace scripts work.

## Failure modes and how they surface

- Access to an unassigned address: MEMTX_DECODE_ERROR, value 0 for reads, a `guest_errors` log line, target-specific consequence.
- Device returns MEMTX_ERROR: propagated; Arm turns it into a synchronous external abort, x86 ignores it.
- Access rejected by `valid` constraints: MEMTX_DECODE_ERROR, as in QEMU (not ACCESS_ERROR).
- Re-entrant access blocked by the reentrancy guard: MEMTX_ACCESS_ERROR and a one-time warning.
- Bounce buffer exhaustion: short mapping, caller retries via map client callback.
- Memslot exhaustion when adding a region under KVM: the listener reports an error, the transaction still commits (QEMU cannot roll back and exits with "kvm_set_phys_mem: error registering slot"); ruvm prevents this earlier by checking the slot budget in memory device pre-plug and failing the hotplug with QMP error text identical to QEMU's.
- Preallocation failure (hugetlbfs pool exhausted): backend creation fails with QEMU's message; for hotplugged backends, the QMP `object-add` fails and nothing is mapped.
- A private (guest_memfd) page touched by device emulation: MEMTX_ACCESS_ERROR, trace event, and for virtio devices the queue is marked broken as QEMU does when descriptor memory is inaccessible.

## Performance budget

The memory core's targets, checked by benchmarks in document 21:

- Commit of a single BAR move on a q35 machine with 64 vCPUs: incremental FlatView and dispatch rebuild at least 10 times faster than QEMU 11.1 on the same host, measured with a qtest script that moves a BAR 10,000 times.
- MMIO dispatch lookup on a view of 128 sections: at most 10 ns on a current x86-64 or Apple silicon core, cache warm.
- RCU read-side enter and exit on Linux: no atomic read-modify-write, no fence, one thread-local store each.
- DMA map of a RAM range for virtio: no heap allocation, no lock, one dispatch lookup (normally an MRU hit).
- Preallocation of 64 GiB backed by 1 GiB huge pages with 16 threads: within 5% of the kernel's MADV_POPULATE_WRITE throughput limit.

## References

- QEMU memory API documentation: https://www.qemu.org/docs/master/devel/memory.html
- QEMU source: system/memory.c, system/physmem.c, include/system/memory.h, include/exec/memattrs.h, include/system/ram_addr.h, backends/hostmem.c, hw/virtio/virtio-mem.c, https://gitlab.com/qemu-project/qemu
- Linux KVM API (dirty ring, KVM_SET_USER_MEMORY_REGION2, KVM_CREATE_GUEST_MEMFD, immediate_exit): https://docs.kernel.org/virt/kvm/api.html
- guest_memfd mmap support, merged for Linux 6.18: https://ratatoskr.run/lkml/2025/07/3152629/t
- guest_memfd in-place conversion series, v13 in September 2026: https://ratatoskr.run/kvm/2026/09/17549361/t
- rust-vmm vm-memory: https://github.com/rust-vmm/vm-memory
- Rust RFC 3301, AtomicPerByte: https://github.com/rust-lang/rfcs/pull/3301
- Firecracker (Agache et al., NSDI 2020): https://www.usenix.org/conference/nsdi20/presentation/agache
