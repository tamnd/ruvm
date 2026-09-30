// SPDX-License-Identifier: MIT OR Apache-2.0

//! The memory system: the region tree, the address spaces over it, the listeners following it
//! and the transaction that ties changes together.
//!
//! QEMU keeps all of this in globals under the big lock. Here it is one [`MemorySystem`] value
//! with a mutex inside, so tests can build as many as they like and a machine owns its own.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use crate::access::MmioOps;
use crate::address_space::AddressSpace;
use crate::dirty::{DirtyClient, DirtySnapshot};
use crate::error::MemError;
use crate::flatview::{FlatView, RenderFlags, SPACE_END, dirty_log_mask, render};
use crate::iommu::IommuOps;
use crate::listener::{ListenerId, MemoryListener};
use crate::ram::RamBlock;
use crate::region::{Arena, Node, RegionId, RegionInfo, RegionTarget, RegionType, TargetKind};

/// `GLOBAL_DIRTY_MIGRATION`: global dirty tracking for live migration.
pub const GLOBAL_DIRTY_MIGRATION: u32 = 1 << 0;
/// `GLOBAL_DIRTY_DIRTY_RATE`: global dirty tracking for dirty rate measurement.
pub const GLOBAL_DIRTY_DIRTY_RATE: u32 = 1 << 1;
/// `GLOBAL_DIRTY_LIMIT`: global dirty tracking for the dirty limit.
pub const GLOBAL_DIRTY_LIMIT: u32 = 1 << 2;

thread_local! {
    static IN_LISTENER: Cell<bool> = const { Cell::new(false) };
}

/// Marks the current thread as running listener callbacks until dropped.
struct ListenerScope(bool);

impl ListenerScope {
    fn enter() -> Self {
        ListenerScope(IN_LISTENER.with(|f| f.replace(true)))
    }
}

impl Drop for ListenerScope {
    fn drop(&mut self) {
        let was = self.0;
        IN_LISTENER.with(|f| f.set(was));
    }
}

/// Settings fixed when a [`MemorySystem`] is created.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MemoryConfig {
    /// Dirty tracking granularity as a shift, `TARGET_PAGE_BITS`.
    pub page_bits: u32,
    /// Whether RAM is logged for the CODE client, which a translating accelerator needs to
    /// notice writes to translated code.
    pub code_dirty_log: bool,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig { page_bits: 12, code_dirty_log: false }
    }
}

struct ListenerEntry {
    id: ListenerId,
    priority: i32,
    listener: Arc<dyn MemoryListener>,
    space: Arc<AddressSpace>,
}

struct Inner {
    arena: Arena,
    spaces: Vec<Arc<AddressSpace>>,
    listeners: Vec<ListenerEntry>,
    next_listener: u64,
    depth: u32,
    pending: bool,
    global_dirty: u32,
}

impl Inner {
    /// An address space renders its root at the root's own offset, so moving a root changes
    /// the space even when the root is mapped nowhere visible. QEMU does not render then and
    /// the move shows up with whatever change comes next; rendering now gives the same map
    /// sooner.
    fn root_moved(&mut self, id: RegionId) {
        if self.spaces.iter().any(|s| s.root() == id) {
            self.pending = true;
        }
    }
}

/// Everything memory: regions, address spaces, listeners and dirty tracking.
///
/// Methods that change the tree take effect at the end of the outermost transaction, like
/// `memory_region_transaction_begin()` and `memory_region_transaction_commit()`. Outside a
/// transaction each call is its own. Readers of an [`AddressSpace`] see either the whole old map
/// or the whole new one.
pub struct MemorySystem {
    config: MemoryConfig,
    inner: Mutex<Inner>,
}

/// An open transaction, committed when dropped. See [`MemorySystem::transaction`].
#[must_use = "the transaction commits as soon as the guard is dropped"]
pub struct Transaction<'a> {
    system: &'a MemorySystem,
}

impl fmt::Debug for Transaction<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Transaction")
    }
}

impl Drop for Transaction<'_> {
    fn drop(&mut self) {
        self.system.commit();
    }
}

impl Default for MemorySystem {
    fn default() -> Self {
        MemorySystem::new()
    }
}

impl fmt::Debug for MemorySystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = self.lock();
        f.debug_struct("MemorySystem")
            .field("config", &self.config)
            .field("regions", &g.arena.iter().count())
            .field("address_spaces", &g.spaces.iter().map(|s| s.name()).collect::<Vec<_>>())
            .field("listeners", &g.listeners.len())
            .finish()
    }
}

fn check_size(size: u128) -> Result<(), MemError> {
    if size > SPACE_END { Err(MemError::TooLarge(size)) } else { Ok(()) }
}

impl MemorySystem {
    /// A memory system with the default [`MemoryConfig`].
    pub fn new() -> Self {
        MemorySystem::with_config(MemoryConfig::default())
    }

    /// A memory system with `config`.
    pub fn with_config(config: MemoryConfig) -> Self {
        MemorySystem {
            config,
            inner: Mutex::new(Inner {
                arena: Arena::default(),
                spaces: Vec::new(),
                listeners: Vec::new(),
                next_listener: 0,
                depth: 0,
                pending: false,
                global_dirty: 0,
            }),
        }
    }

    /// The settings the system was created with.
    pub fn config(&self) -> MemoryConfig {
        self.config
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        assert!(
            !IN_LISTENER.with(Cell::get),
            "the memory API must not be called from a memory listener callback"
        );
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn flags(&self, g: &Inner) -> RenderFlags {
        RenderFlags {
            global_dirty_log: g.global_dirty != 0,
            code_dirty_log: self.config.code_dirty_log,
        }
    }

    /// Runs `f` inside a transaction.
    fn edit<R>(&self, f: impl FnOnce(&mut Inner) -> Result<R, MemError>) -> Result<R, MemError> {
        let mut g = self.lock();
        g.depth += 1;
        let r = f(&mut g);
        self.commit_locked(&mut g);
        r
    }

    fn create(
        &self,
        name: &str,
        size: u128,
        ty: RegionType,
        kind: TargetKind,
    ) -> Result<RegionId, MemError> {
        check_size(size)?;
        let mut g = self.lock();
        let id = g.arena.next_id();
        let target = Arc::new(RegionTarget { id, name: name.to_string(), ty, kind });
        Ok(g.arena.insert(Node::new(target, size)))
    }

    fn new_block(&self, name: &str, size: u64) -> Result<Arc<RamBlock>, MemError> {
        let block = RamBlock::new(name, size, self.config.page_bits)?;
        // Like QEMU, every block has a bitmap for every client, and which bits get set is up to
        // the dirty log mask of the region written through.
        for client in DirtyClient::ALL {
            block.start_dirty_log(client);
        }
        Ok(Arc::new(block))
    }

    /// A container, `memory_region_init()`: no contents of its own, only subregions. `size`
    /// may be up to 2^64.
    pub fn new_container(&self, name: &str, size: u128) -> Result<RegionId, MemError> {
        self.create(name, size, RegionType::Container, TargetKind::Nothing)
    }

    /// RAM, `memory_region_init_ram()`, zero filled.
    pub fn new_ram(&self, name: &str, size: u64) -> Result<RegionId, MemError> {
        let block = self.new_block(name, size)?;
        self.create(name, u128::from(size), RegionType::Ram, TargetKind::Ram(block))
    }

    /// ROM, `memory_region_init_rom()`: RAM the guest cannot write. Fill it through
    /// [`MemorySystem::ram_block`] or with a debug write.
    pub fn new_rom(&self, name: &str, size: u64) -> Result<RegionId, MemError> {
        let id = self.new_ram(name, size)?;
        self.lock().arena.node_mut(id)?.readonly = true;
        Ok(id)
    }

    /// A ROM device, `memory_region_init_rom_device()`: reads come from RAM while in romd mode
    /// (the default) and go to `ops` otherwise, and writes always go to `ops`.
    pub fn new_rom_device(
        &self,
        name: &str,
        size: u64,
        ops: Arc<dyn MmioOps>,
    ) -> Result<RegionId, MemError> {
        let block = self.new_block(name, size)?;
        self.create(
            name,
            u128::from(size),
            RegionType::RomDevice,
            TargetKind::RomDevice(block, ops),
        )
    }

    /// A device, `memory_region_init_io()`.
    pub fn new_io(
        &self,
        name: &str,
        size: u128,
        ops: Arc<dyn MmioOps>,
    ) -> Result<RegionId, MemError> {
        self.create(name, size, RegionType::Mmio, TargetKind::Mmio(ops))
    }

    /// A reservation, `memory_region_init_io()` with no callbacks: the range is claimed and
    /// accesses behave as unassigned.
    pub fn new_reservation(&self, name: &str, size: u128) -> Result<RegionId, MemError> {
        self.create(name, size, RegionType::Reservation, TargetKind::Nothing)
    }

    /// An IOMMU region, `memory_region_init_iommu()`.
    pub fn new_iommu(
        &self,
        name: &str,
        size: u128,
        ops: Arc<dyn IommuOps>,
    ) -> Result<RegionId, MemError> {
        self.create(name, size, RegionType::Iommu, TargetKind::Iommu(ops))
    }

    /// An alias, `memory_region_init_alias()`: `size` bytes of `target` starting at `offset`.
    pub fn new_alias(
        &self,
        name: &str,
        target: RegionId,
        offset: u64,
        size: u128,
    ) -> Result<RegionId, MemError> {
        check_size(size)?;
        let mut g = self.lock();
        g.arena.node_mut(target)?.alias_users += 1;
        let id = g.arena.next_id();
        let t = Arc::new(RegionTarget {
            id,
            name: name.to_string(),
            ty: RegionType::Alias,
            kind: TargetKind::Nothing,
        });
        let mut node = Node::new(t, size);
        node.alias = Some((target, offset));
        Ok(g.arena.insert(node))
    }

    /// Frees a region, the end of `memory_region_finalize()`. The region must not be mapped,
    /// aliased or the root of an address space. Its subregions are unmapped. Views that still
    /// show it keep its callbacks and RAM alive until the last reader lets go.
    pub fn destroy_region(&self, id: RegionId) -> Result<(), MemError> {
        self.edit(|g| {
            let n = g.arena.node(id)?;
            let name = n.name().to_string();
            if n.parent.is_some() || n.alias_users > 0 || g.spaces.iter().any(|s| s.root() == id) {
                return Err(MemError::InUse(name));
            }
            let children = n.children.clone();
            let enabled = n.enabled;
            for c in children {
                let child = g.arena.node_mut(c)?;
                child.parent = None;
                g.pending |= enabled && child.enabled;
            }
            if let Some(n) = g.arena.remove(id) {
                if let Some((target, _)) = n.alias {
                    if let Some(t) = g.arena.get_mut(target) {
                        t.alias_users -= 1;
                    }
                }
            }
            Ok(())
        })
    }

    /// Maps `sub` into `container` at `offset` with priority 0,
    /// `memory_region_add_subregion()`.
    pub fn add_subregion(
        &self,
        container: RegionId,
        offset: u64,
        sub: RegionId,
    ) -> Result<(), MemError> {
        self.add_subregion_overlap(container, offset, sub, 0)
    }

    /// Maps `sub` into `container` at `offset` with `priority`,
    /// `memory_region_add_subregion_overlap()`. Where siblings overlap the higher priority wins,
    /// and among equal priorities the one added last.
    pub fn add_subregion_overlap(
        &self,
        container: RegionId,
        offset: u64,
        sub: RegionId,
        priority: i32,
    ) -> Result<(), MemError> {
        self.edit(|g| {
            let parent_enabled = g.arena.node(container)?.enabled;
            let n = g.arena.node(sub)?;
            if n.parent.is_some() {
                return Err(MemError::AlreadyMapped(n.name().to_string()));
            }
            if sub == container || g.arena.reaches(sub, container) {
                return Err(MemError::Cycle(n.name().to_string()));
            }
            let n = g.arena.node_mut(sub)?;
            n.parent = Some(container);
            n.addr = offset;
            n.priority = priority;
            let enabled = n.enabled;
            g.arena.link(container, sub);
            g.pending |= parent_enabled && enabled;
            g.root_moved(sub);
            Ok(())
        })
    }

    /// Unmaps `sub` from `container`, `memory_region_del_subregion()`.
    pub fn del_subregion(&self, container: RegionId, sub: RegionId) -> Result<(), MemError> {
        self.edit(|g| {
            let parent_enabled = g.arena.node(container)?.enabled;
            let n = g.arena.node_mut(sub)?;
            if n.parent != Some(container) {
                return Err(MemError::NotASubregion(n.name().to_string()));
            }
            n.parent = None;
            let enabled = n.enabled;
            g.arena.unlink(container, sub);
            g.pending |= parent_enabled && enabled;
            Ok(())
        })
    }

    /// Shows or hides a region and everything below it, `memory_region_set_enabled()`.
    pub fn set_enabled(&self, id: RegionId, enabled: bool) -> Result<(), MemError> {
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            if n.enabled != enabled {
                n.enabled = enabled;
                g.pending = true;
            }
            Ok(())
        })
    }

    /// Moves a region within its container, `memory_region_set_address()`. Like QEMU this
    /// unmaps and maps it again, so it becomes the newest among siblings of its priority.
    pub fn set_address(&self, id: RegionId, addr: u64) -> Result<(), MemError> {
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            if n.addr == addr {
                return Ok(());
            }
            n.addr = addr;
            let enabled = n.enabled;
            if let Some(parent) = n.parent {
                g.arena.unlink(parent, id);
                g.arena.link(parent, id);
                g.pending |= enabled && g.arena.node(parent)?.enabled;
            }
            g.root_moved(id);
            Ok(())
        })
    }

    /// Changes where an alias starts in its target, `memory_region_set_alias_offset()`.
    pub fn set_alias_offset(&self, id: RegionId, offset: u64) -> Result<(), MemError> {
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            let Some((target, old)) = n.alias else {
                return Err(MemError::WrongKind(n.name().to_string()));
            };
            if old != offset {
                n.alias = Some((target, offset));
                g.pending |= n.enabled;
            }
            Ok(())
        })
    }

    /// Changes a region's size, `memory_region_set_size()`. A region backed by a RAM block
    /// cannot grow past the block.
    pub fn set_size(&self, id: RegionId, size: u128) -> Result<(), MemError> {
        check_size(size)?;
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            if n.target.ram_block().is_some_and(|b| u128::from(b.len()) < size) {
                return Err(MemError::OutOfRange);
            }
            if n.size != size {
                n.size = size;
                g.pending = true;
            }
            Ok(())
        })
    }

    fn set_flag(
        &self,
        id: RegionId,
        v: bool,
        field: fn(&mut Node) -> &mut bool,
    ) -> Result<(), MemError> {
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            let f = field(n);
            if *f != v {
                *f = v;
                g.pending |= n.enabled;
            }
            Ok(())
        })
    }

    /// Makes a region and everything below it refuse writes, `memory_region_set_readonly()`.
    pub fn set_readonly(&self, id: RegionId, readonly: bool) -> Result<(), MemError> {
        self.set_flag(id, readonly, |n| &mut n.readonly)
    }

    /// Marks a region as nonvolatile memory, `memory_region_set_nonvolatile()`.
    pub fn set_nonvolatile(&self, id: RegionId, nonvolatile: bool) -> Result<(), MemError> {
        self.set_flag(id, nonvolatile, |n| &mut n.nonvolatile)
    }

    /// Keeps the flat ranges of a region from being merged, `memory_region_set_unmergeable()`.
    pub fn set_unmergeable(&self, id: RegionId, unmergeable: bool) -> Result<(), MemError> {
        self.set_flag(id, unmergeable, |n| &mut n.unmergeable)
    }

    /// Switches a ROM device between direct reads and device reads,
    /// `memory_region_rom_device_set_romd()`.
    pub fn set_romd(&self, id: RegionId, romd: bool) -> Result<(), MemError> {
        {
            let g = self.lock();
            let n = g.arena.node(id)?;
            if n.ty() != RegionType::RomDevice {
                return Err(MemError::WrongKind(n.name().to_string()));
            }
        }
        self.set_flag(id, romd, |n| &mut n.romd_mode)
    }

    /// Turns dirty logging of a region on or off for `client`, `memory_region_set_log()`. Only
    /// [`DirtyClient::Vga`] can be logged per region. Calls nest: logging stops when every
    /// caller that turned it on has turned it off.
    pub fn set_log(&self, id: RegionId, log: bool, client: DirtyClient) -> Result<(), MemError> {
        if client != DirtyClient::Vga {
            return Err(MemError::InvalidClient);
        }
        self.edit(|g| {
            let n = g.arena.node_mut(id)?;
            let old = n.vga_logging_count;
            n.vga_logging_count = if log { old + 1 } else { old.saturating_sub(1) };
            if (old != 0) != (n.vga_logging_count != 0) {
                n.dirty_log_mask = if log {
                    n.dirty_log_mask.with(client)
                } else {
                    n.dirty_log_mask.without(client)
                };
                g.pending |= n.enabled;
            }
            Ok(())
        })
    }

    /// A snapshot of a region's properties, or `None` if the id is stale.
    pub fn region(&self, id: RegionId) -> Option<RegionInfo> {
        self.lock().arena.get(id).map(RegionInfo::from_node)
    }

    /// The RAM block of a RAM, ROM or ROM device region, `memory_region_get_ram_ptr()` in spirit.
    pub fn ram_block(&self, id: RegionId) -> Option<Arc<RamBlock>> {
        self.lock().arena.get(id).and_then(|n| n.target.ram_block().cloned())
    }

    /// Renders the tree under `root` into a view without publishing it anywhere,
    /// `generate_memory_topology()`. Useful for inspection and tests.
    pub fn render(&self, root: RegionId) -> Result<FlatView, MemError> {
        let g = self.lock();
        g.arena.node(root)?;
        Ok(FlatView::new(render(&g.arena, root, self.flags(&g))))
    }

    /// Creates an address space over `root`, `address_space_init()`.
    pub fn address_space_init(
        &self,
        root: RegionId,
        name: &str,
    ) -> Result<Arc<AddressSpace>, MemError> {
        let mut g = self.lock();
        g.arena.node(root)?;
        let view = Arc::new(FlatView::new(render(&g.arena, root, self.flags(&g))));
        let space = Arc::new(AddressSpace::new(name, root, view));
        g.spaces.push(Arc::clone(&space));
        Ok(space)
    }

    /// Removes an address space, `address_space_destroy()`. Its listeners must be unregistered
    /// first. Holders of the `Arc` see an empty map from then on.
    pub fn address_space_destroy(&self, space: &Arc<AddressSpace>) -> Result<(), MemError> {
        let mut g = self.lock();
        let pos = g
            .spaces
            .iter()
            .position(|s| Arc::ptr_eq(s, space))
            .ok_or(MemError::NoSuchAddressSpace)?;
        if g.listeners.iter().any(|l| Arc::ptr_eq(&l.space, space)) {
            return Err(MemError::InUse(space.name().to_string()));
        }
        g.spaces.remove(pos);
        space.publish(Arc::new(FlatView::empty()));
        Ok(())
    }

    /// The address spaces, in creation order.
    pub fn address_spaces(&self) -> Vec<Arc<AddressSpace>> {
        self.lock().spaces.clone()
    }

    /// Registers `listener` on `space`, `memory_listener_register()`. The listener is told about
    /// the current map at once, as if every range had just been added.
    pub fn register_listener(
        &self,
        listener: Arc<dyn MemoryListener>,
        space: &Arc<AddressSpace>,
    ) -> Result<ListenerId, MemError> {
        let mut g = self.lock();
        if !g.spaces.iter().any(|s| Arc::ptr_eq(s, space)) {
            return Err(MemError::NoSuchAddressSpace);
        }
        let id = ListenerId(g.next_listener);
        g.next_listener += 1;
        let priority = listener.priority();
        let pos =
            g.listeners.iter().position(|l| priority < l.priority).unwrap_or(g.listeners.len());
        g.listeners.insert(
            pos,
            ListenerEntry {
                id,
                priority,
                listener: Arc::clone(&listener),
                space: Arc::clone(space),
            },
        );

        let _scope = ListenerScope::enter();
        if g.global_dirty != 0 {
            // Nothing to undo if this fails: the listener simply does not track.
            let _ = listener.log_global_start();
        }
        let view = space.flatview();
        listener.begin();
        for fr in view.ranges() {
            listener.region_add(space, fr);
            if !fr.dirty_log_mask().is_empty() {
                listener.log_start(space, fr, crate::DirtyMask::NONE, fr.dirty_log_mask());
            }
        }
        listener.commit();
        Ok(id)
    }

    /// Unregisters a listener, `memory_listener_unregister()`. It is told about the current map
    /// going away first.
    pub fn unregister_listener(&self, id: ListenerId) -> Result<(), MemError> {
        let mut g = self.lock();
        let pos = g.listeners.iter().position(|l| l.id == id).ok_or(MemError::NoSuchListener)?;
        let e = g.listeners.remove(pos);
        let _scope = ListenerScope::enter();
        let view = e.space.flatview();
        e.listener.begin();
        for fr in view.ranges() {
            if !fr.dirty_log_mask().is_empty() {
                e.listener.log_stop(&e.space, fr, fr.dirty_log_mask(), crate::DirtyMask::NONE);
            }
            e.listener.region_del(&e.space, fr);
        }
        e.listener.commit();
        Ok(())
    }

    /// Opens a transaction that commits when the guard is dropped.
    pub fn transaction(&self) -> Transaction<'_> {
        self.begin();
        Transaction { system: self }
    }

    /// `memory_region_transaction_begin()`. Every call needs a matching [`MemorySystem::commit`].
    pub fn begin(&self) {
        self.lock().depth += 1;
    }

    /// `memory_region_transaction_commit()`. The outermost commit rebuilds the views that changed
    /// and tells the listeners.
    pub fn commit(&self) {
        let mut g = self.lock();
        self.commit_locked(&mut g);
    }

    fn commit_locked(&self, g: &mut Inner) {
        assert!(g.depth > 0, "memory transaction commit without begin");
        g.depth -= 1;
        if g.depth == 0 && g.pending {
            self.update(g);
        }
    }

    /// Rebuilds every address space's view and tells the listeners what changed.
    fn update(&self, g: &mut Inner) {
        let flags = self.flags(g);
        let _scope = ListenerScope::enter();
        for l in &g.listeners {
            l.listener.begin();
        }
        let mut views: HashMap<RegionId, Arc<FlatView>> = HashMap::new();
        for space in &g.spaces {
            let new =
                Arc::clone(views.entry(space.root()).or_insert_with(|| {
                    Arc::new(FlatView::new(render(&g.arena, space.root(), flags)))
                }));
            let old = space.flatview();
            if g.listeners.iter().any(|l| Arc::ptr_eq(&l.space, space)) {
                let ls: Vec<&dyn MemoryListener> = g
                    .listeners
                    .iter()
                    .filter(|l| Arc::ptr_eq(&l.space, space))
                    .map(|l| &*l.listener)
                    .collect();
                topology_pass(space, &ls, &old, &new, false);
                topology_pass(space, &ls, &old, &new, true);
            }
            space.publish(new);
        }
        g.pending = false;
        for l in &g.listeners {
            l.listener.commit();
        }
    }

    /// Starts global dirty tracking for the reasons in `flags`, `memory_global_dirty_log_start()`.
    /// When tracking goes from off to on every listener's `log_global_start` runs, and if one
    /// fails the ones before it are stopped again and the error is returned.
    pub fn global_dirty_log_start(&self, flags: u32) -> Result<(), MemError> {
        let mut g = self.lock();
        let flags = flags & !g.global_dirty;
        if flags == 0 {
            return Ok(());
        }
        let old = g.global_dirty;
        g.global_dirty |= flags;
        if old == 0 {
            let failed = {
                let _scope = ListenerScope::enter();
                let mut failed = None;
                for (i, l) in g.listeners.iter().enumerate() {
                    if let Err(e) = l.listener.log_global_start() {
                        for prev in g.listeners[..i].iter().rev() {
                            prev.listener.log_global_stop();
                        }
                        failed = Some(e);
                        break;
                    }
                }
                failed
            };
            if let Some(e) = failed {
                g.global_dirty &= !flags;
                return Err(e);
            }
            g.depth += 1;
            g.pending = true;
            self.commit_locked(&mut g);
        }
        Ok(())
    }

    /// Stops global dirty tracking for the reasons in `flags`, `memory_global_dirty_log_stop()`.
    pub fn global_dirty_log_stop(&self, flags: u32) {
        let mut g = self.lock();
        let was = g.global_dirty;
        g.global_dirty &= !flags;
        if was != 0 && g.global_dirty == 0 {
            g.depth += 1;
            g.pending = true;
            self.commit_locked(&mut g);
            let _scope = ListenerScope::enter();
            for l in g.listeners.iter().rev() {
                l.listener.log_global_stop();
            }
        }
    }

    /// The reasons global dirty tracking is on, `global_dirty_tracking`.
    pub fn global_dirty_tracking(&self) -> u32 {
        self.lock().global_dirty
    }

    fn sync_locked(&self, g: &Inner, region: Option<RegionId>, last_stage: bool) {
        let _scope = ListenerScope::enter();
        for l in &g.listeners {
            if l.listener.log_sync_is_global() {
                l.listener.log_sync_global(last_stage);
                continue;
            }
            let view = l.space.flatview();
            for fr in view.ranges() {
                if !fr.dirty_log_mask().is_empty() && region.is_none_or(|r| r == fr.region()) {
                    l.listener.log_sync(&l.space, fr);
                }
            }
        }
    }

    /// Asks the listeners to copy their dirty bits for `region`, or for everything if `None`,
    /// into the RAM blocks, `memory_region_sync_dirty_bitmap()`.
    pub fn sync_dirty_bitmap(&self, region: Option<RegionId>, last_stage: bool) {
        let g = self.lock();
        self.sync_locked(&g, region, last_stage);
    }

    /// `memory_global_dirty_log_sync()`.
    pub fn global_dirty_log_sync(&self, last_stage: bool) {
        self.sync_dirty_bitmap(None, last_stage);
    }

    /// `memory_global_after_dirty_log_sync()`.
    pub fn global_after_dirty_log_sync(&self) {
        let g = self.lock();
        let _scope = ListenerScope::enter();
        for l in &g.listeners {
            l.listener.log_global_after_sync();
        }
    }

    fn with_block<R>(
        &self,
        id: RegionId,
        f: impl FnOnce(&Inner, &Node, &RamBlock) -> R,
    ) -> Result<R, MemError> {
        let g = self.lock();
        let n = g.arena.node(id)?;
        let block =
            n.target.ram_block().ok_or_else(|| MemError::WrongKind(n.name().to_string()))?;
        Ok(f(&g, n, block))
    }

    /// Marks `[addr, addr + size)` of a RAM region dirty for every client logging it, as a
    /// write from outside the guest would, `memory_region_set_dirty()`.
    pub fn set_dirty(&self, id: RegionId, addr: u64, size: u64) -> Result<(), MemError> {
        let flags = self.flags(&self.lock());
        self.with_block(id, |_, n, b| b.set_dirty(addr, size, dirty_log_mask(n, flags)))
    }

    /// Whether any page of `[addr, addr + size)` of a RAM region is dirty for `client`,
    /// `memory_region_get_dirty()`.
    pub fn get_dirty(
        &self,
        id: RegionId,
        addr: u64,
        size: u64,
        client: DirtyClient,
    ) -> Result<bool, MemError> {
        self.with_block(id, |_, _, b| b.get_dirty(addr, size, client))
    }

    /// Clears `[addr, addr + size)` of a RAM region for `client`, `memory_region_reset_dirty()`.
    pub fn reset_dirty(
        &self,
        id: RegionId,
        addr: u64,
        size: u64,
        client: DirtyClient,
    ) -> Result<(), MemError> {
        self.with_block(id, |_, _, b| {
            b.test_and_clear_dirty(addr, size, client);
        })
    }

    /// Syncs the region, then takes and clears its dirty bits for `client` over
    /// `[addr, addr + size)`, `memory_region_snapshot_and_clear_dirty()`. Display devices use
    /// this once per frame.
    pub fn snapshot_and_clear_dirty(
        &self,
        id: RegionId,
        addr: u64,
        size: u64,
        client: DirtyClient,
    ) -> Result<DirtySnapshot, MemError> {
        self.with_block(id, |g, _, b| {
            self.sync_locked(g, Some(id), false);
            b.snapshot_and_clear_dirty(addr, size, client)
        })
    }
}

type Callback<'a> = &'a dyn Fn(&dyn MemoryListener);

/// `address_space_update_topology_pass()`: walks the old and new views side by side. The first
/// pass removes what went away or changed, in reverse listener order. The second adds what is
/// new and reports unchanged ranges and dirty logging changes.
fn topology_pass(
    space: &AddressSpace,
    listeners: &[&dyn MemoryListener],
    old: &FlatView,
    new: &FlatView,
    adding: bool,
) {
    let forward = |f: Callback<'_>| listeners.iter().for_each(|l| f(*l));
    let reverse = |f: Callback<'_>| listeners.iter().rev().for_each(|l| f(*l));
    let (old, new) = (old.ranges(), new.ranges());
    let (mut i, mut j) = (0, 0);
    while i < old.len() || j < new.len() {
        let o = old.get(i);
        let n = new.get(j);
        match (o, n) {
            (Some(o), n)
                if n.is_none_or(|n| o.addr < n.addr || (o.addr == n.addr && !o.same_as(n))) =>
            {
                if !adding {
                    reverse(&|l| l.region_del(space, o));
                }
                i += 1;
            }
            (Some(o), Some(n)) if o.same_as(n) => {
                if adding {
                    forward(&|l| l.region_nop(space, n));
                    let (om, nm) = (o.dirty_log_mask(), n.dirty_log_mask());
                    if !nm.difference(om).is_empty() {
                        forward(&|l| l.log_start(space, n, om, nm));
                    }
                    if !om.difference(nm).is_empty() {
                        reverse(&|l| l.log_stop(space, n, om, nm));
                    }
                }
                i += 1;
                j += 1;
            }
            (_, Some(n)) => {
                if adding {
                    forward(&|l| l.region_add(space, n));
                }
                j += 1;
            }
            _ => break,
        }
    }
}
