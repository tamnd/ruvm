// SPDX-License-Identifier: MIT OR Apache-2.0

//! The region tree: an arena of nodes addressed by generation checked ids.

use std::fmt;
use std::sync::Arc;

use crate::access::MmioOps;
use crate::dirty::DirtyMask;
use crate::error::MemError;
use crate::iommu::IommuOps;
use crate::ram::RamBlock;

/// A handle on a memory region. Ids are not reused while the region lives, and an id kept after
/// the region is destroyed is refused rather than pointing at whatever took the slot.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionId {
    index: u32,
    generation: u32,
}

impl fmt::Debug for RegionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RegionId({}v{})", self.index, self.generation)
    }
}

/// What kind of region a node is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegionType {
    /// Backed by a [`RamBlock`]. ROM is RAM with the readonly flag set.
    Ram,
    /// A RAM block read directly in romd mode and a device otherwise, and always a device for
    /// writes.
    RomDevice,
    /// A device.
    Mmio,
    /// A window onto another region.
    Alias,
    /// Nothing of its own; subregions show through and holes fall through.
    Container,
    /// Accesses are translated and continued in another address space.
    Iommu,
    /// Claims an address range for something outside the emulator. Accesses behave as
    /// unassigned.
    Reservation,
}

/// What a flat range dispatches to. Built once when the region is created and shared by every
/// FlatView the region appears in, which keeps its callbacks alive as long as a reader may use
/// them.
pub(crate) struct RegionTarget {
    pub(crate) id: RegionId,
    pub(crate) name: String,
    pub(crate) ty: RegionType,
    pub(crate) kind: TargetKind,
}

pub(crate) enum TargetKind {
    Ram(Arc<RamBlock>),
    RomDevice(Arc<RamBlock>, Arc<dyn MmioOps>),
    Mmio(Arc<dyn MmioOps>),
    Iommu(Arc<dyn IommuOps>),
    /// Containers and aliases, which never appear in a FlatView.
    Nothing,
}

impl RegionTarget {
    pub(crate) fn ram_block(&self) -> Option<&Arc<RamBlock>> {
        match &self.kind {
            TargetKind::Ram(b) | TargetKind::RomDevice(b, _) => Some(b),
            _ => None,
        }
    }
}

pub(crate) struct Node {
    pub(crate) target: Arc<RegionTarget>,
    pub(crate) size: u128,
    pub(crate) alias: Option<(RegionId, u64)>,
    pub(crate) parent: Option<RegionId>,
    pub(crate) addr: u64,
    pub(crate) priority: i32,
    pub(crate) enabled: bool,
    pub(crate) readonly: bool,
    pub(crate) nonvolatile: bool,
    pub(crate) unmergeable: bool,
    pub(crate) romd_mode: bool,
    /// In rendering order: descending priority, and among equal priorities the most recently
    /// added first.
    pub(crate) children: Vec<RegionId>,
    pub(crate) dirty_log_mask: DirtyMask,
    pub(crate) vga_logging_count: u32,
    /// Aliases that point at this region.
    pub(crate) alias_users: u32,
}

impl Node {
    pub(crate) fn new(target: Arc<RegionTarget>, size: u128) -> Self {
        Node {
            target,
            size,
            alias: None,
            parent: None,
            addr: 0,
            priority: 0,
            enabled: true,
            readonly: false,
            nonvolatile: false,
            unmergeable: false,
            romd_mode: true,
            children: Vec::new(),
            dirty_log_mask: DirtyMask::NONE,
            vga_logging_count: 0,
            alias_users: 0,
        }
    }

    /// `mr->terminates`: the region renders into the view itself.
    pub(crate) fn terminates(&self) -> bool {
        !matches!(self.target.ty, RegionType::Container | RegionType::Alias)
    }

    pub(crate) fn name(&self) -> &str {
        &self.target.name
    }

    pub(crate) fn ty(&self) -> RegionType {
        self.target.ty
    }
}

struct Slot {
    generation: u32,
    node: Option<Node>,
}

/// The node storage.
#[derive(Default)]
pub(crate) struct Arena {
    slots: Vec<Slot>,
    free: Vec<u32>,
}

impl Arena {
    /// The id the next [`Arena::insert`] will return.
    pub(crate) fn next_id(&self) -> RegionId {
        match self.free.last() {
            Some(&index) => RegionId { index, generation: self.slots[index as usize].generation },
            None => RegionId { index: self.slots.len() as u32, generation: 0 },
        }
    }

    pub(crate) fn insert(&mut self, node: Node) -> RegionId {
        let id = self.next_id();
        match self.free.pop() {
            Some(index) => self.slots[index as usize].node = Some(node),
            None => self.slots.push(Slot { generation: 0, node: Some(node) }),
        }
        id
    }

    pub(crate) fn remove(&mut self, id: RegionId) -> Option<Node> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        let node = slot.node.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(id.index);
        Some(node)
    }

    pub(crate) fn get(&self, id: RegionId) -> Option<&Node> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.node.as_ref()
    }

    pub(crate) fn get_mut(&mut self, id: RegionId) -> Option<&mut Node> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation {
            return None;
        }
        slot.node.as_mut()
    }

    pub(crate) fn node(&self, id: RegionId) -> Result<&Node, MemError> {
        self.get(id).ok_or(MemError::NoSuchRegion)
    }

    pub(crate) fn node_mut(&mut self, id: RegionId) -> Result<&mut Node, MemError> {
        self.get_mut(id).ok_or(MemError::NoSuchRegion)
    }

    /// Every live region.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (RegionId, &Node)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.node.as_ref().map(|n| (RegionId { index: i as u32, generation: s.generation }, n))
        })
    }

    /// Whether `to` can be reached from `from` through subregion and alias edges, which is what
    /// makes rendering recurse forever if `to` then contains `from`.
    pub(crate) fn reaches(&self, from: RegionId, to: RegionId) -> bool {
        let mut stack = vec![from];
        let mut seen = std::collections::HashSet::new();
        while let Some(id) = stack.pop() {
            if id == to {
                return true;
            }
            if !seen.insert(id) {
                continue;
            }
            if let Some(n) = self.get(id) {
                stack.extend(n.children.iter().copied());
                stack.extend(n.alias.map(|(t, _)| t));
            }
        }
        false
    }

    /// Inserts `child` into `parent`'s list at the place its priority gives it,
    /// `memory_region_update_container_subregions()`: before the first sibling whose priority is
    /// not higher, so the newest of equal priorities comes first.
    pub(crate) fn link(&mut self, parent: RegionId, child: RegionId) {
        let prio = self.get(child).map_or(0, |n| n.priority);
        let pos = {
            let p = &self.get(parent).expect("parent checked by the caller").children;
            p.iter()
                .position(|c| self.get(*c).is_some_and(|o| prio >= o.priority))
                .unwrap_or(p.len())
        };
        if let Some(p) = self.get_mut(parent) {
            p.children.insert(pos, child);
        }
    }

    pub(crate) fn unlink(&mut self, parent: RegionId, child: RegionId) {
        if let Some(p) = self.get_mut(parent) {
            p.children.retain(|c| *c != child);
        }
    }
}

/// A snapshot of a region's properties, for inspection and for `info mtree` style output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionInfo {
    /// The name given at creation.
    pub name: String,
    /// The kind of region.
    pub ty: RegionType,
    /// The size in bytes, up to 2^64.
    pub size: u128,
    /// The container this region is mapped into.
    pub parent: Option<RegionId>,
    /// The offset in the container.
    pub addr: u64,
    /// The priority among siblings.
    pub priority: i32,
    /// Whether the region renders at all.
    pub enabled: bool,
    /// Writes are refused, inherited by everything below.
    pub readonly: bool,
    /// Nonvolatile memory, inherited by everything below.
    pub nonvolatile: bool,
    /// Flat ranges of this region are never merged with their neighbours.
    pub unmergeable: bool,
    /// For ROM devices, whether reads go straight to the RAM block.
    pub romd_mode: bool,
    /// For aliases, the target and the offset into it.
    pub alias: Option<(RegionId, u64)>,
    /// The subregions in rendering order.
    pub children: Vec<RegionId>,
    /// The dirty clients the region itself asked for with `memory_region_set_log()`.
    pub dirty_log_mask: DirtyMask,
}

impl RegionInfo {
    pub(crate) fn from_node(n: &Node) -> Self {
        RegionInfo {
            name: n.name().to_string(),
            ty: n.ty(),
            size: n.size,
            parent: n.parent,
            addr: n.addr,
            priority: n.priority,
            enabled: n.enabled,
            readonly: n.readonly,
            nonvolatile: n.nonvolatile,
            unmergeable: n.unmergeable,
            romd_mode: n.romd_mode,
            alias: n.alias,
            children: n.children.clone(),
            dirty_log_mask: n.dirty_log_mask,
        }
    }
}
