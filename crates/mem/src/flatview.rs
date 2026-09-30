// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flattening a region tree into sorted, non-overlapping ranges, and finding the range for an
//! address.
//!
//! Rendering follows `render_memory_region()` and `flatview_simplify()` in system/memory.c, since
//! which region answers at an address is guest visible. Lookup does not use QEMU's radix tree.
//! A view keeps the start address of every range and of every hole between them in one sorted
//! array and binary searches it, which needs no subpage machinery and rebuilds in linear time
//! (spec/05 has the numbers). Above [`EYTZINGER_THRESHOLD`] boundaries the array is stored in
//! Eytzinger order, so the first probes of every search share a few cache lines.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::dirty::{DirtyClient, DirtyMask};
use crate::ram::RamBlock;
use crate::region::{Arena, Node, RegionId, RegionTarget, RegionType};

/// 2^64, the end of every address space.
pub(crate) const SPACE_END: u128 = 1 << 64;

/// Views with more boundaries than this are searched in Eytzinger order.
pub const EYTZINGER_THRESHOLD: usize = 64;

/// A piece of an address space served by one region, `FlatRange`.
#[derive(Clone)]
pub struct FlatRange {
    pub(crate) target: Arc<RegionTarget>,
    pub(crate) addr: u64,
    pub(crate) size: u128,
    pub(crate) offset_in_region: u64,
    pub(crate) readonly: bool,
    pub(crate) nonvolatile: bool,
    pub(crate) unmergeable: bool,
    pub(crate) romd_mode: bool,
    pub(crate) dirty_log_mask: DirtyMask,
    /// The region's own readonly flag, not the inherited one. This is what decides whether a
    /// write goes straight to RAM, as `memory_access_is_direct()` looks at `mr->readonly`.
    pub(crate) region_readonly: bool,
}

impl FlatRange {
    /// The region serving this range.
    pub fn region(&self) -> RegionId {
        self.target.id
    }

    /// The region's name.
    pub fn name(&self) -> &str {
        &self.target.name
    }

    /// The region's kind.
    pub fn region_type(&self) -> RegionType {
        self.target.ty
    }

    /// The first address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// The size in bytes. A range covering a whole address space is 2^64 long.
    pub fn size(&self) -> u128 {
        self.size
    }

    /// One past the last address.
    pub fn end(&self) -> u128 {
        u128::from(self.addr) + self.size
    }

    /// Where in the region the range starts.
    pub fn offset_in_region(&self) -> u64 {
        self.offset_in_region
    }

    /// Writes are refused, because the region or a container above it is readonly.
    pub fn readonly(&self) -> bool {
        self.readonly
    }

    /// The memory is nonvolatile.
    pub fn nonvolatile(&self) -> bool {
        self.nonvolatile
    }

    /// The range is never merged with a neighbour.
    pub fn unmergeable(&self) -> bool {
        self.unmergeable
    }

    /// For a ROM device, whether reads go straight to RAM.
    pub fn romd_mode(&self) -> bool {
        self.romd_mode
    }

    /// The dirty clients logging this range.
    pub fn dirty_log_mask(&self) -> DirtyMask {
        self.dirty_log_mask
    }

    /// The RAM block behind the range, for RAM, ROM and ROM devices.
    pub fn ram_block(&self) -> Option<&Arc<RamBlock>> {
        self.target.ram_block()
    }

    /// `flatrange_equal()`: same place, same region, same attributes. The dirty log mask is left
    /// out on purpose, because a change there is a log_start or log_stop, not a new range.
    pub(crate) fn same_as(&self, o: &FlatRange) -> bool {
        self.target.id == o.target.id
            && self.addr == o.addr
            && self.size == o.size
            && self.offset_in_region == o.offset_in_region
            && self.romd_mode == o.romd_mode
            && self.readonly == o.readonly
            && self.nonvolatile == o.nonvolatile
            && self.unmergeable == o.unmergeable
    }

    /// `can_merge()`.
    fn can_merge(&self, next: &FlatRange) -> bool {
        self.end() == u128::from(next.addr)
            && self.target.id == next.target.id
            && u128::from(self.offset_in_region) + self.size == u128::from(next.offset_in_region)
            && self.dirty_log_mask == next.dirty_log_mask
            && self.romd_mode == next.romd_mode
            && self.readonly == next.readonly
            && self.nonvolatile == next.nonvolatile
            && !self.unmergeable
            && !next.unmergeable
    }
}

impl fmt::Debug for FlatRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:016x}-{:016x} '{}' @{:x}",
            self.addr,
            self.end().saturating_sub(1),
            self.name(),
            self.offset_in_region
        )?;
        if self.readonly {
            f.write_str(" ro")?;
        }
        if self.nonvolatile {
            f.write_str(" nv")?;
        }
        if !self.romd_mode {
            f.write_str(" !romd")?;
        }
        Ok(())
    }
}

/// Global inputs to the dirty mask of a range, `memory_region_get_dirty_log_mask()`.
#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct RenderFlags {
    pub(crate) global_dirty_log: bool,
    pub(crate) code_dirty_log: bool,
}

pub(crate) fn dirty_log_mask(n: &Node, flags: RenderFlags) -> DirtyMask {
    let mut mask = n.dirty_log_mask;
    let ram = n.target.ram_block().is_some();
    if flags.global_dirty_log && (ram || n.ty() == RegionType::Iommu) {
        mask = mask.with(DirtyClient::Migration);
    }
    if flags.code_dirty_log && ram {
        mask = mask.with(DirtyClient::Code);
    }
    mask
}

struct Renderer<'a> {
    arena: &'a Arena,
    flags: RenderFlags,
    out: Vec<(i128, i128, FlatRange)>,
}

#[derive(Copy, Clone)]
struct Inherit {
    readonly: bool,
    nonvolatile: bool,
    unmergeable: bool,
}

impl Renderer<'_> {
    /// `render_memory_region()`. `clip` is a half open range; ranges already in the output
    /// obscure this region.
    fn render(&mut self, id: RegionId, mut base: i128, clip: (i128, i128), inh: Inherit) {
        let Some(n) = self.arena.get(id) else { return };
        if !n.enabled {
            return;
        }
        base += i128::from(n.addr);
        let inh = Inherit {
            readonly: inh.readonly | n.readonly,
            nonvolatile: inh.nonvolatile | n.nonvolatile,
            unmergeable: inh.unmergeable | n.unmergeable,
        };
        let start = clip.0.max(base);
        let end = clip.1.min(base + n.size as i128);
        if start >= end {
            return;
        }
        let clip = (start, end);

        if let Some((target, offset)) = n.alias {
            let taddr = self.arena.get(target).map_or(0, |t| t.addr);
            let base = base - i128::from(taddr) - i128::from(offset);
            self.render(target, base, clip, inh);
            return;
        }

        for &child in &n.children {
            self.render(child, base, clip, inh);
        }

        if !n.terminates() {
            return;
        }

        let range = FlatRange {
            target: Arc::clone(&n.target),
            addr: 0,
            size: 0,
            offset_in_region: 0,
            readonly: inh.readonly,
            nonvolatile: inh.nonvolatile,
            unmergeable: inh.unmergeable,
            romd_mode: n.romd_mode,
            dirty_log_mask: dirty_log_mask(n, self.flags),
            region_readonly: n.readonly,
        };
        // Fill the gaps the output leaves in the clip.
        let mut pos = start;
        let mut i = self.out.partition_point(|r| r.1 <= pos);
        while pos < end {
            if let Some(r) = self.out.get(i) {
                if r.0 <= pos {
                    pos = r.1;
                    i += 1;
                    continue;
                }
            }
            let next = self.out.get(i).map_or(end, |r| r.0.min(end));
            let mut piece = range.clone();
            piece.addr = pos as u64;
            piece.size = (next - pos) as u128;
            piece.offset_in_region = (pos - base) as u64;
            self.out.insert(i, (pos, next, piece));
            i += 1;
            pos = next;
        }
    }
}

/// Renders the tree under `root` into sorted, merged flat ranges, `generate_memory_topology()`
/// without the dispatch part.
pub(crate) fn render(arena: &Arena, root: RegionId, flags: RenderFlags) -> Vec<FlatRange> {
    let mut r = Renderer { arena, flags, out: Vec::new() };
    let inh = Inherit { readonly: false, nonvolatile: false, unmergeable: false };
    r.render(root, 0, (0, SPACE_END as i128), inh);
    simplify(r.out.into_iter().map(|(_, _, fr)| fr).collect())
}

/// `flatview_simplify()`: merges neighbours that continue the same region with the same
/// attributes.
fn simplify(ranges: Vec<FlatRange>) -> Vec<FlatRange> {
    let mut out: Vec<FlatRange> = Vec::with_capacity(ranges.len());
    for r in ranges {
        match out.last_mut() {
            Some(last) if last.can_merge(&r) => last.size += r.size,
            _ => out.push(r),
        }
    }
    out
}

/// No range at this boundary: a hole, which dispatches as unassigned.
const HOLE: u32 = u32::MAX;

/// Keys in Eytzinger order and the sorted index of each.
type EytzingerArrays = (Box<[u64]>, Box<[u32]>);

/// The sorted boundary array.
struct Dispatch {
    /// Start of every range and every hole, sorted. The first is always 0.
    starts: Box<[u64]>,
    /// For each start, the index of its range or [`HOLE`].
    targets: Box<[u32]>,
    /// The same starts in Eytzinger order, 1 based, and for each the index into `starts`.
    eytzinger: Option<EytzingerArrays>,
}

impl Dispatch {
    fn new(ranges: &[FlatRange]) -> Self {
        let mut starts = Vec::with_capacity(ranges.len() * 2 + 1);
        let mut targets = Vec::with_capacity(ranges.len() * 2 + 1);
        let mut cursor: u128 = 0;
        for (i, r) in ranges.iter().enumerate() {
            if u128::from(r.addr) > cursor {
                starts.push(cursor as u64);
                targets.push(HOLE);
            }
            starts.push(r.addr);
            targets.push(i as u32);
            cursor = r.end();
        }
        if cursor < SPACE_END {
            starts.push(cursor as u64);
            targets.push(HOLE);
        }
        let eytzinger = (starts.len() > EYTZINGER_THRESHOLD).then(|| eytzinger(&starts));
        Dispatch { starts: starts.into(), targets: targets.into(), eytzinger }
    }

    /// The index of the last start at or below `addr`.
    fn find(&self, addr: u64) -> usize {
        match &self.eytzinger {
            Some((keys, idx)) => {
                let n = keys.len() - 1;
                let mut k = 1;
                while k <= n {
                    k = 2 * k + usize::from(keys[k] <= addr);
                }
                // The answer is the last node the descent went right from: drop the left turns
                // after it and then the right turn itself.
                k >>= k.trailing_zeros() + 1;
                idx[k] as usize
            }
            None => self.starts.partition_point(|&s| s <= addr) - 1,
        }
    }

    fn end_of(&self, b: usize) -> u128 {
        self.starts.get(b + 1).map_or(SPACE_END, |&s| u128::from(s))
    }
}

fn eytzinger(sorted: &[u64]) -> EytzingerArrays {
    fn fill(sorted: &[u64], keys: &mut [u64], idx: &mut [u32], k: usize, next: &mut usize) {
        if k < keys.len() {
            fill(sorted, keys, idx, 2 * k, next);
            keys[k] = sorted[*next];
            idx[k] = *next as u32;
            *next += 1;
            fill(sorted, keys, idx, 2 * k + 1, next);
        }
    }
    let mut keys = vec![0; sorted.len() + 1];
    let mut idx = vec![0; sorted.len() + 1];
    fill(sorted, &mut keys, &mut idx, 1, &mut 0);
    (keys.into(), idx.into())
}

static GENERATION: AtomicU64 = AtomicU64::new(1);

/// The flattened memory map of an address space, `FlatView`, with its dispatch structure.
///
/// A view never changes once built. A topology change builds a new one and publishes it, and a
/// reader holding the old one keeps using it until it lets go.
pub struct FlatView {
    ranges: Vec<FlatRange>,
    dispatch: Dispatch,
    generation: u64,
}

impl FlatView {
    pub(crate) fn new(ranges: Vec<FlatRange>) -> Self {
        let dispatch = Dispatch::new(&ranges);
        FlatView { ranges, dispatch, generation: GENERATION.fetch_add(1, Ordering::Relaxed) }
    }

    /// A view with nothing in it.
    pub fn empty() -> Self {
        FlatView::new(Vec::new())
    }

    /// The ranges, sorted by address.
    pub fn ranges(&self) -> &[FlatRange] {
        &self.ranges
    }

    /// The number of ranges.
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    /// Whether the view has no ranges.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// A number unique to this view, which caches compare to notice that the map changed.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The number of entries in the boundary array, ranges plus holes.
    pub fn boundaries(&self) -> usize {
        self.dispatch.starts.len()
    }

    /// The range containing `addr`, or `None` for a hole.
    pub fn lookup(&self, addr: u64) -> Option<&FlatRange> {
        self.translate(addr).0
    }

    /// The range containing `addr` (or `None` for a hole) and how many bytes from `addr` to its
    /// end.
    pub(crate) fn translate(&self, addr: u64) -> (Option<&FlatRange>, u128) {
        let b = self.dispatch.find(addr);
        let left = self.dispatch.end_of(b) - u128::from(addr);
        match self.dispatch.targets[b] {
            HOLE => (None, left),
            i => (Some(&self.ranges[i as usize]), left),
        }
    }
}

impl fmt::Debug for FlatView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(&self.ranges).finish()
    }
}
