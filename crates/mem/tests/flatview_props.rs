// SPDX-License-Identifier: MIT OR Apache-2.0

//! Property tests: rendered flat views against a reference model.
//!
//! Each case builds a random region tree through the public API, applies random edits (mapping,
//! unmapping, moving, enabling, readonly, alias offsets, nested transactions) and after every
//! committed change compares the published view with a model that answers "who serves this
//! address" by walking the tree directly: highest priority first, newest first among equals,
//! aliases followed, readonly inherited. It also checks that the view is sorted, has no
//! overlaps and no ranges left unmerged, and that a listener replaying region_add and region_del
//! ends up with exactly the view.
//!
//! The number of cases comes from `PROPTEST_CASES` (256 if unset). The M1 exit criterion is a
//! million:
//!
//! ```text
//! PROPTEST_CASES=1000000 cargo test -p ruvm-mem --release --test flatview_props
//! ```

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use proptest::prelude::*;
use proptest::sample::Index;
use ruvm_mem::{AddressSpace, FlatRange, MemError, MemoryListener, MemorySystem, RegionId};

const SPACE: u128 = 1 << 64;

/// A region in the model.
#[derive(Clone, Debug)]
struct MNode {
    id: RegionId,
    size: u128,
    addr: u64,
    priority: i32,
    /// When the region was last mapped. Among equal priorities the highest wins.
    seq: u64,
    enabled: bool,
    readonly: bool,
    parent: Option<usize>,
    alias: Option<(usize, u64)>,
    terminates: bool,
}

struct Model {
    nodes: Vec<MNode>,
    seq: u64,
}

impl Model {
    fn children(&self, i: usize) -> Vec<usize> {
        let mut c: Vec<usize> =
            (0..self.nodes.len()).filter(|&j| self.nodes[j].parent == Some(i)).collect();
        c.sort_by(|&a, &b| {
            let (a, b) = (&self.nodes[a], &self.nodes[b]);
            b.priority.cmp(&a.priority).then(b.seq.cmp(&a.seq))
        });
        c
    }

    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut stack = vec![from];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(i) = stack.pop() {
            if i == to {
                return true;
            }
            if std::mem::replace(&mut seen[i], true) {
                continue;
            }
            stack.extend((0..self.nodes.len()).filter(|&j| self.nodes[j].parent == Some(i)));
            stack.extend(self.nodes[i].alias.map(|(t, _)| t));
        }
        false
    }

    /// Who serves `a`, with the offset into that region and whether writes are refused.
    fn walk(&self, i: usize, a: u64, base: i128, readonly: bool) -> Option<(usize, u64, bool)> {
        let n = &self.nodes[i];
        if !n.enabled {
            return None;
        }
        let base = base + i128::from(n.addr);
        let a = i128::from(a);
        if a < base || a >= base + n.size as i128 {
            return None;
        }
        let readonly = readonly || n.readonly;
        if let Some((t, off)) = n.alias {
            let tb = base - i128::from(self.nodes[t].addr) - i128::from(off);
            return self.walk(t, a as u64, tb, readonly);
        }
        for c in self.children(i) {
            if let Some(hit) = self.walk(c, a as u64, base, readonly) {
                return Some(hit);
            }
        }
        n.terminates.then_some((i, (a - base) as u64, readonly))
    }
}

/// Mirrors region_add and region_del into a map and notes anything inconsistent.
#[derive(Default)]
struct Mirror {
    map: Mutex<BTreeMap<u64, (RegionId, u128, u64, bool)>>,
    errors: Mutex<Vec<String>>,
}

fn key(r: &FlatRange) -> (RegionId, u128, u64, bool) {
    (r.region(), r.size(), r.offset_in_region(), r.readonly())
}

impl MemoryListener for Mirror {
    fn region_add(&self, _s: &AddressSpace, r: &FlatRange) {
        if let Some(old) = self.map.lock().unwrap().insert(r.addr(), key(r)) {
            self.errors.lock().unwrap().push(format!("add over {old:?} at {:x}", r.addr()));
        }
    }
    fn region_del(&self, _s: &AddressSpace, r: &FlatRange) {
        if self.map.lock().unwrap().remove(&r.addr()) != Some(key(r)) {
            self.errors.lock().unwrap().push(format!("del of unknown {r:?}"));
        }
    }
}

#[derive(Clone, Debug)]
enum Kind {
    Leaf,
    Container,
    Alias(Index, u64),
}

#[derive(Clone, Debug)]
enum Op {
    New(Kind, u128),
    /// Parent (`None` for the root, which most mappings go into), child, offset, priority.
    Add(Option<Index>, Index, u64, i32),
    Del(Index),
    Enable(Index, bool),
    Move(Index, u64),
    ReadOnly(Index, bool),
    AliasOffset(Index, u64),
    Begin,
    Commit,
}

fn addr() -> impl Strategy<Value = u64> {
    prop_oneof![
        8 => (0u64..0x50).prop_map(|x| x * 0x100),
        2 => 0u64..0x5000,
        1 => Just(u64::MAX - 0xfff),
        1 => any::<u64>(),
    ]
}

fn size() -> impl Strategy<Value = u128> {
    prop_oneof![
        6 => (1u128..0x30).prop_map(|x| x * 0x100),
        2 => 1u128..0x1000,
        1 => Just(0x10000),
        1 => Just(SPACE),
        1 => Just(0),
    ]
}

fn op() -> impl Strategy<Value = Op> {
    let kind = prop_oneof![
        3 => Just(Kind::Leaf),
        2 => Just(Kind::Container),
        2 => (any::<Index>(), addr()).prop_map(|(t, o)| Kind::Alias(t, o)),
    ];
    let parent = prop_oneof![1 => Just(None), 1 => any::<Index>().prop_map(Some)];
    prop_oneof![
        3 => (kind, size()).prop_map(|(k, s)| Op::New(k, s)),
        6 => (parent, any::<Index>(), addr(), prop_oneof![3 => 0i32..2, 1 => -1i32..3])
            .prop_map(|(p, c, a, pr)| Op::Add(p, c, a, pr)),
        1 => any::<Index>().prop_map(Op::Del),
        1 => (any::<Index>(), any::<bool>()).prop_map(|(r, e)| Op::Enable(r, e)),
        2 => (any::<Index>(), addr()).prop_map(|(r, a)| Op::Move(r, a)),
        1 => (any::<Index>(), any::<bool>()).prop_map(|(r, e)| Op::ReadOnly(r, e)),
        1 => (any::<Index>(), addr()).prop_map(|(r, a)| Op::AliasOffset(r, a)),
        1 => Just(Op::Begin),
        1 => Just(Op::Commit),
    ]
}

struct World {
    sys: MemorySystem,
    space: Arc<AddressSpace>,
    mirror: Arc<Mirror>,
    model: Model,
    depth: u32,
}

impl World {
    fn new() -> Self {
        let sys = MemorySystem::new();
        let root = sys.new_container("root", SPACE).unwrap();
        let space = sys.address_space_init(root, "as").unwrap();
        let mirror = Arc::new(Mirror::default());
        sys.register_listener(mirror.clone(), &space).unwrap();
        let root_node = MNode {
            id: root,
            size: SPACE,
            addr: 0,
            priority: 0,
            seq: 0,
            enabled: true,
            readonly: false,
            parent: None,
            alias: None,
            terminates: false,
        };
        World { sys, space, mirror, model: Model { nodes: vec![root_node], seq: 1 }, depth: 0 }
    }

    fn pick(&self, i: &Index) -> usize {
        i.index(self.model.nodes.len())
    }

    fn apply(&mut self, op: &Op) -> Result<(), TestCaseError> {
        let n = self.model.nodes.len();
        match op {
            Op::New(kind, size) => {
                if n >= 24 {
                    return Ok(());
                }
                let name = format!("r{n}");
                let (id, alias, terminates) = match kind {
                    Kind::Leaf => (self.sys.new_reservation(&name, *size).unwrap(), None, true),
                    Kind::Container => (self.sys.new_container(&name, *size).unwrap(), None, false),
                    Kind::Alias(t, off) => {
                        let t = self.pick(t);
                        let id =
                            self.sys.new_alias(&name, self.model.nodes[t].id, *off, *size).unwrap();
                        (id, Some((t, *off)), false)
                    }
                };
                self.model.nodes.push(MNode {
                    id,
                    size: *size,
                    addr: 0,
                    priority: 0,
                    seq: 0,
                    enabled: true,
                    readonly: false,
                    parent: None,
                    alias,
                    terminates,
                });
            }
            Op::Add(p, c, a, prio) => {
                let (p, c) = (p.as_ref().map_or(0, |p| self.pick(p)), self.pick(c));
                let r = self.sys.add_subregion_overlap(
                    self.model.nodes[p].id,
                    *a,
                    self.model.nodes[c].id,
                    *prio,
                );
                if self.model.nodes[c].parent.is_some() {
                    prop_assert!(matches!(r, Err(MemError::AlreadyMapped(_))), "{:?}", r);
                } else if self.model.reaches(c, p) {
                    prop_assert!(matches!(r, Err(MemError::Cycle(_))), "{:?}", r);
                } else {
                    prop_assert!(r.is_ok(), "{:?}", r);
                    let seq = self.model.seq;
                    self.model.seq += 1;
                    let node = &mut self.model.nodes[c];
                    node.parent = Some(p);
                    node.addr = *a;
                    node.priority = *prio;
                    node.seq = seq;
                }
            }
            Op::Del(c) => {
                let c = self.pick(c);
                let id = self.model.nodes[c].id;
                match self.model.nodes[c].parent {
                    Some(p) => {
                        self.sys.del_subregion(self.model.nodes[p].id, id).unwrap();
                        self.model.nodes[c].parent = None;
                    }
                    None => {
                        let r = self.sys.del_subregion(self.model.nodes[0].id, id);
                        prop_assert!(matches!(r, Err(MemError::NotASubregion(_))), "{:?}", r);
                    }
                }
            }
            Op::Enable(r, e) => {
                let r = self.pick(r);
                self.sys.set_enabled(self.model.nodes[r].id, *e).unwrap();
                self.model.nodes[r].enabled = *e;
            }
            Op::Move(r, a) => {
                let r = self.pick(r);
                self.sys.set_address(self.model.nodes[r].id, *a).unwrap();
                let seq = self.model.seq;
                let node = &mut self.model.nodes[r];
                if node.addr != *a {
                    node.addr = *a;
                    node.seq = seq;
                    self.model.seq += 1;
                }
            }
            Op::ReadOnly(r, v) => {
                let r = self.pick(r);
                self.sys.set_readonly(self.model.nodes[r].id, *v).unwrap();
                self.model.nodes[r].readonly = *v;
            }
            Op::AliasOffset(r, off) => {
                let r = self.pick(r);
                let res = self.sys.set_alias_offset(self.model.nodes[r].id, *off);
                match &mut self.model.nodes[r].alias {
                    Some((_, o)) => {
                        prop_assert!(res.is_ok());
                        *o = *off;
                    }
                    None => prop_assert!(matches!(res, Err(MemError::WrongKind(_)))),
                }
            }
            Op::Begin => {
                if self.depth < 3 {
                    self.sys.begin();
                    self.depth += 1;
                }
            }
            Op::Commit => {
                if self.depth > 0 {
                    self.sys.commit();
                    self.depth -= 1;
                }
            }
        }
        Ok(())
    }

    fn check(&self, probes: &[u64]) -> Result<(), TestCaseError> {
        let view = self.space.flatview();
        let ranges = view.ranges();
        for r in ranges {
            prop_assert!(r.size() > 0 && r.end() <= SPACE, "bad range {:?}", r);
        }
        for w in ranges.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            prop_assert!(a.end() <= u128::from(b.addr()), "overlap {:?} {:?}", a, b);
            let mergeable = a.end() == u128::from(b.addr())
                && a.region() == b.region()
                && u128::from(a.offset_in_region()) + a.size() == u128::from(b.offset_in_region())
                && a.readonly() == b.readonly();
            prop_assert!(!mergeable, "unmerged {:?} {:?}", a, b);
        }

        let mut addrs: Vec<u64> = probes.to_vec();
        addrs.extend((0..0x50).map(|x| x * 0x100));
        for r in ranges {
            addrs.push(r.addr());
            addrs.push(r.addr().wrapping_sub(1));
            addrs.push((r.end() - 1) as u64);
            addrs.push(r.end() as u64);
        }
        for a in addrs {
            let got = view.lookup(a).map(|r| {
                (r.region(), r.offset_in_region().wrapping_add(a - r.addr()), r.readonly())
            });
            let want = self
                .model
                .walk(0, a, 0, false)
                .map(|(i, off, ro)| (self.model.nodes[i].id, off, ro));
            prop_assert_eq!(got, want, "at {:#x}\nview {:#?}", a, view);
        }

        let seen = self.mirror.map.lock().unwrap().clone();
        let expect: BTreeMap<_, _> = ranges.iter().map(|r| (r.addr(), key(r))).collect();
        prop_assert_eq!(seen, expect);
        let errors = self.mirror.errors.lock().unwrap();
        prop_assert!(errors.is_empty(), "{:?}", *errors);
        Ok(())
    }
}

/// Many leaves in one container, enough that lookups go through the Eytzinger layout.
fn check_wide(leaves: &[(u64, u128, i32)], probes: &[u64]) -> Result<(), TestCaseError> {
    let mut w = World::new();
    let root = w.model.nodes[0].id;
    w.sys.begin();
    for (i, &(a, s, p)) in leaves.iter().enumerate() {
        let id = w.sys.new_reservation(&format!("l{i}"), s).unwrap();
        w.sys.add_subregion_overlap(root, a, id, p).unwrap();
        w.model.nodes.push(MNode {
            id,
            size: s,
            addr: a,
            priority: p,
            seq: i as u64 + 1,
            enabled: true,
            readonly: false,
            parent: Some(0),
            alias: None,
            terminates: true,
        });
    }
    w.sys.commit();
    w.check(probes)
}

proptest! {
    #[test]
    fn flatview_matches_reference(
        ops in prop::collection::vec(op(), 1..60),
        probes in prop::collection::vec(addr(), 8),
    ) {
        let mut w = World::new();
        for op in &ops {
            w.apply(op)?;
            if w.depth == 0 {
                w.check(&probes)?;
            }
        }
        while w.depth > 0 {
            w.sys.commit();
            w.depth -= 1;
        }
        w.check(&probes)?;
    }

    #[test]
    fn wide_flatviews_match_reference(
        leaves in prop::collection::vec(((0u64..0x400).prop_map(|x| x * 0x40), 1u128..0x200, -2i32..3), 40..160),
        probes in prop::collection::vec(0u64..0x11000, 32),
    ) {
        check_wide(&leaves, &probes)?;
    }
}
