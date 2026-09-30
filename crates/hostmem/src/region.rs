// SPDX-License-Identifier: GPL-2.0-or-later

//! The `memory-region` QOM type from system/memory.c. It lives here rather than in ruvm-mem,
//! which stays free of the GPL QOM crates. Every region QEMU creates is also an
//! object in the QOM tree, a child of its owner or of `/machine/unattached`, and QMP clients
//! read its `addr`, `size`, `priority` and `container` from there.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use ruvm_base::{Error, Result};
use ruvm_qapi::visit::VisitorExt;
use ruvm_qom::{Object, Property, Registry, TYPE_OBJECT, TypeInfo, WeakObject};

use ruvm_mem::{MemorySystem, RegionId, RegionInfo};

/// `TYPE_MEMORY_REGION`.
pub const TYPE_MEMORY_REGION: &str = "memory-region";

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// What a `memory-region` object stands for.
#[derive(Debug, Default)]
struct RegionState {
    region: OnceLock<(Arc<RegionObjects>, RegionId)>,
}

/// The objects that stand for the regions of one [`MemorySystem`], so a region's `container`
/// link can name the object of the region it is mapped into.
#[derive(Debug)]
pub struct RegionObjects {
    mem: Arc<MemorySystem>,
    objects: Mutex<HashMap<RegionId, WeakObject>>,
}

impl RegionObjects {
    pub fn new(mem: Arc<MemorySystem>) -> Arc<Self> {
        Arc::new(RegionObjects { mem, objects: Mutex::new(HashMap::new()) })
    }

    pub fn memory(&self) -> &Arc<MemorySystem> {
        &self.mem
    }

    /// The object for `id`, if it has one that is still alive.
    pub fn object(&self, id: RegionId) -> Option<Object> {
        lock(&self.objects).get(&id).and_then(WeakObject::upgrade)
    }

    /// The part of `memory_region_init()` that makes the object: a `memory-region` child of
    /// `owner` called `name[N]`, with the first free `N`.
    pub fn add(self: &Arc<Self>, owner: &Object, name: &str, id: RegionId) -> Result<Object> {
        let obj = owner.registry().object_new(TYPE_MEMORY_REGION)?;
        let st = obj.state::<RegionState>().expect("a memory-region object");
        st.region.set((self.clone(), id)).expect("a new object");
        owner.property_try_add_child(&format!("{name}[*]"), &obj)?;
        lock(&self.objects).insert(id, obj.downgrade());
        Ok(obj)
    }
}

fn info(obj: &Object) -> Result<(Arc<RegionObjects>, RegionInfo)> {
    let st = obj.state::<RegionState>().expect("a memory-region object");
    let (objects, id) = st.region.get().ok_or_else(|| Error::generic("region not initialized"))?;
    let info = objects.mem.region(*id).ok_or_else(|| Error::generic("region is gone"))?;
    Ok((objects.clone(), info))
}

/// Registers `memory-region`.
pub fn register_types(registry: &Registry) {
    let mr = TypeInfo::new(TYPE_MEMORY_REGION)
        .parent(TYPE_OBJECT)
        .instance_state(RegionState::default)
        .class_init(|k| {
            // memory_region_get_container(): the path of the container, or "" when unmapped.
            k.property_add(Property::new("container", "link<memory-region>").getter(
                |obj, v, name| {
                    let (objects, info) = info(obj)?;
                    let mut path = info
                        .parent
                        .and_then(|p| objects.object(p))
                        .and_then(|o| o.canonical_path())
                        .unwrap_or_default();
                    v.type_str(Some(name), &mut path)
                },
            ));
            k.property_add(Property::new("addr", "uint64").getter(|obj, v, name| {
                let mut addr = info(obj)?.1.addr;
                v.type_uint64(Some(name), &mut addr)
            }));
            // A 2^64 region reads as UINT64_MAX, as int128_get64() of the saturated size does.
            k.property_add(Property::new("size", "uint64").getter(|obj, v, name| {
                let mut size = u64::try_from(info(obj)?.1.size).unwrap_or(u64::MAX);
                v.type_uint64(Some(name), &mut size)
            }));
            k.property_add(Property::new("priority", "uint32").getter(|obj, v, name| {
                let mut prio = info(obj)?.1.priority as u32;
                v.type_uint32(Some(name), &mut prio)
            }));
        });
    registry.register(mr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_show_up_as_objects() {
        let registry = Registry::new();
        register_types(&registry);
        let mem = Arc::new(MemorySystem::new());
        let objects = RegionObjects::new(mem.clone());
        let root = registry.root();

        let sys = mem.new_container("system", 1 << 64).unwrap();
        let ram = mem.new_ram("ram", 0x1000).unwrap();
        mem.add_subregion_overlap(sys, 0x2000, ram, -1).unwrap();
        let s = objects.add(&root, "system", sys).unwrap();
        let r = objects.add(&root, "ram", ram).unwrap();
        assert_eq!(s.canonical_path().as_deref(), Some("/system[0]"));

        assert_eq!(s.property_get_uint("size").unwrap(), u64::MAX);
        assert_eq!(s.property_get_str("container").unwrap(), "");
        assert_eq!(r.property_get_uint("addr").unwrap(), 0x2000);
        assert_eq!(r.property_get_uint("priority").unwrap(), u64::from(u32::MAX));
        assert_eq!(r.property_get_str("container").unwrap(), "/system[0]");
    }
}
