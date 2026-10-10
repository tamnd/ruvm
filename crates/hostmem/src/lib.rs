// SPDX-License-Identifier: GPL-2.0-or-later

//! The memory backend objects from backends/hostmem.c and backends/hostmem-ram.c:
//! `memory-backend` and `memory-backend-ram`, which is what `-object memory-backend-ram` and
//! `object-add` create to hold guest RAM.
//!
//! ruvm's RAM is plain process memory, so there is no host NUMA binding, and merging and core
//! dump exclusion are only accepted where QEMU accepts them, on Linux.

#![forbid(unsafe_code)]

pub mod region;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::region::RegionObjects;
use ruvm_base::{Error, Result};
use ruvm_mem::RegionId;
use ruvm_qapi::types::{HostMemPolicy, Memdev};
use ruvm_qapi::visit::VisitorExt;
use ruvm_qom::{
    BoolGetter, BoolSetter, EnumGetter, EnumSetter, LinkFlags, LinkSlot, Object, Property,
    Registry, TYPE_OBJECT, TYPE_USER_CREATABLE, TypeInfo, UserCreatableClass, allow_set_link,
};

/// `TYPE_MEMORY_BACKEND`.
pub const TYPE_MEMORY_BACKEND: &str = "memory-backend";
/// `TYPE_MEMORY_BACKEND_RAM`.
pub const TYPE_MEMORY_BACKEND_RAM: &str = "memory-backend-ram";
/// `TYPE_THREAD_CONTEXT`, what `prealloc-context` links to.
pub const TYPE_THREAD_CONTEXT: &str = "thread-context";

/// Whether the host can mark memory mergeable and keep it out of core dumps, which QEMU
/// checks as `QEMU_MADV_MERGEABLE` and `QEMU_MADV_DONTDUMP` being valid.
const HOST_MADVISE: bool = cfg!(target_os = "linux");

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `HostMemoryBackend`.
#[derive(Debug, Default)]
struct Backend {
    size: AtomicU64,
    merge: AtomicBool,
    dump: AtomicBool,
    prealloc: AtomicBool,
    prealloc_threads: AtomicU32,
    prealloc_context: Arc<LinkSlot>,
    policy: AtomicUsize,
    share: AtomicBool,
    reserve: AtomicBool,
    use_canonical_path: AtomicBool,
    mapped: AtomicBool,
    region: Mutex<Option<(Arc<RegionObjects>, RegionId)>>,
}

fn backend(obj: &Object) -> Arc<Backend> {
    obj.state::<Backend>().expect("a memory-backend object")
}

/// `host_memory_backend_mr_inited()`.
fn mr_inited(b: &Backend) -> bool {
    lock(&b.region).is_some()
}

/// `qmp_query_memdev()`: the memory backends under `/objects`. QEMU prepends each to the
/// list, so the newest comes first.
pub fn query_memdev(registry: &Registry) -> Vec<Memdev> {
    let mut out = Vec::new();
    registry.objects_root().child_foreach(|obj| {
        let Some(b) = obj.dynamic_cast(TYPE_MEMORY_BACKEND).and_then(|o| o.state::<Backend>())
        else {
            return 0;
        };
        out.push(Memdev {
            id: obj.canonical_path_component(),
            size: b.size.load(Ordering::Acquire),
            merge: b.merge.load(Ordering::Acquire),
            dump: b.dump.load(Ordering::Acquire),
            prealloc: b.prealloc.load(Ordering::Acquire),
            share: b.share.load(Ordering::Acquire),
            // The property only exists on Linux, as in QEMU.
            reserve: cfg!(target_os = "linux").then(|| b.reserve.load(Ordering::Acquire)),
            host_nodes: Vec::new(),
            policy: HostMemPolicy::ALL
                .get(b.policy.load(Ordering::Acquire))
                .copied()
                .unwrap_or_default(),
        });
        0
    });
    out.reverse();
    out
}

/// The RAM region of a completed backend, `host_memory_backend_get_memory()`.
pub fn backend_region(obj: &Object) -> Option<RegionId> {
    lock(&obj.state::<Backend>()?.region).as_ref().map(|(_, id)| *id)
}

/// The RAM block of a completed backend, for a machine that takes the backend as its memory.
pub fn backend_ram_block(obj: &Object) -> Option<Arc<ruvm_mem::RamBlock>> {
    let b = obj.state::<Backend>()?;
    let region = lock(&b.region);
    let (objects, id) = region.as_ref()?;
    objects.memory().ram_block(*id)
}

/// `host_memory_backend_set_mapped()`: the machine has put the backend's memory in the guest.
pub fn set_mapped(obj: &Object, mapped: bool) {
    backend(obj).mapped.store(mapped, Ordering::Release);
}

/// `host_memory_backend_is_mapped()`.
pub fn is_mapped(obj: &Object) -> bool {
    backend(obj).mapped.load(Ordering::Acquire)
}

/// `host_memory_backend_init()`. QEMU reads the defaults off the machine, so this does too
/// when there is one.
fn instance_init(obj: &Object) {
    let b = backend(obj);
    let machine = obj.registry().resolve_path("/machine").0;
    let get = |name: &str, default: bool| {
        machine.as_ref().and_then(|m| m.property_get_bool(name).ok()).unwrap_or(default)
    };
    b.merge.store(get("mem-merge", HOST_MADVISE), Ordering::Relaxed);
    b.dump.store(get("dump-guest-core", true), Ordering::Relaxed);
    b.reserve.store(true, Ordering::Relaxed);
    let cpus = machine
        .as_ref()
        .and_then(|m| m.property_get_qobject("smp").ok())
        .and_then(|smp| match smp {
            ruvm_qapi::QValue::Dict(d) => d.get("cpus").and_then(ruvm_qapi::QValue::as_i64),
            _ => None,
        })
        .unwrap_or(1);
    b.prealloc_threads.store(cpus as u32, Ordering::Relaxed);
}

/// `host_memory_backend_get_name()`: the RAM block is named after the backend's id, or its
/// whole path with `x-use-canonical-path-for-ramblock-id`.
fn backend_name(obj: &Object, b: &Backend) -> String {
    if b.use_canonical_path.load(Ordering::Acquire) {
        obj.canonical_path().unwrap_or_default()
    } else {
        obj.canonical_path_component().unwrap_or_default()
    }
}

/// `host_memory_backend_memory_complete()` with `ram_backend_memory_alloc()` as the alloc
/// hook.
fn ram_complete(objects: &Arc<RegionObjects>, obj: &Object) -> Result<()> {
    let b = backend(obj);
    let size = b.size.load(Ordering::Acquire);
    if size == 0 {
        return Err(Error::generic("can't create backend with size 0"));
    }
    let name = backend_name(obj, &b);
    let mem = objects.memory();
    // RAM_SHARED or RAM_PRIVATE, so the machine's aux-ram-share has no say.
    let share = b.share.load(Ordering::Acquire);
    let id = mem.new_ram_shared(&name, size, share).map_err(|e| Error::generic(e.to_string()))?;
    if let Err(e) = objects.add(obj, &name, id) {
        let _ = mem.destroy_region(id);
        return Err(e);
    }
    *lock(&b.region) = Some((objects.clone(), id));
    Ok(())
}

/// Frees the RAM when the backend goes, as finalizing its `MemoryRegion` does.
fn instance_finalize(obj: &Object) {
    if let Some((objects, id)) = lock(&backend(obj).region).take() {
        let _ = objects.memory().destroy_region(id);
    }
}

/// `host_memory_backend_prepare_delete()`.
fn prepare_delete(obj: &Object) -> Result<()> {
    if is_mapped(obj) {
        return Err(Error::generic(format!(
            "Cannot delete host memory backend '{}' which is mapped",
            obj.canonical_path_component().unwrap_or_default()
        )));
    }
    Ok(())
}

fn bool_prop(
    get: impl Fn(&Backend) -> bool + Send + Sync + 'static,
    set: impl Fn(&Object, &Backend, bool) -> Result<()> + Send + Sync + 'static,
) -> (Option<BoolGetter>, Option<BoolSetter>) {
    let get: BoolGetter = Arc::new(move |o: &Object| Ok(get(&backend(o))));
    let set: BoolSetter = Arc::new(move |o: &Object, v: bool| set(o, &backend(o), v));
    (Some(get), Some(set))
}

fn class_init(k: &Arc<ruvm_qom::ObjectClass>) {
    let (g, s) = bool_prop(
        |b| b.merge.load(Ordering::Acquire),
        |_, b, v| {
            if !HOST_MADVISE {
                if v {
                    return Err(Error::generic("Memory merging is not supported on this host"));
                }
                return Ok(());
            }
            b.merge.store(v, Ordering::Release);
            Ok(())
        },
    );
    k.property_add_bool("merge", g, s).set_description(Some("Mark memory as mergeable"));

    let (g, s) = bool_prop(
        |b| b.dump.load(Ordering::Acquire),
        |_, b, v| {
            if !HOST_MADVISE {
                if !v {
                    return Err(Error::generic(
                        "Dumping guest memory cannot be disabled on this host",
                    ));
                }
                return Ok(());
            }
            b.dump.store(v, Ordering::Release);
            Ok(())
        },
    );
    k.property_add_bool("dump", g, s)
        .set_description(Some("Set to 'off' to exclude from core dump"));

    // Guest RAM is zeroed memory that is already there, so preallocating is only a flag.
    let (g, s) = bool_prop(
        |b| b.prealloc.load(Ordering::Acquire),
        |_, b, v| {
            if !b.reserve.load(Ordering::Acquire) && v {
                return Err(Error::generic("'prealloc=on' and 'reserve=off' are incompatible"));
            }
            // Once the memory is there, prealloc can be turned on but not off again.
            if !mr_inited(b) || v {
                b.prealloc.store(v, Ordering::Release);
            }
            Ok(())
        },
    );
    k.property_add_bool("prealloc", g, s).set_description(Some("Preallocate memory"));

    k.property_add(
        Property::new("prealloc-threads", "int")
            .getter(|o, v, name| {
                let mut n = backend(o).prealloc_threads.load(Ordering::Acquire);
                v.type_uint32(Some(name), &mut n)
            })
            .setter(|o, v, name| {
                let mut n = 0;
                v.type_uint32(Some(name), &mut n)?;
                if n == 0 {
                    return Err(Error::generic(format!(
                        "property '{name}' of {} doesn't take value '{n}'",
                        o.typename()
                    )));
                }
                backend(o).prealloc_threads.store(n, Ordering::Release);
                Ok(())
            })
            .description("Number of CPU threads to use for prealloc"),
    );

    k.property_add_link(
        "prealloc-context",
        TYPE_THREAD_CONTEXT,
        |o| backend(o).prealloc_context.clone(),
        Some(Arc::new(allow_set_link)),
        LinkFlags::STRONG,
    )
    .set_description(Some("Context to use for creating CPU threads for preallocation"));

    k.property_add(
        Property::new("size", "int")
            .getter(|o, v, name| {
                let mut size = backend(o).size.load(Ordering::Acquire);
                v.type_size(Some(name), &mut size)
            })
            .setter(|o, v, name| {
                let b = backend(o);
                if mr_inited(&b) {
                    return Err(Error::generic(format!(
                        "cannot change property {name} of {} ",
                        o.typename()
                    )));
                }
                let mut size = 0;
                v.type_size(Some(name), &mut size)?;
                if size == 0 {
                    return Err(Error::generic(format!(
                        "property '{name}' of {} doesn't take value '0'",
                        o.typename()
                    )));
                }
                b.size.store(size, Ordering::Release);
                Ok(())
            })
            .description("Size of the memory region (ex: 500M)"),
    );

    // Without libnuma QEMU reports the empty set and refuses to bind.
    k.property_add(
        Property::new("host-nodes", "int")
            .getter(|_, v, name| {
                let mut nodes: Vec<u16> = Vec::new();
                v.visit_list(Some(name), &mut nodes, |v, n| v.type_uint16(None, n))
            })
            .setter(|_, _, _| {
                Err(Error::generic("NUMA node binding are not supported by this QEMU"))
            })
            .description("Binds memory to the list of NUMA host nodes"),
    );

    let get: EnumGetter = Arc::new(|o: &Object| Ok(backend(o).policy.load(Ordering::Acquire)));
    let set: EnumSetter = Arc::new(|o: &Object, v: usize| {
        backend(o).policy.store(v, Ordering::Release);
        if v != HostMemPolicy::Default as usize {
            return Err(Error::generic("NUMA policies are not supported by this QEMU"));
        }
        Ok(())
    });
    k.property_add_enum("policy", "HostMemPolicy", HostMemPolicy::LOOKUP, Some(get), Some(set))
        .set_description(Some("Set the NUMA policy"));

    let (g, s) = bool_prop(
        |b| b.share.load(Ordering::Acquire),
        |_, b, v| {
            if mr_inited(b) {
                return Err(Error::generic("cannot change property value"));
            }
            b.share.store(v, Ordering::Release);
            Ok(())
        },
    );
    k.property_add_bool("share", g, s)
        .set_description(Some("Mark the memory as private to QEMU or shared"));

    if cfg!(target_os = "linux") {
        let (g, s) = bool_prop(
            |b| b.reserve.load(Ordering::Acquire),
            |_, b, v| {
                if mr_inited(b) {
                    return Err(Error::generic("cannot change property value"));
                }
                if b.prealloc.load(Ordering::Acquire) && !v {
                    return Err(Error::generic("'prealloc=on' and 'reserve=off' are incompatible"));
                }
                b.reserve.store(v, Ordering::Release);
                Ok(())
            },
        );
        k.property_add_bool("reserve", g, s)
            .set_description(Some("Reserve swap space (or huge pages) if applicable"));
    }

    let (g, s) = bool_prop(
        |b| b.use_canonical_path.load(Ordering::Acquire),
        |_, b, v| {
            b.use_canonical_path.store(v, Ordering::Release);
            Ok(())
        },
    );
    k.property_add_bool("x-use-canonical-path-for-ramblock-id", g, s);
}

/// Registers `memory-backend` and `memory-backend-ram`. Their RAM goes into the memory system
/// behind `objects`.
pub fn register_types(registry: &Registry, objects: &Arc<RegionObjects>) {
    let base = TypeInfo::new(TYPE_MEMORY_BACKEND)
        .parent(TYPE_OBJECT)
        .abstract_()
        .interface(TYPE_USER_CREATABLE)
        .instance_state(Backend::default)
        .instance_init(instance_init)
        .instance_finalize(instance_finalize)
        .class_init(class_init);

    let objects = objects.clone();
    let ram =
        TypeInfo::new(TYPE_MEMORY_BACKEND_RAM).parent(TYPE_MEMORY_BACKEND).class_init(move |k| {
            let objects = objects.clone();
            let uc = k.interface(TYPE_USER_CREATABLE).expect("backends are user creatable");
            uc.set_ext(UserCreatableClass {
                complete: Some(Arc::new(move |o| ram_complete(&objects, o))),
                prepare_delete: Some(Arc::new(prepare_delete)),
            });
        });
    registry.register_all([base, ram]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_mem::MemorySystem;

    fn setup() -> (Registry, Arc<RegionObjects>) {
        let registry = Registry::new();
        region::register_types(&registry);
        let objects = RegionObjects::new(Arc::new(MemorySystem::new()));
        register_types(&registry, &objects);
        (registry, objects)
    }

    fn add(registry: &Registry, props: &[(&str, &str)]) -> Result<Object> {
        let parent = registry.objects_root();
        registry.object_new_with_props(TYPE_MEMORY_BACKEND_RAM, Some((&parent, "node0")), props)
    }

    #[test]
    fn a_ram_backend_owns_its_region() {
        let (registry, objects) = setup();
        let obj = add(&registry, &[("size", "64K")]).unwrap();
        assert_eq!(obj.property_get_uint("size").unwrap(), 65536);
        assert!(obj.property_get_bool("dump").unwrap());
        assert_eq!(obj.property_get_int("prealloc-threads").unwrap(), 1);
        assert_eq!(obj.property_get_str("policy").unwrap(), "default");
        let id = backend_region(&obj).unwrap();
        let info = objects.memory().region(id).unwrap();
        assert_eq!((info.name.as_str(), info.size), ("node0", 65536));
        let (mr, _) = registry.resolve_path("/objects/node0/node0[0]");
        assert_eq!(mr.unwrap().property_get_uint("size").unwrap(), 65536);

        let e = obj.property_set_str("size", "1M").unwrap_err();
        assert_eq!(e.message(), "cannot change property size of memory-backend-ram ");
        let e = obj.property_set_bool("share", true).unwrap_err();
        assert_eq!(e.message(), "cannot change property value");

        set_mapped(&obj, true);
        let e = ruvm_qom::qmp::object_del(&registry, "node0").unwrap_err();
        assert_eq!(e.message(), "Cannot delete host memory backend 'node0' which is mapped");
        set_mapped(&obj, false);
        drop(obj);
        ruvm_qom::qmp::object_del(&registry, "node0").unwrap();
        assert!(objects.memory().region(id).is_none());
    }

    #[test]
    fn bad_values() {
        let (registry, _) = setup();
        let e = add(&registry, &[]).unwrap_err();
        assert_eq!(e.message(), "can't create backend with size 0");
        let e = add(&registry, &[("size", "0")]).unwrap_err();
        assert_eq!(e.message(), "property 'size' of memory-backend-ram doesn't take value '0'");
        let e = add(&registry, &[("size", "4K"), ("prealloc-threads", "0")]).unwrap_err();
        assert_eq!(
            e.message(),
            "property 'prealloc-threads' of memory-backend-ram doesn't take value '0'"
        );
        let e = add(&registry, &[("size", "4K"), ("policy", "bind")]).unwrap_err();
        assert_eq!(e.message(), "NUMA policies are not supported by this QEMU");
        let e = add(&registry, &[("size", "4K"), ("host-nodes", "0")]).unwrap_err();
        assert_eq!(e.message(), "NUMA node binding are not supported by this QEMU");
    }
}
