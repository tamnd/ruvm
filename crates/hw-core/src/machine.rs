// SPDX-License-Identifier: GPL-2.0-or-later

//! `TYPE_MACHINE` from hw/core/machine.c, machine `none` from hw/core/null-machine.c, and the
//! parts of `qemu_create_machine()` and `memory_map_init()` that build `/machine` and the
//! system memory and I/O address spaces.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::{Error, Result};
use ruvm_hostmem::region::RegionObjects;
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, MemResult, MmioOps, RegionId,
};
use ruvm_qapi::types::{
    BootConfiguration, CacheLevelAndType, CpuTopologyLevel, MemorySizeConfiguration,
    SMPConfiguration, SmpCacheProperties,
};
use ruvm_qapi::visit::{Visit, VisitorExt};
use ruvm_qom::{
    BoolGetter, BoolSetter, LinkFlags, LinkSlot, Object, ObjectClass, Property, Registry,
    StrGetter, StrSetter, TYPE_OBJECT, TypeInfo, allow_set_link,
};

use crate::bus::TYPE_SYSTEM_BUS;

/// `TYPE_MACHINE`.
pub const TYPE_MACHINE: &str = "machine";
/// `TYPE_MEMORY_BACKEND`, what the `memory-backend` link points at.
const TYPE_MEMORY_BACKEND: &str = "memory-backend";
/// `TYPE_CONFIDENTIAL_GUEST_SUPPORT`.
const TYPE_CONFIDENTIAL_GUEST_SUPPORT: &str = "confidential-guest-support";

/// Whether the host can mark memory mergeable, which is what `mem-merge` defaults to.
const HOST_MADVISE_MERGEABLE: bool = cfg!(target_os = "linux");
/// Whether the host can leave memory out of a core dump.
const HOST_MADVISE_DONTDUMP: bool = cfg!(target_os = "linux");

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The parts of `MachineClass` that describe a machine type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MachineClassInfo {
    /// The name `-machine` takes, the type name without `-machine`.
    pub name: &'static str,
    pub desc: &'static str,
    pub max_cpus: u32,
    pub default_cpus: u32,
    pub default_ram_size: u64,
    pub default_ram_id: Option<&'static str>,
}

/// The type name of machine `name`, `MACHINE_TYPE_NAME()`.
pub fn machine_type_name(name: &str) -> String {
    format!("{name}-machine")
}

/// The description of the machine type `typename`, if it is one.
pub fn machine_class_info(registry: &Registry, typename: &str) -> Option<Arc<MachineClassInfo>> {
    registry.class_by_name(typename)?.ext::<MachineClassInfo>()
}

#[derive(Debug, Default)]
struct MemSize {
    ram_size: u64,
    maxram_size: u64,
    ram_slots: u64,
}

/// `MachineState`, with the string and bool properties kept by name.
#[derive(Debug, Default)]
struct MachineState {
    strs: Mutex<BTreeMap<&'static str, String>>,
    flags: Mutex<BTreeMap<&'static str, bool>>,
    phandle_start: AtomicI64,
    mem: Mutex<MemSize>,
    smp: Mutex<SMPConfiguration>,
    boot: Mutex<BootConfiguration>,
    smp_cache: Mutex<Vec<SmpCacheProperties>>,
    cgs: Arc<LinkSlot>,
    memdev: Arc<LinkSlot>,
}

fn state(obj: &Object) -> Arc<MachineState> {
    obj.state::<MachineState>().expect("a machine object")
}

fn class_info(obj: &Object) -> Arc<MachineClassInfo> {
    obj.class().ext::<MachineClassInfo>().expect("a concrete machine type")
}

/// The string properties, with their descriptions.
const STRINGS: &[(&str, &str)] = &[
    ("kernel", "Linux kernel image file"),
    ("shim", "shim.efi file"),
    ("initrd", "Linux initial ramdisk file"),
    ("append", "Linux kernel command line"),
    ("dtb", "Linux kernel device tree file"),
    ("dumpdtb", "Dump current dtb to a file and quit"),
    ("dt-compatible", "Overrides the \"compatible\" property of the dt root node"),
    ("firmware", "Firmware image"),
    ("memory-encryption", "Set memory encryption object to use"),
];

/// The bool properties the class has, with their descriptions.
const FLAGS: &[(&str, &str)] = &[
    ("dump-guest-core", "Include guest memory in a core dump"),
    (
        "x-change-vmfd-on-reset",
        "Set on/off to enable/disable generating new accelerator guest handle on guest reset. Default: off (used only for testing/debugging).",
    ),
    ("mem-merge", "Enable/disable memory merge support"),
    ("aux-ram-share", "Use anonymous shared memory for auxiliary guest RAMs"),
    ("usb", "Set on/off to enable/disable usb"),
    ("graphics", "Set on/off to enable/disable graphics emulation"),
    ("suppress-vmdesc", "Set on to disable self-describing migration"),
];

fn add_flag(k: &ObjectClass, name: &'static str, description: &str) {
    let get: BoolGetter =
        Arc::new(move |o: &Object| Ok(lock(&state(o).flags).get(name).copied().unwrap_or(false)));
    let set: BoolSetter = Arc::new(move |o: &Object, v: bool| {
        // machine_set_dump_guest_core() and machine_set_mem_merge() need madvise() flags
        // only Linux has.
        if name == "dump-guest-core" && !v && !HOST_MADVISE_DONTDUMP {
            return Err(Error::generic("Dumping guest memory cannot be disabled on this host"));
        }
        if name == "mem-merge" && v && !HOST_MADVISE_MERGEABLE {
            return Err(Error::generic("Memory merging is not supported on this host"));
        }
        lock(&state(o).flags).insert(name, v);
        Ok(())
    });
    k.property_add_bool(name, Some(get), Some(set)).set_description(Some(description));
}

/// A struct property kept as it was last set. ruvm has no firmware or CPUs yet, so nothing
/// reads these back but QMP.
fn struct_prop<T: Visit + Clone + Default + Send + 'static>(
    name: &str,
    type_: &str,
    description: &str,
    field: fn(&MachineState) -> &Mutex<T>,
) -> Property {
    Property::new(name, type_)
        .getter(move |o, v, name| {
            let mut value = lock(field(&state(o))).clone();
            T::visit(v, Some(name), &mut value)
        })
        .setter(move |o, v, name| {
            let mut value = T::default();
            T::visit(v, Some(name), &mut value)?;
            *lock(field(&state(o))) = value;
            Ok(())
        })
        .description(description)
}

/// `machine_set_mem()`.
fn set_mem(obj: &Object, mut mem: MemorySizeConfiguration) -> Result<()> {
    let size = mem.size.unwrap_or(class_info(obj).default_ram_size).next_multiple_of(8192);
    mem.size = Some(size);
    let st = state(obj);
    let mut m = lock(&st.mem);
    match mem.max_size {
        Some(max) => {
            if max < size {
                return Err(Error::generic(format!(
                    "invalid value of maxmem: maximum memory size (0x{max:x}) must be at least the initial memory size (0x{size:x})"
                )));
            }
            if mem.slots.is_some_and(|s| s != 0) && max == size {
                return Err(Error::generic(format!(
                    "invalid value of maxmem: memory slots were specified but maximum memory size (0x{max:x}) is equal to the initial memory size (0x{size:x})"
                )));
            }
            m.maxram_size = max;
        }
        None => {
            if mem.slots.is_some() {
                return Err(Error::generic("slots specified but no max-size"));
            }
            m.maxram_size = size;
        }
    }
    m.ram_size = size;
    m.ram_slots = mem.slots.unwrap_or(0);
    Ok(())
}

/// `machine_class_init()`.
fn machine_class_init(k: &Arc<ObjectClass>) {
    for &(name, description) in STRINGS {
        let get: StrGetter = Arc::new(move |o: &Object| {
            Ok(lock(&state(o).strs).get(name).cloned().unwrap_or_default())
        });
        let set: StrSetter = Arc::new(move |o: &Object, v: &str| {
            lock(&state(o).strs).insert(name, v.to_string());
            Ok(())
        });
        k.property_add_str(name, Some(get), Some(set)).set_description(Some(description));
    }
    for &(name, description) in FLAGS {
        // aux-ram-share is only there with CONFIG_POSIX.
        if name != "aux-ram-share" || cfg!(unix) {
            add_flag(k, name, description);
        }
    }

    k.property_add(struct_prop("boot", "BootConfiguration", "Boot configuration", |s| &s.boot));
    k.property_add(struct_prop("smp", "SMPConfiguration", "CPU topology", |s| &s.smp));
    k.property_add(
        Property::new("smp-cache", "SmpCachePropertiesWrapper")
            .getter(|o, v, name| {
                let mut caches = lock(&state(o).smp_cache).clone();
                v.visit_list(Some(name), &mut caches, |v, c| SmpCacheProperties::visit(v, None, c))
            })
            .setter(|o, v, name| {
                let mut caches = Vec::new();
                v.visit_list(Some(name), &mut caches, |v, c| {
                    SmpCacheProperties::visit(v, None, c)
                })?;
                *lock(&state(o).smp_cache) = caches;
                Ok(())
            })
            .description("Cache properties list for SMP machine"),
    );
    k.property_add(
        Property::new("phandle-start", "int")
            .getter(|o, v, name| {
                let mut n = state(o).phandle_start.load(Ordering::Acquire);
                v.type_int64(Some(name), &mut n)
            })
            .setter(|o, v, name| {
                let mut n = 0;
                v.type_int64(Some(name), &mut n)?;
                state(o).phandle_start.store(n, Ordering::Release);
                Ok(())
            })
            .description("The first phandle ID we may generate dynamically"),
    );
    k.property_add_link(
        "confidential-guest-support",
        TYPE_CONFIDENTIAL_GUEST_SUPPORT,
        |o| state(o).cgs.clone(),
        Some(Arc::new(allow_set_link)),
        LinkFlags::STRONG,
    )
    .set_description(Some("Set confidential guest scheme to support"));
    k.property_add_link(
        "memory-backend",
        TYPE_MEMORY_BACKEND,
        |o| state(o).memdev.clone(),
        Some(Arc::new(allow_set_link)),
        LinkFlags::STRONG,
    )
    .set_description(Some("Set RAM backendValid value is ID of hostmem based backend"));
    k.property_add(
        Property::new("memory", "MemorySizeConfiguration")
            .getter(|o, v, name| {
                let st = state(o);
                let m = lock(&st.mem);
                let slots = m.ram_slots != 0;
                let mut mem = MemorySizeConfiguration {
                    size: Some(m.ram_size),
                    max_size: slots.then_some(m.maxram_size),
                    slots: slots.then_some(m.ram_slots),
                };
                drop(m);
                MemorySizeConfiguration::visit(v, Some(name), &mut mem)
            })
            .setter(|o, v, name| {
                let mut mem = MemorySizeConfiguration::default();
                MemorySizeConfiguration::visit(v, Some(name), &mut mem)?;
                set_mem(o, mem)
            })
            .description("Memory size configuration"),
    );
}

/// `machine_initfn()`.
fn machine_init(obj: &Object) {
    let info = class_info(obj);
    let st = state(obj);
    {
        let mut flags = lock(&st.flags);
        flags.insert("dump-guest-core", true);
        flags.insert("mem-merge", HOST_MADVISE_MERGEABLE);
        flags.insert("graphics", true);
    }
    *lock(&st.mem) = MemSize {
        ram_size: info.default_ram_size,
        maxram_size: info.default_ram_size,
        ram_slots: 0,
    };
    let cpus = i64::from(info.default_cpus);
    *lock(&st.smp) = SMPConfiguration {
        cpus: Some(cpus),
        drawers: Some(1),
        books: Some(1),
        sockets: Some(1),
        dies: Some(1),
        clusters: Some(1),
        modules: Some(1),
        cores: Some(1),
        threads: Some(1),
        maxcpus: Some(cpus),
    };
    *lock(&st.smp_cache) = CacheLevelAndType::ALL
        .iter()
        .map(|&cache| SmpCacheProperties { cache, topology: CpuTopologyLevel::Default })
        .collect();

    // The ACPI SPCR switch is an instance property.
    let spcr = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let (g, s) = (spcr.clone(), spcr);
    let get: BoolGetter = Arc::new(move |_: &Object| Ok(g.load(Ordering::Acquire)));
    let set: BoolSetter = Arc::new(move |_: &Object, v: bool| {
        s.store(v, Ordering::Release);
        Ok(())
    });
    obj.property_add_bool("spcr", Some(get), Some(set)).set_description(Some(
        "Set on/off to enable/disable ACPI Serial Port Console Redirection Table (spcr)",
    ));
}

/// The machine types ruvm has. Only `none` for now.
pub const MACHINES: &[MachineClassInfo] = &[MachineClassInfo {
    name: "none",
    desc: "empty machine",
    max_cpus: 1,
    default_cpus: 1,
    default_ram_size: 0,
    default_ram_id: Some("ram"),
}];

pub(crate) fn register_types(registry: &Registry) {
    let machine = TypeInfo::new(TYPE_MACHINE)
        .parent(TYPE_OBJECT)
        .abstract_()
        .instance_state(MachineState::default)
        .instance_init(machine_init)
        .class_init(machine_class_init);
    registry.register(machine);
    for info in MACHINES {
        register_machine_type(registry, info.clone());
    }
}

/// Registers the machine type `info` describes, `MACHINE_TYPE_NAME(name)` under
/// `TYPE_MACHINE`. The boards built outside QOM use it for their `/machine` object.
pub fn register_machine_type(registry: &Registry, info: MachineClassInfo) {
    registry.register(
        TypeInfo::new(machine_type_name(info.name))
            .parent(TYPE_MACHINE)
            .class_init(move |k| k.set_ext(info.clone())),
    );
}

/// `unassigned_io_ops`: I/O ports nobody claimed read as all ones and ignore writes.
#[derive(Debug)]
struct UnassignedIo;

impl MmioOps for UnassignedIo {
    fn read(&self, _cx: &AccessCtx, _offset: u64, _size: AccessSize) -> MemResult<u64> {
        Ok(u64::MAX)
    }

    fn write(
        &self,
        _cx: &AccessCtx,
        _offset: u64,
        _size: AccessSize,
        _value: u64,
    ) -> MemResult<()> {
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 4).allow_unaligned()
    }
}

/// The machine object and what `memory_map_init()` sets up along with it.
#[derive(Debug)]
pub struct Machine {
    pub object: Object,
    pub info: Arc<MachineClassInfo>,
    /// `get_system_memory()`.
    pub system_memory: RegionId,
    /// `get_system_io()`.
    pub system_io: RegionId,
    /// `address_space_memory`.
    pub address_space_memory: Arc<AddressSpace>,
    /// `address_space_io`.
    pub address_space_io: Arc<AddressSpace>,
}

impl Machine {
    /// `ram_size`.
    pub fn ram_size(&self) -> u64 {
        lock(&state(&self.object).mem).ram_size
    }

    /// Sets `ram_size` straight, as `qemu_resolve_machine_memdev()` does with the size of the
    /// backend.
    pub fn set_ram_size(&self, size: u64) {
        lock(&state(&self.object).mem).ram_size = size;
    }

    /// `machine_run_board_init()` and `machine_none_init()`: the RAM backend, made from
    /// `memory.size` when none was named, goes into the guest at address zero.
    pub fn run_board_init(&self, objects: &Arc<RegionObjects>) -> Result<()> {
        let st = state(&self.object);
        let ram_size = lock(&st.mem).ram_size;
        let mut memdev = self.object.property_get_link("memory-backend")?;
        if let Some(backend) = &memdev {
            if backend.property_get_uint("size")? != ram_size {
                return Err(Error::generic(
                    "Machine memory size does not match the size of the memory backend",
                ));
            }
        } else if let (Some(id), true) = (self.info.default_ram_id, ram_size != 0) {
            let registry = self.object.registry();
            if registry.objects_root().property_find(id).is_some() {
                return Err(Error::generic(format!(
                    "object's id '{id}' is reserved for the default RAM backend, it can't be used for any other purposes"
                ))
                .hint(format!(
                    "Change the object's 'id' to something else or disable automatic creation of the default RAM backend by setting 'memory-backend={id}' with '-machine'.\n"
                )));
            }
            // create_default_memdev()
            let backend = registry.object_new(ruvm_hostmem::TYPE_MEMORY_BACKEND_RAM)?;
            backend.property_set_uint("size", ram_size)?;
            registry.objects_root().property_try_add_child(id, &backend)?;
            backend.property_set_bool("x-use-canonical-path-for-ramblock-id", false)?;
            ruvm_qom::user_creatable_complete(&backend)?;
            self.object.property_set_link("memory-backend", Some(&backend))?;
            memdev = Some(backend);
        }

        if let Some(backend) = &memdev {
            // machine_consume_memdev()
            if ruvm_hostmem::is_mapped(backend) {
                let id = backend.canonical_path_component().unwrap_or_default();
                return Err(Error::generic(format!(
                    "memory backend {id} can't be used multiple times."
                )));
            }
            ruvm_hostmem::set_mapped(backend, true);
            let ram = ruvm_hostmem::backend_region(backend)
                .ok_or_else(|| Error::generic("memory backend has no memory"))?;
            objects.memory().add_subregion(self.system_memory, 0, ram).map_err(mem_error)?;
        }

        if lock(&st.strs).contains_key("kernel") {
            return Err(Error::generic(
                "The -kernel parameter is not supported (use the generic 'loader' device instead).",
            ));
        }
        Ok(())
    }

    /// `machine_get_container()`.
    pub fn container(&self, name: &str) -> Option<Object> {
        self.object.resolve_path_component(name)
    }
}

fn mem_error(e: ruvm_mem::MemError) -> Error {
    Error::generic(e.to_string())
}

/// `qemu_create_machine()` up to `machine_memory_init()`: the machine object at `/machine`
/// with its containers, the main system bus, and the system memory and I/O regions.
pub fn create_machine(
    registry: &Registry,
    typename: &str,
    objects: &Arc<RegionObjects>,
) -> Result<Machine> {
    let info = machine_class_info(registry, typename)
        .ok_or_else(|| Error::generic(format!("unsupported machine type: \"{typename}\"")))?;
    let object = registry.object_new(typename)?;
    registry.root().property_try_add_child("machine", &object)?;
    let unattached = object.add_new_container("unattached");
    object.add_new_container("peripheral");
    object.add_new_container("peripheral-anon");
    let sysbus = registry.object_new(TYPE_SYSTEM_BUS)?;
    unattached.property_try_add_child("sysbus", &sysbus)?;

    // memory_map_init(). A size of UINT64_MAX means all of the 64-bit space.
    let mem = objects.memory();
    let system_memory = mem.new_container("system", 1 << 64).map_err(mem_error)?;
    objects.add(&unattached, "system", system_memory)?;
    let address_space_memory =
        mem.address_space_init(system_memory, "memory").map_err(mem_error)?;
    let system_io = mem.new_io("io", 65536, Arc::new(UnassignedIo)).map_err(mem_error)?;
    objects.add(&unattached, "io", system_io)?;
    let address_space_io = mem.address_space_init(system_io, "I/O").map_err(mem_error)?;

    Ok(Machine { object, info, system_memory, system_io, address_space_memory, address_space_io })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_mem::MemorySystem;

    fn machine() -> (Registry, Machine) {
        let registry = Registry::new();
        crate::register_types(&registry);
        ruvm_hostmem::region::register_types(&registry);
        let objects = RegionObjects::new(Arc::new(MemorySystem::new()));
        let m = create_machine(&registry, "none-machine", &objects).unwrap();
        (registry, m)
    }

    #[test]
    fn the_none_machine_tree() {
        let (registry, m) = machine();
        let get = |path: &str, prop: &str| {
            let (obj, _) = registry.resolve_path(path);
            obj.unwrap().property_get_qobject(prop).unwrap().to_json()
        };
        assert_eq!(get("/machine", "type"), "\"none-machine\"");
        assert_eq!(get("/machine", "memory"), "{\"size\": 0}");
        assert_eq!(get("/machine", "boot"), "{}");
        assert_eq!(get("/machine", "kernel"), "\"\"");
        assert_eq!(get("/machine", "graphics"), "true");
        assert_eq!(get("/machine", "spcr"), "true");
        assert_eq!(get("/machine/unattached/sysbus", "type"), "\"System\"");
        assert_eq!(get("/machine/unattached/sysbus", "realized"), "false");
        assert_eq!(get("/machine/unattached/io[0]", "size"), "65536");
        assert_eq!(get("/machine/unattached/system[0]", "size"), "18446744073709551615");
        assert!(m.container("peripheral-anon").is_some());
        let cache = get("/machine", "smp-cache");
        assert!(cache.starts_with("[{\"topology\": \"default\", \"cache\": \"l1d\"}"), "{cache}");
    }

    #[test]
    fn memory_sizes() {
        let (_, m) = machine();
        let o = &m.object;
        let set = |json: &str| {
            let v = ruvm_qapi::json::from_str(json).unwrap();
            o.property_set_qobject("memory", v)
        };
        set("{\"size\": 1000}").unwrap();
        assert_eq!(o.property_get_qobject("memory").unwrap().to_json(), "{\"size\": 8192}");
        let e = set("{\"size\": 16384, \"max-size\": 8192}").unwrap_err();
        assert_eq!(
            e.message(),
            "invalid value of maxmem: maximum memory size (0x2000) must be at least the initial memory size (0x4000)"
        );
        let e = set("{\"slots\": 2}").unwrap_err();
        assert_eq!(e.message(), "slots specified but no max-size");
    }
}

#[cfg(test)]
mod ram_tests {
    use super::*;
    use ruvm_mem::{MemTxAttrs, MemorySystem};

    #[test]
    fn default_ram_is_at_zero() {
        let registry = Registry::new();
        let objects = RegionObjects::new(Arc::new(MemorySystem::new()));
        crate::register_types(&registry);
        ruvm_hostmem::region::register_types(&registry);
        ruvm_hostmem::register_types(&registry, &objects);
        let m = create_machine(&registry, "none-machine", &objects).unwrap();
        let v = ruvm_qapi::json::from_str("{\"size\": 65536}").unwrap();
        m.object.property_set_qobject("memory", v).unwrap();
        m.run_board_init(&objects).unwrap();
        let r = m.address_space_memory.write_u32(0x100, MemTxAttrs::UNSPECIFIED, 0x1234_5678);
        assert!(r.is_ok());
        let (v, r) = m.address_space_memory.read_u32(0x100, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok());
        assert_eq!(v, 0x1234_5678);
        let ram = registry.resolve_path("/objects/ram").0.unwrap();
        assert!(ruvm_hostmem::is_mapped(&ram));
    }
}
