// SPDX-License-Identifier: GPL-2.0-or-later

//! The type table: registration, lazy class creation and class enumeration.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock, Weak};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::ghash::GHashTable;

use crate::class::ObjectClass;
use crate::compat::CompatProps;
use crate::object::Object;
use crate::{TYPE_CONTAINER, TYPE_INTERFACE, TYPE_OBJECT, TYPE_USER_CREATABLE};

/// A class hook: `class_init` or `class_base_init`. Class data, the C `class_data` pointer, is
/// whatever the closure captures.
pub type ClassHook = Arc<dyn Fn(&Arc<ObjectClass>) + Send + Sync>;
/// An instance hook: `instance_init`, `instance_post_init` or `instance_finalize`.
pub type InstanceHook = Arc<dyn Fn(&Object) + Send + Sync>;
pub(crate) type StateFn = Arc<dyn Fn() -> Arc<dyn Any + Send + Sync> + Send + Sync>;

/// `TypeInfo`: what a type registers.
#[derive(Clone, Default)]
pub struct TypeInfo {
    pub(crate) name: String,
    pub(crate) parent: Option<String>,
    pub(crate) abstract_: bool,
    pub(crate) interfaces: Vec<String>,
    pub(crate) class_init: Option<ClassHook>,
    pub(crate) class_base_init: Option<ClassHook>,
    pub(crate) instance_init: Option<InstanceHook>,
    pub(crate) instance_post_init: Option<InstanceHook>,
    pub(crate) instance_finalize: Option<InstanceHook>,
    pub(crate) instance_state: Option<StateFn>,
}

impl fmt::Debug for TypeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TypeInfo")
            .field("name", &self.name)
            .field("parent", &self.parent)
            .field("abstract", &self.abstract_)
            .field("interfaces", &self.interfaces)
            .finish_non_exhaustive()
    }
}

impl TypeInfo {
    pub fn new(name: impl Into<String>) -> Self {
        TypeInfo { name: name.into(), ..Default::default() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn abstract_(mut self) -> Self {
        self.abstract_ = true;
        self
    }

    /// Adds an interface, an entry of the C `interfaces` array.
    pub fn interface(mut self, name: impl Into<String>) -> Self {
        self.interfaces.push(name.into());
        self
    }

    pub fn class_init(mut self, f: impl Fn(&Arc<ObjectClass>) + Send + Sync + 'static) -> Self {
        self.class_init = Some(Arc::new(f));
        self
    }

    pub fn class_base_init(
        mut self,
        f: impl Fn(&Arc<ObjectClass>) + Send + Sync + 'static,
    ) -> Self {
        self.class_base_init = Some(Arc::new(f));
        self
    }

    pub fn instance_init(mut self, f: impl Fn(&Object) + Send + Sync + 'static) -> Self {
        self.instance_init = Some(Arc::new(f));
        self
    }

    pub fn instance_post_init(mut self, f: impl Fn(&Object) + Send + Sync + 'static) -> Self {
        self.instance_post_init = Some(Arc::new(f));
        self
    }

    pub fn instance_finalize(mut self, f: impl Fn(&Object) + Send + Sync + 'static) -> Self {
        self.instance_finalize = Some(Arc::new(f));
        self
    }

    /// The state this type adds to each instance, what the C struct adds to its parent struct.
    /// It is created before any `instance_init` runs and fetched with [`Object::state`].
    pub fn instance_state<T: Any + Send + Sync>(
        mut self,
        f: impl Fn() -> T + Send + Sync + 'static,
    ) -> Self {
        self.instance_state = Some(Arc::new(move || Arc::new(f()) as Arc<dyn Any + Send + Sync>));
        self
    }

    fn has_instance_hooks(&self) -> bool {
        self.instance_init.is_some()
            || self.instance_post_init.is_some()
            || self.instance_finalize.is_some()
            || self.instance_state.is_some()
    }
}

pub(crate) struct TypeImpl {
    pub(crate) info: TypeInfo,
    class: OnceLock<Arc<ObjectClass>>,
}

/// `type_name_is_valid()`.
fn type_name_is_valid(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() > 1
        && b[0].is_ascii_alphanumeric()
        && b[0] != b'0'
        && b.iter().all(|&c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.'))
}

pub(crate) struct RegistryInner {
    types: RwLock<GHashTable<Arc<TypeImpl>>>,
    root: OnceLock<Object>,
    internal_root: OnceLock<Object>,
    pub(crate) compat: CompatProps,
}

impl Drop for RegistryInner {
    fn drop(&mut self) {
        // Tear the tree down while the classes can still reach the registry.
        drop(self.root.take());
        drop(self.internal_root.take());
    }
}

/// The type table and the composition tree that hangs off its root.
#[derive(Clone)]
pub struct Registry(pub(crate) Arc<RegistryInner>);

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let types = self.0.types.read().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("Registry").field("types", &types.len()).finish_non_exhaustive()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}

impl Registry {
    /// A registry with the types QOM itself defines: `interface`, `object`, `container` and
    /// `user-creatable`.
    pub fn new() -> Self {
        let r = Registry(Arc::new(RegistryInner {
            types: RwLock::new(GHashTable::new()),
            root: OnceLock::new(),
            internal_root: OnceLock::new(),
            compat: CompatProps::default(),
        }));
        r.register(TypeInfo::new(TYPE_INTERFACE).abstract_());
        r.register(TypeInfo::new(TYPE_OBJECT).abstract_().class_init(|oc| {
            oc.property_add_str(
                "type",
                Some(Arc::new(|obj: &Object| Ok(obj.typename().to_string()))),
                None,
            );
        }));
        r.register(TypeInfo::new(TYPE_CONTAINER).parent(TYPE_OBJECT));
        r.register(TypeInfo::new(TYPE_USER_CREATABLE).parent(TYPE_INTERFACE));
        r
    }

    /// The registry the emulator uses.
    pub fn global() -> &'static Registry {
        static GLOBAL: OnceLock<Registry> = OnceLock::new();
        GLOBAL.get_or_init(Registry::new)
    }

    pub(crate) fn from_weak(w: &Weak<RegistryInner>) -> Option<Registry> {
        w.upgrade().map(Registry)
    }

    pub(crate) fn downgrade(&self) -> Weak<RegistryInner> {
        Arc::downgrade(&self.0)
    }

    /// `type_register_static()`. Like QEMU, a bad name or a second registration of a name is a
    /// bug in the caller and panics.
    pub fn register(&self, info: TypeInfo) {
        if !type_name_is_valid(&info.name) {
            panic!("Registering '{}' with illegal type name", info.name);
        }
        let mut types = self.0.types.write().unwrap_or_else(|e| e.into_inner());
        if types.contains_key(&info.name) {
            panic!("Registering `{}' which already exists", info.name);
        }
        let name = info.name.clone();
        types.insert(name, Arc::new(TypeImpl { info, class: OnceLock::new() }));
    }

    /// `type_register_static_array()`.
    pub fn register_all(&self, infos: impl IntoIterator<Item = TypeInfo>) {
        for info in infos {
            self.register(info);
        }
    }

    pub(crate) fn lookup(&self, name: &str) -> Option<Arc<TypeImpl>> {
        let types = self.0.types.read().unwrap_or_else(|e| e.into_inner());
        types.get(name).cloned()
    }

    /// Whether a type with this name is registered, without creating its class.
    pub fn type_exists(&self, name: &str) -> bool {
        self.lookup(name).is_some()
    }

    /// `type_is_ancestor()` on registered types, walking parent names without creating classes.
    pub(crate) fn type_is_ancestor(&self, name: &str, target: &str) -> bool {
        let mut cur = self.lookup(name);
        while let Some(t) = cur {
            if t.info.name == target {
                return true;
            }
            cur = t.info.parent.as_deref().and_then(|p| self.lookup(p));
        }
        false
    }

    /// `object_class_by_name()`: the class of a registered type, created on first use.
    pub fn class_by_name(&self, name: &str) -> Option<Arc<ObjectClass>> {
        let ti = self.lookup(name)?;
        Some(self.type_initialize(&ti))
    }

    /// `type_get_or_load_by_name()` with its error.
    pub(crate) fn class_by_name_err(&self, name: &str) -> Result<Arc<ObjectClass>> {
        self.class_by_name(name).ok_or_else(|| Error::generic(format!("unknown type '{name}'")))
    }

    pub(crate) fn type_initialize(&self, ti: &Arc<TypeImpl>) -> Arc<ObjectClass> {
        if let Some(c) = ti.class.get() {
            return c.clone();
        }
        let parent = ti.info.parent.as_deref().map(|p| match self.lookup(p) {
            Some(pt) => self.type_initialize(&pt),
            None => panic!("Type '{}' is missing its parent '{}'", ti.info.name, p),
        });
        let class = self.build_class(ti.info.clone(), parent);
        ti.class.get_or_init(|| class).clone()
    }

    /// `type_initialize()` for a type whose parent class exists already.
    fn build_class(
        &self,
        mut info: TypeInfo,
        parent: Option<Arc<ObjectClass>>,
    ) -> Arc<ObjectClass> {
        let is_interface =
            info.name == TYPE_INTERFACE || parent.as_ref().is_some_and(|p| p.is_interface());
        if is_interface {
            // Interface types have no instance size, which makes them abstract.
            info.abstract_ = true;
            assert!(
                !info.has_instance_hooks() && info.interfaces.is_empty(),
                "interface type '{}' cannot have instance hooks or interfaces",
                info.name
            );
        }
        let mut interfaces: Vec<Arc<ObjectClass>> = Vec::new();
        if let Some(parent) = &parent {
            for iface in parent.interfaces() {
                let itype =
                    iface.interface_type().expect("interface class has its type").to_string();
                interfaces.push(self.interface_class(&info.name, &itype, iface.clone()));
            }
            for iname in &info.interfaces {
                let Some(it) = self.lookup(iname) else {
                    panic!("missing interface '{}' for object '{}'", iname, parent.name());
                };
                if interfaces.iter().any(|c| c.is_descendant_of(&it.info.name)) {
                    continue;
                }
                let iclass = self.type_initialize(&it);
                interfaces.push(self.interface_class(&info.name, &it.info.name, iclass));
            }
        }
        let class =
            Arc::new(ObjectClass::new(info, parent, is_interface, interfaces, self.downgrade()));
        class.run_class_hooks();
        class
    }

    /// `type_initialize_interface()`: the class named `TYPE::IFACE` that holds a type's view of
    /// an interface. Its parent is the interface class or, for inherited interfaces, the parent
    /// type's own view of it, so overrides made by the parent carry over.
    fn interface_class(
        &self,
        type_name: &str,
        interface_type: &str,
        parent: Arc<ObjectClass>,
    ) -> Arc<ObjectClass> {
        let info = TypeInfo::new(format!("{type_name}::{interface_type}")).abstract_();
        let class = self.build_class(info, Some(parent));
        class.set_interface_type(interface_type);
        class
    }

    /// `object_class_foreach()`: every class in type table order, which is GLib hash order.
    pub fn class_foreach(
        &self,
        implements: Option<&str>,
        include_abstract: bool,
        mut f: impl FnMut(&Arc<ObjectClass>),
    ) {
        let all: Vec<Arc<TypeImpl>> = {
            let types = self.0.types.read().unwrap_or_else(|e| e.into_inner());
            types.values().cloned().collect()
        };
        for ti in all {
            let k = self.type_initialize(&ti);
            if !include_abstract && k.is_abstract() {
                continue;
            }
            if let Some(i) = implements {
                if k.dynamic_cast(i).is_none() {
                    continue;
                }
            }
            f(&k);
        }
    }

    /// `object_class_get_list()`. The C code prepends, so the list is in reverse table order.
    pub fn class_get_list(
        &self,
        implements: Option<&str>,
        include_abstract: bool,
    ) -> Vec<Arc<ObjectClass>> {
        let mut v = Vec::new();
        self.class_foreach(implements, include_abstract, |k| v.push(k.clone()));
        v.reverse();
        v
    }

    /// `object_class_get_list_sorted()`, ordered by `g_ascii_strcasecmp()` of the names.
    pub fn class_get_list_sorted(
        &self,
        implements: Option<&str>,
        include_abstract: bool,
    ) -> Vec<Arc<ObjectClass>> {
        let mut v = self.class_get_list(implements, include_abstract);
        v.sort_by(|a, b| {
            let a = a.name().bytes().map(|c| c.to_ascii_lowercase());
            let b = b.name().bytes().map(|c| c.to_ascii_lowercase());
            a.cmp(b)
        });
        v
    }

    /// `object_new()`.
    pub fn object_new(&self, typename: &str) -> Result<Object> {
        let class = self.class_by_name_err(typename)?;
        Object::new_with_class(&class)
    }

    /// `object_get_root()`: the root container, with `audiodevs`, `chardevs`, `objects` and
    /// `backend` under it.
    pub fn root(&self) -> Object {
        self.0
            .root
            .get_or_init(|| {
                let root = self.object_new(TYPE_CONTAINER).expect("container type exists");
                for name in ["audiodevs", "chardevs", "objects", "backend"] {
                    root.add_new_container(name);
                }
                root
            })
            .clone()
    }

    pub(crate) fn is_root(&self, obj: &Object) -> bool {
        self.0.root.get().is_some_and(|r| r.ptr_eq(obj))
    }

    /// `object_get_container()`.
    pub fn container(&self, name: &str) -> Object {
        let c = self.root().resolve_path_component(name).expect("root container exists");
        assert!(c.dynamic_cast(TYPE_CONTAINER).is_some());
        c
    }

    /// `object_get_objects_root()`.
    pub fn objects_root(&self) -> Object {
        self.container("objects")
    }

    /// `object_get_internal_root()`: a container outside the tree for objects that must not
    /// show up in `qom-list`.
    pub fn internal_root(&self) -> Object {
        self.0
            .internal_root
            .get_or_init(|| self.object_new(TYPE_CONTAINER).expect("container type exists"))
            .clone()
    }

    /// The compat and global properties applied to new objects.
    pub fn compat_props(&self) -> &CompatProps {
        &self.0.compat
    }

    /// `object_resolve_path_type()`. The second value is QEMU's `ambiguous` flag.
    pub fn resolve_path_type(&self, path: &str, typename: &str) -> (Option<Object>, bool) {
        let parts: Vec<&str> = if path.is_empty() { Vec::new() } else { path.split('/').collect() };
        let root = self.root();
        if parts.first().is_none_or(|p| !p.is_empty()) {
            let mut ambiguous = false;
            let obj = root.resolve_partial_path(&parts, typename, &mut ambiguous);
            (obj, ambiguous)
        } else {
            (root.resolve_abs_path(&parts[1..], typename), false)
        }
    }

    /// `object_resolve_path()`.
    pub fn resolve_path(&self, path: &str) -> (Option<Object>, bool) {
        self.resolve_path_type(path, TYPE_OBJECT)
    }

    /// `object_resolve_type_unambiguous()`.
    pub fn resolve_type_unambiguous(&self, typename: &str) -> Result<Object> {
        match self.resolve_path_type("", typename) {
            (_, true) => Err(Error::generic(format!("More than one object of type {typename}"))),
            (None, false) => Err(Error::generic(format!("No object found of type {typename}"))),
            (Some(o), false) => Ok(o),
        }
    }

    /// `object_resolve_and_typecheck()`, how link properties turn a path into an object.
    pub fn resolve_and_typecheck(
        &self,
        path: &str,
        name: &str,
        target_type: &str,
    ) -> Result<Object> {
        match self.resolve_path_type(path, target_type) {
            (_, true) => {
                Err(Error::generic(format!("Path '{path}' does not uniquely identify an object")))
            }
            (Some(o), false) => Ok(o),
            (None, false) => match self.resolve_path(path) {
                (Some(_), _) | (None, true) => Err(Error::generic(format!(
                    "Invalid parameter type for '{name}', expected: {target_type}"
                ))),
                (None, false) => Err(Error::new(
                    ErrorClass::DeviceNotFound,
                    format!("Device '{path}' not found"),
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::type_name_is_valid;

    #[test]
    fn type_names() {
        assert!(type_name_is_valid("pc-q35-11.1"));
        assert!(type_name_is_valid("x86_64-cpu"));
        assert!(!type_name_is_valid("a"));
        assert!(!type_name_is_valid("0abc"));
        assert!(!type_name_is_valid("-abc"));
        assert!(!type_name_is_valid("foo bar"));
        assert!(!type_name_is_valid("foo::bar"));
    }
}
