// SPDX-License-Identifier: GPL-2.0-or-later

//! `ObjectClass`: what every instance of a type shares.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};

use ruvm_base::{Error, Result};
use ruvm_qapi::ghash::GHashTable;

use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64};

use ruvm_qapi::visit::QEnumLookup;

use crate::TYPE_INTERFACE;
use crate::link::{LinkCheck, LinkFlags, LinkSlot, new_link};
use crate::lock;
use crate::object::Object;
use crate::property::{
    BoolGetter, BoolSetter, EnumGetter, EnumSetter, PropFlags, Property, StrGetter, StrSetter,
    TmGetter,
};
use crate::types::{Registry, RegistryInner, TypeInfo};

pub(crate) type UnparentFn = Arc<dyn Fn(&Object) + Send + Sync>;

/// A class. It holds the type's hooks, its class properties, the interface classes it
/// implements, and typed extension values that stand in for the fields a C class struct adds to
/// its parent's.
pub struct ObjectClass {
    pub(crate) info: TypeInfo,
    parent: Option<Arc<ObjectClass>>,
    is_interface: bool,
    interfaces: Vec<Arc<ObjectClass>>,
    interface_type: OnceLock<String>,
    pub(crate) props: Mutex<GHashTable<Arc<Property>>>,
    ext: RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>,
    unparent: RwLock<Option<UnparentFn>>,
    registry: Weak<RegistryInner>,
}

impl fmt::Debug for ObjectClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObjectClass")
            .field("name", &self.info.name)
            .field("parent", &self.parent.as_ref().map(|p| p.name().to_string()))
            .field("abstract", &self.info.abstract_)
            .finish_non_exhaustive()
    }
}

impl ObjectClass {
    pub(crate) fn new(
        info: TypeInfo,
        parent: Option<Arc<ObjectClass>>,
        is_interface: bool,
        interfaces: Vec<Arc<ObjectClass>>,
        registry: Weak<RegistryInner>,
    ) -> Self {
        // The memcpy of the parent class: extension values and the unparent hook carry over,
        // the property table and the interface list do not.
        let ext = parent
            .as_ref()
            .map(|p| p.ext.read().unwrap_or_else(|e| e.into_inner()).clone())
            .unwrap_or_default();
        let unparent = parent
            .as_ref()
            .and_then(|p| p.unparent.read().unwrap_or_else(|e| e.into_inner()).clone());
        ObjectClass {
            info,
            parent,
            is_interface,
            interfaces,
            interface_type: OnceLock::new(),
            props: Mutex::new(GHashTable::new()),
            ext: RwLock::new(ext),
            unparent: RwLock::new(unparent),
            registry,
        }
    }

    /// Runs the `class_base_init` hook of every ancestor, nearest first, then `class_init`.
    pub(crate) fn run_class_hooks(self: &Arc<Self>) {
        let mut p = self.parent.clone();
        while let Some(pc) = p {
            if let Some(h) = &pc.info.class_base_init {
                h(self);
            }
            p = pc.parent.clone();
        }
        if let Some(h) = &self.info.class_init {
            h(self);
        }
    }

    pub(crate) fn set_interface_type(&self, name: &str) {
        let _ = self.interface_type.set(name.to_string());
    }

    /// For a `TYPE::IFACE` class, the interface it stands for.
    pub fn interface_type(&self) -> Option<&str> {
        self.interface_type.get().map(String::as_str)
    }

    /// `object_class_get_name()`.
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// `object_class_is_abstract()`.
    pub fn is_abstract(&self) -> bool {
        self.info.abstract_
    }

    pub(crate) fn is_interface(&self) -> bool {
        self.is_interface
    }

    /// `object_class_get_parent()`.
    pub fn parent(&self) -> Option<&Arc<ObjectClass>> {
        self.parent.as_ref()
    }

    /// The interface classes this class implements, `klass->interfaces`.
    pub fn interfaces(&self) -> &[Arc<ObjectClass>] {
        &self.interfaces
    }

    pub fn registry(&self) -> Option<Registry> {
        Registry::from_weak(&self.registry)
    }

    /// `type_is_ancestor()`: whether this class is `name` or descends from it.
    pub(crate) fn is_descendant_of(&self, name: &str) -> bool {
        let mut c = Some(self);
        while let Some(k) = c {
            if k.info.name == name {
                return true;
            }
            c = k.parent.as_deref();
        }
        false
    }

    /// `object_class_dynamic_cast()`. Casting to an interface gives the interface class, and a
    /// class that implements an interface twice over cannot be cast to it.
    pub fn dynamic_cast(self: &Arc<Self>, typename: &str) -> Option<Arc<ObjectClass>> {
        if self.info.name == typename {
            return Some(self.clone());
        }
        let registry = self.registry()?;
        if !registry.type_exists(typename) {
            return None;
        }
        if !self.interfaces.is_empty() && registry.type_is_ancestor(typename, TYPE_INTERFACE) {
            let mut found = self.interfaces.iter().filter(|c| c.is_descendant_of(typename));
            let first = found.next()?;
            if found.next().is_some() {
                return None;
            }
            Some(first.clone())
        } else if self.is_descendant_of(typename) {
            Some(self.clone())
        } else {
            None
        }
    }

    /// The class of the interface `name` as this class implements it, where the class sets
    /// the interface's hooks.
    pub fn interface(self: &Arc<Self>, name: &str) -> Option<Arc<ObjectClass>> {
        self.dynamic_cast(name).filter(|c| c.interface_type().is_some())
    }

    /// The extension value of type `T`, inherited from the parent class unless this class
    /// replaced it.
    pub fn ext<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        let ext = self.ext.read().unwrap_or_else(|e| e.into_inner());
        ext.get(&TypeId::of::<T>()).cloned().and_then(|a| a.downcast::<T>().ok())
    }

    pub fn set_ext<T: Any + Send + Sync>(&self, value: T) {
        let mut ext = self.ext.write().unwrap_or_else(|e| e.into_inner());
        ext.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// Changes the extension value of type `T` in place, starting from the inherited value or
    /// the default, the way a C `class_init` assigns into its class struct.
    pub fn update_ext<T: Any + Send + Sync + Clone + Default>(&self, f: impl FnOnce(&mut T)) {
        let mut v = self.ext::<T>().map(|a| (*a).clone()).unwrap_or_default();
        f(&mut v);
        self.set_ext(v);
    }

    /// `ObjectClass::unparent`, the hook that runs when an object is removed from its parent.
    pub fn set_unparent(&self, f: impl Fn(&Object) + Send + Sync + 'static) {
        *self.unparent.write().unwrap_or_else(|e| e.into_inner()) = Some(Arc::new(f));
    }

    pub(crate) fn unparent_hook(&self) -> Option<UnparentFn> {
        self.unparent.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// `object_class_property_add()`. Adding a name the class chain already has is a bug.
    pub fn property_add(&self, prop: Property) -> Arc<Property> {
        assert!(
            self.property_find(prop.name()).is_none(),
            "class property '{}' already exists on '{}'",
            prop.name(),
            self.name()
        );
        let prop = Arc::new(prop);
        lock(&self.props).insert(prop.name().to_string(), prop.clone());
        prop
    }

    /// `object_class_property_find()`: this class first, then its ancestors.
    pub fn property_find(&self, name: &str) -> Option<Arc<Property>> {
        if let Some(p) = self.parent.as_ref().and_then(|p| p.property_find(name)) {
            return Some(p);
        }
        lock(&self.props).get(name).cloned()
    }

    /// `object_class_property_find_err()`.
    pub fn property_find_err(&self, name: &str) -> Result<Arc<Property>> {
        self.property_find(name)
            .ok_or_else(|| Error::generic(format!("Property '.{name}' not found")))
    }

    /// The class properties in `object_class_property_iter_init()` order: this class in hash
    /// order, then each ancestor.
    pub fn properties(&self) -> Vec<Arc<Property>> {
        let mut v = Vec::new();
        let mut c = Some(self);
        while let Some(k) = c {
            v.extend(lock(&k.props).values().cloned());
            c = k.parent.as_deref();
        }
        v
    }

    /// `object_class_property_set_description()`.
    pub fn property_set_description(&self, name: &str, description: &str) {
        if let Some(p) = lock(&self.props).get(name) {
            p.set_description(Some(description));
        }
    }
}

macro_rules! class_uint_ptr {
    ($fn:ident, $new:ident, $atomic:ty) => {
        impl ObjectClass {
            /// `object_class_property_add_uintN_ptr()`: the cell is shared by every instance.
            pub fn $fn(&self, name: &str, cell: Arc<$atomic>, flags: PropFlags) -> Arc<Property> {
                self.property_add(Property::$new(name, cell, flags))
            }
        }
    };
}

class_uint_ptr!(property_add_uint8_ptr, new_uint8_ptr, AtomicU8);
class_uint_ptr!(property_add_uint16_ptr, new_uint16_ptr, AtomicU16);
class_uint_ptr!(property_add_uint32_ptr, new_uint32_ptr, AtomicU32);
class_uint_ptr!(property_add_uint64_ptr, new_uint64_ptr, AtomicU64);

impl ObjectClass {
    pub fn property_add_str(
        &self,
        name: &str,
        get: Option<StrGetter>,
        set: Option<StrSetter>,
    ) -> Arc<Property> {
        self.property_add(Property::new_str(name, get, set))
    }

    pub fn property_add_bool(
        &self,
        name: &str,
        get: Option<BoolGetter>,
        set: Option<BoolSetter>,
    ) -> Arc<Property> {
        self.property_add(Property::new_bool(name, get, set))
    }

    pub fn property_add_enum(
        &self,
        name: &str,
        typename: &str,
        lookup: QEnumLookup,
        get: Option<EnumGetter>,
        set: Option<EnumSetter>,
    ) -> Arc<Property> {
        self.property_add(Property::new_enum(name, typename, lookup, get, set))
    }

    pub fn property_add_tm(&self, name: &str, get: Option<TmGetter>) -> Arc<Property> {
        self.property_add(Property::new_tm(name, get))
    }

    /// `object_class_property_add_link()`. `slot` finds the link's slot in an instance, what
    /// the C code does with a field offset.
    pub fn property_add_link(
        &self,
        name: &str,
        type_: &str,
        slot: impl Fn(&Object) -> Arc<LinkSlot> + Send + Sync + 'static,
        check: Option<LinkCheck>,
        flags: LinkFlags,
    ) -> Arc<Property> {
        self.property_add(new_link(name, type_, Arc::new(slot), check, flags | LinkFlags::CLASS))
    }
}
