// SPDX-License-Identifier: GPL-2.0-or-later

//! `ObjectProperty` and the typed property helpers from qom/object.c.

use std::fmt;
use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_base::{Result, ResultExt};
use ruvm_qapi::QValue;
use ruvm_qapi::visit::{QEnumLookup, QObjectInputVisitor, Visitor, VisitorExt};

use crate::lock;
use crate::object::{Object, ObjectInner};

/// A getter or a setter: `ObjectPropertyAccessor`. The name is the property's own.
pub type Accessor = Arc<dyn Fn(&Object, &mut dyn Visitor, &str) -> Result<()> + Send + Sync>;
/// `ObjectPropertyRelease`, run when the property goes away.
pub type Release = Arc<dyn Fn(&Object, &str) + Send + Sync>;
/// `ObjectPropertyResolve`, what a path component that names this property leads to.
pub type Resolver = Arc<dyn Fn(&Object, &str) -> Option<Object> + Send + Sync>;
pub(crate) type PropInit = Arc<dyn Fn(&Object, &Property) + Send + Sync>;

pub type StrGetter = Arc<dyn Fn(&Object) -> Result<String> + Send + Sync>;
pub type StrSetter = Arc<dyn Fn(&Object, &str) -> Result<()> + Send + Sync>;
pub type BoolGetter = Arc<dyn Fn(&Object) -> Result<bool> + Send + Sync>;
pub type BoolSetter = Arc<dyn Fn(&Object, bool) -> Result<()> + Send + Sync>;
pub type EnumGetter = Arc<dyn Fn(&Object) -> Result<usize> + Send + Sync>;
pub type EnumSetter = Arc<dyn Fn(&Object, usize) -> Result<()> + Send + Sync>;
pub type TmGetter = Arc<dyn Fn(&Object) -> Result<Tm> + Send + Sync>;

/// `ObjectPropertyFlags` for the integer pointer properties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PropFlags(u8);

impl PropFlags {
    pub const READ: PropFlags = PropFlags(1 << 0);
    pub const WRITE: PropFlags = PropFlags(1 << 1);
    pub const READWRITE: PropFlags = PropFlags(3);

    fn has(self, other: PropFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

/// The fields of `struct tm` that a `struct tm` property shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tm {
    pub year: i32,
    pub mon: i32,
    pub mday: i32,
    pub hour: i32,
    pub min: i32,
    pub sec: i32,
}

/// `ObjectProperty`.
pub struct Property {
    name: String,
    type_: String,
    description: Mutex<Option<String>>,
    defval: Mutex<Option<QValue>>,
    get: Option<Accessor>,
    set: Option<Accessor>,
    resolve: Option<Resolver>,
    release: Mutex<Option<Release>>,
    init: Mutex<Option<PropInit>>,
    pub(crate) child: Option<Weak<ObjectInner>>,
    enum_lookup: Option<QEnumLookup>,
}

impl fmt::Debug for Property {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Property")
            .field("name", &self.name)
            .field("type", &self.type_)
            .field("description", &*lock(&self.description))
            .field("readable", &self.get.is_some())
            .field("writable", &self.set.is_some())
            .finish_non_exhaustive()
    }
}

impl Property {
    pub fn new(name: impl Into<String>, type_: impl Into<String>) -> Self {
        Property {
            name: name.into(),
            type_: type_.into(),
            description: Mutex::new(None),
            defval: Mutex::new(None),
            get: None,
            set: None,
            resolve: None,
            release: Mutex::new(None),
            init: Mutex::new(None),
            child: None,
            enum_lookup: None,
        }
    }

    pub fn getter(
        mut self,
        f: impl Fn(&Object, &mut dyn Visitor, &str) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.get = Some(Arc::new(f));
        self
    }

    pub fn setter(
        mut self,
        f: impl Fn(&Object, &mut dyn Visitor, &str) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.set = Some(Arc::new(f));
        self
    }

    pub fn accessors(mut self, get: Option<Accessor>, set: Option<Accessor>) -> Self {
        self.get = get;
        self.set = set;
        self
    }

    pub fn resolver(
        mut self,
        f: impl Fn(&Object, &str) -> Option<Object> + Send + Sync + 'static,
    ) -> Self {
        self.resolve = Some(Arc::new(f));
        self
    }

    pub fn releaser(mut self, f: impl Fn(&Object, &str) + Send + Sync + 'static) -> Self {
        self.release = Mutex::new(Some(Arc::new(f)));
        self
    }

    pub fn description(mut self, text: impl Into<String>) -> Self {
        self.description = Mutex::new(Some(text.into()));
        self
    }

    pub(crate) fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub(crate) fn with_child(mut self, child: Weak<ObjectInner>) -> Self {
        self.child = Some(child);
        self
    }

    pub(crate) fn with_enum(mut self, lookup: QEnumLookup) -> Self {
        self.enum_lookup = Some(lookup);
        self
    }

    pub(crate) fn with_defval(self, defval: Option<QValue>) -> Self {
        *lock(&self.defval) = defval;
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The type string, such as `bool`, `child<container>` or `link<pci-bus>`.
    pub fn type_name(&self) -> &str {
        &self.type_
    }

    pub fn get_description(&self) -> Option<String> {
        lock(&self.description).clone()
    }

    pub fn set_description(&self, text: Option<&str>) {
        *lock(&self.description) = text.map(str::to_string);
    }

    /// The default value `qom-list-properties` reports.
    pub fn default_value(&self) -> Option<QValue> {
        lock(&self.defval).clone()
    }

    pub fn is_readable(&self) -> bool {
        self.get.is_some()
    }

    pub fn is_writable(&self) -> bool {
        self.set.is_some()
    }

    pub(crate) fn get_fn(&self) -> Option<&Accessor> {
        self.get.as_ref()
    }

    pub(crate) fn set_fn(&self) -> Option<&Accessor> {
        self.set.as_ref()
    }

    pub(crate) fn resolve_fn(&self) -> Option<&Resolver> {
        self.resolve.as_ref()
    }

    pub(crate) fn release_fn(&self) -> Option<Release> {
        lock(&self.release).clone()
    }

    pub(crate) fn take_release(&self) -> Option<Release> {
        lock(&self.release).take()
    }

    pub(crate) fn init_fn(&self) -> Option<PropInit> {
        lock(&self.init).clone()
    }

    pub(crate) fn enum_lookup(&self) -> Option<&QEnumLookup> {
        self.enum_lookup.as_ref()
    }

    /// `object_property_is_child()`.
    pub fn is_child(&self) -> bool {
        self.type_.starts_with("child<")
    }

    /// `object_property_set_default()`: the value is set on every new instance before its
    /// `instance_init` runs, and `qom-list-properties` reports it.
    fn set_default(&self, value: QValue) {
        let mut d = lock(&self.defval);
        assert!(d.is_none(), "property '{}' already has a default", self.name);
        let mut init = lock(&self.init);
        assert!(init.is_none(), "property '{}' already has an init hook", self.name);
        *d = Some(value);
        *init = Some(Arc::new(|obj: &Object, prop: &Property| {
            let defval = prop.default_value().expect("default was set");
            let set = prop.set.as_ref().expect("a property with a default has a setter");
            let mut v = QObjectInputVisitor::new(defval);
            set(obj, &mut v, &prop.name).or_abort();
        }));
    }

    pub fn set_default_bool(&self, value: bool) {
        self.set_default(QValue::Bool(value));
    }

    pub fn set_default_str(&self, value: &str) {
        self.set_default(QValue::str(value));
    }

    pub fn set_default_list(&self) {
        self.set_default(QValue::List(Vec::new()));
    }

    pub fn set_default_int(&self, value: i64) {
        self.set_default(QValue::Int(value));
    }

    pub fn set_default_uint(&self, value: u64) {
        self.set_default(QValue::Uint(value));
    }

    /// A `string` property, `object_property_add_str()`.
    pub(crate) fn new_str(name: &str, get: Option<StrGetter>, set: Option<StrSetter>) -> Self {
        let g = get.map(|get| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = get(obj)?;
                v.type_str(Some(name), &mut value)
            })
        });
        let s = set.map(|set| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = String::new();
                v.type_str(Some(name), &mut value)?;
                set(obj, &value)
            })
        });
        Property::new(name, "string").accessors(g, s)
    }

    /// A `bool` property, `object_property_add_bool()`.
    pub(crate) fn new_bool(name: &str, get: Option<BoolGetter>, set: Option<BoolSetter>) -> Self {
        let g = get.map(|get| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = get(obj)?;
                v.type_bool(Some(name), &mut value)
            })
        });
        let s = set.map(|set| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = false;
                v.type_bool(Some(name), &mut value)?;
                set(obj, value)
            })
        });
        Property::new(name, "bool").accessors(g, s)
    }

    /// An enum property, `object_property_add_enum()`. The type string is the QAPI enum name.
    pub(crate) fn new_enum(
        name: &str,
        typename: &str,
        lookup: QEnumLookup,
        get: Option<EnumGetter>,
        set: Option<EnumSetter>,
    ) -> Self {
        let g = get.map(|get| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = get(obj)?;
                v.type_enum(Some(name), &mut value, &lookup)
            })
        });
        let s = set.map(|set| -> Accessor {
            Arc::new(move |obj, v, name| {
                let mut value = 0;
                v.type_enum(Some(name), &mut value, &lookup)?;
                set(obj, value)
            })
        });
        Property::new(name, typename).accessors(g, s).with_enum(lookup)
    }

    /// A `struct tm` property, `object_property_add_tm()`. It can only be read.
    pub(crate) fn new_tm(name: &str, get: Option<TmGetter>) -> Self {
        let g = get.map(|get| -> Accessor {
            Arc::new(move |obj, v, name| {
                let value = get(obj)?;
                v.start_struct(Some(name))?;
                let r = (|| {
                    for (field, mut x) in [
                        ("tm_year", value.year),
                        ("tm_mon", value.mon),
                        ("tm_mday", value.mday),
                        ("tm_hour", value.hour),
                        ("tm_min", value.min),
                        ("tm_sec", value.sec),
                    ] {
                        v.type_int32(Some(field), &mut x)?;
                    }
                    v.check_struct()
                })();
                v.end_struct();
                r
            })
        });
        Property::new(name, "struct tm").accessors(g, None)
    }
}

macro_rules! uint_ptr_prop {
    ($fn:ident, $atomic:ty, $int:ty, $visit:ident, $tname:literal) => {
        impl Property {
            /// An integer property backed by a shared cell, `object_property_add_uintN_ptr()`.
            pub(crate) fn $fn(name: &str, cell: Arc<$atomic>, flags: PropFlags) -> Self {
                let g = flags.has(PropFlags::READ).then(|| -> Accessor {
                    let cell = cell.clone();
                    Arc::new(move |_obj, v, name| {
                        let mut value: $int = cell.load(Ordering::SeqCst);
                        v.$visit(Some(name), &mut value)
                    })
                });
                let s = flags.has(PropFlags::WRITE).then(|| -> Accessor {
                    let cell = cell.clone();
                    Arc::new(move |_obj, v, name| {
                        let mut value: $int = 0;
                        v.$visit(Some(name), &mut value)?;
                        cell.store(value, Ordering::SeqCst);
                        Ok(())
                    })
                });
                Property::new(name, $tname).accessors(g, s)
            }
        }
    };
}

uint_ptr_prop!(new_uint8_ptr, AtomicU8, u8, type_uint8, "uint8");
uint_ptr_prop!(new_uint16_ptr, AtomicU16, u16, type_uint16, "uint16");
uint_ptr_prop!(new_uint32_ptr, AtomicU32, u32, type_uint32, "uint32");
uint_ptr_prop!(new_uint64_ptr, AtomicU64, u64, type_uint64, "uint64");
