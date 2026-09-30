// SPDX-License-Identifier: GPL-2.0-or-later

//! `Object`: instances, their lifecycle, their properties and the composition tree.

use std::any::Any;
use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::ghash::GHashTable;
use ruvm_qapi::visit::{
    ForwardFieldVisitor, QEnumLookup, QObjectInputVisitor, QObjectOutputVisitor,
    StringInputVisitor, StringOutputVisitor, Visitor,
};
use ruvm_qapi::{QDict, QValue};

use crate::class::ObjectClass;
use crate::link::{LinkCheck, LinkFlags, LinkSlot, new_link};
use crate::lock;
use crate::property::{
    BoolGetter, BoolSetter, EnumGetter, EnumSetter, PropFlags, Property, StrGetter, StrSetter,
    TmGetter,
};
use crate::types::Registry;
use crate::user_creatable::user_creatable_complete;
use crate::{TYPE_CONTAINER, TYPE_OBJECT, TYPE_USER_CREATABLE};

pub(crate) struct ObjectInner {
    class: Arc<ObjectClass>,
    refcnt: AtomicU32,
    finalizing: AtomicBool,
    parent: Mutex<Option<Weak<ObjectInner>>>,
    props: Mutex<GHashTable<Arc<Property>>>,
    state: Vec<Arc<dyn Any + Send + Sync>>,
}

/// A reference to a QOM object. Each handle counts as one QOM reference: clone is
/// `object_ref()`, drop is `object_unref()`, and the last drop finalizes the object.
pub struct Object {
    inner: Arc<ObjectInner>,
}

/// A handle that does not keep the object alive, for back pointers such as weak links.
#[derive(Clone)]
pub struct WeakObject(Weak<ObjectInner>);

impl fmt::Debug for WeakObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("WeakObject")
    }
}

impl WeakObject {
    /// The object, unless it has been finalized or is being finalized.
    pub fn upgrade(&self) -> Option<Object> {
        Object::from_arc(self.0.upgrade()?)
    }
}

impl Clone for Object {
    fn clone(&self) -> Self {
        let r = self.inner.refcnt.fetch_add(1, Ordering::SeqCst);
        assert!(r < i32::MAX as u32);
        Object { inner: self.inner.clone() }
    }
}

impl Drop for Object {
    fn drop(&mut self) {
        if self.inner.refcnt.fetch_sub(1, Ordering::SeqCst) == 1
            && !self.inner.finalizing.swap(true, Ordering::SeqCst)
        {
            self.finalize();
        }
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Object")
            .field("type", &self.typename())
            .field("path", &self.canonical_path())
            .finish()
    }
}

impl PartialEq for Object {
    fn eq(&self, other: &Self) -> bool {
        self.ptr_eq(other)
    }
}

impl Eq for Object {}

impl Object {
    fn from_arc(inner: Arc<ObjectInner>) -> Option<Object> {
        let mut cur = inner.refcnt.load(Ordering::SeqCst);
        loop {
            if cur == 0 || inner.finalizing.load(Ordering::SeqCst) {
                return None;
            }
            match inner.refcnt.compare_exchange(cur, cur + 1, Ordering::SeqCst, Ordering::SeqCst) {
                Ok(_) => return Some(Object { inner }),
                Err(v) => cur = v,
            }
        }
    }

    /// `object_new_with_class()`: runs the class property init hooks, then every
    /// `instance_init` from the root type down, then every `instance_post_init`.
    pub fn new_with_class(class: &Arc<ObjectClass>) -> Result<Object> {
        if class.is_abstract() {
            return Err(Error::generic(format!("object type '{}' is abstract", class.name())));
        }
        let mut chain = Vec::new();
        let mut c = Some(class);
        while let Some(k) = c {
            chain.push(k.clone());
            c = k.parent();
        }
        chain.reverse();
        let state =
            chain.iter().filter_map(|k| k.info.instance_state.as_ref().map(|f| f())).collect();
        let obj = Object {
            inner: Arc::new(ObjectInner {
                class: class.clone(),
                refcnt: AtomicU32::new(1),
                finalizing: AtomicBool::new(false),
                parent: Mutex::new(None),
                props: Mutex::new(GHashTable::new()),
                state,
            }),
        };
        for prop in class.properties() {
            if let Some(init) = prop.init_fn() {
                init(&obj, &prop);
            }
        }
        for k in &chain {
            if let Some(h) = &k.info.instance_init {
                h(&obj);
            }
        }
        for k in &chain {
            if let Some(h) = &k.info.instance_post_init {
                h(&obj);
            }
        }
        Ok(obj)
    }

    /// `object_finalize()`: releases the properties, which drops the children, then runs
    /// every `instance_finalize` from the most derived type up.
    fn finalize(&self) {
        self.property_del_all();
        let mut c = Some(&self.inner.class);
        while let Some(k) = c {
            if let Some(h) = &k.info.instance_finalize {
                h(self);
            }
            c = k.parent();
        }
        debug_assert!(self.parent().is_none());
        let props = std::mem::take(&mut *lock(&self.inner.props));
        drop(props);
    }

    pub fn ptr_eq(&self, other: &Object) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub fn downgrade(&self) -> WeakObject {
        WeakObject(Arc::downgrade(&self.inner))
    }

    /// The number of QOM references, `obj->ref`.
    pub fn ref_count(&self) -> u32 {
        self.inner.refcnt.load(Ordering::SeqCst)
    }

    /// `object_get_class()`.
    pub fn class(&self) -> &Arc<ObjectClass> {
        &self.inner.class
    }

    /// `object_get_typename()`.
    pub fn typename(&self) -> &str {
        self.inner.class.name()
    }

    pub fn registry(&self) -> Registry {
        self.inner.class.registry().expect("the registry outlives its objects")
    }

    /// The state a type in this object's chain attached with `TypeInfo::instance_state`.
    pub fn state<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
        self.inner.state.iter().find_map(|s| s.clone().downcast::<T>().ok())
    }

    /// `object_dynamic_cast()`.
    pub fn dynamic_cast(&self, typename: &str) -> Option<&Object> {
        self.inner.class.dynamic_cast(typename).map(|_| self)
    }

    /// The composition parent, `obj->parent`.
    pub fn parent(&self) -> Option<Object> {
        let p = lock(&self.inner.parent).clone()?;
        Object::from_arc(p.upgrade()?)
    }

    // Properties.

    /// `object_property_try_add()`. A name ending in `[*]` gets the first free `[N]`.
    pub fn property_try_add(&self, prop: Property) -> Result<Arc<Property>> {
        if let Some(base) = prop.name().strip_suffix("[*]") {
            let base = base.to_string();
            let mut prop = Some(prop);
            for i in 0..i16::MAX {
                let full = format!("{base}[{i}]");
                if self.property_find(&full).is_none() {
                    let p = prop.take().expect("used once").with_name(full);
                    return self.property_try_add(p);
                }
            }
            panic!("no free index for property '{base}[*]'");
        }
        if self.property_find(prop.name()).is_some() {
            return Err(Error::generic(format!(
                "attempt to add duplicate property '{}' to object (type '{}')",
                prop.name(),
                self.typename()
            )));
        }
        let prop = Arc::new(prop);
        lock(&self.inner.props).insert(prop.name().to_string(), prop.clone());
        Ok(prop)
    }

    /// `object_property_add()`. A duplicate name is a bug.
    pub fn property_add(&self, prop: Property) -> Arc<Property> {
        match self.property_try_add(prop) {
            Ok(p) => p,
            Err(e) => panic!("{}", e.message()),
        }
    }

    /// `object_property_find()`: class properties first, then instance properties.
    pub fn property_find(&self, name: &str) -> Option<Arc<Property>> {
        if let Some(p) = self.inner.class.property_find(name) {
            return Some(p);
        }
        lock(&self.inner.props).get(name).cloned()
    }

    /// `object_property_find_err()`.
    pub fn property_find_err(&self, name: &str) -> Result<Arc<Property>> {
        self.property_find(name).ok_or_else(|| {
            Error::generic(format!("Property '{}.{}' not found", self.typename(), name))
        })
    }

    fn instance_properties(&self) -> Vec<Arc<Property>> {
        lock(&self.inner.props).values().cloned().collect()
    }

    /// Every property in `object_property_iter_next()` order: instance properties in hash
    /// order, then the class properties of the class and each ancestor.
    pub fn properties(&self) -> Vec<Arc<Property>> {
        let mut v = self.instance_properties();
        v.extend(self.inner.class.properties());
        v
    }

    /// `object_property_del()`.
    pub fn property_del(&self, name: &str) {
        let prop = lock(&self.inner.props).get(name).cloned();
        if let Some(prop) = prop {
            if let Some(release) = prop.release_fn() {
                release(self, name);
            }
            lock(&self.inner.props).remove(name);
        }
    }

    /// `object_property_del_all()`: releases one property at a time and starts over, because a
    /// release hook may remove other properties.
    fn property_del_all(&self) {
        let mut done: HashSet<*const Property> = HashSet::new();
        loop {
            let mut released = false;
            for prop in self.properties() {
                if done.insert(Arc::as_ptr(&prop)) {
                    if let Some(release) = prop.release_fn() {
                        release(self, prop.name());
                        released = true;
                        break;
                    }
                }
            }
            if !released {
                break;
            }
        }
    }

    fn find_child_prop(&self, child: &Object) -> Option<Arc<Property>> {
        lock(&self.inner.props)
            .values()
            .find(|p| p.is_child() && p.child.as_ref().is_some_and(|c| child.is_weak(c)))
            .cloned()
    }

    fn is_weak(&self, w: &Weak<ObjectInner>) -> bool {
        std::ptr::eq(Arc::as_ptr(&self.inner), w.as_ptr())
    }

    /// `object_property_del_child()`.
    fn property_del_child(&self, child: &Object) {
        if let Some(prop) = self.find_child_prop(child) {
            if let Some(release) = prop.take_release() {
                release(self, prop.name());
            }
        }
        if let Some(prop) = self.find_child_prop(child) {
            lock(&self.inner.props).remove_no_resize(prop.name());
        }
    }

    /// `object_unparent()`: removes the object from the composition tree, dropping the
    /// reference its parent held.
    pub fn unparent(&self) {
        if let Some(parent) = self.parent() {
            parent.property_del_child(self);
        }
    }

    /// `object_property_get()`.
    pub fn property_get(&self, name: &str, v: &mut dyn Visitor) -> Result<()> {
        let prop = self.property_find_err(name)?;
        let Some(get) = prop.get_fn() else {
            return Err(Error::generic(format!(
                "Property '{}.{}' is not readable",
                self.typename(),
                name
            )));
        };
        get(self, v, name)
    }

    /// `object_property_set()`.
    pub fn property_set(&self, name: &str, v: &mut dyn Visitor) -> Result<()> {
        let prop = self.property_find_err(name)?;
        let Some(set) = prop.set_fn() else {
            return Err(Error::generic(format!(
                "Property '{}.{}' is not writable",
                self.typename(),
                name
            )));
        };
        set(self, v, name)
    }

    /// `object_property_set_qobject()`.
    pub fn property_set_qobject(&self, name: &str, value: QValue) -> Result<()> {
        let mut v = QObjectInputVisitor::new(value);
        self.property_set(name, &mut v)
    }

    /// `object_property_get_qobject()`.
    pub fn property_get_qobject(&self, name: &str) -> Result<QValue> {
        let mut v = QObjectOutputVisitor::new();
        self.property_get(name, &mut v)?;
        Ok(v.complete())
    }

    pub fn property_set_str(&self, name: &str, value: &str) -> Result<()> {
        self.property_set_qobject(name, QValue::str(value))
    }

    pub fn property_get_str(&self, name: &str) -> Result<String> {
        match self.property_get_qobject(name)? {
            QValue::Str(s) => Ok(s),
            _ => Err(Error::generic(format!(
                "Invalid parameter type for '{name}', expected: string"
            ))),
        }
    }

    pub fn property_set_bool(&self, name: &str, value: bool) -> Result<()> {
        self.property_set_qobject(name, QValue::Bool(value))
    }

    pub fn property_get_bool(&self, name: &str) -> Result<bool> {
        match self.property_get_qobject(name)? {
            QValue::Bool(b) => Ok(b),
            _ => Err(Error::generic(format!(
                "Invalid parameter type for '{name}', expected: boolean"
            ))),
        }
    }

    pub fn property_set_int(&self, name: &str, value: i64) -> Result<()> {
        self.property_set_qobject(name, QValue::Int(value))
    }

    pub fn property_get_int(&self, name: &str) -> Result<i64> {
        self.property_get_qobject(name)?.as_i64().ok_or_else(|| {
            Error::generic(format!("Invalid parameter type for '{name}', expected: int"))
        })
    }

    pub fn property_set_uint(&self, name: &str, value: u64) -> Result<()> {
        self.property_set_qobject(name, QValue::Uint(value))
    }

    pub fn property_get_uint(&self, name: &str) -> Result<u64> {
        self.property_get_qobject(name)?.as_u64().ok_or_else(|| {
            Error::generic(format!("Invalid parameter type for '{name}', expected: uint"))
        })
    }

    /// `object_property_set_link()`: stores the target's canonical path.
    pub fn property_set_link(&self, name: &str, value: Option<&Object>) -> Result<()> {
        let path = value.and_then(Object::canonical_path).unwrap_or_default();
        self.property_set_str(name, &path)
    }

    /// `object_property_get_link()`.
    pub fn property_get_link(&self, name: &str) -> Result<Option<Object>> {
        let s = self.property_get_str(name)?;
        if s.is_empty() {
            return Ok(None);
        }
        match self.registry().resolve_path(&s).0 {
            Some(o) => Ok(Some(o)),
            None => Err(Error::new(ErrorClass::DeviceNotFound, format!("Device '{s}' not found"))),
        }
    }

    /// `object_property_get_enum()`.
    pub fn property_get_enum(&self, name: &str, typename: &str) -> Result<usize> {
        let prop = self.property_find_err(name)?;
        let lookup = match prop.enum_lookup() {
            Some(l) if prop.type_name() == typename => *l,
            _ => {
                return Err(Error::generic(format!(
                    "Property {} on {} is not '{}' enum type",
                    name,
                    self.typename(),
                    typename
                )));
            }
        };
        let s = self.property_get_str(name)?;
        lookup.parse(&s)
    }

    /// `object_property_parse()`: sets a property from a command line string.
    pub fn property_parse(&self, name: &str, string: &str) -> Result<()> {
        let mut v = StringInputVisitor::new(string);
        self.property_set(name, &mut v)
    }

    /// `object_property_print()`.
    pub fn property_print(&self, name: &str, human: bool) -> Result<String> {
        let mut v = StringOutputVisitor::new(human);
        self.property_get(name, &mut v)?;
        Ok(v.complete())
    }

    /// `object_property_get_type()`.
    pub fn property_get_type(&self, name: &str) -> Result<String> {
        Ok(self.property_find_err(name)?.type_name().to_string())
    }

    /// `object_property_set_description()`.
    pub fn property_set_description(&self, name: &str, description: Option<&str>) {
        let prop = self.property_find(name).expect("property exists");
        prop.set_description(description);
    }

    /// `object_set_props()`: parses each value into its property, stopping at the first error.
    pub fn set_props(&self, props: &[(&str, &str)]) -> Result<()> {
        for (name, value) in props {
            self.property_parse(name, value)?;
        }
        Ok(())
    }

    /// `object_set_props_from_qdict()`: sets each key of `qdict` from `v`, a visitor over the
    /// same dictionary, then checks that nothing was left over.
    pub fn set_props_from_qdict(&self, qdict: &QDict, v: &mut dyn Visitor) -> Result<()> {
        v.start_struct(None)?;
        let r = (|| {
            for key in qdict.keys() {
                self.property_set(key, v)?;
            }
            v.check_struct()
        })();
        v.end_struct();
        r
    }

    /// `object_set_props_from_keyval()`.
    pub fn set_props_from_keyval(&self, qdict: &QDict, from_json: bool) -> Result<()> {
        let root = QValue::Dict(qdict.clone());
        let mut v = if from_json {
            QObjectInputVisitor::new(root)
        } else {
            QObjectInputVisitor::new_keyval(root)
        };
        self.set_props_from_qdict(qdict, &mut v)
    }

    // Typed properties.

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

    pub fn property_add_uint8_ptr(
        &self,
        name: &str,
        cell: Arc<AtomicU8>,
        flags: PropFlags,
    ) -> Arc<Property> {
        self.property_add(Property::new_uint8_ptr(name, cell, flags))
    }

    pub fn property_add_uint16_ptr(
        &self,
        name: &str,
        cell: Arc<AtomicU16>,
        flags: PropFlags,
    ) -> Arc<Property> {
        self.property_add(Property::new_uint16_ptr(name, cell, flags))
    }

    pub fn property_add_uint32_ptr(
        &self,
        name: &str,
        cell: Arc<AtomicU32>,
        flags: PropFlags,
    ) -> Arc<Property> {
        self.property_add(Property::new_uint32_ptr(name, cell, flags))
    }

    pub fn property_add_uint64_ptr(
        &self,
        name: &str,
        cell: Arc<AtomicU64>,
        flags: PropFlags,
    ) -> Arc<Property> {
        self.property_add(Property::new_uint64_ptr(name, cell, flags))
    }

    /// `object_property_add_alias()`: a property that reads and writes `target_name` on
    /// `target`. An alias of a child property is a link.
    pub fn property_add_alias(
        &self,
        name: &str,
        target: &Object,
        target_name: &str,
    ) -> Arc<Property> {
        let target_prop = match target.property_find_err(target_name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e.message()),
        };
        let type_ = match target_prop.type_name().strip_prefix("child") {
            Some(rest) if target_prop.is_child() => format!("link{rest}"),
            _ => target_prop.type_name().to_string(),
        };
        let tn = target_name.to_string();
        let (g, s, r) = (target.downgrade(), target.downgrade(), target.downgrade());
        let (gn, sn, rn) = (tn.clone(), tn.clone(), tn);
        let prop = Property::new(name, type_)
            .getter(move |_obj, v, name| {
                let t = g.upgrade().ok_or_else(|| Error::generic("alias target is gone"))?;
                let mut fv = ForwardFieldVisitor::new(v, &gn, name);
                t.property_get(&gn, &mut fv)
            })
            .setter(move |_obj, v, name| {
                let t = s.upgrade().ok_or_else(|| Error::generic("alias target is gone"))?;
                let mut fv = ForwardFieldVisitor::new(v, &sn, name);
                t.property_set(&sn, &mut fv)
            })
            .resolver(move |_obj, _part| r.upgrade()?.resolve_path_component(&rn))
            .releaser(|_, _| {})
            .with_defval(target_prop.default_value());
        let op = self.property_add(prop);
        op.set_description(target_prop.get_description().as_deref());
        op
    }

    // Children and links.

    /// `object_property_try_add_child()`: makes `child` a child of this object, which takes a
    /// reference on it and gives it a canonical path.
    pub fn property_try_add_child(&self, name: &str, child: &Object) -> Result<Arc<Property>> {
        assert!(child.parent().is_none(), "object already has a parent");
        let type_ = format!("child<{}>", child.typename());
        let held = Mutex::new(Some(child.clone()));
        let get_child = child.downgrade();
        let res_child = child.downgrade();
        let prop = Property::new(name, type_)
            .getter(move |_obj, v, name| {
                let mut path =
                    get_child.upgrade().and_then(|c| c.canonical_path()).unwrap_or_default();
                v.type_str(Some(name), &mut path)
            })
            .resolver(move |_obj, _part| res_child.upgrade())
            .releaser(move |_obj, _name| {
                let Some(child) = lock(&held).take() else { return };
                if let Some(unparent) = child.class().unparent_hook() {
                    unparent(&child);
                }
                *lock(&child.inner.parent) = None;
                drop(child);
            })
            .with_child(Arc::downgrade(&child.inner));
        let op = self.property_try_add(prop)?;
        *lock(&child.inner.parent) = Some(Arc::downgrade(&self.inner));
        Ok(op)
    }

    /// `object_property_add_child()`.
    pub fn property_add_child(&self, name: &str, child: &Object) -> Arc<Property> {
        match self.property_try_add_child(name, child) {
            Ok(p) => p,
            Err(e) => panic!("{}", e.message()),
        }
    }

    /// `object_property_add_new_container()`.
    pub fn add_new_container(&self, name: &str) -> Object {
        let c = self.registry().object_new(TYPE_CONTAINER).expect("container type exists");
        self.property_add_child(name, &c);
        c
    }

    /// `object_initialize_child()`: creates an object of `typename` as a child of this one.
    pub fn initialize_child(&self, name: &str, typename: &str) -> Result<Object> {
        let child = self.registry().object_new(typename)?;
        self.property_try_add_child(name, &child)?;
        Ok(child)
    }

    /// `object_property_add_link()`.
    pub fn property_add_link(
        &self,
        name: &str,
        type_: &str,
        slot: Arc<LinkSlot>,
        check: Option<LinkCheck>,
        flags: LinkFlags,
    ) -> Arc<Property> {
        self.property_add(new_link(name, type_, Arc::new(move |_| slot.clone()), check, flags))
    }

    /// `object_property_add_const_link()`: a read only link to `target`.
    pub fn property_add_const_link(&self, name: &str, target: &Object) -> Arc<Property> {
        let slot = Arc::new(LinkSlot::new());
        slot.set(Some(target), false);
        self.property_add_link(name, target.typename(), slot, None, LinkFlags::DIRECT)
    }

    /// `object_child_foreach()`. A non-zero return stops the walk and is returned.
    pub fn child_foreach(&self, mut f: impl FnMut(&Object) -> i32) -> i32 {
        self.do_child_foreach(&mut f, false)
    }

    /// `object_child_foreach_recursive()`.
    pub fn child_foreach_recursive(&self, mut f: impl FnMut(&Object) -> i32) -> i32 {
        self.do_child_foreach(&mut f, true)
    }

    pub(crate) fn children(&self) -> Vec<Object> {
        self.instance_properties()
            .iter()
            .filter(|p| p.is_child())
            .filter_map(|p| p.child.as_ref().and_then(|w| Object::from_arc(w.upgrade()?)))
            .collect()
    }

    fn do_child_foreach(&self, f: &mut dyn FnMut(&Object) -> i32, recurse: bool) -> i32 {
        for child in self.children() {
            let ret = f(&child);
            if ret != 0 {
                return ret;
            }
            if recurse {
                let ret = child.do_child_foreach(f, true);
                if ret != 0 {
                    return ret;
                }
            }
        }
        0
    }

    // Paths.

    /// `object_get_canonical_path_component()`: the name of the child property the parent
    /// holds this object under.
    pub fn canonical_path_component(&self) -> Option<String> {
        let parent = self.parent()?;
        let props = parent.instance_properties();
        let p = props
            .iter()
            .find(|p| p.is_child() && p.child.as_ref().is_some_and(|c| self.is_weak(c)));
        Some(p.expect("a child is in its parent's properties").name().to_string())
    }

    /// `object_get_canonical_path()`. `None` when the object is not in the tree.
    pub fn canonical_path(&self) -> Option<String> {
        let registry = self.inner.class.registry()?;
        if registry.is_root(self) {
            return Some("/".into());
        }
        let mut path = String::new();
        let mut obj = self.clone();
        loop {
            let component = obj.canonical_path_component()?;
            path = format!("/{component}{path}");
            obj = obj.parent().expect("has a parent");
            if registry.is_root(&obj) {
                return Some(path);
            }
        }
    }

    /// `object_resolve_path_component()`.
    pub fn resolve_path_component(&self, part: &str) -> Option<Object> {
        let prop = self.property_find(part)?;
        let resolve = prop.resolve_fn()?;
        resolve(self, part)
    }

    pub(crate) fn resolve_abs_path(&self, parts: &[&str], typename: &str) -> Option<Object> {
        let Some((first, rest)) = parts.split_first() else {
            return self.dynamic_cast(typename).cloned();
        };
        if first.is_empty() {
            return self.resolve_abs_path(rest, typename);
        }
        self.resolve_path_component(first)?.resolve_abs_path(rest, typename)
    }

    pub(crate) fn resolve_partial_path(
        &self,
        parts: &[&str],
        typename: &str,
        ambiguous: &mut bool,
    ) -> Option<Object> {
        let mut obj = self.resolve_abs_path(parts, typename);
        for child in self.children() {
            let found = child.resolve_partial_path(parts, typename, ambiguous);
            if found.is_some() {
                if obj.is_some() {
                    *ambiguous = true;
                    return None;
                }
                obj = found;
            }
            if *ambiguous {
                return None;
            }
        }
        obj
    }

    /// `object_resolve_path_at()`: a path relative to this object, or absolute.
    pub fn resolve_path_at(&self, path: &str) -> Option<Object> {
        let parts: Vec<&str> = if path.is_empty() { Vec::new() } else { path.split('/').collect() };
        if path.starts_with('/') {
            return self.registry().root().resolve_abs_path(&parts[1..], TYPE_OBJECT);
        }
        self.resolve_abs_path(&parts, TYPE_OBJECT)
    }

    /// `object_apply_global_props()`. With `fatal` the first failure is returned; otherwise
    /// failures are warnings and the rest still apply.
    pub fn apply_global_props(
        &self,
        props: &[Arc<crate::GlobalProperty>],
        fatal: bool,
    ) -> Result<()> {
        for p in props {
            if self.dynamic_cast(&p.driver).is_none() {
                continue;
            }
            if p.optional && self.property_find(&p.property).is_none() {
                continue;
            }
            p.used.store(true, Ordering::SeqCst);
            if let Err(e) = self.property_parse(&p.property, &p.value) {
                let e = e.prepend(format!(
                    "can't apply global {}.{}={}: ",
                    p.driver, p.property, p.value
                ));
                if fatal {
                    return Err(e);
                }
                ruvm_base::warn_report(e.message());
            }
        }
        Ok(())
    }
}

impl Registry {
    /// `object_new_with_props()` and `object_new_with_props_parentless()`: creates an object,
    /// runs `set_props` on it, adds it as child `id` of `parent` when there is one, and
    /// completes it if it is user creatable.
    pub fn object_new_with(
        &self,
        typename: &str,
        parent: Option<(&Object, &str)>,
        set_props: impl FnOnce(&Object) -> Result<()>,
    ) -> Result<Object> {
        if let Some((_, id)) = parent {
            if !ruvm_qapi::cutils::id_wellformed(id) {
                return Err(Error::generic("Parameter 'id' expects an identifier").hint(
                    "Identifiers consist of letters, digits, '-', '.', '_', starting with a letter.\n",
                ));
            }
        }
        let Some(class) = self.class_by_name(typename) else {
            return Err(Error::generic(format!("invalid object type: {typename}")));
        };
        if class.is_abstract() {
            return Err(Error::generic(format!("object type '{typename}' is abstract")));
        }
        let obj = Object::new_with_class(&class)?;
        set_props(&obj)?;
        if let Some((parent, id)) = parent {
            parent.property_try_add_child(id, &obj)?;
        }
        if obj.dynamic_cast(TYPE_USER_CREATABLE).is_some() {
            if let Err(e) = user_creatable_complete(&obj) {
                if parent.is_some() {
                    obj.unparent();
                }
                return Err(e);
            }
        }
        Ok(obj)
    }

    /// `object_new_with_props()` with string values.
    pub fn object_new_with_props(
        &self,
        typename: &str,
        parent: Option<(&Object, &str)>,
        props: &[(&str, &str)],
    ) -> Result<Object> {
        self.object_new_with(typename, parent, |o| o.set_props(props))
    }

    /// `object_new_with_props_from_qdict()`.
    pub fn object_new_with_props_from_qdict(
        &self,
        typename: &str,
        parent: Option<(&Object, &str)>,
        props: &QDict,
        v: &mut dyn Visitor,
    ) -> Result<Object> {
        self.object_new_with(typename, parent, |o| o.set_props_from_qdict(props, v))
    }
}
