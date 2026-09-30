// SPDX-License-Identifier: GPL-2.0-or-later

//! `link<>` properties.

use std::fmt;
use std::sync::{Arc, Mutex};

use ruvm_base::Result;
use ruvm_qapi::visit::Visitor;

use crate::lock;
use crate::object::{Object, WeakObject};
use crate::property::{Accessor, Property};

/// `ObjectPropertyLinkFlags`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LinkFlags(u8);

impl LinkFlags {
    pub const NONE: LinkFlags = LinkFlags(0);
    /// `OBJ_PROP_LINK_STRONG`: the link holds a reference on its target.
    pub const STRONG: LinkFlags = LinkFlags(1 << 0);
    /// `OBJ_PROP_LINK_DIRECT`: the slot belongs to the property.
    pub const DIRECT: LinkFlags = LinkFlags(1 << 1);
    /// `OBJ_PROP_LINK_CLASS`: a class property, with the slot found through the object.
    pub const CLASS: LinkFlags = LinkFlags(1 << 2);

    pub fn contains(self, other: LinkFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for LinkFlags {
    type Output = LinkFlags;
    fn bitor(self, rhs: LinkFlags) -> LinkFlags {
        LinkFlags(self.0 | rhs.0)
    }
}

enum Target {
    Strong(Object),
    Weak(WeakObject),
}

/// Where a link keeps its target, the `Object **` of the C API. A strong link keeps a reference
/// and a weak one does not, so a weak link reads as empty once its target is gone.
#[derive(Default)]
pub struct LinkSlot(Mutex<Option<Target>>);

impl fmt::Debug for LinkSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let t = self.get();
        f.debug_tuple("LinkSlot").field(&t.as_ref().map(Object::typename)).finish()
    }
}

impl LinkSlot {
    pub fn new() -> Self {
        Self::default()
    }

    /// The target, if there is one and it is still alive.
    pub fn get(&self) -> Option<Object> {
        match &*lock(&self.0) {
            None => None,
            Some(Target::Strong(o)) => Some(o.clone()),
            Some(Target::Weak(w)) => w.upgrade(),
        }
    }

    /// Stores a target the way board code assigns the C pointer.
    pub fn set(&self, target: Option<&Object>, strong: bool) {
        let new = target
            .map(|t| if strong { Target::Strong(t.clone()) } else { Target::Weak(t.downgrade()) });
        let old = std::mem::replace(&mut *lock(&self.0), new);
        drop(old);
    }

    fn take(&self) -> Option<Target> {
        lock(&self.0).take()
    }
}

/// The check hook of a link, run before a new target is stored.
pub type LinkCheck = Arc<dyn Fn(&Object, &str, Option<&Object>) -> Result<()> + Send + Sync>;

/// `object_property_allow_set_link()`: a check that allows everything.
pub fn allow_set_link(_obj: &Object, _name: &str, _val: Option<&Object>) -> Result<()> {
    Ok(())
}

pub(crate) type SlotFn = Arc<dyn Fn(&Object) -> Arc<LinkSlot> + Send + Sync>;

/// `object_add_link_prop()` and `object_class_property_add_link()`.
pub(crate) fn new_link(
    name: &str,
    type_: &str,
    slot: SlotFn,
    check: Option<LinkCheck>,
    flags: LinkFlags,
) -> Property {
    let strong = flags.contains(LinkFlags::STRONG);
    let get: Accessor = {
        let slot = slot.clone();
        Arc::new(move |obj, v: &mut dyn Visitor, name| {
            let mut path = slot(obj).get().and_then(|t| t.canonical_path()).unwrap_or_default();
            v.type_str(Some(name), &mut path)
        })
    };
    let set = check.map(|check| -> Accessor {
        let slot = slot.clone();
        Arc::new(move |obj, v: &mut dyn Visitor, name| {
            let mut path = String::new();
            v.type_str(Some(name), &mut path)?;
            let new_target = if path.is_empty() {
                None
            } else {
                // object_resolve_link(): the target type is inside "link<...>".
                let type_ = obj.property_get_type(name)?;
                let target_type = &type_[5..type_.len() - 1];
                Some(obj.registry().resolve_and_typecheck(&path, name, target_type)?)
            };
            check(obj, name, new_target.as_ref())?;
            slot(obj).set(new_target.as_ref(), strong);
            Ok(())
        })
    });
    let resolve_slot = slot.clone();
    Property::new(name, format!("link<{type_}>"))
        .accessors(Some(get), set)
        .resolver(move |parent, _part| resolve_slot(parent).get())
        .releaser(move |obj, _name| {
            if strong {
                drop(slot(obj).take());
            }
        })
}
