// SPDX-License-Identifier: GPL-2.0-or-later

//! The `user-creatable` interface, qom/object_interfaces.c: what `-object`, `object-add` and
//! `object-del` work with.

use std::fmt;
use std::sync::Arc;

use ruvm_base::{Error, Result};
use ruvm_qapi::visit::QObjectInputVisitor;
use ruvm_qapi::{QDict, QValue};

use crate::TYPE_USER_CREATABLE;
use crate::object::Object;
use crate::types::Registry;

type CompleteFn = Arc<dyn Fn(&Object) -> Result<()> + Send + Sync>;

/// `UserCreatableClass`. A type sets it on its view of the interface:
/// `klass.interface("user-creatable").unwrap().set_ext(UserCreatableClass { .. })`.
#[derive(Clone, Default)]
pub struct UserCreatableClass {
    /// Runs once the properties are set, and can refuse the object.
    pub complete: Option<CompleteFn>,
    /// Runs before `object-del`, and can refuse the deletion.
    pub prepare_delete: Option<CompleteFn>,
}

impl fmt::Debug for UserCreatableClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UserCreatableClass")
            .field("complete", &self.complete.is_some())
            .field("prepare_delete", &self.prepare_delete.is_some())
            .finish()
    }
}

fn uc_class(obj: &Object) -> Option<Arc<UserCreatableClass>> {
    obj.class().dynamic_cast(TYPE_USER_CREATABLE)?.ext::<UserCreatableClass>()
}

/// `user_creatable_complete()`.
pub fn user_creatable_complete(obj: &Object) -> Result<()> {
    match uc_class(obj).and_then(|c| c.complete.clone()) {
        Some(f) => f(obj),
        None => Ok(()),
    }
}

impl Registry {
    /// `user_creatable_add_qapi()` for the arguments of `object-add`: `qom-type`, `id` and the
    /// type's properties, all in one dictionary. With `keyval` the values are strings from the
    /// command line. The object goes under `/objects`.
    pub fn user_creatable_add(&self, args: &QDict, keyval: bool) -> Result<Object> {
        let qom_type = args
            .get_str("qom-type")
            .ok_or_else(|| Error::generic("Parameter 'qom-type' is missing"))?
            .to_string();
        let id = args
            .get_str("id")
            .ok_or_else(|| Error::generic("Parameter 'id' is missing"))?
            .to_string();
        let mut props = args.clone();
        props.remove("qom-type");
        props.remove("id");
        let root = QValue::Dict(props.clone());
        let mut v = if keyval {
            QObjectInputVisitor::new_keyval(root)
        } else {
            QObjectInputVisitor::new(root)
        };
        let parent = self.objects_root();
        self.object_new_with_props_from_qdict(&qom_type, Some((&parent, &id)), &props, &mut v)
    }

    /// `user_creatable_del()`.
    pub fn user_creatable_del(&self, id: &str) -> Result<()> {
        let container = self.objects_root();
        let Some(obj) = container.resolve_path_component(id) else {
            return Err(Error::generic(format!("object '{id}' not found")));
        };
        if let Some(f) = uc_class(&obj).and_then(|c| c.prepare_delete.clone()) {
            f(&obj)?;
        }
        obj.unparent();
        Ok(())
    }

    /// `user_creatable_cleanup()`.
    pub fn user_creatable_cleanup(&self) {
        self.objects_root().unparent();
    }
}
