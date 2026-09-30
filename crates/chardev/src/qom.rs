// SPDX-License-Identifier: GPL-2.0-or-later

//! The QOM side of chardevs: `TYPE_CHARDEV` and the backend types from chardev/char.c,
//! char-null.c and char-socket.c, and the objects under `/chardevs` that stand for each
//! chardev.

use std::sync::{Arc, OnceLock};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::SocketAddress;
use ruvm_qapi::visit::Visit;
use ruvm_qom::{BoolGetter, Object, Property, Registry, TYPE_OBJECT, TypeInfo};

use crate::{Backend, Chardev};

/// `TYPE_CHARDEV`.
pub const TYPE_CHARDEV: &str = "chardev";
/// `TYPE_CHARDEV_NULL`.
pub const TYPE_CHARDEV_NULL: &str = "chardev-null";
/// `TYPE_CHARDEV_SOCKET`.
pub const TYPE_CHARDEV_SOCKET: &str = "chardev-socket";

#[derive(Debug, Default)]
struct ChardevState {
    chr: OnceLock<Arc<Chardev>>,
}

fn chardev(obj: &Object) -> Result<Arc<Chardev>> {
    let st = obj.state::<ChardevState>().expect("a chardev object");
    st.chr.get().cloned().ok_or_else(|| Error::generic("chardev not opened"))
}

/// Registers the chardev types.
pub fn register_types(registry: &Registry) {
    let base = TypeInfo::new(TYPE_CHARDEV)
        .parent(TYPE_OBJECT)
        .abstract_()
        .instance_state(ChardevState::default);
    let null = TypeInfo::new(TYPE_CHARDEV_NULL).parent(TYPE_CHARDEV);
    let socket = TypeInfo::new(TYPE_CHARDEV_SOCKET).parent(TYPE_CHARDEV).class_init(|k| {
        k.property_add(Property::new("addr", "SocketAddress").getter(|obj, v, name| {
            let chr = chardev(obj)?;
            let s = chr.socket().ok_or_else(|| Error::generic("socket not connected"))?;
            let mut addr: SocketAddress = s.address();
            SocketAddress::visit(v, Some(name), &mut addr)
        }));
        let connected: BoolGetter =
            Arc::new(|obj: &Object| Ok(chardev(obj)?.socket().is_some_and(|s| s.is_connected())));
        k.property_add_bool("connected", Some(connected), None);
    });
    registry.register_all([base, null, socket]);
}

/// The object for `chr` under `/chardevs`.
pub(crate) fn add_object(registry: &Registry, chr: &Arc<Chardev>) -> Result<Object> {
    let typename = match chr.backend {
        Backend::Null => TYPE_CHARDEV_NULL,
        Backend::Socket(_) => TYPE_CHARDEV_SOCKET,
    };
    let obj = registry.object_new(typename)?;
    let st = obj.state::<ChardevState>().expect("a chardev object");
    st.chr.set(chr.clone()).expect("a new object");
    registry.container("chardevs").property_try_add_child(chr.label(), &obj)?;
    Ok(obj)
}

/// Takes the object for chardev `label` out of `/chardevs`.
pub(crate) fn remove_object(registry: &Registry, label: &str) {
    if let Some(obj) = registry.container("chardevs").resolve_path_component(label) {
        obj.unparent();
    }
}
