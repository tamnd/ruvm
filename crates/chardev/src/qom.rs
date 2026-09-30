// SPDX-License-Identifier: GPL-2.0-or-later

//! The QOM side of chardevs: `TYPE_CHARDEV` and the backend types from chardev/*.c, and the
//! objects under `/chardevs` that stand for each chardev.

use std::sync::{Arc, OnceLock};

use ruvm_base::{Error, Result};
use ruvm_qapi::types::SocketAddress;
use ruvm_qapi::visit::Visit;
use ruvm_qom::{BoolGetter, Object, Property, Registry, TYPE_OBJECT, TypeInfo};

use crate::Chardev;

/// `TYPE_CHARDEV`.
pub const TYPE_CHARDEV: &str = "chardev";
/// `TYPE_CHARDEV_NULL`.
pub const TYPE_CHARDEV_NULL: &str = "chardev-null";
/// `TYPE_CHARDEV_SOCKET`.
pub const TYPE_CHARDEV_SOCKET: &str = "chardev-socket";
/// `TYPE_CHARDEV_FD`, the abstract parent of the Unix `file`, `pipe` and `stdio`.
#[cfg(unix)]
pub const TYPE_CHARDEV_FD: &str = "chardev-fd";
/// `TYPE_CHARDEV_FILE`.
pub const TYPE_CHARDEV_FILE: &str = "chardev-file";
/// `TYPE_CHARDEV_PIPE`.
pub const TYPE_CHARDEV_PIPE: &str = "chardev-pipe";
/// `TYPE_CHARDEV_STDIO`.
pub const TYPE_CHARDEV_STDIO: &str = "chardev-stdio";
/// `TYPE_CHARDEV_PTY`.
pub const TYPE_CHARDEV_PTY: &str = "chardev-pty";
/// `TYPE_CHARDEV_RINGBUF`.
pub const TYPE_CHARDEV_RINGBUF: &str = "chardev-ringbuf";
/// `TYPE_CHARDEV_MEMORY`, `ringbuf` under its old name.
pub const TYPE_CHARDEV_MEMORY: &str = "chardev-memory";
/// `TYPE_CHARDEV_MUX`.
pub const TYPE_CHARDEV_MUX: &str = "chardev-mux";

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
    // On Unix QEMU puts the backends that work on descriptors under chardev-fd, on Windows
    // under chardev-win or chardev-win-stdio. Only the parent differs.
    #[cfg(unix)]
    let fd_parent = TYPE_CHARDEV_FD;
    #[cfg(not(unix))]
    let fd_parent = TYPE_CHARDEV;
    #[cfg(unix)]
    let fd_types = [
        TypeInfo::new(TYPE_CHARDEV_FD).parent(TYPE_CHARDEV).abstract_(),
        TypeInfo::new(TYPE_CHARDEV_PTY).parent(TYPE_CHARDEV),
    ];
    #[cfg(not(unix))]
    let fd_types: [TypeInfo; 0] = [];
    let types = [
        base,
        null,
        socket,
        TypeInfo::new(TYPE_CHARDEV_FILE).parent(fd_parent),
        TypeInfo::new(TYPE_CHARDEV_PIPE).parent(fd_parent),
        TypeInfo::new(TYPE_CHARDEV_STDIO).parent(fd_parent),
        TypeInfo::new(TYPE_CHARDEV_RINGBUF).parent(TYPE_CHARDEV),
        TypeInfo::new(TYPE_CHARDEV_MEMORY).parent(TYPE_CHARDEV_RINGBUF),
        TypeInfo::new(TYPE_CHARDEV_MUX).parent(TYPE_CHARDEV),
    ];
    registry.register_all(types.into_iter().chain(fd_types));
}

/// The object for `chr` under `/chardevs`.
pub(crate) fn add_object(registry: &Registry, chr: &Arc<Chardev>) -> Result<Object> {
    let obj = registry.object_new(chr.typename())?;
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
