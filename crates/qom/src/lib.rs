// SPDX-License-Identifier: GPL-2.0-or-later

//! The QEMU object model in Rust: types, interfaces, properties, the composition tree, realize and compat properties.
//!
//! This is a port of qom/ from QEMU 11.1. The shape follows the C code closely because management
//! tools see the result: type names, the parent chain, property names and type strings, the order
//! `qom-list` prints properties in, canonical paths and every error text. What is different is
//! how a type keeps its state. A C type embeds its parent struct and casts pointers; here a type
//! hangs a value off the object with [`TypeInfo::instance_state`] and off the class with
//! [`ObjectClass::set_ext`], and code fetches them by Rust type.
//!
//! An [`Object`] handle is one QOM reference. Cloning it is `object_ref()`, dropping it is
//! `object_unref()`, and dropping the last one finalizes the object the way `object_finalize()`
//! does. A child property holds a reference to its child, so an object in the composition tree
//! stays alive until it is unparented.
//!
//! All types live in a [`Registry`]. QEMU has one global type table and one root object; the
//! emulator uses [`Registry::global`], and tests make their own so they do not see each other.

#![forbid(unsafe_code)]

mod class;
mod compat;
mod help;
mod link;
mod object;
mod property;
pub mod qmp;
mod types;
mod user_creatable;

pub use class::ObjectClass;
pub use compat::{CompatProps, GlobalProperty};
pub use help::{object_property_help, type_print_class_properties, user_creatable_print_types};
pub use link::{LinkCheck, LinkFlags, LinkSlot, allow_set_link};
pub use object::{Object, WeakObject};
pub use property::{
    Accessor, BoolGetter, BoolSetter, EnumGetter, EnumSetter, PropFlags, Property, Release,
    Resolver, StrGetter, StrSetter, Tm, TmGetter,
};
pub use types::{ClassHook, InstanceHook, Registry, TypeInfo};
pub use user_creatable::{UserCreatableClass, user_creatable_complete};

/// `TYPE_OBJECT`, the root of every instantiable type.
pub const TYPE_OBJECT: &str = "object";
/// `TYPE_INTERFACE`, the root of every interface.
pub const TYPE_INTERFACE: &str = "interface";
/// `TYPE_CONTAINER`, the type of `/objects`, `/machine/peripheral` and the like.
pub const TYPE_CONTAINER: &str = "container";
/// `TYPE_USER_CREATABLE`, the interface of types `-object` and `object-add` accept.
pub const TYPE_USER_CREATABLE: &str = "user-creatable";

fn lock<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests;
