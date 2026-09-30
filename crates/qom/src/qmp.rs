// SPDX-License-Identifier: GPL-2.0-or-later

//! The QOM commands of qom/qom-qmp-cmds.c. Each takes its arguments already unpacked and returns
//! the reply as a [`QValue`] built the way the generated QAPI output visitor builds it.

use std::sync::Arc;

use ruvm_base::{Error, ErrorClass, Result};
use ruvm_qapi::{QDict, QValue};

use crate::TYPE_OBJECT;
use crate::object::Object;
use crate::property::Property;
use crate::types::Registry;

/// `qom_resolve_path()`.
fn resolve(registry: &Registry, path: &str) -> Result<Object> {
    match registry.resolve_path(path) {
        (Some(o), _) => Ok(o),
        (None, true) => Err(Error::generic(format!("Path '{path}' is ambiguous"))),
        (None, false) => {
            Err(Error::new(ErrorClass::DeviceNotFound, format!("Device '{path}' not found")))
        }
    }
}

fn not_found(path: &str) -> Error {
    Error::new(ErrorClass::DeviceNotFound, format!("Device '{path}' not found"))
}

/// `ObjectPropertyInfo` with only the name and the type.
fn info_short(p: &Property) -> QValue {
    QValue::Dict(QDict::new().with("name", p.name()).with("type", p.type_name()))
}

/// A full `ObjectPropertyInfo`, with the description and the default when there are any.
fn info_full(p: &Property) -> QValue {
    let mut d = QDict::new().with("name", p.name()).with("type", p.type_name());
    if let Some(desc) = p.get_description() {
        d.put("description", desc);
    }
    if let Some(v) = p.default_value() {
        d.put("default-value", v);
    }
    QValue::Dict(d)
}

/// The C code prepends each property to the list, so replies come out in reverse iteration
/// order.
fn reversed(props: Vec<Arc<Property>>) -> impl Iterator<Item = Arc<Property>> {
    props.into_iter().rev()
}

/// `qom-list`.
pub fn qom_list(registry: &Registry, path: &str) -> Result<QValue> {
    let obj = resolve(registry, path)?;
    Ok(QValue::List(reversed(obj.properties()).map(|p| info_short(&p)).collect()))
}

/// `qom-list-get`.
pub fn qom_list_get(registry: &Registry, paths: &[String]) -> Result<QValue> {
    let mut out = Vec::new();
    for path in paths {
        let obj = resolve(registry, path)?;
        let props = reversed(obj.properties())
            .map(|p| {
                let mut d = QDict::new().with("name", p.name()).with("type", p.type_name());
                if let Ok(v) = obj.property_get_qobject(p.name()) {
                    d.put("value", v);
                }
                QValue::Dict(d)
            })
            .collect();
        out.push(QValue::Dict(QDict::new().with("properties", QValue::List(props))));
    }
    Ok(QValue::List(out))
}

/// `qom-get`.
pub fn qom_get(registry: &Registry, path: &str, property: &str) -> Result<QValue> {
    let obj = registry.resolve_path(path).0.ok_or_else(|| not_found(path))?;
    obj.property_get_qobject(property)
}

/// `qom-set`.
pub fn qom_set(registry: &Registry, path: &str, property: &str, value: QValue) -> Result<()> {
    let obj = registry.resolve_path(path).0.ok_or_else(|| not_found(path))?;
    obj.property_set_qobject(property, value)
}

/// `qom-list-types`.
pub fn qom_list_types(registry: &Registry, implements: Option<&str>, abstract_: bool) -> QValue {
    let mut out = Vec::new();
    registry.class_foreach(implements, abstract_, |k| {
        let mut d = QDict::new().with("name", k.name()).with("abstract", k.is_abstract());
        if let Some(p) = k.parent() {
            d.put("parent", p.name());
        }
        out.push(QValue::Dict(d));
    });
    out.reverse();
    QValue::List(out)
}

/// `device-list-properties`, without the properties every device has.
pub fn device_list_properties(registry: &Registry, typename: &str) -> Result<QValue> {
    let Some(klass) = registry.class_by_name(typename) else {
        return Err(Error::new(
            ErrorClass::DeviceNotFound,
            format!("Device '{typename}' not found"),
        ));
    };
    if klass.dynamic_cast("device").is_none() || klass.is_abstract() {
        return Err(Error::generic("Parameter 'typename' expects a non-abstract device type"));
    }
    let obj = Object::new_with_class(&klass)?;
    let skip = ["type", "realized", "hotpluggable", "hotplugged", "parent_bus"];
    let list = reversed(obj.properties())
        .filter(|p| !skip.contains(&p.name()))
        .map(|p| info_full(&p))
        .collect();
    Ok(QValue::List(list))
}

/// `qom-list-properties`. Abstract types list their class properties only.
pub fn qom_list_properties(registry: &Registry, typename: &str) -> Result<QValue> {
    let Some(klass) = registry.class_by_name(typename) else {
        return Err(Error::new(
            ErrorClass::DeviceNotFound,
            format!("Class '{typename}' not found"),
        ));
    };
    if klass.dynamic_cast(TYPE_OBJECT).is_none() {
        return Err(Error::generic("Parameter 'typename' expects a QOM type"));
    }
    let props = if klass.is_abstract() {
        klass.properties()
    } else {
        Object::new_with_class(&klass)?.properties()
    };
    Ok(QValue::List(reversed(props).map(|p| info_full(&p)).collect()))
}

/// `object-add`, with the arguments as one dictionary.
pub fn object_add(registry: &Registry, args: &QDict) -> Result<()> {
    registry.user_creatable_add(args, false).map(|_| ())
}

/// `object-del`.
pub fn object_del(registry: &Registry, id: &str) -> Result<()> {
    registry.user_creatable_del(id)
}
