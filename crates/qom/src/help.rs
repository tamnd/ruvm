// SPDX-License-Identifier: GPL-2.0-or-later

//! The help text of `-object help` and `-object TYPE,help`.

use std::fmt::Write;

use ruvm_qapi::QValue;

use crate::TYPE_USER_CREATABLE;
use crate::types::Registry;

/// `object_property_help()`: one line of `TYPE,help` output.
pub fn object_property_help(
    name: &str,
    type_: &str,
    defval: Option<&QValue>,
    description: Option<&str>,
) -> String {
    let mut s = format!("  {name}=<{type_}>");
    if description.is_some() || defval.is_some() {
        if s.len() < 24 {
            s.push_str(&" ".repeat(24 - s.len()));
        }
        s.push_str(" - ");
    }
    if let Some(d) = description {
        s.push_str(d);
    }
    if let Some(v) = defval {
        let def = match v {
            QValue::Str(x) => x.clone(),
            QValue::Bool(b) => if *b { "on" } else { "off" }.to_string(),
            other => other.to_json(),
        };
        let _ = write!(s, " (default: {def})");
    }
    s
}

/// `user_creatable_print_types()`.
pub fn user_creatable_print_types(registry: &Registry) -> String {
    let mut out = String::from("List of user creatable objects:\n");
    for oc in registry.class_get_list_sorted(Some(TYPE_USER_CREATABLE), false) {
        let _ = writeln!(out, "  {}", oc.name());
    }
    out
}

/// `type_print_class_properties()`: the writable class properties of `type_`, sorted, or
/// `None` if there is no such type.
pub fn type_print_class_properties(registry: &Registry, type_: &str) -> Option<String> {
    let klass = registry.class_by_name(type_)?;
    let mut lines: Vec<String> = klass
        .properties()
        .iter()
        .filter(|p| p.is_writable())
        .map(|p| {
            object_property_help(
                p.name(),
                p.type_name(),
                p.default_value().as_ref(),
                p.get_description().as_deref(),
            )
        })
        .collect();
    lines.sort();
    let mut out = if lines.is_empty() {
        format!("There are no options for {type_}.\n")
    } else {
        format!("{type_} options:\n")
    };
    for l in lines {
        out.push_str(&l);
        out.push('\n');
    }
    Some(out)
}
