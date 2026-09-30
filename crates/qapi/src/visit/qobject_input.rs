// SPDX-License-Identifier: GPL-2.0-or-later

//! The QObject input visitor, qapi/qobject-input-visitor.c.

use ruvm_base::{Error, ErrorClass, Result};

use super::{
    CompatPolicy, Visitor, VisitorKind, compat_policy_input_ok, invalid_parameter_value,
    missing_parameter,
};
use crate::cutils;
use crate::ghash::GHashTable;
use crate::qvalue::{QType, QValue};

#[derive(Debug)]
struct StackObject {
    /// Name of `obj` in its parent.
    name: Option<String>,
    /// The dict or list being visited.
    obj: QValue,
    /// For a dict, the keys not visited yet, in a GHashTable because the first of them in its
    /// order is the one "is unexpected" names.
    unvisited: Option<GHashTable<()>>,
    /// For a list, the index of the next element, if there is one.
    entry: Option<usize>,
    /// For a list, the index of the element being visited. Starts at `u32::MAX` like the C
    /// `unsigned` that starts at -1.
    index: u32,
}

/// Reads a QAPI value out of a [`QValue`] tree.
///
/// The plain form wants JSON types to match: an integer member needs a JSON number. The keyval
/// form is for trees built by `keyval_parse()` from command line options, where every scalar is a
/// string that gets parsed as whatever the member needs.
#[derive(Debug)]
pub struct QObjectInputVisitor {
    root: QValue,
    keyval: bool,
    stack: Vec<StackObject>,
    policy: CompatPolicy,
}

impl QObjectInputVisitor {
    /// `qobject_input_visitor_new()`.
    pub fn new(root: QValue) -> Self {
        QObjectInputVisitor { root, keyval: false, stack: Vec::new(), policy: Default::default() }
    }

    /// `qobject_input_visitor_new_keyval()`.
    pub fn new_keyval(root: QValue) -> Self {
        QObjectInputVisitor { keyval: true, ..Self::new(root) }
    }

    /// `qobject_input_visitor_new_qmp()`: plain, with the global `-compat` policy.
    pub fn new_qmp(root: QValue, policy: CompatPolicy) -> Self {
        QObjectInputVisitor { policy, ..Self::new(root) }
    }

    /// `visit_set_policy()`.
    pub fn set_policy(&mut self, policy: CompatPolicy) {
        self.policy = policy;
    }

    /// `full_name_nth()`: the dotted path of what is being visited, for error messages. With `n`
    /// above zero it names the `n`th container from the top instead.
    fn full_name_nth(&self, name: Option<&str>, mut n: usize) -> String {
        let mut out = String::new();
        let mut name = name;
        for so in self.stack.iter().rev() {
            if n > 0 {
                n -= 1;
            } else if matches!(so.obj, QValue::Dict(_)) {
                out.insert_str(0, name.unwrap_or("<anonymous>"));
                out.insert(0, '.');
            } else if self.keyval {
                out.insert_str(0, &format!(".{}", so.index));
            } else {
                out.insert_str(0, &format!("[{}]", so.index));
            }
            name = so.name.as_deref();
        }
        assert_eq!(n, 0);
        if let Some(name) = name {
            out.insert_str(0, name);
        } else if out.starts_with('.') {
            out.remove(0);
        } else if out.is_empty() {
            return "<anonymous>".into();
        }
        out
    }

    fn full_name(&self, name: Option<&str>) -> String {
        self.full_name_nth(name, 0)
    }

    fn try_get_object(&mut self, name: Option<&str>, consume: bool) -> Option<QValue> {
        let Some(tos) = self.stack.last_mut() else {
            return Some(self.root.clone());
        };
        match &tos.obj {
            QValue::Dict(d) => {
                let name = name.expect("dict members have names");
                let ret = d.get(name).cloned();
                if consume && ret.is_some() {
                    if let Some(h) = &mut tos.unvisited {
                        let removed = h.remove(name);
                        assert!(removed.is_some());
                    }
                }
                ret
            }
            QValue::List(l) => {
                assert!(name.is_none(), "list elements have no names");
                let ret = tos.entry.map(|i| l[i].clone());
                if consume {
                    if let Some(i) = tos.entry {
                        tos.entry = (i + 1 < l.len()).then_some(i + 1);
                    }
                    tos.index = tos.index.wrapping_add(1);
                }
                ret
            }
            _ => unreachable!("only dicts and lists are pushed"),
        }
    }

    fn get_object(&mut self, name: Option<&str>, consume: bool) -> Result<QValue> {
        match self.try_get_object(name, consume) {
            Some(v) => Ok(v),
            None => Err(missing_parameter(&self.full_name(name))),
        }
    }

    fn get_keyval(&mut self, name: Option<&str>) -> Result<String> {
        match self.get_object(name, true)? {
            QValue::Str(s) => Ok(s),
            QValue::Dict(_) | QValue::List(_) => Err(Error::generic(format!(
                "Parameters '{}.*' are unexpected",
                self.full_name(name)
            ))),
            _ => Err(Error::generic(format!(
                "Internal error: parameter {} invalid",
                self.full_name(name)
            ))),
        }
    }

    fn invalid_type(&self, name: Option<&str>, expected: &str) -> Error {
        Error::generic(format!(
            "Invalid parameter type for '{}', expected: {expected}",
            self.full_name(name)
        ))
    }

    fn push(&mut self, name: Option<&str>, obj: QValue) -> bool {
        let (unvisited, entry) = match &obj {
            QValue::Dict(d) => {
                let mut h = GHashTable::new();
                for k in d.keys() {
                    h.insert(k, ());
                }
                (Some(h), None)
            }
            QValue::List(l) => (None, (!l.is_empty()).then_some(0)),
            _ => unreachable!(),
        };
        let has_entry = entry.is_some();
        self.stack.push(StackObject {
            name: name.map(str::to_string),
            obj,
            unvisited,
            entry,
            index: u32::MAX,
        });
        has_entry
    }
}

impl Visitor for QObjectInputVisitor {
    fn kind(&self) -> VisitorKind {
        VisitorKind::Input
    }

    fn policy(&self) -> CompatPolicy {
        self.policy
    }

    fn start_struct(&mut self, name: Option<&str>) -> Result<()> {
        let obj = self.get_object(name, true)?;
        if obj.qtype() != QType::QDict {
            return Err(self.invalid_type(name, "object"));
        }
        self.push(name, obj);
        Ok(())
    }

    fn check_struct(&mut self) -> Result<()> {
        let tos = self.stack.last().expect("inside a struct");
        let h = tos.unvisited.as_ref().expect("inside a struct");
        if let Some(key) = h.first_key() {
            let key = key.to_string();
            return Err(Error::generic(format!(
                "Parameter '{}' is unexpected",
                self.full_name(Some(&key))
            )));
        }
        Ok(())
    }

    fn end_struct(&mut self) {
        let tos = self.stack.pop().expect("inside a struct");
        assert!(tos.unvisited.is_some());
    }

    fn start_list(&mut self, name: Option<&str>, _len: usize) -> Result<bool> {
        let obj = self.get_object(name, true)?;
        if obj.qtype() != QType::QList {
            return Err(self.invalid_type(name, "array"));
        }
        Ok(self.push(name, obj))
    }

    fn next_list(&mut self, _remaining: usize) -> bool {
        let tos = self.stack.last().expect("inside a list");
        tos.entry.is_some()
    }

    fn check_list(&mut self) -> Result<()> {
        let tos = self.stack.last().expect("inside a list");
        if tos.entry.is_some() {
            return Err(Error::generic(format!(
                "Only {} list elements expected in {}",
                tos.index.wrapping_add(1),
                self.full_name_nth(None, 1)
            )));
        }
        Ok(())
    }

    fn end_list(&mut self) {
        let tos = self.stack.pop().expect("inside a list");
        assert!(tos.unvisited.is_none());
    }

    fn start_alternate(&mut self, name: Option<&str>) -> Result<Option<QType>> {
        Ok(Some(self.get_object(name, false)?.qtype()))
    }

    fn optional(&mut self, name: Option<&str>, _present: bool) -> bool {
        self.try_get_object(name, false).is_some()
    }

    fn policy_reject(&mut self, name: Option<&str>, features: u64) -> Result<()> {
        let name = name.unwrap_or("(null)");
        compat_policy_input_ok(features, &self.policy, ErrorClass::GenericError, "parameter", name)
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()> {
        if self.keyval {
            let s = self.get_keyval(name)?;
            *value = cutils::strtoi64(&s, 0, true)
                .map_err(|_| invalid_parameter_value(&self.full_name(name), "integer"))?
                .0;
            return Ok(());
        }
        match self.get_object(name, true)?.as_i64() {
            Some(v) => {
                *value = v;
                Ok(())
            }
            None => Err(self.invalid_type(name, "integer")),
        }
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        if self.keyval {
            let s = self.get_keyval(name)?;
            *value = cutils::strtou64(&s, 0, true)
                .map_err(|_| invalid_parameter_value(&self.full_name(name), "integer"))?
                .0;
            return Ok(());
        }
        let obj = self.get_object(name, true)?;
        // Negative numbers are accepted for backward compatibility.
        match obj.as_u64().or_else(|| obj.as_i64().map(|v| v as u64)) {
            Some(v) => {
                *value = v;
                Ok(())
            }
            None => Err(invalid_parameter_value(&self.full_name(name), "uint64")),
        }
    }

    fn type_size(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        if !self.keyval {
            return self.type_uint64(name, value);
        }
        let s = self.get_keyval(name)?;
        *value = cutils::strtosz(&s)
            .map_err(|_| invalid_parameter_value(&self.full_name(name), "size"))?;
        Ok(())
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()> {
        if self.keyval {
            let s = self.get_keyval(name)?;
            *value = cutils::bool_parse(&s)
                .ok_or_else(|| invalid_parameter_value(&self.full_name(name), "'on' or 'off'"))?;
            return Ok(());
        }
        match self.get_object(name, true)? {
            QValue::Bool(b) => {
                *value = b;
                Ok(())
            }
            _ => Err(self.invalid_type(name, "boolean")),
        }
    }

    fn type_str(&mut self, name: Option<&str>, value: &mut String) -> Result<()> {
        if self.keyval {
            *value = self.get_keyval(name)?;
            return Ok(());
        }
        match self.get_object(name, true)? {
            QValue::Str(s) => {
                *value = s;
                Ok(())
            }
            _ => Err(self.invalid_type(name, "string")),
        }
    }

    fn type_number(&mut self, name: Option<&str>, value: &mut f64) -> Result<()> {
        if self.keyval {
            let s = self.get_keyval(name)?;
            *value =
                cutils::strtod_finite(&s, true).map_err(|_| self.invalid_type(name, "number"))?.0;
            return Ok(());
        }
        match self.get_object(name, true)?.as_f64() {
            Some(v) => {
                *value = v;
                Ok(())
            }
            None => Err(self.invalid_type(name, "number")),
        }
    }

    fn type_any(&mut self, name: Option<&str>, value: &mut QValue) -> Result<()> {
        *value = self.get_object(name, true)?;
        Ok(())
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()> {
        match self.get_object(name, true)? {
            QValue::Null => Ok(()),
            _ => Err(self.invalid_type(name, "null")),
        }
    }
}
