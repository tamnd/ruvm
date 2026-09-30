// SPDX-License-Identifier: GPL-2.0-or-later

//! The QObject output visitor, qapi/qobject-output-visitor.c.

use ruvm_base::Result;

use super::{CompatPolicy, Visitor, VisitorKind, compat_policy_output_hidden};
use crate::qvalue::{QDict, QValue};

/// Builds a [`QValue`] tree from a QAPI value.
#[derive(Debug, Default)]
pub struct QObjectOutputVisitor {
    /// Unfinished containers, each with its name in the container below it.
    stack: Vec<(Option<String>, QValue)>,
    root: Option<QValue>,
    policy: CompatPolicy,
}

impl QObjectOutputVisitor {
    /// `qobject_output_visitor_new()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `qobject_output_visitor_new_qmp()`.
    pub fn new_qmp(policy: CompatPolicy) -> Self {
        QObjectOutputVisitor { policy, ..Self::default() }
    }

    /// `visit_complete()`: the finished tree. Panics if nothing was visited or a container is
    /// still open, as QEMU asserts.
    pub fn complete(self) -> QValue {
        assert!(self.stack.is_empty(), "visit_complete() with open containers");
        self.root.expect("visit_complete() before any visit")
    }

    fn add(&mut self, name: Option<&str>, value: QValue) {
        match self.stack.last_mut() {
            None => {
                assert!(self.root.is_none(), "an output visitor visits one root");
                self.root = Some(value);
            }
            Some((_, QValue::Dict(d))) => d.put(name.expect("dict members have names"), value),
            Some((_, QValue::List(l))) => {
                assert!(name.is_none(), "list elements have no names");
                l.push(value);
            }
            Some(_) => unreachable!("only dicts and lists are pushed"),
        }
    }

    fn pop(&mut self) {
        let (name, value) = self.stack.pop().expect("a container is open");
        self.add(name.as_deref(), value);
    }
}

impl Visitor for QObjectOutputVisitor {
    fn kind(&self) -> VisitorKind {
        VisitorKind::Output
    }

    fn policy(&self) -> CompatPolicy {
        self.policy
    }

    fn start_struct(&mut self, name: Option<&str>) -> Result<()> {
        self.stack.push((name.map(str::to_string), QValue::Dict(QDict::new())));
        Ok(())
    }

    fn end_struct(&mut self) {
        assert!(matches!(self.stack.last(), Some((_, QValue::Dict(_)))));
        self.pop();
    }

    fn start_list(&mut self, name: Option<&str>, len: usize) -> Result<bool> {
        self.stack.push((name.map(str::to_string), QValue::List(Vec::with_capacity(len))));
        Ok(len > 0)
    }

    fn next_list(&mut self, remaining: usize) -> bool {
        remaining > 0
    }

    fn end_list(&mut self) {
        assert!(matches!(self.stack.last(), Some((_, QValue::List(_)))));
        self.pop();
    }

    fn policy_skip(&mut self, _name: Option<&str>, features: u64) -> bool {
        compat_policy_output_hidden(features, &self.policy)
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()> {
        self.add(name, QValue::Int(*value));
        Ok(())
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        self.add(name, QValue::Uint(*value));
        Ok(())
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()> {
        self.add(name, QValue::Bool(*value));
        Ok(())
    }

    fn type_str(&mut self, name: Option<&str>, value: &mut String) -> Result<()> {
        self.add(name, QValue::Str(value.clone()));
        Ok(())
    }

    fn type_number(&mut self, name: Option<&str>, value: &mut f64) -> Result<()> {
        self.add(name, QValue::Double(*value));
        Ok(())
    }

    fn type_any(&mut self, name: Option<&str>, value: &mut QValue) -> Result<()> {
        self.add(name, value.clone());
        Ok(())
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()> {
        self.add(name, QValue::Null);
        Ok(())
    }
}
