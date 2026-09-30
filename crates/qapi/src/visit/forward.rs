// SPDX-License-Identifier: GPL-2.0-or-later

//! The forward field visitor, qapi/qapi-forward-visitor.c.

use std::fmt;

use ruvm_base::Result;

use super::{CompatPolicy, Visitor, VisitorKind, missing_parameter};
use crate::qvalue::{QType, QValue};

/// Passes a visit through to another visitor with the top level member renamed from `from` to
/// `to`. QOM alias properties use it so that setting the alias reads the value under the alias's
/// name and hands it to the target property under the target's name.
pub struct ForwardFieldVisitor<'a> {
    target: &'a mut dyn Visitor,
    from: String,
    to: String,
    depth: u32,
}

impl fmt::Debug for ForwardFieldVisitor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardFieldVisitor")
            .field("from", &self.from)
            .field("to", &self.to)
            .field("depth", &self.depth)
            .finish_non_exhaustive()
    }
}

impl<'a> ForwardFieldVisitor<'a> {
    /// `visitor_forward_field()`.
    pub fn new(target: &'a mut dyn Visitor, from: &str, to: &str) -> Self {
        ForwardFieldVisitor { target, from: from.into(), to: to.into(), depth: 0 }
    }

    /// The target and the name to use on it. Below the top level names pass through unchanged,
    /// and at the top level only `from` is known.
    fn parts<'n>(
        &'n mut self,
        name: Option<&'n str>,
    ) -> Result<(&'n mut dyn Visitor, Option<&'n str>)> {
        let to = if self.depth > 0 {
            name
        } else {
            match name {
                Some(n) if n == self.from => Some(self.to.as_str()),
                _ => return Err(missing_parameter(name.unwrap_or("(null)"))),
            }
        };
        Ok((&mut *self.target, to))
    }
}

impl Visitor for ForwardFieldVisitor<'_> {
    fn kind(&self) -> VisitorKind {
        self.target.kind()
    }

    fn policy(&self) -> CompatPolicy {
        self.target.policy()
    }

    fn start_struct(&mut self, name: Option<&str>) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.start_struct(n)?;
        self.depth += 1;
        Ok(())
    }

    fn check_struct(&mut self) -> Result<()> {
        self.target.check_struct()
    }

    fn end_struct(&mut self) {
        assert!(self.depth > 0);
        self.depth -= 1;
        self.target.end_struct();
    }

    fn start_list(&mut self, name: Option<&str>, len: usize) -> Result<bool> {
        let (t, n) = self.parts(name)?;
        let r = t.start_list(n, len);
        self.depth += 1;
        r
    }

    fn next_list(&mut self, remaining: usize) -> bool {
        assert!(self.depth > 0);
        self.target.next_list(remaining)
    }

    fn check_list(&mut self) -> Result<()> {
        assert!(self.depth > 0);
        self.target.check_list()
    }

    fn end_list(&mut self) {
        assert!(self.depth > 0);
        self.depth -= 1;
        self.target.end_list();
    }

    fn start_alternate(&mut self, name: Option<&str>) -> Result<Option<QType>> {
        // The alternate's content is visited under the same name, so depth stays put.
        let (t, n) = self.parts(name)?;
        t.start_alternate(n)
    }

    fn end_alternate(&mut self) {
        self.target.end_alternate();
    }

    fn optional(&mut self, name: Option<&str>, present: bool) -> bool {
        match self.parts(name) {
            Ok((t, n)) => t.optional(n, present),
            Err(_) => false,
        }
    }

    fn policy_reject(&mut self, name: Option<&str>, features: u64) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.policy_reject(n, features)
    }

    fn policy_skip(&mut self, name: Option<&str>, features: u64) -> bool {
        match self.parts(name) {
            Ok((t, n)) => t.policy_skip(n, features),
            Err(_) => true,
        }
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_int64(n, value)
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_uint64(n, value)
    }

    fn type_size(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_size(n, value)
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_bool(n, value)
    }

    fn type_str(&mut self, name: Option<&str>, value: &mut String) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_str(n, value)
    }

    fn type_number(&mut self, name: Option<&str>, value: &mut f64) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_number(n, value)
    }

    fn type_any(&mut self, name: Option<&str>, value: &mut QValue) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_any(n, value)
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()> {
        let (t, n) = self.parts(name)?;
        t.type_null(n)
    }
}
