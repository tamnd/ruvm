// SPDX-License-Identifier: GPL-2.0-or-later

//! The QAPI visitor interface from include/qapi/visitor.h and qapi/qapi-visit-core.c.
//!
//! A visitor walks a value one field at a time. Input visitors fill the value in from somewhere
//! else (a QObject tree, a command line string), output visitors turn it into something else.
//! QOM property accessors, QMP argument parsing and command line options all go through this
//! interface, so the error texts it produces are the ones users see.
//!
//! The Rust shape differs from the C one in how lists work. C passes the linked list itself so the
//! visitor can look at `next`. Here [`Visitor::start_list`] gets the number of elements and
//! [`Visitor::next_list`] the number still to come, which is all the output visitors need, and
//! input visitors answer whether there is another element. [`VisitorExt::visit_list`] drives both.

mod forward;
mod qobject_input;
mod qobject_output;
mod string_input;
mod string_output;

pub use forward::ForwardFieldVisitor;
pub use qobject_input::QObjectInputVisitor;
pub use qobject_output::QObjectOutputVisitor;
pub use string_input::{StringInputVisitor, parse_option_size, qapi_bool_parse};
pub use string_output::StringOutputVisitor;

use ruvm_base::{Error, ErrorClass, Result};

use crate::qvalue::{QType, QValue};

/// Whether a visitor reads into the value or out of it. QEMU's clone and dealloc visitors have no
/// counterpart because `Clone` and `Drop` do their jobs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisitorKind {
    Input,
    Output,
}

/// Bit for the `deprecated` special feature in a feature mask.
pub const QAPI_DEPRECATED: u64 = 1 << 0;
/// Bit for the `unstable` special feature in a feature mask.
pub const QAPI_UNSTABLE: u64 = 1 << 1;

/// `CompatPolicyInput`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompatPolicyInput {
    #[default]
    Accept,
    Reject,
    Crash,
}

/// `CompatPolicyOutput`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompatPolicyOutput {
    #[default]
    Accept,
    Hide,
}

/// `CompatPolicy`, set with `-compat`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompatPolicy {
    pub deprecated_input: CompatPolicyInput,
    pub deprecated_output: CompatPolicyOutput,
    pub unstable_input: CompatPolicyInput,
    pub unstable_output: CompatPolicyOutput,
}

fn compat_policy_input_ok1(
    adjective: &str,
    policy: CompatPolicyInput,
    class: ErrorClass,
    kind: &str,
    name: &str,
) -> Result<()> {
    match policy {
        CompatPolicyInput::Accept => Ok(()),
        CompatPolicyInput::Reject => {
            Err(Error::new(class, format!("{adjective} {kind} {name} disabled by policy")))
        }
        CompatPolicyInput::Crash => panic!("{adjective} {kind} {name} used with -compat crash"),
    }
}

/// `compat_policy_input_ok()`. `kind` is "command", "parameter" or "value".
pub fn compat_policy_input_ok(
    features: u64,
    policy: &CompatPolicy,
    class: ErrorClass,
    kind: &str,
    name: &str,
) -> Result<()> {
    if features & QAPI_DEPRECATED != 0 {
        compat_policy_input_ok1("Deprecated", policy.deprecated_input, class, kind, name)?;
    }
    if features & QAPI_UNSTABLE != 0 {
        compat_policy_input_ok1("Unstable", policy.unstable_input, class, kind, name)?;
    }
    Ok(())
}

/// Whether output tagged with `features` is hidden under `policy`, as the QObject output
/// visitor's `policy_skip` decides.
pub fn compat_policy_output_hidden(features: u64, policy: &CompatPolicy) -> bool {
    (features & QAPI_DEPRECATED != 0 && policy.deprecated_output == CompatPolicyOutput::Hide)
        || (features & QAPI_UNSTABLE != 0 && policy.unstable_output == CompatPolicyOutput::Hide)
}

/// `QEnumLookup`: the names of an enum's values in order, and the special features of each.
#[derive(Clone, Copy, Debug)]
pub struct QEnumLookup {
    pub array: &'static [&'static str],
    pub features: Option<&'static [u64]>,
}

impl QEnumLookup {
    pub const fn new(array: &'static [&'static str]) -> Self {
        QEnumLookup { array, features: None }
    }

    pub fn size(&self) -> usize {
        self.array.len()
    }

    /// `qapi_enum_lookup()`.
    pub fn lookup(&self, value: usize) -> &'static str {
        self.array[value]
    }

    /// `qapi_enum_parse()` without a default: the index of `name`, if it is one of the values.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.array.iter().position(|&s| s == name)
    }

    /// `qapi_enum_parse()` with an error, "invalid parameter value: %s".
    pub fn parse(&self, name: &str) -> Result<usize> {
        self.find(name).ok_or_else(|| Error::generic(format!("invalid parameter value: {name}")))
    }
}

/// `QERR_INVALID_PARAMETER_VALUE`, "Parameter '%s' expects %s".
pub(crate) fn invalid_parameter_value(name: &str, expected: &str) -> Error {
    Error::generic(format!("Parameter '{name}' expects {expected}"))
}

/// `QERR_MISSING_PARAMETER`.
pub(crate) fn missing_parameter(name: &str) -> Error {
    Error::generic(format!("Parameter '{name}' is missing"))
}

/// The table of operations a visitor implements, `struct Visitor` in visitor-impl.h.
///
/// `name` is the member name inside a struct and `None` at the top level and for list elements.
/// Methods with a default body are optional in QEMU too.
pub trait Visitor {
    fn kind(&self) -> VisitorKind;

    /// The `-compat` policy this visitor applies.
    fn policy(&self) -> CompatPolicy {
        CompatPolicy::default()
    }

    /// `visit_start_struct()`. Every successful call is paired with [`Visitor::end_struct`].
    fn start_struct(&mut self, name: Option<&str>) -> Result<()>;

    /// `visit_check_struct()`: whether the input had members nobody asked for.
    fn check_struct(&mut self) -> Result<()> {
        Ok(())
    }

    fn end_struct(&mut self);

    /// `visit_start_list()`. `len` is the number of elements an output visitor will see and is
    /// ignored by input visitors. The result says whether there is a first element to visit.
    fn start_list(&mut self, name: Option<&str>, len: usize) -> Result<bool>;

    /// `visit_next_list()`, called after each element. `remaining` is the number of elements
    /// after the one just visited. The result says whether to visit another.
    fn next_list(&mut self, remaining: usize) -> bool;

    /// `visit_check_list()`: whether the input had more elements than were visited.
    fn check_list(&mut self) -> Result<()> {
        Ok(())
    }

    fn end_list(&mut self);

    /// `visit_start_alternate()`. Input visitors say which kind of value is there, and output
    /// visitors return `None` because the caller already knows.
    fn start_alternate(&mut self, name: Option<&str>) -> Result<Option<QType>> {
        let _ = name;
        assert_eq!(self.kind(), VisitorKind::Output, "this input visitor cannot visit alternates");
        Ok(None)
    }

    fn end_alternate(&mut self) {}

    /// `visit_optional()`: whether an optional member is present. Output visitors keep the
    /// caller's answer.
    fn optional(&mut self, name: Option<&str>, present: bool) -> bool {
        let _ = name;
        present
    }

    /// `visit_policy_reject()`: an error if input tagged with `features` is not allowed.
    fn policy_reject(&mut self, name: Option<&str>, features: u64) -> Result<()> {
        let _ = (name, features);
        Ok(())
    }

    /// `visit_policy_skip()`: whether output tagged with `features` is hidden.
    fn policy_skip(&mut self, name: Option<&str>, features: u64) -> bool {
        let _ = (name, features);
        false
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()>;

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()>;

    /// `visit_type_size()`, which is `type_uint64` unless the visitor knows about suffixes.
    fn type_size(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        self.type_uint64(name, value)
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()>;

    fn type_str(&mut self, name: Option<&str>, value: &mut String) -> Result<()>;

    fn type_number(&mut self, name: Option<&str>, value: &mut f64) -> Result<()>;

    fn type_any(&mut self, name: Option<&str>, value: &mut QValue) -> Result<()> {
        let _ = (name, value);
        panic!("this visitor cannot visit 'any' values");
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()>;
}

fn int_n(
    v: &mut (impl Visitor + ?Sized),
    name: Option<&str>,
    value: i64,
    min: i64,
    max: i64,
    ty: &str,
) -> Result<i64> {
    let mut value = value;
    v.type_int64(name, &mut value)?;
    if value < min || value > max {
        return Err(invalid_parameter_value(name.unwrap_or("null"), ty));
    }
    Ok(value)
}

fn uint_n(
    v: &mut (impl Visitor + ?Sized),
    name: Option<&str>,
    value: u64,
    max: u64,
    ty: &str,
) -> Result<u64> {
    let mut value = value;
    v.type_uint64(name, &mut value)?;
    if value > max {
        return Err(invalid_parameter_value(name.unwrap_or("null"), ty));
    }
    Ok(value)
}

/// The helpers from qapi-visit-core.c that are built on top of the [`Visitor`] methods.
pub trait VisitorExt: Visitor {
    fn is_input(&self) -> bool {
        self.kind() == VisitorKind::Input
    }

    fn type_int8(&mut self, name: Option<&str>, value: &mut i8) -> Result<()> {
        *value =
            int_n(self, name, (*value).into(), i8::MIN.into(), i8::MAX.into(), "int8_t")? as i8;
        Ok(())
    }

    fn type_int16(&mut self, name: Option<&str>, value: &mut i16) -> Result<()> {
        let min = i16::MIN.into();
        *value = int_n(self, name, (*value).into(), min, i16::MAX.into(), "int16_t")? as i16;
        Ok(())
    }

    fn type_int32(&mut self, name: Option<&str>, value: &mut i32) -> Result<()> {
        let min = i32::MIN.into();
        *value = int_n(self, name, (*value).into(), min, i32::MAX.into(), "int32_t")? as i32;
        Ok(())
    }

    fn type_uint8(&mut self, name: Option<&str>, value: &mut u8) -> Result<()> {
        *value = uint_n(self, name, (*value).into(), u8::MAX.into(), "uint8_t")? as u8;
        Ok(())
    }

    fn type_uint16(&mut self, name: Option<&str>, value: &mut u16) -> Result<()> {
        *value = uint_n(self, name, (*value).into(), u16::MAX.into(), "uint16_t")? as u16;
        Ok(())
    }

    fn type_uint32(&mut self, name: Option<&str>, value: &mut u32) -> Result<()> {
        *value = uint_n(self, name, (*value).into(), u32::MAX.into(), "uint32_t")? as u32;
        Ok(())
    }

    /// `visit_type_enum()`: the value travels as its name.
    fn type_enum(
        &mut self,
        name: Option<&str>,
        value: &mut usize,
        lookup: &QEnumLookup,
    ) -> Result<()> {
        if !self.is_input() {
            let mut s = lookup.lookup(*value).to_string();
            return self.type_str(name, &mut s);
        }
        let mut s = String::new();
        self.type_str(name, &mut s)?;
        let Some(index) = lookup.find(&s) else {
            return Err(Error::generic(format!(
                "Parameter '{}' does not accept value '{s}'",
                name.unwrap_or("null")
            )));
        };
        if let Some(features) = lookup.features {
            let policy = self.policy();
            compat_policy_input_ok(
                features[index],
                &policy,
                ErrorClass::GenericError,
                "value",
                &s,
            )?;
        }
        *value = index;
        Ok(())
    }

    /// Visits a whole list the way generated `visit_type_FooList()` functions do. Input visitors
    /// replace the contents of `list`.
    fn visit_list<T: Default>(
        &mut self,
        name: Option<&str>,
        list: &mut Vec<T>,
        mut elem: impl FnMut(&mut Self, &mut T) -> Result<()>,
    ) -> Result<()> {
        let input = self.is_input();
        let len = if input { 0 } else { list.len() };
        if input {
            list.clear();
        }
        let mut more = self.start_list(name, len)?;
        let mut i = 0;
        let mut result = Ok(());
        while more && (input || i < len) {
            if input {
                let mut item = T::default();
                result = elem(self, &mut item);
                list.push(item);
            } else {
                result = elem(self, &mut list[i]);
            }
            if result.is_err() {
                break;
            }
            i += 1;
            more = self.next_list(len.saturating_sub(i));
        }
        if result.is_ok() {
            result = self.check_list();
        }
        self.end_list();
        if result.is_err() && input {
            list.clear();
        }
        result
    }
}

impl<V: Visitor + ?Sized> VisitorExt for V {}

#[cfg(test)]
mod tests;
