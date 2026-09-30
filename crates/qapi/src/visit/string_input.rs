// SPDX-License-Identifier: GPL-2.0-or-later

//! The string input visitor, qapi/string-input-visitor.c.

use ruvm_base::{Error, Result};

use super::{Visitor, VisitorKind, invalid_parameter_value};
use crate::cutils::{self, Errno};

/// At most this many elements come out of one range, so "0-9999999999" cannot eat all memory.
const RANGE_MAX_ELEMENTS: u64 = 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListMode {
    /// Not in a list.
    None,
    /// Some of the string is left to parse.
    Unparsed,
    /// In the middle of a signed range.
    Int64Range { next: i64, end: i64 },
    /// In the middle of an unsigned range.
    Uint64Range { next: u64, end: u64 },
    /// All of the string is used up.
    End,
}

/// Parses one scalar from a string, or a list of integers written as "1,3-5,8".
///
/// This is what QOM uses for `-global` values and for `object_property_parse()`.
#[derive(Debug)]
pub struct StringInputVisitor {
    string: String,
    unparsed: usize,
    lm: ListMode,
}

fn null_name(name: Option<&str>) -> &str {
    name.unwrap_or("null")
}

impl StringInputVisitor {
    /// `string_input_visitor_new()`.
    pub fn new(s: impl Into<String>) -> Self {
        StringInputVisitor { string: s.into(), unparsed: 0, lm: ListMode::None }
    }

    fn rest(&self) -> &str {
        &self.string[self.unparsed..]
    }

    /// Moves past the separator after an element: nothing at the end, or a comma.
    fn finish_entry(&mut self, at: usize) -> bool {
        match self.string.as_bytes().get(at) {
            None => {
                self.unparsed = at;
                true
            }
            Some(b',') => {
                self.unparsed = at + 1;
                true
            }
            Some(_) => false,
        }
    }

    fn try_parse_int64_list_entry(&mut self) -> bool {
        let base = self.unparsed;
        let Ok((start, used)) = cutils::strtoi64(self.rest(), 0, false) else {
            return false;
        };
        let at = base + used;
        let mut end = start;
        if self.string.as_bytes().get(at) == Some(&b'-') {
            let Ok((e, used)) = cutils::strtoi64(&self.string[at + 1..], 0, false) else {
                return false;
            };
            end = e;
            if start > end || end.wrapping_sub(start) as u64 >= RANGE_MAX_ELEMENTS {
                return false;
            }
            if !self.finish_entry(at + 1 + used) {
                return false;
            }
        } else if !self.finish_entry(at) {
            return false;
        }
        self.lm = ListMode::Int64Range { next: start, end };
        true
    }

    fn try_parse_uint64_list_entry(&mut self) -> bool {
        let base = self.unparsed;
        let Ok((start, used)) = cutils::strtou64(self.rest(), 0, false) else {
            return false;
        };
        let at = base + used;
        let mut end = start;
        if self.string.as_bytes().get(at) == Some(&b'-') {
            let Ok((e, used)) = cutils::strtou64(&self.string[at + 1..], 0, false) else {
                return false;
            };
            end = e;
            if start > end || end - start >= RANGE_MAX_ELEMENTS {
                return false;
            }
            if !self.finish_entry(at + 1 + used) {
                return false;
            }
        } else if !self.finish_entry(at) {
            return false;
        }
        self.lm = ListMode::Uint64Range { next: start, end };
        true
    }

    fn after_range(&self) -> ListMode {
        if self.unparsed < self.string.len() { ListMode::Unparsed } else { ListMode::End }
    }

    fn fewer_elements() -> Error {
        Error::generic("Fewer list elements expected")
    }
}

impl Visitor for StringInputVisitor {
    fn kind(&self) -> VisitorKind {
        VisitorKind::Input
    }

    fn start_struct(&mut self, _name: Option<&str>) -> Result<()> {
        panic!("the string input visitor cannot visit structs");
    }

    fn end_struct(&mut self) {
        panic!("the string input visitor cannot visit structs");
    }

    fn start_list(&mut self, _name: Option<&str>, _len: usize) -> Result<bool> {
        assert_eq!(self.lm, ListMode::None);
        self.unparsed = 0;
        if self.string.is_empty() {
            self.lm = ListMode::End;
            Ok(false)
        } else {
            self.lm = ListMode::Unparsed;
            Ok(true)
        }
    }

    fn next_list(&mut self, _remaining: usize) -> bool {
        match self.lm {
            ListMode::End => false,
            ListMode::None => panic!("next_list() outside a list"),
            _ => true,
        }
    }

    fn check_list(&mut self) -> Result<()> {
        match self.lm {
            ListMode::End => Ok(()),
            ListMode::None => panic!("check_list() outside a list"),
            _ => Err(Self::fewer_elements()),
        }
    }

    fn end_list(&mut self) {
        assert_ne!(self.lm, ListMode::None);
        self.unparsed = 0;
        self.lm = ListMode::None;
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()> {
        match self.lm {
            ListMode::None => {
                *value = cutils::strtoi64(&self.string, 0, true)
                    .map_err(|_| invalid_parameter_value(null_name(name), "int64"))?
                    .0;
                return Ok(());
            }
            ListMode::End => return Err(Self::fewer_elements()),
            ListMode::Unparsed => {
                if !self.try_parse_int64_list_entry() {
                    return Err(invalid_parameter_value(
                        null_name(name),
                        "list of int64 values or ranges",
                    ));
                }
            }
            ListMode::Int64Range { .. } => {}
            ListMode::Uint64Range { .. } => panic!("int64 visit in the middle of a uint64 range"),
        }
        let ListMode::Int64Range { next, end } = self.lm else { unreachable!() };
        *value = next;
        let next = next.wrapping_add(1);
        self.lm = if next > end || *value == i64::MAX {
            self.after_range()
        } else {
            ListMode::Int64Range { next, end }
        };
        Ok(())
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        match self.lm {
            ListMode::None => {
                *value = cutils::strtou64(&self.string, 0, true)
                    .map_err(|_| invalid_parameter_value(null_name(name), "uint64"))?
                    .0;
                return Ok(());
            }
            ListMode::End => return Err(Self::fewer_elements()),
            ListMode::Unparsed => {
                if !self.try_parse_uint64_list_entry() {
                    return Err(invalid_parameter_value(
                        null_name(name),
                        "list of uint64 values or ranges",
                    ));
                }
            }
            ListMode::Uint64Range { .. } => {}
            ListMode::Int64Range { .. } => panic!("uint64 visit in the middle of an int64 range"),
        }
        let ListMode::Uint64Range { next, end } = self.lm else { unreachable!() };
        *value = next;
        let next = next.wrapping_add(1);
        self.lm = if next > end || *value == u64::MAX {
            self.after_range()
        } else {
            ListMode::Uint64Range { next, end }
        };
        Ok(())
    }

    fn type_size(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        assert_eq!(self.lm, ListMode::None);
        *value = parse_option_size(name.unwrap_or("(null)"), &self.string)?;
        Ok(())
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()> {
        assert_eq!(self.lm, ListMode::None);
        *value = qapi_bool_parse(null_name(name), &self.string)?;
        Ok(())
    }

    fn type_str(&mut self, _name: Option<&str>, value: &mut String) -> Result<()> {
        assert_eq!(self.lm, ListMode::None);
        value.clone_from(&self.string);
        Ok(())
    }

    fn type_number(&mut self, name: Option<&str>, value: &mut f64) -> Result<()> {
        assert_eq!(self.lm, ListMode::None);
        *value = cutils::strtod_finite(&self.string, true)
            .map_err(|_| {
                Error::generic(format!(
                    "Invalid parameter type for '{}', expected: number",
                    null_name(name)
                ))
            })?
            .0;
        Ok(())
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()> {
        assert_eq!(self.lm, ListMode::None);
        if !self.string.is_empty() {
            return Err(Error::generic(format!(
                "Invalid parameter type for '{}', expected: null",
                null_name(name)
            )));
        }
        Ok(())
    }
}

/// `qapi_bool_parse()`.
pub fn qapi_bool_parse(name: &str, value: &str) -> Result<bool> {
    cutils::bool_parse(value).ok_or_else(|| invalid_parameter_value(name, "'on' or 'off'"))
}

/// `parse_option_size()` from util/qemu-option.c.
pub fn parse_option_size(name: &str, value: &str) -> Result<u64> {
    match cutils::strtosz(value) {
        Ok(v) => Ok(v),
        Err(Errno::Range) => {
            Err(Error::generic(format!("Value '{value}' is out of range for parameter '{name}'")))
        }
        Err(Errno::Inval) => Err(invalid_parameter_value(name, "a non-negative number below 2^64")
            .hint(
                "Optional suffix k, M, G, T, P or E means kilo-, mega-, giga-, tera-, peta-\n\
                 and exabytes, respectively.\n",
            )),
    }
}
