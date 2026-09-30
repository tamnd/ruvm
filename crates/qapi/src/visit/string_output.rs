// SPDX-License-Identifier: GPL-2.0-or-later

//! The string output visitor, qapi/string-output-visitor.c.

use std::fmt::Write as _;

use ruvm_base::Result;

use super::{Visitor, VisitorKind};
use crate::cutils;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListMode {
    None,
    Started,
    InProgress,
    End,
}

/// A range of `Range` from qemu/range.h, inclusive at both ends and kept as unsigned values
/// because that is how `range_list_insert()` compares them.
#[derive(Clone, Copy, Debug)]
struct Range {
    lob: u64,
    upb: u64,
}

impl Range {
    /// `range_compare()`: -1 if `self` is below `b` with a gap, 1 if above with a gap, 0 if the
    /// two overlap or touch.
    fn compare(&self, b: &Range) -> i32 {
        if b.lob != 0 && b.lob - 1 > self.upb {
            -1
        } else if self.lob != 0 && self.lob - 1 > b.upb {
            1
        } else {
            0
        }
    }

    fn extend(&mut self, b: &Range) {
        self.lob = self.lob.min(b.lob);
        self.upb = self.upb.max(b.upb);
    }
}

/// `range_list_insert()`: inserts into a sorted list, merging ranges that overlap or touch.
fn range_list_insert(list: &mut Vec<Range>, r: Range) {
    let mut i = 0;
    while i < list.len() && list[i].compare(&r) < 0 {
        i += 1;
    }
    if i == list.len() || list[i].compare(&r) > 0 {
        list.insert(i, r);
        return;
    }
    list[i].extend(&r);
    while i + 1 < list.len() && list[i].compare(&list[i + 1]) == 0 {
        let next = list.remove(i + 1);
        list[i].extend(&next);
    }
}

/// Prints a value as a string, the way `object_property_print()` and `info qtree` show it.
///
/// Integer lists come out as ranges ("1-3,5"), and in human mode each value is followed by its
/// hex form in parentheses. Structs print as `<omitted>`.
#[derive(Debug)]
pub struct StringOutputVisitor {
    human: bool,
    string: String,
    list_mode: ListMode,
    range_start: i64,
    range_end: i64,
    ranges: Vec<Range>,
    struct_nesting: u32,
}

impl StringOutputVisitor {
    /// `string_output_visitor_new()`.
    pub fn new(human: bool) -> Self {
        StringOutputVisitor {
            human,
            string: String::new(),
            list_mode: ListMode::None,
            range_start: 0,
            range_end: 0,
            ranges: Vec::new(),
            struct_nesting: 0,
        }
    }

    /// `visit_complete()`.
    pub fn complete(self) -> String {
        self.string
    }

    fn set(&mut self, s: String) {
        match self.list_mode {
            ListMode::Started | ListMode::None => {
                if self.list_mode == ListMode::Started {
                    self.list_mode = ListMode::InProgress;
                }
                self.string = s;
            }
            ListMode::InProgress | ListMode::End => {
                self.string.push_str(", ");
                self.string.push_str(&s);
            }
        }
    }

    fn append(&mut self, a: i64) {
        self.append_range(a, a);
    }

    fn append_range(&mut self, s: i64, e: i64) {
        range_list_insert(&mut self.ranges, Range { lob: s as u64, upb: e as u64 });
    }

    fn format_ranges(&mut self, human: bool) {
        let n = self.ranges.len();
        for (i, r) in self.ranges.iter().enumerate() {
            let (lob, upb) = (r.lob, r.upb);
            let _ = match (lob != upb, human) {
                (true, true) => write!(self.string, "0x{lob:x}-0x{upb:x}"),
                (true, false) => write!(self.string, "{}-{}", lob as i64, upb as i64),
                (false, true) => write!(self.string, "0x{lob:x}"),
                (false, false) => write!(self.string, "{}", lob as i64),
            };
            if i + 1 < n {
                self.string.push(',');
            }
        }
    }

    fn flush_pending(&mut self) {
        if self.range_start == self.range_end {
            self.append(self.range_end);
        } else {
            assert!(self.range_start < self.range_end);
            self.append_range(self.range_start, self.range_end);
        }
    }
}

impl Visitor for StringOutputVisitor {
    fn kind(&self) -> VisitorKind {
        VisitorKind::Output
    }

    fn start_struct(&mut self, _name: Option<&str>) -> Result<()> {
        self.struct_nesting += 1;
        Ok(())
    }

    fn end_struct(&mut self) {
        self.struct_nesting -= 1;
        if self.struct_nesting == 0 {
            self.set("<omitted>".into());
        }
    }

    fn start_list(&mut self, _name: Option<&str>, len: usize) -> Result<bool> {
        if self.struct_nesting > 0 {
            return Ok(len > 0);
        }
        assert_eq!(self.list_mode, ListMode::None, "no lists in lists");
        // List handling is only needed when there are at least two elements.
        if len >= 2 {
            self.list_mode = ListMode::Started;
        }
        Ok(len > 0)
    }

    fn next_list(&mut self, remaining: usize) -> bool {
        if self.struct_nesting == 0 && remaining == 1 {
            self.list_mode = ListMode::End;
        }
        remaining > 0
    }

    fn end_list(&mut self) {
        if self.struct_nesting == 0 {
            self.list_mode = ListMode::None;
        }
    }

    fn type_int64(&mut self, _name: Option<&str>, value: &mut i64) -> Result<()> {
        if self.struct_nesting > 0 {
            return Ok(());
        }
        let v = *value;
        match self.list_mode {
            ListMode::None => self.append(v),
            ListMode::Started => {
                self.range_start = v;
                self.range_end = v;
                self.list_mode = ListMode::InProgress;
                return Ok(());
            }
            ListMode::InProgress => {
                if self.range_end.wrapping_add(1) == v {
                    self.range_end += 1;
                } else {
                    self.flush_pending();
                    self.range_start = v;
                    self.range_end = v;
                }
                return Ok(());
            }
            ListMode::End => {
                if self.range_end.wrapping_add(1) == v {
                    self.range_end += 1;
                    assert!(self.range_start < self.range_end);
                    self.append_range(self.range_start, self.range_end);
                } else {
                    self.flush_pending();
                    self.append(v);
                }
            }
        }
        self.format_ranges(false);
        if self.human {
            self.string.push_str(" (");
            self.format_ranges(true);
            self.string.push(')');
        }
        Ok(())
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        // QEMU prints values above INT64_MAX as negative numbers here, and so does this.
        let mut i = *value as i64;
        self.type_int64(name, &mut i)
    }

    fn type_size(&mut self, _name: Option<&str>, value: &mut u64) -> Result<()> {
        if self.struct_nesting > 0 {
            return Ok(());
        }
        let s = if self.human {
            format!("{} ({})", *value, cutils::size_to_str(*value))
        } else {
            value.to_string()
        };
        self.set(s);
        Ok(())
    }

    fn type_bool(&mut self, _name: Option<&str>, value: &mut bool) -> Result<()> {
        if self.struct_nesting == 0 {
            self.set(if *value { "true" } else { "false" }.into());
        }
        Ok(())
    }

    fn type_str(&mut self, _name: Option<&str>, value: &mut String) -> Result<()> {
        if self.struct_nesting == 0 {
            let s = if self.human { format!("\"{value}\"") } else { value.clone() };
            self.set(s);
        }
        Ok(())
    }

    fn type_number(&mut self, _name: Option<&str>, value: &mut f64) -> Result<()> {
        if self.struct_nesting == 0 {
            self.set(cutils::format_g(*value, 17));
        }
        Ok(())
    }

    fn type_null(&mut self, _name: Option<&str>) -> Result<()> {
        if self.struct_nesting == 0 {
            self.set(if self.human { "<null>" } else { "" }.into());
        }
        Ok(())
    }
}
