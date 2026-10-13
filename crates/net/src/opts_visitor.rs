// SPDX-License-Identifier: GPL-2.0-or-later

//! The options visitor of qapi/opts-visitor.c, which `-netdev`, `-nic` and `-net` go through.
//!
//! It reads a flat [`QemuOpts`] as if it were a QAPI struct. Nested structs, such as the branch
//! of a union, share the one namespace. Repeating a key makes a list, and for scalars the last
//! occurrence wins. Whatever is left over once the outermost struct is done is an error.
//!
//! Inside a list an integer may be a range such as `1-3`, which gives each number in it.

use std::collections::VecDeque;

use ruvm_base::{Error, Result};
use ruvm_qapi::QValue;
use ruvm_qapi::cutils::{parse_uint, strtoi64, strtosz};
use ruvm_qapi::ghash::GHashTable;
use ruvm_qapi::opts::QemuOpts;
use ruvm_qapi::visit::qapi_bool_parse;
use ruvm_qapi::visit::{Visitor, VisitorKind};

fn invalid_value(name: &str, expected: &str) -> Error {
    Error::generic(format!("Parameter '{name}' expects {expected}"))
}

/// `OPTS_VISITOR_RANGE_MAX`: the most numbers one range may give.
const RANGE_MAX: u64 = 65536;

fn missing(name: &str) -> Error {
    Error::generic(format!("Parameter '{name}' is missing"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListMode {
    None,
    InProgress,
    /// In a range of signed numbers, `range_next` up to `range_limit`.
    SignedInterval,
    /// In a range of unsigned numbers.
    UnsignedInterval,
    Traversed,
}

/// `OptsVisitor`.
#[derive(Debug)]
pub struct OptsVisitor {
    /// Every option as (name, value), with the id first when there is one.
    opts: Vec<(String, String)>,
    unprocessed: GHashTable<VecDeque<usize>>,
    depth: usize,
    list_mode: ListMode,
    list_key: Option<String>,
    /// The number the range in progress gives next, as the bits of an `i64` or a `u64`.
    range_next: u64,
    range_limit: u64,
}

impl OptsVisitor {
    /// `opts_visitor_new()`.
    pub fn new(opts: &QemuOpts) -> Self {
        let mut all: Vec<(String, String)> =
            opts.opts().iter().map(|o| (o.name().to_string(), o.value().to_string())).collect();
        if let Some(id) = opts.id() {
            all.push(("id".to_string(), id.to_string()));
        }
        OptsVisitor {
            opts: all,
            unprocessed: GHashTable::new(),
            depth: 0,
            list_mode: ListMode::None,
            list_key: None,
            range_next: 0,
            range_limit: 0,
        }
    }

    fn name_of(name: Option<&str>) -> &str {
        name.unwrap_or("null")
    }

    /// `lookup_scalar()`: the option a scalar visit reads, as an index into `opts`.
    fn lookup_scalar(&self, name: Option<&str>) -> Result<usize> {
        match self.list_mode {
            ListMode::None => {
                let name = Self::name_of(name);
                match self.unprocessed.get(name).and_then(|q| q.back()) {
                    Some(&i) => Ok(i),
                    None => Err(missing(name)),
                }
            }
            ListMode::Traversed => Err(Error::generic("Fewer list elements than expected")),
            ListMode::InProgress | ListMode::SignedInterval | ListMode::UnsignedInterval => {
                let key = self.list_key.as_deref().unwrap_or_default();
                Ok(*self.unprocessed.get(key).and_then(|q| q.front()).expect("list in progress"))
            }
        }
    }

    /// `processed()`.
    fn processed(&mut self, name: Option<&str>) {
        if self.list_mode == ListMode::None {
            self.unprocessed.remove(Self::name_of(name));
        }
    }

    fn scalar(&self, name: Option<&str>) -> Result<(String, String)> {
        let i = self.lookup_scalar(name)?;
        Ok(self.opts[i].clone())
    }

    /// Drops the option the list is on, and gives whether another one with the same name
    /// follows it.
    fn next_option(&mut self) -> bool {
        let key = self.list_key.clone().unwrap_or_default();
        let empty = match self.unprocessed.get_mut(&key) {
            Some(q) => {
                q.pop_front();
                q.is_empty()
            }
            None => true,
        };
        if empty {
            self.unprocessed.remove(&key);
            self.list_key = None;
            self.list_mode = ListMode::Traversed;
            return false;
        }
        true
    }
}

impl Visitor for OptsVisitor {
    fn kind(&self) -> VisitorKind {
        VisitorKind::Input
    }

    fn start_struct(&mut self, _name: Option<&str>) -> Result<()> {
        self.depth += 1;
        if self.depth > 1 {
            return Ok(());
        }
        let mut table: GHashTable<VecDeque<usize>> = GHashTable::new();
        for (i, (name, _)) in self.opts.iter().enumerate() {
            match table.get_mut(name) {
                Some(q) => q.push_back(i),
                None => {
                    table.insert(name.clone(), VecDeque::from([i]));
                }
            }
        }
        self.unprocessed = table;
        Ok(())
    }

    fn check_struct(&mut self) -> Result<()> {
        if self.depth > 1 {
            return Ok(());
        }
        if let Some(key) = self.unprocessed.first_key() {
            let first = self.unprocessed.get(key).and_then(|q| q.front()).copied();
            let name = first.map_or(key, |i| self.opts[i].0.as_str());
            return Err(Error::generic(format!("Invalid parameter '{name}'")));
        }
        Ok(())
    }

    fn end_struct(&mut self) {
        self.depth = self.depth.saturating_sub(1);
        if self.depth == 0 {
            self.unprocessed = GHashTable::new();
        }
    }

    fn start_list(&mut self, name: Option<&str>, _len: usize) -> Result<bool> {
        assert_eq!(self.list_mode, ListMode::None, "a list inside a list");
        let key = Self::name_of(name);
        if !self.unprocessed.contains_key(key) {
            return Err(missing(key));
        }
        self.list_key = Some(key.to_string());
        self.list_mode = ListMode::InProgress;
        Ok(true)
    }

    fn next_list(&mut self, _remaining: usize) -> bool {
        match self.list_mode {
            ListMode::SignedInterval if (self.range_next as i64) < self.range_limit as i64 => {
                self.range_next = (self.range_next as i64 + 1) as u64;
                true
            }
            ListMode::UnsignedInterval if self.range_next < self.range_limit => {
                self.range_next += 1;
                true
            }
            // The range is done: go on to the next option.
            ListMode::SignedInterval | ListMode::UnsignedInterval => {
                self.list_mode = ListMode::InProgress;
                self.next_option()
            }
            ListMode::Traversed => false,
            ListMode::InProgress => self.next_option(),
            ListMode::None => panic!("next_list outside a list"),
        }
    }

    fn end_list(&mut self) {
        self.list_key = None;
        self.list_mode = ListMode::None;
    }

    fn optional(&mut self, name: Option<&str>, _present: bool) -> bool {
        assert_eq!(self.list_mode, ListMode::None, "optional member inside a list");
        self.unprocessed.contains_key(Self::name_of(name))
    }

    fn type_int64(&mut self, name: Option<&str>, value: &mut i64) -> Result<()> {
        if self.list_mode == ListMode::SignedInterval {
            *value = self.range_next as i64;
            return Ok(());
        }
        let (opt_name, s) = self.scalar(name)?;
        if let Ok((v, used)) = strtoi64(&s, 0, false) {
            if used == s.len() {
                *value = v;
                self.processed(name);
                return Ok(());
            }
            if s.as_bytes()[used] == b'-' && self.list_mode == ListMode::InProgress {
                // A range: "as if entering on the top".
                if let Ok((v2, _)) = strtoi64(&s[used + 1..], 0, true) {
                    if v <= v2 && (v2 as i128 - v as i128) < RANGE_MAX as i128 {
                        (self.range_next, self.range_limit) = (v as u64, v2 as u64);
                        self.list_mode = ListMode::SignedInterval;
                        *value = v;
                        return Ok(());
                    }
                }
            }
        }
        Err(invalid_value(
            &opt_name,
            if self.list_mode == ListMode::None {
                "an int64 value"
            } else {
                "an int64 value or range"
            },
        ))
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        if self.list_mode == ListMode::UnsignedInterval {
            *value = self.range_next;
            return Ok(());
        }
        let (opt_name, s) = self.scalar(name)?;
        if let Ok((v, used)) = parse_uint(&s, 0, false) {
            if used == s.len() {
                *value = v;
                self.processed(name);
                return Ok(());
            }
            if s.as_bytes()[used] == b'-' && self.list_mode == ListMode::InProgress {
                if let Ok((v2, _)) = parse_uint(&s[used + 1..], 0, true) {
                    if v <= v2 && v2 - v < RANGE_MAX {
                        (self.range_next, self.range_limit) = (v, v2);
                        self.list_mode = ListMode::UnsignedInterval;
                        *value = v;
                        return Ok(());
                    }
                }
            }
        }
        Err(invalid_value(
            &opt_name,
            if self.list_mode == ListMode::None {
                "a uint64 value"
            } else {
                "a uint64 value or range"
            },
        ))
    }

    fn type_size(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        let (opt_name, s) = self.scalar(name)?;
        match strtosz(&s) {
            Ok(v) => {
                *value = v;
                self.processed(name);
                Ok(())
            }
            Err(_) => Err(invalid_value(&opt_name, "a size value")),
        }
    }

    fn type_bool(&mut self, name: Option<&str>, value: &mut bool) -> Result<()> {
        let (opt_name, s) = self.scalar(name)?;
        *value = qapi_bool_parse(&opt_name, &s)?;
        self.processed(name);
        Ok(())
    }

    fn type_str(&mut self, name: Option<&str>, value: &mut String) -> Result<()> {
        let (_, s) = self.scalar(name)?;
        *value = s;
        self.processed(name);
        Ok(())
    }

    fn type_number(&mut self, name: Option<&str>, _value: &mut f64) -> Result<()> {
        Err(invalid_value(Self::name_of(name), "a number, which -netdev cannot give"))
    }

    fn type_null(&mut self, name: Option<&str>) -> Result<()> {
        Err(invalid_value(Self::name_of(name), "null, which -netdev cannot give"))
    }

    fn type_any(&mut self, name: Option<&str>, _value: &mut QValue) -> Result<()> {
        Err(invalid_value(Self::name_of(name), "a value -netdev cannot give"))
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::opts::QemuOptsList;
    use ruvm_qapi::types::{NumaCpuOptions, NumaOptions, NumaOptionsU};
    use ruvm_qapi::visit::Visit;

    use super::*;

    fn numa(arg: &str) -> Result<NumaOptions> {
        let mut list = QemuOptsList::new("numa", &[]).with_implied_opt_name("type");
        let opts = list.parse(arg, true)?;
        let mut o = NumaOptions::default();
        NumaOptions::visit(&mut OptsVisitor::new(opts), None, &mut o)?;
        Ok(o)
    }

    #[test]
    fn ranges_give_each_number() {
        let o = numa("node,cpus=0-2,cpus=5,cpus=7-8").unwrap();
        let NumaOptionsU::Node(n) = o.u else { panic!("{o:?}") };
        assert_eq!(n.cpus, Some(vec![0, 1, 2, 5, 7, 8]));
        let o = numa("cpu,node-id=1,socket-id=0").unwrap();
        let cpu = NumaCpuOptions { node_id: Some(1), socket_id: Some(0), ..Default::default() };
        assert_eq!(o.u, NumaOptionsU::Cpu(cpu));
    }

    #[test]
    fn bad_ranges() {
        let e = numa("node,cpus=3-1").unwrap_err();
        assert_eq!(e.message(), "Parameter 'cpus' expects a uint64 value or range");
        let e = numa("node,cpus=0-65536").unwrap_err();
        assert_eq!(e.message(), "Parameter 'cpus' expects a uint64 value or range");
        let e = numa("node,nodeid=1-2").unwrap_err();
        assert_eq!(e.message(), "Parameter 'nodeid' expects a uint64 value");
    }
}
