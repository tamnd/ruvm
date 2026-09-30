// SPDX-License-Identifier: GPL-2.0-or-later

//! The options visitor of qapi/opts-visitor.c, which `-netdev`, `-nic` and `-net` go through.
//!
//! It reads a flat [`QemuOpts`] as if it were a QAPI struct. Nested structs, such as the branch
//! of a union, share the one namespace. Repeating a key makes a list, and for scalars the last
//! occurrence wins. Whatever is left over once the outermost struct is done is an error.
//!
//! Integer ranges such as `1-3` inside lists are not supported; nothing in the net options uses
//! them.

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

fn missing(name: &str) -> Error {
    Error::generic(format!("Parameter '{name}' is missing"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ListMode {
    None,
    InProgress,
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
            ListMode::InProgress => {
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
            ListMode::Traversed => false,
            ListMode::InProgress => {
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
        let (opt_name, s) = self.scalar(name)?;
        match strtoi64(&s, 0, true) {
            Ok((v, _)) => {
                *value = v;
                self.processed(name);
                Ok(())
            }
            Err(_) => Err(invalid_value(
                &opt_name,
                if self.list_mode == ListMode::None {
                    "an int64 value"
                } else {
                    "an int64 value or range"
                },
            )),
        }
    }

    fn type_uint64(&mut self, name: Option<&str>, value: &mut u64) -> Result<()> {
        let (opt_name, s) = self.scalar(name)?;
        match parse_uint(&s, 0, true) {
            Ok((v, _)) => {
                *value = v;
                self.processed(name);
                Ok(())
            }
            Err(_) => Err(invalid_value(
                &opt_name,
                if self.list_mode == ListMode::None {
                    "a uint64 value"
                } else {
                    "a uint64 value or range"
                },
            )),
        }
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
