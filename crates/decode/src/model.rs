// SPDX-License-Identifier: GPL-2.0-or-later

//! The parsed form of a decodetree description: fields, argument sets, formats, patterns and the
//! groups that nest them. The shapes follow the classes in scripts/decodetree.py one for one, so
//! that the tree building and the output can follow the script line by line.

use std::collections::HashMap;
use std::rc::Rc;

/// One instruction field, the union of decodetree.py's `Field`, `MultiField`, `ConstField`,
/// `FunctionField`, `ParameterField` and `NamedField` classes.
#[derive(Debug)]
pub(crate) enum FieldDef {
    /// `pos:len` or `pos:slen`, a contiguous run of instruction bits.
    Simple { sign: bool, pos: u32, len: u32 },
    /// Several parts concatenated, the first part most significant.
    Multi { subs: Vec<Rc<FieldDef>>, mask: u64 },
    /// `name=value` in a pattern or format.
    Const(i128),
    /// `!function=f` applied to a field.
    Function { func: String, base: Rc<FieldDef> },
    /// `!function=f` with no bits, the value comes from calling `f(ctx)`.
    Parameter { func: String },
    /// `name:len`, a reference to another field of the same argument set.
    Named { name: String, sign: bool, len: u32 },
}

impl FieldDef {
    pub(crate) fn mask(&self) -> u64 {
        match self {
            FieldDef::Simple { pos, len, .. } => low_mask(*len).checked_shl(*pos).unwrap_or(0),
            FieldDef::Multi { mask, .. } => *mask,
            FieldDef::Function { base, .. } => base.mask(),
            FieldDef::Const(_) | FieldDef::Parameter { .. } | FieldDef::Named { .. } => 0,
        }
    }

    fn sign(&self) -> bool {
        match self {
            FieldDef::Simple { sign, .. } | FieldDef::Named { sign, .. } => *sign,
            FieldDef::Multi { subs, .. } => subs[0].sign(),
            FieldDef::Const(v) => *v < 0,
            FieldDef::Function { base, .. } => base.sign(),
            FieldDef::Parameter { .. } => false,
        }
    }

    fn class(&self) -> u8 {
        match self {
            FieldDef::Simple { .. } => 0,
            FieldDef::Multi { .. } => 1,
            FieldDef::Const(_) => 2,
            FieldDef::Function { .. } => 3,
            FieldDef::Parameter { .. } => 4,
            FieldDef::Named { .. } => 5,
        }
    }

    /// The names of other fields this one reads, `referenced_fields` in the script.
    pub(crate) fn referenced_fields(&self, out: &mut Vec<String>) {
        match self {
            FieldDef::Multi { subs, .. } => {
                for s in subs {
                    s.referenced_fields(out);
                }
            }
            FieldDef::Function { base, .. } => base.referenced_fields(out),
            FieldDef::Named { name, .. } => out.push(name.clone()),
            _ => {}
        }
    }

    /// Python's `a == b` for two field objects, with the script's `__eq__` methods. The places
    /// where the script would raise an AttributeError (comparing a MultiField or NamedField with
    /// a field of another class, which only happens through a FunctionField base) count as not
    /// equal. A ConstField has no `__eq__`, so it is only equal to itself, and two fields parsed
    /// from different lines are never the same object.
    pub(crate) fn py_eq(&self, other: &FieldDef) -> bool {
        match self {
            FieldDef::Simple { .. } => self.sign() == other.sign() && self.mask() == other.mask(),
            FieldDef::Multi { .. } => matches!(other, FieldDef::Multi { .. }) && !self.py_ne(other),
            FieldDef::Const(_) => std::ptr::eq(self, other),
            FieldDef::Function { func, base } => match other {
                FieldDef::Function { func: of, base: ob } => func == of && base.py_eq(ob),
                _ => false,
            },
            FieldDef::Parameter { func } => match other {
                FieldDef::Parameter { func: of } | FieldDef::Function { func: of, .. } => {
                    func == of
                }
                _ => false,
            },
            FieldDef::Named { name, .. } => match other {
                FieldDef::Named { name: on, .. } => name == on,
                _ => false,
            },
        }
    }

    fn py_ne(&self, other: &FieldDef) -> bool {
        match (self, other) {
            (FieldDef::Multi { subs: a, .. }, FieldDef::Multi { subs: b, .. }) => {
                if a.len() != b.len() {
                    return true;
                }
                a.iter().zip(b).any(|(x, y)| x.class() != y.class() || x.py_ne(y))
            }
            (FieldDef::Multi { .. }, _) => true,
            (FieldDef::Const(_), _) => !std::ptr::eq(self, other),
            _ => !self.py_eq(other),
        }
    }

    /// `a.__class__ != b.__class__ or a != b`, the comparison used for format fields.
    pub(crate) fn differs(&self, other: &FieldDef) -> bool {
        self.class() != other.class() || self.py_ne(other)
    }
}

pub(crate) fn low_mask(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1u64 << len) - 1 }
}

/// An ordered map of field name to field, standing in for the Python dicts keyed by name.
pub(crate) type Fields = Vec<(String, Rc<FieldDef>)>;

pub(crate) fn fields_get<'a>(fields: &'a Fields, name: &str) -> Option<&'a Rc<FieldDef>> {
    fields.iter().find(|(n, _)| n == name).map(|(_, f)| f)
}

/// `&name field field:type ... !extern`.
#[derive(Debug)]
pub(crate) struct ArgSet {
    pub(crate) name: String,
    pub(crate) fields: Vec<String>,
    pub(crate) types: Vec<String>,
    pub(crate) is_extern: bool,
}

/// The parts shared by formats and patterns, decodetree.py's `General`.
#[derive(Debug)]
pub(crate) struct General {
    pub(crate) name: String,
    pub(crate) file: String,
    pub(crate) lineno: usize,
    /// An argument set index for a format, a format index for a pattern.
    pub(crate) base: usize,
    pub(crate) fixedbits: u64,
    pub(crate) fixedmask: u64,
    pub(crate) undefmask: u64,
    pub(crate) fieldmask: u64,
    pub(crate) fields: Fields,
    pub(crate) width: u32,
}

impl General {
    /// Named references that no field of this format or pattern satisfies.
    pub(crate) fn dangling_references(&self) -> Vec<String> {
        let mut dangling = Vec::new();
        for (_, f) in &self.fields {
            let mut refs = Vec::new();
            f.referenced_fields(&mut refs);
            for r in refs {
                if fields_get(&self.fields, &r).is_none() {
                    dangling.push(r);
                }
            }
        }
        dangling
    }
}

/// A child of a group: a pattern, or a nested group of either kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Node {
    Pattern(usize),
    Group(usize),
}

/// A `{ }` (overlapping) or `[ ]` (non-overlapping) group, and the implicit top level.
#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) overlapping: bool,
    pub(crate) file: String,
    pub(crate) lineno: usize,
    pub(crate) pats: Vec<Node>,
    pub(crate) fixedbits: u64,
    pub(crate) fixedmask: u64,
    pub(crate) undefmask: u64,
    pub(crate) width: Option<u32>,
    pub(crate) tree: Option<Tree>,
}

/// A node of the decision tree built for a non-overlapping group.
#[derive(Debug)]
pub(crate) struct Tree {
    pub(crate) thismask: u64,
    pub(crate) subs: Vec<(u64, TreeSub)>,
    /// The format every leaf below shares, if they all share one.
    pub(crate) base: Option<usize>,
}

#[derive(Debug)]
pub(crate) enum TreeSub {
    Node(Node),
    Tree(Box<Tree>),
}

/// The whole description after parsing.
#[derive(Debug, Default)]
pub(crate) struct Model {
    pub(crate) fields: HashMap<String, Rc<FieldDef>>,
    pub(crate) arguments: Vec<ArgSet>,
    pub(crate) formats: Vec<General>,
    pub(crate) patterns: Vec<General>,
    pub(crate) groups: Vec<Group>,
    /// Patterns in the order they were parsed, `allpatterns` in the script.
    pub(crate) allpatterns: Vec<usize>,
    pub(crate) anyextern: bool,
}

impl Model {
    pub(crate) fn arg_index(&self, name: &str) -> Option<usize> {
        self.arguments.iter().position(|a| a.name == name)
    }

    pub(crate) fn format_index(&self, name: &str) -> Option<usize> {
        self.formats.iter().position(|f| f.name == name)
    }

    pub(crate) fn node_fixed(&self, n: Node) -> (u64, u64) {
        match n {
            Node::Pattern(p) => (self.patterns[p].fixedbits, self.patterns[p].fixedmask),
            Node::Group(g) => (self.groups[g].fixedbits, self.groups[g].fixedmask),
        }
    }

    pub(crate) fn node_file_line(&self, n: Node) -> (&str, usize) {
        match n {
            Node::Pattern(p) => (&self.patterns[p].file, self.patterns[p].lineno),
            Node::Group(g) => (&self.groups[g].file, self.groups[g].lineno),
        }
    }

    /// The argument set of a pattern, through its format.
    pub(crate) fn pattern_args(&self, p: usize) -> usize {
        self.formats[self.patterns[p].base].base
    }
}
