// SPDX-License-Identifier: GPL-2.0-or-later

//! The introspection generator, ported from scripts/qapi/introspect.py.
//!
//! QEMU turns the schema into a `QLitObject` tree with `#if` around conditional parts and
//! `query-qmp-schema` returns that tree. This module builds the same tree with the conditions
//! kept as data, so the caller can evaluate them for its configuration. Type names are masked
//! into decimal strings in order of first use, exactly as QEMU does without
//! `--unmask-non-abi-names`. The numbering happens before any condition is evaluated, so it is
//! the same for every configuration.

use std::collections::HashMap;

use crate::schema::{Entity, Feature, IfCond, Kind, Schema};

/// A node of the tree, QEMU's `QLitObject`.
#[derive(Clone, Debug, PartialEq)]
pub enum Lit {
    Null,
    Bool(bool),
    Str(String),
    List(Vec<Annotated>),
    /// Keys are kept in the order the generator adds them. QEMU sorts them when it prints the
    /// C literal, and the order they end up in a `QDict` is decided by the dict itself.
    Dict(Vec<(String, Lit)>),
}

/// A list element with the condition QEMU wraps it in.
#[derive(Clone, Debug, PartialEq)]
pub struct Annotated {
    pub value: Lit,
    pub ifcond: IfCond,
}

impl Annotated {
    fn new(value: Lit, ifcond: IfCond) -> Self {
        Annotated { value, ifcond }
    }

    fn plain(value: Lit) -> Self {
        Annotated { value, ifcond: None }
    }
}

fn s(v: &str) -> Lit {
    Lit::Str(v.to_string())
}

/// `QAPISchemaGenIntrospectVisitor`.
struct Visitor<'a> {
    schema: &'a Schema,
    unmask: bool,
    trees: Vec<Annotated>,
    used_types: Vec<&'a str>,
    name_map: HashMap<String, String>,
}

/// Builds the `SchemaInfo` list for `query-qmp-schema`.
pub fn introspect(schema: &Schema, unmask: bool) -> Vec<Annotated> {
    let mut v = Visitor {
        schema,
        unmask,
        trees: Vec::new(),
        used_types: Vec::new(),
        name_map: HashMap::new(),
    };
    for e in schema.visit_order() {
        // visit_needed(): types are only emitted once something uses them.
        if !e.is_type() {
            v.visit(e);
        }
    }
    let mut i = 0;
    while i < v.used_types.len() {
        let t = schema.lookup(v.used_types[i]).expect("used types exist");
        v.visit(t);
        i += 1;
    }
    v.trees
}

impl<'a> Visitor<'a> {
    fn name(&mut self, name: &str) -> String {
        if self.unmask {
            return name.to_string();
        }
        let n = self.name_map.len();
        self.name_map.entry(name.to_string()).or_insert_with(|| n.to_string()).clone()
    }

    fn use_type(&mut self, name: &'a str) -> String {
        let schema = self.schema;
        let mut typ = schema.lookup(name).expect("types are resolved by Schema::check");
        if schema.json_type(typ) == "int" {
            typ = schema.lookup("int").expect("built-in");
        } else if let Kind::Array { element } = &typ.kind {
            let el = schema.lookup(element).expect("resolved");
            if schema.json_type(el) == "int" {
                typ = schema.lookup("intList").expect("built-in");
            }
        }
        if !self.used_types.contains(&typ.name.as_str()) {
            self.used_types.push(&typ.name);
        }
        match &typ.kind {
            Kind::Builtin { .. } => typ.name.clone(),
            Kind::Array { element } => format!("[{}]", self.use_type(element)),
            _ => self.name(&typ.name),
        }
    }

    fn features(features: &[Feature]) -> Lit {
        Lit::List(features.iter().map(|f| Annotated::new(s(&f.name), f.ifcond.clone())).collect())
    }

    fn gen_tree(
        &mut self,
        name: &str,
        mtype: &str,
        mut obj: Vec<(String, Lit)>,
        ifcond: IfCond,
        features: &[Feature],
    ) {
        let name = match mtype {
            "command" | "event" | "builtin" | "array" => name.to_string(),
            _ => self.name(name),
        };
        obj.push(("name".into(), Lit::Str(name)));
        obj.push(("meta-type".into(), s(mtype)));
        if !features.is_empty() {
            obj.push(("features".into(), Self::features(features)));
        }
        self.trees.push(Annotated::new(Lit::Dict(obj), ifcond));
    }

    fn visit(&mut self, e: &'a Entity) {
        let schema = self.schema;
        match &e.kind {
            Kind::Include { .. } => {}
            Kind::Builtin { json_type } => {
                self.gen_tree(
                    &e.name,
                    "builtin",
                    vec![("json-type".into(), s(json_type))],
                    None,
                    &[],
                );
            }
            Kind::Enum { members, .. } => {
                let ms = members
                    .iter()
                    .map(|m| {
                        let mut obj = vec![("name".to_string(), s(&m.name))];
                        if !m.features.is_empty() {
                            obj.push(("features".into(), Self::features(&m.features)));
                        }
                        Annotated::new(Lit::Dict(obj), m.ifcond.clone())
                    })
                    .collect();
                let values =
                    members.iter().map(|m| Annotated::new(s(&m.name), m.ifcond.clone())).collect();
                let obj =
                    vec![("members".into(), Lit::List(ms)), ("values".into(), Lit::List(values))];
                self.gen_tree(&e.name, "enum", obj, e.ifcond.clone(), &e.features);
            }
            Kind::Array { element } => {
                let el = self.use_type(element);
                let cond = schema.ifcond_of(e);
                self.gen_tree(
                    &format!("[{el}]"),
                    "array",
                    vec![("element-type".into(), Lit::Str(el))],
                    cond,
                    &[],
                );
            }
            Kind::Object { branches, .. } => {
                let mut ms = Vec::new();
                for m in schema.members(e) {
                    let mut obj = vec![
                        ("name".to_string(), s(&m.name)),
                        ("type".into(), Lit::Str(self.use_type(&m.typ))),
                    ];
                    if m.optional {
                        obj.push(("default".into(), Lit::Null));
                    }
                    if !m.features.is_empty() {
                        obj.push(("features".into(), Self::features(&m.features)));
                    }
                    ms.push(Annotated::new(Lit::Dict(obj), m.ifcond.clone()));
                }
                let mut obj = vec![("members".to_string(), Lit::List(ms))];
                if let Some(b) = branches {
                    obj.push(("tag".into(), s(&b.tag)));
                    let mut vs = Vec::new();
                    for v in &b.variants {
                        let d = vec![
                            ("case".to_string(), s(&v.name)),
                            ("type".into(), Lit::Str(self.use_type(&v.typ))),
                        ];
                        vs.push(Annotated::new(Lit::Dict(d), v.ifcond.clone()));
                    }
                    obj.push(("variants".into(), Lit::List(vs)));
                }
                self.gen_tree(&e.name, "object", obj, e.ifcond.clone(), &e.features);
            }
            Kind::Alternate { variants } => {
                let mut ms = Vec::new();
                for v in variants {
                    let d = vec![("type".to_string(), Lit::Str(self.use_type(&v.typ)))];
                    ms.push(Annotated::new(Lit::Dict(d), v.ifcond.clone()));
                }
                self.gen_tree(
                    &e.name,
                    "alternate",
                    vec![("members".into(), Lit::List(ms))],
                    e.ifcond.clone(),
                    &e.features,
                );
            }
            Kind::Command(c) => {
                let arg = self.use_type(c.arg_type.as_deref().unwrap_or("q_empty"));
                let ret = self.use_type(c.ret_type.as_deref().unwrap_or("q_empty"));
                let mut obj = vec![
                    ("arg-type".to_string(), Lit::Str(arg)),
                    ("ret-type".into(), Lit::Str(ret)),
                ];
                if c.allow_oob {
                    obj.push(("allow-oob".into(), Lit::Bool(true)));
                }
                self.gen_tree(&e.name, "command", obj, e.ifcond.clone(), &e.features);
            }
            Kind::Event { arg_type, .. } => {
                let arg = self.use_type(arg_type.as_deref().unwrap_or("q_empty"));
                self.gen_tree(
                    &e.name,
                    "event",
                    vec![("arg-type".into(), Lit::Str(arg))],
                    e.ifcond.clone(),
                    &e.features,
                );
            }
        }
    }
}

/// Drops everything whose condition is false, the job the C preprocessor does in QEMU.
pub fn resolve(trees: &[Annotated], is_set: &dyn Fn(&str) -> bool) -> Vec<Lit> {
    trees
        .iter()
        .filter(|a| a.ifcond.as_ref().is_none_or(|c| c.eval(is_set)))
        .map(|a| resolve_lit(&a.value, is_set))
        .collect()
}

fn resolve_lit(l: &Lit, is_set: &dyn Fn(&str) -> bool) -> Lit {
    match l {
        Lit::List(items) => {
            Lit::List(resolve(items, is_set).into_iter().map(Annotated::plain).collect())
        }
        Lit::Dict(d) => {
            Lit::Dict(d.iter().map(|(k, v)| (k.clone(), resolve_lit(v, is_set))).collect())
        }
        other => other.clone(),
    }
}

/// Every config symbol a tree depends on, sorted and without duplicates.
pub fn symbols(trees: &[Annotated]) -> Vec<String> {
    fn walk<'a>(a: &'a Annotated, out: &mut Vec<&'a str>) {
        if let Some(c) = &a.ifcond {
            c.names(out);
        }
        lit(&a.value, out);
    }
    fn lit<'a>(l: &'a Lit, out: &mut Vec<&'a str>) {
        match l {
            Lit::List(items) => items.iter().for_each(|a| walk(a, out)),
            Lit::Dict(d) => d.iter().for_each(|(_, v)| lit(v, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    trees.iter().for_each(|a| walk(a, &mut out));
    let mut v: Vec<String> = out.into_iter().map(str::to_string).collect();
    v.sort();
    v.dedup();
    v
}
