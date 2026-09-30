// SPDX-License-Identifier: GPL-2.0-or-later

//! The schema model, ported from scripts/qapi/schema.py.
//!
//! Entities live in one list in definition order, as in `QAPISchema._entity_list`, and are looked
//! up by name through a map. Implicit types are made the same way QEMU makes them: `q_obj_NAME-arg`
//! for inline command and event arguments, `q_obj_NAME-base` for inline union bases, and `TList`
//! for `['T']`. The generators depend on that order and on those names.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::parser::{Error, Expression, Parser, Result, SourceInfo, Value};

/// A condition from an `'if'`: a config symbol, or `all`, `any` and `not` over conditions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    Name(String),
    Not(Box<Cond>),
    All(Vec<Cond>),
    Any(Vec<Cond>),
}

impl Cond {
    fn from_value(v: &Value, info: &SourceInfo) -> Result<Cond> {
        match v {
            Value::Str(s) => Ok(Cond::Name(s.clone())),
            Value::Dict(d) if d.len() == 1 => {
                let (op, arg) = &d[0];
                match op.as_str() {
                    "not" => Ok(Cond::Not(Box::new(Cond::from_value(arg, info)?))),
                    "all" | "any" => {
                        let Some(list) = arg.as_list() else {
                            return Err(Error::new(
                                info,
                                format!("'if' condition [{op}] of {op} must be an array"),
                            ));
                        };
                        let conds = list
                            .iter()
                            .map(|c| Cond::from_value(c, info))
                            .collect::<Result<Vec<_>>>()?;
                        Ok(if op == "all" { Cond::All(conds) } else { Cond::Any(conds) })
                    }
                    _ => {
                        Err(Error::new(info, format!("'if' condition has unknown operator '{op}'")))
                    }
                }
            }
            _ => Err(Error::new(info, "'if' condition must be a string or an object")),
        }
    }

    /// Evaluates the condition with `is_set` saying which symbols are defined.
    pub fn eval(&self, is_set: &dyn Fn(&str) -> bool) -> bool {
        match self {
            Cond::Name(n) => is_set(n),
            Cond::Not(c) => !c.eval(is_set),
            Cond::All(cs) => cs.iter().all(|c| c.eval(is_set)),
            Cond::Any(cs) => cs.iter().any(|c| c.eval(is_set)),
        }
    }

    /// Every symbol the condition mentions.
    pub fn names<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Cond::Name(n) => out.push(n),
            Cond::Not(c) => c.names(out),
            Cond::All(cs) | Cond::Any(cs) => cs.iter().for_each(|c| c.names(out)),
        }
    }

    /// `cgen_ifcond()`: the C preprocessor expression QEMU writes after `#if`.
    pub fn cgen(&self) -> String {
        fn render(c: &Cond, parens: bool) -> String {
            match c {
                Cond::Name(n) => format!("defined({n})"),
                Cond::Not(c) => format!("!{}", render(c, true)),
                Cond::All(cs) | Cond::Any(cs) => {
                    let op = if matches!(c, Cond::All(_)) { " && " } else { " || " };
                    let s = cs.iter().map(|c| render(c, true)).collect::<Vec<_>>().join(op);
                    if parens { format!("({s})") } else { s }
                }
            }
        }
        render(self, false)
    }
}

pub type IfCond = Option<Cond>;

fn ifcond(v: Option<&Value>, info: &SourceInfo) -> Result<IfCond> {
    v.map(|v| Cond::from_value(v, info)).transpose()
}

#[derive(Clone, Debug)]
pub struct Feature {
    pub name: String,
    pub ifcond: IfCond,
}

#[derive(Clone, Debug)]
pub struct EnumMember {
    pub name: String,
    pub ifcond: IfCond,
    pub features: Vec<Feature>,
}

#[derive(Clone, Debug)]
pub struct Member {
    pub name: String,
    pub typ: String,
    pub optional: bool,
    pub ifcond: IfCond,
    pub features: Vec<Feature>,
}

#[derive(Clone, Debug)]
pub struct Variant {
    pub name: String,
    pub typ: String,
    pub ifcond: IfCond,
}

#[derive(Clone, Debug)]
pub struct Branches {
    pub tag: String,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug)]
pub enum Kind {
    Include { module: String },
    Builtin { json_type: &'static str },
    Enum { members: Vec<EnumMember>, prefix: Option<String> },
    Array { element: String },
    Object { base: Option<String>, local_members: Vec<Member>, branches: Option<Branches> },
    Alternate { variants: Vec<Variant> },
    Command(Command),
    Event { arg_type: Option<String>, boxed: bool },
}

#[derive(Clone, Debug)]
pub struct Command {
    pub arg_type: Option<String>,
    pub ret_type: Option<String>,
    pub generate: bool,
    pub success_response: bool,
    pub boxed: bool,
    pub allow_oob: bool,
    pub allow_preconfig: bool,
    pub coroutine: bool,
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub name: String,
    pub info: Option<SourceInfo>,
    pub ifcond: IfCond,
    pub features: Vec<Feature>,
    pub kind: Kind,
}

impl Entity {
    pub fn is_type(&self) -> bool {
        !matches!(self.kind, Kind::Include { .. } | Kind::Command(_) | Kind::Event { .. })
    }

    pub fn meta(&self) -> &'static str {
        match self.kind {
            Kind::Include { .. } => "include",
            Kind::Builtin { .. } => "built-in",
            Kind::Enum { .. } => "enum",
            Kind::Array { .. } => "array",
            Kind::Object { .. } => "object",
            Kind::Alternate { .. } => "alternate",
            Kind::Command(_) => "command",
            Kind::Event { .. } => "event",
        }
    }
}

pub const BUILTIN_MODULE: &str = "./builtin";

/// `QAPISchema`.
#[derive(Debug)]
pub struct Schema {
    pub entities: Vec<Entity>,
    by_name: HashMap<String, usize>,
    schema_dir: PathBuf,
    /// Module names in the order they were first seen: the built-in module, the main file,
    /// then each included file.
    pub modules: Vec<String>,
    /// Every feature name, special ones first, in order of first use.
    pub feature_names: Vec<String>,
}

pub const SPECIAL_FEATURES: [&str; 2] = ["deprecated", "unstable"];

const BUILTINS: [(&str, &str); 15] = [
    ("str", "string"),
    ("number", "number"),
    ("int", "int"),
    ("int8", "int"),
    ("int16", "int"),
    ("int32", "int"),
    ("int64", "int"),
    ("uint8", "int"),
    ("uint16", "int"),
    ("uint32", "int"),
    ("uint64", "int"),
    ("size", "int"),
    ("bool", "boolean"),
    ("any", "value"),
    ("null", "null"),
];

impl Schema {
    /// Reads and checks the schema rooted at `fname`.
    pub fn load(fname: &Path) -> Result<Schema> {
        let exprs = Parser::parse_file(fname)?;
        let mut s = Schema {
            entities: Vec::new(),
            by_name: HashMap::new(),
            schema_dir: fname.parent().unwrap_or(Path::new("")).to_path_buf(),
            modules: Vec::new(),
            feature_names: SPECIAL_FEATURES.iter().map(|s| s.to_string()).collect(),
        };
        s.make_module(BUILTIN_MODULE.to_string());
        let main = s.module_name(fname);
        s.make_module(main);
        s.def_predefineds()?;
        for e in &exprs {
            s.def_expr(e)?;
        }
        s.check()?;
        s.add_implicit_variants()?;
        Ok(s)
    }

    pub fn lookup(&self, name: &str) -> Option<&Entity> {
        self.by_name.get(name).map(|&i| &self.entities[i])
    }

    fn module_name(&self, fname: &Path) -> String {
        let rel = fname.strip_prefix(&self.schema_dir).unwrap_or(fname);
        rel.to_string_lossy().into_owned()
    }

    fn make_module(&mut self, name: String) {
        if !self.modules.contains(&name) {
            self.modules.push(name);
        }
    }

    /// The module an entity belongs to, `set_module()` in QEMU.
    pub fn module_of(&self, e: &Entity) -> String {
        let info = match &e.kind {
            Kind::Array { element } => self.lookup(element).and_then(|t| t.info.as_ref()),
            _ => e.info.as_ref(),
        };
        match info {
            Some(i) => self.module_name(&i.fname),
            None => BUILTIN_MODULE.to_string(),
        }
    }

    fn def_entity(&mut self, e: Entity) -> Result<()> {
        if !matches!(e.kind, Kind::Include { .. }) {
            if let Some(&other) = self.by_name.get(&e.name) {
                let o = &self.entities[other];
                let info = e.info.as_ref().expect("user definitions have a location");
                let msg = match &o.info {
                    Some(oi) => {
                        format!("'{}' is already defined\n{}: previous definition", e.name, oi)
                    }
                    None => format!("built-in type '{}' is already defined", o.name),
                };
                return Err(Error::new(info, msg));
            }
            self.by_name.insert(e.name.clone(), self.entities.len());
        }
        self.entities.push(e);
        Ok(())
    }

    fn def_builtin(&mut self, name: &str, json_type: &'static str) -> Result<()> {
        self.def_entity(Entity {
            name: name.to_string(),
            info: None,
            ifcond: None,
            features: Vec::new(),
            kind: Kind::Builtin { json_type },
        })?;
        self.make_array_type(name, None)?;
        Ok(())
    }

    fn def_predefineds(&mut self) -> Result<()> {
        for (name, json) in BUILTINS {
            self.def_builtin(name, json)?;
        }
        self.def_entity(Entity {
            name: "q_empty".into(),
            info: None,
            ifcond: None,
            features: Vec::new(),
            kind: Kind::Object { base: None, local_members: Vec::new(), branches: None },
        })?;
        let members = ["none", "qnull", "qnum", "qstring", "qdict", "qlist", "qbool"]
            .iter()
            .map(|n| EnumMember { name: n.to_string(), ifcond: None, features: Vec::new() })
            .collect();
        self.def_entity(Entity {
            name: "QType".into(),
            info: None,
            ifcond: None,
            features: Vec::new(),
            kind: Kind::Enum { members, prefix: None },
        })
    }

    fn make_features(&mut self, v: Option<&Value>, info: &SourceInfo) -> Result<Vec<Feature>> {
        let Some(v) = v else {
            return Ok(Vec::new());
        };
        let Some(list) = v.as_list() else {
            return Err(Error::new(info, "'features' must be an array"));
        };
        let mut out = Vec::new();
        for f in list {
            let (name, cond) = match f {
                Value::Str(s) => (s.clone(), None),
                Value::Dict(_) => {
                    let Some(name) = f.get("name").and_then(Value::as_str) else {
                        return Err(Error::new(info, "feature requires a 'name'"));
                    };
                    (name.to_string(), ifcond(f.get("if"), info)?)
                }
                _ => return Err(Error::new(info, "'features' members must be strings or objects")),
            };
            if !self.feature_names.contains(&name) {
                self.feature_names.push(name.clone());
            }
            out.push(Feature { name, ifcond: cond });
        }
        Ok(out)
    }

    fn make_array_type(&mut self, element: &str, info: Option<&SourceInfo>) -> Result<String> {
        let name = format!("{element}List");
        if self.lookup(&name).is_none() {
            self.def_entity(Entity {
                name: name.clone(),
                info: info.cloned(),
                ifcond: None,
                features: Vec::new(),
                kind: Kind::Array { element: element.to_string() },
            })?;
        }
        Ok(name)
    }

    /// A type reference: `'T'` or `['T']`.
    fn type_ref(&mut self, v: &Value, info: &SourceInfo) -> Result<String> {
        match v {
            Value::Str(s) => Ok(s.clone()),
            Value::List(l) if l.len() == 1 => match &l[0] {
                Value::Str(s) => self.make_array_type(s, Some(info)),
                _ => Err(Error::new(info, "array type must be a string")),
            },
            _ => Err(Error::new(info, "type must be a string or a one element array")),
        }
    }

    fn make_members(&mut self, data: &Value, info: &SourceInfo) -> Result<Vec<Member>> {
        let Some(d) = data.as_dict() else {
            return Err(Error::new(info, "'data' must be an object"));
        };
        let mut out = Vec::new();
        for (key, value) in d {
            // normalize_members(): 'T' is short for {'type': 'T'}.
            let (typ, cond, feats) = match value {
                Value::Dict(_) => {
                    let Some(t) = value.get("type") else {
                        return Err(Error::new(info, format!("member '{key}' misses key 'type'")));
                    };
                    (t.clone(), ifcond(value.get("if"), info)?, value.get("features").cloned())
                }
                other => (other.clone(), None, None),
            };
            let typ = self.type_ref(&typ, info)?;
            let features = self.make_features(feats.as_ref(), info)?;
            let (name, optional) = match key.strip_prefix('*') {
                Some(n) => (n.to_string(), true),
                None => (key.clone(), false),
            };
            out.push(Member { name, typ, optional, ifcond: cond, features });
        }
        Ok(out)
    }

    fn make_variants(&mut self, data: &Value, info: &SourceInfo) -> Result<Vec<Variant>> {
        let Some(d) = data.as_dict() else {
            return Err(Error::new(info, "'data' must be an object"));
        };
        let mut out = Vec::new();
        for (key, value) in d {
            let (typ, cond) = match value {
                Value::Dict(_) => {
                    let Some(t) = value.get("type") else {
                        return Err(Error::new(info, format!("branch '{key}' misses key 'type'")));
                    };
                    (t.clone(), ifcond(value.get("if"), info)?)
                }
                other => (other.clone(), None),
            };
            let typ = self.type_ref(&typ, info)?;
            out.push(Variant { name: key.clone(), typ, ifcond: cond });
        }
        Ok(out)
    }

    fn make_implicit_object_type(
        &mut self,
        name: &str,
        info: &SourceInfo,
        cond: &IfCond,
        role: &str,
        members: Vec<Member>,
    ) -> Result<Option<String>> {
        if members.is_empty() {
            return Ok(None);
        }
        let name = format!("q_obj_{name}-{role}");
        if self.lookup(&name).is_none() {
            self.def_entity(Entity {
                name: name.clone(),
                info: Some(info.clone()),
                ifcond: cond.clone(),
                features: Vec::new(),
                kind: Kind::Object { base: None, local_members: members, branches: None },
            })?;
        }
        Ok(Some(name))
    }

    fn def_expr(&mut self, e: &Expression) -> Result<()> {
        const METAS: [&str; 6] = ["enum", "struct", "union", "alternate", "command", "event"];
        if let Some(inc) = e.get("include").and_then(Value::as_str) {
            let module = self.module_name(Path::new(inc));
            self.make_module(module.clone());
            return self.def_entity(Entity {
                name: String::new(),
                info: Some(e.info.clone()),
                ifcond: None,
                features: Vec::new(),
                kind: Kind::Include { module },
            });
        }
        let metas: Vec<&str> = METAS.iter().copied().filter(|m| e.has(m)).collect();
        let [meta] = metas[..] else {
            return Err(Error::new(
                &e.info,
                "expression must have exactly one key 'enum', 'struct', 'union', 'alternate', 'command', 'event'",
            ));
        };
        let Some(name) = e.get(meta).and_then(Value::as_str).map(str::to_string) else {
            return Err(Error::new(&e.info, format!("'{meta}' requires a string name")));
        };
        let mut info = e.info.clone();
        info.defn = Some((meta.to_string(), name.clone()));
        let info = &info;
        let cond = ifcond(e.get("if"), info)?;
        let features = self.make_features(e.get("features"), info)?;
        let flag =
            |key: &str, default: bool| e.get(key).and_then(Value::as_bool).unwrap_or(default);
        let empty = Value::Dict(Vec::new());
        let kind = match meta {
            "enum" => {
                let Some(data) = e.get("data").and_then(Value::as_list) else {
                    return Err(Error::new(info, "'data' must be an array"));
                };
                let mut members = Vec::new();
                for m in data {
                    let (mname, mcond, mfeat) = match m {
                        Value::Str(s) => (s.clone(), None, None),
                        Value::Dict(_) => (
                            m.get("name").and_then(Value::as_str).unwrap_or_default().to_string(),
                            ifcond(m.get("if"), info)?,
                            m.get("features"),
                        ),
                        _ => {
                            return Err(Error::new(
                                info,
                                "enum member must be a string or an object",
                            ));
                        }
                    };
                    let features = self.make_features(mfeat, info)?;
                    members.push(EnumMember { name: mname, ifcond: mcond, features });
                }
                Kind::Enum {
                    members,
                    prefix: e.get("prefix").and_then(Value::as_str).map(str::to_string),
                }
            }
            "struct" => {
                let local_members = self.make_members(e.get("data").unwrap_or(&empty), info)?;
                let base = e.get("base").and_then(Value::as_str).map(str::to_string);
                Kind::Object { base, local_members, branches: None }
            }
            "union" => {
                let base = match e.get("base") {
                    Some(Value::Str(s)) => Some(s.clone()),
                    Some(d @ Value::Dict(_)) => {
                        let members = self.make_members(d, info)?;
                        self.make_implicit_object_type(&name, info, &cond, "base", members)?
                    }
                    _ => return Err(Error::new(info, "union requires 'base'")),
                };
                let Some(tag) = e.get("discriminator").and_then(Value::as_str) else {
                    return Err(Error::new(info, "union requires 'discriminator'"));
                };
                let variants = self.make_variants(e.get("data").unwrap_or(&empty), info)?;
                Kind::Object {
                    base,
                    local_members: Vec::new(),
                    branches: Some(Branches { tag: tag.to_string(), variants }),
                }
            }
            "alternate" => Kind::Alternate {
                variants: self.make_variants(e.get("data").unwrap_or(&empty), info)?,
            },
            "command" => {
                let arg_type = match e.get("data") {
                    None => None,
                    Some(Value::Str(s)) => Some(s.clone()),
                    Some(d) => {
                        let members = self.make_members(d, info)?;
                        self.make_implicit_object_type(&name, info, &cond, "arg", members)?
                    }
                };
                let ret_type = e.get("returns").map(|r| self.type_ref(r, info)).transpose()?;
                Kind::Command(Command {
                    arg_type,
                    ret_type,
                    generate: flag("gen", true),
                    success_response: flag("success-response", true),
                    boxed: flag("boxed", false),
                    allow_oob: flag("allow-oob", false),
                    allow_preconfig: flag("allow-preconfig", false),
                    coroutine: flag("coroutine", false),
                })
            }
            "event" => {
                let arg_type = match e.get("data") {
                    None => None,
                    Some(Value::Str(s)) => Some(s.clone()),
                    Some(d) => {
                        let members = self.make_members(d, info)?;
                        self.make_implicit_object_type(&name, info, &cond, "arg", members)?
                    }
                };
                Kind::Event { arg_type, boxed: flag("boxed", false) }
            }
            _ => unreachable!("meta is one of METAS"),
        };
        self.def_entity(Entity { name, info: Some(info.clone()), ifcond: cond, features, kind })
    }

    /// Resolves every type reference, the part of `QAPISchema.check()` the generators rely on.
    fn check(&self) -> Result<()> {
        let resolve = |name: &str, info: &Option<SourceInfo>, what: &str| -> Result<()> {
            match self.lookup(name) {
                Some(t) if t.is_type() => Ok(()),
                _ => Err(Error::new(
                    info.as_ref().expect("built-ins always resolve"),
                    format!("{what} uses unknown type '{name}'"),
                )),
            }
        };
        for e in &self.entities {
            match &e.kind {
                Kind::Array { element } => resolve(element, &e.info, "array")?,
                Kind::Object { base, local_members, branches } => {
                    if let Some(b) = base {
                        resolve(b, &e.info, "'base'")?;
                    }
                    for m in local_members {
                        resolve(&m.typ, &e.info, &format!("member '{}'", m.name))?;
                    }
                    for v in branches.iter().flat_map(|b| &b.variants) {
                        resolve(&v.typ, &e.info, &format!("branch '{}'", v.name))?;
                    }
                }
                Kind::Alternate { variants } => {
                    for v in variants {
                        resolve(&v.typ, &e.info, &format!("branch '{}'", v.name))?;
                    }
                }
                Kind::Command(c) => {
                    for t in c.arg_type.iter().chain(&c.ret_type) {
                        resolve(t, &e.info, "command")?;
                    }
                }
                Kind::Event { arg_type: Some(t), .. } => resolve(t, &e.info, "event")?,
                _ => {}
            }
        }
        Ok(())
    }

    /// `QAPISchemaBranches.check()`: every value of the tag's enum that has no branch gets one of
    /// type `q_empty`, with the enum value's condition.
    fn add_implicit_variants(&mut self) -> Result<()> {
        let mut additions = Vec::new();
        for (i, e) in self.entities.iter().enumerate() {
            let Kind::Object { branches: Some(b), .. } = &e.kind else {
                continue;
            };
            let info = e.info.as_ref().expect("unions are user definitions");
            let Some(tag) = self.members(e).into_iter().find(|m| m.name == b.tag) else {
                return Err(Error::new(
                    info,
                    format!("discriminator '{}' is not a member of 'base'", b.tag),
                ));
            };
            let Some(Kind::Enum { members, .. }) = self.lookup(&tag.typ).map(|t| &t.kind) else {
                return Err(Error::new(
                    info,
                    format!("discriminator member '{}' must be of enum type", b.tag),
                ));
            };
            let extra: Vec<Variant> = members
                .iter()
                .filter(|m| !b.variants.iter().any(|v| v.name == m.name))
                .map(|m| Variant {
                    name: m.name.clone(),
                    typ: "q_empty".into(),
                    ifcond: m.ifcond.clone(),
                })
                .collect();
            additions.push((i, extra));
        }
        for (i, extra) in additions {
            if let Kind::Object { branches: Some(b), .. } = &mut self.entities[i].kind {
                b.variants.extend(extra);
            }
        }
        Ok(())
    }

    /// The type's condition. Arrays take their element's.
    pub fn ifcond_of(&self, e: &Entity) -> IfCond {
        match &e.kind {
            Kind::Array { element } => self.lookup(element).and_then(|t| self.ifcond_of(t)),
            _ => e.ifcond.clone(),
        }
    }

    /// `json_type()`.
    pub fn json_type(&self, e: &Entity) -> &'static str {
        match &e.kind {
            Kind::Builtin { json_type } => json_type,
            Kind::Enum { .. } => "string",
            Kind::Array { .. } => "array",
            Kind::Object { .. } => "object",
            Kind::Alternate { .. } => "value",
            _ => unreachable!("only types have a JSON type"),
        }
    }

    /// All members of an object type, the base's first, as `QAPISchemaObjectType.members`.
    pub fn members<'a>(&'a self, e: &'a Entity) -> Vec<&'a Member> {
        let Kind::Object { base, local_members, .. } = &e.kind else {
            return Vec::new();
        };
        let mut out = base
            .as_deref()
            .and_then(|b| self.lookup(b))
            .map(|b| self.members(b))
            .unwrap_or_default();
        out.extend(local_members);
        out
    }

    /// Entities grouped by module in module order, which is the order `QAPISchema.visit()` uses.
    pub fn visit_order(&self) -> Vec<&Entity> {
        let mut by_module: Vec<Vec<&Entity>> = vec![Vec::new(); self.modules.len()];
        for e in &self.entities {
            let m = self.module_of(e);
            let i = self.modules.iter().position(|x| *x == m).expect("every module is registered");
            by_module[i].push(e);
        }
        by_module.into_iter().flatten().collect()
    }
}
