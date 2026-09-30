// SPDX-License-Identifier: GPL-2.0-or-later

//! The Rust type generator, the counterpart of scripts/qapi/types.py and visit.py.
//!
//! Every enum, struct, union and alternate in the schema becomes a Rust type with a `Visit`
//! implementation that makes the same visitor calls, in the same order, as the C function QEMU
//! generates. That order is what decides which error a bad QMP argument gets, so it is kept even
//! where a Rust program would do it differently.
//!
//! The mapping:
//!
//! - An enum becomes a fieldless Rust enum with a `LOOKUP` table. Values whose condition is off
//!   are left out, as the C preprocessor leaves them out.
//! - A struct becomes a Rust struct with the base's members first. Optional members are `Option`.
//! - A union becomes a struct with the common members and a field `u` that holds the branch. The
//!   tag is not stored separately: `u` knows it, so the two cannot disagree.
//! - An alternate becomes an enum with one variant per alternative.
//! - Arrays are `Vec`, and the built-in types are Rust's own, with `any` as `QValue`.
//!
//! A member whose type contains the struct it is in, directly or through other members, is boxed.
//! Implicit types get names from their role: `q_obj_qom-list-arg` is `QomListArg`.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::schema::{Entity, Feature, IfCond, Kind, Member, Schema, Variant};

/// Names the generated module imports, which no generated type may take.
const RESERVED: &[&str] = &[
    "Box",
    "Default",
    "Error",
    "Option",
    "QEnumLookup",
    "QType",
    "QValue",
    "Result",
    "String",
    "Vec",
    "Visit",
    "Visitor",
    "VisitorExt",
];

const KEYWORDS: &[&str] = &[
    "abstract", "as", "async", "await", "become", "box", "break", "const", "continue", "crate",
    "do", "dyn", "else", "enum", "extern", "false", "final", "fn", "for", "gen", "if", "impl",
    "in", "let", "loop", "macro", "match", "mod", "move", "mut", "override", "priv", "pub", "ref",
    "return", "self", "static", "struct", "super", "trait", "true", "try", "type", "typeof",
    "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
];

/// Joins the parts of a QAPI name into a Rust type or variant name. Each part gets an upper case
/// first letter, and two parts that would run digits together keep an underscore between them.
pub fn camel(name: &str) -> String {
    let mut out = String::new();
    for part in name.split(['-', '_', '.', '+']).filter(|p| !p.is_empty()) {
        let prev_digit = out.chars().last().is_some_and(|c| c.is_ascii_digit());
        if prev_digit && part.starts_with(|c: char| c.is_ascii_digit()) {
            out.push('_');
        }
        let mut chars = part.chars();
        if let Some(c) = chars.next() {
            out.extend(c.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, 'V');
    }
    out
}

/// The Rust field name of a member.
pub fn field_name(name: &str) -> String {
    let mut out: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    if out.starts_with(|c: char| c.is_ascii_digit()) {
        out.insert(0, '_');
    }
    if KEYWORDS.contains(&out.as_str()) {
        out.push('_');
    }
    out
}

/// The Rust name of a schema type.
pub fn type_name(name: &str) -> String {
    match name.strip_prefix("q_obj_") {
        Some(rest) => {
            let (base, role) = rest.rsplit_once('-').unwrap_or((rest, ""));
            // Event names are upper case, and their argument types read better in camel case.
            if base.chars().any(|c| c.is_ascii_lowercase()) {
                camel(base) + &camel(role)
            } else {
                camel(&base.to_ascii_lowercase()) + &camel(role)
            }
        }
        None => camel(name),
    }
}

fn special_features(features: &[Feature], on: &dyn Fn(&IfCond) -> bool) -> u64 {
    let mut mask = 0;
    for f in features.iter().filter(|f| on(&f.ifcond)) {
        match f.name.as_str() {
            "deprecated" => mask |= 1,
            "unstable" => mask |= 2,
            _ => {}
        }
    }
    mask
}

fn features_expr(mask: u64) -> String {
    let mut parts = Vec::new();
    if mask & 1 != 0 {
        parts.push("crate::visit::QAPI_DEPRECATED");
    }
    if mask & 2 != 0 {
        parts.push("crate::visit::QAPI_UNSTABLE");
    }
    parts.join(" | ")
}

struct Gen<'a> {
    schema: &'a Schema,
    is_set: &'a dyn Fn(&str) -> bool,
    out: String,
    names: HashMap<&'a str, String>,
    /// Pairs (outer, member type) where the member has to be boxed.
    boxed: HashSet<(&'a str, &'a str)>,
}

impl<'a> Gen<'a> {
    fn on(&self, c: &IfCond) -> bool {
        c.as_ref().is_none_or(|c| c.eval(self.is_set))
    }

    fn entity(&self, name: &str) -> &'a Entity {
        self.schema.lookup(name).unwrap_or_else(|| panic!("unknown type {name}"))
    }

    fn members(&self, e: &'a Entity) -> Vec<&'a Member> {
        self.schema.members(e).into_iter().filter(|m| self.on(&m.ifcond)).collect()
    }

    fn variants(&self, vs: &'a [Variant]) -> Vec<&'a Variant> {
        vs.iter().filter(|v| self.on(&v.ifcond)).collect()
    }

    /// The types a value of `e` holds inline, without a `Vec` in between.
    fn inline_types(&self, e: &'a Entity) -> Vec<&'a str> {
        let mut out = Vec::new();
        match &e.kind {
            Kind::Object { branches, .. } => {
                out.extend(self.members(e).iter().map(|m| m.typ.as_str()));
                if let Some(b) = branches {
                    out.extend(self.variants(&b.variants).iter().map(|v| v.typ.as_str()));
                }
            }
            Kind::Alternate { variants } => {
                out.extend(self.variants(variants).iter().map(|v| v.typ.as_str()));
            }
            _ => {}
        }
        out.retain(|t| matches!(self.entity(t).kind, Kind::Object { .. } | Kind::Alternate { .. }));
        out
    }

    fn reaches(&self, from: &'a str, to: &str) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![from];
        while let Some(t) = stack.pop() {
            if t == to {
                return true;
            }
            if seen.insert(t) {
                stack.extend(self.inline_types(self.entity(t)));
            }
        }
        false
    }

    fn rust_type(&self, typ: &str) -> String {
        let e = self.entity(typ);
        match &e.kind {
            Kind::Builtin { .. } => match typ {
                "str" => "String".into(),
                "number" => "f64".into(),
                "int" | "int64" => "i64".into(),
                "int8" | "int16" | "int32" => format!("i{}", &typ[3..]),
                "uint8" | "uint16" | "uint32" | "uint64" => format!("u{}", &typ[4..]),
                "size" => "u64".into(),
                "bool" => "bool".into(),
                "any" => "QValue".into(),
                "null" => "()".into(),
                _ => unreachable!("built-in type {typ}"),
            },
            Kind::Array { element } => format!("Vec<{}>", self.rust_type(element)),
            _ => self.names[e.name.as_str()].clone(),
        }
    }

    fn member_type(&self, outer: &str, typ: &str) -> String {
        let t = self.rust_type(typ);
        if self.boxed.contains(&(outer, typ)) { format!("Box<{t}>") } else { t }
    }

    /// Code that visits the value at `place`, a `&mut` expression, named `name`, a
    /// `Option<&str>` expression. It evaluates to `Result<()>`.
    fn visit_expr(&self, typ: &str, place: &str, name: &str) -> String {
        let e = self.entity(typ);
        match &e.kind {
            Kind::Builtin { .. } => match typ {
                "str" => format!("v.type_str({name}, {place})"),
                "number" => format!("v.type_number({name}, {place})"),
                "int" | "int64" => format!("v.type_int64({name}, {place})"),
                "uint64" => format!("v.type_uint64({name}, {place})"),
                "size" => format!("v.type_size({name}, {place})"),
                "bool" => format!("v.type_bool({name}, {place})"),
                "any" => format!("v.type_any({name}, {place})"),
                "null" => format!("v.type_null({name})"),
                _ => format!("v.type_{typ}({name}, {place})"),
            },
            Kind::Array { element } => {
                let inner = self.visit_expr(element, "e", "None");
                format!("v.visit_list({name}, {place}, |v, e| {inner})")
            }
            _ => format!("{}::visit(v, {name}, {place})", self.names[e.name.as_str()]),
        }
    }

    fn line(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }

    fn gen_enum(&mut self, e: &'a Entity) {
        let Kind::Enum { members, .. } = &e.kind else { unreachable!() };
        let name = self.names[e.name.as_str()].clone();
        let members: Vec<_> = members.iter().filter(|m| self.on(&m.ifcond)).collect();
        let mut variants: Vec<String> = Vec::new();
        for m in &members {
            let mut v = camel(&m.name);
            while variants.contains(&v) {
                v.push('_');
            }
            variants.push(v);
        }
        let feats: Vec<u64> =
            members.iter().map(|m| special_features(&m.features, &|c| self.on(c))).collect();
        let _ = writeln!(self.out, "\n/// QAPI enum `{}`.", e.name);
        self.line("#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]");
        let _ = writeln!(self.out, "pub enum {name} {{");
        for (i, v) in variants.iter().enumerate() {
            let _ = writeln!(self.out, "    /// `{}`", members[i].name);
            if i == 0 {
                self.line("    #[default]");
            }
            let _ = writeln!(self.out, "    {v},");
        }
        self.line("}\n");
        let _ = writeln!(self.out, "impl {name} {{");
        let _ = write!(self.out, "    pub const ALL: &'static [{name}] = &[");
        for v in &variants {
            let _ = write!(self.out, "{name}::{v}, ");
        }
        self.line("];");
        self.out.push_str("    pub const LOOKUP: QEnumLookup = QEnumLookup { array: &[");
        for m in &members {
            let _ = write!(self.out, "{:?}, ", m.name);
        }
        self.out.push_str("], features: ");
        if feats.iter().any(|&f| f != 0) {
            let list: Vec<String> = feats.iter().map(|f| f.to_string()).collect();
            let _ = writeln!(self.out, "Some(&[{}]) }};", list.join(", "));
        } else {
            self.line("None };");
        }
        self.line("\n    /// The name QMP uses for this value.");
        self.line("    pub fn as_str(self) -> &'static str {");
        self.line("        Self::LOOKUP.array[self as usize]");
        self.line("    }\n");
        self.line("    /// The value QMP calls `name`.");
        self.line("    pub fn from_name(name: &str) -> Option<Self> {");
        self.line("        Self::LOOKUP.find(name).map(|i| Self::ALL[i])");
        self.line("    }");
        self.line("}\n");
        let _ = writeln!(self.out, "impl Visit for {name} {{");
        self.line(
            "    fn visit(v: &mut dyn Visitor, name: Option<&str>, obj: &mut Self) -> Result<()> {",
        );
        self.line("        let mut value = *obj as usize;");
        self.line("        v.type_enum(name, &mut value, &Self::LOOKUP)?;");
        self.line("        *obj = Self::ALL[value];");
        self.line("        Ok(())");
        self.line("    }");
        self.line("}");
    }

    fn tag_variant(&self, tag_enum: &Entity, value: &str) -> String {
        let Kind::Enum { members, .. } = &tag_enum.kind else { unreachable!() };
        let mut seen: Vec<String> = Vec::new();
        for m in members.iter().filter(|m| self.on(&m.ifcond)) {
            let mut v = camel(&m.name);
            while seen.contains(&v) {
                v.push('_');
            }
            if m.name == value {
                return v;
            }
            seen.push(v);
        }
        panic!("{value} is not a value of {}", tag_enum.name)
    }

    fn gen_object(&mut self, e: &'a Entity) {
        let Kind::Object { branches, .. } = &e.kind else { unreachable!() };
        let name = self.names[e.name.as_str()].clone();
        let members = self.members(e);
        let tag = branches.as_ref().map(|b| b.tag.as_str());
        let _ = writeln!(
            self.out,
            "\n/// QAPI {} `{}`.",
            if tag.is_some() { "union" } else { "struct" },
            e.name
        );
        self.line("#[derive(Clone, Debug, Default, PartialEq)]");
        let _ = writeln!(self.out, "pub struct {name} {{");
        for m in &members {
            if Some(m.name.as_str()) == tag {
                continue;
            }
            let t = self.member_type(&e.name, &m.typ);
            let t = if m.optional { format!("Option<{t}>") } else { t };
            let _ = writeln!(self.out, "    /// `{}`", m.name);
            let _ = writeln!(self.out, "    pub {}: {t},", field_name(&m.name));
        }
        if tag.is_some() {
            self.line("    /// The branch, which also decides the tag.");
            let _ = writeln!(self.out, "    pub u: {name}U,");
        }
        self.line("}");

        // The branch enum.
        let mut tag_info = None;
        if let (Some(b), Some(tag)) = (branches, tag) {
            let tag_member = members.iter().find(|m| m.name == tag).expect("the tag is a member");
            let tag_enum = self.entity(&tag_member.typ);
            let tag_rust = self.names[tag_enum.name.as_str()].clone();
            let variants = self.variants(&b.variants);
            let Kind::Enum { members: tag_members, .. } = &tag_enum.kind else { unreachable!() };
            let first = tag_members.iter().find(|m| self.on(&m.ifcond)).expect("tag has values");
            let _ =
                writeln!(self.out, "\n/// The branches of `{}`, one per value of `{tag}`.", e.name);
            self.line("#[derive(Clone, Debug, PartialEq)]");
            let _ = writeln!(self.out, "pub enum {name}U {{");
            let mut arms = Vec::new();
            for v in variants.iter().copied() {
                let vn = self.tag_variant(tag_enum, &v.name);
                if v.typ == "q_empty" {
                    let _ = writeln!(self.out, "    {vn},");
                } else {
                    let t = self.member_type(&e.name, &v.typ);
                    let _ = writeln!(self.out, "    {vn}({t}),");
                }
                arms.push((vn, v));
            }
            self.line("}\n");
            let _ = writeln!(self.out, "impl {name}U {{");
            let _ = writeln!(self.out, "    /// The value of `{tag}` this branch goes with.");
            let _ = writeln!(self.out, "    pub fn tag(&self) -> {tag_rust} {{");
            self.line("        match self {");
            for (vn, v) in &arms {
                let pat = if v.typ == "q_empty" { String::new() } else { "(_)".into() };
                let _ = writeln!(self.out, "            {name}U::{vn}{pat} => {tag_rust}::{vn},");
            }
            self.line("        }");
            self.line("    }\n");
            self.line("    /// The branch for `tag`, with its members at their defaults.");
            let _ = writeln!(self.out, "    pub fn for_tag(tag: {tag_rust}) -> Self {{");
            self.line("        match tag {");
            for (vn, v) in &arms {
                let val =
                    if v.typ == "q_empty" { String::new() } else { "(Default::default())".into() };
                let _ = writeln!(self.out, "            {tag_rust}::{vn} => {name}U::{vn}{val},");
            }
            self.line("        }");
            self.line("    }");
            self.line("}\n");
            let _ = writeln!(self.out, "impl Default for {name}U {{");
            self.line("    fn default() -> Self {");
            let first_v = self.tag_variant(tag_enum, &first.name);
            let _ = writeln!(self.out, "        Self::for_tag({tag_rust}::{first_v})");
            self.line("    }");
            self.line("}");
            tag_info = Some((tag.to_string(), tag_rust, arms));
        }

        // visit_type_FOO_members().
        let _ = writeln!(self.out, "\nimpl {name} {{");
        self.line("    /// Visits the members without the struct around them, like");
        let c_name: String =
            e.name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
        let _ = writeln!(self.out, "    /// `visit_type_{c_name}_members()`.");
        self.line("    pub fn visit_members(v: &mut dyn Visitor, obj: &mut Self) -> Result<()> {");
        let mut body = String::new();
        for m in &members {
            let is_tag = Some(m.name.as_str()) == tag;
            let fname = field_name(&m.name);
            let nm = format!("Some({:?})", m.name);
            let mut inner = if is_tag {
                let tag_rust = &tag_info.as_ref().expect("union").1;
                format!(
                    "let mut tag: {tag_rust} = obj.u.tag();\n{}?;\nif tag != obj.u.tag() {{\n    obj.u = {name}U::for_tag(tag);\n}}\n",
                    self.visit_expr(&m.typ, "&mut tag", &nm)
                )
            } else if m.optional {
                format!("{}?;\n", self.visit_expr(&m.typ, "p", &nm))
            } else {
                format!("{}?;\n", self.visit_expr(&m.typ, &format!("&mut obj.{fname}"), &nm))
            };
            if m.optional && !is_tag {
                inner =
                    format!("let p = obj.{fname}.get_or_insert_with(Default::default);\n{inner}");
            }
            let feats = special_features(&m.features, &|c| self.on(c));
            if feats != 0 {
                let f = features_expr(feats);
                inner = format!(
                    "v.policy_reject({nm}, {f})?;\nif !v.policy_skip({nm}, {f}) {{\n{}}}\n",
                    indent(&inner)
                );
            }
            if m.optional && !is_tag {
                inner = format!(
                    "if v.optional({nm}, obj.{fname}.is_some()) {{\n{}}}\n",
                    indent(&inner)
                );
            }
            body.push_str(&inner);
        }
        if let Some((_, _, arms)) = &tag_info {
            body.push_str("match &mut obj.u {\n");
            for (vn, v) in arms {
                if v.typ == "q_empty" {
                    let _ = writeln!(body, "    {name}U::{vn} => {{}}");
                } else {
                    let vt = self.names[v.typ.as_str()].clone();
                    let _ = writeln!(body, "    {name}U::{vn}(u) => {vt}::visit_members(v, u)?,");
                }
            }
            body.push_str("}\n");
        }
        if members.is_empty() && tag_info.is_none() {
            body.push_str("let _ = (v, obj);\n");
        }
        body.push_str("Ok(())\n");
        self.out.push_str(&indent_by(&body, 8));
        self.line("    }");
        self.line("}\n");

        // visit_type_FOO().
        let _ = writeln!(self.out, "impl Visit for {name} {{");
        self.line(
            "    fn visit(v: &mut dyn Visitor, name: Option<&str>, obj: &mut Self) -> Result<()> {",
        );
        self.line("        v.start_struct(name)?;");
        self.line("        let r = Self::visit_members(v, obj).and_then(|()| v.check_struct());");
        self.line("        v.end_struct();");
        self.line("        if r.is_err() && v.is_input() {");
        self.line("            *obj = Self::default();");
        self.line("        }");
        self.line("        r");
        self.line("    }");
        self.line("}");
    }

    fn alternate_qtype(&self, typ: &str) -> &'static str {
        let e = self.entity(typ);
        match &e.kind {
            Kind::Builtin { json_type } => match *json_type {
                "string" => "QString",
                "number" | "int" => "QNum",
                "boolean" => "QBool",
                "null" => "QNull",
                other => panic!("an alternate cannot hold {other}"),
            },
            Kind::Enum { .. } => "QString",
            Kind::Array { .. } => "QList",
            Kind::Object { .. } => "QDict",
            _ => panic!("an alternate cannot hold {typ}"),
        }
    }

    fn gen_alternate(&mut self, e: &'a Entity) {
        let Kind::Alternate { variants } = &e.kind else { unreachable!() };
        let name = self.names[e.name.as_str()].clone();
        let variants = self.variants(variants);
        let arms: Vec<(String, &Variant, &str)> =
            variants.iter().map(|v| (camel(&v.name), *v, self.alternate_qtype(&v.typ))).collect();
        let _ = writeln!(self.out, "\n/// QAPI alternate `{}`.", e.name);
        self.line("#[derive(Clone, Debug, PartialEq)]");
        let _ = writeln!(self.out, "pub enum {name} {{");
        for (vn, v, _) in &arms {
            let t = self.member_type(&e.name, &v.typ);
            let _ = writeln!(self.out, "    /// `{}`", v.name);
            let _ = writeln!(self.out, "    {vn}({t}),");
        }
        self.line("}\n");
        // The default has to be finite, so prefer an alternative that is not a struct.
        let (dv, _, _) = arms
            .iter()
            .find(|(_, _, q)| *q != "QDict")
            .or(arms.first())
            .expect("an alternate has alternatives");
        let _ = writeln!(self.out, "impl Default for {name} {{");
        self.line("    fn default() -> Self {");
        let _ = writeln!(self.out, "        {name}::{dv}(Default::default())");
        self.line("    }");
        self.line("}\n");
        let _ = writeln!(self.out, "impl {name} {{");
        self.line("    fn qtype(&self) -> QType {");
        self.line("        match self {");
        for (vn, _, q) in &arms {
            let _ = writeln!(self.out, "            {name}::{vn}(_) => QType::{q},");
        }
        self.line("        }");
        self.line("    }");
        self.line("}\n");
        let _ = writeln!(self.out, "impl Visit for {name} {{");
        self.line(
            "    fn visit(v: &mut dyn Visitor, name: Option<&str>, obj: &mut Self) -> Result<()> {",
        );
        self.line("        let qtype = v.start_alternate(name)?.unwrap_or_else(|| obj.qtype());");
        self.line("        let input = v.is_input();");
        self.line("        let r = match qtype {");
        for (vn, v, q) in &arms {
            let _ = writeln!(self.out, "            QType::{q} => {{");
            let _ = writeln!(
                self.out,
                "                if input && !matches!(obj, {name}::{vn}(_)) {{"
            );
            let _ =
                writeln!(self.out, "                    *obj = {name}::{vn}(Default::default());");
            self.line("                }");
            if *q == "QNull" {
                self.line("                v.type_null(name)");
                self.line("            }");
                continue;
            }
            let _ = writeln!(
                self.out,
                "                let {name}::{vn}(x) = obj else {{ unreachable!() }};"
            );
            if *q == "QDict" {
                let vt = self.names[v.typ.as_str()].clone();
                self.line("                v.start_struct(name).and_then(|()| {");
                let _ = writeln!(
                    self.out,
                    "                    let r = {vt}::visit_members(v, x).and_then(|()| v.check_struct());"
                );
                self.line("                    v.end_struct();");
                self.line("                    r");
                self.line("                })");
            } else {
                let _ =
                    writeln!(self.out, "                {}", self.visit_expr(&v.typ, "x", "name"));
            }
            self.line("            }");
        }
        self.line("            QType::None => unreachable!(\"a value always has a type\"),");
        self.line("            #[allow(unreachable_patterns)]");
        self.line("            _ => Err(Error::generic(format!(");
        let _ = writeln!(
            self.out,
            "                \"Invalid parameter type for '{{}}', expected: {}\",",
            e.name
        );
        self.line("                name.unwrap_or(\"null\")");
        self.line("            ))),");
        self.line("        };");
        self.line("        v.end_alternate();");
        self.line("        if r.is_err() && input {");
        self.line("            *obj = Self::default();");
        self.line("        }");
        self.line("        r");
        self.line("    }");
        self.line("}");
    }
}

fn indent(s: &str) -> String {
    indent_by(s, 4)
}

fn indent_by(s: &str, n: usize) -> String {
    let pad = " ".repeat(n);
    s.lines().map(|l| if l.is_empty() { "\n".to_string() } else { format!("{pad}{l}\n") }).collect()
}

impl<'a> Gen<'a> {
    fn new(schema: &'a Schema, is_set: &'a dyn Fn(&str) -> bool) -> (Self, Vec<&'a Entity>) {
        let mut g = Gen {
            schema,
            is_set,
            out: String::new(),
            names: HashMap::new(),
            boxed: HashSet::new(),
        };
        let types: Vec<&Entity> = schema
            .visit_order()
            .into_iter()
            .filter(|e| e.info.is_some() && g.on(&e.ifcond))
            .filter(|e| {
                matches!(e.kind, Kind::Enum { .. } | Kind::Object { .. } | Kind::Alternate { .. })
            })
            .collect();

        let mut taken: HashMap<String, &str> =
            RESERVED.iter().map(|r| (r.to_string(), "the generated module")).collect();
        for e in &types {
            let n = type_name(&e.name);
            if let Some(other) = taken.insert(n.clone(), &e.name) {
                panic!("QAPI types {} and {other} both map to the Rust name {n}", e.name);
            }
            g.names.insert(&e.name, n);
        }
        for e in &types {
            for t in g.inline_types(e) {
                if g.reaches(t, &e.name) {
                    g.boxed.insert((&e.name, t));
                }
            }
        }
        (g, types)
    }

    /// The commands or events of the schema whose condition holds, in schema order.
    fn entities(&self, want: &str) -> Vec<&'a Entity> {
        self.schema.entities.iter().filter(|e| e.meta() == want && self.on(&e.ifcond)).collect()
    }

    fn gen_command(&mut self, e: &'a Entity) {
        let Kind::Command(c) = &e.kind else { unreachable!() };
        let fname = field_name(&e.name);
        let arg = c.arg_type.as_deref().map(|t| self.names[t].clone());
        let ret = c.ret_type.as_deref().map(|t| self.rust_type(t));
        let mut options = Vec::new();
        if !c.success_response {
            options.push("QmpCommandOptions::NO_SUCCESS_RESP");
        }
        if c.allow_oob {
            options.push("QmpCommandOptions::ALLOW_OOB");
        }
        if c.allow_preconfig {
            options.push("QmpCommandOptions::ALLOW_PRECONFIG");
        }
        if c.coroutine {
            options.push("QmpCommandOptions::COROUTINE");
        }
        let options =
            if options.is_empty() { "QmpCommandOptions::NONE".into() } else { options.join(" | ") };
        let features = features_expr(special_features(&e.features, &|c| self.on(c)));
        let features = if features.is_empty() { "0".into() } else { features };

        let handler_args = match &arg {
            Some(a) => format!("&C, {a}"),
            None => "&C".into(),
        };
        let handler_ret = ret.clone().unwrap_or_else(|| "()".into());
        let _ = writeln!(
            self.out,
            "\n/// Registers `{}` with `handler`, the `qmp_marshal_{}()` of QEMU.",
            e.name,
            c_name(&e.name)
        );
        let _ = writeln!(self.out, "pub fn register_{fname}<C: 'static>(");
        self.line("    cmds: &mut QmpCommandList<C>,");
        let _ = writeln!(
            self.out,
            "    handler: impl Fn({handler_args}) -> Result<{handler_ret}> + Send + Sync + 'static,"
        );
        self.line(") {");
        self.line("    let func = move |ctx: &C, args: QDict, policy: &CompatPolicy| -> Result<Option<QValue>> {");
        let mut body = String::new();
        body.push_str("let mut iv = QObjectInputVisitor::new_qmp(QValue::Dict(args), *policy);\n");
        body.push_str("let v: &mut dyn Visitor = &mut iv;\n");
        match &arg {
            Some(a) => {
                let _ = writeln!(body, "let mut arg = {a}::default();");
                body.push_str("v.start_struct(None)?;\n");
                let _ = writeln!(
                    body,
                    "let r = {a}::visit_members(v, &mut arg).and_then(|()| v.check_struct());"
                );
            }
            None => {
                body.push_str("v.start_struct(None)?;\n");
                body.push_str("let r = v.check_struct();\n");
            }
        }
        body.push_str("v.end_struct();\nr?;\n");
        let call = if arg.is_some() { "handler(ctx, arg)?" } else { "handler(ctx)?" };
        match c.ret_type.as_deref() {
            Some(t) => {
                let _ = writeln!(body, "#[allow(unused_mut)]\nlet mut ret = {call};");
                body.push_str("let mut ov = QObjectOutputVisitor::new_qmp(*policy);\n");
                body.push_str("let v: &mut dyn Visitor = &mut ov;\n");
                let _ = writeln!(body, "{}?;", self.visit_expr(t, "&mut ret", "Some(\"unused\")"));
                body.push_str("Ok(Some(ov.complete()))\n");
            }
            None => {
                let _ = writeln!(body, "{call};");
                body.push_str("Ok(None)\n");
            }
        }
        self.out.push_str(&indent_by(&body, 8));
        self.line("    };");
        let _ = writeln!(
            self.out,
            "    cmds.register({:?}, Arc::new(func), {options}, {features});",
            e.name
        );
        self.line("}");
    }

    fn gen_event(&mut self, e: &'a Entity) {
        let Kind::Event { arg_type, .. } = &e.kind else { unreachable!() };
        let fname = field_name(&e.name.to_ascii_lowercase());
        let features = special_features(&e.features, &|c| self.on(c));
        let _ = writeln!(
            self.out,
            "\n/// Builds the `{}` event, `qapi_event_send_{}()` in QEMU.",
            e.name,
            c_name(&e.name.to_ascii_lowercase())
        );
        self.line("/// It is `None` when the `-compat` policy hides the event.");
        let arg = arg_type.as_deref().map(|t| (t, self.names[t].clone()));
        match &arg {
            Some((_, a)) => {
                let _ = writeln!(
                    self.out,
                    "pub fn event_{fname}(policy: &CompatPolicy, mut arg: {a}) -> Option<QDict> {{"
                );
            }
            None => {
                let _ = writeln!(
                    self.out,
                    "pub fn event_{fname}(policy: &CompatPolicy) -> Option<QDict> {{"
                );
            }
        }
        let mut body = String::new();
        if features != 0 {
            let _ = writeln!(
                body,
                "if compat_policy_output_hidden({}, policy) {{\n    return None;\n}}",
                features_expr(features)
            );
        }
        let m = if arg.is_some() { "mut " } else { "" };
        let _ = writeln!(body, "let {m}qmp = qmp_event_build_dict({:?});", e.name);
        match &arg {
            Some((t, a)) => {
                body.push_str("let mut ov = QObjectOutputVisitor::new_qmp(*policy);\n");
                body.push_str("let v: &mut dyn Visitor = &mut ov;\n");
                if t.starts_with("q_obj_") {
                    body.push_str(
                        "v.start_struct(Some(\"QAPIEvent\")).expect(\"output cannot fail\");\n",
                    );
                    let _ = writeln!(
                        body,
                        "{a}::visit_members(v, &mut arg).expect(\"output cannot fail\");"
                    );
                    body.push_str("v.check_struct().expect(\"output cannot fail\");\n");
                    body.push_str("v.end_struct();\n");
                } else {
                    let _ = writeln!(
                        body,
                        "{a}::visit(v, Some(\"QAPIEvent\"), &mut arg).expect(\"output cannot fail\");"
                    );
                }
                body.push_str("if let QValue::Dict(data) = ov.complete() {\n");
                body.push_str(
                    "    if !data.is_empty() {\n        qmp.put(\"data\", data);\n    }\n}\n",
                );
            }
            None if features == 0 => body.push_str("let _ = policy;\n"),
            None => {}
        }
        body.push_str("Some(qmp)\n");
        self.out.push_str(&indent(&body));
        self.line("}");
    }
}

/// The C spelling of a name, which QEMU uses in function names.
fn c_name(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

/// Generates the Rust types for every type of `schema` whose condition holds under `is_set`.
/// The result is meant to be `include!`d into a module of `ruvm-qapi`.
pub fn gen_types(schema: &Schema, is_set: &dyn Fn(&str) -> bool) -> String {
    let (mut g, types) = Gen::new(schema, is_set);
    g.out.push_str(
        "// Generated by ruvm-qapi-gen from the vendored QAPI schema. Do not edit.\n\n\
         use crate::visit::{QEnumLookup, Visit, Visitor, VisitorExt};\n\
         use crate::{QType, QValue};\n\
         use ruvm_base::{Error, Result};\n",
    );
    for e in types {
        match e.kind {
            Kind::Enum { .. } => g.gen_enum(e),
            Kind::Object { .. } => g.gen_object(e),
            Kind::Alternate { .. } => g.gen_alternate(e),
            _ => unreachable!(),
        }
    }
    g.out
}

/// Generates a `register_*` function for every command of `schema` that QEMU generates a
/// marshaller for. The result goes into a module next to the one [`gen_types`] fills.
pub fn gen_commands(schema: &Schema, is_set: &dyn Fn(&str) -> bool) -> String {
    let (mut g, _) = Gen::new(schema, is_set);
    g.out.push_str(
        "// Generated by ruvm-qapi-gen from the vendored QAPI schema. Do not edit.\n\n\
         use std::sync::Arc;\n\n\
         use crate::dispatch::{QmpCommandList, QmpCommandOptions};\n\
         use crate::types::*;\n\
         use crate::visit::{CompatPolicy, QObjectInputVisitor, QObjectOutputVisitor, Visit, Visitor, VisitorExt};\n\
         use crate::{QDict, QValue};\n\
         use ruvm_base::Result;\n",
    );
    for e in g.entities("command") {
        let Kind::Command(c) = &e.kind else { unreachable!() };
        if c.generate {
            g.gen_command(e);
        }
    }
    g.out
}

/// Generates an `event_*` function for every event of `schema`, plus the `QapiEvent` enum.
pub fn gen_events(schema: &Schema, is_set: &dyn Fn(&str) -> bool) -> String {
    let (mut g, _) = Gen::new(schema, is_set);
    g.out.push_str(
        "// Generated by ruvm-qapi-gen from the vendored QAPI schema. Do not edit.\n\n\
         use crate::dispatch::qmp_event_build_dict;\n\
         use crate::types::*;\n\
         use crate::visit::{CompatPolicy, QObjectOutputVisitor, Visit, Visitor, compat_policy_output_hidden};\n\
         use crate::{QDict, QValue};\n",
    );
    let events = g.entities("event");
    g.line("\n/// `QAPIEvent`, one value per event.");
    g.line("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]");
    g.line("pub enum QapiEvent {");
    for e in &events {
        let _ = writeln!(g.out, "    {},", camel(&e.name.to_ascii_lowercase()));
    }
    g.line("}\n");
    g.line("impl QapiEvent {");
    g.line("    pub const ALL: &[QapiEvent] = &[");
    for e in &events {
        let _ = writeln!(g.out, "        QapiEvent::{},", camel(&e.name.to_ascii_lowercase()));
    }
    g.line("    ];\n");
    g.line("    pub fn as_str(self) -> &'static str {");
    g.line("        match self {");
    for e in &events {
        let _ = writeln!(
            g.out,
            "            QapiEvent::{} => {:?},",
            camel(&e.name.to_ascii_lowercase()),
            e.name
        );
    }
    g.line("        }");
    g.line("    }\n");
    g.line("    pub fn from_name(name: &str) -> Option<Self> {");
    g.line("        Self::ALL.iter().copied().find(|e| e.as_str() == name)");
    g.line("    }");
    g.line("}");
    for e in events {
        g.gen_event(e);
    }
    g.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(camel("x86_64"), "X86_64");
        assert_eq!(camel("qcow2"), "Qcow2");
        assert_eq!(camel("host_device"), "HostDevice");
        assert_eq!(camel("2.12"), "V2_12");
        assert_eq!(type_name("q_obj_qom-list-arg"), "QomListArg");
        assert_eq!(type_name("q_obj_BlockdevOptions-base"), "BlockdevOptionsBase");
        assert_eq!(type_name("BlockdevOptions"), "BlockdevOptions");
        assert_eq!(type_name("q_obj_DEVICE_DELETED-arg"), "DeviceDeletedArg");
        assert_eq!(field_name("node-name"), "node_name");
        assert_eq!(field_name("type"), "type_");
    }
}
