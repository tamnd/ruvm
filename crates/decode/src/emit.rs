// SPDX-License-Identifier: GPL-2.0-or-later

//! Rust output. This follows the `output_*` methods of scripts/decodetree.py: the same items in
//! the same order, the same decision tree walked the same way, and the same errors raised at the
//! same points of the walk. Where the script writes C, this writes the Rust equivalent, as
//! described in the crate documentation.

use crate::model::{FieldDef, General, Model, Node, Tree, TreeSub, fields_get, low_mask};
use crate::parse::Ctx;
use crate::tree::SizeTree;
use crate::{Error, rust_ident};

/// The lints the generated code opts out of. It is included into crates that build with
/// `-D warnings`, and names like `arg_rrr` and `trans_ADD_i` come from the QEMU files.
const ALLOW: &str = "#[allow(non_camel_case_types, non_snake_case, non_upper_case_globals, \
dead_code, unused_mut, unused_variables, unused_parens, unused_assignments, unreachable_code, \
unreachable_patterns, unreachable_pub, clippy::all)]";

/// The kind of value a field expression produces before it is stored.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// The unsigned bit operation type, `u32` or `u64`.
    Unsigned,
    /// The signed bit operation type, `i32` or `i64`, from `sextract` or a function call.
    Signed,
    /// A constant from the pattern.
    Const(i128),
}

/// The Rust type for a C type named in an argument set.
pub(crate) fn rust_type(c: &str) -> &str {
    match c {
        "int" | "int32_t" | "signed" => "i32",
        "unsigned" | "uint32_t" => "u32",
        "bool" | "_Bool" => "bool",
        "int8_t" => "i8",
        "uint8_t" => "u8",
        "int16_t" => "i16",
        "uint16_t" => "u16",
        "long" | "int64_t" => "i64",
        "uint64_t" => "u64",
        "size_t" | "uintptr_t" => "usize",
        "ssize_t" | "intptr_t" => "isize",
        other => other,
    }
}

/// The value C would store when converting `v` to an integer type, as a Rust literal.
fn const_literal(v: i128, ty: &str) -> Option<String> {
    Some(match ty {
        "i8" => (v as i8).to_string(),
        "u8" => (v as u8).to_string(),
        "i16" => (v as i16).to_string(),
        "u16" => (v as u16).to_string(),
        "i32" => (v as i32).to_string(),
        "u32" => (v as u32).to_string(),
        "i64" => (v as i64).to_string(),
        "u64" => (v as u64).to_string(),
        "isize" => (v as i64).to_string(),
        "usize" => (v as u64).to_string(),
        "bool" => (v != 0).to_string(),
        _ => return None,
    })
}

/// `is_contiguous`: the shift of a contiguous run of ones, or -1.
fn is_contiguous(bits: u64) -> i32 {
    if bits == 0 {
        return -1;
    }
    let shift = bits.trailing_zeros();
    let rest = bits >> shift;
    if rest & rest.wrapping_add(1) == 0 { shift as i32 } else { -1 }
}

/// The part of the pattern table and trait that the output and `inspect` share.
pub(crate) fn functions(m: &Model) -> Result<Vec<(String, bool)>, Error> {
    let mut out: Vec<(String, bool)> = Vec::new();
    fn walk(f: &FieldDef, out: &mut Vec<(String, bool)>) -> Result<(), Error> {
        let (name, takes) = match f {
            FieldDef::Function { func, base } => {
                walk(base, out)?;
                (func, true)
            }
            FieldDef::Parameter { func } => (func, false),
            FieldDef::Multi { subs, .. } => {
                for s in subs {
                    walk(s, out)?;
                }
                return Ok(());
            }
            _ => return Ok(()),
        };
        match out.iter().find(|(n, _)| n == name) {
            Some((_, t)) if *t != takes => Err(Error {
                file: String::new(),
                line: 0,
                message: format!("function {name} is called both with and without an argument"),
            }),
            Some(_) => Ok(()),
            None => {
                out.push((name.clone(), takes));
                Ok(())
            }
        }
    }
    let mut formats: Vec<&General> = m.formats.iter().collect();
    formats.sort_by(|a, b| a.name.cmp(&b.name));
    for g in formats.into_iter().chain(m.allpatterns.iter().map(|&p| &m.patterns[p])) {
        for (_, f) in &g.fields {
            walk(f, &mut out)?;
        }
    }
    Ok(out)
}

pub(crate) struct Emitter<'a> {
    pub(crate) m: &'a Model,
    pub(crate) ctx: &'a Ctx,
    pub(crate) trait_name: String,
    pub(crate) translate_prefix: String,
    pub(crate) public_decode: bool,
    pub(crate) sources: Vec<String>,
    out: String,
    bw: u32,
    ut: &'static str,
    st: &'static str,
    insn_ty: &'static str,
}

impl<'a> Emitter<'a> {
    pub(crate) fn new(
        m: &'a Model,
        ctx: &'a Ctx,
        trait_name: String,
        translate_prefix: String,
        public_decode: bool,
        sources: Vec<String>,
    ) -> Self {
        let (bw, ut, st) =
            if ctx.insnwidth == 64 { (64, "u64", "i64") } else { (32, "u32", "i32") };
        let insn_ty = match ctx.insnwidth {
            16 => "u16",
            64 => "u64",
            _ => "u32",
        };
        Emitter {
            m,
            ctx,
            trait_name,
            translate_prefix,
            public_decode,
            sources,
            out: String::new(),
            bw,
            ut,
            st,
            insn_ty,
        }
    }

    fn line(&mut self, indent: usize, text: &str) {
        for _ in 0..indent {
            self.out.push(' ');
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    fn df(&self) -> &str {
        &self.ctx.decode_function
    }

    fn bound(&self) -> String {
        format!("T: {} + ?Sized", rust_ident(&self.trait_name))
    }

    fn arg_struct(&self, set: usize) -> String {
        format!("arg_{}", self.m.arguments[set].name)
    }

    fn extract_name(&self, fmt: usize) -> String {
        rust_ident(&format!("{}_extract_{}", self.df(), self.m.formats[fmt].name))
    }

    fn trans_name(&self, pat: &str) -> String {
        rust_ident(&format!("{}_{}", self.translate_prefix, pat))
    }

    // Field expressions.

    fn to_unsigned(&self, (e, k): (String, Kind)) -> String {
        match k {
            Kind::Unsigned => e,
            Kind::Signed => format!("({e} as {})", self.ut),
            Kind::Const(v) => {
                format!("{}{}", const_literal(v, self.ut).unwrap_or_default(), self.ut)
            }
        }
    }

    fn to_signed(&self, (e, k): (String, Kind)) -> String {
        match k {
            Kind::Signed => e,
            Kind::Unsigned => format!("({e} as {})", self.st),
            Kind::Const(v) => {
                format!("{}{}", const_literal(v, self.st).unwrap_or_default(), self.st)
            }
        }
    }

    /// `extract32(x, pos, len)` or `sextract32(x, pos, len)` on an unsigned value `x` of the bit
    /// operation width.
    fn extract(&self, x: &str, sign: bool, pos: u32, len: u32) -> (String, Kind) {
        let bw = self.bw;
        if len == 0 || pos >= bw {
            // QEMU's extract functions assert on an empty field.
            return (
                format!("0{}", if sign { self.st } else { self.ut }),
                if sign { Kind::Signed } else { Kind::Unsigned },
            );
        }
        let len = len.min(bw - pos);
        if sign {
            let left = bw - pos - len;
            let right = bw - len;
            let mut e = x.to_string();
            if left > 0 {
                e = format!("({e} << {left})");
            }
            e = format!("({e} as {})", self.st);
            if right > 0 {
                e = format!("({e} >> {right})");
            }
            (e, Kind::Signed)
        } else {
            let m = low_mask(len);
            let e = if pos == 0 && len >= bw {
                x.to_string()
            } else if pos == 0 {
                format!("({x} & {m:#x})")
            } else if pos + len >= bw {
                format!("({x} >> {pos})")
            } else {
                format!("(({x} >> {pos}) & {m:#x})")
            };
            (e, Kind::Unsigned)
        }
    }

    fn insn_value(&self) -> String {
        if self.insn_ty == self.ut { "insn".to_string() } else { format!("(insn as {})", self.ut) }
    }

    fn expr(&self, f: &FieldDef, lv: &dyn Fn(&str) -> String) -> (String, Kind) {
        match f {
            FieldDef::Simple { sign, pos, len } => {
                self.extract(&self.insn_value(), *sign, *pos, *len)
            }
            FieldDef::Named { name, sign, len } => {
                let x = format!("({} as {})", lv(name), self.ut);
                self.extract(&x, *sign, 0, *len)
            }
            FieldDef::Multi { subs, .. } => {
                let mut ret: Option<(String, Kind)> = None;
                let mut pos = 0u32;
                for s in subs.iter().rev() {
                    let ext = self.expr(s, lv);
                    ret = Some(match ret {
                        None => ext,
                        Some(r) if pos >= self.bw => (self.to_unsigned(r), Kind::Unsigned),
                        Some(r) => (
                            format!(
                                "(({} & {:#x}) | ({} << {pos}))",
                                self.to_unsigned(r),
                                low_mask(pos),
                                self.to_unsigned(ext)
                            ),
                            Kind::Unsigned,
                        ),
                    });
                    let len = match &**s {
                        FieldDef::Simple { len, .. } | FieldDef::Named { len, .. } => *len,
                        _ => 0,
                    };
                    pos = pos.saturating_add(len);
                }
                ret.unwrap_or_else(|| ("0".into(), Kind::Unsigned))
            }
            FieldDef::Const(v) => (v.to_string(), Kind::Const(*v)),
            FieldDef::Function { func, base } => {
                let arg = self.to_signed(self.expr(base, lv));
                (format!("T::{}(ctx, {arg})", rust_ident(func)), Kind::Signed)
            }
            FieldDef::Parameter { func } => (format!("T::{}(ctx)", rust_ident(func)), Kind::Signed),
        }
    }

    /// `lvalue = value;` with the conversion C would do on the assignment.
    fn assign(&self, lvalue: &str, c_type: &str, value: (String, Kind)) -> String {
        let ty = rust_type(c_type);
        let rhs = match value.1 {
            Kind::Const(v) => match const_literal(v, ty) {
                Some(lit) => lit,
                None => format!("({v}i128) as {ty}"),
            },
            _ if ty == "bool" => format!("{} != 0", value.0),
            Kind::Unsigned if ty == self.ut => value.0,
            Kind::Signed if ty == self.st => value.0,
            _ => format!("{} as {ty}", value.0),
        };
        format!("{lvalue} = {rhs};")
    }

    /// `output_fields`: assignments in an order where every named reference is set first.
    fn output_fields(
        &mut self,
        indent: usize,
        g: &General,
        set: usize,
        lv: &dyn Fn(&str) -> String,
    ) -> Result<(), Error> {
        let mut data: Vec<(String, Vec<String>)> = Vec::new();
        for (n, f) in &g.fields {
            let mut refs = Vec::new();
            f.referenced_fields(&mut refs);
            data.push((n.clone(), refs));
        }
        let args = &self.m.arguments[set];
        loop {
            let ready: Vec<String> = data
                .iter()
                .filter(|(_, deps)| deps.iter().all(|d| !data.iter().any(|(k, _)| k == d)))
                .map(|(k, _)| k.clone())
                .collect();
            if ready.is_empty() {
                break;
            }
            for n in &ready {
                if let Some(f) = fields_get(&g.fields, n) {
                    let c_type = args
                        .fields
                        .iter()
                        .position(|x| x == n)
                        .map(|i| args.types[i].as_str())
                        .unwrap_or("int");
                    let value = self.expr(f, lv);
                    let text = self.assign(&lv(n), c_type, value);
                    self.line(indent, &text);
                }
            }
            data.retain(|(k, _)| !ready.contains(k));
        }
        if !data.is_empty() {
            let cycle: Vec<&str> = data.iter().map(|(k, _)| k.as_str()).collect();
            return Err(self.ctx.error(
                g.lineno,
                format!("field definitions form a cycle: {}", cycle.join(" => ")),
            ));
        }
        Ok(())
    }

    pub(crate) fn emit(mut self, varwidth: Option<&SizeTree>) -> Result<String, Error> {
        let m = self.m;
        let sources = self.sources.join(", ");
        self.line(0, &format!("// This file is autogenerated by ruvm-decode from {sources}."));
        self.line(0, "// It is the Rust counterpart of what QEMU's scripts/decodetree.py writes.");
        self.line(0, "");

        // Argument sets.
        let mut sets: Vec<usize> = (0..m.arguments.len()).collect();
        sets.sort_by(|&a, &b| m.arguments[a].name.cmp(&m.arguments[b].name));
        for &i in &sets {
            let a = &m.arguments[i];
            if a.is_extern {
                continue;
            }
            self.line(0, ALLOW);
            self.line(0, "#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]");
            self.line(0, &format!("pub struct {} {{", self.arg_struct(i)));
            for (n, t) in a.fields.iter().zip(&a.types) {
                self.line(4, &format!("pub {}: {},", rust_ident(n), rust_type(t)));
            }
            self.line(0, "}");
            self.line(0, "");
        }

        // One name per pattern, checking that the argument sets agree.
        let mut out_pats: Vec<usize> = Vec::new();
        for &p in &m.allpatterns {
            let name = &m.patterns[p].name;
            match out_pats.iter().find(|&&q| m.patterns[q].name == *name) {
                Some(&q) => {
                    if m.pattern_args(p) != m.pattern_args(q) {
                        return Err(self
                            .ctx
                            .error(0, format!("{name}  has conflicting argument sets")));
                    }
                }
                None => out_pats.push(p),
            }
        }
        for &p in &out_pats {
            let name = &m.patterns[p].name;
            let set = m.pattern_args(p);
            if *name != m.arguments[set].name {
                self.line(0, ALLOW);
                self.line(0, &format!("pub type arg_{name} = {};", self.arg_struct(set)));
            }
        }
        self.line(0, "");

        // The trait the caller implements.
        let funcs = functions(m)?;
        let ins = self.insn_ty;
        self.line(
            0,
            &format!(
                "/// The translator interface the decoder from {sources} calls into. Each `{}_*` \
             method returns true if it handled the instruction, as in QEMU.",
                self.translate_prefix
            ),
        );
        self.line(0, ALLOW);
        self.line(0, &format!("pub trait {} {{", rust_ident(&self.trait_name)));
        for &p in &out_pats {
            let name = &m.patterns[p].name;
            let ty = if *name != m.arguments[m.pattern_args(p)].name {
                format!("arg_{name}")
            } else {
                self.arg_struct(m.pattern_args(p))
            };
            let t = self.trans_name(name);
            self.line(4, &format!("fn {t}(&mut self, a: &mut {ty}) -> bool;"));
        }
        let st = self.st;
        for (f, takes) in &funcs {
            let f = rust_ident(f);
            if *takes {
                self.line(4, &format!("fn {f}(&mut self, x: {st}) -> {st};"));
            } else {
                self.line(4, &format!("fn {f}(&mut self) -> {st};"));
            }
        }
        if varwidth.is_some() {
            let f = rust_ident(&format!("{}_load_bytes", self.df()));
            self.line(4, &format!("fn {f}(&mut self, insn: {ins}, i: i32, n: i32) -> {ins};"));
        }
        self.line(0, "}");
        self.line(0, "");

        // The pattern table for fuzzers: mask, fixed bits and name of every pattern.
        let table = rust_ident(&format!("{}_PATTERNS", self.df().to_uppercase()));
        self.line(0, ALLOW);
        self.line(0, &format!("pub const {table}: &[(u64, u64, &str)] = &["));
        for &p in &m.allpatterns {
            let pt = &m.patterns[p];
            self.line(
                4,
                &format!(
                    "({}, {}, \"{}\"),",
                    self.ctx.whex(pt.fixedmask),
                    self.ctx.whex(pt.fixedbits),
                    pt.name
                ),
            );
        }
        self.line(0, "];");
        self.line(0, "");

        // Extract functions, one per format.
        let mut fmts: Vec<usize> = (0..m.formats.len()).collect();
        fmts.sort_by(|&a, &b| m.formats[a].name.cmp(&m.formats[b].name));
        let bound = self.bound();
        for &i in &fmts {
            let set = m.formats[i].base;
            self.line(0, ALLOW);
            self.line(
                0,
                &format!(
                    "fn {}<{bound}>(ctx: &mut T, a: &mut {}, insn: {ins}) {{",
                    self.extract_name(i),
                    self.arg_struct(set)
                ),
            );
            let lv = |n: &str| format!("a.{}", rust_ident(n));
            self.output_fields(4, &m.formats[i], set, &lv)?;
            self.line(0, "}");
            self.line(0, "");
        }

        // The decoder.
        let vis = if self.public_decode { "pub " } else { "" };
        self.line(0, ALLOW);
        self.line(
            0,
            &format!(
                "{vis}fn {}<{bound}>(ctx: &mut T, insn: {ins}) -> bool {{",
                rust_ident(self.df())
            ),
        );
        if !m.allpatterns.is_empty() {
            self.node_code(4, Node::Group(0), false, 0, 0)?;
        }
        self.line(4, "false");
        self.line(0, "}");

        if let Some(stree) = varwidth {
            self.line(0, "");
            self.line(0, ALLOW);
            self.line(
                0,
                &format!(
                    "{vis}fn {}<{bound}>(ctx: &mut T) -> {ins} {{",
                    rust_ident(&format!("{}_load", self.df()))
                ),
            );
            self.line(4, &format!("let mut insn: {ins} = 0;"));
            self.size_code(4, stree, 0, 0, 0);
            self.line(0, "}");
        }
        Ok(self.out)
    }

    fn node_code(
        &mut self,
        i: usize,
        n: Node,
        extracted: bool,
        outerbits: u64,
        outermask: u64,
    ) -> Result<(), Error> {
        match n {
            Node::Pattern(p) => self.pattern_code(i, p, extracted),
            Node::Group(g) => {
                let grp = &self.m.groups[g];
                if grp.overlapping {
                    for &p in &grp.pats {
                        let (pb, pm) = self.m.node_fixed(p);
                        if outermask != pm {
                            let innermask = pm & !outermask;
                            let innerbits = pb & !outermask;
                            self.line(
                                i,
                                &format!(
                                    "if (insn & {}) == {} {{",
                                    self.ctx.whex(innermask),
                                    self.ctx.whex(innerbits)
                                ),
                            );
                            self.line(i + 4, &format!("// {}", self.ctx.str_match_bits(pb, pm)));
                            self.node_code(i + 4, p, extracted, pb, pm)?;
                            self.line(i, "}");
                        } else {
                            self.node_code(i, p, extracted, pb, pm)?;
                        }
                    }
                    Ok(())
                } else {
                    match &grp.tree {
                        Some(t) => self.tree_code(i, t, extracted, outerbits, outermask),
                        None => Ok(()),
                    }
                }
            }
        }
    }

    fn declare(&mut self, i: usize, set: usize) {
        let s = self.arg_struct(set);
        let name = &self.m.arguments[set].name;
        self.line(i, &format!("let mut u_{name} = {s}::default();"));
    }

    fn tree_code(
        &mut self,
        i: usize,
        t: &Tree,
        mut extracted: bool,
        outerbits: u64,
        outermask: u64,
    ) -> Result<(), Error> {
        if !extracted {
            if let Some(base) = t.base {
                if self.m.formats[base].dangling_references().is_empty() {
                    let set = self.m.formats[base].base;
                    self.declare(i, set);
                    let name = &self.m.arguments[set].name;
                    self.line(
                        i,
                        &format!("{}(ctx, &mut u_{name}, insn);", self.extract_name(base)),
                    );
                    extracted = true;
                }
            }
        }

        let sh = is_contiguous(t.thismask);
        let (scrutinee, shift) = if sh > 0 {
            (format!("(insn >> {sh}) & {:#x}", t.thismask >> sh), sh as u32)
        } else {
            (format!("insn & {}", self.ctx.whex(t.thismask)), 0)
        };
        self.line(i, &format!("match {scrutinee} {{"));
        let mut subs: Vec<&(u64, TreeSub)> = t.subs.iter().collect();
        subs.sort_by_key(|(b, _)| *b);
        for (b, s) in subs {
            let case = if sh > 0 { format!("{:#x}", b >> shift) } else { self.ctx.whex(*b) };
            let innermask = outermask | t.thismask;
            let innerbits = outerbits | b;
            self.line(i + 4, &format!("{case} => {{"));
            self.line(i + 8, &format!("// {}", self.ctx.str_match_bits(innerbits, innermask)));
            match s {
                TreeSub::Tree(sub) => {
                    self.tree_code(i + 8, sub, extracted, innerbits, innermask)?
                }
                TreeSub::Node(n) => self.node_code(i + 8, *n, extracted, innerbits, innermask)?,
            }
            self.line(i + 4, "}");
        }
        self.line(i + 4, "_ => {}");
        self.line(i, "}");
        Ok(())
    }

    fn pattern_code(&mut self, i: usize, p: usize, extracted: bool) -> Result<(), Error> {
        let m = self.m;
        let pat = &m.patterns[p];
        let fmt = &m.formats[pat.base];
        let set = fmt.base;
        let arg = &m.arguments[set].name;
        self.line(i, &format!("// {}:{}", pat.file, pat.lineno));

        let fmt_refs = fmt.dangling_references();
        for r in &fmt_refs {
            if fields_get(&pat.fields, r).is_none() {
                return Err(self
                    .ctx
                    .error(pat.lineno, format!("format refers to undefined field {r}")));
            }
        }
        let pat_refs = pat.dangling_references();
        for r in &pat_refs {
            if fields_get(&fmt.fields, r).is_none() {
                return Err(self
                    .ctx
                    .error(pat.lineno, format!("pattern refers to undefined field {r}")));
            }
        }
        if !pat_refs.is_empty() && !fmt_refs.is_empty() {
            return Err(self.ctx.error(
                pat.lineno,
                "pattern that uses fields defined in format cannot use format that uses fields defined in pattern",
            ));
        }

        let var = format!("u_{arg}");
        let lv = |n: &str| format!("{var}.{}", rust_ident(n));
        if !extracted {
            self.declare(i, set);
        }
        if !fmt_refs.is_empty() {
            self.output_fields(i, pat, set, &lv)?;
        }
        if !extracted {
            self.line(i, &format!("{}(ctx, &mut {var}, insn);", self.extract_name(pat.base)));
        }
        if fmt_refs.is_empty() {
            self.output_fields(i, pat, set, &lv)?;
        }
        let t = self.trans_name(&pat.name);
        self.line(i, &format!("if T::{t}(ctx, &mut {var}) {{"));
        self.line(i + 4, "return true;");
        self.line(i, "}");
        Ok(())
    }

    fn size_code(
        &mut self,
        i: usize,
        t: &SizeTree,
        mut extracted: u32,
        outerbits: u64,
        outermask: u64,
    ) {
        let width = t.node_width();
        if extracted < width {
            let f = rust_ident(&format!("{}_load_bytes", self.df()));
            self.line(i, &format!("insn = T::{f}(ctx, insn, {}, {});", extracted / 8, width / 8));
            extracted = width;
        }
        match t {
            SizeTree::Leaf { .. } => self.line(i, "return insn;"),
            SizeTree::Node { mask, subs, .. } => {
                let sh = is_contiguous(*mask);
                let scrutinee = if sh > 0 {
                    format!("(insn >> {sh}) & {:#x}", mask >> sh)
                } else {
                    format!("insn & {}", self.ctx.whex(*mask))
                };
                self.line(i, &format!("match {scrutinee} {{"));
                let mut subs: Vec<&(u64, SizeTree)> = subs.iter().collect();
                subs.sort_by_key(|(b, _)| *b);
                for (b, s) in subs {
                    let case = if sh > 0 { format!("{:#x}", b >> sh) } else { self.ctx.whex(*b) };
                    let innermask = outermask | mask;
                    let innerbits = outerbits | b;
                    self.line(i + 4, &format!("{case} => {{"));
                    self.line(
                        i + 8,
                        &format!("// {}", self.ctx.str_match_bits(innerbits, innermask)),
                    );
                    self.size_code(i + 8, s, extracted, innerbits, innermask);
                    self.line(i + 4, "}");
                }
                self.line(i + 4, "_ => {}");
                self.line(i, "}");
                self.line(i, "return insn;");
            }
        }
    }
}
