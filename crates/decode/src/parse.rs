// SPDX-License-Identifier: GPL-2.0-or-later

//! The decodetree language parser, following `parse_file`, `parse_field`, `parse_arguments` and
//! `parse_generic` in scripts/decodetree.py, including the order in which it checks things, so
//! that a bad file fails with the same message on the same line.

use std::rc::Rc;

use crate::Error;
use crate::model::{ArgSet, FieldDef, Fields, General, Group, Model, Node, fields_get};

/// The settings the parser and the tree builder need, derived from the options.
#[derive(Debug, Clone)]
pub(crate) struct Ctx {
    pub(crate) insnwidth: u32,
    pub(crate) insnmask: u64,
    pub(crate) variablewidth: bool,
    pub(crate) decode_function: String,
    /// The file being parsed, and after parsing the last file, as in the script's global.
    pub(crate) input_file: String,
}

impl Ctx {
    pub(crate) fn error(&self, lineno: usize, message: impl Into<String>) -> Error {
        Error { file: self.input_file.clone(), line: lineno, message: message.into() }
    }

    /// `whex`, a hex number padded to the instruction width.
    pub(crate) fn whex(&self, val: u64) -> String {
        format!("0x{:0w$x}", val, w = (self.insnwidth / 4) as usize)
    }

    /// `str_match_bits`, the bits of a pattern as 0, 1 and dots.
    pub(crate) fn str_match_bits(&self, bits: u64, mask: u64) -> String {
        let space: u64 = 0x0101_0100;
        let mut r = String::new();
        let mut i: u64 = 1u64 << (self.insnwidth - 1);
        while i != 0 {
            if i & mask != 0 {
                r.push(if i & bits != 0 { '1' } else { '0' });
            } else {
                r.push('.');
            }
            if i & space != 0 {
                r.push(' ');
            }
            i >>= 1;
        }
        r
    }
}

fn is_ident_start(c: char) -> bool {
    c.is_ascii_alphabetic()
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// `[a-zA-Z][a-zA-Z0-9_]*`
fn is_c_ident(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if is_ident_start(c) => it.all(is_ident_char),
        _ => false,
    }
}

/// `[a-zA-Z0-9_]*`
fn is_word(s: &str) -> bool {
    s.chars().all(is_ident_char)
}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Python's `int()` on a run of digits, saturating where Python would grow without bound. Every
/// place that saturates goes on to report the value as too large.
fn int(s: &str) -> u64 {
    s.bytes().fold(0u64, |acc, b| acc.saturating_mul(10).saturating_add(u64::from(b - b'0')))
}

/// A token of the form `prefix` followed by a word, the `&name`, `@name` and `%name` forms.
fn sigil(t: &str, c: char) -> Option<&str> {
    let rest = t.strip_prefix(c)?;
    is_word(rest).then_some(rest)
}

/// Split `a:b` where `a` satisfies `left` and `b` is digits, optionally with an `s` in front.
/// Returns the left part, whether the `s` was there, and the digits.
fn split_len(t: &str, left: fn(&str) -> bool) -> Option<(&str, bool, &str)> {
    let (a, b) = t.split_once(':')?;
    if !left(a) {
        return None;
    }
    let (sign, digits) = match b.strip_prefix('s') {
        Some(d) => (true, d),
        None => (false, b),
    };
    is_digits(digits).then_some((a, sign, digits))
}

/// Python's `str.expandtabs()` with the default tab size of 8.
fn expandtabs(line: &str) -> String {
    let mut out = String::new();
    let mut col = 0usize;
    for c in line.chars() {
        if c == '\t' {
            let n = 8 - col % 8;
            out.extend(std::iter::repeat_n(' ', n));
            col += n;
        } else {
            out.push(c);
            col += 1;
        }
    }
    out
}

/// Python's universal newline split, as used by `for line in f` on a text file.
fn lines(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                out.push(&text[start..i]);
                start = i + 1;
            }
            b'\r' => {
                out.push(&text[start..i]);
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    i += 1;
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < bytes.len() {
        out.push(&text[start..]);
    }
    out
}

pub(crate) struct Parser<'m> {
    pub(crate) ctx: &'m mut Ctx,
    pub(crate) m: &'m mut Model,
}

impl Parser<'_> {
    fn error(&self, lineno: usize, msg: impl Into<String>) -> Error {
        self.ctx.error(lineno, msg)
    }

    fn parse_field(&mut self, lineno: usize, name: &str, toks: &[&str]) -> Result<(), Error> {
        let insnwidth = u64::from(self.ctx.insnwidth);
        let mut subs: Vec<Rc<FieldDef>> = Vec::new();
        let mut width: u64 = 0;
        let mut func: Option<String> = None;
        for &t in toks {
            if let Some(rest) = t.strip_prefix("!function=") {
                if func.is_some() {
                    return Err(self.error(lineno, "duplicate function"));
                }
                func = Some(rest.split('=').next().unwrap_or_default().to_string());
                continue;
            }
            if let Some((n, sign, digits)) = split_len(t, is_c_ident) {
                let le = int(digits);
                subs.push(Rc::new(FieldDef::Named {
                    name: n.to_string(),
                    sign,
                    len: le.min(u64::from(u32::MAX)) as u32,
                }));
                width = width.saturating_add(le);
                continue;
            }
            let Some((po, sign, le)) = split_len(t, is_digits) else {
                return Err(self.error(lineno, format!("invalid field token \"{t}\"")));
            };
            let po = int(po);
            let le = int(le);
            if po.saturating_add(le) > insnwidth {
                return Err(self.error(lineno, format!("field {t} too large")));
            }
            subs.push(Rc::new(FieldDef::Simple { sign, pos: po as u32, len: le as u32 }));
            width += le;
        }

        if width > insnwidth {
            return Err(self.error(lineno, "field too large"));
        }
        let f = if subs.is_empty() {
            match func {
                Some(func) => Rc::new(FieldDef::Parameter { func }),
                None => return Err(self.error(lineno, "field with no value")),
            }
        } else {
            let f = if subs.len() == 1 {
                subs.pop().unwrap_or_else(|| unreachable!())
            } else {
                let mut mask = 0;
                for s in &subs {
                    if mask & s.mask() != 0 {
                        return Err(self.error(lineno, "field components overlap"));
                    }
                    mask |= s.mask();
                }
                Rc::new(FieldDef::Multi { subs, mask })
            };
            match func {
                Some(func) => Rc::new(FieldDef::Function { func, base: f }),
                None => f,
            }
        };

        if self.m.fields.contains_key(name) {
            return Err(self.error(lineno, format!("duplicate field {name}")));
        }
        self.m.fields.insert(name.to_string(), f);
        Ok(())
    }

    fn parse_arguments(&mut self, lineno: usize, name: &str, toks: &[&str]) -> Result<(), Error> {
        let mut flds: Vec<String> = Vec::new();
        let mut types = Vec::new();
        let mut is_extern = false;
        for &n in toks {
            if n == "!extern" {
                is_extern = true;
                self.m.anyextern = true;
                continue;
            }
            let (n, t) = match n.split_once(':') {
                Some((a, b)) if is_c_ident(a) && is_c_ident(b) => (a, b),
                _ if is_c_ident(n) => (n, "int"),
                _ => {
                    return Err(self.error(lineno, format!("invalid argument set token \"{n}\"")));
                }
            };
            if flds.iter().any(|f| f == n) {
                return Err(self.error(lineno, format!("duplicate argument \"{n}\"")));
            }
            flds.push(n.to_string());
            types.push(t.to_string());
        }
        if self.m.arg_index(name).is_some() {
            return Err(self.error(lineno, format!("duplicate argument set {name}")));
        }
        self.m.arguments.push(ArgSet { name: name.to_string(), fields: flds, types, is_extern });
        Ok(())
    }

    fn lookup_field(&self, lineno: usize, name: &str) -> Result<Rc<FieldDef>, Error> {
        match self.m.fields.get(name) {
            Some(f) => Ok(f.clone()),
            None => Err(self.error(lineno, format!("undefined field {name}"))),
        }
    }

    fn add_field(
        &self,
        lineno: usize,
        flds: &mut Fields,
        new_name: &str,
        f: Rc<FieldDef>,
    ) -> Result<(), Error> {
        if fields_get(flds, new_name).is_some() {
            return Err(self.error(lineno, format!("duplicate field {new_name}")));
        }
        flds.push((new_name.to_string(), f));
        Ok(())
    }

    fn infer_argument_set(&mut self, flds: &Fields) -> usize {
        for (i, arg) in self.m.arguments.iter().enumerate() {
            // eq_fields_for_args: same count, only int fields, every name present.
            if flds.len() != arg.fields.len() {
                continue;
            }
            if arg.types.iter().any(|t| t != "int") {
                continue;
            }
            if flds.iter().all(|(k, _)| arg.fields.contains(k)) {
                return i;
            }
        }
        let name = format!("{}{}", self.ctx.decode_function, self.m.arguments.len());
        self.m.arguments.push(ArgSet {
            name,
            fields: flds.iter().map(|(n, _)| n.clone()).collect(),
            types: vec!["int".to_string(); flds.len()],
            is_extern: false,
        });
        self.m.arguments.len() - 1
    }

    /// `infer_format`. The script sorts constant fields out with `c is ConstField`, which is
    /// never true for an instance, so every field goes into the inferred format and the pattern
    /// keeps none. That is reproduced here because it decides which formats get shared.
    fn infer_format(
        &mut self,
        arg: Option<usize>,
        fieldmask: u64,
        flds: Fields,
        width: u32,
    ) -> (usize, Fields) {
        let arg = match arg {
            Some(a) => a,
            None => self.infer_argument_set(&flds),
        };
        for (i, fmt) in self.m.formats.iter().enumerate() {
            if fmt.base != arg || fieldmask != fmt.fieldmask || width != fmt.width {
                continue;
            }
            if flds.len() != fmt.fields.len() {
                continue;
            }
            let same = flds.iter().all(|(k, a)| match fields_get(&fmt.fields, k) {
                Some(b) => !a.differs(b),
                None => false,
            });
            if same {
                return (i, Vec::new());
            }
        }
        let name = format!("{}_Fmt_{}", self.ctx.decode_function, self.m.formats.len());
        self.m.formats.push(General {
            name,
            file: self.ctx.input_file.clone(),
            lineno: 0,
            base: arg,
            fixedbits: 0,
            fixedmask: 0,
            undefmask: 0,
            fieldmask,
            fields: flds,
            width,
        });
        (self.m.formats.len() - 1, Vec::new())
    }

    fn parse_generic(
        &mut self,
        lineno: usize,
        parent: Option<usize>,
        name: &str,
        toks: &[&str],
    ) -> Result<(), Error> {
        let insnwidth = self.ctx.insnwidth;
        let is_format = parent.is_none();
        // Wide enough that a definition longer than any instruction still counts its bits.
        let mut fixedmask: u128 = 0;
        let mut fixedbits: u128 = 0;
        let mut undefmask: u128 = 0;
        let mut width: u64 = 0;
        let mut flds: Fields = Vec::new();
        let mut arg: Option<usize> = None;
        let mut fmt: Option<usize> = None;

        for &t in toks {
            if let Some(tt) = sigil(t, '&') {
                if arg.is_some() {
                    return Err(self.error(lineno, "multiple argument sets"));
                }
                match self.m.arg_index(tt) {
                    Some(a) => arg = Some(a),
                    None => return Err(self.error(lineno, format!("undefined argument set {t}"))),
                }
                continue;
            }
            if let Some(tt) = sigil(t, '@') {
                if fmt.is_some() {
                    return Err(self.error(lineno, "multiple formats"));
                }
                match self.m.format_index(tt) {
                    Some(f) => fmt = Some(f),
                    None => return Err(self.error(lineno, format!("undefined format {t}"))),
                }
                continue;
            }
            if let Some(tt) = sigil(t, '%') {
                let f = self.lookup_field(lineno, tt)?;
                self.add_field(lineno, &mut flds, tt, f)?;
                continue;
            }
            if let Some((fname, rest)) = t.split_once('=') {
                if is_c_ident(fname) {
                    if let Some(iname) = sigil(rest, '%') {
                        let f = self.lookup_field(lineno, iname)?;
                        self.add_field(lineno, &mut flds, fname, f)?;
                        continue;
                    }
                    let digits = rest.strip_prefix(['+', '-']).unwrap_or(rest);
                    if is_digits(digits) {
                        let mag = digits.bytes().fold(0i128, |acc, b| {
                            acc.saturating_mul(10).saturating_add(i128::from(b - b'0'))
                        });
                        let value = if rest.starts_with('-') { -mag } else { mag };
                        self.add_field(lineno, &mut flds, fname, Rc::new(FieldDef::Const(value)))?;
                        continue;
                    }
                }
            }

            let shift: u64;
            if !t.is_empty() && t.bytes().all(|b| matches!(b, b'0' | b'1' | b'.' | b'-')) {
                shift = t.len() as u64;
                for b in t.bytes() {
                    fixedbits = (fixedbits << 1) | u128::from(b == b'1');
                    fixedmask = (fixedmask << 1) | u128::from(b == b'0' || b == b'1');
                    undefmask = (undefmask << 1) | u128::from(b == b'-');
                }
            } else if let Some((fname, sign, flen)) = split_len(t, is_c_ident) {
                shift = int(flen);
                if shift.saturating_add(width) > u64::from(insnwidth) {
                    return Err(self.error(lineno, format!("field {fname} exceeds insnwidth")));
                }
                let pos = (u64::from(insnwidth) - width - shift) as u32;
                let f = Rc::new(FieldDef::Simple { sign, pos, len: shift as u32 });
                self.add_field(lineno, &mut flds, fname, f)?;
                let sh = shift as u32;
                fixedbits = fixedbits.checked_shl(sh).unwrap_or(0);
                fixedmask = fixedmask.checked_shl(sh).unwrap_or(0);
                undefmask = undefmask.checked_shl(sh).unwrap_or(0);
            } else {
                return Err(self.error(lineno, format!("invalid token \"{t}\"")));
            }
            width = width.saturating_add(shift);
        }

        let insnw = u64::from(insnwidth);
        if self.ctx.variablewidth && width < insnw && width % 8 == 0 {
            let shift = (insnw - width) as u32;
            fixedbits <<= shift;
            fixedmask <<= shift;
            undefmask <<= shift;
            undefmask |= (1u128 << shift) - 1;
        } else if !(is_format && width == 0) && width != insnw {
            return Err(self.error(lineno, format!("definition has {width} bits")));
        }
        let width = width as u32;
        let mut fixedbits = fixedbits as u64;
        let mut fixedmask = fixedmask as u64;
        let mut undefmask = undefmask as u64;

        let mut fieldmask = 0u64;
        for (_, f) in &flds {
            fieldmask |= f.mask();
        }

        if is_format {
            if fmt.is_some() {
                return Err(self.error(lineno, "format referencing format"));
            }
            let arg = match arg {
                Some(a) => {
                    let set = &self.m.arguments[a];
                    for (f, _) in &flds {
                        if !set.fields.contains(f) {
                            return Err(self.error(
                                lineno,
                                format!("field {f} not in argument set {}", set.name),
                            ));
                        }
                    }
                    a
                }
                None => self.infer_argument_set(&flds),
            };
            if self.m.format_index(name).is_some() {
                return Err(self.error(lineno, format!("duplicate format name {name}")));
            }
            self.m.formats.push(General {
                name: name.to_string(),
                file: self.ctx.input_file.clone(),
                lineno,
                base: arg,
                fixedbits,
                fixedmask,
                undefmask,
                fieldmask,
                fields: flds,
                width,
            });
        } else {
            let (fmt, flds) = match fmt {
                Some(f) => {
                    if arg.is_some() {
                        return Err(
                            self.error(lineno, "pattern specifies both format and argument set")
                        );
                    }
                    let fm = &self.m.formats[f];
                    if fixedmask & fm.fixedmask != 0 {
                        return Err(
                            self.error(lineno, "pattern fixed bits overlap format fixed bits")
                        );
                    }
                    if width != fm.width {
                        return Err(self.error(lineno, "pattern uses format of different width"));
                    }
                    fieldmask |= fm.fieldmask;
                    fixedbits |= fm.fixedbits;
                    fixedmask |= fm.fixedmask;
                    undefmask |= fm.undefmask;
                    (f, flds)
                }
                None => self.infer_format(arg, fieldmask, flds, width),
            };
            let fm = &self.m.formats[fmt];
            let set = &self.m.arguments[fm.base];
            for (f, _) in &flds {
                if !set.fields.contains(f) {
                    return Err(
                        self.error(lineno, format!("field {f} not in argument set {}", set.name))
                    );
                }
                if fields_get(&fm.fields, f).is_some() {
                    return Err(self.error(lineno, format!("field {f} set by format and pattern")));
                }
            }
            for f in &set.fields {
                if fields_get(&flds, f).is_none() && fields_get(&fm.fields, f).is_none() {
                    return Err(self.error(lineno, format!("field {f} not initialized")));
                }
            }
            self.m.patterns.push(General {
                name: name.to_string(),
                file: self.ctx.input_file.clone(),
                lineno,
                base: fmt,
                fixedbits,
                fixedmask,
                undefmask,
                fieldmask,
                fields: flds,
                width,
            });
            let p = self.m.patterns.len() - 1;
            if let Some(g) = parent {
                self.m.groups[g].pats.push(Node::Pattern(p));
            }
            self.m.allpatterns.push(p);
        }

        let whex = |v| self.ctx.whex(v);
        if fieldmask & fixedmask != 0 {
            return Err(self.error(
                lineno,
                format!(
                    "fieldmask overlaps fixedmask  ({} & {})",
                    whex(fieldmask),
                    whex(fixedmask)
                ),
            ));
        }
        if fieldmask & undefmask != 0 {
            return Err(self.error(
                lineno,
                format!(
                    "fieldmask overlaps undefmask  ({} & {})",
                    whex(fieldmask),
                    whex(undefmask)
                ),
            ));
        }
        if fixedmask & undefmask != 0 {
            return Err(self.error(
                lineno,
                format!(
                    "fixedmask overlaps undefmask  ({} & {})",
                    whex(fixedmask),
                    whex(undefmask)
                ),
            ));
        }
        if !is_format {
            let allbits = fieldmask | fixedmask | undefmask;
            if allbits != self.ctx.insnmask {
                return Err(self.error(
                    lineno,
                    format!("bits left unspecified  ({})", whex(allbits ^ self.ctx.insnmask)),
                ));
            }
        }
        Ok(())
    }

    fn new_group(&mut self, overlapping: bool, lineno: usize) -> usize {
        self.m.groups.push(Group {
            overlapping,
            file: self.ctx.input_file.clone(),
            lineno,
            pats: Vec::new(),
            fixedbits: 0,
            fixedmask: 0,
            undefmask: 0,
            width: None,
            tree: None,
        });
        self.m.groups.len() - 1
    }

    /// `parse_file`: one input file, adding its patterns to the top level group `top`.
    pub(crate) fn parse_file(&mut self, text: &str, top: usize) -> Result<(), Error> {
        let mut toks: Vec<&str> = Vec::new();
        let mut lineno = 0usize;
        let mut nesting = 0usize;
        let mut nesting_pats: Vec<usize> = Vec::new();
        let mut parent = top;
        let mut indent = 0usize;
        let mut start_lineno = 0usize;

        for raw in lines(text) {
            lineno += 1;
            let line = expandtabs(raw.trim_end());
            let len1 = line.chars().count();
            let line = line.trim_start();
            let len2 = line.chars().count();
            let line = match line.find('#') {
                Some(end) => &line[..end],
                None => line,
            };
            // The tokens borrow from `line`, which is local, so copy them into the text's
            // lifetime by finding them again in the raw line.
            let t: Vec<&str> = split_tokens(raw, line);
            if !toks.is_empty() {
                toks.extend(t);
            } else {
                if len1 == 0 {
                    continue;
                }
                indent = len1 - len2;
                if t.is_empty() {
                    if indent != nesting {
                        return Err(
                            self.error(lineno, format!("indentation  {indent}  !=  {nesting}"))
                        );
                    }
                    continue;
                }
                start_lineno = lineno;
                toks = t;
            }

            if toks.last() == Some(&"\\") {
                toks.pop();
                continue;
            }

            let name = toks.remove(0);

            if name == "}" || name == "]" {
                if !toks.is_empty() {
                    return Err(self.error(start_lineno, "extra tokens after close brace"));
                }
                let parent_is_inc = parent != top && self.m.groups[parent].overlapping;
                if (name == "}") != parent_is_inc {
                    return Err(self.error(lineno, "mismatched close brace"));
                }
                match nesting_pats.pop() {
                    Some(p) => parent = p,
                    None => return Err(self.error(lineno, "extra close brace")),
                }
                nesting -= 2;
                if indent != nesting {
                    return Err(self.error(lineno, format!("indentation  {indent}  !=  {nesting}")));
                }
                toks = Vec::new();
                continue;
            }

            if indent != nesting {
                return Err(
                    self.error(start_lineno, format!("indentation  {indent}  !=  {nesting}"))
                );
            }

            if name == "{" || name == "[" {
                if !toks.is_empty() {
                    return Err(self.error(start_lineno, "extra tokens after open brace"));
                }
                let g = self.new_group(name == "{", start_lineno);
                self.m.groups[parent].pats.push(Node::Group(g));
                nesting_pats.push(parent);
                parent = g;
                nesting += 2;
                toks = Vec::new();
                continue;
            }

            let rest = std::mem::take(&mut toks);
            if let Some(n) = sigil(name, '%') {
                self.parse_field(start_lineno, n, &rest)?;
            } else if let Some(n) = sigil(name, '&') {
                self.parse_arguments(start_lineno, n, &rest)?;
            } else if let Some(n) = sigil(name, '@') {
                self.parse_generic(start_lineno, None, n, &rest)?;
            } else if is_word(name) {
                self.parse_generic(start_lineno, Some(parent), name, &rest)?;
            } else {
                return Err(self.error(lineno, format!("invalid token \"{name}\"")));
            }
        }

        if nesting != 0 {
            return Err(self.error(lineno, "missing close brace"));
        }
        Ok(())
    }
}

/// Split the comment-stripped `line` into whitespace separated tokens. The tokens of a line never
/// contain a tab, and expanding tabs only changes whitespace, so each token is also a substring
/// of the raw line, which is what lets them outlive the expanded copy.
fn split_tokens<'a>(raw: &'a str, line: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut from = 0usize;
    for tok in line.split(is_py_space).filter(|s| !s.is_empty()) {
        match raw[from..].find(tok) {
            Some(at) => {
                out.push(&raw[from + at..from + at + tok.len()]);
                from += at + tok.len();
            }
            None => unreachable!("a token is always part of its line"),
        }
    }
    out
}

/// Python's `str.isspace` for the characters that can appear in a decode file.
fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

#[cfg(test)]
mod tests {
    use super::{expandtabs, lines};

    #[test]
    fn tabs_expand_to_columns_of_eight() {
        assert_eq!(expandtabs("ab\tc"), "ab      c");
        assert_eq!(expandtabs("\t"), "        ");
    }

    #[test]
    fn universal_newlines() {
        assert_eq!(lines("a\r\nb\rc\nd"), ["a", "b", "c", "d"]);
        assert_eq!(lines("a\n"), ["a"]);
    }
}
