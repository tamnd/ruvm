// SPDX-License-Identifier: GPL-2.0-or-later

//! Reads QEMU's `.hx` tables, the way scripts/hxtool and the C preprocessor do.
//!
//! An `.hx` file is C with documentation mixed in. hxtool drops the `HXCOMM` lines and
//! everything between `SRST` and `ERST`, and what is left is included into C files that define
//! `DEF()`, `DEFHEADING()` and `ARCHHEADING()` as they need. [`parse_options`] does both steps
//! for qemu-options.hx: it evaluates the `#if` lines against the build's conditions and returns
//! the headings and option definitions in file order, with the help strings joined as the C
//! compiler would join them.

use std::fmt::Write as _;

/// One entry of qemu-options.hx.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptionsEntry {
    /// `DEFHEADING(text)` or `ARCHHEADING(text, mask)`. `help()` prints the text as a line of
    /// its own, so an empty heading is a blank line.
    Heading {
        text: String,
        arch: Vec<String>,
    },
    Def(OptionDef),
}

/// `DEF(option, opt_arg, opt_enum, opt_help, arch_mask)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionDef {
    /// The option without its dash.
    pub name: String,
    pub has_arg: bool,
    /// The `QEMU_OPTION_*` enumerator.
    pub enum_name: String,
    pub help: String,
    /// The `QEMU_ARCH_*` names or'ed together in the mask.
    pub arch: Vec<String>,
}

/// What hxtool leaves for the C compiler: no comments and no rST.
fn hxtool(text: &str) -> String {
    let mut out = String::new();
    let mut in_rst = false;
    for line in text.lines() {
        if line.starts_with("SRST") {
            in_rst = true;
        } else if line.starts_with("ERST") {
            in_rst = false;
        } else if !in_rst && !line.starts_with("HXCOMM") {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Evaluates the preprocessor lines and returns the text of the branches that are taken.
/// Backslash newlines are joined first, as translation phase 2 does.
fn preprocess(text: &str, is_set: &dyn Fn(&str) -> Option<bool>) -> Result<String, String> {
    struct Frame {
        /// Whether the enclosing text is live.
        outer: bool,
        /// Whether some branch of this `#if` was taken already.
        taken: bool,
        /// Whether the current branch is live.
        live: bool,
    }
    let joined = text.replace("\\\n", "");
    let mut out = String::new();
    let mut stack: Vec<Frame> = Vec::new();
    let live = |stack: &[Frame]| stack.last().is_none_or(|f| f.live);
    for (i, line) in joined.lines().enumerate() {
        let Some(directive) = line.trim_start().strip_prefix('#') else {
            if live(&stack) {
                out.push_str(line);
            }
            out.push('\n');
            continue;
        };
        let directive = directive.trim_start();
        let (word, rest) = directive.split_once(char::is_whitespace).unwrap_or((directive, ""));
        let err = |e: String| format!("line {}: {e}", i + 1);
        match word {
            "if" | "ifdef" | "ifndef" => {
                let value = match word {
                    "if" => eval(rest, is_set).map_err(err)?,
                    "ifdef" => lookup(rest.trim(), is_set).map_err(err)?,
                    _ => !lookup(rest.trim(), is_set).map_err(err)?,
                };
                let outer = live(&stack);
                stack.push(Frame { outer, taken: value, live: outer && value });
            }
            "elif" => {
                let value = eval(rest, is_set).map_err(err)?;
                let f = stack.last_mut().ok_or_else(|| err("#elif without #if".into()))?;
                f.live = f.outer && !f.taken && value;
                f.taken |= value;
            }
            "else" => {
                let f = stack.last_mut().ok_or_else(|| err("#else without #if".into()))?;
                f.live = f.outer && !f.taken;
                f.taken = true;
            }
            "endif" => {
                stack.pop().ok_or_else(|| err("#endif without #if".into()))?;
            }
            // The HMP tables undefine their macros at the end, which means nothing here.
            "undef" | "define" => {}
            _ => return Err(err(format!("unknown directive #{word}"))),
        }
        out.push('\n');
    }
    if !stack.is_empty() {
        return Err("#if without #endif".into());
    }
    Ok(out)
}

fn lookup(sym: &str, is_set: &dyn Fn(&str) -> Option<bool>) -> Result<bool, String> {
    is_set(sym).ok_or_else(|| format!("unknown condition {sym}"))
}

/// Evaluates an `#if` expression made of `defined(X)`, `!`, `&&`, `||` and parentheses.
fn eval(expr: &str, is_set: &dyn Fn(&str) -> Option<bool>) -> Result<bool, String> {
    let mut toks = Vec::new();
    let mut chars = expr.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c.is_ascii_alphanumeric() || c == '_' {
            let mut id = String::new();
            while let Some(&c) = chars.peek() {
                if !(c.is_ascii_alphanumeric() || c == '_') {
                    break;
                }
                id.push(c);
                chars.next();
            }
            toks.push(id);
        } else if c == '&' || c == '|' {
            chars.next();
            if chars.next() != Some(c) {
                return Err(format!("bad operator in #if {expr}"));
            }
            toks.push(format!("{c}{c}"));
        } else if "()!".contains(c) {
            chars.next();
            toks.push(c.to_string());
        } else {
            return Err(format!("unexpected {c:?} in #if {expr}"));
        }
    }
    let mut p = Expr { toks: &toks, pos: 0, is_set };
    let v = p.or()?;
    if p.pos != toks.len() {
        return Err(format!("trailing tokens in #if {expr}"));
    }
    Ok(v)
}

struct Expr<'a> {
    toks: &'a [String],
    pos: usize,
    is_set: &'a dyn Fn(&str) -> Option<bool>,
}

impl Expr<'_> {
    fn peek(&self) -> Option<&str> {
        self.toks.get(self.pos).map(String::as_str)
    }

    fn expect(&mut self, t: &str) -> Result<(), String> {
        if self.peek() == Some(t) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected {t} in #if"))
        }
    }

    fn or(&mut self) -> Result<bool, String> {
        let mut v = self.and()?;
        while self.peek() == Some("||") {
            self.pos += 1;
            v |= self.and()?;
        }
        Ok(v)
    }

    fn and(&mut self) -> Result<bool, String> {
        let mut v = self.unary()?;
        while self.peek() == Some("&&") {
            self.pos += 1;
            v &= self.unary()?;
        }
        Ok(v)
    }

    fn unary(&mut self) -> Result<bool, String> {
        match self.peek() {
            Some("!") => {
                self.pos += 1;
                Ok(!self.unary()?)
            }
            Some("(") => {
                self.pos += 1;
                let v = self.or()?;
                self.expect(")")?;
                Ok(v)
            }
            Some("defined") => {
                self.pos += 1;
                let paren = self.peek() == Some("(");
                if paren {
                    self.pos += 1;
                }
                let sym = self.peek().ok_or("defined without a name")?.to_string();
                self.pos += 1;
                if paren {
                    self.expect(")")?;
                }
                lookup(&sym, self.is_set)
            }
            Some(t) => Err(format!("unexpected {t} in #if")),
            None => Err("unexpected end of #if".into()),
        }
    }
}

/// A cursor over the preprocessed text, for the macro calls in it.
struct Cursor<'a> {
    s: &'a str,
    pos: usize,
    line: usize,
    macros: &'a dyn Fn(&str) -> Option<String>,
}

impl Cursor<'_> {
    fn err(&self, msg: &str) -> String {
        format!("line {}: {msg}", self.line)
    }

    fn peek(&self) -> Option<char> {
        self.s[self.pos..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        if c == '\n' {
            self.line += 1;
        }
        Some(c)
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn ident(&mut self) -> String {
        let mut id = String::new();
        while let Some(c) = self.peek() {
            if !(c.is_ascii_alphanumeric() || c == '_') {
                break;
            }
            id.push(c);
            self.bump();
        }
        id
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        self.skip_ws();
        if self.bump() == Some(c) { Ok(()) } else { Err(self.err(&format!("expected {c:?}"))) }
    }

    /// Raw text up to one of `stops` at nesting depth 0, which is what a macro argument is
    /// before it is stringified.
    fn raw_arg(&mut self, stops: &[char]) -> Result<String, String> {
        let mut out = String::new();
        let mut depth = 0;
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated macro call"))?;
            if depth == 0 && stops.contains(&c) {
                return Ok(out);
            }
            match c {
                '(' => depth += 1,
                ')' => depth -= 1,
                _ => {}
            }
            out.push(c);
            self.bump();
        }
    }

    /// One or more adjacent string literals, joined. A macro between them stands for the
    /// string it expands to.
    fn strings(&mut self) -> Result<String, String> {
        let mut out = String::new();
        let mut any = false;
        loop {
            self.skip_ws();
            if self.peek().is_some_and(|c| c.is_ascii_uppercase()) {
                let save = (self.pos, self.line);
                let id = self.ident();
                if let Some(text) = (self.macros)(&id) {
                    out.push_str(&text);
                    any = true;
                    continue;
                }
                (self.pos, self.line) = save;
                break;
            }
            if self.peek() != Some('"') {
                break;
            }
            any = true;
            self.bump();
            loop {
                match self.bump().ok_or_else(|| self.err("unterminated string"))? {
                    '"' => break,
                    '\\' => {
                        let e = self.bump().ok_or_else(|| self.err("unterminated string"))?;
                        out.push(match e {
                            'n' => '\n',
                            't' => '\t',
                            '"' => '"',
                            '\'' => '\'',
                            '\\' => '\\',
                            _ => return Err(self.err(&format!("unknown escape \\{e}"))),
                        });
                    }
                    c => out.push(c),
                }
            }
        }
        if any { Ok(out) } else { Err(self.err("expected a string")) }
    }

    fn arch_mask(&mut self) -> Result<Vec<String>, String> {
        let mut arch = Vec::new();
        loop {
            self.skip_ws();
            let id = self.ident();
            if id.is_empty() {
                return Err(self.err("expected a QEMU_ARCH_ name"));
            }
            arch.push(id);
            self.skip_ws();
            if self.peek() != Some('|') {
                return Ok(arch);
            }
            self.bump();
        }
    }
}

/// Collapses whitespace the way `#` stringification does.
fn stringify(raw: &str) -> String {
    raw.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Reads qemu-options.hx. `is_set` answers for each condition an `#if` line names and returns
/// `None` for one the caller does not know, which is an error. `macros` expands the few macros
/// the help strings use, such as `DEFAULT_BRIDGE_INTERFACE`.
pub fn parse_options(
    text: &str,
    is_set: &dyn Fn(&str) -> Option<bool>,
    macros: &dyn Fn(&str) -> Option<String>,
) -> Result<Vec<OptionsEntry>, String> {
    let text = preprocess(&hxtool(text), is_set)?;
    let mut c = Cursor { s: &text, pos: 0, line: 1, macros };
    let mut out = Vec::new();
    loop {
        c.skip_ws();
        if c.peek().is_none() {
            return Ok(out);
        }
        let name = c.ident();
        c.expect('(')?;
        match name.as_str() {
            "DEFHEADING" => {
                let text = stringify(&c.raw_arg(&[')'])?);
                out.push(OptionsEntry::Heading { text, arch: vec!["QEMU_ARCH_ALL".into()] });
            }
            "ARCHHEADING" => {
                let text = stringify(&c.raw_arg(&[','])?);
                c.expect(',')?;
                let arch = c.arch_mask()?;
                out.push(OptionsEntry::Heading { text, arch });
            }
            "DEF" => {
                let name = c.strings()?;
                c.expect(',')?;
                c.skip_ws();
                let has_arg = match c.ident().as_str() {
                    "HAS_ARG" => true,
                    "0" => false,
                    _ => return Err(c.err("expected HAS_ARG or 0")),
                };
                c.expect(',')?;
                c.skip_ws();
                let enum_name = c.ident();
                c.expect(',')?;
                let help = c.strings()?;
                c.expect(',')?;
                let arch = c.arch_mask()?;
                out.push(OptionsEntry::Def(OptionDef { name, has_arg, enum_name, help, arch }));
            }
            "" => return Err(c.err("expected a macro call")),
            other => return Err(c.err(&format!("unknown macro {other}"))),
        }
        c.expect(')')?;
    }
}

/// Writes a string as a Rust literal.
pub fn rust_str(s: &str) -> String {
    let mut out = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => {
                let _ = write!(out, "{c}");
            }
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(sym: &str) -> Option<bool> {
        match sym {
            "CONFIG_A" => Some(true),
            "CONFIG_B" => Some(false),
            _ => None,
        }
    }

    const TEXT: &str = r#"HXCOMM a comment
DEFHEADING(Standard options:)

DEF("help", 0, QEMU_OPTION_h,
    "-h              help\n", QEMU_ARCH_ALL)
SRST
``-h``
    DEF("fake", 0, QEMU_OPTION_fake, "", QEMU_ARCH_ALL)
ERST
DEF("machine", HAS_ARG, QEMU_OPTION_machine, \
    "-machine [type=]name\n" DEFAULT_X "\n"
#ifdef CONFIG_A
    "  a is on\n"
#endif
#if defined(CONFIG_B) || !defined(CONFIG_A)
    "  b is on\n"
#elif defined(CONFIG_A) && !defined(CONFIG_B)
    "  elif \"taken\"\n"
#else
    "  else\n"
#endif
    , QEMU_ARCH_ARM | QEMU_ARCH_I386)
#ifdef CONFIG_B
DEF("gone", 0, QEMU_OPTION_gone, "", QEMU_ARCH_ALL)
#endif
ARCHHEADING(i386 target  only:, QEMU_ARCH_I386)
DEFHEADING()
"#;

    #[test]
    fn options() {
        let macros = |m: &str| (m == "DEFAULT_X").then(|| "x".to_string());
        let e = parse_options(TEXT, &set, &macros).unwrap();
        assert_eq!(e.len(), 5);
        assert_eq!(
            e[0],
            OptionsEntry::Heading {
                text: "Standard options:".into(),
                arch: vec!["QEMU_ARCH_ALL".into()]
            }
        );
        let OptionsEntry::Def(h) = &e[1] else { panic!() };
        assert_eq!(
            (h.name.as_str(), h.has_arg, h.enum_name.as_str()),
            ("help", false, "QEMU_OPTION_h")
        );
        let OptionsEntry::Def(m) = &e[2] else { panic!() };
        assert!(m.has_arg);
        assert_eq!(m.help, "-machine [type=]name\nx\n  a is on\n  elif \"taken\"\n");
        assert_eq!(m.arch, ["QEMU_ARCH_ARM", "QEMU_ARCH_I386"]);
        assert_eq!(
            e[3],
            OptionsEntry::Heading {
                text: "i386 target only:".into(),
                arch: vec!["QEMU_ARCH_I386".into()]
            }
        );
        assert_eq!(
            e[4],
            OptionsEntry::Heading { text: String::new(), arch: vec!["QEMU_ARCH_ALL".into()] }
        );
    }

    #[test]
    fn unknown_conditions_are_errors() {
        let err = parse_options("#ifdef CONFIG_C\n#endif\n", &set, &|_| None).unwrap_err();
        assert_eq!(err, "line 1: unknown condition CONFIG_C");
        assert!(parse_options("#if defined(CONFIG_A)\n", &set, &|_| None).is_err());
    }
}
