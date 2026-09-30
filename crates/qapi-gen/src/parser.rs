// SPDX-License-Identifier: GPL-2.0-or-later

//! The schema file reader, ported from scripts/qapi/parser.py and source.py.
//!
//! A schema file is a sequence of JSON-like objects with single quoted strings and `#` comments.
//! `include` directives pull other files in at that point, each file at most once. Doc comments
//! are skipped: ruvm does not generate documentation from them.

use std::fmt;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// A parsed schema value. Objects keep their key order, which the generators depend on.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Str(String),
    Bool(bool),
    List(Vec<Value>),
    Dict(Vec<(String, Value)>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l),
            _ => None,
        }
    }

    pub fn as_dict(&self) -> Option<&[(String, Value)]> {
        match self {
            Value::Dict(d) => Some(d),
            _ => None,
        }
    }

    /// Looks up a key in an object.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

/// Where something came from: `QAPISourceInfo`.
#[derive(Clone, Debug)]
pub struct SourceInfo {
    pub fname: PathBuf,
    pub line: u32,
    pub parent: Option<Rc<SourceInfo>>,
    pub defn: Option<(String, String)>,
}

impl SourceInfo {
    fn new(fname: PathBuf, parent: Option<Rc<SourceInfo>>) -> Self {
        SourceInfo { fname, line: 1, parent, defn: None }
    }

    pub fn loc(&self) -> String {
        format!("{}:{}", self.fname.display(), self.line)
    }
}

impl fmt::Display for SourceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut chain = Vec::new();
        let mut p = self.parent.as_deref();
        while let Some(i) = p {
            chain.push(i);
            p = i.parent.as_deref();
        }
        for i in chain.iter().rev() {
            writeln!(f, "In file included from {}:", i.loc())?;
        }
        if let Some((meta, name)) = &self.defn {
            writeln!(f, "{}: In {} '{}':", self.fname.display(), meta, name)?;
        }
        f.write_str(&self.loc())
    }
}

/// `QAPIError` and its subclasses, printed the way QEMU prints them.
#[derive(Clone, Debug)]
pub struct Error {
    pub info: Option<Box<SourceInfo>>,
    pub col: Option<usize>,
    pub msg: String,
}

impl Error {
    pub fn new(info: &SourceInfo, msg: impl Into<String>) -> Self {
        Error { info: Some(Box::new(info.clone())), col: None, msg: msg.into() }
    }

    pub fn bare(msg: impl Into<String>) -> Self {
        Error { info: None, col: None, msg: msg.into() }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&self.info, self.col) {
            (None, _) => f.write_str(&self.msg),
            (Some(info), None) => write!(f, "{}: {}", info, self.msg),
            (Some(info), Some(col)) => write!(f, "{}:{}: {}", info, col, self.msg),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// One top level expression with the place it was defined.
#[derive(Clone, Debug)]
pub struct Expression {
    pub expr: Value,
    pub info: SourceInfo,
}

impl Expression {
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.expr.get(key)
    }

    pub fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Punct(char),
    Str(String),
    Bool(bool),
    Eof,
}

/// `QAPISchemaParser`.
#[derive(Debug)]
pub struct Parser {
    src: Vec<char>,
    info: SourceInfo,
    tok: Tok,
    pos: usize,
    cursor: usize,
    line_pos: usize,
    pub exprs: Vec<Expression>,
}

impl Parser {
    /// Parses `fname` and everything it includes.
    pub fn parse_file(fname: &Path) -> Result<Vec<Expression>> {
        let mut included = Vec::new();
        let p = Parser::new(fname, &mut included, None).map_err(|e| match e {
            Ok(e) => e,
            Err(io) => Error::bare(format!("can't read schema file '{}': {}", fname.display(), io)),
        })?;
        Ok(p.exprs)
    }

    fn new(
        fname: &Path,
        included: &mut Vec<PathBuf>,
        parent: Option<Rc<SourceInfo>>,
    ) -> std::result::Result<Parser, std::result::Result<Error, String>> {
        included.push(absolute(fname));
        let mut text = std::fs::read_to_string(fname).map_err(|e| Err(strerror(&e)))?;
        if !text.ends_with('\n') {
            text.push('\n');
        }
        let mut p = Parser {
            src: text.chars().collect(),
            info: SourceInfo::new(fname.to_path_buf(), parent),
            tok: Tok::Eof,
            pos: 0,
            cursor: 0,
            line_pos: 0,
            exprs: Vec::new(),
        };
        p.run(fname, included).map_err(Ok)?;
        Ok(p)
    }

    fn run(&mut self, fname: &Path, included: &mut Vec<PathBuf>) -> Result<()> {
        self.accept()?;
        while self.tok != Tok::Eof {
            let info = self.info.clone();
            let expr = self.get_expr()?;
            let Value::Dict(members) = &expr else {
                return Err(Error::new(&info, "top-level expression must be an object"));
            };
            if let Some(include) = expr.get("include") {
                if members.len() != 1 {
                    return Err(Error::new(&info, "invalid 'include' directive"));
                }
                let Some(include) = include.as_str() else {
                    return Err(Error::new(&info, "value of 'include' must be a string"));
                };
                let incl_fname = fname.parent().unwrap_or(Path::new("")).join(include);
                self.exprs.push(Expression {
                    expr: Value::Dict(vec![(
                        "include".into(),
                        Value::Str(incl_fname.to_string_lossy().into_owned()),
                    )]),
                    info: info.clone(),
                });
                let abs = absolute(&incl_fname);
                let mut inf = Some(&info);
                while let Some(i) = inf {
                    if absolute(&i.fname) == abs {
                        return Err(Error::new(&info, format!("inclusion loop for {include}")));
                    }
                    inf = i.parent.as_deref();
                }
                if included.contains(&abs) {
                    continue;
                }
                match Parser::new(&incl_fname, included, Some(Rc::new(info.clone()))) {
                    Ok(p) => self.exprs.extend(p.exprs),
                    Err(Ok(e)) => return Err(e),
                    Err(Err(io)) => {
                        return Err(Error::new(
                            &info,
                            format!("can't read include file '{}': {}", incl_fname.display(), io),
                        ));
                    }
                }
            } else if let Some(pragma) = expr.get("pragma") {
                if members.len() != 1 {
                    return Err(Error::new(&info, "invalid 'pragma' directive"));
                }
                let Some(pragma) = pragma.as_dict() else {
                    return Err(Error::new(&info, "value of 'pragma' must be an object"));
                };
                for (name, value) in pragma {
                    check_pragma(name, value, &info)?;
                }
            } else {
                self.exprs.push(Expression { expr, info });
            }
        }
        Ok(())
    }

    fn error(&self, msg: impl Into<String>) -> Error {
        // QAPIParseError: the column counts tabs to the next multiple of 8.
        let mut col = 1;
        for &c in &self.src[self.line_pos..self.pos] {
            if c == '\t' {
                col = (col + 7) % 8 + 1;
            } else {
                col += 1;
            }
        }
        Error { info: Some(Box::new(self.info.clone())), col: Some(col), msg: msg.into() }
    }

    fn peek(&self, at: usize) -> char {
        self.src.get(at).copied().unwrap_or('\n')
    }

    fn starts_with(&self, at: usize, word: &str) -> bool {
        word.chars().enumerate().all(|(i, c)| self.src.get(at + i) == Some(&c))
    }

    fn accept(&mut self) -> Result<()> {
        loop {
            let c = self.peek(self.cursor);
            self.pos = self.cursor;
            self.cursor += 1;
            match c {
                '#' => {
                    while self.peek(self.cursor) != '\n' {
                        self.cursor += 1;
                    }
                }
                '{' | '}' | ':' | ',' | '[' | ']' => {
                    self.tok = Tok::Punct(c);
                    return Ok(());
                }
                '\'' => {
                    let mut s = String::new();
                    let mut esc = false;
                    loop {
                        let ch = self.peek(self.cursor);
                        self.cursor += 1;
                        if ch == '\n' {
                            return Err(self.error("missing terminating \"'\""));
                        }
                        if esc {
                            if ch != '\\' {
                                return Err(self.error(format!("unknown escape \\{ch}")));
                            }
                            esc = false;
                        } else if ch == '\\' {
                            esc = true;
                            continue;
                        } else if ch == '\'' {
                            self.tok = Tok::Str(s);
                            return Ok(());
                        }
                        if (ch as u32) < 32 || (ch as u32) >= 127 {
                            return Err(self.error("funny character in string"));
                        }
                        s.push(ch);
                    }
                }
                _ if self.starts_with(self.pos, "true") => {
                    self.cursor += 3;
                    self.tok = Tok::Bool(true);
                    return Ok(());
                }
                _ if self.starts_with(self.pos, "false") => {
                    self.cursor += 4;
                    self.tok = Tok::Bool(false);
                    return Ok(());
                }
                '\n' => {
                    if self.cursor >= self.src.len() {
                        self.tok = Tok::Eof;
                        return Ok(());
                    }
                    self.info.line += 1;
                    self.line_pos = self.cursor;
                }
                _ if c.is_whitespace() => {}
                _ => {
                    let stray: String = self.src[self.pos..]
                        .iter()
                        .take_while(|c| {
                            !matches!(c, '[' | ']' | '{' | '}' | ':' | ',' | '\'')
                                && !c.is_whitespace()
                        })
                        .collect();
                    return Err(self.error(format!("stray '{stray}'")));
                }
            }
        }
    }

    fn get_members(&mut self) -> Result<Value> {
        let mut out: Vec<(String, Value)> = Vec::new();
        if self.tok == Tok::Punct('}') {
            self.accept()?;
            return Ok(Value::Dict(out));
        }
        let Tok::Str(_) = self.tok else {
            return Err(self.error("expected string or '}'"));
        };
        loop {
            let Tok::Str(key) = std::mem::replace(&mut self.tok, Tok::Eof) else {
                unreachable!("checked by the caller or the loop");
            };
            self.accept()?;
            if self.tok != Tok::Punct(':') {
                return Err(self.error("expected ':'"));
            }
            self.accept()?;
            if out.iter().any(|(k, _)| *k == key) {
                return Err(self.error(format!("duplicate key '{key}'")));
            }
            let v = self.get_expr()?;
            out.push((key, v));
            if self.tok == Tok::Punct('}') {
                self.accept()?;
                return Ok(Value::Dict(out));
            }
            if self.tok != Tok::Punct(',') {
                return Err(self.error("expected ',' or '}'"));
            }
            self.accept()?;
            if !matches!(self.tok, Tok::Str(_)) {
                return Err(self.error("expected string"));
            }
        }
    }

    fn get_values(&mut self) -> Result<Value> {
        let mut out = Vec::new();
        if self.tok == Tok::Punct(']') {
            self.accept()?;
            return Ok(Value::List(out));
        }
        if !matches!(self.tok, Tok::Punct('{') | Tok::Punct('[') | Tok::Str(_) | Tok::Bool(_)) {
            return Err(self.error("expected '{', '[', ']', string, or boolean"));
        }
        loop {
            out.push(self.get_expr()?);
            if self.tok == Tok::Punct(']') {
                self.accept()?;
                return Ok(Value::List(out));
            }
            if self.tok != Tok::Punct(',') {
                return Err(self.error("expected ',' or ']'"));
            }
            self.accept()?;
        }
    }

    fn get_expr(&mut self) -> Result<Value> {
        match self.tok.clone() {
            Tok::Punct('{') => {
                self.accept()?;
                self.get_members()
            }
            Tok::Punct('[') => {
                self.accept()?;
                self.get_values()
            }
            Tok::Str(s) => {
                self.accept()?;
                Ok(Value::Str(s))
            }
            Tok::Bool(b) => {
                self.accept()?;
                Ok(Value::Bool(b))
            }
            _ => Err(self.error("expected '{', '[', string, or boolean")),
        }
    }
}

fn check_pragma(name: &str, value: &Value, info: &SourceInfo) -> Result<()> {
    match name {
        "doc-required" => {
            if value.as_bool().is_none() {
                return Err(Error::new(info, "pragma 'doc-required' must be boolean"));
            }
        }
        "command-name-exceptions"
        | "command-returns-exceptions"
        | "documentation-exceptions"
        | "member-name-exceptions" => {
            let ok = value.as_list().is_some_and(|l| l.iter().all(|v| v.as_str().is_some()));
            if !ok {
                return Err(Error::new(info, format!("pragma {name} must be a list of strings")));
            }
        }
        _ => return Err(Error::new(info, format!("unknown pragma '{name}'"))),
    }
    Ok(())
}

fn absolute(p: &Path) -> PathBuf {
    std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf())
}

fn strerror(e: &std::io::Error) -> String {
    let s = e.to_string();
    match s.find(" (os error") {
        Some(i) => s[..i].to_string(),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Result<Vec<Expression>> {
        let dir = std::env::temp_dir().join(format!(
            "ruvm-qapi-gen-{}-{}",
            std::process::id(),
            text.len()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("t.json");
        std::fs::write(&f, text).unwrap();
        let r = Parser::parse_file(&f);
        std::fs::remove_dir_all(&dir).unwrap();
        r
    }

    fn err(text: &str) -> String {
        let e = parse(text).unwrap_err();
        let s = e.to_string();
        s[s.find(".json:").unwrap() + 6..].to_string()
    }

    #[test]
    fn keeps_key_order_and_skips_comments() {
        let exprs = parse("# a comment\n##\n# doc\n##\n{ 'struct': 'S', 'data': { 'b': 'int', '*a': ['str'] } }\n").unwrap();
        assert_eq!(exprs.len(), 1);
        assert_eq!(exprs[0].info.line, 5);
        let data = exprs[0].get("data").unwrap().as_dict().unwrap();
        assert_eq!(data[0].0, "b");
        assert_eq!(data[1].0, "*a");
    }

    #[test]
    fn errors_read_like_qemu() {
        assert_eq!(err("{ 'a': 1 }"), "1:8: stray '1'");
        assert_eq!(err("{ 'a' 'b' }"), "1:7: expected ':'");
        assert_eq!(err("{ 'a': 'b', 'a': 'c' }"), "1:18: duplicate key 'a'");
        assert_eq!(err("\t{ 'a': 'b\n"), "1:8: missing terminating \"'\"");
        assert_eq!(err("[ 'a' ]"), "1: top-level expression must be an object");
        assert_eq!(err("{ 'pragma': { 'x': true } }"), "1: unknown pragma 'x'");
    }
}
