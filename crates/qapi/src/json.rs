// SPDX-License-Identifier: GPL-2.0-or-later

//! QEMU's JSON dialect, ported from qobject/json-lexer.c, json-streamer.c, json-parser.c and
//! json-writer.c.
//!
//! The dialect is JSON plus single quoted strings, and it is read as a stream: [`Streamer`] takes
//! bytes as they arrive on a socket and hands back one value or one error per top level message.
//! After an error it skips input until the braces and brackets balance again, the same way QEMU
//! does, so a client that sends garbage gets exactly the errors QEMU would send and then carries
//! on. The error texts are QEMU's, including the `line:column` prefix.

use std::collections::VecDeque;

use ruvm_base::{Error, Result};

use crate::qvalue::{QDict, QValue};

/// A single token may not grow past this, and neither may the sum of the tokens in one message.
pub const MAX_TOKEN_SIZE: usize = 64 << 20;
/// The most tokens one message may have.
pub const MAX_TOKEN_COUNT: usize = 2 << 20;
/// The deepest one message may nest.
pub const MAX_NESTING: usize = 1 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tok {
    Error,
    LCurly,
    RCurly,
    LSquare,
    RSquare,
    Colon,
    Comma,
    Integer,
    Float,
    Keyword,
    String,
    EndOfInput,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Recovery,
    DqEscape,
    Dq,
    SqEscape,
    Sq,
    Zero,
    ExpDigits,
    ExpSign,
    ExpE,
    Mantissa,
    MantissaDigits,
    Digits,
    Sign,
    Keyword,
    Start,
}

/// What the transition table says for one byte in one state.
enum Next {
    /// Consume the byte and move to a state.
    Go(State),
    /// Consume the byte and finish a token.
    Emit(Tok),
    /// Finish a token without consuming the byte, which is then looked at again from the start.
    EmitLookahead(Tok),
    /// Go back to the start state without consuming the byte.
    StartLookahead,
    /// Consume the byte, report it as an error and enter recovery.
    Error,
}

fn next(state: State, b: u8) -> Next {
    use Next::*;
    match state {
        State::Recovery => match b {
            b'\t' => Go(State::Recovery),
            0..=0x1f | 0xfe | 0xff | b'[' | b']' | b'{' | b'}' | b':' | b',' => StartLookahead,
            _ => Go(State::Recovery),
        },
        State::DqEscape => string_byte(b, State::Dq),
        State::SqEscape => string_byte(b, State::Sq),
        State::Dq => match b {
            b'\\' => Go(State::DqEscape),
            b'"' => Emit(Tok::String),
            _ => string_byte(b, State::Dq),
        },
        State::Sq => match b {
            b'\\' => Go(State::SqEscape),
            b'\'' => Emit(Tok::String),
            _ => string_byte(b, State::Sq),
        },
        State::Zero => match b {
            b'0'..=b'9' => Error,
            b'.' => Go(State::Mantissa),
            _ => EmitLookahead(Tok::Integer),
        },
        State::ExpDigits => match b {
            b'0'..=b'9' => Go(State::ExpDigits),
            _ => EmitLookahead(Tok::Float),
        },
        State::ExpSign => match b {
            b'0'..=b'9' => Go(State::ExpDigits),
            _ => Error,
        },
        State::ExpE => match b {
            b'-' | b'+' => Go(State::ExpSign),
            b'0'..=b'9' => Go(State::ExpDigits),
            _ => Error,
        },
        State::MantissaDigits => match b {
            b'0'..=b'9' => Go(State::MantissaDigits),
            b'e' | b'E' => Go(State::ExpE),
            _ => EmitLookahead(Tok::Float),
        },
        State::Mantissa => match b {
            b'0'..=b'9' => Go(State::MantissaDigits),
            _ => Error,
        },
        State::Digits => match b {
            b'0'..=b'9' => Go(State::Digits),
            b'e' | b'E' => Go(State::ExpE),
            b'.' => Go(State::Mantissa),
            _ => EmitLookahead(Tok::Integer),
        },
        State::Sign => match b {
            b'0' => Go(State::Zero),
            b'1'..=b'9' => Go(State::Digits),
            _ => Error,
        },
        State::Keyword => match b {
            b'a'..=b'z' => Go(State::Keyword),
            _ => EmitLookahead(Tok::Keyword),
        },
        State::Start => match b {
            b'"' => Go(State::Dq),
            b'\'' => Go(State::Sq),
            b'0' => Go(State::Zero),
            b'1'..=b'9' => Go(State::Digits),
            b'-' => Go(State::Sign),
            b'{' => Emit(Tok::LCurly),
            b'}' => Emit(Tok::RCurly),
            b'[' => Emit(Tok::LSquare),
            b']' => Emit(Tok::RSquare),
            b',' => Emit(Tok::Comma),
            b':' => Emit(Tok::Colon),
            b'a'..=b'z' => Go(State::Keyword),
            b' ' | b'\t' | b'\r' | b'\n' => Go(State::Start),
            _ => Error,
        },
    }
}

fn string_byte(b: u8, state: State) -> Next {
    if (0x20..=0xfd).contains(&b) { Next::Go(state) } else { Next::Error }
}

/// The parser's stack, one entry per open container plus one per key waiting for its value.
#[derive(Debug)]
enum Frame {
    Dict(QDict, ParseState),
    List(Vec<QValue>, ParseState),
    Key(String, ParseState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParseState {
    AfterLCurly,
    AfterLSquare,
    BeforeKey,
    BeforeValue,
    EndOfKey,
    EndOfValue,
}

impl Frame {
    fn state_mut(&mut self) -> &mut ParseState {
        match self {
            Frame::Dict(_, s) | Frame::List(_, s) | Frame::Key(_, s) => s,
        }
    }
}

/// QEMU's `JSONMessageParser`: a lexer, the message splitter and the push parser in one.
#[derive(Debug)]
pub struct Streamer {
    state: State,
    token: Vec<u8>,
    x: u32,
    y: u32,
    cur_x: u32,
    cur_y: u32,

    brace_count: usize,
    bracket_count: usize,
    token_count: usize,
    token_size: usize,
    error: bool,

    stack: Vec<Frame>,
    out: VecDeque<Result<QValue>>,
}

impl Default for Streamer {
    fn default() -> Self {
        Self::new()
    }
}

impl Streamer {
    pub fn new() -> Self {
        Streamer {
            state: State::Start,
            token: Vec::new(),
            x: 1,
            y: 1,
            cur_x: 1,
            cur_y: 1,
            brace_count: 0,
            bracket_count: 0,
            token_count: 0,
            token_size: 0,
            error: false,
            stack: Vec::new(),
            out: VecDeque::new(),
        }
    }

    /// `json_message_parser_feed()`. Finished values and errors queue up for [`Streamer::next`].
    pub fn feed(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.feed_byte(b);
        }
    }

    /// `json_message_parser_flush()`: finish whatever token is open and report end of input.
    pub fn flush(&mut self) {
        self.cur_x += 1;
        while self.state != State::Start {
            match next(self.state, 0) {
                Next::Emit(t) | Next::EmitLookahead(t) => self.finish(t),
                Next::StartLookahead | Next::Go(_) => self.restart(),
                Next::Error => {
                    self.process(Tok::Error);
                    self.token.clear();
                    self.state = State::Recovery;
                }
            }
        }
        self.process(Tok::EndOfInput);
    }

    /// The next finished value or error, oldest first.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Result<QValue>> {
        self.out.pop_front()
    }

    fn restart(&mut self) {
        self.token.clear();
        self.state = State::Start;
        self.x = self.cur_x;
        self.y = self.cur_y;
    }

    fn finish(&mut self, tok: Tok) {
        self.process(tok);
        self.restart();
    }

    fn feed_byte(&mut self, b: u8) {
        self.cur_x += 1;
        if b == b'\n' {
            self.cur_x = 1;
            self.cur_y += 1;
        }
        loop {
            match next(self.state, b) {
                Next::Go(State::Start) => {
                    self.restart();
                    break;
                }
                Next::Go(State::Recovery) => {
                    self.token.clear();
                    self.state = State::Recovery;
                    break;
                }
                Next::Go(s) => {
                    self.token.push(b);
                    self.state = s;
                    break;
                }
                Next::Emit(t) => {
                    self.token.push(b);
                    self.finish(t);
                    break;
                }
                Next::EmitLookahead(t) => self.finish(t),
                Next::StartLookahead => self.restart(),
                Next::Error => {
                    self.token.push(b);
                    self.process(Tok::Error);
                    self.token.clear();
                    self.state = State::Recovery;
                    break;
                }
            }
        }
        if self.token.len() > MAX_TOKEN_SIZE {
            // QEMU hands the parser a token typed with the lexer state here, which ends up as a
            // parse error for anything that is not a complete token. Report it as a stray token.
            self.process(Tok::Error);
            self.token.clear();
            self.state = State::Start;
        }
    }

    /// `json_message_process_token()`.
    fn process(&mut self, tok: Tok) {
        self.token_size += self.token.len();
        self.token_count += 1;

        let mut unbalanced = false;
        match tok {
            Tok::LCurly => self.brace_count += 1,
            Tok::RCurly if self.brace_count > 0 => self.brace_count -= 1,
            Tok::LSquare => self.bracket_count += 1,
            Tok::RSquare if self.bracket_count > 0 => self.bracket_count -= 1,
            Tok::RCurly | Tok::RSquare | Tok::Error => unbalanced = true,
            _ => {}
        }
        if unbalanced {
            self.brace_count = 0;
            self.bracket_count = 0;
        }

        if !self.error {
            let result = if self.token_size >= MAX_TOKEN_SIZE {
                Err(Error::generic("JSON token size limit exceeded"))
            } else if self.token_count > MAX_TOKEN_COUNT {
                Err(Error::generic("JSON token count limit exceeded"))
            } else if self.bracket_count + self.brace_count > MAX_NESTING {
                Err(Error::generic("JSON nesting depth limit exceeded"))
            } else {
                let token = std::mem::take(&mut self.token);
                let r = self.parse_feed(tok, &token);
                self.token = token;
                r.map(|v| v.map(|v| self.out.push_back(Ok(v)))).map(|_| ())
            };
            if let Err(e) = result {
                self.out.push_back(Err(e));
                self.error = true;
            }
        }

        if (self.brace_count == 0 && self.bracket_count == 0) || tok == Tok::EndOfInput {
            self.stack.clear();
            self.error = false;
            self.brace_count = 0;
            self.bracket_count = 0;
            self.token_count = 0;
            self.token_size = 0;
        }
    }

    fn parse_error(&self, msg: impl std::fmt::Display) -> Error {
        Error::generic(format!("{}:{}: JSON parse error, {}", self.y, self.x, msg))
    }

    /// `json_parser_feed()`.
    fn parse_feed(&mut self, tok: Tok, text: &[u8]) -> Result<Option<QValue>> {
        match tok {
            Tok::Error => {
                Err(self.parse_error(format_args!("stray '{}'", String::from_utf8_lossy(text))))
            }
            Tok::EndOfInput => {
                if self.stack.is_empty() {
                    Ok(None)
                } else {
                    Err(self.parse_error("premature end of input"))
                }
            }
            _ => self.parse_token(tok, text),
        }
    }

    /// Starts a value: pushes a container or returns a scalar.
    fn begin_value(&mut self, tok: Tok, text: &[u8]) -> Result<Option<QValue>> {
        match tok {
            Tok::LCurly => {
                self.stack.push(Frame::Dict(QDict::new(), ParseState::AfterLCurly));
                Ok(None)
            }
            Tok::LSquare => {
                self.stack.push(Frame::List(Vec::new(), ParseState::AfterLSquare));
                Ok(None)
            }
            Tok::String => self.parse_string(text).map(|s| Some(QValue::Str(s))),
            Tok::Integer => Ok(Some(parse_integer(text))),
            Tok::Float => Ok(Some(QValue::Double(parse_float(text)))),
            Tok::Keyword => match text {
                b"true" => Ok(Some(QValue::Bool(true))),
                b"false" => Ok(Some(QValue::Bool(false))),
                b"null" => Ok(Some(QValue::Null)),
                _ => Err(self.parse_error(format_args!(
                    "invalid keyword '{}'",
                    String::from_utf8_lossy(text)
                ))),
            },
            _ => Err(self.parse_error("expecting value")),
        }
    }

    fn parse_token(&mut self, tok: Tok, text: &[u8]) -> Result<Option<QValue>> {
        let state = self.stack.last().map_or(ParseState::BeforeValue, |f| match f {
            Frame::Dict(_, s) | Frame::List(_, s) | Frame::Key(_, s) => *s,
        });
        let value = match state {
            ParseState::AfterLCurly | ParseState::BeforeKey => {
                if state == ParseState::AfterLCurly && tok == Tok::RCurly {
                    self.pop_container()
                } else {
                    *self.top().state_mut() = ParseState::BeforeKey;
                    if tok != Tok::String {
                        return Err(self.parse_error("expecting key"));
                    }
                    let key = self.parse_string(text)?;
                    self.stack.push(Frame::Key(key, ParseState::EndOfKey));
                    return Ok(None);
                }
            }
            ParseState::EndOfKey => {
                if tok != Tok::Colon {
                    return Err(self.parse_error("expecting ':'"));
                }
                *self.top().state_mut() = ParseState::BeforeValue;
                return Ok(None);
            }
            ParseState::AfterLSquare | ParseState::BeforeValue => {
                if state == ParseState::AfterLSquare && tok == Tok::RSquare {
                    self.pop_container()
                } else {
                    if state == ParseState::AfterLSquare {
                        *self.top().state_mut() = ParseState::BeforeValue;
                    }
                    match self.begin_value(tok, text)? {
                        Some(v) => v,
                        None => return Ok(None),
                    }
                }
            }
            ParseState::EndOfValue => {
                let (is_list, close) = match self.stack.last() {
                    Some(Frame::List(..)) => (true, Tok::RSquare),
                    _ => (false, Tok::RCurly),
                };
                if tok != close {
                    if tok == Tok::Comma {
                        *self.top().state_mut() =
                            if is_list { ParseState::BeforeValue } else { ParseState::BeforeKey };
                        return Ok(None);
                    }
                    return Err(self.parse_error(if is_list {
                        "expected ',' or ']'"
                    } else {
                        "expected ',' or '}'"
                    }));
                }
                self.pop_container()
            }
        };

        match self.stack.pop() {
            None => Ok(Some(value)),
            Some(Frame::Key(key, _)) => {
                let Some(Frame::Dict(dict, state)) = self.stack.last_mut() else {
                    unreachable!("a key always sits on a dict");
                };
                if dict.contains_key(&key) {
                    return Err(self.parse_error("duplicate key"));
                }
                dict.put(key, value);
                *state = ParseState::EndOfValue;
                Ok(None)
            }
            Some(Frame::List(mut list, _)) => {
                list.push(value);
                self.stack.push(Frame::List(list, ParseState::EndOfValue));
                Ok(None)
            }
            Some(Frame::Dict(..)) => unreachable!("a value never lands directly on a dict"),
        }
    }

    fn top(&mut self) -> &mut Frame {
        self.stack.last_mut().expect("parser stack is not empty here")
    }

    fn pop_container(&mut self) -> QValue {
        match self.stack.pop() {
            Some(Frame::Dict(d, _)) => QValue::Dict(d),
            Some(Frame::List(l, _)) => QValue::List(l),
            _ => unreachable!("only containers are closed"),
        }
    }

    /// `parse_string()`: the token still has its quotes.
    fn parse_string(&self, text: &[u8]) -> Result<String> {
        let quote = text[0];
        let body = &text[1..];
        let mut out = String::new();
        let mut i = 0;
        while body[i] != quote {
            match body[i] {
                b'\\' => {
                    let beg = i;
                    let esc = body[i + 1];
                    i += 2;
                    match esc {
                        b'"' => out.push('"'),
                        b'\'' => out.push('\''),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = cvt4hex(&body[i..]);
                            i += 4;
                            if (0xD800..=0xDBFF).contains(&cp)
                                && body.get(i) == Some(&b'\\')
                                && body.get(i + 1) == Some(&b'u')
                            {
                                let trailing = cvt4hex(&body[i + 2..]);
                                if (0xDC00..=0xDFFF).contains(&trailing) {
                                    cp = (0x10000 + ((cp & 0x3FF) << 10)) | (trailing & 0x3FF);
                                    i += 6;
                                } else {
                                    cp = -1;
                                }
                            }
                            match valid_char(cp) {
                                Some(c) => out.push(c),
                                None => {
                                    let end = i.min(body.len());
                                    return Err(self.parse_error(format_args!(
                                        "{} is not a valid Unicode character",
                                        String::from_utf8_lossy(&body[beg..end])
                                    )));
                                }
                            }
                        }
                        _ => return Err(self.parse_error("invalid escape sequence in string")),
                    }
                }
                _ => match mod_utf8_codepoint(&body[i..]) {
                    (Some(c), len) => {
                        out.push(c);
                        i += len;
                    }
                    (None, _) => return Err(self.parse_error("invalid UTF-8 sequence in string")),
                },
            }
        }
        Ok(out)
    }
}

/// `cvt4hex()`, which also gives -1 when fewer than four bytes are left.
fn cvt4hex(s: &[u8]) -> i32 {
    let mut cp = 0i32;
    for i in 0..4 {
        let Some(d) = s.get(i).and_then(|&b| (b as char).to_digit(16)) else {
            return -1;
        };
        cp = cp << 4 | d as i32;
    }
    cp
}

/// `is_valid_codepoint()` from util/unicode.c, which also rules out noncharacters.
fn is_valid_codepoint(cp: i32) -> bool {
    if !(0..=0x10FFFF).contains(&cp) {
        return false;
    }
    if (0xFDD0..=0xFDEF).contains(&cp) || cp & 0xFFFE == 0xFFFE {
        return false;
    }
    !(0xD800..=0xDFFF).contains(&cp)
}

fn valid_char(cp: i32) -> Option<char> {
    if is_valid_codepoint(cp) { char::from_u32(cp as u32) } else { None }
}

/// `mod_utf8_codepoint()`: decodes one character of modified UTF-8, where NUL is `C0 80`.
/// Returns the character, or `None` if the sequence is invalid, and the bytes it covered.
fn mod_utf8_codepoint(s: &[u8]) -> (Option<char>, usize) {
    const MIN_CP: [i32; 5] = [0x80, 0x800, 0x10000, 0x200000, 0x4000000];
    let Some(&first) = s.first() else {
        return (None, 0);
    };
    if first == 0 {
        return (None, 0);
    }
    if first < 0x80 {
        return (Some(first as char), 1);
    }
    if first >= 0xFE || first & 0x40 == 0 {
        return (None, 1);
    }
    let len = first.leading_ones() as usize;
    let mut cp = (first & (0xFF >> (len + 1))) as i32;
    for i in 1..len {
        let b = s.get(i).copied().unwrap_or(0);
        if b & 0xC0 != 0x80 {
            return (None, i);
        }
        cp = cp << 6 | (b & 0x3F) as i32;
    }
    if !is_valid_codepoint(cp) || (cp < MIN_CP[len - 2] && !(cp == 0 && len == 2)) {
        return (None, len);
    }
    (char::from_u32(cp as u32), len)
}

/// A JSON integer is an `Int` if it fits, else a `Uint` if it is not negative, else a `Double`.
fn parse_integer(text: &[u8]) -> QValue {
    let s = std::str::from_utf8(text).unwrap_or_default();
    if let Ok(v) = s.parse::<i64>() {
        QValue::Int(v)
    } else if let (false, Ok(v)) = (s.starts_with('-'), s.parse::<u64>()) {
        QValue::Uint(v)
    } else {
        QValue::Double(parse_float(text))
    }
}

fn parse_float(text: &[u8]) -> f64 {
    // The lexer only lets through what strtod and Rust agree on.
    std::str::from_utf8(text).ok().and_then(|s| s.parse().ok()).unwrap_or(0.0)
}

/// `qobject_from_json()`: exactly one value, or an error. Input stops at the first NUL, since
/// QEMU treats it as a C string.
pub fn from_str(input: &str) -> Result<QValue> {
    from_bytes(input.as_bytes())
}

/// [`from_str`] for raw bytes, which may be invalid UTF-8.
pub fn from_bytes(input: &[u8]) -> Result<QValue> {
    let input = input.split(|&b| b == 0).next().unwrap_or_default();
    let mut s = Streamer::new();
    s.feed(input);
    s.flush();
    let mut value = None;
    while let Some(r) = s.next() {
        // parse_json() keeps the first error and complains about a second value.
        let v = r?;
        if value.is_some() {
            return Err(Error::generic("Expecting at most one JSON value"));
        }
        value = Some(v);
    }
    value.ok_or_else(|| Error::generic("Expecting a JSON value"))
}

/// `qobject_to_json_pretty()`.
pub fn to_string(value: &QValue, pretty: bool) -> String {
    let mut w = Writer { out: String::new(), pretty, stack: Vec::new(), need_comma: false };
    w.value(None, value);
    w.out
}

struct Writer {
    out: String,
    pretty: bool,
    /// One entry per open container: true for an array.
    stack: Vec<bool>,
    need_comma: bool,
}

impl Writer {
    fn newline(&mut self) {
        if self.pretty {
            self.out.push('\n');
            for _ in 0..self.stack.len() * 4 {
                self.out.push(' ');
            }
        }
    }

    fn comma_name(&mut self, name: Option<&str>) {
        if self.need_comma {
            self.out.push(',');
            if self.pretty {
                self.newline();
            } else {
                self.out.push(' ');
            }
        } else {
            if !self.out.is_empty() {
                self.newline();
            }
            self.need_comma = true;
        }
        if self.stack.last() == Some(&false) {
            quoted_str(&mut self.out, name.unwrap_or_default());
            self.out.push_str(": ");
        }
    }

    fn value(&mut self, name: Option<&str>, v: &QValue) {
        self.comma_name(name);
        match v {
            QValue::Null => self.out.push_str("null"),
            QValue::Bool(b) => self.out.push_str(if *b { "true" } else { "false" }),
            QValue::Int(i) => self.out.push_str(&i.to_string()),
            QValue::Uint(u) => self.out.push_str(&u.to_string()),
            QValue::Double(d) => self.out.push_str(&format_g17(*d)),
            QValue::Str(s) => quoted_str(&mut self.out, s),
            QValue::List(l) => {
                self.out.push('[');
                self.stack.push(true);
                self.need_comma = false;
                for item in l {
                    self.value(None, item);
                }
                self.stack.pop();
                self.need_comma = true;
                self.newline();
                self.out.push(']');
            }
            QValue::Dict(d) => {
                self.out.push('{');
                self.stack.push(false);
                self.need_comma = false;
                for (k, item) in d.iter() {
                    self.value(Some(k), item);
                }
                self.stack.pop();
                self.need_comma = true;
                self.newline();
                self.out.push('}');
            }
        }
    }
}

/// `quoted_str()` from qobject/json-writer.c: ASCII printable characters go out as they are,
/// everything else as `\uXXXX` with upper case hex.
pub fn quoted_str(out: &mut String, s: &str) {
    use std::fmt::Write;
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => {
                let mut cp = c as u32;
                if !is_valid_codepoint(cp as i32) {
                    cp = 0xFFFD;
                }
                if cp > 0xFFFF {
                    let _ = write!(
                        out,
                        "\\u{:04X}\\u{:04X}",
                        0xD800 + ((cp - 0x10000) >> 10),
                        0xDC00 + ((cp - 0x10000) & 0x3FF)
                    );
                } else if !(0x20..0x7F).contains(&cp) {
                    let _ = write!(out, "\\u{cp:04X}");
                } else {
                    out.push(c);
                }
            }
        }
    }
    out.push('"');
}

/// C's `printf("%.17g")`, which is how QEMU prints every double.
pub fn format_g17(v: f64) -> String {
    const P: i32 = 17;
    if v.is_nan() {
        return if v.is_sign_negative() { "-nan".into() } else { "nan".into() };
    }
    if v.is_infinite() {
        return if v < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let sci = format!("{:.*e}", (P - 1) as usize, v);
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    if (-4..P).contains(&exp) {
        let fixed = format!("{:.*}", (P - 1 - exp) as usize, v);
        strip_zeros(&fixed).to_string()
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{}{:02}", strip_zeros(mantissa), sign, exp.abs())
    }
}

fn strip_zeros(s: &str) -> &str {
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.') } else { s }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(input: &[u8]) -> Vec<std::result::Result<String, String>> {
        let mut s = Streamer::new();
        s.feed(input);
        let mut v = Vec::new();
        while let Some(r) = s.next() {
            v.push(r.map(|q| q.to_json()).map_err(|e| e.message().to_string()));
        }
        v
    }

    #[test]
    fn round_trips_in_qemu_format() {
        let v = from_str(
            r#"{"execute": "query-version", "arguments": {"a": [1, -2, 1.5, true, null, 'x']}}"#,
        )
        .unwrap();
        let d = v.as_dict().unwrap();
        assert_eq!(d.get_str("execute"), Some("query-version"));
        let again = from_str(&v.to_json()).unwrap();
        assert_eq!(v, again);
        assert_eq!(from_str("[]").unwrap().to_json(), "[]");
        assert_eq!(from_str("{}").unwrap().to_json(), "{}");
        assert_eq!(from_str("[1,2]").unwrap().to_json(), "[1, 2]");
    }

    #[test]
    fn greeting_prints_like_qemu() {
        let version = QDict::new().with("major", 11i64).with("minor", 1i64).with("micro", 0i64);
        let v = QDict::new().with("qemu", version).with("package", "");
        let qmp =
            QDict::new().with("version", v).with("capabilities", QValue::List(vec!["oob".into()]));
        let g = QValue::Dict(QDict::new().with("QMP", qmp));
        assert_eq!(
            g.to_json(),
            r#"{"QMP": {"version": {"qemu": {"micro": 0, "minor": 1, "major": 11}, "package": ""}, "capabilities": ["oob"]}}"#
        );
    }

    #[test]
    fn pretty_output() {
        let v = from_str(r#"{"a": [1, {}], "b": []}"#).unwrap();
        let p = v.to_json_pretty();
        assert_eq!(from_str(&p).unwrap(), v);
        assert!(p.starts_with("{\n    \""));
        assert!(p.contains("\n        1,\n        {\n        }"));
    }

    #[test]
    fn numbers() {
        assert_eq!(from_str("9223372036854775807").unwrap(), QValue::Int(i64::MAX));
        assert!(matches!(
            from_str("9223372036854775808").unwrap(),
            QValue::Uint(9223372036854775808)
        ));
        assert!(matches!(from_str("-9223372036854775809").unwrap(), QValue::Double(_)));
        assert!(matches!(from_str("18446744073709551616").unwrap(), QValue::Double(_)));
        assert_eq!(format_g17(0.0), "0");
        assert_eq!(format_g17(1.5), "1.5");
        assert_eq!(format_g17(0.1), "0.10000000000000001");
        assert_eq!(format_g17(1e20), "1e+20");
        assert_eq!(format_g17(-2.5e-7), "-2.4999999999999999e-07");
        assert_eq!(format_g17(123456.0), "123456");
        assert!(from_str("01").is_err());
        assert!(from_str("-").is_err());
        assert!(from_str("1.").is_err());
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(from_str(r#""é😀\/""#).unwrap(), QValue::str("é😀/"));
        assert_eq!(QValue::str("é😀\u{7f}\0\"").to_json(), r#""\u00E9\uD83D\uDE00\u007F\u0000\"""#);
        assert_eq!(from_bytes(b"\"\xC0\x80\"").unwrap(), QValue::str("\0"));
        assert_eq!(
            from_str(r#""\uD800""#).unwrap_err().message(),
            r#"1:1: JSON parse error, \uD800 is not a valid Unicode character"#
        );
        assert_eq!(
            from_str(r#""\q""#).unwrap_err().message(),
            "1:1: JSON parse error, invalid escape sequence in string"
        );
        assert_eq!(
            from_bytes(b"\"\xC1\x81\"").unwrap_err().message(),
            "1:1: JSON parse error, invalid UTF-8 sequence in string"
        );
    }

    #[test]
    fn qobject_from_json_errors() {
        assert_eq!(from_str("").unwrap_err().message(), "Expecting a JSON value");
        assert_eq!(from_str("1 2").unwrap_err().message(), "Expecting at most one JSON value");
        assert_eq!(
            from_str("[1").unwrap_err().message(),
            "1:4: JSON parse error, premature end of input"
        );
        assert_eq!(
            from_str("nul").unwrap_err().message(),
            "1:1: JSON parse error, invalid keyword 'nul'"
        );
        assert_eq!(
            from_str(r#"{"a":1,"a":2}"#).unwrap_err().message(),
            "1:12: JSON parse error, duplicate key"
        );
    }

    #[test]
    fn qmp_test_malformed_input_recovers() {
        // The cases from QEMU's tests/qtest/qmp-test.c, each followed by a good command.
        let good = br#"{"execute": "no-such-cmd"}"#;
        let cases: &[(&[u8], &str)] = &[
            (b"{]", "1:2: JSON parse error, expecting key"),
            (b"{\xFF", "1:2: JSON parse error, stray '\u{FFFD}'"),
            (b"{\x01", "1:2: JSON parse error, stray '\x01'"),
            (b"{'bad \xFF", "1:2: JSON parse error, stray ''bad \u{FFFD}'"),
            (b"{\"a\": \"\x01\"}", "1:7: JSON parse error, stray '\"\x01'"),
            (b"{]", "1:2: JSON parse error, expecting key"),
            (b"%p", "1:1: JSON parse error, stray '%'"),
        ];
        for (bad, msg) in cases {
            let mut input = bad.to_vec();
            input.extend_from_slice(good);
            let out = all(&input);
            assert_eq!(
                out.first(),
                Some(&Err(msg.to_string())),
                "input {:?}",
                String::from_utf8_lossy(bad)
            );
            assert_eq!(
                out.last(),
                Some(&Ok(r#"{"execute": "no-such-cmd"}"#.to_string())),
                "input {:?}",
                String::from_utf8_lossy(bad)
            );
        }
    }

    #[test]
    fn stream_splits_messages_and_tracks_lines() {
        let out = all(b"{\"a\": 1}\n[2]\n  {\"b\" 3}\n");
        assert_eq!(out[0], Ok(r#"{"a": 1}"#.into()));
        assert_eq!(out[1], Ok("[2]".into()));
        assert_eq!(out[2], Err("3:8: JSON parse error, expecting ':'".into()));
        let mut s = Streamer::new();
        s.feed(b"{\"exec");
        assert!(s.next().is_none());
        s.feed(b"ute\": 1}");
        assert!(s.next().unwrap().is_ok());
    }

    #[test]
    fn nesting_limit() {
        let deep = vec![b'['; MAX_NESTING + 1];
        let out = all(&deep);
        assert_eq!(out, vec![Err("JSON nesting depth limit exceeded".to_string())]);
    }
}
