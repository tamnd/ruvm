// SPDX-License-Identifier: GPL-2.0-or-later

//! A generator that turns QEMU decodetree files into Rust decoders.
//!
//! This is a Rust reimplementation of QEMU 11.1's scripts/decodetree.py, meant to run from a
//! target crate's build.rs. It reads the same `.decode` files, unmodified, parses the whole
//! language described in docs/devel/decodetree.rst (fields, including signed, multi-part,
//! `!function=` and parameter fields, argument sets including `!extern` and typed members,
//! formats, patterns, overlapping `{ }` and non-overlapping `[ ]` groups), accepts the same
//! options (`--insnwidth`, `--varinsnwidth`, `--decode`, `--static-decode`, `--translate`), and
//! fails on bad input with the same messages on the same lines. The decision tree is built the
//! way the script builds it, so the generated decoder tests bits in the same order and resolves
//! overlaps the same way as QEMU's.
//!
//! ```
//! let src = "&r rd rs\nadd 0000 rd:4 rs:4 0000 0000 0000 0000 0000\n";
//! let rust = ruvm_decode::generate(src, &ruvm_decode::Options::default()).unwrap();
//! assert!(rust.contains("fn trans_add(&mut self, a: &mut arg_add) -> bool;"));
//! ```
//!
//! # What the output looks like
//!
//! For an input the script would turn into C, the output has the same items in Rust.
//!
//! - One `pub struct arg_<set>` per argument set that is not `!extern`, with `pub` members. The
//!   C member types map to Rust ones: `int` is `i32`, `bool` is `bool`, the `<stdint.h>` types
//!   are the matching Rust integers, and any other name is used as the Rust type as written. The
//!   structs derive `Clone, Copy, Debug, Default, PartialEq, Eq`. An `!extern` set is expected to
//!   be in scope already, usually with a `use` of the module generated from the file that owns
//!   it, which is how QEMU shares sets between a32.decode and t32.decode.
//! - A `pub type arg_<pattern> = arg_<set>;` alias per pattern name, like the C typedefs, except
//!   when the pattern and its set have the same name.
//! - A trait, named by [`Options::trait_name`] or else the decode function name in CamelCase
//!   (`decode_insn32` becomes `DecodeInsn32`), with one `fn trans_<name>(&mut self, a: &mut
//!   arg_<name>) -> bool` per pattern name, one method per `!function=` name, and for
//!   `--varinsnwidth` the `<decode>_load_bytes` method. The type that implements it plays the
//!   part of the C `DisasContext`. Because every method is required, a pattern without an
//!   implementation, or an implementation for a pattern that is not in the file, is a compile
//!   error rather than an undefined instruction at run time.
//! - `<DECODE>_PATTERNS`, a table of `(fixed mask, fixed bits, name)` for every pattern, which
//!   a fuzzer can use to aim at each encoding.
//! - One private extract function per format, and the decode function, `fn <decode><T>(ctx: &mut
//!   T, insn: u32) -> bool`, `pub` for `--decode` and private for `--static-decode` or neither,
//!   with `u16` or `u64` for the other widths. With `--varinsnwidth` there is also
//!   `<decode>_load`, which reads the instruction bytes through `<decode>_load_bytes`.
//!
//! The text is meant for `include!`, so every item carries its own `#[allow]` for the lints the
//! QEMU names trip, and the whole file can go into a module of a crate built with `-D warnings`.
//!
//! # Differences from decodetree.py
//!
//! - The output is Rust rather than C, as above. `switch` becomes `match` on the same masked
//!   value, `if ((insn & m) == b)` becomes the same `if`, and the C union of argument structs
//!   becomes one local per argument set, declared where the C code first extracts into it.
//!   Values are converted on assignment the way C converts them: a field that is `bool` gets
//!   `value != 0`, and integers are truncated or sign extended with `as`.
//! - A `!function=` function is a trait method. It takes and returns `i32` when the bit
//!   operations are 32 bits wide (16 and 32 bit instructions) and `i64` for 64 bit
//!   instructions. In C its signature is whatever the including file declares, almost always
//!   `int`. A parameter field (`!function=` with no bits) is a method with no argument.
//! - Within one topological level, field assignments are written in declaration order. The
//!   script uses a Python set there, so its order varies between runs; the levels themselves,
//!   which are what make named field references work, are the same.
//! - QEMU's `extract32` family asserts on a zero length field, which the script lets through.
//!   The generated code reads such a field as zero.
//! - Where the script would crash with a Python exception rather than print a message, this
//!   crate reports an error or carries on sensibly: comparing a multi-part or named field with
//!   a field of another class while looking for a format to share counts as not equal, an empty
//!   `[ ]` group inside another group bins as if its fixed bits were zero, `-v` is rejected with
//!   "unhandled option -v", and a width that is not a number is reported as one that cannot be
//!   handled. A function used both with and without bits is an error here, since it would be two
//!   trait methods with one name; in C it fails to compile.
//! - The script writes to a file, stdout, or nowhere, and exits; [`generate`] returns the text
//!   or an [`Error`] whose [`Error::render`] is the line the script prints.
//! - The script keeps the C types from `--insnwidth` when a later option sets the width back to
//!   32. [`Options`] has one width, and the last option wins.

#![forbid(unsafe_code)]

mod cli;
mod emit;
mod model;
mod parse;
mod tree;

use std::fmt;

pub use cli::Invocation;

use crate::model::{Group, Model, Node};
use crate::parse::{Ctx, Parser};

/// The settings for one run, the counterparts of decodetree.py's command line options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// Instruction width in bits: 16, 32 or 64 (`--insnwidth`, `-w`).
    pub insn_width: u32,
    /// Patterns may be shorter than `insn_width` by a whole number of bytes, and a `_load`
    /// function is generated to read them (`--varinsnwidth`).
    pub var_insn_width: bool,
    /// Name of the decode function, also the prefix of every generated helper (`--decode`,
    /// `--static-decode`).
    pub decode_function: String,
    /// Whether the decode function is `pub` (`--decode`) or private (the default, and
    /// `--static-decode`).
    pub public_decode: bool,
    /// Prefix of the translate methods, `trans` unless `--translate` says otherwise.
    pub translate_prefix: String,
    /// Name of the generated trait. `None` means the decode function name in CamelCase.
    pub trait_name: Option<String>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            insn_width: 32,
            var_insn_width: false,
            decode_function: "decode".to_string(),
            public_decode: false,
            translate_prefix: "trans".to_string(),
            trait_name: None,
        }
    }
}

impl Options {
    /// The trait name that will be used, explicit or derived from the decode function.
    pub fn effective_trait_name(&self) -> String {
        match &self.trait_name {
            Some(t) => t.clone(),
            None => camel_case(&self.decode_function),
        }
    }
}

fn camel_case(s: &str) -> String {
    let mut out = String::new();
    for part in s.split('_').filter(|p| !p.is_empty()) {
        let mut c = part.chars();
        if let Some(first) = c.next() {
            out.extend(first.to_uppercase());
            out.push_str(c.as_str());
        }
    }
    if out.is_empty() { "Decode".to_string() } else { out }
}

/// A failure, located the way decodetree.py locates it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// The file the script would name, empty when it names none.
    pub file: String,
    /// The line, 0 when the script names none.
    pub line: usize,
    /// The message, which for overlapping patterns spans several lines.
    pub message: String,
}

impl Error {
    /// The text decodetree.py prints to stderr, without the final newline. With
    /// `test_for_error` (the script's `--test-for-error`) the word `error:` becomes `detected:`.
    pub fn render(&self, test_for_error: bool) -> String {
        let mut prefix = String::new();
        if !self.file.is_empty() {
            prefix += &format!("{}:", self.file);
        }
        if self.line != 0 {
            prefix += &format!("{}:", self.line);
        }
        if !prefix.is_empty() {
            prefix.push(' ');
        }
        let word = if test_for_error { "detected: " } else { "error: " };
        format!("{prefix}{word}{}", self.message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render(false))
    }
}

impl std::error::Error for Error {}

/// Generate the Rust decoder for one decodetree file. Messages name the file `input.decode`;
/// use [`generate_files`] to give it a name.
pub fn generate(input: &str, opts: &Options) -> Result<String, Error> {
    generate_files(&[("input.decode", input)], opts)
}

/// Generate one Rust decoder from several decodetree files, given as `(name, text)` pairs, as
/// the script does when it is given several inputs.
pub fn generate_files(inputs: &[(&str, &str)], opts: &Options) -> Result<String, Error> {
    let (m, ctx, stree) = run(inputs, opts)?;
    let names = inputs.iter().map(|(n, _)| (*n).to_string()).collect();
    emit::Emitter::new(
        &m,
        &ctx,
        opts.effective_trait_name(),
        opts.translate_prefix.clone(),
        opts.public_decode,
        names,
    )
    .emit(stree.as_ref())
}

/// What a decodetree description contains, for tools such as fuzzers and test harnesses that
/// need to know the generated names and shapes without parsing the Rust.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// Every argument set, including inferred ones, in declaration order.
    pub arg_sets: Vec<ArgSetInfo>,
    /// Every pattern in declaration order. A name can appear more than once.
    pub patterns: Vec<PatternInfo>,
    /// Every `!function=` name the generated code calls.
    pub functions: Vec<FunctionInfo>,
}

/// One argument set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgSetInfo {
    /// The name without the `&`; the struct is `arg_<name>`.
    pub name: String,
    /// The members in declaration order.
    pub fields: Vec<ArgFieldInfo>,
    /// Marked `!extern`, so the struct is not generated.
    pub is_extern: bool,
}

/// One member of an argument set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgFieldInfo {
    /// The member name as written; see [`rust_ident`] for the Rust spelling.
    pub name: String,
    /// The C type as written, `int` when none is given.
    pub c_type: String,
    /// The Rust type used in the struct.
    pub rust_type: String,
}

/// One pattern.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatternInfo {
    /// The pattern name; the translate method is `trans_<name>`.
    pub name: String,
    /// The argument set its translate method receives.
    pub arg_set: String,
    /// The bits the pattern fixes, including those from its format.
    pub fixed_mask: u64,
    /// The values of those bits.
    pub fixed_bits: u64,
    /// The pattern width in bits.
    pub width: u32,
    /// Where it was defined.
    pub file: String,
    /// The line it starts on.
    pub line: usize,
}

/// One `!function=` name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FunctionInfo {
    /// The function name, which is also the trait method name.
    pub name: String,
    /// False for a parameter field, which has no bits and calls the function with no value.
    pub takes_value: bool,
}

/// Parse and check the inputs as [`generate_files`] does, and describe what they contain.
pub fn inspect(inputs: &[(&str, &str)], opts: &Options) -> Result<Info, Error> {
    let (m, _, _) = run(inputs, opts)?;
    let arg_sets = m
        .arguments
        .iter()
        .map(|a| ArgSetInfo {
            name: a.name.clone(),
            fields: a
                .fields
                .iter()
                .zip(&a.types)
                .map(|(n, t)| ArgFieldInfo {
                    name: n.clone(),
                    c_type: t.clone(),
                    rust_type: emit::rust_type(t).to_string(),
                })
                .collect(),
            is_extern: a.is_extern,
        })
        .collect();
    let patterns = m
        .allpatterns
        .iter()
        .map(|&p| {
            let pt = &m.patterns[p];
            PatternInfo {
                name: pt.name.clone(),
                arg_set: m.arguments[m.pattern_args(p)].name.clone(),
                fixed_mask: pt.fixedmask,
                fixed_bits: pt.fixedbits,
                width: pt.width,
                file: pt.file.clone(),
                line: pt.lineno,
            }
        })
        .collect();
    let functions = emit::functions(&m)?
        .into_iter()
        .map(|(name, takes_value)| FunctionInfo { name, takes_value })
        .collect();
    Ok(Info { arg_sets, patterns, functions })
}

/// The Rust spelling of a decodetree name: a raw identifier for a Rust keyword, and a trailing
/// underscore for the few keywords that cannot be raw.
pub fn rust_ident(name: &str) -> String {
    const STRICT: &[&str] = &[
        "as", "break", "const", "continue", "else", "enum", "extern", "false", "fn", "for", "if",
        "impl", "in", "let", "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
        "static", "struct", "trait", "true", "type", "unsafe", "use", "where", "while", "async",
        "await", "dyn", "abstract", "become", "box", "do", "final", "macro", "override", "priv",
        "typeof", "unsized", "virtual", "yield", "try", "gen",
    ];
    if matches!(name, "self" | "Self" | "super" | "crate" | "_") {
        format!("{name}_")
    } else if STRICT.contains(&name) {
        format!("r#{name}")
    } else {
        name.to_string()
    }
}

/// The Rust type the generated structs use for a C type from an argument set.
pub fn rust_type(c_type: &str) -> &str {
    emit::rust_type(c_type)
}

/// Parse, check and build the trees, as `main` in the script does before it writes anything.
fn run(
    inputs: &[(&str, &str)],
    opts: &Options,
) -> Result<(Model, Ctx, Option<tree::SizeTree>), Error> {
    let insnmask = match opts.insn_width {
        16 => 0xffff,
        32 => 0xffff_ffff,
        64 => u64::MAX,
        w => {
            return Err(Error {
                file: String::new(),
                line: 0,
                message: format!("cannot handle insns of width {w}"),
            });
        }
    };
    if inputs.is_empty() {
        return Err(Error { file: String::new(), line: 0, message: "missing input file".into() });
    }
    let mut ctx = Ctx {
        insnwidth: opts.insn_width,
        insnmask,
        variablewidth: opts.var_insn_width,
        decode_function: opts.decode_function.clone(),
        input_file: String::new(),
    };
    let mut m = Model::default();
    m.groups.push(Group {
        overlapping: false,
        file: String::new(),
        lineno: 0,
        pats: Vec::new(),
        fixedbits: 0,
        fixedmask: 0,
        undefmask: 0,
        width: None,
        tree: None,
    });

    for (name, text) in inputs {
        ctx.input_file = (*name).to_string();
        Parser { ctx: &mut ctx, m: &mut m }.parse_file(text, 0)?;
    }

    // The top level keeps a zero mask: decoding starts from nothing.
    let top: Vec<Node> = m.groups[0].pats.clone();
    for &p in &top {
        if let Node::Group(g) = p {
            tree::prop_masks(&mut m, &ctx, g);
        }
    }
    tree::build_tree(&mut m, &ctx, 0)?;
    tree::prop_format(&mut m, 0);

    let stree = if opts.var_insn_width {
        for &p in &top {
            if let Node::Group(g) = p {
                tree::prop_width(&mut m, g)?;
            }
        }
        let mut s = tree::build_size_tree(&m, &ctx, &top, 8, 0, 0)?;
        tree::prop_size(&mut s);
        Some(s)
    } else {
        None
    };
    Ok((m, ctx, stree))
}

#[cfg(test)]
mod tests {
    use super::{Error, Options, camel_case, generate, rust_ident};

    #[test]
    fn trait_names() {
        assert_eq!(camel_case("decode"), "Decode");
        assert_eq!(camel_case("disas_a64"), "DisasA64");
        assert_eq!(camel_case("decode_XVentanaCodeOps"), "DecodeXVentanaCodeOps");
    }

    #[test]
    fn keywords_are_escaped() {
        assert_eq!(rust_ident("type"), "r#type");
        assert_eq!(rust_ident("self"), "self_");
        assert_eq!(rust_ident("rd"), "rd");
    }

    #[test]
    fn errors_render_like_the_script() {
        let e = Error { file: "a.decode".into(), line: 3, message: "bad".into() };
        assert_eq!(e.render(false), "a.decode:3: error: bad");
        assert_eq!(e.render(true), "a.decode:3: detected: bad");
        let e = Error { file: String::new(), line: 0, message: "missing input file".into() };
        assert_eq!(e.to_string(), "error: missing input file");
    }

    #[test]
    fn bad_width() {
        let opts = Options { insn_width: 8, ..Options::default() };
        let e = generate("", &opts).unwrap_err();
        assert_eq!(e.to_string(), "error: cannot handle insns of width 8");
    }
}
