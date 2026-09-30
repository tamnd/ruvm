// SPDX-License-Identifier: GPL-2.0-or-later

//! The decodetree.py command line, parsed the way Python's `getopt.gnu_getopt` parses it for
//! the script: options and file names in any order, `--` to end options, long options
//! abbreviated to any unique prefix, and the same messages for mistakes.

use crate::{Error, Options};

/// A parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Invocation {
    /// The generator settings.
    pub options: Options,
    /// `-o`/`--output`: where to write, stdout when `None`.
    pub output: Option<String>,
    /// `--output-null`: generate and discard.
    pub output_null: bool,
    /// `--test-for-error`: failure is the expected outcome, see [`Error::render`].
    pub test_for_error: bool,
    /// The input files.
    pub files: Vec<String>,
}

const LONG: &[&str] = &[
    "decode=",
    "translate=",
    "output=",
    "insnwidth=",
    "static-decode=",
    "varinsnwidth=",
    "test-for-error",
    "output-null",
];

fn fail(message: String) -> Error {
    Error { file: String::new(), line: 0, message }
}

impl Invocation {
    /// Parse the arguments after the program name.
    pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<Invocation, Error> {
        let mut opts: Vec<(String, String)> = Vec::new();
        let mut files = Vec::new();
        let mut i = 0;
        while i < args.len() {
            let a = args[i].as_ref();
            i += 1;
            if a == "--" {
                files.extend(args[i..].iter().map(|s| s.as_ref().to_string()));
                break;
            }
            if let Some(long) = a.strip_prefix("--") {
                let (opt, value) = match long.split_once('=') {
                    Some((o, v)) => (o, Some(v.to_string())),
                    None => (long, None),
                };
                let possible: Vec<&str> =
                    LONG.iter().copied().filter(|o| o.starts_with(opt)).collect();
                if possible.is_empty() {
                    return Err(fail(format!("option --{opt} not recognized")));
                }
                let (has_arg, name) = if possible.contains(&opt) {
                    (false, opt.to_string())
                } else if possible.iter().any(|p| p.strip_suffix('=') == Some(opt)) {
                    (true, opt.to_string())
                } else if possible.len() > 1 {
                    return Err(fail(format!("option --{opt} not a unique prefix")));
                } else {
                    let p = possible[0];
                    match p.strip_suffix('=') {
                        Some(n) => (true, n.to_string()),
                        None => (false, p.to_string()),
                    }
                };
                let value = if has_arg {
                    match value {
                        Some(v) => v,
                        None => {
                            if i >= args.len() {
                                return Err(fail(format!("option --{name} requires argument")));
                            }
                            i += 1;
                            args[i - 1].as_ref().to_string()
                        }
                    }
                } else {
                    if value.is_some() {
                        return Err(fail(format!("option --{name} must not have an argument")));
                    }
                    String::new()
                };
                opts.push((format!("--{name}"), value));
            } else if a.len() > 1 && a.starts_with('-') {
                let mut rest = &a[1..];
                while let Some(c) = rest.chars().next() {
                    rest = &rest[c.len_utf8()..];
                    match c {
                        'o' | 'w' => {
                            let value = if !rest.is_empty() {
                                std::mem::take(&mut rest).to_string()
                            } else if i < args.len() {
                                i += 1;
                                args[i - 1].as_ref().to_string()
                            } else {
                                return Err(fail(format!("option -{c} requires argument")));
                            };
                            opts.push((format!("-{c}"), value));
                        }
                        'v' => opts.push(("-v".into(), String::new())),
                        _ => return Err(fail(format!("option -{c} not recognized"))),
                    }
                }
            } else {
                files.push(a.to_string());
            }
        }

        let mut inv = Invocation {
            options: Options::default(),
            output: None,
            output_null: false,
            test_for_error: false,
            files,
        };
        for (o, a) in opts {
            match o.as_str() {
                "-o" | "--output" => inv.output = Some(a),
                "--decode" => {
                    inv.options.decode_function = a;
                    inv.options.public_decode = true;
                }
                "--static-decode" => inv.options.decode_function = a,
                "--translate" => inv.options.translate_prefix = a,
                "-w" | "--insnwidth" | "--varinsnwidth" => {
                    if o == "--varinsnwidth" {
                        inv.options.var_insn_width = true;
                    }
                    let w = a.trim().parse::<u32>().ok().filter(|w| matches!(w, 16 | 32 | 64));
                    match w {
                        Some(w) => inv.options.insn_width = w,
                        None => return Err(fail(format!("cannot handle insns of width {a}"))),
                    }
                }
                "--test-for-error" => inv.test_for_error = true,
                "--output-null" => inv.output_null = true,
                _ => return Err(fail(format!("unhandled option {o}"))),
            }
        }
        if inv.files.is_empty() {
            return Err(fail("missing input file".into()));
        }
        Ok(inv)
    }
}

#[cfg(test)]
mod tests {
    use super::Invocation;

    fn msg(args: &[&str]) -> String {
        Invocation::parse(args).unwrap_err().to_string()
    }

    #[test]
    fn options_like_meson_passes_them() {
        let inv =
            Invocation::parse(&["t16.decode", "-w", "16", "--static-decode=disas_t16"]).unwrap();
        assert_eq!(inv.options.insn_width, 16);
        assert_eq!(inv.options.decode_function, "disas_t16");
        assert!(!inv.options.public_decode);
        assert_eq!(inv.files, ["t16.decode"]);

        let inv =
            Invocation::parse(&["--decode", "disas_sve", "-o", "out.rs", "sve.decode"]).unwrap();
        assert!(inv.options.public_decode);
        assert_eq!(inv.output.as_deref(), Some("out.rs"));

        let inv = Invocation::parse(&["--varinsn=32", "--output-n", "x.decode"]).unwrap();
        assert!(inv.options.var_insn_width && inv.output_null);
    }

    #[test]
    fn getopt_messages() {
        assert_eq!(msg(&["--bogus", "x"]), "error: option --bogus not recognized");
        assert_eq!(msg(&["--output"]), "error: option --output requires argument");
        assert_eq!(
            msg(&["--output-null=1", "x"]),
            "error: option --output-null must not have an argument"
        );
        assert_eq!(msg(&["--output", "x"]), "error: missing input file");
        assert_eq!(msg(&["--o", "x"]), "error: option --o not a unique prefix");
        assert_eq!(msg(&["-x"]), "error: option -x not recognized");
        assert_eq!(msg(&["-w"]), "error: option -w requires argument");
        assert_eq!(msg(&["-w", "8", "x"]), "error: cannot handle insns of width 8");
        assert_eq!(msg(&[]), "error: missing input file");
    }
}
