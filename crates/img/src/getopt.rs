// SPDX-License-Identifier: GPL-2.0-or-later

//! `getopt()` and `getopt_long()` as the QEMU tools see them.
//!
//! Options and non-options are permuted the GNU way unless the option string starts with `+`
//! (stop at the first non-option) or `-` (non-options come back as option 1). Long options may
//! be abbreviated to any unambiguous prefix.
//!
//! The messages are those of the C library QEMU is built against: glibc on Linux, which names
//! the program by `argv[0]` (after the tools rewrote it to `qemu-img info` and the like), and the
//! BSD `getopt_long()` of macOS and the other hosts, which uses the program name.

/// Whether a long option takes an argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HasArg {
    No,
    Required,
    Optional,
}

/// One entry of the long option table: `struct option`.
#[derive(Clone, Copy, Debug)]
pub struct LongOpt {
    pub name: &'static str,
    pub has_arg: HasArg,
    /// What `getopt_long()` returns for it.
    pub val: i32,
}

/// Shorthand for a long option table entry.
pub const fn lopt(name: &'static str, has_arg: HasArg, val: i32) -> LongOpt {
    LongOpt { name, has_arg, val }
}

/// What one call returns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Opt {
    /// An option, by its value, with its argument.
    Opt(i32, Option<String>),
    /// A non-option argument, for `-` option strings (`getopt()` returns 1).
    NonOpt(String),
    /// An unknown option or a missing argument; the message has been printed. `getopt()`
    /// returns `'?'` (or `':'`).
    Err,
}

/// The state of a parse: `optind`, `optarg` and the permutation.
#[derive(Debug)]
pub struct Getopt<'a> {
    args: Vec<String>,
    shorts: &'a str,
    longs: &'a [LongOpt],
    /// `optind`: the next argument to look at.
    pub optind: usize,
    /// Position within a group of short options.
    nextchar: usize,
    first_nonopt: usize,
    last_nonopt: usize,
    /// `argv[0]` for the glibc messages.
    argv0: String,
    /// `opterr`: print messages.
    pub opterr: bool,
    /// `optopt`: the option character of the last error.
    pub optopt: i32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ordering {
    Permute,
    RequireOrder,
    ReturnInOrder,
}

fn msg(argv0: &str, text: &str) {
    if cfg!(target_os = "linux") {
        eprintln!("{argv0}: {text}");
    } else {
        eprintln!("{}: {text}", ruvm_base::report::program_name());
    }
}

impl<'a> Getopt<'a> {
    /// A parse of `args`, where `args[0]` is the program name, as `optind = 1` starts it.
    pub fn new(args: Vec<String>, shorts: &'a str, longs: &'a [LongOpt]) -> Self {
        let argv0 = args.first().cloned().unwrap_or_default();
        Getopt {
            args,
            shorts,
            longs,
            optind: 1,
            nextchar: 0,
            first_nonopt: 1,
            last_nonopt: 1,
            argv0,
            opterr: true,
            optopt: 0,
        }
    }

    /// The arguments, permuted so far: after the last option they are the operands.
    pub fn args(&self) -> &[String] {
        &self.args
    }

    /// `argv[optind..]`, the operands once parsing is over.
    pub fn rest(&self) -> &[String] {
        &self.args[self.optind.min(self.args.len())..]
    }

    fn ordering(&self) -> (Ordering, &'a str) {
        let s = self.shorts;
        if let Some(r) = s.strip_prefix('+') {
            (Ordering::RequireOrder, r)
        } else if let Some(r) = s.strip_prefix('-') {
            (Ordering::ReturnInOrder, r)
        } else if std::env::var_os("POSIXLY_CORRECT").is_some() {
            (Ordering::RequireOrder, s)
        } else {
            (Ordering::Permute, s)
        }
    }

    /// Moves the non-options skipped so far after the options that followed them.
    fn exchange(&mut self) {
        let a = self.first_nonopt;
        let b = self.last_nonopt;
        let c = self.optind;
        self.args[a..c].rotate_left(b - a);
        self.first_nonopt += c - b;
        self.last_nonopt = c;
    }

    fn is_nonopt(&self, i: usize) -> bool {
        let a = &self.args[i];
        !a.starts_with('-') || a.len() == 1
    }

    /// `getopt_long()`: the next option, or `None` at the end.
    #[allow(clippy::should_implement_trait, reason = "it is getopt_long()")]
    pub fn next(&mut self) -> Option<Opt> {
        let (ordering, shorts) = self.ordering();
        let (shorts, colon) = match shorts.strip_prefix(':') {
            Some(s) => (s, true),
            None => (shorts, false),
        };
        let argc = self.args.len();
        if self.nextchar == 0 {
            if self.last_nonopt > self.optind {
                self.last_nonopt = self.optind;
            }
            if self.first_nonopt > self.optind {
                self.first_nonopt = self.optind;
            }
            if ordering == Ordering::Permute {
                if self.first_nonopt != self.last_nonopt && self.last_nonopt != self.optind {
                    self.exchange();
                } else if self.last_nonopt != self.optind {
                    self.first_nonopt = self.optind;
                }
                while self.optind < argc && self.is_nonopt(self.optind) {
                    self.optind += 1;
                }
                self.last_nonopt = self.optind;
            }
            if self.optind != argc && self.args[self.optind] == "--" {
                self.optind += 1;
                if self.first_nonopt != self.last_nonopt && self.last_nonopt != self.optind {
                    self.exchange();
                } else if self.first_nonopt == self.last_nonopt {
                    self.first_nonopt = self.optind;
                }
                self.last_nonopt = argc;
                self.optind = argc;
            }
            if self.optind == argc {
                if self.first_nonopt != self.last_nonopt {
                    self.optind = self.first_nonopt;
                }
                return None;
            }
            if self.is_nonopt(self.optind) {
                if ordering == Ordering::RequireOrder {
                    return None;
                }
                let a = self.args[self.optind].clone();
                self.optind += 1;
                return Some(Opt::NonOpt(a));
            }
            let a = &self.args[self.optind];
            if !self.longs.is_empty() && a.starts_with("--") {
                return Some(self.long(colon));
            }
            self.nextchar = 1;
        }
        Some(self.short(shorts, colon))
    }

    fn long(&mut self, colon: bool) -> Opt {
        let arg = self.args[self.optind].clone();
        let body = &arg[2..];
        self.optind += 1;
        let (name, value) = match body.find('=') {
            Some(i) => (&body[..i], Some(body[i + 1..].to_string())),
            None => (body, None),
        };
        let mut found: Option<&LongOpt> = self.longs.iter().find(|o| o.name == name);
        let mut ambiguous = false;
        if found.is_none() {
            for o in self.longs.iter().filter(|o| o.name.starts_with(name)) {
                match found {
                    None => found = Some(o),
                    Some(f) if f.has_arg != o.has_arg || f.val != o.val => ambiguous = true,
                    Some(_) => {}
                }
            }
        }
        if ambiguous {
            if self.opterr {
                if cfg!(target_os = "linux") {
                    let mut t = format!("option '--{name}' is ambiguous; possibilities:");
                    for o in self.longs.iter().filter(|o| o.name.starts_with(name)) {
                        t.push_str(&format!(" '--{}'", o.name));
                    }
                    msg(&self.argv0, &t);
                } else {
                    msg(&self.argv0, &format!("option `--{name}' is ambiguous"));
                }
            }
            self.optopt = 0;
            return Opt::Err;
        }
        let Some(o) = found.copied() else {
            if self.opterr {
                if cfg!(target_os = "linux") {
                    msg(&self.argv0, &format!("unrecognized option '--{name}'"));
                } else {
                    msg(&self.argv0, &format!("unrecognized option `{arg}'"));
                }
            }
            self.optopt = 0;
            return Opt::Err;
        };
        match o.has_arg {
            HasArg::No => {
                if value.is_some() {
                    if self.opterr {
                        if cfg!(target_os = "linux") {
                            msg(
                                &self.argv0,
                                &format!("option '--{}' doesn't allow an argument", o.name),
                            );
                        } else {
                            msg(
                                &self.argv0,
                                &format!("option `--{name}' doesn't allow an argument"),
                            );
                        }
                    }
                    self.optopt = o.val;
                    return Opt::Err;
                }
                Opt::Opt(o.val, None)
            }
            HasArg::Optional => Opt::Opt(o.val, value),
            HasArg::Required => {
                if value.is_some() {
                    return Opt::Opt(o.val, value);
                }
                if self.optind < self.args.len() {
                    let v = self.args[self.optind].clone();
                    self.optind += 1;
                    return Opt::Opt(o.val, Some(v));
                }
                if self.opterr && !colon {
                    if cfg!(target_os = "linux") {
                        msg(&self.argv0, &format!("option '--{}' requires an argument", o.name));
                    } else {
                        msg(&self.argv0, &format!("option `--{name}' requires an argument"));
                    }
                }
                self.optopt = o.val;
                Opt::Err
            }
        }
    }

    fn short(&mut self, shorts: &str, colon: bool) -> Opt {
        let arg = self.args[self.optind].clone();
        let bytes = arg.as_bytes();
        let c = bytes[self.nextchar];
        self.nextchar += 1;
        let at_end = self.nextchar >= bytes.len();
        let pos = if c == b':' { None } else { shorts.bytes().position(|b| b == c) };
        let Some(pos) = pos else {
            if at_end {
                self.optind += 1;
                self.nextchar = 0;
            }
            if self.opterr && !colon {
                if cfg!(target_os = "linux") {
                    msg(&self.argv0, &format!("invalid option -- '{}'", c as char));
                } else {
                    msg(&self.argv0, &format!("invalid option -- {}", c as char));
                }
            }
            self.optopt = i32::from(c);
            return Opt::Err;
        };
        let spec = &shorts.as_bytes()[pos + 1..];
        let takes = spec.first() == Some(&b':');
        let optional = takes && spec.get(1) == Some(&b':');
        if !takes {
            if at_end {
                self.optind += 1;
                self.nextchar = 0;
            }
            return Opt::Opt(i32::from(c), None);
        }
        if !at_end {
            let v = arg[self.nextchar..].to_string();
            self.optind += 1;
            self.nextchar = 0;
            return Opt::Opt(i32::from(c), Some(v));
        }
        self.optind += 1;
        self.nextchar = 0;
        if optional {
            return Opt::Opt(i32::from(c), None);
        }
        if self.optind < self.args.len() {
            let v = self.args[self.optind].clone();
            self.optind += 1;
            return Opt::Opt(i32::from(c), Some(v));
        }
        if self.opterr && !colon {
            if cfg!(target_os = "linux") {
                msg(&self.argv0, &format!("option requires an argument -- '{}'", c as char));
            } else {
                msg(&self.argv0, &format!("option requires an argument -- {}", c as char));
            }
        }
        self.optopt = i32::from(c);
        if colon { Opt::Opt(i32::from(b':'), None) } else { Opt::Err }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn permutes_operands_to_the_end() {
        let longs = [lopt("format", HasArg::Required, i32::from(b'f'))];
        let mut g = Getopt::new(args(&["p", "a", "-f", "raw", "b", "-q"]), "f:q", &longs);
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'f'), Some("raw".into()))));
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'q'), None)));
        assert_eq!(g.next(), None);
        assert_eq!(g.rest(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn long_prefixes_and_values() {
        let longs = [
            lopt("output", HasArg::Required, 256),
            lopt("object", HasArg::Required, 257),
            lopt("force-share", HasArg::No, i32::from(b'U')),
        ];
        let mut g = Getopt::new(args(&["p", "--outp=json", "--force", "x"]), "U", &longs);
        g.opterr = false;
        assert_eq!(g.next(), Some(Opt::Opt(256, Some("json".into()))));
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'U'), None)));
        assert_eq!(g.next(), None);
        let mut g = Getopt::new(args(&["p", "--o", "x"]), "", &longs);
        g.opterr = false;
        assert_eq!(g.next(), Some(Opt::Err));
    }

    #[test]
    fn require_order_and_in_order() {
        let mut g = Getopt::new(args(&["p", "-h", "info", "-f"]), "+hf:", &[]);
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'h'), None)));
        assert_eq!(g.next(), None);
        assert_eq!(g.optind, 2);
        let mut g = Getopt::new(args(&["p", "img", "-1G"]), "-hf:q", &[]);
        g.opterr = false;
        assert_eq!(g.next(), Some(Opt::NonOpt("img".into())));
        assert_eq!(g.next(), Some(Opt::Err));
        assert_eq!(g.optopt, i32::from(b'1'));
    }

    #[test]
    fn grouped_shorts_and_attached_argument() {
        let mut g = Getopt::new(args(&["p", "-qUfraw", "--", "-x"]), "qUf:", &[]);
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'q'), None)));
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'U'), None)));
        assert_eq!(g.next(), Some(Opt::Opt(i32::from(b'f'), Some("raw".into()))));
        assert_eq!(g.next(), None);
        assert_eq!(g.rest(), &["-x".to_string()]);
    }
}
