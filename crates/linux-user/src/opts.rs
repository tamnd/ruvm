// SPDX-License-Identifier: GPL-2.0-or-later

//! The command line of `qemu-<target>`: `parse_args()` and `usage()` from
//! `linux-user/main.c`, with the same table, environment variables and messages.

use std::process::ExitCode;

/// The default `-s`, `TARGET_DEFAULT_STACK_SIZE`.
pub const DEFAULT_STACK_SIZE: u64 = 8 << 20;

/// What the command line asked for.
#[derive(Clone, Debug)]
pub struct Options {
    /// `-L`, `interp_prefix`.
    pub ld_prefix: String,
    /// `-s`, `guest_stack_size`.
    pub stack_size: u64,
    /// `-cpu`.
    pub cpu: Option<String>,
    /// `-0`.
    pub argv0: Option<String>,
    /// `-r`.
    pub uname_release: Option<String>,
    /// `-R`, 0 when not given.
    pub reserved_va: u64,
    /// `-one-insn-per-tb`.
    pub one_insn_per_tb: bool,
    /// `-tb-size` in MiB.
    pub tb_size: Option<u64>,
    /// The guest's environment, in the order `envlist_to_environ()` gives it.
    pub env: Vec<String>,
    /// The program, `exec_path`.
    pub exec_path: String,
    /// The program's arguments, its `argv[0]` first.
    pub args: Vec<String>,
}

/// One row of `arg_table`.
struct Arg {
    argv: &'static str,
    env: &'static str,
    has_arg: bool,
    example: &'static str,
    help: &'static str,
}

const fn a(
    argv: &'static str,
    env: &'static str,
    has_arg: bool,
    example: &'static str,
    help: &'static str,
) -> Arg {
    Arg { argv, env, has_arg, example, help }
}

const TABLE: &[Arg] = &[
    a("h", "", false, "", "print this help"),
    a("help", "", false, "", ""),
    a("g", "QEMU_GDB", true, "port", "wait gdb connection to 'port'"),
    a("L", "QEMU_LD_PREFIX", true, "path", "set the elf interpreter prefix to 'path'"),
    a("s", "QEMU_STACK_SIZE", true, "size", "set the stack size to 'size' bytes"),
    a("cpu", "QEMU_CPU", true, "model", "select CPU (-cpu help for list)"),
    a("E", "QEMU_SET_ENV", true, "var=value", "sets targets environment variable (see below)"),
    a("U", "QEMU_UNSET_ENV", true, "var", "unsets targets environment variable (see below)"),
    a("0", "QEMU_ARGV0", true, "argv0", "forces target process argv[0] to be 'argv0'"),
    a("r", "QEMU_UNAME", true, "uname", "set qemu uname release string to 'uname'"),
    a("B", "QEMU_GUEST_BASE", true, "address", "set guest_base address to 'address'"),
    a(
        "R",
        "QEMU_RESERVED_VA",
        true,
        "size",
        "reserve 'size' bytes for guest virtual address space",
    ),
    a(
        "t",
        "QEMU_RTSIG_MAP",
        true,
        "tsig hsig n[,...]",
        "map target rt signals [tsig,tsig+n) to [hsig,hsig+n]",
    ),
    a(
        "d",
        "QEMU_LOG",
        true,
        "item[,...]",
        "enable logging of specified items (use '-d help' for a list of items)",
    ),
    a("dfilter", "QEMU_DFILTER", true, "range[,...]", "filter logging based on address range"),
    a("D", "QEMU_LOG_FILENAME", true, "logfile", "write logs to 'logfile' (default stderr)"),
    a(
        "one-insn-per-tb",
        "QEMU_ONE_INSN_PER_TB",
        false,
        "",
        "run with one guest instruction per emulated TB",
    ),
    a("tb-size", "QEMU_TB_SIZE", true, "size", "TCG translation block cache size"),
    a("strace", "QEMU_STRACE", false, "", "log system calls"),
    a("seed", "QEMU_RAND_SEED", true, "", "Seed for pseudo-random number generator"),
    a("trace", "QEMU_TRACE", true, "", "[[enable=]<pattern>][,events=<file>][,file=<file>]"),
    a("plugin", "QEMU_PLUGIN", true, "", "[file=]<file>[,<argname>=<argvalue>]"),
    a("version", "QEMU_VERSION", false, "", "display version information and exit"),
    a("perfmap", "QEMU_PERFMAP", false, "", "Generate a /tmp/perf-${pid}.map file for perf"),
    a("jitdump", "QEMU_JITDUMP", false, "", "Generate a jit-${pid}.dump file for perf"),
];

/// How parsing ended when it did not produce [`Options`].
#[derive(Debug)]
pub enum Exit {
    /// Print this to stdout and exit with the code.
    Usage(String, ExitCode),
    /// Print this to stderr and exit with failure.
    Error(String),
}

/// `usage()`.
pub fn usage(target: &str, stack_size: u64) -> String {
    let mut maxarglen = "Argument".len();
    let mut maxenvlen = "Env-variable".len();
    for arg in TABLE {
        let mut arglen = arg.argv.len();
        if arg.has_arg {
            arglen += arg.example.len() + 1;
        }
        maxenvlen = maxenvlen.max(arg.env.len());
        maxarglen = maxarglen.max(arglen);
    }
    let mut s = format!(
        "usage: qemu-{target} [options] program [arguments...]\nLinux CPU emulator (compiled for {target} emulation)\n\nOptions and associated environment variables:\n\n"
    );
    s += &format!(
        "{:<w$} {:<e$} Description\n",
        "Argument",
        "Env-variable",
        w = maxarglen + 1,
        e = maxenvlen
    );
    for arg in TABLE {
        if arg.has_arg {
            let w = maxarglen - arg.argv.len() - 1;
            s += &format!(
                "-{} {:<w$} {:<e$} {}\n",
                arg.argv,
                arg.example,
                arg.env,
                arg.help,
                e = maxenvlen
            );
        } else {
            s += &format!(
                "-{:<w$} {:<e$} {}\n",
                arg.argv,
                arg.env,
                arg.help,
                w = maxarglen,
                e = maxenvlen
            );
        }
    }
    s += &format!(
        "\nDefaults:\nQEMU_LD_PREFIX  = /usr/gnemul/qemu-{target}\nQEMU_STACK_SIZE = {stack_size} byte\n"
    );
    s += "\nYou can use -E and -U options or the QEMU_SET_ENV and\n\
          QEMU_UNSET_ENV environment variables to set and unset\n\
          environment variables for the target process.\n\
          It is possible to provide several variables by separating them\n\
          by commas in getsubopt(3) style. Additionally it is possible to\n\
          provide the -E and -U options multiple times.\n\
          The following lines are equivalent:\n    \
          -E var1=val2 -E var2=val2 -U LD_PRELOAD -U LD_DEBUG\n    \
          -E var1=val2,var2=val2 -U LD_PRELOAD,LD_DEBUG\n    \
          QEMU_SET_ENV=var1=val2,var2=val2 QEMU_UNSET_ENV=LD_PRELOAD,LD_DEBUG\n\
          Note that if you provide several changes to a single variable\n\
          the last change will stay in effect.\n\
          \n\
          See <https://qemu.org/contribute/report-a-bug> for how to report bugs.\n\
          More information on the QEMU project at <https://qemu.org>.\n";
    s
}

/// `strtoul(s, &end, 0)`: the number at the start of `s` in C syntax and the rest.
fn strtoul(s: &str) -> (u64, &str) {
    let t = s.trim_start();
    let (radix, digits) = if let Some(r) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        (16, r)
    } else if t.starts_with('0') && t.len() > 1 {
        (8, &t[1..])
    } else {
        (10, t)
    };
    let end = digits.find(|c: char| !c.is_digit(radix)).unwrap_or(digits.len());
    if end == 0 {
        // "0x" alone is 0 followed by "x"; no digits is 0 with nothing consumed.
        return if radix == 8 { (0, &t[1..]) } else { (0, s) };
    }
    let v = u64::from_str_radix(&digits[..end], radix).unwrap_or(u64::MAX);
    (v, &digits[end..])
}

fn set_env(env: &mut Vec<String>, var: &str) {
    let name = var.split('=').next().unwrap_or(var);
    env.retain(|e| e.split('=').next() != Some(name));
    env.insert(0, var.to_string());
}

fn unset_env(env: &mut Vec<String>, name: &str) {
    env.retain(|e| e.split('=').next() != Some(name));
}

/// The state `parse_args()` fills in.
struct Parser<'a> {
    target: &'a str,
    o: Options,
}

impl Parser<'_> {
    fn usage(&self, code: ExitCode) -> Exit {
        Exit::Usage(usage(self.target, self.o.stack_size), code)
    }

    fn handle(&mut self, name: &str, arg: &str) -> Result<(), Exit> {
        let o = &mut self.o;
        match name {
            "h" | "help" => return Err(self.usage(ExitCode::SUCCESS)),
            "L" => o.ld_prefix = arg.to_string(),
            "s" => {
                let (v, rest) = strtoul(arg);
                if v == 0 {
                    return Err(self.usage(ExitCode::FAILURE));
                }
                o.stack_size = match rest.chars().next() {
                    Some('M') => v.saturating_mul(1 << 20),
                    Some('k' | 'K') => v.saturating_mul(1 << 10),
                    _ => v,
                };
            }
            "cpu" => o.cpu = Some(arg.to_string()),
            "E" => {
                for t in arg.split(',') {
                    set_env(&mut o.env, t);
                }
            }
            "U" => {
                for t in arg.split(',') {
                    unset_env(&mut o.env, t);
                }
            }
            "0" => o.argv0 = Some(arg.to_string()),
            "r" => o.uname_release = Some(arg.to_string()),
            "R" => {
                let (v, rest) = strtoul(arg);
                let shift = match rest.chars().next() {
                    Some('k' | 'K') => 10,
                    Some('M') => 20,
                    Some('G') => 30,
                    _ => 0,
                };
                let mut val = v;
                let mut rest = rest;
                if shift != 0 {
                    rest = &rest[1..];
                    val = v << shift;
                    if val >> shift != v {
                        return Err(Exit::Error("Reserved virtual address too big".into()));
                    }
                }
                if !rest.is_empty() {
                    return Err(Exit::Error(format!("Unrecognised -R size suffix '{rest}'")));
                }
                // reserved_va is the last address, so one less than the size.
                o.reserved_va = val;
            }
            "one-insn-per-tb" => o.one_insn_per_tb = true,
            "tb-size" => {
                let (v, _) = strtoul(arg);
                o.tb_size = Some(v);
            }
            // Logging, the random seed and perf maps change nothing a program sees.
            "d" | "dfilter" | "D" | "seed" | "perfmap" | "jitdump" | "B" => {}
            "g" | "t" | "strace" | "trace" | "plugin" => {
                return Err(Exit::Error(format!(
                    "qemu: option '-{name}' is not supported by ruvm yet"
                )));
            }
            _ => {}
        }
        Ok(())
    }
}

/// `parse_args()`: `env` is the host environment, `args` the arguments after the program
/// name.
pub fn parse(target: &str, env: Vec<(String, String)>, args: &[String]) -> Result<Options, Exit> {
    let mut p = Parser {
        target,
        o: Options {
            ld_prefix: format!("/usr/gnemul/qemu-{target}"),
            stack_size: DEFAULT_STACK_SIZE,
            cpu: None,
            argv0: None,
            uname_release: None,
            reserved_va: 0,
            one_insn_per_tb: false,
            tb_size: None,
            env: env.iter().map(|(k, v)| format!("{k}={v}")).collect(),
            exec_path: String::new(),
            args: Vec::new(),
        },
    };
    for arg in TABLE {
        if arg.env.is_empty() {
            continue;
        }
        if let Some((_, v)) = env.iter().find(|(k, _)| k == arg.env) {
            p.handle(arg.argv, v)?;
        }
    }
    let mut i = 0;
    while i < args.len() {
        let Some(r) = args[i].strip_prefix('-') else { break };
        i += 1;
        if r == "-" {
            break;
        }
        let r = r.strip_prefix('-').unwrap_or(r);
        let Some(arg) = TABLE.iter().find(|a| a.argv == r) else {
            return Err(Exit::Error(format!("qemu: unknown option '{r}'")));
        };
        if arg.has_arg {
            let Some(v) = args.get(i) else {
                return Err(Exit::Error(format!("qemu: missing argument for option '{r}'")));
            };
            i += 1;
            p.handle(arg.argv, v)?;
        } else {
            p.handle(arg.argv, "")?;
        }
    }
    let Some(exec) = args.get(i) else {
        return Err(Exit::Error("qemu: no user program specified".into()));
    };
    p.o.exec_path = exec.clone();
    p.o.args = args[i..].to_vec();
    if let Some(a0) = &p.o.argv0 {
        p.o.args[0] = a0.clone();
    }
    Ok(p.o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn options_and_program() {
        let o =
            parse("x86_64", vec![], &s(&["-L", "/sysroot", "--s", "1M", "/bin/ls", "-l"])).unwrap();
        assert_eq!(o.ld_prefix, "/sysroot");
        assert_eq!(o.stack_size, 1 << 20);
        assert_eq!(o.exec_path, "/bin/ls");
        assert_eq!(o.args, s(&["/bin/ls", "-l"]));
    }

    #[test]
    fn environment_edits() {
        let env = vec![("A".to_string(), "1".to_string()), ("B".to_string(), "2".to_string())];
        let o = parse("x86_64", env, &s(&["-E", "C=3,A=4", "-U", "B", "-0", "zero", "p"])).unwrap();
        assert_eq!(o.env, s(&["A=4", "C=3"]));
        assert_eq!(o.args, s(&["zero"]));
    }

    #[test]
    fn errors() {
        assert!(
            matches!(parse("x86_64", vec![], &s(&["-nope"])), Err(Exit::Error(e)) if e == "qemu: unknown option 'nope'")
        );
        assert!(
            matches!(parse("x86_64", vec![], &s(&["-L"])), Err(Exit::Error(e)) if e == "qemu: missing argument for option 'L'")
        );
        assert!(
            matches!(parse("x86_64", vec![], &s(&[])), Err(Exit::Error(e)) if e == "qemu: no user program specified")
        );
        assert!(
            matches!(parse("x86_64", vec![], &s(&["-R", "1X", "p"])), Err(Exit::Error(e)) if e == "Unrecognised -R size suffix 'X'")
        );
    }

    #[test]
    fn usage_columns() {
        let u = usage("x86_64", DEFAULT_STACK_SIZE);
        assert!(u.starts_with("usage: qemu-x86_64 [options] program [arguments...]\n"));
        assert!(u.contains("\nArgument             Env-variable         Description\n"));
        assert!(u.contains(
            "\n-L path              QEMU_LD_PREFIX       set the elf interpreter prefix to 'path'\n"
        ));
        assert!(u.contains("\n-strace              QEMU_STRACE          log system calls\n"));
        assert!(u.contains("QEMU_STACK_SIZE = 8388608 byte\n"));
    }
}
