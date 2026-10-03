// SPDX-License-Identifier: GPL-2.0-or-later

//! `-plugin file=...,arg=...`: `qemu_plugin_opt_parse()` and `plugin_add()` from
//! `plugins/loader.c`.

#![forbid(unsafe_code)]

use ruvm_base::Error;
use ruvm_base::report::warn_report;
use ruvm_qapi::cutils::bool_parse;
use ruvm_qapi::opts::{ParseFailure, QemuOptsList, is_help_option};

/// What `-plugin help` prints before QEMU exits with status 0.
pub const PLUGIN_HELP: &str =
    "Plugin options\n  file=<path/to/plugin.so>\n  plugin specific arguments\n";

/// A plugin to load, `struct qemu_plugin_desc`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PluginDesc {
    /// The shared object.
    pub path: String,
    /// The arguments passed to `qemu_plugin_install()`, each `name=value`.
    pub argv: Vec<String>,
}

/// Why [`qemu_plugin_opt_parse`] stopped.
#[derive(Debug)]
pub enum OptError {
    /// A value was `help` or `?`. QEMU prints [`PLUGIN_HELP`] to stdout and exits with 0.
    Help,
    /// The option string is bad. QEMU reports the error and exits with 1.
    Invalid(Error),
}

/// `qemu_plugin_opt_parse()`: add the plugins and arguments of one `-plugin` option to
/// `head`. A `file=` naming a plugin already in the list adds the arguments that follow to it.
///
/// The deprecation warning for `arg=` is printed here, as QEMU does.
pub fn qemu_plugin_opt_parse(optstr: &str, head: &mut Vec<PluginDesc>) -> Result<(), OptError> {
    let mut list = QemuOptsList::new("plugin", &[]).with_implied_opt_name("file");
    let mut warnings = Vec::new();
    let r = list.parse_detailed(optstr, true, &mut warnings);
    for w in &warnings {
        w.report();
    }
    let opts = match r {
        Ok(o) => o,
        Err(ParseFailure::HelpWanted) => return Err(OptError::Help),
        Err(ParseFailure::Error(e)) => return Err(OptError::Invalid(e)),
    };
    let mut curr: Option<usize> = None;
    for (name, value) in opts.iter() {
        if is_help_option(value) {
            return Err(OptError::Help);
        } else if name == "file" {
            if value.is_empty() {
                return Err(OptError::Invalid(Error::generic("requires a non-empty argument")));
            }
            let i = match head.iter().position(|d| d.path == value) {
                Some(i) => i,
                None => {
                    head.push(PluginDesc { path: value.to_string(), argv: Vec::new() });
                    head.len() - 1
                }
            };
            curr = Some(i);
        } else {
            let Some(i) = curr else {
                return Err(OptError::Invalid(Error::generic(
                    "missing earlier '-plugin file=' option",
                )));
            };
            let fullarg = if name == "arg" && bool_parse(value).is_none() {
                // Will treat arg="argname" as "argname=on".
                let fullarg =
                    if value.contains('=') { value.to_string() } else { format!("{value}=on") };
                warn_report(&format!("using 'arg={value}' is deprecated"));
                eprintln!("Please use '{fullarg}' directly");
                fullarg
            } else {
                format!("{name}={value}")
            };
            head[i].argv.push(fullarg);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(list: &[&str]) -> Result<Vec<PluginDesc>, OptError> {
        let mut head = Vec::new();
        for s in list {
            qemu_plugin_opt_parse(s, &mut head)?;
        }
        Ok(head)
    }

    fn msg(r: Result<Vec<PluginDesc>, OptError>) -> String {
        match r {
            Err(OptError::Invalid(e)) => e.message().to_string(),
            other => panic!("expected an error, got {other:?}"),
        }
    }

    #[test]
    fn files_and_arguments() {
        let head = parse(&["./a.so,inline=on,x=1", "file=./b.so", "./a.so,y=2"]).unwrap();
        assert_eq!(head.len(), 2);
        assert_eq!(head[0].path, "./a.so");
        assert_eq!(head[0].argv, ["inline=on", "x=1", "y=2"]);
        assert_eq!(head[1].path, "./b.so");
        assert!(head[1].argv.is_empty());
    }

    #[test]
    fn deprecated_arg() {
        let head = parse(&["./a.so,arg=inline,arg=x=3,arg=on"]).unwrap();
        assert_eq!(head[0].argv, ["inline=on", "x=3", "arg=on"]);
    }

    #[test]
    fn errors_and_help() {
        assert_eq!(msg(parse(&["file="])), "requires a non-empty argument");
        assert_eq!(msg(parse(&["file=./a.so,file=,x=1"])), "requires a non-empty argument");
        assert_eq!(msg(parse(&["x=1"])), "missing earlier '-plugin file=' option");
        assert!(matches!(parse(&["help"]), Err(OptError::Help)));
        assert!(matches!(parse(&["./a.so,inline=help"]), Err(OptError::Help)));
        assert!(PLUGIN_HELP.starts_with("Plugin options\n"));
    }
}
