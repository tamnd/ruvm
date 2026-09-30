// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-img, the disk image utility: qemu-img.c.
//!
//! [`main`] takes the arguments after `argv[0]` and returns the exit status. Every command
//! and option of QEMU 11.1 is there, with the same help texts, messages and exit statuses.
//! The commands open their images through [`ruvm_block::tools`], so what a command can do
//! with an image depends on which format drivers the block layer has.
//!
//! Differences from QEMU:
//!
//! - `-T`/`--trace` is accepted and ignored; there are no trace events.
//! - `--object` can only make the object types the lower layers register, which today is
//!   `secret`. `--object help` lists those.
//! - There are no block jobs. `commit` copies the data itself the way the active commit
//!   (mirror) job would, one request at a time: it marks what the images above the base
//!   allocate in chunks of the bitmap granularity, then copies, zeroes or discards each
//!   chunk as `mirror_iteration()` decides. The job's zero bitmap is left out, so a range
//!   the base already reads as zero gets its zeroes written again. `-r` and `-p` work, but
//!   `-p` prints its progress lines as the copy goes rather than once per job loop
//!   iteration, so the number of lines differs from QEMU's.
//! - `convert` has no coroutines and copies with a single loop, one request at a time; `-m`
//!   and `-W` are checked and otherwise have no effect, and `-C` never offloads the copy.
//! - `bench` issues its requests one after the other. The requests QEMU would keep in
//!   flight complete first in, first out, so the offsets and the flushes are the ones QEMU
//!   sends, but `-d` does not make them run in parallel. `-i` takes `threads` and `native`
//!   (the same as `-n`); `io_uring` is refused as in a QEMU build without liburing.
//! - `bitmap` makes the open images give up their permissions before the nodes are
//!   inactivated, which is what `blk_root_inactivate()` does in QEMU.
//! - Images are not opened with `BDRV_O_CHECK`, and there is no SIGUSR1 handler to print
//!   the progress on demand.
//! - Errors carry messages rather than errno values, so where QEMU tests for a particular
//!   errno (`-ENOTSUP` of `blk_make_empty()`, `-ENOSPC` when changing the backing file) the
//!   message is looked at instead.
//! - Snapshot dates are local time on Unix hosts and UTC on Windows.
//! - A format driver whose `create_opts` list has not been ported yet takes any `-o` option
//!   and leaves the checking to the driver.

use ruvm_base::report::set_program_name;
use ruvm_block::tools;

mod amend;
mod bench;
mod bitmap;
mod buf;
mod check;
mod commit;
mod common;
mod compare;
mod convert;
mod create;
mod dd;
mod dump;
pub mod getopt;
mod info;
mod map;
mod measure;
mod progress;
mod rebase;
mod resize;
mod snapshot;

pub use common::{Exit, Flow};

/// What qemu-io shares with qemu-img.
pub mod shared {
    pub use crate::common::{
        BDRV_DEFAULT_CACHE, CacheMode, fatal, graph, localtime, object_add, parse_cache_mode,
        register_object_types,
    };
}

use common::{error_exit, tryhelp};
use getopt::{Getopt, HasArg, LongOpt, Opt, lopt};

/// `img_cmd_t`.
pub(crate) struct Cmd {
    pub name: &'static str,
    pub handler: fn(&Cmd, Vec<String>) -> Flow<i32>,
    pub description: &'static str,
}

impl Cmd {
    /// `cmd_help()` of this command.
    pub(crate) fn help(&self, syntax: &str, arguments: &str) -> Exit {
        common::cmd_help(self.name, self.description, syntax, arguments)
    }
}

/// `img_cmds`.
static CMDS: &[Cmd] = &[
    Cmd {
        name: "amend",
        handler: amend::run,
        description: "Update format-specific options of the image",
    },
    Cmd { name: "bench", handler: bench::run, description: "Run a simple image benchmark" },
    Cmd {
        name: "bitmap",
        handler: bitmap::run,
        description: "Perform modifications of the persistent bitmap in the image",
    },
    Cmd { name: "check", handler: check::run, description: "Check basic image integrity" },
    Cmd { name: "commit", handler: commit::run, description: "Commit image to its backing file" },
    Cmd {
        name: "compare",
        handler: compare::run,
        description: "Check if two images have the same contents",
    },
    Cmd {
        name: "convert",
        handler: convert::run,
        description: "Copy one or more images to another with optional format conversion",
    },
    Cmd {
        name: "create",
        handler: create::run,
        description: "Create and format a new image file",
    },
    Cmd {
        name: "dd",
        handler: dd::run,
        description: "Copy input to output with optional format conversion",
    },
    Cmd { name: "info", handler: info::run, description: "Display information about the image" },
    Cmd { name: "map", handler: map::run, description: "Dump image metadata" },
    Cmd {
        name: "measure",
        handler: measure::run,
        description: "Calculate the file size required for a new image",
    },
    Cmd {
        name: "rebase",
        handler: rebase::run,
        description: "Change the backing file of the image",
    },
    Cmd { name: "resize", handler: resize::run, description: "Resize the image" },
    Cmd {
        name: "snapshot",
        handler: snapshot::run,
        description: "List or manipulate snapshots in the image",
    },
];

const QEMU_HELP_BOTTOM: &str = "See <https://qemu.org/contribute/report-a-bug> for how to report \
                                bugs.\nMore information on the QEMU project at \
                                <https://qemu.org>.";

/// The help text of `qemu-img --help`.
fn help_text(version: &str) -> String {
    let mut out = format!(
        "{version}QEMU disk image utility.  Usage:\n\n  qemu-img [standard options] COMMAND \
         [--help | command options]\n\nStandard options:\n  -h, --help\n     display this help \
         and exit\n  -V, --version\n     display version info and exit\n  -T,--trace TRACE\n     \
         specify tracing options:\n        [[enable=]<pattern>][,events=<file>][,file=<file>]\n\n\
         Recognized commands (run qemu-img COMMAND --help for command-specific help):\n\n"
    );
    for cmd in CMDS {
        out.push_str(&format!("  {} - {}\n", cmd.name, cmd.description));
    }
    out.push_str("\nSupported image formats:\n");
    // format_print(), assuming 76 columns.
    let mut c = 99;
    for name in tools::format_names() {
        if c + name.len() > 75 {
            out.push_str("\n ");
            c = 1;
        }
        out.push_str(&format!(" {name}"));
        c += name.len() + 1;
    }
    if c != 0 {
        out.push('\n');
    }
    out.push_str(&format!("\n{QEMU_HELP_BOTTOM}\n"));
    out
}

/// qemu-img's `main()`. `argv0` is how the program was called, `args` what came after it and
/// `version` the text of `-V` (`QEMU_IMG_VERSION`). Returns the exit status.
pub fn main(argv0: &str, args: &[String], version: &str) -> u8 {
    let prgname = argv0.rsplit('/').next().unwrap_or(argv0);
    set_program_name(prgname);
    common::register_object_types();
    let r = run(argv0, args, version);
    use std::io::Write;
    let _ = std::io::stdout().flush();
    match r {
        Ok(code) => code as u8,
        Err(Exit(code)) => code,
    }
}

const MAIN_LONGS: &[LongOpt] = &[
    lopt("help", HasArg::No, b'h' as i32),
    lopt("version", HasArg::No, b'V' as i32),
    lopt("trace", HasArg::Required, b'T' as i32),
];

fn run(argv0: &str, args: &[String], version: &str) -> Flow<i32> {
    let mut argv = vec![argv0.to_string()];
    argv.extend_from_slice(args);
    let mut g = Getopt::new(argv, "+hVT:", MAIN_LONGS);
    while let Some(o) = g.next() {
        match o {
            Opt::Opt(c, _) if c == b'h' as i32 => {
                print!("{}", help_text(version));
                return Ok(0);
            }
            Opt::Opt(c, _) if c == b'V' as i32 => {
                print!("{version}");
                return Ok(0);
            }
            Opt::Opt(c, _) if c == b'T' as i32 => {}
            _ => return Err(tryhelp(argv0)),
        }
    }
    let rest = g.rest().to_vec();
    let Some(cmdname) = rest.first() else {
        return Err(error_exit(argv0, "Not enough arguments"));
    };
    let Some(cmd) = CMDS.iter().find(|c| c.name == cmdname) else {
        return Err(error_exit(argv0, &format!("Command not found: {cmdname}")));
    };
    let mut cargs = rest.clone();
    cargs[0] = format!("{argv0} {cmdname}");
    (cmd.handler)(cmd, cargs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_are_sorted() {
        assert!(CMDS.windows(2).all(|w| w[0].name < w[1].name));
    }
}
