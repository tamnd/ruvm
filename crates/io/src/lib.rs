// SPDX-License-Identifier: GPL-2.0-or-later

//! qemu-io, the block I/O exerciser: qemu-io.c and qemu-io-cmds.c.
//!
//! [`main`] takes the arguments after `argv[0]` and returns the exit status. It opens the image
//! given on the command line, runs the `-c` commands or else reads commands from standard input
//! after a `qemu-io> ` prompt, and exits with 1 if any command failed. Every option and command of
//! QEMU 11.1 is there, with the same help texts, messages, statistics lines and exit statuses.
//!
//! Differences from QEMU:
//!
//! - There is no readline. Lines are read the way `fetchline_fgets()` reads them even from a
//!   terminal, so there is no line editing, history or command completion. When standard input
//!   is a terminal, the end of input prints a newline as QEMU's readline does.
//! - The `aio_*` commands run to completion before they return, so their statistics are printed
//!   right away instead of from a later command, and `aio_flush` has nothing to wait for. A
//!   request stopped at a blkdebug `break` point therefore holds up the command that issued it.
//! - `aio_write -z` reports the time the request took. QEMU never starts its clock for it and
//!   prints the time since boot.
//! - `-s` (`snapshot=on`) fails: there are no temporary snapshot overlays.
//! - `-T`/`--trace` is accepted and ignored; there are no trace events.
//! - `-m` (misalign) and the `-r` (registered buffer) flags of the I/O commands are accepted and
//!   change nothing, since buffers are not handed to the host with `O_DIRECT` alignment rules.
//!   `write -n` (no fallback) is accepted and the zero write may still fall back.
//! - `-i io_uring` is refused as in a QEMU built without liburing; `aio=native` uses io_uring
//!   underneath on Linux anyway.
//! - The zone commands behave as they do in QEMU on an image that is not zoned, which is every
//!   image the block layer can open: they fail with "Operation not supported", after the argument
//!   checks QEMU does. Their argument parsing keeps QEMU's quirk of starting at `argv[2]` on the
//!   BSDs and macOS, where `qemu_reset_optind()` sets `optind` to 1.
//! - Block statistics are counted by the block backend for every request, so `aio_flush` and
//!   the `aio_*` commands add no accounting of their own; nothing in qemu-io prints them.
//! - `sigraise` on Windows ends the process with `abort()` whatever the signal.
//! - Options that the block driver does not know fail with "Parameter 'x' is unexpected" from
//!   the block layer rather than "Block format 'raw' does not support the option 'x'".

#![deny(unsafe_code)]

use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex, OnceLock};

use ruvm_base::report::{error_report, report_error, set_program_name};
use ruvm_block::BlockBackend;
use ruvm_block::tools::OpenFlags;
use ruvm_img::getopt::{Getopt, HasArg, LongOpt, Opt, lopt};
use ruvm_img::shared::{graph, object_add, parse_cache_mode, register_object_types};
use ruvm_img::{Exit, Flow};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::QemuOptsList;

mod cmds;
pub mod fmt;

/// The size of the stdio buffer glibc gives a standard output that is a pipe or a file.
const STDOUT_BUFSIZ: usize = 4096;

/// What `printf()` has buffered and not written yet, when standard output is not a terminal.
static STDOUT_BUF: Mutex<Vec<u8>> = Mutex::new(Vec::new());

/// Whether standard output is fully buffered, as stdio does when it is not a terminal.
fn fully_buffered() -> bool {
    static FULL: OnceLock<bool> = OnceLock::new();
    *FULL.get_or_init(|| !std::io::stdout().is_terminal())
}

/// Writes to standard output the way `printf()` does, ignoring a closed pipe. Into a pipe or a
/// file the output is written in blocks of [`STDOUT_BUFSIZ`] bytes and the rest at
/// [`flush_out`], so the messages on standard error come out in the same place relative to it
/// as QEMU's.
pub(crate) fn out(s: &str) {
    if !fully_buffered() {
        let _ = std::io::stdout().write_all(s.as_bytes());
        return;
    }
    let mut buf = STDOUT_BUF.lock().unwrap_or_else(|e| e.into_inner());
    buf.extend_from_slice(s.as_bytes());
    if buf.len() >= STDOUT_BUFSIZ {
        let whole = buf.len() / STDOUT_BUFSIZ * STDOUT_BUFSIZ;
        let mut o = std::io::stdout().lock();
        let _ = o.write_all(&buf[..whole]);
        let _ = o.flush();
        buf.drain(..whole);
    }
}

/// `fflush(stdout)`.
pub(crate) fn flush_out() {
    let mut buf = STDOUT_BUF.lock().unwrap_or_else(|e| e.into_inner());
    let mut o = std::io::stdout().lock();
    let _ = o.write_all(&buf);
    let _ = o.flush();
    buf.clear();
}

/// `printf()`.
macro_rules! p {
    ($($arg:tt)*) => {
        $crate::out(&format!($($arg)*))
    };
}
pub(crate) use p;

/// The global state of qemu-io.c.
pub(crate) struct State {
    /// `qemuio_blk`.
    pub blk: Option<Arc<BlockBackend>>,
    /// `quit_qemu_io`.
    pub quit: bool,
    /// `imageOpts`.
    pub image_opts: bool,
    /// `empty_opts` of qemu-io.c: what `open -o` has collected.
    pub drive_opts: QemuOptsList,
    /// `reopen_opts` of qemu-io-cmds.c: what `reopen -o` has collected.
    pub reopen_opts: QemuOptsList,
}

/// `QEMU_HELP_BOTTOM`.
const QEMU_HELP_BOTTOM: &str = "See <https://qemu.org/contribute/report-a-bug> for how to report \
                                bugs.\nMore information on the QEMU project at \
                                <https://qemu.org>.";

/// `usage()`.
fn usage(name: &str) {
    p!("Usage: {name} [OPTIONS]... [-c STRING]... [file]\nQEMU Disk exerciser\n\n  --object \
         OBJECTDEF   define an object such as 'secret' for\n                       passwords \
         and/or encryption keys\n  --image-opts         treat file as option string\n  -c, \
         --cmd STRING     execute command with its arguments\n                       from the \
         given string\n  -f, --format FMT     specifies the block driver to use\n  -r, \
         --read-only      export read-only\n  -s, --snapshot       use snapshot file\n  -n, \
         --nocache        disable host cache, short for -t none\n  -C, --copy-on-read   enable \
         copy-on-read\n  -m, --misalign       misalign allocations for O_DIRECT\n  -k, \
         --native-aio     use kernel AIO implementation\n                       (Linux only, \
         prefer use of -i)\n  -i, --aio=MODE       use AIO mode (threads, native or io_uring)\n  \
         -t, --cache=MODE     use the given cache mode for the image\n  -d, --discard=MODE   use \
         the given discard mode for the image\n  -T, --trace \
         [[enable=]<pattern>][,events=<file>][,file=<file>]\n                       specify \
         tracing options\n                       see qemu-img(1) man page for full \
         description\n  -U, --force-share    force shared permissions\n  -h, --help           \
         display this help and exit\n  -V, --version        output version information and \
         exit\n\nSee '{name} -c help' for information on available commands.\n\n\
         {QEMU_HELP_BOTTOM}\n");
}

/// `bdrv_parse_discard_flags()`.
pub(crate) fn parse_discard(mode: &str, flags: &mut OpenFlags) -> bool {
    match mode {
        "off" | "ignore" => flags.unmap = false,
        "on" | "unmap" => flags.unmap = true,
        _ => return false,
    }
    true
}

/// `bdrv_parse_aio()`. There is no `io_uring` mode, as in a QEMU built without liburing.
pub(crate) fn parse_aio(mode: &str, flags: &mut OpenFlags) -> bool {
    match mode {
        "threads" => {}
        "native" => flags.native_aio = true,
        _ => return false,
    }
    true
}

/// `bdrv_parse_cache_mode()` onto `flags` and `writethrough`.
pub(crate) fn apply_cache_mode(mode: &str, flags: &mut OpenFlags, writethrough: &mut bool) -> bool {
    match parse_cache_mode(mode) {
        Some(c) => {
            c.apply(flags);
            *writethrough = c.writethrough;
            true
        }
        None => false,
    }
}

/// What `open` and the command line ask of `openfile()` besides the flags.
pub(crate) struct OpenArgs {
    pub flags: OpenFlags,
    /// `BDRV_O_SNAPSHOT`.
    pub snapshot: bool,
    pub writethrough: bool,
    pub force_share: bool,
}

/// `openfile()`: 0 on success, 1 on failure.
pub(crate) fn openfile(
    state: &mut State,
    name: Option<&str>,
    args: &OpenArgs,
    opts: Option<QDict>,
) -> i32 {
    if state.blk.is_some() {
        error_report("file open already, try 'help close'");
        return 1;
    }
    let mut opts = opts;
    if args.force_share {
        let o = opts.get_or_insert_with(QDict::new);
        if o.contains_key("force-share") && o.get_str("force-share") != Some("on") {
            error_report("-U conflicts with image options");
            return 1;
        }
        o.put("force-share", "on");
    }
    let prefix = match name {
        Some(n) => format!("can't open device {n}: "),
        None => "can't open: ".to_string(),
    };
    if args.snapshot {
        error_report(&format!("{prefix}snapshot=on is not supported yet"));
        return 1;
    }
    match graph().blk_new_open(name, opts.unwrap_or_default(), args.flags) {
        Ok(blk) => {
            blk.set_enable_write_cache(!args.writethrough);
            state.blk = Some(blk);
            0
        }
        Err(e) => {
            report_error(&e.prepend(prefix));
            1
        }
    }
}

/// `MAXREADLINESZ`.
const MAXREADLINESZ: usize = 1024;

/// `fetchline_fgets()`: one line of at most `MAXREADLINESZ - 1` bytes without its newline, or
/// `None` at the end of the input.
fn fetchline(input: &mut impl std::io::BufRead) -> Option<String> {
    let mut line = Vec::new();
    while line.len() < MAXREADLINESZ - 1 {
        let buf = match input.fill_buf() {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        };
        if buf.is_empty() {
            break;
        }
        let room = MAXREADLINESZ - 1 - line.len();
        let n = buf.len().min(room);
        match buf[..n].iter().position(|&b| b == b'\n') {
            Some(i) => {
                line.extend_from_slice(&buf[..=i]);
                input.consume(i + 1);
                break;
            }
            None => {
                line.extend_from_slice(&buf[..n]);
                input.consume(n);
            }
        }
    }
    if line.is_empty() {
        return None;
    }
    // The line ends at the first NUL, as strlen() sees it.
    if let Some(i) = line.iter().position(|&b| b == 0) {
        line.truncate(i);
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    }
    Some(String::from_utf8_lossy(&line).into_owned())
}

/// `command_loop()`.
fn command_loop(state: &mut State, cmdline: &[String], prompt: &str) -> i32 {
    let mut last_error = 0;
    for c in cmdline {
        if state.quit {
            break;
        }
        let ret = cmds::qemuio_command(state, c);
        if ret < 0 {
            last_error = ret;
        }
    }
    if !cmdline.is_empty() {
        return last_error;
    }

    let tty = std::io::stdin().is_terminal();
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    while !state.quit {
        out(prompt);
        flush_out();
        let Some(line) = fetchline(&mut input) else {
            if tty {
                out("\n");
            }
            break;
        };
        let ret = cmds::qemuio_command(state, &line);
        if ret < 0 {
            last_error = ret;
        }
    }
    last_error
}

/// `OPTION_OBJECT`.
const OPTION_OBJECT: i32 = 256;
/// `OPTION_IMAGE_OPTS`.
const OPTION_IMAGE_OPTS: i32 = 257;

const LONGS: &[LongOpt] = &[
    lopt("help", HasArg::No, b'h' as i32),
    lopt("version", HasArg::No, b'V' as i32),
    lopt("cmd", HasArg::Required, b'c' as i32),
    lopt("format", HasArg::Required, b'f' as i32),
    lopt("read-only", HasArg::No, b'r' as i32),
    lopt("snapshot", HasArg::No, b's' as i32),
    lopt("nocache", HasArg::No, b'n' as i32),
    lopt("copy-on-read", HasArg::No, b'C' as i32),
    lopt("misalign", HasArg::No, b'm' as i32),
    lopt("native-aio", HasArg::No, b'k' as i32),
    lopt("aio", HasArg::Required, b'i' as i32),
    lopt("discard", HasArg::Required, b'd' as i32),
    lopt("cache", HasArg::Required, b't' as i32),
    lopt("trace", HasArg::Required, b'T' as i32),
    lopt("object", HasArg::Required, OPTION_OBJECT),
    lopt("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lopt("force-share", HasArg::No, b'U' as i32),
];

/// qemu-io's `main()`. `argv0` is how the program was called, `args` what came after it and
/// `version` the text of `-V` (`qemu-io version ...` and the copyright line). Returns the exit
/// status.
pub fn main(argv0: &str, args: &[String], version: &str) -> u8 {
    let prgname = argv0.rsplit(['/', '\\']).next().unwrap_or(argv0);
    set_program_name(prgname);
    register_object_types();
    let r = run(prgname, argv0, args, version);
    flush_out();
    match r {
        Ok(code) => code,
        Err(Exit(code)) => code,
    }
}

fn run(prgname: &str, argv0: &str, args: &[String], version: &str) -> Flow<u8> {
    let mut argv = vec![argv0.to_string()];
    argv.extend_from_slice(args);
    let mut g = Getopt::new(argv, "hVc:d:f:rsnCmki:t:T:U", LONGS);
    let mut open = OpenArgs {
        flags: OpenFlags { unmap: true, ..OpenFlags::default() },
        snapshot: false,
        writethrough: true,
        force_share: false,
    };
    let mut readonly = false;
    let mut format: Option<String> = None;
    let mut cmdline: Vec<String> = Vec::new();
    let mut image_opts = false;
    while let Some(o) = g.next() {
        let (c, arg) = match o {
            Opt::Opt(c, a) => (c, a.unwrap_or_default()),
            _ => {
                usage(prgname);
                return Err(Exit(1));
            }
        };
        match c {
            OPTION_OBJECT => object_add(&arg)?,
            OPTION_IMAGE_OPTS => image_opts = true,
            _ => match u8::try_from(c).map(char::from).unwrap_or('?') {
                's' => open.snapshot = true,
                'n' => {
                    open.flags.nocache = true;
                    open.writethrough = false;
                }
                'C' => open.flags.copy_on_read = true,
                'd' => {
                    if !parse_discard(&arg, &mut open.flags) {
                        error_report(&format!("Invalid discard option: {arg}"));
                        return Err(Exit(1));
                    }
                }
                'f' => format = Some(arg),
                'c' => cmdline.push(arg),
                'r' => readonly = true,
                'm' => {}
                'k' => open.flags.native_aio = true,
                'i' => {
                    if !parse_aio(&arg, &mut open.flags) {
                        error_report(&format!("Invalid aio option: {arg}"));
                        return Err(Exit(1));
                    }
                }
                't' => {
                    if !apply_cache_mode(&arg, &mut open.flags, &mut open.writethrough) {
                        error_report(&format!("Invalid cache option: {arg}"));
                        return Err(Exit(1));
                    }
                }
                'T' => {}
                'V' => {
                    let rest = version.strip_prefix("qemu-io").unwrap_or(version);
                    p!("{prgname}{rest}");
                    return Err(Exit(0));
                }
                'h' => {
                    usage(prgname);
                    return Err(Exit(0));
                }
                'U' => open.force_share = true,
                _ => {
                    usage(prgname);
                    return Err(Exit(1));
                }
            },
        }
    }
    let rest = g.rest().to_vec();
    if rest.len() > 1 {
        usage(prgname);
        return Err(Exit(1));
    }
    if format.is_some() && image_opts {
        error_report("--image-opts and -f are mutually exclusive");
        return Err(Exit(1));
    }

    let mut state = State {
        blk: None,
        quit: false,
        image_opts,
        drive_opts: cmds::drive_opts(),
        reopen_opts: cmds::reopen_opts(),
    };
    open.flags.rdwr = !readonly;
    if let Some(file) = rest.first() {
        if image_opts {
            let mut file_opts = QemuOptsList::new("file", &[]).with_implied_opt_name("file");
            let Some(qopts) = file_opts.parse_noisily(file, false) else {
                return Err(Exit(1));
            };
            let opts = qopts.to_qdict();
            if openfile(&mut state, None, &open, Some(opts)) != 0 {
                return Err(Exit(1));
            }
        } else {
            let opts = format.map(|f| {
                let mut d = QDict::new();
                d.put("driver", f);
                d
            });
            if openfile(&mut state, Some(file), &open, opts) != 0 {
                return Err(Exit(1));
            }
        }
    }
    let ret = command_loop(&mut state, &cmdline, &format!("{prgname}> "));

    // Make sure all outstanding requests complete before the program exits.
    graph().drain_all();
    state.blk = None;
    Ok(if ret < 0 { 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fetchline_splits_long_lines() {
        let long = "a".repeat(1500);
        let text = format!("one\n{long}\nlast");
        let mut r = std::io::Cursor::new(text.into_bytes());
        assert_eq!(fetchline(&mut r).as_deref(), Some("one"));
        assert_eq!(fetchline(&mut r).map(|l| l.len()), Some(1023));
        assert_eq!(fetchline(&mut r).map(|l| l.len()), Some(477));
        assert_eq!(fetchline(&mut r).as_deref(), Some("last"));
        assert_eq!(fetchline(&mut r), None);
    }

    #[test]
    fn fetchline_stops_at_nul() {
        let mut r = std::io::Cursor::new(b"ab\0cd\n".to_vec());
        assert_eq!(fetchline(&mut r).as_deref(), Some("ab"));
    }

    #[test]
    fn discard_and_aio_modes() {
        let mut f = OpenFlags::default();
        assert!(parse_discard("unmap", &mut f) && f.unmap);
        assert!(parse_discard("ignore", &mut f) && !f.unmap);
        assert!(!parse_discard("x", &mut f));
        assert!(parse_aio("native", &mut f) && f.native_aio);
        assert!(!parse_aio("io_uring", &mut f));
    }
}
