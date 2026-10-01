// SPDX-License-Identifier: GPL-2.0-or-later

//! The commands: qemu-io-cmds.c, and `open`, `close` and `quit` of qemu-io.c.

use std::io::{self, Read, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error};
use ruvm_block::accounting::BlockAcctType;
use ruvm_block::tools::{OpenFlags, protocol_name};
use ruvm_block::{BLK_PERM_RESIZE, BLK_PERM_WRITE, BLK_PERM_WRITE_UNCHANGED, BlockBackend};
use ruvm_img::getopt::{Getopt, Opt};
use ruvm_img::shared::graph;
use ruvm_qapi::QDict;
use ruvm_qapi::cutils::{Errno, strtoi64, strtosz};
use ruvm_qapi::opts::QemuOptsList;
use ruvm_qapi::types::PreallocMode;

use crate::fmt::{cvtstr, dump_buffer, image_info_specific_dump, report};
use crate::{OpenArgs, State, apply_cache_mode, openfile, out, p, parse_aio, parse_discard};

const EIO: i32 = 5;
const EBUSY: i32 = 16;
const EINVAL: i32 = 22;
const ERANGE: i32 = 34;

/// `BDRV_REQUEST_MAX_BYTES`.
const BDRV_REQUEST_MAX_BYTES: i64 = 2147483136;
/// `BDRV_MAX_LENGTH`.
const BDRV_MAX_LENGTH: i64 = i64::MAX & !((1 << 30) - 1);
/// `BDRV_SECTOR_SIZE`.
const BDRV_SECTOR_SIZE: i64 = 512;

/// `CMD_NOFILE_OK`.
const NOFILE_OK: u8 = 1;
/// `CMD_FLAG_GLOBAL`.
const GLOBAL: u8 = 2;

/// Where the zone commands start reading arguments: `qemu_reset_optind()` leaves `optind` at
/// 1 where there is `optreset` and at 0 with glibc, and they increment it first.
const ZONE_OPTIND: usize = if cfg!(target_os = "linux") { 0 } else { 1 };

/// `cmdinfo_t`.
pub(crate) struct Cmd {
    name: &'static str,
    altname: Option<&'static str>,
    cfunc: fn(&mut State, &[String]) -> i32,
    perm: u64,
    argmin: i32,
    argmax: i32,
    flags: u8,
    args: Option<&'static str>,
    oneline: &'static str,
    help: Option<fn()>,
}

const DEFAULT: Cmd = Cmd {
    name: "",
    altname: None,
    cfunc: |_, _| 0,
    perm: 0,
    argmin: 0,
    argmax: 0,
    flags: 0,
    args: None,
    oneline: "",
    help: None,
};

/// `cmdtab`, sorted by name as `qemuio_add_command()` keeps it.
static CMDTAB: &[Cmd] = &[
    Cmd {
        name: "abort",
        cfunc: abort_f,
        flags: NOFILE_OK,
        oneline: "simulate a program crash using abort(3)",
        ..DEFAULT
    },
    Cmd {
        name: "aio_discard",
        cfunc: aio_discard_f,
        perm: BLK_PERM_WRITE,
        argmin: 2,
        argmax: -1,
        args: Some("[-Cq] off len"),
        oneline: "asynchronously discards a number of bytes",
        help: Some(aio_discard_help),
        ..DEFAULT
    },
    Cmd {
        name: "aio_flush",
        cfunc: aio_flush_f,
        oneline: "completes all outstanding aio requests",
        ..DEFAULT
    },
    Cmd {
        name: "aio_read",
        cfunc: aio_read_f,
        argmin: 2,
        argmax: -1,
        args: Some("[-Ciqrv] [-P pattern] off len [len..]"),
        oneline: "asynchronously reads a number of bytes",
        help: Some(aio_read_help),
        ..DEFAULT
    },
    Cmd {
        name: "aio_write",
        cfunc: aio_write_f,
        perm: BLK_PERM_WRITE,
        argmin: 2,
        argmax: -1,
        args: Some("[-Cfiqruz] [-P pattern] off len [len..]"),
        oneline: "asynchronously writes a number of bytes",
        help: Some(aio_write_help),
        ..DEFAULT
    },
    Cmd {
        name: "alloc",
        altname: Some("a"),
        cfunc: alloc_f,
        argmin: 1,
        argmax: 2,
        args: Some("offset [count]"),
        oneline: "checks if offset is allocated in the file",
        ..DEFAULT
    },
    Cmd {
        name: "break",
        cfunc: break_f,
        argmin: 2,
        argmax: 2,
        args: Some("event tag"),
        oneline: "sets a breakpoint on event and tags the stopped request as tag",
        ..DEFAULT
    },
    Cmd {
        name: "close",
        altname: Some("c"),
        cfunc: close_f,
        oneline: "close the current open file",
        ..DEFAULT
    },
    Cmd {
        name: "discard",
        altname: Some("d"),
        cfunc: discard_f,
        perm: BLK_PERM_WRITE,
        argmin: 2,
        argmax: -1,
        args: Some("[-Cq] off len"),
        oneline: "discards a number of bytes at a specified offset",
        help: Some(discard_help),
        ..DEFAULT
    },
    Cmd {
        name: "flush",
        altname: Some("f"),
        cfunc: flush_f,
        oneline: "flush all in-core file state to disk",
        ..DEFAULT
    },
    Cmd {
        name: "help",
        altname: Some("?"),
        cfunc: help_f,
        argmin: 0,
        argmax: 1,
        flags: GLOBAL,
        args: Some("[command]"),
        oneline: "help for one or all commands",
        ..DEFAULT
    },
    Cmd {
        name: "info",
        altname: Some("i"),
        cfunc: info_f,
        oneline: "prints information about the current file",
        ..DEFAULT
    },
    Cmd {
        name: "length",
        altname: Some("l"),
        cfunc: length_f,
        oneline: "gets the length of the current file",
        ..DEFAULT
    },
    Cmd {
        name: "map",
        cfunc: map_f,
        args: Some(""),
        oneline: "prints the allocated areas of a file",
        ..DEFAULT
    },
    Cmd {
        name: "open",
        altname: Some("o"),
        cfunc: open_f,
        argmin: 1,
        argmax: -1,
        flags: NOFILE_OK,
        args: Some("[-rsCnkU] [-t cache] [-d discard] [-o options] [path]"),
        oneline: "open the file specified by path",
        help: Some(open_help),
        ..DEFAULT
    },
    Cmd {
        name: "quit",
        altname: Some("q"),
        cfunc: quit_f,
        argmin: -1,
        argmax: -1,
        flags: GLOBAL,
        oneline: "exit the program",
        ..DEFAULT
    },
    Cmd {
        name: "read",
        altname: Some("r"),
        cfunc: read_f,
        argmin: 2,
        argmax: -1,
        args: Some("[-abCqrv] [-P pattern [-s off] [-l len]] off len"),
        oneline: "reads a number of bytes at a specified offset",
        help: Some(read_help),
        ..DEFAULT
    },
    Cmd {
        name: "readv",
        cfunc: readv_f,
        argmin: 2,
        argmax: -1,
        args: Some("[-Cqrv] [-P pattern] off len [len..]"),
        oneline: "reads a number of bytes at a specified offset",
        help: Some(readv_help),
        ..DEFAULT
    },
    Cmd {
        name: "remove_break",
        cfunc: remove_break_f,
        argmin: 1,
        argmax: 1,
        args: Some("tag"),
        oneline: "remove a breakpoint by tag",
        ..DEFAULT
    },
    Cmd {
        name: "reopen",
        cfunc: reopen_f,
        argmin: 0,
        argmax: -1,
        args: Some("[(-r|-w)] [-c cache] [-o options]"),
        oneline: "reopens an image with new options",
        help: Some(reopen_help),
        ..DEFAULT
    },
    Cmd {
        name: "resume",
        cfunc: resume_f,
        argmin: 1,
        argmax: 1,
        args: Some("tag"),
        oneline: "resumes the request tagged as tag",
        ..DEFAULT
    },
    Cmd {
        name: "sigraise",
        cfunc: sigraise_f,
        argmin: 1,
        argmax: 1,
        flags: NOFILE_OK,
        args: Some("signal"),
        oneline: "raises a signal",
        help: Some(sigraise_help),
        ..DEFAULT
    },
    Cmd {
        name: "sleep",
        cfunc: sleep_f,
        argmin: 1,
        argmax: 1,
        flags: NOFILE_OK,
        oneline: "waits for the given value in milliseconds",
        ..DEFAULT
    },
    Cmd {
        name: "truncate",
        altname: Some("t"),
        cfunc: truncate_f,
        perm: BLK_PERM_WRITE | BLK_PERM_RESIZE,
        argmin: 1,
        argmax: 3,
        args: Some("[-m prealloc_mode] off"),
        oneline: "truncates the current file at the given offset",
        ..DEFAULT
    },
    Cmd {
        name: "wait_break",
        cfunc: wait_break_f,
        argmin: 1,
        argmax: 1,
        args: Some("tag"),
        oneline: "waits for the suspension of a request",
        ..DEFAULT
    },
    Cmd {
        name: "write",
        altname: Some("w"),
        cfunc: write_f,
        perm: BLK_PERM_WRITE,
        argmin: 2,
        argmax: -1,
        args: Some("[-bcCfnqruz] [-P pattern | -s source_file] off len"),
        oneline: "writes a number of bytes at a specified offset",
        help: Some(write_help),
        ..DEFAULT
    },
    Cmd {
        name: "writev",
        cfunc: writev_f,
        perm: BLK_PERM_WRITE,
        argmin: 2,
        argmax: -1,
        args: Some("[-Cfqr] [-P pattern] off len [len..]"),
        oneline: "writes a number of bytes at a specified offset",
        help: Some(writev_help),
        ..DEFAULT
    },
    Cmd {
        name: "zone_append",
        altname: Some("zap"),
        cfunc: zone_append_f,
        argmin: 3,
        argmax: 4,
        args: Some("offset len [len..]"),
        oneline: "append write a number of bytes at a specified offset",
        ..DEFAULT
    },
    Cmd {
        name: "zone_close",
        altname: Some("zc"),
        cfunc: zone_close_f,
        argmin: 2,
        argmax: 2,
        args: Some("offset len"),
        oneline: "close a range of zones in zone block device",
        ..DEFAULT
    },
    Cmd {
        name: "zone_finish",
        altname: Some("zf"),
        cfunc: zone_finish_f,
        argmin: 2,
        argmax: 2,
        args: Some("offset len"),
        oneline: "finish a range of zones in zone block device",
        ..DEFAULT
    },
    Cmd {
        name: "zone_open",
        altname: Some("zo"),
        cfunc: zone_open_f,
        argmin: 2,
        argmax: 2,
        args: Some("offset len"),
        oneline: "explicit open a range of zones in zone block device",
        ..DEFAULT
    },
    Cmd {
        name: "zone_report",
        altname: Some("zrp"),
        cfunc: zone_report_f,
        argmin: 2,
        argmax: 2,
        args: Some("offset number"),
        oneline: "report zone information",
        ..DEFAULT
    },
    Cmd {
        name: "zone_reset",
        altname: Some("zrs"),
        cfunc: zone_reset_f,
        argmin: 2,
        argmax: 2,
        args: Some("offset len"),
        oneline: "reset a zone write pointer in zone block device",
        ..DEFAULT
    },
];

/// `find_command()`.
fn find_command(name: &str) -> Option<&'static Cmd> {
    CMDTAB.iter().find(|c| c.name == name || c.altname == Some(name))
}

/// `%s` of a string that may be NULL.
fn or_null(s: Option<&str>) -> &str {
    s.unwrap_or("(null)")
}

/// `qemuio_command_usage()`.
fn usage(ct: &Cmd) {
    p!("{} {} -- {}\n", ct.name, or_null(ct.args), ct.oneline);
}

/// `init_check_command()`.
fn init_check(state: &State, ct: &Cmd) -> bool {
    if ct.flags & GLOBAL != 0 {
        return true;
    }
    if ct.flags & NOFILE_OK == 0 && state.blk.is_none() {
        eprintln!("no file open, try 'help open'");
        return false;
    }
    true
}

/// `command()`.
fn command(state: &mut State, ct: &Cmd, argv: &[String]) -> i32 {
    let cmd = &argv[0];
    if !init_check(state, ct) {
        return -EINVAL;
    }
    let argc = argv.len() as i32;
    if argc - 1 < ct.argmin || (ct.argmax != -1 && argc - 1 > ct.argmax) {
        if ct.argmax == -1 {
            eprintln!(
                "bad argument count {} to {cmd}, expected at least {} arguments",
                argc - 1,
                ct.argmin
            );
        } else if ct.argmin == ct.argmax {
            eprintln!("bad argument count {} to {cmd}, expected {} arguments", argc - 1, ct.argmin);
        } else {
            eprintln!(
                "bad argument count {} to {cmd}, expected between {} and {} arguments",
                argc - 1,
                ct.argmin,
                ct.argmax
            );
        }
        return -EINVAL;
    }

    // Request additional permissions if necessary for this command. The caller is
    // responsible for restoring the original permissions afterwards if this is what it wants.
    if ct.perm != 0 {
        if let Some(blk) = state.blk.as_ref().filter(|b| b.is_inserted()) {
            let (perm, shared) = blk.perm();
            if ct.perm & !perm != 0 {
                if let Err(e) = blk.set_perm(perm | ct.perm, shared) {
                    report_error(&e);
                    return -EINVAL;
                }
            }
        }
    }
    (ct.cfunc)(state, argv)
}

/// `breakline()`: the words of `input`, split at single spaces.
fn breakline(input: &str) -> Vec<String> {
    input.split(' ').filter(|w| !w.is_empty()).map(str::to_string).collect()
}

/// `qemuio_command()`: runs one command line and returns its result, negative on failure.
pub(crate) fn qemuio_command(state: &mut State, cmd: &str) -> i32 {
    let v = breakline(cmd);
    let Some(name) = v.first() else {
        return 0;
    };
    match find_command(name) {
        Some(ct) => command(state, ct, &v),
        None => {
            eprintln!("command \"{name}\" not found");
            -EINVAL
        }
    }
}

/// The option string of a command: glibc permutes the arguments, the BSD `getopt()` stops at
/// the first operand.
macro_rules! shorts {
    ($s:literal) => {
        if cfg!(target_os = "linux") { $s } else { concat!("+", $s) }
    };
}

/// `getopt()` on the words of a command, with the messages of the host's libc.
struct CmdOpts {
    g: Getopt<'static>,
    shorts: &'static str,
}

impl CmdOpts {
    fn new(argv: &[String], shorts: &'static str) -> Self {
        let mut g = Getopt::new(argv.to_vec(), shorts, &[]);
        g.opterr = false;
        CmdOpts { g, shorts }
    }

    /// The next option, `Some(None)` after printing the message for a bad one.
    fn next(&mut self) -> Option<Option<(u8, String)>> {
        match self.g.next()? {
            Opt::Opt(c, arg) => Some(Some((c as u8, arg.unwrap_or_default()))),
            Opt::NonOpt(_) => None,
            Opt::Err => {
                let c = char::from(self.g.optopt as u8);
                let known = c != ':' && c != '+' && self.shorts.contains(c);
                let cmd = &self.g.args()[0];
                match (known, cfg!(target_os = "linux")) {
                    (true, true) => eprintln!("{cmd}: option requires an argument -- '{c}'"),
                    (true, false) => eprintln!("{cmd}: option requires an argument -- {c}"),
                    (false, true) => eprintln!("{cmd}: invalid option -- '{c}'"),
                    (false, false) => eprintln!("{cmd}: illegal option -- {c}"),
                }
                Some(None)
            }
        }
    }

    fn optind(&self) -> usize {
        self.g.optind
    }

    fn argc(&self) -> usize {
        self.g.args().len()
    }

    fn argv(&self) -> &[String] {
        self.g.args()
    }
}

/// `argv[i]`, NULL past the end.
fn arg(argv: &[String], i: usize) -> Option<&str> {
    argv.get(i).map(String::as_str)
}

/// `cvtnum()`: a size with an optional suffix, or the negative errno.
fn cvtnum(s: Option<&str>) -> Result<i64, i32> {
    let Some(s) = s else {
        return Err(-EINVAL);
    };
    match strtosz(s) {
        Ok(v) if v > i64::MAX as u64 => Err(-ERANGE),
        Ok(v) => Ok(v as i64),
        Err(Errno::Inval) => Err(-EINVAL),
        Err(Errno::Range) => Err(-ERANGE),
    }
}

/// `print_cvtnum_err()`.
fn print_cvtnum_err(rc: i32, arg: Option<&str>) {
    let arg = or_null(arg);
    match -rc {
        EINVAL => {
            p!("Parsing error: non-numeric argument, or extraneous/unrecognized suffix -- {arg}\n")
        }
        ERANGE => p!("Parsing error: argument too large -- {arg}\n"),
        _ => p!("Parsing error: {arg}\n"),
    }
}

/// `cvtnum()` and `print_cvtnum_err()` together, as nearly every caller does.
fn num(s: Option<&str>) -> Result<i64, i32> {
    cvtnum(s).inspect_err(|&rc| print_cvtnum_err(rc, s))
}

/// `strtol(s, &end, 0)` where the whole string has to be a number.
fn strtol_all(s: &str) -> Option<i64> {
    match strtoi64(s, 0, false) {
        Ok((v, used)) if used == s.len() => Some(v),
        Err((Errno::Range, v)) => Some(v),
        _ if s.is_empty() => Some(0),
        _ => None,
    }
}

/// `parse_pattern()`.
fn parse_pattern(arg: &str) -> Option<u8> {
    match strtol_all(arg).and_then(|v| u8::try_from(v).ok()) {
        Some(v) => Some(v),
        None => {
            p!("{arg} is not a valid pattern byte\n");
            None
        }
    }
}

/// The negative errno of an I/O error.
fn neg_errno(e: &io::Error) -> i32 {
    -e.raw_os_error().unwrap_or(EIO)
}

/// `create_iovec()`: one buffer with the total size of the lengths, filled with `pattern`.
fn create_iovec(args: &[String], pattern: u8) -> Option<Vec<u8>> {
    let mut count: i64 = 0;
    for a in args {
        let len = num(Some(a)).ok()?;
        if len > BDRV_REQUEST_MAX_BYTES {
            p!("Argument '{a}' exceeds maximum size {BDRV_REQUEST_MAX_BYTES}\n");
            return None;
        }
        if count > BDRV_REQUEST_MAX_BYTES - len {
            p!("The total number of bytes exceed the maximum size {BDRV_REQUEST_MAX_BYTES}\n");
            return None;
        }
        count += len;
    }
    Some(vec![pattern; count as usize])
}

fn blk(state: &State) -> Arc<BlockBackend> {
    state.blk.clone().expect("init_check_command() checked for a file")
}

fn node(blk: &BlockBackend) -> String {
    blk.node_name().unwrap_or_default()
}

/// `BDRV_REQ_FUA`: the backend already flushes after writes when its write cache is off.
fn fua(blk: &BlockBackend) -> io::Result<()> {
    if blk.enable_write_cache() { blk.flush() } else { Ok(()) }
}

fn is_sector_aligned(offset: i64, count: i64) -> bool {
    if offset % BDRV_SECTOR_SIZE != 0 {
        p!("{offset} is not a sector-aligned value for 'offset'\n");
        return false;
    }
    if count % BDRV_SECTOR_SIZE != 0 {
        p!("{count} is not a sector-aligned value for 'count'\n");
        return false;
    }
    true
}

fn print_report(
    op: &str,
    t: Duration,
    offset: i64,
    count: i64,
    total: i64,
    cnt: u32,
    machine: bool,
) {
    out(&report(op, t, offset as u64, count as u64, total as u64, cnt, machine));
}

fn read_help() {
    out(
        "\n reads a range of bytes from the given offset\n\n Example:\n 'read -v 512 1k' - dumps 1 \
         kilobyte read from 512 bytes into the file\n\n Reads a segment of the currently open \
         file, optionally dumping it to the\n standard output stream (with -v option) for \
         subsequent inspection.\n -b, -- read from the VM state rather than the virtual disk\n \
         -C, -- report statistics in a machine parsable format\n -l, -- length for pattern \
         verification (only with -P)\n -p, -- ignored for backwards compatibility\n -P, -- use a \
         pattern to verify read data\n -q, -- quiet mode, do not show I/O statistics\n -r, -- \
         register I/O buffer\n -s, -- start offset for pattern verification (only with -P)\n \
         -v, -- dump buffer to standard output\n\n",
    );
}

fn read_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("read").unwrap();
    let (mut cflag, mut qflag, mut vflag) = (false, false, false);
    let (mut pflag, mut sflag, mut lflag, mut bflag) = (false, false, false, false);
    let mut pattern = 0u8;
    let (mut pattern_offset, mut pattern_count) = (0i64, 0i64);
    let mut rflag = false;
    let mut o = CmdOpts::new(argv, shorts!("bCl:pP:qrs:v"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'b' => bflag = true,
            b'C' => cflag = true,
            b'l' => {
                lflag = true;
                pattern_count = match num(Some(&optarg)) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
            }
            b'p' => {}
            b'P' => {
                pflag = true;
                pattern = match parse_pattern(&optarg) {
                    Some(v) => v,
                    None => return -EINVAL,
                };
            }
            b'q' => qflag = true,
            b'r' => rflag = true,
            b's' => {
                sflag = true;
                pattern_offset = match num(Some(&optarg)) {
                    Ok(v) => v,
                    Err(e) => return e,
                };
            }
            b'v' => vflag = true,
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 != o.argc() {
        usage(ct);
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let count = match num(arg(argv, i + 1)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if count > BDRV_REQUEST_MAX_BYTES {
        p!("length cannot exceed {BDRV_REQUEST_MAX_BYTES}, given {}\n", argv[i + 1]);
        return -EINVAL;
    }
    if !pflag && (lflag || sflag) {
        usage(ct);
        return -EINVAL;
    }
    if !lflag {
        pattern_count = count - pattern_offset;
    }
    if pattern_count < 0 || pattern_count + pattern_offset > count {
        p!("pattern verification range exceeds end of read data\n");
        return -EINVAL;
    }
    if bflag {
        if !is_sector_aligned(offset, count) {
            return -EINVAL;
        }
        if rflag {
            p!("I/O buffer registration is not supported when reading from vmstate\n");
            return -EINVAL;
        }
    }

    let mut buf = vec![0xabu8; count as usize];
    let t1 = Instant::now();
    let r = if bflag {
        blk.load_vmstate(offset as u64, &mut buf)
    } else {
        blk.pread(offset as u64, &mut buf)
    };
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("read failed: {}\n", strerror(&e));
        return neg_errno(&e);
    }

    let mut ret = 0;
    if pflag {
        let range = &buf[pattern_offset as usize..(pattern_offset + pattern_count) as usize];
        if range.iter().any(|&b| b != pattern) {
            p!(
                "Pattern verification failed at offset {}, {pattern_count} bytes\n",
                offset + pattern_offset
            );
            ret = -EINVAL;
        }
    }
    if qflag {
        return ret;
    }
    if vflag {
        out(&dump_buffer(&buf, offset as u64));
    }
    print_report("read", t, offset, count, count, 1, cflag);
    ret
}

fn readv_help() {
    out("\n reads a range of bytes from the given offset into multiple buffers\n\n Example:\n \
         'readv -v 512 1k 1k ' - dumps 2 kilobytes read from 512 bytes into the file\n\n Reads a \
         segment of the currently open file, optionally dumping it to the\n standard output \
         stream (with -v option) for subsequent inspection.\n Uses multiple iovec buffers if more \
         than one byte range is specified.\n -C, -- report statistics in a machine parsable \
         format\n -P, -- use a pattern to verify read data\n -q, -- quiet mode, do not show I/O \
         statistics\n -r, -- register I/O buffer\n -v, -- dump buffer to standard output\n\n");
}

fn readv_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("readv").unwrap();
    let (mut cflag, mut qflag, mut vflag) = (false, false, false);
    let mut pattern: Option<u8> = None;
    let mut o = CmdOpts::new(argv, shorts!("CP:qrv"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'C' => cflag = true,
            b'P' => match parse_pattern(&optarg) {
                Some(v) => pattern = Some(v),
                None => return -EINVAL,
            },
            b'q' => qflag = true,
            b'r' => {}
            b'v' => vflag = true,
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 > o.argc() {
        usage(ct);
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(mut buf) = create_iovec(&argv[i + 1..], 0xab) else {
        return -EINVAL;
    };

    let t1 = Instant::now();
    let r = blk.pread(offset as u64, &mut buf);
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("readv failed: {}\n", strerror(&e));
        return neg_errno(&e);
    }
    let size = buf.len() as i64;
    let mut ret = 0;
    if let Some(pattern) = pattern {
        if buf.iter().any(|&b| b != pattern) {
            p!("Pattern verification failed at offset {offset}, {size} bytes\n");
            ret = -EINVAL;
        }
    }
    if qflag {
        return ret;
    }
    if vflag {
        out(&dump_buffer(&buf, offset as u64));
    }
    print_report("read", t, offset, size, size, 1, cflag);
    ret
}

fn write_help() {
    out(
        "\n writes a range of bytes from the given offset\n\n Example:\n 'write 512 1k' - writes 1 \
         kilobyte at 512 bytes into the open file\n\n Writes into a segment of the currently open \
         file, using a buffer\n filled with a set pattern (0xcdcdcdcd).\n -b, -- write to the VM \
         state rather than the virtual disk\n -c, -- write compressed data with \
         blk_write_compressed\n -C, -- report statistics in a machine parsable format\n -f, -- \
         use Force Unit Access semantics\n -n, -- with -z, don't allow slow fallback\n -p, -- \
         ignored for backwards compatibility\n -P, -- use different pattern to fill file\n -q, \
         -- quiet mode, do not show I/O statistics\n -r, -- register I/O buffer\n -s, -- use a \
         pattern file to fill the write buffer\n -u, -- with -z, allow unmapping\n -z, -- write \
         zeroes using blk_pwrite_zeroes\n\n",
    );
}

/// `qemu_io_alloc_from_file()`: `len` bytes of `file_name`, repeated as often as needed.
fn alloc_from_file(len: usize, file_name: &str) -> Option<Vec<u8>> {
    let mut f = match std::fs::File::open(file_name) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{file_name}: {}", strerror(&e));
            return None;
        }
    };
    let mut buf = vec![0u8; len];
    let mut pattern_len = 0;
    while pattern_len < len {
        match f.read(&mut buf[pattern_len..]) {
            Ok(0) => break,
            Ok(n) => pattern_len += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                eprintln!("{file_name}: {}", strerror(&e));
                return None;
            }
        }
    }
    if pattern_len == 0 {
        eprintln!("{file_name}: file is empty");
        return None;
    }
    let mut p = pattern_len;
    while p < len {
        let n = pattern_len.min(len - p);
        buf.copy_within(..n, p);
        p += pattern_len;
    }
    Some(buf)
}

fn write_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("write").unwrap();
    let (mut cflag_c, mut qflag, mut bflag) = (false, false, false);
    let (mut pflag, mut zflag, mut cflag, mut sflag) = (false, false, false, false);
    let (mut fua_flag, mut no_fallback, mut rflag, mut unmap) = (false, false, false, false);
    let mut pattern = 0xcdu8;
    let mut file_name = String::new();
    let mut o = CmdOpts::new(argv, shorts!("bcCfnpP:qrs:uz"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'b' => bflag = true,
            b'c' => cflag = true,
            b'C' => cflag_c = true,
            b'f' => fua_flag = true,
            b'n' => no_fallback = true,
            b'p' => {}
            b'P' => {
                pflag = true;
                pattern = match parse_pattern(&optarg) {
                    Some(v) => v,
                    None => return -EINVAL,
                };
            }
            b'q' => qflag = true,
            b'r' => rflag = true,
            b's' => {
                sflag = true;
                file_name = optarg;
            }
            b'u' => unmap = true,
            b'z' => zflag = true,
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 != o.argc() {
        usage(ct);
        return -EINVAL;
    }
    if bflag && zflag {
        p!("-b and -z cannot be specified at the same time\n");
        return -EINVAL;
    }
    if fua_flag && (bflag || cflag) {
        p!("-f and -b or -c cannot be specified at the same time\n");
        return -EINVAL;
    }
    if no_fallback && !zflag {
        p!("-n requires -z to be specified\n");
        return -EINVAL;
    }
    if unmap && !zflag {
        p!("-u requires -z to be specified\n");
        return -EINVAL;
    }
    if u8::from(zflag) + u8::from(pflag) + u8::from(sflag) > 1 {
        p!("Only one of -z, -P, and -s can be specified at the same time\n");
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let count = match num(arg(argv, i + 1)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if count > BDRV_REQUEST_MAX_BYTES && !no_fallback {
        p!("length cannot exceed {BDRV_REQUEST_MAX_BYTES} without -n, given {}\n", argv[i + 1]);
        return -EINVAL;
    }
    if (bflag || cflag) && !is_sector_aligned(offset, count) {
        return -EINVAL;
    }

    let mut buf = Vec::new();
    if zflag {
        if rflag {
            p!("cannot combine zero write with registered I/O buffer\n");
            return -EINVAL;
        }
    } else if sflag {
        match alloc_from_file(count as usize, &file_name) {
            Some(b) => buf = b,
            None => return -EINVAL,
        }
    } else {
        buf = vec![pattern; count as usize];
    }

    let t1 = Instant::now();
    let r = if bflag {
        blk.save_vmstate(offset as u64, &buf)
    } else if zflag {
        blk.pwrite_zeroes(offset as u64, count as u64, unmap)
            .and_then(|()| if fua_flag { fua(&blk) } else { Ok(()) })
    } else if cflag {
        blk.pwrite_compressed(offset as u64, &buf)
    } else {
        blk.pwrite(offset as u64, &buf).and_then(|()| if fua_flag { fua(&blk) } else { Ok(()) })
    };
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("write failed: {}\n", strerror(&e));
        return neg_errno(&e);
    }
    if !qflag {
        print_report("wrote", t, offset, count, count, 1, cflag_c);
    }
    0
}

fn writev_help() {
    out("\n writes a range of bytes from the given offset source from multiple buffers\n\n \
         Example:\n 'writev 512 1k 1k' - writes 2 kilobytes at 512 bytes into the open file\n\n \
         Writes into a segment of the currently open file, using a buffer\n filled with a set \
         pattern (0xcdcdcdcd).\n -C, -- report statistics in a machine parsable format\n -f, -- \
         use Force Unit Access semantics\n -P, -- use different pattern to fill file\n -q, -- \
         quiet mode, do not show I/O statistics\n -r, -- register I/O buffer\n\n");
}

fn writev_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("writev").unwrap();
    let (mut cflag, mut qflag, mut fua_flag) = (false, false, false);
    let mut pattern = 0xcdu8;
    let mut o = CmdOpts::new(argv, shorts!("CfP:qr"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'C' => cflag = true,
            b'f' => fua_flag = true,
            b'q' => qflag = true,
            b'r' => {}
            b'P' => {
                pattern = match parse_pattern(&optarg) {
                    Some(v) => v,
                    None => return -EINVAL,
                };
            }
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 > o.argc() {
        usage(ct);
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(buf) = create_iovec(&argv[i + 1..], pattern) else {
        return -EINVAL;
    };

    let t1 = Instant::now();
    let r =
        blk.pwrite(offset as u64, &buf).and_then(|()| if fua_flag { fua(&blk) } else { Ok(()) });
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("writev failed: {}\n", strerror(&e));
        return neg_errno(&e);
    }
    if !qflag {
        let size = buf.len() as i64;
        print_report("wrote", t, offset, size, size, 1, cflag);
    }
    0
}

fn aio_read_help() {
    out("\n asynchronously reads a range of bytes from the given offset\n\n Example:\n 'aio_read \
         -v 512 1k 1k ' - dumps 2 kilobytes read from 512 bytes into the file\n\n Reads a segment \
         of the currently open file, optionally dumping it to the\n standard output stream (with \
         -v option) for subsequent inspection.\n The read is performed asynchronously and the \
         aio_flush command must be\n used to ensure all outstanding aio requests have been \
         completed.\n Note that due to its asynchronous nature, this command will be\n considered \
         successful once the request is submitted, independently\n of potential I/O errors or \
         pattern mismatches.\n -C, -- report statistics in a machine parsable format\n -i, -- \
         treat request as invalid, for exercising stats\n -P, -- use a pattern to verify read \
         data\n -q, -- quiet mode, do not show I/O statistics\n -r, -- register I/O buffer\n -v, \
         -- dump buffer to standard output\n\n");
}

fn aio_read_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("aio_read").unwrap();
    let (mut cflag, mut qflag, mut vflag) = (false, false, false);
    let mut pattern: Option<u8> = None;
    let mut o = CmdOpts::new(argv, shorts!("CiP:qrv"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'C' => cflag = true,
            b'P' => match parse_pattern(&optarg) {
                Some(v) => pattern = Some(v),
                None => return -EINVAL,
            },
            b'i' => {
                p!("injecting invalid read request\n");
                blk.acct_stats().invalid(BlockAcctType::Read);
                return 0;
            }
            b'q' => qflag = true,
            b'r' => {}
            b'v' => vflag = true,
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 > o.argc() {
        usage(ct);
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some(mut buf) = create_iovec(&argv[i + 1..], 0xab) else {
        blk.acct_stats().invalid(BlockAcctType::Read);
        return -EINVAL;
    };

    // aio_read_done()
    let t1 = Instant::now();
    let r = blk.pread(offset as u64, &mut buf);
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("readv failed: {}\n", strerror(&e));
        return 0;
    }
    let size = buf.len() as i64;
    if let Some(pattern) = pattern {
        if buf.iter().any(|&b| b != pattern) {
            p!("Pattern verification failed at offset {offset}, {size} bytes\n");
        }
    }
    if qflag {
        return 0;
    }
    if vflag {
        out(&dump_buffer(&buf, offset as u64));
    }
    print_report("read", t, offset, size, size, 1, cflag);
    0
}

fn aio_write_help() {
    out("\n asynchronously writes a range of bytes from the given offset source\n from multiple \
         buffers\n\n Example:\n 'aio_write 512 1k 1k' - writes 2 kilobytes at 512 bytes into the \
         open file\n\n Writes into a segment of the currently open file, using a buffer\n filled \
         with a set pattern (0xcdcdcdcd).\n The write is performed asynchronously and the \
         aio_flush command must be\n used to ensure all outstanding aio requests have been \
         completed.\n Note that due to its asynchronous nature, this command will be\n considered \
         successful once the request is submitted, independently\n of potential I/O errors or \
         pattern mismatches.\n -C, -- report statistics in a machine parsable format\n -f, -- \
         use Force Unit Access semantics\n -i, -- treat request as invalid, for exercising \
         stats\n -P, -- use different pattern to fill file\n -q, -- quiet mode, do not show I/O \
         statistics\n -r, -- register I/O buffer\n -u, -- with -z, allow unmapping\n -z, -- \
         write zeroes using blk_aio_pwrite_zeroes\n\n");
}

fn aio_write_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("aio_write").unwrap();
    let (mut cflag, mut qflag, mut zflag) = (false, false, false);
    let (mut fua_flag, mut rflag, mut unmap) = (false, false, false);
    let mut pattern = 0xcdu8;
    let mut o = CmdOpts::new(argv, shorts!("CfiP:qruz"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'C' => cflag = true,
            b'f' => fua_flag = true,
            b'q' => qflag = true,
            b'r' => rflag = true,
            b'u' => unmap = true,
            b'P' => {
                pattern = match parse_pattern(&optarg) {
                    Some(v) => v,
                    None => return -EINVAL,
                };
            }
            b'i' => {
                p!("injecting invalid write request\n");
                blk.acct_stats().invalid(BlockAcctType::Write);
                return 0;
            }
            b'z' => zflag = true,
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() + 2 > o.argc() {
        usage(ct);
        return -EINVAL;
    }
    if zflag && o.optind() + 2 != o.argc() {
        p!("-z supports only a single length parameter\n");
        return -EINVAL;
    }
    if unmap && !zflag {
        p!("-u requires -z to be specified\n");
        return -EINVAL;
    }
    // QEMU also refuses -z with -P here, but never records that -P was given.
    if zflag && rflag {
        p!("cannot combine zero write with registered I/O buffer\n");
        return -EINVAL;
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let fua_after = |r: io::Result<()>| r.and_then(|()| if fua_flag { fua(&blk) } else { Ok(()) });
    let (t1, r, size) = if zflag {
        let count = match num(arg(argv, i + 1)) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let t1 = Instant::now();
        (t1, fua_after(blk.pwrite_zeroes(offset as u64, count as u64, unmap)), count)
    } else {
        let Some(buf) = create_iovec(&argv[i + 1..], pattern) else {
            blk.acct_stats().invalid(BlockAcctType::Write);
            return -EINVAL;
        };
        let t1 = Instant::now();
        (t1, fua_after(blk.pwrite(offset as u64, &buf)), buf.len() as i64)
    };

    // aio_write_done()
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("aio_write failed: {}\n", strerror(&e));
        return 0;
    }
    if !qflag {
        print_report("wrote", t, offset, size, size, 1, cflag);
    }
    0
}

fn aio_flush_f(_state: &mut State, _argv: &[String]) -> i32 {
    graph().drain_all();
    0
}

fn flush_f(state: &mut State, _argv: &[String]) -> i32 {
    match blk(state).flush() {
        Ok(()) => 0,
        Err(e) => neg_errno(&e),
    }
}

/// The zone commands: every image here is `BLK_Z_NONE`.
fn zone_report_f(state: &mut State, argv: &[String]) -> i32 {
    let _ = blk(state);
    let mut i = ZONE_OPTIND + 1;
    if let Err(e) = num(arg(argv, i)) {
        return e;
    }
    i += 1;
    let val = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if val > i64::from(u32::MAX) {
        p!("Number of zones must be less than 2^32\n");
        return -ERANGE;
    }
    p!("zone report failed: Operation not supported\n");
    -95
}

/// `zone_open_f()` and the others that call `blk_zone_mgmt()`.
fn zone_mgmt(state: &mut State, argv: &[String], op: &str) -> i32 {
    let blk = blk(state);
    let mut i = ZONE_OPTIND + 1;
    let offset = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    i += 1;
    let len = match num(arg(argv, i)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    // blk_check_byte_request()
    let err = match blk.getlength() {
        Err(e) => Some(e),
        Ok(size) if offset > size as i64 || size as i64 - offset < len => {
            Some(io::Error::from_raw_os_error(EIO))
        }
        Ok(_) => None,
    };
    match err {
        Some(e) => {
            p!("zone {op} failed: {}\n", strerror(&e));
            neg_errno(&e)
        }
        None => {
            p!("zone {op} failed: Operation not supported\n");
            -95
        }
    }
}

fn zone_open_f(state: &mut State, argv: &[String]) -> i32 {
    zone_mgmt(state, argv, "open")
}

fn zone_close_f(state: &mut State, argv: &[String]) -> i32 {
    zone_mgmt(state, argv, "close")
}

fn zone_finish_f(state: &mut State, argv: &[String]) -> i32 {
    zone_mgmt(state, argv, "finish")
}

fn zone_reset_f(state: &mut State, argv: &[String]) -> i32 {
    zone_mgmt(state, argv, "reset")
}

fn zone_append_f(state: &mut State, argv: &[String]) -> i32 {
    let _ = blk(state);
    let mut optind = if cfg!(target_os = "linux") { 1 } else { ZONE_OPTIND };
    if optind + 3 > argv.len() {
        return -EINVAL;
    }
    // One getopt() call; any option at all counts as -p.
    let mut o = CmdOpts::new(argv, shorts!("p"));
    if let Some(opt) = o.next() {
        let _ = opt;
        optind = o.optind();
    }
    let argv = o.argv();
    let offset = match num(arg(argv, optind)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    optind += 1;
    let Some(buf) = create_iovec(&argv[optind.min(argv.len())..], 0xcd) else {
        return -EINVAL;
    };
    // bdrv_check_qiov_request()
    let bytes = buf.len() as i64;
    if offset > BDRV_MAX_LENGTH || offset > BDRV_MAX_LENGTH - bytes {
        p!("zone append failed: {}\n", strerror(&io::Error::from_raw_os_error(EIO)));
        return -EIO;
    }
    p!("zone append failed: Operation not supported\n");
    -95
}

fn truncate_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("truncate").unwrap();
    let mut prealloc = PreallocMode::Off;
    let mut o = CmdOpts::new(argv, shorts!("m:"));
    while let Some(opt) = o.next() {
        match opt {
            Some((b'm', optarg)) => match PreallocMode::from_name(&optarg) {
                Some(m) => prealloc = m,
                None => {
                    error_report(&format!("Invalid preallocation mode '{optarg}'"));
                    return -EINVAL;
                }
            },
            _ => {
                usage(ct);
                return -EINVAL;
            }
        }
    }
    let offset = match cvtnum(arg(o.argv(), o.optind())) {
        Ok(v) => v,
        Err(e) => {
            print_cvtnum_err(e, arg(argv, 1));
            return e;
        }
    };
    if let Err(e) = blk.truncate_full(offset as u64, false, prealloc) {
        report_error(&e);
        return -EINVAL;
    }
    0
}

fn length_f(state: &mut State, _argv: &[String]) -> i32 {
    match blk(state).getlength() {
        Ok(size) => {
            p!("{}\n", cvtstr(size as f64));
            0
        }
        Err(e) => {
            p!("getlength: {}\n", strerror(&e));
            neg_errno(&e)
        }
    }
}

fn info_f(state: &mut State, _argv: &[String]) -> i32 {
    let blk = blk(state);
    let node = node(&blk);
    let g = graph();
    if let Ok(d) = g.node_details(&node) {
        p!("format name: {}\n", d.driver);
        if let Some(proto) = protocol_name(&d.driver) {
            p!("format name: {proto}\n");
        }
    }
    let bdi = match g.driver_info(&node) {
        Ok(i) => i,
        Err(e) => return neg_errno(&e),
    };
    p!("cluster size: {}\n", cvtstr(bdi.cluster_size as f64));
    p!("vm state offset: {}\n", cvtstr(bdi.vm_state_offset as f64));
    match g.specific_info(&node) {
        Err(e) => {
            report_error(&e);
            return -EIO;
        }
        Ok(Some(spec)) => out(&image_info_specific_dump(&spec)),
        Ok(None) => {}
    }
    0
}

fn discard_help() {
    out("\n discards a range of bytes from the given offset\n\n Example:\n 'discard 512 1k' - \
         discards 1 kilobyte from 512 bytes into the file\n\n Discards a segment of the \
         currently open file.\n -C, -- report statistics in a machine parsable format\n -q, -- \
         quiet mode, do not show I/O statistics\n\n");
}

/// The options and operands of `discard` and `aio_discard`.
fn discard_args(argv: &[String], ct: &Cmd, max: bool) -> Result<(bool, bool, i64, i64), i32> {
    let (mut cflag, mut qflag) = (false, false);
    let mut o = CmdOpts::new(argv, shorts!("Cq"));
    while let Some(opt) = o.next() {
        match opt {
            Some((b'C', _)) => cflag = true,
            Some((b'q', _)) => qflag = true,
            _ => {
                usage(ct);
                return Err(-EINVAL);
            }
        }
    }
    if o.optind() + 2 != o.argc() {
        usage(ct);
        return Err(-EINVAL);
    }
    let argv = o.argv();
    let i = o.optind();
    let offset = num(arg(argv, i))?;
    let bytes = num(arg(argv, i + 1))?;
    if max && bytes > BDRV_REQUEST_MAX_BYTES {
        p!("length cannot exceed {BDRV_REQUEST_MAX_BYTES}, given {}\n", argv[i + 1]);
        return Err(-EINVAL);
    }
    Ok((cflag, qflag, offset, bytes))
}

fn discard_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("discard").unwrap();
    let (cflag, qflag, offset, bytes) = match discard_args(argv, ct, true) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let t1 = Instant::now();
    let r = blk.pdiscard(offset as u64, bytes as u64);
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("discard failed: {}\n", strerror(&e));
        return neg_errno(&e);
    }
    if !qflag {
        print_report("discard", t, offset, bytes, bytes, 1, cflag);
    }
    0
}

fn aio_discard_help() {
    out("\n asynchronously discards a range of bytes from the given offset\n\n Example:\n \
         'aio_discard 512 1k' - discards 1 kilobyte from 512 bytes into the file\n\n Discards a \
         segment of the currently open file.\n -C, -- report statistics in a machine parsable \
         format\n -q, -- quiet mode, do not show I/O statistics\n The discard is performed \
         asynchronously and the aio_flush command must be\n used to ensure all outstanding aio \
         requests have been completed.\n Note that due to its asynchronous nature, this command \
         will be\n considered successful once the request is submitted, independently\n of \
         potential I/O errors.\n\n");
}

fn aio_discard_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("aio_discard").unwrap();
    let (cflag, qflag, offset, bytes) = match discard_args(argv, ct, false) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let t1 = Instant::now();
    let r = blk.pdiscard(offset as u64, bytes as u64);
    let t = t1.elapsed();
    if let Err(e) = r {
        p!("aio_discard failed: {}\n", strerror(&e));
        return 0;
    }
    if !qflag {
        print_report("discarded ", t, offset, bytes, bytes, 1, cflag);
    }
    0
}

fn alloc_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let node = node(&blk);
    let start = match num(arg(argv, 1)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let mut count = if argv.len() == 3 {
        match num(arg(argv, 2)) {
            Ok(v) => v,
            Err(e) => return e,
        }
    } else {
        BDRV_SECTOR_SIZE
    };
    let mut offset = start;
    let mut remaining = count;
    let mut sum_alloc = 0;
    while remaining != 0 {
        let (allocated, n) = match graph().is_allocated(&node, offset as u64, remaining as u64) {
            Ok(v) => v,
            Err(e) => {
                p!("is_allocated failed: {}\n", strerror(&e));
                return neg_errno(&e);
            }
        };
        let n = n as i64;
        offset += n;
        remaining -= n;
        if allocated {
            sum_alloc += n;
        }
        if n == 0 {
            count -= remaining;
            remaining = 0;
        }
    }
    p!("{sum_alloc}/{count} bytes allocated at offset {}\n", cvtstr(start as f64));
    0
}

/// `map_is_allocated()`: the status at `offset` and how far it goes.
fn map_is_allocated(node: &str, mut offset: u64, mut bytes: u64) -> io::Result<(bool, u64)> {
    let g = graph();
    let (first, mut num) = g.is_allocated(node, offset, bytes)?;
    let mut pnum = num;
    while bytes > 0 {
        offset += num;
        bytes -= num;
        match g.is_allocated(node, offset, bytes) {
            Ok((ret, n)) if ret == first && n != 0 => {
                pnum += n;
                num = n;
            }
            _ => break,
        }
    }
    Ok((first, pnum))
}

fn map_f(state: &mut State, _argv: &[String]) -> i32 {
    let blk = blk(state);
    let node = node(&blk);
    let mut offset = 0u64;
    let mut bytes = match blk.getlength() {
        Ok(b) => b,
        Err(e) => {
            error_report(&format!("Failed to query image length: {}", strerror(&e)));
            return neg_errno(&e);
        }
    };
    while bytes != 0 {
        let (allocated, n) = match map_is_allocated(&node, offset, bytes) {
            Ok(v) => v,
            Err(e) => {
                error_report(&format!("Failed to get allocation status: {}", strerror(&e)));
                return neg_errno(&e);
            }
        };
        if n == 0 {
            error_report("Unexpected end of image");
            return -EIO;
        }
        p!(
            "{} (0x{n:x}) bytes {} at offset {} (0x{offset:x})\n",
            cvtstr(n as f64),
            if allocated { "    allocated" } else { "not allocated" },
            cvtstr(offset as f64)
        );
        offset += n;
        bytes -= n;
    }
    0
}

fn reopen_help() {
    out("\n Changes the open options of an already opened image\n\n Example:\n 'reopen -o \
         lazy-refcounts=on' - activates lazy refcount writeback on a qcow2 image\n\n -r, -- \
         Reopen the image read-only\n -w, -- Reopen the image read-write\n -c, -- Change the \
         cache mode to the given value\n -o, -- Changes block driver options (cf. 'open' \
         command)\n\n");
}

fn reopen_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    let ct = find_command("reopen").unwrap();
    let node = node(&blk);
    let g = graph();
    let mut flags = OpenFlags::default();
    if let Ok(d) = g.node_details(&node) {
        flags.rdwr = !d.read_only;
    }
    if let Ok((direct, no_flush)) = g.node_cache_flags(&node) {
        flags.nocache = direct;
        flags.no_flush = no_flush;
    }
    let mut writethrough = !blk.enable_write_cache();
    let mut has_rw_option = false;
    let mut has_cache_option = false;

    let mut o = CmdOpts::new(argv, shorts!("c:o:rw"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            state.reopen_opts.reset();
            usage(ct);
            return -EINVAL;
        };
        match c {
            b'c' => {
                if !apply_cache_mode(&optarg, &mut flags, &mut writethrough) {
                    error_report(&format!("Invalid cache option: {optarg}"));
                    return -EINVAL;
                }
                has_cache_option = true;
            }
            b'o' => {
                if state.reopen_opts.parse_noisily(&optarg, false).is_none() {
                    state.reopen_opts.reset();
                    return -EINVAL;
                }
            }
            b'r' | b'w' => {
                if has_rw_option {
                    error_report("Only one -r/-w option may be given");
                    return -EINVAL;
                }
                flags.rdwr = c == b'w';
                has_rw_option = true;
            }
            _ => {
                state.reopen_opts.reset();
                usage(ct);
                return -EINVAL;
            }
        }
    }
    if o.optind() != o.argc() {
        state.reopen_opts.reset();
        usage(ct);
        return -EINVAL;
    }
    if !writethrough != blk.enable_write_cache() && blk.attached_dev().is_some() {
        error_report("Cannot change cache.writeback: Device attached");
        state.reopen_opts.reset();
        return -EBUSY;
    }
    if !flags.rdwr {
        g.drain_all();
        let (perm, shared) = blk.perm();
        let _ = blk.set_perm(perm & !(BLK_PERM_WRITE | BLK_PERM_WRITE_UNCHANGED), shared);
    }
    let mut opts = state.reopen_opts.find(None).map(|q| q.to_qdict()).unwrap_or_default();
    state.reopen_opts.reset();
    if opts.contains_key("read-only") {
        if has_rw_option {
            error_report("Cannot set both -r/-w and 'read-only'");
            return -EINVAL;
        }
    } else {
        opts.put("read-only", !flags.rdwr);
    }
    if opts.contains_key("cache.direct") || opts.contains_key("cache.no-flush") {
        if has_cache_option {
            error_report("Cannot set both -c and the cache options");
            return -EINVAL;
        }
    } else {
        opts.put("cache.direct", flags.nocache);
        opts.put("cache.no-flush", flags.no_flush);
    }
    if let Err(e) = g.reopen_node(&node, opts, true) {
        report_error(&e);
        return -EINVAL;
    }
    blk.set_enable_write_cache(!writethrough);
    0
}

fn break_f(state: &mut State, argv: &[String]) -> i32 {
    match blk(state).debug_breakpoint(&argv[1], &argv[2]) {
        Ok(()) => 0,
        Err(e) => {
            p!("Could not set breakpoint: {}\n", strerror(&e));
            neg_errno(&e)
        }
    }
}

fn remove_break_f(state: &mut State, argv: &[String]) -> i32 {
    match blk(state).debug_remove_breakpoint(&argv[1]) {
        Ok(()) => 0,
        Err(e) => {
            p!("Could not remove breakpoint {}: {}\n", argv[1], strerror(&e));
            neg_errno(&e)
        }
    }
}

fn resume_f(state: &mut State, argv: &[String]) -> i32 {
    match blk(state).debug_resume(&argv[1]) {
        Ok(()) => 0,
        Err(e) => {
            p!("Could not resume request: {}\n", strerror(&e));
            neg_errno(&e)
        }
    }
}

fn wait_break_f(state: &mut State, argv: &[String]) -> i32 {
    let blk = blk(state);
    while !blk.debug_is_suspended(&argv[1]) {
        std::thread::sleep(Duration::from_millis(1));
    }
    0
}

fn abort_f(_state: &mut State, _argv: &[String]) -> i32 {
    std::process::abort()
}

/// `NSIG`.
const NSIG: i64 = if cfg!(target_os = "linux") {
    65
} else if cfg!(windows) {
    23
} else {
    32
};

/// `SIGTERM`.
const SIGTERM: i32 = 15;

fn sigraise_help() {
    p!("\n raises the given signal\n\n Example:\n 'sigraise {SIGTERM}' - raises SIGTERM\n\n \
         Invokes raise(signal), where \"signal\" is the mandatory integer argument\n given to \
         sigraise.\n\n");
}

fn sigraise_f(_state: &mut State, argv: &[String]) -> i32 {
    let sig = match num(arg(argv, 1)) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if sig > NSIG {
        p!("signal argument '{}' is too large to be a valid signal\n", argv[1]);
        return -EINVAL;
    }
    // Using raise() to kill this process does not necessarily flush all open streams. At
    // least stdout and stderr should be flushed, though.
    crate::flush_out();
    let _ = io::stderr().flush();
    raise(sig as i32);
    0
}

#[cfg(unix)]
#[allow(unsafe_code)]
fn raise(sig: i32) {
    // SAFETY: raise() takes any integer and only delivers a signal to this thread; an invalid
    // number makes it fail with EINVAL, which qemu-io ignores too.
    unsafe {
        libc::raise(sig);
    }
}

#[cfg(not(unix))]
fn raise(sig: i32) {
    if sig != 0 {
        std::process::abort();
    }
}

fn sleep_f(_state: &mut State, argv: &[String]) -> i32 {
    match strtol_all(&argv[1]) {
        Some(ms) if ms >= 0 => {
            std::thread::sleep(Duration::from_millis(ms as u64));
            0
        }
        _ => {
            p!("{} is not a valid number\n", argv[1]);
            -EINVAL
        }
    }
}

/// `help_oneline()`.
fn help_oneline(cmd: &str, ct: &Cmd) {
    p!("{cmd} ");
    if let Some(a) = ct.args {
        p!("{a} ");
    }
    p!("-- {}\n", ct.oneline);
}

fn help_f(_state: &mut State, argv: &[String]) -> i32 {
    let Some(name) = argv.get(1) else {
        for ct in CMDTAB {
            help_oneline(ct.name, ct);
        }
        p!("\nUse 'help commandname' for extended help.\n");
        return 0;
    };
    let Some(ct) = find_command(name) else {
        p!("command {name} not found\n");
        return -EINVAL;
    };
    help_oneline(name, ct);
    if let Some(h) = ct.help {
        h();
    }
    0
}

fn close_f(state: &mut State, _argv: &[String]) -> i32 {
    state.blk = None;
    0
}

fn open_help() {
    out("\n opens a new file in the requested mode\n\n Example:\n 'open -n -o driver=raw \
         /tmp/data' - opens raw data file read-write, uncached\n\n Opens a file for subsequent \
         use by all of the other qemu-io commands.\n -r, -- open file read-only\n -s, -- use \
         snapshot file\n -C, -- use copy-on-read\n -n, -- disable host cache, short for -t \
         none\n -U, -- force shared permissions\n -k, -- use kernel AIO implementation (Linux \
         only, prefer use of -i)\n -i, -- use AIO mode (threads, native or io_uring)\n -t, -- \
         use the given cache mode for the image\n -d, -- use the given discard mode for the \
         image\n -o, -- options to be given to the block driver\n");
}

fn open_f(state: &mut State, argv: &[String]) -> i32 {
    let ct = find_command("open").unwrap();
    let mut args = OpenArgs {
        flags: OpenFlags { unmap: true, ..OpenFlags::default() },
        snapshot: false,
        writethrough: true,
        force_share: false,
    };
    let mut readonly = false;
    let mut o = CmdOpts::new(argv, shorts!("snCro:ki:t:d:U"));
    while let Some(opt) = o.next() {
        let Some((c, optarg)) = opt else {
            state.drive_opts.reset();
            usage(ct);
            return -EINVAL;
        };
        match c {
            b's' => args.snapshot = true,
            b'n' => {
                args.flags.nocache = true;
                args.writethrough = false;
            }
            b'C' => args.flags.copy_on_read = true,
            b'r' => readonly = true,
            b'k' => args.flags.native_aio = true,
            b't' => {
                if !apply_cache_mode(&optarg, &mut args.flags, &mut args.writethrough) {
                    error_report(&format!("Invalid cache option: {optarg}"));
                    state.drive_opts.reset();
                    return -EINVAL;
                }
            }
            b'd' => {
                if !parse_discard(&optarg, &mut args.flags) {
                    error_report(&format!("Invalid discard option: {optarg}"));
                    state.drive_opts.reset();
                    return -EINVAL;
                }
            }
            b'i' => {
                if !parse_aio(&optarg, &mut args.flags) {
                    error_report(&format!("Invalid aio option: {optarg}"));
                    state.drive_opts.reset();
                    return -EINVAL;
                }
            }
            b'o' => {
                if state.image_opts {
                    p!("--image-opts and 'open -o' are mutually exclusive\n");
                    state.drive_opts.reset();
                    return -EINVAL;
                }
                if state.drive_opts.parse_noisily(&optarg, false).is_none() {
                    state.drive_opts.reset();
                    return -EINVAL;
                }
            }
            b'U' => args.force_share = true,
            _ => {
                state.drive_opts.reset();
                usage(ct);
                return -EINVAL;
            }
        }
    }
    args.flags.rdwr = !readonly;
    let argv = o.argv().to_vec();
    let argc = argv.len();
    let mut optind = o.optind();
    if state.image_opts && optind + 1 == argc {
        if state.drive_opts.parse_noisily(&argv[optind], false).is_none() {
            state.drive_opts.reset();
            return -EINVAL;
        }
        optind += 1;
    }
    let opts: Option<QDict> = state.drive_opts.find(None).map(|q| q.to_qdict());
    state.drive_opts.reset();
    let ret = if optind + 1 == argc {
        openfile(state, Some(&argv[optind]), &args, opts)
    } else if optind == argc {
        openfile(state, None, &args, opts)
    } else {
        usage(ct);
        return -EINVAL;
    };
    if ret != 0 { -EINVAL } else { 0 }
}

fn quit_f(state: &mut State, _argv: &[String]) -> i32 {
    state.quit = true;
    0
}

/// `empty_opts` of qemu-io.c, the list `open -o` fills.
pub(crate) fn drive_opts() -> QemuOptsList {
    QemuOptsList::new("drive", &[]).with_merge_lists()
}

/// `reopen_opts`, the list `reopen -o` fills.
pub(crate) fn reopen_opts() -> QemuOptsList {
    QemuOptsList::new("reopen", &[]).with_merge_lists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_sorted() {
        assert!(CMDTAB.windows(2).all(|w| w[0].name < w[1].name));
        assert!(CMDTAB.iter().all(|c| c.perm == 0 || c.flags == 0));
    }

    #[test]
    fn words() {
        assert_eq!(breakline("  read  0 1 "), ["read", "0", "1"]);
        assert!(breakline("   ").is_empty());
    }

    #[test]
    fn numbers() {
        assert_eq!(cvtnum(Some("1k")), Ok(1024));
        assert_eq!(cvtnum(Some("x")), Err(-EINVAL));
        assert_eq!(cvtnum(None), Err(-EINVAL));
        assert_eq!(cvtnum(Some("16E")), Err(-ERANGE));
        assert_eq!(strtol_all("0x10"), Some(16));
        assert_eq!(strtol_all("010"), Some(8));
        assert_eq!(strtol_all("1a"), None);
    }

    #[test]
    fn lookup() {
        assert_eq!(find_command("zrp").map(|c| c.name), Some("zone_report"));
        assert_eq!(find_command("?").map(|c| c.name), Some("help"));
        assert!(find_command("nope").is_none());
    }
}
