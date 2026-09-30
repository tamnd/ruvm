// SPDX-License-Identifier: GPL-2.0-or-later

//! The helpers at the top of qemu-img.c that every command uses: help and error exits, option
//! parsing helpers, `--object`, and opening images with `img_open()`.

use std::sync::{Arc, OnceLock};

use ruvm_base::Error;
use ruvm_base::report::{error_report, report_error};

use crate::getopt::{Getopt, HasArg, LongOpt, Opt, lopt};
use ruvm_block::tools::{self, OpenFlags};
use ruvm_block::{BlockBackend, BlockGraph};
use ruvm_qapi::cutils::{self, Errno};
use ruvm_qapi::keyval::keyval_parse;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptsList};
use ruvm_qapi::types::ObjectOptions;
use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
use ruvm_qapi::{QDict, QValue, json};
use ruvm_qom::{Registry, type_print_class_properties, user_creatable_print_types};

/// An early end of the program with this exit status, after whatever had to be printed. It is
/// how `exit()` in the middle of a command gets back to `main()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Exit(pub u8);

/// What a step that may end the program returns.
pub type Flow<T> = Result<T, Exit>;

/// The block graph every image of this run is opened in.
pub fn graph() -> &'static BlockGraph {
    static GRAPH: OnceLock<BlockGraph> = OnceLock::new();
    GRAPH.get_or_init(BlockGraph::new)
}

/// `tryhelp()`.
pub(crate) fn tryhelp(argv0: &str) -> Exit {
    eprintln!("Try '{argv0} --help' for more information");
    Exit(1)
}

/// `error_exit()`.
pub(crate) fn error_exit(argv0: &str, msg: &str) -> Exit {
    error_report(msg);
    tryhelp(argv0)
}

/// `error_report_err()` and `exit(1)`, what `&error_fatal` does.
pub fn fatal(e: &Error) -> Exit {
    report_error(e);
    Exit(1)
}

/// `cmd_help()`: the usage of one command.
pub(crate) fn cmd_help(name: &str, description: &str, syntax: &str, arguments: &str) -> Exit {
    print!(
        "Usage:\n  qemu-img {name} {syntax}\n{description}.\n\nArguments:\n  -h, --help\n     print \
         this help and exit\n{arguments}\n"
    );
    Exit(0)
}

/// `OutputFormat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Json,
    Human,
}

/// `parse_output_format()`.
pub(crate) fn parse_output_format(argv0: &str, arg: &str) -> Flow<OutputFormat> {
    match arg {
        "json" => Ok(OutputFormat::Json),
        "human" => Ok(OutputFormat::Human),
        _ => Err(error_exit(argv0, &format!("--output expects 'human' or 'json', not '{arg}'"))),
    }
}

/// `is_valid_option_list()`.
fn is_valid_option_list(list: &str) -> bool {
    if list.is_empty() || list.starts_with(',') {
        return false;
    }
    let trailing = list.len() - list.trim_end_matches(',').len();
    trailing % 2 == 0
}

/// `accumulate_options()`: false after reporting a bad list.
pub(crate) fn accumulate_options(options: &mut Option<String>, list: &str) -> bool {
    if !is_valid_option_list(list) {
        error_report(&format!("Invalid option list: {list}"));
        return false;
    }
    *options = Some(match options.take() {
        None => list.to_string(),
        Some(o) => format!("{o},{list}"),
    });
    true
}

/// `cvtnum_full()`: `None` after reporting the error.
pub(crate) fn cvtnum_full(
    name: &str,
    value: &str,
    is_size: bool,
    min: i64,
    max: i64,
) -> Option<i64> {
    let r = if is_size {
        cutils::strtosz(value)
    } else {
        cutils::strtou64(value, 0, true).map(|(v, _)| v).map_err(|(e, _)| e)
    };
    let range = || {
        error_report(&format!("Invalid {name} specified. Must be between {min} and {max}."));
        None
    };
    match r {
        Err(Errno::Range) => range(),
        Err(_) => {
            error_report(&format!("Invalid {name} specified: '{value}'"));
            None
        }
        Ok(v) if v > max as u64 || v < min as u64 => range(),
        Ok(v) => Some(v as i64),
    }
}

/// `cvtnum()`.
pub(crate) fn cvtnum(name: &str, value: &str, is_size: bool) -> Option<i64> {
    cvtnum_full(name, value, is_size, 0, i64::MAX)
}

/// Makes sure the QOM types `--object` can make are registered.
pub fn register_object_types() {
    ruvm_crypto::secret::register_types(Registry::global());
}

/// `user_creatable_process_cmdline()`: `--object`.
pub fn object_add(arg: &str) -> Flow<()> {
    let registry = Registry::global();
    let (value, keyval) = if arg.starts_with('{') {
        (json::from_str(arg).map_err(|e| fatal(&e))?, false)
    } else {
        let mut help = false;
        let args = keyval_parse(arg, Some("qom-type"), Some(&mut help)).map_err(|e| fatal(&e))?;
        if help {
            // user_creatable_print_help_from_qdict()
            let text = args
                .get_str("qom-type")
                .and_then(|ty| type_print_class_properties(registry, ty))
                .unwrap_or_else(|| user_creatable_print_types(registry));
            print!("{text}");
            return Err(Exit(0));
        }
        (QValue::Dict(args), true)
    };
    let mut v = if keyval {
        QObjectInputVisitor::new_keyval(value.clone())
    } else {
        QObjectInputVisitor::new(value.clone())
    };
    let mut options = ObjectOptions::default();
    ObjectOptions::visit(&mut v, None, &mut options).map_err(|e| fatal(&e))?;
    let QValue::Dict(dict) = value else {
        return Err(fatal(&Error::generic("Invalid parameter type, expected: object")));
    };
    registry.user_creatable_add(&dict, keyval).map_err(|e| fatal(&e))?;
    Ok(())
}

/// The cache flags of `bdrv_parse_cache_mode()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheMode {
    pub nocache: bool,
    pub no_flush: bool,
    pub writethrough: bool,
}

/// `BDRV_DEFAULT_CACHE`: data integrity does not matter much to qemu-img.
pub const BDRV_DEFAULT_CACHE: &str = "writeback";

/// `bdrv_parse_cache_mode()`.
pub fn parse_cache_mode(mode: &str) -> Option<CacheMode> {
    let (nocache, no_flush, writethrough) = match mode {
        "off" | "none" => (true, false, false),
        "directsync" => (true, false, true),
        "writeback" => (false, false, false),
        "unsafe" => (false, true, false),
        "writethrough" => (false, false, true),
        _ => return None,
    };
    Some(CacheMode { nocache, no_flush, writethrough })
}

impl CacheMode {
    /// Puts the cache flags into `flags`.
    pub fn apply(self, flags: &mut OpenFlags) {
        flags.nocache = self.nocache;
        flags.no_flush = self.no_flush;
    }
}

/// `qemu_source_opts`: `--image-opts` strings, with `file` as the implied key.
pub(crate) fn source_opts() -> QemuOptsList {
    QemuOptsList::new("source", &[]).with_implied_opt_name("file")
}

/// `qemu_opts_parse_noisily(qemu_find_opts("source"), optstr, true)` turned into a dict.
pub(crate) fn parse_source_opts(optstr: &str) -> Option<QDict> {
    let mut list = source_opts();
    list.parse_noisily(optstr, true).map(|o| o.to_qdict())
}

/// `img_open_opts()`.
pub(crate) fn img_open_opts(
    optstr: &str,
    mut options: QDict,
    flags: OpenFlags,
    writethrough: bool,
    force_share: bool,
) -> Option<Arc<BlockBackend>> {
    if force_share {
        if options.contains_key("force-share") && options.get_str("force-share") != Some("on") {
            error_report("--force-share/-U conflicts with image options");
            return None;
        }
        options.put("force-share", "on");
    }
    match graph().blk_new_open(None, options, flags) {
        Ok(blk) => {
            blk.set_enable_write_cache(!writethrough);
            Some(blk)
        }
        Err(e) => {
            report_error(&e.prepend(format!("Could not open '{optstr}': ")));
            None
        }
    }
}

/// `img_open_file()`.
pub(crate) fn img_open_file(
    filename: &str,
    options: Option<QDict>,
    fmt: Option<&str>,
    flags: OpenFlags,
    writethrough: bool,
    force_share: bool,
) -> Option<Arc<BlockBackend>> {
    let mut options = options.unwrap_or_default();
    if let Some(fmt) = fmt {
        options.put("driver", fmt);
    }
    if force_share {
        options.put("force-share", true);
    }
    match graph().blk_new_open(Some(filename), options, flags) {
        Ok(blk) => {
            blk.set_enable_write_cache(!writethrough);
            Some(blk)
        }
        Err(e) => {
            report_error(&e.prepend(format!("Could not open '{filename}': ")));
            None
        }
    }
}

/// `img_open()`.
pub(crate) fn img_open(
    image_opts: bool,
    filename: &str,
    fmt: Option<&str>,
    flags: OpenFlags,
    writethrough: bool,
    force_share: bool,
) -> Option<Arc<BlockBackend>> {
    if image_opts {
        if fmt.is_some() {
            error_report("--image-opts and --format are mutually exclusive");
            return None;
        }
        let opts = parse_source_opts(filename)?;
        img_open_opts(filename, opts, flags, writethrough, force_share)
    } else {
        img_open_file(filename, None, fmt, flags, writethrough, force_share)
    }
}

/// The node name at the root of `blk`, which every open backend has.
pub(crate) fn root(blk: &BlockBackend) -> String {
    blk.node_name().unwrap_or_default()
}

/// `qemu_opts_append()` of a description table onto a list.
pub(crate) fn opts_append(dst: Option<QemuOptsList>, desc: &[QemuOptDesc]) -> QemuOptsList {
    QemuOptsList::append(dst, &QemuOptsList::new("", desc))
}

/// `print_block_option_help()`: the exit status.
pub(crate) fn print_block_option_help(filename: Option<&str>, fmt: &str) -> i32 {
    if !tools::format_exists(fmt) {
        error_report(&format!("Unknown file format '{fmt}'"));
        return 1;
    }
    let Some(desc) = tools::create_opts_list(fmt) else {
        error_report(&format!("Format driver '{fmt}' does not support image creation"));
        return 1;
    };
    let mut create_opts = opts_append(None, desc);
    if let Some(filename) = filename {
        let proto = match tools::find_protocol(filename) {
            Ok(p) => p,
            Err(e) => {
                report_error(&e);
                return 1;
            }
        };
        let Some(pdesc) = tools::create_opts_list(proto) else {
            error_report(&format!("Protocol driver '{proto}' does not support image creation"));
            return 1;
        };
        create_opts = QemuOptsList::append(Some(create_opts), &QemuOptsList::new("", pdesc));
    }
    match filename {
        Some(_) => println!("Supported options:"),
        None => println!("Supported {fmt} options:"),
    }
    print!("{}", create_opts.help_text(false));
    if filename.is_none() {
        print!(
            "\nThe protocol level may support further options.\nSpecify the target filename to \
             include those options.\n"
        );
    }
    0
}

/// `has_help_option()` of an `-o` string.
pub(crate) fn has_help_option(options: &str) -> bool {
    ruvm_qapi::opts::has_help_option(options)
}

/// `bdrv_img_create()`.
#[allow(clippy::too_many_arguments, reason = "it is bdrv_img_create()")]
pub(crate) fn bdrv_img_create(
    filename: &str,
    fmt: &str,
    base_filename: Option<&str>,
    base_fmt: Option<&str>,
    options: Option<&str>,
    img_size: u64,
    no_backing: bool,
    quiet: bool,
) -> Result<(), Error> {
    if !tools::format_exists(fmt) {
        return Err(Error::generic(format!("Unknown file format '{fmt}'")));
    }
    let proto = tools::find_protocol(filename)?;
    let Some(desc) = tools::create_opts_list(fmt) else {
        return Err(Error::generic(format!(
            "Format driver '{fmt}' does not support image creation"
        )));
    };
    let Some(pdesc) = tools::create_opts_list(proto) else {
        return Err(Error::generic(format!(
            "Protocol driver '{proto}' does not support image creation"
        )));
    };
    // A driver whose create_opts are not ported yet has an empty list; take any option then
    // and leave the checking to the driver.
    let mut list = if desc.is_empty() {
        QemuOptsList::new("", &[])
    } else {
        opts_append(Some(opts_append(None, desc)), pdesc)
    };
    let handle = list.create(None, false)?.handle();
    let opts = list.get_mut(handle).expect("just made");
    if let Some(o) = options {
        opts.do_parse(o, None)?;
    }
    if opts.get("size").is_none() {
        opts.set_number("size", img_size as i64)?;
    } else if img_size != u64::MAX {
        return Err(Error::generic("The image size must be specified only once"));
    }
    let has = |name: &str| desc.is_empty() || desc.iter().any(|d| d.name == name);
    if let Some(b) = base_filename {
        if !has("backing_file") {
            return Err(Error::generic(format!(
                "Backing file not supported for file format '{fmt}'"
            )));
        }
        opts.set("backing_file", b)?;
    }
    if let Some(b) = base_fmt {
        if !has("backing_fmt") {
            return Err(Error::generic(format!(
                "Backing file format not supported for file format '{fmt}'"
            )));
        }
        opts.set("backing_fmt", b)?;
    }
    let backing_file = opts.get("backing_file").map(str::to_string);
    if let Some(b) = &backing_file {
        if b == filename {
            return Err(Error::generic(
                "Error: Trying to create an image with the same filename as the backing file",
            ));
        }
        if b.is_empty() {
            return Err(Error::generic("Expected backing file name, got empty string"));
        }
    }
    let backing_fmt = opts.get("backing_fmt").map(str::to_string);
    let mut size = opts.get_size("size", img_size) as i64;
    if let Some(backing_file) = backing_file.as_deref().filter(|_| !no_backing) {
        let full = tools::full_backing_filename_from_filename(filename, backing_file)?;
        let mut bo = QDict::new();
        if let Some(f) = &backing_fmt {
            bo.put("driver", f.as_str());
        }
        bo.put("force-share", true);
        let flags = OpenFlags { no_io: true, ..OpenFlags::default() };
        let blk = match graph().blk_new_open(Some(&full), bo, flags) {
            Ok(b) => b,
            Err(e) => return Err(e.hint("Could not open backing image.\n")),
        };
        if backing_fmt.is_none() {
            let drv = graph().node_details(&root(&blk)).map(|d| d.driver).unwrap_or_default();
            return Err(Error::generic("Backing file specified without backing format")
                .hint(format!("Detected format of {drv}.\n")));
        }
        if size == -1 {
            size = match blk.getlength() {
                Ok(s) => s as i64,
                Err(e) => {
                    return Err(Error::from_io(
                        format!("Could not get size of '{backing_file}'"),
                        e,
                    ));
                }
            };
            opts.set_number("size", size)?;
        }
    } else if backing_file.is_some() && backing_fmt.is_none() {
        return Err(Error::generic("Backing file specified without backing format"));
    }
    if size == -1 && !(fmt == "luks" && opts.get_bool("detached-header", false)) {
        return Err(Error::generic("Image creation needs a size parameter"));
    }
    if !quiet {
        println!("Formatting '{filename}', fmt={fmt} {}", opts.to_print_string(" "));
    }
    let mut dict = opts.to_qdict();
    let cluster_size = opts.get_size("cluster_size", 0);
    match graph().create_image(fmt, filename, &mut dict) {
        Ok(()) => Ok(()),
        Err(e) if e.message().ends_with("File too large") => {
            let hint = if cluster_size != 0 { " (try using a larger cluster size)" } else { "" };
            Err(Error::generic(format!(
                "The image size is too large for file format '{fmt}'{hint}"
            )))
        }
        Err(e) => Err(e),
    }
}

/// `localtime_r()` of `secs`: year, month (1 based), day, hour, minute, second.
#[cfg(unix)]
#[allow(unsafe_code, reason = "localtime_r() is the only way to the host time zone")]
pub fn localtime(secs: i64) -> (i32, i32, i32, i32, i32, i32) {
    let t: libc::time_t = secs as libc::time_t;
    // SAFETY: an all-zero `struct tm` is a valid value of a plain C struct.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: both pointers point to live locals of the right types for the whole call, and
    // localtime_r() is the thread safe variant.
    unsafe { libc::localtime_r(&t, &mut tm) };
    (tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec)
}

/// Windows has no `localtime_r()`; the dates are in UTC there.
#[cfg(not(unix))]
pub fn localtime(secs: i64) -> (i32, i32, i32, i32, i32, i32) {
    let days = secs.div_euclid(86400);
    let rem = secs.rem_euclid(86400);
    // Howard Hinnant's civil_from_days().
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y as i32, m as i32, d as i32, (rem / 3600) as i32, (rem / 60 % 60) as i32, (rem % 60) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_lists_accumulate() {
        let mut o = None;
        assert!(accumulate_options(&mut o, "a=1"));
        assert!(accumulate_options(&mut o, "b=2,,"));
        assert_eq!(o.as_deref(), Some("a=1,b=2,,"));
        assert!(!is_valid_option_list(""));
        assert!(!is_valid_option_list(",a"));
        assert!(!is_valid_option_list("a,"));
    }

    #[test]
    fn cache_modes() {
        assert!(parse_cache_mode("none").unwrap().nocache);
        assert!(parse_cache_mode("unsafe").unwrap().no_flush);
        assert!(parse_cache_mode("bogus").is_none());
    }

    #[test]
    fn numbers() {
        assert_eq!(cvtnum("size", "1k", true), Some(1024));
        assert_eq!(cvtnum_full("x", "17", false, 1, 16), None);
        assert_eq!(cvtnum_full("x", "0x10", false, 1, 16), Some(16));
    }
}

/// The values of the long options without a short one, as characters so they can be matched
/// next to the short options.
pub(crate) const OPTION_OUTPUT: char = '\u{100}';
pub(crate) const OPTION_BACKING_CHAIN: char = '\u{101}';
pub(crate) const OPTION_OBJECT: char = '\u{102}';
pub(crate) const OPTION_IMAGE_OPTS: char = '\u{103}';
pub(crate) const OPTION_PATTERN: char = '\u{104}';
pub(crate) const OPTION_FLUSH_INTERVAL: char = '\u{105}';
pub(crate) const OPTION_NO_DRAIN: char = '\u{106}';
pub(crate) const OPTION_TARGET_IMAGE_OPTS: char = '\u{107}';
pub(crate) const OPTION_PREALLOCATION: char = '\u{109}';
pub(crate) const OPTION_SHRINK: char = '\u{10a}';
pub(crate) const OPTION_SALVAGE: char = '\u{10b}';
pub(crate) const OPTION_TARGET_IS_ZERO: char = '\u{10c}';
pub(crate) const OPTION_ADD: char = '\u{10d}';
pub(crate) const OPTION_REMOVE: char = '\u{10e}';
pub(crate) const OPTION_CLEAR: char = '\u{10f}';
pub(crate) const OPTION_ENABLE: char = '\u{110}';
pub(crate) const OPTION_DISABLE: char = '\u{111}';
pub(crate) const OPTION_MERGE: char = '\u{112}';
pub(crate) const OPTION_BITMAPS: char = '\u{113}';
pub(crate) const OPTION_FORCE: char = '\u{114}';
pub(crate) const OPTION_SKIP_BROKEN: char = '\u{115}';
pub(crate) const OPTION_LIMITS: char = '\u{116}';
pub(crate) const OPTION_REMOVE_ALL: char = '\u{117}';

/// What `getopt()` returns for a non-option argument with a `-` option string.
pub(crate) const NONOPT: char = '\u{1}';

/// A long option table entry whose value is a character.
pub(crate) const fn lo(name: &'static str, has_arg: HasArg, val: char) -> LongOpt {
    lopt(name, has_arg, val as i32)
}

/// A `getopt_long()` loop of a command.
pub(crate) struct Opts<'a> {
    g: Getopt<'a>,
    /// `argv[0]`, "qemu-img <command>".
    pub argv0: String,
}

impl<'a> Opts<'a> {
    pub(crate) fn new(args: Vec<String>, shorts: &'a str, longs: &'a [LongOpt]) -> Self {
        let argv0 = args.first().cloned().unwrap_or_default();
        Opts { g: Getopt::new(args, shorts, longs), argv0 }
    }

    /// The next option and its argument (empty when it has none). An unknown option or a
    /// missing argument ends the program the way `default: tryhelp(argv[0])` does.
    pub(crate) fn next(&mut self) -> Flow<Option<(char, String)>> {
        match self.g.next() {
            None => Ok(None),
            Some(Opt::Opt(c, a)) => {
                Ok(Some((char::from_u32(c as u32).unwrap_or('?'), a.unwrap_or_default())))
            }
            Some(Opt::NonOpt(a)) => Ok(Some((NONOPT, a))),
            Some(Opt::Err) => Err(tryhelp(&self.argv0)),
        }
    }

    /// `argv[optind..]`.
    pub(crate) fn rest(&self) -> Vec<String> {
        self.g.rest().to_vec()
    }

    /// `argv[optind]`, the argument the loop looks at next.
    pub(crate) fn peek(&self) -> Option<String> {
        self.g.args().get(self.g.optind).cloned()
    }

    /// `++optind`.
    pub(crate) fn skip(&mut self) {
        self.g.optind += 1;
    }
}

/// `qprintf()`.
macro_rules! qprintf {
    ($quiet:expr, $($arg:tt)*) => {
        if !$quiet {
            print!($($arg)*);
        }
    };
}
pub(crate) use qprintf;

/// What `-l` of `convert` and `measure` names: `snapshot.id=...,snapshot.name=...` or a
/// plain id or name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SnapshotArg {
    Opts { id: Option<String>, name: Option<String> },
    IdOrName(String),
}

const INTERNAL_SNAPSHOT_OPTS: &[QemuOptDesc] = &[
    QemuOptDesc::new("snapshot.id", ruvm_qapi::opts::QemuOptType::String).help("snapshot id"),
    QemuOptDesc::new("snapshot.name", ruvm_qapi::opts::QemuOptType::String).help("snapshot name"),
];

/// Parses the `-l` argument; `None` after the error was reported.
pub(crate) fn parse_snapshot_arg(arg: &str) -> Option<SnapshotArg> {
    if !arg.starts_with("snapshot.") {
        return Some(SnapshotArg::IdOrName(arg.to_string()));
    }
    let mut list = QemuOptsList::new("snapshot", INTERNAL_SNAPSHOT_OPTS);
    let Some(opts) = list.parse_noisily(arg, false) else {
        error_report(&format!("Failed in parsing snapshot param '{arg}'"));
        return None;
    };
    Some(SnapshotArg::Opts {
        id: opts.get("snapshot.id").map(str::to_string),
        name: opts.get("snapshot.name").map(str::to_string),
    })
}

/// `bdrv_snapshot_load_tmp()` or `bdrv_snapshot_load_tmp_by_id_or_name()`, with the error
/// reported as "Failed to load snapshot: ". Returns whether it worked.
pub(crate) fn load_snapshot(node: &str, sn: &SnapshotArg) -> bool {
    let r = match sn {
        SnapshotArg::Opts { id, name } => {
            graph().snapshot_load_tmp(node, id.as_deref(), name.as_deref())
        }
        SnapshotArg::IdOrName(s) => graph().snapshot_load_tmp_by_id_or_name(node, s),
    };
    match r {
        Ok(()) => true,
        Err(e) => {
            report_error(&e.prepend("Failed to load snapshot: "));
            false
        }
    }
}
