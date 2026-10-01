// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img amend`.

use ruvm_base::Result;
use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools::{self, OpenFlags};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::{QemuOptDesc, QemuOptsList};

use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_FORCE, OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, accumulate_options,
    error_exit, graph, has_help_option, img_open, lo, object_add, opts_append, parse_cache_mode,
    root, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow, progress};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("options", HasArg::Required, 'o'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("cache", HasArg::Required, 't'),
    lo("force", HasArg::No, OPTION_FORCE),
    lo("progress", HasArg::No, 'p'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `print_amend_option_help()`: the exit status.
fn print_amend_option_help(format: &str) -> i32 {
    if !tools::format_exists(format) {
        error_report(&format!("Unknown file format '{format}'"));
        return 1;
    }
    let Some(desc) = tools::amend_opts_list(format) else {
        error_report(&format!("Format driver '{format}' does not support option amendment"));
        return 1;
    };
    println!("Amend options for '{format}':");
    print!("{}", QemuOptsList::new("", desc).help_text(false));
    0
}

/// `qemu_opts_do_parse()` of `options` against `list`, as a dict of strings.
fn parse(list: QemuOptsList, options: &str) -> Result<QDict> {
    let mut list = list;
    let handle = list.create(None, false)?.handle();
    let opts = list.get_mut(handle).expect("just made");
    opts.do_parse(options, None)?;
    Ok(opts.to_qdict())
}

/// Parses the `-o` options against the amend options of `fmt`. When that fails but the
/// create options would take them, the error says so.
fn parse_amend_options(
    fmt: &str,
    amend_desc: &'static [QemuOptDesc],
    options: &str,
) -> Result<QDict> {
    let amend = opts_append(None, amend_desc);
    match parse(amend.clone(), options) {
        Ok(d) => Ok(d),
        Err(e) => {
            let with_create = match tools::create_opts_list(fmt) {
                Some(c) => opts_append(Some(amend), c),
                None => amend,
            };
            if parse(with_create, options).is_ok() {
                return Err(e.hint("This option is only supported for image creation\n"));
            }
            Err(e)
        }
    }
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut options = None;
    let mut fmt = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut quiet = false;
    let mut progress_on = false;
    let mut image_opts = false;
    let mut force = false;
    let mut o = Opts::new(args, "ho:f:t:pq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "-o FMT_OPTS [-f FMT | --image-opts]\n",
                        "        [-t CACHE] [--force] [-p] [-q] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -o, --options FMT_OPTS\n",
                        "     FMT-specfic format options (required)\n",
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -t, --cache CACHE\n",
                        "     cache mode for FILE (default: writeback)\n",
                        "  --force\n",
                        "     allow certain unsafe operations\n",
                        "  -p, --progres\n",
                        "     show operation progress\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                    ),
                ));
            }
            'o' => {
                if !accumulate_options(&mut options, &arg) {
                    return Ok(1);
                }
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            't' => cache = arg,
            OPTION_FORCE => force = true,
            'p' => progress_on = true,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let Some(options) = options else {
        return Err(error_exit(&o.argv0, "Must specify options (-o)"));
    };
    if quiet {
        progress_on = false;
    }
    progress::init(progress_on, 1.0);
    let ret = amend(&o.rest(), &options, fmt.as_deref(), &cache, image_opts, force);
    progress::end();
    Ok(ret)
}

fn amend(
    rest: &[String],
    options: &str,
    fmt: Option<&str>,
    cache: &str,
    image_opts: bool,
    force: bool,
) -> i32 {
    if let Some(fmt) = fmt {
        if has_help_option(options) {
            return print_amend_option_help(fmt);
        }
    }
    if rest.len() != 1 {
        error_report("Expecting one image file name");
        return 1;
    }
    let filename = &rest[0];
    let Some(mode) = parse_cache_mode(cache) else {
        error_report(&format!("Invalid cache option: {cache}"));
        return 1;
    };
    let mut flags = OpenFlags { rdwr: true, ..OpenFlags::default() };
    mode.apply(&mut flags);
    let Some(blk) = img_open(image_opts, filename, fmt, flags, mode.writethrough, false) else {
        return 1;
    };
    let node = root(&blk);
    let fmt = graph().node_details(&node).map(|d| d.driver).unwrap_or_default();
    if has_help_option(options) {
        return print_amend_option_help(&fmt);
    }
    let Some(desc) = tools::amend_opts_list(&fmt) else {
        error_report(&format!("Format driver '{fmt}' does not support option amendment"));
        return 1;
    };
    let mut dict = match parse_amend_options(&fmt, desc, options) {
        Ok(d) => d,
        Err(e) => {
            report_error(&e);
            return 1;
        }
    };
    progress::print(0.0, 0);
    // amend_status_cb().
    let mut status = |offset: u64, total: u64| {
        progress::print(100.0 * offset as f32 / total as f32, 0);
    };
    let r = graph().amend_options_status(&node, &mut dict, &mut status, force);
    progress::print(100.0, 0);
    match r {
        Ok(()) => 0,
        Err(e) => {
            report_error(&e);
            1
        }
    }
}
