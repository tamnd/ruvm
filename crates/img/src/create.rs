// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img create`.

use ruvm_base::report::report_error;

use crate::common::{
    OPTION_OBJECT, Opts, accumulate_options, bdrv_img_create, cvtnum, error_exit, has_help_option,
    lo, object_add, print_block_option_help,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("options", HasArg::Required, 'o'),
    lo("backing", HasArg::Required, 'b'),
    lo("backing-format", HasArg::Required, 'B'),
    lo("backing-unsafe", HasArg::No, 'u'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut img_size = u64::MAX;
    let mut fmt = "raw".to_string();
    let mut base_fmt = None;
    let mut base_filename = None;
    let mut options = None;
    let mut quiet = false;
    let mut no_backing = false;
    let mut o = Opts::new(args, "hf:o:b:F:B:uq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    "[-f FMT] [-o FMT_OPTS]\n        [-b BACKING_FILE [-B BACKING_FMT]] [-u]\n        \
                     [-q] [--object OBJDEF] FILE [SIZE]\n",
                    "  -f, --format FMT\n     specifies the format of the new image (default: \
                     raw)\n  -o, --options FMT_OPTS\n     format-specific options (specify '-o \
                     help' for help)\n  -b, --backing BACKING_FILE\n     create target image to \
                     be a CoW on top of BACKING_FILE\n  -B, --backing-format BACKING_FMT (was -F \
                     in <= 10.0)\n     specifies the format of BACKING_FILE (default: probing is \
                     used)\n  -u, --backing-unsafe\n     do not fail if BACKING_FILE can not be \
                     read\n  -q, --quiet\n     quiet mode (produce only error messages if any)\n  \
                     --object OBJDEF\n     defines QEMU user-creatable object\n  FILE\n     name \
                     of the image file to create (will be overritten if already exists)\n  \
                     SIZE[bKMGTPE]\n     image size with optional multiplier suffix (powers of \
                     1024)\n     (required unless BACKING_FILE is specified)\n",
                ));
            }
            'f' => fmt = arg,
            'o' => {
                if !accumulate_options(&mut options, &arg) {
                    return Ok(1);
                }
            }
            'b' => base_filename = Some(arg),
            'F' | 'B' => base_fmt = Some(arg),
            'u' => no_backing = true,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(crate::common::tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    let filename = rest.first().cloned();
    if options.as_deref().is_some_and(has_help_option) {
        return Ok(print_block_option_help(filename.as_deref(), &fmt));
    }
    let Some(filename) = filename else {
        return Err(error_exit(&o.argv0, "Expecting image file name"));
    };
    let mut i = 1;
    if let Some(s) = rest.get(i) {
        i += 1;
        match cvtnum("image size", s, true) {
            Some(v) => img_size = v as u64,
            None => return Ok(1),
        }
    }
    if let Some(extra) = rest.get(i) {
        return Err(error_exit(&o.argv0, &format!("Unexpected argument: {extra}")));
    }
    if let Err(e) = bdrv_img_create(
        &filename,
        &fmt,
        base_filename.as_deref(),
        base_fmt.as_deref(),
        options.as_deref(),
        img_size,
        no_backing,
        quiet,
    ) {
        report_error(&e.prepend(format!("{filename}: ")));
        return Ok(1);
    }
    Ok(0)
}
