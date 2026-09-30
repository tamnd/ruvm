// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img resize`.

use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error, warn_report};
use ruvm_block::tools::OpenFlags;
use ruvm_qapi::types::PreallocMode;
use ruvm_qapi::visit::parse_option_size;

use crate::common::{
    NONOPT, OPTION_IMAGE_OPTS, OPTION_OBJECT, OPTION_PREALLOCATION, OPTION_SHRINK, Opts,
    error_exit, img_open, lo, object_add, qprintf, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("preallocation", HasArg::Required, OPTION_PREALLOCATION),
    lo("shrink", HasArg::No, OPTION_SHRINK),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut filename: Option<String> = None;
    let mut fmt = None;
    let mut size: Option<String> = None;
    let mut quiet = false;
    let mut prealloc = PreallocMode::Off;
    let mut image_opts = false;
    let mut shrink = false;
    let mut o = Opts::new(args, "-hf:q", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [--preallocation PREALLOC] [--shrink]\n",
                        "        [-q] [--object OBJDEF] FILE [+-]SIZE[bkKMGTPE]\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,...), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  --shrink\n",
                        "     allow operation when the new size is smaller than the original\n",
                        "  --preallocation PREALLOC\n",
                        "     specify FMT-specific preallocation type for the new areas\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                        "  [+-]SIZE[bkKMGTPE]\n",
                        "     new image size or amount by which to shrink (-)/grow (+),\n",
                        "     with optional multiplier suffix (powers of 1024, default is bytes)\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            OPTION_PREALLOCATION => match PreallocMode::from_name(&arg) {
                Some(p) => prealloc = p,
                None => {
                    error_report(&format!("Invalid preallocation mode '{arg}'"));
                    return Ok(1);
                }
            },
            OPTION_SHRINK => shrink = true,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            NONOPT => {
                if filename.is_none() {
                    filename = Some(arg);
                    // A negative size right after the file name looks like an option.
                    if let Some(next) = o.peek() {
                        let b = next.as_bytes();
                        if b.len() >= 2 && b[0] == b'-' && b[1].is_ascii_digit() {
                            o.skip();
                            size = Some(next);
                        }
                    }
                } else if size.is_none() {
                    size = Some(arg);
                } else {
                    return Err(error_exit(&o.argv0, "Extra argument(s) in command line"));
                }
            }
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let mut rest = o.rest().into_iter();
    if filename.is_none() {
        filename = rest.next();
    }
    if size.is_none() {
        size = rest.next();
    }
    let (Some(filename), Some(size)) = (filename, size) else {
        return Err(error_exit(&o.argv0, "Expecting image file name and size"));
    };
    if rest.next().is_some() {
        return Err(error_exit(&o.argv0, "Expecting image file name and size"));
    }

    let (relative, size) = match size.as_bytes().first() {
        Some(b'+') => (1i64, &size[1..]),
        Some(b'-') => (-1, &size[1..]),
        _ => (0, &size[..]),
    };
    let n = match parse_option_size("size", size) {
        Ok(n) => n as i64,
        Err(e) => {
            report_error(&e);
            return Ok(1);
        }
    };

    let flags = OpenFlags { rdwr: true, resize: true, ..OpenFlags::default() };
    let Some(blk) = img_open(image_opts, &filename, fmt.as_deref(), flags, false, false) else {
        return Ok(1);
    };
    let current_size = match blk.getlength() {
        Ok(l) => l as i64,
        Err(e) => {
            error_report(&format!("Failed to inquire current image length: {}", strerror(&e)));
            return Ok(1);
        }
    };
    let total_size =
        if relative != 0 { current_size.wrapping_add(n.wrapping_mul(relative)) } else { n };
    if total_size <= 0 {
        error_report("New image size must be positive");
        return Ok(1);
    }
    if total_size <= current_size && prealloc != PreallocMode::Off {
        error_report("Preallocation can only be used for growing images");
        return Ok(1);
    }
    if total_size < current_size && !shrink {
        error_report("Use the --shrink option to perform a shrink operation.");
        warn_report(
            "Shrinking an image will delete all data beyond the shrunken image's end. Before \
             performing such an operation, make sure there is no important data there.",
        );
        return Ok(1);
    }
    match blk.truncate_full(total_size as u64, true, prealloc) {
        Ok(()) => {
            qprintf!(quiet, "Image resized.\n");
            Ok(0)
        }
        Err(e) => {
            report_error(&e);
            Ok(1)
        }
    }
}
