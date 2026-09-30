// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img dd`: copies an image, or a part of it, into a new image, with operands in the
//! style of dd(1).

use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools::{self, OpenFlags};
use ruvm_qapi::opts::QemuOptsList;

use crate::common::{
    OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, cvtnum, cvtnum_full, graph, img_open, img_open_file,
    lo, object_add, opts_append, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("output-format", HasArg::Required, 'O'),
    lo("force-share", HasArg::No, 'U'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut out_fmt = "raw".to_string();
    let mut image_opts = false;
    let mut force_share = false;
    let mut o = Opts::new(args, "hf:O:U", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT|--image-opts] [-O OUTPUT_FMT] [-U]\n",
                        "        [--object OBJDEF] [bs=BLOCK_SIZE] [count=BLOCKS] if=INPUT \
                         of=OUTPUT\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify format for INPUT explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat INPUT as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -O, --output-format OUTPUT_FMT\n",
                        "     format of the OUTPUT (default: raw)\n",
                        "  -U, --force-share\n",
                        "     open images in shared mode for concurrent access\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  bs=BLOCK_SIZE[bKMGTP]\n",
                        "     size of the I/O block, with optional multiplier suffix (powers \
                         of 1024)\n",
                        "     (default: 512)\n",
                        "  count=COUNT\n",
                        "     number of blocks to convert (default whole INPUT)\n",
                        "  if=INPUT\n",
                        "     name of the file, or option string (key=value,..)\n",
                        "     with --image-opts, to use for input\n",
                        "  of=OUTPUT\n",
                        "     output file name to create (will be overridden if alrady \
                         exists)\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            'O' => out_fmt = arg,
            'U' => force_share = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }

    // The operands: bs=, count=, if=, of= and skip=.
    let mut bsz = 512i64;
    let mut count = None;
    let mut in_filename = None;
    let mut out_filename = None;
    let mut skip = None;
    for arg in o.rest() {
        let Some((name, value)) = arg.split_once('=') else {
            error_report(&format!("unrecognized operand {arg}"));
            return Ok(1);
        };
        let ok = match name {
            "bs" => cvtnum_full("bs", value, true, 1, i64::from(i32::MAX)).map(|v| bsz = v),
            "count" => cvtnum("count", value, true).map(|v| count = Some(v)),
            "if" => {
                in_filename = Some(value.to_string());
                Some(())
            }
            "of" => {
                out_filename = Some(value.to_string());
                Some(())
            }
            "skip" => cvtnum("skip", value, true).map(|v| skip = Some(v)),
            _ => {
                error_report(&format!("unrecognized operand {name}"));
                return Ok(1);
            }
        };
        if ok.is_none() {
            return Ok(1);
        }
    }
    let (Some(in_filename), Some(out_filename)) = (in_filename, out_filename) else {
        error_report("Must specify both input and output files");
        return Ok(1);
    };
    Ok(dd(
        &in_filename,
        &out_filename,
        fmt.as_deref(),
        &out_fmt,
        image_opts,
        force_share,
        bsz,
        count,
        skip,
    ))
}

#[allow(clippy::too_many_arguments, reason = "the locals of img_dd()")]
fn dd(
    in_filename: &str,
    out_filename: &str,
    fmt: Option<&str>,
    out_fmt: &str,
    image_opts: bool,
    force_share: bool,
    bsz: i64,
    count: Option<i64>,
    skip: Option<i64>,
) -> i32 {
    let Some(blk1) =
        img_open(image_opts, in_filename, fmt, OpenFlags::default(), false, force_share)
    else {
        return 1;
    };

    if !tools::format_exists(out_fmt) {
        error_report("Unknown file format");
        return 1;
    }
    let proto = match tools::find_protocol(out_filename) {
        Ok(p) => p,
        Err(e) => {
            report_error(&e);
            return 1;
        }
    };
    let Some(desc) = tools::create_opts_list(out_fmt) else {
        error_report(&format!("Format driver '{out_fmt}' does not support image creation"));
        return 1;
    };
    let Some(pdesc) = tools::create_opts_list(proto) else {
        error_report(&format!("Protocol driver '{proto}' does not support image creation"));
        return 1;
    };
    let mut list: QemuOptsList = opts_append(Some(opts_append(None, desc)), pdesc);
    let mut opts = list.create(None, false).expect("an unnamed list takes new options").clone();

    let Ok(size) = blk1.getlength() else {
        error_report(&format!("Failed to get size for '{in_filename}'"));
        return 1;
    };
    let mut size = size as i64;
    if let Some(count) = count {
        if count <= i64::MAX / bsz && count * bsz < size {
            size = count * bsz;
        }
    }

    // An offset that overflows is beyond the end of the input.
    let skip_too_far = skip.is_some_and(|off| off > i64::MAX / bsz || size < bsz * off);
    let offset = skip.unwrap_or(0);
    let out_size = if skip_too_far { 0 } else { size - bsz * offset };
    opts.set_number("size", out_size).expect("every format has size");

    let mut dict = opts.to_qdict();
    if let Err(e) = graph().create_image(out_fmt, out_filename, &mut dict) {
        report_error(&e.prepend(format!("{out_filename}: error while creating output image: ")));
        return 1;
    }

    // --image-opts does not work for the output: it has to be in the form bdrv_create()
    // takes, and that has no image options.
    let flags = OpenFlags { rdwr: true, ..OpenFlags::default() };
    let Some(blk2) = img_open_file(out_filename, None, Some(out_fmt), flags, false, false) else {
        return 1;
    };

    let mut in_pos = if skip_too_far {
        // Like dd(1), an offset beyond the input gives a warning and an empty output.
        error_report(&format!("{in_filename}: cannot skip to specified offset"));
        size
    } else {
        offset * bsz
    };

    let mut buf = vec![0u8; bsz as usize];
    let mut out_pos = 0i64;
    while in_pos < size {
        let bytes = if in_pos + bsz > size { size - in_pos } else { bsz } as usize;
        if let Err(e) = blk1.pread(in_pos as u64, &mut buf[..bytes]) {
            error_report(&format!("error while reading from input image file: {}", strerror(&e)));
            return 1;
        }
        in_pos += bytes as i64;
        if let Err(e) = blk2.pwrite(out_pos as u64, &buf[..bytes]) {
            error_report(&format!("error while writing to output image file: {}", strerror(&e)));
            return 1;
        }
        out_pos += bytes as i64;
    }
    0
}
