// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img measure`.

use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools;

use crate::common::{
    OPTION_IMAGE_OPTS, OPTION_OBJECT, OPTION_OUTPUT, Opts, OutputFormat, SnapshotArg,
    accumulate_options, cvtnum, graph, img_open, lo, load_snapshot, object_add, opts_append,
    parse_output_format, parse_snapshot_arg, root, tryhelp,
};
use crate::dump::to_json_pretty;
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("source-format", HasArg::Required, 'f'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("source-image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("snapshot", HasArg::Required, 'l'),
    lo("target-format", HasArg::Required, 'O'),
    lo("target-format-options", HasArg::Required, 'o'),
    lo("options", HasArg::Required, 'o'),
    lo("force-share", HasArg::No, 'U'),
    lo("output", HasArg::Required, OPTION_OUTPUT),
    lo("object", HasArg::Required, OPTION_OBJECT),
    lo("size", HasArg::Required, 's'),
];

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut output_format = OutputFormat::Human;
    let mut fmt = None;
    let mut out_fmt = "raw".to_string();
    let mut options = None;
    let mut snapshot: Option<SnapshotArg> = None;
    let mut force_share = false;
    let mut image_opts = false;
    let mut img_size = None;
    let mut o = Opts::new(args, "hf:l:O:o:Us:", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT|--image-opts] [-l SNAPSHOT]\n",
                        "       [-O TARGET_FMT] [-o TARGET_FMT_OPTS] [--output human|json]\n",
                        "       [--object OBJDEF] (--size SIZE | FILE)\n",
                    ),
                    concat!(
                        "  -f, --format\n",
                        "     specify format of FILE explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     indicates that FILE is a complete image specification\n",
                        "     instead of a file name (incompatible with --format)\n",
                        "  -l, --snapshot SNAPSHOT\n",
                        "     use this snapshot in FILE as source\n",
                        "  -O, --target-format TARGET_FMT\n",
                        "     desired target/output image format (default: raw)\n",
                        "  -o TARGET_FMT_OPTS\n",
                        "     options specific to TARGET_FMT\n",
                        "  --output human|json\n",
                        "     output format (default: human)\n",
                        "  -U, --force-share\n",
                        "     open images in shared mode for concurrent access\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  -s, --size SIZE[bKMGTPE]\n",
                        "     measure file size for given image size,\n",
                        "     with optional multiplier suffix (powers of 1024)\n",
                        "  FILE\n",
                        "     measure file size required to convert from FILE (either a file \
                         name\n",
                        "     or an option string (key=value,..) with --image-options)\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            'l' => match parse_snapshot_arg(&arg) {
                Some(s) => snapshot = Some(s),
                None => return Ok(1),
            },
            'O' => out_fmt = arg,
            'o' => {
                if !accumulate_options(&mut options, &arg) {
                    return Ok(1);
                }
            }
            'U' => force_share = true,
            OPTION_OUTPUT => output_format = parse_output_format(&o.argv0, &arg)?,
            OPTION_OBJECT => object_add(&arg)?,
            's' => match cvtnum("image size", &arg, true) {
                Some(v) => img_size = Some(v),
                None => return Ok(1),
            },
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    if rest.len() > 1 {
        error_report("At most one filename argument is allowed.");
        return Ok(1);
    }
    let filename = rest.first();
    if filename.is_none() && (image_opts || fmt.is_some() || snapshot.is_some()) {
        error_report("--image-opts, -f, and -l require a filename argument.");
        return Ok(1);
    }
    if filename.is_some() && img_size.is_some() {
        error_report("--size N cannot be used together with a filename.");
        return Ok(1);
    }
    if filename.is_none() && img_size.is_none() {
        error_report("Either --size N or one filename must be specified.");
        return Ok(1);
    }

    let mut in_blk = None;
    if let Some(filename) = filename {
        let Some(blk) =
            img_open(image_opts, filename, fmt.as_deref(), Default::default(), false, force_share)
        else {
            return Ok(1);
        };
        if let Some(sn) = &snapshot {
            if !load_snapshot(&root(&blk), sn) {
                return Ok(1);
            }
        }
        in_blk = Some(blk);
    }

    if !tools::format_exists(&out_fmt) {
        error_report(&format!("Unknown file format '{out_fmt}'"));
        return Ok(1);
    }
    let Some(desc) = tools::create_opts_list(&out_fmt) else {
        error_report(&format!("Format driver '{out_fmt}' does not support image creation"));
        return Ok(1);
    };
    let file_desc = tools::create_opts_list("file").unwrap_or(&[]);
    let mut list = opts_append(Some(opts_append(None, desc)), file_desc);
    let handle = match list.create(None, false) {
        Ok(o) => o.handle(),
        Err(e) => {
            report_error(&e);
            return Ok(1);
        }
    };
    let opts = list.get_mut(handle).expect("just made");
    if let Some(options) = &options {
        if let Err(e) = opts.do_parse(options, None) {
            report_error(&e);
            error_report(&format!("Invalid options for file format '{out_fmt}'"));
            return Ok(1);
        }
    }
    if let Some(size) = img_size {
        if let Err(e) = opts.set_number("size", size) {
            report_error(&e);
            return Ok(1);
        }
    }
    let mut dict = opts.to_qdict();
    let in_node = in_blk.as_ref().map(|b| root(b));
    let info = match graph().measure(&out_fmt, &mut dict, in_node.as_deref()) {
        Ok(i) => i,
        Err(e) => {
            report_error(&e);
            return Ok(1);
        }
    };
    match output_format {
        OutputFormat::Human => {
            println!("required size: {}", info.required);
            println!("fully allocated size: {}", info.fully_allocated);
            if let Some(b) = info.bitmaps {
                println!("bitmaps size: {b}");
            }
        }
        OutputFormat::Json => println!("{}", to_json_pretty(&info)),
    }
    Ok(0)
}
