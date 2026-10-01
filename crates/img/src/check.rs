// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img check`.

use ruvm_base::Error;
use ruvm_base::report::error_report;
use ruvm_block::tools::OpenFlags;
use ruvm_block::{BDRV_FIX_ERRORS, BDRV_FIX_LEAKS};
use ruvm_qapi::types::ImageCheck;

use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_IMAGE_OPTS, OPTION_OBJECT, OPTION_OUTPUT, Opts, OutputFormat,
    error_exit, graph, img_open, lo, object_add, parse_cache_mode, parse_output_format, qprintf,
    root, tryhelp,
};
use crate::dump::to_json_pretty;
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("cache", HasArg::Required, 'T'),
    lo("repair", HasArg::Required, 'r'),
    lo("force-share", HasArg::No, 'U'),
    lo("output", HasArg::Required, OPTION_OUTPUT),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// The message [`ruvm_block::BlockGraph::check`] fails with for `-ENOTSUP`.
const NOT_SUPPORTED: &str = "This image format does not support checks";

/// `dump_human_image_check()`.
fn dump_human_image_check(check: &ImageCheck, quiet: bool) {
    let corruptions = check.corruptions.unwrap_or(0);
    let leaks = check.leaks.unwrap_or(0);
    if corruptions == 0 && leaks == 0 && check.check_errors == 0 {
        qprintf!(quiet, "No errors were found on the image.\n");
    } else {
        if corruptions != 0 {
            qprintf!(
                quiet,
                "\n{corruptions} errors were found on the image.\nData may be corrupted, or \
                 further writes to the image may corrupt it.\n"
            );
        }
        if leaks != 0 {
            qprintf!(
                quiet,
                "\n{leaks} leaked clusters were found on the image.\nThis means waste of disk \
                 space, but no harm to data.\n"
            );
        }
        if check.check_errors != 0 {
            qprintf!(
                quiet,
                "\n{} internal errors have occurred during the check.\n",
                check.check_errors
            );
        }
    }
    let total = check.total_clusters.unwrap_or(0);
    let allocated = check.allocated_clusters.unwrap_or(0);
    if total != 0 && allocated != 0 {
        let fragmented = check.fragmented_clusters.unwrap_or(0);
        let compressed = check.compressed_clusters.unwrap_or(0);
        qprintf!(
            quiet,
            "{allocated}/{total} = {:.2}% allocated, {:.2}% fragmented, {:.2}% compressed \
             clusters\n",
            allocated as f64 * 100.0 / total as f64,
            fragmented as f64 * 100.0 / allocated as f64,
            compressed as f64 * 100.0 / allocated as f64
        );
    }
    if let Some(end) = check.image_end_offset {
        qprintf!(quiet, "Image end offset: {end}\n");
    }
}

fn nonzero(v: i64) -> Option<i64> {
    (v != 0).then_some(v)
}

/// `collect_image_check()`.
fn collect_image_check(node: &str, filename: &str, fix: u32) -> Result<ImageCheck, Error> {
    let r = graph().check(node, fix)?;
    let format = graph().node_details(node).map(|d| d.driver).unwrap_or_default();
    Ok(ImageCheck {
        filename: filename.to_string(),
        format,
        check_errors: r.check_errors,
        image_end_offset: nonzero(r.image_end_offset),
        corruptions: nonzero(r.corruptions),
        leaks: nonzero(r.leaks),
        corruptions_fixed: nonzero(r.corruptions_fixed),
        leaks_fixed: nonzero(r.leaks_fixed),
        total_clusters: nonzero(r.bfi.total_clusters as i64),
        allocated_clusters: nonzero(r.bfi.allocated_clusters as i64),
        fragmented_clusters: nonzero(r.bfi.fragmented_clusters as i64),
        compressed_clusters: nonzero(r.bfi.compressed_clusters as i64),
    })
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut output_format = OutputFormat::Human;
    let mut fmt = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut fix = 0;
    let mut flags = OpenFlags { check: true, ..OpenFlags::default() };
    let mut quiet = false;
    let mut image_opts = false;
    let mut force_share = false;
    let mut o = Opts::new(args, "hf:T:r:Uq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [-T CACHE_MODE] [-r leaks|all]\n",
                        "        [-U] [--output human|json] [-q] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specifies the format of the image explicitly (default: probing is \
                         used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -T, --cache CACHE_MODE\n",
                        "     cache mode (default: writeback)\n",
                        "  -r, --repair leaks|all\n",
                        "     repair errors of the given category in the image (image will be\n",
                        "     opened in read-write mode, incompatible with -U|--force-share)\n",
                        "  -U, --force-share\n",
                        "     open image in shared mode for concurrent access\n",
                        "  --output human|json\n",
                        "     output format (default: human)\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or an option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            'T' => cache = arg,
            'r' => {
                flags.rdwr = true;
                fix = match arg.as_str() {
                    "leaks" => BDRV_FIX_LEAKS,
                    "all" => BDRV_FIX_LEAKS | BDRV_FIX_ERRORS,
                    _ => {
                        return Err(error_exit(
                            &o.argv0,
                            &format!("--repair (-r) expects 'leaks' or 'all', not '{arg}'"),
                        ));
                    }
                };
            }
            'U' => force_share = true,
            OPTION_OUTPUT => output_format = parse_output_format(&o.argv0, &arg)?,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let filename = &rest[0];
    let Some(mode) = parse_cache_mode(&cache) else {
        error_report(&format!("Invalid source cache option: {cache}"));
        return Ok(1);
    };
    mode.apply(&mut flags);
    let Some(blk) =
        img_open(image_opts, filename, fmt.as_deref(), flags, mode.writethrough, force_share)
    else {
        return Ok(1);
    };
    let node = root(&blk);

    let mut result = collect_image_check(&node, filename, fix);
    if let Err(e) = &result {
        if e.to_string() == NOT_SUPPORTED {
            error_report(NOT_SUPPORTED);
            return Ok(63);
        }
    }
    if let Ok(check) = &result {
        if check.corruptions_fixed.is_some() || check.leaks_fixed.is_some() {
            let leaks_fixed = check.leaks_fixed;
            let corruptions_fixed = check.corruptions_fixed;
            if output_format == OutputFormat::Human {
                qprintf!(
                    quiet,
                    "The following inconsistencies were found and repaired:\n\n    {} leaked \
                     clusters\n    {} corruptions\n\nDouble checking the fixed image now...\n",
                    leaks_fixed.unwrap_or(0),
                    corruptions_fixed.unwrap_or(0)
                );
            }
            result = collect_image_check(&node, filename, 0).map(|mut c| {
                c.leaks_fixed = leaks_fixed;
                c.corruptions_fixed = corruptions_fixed;
                c
            });
        }
    }

    match result {
        Err(e) => {
            error_report(&format!("Check failed: {e}"));
            Ok(1)
        }
        Ok(check) => {
            match output_format {
                OutputFormat::Human => dump_human_image_check(&check, quiet),
                OutputFormat::Json => qprintf!(quiet, "{}\n", to_json_pretty(&check)),
            }
            if check.check_errors != 0 {
                error_report("Check failed");
                Ok(1)
            } else if check.corruptions.is_some() {
                Ok(2)
            } else if check.leaks.is_some() {
                Ok(3)
            } else {
                Ok(0)
            }
        }
    }
}
