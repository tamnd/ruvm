// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img info`.

use std::collections::HashSet;

use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools::OpenFlags;
use ruvm_qapi::types::BlockGraphInfo;
use ruvm_qapi::{QValue, json};

use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_BACKING_CHAIN, OPTION_IMAGE_OPTS, OPTION_LIMITS, OPTION_OBJECT,
    OPTION_OUTPUT, Opts, OutputFormat, error_exit, graph, img_open, lo, object_add,
    parse_cache_mode, parse_output_format, root, tryhelp,
};
use crate::dump::{human_image_info, to_json_pretty, to_qobject};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("backing-chain", HasArg::No, OPTION_BACKING_CHAIN),
    lo("cache", HasArg::Required, 't'),
    lo("force-share", HasArg::No, 'U'),
    lo("limits", HasArg::No, OPTION_LIMITS),
    lo("output", HasArg::Required, OPTION_OUTPUT),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `collect_image_info_list()`.
fn collect_image_info_list(
    mut image_opts: bool,
    filename: &str,
    fmt: Option<&str>,
    cache: &str,
    chain: bool,
    limits: bool,
    force_share: bool,
) -> Option<Vec<BlockGraphInfo>> {
    let Some(mode) = parse_cache_mode(cache) else {
        error_report(&format!("Invalid cache option: {cache}"));
        return None;
    };
    let mut flags = OpenFlags { no_backing: true, no_io: true, ..OpenFlags::default() };
    mode.apply(&mut flags);
    let mut filenames = HashSet::new();
    let mut list = Vec::new();
    let mut filename = Some(filename.to_string());
    let mut fmt = fmt.map(str::to_string);
    while let Some(f) = filename.take() {
        if !filenames.insert(f.clone()) {
            error_report(&format!("Backing file '{f}' creates an infinite loop."));
            return None;
        }
        let blk = img_open(image_opts, &f, fmt.as_deref(), flags, mode.writethrough, force_share)?;
        let info = match graph().query_block_graph_info(&root(&blk), limits) {
            Ok(i) => i,
            Err(e) => {
                report_error(&e);
                return None;
            }
        };
        drop(blk);
        fmt = None;
        image_opts = false;
        if chain {
            if let Some(full) = &info.full_backing_filename {
                filename = Some(full.clone());
            } else if let Some(b) = &info.backing_filename {
                error_report(&format!(
                    "Could not determine absolute backing filename, but backing filename '{b}' \
                     present"
                ));
                return None;
            }
            if let Some(bf) = &info.backing_filename_format {
                fmt = Some(bf.clone());
            }
        }
        list.push(info);
    }
    Some(list)
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut output_format = OutputFormat::Human;
    let mut chain = false;
    let mut fmt = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut image_opts = false;
    let mut force_share = false;
    let mut limits = false;
    let mut o = Opts::new(args, "hf:t:U", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [--backing-chain] [-U]\n",
                        "        [--output human|json] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE image format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  --backing-chain\n",
                        "     display information about the backing chain for copy-on-write overlays\n",
                        "  -t, --cache CACHE\n",
                        "     cache mode for FILE (default: writeback)\n",
                        "  -U, --force-share\n",
                        "     open image in shared mode for concurrent access\n",
                        "  --limits\n",
                        "     show detected block limits (may depend on options, e.g. cache mode)\n",
                        "  --output human|json\n",
                        "     specify output format (default: human)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            OPTION_BACKING_CHAIN => chain = true,
            't' => cache = arg,
            'U' => force_share = true,
            OPTION_LIMITS => limits = true,
            OPTION_OUTPUT => output_format = parse_output_format(&o.argv0, &arg)?,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let Some(list) = collect_image_info_list(
        image_opts,
        &rest[0],
        fmt.as_deref(),
        &cache,
        chain,
        limits,
        force_share,
    ) else {
        return Ok(1);
    };
    match output_format {
        OutputFormat::Human => {
            let mut out = String::new();
            for (i, info) in list.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                human_image_info(&mut out, info, 0, "/");
            }
            print!("{out}");
        }
        OutputFormat::Json => {
            if chain {
                let v = QValue::List(list.iter().map(to_qobject).collect());
                println!("{}", json::to_string(&v, true));
            } else {
                println!("{}", to_json_pretty(&list[0]));
            }
        }
    }
    Ok(0)
}
