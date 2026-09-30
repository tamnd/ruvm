// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img snapshot`.

use ruvm_base::Error;
use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools::OpenFlags;

use crate::common::{
    OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, error_exit, graph, img_open, lo, object_add, root,
    tryhelp,
};
use crate::dump::snapshot_dump;
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("list", HasArg::No, 'l'),
    lo("apply", HasArg::Required, 'a'),
    lo("create", HasArg::Required, 'c'),
    lo("delete", HasArg::Required, 'd'),
    lo("force-share", HasArg::No, 'U'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// The `strerror()` text of the errno an error carries, or its message when it has none.
pub(crate) fn errno_text(e: &Error) -> String {
    let mut src: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(e);
    while let Some(s) = src {
        if let Some(io) = s.downcast_ref::<std::io::Error>() {
            return strerror(io);
        }
        src = s.source();
    }
    e.to_string()
}

/// `dump_snapshots()`.
pub(crate) fn dump_snapshots(node: &str) -> String {
    let mut out = String::new();
    let list = match graph().snapshot_list(node) {
        Ok(Some(l)) if !l.is_empty() => l,
        _ => return out,
    };
    out.push_str(&format!("Snapshot list:\n{}\n", snapshot_dump(None)));
    for sn in &list {
        out.push_str(&format!("{}\n", snapshot_dump(Some(sn))));
    }
    out
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut snapshot_name = String::new();
    let mut action = None;
    let mut quiet = false;
    let mut image_opts = false;
    let mut force_share = false;
    let mut o = Opts::new(args, "hf:la:c:d:Uq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [-l | -a|-c|-d SNAPSHOT]\n",
                        "        [-U] [-q] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -l, --list\n",
                        "     list snapshots in FILE (default action if no -l|-c|-a|-d is \
                         given)\n",
                        "  -c, --create SNAPSHOT\n",
                        "     create named snapshot\n",
                        "  -a, --apply SNAPSHOT\n",
                        "     apply named snapshot to the base\n",
                        "  -d, --delete SNAPSHOT\n",
                        "     delete named snapshot\n",
                        "  (only one of -l|-c|-a|-d can be specified)\n",
                        "  -U, --force-share\n",
                        "     open image in shared mode for concurrent access\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or option string (key=value,..)\n",
                        "     with --image-opts) to operate on\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            'l' | 'a' | 'c' | 'd' => {
                if action.is_some() {
                    return Err(error_exit(&o.argv0, "Cannot mix '-l', '-a', '-c', '-d'"));
                }
                action = Some(c);
                snapshot_name = arg;
            }
            'U' => force_share = true,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let _ = quiet;
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let filename = &rest[0];
    let action = action.unwrap_or('l');

    let flags = OpenFlags { rdwr: action != 'l', ..OpenFlags::default() };
    let Some(blk) = img_open(image_opts, filename, fmt.as_deref(), flags, false, force_share)
    else {
        return Ok(1);
    };
    let node = root(&blk);
    let g = graph();
    let ok = match action {
        'l' => {
            print!("{}", dump_snapshots(&node));
            true
        }
        'c' => match g.snapshot_create(&node, &snapshot_name) {
            Ok(()) => true,
            Err(e) => {
                error_report(&format!(
                    "Could not create snapshot '{snapshot_name}': {}",
                    errno_text(&e)
                ));
                false
            }
        },
        'a' => match g.snapshot_goto(&node, &snapshot_name) {
            Ok(()) => true,
            Err(e) => {
                report_error(&e.prepend(format!("Could not apply snapshot '{snapshot_name}': ")));
                false
            }
        },
        _ => match g.snapshot_find(&node, &snapshot_name) {
            Ok(Some(sn)) => match g.snapshot_delete(&node, Some(&sn.id), Some(&sn.name)) {
                Ok(()) => true,
                Err(e) => {
                    report_error(
                        &e.prepend(format!("Could not delete snapshot '{snapshot_name}': ")),
                    );
                    false
                }
            },
            _ => {
                error_report(&format!(
                    "Could not delete snapshot '{snapshot_name}': snapshot not found"
                ));
                false
            }
        },
    };
    drop(blk);
    Ok(i32::from(!ok))
}
