// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img bitmap`: works on the persistent dirty bitmaps of an image.

use ruvm_base::report::{error_report, report_error};
use std::sync::Arc;

use ruvm_block::tools::OpenFlags;
use ruvm_block::{BLK_PERM_ALL, BlockBackend};
use ruvm_qapi::types::{BlockDirtyBitmap, BlockDirtyBitmapAdd};

use crate::common::{
    OPTION_ADD, OPTION_CLEAR, OPTION_DISABLE, OPTION_ENABLE, OPTION_IMAGE_OPTS, OPTION_MERGE,
    OPTION_OBJECT, OPTION_REMOVE, OPTION_REMOVE_ALL, Opts, cvtnum, graph, img_open, lo, object_add,
    root, tryhelp,
};
use crate::convert::dirty_bitmap_merge;
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("add", HasArg::No, OPTION_ADD),
    lo("granularity", HasArg::Required, 'g'),
    lo("remove", HasArg::No, OPTION_REMOVE),
    lo("remove-all", HasArg::No, OPTION_REMOVE_ALL),
    lo("clear", HasArg::No, OPTION_CLEAR),
    lo("enable", HasArg::No, OPTION_ENABLE),
    lo("disable", HasArg::No, OPTION_DISABLE),
    lo("merge", HasArg::Required, OPTION_MERGE),
    lo("source-file", HasArg::Required, 'b'),
    lo("source-format", HasArg::Required, 'F'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `ImgBitmapAction`.
enum Action {
    Add,
    Remove,
    Clear,
    Enable,
    Disable,
    /// The source bitmap.
    Merge(String),
    RemoveAll,
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut src_fmt = None;
    let mut src_filename = None;
    let mut image_opts = false;
    let mut granularity = 0i64;
    let mut add = false;
    let mut merge = false;
    let mut need_bitmap_name = false;
    let mut actions = Vec::new();
    let mut o = Opts::new(args, "hf:g:b:F:", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts]\n",
                        "        ( --add [-g SIZE] | --remove | --remove-all | --clear | \
                         --enable |\n",
                        "          --disable | --merge SOURCE [-b SRC_FILE [-F SRC_FMT]] )..\n",
                        "        [--object OBJDEF] FILE [BITMAP]\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  --add\n",
                        "     creates BITMAP in FILE, enables to record future edits\n",
                        "  -g, --granularity SIZE[bKMGTPE]\n",
                        "     sets non-default granularity for the bitmap being added,\n",
                        "     with optional multiplier suffix (in powers of 1024)\n",
                        "  --remove\n",
                        "     removes BITMAP from FILE\n",
                        "  --remove-all\n",
                        "     removes all bitmaps from FILE\n",
                        "  --clear\n",
                        "     clears BITMAP in FILE\n",
                        "  --enable, --disable\n",
                        "     starts and stops recording future edits to BITMAP in FILE\n",
                        "  --merge SOURCE\n",
                        "     merges contents of the SOURCE bitmap into BITMAP in FILE\n",
                        "  -b, --source-file SRC_FILE\n",
                        "     select alternative source file for --merge\n",
                        "  -F, --source-format SRC_FMT\n",
                        "     specify format for SRC_FILE explicitly\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE\n",
                        "     name of the image file, or option string (key=value,..)\n",
                        "     with --image-opts, to operate on\n",
                        "  BITMAP\n",
                        "     name of the bitmap to add, remove, clear, enable, disable or \
                         merge to\n",
                    ),
                ));
            }
            'f' => fmt = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            OPTION_ADD => {
                actions.push(Action::Add);
                add = true;
                need_bitmap_name = true;
            }
            'g' => match cvtnum("granularity", &arg, true) {
                Some(v) => granularity = v,
                None => return Ok(1),
            },
            OPTION_REMOVE => {
                actions.push(Action::Remove);
                need_bitmap_name = true;
            }
            OPTION_REMOVE_ALL => actions.push(Action::RemoveAll),
            OPTION_CLEAR => {
                actions.push(Action::Clear);
                need_bitmap_name = true;
            }
            OPTION_ENABLE => {
                actions.push(Action::Enable);
                need_bitmap_name = true;
            }
            OPTION_DISABLE => {
                actions.push(Action::Disable);
                need_bitmap_name = true;
            }
            OPTION_MERGE => {
                actions.push(Action::Merge(arg));
                merge = true;
                need_bitmap_name = true;
            }
            'b' => src_filename = Some(arg),
            'F' => src_fmt = Some(arg),
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }

    let rest = o.rest();
    // The images stay open until after they are inactivated, as with blk_unref() in QEMU.
    let mut open = Vec::new();
    let ret = bitmap(
        &mut open,
        &rest,
        actions,
        fmt.as_deref(),
        image_opts,
        granularity,
        (add, merge, need_bitmap_name),
        src_filename.as_deref(),
        src_fmt.as_deref(),
    );

    // bdrv_inactivate_all() writes the bitmaps out and tells whether that worked, error or
    // not.
    let mut ret = i32::from(ret.is_err());
    // blk_root_inactivate(): the backends give up their permissions first.
    for blk in &open {
        let _ = blk.set_perm(0, BLK_PERM_ALL);
    }
    if let Err(e) = graph().blockdev_set_active(None, false) {
        report_error(&e.prepend("Error while closing the image: "));
        ret = 1;
    }
    drop(open);
    Ok(ret)
}

#[allow(clippy::too_many_arguments, reason = "the locals of img_bitmap()")]
fn bitmap(
    open: &mut Vec<Arc<BlockBackend>>,
    rest: &[String],
    actions: Vec<Action>,
    fmt: Option<&str>,
    image_opts: bool,
    granularity: i64,
    (add, merge, need_bitmap_name): (bool, bool, bool),
    src_filename: Option<&str>,
    src_fmt: Option<&str>,
) -> Result<(), ()> {
    if actions.is_empty() {
        error_report(
            "Need at least one of --add, --remove, --remove-all, --clear, --enable, --disable, \
             or --merge",
        );
        return Err(());
    }
    if granularity != 0 && !add {
        error_report("granularity only supported with --add");
        return Err(());
    }
    if src_fmt.is_some() && src_filename.is_none() {
        error_report("-F only supported with -b");
        return Err(());
    }
    if src_filename.is_some() && !merge {
        error_report("Merge bitmap source file only supported with --merge");
        return Err(());
    }
    if need_bitmap_name && rest.len() != 2 {
        error_report("Expecting filename and bitmap name");
        return Err(());
    }
    // Every action but --remove-all needs a bitmap name, so here --remove-all is all there
    // is and there must not be a bitmap name.
    if !need_bitmap_name && rest.len() != 1 {
        error_report("Expecting filename");
        return Err(());
    }
    let filename = &rest[0];
    let bitmap = rest.get(1).map_or("(null)", String::as_str);

    // The backing chain is not needed: the bitmaps are changed right in this image, no
    // matter what the image holds.
    let flags = OpenFlags { rdwr: true, no_backing: true, ..OpenFlags::default() };
    let blk = img_open(image_opts, filename, fmt, flags, false, false).ok_or(())?;
    let bs = root(&blk);
    open.push(blk);
    let src_bs = match src_filename {
        Some(src_filename) => {
            let flags = OpenFlags { no_backing: true, ..OpenFlags::default() };
            let src = img_open(false, src_filename, src_fmt, flags, false, false).ok_or(())?;
            let n = root(&src);
            open.push(src);
            n
        }
        None => bs.clone(),
    };

    let g = graph();
    let target = |name: &str| BlockDirtyBitmap { node: bs.clone(), name: name.to_string() };
    for act in actions {
        let mut name = bitmap.to_string();
        let (op, res) = match act {
            Action::Add => (
                "add",
                g.block_dirty_bitmap_add(&BlockDirtyBitmapAdd {
                    node: bs.clone(),
                    name: name.clone(),
                    granularity: (granularity != 0).then_some(granularity as u32),
                    persistent: Some(true),
                    disabled: None,
                }),
            ),
            Action::Remove => ("remove", g.block_dirty_bitmap_remove(&target(bitmap))),
            Action::RemoveAll => {
                let mut res = Ok(());
                // bdrv_dirty_bitmap_first() until there are none left.
                while let Some(bm) =
                    g.dirty_bitmap_details(&bs).unwrap_or_default().into_iter().next()
                {
                    if let Err(e) = g.block_dirty_bitmap_remove(&target(&bm.name)) {
                        // The name of the bitmap that failed goes into the message.
                        name = bm.name;
                        res = Err(e);
                        break;
                    }
                }
                ("remove-all", res)
            }
            Action::Clear => ("clear", g.block_dirty_bitmap_clear(&target(bitmap))),
            Action::Enable => ("enable", g.block_dirty_bitmap_enable(&target(bitmap))),
            Action::Disable => ("disable", g.block_dirty_bitmap_disable(&target(bitmap))),
            Action::Merge(src) => ("merge", dirty_bitmap_merge(&bs, bitmap, &src_bs, &src)),
        };
        if let Err(e) = res {
            report_error(&e.prepend(format!("Operation {op} on bitmap {name} failed: ")));
            return Err(());
        }
    }
    Ok(())
}
