// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img compare`: 0 when the images are the same, 1 when they differ, more on errors.

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_block::BlockBackend;
use ruvm_block::tools::{BDRV_BLOCK_ALLOCATED, BDRV_BLOCK_ZERO, OpenFlags};

use crate::buf::{IO_BUF_SIZE, compare_buffers, find_nonzero};
use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, error_exit, graph, img_open, lo,
    object_add, parse_cache_mode, qprintf, root, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow, progress};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("a-format", HasArg::Required, 'f'),
    lo("b-format", HasArg::Required, 'F'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("strict", HasArg::No, 's'),
    lo("cache", HasArg::Required, 'T'),
    lo("force-share", HasArg::No, 'U'),
    lo("progress", HasArg::No, 'p'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `check_empty_sectors()`: 0 when the range reads as zeroes, 1 on a mismatch (printed) and
/// 4 when reading fails.
pub(crate) fn check_empty_sectors(
    blk: &BlockBackend,
    offset: u64,
    bytes: usize,
    filename: &str,
    buffer: &mut [u8],
    quiet: bool,
) -> i32 {
    let buf = &mut buffer[..bytes];
    if let Err(e) = blk.pread(offset, buf) {
        error_report(&format!(
            "Error while reading offset {offset} of {filename}: {}",
            strerror(&e)
        ));
        return 4;
    }
    if let Some(idx) = find_nonzero(buf) {
        qprintf!(quiet, "Content mismatch at offset {}!\n", offset + idx as u64);
        return 1;
    }
    0
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt1 = None;
    let mut fmt2 = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut progress_on = false;
    let mut quiet = false;
    let mut strict = false;
    let mut image_opts = false;
    let mut force_share = false;
    let mut o = Opts::new(args, "hf:F:sT:Upq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[[-f FMT] [-F FMT] | --image-opts] [-s] [-T CACHE]\n",
                        "        [-U] [-p] [-q] [--object OBJDEF] FILE1 FILE2\n",
                    ),
                    concat!(
                        "  -f, --a-format FMT\n",
                        "     specify FILE1 image format explicitly (default: probing is used)\n",
                        "  -F, --b-format FMT\n",
                        "     specify FILE2 image format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE1 and FILE2 as option strings (key=value,..), not file \
                         names\n",
                        "     (incompatible with -f|--a-format and -F|--b-format)\n",
                        "  -s, --strict\n",
                        "     strict mode, also check if sizes are equal\n",
                        "  -T, --cache CACHE_MODE\n",
                        "     images caching mode (default: writeback)\n",
                        "  -U, --force-share\n",
                        "     open images in shared mode for concurrent access\n",
                        "  -p, --progress\n",
                        "     display progress information\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
                        "  --object OBJDEF\n",
                        "     defines QEMU user-creatable object\n",
                        "  FILE1, FILE2\n",
                        "     names of the image files, or option strings (key=value,..)\n",
                        "     with --image-opts, to compare\n",
                    ),
                ));
            }
            'f' => fmt1 = Some(arg),
            'F' => fmt2 = Some(arg),
            OPTION_IMAGE_OPTS => image_opts = true,
            's' => strict = true,
            'T' => cache = arg,
            'U' => force_share = true,
            'p' => progress_on = true,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    // Progress is not shown in quiet mode.
    if quiet {
        progress_on = false;
    }
    let rest = o.rest();
    if rest.len() != 2 {
        return Err(error_exit(&o.argv0, "Expecting two image file names"));
    }
    progress::init(progress_on, 2.0);
    let ret =
        compare(&rest[0], &rest[1], fmt1, fmt2, &cache, image_opts, force_share, strict, quiet);
    progress::end();
    Ok(ret)
}

#[allow(clippy::too_many_arguments, reason = "the locals of img_compare()")]
fn compare(
    filename1: &str,
    filename2: &str,
    fmt1: Option<String>,
    fmt2: Option<String>,
    cache: &str,
    image_opts: bool,
    force_share: bool,
    strict: bool,
    quiet: bool,
) -> i32 {
    let Some(mode) = parse_cache_mode(cache) else {
        error_report(&format!("Invalid source cache option: {cache}"));
        return 2;
    };
    let mut flags = OpenFlags::default();
    mode.apply(&mut flags);
    let Some(blk1) =
        img_open(image_opts, filename1, fmt1.as_deref(), flags, mode.writethrough, force_share)
    else {
        return 2;
    };
    let Some(blk2) =
        img_open(image_opts, filename2, fmt2.as_deref(), flags, mode.writethrough, force_share)
    else {
        return 2;
    };
    let (bs1, bs2) = (root(&blk1), root(&blk2));
    let mut buf1 = vec![0u8; IO_BUF_SIZE];
    let mut buf2 = vec![0u8; IO_BUF_SIZE];
    let total_size1 = match blk1.getlength() {
        Ok(l) => l,
        Err(e) => {
            error_report(&format!("Can't get size of {filename1}: {}", strerror(&e)));
            return 4;
        }
    };
    let total_size2 = match blk2.getlength() {
        Ok(l) => l,
        Err(e) => {
            error_report(&format!("Can't get size of {filename2}: {}", strerror(&e)));
            return 4;
        }
    };
    let total_size = total_size1.min(total_size2);
    let progress_base = total_size1.max(total_size2);
    progress::print(0.0, 100);

    if strict && total_size1 != total_size2 {
        qprintf!(quiet, "Strict mode: Image size mismatch!\n");
        return 1;
    }

    let g = graph();
    let mut offset = 0u64;
    while offset < total_size {
        let Ok(st1) = g.block_status_above(&bs1, None, false, true, offset, total_size1 - offset)
        else {
            error_report(&format!("Sector allocation test failed for {filename1}"));
            return 3;
        };
        let Ok(st2) = g.block_status_above(&bs2, None, false, true, offset, total_size2 - offset)
        else {
            error_report(&format!("Sector allocation test failed for {filename2}"));
            return 3;
        };
        let allocated1 = st1.ret & BDRV_BLOCK_ALLOCATED != 0;
        let allocated2 = st2.ret & BDRV_BLOCK_ALLOCATED != 0;
        assert!(st1.pnum != 0 && st2.pnum != 0);
        let mut chunk = st1.pnum.min(st2.pnum);

        if strict && st1.ret != st2.ret {
            qprintf!(quiet, "Strict mode: Offset {offset} block status mismatch!\n");
            return 1;
        }
        if st1.ret & BDRV_BLOCK_ZERO != 0 && st2.ret & BDRV_BLOCK_ZERO != 0 {
            // Nothing to do.
        } else if allocated1 == allocated2 {
            if allocated1 {
                chunk = chunk.min(IO_BUF_SIZE as u64);
                let n = chunk as usize;
                if let Err(e) = blk1.pread(offset, &mut buf1[..n]) {
                    error_report(&format!(
                        "Error while reading offset {offset} of {filename1}: {}",
                        strerror(&e)
                    ));
                    return 4;
                }
                if let Err(e) = blk2.pread(offset, &mut buf2[..n]) {
                    error_report(&format!(
                        "Error while reading offset {offset} of {filename2}: {}",
                        strerror(&e)
                    ));
                    return 4;
                }
                let (differ, pnum) = compare_buffers(&buf1[..n], &buf2[..n], 0);
                if differ || pnum != n {
                    let at = offset + if differ { 0 } else { pnum as u64 };
                    qprintf!(quiet, "Content mismatch at offset {at}!\n");
                    return 1;
                }
            }
        } else {
            chunk = chunk.min(IO_BUF_SIZE as u64);
            let ret = if allocated1 {
                check_empty_sectors(&blk1, offset, chunk as usize, filename1, &mut buf1, quiet)
            } else {
                check_empty_sectors(&blk2, offset, chunk as usize, filename2, &mut buf1, quiet)
            };
            if ret != 0 {
                return ret;
            }
        }
        offset += chunk;
        progress::print(chunk as f32 / progress_base as f32 * 100.0, 100);
    }

    if total_size1 != total_size2 {
        qprintf!(quiet, "Warning: Image size mismatch!\n");
        let (blk_over, bs_over, filename_over) = if total_size1 > total_size2 {
            (&blk1, &bs1, filename1)
        } else {
            (&blk2, &bs2, filename2)
        };
        while offset < progress_base {
            let Ok(st) =
                g.block_status_above(bs_over, None, false, true, offset, progress_base - offset)
            else {
                error_report(&format!("Sector allocation test failed for {filename_over}"));
                return 3;
            };
            let mut chunk = st.pnum;
            if st.ret & BDRV_BLOCK_ALLOCATED != 0 && st.ret & BDRV_BLOCK_ZERO == 0 {
                chunk = chunk.min(IO_BUF_SIZE as u64);
                let ret = check_empty_sectors(
                    blk_over,
                    offset,
                    chunk as usize,
                    filename_over,
                    &mut buf1,
                    quiet,
                );
                if ret != 0 {
                    return ret;
                }
            }
            offset += chunk;
            progress::print(chunk as f32 / progress_base as f32 * 100.0, 100);
        }
    }

    qprintf!(quiet, "Images are identical.\n");
    0
}
