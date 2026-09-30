// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img commit`: writes what an image holds into a file of its backing chain.
//!
//! QEMU runs the active commit job (the mirror job in commit mode) for this. There are no
//! block jobs here, so [`copy_to_base`] does what that job does with a single request at a
//! time: it marks the ranges the images above the base allocate in chunks of the default
//! bitmap granularity of the base, and then copies them, writes zeroes or discards the way
//! `mirror_iteration()` decides. The zero bitmap the job keeps to skip zeroing a range the
//! base already reads as zero is left out, so such a range gets its zeroes written again.

use std::sync::Arc;

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_block::tools::{BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO, OpenFlags};
use ruvm_block::{
    BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_RESIZE, BLK_PERM_WRITE, BlockBackend,
};

use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, cvtnum, error_exit, graph,
    img_open, lo, object_add, parse_cache_mode, qprintf, root, tryhelp,
};
use crate::convert::set_rate_limit;
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow, progress};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("cache", HasArg::Required, 't'),
    lo("drop", HasArg::No, 'd'),
    lo("base", HasArg::Required, 'b'),
    lo("rate-limit", HasArg::Required, 'r'),
    lo("progress", HasArg::No, 'p'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `MAX_IO_BYTES` of mirror.c, which with the default buffer size is also the largest
/// request the job makes.
const MAX_IO_BYTES: u64 = 1 << 20;

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut base = None;
    let mut drop = false;
    let mut progress_on = false;
    let mut quiet = false;
    let mut image_opts = false;
    let mut rate_limit = 0i64;
    let mut o = Opts::new(args, "hf:t:db:r:pq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [-t CACHE_MODE] [-b BASE_IMG]\n",
                        "        [-d] [-r RATE] [-q] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE image format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -t, --cache CACHE_MODE image cache mode (default: writeback)\n",
                        "  -d, --drop\n",
                        "     skip emptying FILE on completion\n",
                        "  -b, --base BASE_IMG\n",
                        "     image in the backing chain to commit change to\n",
                        "     (default: immediate backing file; implies --drop)\n",
                        "  -r, --rate-limit RATE\n",
                        "     I/O rate limit, in bytes per second\n",
                        "  -p, --progress\n",
                        "     display progress information\n",
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
            't' => cache = arg,
            'd' => drop = true,
            'b' => {
                base = Some(arg);
                // -b implies -d.
                drop = true;
            }
            'r' => match cvtnum("rate limit", &arg, true) {
                Some(v) => rate_limit = v,
                None => return Ok(1),
            },
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
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let filename = &rest[0];

    let Some(mode) = parse_cache_mode(&cache) else {
        error_report(&format!("Invalid cache option: {cache}"));
        return Ok(1);
    };
    let mut flags = OpenFlags { rdwr: true, unmap: true, ..OpenFlags::default() };
    mode.apply(&mut flags);
    let Some(blk) = img_open(image_opts, filename, fmt.as_deref(), flags, mode.writethrough, false)
    else {
        return Ok(1);
    };

    progress::init(progress_on, 1.0);
    progress::print(0.0, 100);
    let res = commit(&blk, filename, base.as_deref(), drop, rate_limit);
    progress::end();

    let res = res
        .and_then(|()| blk.flush().map_err(|e| Error::from_io("Error while closing the image", e)));
    match res {
        Ok(()) => {
            qprintf!(quiet, "Image committed.\n");
            Ok(0)
        }
        Err(e) => {
            ruvm_base::report::report_error(&e);
            Ok(1)
        }
    }
}

fn commit(
    blk: &BlockBackend,
    filename: &str,
    base: Option<&str>,
    drop: bool,
    rate_limit: i64,
) -> Result<()> {
    let g = graph();
    let bs = root(blk);
    let base_bs = match base {
        Some(base) => g.find_backing_image(&bs, base).ok_or_else(|| {
            Error::generic(format!("Did not find '{base}' in the backing chain of '{filename}'"))
        })?,
        // Unlike QMP, which defaults to the bottom of the chain, qemu-img commit has always
        // used the immediate backing file.
        None => g
            .backing_chain_next(&bs)?
            .ok_or_else(|| Error::generic("Image does not have a backing file"))?,
    };

    // commit_active_start(): the base is written, so it is made writable for the job.
    let base_read_only = g.node_details(&base_bs).map(|d| d.read_only).unwrap_or(false);
    if base_read_only {
        g.reopen_set_read_only(&base_bs, false)?;
    }
    let res = copy_to_base(&bs, &base_bs, rate_limit);
    if base_read_only {
        let _ = g.reopen_set_read_only(&base_bs, true);
    }
    res?;
    progress::print(100.0, 0);

    if !drop {
        let old_backing_blk = BlockBackend::new(g, &bs, BLK_PERM_WRITE, BLK_PERM_ALL)?;
        match old_backing_blk.make_empty() {
            Ok(()) => {}
            // -ENOTSUP from a driver that cannot empty an image is not an error here.
            Err(e) if e.to_string().contains("does not support emptying") => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The work of the active commit job, from `mirror_run()` on.
fn copy_to_base(bs: &str, base_bs: &str, rate_limit: i64) -> Result<()> {
    let g = graph();
    let job_err = |e: std::io::Error| Error::generic(strerror(&e));
    let source: Arc<BlockBackend> =
        BlockBackend::new(g, bs, BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL)?;
    let target = BlockBackend::new(
        g,
        base_bs,
        BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE | BLK_PERM_RESIZE,
        BLK_PERM_ALL,
    )?;
    if rate_limit > 0 {
        set_rate_limit(&target, rate_limit);
    }

    // bdrv_get_default_bitmap_granularity() of the target.
    let target_info = g.driver_info(base_bs).ok();
    let granularity = match target_info.as_ref().map(|i| i.cluster_size) {
        Some(c) if c > 0 => c.clamp(4096, 65536),
        _ => 65536,
    };
    let subcluster = target_info.map_or(0, |i| i.subcluster_size);

    let length = source.getlength().map_err(job_err)?;
    let target_length = target.getlength().map_err(job_err)?;
    // Active commit resizes the base when it is smaller than the active layer.
    if length > target_length {
        target.truncate(length)?;
    }

    // mirror_dirty_init(): what the images above the base allocate, in whole chunks.
    let mut dirty: Vec<(u64, u64)> = Vec::new();
    let mut offset = 0;
    while offset < length {
        let (depth, count) = g
            .is_allocated_above(bs, Some(base_bs), false, offset, length - offset)
            .map_err(job_err)?;
        assert!(count > 0);
        if depth > 0 {
            let start = offset / granularity * granularity;
            let end = (offset + count).div_ceil(granularity) * granularity;
            match dirty.last_mut() {
                Some(last) if last.1 >= start => last.1 = last.1.max(end),
                _ => dirty.push((start, end)),
            }
        }
        offset += count;
    }
    let total: u64 = dirty.iter().map(|&(s, e)| e.min(length) - s).sum();

    let mut buf = vec![0u8; MAX_IO_BYTES as usize];
    let mut done = 0u64;
    for (start, end) in dirty {
        let end = end.min(length);
        let mut offset = start;
        while offset < end {
            // mirror_iteration()
            let st = g.block_status_above(bs, None, false, true, offset, end - offset);
            let mut io_bytes = match &st {
                Ok(st) if st.ret & BDRV_BLOCK_DATA != 0 => st.pnum.min(MAX_IO_BYTES),
                Ok(st) => st.pnum,
                Err(_) => (end - offset).min(MAX_IO_BYTES),
            };
            io_bytes -= io_bytes % granularity;
            let mut zero_method = None;
            if io_bytes < granularity {
                io_bytes = granularity;
            } else if let Ok(st) = &st {
                if st.ret & BDRV_BLOCK_DATA == 0 {
                    // bdrv_round_to_subclusters() of the target leaves the range as it is.
                    let aligned =
                        subcluster == 0 || (offset % subcluster == 0 && io_bytes % subcluster == 0);
                    if aligned {
                        zero_method = Some(st.ret & BDRV_BLOCK_ZERO != 0);
                    }
                }
            }
            // mirror_clip_bytes()
            let io_bytes = io_bytes.min(length - offset);
            match zero_method {
                Some(true) => target.pwrite_zeroes(offset, io_bytes, true).map_err(job_err)?,
                Some(false) => target.pdiscard(offset, io_bytes).map_err(job_err)?,
                None => {
                    let mut left = io_bytes;
                    let mut at = offset;
                    while left > 0 {
                        let n = left.min(MAX_IO_BYTES);
                        let b = &mut buf[..n as usize];
                        source.pread(at, b).map_err(job_err)?;
                        target.pwrite(at, b).map_err(job_err)?;
                        at += n;
                        left -= n;
                    }
                }
            }
            offset += io_bytes;
            done += io_bytes;
            if total > 0 {
                progress::print(done as f32 / total as f32 * 100.0, 0);
            }
        }
    }
    target.flush().map_err(job_err)?;
    Ok(())
}
