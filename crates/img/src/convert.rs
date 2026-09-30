// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img convert`.
//!
//! QEMU copies with up to 16 coroutines; here one loop does the requests in order, which is
//! what QEMU does with `-m 1`. `-m` is still checked, and `-W` (out of order writes) changes
//! nothing. `-C` asks for copy offloading, which the block layer does not have: QEMU falls
//! back to reading and writing when `blk_co_copy_range()` fails, and that is what happens
//! every time here.

use std::sync::Arc;

use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error, warn_report};
use ruvm_block::BlockBackend;
use ruvm_block::throttle::{BucketType, ThrottleConfig};
use ruvm_block::tools::{self, BDRV_BLOCK_DATA, BDRV_BLOCK_ZERO, OpenFlags};
use ruvm_qapi::QDict;
use ruvm_qapi::opts::QemuOptsList;
use ruvm_qapi::types::{
    BlockDirtyBitmap, BlockDirtyBitmapAdd, BlockDirtyBitmapMerge, BlockDirtyBitmapOrStr,
};

use crate::buf::{self, IO_BUF_SIZE, SECTOR};
use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_BITMAPS, OPTION_IMAGE_OPTS, OPTION_OBJECT, OPTION_SALVAGE,
    OPTION_SKIP_BROKEN, OPTION_TARGET_IMAGE_OPTS, OPTION_TARGET_IS_ZERO, Opts, SnapshotArg,
    accumulate_options, cvtnum, cvtnum_full, graph, has_help_option, img_open, img_open_file, lo,
    load_snapshot, object_add, opts_append, parse_cache_mode, parse_snapshot_arg,
    print_block_option_help, root, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow, progress};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("source-format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("source-cache", HasArg::Required, 'T'),
    lo("snapshot", HasArg::Required, 'l'),
    lo("bitmaps", HasArg::No, OPTION_BITMAPS),
    lo("skip-broken-bitmaps", HasArg::No, OPTION_SKIP_BROKEN),
    lo("salvage", HasArg::No, OPTION_SALVAGE),
    lo("target-format", HasArg::Required, 'O'),
    lo("target-image-opts", HasArg::No, OPTION_TARGET_IMAGE_OPTS),
    lo("target-format-options", HasArg::Required, 'o'),
    lo("target-cache", HasArg::Required, 't'),
    lo("backing", HasArg::Required, 'b'),
    lo("backing-format", HasArg::Required, 'F'),
    lo("sparse-size", HasArg::Required, 'S'),
    lo("no-create", HasArg::No, 'n'),
    lo("target-is-zero", HasArg::No, OPTION_TARGET_IS_ZERO),
    lo("force-share", HasArg::No, 'U'),
    lo("rate-limit", HasArg::Required, 'r'),
    lo("parallel", HasArg::Required, 'm'),
    lo("oob-writes", HasArg::No, 'W'),
    lo("copy-range-offloading", HasArg::No, 'C'),
    lo("progress", HasArg::No, 'p'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

/// `MAX_COROUTINES`.
const MAX_COROUTINES: i64 = 16;
/// `MAX_BUF_SECTORS`.
const MAX_BUF_SECTORS: usize = 32768;
/// `BDRV_REQUEST_MAX_SECTORS`.
const REQUEST_MAX_SECTORS: i64 = (i32::MAX >> 9) as i64;
/// `CONVERT_THROTTLE_GROUP`.
const CONVERT_THROTTLE_GROUP: &str = "img_convert";

/// `enum ImgConvertBlockStatus`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    Data,
    Zero,
    BackingFile,
}

/// `ImgConvertState`.
struct State {
    src: Vec<Arc<BlockBackend>>,
    src_nodes: Vec<String>,
    src_sectors: Vec<i64>,
    src_alignment: Vec<i64>,
    total_sectors: i64,
    allocated_sectors: i64,
    allocated_done: i64,
    status: Status,
    sector_next_status: i64,
    target: Option<Arc<BlockBackend>>,
    has_zero_init: bool,
    compressed: bool,
    target_is_new: bool,
    target_has_backing: bool,
    /// Negative if unknown.
    target_backing_sectors: i64,
    copy_range: bool,
    salvage: bool,
    quiet: bool,
    min_sparse: usize,
    alignment: usize,
    cluster_sectors: usize,
    buf_sectors: usize,
}

impl State {
    /// `convert_select_part()`: the source that holds `sector_num` and where it starts.
    fn select_part(&self, sector_num: i64) -> (usize, i64) {
        let mut cur = 0;
        let mut cur_offset = 0;
        while sector_num - cur_offset >= self.src_sectors[cur] {
            cur_offset += self.src_sectors[cur];
            cur += 1;
            assert!(cur < self.src.len());
        }
        (cur, cur_offset)
    }

    /// `convert_iteration_sectors()`: how many sectors from `sector_num` go in one request,
    /// with their status in `self.status`. An error was reported already.
    fn iteration_sectors(&mut self, sector_num: i64) -> Result<i64, ()> {
        let (src_cur, src_cur_offset) = self.select_part(sector_num);
        assert!(self.total_sectors > sector_num);
        let mut n = (self.total_sectors - sector_num).min(REQUEST_MAX_SECTORS);
        let mut post_backing_zero = false;
        if self.target_backing_sectors >= 0 {
            if sector_num >= self.target_backing_sectors {
                post_backing_zero = true;
            } else if sector_num + n > self.target_backing_sectors {
                // Zeroes are handled differently from there on, so split the request.
                n = self.target_backing_sectors - sector_num;
            }
        }

        if self.sector_next_status <= sector_num {
            let offset = ((sector_num - src_cur_offset) * SECTOR as i64) as u64;
            let src_bs = &self.src_nodes[src_cur];
            let g = graph();
            let base = if self.target_has_backing {
                g.skip_filters(src_bs).ok().and_then(|n| g.cow_bs(&n).ok().flatten())
            } else {
                None
            };
            let (ret, count) = loop {
                let count = n as u64 * SECTOR as u64;
                match g.block_status_above(src_bs, base.as_deref(), false, true, offset, count) {
                    Ok(st) => break (st.ret, st.pnum),
                    Err(e) => {
                        if !self.salvage {
                            error_report(&format!(
                                "error while reading block status at offset {offset}: {}",
                                strerror(&e)
                            ));
                            return Err(());
                        }
                        if n == 1 {
                            if !self.quiet {
                                warn_report(&format!(
                                    "error while reading block status at offset {offset}: {}",
                                    strerror(&e)
                                ));
                            }
                            // Just try to read the data then.
                            break (BDRV_BLOCK_DATA, SECTOR as u64);
                        }
                        // Retry on a shorter range.
                        n = (n + 3) / 4;
                    }
                }
            };
            n = count.div_ceil(SECTOR as u64) as i64;
            // Keep the next status query aligned to the source's alignment and cluster size
            // so that nothing is read twice.
            let tail = (sector_num - src_cur_offset + n) % self.src_alignment[src_cur];
            if n > tail {
                n -= tail;
            }
            self.status = if ret & BDRV_BLOCK_ZERO != 0 {
                if post_backing_zero { Status::BackingFile } else { Status::Zero }
            } else if ret & BDRV_BLOCK_DATA != 0 {
                Status::Data
            } else if self.target_has_backing {
                Status::BackingFile
            } else {
                Status::Data
            };
            self.sector_next_status = sector_num + n;
        }

        n = n.min(self.sector_next_status - sector_num);
        if self.status == Status::Data {
            n = n.min(self.buf_sectors as i64);
        }
        // Compressed images are written in whole clusters, so an unallocated area shorter
        // than a cluster counts as allocated.
        if self.compressed {
            let cs = self.cluster_sectors as i64;
            if n < cs {
                n = cs.min(self.total_sectors - sector_num);
                self.status = Status::Data;
            } else {
                n = n / cs * cs;
            }
        }
        Ok(n)
    }

    /// `convert_co_read()`.
    fn read(
        &self,
        mut sector_num: i64,
        mut nb_sectors: i64,
        buf: &mut [u8],
    ) -> std::io::Result<()> {
        assert!(nb_sectors <= self.buf_sectors as i64);
        let mut single_read_until = 0u64;
        let mut pos = 0usize;
        while nb_sectors > 0 {
            // With compression and several sources a request can reach into the next one.
            let (src_cur, src_cur_offset) = self.select_part(sector_num);
            let blk = &self.src[src_cur];
            let bs_sectors = self.src_sectors[src_cur];
            let offset = ((sector_num - src_cur_offset) as u64) << 9;
            let mut n = nb_sectors.min(bs_sectors - (sector_num - src_cur_offset));
            if single_read_until > offset {
                n = 1;
            }
            let len = n as usize * SECTOR;
            if let Err(e) = blk.pread(offset, &mut buf[pos..pos + len]) {
                if !self.salvage {
                    return Err(e);
                }
                if n > 1 {
                    single_read_until = offset + len as u64;
                    continue;
                }
                if !self.quiet {
                    warn_report(&format!("error while reading offset {offset}: {}", strerror(&e)));
                }
                buf[pos..pos + SECTOR].fill(0);
            }
            sector_num += n;
            nb_sectors -= n;
            pos += len;
        }
        Ok(())
    }

    /// `convert_co_write()`.
    fn write(
        &self,
        mut sector_num: i64,
        mut nb_sectors: i64,
        buf: &[u8],
        status: Status,
    ) -> std::io::Result<()> {
        let target = self.target.as_ref().expect("target is open");
        let mut pos = 0usize;
        while nb_sectors > 0 {
            let mut n = nb_sectors as usize;
            let offset = (sector_num as u64) << 9;
            let mut zero = false;
            match status {
                Status::BackingFile => {
                    // Leave what is unallocated in the source unallocated, so that the
                    // backing file shows through.
                    assert!(self.target_has_backing);
                }
                Status::Data => {
                    let chunk = &buf[pos..pos + n * SECTOR];
                    // Write it when told to keep the target fully allocated (-S 0) or when
                    // there is data. A compressed cluster is written whole unless it is all
                    // zeroes.
                    let write = if self.min_sparse == 0 {
                        true
                    } else if !self.compressed {
                        let (data, pnum) = buf::is_allocated_sectors_min(
                            chunk,
                            n,
                            self.min_sparse,
                            sector_num as u64,
                            self.alignment,
                        );
                        n = pnum;
                        data
                    } else {
                        !buf::is_zero(chunk)
                    };
                    if write {
                        let data = &chunk[..n * SECTOR];
                        if self.compressed {
                            target.pwrite_compressed(offset, data)?;
                        } else {
                            target.pwrite(offset, data)?;
                        }
                    } else {
                        zero = true;
                    }
                }
                Status::Zero => zero = true,
            }
            if zero && !self.has_zero_init {
                target.pwrite_zeroes(offset, (n * SECTOR) as u64, true)?;
            } else if zero {
                assert!(!self.target_has_backing);
            }
            sector_num += n as i64;
            nb_sectors -= n as i64;
            if status == Status::Data {
                pos += n * SECTOR;
            }
        }
        Ok(())
    }

    /// `convert_co_do_copy()` with one coroutine.
    fn do_copy_loop(&mut self) -> Result<(), ()> {
        let mut buf = vec![0u8; self.buf_sectors * SECTOR];
        let mut next = 0i64;
        while next < self.total_sectors {
            let mut n = self.iteration_sectors(next)?;
            let sector_num = next;
            let mut status = self.status;
            if self.min_sparse == 0 && status == Status::Zero {
                n = n.min(self.buf_sectors as i64);
            }
            next += n;
            if status == Status::Data || (self.min_sparse == 0 && status == Status::Zero) {
                self.allocated_done += n;
                progress::print(
                    (100.0 * self.allocated_done as f64 / self.allocated_sectors as f64) as f32,
                    0,
                );
            }
            // There is no copy offloading, so a copy_range request fails and QEMU retries
            // with a read and a write.
            if status == Status::Data {
                let len = n as usize * SECTOR;
                if let Err(e) = self.read(sector_num, n, &mut buf[..len]) {
                    error_report(&format!(
                        "error while reading at byte {}: {}",
                        sector_num * SECTOR as i64,
                        strerror(&e)
                    ));
                    return Err(());
                }
            } else if self.min_sparse == 0 && status == Status::Zero {
                status = Status::Data;
                buf[..n as usize * SECTOR].fill(0);
            }
            if let Err(e) = self.write(sector_num, n, &buf, status) {
                error_report(&format!(
                    "error while writing at byte {}: {}",
                    sector_num * SECTOR as i64,
                    strerror(&e)
                ));
                return Err(());
            }
        }
        Ok(())
    }

    /// `convert_do_copy()`.
    fn do_copy(&mut self) -> Result<(), ()> {
        let target = self.target.clone().expect("target is open");
        // Is the target zero initialised, or can it be made so cheaply?
        if !self.has_zero_init
            && self.target_is_new
            && self.min_sparse != 0
            && !self.target_has_backing
        {
            self.has_zero_init = graph().has_zero_init(&root(&target));
        }
        // Compressed images are copied one cluster at a time.
        if self.compressed {
            if self.cluster_sectors == 0 || self.cluster_sectors > self.buf_sectors {
                error_report("invalid cluster size");
                return Err(());
            }
            self.buf_sectors = self.cluster_sectors;
        }
        let mut sector_num = 0;
        while sector_num < self.total_sectors {
            let n = self.iteration_sectors(sector_num)?;
            if self.status == Status::Data || (self.min_sparse == 0 && self.status == Status::Zero)
            {
                self.allocated_sectors += n;
            }
            sector_num += n;
        }

        self.sector_next_status = 0;
        self.do_copy_loop()?;

        if self.compressed {
            // Signal the end so that the driver can align the file.
            if target.pwrite_compressed(0, &[]).is_err() {
                return Err(());
            }
        }
        Ok(())
    }
}

/// `convert_check_bitmaps()`.
fn check_bitmaps(src: &str, skip_broken: bool) -> Result<(), ()> {
    let g = graph();
    if !g.supports_persistent_dirty_bitmap(src) {
        error_report("Source lacks bitmap support");
        return Err(());
    }
    for bm in g.dirty_bitmap_details(src).unwrap_or_default() {
        if !bm.persistent {
            continue;
        }
        if !skip_broken && bm.inconsistent {
            error_report(&format!("Cannot copy inconsistent bitmap '{}'", bm.name));
            eprintln!("Try --skip-broken-bitmaps, or use 'qemu-img bitmap --remove' to delete it");
            return Err(());
        }
    }
    Ok(())
}

/// `do_dirty_bitmap_merge()`.
pub(crate) fn dirty_bitmap_merge(
    dst_node: &str,
    dst_name: &str,
    src_node: &str,
    src_name: &str,
) -> ruvm_base::Result<()> {
    graph().block_dirty_bitmap_merge(&BlockDirtyBitmapMerge {
        node: dst_node.to_string(),
        target: dst_name.to_string(),
        bitmaps: vec![BlockDirtyBitmapOrStr::External(BlockDirtyBitmap {
            node: src_node.to_string(),
            name: src_name.to_string(),
        })],
    })
}

/// `convert_copy_bitmaps()`.
fn copy_bitmaps(src: &str, dst: &str, skip_broken: bool) -> Result<(), ()> {
    let g = graph();
    for bm in g.dirty_bitmap_details(src).unwrap_or_default() {
        if !bm.persistent {
            continue;
        }
        let name = &bm.name;
        if skip_broken && bm.inconsistent {
            warn_report(&format!("Skipping inconsistent bitmap '{name}'"));
            continue;
        }
        let add = BlockDirtyBitmapAdd {
            node: dst.to_string(),
            name: name.clone(),
            granularity: Some(bm.granularity),
            persistent: Some(true),
            disabled: Some(!bm.enabled),
        };
        if let Err(e) = g.block_dirty_bitmap_add(&add) {
            report_error(&e.prepend(format!("Failed to create bitmap {name}: ")));
            return Err(());
        }
        if let Err(e) = dirty_bitmap_merge(dst, name, src, name) {
            report_error(&e.prepend(format!("Failed to populate bitmap {name}: ")));
            let _ = g.block_dirty_bitmap_remove(&BlockDirtyBitmap {
                node: dst.to_string(),
                name: name.clone(),
            });
            return Err(());
        }
    }
    Ok(())
}

/// `set_rate_limit()`.
pub(crate) fn set_rate_limit(blk: &BlockBackend, rate_limit: i64) {
    let mut cfg = ThrottleConfig::new();
    cfg.bucket_mut(BucketType::BpsWrite).avg = rate_limit as u64;
    blk.io_limits_enable(CONVERT_THROTTLE_GROUP);
    blk.set_io_limits(&cfg);
}

/// `add_old_style_options()`: `-b` and `-F` as create options.
pub(crate) fn add_old_style_options(
    fmt: &str,
    opts: &mut ruvm_qapi::opts::QemuOpts,
    base_filename: Option<&str>,
    base_fmt: Option<&str>,
) -> Result<(), ()> {
    if let Some(b) = base_filename {
        if opts.set("backing_file", b).is_err() {
            error_report(&format!("Backing file not supported for file format '{fmt}'"));
            return Err(());
        }
    }
    if let Some(b) = base_fmt {
        if opts.set("backing_fmt", b).is_err() {
            error_report(&format!("Backing file format not supported for file format '{fmt}'"));
            return Err(());
        }
    }
    Ok(())
}

/// Everything the option loop collects.
#[derive(Default)]
struct Args {
    fmt: Option<String>,
    out_fmt: Option<String>,
    cache: String,
    src_cache: String,
    out_baseimg: Option<String>,
    backing_fmt: Option<String>,
    snapshot: Option<SnapshotArg>,
    options: Option<String>,
    image_opts: bool,
    skip_create: bool,
    progress: bool,
    tgt_image_opts: bool,
    force_share: bool,
    explicit_min_sparse: bool,
    bitmaps: bool,
    skip_broken: bool,
    rate_limit: i64,
    min_sparse: usize,
    has_zero_init: bool,
    compressed: bool,
    copy_range: bool,
    salvage: bool,
    quiet: bool,
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut a = Args {
        cache: "unsafe".to_string(),
        src_cache: BDRV_DEFAULT_CACHE.to_string(),
        // At least 4k of zeroes for sparse detection.
        min_sparse: 8,
        ..Args::default()
    };
    let mut o = Opts::new(args, "hf:O:b:B:CcF:o:l:S:pt:T:nm:WUr:q", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => return Err(help(cmd)),
            'f' => a.fmt = Some(arg),
            OPTION_IMAGE_OPTS => a.image_opts = true,
            'T' => a.src_cache = arg,
            'l' => match parse_snapshot_arg(&arg) {
                Some(s) => a.snapshot = Some(s),
                None => return Ok(1),
            },
            OPTION_BITMAPS => a.bitmaps = true,
            OPTION_SKIP_BROKEN => a.skip_broken = true,
            OPTION_SALVAGE => a.salvage = true,
            'O' => a.out_fmt = Some(arg),
            OPTION_TARGET_IMAGE_OPTS => a.tgt_image_opts = true,
            'o' => {
                if !accumulate_options(&mut a.options, &arg) {
                    return Ok(1);
                }
            }
            't' => a.cache = arg,
            // -B was -b up to 10.0.
            'B' | 'b' => a.out_baseimg = Some(arg),
            'F' => a.backing_fmt = Some(arg),
            'S' => {
                let Some(sval) = cvtnum("buffer size for sparse output", &arg, true) else {
                    return Ok(1);
                };
                if sval % SECTOR as i64 != 0 || sval / SECTOR as i64 > MAX_BUF_SECTORS as i64 {
                    error_report(&format!(
                        "Invalid buffer size for sparse output specified. Valid sizes are \
                         multiples of {SECTOR} up to {}. Select 0 to disable sparse detection \
                         (fully allocates output).",
                        MAX_BUF_SECTORS * SECTOR
                    ));
                    return Ok(1);
                }
                a.min_sparse = (sval / SECTOR as i64) as usize;
                a.explicit_min_sparse = true;
            }
            'n' => a.skip_create = true,
            // Saying the target is blank is the same as the driver having zero init.
            OPTION_TARGET_IS_ZERO => a.has_zero_init = true,
            'c' => a.compressed = true,
            'U' => a.force_share = true,
            'r' => match cvtnum("rate limit", &arg, true) {
                Some(r) => a.rate_limit = r,
                None => return Ok(1),
            },
            'm' => {
                if cvtnum_full("number of coroutines", &arg, false, 1, MAX_COROUTINES).is_none() {
                    return Ok(1);
                }
            }
            'W' => {}
            'C' => a.copy_range = true,
            'p' => a.progress = true,
            'q' => a.quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    if a.out_fmt.is_none() && !a.tgt_image_opts {
        a.out_fmt = Some("raw".to_string());
    }
    let checks: [(bool, &str); 7] = [
        (a.skip_broken && !a.bitmaps, "Use of --skip-broken-bitmaps requires --bitmaps"),
        (a.compressed && a.copy_range, "Cannot enable copy offloading when -c is used"),
        (a.explicit_min_sparse && a.copy_range, "Cannot enable copy offloading when -S is used"),
        (a.copy_range && a.salvage, "Cannot use copy offloading in salvaging mode"),
        (a.tgt_image_opts && !a.skip_create, "--target-image-opts requires use of -n flag"),
        (a.skip_create && a.options.is_some(), "-o has no effect when skipping image creation"),
        (a.has_zero_init && !a.skip_create, "--target-is-zero requires use of -n flag"),
    ];
    for (bad, msg) in checks {
        if bad {
            error_report(msg);
            return Ok(1);
        }
    }
    let rest = o.rest();
    let src_num = rest.len().saturating_sub(1);
    let out_filename = if src_num >= 1 { rest.last().cloned() } else { None };
    if let Some(options) = &a.options {
        if has_help_option(options) {
            return Ok(match &a.out_fmt {
                Some(f) => print_block_option_help(out_filename.as_deref(), f),
                None => {
                    error_report("Option help requires a format be specified");
                    1
                }
            });
        }
    }
    if src_num < 1 {
        error_report("Must specify image file name");
        return Ok(1);
    }
    let Some(src_mode) = parse_cache_mode(&a.src_cache) else {
        error_report(&format!("Invalid source cache option: {}", a.src_cache));
        return Ok(1);
    };

    // Progress is not shown in quiet mode.
    if a.quiet {
        a.progress = false;
    }
    progress::init(a.progress, 1.0);
    progress::print(0.0, 100);
    let out_filename = out_filename.expect("there is a source");
    let ret = convert(&mut a, &rest[..src_num], &out_filename, src_mode);
    if ret == 0 {
        progress::print(100.0, 0);
    }
    progress::end();
    Ok(ret)
}

fn help(cmd: &Cmd) -> crate::Exit {
    cmd.help(
        concat!(
            "[-f SRC_FMT | --image-opts] [-T SRC_CACHE]\n",
            "        [-l SNAPSHOT] [--bitmaps [--skip-broken-bitmaps]] [--salvage]\n",
            "        [-O TGT_FMT | --target-image-opts] [-o TGT_FMT_OPTS] [-t TGT_CACHE]\n",
            "        [-b BACKING_FILE [-F BACKING_FMT]] [-S SPARSE_SIZE]\n",
            "        [-n] [--target-is-zero] [-c]\n",
            "        [-U] [-r RATE] [-m NUM_PARALLEL] [-W] [-C] [-p] [-q] [--object OBJDEF]\n",
            "        SRC_FILE [SRC_FILE2...] TGT_FILE\n",
        ),
        concat!(
            "  -f, --source-format SRC_FMT\n",
            "     specify format of all SRC_FILEs explicitly (default: probing is used)\n",
            "  --image-opts\n",
            "     treat each SRC_FILE as an option string (key=value,...), not a file name\n",
            "     (incompatible with -f|--source-format)\n",
            "  -T, --source-cache SRC_CACHE\n",
            "     source image(s) cache mode (writeback)\n",
            "  -l, --snapshot SNAPSHOT\n",
            "     specify source snapshot\n",
            "  --bitmaps\n",
            "     also copy any persistent bitmaps present in source\n",
            "  --skip-broken-bitmaps\n",
            "     skip (do not error out) any broken bitmaps\n",
            "  --salvage\n",
            "     ignore errors on input (convert unreadable areas to zeros)\n",
            "  -O, --target-format TGT_FMT\n",
            "     specify TGT_FILE image format (default: raw)\n",
            "  --target-image-opts\n",
            "     treat TGT_FILE as an option string (key=value,...), not a file name\n",
            "     (incompatible with -O|--target-format)\n",
            "  -o, --target-format-options TGT_FMT_OPTS\n",
            "     TGT_FMT-specific options\n",
            "  -t, --target-cache TGT_CACHE\n",
            "     cache mode when opening output image (default: unsafe)\n",
            "  -b, --backing BACKING_FILE (was -B in <= 10.0)\n",
            "     create target image to be a CoW on top of BACKING_FILE\n",
            "  -F, --backing-format BACKING_FMT\n",
            "     specify BACKING_FILE image format explicitly (default: probing is used)\n",
            "  -S, --sparse-size SPARSE_SIZE[bkKMGTPE]\n",
            "     specify number of consecutive zero bytes to treat as a gap on output\n",
            "     (rounded down to nearest 512 bytes), with optional multiplier suffix\n",
            "  -n, --no-create\n",
            "     omit target volume creation (e.g. on rbd)\n",
            "  --target-is-zero\n",
            "     indicates that the target volume is pre-zeroed\n",
            "  -c, --compress\n",
            "     create compressed output image (qcow and qcow2 formats only)\n",
            "  -U, --force-share\n",
            "     open images in shared mode for concurrent access\n",
            "  -r, --rate-limit RATE\n",
            "     I/O rate limit, in bytes per second\n",
            "  -m, --parallel NUM_PARALLEL\n",
            "     specify parallelism (default: 8)\n",
            "  -C, --copy-range-offloading\n",
            "     try to use copy offloading\n",
            "  -W, --oob-writes\n",
            "     enable out-of-order writes to improve performance\n",
            "  -p, --progress\n",
            "     display progress information\n",
            "  -q, --quiet\n",
            "     quiet mode (produce only error messages if any)\n",
            "  --object OBJDEF\n",
            "     defines QEMU user-creatable object\n",
            "  SRC_FILE...\n",
            "     one or more source image file names,\n",
            "     or option strings (key=value,..) with --source-image-opts\n",
            "  TGT_FILE\n",
            "     target (output) image file name,\n",
            "     or option string (key=value,..) with --target-image-opts\n",
        ),
    )
}

/// The part of `img_convert()` after the option checks: 0 or 1.
fn convert(
    a: &mut Args,
    srcs: &[String],
    out_filename: &str,
    src_mode: crate::common::CacheMode,
) -> i32 {
    let g = graph();
    let mut s = State {
        src: Vec::new(),
        src_nodes: Vec::new(),
        src_sectors: Vec::new(),
        src_alignment: Vec::new(),
        total_sectors: 0,
        allocated_sectors: 0,
        allocated_done: 0,
        status: Status::Data,
        sector_next_status: 0,
        target: None,
        has_zero_init: a.has_zero_init,
        compressed: a.compressed,
        target_is_new: false,
        target_has_backing: false,
        target_backing_sectors: -1,
        copy_range: a.copy_range,
        salvage: a.salvage,
        quiet: a.quiet,
        min_sparse: a.min_sparse,
        alignment: 0,
        cluster_sectors: 0,
        buf_sectors: IO_BUF_SIZE / SECTOR,
    };
    let _ = s.copy_range;

    let mut src_flags = OpenFlags { no_share: true, ..OpenFlags::default() };
    src_mode.apply(&mut src_flags);
    for name in srcs {
        let Some(blk) = img_open(
            a.image_opts,
            name,
            a.fmt.as_deref(),
            src_flags,
            src_mode.writethrough,
            a.force_share,
        ) else {
            return 1;
        };
        let sectors = match blk.getlength() {
            Ok(l) => (l / SECTOR as u64) as i64,
            Err(e) => {
                error_report(&format!("Could not get size of {name}: {}", strerror(&e)));
                return 1;
            }
        };
        let node = root(&blk);
        let req = g.limits(&node).map(|l| l.request_alignment).unwrap_or(1);
        let mut align = u64::from(req).div_ceil(SECTOR as u64) as i64;
        if let Ok(bdi) = g.driver_info(&node) {
            align = align.max((bdi.cluster_size / SECTOR as u64) as i64);
        }
        s.src.push(blk);
        s.src_nodes.push(node);
        s.src_sectors.push(sectors);
        s.src_alignment.push(align);
        s.total_sectors += sectors;
    }

    match &a.snapshot {
        Some(sn @ SnapshotArg::Opts { .. }) => {
            if !load_snapshot(&s.src_nodes[0], sn) {
                return 1;
            }
        }
        Some(sn @ SnapshotArg::IdOrName(_)) => {
            if srcs.len() > 1 {
                error_report("No support for concatenating multiple snapshot");
                return 1;
            }
            if !load_snapshot(&s.src_nodes[0], sn) {
                return 1;
            }
        }
        None => {}
    }

    let out_fmt = a.out_fmt.clone();
    let mut opts: Option<ruvm_qapi::opts::QemuOpts> = None;
    if !a.skip_create {
        let fmt = out_fmt.as_deref().expect("a target format without -n");
        if !tools::format_exists(fmt) {
            error_report(&format!("Unknown file format '{fmt}'"));
            return 1;
        }
        let proto = match tools::find_protocol(out_filename) {
            Ok(p) => p,
            Err(e) => {
                report_error(&e);
                return 1;
            }
        };
        let Some(desc) = tools::create_opts_list(fmt) else {
            error_report(&format!("Format driver '{fmt}' does not support image creation"));
            return 1;
        };
        let Some(pdesc) = tools::create_opts_list(proto) else {
            error_report(&format!("Protocol driver '{proto}' does not support image creation"));
            return 1;
        };
        let mut list: QemuOptsList = opts_append(Some(opts_append(None, desc)), pdesc);
        let mut o = match list.create(None, false) {
            Ok(o) => o.clone(),
            Err(e) => {
                report_error(&e);
                return 1;
            }
        };
        if let Some(options) = &a.options {
            if let Err(e) = o.do_parse(options, None) {
                report_error(&e);
                return 1;
            }
        }
        o.set_number("size", s.total_sectors * SECTOR as i64).expect("every format has size");
        if add_old_style_options(fmt, &mut o, a.out_baseimg.as_deref(), a.backing_fmt.as_deref())
            .is_err()
        {
            return 1;
        }
        opts = Some(o);
    }

    // The backing file name of -o backing_file.
    let out_baseimg_param = opts.as_ref().and_then(|o| o.get("backing_file")).map(str::to_string);
    if let Some(p) = &out_baseimg_param {
        a.out_baseimg = Some(p.clone());
    }
    s.target_has_backing = a.out_baseimg.is_some();

    if s.has_zero_init && s.target_has_backing {
        error_report("Cannot use --target-is-zero when the destination image has a backing file");
        // QEMU jumps out with the status of the last call that worked, which is 0.
        return 0;
    }
    if srcs.len() > 1 && a.out_baseimg.is_some() {
        error_report(
            "Having a backing file for the target makes no sense when concatenating multiple \
             input images",
        );
        return 1;
    }
    if out_baseimg_param.is_some() && opts.as_ref().and_then(|o| o.get("backing_fmt")).is_none() {
        error_report("Use of backing file requires explicit backing format");
        return 1;
    }

    if s.compressed {
        let encryption = opts.as_ref().is_some_and(|o| o.get_bool("encryption", false));
        let encryptfmt = opts.as_ref().and_then(|o| o.get("encrypt.format")).is_some();
        let preallocation = opts.as_ref().and_then(|o| o.get("preallocation"));
        if let Some(fmt) = out_fmt.as_deref().filter(|_| !a.skip_create) {
            if !tools::format_can_compress(fmt) {
                error_report("Compression not supported for this file format");
                return 1;
            }
        }
        if encryption || encryptfmt {
            error_report("Compression and encryption not supported at the same time");
            return 1;
        }
        if preallocation.is_some_and(|p| p != "off") {
            error_report("Compression and preallocation not supported at the same time");
            return 1;
        }
    }

    if a.bitmaps {
        if srcs.len() > 1 {
            error_report("Copying bitmaps only possible with single source");
            return 1;
        }
        if check_bitmaps(&s.src_nodes[0], a.skip_broken).is_err() {
            return 1;
        }
    }

    // The open below needs the secrets, and creating the image uses up the options, so
    // take them out now.
    let mut open_opts = None;
    if let Some(o) = &opts {
        let mut secrets = QDict::new();
        for (k, v) in o.iter() {
            if k.ends_with("key-secret") {
                secrets.put(k, v);
            }
        }
        open_opts = Some(secrets);
        let fmt = out_fmt.as_deref().expect("a target format without -n");
        let mut dict = o.to_qdict();
        if let Err(e) = g.create_image(fmt, out_filename, &mut dict) {
            report_error(&e.prepend(format!("{out_filename}: error while converting {fmt}: ")));
            return 1;
        }
    }
    s.target_is_new = !a.skip_create;

    let Some(mode) = parse_cache_mode(&a.cache) else {
        error_report(&format!("Invalid cache option: {}", a.cache));
        return 1;
    };
    let mut flags = OpenFlags { rdwr: true, unmap: s.min_sparse != 0, ..OpenFlags::default() };
    mode.apply(&mut flags);
    // With O_DIRECT the target may have to grow to the physical sector size.
    if flags.nocache {
        flags.resize = true;
    }
    let target = if a.skip_create {
        img_open(
            a.tgt_image_opts,
            out_filename,
            out_fmt.as_deref(),
            flags,
            mode.writethrough,
            false,
        )
    } else {
        img_open_file(out_filename, open_opts, out_fmt.as_deref(), flags, mode.writethrough, false)
    };
    let Some(target) = target else {
        return 1;
    };
    let out_bs = root(&target);
    s.target = Some(target.clone());
    let details = g.node_details(&out_bs).unwrap_or_default();

    if a.bitmaps && !g.supports_persistent_dirty_bitmap(&out_bs) {
        error_report(&format!("Format driver '{}' does not support bitmaps", details.driver));
        return 1;
    }
    if s.compressed && !g.can_compress(&out_bs) {
        error_report("Compression not supported for this file format");
        return 1;
    }

    // A bigger buffer than 2M when the target wants larger requests, up to 16M.
    let limits = g.limits(&out_bs).unwrap_or_default();
    s.buf_sectors = MAX_BUF_SECTORS.min(
        s.buf_sectors
            .max((limits.opt_transfer as usize >> 9).max(limits.pdiscard_alignment as usize >> 9)),
    );
    // Align the writes to the target to avoid read-modify-write cycles.
    let pow2floor =
        |n: usize| if n == 0 { 0 } else { 1usize << (usize::BITS - 1 - n.leading_zeros()) };
    s.alignment = pow2floor(s.min_sparse).max((limits.request_alignment as usize).div_ceil(SECTOR));
    assert!(s.alignment.is_power_of_two());

    if a.skip_create {
        match target.getlength() {
            Err(e) => {
                error_report(&format!("unable to get output image length: {}", strerror(&e)));
                return 1;
            }
            Ok(l) if ((l / SECTOR as u64) as i64) < s.total_sectors => {
                error_report("output file is smaller than input file");
                return 1;
            }
            Ok(_) => {}
        }
    }

    s.target_backing_sectors = -1;
    if s.target_has_backing && s.target_is_new {
        // Only an optimization, so an error just means the length is not known.
        if let Ok(Some(b)) = g.backing_chain_next(&out_bs) {
            if let Ok(l) = g.node_getlength(&b) {
                s.target_backing_sectors = (l / SECTOR as u64) as i64;
            }
        }
    }

    match g.driver_info(&out_bs) {
        Err(_) => {
            if s.compressed {
                error_report("could not get block driver info");
                return 1;
            }
        }
        Ok(bdi) => {
            s.compressed = s.compressed || bdi.needs_compressed_writes;
            s.cluster_sectors = (bdi.cluster_size / SECTOR as u64) as usize;
        }
    }

    if a.rate_limit != 0 {
        set_rate_limit(&target, a.rate_limit);
    }

    if s.do_copy().is_err() {
        return 1;
    }
    if a.bitmaps && copy_bitmaps(&s.src_nodes[0], &out_bs, a.skip_broken).is_err() {
        return 1;
    }
    0
}
