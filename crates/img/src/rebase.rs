// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img rebase`: changes the backing file of an image, copying whatever differs
//! between the old and the new backing file into the image first unless `-u` is given.

use std::io::ErrorKind;
use std::sync::Arc;

use ruvm_base::error::strerror;
use ruvm_base::report::{error_report, report_error};
use ruvm_block::tools::{self, OpenFlags};
use ruvm_block::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BlockBackend};
use ruvm_qapi::QDict;

use crate::buf::{IO_BUF_SIZE, compare_buffers};
use crate::common::{
    BDRV_DEFAULT_CACHE, OPTION_IMAGE_OPTS, OPTION_OBJECT, Opts, error_exit, graph, img_open, lo,
    object_add, parse_cache_mode, root, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow, progress};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("cache", HasArg::Required, 't'),
    lo("compress", HasArg::No, 'c'),
    lo("backing", HasArg::Required, 'b'),
    lo("backing-format", HasArg::Required, 'B'),
    lo("backing-cache", HasArg::Required, 'T'),
    lo("backing-unsafe", HasArg::No, 'u'),
    lo("force-share", HasArg::No, 'U'),
    lo("progress", HasArg::No, 'p'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

struct Args {
    filename: String,
    fmt: Option<String>,
    cache: String,
    src_cache: String,
    out_baseimg: Option<String>,
    out_basefmt: Option<String>,
    unsafe_: bool,
    compress: bool,
    force_share: bool,
    image_opts: bool,
}

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut cache = BDRV_DEFAULT_CACHE.to_string();
    let mut src_cache = BDRV_DEFAULT_CACHE.to_string();
    let mut out_baseimg = None;
    let mut out_basefmt = None;
    let mut unsafe_ = false;
    let mut compress = false;
    let mut force_share = false;
    let mut progress_on = false;
    let mut quiet = false;
    let mut image_opts = false;
    let mut o = Opts::new(args, "hf:t:cb:F:B:T:uUpq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [-t CACHE]\n",
                        "        [-b BACKING_FILE [-B BACKING_FMT] [-T BACKING_CACHE]] [-u]\n",
                        "        [-c] [-U] [-p] [-q] [--object OBJDEF] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly (default: probing is used)\n",
                        "  --image-opts\n",
                        "     treat FILE as an option string (key=value,..), not a file name\n",
                        "     (incompatible with -f|--format)\n",
                        "  -t, --cache CACHE\n",
                        "     cache mode for FILE (default: writeback)\n",
                        "  -b, --backing BACKING_FILE|\"\"\n",
                        "     rebase onto this file (specify empty name for no backing file)\n",
                        "  -B, --backing-format BACKING_FMT (was -F in <=10.0)\n",
                        "     specify format for BACKING_FILE explicitly (default: probing is \
                         used)\n",
                        "  -T, --backing-cache CACHE\n",
                        "     BACKING_FILE cache mode (default: writeback)\n",
                        "  -u, --backing-unsafe\n",
                        "     do not fail if BACKING_FILE can not be read\n",
                        "  -c, --compress\n",
                        "     compress image (when image supports this)\n",
                        "  -U, --force-share\n",
                        "     open image in shared mode for concurrent access\n",
                        "  -p, --progress\n",
                        "     display progress information\n",
                        "  -q, --quiet\n",
                        "     quiet mode (produce only error messages if any)\n",
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
            't' => cache = arg,
            'b' => out_baseimg = Some(arg),
            // -F is the name of -B up to QEMU 10.0.
            'F' | 'B' => out_basefmt = Some(arg),
            'u' => unsafe_ = true,
            'c' => compress = true,
            'U' => force_share = true,
            'p' => progress_on = true,
            'T' => src_cache = arg,
            'q' => quiet = true,
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    if quiet {
        progress_on = false;
    }
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    if !unsafe_ && out_baseimg.is_none() {
        return Err(error_exit(&o.argv0, "Must specify backing file (-b) or use unsafe mode (-u)"));
    }
    let a = Args {
        filename: rest[0].clone(),
        fmt,
        cache,
        src_cache,
        out_baseimg,
        out_basefmt,
        unsafe_,
        compress,
        force_share,
        image_opts,
    };

    progress::init(progress_on, 2.0);
    progress::print(0.0, 100);
    let ret = rebase(&a);
    progress::end();
    Ok(i32::from(ret != 0))
}

fn rebase(a: &Args) -> i32 {
    let Some(mode) = parse_cache_mode(&a.cache) else {
        error_report(&format!("Invalid cache option: {}", a.cache));
        return -1;
    };
    let mut flags = OpenFlags { rdwr: true, no_backing: a.unsafe_, ..OpenFlags::default() };
    mode.apply(&mut flags);
    let Some(src_mode) = parse_cache_mode(&a.src_cache) else {
        error_report(&format!("Invalid source cache option: {}", a.src_cache));
        return -1;
    };
    let mut src_flags = OpenFlags::default();
    src_mode.apply(&mut src_flags);

    // Open the images. The old backing file is ignored for an unsafe rebase, in case the
    // reference to a renamed or moved backing file is what is being fixed.
    let Some(blk) =
        img_open(a.image_opts, &a.filename, a.fmt.as_deref(), flags, mode.writethrough, false)
    else {
        return -1;
    };
    let g = graph();
    let bs = root(&blk);
    let Ok(unfiltered_bs) = g.skip_filters(&bs) else {
        return -1;
    };
    let unfiltered_bs_cow = g.cow_bs(&unfiltered_bs).ok().flatten();

    if a.compress && !g.can_compress(&unfiltered_bs) {
        error_report("Compression not supported for this file format");
        return -1;
    }

    if let Some(f) = &a.out_basefmt {
        if !tools::format_exists(f) {
            error_report(&format!("Invalid format name: '{f}'"));
            return -1;
        }
    }

    // The overlay subcluster size (or cluster size for compressed writes) keeps the write
    // requests aligned.
    let mut bdi = match g.driver_info(&unfiltered_bs) {
        Ok(i) => i,
        Err(_) => {
            error_report("could not get block driver info");
            return -1;
        }
    };
    if bdi.subcluster_size == 0 {
        bdi.cluster_size = 1;
        bdi.subcluster_size = 1;
    }
    let write_align = if a.compress { bdi.cluster_size } else { bdi.subcluster_size };

    let mut blk_old_backing: Option<Arc<BlockBackend>> = None;
    let mut blk_new_backing: Option<Arc<BlockBackend>> = None;
    let mut prefix_chain_bs = None;

    // A safe rebase compares the old and the new backing file.
    if !a.unsafe_ {
        if let Some(base_bs) = &unfiltered_bs_cow {
            match BlockBackend::new(g, base_bs, BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL) {
                Ok(b) => blk_old_backing = Some(b),
                Err(e) => {
                    let name = g.node_details(base_bs).map(|d| d.filename).unwrap_or_default();
                    report_error(
                        &e.prepend(format!("Could not reuse old backing file '{name}': ")),
                    );
                    return -1;
                }
            }
        }

        let out_baseimg = a.out_baseimg.as_deref().unwrap_or_default();
        if !out_baseimg.is_empty() {
            let mut options = QDict::new();
            if let Some(f) = &a.out_basefmt {
                options.put("driver", f.as_str());
            }
            if a.force_share {
                options.put("force-share", true);
            }
            let overlay_filename = g.exact_or_filename(&bs).unwrap_or_default();
            let out_real_path =
                match tools::full_backing_filename_from_filename(&overlay_filename, out_baseimg) {
                    Ok(p) => p,
                    Err(e) => {
                        report_error(&e.prepend("Could not resolve backing filename: "));
                        return -1;
                    }
                };

            // Is the image rebased onto an image further down its own chain?
            prefix_chain_bs = g.find_backing_image(&bs, &out_real_path);
            if let Some(prefix) = &prefix_chain_bs {
                match BlockBackend::new(g, prefix, BLK_PERM_CONSISTENT_READ, BLK_PERM_ALL) {
                    Ok(b) => blk_new_backing = Some(b),
                    Err(e) => {
                        report_error(
                            &e.prepend(format!("Could not reuse backing file '{out_baseimg}': ")),
                        );
                        return -1;
                    }
                }
            } else {
                match g.blk_new_open(Some(&out_real_path), options, src_flags) {
                    Ok(b) => blk_new_backing = Some(b),
                    Err(e) => {
                        report_error(
                            &e.prepend(format!(
                                "Could not open new backing file '{out_baseimg}': "
                            )),
                        );
                        return -1;
                    }
                }
            }
        }
    }

    // Every cluster the image does not have itself reads from the backing file, so it is
    // compared between the old and the new backing file and copied into the image where
    // they differ. Stopping half way does no harm: the image content stays the same all the
    // time.
    if !a.unsafe_ {
        let ret = copy_differences(
            a,
            &blk,
            &bs,
            &unfiltered_bs,
            unfiltered_bs_cow.as_deref(),
            prefix_chain_bs.as_deref(),
            blk_old_backing.as_deref(),
            blk_new_backing.as_deref(),
            write_align,
        );
        if ret != 0 {
            return ret;
        }
    }

    // Now switch the backing file. What differed has been written into the image, so what
    // the guest sees does not change.
    let out_baseimg = a.out_baseimg.as_deref().filter(|s| !s.is_empty());
    let res = match out_baseimg {
        Some(b) => g.change_backing_file(&unfiltered_bs, Some(b), a.out_basefmt.as_deref(), true),
        None => g.change_backing_file(&unfiltered_bs, None, None, false),
    };
    let shown = a.out_baseimg.as_deref().unwrap_or("(null)");
    let ret = match res {
        Ok(()) => 0,
        Err(e) if e.kind() == ErrorKind::StorageFull => {
            error_report(&format!(
                "Could not change the backing file to '{shown}': No space left in the file \
                 header"
            ));
            -1
        }
        Err(e)
            if e.kind() == ErrorKind::InvalidInput
                && a.out_baseimg.is_some()
                && a.out_basefmt.is_none() =>
        {
            error_report(&format!(
                "Could not change the backing file to '{shown}': backing format must be \
                 specified"
            ));
            -1
        }
        Err(e) => {
            error_report(&format!(
                "Could not change the backing file to '{shown}': {}",
                strerror(&e)
            ));
            -1
        }
    };
    progress::print(100.0, 0);
    // QEMU returns what the backing file change returned even after printing its error.
    ret
}

#[allow(clippy::too_many_arguments, reason = "the locals of img_rebase()")]
fn copy_differences(
    a: &Args,
    blk: &BlockBackend,
    bs: &str,
    unfiltered_bs: &str,
    unfiltered_bs_cow: Option<&str>,
    prefix_chain_bs: Option<&str>,
    blk_old_backing: Option<&BlockBackend>,
    blk_new_backing: Option<&BlockBackend>,
    write_align: u64,
) -> i32 {
    let g = graph();
    let mut buf_old = vec![0u8; IO_BUF_SIZE];
    let mut buf_new = vec![0u8; IO_BUF_SIZE];

    let size = match blk.getlength() {
        Ok(s) => s,
        Err(e) => {
            error_report(&format!("Could not get size of '{}': {}", a.filename, strerror(&e)));
            return -1;
        }
    };
    let mut old_backing_size = 0u64;
    if let Some(old) = blk_old_backing {
        match old.getlength() {
            Ok(s) => old_backing_size = s,
            Err(e) => {
                let name = g.backing_filename(bs).unwrap_or_default();
                error_report(&format!("Could not get size of '{name}': {}", strerror(&e)));
                return -1;
            }
        }
    }
    let mut new_backing_size = 0u64;
    if let Some(new) = blk_new_backing {
        match new.getlength() {
            Ok(s) => new_backing_size = s,
            Err(e) => {
                error_report(&format!(
                    "Could not get size of '{}': {}",
                    a.out_baseimg.as_deref().unwrap_or_default(),
                    strerror(&e)
                ));
                return -1;
            }
        }
    }

    let mut local_progress = 0f32;
    if size != 0 {
        local_progress = 100.0 / (size / size.min(IO_BUF_SIZE as u64)) as f32;
    }

    let mut offset = 0u64;
    while offset < size {
        let mut old_backing_eof = false;

        // How many bytes can the next read handle?
        let mut n = (IO_BUF_SIZE as u64).min(size - offset);

        // Clusters the image has itself need nothing.
        match g.is_allocated(unfiltered_bs, offset, n) {
            Ok((allocated, pnum)) => {
                n = pnum;
                if allocated {
                    offset += n;
                    continue;
                }
            }
            Err(e) => {
                error_report(&format!("error while reading image metadata: {}", strerror(&e)));
                return -1;
            }
        }

        if let Some(prefix) = prefix_chain_bs {
            let bytes = n;
            // Clusters that did not change since the prefix of the chain need nothing.
            let cow = unfiltered_bs_cow.unwrap_or_default();
            match g.is_allocated_above(cow, Some(prefix), false, offset, n) {
                Ok((depth, pnum)) => {
                    n = pnum;
                    if depth == 0 && n != 0 {
                        offset += n;
                        continue;
                    }
                }
                Err(e) => {
                    error_report(&format!("error while reading image metadata: {}", strerror(&e)));
                    return -1;
                }
            }
            if n == 0 {
                // At the end of the old backing file, where the offsets read as zeroes.
                // The cluster is zeroed explicitly to keep that after the rebase.
                n = bytes;
            }
        }

        // [offset, offset + n) is not allocated in the image, but it may not be aligned to
        // the image's (sub)clusters when the old backing file has smaller ones. Widen it to
        // the aligned boundaries so that the writes need no copy on write.
        let aligned = offset / write_align * write_align;
        n += offset - aligned;
        offset = aligned;
        n = (offset + n).div_ceil(write_align) * write_align - offset;
        n = n.min((size - offset).min(IO_BUF_SIZE as u64));

        // Read as much of the old and the new backing file as there is.
        let n_old = n.min(old_backing_size.saturating_sub(offset));
        let n_new = n.min(new_backing_size.saturating_sub(offset));
        let (nu, n_old_u, n_new_u) = (n as usize, n_old as usize, n_new as usize);

        buf_old[n_old_u..nu].fill(0);
        if n_old == 0 {
            old_backing_eof = true;
        } else if blk_old_backing
            .expect("the image has a backing file")
            .pread(offset, &mut buf_old[..n_old_u])
            .is_err()
        {
            error_report("error while reading from old backing file");
            return -1;
        }

        buf_new[n_new_u..nu].fill(0);
        if n_new != 0
            && blk_new_backing
                .expect("there is a new backing file")
                .pread(offset, &mut buf_new[..n_new_u])
                .is_err()
        {
            error_report("error while reading from new backing file");
            return -1;
        }

        // Where they differ, the old content goes into the image.
        let mut written = 0usize;
        while written < nu {
            let (differ, pnum) =
                compare_buffers(&buf_old[written..nu], &buf_new[written..nu], write_align as usize);
            if differ {
                let at = offset + written as u64;
                let res = if old_backing_eof {
                    blk.pwrite_zeroes(at, pnum as u64, false)
                } else if a.compress {
                    blk.pwrite_compressed(at, &buf_old[written..written + pnum])
                } else {
                    blk.pwrite(at, &buf_old[written..written + pnum])
                };
                if let Err(e) = res {
                    error_report(&format!("Error while writing to COW image: {}", strerror(&e)));
                    return -1;
                }
            }
            written += pnum;
            if offset + written as u64 >= old_backing_size {
                old_backing_eof = true;
            }
        }
        progress::print(local_progress, 100);
        offset += n;
    }
    0
}
