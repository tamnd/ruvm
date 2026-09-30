// SPDX-License-Identifier: GPL-2.0-or-later

//! `qemu-img bench`: times a run of read or write requests.
//!
//! QEMU keeps `-d` requests in flight with AIO. Here the requests run one at a time, but in
//! the order QEMU submits them: [`Bench`] walks the state machine of `bench_cb()` with the
//! requests in flight completing first in, first out, so the offsets and the flushes are the
//! same as QEMU's.

use std::collections::VecDeque;
use std::time::Instant;

use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_block::BlockBackend;
use ruvm_block::tools::OpenFlags;

use crate::common::{
    OPTION_FLUSH_INTERVAL, OPTION_IMAGE_OPTS, OPTION_NO_DRAIN, OPTION_OBJECT, OPTION_PATTERN, Opts,
    cvtnum, cvtnum_full, error_exit, img_open, lo, object_add, parse_cache_mode, tryhelp,
};
use crate::getopt::{HasArg, LongOpt};
use crate::{Cmd, Flow};

const LONGS: &[LongOpt] = &[
    lo("help", HasArg::No, 'h'),
    lo("format", HasArg::Required, 'f'),
    lo("image-opts", HasArg::No, OPTION_IMAGE_OPTS),
    lo("cache", HasArg::Required, 't'),
    lo("count", HasArg::Required, 'c'),
    lo("depth", HasArg::Required, 'd'),
    lo("offset", HasArg::Required, 'o'),
    lo("buffer-size", HasArg::Required, 's'),
    lo("step-size", HasArg::Required, 'S'),
    lo("write", HasArg::No, 'w'),
    lo("pattern", HasArg::Required, OPTION_PATTERN),
    lo("flush-interval", HasArg::Required, OPTION_FLUSH_INTERVAL),
    lo("no-drain", HasArg::No, OPTION_NO_DRAIN),
    lo("aio", HasArg::Required, 'i'),
    lo("native", HasArg::No, 'n'),
    lo("force-share", HasArg::No, 'U'),
    lo("quiet", HasArg::No, 'q'),
    lo("object", HasArg::Required, OPTION_OBJECT),
];

const INT_MAX: i64 = i32::MAX as i64;

pub(crate) fn run(cmd: &Cmd, args: Vec<String>) -> Flow<i32> {
    let mut fmt = None;
    let mut image_opts = false;
    let mut is_write = false;
    let mut count = 75000i64;
    let mut depth = 64i64;
    let mut offset = 0i64;
    let mut bufsize = 4096i64;
    let mut pattern = 0i64;
    let mut step = 0i64;
    let mut flush_interval = 0i64;
    let mut drain_on_flush = true;
    let mut flags = OpenFlags::default();
    let mut writethrough = false;
    let mut force_share = false;
    let mut o = Opts::new(args, "hf:t:c:d:o:s:S:wi:nUq", LONGS);
    while let Some((c, arg)) = o.next()? {
        match c {
            'h' => {
                return Err(cmd.help(
                    concat!(
                        "[-f FMT | --image-opts] [-t CACHE]\n",
                        "        [-c COUNT] [-d DEPTH] [-o OFFSET] [-s BUFFER_SIZE] [-S \
                         STEP_SIZE]\n",
                        "        [-w [--pattern PATTERN] [--flush-interval INTERVAL \
                         [--no-drain]]]\n",
                        "        [-i AIO] [-n] [-U] [-q] FILE\n",
                    ),
                    concat!(
                        "  -f, --format FMT\n",
                        "     specify FILE format explicitly\n",
                        "  --image-opts\n",
                        "     indicates that FILE is a complete image specification\n",
                        "     instead of a file name (incompatible with --format)\n",
                        "  -t, --cache CACHE\n",
                        "     cache mode for FILE (default: writeback)\n",
                        "  -c, --count COUNT\n",
                        "     number of I/O requests to perform\n",
                        "  -d, --depth DEPTH\n",
                        "     number of requests to perform in parallel\n",
                        "  -o, --offset OFFSET\n",
                        "     start first request at this OFFSET\n",
                        "  -s, --buffer-size BUFFER_SIZE[bkKMGTPE]\n",
                        "     size of each I/O request, with optional multiplier suffix\n",
                        "     (powers of 1024, default is 4K)\n",
                        "  -S, --step-size STEP_SIZE[bkKMGTPE]\n",
                        "     each next request offset increment, with optional multiplier \
                         suffix\n",
                        "     (powers of 1024, default is the same as BUFFER_SIZE)\n",
                        "  -w, --write\n",
                        "     perform write test (default is read)\n",
                        "  --pattern PATTERN\n",
                        "     write this pattern byte instead of zero\n",
                        "  --flush-interval FLUSH_INTERVAL\n",
                        "     issue flush after this number of requests\n",
                        "  --no-drain\n",
                        "     do not wait when flushing pending requests\n",
                        "  -i, --aio AIO\n",
                        "     async-io backend (threads, native, io_uring)\n",
                        "  -n, --native\n",
                        "     use native AIO backend if possible\n",
                        "  -U, --force-share\n",
                        "     open images in shared mode for concurrent access\n",
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
            't' => match parse_cache_mode(&arg) {
                Some(mode) => {
                    mode.apply(&mut flags);
                    writethrough = mode.writethrough;
                }
                None => {
                    error_report("Invalid cache mode");
                    return Ok(1);
                }
            },
            'c' => match cvtnum_full("request count", &arg, false, 1, INT_MAX) {
                Some(v) => count = v,
                None => return Ok(1),
            },
            'd' => match cvtnum_full("queue depth", &arg, false, 1, INT_MAX) {
                Some(v) => depth = v,
                None => return Ok(1),
            },
            'n' => flags.native_aio = true,
            // bdrv_parse_aio(); io_uring only exists in Linux builds with liburing, which
            // this is not.
            'i' => match arg.as_str() {
                "threads" => flags.native_aio = false,
                "native" => flags.native_aio = true,
                _ => {
                    error_report(&format!("Invalid aio option: {arg}"));
                    return Ok(1);
                }
            },
            'o' => match cvtnum("offset", &arg, true) {
                Some(v) => offset = v,
                None => return Ok(1),
            },
            's' => match cvtnum_full("buffer size", &arg, true, 1, INT_MAX) {
                Some(v) => bufsize = v,
                None => return Ok(1),
            },
            'S' => match cvtnum_full("step size", &arg, true, 0, INT_MAX) {
                Some(v) => step = v,
                None => return Ok(1),
            },
            'w' => {
                flags.rdwr = true;
                is_write = true;
            }
            OPTION_PATTERN => match cvtnum_full("pattern byte", &arg, false, 0, 0xff) {
                Some(v) => pattern = v,
                None => return Ok(1),
            },
            OPTION_FLUSH_INTERVAL => match cvtnum_full("flush interval", &arg, false, 0, INT_MAX) {
                Some(v) => flush_interval = v,
                None => return Ok(1),
            },
            OPTION_NO_DRAIN => drain_on_flush = false,
            'U' => force_share = true,
            // Nothing is printed besides the results anyway.
            'q' => {}
            OPTION_OBJECT => object_add(&arg)?,
            _ => return Err(tryhelp(&o.argv0)),
        }
    }
    let rest = o.rest();
    if rest.len() != 1 {
        return Err(error_exit(&o.argv0, "Expecting one image file name"));
    }
    let filename = &rest[0];

    if !is_write && flush_interval != 0 {
        error_report("--flush-interval is only available in write tests");
        return Ok(1);
    }
    if flush_interval != 0 && flush_interval < depth {
        error_report("Flush interval can't be smaller than depth");
        return Ok(1);
    }

    let Some(blk) =
        img_open(image_opts, filename, fmt.as_deref(), flags, writethrough, force_share)
    else {
        return Ok(1);
    };
    let Ok(image_size) = blk.getlength() else {
        return Ok(1);
    };

    let mut b = Bench {
        blk: &blk,
        image_size,
        write: is_write,
        bufsize: bufsize as u64,
        step: if step != 0 { step as u64 } else { bufsize as u64 },
        nrreq: depth,
        n: count,
        flush_interval,
        drain_on_flush,
        buf: vec![pattern as u8; bufsize as usize],
        in_flight: 0,
        in_flush: false,
        offset: offset as u64,
        queue: VecDeque::new(),
    };
    println!(
        "Sending {} {} requests, {} bytes each, {} in parallel (starting at offset {}, step \
         size {})",
        b.n,
        if b.write { "write" } else { "read" },
        b.bufsize,
        b.nrreq,
        b.offset,
        b.step
    );
    if flush_interval != 0 {
        println!("Sending flush every {flush_interval} requests");
    }

    let t1 = Instant::now();
    if b.run().is_err() {
        return Ok(1);
    }
    println!("Run completed in {:3.3} seconds.", t1.elapsed().as_secs_f64());
    Ok(0)
}

/// A request in flight.
enum Op {
    Io(u64),
    /// A flush, and whether `bench_cb()` or `bench_undrained_flush_cb()` completes it.
    Flush(bool),
}

/// `BenchData`.
struct Bench<'a> {
    blk: &'a BlockBackend,
    image_size: u64,
    write: bool,
    bufsize: u64,
    step: u64,
    nrreq: i64,
    n: i64,
    flush_interval: i64,
    drain_on_flush: bool,
    buf: Vec<u8>,
    in_flight: i64,
    in_flush: bool,
    offset: u64,
    queue: VecDeque<Op>,
}

impl Bench<'_> {
    fn run(&mut self) -> Result<(), ()> {
        self.bench_cb();
        while self.n > 0 {
            let op = self.queue.pop_front().expect("requests are in flight while n > 0");
            let res = match op {
                Op::Io(offset) => {
                    let mut buf = std::mem::take(&mut self.buf);
                    let r = if self.write {
                        self.blk.pwrite(offset, &buf)
                    } else {
                        self.blk.pread(offset, &mut buf)
                    };
                    self.buf = buf;
                    r
                }
                Op::Flush(_) => self.blk.flush(),
            };
            match (op, res) {
                (Op::Flush(false), Err(e)) => {
                    error_report(&format!("Failed flush request: {}", strerror(&e)));
                    return Err(());
                }
                (Op::Flush(false), Ok(())) => {}
                (_, Err(e)) => {
                    error_report(&format!("Failed request: {}", strerror(&e)));
                    return Err(());
                }
                (_, Ok(())) => self.bench_cb(),
            }
        }
        Ok(())
    }

    /// `bench_cb()` after a request that succeeded.
    fn bench_cb(&mut self) {
        if self.in_flush {
            // A flush with the queue drained finished: start the next requests.
            assert_eq!(self.in_flight, 0);
            self.in_flush = false;
        } else if self.in_flight > 0 {
            let remaining = self.n - self.in_flight;
            self.n -= 1;
            self.in_flight -= 1;

            // Time for a flush? Drain the queue if asked to, then flush.
            if self.flush_interval != 0 && remaining % self.flush_interval == 0 {
                if self.in_flight == 0 || !self.drain_on_flush {
                    if self.drain_on_flush {
                        self.in_flush = true;
                    }
                    self.queue.push_back(Op::Flush(self.drain_on_flush));
                }
                if self.drain_on_flush {
                    return;
                }
            }
        }

        while self.n > self.in_flight && self.in_flight < self.nrreq {
            let offset = self.offset;
            self.in_flight += 1;
            self.offset += self.step;
            if self.image_size <= self.bufsize {
                self.offset = 0;
            } else {
                self.offset %= self.image_size - self.bufsize;
            }
            self.queue.push_back(Op::Io(offset));
        }
    }
}
