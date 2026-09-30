// SPDX-License-Identifier: GPL-2.0-or-later

//! `null-co` and `null-aio` from block/null.c: a node of a given size that throws writes away
//! and reads as zeroes when `read-zeroes` is on (and as whatever the buffer held otherwise).
//!
//! Both drivers behave the same here: requests are synchronous, so the difference between the
//! coroutine and the AIO callback versions does not show. `latency-ns` sleeps the calling
//! thread.

use std::io;
use std::time::Duration;

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::{BlockdevOptionsNull, BlockdevOptionsU};

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_BLOCK_OFFSET_VALID, BDRV_BLOCK_ZERO, BDRV_REQ_FUA, BlockStatus, Driver, Node, ReopenState,
};

/// `bdrv_null_co`.
pub(crate) static NULL_CO: DriverDef = DriverDef::protocol("null-co", "null-co", null_open)
    .with_parse_filename(null_co_parse_filename)
    .with_strong_opts(NULL_STRONG_OPTS);

/// `bdrv_null_aio`.
pub(crate) static NULL_AIO: DriverDef = DriverDef::protocol("null-aio", "null-aio", null_open)
    .with_parse_filename(null_aio_parse_filename)
    .with_strong_opts(NULL_STRONG_OPTS);

/// `null_strong_runtime_opts`.
const NULL_STRONG_OPTS: &[&str] = &["size", "read-zeroes"];

/// `BDRVNullState`.
#[derive(Debug)]
pub(crate) struct NullDriver {
    length: u64,
    latency_ns: u64,
    read_zeroes: bool,
    format_name: &'static str,
    /// Whether the node was opened with options besides the ones that do not matter for its
    /// file name, see `null_refresh_filename()`.
    plain: bool,
}

fn null_co_parse_filename(filename: &str, _options: &mut QDict) -> Result<()> {
    if filename != "null-co://" {
        return Err(Error::generic("The only allowed filename for this driver is 'null-co://'"));
    }
    Ok(())
}

fn null_aio_parse_filename(filename: &str, _options: &mut QDict) -> Result<()> {
    if filename != "null-aio://" {
        return Err(Error::generic("The only allowed filename for this driver is 'null-aio://'"));
    }
    Ok(())
}

fn null_open(_args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let (o, name) = match opts {
        BlockdevOptionsU::NullCo(o) => (o, "null-co"),
        BlockdevOptionsU::NullAio(o) => (o, "null-aio"),
        _ => unreachable!("the null drivers get null options"),
    };
    Ok(Box::new(NullDriver::open(&o, name)?))
}

impl NullDriver {
    /// `null_open()`.
    pub(crate) fn open(o: &BlockdevOptionsNull, format_name: &'static str) -> Result<NullDriver> {
        // QEMU reads latency-ns as a signed number, so the top half of the range is negative.
        if o.latency_ns.is_some_and(|l| l > i64::MAX as u64) {
            return Err(Error::generic("latency-ns is invalid"));
        }
        Ok(NullDriver {
            length: o.size.unwrap_or(1 << 30).max(0) as u64,
            latency_ns: o.latency_ns.unwrap_or(0),
            read_zeroes: o.read_zeroes.unwrap_or(false),
            format_name,
            plain: o.size.is_none() && o.read_zeroes.is_none(),
        })
    }

    /// `null_co_common()`.
    fn common(&self) -> io::Result<()> {
        if self.latency_ns != 0 {
            std::thread::sleep(Duration::from_nanos(self.latency_ns));
        }
        Ok(())
    }
}

impl Driver for NullDriver {
    fn pread(&self, _bs: &Node, _offset: u64, buf: &mut [u8]) -> io::Result<()> {
        if self.read_zeroes {
            buf.fill(0);
        }
        self.common()
    }

    fn pwrite(&self, _bs: &Node, _offset: u64, _buf: &[u8]) -> io::Result<()> {
        self.common()
    }

    fn supported_write_flags(&self) -> u32 {
        BDRV_REQ_FUA
    }

    fn supported_zero_flags(&self) -> u32 {
        0
    }

    /// null.c has no `.bdrv_co_pwrite_zeroes`, so zero writes become writes of a zeroed buffer.
    fn pwrite_zeroes(&self, _: &Node, _: u64, _: u64, _: bool) -> io::Result<()> {
        Err(crate::node::errno(libc::ENOTSUP))
    }

    fn has_pwrite_zeroes(&self) -> bool {
        false
    }

    fn flush_to_disk(&self, _bs: &Node) -> io::Result<()> {
        self.common()
    }

    fn getlength(&self, _bs: &Node) -> io::Result<u64> {
        Ok(self.length)
    }

    fn get_allocated_file_size(&self, _bs: &Node) -> Option<io::Result<u64>> {
        Some(Ok(0))
    }

    fn has_truncate(&self) -> bool {
        false
    }

    fn reopen_prepare(&self, _bs: &Node, _state: &mut ReopenState) -> Option<Result<()>> {
        Some(Ok(()))
    }

    fn block_status(
        &self,
        bs: &Node,
        _want: u32,
        offset: u64,
        bytes: u64,
    ) -> Option<io::Result<BlockStatus>> {
        let mut ret = BDRV_BLOCK_OFFSET_VALID;
        if self.read_zeroes {
            ret |= BDRV_BLOCK_ZERO;
        }
        Some(Ok(BlockStatus { ret, pnum: bytes, map: offset, file: Some(bs.arc()) }))
    }

    fn exact_filename(&self, _bs: &Node) -> Option<String> {
        self.plain.then(|| format!("{}://", self.format_name))
    }
}
