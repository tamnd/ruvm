// SPDX-License-Identifier: GPL-2.0-or-later

//! `blkverify` from block/blkverify.c: a filter that sends every read and write to two
//! children, the image under test (`test`) and a raw copy of it (`raw`), and checks that both
//! give the same result.
//!
//! On a mismatch QEMU prints what went wrong to stderr and exits with status 1:
//!
//! ```text
//! blkverify: read offset=512 bytes=512 contents mismatch at offset 700
//! blkverify: write offset=0 bytes=512 return value mismatch -5 != 0
//! ```
//!
//! This port does the same. Reads give the data of `test`, lengths come from `test`, and a
//! flush only flushes `test`, as in QEMU.
//!
//! Differences from QEMU:
//!
//! - The file names in `blkverify:raw:image` are opened with the protocol driver the name
//!   picks, without format probing. Give the children as options (`raw`, `test`) to put a
//!   format driver on top of them.
//! - The two requests run one after the other, `test` first, not in parallel coroutines.
//! - Contents are only compared when both reads succeed.
//! - `bdrv_dirname()` and `bdrv_recurse_can_replace()` have no counterpart here yet.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::QDict;
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::filter::blkdebug::put_filename_child;
use crate::node::{
    BDRV_CHILD_DATA, BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_WRITE_UNCHANGED, Driver,
    Node,
};

/// `bdrv_blkverify`.
pub(crate) static BLKVERIFY: DriverDef = DriverDef::filter("blkverify", blkverify_open)
    .with_protocol("blkverify")
    .with_parse_filename(blkverify_parse_filename);

struct BlkverifyDriver {
    /// Whether a mismatch ends the process, as in QEMU. Tests turn this off to get `EIO`.
    exit_on_mismatch: bool,
}

/// `blkverify_parse_filename()`: `blkverify:raw:image`. Without the prefix the whole name is
/// the image under test.
fn blkverify_parse_filename(filename: &str, options: &mut QDict) -> Result<()> {
    let Some(rest) = filename.strip_prefix("blkverify:") else {
        return put_filename_child(options, "test", filename);
    };
    let Some(c) = rest.find(':') else {
        return Err(Error::generic("blkverify requires raw copy and original image path"));
    };
    put_filename_child(options, "raw", &rest[..c])?;
    put_filename_child(options, "test", &rest[c + 1..])
}

/// `blkverify_open()`.
fn blkverify_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Blkverify(o) = opts else {
        unreachable!("the blkverify driver gets blkverify options");
    };
    // The raw file first, then the image under test.
    args.open_child(*o.raw, "raw", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;
    args.open_child(*o.test, "test", BDRV_CHILD_DATA)?;
    Ok(Box::new(BlkverifyDriver { exit_on_mismatch: true }))
}

/// A request as `blkverify_err()` describes it.
struct Req {
    is_write: bool,
    offset: u64,
    bytes: u64,
}

/// The `-errno` QEMU's request functions return, 0 for success.
fn ret_of(r: &io::Result<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => -e.raw_os_error().unwrap_or(libc::EIO),
    }
}

/// The line `blkverify_err()` prints, without the newline.
fn mismatch_message(r: &Req, what: &str) -> String {
    format!(
        "blkverify: {} offset={} bytes={} {what}",
        if r.is_write { "write" } else { "read" },
        r.offset,
        r.bytes
    )
}

impl BlkverifyDriver {
    /// `blkverify_err()`.
    fn err(&self, r: &Req, what: &str) -> io::Error {
        eprintln!("{}", mismatch_message(r, what));
        if self.exit_on_mismatch {
            std::process::exit(1);
        }
        io::Error::from_raw_os_error(libc::EIO)
    }

    fn test(bs: &Node) -> std::sync::Arc<Node> {
        bs.child("test").expect("blkverify has a test child").node
    }

    /// The end of `blkverify_co_prwv()`: both requests must have the same result.
    fn check_ret(&self, r: &Req, test: &io::Result<()>, raw: &io::Result<()>) -> io::Result<()> {
        let (a, b) = (ret_of(test), ret_of(raw));
        if a != b {
            return Err(self.err(r, &format!("return value mismatch {a} != {b}")));
        }
        match test {
            Ok(()) => Ok(()),
            Err(e) => Err(io::Error::new(e.kind(), e.to_string())),
        }
    }
}

impl Driver for BlkverifyDriver {
    /// `blkverify_co_preadv()`.
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        let r = Req { is_write: false, offset, bytes: buf.len() as u64 };
        let mut raw_buf = vec![0u8; buf.len()];
        let test = Self::test(bs).pread(offset, buf);
        let raw = bs.file().pread(offset, &mut raw_buf);
        self.check_ret(&r, &test, &raw)?;
        if let Some(i) = buf.iter().zip(&raw_buf).position(|(a, b)| a != b) {
            return Err(self.err(&r, &format!("contents mismatch at offset {}", offset + i as u64)));
        }
        Ok(())
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    /// `blkverify_co_pwritev()`.
    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        let r = Req { is_write: true, offset, bytes: buf.len() as u64 };
        let test = Self::test(bs).pwrite_flags(offset, buf, flags);
        let raw = bs.file().pwrite_flags(offset, buf, flags);
        self.check_ret(&r, &test, &raw)
    }

    fn supported_write_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED
    }

    fn supported_zero_flags(&self) -> u32 {
        BDRV_REQ_WRITE_UNCHANGED
    }

    fn has_pwrite_zeroes(&self) -> bool {
        false
    }

    /// `blkverify_co_flush()`: only the image under test, the raw file does not matter.
    fn flush_to_disk(&self, bs: &Node) -> io::Result<()> {
        Self::test(bs).flush()
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        Self::test(bs).getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    /// `blkverify_refresh_filename()`.
    fn exact_filename(&self, bs: &Node) -> Option<String> {
        let raw = bs.file().meta.lock().unwrap().exact_filename.clone();
        let test = Self::test(bs).meta.lock().unwrap().exact_filename.clone();
        if raw.is_empty() || test.is_empty() {
            return None;
        }
        let f = format!("blkverify:{raw}:{test}");
        // An overflow makes the file name unusable, so there is none.
        (f.len() < 4096).then_some(f)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ruvm_qapi::QValue;

    use super::*;
    use crate::filter::copy_on_read::tests::{Mem, mem, mem_node};
    use crate::node::{NodeFlags, NodeMeta, NodeSpec};

    fn verify_node(raw: &Arc<Node>, test: &Arc<Node>) -> Arc<Node> {
        Node::build(NodeSpec {
            name: "v".into(),
            driver_name: "blkverify",
            driver: Box::new(BlkverifyDriver { exit_on_mismatch: false }),
            def: Some(&BLKVERIFY),
            flags: NodeFlags::default(),
            meta: NodeMeta::default(),
            children: vec![
                ("raw".into(), raw.clone(), BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY),
                ("test".into(), test.clone(), BDRV_CHILD_DATA),
            ],
        })
        .unwrap()
    }

    #[test]
    fn detects_mismatch() {
        let raw = mem_node("raw", Mem::with_data(vec![5; 4096], true), None);
        let test = mem_node("test", Mem::with_data(vec![5; 4096], true), None);
        let v = verify_node(&raw, &test);

        let mut buf = [0u8; 1024];
        v.pread(0, &mut buf).unwrap();
        assert_eq!(buf, [5u8; 1024]);

        // Writes reach both children.
        v.pwrite(512, &[9u8; 512]).unwrap();
        assert_eq!(mem(&raw).data.lock().unwrap()[512], 9);
        assert_eq!(mem(&test).data.lock().unwrap()[512], 9);

        // Now make the image under test differ.
        mem(&test).data.lock().unwrap()[700] = 1;
        let e = v.pread(512, &mut [0u8; 512]).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EIO));
        assert_eq!(v.getlength().unwrap(), 4096);
    }

    #[test]
    fn messages() {
        let r = Req { is_write: false, offset: 512, bytes: 512 };
        assert_eq!(
            mismatch_message(&r, "contents mismatch at offset 700"),
            "blkverify: read offset=512 bytes=512 contents mismatch at offset 700"
        );
        let d = BlkverifyDriver { exit_on_mismatch: false };
        let w = Req { is_write: true, offset: 0, bytes: 512 };
        let e =
            d.check_ret(&w, &Err(io::Error::from_raw_os_error(libc::EIO)), &Ok(())).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(libc::EIO));
        assert_eq!(
            mismatch_message(&w, "return value mismatch -5 != 0"),
            "blkverify: write offset=0 bytes=512 return value mismatch -5 != 0"
        );
    }

    #[test]
    fn parse_filename_and_open() {
        let mut d = QDict::new();
        blkverify_parse_filename("null-aio://", &mut d).unwrap();
        let test = d.get("test").and_then(QValue::as_dict).unwrap();
        assert_eq!(test.get_str("driver"), Some("null-aio"));
        assert!(!d.contains_key("raw"));
        #[cfg(unix)]
        {
            let mut d = QDict::new();
            blkverify_parse_filename("blkverify:/tmp/raw.img:null-co://", &mut d).unwrap();
            let raw = d.get("raw").and_then(QValue::as_dict).unwrap();
            assert_eq!(raw.get_str("driver"), Some("file"));
            assert_eq!(raw.get_str("filename"), Some("/tmp/raw.img"));
            let test = d.get("test").and_then(QValue::as_dict).unwrap();
            assert_eq!(test.get_str("driver"), Some("null-co"));
        }
        let e = blkverify_parse_filename("blkverify:x", &mut QDict::new()).unwrap_err();
        assert_eq!(e.message(), "blkverify requires raw copy and original image path");

        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "blkverify");
        o.put("read-zeroes", "on");
        o.put("raw.driver", "null-co");
        o.put("test.driver", "null-co");
        let e = g.open_image(None, o).unwrap_err();
        // `read-zeroes` belongs to the children, not to blkverify.
        assert!(e.message().contains("read-zeroes"), "{}", e.message());

        let mut o = QDict::new();
        o.put("driver", "blkverify");
        o.put("raw.read-zeroes", "on");
        o.put("test.read-zeroes", "on");
        o.put("raw.driver", "null-co");
        o.put("test.driver", "null-co");
        o.put("node-name", "v0");
        let name = g.open_image(None, o).unwrap();
        let v = g.find_node(&name).unwrap();
        assert_eq!(v.driver_name, "blkverify");
        let mut buf = [1u8; 512];
        v.pread(0, &mut buf).unwrap();
        assert_eq!(buf, [0u8; 512]);
    }
}
