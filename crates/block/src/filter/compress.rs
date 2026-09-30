// SPDX-License-Identifier: GPL-2.0-or-later

//! `compress` from block/filter-compress.c: a filter that turns every write into a compressed
//! write on its child, whose format must support compressed writes.
//!
//! Difference from QEMU: whether the child's format can compress is what its driver says in
//! [`Driver::can_compress`], QEMU's `block_driver_can_compress()`.

use std::io;

use ruvm_base::{Error, Result};
use ruvm_qapi::types::BlockdevOptionsU;

use crate::drivers::{DriverDef, OpenArgs};
use crate::node::{
    BDRV_CHILD_FILTERED, BDRV_CHILD_PRIMARY, BDRV_REQ_FUA, BDRV_REQ_MAY_UNMAP,
    BDRV_REQ_NO_FALLBACK, BDRV_REQ_WRITE_COMPRESSED, BDRV_REQ_WRITE_UNCHANGED, BlockLimits, Driver,
    Node,
};

/// `bdrv_compress`.
pub(crate) static COMPRESS: DriverDef = DriverDef::filter("compress", compress_open);

struct CompressDriver {
    /// `bs->supported_write_flags`.
    write_flags: u32,
    /// `bs->supported_zero_flags`.
    zero_flags: u32,
}

/// The check of `compress_open()`: the child's driver must take compressed writes.
fn check_can_compress(file: &Node) -> Result<()> {
    if !file.driver.can_compress() {
        return Err(Error::generic(format!(
            "Compression is not supported for underlying format: {}",
            file.driver_name
        )));
    }
    Ok(())
}

impl CompressDriver {
    fn new(file: &Node) -> Self {
        CompressDriver {
            write_flags: BDRV_REQ_WRITE_UNCHANGED
                | (BDRV_REQ_FUA & file.driver.supported_write_flags()),
            zero_flags: BDRV_REQ_WRITE_UNCHANGED
                | ((BDRV_REQ_FUA | BDRV_REQ_MAY_UNMAP | BDRV_REQ_NO_FALLBACK)
                    & file.driver.supported_zero_flags()),
        }
    }
}

/// `compress_open()`.
fn compress_open(args: &mut OpenArgs<'_>, opts: BlockdevOptionsU) -> Result<Box<dyn Driver>> {
    let BlockdevOptionsU::Compress(o) = opts else {
        unreachable!("the compress driver gets compress options");
    };
    let file = args.open_child(*o.file, "file", BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)?;
    check_can_compress(&file)?;
    Ok(Box::new(CompressDriver::new(&file)))
}

impl Driver for CompressDriver {
    fn pread(&self, bs: &Node, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        bs.file().pread(offset, buf)
    }

    fn pwrite(&self, bs: &Node, offset: u64, buf: &[u8]) -> io::Result<()> {
        self.pwrite_flags(bs, offset, buf, 0)
    }

    fn pwrite_flags(&self, bs: &Node, offset: u64, buf: &[u8], flags: u32) -> io::Result<()> {
        bs.file().pwrite_flags(offset, buf, flags | BDRV_REQ_WRITE_COMPRESSED)
    }

    fn supported_write_flags(&self) -> u32 {
        self.write_flags
    }

    fn pwrite_zeroes(&self, bs: &Node, offset: u64, bytes: u64, may_unmap: bool) -> io::Result<()> {
        self.pwrite_zeroes_flags(bs, offset, bytes, if may_unmap { BDRV_REQ_MAY_UNMAP } else { 0 })
    }

    fn pwrite_zeroes_flags(
        &self,
        bs: &Node,
        offset: u64,
        bytes: u64,
        flags: u32,
    ) -> io::Result<()> {
        bs.file().pwrite_zeroes_flags(offset, bytes, flags)
    }

    fn supported_zero_flags(&self) -> u32 {
        self.zero_flags
    }

    fn pdiscard(&self, bs: &Node, offset: u64, bytes: u64) -> io::Result<()> {
        bs.file().pdiscard(offset, bytes)
    }

    fn getlength(&self, bs: &Node) -> io::Result<u64> {
        bs.file().getlength()
    }

    fn has_truncate(&self) -> bool {
        false
    }

    /// `compress_refresh_limits()`: requests cover whole clusters of the child.
    fn refresh_limits(&self, bs: &Node, bl: &mut BlockLimits) -> Result<()> {
        let Some(file) = bs.filter_child() else {
            return Ok(());
        };
        if let Ok(bdi) = file.node.get_info() {
            if bdi.cluster_size != 0 {
                bl.request_alignment = bdi.cluster_size as u32;
            }
        }
        Ok(())
    }

    fn eject(&self, bs: &Node, eject_flag: bool) {
        let file = bs.file();
        file.driver.eject(&file, eject_flag);
    }

    fn lock_medium(&self, bs: &Node, locked: bool) {
        let file = bs.file();
        file.driver.lock_medium(&file, locked);
    }
}

#[cfg(test)]
mod tests {
    use ruvm_qapi::QDict;

    use super::*;
    use crate::filter::copy_on_read::tests::{Mem, mem, mem_node};
    use crate::node::{NodeFlags, NodeMeta, NodeSpec};

    #[test]
    fn refuses_formats_without_compression() {
        let g = crate::graph::BlockGraph::new();
        let mut o = QDict::new();
        o.put("driver", "compress");
        o.put("file.driver", "null-co");
        let e = g.open_image(None, o).unwrap_err();
        assert_eq!(e.message(), "Compression is not supported for underlying format: null-co");

        let plain = mem_node("plain", Mem::with_data(vec![0; 4096], true), None);
        let e = check_can_compress(&plain).unwrap_err();
        assert_eq!(e.message(), "Compression is not supported for underlying format: test-mem");
    }

    #[test]
    fn writes_are_compressed() {
        let m = Mem { compress: true, ..Mem::with_data(vec![0; 8192], true) };
        let file = mem_node("file", m, None);
        check_can_compress(&file).unwrap();
        let bs = Node::build(NodeSpec {
            name: "c".into(),
            driver_name: "compress",
            driver: Box::new(CompressDriver::new(&file)),
            def: Some(&COMPRESS),
            flags: NodeFlags::default(),
            meta: NodeMeta::default(),
            children: vec![("file".into(), file.clone(), BDRV_CHILD_FILTERED | BDRV_CHILD_PRIMARY)],
        })
        .unwrap();
        bs.pwrite(4096, &[7u8; 4096]).unwrap();
        assert_eq!(*mem(&file).compressed.lock().unwrap(), vec![(4096, 4096)]);
        let mut buf = [0u8; 512];
        bs.pread(4096, &mut buf).unwrap();
        assert_eq!(buf, [7u8; 512]);
        assert_eq!(bs.getlength().unwrap(), 8192);
    }
}
