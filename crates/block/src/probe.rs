// SPDX-License-Identifier: GPL-2.0-or-later

//! Format probing: `bdrv_probe_all()` and `find_image_format()` from block.c.
//!
//! Every registered driver with a `.bdrv_probe` scores the first bytes of the image, and the
//! highest score wins. `raw` always scores 1, so it wins when nothing else claims the data.
//!
//! Difference from QEMU: a QEMU binary built without, say, the vmdk driver probes a vmdk
//! image as raw. Here the magic numbers of the formats QEMU builds by default are known even
//! when their driver is not in this build, and such an image is refused with "Driver '%s' is
//! not supported yet" rather than handed to the guest as raw.

use ruvm_base::{Error, Result};

use crate::drivers::{self, DriverDef};
use crate::node::Node;

/// `BLOCK_PROBE_BUF_SIZE`.
pub(crate) const BLOCK_PROBE_BUF_SIZE: usize = 512;

/// The probes of the formats QEMU builds by default, by magic number, with the scores their
/// `.bdrv_probe` functions give.
fn known_format(buf: &[u8], filename: Option<&str>) -> Option<(&'static str, i32)> {
    let be32 =
        |off: usize| buf.get(off..off + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let le32 =
        |off: usize| buf.get(off..off + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let starts = |magic: &[u8]| buf.starts_with(magic);
    if starts(b"QFI\xfb") {
        return match be32(4) {
            Some(1) => Some(("qcow", 100)),
            Some(v) if v >= 2 => Some(("qcow2", 100)),
            _ => None,
        };
    }
    if starts(b"QED\0") {
        return Some(("qed", 100));
    }
    if le32(0x40) == Some(0xbeda_107f) {
        return Some(("vdi", 100));
    }
    if starts(b"KDMV") || starts(b"COWD") {
        return Some(("vmdk", 100));
    }
    if starts(b"# Disk DescriptorFile") {
        return Some(("vmdk", 100));
    }
    if starts(b"vhdxfile") {
        return Some(("vhdx", 100));
    }
    if starts(b"conectix") {
        return Some(("vpc", 100));
    }
    if starts(b"LUKS\xba\xbe") {
        return Some(("luks", 100));
    }
    if starts(b"Bochs Virtual HD Image") {
        return Some(("bochs", 100));
    }
    if starts(b"WithoutFreeSpace") || starts(b"WithouFreSpacExt") {
        return Some(("parallels", 100));
    }
    if starts(b"#!/bin/sh\n#V2.0 Format\n") {
        return Some(("cloop", 2));
    }
    // dmg_probe(): only by the file name.
    if filename.is_some_and(|f| f.len() > 4 && f.ends_with(".dmg")) {
        return Some(("dmg", 2));
    }
    None
}

/// `bdrv_probe_all()` with the known formats of other builds: the name of the format the data
/// looks like, and its driver if this build has one.
pub(crate) fn probe(
    buf: &[u8],
    filename: Option<&str>,
) -> (&'static str, Option<&'static DriverDef>) {
    let mut best: Option<(&'static DriverDef, i32)> = None;
    for d in drivers::DRIVERS {
        if let Some(p) = d.probe {
            let score = p(buf, filename);
            if score > best.map_or(0, |b| b.1) {
                best = Some((d, score));
            }
        }
    }
    if let Some((name, score)) = known_format(buf, filename) {
        if drivers::find_format(name).is_none() && score > best.map_or(0, |b| b.1) {
            return (name, None);
        }
    }
    match best {
        Some((d, _)) => (d.format_name, Some(d)),
        None => ("raw", None),
    }
}

/// The format name the data looks like, `raw` when nothing else claims it.
pub(crate) fn probe_format(buf: &[u8]) -> &'static str {
    probe(buf, None).0
}

/// `find_image_format()`: reads the start of `file` and picks the driver.
pub(crate) fn find_image_format(file: &Node, filename: Option<&str>) -> Result<&'static DriverDef> {
    let raw = || drivers::find_format("raw").expect("raw is always built");
    let len = file.getlength().unwrap_or(0);
    if len == 0 || !file.is_inserted() {
        return Ok(raw());
    }
    // QEMU always reads the whole buffer; the part past the end of a short image reads as
    // zeroes, which the generic read path gives here too.
    let mut buf = [0u8; BLOCK_PROBE_BUF_SIZE];
    file.pread(0, &mut buf)
        .map_err(|e| Error::from_io("Could not read image for determining its format", e))?;
    match probe(&buf, filename) {
        (_, Some(d)) => Ok(d),
        ("raw", None) => {
            Err(Error::generic("Could not determine image format: No compatible driver found"))
        }
        (name, None) => Err(Error::generic(format!("Driver '{name}' is not supported yet"))),
    }
}

/// `raw_probe()`: raw claims everything, with the lowest score.
pub(crate) fn raw_probe(_buf: &[u8], _filename: Option<&str>) -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores() {
        let mut b = [0u8; 512];
        assert_eq!(probe_format(&b), "raw");
        b[..8].copy_from_slice(b"QFI\xfb\0\0\0\x03");
        assert_eq!(probe_format(&b), "qcow2");
        b[..8].copy_from_slice(b"QFI\xfb\0\0\0\x01");
        assert_eq!(probe_format(&b), "qcow");
        b[..8].copy_from_slice(b"conectix");
        assert_eq!(probe_format(&b), "vpc");
        b[..8].copy_from_slice(b"QFI\xfb\0\0\0\0");
        assert_eq!(probe_format(&b), "raw");
        b[..8].fill(0);
        assert_eq!(probe(&b, Some("x.dmg")).0, "dmg");
        assert_eq!(probe(&b, Some(".dmg")).0, "raw");
        b[..20].copy_from_slice(b"#!/bin/sh\n#V2.0 Form");
        assert_eq!(probe(&b, None).0, "raw");
    }
}
