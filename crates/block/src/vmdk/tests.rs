// SPDX-License-Identifier: GPL-2.0-or-later

//! Unit tests of the parts of the vmdk driver that need no files.

use super::create::{descriptor, filename_decompose, sparse_header};
use super::io::{write_cid_buf, zlib_compress};
use super::open::{parse_description, read_cid_from, scan_extent_line};
use super::*;

#[test]
fn probe() {
    assert_eq!(vmdk_probe(b"KDMV\x01\0\0\0", None), 100);
    assert_eq!(vmdk_probe(b"COWD\x01\0\0\0", None), 100);
    assert_eq!(vmdk_probe(b"KDM", None), 0);
    assert_eq!(vmdk_probe(b"# Disk DescriptorFile\nversion=1\nCID=1\n", None), 100);
    assert_eq!(vmdk_probe(b"# comment\r\n   \r\nversion=2\r\n", None), 100);
    assert_eq!(vmdk_probe(b"\nversion=1\n", None), 0);
    assert_eq!(vmdk_probe(b"  x\nversion=1\n", None), 0);
    assert_eq!(vmdk_probe(b"version=4\n", None), 0);
    assert_eq!(vmdk_probe(b"# only a comment", None), 0);
}

#[test]
fn round_up() {
    assert_eq!(round_up_mask(0, 128), 0);
    assert_eq!(round_up_mask(1, 128), 128);
    assert_eq!(round_up_mask(128, 128), 128);
    assert_eq!(round_up_mask(5, 0), 0);
    // A mask, not a division: 3 is not a power of two.
    assert_eq!(round_up_mask(4, 3), 4);
}

#[test]
fn extent_lines() {
    let l = scan_extent_line(b"RW 2048 FLAT \"a-flat.vmdk\" 0\nRW 1 x");
    assert_eq!(l.matches, 5);
    assert_eq!(l.access, b"RW");
    assert_eq!(l.sectors, 2048);
    assert_eq!(l.ty, b"FLAT");
    assert_eq!(l.fname, b"a-flat.vmdk");
    assert_eq!(l.flat_offset, 0);

    // sscanf() goes on to the next line for the offset.
    let l = scan_extent_line(b"RW 16 SPARSE \"s.vmdk\"\nRW 16 SPARSE \"t.vmdk\"\n");
    assert_eq!(l.matches, 4);
    assert_eq!(l.fname, b"s.vmdk");

    let l = scan_extent_line(b"RW 16 SPARSE s.vmdk\n");
    assert_eq!(l.matches, 3);
    let l = scan_extent_line(b"createType=\"monolithicFlat\"\n");
    assert_eq!(l.matches, 1);
    assert_eq!(scan_extent_line(b"   \n").matches, -1);
    // Access modes are at most 10 characters.
    let l = scan_extent_line(b"RWRWRWRWRWRW 1");
    assert_eq!(l.access, b"RWRWRWRWRW");
}

#[test]
fn description() {
    let d = b"version=1\nCID=fffffffe\nparentCID=ffffffff\ncreateType=\"monolithicSparse\"\n\0";
    assert_eq!(parse_description(d, "createType").as_deref(), Some("monolithicSparse"));
    assert_eq!(parse_description(d, "parentFileNameHint"), None);
    assert_eq!(read_cid_from(d, false), Some(0xffff_fffe));
    assert_eq!(read_cid_from(d, true), Some(0xffff_ffff));
    assert_eq!(read_cid_from(b"CID=zz\n", false), None);
}

#[test]
fn write_cid() {
    let text =
        b"# Disk DescriptorFile\nversion=1\nCID=12345678\nparentCID=ffffffff\ncreateType=\"x\"\n";
    let mut desc = text.to_vec();
    desc.resize(200, 0);
    write_cid_buf(&mut desc, 0xab).unwrap();
    let want = b"# Disk DescriptorFile\nversion=1\nCID=ab\nparentCID=ffffffff\ncreateType=\"x\"\n";
    assert_eq!(&desc[..want.len()], want);
    assert_eq!(desc[want.len()], 0);
    // The old tail stays behind the new end of the text.
    assert_eq!(&desc[want.len() + 1..text.len()], &text[want.len() + 1..]);

    let mut desc = b"CID=1\n\0".to_vec();
    assert!(write_cid_buf(&mut desc, 2).is_err());
}

#[test]
fn names() {
    let (p, a, b) = filename_decompose("/tmp/dir/disk.vmdk").unwrap();
    assert_eq!((p.as_str(), a.as_str(), b.as_str()), ("/tmp/dir/", "disk", ".vmdk"));
    let (p, a, b) = filename_decompose("disk").unwrap();
    assert_eq!((p.as_str(), a.as_str(), b.as_str()), ("", "disk", ""));
    let (p, a, b) = filename_decompose("file:x.y.z").unwrap();
    assert_eq!((p.as_str(), a.as_str(), b.as_str()), ("file:", "x.y", ".z"));
    assert_eq!(filename_decompose("").unwrap_err().message(), "No filename provided");
}

#[test]
fn header() {
    // 1 GiB: 16384 grains in 32 grain tables of 4 sectors, a grain directory of 1 sector.
    let h = sparse_header(1 << 30, false, false);
    assert_eq!(le32(&h, 0), 1);
    assert_eq!(le32(&h, 4), VMDK4_FLAG_RGD | VMDK4_FLAG_NL_DETECT);
    assert_eq!(le64(&h, 8), 2 << 20);
    assert_eq!(le64(&h, 44), 21);
    assert_eq!(le64(&h, 52), 21 + 1 + 128);
    assert_eq!(le64(&h, 60), 384);
    assert_eq!(&h[69..73], &[0x0a, 0x20, 0x0d, 0x0a]);
    let h = sparse_header(1 << 20, true, false);
    assert_eq!(le32(&h, 0), 3);
    assert_eq!(le16(&h, 73), 1);
    assert_eq!(le32(&sparse_header(1 << 20, false, true), 0), 2);
}

#[test]
fn descriptor_text() {
    let d = descriptor(
        0x1234,
        0xffff_ffff,
        "twoGbMaxExtentFlat",
        "",
        "RW 2048 FLAT \"b-f001.vmdk\" 0\n",
        "4",
        2,
        16,
        "ide",
        "2147483647",
    );
    assert_eq!(
        d,
        "# Disk DescriptorFile\nversion=1\nCID=1234\nparentCID=ffffffff\n\
         createType=\"twoGbMaxExtentFlat\"\n\n# Extent description\n\
         RW 2048 FLAT \"b-f001.vmdk\" 0\n\n# The Disk Data Base\n#DDB\n\n\
         ddb.virtualHWVersion = \"4\"\nddb.geometry.cylinders = \"2\"\n\
         ddb.geometry.heads = \"16\"\nddb.geometry.sectors = \"63\"\n\
         ddb.adapterType = \"ide\"\nddb.toolsVersion = \"2147483647\"\n"
    );
}

#[test]
fn compress_round_trip() {
    let data: Vec<u8> = (0..65536u32).map(|i| (i % 7) as u8).collect();
    let z = zlib_compress(&data, 2 * 65536).unwrap();
    // A zlib stream.
    assert_eq!(z[0], 0x78);
    let mut d = flate2::Decompress::new(true);
    let mut out = vec![0u8; 65536];
    d.decompress(&z, &mut out, flate2::FlushDecompress::Finish).unwrap();
    assert_eq!(out, data);
    assert!(zlib_compress(&data, 4).is_none());
}
