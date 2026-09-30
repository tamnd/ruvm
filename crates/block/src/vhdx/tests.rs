// SPDX-License-Identifier: GPL-2.0-or-later

use super::*;

#[test]
fn crc32c_known_value() {
    // The standard CRC-32C check value.
    assert_eq!(crc32c(0xffff_ffff, b"123456789"), 0xe306_9283);
}

#[test]
fn crc32c_chains_like_qemu() {
    // The log checksum is built sector by sector, inverting between the pieces.
    let data = b"The quick brown fox jumps over the lazy dog";
    let whole = crc32c(0xffff_ffff, data);
    let first = crc32c(0xffff_ffff, &data[..10]) ^ 0xffff_ffff;
    assert_eq!(crc32c(first, &data[10..]), whole);
    // A checksum field in the middle is read as zeroes.
    let mut buf = data.to_vec();
    buf[4..8].copy_from_slice(&[1, 2, 3, 4]);
    let mut zeroed = buf.clone();
    zeroed[4..8].fill(0);
    assert_eq!(checksum_calc(0xffff_ffff, &buf, Some(4)), crc32c(0xffff_ffff, &zeroed));
    update_checksum(&mut buf, 4);
    assert!(checksum_is_valid(&buf, 4));
    buf[20] ^= 1;
    assert!(!checksum_is_valid(&buf, 4));
}

#[test]
fn guids_are_little_endian() {
    assert_eq!(
        BAT_GUID,
        [
            0x66, 0x77, 0xc2, 0x2d, 0x23, 0xf6, 0x00, 0x42, 0x9d, 0x64, 0x11, 0x5e, 0x9b, 0xfd,
            0x4a, 0x08
        ]
    );
    let g = guid_generate().unwrap();
    assert_eq!(g[7] >> 4, 4);
    assert_eq!(g[8] >> 6, 2);
}

#[test]
fn bat_geometry() {
    let g = Geometry::new(8 << 20, 512, 20 << 20);
    assert_eq!((g.chunk_ratio, g.chunk_ratio_bits, g.sectors_per_block_bits), (512, 9, 14));
    assert_eq!(g.bat_entries, 3);
    // 513 payload blocks need a sector bitmap entry after the first 512.
    let g = Geometry::new(1 << 20, 512, 513 << 20);
    assert_eq!(g.chunk_ratio, 4096);
    let g2 = Geometry::new(1 << 20, 512, (4097u64) << 20);
    assert_eq!((g.bat_entries, g2.bat_entries), (513, 4098));
    // An empty disk wraps around in 32 bits like QEMU's uint32_t arithmetic.
    assert_eq!(Geometry::new(8 << 20, 512, 0).bat_entries, 8_388_607);

    // Block 4096 of a 1 MiB block image comes after the bitmap entry at index 4096.
    let bat = vec![0u64; 4098];
    let s = g2.translate(&bat, 4096 * 2048 + 3, 10);
    assert_eq!((s.bat_idx, s.sectors_avail, s.bytes_avail, s.block_offset), (4097, 10, 5120, 1536));
    let s = g2.translate(&bat, 2047, 10);
    assert_eq!((s.bat_idx, s.sectors_avail), (0, 1));
}

#[test]
fn bat_entry_states() {
    let g = Geometry { bat_offset: 3 << 20, ..Geometry::new(1 << 20, 512, 4 << 20) };
    let mut bat = vec![0u64; 4];
    let sinfo = SectorInfo { bat_idx: 2, file_offset: 7 << 20, ..Default::default() };
    assert_eq!(
        g.update_bat_table_entry(&mut bat, &sinfo, PAYLOAD_BLOCK_FULLY_PRESENT),
        ((7 << 20) | 6, (3 << 20) + 16)
    );
    // Zero blocks must not keep an offset: Hyper-V refuses them otherwise.
    assert_eq!(g.update_bat_table_entry(&mut bat, &sinfo, PAYLOAD_BLOCK_ZERO).0, 2);
    assert_eq!(bat[2], 2);
}

#[test]
fn headers_round_trip() {
    let h = Header {
        signature: HEADER_SIGNATURE,
        sequence_number: 77,
        file_write_guid: [1; 16],
        data_write_guid: [2; 16],
        log_guid: [3; 16],
        version: 1,
        log_length: 1 << 20,
        log_offset: 1 << 20,
        ..Default::default()
    };
    let mut b = [0u8; HEADER_STRUCT_SIZE];
    h.write(&mut b);
    assert_eq!(&b[..4], b"head");
    assert_eq!(Header::parse(&b), h);
}

#[test]
fn probe_scores() {
    assert_eq!(probe(b"vhdxfile\0\0\0\0", None), 100);
    assert_eq!(probe(b"vhdxfil", None), 0);
    assert_eq!(probe(b"conectix", None), 0);
}

#[test]
fn round_up_wraps() {
    assert_eq!(round_up(1, MIB), MIB);
    assert_eq!(round_up(MIB, MIB), MIB);
    assert_eq!(round_up(u64::MAX, 512), 0);
}
