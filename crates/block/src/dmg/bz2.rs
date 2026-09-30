// SPDX-License-Identifier: GPL-2.0-or-later

//! The bzip2 chunk decompressor, what QEMU's dmg-bz2 module does with libbz2: one call of
//! `BZ2_bzDecompress()` on the whole chunk, which must end the stream and fill the output
//! exactly.
//!
//! The decoder follows libbz2 1.0.8 (decompress.c), including its consistency checks, so the
//! same streams are rejected. The one difference: blocks with the "randomised" bit, which no
//! bzip2 since 0.9.5 (1999) writes, are rejected instead of being decoded.

/// `BZ_MAX_SELECTORS`.
const MAX_SELECTORS: usize = 2 + 900_000 / 50;
/// `BZ_MAX_ALPHA_SIZE`.
const MAX_ALPHA_SIZE: usize = 258;
/// `BZ_MAX_CODE_LEN`.
const MAX_CODE_LEN: usize = 23;
/// `BZ_G_SIZE`.
const G_SIZE: u32 = 50;
const RUNA: u32 = 0;
const RUNB: u32 = 1;

/// A stream that does not decode, `BZ_DATA_ERROR` and friends.
#[derive(Debug)]
pub(crate) struct DataError;

type R<T> = Result<T, DataError>;

/// `dmg_uncompress_bz2_do()`: decompresses the bzip2 stream at the start of `input` into
/// `out`, which it must fill exactly.
pub(crate) fn uncompress(input: &[u8], out: &mut [u8]) -> R<()> {
    let n = decompress(input, out)?;
    if n != out.len() {
        return Err(DataError);
    }
    Ok(())
}

/// Decodes the first bzip2 stream in `input` into `out`. Returns how many bytes it wrote, or
/// an error if the stream is corrupt, truncated, or does not fit.
pub(crate) fn decompress(input: &[u8], out: &mut [u8]) -> R<usize> {
    let mut br = BitReader { data: input, pos: 0, buf: 0, live: 0 };
    if br.bits(8)? != u32::from(b'B')
        || br.bits(8)? != u32::from(b'Z')
        || br.bits(8)? != u32::from(b'h')
    {
        return Err(DataError);
    }
    let level = br.bits(8)?;
    if !(u32::from(b'1')..=u32::from(b'9')).contains(&level) {
        return Err(DataError);
    }
    let block_size_100k = level - u32::from(b'0');
    let mut tt = Vec::new();
    let mut out_pos = 0usize;
    let mut combined_crc: u32 = 0;

    loop {
        let uc = br.bits(8)?;
        if uc == 0x17 {
            for m in [0x72, 0x45, 0x38, 0x50, 0x90] {
                if br.bits(8)? != m {
                    return Err(DataError);
                }
            }
            let stored = br.bits(32)?;
            if stored != combined_crc {
                return Err(DataError);
            }
            return Ok(out_pos);
        }
        if uc != 0x31 {
            return Err(DataError);
        }
        for m in [0x41, 0x59, 0x26, 0x53, 0x59] {
            if br.bits(8)? != m {
                return Err(DataError);
            }
        }
        let stored_block_crc = br.bits(32)?;
        let block_crc = decode_block(&mut br, block_size_100k, &mut tt, out, &mut out_pos)?;
        if block_crc != stored_block_crc {
            return Err(DataError);
        }
        combined_crc = combined_crc.rotate_left(1) ^ block_crc;
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    live: u32,
}

impl BitReader<'_> {
    /// `GET_BITS`: the next `n` bits, most significant first. Running out of input is an
    /// error here since the whole chunk is given at once.
    fn bits(&mut self, n: u32) -> R<u32> {
        while self.live < n {
            let b = *self.data.get(self.pos).ok_or(DataError)?;
            self.pos += 1;
            self.buf = (self.buf << 8) | u64::from(b);
            self.live += 8;
        }
        self.live -= n;
        Ok(((self.buf >> self.live) & ((1u64 << n) - 1)) as u32)
    }

    fn bit(&mut self) -> R<u32> {
        self.bits(1)
    }
}

/// The decoding tables of one Huffman group, `BZ2_hbCreateDecodeTables()`.
#[derive(Clone)]
struct Group {
    limit: [i32; MAX_CODE_LEN],
    base: [i32; MAX_CODE_LEN],
    perm: [i32; MAX_ALPHA_SIZE],
    min_len: u32,
}

impl Group {
    fn new(length: &[u8], min_len: u32, max_len: u32) -> Group {
        let mut g = Group {
            limit: [0; MAX_CODE_LEN],
            base: [0; MAX_CODE_LEN],
            perm: [0; MAX_ALPHA_SIZE],
            min_len,
        };
        let mut pp = 0;
        for i in min_len..=max_len {
            for (j, &l) in length.iter().enumerate() {
                if u32::from(l) == i {
                    g.perm[pp] = j as i32;
                    pp += 1;
                }
            }
        }
        for &l in length {
            g.base[usize::from(l) + 1] += 1;
        }
        for i in 1..MAX_CODE_LEN {
            g.base[i] += g.base[i - 1];
        }
        let mut vec: i32 = 0;
        for i in min_len as usize..=max_len as usize {
            vec += g.base[i + 1] - g.base[i];
            g.limit[i] = vec - 1;
            vec <<= 1;
        }
        for i in min_len as usize + 1..=max_len as usize {
            g.base[i] = ((g.limit[i - 1] + 1) << 1) - g.base[i];
        }
        g
    }
}

/// The symbol decoder state of a block, `GET_MTF_VAL`.
struct Symbols<'a> {
    groups: &'a [Group],
    selectors: &'a [u8],
    group_no: usize,
    group_pos: u32,
}

impl Symbols<'_> {
    fn next(&mut self, br: &mut BitReader<'_>) -> R<u32> {
        if self.group_pos == 0 {
            self.group_no = self.group_no.wrapping_add(1);
            if self.group_no >= self.selectors.len() {
                return Err(DataError);
            }
            self.group_pos = G_SIZE;
        }
        self.group_pos -= 1;
        let g = &self.groups[usize::from(self.selectors[self.group_no])];
        let mut zn = g.min_len;
        let mut zvec = br.bits(zn)? as i32;
        loop {
            if zn > 20 {
                return Err(DataError);
            }
            if zvec <= g.limit[zn as usize] {
                break;
            }
            zn += 1;
            zvec = (zvec << 1) | br.bit()? as i32;
        }
        let idx = zvec - g.base[zn as usize];
        if !(0..MAX_ALPHA_SIZE as i32).contains(&idx) {
            return Err(DataError);
        }
        Ok(g.perm[idx as usize] as u32)
    }
}

/// Decodes one block after its CRC into `out[*out_pos..]`, returning the CRC of what it
/// wrote.
fn decode_block(
    br: &mut BitReader<'_>,
    block_size_100k: u32,
    tt: &mut Vec<u32>,
    out: &mut [u8],
    out_pos: &mut usize,
) -> R<u32> {
    let randomised = br.bit()?;
    let orig_ptr = br.bits(24)?;
    if orig_ptr > 10 + 100_000 * block_size_100k {
        return Err(DataError);
    }
    if randomised != 0 {
        return Err(DataError);
    }

    // The symbol map.
    let mut in_use16 = [false; 16];
    for u in &mut in_use16 {
        *u = br.bit()? == 1;
    }
    let mut seq_to_unseq = Vec::with_capacity(256);
    for (i, &u) in in_use16.iter().enumerate() {
        if u {
            for j in 0..16 {
                if br.bit()? == 1 {
                    seq_to_unseq.push((i * 16 + j) as u8);
                }
            }
        }
    }
    if seq_to_unseq.is_empty() {
        return Err(DataError);
    }
    let alpha_size = seq_to_unseq.len() + 2;

    // The selectors.
    let n_groups = br.bits(3)? as usize;
    if !(2..=6).contains(&n_groups) {
        return Err(DataError);
    }
    let n_selectors = br.bits(15)? as usize;
    if n_selectors < 1 {
        return Err(DataError);
    }
    let mut selector_mtf = Vec::with_capacity(n_selectors.min(MAX_SELECTORS));
    for i in 0..n_selectors {
        let mut j = 0;
        while br.bit()? != 0 {
            j += 1;
            if j >= n_groups {
                return Err(DataError);
            }
        }
        if i < MAX_SELECTORS {
            selector_mtf.push(j as u8);
        }
    }
    let mut pos: Vec<u8> = (0..n_groups as u8).collect();
    let selectors: Vec<u8> = selector_mtf
        .iter()
        .map(|&v| {
            let v = usize::from(v);
            let tmp = pos[v];
            pos.copy_within(0..v, 1);
            pos[0] = tmp;
            tmp
        })
        .collect();

    // The coding tables.
    let mut groups = Vec::with_capacity(n_groups);
    for _ in 0..n_groups {
        let mut curr = br.bits(5)? as i32;
        let mut len = vec![0u8; alpha_size];
        for l in &mut len {
            loop {
                if !(1..=20).contains(&curr) {
                    return Err(DataError);
                }
                if br.bit()? == 0 {
                    break;
                }
                if br.bit()? == 0 {
                    curr += 1;
                } else {
                    curr -= 1;
                }
            }
            *l = curr as u8;
        }
        let min_len = u32::from(*len.iter().min().unwrap());
        let max_len = u32::from(*len.iter().max().unwrap());
        groups.push(Group::new(&len, min_len, max_len));
    }

    // The MTF values.
    let eob = seq_to_unseq.len() as u32 + 1;
    let nblock_max = 100_000 * block_size_100k as usize;
    let mut unzftab = [0i64; 256];
    let mut yy: [u8; 256] = std::array::from_fn(|i| i as u8);
    tt.clear();
    let mut syms =
        Symbols { groups: &groups, selectors: &selectors, group_no: usize::MAX, group_pos: 0 };
    let mut next_sym = syms.next(br)?;
    loop {
        if next_sym == eob {
            break;
        }
        if next_sym == RUNA || next_sym == RUNB {
            let mut es: i64 = -1;
            let mut n: i64 = 1;
            loop {
                if n >= 2 * 1024 * 1024 {
                    return Err(DataError);
                }
                es += if next_sym == RUNA { n } else { 2 * n };
                n *= 2;
                next_sym = syms.next(br)?;
                if next_sym != RUNA && next_sym != RUNB {
                    break;
                }
            }
            es += 1;
            let uc = seq_to_unseq[usize::from(yy[0])];
            unzftab[usize::from(uc)] += es;
            if tt.len() as i64 + es > nblock_max as i64 {
                return Err(DataError);
            }
            tt.resize(tt.len() + es as usize, u32::from(uc));
        } else {
            if tt.len() >= nblock_max {
                return Err(DataError);
            }
            let nn = (next_sym - 1) as usize;
            let uc = yy[nn];
            yy.copy_within(0..nn, 1);
            yy[0] = uc;
            let c = *seq_to_unseq.get(usize::from(uc)).ok_or(DataError)?;
            unzftab[usize::from(c)] += 1;
            tt.push(u32::from(c));
            next_sym = syms.next(br)?;
        }
    }

    let nblock = tt.len();
    if orig_ptr as usize >= nblock {
        return Err(DataError);
    }
    if unzftab.iter().any(|&c| c < 0 || c > nblock as i64) {
        return Err(DataError);
    }
    let mut cftab = [0i64; 257];
    for i in 1..=256 {
        cftab[i] = cftab[i - 1] + unzftab[i - 1];
    }
    if cftab.iter().any(|&c| c < 0 || c > nblock as i64) {
        return Err(DataError);
    }

    // The inverse BWT.
    for i in 0..nblock {
        let uc = (tt[i] & 0xff) as usize;
        let slot = cftab[uc] as usize;
        tt[slot] |= (i as u32) << 8;
        cftab[uc] += 1;
    }
    let mut t_pos = tt[orig_ptr as usize] >> 8;
    let get = |t_pos: &mut u32| -> R<u8> {
        let v = *tt.get(*t_pos as usize).ok_or(DataError)?;
        *t_pos = v >> 8;
        Ok((v & 0xff) as u8)
    };

    // The run length decoding, unRLE_obuf_to_output_FAST().
    let mut crc = Crc::new();
    let done = nblock + 1;
    let mut k0 = get(&mut t_pos)?;
    let mut used = 1usize;
    let mut out_ch = 0u8;
    let mut out_len = 0usize;
    loop {
        if out_len > 0 {
            let dst = out.get_mut(*out_pos..*out_pos + out_len).ok_or(DataError)?;
            dst.fill(out_ch);
            crc.update(dst);
            *out_pos += out_len;
        }
        if used == done {
            break;
        }
        if used > done {
            return Err(DataError);
        }
        out_len = 1;
        out_ch = k0;
        let k1 = get(&mut t_pos)?;
        used += 1;
        if used == done {
            continue;
        }
        if k1 != k0 {
            k0 = k1;
            continue;
        }
        out_len = 2;
        let k1 = get(&mut t_pos)?;
        used += 1;
        if used == done {
            continue;
        }
        if k1 != k0 {
            k0 = k1;
            continue;
        }
        out_len = 3;
        let k1 = get(&mut t_pos)?;
        used += 1;
        if used == done {
            continue;
        }
        if k1 != k0 {
            k0 = k1;
            continue;
        }
        let k1 = get(&mut t_pos)?;
        used += 1;
        out_len = usize::from(k1) + 4;
        k0 = get(&mut t_pos)?;
        used += 1;
    }
    Ok(crc.finish())
}

/// The CRC-32 of bzip2: polynomial 0x04c11db7, most significant bit first.
struct Crc(u32);

impl Crc {
    const TABLE: [u32; 256] = {
        let mut t = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = (i as u32) << 24;
            let mut k = 0;
            while k < 8 {
                c = if c & 0x8000_0000 != 0 { (c << 1) ^ 0x04c1_1db7 } else { c << 1 };
                k += 1;
            }
            t[i] = c;
            i += 1;
        }
        t
    };

    fn new() -> Crc {
        Crc(0xffff_ffff)
    }

    fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.0 = (self.0 << 8) ^ Self::TABLE[((self.0 >> 24) ^ u32::from(b)) as usize];
        }
    }

    fn finish(&self) -> u32 {
        !self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `printf 'hello hello hello hello\n' | bzip2 -9`
    const HELLO: &[u8] = &[
        0x42, 0x5a, 0x68, 0x39, 0x31, 0x41, 0x59, 0x26, 0x53, 0x59, 0x6f, 0x4f, 0x10, 0xf3, 0x00,
        0x00, 0x05, 0xd1, 0x00, 0x00, 0x10, 0x40, 0x00, 0x02, 0x44, 0xa0, 0x00, 0x30, 0xc0, 0x02,
        0xa8, 0x34, 0x71, 0x0d, 0xad, 0x87, 0x0f, 0x17, 0x72, 0x45, 0x38, 0x50, 0x90, 0x6f, 0x4f,
        0x10, 0xf3,
    ];

    #[test]
    fn small_stream() {
        let want = b"hello hello hello hello\n";
        let mut out = vec![0u8; want.len()];
        uncompress(HELLO, &mut out).unwrap();
        assert_eq!(&out, want);
        // Too small or too large an output buffer fails, as in QEMU.
        assert!(uncompress(HELLO, &mut vec![0u8; want.len() - 1]).is_err());
        assert!(uncompress(HELLO, &mut vec![0u8; want.len() + 1]).is_err());
        // Truncated or corrupt streams fail.
        assert!(uncompress(&HELLO[..HELLO.len() - 1], &mut out).is_err());
        let mut bad = HELLO.to_vec();
        bad[20] ^= 0x10;
        assert!(uncompress(&bad, &mut out).is_err());
    }

    /// Round trips through the bzip2 tool, when it is installed, with data that exercises
    /// long runs, several blocks and all byte values.
    #[test]
    fn against_bzip2_tool() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut data = Vec::new();
        let mut x: u32 = 1;
        for i in 0..300_000u32 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            match (i / 10_000) % 3 {
                0 => data.push((x >> 16) as u8),
                1 => data.push(b'a' + (i % 7) as u8),
                _ => data.push(0),
            }
        }
        let mut child = match Command::new("bzip2")
            .arg("-1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(_) => {
                eprintln!("bzip2 not found, skipping");
                return;
            }
        };
        let mut stdin = child.stdin.take().unwrap();
        let input = data.clone();
        let writer = std::thread::spawn(move || stdin.write_all(&input));
        let packed = child.wait_with_output().unwrap().stdout;
        writer.join().unwrap().unwrap();
        let mut out = vec![0u8; data.len()];
        uncompress(&packed, &mut out).unwrap();
        assert!(out == data);
    }
}
