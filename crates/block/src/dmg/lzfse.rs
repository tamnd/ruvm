// SPDX-License-Identifier: GPL-2.0-or-later

//! The lzfse chunk decompressor, what QEMU's dmg-lzfse module does with Apple's liblzfse:
//! `lzfse_decode_buffer()` on the whole chunk.
//!
//! An lzfse stream is a sequence of blocks, each starting with a 4 byte magic: "bvx-" for
//! stored bytes, "bvxn" for LZVN, "bvx1" and "bvx2" for LZFSE proper (literals and
//! literal/match/distance triples coded with finite state entropy, with a plain or a packed
//! header), and "bvx$" to end the stream. The decoder follows the reference implementation
//! (lzfse_decode_base.c, lzfse_fse.c, lzvn_decode_base.c), its checks included, so the same
//! streams are accepted and rejected, and it fills the output the same way: a stream that
//! decodes to more than the output holds fills it and counts as success, as
//! `lzfse_decode_buffer()` returns the output size then.

/// `LZFSE_ENCODE_*_SYMBOLS` and `LZFSE_ENCODE_*_STATES`.
const L_SYMBOLS: usize = 20;
const M_SYMBOLS: usize = 20;
const D_SYMBOLS: usize = 64;
const LITERAL_SYMBOLS: usize = 256;
const L_STATES: usize = 64;
const M_STATES: usize = 64;
const D_STATES: usize = 256;
const LITERAL_STATES: usize = 1024;
/// `LZFSE_MATCHES_PER_BLOCK` and `LZFSE_LITERALS_PER_BLOCK`.
const MATCHES_PER_BLOCK: u32 = 10_000;
const LITERALS_PER_BLOCK: u32 = 4 * MATCHES_PER_BLOCK;

const ENDOFSTREAM_MAGIC: u32 = 0x2478_7662; // bvx$
const UNCOMPRESSED_MAGIC: u32 = 0x2d78_7662; // bvx-
const COMPRESSEDV1_MAGIC: u32 = 0x3178_7662; // bvx1
const COMPRESSEDV2_MAGIC: u32 = 0x3278_7662; // bvx2
const COMPRESSEDLZVN_MAGIC: u32 = 0x6e78_7662; // bvxn

const L_EXTRA_BITS: [u8; L_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 5, 8];
const M_EXTRA_BITS: [u8; M_SYMBOLS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 5, 8, 11];
const D_EXTRA_BITS: [u8; D_SYMBOLS] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7,
    8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14,
    14, 14, 15, 15, 15, 15,
];

/// The base values that go with the extra bits: each symbol starts where the previous one's
/// range ends, which is how the reference tables are laid out.
const fn base_values<const N: usize>(extra: &[u8; N]) -> [i32; N] {
    let mut b = [0i32; N];
    let mut i = 1;
    while i < N {
        b[i] = b[i - 1] + (1 << extra[i - 1]);
        i += 1;
    }
    b
}

const L_BASE_VALUE: [i32; L_SYMBOLS] = base_values(&L_EXTRA_BITS);
const M_BASE_VALUE: [i32; M_SYMBOLS] = base_values(&M_EXTRA_BITS);
const D_BASE_VALUE: [i32; D_SYMBOLS] = base_values(&D_EXTRA_BITS);

/// `LZFSE_STATUS_*` other than OK.
#[derive(Debug, PartialEq)]
enum Stop {
    /// `LZFSE_STATUS_DST_FULL`.
    DstFull,
    /// `LZFSE_STATUS_SRC_EMPTY` or `LZFSE_STATUS_ERROR`, which `lzfse_decode_buffer()` treats
    /// alike.
    Fail,
}

/// `dmg_uncompress_lzfse_do()` with `lzfse_decode_buffer()`: how many bytes were decoded
/// into `out`, or `None` where the C code returns -1.
pub(crate) fn uncompress(input: &[u8], out: &mut [u8]) -> Option<usize> {
    let n = decode_buffer(input, out);
    if n > 0 { Some(n) } else { None }
}

/// `lzfse_decode_buffer()`: the number of bytes written, `out.len()` when the output filled
/// up, 0 on error.
pub(crate) fn decode_buffer(input: &[u8], out: &mut [u8]) -> usize {
    let mut d = Decoder {
        src: input,
        sp: 0,
        dst: out,
        dp: 0,
        literals: vec![0; LITERALS_PER_BLOCK as usize + 64],
    };
    match d.decode() {
        Ok(()) => d.dp,
        Err(Stop::DstFull) => d.dst.len(),
        Err(Stop::Fail) => 0,
    }
}

fn load4(b: &[u8], at: usize) -> Option<u32> {
    b.get(at..at + 4).map(|s| u32::from_le_bytes(s.try_into().unwrap()))
}

fn load8(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn get_field(v: u64, offset: u32, nbits: u32) -> u32 {
    ((v >> offset) & ((1u64 << nbits) - 1)) as u32
}

/// `lzfse_compressed_block_header_v1`, the decoded form of both compressed headers.
struct HeaderV1 {
    n_literals: u32,
    n_matches: u32,
    n_literal_payload_bytes: u32,
    n_lmd_payload_bytes: u32,
    literal_bits: i32,
    literal_state: [u16; 4],
    lmd_bits: i32,
    l_state: u16,
    m_state: u16,
    d_state: u16,
    l_freq: [u16; L_SYMBOLS],
    m_freq: [u16; M_SYMBOLS],
    d_freq: [u16; D_SYMBOLS],
    literal_freq: [u16; LITERAL_SYMBOLS],
}

/// `sizeof(lzfse_compressed_block_header_v1)`: 770 bytes of fields padded to a multiple of
/// 4, which is what the reference code skips.
const HEADER_V1_SIZE: usize =
    (4 * 8 + 2 * 4 + 4 + 2 * 3 + 2 * (L_SYMBOLS + M_SYMBOLS + D_SYMBOLS + LITERAL_SYMBOLS))
        .next_multiple_of(4);
/// `offsetof(lzfse_compressed_block_header_v2, freq)`.
const HEADER_V2_FIXED: usize = 4 + 4 + 3 * 8;

impl HeaderV1 {
    fn zeroed() -> HeaderV1 {
        HeaderV1 {
            n_literals: 0,
            n_matches: 0,
            n_literal_payload_bytes: 0,
            n_lmd_payload_bytes: 0,
            literal_bits: 0,
            literal_state: [0; 4],
            lmd_bits: 0,
            l_state: 0,
            m_state: 0,
            d_state: 0,
            l_freq: [0; L_SYMBOLS],
            m_freq: [0; M_SYMBOLS],
            d_freq: [0; D_SYMBOLS],
            literal_freq: [0; LITERAL_SYMBOLS],
        }
    }

    fn freqs_mut(&mut self) -> impl Iterator<Item = &mut u16> {
        self.l_freq
            .iter_mut()
            .chain(self.m_freq.iter_mut())
            .chain(self.d_freq.iter_mut())
            .chain(self.literal_freq.iter_mut())
    }

    /// Reads the plain header at the start of `b`, which holds at least `HEADER_V1_SIZE`
    /// bytes.
    fn parse_v1(b: &[u8]) -> HeaderV1 {
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u16_at = |o: usize| u16::from_le_bytes(b[o..o + 2].try_into().unwrap());
        let mut h = HeaderV1::zeroed();
        h.n_literals = u32_at(12);
        h.n_matches = u32_at(16);
        h.n_literal_payload_bytes = u32_at(20);
        h.n_lmd_payload_bytes = u32_at(24);
        h.literal_bits = u32_at(28) as i32;
        for i in 0..4 {
            h.literal_state[i] = u16_at(32 + 2 * i);
        }
        h.lmd_bits = u32_at(40) as i32;
        h.l_state = u16_at(44);
        h.m_state = u16_at(46);
        h.d_state = u16_at(48);
        let mut o = 50;
        for f in h.freqs_mut() {
            *f = u16_at(o);
            o += 2;
        }
        h
    }

    /// `lzfse_decode_v1()`: unpacks the header at the start of `b`, which holds the whole
    /// `header_size` bytes of it.
    fn parse_v2(b: &[u8], header_size: usize) -> Option<HeaderV1> {
        let v0 = load8(b, 8);
        let v1 = load8(b, 16);
        let v2 = load8(b, 24);
        let mut h = HeaderV1::zeroed();

        h.n_literals = get_field(v0, 0, 20);
        h.n_literal_payload_bytes = get_field(v0, 20, 20);
        h.literal_bits = get_field(v0, 60, 3) as i32 - 7;
        h.literal_state = [
            get_field(v1, 0, 10) as u16,
            get_field(v1, 10, 10) as u16,
            get_field(v1, 20, 10) as u16,
            get_field(v1, 30, 10) as u16,
        ];

        h.n_matches = get_field(v0, 40, 20);
        h.n_lmd_payload_bytes = get_field(v1, 40, 20);
        h.lmd_bits = get_field(v1, 60, 3) as i32 - 7;
        h.l_state = get_field(v2, 32, 10) as u16;
        h.m_state = get_field(v2, 42, 10) as u16;
        h.d_state = get_field(v2, 52, 10) as u16;

        // No freq tables?
        if header_size == HEADER_V2_FIXED {
            return Some(h);
        }
        let mut src = HEADER_V2_FIXED;
        let src_end = header_size;
        let mut accum: u32 = 0;
        let mut accum_nbits = 0;
        for f in h.freqs_mut() {
            // Refill accum, one byte at a time, until we reach end of header, or accum is
            // full
            while src < src_end && accum_nbits + 8 <= 32 {
                accum |= u32::from(b[src]) << accum_nbits;
                accum_nbits += 8;
                src += 1;
            }
            let (value, nbits) = decode_v1_freq_value(accum);
            if nbits > accum_nbits {
                return None;
            }
            *f = value;
            accum >>= nbits;
            accum_nbits -= nbits;
        }
        // We need to end up exactly at the end of header, with less than 8 bits in the
        // accumulator
        if accum_nbits >= 8 || src != src_end {
            return None;
        }
        Some(h)
    }

    /// `lzfse_check_block_header_v1()`.
    fn check(&self) -> bool {
        self.n_literals <= LITERALS_PER_BLOCK
            && self.n_matches <= MATCHES_PER_BLOCK
            && self.literal_state.iter().all(|&s| usize::from(s) < LITERAL_STATES)
            && usize::from(self.l_state) < L_STATES
            && usize::from(self.m_state) < M_STATES
            && usize::from(self.d_state) < D_STATES
            && check_freq(&self.l_freq, L_STATES)
            && check_freq(&self.m_freq, M_STATES)
            && check_freq(&self.d_freq, D_STATES)
            && check_freq(&self.literal_freq, LITERAL_STATES)
    }
}

/// `lzfse_decode_v1_freq_value()`: a frequency from the low bits of `bits`, and how many
/// bits it took.
fn decode_v1_freq_value(bits: u32) -> (u16, u32) {
    const NBITS: [u8; 32] = [
        2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3, 2, 14, 2, 3, 2, 5, 2, 3, 2, 8, 2, 3, 2, 5, 2, 3,
        2, 14,
    ];
    const VALUE: [u8; 32] = [
        0, 2, 1, 4, 0, 3, 1, 0, 0, 2, 1, 5, 0, 3, 1, 0, 0, 2, 1, 6, 0, 3, 1, 0, 0, 2, 1, 7, 0, 3,
        1, 0,
    ];
    let b = (bits & 31) as usize;
    let n = u32::from(NBITS[b]);
    match n {
        8 => (8 + ((bits >> 4) & 0xf) as u16, n),
        14 => (24 + ((bits >> 4) & 0x3ff) as u16, n),
        _ => (u16::from(VALUE[b]), n),
    }
}

/// `fse_check_freq()`: the frequencies must not add up to more than the number of states.
fn check_freq(freq: &[u16], nstates: usize) -> bool {
    freq.iter().map(|&f| usize::from(f)).sum::<usize>() <= nstates
}

/// `fse_decoder_entry`.
#[derive(Clone, Copy, Default)]
struct Entry {
    k: u32,
    symbol: u8,
    delta: i32,
}

/// `fse_init_decoder_table()`.
fn init_decoder_table(nstates: usize, freq: &[u16]) -> Vec<Entry> {
    let mut t = vec![Entry::default(); nstates];
    let n_clz = (nstates as u32).leading_zeros();
    let mut at = 0;
    for (i, &f) in freq.iter().enumerate() {
        let f = u32::from(f);
        if f == 0 {
            continue; // skip this symbol, no occurrences
        }
        let k = f.leading_zeros() - n_clz; // shift needed to ensure N <= (F<<K) < 2*N
        let j0 = ((2 * nstates as u32) >> k) - f;
        for j in 0..f {
            t[at] = if j < j0 {
                Entry { k, symbol: i as u8, delta: (((f + j) << k) as i32) - nstates as i32 }
            } else {
                Entry { k: k - 1, symbol: i as u8, delta: ((j - j0) << (k - 1)) as i32 }
            };
            at += 1;
        }
    }
    t
}

/// `fse_value_decoder_entry`.
#[derive(Clone, Copy, Default)]
struct ValueEntry {
    total_bits: u32,
    value_bits: u32,
    delta: i32,
    vbase: i32,
}

/// `fse_init_value_decoder_table()`.
fn init_value_decoder_table(
    nstates: usize,
    freq: &[u16],
    symbol_vbits: &[u8],
    symbol_vbase: &[i32],
) -> Vec<ValueEntry> {
    let mut t = vec![ValueEntry::default(); nstates];
    let n_clz = (nstates as u32).leading_zeros();
    let mut at = 0;
    for (i, &f) in freq.iter().enumerate() {
        let f = u32::from(f);
        if f == 0 {
            continue;
        }
        let k = f.leading_zeros() - n_clz;
        let j0 = ((2 * nstates as u32) >> k) - f;
        let value_bits = u32::from(symbol_vbits[i]);
        let vbase = symbol_vbase[i];
        for j in 0..f {
            t[at] = if j < j0 {
                ValueEntry {
                    total_bits: k + value_bits,
                    value_bits,
                    delta: (((f + j) << k) as i32) - nstates as i32,
                    vbase,
                }
            } else {
                ValueEntry {
                    total_bits: k - 1 + value_bits,
                    value_bits,
                    delta: ((j - j0) << (k - 1)) as i32,
                    vbase,
                }
            };
            at += 1;
        }
    }
    t
}

/// `fse_in_stream64`: bits read backwards from the end of a buffer.
struct InStream {
    accum: u64,
    accum_nbits: i32,
    /// The read position, bytes before it are still to come.
    buf: usize,
}

impl InStream {
    /// `fse_in_checked_init64()`: starts reading backwards from `buf`, with `n` (at most 0)
    /// bits of the first word unused.
    fn init(src: &[u8], n: i32, buf: usize, buf_start: usize) -> Result<InStream, Stop> {
        let mut s = InStream { accum: 0, accum_nbits: 0, buf };
        if n != 0 {
            if buf < buf_start + 8 {
                return Err(Stop::Fail);
            }
            s.buf -= 8;
            s.accum = load8(src, s.buf);
            s.accum_nbits = n.wrapping_add(64);
        } else {
            if buf < buf_start + 7 {
                return Err(Stop::Fail);
            }
            s.buf -= 7;
            let mut b = [0u8; 8];
            b[..7].copy_from_slice(&src[s.buf..s.buf + 7]);
            s.accum = u64::from_le_bytes(b);
            s.accum_nbits = n + 56;
        }
        if !(56..64).contains(&s.accum_nbits) || (s.accum >> s.accum_nbits) != 0 {
            return Err(Stop::Fail);
        }
        Ok(s)
    }

    /// `fse_in_checked_flush64()`: tops the accumulator up to at least 56 bits.
    fn flush(&mut self, src: &[u8], buf_start: usize) -> Result<(), Stop> {
        let nbits = (63 - self.accum_nbits) & -8;
        let nbytes = (nbits >> 3) as usize;
        if self.buf < buf_start + nbytes {
            return Err(Stop::Fail);
        }
        self.buf -= nbytes;
        if nbits > 0 {
            let mut b = [0u8; 8];
            b[..nbytes].copy_from_slice(&src[self.buf..self.buf + nbytes]);
            self.accum = (self.accum << nbits) | u64::from_le_bytes(b);
            self.accum_nbits += nbits;
        }
        Ok(())
    }

    /// `fse_in_pull64()`.
    fn pull(&mut self, n: u32) -> Result<u64, Stop> {
        let n = n as i32;
        if n > self.accum_nbits {
            return Err(Stop::Fail);
        }
        self.accum_nbits -= n;
        let result = self.accum >> self.accum_nbits;
        self.accum &= (1u64 << self.accum_nbits) - 1;
        Ok(result)
    }

    /// `fse_decode()`.
    fn decode(&mut self, state: &mut u16, t: &[Entry]) -> Result<u8, Stop> {
        let e = t[usize::from(*state)];
        *state = (e.delta + self.pull(e.k)? as i32) as u16;
        Ok(e.symbol)
    }

    /// `fse_value_decode()`.
    fn value_decode(&mut self, state: &mut u16, t: &[ValueEntry]) -> Result<i32, Stop> {
        let e = t[usize::from(*state)];
        let state_and_value_bits = self.pull(e.total_bits)? as u32;
        *state = (e.delta + (state_and_value_bits >> e.value_bits) as i32) as u16;
        Ok(e.vbase + (state_and_value_bits & ((1u32 << e.value_bits) - 1)) as i32)
    }
}

/// `lzfse_decoder_state`, for one call.
struct Decoder<'a> {
    src: &'a [u8],
    sp: usize,
    dst: &'a mut [u8],
    dp: usize,
    /// The literal buffer of the compressed block state, kept across blocks as in C.
    literals: Vec<u8>,
}

impl Decoder<'_> {
    /// `lzfse_decode()`, run until the end of stream block.
    fn decode(&mut self) -> Result<(), Stop> {
        loop {
            // We need at least 4 bytes of magic number to identify next block
            let magic = load4(self.src, self.sp).ok_or(Stop::Fail)?;
            match magic {
                ENDOFSTREAM_MAGIC => return Ok(()),
                UNCOMPRESSED_MAGIC => {
                    let n_raw_bytes = load4(self.src, self.sp + 4).ok_or(Stop::Fail)?;
                    self.sp += 8;
                    self.uncompressed_block(n_raw_bytes as usize)?;
                }
                COMPRESSEDLZVN_MAGIC => {
                    let n_raw_bytes = load4(self.src, self.sp + 4).ok_or(Stop::Fail)?;
                    let n_payload_bytes = load4(self.src, self.sp + 8).ok_or(Stop::Fail)?;
                    self.sp += 12;
                    self.lzvn_block(n_raw_bytes as usize, n_payload_bytes as usize)?;
                }
                COMPRESSEDV1_MAGIC | COMPRESSEDV2_MAGIC => self.lzfse_block(magic)?,
                // Here we have an invalid magic number
                _ => return Err(Stop::Fail),
            }
        }
    }

    fn uncompressed_block(&mut self, mut n_raw_bytes: usize) -> Result<(), Stop> {
        while n_raw_bytes > 0 {
            if self.sp >= self.src.len() {
                return Err(Stop::Fail); // need more SRC data
            }
            if self.dp >= self.dst.len() {
                return Err(Stop::DstFull);
            }
            let n = n_raw_bytes.min(self.src.len() - self.sp).min(self.dst.len() - self.dp);
            self.dst[self.dp..self.dp + n].copy_from_slice(&self.src[self.sp..self.sp + n]);
            self.sp += n;
            self.dp += n;
            n_raw_bytes -= n;
        }
        Ok(())
    }

    fn lzvn_block(&mut self, n_raw_bytes: usize, n_payload_bytes: usize) -> Result<(), Stop> {
        let src_end = self.src.len().min(self.sp + n_payload_bytes);
        let dst_end = self.dst.len().min(self.dp + n_raw_bytes);
        let mut st = Lzvn {
            src: &self.src[..src_end],
            sp: self.sp,
            dst: &mut self.dst[..dst_end],
            dp: self.dp,
            d_prev: 0,
            end_of_stream: false,
        };
        st.decode();
        let (sp, dp, eos) = (st.sp, st.dp, st.end_of_stream);
        let src_used = sp - self.sp;
        let dst_used = dp - self.dp;
        self.sp = sp;
        self.dp = dp;
        if eos {
            // We reached end of block marker
            if src_used == n_payload_bytes && dst_used == n_raw_bytes {
                return Ok(());
            }
            return Err(Stop::Fail);
        }
        if self.dp == self.dst.len() {
            return Err(Stop::DstFull);
        }
        Err(Stop::Fail)
    }

    fn lzfse_block(&mut self, magic: u32) -> Result<(), Stop> {
        let rest = &self.src[self.sp..];
        let (h, header_size) = if magic == COMPRESSEDV2_MAGIC {
            // Check we have the fixed part of the structure
            if rest.len() < HEADER_V2_FIXED {
                return Err(Stop::Fail);
            }
            let header_size = get_field(load8(rest, 24), 0, 32) as usize;
            if header_size > rest.len() {
                return Err(Stop::Fail);
            }
            if header_size < HEADER_V2_FIXED {
                // The frequency tables would start past their end, which the reference code
                // rejects since it cannot end on the exact header size.
                return Err(Stop::Fail);
            }
            (HeaderV1::parse_v2(rest, header_size).ok_or(Stop::Fail)?, header_size)
        } else {
            if rest.len() < HEADER_V1_SIZE {
                return Err(Stop::Fail);
            }
            (HeaderV1::parse_v1(rest), HEADER_V1_SIZE)
        };

        // We require the header + entire encoded block to be present in SRC during the
        // entire block decoding.
        let need = header_size as u64
            + u64::from(h.n_literal_payload_bytes)
            + u64::from(h.n_lmd_payload_bytes);
        if need > rest.len() as u64 {
            return Err(Stop::Fail);
        }
        if !h.check() {
            return Err(Stop::Fail);
        }
        self.sp += header_size;

        let literal_decoder = init_decoder_table(LITERAL_STATES, &h.literal_freq);
        let l_decoder = init_value_decoder_table(L_STATES, &h.l_freq, &L_EXTRA_BITS, &L_BASE_VALUE);
        let m_decoder = init_value_decoder_table(M_STATES, &h.m_freq, &M_EXTRA_BITS, &M_BASE_VALUE);
        let d_decoder = init_value_decoder_table(D_STATES, &h.d_freq, &D_EXTRA_BITS, &D_BASE_VALUE);

        // Decode literals, reading bits backwards from the end of their payload; the reads
        // are bounded by the start of the whole input, as in C.
        self.sp += h.n_literal_payload_bytes as usize;
        let mut inp = InStream::init(self.src, h.literal_bits, self.sp, 0)?;
        let mut state = h.literal_state;
        let mut i = 0;
        while i < h.n_literals as usize {
            inp.flush(self.src, 0)?;
            for (j, s) in state.iter_mut().enumerate() {
                self.literals[i + j] = inp.decode(s, &literal_decoder)?;
            }
            i += 4;
        }

        // The L, M, D triples.
        let lmd_start = self.sp;
        let mut inp = InStream::init(
            self.src,
            h.lmd_bits,
            lmd_start + h.n_lmd_payload_bytes as usize,
            lmd_start,
        )?;
        let (mut l_state, mut m_state, mut d_state) = (h.l_state, h.m_state, h.d_state);
        let mut lit = 0usize;
        // Initialize D to an illegal value so we can't erroneously use an uninitialized
        // "previous" value.
        let mut d: i32 = -1;
        for _ in 0..h.n_matches {
            inp.flush(self.src, 0)?;
            let l = inp.value_decode(&mut l_state, &l_decoder)? as usize;
            if lit + l >= LITERALS_PER_BLOCK as usize + 64 {
                return Err(Stop::Fail);
            }
            let m = inp.value_decode(&mut m_state, &m_decoder)? as usize;
            let new_d = inp.value_decode(&mut d_state, &d_decoder)?;
            if new_d != 0 {
                d = new_d;
            }

            // Error if D is out of range, so that we avoid passing through uninitialized
            // data or accessing memory out of the destination buffer.
            if d as u32 as usize > self.dp + l {
                return Err(Stop::Fail);
            }
            let room = self.dst.len() - self.dp;
            let lc = l.min(room);
            self.dst[self.dp..self.dp + lc].copy_from_slice(&self.literals[lit..lit + lc]);
            self.dp += lc;
            lit += lc;
            if lc < l {
                return Err(Stop::DstFull);
            }
            let room = self.dst.len() - self.dp;
            let mc = m.min(room);
            copy_match(self.dst, self.dp, d as usize, mc);
            self.dp += mc;
            if mc < m {
                return Err(Stop::DstFull);
            }
        }
        self.sp += h.n_lmd_payload_bytes as usize;
        Ok(())
    }
}

/// Copies `len` bytes from `distance` bytes back to `at`, a byte at a time so that
/// overlapping matches repeat the pattern.
fn copy_match(dst: &mut [u8], at: usize, distance: usize, len: usize) {
    for i in at..at + len {
        dst[i] = dst[i - distance];
    }
}

/// `lzvn_decoder_state` and `lzvn_decode()`.
struct Lzvn<'a> {
    src: &'a [u8],
    sp: usize,
    dst: &'a mut [u8],
    dp: usize,
    d_prev: usize,
    end_of_stream: bool,
}

impl Lzvn<'_> {
    /// Decodes opcodes until the end of stream opcode, an error, or the end of either buffer.
    /// `sp` and `dp` are left after the last opcode done in full (or at the end of the output
    /// when it filled up), as the reference code leaves `state->src` and `state->dst`.
    fn decode(&mut self) {
        if self.sp >= self.src.len() || self.dp >= self.dst.len() {
            return; // empty buffer
        }
        let mut d = self.d_prev;
        loop {
            let src_len = self.src.len() - self.sp;
            if src_len == 0 {
                return;
            }
            let opc = self.src[self.sp];
            let op = |shift: u32, bits: u32| usize::from(opc >> shift) & ((1 << bits) - 1);
            let (opc_len, l, m);
            match opc {
                // eos
                0x06 => {
                    if src_len < 8 {
                        return; // source truncated
                    }
                    self.sp += 8;
                    self.end_of_stream = true;
                    self.d_prev = d;
                    return;
                }
                // nop
                0x0e | 0x16 => {
                    if src_len <= 1 {
                        return;
                    }
                    self.sp += 1;
                    continue;
                }
                // udef
                0x1e | 0x26 | 0x2e | 0x36 | 0x3e | 0x70..=0x7f | 0xd0..=0xdf => return,
                // lrg_l, sml_l
                0xe0..=0xef => {
                    let (opc_len, l) = if opc == 0xe0 {
                        if src_len <= 2 {
                            return;
                        }
                        (2, usize::from(self.src[self.sp + 1]) + 16)
                    } else {
                        (1, op(0, 4))
                    };
                    if src_len <= opc_len + l {
                        return; // source truncated
                    }
                    if !self.copy_literal(opc_len, l) {
                        return;
                    }
                    continue;
                }
                // lrg_m, sml_m
                0xf0..=0xff => {
                    let (opc_len, m) = if opc == 0xf0 {
                        if src_len <= 2 {
                            return;
                        }
                        (2, usize::from(self.src[self.sp + 1]) + 16)
                    } else {
                        if src_len <= 1 {
                            return;
                        }
                        (1, op(0, 4))
                    };
                    self.sp += opc_len;
                    if !self.copy_match(d, m) {
                        return;
                    }
                    continue;
                }
                // med_d
                0xa0..=0xbf => {
                    opc_len = 3;
                    l = op(3, 2);
                    if src_len <= opc_len + l {
                        return;
                    }
                    let opc23 = usize::from(u16::from_le_bytes([
                        self.src[self.sp + 1],
                        self.src[self.sp + 2],
                    ]));
                    m = ((op(0, 3) << 2) | (opc23 & 3)) + 3;
                    d = opc23 >> 2;
                }
                // lrg_d
                _ if opc & 7 == 7 => {
                    opc_len = 3;
                    l = op(6, 2);
                    m = op(3, 3) + 3;
                    if src_len <= opc_len + l {
                        return;
                    }
                    d = usize::from(u16::from_le_bytes([
                        self.src[self.sp + 1],
                        self.src[self.sp + 2],
                    ]));
                }
                // pre_d
                _ if opc & 7 == 6 => {
                    opc_len = 1;
                    l = op(6, 2);
                    m = op(3, 3) + 3;
                    if src_len <= opc_len + l {
                        return;
                    }
                }
                // sml_d
                _ => {
                    opc_len = 2;
                    l = op(6, 2);
                    m = op(3, 3) + 3;
                    if src_len <= opc_len + l {
                        return;
                    }
                    d = (op(0, 3) << 8) | usize::from(self.src[self.sp + 1]);
                }
            }

            // copy_literal_and_match
            if !self.copy_literal(opc_len, l) {
                return;
            }
            // Matches may not reference data before the beginning of the output.
            if d > self.dp || d == 0 {
                return; // invalid match distance
            }
            if !self.copy_match(d, m) {
                return;
            }
            self.d_prev = d;
        }
    }

    /// Copies an `l` byte literal that follows an `opc_len` byte opcode. When the output
    /// fills up first, fills it and returns false.
    fn copy_literal(&mut self, opc_len: usize, l: usize) -> bool {
        let room = self.dst.len() - self.dp;
        let n = l.min(room);
        let from = self.sp + opc_len;
        self.dst[self.dp..self.dp + n].copy_from_slice(&self.src[from..from + n]);
        if n < l {
            self.dp += n;
            return false; // destination truncated
        }
        self.sp += opc_len + l;
        self.dp += l;
        true
    }

    /// Copies an `m` byte match from `d` bytes back. When the output fills up first, fills
    /// it and returns false.
    fn copy_match(&mut self, d: usize, m: usize) -> bool {
        let room = self.dst.len() - self.dp;
        let n = m.min(room);
        copy_match(self.dst, self.dp, d, n);
        self.dp += n;
        n == m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `printf 'hello hello hello hello\n' | compression_tool -encode -a lzfse`, an LZVN
    /// block.
    const HELLO_LZVN: &[u8] = &[
        0x62, 0x76, 0x78, 0x6e, 0x18, 0x00, 0x00, 0x00, 0x14, 0x00, 0x00, 0x00, 0xe6, 0x68, 0x65,
        0x6c, 0x6c, 0x6f, 0x20, 0x38, 0x06, 0xf7, 0xe1, 0x0a, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x62, 0x76, 0x78, 0x24,
    ];

    #[test]
    fn lzvn_block() {
        let want = b"hello hello hello hello\n";
        let mut out = vec![0u8; want.len()];
        assert_eq!(uncompress(HELLO_LZVN, &mut out), Some(want.len()));
        assert_eq!(&out, want);
        // A larger buffer gets what the stream has, a smaller one is filled.
        let mut big = vec![0u8; 100];
        assert_eq!(uncompress(HELLO_LZVN, &mut big), Some(want.len()));
        let mut small = vec![0u8; 10];
        assert_eq!(uncompress(HELLO_LZVN, &mut small), Some(10));
        assert_eq!(&small, &want[..10]);
        // No end of stream block, or a bad magic, is an error.
        assert_eq!(uncompress(&HELLO_LZVN[..HELLO_LZVN.len() - 4], &mut out), None);
        assert_eq!(uncompress(b"bvxz", &mut out), None);
    }

    #[test]
    fn stored_block() {
        let mut s = b"bvx-\x05\x00\x00\x00abcde".to_vec();
        s.extend_from_slice(b"bvx$");
        let mut out = [0u8; 5];
        assert_eq!(uncompress(&s, &mut out), Some(5));
        assert_eq!(&out, b"abcde");
    }

    #[test]
    fn base_values_match_reference() {
        assert_eq!(L_BASE_VALUE[16..], [16, 20, 28, 60]);
        assert_eq!(M_BASE_VALUE[16..], [16, 24, 56, 312]);
        assert_eq!(D_BASE_VALUE[60..], [131068, 163836, 196604, 229372]);
    }

    /// Round trips through Apple's compression_tool, when it is there: bvx2 blocks for large
    /// compressible input, stored blocks for random input.
    #[test]
    fn against_compression_tool() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let mut data = Vec::new();
        let mut x: u32 = 7;
        for i in 0..600_000u32 {
            x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
            match (i / 50_000) % 3 {
                0 => data.push((x >> 16) as u8),
                1 => data.extend_from_slice(format!("line {} ", i % 1000).as_bytes()),
                _ => data.push(b"abcdefgh"[(x >> 29) as usize]),
            }
        }
        for len in [100, 5000, data.len()] {
            let input = data[..len].to_vec();
            let mut child = match Command::new("compression_tool")
                .args(["-encode", "-a", "lzfse"])
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
            {
                Ok(c) => c,
                Err(_) => {
                    eprintln!("compression_tool not found, skipping");
                    return;
                }
            };
            let mut stdin = child.stdin.take().unwrap();
            let feed = input.clone();
            let writer = std::thread::spawn(move || stdin.write_all(&feed));
            let packed = child.wait_with_output().unwrap().stdout;
            writer.join().unwrap().unwrap();
            let mut out = vec![0u8; len];
            assert_eq!(uncompress(&packed, &mut out), Some(len), "length {len}");
            assert!(out == input, "length {len}");
        }
    }
}
