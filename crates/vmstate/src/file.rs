// SPDX-License-Identifier: GPL-2.0-or-later

//! `QEMUFile`, migration/qemu-file.c, over a byte buffer or a channel.
//!
//! QEMU keeps a sticky error in the file instead of failing each call. A read past the end of the
//! stream returns zeros and records `-EIO`, a write after an error is dropped, and the caller
//! checks `qemu_file_get_error()` at the points where it matters. The VMState loader depends on
//! exactly that behaviour (it loads a field, then asks whether the stream went bad), so these types
//! keep it rather than returning a `Result` from every accessor.

use std::borrow::Cow;
use std::fmt;
use std::io::{self, Read};

/// `EIO`, the error a read past the end of the stream leaves behind.
pub const EIO: i32 = 5;
/// `EINVAL`, the error the loader leaves in the stream when a field fails to load.
pub const EINVAL: i32 = 22;

/// The writing side of a `QEMUFile`, collecting the stream in a `Vec<u8>`.
///
/// A migration sends the stream in pieces: [`take`](Self::take) hands over what was written so
/// far and leaves the writer empty, while [`transferred`](Self::transferred) keeps counting from
/// the start of the stream.
#[derive(Debug, Default, Clone)]
pub struct StreamWriter {
    buf: Vec<u8>,
    taken: u64,
    last_error: i32,
}

impl StreamWriter {
    /// `qemu_file_new_output()` over an empty buffer.
    pub fn new() -> Self {
        StreamWriter::default()
    }

    /// `qemu_file_get_error()`: zero, or the first negative errno recorded.
    pub fn get_error(&self) -> i32 {
        self.last_error
    }

    /// `qemu_file_set_error()`. Only the first error sticks.
    pub fn set_error(&mut self, ret: i32) {
        if self.last_error == 0 {
            self.last_error = ret;
        }
    }

    /// `qemu_file_transferred()`: the number of bytes written so far, including any already
    /// taken out.
    pub fn transferred(&self) -> u64 {
        self.taken + self.buf.len() as u64
    }

    /// The bytes written and not taken yet.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Takes the stream out of the writer.
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }

    /// Takes the bytes written since the last call, the way `qemu_fflush()` hands its buffer to
    /// the channel.
    pub fn take(&mut self) -> Vec<u8> {
        let buf = std::mem::take(&mut self.buf);
        self.taken += buf.len() as u64;
        buf
    }

    /// `qemu_put_byte()`.
    pub fn put_byte(&mut self, v: u8) {
        if self.last_error != 0 {
            return;
        }
        self.buf.push(v);
    }

    /// `qemu_put_buffer()`.
    pub fn put_buffer(&mut self, buf: &[u8]) {
        if self.last_error != 0 {
            return;
        }
        self.buf.extend_from_slice(buf);
    }

    /// `qemu_put_be16()`.
    pub fn put_be16(&mut self, v: u16) {
        self.put_buffer(&v.to_be_bytes());
    }

    /// `qemu_put_be32()`.
    pub fn put_be32(&mut self, v: u32) {
        self.put_buffer(&v.to_be_bytes());
    }

    /// `qemu_put_be64()`.
    pub fn put_be64(&mut self, v: u64) {
        self.put_buffer(&v.to_be_bytes());
    }
}

/// How much a reader with a channel behind it asks for at a time.
const IO_BUF_SIZE: usize = 32768;

/// The reading side of a `QEMUFile`.
///
/// It reads either from a byte slice that holds the whole stream, or from a channel through a
/// buffer that is refilled as the loader peeks ahead, like `qemu_fill_buffer()`. Either way the
/// end of the data is a sticky `-EIO`.
pub struct StreamReader<'a> {
    buf: Cow<'a, [u8]>,
    pos: usize,
    // Bytes dropped from the front of `buf` when it was compacted.
    dropped: u64,
    src: Option<Box<dyn Read + Send + 'a>>,
    last_error: i32,
}

impl fmt::Debug for StreamReader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamReader")
            .field("position", &self.position())
            .field("buffered", &(self.buf.len() - self.pos))
            .field("channel", &self.src.is_some())
            .field("last_error", &self.last_error)
            .finish()
    }
}

impl<'a> StreamReader<'a> {
    /// `qemu_file_new_input()` over `buf`, which holds the whole stream.
    pub fn new(buf: &'a [u8]) -> Self {
        StreamReader { buf: Cow::Borrowed(buf), pos: 0, dropped: 0, src: None, last_error: 0 }
    }

    /// `qemu_file_new_input()` over a channel. Reads block until the channel has the bytes or
    /// reports end of file.
    pub fn from_reader(src: impl Read + Send + 'a) -> Self {
        StreamReader {
            buf: Cow::Owned(Vec::new()),
            pos: 0,
            dropped: 0,
            src: Some(Box::new(src)),
            last_error: 0,
        }
    }

    /// `qemu_file_get_error()`: zero, or the first negative errno recorded.
    pub fn get_error(&self) -> i32 {
        self.last_error
    }

    /// `qemu_file_set_error()`. Only the first error sticks.
    pub fn set_error(&mut self, ret: i32) {
        if self.last_error == 0 {
            self.last_error = ret;
        }
    }

    /// How far into the stream the reader is.
    pub fn position(&self) -> u64 {
        self.dropped + self.pos as u64
    }

    /// The bytes read from the source and not consumed yet. Over a slice this is the rest of
    /// the stream.
    pub fn remaining(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    /// `qemu_fill_buffer()`: makes sure `n` bytes past the read position are buffered, if the
    /// channel has them. Returns whether they are.
    fn fill(&mut self, n: usize) -> bool {
        if self.buf.len() - self.pos >= n {
            return true;
        }
        let Some(src) = self.src.as_mut() else {
            return false;
        };
        if self.last_error != 0 {
            return false;
        }
        let buf = self.buf.to_mut();
        if self.pos > 0 {
            buf.drain(..self.pos);
            self.dropped += self.pos as u64;
            self.pos = 0;
        }
        while buf.len() < n {
            let old = buf.len();
            buf.resize(old + (n - old).max(IO_BUF_SIZE), 0);
            let got = loop {
                match src.read(&mut buf[old..]) {
                    Ok(got) => break Ok(got),
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => break Err(e),
                }
            };
            match got {
                Ok(got) if got > 0 => buf.truncate(old + got),
                _ => {
                    buf.truncate(old);
                    // A failed channel is the same -EIO as the end of the stream.
                    self.src = None;
                    return false;
                }
            }
        }
        true
    }

    /// What `qemu_fill_buffer()` does when the channel has nothing more to give.
    fn hit_eof(&mut self) {
        self.set_error(-EIO);
    }

    /// `qemu_peek_byte()`. Past the end of the stream it returns 0 and records `-EIO`.
    pub fn peek_byte(&mut self, offset: usize) -> u8 {
        if !self.fill(offset + 1) {
            self.hit_eof();
            return 0;
        }
        self.buf[self.pos + offset]
    }

    /// `qemu_peek_buffer()`: up to `size` bytes starting `offset` bytes ahead, without consuming
    /// them. A short result means the stream ended, and `-EIO` is recorded.
    pub fn peek_buffer(&mut self, size: usize, offset: usize) -> &[u8] {
        if !self.fill(offset.saturating_add(size)) {
            self.hit_eof();
        }
        let start = (self.pos + offset).min(self.buf.len());
        let end = start.saturating_add(size).min(self.buf.len());
        &self.buf[start..end]
    }

    /// `qemu_file_skip()`. Like QEMU it does nothing when fewer than `size` bytes are left.
    pub fn skip(&mut self, size: usize) {
        if self.fill(size) {
            self.pos += size;
        }
    }

    /// `qemu_get_byte()`.
    pub fn get_byte(&mut self) -> u8 {
        let b = self.peek_byte(0);
        self.skip(1);
        b
    }

    /// `qemu_get_buffer()`: fills as much of `buf` as the stream allows and returns how much
    /// that was. The rest of `buf` is left alone.
    pub fn get_buffer(&mut self, buf: &mut [u8]) -> usize {
        let src = self.peek_buffer(buf.len(), 0);
        let n = src.len();
        buf[..n].copy_from_slice(src);
        self.pos += n;
        n
    }

    /// `qemu_get_be16()`.
    pub fn get_be16(&mut self) -> u16 {
        let hi = u16::from(self.get_byte());
        let lo = u16::from(self.get_byte());
        (hi << 8) | lo
    }

    /// `qemu_get_be32()`.
    pub fn get_be32(&mut self) -> u32 {
        let mut v = 0;
        for _ in 0..4 {
            v = (v << 8) | u32::from(self.get_byte());
        }
        v
    }

    /// `qemu_get_be64()`.
    pub fn get_be64(&mut self) -> u64 {
        let hi = u64::from(self.get_be32());
        let lo = u64::from(self.get_be32());
        (hi << 32) | lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn big_endian_round_trip() {
        let mut w = StreamWriter::new();
        w.put_byte(0x82);
        w.put_be16(0x0200);
        w.put_be32(70000);
        w.put_be64(u64::MAX - 1);
        let bytes = w.into_inner();
        assert_eq!(bytes[..7], [0x82, 0x02, 0x00, 0x00, 0x01, 0x11, 0x70]);

        let mut r = StreamReader::new(&bytes);
        assert_eq!(r.get_byte(), 0x82);
        assert_eq!(r.get_be16(), 0x0200);
        assert_eq!(r.get_be32(), 70000);
        assert_eq!(r.get_be64(), u64::MAX - 1);
        assert_eq!(r.get_error(), 0);
    }

    #[test]
    fn reading_past_the_end_is_a_sticky_eio() {
        let mut r = StreamReader::new(&[0x12]);
        assert_eq!(r.get_be16(), 0x1200);
        assert_eq!(r.get_error(), -EIO);
        r.set_error(-EINVAL);
        assert_eq!(r.get_error(), -EIO);
    }

    #[test]
    fn short_skip_does_not_move() {
        let mut r = StreamReader::new(&[1, 2, 3]);
        r.skip(4);
        assert_eq!(r.position(), 0);
        r.skip(3);
        assert_eq!(r.position(), 3);
    }

    /// A channel that hands out at most `chunk` bytes per read.
    struct Trickle<'a> {
        data: &'a [u8],
        chunk: usize,
    }

    impl Read for Trickle<'_> {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.chunk).min(self.data.len());
            buf[..n].copy_from_slice(&self.data[..n]);
            self.data = &self.data[n..];
            Ok(n)
        }
    }

    #[test]
    fn a_channel_reader_refills_as_it_peeks() {
        let data: Vec<u8> = (0..100_000u32).map(|i| i as u8).collect();
        let mut r = StreamReader::from_reader(Trickle { data: &data, chunk: 7 });
        assert_eq!(r.peek_buffer(5, 3), [3, 4, 5, 6, 7]);
        assert_eq!(r.get_be32(), 0x0001_0203);
        let mut big = vec![0; 70_000];
        assert_eq!(r.get_buffer(&mut big), 70_000);
        assert_eq!(big[..2], [4, 5]);
        assert_eq!(r.position(), 70_004);
        r.skip(29_995);
        assert_eq!(r.get_byte(), data[99_999]);
        assert_eq!(r.get_error(), 0);
        assert_eq!(r.get_byte(), 0);
        assert_eq!(r.get_error(), -EIO);
    }

    #[test]
    fn writer_take_keeps_the_count() {
        let mut w = StreamWriter::new();
        w.put_be32(1);
        assert_eq!(w.take(), [0, 0, 0, 1]);
        w.put_byte(2);
        assert_eq!(w.transferred(), 5);
        assert_eq!(w.as_bytes(), [2]);
    }

    #[test]
    fn writes_after_an_error_are_dropped() {
        let mut w = StreamWriter::new();
        w.put_byte(1);
        w.set_error(-EIO);
        w.put_be32(2);
        assert_eq!(w.as_bytes(), [1]);
    }
}
