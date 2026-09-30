// SPDX-License-Identifier: GPL-2.0-or-later

//! In memory stand-ins for `QEMUFile`, migration/qemu-file.c.
//!
//! QEMU keeps a sticky error in the file instead of failing each call. A read past the end of the
//! stream returns zeros and records `-EIO`, a write after an error is dropped, and the caller
//! checks `qemu_file_get_error()` at the points where it matters. The VMState loader depends on
//! exactly that behaviour (it loads a field, then asks whether the stream went bad), so these types
//! keep it rather than returning a `Result` from every accessor.

/// `EIO`, the error a read past the end of the stream leaves behind.
pub const EIO: i32 = 5;
/// `EINVAL`, the error the loader leaves in the stream when a field fails to load.
pub const EINVAL: i32 = 22;

/// The writing side of a `QEMUFile`, collecting the stream in a `Vec<u8>`.
#[derive(Debug, Default, Clone)]
pub struct StreamWriter {
    buf: Vec<u8>,
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

    /// `qemu_file_transferred()`: the number of bytes written so far.
    pub fn transferred(&self) -> u64 {
        self.buf.len() as u64
    }

    /// The bytes written so far.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Takes the stream out of the writer.
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
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

/// The reading side of a `QEMUFile`, over a byte slice.
#[derive(Debug, Clone)]
pub struct StreamReader<'a> {
    buf: &'a [u8],
    pos: usize,
    last_error: i32,
}

impl<'a> StreamReader<'a> {
    /// `qemu_file_new_input()` over `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        StreamReader { buf, pos: 0, last_error: 0 }
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
    pub fn position(&self) -> usize {
        self.pos
    }

    /// The bytes not read yet.
    pub fn remaining(&self) -> &'a [u8] {
        &self.buf[self.pos..]
    }

    /// What `qemu_fill_buffer()` does when the channel has nothing more to give.
    fn hit_eof(&mut self) {
        self.set_error(-EIO);
    }

    /// `qemu_peek_byte()`. Past the end of the stream it returns 0 and records `-EIO`.
    pub fn peek_byte(&mut self, offset: usize) -> u8 {
        match self.buf.get(self.pos + offset) {
            Some(&b) => b,
            None => {
                self.hit_eof();
                0
            }
        }
    }

    /// `qemu_peek_buffer()`: up to `size` bytes starting `offset` bytes ahead, without consuming
    /// them. A short result means the stream ended, and `-EIO` is recorded.
    pub fn peek_buffer(&mut self, size: usize, offset: usize) -> &'a [u8] {
        let start = (self.pos + offset).min(self.buf.len());
        let end = start.saturating_add(size).min(self.buf.len());
        if end - start < size {
            self.hit_eof();
        }
        &self.buf[start..end]
    }

    /// `qemu_file_skip()`. Like QEMU it does nothing when fewer than `size` bytes are left.
    pub fn skip(&mut self, size: usize) {
        if self.pos + size <= self.buf.len() {
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
        self.skip(n);
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

    #[test]
    fn writes_after_an_error_are_dropped() {
        let mut w = StreamWriter::new();
        w.put_byte(1);
        w.set_error(-EIO);
        w.put_be32(2);
        assert_eq!(w.as_bytes(), [1]);
    }
}
