// SPDX-License-Identifier: MIT OR Apache-2.0

//! Descriptor chains as the device sees them, and cursors that copy data in and out of them.

use std::fmt;

use crate::error::QueueError;
use crate::memory::GuestMemory;

/// One guest buffer of a descriptor chain.
///
/// This is the resolved form: indirect tables have already been expanded and the ring specific
/// flags are gone. What is left is where the buffer is, how long it is and who may write it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Descriptor {
    addr: u64,
    len: u32,
    write_only: bool,
}

impl Descriptor {
    /// A buffer of `len` bytes at `addr`, writable by the device if `write_only` is set.
    #[must_use]
    pub fn new(addr: u64, len: u32, write_only: bool) -> Self {
        Self { addr, len, write_only }
    }

    /// Guest physical address of the buffer.
    #[must_use]
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// Length of the buffer in bytes.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Whether the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the device may write the buffer (and must not read it).
    #[must_use]
    pub fn is_write_only(&self) -> bool {
        self.write_only
    }
}

/// A buffer the driver made available, walked and checked.
///
/// Popping a chain from a queue walks every descriptor, follows an indirect table if there is one,
/// and checks the rules from the specification before handing it over. So a `DescriptorChain` is
/// always well formed: it has at least one descriptor, and all device readable descriptors come
/// before all device writable ones.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DescriptorChain {
    head: u16,
    ring_slots: u16,
    descriptors: Vec<Descriptor>,
    readable: usize,
}

impl DescriptorChain {
    pub(crate) fn new(head: u16, ring_slots: u16, builder: ChainBuilder) -> Self {
        Self { head, ring_slots, readable: builder.readable, descriptors: builder.descriptors }
    }

    /// The value to hand back when the chain is used.
    ///
    /// For a split queue this is the index of the head descriptor. For a packed queue it is the
    /// buffer ID the driver put in the last descriptor of the chain.
    #[must_use]
    pub fn head(&self) -> u16 {
        self.head
    }

    /// How many ring entries the chain took up.
    ///
    /// A packed queue needs this back in `add_used` to know how far to move its used position. For
    /// a split queue it is the number of descriptors read from the main table, which is only of
    /// interest for statistics.
    #[must_use]
    pub fn ring_slots(&self) -> u16 {
        self.ring_slots
    }

    /// Every buffer of the chain in order.
    #[must_use]
    pub fn descriptors(&self) -> &[Descriptor] {
        &self.descriptors
    }

    /// Iterates over every buffer of the chain in order.
    pub fn iter(&self) -> std::slice::Iter<'_, Descriptor> {
        self.descriptors.iter()
    }

    /// Number of buffers in the chain, counting the entries of an indirect table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    /// Always false, since the queues never produce an empty chain. Here for completeness.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// The device readable buffers.
    #[must_use]
    pub fn readable(&self) -> &[Descriptor] {
        &self.descriptors[..self.readable]
    }

    /// The device writable buffers.
    #[must_use]
    pub fn writable(&self) -> &[Descriptor] {
        &self.descriptors[self.readable..]
    }

    /// Total length of the device readable buffers.
    #[must_use]
    pub fn readable_len(&self) -> u64 {
        self.readable().iter().map(|d| u64::from(d.len)).sum()
    }

    /// Total length of the device writable buffers.
    #[must_use]
    pub fn writable_len(&self) -> u64 {
        self.writable().iter().map(|d| u64::from(d.len)).sum()
    }

    /// A cursor that reads the device readable buffers as one byte stream.
    pub fn reader<'a, M: GuestMemory + ?Sized>(&'a self, mem: &'a M) -> Reader<'a, M> {
        Reader { cursor: Cursor::new(self.readable()), mem }
    }

    /// A cursor that writes the device writable buffers as one byte stream.
    pub fn writer<'a, M: GuestMemory + ?Sized>(&'a self, mem: &'a M) -> Writer<'a, M> {
        Writer { cursor: Cursor::new(self.writable()), mem }
    }
}

impl<'a> IntoIterator for &'a DescriptorChain {
    type Item = &'a Descriptor;
    type IntoIter = std::slice::Iter<'a, Descriptor>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Collects descriptors while a ring is walked and enforces the ordering rule.
#[derive(Debug, Default)]
pub(crate) struct ChainBuilder {
    descriptors: Vec<Descriptor>,
    readable: usize,
}

impl ChainBuilder {
    pub(crate) fn push(&mut self, addr: u64, len: u32, write_only: bool) -> Result<(), QueueError> {
        if addr.checked_add(u64::from(len)).is_none() {
            return Err(QueueError::BufferOverflow { addr, len });
        }
        if write_only {
            self.descriptors.push(Descriptor::new(addr, len, true));
        } else {
            // The specification puts every device readable buffer before the writable ones.
            if self.readable != self.descriptors.len() {
                return Err(QueueError::ReadableAfterWritable);
            }
            self.descriptors.push(Descriptor::new(addr, len, false));
            self.readable += 1;
        }
        Ok(())
    }
}

/// Position within a list of buffers.
#[derive(Clone, Debug)]
struct Cursor<'a> {
    bufs: &'a [Descriptor],
    index: usize,
    offset: u32,
    done: u64,
}

impl<'a> Cursor<'a> {
    fn new(bufs: &'a [Descriptor]) -> Self {
        Self { bufs, index: 0, offset: 0, done: 0 }
    }

    fn remaining(&self) -> u64 {
        let total: u64 =
            self.bufs[self.index.min(self.bufs.len())..].iter().map(|d| u64::from(d.len)).sum();
        total - u64::from(self.offset)
    }

    /// The next contiguous piece of at most `max` bytes, without consuming it.
    fn next_piece(&mut self, max: usize) -> Option<(u64, usize)> {
        while let Some(d) = self.bufs.get(self.index) {
            let left = d.len - self.offset;
            if left == 0 {
                self.index += 1;
                self.offset = 0;
                continue;
            }
            let n = max.min(left as usize);
            return Some((d.addr + u64::from(self.offset), n));
        }
        None
    }

    fn consume(&mut self, n: usize) {
        // `n` is never more than the piece `next_piece` returned, which fits in u32.
        self.offset += n as u32;
        self.done += n as u64;
    }

    fn skip(&mut self, mut n: u64) -> u64 {
        let mut skipped = 0;
        while n > 0 {
            let Some((_, piece)) = self.next_piece(usize::try_from(n).unwrap_or(usize::MAX)) else {
                break;
            };
            self.consume(piece);
            n -= piece as u64;
            skipped += piece as u64;
        }
        skipped
    }
}

/// Reads the device readable part of a chain as one stream of bytes.
///
/// Buffer boundaries are invisible to the caller: a read that starts near the end of one buffer
/// continues in the next.
pub struct Reader<'a, M: GuestMemory + ?Sized> {
    cursor: Cursor<'a>,
    mem: &'a M,
}

impl<M: GuestMemory + ?Sized> fmt::Debug for Reader<'_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Reader")
            .field("bytes_read", &self.cursor.done)
            .field("remaining", &self.cursor.remaining())
            .finish_non_exhaustive()
    }
}

impl<M: GuestMemory + ?Sized> Reader<'_, M> {
    /// Copies as much as fits into `buf` and returns how many bytes that was.
    ///
    /// Returns less than `buf.len()` only when the readable buffers run out.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, QueueError> {
        let mut filled = 0;
        while filled < buf.len() {
            let Some((addr, n)) = self.cursor.next_piece(buf.len() - filled) else {
                break;
            };
            self.mem.read(addr, &mut buf[filled..filled + n])?;
            self.cursor.consume(n);
            filled += n;
        }
        Ok(filled)
    }

    /// Fills all of `buf`, or fails without consuming anything if the chain is too short.
    pub fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), QueueError> {
        let available = self.remaining();
        if (buf.len() as u64) > available {
            return Err(QueueError::ChainExhausted { wanted: buf.len(), available });
        }
        self.read(buf).map(|_| ())
    }

    /// Reads everything that is left into a vector.
    pub fn read_to_vec(&mut self) -> Result<Vec<u8>, QueueError> {
        let len = usize::try_from(self.remaining()).unwrap_or(usize::MAX);
        let mut v = vec![0; len];
        let n = self.read(&mut v)?;
        v.truncate(n);
        Ok(v)
    }

    /// Moves past `n` bytes without reading them and returns how many were skipped.
    pub fn skip(&mut self, n: u64) -> u64 {
        self.cursor.skip(n)
    }

    /// Bytes left to read.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.cursor.remaining()
    }

    /// Bytes read or skipped so far.
    #[must_use]
    pub fn bytes_read(&self) -> u64 {
        self.cursor.done
    }
}

/// Writes the device writable part of a chain as one stream of bytes.
///
/// [`bytes_written`](Writer::bytes_written) is the number the device usually reports as the used
/// length.
pub struct Writer<'a, M: GuestMemory + ?Sized> {
    cursor: Cursor<'a>,
    mem: &'a M,
}

impl<M: GuestMemory + ?Sized> fmt::Debug for Writer<'_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Writer")
            .field("bytes_written", &self.cursor.done)
            .field("remaining", &self.cursor.remaining())
            .finish_non_exhaustive()
    }
}

impl<M: GuestMemory + ?Sized> Writer<'_, M> {
    /// Copies as much of `buf` as fits and returns how many bytes that was.
    pub fn write(&mut self, buf: &[u8]) -> Result<usize, QueueError> {
        let mut done = 0;
        while done < buf.len() {
            let Some((addr, n)) = self.cursor.next_piece(buf.len() - done) else {
                break;
            };
            self.mem.write(addr, &buf[done..done + n])?;
            self.cursor.consume(n);
            done += n;
        }
        Ok(done)
    }

    /// Writes all of `buf`, or fails without writing anything if the chain is too short.
    pub fn write_all(&mut self, buf: &[u8]) -> Result<(), QueueError> {
        let available = self.remaining();
        if (buf.len() as u64) > available {
            return Err(QueueError::ChainExhausted { wanted: buf.len(), available });
        }
        self.write(buf).map(|_| ())
    }

    /// Moves past `n` bytes without writing them and returns how many were skipped.
    ///
    /// Skipped bytes count towards [`bytes_written`](Writer::bytes_written), which suits a device
    /// that fills in a header after the payload.
    pub fn skip(&mut self, n: u64) -> u64 {
        self.cursor.skip(n)
    }

    /// Bytes of writable space left.
    #[must_use]
    pub fn remaining(&self) -> u64 {
        self.cursor.remaining()
    }

    /// Bytes written or skipped so far.
    #[must_use]
    pub fn bytes_written(&self) -> u64 {
        self.cursor.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::VecMemory;

    fn chain(parts: &[(u64, u32, bool)]) -> DescriptorChain {
        let mut b = ChainBuilder::default();
        for &(addr, len, w) in parts {
            b.push(addr, len, w).unwrap();
        }
        DescriptorChain::new(0, parts.len() as u16, b)
    }

    #[test]
    fn builder_enforces_order_and_overflow() {
        let mut b = ChainBuilder::default();
        b.push(0, 4, false).unwrap();
        b.push(8, 4, true).unwrap();
        assert_eq!(b.push(16, 4, false), Err(QueueError::ReadableAfterWritable));
        assert_eq!(
            b.push(u64::MAX - 1, 4, true),
            Err(QueueError::BufferOverflow { addr: u64::MAX - 1, len: 4 })
        );
        // Ending exactly at the top of the address space is fine.
        b.push(u64::MAX - 3, 3, true).unwrap();
    }

    #[test]
    fn reader_crosses_buffers() {
        let mem = VecMemory::new(64);
        mem.write(0, b"abc").unwrap();
        mem.write(10, b"").unwrap();
        mem.write(20, b"defgh").unwrap();
        let c = chain(&[(0, 3, false), (10, 0, false), (20, 5, false), (40, 8, true)]);
        assert_eq!(c.readable_len(), 8);
        assert_eq!(c.writable_len(), 8);
        let mut r = c.reader(&mem);
        let mut buf = [0; 4];
        assert_eq!(r.read(&mut buf).unwrap(), 4);
        assert_eq!(&buf, b"abcd");
        assert_eq!(r.remaining(), 4);
        assert_eq!(r.skip(1), 1);
        assert_eq!(r.read_to_vec().unwrap(), b"fgh");
        assert_eq!(r.bytes_read(), 8);
        assert_eq!(r.read(&mut buf).unwrap(), 0);
    }

    #[test]
    fn read_exact_and_write_all_refuse_short_chains() {
        let mem = VecMemory::new(64);
        let c = chain(&[(0, 2, false), (8, 3, true), (16, 3, true)]);
        let mut r = c.reader(&mem);
        let mut buf = [0; 3];
        assert_eq!(
            r.read_exact(&mut buf),
            Err(QueueError::ChainExhausted { wanted: 3, available: 2 })
        );
        assert_eq!(r.bytes_read(), 0);

        let mut w = c.writer(&mem);
        assert!(w.write_all(b"0123456").is_err());
        assert_eq!(w.bytes_written(), 0);
        w.write_all(b"01234").unwrap();
        assert_eq!(w.write(b"xyz").unwrap(), 1);
        assert_eq!(w.bytes_written(), 6);
        assert_eq!(w.remaining(), 0);
        let mut out = [0; 3];
        mem.read(8, &mut out).unwrap();
        assert_eq!(&out, b"012");
        mem.read(16, &mut out).unwrap();
        assert_eq!(&out, b"34x");
    }

    #[test]
    fn writer_skip_counts_as_written() {
        let mem = VecMemory::new(64);
        let c = chain(&[(0, 4, true), (32, 4, true)]);
        let mut w = c.writer(&mem);
        assert_eq!(w.skip(6), 6);
        w.write_all(b"zz").unwrap();
        assert_eq!(w.skip(10), 0);
        assert_eq!(w.bytes_written(), 8);
        assert_eq!(mem.read_u16(34).unwrap(), u16::from_le_bytes(*b"zz"));
        assert!(format!("{w:?}").contains("bytes_written: 8"));
    }

    #[test]
    fn chain_accessors() {
        let c = chain(&[(0, 1, false), (8, 2, true)]);
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
        assert_eq!(c.readable(), &[Descriptor::new(0, 1, false)]);
        assert_eq!(c.writable(), &[Descriptor::new(8, 2, true)]);
        assert_eq!((&c).into_iter().count(), 2);
        assert!(c.descriptors()[1].is_write_only());
        assert!(!c.descriptors()[0].is_empty());
    }
}
