// SPDX-License-Identifier: MIT OR Apache-2.0

//! A fixed size bitmap with the range operations of util/bitmap.c. Dirty memory tracking and the
//! block layer's dirty bitmaps are built on it.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bitmap {
    words: Vec<u64>,
    len: usize,
}

impl Bitmap {
    pub fn new(len: usize) -> Self {
        Bitmap { words: vec![0; len.div_ceil(64)], len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn test(&self, bit: usize) -> bool {
        assert!(bit < self.len, "bit {bit} out of range for a bitmap of {}", self.len);
        self.words[bit / 64] & (1 << (bit % 64)) != 0
    }

    pub fn set(&mut self, bit: usize) {
        assert!(bit < self.len, "bit {bit} out of range for a bitmap of {}", self.len);
        self.words[bit / 64] |= 1 << (bit % 64);
    }

    pub fn clear(&mut self, bit: usize) {
        assert!(bit < self.len, "bit {bit} out of range for a bitmap of {}", self.len);
        self.words[bit / 64] &= !(1 << (bit % 64));
    }

    /// `bitmap_set()`: sets `count` bits starting at `start`.
    pub fn set_range(&mut self, start: usize, count: usize) {
        self.apply_range(start, count, |w, m| *w |= m);
    }

    /// `bitmap_clear()`.
    pub fn clear_range(&mut self, start: usize, count: usize) {
        self.apply_range(start, count, |w, m| *w &= !m);
    }

    /// `bitmap_test_and_clear()`: clears the range and says whether any bit in it was set.
    pub fn test_and_clear_range(&mut self, start: usize, count: usize) -> bool {
        let mut any = false;
        self.apply_range(start, count, |w, m| {
            any |= *w & m != 0;
            *w &= !m;
        });
        any
    }

    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// `find_next_bit()`: the first set bit at or after `from`, or `None`.
    pub fn next_set(&self, from: usize) -> Option<usize> {
        self.next_where(from, |w| w)
    }

    /// `find_next_zero_bit()`.
    pub fn next_clear(&self, from: usize) -> Option<usize> {
        self.next_where(from, |w| !w)
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        let mut at = 0;
        std::iter::from_fn(move || {
            let bit = self.next_set(at)?;
            at = bit + 1;
            Some(bit)
        })
    }

    /// Grows or shrinks the bitmap. New bits are clear.
    pub fn resize(&mut self, len: usize) {
        if len < self.len {
            let tail = self.len - len;
            self.clear_range(len, tail);
        }
        self.words.resize(len.div_ceil(64), 0);
        self.len = len;
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    fn next_where(&self, from: usize, f: impl Fn(u64) -> u64) -> Option<usize> {
        if from >= self.len {
            return None;
        }
        let mut i = from / 64;
        let mut w = f(self.words[i]) & (u64::MAX << (from % 64));
        loop {
            if w != 0 {
                let bit = i * 64 + w.trailing_zeros() as usize;
                return (bit < self.len).then_some(bit);
            }
            i += 1;
            if i == self.words.len() {
                return None;
            }
            w = f(self.words[i]);
        }
    }

    fn apply_range(&mut self, start: usize, count: usize, mut op: impl FnMut(&mut u64, u64)) {
        if count == 0 {
            return;
        }
        let end = start.checked_add(count).expect("bitmap range overflows");
        assert!(end <= self.len, "range {start}+{count} out of range for a bitmap of {}", self.len);
        let (first, last) = (start / 64, (end - 1) / 64);
        for i in first..=last {
            let lo = if i == first { start % 64 } else { 0 };
            let hi = if i == last { (end - 1) % 64 } else { 63 };
            let mask = (u64::MAX >> (63 - hi)) & (u64::MAX << lo);
            op(&mut self.words[i], mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Bitmap;

    #[test]
    fn ranges_across_word_boundaries() {
        let mut b = Bitmap::new(200);
        b.set_range(60, 10);
        assert_eq!(b.count_ones(), 10);
        assert!(!b.test(59) && b.test(60) && b.test(69) && !b.test(70));
        assert_eq!(b.next_set(0), Some(60));
        assert_eq!(b.next_clear(60), Some(70));
        assert!(b.test_and_clear_range(65, 100));
        assert_eq!(b.iter_ones().collect::<Vec<_>>(), (60..65).collect::<Vec<_>>());
        assert!(!b.test_and_clear_range(100, 100));
    }

    #[test]
    fn against_a_vector_of_bools() {
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let len = 333;
        let mut b = Bitmap::new(len);
        let mut model = vec![false; len];
        for _ in 0..5000 {
            let start = (rnd() % len as u64) as usize;
            let count = (rnd() % (len - start) as u64) as usize;
            match rnd() % 3 {
                0 => {
                    b.set_range(start, count);
                    model[start..start + count].iter_mut().for_each(|x| *x = true);
                }
                1 => {
                    b.clear_range(start, count);
                    model[start..start + count].iter_mut().for_each(|x| *x = false);
                }
                _ => {
                    let want = model[start..start + count].iter().any(|&x| x);
                    assert_eq!(b.test_and_clear_range(start, count), want);
                    model[start..start + count].iter_mut().for_each(|x| *x = false);
                }
            }
            let from = (rnd() % len as u64) as usize;
            assert_eq!(b.next_set(from), (from..len).find(|&i| model[i]));
            assert_eq!(b.next_clear(from), (from..len).find(|&i| !model[i]));
        }
        assert_eq!(b.count_ones(), model.iter().filter(|&&x| x).count());
    }

    #[test]
    fn shrinking_drops_the_tail() {
        let mut b = Bitmap::new(130);
        b.set_range(0, 130);
        b.resize(65);
        b.resize(130);
        assert_eq!(b.count_ones(), 65);
    }
}
