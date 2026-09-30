// SPDX-License-Identifier: MIT OR Apache-2.0

//! The event index comparison shared by both ring formats.

/// Decides whether moving an index from `old` to `new` passed the point `event`.
///
/// With `VIRTIO_F_EVENT_IDX` one side publishes the index at which it wants to hear from the other
/// side. The other side advances its own index from `old` to `new` and has to notify when `event`
/// lies in the half open range `[old, new)`. All three values are free running 16 bit counters, so
/// the comparison is done with wrapping arithmetic: the distance from `event` to `new` must be
/// smaller than the distance the index moved.
///
/// ```
/// use ruvm_virtio_queue::need_event;
///
/// // The driver asked to hear about index 5. Moving from 3 to 6 crosses it.
/// assert!(need_event(5, 6, 3));
/// // Moving from 6 to 8 does not, it was already passed.
/// assert!(!need_event(5, 8, 6));
/// // The counters wrap at 65536.
/// assert!(need_event(0xffff, 2, 0xfffe));
/// ```
#[must_use]
pub fn need_event(event: u16, new: u16, old: u16) -> bool {
    new.wrapping_sub(event).wrapping_sub(1) < new.wrapping_sub(old)
}

#[cfg(test)]
mod tests {
    use super::need_event;

    /// Checks the wrapping formula against a direct walk over the indices that were crossed.
    fn by_walking(event: u16, new: u16, old: u16) -> bool {
        let mut i = old;
        while i != new {
            if i == event {
                return true;
            }
            i = i.wrapping_add(1);
        }
        false
    }

    #[test]
    fn matches_a_walk_over_many_points() {
        let points = [0u16, 1, 2, 5, 100, 0x7fff, 0x8000, 0xfffe, 0xffff];
        for &old in &points {
            for delta in [0u16, 1, 2, 3, 7, 200] {
                let new = old.wrapping_add(delta);
                for offset in [0u16, 1, 2, 3, 6, 7, 8, 199, 200, 201, 0xffff] {
                    let event = old.wrapping_add(offset);
                    assert_eq!(
                        need_event(event, new, old),
                        by_walking(event, new, old),
                        "event {event} new {new} old {old}"
                    );
                }
            }
        }
    }

    #[test]
    fn nothing_moved_means_no_event() {
        assert!(!need_event(3, 3, 3));
        assert!(!need_event(4, 3, 3));
    }
}
