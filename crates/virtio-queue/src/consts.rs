// SPDX-License-Identifier: MIT OR Apache-2.0

//! Flag values and layout constants from the specification, under the names it uses.

/// The descriptor continues through the `next` field (split) or the next ring slot (packed).
pub const VIRTQ_DESC_F_NEXT: u16 = 1;
/// The buffer is write only for the device. Without it the buffer is read only.
pub const VIRTQ_DESC_F_WRITE: u16 = 2;
/// The buffer holds a table of indirect descriptors.
pub const VIRTQ_DESC_F_INDIRECT: u16 = 4;

/// Packed ring only: the descriptor's avail bit.
pub const VIRTQ_DESC_F_AVAIL: u16 = 1 << 7;
/// Packed ring only: the descriptor's used bit.
pub const VIRTQ_DESC_F_USED: u16 = 1 << 15;

/// Set by the driver in the available ring flags to ask the device not to interrupt it.
pub const VIRTQ_AVAIL_F_NO_INTERRUPT: u16 = 1;
/// Set by the device in the used ring flags to ask the driver not to notify it.
pub const VIRTQ_USED_F_NO_NOTIFY: u16 = 1;

/// Packed event suppression: notifications are enabled.
pub const RING_EVENT_FLAGS_ENABLE: u16 = 0;
/// Packed event suppression: notifications are disabled.
pub const RING_EVENT_FLAGS_DISABLE: u16 = 1;
/// Packed event suppression: notify only when the ring reaches the given offset and wrap counter.
/// Only valid when `VIRTIO_F_EVENT_IDX` was negotiated.
pub const RING_EVENT_FLAGS_DESC: u16 = 2;

/// Feature bit for indirect descriptors.
pub const VIRTIO_F_INDIRECT_DESC: u32 = 28;
/// Feature bit for the `used_event` and `avail_event` fields.
pub const VIRTIO_F_EVENT_IDX: u32 = 29;
/// Feature bit for the packed ring format.
pub const VIRTIO_F_RING_PACKED: u32 = 34;

/// Size in bytes of one descriptor in either format.
pub const DESCRIPTOR_SIZE: u64 = 16;
