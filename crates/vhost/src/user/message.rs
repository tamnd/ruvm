// SPDX-License-Identifier: MIT OR Apache-2.0

//! The vhost-user wire format: request codes, header flags, protocol feature bits and the
//! payload layouts, all from the "Message Specification" section of vhost-user.rst.
//!
//! Every number on the wire is in the host's byte order, since both ends run on the same
//! machine.

/// Frontend request codes, the requests the VMM sends to the backend.
pub mod request {
    /// `VHOST_USER_GET_FEATURES`
    pub const GET_FEATURES: u32 = 1;
    /// `VHOST_USER_SET_FEATURES`
    pub const SET_FEATURES: u32 = 2;
    /// `VHOST_USER_SET_OWNER`
    pub const SET_OWNER: u32 = 3;
    /// `VHOST_USER_RESET_OWNER`
    pub const RESET_OWNER: u32 = 4;
    /// `VHOST_USER_SET_MEM_TABLE`
    pub const SET_MEM_TABLE: u32 = 5;
    /// `VHOST_USER_SET_LOG_BASE`
    pub const SET_LOG_BASE: u32 = 6;
    /// `VHOST_USER_SET_LOG_FD`
    pub const SET_LOG_FD: u32 = 7;
    /// `VHOST_USER_SET_VRING_NUM`
    pub const SET_VRING_NUM: u32 = 8;
    /// `VHOST_USER_SET_VRING_ADDR`
    pub const SET_VRING_ADDR: u32 = 9;
    /// `VHOST_USER_SET_VRING_BASE`
    pub const SET_VRING_BASE: u32 = 10;
    /// `VHOST_USER_GET_VRING_BASE`
    pub const GET_VRING_BASE: u32 = 11;
    /// `VHOST_USER_SET_VRING_KICK`
    pub const SET_VRING_KICK: u32 = 12;
    /// `VHOST_USER_SET_VRING_CALL`
    pub const SET_VRING_CALL: u32 = 13;
    /// `VHOST_USER_SET_VRING_ERR`
    pub const SET_VRING_ERR: u32 = 14;
    /// `VHOST_USER_GET_PROTOCOL_FEATURES`
    pub const GET_PROTOCOL_FEATURES: u32 = 15;
    /// `VHOST_USER_SET_PROTOCOL_FEATURES`
    pub const SET_PROTOCOL_FEATURES: u32 = 16;
    /// `VHOST_USER_GET_QUEUE_NUM`
    pub const GET_QUEUE_NUM: u32 = 17;
    /// `VHOST_USER_SET_VRING_ENABLE`
    pub const SET_VRING_ENABLE: u32 = 18;
    /// `VHOST_USER_SEND_RARP`
    pub const SEND_RARP: u32 = 19;
    /// `VHOST_USER_NET_SET_MTU`
    pub const NET_SET_MTU: u32 = 20;
    /// `VHOST_USER_SET_BACKEND_REQ_FD`, once called `SET_SLAVE_REQ_FD`.
    pub const SET_BACKEND_REQ_FD: u32 = 21;
    /// `VHOST_USER_IOTLB_MSG`
    pub const IOTLB_MSG: u32 = 22;
    /// `VHOST_USER_SET_VRING_ENDIAN`
    pub const SET_VRING_ENDIAN: u32 = 23;
    /// `VHOST_USER_GET_CONFIG`
    pub const GET_CONFIG: u32 = 24;
    /// `VHOST_USER_SET_CONFIG`
    pub const SET_CONFIG: u32 = 25;
    /// `VHOST_USER_RESET_DEVICE`
    pub const RESET_DEVICE: u32 = 34;
    /// `VHOST_USER_VRING_KICK`
    pub const VRING_KICK: u32 = 35;
    /// `VHOST_USER_GET_MAX_MEM_SLOTS`
    pub const GET_MAX_MEM_SLOTS: u32 = 36;
    /// `VHOST_USER_ADD_MEM_REG`
    pub const ADD_MEM_REG: u32 = 37;
    /// `VHOST_USER_REM_MEM_REG`
    pub const REM_MEM_REG: u32 = 38;
    /// `VHOST_USER_SET_STATUS`
    pub const SET_STATUS: u32 = 39;
    /// `VHOST_USER_GET_STATUS`
    pub const GET_STATUS: u32 = 40;
}

/// Backend request codes, the requests a backend sends on the channel set up with
/// `SET_BACKEND_REQ_FD`.
pub mod backend_request {
    /// `VHOST_USER_BACKEND_IOTLB_MSG`
    pub const IOTLB_MSG: u32 = 1;
    /// `VHOST_USER_BACKEND_CONFIG_CHANGE_MSG`
    pub const CONFIG_CHANGE_MSG: u32 = 2;
    /// `VHOST_USER_BACKEND_VRING_HOST_NOTIFIER_MSG`
    pub const VRING_HOST_NOTIFIER_MSG: u32 = 3;
    /// `VHOST_USER_BACKEND_VRING_CALL`
    pub const VRING_CALL: u32 = 4;
    /// `VHOST_USER_BACKEND_VRING_ERR`
    pub const VRING_ERR: u32 = 5;
}

/// Protocol feature bits, as masks, from `GET_PROTOCOL_FEATURES`.
pub mod protocol {
    /// `VHOST_USER_PROTOCOL_F_MQ`
    pub const MQ: u64 = 1 << 0;
    /// `VHOST_USER_PROTOCOL_F_LOG_SHMFD`
    pub const LOG_SHMFD: u64 = 1 << 1;
    /// `VHOST_USER_PROTOCOL_F_RARP`
    pub const RARP: u64 = 1 << 2;
    /// `VHOST_USER_PROTOCOL_F_REPLY_ACK`
    pub const REPLY_ACK: u64 = 1 << 3;
    /// `VHOST_USER_PROTOCOL_F_MTU`
    pub const MTU: u64 = 1 << 4;
    /// `VHOST_USER_PROTOCOL_F_BACKEND_REQ`
    pub const BACKEND_REQ: u64 = 1 << 5;
    /// `VHOST_USER_PROTOCOL_F_CROSS_ENDIAN`
    pub const CROSS_ENDIAN: u64 = 1 << 6;
    /// `VHOST_USER_PROTOCOL_F_CRYPTO_SESSION`
    pub const CRYPTO_SESSION: u64 = 1 << 7;
    /// `VHOST_USER_PROTOCOL_F_PAGEFAULT`
    pub const PAGEFAULT: u64 = 1 << 8;
    /// `VHOST_USER_PROTOCOL_F_CONFIG`
    pub const CONFIG: u64 = 1 << 9;
    /// `VHOST_USER_PROTOCOL_F_BACKEND_SEND_FD`
    pub const BACKEND_SEND_FD: u64 = 1 << 10;
    /// `VHOST_USER_PROTOCOL_F_HOST_NOTIFIER`
    pub const HOST_NOTIFIER: u64 = 1 << 11;
    /// `VHOST_USER_PROTOCOL_F_INFLIGHT_SHMFD`
    pub const INFLIGHT_SHMFD: u64 = 1 << 12;
    /// `VHOST_USER_PROTOCOL_F_RESET_DEVICE`
    pub const RESET_DEVICE: u64 = 1 << 13;
    /// `VHOST_USER_PROTOCOL_F_INBAND_NOTIFICATIONS`
    pub const INBAND_NOTIFICATIONS: u64 = 1 << 14;
    /// `VHOST_USER_PROTOCOL_F_CONFIGURE_MEM_SLOTS`
    pub const CONFIGURE_MEM_SLOTS: u64 = 1 << 15;
    /// `VHOST_USER_PROTOCOL_F_STATUS`
    pub const STATUS: u64 = 1 << 16;
    /// `VHOST_USER_PROTOCOL_F_XEN_MMAP`
    pub const XEN_MMAP: u64 = 1 << 17;
    /// `VHOST_USER_PROTOCOL_F_SHARED_OBJECT`
    pub const SHARED_OBJECT: u64 = 1 << 18;
    /// `VHOST_USER_PROTOCOL_F_DEVICE_STATE`
    pub const DEVICE_STATE: u64 = 1 << 19;
    /// `VHOST_USER_PROTOCOL_F_GET_VRING_BASE_INFLIGHT`
    pub const GET_VRING_BASE_INFLIGHT: u64 = 1 << 20;
    /// `VHOST_USER_PROTOCOL_F_GPA_ADDRESSES`
    pub const GPA_ADDRESSES: u64 = 1 << 21;
    /// `VHOST_USER_PROTOCOL_F_SHMEM_MAP`
    pub const SHMEM_MAP: u64 = 1 << 22;
}

/// The protocol version, in the low two bits of the header flags.
pub const VERSION: u32 = 0x1;
/// Mask for the version bits.
pub const VERSION_MASK: u32 = 0x3;
/// Set on every reply.
pub const REPLY_FLAG: u32 = 1 << 2;
/// Set by the sender to ask for a `REPLY_ACK` status reply.
pub const NEED_REPLY_FLAG: u32 = 1 << 3;

/// In the `u64` payload of `SET_VRING_KICK`, `SET_VRING_CALL` and `SET_VRING_ERR`, the bits that
/// hold the ring index.
pub const VRING_IDX_MASK: u64 = 0xff;
/// In the same payloads, set when no eventfd comes with the message.
pub const VRING_NOFD_MASK: u64 = 1 << 8;

/// Size of the message header: request, flags and payload size, 32 bits each.
pub const HEADER_SIZE: usize = 12;

/// `VHOST_MEMORY_BASELINE_NREGIONS`: the regions one `SET_MEM_TABLE` can hold, which is also the
/// most file descriptors one message may carry.
pub const MAX_REGIONS: usize = 8;
/// The most file descriptors one message may carry.
pub const MAX_FDS: usize = MAX_REGIONS;

/// Size of one memory region description on the wire.
pub const REGION_SIZE: usize = 32;
/// Size of the vring address description on the wire.
pub const VRING_ADDR_SIZE: usize = 40;
/// Size of the header in front of the config space bytes in `GET_CONFIG` and `SET_CONFIG`.
pub const CONFIG_HEADER_SIZE: usize = 12;
/// `VHOST_USER_MAX_CONFIG_SIZE`: the largest config space access.
pub const MAX_CONFIG_SIZE: usize = 256;

/// `SET_CONFIG` flags value for a write the driver made.
pub const CONFIG_TYPE_FRONTEND: u32 = 0;
/// `SET_CONFIG` flags value for a write made while loading migration state.
pub const CONFIG_TYPE_MIGRATION: u32 = 1;

/// The largest payload this implementation accepts. The biggest message the protocol defines is
/// the shared memory config reply at a little over 2 KiB; anything much larger is a broken peer.
pub const MAX_PAYLOAD: usize = 4096;

/// A vhost-user message header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// The request code.
    pub request: u32,
    /// Version, reply and need_reply bits.
    pub flags: u32,
    /// Payload size in bytes.
    pub size: u32,
}

impl Header {
    /// A request header with the current version and no other flags.
    pub fn new(request: u32, size: u32) -> Self {
        Header { request, flags: VERSION, size }
    }

    /// The header in wire format.
    pub fn to_bytes(self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        out[0..4].copy_from_slice(&self.request.to_ne_bytes());
        out[4..8].copy_from_slice(&self.flags.to_ne_bytes());
        out[8..12].copy_from_slice(&self.size.to_ne_bytes());
        out
    }

    /// Decode a header from wire format.
    pub fn from_bytes(bytes: &[u8; HEADER_SIZE]) -> Self {
        Header { request: u32_at(bytes, 0), flags: u32_at(bytes, 4), size: u32_at(bytes, 8) }
    }

    /// The version bits.
    pub fn version(self) -> u32 {
        self.flags & VERSION_MASK
    }

    /// Whether the reply bit is set.
    pub fn is_reply(self) -> bool {
        self.flags & REPLY_FLAG != 0
    }

    /// Whether the need_reply bit is set.
    pub fn needs_reply(self) -> bool {
        self.flags & NEED_REPLY_FLAG != 0
    }
}

/// Read a native endian `u32` at `offset`.
///
/// # Panics
///
/// If `bytes` is shorter than `offset + 4`.
pub fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_ne_bytes(raw)
}

/// Read a native endian `u64` at `offset`.
///
/// # Panics
///
/// If `bytes` is shorter than `offset + 8`.
pub fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    u64::from_ne_bytes(raw)
}

/// Builds a payload out of native endian fields.
#[derive(Debug, Default)]
pub struct Payload(pub Vec<u8>);

impl Payload {
    /// Append a `u32`.
    pub fn u32(mut self, v: u32) -> Self {
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// Append a `u64`.
    pub fn u64(mut self, v: u64) -> Self {
        self.0.extend_from_slice(&v.to_ne_bytes());
        self
    }

    /// Append raw bytes.
    pub fn bytes(mut self, v: &[u8]) -> Self {
        self.0.extend_from_slice(v);
        self
    }

    /// Append a memory region description.
    pub fn region(self, r: &crate::MemoryRegion<'_>) -> Self {
        self.u64(r.guest_phys_addr).u64(r.memory_size).u64(r.userspace_addr).u64(r.mmap_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::{HEADER_SIZE, Header, NEED_REPLY_FLAG, REPLY_FLAG, VERSION};

    #[test]
    fn header_round_trips() {
        let h = Header { request: 24, flags: VERSION | NEED_REPLY_FLAG, size: 268 };
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), HEADER_SIZE);
        assert_eq!(&bytes[0..4], &24u32.to_ne_bytes());
        assert_eq!(&bytes[8..12], &268u32.to_ne_bytes());
        let back = Header::from_bytes(&bytes);
        assert_eq!(back, h);
        assert_eq!(back.version(), 1);
        assert!(back.needs_reply());
        assert!(!back.is_reply());
        assert!(Header { flags: VERSION | REPLY_FLAG, ..h }.is_reply());
    }
}
