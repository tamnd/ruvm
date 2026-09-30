// SPDX-License-Identifier: MIT OR Apache-2.0

//! [`Frontend`], the requests the VMM sends.

use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::net::UnixStream;
use std::path::Path;

use super::backend::{BackendChannel, BackendHandler};
use super::connection::Connection;
use super::message::{
    self, CONFIG_HEADER_SIZE, Header, MAX_CONFIG_SIZE, MAX_REGIONS, NEED_REPLY_FLAG, Payload,
    REGION_SIZE, VRING_IDX_MASK, VRING_NOFD_MASK, protocol, request, u32_at, u64_at,
};

use crate::{
    Error, LogRegion, MemoryRegion, Result, VHOST_USER_F_PROTOCOL_FEATURES, VhostBackend,
    VringAddr, check_regions,
};

/// The VMM end of a vhost-user connection.
#[derive(Debug)]
pub struct Frontend {
    conn: Connection,
    /// What `GET_FEATURES` returned, once it has been asked.
    backend_features: Option<u64>,
    /// What `SET_FEATURES` last sent.
    acked_features: u64,
    /// What `GET_PROTOCOL_FEATURES` returned, once it has been asked.
    backend_protocol_features: Option<u64>,
    /// What `SET_PROTOCOL_FEATURES` sent, or `None` before it has been.
    acked_protocol_features: Option<u64>,
}

impl Frontend {
    /// Take over a connected socket.
    pub fn new(stream: UnixStream) -> Self {
        Frontend {
            conn: Connection::new(stream),
            backend_features: None,
            acked_features: 0,
            backend_protocol_features: None,
            acked_protocol_features: None,
        }
    }

    /// Connect to a backend listening on `path`.
    pub fn connect(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Frontend::new(UnixStream::connect(path)?))
    }

    /// The socket, for setting timeouts or polling.
    pub fn stream(&self) -> &UnixStream {
        self.conn.stream()
    }

    /// The features last sent with `SET_FEATURES`.
    pub fn acked_features(&self) -> u64 {
        self.acked_features
    }

    /// The protocol features both sides agreed on, zero before `SET_PROTOCOL_FEATURES`.
    pub fn acked_protocol_features(&self) -> u64 {
        self.acked_protocol_features.unwrap_or(0)
    }

    /// Whether protocol feature `bit` (a mask from [`message::protocol`]) was negotiated.
    pub fn has_protocol_feature(&self, bit: u64) -> bool {
        self.acked_protocol_features() & bit == bit
    }

    fn require(&self, bit: u64, name: &'static str) -> Result<()> {
        if self.has_protocol_feature(bit) { Ok(()) } else { Err(Error::NotNegotiated(name)) }
    }

    /// Send a request that has a reply of its own and return the reply payload, checked to be
    /// `expected` bytes long when `expected` is given.
    fn call(
        &mut self,
        req: u32,
        payload: &[u8],
        fds: &[BorrowedFd<'_>],
        expected: Option<usize>,
    ) -> Result<Vec<u8>> {
        self.conn.send(Header::new(req, wire_size(payload)?), payload, fds)?;
        self.reply(req, expected)
    }

    /// Send a request that has no reply of its own, waiting for a `REPLY_ACK` status if that
    /// was negotiated.
    fn send(&mut self, req: u32, payload: &[u8], fds: &[BorrowedFd<'_>]) -> Result<()> {
        let mut header = Header::new(req, wire_size(payload)?);
        let ack = self.has_protocol_feature(protocol::REPLY_ACK);
        if ack {
            header.flags |= NEED_REPLY_FLAG;
        }
        self.conn.send(header, payload, fds)?;
        if ack {
            let status = u64_at(&self.reply(req, Some(8))?, 0);
            if status != 0 {
                return Err(Error::BackendFailed { request: req, status });
            }
        }
        Ok(())
    }

    fn reply(&mut self, req: u32, expected: Option<usize>) -> Result<Vec<u8>> {
        let message = self.conn.recv()?;
        let header = message.header;
        if header.request != req {
            return Err(Error::UnexpectedReply { expected: req, got: header.request });
        }
        if !header.is_reply() {
            return Err(Error::BadFlags { request: req, flags: header.flags });
        }
        if let Some(size) = expected {
            if message.payload.len() != size {
                return Err(Error::BadPayloadSize {
                    request: req,
                    expected: size,
                    got: message.payload.len(),
                });
            }
        }
        Ok(message.payload)
    }

    fn get_u64(&mut self, req: u32) -> Result<u64> {
        Ok(u64_at(&self.call(req, &[], &[], Some(8))?, 0))
    }

    /// `GET_FEATURES`: the virtio and vhost-user features the backend offers.
    pub fn get_features(&mut self) -> Result<u64> {
        let features = self.get_u64(request::GET_FEATURES)?;
        self.backend_features = Some(features);
        Ok(features)
    }

    /// `SET_FEATURES`. If the backend offered [`VHOST_USER_F_PROTOCOL_FEATURES`] the bit stays
    /// set, because a backend that sees it cleared enables every ring at once instead of waiting
    /// for `SET_VRING_ENABLE`.
    pub fn set_features(&mut self, features: u64) -> Result<()> {
        let offered = self.offered_features()?;
        let features = features | (offered & VHOST_USER_F_PROTOCOL_FEATURES);
        if features & !offered != 0 {
            return Err(Error::InvalidArgument("feature the backend did not offer"));
        }
        self.send(request::SET_FEATURES, &Payload::default().u64(features).0, &[])?;
        self.acked_features = features;
        Ok(())
    }

    fn offered_features(&mut self) -> Result<u64> {
        match self.backend_features {
            Some(features) => Ok(features),
            None => self.get_features(),
        }
    }

    /// `GET_PROTOCOL_FEATURES`. Only legal when the backend offers
    /// [`VHOST_USER_F_PROTOCOL_FEATURES`], which this checks first.
    pub fn get_protocol_features(&mut self) -> Result<u64> {
        if self.offered_features()? & VHOST_USER_F_PROTOCOL_FEATURES == 0 {
            return Err(Error::NotNegotiated("VHOST_USER_F_PROTOCOL_FEATURES"));
        }
        let features = self.get_u64(request::GET_PROTOCOL_FEATURES)?;
        self.backend_protocol_features = Some(features);
        Ok(features)
    }

    /// `SET_PROTOCOL_FEATURES`. The bits must all be ones the backend offered.
    ///
    /// The reply to this one is never acked even when it turns `REPLY_ACK` on, since the backend
    /// only starts acking after it has seen the message.
    pub fn set_protocol_features(&mut self, features: u64) -> Result<()> {
        let offered = match self.backend_protocol_features {
            Some(offered) => offered,
            None => self.get_protocol_features()?,
        };
        if features & !offered != 0 {
            return Err(Error::InvalidArgument("protocol feature the backend did not offer"));
        }
        let payload = Payload::default().u64(features).0;
        self.conn.send(Header::new(request::SET_PROTOCOL_FEATURES, 8), &payload, &[])?;
        self.acked_protocol_features = Some(features);
        Ok(())
    }

    /// Negotiate protocol features: ask the backend for its set, keep the bits `supported`
    /// also has, and send the result back. Returns what was agreed on, or zero if the backend
    /// does not do protocol features at all.
    ///
    /// `INBAND_NOTIFICATIONS` is dropped unless `BACKEND_REQ` and `REPLY_ACK` both survive,
    /// since the protocol says a backend closes the connection on that combination.
    pub fn negotiate_protocol_features(&mut self, supported: u64) -> Result<u64> {
        if self.offered_features()? & VHOST_USER_F_PROTOCOL_FEATURES == 0 {
            return Ok(0);
        }
        let mut agreed = self.get_protocol_features()? & supported;
        let needed = protocol::BACKEND_REQ | protocol::REPLY_ACK;
        if agreed & needed != needed {
            agreed &= !protocol::INBAND_NOTIFICATIONS;
        }
        if agreed & protocol::INFLIGHT_SHMFD == 0 {
            agreed &= !protocol::GET_VRING_BASE_INFLIGHT;
        }
        self.set_protocol_features(agreed)?;
        Ok(agreed)
    }

    /// `SET_OWNER`, the first request of a session.
    pub fn set_owner(&mut self) -> Result<()> {
        self.send(request::SET_OWNER, &[], &[])
    }

    /// `RESET_OWNER`. Deprecated by the protocol; [`Frontend::reset`] picks the right request.
    pub fn reset_owner(&mut self) -> Result<()> {
        self.send(request::RESET_OWNER, &[], &[])
    }

    /// `RESET_DEVICE`, which needs the `RESET_DEVICE` protocol feature.
    pub fn reset_device(&mut self) -> Result<()> {
        self.require(protocol::RESET_DEVICE, "VHOST_USER_PROTOCOL_F_RESET_DEVICE")?;
        self.send(request::RESET_DEVICE, &[], &[])
    }

    /// Reset the device: `RESET_DEVICE` when the backend has it, `RESET_OWNER` otherwise.
    pub fn reset(&mut self) -> Result<()> {
        if self.has_protocol_feature(protocol::RESET_DEVICE) {
            self.reset_device()
        } else {
            self.reset_owner()
        }
    }

    /// `SET_MEM_TABLE` with up to [`MAX_REGIONS`] regions, each with its file descriptor.
    pub fn set_mem_table(&mut self, regions: &[MemoryRegion<'_>]) -> Result<()> {
        if regions.len() > MAX_REGIONS {
            return Err(Error::TooManyRegions { count: regions.len(), max: MAX_REGIONS });
        }
        check_regions(regions)?;
        let fds = region_fds(regions)?;
        let count = u32::try_from(regions.len()).unwrap_or(u32::MAX);
        let mut payload = Payload::default().u32(count).u32(0);
        for region in regions {
            payload = payload.region(region);
        }
        debug_assert_eq!(payload.0.len(), 8 + regions.len() * REGION_SIZE);
        self.send(request::SET_MEM_TABLE, &payload.0, &fds)
    }

    /// `GET_MAX_MEM_SLOTS`, which needs `CONFIGURE_MEM_SLOTS`.
    pub fn get_max_mem_slots(&mut self) -> Result<u64> {
        self.require(protocol::CONFIGURE_MEM_SLOTS, "VHOST_USER_PROTOCOL_F_CONFIGURE_MEM_SLOTS")?;
        self.get_u64(request::GET_MAX_MEM_SLOTS)
    }

    /// `ADD_MEM_REG`: map one more region. Needs `CONFIGURE_MEM_SLOTS`.
    pub fn add_mem_reg(&mut self, region: &MemoryRegion<'_>) -> Result<()> {
        self.require(protocol::CONFIGURE_MEM_SLOTS, "VHOST_USER_PROTOCOL_F_CONFIGURE_MEM_SLOTS")?;
        check_regions(std::slice::from_ref(region))?;
        let fds = region_fds(std::slice::from_ref(region))?;
        let payload = Payload::default().u64(0).region(region).0;
        self.send(request::ADD_MEM_REG, &payload, &fds)
    }

    /// `REM_MEM_REG`: unmap a region, identified by its guest address, VMM address and size.
    /// No file descriptor goes with it.
    pub fn rem_mem_reg(&mut self, region: &MemoryRegion<'_>) -> Result<()> {
        self.require(protocol::CONFIGURE_MEM_SLOTS, "VHOST_USER_PROTOCOL_F_CONFIGURE_MEM_SLOTS")?;
        let payload = Payload::default().u64(0).region(region).0;
        self.send(request::REM_MEM_REG, &payload, &[])
    }

    /// `SET_LOG_BASE`. With a shared memory `region` this needs `LOG_SHMFD`, sends the log's
    /// size and offset with its file descriptor, and waits for the backend to say it has mapped
    /// it. Without one it sends `base` alone.
    pub fn set_log_base(&mut self, base: u64, region: Option<LogRegion<'_>>) -> Result<()> {
        match region {
            Some(log) => {
                self.require(protocol::LOG_SHMFD, "VHOST_USER_PROTOCOL_F_LOG_SHMFD")?;
                let payload = Payload::default().u64(log.size).u64(log.offset).0;
                self.call(request::SET_LOG_BASE, &payload, &[log.fd], None)?;
                Ok(())
            }
            None => self.send(request::SET_LOG_BASE, &Payload::default().u64(base).0, &[]),
        }
    }

    /// `SET_LOG_FD`: the eventfd the backend signals after writing to the log.
    pub fn set_log_fd(&mut self, fd: BorrowedFd<'_>) -> Result<()> {
        self.send(request::SET_LOG_FD, &[], &[fd])
    }

    fn vring_state(&mut self, req: u32, index: u32, num: u32) -> Result<()> {
        self.send(req, &Payload::default().u32(index).u32(num).0, &[])
    }

    /// `SET_VRING_NUM`: the size of ring `index`.
    pub fn set_vring_num(&mut self, index: u32, num: u32) -> Result<()> {
        self.vring_state(request::SET_VRING_NUM, index, num)
    }

    /// `SET_VRING_ADDR`.
    pub fn set_vring_addr(&mut self, addr: &VringAddr) -> Result<()> {
        let payload = Payload::default()
            .u32(addr.index)
            .u32(addr.flags)
            .u64(addr.desc_user_addr)
            .u64(addr.used_user_addr)
            .u64(addr.avail_user_addr)
            .u64(addr.log_guest_addr)
            .0;
        debug_assert_eq!(payload.len(), message::VRING_ADDR_SIZE);
        self.send(request::SET_VRING_ADDR, &payload, &[])
    }

    /// `SET_VRING_BASE`. For a split ring `base` is the next available index; for a packed
    /// ring it packs both indices and wrap counters as the protocol describes.
    pub fn set_vring_base(&mut self, index: u32, base: u32) -> Result<()> {
        self.vring_state(request::SET_VRING_BASE, index, base)
    }

    /// `GET_VRING_BASE`: stop ring `index` and return where it stopped.
    pub fn get_vring_base(&mut self, index: u32) -> Result<u32> {
        let payload = Payload::default().u32(index).u32(0).0;
        let reply = self.call(request::GET_VRING_BASE, &payload, &[], Some(8))?;
        let got = u32_at(&reply, 0);
        if got != index {
            return Err(Error::InvalidArgument("GET_VRING_BASE reply names another ring"));
        }
        Ok(u32_at(&reply, 4))
    }

    fn vring_fd(&mut self, req: u32, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        let index = u64::from(index);
        if index > VRING_IDX_MASK {
            return Err(Error::InvalidArgument("ring index above 255"));
        }
        match fd {
            Some(fd) => self.send(req, &Payload::default().u64(index).0, &[fd]),
            None => self.send(req, &Payload::default().u64(index | VRING_NOFD_MASK).0, &[]),
        }
    }

    /// `SET_VRING_KICK`. `None` asks the backend to poll the ring instead.
    pub fn set_vring_kick(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_fd(request::SET_VRING_KICK, index, fd)
    }

    /// `SET_VRING_CALL`. `None` means the VMM will poll for used buffers.
    pub fn set_vring_call(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_fd(request::SET_VRING_CALL, index, fd)
    }

    /// `SET_VRING_ERR`.
    pub fn set_vring_err(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        self.vring_fd(request::SET_VRING_ERR, index, fd)
    }

    /// `SET_VRING_ENABLE`, which is only legal once protocol features are negotiated.
    pub fn set_vring_enable(&mut self, index: u32, enable: bool) -> Result<()> {
        if self.acked_protocol_features.is_none() {
            return Err(Error::NotNegotiated("VHOST_USER_F_PROTOCOL_FEATURES"));
        }
        self.vring_state(request::SET_VRING_ENABLE, index, u32::from(enable))
    }

    /// `GET_QUEUE_NUM`: the most queues the backend supports. Needs `MQ`.
    pub fn get_queue_num(&mut self) -> Result<u64> {
        self.require(protocol::MQ, "VHOST_USER_PROTOCOL_F_MQ")?;
        self.get_u64(request::GET_QUEUE_NUM)
    }

    /// `GET_CONFIG`: read `size` bytes of device config space from `offset`. Needs `CONFIG`.
    pub fn get_config(&mut self, offset: u32, size: u32, flags: u32) -> Result<Vec<u8>> {
        self.require(protocol::CONFIG, "VHOST_USER_PROTOCOL_F_CONFIG")?;
        let len = size as usize;
        if len == 0 || len > MAX_CONFIG_SIZE {
            return Err(Error::InvalidArgument("config space access size"));
        }
        let payload = Payload::default().u32(offset).u32(size).u32(flags).bytes(&vec![0; len]).0;
        let reply = self.call(request::GET_CONFIG, &payload, &[], None)?;
        if reply.is_empty() {
            return Err(Error::BackendFailed { request: request::GET_CONFIG, status: 1 });
        }
        if reply.len() != CONFIG_HEADER_SIZE + len {
            return Err(Error::BadPayloadSize {
                request: request::GET_CONFIG,
                expected: CONFIG_HEADER_SIZE + len,
                got: reply.len(),
            });
        }
        Ok(reply[CONFIG_HEADER_SIZE..].to_vec())
    }

    /// `SET_CONFIG`: write `data` to device config space at `offset`. `flags` is
    /// [`message::CONFIG_TYPE_FRONTEND`] for a driver write or
    /// [`message::CONFIG_TYPE_MIGRATION`] when loading state. Needs `CONFIG`.
    pub fn set_config(&mut self, offset: u32, flags: u32, data: &[u8]) -> Result<()> {
        self.require(protocol::CONFIG, "VHOST_USER_PROTOCOL_F_CONFIG")?;
        if data.is_empty() || data.len() > MAX_CONFIG_SIZE {
            return Err(Error::InvalidArgument("config space access size"));
        }
        let size = u32::try_from(data.len()).unwrap_or(u32::MAX);
        let payload = Payload::default().u32(offset).u32(size).u32(flags).bytes(data).0;
        self.send(request::SET_CONFIG, &payload, &[])
    }

    /// `SET_STATUS`: the virtio device status byte. Needs `STATUS`.
    pub fn set_status(&mut self, status: u8) -> Result<()> {
        self.require(protocol::STATUS, "VHOST_USER_PROTOCOL_F_STATUS")?;
        self.send(request::SET_STATUS, &Payload::default().u64(u64::from(status)).0, &[])
    }

    /// `GET_STATUS`. Needs `STATUS`.
    pub fn get_status(&mut self) -> Result<u8> {
        self.require(protocol::STATUS, "VHOST_USER_PROTOCOL_F_STATUS")?;
        let status = self.get_u64(request::GET_STATUS)?;
        u8::try_from(status).map_err(|_| Error::InvalidArgument("device status above 0xff"))
    }

    /// `SET_BACKEND_REQ_FD` with a fresh socket pair. The backend gets one end, and the other
    /// comes back as a [`BackendChannel`] that passes each request to `handler`. Needs
    /// `BACKEND_REQ`.
    pub fn set_backend_req_fd(&mut self, handler: BackendHandler) -> Result<BackendChannel> {
        self.require(protocol::BACKEND_REQ, "VHOST_USER_PROTOCOL_F_BACKEND_REQ")?;
        let (ours, theirs) = UnixStream::pair()?;
        self.send(request::SET_BACKEND_REQ_FD, &[], &[theirs.as_fd()])?;
        Ok(BackendChannel::new(ours, handler))
    }
}

fn wire_size(payload: &[u8]) -> Result<u32> {
    u32::try_from(payload.len()).map_err(|_| Error::PayloadTooLarge(payload.len()))
}

fn region_fds<'a>(regions: &[MemoryRegion<'a>]) -> Result<Vec<BorrowedFd<'a>>> {
    regions
        .iter()
        .map(|r| r.fd.ok_or(Error::InvalidArgument("vhost-user memory region without a file")))
        .collect()
}

impl VhostBackend for Frontend {
    fn set_owner(&mut self) -> Result<()> {
        Frontend::set_owner(self)
    }

    fn reset_owner(&mut self) -> Result<()> {
        self.reset()
    }

    fn get_features(&mut self) -> Result<u64> {
        Frontend::get_features(self)
    }

    fn set_features(&mut self, features: u64) -> Result<()> {
        Frontend::set_features(self, features)
    }

    fn set_mem_table(&mut self, regions: &[MemoryRegion<'_>]) -> Result<()> {
        Frontend::set_mem_table(self, regions)
    }

    fn set_log_base(&mut self, base: u64, region: Option<LogRegion<'_>>) -> Result<()> {
        Frontend::set_log_base(self, base, region)
    }

    fn set_log_fd(&mut self, fd: BorrowedFd<'_>) -> Result<()> {
        Frontend::set_log_fd(self, fd)
    }

    fn set_vring_num(&mut self, index: u32, num: u32) -> Result<()> {
        Frontend::set_vring_num(self, index, num)
    }

    fn set_vring_addr(&mut self, addr: &VringAddr) -> Result<()> {
        Frontend::set_vring_addr(self, addr)
    }

    fn set_vring_base(&mut self, index: u32, base: u32) -> Result<()> {
        Frontend::set_vring_base(self, index, base)
    }

    fn get_vring_base(&mut self, index: u32) -> Result<u32> {
        Frontend::get_vring_base(self, index)
    }

    fn set_vring_kick(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        Frontend::set_vring_kick(self, index, fd)
    }

    fn set_vring_call(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        Frontend::set_vring_call(self, index, fd)
    }

    fn set_vring_err(&mut self, index: u32, fd: Option<BorrowedFd<'_>>) -> Result<()> {
        Frontend::set_vring_err(self, index, fd)
    }
}
