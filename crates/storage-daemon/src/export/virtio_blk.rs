// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio-blk request handler shared by the vhost-user-blk and VDUSE exports,
//! block/export/virtio-blk-handler.c, plus the `virtio_blk_config` layout from
//! `linux/virtio_blk.h` that both exports hand to the driver.
//!
//! Differences from QEMU:
//!
//! - QEMU works on iovecs that libvhost-user or libvduse already mapped. Here a request is a
//!   [`DescriptorChain`] and the data is copied through [`GuestMemory`], at most
//!   [`CHUNK`] bytes at a time, so a large request becomes several block layer calls.
//! - A buffer outside guest memory is only found while copying. The request is then given up
//!   with [`ReqError::Memory`] and the caller treats the device as broken, which is what
//!   libvhost-user does when it cannot map a buffer at pop time.

use std::sync::Arc;

use ruvm_base::error_report;
use ruvm_block::BlockBackend;
use ruvm_virtio_queue::{DescriptorChain, GuestMemory, QueueError};

/// `VIRTIO_BLK_SECTOR_BITS`
pub const VIRTIO_BLK_SECTOR_BITS: u32 = 9;
/// `VIRTIO_BLK_SECTOR_SIZE`
pub const VIRTIO_BLK_SECTOR_SIZE: u64 = 1 << VIRTIO_BLK_SECTOR_BITS;
/// `VIRTIO_BLK_MAX_DISCARD_SECTORS`
pub const VIRTIO_BLK_MAX_DISCARD_SECTORS: u32 = 32768;
/// `VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS`
pub const VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS: u32 = 32768;
/// `BDRV_REQUEST_MAX_SECTORS`: `INT_MAX >> BDRV_SECTOR_BITS` on a 64-bit host.
const BDRV_REQUEST_MAX_SECTORS: u64 = (i32::MAX as u64) >> VIRTIO_BLK_SECTOR_BITS;

/// `VIRTIO_ID_BLOCK`
pub const VIRTIO_ID_BLOCK: u32 = 2;

/// `VIRTIO_BLK_T_IN`
pub const VIRTIO_BLK_T_IN: u32 = 0;
/// `VIRTIO_BLK_T_OUT`
pub const VIRTIO_BLK_T_OUT: u32 = 1;
/// `VIRTIO_BLK_T_FLUSH`
pub const VIRTIO_BLK_T_FLUSH: u32 = 4;
/// `VIRTIO_BLK_T_GET_ID`
pub const VIRTIO_BLK_T_GET_ID: u32 = 8;
/// `VIRTIO_BLK_T_DISCARD`
pub const VIRTIO_BLK_T_DISCARD: u32 = 11;
/// `VIRTIO_BLK_T_WRITE_ZEROES`
pub const VIRTIO_BLK_T_WRITE_ZEROES: u32 = 13;
/// `VIRTIO_BLK_T_BARRIER`
pub const VIRTIO_BLK_T_BARRIER: u32 = 0x8000_0000;

/// `VIRTIO_BLK_S_OK`
pub const VIRTIO_BLK_S_OK: u8 = 0;
/// `VIRTIO_BLK_S_IOERR`
pub const VIRTIO_BLK_S_IOERR: u8 = 1;
/// `VIRTIO_BLK_S_UNSUPP`
pub const VIRTIO_BLK_S_UNSUPP: u8 = 2;

/// `VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP`
pub const VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP: u32 = 1;
/// `VIRTIO_BLK_ID_BYTES`
pub const VIRTIO_BLK_ID_BYTES: usize = 20;

/// `VIRTIO_BLK_F_SIZE_MAX`
pub const VIRTIO_BLK_F_SIZE_MAX: u32 = 1;
/// `VIRTIO_BLK_F_SEG_MAX`
pub const VIRTIO_BLK_F_SEG_MAX: u32 = 2;
/// `VIRTIO_BLK_F_RO`
pub const VIRTIO_BLK_F_RO: u32 = 5;
/// `VIRTIO_BLK_F_BLK_SIZE`
pub const VIRTIO_BLK_F_BLK_SIZE: u32 = 6;
/// `VIRTIO_BLK_F_FLUSH`
pub const VIRTIO_BLK_F_FLUSH: u32 = 9;
/// `VIRTIO_BLK_F_TOPOLOGY`
pub const VIRTIO_BLK_F_TOPOLOGY: u32 = 10;
/// `VIRTIO_BLK_F_CONFIG_WCE`
pub const VIRTIO_BLK_F_CONFIG_WCE: u32 = 11;
/// `VIRTIO_BLK_F_MQ`
pub const VIRTIO_BLK_F_MQ: u32 = 12;
/// `VIRTIO_BLK_F_DISCARD`
pub const VIRTIO_BLK_F_DISCARD: u32 = 13;
/// `VIRTIO_BLK_F_WRITE_ZEROES`
pub const VIRTIO_BLK_F_WRITE_ZEROES: u32 = 14;
/// `VIRTIO_F_NOTIFY_ON_EMPTY`
pub const VIRTIO_F_NOTIFY_ON_EMPTY: u32 = 24;
/// `VIRTIO_F_VERSION_1`
pub const VIRTIO_F_VERSION_1: u32 = 32;
/// `VIRTIO_F_IOMMU_PLATFORM`
pub const VIRTIO_F_IOMMU_PLATFORM: u32 = 33;

/// `sizeof(struct virtio_blk_outhdr)`: type, ioprio and sector.
const OUTHDR_SIZE: u64 = 16;
/// `sizeof(struct virtio_blk_discard_write_zeroes)`: sector, num_sectors and flags.
const DWZ_SIZE: usize = 16;

/// The most bytes moved between guest memory and the block layer in one call.
pub const CHUNK: usize = 1 << 20;

/// `sizeof(struct virtio_blk_config)`.
pub const VIRTIO_BLK_CONFIG_SIZE: usize = 96;
/// `offsetof(struct virtio_blk_config, capacity)`.
pub const VIRTIO_BLK_CONFIG_CAPACITY: usize = 0;
/// `offsetof(struct virtio_blk_config, wce)`.
pub const VIRTIO_BLK_CONFIG_WCE: usize = 32;

/// `struct virtio_blk_config`, the fields the exports fill in. The rest (geometry, the secure
/// erase and zoned fields) stay zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtioBlkConfig {
    /// Size in 512 byte sectors.
    pub capacity: u64,
    /// Largest segment, zero for no limit.
    pub size_max: u32,
    /// Most segments in one request.
    pub seg_max: u32,
    /// Logical block size.
    pub blk_size: u32,
    /// Minimum I/O size in logical blocks.
    pub min_io_size: u16,
    /// Optimal I/O size in logical blocks.
    pub opt_io_size: u32,
    /// Write cache enabled.
    pub wce: u8,
    /// Number of request queues.
    pub num_queues: u16,
    /// `max_discard_sectors`
    pub max_discard_sectors: u32,
    /// `max_discard_seg`
    pub max_discard_seg: u32,
    /// `discard_sector_alignment`
    pub discard_sector_alignment: u32,
    /// `max_write_zeroes_sectors`
    pub max_write_zeroes_sectors: u32,
    /// `max_write_zeroes_seg`
    pub max_write_zeroes_seg: u32,
}

impl VirtioBlkConfig {
    /// The config space bytes, little endian as a VIRTIO 1.0 device has them.
    pub fn to_bytes(&self) -> [u8; VIRTIO_BLK_CONFIG_SIZE] {
        let mut b = [0u8; VIRTIO_BLK_CONFIG_SIZE];
        b[0..8].copy_from_slice(&self.capacity.to_le_bytes());
        b[8..12].copy_from_slice(&self.size_max.to_le_bytes());
        b[12..16].copy_from_slice(&self.seg_max.to_le_bytes());
        b[20..24].copy_from_slice(&self.blk_size.to_le_bytes());
        b[26..28].copy_from_slice(&self.min_io_size.to_le_bytes());
        b[28..32].copy_from_slice(&self.opt_io_size.to_le_bytes());
        b[VIRTIO_BLK_CONFIG_WCE] = self.wce;
        b[34..36].copy_from_slice(&self.num_queues.to_le_bytes());
        b[36..40].copy_from_slice(&self.max_discard_sectors.to_le_bytes());
        b[40..44].copy_from_slice(&self.max_discard_seg.to_le_bytes());
        b[44..48].copy_from_slice(&self.discard_sector_alignment.to_le_bytes());
        b[48..52].copy_from_slice(&self.max_write_zeroes_sectors.to_le_bytes());
        b[52..56].copy_from_slice(&self.max_write_zeroes_seg.to_le_bytes());
        b
    }
}

/// Why a request produced no used element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReqError {
    /// The request had no headers. The message has been reported and, as in QEMU, the chain is
    /// dropped without being returned to the driver.
    Malformed,
    /// A buffer is not in guest memory.
    Memory(QueueError),
}

impl From<QueueError> for ReqError {
    fn from(e: QueueError) -> Self {
        ReqError::Memory(e)
    }
}

impl From<ruvm_virtio_queue::MemoryError> for ReqError {
    fn from(e: ruvm_virtio_queue::MemoryError) -> Self {
        ReqError::Memory(QueueError::Memory(e))
    }
}

/// `MIN_BLOCK_SIZE` of util/block-helpers.h.
const MIN_BLOCK_SIZE: u64 = 512;
/// `MAX_BLOCK_SIZE` of util/block-helpers.h.
const MAX_BLOCK_SIZE: u64 = 2 << 20;

/// `check_block_size()` of util/block-helpers.c. Zero means unset and passes.
pub fn check_block_size(name: &str, value: u64) -> ruvm_base::Result<()> {
    if value == 0 {
        return Ok(());
    }
    if !(MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&value) || !value.is_power_of_two() {
        return Err(ruvm_base::Error::generic(format!(
            "parameter {name} must be a power of 2 between {MIN_BLOCK_SIZE} and {MAX_BLOCK_SIZE}"
        )));
    }
    Ok(())
}

/// `VirtioBlkHandler`.
#[derive(Debug)]
pub struct VirtioBlkHandler {
    /// The export's backend.
    pub blk: Arc<BlockBackend>,
    /// What `VIRTIO_BLK_T_GET_ID` returns.
    pub serial: String,
    /// Requests must start on a multiple of this.
    pub logical_block_size: u32,
    /// Whether writes, discards and write zeroes are allowed.
    pub writable: bool,
}

impl VirtioBlkHandler {
    /// `virtio_blk_sect_range_ok()`.
    fn sect_range_ok(&self, sector: u64, size: u64) -> bool {
        if size % VIRTIO_BLK_SECTOR_SIZE != 0 {
            return false;
        }
        let nb_sectors = size >> VIRTIO_BLK_SECTOR_BITS;
        if nb_sectors > BDRV_REQUEST_MAX_SECTORS {
            return false;
        }
        if (sector << VIRTIO_BLK_SECTOR_BITS) % u64::from(self.logical_block_size.max(1)) != 0 {
            return false;
        }
        let total_sectors = self.blk.getlength().unwrap_or(0) >> VIRTIO_BLK_SECTOR_BITS;
        !(sector > total_sectors || nb_sectors > total_sectors - sector)
    }

    /// `virtio_blk_process_req()`: carry out the request in `chain` and write its status byte.
    /// Returns the length to put in the used element, which is every byte the driver made
    /// writable, as QEMU reports it.
    pub fn process_req<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
        chain: &DescriptorChain,
    ) -> Result<u32, ReqError> {
        let (out, inp) = (chain.readable(), chain.writable());
        if out.is_empty() || inp.is_empty() {
            error_report("virtio-blk request missing headers");
            return Err(ReqError::Malformed);
        }
        let mut reader = chain.reader(mem);
        let mut hdr = [0u8; OUTHDR_SIZE as usize];
        if chain.readable_len() < OUTHDR_SIZE {
            error_report("virtio-blk request outhdr too short");
            return Err(ReqError::Malformed);
        }
        reader.read_exact(&mut hdr)?;
        let last = inp[inp.len() - 1];
        if last.is_empty() {
            error_report("virtio-blk request inhdr too short");
            return Err(ReqError::Malformed);
        }
        let in_len = chain.writable_len();
        let status_addr = last.addr() + u64::from(last.len()) - 1;
        let data_in = in_len - 1;

        let ty = u32::from_le_bytes(hdr[0..4].try_into().unwrap_or_default());
        let sector = u64::from_le_bytes(hdr[8..16].try_into().unwrap_or_default());
        let status = match ty & !VIRTIO_BLK_T_BARRIER {
            VIRTIO_BLK_T_IN | VIRTIO_BLK_T_OUT => {
                let is_write = ty & VIRTIO_BLK_T_OUT != 0;
                if is_write && !self.writable {
                    VIRTIO_BLK_S_IOERR
                } else {
                    let size = if is_write { reader.remaining() } else { data_in };
                    if !self.sect_range_ok(sector, size) {
                        VIRTIO_BLK_S_IOERR
                    } else if is_write {
                        self.write(&mut reader, sector << VIRTIO_BLK_SECTOR_BITS, size)?
                    } else {
                        self.read(mem, chain, sector << VIRTIO_BLK_SECTOR_BITS, size)?
                    }
                }
            }
            VIRTIO_BLK_T_FLUSH => {
                if self.blk.flush().is_ok() {
                    VIRTIO_BLK_S_OK
                } else {
                    VIRTIO_BLK_S_IOERR
                }
            }
            VIRTIO_BLK_T_GET_ID => {
                let mut id = self.serial.as_bytes().to_vec();
                id.push(0);
                let size = id.len().min((data_in as usize).min(VIRTIO_BLK_ID_BYTES));
                chain.writer(mem).write_all(&id[..size])?;
                VIRTIO_BLK_S_OK
            }
            t @ (VIRTIO_BLK_T_DISCARD | VIRTIO_BLK_T_WRITE_ZEROES) => {
                if !self.writable {
                    VIRTIO_BLK_S_IOERR
                } else {
                    self.discard_write_zeroes(&mut reader, t)?
                }
            }
            _ => VIRTIO_BLK_S_UNSUPP,
        };
        mem.write(status_addr, &[status])?;
        Ok(u32::try_from(in_len).unwrap_or(u32::MAX))
    }

    fn read<M: GuestMemory + ?Sized>(
        &self,
        mem: &M,
        chain: &DescriptorChain,
        offset: u64,
        size: u64,
    ) -> Result<u8, ReqError> {
        let mut writer = chain.writer(mem);
        let mut buf = vec![0u8; (size as usize).min(CHUNK)];
        let mut done = 0;
        while done < size {
            let n = (size - done).min(CHUNK as u64) as usize;
            if self.blk.pread(offset + done, &mut buf[..n]).is_err() {
                return Ok(VIRTIO_BLK_S_IOERR);
            }
            writer.write_all(&buf[..n])?;
            done += n as u64;
        }
        Ok(VIRTIO_BLK_S_OK)
    }

    fn write<M: GuestMemory + ?Sized>(
        &self,
        reader: &mut ruvm_virtio_queue::Reader<'_, M>,
        offset: u64,
        size: u64,
    ) -> Result<u8, ReqError> {
        let mut buf = vec![0u8; (size as usize).min(CHUNK)];
        let mut done = 0;
        while done < size {
            let n = (size - done).min(CHUNK as u64) as usize;
            reader.read_exact(&mut buf[..n])?;
            if self.blk.pwrite(offset + done, &buf[..n]).is_err() {
                return Ok(VIRTIO_BLK_S_IOERR);
            }
            done += n as u64;
        }
        Ok(VIRTIO_BLK_S_OK)
    }

    /// `virtio_blk_discard_write_zeroes()`, with `reader` just past the request header.
    fn discard_write_zeroes<M: GuestMemory + ?Sized>(
        &self,
        reader: &mut ruvm_virtio_queue::Reader<'_, M>,
        ty: u32,
    ) -> Result<u8, ReqError> {
        // Only one desc is currently supported.
        if reader.remaining() > DWZ_SIZE as u64 {
            return Ok(VIRTIO_BLK_S_UNSUPP);
        }
        let mut desc = [0u8; DWZ_SIZE];
        let size = reader.read(&mut desc)?;
        if size != DWZ_SIZE {
            error_report(&format!("Invalid size {size}, expected {DWZ_SIZE}"));
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        let sector = u64::from_le_bytes(desc[0..8].try_into().unwrap_or_default());
        let num_sectors = u32::from_le_bytes(desc[8..12].try_into().unwrap_or_default());
        let flags = u32::from_le_bytes(desc[12..16].try_into().unwrap_or_default());
        let max_sectors = if ty == VIRTIO_BLK_T_WRITE_ZEROES {
            VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS
        } else {
            VIRTIO_BLK_MAX_DISCARD_SECTORS
        };
        if num_sectors > max_sectors {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        let bytes = u64::from(num_sectors) << VIRTIO_BLK_SECTOR_BITS;
        if !self.sect_range_ok(sector, bytes) {
            return Ok(VIRTIO_BLK_S_IOERR);
        }
        // The device MUST set the status byte to VIRTIO_BLK_S_UNSUPP for discard and write
        // zeroes commands if any unknown flag is set.
        if flags & !VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP != 0 {
            return Ok(VIRTIO_BLK_S_UNSUPP);
        }
        let offset = sector << VIRTIO_BLK_SECTOR_BITS;
        let unmap = flags & VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP != 0;
        let ok = if ty == VIRTIO_BLK_T_WRITE_ZEROES {
            self.blk.pwrite_zeroes(offset, bytes, unmap).is_ok()
        } else {
            // The device MUST set the status byte to VIRTIO_BLK_S_UNSUPP for discard commands
            // if the unmap flag is set.
            if unmap {
                return Ok(VIRTIO_BLK_S_UNSUPP);
            }
            self.blk.pdiscard(offset, bytes).is_ok()
        };
        Ok(if ok { VIRTIO_BLK_S_OK } else { VIRTIO_BLK_S_IOERR })
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use ruvm_block::{BLK_PERM_ALL, BLK_PERM_CONSISTENT_READ, BLK_PERM_WRITE, BlockGraph};
    use ruvm_qapi::types::BlockdevOptions;
    use ruvm_qapi::visit::{QObjectInputVisitor, Visit};
    use ruvm_virtio_queue::{
        RingAddresses, SplitQueue, VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE, VecMemory,
    };

    use super::*;

    const QSIZE: u16 = 16;
    const ADDRS: RingAddresses =
        RingAddresses { desc_table: 0x1000, driver_area: 0x2000, device_area: 0x3000 };
    const DATA: u64 = 0x10000;

    struct Image {
        path: PathBuf,
        _graph: BlockGraph,
        blk: Arc<BlockBackend>,
    }

    impl Drop for Image {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn image(name: &str, size: usize) -> Image {
        let path =
            std::env::temp_dir().join(format!("ruvm-vblk-{}-{name}.raw", std::process::id()));
        let content: Vec<u8> = (0..size).map(|i| (i / 512) as u8).collect();
        std::fs::write(&path, content).unwrap();
        let json = format!(
            r#"{{"driver": "raw", "node-name": "n",
                "file": {{"driver": "file", "filename": "{}"}}}}"#,
            path.display()
        );
        let mut v = QObjectInputVisitor::new(ruvm_qapi::json::from_str(&json).unwrap());
        let mut opts = BlockdevOptions::default();
        BlockdevOptions::visit(&mut v, None, &mut opts).unwrap();
        let graph = BlockGraph::new();
        graph.blockdev_add(opts).unwrap();
        let blk =
            BlockBackend::new(&graph, "n", BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE, BLK_PERM_ALL)
                .unwrap();
        Image { path, _graph: graph, blk }
    }

    fn handler(img: &Image, writable: bool) -> VirtioBlkHandler {
        VirtioBlkHandler {
            blk: img.blk.clone(),
            serial: "serial-number".into(),
            logical_block_size: 512,
            writable,
        }
    }

    /// Puts a chain of `(addr, len, write)` buffers in the ring and pops it.
    fn chain(mem: &VecMemory, bufs: &[(u64, u32, bool)]) -> DescriptorChain {
        for (i, &(addr, len, write)) in bufs.iter().enumerate() {
            let d = ADDRS.desc_table + 16 * i as u64;
            mem.write_u64(d, addr).unwrap();
            mem.write_u32(d + 8, len).unwrap();
            let mut flags = if write { VIRTQ_DESC_F_WRITE } else { 0 };
            if i + 1 < bufs.len() {
                flags |= VIRTQ_DESC_F_NEXT;
            }
            mem.write_u16(d + 12, flags).unwrap();
            mem.write_u16(d + 14, i as u16 + 1).unwrap();
        }
        mem.write_u16(ADDRS.driver_area + 4, 0).unwrap();
        mem.write_u16(ADDRS.driver_area + 2, 1).unwrap();
        let mut q = SplitQueue::new(QSIZE, ADDRS).unwrap();
        q.pop(mem).unwrap().unwrap()
    }

    fn outhdr(mem: &VecMemory, ty: u32, sector: u64) {
        mem.write_u32(DATA, ty).unwrap();
        mem.write_u32(DATA + 4, 0).unwrap();
        mem.write_u64(DATA + 8, sector).unwrap();
    }

    fn status(mem: &VecMemory, addr: u64) -> u8 {
        let mut b = [0u8];
        mem.read(addr, &mut b).unwrap();
        b[0]
    }

    #[test]
    fn config_layout() {
        let c = VirtioBlkConfig {
            capacity: 0x1122,
            blk_size: 4096,
            wce: 1,
            num_queues: 3,
            max_write_zeroes_seg: 1,
            ..Default::default()
        };
        let b = c.to_bytes();
        assert_eq!(&b[0..8], &0x1122u64.to_le_bytes());
        assert_eq!(&b[20..24], &4096u32.to_le_bytes());
        assert_eq!(b[32], 1);
        assert_eq!(&b[34..36], &3u16.to_le_bytes());
        assert_eq!(&b[52..56], &1u32.to_le_bytes());
    }

    #[test]
    fn read_write_and_status() {
        let img = image("rw", 8192);
        let h = handler(&img, true);
        let mem = VecMemory::new(0x40000);

        // Read sectors 2 and 3 into two buffers, status in a third.
        outhdr(&mem, VIRTIO_BLK_T_IN, 2);
        let c = chain(&mem, &[(DATA, 16, false), (0x20000, 512, true), (0x21000, 513, true)]);
        assert_eq!(h.process_req(&mem, &c), Ok(1025));
        let mut got = vec![0u8; 512];
        mem.read(0x20000, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 2));
        mem.read(0x21000, &mut got).unwrap();
        assert!(got.iter().all(|&b| b == 3));
        assert_eq!(status(&mem, 0x21000 + 512), VIRTIO_BLK_S_OK);

        // Write sector 5.
        outhdr(&mem, VIRTIO_BLK_T_OUT, 5);
        mem.write(0x22000, &[0xab; 512]).unwrap();
        let c = chain(&mem, &[(DATA, 16, false), (0x22000, 512, false), (0x23000, 1, true)]);
        assert_eq!(h.process_req(&mem, &c), Ok(1));
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_OK);
        let mut back = [0u8; 512];
        img.blk.pread(5 * 512, &mut back).unwrap();
        assert!(back.iter().all(|&b| b == 0xab));

        // Past the end, and not a whole sector.
        outhdr(&mem, VIRTIO_BLK_T_IN, 16);
        let c = chain(&mem, &[(DATA, 16, false), (0x20000, 513, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x20000 + 512), VIRTIO_BLK_S_IOERR);
        outhdr(&mem, VIRTIO_BLK_T_IN, 0);
        let c = chain(&mem, &[(DATA, 16, false), (0x20000, 100, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x20000 + 99), VIRTIO_BLK_S_IOERR);
    }

    #[test]
    fn read_only_refuses_writes() {
        let img = image("ro", 4096);
        let h = handler(&img, false);
        let mem = VecMemory::new(0x40000);
        for ty in [VIRTIO_BLK_T_OUT, VIRTIO_BLK_T_DISCARD, VIRTIO_BLK_T_WRITE_ZEROES] {
            outhdr(&mem, ty, 0);
            let c = chain(&mem, &[(DATA, 16, false), (0x22000, 16, false), (0x23000, 1, true)]);
            h.process_req(&mem, &c).unwrap();
            assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_IOERR);
        }
    }

    #[test]
    fn get_id_flush_and_unknown() {
        let img = image("id", 4096);
        let h = handler(&img, true);
        let mem = VecMemory::new(0x40000);
        outhdr(&mem, VIRTIO_BLK_T_GET_ID, 0);
        let c = chain(&mem, &[(DATA, 16, false), (0x20000, 20, true), (0x21000, 1, true)]);
        assert_eq!(h.process_req(&mem, &c), Ok(21));
        let mut id = [0u8; 14];
        mem.read(0x20000, &mut id).unwrap();
        assert_eq!(&id, b"serial-number\0");
        assert_eq!(status(&mem, 0x21000), VIRTIO_BLK_S_OK);

        outhdr(&mem, VIRTIO_BLK_T_FLUSH | VIRTIO_BLK_T_BARRIER, 0);
        let c = chain(&mem, &[(DATA, 16, false), (0x21000, 1, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x21000), VIRTIO_BLK_S_OK);

        outhdr(&mem, 99, 0);
        let c = chain(&mem, &[(DATA, 16, false), (0x21000, 1, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x21000), VIRTIO_BLK_S_UNSUPP);
    }

    #[test]
    fn write_zeroes_and_discard() {
        let img = image("wz", 8192);
        let h = handler(&img, true);
        let mem = VecMemory::new(0x40000);
        let dwz = |sector: u64, n: u32, flags: u32| {
            mem.write_u64(0x22000, sector).unwrap();
            mem.write_u32(0x22008, n).unwrap();
            mem.write_u32(0x2200c, flags).unwrap();
        };

        outhdr(&mem, VIRTIO_BLK_T_WRITE_ZEROES, 0);
        dwz(1, 2, 0);
        let c = chain(&mem, &[(DATA, 16, false), (0x22000, 16, false), (0x23000, 1, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_OK);
        let mut back = [0xffu8; 1024];
        img.blk.pread(512, &mut back).unwrap();
        assert!(back.iter().all(|&b| b == 0));

        // Unknown flag, too many sectors, the unmap flag on discard, two descriptors.
        dwz(0, 1, 2);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_UNSUPP);
        dwz(0, VIRTIO_BLK_MAX_WRITE_ZEROES_SECTORS + 1, 0);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_IOERR);
        outhdr(&mem, VIRTIO_BLK_T_DISCARD, 0);
        dwz(0, 1, VIRTIO_BLK_WRITE_ZEROES_FLAG_UNMAP);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_UNSUPP);
        dwz(0, 1, 0);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_OK);
        let c = chain(&mem, &[(DATA, 16, false), (0x22000, 32, false), (0x23000, 1, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_UNSUPP);
        let c = chain(&mem, &[(DATA, 16, false), (0x22000, 8, false), (0x23000, 1, true)]);
        h.process_req(&mem, &c).unwrap();
        assert_eq!(status(&mem, 0x23000), VIRTIO_BLK_S_IOERR);
    }

    #[test]
    fn malformed_requests_are_dropped() {
        let img = image("bad", 4096);
        let h = handler(&img, true);
        let mem = VecMemory::new(0x40000);
        outhdr(&mem, VIRTIO_BLK_T_IN, 0);
        let c = chain(&mem, &[(DATA, 16, false)]);
        assert_eq!(h.process_req(&mem, &c), Err(ReqError::Malformed));
        let c = chain(&mem, &[(DATA, 8, false), (0x21000, 1, true)]);
        assert_eq!(h.process_req(&mem, &c), Err(ReqError::Malformed));
        let c = chain(&mem, &[(DATA, 16, false), (0x21000, 0, true)]);
        assert_eq!(h.process_req(&mem, &c), Err(ReqError::Malformed));
        // A data buffer outside guest memory.
        let c = chain(&mem, &[(DATA, 16, false), (0x100000, 512, true), (0x21000, 1, true)]);
        assert!(matches!(h.process_req(&mem, &c), Err(ReqError::Memory(_))));
    }
}
