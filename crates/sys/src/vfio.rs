// SPDX-License-Identifier: MIT OR Apache-2.0

//! The Linux VFIO uapi that device assignment needs: the legacy container (`/dev/vfio/vfio`),
//! the IOMMU groups under `/dev/vfio`, and the device a group hands out.
//!
//! A container holds an IOMMU context. Groups are added to it with
//! [`VfioGroup::set_container`], the container picks its IOMMU model with
//! [`VfioContainer::set_iommu`], and only then does a group give out device descriptors. A device
//! is a set of regions (BARs, ROM and config space for vfio-pci) read and written at fixed offsets
//! of its descriptor, and a set of interrupts.
//!
//! This is the subset of `linux/vfio.h` that QEMU's hw/vfio/container-legacy.c, device.c and
//! region.c use.

use std::ffi::CString;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::fs::{FileExt, OpenOptionsExt};

/// `VFIO_API_VERSION`.
pub const API_VERSION: i32 = 0;
/// `VFIO_TYPE1_IOMMU`.
pub const TYPE1_IOMMU: u32 = 1;
/// `VFIO_TYPE1v2_IOMMU`.
pub const TYPE1V2_IOMMU: u32 = 3;
/// `VFIO_UNMAP_ALL`: the extension for [`VfioContainer::unmap_dma_all`].
pub const UNMAP_ALL: u32 = 9;

/// `VFIO_GROUP_FLAGS_VIABLE`: every device in the group is bound to a VFIO driver.
pub const GROUP_FLAGS_VIABLE: u32 = 1;

/// `VFIO_DEVICE_FLAGS_RESET`: `VFIO_DEVICE_RESET` works.
pub const DEVICE_FLAGS_RESET: u32 = 1;
/// `VFIO_DEVICE_FLAGS_PCI`: a vfio-pci device.
pub const DEVICE_FLAGS_PCI: u32 = 1 << 1;

/// `VFIO_REGION_INFO_FLAG_READ`.
pub const REGION_INFO_FLAG_READ: u32 = 1;
/// `VFIO_REGION_INFO_FLAG_WRITE`.
pub const REGION_INFO_FLAG_WRITE: u32 = 1 << 1;
/// `VFIO_REGION_INFO_FLAG_MMAP`.
pub const REGION_INFO_FLAG_MMAP: u32 = 1 << 2;
/// `VFIO_REGION_INFO_FLAG_CAPS`: the region info has a capability chain.
pub const REGION_INFO_FLAG_CAPS: u32 = 1 << 3;
/// `VFIO_REGION_INFO_CAP_SPARSE_MMAP`: only the listed areas of the region can be mapped.
pub const REGION_INFO_CAP_SPARSE_MMAP: u16 = 1;
/// `VFIO_REGION_INFO_CAP_MSIX_MAPPABLE`: the MSI-X table of the BAR can be mapped too.
pub const REGION_INFO_CAP_MSIX_MAPPABLE: u16 = 3;

/// `VFIO_DEVICE_FEATURE_GET`.
pub const DEVICE_FEATURE_GET: u32 = 1 << 16;
/// `VFIO_DEVICE_FEATURE_DMA_BUF`.
pub const DEVICE_FEATURE_DMA_BUF: u32 = 11;

/// `VFIO_PCI_BAR0_REGION_INDEX`.
pub const PCI_BAR0_REGION_INDEX: u32 = 0;
/// `VFIO_PCI_ROM_REGION_INDEX`.
pub const PCI_ROM_REGION_INDEX: u32 = 6;
/// `VFIO_PCI_CONFIG_REGION_INDEX`.
pub const PCI_CONFIG_REGION_INDEX: u32 = 7;
/// `VFIO_PCI_INTX_IRQ_INDEX`.
pub const PCI_INTX_IRQ_INDEX: u32 = 0;
/// `VFIO_PCI_MSI_IRQ_INDEX`.
pub const PCI_MSI_IRQ_INDEX: u32 = 1;
/// `VFIO_PCI_MSIX_IRQ_INDEX`.
pub const PCI_MSIX_IRQ_INDEX: u32 = 2;
/// `VFIO_PCI_ERR_IRQ_INDEX`.
pub const PCI_ERR_IRQ_INDEX: u32 = 3;

/// `VFIO_DMA_MAP_FLAG_READ`: the device may read the mapping.
pub const DMA_MAP_FLAG_READ: u32 = 1;
/// `VFIO_DMA_MAP_FLAG_WRITE`: the device may write the mapping.
pub const DMA_MAP_FLAG_WRITE: u32 = 1 << 1;
/// `VFIO_DMA_UNMAP_FLAG_ALL`: the unmap covers every mapping.
pub const DMA_UNMAP_FLAG_ALL: u32 = 1 << 1;

/// `VFIO_IRQ_SET_DATA_NONE`: the call carries no data.
pub const IRQ_SET_DATA_NONE: u32 = 1;
/// `VFIO_IRQ_SET_DATA_EVENTFD`: the call carries one eventfd per interrupt.
pub const IRQ_SET_DATA_EVENTFD: u32 = 1 << 2;
/// `VFIO_IRQ_SET_ACTION_MASK`.
pub const IRQ_SET_ACTION_MASK: u32 = 1 << 3;
/// `VFIO_IRQ_SET_ACTION_UNMASK`.
pub const IRQ_SET_ACTION_UNMASK: u32 = 1 << 4;
/// `VFIO_IRQ_SET_ACTION_TRIGGER`.
pub const IRQ_SET_ACTION_TRIGGER: u32 = 1 << 5;

/// `VFIO_IRQ_INFO_NORESIZE`: the MSI-X vector count cannot change while enabled.
pub const IRQ_INFO_NORESIZE: u32 = 1 << 3;

/// A device a PCI hot reset resets too, `struct vfio_pci_dependent_device` as a device opened
/// through its group sees it.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct DependentDevice {
    /// The IOMMU group the device is in.
    pub group_id: u32,
    pub segment: u16,
    pub bus: u8,
    pub devfn: u8,
}

impl DependentDevice {
    /// The device's name in sysfs, `0000:01:00.0`, which is what `vfio_pci_host_match()`
    /// compares.
    pub fn host_name(&self) -> String {
        format!(
            "{:04x}:{:02x}:{:02x}.{:x}",
            self.segment,
            self.bus,
            self.devfn >> 3,
            self.devfn & 7
        )
    }
}

/// The devices of a `vfio_pci_hot_reset_info` that `count` says were filled in: two words each,
/// the group and then the segment, bus and devfn, after the three words of the header.
fn dependent_devices(words: &[u32], count: u32) -> Vec<DependentDevice> {
    words[3..]
        .chunks_exact(2)
        .take(count as usize)
        .map(|d| {
            let [s0, s1, bus, devfn] = d[1].to_ne_bytes();
            DependentDevice { group_id: d[0], segment: u16::from_ne_bytes([s0, s1]), bus, devfn }
        })
        .collect()
}

/// `_IO(';', 100 + nr)`. Every VFIO ioctl is `_IO`, with the size in the `argsz` field instead.
const fn io(nr: u64) -> u64 {
    ((b';' as u64) << 8) | (100 + nr)
}

const VFIO_GET_API_VERSION: u64 = io(0);
const VFIO_CHECK_EXTENSION: u64 = io(1);
const VFIO_SET_IOMMU: u64 = io(2);
const VFIO_GROUP_GET_STATUS: u64 = io(3);
const VFIO_GROUP_SET_CONTAINER: u64 = io(4);
const VFIO_GROUP_GET_DEVICE_FD: u64 = io(6);
const VFIO_DEVICE_GET_INFO: u64 = io(7);
const VFIO_DEVICE_GET_REGION_INFO: u64 = io(8);
const VFIO_DEVICE_GET_IRQ_INFO: u64 = io(9);
const VFIO_DEVICE_SET_IRQS: u64 = io(10);
const VFIO_DEVICE_RESET: u64 = io(11);
const VFIO_DEVICE_GET_PCI_HOT_RESET_INFO: u64 = io(12);
const VFIO_DEVICE_PCI_HOT_RESET: u64 = io(13);
const VFIO_IOMMU_MAP_DMA: u64 = io(13);
const VFIO_IOMMU_UNMAP_DMA: u64 = io(14);
const VFIO_DEVICE_FEATURE: u64 = io(17);

#[repr(C)]
#[derive(Default)]
struct GroupStatus {
    argsz: u32,
    flags: u32,
}

/// `struct vfio_device_info`, without the capability chain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceInfo {
    argsz: u32,
    /// `VFIO_DEVICE_FLAGS_*`.
    pub flags: u32,
    pub num_regions: u32,
    pub num_irqs: u32,
    cap_offset: u32,
    pad: u32,
}

/// `struct vfio_region_info`, without the capability chain.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionInfo {
    argsz: u32,
    /// `VFIO_REGION_INFO_FLAG_*`.
    pub flags: u32,
    pub index: u32,
    cap_offset: u32,
    /// The size in bytes, 0 for a region the device does not have.
    pub size: u64,
    /// Where the region is in the device descriptor.
    pub offset: u64,
}

/// The capability chain of a region's info, the bytes `VFIO_DEVICE_GET_REGION_INFO` returned
/// when its `argsz` was large enough for them.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegionCaps {
    buf: Vec<u8>,
    first: u32,
}

impl RegionCaps {
    /// A chain from the whole `vfio_region_info` buffer and its `cap_offset`.
    pub fn new(buf: Vec<u8>, first: u32) -> Self {
        RegionCaps { buf, first }
    }

    /// `vfio_get_cap()`: the capability `id`, from its header on. The chain ends at a `next`
    /// of 0, which is where the info starts.
    pub fn find(&self, id: u16) -> Option<&[u8]> {
        let mut off = self.first as usize;
        // A chain that loops back on itself would never reach 0.
        for _ in 0..self.buf.len() / 8 {
            if off == 0 {
                return None;
            }
            let hdr = self.buf.get(off..off.checked_add(8)?)?;
            if u16::from_ne_bytes([hdr[0], hdr[1]]) == id {
                return Some(&self.buf[off..]);
            }
            off = u32::from_ne_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
        }
        None
    }

    /// Whether the chain has capability `id`.
    pub fn has(&self, id: u16) -> bool {
        self.find(id).is_some()
    }

    /// The areas of `struct vfio_region_info_cap_sparse_mmap` as offset and size pairs, in the
    /// order the kernel gave them.
    pub fn sparse_mmap(&self) -> Option<Vec<(u64, u64)>> {
        let cap = self.find(REGION_INFO_CAP_SPARSE_MMAP)?;
        let nr = u32::from_ne_bytes(cap.get(8..12)?.try_into().ok()?) as usize;
        let u64_at =
            |o: usize| cap.get(o..o + 8).map(|b| u64::from_ne_bytes(b.try_into().unwrap()));
        (0..nr).map(|i| Some((u64_at(16 + 16 * i)?, u64_at(24 + 16 * i)?))).collect()
    }
}

/// `struct vfio_irq_info`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IrqInfo {
    argsz: u32,
    /// `VFIO_IRQ_INFO_*`.
    pub flags: u32,
    pub index: u32,
    /// How many interrupts of this kind the device has.
    pub count: u32,
}

/// `struct vfio_iommu_type1_dma_map`.
#[repr(C)]
#[derive(Default)]
struct DmaMap {
    argsz: u32,
    flags: u32,
    vaddr: u64,
    iova: u64,
    size: u64,
}

/// `struct vfio_iommu_type1_dma_unmap`, without the dirty bitmap.
#[repr(C)]
#[derive(Default)]
struct DmaUnmap {
    argsz: u32,
    flags: u32,
    iova: u64,
    size: u64,
}

/// `struct vfio_irq_set` without its data.
#[repr(C)]
#[derive(Default)]
struct IrqSet {
    argsz: u32,
    flags: u32,
    index: u32,
    start: u32,
    count: u32,
}

/// The argument structures of the ioctls this module issues. A `c_char` is the first byte of a
/// string, a `u32` the first word of a `vfio_irq_set` followed by its eventfds, and a `u64` the
/// first word of a buffer as long as the `argsz` it starts with.
trait Arg {}
impl Arg for u64 {}
impl Arg for IrqSet {}
impl Arg for u32 {}
impl Arg for GroupStatus {}
impl Arg for DeviceInfo {}
impl Arg for RegionInfo {}
impl Arg for IrqInfo {}
impl Arg for DmaMap {}
impl Arg for DmaUnmap {}
impl Arg for libc::c_int {}
impl Arg for libc::c_char {}

/// An ioctl that takes a plain integer or nothing, and returns an integer.
fn ioctl_val(file: &File, req: u64, arg: libc::c_ulong) -> io::Result<i32> {
    // SAFETY: `req` is one of the VFIO ioctls that take no argument or an integer by value,
    // which the kernel does not dereference.
    let ret = unsafe { libc::ioctl(file.as_raw_fd(), req as _, arg) };
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(ret) }
}

/// An ioctl that takes a pointer to `T`, and returns an integer.
fn ioctl_ptr<T: Arg>(file: &File, req: u64, arg: *mut T) -> io::Result<i32> {
    // SAFETY: `req` is one of the VFIO ioctls whose argument is `T` (the `Arg` types mirror the
    // kernel structures, each starting with an `argsz` the callers fill in, and every caller
    // pairs them right), and `arg` points to a live `T` valid for reads and writes for the
    // whole call.
    let ret = unsafe { libc::ioctl(file.as_raw_fd(), req as _, arg) };
    if ret < 0 { Err(io::Error::last_os_error()) } else { Ok(ret) }
}

/// Two `u32` fields that follow each other in a structure, as the `u64` word they fill.
fn word(first: u32, second: u32) -> u64 {
    let [a, b, c, d] = first.to_ne_bytes();
    let [e, f, g, h] = second.to_ne_bytes();
    u64::from_ne_bytes([a, b, c, d, e, f, g, h])
}

fn argsz<T>() -> u32 {
    size_of::<T>() as u32
}

fn open_rw(path: &str) -> io::Result<File> {
    OpenOptions::new().read(true).write(true).custom_flags(libc::O_CLOEXEC).open(path)
}

/// A legacy VFIO container, `/dev/vfio/vfio`.
#[derive(Debug)]
pub struct VfioContainer {
    file: File,
}

impl VfioContainer {
    /// Opens a new container.
    pub fn open() -> io::Result<Self> {
        Ok(VfioContainer { file: open_rw("/dev/vfio/vfio")? })
    }

    /// `VFIO_GET_API_VERSION`.
    pub fn api_version(&self) -> io::Result<i32> {
        ioctl_val(&self.file, VFIO_GET_API_VERSION, 0)
    }

    /// `VFIO_CHECK_EXTENSION`: whether the container supports the IOMMU model or extension
    /// `ext`.
    pub fn check_extension(&self, ext: u32) -> io::Result<bool> {
        ioctl_val(&self.file, VFIO_CHECK_EXTENSION, ext.into()).map(|r| r > 0)
    }

    /// `VFIO_SET_IOMMU`: picks the IOMMU model. A group must be in the container first.
    pub fn set_iommu(&self, model: u32) -> io::Result<()> {
        ioctl_val(&self.file, VFIO_SET_IOMMU, model.into()).map(drop)
    }

    /// `VFIO_IOMMU_MAP_DMA`: lets the devices in the container reach the `size` bytes at
    /// `vaddr` through bus address `iova`, read only or read and write. The kernel pins the
    /// pages until they are unmapped.
    ///
    /// # Safety
    ///
    /// The devices write the memory behind the program's back for as long as it is mapped, so
    /// nothing may hold a Rust reference to it in that time. Guest RAM, which is only reached
    /// through raw accesses, is the memory this is for.
    pub unsafe fn map_dma(
        &self,
        iova: u64,
        size: u64,
        vaddr: usize,
        readonly: bool,
    ) -> io::Result<()> {
        let mut flags = DMA_MAP_FLAG_READ;
        if !readonly {
            flags |= DMA_MAP_FLAG_WRITE;
        }
        let mut map = DmaMap { argsz: argsz::<DmaMap>(), flags, vaddr: vaddr as u64, iova, size };
        ioctl_ptr(&self.file, VFIO_IOMMU_MAP_DMA, &mut map).map(drop)
    }

    /// `VFIO_IOMMU_UNMAP_DMA`: removes the mappings in `iova..iova + size`. Returns how many
    /// bytes were unmapped.
    pub fn unmap_dma(&self, iova: u64, size: u64) -> io::Result<u64> {
        let mut unmap = DmaUnmap { argsz: argsz::<DmaUnmap>(), flags: 0, iova, size };
        ioctl_ptr(&self.file, VFIO_IOMMU_UNMAP_DMA, &mut unmap)?;
        Ok(unmap.size)
    }

    /// `VFIO_IOMMU_UNMAP_DMA` with `VFIO_DMA_UNMAP_FLAG_ALL`: removes every mapping, where
    /// [`UNMAP_ALL`] is supported. Returns how many bytes were unmapped.
    pub fn unmap_dma_all(&self) -> io::Result<u64> {
        let flags = DMA_UNMAP_FLAG_ALL;
        let mut unmap = DmaUnmap { argsz: argsz::<DmaUnmap>(), flags, iova: 0, size: 0 };
        ioctl_ptr(&self.file, VFIO_IOMMU_UNMAP_DMA, &mut unmap)?;
        Ok(unmap.size)
    }
}

/// An IOMMU group, `/dev/vfio/<id>`.
#[derive(Debug)]
pub struct VfioGroup {
    file: File,
}

impl AsFd for VfioGroup {
    /// The group descriptor, which a hot reset takes as proof that the group is owned.
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl VfioGroup {
    /// Opens group `id`.
    pub fn open(id: u32) -> io::Result<Self> {
        Ok(VfioGroup { file: open_rw(&format!("/dev/vfio/{id}"))? })
    }

    /// `VFIO_GROUP_GET_STATUS`: the `VFIO_GROUP_FLAGS_*`.
    pub fn status(&self) -> io::Result<u32> {
        let mut status = GroupStatus { argsz: argsz::<GroupStatus>(), flags: 0 };
        ioctl_ptr(&self.file, VFIO_GROUP_GET_STATUS, &mut status)?;
        Ok(status.flags)
    }

    /// `VFIO_GROUP_SET_CONTAINER`.
    pub fn set_container(&self, container: &VfioContainer) -> io::Result<()> {
        let mut fd: libc::c_int = container.file.as_raw_fd();
        ioctl_ptr(&self.file, VFIO_GROUP_SET_CONTAINER, &mut fd).map(drop)
    }

    /// `VFIO_GROUP_GET_DEVICE_FD`: the device called `name`, a PCI address such as
    /// `0000:00:04.0` for vfio-pci, possibly followed by ` vf_token=<uuid>`.
    pub fn device(&self, name: &str) -> io::Result<VfioDevice> {
        let name = CString::new(name).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let fd = ioctl_ptr(&self.file, VFIO_GROUP_GET_DEVICE_FD, name.as_ptr().cast_mut())?;
        // SAFETY: the kernel just returned this descriptor and nothing else owns it.
        let file = unsafe { File::from_raw_fd(fd) };
        Ok(VfioDevice { file })
    }
}

/// A device a group handed out.
#[derive(Debug)]
pub struct VfioDevice {
    file: File,
}

impl AsFd for VfioDevice {
    /// The device descriptor, which the regions that allow it are mapped from.
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.file.as_fd()
    }
}

impl VfioDevice {
    /// `VFIO_DEVICE_GET_INFO`.
    pub fn info(&self) -> io::Result<DeviceInfo> {
        let mut info = DeviceInfo { argsz: argsz::<DeviceInfo>(), ..DeviceInfo::default() };
        ioctl_ptr(&self.file, VFIO_DEVICE_GET_INFO, &mut info)?;
        Ok(info)
    }

    /// `VFIO_DEVICE_GET_REGION_INFO` for region `index`.
    pub fn region_info(&self, index: u32) -> io::Result<RegionInfo> {
        let mut info = RegionInfo { argsz: argsz::<RegionInfo>(), index, ..RegionInfo::default() };
        ioctl_ptr(&self.file, VFIO_DEVICE_GET_REGION_INFO, &mut info)?;
        Ok(info)
    }

    /// `vfio_device_get_region_info()`: `VFIO_DEVICE_GET_REGION_INFO` for region `index` with
    /// its capability chain, asking again with the `argsz` the kernel wants when the chain did
    /// not fit.
    pub fn region_info_caps(&self, index: u32) -> io::Result<(RegionInfo, RegionCaps)> {
        let mut argsz = argsz::<RegionInfo>();
        loop {
            let mut buf = vec![0u64; (argsz as usize).div_ceil(8)];
            buf[0] = word(argsz, 0);
            buf[1] = word(index, 0);
            ioctl_ptr(&self.file, VFIO_DEVICE_GET_REGION_INFO, buf.as_mut_ptr())?;
            let bytes: Vec<u8> = buf.iter().flat_map(|w| w.to_ne_bytes()).collect();
            let u32_at = |o: usize| u32::from_ne_bytes(bytes[o..o + 4].try_into().unwrap());
            let u64_at = |o: usize| u64::from_ne_bytes(bytes[o..o + 8].try_into().unwrap());
            let want = u32_at(0);
            if want > argsz {
                argsz = want;
                continue;
            }
            let info = RegionInfo {
                argsz: want,
                flags: u32_at(4),
                index: u32_at(8),
                cap_offset: u32_at(12),
                size: u64_at(16),
                offset: u64_at(24),
            };
            let first = if info.flags & REGION_INFO_FLAG_CAPS != 0 { info.cap_offset } else { 0 };
            return Ok((info, RegionCaps::new(bytes, first)));
        }
    }

    /// `VFIO_DEVICE_FEATURE` with `VFIO_DEVICE_FEATURE_DMA_BUF`: a dma-buf of the `ranges` of
    /// region `index`, each an offset and a length, as `vfio_region_create_dma_buf()` asks for
    /// one.
    pub fn dma_buf(
        &self,
        index: u32,
        open_flags: u32,
        ranges: &[(u64, u64)],
    ) -> io::Result<OwnedFd> {
        let bytes = 8 + 16 + 16 * ranges.len();
        let argsz =
            u32::try_from(bytes).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut buf = Vec::with_capacity(bytes / 8);
        buf.push(word(argsz, DEVICE_FEATURE_GET | DEVICE_FEATURE_DMA_BUF));
        buf.push(word(index, open_flags));
        buf.push(word(0, ranges.len() as u32));
        buf.extend(ranges.iter().flat_map(|&(offset, length)| [offset, length]));
        let fd = ioctl_ptr(&self.file, VFIO_DEVICE_FEATURE, buf.as_mut_ptr())?;
        // SAFETY: the kernel just returned this descriptor and nothing else owns it.
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// `VFIO_DEVICE_GET_IRQ_INFO` for interrupt `index`.
    pub fn irq_info(&self, index: u32) -> io::Result<IrqInfo> {
        let mut info = IrqInfo { argsz: argsz::<IrqInfo>(), index, ..IrqInfo::default() };
        ioctl_ptr(&self.file, VFIO_DEVICE_GET_IRQ_INFO, &mut info)?;
        Ok(info)
    }

    /// `VFIO_DEVICE_SET_IRQS` without data: `action` on interrupts `start..start + count` of
    /// `index`. A trigger with a count of 0 turns the whole index off.
    pub fn set_irqs(&self, index: u32, start: u32, count: u32, action: u32) -> io::Result<()> {
        let mut set = IrqSet {
            argsz: argsz::<IrqSet>(),
            flags: IRQ_SET_DATA_NONE | action,
            index,
            start,
            count,
        };
        ioctl_ptr(&self.file, VFIO_DEVICE_SET_IRQS, &mut set).map(drop)
    }

    /// `VFIO_DEVICE_SET_IRQS` with eventfds: interrupt `start + i` of `index` gets `fds[i]` for
    /// `action`, and a `None` takes the one it had away.
    pub fn set_irqs_eventfds(
        &self,
        index: u32,
        start: u32,
        action: u32,
        fds: &[Option<BorrowedFd<'_>>],
    ) -> io::Result<()> {
        let words = size_of::<IrqSet>() / 4 + fds.len();
        let argsz =
            u32::try_from(4 * words).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut set = Vec::with_capacity(words);
        set.extend([argsz, IRQ_SET_DATA_EVENTFD | action, index, start, fds.len() as u32]);
        set.extend(fds.iter().map(|fd| fd.map_or(-1, |fd| fd.as_raw_fd()) as u32));
        ioctl_ptr(&self.file, VFIO_DEVICE_SET_IRQS, set.as_mut_ptr()).map(drop)
    }

    /// `VFIO_DEVICE_RESET`.
    pub fn reset(&self) -> io::Result<()> {
        ioctl_val(&self.file, VFIO_DEVICE_RESET, 0).map(drop)
    }

    /// `VFIO_DEVICE_GET_PCI_HOT_RESET_INFO` with room for `room` devices: how many a hot reset
    /// of this device would reset, and the devices. The kernel says how many there are even
    /// when they do not fit, which it fails with `ENOSPC`.
    pub fn pci_hot_reset_info(&self, room: u32) -> (u32, io::Result<Vec<DependentDevice>>) {
        let mut buf = vec![0u32; 3 + 2 * room as usize];
        buf[0] = 12 + 8 * room;
        let r = ioctl_ptr(&self.file, VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, buf.as_mut_ptr());
        (buf[2], r.map(|_| dependent_devices(&buf, buf[2].min(room))))
    }

    /// `VFIO_DEVICE_PCI_HOT_RESET`: a bus or slot reset of this device and every device
    /// [`VfioDevice::pci_hot_reset_info`] lists, with the descriptors of all of their groups as
    /// proof that they are owned.
    pub fn pci_hot_reset(&self, group_fds: &[BorrowedFd<'_>]) -> io::Result<()> {
        let argsz = u32::try_from(12 + 4 * group_fds.len())
            .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut buf = Vec::with_capacity(3 + group_fds.len());
        buf.extend([argsz, 0, group_fds.len() as u32]);
        buf.extend(group_fds.iter().map(|fd| fd.as_raw_fd() as u32));
        ioctl_ptr(&self.file, VFIO_DEVICE_PCI_HOT_RESET, buf.as_mut_ptr()).map(drop)
    }

    /// Reads at `offset` of the descriptor, which is a region's offset plus the offset inside
    /// it. Returns how many bytes were read.
    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> io::Result<usize> {
        self.file.read_at(buf, offset)
    }

    /// Writes at `offset` of the descriptor. Returns how many bytes were written.
    pub fn write_at(&self, buf: &[u8], offset: u64) -> io::Result<usize> {
        self.file.write_at(buf, offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_numbers() {
        // From linux/vfio.h on x86_64.
        assert_eq!(VFIO_GET_API_VERSION, 0x3b64);
        assert_eq!(VFIO_GROUP_GET_DEVICE_FD, 0x3b6a);
        assert_eq!(VFIO_DEVICE_SET_IRQS, 0x3b6e);
        assert_eq!(VFIO_DEVICE_RESET, 0x3b6f);
        assert_eq!(VFIO_DEVICE_GET_PCI_HOT_RESET_INFO, 0x3b70);
        assert_eq!(VFIO_DEVICE_PCI_HOT_RESET, 0x3b71);
        assert_eq!(size_of::<IrqSet>(), 20);
        assert_eq!(VFIO_IOMMU_MAP_DMA, 0x3b71);
        assert_eq!(VFIO_IOMMU_UNMAP_DMA, 0x3b72);
        assert_eq!(size_of::<DmaMap>(), 32);
        assert_eq!(size_of::<DmaUnmap>(), 24);
        assert_eq!(size_of::<DeviceInfo>(), 24);
        assert_eq!(size_of::<RegionInfo>(), 32);
        assert_eq!(size_of::<IrqInfo>(), 16);
        assert_eq!(VFIO_DEVICE_FEATURE, 0x3b75);
    }

    /// A region info of 32 bytes followed by a chain at `first`.
    fn chain(caps: &[(usize, u16, u32, &[u8])], len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        for &(off, id, next, body) in caps {
            buf[off..off + 2].copy_from_slice(&id.to_ne_bytes());
            buf[off + 2..off + 4].copy_from_slice(&1u16.to_ne_bytes());
            buf[off + 4..off + 8].copy_from_slice(&next.to_ne_bytes());
            buf[off + 8..off + 8 + body.len()].copy_from_slice(body);
        }
        buf
    }

    #[test]
    fn region_caps() {
        let mut sparse = Vec::new();
        sparse.extend(2u32.to_ne_bytes());
        sparse.extend(0u32.to_ne_bytes());
        for v in [0x3000u64, 0x1000, 0, 0x2000] {
            sparse.extend(v.to_ne_bytes());
        }
        let buf = chain(&[(32, 3, 40, &[]), (40, 1, 0, &sparse)], 96);
        let caps = RegionCaps::new(buf.clone(), 32);
        assert!(caps.has(REGION_INFO_CAP_MSIX_MAPPABLE));
        assert_eq!(caps.sparse_mmap(), Some(vec![(0x3000, 0x1000), (0, 0x2000)]));
        assert!(!caps.has(2));
        // No chain at all, and a chain that points back at itself.
        assert_eq!(RegionCaps::new(buf, 0).sparse_mmap(), None);
        let looped = chain(&[(32, 3, 32, &[])], 40);
        assert!(!RegionCaps::new(looped, 32).has(1));
        // A sparse cap cut short by the buffer.
        let short = chain(&[(32, 1, 0, &sparse[..24])], 64);
        assert_eq!(RegionCaps::new(short, 32).sparse_mmap(), None);
        assert_eq!(word(1, 2).to_ne_bytes()[..4], 1u32.to_ne_bytes());
    }

    #[test]
    fn hot_reset_info() {
        let dev = |segment: u16, bus: u8, devfn: u8| {
            let [a, b] = segment.to_ne_bytes();
            u32::from_ne_bytes([a, b, bus, devfn])
        };
        let words = [28, 0, 2, 7, dev(0, 1, 0), 9, dev(1, 0x3a, 0x1b)];
        let devices = dependent_devices(&words, 2);
        assert_eq!(devices[0], DependentDevice { group_id: 7, segment: 0, bus: 1, devfn: 0 });
        assert_eq!(devices[0].host_name(), "0000:01:00.0");
        assert_eq!(devices[1].host_name(), "0001:3a:03.3");
        // The kernel never fills in more than there was room for.
        assert_eq!(dependent_devices(&words, 1).len(), 1);
        assert!(dependent_devices(&words[..3], 0).is_empty());
    }

    #[test]
    fn container() {
        // Hosts without the vfio module, or without access to it, have nothing to check.
        let Ok(c) = VfioContainer::open() else { return };
        assert_eq!(c.api_version().unwrap(), API_VERSION);
        // Type1 is all there is on x86 and Arm hosts, and v2 came with Linux 3.13.
        assert!(c.check_extension(TYPE1_IOMMU).is_ok());
        // No group yet, so the container has no IOMMU to set.
        assert!(c.set_iommu(TYPE1V2_IOMMU).is_err());
    }

    #[test]
    fn missing_group() {
        let e = VfioGroup::open(u32::MAX).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::NotFound);
    }
}
