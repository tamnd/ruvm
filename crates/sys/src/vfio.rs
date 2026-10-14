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
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{FileExt, OpenOptionsExt};

/// `VFIO_API_VERSION`.
pub const API_VERSION: i32 = 0;
/// `VFIO_TYPE1_IOMMU`.
pub const TYPE1_IOMMU: u32 = 1;
/// `VFIO_TYPE1v2_IOMMU`.
pub const TYPE1V2_IOMMU: u32 = 3;

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

/// `VFIO_IRQ_INFO_NORESIZE`: the MSI-X vector count cannot change while enabled.
pub const IRQ_INFO_NORESIZE: u32 = 1 << 3;

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
const VFIO_DEVICE_RESET: u64 = io(11);
const VFIO_IOMMU_MAP_DMA: u64 = io(13);
const VFIO_IOMMU_UNMAP_DMA: u64 = io(14);

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

/// The argument structures of the ioctls this module issues.
trait Arg {}
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
}

/// An IOMMU group, `/dev/vfio/<id>`.
#[derive(Debug)]
pub struct VfioGroup {
    file: File,
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

    /// `VFIO_DEVICE_GET_IRQ_INFO` for interrupt `index`.
    pub fn irq_info(&self, index: u32) -> io::Result<IrqInfo> {
        let mut info = IrqInfo { argsz: argsz::<IrqInfo>(), index, ..IrqInfo::default() };
        ioctl_ptr(&self.file, VFIO_DEVICE_GET_IRQ_INFO, &mut info)?;
        Ok(info)
    }

    /// `VFIO_DEVICE_RESET`.
    pub fn reset(&self) -> io::Result<()> {
        ioctl_val(&self.file, VFIO_DEVICE_RESET, 0).map(drop)
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
        assert_eq!(VFIO_DEVICE_RESET, 0x3b6f);
        assert_eq!(VFIO_IOMMU_MAP_DMA, 0x3b71);
        assert_eq!(VFIO_IOMMU_UNMAP_DMA, 0x3b72);
        assert_eq!(size_of::<DmaMap>(), 32);
        assert_eq!(size_of::<DmaUnmap>(), 24);
        assert_eq!(size_of::<DeviceInfo>(), 24);
        assert_eq!(size_of::<RegionInfo>(), 32);
        assert_eq!(size_of::<IrqInfo>(), 16);
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
