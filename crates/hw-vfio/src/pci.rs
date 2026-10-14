// SPDX-License-Identifier: GPL-2.0-or-later

//! `vfio-pci`, hw/vfio/pci.c over the legacy container of hw/vfio/container-legacy.c.
//!
//! The device is found the way `vfio_pci_realize()` finds it: its IOMMU group is opened and put
//! in a container, then the group hands out the device. The config space is copied from the
//! host and the parts QEMU emulates are marked in `emulated`, the BARs and the ROM are trapped
//! and go to the device with `pread()` and `pwrite()`, and the capabilities are rebuilt the way
//! `vfio_pci_add_capabilities()` rebuilds them. A container listens to the address space its
//! devices DMA into and maps the guest's RAM for them, as hw/vfio/listener.c does. INTx comes
//! in through an eventfd the kernel signals, which a thread of the device turns into a raised
//! pin, and the first BAR access after it lowers the pin and unmasks the host interrupt. When the
//! guest turns MSI on, each vector it enabled gets an eventfd of its own and the thread sends
//! the vector's message when the kernel signals it. MSI-X works the same way, except that a
//! vector gets its eventfd when the guest first unmasks it, and one that fires while masked
//! goes into the emulated PBA. DMA through a vIOMMU, mmap of the BARs, hot reset and the device
//! quirks are not here yet.

use std::fmt;
use std::io;
use std::mem;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::{self, JoinHandle};

use rustix::event::{EventfdFlags, PollFd, PollFlags, eventfd, poll};

use ruvm_base::error::{Error, Result, strerror};
use ruvm_base::report::{error_report, report_error, warn_report};
use ruvm_hw_core::IrqLine;
use ruvm_hw_pci::pcie::{
    PCI_EXP_DEVCAP, PCI_EXP_FLAGS_TYPE, PCI_EXP_LNKCAP, PCI_EXP_LNKCAP2, PCI_EXP_LNKCTL,
    PCI_EXP_LNKCTL2, PCI_EXP_LNKSTA, PCI_EXP_LNKSTA2, PCI_EXP_TYPE_ENDPOINT, PCI_EXP_TYPE_LEG_END,
    PCI_EXP_TYPE_RC_END, PCI_EXT_CAP_ID_ARI, pci_ext_cap, pci_ext_cap_id, pci_ext_cap_next,
    pci_ext_cap_ver, pcie_add_capability,
};
use ruvm_hw_pci::regs::{
    MSIX_CAP_LENGTH, PCI_BASE_ADDRESS_0, PCI_BASE_ADDRESS_SPACE_IO, PCI_BASE_ADDRESS_SPACE_MEMORY,
    PCI_CAP_FLAGS, PCI_CAP_ID_AF, PCI_CAP_ID_EXP, PCI_CAP_ID_MSI, PCI_CAP_ID_MSIX, PCI_CAP_ID_PM,
    PCI_CAP_ID_VNDR, PCI_CAP_LIST_NEXT, PCI_CAPABILITY_LIST, PCI_CLASS_REVISION, PCI_COMMAND,
    PCI_COMMAND_INTX_DISABLE, PCI_COMMAND_IO, PCI_COMMAND_MASTER, PCI_COMMAND_MEMORY,
    PCI_CONFIG_SPACE_SIZE, PCI_DEVICE_ID, PCI_HEADER_TYPE, PCI_HEADER_TYPE_MULTI_FUNCTION,
    PCI_INTERRUPT_PIN, PCI_MSI_FLAGS_64BIT, PCI_MSI_FLAGS_MASKBIT, PCI_MSI_FLAGS_QMASK,
    PCI_MSIX_FLAGS, PCI_MSIX_FLAGS_BIRMASK, PCI_MSIX_FLAGS_QSIZE, PCI_MSIX_PBA, PCI_MSIX_TABLE,
    PCI_NUM_PINS, PCI_PM_CTRL, PCI_PM_CTRL_NO_SOFT_RESET, PCI_PM_CTRL_STATE_MASK, PCI_ROM_ADDRESS,
    PCI_ROM_ADDRESS_MASK, PCI_ROM_SLOT, PCI_STATUS, PCI_STATUS_CAP_LIST, PCI_SUBSYSTEM_ID,
    PCI_SUBSYSTEM_VENDOR_ID, PCI_VENDOR_ID, PCIE_CONFIG_SPACE_SIZE, pci_get_long, pci_get_word,
    pci_set_long, pci_set_word,
};
use ruvm_hw_pci::{MsiMessage, MsixLayout, PciBus, PciDevice, PciDeviceInfo, PciDeviceOps};
use ruvm_mem::{
    AccessConstraints, AccessCtx, AccessSize, AddressSpace, FlatRange, ListenerId, MemError,
    MemResult, MemoryListener, MemorySystem, MmioOps, RegionId, RegionType,
};
use ruvm_sys::vfio::{
    API_VERSION, DEVICE_FLAGS_PCI, DEVICE_FLAGS_RESET, DeviceInfo, GROUP_FLAGS_VIABLE,
    IRQ_INFO_NORESIZE, IRQ_SET_ACTION_TRIGGER, IRQ_SET_ACTION_UNMASK, PCI_CONFIG_REGION_INDEX,
    PCI_ERR_IRQ_INDEX, PCI_INTX_IRQ_INDEX, PCI_MSI_IRQ_INDEX, PCI_MSIX_IRQ_INDEX,
    PCI_ROM_REGION_INDEX, RegionInfo, TYPE1_IOMMU, TYPE1V2_IOMMU, VfioContainer, VfioDevice,
    VfioGroup,
};

use crate::{PCI_ANY_ID, VfioPciConf};

const PCI_EXP_FLAGS_VERS: u16 = 0x000f;
const PCI_EXP_DEVCAP_FLR: u32 = 0x1000_0000;
const PCI_AF_CAP: usize = 3;
const PCI_AF_CAP_TP: u8 = 0x01;
const PCI_AF_CAP_FLR: u8 = 0x02;
const PCI_EXT_CAP_ID_ATS: u16 = 0x0f;
const PCI_EXT_CAP_ID_SRIOV: u16 = 0x10;
const PCI_EXT_CAP_ID_REBAR: u16 = 0x15;
const PCI_EXT_CAP_NEXT_MASK: u32 = 0xfff0_0000;
const PCI_REBAR_CAP: usize = 4;
const PCI_REBAR_CAP_SIZES: u32 = 0xffff_fff0;
const PCI_REBAR_CTRL: usize = 8;
const PCI_REBAR_CTRL_BAR_IDX: u32 = 0x0000_0007;
const PCI_REBAR_CTRL_NBAR_MASK: u32 = 0x0000_00e0;
const PCI_REBAR_CTRL_BAR_SIZE: u32 = 0x0000_1f00;
const PCI_VENDOR_ID_CHELSIO: u16 = 0x1425;
const PCI_VENDOR_ID_BAIDU: u16 = 0x1d22;
const PCI_DEVICE_ID_KUNLUN_VF: u16 = 0x3685;
const PCI_EXP_LNKCAP_X1_2_5GT: u32 = 0x11;
/// Bars 0 to 5, `ARRAY_SIZE(vdev->bars)`.
const NUM_BARS: usize = PCI_ROM_SLOT;
/// What `read()` gives back for a short config read, `EFAULT`.
const EFAULT: i32 = 14;
/// How deep a capability chain is followed. A chain that loops would have QEMU recurse until
/// it crashes; this stops early instead.
const MAX_CAPS: u32 = 48;
/// The same for the extended chain, which has room for 960 headers.
const MAX_EXT_CAPS: u32 = 960;

/// `rom_denylist` of hw/vfio/pci-quirks.c: option ROMs that hang the guest.
const ROM_DENYLIST: [(u16, u16); 1] = [(0x14e4, 0x168e)];

/// An open group and the container it is in, `VFIOGroup`. The container comes first so that
/// it is dropped first: QEMU unregisters the listener before the group leaves the container.
#[derive(Debug)]
struct Group {
    id: u32,
    container: Arc<Container>,
    group: VfioGroup,
    devices: Vec<String>,
}

/// A container and the address space its devices DMA into, `VFIOContainer` with its
/// `VFIOAddressSpace`. When the last group leaves, the listener is unregistered, which unmaps
/// the guest's RAM, and then the container is closed.
struct Container {
    memory: Arc<MemorySystem>,
    space: Arc<AddressSpace>,
    listener: ListenerId,
    dma: Arc<DmaListener>,
}

impl fmt::Debug for Container {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Container")
            .field("space", &self.space.name())
            .field("listener", &self.listener)
            .finish_non_exhaustive()
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        // vfio_listener_unregister(). The listener is registered while the container exists.
        let _ = self.memory.unregister_listener(self.listener);
    }
}

/// `vfio_memory_listener`: maps each page aligned RAM range of the address space into the
/// container with the same bus address as the guest physical one.
struct DmaListener {
    fd: VfioContainer,
    page_size: u64,
    /// `bcontainer->error`: the first failure while the listener is registered.
    error: Mutex<Option<Error>>,
    /// `bcontainer->initialized`: registration is over, so a failure is fatal.
    initialized: AtomicBool,
}

/// `vfio_get_section_iova_range()`: the part of `addr..addr + size` made of whole host pages,
/// as start and length, if there is one.
fn section_iova_range(addr: u64, size: u128, page_size: u64) -> Option<(u64, u64)> {
    let mask = !u128::from(page_size - 1);
    let iova = (u128::from(addr) + u128::from(page_size - 1)) & mask;
    let llend = (u128::from(addr) + size) & mask;
    if iova >= llend {
        return None;
    }
    Some((u64::try_from(iova).ok()?, u64::try_from(llend - iova).ok()?))
}

impl DmaListener {
    /// `vfio_listener_valid_section()` and `vfio_get_section_iova_range()`.
    fn section(&self, fr: &FlatRange) -> Option<(u64, u64)> {
        // vfio_listener_skipped_section(): ROM devices are not RAM to QEMU either, and an
        // address with bit 63 set is a 64-bit BAR being sized.
        if fr.region_type() != RegionType::Ram || fr.addr() & (1 << 63) != 0 {
            return None;
        }
        let off = self.page_size - 1;
        if fr.addr() & off != fr.offset_in_region() & off {
            error_report(&format!(
                "vfio_listener_valid_section received unaligned region {} iova={:#x} \
                 offset_within_region={:#x} qemu_real_host_page_size={:#x}",
                fr.name(),
                fr.addr(),
                fr.offset_in_region(),
                self.page_size
            ));
            return None;
        }
        section_iova_range(fr.addr(), fr.size(), self.page_size)
    }

    /// `vfio_legacy_dma_map()`: an `EBUSY` is answered with an unmap and a second try.
    fn map(&self, iova: u64, size: u64, vaddr: usize, readonly: bool) -> io::Result<()> {
        let mut retried = false;
        loop {
            // SAFETY: `vaddr` is guest RAM held by the range being added, and region_del
            // unmaps it before the range, and so the block, can go away. Guest RAM is only
            // reached through raw accesses that already expect it to change underneath them.
            match unsafe { self.fd.map_dma(iova, size, vaddr, readonly) } {
                Err(e) if !retried && e.kind() == io::ErrorKind::ResourceBusy => {
                    retried = true;
                    self.fd.unmap_dma(iova, size)?;
                }
                r => return r,
            }
        }
    }

    /// The `fail:` path of `vfio_container_region_add()`.
    fn fail(&self, name: &str, err: Error) {
        if self.initialized.load(Ordering::Acquire) {
            report_error(&err);
            ruvm_base::fatal!("vfio: DMA mapping failed, unable to continue");
        }
        let mut first = self.error.lock().unwrap_or_else(PoisonError::into_inner);
        if first.is_none() {
            *first = Some(err.prepend(format_args!("Region {name}: ")));
        }
    }
}

impl MemoryListener for DmaListener {
    fn name(&self) -> &str {
        "vfio"
    }

    fn region_add(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some((iova, size)) = self.section(fr) else {
            return;
        };
        let Some(block) = fr.ram_block() else {
            return;
        };
        // memory_region_get_ram_ptr() + offset_within_region + (iova - offset_within_address_space).
        let vaddr = block.host_addr() + (fr.offset_in_region() + (iova - fr.addr())) as usize;
        if let Err(e) = self.map(iova, size, vaddr, fr.readonly()) {
            let ret = -e.raw_os_error().unwrap_or(0);
            let msg = format!(
                "vfio_container_dma_map({self:p}, {iova:#x}, {size:#x}, {vaddr:#x}) = {ret} ({})",
                strerror(&e)
            );
            self.fail(fr.name(), Error::generic(msg));
        }
    }

    fn region_del(&self, _space: &AddressSpace, fr: &FlatRange) {
        let Some((iova, size)) = self.section(fr) else {
            return;
        };
        if let Err(e) = self.fd.unmap_dma(iova, size) {
            let ret = -e.raw_os_error().unwrap_or(0);
            error_report(&format!(
                "vfio_container_dma_unmap({self:p}, {iova:#x}, {size:#x}) = {ret} ({})",
                strerror(&e)
            ));
        }
    }
}

/// `vfio_group_list`.
static GROUPS: Mutex<Vec<Group>> = Mutex::new(Vec::new());

fn groups() -> MutexGuard<'static, Vec<Group>> {
    GROUPS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A device's place in its group. Dropping it is `vfio_device_put()` plus `vfio_group_put()`:
/// the group goes once it has no devices left.
#[derive(Debug)]
struct Attachment {
    group: u32,
    name: String,
}

impl Drop for Attachment {
    fn drop(&mut self) {
        let mut groups = groups();
        if let Some(i) = groups.iter().position(|g| g.id == self.group) {
            groups[i].devices.retain(|d| *d != self.name);
            if groups[i].devices.is_empty() {
                groups.remove(i);
            }
        }
    }
}

/// `vfio_device_get_groupid()`.
fn group_id(sysfsdev: &str) -> Result<u32> {
    let link = std::fs::read_link(format!("{sysfsdev}/iommu_group"))
        .map_err(|e| Error::from_io("no iommu_group found", e))?;
    let path = link.to_string_lossy();
    let base = basename(&path);
    // sscanf("%d").
    let t = base.trim_start();
    let end = t.find(|c: char| !c.is_ascii_digit()).unwrap_or(t.len());
    t[..end].parse().map_err(|_| {
        Error::from_io(format_args!("failed to read {path}"), io::Error::from_raw_os_error(0))
    })
}

/// `vfio_container_connect()`: puts `group` in a container for `space` it shares with the
/// groups already open, or in a new one that maps the RAM of `space`.
fn container_connect(
    groups: &[Group],
    group: &VfioGroup,
    memory: &Arc<MemorySystem>,
    space: &Arc<AddressSpace>,
) -> Result<Arc<Container>> {
    let mut tried: Vec<&Arc<Container>> = Vec::new();
    for g in groups {
        if !Arc::ptr_eq(&g.container.space, space)
            || tried.iter().any(|c| Arc::ptr_eq(c, &g.container))
        {
            continue;
        }
        tried.push(&g.container);
        if group.set_container(&g.container.dma.fd).is_ok() {
            return Ok(Arc::clone(&g.container));
        }
    }
    let container =
        VfioContainer::open().map_err(|e| Error::from_io("Could not open '/dev/vfio/vfio'", e))?;
    let version = container.api_version().unwrap_or(-1);
    if version != API_VERSION {
        return Err(Error::generic(format!(
            "supported vfio version: {API_VERSION}, reported version: {version}"
        )));
    }
    // vfio_get_iommu_type(): QEMU takes any answer but zero as a yes.
    let model = [TYPE1V2_IOMMU, TYPE1_IOMMU]
        .into_iter()
        .find(|&m| container.check_extension(m).unwrap_or(true))
        .ok_or_else(|| Error::generic("No available IOMMU models"))?;
    group
        .set_container(&container)
        .map_err(|e| Error::from_io("Failed to set group container", e))?;
    container
        .set_iommu(model)
        .map_err(|e| Error::from_io("Failed to set iommu for container", e))?;
    // vfio_listener_register().
    let dma = Arc::new(DmaListener {
        fd: container,
        page_size: rustix::param::page_size() as u64,
        error: Mutex::new(None),
        initialized: AtomicBool::new(false),
    });
    let listener = memory
        .register_listener(Arc::clone(&dma) as Arc<dyn MemoryListener>, space)
        .map_err(mem_error)?;
    let container =
        Container { memory: Arc::clone(memory), space: Arc::clone(space), listener, dma };
    let failed = container.dma.error.lock().unwrap_or_else(PoisonError::into_inner).take();
    if let Some(e) = failed {
        return Err(e.prepend("memory listener initialization failed: "));
    }
    container.dma.initialized.store(true, Ordering::Release);
    Ok(Arc::new(container))
}

/// `vfio_group_get()`: the index of group `id` in `groups`, opened and connected to a container
/// for `space` if it is new.
fn group_get(
    groups: &mut Vec<Group>,
    id: u32,
    memory: &Arc<MemorySystem>,
    space: &Arc<AddressSpace>,
) -> Result<usize> {
    if let Some(i) = groups.iter().position(|g| g.id == id) {
        return Ok(i);
    }
    let group = VfioGroup::open(id)
        .map_err(|e| Error::from_io(format_args!("Could not open '/dev/vfio/{id}'"), e))?;
    let status = group
        .status()
        .map_err(|e| Error::from_io(format_args!("failed to get group {id} status"), e))?;
    if status & GROUP_FLAGS_VIABLE == 0 {
        return Err(Error::generic(format!("group {id} is not viable")).hint(
            "Please ensure all devices within the iommu_group are bound to their vfio bus \
             driver.\n",
        ));
    }
    let container = container_connect(groups, &group, memory, space)
        .map_err(|e| e.prepend(format_args!("failed to setup container for group {id}: ")))?;
    groups.push(Group { id, container, group, devices: Vec::new() });
    Ok(groups.len() - 1)
}

/// `vfio_device_attach()` with the legacy container: the device called `name` in the group
/// of `sysfsdev`, which does DMA into `space`.
fn attach(
    sysfsdev: &str,
    name: &str,
    memory: &Arc<MemorySystem>,
    space: &Arc<AddressSpace>,
) -> Result<(Attachment, VfioDevice, DeviceInfo)> {
    let id = group_id(sysfsdev)?;
    let mut groups = groups();
    let i = group_get(&mut groups, id, memory, space)?;
    let g = &groups[i];
    let got = if g.devices.iter().any(|d| d == name) {
        Err(Error::generic("device is already attached"))
    } else {
        // vfio_device_get().
        g.group
            .device(name)
            .map_err(|e| {
                Error::from_io(format_args!("error getting device from group {id}"), e).hint(
                    format_args!(
                        "Verify all devices in group {id} are bound to vfio-<bus> or pci-stub \
                         and not already in use\n"
                    ),
                )
            })
            .and_then(|device| match device.info() {
                Ok(info) => Ok((device, info)),
                Err(e) => Err(Error::from_io("error getting device info", e)),
            })
    };
    match got {
        Ok((device, info)) => {
            groups[i].devices.push(name.to_string());
            Ok((Attachment { group: id, name: name.to_string() }, device, info))
        }
        Err(e) => {
            if groups[i].devices.is_empty() {
                groups.remove(i);
            }
            Err(e)
        }
    }
}

/// `g_path_get_basename()`.
fn basename(path: &str) -> &str {
    let t = path.trim_end_matches('/');
    if t.is_empty() {
        return if path.is_empty() { "." } else { "/" };
    }
    t.rsplit('/').next().unwrap_or(t)
}

/// `strreaderror()` and `strwriteerror()`: what went wrong with a `pread()` or `pwrite()` of
/// `len` bytes.
fn check_rw(r: io::Result<usize>, len: usize, short: &str) -> Result<(), String> {
    match r {
        Ok(n) if n == len => Ok(()),
        Ok(_) => Err(short.to_string()),
        Err(e) => Err(strerror(&e)),
    }
}

fn mem_error(e: MemError) -> Error {
    Error::generic(e.to_string())
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One eventfd slot per interrupt, filled when the interrupt is set up.
type Slots = Arc<Mutex<Vec<Option<Arc<OwnedFd>>>>>;

/// Eventfds the kernel signals for a device's interrupts and a thread that waits on them, the
/// part of `qemu_set_fd_handler()` vfio relies on. The handler runs with the index of each
/// eventfd that fires, until the notifiers are dropped. Slots can be filled later with
/// [`Notifiers::init`], as MSI-X does when the guest unmasks a vector.
#[derive(Debug)]
struct Notifiers {
    name: String,
    fds: Slots,
    /// Signalled to make the thread look at the slots again, or to stop it.
    wake: Arc<OwnedFd>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Notifiers {
    /// `vfio_notifier_init()` for `n` eventfds called `name`, and the thread that serves them.
    fn new(name: &str, n: usize, handler: impl Fn(usize) + Send + 'static) -> Result<Notifiers> {
        let notifiers = Notifiers::empty(name, n, handler)?;
        for i in 0..n {
            notifiers.init(i)?;
        }
        Ok(notifiers)
    }

    /// `n` empty slots and the thread that serves them once they are filled.
    fn empty(name: &str, n: usize, handler: impl Fn(usize) + Send + 'static) -> Result<Notifiers> {
        let failed = |e: io::Error| Error::from_io(format!("vfio_notifier_init {name} failed"), e);
        let fds: Slots = Arc::new(Mutex::new(vec![None; n]));
        let wake = Arc::new(eventfd(0, EventfdFlags::CLOEXEC).map_err(|e| failed(e.into()))?);
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let (fds, wake, stop) = (Arc::clone(&fds), Arc::clone(&wake), Arc::clone(&stop));
            thread::Builder::new()
                .name(format!("vfio-{name}"))
                .spawn(move || notifier_thread(&fds, &wake, &stop, &handler))
                .map_err(failed)?
        };
        Ok(Notifiers { name: name.to_string(), fds, wake, stop, thread: Some(thread) })
    }

    /// `vfio_notifier_init()` for slot `i`, which the thread then waits on as well.
    fn init(&self, i: usize) -> Result<()> {
        let fd = eventfd(0, EventfdFlags::CLOEXEC | EventfdFlags::NONBLOCK).map_err(|e| {
            Error::from_io(format!("vfio_notifier_init {} failed", self.name), e.into())
        })?;
        lock(&self.fds)[i] = Some(Arc::new(fd));
        self.kick();
        Ok(())
    }

    fn kick(&self) {
        let _ = rustix::io::write(&*self.wake, &1u64.to_ne_bytes());
    }

    fn fd(&self, i: usize) -> Option<Arc<OwnedFd>> {
        lock(&self.fds)[i].clone()
    }
}

impl Drop for Notifiers {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.kick();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// `event_notifier_test_and_clear()`.
fn test_and_clear(fd: &OwnedFd) -> bool {
    let mut b = [0u8; 8];
    matches!(rustix::io::read(fd, &mut b), Ok(8))
}

/// Waits on the filled slots of `fds` until `stop` is set, and calls `handler` for each one the
/// kernel signals. `wake` makes it look at the slots again.
fn notifier_thread(fds: &Slots, wake: &OwnedFd, stop: &AtomicBool, handler: &dyn Fn(usize)) {
    loop {
        let live: Vec<(usize, Arc<OwnedFd>)> =
            lock(fds).iter().enumerate().filter_map(|(i, f)| f.clone().map(|f| (i, f))).collect();
        let mut pfds: Vec<PollFd<'_>> =
            live.iter().map(|(_, f)| PollFd::new(&**f, PollFlags::IN)).collect();
        pfds.push(PollFd::new(wake, PollFlags::IN));
        match poll(&mut pfds, None) {
            Ok(_) => {}
            Err(rustix::io::Errno::INTR) => continue,
            Err(_) => return,
        }
        if !pfds[live.len()].revents().is_empty() {
            test_and_clear(wake);
            if stop.load(Ordering::Acquire) {
                return;
            }
        }
        let ready: Vec<usize> =
            (0..live.len()).filter(|&k| !pfds[k].revents().is_empty()).collect();
        for k in ready {
            if test_and_clear(&live[k].1) {
                handler(live[k].0);
            }
        }
    }
}

/// INTx delivery, `VFIOINTx`. The kernel signals the eventfd when the device raises its pin and
/// keeps the host interrupt masked until it is told to unmask it.
#[derive(Debug)]
struct Intx {
    /// `pending`: the pin is raised and the host interrupt masked.
    pending: Arc<Mutex<bool>>,
    irq: IrqLine,
    _notifier: Notifiers,
}

/// `VFIOPCIDevice::interrupt`: which kind of interrupt the device uses, with the eventfds for
/// it.
#[derive(Debug)]
enum Interrupt {
    None,
    Intx(Intx),
    /// One eventfd per MSI vector the guest enabled.
    Msi {
        _vectors: Notifiers,
    },
    Msix(MsixVectors),
}

/// `VFIOPCIDevice::msi_vectors` while the device is in MSI-X mode.
#[derive(Debug)]
struct MsixVectors {
    /// One slot per table entry, filled the first time the guest unmasks the vector.
    notifiers: Notifiers,
    /// `VFIOMSIVector::use`.
    used: Vec<bool>,
    /// `nr_vectors`: one past the highest vector in use.
    nr_vectors: usize,
    /// `defer_kvm_irq_routing`: set while `msix_set_vector_notifiers()` walks the unmasked
    /// vectors, which are then handed to the kernel in one go.
    batch: bool,
    /// `VFIOMSIXInfo::noresize`: the kernel cannot add vectors while MSI-X is on.
    noresize: bool,
    /// `VFIOMSIXInfo::pending`: vectors that fired while masked, shared with the thread.
    pending: Arc<Mutex<Vec<bool>>>,
    pba: Pba,
}

/// The emulated PBA, `msix_pba_mmio`. vfio-pci only shows it while a vector that fired masked
/// is pending, and lets reads go to the device otherwise.
#[derive(Clone, Debug)]
struct Pba {
    memory: Weak<MemorySystem>,
    region: Option<RegionId>,
}

impl Pba {
    fn show(&self, on: bool) {
        if let (Some(memory), Some(region)) = (self.memory.upgrade(), self.region) {
            let _ = memory.set_enabled(region, on);
        }
    }
}

/// What the BAR, ROM and config callbacks share: the device, where its config space is and the
/// state of its interrupts.
#[derive(Debug)]
struct Shared {
    name: String,
    interrupt: Mutex<Interrupt>,
    device: VfioDevice,
    config_offset: u64,
    _attachment: Attachment,
}

impl Shared {
    /// `vfio_pci_config_space_read()`.
    fn config_read(&self, addr: usize, buf: &mut [u8]) -> Result<(), String> {
        let n = buf.len();
        check_rw(self.device.read_at(buf, self.config_offset + addr as u64), n, "short read")
    }

    /// `vfio_pci_config_space_write()`.
    fn config_write(&self, addr: usize, buf: &[u8]) -> Result<(), String> {
        check_rw(
            self.device.write_at(buf, self.config_offset + addr as u64),
            buf.len(),
            "short write",
        )
    }

    /// A config read of up to 4 bytes as a number.
    fn config_read_val(&self, addr: usize, len: usize) -> Result<u32, String> {
        let mut b = [0u8; 4];
        self.config_read(addr, &mut b[..len])?;
        Ok(u32::from_le_bytes(b))
    }

    /// `VFIO_DEVICE_SET_IRQS` with the trigger eventfds of interrupts `start..` of `index`. An
    /// empty slot is -1 to the kernel, which takes away the eventfd that interrupt had.
    fn set_triggers(&self, index: u32, start: u32, fds: &[Option<Arc<OwnedFd>>]) -> io::Result<()> {
        let fds: Vec<Option<BorrowedFd<'_>>> =
            fds.iter().map(|f| f.as_deref().map(AsFd::as_fd)).collect();
        self.device.set_irqs_eventfds(index, start, IRQ_SET_ACTION_TRIGGER, &fds)
    }

    /// `vfio_enable_msix_no_vec()`: MSI-X on in the device with no vector enabled, by giving
    /// vector 0 an fd of -1.
    fn msix_enable_no_vec(&self) -> io::Result<()> {
        self.set_triggers(PCI_MSIX_IRQ_INDEX, 0, &[None])
    }

    /// `vfio_enable_vectors()` for MSI-X. When the kernel can add vectors later, MSI-X is
    /// turned on first so that only the vectors in use get allocated.
    fn msix_enable_vectors(&self, v: &MsixVectors) -> io::Result<()> {
        if !v.noresize {
            self.msix_enable_no_vec()?;
        }
        let fds: Vec<_> = (0..v.nr_vectors).map(|i| v.notifiers.fd(i)).collect();
        self.set_triggers(PCI_MSIX_IRQ_INDEX, 0, &fds)
    }

    /// `vfio_msix_vector_do_use()`: the guest unmasked vector `nr`. Its eventfd is made the
    /// first time, and the kernel is told about it unless the whole table is handed over in
    /// one go afterwards.
    fn msix_vector_use(&self, dev: &PciDevice, nr: usize) {
        let mut interrupt = lock(&self.interrupt);
        let Interrupt::Msix(v) = &mut *interrupt else {
            return;
        };
        if !v.used[nr] {
            // vfio_pci_vector_init().
            if let Err(e) = v.notifiers.init(nr) {
                report_error(&e);
            }
            v.used[nr] = true;
            dev.msix_vector_use(nr as u32);
        }
        // Without dynamic allocation the kernel only has the vectors up to the highest one in
        // use, so a new highest vector means turning MSI-X off and on again with one more.
        let resizing = v.nr_vectors < nr + 1;
        if resizing {
            v.nr_vectors = nr + 1;
        }
        if !v.batch {
            if v.noresize && resizing {
                // vfio_device_irq_disable().
                let _ = self.device.set_irqs(PCI_MSIX_IRQ_INDEX, 0, 0, IRQ_SET_ACTION_TRIGGER);
                if let Err(e) = self.msix_enable_vectors(v) {
                    error_report(&format!("vfio: failed to enable vectors, {}", strerror(&e)));
                }
            } else if let Err(e) =
                self.set_triggers(PCI_MSIX_IRQ_INDEX, nr as u32, &[v.notifiers.fd(nr)])
            {
                // set_irq_signalling().
                let e = Error::from_io("VFIO_DEVICE_SET_IRQS failure", e)
                    .prepend(format_args!("MSIX-{nr}: "))
                    .prepend("Failed to set up TRIGGER eventfd signaling for interrupt ")
                    .prepend(format_args!("vfio {}: ", self.name));
                report_error(&e);
            }
        }
        // The PBA goes back to the device once nothing is pending.
        let none = {
            let mut p = lock(&v.pending);
            p[nr] = false;
            !p[..v.nr_vectors].contains(&true)
        };
        if none {
            v.pba.show(false);
        }
    }

    /// `vfio_pci_intx_eoi()`: QEMU takes any BAR access as the guest having serviced INTx.
    fn eoi(&self) {
        let interrupt = lock(&self.interrupt);
        let Interrupt::Intx(i) = &*interrupt else {
            return;
        };
        let mut pending = lock(&i.pending);
        if !*pending {
            return;
        }
        *pending = false;
        i.irq.lower();
        // vfio_device_irq_unmask().
        let _ = self.device.set_irqs(PCI_INTX_IRQ_INDEX, 0, 1, IRQ_SET_ACTION_UNMASK);
    }

    /// Takes the interrupt state out if `pick` matches it. The caller drops it with the lock
    /// released, since that waits for the notifier thread.
    fn take_interrupt(&self, pick: fn(&Interrupt) -> bool) -> Option<Interrupt> {
        let mut interrupt = lock(&self.interrupt);
        if !pick(&interrupt) {
            return None;
        }
        Some(mem::replace(&mut *interrupt, Interrupt::None))
    }

    /// `vfio_intx_disable()`.
    fn intx_disable(&self) {
        let Some(Interrupt::Intx(intx)) = self.take_interrupt(|i| matches!(i, Interrupt::Intx(_)))
        else {
            return;
        };
        // vfio_device_irq_disable().
        let _ = self.device.set_irqs(PCI_INTX_IRQ_INDEX, 0, 0, IRQ_SET_ACTION_TRIGGER);
        *lock(&intx.pending) = false;
        intx.irq.lower();
    }

    /// The part of `vfio_intx_enable()` after the pin is known: an eventfd for the kernel to
    /// signal, the thread that waits on it, and the kernel told to use it.
    fn intx_start(&self, irq: IrqLine) -> Result<()> {
        let pending = Arc::new(Mutex::new(false));
        let notifier = {
            let (pending, irq) = (Arc::clone(&pending), irq.clone());
            // vfio_intx_interrupt(). The lock is held over the raise so that an EOI cannot
            // lower the pin in between.
            Notifiers::new("intx-interrupt", 1, move |_| {
                let mut p = lock(&pending);
                *p = true;
                irq.raise();
            })?
        };
        // vfio_device_irq_set_signaling().
        self.set_triggers(PCI_INTX_IRQ_INDEX, 0, &[notifier.fd(0)]).map_err(|e| {
            Error::from_io("VFIO_DEVICE_SET_IRQS failure", e)
                .prepend("Failed to set up TRIGGER eventfd signaling for interrupt INTX-0: ")
        })?;
        *lock(&self.interrupt) = Interrupt::Intx(Intx { pending, irq, _notifier: notifier });
        Ok(())
    }

    /// `vfio_msi_disable_common()`, for MSI and MSI-X.
    fn msi_disable_common(&self) {
        drop(self.take_interrupt(|i| matches!(i, Interrupt::Msi { .. } | Interrupt::Msix(_))));
    }
}

/// A BAR, `vfio_region_ops`: every access goes to the device.
#[derive(Debug)]
struct BarOps {
    shared: Arc<Shared>,
    nr: usize,
    offset: u64,
}

impl MmioOps for BarOps {
    /// `vfio_region_read()`. Like every BAR access that reaches the device, a read is taken
    /// as the EOI for a pending INTx once it is done, after the device has seen it.
    fn read(&self, _cx: &AccessCtx, addr: u64, size: AccessSize) -> MemResult<u64> {
        let n = size.bytes() as usize;
        let mut b = [0u8; 8];
        let r = self.shared.device.read_at(&mut b[..n], self.offset + addr);
        match check_rw(r, n, "short read") {
            Ok(()) => {
                self.shared.eoi();
                Ok(u64::from_le_bytes(b))
            }
            Err(e) => {
                error_report(&format!(
                    "vfio_region_read({}:region{}+0x{addr:x}, {n}) failed: {e}",
                    self.shared.name, self.nr
                ));
                Ok(u64::MAX >> (64 - 8 * n))
            }
        }
    }

    /// `vfio_region_write()`, then the EOI.
    fn write(&self, _cx: &AccessCtx, addr: u64, size: AccessSize, value: u64) -> MemResult<()> {
        let n = size.bytes() as usize;
        let r = self.shared.device.write_at(&value.to_le_bytes()[..n], self.offset + addr);
        if let Err(e) = check_rw(r, n, "short write") {
            error_report(&format!(
                "vfio_region_write({}:region{}+0x{addr:x}, 0x{value:x},{n}) failed: {e}",
                self.shared.name, self.nr
            ));
        }
        self.shared.eoi();
        Ok(())
    }

    fn valid(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }

    fn impl_constraints(&self) -> AccessConstraints {
        AccessConstraints::any_size(1, 8)
    }
}

#[derive(Debug, Default)]
struct RomState {
    data: Vec<u8>,
    failed: bool,
}

/// The option ROM, `vfio_rom_ops`: read from the device the first time the guest looks.
#[derive(Debug)]
struct RomOps {
    shared: Arc<Shared>,
    vendor_id: u16,
    device_id: u16,
    state: Mutex<RomState>,
}

impl RomOps {
    /// `vfio_pci_load_rom()`.
    fn load(&self) -> Result<Vec<u8>> {
        let dev = &self.shared.device;
        let info = dev
            .region_info(PCI_ROM_REGION_INDEX)
            .map_err(|e| Error::from_io("vfio: Error getting ROM info", e))?;
        if info.size == 0 {
            return Err(Error::generic(format!(
                "vfio-pci: Device ROM size is zero at {}",
                self.shared.name
            ))
            .hint(
                "Device option ROM contents are probably invalid (check dmesg).\nSkip option ROM \
                 probe with rombar=0, or load from file with romfile=\n",
            ));
        }
        let mut rom = vec![0xffu8; info.size as usize];
        let mut off = 0;
        while off < rom.len() {
            match dev.read_at(&mut rom[off..], info.offset + off as u64) {
                Ok(0) => break,
                Ok(n) => off += n,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                    ) => {}
                Err(e) => return Err(Error::from_io("vfio: Error reading device ROM", e)),
            }
        }
        // A ROM with the right vendor but another device ID gets this device's ID and a new
        // checksum.
        if rom.len() >= 0x1a && pci_get_word(&rom, 0) == 0xaa55 {
            let ptr = usize::from(pci_get_word(&rom, 0x18));
            if ptr + 8 < rom.len() && &rom[ptr..ptr + 4] == b"PCIR" {
                let vid = pci_get_word(&rom, ptr + 4);
                let did = pci_get_word(&rom, ptr + 6);
                if vid == self.vendor_id && did != self.device_id {
                    pci_set_word(&mut rom, ptr + 6, self.device_id);
                    rom[6] = 0;
                    let sum = rom.iter().fold(0u8, |a, b| a.wrapping_add(*b));
                    rom[6] = sum.wrapping_neg();
                }
            }
        }
        Ok(rom)
    }
}

impl MmioOps for RomOps {
    /// `vfio_rom_read()`.
    fn read(&self, _cx: &AccessCtx, addr: u64, size: AccessSize) -> MemResult<u64> {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if s.data.is_empty() && !s.failed {
            match self.load() {
                Ok(data) => s.data = data,
                Err(e) => {
                    s.failed = true;
                    report_error(&e);
                }
            }
        }
        let n = size.bytes() as usize;
        let mut b = [0xffu8; 8];
        if let Ok(a) = usize::try_from(addr) {
            if a < s.data.len() {
                let k = n.min(s.data.len() - a);
                b[..k].copy_from_slice(&s.data[a..a + k]);
            }
        }
        Ok(u64::from_le_bytes(b) & (u64::MAX >> (64 - 8 * n)))
    }

    /// `vfio_rom_write()`: the ROM is read only.
    fn write(&self, _cx: &AccessCtx, _addr: u64, _size: AccessSize, _value: u64) -> MemResult<()> {
        Ok(())
    }
}

/// The config space callbacks and the reset of `vfio-pci`.
#[derive(Debug)]
struct VfioPciOps {
    shared: Arc<Shared>,
    /// The device itself, for the threads that send its MSI messages.
    dev: OnceLock<Weak<PciDevice>>,
    msi_cap_size: u8,
    /// `VFIOMSIXInfo::noresize`.
    msix_noresize: bool,
    memory: Weak<MemorySystem>,
    /// `emulated_config_bits`: the bits QEMU answers itself instead of the device.
    emulated: Vec<u8>,
    pm_cap: u8,
    reset_works: bool,
    has_flr: bool,
    has_pm_reset: bool,
}

impl VfioPciOps {
    /// `vfio_intx_enable()`.
    fn intx_enable(&self, dev: &PciDevice) -> Result<()> {
        let val = self.config_read(dev, PCI_INTERRUPT_PIN as u32, 1);
        if val == u32::MAX {
            return Err(Error::generic("failed to read PCI_INTERRUPT_PIN"));
        }
        let pin = val as u8;
        if pin == 0 {
            return Ok(());
        }
        if usize::from(pin) > PCI_NUM_PINS {
            return Err(Error::generic(format!("invalid PCI interrupt pin {pin}")));
        }
        self.disable_interrupts(dev);
        dev.with_config(|c| c.config[PCI_INTERRUPT_PIN] = pin);
        self.shared.intx_start(dev.allocate_irq())
    }

    /// `vfio_disable_interrupts()`. Turning MSI or MSI-X off goes back to INTx, so INTx is
    /// torn down last.
    fn disable_interrupts(&self, dev: &PciDevice) {
        let (msi, msix) = match *lock(&self.shared.interrupt) {
            Interrupt::Msi { .. } => (true, false),
            Interrupt::Msix(_) => (false, true),
            _ => (false, false),
        };
        if msix {
            self.msix_disable(dev);
        } else if msi {
            self.msi_disable(dev);
        }
        self.shared.intx_disable();
    }

    /// `vfio_msix_enable()`. The guest's driver usually turns MSI-X on with every vector still
    /// masked, so the device goes into MSI-X mode with no vector, and each vector gets its
    /// eventfd when the guest unmasks it.
    fn msix_enable(&self, dev: &PciDevice) {
        self.disable_interrupts(dev);
        let entries = dev.msix_nr_vectors_allocated() as usize;
        let pending = Arc::new(Mutex::new(vec![false; entries]));
        let pba = Pba { memory: self.memory.clone(), region: dev.msix_regions().map(|r| r.1) };
        let handler = {
            let (weak, pending, pba) =
                (self.dev.get().cloned().unwrap_or_default(), Arc::clone(&pending), pba.clone());
            // vfio_msi_interrupt(). A masked vector firing needs the PBA, so it is shown.
            move |nr: usize| {
                let Some(d) = weak.upgrade() else { return };
                if d.msix_is_masked(nr as u32) {
                    lock(&pending)[nr] = true;
                    pba.show(true);
                }
                d.msix_notify(nr as u32);
            }
        };
        let notifiers = match Notifiers::empty("interrupt", entries, handler) {
            Ok(n) => n,
            Err(e) => {
                report_error(&e);
                return;
            }
        };
        *lock(&self.shared.interrupt) = Interrupt::Msix(MsixVectors {
            notifiers,
            used: vec![false; entries],
            nr_vectors: 0,
            batch: true,
            noresize: self.msix_noresize,
            pending,
            pba,
        });
        let shared = Arc::clone(&self.shared);
        dev.msix_set_vector_notifiers(
            Arc::new(move |d: &PciDevice, nr: u32, _: MsiMessage| {
                shared.msix_vector_use(d, nr as usize);
            }),
            // vfio_msix_vector_release() only has a KVM irqfd to undo.
            Arc::new(|_: &PciDevice, _: u32| {}),
        );
        let mut interrupt = lock(&self.shared.interrupt);
        let Interrupt::Msix(v) = &mut *interrupt else {
            return;
        };
        v.batch = false;
        if v.nr_vectors != 0 {
            if let Err(e) = self.shared.msix_enable_vectors(v) {
                error_report(&format!("vfio: failed to enable vectors, {}", strerror(&e)));
            }
        } else if let Err(e) = self.shared.msix_enable_no_vec() {
            error_report(&format!("vfio: failed to enable MSI-X, {}", strerror(&e)));
        }
    }

    /// `vfio_msix_disable()`: back to INTx.
    fn msix_disable(&self, dev: &PciDevice) {
        dev.msix_unset_vector_notifiers();
        // MSI-X only releases vectors while it is still on, so the rest are let go here.
        let used: Vec<u32> = match &*lock(&self.shared.interrupt) {
            Interrupt::Msix(v) => {
                (0..v.nr_vectors).filter(|&i| v.used[i]).map(|i| i as u32).collect()
            }
            _ => Vec::new(),
        };
        for nr in used {
            dev.msix_vector_unuse(nr);
        }
        // The index is cleared even with no vector in use: the device may have MSI-X on with
        // none, see msix_enable().
        let _ = self.shared.device.set_irqs(PCI_MSIX_IRQ_INDEX, 0, 0, IRQ_SET_ACTION_TRIGGER);
        self.shared.msi_disable_common();
        if let Err(e) = self.intx_enable(dev) {
            report_error(&e.prepend(format_args!("vfio {}: ", self.shared.name)));
        }
    }

    /// `vfio_msi_enable()`: one eventfd per vector the guest enabled, each sending its vector
    /// when the kernel signals it.
    fn msi_enable(&self, dev: &PciDevice) {
        self.disable_interrupts(dev);
        let nr = dev.msi_nr_vectors_allocated() as usize;
        let weak = self.dev.get().cloned().unwrap_or_default();
        // vfio_msi_interrupt().
        let notifiers = match Notifiers::new("interrupt", nr, move |i| {
            if let Some(d) = weak.upgrade() {
                d.msi_notify(i as u32);
            }
        }) {
            Ok(n) => n,
            Err(e) => {
                report_error(&e);
                error_report("vfio: Error: Failed to enable MSI");
                return;
            }
        };
        // vfio_enable_vectors().
        let fds: Vec<_> = (0..nr).map(|i| notifiers.fd(i)).collect();
        let r = self.shared.set_triggers(PCI_MSI_IRQ_INDEX, 0, &fds);
        match r {
            Ok(()) => *lock(&self.shared.interrupt) = Interrupt::Msi { _vectors: notifiers },
            Err(e) => {
                error_report(&format!("vfio: Error: Failed to setup MSI fds: {}", strerror(&e)));
                drop(notifiers);
                // Failing to set up MSI is outside any spec, so interrupts stay off in the
                // hope that the guest falls back to INTx.
                error_report("vfio: Error: Failed to enable MSI");
            }
        }
    }

    /// `vfio_msi_disable()`: back to INTx.
    fn msi_disable(&self, dev: &PciDevice) {
        // vfio_device_irq_disable().
        let _ = self.shared.device.set_irqs(PCI_MSI_IRQ_INDEX, 0, 0, IRQ_SET_ACTION_TRIGGER);
        self.shared.msi_disable_common();
        if let Err(e) = self.intx_enable(dev) {
            report_error(&e.prepend(format_args!("vfio {}: ", self.shared.name)));
        }
    }

    /// The device in D0, the second half of `vfio_pci_pre_reset()`.
    fn power_on(&self, dev: &PciDevice) {
        if self.pm_cap == 0 {
            return;
        }
        let ctrl = u32::from(self.pm_cap) + PCI_PM_CTRL as u32;
        let val = self.config_read(dev, ctrl, 2);
        if val == u32::MAX {
            return;
        }
        let pmcsr = val as u16;
        if pmcsr & PCI_PM_CTRL_STATE_MASK == 0 {
            return;
        }
        self.config_write(dev, ctrl, u32::from(pmcsr & !PCI_PM_CTRL_STATE_MASK), 2);
        let val = self.config_read(dev, ctrl, 2);
        if val == u32::MAX {
            return;
        }
        let state = val as u16 & PCI_PM_CTRL_STATE_MASK;
        if state != 0 {
            error_report(&format!("vfio: Unable to power on device, stuck in D{state}"));
        }
    }
}

impl PciDeviceOps for VfioPciOps {
    /// `vfio_pci_read_config()`.
    fn config_read(&self, dev: &PciDevice, addr: u32, len: u32) -> u32 {
        let (a, n) = (addr as usize, len as usize);
        let mut b = [0u8; 4];
        if let Some(bits) = self.emulated.get(a..a + n) {
            b[..n].copy_from_slice(bits);
        }
        let emu_bits = u32::from_le_bytes(b);
        let emu_val = if emu_bits != 0 { dev.default_read_config(addr, len) } else { 0 };
        let mut phys_val = 0;
        if !emu_bits & (u32::MAX >> (32 - len * 8)) != 0 {
            match self.shared.config_read_val(a, n) {
                Ok(v) => phys_val = v,
                Err(e) => {
                    error_report(&format!(
                        "vfio_pci_read_config({}, 0x{addr:x}, 0x{len:x}) failed: {e}",
                        self.shared.name
                    ));
                    return u32::MAX;
                }
            }
        }
        (emu_val & emu_bits) | (phys_val & !emu_bits)
    }

    /// `vfio_pci_write_config()`: everything goes to the device, which drops what it does not
    /// take, then to the emulated config space.
    fn config_write(&self, dev: &PciDevice, addr: u32, val: u32, len: u32) {
        if let Err(e) = self.shared.config_write(addr as usize, &val.to_le_bytes()[..len as usize])
        {
            error_report(&format!(
                "vfio_pci_write_config({}, 0x{addr:x}, 0x{val:x}, 0x{len:x}) failed: {e}",
                self.shared.name
            ));
        }
        // MSI and MSI-X enabling and disabling. vfio_update_msi() only moves KVM routes, so a
        // write that leaves MSI on needs nothing more.
        let msi_cap = u32::from(dev.msi_cap());
        if dev.msi_present()
            && addr < msi_cap + u32::from(self.msi_cap_size)
            && msi_cap < addr + len
        {
            let was_enabled = dev.msi_enabled();
            dev.default_write_config(addr, val, len);
            let is_enabled = dev.msi_enabled();
            if !was_enabled && is_enabled {
                self.msi_enable(dev);
            } else if was_enabled && !is_enabled {
                self.msi_disable(dev);
            }
        } else if dev.msix_present()
            && addr < u32::from(dev.msix_cap()) + u32::from(MSIX_CAP_LENGTH)
            && u32::from(dev.msix_cap()) < addr + len
        {
            let was_enabled = dev.msix_enabled();
            dev.default_write_config(addr, val, len);
            let is_enabled = dev.msix_enabled();
            if !was_enabled && is_enabled {
                self.msix_enable(dev);
            } else if was_enabled && !is_enabled {
                self.msix_disable(dev);
            }
        } else {
            dev.default_write_config(addr, val, len);
        }
    }

    /// `vfio_pci_reset()`.
    fn reset(&self, dev: &PciDevice) {
        // vfio_pci_pre_reset(): stop interrupts and DMA and put INTx Disable in a known state.
        self.disable_interrupts(dev);
        let val = self.config_read(dev, PCI_COMMAND as u32, 2);
        if val != u32::MAX {
            let cmd = val as u16
                & !(PCI_COMMAND_IO
                    | PCI_COMMAND_MEMORY
                    | PCI_COMMAND_MASTER
                    | PCI_COMMAND_INTX_DISABLE);
            self.config_write(dev, PCI_COMMAND as u32, u32::from(cmd), 2);
        }
        self.power_on(dev);

        let device = &self.shared.device;
        let flr = self.reset_works && (self.has_flr || !self.has_pm_reset);
        // The bus reset QEMU tries next is not done, so a device that only has a PM reset
        // gets that.
        if !(flr && device.reset().is_ok()) && self.reset_works && self.has_pm_reset {
            let _ = device.reset();
        }

        // vfio_pci_post_reset().
        if let Err(e) = self.intx_enable(dev) {
            report_error(&e.prepend(format_args!("vfio {}: ", self.shared.name)));
        }
        for nr in 0..NUM_BARS {
            if let Err(e) = self.shared.config_write(PCI_BASE_ADDRESS_0 + 4 * nr, &[0; 4]) {
                error_report(&format!(
                    "vfio_pci_post_reset({}) reset bar {nr} failed: {e}",
                    self.shared.name
                ));
            }
        }
    }
}

/// Where the MSI-X table and PBA are, `VFIOMSIXInfo`.
#[derive(Copy, Clone, Debug)]
struct MsixInfo {
    table_bar: usize,
    table_offset: u32,
    pba_bar: usize,
    pba_offset: u32,
    entries: u32,
    /// `VFIO_IRQ_INFO_NORESIZE`.
    noresize: bool,
}

/// The device as `vfio_pci_populate_device()` and `vfio_pci_config_setup()` leave it, before
/// it is on the bus.
#[derive(Debug)]
struct Populated {
    shared: Arc<Shared>,
    bars: [RegionInfo; NUM_BARS],
    express: bool,
    reset_works: bool,
    config: Vec<u8>,
}

/// `vfio_pci_realize()`: puts the host function of `conf` on `bus` at `devfn`, or the first
/// free slot. `express_bus` says whether `bus` is PCI Express, `pci_bus_is_express()`, and
/// `dma_as` is the address space the device's DMA goes to, `pci_device_iommu_address_space()`.
pub fn realize(
    bus: &Arc<PciBus>,
    devfn: Option<u8>,
    express_bus: bool,
    dma_as: &Arc<AddressSpace>,
    conf: &VfioPciConf,
) -> Result<Arc<PciDevice>> {
    let sysfsdev = match (&conf.sysfsdev, conf.host) {
        (Some(s), _) => s.clone(),
        (None, Some(h)) => h.sysfsdev(),
        (None, None) => {
            return Err(Error::generic("No provided host device").hint(
                "Use -device vfio-pci,host=DDDD:BB:DD.F or -device vfio-pci,fd=DEVICE_FD or \
                 -device vfio-pci,sysfsdev=PATH_TO_DEVICE\n",
            ));
        }
    };
    // vfio_device_get_name().
    if let Err(e) = std::fs::metadata(&sysfsdev) {
        return Err(
            Error::from_io("no such host device", e).prepend(format_args!("vfio {sysfsdev}: "))
        );
    }
    let name = basename(&sysfsdev).to_string();
    let prefix = |e: Error| e.prepend(format_args!("vfio {name}: "));

    // vfio_device_is_mdev().
    let mdev = std::fs::canonicalize(format!("{sysfsdev}/subsystem"))
        .is_ok_and(|p| p == Path::new("/sys/bus/mdev"));
    if conf.balloon_allowed && !mdev {
        return Err(prefix(Error::generic(
            "x-balloon-allowed only potentially compatible with mdev devices",
        )));
    }
    let p = populate(&sysfsdev, &name, bus.memory(), dma_as).map_err(prefix)?;
    let info = PciDeviceInfo {
        name: "vfio-pci".to_string(),
        id: conf.id.clone(),
        vendor_id: pci_get_word(&p.config, PCI_VENDOR_ID),
        device_id: pci_get_word(&p.config, PCI_DEVICE_ID),
        multifunction: conf.multifunction,
        express: p.express,
        ..PciDeviceInfo::default()
    };
    let dev = bus.register_device(&info, devfn)?;
    let mut s = Setup::new(&dev, bus, p, express_bus, conf);
    match s.run() {
        Ok(ops) => {
            let ops = Arc::new(ops);
            let _ = ops.dev.set(Arc::downgrade(&dev));
            dev.set_ops(Arc::clone(&ops) as Arc<dyn PciDeviceOps>);
            // vfio_pci_interrupt_setup().
            if let Err(e) = ops.intx_enable(&dev) {
                bus.unregister_device(&dev);
                return Err(prefix(e));
            }
            // QEMU resets every device once before the guest runs, and a hotplugged one right
            // after realize. The boards here are built in their reset state, so a passed through
            // device has to be reset now or it keeps whatever the last user left in it.
            dev.reset();
            Ok(dev)
        }
        Err(e) => {
            bus.unregister_device(&dev);
            Err(prefix(e))
        }
    }
}

/// `vfio_device_attach()`, `vfio_pci_populate_device()` and the config read of
/// `vfio_pci_config_setup()`.
fn populate(
    sysfsdev: &str,
    name: &str,
    memory: &Arc<MemorySystem>,
    space: &Arc<AddressSpace>,
) -> Result<Populated> {
    let (attachment, device, info) = attach(sysfsdev, name, memory, space)?;

    if info.flags & DEVICE_FLAGS_PCI == 0 {
        return Err(Error::generic("this isn't a PCI device"));
    }
    if info.num_regions < PCI_CONFIG_REGION_INDEX + 1 {
        return Err(Error::generic(format!(
            "unexpected number of io regions {}",
            info.num_regions
        )));
    }
    if info.num_irqs < PCI_MSIX_IRQ_INDEX + 1 {
        return Err(Error::generic(format!("unexpected number of irqs {}", info.num_irqs)));
    }
    let mut bars = [RegionInfo::default(); NUM_BARS];
    for (i, bar) in bars.iter_mut().enumerate() {
        *bar = device
            .region_info(i as u32)
            .map_err(|e| Error::from_io(format_args!("failed to get region {i} info"), e))?;
    }
    let config_info = device
        .region_info(PCI_CONFIG_REGION_INDEX)
        .map_err(|e| Error::from_io("failed to get config info", e))?;
    let express = config_info.size != PCI_CONFIG_SPACE_SIZE as u64;
    if let Ok(irq) = device.irq_info(PCI_ERR_IRQ_INDEX) {
        if irq.count != 1 {
            warn_report(&format!("vfio {name}: Could not enable error recovery for the device"));
        }
    }

    let size = if express { PCIE_CONFIG_SPACE_SIZE } else { PCI_CONFIG_SPACE_SIZE };
    let mut config = vec![0u8; size];
    let n = config_info.size.min(size as u64) as usize;
    match device.read_at(&mut config[..n], config_info.offset) {
        Ok(r) if r == n => {}
        Ok(_) => {
            return Err(Error::from_io(
                "failed to read device config space",
                io::Error::from_raw_os_error(EFAULT),
            ));
        }
        Err(e) => return Err(Error::from_io("failed to read device config space", e)),
    }
    let shared = Arc::new(Shared {
        name: name.to_string(),
        interrupt: Mutex::new(Interrupt::None),
        device,
        config_offset: config_info.offset,
        _attachment: attachment,
    });
    Ok(Populated {
        shared,
        bars,
        express,
        reset_works: info.flags & DEVICE_FLAGS_RESET != 0,
        config,
    })
}

/// The rest of `vfio_pci_realize()`, once the function is on the bus.
struct Setup<'a> {
    dev: &'a PciDevice,
    bus: &'a PciBus,
    shared: Arc<Shared>,
    bars: [RegionInfo; NUM_BARS],
    reset_works: bool,
    express_bus: bool,
    conf: &'a VfioPciConf,
    emulated: Vec<u8>,
    vendor_id: u16,
    device_id: u16,
    containers: [Option<RegionId>; NUM_BARS],
    msix: Option<MsixInfo>,
    msix_cap: u8,
    msi_cap: u8,
    msi_cap_size: u8,
    pm_cap: u8,
    has_flr: bool,
    has_pm_reset: bool,
}

impl<'a> Setup<'a> {
    fn new(
        dev: &'a PciDevice,
        bus: &'a PciBus,
        p: Populated,
        express_bus: bool,
        conf: &'a VfioPciConf,
    ) -> Self {
        dev.with_config(|c| c.config.copy_from_slice(&p.config));
        Setup {
            dev,
            bus,
            shared: p.shared,
            bars: p.bars,
            reset_works: p.reset_works,
            express_bus,
            conf,
            emulated: vec![0; dev.config_size()],
            vendor_id: 0,
            device_id: 0,
            containers: [None; NUM_BARS],
            msix: None,
            msix_cap: 0,
            msi_cap: 0,
            msi_cap_size: 0,
            pm_cap: 0,
            has_flr: false,
            has_pm_reset: false,
        }
    }

    fn run(&mut self) -> Result<VfioPciOps> {
        self.config_setup()?;
        self.add_capabilities()?;
        // vfio_pci_interrupt_setup(): QEMU emulates all of MSI and MSI-X.
        if self.msix_cap != 0 {
            let c = usize::from(self.msix_cap);
            self.emulated[c..c + usize::from(MSIX_CAP_LENGTH)].fill(0xff);
        }
        if self.msi_cap != 0 {
            let c = usize::from(self.msi_cap);
            self.emulated[c..c + usize::from(self.msi_cap_size)].fill(0xff);
        }
        Ok(VfioPciOps {
            shared: Arc::clone(&self.shared),
            dev: OnceLock::new(),
            msi_cap_size: self.msi_cap_size,
            msix_noresize: self.msix.is_some_and(|m| m.noresize),
            memory: Arc::downgrade(self.bus.memory()),
            emulated: mem::take(&mut self.emulated),
            pm_cap: self.pm_cap,
            reset_works: self.reset_works,
            has_flr: self.has_flr,
            has_pm_reset: self.has_pm_reset,
        })
    }

    fn config(&self) -> Vec<u8> {
        self.dev.config_bytes()
    }

    fn msi_nonbroken(&self) -> bool {
        self.bus.msi_nonbroken()
    }

    /// `vfio_add_emulated_word()`.
    fn add_emulated_word(&mut self, pos: usize, val: u16, mask: u16) {
        let bits = |b: &mut [u8], v: u16| pci_set_word(b, pos, (pci_get_word(b, pos) & !mask) | v);
        self.dev.with_config(|c| {
            bits(c.config, val);
            bits(c.wmask, !mask);
        });
        bits(&mut self.emulated, mask);
    }

    /// `vfio_add_emulated_long()`.
    fn add_emulated_long(&mut self, pos: usize, val: u32, mask: u32) {
        let bits = |b: &mut [u8], v: u32| pci_set_long(b, pos, (pci_get_long(b, pos) & !mask) | v);
        self.dev.with_config(|c| {
            bits(c.config, val);
            bits(c.wmask, !mask);
        });
        bits(&mut self.emulated, mask);
    }

    /// `vfio_pci_config_setup()` after the config read.
    fn config_setup(&mut self) -> Result<()> {
        let conf = self.conf;
        self.emulated[PCI_ROM_ADDRESS..PCI_ROM_ADDRESS + 4].fill(0xff);
        self.emulated[PCI_BASE_ADDRESS_0..PCI_BASE_ADDRESS_0 + 6 * 4].fill(0xff);

        let config = self.config();
        self.vendor_id = pci_get_word(&config, PCI_VENDOR_ID);
        self.device_id = pci_get_word(&config, PCI_DEVICE_ID);
        if conf.vendor_id != PCI_ANY_ID {
            if conf.vendor_id >= 0xffff {
                return Err(Error::generic("invalid PCI vendor ID provided"));
            }
            self.vendor_id = conf.vendor_id as u16;
            self.add_emulated_word(PCI_VENDOR_ID, self.vendor_id, !0);
        }
        if conf.device_id != PCI_ANY_ID {
            if conf.device_id > 0xffff {
                return Err(Error::generic("invalid PCI device ID provided"));
            }
            self.device_id = conf.device_id as u16;
            self.add_emulated_word(PCI_DEVICE_ID, self.device_id, !0);
        }
        if conf.sub_vendor_id != PCI_ANY_ID {
            if conf.sub_vendor_id > 0xffff {
                return Err(Error::generic("invalid PCI subsystem vendor ID provided"));
            }
            self.add_emulated_word(PCI_SUBSYSTEM_VENDOR_ID, conf.sub_vendor_id as u16, !0);
        }
        if conf.sub_device_id != PCI_ANY_ID {
            if conf.sub_device_id > 0xffff {
                return Err(Error::generic("invalid PCI subsystem device ID provided"));
            }
            self.add_emulated_word(PCI_SUBSYSTEM_ID, conf.sub_device_id as u16, !0);
        }
        if conf.class_code != PCI_ANY_ID {
            if conf.class_code > 0xff_ffff {
                return Err(Error::generic("invalid PCI class code provided"));
            }
            self.add_emulated_long(PCI_CLASS_REVISION, conf.class_code << 8, !0xff);
        }

        // Multifunction is always QEMU's to say.
        self.emulated[PCI_HEADER_TYPE] = PCI_HEADER_TYPE_MULTI_FUNCTION;
        self.dev.with_config(|c| {
            if conf.multifunction {
                c.config[PCI_HEADER_TYPE] |= PCI_HEADER_TYPE_MULTI_FUNCTION;
            } else {
                c.config[PCI_HEADER_TYPE] &= !PCI_HEADER_TYPE_MULTI_FUNCTION;
            }
            // The host's BAR and ROM addresses mean nothing to the guest.
            c.config[PCI_BASE_ADDRESS_0..PCI_BASE_ADDRESS_0 + 24].fill(0);
            c.config[PCI_ROM_ADDRESS..PCI_ROM_ADDRESS + 4].fill(0);
        });

        self.size_rom()?;
        let types = self.bars_prepare();
        self.msix_early_setup()?;
        self.bars_register(types)
    }

    /// `vfio_pci_size_rom()`: a ROM BAR the size of the device's, filled when it is read.
    fn size_rom(&mut self) -> Result<()> {
        let conf = self.conf;
        let name = &self.shared.name;
        let denied = ROM_DENYLIST.contains(&(self.vendor_id, self.device_id));
        if conf.romfile.is_some() || conf.rombar == 0 {
            if denied && conf.romfile.is_some() {
                warn_report(&format!(
                    "Device at {name} is known to cause system instability issues during \
                     option rom execution"
                ));
                eprintln!("Proceeding anyway since user specified romfile");
            }
            return Ok(());
        }
        let sh = &self.shared;
        let mut orig = [0u8; 4];
        let mut size = [0u8; 4];
        let ok = sh.config_read(PCI_ROM_ADDRESS, &mut orig).is_ok()
            && sh.config_write(PCI_ROM_ADDRESS, &PCI_ROM_ADDRESS_MASK.to_le_bytes()).is_ok()
            && sh.config_read(PCI_ROM_ADDRESS, &mut size).is_ok()
            && sh.config_write(PCI_ROM_ADDRESS, &orig).is_ok();
        if !ok {
            error_report(&format!("vfio_pci_size_rom({name}) ROM access failed"));
            return Ok(());
        }
        let size = (!(u32::from_le_bytes(size) & PCI_ROM_ADDRESS_MASK)).wrapping_add(1);
        if size == 0 {
            return Ok(());
        }
        if denied {
            if conf.rombar > 0 {
                warn_report(&format!(
                    "Device at {name} is known to cause system instability issues during \
                     option rom execution"
                ));
                eprintln!("Proceeding anyway since user specified positive value for rombar");
            } else {
                warn_report(&format!(
                    "Rom loading for device at {name} has been disabled due to system \
                     instability issues"
                ));
                eprintln!("Specify rombar=1 or romfile to force");
                return Ok(());
            }
        }
        let ops = RomOps {
            shared: Arc::clone(&self.shared),
            vendor_id: self.vendor_id,
            device_id: self.device_id,
            state: Mutex::new(RomState::default()),
        };
        let mem = self.bus.memory();
        let rom = mem
            .new_io(&format!("vfio[{name}].rom"), u128::from(size), Arc::new(ops))
            .map_err(mem_error)?;
        self.dev.register_bar(PCI_ROM_SLOT, PCI_BASE_ADDRESS_SPACE_MEMORY, rom);
        Ok(())
    }

    /// `vfio_bars_prepare()`: the type bits of each BAR the device has, from the device.
    fn bars_prepare(&self) -> [Option<u8>; NUM_BARS] {
        let mut types = [None; NUM_BARS];
        for (nr, t) in types.iter_mut().enumerate() {
            if self.bars[nr].size == 0 {
                continue;
            }
            match self.shared.config_read_val(PCI_BASE_ADDRESS_0 + 4 * nr, 4) {
                Ok(v) => {
                    let io = v as u8 & PCI_BASE_ADDRESS_SPACE_IO != 0;
                    *t = Some(v as u8 & if io { 0x3 } else { 0xf });
                }
                Err(e) => error_report(&format!("vfio: Failed to read BAR {nr}: {e}")),
            }
        }
        types
    }

    /// `vfio_bars_register()`: each BAR is a container with the trapped region in it, so the
    /// MSI-X table and PBA can sit on top.
    fn bars_register(&mut self, types: [Option<u8>; NUM_BARS]) -> Result<()> {
        let mem = self.bus.memory();
        let name = &self.shared.name;
        for (nr, t) in types.iter().enumerate() {
            let Some(type_) = *t else { continue };
            let size = u128::from(self.bars[nr].size);
            let bar =
                mem.new_container(&format!("{name} base BAR {nr}"), size).map_err(mem_error)?;
            let ops = BarOps { shared: Arc::clone(&self.shared), nr, offset: self.bars[nr].offset };
            let region =
                mem.new_io(&format!("{name} BAR {nr}"), size, Arc::new(ops)).map_err(mem_error)?;
            mem.add_subregion(bar, 0, region).map_err(mem_error)?;
            self.dev.register_bar(nr, type_, bar);
            self.containers[nr] = Some(bar);
        }
        Ok(())
    }

    /// `vfio_msix_early_setup()`: where the MSI-X table and PBA are, from the device.
    fn msix_early_setup(&mut self) -> Result<()> {
        let pos = usize::from(self.dev.find_capability(PCI_CAP_ID_MSIX));
        if pos == 0 {
            return Ok(());
        }
        let read = |off: usize, len: usize, what: &str| {
            self.shared
                .config_read_val(pos + off, len)
                .map_err(|e| Error::generic(format!("failed to read PCI MSIX {what}: {e}")))
        };
        let ctrl = read(PCI_MSIX_FLAGS, 2, "FLAGS")? as u16;
        let table = read(PCI_MSIX_TABLE, 4, "TABLE")?;
        let pba = read(PCI_MSIX_PBA, 4, "PBA")?;
        let mut msix = MsixInfo {
            table_bar: (table & PCI_MSIX_FLAGS_BIRMASK) as usize,
            table_offset: table & !PCI_MSIX_FLAGS_BIRMASK,
            pba_bar: (pba & PCI_MSIX_FLAGS_BIRMASK) as usize,
            pba_offset: pba & !PCI_MSIX_FLAGS_BIRMASK,
            entries: u32::from(ctrl & PCI_MSIX_FLAGS_QSIZE) + 1,
            noresize: false,
        };
        if msix.table_bar >= NUM_BARS || msix.pba_bar >= NUM_BARS {
            return Err(Error::generic(format!(
                "invalid MSI-X BIR, table_bar={} pba_bar={}",
                msix.table_bar, msix.pba_bar
            )));
        }
        let info = self
            .shared
            .device
            .irq_info(PCI_MSIX_IRQ_INDEX)
            .map_err(|e| Error::from_io("failed to get MSI-X irq info", e))?;
        msix.noresize = info.flags & IRQ_INFO_NORESIZE != 0;
        if u64::from(msix.pba_offset) >= self.bars[msix.pba_bar].size {
            if self.vendor_id == PCI_VENDOR_ID_CHELSIO && self.device_id & 0xff00 == 0x5800 {
                // Chelsio T5 VFs report 0x8000 for a PBA at 0x1000 of an 8k BAR.
                msix.pba_offset = 0x1000;
            } else if (self.vendor_id, self.device_id)
                == (PCI_VENDOR_ID_BAIDU, PCI_DEVICE_ID_KUNLUN_VF)
            {
                // Baidu Kunlun VFs report 0x460000 for a PBA at 0xb400 of a 64k BAR.
                msix.pba_offset = 0xb400;
            } else {
                return Err(Error::generic(
                    "hardware reports invalid configuration, MSIX PBA outside of specified BAR",
                ));
            }
        }
        self.msix = Some(msix);
        Ok(())
    }

    /// `vfio_pci_add_capabilities()`.
    fn add_capabilities(&mut self) -> Result<()> {
        let config = self.config();
        if pci_get_word(&config, PCI_STATUS) & PCI_STATUS_CAP_LIST == 0
            || config[PCI_CAPABILITY_LIST] == 0
        {
            return Ok(());
        }
        self.add_std_cap(config[PCI_CAPABILITY_LIST], 0)?;
        // vfio_pci_ats_requested_and_supported(): the legacy container cannot say, so ATS is
        // wanted unless it is off.
        let ats_needed = self.conf.ats != Some(false);
        self.add_ext_cap(ats_needed);
        Ok(())
    }

    /// `vfio_add_std_cap()`: adds the chain from its tail, so it comes out in the device's
    /// order.
    fn add_std_cap(&mut self, pos: u8, depth: u32) -> Result<()> {
        let config = self.config();
        let p = usize::from(pos);
        let cap_id = config[p];
        let next = config[p + PCI_CAP_LIST_NEXT];
        let size = std_cap_max_size(&config, pos);
        if next != 0 && depth < MAX_CAPS {
            self.add_std_cap(next, depth + 1)?;
        } else {
            // Begin the rebuild with QEMU's list bits.
            self.dev.with_config(|c| c.config[PCI_CAPABILITY_LIST] = 0);
            self.emulated[PCI_CAPABILITY_LIST] = 0xff;
            self.emulated[PCI_STATUS] |= PCI_STATUS_CAP_LIST as u8;
        }
        let size = size.min(std_cap_max_size(&self.config(), pos));
        // The next pointer is emulated so caps can be dropped.
        self.emulated[p + PCI_CAP_LIST_NEXT] = 0xff;

        let r = match cap_id {
            PCI_CAP_ID_MSI => self.msi_setup(pos),
            PCI_CAP_ID_EXP => {
                if pci_get_long(&self.config(), p + PCI_EXP_DEVCAP) & PCI_EXP_DEVCAP_FLR != 0 {
                    self.has_flr = true;
                }
                self.setup_pcie_cap(pos, size)
            }
            PCI_CAP_ID_MSIX => self.msix_setup(pos),
            PCI_CAP_ID_PM => {
                if pci_get_word(&self.config(), p + PCI_PM_CTRL) & PCI_PM_CTRL_NO_SOFT_RESET == 0 {
                    self.has_pm_reset = true;
                }
                let r = self.dev.pm_init(pos).map(|cap| self.pm_cap = cap);
                // The PCI core tracks the power state for the BAR mappings, so it has to
                // see writes to it.
                self.dev.with_config(|c| {
                    pci_set_word(c.wmask, p + PCI_PM_CTRL, PCI_PM_CTRL_STATE_MASK)
                });
                r
            }
            PCI_CAP_ID_AF => {
                let cap = self.config()[p + PCI_AF_CAP];
                if cap & PCI_AF_CAP_TP != 0 && cap & PCI_AF_CAP_FLR != 0 {
                    self.has_flr = true;
                }
                self.dev.add_capability(cap_id, pos, size).map(drop)
            }
            PCI_CAP_ID_VNDR => self.dev.add_capability(cap_id, pos, size).map(|at| {
                // The vendor data is not checked on load, only its 3 byte header.
                if self.conf.skip_vsc_check && size > 3 {
                    let at = usize::from(at);
                    self.dev.with_config(|c| c.cmask[at + 3..at + usize::from(size)].fill(0));
                }
            }),
            _ => self.dev.add_capability(cap_id, pos, size).map(drop),
        };
        r.map_err(|e| {
            e.prepend(format_args!(
                "failed to add PCI capability 0x{cap_id:x}[0x{size:x}]@0x{pos:x}: "
            ))
        })
    }

    /// `vfio_msi_setup()`.
    fn msi_setup(&mut self, pos: u8) -> Result<()> {
        let ctrl = self
            .shared
            .config_read_val(usize::from(pos) + PCI_CAP_FLAGS, 2)
            .map_err(|e| Error::generic(format!("failed reading MSI PCI_CAP_FLAGS: {e}")))?
            as u16;
        if !self.msi_nonbroken() {
            // msi_init() says ENOTSUP and the device goes without MSI.
            return Ok(());
        }
        let msi_64bit = ctrl & PCI_MSI_FLAGS_64BIT != 0;
        let msi_maskbit = ctrl & PCI_MSI_FLAGS_MASKBIT != 0;
        let entries = 1 << ((ctrl & PCI_MSI_FLAGS_QMASK) >> 1);
        let cap = self
            .dev
            .msi_init(pos, entries, msi_64bit, msi_maskbit)
            .map_err(|e| e.prepend("msi_init failed: "))?;
        self.msi_cap = cap;
        self.msi_cap_size =
            0xa + if msi_maskbit { 0xa } else { 0 } + if msi_64bit { 0x4 } else { 0 };
        Ok(())
    }

    /// `vfio_msix_setup()`.
    fn msix_setup(&mut self, pos: u8) -> Result<()> {
        let Some(m) = self.msix else {
            return self.dev.add_capability(PCI_CAP_ID_MSIX, pos, MSIX_CAP_LENGTH).map(drop);
        };
        if !self.msi_nonbroken() {
            warn_report("MSI-X is not supported by interrupt controller");
            return Ok(());
        }
        let (Some(table_bar), Some(pba_bar)) =
            (self.containers[m.table_bar], self.containers[m.pba_bar])
        else {
            return Err(Error::generic(format!(
                "MSI-X table or PBA in BAR {} or {}, which the device does not have",
                m.table_bar, m.pba_bar
            )));
        };
        let layout = MsixLayout {
            table_bar,
            table_bar_nr: m.table_bar as u8,
            table_offset: m.table_offset,
            pba_bar,
            pba_bar_nr: m.pba_bar as u8,
            pba_offset: m.pba_offset,
        };
        self.msix_cap = self.dev.msix_init(m.entries, layout, pos)?;
        // The PBA is only shown when a masked vector fires through QEMU. Until then reads go to
        // the device.
        if let Some((_, pba)) = self.dev.msix_regions() {
            self.bus.memory().set_enabled(pba, false).map_err(mem_error)?;
        }
        Ok(())
    }

    /// `vfio_setup_pcie_cap()`.
    fn setup_pcie_cap(&mut self, pos: u8, size: u8) -> Result<()> {
        let p = usize::from(pos);
        let flags = pci_get_word(&self.config(), p + PCI_CAP_FLAGS);
        let type_ = ((flags & PCI_EXP_FLAGS_TYPE) >> 4) as u8;
        if !matches!(type_, PCI_EXP_TYPE_ENDPOINT | PCI_EXP_TYPE_LEG_END | PCI_EXP_TYPE_RC_END) {
            return Err(Error::generic(format!(
                "assignment of PCIe type 0x{type_:x} devices is not currently supported"
            )));
        }
        let size_u = usize::from(size);
        if !self.express_bus {
            // The device sits on a root bus, so the root is this bus and it is not express:
            // the capability stays as it is.
        } else if self.bus.is_root() {
            // On a Root Complex bus Endpoints become Root Complex Integrated Endpoints,
            // without the link registers.
            if type_ == PCI_EXP_TYPE_ENDPOINT {
                self.add_emulated_word(
                    p + PCI_CAP_FLAGS,
                    u16::from(PCI_EXP_TYPE_RC_END) << 4,
                    PCI_EXP_FLAGS_TYPE,
                );
                if size_u > PCI_EXP_LNKCTL {
                    self.add_emulated_long(p + PCI_EXP_LNKCAP, 0, !0);
                    self.add_emulated_word(p + PCI_EXP_LNKCTL, 0, !0);
                    self.add_emulated_word(p + PCI_EXP_LNKSTA, 0, !0);
                    if size_u > PCI_EXP_LNKCAP2 {
                        self.add_emulated_long(p + PCI_EXP_LNKCAP2, 0, !0);
                        self.add_emulated_word(p + PCI_EXP_LNKCTL2, 0, !0);
                        self.add_emulated_word(p + PCI_EXP_LNKSTA2, 0, !0);
                    }
                }
            } else if type_ == PCI_EXP_TYPE_LEG_END {
                // Legacy endpoints don't belong on the root complex.
                return Ok(());
            }
        } else if type_ == PCI_EXP_TYPE_RC_END {
            // Integrated endpoints behind a root port become endpoints with a made up x1
            // 2.5 GT/s link.
            self.add_emulated_word(
                p + PCI_CAP_FLAGS,
                u16::from(PCI_EXP_TYPE_ENDPOINT) << 4,
                PCI_EXP_FLAGS_TYPE,
            );
            self.add_emulated_long(p + PCI_EXP_LNKCAP, PCI_EXP_LNKCAP_X1_2_5GT, !0);
            self.add_emulated_word(p + PCI_EXP_LNKCTL, 0, !0);
        }
        // Intel 82599 VFs report version 0, which Windows refuses on q35.
        if flags & PCI_EXP_FLAGS_VERS == 0 {
            self.add_emulated_word(p + PCI_CAP_FLAGS, 1, PCI_EXP_FLAGS_VERS);
        }
        self.dev.add_capability(PCI_CAP_ID_EXP, pos, size).map(drop)
    }

    /// `vfio_add_ext_cap()`: the extended chain, walked on a copy since adding a capability
    /// rewrites the next pointers.
    fn add_ext_cap(&mut self, ats_needed: bool) {
        let config = self.config();
        let head = PCI_CONFIG_SPACE_SIZE;
        if !self.dev.is_express() || !self.express_bus || pci_get_long(&config, head) == 0 {
            return;
        }
        // A stub head with id 0xffff, so any capability can be dropped by skipping it.
        self.dev.with_config(|c| {
            pci_set_long(c.config, head, pci_ext_cap(0xffff, 0, 0));
            pci_set_long(c.wmask, head, 0);
        });
        pci_set_long(&mut self.emulated, head, !0);

        let mut ats_present = false;
        let mut next = head as u16;
        let mut seen = 0;
        while next != 0 && seen < MAX_EXT_CAPS {
            seen += 1;
            let n = usize::from(next);
            let header = pci_get_long(&config, n);
            let cap_id = pci_ext_cap_id(header);
            let cap_ver = pci_ext_cap_ver(header);
            let size = ext_cap_max_size(&config, next);
            let e = pci_get_long(&self.emulated, n);
            pci_set_long(&mut self.emulated, n, e | PCI_EXT_CAP_NEXT_MASK);
            match cap_id {
                // Masked by the kernel, SR-IOV whose read only VF BARs confuse OVMF, and ARI
                // which needs the next function virtualized.
                0 | PCI_EXT_CAP_ID_SRIOV | PCI_EXT_CAP_ID_ARI => {}
                PCI_EXT_CAP_ID_REBAR => {
                    if self.setup_rebar_ecap(n) {
                        pcie_add_capability(self.dev, cap_id, cap_ver, next, size);
                    }
                }
                PCI_EXT_CAP_ID_ATS => {
                    ats_present = true;
                    if ats_needed {
                        pcie_add_capability(self.dev, cap_id, cap_ver, next, size);
                    }
                }
                _ => pcie_add_capability(self.dev, cap_id, cap_ver, next, size),
            }
            next = pci_ext_cap_next(header);
        }
        if self.conf.ats == Some(true) && !ats_present {
            warn_report(
                "vfio-pci: ats=on requested, but host device has no ATS extended capability",
            );
        }
        if pci_get_word(&self.config(), head) == 0xffff {
            self.dev.with_config(|c| pci_set_word(c.config, head, 0));
        }
    }

    /// `vfio_setup_rebar_ecap()`: each resizable BAR shows only its current size, and the
    /// size is read only. False hides the capability.
    fn setup_rebar_ecap(&mut self, pos: usize) -> bool {
        let config = self.config();
        let ctrl = pci_get_long(&config, pos + PCI_REBAR_CTRL);
        let nbar = ((ctrl & PCI_REBAR_CTRL_NBAR_MASK) >> 5) as usize;
        for i in 0..nbar {
            let at = pos + PCI_REBAR_CTRL + i * 8;
            if at + 4 > config.len() {
                return false;
            }
            let ctrl = pci_get_long(&self.config(), at);
            let size = (ctrl & PCI_REBAR_CTRL_BAR_SIZE) >> 8;
            // The cap register has sizes from 1 MB to 128 TB above 4 reserved bits.
            let cap = if size <= 27 { 1u32 << (size + 4) } else { 0 };
            if cap & PCI_REBAR_CAP_SIZES == 0 {
                return false;
            }
            let ctrl = ctrl
                & (PCI_REBAR_CTRL_BAR_SIZE | PCI_REBAR_CTRL_NBAR_MASK | PCI_REBAR_CTRL_BAR_IDX);
            self.add_emulated_long(pos + PCI_REBAR_CAP + i * 8, cap, !0);
            self.add_emulated_long(at, ctrl, !0);
        }
        true
    }
}

/// `vfio_std_cap_max_size()`: from `pos` to the next capability or the end of the header
/// space.
fn std_cap_max_size(config: &[u8], pos: u8) -> u8 {
    let mut next = PCI_CONFIG_SPACE_SIZE as u16;
    let mut tmp = config[PCI_CAPABILITY_LIST];
    let mut seen = 0;
    while tmp != 0 && seen < MAX_CAPS {
        seen += 1;
        if tmp > pos && u16::from(tmp) < next {
            next = u16::from(tmp);
        }
        tmp = config[usize::from(tmp) + PCI_CAP_LIST_NEXT];
    }
    (next - u16::from(pos)) as u8
}

/// `vfio_ext_cap_max_size()`.
fn ext_cap_max_size(config: &[u8], pos: u16) -> u16 {
    let mut next = PCIE_CONFIG_SPACE_SIZE as u16;
    let mut tmp = PCI_CONFIG_SPACE_SIZE as u16;
    let mut seen = 0;
    while tmp != 0 && seen < MAX_EXT_CAPS {
        seen += 1;
        if tmp > pos && tmp < next {
            next = tmp;
        }
        tmp = pci_ext_cap_next(pci_get_long(config, usize::from(tmp)));
    }
    next - pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basename_like_glib() {
        assert_eq!(basename("/sys/bus/pci/devices/0000:00:04.0"), "0000:00:04.0");
        assert_eq!(basename("/sys/bus/pci/devices/0000:00:04.0/"), "0000:00:04.0");
        assert_eq!(basename("/"), "/");
        assert_eq!(basename(""), ".");
        assert_eq!(basename("dev"), "dev");
    }

    #[test]
    fn cap_sizes() {
        let mut config = vec![0u8; PCIE_CONFIG_SPACE_SIZE];
        config[PCI_CAPABILITY_LIST] = 0x40;
        config[0x41] = 0x70;
        config[0x71] = 0x50;
        assert_eq!(std_cap_max_size(&config, 0x40), 0x10);
        assert_eq!(std_cap_max_size(&config, 0x50), 0x20);
        assert_eq!(std_cap_max_size(&config, 0x70), 0x90);
        // A loop ends.
        config[0x51] = 0x40;
        assert_eq!(std_cap_max_size(&config, 0x70), 0x90);

        pci_set_long(&mut config, 0x100, pci_ext_cap(1, 1, 0x140));
        pci_set_long(&mut config, 0x140, pci_ext_cap(0x15, 1, 0));
        assert_eq!(ext_cap_max_size(&config, 0x100), 0x40);
        assert_eq!(ext_cap_max_size(&config, 0x140), 0xec0);
    }

    #[test]
    fn iova_ranges() {
        assert_eq!(section_iova_range(0, 0xa0000, 0x1000), Some((0, 0xa0000)));
        assert_eq!(section_iova_range(0x100000, 0x7ff00000, 0x1000), Some((0x100000, 0x7ff00000)));
        // Only the whole pages inside.
        assert_eq!(section_iova_range(0x800, 0x2000, 0x1000), Some((0x1000, 0x1000)));
        assert_eq!(section_iova_range(0x800, 0x1000, 0x1000), None);
        assert_eq!(section_iova_range(0xfee00000, 0x10, 0x1000), None);
        assert_eq!(section_iova_range(0, 0x1_0000_0000_0000_0000, 0x1000), None);
    }
}
