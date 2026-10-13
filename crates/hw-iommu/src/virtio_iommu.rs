// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-iommu, from hw/virtio/virtio-iommu.c and hw/virtio/virtio-iommu-pci.c.
//!
//! The device is split in two. [`Viommu`] holds the domains, the endpoints and the config bypass
//! bit, and hands out an IOMMU region for each PCI function through [`Viommu::device_ops`]. The
//! board makes an address space over that region for every function that does its DMA through
//! the IOMMU, as `virtio_iommu_find_add_as()` does. [`VirtioIommu`] is the virtio device model
//! that takes ATTACH, DETACH, MAP, UNMAP and PROBE requests from the driver and changes the
//! shared state, and that reports faults on the event queue.
//!
//! The endpoint ID is the requester ID of the function, `PCI_BUILD_BDF(bus, devfn)`. The board
//! tells the IOMMU which endpoints exist with [`Viommu::set_endpoint_check`], since an attach to
//! a function that is not there fails with `S_NOENT`.
//!
//! # Differences from QEMU
//!
//! - A translation looks for the mapping that holds the address. QEMU looks up the interval
//!   `[addr, addr + 1]`, which also matches a mapping that starts at `addr + 1`.
//! - There are no IOMMU notifiers and no replay, so nothing like vhost or VFIO can sit behind the
//!   IOMMU, and there are no host IOVA ranges from host IOMMU devices.
//! - All requests waiting on the request queue are handled in one go. QEMU stops after as many
//!   as the queue holds and arms a timer for the rest.
//! - A fault found while the device itself is busy, on the same thread, is kept and reported
//!   once the device is done. QEMU reports it on the spot.
//! - There is no VMState, and the `primary-bus` link is not a property: the board picks the bus.

use std::any::Any;
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use ruvm_base::{Error, Result, error_report};
use ruvm_hw_virtio::virtio::{VIRTIO_F_VERSION_1, VirtIODevice, VirtioDeviceClass, feature};
use ruvm_mem::{AddressSpace, IommuAccessFlags, IommuOps, IommuTlbEntry};
use ruvm_virtio_queue::{VIRTIO_F_EVENT_IDX, VIRTIO_F_INDIRECT_DESC};

/// `TYPE_VIRTIO_IOMMU`.
pub const TYPE_VIRTIO_IOMMU: &str = "virtio-iommu-device";
/// `TYPE_VIRTIO_IOMMU_PCI`.
pub const TYPE_VIRTIO_IOMMU_PCI: &str = "virtio-iommu-pci";
/// `TYPE_VIRTIO_IOMMU_MEMORY_REGION`, the stem of the region names.
pub const TYPE_VIRTIO_IOMMU_MEMORY_REGION: &str = "virtio-iommu-memory-region";

/// `VIRTIO_ID_IOMMU`.
pub const VIRTIO_ID_IOMMU: u16 = 23;

/// `VIOMMU_DEFAULT_QUEUE_SIZE`.
const QUEUE_SIZE: u16 = 256;
/// `VIOMMU_PROBE_SIZE`.
const PROBE_SIZE: u32 = 512;
/// `sizeof(struct virtio_iommu_config)`.
const CONFIG_SIZE: usize = 40;
/// Where `bypass` sits in the config space.
const CONFIG_BYPASS: usize = 36;

/// Feature bits.
pub const VIRTIO_IOMMU_F_INPUT_RANGE: u32 = 0;
pub const VIRTIO_IOMMU_F_DOMAIN_RANGE: u32 = 1;
pub const VIRTIO_IOMMU_F_MAP_UNMAP: u32 = 2;
pub const VIRTIO_IOMMU_F_BYPASS: u32 = 3;
pub const VIRTIO_IOMMU_F_PROBE: u32 = 4;
pub const VIRTIO_IOMMU_F_MMIO: u32 = 5;
pub const VIRTIO_IOMMU_F_BYPASS_CONFIG: u32 = 6;

/// Request types.
const T_ATTACH: u8 = 1;
const T_DETACH: u8 = 2;
const T_MAP: u8 = 3;
const T_UNMAP: u8 = 4;
const T_PROBE: u8 = 5;

/// Request status codes.
pub const VIRTIO_IOMMU_S_OK: u8 = 0;
pub const VIRTIO_IOMMU_S_IOERR: u8 = 1;
pub const VIRTIO_IOMMU_S_UNSUPP: u8 = 2;
pub const VIRTIO_IOMMU_S_DEVERR: u8 = 3;
pub const VIRTIO_IOMMU_S_INVAL: u8 = 4;
pub const VIRTIO_IOMMU_S_RANGE: u8 = 5;
pub const VIRTIO_IOMMU_S_NOENT: u8 = 6;
pub const VIRTIO_IOMMU_S_FAULT: u8 = 7;
pub const VIRTIO_IOMMU_S_NOMEM: u8 = 8;

/// `VIRTIO_IOMMU_ATTACH_F_BYPASS`.
const ATTACH_F_BYPASS: u32 = 1;

/// Mapping flags.
pub const VIRTIO_IOMMU_MAP_F_READ: u32 = 1;
pub const VIRTIO_IOMMU_MAP_F_WRITE: u32 = 2;
pub const VIRTIO_IOMMU_MAP_F_MMIO: u32 = 4;
const MAP_F_MASK: u32 =
    VIRTIO_IOMMU_MAP_F_READ | VIRTIO_IOMMU_MAP_F_WRITE | VIRTIO_IOMMU_MAP_F_MMIO;

/// `VIRTIO_IOMMU_PROBE_T_RESV_MEM`.
const PROBE_T_RESV_MEM: u16 = 1;
/// `sizeof(struct virtio_iommu_probe_resv_mem)`.
const RESV_MEM_SIZE: usize = 24;

/// Reserved region subtypes, `VIRTIO_IOMMU_RESV_MEM_T_*`.
pub const VIRTIO_IOMMU_RESV_MEM_T_RESERVED: u32 = 0;
pub const VIRTIO_IOMMU_RESV_MEM_T_MSI: u32 = 1;

/// Fault reasons.
const FAULT_R_UNKNOWN: u8 = 0;
const FAULT_R_DOMAIN: u8 = 1;
const FAULT_R_MAPPING: u8 = 2;
/// Fault flags.
const FAULT_F_READ: u32 = 1;
const FAULT_F_WRITE: u32 = 2;
const FAULT_F_ADDRESS: u32 = 0x100;
/// `sizeof(struct virtio_iommu_fault)`.
const FAULT_SIZE: usize = 24;

/// `sizeof(struct virtio_iommu_req_head)` and `sizeof(struct virtio_iommu_req_tail)`.
const HEAD_SIZE: usize = 4;
const TAIL_SIZE: usize = 4;

/// The `granule` property, `GranuleMode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GranuleMode {
    G4k,
    G8k,
    G16k,
    G64k,
    /// The page size of the host, the default.
    #[default]
    Host,
}

impl GranuleMode {
    /// Parses the property value as QEMU's `GranuleMode` enum does.
    pub fn parse(v: &str) -> Option<Self> {
        Some(match v {
            "4k" => Self::G4k,
            "8k" => Self::G8k,
            "16k" => Self::G16k,
            "64k" => Self::G64k,
            "host" => Self::Host,
            _ => return None,
        })
    }

    /// `config.page_size_mask` for this granule.
    pub fn page_size_mask(self) -> u64 {
        let size: u64 = match self {
            Self::G4k => 4 << 10,
            Self::G8k => 8 << 10,
            Self::G16k => 16 << 10,
            Self::G64k => 64 << 10,
            Self::Host => host_page_size(),
        };
        size.wrapping_neg()
    }
}

/// `qemu_real_host_page_size()`.
fn host_page_size() -> u64 {
    #[cfg(unix)]
    {
        rustix::param::page_size() as u64
    }
    #[cfg(not(unix))]
    {
        4096
    }
}

/// One entry of the `reserved-regions` property, `ReservedRegion`. Both ends are inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReservedRegion {
    pub low: u64,
    pub high: u64,
    /// [`VIRTIO_IOMMU_RESV_MEM_T_RESERVED`] or [`VIRTIO_IOMMU_RESV_MEM_T_MSI`].
    pub kind: u32,
}

/// The properties of `virtio-iommu-device` and `virtio-iommu-pci`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VirtioIommuConf {
    /// `boot-bypass`: whether functions that are not attached to a domain bypass the IOMMU
    /// until the driver says otherwise.
    pub boot_bypass: bool,
    /// `granule`.
    pub granule: GranuleMode,
    /// `aw-bits`: the input address width. The boards change the default of 64.
    pub aw_bits: u8,
    /// `reserved-regions`, which the boards set in their pre-plug hook.
    pub reserved_regions: Vec<ReservedRegion>,
}

impl Default for VirtioIommuConf {
    fn default() -> Self {
        VirtioIommuConf {
            boot_bypass: true,
            granule: GranuleMode::Host,
            aw_bits: 64,
            reserved_regions: Vec::new(),
        }
    }
}

impl VirtioIommuConf {
    /// `virtio_iommu_pci_realize()`: the checks the PCI proxy makes before the device is
    /// realized.
    pub fn check_pci(&self) -> Result<()> {
        for (i, r) in self.reserved_regions.iter().enumerate() {
            if r.kind != VIRTIO_IOMMU_RESV_MEM_T_RESERVED && r.kind != VIRTIO_IOMMU_RESV_MEM_T_MSI {
                return Err(Error::generic(format!("reserved region {i} has an invalid type"))
                    .hint("Valid values are 0 and 1\n"));
            }
        }
        Ok(())
    }
}

/// A mapping of a domain, keyed by the start of its input range.
#[derive(Clone, Copy, Debug)]
struct Mapping {
    /// The last input address, inclusive.
    high: u64,
    phys: u64,
    flags: u32,
}

/// `VirtIOIOMMUDomain`.
#[derive(Debug, Default)]
struct Domain {
    bypass: bool,
    mappings: BTreeMap<u64, Mapping>,
    endpoints: Vec<u32>,
}

impl Domain {
    /// The mapping holding `addr`.
    fn lookup(&self, addr: u64) -> Option<(u64, Mapping)> {
        let (&low, m) = self.mappings.range(..=addr).next_back()?;
        (addr <= m.high).then_some((low, *m))
    }

    /// The first mapping that overlaps `[low, high]`.
    fn overlapping(&self, low: u64, high: u64) -> Option<(u64, Mapping)> {
        if let Some(found) = self.lookup(low) {
            return Some(found);
        }
        let (&l, m) = self.mappings.range(low..).next()?;
        (l <= high).then_some((l, *m))
    }
}

/// What the request handlers and the translations share, under one lock.
#[derive(Debug, Default)]
struct State {
    /// `config.bypass`.
    bypass: bool,
    domains: BTreeMap<u32, Domain>,
    /// The endpoints the driver attached, with their domain.
    endpoints: BTreeMap<u32, Option<u32>>,
    /// Faults waiting to go on the event queue.
    faults: VecDeque<[u8; FAULT_SIZE]>,
}

impl State {
    /// `virtio_iommu_device_bypassed()`.
    fn bypassed(&self, sid: u32) -> bool {
        match self.endpoints.get(&sid) {
            Some(Some(d)) => self.domains.get(d).is_some_and(|d| d.bypass),
            _ => self.bypass,
        }
    }

    /// `virtio_iommu_detach_endpoint_from_domain()`.
    fn detach_from_domain(&mut self, ep: u32) {
        let Some(slot) = self.endpoints.get_mut(&ep) else { return };
        let Some(d) = slot.take() else { return };
        if let Some(domain) = self.domains.get_mut(&d) {
            domain.endpoints.retain(|&e| e != ep);
        }
    }

    /// Drops domain `d` if no endpoint is left in it.
    fn put_domain_if_empty(&mut self, d: u32) {
        if self.domains.get(&d).is_some_and(|d| d.endpoints.is_empty()) {
            self.domains.remove(&d);
        }
    }

    fn push_fault(&mut self, reason: u8, flags: u32, endpoint: u32, address: u64) {
        let mut f = [0; FAULT_SIZE];
        f[0] = reason;
        f[4..8].copy_from_slice(&flags.to_le_bytes());
        f[8..12].copy_from_slice(&endpoint.to_le_bytes());
        f[16..24].copy_from_slice(&address.to_le_bytes());
        self.faults.push_back(f);
    }
}

/// `error_report_once()`, one flag for each place that uses it.
struct Once(AtomicBool);

impl Once {
    const fn new() -> Self {
        Once(AtomicBool::new(false))
    }

    fn report(&self, msg: impl FnOnce() -> String) {
        if !self.0.swap(true, Ordering::Relaxed) {
            error_report(&msg());
        }
    }
}

static SID_UNKNOWN: Once = Once::new();
static NO_DOMAIN: Once = Once::new();
static NO_MAPPING: Once = Once::new();
static PERMISSION: Once = Once::new();
static NO_EVENT_BUFFER: Once = Once::new();

type EndpointCheck = Box<dyn Fn(u32) -> bool + Send + Sync>;
type FaultSink = Box<dyn Fn() + Send + Sync>;

/// The state of a virtio-iommu that the device model and the IOMMU regions share.
pub struct Viommu {
    state: Mutex<State>,
    /// `address_space_memory`, where translated accesses go.
    memory: Weak<AddressSpace>,
    boot_bypass: bool,
    page_size_mask: u64,
    input_end: u64,
    resv: Vec<ReservedRegion>,
    endpoint_check: OnceLock<EndpointCheck>,
    fault_sink: OnceLock<FaultSink>,
}

impl fmt::Debug for Viommu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Viommu")
            .field("boot_bypass", &self.boot_bypass)
            .field("page_size_mask", &format_args!("{:#x}", self.page_size_mask))
            .field("resv", &self.resv)
            .finish_non_exhaustive()
    }
}

impl Viommu {
    /// A virtio-iommu translating into `memory` with the properties `conf`, which
    /// [`VirtioIommu::realize`](VirtioDeviceClass::realize) checks.
    pub fn new(memory: &Arc<AddressSpace>, conf: &VirtioIommuConf) -> Arc<Self> {
        let mut resv = conf.reserved_regions.clone();
        resv.sort_by_key(|r| r.low);
        let input_end = match conf.aw_bits {
            64.. => u64::MAX,
            b => (1u64 << b) - 1,
        };
        Arc::new(Viommu {
            state: Mutex::new(State { bypass: conf.boot_bypass, ..State::default() }),
            memory: Arc::downgrade(memory),
            boot_bypass: conf.boot_bypass,
            page_size_mask: conf.granule.page_size_mask(),
            input_end,
            resv,
            endpoint_check: OnceLock::new(),
            fault_sink: OnceLock::new(),
        })
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Tells the IOMMU which endpoint IDs have a function behind them, `virtio_iommu_mr()`.
    pub fn set_endpoint_check(&self, check: impl Fn(u32) -> bool + Send + Sync + 'static) {
        let _ = self.endpoint_check.set(Box::new(check));
    }

    /// Gives the IOMMU a way to report faults: `sink` should call
    /// [`VirtioIommu::report_faults`] on the device. It may find the device busy, in which case
    /// the faults go out when the device is next done with a request.
    pub fn set_fault_sink(&self, sink: impl Fn() + Send + Sync + 'static) {
        let _ = self.fault_sink.set(Box::new(sink));
    }

    fn endpoint_exists(&self, sid: u32) -> bool {
        self.endpoint_check.get().is_some_and(|c| c(sid))
    }

    /// `config.page_size_mask`.
    pub fn page_size_mask(&self) -> u64 {
        self.page_size_mask
    }

    /// `config.bypass`.
    pub fn bypass(&self) -> bool {
        self.lock().bypass
    }

    /// The domain endpoint `sid` is attached to, if any.
    pub fn endpoint_domain(&self, sid: u32) -> Option<u32> {
        self.lock().endpoints.get(&sid).copied().flatten()
    }

    /// `virtio_iommu_system_reset()` together with `virtio_iommu_device_reset_exit()`: the
    /// domains and endpoints go and `config.bypass` is back to `boot-bypass`.
    pub fn system_reset(&self) {
        let mut st = self.lock();
        st.domains.clear();
        st.endpoints.clear();
        st.faults.clear();
        st.bypass = self.boot_bypass;
    }

    /// The IOMMU region callbacks of the function with requester ID `sid`.
    pub fn device_ops(self: &Arc<Self>, sid: u32) -> Arc<dyn IommuOps> {
        Arc::new(ViommuDevice { viommu: Arc::clone(self), sid })
    }

    /// `virtio_iommu_translate()`, including the bypass that QEMU does by switching the
    /// function to an alias of system memory.
    pub fn translate(&self, sid: u32, addr: u64, flag: IommuAccessFlags) -> IommuTlbEntry {
        let granule = self.page_size_mask.trailing_zeros();
        let mut entry = IommuTlbEntry {
            target_as: self.memory.upgrade(),
            iova: addr,
            translated_addr: addr,
            addr_mask: if granule >= 64 { u64::MAX } else { (1u64 << granule) - 1 },
            perm: IommuAccessFlags::NONE,
        };
        let mut st = self.lock();
        if st.bypassed(sid) {
            return IommuTlbEntry {
                iova: 0,
                translated_addr: 0,
                addr_mask: u64::MAX,
                perm: IommuAccessFlags::RW,
                ..entry
            };
        }
        let fault = 'walk: {
            let Some(&ep) = st.endpoints.get(&sid) else {
                SID_UNKNOWN.report(|| format!("virtio_iommu_translate sid={sid} is not known!!"));
                break 'walk Some((FAULT_R_UNKNOWN, FAULT_F_ADDRESS));
            };
            if let Some(r) = self.resv.iter().find(|r| r.low <= addr && addr <= r.high) {
                if r.kind == VIRTIO_IOMMU_RESV_MEM_T_MSI {
                    entry.perm = flag;
                    break 'walk None;
                }
                break 'walk Some((FAULT_R_MAPPING, FAULT_F_ADDRESS));
            }
            let Some(domain) = ep.and_then(|d| st.domains.get(&d)) else {
                NO_DOMAIN.report(|| {
                    format!(
                        "virtio_iommu_translate {:02x}:{:02x}.{:01x} not attached to any domain",
                        sid >> 8,
                        (sid >> 3) & 0x1f,
                        sid & 7
                    )
                });
                break 'walk Some((FAULT_R_DOMAIN, FAULT_F_ADDRESS));
            };
            let Some((low, m)) = domain.lookup(addr) else {
                NO_MAPPING.report(|| {
                    format!("virtio_iommu_translate no mapping for 0x{addr:x} for sid={sid}")
                });
                break 'walk Some((FAULT_R_MAPPING, FAULT_F_ADDRESS));
            };
            let mut flags = 0;
            if flag.allows(IommuAccessFlags::RO) && m.flags & VIRTIO_IOMMU_MAP_F_READ == 0 {
                flags |= FAULT_F_READ;
            }
            if flag.allows(IommuAccessFlags::WO) && m.flags & VIRTIO_IOMMU_MAP_F_WRITE == 0 {
                flags |= FAULT_F_WRITE;
            }
            if flags != 0 {
                PERMISSION.report(|| {
                    let want = u8::from(flag.allows(IommuAccessFlags::RO))
                        | (u8::from(flag.allows(IommuAccessFlags::WO)) << 1);
                    format!(
                        "virtio_iommu_translate permission error on 0x{addr:x}({want}): allowed={}",
                        m.flags
                    )
                });
                break 'walk Some((FAULT_R_MAPPING, flags | FAULT_F_ADDRESS));
            }
            entry.translated_addr = (addr - low).wrapping_add(m.phys);
            entry.perm = flag;
            None
        };
        if let Some((reason, flags)) = fault {
            st.push_fault(reason, flags, sid, addr);
            drop(st);
            if let Some(sink) = self.fault_sink.get() {
                sink();
            }
        }
        entry
    }

    /// `virtio_iommu_attach()`.
    fn attach(&self, st: &mut State, domain: u32, ep: u32, flags: u32) -> u8 {
        if flags & !ATTACH_F_BYPASS != 0 {
            return VIRTIO_IOMMU_S_INVAL;
        }
        // virtio_iommu_get_endpoint()
        if let Entry::Vacant(e) = st.endpoints.entry(ep) {
            if !self.endpoint_exists(ep) {
                return VIRTIO_IOMMU_S_NOENT;
            }
            e.insert(None);
        }
        if let Some(previous) = st.endpoints.get(&ep).copied().flatten() {
            st.detach_from_domain(ep);
            st.put_domain_if_empty(previous);
        }
        // virtio_iommu_get_domain()
        let bypass = flags & ATTACH_F_BYPASS != 0;
        let d = st.domains.entry(domain).or_insert_with(|| Domain { bypass, ..Domain::default() });
        if d.bypass != bypass {
            return VIRTIO_IOMMU_S_INVAL;
        }
        d.endpoints.insert(0, ep);
        st.endpoints.insert(ep, Some(domain));
        VIRTIO_IOMMU_S_OK
    }

    /// `virtio_iommu_detach()`.
    fn detach(st: &mut State, domain: u32, ep: u32) -> u8 {
        let Some(&current) = st.endpoints.get(&ep) else {
            return VIRTIO_IOMMU_S_NOENT;
        };
        if current != Some(domain) {
            return VIRTIO_IOMMU_S_INVAL;
        }
        st.detach_from_domain(ep);
        st.put_domain_if_empty(domain);
        st.endpoints.remove(&ep);
        VIRTIO_IOMMU_S_OK
    }

    /// `virtio_iommu_map()`.
    fn map(
        st: &mut State,
        domain: u32,
        virt_start: u64,
        virt_end: u64,
        phys: u64,
        flags: u32,
    ) -> u8 {
        if flags & !MAP_F_MASK != 0 {
            return VIRTIO_IOMMU_S_INVAL;
        }
        let Some(d) = st.domains.get_mut(&domain) else {
            return VIRTIO_IOMMU_S_NOENT;
        };
        if d.bypass || d.overlapping(virt_start, virt_end).is_some() {
            return VIRTIO_IOMMU_S_INVAL;
        }
        d.mappings.insert(virt_start, Mapping { high: virt_end, phys, flags });
        VIRTIO_IOMMU_S_OK
    }

    /// `virtio_iommu_unmap()`: drops every mapping inside `[virt_start, virt_end]`, and stops
    /// with `S_RANGE` at one that sticks out.
    fn unmap(st: &mut State, domain: u32, virt_start: u64, virt_end: u64) -> u8 {
        let Some(d) = st.domains.get_mut(&domain) else {
            return VIRTIO_IOMMU_S_NOENT;
        };
        if d.bypass {
            return VIRTIO_IOMMU_S_INVAL;
        }
        while let Some((low, m)) = d.overlapping(virt_start, virt_end) {
            if virt_start <= low && virt_end >= m.high {
                d.mappings.remove(&low);
            } else {
                return VIRTIO_IOMMU_S_RANGE;
            }
        }
        VIRTIO_IOMMU_S_OK
    }

    /// `virtio_iommu_probe()`: fills `buf` with the properties of endpoint `ep`, which are its
    /// reserved regions.
    fn probe(&self, ep: u32, buf: &mut [u8]) -> u8 {
        if !self.endpoint_exists(ep) {
            return VIRTIO_IOMMU_S_NOENT;
        }
        // virtio_iommu_fill_resv_mem_prop()
        if self.resv.len() * RESV_MEM_SIZE > buf.len() {
            return VIRTIO_IOMMU_S_INVAL;
        }
        for (r, out) in self.resv.iter().zip(buf.chunks_exact_mut(RESV_MEM_SIZE)) {
            out[0..2].copy_from_slice(&PROBE_T_RESV_MEM.to_le_bytes());
            out[2..4].copy_from_slice(&((RESV_MEM_SIZE - 4) as u16).to_le_bytes());
            out[4] = r.kind as u8;
            out[8..16].copy_from_slice(&r.low.to_le_bytes());
            out[16..24].copy_from_slice(&r.high.to_le_bytes());
        }
        VIRTIO_IOMMU_S_OK
    }
}

/// The IOMMU region of one function.
struct ViommuDevice {
    viommu: Arc<Viommu>,
    sid: u32,
}

impl IommuOps for ViommuDevice {
    fn translate(&self, addr: u64, flag: IommuAccessFlags, _iommu_idx: u32) -> IommuTlbEntry {
        self.viommu.translate(self.sid, addr, flag)
    }
}

/// The virtio-iommu device model, `VirtIOIOMMU`.
#[derive(Debug)]
pub struct VirtioIommu {
    viommu: Arc<Viommu>,
    conf: VirtioIommuConf,
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap_or_default())
}

fn le64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap_or_default())
}

impl VirtioIommu {
    /// The device model in front of `viommu`, with the properties `conf` that `viommu` was
    /// made with.
    pub fn new(viommu: Arc<Viommu>, conf: VirtioIommuConf) -> Self {
        VirtioIommu { viommu, conf }
    }

    /// The shared state.
    pub fn viommu(&self) -> &Arc<Viommu> {
        &self.viommu
    }

    /// `virtio_iommu_report_fault()` for every fault waiting.
    pub fn report_faults(&mut self, vdev: &mut VirtIODevice) {
        loop {
            let Some(fault) = self.viommu.lock().faults.pop_front() else { return };
            let Some(chain) = vdev.pop(1) else {
                NO_EVENT_BUFFER
                    .report(|| "no buffer available in event queue to report event".to_string());
                continue;
            };
            if chain.writable_len() < FAULT_SIZE as u64 {
                vdev.error("error buffer of wrong size");
                vdev.detach(1, &chain);
                return;
            }
            let mem = Arc::clone(vdev.mem());
            let n = chain.writer(&*mem).write(&fault).unwrap_or(0);
            vdev.push(1, &chain, n as u32);
            vdev.notify(1);
        }
    }

    /// `virtio_iommu_handle_command()`.
    fn handle_command(&mut self, vdev: &mut VirtIODevice) {
        while let Some(chain) = vdev.pop(0) {
            if chain.writable_len() < TAIL_SIZE as u64 || chain.readable_len() < HEAD_SIZE as u64 {
                vdev.error("virtio-iommu bad head/tail size");
                vdev.detach(0, &chain);
                break;
            }
            let mem = Arc::clone(vdev.mem());
            let req = chain.reader(&*mem).read_to_vec().unwrap_or_default();
            let mut tail = [0u8; TAIL_SIZE];
            let mut probe = None;
            if req.len() < HEAD_SIZE {
                tail[0] = VIRTIO_IOMMU_S_DEVERR;
            } else {
                let mut st = self.viommu.lock();
                // virtio_iommu_iov_to_req(): the request without its tail must be there.
                let need = |n: usize| req.len() >= n;
                tail[0] = match req[0] {
                    T_ATTACH if need(20) => {
                        self.viommu.attach(&mut st, le32(&req, 4), le32(&req, 8), le32(&req, 12))
                    }
                    T_DETACH if need(20) => Viommu::detach(&mut st, le32(&req, 4), le32(&req, 8)),
                    T_MAP if need(36) => Viommu::map(
                        &mut st,
                        le32(&req, 4),
                        le64(&req, 8),
                        le64(&req, 16),
                        le64(&req, 24),
                        le32(&req, 32),
                    ),
                    T_UNMAP if need(28) => {
                        Viommu::unmap(&mut st, le32(&req, 4), le64(&req, 8), le64(&req, 16))
                    }
                    T_PROBE => {
                        let mut buf = vec![0u8; PROBE_SIZE as usize + TAIL_SIZE];
                        let status = if need(72) {
                            self.viommu.probe(le32(&req, 4), &mut buf[..PROBE_SIZE as usize])
                        } else {
                            VIRTIO_IOMMU_S_INVAL
                        };
                        buf[PROBE_SIZE as usize] = status;
                        probe = Some(buf);
                        status
                    }
                    T_ATTACH | T_DETACH | T_MAP | T_UNMAP => VIRTIO_IOMMU_S_INVAL,
                    _ => VIRTIO_IOMMU_S_UNSUPP,
                };
            }
            let out: &[u8] = probe.as_deref().unwrap_or(&tail);
            let mut written = 0;
            if chain.writable_len() >= out.len() as u64 {
                written = chain.writer(&*mem).write(out).unwrap_or(0);
            }
            if written != out.len() {
                // The response did not fit, which only a probe can do: send the tail alone.
                tail[0] = VIRTIO_IOMMU_S_DEVERR;
                written = chain.writer(&*mem).write(&tail).unwrap_or(0);
            }
            vdev.push(0, &chain, written as u32);
            vdev.notify(0);
        }
    }
}

impl VirtioDeviceClass for VirtioIommu {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        vdev.init(TYPE_VIRTIO_IOMMU, VIRTIO_ID_IOMMU, CONFIG_SIZE);
        vdev.add_queue(QUEUE_SIZE)?;
        vdev.add_queue(QUEUE_SIZE)?;
        if !(32..=64).contains(&self.conf.aw_bits) {
            return Err(Error::generic("aw-bits must be within [32,64]"));
        }
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        Ok(features
            | feature(VIRTIO_F_EVENT_IDX)
            | feature(VIRTIO_F_INDIRECT_DESC)
            | feature(VIRTIO_F_VERSION_1)
            | feature(VIRTIO_IOMMU_F_INPUT_RANGE)
            | feature(VIRTIO_IOMMU_F_DOMAIN_RANGE)
            | feature(VIRTIO_IOMMU_F_MAP_UNMAP)
            | feature(VIRTIO_IOMMU_F_MMIO)
            | feature(VIRTIO_IOMMU_F_PROBE)
            | feature(VIRTIO_IOMMU_F_BYPASS_CONFIG))
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        if config.len() < CONFIG_SIZE {
            return;
        }
        let v = &self.viommu;
        config[0..8].copy_from_slice(&v.page_size_mask.to_le_bytes());
        config[8..16].copy_from_slice(&0u64.to_le_bytes());
        config[16..24].copy_from_slice(&v.input_end.to_le_bytes());
        config[24..28].copy_from_slice(&0u32.to_le_bytes());
        config[28..32].copy_from_slice(&u32::MAX.to_le_bytes());
        config[32..36].copy_from_slice(&PROBE_SIZE.to_le_bytes());
        config[CONFIG_BYPASS] = u8::from(v.bypass());
    }

    fn set_config(&mut self, vdev: &mut VirtIODevice, config: &mut [u8]) {
        let Some(&bypass) = config.get(CONFIG_BYPASS) else { return };
        let mut st = self.viommu.lock();
        if bypass == u8::from(st.bypass) {
            return;
        }
        if !vdev.has_feature(VIRTIO_IOMMU_F_BYPASS_CONFIG) {
            drop(st);
            vdev.error("cannot set config.bypass");
        } else if bypass > 1 {
            drop(st);
            vdev.error(&format!("invalid config.bypass value '{bypass}'"));
        } else {
            st.bypass = bypass == 1;
        }
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        if queue == 0 {
            self.handle_command(vdev);
        }
        // The event queue has no handler in QEMU. New buffers there are a chance to send what
        // could not go out before.
        self.report_faults(vdev);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn viommu(conf: &VirtioIommuConf) -> Arc<Viommu> {
        let mem = ruvm_mem::MemorySystem::new();
        let root = mem.new_container("system", u64::MAX.into()).unwrap();
        let space = mem.address_space_init(root, "memory").unwrap();
        let v = Viommu::new(&space, conf);
        // Keep the address space alive for the test.
        std::mem::forget(space);
        v.set_endpoint_check(|sid| sid == 0x10 || sid == 0x18);
        v
    }

    #[test]
    fn attach_map_translate() {
        let v = viommu(&VirtioIommuConf { boot_bypass: false, ..VirtioIommuConf::default() });
        let mut st = v.lock();
        assert_eq!(v.attach(&mut st, 1, 0x20, 0), VIRTIO_IOMMU_S_NOENT);
        assert_eq!(v.attach(&mut st, 1, 0x10, 2), VIRTIO_IOMMU_S_INVAL);
        assert_eq!(v.attach(&mut st, 1, 0x10, 0), VIRTIO_IOMMU_S_OK);
        assert_eq!(Viommu::map(&mut st, 2, 0, 0xfff, 0, 3), VIRTIO_IOMMU_S_NOENT);
        assert_eq!(Viommu::map(&mut st, 1, 0x1000, 0x2fff, 0x8000, 1), VIRTIO_IOMMU_S_OK);
        assert_eq!(Viommu::map(&mut st, 1, 0x2000, 0x2fff, 0, 1), VIRTIO_IOMMU_S_INVAL);
        assert_eq!(Viommu::map(&mut st, 1, 0, 0x1000, 0, 1), VIRTIO_IOMMU_S_INVAL);
        assert_eq!(Viommu::map(&mut st, 1, 0x3000, 0x3fff, 0, 8), VIRTIO_IOMMU_S_INVAL);
        drop(st);

        let e = v.translate(0x10, 0x2abc, IommuAccessFlags::RO);
        assert_eq!(e.perm, IommuAccessFlags::RO);
        assert_eq!(e.translated_addr, 0x9abc);
        assert_eq!(e.addr_mask, !v.page_size_mask());
        // A write to a read only mapping and an access outside any mapping both fault.
        assert_eq!(v.translate(0x10, 0x1000, IommuAccessFlags::WO).perm, IommuAccessFlags::NONE);
        assert_eq!(v.translate(0x10, 0x3000, IommuAccessFlags::RO).perm, IommuAccessFlags::NONE);
        // Not attached and no bypass: unknown endpoint.
        assert_eq!(v.translate(0x18, 0, IommuAccessFlags::RO).perm, IommuAccessFlags::NONE);
        let st = v.lock();
        let faults: Vec<_> = st.faults.iter().map(|f| (f[0], le32(f, 4), le32(f, 8))).collect();
        assert_eq!(
            faults,
            [
                (FAULT_R_MAPPING, FAULT_F_WRITE | FAULT_F_ADDRESS, 0x10),
                (FAULT_R_MAPPING, FAULT_F_ADDRESS, 0x10),
                (FAULT_R_UNKNOWN, FAULT_F_ADDRESS, 0x18),
            ]
        );
    }

    #[test]
    fn unmap_ranges() {
        let v = viommu(&VirtioIommuConf::default());
        let mut st = v.lock();
        assert_eq!(v.attach(&mut st, 3, 0x10, 0), VIRTIO_IOMMU_S_OK);
        for i in 0..4u64 {
            let at = i * 0x1000;
            assert_eq!(Viommu::map(&mut st, 3, at, at + 0xfff, at, 3), VIRTIO_IOMMU_S_OK);
        }
        assert_eq!(Viommu::unmap(&mut st, 3, 0, 0x1fff), VIRTIO_IOMMU_S_OK);
        assert_eq!(Viommu::unmap(&mut st, 3, 0x2000, 0x27ff), VIRTIO_IOMMU_S_RANGE);
        assert_eq!(Viommu::unmap(&mut st, 3, 0, u64::MAX), VIRTIO_IOMMU_S_OK);
        assert!(st.domains[&3].mappings.is_empty());
        assert_eq!(Viommu::unmap(&mut st, 4, 0, u64::MAX), VIRTIO_IOMMU_S_NOENT);
    }

    #[test]
    fn bypass_and_domains() {
        let v = viommu(&VirtioIommuConf::default());
        // boot-bypass: nothing attached goes straight through.
        let e = v.translate(0x10, 0x1234, IommuAccessFlags::WO);
        assert_eq!((e.perm, e.translated_addr, e.addr_mask), (IommuAccessFlags::RW, 0, u64::MAX));
        let mut st = v.lock();
        assert_eq!(v.attach(&mut st, 1, 0x10, ATTACH_F_BYPASS), VIRTIO_IOMMU_S_OK);
        assert_eq!(Viommu::map(&mut st, 1, 0, 0xfff, 0, 3), VIRTIO_IOMMU_S_INVAL);
        // A bypass domain cannot take an endpoint asking for translation.
        assert_eq!(v.attach(&mut st, 1, 0x18, 0), VIRTIO_IOMMU_S_INVAL);
        // Moving 0x10 to domain 2 drops domain 1, which is empty now.
        assert_eq!(v.attach(&mut st, 2, 0x10, 0), VIRTIO_IOMMU_S_OK);
        assert!(!st.domains.contains_key(&1));
        assert_eq!(Viommu::detach(&mut st, 1, 0x10), VIRTIO_IOMMU_S_INVAL);
        assert_eq!(Viommu::detach(&mut st, 2, 0x10), VIRTIO_IOMMU_S_OK);
        assert!(st.domains.is_empty() && !st.endpoints.contains_key(&0x10));
        assert_eq!(Viommu::detach(&mut st, 2, 0x10), VIRTIO_IOMMU_S_NOENT);
        drop(st);
        v.system_reset();
        assert!(v.bypass());
    }

    #[test]
    fn reserved_regions_and_probe() {
        let msi = ReservedRegion { low: 0xfee0_0000, high: 0xfeef_ffff, kind: 1 };
        let hole = ReservedRegion { low: 0x1_0000, high: 0x1_ffff, kind: 0 };
        let conf = VirtioIommuConf {
            boot_bypass: false,
            reserved_regions: vec![msi, hole],
            ..VirtioIommuConf::default()
        };
        let v = viommu(&conf);
        let mut st = v.lock();
        assert_eq!(v.attach(&mut st, 1, 0x10, 0), VIRTIO_IOMMU_S_OK);
        drop(st);
        let e = v.translate(0x10, 0xfee0_1004, IommuAccessFlags::WO);
        assert_eq!((e.perm, e.translated_addr), (IommuAccessFlags::WO, 0xfee0_1004));
        assert_eq!(v.translate(0x10, 0x1_0000, IommuAccessFlags::RO).perm, IommuAccessFlags::NONE);

        let mut buf = vec![0; PROBE_SIZE as usize];
        assert_eq!(v.probe(0x20, &mut buf), VIRTIO_IOMMU_S_NOENT);
        assert_eq!(v.probe(0x18, &mut buf), VIRTIO_IOMMU_S_OK);
        // Sorted by address, the hole first.
        assert_eq!(&buf[..8], &[1, 0, 20, 0, 0, 0, 0, 0]);
        assert_eq!(le64(&buf, 8), 0x1_0000);
        assert_eq!(le64(&buf, 16), 0x1_ffff);
        assert_eq!(buf[24 + 4], 1);
        assert_eq!(le64(&buf, 32), 0xfee0_0000);

        let bad = VirtioIommuConf {
            reserved_regions: vec![ReservedRegion { low: 0, high: 1, kind: 2 }],
            ..VirtioIommuConf::default()
        };
        assert_eq!(
            bad.check_pci().unwrap_err().to_string(),
            "reserved region 0 has an invalid type"
        );
    }

    #[test]
    fn granules() {
        assert_eq!(GranuleMode::G4k.page_size_mask(), 0xffff_ffff_ffff_f000);
        assert_eq!(GranuleMode::G64k.page_size_mask(), 0xffff_ffff_ffff_0000);
        assert_eq!(GranuleMode::parse("16k"), Some(GranuleMode::G16k));
        assert_eq!(GranuleMode::parse("2k"), None);
    }
}
