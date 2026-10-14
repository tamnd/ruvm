// SPDX-License-Identifier: MIT OR Apache-2.0

//! IOMMU regions, the translate half of `IOMMUMemoryRegionClass`.
//!
//! Dispatch through an IOMMU region works: an access is translated, clamped to the translated
//! page and continued in the target address space, as `address_space_translate_iommu()` does.
//!
//! IOMMU notifiers let something outside the IOMMU follow its mappings, the way VFIO mirrors
//! them into the host IOMMU. The IOMMU model owns the notifier list of each of its regions,
//! hands it out through [`IommuOps::notifiers`] and sends its map and unmap events to it.
//! QEMU keeps the list in the region itself; here the model holds it, since it is the one that
//! sends the events. There is no generic replay: an IOMMU that takes notifiers brings its own
//! [`IommuOps::replay`], as every IOMMU in QEMU that takes MAP notifiers does.

use std::fmt;
use std::ops::BitOr;
use std::sync::{Arc, Mutex, PoisonError};

use crate::address_space::AddressSpace;
use crate::attrs::MemTxAttrs;

/// Access rights of a translation, `IOMMUAccessFlags`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IommuAccessFlags(u8);

impl IommuAccessFlags {
    /// `IOMMU_NONE`.
    pub const NONE: IommuAccessFlags = IommuAccessFlags(0);
    /// `IOMMU_RO`.
    pub const RO: IommuAccessFlags = IommuAccessFlags(1);
    /// `IOMMU_WO`.
    pub const WO: IommuAccessFlags = IommuAccessFlags(2);
    /// `IOMMU_RW`.
    pub const RW: IommuAccessFlags = IommuAccessFlags(3);

    /// The flag for a read or a write, `IOMMU_ACCESS_FLAG(!is_write, is_write)`.
    pub const fn for_access(is_write: bool) -> Self {
        if is_write { Self::WO } else { Self::RO }
    }

    /// Whether every right in `other` is granted.
    pub const fn allows(self, other: IommuAccessFlags) -> bool {
        self.0 & other.0 == other.0
    }
}

/// The result of a translation, `IOMMUTLBEntry`.
#[derive(Clone)]
pub struct IommuTlbEntry {
    /// Where the translated access goes. `None` together with [`IommuAccessFlags::NONE`] is a
    /// fault.
    pub target_as: Option<Arc<AddressSpace>>,
    /// The input address, page aligned.
    pub iova: u64,
    /// The output address, page aligned.
    pub translated_addr: u64,
    /// The page offset mask, for example `0xfff` for a 4 KiB page.
    pub addr_mask: u64,
    /// What the mapping allows.
    pub perm: IommuAccessFlags,
}

impl fmt::Debug for IommuTlbEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IommuTlbEntry")
            .field("target_as", &self.target_as.as_ref().map(|a| a.name().to_string()))
            .field("iova", &self.iova)
            .field("translated_addr", &self.translated_addr)
            .field("addr_mask", &self.addr_mask)
            .field("perm", &self.perm)
            .finish()
    }
}

/// The events a notifier wants, `IOMMUNotifierFlag`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct IommuNotifierFlag(u8);

impl IommuNotifierFlag {
    /// `IOMMU_NOTIFIER_NONE`.
    pub const NONE: IommuNotifierFlag = IommuNotifierFlag(0);
    /// `IOMMU_NOTIFIER_UNMAP`: a mapping went away.
    pub const UNMAP: IommuNotifierFlag = IommuNotifierFlag(1);
    /// `IOMMU_NOTIFIER_MAP`: a mapping was added.
    pub const MAP: IommuNotifierFlag = IommuNotifierFlag(2);
    /// `IOMMU_NOTIFIER_DEVIOTLB_UNMAP`: a device IOTLB invalidation.
    pub const DEVIOTLB_UNMAP: IommuNotifierFlag = IommuNotifierFlag(4);
    /// `IOMMU_NOTIFIER_IOTLB_EVENTS`, map and unmap.
    pub const IOTLB_EVENTS: IommuNotifierFlag = IommuNotifierFlag(3);

    /// Whether any flag of `other` is set.
    pub const fn intersects(self, other: IommuNotifierFlag) -> bool {
        self.0 & other.0 != 0
    }
}

impl BitOr for IommuNotifierFlag {
    type Output = IommuNotifierFlag;

    fn bitor(self, o: IommuNotifierFlag) -> IommuNotifierFlag {
        IommuNotifierFlag(self.0 | o.0)
    }
}

/// A change of a mapping, `IOMMUTLBEvent`. `ty` is [`IommuNotifierFlag::MAP`],
/// [`IommuNotifierFlag::UNMAP`] or [`IommuNotifierFlag::DEVIOTLB_UNMAP`].
#[derive(Clone, Debug)]
pub struct IommuTlbEvent {
    /// What happened.
    pub ty: IommuNotifierFlag,
    /// The range. An unmap has [`IommuAccessFlags::NONE`].
    pub entry: IommuTlbEntry,
}

/// What a notifier calls with each event in its range, `IOMMUNotify`.
pub type IommuNotify = Box<dyn Fn(&IommuTlbEntry) + Send + Sync>;

/// A notifier on an IOMMU region, `IOMMUNotifier`. It takes the events of `flags` that touch
/// `start..=end`, which are offsets into the region, for IOMMU index `iommu_idx`.
pub struct IommuNotifier {
    notify: IommuNotify,
    /// The events wanted.
    pub flags: IommuNotifierFlag,
    /// The first offset covered.
    pub start: u64,
    /// The last offset covered.
    pub end: u64,
    /// The IOMMU index followed.
    pub iommu_idx: u32,
}

impl fmt::Debug for IommuNotifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IommuNotifier")
            .field("flags", &self.flags)
            .field("start", &self.start)
            .field("end", &self.end)
            .field("iommu_idx", &self.iommu_idx)
            .finish_non_exhaustive()
    }
}

impl IommuNotifier {
    /// `iommu_notifier_init()`.
    pub fn new(
        notify: impl Fn(&IommuTlbEntry) + Send + Sync + 'static,
        flags: IommuNotifierFlag,
        start: u64,
        end: u64,
        iommu_idx: u32,
    ) -> Arc<Self> {
        Arc::new(IommuNotifier { notify: Box::new(notify), flags, start, end, iommu_idx })
    }

    /// `memory_region_notify_iommu_one()`: passes the part of `event` inside the notifier's
    /// range on, if the notifier wants that kind of event.
    pub fn notify_one(&self, event: &IommuTlbEvent) {
        let entry = &event.entry;
        let entry_end = entry.iova.wrapping_add(entry.addr_mask);
        if event.ty == IommuNotifierFlag::UNMAP {
            debug_assert_eq!(entry.perm, IommuAccessFlags::NONE);
        }
        if self.start > entry_end || self.end < entry.iova {
            return;
        }
        let mut tmp = entry.clone();
        tmp.iova = entry.iova.max(self.start);
        tmp.addr_mask = entry_end.min(self.end) - tmp.iova;
        if event.ty.intersects(self.flags) {
            (self.notify)(&tmp);
        }
    }

    /// `memory_region_unmap_iommu_notifier_range()`: an unmap of the notifier's whole range.
    /// `target_as` is the system address space.
    pub fn unmap_range(&self, target_as: Option<Arc<AddressSpace>>) {
        self.notify_one(&IommuTlbEvent {
            ty: IommuNotifierFlag::UNMAP,
            entry: IommuTlbEntry {
                target_as,
                iova: self.start,
                translated_addr: 0,
                addr_mask: self.end - self.start,
                perm: IommuAccessFlags::NONE,
            },
        });
    }
}

#[derive(Default)]
struct NotifierList {
    list: Vec<Arc<IommuNotifier>>,
    /// `iommu_notify_flags`, what the IOMMU was last told the notifiers want.
    flags: IommuNotifierFlag,
}

/// The notifiers of one IOMMU region, `iommu_notify` with `iommu_notify_flags`.
#[derive(Default)]
pub struct IommuNotifiers {
    inner: Mutex<NotifierList>,
}

impl fmt::Debug for IommuNotifiers {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let g = self.lock();
        f.debug_struct("IommuNotifiers")
            .field("len", &g.list.len())
            .field("flags", &g.flags)
            .finish()
    }
}

impl IommuNotifiers {
    /// An empty list.
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, NotifierList> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The notifiers, newest first, `IOMMU_NOTIFIER_FOREACH`. The callbacks run on a copy of
    /// the list, so a notifier may come or go while the events go out.
    pub fn list(&self) -> Vec<Arc<IommuNotifier>> {
        self.lock().list.clone()
    }

    /// `memory_region_notify_iommu()`: sends `event` to the notifiers of `iommu_idx`.
    pub fn notify(&self, iommu_idx: u32, event: &IommuTlbEvent) {
        for n in self.list() {
            if n.iommu_idx == iommu_idx {
                n.notify_one(event);
            }
        }
    }
}

/// `memory_region_update_iommu_notify_flags()`: tells the IOMMU when what the notifiers want
/// changed. The list lock is not held across the call, as the IOMMU may look at the list.
fn update_notify_flags(ops: &dyn IommuOps, list: &IommuNotifiers) -> Result<(), String> {
    let (old, new) = {
        let g = list.lock();
        (g.flags, g.list.iter().fold(IommuNotifierFlag::NONE, |f, n| f | n.flags))
    };
    if old != new {
        ops.notify_flag_changed(old, new)?;
    }
    list.lock().flags = new;
    Ok(())
}

/// `memory_region_register_iommu_notifier()`: adds `n` to the region of `ops`. The IOMMU may
/// refuse the events it asks for, and then `n` is not added.
pub fn register_iommu_notifier(ops: &dyn IommuOps, n: &Arc<IommuNotifier>) -> Result<(), String> {
    assert!(n.flags != IommuNotifierFlag::NONE && n.start <= n.end);
    let Some(list) = ops.notifiers() else {
        return Err("the IOMMU does not take notifiers".to_string());
    };
    list.lock().list.insert(0, Arc::clone(n));
    let r = update_notify_flags(ops, list);
    if r.is_err() {
        list.lock().list.retain(|x| !Arc::ptr_eq(x, n));
    }
    r
}

/// `memory_region_unregister_iommu_notifier()`.
pub fn unregister_iommu_notifier(ops: &dyn IommuOps, n: &Arc<IommuNotifier>) {
    let Some(list) = ops.notifiers() else {
        return;
    };
    list.lock().list.retain(|x| !Arc::ptr_eq(x, n));
    let _ = update_notify_flags(ops, list);
}

/// The callbacks of an IOMMU region.
pub trait IommuOps: Send + Sync {
    /// Translates `addr`, an offset into the region, for an access needing `flag`.
    fn translate(&self, addr: u64, flag: IommuAccessFlags, iommu_idx: u32) -> IommuTlbEntry;

    /// Picks the IOMMU index for an access, `attrs_to_index`. Most IOMMUs have one.
    fn attrs_to_index(&self, attrs: MemTxAttrs) -> u32 {
        let _ = attrs;
        0
    }

    /// The notifier list of the region, or `None` if the IOMMU sends no events.
    fn notifiers(&self) -> Option<&IommuNotifiers> {
        None
    }

    /// `notify_flag_changed`: the notifiers now want `new` instead of `old`. An error refuses
    /// the notifier being registered.
    fn notify_flag_changed(
        &self,
        old: IommuNotifierFlag,
        new: IommuNotifierFlag,
    ) -> Result<(), String> {
        let _ = (old, new);
        Ok(())
    }

    /// `replay`, `memory_region_iommu_replay()`: sends the mappings there are now to `n`
    /// alone.
    fn replay(&self, n: &IommuNotifier) {
        let _ = n;
    }
}
