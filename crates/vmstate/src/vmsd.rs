// SPDX-License-Identifier: GPL-2.0-or-later

//! `VMStateDescription`, include/migration/vmstate.h.

use std::fmt;

use ruvm_base::Result;

use crate::field::VmStateField;
use crate::file::{StreamReader, StreamWriter};
use crate::vmstate;

/// `MigrationPriority`. Sections are saved in decreasing priority, so a higher value goes out
/// earlier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum MigPriority {
    /// `MIG_PRI_UNINITIALIZED`, which `save_state_priority()` treats as `MIG_PRI_DEFAULT`.
    #[default]
    Uninitialized,
    /// `MIG_PRI_LOW`: must happen after default.
    Low,
    /// `MIG_PRI_DEFAULT`.
    Default,
    /// `MIG_PRI_IOMMU`: must happen before PCI devices.
    Iommu,
    /// `MIG_PRI_PCI_BUS`: must happen before the IOMMU.
    PciBus,
    /// `MIG_PRI_VIRTIO_MEM`: must happen before the IOMMU.
    VirtioMem,
    /// `MIG_PRI_APIC`: must happen before PCI devices.
    Apic,
    /// `MIG_PRI_GICV3_ITS`: must happen before PCI devices.
    Gicv3Its,
    /// `MIG_PRI_GICV3`: must happen before the ITS.
    Gicv3,
}

/// A hook that QEMU declares both as a plain `int` callback and as an `_errp` flavour. The two
/// report failure with different messages, so a port keeps whichever the original uses.
pub(crate) enum Hook<F: ?Sized, G: ?Sized> {
    Ret(Box<F>),
    Errp(Box<G>),
}

type PreRet<T> = dyn Fn(&mut T) -> i32 + Send + Sync;
type PreErrp<T> = dyn Fn(&mut T) -> Result<()> + Send + Sync;
type PostLoadRet<T> = dyn Fn(&mut T, i32) -> i32 + Send + Sync;
type PostLoadErrp<T> = dyn Fn(&mut T, i32) -> Result<()> + Send + Sync;

pub(crate) type PreHook<T> = Hook<PreRet<T>, PreErrp<T>>;
pub(crate) type PostLoadHook<T> = Hook<PostLoadRet<T>, PostLoadErrp<T>>;
type Needed<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;
type PostSave<T> = Box<dyn Fn(&mut T) + Send + Sync>;

/// `VMStateDescription`: the name, versions, fields, subsections and hooks for one kind of state.
///
/// Descriptions are built once, usually in a `LazyLock` static, because nested structures and
/// subsections refer to other descriptions by `&'static` reference the way QEMU's do.
///
/// ```
/// use std::sync::LazyLock;
/// use ruvm_vmstate::{StreamWriter, VmStateDescription, VmStateField};
///
/// #[derive(Default)]
/// struct Timer {
///     count: u32,
///     enabled: bool,
/// }
///
/// static VMSTATE_TIMER: LazyLock<VmStateDescription<Timer>> = LazyLock::new(|| {
///     VmStateDescription::new("timer")
///         .version_id(2)
///         .minimum_version_id(1)
///         .field(VmStateField::scalar("count", |s: &mut Timer| &mut s.count))
///         .field(VmStateField::scalar("enabled", |s: &mut Timer| &mut s.enabled).version(2))
/// });
///
/// let mut w = StreamWriter::new();
/// let mut timer = Timer { count: 7, enabled: true };
/// VMSTATE_TIMER.save(&mut w, &mut timer).unwrap();
/// assert_eq!(w.as_bytes(), [0, 0, 0, 7, 1]);
/// ```
pub struct VmStateDescription<T: 'static> {
    /// The section or subsection name.
    pub name: &'static str,
    /// The version this side saves.
    pub version_id: i32,
    /// The oldest version this side can load.
    pub minimum_version_id: i32,
    /// The section priority.
    pub priority: MigPriority,
    /// Whether the section is loaded before the rest of the stream.
    pub early_setup: bool,
    /// Whether a device with this state blocks migration.
    pub unmigratable: bool,
    pub(crate) fields: Vec<VmStateField<T>>,
    pub(crate) subsections: Vec<&'static VmStateDescription<T>>,
    pub(crate) needed: Option<Needed<T>>,
    pub(crate) pre_load: Option<PreHook<T>>,
    pub(crate) post_load: Option<PostLoadHook<T>>,
    pub(crate) pre_save: Option<PreHook<T>>,
    pub(crate) post_save: Option<PostSave<T>>,
}

impl<T: 'static> fmt::Debug for VmStateDescription<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VmStateDescription")
            .field("name", &self.name)
            .field("version_id", &self.version_id)
            .field("minimum_version_id", &self.minimum_version_id)
            .field("priority", &self.priority)
            .field("fields", &self.fields)
            .field("subsections", &self.subsections.iter().map(|s| s.name).collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl<T: 'static> VmStateDescription<T> {
    /// A description with no fields, version 0 and minimum version 0, which is what a C
    /// initializer that only sets `.name` gives.
    pub fn new(name: &'static str) -> Self {
        VmStateDescription {
            name,
            version_id: 0,
            minimum_version_id: 0,
            priority: MigPriority::Uninitialized,
            early_setup: false,
            unmigratable: false,
            fields: Vec::new(),
            subsections: Vec::new(),
            needed: None,
            pre_load: None,
            post_load: None,
            pre_save: None,
            post_save: None,
        }
    }

    /// Sets `version_id`, the version this side saves.
    pub fn version_id(mut self, version_id: i32) -> Self {
        self.version_id = version_id;
        self
    }

    /// Sets `minimum_version_id`, the oldest version this side can load.
    pub fn minimum_version_id(mut self, minimum_version_id: i32) -> Self {
        self.minimum_version_id = minimum_version_id;
        self
    }

    /// Sets `priority`.
    pub fn priority(mut self, priority: MigPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Sets `early_setup`.
    pub fn early_setup(mut self, early_setup: bool) -> Self {
        self.early_setup = early_setup;
        self
    }

    /// Sets `unmigratable`.
    pub fn unmigratable(mut self, unmigratable: bool) -> Self {
        self.unmigratable = unmigratable;
        self
    }

    /// Appends a field. Fields go on the wire in the order they are added.
    pub fn field(mut self, field: VmStateField<T>) -> Self {
        self.fields.push(field);
        self
    }

    /// Appends several fields.
    pub fn fields(mut self, fields: impl IntoIterator<Item = VmStateField<T>>) -> Self {
        self.fields.extend(fields);
        self
    }

    /// Appends a subsection. Subsections work on the same state as their parent, and their names
    /// start with the parent's name followed by a slash.
    pub fn subsection(mut self, sub: &'static VmStateDescription<T>) -> Self {
        self.subsections.push(sub);
        self
    }

    /// Sets `needed`, which decides whether a subsection is sent.
    pub fn needed(mut self, needed: impl Fn(&T) -> bool + Send + Sync + 'static) -> Self {
        self.needed = Some(Box::new(needed));
        self
    }

    /// Sets the plain `pre_load` hook. A nonzero return fails the load.
    pub fn pre_load(mut self, hook: impl Fn(&mut T) -> i32 + Send + Sync + 'static) -> Self {
        self.pre_load = Some(Hook::Ret(Box::new(hook)));
        self
    }

    /// Sets `pre_load_errp`.
    pub fn pre_load_errp(
        mut self,
        hook: impl Fn(&mut T) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.pre_load = Some(Hook::Errp(Box::new(hook)));
        self
    }

    /// Sets the plain `post_load` hook, which gets the incoming version. A negative return fails
    /// the load.
    pub fn post_load(mut self, hook: impl Fn(&mut T, i32) -> i32 + Send + Sync + 'static) -> Self {
        self.post_load = Some(Hook::Ret(Box::new(hook)));
        self
    }

    /// Sets `post_load_errp`.
    pub fn post_load_errp(
        mut self,
        hook: impl Fn(&mut T, i32) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.post_load = Some(Hook::Errp(Box::new(hook)));
        self
    }

    /// Sets the plain `pre_save` hook. A negative return fails the save.
    pub fn pre_save(mut self, hook: impl Fn(&mut T) -> i32 + Send + Sync + 'static) -> Self {
        self.pre_save = Some(Hook::Ret(Box::new(hook)));
        self
    }

    /// Sets `pre_save_errp`.
    pub fn pre_save_errp(
        mut self,
        hook: impl Fn(&mut T) -> Result<()> + Send + Sync + 'static,
    ) -> Self {
        self.pre_save = Some(Hook::Errp(Box::new(hook)));
        self
    }

    /// Sets `post_save`, which runs after saving whether or not the save worked.
    pub fn post_save(mut self, hook: impl Fn(&mut T) + Send + Sync + 'static) -> Self {
        self.post_save = Some(Box::new(hook));
        self
    }

    /// The fields in wire order.
    pub fn field_list(&self) -> &[VmStateField<T>] {
        &self.fields
    }

    /// The subsections in wire order.
    pub fn subsection_list(&self) -> &[&'static VmStateDescription<T>] {
        &self.subsections
    }

    /// `vmstate_save_state()`.
    pub fn save(&self, f: &mut StreamWriter, opaque: &mut T) -> Result<()> {
        vmstate::vmstate_save_state(f, self, opaque)
    }

    /// `vmstate_load_state()`, where `version_id` is the version the stream says it carries.
    pub fn load(&self, f: &mut StreamReader<'_>, opaque: &mut T, version_id: i32) -> Result<()> {
        vmstate::vmstate_load_state(f, self, opaque, version_id)
    }

    /// `vmstate_section_needed()`.
    pub fn section_needed(&self, opaque: &T) -> bool {
        self.needed.as_ref().is_none_or(|needed| needed(opaque))
    }
}
