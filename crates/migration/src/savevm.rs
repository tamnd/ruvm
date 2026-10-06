// SPDX-License-Identifier: GPL-2.0-or-later

//! The migration stream, migration/savevm.c.
//!
//! A stream is the header (`QEMU_VM_FILE_MAGIC`, version 3), the `configuration` section, then
//! sections: `QEMU_VM_SECTION_START`, `PART` and `END` for live state such as RAM that goes out in
//! several passes, and `QEMU_VM_SECTION_FULL` for a device saved in one go. Every section ends with
//! a footer that repeats its id. `QEMU_VM_EOF` ends the state, and the vmdesc JSON, which
//! describes every device section, follows it.
//!
//! [`SaveVm`] is the `SaveStateEntry` list. The machine registers one entry per device with the
//! name and instance id QEMU uses, so that both sides of a migration pair their sections up.

use std::io::Write;
use std::sync::LazyLock;

use ruvm_base::{Error, Result, bail, warn_report};
use ruvm_qapi::types::MigrationCapability;
use ruvm_vmstate::info::{VmStateInfo, VmStateType};
use ruvm_vmstate::{
    JsonWriter, MigPriority, StreamReader, StreamWriter, VmStateDescription, VmStateField,
    vmstate_load_state, vmstate_save_state_vmdesc,
};

/// `QEMU_VM_FILE_MAGIC`, "QEVM".
pub const QEMU_VM_FILE_MAGIC: u32 = 0x5145_564d;
/// `QEMU_VM_FILE_VERSION_COMPAT`, the old format nothing reads any more.
pub const QEMU_VM_FILE_VERSION_COMPAT: u32 = 2;
/// `QEMU_VM_FILE_VERSION`.
pub const QEMU_VM_FILE_VERSION: u32 = 3;

/// `QEMU_VM_EOF`.
pub const QEMU_VM_EOF: u8 = 0x00;
/// `QEMU_VM_SECTION_START`.
pub const QEMU_VM_SECTION_START: u8 = 0x01;
/// `QEMU_VM_SECTION_PART`.
pub const QEMU_VM_SECTION_PART: u8 = 0x02;
/// `QEMU_VM_SECTION_END`.
pub const QEMU_VM_SECTION_END: u8 = 0x03;
/// `QEMU_VM_SECTION_FULL`.
pub const QEMU_VM_SECTION_FULL: u8 = 0x04;
/// `QEMU_VM_SUBSECTION`.
pub const QEMU_VM_SUBSECTION: u8 = 0x05;
/// `QEMU_VM_VMDESCRIPTION`.
pub const QEMU_VM_VMDESCRIPTION: u8 = 0x06;
/// `QEMU_VM_CONFIGURATION`.
pub const QEMU_VM_CONFIGURATION: u8 = 0x07;
/// `QEMU_VM_COMMAND`.
pub const QEMU_VM_COMMAND: u8 = 0x08;
/// `QEMU_VM_SECTION_FOOTER`.
pub const QEMU_VM_SECTION_FOOTER: u8 = 0x7e;

/// `enum qemu_vm_cmd`.
pub mod cmd {
    /// `MIG_CMD_INVALID`.
    pub const INVALID: u16 = 0;
    /// `MIG_CMD_OPEN_RETURN_PATH`.
    pub const OPEN_RETURN_PATH: u16 = 1;
    /// `MIG_CMD_PING`.
    pub const PING: u16 = 2;
    /// `MIG_CMD_POSTCOPY_ADVISE`.
    pub const POSTCOPY_ADVISE: u16 = 3;
    /// `MIG_CMD_POSTCOPY_LISTEN`.
    pub const POSTCOPY_LISTEN: u16 = 4;
    /// `MIG_CMD_POSTCOPY_RUN`.
    pub const POSTCOPY_RUN: u16 = 5;
    /// `MIG_CMD_POSTCOPY_RAM_DISCARD`.
    pub const POSTCOPY_RAM_DISCARD: u16 = 6;
    /// `MIG_CMD_PACKAGED`.
    pub const PACKAGED: u16 = 7;
    /// `MIG_CMD_DEPRECATED_0`, `MIG_CMD_ENABLE_COLO` before 10.2.
    pub const DEPRECATED_0: u16 = 8;
    /// `MIG_CMD_POSTCOPY_RESUME`.
    pub const POSTCOPY_RESUME: u16 = 9;
    /// `MIG_CMD_RECV_BITMAP`.
    pub const RECV_BITMAP: u16 = 10;
    /// `MIG_CMD_SWITCHOVER_START`.
    pub const SWITCHOVER_START: u16 = 11;
    /// `MIG_CMD_MAX`.
    pub const MAX: u16 = 12;
}

/// `mig_cmd_args[]`: the name of each command and its fixed length, or -1.
const MIG_CMD_ARGS: [(&str, i32); cmd::MAX as usize] = [
    ("INVALID", -1),
    ("OPEN_RETURN_PATH", 0),
    ("PING", 4),
    ("POSTCOPY_ADVISE", -1),
    ("POSTCOPY_LISTEN", 0),
    ("POSTCOPY_RUN", 0),
    ("POSTCOPY_RAM_DISCARD", -1),
    ("PACKAGED", 4),
    ("DEPRECATED_0", -1),
    ("POSTCOPY_RESUME", 0),
    ("RECV_BITMAP", -1),
    ("SWITCHOVER_START", 0),
];

/// `VMSTATE_INSTANCE_ID_ANY`: pick the next free instance id for the name.
pub const INSTANCE_ID_ANY: Option<u32> = None;

/// A `QEMUFile` for writing: a [`StreamWriter`] in front of the channel.
///
/// Savers write into the buffer and call [`fflush`](Self::fflush) now and then, which hands
/// what is buffered to the channel. It dereferences to the [`StreamWriter`], so everything that
/// writes a stream writes into it.
pub struct QemuFile<'a> {
    buf: StreamWriter,
    sink: Box<dyn Write + Send + 'a>,
}

impl std::fmt::Debug for QemuFile<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QemuFile").field("transferred", &self.buf.transferred()).finish()
    }
}

/// How much a saver buffers before it should flush.
pub const IO_BUF_SIZE: usize = 32768;

impl<'a> QemuFile<'a> {
    /// `qemu_file_new_output()` over `sink`.
    pub fn new(sink: impl Write + Send + 'a) -> Self {
        QemuFile { buf: StreamWriter::new(), sink: Box::new(sink) }
    }

    /// `qemu_fflush()`: writes out what is buffered. A failed write sticks as `-EIO`.
    pub fn fflush(&mut self) -> Result<()> {
        let data = self.buf.take();
        if self.buf.get_error() != 0 {
            bail!("Failed to write the migration stream: error {}", self.buf.get_error());
        }
        if data.is_empty() {
            return Ok(());
        }
        if let Err(e) = self.sink.write_all(&data).and_then(|()| self.sink.flush()) {
            self.buf.set_error(-ruvm_vmstate::EIO);
            return Err(Error::from_io("Failed to write the migration stream", e));
        }
        Ok(())
    }

    /// Flushes when the buffer holds a good amount, so that a saver writing many pages does not
    /// keep the whole guest in memory.
    pub fn maybe_flush(&mut self) -> Result<()> {
        if self.buf.as_bytes().len() >= IO_BUF_SIZE {
            self.fflush()?;
        }
        Ok(())
    }

    /// `qemu_file_transferred()`.
    pub fn transferred(&self) -> u64 {
        self.buf.transferred()
    }
}

impl std::ops::Deref for QemuFile<'_> {
    type Target = StreamWriter;
    fn deref(&self) -> &StreamWriter {
        &self.buf
    }
}

impl std::ops::DerefMut for QemuFile<'_> {
    fn deref_mut(&mut self) -> &mut StreamWriter {
        &mut self.buf
    }
}

/// The state of a device that goes out in one `QEMU_VM_SECTION_FULL` section, a `vmsd` entry.
pub trait DeviceState: Send {
    /// `vmstate_section_needed()`: whether the section goes out at all.
    fn needed(&mut self) -> bool {
        true
    }

    /// `vmstate_save_vmsd()`. `vmdesc` has the section's object open, with its name and instance
    /// id already in it.
    fn save(&mut self, f: &mut StreamWriter, vmdesc: Option<&mut JsonWriter>) -> Result<()>;

    /// `vmstate_load_vmsd()` at the version the stream carries.
    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()>;
}

/// A [`DeviceState`] described by a `VMStateDescription` over a mirror of the device.
///
/// `get` builds the mirror from the device and `put` hands a loaded mirror back to it. Loading
/// starts from what `get` returns, so fields the stream does not carry keep the device's values.
pub struct VmsdState<T: 'static> {
    vmsd: &'static VmStateDescription<T>,
    get: Box<dyn FnMut() -> Result<T> + Send>,
    put: Box<dyn FnMut(T) -> Result<()> + Send>,
    // What `needed` built, for the `save` that follows it.
    cached: Option<T>,
}

impl<T: 'static> std::fmt::Debug for VmsdState<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VmsdState").field("vmsd", &self.vmsd.name).finish()
    }
}

impl<T: Send + 'static> VmsdState<T> {
    /// A device state over `vmsd`.
    pub fn new(
        vmsd: &'static VmStateDescription<T>,
        get: impl FnMut() -> Result<T> + Send + 'static,
        put: impl FnMut(T) -> Result<()> + Send + 'static,
    ) -> Self {
        VmsdState { vmsd, get: Box::new(get), put: Box::new(put), cached: None }
    }

    fn take(&mut self) -> Result<T> {
        match self.cached.take() {
            Some(t) => Ok(t),
            None => (self.get)(),
        }
    }
}

impl<T: Send + 'static> DeviceState for VmsdState<T> {
    fn needed(&mut self) -> bool {
        match (self.get)() {
            Ok(t) => {
                let needed = self.vmsd.section_needed(&t);
                self.cached = Some(t);
                needed
            }
            // Let `save` report it.
            Err(_) => true,
        }
    }

    fn save(&mut self, f: &mut StreamWriter, vmdesc: Option<&mut JsonWriter>) -> Result<()> {
        let mut t = self.take()?;
        vmstate_save_state_vmdesc(f, self.vmsd, &mut t, vmdesc)
    }

    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()> {
        self.cached = None;
        let mut t = (self.get)()?;
        vmstate_load_state(f, self.vmsd, &mut t, version_id)?;
        (self.put)(t)
    }
}

/// A section the machine accepts from QEMU but has no state for: it is parsed with its
/// description, so a malformed stream still fails, and then dropped. It is never sent.
pub struct Discard<T: 'static> {
    vmsd: &'static VmStateDescription<T>,
    init: Box<dyn Fn() -> T + Send>,
}

impl<T: 'static> std::fmt::Debug for Discard<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Discard").field("vmsd", &self.vmsd.name).finish()
    }
}

impl<T: Default + 'static> Discard<T> {
    /// A discarding entry over `vmsd`.
    pub fn new(vmsd: &'static VmStateDescription<T>) -> Self {
        Discard::with_init(vmsd, T::default)
    }
}

impl<T: 'static> Discard<T> {
    /// A discarding entry over `vmsd` that loads into what `init` makes, for descriptions whose
    /// array lengths come from the device rather than from the stream.
    pub fn with_init(
        vmsd: &'static VmStateDescription<T>,
        init: impl Fn() -> T + Send + 'static,
    ) -> Self {
        Discard { vmsd, init: Box::new(init) }
    }
}

impl<T: 'static> DeviceState for Discard<T> {
    fn needed(&mut self) -> bool {
        false
    }

    fn save(&mut self, _f: &mut StreamWriter, _vmdesc: Option<&mut JsonWriter>) -> Result<()> {
        Ok(())
    }

    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()> {
        let mut t = (self.init)();
        vmstate_load_state(f, self.vmsd, &mut t, version_id)
    }
}

/// State that goes out in several passes while the guest runs, the `SaveVMHandlers` of an
/// iterable entry such as RAM.
pub trait LiveState: Send {
    /// `is_active`.
    fn is_active(&self) -> bool {
        true
    }

    /// `save_setup`: the `QEMU_VM_SECTION_START` payload.
    fn save_setup(&mut self, f: &mut QemuFile<'_>) -> Result<()>;

    /// `save_live_iterate`: one `QEMU_VM_SECTION_PART` payload of at most about `max_bytes`.
    /// Returns true when everything pending at the last sync went out.
    fn save_iterate(&mut self, f: &mut QemuFile<'_>, max_bytes: u64) -> Result<bool>;

    /// `save_complete`: the `QEMU_VM_SECTION_END` payload, with the guest stopped.
    fn save_complete(&mut self, f: &mut QemuFile<'_>) -> Result<()>;

    /// `state_pending_estimate` and, with `exact`, `state_pending_exact` (which syncs the dirty
    /// log first): the bytes still to send.
    fn pending(&mut self, exact: bool) -> u64;

    /// `save_cleanup`.
    fn save_cleanup(&mut self) {}

    /// `load_setup`.
    fn load_setup(&mut self) -> Result<()> {
        Ok(())
    }

    /// `load_state`: one section of any type.
    fn load(&mut self, f: &mut StreamReader<'_>, version_id: i32) -> Result<()>;

    /// `load_cleanup`.
    fn load_cleanup(&mut self) {}
}

enum Kind {
    Device(Box<dyn DeviceState>),
    Live(Box<dyn LiveState>),
}

/// `SaveStateEntry`.
struct Entry {
    idstr: String,
    instance_id: u32,
    version_id: i32,
    section_id: u32,
    load_section_id: Option<u32>,
    priority: MigPriority,
    early_setup: bool,
    kind: Kind,
}

/// How an entry is registered: everything of `vmstate_register_with_alias_id()` except the
/// state itself.
#[derive(Debug, Clone)]
pub struct EntryInfo {
    /// The section name, the vmsd name with the device's qdev path in front for devices on a bus
    /// that has one, like `0000:00:1f.0/ICH9LPC`.
    pub idstr: String,
    /// The instance id, or [`INSTANCE_ID_ANY`] for the next free one.
    pub instance_id: Option<u32>,
    /// The version the section is saved at.
    pub version_id: i32,
    /// `MigrationPriority`.
    pub priority: MigPriority,
    /// `early_setup`: saved with the setup sections rather than at completion.
    pub early_setup: bool,
}

impl EntryInfo {
    /// An entry named `idstr` at `version_id`, with the default priority.
    pub fn new(idstr: impl Into<String>, version_id: i32) -> Self {
        EntryInfo {
            idstr: idstr.into(),
            instance_id: INSTANCE_ID_ANY,
            version_id,
            priority: MigPriority::Default,
            early_setup: false,
        }
    }

    /// Takes the name, version, priority and `early_setup` from `vmsd`, with `prefix` (the qdev
    /// path and a slash) in front of the name.
    pub fn from_vmsd<T>(prefix: &str, vmsd: &VmStateDescription<T>) -> Self {
        EntryInfo {
            idstr: format!("{prefix}{}", vmsd.name),
            instance_id: INSTANCE_ID_ANY,
            version_id: vmsd.version_id,
            priority: vmsd.priority,
            early_setup: vmsd.early_setup,
        }
    }

    /// Sets the instance id.
    pub fn instance(mut self, id: u32) -> Self {
        self.instance_id = Some(id);
        self
    }

    /// Sets the priority.
    pub fn priority(mut self, priority: MigPriority) -> Self {
        self.priority = priority;
        self
    }
}

fn effective_priority(p: MigPriority) -> MigPriority {
    if p == MigPriority::Uninitialized { MigPriority::Default } else { p }
}

/// What the machine tells the configuration section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MachineConfig {
    /// The machine type name, `pc-q35-11.1`.
    pub name: String,
    /// `qemu_target_page_bits()`.
    pub page_bits: u32,
    /// `migration_legacy_page_bits()`: the smallest page size of the target.
    pub legacy_page_bits: u32,
    /// `qemu_uuid`, when `-uuid` set one.
    pub uuid: Option<[u8; 16]>,
}

/// What the stream said about the source, kept after a load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadInfo {
    /// Whether a vmdesc followed `QEMU_VM_EOF`.
    pub vmdesc: Option<String>,
    /// Whether the source asked for a return path.
    pub return_path: bool,
    /// Whether `MIG_CMD_SWITCHOVER_START` came.
    pub switchover_start: bool,
}

/// `SaveState`: the entries of a machine, and the saving and loading of a whole stream.
pub struct SaveVm {
    config: MachineConfig,
    entries: Vec<Entry>,
    next_section_id: u32,
    /// `MigrationState.send_configuration`, on for every current machine type.
    pub send_configuration: bool,
    /// `MigrationState.send_section_footer`, on for every current machine type.
    pub send_section_footer: bool,
    /// `!machine->suppress_vmdesc`.
    pub send_vmdesc: bool,
    /// The capabilities on for this migration, which the configuration section checks.
    pub capabilities: Vec<MigrationCapability>,
    vmdesc: Option<JsonWriter>,
}

impl std::fmt::Debug for SaveVm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaveVm")
            .field("config", &self.config)
            .field(
                "entries",
                &self.entries.iter().map(|e| (&e.idstr, e.instance_id)).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SaveVm {
    /// An empty list for a machine.
    pub fn new(config: MachineConfig) -> Self {
        SaveVm {
            config,
            entries: Vec::new(),
            next_section_id: 0,
            send_configuration: true,
            send_section_footer: true,
            send_vmdesc: true,
            capabilities: Vec::new(),
            vmdesc: None,
        }
    }

    /// The machine configuration.
    pub fn config(&self) -> &MachineConfig {
        &self.config
    }

    /// `calculate_new_instance_id()`.
    fn new_instance_id(&self, idstr: &str) -> u32 {
        self.entries
            .iter()
            .filter(|e| e.idstr == idstr)
            .map(|e| e.instance_id + 1)
            .max()
            .unwrap_or(0)
    }

    fn insert(&mut self, info: EntryInfo, kind: Kind) -> u32 {
        let instance_id = info.instance_id.unwrap_or_else(|| self.new_instance_id(&info.idstr));
        assert!(
            !self.entries.iter().any(|e| e.idstr == info.idstr && e.instance_id == instance_id),
            "savevm entry {} {} registered twice",
            info.idstr,
            instance_id
        );
        let entry = Entry {
            idstr: info.idstr,
            instance_id,
            version_id: info.version_id,
            section_id: self.next_section_id,
            load_section_id: None,
            priority: effective_priority(info.priority),
            early_setup: info.early_setup,
            kind,
        };
        self.next_section_id += 1;
        // savevm_state_handler_insert(): in front of the first entry of lower priority.
        let at = self
            .entries
            .iter()
            .position(|e| e.priority < entry.priority)
            .unwrap_or(self.entries.len());
        self.entries.insert(at, entry);
        instance_id
    }

    /// `vmstate_register()`: returns the instance id the entry got.
    pub fn register_device(&mut self, info: EntryInfo, state: impl DeviceState + 'static) -> u32 {
        self.insert(info, Kind::Device(Box::new(state)))
    }

    /// [`register_device`](Self::register_device) for a mirror over `vmsd`, with the name,
    /// version and priority of the description.
    pub fn register_vmsd<T: Send + 'static>(
        &mut self,
        prefix: &str,
        instance_id: Option<u32>,
        vmsd: &'static VmStateDescription<T>,
        get: impl FnMut() -> Result<T> + Send + 'static,
        put: impl FnMut(T) -> Result<()> + Send + 'static,
    ) -> u32 {
        let mut info = EntryInfo::from_vmsd(prefix, vmsd);
        info.instance_id = instance_id;
        self.register_device(info, VmsdState::new(vmsd, get, put))
    }

    /// Registers a section that is accepted and dropped, see [`Discard`].
    pub fn register_discard<T: Default + Send + 'static>(
        &mut self,
        prefix: &str,
        instance_id: Option<u32>,
        vmsd: &'static VmStateDescription<T>,
    ) -> u32 {
        let mut info = EntryInfo::from_vmsd(prefix, vmsd);
        info.instance_id = instance_id;
        self.register_device(info, Discard::new(vmsd))
    }

    /// [`register_discard`](Self::register_discard) loading into what `init` makes.
    pub fn register_discard_with<T: Send + 'static>(
        &mut self,
        prefix: &str,
        instance_id: Option<u32>,
        vmsd: &'static VmStateDescription<T>,
        init: impl Fn() -> T + Send + 'static,
    ) -> u32 {
        let mut info = EntryInfo::from_vmsd(prefix, vmsd);
        info.instance_id = instance_id;
        self.register_device(info, Discard::with_init(vmsd, init))
    }

    /// Skips a section id, for an entry QEMU registers at this point that this machine does not
    /// have. QEMU looks up `QEMU_VM_SECTION_PART` and `QEMU_VM_SECTION_END` by section id and
    /// takes id 0 for any entry it has not loaded yet, so a live entry must not get id 0, and
    /// keeping QEMU's ids keeps the streams comparable.
    pub fn reserve_section_id(&mut self) {
        self.next_section_id += 1;
    }

    /// `register_savevm_live()`.
    pub fn register_live(&mut self, info: EntryInfo, state: impl LiveState + 'static) -> u32 {
        self.insert(info, Kind::Live(Box::new(state)))
    }

    /// The registered sections in stream order, as `(idstr, instance_id, version_id)`.
    pub fn entries(&self) -> impl Iterator<Item = (&str, u32, i32)> {
        self.entries.iter().map(|e| (e.idstr.as_str(), e.instance_id, e.version_id))
    }

    fn validatable_caps(&self) -> Vec<Capability> {
        self.capabilities
            .iter()
            .filter(|c| should_validate_capability(**c))
            .map(|c| Capability(*c))
            .collect()
    }

    fn section_header(&self, f: &mut StreamWriter, e: &Entry, section_type: u8) {
        f.put_byte(section_type);
        f.put_be32(e.section_id);
        if section_type == QEMU_VM_SECTION_FULL || section_type == QEMU_VM_SECTION_START {
            f.put_byte(e.idstr.len() as u8);
            f.put_buffer(e.idstr.as_bytes());
            f.put_be32(e.instance_id);
            f.put_be32(e.version_id as u32);
        }
    }

    fn section_footer(&self, f: &mut StreamWriter, e: &Entry) {
        if self.send_section_footer {
            f.put_byte(QEMU_VM_SECTION_FOOTER);
            f.put_be32(e.section_id);
        }
    }

    /// `qemu_savevm_state_header()`: the magic, the version and the configuration section. It
    /// also starts the vmdesc.
    pub fn save_header(&mut self, f: &mut StreamWriter) -> Result<()> {
        f.put_be32(QEMU_VM_FILE_MAGIC);
        f.put_be32(QEMU_VM_FILE_VERSION);
        let mut vmdesc = self.send_vmdesc.then(JsonWriter::new);
        if self.send_configuration {
            f.put_byte(QEMU_VM_CONFIGURATION);
            if let Some(d) = vmdesc.as_mut() {
                d.start_object(None);
                d.start_object(Some("configuration"));
            }
            let validate_uuid = self.capabilities.contains(&MigrationCapability::ValidateUuid);
            let mut conf =
                Configuration::for_save(&self.config, self.validatable_caps(), validate_uuid);
            vmstate_save_state_vmdesc(f, &VMSTATE_CONFIGURATION, &mut conf, vmdesc.as_mut())?;
            if let Some(d) = vmdesc.as_mut() {
                d.end_object();
            }
        }
        self.vmdesc = vmdesc;
        Ok(())
    }

    fn vmstate_save(
        f: &mut StreamWriter,
        e: &mut Entry,
        footer: bool,
        vmdesc: Option<&mut JsonWriter>,
    ) -> Result<()> {
        let Kind::Device(dev) = &mut e.kind else { return Ok(()) };
        if !dev.needed() {
            return Ok(());
        }
        f.put_byte(QEMU_VM_SECTION_FULL);
        f.put_be32(e.section_id);
        f.put_byte(e.idstr.len() as u8);
        f.put_buffer(e.idstr.as_bytes());
        f.put_be32(e.instance_id);
        f.put_be32(e.version_id as u32);
        let mut vmdesc = vmdesc;
        if let Some(d) = vmdesc.as_deref_mut() {
            d.start_object(None);
            d.str(Some("name"), &e.idstr);
            d.int64(Some("instance_id"), i64::from(e.instance_id));
        }
        dev.save(f, vmdesc.as_deref_mut())
            .map_err(|err| err.prepend(format!("Failed to save {}: ", e.idstr)))?;
        if footer {
            f.put_byte(QEMU_VM_SECTION_FOOTER);
            f.put_be32(e.section_id);
        }
        if let Some(d) = vmdesc {
            d.end_object();
        }
        Ok(())
    }

    /// `qemu_savevm_state_do_setup()`: the early devices and the `QEMU_VM_SECTION_START` of each
    /// live entry.
    pub fn save_setup(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        if let Some(d) = self.vmdesc.as_mut() {
            d.int64(Some("page_size"), 1i64 << self.config.page_bits);
            d.start_array(Some("devices"));
        }
        let footer = self.send_section_footer;
        for e in self.entries.iter_mut().filter(|e| e.early_setup) {
            Self::vmstate_save(f, e, footer, self.vmdesc.as_mut())?;
        }
        for i in 0..self.entries.len() {
            let active = matches!(&self.entries[i].kind, Kind::Live(l) if l.is_active());
            if !active {
                continue;
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_START);
            if let Kind::Live(l) = &mut self.entries[i].kind {
                l.save_setup(f)?;
            }
            self.section_footer(f, &self.entries[i]);
        }
        f.fflush()
    }

    /// `qemu_savevm_state_iterate()`: one `QEMU_VM_SECTION_PART` per live entry, stopping at
    /// the first that has more to send. Returns true when all of them are done.
    pub fn save_iterate(&mut self, f: &mut QemuFile<'_>, max_bytes: u64) -> Result<bool> {
        for i in 0..self.entries.len() {
            let active = matches!(&self.entries[i].kind, Kind::Live(l) if l.is_active());
            if !active {
                continue;
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_PART);
            let done = match &mut self.entries[i].kind {
                Kind::Live(l) => l.save_iterate(f, max_bytes)?,
                Kind::Device(_) => true,
            };
            self.section_footer(f, &self.entries[i]);
            if !done {
                f.fflush()?;
                return Ok(false);
            }
        }
        f.fflush()?;
        Ok(true)
    }

    /// The bytes the live entries still have to send.
    pub fn pending(&mut self, exact: bool) -> u64 {
        self.entries
            .iter_mut()
            .filter_map(|e| match &mut e.kind {
                Kind::Live(l) if l.is_active() => Some(l.pending(exact)),
                _ => None,
            })
            .sum()
    }

    /// `qemu_savevm_state_complete_precopy()`: the `QEMU_VM_SECTION_END` of each live entry, the
    /// device sections, `QEMU_VM_EOF` and the vmdesc. The guest must be stopped.
    pub fn save_complete(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        for i in 0..self.entries.len() {
            let active = matches!(&self.entries[i].kind, Kind::Live(l) if l.is_active());
            if !active {
                continue;
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_END);
            if let Kind::Live(l) = &mut self.entries[i].kind {
                l.save_complete(f)
                    .map_err(|e| e.prepend("Failed to save iterable device state: "))?;
            }
            self.section_footer(f, &self.entries[i]);
            f.fflush()?;
        }
        let footer = self.send_section_footer;
        for e in self.entries.iter_mut().filter(|e| !e.early_setup) {
            Self::vmstate_save(f, e, footer, self.vmdesc.as_mut())?;
        }
        f.put_byte(QEMU_VM_EOF);
        if let Some(mut d) = self.vmdesc.take() {
            d.end_array();
            d.end_object();
            let json = d.into_string();
            f.put_byte(QEMU_VM_VMDESCRIPTION);
            f.put_be32(json.len() as u32);
            f.put_buffer(json.as_bytes());
        }
        f.fflush()
    }

    /// `qemu_savevm_state_cleanup()`.
    pub fn save_cleanup(&mut self) {
        self.vmdesc = None;
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.save_cleanup();
            }
        }
    }

    /// `qemu_savevm_state()`: a whole stream in one go, with the guest stopped, as `savevm`
    /// and the `file:` channel use it when the guest is not running.
    pub fn save_state(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        let ret = (|| {
            self.save_header(f)?;
            self.save_setup(f)?;
            self.save_complete(f)
        })();
        self.save_cleanup();
        ret
    }

    /// `qemu_loadvm_state()`: reads a whole stream into the entries.
    pub fn load_state(&mut self, f: &mut StreamReader<'_>) -> Result<LoadInfo> {
        for e in &mut self.entries {
            e.load_section_id = None;
        }
        let ret = self.load_inner(f);
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.load_cleanup();
            }
        }
        ret
    }

    fn load_inner(&mut self, f: &mut StreamReader<'_>) -> Result<LoadInfo> {
        self.load_header(f)?;
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                if l.is_active() {
                    let idstr = e.idstr.clone();
                    l.load_setup().map_err(|err| {
                        err.prepend(format!("Load state of device {idstr} failed: "))
                    })?;
                }
            }
        }
        let mut info = LoadInfo::default();
        self.load_main(f, &mut info)?;
        let ret = f.get_error();
        if ret < 0 {
            bail!("Error while loading vmstate: stream error: {}", ret);
        }
        // Read the vmdesc too, so the source is not left writing into a closed channel.
        if self.send_vmdesc {
            let section_type = f.get_byte();
            if f.get_error() < 0 {
                // The source may suppress it, or close right away: nothing is lost.
                warn_report("Expected vmdescription section, but the stream ended");
            } else if section_type != QEMU_VM_VMDESCRIPTION {
                warn_report(&format!("Expected vmdescription section, but got {section_type}"));
            } else {
                let size = f.get_be32() as usize;
                let mut buf = vec![0; size];
                let got = f.get_buffer(&mut buf);
                buf.truncate(got);
                info.vmdesc = Some(String::from_utf8_lossy(&buf).into_owned());
            }
        }
        Ok(info)
    }

    /// `qemu_loadvm_state_header()`.
    fn load_header(&mut self, f: &mut StreamReader<'_>) -> Result<()> {
        let v = f.get_be32();
        if v != QEMU_VM_FILE_MAGIC {
            bail!("Not a migration stream, magic: {:x} != {:x}", v, QEMU_VM_FILE_MAGIC);
        }
        let v = f.get_be32();
        if v == QEMU_VM_FILE_VERSION_COMPAT {
            bail!("SaveVM v2 format is obsolete and no longer supported");
        }
        if v != QEMU_VM_FILE_VERSION {
            bail!(
                "Unsupported migration stream version, file version {:x} != {:x}",
                v,
                QEMU_VM_FILE_VERSION
            );
        }
        if self.send_configuration {
            let v = f.get_byte();
            if v != QEMU_VM_CONFIGURATION {
                bail!("Configuration section missing, {:x} != {:x}", v, QEMU_VM_CONFIGURATION);
            }
            let mut conf = Configuration::for_load(&self.config, self.validatable_caps());
            vmstate_load_state(f, &VMSTATE_CONFIGURATION, &mut conf, 0)?;
        }
        Ok(())
    }

    /// `qemu_loadvm_state_main()`.
    fn load_main(&mut self, f: &mut StreamReader<'_>, info: &mut LoadInfo) -> Result<()> {
        loop {
            let section_type = f.get_byte();
            let ret = f.get_error();
            if ret != 0 {
                bail!("Failed to load section ID: stream error: {}", ret);
            }
            match section_type {
                QEMU_VM_SECTION_START | QEMU_VM_SECTION_FULL => self.load_start_full(f)?,
                QEMU_VM_SECTION_PART | QEMU_VM_SECTION_END => self.load_part_end(f)?,
                QEMU_VM_COMMAND => self.load_command(f, info)?,
                QEMU_VM_EOF => return Ok(()),
                t => bail!("Unknown section type {}", t),
            }
        }
    }

    /// `qemu_loadvm_section_start_full()`.
    fn load_start_full(&mut self, f: &mut StreamReader<'_>) -> Result<()> {
        let section_id = f.get_be32();
        let len = usize::from(f.get_byte());
        let mut id = vec![0; len];
        let got = f.get_buffer(&mut id);
        if got != len {
            bail!("Unable to read ID string for section {}", section_id);
        }
        let idstr = String::from_utf8_lossy(&id).into_owned();
        let instance_id = f.get_be32();
        let version_id = f.get_be32() as i32;
        let ret = f.get_error();
        if ret != 0 {
            bail!("Failed to read instance/version ID: {}", ret);
        }
        let Some(i) =
            self.entries.iter().position(|e| e.idstr == idstr && e.instance_id == instance_id)
        else {
            bail!(
                "Unknown section or instance '{}' {}. Make sure that your current VM setup \
                 matches your saved VM setup, including any hotplugged devices",
                idstr,
                instance_id
            );
        };
        let e = &mut self.entries[i];
        if version_id > e.version_id {
            bail!("unsupported version {} for '{}' v{}", version_id, idstr, e.version_id);
        }
        e.load_section_id = Some(section_id);
        Self::entry_load(e, f, version_id).map_err(|err| {
            err.prepend(format!(
                "error while loading state for instance 0x{instance_id:x} of device '{idstr}': "
            ))
        })?;
        self.check_footer(f, i, section_id)
    }

    /// `qemu_loadvm_section_part_end()`.
    fn load_part_end(&mut self, f: &mut StreamReader<'_>) -> Result<()> {
        let section_id = f.get_be32();
        let ret = f.get_error();
        if ret != 0 {
            bail!("Failed to read section ID: {}", ret);
        }
        let Some(i) = self.entries.iter().position(|e| e.load_section_id == Some(section_id))
        else {
            bail!("Unknown section {}", section_id);
        };
        // The version came with the START section; PART and END reuse it.
        let version_id = self.entries[i].version_id;
        Self::entry_load(&mut self.entries[i], f, version_id)?;
        self.check_footer(f, i, section_id)
    }

    fn entry_load(e: &mut Entry, f: &mut StreamReader<'_>, version_id: i32) -> Result<()> {
        let ret = match &mut e.kind {
            Kind::Device(d) => d.load(f, version_id),
            Kind::Live(l) => l.load(f, version_id),
        };
        ret?;
        let err = f.get_error();
        if err < 0 {
            bail!("stream error {}", err);
        }
        Ok(())
    }

    /// `check_section_footer()`.
    fn check_footer(&self, f: &mut StreamReader<'_>, i: usize, section_id: u32) -> Result<()> {
        if !self.send_section_footer {
            return Ok(());
        }
        let e = &self.entries[i];
        let mark = f.get_byte();
        let ret = f.get_error();
        let why = if ret != 0 {
            format!("Read section footer failed: {ret}")
        } else if mark != QEMU_VM_SECTION_FOOTER {
            format!("Missing section footer for {}", e.idstr)
        } else {
            let read = f.get_be32();
            if Some(read) == e.load_section_id {
                return Ok(());
            }
            format!(
                "Mismatched section id in footer for {} - read 0x{:x} expected 0x{:x}",
                e.idstr,
                read,
                e.load_section_id.unwrap_or(0)
            )
        };
        bail!("Section footer error, section_id: {}: {}", section_id, why)
    }

    /// `loadvm_process_command()`.
    fn load_command(&mut self, f: &mut StreamReader<'_>, info: &mut LoadInfo) -> Result<()> {
        let c = f.get_be16();
        let len = f.get_be16();
        let ret = f.get_error();
        if ret != 0 {
            bail!("Failed to load VM process command: stream error: {}", ret);
        }
        if c >= cmd::MAX || c == cmd::INVALID {
            bail!("MIG_CMD 0x{:x} unknown (len 0x{:x})", c, len);
        }
        let (name, want) = MIG_CMD_ARGS[usize::from(c)];
        if want != -1 && want != i32::from(len) {
            bail!("{} received with bad length - expecting {}, got {}", name, want, len);
        }
        match c {
            cmd::OPEN_RETURN_PATH => {
                // There is no return path yet. The source only asks for one with postcopy or
                // the return-path capability, and then only uses it for PING and its acks.
                info.return_path = true;
                Ok(())
            }
            cmd::PING => {
                let v = f.get_be32();
                bail!("CMD_PING (0x{:x}) received with no return path", v)
            }
            cmd::PACKAGED => {
                let size = f.get_be32() as usize;
                let mut buf = vec![0; size];
                if f.get_buffer(&mut buf) != size {
                    bail!("Unable to read packaged data: stream error {}", f.get_error());
                }
                let mut inner = StreamReader::new(&buf);
                self.load_main(&mut inner, info)
            }
            cmd::SWITCHOVER_START => {
                info.switchover_start = true;
                Ok(())
            }
            cmd::POSTCOPY_ADVISE
            | cmd::POSTCOPY_LISTEN
            | cmd::POSTCOPY_RUN
            | cmd::POSTCOPY_RAM_DISCARD
            | cmd::POSTCOPY_RESUME
            | cmd::RECV_BITMAP => bail!("{} received, but ruvm does not support postcopy", name),
            _ => bail!("MIG_CMD 0x{:x} deprecated (len 0x{:x})", c, len),
        }
    }
}

/// `should_validate_capability()`.
fn should_validate_capability(c: MigrationCapability) -> bool {
    matches!(c, MigrationCapability::XIgnoreShared | MigrationCapability::MappedRam)
}

/// A `MigrationCapability` on the wire, `vmstate_info_capability`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Capability(MigrationCapability);

struct CapabilityInfo;

impl VmStateInfo<Capability> for CapabilityInfo {
    fn name(&self) -> &'static str {
        "capability"
    }

    fn load(&self, f: &mut StreamReader<'_>, v: &mut Capability, _size: usize) -> Result<()> {
        let len = usize::from(f.get_byte());
        let mut buf = vec![0; len];
        f.get_buffer(&mut buf);
        let s = String::from_utf8_lossy(&buf);
        match MigrationCapability::from_name(&s) {
            Some(c) => {
                v.0 = c;
                Ok(())
            }
            None => bail!("Received unknown capability {}", s),
        }
    }

    fn save(&self, f: &mut StreamWriter, v: &Capability, _size: usize) -> Result<()> {
        let s = v.0.as_str();
        f.put_byte(s.len() as u8);
        f.put_buffer(s.as_bytes());
        Ok(())
    }
}

impl VmStateType for Capability {
    fn info() -> &'static dyn VmStateInfo<Capability> {
        &CapabilityInfo
    }
}

/// The `SaveState` fields the configuration section carries, next to what the local side has.
#[derive(Debug, Default)]
struct Configuration {
    len: u32,
    name: Vec<u8>,
    target_page_bits: u32,
    caps_count: u32,
    capabilities: Vec<Capability>,
    uuid: [u8; 16],
    local: Option<MachineConfig>,
    local_caps: Vec<Capability>,
    validate_uuid: bool,
}

impl Configuration {
    /// `configuration_pre_save()`.
    fn for_save(config: &MachineConfig, caps: Vec<Capability>, validate_uuid: bool) -> Self {
        Configuration {
            len: config.name.len() as u32,
            name: config.name.as_bytes().to_vec(),
            target_page_bits: config.page_bits,
            caps_count: caps.len() as u32,
            capabilities: caps.clone(),
            uuid: config.uuid.unwrap_or_default(),
            local: Some(config.clone()),
            local_caps: caps,
            validate_uuid,
        }
    }

    /// `configuration_pre_load()`: without the target-page-bits subsection the source uses the
    /// legacy page size.
    fn for_load(config: &MachineConfig, caps: Vec<Capability>) -> Self {
        Configuration {
            target_page_bits: config.legacy_page_bits,
            local: Some(config.clone()),
            local_caps: caps,
            ..Configuration::default()
        }
    }

    fn local(&self) -> &MachineConfig {
        self.local.as_ref().expect("configuration without a machine")
    }

    /// `configuration_post_load()`.
    fn post_load(&mut self) -> Result<()> {
        let local = self.local();
        let n = (self.len as usize).min(self.name.len());
        let received = &self.name[..n];
        // strncmp() over the received length, with the local name ending in a NUL.
        let local_bytes = local.name.as_bytes();
        let mut same = true;
        for (i, &r) in received.iter().enumerate() {
            let l = local_bytes.get(i).copied().unwrap_or(0);
            if l != r {
                same = false;
                break;
            }
            if l == 0 {
                break;
            }
        }
        if !same {
            bail!(
                "Machine type received is '{}' and local is '{}'",
                String::from_utf8_lossy(received),
                local.name
            );
        }
        if self.target_page_bits != local.page_bits {
            bail!(
                "Received TARGET_PAGE_BITS is {} but local is {}",
                self.target_page_bits,
                local.page_bits
            );
        }
        let mut ok = true;
        for c in MigrationCapability::ALL.iter().copied().filter(|c| should_validate_capability(*c))
        {
            let source = self.capabilities.iter().any(|x| x.0 == c);
            let target = self.local_caps.iter().any(|x| x.0 == c);
            if source != target {
                ruvm_base::error_report(&format!(
                    "Capability {} is {}, but received capability is {}",
                    c.as_str(),
                    if target { "on" } else { "off" },
                    if source { "on" } else { "off" }
                ));
                ok = false;
            }
        }
        if !ok {
            bail!("Failed to validate capabilities");
        }
        Ok(())
    }

    /// `vmstate_uuid_post_load()`.
    fn uuid_post_load(&mut self) -> Result<()> {
        match self.local().uuid {
            None => {
                warn_report(&format!(
                    "UUID is received {}, but local uuid isn't set",
                    uuid_unparse(&self.uuid)
                ));
                Ok(())
            }
            Some(u) if u != self.uuid => bail!(
                "UUID received is {} and local is {}",
                uuid_unparse(&self.uuid),
                uuid_unparse(&u)
            ),
            Some(_) => Ok(()),
        }
    }
}

/// `qemu_uuid_unparse()`.
fn uuid_unparse(u: &[u8; 16]) -> String {
    let h: Vec<String> = u.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        h[0..4].concat(),
        h[4..6].concat(),
        h[6..8].concat(),
        h[8..10].concat(),
        h[10..16].concat()
    )
}

static VMSTATE_TARGET_PAGE_BITS: LazyLock<VmStateDescription<Configuration>> =
    LazyLock::new(|| {
        VmStateDescription::new("configuration/target-page-bits")
            .version_id(1)
            .minimum_version_id(1)
            .needed(|c: &Configuration| c.local().page_bits > c.local().legacy_page_bits)
            .field(VmStateField::scalar("target_page_bits", |c: &mut Configuration| {
                &mut c.target_page_bits
            }))
    });

static VMSTATE_CAPABILITIES: LazyLock<VmStateDescription<Configuration>> = LazyLock::new(|| {
    VmStateDescription::new("configuration/capabilities")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c: &Configuration| !c.local_caps.is_empty())
        .field(
            VmStateField::scalar("caps_count", |c: &mut Configuration| &mut c.caps_count)
                .version(1),
        )
        .field(
            VmStateField::varray_alloc(
                "capabilities",
                |c: &Configuration| c.caps_count as usize,
                |c: &mut Configuration| &mut c.capabilities,
            )
            .version(1),
        )
});

static VMSTATE_UUID: LazyLock<VmStateDescription<Configuration>> = LazyLock::new(|| {
    VmStateDescription::new("configuration/uuid")
        .version_id(1)
        .minimum_version_id(1)
        .needed(|c: &Configuration| c.validate_uuid && c.local().uuid.is_some())
        .post_load_errp(|c: &mut Configuration, _| c.uuid_post_load())
        .field(VmStateField::array("uuid.data", |c: &mut Configuration| &mut c.uuid).version(1))
});

static VMSTATE_CONFIGURATION: LazyLock<VmStateDescription<Configuration>> = LazyLock::new(|| {
    VmStateDescription::new("configuration")
        .version_id(1)
        .post_load_errp(|c: &mut Configuration, _| c.post_load())
        .field(VmStateField::scalar("len", |c: &mut Configuration| &mut c.len))
        .field(VmStateField::vbuffer_alloc(
            "name",
            |c: &Configuration| c.len as usize,
            |c: &mut Configuration| &mut c.name,
        ))
        .subsection(&VMSTATE_TARGET_PAGE_BITS)
        .subsection(&VMSTATE_CAPABILITIES)
        .subsection(&VMSTATE_UUID)
});

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default, Clone, PartialEq)]
    struct Dev {
        a: u32,
        b: u64,
    }

    static VMSTATE_DEV: LazyLock<VmStateDescription<Dev>> = LazyLock::new(|| {
        VmStateDescription::new("dev")
            .version_id(2)
            .minimum_version_id(1)
            .field(VmStateField::scalar("a", |d: &mut Dev| &mut d.a))
            .field(VmStateField::scalar("b", |d: &mut Dev| &mut d.b).version(2))
    });

    fn config() -> MachineConfig {
        MachineConfig {
            name: "pc-q35-11.1".into(),
            page_bits: 12,
            legacy_page_bits: 12,
            uuid: None,
        }
    }

    fn machine(dev: Arc<Mutex<Dev>>) -> SaveVm {
        let mut s = SaveVm::new(config());
        let (g, p) = (dev.clone(), dev);
        s.register_vmsd(
            "",
            None,
            &VMSTATE_DEV,
            move || Ok(g.lock().unwrap().clone()),
            move |d| {
                *p.lock().unwrap() = d;
                Ok(())
            },
        );
        s
    }

    #[test]
    fn device_round_trip() {
        let src = Arc::new(Mutex::new(Dev { a: 7, b: 0x1122_3344_5566_7788 }));
        let mut out = Vec::new();
        machine(src).save_state(&mut QemuFile::new(&mut out)).unwrap();

        // The header and configuration section, byte for byte what QEMU writes.
        let mut want = vec![0x51, 0x45, 0x56, 0x4d, 0, 0, 0, 3, 7, 0, 0, 0, 11];
        want.extend_from_slice(b"pc-q35-11.1");
        // No subsection, then the device in a FULL section.
        want.extend_from_slice(&[0x04, 0, 0, 0, 0, 3, b'd', b'e', b'v', 0, 0, 0, 0, 0, 0, 0, 2]);
        want.extend_from_slice(&[0, 0, 0, 7, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88]);
        want.extend_from_slice(&[0x7e, 0, 0, 0, 0, 0x00, 0x06]);
        assert_eq!(&out[..want.len()], &want[..]);
        let json = std::str::from_utf8(&out[want.len() + 4..]).unwrap();
        assert!(json.starts_with(
            r#"{"configuration": {"vmsd_name": "configuration", "version": 1, "fields": [{"name": "len""#
        ));
        assert!(json.contains(
            r#""page_size": 4096, "devices": [{"name": "dev", "instance_id": 0, "vmsd_name": "dev""#
        ));

        let dst = Arc::new(Mutex::new(Dev::default()));
        let info = machine(dst.clone()).load_state(&mut StreamReader::new(&out)).unwrap();
        assert_eq!(*dst.lock().unwrap(), Dev { a: 7, b: 0x1122_3344_5566_7788 });
        assert_eq!(info.vmdesc.as_deref(), Some(json));
    }

    #[test]
    fn load_errors_match_qemu() {
        let src = Arc::new(Mutex::new(Dev::default()));
        let mut out = Vec::new();
        machine(src).save_state(&mut QemuFile::new(&mut out)).unwrap();
        let dst = || machine(Arc::new(Mutex::new(Dev::default())));

        let mut bad = out.clone();
        bad[0] = 0;
        let e = dst().load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(e.message(), "Not a migration stream, magic: 45564d != 5145564d");

        let mut bad = out.clone();
        bad[7] = 2;
        let e = dst().load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(e.message(), "SaveVM v2 format is obsolete and no longer supported");

        let mut other = SaveVm::new(MachineConfig { name: "pc-q35-10.0".into(), ..config() });
        let e = other.load_state(&mut StreamReader::new(&out)).unwrap_err();
        assert!(
            e.message()
                .contains("Machine type received is 'pc-q35-11.1' and local is 'pc-q35-10.0'"),
            "{}",
            e.message()
        );

        let mut empty = SaveVm::new(config());
        let e = empty.load_state(&mut StreamReader::new(&out)).unwrap_err();
        assert_eq!(
            e.message(),
            "Unknown section or instance 'dev' 0. Make sure that your current VM setup matches \
             your saved VM setup, including any hotplugged devices"
        );

        // A broken footer.
        let at = 13 + 11 + 17 + 12;
        let mut bad = out.clone();
        bad[at] = 0x7f;
        let e = dst().load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(
            e.message(),
            "Section footer error, section_id: 0: Missing section footer for dev"
        );
    }

    #[test]
    fn commands() {
        let mut s = SaveVm::new(config());
        let mut w = StreamWriter::new();
        let mut conf = Configuration::for_save(&config(), Vec::new(), false);
        w.put_be32(QEMU_VM_FILE_MAGIC);
        w.put_be32(QEMU_VM_FILE_VERSION);
        w.put_byte(QEMU_VM_CONFIGURATION);
        VMSTATE_CONFIGURATION.save(&mut w, &mut conf).unwrap();
        let head = w.into_inner();

        let mut ok = head.clone();
        ok.extend_from_slice(&[0x08, 0, 11, 0, 0, 0x00]);
        let info = s.load_state(&mut StreamReader::new(&ok)).unwrap();
        assert!(info.switchover_start);

        let mut bad = head.clone();
        bad.extend_from_slice(&[0x08, 0, 11, 0, 1]);
        let e = s.load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(e.message(), "SWITCHOVER_START received with bad length - expecting 0, got 1");

        let mut bad = head.clone();
        bad.extend_from_slice(&[0x08, 0, 99, 0, 0]);
        let e = s.load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(e.message(), "MIG_CMD 0x63 unknown (len 0x0)");

        let mut bad = head;
        bad.push(0x42);
        let e = s.load_state(&mut StreamReader::new(&bad)).unwrap_err();
        assert_eq!(e.message(), "Unknown section type 66");
    }

    #[test]
    fn priorities_and_instance_ids() {
        struct Nop;
        impl DeviceState for Nop {
            fn save(&mut self, _: &mut StreamWriter, _: Option<&mut JsonWriter>) -> Result<()> {
                Ok(())
            }
            fn load(&mut self, _: &mut StreamReader<'_>, _: i32) -> Result<()> {
                Ok(())
            }
        }
        let mut s = SaveVm::new(config());
        assert_eq!(s.register_device(EntryInfo::new("timer", 2), Nop), 0);
        assert_eq!(s.register_device(EntryInfo::new("i8259", 1), Nop), 0);
        assert_eq!(s.register_device(EntryInfo::new("i8259", 1), Nop), 1);
        s.register_device(EntryInfo::new("apic", 3).priority(MigPriority::Apic), Nop);
        let names: Vec<_> = s.entries().map(|(n, i, _)| format!("{n}.{i}")).collect();
        assert_eq!(names, ["apic.0", "timer.0", "i8259.0", "i8259.1"]);
    }
}
