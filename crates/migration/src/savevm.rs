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
//!
//! Postcopy adds commands to the stream. The source advises the destination early, and at the
//! switch sends the pages to drop, then the device state as one `MIG_CMD_PACKAGED` blob that
//! ends in `MIG_CMD_POSTCOPY_LISTEN` and `MIG_CMD_POSTCOPY_RUN`. On `LISTEN` the destination
//! hands the rest of the main stream, which carries RAM only from then on, to a listen thread,
//! loads the device state from the package and starts the guest on `RUN`.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

use ruvm_base::{Error, Result, bail, error_report, warn_report};
use ruvm_qapi::types::{MigrationCapability, MigrationStatus};
use ruvm_vmstate::info::{VmStateInfo, VmStateType};
use ruvm_vmstate::{
    JsonWriter, MigPriority, StreamReader, StreamWriter, VmStateDescription, VmStateField,
    vmstate_load_state, vmstate_save_state_vmdesc,
};

use ruvm_mem::RamBlock;
use ruvm_qapi::types::ZeroPageDetection;

use crate::channel::{FileChannel, Socket};
use crate::multifd::{MultifdRecv, MultifdSend};
use crate::postcopy::{self, MAX_PACKAGED_SIZE, PING_PACKAGED_LOADED, PageRequests, ReturnPath};

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
    // The seekable side of a `file:` channel.
    file: Option<Arc<FileChannel>>,
    // What went out with `put_buffer_at()`, which counts as transferred too.
    at_bytes: u64,
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
        QemuFile { buf: StreamWriter::new(), sink: Box::new(sink), file: None, at_bytes: 0 }
    }

    /// [`new`](Self::new) over a `file:` channel, which can seek.
    pub fn with_file(sink: impl Write + Send + 'a, file: Option<Arc<FileChannel>>) -> Self {
        QemuFile { file, ..Self::new(sink) }
    }

    /// The seekable side of the channel, for a file.
    pub fn file(&self) -> Option<&Arc<FileChannel>> {
        self.file.as_ref()
    }

    fn seekable(&self) -> Result<&FileChannel> {
        match &self.file {
            Some(f) => Ok(f),
            None => bail!("The migration channel cannot seek"),
        }
    }

    /// `qemu_get_offset()`: flushes, then the position of the stream in the file.
    pub fn offset(&mut self) -> Result<u64> {
        self.fflush()?;
        let ret = self.seekable()?.offset();
        if ret.is_err() {
            self.buf.set_error(-ruvm_vmstate::EIO);
        }
        ret
    }

    /// `qemu_set_offset(SEEK_SET)`: flushes, then moves the stream to `off` in the file.
    pub fn set_offset(&mut self, off: u64) -> Result<()> {
        self.fflush()?;
        let ret = self.seekable()?.set_offset(off);
        if ret.is_err() {
            self.buf.set_error(-ruvm_vmstate::EIO);
        }
        ret
    }

    /// `qemu_put_buffer_at()`: flushes, then writes `buf` at `pos` in the file without moving
    /// the stream.
    pub fn put_buffer_at(&mut self, buf: &[u8], pos: u64) -> Result<()> {
        self.fflush()?;
        let ret = self.seekable()?.write_at(buf, pos);
        match ret {
            Ok(()) => self.at_bytes += buf.len() as u64,
            Err(_) => self.buf.set_error(-ruvm_vmstate::EIO),
        }
        ret
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
        self.buf.transferred() + self.at_bytes
    }

    /// `mig_stats.qemu_file_transferred`: what reached the channel, without what is still
    /// buffered.
    pub fn flushed(&self) -> u64 {
        self.transferred() - self.buf.as_bytes().len() as u64
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

    /// `has_postcopy`: whether the entry keeps going after the switch to postcopy. Only RAM
    /// does; everything else completes when the guest stops.
    fn has_postcopy(&self) -> bool {
        false
    }

    /// `save_postcopy_prepare`: a last `QEMU_VM_SECTION_PART` payload before the guest stops for
    /// postcopy.
    fn save_postcopy_prepare(&mut self, _f: &mut QemuFile<'_>) -> Result<()> {
        Ok(())
    }

    /// For `ram_postcopy_send_discard_bitmap()`, with the guest stopped: syncs a last time and
    /// returns, per RAM block, the byte ranges still to send, which the destination has to drop
    /// because it may hold stale copies of them.
    fn postcopy_discard_ranges(&mut self) -> Vec<(String, Vec<(u64, u64)>)> {
        Vec::new()
    }

    /// The queue the return path puts page requests into, for an entry that serves them.
    fn page_requests(&self) -> Option<Arc<PageRequests>> {
        None
    }

    /// Whether the entry has a RAM block with this name, `qemu_ram_block_by_name()`.
    fn has_ram_block(&self, _name: &str) -> bool {
        false
    }

    /// Destination, on `MIG_CMD_POSTCOPY_ADVISE`: `ram_postcopy_incoming_init()`.
    fn postcopy_advise(&mut self) -> Result<()> {
        Ok(())
    }

    /// Destination, on `MIG_CMD_POSTCOPY_RAM_DISCARD`: `ram_discard_range()`. `None` when the
    /// entry has no such block.
    fn postcopy_discard(&mut self, _name: &str, _start: u64, _len: u64) -> Option<Result<()>> {
        None
    }

    /// Destination, on `MIG_CMD_POSTCOPY_LISTEN`: `postcopy_ram_incoming_setup()`, which
    /// registers RAM for page faults and starts asking for missing pages on `rp`.
    fn postcopy_listen(&mut self, _rp: Option<&Arc<ReturnPath>>) -> Result<()> {
        Ok(())
    }

    /// Destination: `postcopy_ram_incoming_cleanup()`, once the listen thread is done.
    fn postcopy_end(&mut self) {}

    /// Source, before `save_setup`: how pages go out.
    fn set_save_params(&mut self, _p: &SaveParams) {}

    /// Destination, around a load: where pages come in besides the main stream.
    fn set_load_params(&mut self, _p: &LoadParams) {}

    /// Source, with the `background-snapshot` capability once the guest is stopped and the
    /// device state saved: `ram_write_tracking_start()`, which write-protects the RAM still to
    /// send.
    fn write_tracking_start(&mut self) -> Result<()> {
        Ok(())
    }

    /// `ram_write_tracking_stop()`: lifts the protection again.
    fn write_tracking_stop(&mut self) {}

    /// The RAM blocks of the entry, which multifd packets name.
    fn ram_blocks(&self) -> Vec<Arc<RamBlock>> {
        Vec::new()
    }

    /// Prefixes of block names the entry reads and drops when it has no such block.
    fn droppable_blocks(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The migration parameters that change how live state is sent.
#[derive(Clone)]
pub struct SaveParams {
    /// The multifd channels, with the `multifd` capability.
    pub multifd: Option<Arc<MultifdSend>>,
    /// `zero-page-detection`.
    pub zero_page_detection: ZeroPageDetection,
    /// With the `xbzrle` capability, `xbzrle-cache-size`, which `migrate-set-parameters` can
    /// change while the migration runs.
    pub xbzrle_cache_size: Option<Arc<AtomicU64>>,
    /// The `mapped-ram` capability: each page goes to a fixed place in the file.
    pub mapped_ram: bool,
    /// The `background-snapshot` capability: RAM goes out in one pass behind write protection
    /// instead of dirty logging.
    pub background_snapshot: bool,
    /// `migrate_ram_is_ignored()` for every block, which `cpr-transfer` makes so: the next
    /// process maps the very same memory, so the block list goes out but no page does.
    pub ignore_ram: bool,
}

impl std::fmt::Debug for SaveParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SaveParams")
            .field("multifd", &self.multifd)
            .field("zero_page_detection", &self.zero_page_detection)
            .field("xbzrle_cache_size", &self.xbzrle_cache_size)
            .field("mapped_ram", &self.mapped_ram)
            .field("background_snapshot", &self.background_snapshot)
            .field("ignore_ram", &self.ignore_ram)
            .finish()
    }
}

/// What the incoming migration state learns while a stream loads.
pub trait IncomingHooks: Sync {
    /// `migrate_set_state()` on the incoming state, for the postcopy states.
    fn set_state(&self, _status: MigrationStatus) {}

    /// `loadvm_postcopy_handle_run_bh()`: the device state is in, start the guest (or leave it
    /// paused without autostart).
    fn postcopy_run(&self) {}
}

/// How [`SaveVm::load_state_with`] loads a stream.
#[derive(Default)]
pub struct LoadOptions<'a> {
    /// The channel the stream comes in on, when it is a socket, for the return path.
    pub socket: Option<Socket>,
    /// The incoming migration state.
    pub hooks: Option<&'a dyn IncomingHooks>,
    /// The multifd channels, with the `multifd` capability.
    pub multifd: Option<Arc<MultifdRecv>>,
    /// The file the stream comes in from, when it is one, for `mapped-ram`.
    pub file: Option<Arc<FileChannel>>,
}

impl std::fmt::Debug for LoadOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadOptions")
            .field("socket", &self.socket)
            .field("hooks", &self.hooks.is_some())
            .field("multifd", &self.multifd)
            .field("file", &self.file)
            .finish()
    }
}

/// What a live entry needs to know while a stream loads.
#[derive(Debug, Default, Clone)]
pub struct LoadParams {
    /// The multifd channels, with the `multifd` capability.
    pub multifd: Option<Arc<MultifdRecv>>,
    /// The file the stream comes in from, when it is one.
    pub file: Option<Arc<FileChannel>>,
    /// The `mapped-ram` capability: the pages sit at fixed places in the file.
    pub mapped_ram: bool,
}

/// `PostcopyState`, the destination's progress through the postcopy commands.
mod ps {
    pub(super) const NONE: u8 = 0;
    pub(super) const ADVISE: u8 = 1;
    pub(super) const DISCARD: u8 = 2;
    pub(super) const LISTENING: u8 = 3;
    pub(super) const RUNNING: u8 = 4;
    pub(super) const END: u8 = 5;
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `qemu_savevm_command_send()` into a buffer.
fn command(f: &mut StreamWriter, c: u16, data: &[u8]) {
    f.put_byte(QEMU_VM_COMMAND);
    f.put_be16(c);
    f.put_be16(data.len() as u16);
    f.put_buffer(data);
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
    /// Whether the migration finished in postcopy: the guest was started on
    /// `MIG_CMD_POSTCOPY_RUN` and RAM came in after that.
    pub postcopy: bool,
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
    /// `migrate_send_switchover_start()`, on for every current machine type.
    pub send_switchover_start: bool,
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
            send_switchover_start: true,
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

    fn live_active(&self, i: usize) -> Option<bool> {
        match &self.entries[i].kind {
            Kind::Live(l) if l.is_active() => Some(l.has_postcopy()),
            _ => None,
        }
    }

    /// `qemu_savevm_state_iterate()`: one `QEMU_VM_SECTION_PART` per live entry, stopping at
    /// the first that has more to send. In postcopy only the entries that go on in postcopy
    /// take part. Returns true when all of them are done.
    pub fn save_iterate(
        &mut self,
        f: &mut QemuFile<'_>,
        max_bytes: u64,
        in_postcopy: bool,
    ) -> Result<bool> {
        for i in 0..self.entries.len() {
            match self.live_active(i) {
                None => continue,
                Some(false) if in_postcopy => continue,
                Some(_) => {}
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

    /// `ram_write_tracking_start()` for every live entry.
    pub fn write_tracking_start(&mut self) -> Result<()> {
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.write_tracking_start()?;
            }
        }
        Ok(())
    }

    /// `ram_write_tracking_stop()` for every live entry.
    pub fn write_tracking_stop(&mut self) {
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.write_tracking_stop();
            }
        }
    }

    /// The bytes the live entries still have to send.
    pub fn pending(&mut self, exact: bool) -> u64 {
        let (pre, post) = self.pending_split(exact);
        pre + post
    }

    /// `qemu_savevm_state_pending_estimate()` and `qemu_savevm_state_pending_exact()`: the bytes
    /// still to send, as those that must go before the guest stops and those that can follow
    /// in postcopy. With `postcopy-ram` all of RAM can.
    pub fn pending_split(&mut self, exact: bool) -> (u64, u64) {
        let postcopy_ram = self.postcopy_ram();
        let (mut pre, mut post) = (0, 0);
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                if l.is_active() {
                    let p = l.pending(exact);
                    if postcopy_ram && l.has_postcopy() {
                        post += p;
                    } else {
                        pre += p;
                    }
                }
            }
        }
        (pre, post)
    }

    /// Whether the `postcopy-ram` capability is on.
    pub fn postcopy_ram(&self) -> bool {
        self.capabilities.contains(&MigrationCapability::PostcopyRam)
    }

    /// The page request queue of the entry that serves them, RAM.
    pub fn page_requests(&self) -> Option<Arc<PageRequests>> {
        self.entries.iter().find_map(|e| match &e.kind {
            Kind::Live(l) => l.page_requests(),
            Kind::Device(_) => None,
        })
    }

    fn live(&self) -> impl Iterator<Item = &dyn LiveState> {
        self.entries.iter().filter_map(|e| match &e.kind {
            Kind::Live(l) => Some(&**l),
            Kind::Device(_) => None,
        })
    }

    /// Tells every live entry how to send, before [`save_setup`](Self::save_setup).
    pub fn set_save_params(&mut self, p: &SaveParams) {
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.set_save_params(p);
            }
        }
    }

    /// The RAM blocks of every live entry.
    pub fn ram_blocks(&self) -> Vec<Arc<RamBlock>> {
        self.live().flat_map(|l| l.ram_blocks()).collect()
    }

    /// The block name prefixes the live entries drop when they lack the block.
    pub fn droppable_blocks(&self) -> Vec<String> {
        self.live().flat_map(|l| l.droppable_blocks()).collect()
    }

    fn send_command(f: &mut QemuFile<'_>, c: u16, data: &[u8]) -> Result<()> {
        command(f, c, data);
        f.fflush()
    }

    /// `qemu_savevm_send_open_return_path()`.
    pub fn send_open_return_path(&self, f: &mut QemuFile<'_>) -> Result<()> {
        Self::send_command(f, cmd::OPEN_RETURN_PATH, &[])
    }

    /// `qemu_savevm_send_ping()`.
    pub fn send_ping(&self, f: &mut QemuFile<'_>, value: u32) -> Result<()> {
        Self::send_command(f, cmd::PING, &value.to_be_bytes())
    }

    /// `qemu_savevm_send_postcopy_advise()`: with `postcopy-ram` it carries the host and the
    /// target page size.
    pub fn send_postcopy_advise(&self, f: &mut QemuFile<'_>) -> Result<()> {
        let mut data = Vec::new();
        if self.postcopy_ram() {
            data.extend_from_slice(&HOST_PAGE_SIZE.to_be_bytes());
            data.extend_from_slice(&(1u64 << self.config.page_bits).to_be_bytes());
        }
        Self::send_command(f, cmd::POSTCOPY_ADVISE, &data)
    }

    /// `qemu_savevm_maybe_send_switchover_start()`.
    pub fn send_switchover_start(&self, f: &mut QemuFile<'_>) -> Result<()> {
        if self.send_switchover_start {
            Self::send_command(f, cmd::SWITCHOVER_START, &[])?;
        }
        Ok(())
    }

    /// `qemu_savevm_state_postcopy_prepare()`, which also drops the vmdesc: a postcopy stream
    /// has none.
    pub fn postcopy_prepare(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        self.vmdesc = None;
        for i in 0..self.entries.len() {
            if self.live_active(i) != Some(true) {
                continue;
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_PART);
            if let Kind::Live(l) = &mut self.entries[i].kind {
                l.save_postcopy_prepare(f)?;
            }
            self.section_footer(f, &self.entries[i]);
        }
        f.fflush()
    }

    /// `qemu_savevm_state_complete_precopy_iterable()`: the `QEMU_VM_SECTION_END` of each live
    /// entry, or in postcopy of each that does not go on in postcopy.
    pub fn complete_precopy_iterable(
        &mut self,
        f: &mut QemuFile<'_>,
        in_postcopy: bool,
    ) -> Result<()> {
        for i in 0..self.entries.len() {
            match self.live_active(i) {
                None => continue,
                Some(true) if in_postcopy => continue,
                Some(_) => {}
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_END);
            if let Kind::Live(l) = &mut self.entries[i].kind {
                l.save_complete(f)
                    .map_err(|e| e.prepend("Failed to save iterable device state: "))?;
            }
            self.section_footer(f, &self.entries[i]);
            f.fflush()?;
        }
        Ok(())
    }

    /// `qemu_savevm_state_complete_precopy()`: the `QEMU_VM_SECTION_END` of each live entry, the
    /// device sections, `QEMU_VM_EOF` and the vmdesc. The guest must be stopped.
    pub fn save_complete(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        self.complete_precopy_iterable(f, false)?;
        self.save_non_iterable(f)
    }

    /// `qemu_savevm_state_non_iterable()` and `qemu_savevm_state_end_precopy()`: the device
    /// sections, `QEMU_VM_EOF` and the vmdesc. A background snapshot keeps them in a buffer
    /// while RAM goes out.
    pub fn save_non_iterable(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
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

    /// `ram_postcopy_send_discard_bitmap()`: the pages still to send, as
    /// `MIG_CMD_POSTCOPY_RAM_DISCARD` commands of at most `MAX_DISCARDS_PER_COMMAND` ranges.
    pub fn send_postcopy_discard(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        const MAX_DISCARDS_PER_COMMAND: usize = 12;
        for i in 0..self.entries.len() {
            if self.live_active(i) != Some(true) {
                continue;
            }
            let Kind::Live(l) = &mut self.entries[i].kind else { continue };
            for (name, ranges) in l.postcopy_discard_ranges() {
                for chunk in ranges.chunks(MAX_DISCARDS_PER_COMMAND) {
                    // qemu_savevm_send_postcopy_ram_discard(): version 0, the counted name with
                    // a NUL after it, then start and length pairs.
                    let mut data = vec![0u8, name.len() as u8];
                    data.extend_from_slice(name.as_bytes());
                    data.push(0);
                    for (start, len) in chunk {
                        data.extend_from_slice(&start.to_be_bytes());
                        data.extend_from_slice(&len.to_be_bytes());
                    }
                    Self::send_command(f, cmd::POSTCOPY_RAM_DISCARD, &data)?;
                }
            }
        }
        Ok(())
    }

    /// The package `postcopy_start()` builds: `MIG_CMD_POSTCOPY_LISTEN`, the device state, the
    /// pings that tell the source how far the destination got, and `MIG_CMD_POSTCOPY_RUN`.
    pub fn postcopy_package(&mut self, has_return_path: bool) -> Result<Vec<u8>> {
        let mut fb = StreamWriter::new();
        command(&mut fb, cmd::POSTCOPY_LISTEN, &[]);
        // qemu_savevm_state_non_iterable()
        let footer = self.send_section_footer;
        for e in self.entries.iter_mut().filter(|e| !e.early_setup) {
            Self::vmstate_save(&mut fb, e, footer, None)
                .map_err(|err| err.prepend("Postcopy save non-iterable states failed: "))?;
        }
        if self.postcopy_ram() {
            command(&mut fb, cmd::PING, &3u32.to_be_bytes());
        }
        if has_return_path {
            command(&mut fb, cmd::PING, &PING_PACKAGED_LOADED.to_be_bytes());
        }
        command(&mut fb, cmd::POSTCOPY_RUN, &[]);
        if fb.get_error() != 0 {
            bail!("postcopy_start: Migration stream errored (pre package)");
        }
        Ok(fb.into_inner())
    }

    /// `qemu_savevm_send_packaged()`.
    pub fn send_packaged(&self, f: &mut QemuFile<'_>, data: &[u8]) -> Result<()> {
        if data.len() > MAX_PACKAGED_SIZE {
            error_report(&format!(
                "qemu_savevm_send_packaged: Unreasonably large packaged state: {}",
                data.len()
            ));
            bail!("postcopy_start: Failed to send packaged data");
        }
        f.put_byte(QEMU_VM_COMMAND);
        f.put_be16(cmd::PACKAGED);
        f.put_be16(4);
        f.put_be32(data.len() as u32);
        f.put_buffer(data);
        f.fflush()
    }

    /// `qemu_savevm_state_complete_postcopy()`: the `QEMU_VM_SECTION_END` of the entries that
    /// went on in postcopy, then `QEMU_VM_EOF`, with no vmdesc after it.
    pub fn complete_postcopy(&mut self, f: &mut QemuFile<'_>) -> Result<()> {
        for i in 0..self.entries.len() {
            if self.live_active(i) != Some(true) {
                continue;
            }
            self.section_header(f, &self.entries[i], QEMU_VM_SECTION_END);
            if let Kind::Live(l) = &mut self.entries[i].kind {
                l.save_complete(f)?;
            }
            self.section_footer(f, &self.entries[i]);
            f.fflush()?;
        }
        f.put_byte(QEMU_VM_EOF);
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
        self.load_state_with(f, LoadOptions::default())
    }

    /// [`load_state`](Self::load_state) for an incoming migration, which may open a return
    /// path on the socket and go into postcopy.
    pub fn load_state_with(
        &mut self,
        f: &mut StreamReader<'_>,
        opts: LoadOptions<'_>,
    ) -> Result<LoadInfo> {
        for e in &mut self.entries {
            e.load_section_id = None;
        }
        let ctx = LoadCtx {
            footer: self.send_section_footer,
            page_bits: self.config.page_bits,
            postcopy_ram: self.postcopy_ram(),
            socket: Mutex::new(opts.socket),
            rp: Mutex::new(None),
            hooks: opts.hooks,
            ps: AtomicU8::new(ps::NONE),
            device: AtomicBool::new(false),
            info: Mutex::new(LoadInfo::default()),
        };
        let params = LoadParams {
            multifd: opts.multifd,
            file: opts.file,
            mapped_ram: self.capabilities.contains(&MigrationCapability::MappedRam),
        };
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.set_load_params(&params);
            }
        }
        let ret = self.load_inner(f, &ctx);
        // migration_incoming_state_destroy(): tell the source the return path is done.
        if let Some(rp) = ctx.rp() {
            let _ = rp.shut(u32::from(f.get_error() != 0));
        }
        for e in &mut self.entries {
            if let Kind::Live(l) = &mut e.kind {
                l.postcopy_end();
                l.load_cleanup();
                l.set_load_params(&LoadParams::default());
            }
        }
        ret.map(|()| ctx.info.into_inner().unwrap_or_else(|e| e.into_inner()))
    }

    fn load_inner(&mut self, f: &mut StreamReader<'_>, ctx: &LoadCtx<'_>) -> Result<()> {
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
        let flow = Loader { entries: self.entries.iter_mut().collect(), ctx }.load_main(f)?;
        match flow {
            Flow::Eof => {}
            // The listen thread took the rest of the stream and is done with it.
            Flow::Quit => return Ok(()),
            Flow::Listen => bail!("CMD_POSTCOPY_LISTEN received outside of a package"),
        }
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
                ctx.info().vmdesc = Some(String::from_utf8_lossy(&buf).into_owned());
            }
        }
        Ok(())
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
}

/// `ram_pagesize_summary()`: every RAM block uses host pages of this size.
const HOST_PAGE_SIZE: u64 = 4096;

/// What the loaders of one incoming stream share, the parts of `MigrationIncomingState` the
/// commands use. The listen thread and the main thread both see it.
struct LoadCtx<'a> {
    footer: bool,
    page_bits: u32,
    postcopy_ram: bool,
    socket: Mutex<Option<Socket>>,
    // `to_src_file`.
    rp: Mutex<Option<Arc<ReturnPath>>>,
    hooks: Option<&'a dyn IncomingHooks>,
    ps: AtomicU8,
    // Whether the state went to postcopy-device, which `RUN` moves on to postcopy-active.
    device: AtomicBool,
    info: Mutex<LoadInfo>,
}

impl LoadCtx<'_> {
    fn rp(&self) -> Option<Arc<ReturnPath>> {
        lock(&self.rp).clone()
    }

    fn info(&self) -> MutexGuard<'_, LoadInfo> {
        lock(&self.info)
    }

    /// `postcopy_state_set()`: returns the state before.
    fn ps_set(&self, new: u8) -> u8 {
        self.ps.swap(new, Ordering::SeqCst)
    }

    fn set_state(&self, s: MigrationStatus) {
        if let Some(h) = self.hooks {
            h.set_state(s);
        }
    }

    /// Wakes up whoever reads the main stream, after a failure on the other thread.
    fn shutdown(&self) {
        if let Some(rp) = self.rp() {
            rp.shutdown();
        } else if let Some(s) = lock(&self.socket).as_ref() {
            s.shutdown();
        }
    }
}

/// How a run of `qemu_loadvm_state_main()` ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    /// `QEMU_VM_EOF`.
    Eof,
    /// `MIG_CMD_POSTCOPY_LISTEN`: the listen thread takes over the outer stream.
    Listen,
    /// `LOADVM_QUIT`, after `MIG_CMD_POSTCOPY_RUN`: stop reading on every level.
    Quit,
}

/// `qemu_loadvm_state_main()` over some of the entries: all of them, or after `LISTEN` the live
/// ones on the listen thread and the devices on the main thread.
struct Loader<'e, 'c> {
    entries: Vec<&'e mut Entry>,
    ctx: &'c LoadCtx<'c>,
}

impl Loader<'_, '_> {
    /// `qemu_loadvm_state_main()`.
    fn load_main(&mut self, f: &mut StreamReader<'_>) -> Result<Flow> {
        loop {
            let section_type = f.get_byte();
            let ret = f.get_error();
            if ret != 0 {
                bail!("Failed to load section ID: stream error: {}", ret);
            }
            match section_type {
                QEMU_VM_SECTION_START | QEMU_VM_SECTION_FULL => self.load_start_full(f)?,
                QEMU_VM_SECTION_PART | QEMU_VM_SECTION_END => self.load_part_end(f)?,
                QEMU_VM_COMMAND => {
                    if let Some(flow) = self.load_command(f)? {
                        return Ok(flow);
                    }
                }
                QEMU_VM_EOF => return Ok(Flow::Eof),
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
        let e = &mut *self.entries[i];
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
        Self::entry_load(self.entries[i], f, version_id)?;
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
        if !self.ctx.footer {
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

    fn live_entries(&mut self) -> impl Iterator<Item = &mut Box<dyn LiveState>> {
        self.entries.iter_mut().filter_map(|e| match &mut e.kind {
            Kind::Live(l) if l.is_active() => Some(l),
            _ => None,
        })
    }

    /// `loadvm_process_command()`. Returns how loading goes on: `None` to read the next
    /// section.
    fn load_command(&mut self, f: &mut StreamReader<'_>) -> Result<Option<Flow>> {
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
        let ctx = self.ctx;
        match c {
            cmd::OPEN_RETURN_PATH => {
                ctx.info().return_path = true;
                let mut rp = lock(&ctx.rp);
                if rp.is_some() {
                    error_report("CMD_OPEN_RETURN_PATH called when RP already open");
                    return Ok(None);
                }
                let sock = lock(&ctx.socket).as_ref().and_then(|s| s.try_clone().ok());
                match sock.and_then(|s| ReturnPath::new(s).ok()) {
                    Some(r) => *rp = Some(Arc::new(r)),
                    None => bail!("CMD_OPEN_RETURN_PATH failed"),
                }
                Ok(None)
            }
            cmd::PING => {
                let v = f.get_be32();
                match ctx.rp() {
                    // A failed write shows up on the source as a missing pong.
                    Some(rp) => {
                        let _ = rp.pong(v);
                    }
                    None => bail!("CMD_PING (0x{:x}) received with no return path", v),
                }
                Ok(None)
            }
            cmd::PACKAGED => {
                // loadvm_handle_cmd_packaged()
                let size = f.get_be32() as usize;
                if size > MAX_PACKAGED_SIZE {
                    bail!("Unreasonably large packaged state: {}", size);
                }
                let mut buf = vec![0; size];
                let got = f.get_buffer(&mut buf);
                if got != size {
                    bail!("CMD_PACKAGED: Buffer receive fail ret={} length={}", got, size);
                }
                let mut inner = StreamReader::new(&buf);
                match self.load_main(&mut inner)? {
                    Flow::Eof => Ok(None),
                    Flow::Quit => Ok(Some(Flow::Quit)),
                    Flow::Listen => {
                        self.listen(f, &mut inner)?;
                        Ok(Some(Flow::Quit))
                    }
                }
            }
            cmd::POSTCOPY_ADVISE => self.postcopy_advise(f, len).map(|()| None),
            cmd::POSTCOPY_RAM_DISCARD => self.postcopy_discard(f, len).map(|()| None),
            cmd::POSTCOPY_LISTEN => {
                let prev = ctx.ps_set(ps::LISTENING);
                if prev != ps::ADVISE && prev != ps::DISCARD {
                    bail!("CMD_POSTCOPY_LISTEN in wrong postcopy state ({})", prev);
                }
                Ok(Some(Flow::Listen))
            }
            cmd::POSTCOPY_RUN => {
                let cur = ctx.ps.load(Ordering::SeqCst);
                if cur != ps::LISTENING {
                    bail!("CMD_POSTCOPY_RUN in wrong postcopy state ({})", cur);
                }
                // Without a return path the state is postcopy-active already.
                if ctx.device.swap(false, Ordering::SeqCst) {
                    ctx.set_state(MigrationStatus::PostcopyActive);
                }
                ctx.ps_set(ps::RUNNING);
                if let Some(h) = ctx.hooks {
                    h.postcopy_run();
                }
                Ok(Some(Flow::Quit))
            }
            cmd::POSTCOPY_RESUME => {
                // ruvm never enters postcopy-recover, where this would be legal.
                warn_report("loadvm_postcopy_handle_resume: illegal resume received");
                Ok(None)
            }
            cmd::RECV_BITMAP => self.recv_bitmap(f, len).map(|()| None),
            cmd::SWITCHOVER_START => {
                ctx.info().switchover_start = true;
                Ok(None)
            }
            _ => bail!("MIG_CMD 0x{:x} deprecated (len 0x{:x})", c, len),
        }
    }

    /// `loadvm_postcopy_handle_advise()`.
    fn postcopy_advise(&mut self, f: &mut StreamReader<'_>, len: u16) -> Result<()> {
        let ctx = self.ctx;
        let prev = ctx.ps_set(ps::ADVISE);
        if prev != ps::NONE {
            bail!("CMD_POSTCOPY_ADVISE in wrong postcopy state ({})", prev);
        }
        match len {
            0 => {
                if ctx.postcopy_ram {
                    bail!("RAM postcopy is enabled but have 0 byte advise");
                }
                return Ok(());
            }
            16 => {
                if !ctx.postcopy_ram {
                    bail!("RAM postcopy is disabled but have 16 byte advise");
                }
            }
            _ => bail!("CMD_POSTCOPY_ADVISE invalid length ({})", len),
        }
        if let Err(e) = postcopy::supported_by_host() {
            ctx.ps_set(ps::NONE);
            return Err(e);
        }
        let remote = f.get_be64();
        if remote != HOST_PAGE_SIZE {
            bail!("Postcopy needs matching RAM page sizes (s={:x} d={:x})", remote, HOST_PAGE_SIZE);
        }
        let remote_tps = f.get_be64();
        let page = 1u64 << ctx.page_bits;
        if remote_tps != page {
            bail!("Postcopy needs matching target page sizes (s={} d={})", remote_tps as i32, page);
        }
        for l in self.live_entries() {
            l.postcopy_advise().map_err(|e| e.prepend("Postcopy RAM incoming init failed: "))?;
        }
        Ok(())
    }

    /// `loadvm_postcopy_ram_handle_discard()`.
    fn postcopy_discard(&mut self, f: &mut StreamReader<'_>, len: u16) -> Result<()> {
        let ctx = self.ctx;
        match ctx.ps.load(Ordering::SeqCst) {
            // The first discard; postcopy_ram_prepare_discard() has nothing to do here.
            ps::ADVISE => {
                ctx.ps_set(ps::DISCARD);
            }
            ps::DISCARD => {}
            cur => bail!("CMD_POSTCOPY_RAM_DISCARD in wrong postcopy state ({})", cur),
        }
        // A version byte, a block name of at least one byte with its NUL, and a range.
        if len < 1 + 1 + 1 + 1 + 2 * 8 {
            bail!("CMD_POSTCOPY_RAM_DISCARD invalid length ({})", len);
        }
        let version = f.get_byte();
        if version != 0 {
            bail!("CMD_POSTCOPY_RAM_DISCARD invalid version ({})", version);
        }
        let n = usize::from(f.get_byte());
        let mut id = vec![0; n];
        if n == 0 || f.get_buffer(&mut id) != n {
            bail!("CMD_POSTCOPY_RAM_DISCARD Failed to read RAMBlock ID");
        }
        let name = String::from_utf8_lossy(&id).into_owned();
        let nil = f.get_byte();
        if nil != 0 {
            bail!("CMD_POSTCOPY_RAM_DISCARD missing nil ({})", nil);
        }
        let mut len = len.wrapping_sub(3 + n as u16);
        if len % 16 != 0 {
            bail!("CMD_POSTCOPY_RAM_DISCARD invalid length ({})", len);
        }
        while len > 0 {
            let start = f.get_be64();
            let block_len = f.get_be64();
            len -= 16;
            let ret = self.live_entries().find_map(|l| l.postcopy_discard(&name, start, block_len));
            match ret {
                Some(Ok(())) => {}
                Some(Err(e)) => {
                    error_report(e.message());
                    bail!("Failed to discard RAM range {}: -1", name);
                }
                None => {
                    error_report(&format!("ram_discard_range: Failed to find block '{name}'"));
                    bail!("Failed to discard RAM range {}: -1", name);
                }
            }
        }
        Ok(())
    }

    /// `loadvm_handle_recv_bitmap()`.
    fn recv_bitmap(&mut self, f: &mut StreamReader<'_>, len: u16) -> Result<()> {
        let n = usize::from(f.get_byte());
        let mut id = vec![0; n];
        if n == 0 || f.get_buffer(&mut id) != n {
            bail!("failed to read block name");
        }
        let ret = f.get_error();
        if ret < 0 {
            bail!("loadvm failed: stream error: {}", ret);
        }
        if usize::from(len) != n + 1 {
            bail!("invalid payload length ({})", len);
        }
        let name = String::from_utf8_lossy(&id).into_owned();
        if !self.live_entries().any(|l| l.has_ram_block(&name)) {
            bail!("block '{}' not found", name);
        }
        // The source only asks for this in postcopy recovery, which ruvm does not do.
        bail!("MIG_CMD_RECV_BITMAP for block '{}' outside of postcopy recovery", name)
    }

    /// `MIG_CMD_POSTCOPY_LISTEN` inside the package: the listen thread loads the rest of the
    /// outer stream `f` into the live entries while this thread loads the device state from
    /// the package `inner`. Returns once the listen thread is done, after the guest started.
    fn listen(&mut self, f: &mut StreamReader<'_>, inner: &mut StreamReader<'_>) -> Result<()> {
        let ctx = self.ctx;
        let entries = std::mem::take(&mut self.entries);
        let (live, devices): (Vec<_>, Vec<_>) =
            entries.into_iter().partition(|e| matches!(e.kind, Kind::Live(_)));
        let mut live = Loader { entries: live, ctx };
        let mut devices = Loader { entries: devices, ctx };
        let rp = ctx.rp();
        if ctx.postcopy_ram {
            let mut ret = Ok(());
            for l in live.live_entries() {
                ret = l.postcopy_listen(rp.as_ref());
                if ret.is_err() {
                    break;
                }
            }
            if let Err(e) = ret {
                for l in live.live_entries() {
                    l.postcopy_end();
                }
                error_report(e.message());
                bail!("Failed to setup incoming postcopy RAM blocks");
            }
        }
        // postcopy_ram_listen_thread(): postcopy-device until RUN when the source waits for
        // the package to load, postcopy-active at once when it cannot know.
        let device = rp.is_some();
        ctx.device.store(device, Ordering::SeqCst);
        ctx.set_state(if device {
            MigrationStatus::PostcopyDevice
        } else {
            MigrationStatus::PostcopyActive
        });

        let (main, listen) = std::thread::scope(|s| {
            let thread = std::thread::Builder::new()
                .name("mig/dst/listen".to_string())
                .spawn_scoped(s, || {
                    let ret = live.load_main(f);
                    if ret.is_err() && f.get_error() == 0 {
                        f.set_error(-ruvm_vmstate::EINVAL);
                    }
                    let err = f.get_error();
                    for l in live.live_entries() {
                        l.postcopy_end();
                    }
                    ctx.ps_set(ps::END);
                    ret.map(drop).map_err(|e| (e, err))
                });
            let thread = match thread {
                Ok(t) => t,
                Err(e) => {
                    return (Err(Error::from_io("failed to create the listen thread", e)), Ok(()));
                }
            };
            let main = match devices.load_main(inner) {
                Ok(Flow::Quit) => Ok(()),
                Ok(_) => Err(Error::generic("The postcopy package ended without CMD_POSTCOPY_RUN")),
                Err(e) => Err(e),
            };
            if main.is_err() {
                ctx.shutdown();
            }
            let listen = thread.join().unwrap_or_else(|_| {
                Err((Error::generic("the listen thread panicked"), -ruvm_vmstate::EINVAL))
            });
            (main, listen)
        });
        main?;
        if let Err((e, err)) = listen {
            let e = e.prepend(format!("loadvm failed during postcopy: {err}: "));
            ctx.set_state(MigrationStatus::Failed);
            return Err(e);
        }
        ctx.info().postcopy = true;
        Ok(())
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
                error_report(&format!(
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
    fn postcopy_commands() {
        let mut s = SaveVm::new(config());
        let mut w = StreamWriter::new();
        let mut conf = Configuration::for_save(&config(), Vec::new(), false);
        w.put_be32(QEMU_VM_FILE_MAGIC);
        w.put_be32(QEMU_VM_FILE_VERSION);
        w.put_byte(QEMU_VM_CONFIGURATION);
        VMSTATE_CONFIGURATION.save(&mut w, &mut conf).unwrap();
        let head = w.into_inner();
        let mut load = |cmds: &[(u16, Vec<u8>)]| {
            let mut w = StreamWriter::new();
            w.put_buffer(&head);
            for (c, data) in cmds {
                if *c == cmd::PACKAGED {
                    // The length field covers only the size in front of the package.
                    w.put_byte(QEMU_VM_COMMAND);
                    w.put_be16(*c);
                    w.put_be16(4);
                    w.put_buffer(data);
                } else {
                    command(&mut w, *c, data);
                }
            }
            w.put_byte(QEMU_VM_EOF);
            s.load_state(&mut StreamReader::new(&w.into_inner()))
        };
        let err = |r: Result<LoadInfo>| r.unwrap_err().message().to_string();

        assert_eq!(
            err(load(&[(cmd::PING, 1u32.to_be_bytes().to_vec())])),
            "CMD_PING (0x1) received with no return path"
        );
        assert_eq!(
            err(load(&[(cmd::OPEN_RETURN_PATH, Vec::new())])),
            "CMD_OPEN_RETURN_PATH failed"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_ADVISE, vec![0; 16])])),
            "RAM postcopy is disabled but have 16 byte advise"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_ADVISE, vec![0; 8])])),
            "CMD_POSTCOPY_ADVISE invalid length (8)"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_ADVISE, Vec::new()), (cmd::POSTCOPY_ADVISE, Vec::new())])),
            "CMD_POSTCOPY_ADVISE in wrong postcopy state (1)"
        );
        let mut discard = vec![0, 2, b'p', b'c', 0];
        discard.extend_from_slice(&[0; 16]);
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_RAM_DISCARD, discard.clone())])),
            "CMD_POSTCOPY_RAM_DISCARD in wrong postcopy state (0)"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_ADVISE, Vec::new()), (cmd::POSTCOPY_RAM_DISCARD, discard)])),
            "Failed to discard RAM range pc: -1"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_RUN, Vec::new())])),
            "CMD_POSTCOPY_RUN in wrong postcopy state (0)"
        );
        assert_eq!(
            err(load(&[(cmd::POSTCOPY_LISTEN, Vec::new())])),
            "CMD_POSTCOPY_LISTEN in wrong postcopy state (0)"
        );
        assert_eq!(
            err(load(&[(cmd::RECV_BITMAP, vec![3, b'a', b'b', b'c'])])),
            "block 'abc' not found"
        );
        // An empty package, and a resume outside of recovery, are both let through.
        let mut pkg = 1u32.to_be_bytes().to_vec();
        pkg.push(QEMU_VM_EOF);
        assert!(load(&[(cmd::PACKAGED, pkg), (cmd::POSTCOPY_RESUME, Vec::new())]).is_ok());
        let pkg = (MAX_PACKAGED_SIZE as u32 + 1).to_be_bytes().to_vec();
        assert_eq!(
            err(load(&[(cmd::PACKAGED, pkg)])),
            format!("Unreasonably large packaged state: {}", MAX_PACKAGED_SIZE + 1)
        );
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
