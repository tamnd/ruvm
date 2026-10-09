// SPDX-License-Identifier: GPL-2.0-or-later

//! migration/migration.c and migration/options.c: the migration state of a machine.
//!
//! A [`Migration`] holds the capabilities and parameters, runs the outgoing precopy in its own
//! thread the way `migration_thread()` does, takes one incoming migration, and reports both in
//! `query-migrate`. The machine side, its run state and its events, is behind [`MigrationHost`].
//!
//! The outgoing thread works like QEMU's: after the header and the setup sections it sends the
//! live entries in 100 ms windows held to `max-bandwidth`, measures the bandwidth of each window
//! and from it the bytes that fit into `downtime-limit`. Once what is left fits, it stops the
//! guest, sends the rest with the device state and completes.
//!
//! With `postcopy-ram` or `return-path` a second thread reads the return path from the
//! destination. After `migrate-start-postcopy` the outgoing thread switches over as soon as what
//! must go before the switchover fits (`postcopy_start()`): it stops the guest, sends the pages
//! the destination has to drop again and the device state in one package, and from then on
//! sends the rest of RAM with the pages the destination asks for first. The incoming side goes
//! into postcopy in the loader, which reports the state changes back through
//! [`IncomingHooks`].

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result, bail, error_report};
use ruvm_qapi::types::{
    MigMode, MigrationAddressU, MigrationCapability, MigrationCapabilityStatus, MigrationChannel,
    MigrationInfo, MigrationParameters, MigrationRAMStats, MigrationStatus, MultiFDCompression,
    SocketAddress, SocketAddressU, StrOrNull, XBZRLECacheStats, ZeroPageDetection,
};
use ruvm_vmstate::StreamReader;

use crate::channel::{
    Channel, FdResolver, FileChannel, MigrationAddr, Socket, open_file, parse_input,
    parse_input_cpr, supports_multi_channels, supports_seeking,
};
use crate::cpr;
use crate::multifd::{self, Compression, MultifdParams, MultifdRecv, MultifdSend, RecvConfig};
use crate::postcopy::{self, PageRequests, SourceRp, source_return_path};
use crate::ram::RamStats;
use crate::savevm::{IncomingHooks, LoadInfo, LoadOptions, QemuFile, SaveParams, SaveVm};
use crate::write_tracking;

/// `BUFFER_DELAY`: the length of a rate limiting and bandwidth window, in milliseconds.
const BUFFER_DELAY: u64 = 100;
/// `XFER_LIMIT_RATIO`: windows per second.
const XFER_LIMIT_RATIO: u64 = 1000 / BUFFER_DELAY;
/// `MAX_MIGRATE_DOWNTIME`, in milliseconds.
const MAX_MIGRATE_DOWNTIME: u64 = 2000 * 1000;
/// The bytes one iteration sends at most when there is no rate limit, so that cancelling and
/// the window bookkeeping stay responsive.
const UNLIMITED_CHUNK: u64 = 16 << 20;
/// The same for a background snapshot, which has no rate limit at all.
const BG_CHUNK: u64 = 1 << 20;
const PAGE_SIZE: u64 = 4096;
/// `DEFAULT_MIGRATE_XBZRLE_CACHE_SIZE`.
const XBZRLE_CACHE_SIZE: u64 = 64 << 20;

/// What a migration needs from the machine.
pub trait MigrationHost: Send + Sync {
    /// Whether the guest runs (`runstate_is_running()`).
    fn is_running(&self) -> bool;

    /// Whether the machine waits for an incoming migration (`RUN_STATE_INMIGRATE`).
    fn is_incoming(&self) -> bool {
        false
    }

    /// Whether the guest was stopped by a completed migration (`RUN_STATE_POSTMIGRATE`).
    fn is_postmigrate(&self) -> bool {
        false
    }

    /// `migration_stop_vm()`: stores the global state and stops the guest in
    /// `RUN_STATE_FINISH_MIGRATE` for the switchover.
    fn stop_for_switchover(&self) -> Result<()>;

    /// After a failed or cancelled migration: starts the guest again if `was_running`, else
    /// leaves `RUN_STATE_FINISH_MIGRATE` for the old state.
    fn resume_after_failure(&self, was_running: bool);

    /// After a completed migration: `RUN_STATE_POSTMIGRATE`.
    fn set_postmigrate(&self);

    /// `migration_stop_vm(s, RUN_STATE_PAUSED)`: stores the global state and pauses the guest
    /// while a background snapshot saves the device state.
    fn stop_for_snapshot(&self) -> Result<()> {
        self.stop_for_switchover()
    }

    /// `vm_resume(s->vm_old_state)`: once RAM is write-protected the guest goes on as it was
    /// before [`stop_for_snapshot`](Self::stop_for_snapshot).
    fn resume_after_snapshot(&self, was_running: bool) {
        self.resume_after_failure(was_running);
    }

    /// `process_incoming_migration_bh()`: the state is loaded; start the guest (or pause it
    /// without autostart) as the global state from the source says.
    fn incoming_done(&self);

    /// The incoming migration failed with `err`. With `exit_on_error`, QEMU reports the error
    /// and exits.
    fn incoming_failed(&self, err: &Error, exit_on_error: bool);

    /// The `MIGRATION` event, sent only with the `events` capability.
    fn status_event(&self, _status: MigrationStatus) {}

    /// The `MIGRATION_PASS` event, sent only with the `events` capability.
    fn pass_event(&self, _pass: i64) {}

    /// `global_state_store()`: records the run state for the `globalstate` section, before
    /// `savevm` stops the guest.
    fn global_state_store(&self) {}

    /// `qemu_system_reset(SHUTDOWN_CAUSE_SNAPSHOT_LOAD)`, before a snapshot loads.
    fn snapshot_reset(&self) -> Result<()> {
        Ok(())
    }

    /// A snapshot loaded: `global_state_post_load()` hands on whether the guest was
    /// suspended.
    fn snapshot_loaded(&self) {}

    /// `qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL)`, which a snapshot records.
    fn vm_clock_ns(&self) -> u64 {
        0
    }
}

/// The capabilities this side implements. The others are refused, so that a management tool
/// learns at `migrate-set-capabilities` rather than half way through.
const SUPPORTED_CAPS: &[MigrationCapability] = &[
    MigrationCapability::Events,
    MigrationCapability::ValidateUuid,
    MigrationCapability::LateBlockActivate,
    MigrationCapability::PauseBeforeSwitchover,
    MigrationCapability::PostcopyRam,
    MigrationCapability::ReturnPath,
    MigrationCapability::Multifd,
    MigrationCapability::Xbzrle,
    MigrationCapability::MappedRam,
    MigrationCapability::BackgroundSnapshot,
    MigrationCapability::SwitchoverAck,
];

/// `check_caps_background_snapshot`: what `background-snapshot` cannot go with.
const BACKGROUND_SNAPSHOT_CONFLICTS: &[MigrationCapability] = &[
    MigrationCapability::PostcopyRam,
    MigrationCapability::DirtyBitmaps,
    MigrationCapability::PostcopyBlocktime,
    MigrationCapability::LateBlockActivate,
    MigrationCapability::ReturnPath,
    MigrationCapability::Multifd,
    MigrationCapability::PauseBeforeSwitchover,
    MigrationCapability::AutoConverge,
    MigrationCapability::ReleaseRam,
    MigrationCapability::RdmaPinAll,
    MigrationCapability::Xbzrle,
    MigrationCapability::XColo,
    MigrationCapability::ValidateUuid,
    MigrationCapability::ZeroCopySend,
];

/// `migration_channels_and_transport_compatible()`. `direct_io` is the `direct-io` parameter
/// and `mode` the `mode` one.
fn transport_compatible(
    addr: &MigrationAddr,
    caps: &[MigrationCapability],
    direct_io: bool,
    mode: MigMode,
) -> Result<()> {
    let mapped_ram = caps.contains(&MigrationCapability::MappedRam);
    let multifd = caps.contains(&MigrationCapability::Multifd);
    if mapped_ram && !supports_seeking(addr) {
        bail!("Migration requires seekable transport (e.g. file)");
    }
    // migration_needs_multiple_sockets(); postcopy-preempt is not supported here.
    if multifd && !supports_multi_channels(addr, mapped_ram) {
        bail!("Migration requires multi-channel URIs (e.g. tcp)");
    }
    // migration_needs_extra_fds(): migrate_direct_io() opens the file once more per channel.
    if multifd && direct_io && mapped_ram && !supports_seeking(addr) {
        bail!("Migration requires a transport that allows for extra fds (e.g. file)");
    }
    // The memory stays where it is; the stream has to go to a process running at the same time.
    if mode == MigMode::CprTransfer && matches!(addr.u, MigrationAddressU::File(_)) {
        bail!("Migration requires streamable transport (eg unix)");
    }
    Ok(())
}

/// `migration_in_postcopy()`, for the states this side reaches.
fn in_postcopy(s: MigrationStatus) -> bool {
    matches!(s, MigrationStatus::PostcopyDevice | MigrationStatus::PostcopyActive)
}

fn is_running_state(s: MigrationStatus) -> bool {
    // migration_is_running()
    !matches!(
        s,
        MigrationStatus::None
            | MigrationStatus::Completed
            | MigrationStatus::Failed
            | MigrationStatus::Cancelled
            | MigrationStatus::Colo
    )
}

/// The parameters with QEMU's defaults (`migration_properties`).
pub fn default_parameters() -> MigrationParameters {
    MigrationParameters {
        announce_initial: Some(50),
        announce_max: Some(550),
        announce_rounds: Some(5),
        announce_step: Some(100),
        throttle_trigger_threshold: Some(50),
        cpu_throttle_initial: Some(20),
        cpu_throttle_increment: Some(10),
        cpu_throttle_tailslow: Some(false),
        tls_creds: Some(StrOrNull::S(String::new())),
        tls_hostname: Some(StrOrNull::S(String::new())),
        tls_authz: Some(StrOrNull::S(String::new())),
        max_bandwidth: Some(128 << 20),
        avail_switchover_bandwidth: Some(0),
        downtime_limit: Some(300),
        x_checkpoint_delay: Some(20000),
        multifd_channels: Some(2),
        xbzrle_cache_size: Some(XBZRLE_CACHE_SIZE),
        max_postcopy_bandwidth: Some(0),
        max_cpu_throttle: Some(99),
        multifd_compression: Some(MultiFDCompression::None),
        multifd_zlib_level: Some(1),
        multifd_qatzip_level: Some(1),
        multifd_zstd_level: Some(1),
        block_bitmap_mapping: None,
        x_vcpu_dirty_limit_period: Some(1000),
        vcpu_dirty_limit: Some(1),
        mode: Some(MigMode::Normal),
        zero_page_detection: Some(ZeroPageDetection::Multifd),
        direct_io: Some(false),
        x_rdma_chunk_size: Some(1 << 20),
        cpr_exec_command: None,
    }
}

macro_rules! merge {
    ($dst:expr, $src:expr, $($f:ident),* $(,)?) => {
        $( if let Some(v) = $src.$f.clone() { $dst.$f = Some(v); } )*
    };
}

/// `migrate_params_check()` on the parameters with all members present. `mapped_ram_conflict`
/// says whether `mapped-ram` is on while the current parameters, not the new ones, ask for
/// compression or TLS; QEMU checks it that way.
fn check_parameters(p: &MigrationParameters, mapped_ram_conflict: bool) -> Result<()> {
    let u8v = |v: Option<u8>| u64::from(v.unwrap_or(0));
    let u = |v: Option<u64>| v.unwrap_or(0);
    let t = u8v(p.throttle_trigger_threshold);
    if !(1..=100).contains(&t) {
        bail!("Option throttle-trigger-threshold expects an integer in the range of 1 to 100");
    }
    let initial = u8v(p.cpu_throttle_initial);
    if !(1..=99).contains(&initial) {
        bail!("Option cpu-throttle-initial expects an integer in the range of 1 to 99");
    }
    if !(1..=99).contains(&u8v(p.cpu_throttle_increment)) {
        bail!("Option cpu-throttle-increment expects an integer in the range of 1 to 99");
    }
    if u(p.downtime_limit) > MAX_MIGRATE_DOWNTIME {
        bail!("Option downtime-limit expects an integer in the range of 0 to (2000 * 1000) ms");
    }
    if u8v(p.multifd_channels) < 1 {
        bail!("Option multifd-channels expects a value between 1 and 255");
    }
    if u8v(p.multifd_zlib_level) > 9 {
        bail!("Option multifd-zlib-level expects a value between 0 and 9");
    }
    if !(1..=9).contains(&u8v(p.multifd_qatzip_level)) {
        bail!("Option multifd-qatzip-level expects a value between 1 and 9");
    }
    if u8v(p.multifd_zstd_level) > 20 {
        bail!("Option multifd-zstd-level expects a value between 0 and 20");
    }
    let xbzrle = u(p.xbzrle_cache_size);
    if xbzrle < PAGE_SIZE || !xbzrle.is_power_of_two() {
        bail!("Option xbzrle-cache-size expects a power of two no less than the target page size");
    }
    let max_throttle = u8v(p.max_cpu_throttle);
    if max_throttle < initial || max_throttle > 99 {
        bail!(
            "Option max-cpu-throttle expects an integer in the range of cpu-throttle-initial to 99"
        );
    }
    if u(p.announce_initial) > 100000 {
        bail!("Option announce-initial expects a value between 0 and 100000");
    }
    if u(p.announce_max) > 100000 {
        bail!("Option announce-max expects a value between 0 and 100000");
    }
    if u(p.announce_rounds) > 1000 {
        bail!("Option announce-rounds expects a value between 0 and 1000");
    }
    if !(1..=10000).contains(&u(p.announce_step)) {
        bail!("Option announce-step expects a value between 0 and 10000");
    }
    if mapped_ram_conflict {
        bail!("Mapped-ram only available for non-compressed non-TLS multifd migration");
    }
    if !(1..=1000).contains(&u(p.x_vcpu_dirty_limit_period)) {
        bail!("Option x-vcpu-dirty-limit-period expects a value between 1 and 1000");
    }
    if u(p.vcpu_dirty_limit) < 1 {
        bail!("Parameter 'vcpu-dirty-limit' must be greater than 1 MB/s");
    }
    if p.direct_io == Some(true) && ruvm_sys::directio::o_direct().is_none() {
        bail!("No build-time support for direct-io");
    }
    let chunk = u(p.x_rdma_chunk_size);
    if !((1 << 20)..=(1 << 30)).contains(&chunk) || !chunk.is_power_of_two() {
        bail!("Option x_rdma_chunk_size expects a power of 2 in the range 1MiB to 1024MiB");
    }
    Ok(())
}

fn str_or_null(v: &Option<StrOrNull>) -> &str {
    match v {
        Some(StrOrNull::S(s)) => s,
        _ => "",
    }
}

#[derive(Debug)]
struct Shared {
    caps: Vec<MigrationCapability>,
    params: MigrationParameters,
    has_block_bitmap_mapping: bool,

    state: MigrationStatus,
    incoming_state: MigrationStatus,
    incoming_started: bool,
    socket_address: Option<Vec<SocketAddress>>,
    error: Option<String>,

    start_time: Option<Instant>,
    setup_time: i64,
    total_time: i64,
    downtime: i64,
    mbps: f64,
    pages_per_second: u64,
    // Bytes the live entries had pending at the last query and at the last exact one.
    pending_bytes: u64,
    dirty_bytes_last_sync: u64,
    // The page requests of the outgoing migration, for `postcopy-requests`.
    requests: Option<Arc<PageRequests>>,

    thread: Option<JoinHandle<()>>,
    incoming_thread: Option<JoinHandle<()>>,
}

struct Inner {
    host: Arc<dyn MigrationHost>,
    savevm: Arc<Mutex<SaveVm>>,
    ram: Option<Arc<RamStats>>,
    // The blockers ram_block_add_cpr_blocker() adds for `cpr-transfer`, one for each RAM block
    // the next process could not map again, in the order the blocks were made.
    cpr_blockers: Vec<String>,
    fds: Mutex<Option<Arc<FdResolver>>>,
    shared: Mutex<Shared>,
    // Set by `migrate_cancel` and checked by the thread and each write to the channel.
    cancel: Arc<AtomicBool>,
    // `pause_event`, for `pause-before-switchover`.
    resume: Condvar,
    resumed: AtomicBool,
    transferred: AtomicU64,
    // `start_postcopy`, set by `migrate-start-postcopy`.
    start_postcopy: AtomicBool,
    // `xbzrle-cache-size`, which a running migration resizes its cache to.
    xbzrle_cache_size: Arc<AtomicU64>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Inner {
    /// `migration_blockers[mode]`, oldest first.
    fn blockers(&self, mode: MigMode) -> &[String] {
        match mode {
            MigMode::CprTransfer => &self.cpr_blockers,
            _ => &[],
        }
    }

    fn events(&self, s: &Shared) -> bool {
        s.caps.contains(&MigrationCapability::Events)
    }

    /// `migrate_set_state()` for the outgoing side.
    fn set_state(&self, old: Option<MigrationStatus>, new: MigrationStatus) -> bool {
        let mut s = lock(&self.shared);
        if old.is_some_and(|o| o != s.state) {
            return false;
        }
        s.state = new;
        let events = self.events(&s);
        drop(s);
        if events {
            self.host.status_event(new);
        }
        true
    }

    fn set_incoming_state(&self, new: MigrationStatus) {
        let mut s = lock(&self.shared);
        s.incoming_state = new;
        let events = self.events(&s);
        drop(s);
        if events {
            self.host.status_event(new);
        }
    }

    fn set_error(&self, err: &Error) {
        let mut s = lock(&self.shared);
        if s.error.is_none() {
            s.error = Some(err.message().to_string());
        }
    }

    fn state(&self) -> MigrationStatus {
        lock(&self.shared).state
    }

    fn has_error(&self) -> bool {
        lock(&self.shared).error.is_some()
    }

    fn error(&self) -> Error {
        let msg = lock(&self.shared).error.clone();
        Error::generic(msg.unwrap_or_else(|| "migration failed".to_string()))
    }
}

/// The incoming side as the loader sees it.
struct Hooks<'a>(&'a Inner);

impl IncomingHooks for Hooks<'_> {
    fn set_state(&self, status: MigrationStatus) {
        self.0.set_incoming_state(status);
    }

    fn postcopy_run(&self) {
        self.0.host.incoming_done();
    }
}

/// The source end of the return path: the thread reading it and what it found out.
struct RpHandle {
    state: Arc<SourceRp>,
    requests: Option<Arc<PageRequests>>,
    // Another handle on the socket, to wake the thread up on an error.
    ctl: Socket,
    thread: Option<JoinHandle<()>>,
}

/// `migration_connect_error_propagate()` for a migration that failed before its thread ran.
fn connect_error(inner: &Inner, e: &Error) {
    inner.set_error(e);
    inner.set_state(Some(MigrationStatus::Setup), MigrationStatus::Failing);
    inner.set_state(Some(MigrationStatus::Failing), MigrationStatus::Failed);
}

/// A migration whose channel is up, for its thread to run.
struct Outgoing {
    sink: CancelWriter,
    file: Option<Arc<FileChannel>>,
    rp: Option<RpHandle>,
    addr: MigrationAddr,
    background: bool,
    cpr: bool,
}

impl Outgoing {
    fn thread_name(&self) -> &'static str {
        if self.background { "mig/snapshot" } else { "live_migration" }
    }

    fn run(self, inner: Arc<Inner>) {
        if self.background {
            bg_migration_thread(inner, self.sink, self.file, self.addr);
        } else {
            migration_thread(inner, self.sink, self.file, self.rp, self.addr, self.cpr);
        }
    }
}

/// `migration_connect_outgoing()` and the part of `migration_connect()` before the thread:
/// connects the main channel and opens the return path.
fn connect_outgoing(inner: &Arc<Inner>, addr: MigrationAddr, mode: MigMode) -> Result<Outgoing> {
    let fds = lock(&inner.fds).clone();
    let conn = Channel::connect_socket(&addr, fds.as_deref())?;
    let (sink, sock, file) = (conn.out, conn.socket, conn.file);
    let requests = lock(&inner.savevm).page_requests();
    lock(&inner.shared).requests = requests.clone();
    let (want_rp, background) = {
        let s = lock(&inner.shared);
        (
            s.caps.contains(&MigrationCapability::PostcopyRam)
                || s.caps.contains(&MigrationCapability::ReturnPath),
            s.caps.contains(&MigrationCapability::BackgroundSnapshot),
        )
    };
    let rp = if want_rp {
        // QEMU reads the return path from any channel; here only sockets have one.
        let Some(sock) = sock else {
            bail!("Unable to open return-path for postcopy");
        };
        Some(open_return_path(inner, sock, requests)?)
    } else {
        None
    };
    let sink = CancelWriter { inner: sink, cancel: inner.cancel.clone() };
    Ok(Outgoing { sink, file, rp, addr, background, cpr: mode != MigMode::Normal })
}

/// `open_return_path_on_source()`: starts the thread that reads the return path from `sock`.
fn open_return_path(
    inner: &Arc<Inner>,
    mut sock: Socket,
    requests: Option<Arc<PageRequests>>,
) -> Result<RpHandle> {
    let ctl = sock.try_clone().map_err(|e| Error::from_io("Unable to open return-path", e))?;
    // migrate_init(): switchover-ack the old way waits for one acknowledgement for all devices.
    // The new way counts the devices that ask for one, and none here does.
    let ack = lock(&inner.shared).caps.contains(&MigrationCapability::SwitchoverAck);
    let legacy = ack && lock(&inner.savevm).switchover_ack_legacy();
    let state = Arc::new(SourceRp {
        switchover_ack_pending: AtomicU32::new(u32::from(legacy)),
        ..SourceRp::default()
    });
    let (rp_inner, rp_state, rp_requests) = (inner.clone(), state.clone(), requests.clone());
    let thread = std::thread::Builder::new()
        .name("mig/src/rp-thr".to_string())
        .spawn(move || {
            let running = || is_running_state(rp_inner.state());
            let res = source_return_path(&mut sock, &rp_state, rp_requests.as_deref(), &running);
            if let Err(e) = res {
                rp_inner.set_error(&e);
            }
            // The migration thread may be waiting for the package to load; let it see the
            // error.
            if let Some(r) = &rp_requests {
                r.kick();
            }
        })
        .map_err(|e| Error::from_io("Failed to create the return path thread", e))?;
    Ok(RpHandle { state, requests, ctl, thread: Some(thread) })
}

/// A writer that fails once the migration is cancelled, so the thread stops at its next write.
/// The error must not be `Interrupted`, which `write_all()` retries for ever.
struct CancelWriter {
    inner: Box<dyn Write + Send>,
    cancel: Arc<AtomicBool>,
}

impl Write for CancelWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::other("migration cancelled"));
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::other("migration cancelled"));
        }
        self.inner.flush()
    }
}

/// `MigrationState` and `MigrationIncomingState` of a machine.
pub struct Migration {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Migration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = lock(&self.inner.shared);
        f.debug_struct("Migration")
            .field("state", &s.state)
            .field("incoming_state", &s.incoming_state)
            .finish()
    }
}

impl Migration {
    /// The migration state of a machine whose entries are in `savevm`. `ram` are the counters
    /// of its `ram` entry, for `query-migrate`.
    pub fn new(
        savevm: Arc<Mutex<SaveVm>>,
        ram: Option<Arc<RamStats>>,
        host: Arc<dyn MigrationHost>,
    ) -> Self {
        let cpr_blockers = lock(&savevm)
            .ram_blocks()
            .iter()
            .filter(|b| !b.is_shared())
            .map(|b| {
                format!(
                    "Memory region {} is not compatible with CPR. share=on is required for \
                     memory-backend objects, and aux-ram-share=on is required.",
                    b.name()
                )
            })
            .collect();
        Migration {
            inner: Arc::new(Inner {
                host,
                savevm,
                ram,
                cpr_blockers,
                fds: Mutex::new(None),
                shared: Mutex::new(Shared {
                    caps: Vec::new(),
                    params: default_parameters(),
                    has_block_bitmap_mapping: false,
                    state: MigrationStatus::None,
                    incoming_state: MigrationStatus::None,
                    incoming_started: false,
                    socket_address: None,
                    error: None,
                    start_time: None,
                    setup_time: 0,
                    total_time: 0,
                    downtime: 0,
                    mbps: 0.0,
                    pages_per_second: 0,
                    pending_bytes: 0,
                    dirty_bytes_last_sync: 0,
                    requests: None,
                    thread: None,
                    incoming_thread: None,
                }),
                cancel: Arc::new(AtomicBool::new(false)),
                resume: Condvar::new(),
                resumed: AtomicBool::new(false),
                transferred: AtomicU64::new(0),
                start_postcopy: AtomicBool::new(false),
                xbzrle_cache_size: Arc::new(AtomicU64::new(XBZRLE_CACHE_SIZE)),
            }),
        }
    }

    /// How `fd:` channels find a file descriptor by name or number (`monitor_fd_param()`).
    pub fn set_fd_resolver(&self, fds: Arc<FdResolver>) {
        *lock(&self.inner.fds) = Some(fds);
    }

    /// The outgoing status.
    pub fn status(&self) -> MigrationStatus {
        self.inner.state()
    }

    /// The incoming status.
    pub fn incoming_status(&self) -> MigrationStatus {
        lock(&self.inner.shared).incoming_state
    }

    /// `migration_is_running()`.
    pub fn is_running(&self) -> bool {
        is_running_state(self.inner.state())
    }

    /// Whether a capability is on.
    pub fn capability(&self, cap: MigrationCapability) -> bool {
        lock(&self.inner.shared).caps.contains(&cap)
    }

    /// `migrate_can_snapshot()`: snapshots do not go with some capabilities.
    pub fn can_snapshot(&self) -> Result<()> {
        // check_caps_savevm
        for cap in [MigrationCapability::Multifd] {
            if self.capability(cap) {
                bail!("Snapshots are not compatible with {}", cap.as_str());
            }
        }
        Ok(())
    }

    /// [`MigrationHost::global_state_store`].
    pub fn global_state_store(&self) {
        self.inner.host.global_state_store();
    }

    /// [`MigrationHost::vm_clock_ns`].
    pub fn vm_clock_ns(&self) -> u64 {
        self.inner.host.vm_clock_ns()
    }

    /// `qemu_savevm_state()`: the whole state into `out` in one go, with the guest stopped, as
    /// `savevm` saves it into a snapshot. Gives the bytes written, `qemu_file_transferred()`.
    /// The outgoing status goes through `setup` to `completed` or `failed`.
    pub fn save_snapshot_state(&self, out: impl Write + Send) -> Result<u64> {
        let inner = &self.inner;
        if is_running_state(inner.state()) {
            bail!("There's a migration process in progress");
        }
        // The thread takes the state lock on its way out, so it must not be held here.
        let thread = lock(&inner.shared).thread.take();
        if let Some(t) = thread {
            let _ = t.join();
        }
        // migrate_init()
        let caps = {
            let mut s = lock(&inner.shared);
            s.error = None;
            s.start_time = Some(Instant::now());
            s.setup_time = 0;
            s.total_time = 0;
            s.downtime = 0;
            s.mbps = 0.0;
            s.pages_per_second = 0;
            s.pending_bytes = 0;
            s.dirty_bytes_last_sync = 0;
            s.requests = None;
            s.state = MigrationStatus::None;
            s.caps.clone()
        };
        inner.cancel.store(false, Ordering::Relaxed);
        inner.transferred.store(0, Ordering::Relaxed);
        if let Some(r) = &inner.ram {
            r.reset();
            r.guest_running.store(inner.host.is_running(), Ordering::Relaxed);
        }
        inner.start_postcopy.store(false, Ordering::Relaxed);
        inner.set_state(Some(MigrationStatus::None), MigrationStatus::Setup);

        let params = lock(&inner.shared).params.clone();
        let xbzrle_cache_size = caps.contains(&MigrationCapability::Xbzrle).then(|| {
            let size = params.xbzrle_cache_size.unwrap_or(XBZRLE_CACHE_SIZE);
            inner.xbzrle_cache_size.store(size, Ordering::Relaxed);
            inner.xbzrle_cache_size.clone()
        });
        let save = SaveParams {
            multifd: None,
            zero_page_detection: params.zero_page_detection.unwrap_or(ZeroPageDetection::Multifd),
            xbzrle_cache_size,
            mapped_ram: caps.contains(&MigrationCapability::MappedRam),
            background_snapshot: false,
            ignore_ram: false,
        };
        let mut f = QemuFile::new(out);
        let res = {
            let mut vm = lock(&inner.savevm);
            vm.capabilities = caps;
            vm.set_save_params(&save);
            let res = (|| {
                vm.save_header(&mut f)?;
                vm.save_setup(&mut f)?;
                loop {
                    // qemu_fflush() makes every failed write -EIO.
                    let done = vm.save_iterate(&mut f, UNLIMITED_CHUNK, false).map_err(|_| {
                        Error::generic("Error while writing VM state: Input/output error")
                    })?;
                    if done {
                        break;
                    }
                }
                vm.save_complete(&mut f)
            })();
            vm.save_cleanup();
            res
        };
        let transferred = f.transferred();
        inner.transferred.store(transferred, Ordering::Relaxed);
        let status = match &res {
            Ok(()) => MigrationStatus::Completed,
            Err(e) => {
                inner.set_error(e);
                MigrationStatus::Failed
            }
        };
        {
            let mut s = lock(&inner.shared);
            if let Some(t) = s.start_time {
                s.total_time = t.elapsed().as_millis() as i64;
            }
        }
        inner.set_state(Some(MigrationStatus::Setup), status);
        res.map(|()| transferred)
    }

    /// The loading half of `load_snapshot()`: resets the machine and loads the state that
    /// [`save_snapshot_state`](Self::save_snapshot_state) wrote from `input`. The guest must be
    /// stopped.
    pub fn load_snapshot_state(&self, input: impl io::Read + Send) -> Result<()> {
        let inner = &self.inner;
        inner.host.snapshot_reset()?;
        let caps = lock(&inner.shared).caps.clone();
        let mut f = StreamReader::from_reader(input);
        let mut vm = lock(&inner.savevm);
        vm.capabilities = caps;
        vm.load_state(&mut f)?;
        inner.host.snapshot_loaded();
        Ok(())
    }

    /// `query-migrate-capabilities`.
    pub fn query_capabilities(&self) -> Vec<MigrationCapabilityStatus> {
        let s = lock(&self.inner.shared);
        MigrationCapability::ALL
            .iter()
            .map(|&c| MigrationCapabilityStatus { capability: c, state: s.caps.contains(&c) })
            .collect()
    }

    /// `migrate-set-capabilities`.
    pub fn set_capabilities(&self, caps: &[MigrationCapabilityStatus]) -> Result<()> {
        // The blocks for the write tracking check. The state lock comes second everywhere, and
        // a busy entry list belongs to a load or a save, which the checks below refuse anyway.
        let blocks = self.inner.savevm.try_lock().map(|vm| vm.ram_blocks()).unwrap_or_default();
        let mut s = lock(&self.inner.shared);
        if is_running_state(s.state) {
            bail!("There's a migration process in progress");
        }
        let mut new = s.caps.clone();
        for c in caps {
            new.retain(|&x| x != c.capability);
            if c.state {
                new.push(c.capability);
            }
        }
        // migrate_caps_check(): the destination needs userfaultfd for postcopy.
        let postcopy = MigrationCapability::PostcopyRam;
        if new.contains(&postcopy) && !s.caps.contains(&postcopy) && self.inner.host.is_incoming() {
            postcopy::supported_by_host().map_err(|e| e.prepend("Postcopy is not supported: "))?;
        }
        if new.contains(&MigrationCapability::BackgroundSnapshot) {
            // migrate_query_write_tracking()
            if !write_tracking::available() {
                bail!("Background-snapshot is not supported by host kernel");
            }
            if !write_tracking::compatible(&blocks) {
                bail!("Background-snapshot is not compatible with guest memory configuration");
            }
            for c in BACKGROUND_SNAPSHOT_CONFLICTS {
                if new.contains(c) {
                    bail!("Background-snapshot is not compatible with {}", c.as_str());
                }
            }
        }
        let multifd = MigrationCapability::Multifd;
        if new.contains(&multifd) && !s.caps.contains(&multifd) && s.incoming_started {
            bail!("Multifd must be set before incoming starts");
        }
        if new.contains(&MigrationCapability::SwitchoverAck)
            && !new.contains(&MigrationCapability::ReturnPath)
        {
            bail!("Capability 'switchover-ack' requires capability 'return-path'");
        }
        if new.contains(&multifd) && new.contains(&MigrationCapability::Xbzrle) {
            bail!("Multifd is not compatible with xbzrle");
        }
        if new.contains(&MigrationCapability::MappedRam) {
            if new.contains(&MigrationCapability::Xbzrle) {
                bail!("Mapped-ram migration is incompatible with xbzrle");
            }
            if new.contains(&postcopy) {
                bail!("Mapped-ram migration is incompatible with postcopy");
            }
        }
        // After QEMU's own checks, so that a combination QEMU refuses fails as it does there.
        for c in &new {
            if !SUPPORTED_CAPS.contains(c) {
                bail!("Migration capability '{}' is not supported", c.as_str());
            }
        }
        s.caps = new;
        Ok(())
    }

    /// `query-migrate-parameters`.
    pub fn query_parameters(&self) -> MigrationParameters {
        let s = lock(&self.inner.shared);
        let mut p = s.params.clone();
        if !s.has_block_bitmap_mapping {
            p.block_bitmap_mapping = None;
        }
        p
    }

    /// `migrate-set-parameters`.
    pub fn set_parameters(&self, new: &MigrationParameters) -> Result<()> {
        let mut s = lock(&self.inner.shared);
        let mut p = s.params.clone();
        merge!(
            p,
            new,
            announce_initial,
            announce_max,
            announce_rounds,
            announce_step,
            throttle_trigger_threshold,
            cpu_throttle_initial,
            cpu_throttle_increment,
            cpu_throttle_tailslow,
            max_bandwidth,
            avail_switchover_bandwidth,
            downtime_limit,
            x_checkpoint_delay,
            multifd_channels,
            xbzrle_cache_size,
            max_postcopy_bandwidth,
            max_cpu_throttle,
            multifd_compression,
            multifd_zlib_level,
            multifd_qatzip_level,
            multifd_zstd_level,
            block_bitmap_mapping,
            x_vcpu_dirty_limit_period,
            vcpu_dirty_limit,
            mode,
            zero_page_detection,
            direct_io,
            x_rdma_chunk_size,
            cpr_exec_command,
        );
        // A null string means the empty string, as QEMU stores it.
        for (dst, src) in [
            (&mut p.tls_creds, &new.tls_creds),
            (&mut p.tls_hostname, &new.tls_hostname),
            (&mut p.tls_authz, &new.tls_authz),
        ] {
            if src.is_some() {
                *dst = Some(StrOrNull::S(str_or_null(src).to_string()));
            }
        }
        // migrate_mapped_ram() with migrate_multifd_compression() or migrate_tls(), which read
        // the parameters as they were before this command.
        let conflict = s.caps.contains(&MigrationCapability::MappedRam)
            && (s.params.multifd_compression.is_some_and(|c| c != MultiFDCompression::None)
                || !str_or_null(&s.params.tls_creds).is_empty());
        check_parameters(&p, conflict)?;
        if new.block_bitmap_mapping.is_some() {
            s.has_block_bitmap_mapping = true;
        }
        // xbzrle_cache_resize(): a running migration picks the new size up with its next pages.
        if let Some(size) = new.xbzrle_cache_size {
            self.inner.xbzrle_cache_size.store(size, Ordering::Relaxed);
        }
        s.params = p;
        Ok(())
    }

    /// `query-migrate`.
    pub fn query(&self) -> MigrationInfo {
        let inner = &self.inner;
        let s = lock(&inner.shared);
        let mut info = MigrationInfo::default();

        // fill_destination_migration_info()
        if let Some(list) = &s.socket_address {
            info.socket_address = Some(list.clone());
        }
        if s.incoming_state != MigrationStatus::None {
            info.status = Some(s.incoming_state);
            info.error_desc = s.error.clone();
        }

        // fill_source_migration_info()
        let blocked = inner.blockers(s.params.mode.unwrap_or_default());
        if !blocked.is_empty() {
            info.blocked_reasons = Some(blocked.to_vec());
        }
        let page_size = PAGE_SIZE as i64;
        let completed = s.state == MigrationStatus::Completed;
        match s.state {
            MigrationStatus::None => return info,
            MigrationStatus::Active
            | MigrationStatus::Cancelling
            | MigrationStatus::PreSwitchover
            | MigrationStatus::Device
            | MigrationStatus::PostcopyDevice
            | MigrationStatus::PostcopyActive
            | MigrationStatus::Failing
            | MigrationStatus::Completed => {
                // populate_time_info()
                info.setup_time = Some(s.setup_time);
                info.total_time = Some(if completed {
                    s.total_time
                } else {
                    s.start_time.map_or(0, |t| t.elapsed().as_millis() as i64)
                });
                // migrate_show_downtime()
                if completed || in_postcopy(s.state) {
                    info.downtime = Some(s.downtime);
                } else {
                    info.expected_downtime = Some(self.expected_downtime(&s));
                }
                // populate_ram_info()
                let ram = inner.ram.as_deref();
                let rd = |f: fn(&RamStats) -> &AtomicU64| {
                    ram.map_or(0, |r| f(r).load(Ordering::Relaxed))
                };
                let normal = rd(|r| &r.normal) as i64;
                let mut stats = MigrationRAMStats {
                    transferred: inner.transferred.load(Ordering::Relaxed) as i64,
                    total: rd(|r| &r.total) as i64,
                    duplicate: rd(|r| &r.duplicate) as i64,
                    normal,
                    normal_bytes: normal * page_size,
                    mbps: s.mbps,
                    dirty_sync_count: rd(|r| &r.dirty_sync_count) as i64,
                    page_size,
                    pages_per_second: s.pages_per_second,
                    precopy_bytes: rd(|r| &r.precopy_bytes),
                    downtime_bytes: rd(|r| &r.downtime_bytes),
                    postcopy_requests: s.requests.as_ref().map_or(0, |r| r.requests()) as i64,
                    postcopy_bytes: rd(|r| &r.postcopy_bytes),
                    multifd_bytes: rd(|r| &r.multifd_bytes),
                    ..Default::default()
                };
                if !completed {
                    stats.remaining = rd(|r| &r.remaining_pages) as i64 * page_size;
                    stats.dirty_pages_rate = rd(|r| &r.dirty_pages_rate) as i64;
                    // populate_global_info()
                    info.remaining = Some(s.pending_bytes);
                }
                info.ram = Some(stats);
                if s.caps.contains(&MigrationCapability::Xbzrle) {
                    let c = ram.map(|r| &r.xbzrle);
                    let rd = |f: fn(&crate::xbzrle::XbzrleCounters) -> &AtomicU64| {
                        c.map_or(0, |c| f(c).load(Ordering::Relaxed)) as i64
                    };
                    info.xbzrle_cache = Some(XBZRLECacheStats {
                        cache_size: s.params.xbzrle_cache_size.unwrap_or(XBZRLE_CACHE_SIZE),
                        bytes: rd(|c| &c.bytes),
                        pages: rd(|c| &c.pages),
                        cache_miss: rd(|c| &c.cache_miss),
                        cache_miss_rate: c.map_or(0.0, |c| c.cache_miss_rate()),
                        encoding_rate: c.map_or(0.0, |c| c.encoding_rate()),
                        overflow: rd(|c| &c.overflow),
                    });
                }
            }
            _ => {}
        }
        info.status = Some(s.state);
        if s.error.is_some() {
            info.error_desc = s.error.clone();
        }
        info
    }

    /// `migration_downtime_calc_expected()`.
    fn expected_downtime(&self, s: &Shared) -> i64 {
        let syncs =
            self.inner.ram.as_ref().map_or(0, |r| r.dirty_sync_count.load(Ordering::Relaxed));
        let limit = s.params.downtime_limit.unwrap_or(300) as i64;
        if syncs <= 1 {
            return limit;
        }
        let bw = switchover_bw(s);
        let ms = s.dirty_bytes_last_sync as f64 / bw * 1000.0;
        if ms.is_finite() { ms as i64 } else { i64::MAX }
    }

    /// `migrate`: connects to `uri` or the main channel of `channels` and starts the
    /// migration thread. With `cpr-transfer` the CPR state goes to the `cpr` channel first, and
    /// the main channel connects once the destination closed that one.
    pub fn migrate(&self, uri: Option<&str>, channels: Option<&[MigrationChannel]>) -> Result<()> {
        let inner = &self.inner;
        let mode = lock(&inner.shared).params.mode.unwrap_or_default();
        let (addr, cpr_addr) = parse_input_cpr(uri, channels, mode == MigMode::CprTransfer)?;
        {
            let s = lock(&inner.shared);
            transport_compatible(&addr, &s.caps, s.params.direct_io == Some(true), mode)?;
        }
        {
            // migrate_prepare()
            let s = lock(&inner.shared);
            if is_running_state(s.state) {
                bail!("There's a migration process in progress");
            }
            if inner.host.is_incoming() {
                bail!("Guest is waiting for an incoming migration");
            }
            if inner.host.is_postmigrate() {
                bail!("Can't migrate the vm that was paused due to previous migration");
            }
            if mode == MigMode::CprExec && s.params.cpr_exec_command.is_none() {
                bail!("cpr-exec mode requires setting cpr-exec-command");
            }
            // migration_is_blocked(): the blocker added last.
            if let Some(why) = inner.blockers(mode).last() {
                bail!("{why}");
            }
            if s.caps.contains(&MigrationCapability::MappedRam) {
                if !str_or_null(&s.params.tls_creds).is_empty() {
                    bail!("Cannot use TLS with mapped-ram");
                }
                if s.params.multifd_compression.is_some_and(|c| c != MultiFDCompression::None) {
                    bail!("Cannot use compression with mapped-ram");
                }
            }
            if !str_or_null(&s.params.tls_creds).is_empty() {
                bail!("TLS migration is not supported");
            }
            // migrate_mode_is_cpr()
            if mode != MigMode::Normal {
                let conflict = if s.caps.contains(&MigrationCapability::PostcopyRam)
                    || s.caps.contains(&MigrationCapability::DirtyBitmaps)
                {
                    Some("postcopy")
                } else if s.caps.contains(&MigrationCapability::BackgroundSnapshot) {
                    Some("background snapshot")
                } else if s.caps.contains(&MigrationCapability::XColo) {
                    Some("COLO")
                } else {
                    None
                };
                if let Some(c) = conflict {
                    bail!("Cannot use {} with CPR", c);
                }
            }
            if mode == MigMode::CprExec {
                bail!("cpr-exec is not supported by ruvm yet");
            }
        }
        // The thread takes the state lock on its way out, so it must not be held here.
        let thread = lock(&inner.shared).thread.take();
        if let Some(t) = thread {
            let _ = t.join();
        }
        // migrate_init()
        {
            let mut s = lock(&inner.shared);
            s.error = None;
            s.start_time = Some(Instant::now());
            s.setup_time = 0;
            s.total_time = 0;
            s.downtime = 0;
            s.mbps = 0.0;
            s.pages_per_second = 0;
            s.pending_bytes = 0;
            s.dirty_bytes_last_sync = 0;
            s.state = MigrationStatus::None;
        }
        inner.cancel.store(false, Ordering::Relaxed);
        inner.resumed.store(false, Ordering::Relaxed);
        inner.transferred.store(0, Ordering::Relaxed);
        if let Some(r) = &inner.ram {
            r.reset();
        }
        inner.start_postcopy.store(false, Ordering::Relaxed);
        inner.set_state(Some(MigrationStatus::None), MigrationStatus::Setup);

        let Some(cpr_addr) = cpr_addr.filter(|_| mode == MigMode::CprTransfer) else {
            // socket_start_outgoing_migration() connects a TCP or Unix socket in the background,
            // so a destination that is not there fails the migration but not the command.
            let in_background = matches!(
                &addr.u,
                MigrationAddressU::Socket(SocketAddress {
                    u: SocketAddressU::Inet(_) | SocketAddressU::Unix(_)
                })
            );
            let out = match connect_outgoing(inner, addr, mode) {
                Ok(out) => out,
                Err(e) => {
                    connect_error(inner, &e);
                    return if in_background { Ok(()) } else { Err(e) };
                }
            };
            let thread_inner = inner.clone();
            let handle = std::thread::Builder::new()
                .name(out.thread_name().to_string())
                .spawn(move || out.run(thread_inner))
                .map_err(|e| Error::from_io("Failed to create migration thread", e))?;
            lock(&inner.shared).thread = Some(handle);
            return Ok(());
        };
        // cpr_state_save()
        if let Err(e) = cpr::state_save(&cpr_addr) {
            connect_error(inner, &e);
            return Err(e);
        }
        // cpr_transfer_add_hup_watch(): the destination closes the CPR socket once it listens on
        // the main channel. Until then the migration only waits, and a cancel ends it. The
        // thread that waits goes on as the migration thread.
        let thread_inner = inner.clone();
        let handle = std::thread::Builder::new()
            .name("live_migration".to_string())
            .spawn(move || {
                let hup = cpr::wait_hup(&thread_inner.cancel);
                cpr::state_close();
                if !hup {
                    // migration_cancel() calls migration_cleanup() in this window.
                    thread_inner
                        .set_state(Some(MigrationStatus::Cancelling), MigrationStatus::Cancelled);
                    return;
                }
                // migration_connect_outgoing_cb()
                match connect_outgoing(&thread_inner, addr, mode) {
                    Ok(out) => out.run(thread_inner),
                    Err(e) => connect_error(&thread_inner, &e),
                }
            })
            .map_err(|e| Error::from_io("Failed to create migration thread", e))?;
        lock(&inner.shared).thread = Some(handle);
        Ok(())
    }

    /// `migrate_cancel`.
    pub fn cancel(&self) -> Result<()> {
        let inner = &self.inner;
        let mut s = lock(&inner.shared);
        if !is_running_state(s.state) {
            return Ok(());
        }
        if in_postcopy(s.state) {
            // The guest runs on the destination now; neither side can go back.
            bail!("Postcopy migration in progress, cannot cancel.");
        }
        let old = s.state;
        s.state = MigrationStatus::Cancelling;
        let events = inner.events(&s);
        drop(s);
        inner.cancel.store(true, Ordering::Relaxed);
        if old == MigrationStatus::PreSwitchover {
            inner.resumed.store(true, Ordering::Relaxed);
            inner.resume.notify_all();
        }
        if events {
            inner.host.status_event(MigrationStatus::Cancelling);
        }
        Ok(())
    }

    /// `migrate-start-postcopy`: switch the running migration to postcopy as soon as it can.
    pub fn start_postcopy(&self) -> Result<()> {
        let s = lock(&self.inner.shared);
        if !s.caps.contains(&MigrationCapability::PostcopyRam) {
            bail!("Enable postcopy with migrate_set_capability before the start of migration");
        }
        if s.state == MigrationStatus::None {
            bail!("Postcopy must be started after migration has been started");
        }
        // A migration that already finished is not an error, since that would race with the
        // command.
        self.inner.start_postcopy.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// `migrate-continue`: lets a migration paused by `pause-before-switchover` go on.
    pub fn continue_from(&self, state: MigrationStatus) -> Result<()> {
        let cur = self.inner.state();
        if cur != state {
            bail!("Migration not in expected state: {}", cur.as_str());
        }
        let _g = lock(&self.inner.shared);
        self.inner.resumed.store(true, Ordering::Relaxed);
        self.inner.resume.notify_all();
        Ok(())
    }

    /// Waits for the outgoing thread to end, for tests and for `-incoming` style tools.
    pub fn join(&self) {
        let t = lock(&self.inner.shared).thread.take();
        if let Some(t) = t {
            let _ = t.join();
        }
        let t = lock(&self.inner.shared).incoming_thread.take();
        if let Some(t) = t {
            let _ = t.join();
        }
    }

    /// `migrate-incoming` and `-incoming`: listens on `uri` or the main channel of `channels`
    /// and loads the first migration that comes in, in its own thread.
    pub fn incoming(
        &self,
        uri: Option<&str>,
        channels: Option<&[MigrationChannel]>,
        exit_on_error: bool,
    ) -> Result<()> {
        let inner = &self.inner;
        if lock(&inner.shared).incoming_started {
            bail!("The incoming migration has already been started");
        }
        if !inner.host.is_incoming() {
            bail!("'-incoming' was not specified on the command line");
        }
        let addr = parse_input(uri, channels)?;
        let direct_io = {
            let s = lock(&inner.shared);
            let direct_io = s.params.direct_io == Some(true);
            transport_compatible(&addr, &s.caps, direct_io, s.params.mode.unwrap_or_default())?;
            // migrate_direct_io()
            direct_io && s.caps.contains(&MigrationCapability::MappedRam)
        };
        let state = lock(&inner.shared).incoming_state;
        if state != MigrationStatus::None {
            bail!("Illegal migration incoming state: {}", state.as_str());
        }
        let fds = lock(&inner.fds).clone();
        let mut listener = Channel::listen(&addr, fds.as_deref())?;
        // Closing the CPR socket tells a cpr-transfer source that this side listens now.
        cpr::state_close();
        {
            let mut s = lock(&inner.shared);
            s.incoming_started = true;
            s.socket_address = listener.local_addr().map(|a| vec![a]);
        }
        inner.set_incoming_state(MigrationStatus::Setup);
        let thread_inner = inner.clone();
        let handle = std::thread::Builder::new()
            .name("mig/dst/main".to_string())
            .spawn(move || {
                let mf = incoming_multifd(&thread_inner);
                // A file has nothing to accept: the multifd channels open it once more each.
                let file = listener.file();
                let wanted = if file.is_some() { None } else { mf.as_ref().map(|m| m.0) };
                let chans = multifd::accept_channels(&mut listener, wanted);
                drop(listener);
                let res = chans.and_then(|ch| {
                    thread_inner.set_incoming_state(MigrationStatus::Active);
                    let mut f = StreamReader::from_reader(ch.main);
                    let mut vm = lock(&thread_inner.savevm);
                    vm.capabilities = lock(&thread_inner.shared).caps.clone();
                    let recv = match (&mf, &file) {
                        (Some(((n, _), _)), Some(fc)) => {
                            // file_recv_channel_create()
                            let mut files = Vec::with_capacity(usize::from(*n));
                            for _ in 0..*n {
                                files.push(open_file(fc.path(), false, direct_io)?);
                            }
                            Some(MultifdRecv::start_file(files)?)
                        }
                        (Some((_, zlib)), None) => {
                            let cfg = RecvConfig {
                                blocks: vm.ram_blocks(),
                                droppable: vm.droppable_blocks(),
                                entries: vm
                                    .entries()
                                    .map(|(id, inst, _)| (id.to_string(), inst))
                                    .collect(),
                                zlib: *zlib,
                                postcopy_ram: vm.postcopy_ram(),
                            };
                            Some(MultifdRecv::start(ch.multifd, cfg)?)
                        }
                        (None, _) => None,
                    };
                    let hooks = Hooks(&thread_inner);
                    let opts = LoadOptions {
                        socket: ch.socket,
                        hooks: Some(&hooks),
                        multifd: recv.clone(),
                        file: file.clone(),
                    };
                    let ret = vm.load_state_with(&mut f, opts);
                    let errno = f.get_error();
                    // The first error wins, as with migrate_set_error(): when a channel failed,
                    // the main stream usually failed only because the source gave up.
                    let channel_err = recv.as_ref().and_then(|r| r.error());
                    if let Some(r) = recv {
                        r.shutdown();
                    }
                    if let (Err(e), Some(ce)) = (&ret, channel_err) {
                        error_report(e.message());
                        return Err(ce);
                    }
                    ret.map_err(|e| {
                        // A failure in postcopy has its own message.
                        if lock(&thread_inner.shared).incoming_state == MigrationStatus::Failed {
                            return e;
                        }
                        let why = if errno < 0 { "Input/output error" } else { "Invalid argument" };
                        e.prepend(format!("load of migration failed: {why}: "))
                    })
                });
                incoming_finish(&thread_inner, res, exit_on_error);
            })
            .map_err(|e| Error::from_io("Failed to create incoming migration thread", e))?;
        lock(&inner.shared).incoming_thread = Some(handle);
        Ok(())
    }

    /// The address the incoming side listens on, once it does.
    pub fn incoming_address(&self) -> Option<Vec<SocketAddress>> {
        lock(&self.inner.shared).socket_address.clone()
    }
}

/// The multifd setup of the incoming side, `((channels, uuid), zlib)`, when the capability is on.
fn incoming_multifd(inner: &Inner) -> Option<((u8, [u8; 16]), bool)> {
    let s = lock(&inner.shared);
    if !s.caps.contains(&MigrationCapability::Multifd) {
        return None;
    }
    let n = s.params.multifd_channels.unwrap_or(2);
    let zlib = s.params.multifd_compression == Some(MultiFDCompression::Zlib);
    drop(s);
    let uuid = lock(&inner.savevm).config().uuid.unwrap_or([0; 16]);
    Some(((n, uuid), zlib))
}

fn incoming_finish(inner: &Inner, res: Result<LoadInfo>, exit_on_error: bool) {
    match res {
        Ok(info) => {
            // In postcopy the guest started on MIG_CMD_POSTCOPY_RUN already.
            if !info.postcopy {
                inner.host.incoming_done();
            }
            inner.set_incoming_state(MigrationStatus::Completed);
        }
        Err(e) => {
            if lock(&inner.shared).incoming_state != MigrationStatus::Failed {
                inner.set_incoming_state(MigrationStatus::Failed);
            }
            inner.set_error(&e);
            inner.host.incoming_failed(&e, exit_on_error);
        }
    }
    // migration_incoming_state_destroy(): query-migrate no longer shows where it listened.
    lock(&inner.shared).socket_address = None;
}

/// `migration_get_switchover_bw()`, in bytes per second.
fn switchover_bw(s: &Shared) -> f64 {
    let avail = s.params.avail_switchover_bandwidth.unwrap_or(0);
    if avail > 0 { avail as f64 } else { s.mbps * 1000.0 * 1000.0 / 8.0 }
}

struct Thread<'a> {
    inner: &'a Inner,
    f: QemuFile<'static>,
    rp: Option<RpHandle>,
    iteration_start: Instant,
    iteration_bytes: u64,
    iteration_pages: u64,
    threshold: u64,
    downtime_start: Instant,
    was_running: bool,
    stopped: bool,
    // The guest may run on the destination: from `postcopy-active` on, the source must not
    // start it again.
    dest_running: bool,
    // The downtime is known already, as with a background snapshot that resumed the guest.
    downtime_ended: bool,
    multifd: Option<Arc<MultifdSend>>,
    // `migrate_mode_is_cpr()`: the guest stopped before the thread started.
    cpr: bool,
}

impl Thread<'_> {
    /// `ram_get_total_transferred_pages()`.
    fn pages_sent(&self) -> u64 {
        self.inner.ram.as_ref().map_or(0, |r| {
            r.normal.load(Ordering::Relaxed)
                + r.duplicate.load(Ordering::Relaxed)
                + r.xbzrle.pages.load(Ordering::Relaxed)
        })
    }

    /// `migration_transferred_bytes()`: the main stream and the multifd channels.
    fn sent(&self) -> u64 {
        self.f.transferred() + self.multifd.as_ref().map_or(0, |m| m.bytes())
    }

    /// Tells `ram_transferred_add()` whether the guest runs, so that what goes out next counts
    /// as precopy bytes or not.
    fn note_runstate(&self) {
        if let Some(r) = &self.inner.ram {
            r.guest_running.store(self.inner.host.is_running(), Ordering::Relaxed);
        }
    }

    fn publish(&self) {
        self.inner.transferred.store(self.sent(), Ordering::Relaxed);
    }

    fn check_cancel(&self) -> Result<()> {
        if self.inner.cancel.load(Ordering::Relaxed) {
            bail!("migration cancelled");
        }
        Ok(())
    }

    fn requests(&self) -> Option<&PageRequests> {
        self.rp.as_ref().and_then(|rp| rp.requests.as_deref())
    }

    /// `migration_update_counters()`, at the end of a window.
    fn update_counters(&mut self, now: Instant) {
        let spent = now.duration_since(self.iteration_start).as_millis() as u64;
        if spent < BUFFER_DELAY {
            return;
        }
        let bytes = self.sent();
        let pages = self.pages_sent();
        let transferred = bytes - self.iteration_bytes;
        let mut s = lock(&self.inner.shared);
        s.mbps = (transferred as f64 * 8.0) / (spent as f64 / 1000.0) / 1000.0 / 1000.0;
        let bw_per_ms = switchover_bw(&s) / 1000.0;
        self.threshold = (bw_per_ms * s.params.downtime_limit.unwrap_or(300) as f64) as u64;
        s.pages_per_second =
            ((pages - self.iteration_pages) as f64 / (spent as f64 / 1000.0)) as u64;
        drop(s);
        self.iteration_start = now;
        self.iteration_bytes = bytes;
        self.iteration_pages = pages;
    }

    /// The bytes a window may carry, `max-bandwidth / XFER_LIMIT_RATIO`, and after the
    /// switchover to postcopy `max-postcopy-bandwidth / XFER_LIMIT_RATIO`. 0 means no limit.
    fn window_limit(&self, postcopy: bool) -> u64 {
        let s = lock(&self.inner.shared);
        let bw = if postcopy { s.params.max_postcopy_bandwidth } else { s.params.max_bandwidth };
        bw.unwrap_or(0) / XFER_LIMIT_RATIO
    }

    fn run(&mut self, vm: &mut SaveVm) -> Result<()> {
        let setup_start = Instant::now();
        self.note_runstate();
        vm.capabilities = lock(&self.inner.shared).caps.clone();
        vm.save_header(&mut self.f)?;
        if self.rp.is_some() {
            // Have the destination open its end, and ping it so that the traces line up.
            vm.send_open_return_path(&mut self.f)?;
            vm.send_ping(&mut self.f, 1)?;
        }
        if vm.postcopy_ram() {
            vm.send_postcopy_advise(&mut self.f)?;
        }
        vm.save_setup(&mut self.f)?;
        self.publish();
        if !self.inner.set_state(Some(MigrationStatus::Setup), MigrationStatus::Active) {
            self.check_cancel()?;
        }
        lock(&self.inner.shared).setup_time = setup_start.elapsed().as_millis() as i64;
        self.iteration_start = Instant::now();
        self.iteration_bytes = self.sent();
        self.iteration_pages = self.pages_sent();

        // Whether a page request came in while the thread waited for the next window.
        let mut urgent = false;
        loop {
            self.check_cancel()?;
            // The guest may have been stopped or started from the monitor in between.
            self.note_runstate();
            let state = self.inner.state();
            let postcopy = in_postcopy(state);
            let limit = self.window_limit(postcopy);
            let used = self.sent() - self.iteration_bytes;
            if urgent || limit == 0 || used < limit {
                // Over the limit, an urgent iteration only sends what was asked for.
                let budget = match limit {
                    0 => UNLIMITED_CHUNK,
                    l if used < l => l - used,
                    _ => 1,
                };
                let done = if postcopy {
                    self.postcopy_iteration(vm, state, budget)?
                } else {
                    self.precopy_iteration(vm, budget)?
                };
                if done {
                    return Ok(());
                }
                self.publish();
            }
            urgent = false;
            let now = Instant::now();
            let used = self.sent() - self.iteration_bytes;
            let window_end = self.iteration_start + Duration::from_millis(BUFFER_DELAY);
            if limit != 0 && used >= limit && now < window_end {
                // migration_rate_limit(): wait for the next window, or in postcopy until the
                // destination asks for a page.
                match self.requests() {
                    Some(r) if postcopy => {
                        r.wait(window_end - now);
                        urgent = Instant::now() < window_end;
                    }
                    _ => std::thread::sleep(window_end - now),
                }
            }
            self.update_counters(Instant::now());
        }
    }

    /// `migration_can_switchover()`: whether the destination has acknowledged the switchover,
    /// when switchover-ack asks it to.
    fn can_switchover(&self) -> bool {
        if !lock(&self.inner.shared).caps.contains(&MigrationCapability::SwitchoverAck) {
            return true;
        }
        // There is no reason to wait for the acknowledgement when the guest is stopped.
        if !self.inner.host.is_running() {
            return true;
        }
        self.rp
            .as_ref()
            .is_none_or(|rp| rp.state.switchover_ack_pending.load(Ordering::Acquire) == 0)
    }

    /// `migration_iteration_run()` before the switchover. Returns true once the migration is
    /// complete.
    fn precopy_iteration(&mut self, vm: &mut SaveVm, budget: u64) -> Result<bool> {
        // The bytes that must go before the guest stops, and those that can follow in
        // postcopy.
        let (mut pre, mut post) = vm.pending_split(false);
        if pre + post <= self.threshold || pre == 0 {
            // migration_iteration_go_next()
            (pre, post) = vm.pending_split(true);
            let (events, pass) = {
                let mut s = lock(&self.inner.shared);
                s.dirty_bytes_last_sync = pre + post;
                // The count is of iterations, which goes on even when no RAM is sent.
                let pass = self
                    .inner
                    .ram
                    .as_ref()
                    .map_or(0, |r| r.dirty_sync_count.fetch_add(1, Ordering::Relaxed) + 1);
                (self.inner.events(&s), pass)
            };
            if events {
                self.inner.host.pass_event(pass as i64);
            }
        }
        let total = pre + post;
        lock(&self.inner.shared).pending_bytes = total;
        let can_switchover = self.can_switchover();
        // postcopy_should_start()
        if can_switchover
            && pre <= self.threshold
            && self.inner.start_postcopy.load(Ordering::Relaxed)
        {
            if let Err(e) = self.postcopy_start(vm) {
                error_report(e.message());
                return Err(e);
            }
            return Ok(false);
        }
        if can_switchover && total <= self.threshold {
            self.complete(vm)?;
            return Ok(true);
        }
        vm.save_iterate(&mut self.f, budget, false)?;
        Ok(false)
    }

    /// `migration_iteration_run()` after the switchover to postcopy. Returns true once the
    /// migration is complete.
    fn postcopy_iteration(
        &mut self,
        vm: &mut SaveVm,
        state: MigrationStatus,
        budget: u64,
    ) -> Result<bool> {
        let pending = vm.pending(false);
        lock(&self.inner.shared).pending_bytes = pending;
        let complete_ready = pending == 0;
        if state == MigrationStatus::PostcopyDevice {
            let rp = self.rp.as_ref().expect("postcopy-device needs a return path");
            if rp.state.package_loaded.load(Ordering::Acquire) || complete_ready {
                // Before completing, the destination must have loaded the package.
                while !rp.state.package_loaded.load(Ordering::Acquire) {
                    if self.inner.has_error() {
                        self.inner.set_state(
                            Some(MigrationStatus::PostcopyDevice),
                            MigrationStatus::Failing,
                        );
                        return Err(self.inner.error());
                    }
                    match &rp.requests {
                        Some(r) => r.wait(Duration::from_millis(BUFFER_DELAY)),
                        None => std::thread::sleep(Duration::from_millis(10)),
                    }
                }
                self.inner.set_state(
                    Some(MigrationStatus::PostcopyDevice),
                    MigrationStatus::PostcopyActive,
                );
                self.dest_running = true;
            }
        }
        if complete_ready {
            self.complete_postcopy(vm)?;
            return Ok(true);
        }
        vm.save_iterate(&mut self.f, budget, true)?;
        Ok(false)
    }

    /// `migration_stop_vm()`.
    fn stop_vm(&mut self) -> Result<()> {
        self.downtime_start = Instant::now();
        self.was_running = self.inner.host.is_running();
        self.inner.host.stop_for_switchover()?;
        self.stopped = true;
        self.note_runstate();
        Ok(())
    }

    /// `migration_switchover_start()`: `pause-before-switchover`, then the `device` state.
    fn switchover_start(&mut self, vm: &mut SaveVm) -> Result<()> {
        // migration_switchover_prepare()
        let interrupted = || Error::generic("Switchover is interrupted");
        let pause =
            lock(&self.inner.shared).caps.contains(&MigrationCapability::PauseBeforeSwitchover);
        self.check_cancel().map_err(|_| interrupted())?;
        if pause {
            self.inner.set_state(None, MigrationStatus::PreSwitchover);
            let mut g = lock(&self.inner.shared);
            while !self.inner.resumed.load(Ordering::Relaxed) {
                g = self.inner.resume.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            drop(g);
            self.check_cancel().map_err(|_| interrupted())?;
            self.inner.set_state(Some(MigrationStatus::PreSwitchover), MigrationStatus::Device);
        } else {
            self.inner.set_state(None, MigrationStatus::Device);
        }
        vm.send_switchover_start(&mut self.f)
    }

    /// `postcopy_start()`: stops the guest and hands it over to the destination with what
    /// must go first; the rest of RAM follows while the destination runs.
    fn postcopy_start(&mut self, vm: &mut SaveVm) -> Result<()> {
        vm.postcopy_prepare(&mut self.f)?;
        self.stop_vm().map_err(|e| e.prepend("postcopy_start: Failed to stop the VM: "))?;
        self.switchover_start(vm)?;
        vm.complete_precopy_iterable(&mut self.f, true)
            .map_err(|_| Error::generic("Postcopy save non-postcopiable iterables failed"))?;
        // The pages the destination may have received and that got dirty since.
        if vm.postcopy_ram() {
            vm.send_postcopy_discard(&mut self.f)?;
            vm.send_ping(&mut self.f, 2)?;
        }
        // The destination has to read all of the device state before it loads any of it, so
        // that the channel is free for the pages that loading may fault on.
        let package = vm.postcopy_package(self.rp.is_some())?;
        vm.send_packaged(&mut self.f, &package)?;
        {
            // migration_downtime_end()
            let mut s = lock(&self.inner.shared);
            if s.downtime == 0 {
                s.downtime = self.downtime_start.elapsed().as_millis() as i64;
            }
        }
        if vm.postcopy_ram() {
            vm.send_ping(&mut self.f, 4)
                .map_err(|e| e.prepend("postcopy_start: Migration stream error: "))?;
        }
        self.publish();
        let next = if self.rp.is_some() {
            MigrationStatus::PostcopyDevice
        } else {
            self.dest_running = true;
            MigrationStatus::PostcopyActive
        };
        self.inner.set_state(Some(MigrationStatus::Device), next);
        // The window starts over with the postcopy bandwidth.
        self.iteration_start = Instant::now();
        self.iteration_bytes = self.sent();
        Ok(())
    }

    /// `migration_completion()` for precopy.
    fn complete(&mut self, vm: &mut SaveVm) -> Result<()> {
        if !self.cpr {
            self.stop_vm().map_err(|e| e.prepend("Failed to stop the VM: "))?;
        }
        self.switchover_start(vm)?;
        vm.save_complete(&mut self.f)?;
        self.publish();
        self.stop_return_path()?;
        self.completion_end();
        Ok(())
    }

    /// `migration_completion()` in postcopy.
    fn complete_postcopy(&mut self, vm: &mut SaveVm) -> Result<()> {
        vm.complete_postcopy(&mut self.f)?;
        self.publish();
        self.stop_return_path()?;
        self.completion_end();
        Ok(())
    }

    /// `stop_return_path_thread_on_source()`: waits for the destination to close the return
    /// path, which it does once it loaded everything, and fails if anything went wrong.
    fn stop_return_path(&mut self) -> Result<()> {
        let Some(rp) = self.rp.as_mut() else { return Ok(()) };
        if self.inner.has_error() {
            rp.ctl.shutdown();
        }
        if let Some(t) = rp.thread.take() {
            let _ = t.join();
        }
        if self.inner.has_error() {
            return Err(self.inner.error());
        }
        Ok(())
    }

    /// `migration_completion_end()`.
    fn completion_end(&mut self) {
        let bytes = self.sent();
        let mut s = lock(&self.inner.shared);
        // migration_downtime_end(): postcopy and background snapshots set it already.
        if s.downtime == 0 && !self.downtime_ended {
            s.downtime = self.downtime_start.elapsed().as_millis() as i64;
        }
        s.total_time = s.start_time.map_or(0, |t| t.elapsed().as_millis() as i64);
        let transfer_time = s.total_time - s.setup_time;
        if transfer_time > 0 {
            s.mbps = (bytes as f64 * 8.0) / transfer_time as f64 / 1000.0;
        }
        s.pending_bytes = 0;
        drop(s);
        self.inner.set_state(None, MigrationStatus::Completed);
    }
}

impl Drop for RpHandle {
    fn drop(&mut self) {
        // After a failure the destination may never close its end; wake the thread up.
        if let Some(t) = self.thread.take() {
            self.ctl.shutdown();
            let _ = t.join();
        }
    }
}

/// `multifd_send_setup()`: connects the channels and starts their threads. The channels connect
/// one after the other here rather than in the background as in QEMU. `file` is the migration
/// file, whose channels open it once more each.
fn multifd_setup(
    inner: &Inner,
    addr: &MigrationAddr,
    file: Option<&Arc<FileChannel>>,
    vm: &SaveVm,
) -> Result<SaveParams> {
    let (caps, params) = {
        let s = lock(&inner.shared);
        (s.caps.clone(), s.params.clone())
    };
    let zero_page_detection = params.zero_page_detection.unwrap_or(ZeroPageDetection::Multifd);
    let xbzrle_cache_size = caps.contains(&MigrationCapability::Xbzrle).then(|| {
        let size = params.xbzrle_cache_size.unwrap_or(XBZRLE_CACHE_SIZE);
        inner.xbzrle_cache_size.store(size, Ordering::Relaxed);
        inner.xbzrle_cache_size.clone()
    });
    let mapped_ram = caps.contains(&MigrationCapability::MappedRam);
    // migrate_ram_is_ignored(): the next process maps the same memory.
    let ignore_ram = params.mode == Some(MigMode::CprTransfer);
    if !caps.contains(&MigrationCapability::Multifd) {
        return Ok(SaveParams {
            multifd: None,
            zero_page_detection,
            xbzrle_cache_size,
            mapped_ram,
            background_snapshot: caps.contains(&MigrationCapability::BackgroundSnapshot),
            ignore_ram,
        });
    }
    let channels = params.multifd_channels.unwrap_or(2);
    let compression = match params.multifd_compression.unwrap_or(MultiFDCompression::None) {
        MultiFDCompression::Zlib => {
            Compression::Zlib(params.multifd_zlib_level.unwrap_or(1) as u32)
        }
        _ => Compression::None,
    };
    let p = MultifdParams {
        channels,
        compression,
        zero_pages: zero_page_detection == ZeroPageDetection::Multifd,
        uuid: vm.config().uuid.unwrap_or([0; 16]),
    };
    let stats = inner.ram.clone().unwrap_or_default();
    let m = match file {
        Some(fc) => {
            // file_send_channel_create(), with O_DIRECT for migrate_direct_io().
            let direct = params.direct_io == Some(true) && mapped_ram;
            let mut files = Vec::with_capacity(usize::from(channels));
            for _ in 0..channels {
                files.push(open_file(fc.path(), true, direct)?);
            }
            MultifdSend::start_file(files, &p, stats, inner.cancel.clone())?
        }
        None => {
            let mut sockets = Vec::with_capacity(usize::from(channels));
            for _ in 0..channels {
                sockets.push(Channel::connect_raw(addr)?);
            }
            MultifdSend::start(sockets, &p, stats, inner.cancel.clone())?
        }
    };
    Ok(SaveParams {
        multifd: Some(m),
        zero_page_detection,
        xbzrle_cache_size,
        mapped_ram,
        background_snapshot: false,
        ignore_ram,
    })
}

fn migration_thread(
    inner: Arc<Inner>,
    sink: CancelWriter,
    file: Option<Arc<FileChannel>>,
    rp: Option<RpHandle>,
    addr: MigrationAddr,
    cpr: bool,
) {
    let mut t = Thread {
        inner: &inner,
        f: QemuFile::with_file(sink, file.clone()),
        rp,
        iteration_start: Instant::now(),
        iteration_bytes: 0,
        iteration_pages: 0,
        threshold: 0,
        downtime_start: Instant::now(),
        was_running: false,
        stopped: false,
        dest_running: false,
        downtime_ended: false,
        multifd: None,
        cpr,
    };
    // migration_connect(): in the CPR modes the guest stops before anything goes out.
    let stopped = if cpr {
        t.stop_vm().map_err(|e| e.prepend("migration_stop_vm failed, error "))
    } else {
        Ok(())
    };
    let savevm = inner.savevm.clone();
    let res = {
        let mut vm = lock(&savevm);
        let res = stopped.and_then(|()| {
            multifd_setup(&inner, &addr, file.as_ref(), &vm).and_then(|p| {
                t.multifd = p.multifd.clone();
                vm.set_save_params(&p);
                t.run(&mut vm)
            })
        });
        // migration_iteration_finish() comes before migration_cleanup() shuts the multifd
        // channels down, so the guest shows as migrated as soon as the migration does.
        if res.is_ok() && inner.state() == MigrationStatus::Completed {
            inner.host.set_postmigrate();
        }
        vm.save_cleanup();
        if let Some(m) = &t.multifd {
            m.shutdown();
        }
        res
    };
    if let Err(e) = res {
        // A cancel shows as a failed write; keep the error of a real failure only.
        if inner.state() != MigrationStatus::Cancelling {
            inner.set_error(&e);
            inner.set_state(None, MigrationStatus::Failing);
        }
    }
    // migration_iteration_finish()
    match inner.state() {
        MigrationStatus::Completed => {}
        _ => {
            // Once the destination runs the guest, starting it here as well would run it
            // twice. QEMU pauses the migration for a recovery then; this side has none.
            if t.stopped && !t.dest_running {
                inner.host.resume_after_failure(t.was_running);
            }
        }
    }
    drop(t);
    // migration_cleanup()
    if !inner.set_state(Some(MigrationStatus::Cancelling), MigrationStatus::Cancelled) {
        inner.set_state(Some(MigrationStatus::Failing), MigrationStatus::Failed);
    }
}

/// `bg_migration_thread()`: a background snapshot. The device state is saved into a buffer
/// with the guest paused, then RAM is write-protected and the guest goes on while RAM goes out
/// in one pass, the pages the guest wants to write first. The buffered device state follows it,
/// so the stream loads like any other but holds RAM as it was at the pause.
fn bg_migration_thread(
    inner: Arc<Inner>,
    sink: CancelWriter,
    file: Option<Arc<FileChannel>>,
    addr: MigrationAddr,
) {
    let mut t = Thread {
        inner: &inner,
        f: QemuFile::with_file(sink, file.clone()),
        rp: None,
        iteration_start: Instant::now(),
        iteration_bytes: 0,
        iteration_pages: 0,
        threshold: 0,
        downtime_start: Instant::now(),
        was_running: false,
        stopped: false,
        dest_running: false,
        downtime_ended: false,
        multifd: None,
        cpr: false,
    };
    let savevm = inner.savevm.clone();
    let mut resume: Option<JoinHandle<()>> = None;
    {
        let mut vm = lock(&savevm);
        let res = multifd_setup(&inner, &addr, file.as_ref(), &vm).and_then(|p| {
            vm.set_save_params(&p);
            bg_run(&inner, &mut t, &mut vm, &mut resume)
        });
        if let Err(e) = res {
            // A cancel shows as a failed write; keep the error of a real failure only.
            if inner.state() != MigrationStatus::Cancelling {
                inner.set_error(&e);
                inner.set_state(None, MigrationStatus::Failing);
            }
        }
        // bg_migration_iteration_finish(): the protection goes first, which also wakes any
        // writer, then the guest is surely running again.
        vm.write_tracking_stop();
        vm.save_cleanup();
    }
    if let Some(h) = resume {
        let _ = h.join();
    }
    drop(t);
    // migration_cleanup()
    if !inner.set_state(Some(MigrationStatus::Cancelling), MigrationStatus::Cancelled) {
        inner.set_state(Some(MigrationStatus::Failing), MigrationStatus::Failed);
    }
}

/// The body of `bg_migration_thread()`. On an error before the guest resumes it stays paused,
/// as in QEMU.
fn bg_run(
    inner: &Arc<Inner>,
    t: &mut Thread<'_>,
    vm: &mut SaveVm,
    resume: &mut Option<JoinHandle<()>>,
) -> Result<()> {
    let setup_start = Instant::now();
    t.note_runstate();
    vm.capabilities = lock(&inner.shared).caps.clone();
    let blocks = vm.ram_blocks();
    write_tracking::prepare(&blocks);
    vm.save_header(&mut t.f)?;
    vm.save_setup(&mut t.f)?;
    t.publish();
    if !inner.set_state(Some(MigrationStatus::Setup), MigrationStatus::Active) {
        t.check_cancel()?;
    }
    lock(&inner.shared).setup_time = setup_start.elapsed().as_millis() as i64;
    t.iteration_start = Instant::now();
    t.iteration_bytes = t.sent();
    t.iteration_pages = t.pages_sent();

    // migration_stop_vm(s, RUN_STATE_PAUSED)
    t.downtime_start = Instant::now();
    t.was_running = inner.host.is_running();
    inner.host.stop_for_snapshot().map_err(|_| Error::generic("Failed to stop the VM"))?;
    t.stopped = true;
    t.note_runstate();
    let mut device = Vec::new();
    {
        let mut fb = QemuFile::new(&mut device);
        vm.save_non_iterable(&mut fb)
            .map_err(|e| e.prepend("Failed to save non-iterable devices "))?;
    }
    vm.write_tracking_start().map_err(|_| Error::generic("Failed to start write tracking"))?;

    // bg_migration_vm_start_bh(): the guest goes on next to this thread, which serves the
    // write faults it may take on the way.
    let resume_inner = inner.clone();
    let was_running = t.was_running;
    let downtime_start = t.downtime_start;
    *resume = Some(
        std::thread::Builder::new()
            .name("bg_snapshot_resume".to_string())
            .spawn(move || {
                resume_inner.host.resume_after_snapshot(was_running);
                // From here on what goes out counts as precopy bytes.
                if let Some(r) = &resume_inner.ram {
                    r.guest_running.store(resume_inner.host.is_running(), Ordering::Relaxed);
                }
                // migration_downtime_end()
                lock(&resume_inner.shared).downtime = downtime_start.elapsed().as_millis() as i64;
            })
            .map_err(|e| Error::from_io("Failed to resume the VM", e))?,
    );
    t.downtime_ended = true;

    // The pages that go out before the guest actually resumed count as downtime bytes.
    loop {
        t.check_cancel()?;
        // bg_migration_iteration_run(). QEMU's iterations end after 50 ms at the latest; a
        // smaller chunk than a migration's keeps the counters going in the same way.
        if vm.save_iterate(&mut t.f, BG_CHUNK, false)? {
            // bg_migration_completion(): RAM is out, the device state follows.
            t.f.put_buffer(&device);
            t.f.fflush()?;
            if let Some(h) = resume.take() {
                let _ = h.join();
            }
            t.publish();
            t.completion_end();
            return Ok(());
        }
        t.publish();
        t.update_counters(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Write};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use ruvm_mem::RamBlock;
    use ruvm_qapi::types::{
        InetSocketAddress, MigrationAddress, MigrationAddressU, MigrationCapability,
        MigrationCapabilityStatus, MigrationChannel, MigrationChannelType, MigrationParameters,
        MigrationStatus, MultiFDCompression, SocketAddress, SocketAddressU,
    };

    use super::{CancelWriter, Migration, MigrationHost};
    use crate::ram::{NoHooks, RamSection};
    use crate::savevm::{EntryInfo, MachineConfig, SaveVm};

    #[derive(Default)]
    struct Host {
        running: AtomicBool,
        incoming: AtomicBool,
        events: Mutex<Vec<MigrationStatus>>,
        done: AtomicBool,
        snapshot_resumed: AtomicBool,
    }

    impl MigrationHost for Host {
        fn is_running(&self) -> bool {
            self.running.load(Ordering::Relaxed)
        }
        fn is_incoming(&self) -> bool {
            self.incoming.load(Ordering::Relaxed)
        }
        fn stop_for_switchover(&self) -> ruvm_base::Result<()> {
            self.running.store(false, Ordering::Relaxed);
            Ok(())
        }
        fn resume_after_failure(&self, was_running: bool) {
            self.running.store(was_running, Ordering::Relaxed);
        }
        fn set_postmigrate(&self) {}
        fn resume_after_snapshot(&self, was_running: bool) {
            self.running.store(was_running, Ordering::Relaxed);
            self.snapshot_resumed.store(true, Ordering::Release);
        }
        fn incoming_done(&self) {
            self.done.store(true, Ordering::Relaxed);
        }
        fn incoming_failed(&self, err: &ruvm_base::Error, _exit_on_error: bool) {
            panic!("incoming failed: {}", err.message());
        }
        fn status_event(&self, status: MigrationStatus) {
            self.events.lock().unwrap().push(status);
        }
    }

    fn machine(pages: u64, host: Arc<Host>) -> (Migration, Arc<RamBlock>) {
        machine_named("pc-q35-11.1", pages, host)
    }

    fn machine_named(name: &str, pages: u64, host: Arc<Host>) -> (Migration, Arc<RamBlock>) {
        let block = Arc::new(RamBlock::new("pc.ram", pages << 12, 12).unwrap());
        let ram = RamSection::new(vec![block.clone()], NoHooks);
        let stats = ram.stats();
        let config = MachineConfig {
            name: name.to_string(),
            page_bits: 12,
            legacy_page_bits: 12,
            uuid: None,
        };
        let mut vm = SaveVm::new(config);
        vm.register_live(EntryInfo::new("ram", 4), ram);
        (Migration::new(Arc::new(Mutex::new(vm)), Some(stats), host), block)
    }

    #[test]
    fn parameters_and_capabilities() {
        let host = Arc::new(Host::default());
        let (m, _) = machine(4, host);
        let p = m.query_parameters();
        assert_eq!(p.max_bandwidth, Some(134217728));
        assert_eq!(p.downtime_limit, Some(300));
        assert_eq!(p.block_bitmap_mapping, None);
        let err = m
            .set_parameters(&MigrationParameters {
                downtime_limit: Some(3_000_000),
                ..Default::default()
            })
            .unwrap_err();
        assert_eq!(
            err.message(),
            "Option downtime-limit expects an integer in the range of 0 to (2000 * 1000) ms"
        );
        m.set_parameters(&MigrationParameters { downtime_limit: Some(50), ..Default::default() })
            .unwrap();
        assert_eq!(m.query_parameters().downtime_limit, Some(50));
        assert!(
            m.set_capabilities(&[MigrationCapabilityStatus {
                capability: MigrationCapability::ZeroCopySend,
                state: true
            }])
            .is_err()
        );
        m.set_capabilities(&[MigrationCapabilityStatus {
            capability: MigrationCapability::Events,
            state: true,
        }])
        .unwrap();
        assert!(m.capability(MigrationCapability::Events));
        assert_eq!(m.query().status, None);
    }

    #[test]
    fn expected_downtime() {
        let (m, _) = machine(4, Arc::new(Host::default()));
        let syncs = &m.inner.ram.as_ref().unwrap().dirty_sync_count;
        let mut s = super::lock(&m.inner.shared);
        s.params.downtime_limit = Some(42);
        s.dirty_bytes_last_sync = 1_000_000;
        // Before the first iteration is over there is nothing to go by but the limit.
        syncs.store(1, Ordering::Relaxed);
        assert_eq!(m.expected_downtime(&s), 42);
        // Nothing sent yet: forever.
        syncs.store(3, Ordering::Relaxed);
        assert_eq!(m.expected_downtime(&s), i64::MAX);
        // 80 Mbit/s is 10 MB/s, so 1 MB takes 100 ms.
        s.mbps = 80.0;
        assert_eq!(m.expected_downtime(&s), 100);
        // avail-switchover-bandwidth, in bytes per second, comes first.
        s.params.avail_switchover_bandwidth = Some(4_000_000);
        assert_eq!(m.expected_downtime(&s), 250);
    }

    #[test]
    fn cpr_transfer_blocks_private_ram() {
        let (m, _) = machine(4, Arc::new(Host::default()));
        assert_eq!(m.query().blocked_reasons, None);
        m.set_parameters(&MigrationParameters {
            mode: Some(super::MigMode::CprTransfer),
            ..Default::default()
        })
        .unwrap();
        let info = m.query();
        assert_eq!(info.status, None);
        assert_eq!(
            info.blocked_reasons.unwrap(),
            ["Memory region pc.ram is not compatible with CPR. share=on is required for \
              memory-backend objects, and aux-ram-share=on is required."]
        );
    }

    #[test]
    fn precopy_over_tcp() {
        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, dst_block) = machine(256, dst_host.clone());
        dst.set_capabilities(&[MigrationCapabilityStatus {
            capability: MigrationCapability::Events,
            state: true,
        }])
        .unwrap();
        dst.incoming(Some("tcp:127.0.0.1:0"), None, true).unwrap();
        let port = match &dst.incoming_address().unwrap()[0] {
            SocketAddress { u: SocketAddressU::Inet(a) } => a.port.clone(),
            other => panic!("{other:?}"),
        };

        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, src_block) = machine(256, src_host.clone());
        for page in 0..256u64 {
            if page % 3 != 0 {
                src_block.fill(page << 12, 4096, page as u8).unwrap();
            }
        }
        let channels = [MigrationChannel {
            channel_type: MigrationChannelType::Main,
            addr: MigrationAddress {
                u: MigrationAddressU::Socket(SocketAddress {
                    u: SocketAddressU::Inet(InetSocketAddress {
                        host: "127.0.0.1".to_string(),
                        port,
                        ..Default::default()
                    }),
                }),
            },
        }];
        src.migrate(None, Some(&channels)).unwrap();
        src.join();
        dst.join();

        assert_eq!(src.status(), MigrationStatus::Completed);
        assert!(!src_host.is_running());
        let info = src.query();
        assert_eq!(info.status, Some(MigrationStatus::Completed));
        let ram = info.ram.unwrap();
        assert_eq!(ram.total, 256 << 12);
        assert_eq!(ram.normal + ram.duplicate, 256);
        // Everything went out with the guest running, and nothing was left for the switchover.
        assert!(ram.precopy_bytes > 170 << 12, "{ram:?}");
        assert_eq!(ram.downtime_bytes, 0);
        assert!(info.downtime.is_some());

        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
        assert!(dst_host.done.load(Ordering::Relaxed));
        assert_eq!(
            *dst_host.events.lock().unwrap(),
            [MigrationStatus::Setup, MigrationStatus::Active, MigrationStatus::Completed]
        );
        let mut a = vec![0u8; 256 << 12];
        let mut b = vec![0u8; 256 << 12];
        src_block.read(0, &mut a).unwrap();
        dst_block.read(0, &mut b).unwrap();
        assert!(a == b);
    }

    fn caps(m: &Migration, caps: &[MigrationCapability]) {
        let list: Vec<_> = caps
            .iter()
            .map(|&c| MigrationCapabilityStatus { capability: c, state: true })
            .collect();
        m.set_capabilities(&list).unwrap();
    }

    fn port_of(m: &Migration) -> String {
        match &m.incoming_address().unwrap()[0] {
            SocketAddress { u: SocketAddressU::Inet(a) } => a.port.clone(),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn postcopy_over_tcp() {
        // The destination needs userfaultfd, which the host running the tests may not allow.
        if crate::postcopy::supported_by_host().is_err() {
            return;
        }
        const PAGES: u64 = 64;
        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, dst_block) = machine(PAGES, dst_host.clone());
        caps(&dst, &[MigrationCapability::Events, MigrationCapability::PostcopyRam]);
        dst.incoming(Some("tcp:127.0.0.1:0"), None, true).unwrap();
        let uri = format!("tcp:127.0.0.1:{}", port_of(&dst));

        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, src_block) = machine(PAGES, src_host.clone());
        for page in 0..PAGES {
            if page % 3 != 0 {
                src_block.fill(page << 12, 4096, page as u8 + 1).unwrap();
            }
        }
        let err = src.start_postcopy().unwrap_err();
        assert_eq!(
            err.message(),
            "Enable postcopy with migrate_set_capability before the start of migration"
        );
        caps(&src, &[MigrationCapability::PostcopyRam]);
        assert_eq!(
            src.start_postcopy().unwrap_err().message(),
            "Postcopy must be started after migration has been started"
        );
        // One page per window before the switchover and four after it, so that the
        // destination has to ask for the page it reads below.
        src.set_parameters(&MigrationParameters {
            max_bandwidth: Some(4096 * 10),
            max_postcopy_bandwidth: Some(4 * 4096 * 10),
            ..Default::default()
        })
        .unwrap();
        src.migrate(Some(&uri), None).unwrap();
        src.start_postcopy().unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        // The destination registers its RAM for faults before it reports postcopy.
        let postcopy = |s: MigrationStatus| {
            matches!(s, MigrationStatus::PostcopyDevice | MigrationStatus::PostcopyActive)
        };
        while !postcopy(src.status()) || !postcopy(dst.incoming_status()) {
            assert!(std::time::Instant::now() < deadline, "no postcopy: {:?}", src.query());
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(
            src.cancel().unwrap_err().message(),
            "Postcopy migration in progress, cannot cancel."
        );
        // This faults on the destination until the source sent the page.
        let last = PAGES - 2;
        let mut page = vec![0u8; 4096];
        dst_block.read(last << 12, &mut page).unwrap();
        assert!(page.iter().all(|&b| b == last as u8 + 1));
        src.join();
        dst.join();

        assert_eq!(src.status(), MigrationStatus::Completed);
        let info = src.query();
        let ram = info.ram.unwrap();
        assert!(ram.postcopy_requests >= 1, "{ram:?}");
        assert!(ram.postcopy_bytes > 0, "{ram:?}");
        assert!(info.downtime.is_some());
        assert!(!src_host.is_running());

        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
        assert!(dst_host.done.load(Ordering::Relaxed));
        assert_eq!(
            *dst_host.events.lock().unwrap(),
            [
                MigrationStatus::Setup,
                MigrationStatus::Active,
                MigrationStatus::PostcopyDevice,
                MigrationStatus::PostcopyActive,
                MigrationStatus::Completed
            ]
        );
        let mut a = vec![0u8; (PAGES << 12) as usize];
        let mut b = vec![0u8; (PAGES << 12) as usize];
        src_block.read(0, &mut a).unwrap();
        dst_block.read(0, &mut b).unwrap();
        assert!(a == b);
    }

    fn multifd_over_tcp(compression: MultiFDCompression) {
        const PAGES: u64 = 1024;
        let params = MigrationParameters {
            multifd_channels: Some(3),
            multifd_compression: Some(compression),
            ..Default::default()
        };
        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, dst_block) = machine(PAGES, dst_host.clone());
        dst.set_parameters(&params).unwrap();
        caps(&dst, &[MigrationCapability::Multifd]);
        assert_eq!(
            dst.incoming(Some("exec:true"), None, true).unwrap_err().message(),
            "Migration requires multi-channel URIs (e.g. tcp)"
        );
        dst.incoming(Some("tcp:127.0.0.1:0"), None, true).unwrap();
        let port = port_of(&dst);

        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, src_block) = machine(PAGES, src_host.clone());
        src.set_parameters(&params).unwrap();
        caps(&src, &[MigrationCapability::Multifd]);
        for page in 0..PAGES {
            if page % 3 != 0 {
                src_block.fill(page << 12, 4096, page as u8 | 1).unwrap();
            }
        }
        src.migrate(Some(&format!("tcp:127.0.0.1:{port}")), None).unwrap();
        src.join();
        dst.join();

        assert_eq!(src.status(), MigrationStatus::Completed, "{:?}", src.query().error_desc);
        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
        let ram = src.query().ram.unwrap();
        assert!(ram.multifd_bytes > PAGES / 3 * 2, "{ram:?}");
        assert!(ram.transferred as u64 > ram.multifd_bytes, "{ram:?}");
        assert_eq!(ram.normal + ram.duplicate, PAGES as i64, "{ram:?}");
        let mut a = vec![0u8; (PAGES << 12) as usize];
        let mut b = vec![0u8; (PAGES << 12) as usize];
        src_block.read(0, &mut a).unwrap();
        dst_block.read(0, &mut b).unwrap();
        assert!(a == b);

        // Off is fine at any time, on only before the destination listens.
        let set = |on| {
            dst.set_capabilities(&[MigrationCapabilityStatus {
                capability: MigrationCapability::Multifd,
                state: on,
            }])
        };
        set(false).unwrap();
        assert_eq!(set(true).unwrap_err().message(), "Multifd must be set before incoming starts");
        let err = src
            .set_capabilities(&[MigrationCapabilityStatus {
                capability: MigrationCapability::Xbzrle,
                state: true,
            }])
            .unwrap_err();
        assert_eq!(err.message(), "Multifd is not compatible with xbzrle");
    }

    #[test]
    fn multifd_nocomp_over_tcp() {
        multifd_over_tcp(MultiFDCompression::None);
    }

    #[test]
    fn multifd_zlib_over_tcp() {
        multifd_over_tcp(MultiFDCompression::Zlib);
    }

    fn mapped_ram_over_file(multifd: bool, direct_io: bool) {
        const PAGES: u64 = 600;
        let path = std::env::temp_dir()
            .join(format!("ruvm-mapped-ram-{}-{multifd}-{direct_io}.mig", std::process::id()));
        let uri = format!("file:{}", path.display());
        let mut list = vec![MigrationCapability::MappedRam];
        if multifd {
            list.push(MigrationCapability::Multifd);
        }
        let params = MigrationParameters {
            multifd_channels: Some(3),
            direct_io: Some(direct_io),
            ..Default::default()
        };

        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, src_block) = machine(PAGES, src_host.clone());
        src.set_parameters(&params).unwrap();
        caps(&src, &list);
        for page in 0..PAGES {
            if page % 3 != 0 {
                src_block.fill(page << 12, 4096, page as u8 | 1).unwrap();
            }
        }
        src.migrate(Some(&uri), None).unwrap();
        src.join();
        assert_eq!(src.status(), MigrationStatus::Completed, "{:?}", src.query().error_desc);
        let ram = src.query().ram.unwrap();
        assert_eq!(ram.normal + ram.duplicate, PAGES as i64, "{ram:?}");
        assert_eq!(ram.normal, (PAGES - PAGES.div_ceil(3)) as i64, "{ram:?}");
        if multifd {
            assert_eq!(ram.multifd_bytes, ram.normal as u64 * 4096, "{ram:?}");
        }
        // The pages have their place in the file, so it holds RAM and a little more.
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > PAGES << 12 && len < (PAGES << 12) + (4 << 20), "{len}");

        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, dst_block) = machine(PAGES, dst_host.clone());
        dst.set_parameters(&params).unwrap();
        caps(&dst, &list);
        dst.incoming(Some(&uri), None, true).unwrap();
        dst.join();
        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
        let mut a = vec![0u8; (PAGES << 12) as usize];
        let mut b = vec![0u8; (PAGES << 12) as usize];
        src_block.read(0, &mut a).unwrap();
        dst_block.read(0, &mut b).unwrap();
        assert!(a == b);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn background_snapshot() {
        if !crate::write_tracking::available() {
            eprintln!("skipped: no userfaultfd write protection here");
            return;
        }
        const PAGES: u64 = 4096;
        let path = std::env::temp_dir()
            .join(format!("ruvm-background-snapshot-{}.mig", std::process::id()));
        let uri = format!("file:{}", path.display());
        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, src_block) = machine(PAGES, src_host.clone());
        let on = |c| MigrationCapabilityStatus { capability: c, state: true };
        let err = src
            .set_capabilities(&[
                on(MigrationCapability::BackgroundSnapshot),
                on(MigrationCapability::Xbzrle),
            ])
            .unwrap_err();
        assert_eq!(err.message(), "Background-snapshot is not compatible with xbzrle");
        assert!(!src.capability(MigrationCapability::BackgroundSnapshot));
        caps(&src, &[MigrationCapability::BackgroundSnapshot]);
        let before = |page: u64| page as u8 | 1;
        for page in 0..PAGES {
            if page % 4 != 0 {
                src_block.fill(page << 12, 4096, before(page)).unwrap();
            }
        }
        // The guest writes every page once it runs again, last page first so that most of the
        // writes hit pages that did not go out yet.
        let writer = {
            let host = src_host.clone();
            let block = src_block.clone();
            std::thread::spawn(move || {
                while !host.snapshot_resumed.load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                for page in (0..PAGES).rev() {
                    block.fill(page << 12, 4096, 0xee).unwrap();
                }
            })
        };
        src.migrate(Some(&uri), None).unwrap();
        src.join();
        writer.join().unwrap();
        assert_eq!(src.status(), MigrationStatus::Completed, "{:?}", src.query().error_desc);
        assert!(src_host.running.load(Ordering::Relaxed));
        let mut page = vec![0u8; 4096];
        src_block.read(0, &mut page).unwrap();
        assert!(page.iter().all(|&b| b == 0xee));

        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, dst_block) = machine(PAGES, dst_host.clone());
        dst.incoming(Some(&uri), None, true).unwrap();
        dst.join();
        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
        for p in 0..PAGES {
            dst_block.read(p << 12, &mut page).unwrap();
            let want = if p % 4 != 0 { before(p) } else { 0 };
            assert!(page.iter().all(|&b| b == want), "page {p}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn snapshot_state() {
        const PAGES: u64 = 64;
        let host = Arc::new(Host::default());
        let (m, block) = machine(PAGES, host.clone());
        caps(&m, &[MigrationCapability::Events]);
        for page in 0..PAGES {
            block.fill(page << 12, 4096, page as u8 ^ 0x5a).unwrap();
        }
        let mut buf = Vec::new();
        let n = m.save_snapshot_state(&mut buf).unwrap();
        assert_eq!(n, buf.len() as u64);
        assert_eq!(m.status(), MigrationStatus::Completed);
        assert_eq!(
            *host.events.lock().unwrap(),
            [MigrationStatus::Setup, MigrationStatus::Completed]
        );
        let mut want = vec![0u8; (PAGES << 12) as usize];
        block.read(0, &mut want).unwrap();
        block.fill(0, PAGES << 12, 0xff).unwrap();
        m.load_snapshot_state(&buf[..]).unwrap();
        let mut got = vec![0u8; (PAGES << 12) as usize];
        block.read(0, &mut got).unwrap();
        assert!(want == got);
        // A cut short state fails to load.
        assert!(m.load_snapshot_state(&buf[..buf.len() / 2]).is_err());

        caps(&m, &[MigrationCapability::Multifd]);
        assert_eq!(
            m.can_snapshot().unwrap_err().message(),
            "Snapshots are not compatible with multifd"
        );
    }

    #[test]
    fn mapped_ram_file() {
        mapped_ram_over_file(false, false);
    }

    #[test]
    fn mapped_ram_multifd_file() {
        mapped_ram_over_file(true, false);
    }

    #[test]
    fn mapped_ram_direct_io() {
        // Not every file system takes O_DIRECT; tmpfs only does on newer kernels.
        let probe = std::env::temp_dir().join(format!("ruvm-direct-{}", std::process::id()));
        let _ = std::fs::write(&probe, b"");
        let ok = crate::channel::open_file(&probe.display().to_string(), true, true).is_ok()
            && ruvm_sys::directio::o_direct().is_some();
        let _ = std::fs::remove_file(&probe);
        if ok {
            mapped_ram_over_file(true, true);
        }
    }

    /// A migration with switchover-ack on both sides. A machine type older than 11.1 makes the
    /// source wait for the one acknowledgement the destination sends when the return path
    /// opens; with 11.1 no device asks for one and nothing waits.
    fn switchover_ack(name: &str) {
        const PAGES: u64 = 16;
        let ack = [MigrationCapability::ReturnPath, MigrationCapability::SwitchoverAck];
        let dst_host = Arc::new(Host::default());
        dst_host.incoming.store(true, Ordering::Relaxed);
        let (dst, _) = machine_named(name, PAGES, dst_host);
        caps(&dst, &ack);
        dst.incoming(Some("tcp:127.0.0.1:0"), None, true).unwrap();
        let uri = format!("tcp:127.0.0.1:{}", port_of(&dst));
        let src_host = Arc::new(Host::default());
        src_host.running.store(true, Ordering::Relaxed);
        let (src, _) = machine_named(name, PAGES, src_host);
        caps(&src, &ack);
        src.migrate(Some(&uri), None).unwrap();
        src.join();
        dst.join();
        assert_eq!(src.status(), MigrationStatus::Completed, "{:?}", src.query().error_desc);
        assert_eq!(dst.incoming_status(), MigrationStatus::Completed);
    }

    #[test]
    fn switchover_ack_old_and_new() {
        switchover_ack("pc-q35-11.0");
        switchover_ack("pc-q35-11.1");
        let (m, _) = machine(4, Arc::new(Host::default()));
        let only = [MigrationCapabilityStatus {
            capability: MigrationCapability::SwitchoverAck,
            state: true,
        }];
        assert_eq!(
            m.set_capabilities(&only).unwrap_err().message(),
            "Capability 'switchover-ack' requires capability 'return-path'"
        );
    }

    #[test]
    fn a_cancelled_write_fails_rather_than_retrying() {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut w = CancelWriter { inner: Box::new(io::sink()), cancel: cancel.clone() };
        w.write_all(b"before").unwrap();
        cancel.store(true, Ordering::Relaxed);
        assert_eq!(w.write_all(b"after").unwrap_err().to_string(), "migration cancelled");
        assert!(w.flush().is_err());
    }

    #[test]
    fn a_destination_that_is_not_there_fails_the_migration_not_the_command() {
        let host = Arc::new(Host::default());
        let (m, _) = machine(4, host);
        // A listener that is closed again leaves a port that nothing listens on.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        m.migrate(Some(&format!("tcp:127.0.0.1:{port}")), None).unwrap();
        assert_eq!(m.status(), MigrationStatus::Failed);
        let why = m.query().error_desc.unwrap();
        assert!(why.starts_with("Failed to connect to '127.0.0.1:"), "{why}");
    }

    #[test]
    fn mapped_ram_checks() {
        let host = Arc::new(Host::default());
        let (m, _) = machine(4, host);
        let set =
            |c, on| m.set_capabilities(&[MigrationCapabilityStatus { capability: c, state: on }]);
        set(MigrationCapability::MappedRam, true).unwrap();
        let err = set(MigrationCapability::Xbzrle, true).unwrap_err();
        assert_eq!(err.message(), "Mapped-ram migration is incompatible with xbzrle");
        let err = set(MigrationCapability::PostcopyRam, true).unwrap_err();
        assert_eq!(err.message(), "Mapped-ram migration is incompatible with postcopy");
        let err = m.migrate(Some("tcp:127.0.0.1:1"), None).unwrap_err();
        assert_eq!(err.message(), "Migration requires seekable transport (e.g. file)");
        set(MigrationCapability::Multifd, true).unwrap();

        // QEMU checks the compression that was set before, so the first change goes through.
        let zlib = MigrationParameters {
            multifd_compression: Some(MultiFDCompression::Zlib),
            ..Default::default()
        };
        m.set_parameters(&zlib).unwrap();
        let err = m.set_parameters(&zlib).unwrap_err();
        assert_eq!(
            err.message(),
            "Mapped-ram only available for non-compressed non-TLS multifd migration"
        );
        let err = m.migrate(Some("file:/nonexistent/x"), None).unwrap_err();
        assert_eq!(err.message(), "Cannot use compression with mapped-ram");

        // Without mapped-ram a file takes one channel only.
        set(MigrationCapability::MappedRam, false).unwrap();
        let err = m.migrate(Some("file:/nonexistent/x"), None).unwrap_err();
        assert_eq!(err.message(), "Migration requires multi-channel URIs (e.g. tcp)");
    }
}
