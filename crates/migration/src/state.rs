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

use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ruvm_base::{Error, Result, bail};
use ruvm_qapi::types::{
    MigMode, MigrationCapability, MigrationCapabilityStatus, MigrationChannel, MigrationInfo,
    MigrationParameters, MigrationRAMStats, MigrationStatus, MultiFDCompression, SocketAddress,
    StrOrNull, ZeroPageDetection,
};
use ruvm_vmstate::StreamReader;

use crate::channel::{Channel, FdResolver, parse_input};
use crate::ram::RamStats;
use crate::savevm::{QemuFile, SaveVm};

/// `BUFFER_DELAY`: the length of a rate limiting and bandwidth window, in milliseconds.
const BUFFER_DELAY: u64 = 100;
/// `XFER_LIMIT_RATIO`: windows per second.
const XFER_LIMIT_RATIO: u64 = 1000 / BUFFER_DELAY;
/// `MAX_MIGRATE_DOWNTIME`, in milliseconds.
const MAX_MIGRATE_DOWNTIME: u64 = 2000 * 1000;
/// The bytes one iteration sends at most when there is no rate limit, so that cancelling and
/// the window bookkeeping stay responsive.
const UNLIMITED_CHUNK: u64 = 16 << 20;
const PAGE_SIZE: u64 = 4096;

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
}

/// The capabilities this side implements. The others are refused, so that a management tool
/// learns at `migrate-set-capabilities` rather than half way through.
const SUPPORTED_CAPS: &[MigrationCapability] = &[
    MigrationCapability::Events,
    MigrationCapability::ValidateUuid,
    MigrationCapability::LateBlockActivate,
    MigrationCapability::PauseBeforeSwitchover,
];

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
        xbzrle_cache_size: Some(64 << 20),
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

/// `migrate_params_check()` on the parameters with all members present.
fn check_parameters(p: &MigrationParameters) -> Result<()> {
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
    if !(1..=1000).contains(&u(p.x_vcpu_dirty_limit_period)) {
        bail!("Option x-vcpu-dirty-limit-period expects a value between 1 and 1000");
    }
    if u(p.vcpu_dirty_limit) < 1 {
        bail!("Parameter 'vcpu-dirty-limit' must be greater than 1 MB/s");
    }
    if p.direct_io == Some(true) {
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
    precopy_bytes: u64,
    downtime_bytes: u64,

    thread: Option<JoinHandle<()>>,
    incoming_thread: Option<JoinHandle<()>>,
}

struct Inner {
    host: Arc<dyn MigrationHost>,
    savevm: Arc<Mutex<SaveVm>>,
    ram: Option<Arc<RamStats>>,
    fds: Mutex<Option<Arc<FdResolver>>>,
    shared: Mutex<Shared>,
    // Set by `migrate_cancel` and checked by the thread and each write to the channel.
    cancel: Arc<AtomicBool>,
    // `pause_event`, for `pause-before-switchover`.
    resume: Condvar,
    resumed: AtomicBool,
    transferred: AtomicU64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Inner {
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
}

/// A writer that fails once the migration is cancelled, so the thread stops at its next write.
struct CancelWriter {
    inner: Box<dyn Write + Send>,
    cancel: Arc<AtomicBool>,
}

impl Write for CancelWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "migration cancelled"));
        }
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::Interrupted, "migration cancelled"));
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
        Migration {
            inner: Arc::new(Inner {
                host,
                savevm,
                ram,
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
                    precopy_bytes: 0,
                    downtime_bytes: 0,
                    thread: None,
                    incoming_thread: None,
                }),
                cancel: Arc::new(AtomicBool::new(false)),
                resume: Condvar::new(),
                resumed: AtomicBool::new(false),
                transferred: AtomicU64::new(0),
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
        check_parameters(&p)?;
        if new.block_bitmap_mapping.is_some() {
            s.has_block_bitmap_mapping = true;
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
        let page_size = PAGE_SIZE as i64;
        let completed = s.state == MigrationStatus::Completed;
        match s.state {
            MigrationStatus::None => return info,
            MigrationStatus::Active
            | MigrationStatus::Cancelling
            | MigrationStatus::PreSwitchover
            | MigrationStatus::Device
            | MigrationStatus::Failing
            | MigrationStatus::Completed => {
                // populate_time_info()
                info.setup_time = Some(s.setup_time);
                info.total_time = Some(if completed {
                    s.total_time
                } else {
                    s.start_time.map_or(0, |t| t.elapsed().as_millis() as i64)
                });
                if completed {
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
                    precopy_bytes: s.precopy_bytes,
                    downtime_bytes: s.downtime_bytes,
                    ..Default::default()
                };
                if !completed {
                    stats.remaining = rd(|r| &r.remaining_pages) as i64 * page_size;
                    stats.dirty_pages_rate = rd(|r| &r.dirty_pages_rate) as i64;
                    // populate_global_info()
                    info.remaining = Some(s.pending_bytes);
                }
                info.ram = Some(stats);
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
    /// migration thread.
    pub fn migrate(&self, uri: Option<&str>, channels: Option<&[MigrationChannel]>) -> Result<()> {
        let inner = &self.inner;
        let addr = parse_input(uri, channels)?;
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
            if !str_or_null(&s.params.tls_creds).is_empty() {
                bail!("TLS migration is not supported");
            }
            if s.params.mode.is_some_and(|m| m != MigMode::Normal) {
                bail!("Only the normal migration mode is supported");
            }
        }
        if let Some(t) = lock(&inner.shared).thread.take() {
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
            s.precopy_bytes = 0;
            s.downtime_bytes = 0;
            s.state = MigrationStatus::None;
        }
        inner.cancel.store(false, Ordering::Relaxed);
        inner.resumed.store(false, Ordering::Relaxed);
        inner.transferred.store(0, Ordering::Relaxed);
        inner.set_state(Some(MigrationStatus::None), MigrationStatus::Setup);

        let fds = lock(&inner.fds).clone();
        let sink = match Channel::connect(&addr, fds.as_deref()) {
            Ok(sink) => sink,
            Err(e) => {
                // migration_connect_error_propagate()
                inner.set_error(&e);
                inner.set_state(Some(MigrationStatus::Setup), MigrationStatus::Failing);
                inner.set_state(Some(MigrationStatus::Failing), MigrationStatus::Failed);
                return Err(e);
            }
        };
        let sink = CancelWriter { inner: sink, cancel: inner.cancel.clone() };
        let thread_inner = inner.clone();
        let handle = std::thread::Builder::new()
            .name("live_migration".to_string())
            .spawn(move || migration_thread(thread_inner, sink))
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
        let state = lock(&inner.shared).incoming_state;
        if state != MigrationStatus::None {
            bail!("Illegal migration incoming state: {}", state.as_str());
        }
        let fds = lock(&inner.fds).clone();
        let listener = Channel::listen(&addr, fds.as_deref())?;
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
                let res = listener.accept().and_then(|src| {
                    thread_inner.set_incoming_state(MigrationStatus::Active);
                    let mut f = StreamReader::from_reader(src);
                    let mut vm = lock(&thread_inner.savevm);
                    vm.capabilities = lock(&thread_inner.shared).caps.clone();
                    let ret = vm.load_state(&mut f).map(|_| ());
                    let errno = f.get_error();
                    ret.map_err(|e| {
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

fn incoming_finish(inner: &Inner, res: Result<()>, exit_on_error: bool) {
    match res {
        Ok(()) => {
            inner.host.incoming_done();
            inner.set_incoming_state(MigrationStatus::Completed);
        }
        Err(e) => {
            inner.set_incoming_state(MigrationStatus::Failed);
            inner.set_error(&e);
            inner.host.incoming_failed(&e, exit_on_error);
        }
    }
}

/// `migration_get_switchover_bw()`, in bytes per second.
fn switchover_bw(s: &Shared) -> f64 {
    let avail = s.params.avail_switchover_bandwidth.unwrap_or(0);
    if avail > 0 { avail as f64 } else { s.mbps * 1000.0 * 1000.0 / 8.0 }
}

struct Thread<'a> {
    inner: &'a Inner,
    f: QemuFile<'static>,
    iteration_start: Instant,
    iteration_bytes: u64,
    iteration_pages: u64,
    threshold: u64,
    was_running: bool,
    stopped: bool,
}

impl Thread<'_> {
    fn pages_sent(&self) -> u64 {
        self.inner
            .ram
            .as_ref()
            .map_or(0, |r| r.normal.load(Ordering::Relaxed) + r.duplicate.load(Ordering::Relaxed))
    }

    fn publish(&self) {
        self.inner.transferred.store(self.f.transferred(), Ordering::Relaxed);
    }

    fn check_cancel(&self) -> Result<()> {
        if self.inner.cancel.load(Ordering::Relaxed) {
            bail!("migration cancelled");
        }
        Ok(())
    }

    /// `migration_update_counters()`, at the end of a window.
    fn update_counters(&mut self, now: Instant) {
        let spent = now.duration_since(self.iteration_start).as_millis() as u64;
        if spent < BUFFER_DELAY {
            return;
        }
        let bytes = self.f.transferred();
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

    /// The bytes a window may carry, `max-bandwidth / XFER_LIMIT_RATIO`, or 0 for no limit.
    fn window_limit(&self) -> u64 {
        lock(&self.inner.shared).params.max_bandwidth.unwrap_or(0) / XFER_LIMIT_RATIO
    }

    fn run(&mut self, vm: &mut SaveVm) -> Result<()> {
        let setup_start = Instant::now();
        vm.capabilities = lock(&self.inner.shared).caps.clone();
        vm.save_header(&mut self.f)?;
        vm.save_setup(&mut self.f)?;
        self.publish();
        if !self.inner.set_state(Some(MigrationStatus::Setup), MigrationStatus::Active) {
            self.check_cancel()?;
        }
        lock(&self.inner.shared).setup_time = setup_start.elapsed().as_millis() as i64;
        self.iteration_start = Instant::now();
        self.iteration_bytes = self.f.transferred();
        self.iteration_pages = self.pages_sent();

        loop {
            self.check_cancel()?;
            let limit = self.window_limit();
            let used = self.f.transferred() - self.iteration_bytes;
            if limit == 0 || used < limit {
                // migration_iteration_run()
                let mut pending = vm.pending(false);
                if pending <= self.threshold || pending == 0 {
                    // migration_iteration_go_next()
                    pending = vm.pending(true);
                    let (events, pass) = {
                        let mut s = lock(&self.inner.shared);
                        s.dirty_bytes_last_sync = pending;
                        let pass = self
                            .inner
                            .ram
                            .as_ref()
                            .map_or(0, |r| r.dirty_sync_count.load(Ordering::Relaxed));
                        (self.inner.events(&s), pass)
                    };
                    if events {
                        self.inner.host.pass_event(pass as i64);
                    }
                }
                lock(&self.inner.shared).pending_bytes = pending;
                if pending <= self.threshold {
                    return self.complete(vm);
                }
                let budget = if limit == 0 { UNLIMITED_CHUNK } else { limit - used };
                vm.save_iterate(&mut self.f, budget)?;
                self.publish();
            }
            let now = Instant::now();
            let used = self.f.transferred() - self.iteration_bytes;
            let window_end = self.iteration_start + Duration::from_millis(BUFFER_DELAY);
            if limit != 0 && used >= limit && now < window_end {
                // migration_rate_limit(): wait for the next window.
                std::thread::sleep(window_end - now);
            }
            self.update_counters(Instant::now());
        }
    }

    /// `migration_completion()` for precopy.
    fn complete(&mut self, vm: &mut SaveVm) -> Result<()> {
        let downtime_start = Instant::now();
        self.was_running = self.inner.host.is_running();
        self.inner.host.stop_for_switchover().map_err(|e| e.prepend("Failed to stop the VM: "))?;
        self.stopped = true;
        // migration_switchover_prepare()
        let pause =
            lock(&self.inner.shared).caps.contains(&MigrationCapability::PauseBeforeSwitchover);
        self.check_cancel().map_err(|_| Error::generic("Switchover is interrupted"))?;
        if pause {
            self.inner.set_state(None, MigrationStatus::PreSwitchover);
            let mut g = lock(&self.inner.shared);
            while !self.inner.resumed.load(Ordering::Relaxed) {
                g = self.inner.resume.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            drop(g);
            self.check_cancel().map_err(|_| Error::generic("Switchover is interrupted"))?;
            self.inner.set_state(Some(MigrationStatus::PreSwitchover), MigrationStatus::Device);
        } else {
            self.inner.set_state(None, MigrationStatus::Device);
        }
        let before = self.f.transferred();
        lock(&self.inner.shared).precopy_bytes = before;
        vm.save_complete(&mut self.f)?;
        self.publish();
        // migration_completion_end()
        let bytes = self.f.transferred();
        let mut s = lock(&self.inner.shared);
        s.downtime = downtime_start.elapsed().as_millis() as i64;
        s.downtime_bytes = bytes - before;
        s.total_time = s.start_time.map_or(0, |t| t.elapsed().as_millis() as i64);
        let transfer_time = s.total_time - s.setup_time;
        if transfer_time > 0 {
            s.mbps = (bytes as f64 * 8.0) / transfer_time as f64 / 1000.0;
        }
        s.pending_bytes = 0;
        drop(s);
        self.inner.set_state(None, MigrationStatus::Completed);
        Ok(())
    }
}

fn migration_thread(inner: Arc<Inner>, sink: CancelWriter) {
    let mut t = Thread {
        inner: &inner,
        f: QemuFile::new(sink),
        iteration_start: Instant::now(),
        iteration_bytes: 0,
        iteration_pages: 0,
        threshold: 0,
        was_running: false,
        stopped: false,
    };
    let savevm = inner.savevm.clone();
    let res = {
        let mut vm = lock(&savevm);
        let res = t.run(&mut vm);
        vm.save_cleanup();
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
        MigrationStatus::Completed => inner.host.set_postmigrate(),
        _ => {
            if t.stopped {
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, Ordering};

    use ruvm_mem::RamBlock;
    use ruvm_qapi::types::{
        InetSocketAddress, MigrationAddress, MigrationAddressU, MigrationCapability,
        MigrationCapabilityStatus, MigrationChannel, MigrationChannelType, MigrationParameters,
        MigrationStatus, SocketAddress, SocketAddressU,
    };

    use super::{Migration, MigrationHost};
    use crate::ram::{NoHooks, RamSection};
    use crate::savevm::{EntryInfo, MachineConfig, SaveVm};

    #[derive(Default)]
    struct Host {
        running: AtomicBool,
        incoming: AtomicBool,
        events: Mutex<Vec<MigrationStatus>>,
        done: AtomicBool,
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
        let block = Arc::new(RamBlock::new("pc.ram", pages << 12, 12).unwrap());
        let ram = RamSection::new(vec![block.clone()], NoHooks);
        let stats = ram.stats();
        let config = MachineConfig {
            name: "pc-q35-11.1".to_string(),
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
                capability: MigrationCapability::Xbzrle,
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
}
