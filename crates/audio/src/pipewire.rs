// SPDX-License-Identifier: GPL-2.0-or-later

//! The `pipewire` driver, QEMU's `audio/pwaudio.c`, over the `pipewire` crate.
//!
//! QEMU runs a `pw_thread_loop` per audiodev and takes its lock around every call into
//! libpipewire. The crate only makes a thread loop in unsafe code, so here each audiodev owns a
//! worker thread that runs a plain main loop with the context, the core and the streams on it.
//! A voice sends the worker a job through a channel the main loop watches and waits for the
//! answer, which is the lock QEMU takes with one more thread in the way. The audio itself goes
//! through a ring buffer per voice that the stream's `process` callback and the voice share, as
//! in QEMU, and the stream state the voice checks is mirrored from the `state_changed` callback.

use std::cell::RefCell;
use std::collections::HashMap;
use std::io::{self, Cursor};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use pipewire::context::ContextRc;
use pipewire::core::CoreRc;
use pipewire::main_loop::MainLoopRc;
use pipewire::node::Node;
use pipewire::permissions::PermissionFlags;
use pipewire::properties::PropertiesBox;
use pipewire::registry::{GlobalObject, RegistryRc};
use pipewire::spa;
use pipewire::stream::{Stream, StreamFlags, StreamListener, StreamRc, StreamState};
use pipewire::types::ObjectType;
use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{AudioFormat, Audiodev, AudiodevPipewirePerDirectionOptions, AudiodevU};
use spa::param::ParamType;
use spa::param::audio::AudioFormat as SpaFormat;
use spa::param::format::{FormatProperties, MediaSubtype, MediaType};
use spa::pod::serialize::PodSerializer;
use spa::pod::{Object, Pod, Property, Value, ValueArray};
use spa::utils::dict::DictRef;
use spa::utils::{Direction, Id, SpaTypes};

use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, Volume,
    generic_run_buffer_in, generic_run_buffer_out,
};

const RINGBUFFER_SIZE: u32 = 1 << 22;
const RINGBUFFER_MASK: u32 = RINGBUFFER_SIZE - 1;

/// The latency `qpw_init_out()` and `qpw_init_in()` use when the user leaves it out.
const DEFAULT_LATENCY: u32 = 46440;

/// `SPA_ID_INVALID`, what `pw_stream_get_node_id()` returns before the node is exported.
const SPA_ID_INVALID: u32 = u32::MAX;

/// A C string ends at the first NUL.
fn c_str(s: &str) -> &str {
    s.split('\0').next().unwrap_or("")
}

/// The text `g_strerror(errno)` gives for the error the last call on this thread left.
fn errno_str() -> String {
    strerror(&io::Error::last_os_error())
}

/// `spa_strerror()`.
fn spa_strerror(res: i32) -> String {
    const ASYNC_MASK: i32 = 0xf << 28;
    const ASYNC_BIT: i32 = 0x4 << 28;
    let errno = if res & ASYNC_MASK == ASYNC_BIT { libc::EINPROGRESS } else { res.wrapping_neg() };
    strerror(&io::Error::from_raw_os_error(errno))
}

/// `spa_ringbuffer` with the data it indexes.
struct Ring {
    readindex: u32,
    writeindex: u32,
    buf: Vec<u8>,
}

impl Ring {
    fn new() -> Self {
        Ring { readindex: 0, writeindex: 0, buf: vec![0; RINGBUFFER_SIZE as usize] }
    }

    /// `spa_ringbuffer_get_read_index()`: the bytes there are to read and where they start.
    fn read_index(&self) -> (i32, u32) {
        (self.writeindex.wrapping_sub(self.readindex) as i32, self.readindex)
    }

    /// `spa_ringbuffer_get_write_index()`: the bytes in use and where the next ones go.
    fn write_index(&self) -> (i32, u32) {
        (self.writeindex.wrapping_sub(self.readindex) as i32, self.writeindex)
    }

    /// `spa_ringbuffer_read_data()` and `spa_ringbuffer_read_update()`.
    fn read(&mut self, index: u32, data: &mut [u8]) {
        let off = (index & RINGBUFFER_MASK) as usize;
        let l0 = data.len().min(self.buf.len() - off);
        let (a, b) = data.split_at_mut(l0);
        a.copy_from_slice(&self.buf[off..off + l0]);
        b.copy_from_slice(&self.buf[..b.len()]);
        self.readindex = index.wrapping_add(data.len() as u32);
    }

    /// `spa_ringbuffer_write_data()` and `spa_ringbuffer_write_update()`.
    fn write(&mut self, index: u32, data: &[u8]) {
        let off = (index & RINGBUFFER_MASK) as usize;
        let l0 = data.len().min(self.buf.len() - off);
        self.buf[off..off + l0].copy_from_slice(&data[..l0]);
        self.buf[..data.len() - l0].copy_from_slice(&data[l0..]);
        self.writeindex = index.wrapping_add(data.len() as u32);
    }
}

/// What a voice and its stream's callbacks share, the part of QEMU's `PWVoice` the realtime
/// thread touches.
struct Shared {
    ring: Mutex<Ring>,
    /// Whether the stream is `PW_STREAM_STATE_STREAMING`.
    streaming: AtomicBool,
    frame_size: u32,
    req: u32,
    /// The layout `audio_pcm_info_clear_buf()` fills silence for.
    info: PcmInfo,
    /// The controls the stream has announced through `control_info`, what `find_control()`
    /// looks through in libpipewire.
    controls: Mutex<Vec<u32>>,
    /// Whether the stream has a format, from `param_changed`.
    negotiated: AtomicBool,
}

impl Shared {
    fn ring(&self) -> MutexGuard<'_, Ring> {
        self.ring.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// `playback_on_process()`.
fn playback_on_process(stream: &Stream, v: &Shared) {
    let Some(mut b) = stream.dequeue_buffer() else {
        error_report(&format!("out of buffers: {}", errno_str()));
        return;
    };
    // The buffer goes back to the stream when it drops.
    let requested = b.requested();
    let Some(d) = b.datas_mut().first_mut() else { return };
    let maxsize = d.as_raw().maxsize;
    // The total number of bytes to read from the ring.
    let mut req = requested.wrapping_mul(u64::from(v.frame_size)) as u32;
    if req == 0 {
        req = v.req;
    }
    let mut n_bytes = req.min(maxsize);
    {
        let Some(p) = d.data() else { return };
        let mut ring = v.ring();
        let (avail, index) = ring.read_index();
        if avail <= 0 {
            v.info.clear_buf(p, (n_bytes / v.frame_size) as usize);
        } else {
            // PipeWire calls back at once for the rest when this is short of n_bytes.
            n_bytes = n_bytes.min(avail as u32);
            ring.read(index, &mut p[..n_bytes as usize]);
        }
    }
    let chunk = d.chunk_mut();
    *chunk.offset_mut() = 0;
    *chunk.stride_mut() = v.frame_size as i32;
    *chunk.size_mut() = n_bytes;
}

/// `capture_on_process()`.
fn capture_on_process(stream: &Stream, v: &Shared) {
    let Some(mut b) = stream.dequeue_buffer() else {
        error_report(&format!("out of buffers: {}", errno_str()));
        return;
    };
    let Some(d) = b.datas_mut().first_mut() else { return };
    let maxsize = d.as_raw().maxsize;
    let offs = d.chunk().offset().min(maxsize);
    let n_bytes = d.chunk().size().min(maxsize - offs);
    let Some(p) = d.data() else { return };
    let mut ring = v.ring();
    let (filled, index) = ring.write_index();
    if filled < 0 {
        error_report(&format!("{:p}: underrun write:{index} filled:{filled}", p.as_ptr()));
    } else if (filled as u32).wrapping_add(n_bytes) > RINGBUFFER_SIZE {
        error_report(&format!(
            "{:p}: overrun write:{index} filled:{filled} + size:{n_bytes} > max:{RINGBUFFER_SIZE}",
            p.as_ptr()
        ));
    }
    ring.write(index, &p[offs as usize..(offs + n_bytes) as usize]);
}

/// `audfmt_to_pw()`.
fn audfmt_to_pw(fmt: AudioFormat, big_endian: bool) -> SpaFormat {
    match (fmt, big_endian) {
        (AudioFormat::S8, _) => SpaFormat::S8,
        (AudioFormat::U8, _) => SpaFormat::U8,
        (AudioFormat::S16, false) => SpaFormat::S16LE,
        (AudioFormat::S16, true) => SpaFormat::S16BE,
        (AudioFormat::U16, false) => SpaFormat::U16LE,
        (AudioFormat::U16, true) => SpaFormat::U16BE,
        (AudioFormat::S32, false) => SpaFormat::S32LE,
        (AudioFormat::S32, true) => SpaFormat::S32BE,
        (AudioFormat::U32, false) => SpaFormat::U32LE,
        (AudioFormat::U32, true) => SpaFormat::U32BE,
        (AudioFormat::F32, false) => SpaFormat::F32LE,
        (AudioFormat::F32, true) => SpaFormat::F32BE,
    }
}

/// `pw_to_audfmt()`: the format, its byte order and the size of a sample. Every format
/// `audfmt_to_pw()` makes comes back, so the internal error QEMU reports for the rest cannot
/// happen.
fn pw_to_audfmt(fmt: SpaFormat, big_endian: bool) -> (AudioFormat, bool, u32) {
    match fmt {
        SpaFormat::S8 => (AudioFormat::S8, big_endian, 1),
        SpaFormat::S16BE => (AudioFormat::S16, true, 2),
        SpaFormat::S16LE => (AudioFormat::S16, false, 2),
        SpaFormat::U16BE => (AudioFormat::U16, true, 2),
        SpaFormat::U16LE => (AudioFormat::U16, false, 2),
        SpaFormat::S32BE => (AudioFormat::S32, true, 4),
        SpaFormat::S32LE => (AudioFormat::S32, false, 4),
        SpaFormat::U32BE => (AudioFormat::U32, true, 4),
        SpaFormat::U32LE => (AudioFormat::U32, false, 4),
        SpaFormat::F32BE => (AudioFormat::F32, true, 4),
        SpaFormat::F32LE => (AudioFormat::F32, false, 4),
        _ => (AudioFormat::U8, big_endian, 1),
    }
}

/// `qpw_set_position()`. Only usb-audio has more than two channels in QEMU, so the order is
/// the one it uses.
fn set_position(channels: u32) -> Vec<u32> {
    use spa::sys::*;
    let mut position = vec![SPA_AUDIO_CHANNEL_UNKNOWN; channels.min(64) as usize];
    let map: &[u32] = match channels {
        8 => &[
            SPA_AUDIO_CHANNEL_FL,
            SPA_AUDIO_CHANNEL_FR,
            SPA_AUDIO_CHANNEL_FC,
            SPA_AUDIO_CHANNEL_LFE,
            SPA_AUDIO_CHANNEL_RL,
            SPA_AUDIO_CHANNEL_RR,
            SPA_AUDIO_CHANNEL_SL,
            SPA_AUDIO_CHANNEL_SR,
        ],
        6 => &[
            SPA_AUDIO_CHANNEL_FL,
            SPA_AUDIO_CHANNEL_FR,
            SPA_AUDIO_CHANNEL_FC,
            SPA_AUDIO_CHANNEL_LFE,
            SPA_AUDIO_CHANNEL_RL,
            SPA_AUDIO_CHANNEL_RR,
        ],
        2 => &[SPA_AUDIO_CHANNEL_FL, SPA_AUDIO_CHANNEL_FR],
        1 => &[SPA_AUDIO_CHANNEL_MONO],
        _ => {
            error_report(&format!("pipewire: unsupported channel count {channels}"));
            &[]
        }
    };
    position[..map.len()].copy_from_slice(map);
    position
}

/// `spa_format_audio_raw_build()` for `SPA_PARAM_EnumFormat`, as the bytes of the pod.
fn enum_format(format: SpaFormat, rate: u32, position: &[u32]) -> Vec<u8> {
    let mut props = vec![
        Property::new(
            FormatProperties::MediaType.as_raw(),
            Value::Id(Id(MediaType::Audio.as_raw())),
        ),
        Property::new(
            FormatProperties::MediaSubtype.as_raw(),
            Value::Id(Id(MediaSubtype::Raw.as_raw())),
        ),
        Property::new(FormatProperties::AudioFormat.as_raw(), Value::Id(Id(format.as_raw()))),
    ];
    if rate != 0 {
        props.push(Property::new(FormatProperties::AudioRate.as_raw(), Value::Int(rate as i32)));
    }
    if !position.is_empty() {
        props.push(Property::new(
            FormatProperties::AudioChannels.as_raw(),
            Value::Int(position.len() as i32),
        ));
        props.push(Property::new(
            FormatProperties::AudioPosition.as_raw(),
            Value::ValueArray(ValueArray::Id(position.iter().copied().map(Id).collect())),
        ));
    }
    let obj = Value::Object(Object {
        type_: SpaTypes::ObjectParamFormat.as_raw(),
        id: ParamType::EnumFormat.as_raw(),
        properties: props,
    });
    match PodSerializer::serialize(Cursor::new(Vec::new()), &obj) {
        Ok((c, _)) => c.into_inner(),
        Err(_) => Vec::new(),
    }
}

/// One stream on the worker. The node proxy and the listener go first, so the stream never calls
/// back into a listener that is gone.
struct Slot {
    node: Option<Node>,
    _listener: StreamListener<Arc<Shared>>,
    stream: StreamRc,
    shared: Arc<Shared>,
    /// The last value of each control asked for that has not gone out yet.
    pending: Vec<(u32, Value)>,
}

/// What a connection's worker thread owns besides the main loop: QEMU's `AudioPw` less the
/// thread loop.
struct Worker {
    core: CoreRc,
    /// The worker's own queue, for work a stream callback hands back to it.
    tx: pipewire::channel::Sender<Msg>,
    /// Made the first time a volume goes out, for binding the streams' own nodes.
    registry: Option<RegistryRc>,
    slots: HashMap<u64, Slot>,
}

type Job = Box<dyn FnOnce(&mut Worker) + Send>;

enum Msg {
    Job(Job),
    Quit,
}

/// What `qpw_stream_new()` takes.
struct NewStream {
    name: String,
    target: Option<String>,
    latency: String,
    in_: bool,
    format: Vec<u8>,
    shared: Arc<Shared>,
}

impl Worker {
    /// `qpw_stream_new()`, with the error to report returned.
    fn stream_new(&mut self, id: u64, new: NewStream) -> std::result::Result<(), String> {
        let mut props = PropertiesBox::new();
        props.insert("node.latency", new.latency);
        if let Some(t) = new.target {
            props.insert("target.object", t);
        }
        let stream = StreamRc::new(self.core.clone(), &new.name, props)
            .map_err(|_| format!("Failed to create PW stream: {}", errno_str()))?;
        let builder = stream
            .add_local_listener_with_user_data(Arc::clone(&new.shared))
            .state_changed(|_, v, _, state| {
                v.streaming.store(matches!(state, StreamState::Streaming), Ordering::Release);
            })
            .control_info(|_, v, id, _| {
                let mut c = v.controls.lock().unwrap_or_else(|e| e.into_inner());
                if !c.contains(&id) {
                    c.push(id);
                }
            })
            .param_changed({
                let tx = self.tx.clone();
                move |_, v, param, pod| {
                    if param != ParamType::Format.as_raw() {
                        return;
                    }
                    let was = v.negotiated.swap(pod.is_some(), Ordering::AcqRel);
                    if pod.is_some() && !was {
                        let _ = tx.send(Msg::Job(Box::new(move |w| w.flush_controls(id))));
                    }
                }
            });
        let builder = if new.in_ {
            builder.process(|s, v| capture_on_process(s, v))
        } else {
            builder.process(|s, v| playback_on_process(s, v))
        };
        let listener = builder
            .register()
            .map_err(|_| format!("Failed to create PW stream: {}", errno_str()))?;

        let mut params: Vec<&Pod> = Pod::from_bytes(&new.format).into_iter().collect();
        let dir = if new.in_ { Direction::Input } else { Direction::Output };
        let flags = StreamFlags::AUTOCONNECT
            | StreamFlags::INACTIVE
            | StreamFlags::MAP_BUFFERS
            | StreamFlags::RT_PROCESS;
        // Connect the stream to a sink or source.
        if stream.connect(dir, None, flags, &mut params).is_err() {
            let e = format!("Failed to connect PW stream: {}", errno_str());
            drop(listener);
            drop(stream);
            return Err(e);
        }
        let slot = Slot {
            node: None,
            _listener: listener,
            stream,
            shared: new.shared,
            pending: Vec::new(),
        };
        self.slots.insert(id, slot);
        Ok(())
    }

    /// `qpw_voice_set_volume()`.
    fn set_volume(&mut self, id: u64, vol: &Volume) {
        let n = vol.channels.min(vol.vol.len());
        let values: Vec<f32> = vol.vol[..n].iter().map(|&v| f32::from(v) / 255.0).collect();
        let Some(slot) = self.slots.get_mut(&id) else { return };
        slot.set_control(
            spa::sys::SPA_PROP_channelVolumes,
            Value::ValueArray(ValueArray::Float(values)),
        );
        slot.set_control(spa::sys::SPA_PROP_mute, Value::Bool(vol.mute));
        self.flush_controls(id);
    }

    /// Sends a negotiated stream's waiting controls in one `Props` object, since a second
    /// `set_param` sent right behind the first gets lost on the way. A control the stream has
    /// not announced is dropped, as `pw_stream_set_control()` drops it.
    fn flush_controls(&mut self, id: u64) {
        let Some(slot) = self.slots.get_mut(&id) else { return };
        if !slot.shared.negotiated.load(Ordering::Acquire) {
            return;
        }
        if slot.node.is_none() {
            let node_id = slot.stream.node_id();
            if node_id == SPA_ID_INVALID {
                return;
            }
            let registry = match &self.registry {
                Some(r) => r,
                None => match self.core.get_registry_rc() {
                    Ok(r) => self.registry.insert(r),
                    Err(_) => return,
                },
            };
            let global = GlobalObject::<&DictRef> {
                id: node_id,
                permissions: PermissionFlags::empty(),
                type_: ObjectType::Node,
                version: 0,
                props: None,
            };
            slot.node = registry.bind::<Node, _>(&global).ok();
        }
        let Some(node) = &slot.node else { return };
        let controls = slot.shared.controls.lock().unwrap_or_else(|e| e.into_inner());
        let properties: Vec<Property> = slot
            .pending
            .drain(..)
            .filter(|(c, _)| controls.contains(c))
            .map(|(c, v)| Property::new(c, v))
            .collect();
        if properties.is_empty() {
            return;
        }
        let obj = Value::Object(Object {
            type_: SpaTypes::ObjectParamProps.as_raw(),
            id: ParamType::Props.as_raw(),
            properties,
        });
        if let Ok((c, _)) = PodSerializer::serialize(Cursor::new(Vec::new()), &obj) {
            if let Some(pod) = Pod::from_bytes(&c.into_inner()) {
                node.set_param(ParamType::Props, 0, pod);
            }
        }
    }
}

impl Slot {
    /// `pw_stream_set_control()` for one control. The crate's wrapper leaves out the 0 that
    /// ends the C function's variadic list, so the worker builds the same `SPA_PARAM_Props`
    /// object itself and hands it to the stream's node through a proxy, as `pw-cli set-param`
    /// would. Props that reach the node that way before it has a format stall the link it is
    /// negotiating, so the last value of each control waits here until
    /// [`Worker::flush_controls`] can send it.
    fn set_control(&mut self, control: u32, value: Value) {
        match self.pending.iter_mut().find(|(c, _)| *c == control) {
            Some(p) => p.1 = value,
            None => self.pending.push((control, value)),
        }
    }
}

/// An audiodev's connection, QEMU's `AudioPw`. Dropping the last reference stops the worker,
/// which runs `audio_pw_finalize()`.
struct PwConn {
    tx: pipewire::channel::Sender<Msg>,
    worker: Option<JoinHandle<()>>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for PwConn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PwConn").field("next_id", &self.next_id).finish_non_exhaustive()
    }
}

impl PwConn {
    /// The part of `audio_pw_realize()` after the generic one.
    fn new() -> Result<Arc<PwConn>> {
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("pipewire".to_string())
            .spawn(move || conn_run(ready_tx))
            .map_err(|e| {
                Error::generic(format!("Could not start PipeWire loop: {}", strerror(&e)))
            })?;
        match ready_rx.recv() {
            Ok(Ok(tx)) => {
                Ok(Arc::new(PwConn { tx, worker: Some(worker), next_id: AtomicU64::new(0) }))
            }
            Ok(Err(msg)) => {
                let _ = worker.join();
                Err(Error::generic(msg))
            }
            Err(_) => {
                let _ = worker.join();
                Err(Error::generic("Could not start PipeWire loop"))
            }
        }
    }

    /// Runs `f` on the worker, the way QEMU calls into libpipewire with the thread loop locked.
    /// `None` means the worker is gone.
    fn call<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Worker) -> R + Send + 'static,
    ) -> Option<R> {
        let (tx, rx) = mpsc::sync_channel(1);
        let job: Job = Box::new(move |w: &mut Worker| {
            let _ = tx.send(f(w));
        });
        self.tx.send(Msg::Job(job)).ok()?;
        rx.recv().ok()
    }
}

impl Drop for PwConn {
    fn drop(&mut self) {
        let _ = self.tx.send(Msg::Quit);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

type Ready = SyncSender<std::result::Result<pipewire::channel::Sender<Msg>, String>>;

/// The worker thread: `audio_pw_realize()` up to `wait_resync()`, then the voices' jobs until
/// the last reference goes, then `audio_pw_finalize()`.
fn conn_run(ready: Ready) {
    pipewire::init();
    let ml = match MainLoopRc::new(None) {
        Ok(ml) => ml,
        Err(_) => {
            let _ = ready.send(Err(format!("Could not create PipeWire loop: {}", errno_str())));
            return;
        }
    };
    let context = match ContextRc::new(&ml, None) {
        Ok(c) => c,
        Err(_) => {
            let _ = ready.send(Err(format!("Could not create PipeWire context: {}", errno_str())));
            return;
        }
    };
    let core = match context.connect_rc(None) {
        Ok(c) => c,
        Err(_) => {
            let msg = format!("Failed to connect to PipeWire instance: {}", errno_str());
            let _ = ready.send(Err(msg));
            return;
        }
    };

    let (tx, rx) = pipewire::channel::channel::<Msg>();
    let own_tx = tx.clone();
    let pending = Rc::new(RefCell::new(None));
    let waiting = Rc::new(RefCell::new(Some((ready, tx))));
    let listener = {
        let (pending, waiting) = (Rc::clone(&pending), Rc::clone(&waiting));
        core.add_listener_local()
            .done(move |id, seq| {
                // `on_core_done()`: the sync realize waits for has come back.
                if id == pipewire::core::PW_ID_CORE && *pending.borrow() == Some(seq) {
                    if let Some((ready, tx)) = waiting.borrow_mut().take() {
                        let _ = ready.send(Ok(tx));
                    }
                }
            })
            .error(|id, seq, res, message| {
                error_report(&format!(
                    "error id:{id} seq:{seq} res:{res} ({}): {message}",
                    spa_strerror(res)
                ));
            })
            .register()
    };
    match core.sync(0) {
        Ok(seq) => *pending.borrow_mut() = Some(seq),
        Err(e) => {
            // QEMU's wait for a sync that never went out would never end.
            if let Some((ready, _)) = waiting.borrow_mut().take() {
                let _ = ready.send(Err(format!("Failed to connect to PipeWire instance: {e}")));
            }
            return;
        }
    }

    let worker = Rc::new(RefCell::new(Worker {
        core: core.clone(),
        tx: own_tx,
        registry: None,
        slots: HashMap::new(),
    }));
    let attached = {
        let (worker, quit) = (Rc::clone(&worker), ml.clone());
        rx.attach(ml.loop_(), move |msg| match msg {
            Msg::Job(job) => job(&mut worker.borrow_mut()),
            Msg::Quit => quit.quit(),
        })
    };
    ml.run();
    drop(attached);
    worker.borrow_mut().slots.clear();
    drop(worker);
    drop(listener);
    drop(core);
    drop(context);
}

fn pw_opts(dev: &Audiodev, in_: bool) -> AudiodevPipewirePerDirectionOptions {
    match &dev.u {
        AudiodevU::Pipewire(o) => (if in_ { &o.in_ } else { &o.out }).clone().unwrap_or_default(),
        _ => AudiodevPipewirePerDirectionOptions::default(),
    }
}

/// The `pipewire` driver, QEMU's `AudioPw`.
#[derive(Debug, Default)]
pub struct PwDriver {
    conn: Option<Arc<PwConn>>,
}

/// A voice of either direction, QEMU's `PWVoice` as the audio thread sees it.
struct PwVoice {
    conn: Arc<PwConn>,
    id: u64,
    v: Arc<Shared>,
    highwater_mark: u32,
}

impl PwVoice {
    /// `qpw_voice_set_enabled()`.
    fn set_enabled(&self, enable: bool) {
        let id = self.id;
        self.conn.call(move |w| {
            if let Some(slot) = w.slots.get(&id) {
                let _ = slot.stream.set_active(enable);
            }
        });
    }

    /// `qpw_voice_set_volume()`.
    fn set_volume(&self, vol: &Volume) {
        let (id, vol) = (self.id, *vol);
        self.conn.call(move |w| w.set_volume(id, &vol));
    }

    /// `qpw_voice_fini()`.
    fn fini(&self) {
        let id = self.id;
        self.conn.call(move |w| drop(w.slots.remove(&id)));
    }
}

impl PwDriver {
    /// `qpw_init_out()` and `qpw_init_in()`.
    fn open(
        &self,
        ctx: &InitCtx<'_>,
        as_: &AudSettings,
        in_: bool,
    ) -> Option<(PwVoice, PcmInfo, usize)> {
        let conn = self.conn.clone()?;
        let ppdo = pw_opts(ctx.dev, in_);
        let tp = u64::from(ctx.dev.timer_period.unwrap_or(10000));

        let format = audfmt_to_pw(as_.fmt, as_.big_endian);
        let channels = as_.nchannels as u32;
        let position = set_position(channels);
        let rate = as_.freq as u32;
        let (fmt, big_endian, sample_size) = pw_to_audfmt(format, as_.big_endian);
        let obt = AudSettings { fmt, big_endian, ..*as_ };
        let frame_size = sample_size.wrapping_mul(channels);
        let req = (tp * u64::from(rate) / 2 / 1_000_000 * u64::from(frame_size)) as u32;
        let info = PcmInfo::new(&obt);

        let shared = Arc::new(Shared {
            ring: Mutex::new(Ring::new()),
            streaming: AtomicBool::new(false),
            frame_size,
            req,
            info,
            controls: Mutex::new(Vec::new()),
            negotiated: AtomicBool::new(false),
        });
        // 75% of the timer period for faster updates.
        let buf_samples = tp * u64::from(rate) * 3 / 4 / 1_000_000;
        let new = NewStream {
            name: c_str(ppdo.stream_name.as_deref().unwrap_or(&ctx.dev.id)).to_string(),
            target: ppdo.name.as_deref().map(|n| c_str(n).to_string()),
            latency: format!("{buf_samples}/{rate}"),
            in_,
            format: enum_format(format, rate, &position),
            shared: Arc::clone(&shared),
        };
        let id = conn.next_id.fetch_add(1, Ordering::Relaxed);
        let r = conn.call(move |w| w.stream_new(id, new));
        match r.unwrap_or_else(|| Err("Failed to create PW stream: worker gone".to_string())) {
            Ok(()) => {}
            Err(e) => {
                error_report(&e);
                return None;
            }
        }

        let samples = ctx.pdo.buffer_frames(&obt, DEFAULT_LATENCY) as usize;
        let latency = u64::from(ppdo.latency.unwrap_or(DEFAULT_LATENCY));
        let highwater_mark = u64::from(RINGBUFFER_SIZE)
            .min(latency * u64::from(rate) / 1_000_000 * u64::from(frame_size))
            as u32;
        Some((PwVoice { conn, id, v: shared, highwater_mark }, info, samples))
    }
}

/// A playback voice, QEMU's `PWVoiceOut`.
struct PwVoiceOut(PwVoice);

/// A capture voice, QEMU's `PWVoiceIn`.
struct PwVoiceIn(PwVoice);

impl PcmOut for PwVoiceOut {
    /// `qpw_write()`.
    fn write(&mut self, _info: &PcmInfo, buf: &[u8]) -> usize {
        let v = &self.0.v;
        if !v.streaming.load(Ordering::Acquire) {
            // Wait for the stream to become ready.
            return 0;
        }
        let mut ring = v.ring();
        let (filled, index) = ring.write_index();
        let avail = (self.0.highwater_mark as i32).wrapping_sub(filled);
        let len = if avail >= 0 { buf.len().min(avail as usize) } else { buf.len() };
        let p = Arc::as_ptr(v);
        if filled < 0 {
            error_report(&format!("{p:p}: underrun write:{index} filled:{filled}"));
        } else if filled as usize + len > RINGBUFFER_SIZE as usize {
            error_report(&format!(
                "{p:p}: overrun write:{index} filled:{filled} + size:{len} > max:{RINGBUFFER_SIZE}"
            ));
        }
        ring.write(index, &buf[..len]);
        len
    }

    /// `qpw_buffer_get_free()`.
    fn buffer_get_free(&mut self, _hw: &HwCore) -> Option<usize> {
        let v = &self.0.v;
        if !v.streaming.load(Ordering::Acquire) {
            // Wait for the stream to become ready.
            return Some(0);
        }
        let (filled, _) = v.ring().write_index();
        Some((self.0.highwater_mark as i32).wrapping_sub(filled).max(0) as usize)
    }

    fn run_buffer_out(&mut self, hw: &mut HwCore) {
        generic_run_buffer_out(self, hw);
    }

    /// `qpw_enable_out()`.
    fn enable_out(&mut self, enable: bool) {
        self.0.set_enabled(enable);
    }

    /// `qpw_volume_out()`.
    fn volume_out(&mut self, vol: &Volume) {
        self.0.set_volume(vol);
    }

    /// `qpw_fini_out()`.
    fn fini_out(&mut self, _hw: &HwCore) {
        self.0.fini();
    }
}

impl PcmIn for PwVoiceIn {
    /// `qpw_read()`.
    fn read(&mut self, _info: &PcmInfo, buf: &mut [u8]) -> usize {
        let v = &self.0.v;
        if !v.streaming.load(Ordering::Acquire) {
            // Wait for the stream to become ready.
            return 0;
        }
        let mut ring = v.ring();
        let (avail, index) = ring.read_index();
        let len = buf.len().min(avail.max(0) as usize).min(RINGBUFFER_SIZE as usize);
        ring.read(index, &mut buf[..len]);
        len
    }

    fn run_buffer_in(&mut self, hw: &mut HwCore) {
        generic_run_buffer_in(self, hw);
    }

    /// `qpw_enable_in()`.
    fn enable_in(&mut self, enable: bool) {
        self.0.set_enabled(enable);
    }

    /// `qpw_volume_in()`.
    fn volume_in(&mut self, vol: &Volume) {
        self.0.set_volume(vol);
    }

    /// `qpw_fini_in()`.
    fn fini_in(&mut self, _hw: &HwCore) {
        self.0.fini();
    }
}

impl Driver for PwDriver {
    fn type_name(&self) -> &'static str {
        "audio-pipewire"
    }

    fn max_voices_out(&self) -> i32 {
        i32::MAX
    }

    fn max_voices_in(&self) -> i32 {
        i32::MAX
    }

    fn volume_out(&self) -> bool {
        true
    }

    fn volume_in(&self) -> bool {
        true
    }

    /// `audio_pw_realize()`.
    fn realize(&mut self, _dev: &Audiodev) -> Result<()> {
        self.conn = Some(PwConn::new()?);
        Ok(())
    }

    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let (voice, info, samples) = self.open(ctx, as_, false)?;
        Some(HwInit { pcm: Box::new(PwVoiceOut(voice)), info, samples, poll_mode: false })
    }

    fn init_in(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        let (voice, info, samples) = self.open(ctx, as_, true)?;
        Some(HwInit { pcm: Box::new(PwVoiceIn(voice)), info, samples, poll_mode: false })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_round_trip() {
        for fmt in [
            AudioFormat::U8,
            AudioFormat::S8,
            AudioFormat::U16,
            AudioFormat::S16,
            AudioFormat::U32,
            AudioFormat::S32,
            AudioFormat::F32,
        ] {
            for be in [false, true] {
                let (f, b, size) = pw_to_audfmt(audfmt_to_pw(fmt, be), be);
                assert_eq!((f, b), (fmt, be));
                assert_eq!(size * 8, crate::mixeng::format_bits(fmt) as u32);
            }
        }
    }

    #[test]
    fn ring_wraps() {
        let mut r = Ring::new();
        r.readindex = RINGBUFFER_SIZE - 2;
        r.writeindex = RINGBUFFER_SIZE - 2;
        let (filled, index) = r.write_index();
        assert_eq!(filled, 0);
        r.write(index, &[1, 2, 3, 4]);
        assert_eq!(r.read_index(), (4, RINGBUFFER_SIZE - 2));
        let mut out = [0; 4];
        r.read(RINGBUFFER_SIZE - 2, &mut out);
        assert_eq!(out, [1, 2, 3, 4]);
        assert_eq!(r.buf[..2], [3, 4]);
        assert_eq!(r.read_index(), (0, RINGBUFFER_SIZE + 2));
    }

    #[test]
    fn positions() {
        use spa::sys::*;
        assert_eq!(set_position(1), [SPA_AUDIO_CHANNEL_MONO]);
        assert_eq!(set_position(6)[5], SPA_AUDIO_CHANNEL_RR);
        assert_eq!(set_position(3), [SPA_AUDIO_CHANNEL_UNKNOWN; 3]);
    }
}
