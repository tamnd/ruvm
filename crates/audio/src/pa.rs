// SPDX-License-Identifier: GPL-2.0-or-later

//! The `pa` driver, QEMU's `audio/paaudio.c`, over the `libpulse-binding` crate.
//!
//! QEMU keeps one threaded main loop and context per server and takes the main loop lock around
//! every call into libpulse. The binding's main loop handle cannot leave the thread that made it,
//! so here each connection owns a worker thread that holds the main loop, the context and the
//! streams of every voice on it. A voice sends the worker a job, the worker runs it with the main
//! loop locked and sends the answer back, which is the same locking QEMU does with one more
//! thread in the way. The worker copies audio in and out, since a stream's buffers cannot leave
//! it either.

use std::collections::HashMap;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex, Weak};
use std::thread::JoinHandle;

use libpulse_binding::channelmap::{Map, Position};
use libpulse_binding::context::{self, Context};
use libpulse_binding::def::BufferAttr;
use libpulse_binding::error::{Code, PAErr};
use libpulse_binding::mainloop::threaded::Mainloop;
use libpulse_binding::sample::{Format, Spec};
use libpulse_binding::stream::{self, PeekResult, SeekMode, Stream};
use libpulse_binding::time::MicroSeconds;
use libpulse_binding::volume::{ChannelVolumes, Volume as PaVolume};
use ruvm_base::error::strerror;
use ruvm_base::report::error_report;
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{AudioFormat, Audiodev, AudiodevPaPerDirectionOptions, AudiodevU};

use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, Volume, application_name,
    error_printf,
};

/// `qpa_validate_per_direction_opts()` fills this in when the user leaves it out.
const DEFAULT_LATENCY: u32 = 46440;

/// The size of the buffer `audio_pa_realize()` prints the pid file path into.
const PIDFILE_LEN: usize = 64;

/// A C string ends at the first NUL.
fn c_str(s: &str) -> &str {
    s.split('\0').next().unwrap_or("")
}

/// `qpa_logerr()`.
fn logerr(err: PAErr, msg: &str) {
    let reason = err.to_string().unwrap_or_else(|| "(null)".to_string());
    error_printf(&format!("pulseaudio: {msg} Reason: {reason}\n"));
}

/// `audfmt_to_pa()`.
fn audfmt_to_pa(fmt: AudioFormat, big_endian: bool) -> Format {
    match (fmt, big_endian) {
        (AudioFormat::S8 | AudioFormat::U8, _) => Format::U8,
        (AudioFormat::S16 | AudioFormat::U16, false) => Format::S16le,
        (AudioFormat::S16 | AudioFormat::U16, true) => Format::S16be,
        (AudioFormat::S32 | AudioFormat::U32, false) => Format::S32le,
        (AudioFormat::S32 | AudioFormat::U32, true) => Format::S32be,
        (AudioFormat::F32, false) => Format::F32le,
        (AudioFormat::F32, true) => Format::F32be,
    }
}

/// `pa_to_audfmt()`, with the endianness it leaves alone for 8-bit samples.
fn pa_to_audfmt(fmt: Format, big_endian: bool) -> (AudioFormat, bool) {
    match fmt {
        Format::S16be => (AudioFormat::S16, true),
        Format::S16le => (AudioFormat::S16, false),
        Format::S32be => (AudioFormat::S32, true),
        Format::S32le => (AudioFormat::S32, false),
        Format::F32be => (AudioFormat::F32, true),
        Format::F32le => (AudioFormat::F32, false),
        _ => (AudioFormat::U8, big_endian),
    }
}

/// The pid file `audio_pa_realize()` looks for, cut to fit its buffer like `snprintf()` does.
fn pidfile(runtime: OsString) -> PathBuf {
    let mut p = runtime.into_vec();
    p.extend_from_slice(b"/pulse/pid");
    p.truncate(PIDFILE_LEN - 1);
    PathBuf::from(OsString::from_vec(p))
}

/// A device volume as libpulse takes it.
fn pa_volume(vol: &Volume) -> ChannelVolumes {
    let mut v = ChannelVolumes::default();
    v.init();
    let n = vol.channels.min(vol.vol.len());
    v.set_len(n as u8);
    for (dst, &src) in v.get_mut().iter_mut().zip(&vol.vol[..n]) {
        *dst = PaVolume((PaVolume::NORMAL.0 - PaVolume::MUTED.0) * u32::from(src) / 255);
    }
    v
}

/// One stream on a connection, with the fragment `pa_stream_peek()` last gave a capture voice.
struct Slot {
    stream: Stream,
    frag: Vec<u8>,
    pos: usize,
}

impl Drop for Slot {
    fn drop(&mut self) {
        // libpulse keeps the stream until the server answers the disconnect and then calls the
        // state callback, whose closure the binding frees along with the stream.
        self.stream.set_state_callback(None);
    }
}

impl Slot {
    /// `read_length`.
    fn left(&self) -> usize {
        self.frag.len() - self.pos
    }
}

/// What a connection's worker thread owns, QEMU's `PAConnection` less the list fields.
struct Worker {
    ml: Mainloop,
    ctx: Context,
    slots: HashMap<u64, Slot>,
}

type Job = Box<dyn FnOnce(&mut Worker) + Send>;

/// A second handle on the main loop for a state callback to wake the waiter with.
fn signaller(ml: &Mainloop) -> Box<dyn FnMut()> {
    let mut ml = Mainloop { _inner: Rc::clone(&ml._inner) };
    Box::new(move || ml.signal(false))
}

impl Worker {
    /// `CHECK_DEAD_GOTO`: whether the context and the stream of `id` still work. It logs
    /// `msg` when they do not.
    fn alive(&self, id: u64, msg: &str) -> bool {
        let cs = self.ctx.get_state();
        let ss = self.slots.get(&id).map(|s| s.stream.get_state());
        if cs.is_good() && ss.is_some_and(stream::State::is_good) {
            return true;
        }
        if cs == context::State::Failed || ss == Some(stream::State::Failed) {
            logerr(self.ctx.errno(), msg);
        } else {
            logerr(PAErr::from(Code::BadState), msg);
        }
        false
    }

    fn slot(&mut self, id: u64) -> &mut Slot {
        self.slots.get_mut(&id).expect("stream of a live voice")
    }

    /// `qpa_simple_new()`, with the error to log returned.
    fn simple_new(&mut self, id: u64, new: NewStream) -> std::result::Result<(), NewError> {
        let mut map = Map::default();
        map.init();
        map.set_len(new.ss.channels);
        use Position::*;
        let positions: &[Position] = match new.ss.channels {
            1 => &[Mono],
            2 => &[FrontLeft, FrontRight],
            6 => &[FrontLeft, FrontRight, FrontCenter, Lfe, RearLeft, RearRight],
            8 => {
                &[FrontLeft, FrontRight, FrontCenter, Lfe, RearLeft, RearRight, SideLeft, SideRight]
            }
            // The caller reports it, where the location of the device is current.
            n => return Err(NewError::Channels(n, self.ctx.errno())),
        };
        map.get_mut().copy_from_slice(positions);

        let Some(mut stream) = Stream::new(&mut self.ctx, &new.name, &new.ss, Some(&map)) else {
            return Err(NewError::Pa(self.ctx.errno()));
        };
        stream.set_state_callback(Some(signaller(&self.ml)));
        let mut flags = stream::FlagSet::EARLY_REQUESTS;
        if new.dev.is_some() {
            // Don't move the stream if the user named a sink or source.
            flags |= stream::FlagSet::DONT_MOVE;
        }
        let dev = new.dev.as_deref();
        let r = if new.in_ {
            stream.connect_record(dev, Some(&new.attr), flags)
        } else {
            stream.connect_playback(dev, Some(&new.attr), flags, None, None)
        };
        if r.is_err() {
            // Read the error before dropping the stream, which disconnects it again.
            let e = self.ctx.errno();
            stream.set_state_callback(None);
            drop(stream);
            return Err(NewError::Pa(e));
        }
        self.slots.insert(id, Slot { stream, frag: Vec::new(), pos: 0 });
        Ok(())
    }

    /// `qpa_simple_disconnect()`.
    fn simple_disconnect(&mut self, id: u64) {
        let Some(mut slot) = self.slots.remove(&id) else { return };
        // Wait until it is connected, for PulseAudio bug #247.
        while slot.stream.get_state() == stream::State::Creating {
            self.ml.wait();
        }
        if let Err(e) = slot.stream.disconnect() {
            error_report(&format!("pulseaudio: Failed to disconnect! err={}", e.0));
        }
    }

    /// Peeks the next fragment of a capture stream when the last one is used up.
    fn peek(&mut self, id: u64) -> bool {
        let slot = self.slot(id);
        if slot.left() > 0 {
            return true;
        }
        let frag = match slot.stream.peek() {
            Ok(PeekResult::Empty) => Vec::new(),
            // QEMU would read a hole from a null pointer; silence is what it stands for.
            Ok(PeekResult::Hole(n)) => vec![0; n],
            Ok(PeekResult::Data(d)) => d.to_vec(),
            Err(_) => {
                logerr(self.ctx.errno(), "pa_stream_peek failed");
                return false;
            }
        };
        let slot = self.slot(id);
        slot.frag = frag;
        slot.pos = 0;
        true
    }

    /// Uses `n` bytes of the fragment and drops it once it is used up.
    fn advance(&mut self, id: u64, n: usize) -> bool {
        let slot = self.slot(id);
        assert!(n <= slot.left());
        slot.pos += n;
        if n > 0 && slot.left() == 0 {
            slot.frag.clear();
            slot.pos = 0;
            if slot.stream.discard().is_err() {
                logerr(self.ctx.errno(), "pa_stream_drop failed");
                return false;
            }
        }
        true
    }

    /// `pa_context_set_sink_input_volume()` and the rest, which return no operation when they
    /// refuse. The binding cannot take that, so the checks libpulse makes come first.
    fn set_volume(&mut self, id: u64, vol: &Volume, in_: bool) {
        let index = self.slot(id).stream.get_index();
        let v = pa_volume(vol);
        let ready = self.ctx.get_state() == context::State::Ready;
        let mut intro = self.ctx.introspect();
        let (vname, mname) = if in_ {
            ("set_source_output_volume() failed", "set_source_output_mute() failed")
        } else {
            ("set_sink_input_volume() failed", "set_sink_input_mute() failed")
        };
        let check = |valid: bool| match (ready, index) {
            (false, _) => Err(PAErr::from(Code::BadState)),
            (true, None) => Err(PAErr::from(Code::Invalid)),
            (true, Some(_)) if !valid => Err(PAErr::from(Code::Invalid)),
            (true, Some(i)) => Ok(i),
        };
        match check(v.is_valid()) {
            Ok(i) if in_ => drop(intro.set_source_output_volume(i, &v, None)),
            Ok(i) => drop(intro.set_sink_input_volume(i, &v, None)),
            Err(e) => logerr(e, vname),
        }
        match check(true) {
            Ok(i) if in_ => drop(intro.set_source_output_mute(i, vol.mute, None)),
            Ok(i) => drop(intro.set_sink_input_mute(i, vol.mute, None)),
            Err(e) => logerr(e, mname),
        }
    }
}

/// Why `qpa_simple_new()` failed: a channel count it has no map for, or a libpulse error.
enum NewError {
    Channels(u8, PAErr),
    Pa(PAErr),
}

/// What `qpa_simple_new()` takes.
struct NewStream {
    name: String,
    in_: bool,
    dev: Option<String>,
    ss: Spec,
    attr: BufferAttr,
}

/// A connection to one server, QEMU's `PAConnection`. Dropping the last reference stops the
/// worker, which runs `qpa_conn_fini()`.
#[derive(Debug)]
struct PaConn {
    tx: Option<Sender<Job>>,
    worker: Option<JoinHandle<()>>,
    next_id: AtomicU64,
}

/// `pa_conns`, oldest first.
static CONNS: Mutex<Vec<Weak<PaConn>>> = Mutex::new(Vec::new());

impl PaConn {
    /// `qpa_conn_init()`.
    fn new(server: Option<String>) -> Option<Arc<PaConn>> {
        let name = application_name();
        let (tx, rx) = mpsc::channel::<Job>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = std::thread::Builder::new()
            .name("pulseaudio".to_string())
            .spawn(move || conn_run(server, name, ready_tx, rx))
            .ok()?;
        if ready_rx.recv() != Ok(true) {
            let _ = worker.join();
            return None;
        }
        Some(Arc::new(PaConn { tx: Some(tx), worker: Some(worker), next_id: AtomicU64::new(0) }))
    }

    /// Runs `f` on the worker with the main loop locked. `None` means the worker is gone.
    fn call<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Worker) -> R + Send + 'static,
    ) -> Option<R> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.tx
            .as_ref()?
            .send(Box::new(move |w: &mut Worker| {
                let _ = tx.send(f(w));
            }))
            .ok()?;
        rx.recv().ok()
    }
}

impl Drop for PaConn {
    fn drop(&mut self) {
        self.tx = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// `qpa_conn_fini()`. The main loop goes last, when `ml` drops.
fn conn_fini(mut ml: Mainloop, ctx: Option<Context>) {
    ml.stop();
    if let Some(mut ctx) = ctx {
        ctx.disconnect();
    }
}

/// The worker thread of a connection: `qpa_conn_init()`, then the voices' jobs until the last
/// reference goes, then `qpa_conn_fini()`.
fn conn_run(server: Option<String>, name: String, ready: SyncSender<bool>, rx: Receiver<Job>) {
    let Some(mut ml) = Mainloop::new() else {
        let _ = ready.send(false);
        return;
    };
    let Some(mut ctx) = Context::new(&ml, c_str(&name)) else {
        conn_fini(ml, None);
        let _ = ready.send(false);
        return;
    };
    ctx.set_state_callback(Some(signaller(&ml)));
    if ctx.connect(server.as_deref().map(c_str), context::FlagSet::NOFLAGS, None).is_err() {
        logerr(ctx.errno(), "pa_context_connect() failed");
        conn_fini(ml, Some(ctx));
        let _ = ready.send(false);
        return;
    }
    ml.lock();
    let mut ok = ml.start().is_ok();
    while ok {
        let state = ctx.get_state();
        if state == context::State::Ready {
            break;
        }
        if !state.is_good() {
            logerr(ctx.errno(), "Wrong context state");
            ok = false;
            break;
        }
        // Wait until the context is ready.
        ml.wait();
    }
    ml.unlock();
    if !ok {
        conn_fini(ml, Some(ctx));
        let _ = ready.send(false);
        return;
    }
    let _ = ready.send(true);

    let mut w = Worker { ml, ctx, slots: HashMap::new() };
    for job in rx {
        w.ml.lock();
        job(&mut w);
        w.ml.unlock();
    }
    let Worker { mut ml, ctx, slots } = w;
    ml.stop();
    drop(slots);
    conn_fini(ml, Some(ctx));
}

/// The `pa` driver, QEMU's `AudioPa`.
#[derive(Debug, Default)]
pub struct PaDriver {
    conn: Option<Arc<PaConn>>,
}

fn pa_opts(dev: &Audiodev, in_: bool) -> AudiodevPaPerDirectionOptions {
    match &dev.u {
        AudiodevU::Pa(o) => (if in_ { &o.in_ } else { &o.out }).clone().unwrap_or_default(),
        _ => AudiodevPaPerDirectionOptions::default(),
    }
}

impl PaDriver {
    /// `qpa_init_out()` and `qpa_init_in()`: the stream and the layout it runs at.
    fn open(
        &self,
        ctx: &InitCtx<'_>,
        as_: &AudSettings,
        in_: bool,
    ) -> Option<(Arc<PaConn>, u64, HwInitParts)> {
        let conn = self.conn.clone()?;
        let ppdo = pa_opts(ctx.dev, in_);
        let latency = ppdo.latency.unwrap_or(DEFAULT_LATENCY);
        let tp = ctx.dev.timer_period.unwrap_or(10000);
        let ss = Spec {
            format: audfmt_to_pa(as_.fmt, as_.big_endian),
            channels: as_.nchannels as u8,
            rate: as_.freq as u32,
        };
        let bytes = |usec: u32| ss.usec_to_bytes(MicroSeconds(u64::from(usec))) as u32;
        let attr = if in_ {
            BufferAttr {
                maxlength: bytes(latency.max(tp.wrapping_mul(3))),
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: bytes((tp >> 1).wrapping_mul(3)),
            }
        } else {
            BufferAttr {
                maxlength: u32::MAX,
                tlength: bytes(latency),
                prebuf: u32::MAX,
                minreq: bytes((latency >> 2).min((tp >> 2).wrapping_mul(3))),
                fragsize: u32::MAX,
            }
        };
        let (fmt, big_endian) = pa_to_audfmt(ss.format, as_.big_endian);
        let obt = AudSettings { fmt, big_endian, ..*as_ };

        let new = NewStream {
            name: c_str(ppdo.stream_name.as_deref().unwrap_or(&ctx.dev.id)).to_string(),
            in_,
            dev: ppdo.name.as_deref().map(|d| c_str(d).to_string()),
            ss,
            attr,
        };
        let id = conn.next_id.fetch_add(1, Ordering::Relaxed);
        let r = conn.call(move |w| w.simple_new(id, new));
        let r = r.unwrap_or(Err(NewError::Pa(PAErr::from(Code::Killed))));
        if let Err(e) = r {
            let e = match e {
                NewError::Channels(n, e) => {
                    error_report(&format!("pulseaudio: unsupported channel count {n}"));
                    e
                }
                NewError::Pa(e) => e,
            };
            let what = if in_ { "capture" } else { "playback" };
            logerr(e, &format!("pa_simple_new for {what} failed"));
            return None;
        }
        let samples = ctx.pdo.buffer_frames(&obt, DEFAULT_LATENCY);
        Some((conn, id, HwInitParts { info: PcmInfo::new(&obt), samples: samples as usize }))
    }
}

struct HwInitParts {
    info: PcmInfo,
    samples: usize,
}

const LOCK_FAILED: &str = "pa_threaded_mainloop_lock failed";

/// A playback voice, QEMU's `PAVoiceOut`. `buf` stands in for the buffer `pa_stream_begin_write()`
/// hands out.
struct PaVoiceOut {
    conn: Arc<PaConn>,
    id: u64,
    buf: Vec<u8>,
}

/// A capture voice, QEMU's `PAVoiceIn`. `buf` holds the copy of the fragment `get_buffer_in` returns.
struct PaVoiceIn {
    conn: Arc<PaConn>,
    id: u64,
    buf: Vec<u8>,
}

impl PcmOut for PaVoiceOut {
    /// `qpa_write()`.
    fn write(&mut self, _info: &PcmInfo, buf: &[u8]) -> usize {
        let id = self.id;
        let data = buf.to_vec();
        self.conn
            .call(move |w| {
                if !w.alive(id, LOCK_FAILED) {
                    return 0;
                }
                let slot = w.slot(id);
                if slot.stream.get_state() != stream::State::Ready {
                    // Wait for the stream to become ready.
                    return 0;
                }
                let Some(l) = slot.stream.writable_size() else {
                    logerr(w.ctx.errno(), "pa_stream_writable_size failed");
                    return 0;
                };
                let l = l.min(data.len());
                if slot.stream.write(&data[..l], None, 0, SeekMode::Relative).is_err() {
                    logerr(w.ctx.errno(), "pa_stream_write failed");
                    return 0;
                }
                l
            })
            .unwrap_or(0)
    }

    /// `qpa_buffer_get_free()`.
    fn buffer_get_free(&mut self, _hw: &HwCore) -> Option<usize> {
        let id = self.id;
        let free = self.conn.call(move |w| {
            if !w.alive(id, LOCK_FAILED) {
                return 0;
            }
            let slot = w.slot(id);
            if slot.stream.get_state() != stream::State::Ready {
                // Wait for the stream to become ready.
                return 0;
            }
            match slot.stream.writable_size() {
                Some(l) => l,
                None => {
                    logerr(w.ctx.errno(), "pa_stream_writable_size failed");
                    0
                }
            }
        });
        Some(free.unwrap_or(0))
    }

    /// `qpa_get_buffer_out()`: the size libpulse would hand out, with a local buffer of that
    /// size to mix into.
    fn get_buffer_out<'a>(
        &'a mut self,
        _hw: &'a mut HwCore,
        _size: usize,
    ) -> (Option<&'a mut [u8]>, usize) {
        let id = self.id;
        let size = self.conn.call(move |w| {
            if !w.alive(id, LOCK_FAILED) {
                return 0;
            }
            let slot = w.slot(id);
            let n = match slot.stream.begin_write(None) {
                Ok(b) => b.map_or(0, |b| b.len()),
                Err(_) => {
                    logerr(w.ctx.errno(), "pa_stream_begin_write failed");
                    return 0;
                }
            };
            let _ = slot.stream.cancel_write();
            n
        });
        match size {
            Some(n) if n > 0 => {
                self.buf.resize(n, 0);
                (Some(&mut self.buf[..n]), n)
            }
            _ => (None, 0),
        }
    }

    /// `qpa_put_buffer_out()`.
    fn put_buffer_out(&mut self, _hw: &mut HwCore, size: usize) -> usize {
        let id = self.id;
        let data = self.buf[..size].to_vec();
        self.conn
            .call(move |w| {
                if !w.alive(id, LOCK_FAILED) {
                    return 0;
                }
                if w.slot(id).stream.write(&data, None, 0, SeekMode::Relative).is_err() {
                    logerr(w.ctx.errno(), "pa_stream_write failed");
                    return 0;
                }
                data.len()
            })
            .unwrap_or(0)
    }

    /// `qpa_volume_out()`.
    fn volume_out(&mut self, vol: &Volume) {
        let (id, vol) = (self.id, *vol);
        self.conn.call(move |w| w.set_volume(id, &vol, false));
    }

    /// `qpa_fini_out()`.
    fn fini_out(&mut self, _hw: &HwCore) {
        let id = self.id;
        self.conn.call(move |w| w.simple_disconnect(id));
    }
}

impl PcmIn for PaVoiceIn {
    /// `qpa_read()`.
    fn read(&mut self, _info: &PcmInfo, buf: &mut [u8]) -> usize {
        let (id, length) = (self.id, buf.len());
        let data = self.conn.call(move |w| {
            let mut data = Vec::new();
            if !w.alive(id, LOCK_FAILED) {
                return data;
            }
            if w.slot(id).stream.get_state() != stream::State::Ready {
                // Wait for the stream to become ready.
                return data;
            }
            while data.len() < length {
                if !w.peek(id) {
                    return Vec::new();
                }
                let slot = w.slot(id);
                if slot.left() == 0 {
                    // The buffer is empty.
                    break;
                }
                let l = slot.left().min(length - data.len());
                data.extend_from_slice(&slot.frag[slot.pos..slot.pos + l]);
                if !w.advance(id, l) {
                    return Vec::new();
                }
            }
            data
        });
        let data = data.unwrap_or_default();
        buf[..data.len()].copy_from_slice(&data);
        data.len()
    }

    /// `qpa_get_buffer_in()`.
    fn get_buffer_in<'a>(&'a mut self, _hw: &'a mut HwCore, size: usize) -> &'a [u8] {
        let id = self.id;
        let data = self.conn.call(move |w| {
            if !w.alive(id, LOCK_FAILED) || !w.peek(id) {
                return Vec::new();
            }
            let slot = w.slot(id);
            let n = slot.left().min(size);
            slot.frag[slot.pos..slot.pos + n].to_vec()
        });
        self.buf = data.unwrap_or_default();
        &self.buf
    }

    /// `qpa_put_buffer_in()`.
    fn put_buffer_in(&mut self, _hw: &mut HwCore, size: usize) {
        let id = self.id;
        self.conn.call(move |w| {
            if w.alive(id, LOCK_FAILED) {
                w.advance(id, size);
            }
        });
    }

    /// `qpa_volume_in()`.
    fn volume_in(&mut self, vol: &Volume) {
        let (id, vol) = (self.id, *vol);
        self.conn.call(move |w| w.set_volume(id, &vol, true));
    }

    /// `qpa_fini_in()`.
    fn fini_in(&mut self, _hw: &HwCore) {
        let id = self.id;
        self.conn.call(move |w| {
            let Some(slot) = w.slots.get_mut(&id) else { return };
            if slot.left() > 0 {
                slot.frag.clear();
                slot.pos = 0;
                if slot.stream.discard().is_err() {
                    logerr(w.ctx.errno(), "pa_stream_drop failed");
                }
            }
            w.simple_disconnect(id);
        });
    }
}

impl Driver for PaDriver {
    fn type_name(&self) -> &'static str {
        "audio-pa"
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

    /// `audio_pa_realize()`.
    fn realize(&mut self, dev: &Audiodev) -> Result<()> {
        let server = match &dev.u {
            AudiodevU::Pa(o) => o.server.as_deref().map(|s| c_str(s).to_string()),
            _ => None,
        };
        if server.is_none() {
            let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") else {
                return Err(Error::generic("XDG_RUNTIME_DIR not set"));
            };
            let pidfile = pidfile(runtime);
            if let Err(e) = std::fs::metadata(&pidfile) {
                return Err(Error::generic(format!(
                    "could not stat pidfile {}: {}",
                    String::from_utf8_lossy(pidfile.as_os_str().as_bytes()),
                    strerror(&e)
                )));
            }
        }

        let mut conns = CONNS.lock().unwrap_or_else(|e| e.into_inner());
        conns.retain(|c| c.strong_count() > 0);
        // QEMU never records the server of a connection, so an audiodev without one shares the
        // first connection there is and one with a server always makes its own.
        let mut conn = match server {
            None => conns.iter().find_map(Weak::upgrade),
            Some(_) => None,
        };
        if conn.is_none() {
            conn = PaConn::new(server);
            if let Some(c) = &conn {
                conns.push(Arc::downgrade(c));
            }
        }
        match conn {
            Some(c) => {
                self.conn = Some(c);
                Ok(())
            }
            None => Err(Error::generic("could not connect to PulseAudio server")),
        }
    }

    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let (conn, id, p) = self.open(ctx, as_, false)?;
        Some(HwInit {
            pcm: Box::new(PaVoiceOut { conn, id, buf: Vec::new() }),
            info: p.info,
            samples: p.samples,
            poll_mode: false,
        })
    }

    fn init_in(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        let (conn, id, p) = self.open(ctx, as_, true)?;
        Some(HwInit {
            pcm: Box::new(PaVoiceIn { conn, id, buf: Vec::new() }),
            info: p.info,
            samples: p.samples,
            poll_mode: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats() {
        assert_eq!(audfmt_to_pa(AudioFormat::S8, true), Format::U8);
        assert_eq!(audfmt_to_pa(AudioFormat::U16, true), Format::S16be);
        assert_eq!(audfmt_to_pa(AudioFormat::U32, false), Format::S32le);
        assert_eq!(pa_to_audfmt(Format::U8, true), (AudioFormat::U8, true));
        assert_eq!(pa_to_audfmt(Format::F32be, false), (AudioFormat::F32, true));
        for fmt in [AudioFormat::S16, AudioFormat::S32, AudioFormat::F32] {
            for be in [false, true] {
                assert_eq!(pa_to_audfmt(audfmt_to_pa(fmt, be), !be), (fmt, be));
            }
        }
    }

    #[test]
    fn pidfile_is_cut_like_snprintf() {
        assert_eq!(pidfile("/run/user/0".into()), PathBuf::from("/run/user/0/pulse/pid"));
        let long = "/".to_string() + &"d".repeat(60);
        let p = pidfile(long.clone().into());
        assert_eq!(p.as_os_str().len(), PIDFILE_LEN - 1);
        assert_eq!(p, PathBuf::from(long + "/p"));
    }

    #[test]
    fn volumes() {
        let mut vol = Volume { mute: false, channels: 2, vol: [0; 16] };
        vol.vol[0] = 255;
        vol.vol[1] = 128;
        let v = pa_volume(&vol);
        assert_eq!(v.len(), 2);
        assert_eq!(v.get()[0], PaVolume(0x10000));
        assert_eq!(v.get()[1], PaVolume(0x10000 * 128 / 255));
        assert!(!pa_volume(&Volume::default()).is_valid());
    }
}
