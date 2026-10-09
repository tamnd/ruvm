// SPDX-License-Identifier: GPL-2.0-or-later

//! An audio backend with the mixing engine, QEMU's `AudioMixengBackend`.
//!
//! This ports `audio/audio-mixeng-be.c` and `audio/audio_template.h`. A device opens soft
//! voices on a backend. Each soft voice resamples into the mix buffer of a hard voice, which is a
//! stream the host driver opened, and the audio timer on the virtual clock clips the mix out to
//! the driver every `timer-period` microseconds. Captures tap the mix of every hard voice.
//!
//! The C code calls device callbacks and capture callbacks with the backend in whatever state it
//! is in. Here the backend state sits behind a lock, and callbacks always run with it released:
//! device callbacks are called between steps of the timer run, and capture callbacks are queued
//! and called when the step that raised them lets go of the lock. Voices and hard voices are
//! looked up by id again after every callback, so a callback that closes its own voice is safe.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use ruvm_base::report::{error_report, warn_report};
use ruvm_hw_core::timer::{Clock, Timer, muldiv64};
use ruvm_qapi::types::Audiodev;

use crate::mixeng::{self, MixengVolume, NOMINAL_VOLUME, Rate, SampleFmt, StSample};
use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, Pdo, VClock, Volume,
    ring_posb,
};

/// A device's voice callback. It gets the number of bytes the voice can take or has ready.
pub type AudioCallback = Arc<dyn Fn(usize) + Send + Sync>;

/// What a capture is told when the playback it taps starts or stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureNotify {
    /// Some voice on the backend is playing.
    Enable,
    /// No voice is playing.
    Disable,
}

/// The callbacks of a capture, QEMU's `struct audio_capture_ops`.
pub trait CaptureOps: Send + Sync {
    /// Playback started or stopped.
    fn notify(&self, cmd: CaptureNotify);
    /// Mixed audio in the capture's format.
    fn capture(&self, buf: &[u8]);
    /// The capture is going away.
    fn destroy(&self);
}

/// A playback voice a device opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwVoiceOut(u64);

/// A capture voice a device opened.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SwVoiceIn(u64);

/// One callback set added with [`AudioBackend::add_capture`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CaptureHandle {
    cap: u64,
    cb: u64,
}

enum Deferred {
    Notify(Arc<dyn CaptureOps>, CaptureNotify),
    Capture(Arc<dyn CaptureOps>, Vec<u8>),
    Destroy(Arc<dyn CaptureOps>),
}

struct HwOut {
    id: u64,
    enabled: bool,
    poll_mode: bool,
    pending_disable: bool,
    core: HwCore,
    clip: SampleFmt,
    mix_buf: Vec<StSample>,
    mix_pos: usize,
    sw_head: Vec<u64>,
    cap_head: Vec<u64>,
    pcm: Box<dyn PcmOut>,
}

struct HwIn {
    id: u64,
    enabled: bool,
    poll_mode: bool,
    core: HwCore,
    conv: SampleFmt,
    conv_buf: Vec<StSample>,
    conv_pos: usize,
    total_samples_captured: usize,
    sw_head: Vec<u64>,
    pcm: Box<dyn PcmIn>,
}

struct SwOut {
    hw: u64,
    info: PcmInfo,
    conv: SampleFmt,
    resample_buf: Vec<StSample>,
    rb_pos: usize,
    rate: Option<Rate>,
    total_hw_samples_mixed: usize,
    active: bool,
    empty: bool,
    name: String,
    vol: MixengVolume,
    callback: Option<AudioCallback>,
}

struct SwIn {
    hw: u64,
    info: PcmInfo,
    clip: SampleFmt,
    resample_buf: Vec<StSample>,
    rate: Option<Rate>,
    total_hw_samples_acquired: usize,
    active: bool,
    name: String,
    vol: MixengVolume,
    callback: Option<AudioCallback>,
}

/// The soft voice a capture keeps on each hard voice, QEMU's `SWVoiceCap`.
struct SwCap {
    hw: u64,
    cap: u64,
    rate: Rate,
    total_hw_samples_mixed: usize,
    active: bool,
    empty: bool,
}

struct Capture {
    id: u64,
    enabled: bool,
    info: PcmInfo,
    clip: SampleFmt,
    mix_buf: Vec<StSample>,
    mix_pos: usize,
    buf: Vec<u8>,
    sw_head: Vec<u64>,
    cbs: Vec<(u64, Arc<dyn CaptureOps>)>,
}

struct State {
    nb_hw_voices_out: i32,
    nb_hw_voices_in: i32,
    hw_out: Vec<HwOut>,
    hw_in: Vec<HwIn>,
    caps: Vec<Capture>,
    sw_out: BTreeMap<u64, SwOut>,
    sw_in: BTreeMap<u64, SwIn>,
    sw_cap: BTreeMap<u64, SwCap>,
    next_id: u64,
    deferred: Vec<Deferred>,
}

impl State {
    fn new_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }
}

fn find_out(v: &mut [HwOut], id: u64) -> Option<&mut HwOut> {
    v.iter_mut().find(|h| h.id == id)
}

fn find_in(v: &mut [HwIn], id: u64) -> Option<&mut HwIn> {
    v.iter_mut().find(|h| h.id == id)
}

fn find_cap(v: &mut [Capture], id: u64) -> Option<&mut Capture> {
    v.iter_mut().find(|c| c.id == id)
}

/// `audio_bug()`, which QEMU prints with the function name.
fn audio_bug(func: &str, msg: &str) {
    error_report(&format!("{func}: {msg}"));
}

fn sw_name(name: &str) -> &str {
    if name.is_empty() { "unknown" } else { name }
}

/// `audio_pcm_hw_get_free()`: free frames in the host buffer.
fn hw_get_free(hw: &mut HwOut) -> usize {
    hw.pcm.buffer_get_free(&hw.core).unwrap_or(i32::MAX as usize) / hw.core.info.bytes_per_frame
}

/// `audio_pcm_hw_find_min_out()` and `audio_pcm_hw_get_live_out()` over any list of voices,
/// given as (active, empty, mixed).
fn get_live_out(
    func: &str,
    size: usize,
    sws: impl Iterator<Item = (bool, bool, usize)>,
) -> (usize, i32) {
    let mut m = usize::MAX;
    let mut nb_live = 0;
    for (active, empty, mixed) in sws {
        if active || !empty {
            m = m.min(mixed);
            nb_live += 1;
        }
    }
    if nb_live == 0 {
        return (0, 0);
    }
    if m > size {
        audio_bug(func, &format!("live={m} hw->mix_buf.size={size}"));
        return (0, nb_live);
    }
    (m, nb_live)
}

/// `audio_pcm_sw_resample_out()`: resamples `src` into the mix ring after the `live` frames
/// already there.
fn resample_out(
    rate: &mut Rate,
    src: &[StSample],
    frames_in_max: usize,
    frames_out_max: usize,
    mix_buf: &mut [StSample],
    mix_pos: usize,
    live: usize,
) -> (usize, usize) {
    let size = mix_buf.len();
    let mut wpos = (mix_pos + live) % size;
    let out_lim = frames_out_max.min(size - wpos);
    let (fi, fo) = rate.flow(&src[..frames_in_max], &mut mix_buf[wpos..wpos + out_lim], true);
    wpos += fo;
    let (mut total_in, mut total_out) = (fi, fo);
    if frames_in_max - fi > 0 && wpos == size {
        let (fi2, fo2) =
            rate.flow(&src[fi..frames_in_max], &mut mix_buf[..frames_out_max - fo], true);
        total_in += fi2;
        total_out += fo2;
    }
    (total_in, total_out)
}

/// `audio_pcm_hw_clip_out()`: clips `len` frames from the mix position into `dst`.
fn clip_out(
    hw_clip: &SampleFmt,
    mix_buf: &[StSample],
    pos: usize,
    bpf: usize,
    dst: &mut [u8],
    len: usize,
) {
    let size = mix_buf.len();
    let mut pos = pos;
    let mut clipped = 0;
    let mut len = len;
    while len > 0 {
        let n = len.min(size - pos);
        hw_clip.clip(&mut dst[clipped * bpf..], &mix_buf[pos..pos + n]);
        pos = (pos + n) % size;
        len -= n;
        clipped += n;
    }
}

/// `audio_capture_maybe_changed()`.
fn capture_maybe_changed(cap: &mut Capture, enabled: bool, deferred: &mut Vec<Deferred>) {
    if cap.enabled != enabled {
        cap.enabled = enabled;
        let cmd = if enabled { CaptureNotify::Enable } else { CaptureNotify::Disable };
        for (_, ops) in &cap.cbs {
            deferred.push(Deferred::Notify(ops.clone(), cmd));
        }
    }
}

/// `audio_recalc_and_notify_capture()`.
fn recalc_and_notify_capture(
    caps: &mut [Capture],
    sw_cap: &BTreeMap<u64, SwCap>,
    cap_id: u64,
    deferred: &mut Vec<Deferred>,
) {
    let Some(cap) = find_cap(caps, cap_id) else { return };
    let enabled = cap.sw_head.iter().any(|id| sw_cap.get(id).is_some_and(|s| s.active));
    capture_maybe_changed(cap, enabled, deferred);
}

/// A backend for one audiodev.
pub struct AudioBackend {
    dev: Audiodev,
    pdo_in: Pdo,
    pdo_out: Pdo,
    driver: Box<dyn Driver>,
    state: Mutex<State>,
    clock: VClock,
    timer: OnceLock<Timer>,
    period_ticks: i64,
    running: AtomicBool,
    weak: Weak<AudioBackend>,
}

impl fmt::Debug for AudioBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioBackend")
            .field("id", &self.dev.id)
            .field("driver", &self.driver.type_name())
            .finish_non_exhaustive()
    }
}

impl AudioBackend {
    /// Creates a backend for a validated audiodev, like `audio_mixeng_backend_realize()`.
    pub(crate) fn new(
        dev: Audiodev,
        pdo_in: Pdo,
        pdo_out: Pdo,
        driver: Box<dyn Driver>,
        running: bool,
    ) -> Arc<Self> {
        let be_name = driver.type_name();
        let nb_out =
            init_nb_voices(be_name, true, pdo_out.voices as i32, driver.max_voices_out(), 1);
        let nb_in = init_nb_voices(be_name, false, pdo_in.voices as i32, driver.max_voices_in(), 0);
        let period_ticks = match dev.timer_period.unwrap_or(10000) {
            0 => 1,
            p => i64::from(p) * 1000,
        };
        Arc::new_cyclic(|weak| AudioBackend {
            dev,
            pdo_in,
            pdo_out,
            driver,
            state: Mutex::new(State {
                nb_hw_voices_out: nb_out,
                nb_hw_voices_in: nb_in,
                hw_out: Vec::new(),
                hw_in: Vec::new(),
                caps: Vec::new(),
                sw_out: BTreeMap::new(),
                sw_in: BTreeMap::new(),
                sw_cap: BTreeMap::new(),
                next_id: 0,
                deferred: Vec::new(),
            }),
            clock: VClock::default(),
            timer: OnceLock::new(),
            period_ticks,
            running: AtomicBool::new(running),
            weak: weak.clone(),
        })
    }

    /// The audiodev id, `audio_be_get_id()`.
    pub fn id(&self) -> &str {
        &self.dev.id
    }

    /// The audiodev this backend was created from.
    pub fn audiodev(&self) -> &Audiodev {
        &self.dev
    }

    /// The validated playback options.
    pub fn pdo_out(&self) -> &Pdo {
        &self.pdo_out
    }

    /// The validated capture options.
    pub fn pdo_in(&self) -> &Pdo {
        &self.pdo_in
    }

    /// Paces the backend by `clock`, the machine's virtual clock. QEMU's backends use the one
    /// global virtual clock; here the machine hands it over once it exists. Only the first
    /// clock counts.
    pub fn attach_clock(&self, clock: &Arc<Clock>) {
        if !self.clock.set(clock.clone()) {
            return;
        }
        let weak = self.weak.clone();
        let timer = clock.new_timer(move || {
            if let Some(be) = weak.upgrade() {
                be.timer_fire();
            }
        });
        let _ = self.timer.set(timer);
        let st = self.lock();
        self.reset_timer(&st);
        self.unlock(st);
    }

    /// The VM started or stopped, `audio_vm_change_state_handler()`.
    pub fn vm_state_change(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
        self.with_state(|st| {
            for hw in st.hw_out.iter_mut().filter(|h| h.enabled) {
                hw.pcm.enable_out(running);
            }
            for hw in st.hw_in.iter_mut().filter(|h| h.enabled) {
                hw.pcm.enable_in(running);
            }
            self.reset_timer(st);
        });
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn unlock(&self, mut st: MutexGuard<'_, State>) {
        let deferred = std::mem::take(&mut st.deferred);
        drop(st);
        for d in deferred {
            match d {
                Deferred::Notify(ops, cmd) => ops.notify(cmd),
                Deferred::Capture(ops, buf) => ops.capture(&buf),
                Deferred::Destroy(ops) => ops.destroy(),
            }
        }
    }

    fn with_state<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut st = self.lock();
        let r = f(&mut st);
        self.unlock(st);
        r
    }

    fn timer_fire(&self) {
        self.run();
        self.with_state(|st| self.reset_timer(st));
    }

    /// `audio_reset_timer()`.
    fn reset_timer(&self, st: &State) {
        let Some(timer) = self.timer.get() else { return };
        let needed = st.hw_out.iter().any(|h| h.enabled && !h.poll_mode)
            || st.hw_in.iter().any(|h| h.enabled && !h.poll_mode);
        if needed {
            timer.modify_anticipate(self.clock.get_ns() + self.period_ticks);
        } else {
            timer.del();
        }
    }

    /// One round of the audio timer, `audio_run()`.
    pub fn run(&self) {
        self.run_out();
        self.run_in();
        self.with_state(|st| self.run_capture(st));
    }

    // Playback voices.

    /// Opens or reconfigures a playback voice, `audio_be_open_out()`.
    pub fn open_out(
        &self,
        sw: Option<SwVoiceOut>,
        name: &str,
        callback: AudioCallback,
        as_: &AudSettings,
    ) -> Option<SwVoiceOut> {
        self.with_state(|st| self.open_out_locked(st, sw, name, callback, as_))
    }

    fn open_out_locked(
        &self,
        st: &mut State,
        sw: Option<SwVoiceOut>,
        name: &str,
        callback: AudioCallback,
        as_: &AudSettings,
    ) -> Option<SwVoiceOut> {
        let mut sw = sw.filter(|s| st.sw_out.contains_key(&s.0));
        if !as_.is_valid() {
            error_report(&format!("audio: Invalid audio settings: {as_}"));
            self.close_out_locked(st, sw);
            return None;
        }
        if let Some(s) = sw {
            if st.sw_out[&s.0].info.eq_settings(as_) {
                return Some(s);
            }
        }
        if !self.pdo_out.fixed_settings && sw.is_some() {
            self.close_out_locked(st, sw);
            sw = None;
        }
        let id = match sw {
            Some(s) => {
                let hw = st.sw_out[&s.0].hw;
                if !self.sw_init_out(st, s.0, hw, name, as_) {
                    self.close_out_locked(st, Some(s));
                    return None;
                }
                s.0
            }
            None => self.create_voice_pair_out(st, name, as_)?,
        };
        let s = st.sw_out.get_mut(&id)?;
        s.vol = NOMINAL_VOLUME;
        s.callback = Some(callback);
        Some(SwVoiceOut(id))
    }

    /// `audio_pcm_sw_init_out()`. The voice is stored even when this fails, so the caller can
    /// close it.
    fn sw_init_out(
        &self,
        st: &mut State,
        id: u64,
        hw_id: u64,
        name: &str,
        as_: &AudSettings,
    ) -> bool {
        let info = PcmInfo::new(as_);
        let Some(hw) = find_out(&mut st.hw_out, hw_id) else { return false };
        let mut ok = true;
        let (resample_buf, rate) = if self.pdo_out.mixing_engine {
            let samples =
                muldiv64(hw.mix_buf.len() as u64, info.freq as u32, hw.core.info.freq as u32);
            if samples == 0 {
                ok = false;
                (Vec::new(), None)
            } else {
                (
                    vec![StSample::default(); samples as usize + 1],
                    Some(Rate::new(info.freq, hw.core.info.freq)),
                )
            }
        } else {
            (Vec::new(), None)
        };
        let old = st.sw_out.remove(&id);
        st.sw_out.insert(
            id,
            SwOut {
                hw: hw_id,
                info,
                conv: info.sample_fmt(),
                resample_buf,
                rb_pos: 0,
                rate,
                total_hw_samples_mixed: 0,
                active: false,
                empty: true,
                name: if ok { name.to_string() } else { String::new() },
                vol: old.as_ref().map_or(NOMINAL_VOLUME, |o| o.vol),
                callback: old.and_then(|o| o.callback),
            },
        );
        ok
    }

    /// `audio_pcm_create_voice_pair_out()`.
    fn create_voice_pair_out(&self, st: &mut State, name: &str, as_: &AudSettings) -> Option<u64> {
        let hw_as = if self.pdo_out.fixed_settings { self.pdo_out.settings() } else { *as_ };
        let Some(hw_id) = self.hw_add_out(st, &hw_as) else {
            error_report(&format!("audio: Could not create a backend for voice '{name}'"));
            return None;
        };
        let id = st.new_id();
        if let Some(hw) = find_out(&mut st.hw_out, hw_id) {
            hw.sw_head.insert(0, id);
        }
        if !self.sw_init_out(st, id, hw_id, name, as_) {
            st.sw_out.remove(&id);
            if let Some(hw) = find_out(&mut st.hw_out, hw_id) {
                hw.sw_head.retain(|&s| s != id);
            }
            self.hw_gc_out(st, hw_id);
            return None;
        }
        Some(id)
    }

    /// `audio_pcm_hw_add_out()`.
    fn hw_add_out(&self, st: &mut State, as_: &AudSettings) -> Option<u64> {
        let pdo = &self.pdo_out;
        if !pdo.mixing_engine || pdo.fixed_settings {
            let hw = self.hw_add_new_out(st, as_);
            if !pdo.mixing_engine || hw.is_some() {
                return hw;
            }
        }
        if let Some(hw) = st.hw_out.iter().find(|h| h.core.info.eq_settings(as_)) {
            return Some(hw.id);
        }
        if let Some(hw) = self.hw_add_new_out(st, as_) {
            return Some(hw);
        }
        st.hw_out.first().map(|h| h.id)
    }

    /// `audio_pcm_hw_add_new_out()`.
    fn hw_add_new_out(&self, st: &mut State, as_: &AudSettings) -> Option<u64> {
        if st.nb_hw_voices_out == 0 {
            return None;
        }
        let ctx = InitCtx { dev: &self.dev, pdo: &self.pdo_out, clock: &self.clock };
        let HwInit { mut pcm, info, samples, poll_mode } = self.driver.init_out(&ctx, as_)?;
        let core = HwCore::new(info, samples);
        if samples == 0 {
            audio_bug("audio_pcm_hw_add_new_out", "hw->samples=0");
            pcm.fini_out(&core);
            return None;
        }
        let mix_buf = if self.pdo_out.mixing_engine {
            vec![StSample::default(); samples]
        } else {
            Vec::new()
        };
        let id = st.new_id();
        st.hw_out.insert(
            0,
            HwOut {
                id,
                enabled: false,
                poll_mode,
                pending_disable: false,
                clip: info.sample_fmt(),
                core,
                mix_buf,
                mix_pos: 0,
                sw_head: Vec::new(),
                cap_head: Vec::new(),
                pcm,
            },
        );
        st.nb_hw_voices_out -= 1;
        self.attach_capture(st, id);
        Some(id)
    }

    /// `audio_pcm_hw_gc_out()`.
    fn hw_gc_out(&self, st: &mut State, hw_id: u64) {
        let Some(idx) = st.hw_out.iter().position(|h| h.id == hw_id) else { return };
        if !st.hw_out[idx].sw_head.is_empty() {
            return;
        }
        self.detach_capture(st, hw_id);
        let mut hw = st.hw_out.remove(idx);
        hw.pcm.fini_out(&hw.core);
        st.nb_hw_voices_out += 1;
    }

    /// Closes a playback voice, `audio_be_close_out()`.
    pub fn close_out(&self, sw: Option<SwVoiceOut>) {
        self.with_state(|st| self.close_out_locked(st, sw));
    }

    fn close_out_locked(&self, st: &mut State, sw: Option<SwVoiceOut>) {
        let Some(sw) = sw else { return };
        let Some(s) = st.sw_out.remove(&sw.0) else { return };
        if let Some(hw) = find_out(&mut st.hw_out, s.hw) {
            hw.sw_head.retain(|&id| id != sw.0);
        }
        self.hw_gc_out(st, s.hw);
    }

    /// Whether a playback voice is running, `audio_be_is_active_out()`.
    pub fn is_active_out(&self, sw: Option<SwVoiceOut>) -> bool {
        let Some(sw) = sw else { return false };
        self.with_state(|st| st.sw_out.get(&sw.0).is_some_and(|s| s.active))
    }

    /// Mixes PCM into a playback voice and returns the bytes taken, `audio_be_write()`.
    pub fn write(&self, sw: Option<SwVoiceOut>, buf: &[u8]) -> usize {
        let Some(sw) = sw else { return 0 };
        self.with_state(|st| {
            let State { hw_out, sw_out, .. } = st;
            let Some(s) = sw_out.get_mut(&sw.0) else { return 0 };
            let Some(hw) = find_out(hw_out, s.hw) else { return 0 };
            if !hw.enabled {
                warn_report(&format!("audio: Writing to disabled voice {}", sw_name(&s.name)));
                return 0;
            }
            if self.pdo_out.mixing_engine {
                self.sw_write(s, hw, buf)
            } else {
                hw.pcm.write(&hw.core.info, buf)
            }
        })
    }

    /// `audio_pcm_sw_write()`.
    fn sw_write(&self, sw: &mut SwOut, hw: &mut HwOut, buf: &[u8]) -> usize {
        let size = hw.mix_buf.len();
        let live = sw.total_hw_samples_mixed;
        if live > size {
            audio_bug("audio_pcm_sw_write", &format!("live={live} hw->mix_buf.size={size}"));
            return 0;
        }
        if live == size {
            return 0;
        }
        let dead = size - live;
        let hw_free = hw_get_free(hw);
        let hw_free = hw_free.saturating_sub(live);
        let frames_out_max = dead.min(hw_free);
        let Some(rate) = sw.rate.as_mut() else { return 0 };
        let sw_max = rate.frames_in(frames_out_max as u32) as usize;
        let bpf = sw.info.bytes_per_frame;
        let fe_max = (buf.len() / bpf + sw.rb_pos).min(sw.resample_buf.len());
        let frames_in_max = sw_max.min(fe_max);
        if frames_in_max == 0 {
            return 0;
        }
        if frames_in_max > sw.rb_pos {
            let dst = &mut sw.resample_buf[sw.rb_pos..frames_in_max];
            sw.conv.conv(dst, buf);
            if !self.driver.volume_out() {
                mixeng::volume(dst, &sw.vol);
            }
        }
        let (mut total_in, total_out) = resample_out(
            rate,
            &sw.resample_buf,
            frames_in_max,
            frames_out_max,
            &mut hw.mix_buf,
            hw.mix_pos,
            live,
        );
        sw.total_hw_samples_mixed += total_out;
        sw.empty = sw.total_hw_samples_mixed == 0;
        // Upsampling can leave one frame in the resample buffer for the next pass.
        if frames_in_max - total_in == 1 {
            sw.resample_buf[0] = sw.resample_buf[total_in];
            total_in = total_in + 1 - sw.rb_pos;
            sw.rb_pos = 1;
        } else if total_in >= sw.rb_pos {
            total_in -= sw.rb_pos;
            sw.rb_pos = 0;
        }
        total_in * bpf
    }

    /// The bytes a playback voice buffers, `audio_be_get_buffer_size_out()`.
    pub fn get_buffer_size_out(&self, sw: Option<SwVoiceOut>) -> usize {
        let Some(sw) = sw else { return 0 };
        self.with_state(|st| {
            let State { hw_out, sw_out, .. } = st;
            let Some(s) = sw_out.get(&sw.0) else { return 0 };
            if self.pdo_out.mixing_engine {
                return s.resample_buf.len() * s.info.bytes_per_frame;
            }
            find_out(hw_out, s.hw).map_or(0, |hw| hw.core.samples * hw.core.info.bytes_per_frame)
        })
    }

    /// Starts or stops a playback voice, `audio_be_set_active_out()`.
    pub fn set_active_out(&self, sw: Option<SwVoiceOut>, on: bool) {
        let Some(sw) = sw else { return };
        self.with_state(|st| {
            let Some(s) = st.sw_out.get(&sw.0) else { return };
            if s.active == on {
                return;
            }
            let hw_id = s.hw;
            let running = self.running.load(Ordering::SeqCst);
            let mut reset = false;
            let State { hw_out, sw_out, sw_cap, caps, deferred, .. } = st;
            let Some(hw) = find_out(hw_out, hw_id) else { return };
            if on {
                hw.pending_disable = false;
                if !hw.enabled {
                    hw.enabled = true;
                    if running {
                        hw.pcm.enable_out(true);
                        reset = true;
                    }
                }
            } else if hw.enabled {
                let nb_active =
                    hw.sw_head.iter().filter(|id| sw_out.get(id).is_some_and(|s| s.active)).count();
                hw.pending_disable = nb_active == 1;
            }
            let enabled = hw.enabled;
            for sc_id in &hw.cap_head {
                let Some(sc) = sw_cap.get_mut(sc_id) else { continue };
                sc.active = enabled;
                if enabled {
                    if let Some(cap) = find_cap(caps, sc.cap) {
                        capture_maybe_changed(cap, true, deferred);
                    }
                }
            }
            if let Some(s) = sw_out.get_mut(&sw.0) {
                s.active = on;
            }
            if reset {
                self.reset_timer(st);
            }
        });
    }

    /// Sets the volume of a playback voice, `audio_be_set_volume_out()`.
    pub fn set_volume_out(&self, sw: Option<SwVoiceOut>, vol: &Volume) {
        let Some(sw) = sw else { return };
        self.with_state(|st| {
            let State { hw_out, sw_out, .. } = st;
            let Some(s) = sw_out.get_mut(&sw.0) else { return };
            s.vol = mixeng_volume(vol);
            if self.driver.volume_out() {
                if let Some(hw) = find_out(hw_out, s.hw) {
                    hw.pcm.volume_out(vol);
                }
            }
        });
    }

    fn run_out(&self) {
        let hws: Vec<u64> = self.with_state(|st| st.hw_out.iter().map(|h| h.id).collect());
        for hw_id in hws {
            if !self.pdo_out.mixing_engine {
                // There is exactly one voice on each hard voice without the mixing engine.
                let step = self.with_state(|st| {
                    let State { hw_out, sw_out, .. } = st;
                    let hw = find_out(hw_out, hw_id).filter(|h| h.enabled)?;
                    let hw_free = hw_get_free(hw);
                    if hw.pending_disable {
                        hw.enabled = false;
                        hw.pending_disable = false;
                        hw.pcm.enable_out(false);
                    }
                    let sw = hw.sw_head.first().and_then(|id| sw_out.get(id));
                    Some(sw.filter(|s| s.active).and_then(|s| {
                        s.callback.clone().map(|cb| (cb, hw_free * s.info.bytes_per_frame))
                    }))
                });
                let Some(cb) = step else { continue };
                if let Some((cb, n)) = cb {
                    cb(n);
                }
                self.with_state(|st| {
                    if let Some(hw) = find_out(&mut st.hw_out, hw_id) {
                        hw.pcm.run_buffer_out(&mut hw.core);
                    }
                });
                continue;
            }

            let start = self.with_state(|st| {
                let hw = find_out(&mut st.hw_out, hw_id).filter(|h| h.enabled)?;
                Some((hw_get_free(hw), hw.sw_head.clone()))
            });
            let Some((hw_free, sws)) = start else { continue };
            for sw_id in sws {
                let cb = self.with_state(|st| {
                    let State { hw_out, sw_out, .. } = st;
                    let sw = sw_out.get(&sw_id).filter(|s| s.active && s.hw == hw_id)?;
                    let hw = find_out(hw_out, hw_id)?;
                    let size = hw.mix_buf.len();
                    let live = sw.total_hw_samples_mixed;
                    let sw_free = if live > size {
                        audio_bug(
                            "audio_get_free",
                            &format!("live={live} sw->hw->mix_buf.size={size}"),
                        );
                        0
                    } else {
                        size - live
                    };
                    let free = if hw_free > live {
                        let rate = sw.rate.as_ref()?;
                        rate.frames_in(sw_free.min(hw_free - live) as u32) as usize
                    } else {
                        0
                    };
                    if free > sw.rb_pos {
                        let free = free.min(sw.resample_buf.len()) - sw.rb_pos;
                        return sw.callback.clone().map(|cb| (cb, free * sw.info.bytes_per_frame));
                    }
                    None
                });
                if let Some((cb, n)) = cb {
                    cb(n);
                }
            }
            self.with_state(|st| self.run_out_finish(st, hw_id));
        }
    }

    /// The rest of `audio_run_out()` for one hard voice, once the voices have been fed.
    fn run_out_finish(&self, st: &mut State, hw_id: u64) {
        let State { hw_out, sw_out, sw_cap, caps, deferred, .. } = st;
        let Some(hw) = find_out(hw_out, hw_id) else { return };
        let size = hw.mix_buf.len();
        let (live, nb_live) = get_live_out(
            "audio_pcm_hw_get_live_out",
            size,
            hw.sw_head
                .iter()
                .filter_map(|id| sw_out.get(id))
                .map(|s| (s.active, s.empty, s.total_hw_samples_mixed)),
        );
        if live > size {
            audio_bug("audio_run_out", &format!("live={live} hw->mix_buf.size={size}"));
            return;
        }
        if hw.pending_disable && nb_live == 0 {
            hw.enabled = false;
            hw.pending_disable = false;
            hw.pcm.enable_out(false);
            for sc_id in hw.cap_head.clone() {
                let Some(sc) = sw_cap.get_mut(&sc_id) else { continue };
                sc.active = false;
                let cap = sc.cap;
                recalc_and_notify_capture(caps, sw_cap, cap, deferred);
            }
            return;
        }
        if live == 0 {
            hw.pcm.run_buffer_out(&mut hw.core);
            return;
        }
        let prev_rpos = hw.mix_pos;
        let mut played = hw_run_out(hw, live);
        if hw.mix_pos >= size {
            audio_bug(
                "audio_run_out",
                &format!("hw->mix_buf.pos={} hw->mix_buf.size={size} played={played}", hw.mix_pos),
            );
            hw.mix_pos = 0;
        }
        if played > 0 {
            capture_mix_and_clear(hw, sw_cap, caps, prev_rpos, played);
        }
        for id in &hw.sw_head {
            let Some(sw) = sw_out.get_mut(id) else { continue };
            if !sw.active && sw.empty {
                continue;
            }
            if played > sw.total_hw_samples_mixed {
                audio_bug(
                    "audio_run_out",
                    &format!(
                        "played={played} sw->total_hw_samples_mixed={}",
                        sw.total_hw_samples_mixed
                    ),
                );
                played = sw.total_hw_samples_mixed;
            }
            sw.total_hw_samples_mixed -= played;
            if sw.total_hw_samples_mixed == 0 {
                sw.empty = true;
            }
        }
    }

    // Capture voices.

    /// Opens or reconfigures a capture voice, `audio_be_open_in()`.
    pub fn open_in(
        &self,
        sw: Option<SwVoiceIn>,
        name: &str,
        callback: AudioCallback,
        as_: &AudSettings,
    ) -> Option<SwVoiceIn> {
        self.with_state(|st| self.open_in_locked(st, sw, name, callback, as_))
    }

    fn open_in_locked(
        &self,
        st: &mut State,
        sw: Option<SwVoiceIn>,
        name: &str,
        callback: AudioCallback,
        as_: &AudSettings,
    ) -> Option<SwVoiceIn> {
        let mut sw = sw.filter(|s| st.sw_in.contains_key(&s.0));
        if !as_.is_valid() {
            error_report(&format!("audio: Invalid audio settings: {as_}"));
            self.close_in_locked(st, sw);
            return None;
        }
        if !self.driver.has_init_in() {
            error_report(&format!("audio: Can not open `{name}' (no host audio driver)"));
            self.close_in_locked(st, sw);
            return None;
        }
        if let Some(s) = sw {
            if st.sw_in[&s.0].info.eq_settings(as_) {
                return Some(s);
            }
        }
        if !self.pdo_in.fixed_settings && sw.is_some() {
            self.close_in_locked(st, sw);
            sw = None;
        }
        let id = match sw {
            Some(s) => {
                let hw = st.sw_in[&s.0].hw;
                if !self.sw_init_in(st, s.0, hw, name, as_) {
                    self.close_in_locked(st, Some(s));
                    return None;
                }
                s.0
            }
            None => self.create_voice_pair_in(st, name, as_)?,
        };
        let s = st.sw_in.get_mut(&id)?;
        s.vol = NOMINAL_VOLUME;
        s.callback = Some(callback);
        Some(SwVoiceIn(id))
    }

    /// `audio_pcm_sw_init_in()`.
    fn sw_init_in(
        &self,
        st: &mut State,
        id: u64,
        hw_id: u64,
        name: &str,
        as_: &AudSettings,
    ) -> bool {
        let info = PcmInfo::new(as_);
        let Some(hw) = find_in(&mut st.hw_in, hw_id) else { return false };
        let mut ok = true;
        let (resample_buf, rate) = if self.pdo_in.mixing_engine {
            let samples =
                muldiv64(hw.conv_buf.len() as u64, info.freq as u32, hw.core.info.freq as u32);
            if samples == 0 {
                ok = false;
                (Vec::new(), None)
            } else {
                (
                    vec![StSample::default(); samples as usize + 1],
                    Some(Rate::new(hw.core.info.freq, info.freq)),
                )
            }
        } else {
            (Vec::new(), None)
        };
        let old = st.sw_in.remove(&id);
        st.sw_in.insert(
            id,
            SwIn {
                hw: hw_id,
                info,
                clip: info.sample_fmt(),
                resample_buf,
                rate,
                total_hw_samples_acquired: old.as_ref().map_or(0, |o| o.total_hw_samples_acquired),
                active: false,
                name: if ok { name.to_string() } else { String::new() },
                vol: old.as_ref().map_or(NOMINAL_VOLUME, |o| o.vol),
                callback: old.and_then(|o| o.callback),
            },
        );
        ok
    }

    /// `audio_pcm_create_voice_pair_in()`.
    fn create_voice_pair_in(&self, st: &mut State, name: &str, as_: &AudSettings) -> Option<u64> {
        let hw_as = if self.pdo_in.fixed_settings { self.pdo_in.settings() } else { *as_ };
        let Some(hw_id) = self.hw_add_in(st, &hw_as) else {
            error_report(&format!("audio: Could not create a backend for voice '{name}'"));
            return None;
        };
        let id = st.new_id();
        if let Some(hw) = find_in(&mut st.hw_in, hw_id) {
            hw.sw_head.insert(0, id);
        }
        if !self.sw_init_in(st, id, hw_id, name, as_) {
            st.sw_in.remove(&id);
            if let Some(hw) = find_in(&mut st.hw_in, hw_id) {
                hw.sw_head.retain(|&s| s != id);
            }
            self.hw_gc_in(st, hw_id);
            return None;
        }
        Some(id)
    }

    /// `audio_pcm_hw_add_in()`.
    fn hw_add_in(&self, st: &mut State, as_: &AudSettings) -> Option<u64> {
        let pdo = &self.pdo_in;
        if !pdo.mixing_engine || pdo.fixed_settings {
            let hw = self.hw_add_new_in(st, as_);
            if !pdo.mixing_engine || hw.is_some() {
                return hw;
            }
        }
        if let Some(hw) = st.hw_in.iter().find(|h| h.core.info.eq_settings(as_)) {
            return Some(hw.id);
        }
        if let Some(hw) = self.hw_add_new_in(st, as_) {
            return Some(hw);
        }
        st.hw_in.first().map(|h| h.id)
    }

    /// `audio_pcm_hw_add_new_in()`.
    fn hw_add_new_in(&self, st: &mut State, as_: &AudSettings) -> Option<u64> {
        if st.nb_hw_voices_in == 0 {
            return None;
        }
        if !self.driver.has_init_in() {
            audio_bug("audio_pcm_hw_add_new_in", "No host audio driver or missing init_capture");
            return None;
        }
        let ctx = InitCtx { dev: &self.dev, pdo: &self.pdo_in, clock: &self.clock };
        let HwInit { mut pcm, info, samples, poll_mode } = self.driver.init_in(&ctx, as_)?;
        let core = HwCore::new(info, samples);
        if samples == 0 {
            audio_bug("audio_pcm_hw_add_new_in", "hw->samples=0");
            pcm.fini_in(&core);
            return None;
        }
        let conv_buf =
            if self.pdo_in.mixing_engine { vec![StSample::default(); samples] } else { Vec::new() };
        let id = st.new_id();
        st.hw_in.insert(
            0,
            HwIn {
                id,
                enabled: false,
                poll_mode,
                conv: info.sample_fmt(),
                core,
                conv_buf,
                conv_pos: 0,
                total_samples_captured: 0,
                sw_head: Vec::new(),
                pcm,
            },
        );
        st.nb_hw_voices_in -= 1;
        Some(id)
    }

    /// `audio_pcm_hw_gc_in()`.
    fn hw_gc_in(&self, st: &mut State, hw_id: u64) {
        let Some(idx) = st.hw_in.iter().position(|h| h.id == hw_id) else { return };
        if !st.hw_in[idx].sw_head.is_empty() {
            return;
        }
        let mut hw = st.hw_in.remove(idx);
        hw.pcm.fini_in(&hw.core);
        st.nb_hw_voices_in += 1;
    }

    /// Closes a capture voice, `audio_be_close_in()`.
    pub fn close_in(&self, sw: Option<SwVoiceIn>) {
        self.with_state(|st| self.close_in_locked(st, sw));
    }

    fn close_in_locked(&self, st: &mut State, sw: Option<SwVoiceIn>) {
        let Some(sw) = sw else { return };
        let Some(s) = st.sw_in.remove(&sw.0) else { return };
        if let Some(hw) = find_in(&mut st.hw_in, s.hw) {
            hw.sw_head.retain(|&id| id != sw.0);
        }
        self.hw_gc_in(st, s.hw);
    }

    /// Whether a capture voice is running, `audio_be_is_active_in()`.
    pub fn is_active_in(&self, sw: Option<SwVoiceIn>) -> bool {
        let Some(sw) = sw else { return false };
        self.with_state(|st| st.sw_in.get(&sw.0).is_some_and(|s| s.active))
    }

    /// Reads captured PCM from a capture voice, `audio_be_read()`.
    pub fn read(&self, sw: Option<SwVoiceIn>, buf: &mut [u8]) -> usize {
        let Some(sw) = sw else { return 0 };
        self.with_state(|st| {
            let State { hw_in, sw_in, .. } = st;
            let Some(s) = sw_in.get_mut(&sw.0) else { return 0 };
            let Some(hw) = find_in(hw_in, s.hw) else { return 0 };
            if !hw.enabled {
                warn_report(&format!("audio: Reading from disabled voice {}", sw_name(&s.name)));
                return 0;
            }
            if self.pdo_in.mixing_engine {
                self.sw_read(s, hw, buf)
            } else {
                hw.pcm.read(&hw.core.info, buf)
            }
        })
    }

    /// `audio_pcm_sw_read()`.
    fn sw_read(&self, sw: &mut SwIn, hw: &mut HwIn, buf: &mut [u8]) -> usize {
        let size = hw.conv_buf.len();
        let live = hw.total_samples_captured.wrapping_sub(sw.total_hw_samples_acquired);
        if live == 0 {
            return 0;
        }
        if live > size {
            audio_bug("audio_pcm_sw_read", &format!("live={live} hw->conv_buf.size={size}"));
            return 0;
        }
        let bpf = sw.info.bytes_per_frame;
        let frames_out_max = (buf.len() / bpf).min(sw.resample_buf.len());
        let Some(rate) = sw.rate.as_mut() else { return 0 };
        // audio_pcm_sw_resample_in()
        let mut rpos = ring_posb(hw.conv_pos, live, size);
        let in_lim = live.min(size - rpos);
        let (fi, fo) = rate.flow(
            &hw.conv_buf[rpos..rpos + in_lim],
            &mut sw.resample_buf[..frames_out_max],
            false,
        );
        rpos += fi;
        let (mut total_in, mut total_out) = (fi, fo);
        if live - fi != 0 && rpos == size {
            let (fi2, fo2) = rate.flow(
                &hw.conv_buf[..live - fi],
                &mut sw.resample_buf[fo..frames_out_max],
                false,
            );
            total_in += fi2;
            total_out += fo2;
        }
        if !self.driver.volume_in() {
            mixeng::volume(&mut sw.resample_buf[..total_out], &sw.vol);
        }
        sw.clip.clip(buf, &sw.resample_buf[..total_out]);
        sw.total_hw_samples_acquired = sw.total_hw_samples_acquired.wrapping_add(total_in);
        total_out * bpf
    }

    /// Starts or stops a capture voice, `audio_be_set_active_in()`.
    pub fn set_active_in(&self, sw: Option<SwVoiceIn>, on: bool) {
        let Some(sw) = sw else { return };
        self.with_state(|st| {
            let Some(s) = st.sw_in.get(&sw.0) else { return };
            if s.active == on {
                return;
            }
            let hw_id = s.hw;
            let running = self.running.load(Ordering::SeqCst);
            let mut reset = false;
            let State { hw_in, sw_in, .. } = st;
            let Some(hw) = find_in(hw_in, hw_id) else { return };
            if on {
                if !hw.enabled {
                    hw.enabled = true;
                    if running {
                        hw.pcm.enable_in(true);
                        reset = true;
                    }
                }
                if let Some(s) = sw_in.get_mut(&sw.0) {
                    s.total_hw_samples_acquired = hw.total_samples_captured;
                }
            } else if hw.enabled {
                let nb_active =
                    hw.sw_head.iter().filter(|id| sw_in.get(id).is_some_and(|s| s.active)).count();
                if nb_active == 1 {
                    hw.enabled = false;
                    hw.pcm.enable_in(false);
                }
            }
            if let Some(s) = sw_in.get_mut(&sw.0) {
                s.active = on;
            }
            if reset {
                self.reset_timer(st);
            }
        });
    }

    /// Sets the volume of a capture voice, `audio_be_set_volume_in()`.
    pub fn set_volume_in(&self, sw: Option<SwVoiceIn>, vol: &Volume) {
        let Some(sw) = sw else { return };
        self.with_state(|st| {
            let State { hw_in, sw_in, .. } = st;
            let Some(s) = sw_in.get_mut(&sw.0) else { return };
            s.vol = mixeng_volume(vol);
            if self.driver.volume_in() {
                if let Some(hw) = find_in(hw_in, s.hw) {
                    hw.pcm.volume_in(vol);
                }
            }
        });
    }

    fn run_in(&self) {
        let hws: Vec<u64> = self.with_state(|st| st.hw_in.iter().map(|h| h.id).collect());
        if !self.pdo_in.mixing_engine {
            for hw_id in hws {
                let cb = self.with_state(|st| {
                    let State { hw_in, sw_in, .. } = st;
                    let hw = find_in(hw_in, hw_id).filter(|h| h.enabled)?;
                    let sw = sw_in.get(hw.sw_head.first()?)?;
                    if sw.active { sw.callback.clone() } else { None }
                });
                if let Some(cb) = cb {
                    cb(i32::MAX as usize);
                }
            }
            return;
        }
        for hw_id in hws {
            let sws = self.with_state(|st| {
                let State { hw_in, sw_in, .. } = st;
                let hw = find_in(hw_in, hw_id).filter(|h| h.enabled)?;
                let size = hw.conv_buf.len();
                let min_before = find_min_in(hw, sw_in);
                let mut live = hw.total_samples_captured.wrapping_sub(min_before);
                if live > size {
                    audio_bug(
                        "audio_pcm_hw_get_live_in",
                        &format!("live={live} hw->conv_buf.size={size}"),
                    );
                    live = 0;
                }
                let captured = hw_run_in(hw, size - live);
                assert!(captured <= size);
                let min = find_min_in(hw, sw_in);
                hw.total_samples_captured =
                    hw.total_samples_captured.wrapping_add(captured.wrapping_sub(min));
                Some((hw.sw_head.clone(), min))
            });
            let Some((sws, min)) = sws else { continue };
            for sw_id in sws {
                let cb = self.with_state(|st| {
                    let State { hw_in, sw_in, .. } = st;
                    let hw = find_in(hw_in, hw_id)?;
                    let sw = sw_in.get_mut(&sw_id)?;
                    sw.total_hw_samples_acquired = sw.total_hw_samples_acquired.wrapping_sub(min);
                    if !sw.active {
                        return None;
                    }
                    // audio_get_avail()
                    let size = hw.conv_buf.len();
                    let mut sw_avail =
                        hw.total_samples_captured.wrapping_sub(sw.total_hw_samples_acquired);
                    if sw_avail > size {
                        audio_bug(
                            "audio_get_avail",
                            &format!("live={sw_avail} sw->hw->conv_buf.size={size}"),
                        );
                        sw_avail = 0;
                    }
                    let avail = sw.rate.as_ref()?.frames_out(sw_avail as u32) as usize;
                    if avail > 0 {
                        let avail = avail.min(sw.resample_buf.len());
                        return sw.callback.clone().map(|cb| (cb, avail * sw.info.bytes_per_frame));
                    }
                    None
                });
                if let Some((cb, n)) = cb {
                    cb(n);
                }
            }
        }
    }

    // Captures.

    /// Taps the playback mix in the format `as_`, `audio_be_add_capture()`.
    pub fn add_capture(
        &self,
        as_: &AudSettings,
        ops: Arc<dyn CaptureOps>,
    ) -> Option<CaptureHandle> {
        if !self.pdo_out.mixing_engine {
            error_report("audio: Can't capture with mixeng disabled");
            return None;
        }
        if !as_.is_valid() {
            error_report(&format!(
                "audio: Invalid audio settings when trying to add capture: {as_}"
            ));
            return None;
        }
        self.with_state(|st| {
            let cb = st.new_id();
            if let Some(cap) = st.caps.iter_mut().find(|c| c.info.eq_settings(as_)) {
                cap.cbs.insert(0, (cb, ops));
                return Some(CaptureHandle { cap: cap.id, cb });
            }
            let id = st.new_id();
            let samples = 4096 * 4;
            let info = PcmInfo::new(as_);
            st.caps.insert(
                0,
                Capture {
                    id,
                    enabled: false,
                    info,
                    clip: info.sample_fmt(),
                    mix_buf: vec![StSample::default(); samples],
                    mix_pos: 0,
                    buf: vec![0; samples * info.bytes_per_frame],
                    sw_head: Vec::new(),
                    cbs: vec![(cb, ops)],
                },
            );
            let hws: Vec<u64> = st.hw_out.iter().map(|h| h.id).collect();
            for hw in hws {
                self.attach_capture(st, hw);
            }
            Some(CaptureHandle { cap: id, cb })
        })
    }

    /// Removes a capture callback set, `audio_be_del_capture()`. The capture goes away with
    /// its last callback set.
    pub fn del_capture(&self, h: CaptureHandle) {
        self.with_state(|st| {
            let State { hw_out, sw_cap, caps, deferred, .. } = st;
            let Some(idx) = caps.iter().position(|c| c.id == h.cap) else { return };
            let cap = &mut caps[idx];
            let Some(ci) = cap.cbs.iter().position(|(id, _)| *id == h.cb) else { return };
            let (_, ops) = cap.cbs.remove(ci);
            deferred.push(Deferred::Destroy(ops));
            if !cap.cbs.is_empty() {
                return;
            }
            for sc_id in std::mem::take(&mut cap.sw_head) {
                if let Some(sc) = sw_cap.remove(&sc_id) {
                    if let Some(hw) = find_out(hw_out, sc.hw) {
                        hw.cap_head.retain(|&id| id != sc_id);
                    }
                }
            }
            caps.remove(idx);
        });
    }

    /// `audio_attach_capture()`.
    fn attach_capture(&self, st: &mut State, hw_id: u64) {
        self.detach_capture(st, hw_id);
        let cap_ids: Vec<u64> = st.caps.iter().map(|c| c.id).collect();
        for cap_id in cap_ids {
            let sc_id = st.new_id();
            let State { hw_out, sw_cap, caps, deferred, .. } = st;
            let Some(hw) = find_out(hw_out, hw_id) else { return };
            let Some(cap) = find_cap(caps, cap_id) else { continue };
            let active = hw.enabled;
            sw_cap.insert(
                sc_id,
                SwCap {
                    hw: hw_id,
                    cap: cap_id,
                    rate: Rate::new(hw.core.info.freq, cap.info.freq),
                    total_hw_samples_mixed: 0,
                    active,
                    empty: true,
                },
            );
            cap.sw_head.insert(0, sc_id);
            hw.cap_head.insert(0, sc_id);
            if active {
                capture_maybe_changed(cap, true, deferred);
            }
        }
    }

    /// `audio_detach_capture()`.
    fn detach_capture(&self, st: &mut State, hw_id: u64) {
        let State { hw_out, sw_cap, caps, deferred, .. } = st;
        let Some(hw) = find_out(hw_out, hw_id) else { return };
        for sc_id in std::mem::take(&mut hw.cap_head) {
            let Some(sc) = sw_cap.remove(&sc_id) else { continue };
            if let Some(cap) = find_cap(caps, sc.cap) {
                cap.sw_head.retain(|&id| id != sc_id);
            }
            if sc.active {
                recalc_and_notify_capture(caps, sw_cap, sc.cap, deferred);
            }
        }
    }

    /// `audio_run_capture()`.
    fn run_capture(&self, st: &mut State) {
        let State { sw_cap, caps, deferred, .. } = st;
        for cap in caps.iter_mut() {
            let size = cap.mix_buf.len();
            let (live, _) = get_live_out(
                "audio_pcm_hw_get_live_out",
                size,
                cap.sw_head
                    .iter()
                    .filter_map(|id| sw_cap.get(id))
                    .map(|s| (s.active, s.empty, s.total_hw_samples_mixed)),
            );
            let mut captured = live;
            let mut live = live;
            let mut rpos = cap.mix_pos;
            let bpf = cap.info.bytes_per_frame;
            while live > 0 {
                let to_capture = live.min(size - rpos);
                cap.clip.clip(&mut cap.buf, &cap.mix_buf[rpos..rpos + to_capture]);
                mixeng::clear(&mut cap.mix_buf[rpos..rpos + to_capture]);
                for (_, ops) in &cap.cbs {
                    deferred
                        .push(Deferred::Capture(ops.clone(), cap.buf[..to_capture * bpf].to_vec()));
                }
                rpos = (rpos + to_capture) % size;
                live -= to_capture;
            }
            cap.mix_pos = rpos;
            for id in &cap.sw_head {
                let Some(sw) = sw_cap.get_mut(id) else { continue };
                if !sw.active && sw.empty {
                    continue;
                }
                if captured > sw.total_hw_samples_mixed {
                    audio_bug(
                        "audio_run_capture",
                        &format!(
                            "captured={captured} sw->total_hw_samples_mixed={}",
                            sw.total_hw_samples_mixed
                        ),
                    );
                    captured = sw.total_hw_samples_mixed;
                }
                sw.total_hw_samples_mixed -= captured;
                sw.empty = sw.total_hw_samples_mixed == 0;
            }
        }
    }

    /// Closes every host stream, `audio_mixeng_backend_finalize()`. Voices that devices still
    /// hold become inert.
    pub fn shutdown(&self) {
        if let Some(t) = self.timer.get() {
            t.del();
        }
        self.with_state(|st| {
            for mut hw in std::mem::take(&mut st.hw_out) {
                if hw.enabled {
                    hw.pcm.enable_out(false);
                }
                hw.pcm.fini_out(&hw.core);
                for sc_id in &hw.cap_head {
                    let Some(sc) = st.sw_cap.get(sc_id) else { continue };
                    if let Some(cap) = st.caps.iter().find(|c| c.id == sc.cap) {
                        for (_, ops) in &cap.cbs {
                            st.deferred.push(Deferred::Destroy(ops.clone()));
                        }
                    }
                }
            }
            for mut hw in std::mem::take(&mut st.hw_in) {
                if hw.enabled {
                    hw.pcm.enable_in(false);
                }
                hw.pcm.fini_in(&hw.core);
            }
            st.sw_out.clear();
            st.sw_in.clear();
            st.sw_cap.clear();
            st.caps.clear();
        });
    }
}

/// `audio_init_nb_voices_out()` and `audio_init_nb_voices_in()`.
fn init_nb_voices(be_name: &str, out: bool, voices: i32, max_voices: i32, min_voices: i32) -> i32 {
    let what = if out { "playback" } else { "capture" };
    let mut nb = voices;
    if nb > max_voices {
        if max_voices == 0 {
            if out {
                warn_report(&format!("audio: '{be_name}' backend does not support {what}"));
            }
        } else {
            warn_report(&format!(
                "audio: '{be_name}' backend does not support {nb} {what} voices, max {max_voices}"
            ));
        }
        nb = max_voices;
    }
    if nb < min_voices {
        warn_report(&format!("audio: Bogus number of {what} voices {nb}, setting to {min_voices}"));
    }
    nb
}

fn mixeng_volume(vol: &Volume) -> MixengVolume {
    let r = vol.vol[if vol.channels > 1 { 1 } else { 0 }];
    MixengVolume {
        mute: vol.mute,
        l: NOMINAL_VOLUME.l * i64::from(vol.vol[0]) / 255,
        r: NOMINAL_VOLUME.r * i64::from(r) / 255,
    }
}

/// `audio_pcm_hw_find_min_in()`.
fn find_min_in(hw: &HwIn, sw_in: &BTreeMap<u64, SwIn>) -> usize {
    let mut m = hw.total_samples_captured;
    for id in &hw.sw_head {
        if let Some(sw) = sw_in.get(id) {
            if sw.active {
                m = m.min(sw.total_hw_samples_acquired);
            }
        }
    }
    m
}

/// `audio_pcm_hw_run_out()`: clips up to `live` frames out to the driver.
fn hw_run_out(hw: &mut HwOut, live: usize) -> usize {
    let bpf = hw.core.info.bytes_per_frame;
    let mut clipped = 0;
    let mut live = live;
    while live > 0 {
        let HwOut { pcm, core, mix_buf, mix_pos, clip, .. } = hw;
        let (buf, size) = pcm.get_buffer_out(core, live * bpf);
        if size == 0 {
            break;
        }
        let decr = (size / bpf).min(live);
        if let Some(buf) = buf {
            clip_out(clip, mix_buf, *mix_pos, bpf, buf, decr);
        }
        let proc = pcm.put_buffer_out(core, decr * bpf) / bpf;
        live -= proc;
        clipped += proc;
        hw.mix_pos = (hw.mix_pos + proc) % hw.mix_buf.len();
        if proc == 0 || proc < decr {
            break;
        }
    }
    hw.pcm.run_buffer_out(&mut hw.core);
    clipped
}

/// `audio_pcm_hw_run_in()`: converts up to `samples` captured frames into the ring.
fn hw_run_in(hw: &mut HwIn, samples: usize) -> usize {
    hw.pcm.run_buffer_in(&mut hw.core);
    let bpf = hw.core.info.bytes_per_frame;
    let mut conv = 0;
    let mut samples = samples;
    while samples > 0 {
        let HwIn { pcm, core, conv_buf, conv_pos, conv: fmt, .. } = hw;
        let buf = pcm.get_buffer_in(core, samples * bpf);
        let size = buf.len();
        assert!(size % bpf == 0);
        if size == 0 {
            break;
        }
        // audio_pcm_hw_conv_in()
        let mut frames = size / bpf;
        let mut proc = 0;
        let cs = conv_buf.len();
        while frames > 0 {
            let n = frames.min(cs - *conv_pos);
            fmt.conv(&mut conv_buf[*conv_pos..*conv_pos + n], &buf[proc * bpf..]);
            *conv_pos = (*conv_pos + n) % cs;
            frames -= n;
            proc += n;
        }
        pcm.put_buffer_in(core, proc * bpf);
        samples -= proc;
        conv += proc;
    }
    conv
}

/// `audio_capture_mix_and_clear()`: feeds the frames just played to every capture, then
/// clears them from the mix.
fn capture_mix_and_clear(
    hw: &mut HwOut,
    sw_cap: &mut BTreeMap<u64, SwCap>,
    caps: &mut [Capture],
    rpos: usize,
    samples: usize,
) {
    let size = hw.mix_buf.len();
    if hw.enabled {
        for sc_id in &hw.cap_head {
            let Some(sc) = sw_cap.get_mut(sc_id) else { continue };
            let Some(cap) = find_cap(caps, sc.cap) else { continue };
            let mut rpos2 = rpos;
            let mut n = samples;
            while n > 0 {
                let to_read = (size - rpos2).min(n);
                let live = sc.total_hw_samples_mixed;
                let cap_size = cap.mix_buf.len();
                let (frames_in, frames_out) = resample_out(
                    &mut sc.rate,
                    &hw.mix_buf[rpos2..rpos2 + to_read],
                    to_read,
                    cap_size - live,
                    &mut cap.mix_buf,
                    cap.mix_pos,
                    live,
                );
                sc.total_hw_samples_mixed += frames_out;
                sc.empty = sc.total_hw_samples_mixed == 0;
                if to_read - frames_in != 0 {
                    audio_bug(
                        "audio_capture_mix_and_clear",
                        &format!(
                            "Could not mix {to_read} frames into a capture buffer, mixed {frames_in}"
                        ),
                    );
                    break;
                }
                n -= to_read;
                rpos2 = (rpos2 + to_read) % size;
            }
        }
    }
    let n = samples.min(size - rpos);
    mixeng::clear(&mut hw.mix_buf[rpos..rpos + n]);
    mixeng::clear(&mut hw.mix_buf[..samples - n]);
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use ruvm_base::ClockType;
    use ruvm_qapi::types::{AudioFormat, AudiodevU, AudiodevWavOptions};

    use super::*;
    use crate::none::NoneDriver;
    use crate::wav::WavDriver;

    fn backend(u: AudiodevU, driver: Box<dyn Driver>) -> (Arc<AudioBackend>, Arc<Clock>) {
        let mut dev = Audiodev { id: "t".into(), timer_period: None, u };
        crate::registry::validate_opts(&mut dev).unwrap();
        let pdo = Pdo::from_qapi(&Default::default());
        let be = AudioBackend::new(dev, pdo.clone(), pdo, driver, true);
        let clock = Clock::manual(ClockType::Virtual);
        be.attach_clock(&clock);
        (be, clock)
    }

    fn run_to(clock: &Clock, ms: i64) {
        for t in 1..=ms / 10 {
            clock.set_ns(t * 10_000_000);
            clock.run_timers();
        }
    }

    fn sample(frame: usize, ch: usize) -> i16 {
        (frame as i32 * 7 + ch as i32 * 1000) as i16
    }

    /// A voice that plays the ramp of `sample()` for as long as the backend asks.
    fn ramp_voice(be: &Arc<AudioBackend>, as_: &AudSettings) -> SwVoiceOut {
        let slot: Arc<OnceLock<SwVoiceOut>> = Arc::new(OnceLock::new());
        let frames = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(be);
        let nch = as_.nchannels as usize;
        let (s, f) = (slot.clone(), frames.clone());
        let cb: AudioCallback = Arc::new(move |avail| {
            let Some(be) = weak.upgrade() else { return };
            let start = f.load(Ordering::SeqCst);
            let n = avail / (2 * nch);
            let mut buf = Vec::with_capacity(n * 2 * nch);
            for k in start..start + n {
                for c in 0..nch {
                    buf.extend_from_slice(&sample(k, c).to_le_bytes());
                }
            }
            let took = be.write(s.get().copied(), &buf);
            f.fetch_add(took / (2 * nch), Ordering::SeqCst);
        });
        let sw = be.open_out(None, "ramp", cb, as_).unwrap();
        slot.set(sw).unwrap();
        be.set_active_out(Some(sw), true);
        sw
    }

    #[test]
    fn wav_records_the_mix_at_clock_rate() {
        let path = std::env::temp_dir().join(format!("ruvm-audio-{}.wav", std::process::id()));
        let u = AudiodevU::Wav(AudiodevWavOptions {
            path: Some(path.to_str().unwrap().into()),
            ..Default::default()
        });
        let (be, clock) = backend(u, Box::new(WavDriver));
        let as_ =
            AudSettings { freq: 44100, nchannels: 2, fmt: AudioFormat::S16, big_endian: false };
        let sw = ramp_voice(&be, &as_);
        run_to(&clock, 100);
        be.close_out(Some(sw));
        be.shutdown();
        let data = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&data[..4], b"RIFF");
        assert_eq!(&data[36..40], b"data");
        let datalen = u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize;
        let rifflen = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        assert_eq!(datalen, 17640);
        assert_eq!(rifflen, datalen + 36);
        assert_eq!(data.len(), 44 + datalen);
        for (k, frame) in data[44..].chunks_exact(4).enumerate() {
            assert_eq!(i16::from_le_bytes([frame[0], frame[1]]), sample(k, 0), "frame {k}");
            assert_eq!(i16::from_le_bytes([frame[2], frame[3]]), sample(k, 1), "frame {k}");
        }
    }

    struct Tap {
        bytes: AtomicUsize,
        enabled: AtomicBool,
        destroyed: AtomicBool,
    }

    impl CaptureOps for Tap {
        fn notify(&self, cmd: CaptureNotify) {
            self.enabled.store(cmd == CaptureNotify::Enable, Ordering::SeqCst);
        }
        fn capture(&self, buf: &[u8]) {
            self.bytes.fetch_add(buf.len(), Ordering::SeqCst);
        }
        fn destroy(&self) {
            self.destroyed.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn resampled_voice_feeds_a_capture() {
        let (be, clock) = backend(
            AudiodevU::for_tag(ruvm_qapi::types::AudiodevDriver::None),
            Box::new(NoneDriver),
        );
        let tap = Arc::new(Tap {
            bytes: AtomicUsize::new(0),
            enabled: AtomicBool::new(false),
            destroyed: AtomicBool::new(false),
        });
        let cap_as =
            AudSettings { freq: 44100, nchannels: 2, fmt: AudioFormat::S16, big_endian: false };
        let h = be.add_capture(&cap_as, tap.clone()).unwrap();
        let as_ =
            AudSettings { freq: 22050, nchannels: 1, fmt: AudioFormat::S16, big_endian: false };
        let sw = ramp_voice(&be, &as_);
        assert!(be.is_active_out(Some(sw)));
        assert!(tap.enabled.load(Ordering::SeqCst));
        run_to(&clock, 200);
        let got = tap.bytes.load(Ordering::SeqCst);
        // The capture sees what goes into the host buffer, so it runs up to one buffer of 1024
        // frames ahead of the 200ms of 44.1kHz stereo the clock allows.
        assert!(got % 4 == 0 && got > 30000 && got <= 35280 + 4096, "captured {got}");
        be.set_active_out(Some(sw), false);
        run_to(&clock, 300);
        assert!(!tap.enabled.load(Ordering::SeqCst));
        be.del_capture(h);
        assert!(tap.destroyed.load(Ordering::SeqCst));
        be.close_out(Some(sw));
        assert!(!be.is_active_out(Some(sw)));
        assert_eq!(be.write(Some(sw), &[0; 4]), 0);
    }

    #[test]
    fn none_records_silence() {
        let (be, clock) = backend(
            AudiodevU::for_tag(ruvm_qapi::types::AudiodevDriver::None),
            Box::new(NoneDriver),
        );
        let as_ = AudSettings { freq: 8000, nchannels: 1, fmt: AudioFormat::U8, big_endian: false };
        let ready = Arc::new(AtomicUsize::new(0));
        let r = ready.clone();
        let sw = be
            .open_in(
                None,
                "mic",
                Arc::new(move |n| {
                    r.fetch_add(n, Ordering::SeqCst);
                }),
                &as_,
            )
            .unwrap();
        be.set_active_in(Some(sw), true);
        run_to(&clock, 50);
        assert!(ready.load(Ordering::SeqCst) > 0);
        let mut buf = [0u8; 64];
        let n = be.read(Some(sw), &mut buf);
        assert_eq!(n, 64);
        assert!(buf.iter().all(|&b| b == 0x80));
    }

    #[test]
    fn invalid_settings_close_the_voice() {
        let (be, _clock) = backend(
            AudiodevU::for_tag(ruvm_qapi::types::AudiodevDriver::None),
            Box::new(NoneDriver),
        );
        let bad = AudSettings { freq: 0, nchannels: 2, fmt: AudioFormat::S16, big_endian: false };
        assert!(be.open_out(None, "x", Arc::new(|_| {}), &bad).is_none());
        assert_eq!(init_nb_voices("audio-wav", false, 1, 0, 0), 0);
    }
}
