// SPDX-License-Identifier: GPL-2.0-or-later

//! The `dsound` driver, QEMU's `audio/dsoundaudio.c` with `dsound_template.h` and
//! `audio_win_int.c`, over the `windows` crate.
//!
//! Each voice is a looping DirectSound buffer. The engine mixes straight into the part of the
//! buffer `get_buffer_out` locks and `put_buffer_out` unlocks, and the free space is the distance
//! from the play cursor back to where the voice last wrote. Capture runs the same way the other
//! direction. QEMU keeps the buffer position in `hw->pos_emul` and `hw->size_emul`, which here sit
//! in the voice. The playback and capture halves of the template share the [`DsBuf`] trait.
//!
//! The COM calls are unsafe. They are kept to the small functions here, as `ruvm-ui` does for
//! cocoa.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::fmt;
use std::ptr;

use ruvm_base::report::{error_report, warn_report};
use ruvm_base::{Error, Result};
use ruvm_qapi::types::{AudioFormat, Audiodev};
use windows::Win32::Foundation::{
    CLASS_E_NOAGGREGATION, E_ACCESSDENIED, E_FAIL, E_INVALIDARG, E_NOINTERFACE, E_NOTIMPL,
    E_OUTOFMEMORY,
};
use windows::Win32::Media::Audio::DirectSound::{
    CLSID_DirectSound, CLSID_DirectSoundCapture, DS_NO_VIRTUALIZATION, DSBCAPS,
    DSBCAPS_GETCURRENTPOSITION2, DSBCAPS_GLOBALFOCUS, DSBLOCK_ENTIREBUFFER, DSBPLAY_LOOPING,
    DSBSTATUS_BUFFERLOST, DSBSTATUS_PLAYING, DSBUFFERDESC, DSCBCAPS, DSCBLOCK_ENTIREBUFFER,
    DSCBSTART_LOOPING, DSCBSTATUS_CAPTURING, DSCBUFFERDESC, DSSCL_PRIORITY, IDirectSound,
    IDirectSoundBuffer, IDirectSoundCapture, IDirectSoundCaptureBuffer,
};
use windows::Win32::Media::Audio::{WAVE_FORMAT_PCM, WAVEFORMATEX};
use windows::Win32::Media::Multimedia::WAVE_FORMAT_IEEE_FLOAT;
use windows::Win32::System::Com::{CLSCTX_ALL, CoCreateInstance, CoInitialize};
use windows::Win32::UI::WindowsAndMessaging::GetDesktopWindow;
use windows::core::{HRESULT, Interface};

use crate::mixeng::format_bits;
use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmIn, PcmInfo, PcmOut, Pdo, error_printf,
};

/// The buffer length `dsound_init_out()` and `dsound_init_in()` ask for when the user leaves it
/// out.
const DEFAULT_BUFFER_USECS: u32 = 92880;

/// `MAKE_HRESULT(sev, _FACDS, code)` from dsound.h, which the `windows` crate leaves out.
const fn ds_hresult(sev: u32, code: u32) -> HRESULT {
    HRESULT(((sev << 31) | (0x878 << 16) | code) as i32)
}

const DS_OK: HRESULT = HRESULT(0);
const DS_INCOMPLETE: HRESULT = ds_hresult(0, 20);
const DSERR_ALLOCATED: HRESULT = ds_hresult(1, 10);
const DSERR_CONTROLUNAVAIL: HRESULT = ds_hresult(1, 30);
const DSERR_INVALIDCALL: HRESULT = ds_hresult(1, 50);
const DSERR_PRIOLEVELNEEDED: HRESULT = ds_hresult(1, 70);
const DSERR_BADFORMAT: HRESULT = ds_hresult(1, 100);
const DSERR_NODRIVER: HRESULT = ds_hresult(1, 120);
const DSERR_ALREADYINITIALIZED: HRESULT = ds_hresult(1, 130);
const DSERR_BUFFERLOST: HRESULT = ds_hresult(1, 150);
const DSERR_OTHERAPPHASPRIO: HRESULT = ds_hresult(1, 160);
const DSERR_UNINITIALIZED: HRESULT = ds_hresult(1, 170);
const DSERR_BUFFERTOOSMALL: HRESULT = ds_hresult(1, 180);
const DSERR_DS8_REQUIRED: HRESULT = ds_hresult(1, 190);
const DSERR_SENDLOOP: HRESULT = ds_hresult(1, 200);
const DSERR_BADSENDBUFFERGUID: HRESULT = ds_hresult(1, 210);
const DSERR_FXUNAVAILABLE: HRESULT = ds_hresult(1, 220);
const DSERR_OBJECTNOTFOUND: HRESULT = ds_hresult(1, 4449);

/// `dserror()`: what each result code means, in QEMU's words.
const DSERRORS: &[(HRESULT, &str)] = &[
    (DS_OK, "The method succeeded"),
    (DS_NO_VIRTUALIZATION, "The buffer was created, but another 3D algorithm was substituted"),
    (DS_INCOMPLETE, "The method succeeded, but not all the optional effects were obtained"),
    (E_ACCESSDENIED, "The request failed because access was denied"),
    (
        DSERR_ALLOCATED,
        "The request failed because resources, such as a priority level, were already in use \
         by another caller",
    ),
    (DSERR_ALREADYINITIALIZED, "The object is already initialized"),
    (DSERR_BADFORMAT, "The specified wave format is not supported"),
    (
        DSERR_BADSENDBUFFERGUID,
        "The GUID specified in an audiopath file does not match a valid mix-in buffer",
    ),
    (DSERR_BUFFERLOST, "The buffer memory has been lost and must be restored"),
    (DSERR_BUFFERTOOSMALL, "The buffer size is not great enough to enable effects processing"),
    (
        DSERR_CONTROLUNAVAIL,
        "The buffer control (volume, pan, and so on) requested by the caller is not available. \
         Controls must be specified when the buffer is created, using the dwFlags member of \
         DSBUFFERDESC",
    ),
    (
        DSERR_DS8_REQUIRED,
        "A DirectSound object of class CLSID_DirectSound8 or later is required for the \
         requested functionality. For more information, see IDirectSound8 Interface",
    ),
    (
        DSERR_FXUNAVAILABLE,
        "The effects requested could not be found on the system, or they are in the wrong order \
         or in the wrong location; for example, an effect expected in hardware was found in \
         software",
    ),
    (E_FAIL, "An undetermined error occurred inside the DirectSound subsystem"),
    (DSERR_INVALIDCALL, "This function is not valid for the current state of this object"),
    (E_INVALIDARG, "An invalid parameter was passed to the returning function"),
    (CLASS_E_NOAGGREGATION, "The object does not support aggregation"),
    (
        DSERR_NODRIVER,
        "No sound driver is available for use, or the given GUID is not a valid DirectSound \
         device ID",
    ),
    (E_NOINTERFACE, "The requested COM interface is not available"),
    (DSERR_OBJECTNOTFOUND, "The requested object was not found"),
    (
        DSERR_OTHERAPPHASPRIO,
        "Another application has a higher priority level, preventing this call from succeeding",
    ),
    (
        E_OUTOFMEMORY,
        "The DirectSound subsystem could not allocate sufficient memory to complete the \
         caller's request",
    ),
    (DSERR_PRIOLEVELNEEDED, "A cooperative level of DSSCL_PRIORITY or higher is required"),
    (DSERR_SENDLOOP, "A circular loop of send effects was detected"),
    (
        DSERR_UNINITIALIZED,
        "The Initialize method has not been called or has not been called successfully before \
         other methods were called",
    ),
    (E_NOTIMPL, "The function called is not supported at this time"),
];

/// `dserror()`.
fn dserror(hr: HRESULT) -> Option<&'static str> {
    DSERRORS.iter().find(|(c, _)| *c == hr).map(|(_, s)| *s)
}

/// The reason `dserror_set()` and `dsound_log_hresult()` give for `hr`.
fn reason(hr: HRESULT) -> String {
    match dserror(hr) {
        Some(s) => s.to_string(),
        None => format!("Unknown (HRESULT: 0x{:x})", hr.0 as u32),
    }
}

/// `dserror_set()`.
fn dserror_set(hr: HRESULT, msg: &str) -> Error {
    Error::generic(format!("{msg}: {}", reason(hr)))
}

/// `dsound_logerr()`.
fn logerr(hr: HRESULT, msg: &str) {
    error_printf(&format!("dsound: {msg} Reason: {}\n", reason(hr)));
}

/// `dsound_logerr2()`.
fn logerr2(hr: HRESULT, typ: &str, msg: &str) {
    error_printf(&format!("dsound: Could not initialize {typ}: {msg} Reason: {}\n", reason(hr)));
}

/// `audio_ring_dist()`.
fn ring_dist(dst: usize, src: usize, len: usize) -> usize {
    if dst >= src { dst - src } else { len - src + dst }
}

/// `waveformat_from_audio_settings()`.
fn waveformat_from_audio_settings(as_: &AudSettings) -> WAVEFORMATEX {
    let stereo = u32::from(as_.nchannels == 2);
    let mut wfx = WAVEFORMATEX {
        nChannels: as_.nchannels as u16,
        nSamplesPerSec: as_.freq as u32,
        nAvgBytesPerSec: (as_.freq as u32) << stereo,
        nBlockAlign: 1 << stereo,
        cbSize: 0,
        ..WAVEFORMATEX::default()
    };
    let shift = match as_.fmt {
        AudioFormat::S8 | AudioFormat::U8 => 0,
        AudioFormat::S16 | AudioFormat::U16 => 1,
        AudioFormat::S32 | AudioFormat::U32 | AudioFormat::F32 => 2,
    };
    let tag = if as_.fmt == AudioFormat::F32 { WAVE_FORMAT_IEEE_FLOAT } else { WAVE_FORMAT_PCM };
    wfx.wFormatTag = tag as u16;
    wfx.wBitsPerSample = 8 << shift;
    wfx.nAvgBytesPerSec <<= shift;
    wfx.nBlockAlign <<= shift;
    wfx
}

/// `waveformat_to_audio_settings()`.
fn waveformat_to_audio_settings(wfx: &WAVEFORMATEX) -> Option<AudSettings> {
    let (tag, nch, freq, bits) =
        (wfx.wFormatTag, wfx.nChannels, wfx.nSamplesPerSec, wfx.wBitsPerSample);
    if freq == 0 {
        error_report("dsound: Invalid wave format, frequency is zero");
        return None;
    }
    if nch != 1 && nch != 2 {
        error_report(&format!(
            "dsound: Invalid wave format, number of channels is not 1 or 2, but {nch}"
        ));
        return None;
    }
    let fmt = if u32::from(tag) == WAVE_FORMAT_PCM {
        match bits {
            8 => AudioFormat::U8,
            16 => AudioFormat::S16,
            32 => AudioFormat::S32,
            _ => {
                error_report(&format!(
                    "dsound: Invalid PCM wave format, bits per sample is not 8, 16 or 32, but \
                     {bits}"
                ));
                return None;
            }
        }
    } else if u32::from(tag) == WAVE_FORMAT_IEEE_FLOAT {
        if bits != 32 {
            error_report(&format!(
                "dsound: Invalid IEEE_FLOAT wave format, bits per sample is not 32, but {bits}"
            ));
            return None;
        }
        AudioFormat::F32
    } else {
        error_report(&format!(
            "dsound: Invalid wave format, tag is not PCM and not IEEE_FLOAT, but {tag}"
        ));
        return None;
    };
    Some(AudSettings { freq: freq as i32, nchannels: i32::from(nch), fmt, big_endian: false })
}

/// `audio_buffer_bytes()`.
fn buffer_bytes(pdo: &Pdo, as_: &AudSettings, def_usecs: u32) -> u32 {
    let samples = pdo.buffer_frames(as_, def_usecs).wrapping_mul(as_.nchannels);
    samples.wrapping_mul(format_bits(as_.fmt) as i32 / 8) as u32
}

/// A COM object or locked buffer pointer the engine's threads pass around. DirectSound objects
/// are free-threaded, and QEMU calls them from whichever thread holds the BQL.
struct Com<T>(T);

// SAFETY: see above; every use of the pointer is serialized by the engine's lock.
unsafe impl<T> Send for Com<T> {}
// SAFETY: as for `Send`.
unsafe impl<T> Sync for Com<T> {}

impl<T> fmt::Debug for Com<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Com")
    }
}

/// The bytes of a locked region, empty when DirectSound gave none.
///
/// # Safety
///
/// `p` is null or points at `len` bytes of a locked DirectSound buffer, which stay locked and
/// unaliased for `'a`.
unsafe fn locked_bytes<'a>(p: *mut c_void, len: u32) -> &'a mut [u8] {
    if p.is_null() {
        return &mut [];
    }
    // SAFETY: the caller's promise.
    unsafe { std::slice::from_raw_parts_mut(p.cast::<u8>(), len as usize) }
}

/// What a `Lock` call handed out.
#[derive(Debug)]
struct Locked {
    p1: *mut c_void,
    blen1: u32,
    p2: *mut c_void,
    blen2: u32,
}

/// The parts of `dsound_template.h` that differ between playback and capture buffers.
trait DsBuf {
    /// `NAME`.
    const NAME: &'static str;
    /// The lock flag for the whole buffer.
    const LOCK_ENTIRE: u32;

    /// `Lock()`, with the second region only when `two` is set.
    fn lock(
        &self,
        pos: u32,
        len: u32,
        l: &mut Locked,
        two: bool,
        flags: u32,
    ) -> windows::core::Result<()>;
    /// `Unlock()`.
    fn unlock(
        &self,
        p1: *mut c_void,
        p2: *mut c_void,
        blen1: u32,
        blen2: u32,
    ) -> windows::core::Result<()>;
    /// `Stop()`.
    fn stop(&self) -> windows::core::Result<()>;
    /// `GetFormat()`.
    fn get_format(&self, wfx: &mut WAVEFORMATEX) -> windows::core::Result<()>;
    /// `GetCaps()`, for `dwBufferBytes`.
    fn buffer_bytes(&self) -> windows::core::Result<u32>;
    /// Restores a lost buffer after `Lock()` failed with `hr`, returning whether `hr` was that.
    /// Only playback buffers get lost.
    fn lost(&self, _hr: HRESULT) -> bool {
        false
    }
}

impl DsBuf for IDirectSoundBuffer {
    const NAME: &'static str = "playback buffer";
    const LOCK_ENTIRE: u32 = DSBLOCK_ENTIREBUFFER;

    fn lock(
        &self,
        pos: u32,
        len: u32,
        l: &mut Locked,
        two: bool,
        flags: u32,
    ) -> windows::core::Result<()> {
        let (p2, b2) =
            if two { (Some(&raw mut l.p2), Some(&raw mut l.blen2)) } else { (None, None) };
        // SAFETY: the out pointers are live for the call.
        unsafe { self.Lock(pos, len, &mut l.p1, &mut l.blen1, p2, b2, flags) }
    }

    fn unlock(
        &self,
        p1: *mut c_void,
        p2: *mut c_void,
        blen1: u32,
        blen2: u32,
    ) -> windows::core::Result<()> {
        // SAFETY: the pointers come from this buffer's `Lock()`, or are null.
        unsafe { self.Unlock(p1, blen1, Some(p2.cast_const()), blen2) }
    }

    fn stop(&self) -> windows::core::Result<()> {
        // SAFETY: takes no arguments.
        unsafe { self.Stop() }
    }

    fn get_format(&self, wfx: &mut WAVEFORMATEX) -> windows::core::Result<()> {
        // SAFETY: `wfx` has room for the size passed.
        unsafe { self.GetFormat(Some(ptr::from_mut(wfx)), size_of::<WAVEFORMATEX>() as u32, None) }
    }

    fn buffer_bytes(&self) -> windows::core::Result<u32> {
        let mut bc = DSBCAPS { dwSize: size_of::<DSBCAPS>() as u32, ..DSBCAPS::default() };
        // SAFETY: `bc` is live for the call and its size is set.
        unsafe { self.GetCaps(&mut bc) }?;
        Ok(bc.dwBufferBytes)
    }

    fn lost(&self, hr: HRESULT) -> bool {
        if hr != DSERR_BUFFERLOST {
            return false;
        }
        if restore_out(self).is_err() {
            logerr(hr, &format!("Could not lock {}", Self::NAME));
        }
        true
    }
}

impl DsBuf for IDirectSoundCaptureBuffer {
    const NAME: &'static str = "capture buffer";
    const LOCK_ENTIRE: u32 = DSCBLOCK_ENTIREBUFFER;

    fn lock(
        &self,
        pos: u32,
        len: u32,
        l: &mut Locked,
        two: bool,
        flags: u32,
    ) -> windows::core::Result<()> {
        let (p2, b2) =
            if two { (Some(&raw mut l.p2), Some(&raw mut l.blen2)) } else { (None, None) };
        // SAFETY: the out pointers are live for the call.
        unsafe { self.Lock(pos, len, &mut l.p1, &mut l.blen1, p2, b2, flags) }
    }

    fn unlock(
        &self,
        p1: *mut c_void,
        p2: *mut c_void,
        blen1: u32,
        blen2: u32,
    ) -> windows::core::Result<()> {
        // SAFETY: the pointers come from this buffer's `Lock()`, or are null.
        unsafe { self.Unlock(p1, blen1, Some(p2.cast_const()), blen2) }
    }

    fn stop(&self) -> windows::core::Result<()> {
        // SAFETY: takes no arguments.
        unsafe { self.Stop() }
    }

    fn get_format(&self, wfx: &mut WAVEFORMATEX) -> windows::core::Result<()> {
        // SAFETY: `wfx` has room for the size passed.
        unsafe { self.GetFormat(Some(ptr::from_mut(wfx)), size_of::<WAVEFORMATEX>() as u32, None) }
    }

    fn buffer_bytes(&self) -> windows::core::Result<u32> {
        // The crate's `GetCaps()` passes a zeroed struct, and DirectSound wants `dwSize` set.
        let mut bc = DSCBCAPS { dwSize: size_of::<DSCBCAPS>() as u32, ..DSCBCAPS::default() };
        // SAFETY: the vtable entry of a live interface, with `bc` live for the call.
        unsafe { (Interface::vtable(self).GetCaps)(Interface::as_raw(self), &mut bc) }.ok()?;
        Ok(bc.dwBufferBytes)
    }
}

/// `dsound_restore_out()`.
fn restore_out(dsb: &IDirectSoundBuffer) -> std::result::Result<(), ()> {
    // SAFETY: takes no arguments.
    if let Err(e) = unsafe { dsb.Restore() } {
        logerr(e.code(), "Could not restore playback buffer");
        return Err(());
    }
    Ok(())
}

/// `dsound_unlock_out()` and `dsound_unlock_in()`.
fn unlock<B: DsBuf>(
    buf: &B,
    p1: *mut c_void,
    p2: *mut c_void,
    blen1: u32,
    blen2: u32,
) -> std::result::Result<(), ()> {
    if let Err(e) = buf.unlock(p1, p2, blen1, blen2) {
        logerr(e.code(), &format!("Could not unlock {}", B::NAME));
        return Err(());
    }
    Ok(())
}

/// `dsound_lock_out()` and `dsound_lock_in()`. The second region is asked for when `two` is set.
fn lock<B: DsBuf>(
    buf: &B,
    info: &PcmInfo,
    pos: u32,
    len: u32,
    two: bool,
    entire: bool,
) -> Option<Locked> {
    let mut l = Locked { p1: ptr::null_mut(), blen1: 0, p2: ptr::null_mut(), blen2: 0 };
    let flags = if entire { B::LOCK_ENTIRE } else { 0 };
    if let Err(e) = buf.lock(pos, len, &mut l, two, flags) {
        if !buf.lost(e.code()) {
            logerr(e.code(), &format!("Could not lock {}", B::NAME));
        }
        return None;
    }

    let bpf = info.bytes_per_frame as u32;
    if (!l.p1.is_null() && l.blen1 % bpf != 0) || (two && !l.p2.is_null() && l.blen2 % bpf != 0) {
        error_report(&format!("dsound: returned misaligned buffer {} {}", l.blen1, l.blen2));
        let _ = unlock(buf, l.p1, l.p2, l.blen1, l.blen2);
        return None;
    }

    if l.p1.is_null() && l.blen1 != 0 {
        warn_report(&format!("dsound: !p1 && blen1={}", l.blen1));
        l.blen1 = 0;
    }
    if two && l.p2.is_null() && l.blen2 != 0 {
        warn_report(&format!("dsound: !p2 && blen2={}", l.blen2));
        l.blen2 = 0;
    }
    Some(l)
}

/// `dsound_fini_out()` and `dsound_fini_in()`.
fn fini<B: DsBuf>(slot: &mut Option<Com<B>>) {
    if let Some(Com(buf)) = slot.take() {
        if let Err(e) = buf.stop() {
            logerr(e.code(), &format!("Could not stop {}", B::NAME));
        }
        // Dropping the interface releases it.
    }
}

/// The rest of `dsound_init_out()` and `dsound_init_in()` once the buffer exists: the format and
/// size it came out with.
fn init_buffer<B: DsBuf>(
    buf: &B,
    typ: &str,
    mut wfx: WAVEFORMATEX,
) -> Option<(PcmInfo, usize, usize)> {
    if let Err(e) = buf.get_format(&mut wfx) {
        logerr2(e.code(), typ, &format!("Could not get {} format", B::NAME));
        return None;
    }
    let buffer_bytes = match buf.buffer_bytes() {
        Ok(n) => n,
        Err(e) => {
            logerr2(e.code(), typ, &format!("Could not get {} caps", B::NAME));
            return None;
        }
    };
    let obt_as = waveformat_to_audio_settings(&wfx)?;
    let info = PcmInfo::new(&obt_as);
    if buffer_bytes as usize % info.bytes_per_frame != 0 {
        warn_report(&format!(
            "dsound: GetCaps returned misaligned buffer size {buffer_bytes}, alignment {}",
            info.bytes_per_frame
        ));
    }
    let size_emul = buffer_bytes as usize;
    Some((info, size_emul, size_emul / info.bytes_per_frame))
}

/// The `dsound` driver, QEMU's `AudioDsound`.
#[derive(Debug, Default)]
pub struct DsoundDriver {
    dsound: Option<Com<IDirectSound>>,
    dsound_capture: Option<Com<IDirectSoundCapture>>,
}

impl DsoundDriver {
    /// The COM half of `audio_dsound_realize()`. On error the objects made so far stay in
    /// `self`, which releases them, as `audio_dsound_finalize()` does.
    fn create(&mut self) -> std::result::Result<(), (HRESULT, &'static str)> {
        let e = |msg| move |e: windows::core::Error| (e.code(), msg);
        // SAFETY: plain COM calls with no pointers but the constant GUIDs, on interfaces just
        // created.
        unsafe {
            CoInitialize(None).ok().map_err(e("Could not initialize COM"))?;
            let ds: IDirectSound = CoCreateInstance(&CLSID_DirectSound, None, CLSCTX_ALL)
                .map_err(e("Could not create DirectSound instance"))?;
            let ds = &self.dsound.insert(Com(ds)).0;
            ds.Initialize(None).map_err(e("Could not initialize DirectSound"))?;

            let dsc: IDirectSoundCapture =
                CoCreateInstance(&CLSID_DirectSoundCapture, None, CLSCTX_ALL)
                    .map_err(e("Could not create DirectSoundCapture instance"))?;
            let dsc = &self.dsound_capture.insert(Com(dsc)).0;
            dsc.Initialize(None).map_err(e("Could not initialize DirectSoundCapture"))?;

            ds.SetCooperativeLevel(GetDesktopWindow(), DSSCL_PRIORITY)
                .map_err(e("Could not set cooperative level"))?;
        }
        Ok(())
    }
}

impl Driver for DsoundDriver {
    fn type_name(&self) -> &'static str {
        "audio-dsound"
    }

    fn max_voices_out(&self) -> i32 {
        i32::MAX
    }

    fn max_voices_in(&self) -> i32 {
        1
    }

    /// `audio_dsound_realize()`. The `latency` default it sets is filled in with the other
    /// defaults by `validate_opts()`, which is where `query-audiodevs` looks.
    fn realize(&mut self, _dev: &Audiodev) -> Result<()> {
        self.create().map_err(|(hr, msg)| dserror_set(hr, msg))
    }

    /// `dsound_init_out()`.
    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let Some(Com(dsound)) = &self.dsound else {
            error_report("dsound: Attempt to initialize voice without DirectSound object");
            return None;
        };
        let mut wfx = waveformat_from_audio_settings(as_);
        let bd = DSBUFFERDESC {
            dwSize: size_of::<DSBUFFERDESC>() as u32,
            dwFlags: DSBCAPS_GLOBALFOCUS | DSBCAPS_GETCURRENTPOSITION2,
            dwBufferBytes: buffer_bytes(ctx.pdo, as_, DEFAULT_BUFFER_USECS),
            lpwfxFormat: &mut wfx,
            ..DSBUFFERDESC::default()
        };
        let mut dsb = None;
        // SAFETY: `bd`, the format it points at and `dsb` are live for the call.
        if let Err(e) = unsafe { dsound.CreateSoundBuffer(&bd, &mut dsb, None) } {
            logerr2(e.code(), "DAC", "Could not create playback buffer");
            return None;
        }
        let mut slot = dsb.map(Com);
        let Some(Com(buf)) = &slot else { return None };
        let Some((info, size_emul, samples)) = init_buffer(buf, "DAC", wfx) else {
            fini(&mut slot);
            return None;
        };
        let voice = DsoundVoiceOut {
            dsb: slot,
            first_time: true,
            info,
            pos_emul: 0,
            size_emul,
            locked: Com(ptr::null_mut()),
        };
        Some(HwInit { pcm: Box::new(voice), info, samples, poll_mode: false })
    }

    /// `dsound_init_in()`.
    fn init_in(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmIn>>> {
        let Some(Com(capture)) = &self.dsound_capture else {
            error_report("dsound: Attempt to initialize voice without DirectSoundCapture object");
            return None;
        };
        let mut wfx = waveformat_from_audio_settings(as_);
        let bd = DSCBUFFERDESC {
            dwSize: size_of::<DSCBUFFERDESC>() as u32,
            dwBufferBytes: buffer_bytes(ctx.pdo, as_, DEFAULT_BUFFER_USECS),
            lpwfxFormat: &mut wfx,
            ..DSCBUFFERDESC::default()
        };
        let mut dscb = None;
        // SAFETY: `bd`, the format it points at and `dscb` are live for the call.
        if let Err(e) = unsafe { capture.CreateCaptureBuffer(&bd, &mut dscb, None) } {
            logerr2(e.code(), "ADC", "Could not create capture buffer");
            return None;
        }
        let mut slot = dscb.map(Com);
        let Some(Com(buf)) = &slot else { return None };
        let Some((info, size_emul, samples)) = init_buffer(buf, "ADC", wfx) else {
            fini(&mut slot);
            return None;
        };
        let voice = DsoundVoiceIn {
            dscb: slot,
            first_time: true,
            info,
            pos_emul: 0,
            size_emul,
            locked: Com(ptr::null_mut()),
        };
        Some(HwInit { pcm: Box::new(voice), info, samples, poll_mode: false })
    }
}

/// A playback voice, QEMU's `DSoundVoiceOut`.
#[derive(Debug)]
struct DsoundVoiceOut {
    dsb: Option<Com<IDirectSoundBuffer>>,
    first_time: bool,
    info: PcmInfo,
    /// `hw->pos_emul`: where the next write goes.
    pos_emul: usize,
    /// `hw->size_emul`: the buffer size in bytes.
    size_emul: usize,
    /// The region `get_buffer_out` locked, for `put_buffer_out` to unlock.
    locked: Com<*mut c_void>,
}

impl DsoundVoiceOut {
    /// `dsound_get_status_out()`.
    fn get_status(&self, dsb: &IDirectSoundBuffer) -> Option<u32> {
        // SAFETY: takes no arguments.
        let status = match unsafe { dsb.GetStatus() } {
            Ok(s) => s,
            Err(e) => {
                logerr(e.code(), "Could not get playback buffer status");
                return None;
            }
        };
        if status & DSBSTATUS_BUFFERLOST != 0 {
            let _ = restore_out(dsb);
            return None;
        }
        Some(status)
    }

    /// `dsound_clear_sample()`.
    fn clear_sample(&self, dsb: &IDirectSoundBuffer) {
        let size = self.size_emul as u32;
        let Some(l) = lock(dsb, &self.info, 0, size, true, true) else { return };
        let bpf = self.info.bytes_per_frame as u32;
        // SAFETY: both regions are locked until the unlock below and nothing else points there.
        let (b1, b2) = unsafe { (locked_bytes(l.p1, l.blen1), locked_bytes(l.p2, l.blen2)) };
        self.info.clear_buf(b1, (l.blen1 / bpf) as usize);
        self.info.clear_buf(b2, (l.blen2 / bpf) as usize);
        let _ = unlock(dsb, l.p1, l.p2, l.blen1, l.blen2);
    }

    /// `dsound_buffer_get_free()`.
    fn free(&mut self) -> usize {
        let Some(Com(dsb)) = &self.dsb else { return 0 };
        let (mut ppos, mut wpos) = (0u32, 0u32);
        let w = if self.first_time { Some(&raw mut wpos) } else { None };
        // SAFETY: the cursors are live for the call.
        if let Err(e) = unsafe { dsb.GetCurrentPosition(Some(&raw mut ppos), w) } {
            logerr(e.code(), "Could not get playback buffer position");
            return 0;
        }
        if self.first_time {
            self.pos_emul = wpos as usize;
            self.first_time = false;
        }
        ring_dist(ppos as usize, self.pos_emul, self.size_emul)
    }

    /// `dsound_get_buffer_out()`.
    fn lock_next(&mut self, size: usize) -> &mut [u8] {
        let req_size = size.min(self.size_emul - self.pos_emul);
        assert!(req_size > 0);
        let Some(Com(dsb)) = &self.dsb else { return &mut [] };
        let Some(l) = lock(dsb, &self.info, self.pos_emul as u32, req_size as u32, false, false)
        else {
            error_report("dsound: Failed to lock buffer");
            return &mut [];
        };
        self.locked = Com(l.p1);
        // SAFETY: the region stays locked until `unlock_next`, which takes `&mut self` and so
        // ends this borrow first.
        unsafe { locked_bytes(l.p1, l.blen1) }
    }

    /// `dsound_put_buffer_out()`.
    fn unlock_next(&mut self, len: usize) -> usize {
        let Some(Com(dsb)) = &self.dsb else { return 0 };
        let p1 = std::mem::replace(&mut self.locked.0, ptr::null_mut());
        if unlock(dsb, p1, ptr::null_mut(), len as u32, 0).is_err() {
            error_report("dsound: Failed to unlock buffer");
            return 0;
        }
        self.pos_emul = (self.pos_emul + len) % self.size_emul;
        len
    }
}

impl PcmOut for DsoundVoiceOut {
    /// `audio_generic_write()` over the hooks below.
    fn write(&mut self, _info: &PcmInfo, buf: &[u8]) -> usize {
        let size = buf.len().min(self.free());
        let mut total = 0;
        while total < size {
            let dst = self.lock_next(size - total);
            if dst.is_empty() {
                break;
            }
            let copy_size = (size - total).min(dst.len());
            dst[..copy_size].copy_from_slice(&buf[total..total + copy_size]);
            let proc = self.unlock_next(copy_size);
            total += proc;
            if proc == 0 || proc < copy_size {
                break;
            }
        }
        total
    }

    fn buffer_get_free(&mut self, _hw: &HwCore) -> Option<usize> {
        Some(self.free())
    }

    fn get_buffer_out<'a>(
        &'a mut self,
        _hw: &'a mut HwCore,
        size: usize,
    ) -> (Option<&'a mut [u8]>, usize) {
        let b = self.lock_next(size);
        let n = b.len();
        if n == 0 { (None, 0) } else { (Some(b), n) }
    }

    fn put_buffer_out(&mut self, _hw: &mut HwCore, size: usize) -> usize {
        self.unlock_next(size)
    }

    /// `dsound_enable_out()`.
    fn enable_out(&mut self, enable: bool) {
        let Some(Com(dsb)) = &self.dsb else {
            error_report("dsound: Attempt to control voice without a buffer");
            return;
        };
        let Some(status) = self.get_status(dsb) else { return };
        if enable {
            if status & DSBSTATUS_PLAYING != 0 {
                warn_report("dsound: Voice is already playing");
                return;
            }
            self.clear_sample(dsb);
            // SAFETY: takes no pointers.
            if let Err(e) = unsafe { dsb.Play(0, 0, DSBPLAY_LOOPING) } {
                logerr(e.code(), "Could not start playing buffer");
            }
        } else if status & DSBSTATUS_PLAYING != 0 {
            if let Err(e) = dsb.stop() {
                logerr(e.code(), "Could not stop playing buffer");
            }
        } else {
            warn_report("dsound: Voice is not playing");
        }
    }

    fn fini_out(&mut self, _hw: &HwCore) {
        fini(&mut self.dsb);
    }
}

/// A capture voice, QEMU's `DSoundVoiceIn`.
#[derive(Debug)]
struct DsoundVoiceIn {
    dscb: Option<Com<IDirectSoundCaptureBuffer>>,
    first_time: bool,
    info: PcmInfo,
    /// `hw->pos_emul`: where the next read comes from.
    pos_emul: usize,
    /// `hw->size_emul`: the buffer size in bytes.
    size_emul: usize,
    /// The region `get_buffer_in` locked, for `put_buffer_in` to unlock.
    locked: Com<*mut c_void>,
}

impl DsoundVoiceIn {
    /// `dsound_get_buffer_in()`.
    fn lock_next(&mut self, size: usize) -> &[u8] {
        let Some(Com(dscb)) = &self.dscb else { return &[] };
        let mut rpos = 0u32;
        // SAFETY: the cursor is live for the call.
        if let Err(e) = unsafe { dscb.GetCurrentPosition(None, Some(&raw mut rpos)) } {
            logerr(e.code(), "Could not get capture buffer position");
            return &[];
        }
        if self.first_time {
            self.pos_emul = rpos as usize;
            self.first_time = false;
        }
        let req_size = ring_dist(rpos as usize, self.pos_emul, self.size_emul);
        let req_size = size.min(req_size.min(self.size_emul - self.pos_emul));
        if req_size == 0 {
            return &[];
        }
        let Some(l) = lock(dscb, &self.info, self.pos_emul as u32, req_size as u32, false, false)
        else {
            error_report("dsound: Failed to lock buffer");
            return &[];
        };
        self.locked = Com(l.p1);
        // SAFETY: the region stays locked until `unlock_next`, which takes `&mut self` and so
        // ends this borrow first.
        unsafe { locked_bytes(l.p1, l.blen1) }
    }

    /// `dsound_put_buffer_in()`.
    fn unlock_next(&mut self, len: usize) {
        let Some(Com(dscb)) = &self.dscb else { return };
        let p1 = std::mem::replace(&mut self.locked.0, ptr::null_mut());
        if unlock(dscb, p1, ptr::null_mut(), len as u32, 0).is_err() {
            error_report("dsound: Failed to unlock buffer");
            return;
        }
        self.pos_emul = (self.pos_emul + len) % self.size_emul;
    }
}

impl PcmIn for DsoundVoiceIn {
    /// `audio_generic_read()` over the hooks below.
    fn read(&mut self, _info: &PcmInfo, buf: &mut [u8]) -> usize {
        let size = buf.len();
        let mut total = 0;
        while total < size {
            let src = self.lock_next(size - total);
            let src_size = src.len();
            if src_size == 0 {
                break;
            }
            buf[total..total + src_size].copy_from_slice(src);
            self.unlock_next(src_size);
            total += src_size;
        }
        total
    }

    fn get_buffer_in<'a>(&'a mut self, _hw: &'a mut HwCore, size: usize) -> &'a [u8] {
        self.lock_next(size)
    }

    fn put_buffer_in(&mut self, _hw: &mut HwCore, size: usize) {
        self.unlock_next(size);
    }

    /// `dsound_enable_in()`.
    fn enable_in(&mut self, enable: bool) {
        let Some(Com(dscb)) = &self.dscb else {
            error_report("dsound: Attempt to control capture voice without a buffer");
            return;
        };
        // SAFETY: takes no arguments.
        let status = match unsafe { dscb.GetStatus() } {
            Ok(s) => s,
            Err(e) => {
                logerr(e.code(), "Could not get capture buffer status");
                return;
            }
        };
        if enable {
            if status & DSCBSTATUS_CAPTURING != 0 {
                warn_report("dsound: Voice is already capturing");
                return;
            }
            // SAFETY: takes no pointers.
            if let Err(e) = unsafe { dscb.Start(DSCBSTART_LOOPING) } {
                logerr(e.code(), "Could not start capturing");
            }
        } else if status & DSCBSTATUS_CAPTURING != 0 {
            if let Err(e) = dscb.stop() {
                logerr(e.code(), "Could not stop capturing");
            }
        } else {
            warn_report("dsound: Voice is not capturing");
        }
    }

    fn fini_in(&mut self, _hw: &HwCore) {
        fini(&mut self.dscb);
    }
}
