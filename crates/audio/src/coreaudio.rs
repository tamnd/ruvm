// SPDX-License-Identifier: GPL-2.0-or-later

//! The `coreaudio` driver, QEMU's `audio/coreaudio.m`, over the `objc2-core-audio` crate.
//!
//! QEMU plays through an IOProc on the default output device. The IOProc runs on a CoreAudio
//! thread and copies out of the voice's emulated ring buffer under a mutex, and a listener on the
//! default output device closes and reopens the device when the user picks another one. Here the
//! ring buffer lives in the driver instead of the engine's `HwCore`, since the IOProc reads it
//! from another thread, and the voice's `get_buffer_out` hands the engine a staging buffer that
//! `put_buffer_out` copies in under the lock. The device state the listener and `enable_out`
//! share sits behind a second mutex, which stands in for the BQL the listener takes in QEMU.
//!
//! The FFI calls are unsafe. They are kept to a handful of small functions here, as `ruvm-ui`
//! does for cocoa.

#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, MutexGuard};

use objc2_core_audio::{
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioObjectAddPropertyListener, AudioObjectGetPropertyData, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectRemovePropertyListener, AudioObjectSetPropertyData,
    kAudioDevicePermissionsError, kAudioDevicePropertyBufferFrameSize,
    kAudioDevicePropertyBufferFrameSizeRange, kAudioDevicePropertyDeviceIsRunning,
    kAudioDevicePropertyScopeOutput, kAudioDevicePropertyStreamFormat, kAudioDeviceUnknown,
    kAudioDeviceUnsupportedFormatError, kAudioHardwareBadDeviceError, kAudioHardwareBadObjectError,
    kAudioHardwareBadPropertySizeError, kAudioHardwareBadStreamError,
    kAudioHardwareIllegalOperationError, kAudioHardwareNoError, kAudioHardwareNotRunningError,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioHardwareUnknownPropertyError,
    kAudioHardwareUnspecifiedError, kAudioHardwareUnsupportedOperationError,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
};
use objc2_core_audio_types::{
    AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, AudioValueRange,
    kAudioFormatLinearPCM, kLinearPCMFormatFlagIsFloat,
};
use ruvm_base::report::{error_report, warn_report};
use ruvm_qapi::types::{AudioFormat, Audiodev, AudiodevCoreaudioPerDirectionOptions, AudiodevU};

use crate::mixeng::format_bits;
use crate::pcm::{
    AudSettings, Driver, HwCore, HwInit, InitCtx, PcmInfo, PcmOut, error_printf, ring_posb,
};

/// CoreAudio's `OSStatus`.
type OsStatus = i32;

/// The buffer length `coreaudio_init_out()` asks for when the user leaves it out.
const DEFAULT_BUFFER_USECS: u32 = 11610;

/// The ring buffer is that many device buffers long when `buffer-count` is not set.
const DEFAULT_BUFFER_COUNT: u32 = 4;

/// `voice_out_addr`: the system object's default output device.
const VOICE_OUT_ADDR: AudioObjectPropertyAddress = AudioObjectPropertyAddress {
    mSelector: kAudioHardwarePropertyDefaultOutputDevice,
    mScope: kAudioObjectPropertyScopeGlobal,
    mElement: kAudioObjectPropertyElementMain,
};

/// A device property of the output scope.
fn out_addr(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioDevicePropertyScopeOutput,
        mElement: kAudioObjectPropertyElementMain,
    }
}

/// The property types read and written here, plain C structs any bit pattern of which is valid.
trait Property: Copy {}
impl Property for u32 {}
impl Property for AudioValueRange {}
impl Property for AudioStreamBasicDescription {}

/// `AudioObjectGetPropertyData()` into `value`, with no qualifier.
fn get_property<T: Property>(
    id: AudioObjectID,
    addr: &AudioObjectPropertyAddress,
    value: &mut T,
) -> OsStatus {
    let mut size = size_of::<T>() as u32;
    // SAFETY: `addr` and `size` are live for the call and `value` has room for `size` bytes of a
    // type that takes any bit pattern.
    unsafe {
        AudioObjectGetPropertyData(
            id,
            NonNull::from(addr),
            0,
            ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(value).cast(),
        )
    }
}

/// `AudioObjectSetPropertyData()` from `value`, with no qualifier.
fn set_property<T: Property>(
    id: AudioObjectID,
    addr: &AudioObjectPropertyAddress,
    value: &T,
) -> OsStatus {
    // SAFETY: `addr` and `value` are live for the call, and CoreAudio only reads `value`.
    unsafe {
        AudioObjectSetPropertyData(
            id,
            NonNull::from(addr),
            0,
            ptr::null(),
            size_of::<T>() as u32,
            NonNull::from(value).cast(),
        )
    }
}

/// `coreaudio_get_voice_out()`.
fn get_voice_out(id: &mut AudioObjectID) -> OsStatus {
    get_property(kAudioObjectSystemObject as AudioObjectID, &VOICE_OUT_ADDR, id)
}

/// `coreaudio_get_out_isrunning()`.
fn get_out_isrunning(id: AudioObjectID, result: &mut u32) -> OsStatus {
    get_property(id, &out_addr(kAudioDevicePropertyDeviceIsRunning), result)
}

/// The IOProc calls the voice gives CoreAudio. The IOProc ID is an opaque handle, so they are
/// all one unsafe block.
enum IoCall {
    Start,
    Stop,
    Destroy,
}

fn ioproc_call(call: IoCall, id: AudioObjectID, ioprocid: AudioDeviceIOProcID) -> OsStatus {
    // SAFETY: the calls take the device and the IOProc ID by value and CoreAudio checks both.
    unsafe {
        match call {
            IoCall::Start => AudioDeviceStart(id, ioprocid),
            IoCall::Stop => AudioDeviceStop(id, ioprocid),
            IoCall::Destroy => AudioDeviceDestroyIOProcID(id, ioprocid),
        }
    }
}

/// The status codes `coreaudio_logstatus()` knows by name.
const STATUS_NAMES: &[(OsStatus, &str)] = {
    macro_rules! names {
        ($($c:ident),*) => { &[$(($c, stringify!($c))),*] };
    }
    names!(
        kAudioHardwareNoError,
        kAudioHardwareNotRunningError,
        kAudioHardwareUnspecifiedError,
        kAudioHardwareUnknownPropertyError,
        kAudioHardwareBadPropertySizeError,
        kAudioHardwareIllegalOperationError,
        kAudioHardwareBadDeviceError,
        kAudioHardwareBadStreamError,
        kAudioHardwareUnsupportedOperationError,
        kAudioDeviceUnsupportedFormatError,
        kAudioDevicePermissionsError
    )
};

/// `coreaudio_logstatus()`.
fn logstatus(status: OsStatus) {
    match STATUS_NAMES.iter().find(|(c, _)| *c == status) {
        Some((_, s)) => error_printf(&format!(" Reason: {s}")),
        None => error_printf(&format!(" Reason: status code {status}")),
    }
}

/// `coreaudio_logerr()`.
fn logerr(status: OsStatus, msg: &str) {
    error_printf(&format!("coreaudio: {msg}"));
    logstatus(status);
    error_printf("\n");
}

/// `coreaudio_playback_logerr()`.
fn playback_logerr(status: OsStatus, msg: &str) {
    error_printf(&format!("coreaudio: Could not initialize playback: {msg}"));
    logstatus(status);
    error_printf("\n");
}

/// The emulated ring buffer of `HWVoiceOut` with what the IOProc needs, behind QEMU's
/// `buf_mutex`.
#[derive(Debug, Default)]
struct Ring {
    buf: Vec<u8>,
    pos: usize,
    pending: usize,
    bytes_per_frame: usize,
    /// The device the IOProc plays to.
    device_id: AudioObjectID,
    /// The device's buffer in frames.
    device_frame_size: u32,
    /// `hw.samples`, the ring length in frames.
    samples: usize,
}

impl Ring {
    /// `audio_generic_initialize_buffer_out()`.
    fn initialize(&mut self) {
        self.buf = vec![0; self.samples * self.bytes_per_frame];
        self.pos = 0;
        self.pending = 0;
    }

    /// `audio_generic_buffer_get_free()`.
    fn free(&self) -> usize {
        if self.buf.is_empty() {
            self.samples * self.bytes_per_frame
        } else {
            self.buf.len() - self.pending
        }
    }

    /// The size `audio_generic_get_buffer_out()` gives back: the free bytes up to the end of the
    /// ring.
    fn free_run(&mut self) -> usize {
        if self.buf.is_empty() {
            self.initialize();
        }
        (self.buf.len() - self.pending).min(self.buf.len() - self.pos)
    }

    /// `audio_generic_put_buffer_out()` for bytes already at the write position.
    fn commit(&mut self, size: usize) {
        self.pending += size;
        self.pos = (self.pos + size) % self.buf.len();
    }

    /// The body of `out_device_ioproc()`: one device buffer of audio, or nothing when the ring
    /// holds less than that.
    fn feed(&mut self, in_device: AudioObjectID, mut out: &mut [u8]) {
        if in_device != self.device_id {
            return;
        }
        let pending_frames = self.pending / self.bytes_per_frame;
        if pending_frames < self.device_frame_size as usize {
            return;
        }
        let mut len = (self.device_frame_size as usize * self.bytes_per_frame).min(out.len());
        while len > 0 {
            let start = ring_posb(self.pos, self.pending, self.buf.len());
            assert!(start < self.buf.len());
            let write_len = self.pending.min(len).min(self.buf.len() - start);
            let (head, rest) = out.split_at_mut(write_len);
            head.copy_from_slice(&self.buf[start..start + write_len]);
            self.pending -= write_len;
            len -= write_len;
            out = rest;
        }
    }
}

/// The device half of `CoreaudioVoiceOut`, which the listener changes. QEMU guards it with the
/// BQL.
#[derive(Debug)]
struct Dev {
    device_id: AudioObjectID,
    ioprocid: AudioDeviceIOProcID,
    enabled: bool,
    frame_size_setting: i32,
    buffer_count: u32,
}

/// What the voice, its IOProc and its listener share.
#[derive(Debug)]
struct Shared {
    info: PcmInfo,
    ring: Mutex<Ring>,
    dev: Mutex<Dev>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `out_device_ioproc()`. Runs on a CoreAudio thread and takes only the ring lock.
extern "C-unwind" fn out_device_ioproc(
    in_device: AudioObjectID,
    _in_now: NonNull<AudioTimeStamp>,
    _in_input_data: NonNull<AudioBufferList>,
    _in_input_time: NonNull<AudioTimeStamp>,
    out_output_data: NonNull<AudioBufferList>,
    _in_output_time: NonNull<AudioTimeStamp>,
    hwptr: *mut c_void,
) -> OsStatus {
    // SAFETY: `hwptr` is the `Shared` the voice registered, which is never freed (see
    // `CoreaudioDriver::init_out`). CoreAudio passes an output buffer list whose first buffer
    // holds `mDataByteSize` bytes at `mData` for the length of the call.
    let (shared, out) = unsafe {
        let b = &out_output_data.as_ref().mBuffers[0];
        let out = if b.mData.is_null() {
            None
        } else {
            Some(std::slice::from_raw_parts_mut(b.mData.cast::<u8>(), b.mDataByteSize as usize))
        };
        (&*hwptr.cast::<Shared>(), out)
    };
    if let Some(out) = out {
        lock(&shared.ring).feed(in_device, out);
    }
    0
}

/// `handle_voice_out_change()`: reopens the voice on the new default output device.
extern "C-unwind" fn handle_voice_out_change(
    _in_object_id: AudioObjectID,
    _in_number_addresses: u32,
    _in_addresses: NonNull<AudioObjectPropertyAddress>,
    in_client_data: *mut c_void,
) -> OsStatus {
    // SAFETY: `in_client_data` is the `Shared` the voice registered, which is never freed.
    let shared = unsafe { &*in_client_data.cast::<Shared>() };
    let mut dev = lock(&shared.dev);
    if dev.device_id != 0 {
        shared.fini_out_device(&mut dev);
    }
    shared.init_out_device(&mut dev);
    if dev.device_id != 0 {
        update_out_device_playback_state(&dev);
    }
    0
}

/// `AudioObjectAddPropertyListener()` or `AudioObjectRemovePropertyListener()` for the default
/// output device, with `shared` as the client data.
fn voice_out_listener(shared: &Arc<Shared>, add: bool) -> OsStatus {
    let client = Arc::as_ptr(shared).cast_mut().cast::<c_void>();
    let system = kAudioObjectSystemObject as AudioObjectID;
    let addr = NonNull::from(&VOICE_OUT_ADDR);
    // SAFETY: the address is a constant, and `client` stays valid for the rest of the process
    // because the voice leaks one reference to it.
    unsafe {
        if add {
            AudioObjectAddPropertyListener(system, addr, Some(handle_voice_out_change), client)
        } else {
            AudioObjectRemovePropertyListener(system, addr, Some(handle_voice_out_change), client)
        }
    }
}

impl Shared {
    /// `init_out_device()`.
    fn init_out_device(self: &Shared, dev: &mut Dev) -> OsStatus {
        let info = self.info;
        let bpf = info.bytes_per_frame as u32;
        let stream_basic_description = AudioStreamBasicDescription {
            mBitsPerChannel: format_bits(info.af) as u32,
            mBytesPerFrame: bpf,
            mBytesPerPacket: bpf,
            mChannelsPerFrame: info.nchannels as u32,
            mFormatFlags: kLinearPCMFormatFlagIsFloat,
            mFormatID: kAudioFormatLinearPCM,
            mFramesPerPacket: 1,
            mSampleRate: f64::from(info.freq),
            mReserved: 0,
        };

        let mut device_id: AudioObjectID = 0;
        let status = get_voice_out(&mut device_id);
        if status != kAudioHardwareNoError {
            playback_logerr(status, "Could not get default output device");
            return status;
        }
        if device_id == kAudioDeviceUnknown {
            error_report("coreaudio: Could not initialize playback: Unknown audio device");
            return status;
        }

        // Get the minimum and maximum buffer frame sizes.
        let mut value_range = AudioValueRange { mMinimum: 0.0, mMaximum: 0.0 };
        let status = get_property(
            device_id,
            &out_addr(kAudioDevicePropertyBufferFrameSizeRange),
            &mut value_range,
        );
        if status == kAudioHardwareBadObjectError {
            return 0;
        }
        if status != kAudioHardwareNoError {
            playback_logerr(status, "Could not get device buffer frame range");
            return status;
        }

        let setting = f64::from(dev.frame_size_setting);
        let mut device_frame_size = if value_range.mMinimum > setting {
            warn_report(&format!(
                "coreaudio: Upsizing buffer frames to {:.6}",
                value_range.mMinimum
            ));
            value_range.mMinimum as u32
        } else if value_range.mMaximum < setting {
            warn_report(&format!(
                "coreaudio: Downsizing buffer frames to {:.6}",
                value_range.mMaximum
            ));
            value_range.mMaximum as u32
        } else {
            dev.frame_size_setting as u32
        };

        // Set the buffer frame size.
        let addr = out_addr(kAudioDevicePropertyBufferFrameSize);
        let status = set_property(device_id, &addr, &device_frame_size);
        if status == kAudioHardwareBadObjectError {
            return 0;
        }
        if status != kAudioHardwareNoError {
            playback_logerr(
                status,
                &format!("Could not set device buffer frame size {device_frame_size}"),
            );
            return status;
        }

        // Get the buffer frame size.
        let status = get_property(device_id, &addr, &mut device_frame_size);
        if status == kAudioHardwareBadObjectError {
            return 0;
        }
        if status != kAudioHardwareNoError {
            playback_logerr(status, "Could not get device buffer frame size");
            return status;
        }

        // Set the sample rate.
        let status = set_property(
            device_id,
            &out_addr(kAudioDevicePropertyStreamFormat),
            &stream_basic_description,
        );
        if status == kAudioHardwareBadObjectError {
            return 0;
        }
        if status != kAudioHardwareNoError {
            playback_logerr(
                status,
                &format!("Could not set samplerate {:.6}", stream_basic_description.mSampleRate),
            );
            return status;
        }

        // Set the callback. CoreAudio calls the IOProc with a HAL mutex held that
        // AudioObjectGetPropertyData() takes too, so nothing here calls into CoreAudio with the
        // ring lock held.
        let mut ioprocid: AudioDeviceIOProcID = None;
        let client = ptr::from_ref(self).cast_mut().cast::<c_void>();
        // SAFETY: `client` is the leaked `Shared` (see `CoreaudioDriver::init_out`), and
        // `ioprocid` is live for the call.
        let status = unsafe {
            AudioDeviceCreateIOProcID(
                device_id,
                Some(out_device_ioproc),
                client,
                NonNull::from(&mut ioprocid),
            )
        };
        if status == kAudioHardwareBadDeviceError {
            return 0;
        }
        if status != kAudioHardwareNoError || ioprocid.is_none() {
            playback_logerr(status, "Could not set IOProc");
            return status;
        }

        dev.device_id = device_id;
        dev.ioprocid = ioprocid;
        let mut ring = lock(&self.ring);
        ring.device_id = device_id;
        ring.device_frame_size = device_frame_size;
        ring.samples = dev.buffer_count as usize * device_frame_size as usize;
        ring.initialize();
        0
    }

    /// `fini_out_device()`.
    fn fini_out_device(&self, dev: &mut Dev) {
        // Stop playback.
        let mut isrunning = 0;
        let status = get_out_isrunning(dev.device_id, &mut isrunning);
        if status != kAudioHardwareBadObjectError {
            if status != kAudioHardwareNoError {
                logerr(status, "Could not determine whether device is playing");
            }
            if isrunning != 0 {
                let status = ioproc_call(IoCall::Stop, dev.device_id, dev.ioprocid);
                if status != kAudioHardwareBadDeviceError && status != kAudioHardwareNoError {
                    logerr(status, "Could not stop playback");
                }
            }
        }

        // Remove the callback.
        let status = ioproc_call(IoCall::Destroy, dev.device_id, dev.ioprocid);
        if status != kAudioHardwareBadDeviceError && status != kAudioHardwareNoError {
            logerr(status, "Could not remove IOProc");
        }
        dev.device_id = kAudioDeviceUnknown;
        lock(&self.ring).device_id = kAudioDeviceUnknown;
    }
}

/// `update_out_device_playback_state()`.
fn update_out_device_playback_state(dev: &Dev) {
    let mut isrunning = 0;
    let status = get_out_isrunning(dev.device_id, &mut isrunning);
    if status != kAudioHardwareNoError {
        if status != kAudioHardwareBadObjectError {
            logerr(status, "Could not determine whether device is playing");
        }
        return;
    }

    if dev.enabled {
        // Start playback.
        if isrunning == 0 {
            let status = ioproc_call(IoCall::Start, dev.device_id, dev.ioprocid);
            if status != kAudioHardwareBadDeviceError && status != kAudioHardwareNoError {
                logerr(status, "Could not resume playback");
            }
        }
    } else if isrunning != 0 {
        // Stop playback.
        let status = ioproc_call(IoCall::Stop, dev.device_id, dev.ioprocid);
        if status != kAudioHardwareBadDeviceError && status != kAudioHardwareNoError {
            logerr(status, "Could not pause playback");
        }
    }
}

/// The `out` options of a coreaudio audiodev.
fn coreaudio_pdo(dev: &Audiodev) -> AudiodevCoreaudioPerDirectionOptions {
    match &dev.u {
        AudiodevU::Coreaudio(o) => o.out.clone().unwrap_or_default(),
        _ => AudiodevCoreaudioPerDirectionOptions::default(),
    }
}

/// The `coreaudio` driver, QEMU's `AudioCoreaudio`.
#[derive(Debug, Default)]
pub struct CoreaudioDriver;

impl Driver for CoreaudioDriver {
    fn type_name(&self) -> &'static str {
        "audio-coreaudio"
    }

    fn max_voices_out(&self) -> i32 {
        1
    }

    fn max_voices_in(&self) -> i32 {
        0
    }

    fn has_init_in(&self) -> bool {
        false
    }

    /// `coreaudio_init_out()`.
    fn init_out(&self, ctx: &InitCtx<'_>, as_: &AudSettings) -> Option<HwInit<Box<dyn PcmOut>>> {
        let mut obt = *as_;
        obt.fmt = AudioFormat::F32;
        let info = PcmInfo::new(&obt);
        let cpdo = coreaudio_pdo(ctx.dev);
        let shared = Arc::new(Shared {
            info,
            ring: Mutex::new(Ring { bytes_per_frame: info.bytes_per_frame, ..Ring::default() }),
            dev: Mutex::new(Dev {
                device_id: kAudioDeviceUnknown,
                ioprocid: None,
                enabled: false,
                frame_size_setting: ctx.pdo.buffer_frames(&obt, DEFAULT_BUFFER_USECS),
                buffer_count: cpdo.buffer_count.unwrap_or(DEFAULT_BUFFER_COUNT),
            }),
        });
        // CoreAudio may still be inside the listener or the IOProc when removing them returns,
        // so the pointer it holds has to stay valid for good. The voice leaks one reference and
        // drops the ring buffer when it closes, which leaves a few words behind.
        let _ = Arc::into_raw(Arc::clone(&shared));

        let mut dev = lock(&shared.dev);
        let status = voice_out_listener(&shared, true);
        if status != kAudioHardwareNoError {
            playback_logerr(status, "Could not listen to voice property change");
            return None;
        }

        if shared.init_out_device(&mut dev) != 0 {
            let status = voice_out_listener(&shared, false);
            if status != kAudioHardwareNoError {
                playback_logerr(status, "Could not remove voice property change listener");
            }
            return None;
        }
        drop(dev);

        let samples = lock(&shared.ring).samples;
        Some(HwInit {
            pcm: Box::new(CoreaudioVoiceOut { shared, stage: Vec::new() }),
            info,
            samples,
            poll_mode: false,
        })
    }
}

/// A playback voice, QEMU's `CoreaudioVoiceOut`.
struct CoreaudioVoiceOut {
    shared: Arc<Shared>,
    /// Where the engine mixes before `put_buffer_out` copies into the ring.
    stage: Vec<u8>,
}

impl PcmOut for CoreaudioVoiceOut {
    /// `coreaudio_write()`: `audio_generic_write()` with the ring locked once.
    fn write(&mut self, _info: &PcmInfo, buf: &[u8]) -> usize {
        let mut ring = lock(&self.shared.ring);
        let size = buf.len().min(ring.free());
        let mut total = 0;
        while total < size {
            let dst_size = ring.free_run();
            if dst_size == 0 {
                break;
            }
            let copy_size = (size - total).min(dst_size);
            let pos = ring.pos;
            ring.buf[pos..pos + copy_size].copy_from_slice(&buf[total..total + copy_size]);
            ring.commit(copy_size);
            total += copy_size;
        }
        total
    }

    /// `coreaudio_buffer_get_free()`.
    fn buffer_get_free(&mut self, _hw: &HwCore) -> Option<usize> {
        Some(lock(&self.shared.ring).free())
    }

    /// `coreaudio_get_buffer_out()`.
    fn get_buffer_out<'a>(
        &'a mut self,
        _hw: &'a mut HwCore,
        _size: usize,
    ) -> (Option<&'a mut [u8]>, usize) {
        let n = lock(&self.shared.ring).free_run();
        self.stage.resize(n, 0);
        (Some(&mut self.stage[..n]), n)
    }

    /// `coreaudio_put_buffer_out()`.
    fn put_buffer_out(&mut self, _hw: &mut HwCore, size: usize) -> usize {
        let mut ring = lock(&self.shared.ring);
        let size = size.min(ring.free_run()).min(self.stage.len());
        let pos = ring.pos;
        ring.buf[pos..pos + size].copy_from_slice(&self.stage[..size]);
        ring.commit(size);
        size
    }

    /// `coreaudio_enable_out()`.
    fn enable_out(&mut self, enable: bool) {
        let mut dev = lock(&self.shared.dev);
        dev.enabled = enable;
        update_out_device_playback_state(&dev);
    }

    /// `coreaudio_fini_out()`.
    fn fini_out(&mut self, _hw: &HwCore) {
        let status = voice_out_listener(&self.shared, false);
        if status != kAudioHardwareNoError {
            logerr(status, "Could not remove voice property change listener");
        }
        self.shared.fini_out_device(&mut lock(&self.shared.dev));
        lock(&self.shared.ring).buf = Vec::new();
    }
}
