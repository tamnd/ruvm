// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-sound, a port of `hw/audio/virtio-snd.c`.
//!
//! Four queues of 64 entries: control, event, tx and rx. The control queue carries the PCM
//! requests, the tx queue the guest's playback buffers and the rx queue the buffers capture
//! fills. Streams are split in two: the first half (rounded up) play, the rest capture. Every
//! stream gets the default parameters, 48 kHz stereo S16, and its voice when the device is
//! realized, and the driver may change them with `SET_PARAMS` and `PREPARE`.
//!
//! As in QEMU, a playback buffer is only taken from the guest when the voice asks for data,
//! from the backend's timer, and goes back with its status once all of it is written. Jacks,
//! channel maps and the event queue are not implemented: their requests answer
//! `VIRTIO_SND_S_NOT_SUPP` and event buffers stay queued.
//!
//! Differences from QEMU:
//!
//! - The voice callbacks need the device and the core state, which the transport owns. Whoever
//!   plugs the device hands it a [`DeviceAccess`] with [`VirtioSnd::connect`], and until then
//!   the callbacks do nothing.
//! - The `LOG_GUEST_ERROR` and `LOG_UNIMP` messages are not printed, since the workspace has no
//!   log mask. The `error_report()` ones are.
//!
//! Migration: QEMU marks the device unmigratable. Unrealize, which closes the voices, is not
//! ported.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, OnceLock};

use ruvm_audio::{AudSettings, AudioBackend, AudioCallback, SwVoiceIn, SwVoiceOut, Volume};
use ruvm_base::{Error, Result, error_report};
use ruvm_hw_virtio::virtio::{VIRTIO_F_VERSION_1, feature};
use ruvm_hw_virtio::{VirtIODevice, VirtioDeviceClass};
use ruvm_qapi::types::AudioFormat;
use ruvm_virtio_queue::DescriptorChain;

/// `TYPE_VIRTIO_SND`.
pub const TYPE_VIRTIO_SND: &str = "virtio-sound-device";
/// `TYPE_VIRTIO_SND_PCI`.
pub const TYPE_VIRTIO_SND_PCI: &str = "virtio-sound-pci";
/// `VIRTIO_ID_SOUND`.
pub const VIRTIO_ID_SOUND: u16 = 25;

/// The size of every queue.
const QUEUE_SIZE: u16 = 64;

const VQ_CONTROL: u16 = 0;
const VQ_EVENT: u16 = 1;
const VQ_TX: u16 = 2;
const VQ_RX: u16 = 3;

/// `sizeof(virtio_snd_config)`: jacks, streams, chmaps and controls.
const CONFIG_SIZE: usize = 16;
/// `VIRTIO_SND_CHMAP_MAX_SIZE`.
const CHMAP_MAX_SIZE: u32 = 18;
/// `AUDIO_MAX_CHANNELS`.
const AUDIO_MAX_CHANNELS: u8 = 16;
/// `VIRTIO_SOUND_HDA_FN_NID`.
const HDA_FN_NID: u32 = 0;

const R_JACK_INFO: u32 = 1;
const R_JACK_REMAP: u32 = 2;
const R_PCM_INFO: u32 = 0x0100;
const R_PCM_SET_PARAMS: u32 = 0x0101;
const R_PCM_PREPARE: u32 = 0x0102;
const R_PCM_RELEASE: u32 = 0x0103;
const R_PCM_START: u32 = 0x0104;
const R_PCM_STOP: u32 = 0x0105;
const R_CHMAP_INFO: u32 = 0x0200;

const S_OK: u32 = 0x8000;
const S_BAD_MSG: u32 = 0x8001;
const S_NOT_SUPP: u32 = 0x8002;

const D_OUTPUT: u8 = 0;
const D_INPUT: u8 = 1;

const FMT_S8: u8 = 3;
const FMT_U8: u8 = 4;
const FMT_S16: u8 = 5;
const FMT_U16: u8 = 6;
const FMT_S32: u8 = 17;
const FMT_U32: u8 = 18;
const FMT_FLOAT: u8 = 19;

/// `VIRTIO_SND_PCM_RATE_48000`.
const RATE_48000: u8 = 7;
/// The frame rates of `VIRTIO_SND_PCM_RATE_5512` to `VIRTIO_SND_PCM_RATE_384000`, all of
/// which the device takes.
const RATES: [u32; 14] = [
    5512, 8000, 11025, 16000, 22050, 32000, 44100, 48000, 64000, 88200, 96000, 176400, 192000,
    384000,
];

/// `supported_formats`.
const SUPPORTED_FORMATS: u32 = 1 << FMT_S8
    | 1 << FMT_U8
    | 1 << FMT_S16
    | 1 << FMT_U16
    | 1 << FMT_S32
    | 1 << FMT_U32
    | 1 << FMT_FLOAT;
/// `supported_rates`.
const SUPPORTED_RATES: u32 = (1 << RATES.len()) - 1;

/// `sizeof(virtio_snd_hdr)`.
const HDR_SIZE: usize = 4;
/// `sizeof(virtio_snd_query_info)`.
const QUERY_INFO_SIZE: usize = 16;
/// `sizeof(virtio_snd_pcm_info)`.
const PCM_INFO_SIZE: usize = 32;
/// `sizeof(virtio_snd_pcm_hdr)`.
const PCM_HDR_SIZE: usize = 8;
/// `sizeof(virtio_snd_pcm_set_params)`.
const SET_PARAMS_SIZE: usize = 24;
/// `sizeof(virtio_snd_pcm_xfer)`.
const XFER_SIZE: usize = 4;
/// `sizeof(virtio_snd_pcm_status)`.
const STATUS_SIZE: usize = 8;

/// The virtio-sound properties, `virtio_snd_config` as the command line sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtioSndConf {
    /// `jacks`, at most 8.
    pub jacks: u32,
    /// `streams`, 1 to 10.
    pub streams: u32,
    /// `chmaps`, at most 18.
    pub chmaps: u32,
}

impl Default for VirtioSndConf {
    fn default() -> Self {
        VirtioSndConf { jacks: 0, streams: 2, chmaps: 0 }
    }
}

impl VirtioSndConf {
    /// The property checks at the start of `virtio_snd_realize()`, which come before the
    /// audiodev is looked at.
    pub fn check(&self) -> Result<()> {
        if self.jacks > 8 {
            return Err(Error::generic(format!("Invalid number of jacks: {}", self.jacks)));
        }
        if self.streams < 1 || self.streams > 10 {
            return Err(Error::generic(format!("Invalid number of streams: {}", self.streams)));
        }
        if self.chmaps > CHMAP_MAX_SIZE {
            return Err(Error::generic(format!("Invalid number of channel maps: {}", self.chmaps)));
        }
        Ok(())
    }
}

/// Runs a closure on the plugged device and its core state, under the transport's lock. It
/// must not keep the transport alive.
pub type DeviceAccess =
    Box<dyn Fn(&mut dyn FnMut(&mut VirtIODevice, &mut VirtioSnd)) + Send + Sync>;

/// `virtio_snd_pcm_set_params` without its header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PcmParams {
    buffer_bytes: u32,
    period_bytes: u32,
    features: u32,
    channels: u8,
    format: u8,
    rate: u8,
}

impl PcmParams {
    /// The defaults `virtio_snd_realize()` gives every stream.
    fn default_params() -> Self {
        PcmParams {
            buffer_bytes: 8192,
            period_bytes: 2048,
            features: 0,
            channels: 2,
            format: FMT_S16,
            rate: RATE_48000,
        }
    }

    /// `virtio_snd_get_qemu_audsettings()`.
    fn audsettings(&self) -> AudSettings {
        let fmt = match self.format {
            FMT_U8 => AudioFormat::U8,
            FMT_S8 => AudioFormat::S8,
            FMT_U16 => AudioFormat::U16,
            FMT_S16 => AudioFormat::S16,
            FMT_U32 => AudioFormat::U32,
            FMT_S32 => AudioFormat::S32,
            _ => AudioFormat::F32,
        };
        AudSettings {
            freq: RATES[usize::from(self.rate)] as i32,
            nchannels: i32::from(self.channels.min(AUDIO_MAX_CHANNELS)),
            fmt,
            // Conforming to VIRTIO 1.0: always little endian.
            big_endian: false,
        }
    }
}

/// `virtio_snd_pcm_info` without its padding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PcmInfo {
    hda_fn_nid: u32,
    features: u32,
    formats: u64,
    rates: u64,
    direction: u8,
    channels_min: u8,
    channels_max: u8,
}

impl PcmInfo {
    fn to_bytes(self) -> [u8; PCM_INFO_SIZE] {
        let mut b = [0; PCM_INFO_SIZE];
        b[0..4].copy_from_slice(&self.hda_fn_nid.to_le_bytes());
        b[4..8].copy_from_slice(&self.features.to_le_bytes());
        b[8..16].copy_from_slice(&self.formats.to_le_bytes());
        b[16..24].copy_from_slice(&self.rates.to_le_bytes());
        b[24] = self.direction;
        b[25] = self.channels_min;
        b[26] = self.channels_max;
        b
    }
}

/// `VirtIOSoundPCMBuffer`: an I/O message waiting on a stream.
struct PcmBuffer {
    chain: DescriptorChain,
    vq: u16,
    /// For playback the bytes not written yet, for capture the bytes read so far.
    size: usize,
    /// For playback the first byte of `data` not written yet.
    offset: usize,
    /// For playback whether `data` has been copied from the guest.
    populated: bool,
    data: Vec<u8>,
}

/// `VirtIOSoundPCMStream`.
struct PcmStream {
    info: PcmInfo,
    params: PcmParams,
    active: bool,
    latency_bytes: u32,
    voice_out: Option<SwVoiceOut>,
    voice_in: Option<SwVoiceIn>,
    queue: VecDeque<PcmBuffer>,
}

impl PcmStream {
    /// `update_latency()`.
    fn update_latency(&mut self, used: usize) {
        self.latency_bytes =
            if self.latency_bytes as usize > used { self.latency_bytes - used as u32 } else { 0 };
    }
}

/// The virtio-sound device model, `VirtIOSound`.
pub struct VirtioSnd {
    be: Arc<AudioBackend>,
    conf: VirtioSndConf,
    pcm_params: Vec<PcmParams>,
    streams: Vec<Option<PcmStream>>,
    access: Arc<OnceLock<DeviceAccess>>,
}

impl fmt::Debug for VirtioSnd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtioSnd")
            .field("audiodev", &self.be.id())
            .field("conf", &self.conf)
            .field("pcm_params", &self.pcm_params)
            .finish_non_exhaustive()
    }
}

/// `audio_be_set_volume_out_lr(be, voice, 0, 255, 255)`.
fn full_volume() -> Volume {
    let mut v = Volume { mute: false, channels: 2, vol: [0; 16] };
    v.vol[0] = 255;
    v.vol[1] = 255;
    v
}

/// `iov_to_buf()` on the readable part of `chain` from `offset`.
fn read_at(vdev: &VirtIODevice, chain: &DescriptorChain, offset: usize, buf: &mut [u8]) -> usize {
    let mem = Arc::clone(vdev.mem());
    let mut r = chain.reader(&*mem);
    if r.skip(offset as u64) != offset as u64 {
        return 0;
    }
    r.read(buf).unwrap_or(0)
}

/// `iov_from_buf()` on the writable part of `chain` from offset 0, the pieces one after the
/// other.
fn write_parts(vdev: &VirtIODevice, chain: &DescriptorChain, parts: &[&[u8]]) {
    let mem = Arc::clone(vdev.mem());
    let mut w = chain.writer(&*mem);
    for p in parts {
        if w.write(p).unwrap_or(0) < p.len() {
            return;
        }
    }
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// `virtio_snd_pcm_status` with `status`.
fn status_bytes(status: u32, latency_bytes: u32) -> [u8; STATUS_SIZE] {
    let mut b = [0; STATUS_SIZE];
    b[0..4].copy_from_slice(&status.to_le_bytes());
    b[4..8].copy_from_slice(&latency_bytes.to_le_bytes());
    b
}

/// `return_tx_buffer()` for the first buffer of `stream`.
fn return_tx_buffer(vdev: &mut VirtIODevice, stream: &mut PcmStream) {
    let Some(buffer) = stream.queue.pop_front() else {
        return;
    };
    stream.update_latency(buffer.size);
    write_parts(vdev, &buffer.chain, &[&status_bytes(S_OK, stream.latency_bytes)]);
    vdev.push(buffer.vq, &buffer.chain, STATUS_SIZE as u32);
    vdev.notify(buffer.vq);
}

/// `return_rx_buffer()` for the first buffer of `stream`.
fn return_rx_buffer(vdev: &mut VirtIODevice, stream: &mut PcmStream) {
    let Some(buffer) = stream.queue.pop_front() else {
        return;
    };
    // Copy data -if any- to guest.
    write_parts(vdev, &buffer.chain, &[&buffer.data[..buffer.size], &status_bytes(S_OK, 0)]);
    vdev.push(buffer.vq, &buffer.chain, (STATUS_SIZE + buffer.size) as u32);
    vdev.notify(buffer.vq);
}

/// `virtio_snd_pcm_flush()`.
fn flush(vdev: &mut VirtIODevice, stream: &mut PcmStream) {
    while !stream.queue.is_empty() {
        if stream.info.direction == D_OUTPUT {
            return_tx_buffer(vdev, stream);
        } else {
            return_rx_buffer(vdev, stream);
        }
    }
}

impl VirtioSnd {
    /// A device playing and recording on `be`, with properties `conf`. They are checked when
    /// the device is realized.
    pub fn new(be: Arc<AudioBackend>, conf: VirtioSndConf) -> Self {
        VirtioSnd {
            be,
            conf,
            pcm_params: Vec::new(),
            streams: Vec::new(),
            access: Arc::new(OnceLock::new()),
        }
    }

    /// The properties.
    pub fn conf(&self) -> VirtioSndConf {
        self.conf
    }

    /// Lets the voice callbacks reach the device. Only the first call counts.
    pub fn connect(&self, access: DeviceAccess) {
        let _ = self.access.set(access);
    }

    fn stream_mut(&mut self, stream_id: u32) -> Option<&mut PcmStream> {
        self.streams.get_mut(stream_id as usize).and_then(Option::as_mut)
    }

    /// `virtio_snd_set_pcm_params()`.
    fn set_pcm_params(&mut self, vdev: &mut VirtIODevice, stream_id: u32, p: PcmParams) -> u32 {
        let Some(st_params) = self.pcm_params.get_mut(stream_id as usize) else {
            vdev.error("Streams have not been initialized.\n");
            return S_BAD_MSG;
        };
        if p.channels < 1 || p.channels > AUDIO_MAX_CHANNELS {
            error_report("Number of channels is not supported.");
            return S_NOT_SUPP;
        }
        if p.format >= 32 || SUPPORTED_FORMATS & (1 << p.format) == 0 {
            error_report("Stream format is not supported.");
            return S_NOT_SUPP;
        }
        if p.rate >= 32 || SUPPORTED_RATES & (1 << p.rate) == 0 {
            error_report("Stream rate is not supported.");
            return S_NOT_SUPP;
        }
        *st_params = p;
        S_OK
    }

    /// `virtio_snd_pcm_prepare()`.
    fn pcm_prepare(&mut self, stream_id: u32) -> u32 {
        let Some(&params) = self.pcm_params.get(stream_id as usize) else {
            return S_BAD_MSG;
        };
        let streams = self.conf.streams;
        let stream = self.streams[stream_id as usize].get_or_insert_with(|| PcmStream {
            info: PcmInfo::default(),
            params,
            active: false,
            latency_bytes: 0,
            voice_out: None,
            voice_in: None,
            queue: VecDeque::new(),
        });
        let as_ = params.audsettings();
        stream.info = PcmInfo {
            hda_fn_nid: HDA_FN_NID,
            features: 0,
            formats: u64::from(SUPPORTED_FORMATS),
            rates: u64::from(SUPPORTED_RATES),
            direction: if stream_id < streams / 2 + (streams & 1) { D_OUTPUT } else { D_INPUT },
            channels_min: 1,
            channels_max: as_.nchannels as u8,
        };
        stream.params = params;
        let access = Arc::clone(&self.access);
        if stream.info.direction == D_OUTPUT {
            let cb: AudioCallback = Arc::new(move |avail| {
                if let Some(a) = access.get() {
                    a(&mut |vdev: &mut VirtIODevice, s: &mut VirtioSnd| {
                        s.pcm_out_cb(vdev, stream_id, avail);
                    });
                }
            });
            stream.voice_out = self.be.open_out(stream.voice_out, "virtio-sound.out", cb, &as_);
            self.be.set_volume_out(stream.voice_out, &full_volume());
        } else {
            let cb: AudioCallback = Arc::new(move |avail| {
                if let Some(a) = access.get() {
                    a(&mut |vdev: &mut VirtIODevice, s: &mut VirtioSnd| {
                        s.pcm_in_cb(vdev, stream_id, avail);
                    });
                }
            });
            stream.voice_in = self.be.open_in(stream.voice_in, "virtio-sound.in", cb, &as_);
            self.be.set_volume_in(stream.voice_in, &full_volume());
        }
        S_OK
    }

    /// `virtio_snd_handle_pcm_info()`. Gives the status and the payload size.
    fn handle_pcm_info(&mut self, vdev: &VirtIODevice, chain: &DescriptorChain) -> (u32, usize) {
        let mut req = [0; QUERY_INFO_SIZE];
        if read_at(vdev, chain, 0, &mut req) != QUERY_INFO_SIZE {
            return (S_BAD_MSG, 0);
        }
        let start_id = le32(&req, 4);
        let count = le32(&req, 8);
        let size = le32(&req, 12);
        let streams = self.conf.streams;
        if start_id > streams || start_id.checked_add(count).is_none_or(|end| end > streams) {
            error_report(&format!(
                "pcm info: start_id + count is greater than the total number of streams, \
                 got: start_id = {start_id}, count = {count}"
            ));
            return (S_BAD_MSG, 0);
        }
        let in_size = chain.writable_len();
        let needed = size.checked_mul(count).and_then(|t| t.checked_add(HDR_SIZE as u32));
        if needed.is_none_or(|n| in_size < u64::from(n)) {
            error_report(&format!(
                "pcm info: buffer too small, got: {in_size}, needed: {}",
                PCM_INFO_SIZE * count as usize
            ));
            return (S_BAD_MSG, 0);
        }
        let mut payload = Vec::with_capacity(PCM_INFO_SIZE * count as usize);
        for i in 0..count {
            let stream_id = start_id + i;
            let Some(stream) = self.stream_mut(stream_id) else {
                error_report(&format!("Invalid stream id: {stream_id}"));
                return (S_BAD_MSG, 0);
            };
            payload.extend_from_slice(&stream.info.to_bytes());
        }
        // The header goes in front once the status is known.
        write_parts(vdev, chain, &[&[0; HDR_SIZE], &payload]);
        (S_OK, payload.len())
    }

    /// `virtio_snd_handle_pcm_set_params()`.
    fn handle_pcm_set_params(&mut self, vdev: &mut VirtIODevice, chain: &DescriptorChain) -> u32 {
        let mut req = [0; SET_PARAMS_SIZE];
        if read_at(vdev, chain, 0, &mut req) != SET_PARAMS_SIZE {
            return S_BAD_MSG;
        }
        let params = PcmParams {
            buffer_bytes: le32(&req, 8),
            period_bytes: le32(&req, 12),
            features: le32(&req, 16),
            channels: req[20],
            format: req[21],
            rate: req[22],
        };
        self.set_pcm_params(vdev, le32(&req, 4), params)
    }

    /// The `stream_id` after the header, for `PREPARE` and `RELEASE`.
    fn read_stream_id(vdev: &VirtIODevice, chain: &DescriptorChain) -> Option<u32> {
        let mut b = [0; 4];
        (read_at(vdev, chain, HDR_SIZE, &mut b) == 4).then(|| u32::from_le_bytes(b))
    }

    /// `virtio_snd_handle_pcm_start_stop()`.
    fn handle_pcm_start_stop(
        &mut self,
        vdev: &VirtIODevice,
        chain: &DescriptorChain,
        start: bool,
    ) -> u32 {
        let mut req = [0; PCM_HDR_SIZE];
        if read_at(vdev, chain, 0, &mut req) != PCM_HDR_SIZE {
            return S_BAD_MSG;
        }
        let stream_id = le32(&req, 4);
        let be = Arc::clone(&self.be);
        let Some(stream) = self.stream_mut(stream_id) else {
            error_report(&format!("Invalid stream id: {stream_id}"));
            return S_BAD_MSG;
        };
        stream.active = start;
        if stream.info.direction == D_OUTPUT {
            be.set_active_out(stream.voice_out, start);
        } else {
            be.set_active_in(stream.voice_in, start);
        }
        S_OK
    }

    /// `virtio_snd_handle_pcm_release()`.
    fn handle_pcm_release(&mut self, vdev: &mut VirtIODevice, chain: &DescriptorChain) -> u32 {
        let Some(stream_id) = Self::read_stream_id(vdev, chain) else {
            return S_BAD_MSG;
        };
        let Some(stream) = self.stream_mut(stream_id) else {
            error_report(&format!("already released stream {stream_id}"));
            vdev.error(&format!("already released stream {stream_id}"));
            return S_BAD_MSG;
        };
        // virtio-v1.2-csd01, 5.14.6.6.5.1: the device completes the pending I/O messages
        // before the request.
        flush(vdev, stream);
        S_OK
    }

    /// `process_cmd()`.
    fn process_cmd(&mut self, vdev: &mut VirtIODevice, chain: &DescriptorChain) {
        let mut hdr = [0; HDR_SIZE];
        if read_at(vdev, chain, 0, &mut hdr) != HDR_SIZE {
            // QEMU drops the element without answering.
            return;
        }
        let mut payload_size = 0;
        let code = match u32::from_le_bytes(hdr) {
            R_JACK_INFO | R_JACK_REMAP | R_CHMAP_INFO => S_NOT_SUPP,
            R_PCM_INFO => {
                let (code, size) = self.handle_pcm_info(vdev, chain);
                payload_size = size;
                code
            }
            R_PCM_START => self.handle_pcm_start_stop(vdev, chain, true),
            R_PCM_STOP => self.handle_pcm_start_stop(vdev, chain, false),
            R_PCM_SET_PARAMS => self.handle_pcm_set_params(vdev, chain),
            R_PCM_PREPARE => match Self::read_stream_id(vdev, chain) {
                Some(id) => self.pcm_prepare(id),
                None => S_BAD_MSG,
            },
            R_PCM_RELEASE => self.handle_pcm_release(vdev, chain),
            code => {
                error_report(&format!("virtio snd header not recognized: {code}"));
                S_BAD_MSG
            }
        };
        write_parts(vdev, chain, &[&code.to_le_bytes()]);
        vdev.push(VQ_CONTROL, chain, (HDR_SIZE + payload_size) as u32);
        vdev.notify(VQ_CONTROL);
    }

    /// `virtio_snd_handle_ctrl()`.
    fn handle_ctrl(&mut self, vdev: &mut VirtIODevice) {
        if !vdev.queue_ready(VQ_CONTROL) {
            return;
        }
        let mut cmdq = Vec::new();
        while let Some(chain) = vdev.pop(VQ_CONTROL) {
            cmdq.push(chain);
        }
        for chain in &cmdq {
            self.process_cmd(vdev, chain);
        }
    }

    /// `empty_invalid_queue()`.
    fn empty_invalid_queue(vdev: &mut VirtIODevice, vq: u16, invalid: &[DescriptorChain]) {
        for chain in invalid {
            write_parts(vdev, chain, &[&status_bytes(S_BAD_MSG, 0)]);
            vdev.push(vq, chain, STATUS_SIZE as u32);
        }
        // Notify vq about virtio_snd_pcm_status responses.
        vdev.notify(vq);
    }

    /// `virtio_snd_handle_tx_xfer()` and `virtio_snd_handle_rx_xfer()`: queues the I/O messages
    /// on their streams and answers the invalid ones.
    fn handle_xfer(&mut self, vdev: &mut VirtIODevice, vq: u16) {
        if !vdev.queue_ready(vq) {
            return;
        }
        let tx = vq == VQ_TX;
        let mut invalid = Vec::new();
        while let Some(chain) = vdev.pop(vq) {
            let mut hdr = [0; XFER_SIZE];
            let stream = if read_at(vdev, &chain, 0, &mut hdr) == XFER_SIZE {
                self.stream_mut(u32::from_le_bytes(hdr))
            } else {
                None
            };
            let in_size = chain.writable_len() as usize;
            let stream = stream.filter(|s| {
                if tx {
                    s.info.direction == D_OUTPUT
                } else {
                    s.info.direction == D_INPUT && in_size >= STATUS_SIZE
                }
            });
            let Some(stream) = stream else {
                invalid.push(chain);
                continue;
            };
            let buffer = if tx {
                let size = chain.readable_len() as usize - XFER_SIZE;
                stream.latency_bytes = stream.latency_bytes.wrapping_add(size as u32);
                PcmBuffer { chain, vq, size, offset: 0, populated: false, data: Vec::new() }
            } else {
                let size = in_size - STATUS_SIZE;
                PcmBuffer { chain, vq, size: 0, offset: 0, populated: false, data: vec![0; size] }
            };
            stream.queue.push_back(buffer);
        }
        if !invalid.is_empty() {
            Self::empty_invalid_queue(vdev, vq, &invalid);
        }
    }

    /// `virtio_snd_pcm_out_cb()`: the voice of stream `stream_id` takes `available` bytes.
    fn pcm_out_cb(&mut self, vdev: &mut VirtIODevice, stream_id: u32, available: usize) {
        let be = Arc::clone(&self.be);
        let Some(stream) = self.stream_mut(stream_id) else {
            return;
        };
        let mut available = available;
        while let Some(vq) = stream.queue.front().map(|b| b.vq) {
            if !vdev.queue_ready(vq) {
                return;
            }
            if !stream.active {
                // Stream has stopped, so do not write to the voice.
                return_tx_buffer(vdev, stream);
                continue;
            }
            if let Some(buffer) = stream.queue.front_mut().filter(|b| !b.populated) {
                let mut data = vec![0; buffer.size];
                read_at(vdev, &buffer.chain, XFER_SIZE, &mut data);
                buffer.data = data;
                buffer.populated = true;
            }
            while let Some(buffer) = stream.queue.front_mut() {
                let n = buffer.size.min(available);
                let data = &buffer.data[buffer.offset..buffer.offset + n];
                let size = be.write(stream.voice_out, data);
                if size == 0 {
                    // Break out of both loops.
                    available = 0;
                    break;
                }
                buffer.size -= size;
                buffer.offset += size;
                available -= size;
                let done = buffer.size < 1;
                stream.update_latency(size);
                if done {
                    return_tx_buffer(vdev, stream);
                    break;
                }
                if available == 0 {
                    break;
                }
            }
            if available == 0 {
                break;
            }
        }
    }

    /// `virtio_snd_pcm_in_cb()`: the voice of stream `stream_id` has `available` bytes.
    fn pcm_in_cb(&mut self, vdev: &mut VirtIODevice, stream_id: u32, available: usize) {
        let be = Arc::clone(&self.be);
        let Some(stream) = self.stream_mut(stream_id) else {
            return;
        };
        let period_bytes = stream.params.period_bytes as usize;
        let mut available = available;
        while let Some((vq, max_size)) =
            stream.queue.front().map(|b| (b.vq, b.chain.writable_len() as usize))
        {
            if !vdev.queue_ready(vq) {
                return;
            }
            if !stream.active {
                // Stream has stopped, so do not read from the voice.
                return_rx_buffer(vdev, stream);
                continue;
            }
            if max_size <= STATUS_SIZE {
                return_rx_buffer(vdev, stream);
                continue;
            }
            let max_size = max_size - STATUS_SIZE;
            while let Some(buffer) = stream.queue.front_mut() {
                if buffer.size >= max_size {
                    return_rx_buffer(vdev, stream);
                    break;
                }
                // size_t arithmetic, as in QEMU.
                let to_read = period_bytes
                    .wrapping_sub(buffer.size)
                    .min(available)
                    .min(max_size - buffer.size);
                let at = buffer.size;
                let size = be.read(stream.voice_in, &mut buffer.data[at..at + to_read]);
                if size == 0 {
                    available = 0;
                    break;
                }
                buffer.size += size;
                available -= size;
                if buffer.size >= period_bytes {
                    return_rx_buffer(vdev, stream);
                    break;
                }
                if available == 0 {
                    break;
                }
            }
            if available == 0 {
                break;
            }
        }
    }
}

impl VirtioDeviceClass for VirtioSnd {
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        self.conf.check()?;
        let n = self.conf.streams as usize;
        self.streams = (0..n).map(|_| None).collect();
        self.pcm_params = vec![PcmParams::default(); n];
        vdev.init(TYPE_VIRTIO_SND, VIRTIO_ID_SOUND, CONFIG_SIZE);
        for _ in [VQ_CONTROL, VQ_EVENT, VQ_TX, VQ_RX] {
            vdev.add_queue(QUEUE_SIZE)?;
        }
        for i in 0..self.conf.streams {
            // Neither can fail with the default parameters.
            self.set_pcm_params(vdev, i, PcmParams::default_params());
            self.pcm_prepare(i);
        }
        Ok(())
    }

    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        // virtio-v1.2-csd01, 5.14.3: no feature bits are defined.
        Ok(features | feature(VIRTIO_F_VERSION_1))
    }

    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let fields = [self.conf.jacks, self.conf.streams, self.conf.chmaps, 0];
        for (dst, v) in config.chunks_exact_mut(4).zip(fields) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        match queue {
            VQ_CONTROL => self.handle_ctrl(vdev),
            VQ_TX | VQ_RX => self.handle_xfer(vdev, queue),
            // The event queue is unimplemented.
            _ => {}
        }
    }

    fn legacy_features(&self) -> u64 {
        0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_properties_are_checked_in_qemus_order() {
        let ok = VirtioSndConf::default();
        assert_eq!(ok, VirtioSndConf { jacks: 0, streams: 2, chmaps: 0 });
        assert!(ok.check().is_ok());
        let msg = |c: VirtioSndConf| c.check().unwrap_err().message().to_string();
        let bad = VirtioSndConf { jacks: 9, streams: 0, chmaps: 19 };
        assert_eq!(msg(bad), "Invalid number of jacks: 9");
        assert_eq!(msg(VirtioSndConf { jacks: 8, ..bad }), "Invalid number of streams: 0");
        let c = VirtioSndConf { jacks: 8, streams: 11, chmaps: 18 };
        assert_eq!(msg(c), "Invalid number of streams: 11");
        let c = VirtioSndConf { jacks: 8, streams: 10, chmaps: 19 };
        assert_eq!(msg(c), "Invalid number of channel maps: 19");
    }

    #[test]
    fn pcm_info_has_the_wire_layout() {
        let info = PcmInfo {
            hda_fn_nid: 0x0403_0201,
            features: 0x0807_0605,
            formats: u64::from(SUPPORTED_FORMATS),
            rates: u64::from(SUPPORTED_RATES),
            direction: D_INPUT,
            channels_min: 1,
            channels_max: 16,
        };
        let b = info.to_bytes();
        assert_eq!(b[..8], [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(b[8..16], 0x000e_0078u64.to_le_bytes());
        assert_eq!(b[16..24], 0x3fffu64.to_le_bytes());
        assert_eq!(b[24..], [1, 1, 16, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn the_defaults_play_48khz_stereo_s16() {
        let a = PcmParams::default_params().audsettings();
        assert_eq!((a.freq, a.nchannels, a.fmt, a.big_endian), (48000, 2, AudioFormat::S16, false));
        let p = PcmParams { channels: 20, format: FMT_FLOAT, rate: 0, ..PcmParams::default() };
        let a = p.audsettings();
        assert_eq!((a.freq, a.nchannels, a.fmt), (5512, 16, AudioFormat::F32));
    }

    #[test]
    fn latency_never_goes_below_zero() {
        let mut s = PcmStream {
            info: PcmInfo::default(),
            params: PcmParams::default(),
            active: false,
            latency_bytes: 100,
            voice_out: None,
            voice_in: None,
            queue: VecDeque::new(),
        };
        s.update_latency(40);
        assert_eq!(s.latency_bytes, 60);
        s.update_latency(60);
        assert_eq!(s.latency_bytes, 0);
        s.update_latency(1);
        assert_eq!(s.latency_bytes, 0);
        assert_eq!(status_bytes(S_BAD_MSG, 0x1234), [1, 0x80, 0, 0, 0x34, 0x12, 0, 0]);
    }
}
