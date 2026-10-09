// SPDX-License-Identifier: GPL-2.0-or-later

//! The HDA codecs QEMU emulates, `hw/audio/hda-codec.c`: `hda-output` (a line out),
//! `hda-duplex` (line out and line in) and `hda-micro` (speaker and microphone).
//!
//! Each codec has a DAC and, apart from `hda-output`, an ADC. A 1 ms timer on the virtual
//! clock moves stream data between the controller's DMA engines and an 8 KiB ring at the rate
//! the stream format asks for, and the voice callbacks move the ring to and from the backend.
//! With `mixer=on`, the default, the amplifier verbs set the voice volume.
//!
//! The controller owns the codecs and calls them with its state locked, so a codec reaches
//! the controller through the closures it is handed rather than through a bus object.
//!
//! Migration of the codec state is not ported yet.

use std::sync::Arc;

use ruvm_audio::{AudSettings, AudioBackend, SwVoiceIn, SwVoiceOut, Volume};
use ruvm_hw_core::timer::{Clock, Timer, muldiv64};
use ruvm_qapi::types::AudioFormat;

/// `hda-output`.
pub const TYPE_HDA_OUTPUT: &str = "hda-output";
/// `hda-duplex`.
pub const TYPE_HDA_DUPLEX: &str = "hda-duplex";
/// `hda-micro`.
pub const TYPE_HDA_MICRO: &str = "hda-micro";

/// One of the three codec types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdaCodecKind {
    /// `hda-output`.
    Output,
    /// `hda-duplex`.
    Duplex,
    /// `hda-micro`.
    Micro,
}

impl HdaCodecKind {
    /// The kind a device type names.
    pub fn from_type(typename: &str) -> Option<HdaCodecKind> {
        match typename {
            TYPE_HDA_OUTPUT => Some(HdaCodecKind::Output),
            TYPE_HDA_DUPLEX => Some(HdaCodecKind::Duplex),
            TYPE_HDA_MICRO => Some(HdaCodecKind::Micro),
            _ => None,
        }
    }

    /// The device type.
    pub fn typename(self) -> &'static str {
        match self {
            HdaCodecKind::Output => TYPE_HDA_OUTPUT,
            HdaCodecKind::Duplex => TYPE_HDA_DUPLEX,
            HdaCodecKind::Micro => TYPE_HDA_MICRO,
        }
    }
}

// Parameters, `intel-hda-defs.h`.
const AC_PAR_VENDOR_ID: u32 = 0x00;
const AC_PAR_SUBSYSTEM_ID: u32 = 0x01;
const AC_PAR_REV_ID: u32 = 0x02;
const AC_PAR_NODE_COUNT: u32 = 0x04;
const AC_PAR_FUNCTION_TYPE: u32 = 0x05;
const AC_PAR_AUDIO_FG_CAP: u32 = 0x08;
const AC_PAR_AUDIO_WIDGET_CAP: u32 = 0x09;
const AC_PAR_PCM: u32 = 0x0a;
const AC_PAR_STREAM: u32 = 0x0b;
const AC_PAR_PIN_CAP: u32 = 0x0c;
const AC_PAR_AMP_IN_CAP: u32 = 0x0d;
const AC_PAR_CONNLIST_LEN: u32 = 0x0e;
const AC_PAR_POWER_STATE: u32 = 0x0f;
const AC_PAR_GPIO_CAP: u32 = 0x11;
const AC_PAR_AMP_OUT_CAP: u32 = 0x12;

// Verbs.
const AC_VERB_GET_STREAM_FORMAT: u32 = 0x0a00;
const AC_VERB_GET_AMP_GAIN_MUTE: u32 = 0x0b00;
const AC_VERB_PARAMETERS: u32 = 0x0f00;
const AC_VERB_GET_CONNECT_LIST: u32 = 0x0f02;
const AC_VERB_GET_SDI_SELECT: u32 = 0x0f04;
const AC_VERB_GET_POWER_STATE: u32 = 0x0f05;
const AC_VERB_GET_CONV: u32 = 0x0f06;
const AC_VERB_GET_PIN_WIDGET_CONTROL: u32 = 0x0f07;
const AC_VERB_GET_CONFIG_DEFAULT: u32 = 0x0f1c;
const AC_VERB_GET_SUBSYSTEM_ID: u32 = 0x0f20;
const AC_VERB_SET_STREAM_FORMAT: u32 = 0x200;
const AC_VERB_SET_AMP_GAIN_MUTE: u32 = 0x300;
const AC_VERB_SET_POWER_STATE: u32 = 0x705;
const AC_VERB_SET_CHANNEL_STREAMID: u32 = 0x706;
const AC_VERB_SET_PIN_WIDGET_CONTROL: u32 = 0x707;

const AC_GRP_AUDIO_FUNCTION: u32 = 0x01;

const AC_WID_AUD_OUT: u32 = 0;
const AC_WID_AUD_IN: u32 = 1;
const AC_WID_PIN: u32 = 4;

const AC_WCAP_STEREO: u32 = 1 << 0;
const AC_WCAP_IN_AMP: u32 = 1 << 1;
const AC_WCAP_OUT_AMP: u32 = 1 << 2;
const AC_WCAP_AMP_OVRD: u32 = 1 << 3;
const AC_WCAP_FORMAT_OVRD: u32 = 1 << 4;
const AC_WCAP_CONN_LIST: u32 = 1 << 8;
const AC_WCAP_TYPE: u32 = 0xf << 20;
const AC_WCAP_TYPE_SHIFT: u32 = 20;

const AC_SUPPCM_BITS_16: u32 = 1 << 17;
const AC_SUPFMT_PCM: u32 = 1 << 0;
const AC_PINCAP_OUT: u32 = 1 << 4;
const AC_PINCAP_IN: u32 = 1 << 5;
const AC_PINCTL_IN_EN: u32 = 1 << 5;
const AC_PINCTL_OUT_EN: u32 = 1 << 6;

const AC_AMPCAP_OFFSET_SHIFT: u32 = 0;
const AC_AMPCAP_NUM_STEPS_SHIFT: u32 = 8;
const AC_AMPCAP_STEP_SIZE_SHIFT: u32 = 16;
const AC_AMPCAP_MUTE: u32 = 1 << 31;

const AC_AMP_MUTE: u32 = 1 << 7;
const AC_AMP_GAIN: u32 = 0x7f;
const AC_AMP_GET_LEFT: u32 = 1 << 13;
const AC_AMP_SET_RIGHT: u32 = 1 << 12;
const AC_AMP_SET_LEFT: u32 = 1 << 13;

const AC_FMT_CHAN_MASK: u32 = 0x0f;
const AC_FMT_BITS_MASK: u32 = 7 << 4;
const AC_FMT_BITS_8: u32 = 0;
const AC_FMT_BITS_16: u32 = 1 << 4;
const AC_FMT_BITS_32: u32 = 4 << 4;
const AC_FMT_DIV_MASK: u32 = 7 << 8;
const AC_FMT_DIV_SHIFT: u32 = 8;
const AC_FMT_MULT_MASK: u32 = 7 << 11;
const AC_FMT_MULT_SHIFT: u32 = 11;
const AC_FMT_BASE_44K: u32 = 1 << 14;
const AC_FMT_TYPE_NON_PCM: u32 = 1 << 15;

// The pin default configuration.
const AC_DEFCFG_COLOR_SHIFT: u32 = 12;
const AC_DEFCFG_CONN_TYPE_SHIFT: u32 = 16;
const AC_DEFCFG_DEVICE_SHIFT: u32 = 20;
const AC_DEFCFG_PORT_CONN_SHIFT: u32 = 30;
const AC_JACK_PORT_COMPLEX: u32 = 0;
const AC_JACK_LINE_OUT: u32 = 0;
const AC_JACK_SPEAKER: u32 = 1;
const AC_JACK_LINE_IN: u32 = 8;
const AC_JACK_MIC_IN: u32 = 0xa;
const AC_JACK_CONN_UNKNOWN: u32 = 0;
const AC_JACK_COLOR_GREEN: u32 = 4;
const AC_JACK_COLOR_RED: u32 = 5;

const QEMU_HDA_ID_VENDOR: u32 = 0x1af4;
/// 16 bit samples, 16 to 96 kHz.
const QEMU_HDA_PCM_FORMATS: u32 = AC_SUPPCM_BITS_16 | 0x1fc;
const QEMU_HDA_AMP_NONE: u32 = 0;
const QEMU_HDA_AMP_STEPS: u32 = 0x4a;

/// `HDA_TIMER_TICKS`, one millisecond.
const HDA_TIMER_TICKS: i64 = 1_000_000;
/// The size of a stream's ring. It must be a power of two.
const B_SIZE: usize = 8192;
const B_MASK: i64 = B_SIZE as i64 - 1;

/// `desc_node`.
struct Node {
    nid: u32,
    name: &'static str,
    params: Vec<(u32, u32)>,
    config: u32,
    pinctl: u32,
    conn: &'static [u32],
    stindex: usize,
}

impl Node {
    fn new(nid: u32, name: &'static str, params: Vec<(u32, u32)>) -> Node {
        Node { nid, name, params, config: 0, pinctl: 0, conn: &[], stindex: 0 }
    }

    /// `hda_codec_find_param()`.
    fn param(&self, id: u32) -> Option<u32> {
        self.params.iter().find(|p| p.0 == id).map(|p| p.1)
    }
}

/// `desc_codec`.
struct Desc {
    iid: u32,
    nodes: Vec<Node>,
}

fn pin_config(device: u32, color: u32, misc: u32) -> u32 {
    (AC_JACK_PORT_COMPLEX << AC_DEFCFG_PORT_CONN_SHIFT)
        | (device << AC_DEFCFG_DEVICE_SHIFT)
        | (AC_JACK_CONN_UNKNOWN << AC_DEFCFG_CONN_TYPE_SHIFT)
        | (color << AC_DEFCFG_COLOR_SHIFT)
        | misc
}

/// The tables of `hda-codec-common.h`, which QEMU builds twice: with the mixer emulation
/// (`mixemu`) the IDs end in 2 and the amplifiers have 0x4a steps, without it (`nomixemu`)
/// the IDs end in 1 and there are no amplifiers.
fn desc(kind: HdaCodecKind, mixer: bool) -> Desc {
    let low = match kind {
        HdaCodecKind::Output => 0x10,
        HdaCodecKind::Duplex => 0x20,
        HdaCodecKind::Micro => 0x30,
    } + if mixer { 2 } else { 1 };
    let iid = (QEMU_HDA_ID_VENDOR << 16) | low;
    let amp_caps = if mixer {
        AC_AMPCAP_MUTE
            | (QEMU_HDA_AMP_STEPS << AC_AMPCAP_OFFSET_SHIFT)
            | (QEMU_HDA_AMP_STEPS << AC_AMPCAP_NUM_STEPS_SHIFT)
            | (3 << AC_AMPCAP_STEP_SIZE_SHIFT)
    } else {
        QEMU_HDA_AMP_NONE
    };
    let root = vec![
        (AC_PAR_VENDOR_ID, iid),
        (AC_PAR_SUBSYSTEM_ID, iid),
        (AC_PAR_REV_ID, 0x0010_0101),
        (AC_PAR_NODE_COUNT, 0x0001_0001),
    ];
    let func = vec![
        (AC_PAR_FUNCTION_TYPE, AC_GRP_AUDIO_FUNCTION),
        (AC_PAR_SUBSYSTEM_ID, iid),
        (AC_PAR_NODE_COUNT, if kind == HdaCodecKind::Output { 0x0002_0002 } else { 0x0002_0004 }),
        (AC_PAR_PCM, QEMU_HDA_PCM_FORMATS),
        (AC_PAR_STREAM, AC_SUPFMT_PCM),
        (AC_PAR_AMP_IN_CAP, QEMU_HDA_AMP_NONE),
        (AC_PAR_AMP_OUT_CAP, QEMU_HDA_AMP_NONE),
        (AC_PAR_GPIO_CAP, 0),
        (AC_PAR_AUDIO_FG_CAP, 0x0000_0808),
        (AC_PAR_POWER_STATE, 0),
    ];
    let dac = vec![
        (
            AC_PAR_AUDIO_WIDGET_CAP,
            (AC_WID_AUD_OUT << AC_WCAP_TYPE_SHIFT)
                | AC_WCAP_FORMAT_OVRD
                | AC_WCAP_AMP_OVRD
                | AC_WCAP_OUT_AMP
                | AC_WCAP_STEREO,
        ),
        (AC_PAR_PCM, QEMU_HDA_PCM_FORMATS),
        (AC_PAR_STREAM, AC_SUPFMT_PCM),
        (AC_PAR_AMP_IN_CAP, QEMU_HDA_AMP_NONE),
        (AC_PAR_AMP_OUT_CAP, amp_caps),
    ];
    let lineout = vec![
        (
            AC_PAR_AUDIO_WIDGET_CAP,
            (AC_WID_PIN << AC_WCAP_TYPE_SHIFT) | AC_WCAP_CONN_LIST | AC_WCAP_STEREO,
        ),
        (AC_PAR_PIN_CAP, AC_PINCAP_OUT),
        (AC_PAR_CONNLIST_LEN, 1),
        (AC_PAR_AMP_IN_CAP, QEMU_HDA_AMP_NONE),
        (AC_PAR_AMP_OUT_CAP, QEMU_HDA_AMP_NONE),
    ];
    let out_device = if kind == HdaCodecKind::Micro { AC_JACK_SPEAKER } else { AC_JACK_LINE_OUT };
    let mut nodes = vec![
        Node::new(0, "root", root),
        Node::new(1, "func", func),
        Node::new(2, "dac", dac),
        Node {
            config: pin_config(out_device, AC_JACK_COLOR_GREEN, 0x10),
            pinctl: AC_PINCTL_OUT_EN,
            conn: &[2],
            ..Node::new(3, "out", lineout)
        },
    ];
    if kind != HdaCodecKind::Output {
        let adc = vec![
            (
                AC_PAR_AUDIO_WIDGET_CAP,
                (AC_WID_AUD_IN << AC_WCAP_TYPE_SHIFT)
                    | AC_WCAP_CONN_LIST
                    | AC_WCAP_FORMAT_OVRD
                    | AC_WCAP_AMP_OVRD
                    | AC_WCAP_IN_AMP
                    | AC_WCAP_STEREO,
            ),
            (AC_PAR_CONNLIST_LEN, 1),
            (AC_PAR_PCM, QEMU_HDA_PCM_FORMATS),
            (AC_PAR_STREAM, AC_SUPFMT_PCM),
            (AC_PAR_AMP_IN_CAP, amp_caps),
            (AC_PAR_AMP_OUT_CAP, QEMU_HDA_AMP_NONE),
        ];
        let linein = vec![
            (AC_PAR_AUDIO_WIDGET_CAP, (AC_WID_PIN << AC_WCAP_TYPE_SHIFT) | AC_WCAP_STEREO),
            (AC_PAR_PIN_CAP, AC_PINCAP_IN),
            (AC_PAR_AMP_IN_CAP, QEMU_HDA_AMP_NONE),
            (AC_PAR_AMP_OUT_CAP, QEMU_HDA_AMP_NONE),
        ];
        let in_device = if kind == HdaCodecKind::Micro { AC_JACK_MIC_IN } else { AC_JACK_LINE_IN };
        nodes.push(Node { stindex: 1, conn: &[5], ..Node::new(4, "adc", adc) });
        nodes.push(Node {
            config: pin_config(in_device, AC_JACK_COLOR_RED, 0x20),
            pinctl: AC_PINCTL_IN_EN,
            ..Node::new(5, "in", linein)
        });
    }
    Desc { iid, nodes }
}

/// `hda_codec_parse_fmt()`: a stream format word into `as_`. Bit depths other than 8, 16
/// and 32 leave the sample format alone, and a non-PCM format leaves everything alone.
fn parse_fmt(format: u32, as_: &mut AudSettings) {
    if format & AC_FMT_TYPE_NON_PCM != 0 {
        return;
    }
    as_.freq = if format & AC_FMT_BASE_44K != 0 { 44100 } else { 48000 };
    match (format & AC_FMT_MULT_MASK) >> AC_FMT_MULT_SHIFT {
        1 => as_.freq *= 2,
        2 => as_.freq *= 3,
        3 => as_.freq *= 4,
        _ => {}
    }
    let div = (format & AC_FMT_DIV_MASK) >> AC_FMT_DIV_SHIFT;
    as_.freq /= div as i32 + 1;
    match format & AC_FMT_BITS_MASK {
        AC_FMT_BITS_8 => as_.fmt = AudioFormat::S8,
        AC_FMT_BITS_16 => as_.fmt = AudioFormat::S16,
        AC_FMT_BITS_32 => as_.fmt = AudioFormat::S32,
        _ => {}
    }
    as_.nchannels = (format & AC_FMT_CHAN_MASK) as i32 + 1;
}

/// What a codec's timers and voices ask for. The controller locks its state and hands the
/// event back to the codec through [`HdaAudio::timer`] or [`HdaAudio::voice`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum Event {
    /// The timer of stream `0` fired.
    Timer(usize),
    /// The voice of stream `0` can take or give `1` bytes.
    Voice(usize, usize),
}

/// Delivers an [`Event`] to the controller the codec sits on.
pub(crate) type Hook = Arc<dyn Fn(Event) + Send + Sync>;

/// The controller's side of a transfer, `hda_codec_xfer()`: moves the buffer to or from the
/// DMA engine running stream `stnr` and says whether there was one.
pub(crate) type Xfer<'a> = dyn FnMut(u32, bool, &mut [u8]) -> bool + 'a;

/// `HDAAudioStream`.
struct Stream {
    /// The index of the converter node in the descriptor, `None` for an unused slot.
    node: Option<usize>,
    output: bool,
    running: bool,
    stream: u32,
    channel: u32,
    format: u32,
    gain_left: u32,
    gain_right: u32,
    mute_left: bool,
    mute_right: bool,
    as_: AudSettings,
    voice_out: Option<SwVoiceOut>,
    voice_in: Option<SwVoiceIn>,
    buf: Box<[u8; B_SIZE]>,
    rpos: i64,
    wpos: i64,
    buft: Option<Timer>,
    buft_start: i64,
}

impl Stream {
    fn new() -> Stream {
        Stream {
            node: None,
            output: false,
            running: false,
            stream: 0,
            channel: 0,
            format: 0,
            gain_left: 0,
            gain_right: 0,
            mute_left: false,
            mute_right: false,
            as_: AudSettings { freq: 0, nchannels: 0, fmt: AudioFormat::U8, big_endian: false },
            voice_out: None,
            voice_in: None,
            buf: Box::new([0; B_SIZE]),
            rpos: 0,
            wpos: 0,
            buft: None,
            buft_start: 0,
        }
    }

    /// `hda_bytes_per_second()`. It counts two bytes a sample whatever the format, as QEMU
    /// does.
    fn bytes_per_second(&self) -> u32 {
        (2 * self.as_.nchannels as u32).wrapping_mul(self.as_.freq as u32)
    }

    /// `hda_timer_sync_adjust()`: nudges the timer base so the ring stays half full.
    fn sync_adjust(&mut self, target_pos: i64) {
        let limit = B_SIZE as i64 / 8;
        let mut corr = 0;
        if target_pos > limit {
            corr = HDA_TIMER_TICKS;
        }
        if target_pos < -limit {
            corr = -HDA_TIMER_TICKS;
        }
        if target_pos < -(2 * limit) {
            corr = -(4 * HDA_TIMER_TICKS);
        }
        self.buft_start += corr;
    }

    fn rearm(&self, now: i64) {
        if self.running {
            if let Some(t) = &self.buft {
                t.modify_anticipate(now + HDA_TIMER_TICKS);
            }
        }
    }
}

/// `HDAAudioState`: one codec on an HDA bus.
pub(crate) struct HdaAudio {
    pub(crate) cad: u32,
    be: Arc<AudioBackend>,
    clock: Arc<Clock>,
    hook: Hook,
    desc: Desc,
    st: [Stream; 4],
    running_real: [bool; 2 * 16],
    mixer: bool,
}

impl HdaAudio {
    /// `hda_audio_init()`: a stream for every converter node, each opened with the default
    /// format, 48 kHz 16 bit stereo. Output amplifiers start at full gain.
    pub(crate) fn new(
        kind: HdaCodecKind,
        cad: u32,
        mixer: bool,
        be: Arc<AudioBackend>,
        clock: Arc<Clock>,
        hook: Hook,
    ) -> HdaAudio {
        let mut a = HdaAudio {
            cad,
            be,
            clock,
            hook,
            desc: desc(kind, mixer),
            st: std::array::from_fn(|_| Stream::new()),
            running_real: [false; 32],
            mixer,
        };
        for i in 0..a.desc.nodes.len() {
            let node = &a.desc.nodes[i];
            let Some(wcap) = node.param(AC_PAR_AUDIO_WIDGET_CAP) else { continue };
            let ty = (wcap & AC_WCAP_TYPE) >> AC_WCAP_TYPE_SHIFT;
            if ty != AC_WID_AUD_OUT && ty != AC_WID_AUD_IN {
                continue;
            }
            let si = node.stindex;
            let hook = a.hook.clone();
            let timer = a.clock.new_timer(move || hook(Event::Timer(si)));
            let st = &mut a.st[si];
            st.node = Some(i);
            st.output = ty == AC_WID_AUD_OUT;
            if st.output {
                st.gain_left = QEMU_HDA_AMP_STEPS;
                st.gain_right = QEMU_HDA_AMP_STEPS;
            }
            st.buft = Some(timer);
            st.format = AC_FMT_BITS_16 | 1;
            parse_fmt(st.format, &mut st.as_);
            a.setup(si);
        }
        a
    }

    fn node_name(&self, si: usize) -> &'static str {
        self.st[si].node.map_or("?", |n| self.desc.nodes[n].name)
    }

    /// `hda_audio_setup()`: (re)opens the voice with the stream's format.
    fn setup(&mut self, si: usize) {
        if self.st[si].node.is_none() {
            return;
        }
        let name = self.node_name(si);
        let hook = self.hook.clone();
        let cb = Arc::new(move |avail: usize| hook(Event::Voice(si, avail)));
        let st = &mut self.st[si];
        if let Some(t) = &st.buft {
            t.del();
        }
        if st.output {
            st.voice_out = self.be.open_out(st.voice_out, name, cb, &st.as_);
        } else {
            st.voice_in = self.be.open_in(st.voice_in, name, cb, &st.as_);
        }
    }

    /// `hda_audio_set_running()`.
    fn set_running(&mut self, si: usize, running: bool) {
        let now = self.clock.get_ns();
        let st = &mut self.st[si];
        if st.node.is_none() || st.running == running {
            return;
        }
        st.running = running;
        if running {
            st.rpos = 0;
            st.wpos = 0;
            st.buft_start = now;
            st.rearm(now);
        } else if let Some(t) = &st.buft {
            t.del();
        }
        if st.output {
            self.be.set_active_out(st.voice_out, st.running);
        } else {
            self.be.set_active_in(st.voice_in, st.running);
        }
    }

    /// `hda_audio_set_amp()`. QEMU passes the scaled gain as a `uint8_t`, so gains above the
    /// 0x4a steps wrap; that is kept.
    fn set_amp(&self, si: usize) {
        let st = &self.st[si];
        if st.node.is_none() {
            return;
        }
        let muted = st.mute_left && st.mute_right;
        let left = if st.mute_left { 0 } else { st.gain_left };
        let right = if st.mute_right { 0 } else { st.gain_right };
        let left = left * 255 / QEMU_HDA_AMP_STEPS;
        let right = right * 255 / QEMU_HDA_AMP_STEPS;
        if !self.mixer {
            return;
        }
        let mut vol = Volume { mute: muted, channels: 2, vol: [0; 16] };
        vol.vol[0] = left as u8;
        vol.vol[1] = right as u8;
        if st.output {
            self.be.set_volume_out(st.voice_out, &vol);
        } else {
            self.be.set_volume_in(st.voice_in, &vol);
        }
    }

    /// `hda_audio_command()`: runs a verb and returns the solicited response. Whatever the
    /// codec does not handle answers 0.
    pub(crate) fn command(&mut self, nid: u32, data: u32) -> u32 {
        let (verb, payload) = if data & 0x70000 == 0x70000 {
            // 12/8 id/payload
            ((data >> 8) & 0xfff, data & 0x00ff)
        } else {
            // 4/16 id/payload
            ((data >> 8) & 0xf00, data & 0xffff)
        };
        let Some(ni) = self.desc.nodes.iter().position(|n| n.nid == nid) else { return 0 };
        self.verb(ni, verb, payload).unwrap_or(0)
    }

    fn verb(&mut self, ni: usize, verb: u32, mut payload: u32) -> Option<u32> {
        let node = &self.desc.nodes[ni];
        let si = node.stindex;
        match verb {
            AC_VERB_PARAMETERS => node.param(payload),
            AC_VERB_GET_SUBSYSTEM_ID => Some(self.desc.iid),
            AC_VERB_GET_CONNECT_LIST => {
                let count = node.param(AC_PAR_CONNLIST_LEN).unwrap_or(0);
                let mut response = 0;
                let mut shift = 0;
                while payload < count && shift < 32 {
                    response |= node.conn[payload as usize] << shift;
                    payload += 1;
                    shift += 8;
                }
                Some(response)
            }
            AC_VERB_GET_CONFIG_DEFAULT => Some(node.config),
            AC_VERB_GET_PIN_WIDGET_CONTROL => Some(node.pinctl),
            AC_VERB_SET_PIN_WIDGET_CONTROL => Some(0),
            // The converter verbs act on the node's stream. Every node but the ADC has
            // stream index 0, so on a pin or the root they reach the DAC, as in QEMU.
            AC_VERB_SET_CHANNEL_STREAMID => {
                self.st[si].node?;
                self.set_running(si, false);
                let st = &mut self.st[si];
                st.stream = (payload >> 4) & 0x0f;
                st.channel = payload & 0x0f;
                let running = self.running_real[usize::from(st.output) * 16 + st.stream as usize];
                self.set_running(si, running);
                Some(0)
            }
            AC_VERB_GET_CONV => {
                let st = &self.st[si];
                st.node?;
                Some(st.stream << 4 | st.channel)
            }
            AC_VERB_SET_STREAM_FORMAT => {
                let st = &mut self.st[si];
                st.node?;
                st.format = payload;
                parse_fmt(st.format, &mut st.as_);
                self.setup(si);
                Some(0)
            }
            AC_VERB_GET_STREAM_FORMAT => {
                let st = &self.st[si];
                st.node?;
                Some(st.format)
            }
            AC_VERB_GET_AMP_GAIN_MUTE => {
                let st = &self.st[si];
                st.node?;
                let mute = |m: bool| if m { AC_AMP_MUTE } else { 0 };
                Some(if payload & AC_AMP_GET_LEFT != 0 {
                    st.gain_left | mute(st.mute_left)
                } else {
                    st.gain_right | mute(st.mute_right)
                })
            }
            AC_VERB_SET_AMP_GAIN_MUTE => {
                let st = &mut self.st[si];
                st.node?;
                if payload & AC_AMP_SET_LEFT != 0 {
                    st.gain_left = payload & AC_AMP_GAIN;
                    st.mute_left = payload & AC_AMP_MUTE != 0;
                }
                if payload & AC_AMP_SET_RIGHT != 0 {
                    st.gain_right = payload & AC_AMP_GAIN;
                    st.mute_right = payload & AC_AMP_MUTE != 0;
                }
                self.set_amp(si);
                Some(0)
            }
            // Not supported.
            AC_VERB_SET_POWER_STATE | AC_VERB_GET_POWER_STATE | AC_VERB_GET_SDI_SELECT => Some(0),
            _ => None,
        }
    }

    /// `hda_audio_stream()`: the controller started or stopped stream `stnr`.
    pub(crate) fn stream(&mut self, stnr: u32, running: bool, output: bool) {
        self.running_real[usize::from(output) * 16 + stnr as usize] = running;
        for si in 0..self.st.len() {
            let st = &self.st[si];
            if st.node.is_none() || st.output != output || st.stream != stnr {
                continue;
            }
            self.set_running(si, running);
        }
    }

    /// `hda_audio_reset()`.
    pub(crate) fn reset(&mut self) {
        for si in 0..self.st.len() {
            if self.st[si].node.is_some() {
                self.set_running(si, false);
            }
        }
    }

    /// The stream timer, `hda_audio_output_timer()` or `hda_audio_input_timer()`: moves what
    /// the elapsed time calls for between the ring and the DMA engine.
    pub(crate) fn timer(&mut self, si: usize, xfer: &mut Xfer<'_>) {
        let now = self.clock.get_ns();
        let st = &mut self.st[si];
        let uptime = now - st.buft_start;
        if uptime > 0 {
            let wanted = muldiv64(uptime as u64, st.bytes_per_second(), 1_000_000_000) as i64;
            // Clip to frames.
            let wanted = wanted & -4;
            if st.output && wanted > st.wpos {
                let mut to_transfer = (B_SIZE as i64 - (st.wpos - st.rpos)).min(wanted - st.wpos);
                while to_transfer > 0 {
                    let start = (st.wpos & B_MASK) as usize;
                    let chunk = (B_SIZE - start).min(to_transfer as usize);
                    if !xfer(st.stream, true, &mut st.buf[start..start + chunk]) {
                        break;
                    }
                    to_transfer -= chunk as i64;
                    st.wpos += chunk as i64;
                }
            } else if !st.output && wanted > st.rpos {
                let mut to_transfer = (st.wpos - st.rpos).min(wanted - st.rpos);
                while to_transfer > 0 {
                    let start = (st.rpos & B_MASK) as usize;
                    let chunk = (B_SIZE - start).min(to_transfer as usize);
                    if !xfer(st.stream, false, &mut st.buf[start..start + chunk]) {
                        break;
                    }
                    to_transfer -= chunk as i64;
                    st.rpos += chunk as i64;
                }
            }
        }
        st.rearm(now);
    }

    /// The voice callback, `hda_audio_output_cb()` or `hda_audio_input_cb()`: moves up to
    /// `avail` bytes between the ring and the backend.
    pub(crate) fn voice(&mut self, si: usize, avail: usize) {
        let now = self.clock.get_ns();
        let be = &self.be;
        let st = &mut self.st[si];
        let avail = avail as i64;
        if st.output {
            let (wpos, mut rpos) = (st.wpos, st.rpos);
            if wpos - rpos == B_SIZE as i64 {
                // Drop the buffer and start the timer adjustment over.
                st.rpos = 0;
                st.wpos = 0;
                st.buft_start = now;
                return;
            }
            let mut to_transfer = (wpos - rpos).min(avail);
            while to_transfer > 0 {
                let start = (rpos & B_MASK) as usize;
                let chunk = (B_SIZE - start).min(to_transfer as usize);
                let written = be.write(st.voice_out, &st.buf[start..start + chunk]);
                rpos += written as i64;
                to_transfer -= written as i64;
                st.rpos += written as i64;
                if chunk != written {
                    break;
                }
            }
            st.sync_adjust((wpos - rpos) - (B_SIZE as i64 >> 1));
        } else {
            let (mut wpos, rpos) = (st.wpos, st.rpos);
            let mut to_transfer = (B_SIZE as i64 - (wpos - rpos)).min(avail);
            while to_transfer > 0 {
                let start = (wpos & B_MASK) as usize;
                let chunk = (B_SIZE - start).min(to_transfer as usize);
                let read = be.read(st.voice_in, &mut st.buf[start..start + chunk]);
                wpos += read as i64;
                to_transfer -= read as i64;
                st.wpos += read as i64;
                if chunk != read {
                    break;
                }
            }
            st.sync_adjust(-((wpos - rpos) - (B_SIZE as i64 >> 1)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_parse_like_qemu() {
        let mut as_ =
            AudSettings { freq: 0, nchannels: 0, fmt: AudioFormat::U8, big_endian: false };
        parse_fmt(0x11, &mut as_);
        assert_eq!((as_.freq, as_.nchannels, as_.fmt), (48000, 2, AudioFormat::S16));
        // 44.1 kHz times 2 divided by 3, 32 bit, mono.
        parse_fmt(AC_FMT_BASE_44K | (1 << 11) | (2 << 8) | AC_FMT_BITS_32, &mut as_);
        assert_eq!((as_.freq, as_.nchannels, as_.fmt), (29400, 1, AudioFormat::S32));
        // 24 bit keeps the sample format; non-PCM changes nothing.
        parse_fmt((3 << 4) | 1, &mut as_);
        assert_eq!((as_.freq, as_.nchannels, as_.fmt), (48000, 2, AudioFormat::S32));
        parse_fmt(AC_FMT_TYPE_NON_PCM | 7, &mut as_);
        assert_eq!(as_.nchannels, 2);
    }

    #[test]
    fn descriptors_match_qemu() {
        let d = desc(HdaCodecKind::Duplex, true);
        assert_eq!(d.iid, 0x1af4_0022);
        assert_eq!(d.nodes.len(), 6);
        assert_eq!(d.nodes[2].param(AC_PAR_AUDIO_WIDGET_CAP), Some(0x1d));
        assert_eq!(d.nodes[2].param(AC_PAR_AMP_OUT_CAP), Some(0x8003_4a4a));
        assert_eq!(d.nodes[4].param(AC_PAR_AUDIO_WIDGET_CAP), Some(0x0010_011b));
        assert_eq!(d.nodes[3].config, 0x4010);
        assert_eq!(d.nodes[5].config, 0x0080_5020);
        let d = desc(HdaCodecKind::Micro, false);
        assert_eq!(d.iid, 0x1af4_0031);
        assert_eq!(d.nodes[2].param(AC_PAR_AMP_OUT_CAP), Some(0));
        assert_eq!(d.nodes[3].config, 0x0010_4010);
        assert_eq!(d.nodes[5].config, 0x00a0_5020);
        let d = desc(HdaCodecKind::Output, true);
        assert_eq!(d.nodes.len(), 4);
        assert_eq!(d.nodes[1].param(AC_PAR_NODE_COUNT), Some(0x0002_0002));
    }
}
