// SPDX-License-Identifier: GPL-2.0-or-later

//! One VNC client, the `VncState` half of QEMU's ui/vnc.c and the job half of ui/vnc-jobs.c.
//!
//! The protocol is a chain of handlers, each waiting for a number of bytes as
//! `vnc_read_when()` sets them up: the version, the security type, the VNC authentication
//! response, ClientInit and then client messages for good. A handler that needs more bytes
//! than it was given returns the count it wants and is called again once they are there.
//!
//! Key and pointer messages update the display's keyboard state and the client's pointer
//! state here, and queue what goes to the input layer as [`Deferred`] work. Cut text messages
//! are read in full and dropped.

use std::sync::Arc;

use ruvm_base::report::error_report;
use ruvm_qapi::types::{
    VncBasicInfo, VncClientInfo, VncConnectedArg, VncDisconnectedArg, VncInitializedArg,
};

use crate::input::key_number_to_linux;
use crate::kbd_state::QKbdModifier;
use crate::keymaps::{self, SCANCODE_KEYMASK, keycode_is_keypad, keysym_is_numlock};

use super::net::{AddrInfo, ClientIo};
use super::pixels::{PixelWriter, VncPixelFormat};
use super::tight::Tight;
use super::zlib::ZStream;
use super::{
    Auth, DIRTY_BPL, DIRTY_PIXELS_PER_BIT, Deferred, DirtyMap, ENCODING_ALPHA_CURSOR,
    ENCODING_AUDIO, ENCODING_COMPRESSLEVEL0, ENCODING_DESKTOP_RESIZE_EXT, ENCODING_DESKTOPRESIZE,
    ENCODING_EXT_KEY_EVENT, ENCODING_HEXTILE, ENCODING_LED_STATE, ENCODING_POINTER_TYPE_CHANGE,
    ENCODING_QUALITYLEVEL0, ENCODING_RAW, ENCODING_RICH_CURSOR, ENCODING_TIGHT, ENCODING_WMVI,
    ENCODING_XVP, ENCODING_ZLIB, Fb, Motion, REFRESH_INTERVAL_BASE, SharePolicy, VdState,
    VncDisplay, VncEvent, auth, framebuffer_update, hextile, raw, set_area_dirty, tight, zlib,
};

/// Server to client message types.
const MSG_SERVER_FRAMEBUFFER_UPDATE: u8 = 0;
const MSG_SERVER_SET_COLOUR_MAP_ENTRIES: u8 = 1;
const MSG_SERVER_XVP: u8 = 250;

/// Client to server message types.
const MSG_CLIENT_SET_PIXEL_FORMAT: u8 = 0;
const MSG_CLIENT_SET_ENCODINGS: u8 = 2;
const MSG_CLIENT_FRAMEBUFFER_UPDATE_REQUEST: u8 = 3;
const MSG_CLIENT_KEY_EVENT: u8 = 4;
const MSG_CLIENT_POINTER_EVENT: u8 = 5;
const MSG_CLIENT_CUT_TEXT: u8 = 6;
const MSG_CLIENT_XVP: u8 = 250;
const MSG_CLIENT_SET_DESKTOP_SIZE: u8 = 251;
const MSG_CLIENT_QEMU: u8 = 255;
const MSG_CLIENT_QEMU_EXT_KEY_EVENT: u8 = 0;

const XVP_CODE_FAIL: u8 = 0;
const XVP_CODE_INIT: u8 = 1;
const XVP_ACTION_SHUTDOWN: u8 = 2;
const XVP_ACTION_RESET: u8 = 4;

const KEY_1: u32 = 2;
const KEY_9: u32 = 10;
const KEY_CAPSLOCK: u32 = 58;
const KEY_NUMLOCK: u32 = 69;

/// `VNC_AUTH_INVALID`.
const AUTH_INVALID: u32 = 0;

/// `VNC_THROTTLE_OUTPUT_LIMIT_SCALE`.
const THROTTLE_OUTPUT_LIMIT_SCALE: usize = 5;

/// The `VNC_FEATURE_*` bits.
const FEATURE_RESIZE: u32 = 1 << 0;
const FEATURE_HEXTILE: u32 = 1 << 1;
const FEATURE_POINTER_TYPE_CHANGE: u32 = 1 << 2;
const FEATURE_WMVI: u32 = 1 << 3;
const FEATURE_TIGHT: u32 = 1 << 4;
const FEATURE_ZLIB: u32 = 1 << 5;
const FEATURE_RICH_CURSOR: u32 = 1 << 8;
const FEATURE_LED_STATE: u32 = 1 << 10;
const FEATURE_XVP: u32 = 1 << 11;
const FEATURE_RESIZE_EXT: u32 = 1 << 13;
const FEATURE_ALPHA_CURSOR: u32 = 1 << 14;

/// `VncShareMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShareMode {
    /// The zero a new `VncState` starts with, before `vnc_connect()` sets connecting.
    Unset,
    Connecting,
    Shared,
    Exclusive,
    Disconnected,
}

/// `VncStateUpdate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Update {
    None,
    Incremental,
    Force,
}

/// What the next `vnc_read_when()` bytes go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handler {
    Version,
    Auth,
    AuthVnc,
    Init,
    Msg,
}

/// Where a client is, the `VncClientInfo` QEMU caches when it connects.
pub(crate) struct ClientAddr(AddrInfo);

impl ClientAddr {
    pub(crate) fn basic_info(&self) -> VncBasicInfo {
        VncBasicInfo {
            host: self.0.host.clone(),
            service: self.0.service.clone(),
            family: self.0.family,
            websocket: false,
        }
    }

    pub(crate) fn client_info(&self) -> VncClientInfo {
        VncClientInfo {
            host: self.0.host.clone(),
            service: self.0.service.clone(),
            family: self.0.family,
            websocket: false,
            x509_dname: None,
            sasl_username: None,
        }
    }
}

/// `VncState`.
pub(crate) struct Client {
    pub(crate) id: u64,
    io: Arc<ClientIo>,
    pub(crate) info: ClientAddr,
    /// `vs->output` not yet handed to the writer.
    out: Vec<u8>,
    input: Vec<u8>,
    expect: usize,
    handler: Handler,
    minor: i32,
    auth: Auth,
    challenge: [u8; auth::CHALLENGE_SIZE],
    pub(crate) share_mode: ShareMode,
    pub(crate) update: Update,
    pub(crate) job_update: Update,
    pub(crate) dirty: DirtyMap,
    has_dirty: i64,
    client_width: usize,
    client_height: usize,
    pw: PixelWriter,
    features: u32,
    encoding: i32,
    absolute: i32,
    last_x: i32,
    last_y: i32,
    last_bmask: u8,
    tight: Tight,
    zlib: Option<ZStream>,
    throttle_output_offset: usize,
    disconnecting: bool,
}

impl Client {
    fn has_feature(&self, feature: u32) -> bool {
        self.features & feature != 0
    }

    /// `vnc_write()`.
    fn write(&mut self, data: &[u8]) {
        if self.disconnecting {
            return;
        }
        // The guard against a client that stopped reading. Updates are throttled well before
        // this, so only a pile of pseudo encodings gets here.
        if self.throttle_output_offset != 0
            && (self.io.pending() + self.out.len()) / THROTTLE_OUTPUT_LIMIT_SCALE
                >= self.throttle_output_offset
        {
            self.disconnect_start();
            return;
        }
        self.out.extend_from_slice(data);
    }

    fn write_u8(&mut self, v: u8) {
        self.write(&[v]);
    }

    fn write_u16(&mut self, v: u16) {
        self.write(&v.to_be_bytes());
    }

    fn write_u32(&mut self, v: u32) {
        self.write(&v.to_be_bytes());
    }

    /// `vnc_framebuffer_update()` into the output.
    fn write_rect_header(&mut self, x: usize, y: usize, w: usize, h: usize, encoding: i32) {
        let mut b = Vec::with_capacity(12);
        framebuffer_update(&mut b, x, y, w, h, encoding);
        self.write(&b);
    }

    /// A FramebufferUpdate message of one pseudo rectangle.
    fn write_one_rect(&mut self, x: usize, y: usize, w: usize, h: usize, encoding: i32) {
        self.write_u8(MSG_SERVER_FRAMEBUFFER_UPDATE);
        self.write_u8(0);
        self.write_u16(1);
        self.write_rect_header(x, y, w, h, encoding);
    }

    /// `vnc_flush()`: hands the output to the writer.
    pub(crate) fn flush(&mut self) {
        if !self.out.is_empty() && !self.disconnecting {
            self.io.push(std::mem::take(&mut self.out));
        }
    }

    /// `vnc_read_when()`.
    fn read_when(&mut self, handler: Handler, expect: usize) {
        self.handler = handler;
        self.expect = expect;
    }

    /// `vnc_update_throttle_offset()`.
    pub(crate) fn update_throttle_offset(&mut self) {
        let offset = self.client_width * self.client_height * self.pw.pf.bytes_per_pixel;
        self.throttle_output_offset = offset.max(1024 * 1024);
    }

    /// `vnc_disconnect_start()`. Output not flushed yet is lost with the socket, as in QEMU.
    pub(crate) fn disconnect_start(&mut self) {
        if self.disconnecting {
            return;
        }
        self.share_mode = ShareMode::Disconnected;
        self.out.clear();
        self.io.close();
        self.disconnecting = true;
    }

    /// `vnc_should_update()`.
    fn should_update(&self) -> bool {
        match self.update {
            Update::None => false,
            Update::Incremental => {
                self.io.pending() + self.out.len() < self.throttle_output_offset
                    && self.job_update == Update::None
            }
            Update::Force => self.io.force_pending() == 0 && self.job_update == Update::None,
        }
    }

    /// `pixel_format_message()`: the server format, which the client uses from then on.
    fn pixel_format_message(&mut self) {
        self.pw = PixelWriter::server_default();
        let pf = self.pw.pf;
        self.write_u8(pf.bits_per_pixel);
        self.write_u8(pf.depth);
        self.write_u8(u8::from(cfg!(target_endian = "big")));
        self.write_u8(1);
        self.write_u16(pf.rmax as u16);
        self.write_u16(pf.gmax as u16);
        self.write_u16(pf.bmax as u16);
        self.write_u8(pf.rshift as u8);
        self.write_u8(pf.gshift as u8);
        self.write_u8(pf.bshift as u8);
        self.write(&[0; 3]);
    }

    /// `send_color_map()`.
    fn send_color_map(&mut self) {
        let pf = self.pw.pf;
        self.write_u8(MSG_SERVER_SET_COLOUR_MAP_ENTRIES);
        self.write_u8(0);
        self.write_u16(0);
        self.write_u16(256);
        for i in 0u32..256 {
            self.write_u16((((i >> pf.rshift) & pf.rmax) << (16 - pf.rbits)) as u16);
            self.write_u16((((i >> pf.gshift) & pf.gmax) << (16 - pf.gbits)) as u16);
            self.write_u16((((i >> pf.bshift) & pf.bmax) << (16 - pf.bbits)) as u16);
        }
    }

    /// `send_xvp_message()`.
    fn send_xvp_message(&mut self, code: u8) {
        self.write(&[MSG_SERVER_XVP, 0, 1, code]);
    }

    /// `vnc_desktop_resize_ext()`.
    fn desktop_resize_ext(&mut self, reject_reason: usize) {
        let (w, h) = (self.client_width, self.client_height);
        self.write_one_rect(
            usize::from(reject_reason != 0),
            reject_reason,
            w,
            h,
            ENCODING_DESKTOP_RESIZE_EXT,
        );
        self.write(&[1, 0, 0, 0]);
        self.write_u32(0);
        self.write_u16(0);
        self.write_u16(0);
        self.write_u16(w as u16);
        self.write_u16(h as u16);
        self.write_u32(0);
    }

    /// `authentication_failed()`.
    fn authentication_failed(&mut self) {
        self.write_u32(1);
        if self.minor >= 8 {
            const ERR: &[u8] = b"Authentication failed\0";
            self.write_u32(ERR.len() as u32);
            self.write(ERR);
        }
        self.flush();
        self.disconnect_start();
    }

    /// `start_auth_vnc()`.
    fn start_auth_vnc(&mut self) {
        let Ok(challenge) = auth::make_challenge() else {
            self.authentication_failed();
            return;
        };
        self.challenge = challenge;
        self.write(&challenge);
        self.flush();
        self.read_when(Handler::AuthVnc, auth::CHALLENGE_SIZE);
    }

    /// `start_client_init()`.
    fn start_client_init(&mut self) {
        self.read_when(Handler::Init, 1);
    }
}

fn server_info(vd: &VncDisplay) -> Option<ruvm_qapi::types::VncServerInfo> {
    vd.server_info()
}

/// `vnc_qmp_event()` for a connected client.
fn event_connected(vd: &VncDisplay, st: &mut VdState, i: usize) {
    let Some(server) = server_info(vd) else { return };
    let client = st.clients[i].info.basic_info();
    st.deferred.push(Deferred::Event(VncEvent::Connected(VncConnectedArg { server, client })));
}

/// `vnc_connect()`.
pub(crate) fn connect(
    vd: &VncDisplay,
    st: &mut VdState,
    id: u64,
    io: Arc<ClientIo>,
    peer: AddrInfo,
) {
    st.set_refresh(REFRESH_INTERVAL_BASE);
    let first_client = st.clients.is_empty();
    st.clients.push(Client {
        id,
        io,
        info: ClientAddr(peer),
        out: Vec::new(),
        input: Vec::new(),
        expect: 0,
        handler: Handler::Version,
        minor: 0,
        auth: vd.cfg.auth,
        challenge: [0; auth::CHALLENGE_SIZE],
        share_mode: ShareMode::Unset,
        update: Update::None,
        job_update: Update::None,
        dirty: DirtyMap::new(),
        has_dirty: 0,
        client_width: 0,
        client_height: 0,
        pw: PixelWriter::server_default(),
        features: 0,
        encoding: ENCODING_RAW,
        absolute: -1,
        last_x: -1,
        last_y: -1,
        last_bmask: 0,
        tight: Tight::default(),
        zlib: None,
        throttle_output_offset: 0,
        disconnecting: false,
    });
    let i = st.clients.len() - 1;
    event_connected(vd, st, i);
    st.clients[i].share_mode = ShareMode::Connecting;
    if first_client {
        st.update_server_surface();
    }
    st.deferred.push(Deferred::HwUpdate);

    // vnc_start_protocol()
    let c = &mut st.clients[i];
    c.write(b"RFB 003.008\n");
    c.flush();
    c.read_when(Handler::Version, 12);

    if st.num_mode(ShareMode::Connecting) > vd.cfg.connections_limit {
        if let Some(c) = st.clients.iter_mut().find(|c| c.share_mode == ShareMode::Connecting) {
            c.disconnect_start();
        }
    }
}

/// `vnc_disconnect_start()` by index.
pub(crate) fn disconnect_start(st: &mut VdState, i: usize) {
    st.clients[i].disconnect_start();
}

/// `vnc_disconnect_finish()`.
pub(crate) fn disconnect_finish(vd: &VncDisplay, st: &mut VdState, i: usize) {
    let c = st.clients.remove(i);
    if let Some(server) = server_info(vd) {
        let client = c.info.client_info();
        st.deferred
            .push(Deferred::Event(VncEvent::Disconnected(VncDisconnectedArg { server, client })));
    }
    c.io.close();
    let mut keys = Vec::new();
    st.kbd.lift_all_keys(&mut keys);
    if !keys.is_empty() {
        st.deferred.push(Deferred::Keys(keys));
    }
    if st.clients.is_empty() {
        st.update_server_surface();
    }
}

/// `vnc_client_read()` and the handler loop after it. False once the client is gone.
pub(crate) fn input(vd: &VncDisplay, st: &mut VdState, id: u64, data: &[u8]) -> bool {
    let Some(mut i) = st.client_index(id) else { return false };
    if st.clients[i].disconnecting {
        disconnect_finish(vd, st, i);
        return false;
    }
    st.clients[i].input.extend_from_slice(data);
    loop {
        let c = &st.clients[i];
        if c.input.len() < c.expect {
            return true;
        }
        let len = c.expect;
        let handler = c.handler;
        let msg = c.input[..len].to_vec();
        let ret = match handler {
            Handler::Version => protocol_version(st, i, &msg),
            Handler::Auth => protocol_client_auth(st, i, &msg),
            Handler::AuthVnc => protocol_client_auth_vnc(st, i, &msg),
            Handler::Init => protocol_client_init(vd, st, i, &msg),
            Handler::Msg => protocol_client_msg(vd, st, i, &msg),
        };
        // The handlers may have dropped other clients but never removed one.
        i = match st.client_index(id) {
            Some(i) => i,
            None => return false,
        };
        if st.clients[i].disconnecting {
            disconnect_finish(vd, st, i);
            return false;
        }
        let c = &mut st.clients[i];
        if ret == 0 {
            c.input.drain(..len);
        } else {
            c.expect = ret;
        }
    }
}

/// `sscanf(local, "RFB %03d.%03d\n", ...)`: the number of fields read and the two values.
fn scan_version(s: &[u8]) -> (usize, i32, i32) {
    let s = &s[..s.iter().position(|&b| b == 0).unwrap_or(s.len())];
    let mut pos = 0;
    let skip_ws = |pos: &mut usize| {
        while *pos < s.len() && s[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
    };
    let number = |pos: &mut usize| -> Option<i32> {
        skip_ws(pos);
        let start = *pos;
        let mut neg = false;
        if *pos < s.len() && (s[*pos] == b'-' || s[*pos] == b'+') {
            neg = s[*pos] == b'-';
            *pos += 1;
        }
        let digits = *pos;
        let mut v: i32 = 0;
        while *pos < s.len() && *pos - start < 3 && s[*pos].is_ascii_digit() {
            v = v * 10 + i32::from(s[*pos] - b'0');
            *pos += 1;
        }
        if *pos == digits {
            return None;
        }
        Some(if neg { -v } else { v })
    };
    if !s.starts_with(b"RFB") {
        return (0, 0, 0);
    }
    pos += 3;
    skip_ws(&mut pos);
    let Some(major) = number(&mut pos) else { return (0, 0, 0) };
    if pos >= s.len() || s[pos] != b'.' {
        return (1, major, 0);
    }
    pos += 1;
    match number(&mut pos) {
        Some(minor) => (2, major, minor),
        None => (1, major, 0),
    }
}

/// `protocol_version()`.
fn protocol_version(st: &mut VdState, i: usize, data: &[u8]) -> usize {
    let c = &mut st.clients[i];
    let (n, major, minor) = scan_version(data);
    if n != 2 {
        c.disconnect_start();
        return 0;
    }
    c.minor = minor;
    if major != 3 || !matches!(minor, 3 | 4 | 5 | 7 | 8) {
        c.write_u32(AUTH_INVALID);
        c.flush();
        c.disconnect_start();
        return 0;
    }
    // Some broken clients report 3.4 or 3.5, which the specification says are 3.3.
    if minor == 4 || minor == 5 {
        c.minor = 3;
    }
    if c.minor == 3 {
        c.write_u32(c.auth as u32);
        c.flush();
        match c.auth {
            Auth::None => c.start_client_init(),
            Auth::Vnc => c.start_auth_vnc(),
        }
    } else {
        c.write_u8(1);
        c.write_u8(c.auth as u8);
        c.read_when(Handler::Auth, 1);
        c.flush();
    }
    0
}

/// `protocol_client_auth()`.
fn protocol_client_auth(st: &mut VdState, i: usize, data: &[u8]) -> usize {
    let c = &mut st.clients[i];
    if u32::from(data[0]) != c.auth as u32 {
        c.write_u32(1);
        if c.minor >= 8 {
            const ERR: &[u8] = b"Authentication failed\0";
            c.write_u32(ERR.len() as u32);
            c.write(ERR);
        }
        // QEMU closes the socket without flushing, so the client sees none of this.
        c.disconnect_start();
        return 0;
    }
    match c.auth {
        Auth::None => {
            if c.minor >= 8 {
                c.write_u32(0);
                c.flush();
            }
            c.start_client_init();
        }
        Auth::Vnc => c.start_auth_vnc(),
    }
    0
}

/// `protocol_client_auth_vnc()`.
fn protocol_client_auth_vnc(st: &mut VdState, i: usize, data: &[u8]) -> usize {
    let accept = match &st.password {
        Some(pw) if st.expires >= super::now() => {
            matches!(auth::expected_response(pw.as_bytes(), &st.clients[i].challenge),
                     Ok(r) if r[..] == data[..auth::CHALLENGE_SIZE])
        }
        _ => false,
    };
    let c = &mut st.clients[i];
    if accept {
        c.write_u32(0);
        c.flush();
        c.start_client_init();
    } else {
        c.authentication_failed();
    }
    0
}

/// `protocol_client_init()`.
fn protocol_client_init(vd: &VncDisplay, st: &mut VdState, i: usize, data: &[u8]) -> usize {
    let mode = if data[0] != 0 { ShareMode::Shared } else { ShareMode::Exclusive };
    match vd.cfg.share_policy {
        // The traditional QEMU behaviour, against the specification: the flag means nothing.
        SharePolicy::Ignore => {}
        // Exclusive access drops everybody else, and shared access is refused while a client
        // has exclusive access. This is what the specification suggests.
        SharePolicy::AllowExclusive => {
            if mode == ShareMode::Exclusive {
                for (j, c) in st.clients.iter_mut().enumerate() {
                    if j != i && matches!(c.share_mode, ShareMode::Exclusive | ShareMode::Shared) {
                        c.disconnect_start();
                    }
                }
            }
            if mode == ShareMode::Shared && st.num_mode(ShareMode::Exclusive) > 0 {
                st.clients[i].disconnect_start();
                return 0;
            }
        }
        SharePolicy::ForceShared => {
            if mode == ShareMode::Exclusive {
                st.clients[i].disconnect_start();
                return 0;
            }
        }
    }
    st.clients[i].share_mode = mode;
    if st.num_mode(ShareMode::Shared) > vd.cfg.connections_limit {
        st.clients[i].disconnect_start();
        return 0;
    }

    let (width, height) = st.server_dims();
    let c = &mut st.clients[i];
    c.client_width = width;
    c.client_height = height;
    c.write_u16(width as u16);
    c.write_u16(height as u16);
    c.pixel_format_message();

    let mut name = match vd.name() {
        Some(n) => format!("QEMU ({n})").into_bytes(),
        None => b"QEMU".to_vec(),
    };
    // snprintf() into 1024 bytes keeps 1023 of them and the NUL.
    if name.len() >= 1024 {
        name.truncate(1023);
        name.push(0);
    }
    c.write_u32(name.len() as u32);
    c.write(&name);
    c.flush();

    if let Some(server) = server_info(vd) {
        let client = c.info.client_info();
        st.deferred
            .push(Deferred::Event(VncEvent::Initialized(VncInitializedArg { server, client })));
    }
    st.clients[i].read_when(Handler::Msg, 1);
    0
}

/// `set_pixel_conversion()`.
fn set_pixel_conversion(c: &mut Client) {
    c.pw = PixelWriter::for_format(c.pw.pf);
}

/// `set_pixel_format()`.
fn set_pixel_format(st: &mut VdState, i: usize, data: &[u8]) {
    let c = &mut st.clients[i];
    let true_color = data[7];
    let Some(pf) = VncPixelFormat::from_client(
        data[4],
        data[6],
        true_color,
        u16::from_be_bytes([data[8], data[9]]),
        u16::from_be_bytes([data[10], data[11]]),
        u16::from_be_bytes([data[12], data[13]]),
        data[14],
        data[15],
        data[16],
    ) else {
        c.disconnect_start();
        return;
    };
    c.pw.pf = pf;
    if true_color == 0 {
        c.send_color_map();
    }
    set_pixel_conversion(c);
    st.deferred.push(Deferred::HwInvalidate);
    st.deferred.push(Deferred::HwUpdate);
}

/// `vnc_colordepth()`.
pub(crate) fn colordepth(st: &mut VdState, i: usize) {
    let c = &mut st.clients[i];
    if c.has_feature(FEATURE_WMVI) {
        let (w, h) = (c.client_width, c.client_height);
        c.write_one_rect(0, 0, w, h, ENCODING_WMVI);
        c.pixel_format_message();
        c.flush();
    } else {
        set_pixel_conversion(c);
    }
}

/// `vnc_desktop_resize()`.
pub(crate) fn desktop_resize(st: &mut VdState, i: usize) {
    let true_width = st.true_width;
    let (_, height) = st.server_dims();
    let c = &mut st.clients[i];
    if !c.has_feature(FEATURE_RESIZE) && !c.has_feature(FEATURE_RESIZE_EXT) {
        return;
    }
    if c.client_width == true_width && c.client_height == height {
        return;
    }
    c.client_width = true_width;
    c.client_height = height;
    if c.has_feature(FEATURE_RESIZE_EXT) {
        c.desktop_resize_ext(0);
        return;
    }
    c.write_one_rect(0, 0, true_width, height, ENCODING_DESKTOPRESIZE);
    c.flush();
}

/// `set_encodings()`.
fn set_encodings(vd: &VncDisplay, st: &mut VdState, i: usize, encodings: &[i32]) {
    let (sw, sh) = st.server_dims();
    let c = &mut st.clients[i];
    c.features = 0;
    c.encoding = 0;
    c.tight.compression = 9;
    c.tight.quality = -1;
    c.absolute = -1;
    // The list is in order of preference, so the first one the server knows wins.
    for &enc in encodings.iter().rev() {
        match enc {
            ENCODING_RAW => c.encoding = enc,
            ENCODING_HEXTILE => {
                c.features |= FEATURE_HEXTILE;
                c.encoding = enc;
            }
            ENCODING_TIGHT => {
                c.features |= FEATURE_TIGHT;
                c.encoding = enc;
            }
            ENCODING_ZLIB => {
                c.features |= FEATURE_ZLIB;
                c.encoding = enc;
            }
            ENCODING_DESKTOPRESIZE => c.features |= FEATURE_RESIZE,
            ENCODING_DESKTOP_RESIZE_EXT => c.features |= FEATURE_RESIZE_EXT,
            ENCODING_POINTER_TYPE_CHANGE => c.features |= FEATURE_POINTER_TYPE_CHANGE,
            ENCODING_RICH_CURSOR => c.features |= FEATURE_RICH_CURSOR,
            ENCODING_ALPHA_CURSOR => c.features |= FEATURE_ALPHA_CURSOR,
            ENCODING_EXT_KEY_EVENT => {
                // send_ext_key_event_ack()
                c.write_one_rect(0, 0, sw, sh, ENCODING_EXT_KEY_EVENT);
                c.flush();
            }
            // Without an audio backend the client never hears about audio.
            ENCODING_AUDIO => {}
            ENCODING_WMVI => c.features |= FEATURE_WMVI,
            ENCODING_LED_STATE => c.features |= FEATURE_LED_STATE,
            ENCODING_XVP => {
                if vd.cfg.power_control {
                    c.features |= FEATURE_XVP;
                    c.send_xvp_message(XVP_CODE_INIT);
                    c.flush();
                }
            }
            e if (ENCODING_COMPRESSLEVEL0..=ENCODING_COMPRESSLEVEL0 + 9).contains(&e) => {
                c.tight.compression = (e & 0x0f) as u8;
            }
            e if (ENCODING_QUALITYLEVEL0..=ENCODING_QUALITYLEVEL0 + 9).contains(&e)
                && vd.cfg.lossy =>
            {
                c.tight.quality = e & 0x0f;
            }
            // ZRLE, ZYWRLE, tight PNG, the extended clipboard and anything unknown.
            _ => {}
        }
    }
    desktop_resize(st, i);
    check_pointer_type_change(st, i, vd.is_absolute());
    led_state_change(st, i);
}

/// `check_pointer_type_change()` with `qemu_input_is_absolute()` already asked.
pub(crate) fn check_pointer_type_change(st: &mut VdState, i: usize, absolute: i32) {
    let (sw, sh) = st.server_dims();
    let c = &mut st.clients[i];
    if c.has_feature(FEATURE_POINTER_TYPE_CHANGE) && c.absolute != absolute {
        c.write_one_rect(absolute as usize, 0, sw, sh, ENCODING_POINTER_TYPE_CHANGE);
        c.flush();
    }
    c.absolute = absolute;
}

/// `vnc_led_state_change()`.
pub(crate) fn led_state_change(st: &mut VdState, i: usize) {
    let ledstate = st.ledstate;
    let c = &mut st.clients[i];
    if c.has_feature(FEATURE_LED_STATE) {
        c.write_one_rect(0, 0, 1, 1, ENCODING_LED_STATE);
        c.write_u8(ledstate);
        c.flush();
    }
}

/// `pointer_event()`.
fn pointer_event(st: &mut VdState, i: usize, button_mask: u8, x: u16, y: u16) {
    let (width, height) = st.server_dims();
    let c = &mut st.clients[i];
    let old = c.last_bmask;
    c.last_bmask = button_mask;
    let (x, y) = (i32::from(x), i32::from(y));
    // A client that never sent its encodings has -1 here, which counts as absolute.
    let motion = if c.absolute != 0 {
        Motion::Abs { x, y, width: width as i32, height: height as i32 }
    } else if c.has_feature(FEATURE_POINTER_TYPE_CHANGE) {
        Motion::Rel { dx: i64::from(x - 0x7FFF), dy: i64::from(y - 0x7FFF) }
    } else {
        let m = if c.last_x != -1 {
            Motion::Rel { dx: i64::from(x - c.last_x), dy: i64::from(y - c.last_y) }
        } else {
            Motion::None
        };
        c.last_x = x;
        c.last_y = y;
        m
    };
    st.deferred.push(Deferred::Pointer { old, new: button_mask, motion });
}

/// `press_key()`.
fn press_key(st: &mut VdState, lnx: u32, out: &mut Vec<super::KbdOut>) {
    st.kbd.key_event(lnx, true, out);
    st.kbd.key_event(lnx, false, out);
}

/// `do_key_event()`: `keycode` is a QEMU key number, `sym` the keysym the client sent.
fn do_key_event(vd: &VncDisplay, st: &mut VdState, i: usize, down: bool, keycode: u32, sym: u32) {
    let lnx = key_number_to_linux(i64::from(keycode));
    let mut out = Vec::new();

    // The console switch keys.
    if (KEY_1..=KEY_9).contains(&lnx)
        && down
        && st.kbd.modifier_get(QKbdModifier::Ctrl)
        && st.kbd.modifier_get(QKbdModifier::Alt)
    {
        if let Some(con) = vd.ds.lookup_by_index(lnx - KEY_1) {
            st.kbd.switch_console(Some(con), &mut out);
            st.deferred.push(Deferred::Keys(out));
        }
        return;
    }

    // A client with the LED state extension keeps the lock keys in step itself.
    let sync = down && vd.cfg.lock_key_sync && !st.clients[i].has_feature(FEATURE_LED_STATE);
    if sync && keycode_is_keypad(keycode) {
        // Press numlock first when it changed while the VNC window was not looking.
        let numlock = st.kbd.modifier_get(QKbdModifier::NumLock);
        if keysym_is_numlock(sym & 0xFFFF) != numlock {
            press_key(st, KEY_NUMLOCK, &mut out);
        }
    }
    let upper = (u32::from(b'A')..=u32::from(b'Z')).contains(&sym);
    let lower = (u32::from(b'a')..=u32::from(b'z')).contains(&sym);
    if sync && (upper || lower) {
        // The same for capslock.
        let shift = st.kbd.modifier_get(QKbdModifier::Shift);
        let capslock = st.kbd.modifier_get(QKbdModifier::CapsLock);
        if capslock == (upper == shift) {
            press_key(st, KEY_CAPSLOCK, &mut out);
        }
    }

    // Every console is graphic, so there is no text console emulation to feed.
    st.kbd.key_event(lnx, down, &mut out);
    if !out.is_empty() {
        st.deferred.push(Deferred::Keys(out));
    }
}

/// `key_event()`: a keysym, mapped through the display's layout.
fn key_event(vd: &VncDisplay, st: &mut VdState, i: usize, down: bool, sym: u32) {
    let mut lsym = sym;
    if (u32::from(b'A')..=u32::from(b'Z')).contains(&lsym) {
        lsym = lsym - u32::from(b'A') + u32::from(b'a');
    }
    let keycode = vd.layout.keysym2scancode(lsym & 0xFFFF, Some(&st.kbd), down) & SCANCODE_KEYMASK;
    do_key_event(vd, st, i, down, keycode, sym);
}

/// `ext_key_event()`: the QEMU extended key event, which carries the key number too.
fn ext_key_event(vd: &VncDisplay, st: &mut VdState, i: usize, down: bool, sym: u32, keycode: u32) {
    // A layout given with -k always wins.
    if keymaps::keyboard_layout().is_some() {
        key_event(vd, st, i, down, sym);
    } else {
        do_key_event(vd, st, i, down, keycode, sym);
    }
}

/// `framebuffer_update_request()`.
fn framebuffer_update_request(st: &mut VdState, i: usize, incremental: bool, rect: [u16; 4]) {
    let dims = st.guest_dims();
    let c = &mut st.clients[i];
    if incremental {
        if c.update != Update::Force {
            c.update = Update::Incremental;
        }
        return;
    }
    c.update = Update::Force;
    let [x, y, w, h] = rect.map(i64::from);
    set_area_dirty(&mut c.dirty, dims, x, y, w, h);
    if c.has_feature(FEATURE_RESIZE_EXT) {
        c.desktop_resize_ext(0);
    }
}

/// Sends an XVP failure to a client that is still there, after the power hook refused.
pub(crate) fn xvp_fail(st: &mut VdState, id: u64) {
    if let Some(i) = st.client_index(id) {
        let c = &mut st.clients[i];
        c.send_xvp_message(XVP_CODE_FAIL);
        c.flush();
    }
}

fn read_u16(data: &[u8], at: usize) -> u16 {
    u16::from_be_bytes([data[at], data[at + 1]])
}

fn read_s32(data: &[u8], at: usize) -> i32 {
    i32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

fn read_u32(data: &[u8], at: usize) -> u32 {
    read_s32(data, at) as u32
}

/// `protocol_client_msg()`.
fn protocol_client_msg(vd: &VncDisplay, st: &mut VdState, i: usize, data: &[u8]) -> usize {
    let len = data.len();
    if data[0] > 3 {
        st.set_refresh(REFRESH_INTERVAL_BASE);
    }
    match data[0] {
        MSG_CLIENT_SET_PIXEL_FORMAT => {
            if len == 1 {
                return 20;
            }
            set_pixel_format(st, i, data);
        }
        MSG_CLIENT_SET_ENCODINGS => {
            if len == 1 {
                return 4;
            }
            let limit = usize::from(read_u16(data, 2));
            if len == 4 && limit > 0 {
                return 4 + limit * 4;
            }
            let encodings: Vec<i32> = (0..limit).map(|n| read_s32(data, 4 + n * 4)).collect();
            set_encodings(vd, st, i, &encodings);
        }
        MSG_CLIENT_FRAMEBUFFER_UPDATE_REQUEST => {
            if len == 1 {
                return 10;
            }
            let rect = [read_u16(data, 2), read_u16(data, 4), read_u16(data, 6), read_u16(data, 8)];
            framebuffer_update_request(st, i, data[1] != 0, rect);
        }
        MSG_CLIENT_KEY_EVENT => {
            if len == 1 {
                return 8;
            }
            key_event(vd, st, i, data[1] != 0, read_u32(data, 4));
        }
        MSG_CLIENT_POINTER_EVENT => {
            if len == 1 {
                return 6;
            }
            pointer_event(st, i, data[1], read_u16(data, 2), read_u16(data, 4));
        }
        MSG_CLIENT_CUT_TEXT => {
            if len == 1 {
                return 8;
            }
            let raw = read_s32(data, 4);
            let dlen = raw.unsigned_abs();
            let mut fail = false;
            if len == 8 {
                if dlen > 1 << 20 {
                    error_report(&format!(
                        "vnc: client_cut_text msg payload has {dlen} bytes which exceeds our limit of 1MB."
                    ));
                    fail = true;
                } else if dlen > 0 {
                    return 8 + dlen as usize;
                }
            }
            // No extended clipboard is offered, so a negative length is a broken client.
            if !fail && raw < 0 {
                error_report("vnc: extended clipboard message while disabled");
                fail = true;
            }
            if fail {
                st.clients[i].disconnect_start();
            }
        }
        MSG_CLIENT_XVP => {
            let c = &mut st.clients[i];
            if !c.has_feature(FEATURE_XVP) {
                error_report("vnc: xvp client message while disabled");
                c.disconnect_start();
            } else if len == 1 {
                return 4;
            } else if data[2] != 1 {
                error_report(&format!("vnc: xvp client message version {} != 1", data[2]));
                c.disconnect_start();
            } else {
                let id = c.id;
                match data[3] {
                    XVP_ACTION_SHUTDOWN => st.deferred.push(Deferred::XvpPowerdown(id)),
                    XVP_ACTION_RESET => st.deferred.push(Deferred::XvpReset(id)),
                    _ => {
                        c.send_xvp_message(XVP_CODE_FAIL);
                        c.flush();
                    }
                }
            }
        }
        MSG_CLIENT_QEMU => {
            if len == 1 {
                return 2;
            }
            match data[1] {
                MSG_CLIENT_QEMU_EXT_KEY_EVENT => {
                    if len == 2 {
                        return 12;
                    }
                    let down = read_u16(data, 2) != 0;
                    ext_key_event(vd, st, i, down, read_u32(data, 4), read_u32(data, 8));
                }
                1 => {
                    let op = data.get(2).copied().unwrap_or(0);
                    error_report(&format!("Audio message {op} with audio disabled"));
                    st.clients[i].disconnect_start();
                }
                _ => st.clients[i].disconnect_start(),
            }
        }
        MSG_CLIENT_SET_DESKTOP_SIZE => {
            if len < 8 {
                return 8;
            }
            let size = 8 + usize::from(data[6]) * 16;
            if len < size {
                return size;
            }
            let w = u32::from(read_u16(data, 2));
            let h = u32::from(read_u16(data, 4));
            if vd.console().is_some_and(|c| c.ui_info_supported()) {
                st.deferred.push(Deferred::SetUiInfo(w, h));
                st.clients[i].desktop_resize_ext(4);
            } else {
                st.clients[i].desktop_resize_ext(3);
            }
        }
        _ => st.clients[i].disconnect_start(),
    }
    let c = &mut st.clients[i];
    c.update_throttle_offset();
    c.read_when(Handler::Msg, 1);
    0
}

/// `find_and_clear_dirty_height()`.
fn find_and_clear_dirty_height(
    c: &mut Client,
    y: usize,
    last_x: usize,
    x: usize,
    height: usize,
) -> usize {
    let mut h = 1;
    while h < height - y {
        if !c.dirty.test(y + h, last_x) {
            break;
        }
        c.dirty.clear_bits(y + h, last_x, x - last_x);
        h += 1;
    }
    h
}

/// `vnc_update_client()` and the job it pushes, run at once. Returns the rectangles found.
pub(crate) fn update_client(vd: &VncDisplay, st: &mut VdState, i: usize, has_dirty: i64) -> i64 {
    if st.clients[i].disconnecting {
        disconnect_finish(vd, st, i);
        return 0;
    }
    let VdState { server, clients, .. } = st;
    let c = &mut clients[i];
    c.has_dirty += has_dirty;
    if !c.should_update() {
        return 0;
    }
    if c.has_dirty == 0 && c.update != Update::Force {
        return 0;
    }
    let Some(server) = server.as_ref() else { return 0 };
    let (width, height) = (server.width, server.height);

    // Collected the way QEMU puts them on the job's list, which runs them newest first.
    let mut rects = Vec::new();
    let size = height * DIRTY_BPL;
    let mut y = 0;
    loop {
        let offset = c.dirty.find_next_bit(size, y * DIRTY_BPL);
        if offset == size {
            break;
        }
        y = offset / DIRTY_BPL;
        let x = offset % DIRTY_BPL;
        let mut x2 = c.dirty.find_next_zero_bit(y, x);
        c.dirty.clear_bits(y, x, x2 - x);
        let h = find_and_clear_dirty_height(c, y, x, x2, height);
        x2 = x2.min(width / DIRTY_PIXELS_PER_BIT);
        if x2 > x {
            rects.push((x * DIRTY_PIXELS_PER_BIT, y, (x2 - x) * DIRTY_PIXELS_PER_BIT, h));
        }
        if x == 0 && x2 == width / DIRTY_PIXELS_PER_BIT {
            y += h;
            if y == height {
                break;
            }
        }
    }
    let n = rects.len() as i64;
    c.job_update = c.update;
    c.update = Update::None;
    c.has_dirty = 0;
    // vnc_job_push() drops a job without rectangles and leaves job_update as it is.
    if rects.is_empty() {
        return n;
    }

    // vnc_worker_thread_loop()
    let fb = Fb::new(&server.data, server.width);
    let mut buf = vec![MSG_SERVER_FRAMEBUFFER_UPDATE, 0, 0, 0];
    let mut n_rectangles: i32 = 0;
    for &(x, y, w, h) in rects.iter().rev() {
        // vnc_worker_clamp_rect()
        if x >= c.client_width || y >= c.client_height {
            continue;
        }
        let w = w.min(c.client_width - x);
        let h = h.min(c.client_height - y);
        if w == 0 || h == 0 {
            continue;
        }
        let pw = c.pw;
        let n = match c.encoding {
            ENCODING_ZLIB => {
                zlib::send(&mut buf, &mut c.zlib, c.tight.compression.into(), &fb, &pw, x, y, w, h)
            }
            ENCODING_HEXTILE => {
                framebuffer_update(&mut buf, x, y, w, h, ENCODING_HEXTILE);
                hextile::send(&mut buf, &fb, &pw, x, y, w, h)
            }
            ENCODING_TIGHT => tight::send(&mut buf, &mut c.tight, &fb, &pw, x, y, w, h),
            _ => {
                framebuffer_update(&mut buf, x, y, w, h, ENCODING_RAW);
                raw::send(&mut buf, &fb, &pw, x, y, w, h)
            }
        };
        if n >= 0 {
            n_rectangles += n;
        }
    }
    buf[2..4].copy_from_slice(&(n_rectangles as u16).to_be_bytes());

    // vnc_jobs_consume_buffer()
    c.flush();
    if !c.disconnecting {
        c.io.push(buf);
        if c.job_update == Update::Force {
            c.io.set_force_pending(c.io.pending());
        }
    }
    c.job_update = Update::None;
    n
}

#[cfg(test)]
mod tests {
    use super::scan_version;

    #[test]
    fn version_scan_follows_sscanf() {
        assert_eq!(scan_version(b"RFB 003.008\n"), (2, 3, 8));
        assert_eq!(scan_version(b"RFB 003.003\n"), (2, 3, 3));
        assert_eq!(scan_version(b"RFB3.8\n\0\0\0\0\0"), (2, 3, 8));
        assert_eq!(scan_version(b"RFB 0038.000"), (1, 3, 0));
        assert_eq!(scan_version(b"XFB 003.008\n"), (0, 0, 0));
        assert_eq!(scan_version(b"RFB 003,008\n"), (1, 3, 0));
    }
}
