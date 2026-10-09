// SPDX-License-Identifier: GPL-2.0-or-later

//! Tests of the VNC server: the encoders against small decoders written from the RFB
//! specification, the handshake over loopback, VNC authentication and the option errors.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use flate2::{Decompress, FlushDecompress};
use ruvm_qapi::types::{
    ExpirePasswordOptions, ExpirePasswordOptionsU, ExpirePasswordOptionsVnc, SetPasswordAction,
    SetPasswordOptions, SetPasswordOptionsU, SetPasswordOptionsVnc,
};

use super::pixels::{PixelWriter, VncPixelFormat};
use super::tight::Tight;
use super::zlib::ZStream;
use super::*;
use crate::console::DisplayState;

/// A server surface with flat areas, a two colour stripe, a few colour blocks and noise.
fn test_image(w: usize, h: usize) -> Vec<u32> {
    let mut seed = 0x1234_5678u32;
    let mut img = vec![0u32; w * h];
    for y in 0..h {
        for x in 0..w {
            img[y * w + x] = if y < 20 {
                0x0020_4060
            } else if y < 30 {
                if (x / 3 + y) % 2 == 0 { 0x00ff_ffff } else { 0x0000_0000 }
            } else if y < 45 {
                [0x00ff_0000, 0x0000_ff00, 0x0000_00ff, 0x00ff_ff00, 0x0012_3456][(x / 7) % 5]
            } else {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
                seed >> 8 & 0x00ff_ffff
            };
        }
    }
    img
}

fn formats() -> Vec<PixelWriter> {
    let rgb565 = VncPixelFormat::from_client(16, 0, 1, 31, 63, 31, 11, 5, 0).unwrap();
    let rgb565be = VncPixelFormat::from_client(16, 1, 1, 31, 63, 31, 11, 5, 0).unwrap();
    let bgr233 = VncPixelFormat::from_client(8, 0, 1, 7, 7, 3, 0, 3, 6).unwrap();
    let bgr32 = VncPixelFormat::from_client(32, 1, 1, 255, 255, 255, 0, 8, 16).unwrap();
    vec![
        PixelWriter::server_default(),
        PixelWriter::for_format(rgb565),
        PixelWriter::for_format(rgb565be),
        PixelWriter::for_format(bgr233),
        PixelWriter::for_format(bgr32),
    ]
}

/// The client pixels of a rectangle, `bpp` bytes each, row by row.
type Pixels = Vec<Vec<u8>>;

fn expected(fb: &Fb<'_>, pw: &PixelWriter, x: usize, y: usize, w: usize, h: usize) -> Pixels {
    let mut raw = Vec::new();
    raw::send(&mut raw, fb, pw, x, y, w, h);
    raw.chunks_exact(pw.bytes_per_pixel()).map(<[u8]>::to_vec).collect()
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u16(&mut self) -> usize {
        let s = self.take(2);
        usize::from(u16::from_be_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> u32 {
        let s = self.take(4);
        u32::from_be_bytes([s[0], s[1], s[2], s[3]])
    }
    fn header(&mut self) -> (usize, usize, usize, usize, i32) {
        let (x, y, w, h) = (self.u16(), self.u16(), self.u16(), self.u16());
        (x, y, w, h, self.u32() as i32)
    }
}

/// A frame the decoders paint into, in client pixels.
struct Frame {
    w: usize,
    px: Pixels,
}

impl Frame {
    fn new(w: usize, h: usize, bpp: usize) -> Frame {
        Frame { w, px: vec![vec![0xee; bpp]; w * h] }
    }
    fn fill(&mut self, x: usize, y: usize, w: usize, h: usize, p: &[u8]) {
        for j in y..y + h {
            for i in x..x + w {
                self.px[j * self.w + i] = p.to_vec();
            }
        }
    }
    fn rect(&self, x: usize, y: usize, w: usize, h: usize) -> Pixels {
        let mut out = Vec::new();
        for j in y..y + h {
            out.extend_from_slice(&self.px[j * self.w + x..j * self.w + x + w]);
        }
        out
    }
}

fn inflate(z: &mut Decompress, data: &[u8], out_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(out_len + 64);
    let start = z.total_in();
    loop {
        let done = (z.total_in() - start) as usize;
        z.decompress_vec(&data[done..], &mut out, FlushDecompress::Sync).unwrap();
        if (z.total_in() - start) as usize == data.len() || out.len() >= out_len {
            break;
        }
        out.reserve(4096);
    }
    assert_eq!(out.len(), out_len);
    out
}

fn decode_hextile(
    r: &mut Reader<'_>,
    f: &mut Frame,
    rect: (usize, usize, usize, usize),
    bpp: usize,
) {
    let (x, y, w, h) = rect;
    let (mut bg, mut fg) = (vec![0; bpp], vec![0; bpp]);
    for ty in (y..y + h).step_by(16) {
        for tx in (x..x + w).step_by(16) {
            let tw = 16.min(x + w - tx);
            let th = 16.min(y + h - ty);
            let sub = r.u8();
            if sub & 1 != 0 {
                for j in 0..th {
                    for i in 0..tw {
                        let p = r.take(bpp).to_vec();
                        f.px[(ty + j) * f.w + tx + i] = p;
                    }
                }
                continue;
            }
            if sub & 2 != 0 {
                bg = r.take(bpp).to_vec();
            }
            f.fill(tx, ty, tw, th, &bg);
            if sub & 4 != 0 {
                fg = r.take(bpp).to_vec();
            }
            if sub & 8 != 0 {
                let n = r.u8();
                for _ in 0..n {
                    let color = if sub & 16 != 0 { r.take(bpp).to_vec() } else { fg.clone() };
                    let xy = r.u8();
                    let wh = r.u8();
                    let (sx, sy) = (usize::from(xy >> 4), usize::from(xy & 15));
                    let (sw, sh) = (usize::from(wh >> 4) + 1, usize::from(wh & 15) + 1);
                    f.fill(tx + sx, ty + sy, sw, sh, &color);
                }
            }
        }
    }
}

/// The tight client state: its four zlib streams.
struct TightDec {
    z: [Decompress; 4],
}

impl TightDec {
    fn new() -> TightDec {
        TightDec { z: std::array::from_fn(|_| Decompress::new(true)) }
    }
}

fn compact_len(r: &mut Reader<'_>) -> usize {
    let b = r.u8();
    let mut len = usize::from(b & 0x7f);
    if b & 0x80 != 0 {
        let b = r.u8();
        len |= usize::from(b & 0x7f) << 7;
        if b & 0x80 != 0 {
            len |= usize::from(r.u8()) << 14;
        }
    }
    len
}

fn decode_tight(
    r: &mut Reader<'_>,
    f: &mut Frame,
    rect: (usize, usize, usize, usize),
    pw: &PixelWriter,
    d: &mut TightDec,
) {
    let (x, y, w, h) = rect;
    let pf = pw.pf;
    let bpp = pw.bytes_per_pixel();
    let pixel24 = pf.bytes_per_pixel == 4 && pf.rmax == 0xff && pf.gmax == 0xff && pf.bmax == 0xff;
    let tsize = if pixel24 { 3 } else { bpp };
    // A packed pixel is R, G, B; put it back together in the client's format.
    let tpixel = |b: &[u8]| -> Vec<u8> {
        if !pixel24 {
            return b.to_vec();
        }
        let p = (u32::from(b[0]) << pf.rshift)
            | (u32::from(b[1]) << pf.gshift)
            | (u32::from(b[2]) << pf.bshift);
        if pf.big_endian { p.to_be_bytes().to_vec() } else { p.to_le_bytes().to_vec() }
    };
    let ctl = r.u8();
    for (i, z) in d.z.iter_mut().enumerate() {
        if ctl & (1 << i) != 0 {
            *z = Decompress::new(true);
        }
    }
    let ctl = ctl >> 4;
    if ctl == 8 {
        let p = tpixel(r.take(tsize));
        f.fill(x, y, w, h, &p);
        return;
    }
    assert!(ctl & 8 == 0, "JPEG or an unknown tight compression: {ctl:#x}");
    let stream = usize::from(ctl & 3);
    let filter = if ctl & 4 != 0 { r.u8() } else { 0 };
    let palette: Vec<Vec<u8>> = match filter {
        0 => Vec::new(),
        1 => {
            let n = usize::from(r.u8()) + 1;
            (0..n).map(|_| tpixel(r.take(tsize))).collect()
        }
        _ => panic!("tight filter {filter} is lossy only"),
    };
    let len = match filter {
        0 => w * h * tsize,
        _ if palette.len() == 2 => w.div_ceil(8) * h,
        _ => w * h,
    };
    let data = if len < 12 {
        r.take(len).to_vec()
    } else {
        let clen = compact_len(r);
        let comp = r.take(clen).to_vec();
        inflate(&mut d.z[stream], &comp, len)
    };
    for j in 0..h {
        for i in 0..w {
            let p = match filter {
                0 => tpixel(&data[(j * w + i) * tsize..(j * w + i + 1) * tsize]),
                _ if palette.len() == 2 => {
                    let byte = data[j * w.div_ceil(8) + i / 8];
                    palette[usize::from(byte >> (7 - i % 8) & 1)].clone()
                }
                _ => palette[usize::from(data[j * w + i])].clone(),
            };
            f.px[(y + j) * f.w + x + i] = p;
        }
    }
}

/// Decodes `count` rectangles of a framebuffer update body.
fn decode_rects(
    r: &mut Reader<'_>,
    count: usize,
    f: &mut Frame,
    pw: &PixelWriter,
    zlib: &mut Decompress,
    tight: &mut TightDec,
) {
    let bpp = pw.bytes_per_pixel();
    for _ in 0..count {
        let (x, y, w, h, enc) = r.header();
        match enc {
            ENCODING_RAW => {
                for j in 0..h {
                    for i in 0..w {
                        f.px[(y + j) * f.w + x + i] = r.take(bpp).to_vec();
                    }
                }
            }
            ENCODING_HEXTILE => decode_hextile(r, f, (x, y, w, h), bpp),
            ENCODING_ZLIB => {
                let len = r.u32() as usize;
                let comp = r.take(len).to_vec();
                let data = inflate(zlib, &comp, w * h * bpp);
                for (k, p) in data.chunks_exact(bpp).enumerate() {
                    f.px[(y + k / w) * f.w + x + k % w] = p.to_vec();
                }
            }
            ENCODING_TIGHT => decode_tight(r, f, (x, y, w, h), pw, tight),
            e => panic!("unexpected encoding {e}"),
        }
    }
}

#[test]
fn encoders_round_trip() {
    let (w, h, stride) = (100, 70, 112);
    let img = test_image(stride, h);
    let fb = Fb::new(&img, stride);
    let rects =
        [(0, 0, 100, 70), (3, 5, 90, 60), (17, 21, 1, 1), (0, 20, 100, 10), (40, 45, 13, 25)];
    for pw in formats() {
        for enc in [ENCODING_RAW, ENCODING_HEXTILE, ENCODING_ZLIB, ENCODING_TIGHT] {
            let mut zs: Option<ZStream> = None;
            let mut tight = Tight::default();
            let mut zd = Decompress::new(true);
            let mut td = TightDec::new();
            let mut frame = Frame::new(w, h, pw.bytes_per_pixel());
            for (level, &(x, y, rw, rh)) in rects.iter().enumerate() {
                let mut out = Vec::new();
                tight.compression = (level * 2) as u8;
                let n = match enc {
                    ENCODING_ZLIB => {
                        zlib::send(&mut out, &mut zs, level as u32 * 2, &fb, &pw, x, y, rw, rh)
                    }
                    ENCODING_TIGHT => tight::send(&mut out, &mut tight, &fb, &pw, x, y, rw, rh),
                    ENCODING_HEXTILE => {
                        framebuffer_update(&mut out, x, y, rw, rh, ENCODING_HEXTILE);
                        hextile::send(&mut out, &fb, &pw, x, y, rw, rh)
                    }
                    _ => {
                        framebuffer_update(&mut out, x, y, rw, rh, ENCODING_RAW);
                        raw::send(&mut out, &fb, &pw, x, y, rw, rh)
                    }
                };
                assert!(n >= 1, "encoding {enc} sent no rectangle");
                let mut r = Reader { b: &out, pos: 0 };
                decode_rects(&mut r, n as usize, &mut frame, &pw, &mut zd, &mut td);
                assert_eq!(r.pos, out.len(), "encoding {enc} left bytes over");
                assert!(
                    frame.rect(x, y, rw, rh) == expected(&fb, &pw, x, y, rw, rh),
                    "encoding {enc} at {}bpp, rect {x},{y} {rw}x{rh}",
                    pw.pf.bits_per_pixel
                );
            }
        }
    }
}

#[test]
fn des_vector() {
    // FIPS 81's DES example, key 133457799bbcdff1, with the key bits in RFB order.
    let key = [0x13u8, 0x34, 0x57, 0x79, 0x9b, 0xbc, 0xdf, 0xf1].map(u8::reverse_bits);
    let block = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
    let mut challenge = [0u8; 16];
    challenge[..8].copy_from_slice(&block);
    challenge[8..].copy_from_slice(&block);
    let r = auth::expected_response(&key, &challenge).unwrap();
    let want = [0x85, 0xe8, 0x13, 0x54, 0x0f, 0x0a, 0xb4, 0x05];
    assert_eq!(r[..8], want);
    assert_eq!(r[8..], want);
    // A password shorter than eight bytes is zero padded and stops at a NUL.
    assert_eq!(
        auth::password_key(b"ab\0cd"),
        [b'a'.reverse_bits(), b'b'.reverse_bits(), 0, 0, 0, 0, 0, 0]
    );
}

#[test]
fn full_palette_sends_the_last_colour_as_the_first() {
    let mut palette = palette::Palette::new(256);
    let data: Vec<u32> = (0..256).map(|i| i * 0x10101).collect();
    for &c in &data {
        palette.put(c);
    }
    let idx = tight::encode_indexed_rect(&data, &palette);
    assert_eq!(idx[..255], (0..=254).collect::<Vec<u8>>()[..]);
    assert_eq!(idx[255], 0);
}

#[test]
fn dirty_map_bits() {
    let mut d = DirtyMap::new();
    assert_eq!(d.find_next_bit(10 * DIRTY_BPL, 0), 10 * DIRTY_BPL);
    d.set_bits(3, 70, 5);
    assert_eq!(d.find_next_bit(10 * DIRTY_BPL, 0), 3 * DIRTY_BPL + 70);
    assert_eq!(d.find_next_zero_bit(3, 70), 75);
    assert!(d.test_and_clear(3, 72));
    assert_eq!(d.find_next_zero_bit(3, 70), 72);
    let mut d = DirtyMap::new();
    set_area_dirty(&mut d, (100, 50), 17, 2, 20, 3);
    assert!(d.test(2, 1) && d.test(2, 2) && !d.test(2, 3) && !d.test(2, 0));
    assert!(d.test(4, 1) && !d.test(5, 1) && !d.test(1, 1));
    assert_eq!(vnc_width(100), 112);
}

#[derive(Default)]
struct TestHooks(Mutex<Vec<&'static str>>);

impl Hooks for TestHooks {
    fn event(&self, event: VncEvent) {
        let name = match event {
            VncEvent::Connected(_) => "connected",
            VncEvent::Initialized(_) => "initialized",
            VncEvent::Disconnected(_) => "disconnected",
        };
        self.0.lock().unwrap().push(name);
    }
}

fn open_str(arg: &str, id: &str, hooks: Arc<TestHooks>) -> Result<Arc<VncDisplay>> {
    open_with_input(arg, id, InputState::new(), hooks)
}

fn open_with_input(
    arg: &str,
    id: &str,
    input: Arc<InputState>,
    hooks: Arc<TestHooks>,
) -> Result<Arc<VncDisplay>> {
    let mut list = opts::opts_list();
    let o = list.parse_noisily(arg, true).expect("options parse");
    opts::open(o, id, Some("test"), DisplayState::new(), input, hooks)
}

fn connect(vd: &VncDisplay) -> TcpStream {
    let port = vd.server_info().unwrap().service;
    let s = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    s
}

fn read_n(s: &mut TcpStream, n: usize) -> Vec<u8> {
    let mut b = vec![0; n];
    s.read_exact(&mut b).unwrap();
    b
}

fn read_u32(s: &mut TcpStream) -> u32 {
    let b = read_n(s, 4);
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

fn wait_for(hooks: &TestHooks, what: &str) {
    let start = Instant::now();
    while !hooks.0.lock().unwrap().contains(&what) {
        assert!(start.elapsed() < Duration::from_secs(10), "no {what} event");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// ServerInit, checked, after the security handshake.
fn server_init(s: &mut TcpStream) -> (usize, usize) {
    s.write_all(&[1]).unwrap();
    let b = read_n(s, 20);
    let (w, h) = (
        usize::from(u16::from_be_bytes([b[0], b[1]])),
        usize::from(u16::from_be_bytes([b[2], b[3]])),
    );
    let be = u8::from(cfg!(target_endian = "big"));
    assert_eq!(b[4..20], [32, 24, be, 1, 0, 255, 0, 255, 0, 255, 16, 8, 0, 0, 0, 0]);
    let len = read_u32(s) as usize;
    assert_eq!(read_n(s, len), b"QEMU (test)");
    (w, h)
}

#[test]
fn handshake_and_update_over_loopback() {
    let hooks = Arc::new(TestHooks::default());
    let vd = open_str("127.0.0.1:101,to=899", "t-handshake", Arc::clone(&hooks))
        .unwrap_or_else(|e| panic!("{}", e.message()));
    let mut s = connect(&vd);
    assert_eq!(read_n(&mut s, 12), b"RFB 003.008\n");
    s.write_all(b"RFB 003.008\n").unwrap();
    assert_eq!(read_n(&mut s, 2), [1, 1]);
    s.write_all(&[1]).unwrap();
    assert_eq!(read_u32(&mut s), 0);
    let (w, h) = server_init(&mut s);
    assert_eq!((w, h), (640, 480));
    wait_for(&hooks, "initialized");

    // SetEncodings: raw only, then a full update request.
    let mut msg = vec![2, 0, 0, 1];
    msg.extend_from_slice(&ENCODING_RAW.to_be_bytes());
    msg.extend_from_slice(&[3, 0, 0, 0, 0, 0, 2, 128, 1, 224]);
    s.write_all(&msg).unwrap();
    let mut frame = vec![0u8; w * h * 4];
    let mut seen = vec![false; w * h];
    while seen.iter().any(|s| !s) {
        let hdr = read_n(&mut s, 4);
        assert_eq!(hdr[0], 0);
        for _ in 0..u16::from_be_bytes([hdr[2], hdr[3]]) {
            let rh = read_n(&mut s, 12);
            let field = |i: usize| usize::from(u16::from_be_bytes([rh[i], rh[i + 1]]));
            let (x, y, rw, rhh) = (field(0), field(2), field(4), field(6));
            assert_eq!(i32::from_be_bytes([rh[8], rh[9], rh[10], rh[11]]), ENCODING_RAW);
            for j in y..y + rhh {
                let row = read_n(&mut s, rw * 4);
                frame[(j * w + x) * 4..(j * w + x + rw) * 4].copy_from_slice(&row);
                seen[j * w + x..j * w + x + rw].fill(true);
            }
        }
    }
    let surface = DisplaySurface::placeholder(640, 480, NODEV_MSG);
    let pw = PixelWriter::server_default();
    for y in 0..h {
        let px: Vec<u32> = surface.image().row(y)[..w * 4]
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut want = Vec::new();
        pw.write(&mut want, &px);
        assert!(frame[y * w * 4..(y + 1) * w * 4] == want[..], "row {y}");
    }
    let servers = query_vnc_servers().unwrap();
    let me = servers.iter().find(|v| v.id == "t-handshake").unwrap();
    assert_eq!(me.clients.len(), 1);
    drop(s);
    wait_for(&hooks, "disconnected");
    assert_eq!(*hooks.0.lock().unwrap(), ["connected", "initialized", "disconnected"]);
}

/// An input handler that writes down what it gets.
struct Rec {
    mask: u32,
    log: Mutex<Vec<String>>,
}

impl crate::input::InputHandler for Rec {
    fn name(&self) -> &str {
        "rec"
    }

    fn mask(&self) -> u32 {
        self.mask
    }

    fn event(&self, _src: Option<&QemuConsole>, evt: &crate::input::QemuInputEvent) {
        use crate::input::QemuInputEvent as E;
        let s = match evt {
            E::Key { key, down } => format!("key {key} {down}"),
            E::Btn(b) => format!("btn {} {}", b.button.as_str(), b.down),
            E::Rel(m) => format!("rel {} {}", m.axis.as_str(), m.value),
            E::Abs(m) => format!("abs {} {}", m.axis.as_str(), m.value),
            E::Mtt(_) => "mtt".to_string(),
        };
        self.log.lock().unwrap().push(s);
    }

    fn sync(&self) {
        self.log.lock().unwrap().push("sync".to_string());
    }
}

/// Waits for `want` in the log of `rec` and clears it.
fn expect_log(rec: &Rec, want: &[&str]) {
    let start = Instant::now();
    while rec.log.lock().unwrap().len() < want.len() {
        assert!(start.elapsed() < Duration::from_secs(10), "log {:?}", rec.log.lock().unwrap());
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(std::mem::take(&mut *rec.log.lock().unwrap()), want);
}

#[test]
fn keys_and_pointer_reach_the_input_layer() {
    use crate::input::{INPUT_EVENT_MASK_BTN, INPUT_EVENT_MASK_KEY, INPUT_EVENT_MASK_REL};
    let input = InputState::new();
    let kbd = Arc::new(Rec { mask: INPUT_EVENT_MASK_KEY, log: Mutex::new(Vec::new()) });
    let mask = INPUT_EVENT_MASK_BTN | INPUT_EVENT_MASK_REL;
    let mouse = Arc::new(Rec { mask, log: Mutex::new(Vec::new()) });
    input.register(kbd.clone());
    input.register(mouse.clone());
    let hooks = Arc::new(TestHooks::default());
    let vd = open_with_input("127.0.0.1:102,to=899", "t-input", input, Arc::clone(&hooks))
        .unwrap_or_else(|e| panic!("{}", e.message()));
    let mut s = connect(&vd);
    read_n(&mut s, 12);
    s.write_all(b"RFB 003.008\n").unwrap();
    read_n(&mut s, 2);
    s.write_all(&[1]).unwrap();
    read_u32(&mut s);
    server_init(&mut s);
    let mut msg = vec![2, 0, 0, 1];
    msg.extend_from_slice(&ENCODING_RAW.to_be_bytes());
    s.write_all(&msg).unwrap();

    // "a" down and up, through the en-us layout.
    s.write_all(&[4, 1, 0, 0, 0, 0, 0, b'a', 4, 0, 0, 0, 0, 0, 0, b'a']).unwrap();
    expect_log(&kbd, &["key 30 true", "sync", "key 30 false", "sync"]);
    // "A" without shift held presses capslock first.
    s.write_all(&[4, 1, 0, 0, 0, 0, 0, b'A']).unwrap();
    let caps = ["key 58 true", "sync", "key 58 false", "sync"];
    expect_log(&kbd, &[&caps[..], &["key 30 true", "sync"]].concat());

    // The mouse is relative, so the first position only sets where the pointer is.
    s.write_all(&[5, 1, 0, 10, 0, 10]).unwrap();
    expect_log(&mouse, &["btn left true", "sync"]);
    s.write_all(&[5, 4, 0, 15, 0, 7]).unwrap();
    let want = ["btn left false", "btn right true", "rel x 5", "rel y -3", "sync"];
    expect_log(&mouse, &want);

    // The key still down goes up when the client leaves.
    drop(s);
    wait_for(&hooks, "disconnected");
    expect_log(&kbd, &["key 30 false", "sync"]);
}

#[test]
fn rfb_33_and_bad_versions() {
    let hooks = Arc::new(TestHooks::default());
    let vd = open_str("127.0.0.1:101,to=899", "t-33", Arc::clone(&hooks))
        .unwrap_or_else(|e| panic!("{}", e.message()));
    let mut s = connect(&vd);
    read_n(&mut s, 12);
    s.write_all(b"RFB 003.005\n").unwrap();
    assert_eq!(read_u32(&mut s), 1);
    server_init(&mut s);

    let mut s = connect(&vd);
    read_n(&mut s, 12);
    s.write_all(b"RFB 004.000\n").unwrap();
    assert_eq!(read_u32(&mut s), 0);
    let mut rest = Vec::new();
    s.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty());
}

#[test]
fn vnc_password_auth() {
    let hooks = Arc::new(TestHooks::default());
    let vd = open_str("127.0.0.1:101,to=899,password=on", "t-auth", Arc::clone(&hooks))
        .unwrap_or_else(|e| panic!("{}", e.message()));
    let opts = |pw: &str| SetPasswordOptions {
        password: pw.to_string(),
        connected: None,
        u: SetPasswordOptionsU::Vnc(SetPasswordOptionsVnc { display: Some("t-auth".to_string()) }),
    };
    qmp_set_password(opts("sesame")).unwrap();

    for (minor, pw, ok) in
        [(8, "sesame", true), (8, "wrong", false), (7, "wrong", false), (3, "sesame", true)]
    {
        let mut s = connect(&vd);
        read_n(&mut s, 12);
        s.write_all(format!("RFB 003.00{minor}\n").as_bytes()).unwrap();
        if minor == 3 {
            assert_eq!(read_u32(&mut s), 2);
        } else {
            assert_eq!(read_n(&mut s, 2), [1, 2]);
            s.write_all(&[2]).unwrap();
        }
        let challenge: [u8; 16] = read_n(&mut s, 16).try_into().unwrap();
        s.write_all(&auth::expected_response(pw.as_bytes(), &challenge).unwrap()).unwrap();
        if ok {
            assert_eq!(read_u32(&mut s), 0);
            server_init(&mut s);
        } else {
            assert_eq!(read_u32(&mut s), 1);
            if minor >= 8 {
                let len = read_u32(&mut s) as usize;
                assert_eq!(read_n(&mut s, len), b"Authentication failed\0");
            }
        }
    }

    // An expired password turns everybody away.
    qmp_expire_password(ExpirePasswordOptions {
        time: "now".to_string(),
        u: ExpirePasswordOptionsU::Vnc(ExpirePasswordOptionsVnc {
            display: Some("t-auth".to_string()),
        }),
    })
    .unwrap();
    let mut s = connect(&vd);
    read_n(&mut s, 12);
    s.write_all(b"RFB 003.008\n").unwrap();
    read_n(&mut s, 2);
    s.write_all(&[2]).unwrap();
    let challenge: [u8; 16] = read_n(&mut s, 16).try_into().unwrap();
    s.write_all(&auth::expected_response(b"sesame", &challenge).unwrap()).unwrap();
    assert_eq!(read_u32(&mut s), 1);

    let mut bad = opts("x");
    bad.connected = Some(SetPasswordAction::Fail);
    assert_eq!(
        qmp_set_password(bad).unwrap_err().message(),
        "parameter 'connected' must be 'keep' when 'protocol' is 'vnc'"
    );
    let e = qmp_expire_password(ExpirePasswordOptions {
        time: "+x".to_string(),
        u: ExpirePasswordOptionsU::Vnc(ExpirePasswordOptionsVnc {
            display: Some("t-auth".to_string()),
        }),
    })
    .unwrap_err();
    assert_eq!(e.message(), "Parameter 'time' doesn't take value '+x'");
    let e = qmp_set_password(SetPasswordOptions {
        password: "x".to_string(),
        connected: None,
        u: SetPasswordOptionsU::Vnc(SetPasswordOptionsVnc { display: Some("nope".to_string()) }),
    })
    .unwrap_err();
    assert_eq!(e.message(), "No VNC display is present");
}

#[test]
fn password_on_a_display_without_auth() {
    let hooks = Arc::new(TestHooks::default());
    assert!(open_str("none", "t-noauth", hooks).is_ok());
    let e = display_password(Some("t-noauth"), "x").unwrap_err();
    assert_eq!(e.message(), "VNC password authentication is disabled");
}

#[test]
fn option_errors() {
    let err = |arg: &str| {
        let hooks = Arc::new(TestHooks::default());
        match open_str(arg, "t-err", hooks) {
            Ok(_) => panic!("{arg} opened"),
            Err(e) => e.message().to_string(),
        }
    };
    assert_eq!(err("localhost"), "no vnc port specified");
    assert_eq!(err("localhost:"), "vnc port cannot be empty");
    assert_eq!(err(":x1"), "can't convert to a number: x1");
    assert_eq!(err(":59636"), "port 59636 out of range");
    assert_eq!(err(":-1"), "can't convert to a number: -1");
    assert_eq!(err("unix:/tmp/x,to=3"), "Port range not support with UNIX socket");
    assert_eq!(err(":0,share=foo"), "unknown vnc share= option");
    assert_eq!(err(":0,tls-authz=a"), "'tls-authz' provided but TLS is not enabled");
    assert_eq!(err(":0,sasl-authz=a"), "'sasl-authz' provided but SASL auth is not enabled");
    assert_eq!(
        err(":0,password=on,password-secret=s"),
        "'password' flag is redundant with 'password-secret'"
    );
    assert_eq!(err(":0,ipv4=off,ipv6=off"), "Cannot disable IPv4 and IPv6 at same time");
    assert_eq!(err(":0,display=nodev"), "Device 'nodev' not found");
    assert_eq!(err(":0,sasl=on"), "VNC SASL auth is not supported by ruvm yet");
    assert_eq!(err(":0,tls-creds=t"), "VNC TLS is not supported by ruvm yet");
    assert_eq!(err(":0,websocket=5700"), "VNC websocket is not supported by ruvm yet");
    assert_eq!(err(":0,audiodev=a"), "VNC audio is not supported by ruvm yet");
    assert_eq!(err(":0,reverse=on"), "VNC reverse connections are not supported by ruvm yet");
    // Without an address nothing listens, so reverse and websocket are never looked at.
    let hooks = Arc::new(TestHooks::default());
    assert!(open_str("none,reverse=on,websocket=1", "t-none", hooks).is_ok());
}

#[test]
fn bracketed_ipv6_host() {
    match opts::get_address("[::1]:3", 0, None, Some(true)).unwrap() {
        net::ListenAddr::Inet { host, port, .. } => {
            assert_eq!((host.as_str(), port), ("::1", 5903))
        }
        net::ListenAddr::Unix(_) => unreachable!(),
    }
}
