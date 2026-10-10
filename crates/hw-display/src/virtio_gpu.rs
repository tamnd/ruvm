// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-gpu in 2D mode, a port of `hw/display/virtio-gpu.c` and `hw/display/virtio-gpu-base.c`.
//!
//! The device has a control queue of 64 entries and a cursor queue of 16. The guest creates 2D
//! resources (host side images), attaches guest pages to them as backing, copies rectangles from
//! the backing into the image with `TRANSFER_TO_HOST_2D`, points a scanout at a rectangle of a
//! resource with `SET_SCANOUT` and tells the UI what changed with `RESOURCE_FLUSH`. Each scanout
//! is a graphic console, one per `max_outputs`.
//!
//! The config space is `virtio_gpu_config`: `events_read`, `events_clear`, `num_scanouts`,
//! `num_capsets` and `blob_alignment`. When the UI asks for a new window size
//! ([`GraphicHwOps::ui_info`]) the device sets `VIRTIO_GPU_EVENT_DISPLAY` in `events_read` and
//! raises a config interrupt, which goes through the notifier set with
//! [`VirtioGpu::set_config_notifier`].
//!
//! Differences from QEMU:
//!
//! - QEMU's scanout surface is a view of the resource image, so a transfer shows on the console
//!   without a flush. Here the console surface is a copy of the scanout rectangle, refreshed on
//!   every transfer into the resource it shows, so a screendump sees the same pixels.
//! - Backing pages are not mapped. An entry is checked to be guest memory at its first and last
//!   byte when it is attached, and read from guest memory at transfer time.
//! - The cursor image is kept but not handed to the UI, which has no cursor define call yet. The
//!   cursor position goes to the console with `qemu_console_set_mouse()` as in QEMU.
//! - Commands run when the queue is kicked, not from a bottom half.
//! - `blob=on` fails with "need rutabaga or udmabuf for blob resources", as on a host without
//!   udmabuf. virgl, rutabaga, `hostmem`, `outputs` and the statistics are not ported.
//! - The guest errors QEMU logs with `qemu_log_mask(LOG_GUEST_ERROR)` are not printed.
//!
//! Migration (the `virtio-gpu` vmstate with the resources and scanouts) is not ported.

use std::any::Any;
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ruvm_base::{Error, Result, warn_report};
use ruvm_hw_virtio::{VirtIODevice, VirtioDeviceClass};
use ruvm_ui::console::{ConsoleDevice, DisplayState, GraphicHwOps, QemuConsole, QemuUiInfo};
use ruvm_ui::pixman::{self, Image, PixelFormat};
use ruvm_ui::surface::DisplaySurface;
use ruvm_virtio_queue::GuestMemory;

use crate::edid::{EdidInfo, edid_generate};

/// `TYPE_VIRTIO_GPU`.
pub const TYPE_VIRTIO_GPU: &str = "virtio-gpu-device";
/// The name of the virtio PCI function carrying the device.
pub const TYPE_VIRTIO_GPU_PCI: &str = "virtio-gpu-pci";
/// `VIRTIO_ID_GPU`.
pub const VIRTIO_ID_GPU: u16 = 16;
/// `VIRTIO_GPU_MAX_SCANOUTS`.
pub const VIRTIO_GPU_MAX_SCANOUTS: u32 = 16;
/// `VIRTIO_GPU_F_EDID`.
pub const VIRTIO_GPU_F_EDID: u32 = 1;
/// `VIRTIO_GPU_EVENT_DISPLAY`.
pub const VIRTIO_GPU_EVENT_DISPLAY: u32 = 1;
/// `sizeof(struct virtio_gpu_config)`.
pub const VIRTIO_GPU_CONFIG_SIZE: usize = 20;
/// `VIRTIO_GPU_FLAG_FENCE`.
pub const VIRTIO_GPU_FLAG_FENCE: u32 = 1;

pub const VIRTIO_GPU_CMD_GET_DISPLAY_INFO: u32 = 0x100;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_2D: u32 = 0x101;
pub const VIRTIO_GPU_CMD_RESOURCE_UNREF: u32 = 0x102;
pub const VIRTIO_GPU_CMD_SET_SCANOUT: u32 = 0x103;
pub const VIRTIO_GPU_CMD_RESOURCE_FLUSH: u32 = 0x104;
pub const VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D: u32 = 0x105;
pub const VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING: u32 = 0x106;
pub const VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING: u32 = 0x107;
pub const VIRTIO_GPU_CMD_GET_EDID: u32 = 0x10a;
pub const VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB: u32 = 0x10c;
pub const VIRTIO_GPU_CMD_SET_SCANOUT_BLOB: u32 = 0x10d;
pub const VIRTIO_GPU_CMD_UPDATE_CURSOR: u32 = 0x300;
pub const VIRTIO_GPU_CMD_MOVE_CURSOR: u32 = 0x301;

pub const VIRTIO_GPU_RESP_OK_NODATA: u32 = 0x1100;
pub const VIRTIO_GPU_RESP_OK_DISPLAY_INFO: u32 = 0x1101;
pub const VIRTIO_GPU_RESP_OK_EDID: u32 = 0x1104;
pub const VIRTIO_GPU_RESP_ERR_UNSPEC: u32 = 0x1200;
pub const VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY: u32 = 0x1201;
pub const VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID: u32 = 0x1202;
pub const VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID: u32 = 0x1203;
pub const VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER: u32 = 0x1205;

pub const VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM: u32 = 1;
pub const VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM: u32 = 2;
pub const VIRTIO_GPU_FORMAT_A8R8G8B8_UNORM: u32 = 3;
pub const VIRTIO_GPU_FORMAT_X8R8G8B8_UNORM: u32 = 4;
pub const VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM: u32 = 67;
pub const VIRTIO_GPU_FORMAT_X8B8G8R8_UNORM: u32 = 68;
pub const VIRTIO_GPU_FORMAT_A8B8G8R8_UNORM: u32 = 121;
pub const VIRTIO_GPU_FORMAT_R8G8B8X8_UNORM: u32 = 134;

const CTRL_QUEUE_SIZE: u16 = 64;
const CURSOR_QUEUE_SIZE: u16 = 16;
/// `sizeof(struct virtio_gpu_ctrl_hdr)`.
const HDR_SIZE: usize = 24;
const MAX_MEM_ENTRIES: u32 = 16384;
const MEM_ENTRY_SIZE: usize = 16;
/// `sizeof(struct virtio_gpu_resource_attach_backing)`.
const ATTACH_BACKING_SIZE: usize = 32;
/// The most of a control request any command reads: attach_backing with the most entries.
const MAX_REQUEST: u64 = (ATTACH_BACKING_SIZE + MAX_MEM_ENTRIES as usize * MEM_ENTRY_SIZE) as u64;
/// `sizeof(struct virtio_gpu_update_cursor)`.
const CURSOR_CMD_SIZE: usize = 56;
/// `sizeof(struct virtio_gpu_resp_display_info)`.
const DISPLAY_INFO_SIZE: usize = HDR_SIZE + VIRTIO_GPU_MAX_SCANOUTS as usize * 24;
/// The EDID blob of `struct virtio_gpu_resp_edid`.
const EDID_BLOB_SIZE: usize = 1024;
/// The cursor size, `cursor_alloc(64, 64)`.
const CURSOR_DIM: usize = 64;

/// `virtio_gpu_get_pixman_format()` on a little endian host.
pub fn virtio_gpu_pixman_format(format: u32) -> Option<PixelFormat> {
    Some(match format {
        VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM => pixman::X8R8G8B8,
        VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM => pixman::A8R8G8B8,
        VIRTIO_GPU_FORMAT_X8R8G8B8_UNORM => pixman::B8G8R8X8,
        VIRTIO_GPU_FORMAT_A8R8G8B8_UNORM => pixman::B8G8R8A8,
        VIRTIO_GPU_FORMAT_R8G8B8X8_UNORM => pixman::X8B8G8R8,
        VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM => pixman::A8B8G8R8,
        VIRTIO_GPU_FORMAT_X8B8G8R8_UNORM => pixman::R8G8B8X8,
        VIRTIO_GPU_FORMAT_A8B8G8R8_UNORM => pixman::R8G8B8A8,
        _ => return None,
    })
}

/// The virtio-gpu properties that this port has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VirtioGpuConf {
    /// `max_outputs`: the number of scanouts, at most 16.
    pub max_outputs: u32,
    /// `edid`: offer `VIRTIO_GPU_F_EDID`.
    pub edid: bool,
    /// `xres` and `yres`: the size the first output asks for.
    pub xres: u32,
    pub yres: u32,
    /// `max_hostmem`: the most memory the resource images may take.
    pub max_hostmem: u64,
    /// `blob`: blob resources, which need udmabuf and are refused.
    pub blob: bool,
}

impl Default for VirtioGpuConf {
    fn default() -> Self {
        VirtioGpuConf {
            max_outputs: 1,
            edid: true,
            xres: 1280,
            yres: 800,
            max_hostmem: 256 << 20,
            blob: false,
        }
    }
}

/// `struct virtio_gpu_requested_state`: what the UI asked for on one output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ReqState {
    width_mm: u16,
    height_mm: u16,
    width: u32,
    height: u32,
    refresh_rate: u32,
    x: i32,
    y: i32,
}

/// The part of `VirtIOGPUBase` the UI side changes.
#[derive(Debug, Default)]
struct Shared {
    req_state: [ReqState; VIRTIO_GPU_MAX_SCANOUTS as usize],
    enabled_output_bitmask: u32,
    events_read: u32,
    /// A config interrupt is due that could not be raised yet.
    config_pending: bool,
}

type ConfigNotifier = Box<dyn Fn() + Send + Sync>;

/// `virtio_gpu_ops`, the console side of the device.
struct GpuOps {
    shared: Arc<Mutex<Shared>>,
    max_outputs: u32,
    notifier: Mutex<Option<ConfigNotifier>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

impl GraphicHwOps for GpuOps {
    // virtio_gpu_invalidate_display(), virtio_gpu_update_display() and virtio_gpu_text_update()
    // do nothing, which the defaults also do.

    /// `virtio_gpu_ui_info()` with `virtio_gpu_notify_event()`.
    fn ui_info(&self, head: u32, info: &QemuUiInfo) {
        if head >= self.max_outputs {
            return;
        }
        {
            let mut s = lock(&self.shared);
            let r = &mut s.req_state[head as usize];
            r.x = info.xoff;
            r.y = info.yoff;
            r.refresh_rate = info.refresh_rate;
            r.width = info.width;
            r.height = info.height;
            r.width_mm = info.width_mm;
            r.height_mm = info.height_mm;
            if info.width != 0 && info.height != 0 {
                s.enabled_output_bitmask |= 1 << head;
            } else {
                s.enabled_output_bitmask &= !(1 << head);
            }
            s.events_read |= VIRTIO_GPU_EVENT_DISPLAY;
            s.config_pending = true;
        }
        if let Some(notify) = lock(&self.notifier).as_ref() {
            notify();
        }
    }

    fn has_ui_info(&self) -> bool {
        true
    }
}

/// `struct virtio_gpu_rect`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

/// `struct virtio_gpu_framebuffer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Framebuffer {
    format: PixelFormat,
    width: u32,
    height: u32,
    stride: u32,
    offset: u32,
}

/// `struct virtio_gpu_simple_resource` for a 2D resource.
#[derive(Debug)]
struct Resource {
    id: u32,
    image: Image,
    hostmem: u64,
    /// The backing entries as (guest address, length). `None` is QEMU's NULL `iov`.
    backing: Option<Vec<(u64, u64)>>,
    scanout_bitmask: u32,
}

/// `QEMUCursor`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub hot_x: u32,
    pub hot_y: u32,
    /// 64 by 64 pixels, `a8r8g8b8`.
    pub data: Vec<u32>,
}

/// `struct virtio_gpu_update_cursor`, the fields used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CursorCmd {
    kind: u32,
    scanout_id: u32,
    x: u32,
    y: u32,
    resource_id: u32,
    hot_x: u32,
    hot_y: u32,
}

/// `struct virtio_gpu_scanout`.
#[derive(Debug)]
struct Scanout {
    con: QemuConsole,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    resource_id: u32,
    /// `ds`: the console surface this device set, as the resource and offset its pixels come
    /// from. QEMU compares the surface's data pointer, which is the same thing.
    ds: Option<(u32, u32)>,
    /// The cursor position from the last cursor command, `cursor.pos`.
    cursor_pos: (u32, u32),
    current_cursor: Option<Cursor>,
}

/// QEMU's `QemuRect`, which has 16 bit fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QemuRect {
    x: i16,
    y: i16,
    width: u16,
    height: u16,
}

impl QemuRect {
    fn new(x: u32, y: u32, width: u32, height: u32) -> QemuRect {
        QemuRect { x: x as i16, y: y as i16, width: width as u16, height: height as u16 }
    }

    /// `qemu_rect_intersect()`.
    fn intersect(&self, b: &QemuRect) -> Option<QemuRect> {
        let x1 = self.x.max(b.x);
        let y1 = self.y.max(b.y);
        let x2 = (i32::from(self.x) + i32::from(self.width))
            .min(i32::from(b.x) + i32::from(b.width)) as i16;
        let y2 = (i32::from(self.y) + i32::from(self.height))
            .min(i32::from(b.y) + i32::from(b.height)) as i16;
        if x1 >= x2 || y1 >= y2 {
            return None;
        }
        Some(QemuRect {
            x: x1,
            y: y1,
            width: (i32::from(x2) - i32::from(x1)) as u16,
            height: (i32::from(y2) - i32::from(y1)) as u16,
        })
    }
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from(le32(b, off)) | u64::from(le32(b, off + 4)) << 32
}

fn put32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn rect_at(b: &[u8], off: usize) -> Rect {
    Rect {
        x: le32(b, off),
        y: le32(b, off + 4),
        width: le32(b, off + 8),
        height: le32(b, off + 12),
    }
}

/// `VIRTIO_GPU_FILL_CMD`: the first `n` bytes of the request, or `ERR_INVALID_PARAMETER`.
fn fill(req: &[u8], n: usize) -> std::result::Result<&[u8], u32> {
    req.get(..n).ok_or(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER)
}

/// A response header of `len` bytes with type `kind`.
fn response(kind: u32, len: usize) -> Vec<u8> {
    let mut v = vec![0; len];
    put32(&mut v, 0, kind);
    v
}

/// `iov_to_buf()` over the backing entries: copies from byte `offset` of the backing into
/// `dst` and returns how much was copied.
fn iov_to_buf(mem: &dyn GuestMemory, iov: &[(u64, u64)], offset: u64, dst: &mut [u8]) -> usize {
    let mut offset = offset;
    let mut done = 0;
    for &(addr, len) in iov {
        if done == dst.len() {
            break;
        }
        if offset >= len {
            offset -= len;
            continue;
        }
        let n = usize::try_from(len - offset).unwrap_or(usize::MAX).min(dst.len() - done);
        if mem.read(addr + offset, &mut dst[done..done + n]).is_err() {
            break;
        }
        done += n;
        offset = 0;
    }
    done
}

/// The bounds check of transfer and flush, with QEMU's 32 bit sums.
fn outside(r: &Rect, width: u32, height: u32) -> bool {
    r.x > width
        || r.y > height
        || r.width > width
        || r.height > height
        || r.x.wrapping_add(r.width) > width
        || r.y.wrapping_add(r.height) > height
}

/// The virtio-gpu device model, `VirtIOGPU` with its `VirtIOGPUBase`.
pub struct VirtioGpu {
    conf: VirtioGpuConf,
    display: Arc<DisplayState>,
    dev: ConsoleDevice,
    ops: Arc<GpuOps>,
    resources: Vec<Resource>,
    hostmem: u64,
    scanouts: Vec<Scanout>,
    enable: bool,
}

impl fmt::Debug for VirtioGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtioGpu")
            .field("conf", &self.conf)
            .field("resources", &self.resources.len())
            .field("enable", &self.enable)
            .finish_non_exhaustive()
    }
}

impl VirtioGpu {
    /// A device with properties `conf` whose consoles belong to `dev` in `display`. The
    /// properties are checked and the consoles made when the device is realized.
    pub fn new(conf: VirtioGpuConf, display: Arc<DisplayState>, dev: ConsoleDevice) -> Self {
        let ops = Arc::new(GpuOps {
            shared: Arc::new(Mutex::new(Shared::default())),
            max_outputs: conf.max_outputs,
            notifier: Mutex::new(None),
        });
        VirtioGpu {
            conf,
            display,
            dev,
            ops,
            resources: Vec::new(),
            hostmem: 0,
            scanouts: Vec::new(),
            enable: false,
        }
    }

    /// The properties.
    pub fn conf(&self) -> VirtioGpuConf {
        self.conf
    }

    /// Sets what raises a config interrupt from outside the device, normally a call to
    /// [`VirtioGpu::config_notify`] through the transport.
    pub fn set_config_notifier(&self, notifier: Option<Box<dyn Fn() + Send + Sync>>) {
        *lock(&self.ops.notifier) = notifier;
    }

    /// Raises the config interrupt a UI size change asked for, if one is due.
    pub fn config_notify(&mut self, vdev: &mut VirtIODevice) {
        let due = std::mem::take(&mut lock(&self.ops.shared).config_pending);
        if due {
            vdev.notify_config();
        }
    }

    /// The consoles of the scanouts, in order.
    pub fn consoles(&self) -> Vec<QemuConsole> {
        self.scanouts.iter().map(|s| s.con.clone()).collect()
    }

    /// The cursor last defined on `scanout`.
    pub fn cursor(&self, scanout: usize) -> Option<&Cursor> {
        self.scanouts.get(scanout)?.current_cursor.as_ref()
    }

    /// Where the last cursor command on `scanout` put the cursor.
    pub fn cursor_position(&self, scanout: usize) -> Option<(u32, u32)> {
        Some(self.scanouts.get(scanout)?.cursor_pos)
    }

    /// `g->parent_obj.enable`: a scanout was set since the last reset.
    pub fn enabled(&self) -> bool {
        self.enable
    }

    fn find(&self, id: u32) -> Option<usize> {
        self.resources.iter().position(|r| r.id == id)
    }

    /// `virtio_gpu_find_check_resource()`.
    fn find_check(&self, id: u32, require_backing: bool) -> std::result::Result<usize, u32> {
        let i = self.find(id).ok_or(VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID)?;
        if require_backing && self.resources[i].backing.is_none() {
            return Err(VIRTIO_GPU_RESP_ERR_UNSPEC);
        }
        Ok(i)
    }

    /// `virtio_gpu_simple_process_cmd()` with `virtio_gpu_ctrl_response()`: runs the request
    /// `req` and gives the response.
    fn process_cmd(&mut self, mem: &dyn GuestMemory, req: &[u8]) -> Vec<u8> {
        let Ok(hdr) = fill(req, HDR_SIZE) else {
            return response(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER, HDR_SIZE);
        };
        let kind = le32(hdr, 0);
        let flags = le32(hdr, 4);
        let result = match kind {
            VIRTIO_GPU_CMD_GET_DISPLAY_INFO => Ok(Some(self.get_display_info())),
            VIRTIO_GPU_CMD_GET_EDID => self.get_edid(req).map(Some),
            VIRTIO_GPU_CMD_RESOURCE_CREATE_2D => self.resource_create_2d(req),
            VIRTIO_GPU_CMD_RESOURCE_UNREF => self.resource_unref(req),
            VIRTIO_GPU_CMD_RESOURCE_FLUSH => self.resource_flush(req),
            VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D => self.transfer_to_host_2d(mem, req),
            VIRTIO_GPU_CMD_SET_SCANOUT => self.set_scanout(req),
            VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING => self.resource_attach_backing(mem, req),
            VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING => self.resource_detach_backing(req),
            // Blob resources are never enabled.
            VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB | VIRTIO_GPU_CMD_SET_SCANOUT_BLOB => {
                Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER)
            }
            _ => Err(VIRTIO_GPU_RESP_ERR_UNSPEC),
        };
        let mut resp = match result {
            Ok(Some(resp)) => resp,
            Ok(None) => response(VIRTIO_GPU_RESP_OK_NODATA, HDR_SIZE),
            Err(e) => response(e, HDR_SIZE),
        };
        if flags & VIRTIO_GPU_FLAG_FENCE != 0 {
            let resp_flags = le32(&resp, 4) | VIRTIO_GPU_FLAG_FENCE;
            put32(&mut resp, 4, resp_flags);
            resp[8..16].copy_from_slice(&hdr[8..16]);
            resp[16..20].copy_from_slice(&hdr[16..20]);
        }
        resp
    }

    /// `virtio_gpu_get_display_info()` with `virtio_gpu_base_fill_display_info()`.
    fn get_display_info(&self) -> Vec<u8> {
        let mut resp = response(VIRTIO_GPU_RESP_OK_DISPLAY_INFO, DISPLAY_INFO_SIZE);
        let s = lock(&self.ops.shared);
        for i in 0..self.conf.max_outputs as usize {
            if s.enabled_output_bitmask & (1 << i) != 0 {
                let p = HDR_SIZE + i * 24;
                put32(&mut resp, p + 8, s.req_state[i].width);
                put32(&mut resp, p + 12, s.req_state[i].height);
                put32(&mut resp, p + 16, 1);
            }
        }
        resp
    }

    /// `virtio_gpu_get_edid()` with `virtio_gpu_base_generate_edid()`.
    fn get_edid(&self, req: &[u8]) -> std::result::Result<Vec<u8>, u32> {
        let c = fill(req, 32)?;
        let scanout = le32(c, 24);
        if scanout >= self.conf.max_outputs {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
        let r = lock(&self.ops.shared).req_state[scanout as usize];
        let mut info = EdidInfo {
            width_mm: r.width_mm,
            height_mm: r.height_mm,
            prefx: r.width,
            prefy: r.height,
            refresh_rate: r.refresh_rate,
            ..EdidInfo::default()
        };
        let mut resp = response(VIRTIO_GPU_RESP_OK_EDID, HDR_SIZE + 8 + EDID_BLOB_SIZE);
        put32(&mut resp, HDR_SIZE, EDID_BLOB_SIZE as u32);
        edid_generate(&mut resp[HDR_SIZE + 8..], &mut info);
        Ok(resp)
    }

    /// `virtio_gpu_resource_create_2d()`.
    fn resource_create_2d(&mut self, req: &[u8]) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 40)?;
        let (id, format, width, height) = (le32(c, 24), le32(c, 28), le32(c, 32), le32(c, 36));
        if id == 0 || self.find(id).is_some() {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID);
        }
        let pformat =
            virtio_gpu_pixman_format(format).ok_or(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER)?;
        // calc_image_hostmem()
        let bpp = u64::from(pformat.bpp());
        let stride = ((u64::from(width) * bpp + 0x1f) >> 5) * 4;
        let size = u64::from(height) * stride;
        if size > u64::from(u32::MAX) || size + self.hostmem >= self.conf.max_hostmem {
            return Err(VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY);
        }
        if size == 0 {
            // qemu_pixman_image_new_shareable() cannot map an empty memfd.
            warn_report("failed to allocate shared memory: Invalid argument");
            return Err(VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY);
        }
        let image = Image::new(pformat, width as usize, height as usize, stride as usize);
        self.resources.push(Resource {
            id,
            image,
            hostmem: size,
            backing: None,
            scanout_bitmask: 0,
        });
        self.hostmem += size;
        Ok(None)
    }

    /// `virtio_gpu_disable_scanout()`.
    fn disable_scanout(&mut self, scanout_id: usize) {
        let id = self.scanouts[scanout_id].resource_id;
        if id == 0 {
            return;
        }
        if let Some(i) = self.find(id) {
            self.resources[i].scanout_bitmask &= !(1 << scanout_id);
        }
        let s = &mut self.scanouts[scanout_id];
        s.con.set_surface(None);
        s.resource_id = 0;
        s.ds = None;
        s.width = 0;
        s.height = 0;
    }

    /// `virtio_gpu_resource_destroy()`.
    fn resource_destroy(&mut self, index: usize) {
        let mask = self.resources[index].scanout_bitmask;
        if mask != 0 {
            for i in 0..self.conf.max_outputs as usize {
                if mask & (1 << i) != 0 {
                    self.disable_scanout(i);
                }
            }
        }
        let res = self.resources.remove(index);
        self.hostmem -= res.hostmem;
    }

    /// `virtio_gpu_resource_unref()`.
    fn resource_unref(&mut self, req: &[u8]) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 32)?;
        let i = self.find(le32(c, 24)).ok_or(VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID)?;
        self.resource_destroy(i);
        Ok(None)
    }

    /// `virtio_gpu_transfer_to_host_2d()`.
    fn transfer_to_host_2d(
        &mut self,
        mem: &dyn GuestMemory,
        req: &[u8],
    ) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 56)?;
        let r = rect_at(c, 24);
        let offset = le64(c, 40);
        let index = self.find_check(le32(c, 48), true)?;
        let res = &mut self.resources[index];
        let (width, height) = (res.image.width() as u32, res.image.height() as u32);
        if outside(&r, width, height) {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
        let bpp = res.image.format().bpp().div_ceil(8);
        let stride = res.image.stride() as u32;
        let iov = res.backing.as_deref().unwrap_or(&[]);
        let data = res.image.data_mut();
        // The bytes of the image that may have changed.
        let dirty = if r.x != 0 || r.width != width {
            for h in 0..r.height {
                // QEMU's offsets are 32 bit.
                let src = (offset as u32).wrapping_add(stride.wrapping_mul(h));
                let dst = ((r.y + h) * stride + r.x * bpp) as usize;
                let len = (r.width * bpp) as usize;
                iov_to_buf(mem, iov, u64::from(src), &mut data[dst..dst + len]);
            }
            (r.y * stride + r.x * bpp, (r.y + r.height) * stride)
        } else {
            let dst = (r.y * stride + r.x * bpp) as usize;
            let len = (stride * r.height) as usize;
            iov_to_buf(mem, iov, u64::from(offset as u32), &mut data[dst..dst + len]);
            (dst as u32, dst as u32 + len as u32)
        };
        self.refresh_surfaces(index, dirty);
        Ok(None)
    }

    /// Copies the bytes `dirty` of resource `index` to the console surfaces showing it, which
    /// in QEMU are views of the resource image.
    fn refresh_surfaces(&mut self, index: usize, dirty: (u32, u32)) {
        if dirty.1 <= dirty.0 {
            return;
        }
        let res = &self.resources[index];
        let stride = res.image.stride() as u32;
        let first = dirty.0 / stride;
        let last = (dirty.1 - 1) / stride;
        for s in &self.scanouts {
            let Some((id, offset)) = s.ds else { continue };
            if id != res.id {
                continue;
            }
            let top = offset / stride;
            let from = first.max(top);
            let to = last.min(top + s.height - 1);
            if from > to {
                continue;
            }
            s.con.with_surface_mut(|surface| {
                let Some(surface) = surface else { return };
                if surface.width() != s.width as usize || surface.height() != s.height as usize {
                    return;
                }
                copy_rows(&res.image, offset, s.width, from - top..to - top + 1, surface);
            });
        }
    }

    /// `virtio_gpu_resource_flush()` for a 2D resource.
    fn resource_flush(&mut self, req: &[u8]) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 48)?;
        let r = rect_at(c, 24);
        let res = &self.resources[self.find_check(le32(c, 40), false)?];
        if outside(&r, res.image.width() as u32, res.image.height() as u32) {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
        let flush = QemuRect::new(r.x, r.y, r.width, r.height);
        for (i, s) in self.scanouts.iter().enumerate() {
            if res.scanout_bitmask & (1 << i) == 0 {
                continue;
            }
            let rect = QemuRect::new(s.x, s.y, s.width, s.height);
            if let Some(mut rect) = flush.intersect(&rect) {
                rect.x = rect.x.wrapping_sub(s.x as i16);
                rect.y = rect.y.wrapping_sub(s.y as i16);
                s.con.update(
                    i32::from(rect.x),
                    i32::from(rect.y),
                    i32::from(rect.width),
                    i32::from(rect.height),
                );
            }
        }
        Ok(None)
    }

    /// `virtio_gpu_update_scanout()`.
    fn update_scanout(&mut self, scanout_id: usize, index: usize, r: &Rect) {
        let old = self.scanouts[scanout_id].resource_id;
        if let Some(o) = self.find(old) {
            self.resources[o].scanout_bitmask &= !(1 << scanout_id);
        }
        let res = &mut self.resources[index];
        res.scanout_bitmask |= 1 << scanout_id;
        let s = &mut self.scanouts[scanout_id];
        s.resource_id = res.id;
        s.x = r.x;
        s.y = r.y;
        s.width = r.width;
        s.height = r.height;
    }

    /// `virtio_gpu_set_scanout()` with `virtio_gpu_do_set_scanout()`.
    fn set_scanout(&mut self, req: &[u8]) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 48)?;
        let r = rect_at(c, 24);
        let (scanout_id, resource_id) = (le32(c, 40), le32(c, 44));
        if scanout_id >= self.conf.max_outputs {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID);
        }
        let scanout_id = scanout_id as usize;
        if resource_id == 0 {
            self.disable_scanout(scanout_id);
            return Ok(None);
        }
        let index = self.find_check(resource_id, true)?;
        let image = &self.resources[index].image;
        let bytes_pp = image.format().bpp().div_ceil(8);
        let stride = image.stride() as u32;
        let fb = Framebuffer {
            format: image.format(),
            width: image.width() as u32,
            height: image.height() as u32,
            stride,
            offset: r.x.wrapping_mul(bytes_pp).wrapping_add(r.y.wrapping_mul(stride)),
        };
        // virtio_gpu_check_scanout_bounds()
        if r.width < 16
            || r.height < 16
            || u64::from(r.x) + u64::from(r.width) > u64::from(fb.width)
            || u64::from(r.y) + u64::from(r.height) > u64::from(fb.height)
        {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
        if u64::from(fb.stride) < u64::from(fb.width) * u64::from(bytes_pp)
            || fb.stride > i32::MAX as u32
        {
            return Err(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
        self.enable = true;
        let s = &self.scanouts[scanout_id];
        if s.ds != Some((resource_id, fb.offset)) || s.width != r.width || s.height != r.height {
            let mut surface = DisplaySurface::new_from(
                r.width as usize,
                r.height as usize,
                fb.format,
                stride as usize,
            );
            copy_rows(image, fb.offset, r.width, 0..r.height, &mut surface);
            let s = &mut self.scanouts[scanout_id];
            s.ds = Some((resource_id, fb.offset));
            s.con.set_surface(Some(surface));
        }
        self.update_scanout(scanout_id, index, &r);
        Ok(None)
    }

    /// `virtio_gpu_resource_attach_backing()` with `virtio_gpu_create_mapping_iov()`.
    fn resource_attach_backing(
        &mut self,
        mem: &dyn GuestMemory,
        req: &[u8],
    ) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, ATTACH_BACKING_SIZE)?;
        let (id, nr_entries) = (le32(c, 24), le32(c, 28));
        let index = self.find(id).ok_or(VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID)?;
        if self.resources[index].backing.is_some() || nr_entries > MAX_MEM_ENTRIES {
            return Err(VIRTIO_GPU_RESP_ERR_UNSPEC);
        }
        let len = nr_entries as usize * MEM_ENTRY_SIZE;
        let ents = req
            .get(ATTACH_BACKING_SIZE..ATTACH_BACKING_SIZE + len)
            .ok_or(VIRTIO_GPU_RESP_ERR_UNSPEC)?;
        let mut iov = Vec::with_capacity(nr_entries as usize);
        for e in ents.chunks_exact(MEM_ENTRY_SIZE) {
            let (addr, len) = (le64(e, 0), u64::from(le32(e, 8)));
            // dma_memory_map() fails for an empty range and for one that is not memory.
            let mut b = [0u8];
            let mapped = len != 0
                && addr.checked_add(len - 1).is_some()
                && mem.read(addr, &mut b).is_ok()
                && mem.read(addr + len - 1, &mut b).is_ok();
            if !mapped {
                return Err(VIRTIO_GPU_RESP_ERR_UNSPEC);
            }
            iov.push((addr, len));
        }
        // With no entries QEMU's iov stays NULL, which is no backing.
        self.resources[index].backing = (!iov.is_empty()).then_some(iov);
        Ok(None)
    }

    /// `virtio_gpu_resource_detach_backing()`.
    fn resource_detach_backing(&mut self, req: &[u8]) -> std::result::Result<Option<Vec<u8>>, u32> {
        let c = fill(req, 32)?;
        let index = self.find_check(le32(c, 24), true)?;
        self.resources[index].backing = None;
        Ok(None)
    }

    /// `update_cursor()` with `virtio_gpu_update_cursor_data()`.
    fn update_cursor(&mut self, b: &[u8]) {
        let cmd = CursorCmd {
            kind: le32(b, 0),
            scanout_id: le32(b, 24),
            x: le32(b, 28),
            y: le32(b, 32),
            resource_id: le32(b, 40),
            hot_x: le32(b, 44),
            hot_y: le32(b, 48),
        };
        if cmd.scanout_id >= self.conf.max_outputs {
            return;
        }
        let i = cmd.scanout_id as usize;
        if cmd.kind != VIRTIO_GPU_CMD_MOVE_CURSOR {
            let data = (cmd.resource_id > 0)
                .then(|| self.find(cmd.resource_id))
                .flatten()
                .map(|r| &self.resources[r].image)
                .filter(|img| img.width() == CURSOR_DIM && img.height() == CURSOR_DIM)
                .map(|img| {
                    img.data()
                        .chunks_exact(4)
                        .take(CURSOR_DIM * CURSOR_DIM)
                        .map(|p| u32::from_ne_bytes([p[0], p[1], p[2], p[3]]))
                        .collect::<Vec<u32>>()
                });
            let s = &mut self.scanouts[i];
            let cursor = s.current_cursor.get_or_insert_with(|| Cursor {
                hot_x: 0,
                hot_y: 0,
                data: vec![0; CURSOR_DIM * CURSOR_DIM],
            });
            cursor.hot_x = cmd.hot_x;
            cursor.hot_y = cmd.hot_y;
            if let Some(data) = data {
                cursor.data = data;
            }
        }
        // A full update stores the whole command, a move only the position.
        self.scanouts[i].cursor_pos = (cmd.x, cmd.y);
        self.scanouts[i].con.set_mouse(cmd.x as i32, cmd.y as i32, cmd.resource_id != 0);
    }

    /// `virtio_gpu_handle_ctrl()` with `virtio_gpu_process_cmdq()`.
    fn handle_ctrl(&mut self, vdev: &mut VirtIODevice) {
        if !vdev.queue_ready(0) {
            return;
        }
        let mem = Arc::clone(vdev.mem());
        while let Some(chain) = vdev.pop(0) {
            let len = chain.readable_len().min(MAX_REQUEST) as usize;
            let mut req = vec![0; len];
            let n = chain.reader(&*mem).read(&mut req).unwrap_or(0);
            req.truncate(n);
            let resp = self.process_cmd(&*mem, &req);
            let written = chain.writer(&*mem).write(&resp).unwrap_or(0);
            vdev.push(0, &chain, written as u32);
            vdev.notify(0);
        }
    }

    /// `virtio_gpu_handle_cursor()`.
    fn handle_cursor(&mut self, vdev: &mut VirtIODevice) {
        if !vdev.queue_ready(1) {
            return;
        }
        let mem = Arc::clone(vdev.mem());
        while let Some(chain) = vdev.pop(1) {
            let mut b = [0u8; CURSOR_CMD_SIZE];
            if chain.reader(&*mem).read(&mut b).unwrap_or(0) == CURSOR_CMD_SIZE {
                self.update_cursor(&b);
            }
            vdev.push(1, &chain, 0);
            vdev.notify(1);
        }
    }
}

/// Copies the rows `rows` of a `width` pixel wide rectangle starting `offset` bytes into
/// `image` to the same rows of `surface`, which has the image's stride.
fn copy_rows(
    image: &Image,
    offset: u32,
    width: u32,
    rows: std::ops::Range<u32>,
    surface: &mut DisplaySurface,
) {
    let stride = image.stride();
    let len = width as usize * image.format().bpp().div_ceil(8) as usize;
    let src = image.data();
    let dst = surface.data_mut();
    for row in rows {
        let s = offset as usize + row as usize * stride;
        let d = row as usize * stride;
        if s + len <= src.len() && d + len <= dst.len() {
            dst[d..d + len].copy_from_slice(&src[s..s + len]);
        }
    }
}

impl VirtioDeviceClass for VirtioGpu {
    /// `virtio_gpu_device_realize()` with `virtio_gpu_base_device_realize()`.
    fn realize(&mut self, vdev: &mut VirtIODevice) -> Result<()> {
        if self.conf.blob {
            return Err(Error::generic("need rutabaga or udmabuf for blob resources"));
        }
        if self.conf.max_outputs > VIRTIO_GPU_MAX_SCANOUTS {
            return Err(Error::generic(format!("invalid max_outputs > {VIRTIO_GPU_MAX_SCANOUTS}")));
        }
        {
            let mut s = lock(&self.ops.shared);
            s.enabled_output_bitmask = 1;
            s.req_state[0].width = self.conf.xres;
            s.req_state[0].height = self.conf.yres;
        }
        vdev.init(TYPE_VIRTIO_GPU, VIRTIO_ID_GPU, VIRTIO_GPU_CONFIG_SIZE);
        vdev.add_queue(CTRL_QUEUE_SIZE)?;
        vdev.add_queue(CURSOR_QUEUE_SIZE)?;
        let ops: Arc<dyn GraphicHwOps> = self.ops.clone();
        self.scanouts = (0..self.conf.max_outputs)
            .map(|i| {
                let con = self.display.graphic_console_create(
                    Some(self.dev.clone()),
                    i,
                    Arc::clone(&ops),
                );
                // The system reset that follows in QEMU puts up the "not active" placeholder.
                con.set_surface(None);
                Scanout {
                    con,
                    width: 0,
                    height: 0,
                    x: 0,
                    y: 0,
                    resource_id: 0,
                    ds: None,
                    cursor_pos: (0, 0),
                    current_cursor: None,
                }
            })
            .collect();
        Ok(())
    }

    /// `virtio_gpu_base_get_features()`.
    fn get_features(&mut self, _vdev: &VirtIODevice, features: u64) -> Result<u64> {
        let edid = if self.conf.edid { 1 << VIRTIO_GPU_F_EDID } else { 0 };
        Ok(features | edid)
    }

    /// `virtio_gpu_get_config()`: `events_clear` always reads 0.
    fn get_config(&mut self, _vdev: &VirtIODevice, config: &mut [u8]) {
        let events_read = lock(&self.ops.shared).events_read;
        config[..VIRTIO_GPU_CONFIG_SIZE].fill(0);
        put32(config, 0, events_read);
        put32(config, 8, self.conf.max_outputs);
    }

    /// `virtio_gpu_set_config()`.
    fn set_config(&mut self, _vdev: &mut VirtIODevice, config: &mut [u8]) {
        let clear = le32(config, 4);
        if clear != 0 {
            lock(&self.ops.shared).events_read &= !clear;
        }
    }

    /// `virtio_gpu_reset()`: the resources go, every console shows "Display output is not
    /// active." and the scanouts are cleared. What the UI asked for and the cursors stay.
    fn reset(&mut self, _vdev: &mut VirtIODevice) {
        while !self.resources.is_empty() {
            self.resource_destroy(0);
        }
        for s in &self.scanouts {
            s.con.set_surface(None);
        }
        // virtio_gpu_base_reset()
        self.enable = false;
        for s in &mut self.scanouts {
            s.resource_id = 0;
            s.width = 0;
            s.height = 0;
            s.x = 0;
            s.y = 0;
            s.ds = None;
        }
    }

    fn handle_output(&mut self, vdev: &mut VirtIODevice, queue: u16) {
        self.config_notify(vdev);
        match queue {
            0 => self.handle_ctrl(vdev),
            1 => self.handle_cursor(vdev),
            _ => {}
        }
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
    use ruvm_virtio_queue::VecMemory;

    const MIB: usize = 1 << 20;

    fn gpu(conf: VirtioGpuConf) -> (VirtioGpu, VirtIODevice, Arc<DisplayState>) {
        let ds = DisplayState::new();
        let dev = ConsoleDevice { id: Some("gpu0".into()), typename: TYPE_VIRTIO_GPU_PCI.into() };
        let mut g = VirtioGpu::new(conf, Arc::clone(&ds), dev);
        let mut vdev = VirtIODevice::new(Arc::new(VecMemory::new(MIB)));
        g.realize(&mut vdev).unwrap();
        (g, vdev, ds)
    }

    fn hdr(kind: u32, len: usize) -> Vec<u8> {
        let mut v = vec![0; len];
        put32(&mut v, 0, kind);
        v
    }

    fn cmd(kind: u32, words: &[u32]) -> Vec<u8> {
        let mut v = hdr(kind, HDR_SIZE + words.len() * 4);
        for (i, w) in words.iter().enumerate() {
            put32(&mut v, HDR_SIZE + i * 4, *w);
        }
        v
    }

    fn run(g: &mut VirtioGpu, mem: &VecMemory, req: &[u8]) -> u32 {
        le32(&g.process_cmd(mem, req), 0)
    }

    fn create(g: &mut VirtioGpu, mem: &VecMemory, id: u32, format: u32, w: u32, h: u32) -> u32 {
        run(g, mem, &cmd(VIRTIO_GPU_CMD_RESOURCE_CREATE_2D, &[id, format, w, h]))
    }

    fn attach(g: &mut VirtioGpu, mem: &VecMemory, id: u32, ents: &[(u64, u32)]) -> u32 {
        let mut v = cmd(VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING, &[id, ents.len() as u32]);
        for &(a, l) in ents {
            v.extend_from_slice(&a.to_le_bytes());
            v.extend_from_slice(&l.to_le_bytes());
            v.extend_from_slice(&[0; 4]);
        }
        run(g, mem, &v)
    }

    fn transfer(g: &mut VirtioGpu, mem: &VecMemory, id: u32, r: [u32; 4], offset: u64) -> u32 {
        let mut v = cmd(VIRTIO_GPU_CMD_TRANSFER_TO_HOST_2D, &r);
        v.extend_from_slice(&offset.to_le_bytes());
        v.extend_from_slice(&id.to_le_bytes());
        v.extend_from_slice(&[0; 4]);
        run(g, mem, &v)
    }

    fn set_scanout(g: &mut VirtioGpu, mem: &VecMemory, scanout: u32, id: u32, r: [u32; 4]) -> u32 {
        run(g, mem, &cmd(VIRTIO_GPU_CMD_SET_SCANOUT, &[r[0], r[1], r[2], r[3], scanout, id]))
    }

    fn pixel(con: &QemuConsole, x: usize, y: usize) -> u32 {
        con.with_surface(|s| s.unwrap().image().pixel(x, y))
    }

    fn placeholder(con: &QemuConsole) -> bool {
        con.with_surface(|s| s.unwrap().is_placeholder())
    }

    #[test]
    fn realize_checks_and_sets_up() {
        let ds = DisplayState::new();
        let dev = ConsoleDevice { id: None, typename: TYPE_VIRTIO_GPU.into() };
        let conf = VirtioGpuConf { max_outputs: 17, ..VirtioGpuConf::default() };
        let mut g = VirtioGpu::new(conf, Arc::clone(&ds), dev.clone());
        let mut vdev = VirtIODevice::new(Arc::new(VecMemory::new(4096)));
        let e = g.realize(&mut vdev).unwrap_err();
        assert_eq!(e.to_string(), "invalid max_outputs > 16");
        let conf = VirtioGpuConf { blob: true, ..VirtioGpuConf::default() };
        let mut g = VirtioGpu::new(conf, ds, dev);
        let e = g.realize(&mut vdev).unwrap_err();
        assert_eq!(e.to_string(), "need rutabaga or udmabuf for blob resources");

        let (mut g, vdev, ds) = gpu(VirtioGpuConf { max_outputs: 2, ..VirtioGpuConf::default() });
        assert_eq!(vdev.device_id(), VIRTIO_ID_GPU);
        assert_eq!(vdev.num_queues(), 2);
        assert_eq!(vdev.queue_num_max(0), 64);
        assert_eq!(vdev.queue_num_max(1), 16);
        assert_eq!(ds.consoles().len(), 2);
        assert!(placeholder(&g.consoles()[1]));
        assert_eq!(g.get_features(&vdev, 0).unwrap(), 1 << VIRTIO_GPU_F_EDID);
        let mut config = [0xffu8; 20];
        g.get_config(&vdev, &mut config);
        assert_eq!(config, [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn display_info_edid_and_ui_info() {
        let (mut g, mut vdev, _ds) = gpu(VirtioGpuConf { max_outputs: 2, ..Default::default() });
        let mem = VecMemory::new(4096);
        let resp = g.process_cmd(&mem, &hdr(VIRTIO_GPU_CMD_GET_DISPLAY_INFO, HDR_SIZE));
        assert_eq!(resp.len(), 408);
        assert_eq!(le32(&resp, 0), VIRTIO_GPU_RESP_OK_DISPLAY_INFO);
        assert_eq!((le32(&resp, 32), le32(&resp, 36), le32(&resp, 40)), (1280, 800, 1));
        assert_eq!(le32(&resp, 24 + 24 + 16), 0);

        let resp = g.process_cmd(&mem, &cmd(VIRTIO_GPU_CMD_GET_EDID, &[0, 0]));
        assert_eq!(
            (le32(&resp, 0), le32(&resp, 24), resp.len()),
            (VIRTIO_GPU_RESP_OK_EDID, 1024, 1056)
        );
        assert_eq!(&resp[32..40], &[0, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0]);
        let sum = resp[32..160].iter().fold(0u8, |a, b| a.wrapping_add(*b));
        assert_eq!(sum, 0);
        assert_eq!(
            run(&mut g, &mem, &cmd(VIRTIO_GPU_CMD_GET_EDID, &[2, 0])),
            VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER
        );

        let con = g.consoles()[1].clone();
        assert!(con.set_ui_info(QemuUiInfo { width: 800, height: 600, ..Default::default() }));
        let resp = g.process_cmd(&mem, &hdr(VIRTIO_GPU_CMD_GET_DISPLAY_INFO, HDR_SIZE));
        assert_eq!(
            (le32(&resp, 48 + 8), le32(&resp, 48 + 12), le32(&resp, 48 + 16)),
            (800, 600, 1)
        );
        let mut config = [0u8; 20];
        g.get_config(&vdev, &mut config);
        assert_eq!(le32(&config, 0), VIRTIO_GPU_EVENT_DISPLAY);
        put32(&mut config, 4, VIRTIO_GPU_EVENT_DISPLAY);
        g.set_config(&mut vdev, &mut config);
        g.get_config(&vdev, &mut config);
        assert_eq!((le32(&config, 0), le32(&config, 4)), (0, 0));
        assert!(lock(&g.ops.shared).config_pending);
        g.config_notify(&mut vdev);
        assert!(!lock(&g.ops.shared).config_pending);
        // A zero size turns the output off.
        con.set_ui_info(QemuUiInfo::default());
        let resp = g.process_cmd(&mem, &hdr(VIRTIO_GPU_CMD_GET_DISPLAY_INFO, HDR_SIZE));
        assert_eq!(le32(&resp, 48 + 16), 0);
    }

    #[test]
    fn header_errors_and_fences() {
        let (mut g, _vdev, _ds) = gpu(VirtioGpuConf::default());
        let mem = VecMemory::new(4096);
        let resp = g.process_cmd(&mem, &[0; 10]);
        assert_eq!(resp, response(VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER, HDR_SIZE));
        let mut req = hdr(0x999, HDR_SIZE);
        put32(&mut req, 4, VIRTIO_GPU_FLAG_FENCE);
        req[8..16].copy_from_slice(&0x1122_3344_5566_7788u64.to_le_bytes());
        put32(&mut req, 16, 7);
        req[20] = 3;
        let resp = g.process_cmd(&mem, &req);
        assert_eq!(le32(&resp, 0), VIRTIO_GPU_RESP_ERR_UNSPEC);
        assert_eq!(le32(&resp, 4), VIRTIO_GPU_FLAG_FENCE);
        assert_eq!(le64(&resp, 8), 0x1122_3344_5566_7788);
        assert_eq!((le32(&resp, 16), resp[20]), (7, 0));
        // A short command still carries the fence.
        put32(&mut req, 0, VIRTIO_GPU_CMD_RESOURCE_CREATE_2D);
        let resp = g.process_cmd(&mem, &req);
        assert_eq!((le32(&resp, 0), le32(&resp, 4)), (VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER, 1));
        for blob in [VIRTIO_GPU_CMD_RESOURCE_CREATE_BLOB, VIRTIO_GPU_CMD_SET_SCANOUT_BLOB] {
            assert_eq!(run(&mut g, &mem, &hdr(blob, 64)), VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        }
    }

    #[test]
    fn resource_lifecycle_errors() {
        let (mut g, _vdev, _ds) =
            gpu(VirtioGpuConf { max_hostmem: 64 * 64 * 4 * 2 + 1, ..Default::default() });
        let mem = VecMemory::new(MIB);
        let ok = VIRTIO_GPU_RESP_OK_NODATA;
        assert_eq!(create(&mut g, &mem, 0, 2, 64, 64), VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID);
        assert_eq!(create(&mut g, &mem, 1, 5, 64, 64), VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER);
        assert_eq!(create(&mut g, &mem, 1, 2, 0, 64), VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY);
        assert_eq!(create(&mut g, &mem, 1, 2, 64, 64), ok);
        assert_eq!(create(&mut g, &mem, 1, 2, 64, 64), VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID);
        assert_eq!(create(&mut g, &mem, 2, 2, 64, 64), ok);
        assert_eq!(g.hostmem, 2 * 64 * 64 * 4);
        assert_eq!(create(&mut g, &mem, 3, 2, 1, 1), VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY);
        assert_eq!(create(&mut g, &mem, 4, 2, 0x10000, 0x10000), VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY);

        assert_eq!(attach(&mut g, &mem, 9, &[]), VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID);
        assert_eq!(attach(&mut g, &mem, 1, &[(0x1000, 0)]), VIRTIO_GPU_RESP_ERR_UNSPEC);
        assert_eq!(attach(&mut g, &mem, 1, &[(MIB as u64 - 4, 8)]), VIRTIO_GPU_RESP_ERR_UNSPEC);
        let mut short = cmd(VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING, &[1, 2]);
        short.extend_from_slice(&[0; 16]);
        assert_eq!(run(&mut g, &mem, &short), VIRTIO_GPU_RESP_ERR_UNSPEC);
        let big = cmd(VIRTIO_GPU_CMD_RESOURCE_ATTACH_BACKING, &[1, 16385]);
        assert_eq!(run(&mut g, &mem, &big), VIRTIO_GPU_RESP_ERR_UNSPEC);
        // No entries leaves the resource without backing.
        assert_eq!(attach(&mut g, &mem, 1, &[]), ok);
        let detach = cmd(VIRTIO_GPU_CMD_RESOURCE_DETACH_BACKING, &[1, 0]);
        assert_eq!(run(&mut g, &mem, &detach), VIRTIO_GPU_RESP_ERR_UNSPEC);
        assert_eq!(transfer(&mut g, &mem, 1, [0, 0, 64, 64], 0), VIRTIO_GPU_RESP_ERR_UNSPEC);
        assert_eq!(attach(&mut g, &mem, 1, &[(0x1000, 0x4000)]), ok);
        assert_eq!(attach(&mut g, &mem, 1, &[(0x1000, 0x4000)]), VIRTIO_GPU_RESP_ERR_UNSPEC);
        assert_eq!(
            transfer(&mut g, &mem, 1, [1, 0, 64, 1], 0),
            VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER
        );
        assert_eq!(
            transfer(&mut g, &mem, 7, [0, 0, 1, 1], 0),
            VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID
        );
        assert_eq!(run(&mut g, &mem, &detach), ok);
        assert_eq!(run(&mut g, &mem, &detach), VIRTIO_GPU_RESP_ERR_UNSPEC);

        let unref = |id| cmd(VIRTIO_GPU_CMD_RESOURCE_UNREF, &[id, 0]);
        assert_eq!(run(&mut g, &mem, &unref(1)), ok);
        assert_eq!(run(&mut g, &mem, &unref(1)), VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID);
        assert_eq!(g.hostmem, 64 * 64 * 4);
    }

    #[test]
    fn scanout_transfer_and_flush() {
        let (mut g, mut vdev, _ds) = gpu(VirtioGpuConf { max_outputs: 2, ..Default::default() });
        let mem = VecMemory::new(MIB);
        let ok = VIRTIO_GPU_RESP_OK_NODATA;
        let con = g.consoles()[0].clone();
        // 64x32 B8G8R8X8, backed by two entries, pixel (x, y) = 0x00yyxx01.
        for y in 0..32u32 {
            for x in 0..64u32 {
                let p = 0x0001 | x << 8 | y << 16;
                mem.write(0x10000 + u64::from(y * 256 + x * 4), &p.to_le_bytes()).unwrap();
            }
        }
        assert_eq!(create(&mut g, &mem, 5, VIRTIO_GPU_FORMAT_B8G8R8X8_UNORM, 64, 32), ok);
        assert_eq!(attach(&mut g, &mem, 5, &[(0x10000, 0x1000), (0x11000, 0x1000)]), ok);

        assert_eq!(
            set_scanout(&mut g, &mem, 2, 5, [0, 0, 64, 32]),
            VIRTIO_GPU_RESP_ERR_INVALID_SCANOUT_ID
        );
        assert_eq!(
            set_scanout(&mut g, &mem, 0, 5, [0, 0, 15, 32]),
            VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER
        );
        assert_eq!(
            set_scanout(&mut g, &mem, 0, 5, [8, 0, 64, 32]),
            VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER
        );
        assert!(!g.enabled());
        assert_eq!(set_scanout(&mut g, &mem, 0, 5, [8, 4, 32, 16]), ok);
        assert!(g.enabled());
        assert!(!placeholder(&con));
        assert_eq!((con.width(0), con.height(0)), (32, 16));
        assert_eq!(pixel(&con, 0, 0), 0);
        // A transfer shows without a flush, as in QEMU where the surface is a view.
        assert_eq!(transfer(&mut g, &mem, 5, [0, 0, 64, 32], 0), ok);
        assert_eq!(pixel(&con, 0, 0), 0x0004_0801);
        assert_eq!(pixel(&con, 31, 15), 0x0013_2701);
        // A sub-rectangle transfer reads rows a stride apart from the offset.
        mem.write(0x10000, &0xabcd_ef01u32.to_le_bytes()).unwrap();
        assert_eq!(transfer(&mut g, &mem, 5, [9, 5, 1, 1], 0), ok);
        assert_eq!(pixel(&con, 1, 1), 0xabcd_ef01);
        // Short backing leaves the rest alone.
        assert_eq!(transfer(&mut g, &mem, 5, [0, 0, 64, 32], 0x1ff0), ok);
        assert_eq!(pixel(&con, 0, 0), 0x0004_0801);

        let flush =
            |r: [u32; 4], id| cmd(VIRTIO_GPU_CMD_RESOURCE_FLUSH, &[r[0], r[1], r[2], r[3], id, 0]);
        assert_eq!(run(&mut g, &mem, &flush([0, 0, 64, 32], 5)), ok);
        assert_eq!(
            run(&mut g, &mem, &flush([0, 0, 65, 32], 5)),
            VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER
        );
        assert_eq!(
            run(&mut g, &mem, &flush([0, 0, 1, 1], 6)),
            VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID
        );

        // The same resource on the second head, then unref turns both off.
        assert_eq!(set_scanout(&mut g, &mem, 1, 5, [0, 0, 64, 32]), ok);
        let con1 = g.consoles()[1].clone();
        assert_eq!(pixel(&con1, 63, 31), 0x001f_3f01);
        assert_eq!(g.resources[0].scanout_bitmask, 3);
        assert_eq!(run(&mut g, &mem, &cmd(VIRTIO_GPU_CMD_RESOURCE_UNREF, &[5, 0])), ok);
        assert!(placeholder(&con) && placeholder(&con1));
        assert_eq!((con.width(0), con.height(0)), (32, 16));

        // Disabling a scanout, and reset.
        assert_eq!(create(&mut g, &mem, 6, VIRTIO_GPU_FORMAT_R8G8B8A8_UNORM, 64, 32), ok);
        assert_eq!(attach(&mut g, &mem, 6, &[(0x10000, 0x2000)]), ok);
        assert_eq!(set_scanout(&mut g, &mem, 0, 6, [0, 0, 64, 32]), ok);
        assert_eq!(con.with_surface(|s| s.unwrap().format()), pixman::A8B8G8R8);
        assert_eq!(set_scanout(&mut g, &mem, 0, 0, [0; 4]), ok);
        assert!(placeholder(&con));
        assert_eq!(set_scanout(&mut g, &mem, 0, 6, [0, 0, 64, 32]), ok);
        g.reset(&mut vdev);
        assert!(placeholder(&con) && g.resources.is_empty() && !g.enabled());
        assert_eq!((g.hostmem, g.scanouts[0].resource_id), (0, 0));
    }

    #[test]
    fn cursor_updates() {
        let (mut g, _vdev, _ds) = gpu(VirtioGpuConf::default());
        let mem = VecMemory::new(MIB);
        assert_eq!(create(&mut g, &mem, 1, VIRTIO_GPU_FORMAT_B8G8R8A8_UNORM, 64, 64), 0x1100);
        g.resources[0].image.set_pixel(0, 0, 0xff00_ff00);
        let mut c = cmd(VIRTIO_GPU_CMD_UPDATE_CURSOR, &[0, 10, 20, 0, 1, 3, 4, 0]);
        g.update_cursor(&c);
        let cur = g.cursor(0).unwrap();
        assert_eq!((cur.hot_x, cur.hot_y, cur.data[0], cur.data.len()), (3, 4, 0xff00_ff00, 4096));
        put32(&mut c, 0, VIRTIO_GPU_CMD_MOVE_CURSOR);
        put32(&mut c, 28, 30);
        g.update_cursor(&c);
        assert_eq!(g.cursor_position(0), Some((30, 20)));
        put32(&mut c, 24, 1);
        put32(&mut c, 28, 50);
        g.update_cursor(&c);
        assert_eq!(g.cursor_position(0), Some((30, 20)));
    }

    #[test]
    fn rect_intersection_is_16_bit() {
        let a = QemuRect::new(0, 0, 100, 100);
        let b = QemuRect::new(50, 60, 100, 100);
        assert_eq!(a.intersect(&b), Some(QemuRect { x: 50, y: 60, width: 50, height: 40 }));
        assert_eq!(a.intersect(&QemuRect::new(100, 0, 10, 10)), None);
        assert_eq!(QemuRect::new(0x10005, 0, 1, 1), QemuRect { x: 5, y: 0, width: 1, height: 1 });
    }
}
