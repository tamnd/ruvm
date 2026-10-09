// SPDX-License-Identifier: GPL-2.0-or-later

//! `ramfb`, QEMU's `hw/display/ramfb.c` and `ramfb-standalone.c`: a framebuffer in guest RAM
//! that the firmware sets up through the fw_cfg file "etc/ramfb".
//!
//! The guest writes 28 big endian bytes with fw_cfg DMA: the address of the framebuffer, a DRM
//! fourcc, flags, width, height and stride. From then on the console shows that memory.
//!
//! Where this differs from QEMU: QEMU maps the guest memory and points the surface at it. Here
//! the surface owns its pixels and each update reads the framebuffer out of guest memory. A
//! configuration is rejected when that read fails, where QEMU rejects memory it cannot map.
//! The `use-legacy-x86-rom` property is the board code's, which adds the ROM to fw_cfg, and
//! there is no migration state.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use ruvm_base::error::{Error, Result};
use ruvm_hw_core::fw_cfg::{DmaMemory, FwCfgState};
use ruvm_ui::console::{ConsoleDevice, DisplayState, GraphicHwOps, QemuConsole};
use ruvm_ui::pixman::{PixelFormat, drm_format_to_pixman};
use ruvm_ui::surface::DisplaySurface;

use crate::vga::{VBE_DISPI_MAX_XRES, VBE_DISPI_MAX_YRES};

/// The fw_cfg file the guest writes the configuration to.
pub const RAMFB_FILE: &str = "etc/ramfb";

/// `sizeof(RAMFBCfg)`.
pub const RAMFB_CFG_SIZE: usize = 28;

/// A framebuffer the guest configured, what `ramfb_create_display_surface()` checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fb {
    addr: u64,
    format: PixelFormat,
    width: u32,
    height: u32,
    stride: u64,
    /// `stride * (height - 1) + linesize`: the bytes the surface covers.
    size: u64,
}

#[derive(Debug, Default)]
struct Inner {
    /// `RAMFBState::width` and `height`: zero until the guest set up a valid framebuffer.
    width: u32,
    height: u32,
    /// The framebuffer shown on the console.
    cur: Option<Fb>,
    /// A new framebuffer the next update switches to, `RAMFBState::ds`.
    pending: Option<Fb>,
    /// The raw configuration, `RAMFBState::cfg`.
    cfg: [u8; RAMFB_CFG_SIZE],
}

/// `RAMFBState`.
pub struct RamfbState {
    inner: Mutex<Inner>,
    mem: Arc<dyn DmaMemory>,
}

impl fmt::Debug for RamfbState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RamfbState").finish_non_exhaustive()
    }
}

fn be32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

impl RamfbState {
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// `ramfb_create_display_surface()`'s checks.
    fn check(
        &self,
        width: u32,
        height: u32,
        format: Option<PixelFormat>,
        stride: u64,
        addr: u64,
    ) -> Option<Fb> {
        let format = format?;
        if !(16..=u32::from(VBE_DISPI_MAX_XRES)).contains(&width)
            || !(16..=u32::from(VBE_DISPI_MAX_YRES)).contains(&height)
        {
            return None;
        }
        let linesize = u64::from(width) * u64::from(format.bpp()) / 8;
        let stride = if stride == 0 { linesize } else { stride };
        let size = stride * u64::from(height - 1) + linesize;
        // physical_memory_map() of the whole buffer.
        let mut probe = vec![0u8; usize::try_from(size).ok()?];
        if !self.mem.read(addr, &mut probe) {
            return None;
        }
        Some(Fb { addr, format, width, height, stride, size })
    }

    /// `ramfb_fw_cfg_write()`: the guest wrote the configuration.
    pub fn cfg_write(&self, cfg: &[u8]) {
        let mut g = self.lock();
        let n = cfg.len().min(RAMFB_CFG_SIZE);
        g.cfg[..n].copy_from_slice(&cfg[..n]);
        let c = g.cfg;
        drop(g);
        let addr = u64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        let fourcc = be32(&c, 8);
        let width = be32(&c, 16);
        let height = be32(&c, 20);
        let stride = u64::from(be32(&c, 24));
        let Some(fb) = self.check(width, height, drm_format_to_pixman(fourcc), stride, addr) else {
            return;
        };
        let mut g = self.lock();
        g.width = width;
        g.height = height;
        g.pending = Some(fb);
    }

    /// `ramfb_display_update()`.
    pub fn display_update(&self, con: &QemuConsole) {
        let mut g = self.lock();
        if g.width == 0 || g.height == 0 {
            return;
        }
        if let Some(fb) = g.pending.take() {
            con.set_surface(Some(DisplaySurface::new_from(
                fb.width as usize,
                fb.height as usize,
                fb.format,
                fb.stride as usize,
            )));
            g.cur = Some(fb);
        }
        let Some(fb) = g.cur else { return };
        drop(g);
        con.with_surface_mut(|surf| {
            let Some(surf) = surf else { return };
            let data = surf.data_mut();
            let len = (fb.size as usize).min(data.len());
            let _ = self.mem.read(fb.addr, &mut data[..len]);
        });
        // A simple full screen update.
        con.update_full();
    }
}

impl GraphicHwOps for RamfbState {
    fn gfx_update(&self, con: &QemuConsole) -> bool {
        self.display_update(con);
        true
    }
}

/// The `ramfb` device, `RAMFBStandaloneState`.
#[derive(Debug)]
pub struct Ramfb {
    state: Arc<RamfbState>,
}

impl Ramfb {
    /// `ramfb_realizefn()` and `ramfb_setup()`: makes the console in `ds` and adds "etc/ramfb"
    /// to `fw_cfg`. `mem` is the memory fw_cfg DMA and the framebuffer live in.
    pub fn realize(
        id: Option<String>,
        fw_cfg: Option<&FwCfgState>,
        mem: Arc<dyn DmaMemory>,
        ds: &DisplayState,
    ) -> Result<Ramfb> {
        let state = Arc::new(RamfbState { inner: Mutex::new(Inner::default()), mem });
        drop(ds.graphic_console_create(
            Some(ConsoleDevice { id, typename: "ramfb".to_string() }),
            0,
            Arc::clone(&state) as Arc<dyn GraphicHwOps>,
        ));
        let fw_cfg = match fw_cfg {
            Some(f) if f.dma_enabled() => f,
            _ => return Err(Error::generic("ramfb device requires fw_cfg with DMA".to_string())),
        };
        let s = Arc::clone(&state);
        fw_cfg.add_file_callback(
            RAMFB_FILE,
            None,
            Some(Box::new(move |data: &[u8], _off: u32, _len: u32| s.cfg_write(data))),
            vec![0; RAMFB_CFG_SIZE],
            false,
        )?;
        Ok(Ramfb { state })
    }

    /// The framebuffer state.
    pub fn state(&self) -> &Arc<RamfbState> {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvm_ui::pixman::DRM_FORMAT_XRGB8888;

    #[derive(Debug)]
    struct Ram(Mutex<Vec<u8>>);

    impl DmaMemory for Ram {
        fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
            let m = self.0.lock().unwrap();
            let a = addr as usize;
            match m.get(a..a + buf.len()) {
                Some(src) => {
                    buf.copy_from_slice(src);
                    true
                }
                None => false,
            }
        }

        fn write(&self, addr: u64, buf: &[u8]) -> bool {
            let mut m = self.0.lock().unwrap();
            let a = addr as usize;
            m[a..a + buf.len()].copy_from_slice(buf);
            true
        }
    }

    fn cfg(addr: u64, fourcc: u32, w: u32, h: u32, stride: u32) -> Vec<u8> {
        let mut v = addr.to_be_bytes().to_vec();
        for x in [fourcc, 0, w, h, stride] {
            v.extend_from_slice(&x.to_be_bytes());
        }
        v
    }

    #[test]
    fn shows_guest_memory_once_configured() {
        let ram = Arc::new(Ram(Mutex::new(vec![0; 1 << 20])));
        let state = Arc::new(RamfbState {
            inner: Mutex::new(Inner::default()),
            mem: Arc::clone(&ram) as Arc<dyn DmaMemory>,
        });
        let ds = DisplayState::new();
        let con = ds.graphic_console_create(None, 0, Arc::clone(&state) as Arc<dyn GraphicHwOps>);
        con.hw_update();
        con.with_surface(|s| assert!(s.unwrap().is_placeholder()));

        // Too small, then off the end of RAM: both ignored.
        state.cfg_write(&cfg(0x1000, DRM_FORMAT_XRGB8888, 8, 8, 0));
        state.cfg_write(&cfg(0xf_0000, DRM_FORMAT_XRGB8888, 640, 480, 0));
        con.hw_update();
        con.with_surface(|s| assert!(s.unwrap().is_placeholder()));

        ram.write(0x1000 + 4 * (32 * 2 + 1), &[0x10, 0x20, 0x30, 0]);
        state.cfg_write(&cfg(0x1000, DRM_FORMAT_XRGB8888, 32, 16, 0));
        con.hw_update();
        con.with_surface(|s| {
            let s = s.unwrap();
            assert_eq!((s.width(), s.height(), s.stride()), (32, 16, 128));
            assert_eq!(s.image().pixel(1, 2), 0x0030_2010);
        });
        // Later writes show up on the next update.
        ram.write(0x1000, &[0xff, 0, 0, 0]);
        con.hw_update();
        con.with_surface(|s| assert_eq!(s.unwrap().image().pixel(0, 0), 0xff));
    }
}
