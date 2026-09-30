// SPDX-License-Identifier: GPL-2.0-or-later

//! The virtio-mmio transport, a port of `hw/virtio/virtio-mmio.c`.
//!
//! [`VirtioMmio`] is a 0x200 byte register window implementing [`MmioOps`], with one interrupt
//! line. It speaks the legacy (version 1) register layout when `force-legacy` is on, which is
//! QEMU's default, and the virtio 1.0 (version 2) layout when it is off. A transport can exist
//! with no device behind it: the magic value, version and vendor ID read back as usual and
//! every other register reads as zero, which makes Linux skip the slot.
//!
//! Differences from QEMU:
//!
//! - The register window is always little endian. QEMU makes the legacy window native endian,
//!   which only matters on big endian targets.
//! - Guest errors that QEMU logs under `LOG_GUEST_ERROR` (bad access sizes, writes to read-only
//!   registers, legacy-only registers in modern mode and the reverse) are ignored silently.
//! - A register access from inside the device's own queue processing (a DMA that lands back on
//!   this window) is dropped, the way QEMU's memory re-entrancy guard drops it.
//!
//! Not ported: VMState, trace points, QOM registration and properties other than
//! `force-legacy`, ioeventfd and irqfd (every notify is handled synchronously in the vCPU
//! thread), `VIRTIO_F_NOTIFICATION_DATA` (the shadow available index in the notify value is
//! ignored), shared memory regions (`SHM_LEN` reads as all ones, meaning no region), and the
//! device tree and ACPI glue that boards use to describe the window.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

use ruvm_base::Result;
use ruvm_hw_core::IrqPin;
use ruvm_mem::{AccessCtx, AccessSize, MemResult, MmioOps};

use crate::virtio::{
    VIRTIO_CONFIG_S_FEATURES_OK, VIRTIO_F_VERSION_1, VIRTIO_QUEUE_MAX, VirtIODevice, VirtioBackend,
    VirtioDeviceClass, VirtioTransport, feature,
};

/// `TYPE_VIRTIO_MMIO`.
pub const TYPE_VIRTIO_MMIO: &str = "virtio-mmio";

/// Size of the register window.
pub const VIRTIO_MMIO_REGION_SIZE: u64 = 0x200;

/// `VIRT_MAGIC`: "virt" as a little endian word.
pub const VIRT_MAGIC: u32 = 0x7472_6976;
/// `VIRT_VERSION`: the virtio 1.0 register layout.
pub const VIRT_VERSION: u32 = 2;
/// `VIRT_VERSION_LEGACY`: the legacy register layout.
pub const VIRT_VERSION_LEGACY: u32 = 1;
/// `VIRT_VENDOR`: "QEMU" as a little endian word.
pub const VIRT_VENDOR: u32 = 0x554d_4551;

/// QEMU's default for the `force-legacy` property.
pub const VIRTIO_MMIO_FORCE_LEGACY_DEFAULT: bool = true;

/// Magic value register.
pub const VIRTIO_MMIO_MAGIC_VALUE: u64 = 0x000;
/// Version register.
pub const VIRTIO_MMIO_VERSION: u64 = 0x004;
/// Device ID register.
pub const VIRTIO_MMIO_DEVICE_ID: u64 = 0x008;
/// Vendor ID register.
pub const VIRTIO_MMIO_VENDOR_ID: u64 = 0x00c;
/// Device features, 32 bits at a time.
pub const VIRTIO_MMIO_DEVICE_FEATURES: u64 = 0x010;
/// Selects which 32 bits `DEVICE_FEATURES` shows.
pub const VIRTIO_MMIO_DEVICE_FEATURES_SEL: u64 = 0x014;
/// Driver features, 32 bits at a time.
pub const VIRTIO_MMIO_DRIVER_FEATURES: u64 = 0x020;
/// Selects which 32 bits `DRIVER_FEATURES` sets.
pub const VIRTIO_MMIO_DRIVER_FEATURES_SEL: u64 = 0x024;
/// Legacy: the guest page size, the unit of `QUEUE_PFN`.
pub const VIRTIO_MMIO_GUEST_PAGE_SIZE: u64 = 0x028;
/// Selects the queue the queue registers refer to.
pub const VIRTIO_MMIO_QUEUE_SEL: u64 = 0x030;
/// Largest size the selected queue supports.
pub const VIRTIO_MMIO_QUEUE_NUM_MAX: u64 = 0x034;
/// Size of the selected queue.
pub const VIRTIO_MMIO_QUEUE_NUM: u64 = 0x038;
/// Legacy: used ring alignment of the selected queue.
pub const VIRTIO_MMIO_QUEUE_ALIGN: u64 = 0x03c;
/// Legacy: page number of the selected queue.
pub const VIRTIO_MMIO_QUEUE_PFN: u64 = 0x040;
/// Modern: the selected queue is ready.
pub const VIRTIO_MMIO_QUEUE_READY: u64 = 0x044;
/// Queue notifier.
pub const VIRTIO_MMIO_QUEUE_NOTIFY: u64 = 0x050;
/// Interrupt status.
pub const VIRTIO_MMIO_INTERRUPT_STATUS: u64 = 0x060;
/// Interrupt acknowledge.
pub const VIRTIO_MMIO_INTERRUPT_ACK: u64 = 0x064;
/// Device status.
pub const VIRTIO_MMIO_STATUS: u64 = 0x070;
/// Modern: descriptor table address, low half.
pub const VIRTIO_MMIO_QUEUE_DESC_LOW: u64 = 0x080;
/// Modern: descriptor table address, high half.
pub const VIRTIO_MMIO_QUEUE_DESC_HIGH: u64 = 0x084;
/// Modern: available ring address, low half.
pub const VIRTIO_MMIO_QUEUE_AVAIL_LOW: u64 = 0x090;
/// Modern: available ring address, high half.
pub const VIRTIO_MMIO_QUEUE_AVAIL_HIGH: u64 = 0x094;
/// Modern: used ring address, low half.
pub const VIRTIO_MMIO_QUEUE_USED_LOW: u64 = 0x0a0;
/// Modern: used ring address, high half.
pub const VIRTIO_MMIO_QUEUE_USED_HIGH: u64 = 0x0a4;
/// Shared memory region select.
pub const VIRTIO_MMIO_SHM_SEL: u64 = 0x0ac;
/// Shared memory region length, low half.
pub const VIRTIO_MMIO_SHM_LEN_LOW: u64 = 0x0b0;
/// Shared memory region length, high half.
pub const VIRTIO_MMIO_SHM_LEN_HIGH: u64 = 0x0b4;
/// Shared memory region base, low half.
pub const VIRTIO_MMIO_SHM_BASE_LOW: u64 = 0x0b8;
/// Shared memory region base, high half.
pub const VIRTIO_MMIO_SHM_BASE_HIGH: u64 = 0x0bc;
/// Modern: config space generation.
pub const VIRTIO_MMIO_CONFIG_GENERATION: u64 = 0x0fc;
/// Start of the device config space.
pub const VIRTIO_MMIO_CONFIG: u64 = 0x100;

/// Interrupt status bit: a queue has used buffers.
pub const VIRTIO_MMIO_INT_VRING: u32 = 1 << 0;
/// Interrupt status bit: the config space changed.
pub const VIRTIO_MMIO_INT_CONFIG: u32 = 1 << 1;

/// The transport side of the interrupt: one level triggered line that is high while any
/// interrupt status bit is set, `virtio_mmio_update_irq()`.
#[derive(Debug)]
struct MmioIrq {
    pin: Arc<IrqPin>,
}

impl VirtioTransport for MmioIrq {
    fn notify(&self, _vector: u16, isr: u8) {
        self.pin.set_bool(isr != 0);
    }
}

/// `VirtIOMMIOQueue`: what a modern driver programs before setting `QUEUE_READY`.
#[derive(Clone, Copy, Debug, Default)]
struct MmioQueue {
    num: u16,
    enabled: bool,
    desc: [u32; 2],
    avail: [u32; 2],
    used: [u32; 2],
}

fn join(halves: [u32; 2]) -> u64 {
    u64::from(halves[1]) << 32 | u64::from(halves[0])
}

#[derive(Debug)]
struct MmioState {
    backend: Option<VirtioBackend>,
    host_features_sel: u32,
    guest_features_sel: u32,
    guest_page_shift: u32,
    guest_features: [u32; 2],
    vqs: Vec<MmioQueue>,
}

/// A virtio-mmio transport, `VirtIOMMIOProxy`.
pub struct VirtioMmio {
    legacy: bool,
    irq: Arc<IrqPin>,
    state: Mutex<MmioState>,
    owner: Mutex<Option<ThreadId>>,
}

impl fmt::Debug for VirtioMmio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VirtioMmio").field("legacy", &self.legacy).finish_non_exhaustive()
    }
}

/// Holds the state lock and remembers which thread holds it, so a nested access from the same
/// thread can be refused instead of deadlocking.
struct StateGuard<'a> {
    state: MutexGuard<'a, MmioState>,
    owner: &'a Mutex<Option<ThreadId>>,
}

impl Deref for StateGuard<'_> {
    type Target = MmioState;
    fn deref(&self) -> &MmioState {
        &self.state
    }
}

impl DerefMut for StateGuard<'_> {
    fn deref_mut(&mut self) -> &mut MmioState {
        &mut self.state
    }
}

impl Drop for StateGuard<'_> {
    fn drop(&mut self) {
        *self.owner.lock().unwrap_or_else(PoisonError::into_inner) = None;
    }
}

impl VirtioMmio {
    /// Creates the transport and plugs `backend` into it, if there is one.
    ///
    /// `force_legacy` is the `force-legacy` property. With it off the device is offered
    /// `VIRTIO_F_VERSION_1`, as `virtio_mmio_pre_plugged()` does.
    pub fn new(backend: Option<VirtioBackend>, force_legacy: bool) -> Result<Self> {
        let irq = Arc::new(IrqPin::new());
        let mut backend = backend;
        if let Some(b) = backend.as_mut() {
            let transport_features = if force_legacy { 0 } else { feature(VIRTIO_F_VERSION_1) };
            b.plug(Arc::new(MmioIrq { pin: Arc::clone(&irq) }), transport_features)?;
        }
        Ok(VirtioMmio {
            legacy: force_legacy,
            irq,
            state: Mutex::new(MmioState {
                backend,
                host_features_sel: 0,
                guest_features_sel: 0,
                guest_page_shift: 0,
                guest_features: [0; 2],
                vqs: vec![MmioQueue::default(); VIRTIO_QUEUE_MAX],
            }),
            owner: Mutex::new(None),
        })
    }

    /// Whether the transport uses the legacy register layout.
    pub fn is_legacy(&self) -> bool {
        self.legacy
    }

    /// The interrupt output. Connect it to the interrupt controller.
    pub fn irq(&self) -> &IrqPin {
        &self.irq
    }

    fn lock(&self) -> Option<StateGuard<'_>> {
        let me = thread::current().id();
        if *self.owner.lock().unwrap_or_else(PoisonError::into_inner) == Some(me) {
            return None;
        }
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *self.owner.lock().unwrap_or_else(PoisonError::into_inner) = Some(me);
        Some(StateGuard { state, owner: &self.owner })
    }

    /// Runs `f` on the plugged device. Returns `None` if there is no device, or if called from
    /// inside the device's own processing.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut VirtioBackend) -> R) -> Option<R> {
        let mut st = self.lock()?;
        st.backend.as_mut().map(f)
    }

    /// Runs `f` on the plugged device model as its concrete type `D`, together with the core
    /// state. Returns `None` if there is no device or it is not a `D`.
    ///
    /// This is how the host side talks to a device, for example to feed console input.
    pub fn with_device<D: VirtioDeviceClass, R>(
        &self,
        f: impl FnOnce(&mut VirtIODevice, &mut D) -> R,
    ) -> Option<R> {
        let mut st = self.lock()?;
        let (vdev, dev) = st.backend.as_mut()?.downcast_mut::<D>()?;
        Some(f(vdev, dev))
    }

    /// `virtio_mmio_reset()`: the device reset of the transport.
    pub fn reset(&self) {
        let Some(mut st) = self.lock() else {
            return;
        };
        self.soft_reset(&mut st);
        st.host_features_sel = 0;
        st.guest_features_sel = 0;
        st.guest_page_shift = 0;
        if !self.legacy {
            st.guest_features = [0; 2];
            for q in &mut st.vqs {
                *q = MmioQueue::default();
            }
        }
    }

    /// `virtio_mmio_soft_reset()`: what the guest can trigger by writing 0 to `STATUS`.
    fn soft_reset(&self, st: &mut MmioState) {
        if let Some(b) = st.backend.as_mut() {
            b.reset();
        }
        if !self.legacy {
            for q in &mut st.vqs {
                q.enabled = false;
            }
        }
    }

    fn read_reg(&self, st: &mut MmioState, offset: u64, size: AccessSize) -> u64 {
        let legacy = self.legacy;
        let Some(backend) = st.backend.as_mut() else {
            return u64::from(match offset {
                VIRTIO_MMIO_MAGIC_VALUE => VIRT_MAGIC,
                VIRTIO_MMIO_VERSION if legacy => VIRT_VERSION_LEGACY,
                VIRTIO_MMIO_VERSION => VIRT_VERSION,
                VIRTIO_MMIO_VENDOR_ID => VIRT_VENDOR,
                _ => 0,
            });
        };

        if offset >= VIRTIO_MMIO_CONFIG {
            return match size.bytes() {
                n @ (1 | 2 | 4) => {
                    u64::from(backend.config_read(offset - VIRTIO_MMIO_CONFIG, n)) & size.mask()
                }
                _ => 0,
            };
        }
        if size.bytes() != 4 {
            return 0;
        }
        let vdev = backend.vdev();
        let sel = vdev.queue_sel();
        let value = match offset {
            VIRTIO_MMIO_MAGIC_VALUE => VIRT_MAGIC,
            VIRTIO_MMIO_VERSION if legacy => VIRT_VERSION_LEGACY,
            VIRTIO_MMIO_VERSION => VIRT_VERSION,
            VIRTIO_MMIO_DEVICE_ID => u32::from(vdev.device_id()),
            VIRTIO_MMIO_VENDOR_ID => VIRT_VENDOR,
            VIRTIO_MMIO_DEVICE_FEATURES if legacy => {
                if st.host_features_sel != 0 {
                    0
                } else {
                    vdev.host_features() as u32
                }
            }
            VIRTIO_MMIO_DEVICE_FEATURES => {
                let features = vdev.host_features() & !backend.class().legacy_features();
                (features >> (32 * st.host_features_sel)) as u32
            }
            VIRTIO_MMIO_QUEUE_NUM_MAX => u32::from(vdev.queue_num_max(sel)),
            VIRTIO_MMIO_QUEUE_PFN if legacy => (vdev.queue_addr(sel) >> st.guest_page_shift) as u32,
            VIRTIO_MMIO_QUEUE_READY if !legacy => u32::from(st.vqs[usize::from(sel)].enabled),
            VIRTIO_MMIO_INTERRUPT_STATUS => u32::from(vdev.isr()),
            VIRTIO_MMIO_STATUS => u32::from(vdev.status()),
            VIRTIO_MMIO_CONFIG_GENERATION if !legacy => vdev.generation(),
            VIRTIO_MMIO_SHM_LEN_LOW | VIRTIO_MMIO_SHM_LEN_HIGH => u32::MAX,
            // Write-only registers, registers of the other layout and holes read as zero.
            _ => 0,
        };
        u64::from(value)
    }

    fn write_reg(&self, st: &mut MmioState, offset: u64, size: AccessSize, value: u64) {
        let legacy = self.legacy;
        let Some(backend) = st.backend.as_mut() else {
            return;
        };

        if offset >= VIRTIO_MMIO_CONFIG {
            if let n @ (1 | 2 | 4) = size.bytes() {
                backend.config_write(offset - VIRTIO_MMIO_CONFIG, n, value as u32);
            }
            return;
        }
        if size.bytes() != 4 {
            return;
        }
        let value = value as u32;
        let sel = backend.vdev().queue_sel();
        let idx = usize::from(sel);
        match offset {
            VIRTIO_MMIO_DEVICE_FEATURES_SEL => {
                st.host_features_sel = u32::from(value != 0);
            }
            VIRTIO_MMIO_DRIVER_FEATURES if legacy => {
                if st.guest_features_sel == 0 {
                    let _ = backend.set_features(u64::from(value));
                }
            }
            VIRTIO_MMIO_DRIVER_FEATURES => {
                st.guest_features[st.guest_features_sel as usize] = value;
            }
            VIRTIO_MMIO_DRIVER_FEATURES_SEL => {
                st.guest_features_sel = u32::from(value != 0);
            }
            VIRTIO_MMIO_GUEST_PAGE_SIZE if legacy => {
                let shift = value.trailing_zeros();
                st.guest_page_shift = if shift > 31 { 0 } else { shift };
            }
            VIRTIO_MMIO_QUEUE_SEL => {
                if (value as usize) < VIRTIO_QUEUE_MAX {
                    backend.vdev_mut().set_queue_sel(value as u16);
                }
            }
            VIRTIO_MMIO_QUEUE_NUM => {
                let vdev = backend.vdev_mut();
                vdev.set_queue_num(sel, value);
                if legacy {
                    vdev.update_queue_rings(sel);
                } else {
                    st.vqs[idx].num = value as u16;
                }
            }
            VIRTIO_MMIO_QUEUE_ALIGN if legacy => {
                backend.vdev_mut().set_queue_align(sel, value);
            }
            VIRTIO_MMIO_QUEUE_PFN if legacy => {
                if value == 0 {
                    self.soft_reset(st);
                } else {
                    let addr = u64::from(value) << st.guest_page_shift;
                    backend.vdev_mut().set_queue_addr(sel, addr);
                }
            }
            VIRTIO_MMIO_QUEUE_READY if !legacy => {
                let q = st.vqs[idx];
                if value != 0 {
                    let vdev = backend.vdev_mut();
                    vdev.set_queue_num(sel, u32::from(q.num));
                    vdev.set_queue_rings(sel, join(q.desc), join(q.avail), join(q.used));
                    st.vqs[idx].enabled = true;
                } else {
                    st.vqs[idx].enabled = false;
                }
            }
            VIRTIO_MMIO_QUEUE_NOTIFY => {
                let n = value as u16;
                if usize::from(n) < VIRTIO_QUEUE_MAX && backend.vdev().queue_num(n) != 0 {
                    backend.queue_notify(n);
                }
            }
            VIRTIO_MMIO_INTERRUPT_ACK => {
                let vdev = backend.vdev_mut();
                vdev.clear_isr(value as u8);
                vdev.update_irq();
            }
            VIRTIO_MMIO_STATUS => {
                if !legacy && value & u32::from(VIRTIO_CONFIG_S_FEATURES_OK) != 0 {
                    let _ = backend.set_features(join(st.guest_features));
                }
                let _ = backend.set_status(value as u8);
                if backend.vdev().status() == 0 {
                    self.soft_reset(st);
                }
            }
            VIRTIO_MMIO_QUEUE_DESC_LOW if !legacy => st.vqs[idx].desc[0] = value,
            VIRTIO_MMIO_QUEUE_DESC_HIGH if !legacy => st.vqs[idx].desc[1] = value,
            VIRTIO_MMIO_QUEUE_AVAIL_LOW if !legacy => st.vqs[idx].avail[0] = value,
            VIRTIO_MMIO_QUEUE_AVAIL_HIGH if !legacy => st.vqs[idx].avail[1] = value,
            VIRTIO_MMIO_QUEUE_USED_LOW if !legacy => st.vqs[idx].used[0] = value,
            VIRTIO_MMIO_QUEUE_USED_HIGH if !legacy => st.vqs[idx].used[1] = value,
            // Read-only registers, registers of the other layout and holes ignore writes.
            _ => {}
        }
    }
}

impl MmioOps for VirtioMmio {
    fn read(&self, _cx: &AccessCtx, offset: u64, size: AccessSize) -> MemResult<u64> {
        let Some(mut st) = self.lock() else {
            return Ok(0);
        };
        Ok(self.read_reg(&mut st, offset, size))
    }

    fn write(&self, _cx: &AccessCtx, offset: u64, size: AccessSize, value: u64) -> MemResult<()> {
        if let Some(mut st) = self.lock() {
            self.write_reg(&mut st, offset, size, value);
        }
        Ok(())
    }
}
