// SPDX-License-Identifier: GPL-2.0-or-later

//! virtio-keyboard, virtio-mouse and virtio-tablet driven through a virtio-mmio transport, the
//! way the Linux virtio_input driver uses them: config selects, a full event queue, batches that
//! do not fit, and LED updates on the status queue.

use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use ruvm_hw_core::IrqLine;
use ruvm_hw_input::virtio_input::*;
use ruvm_hw_virtio::mmio::*;
use ruvm_hw_virtio::virtio::*;
use ruvm_hw_virtio::{AddressSpaceMemory, SharedGuestMemory, VirtioBackend, VirtioMmio};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};
use ruvm_qapi::types::{InputAxis, InputBtnEvent, InputButton, InputMoveEvent};
use ruvm_ui::console::DisplayState;
use ruvm_ui::input::{InputState, QEMU_CAPS_LOCK_LED, QEMU_NUM_LOCK_LED, QemuInputEvent};

const RAM_SIZE: u64 = 1 << 20;
const MMIO_BASE: u64 = 0x1000_0000;
const VRING_DESC_F_WRITE: u16 = 2;
const QSIZE: u16 = 8;

/// RAM at 0 and the device behind one modern virtio-mmio window, with the kick
/// `connect_input_mmio()` in ruvm-system sets.
struct Guest {
    space: Arc<AddressSpace>,
    level: Arc<AtomicI32>,
    input: Arc<InputState>,
    next_alloc: u64,
}

/// The driver side of a split ring, with one descriptor per chain.
struct Ring {
    index: u16,
    desc: u64,
    avail: u64,
    used: u64,
    next: u16,
    last_used: u16,
}

impl Guest {
    fn new(kind: VirtioInputKind, conf: VirtioInputConf) -> Guest {
        let sys = MemorySystem::new();
        let root = sys.new_container("system", u128::from(u64::MAX)).unwrap();
        let ram = sys.new_ram("ram", RAM_SIZE).unwrap();
        sys.add_subregion(root, 0, ram).unwrap();
        let space = sys.address_space_init(root, "memory").unwrap();
        let mem: SharedGuestMemory = Arc::new(AddressSpaceMemory::new(Arc::clone(&space)));
        let input = InputState::new();
        let dev = VirtioInput::new(kind, conf, Arc::clone(&input), DisplayState::new());
        let backend = VirtioBackend::new(Box::new(dev), mem).unwrap();
        let mmio = Arc::new(VirtioMmio::new(Some(backend), false).unwrap());
        let level = Arc::new(AtomicI32::new(0));
        let l = Arc::clone(&level);
        mmio.irq().connect(IrqLine::from_fn(move |v| l.store(v, Ordering::SeqCst)));
        let region =
            sys.new_io("virtio-mmio", u128::from(VIRTIO_MMIO_REGION_SIZE), mmio.clone()).unwrap();
        sys.add_subregion(root, MMIO_BASE, region).unwrap();
        let weak = Arc::downgrade(&mmio);
        mmio.with_device::<VirtioInput, _>(|_, d| {
            d.set_kick(Some(Box::new(move || {
                if let Some(t) = weak.upgrade() {
                    t.with_device::<VirtioInput, _>(|vdev, d| d.flush(vdev));
                }
            })));
        });
        Guest { space, level, input, next_alloc: 0x1000 }
    }

    fn readl(&self, off: u64) -> u32 {
        let (v, r) = self.space.load(MMIO_BASE + off, 4, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok());
        v as u32
    }

    fn store(&self, off: u64, size: u32, value: u64) {
        let r =
            self.space.store(MMIO_BASE + off, size, value, Endian::Little, MemTxAttrs::UNSPECIFIED);
        assert!(r.is_ok());
    }

    fn writel(&self, off: u64, value: u32) {
        self.store(off, 4, u64::from(value));
    }

    fn config_readb(&self, off: u64) -> u8 {
        let (v, r) = self.space.load(
            MMIO_BASE + VIRTIO_MMIO_CONFIG + off,
            1,
            Endian::Little,
            MemTxAttrs::UNSPECIFIED,
        );
        assert!(r.is_ok());
        v as u8
    }

    /// What `virtinput_cfg_select()` does: writes select and subsel, then reads the size.
    fn select(&self, select: u8, subsel: u8) -> Vec<u8> {
        self.store(VIRTIO_MMIO_CONFIG, 1, u64::from(select));
        self.store(VIRTIO_MMIO_CONFIG + 1, 1, u64::from(subsel));
        let size = self.config_readb(2);
        (0..u64::from(size)).map(|i| self.config_readb(8 + i)).collect()
    }

    fn status(&self) -> u8 {
        self.readl(VIRTIO_MMIO_STATUS) as u8
    }

    fn set_status(&self, s: u8) {
        self.writel(VIRTIO_MMIO_STATUS, u32::from(s));
    }

    /// Reset through FEATURES_OK with VERSION_1, then both queues.
    fn probe(&mut self) -> (Ring, Ring) {
        self.set_status(0);
        self.set_status(VIRTIO_CONFIG_S_ACKNOWLEDGE | VIRTIO_CONFIG_S_DRIVER);
        self.writel(VIRTIO_MMIO_DRIVER_FEATURES_SEL, 1);
        self.writel(VIRTIO_MMIO_DRIVER_FEATURES, 1);
        let s = self.status() | VIRTIO_CONFIG_S_FEATURES_OK;
        self.set_status(s);
        assert_ne!(self.status() & VIRTIO_CONFIG_S_FEATURES_OK, 0);
        (self.setup_queue(0), self.setup_queue(1))
    }

    fn driver_ok(&self) {
        let s = self.status() | VIRTIO_CONFIG_S_DRIVER_OK;
        self.set_status(s);
    }

    fn alloc(&mut self, size: u64) -> u64 {
        let addr = self.next_alloc.div_ceil(16) * 16;
        self.next_alloc = addr + size;
        addr
    }

    fn write_mem(&self, addr: u64, data: &[u8]) {
        assert!(self.space.write(addr, MemTxAttrs::UNSPECIFIED, data).is_ok());
    }

    fn read_mem(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0; len];
        assert!(self.space.read(addr, MemTxAttrs::UNSPECIFIED, &mut v).is_ok());
        v
    }

    fn read_u16(&self, addr: u64) -> u16 {
        u16::from_le_bytes(self.read_mem(addr, 2).try_into().unwrap())
    }

    fn read_u32(&self, addr: u64) -> u32 {
        u32::from_le_bytes(self.read_mem(addr, 4).try_into().unwrap())
    }

    fn setup_queue(&mut self, index: u16) -> Ring {
        self.writel(VIRTIO_MMIO_QUEUE_SEL, u32::from(index));
        assert_eq!(self.readl(VIRTIO_MMIO_QUEUE_NUM_MAX), u32::from(VIRTIO_INPUT_QUEUE_SIZE));
        let n = u64::from(QSIZE);
        let desc = self.alloc(16 * n);
        let avail = self.alloc(6 + 2 * n);
        let used = self.alloc(6 + 8 * n);
        self.writel(VIRTIO_MMIO_QUEUE_NUM, u32::from(QSIZE));
        self.writel(VIRTIO_MMIO_QUEUE_DESC_LOW, desc as u32);
        self.writel(VIRTIO_MMIO_QUEUE_AVAIL_LOW, avail as u32);
        self.writel(VIRTIO_MMIO_QUEUE_USED_LOW, used as u32);
        self.writel(VIRTIO_MMIO_QUEUE_READY, 1);
        Ring { index, desc, avail, used, next: 0, last_used: 0 }
    }

    /// Makes one buffer of 8 bytes available and gives its address.
    fn submit(&mut self, ring: &mut Ring, write: bool, data: &[u8; 8]) -> u64 {
        let addr = self.alloc(8);
        self.write_mem(addr, data);
        let slot = ring.next % QSIZE;
        let mut d = Vec::new();
        d.extend_from_slice(&addr.to_le_bytes());
        d.extend_from_slice(&8u32.to_le_bytes());
        d.extend_from_slice(&(if write { VRING_DESC_F_WRITE } else { 0 }).to_le_bytes());
        d.extend_from_slice(&0u16.to_le_bytes());
        self.write_mem(ring.desc + 16 * u64::from(slot), &d);
        self.write_mem(ring.avail + 4 + 2 * u64::from(slot), &slot.to_le_bytes());
        ring.next = ring.next.wrapping_add(1);
        self.write_mem(ring.avail + 2, &ring.next.to_le_bytes());
        addr
    }

    fn kick(&self, ring: &Ring) {
        self.writel(VIRTIO_MMIO_QUEUE_NOTIFY, u32::from(ring.index));
    }

    /// The used elements since the last call: id and length.
    fn used(&self, ring: &mut Ring) -> Vec<(u32, u32)> {
        let mut v = Vec::new();
        while self.read_u16(ring.used + 2) != ring.last_used {
            let e = ring.used + 4 + 8 * u64::from(ring.last_used % QSIZE);
            v.push((self.read_u32(e), self.read_u32(e + 4)));
            ring.last_used = ring.last_used.wrapping_add(1);
        }
        v
    }

    fn irq(&self) -> bool {
        self.level.load(Ordering::SeqCst) != 0
    }

    fn ack(&self) {
        let isr = self.readl(VIRTIO_MMIO_INTERRUPT_STATUS);
        self.writel(VIRTIO_MMIO_INTERRUPT_ACK, isr);
    }

    fn key(&self, key: u32, down: bool) {
        self.input.event_send(None, &QemuInputEvent::Key { key, down });
        self.input.event_sync();
    }
}

fn ev(type_: u16, code: u16, value: u32) -> [u8; 8] {
    let mut e = [0; 8];
    e[0..2].copy_from_slice(&type_.to_le_bytes());
    e[2..4].copy_from_slice(&code.to_le_bytes());
    e[4..8].copy_from_slice(&value.to_le_bytes());
    e
}

#[test]
fn keyboard_config() {
    let g = Guest::new(
        VirtioInputKind::Keyboard,
        VirtioInputConf { serial: Some("kbd0".into()), ..Default::default() },
    );
    assert_eq!(g.readl(VIRTIO_MMIO_DEVICE_ID), u32::from(VIRTIO_ID_INPUT));
    assert_eq!(g.select(VIRTIO_INPUT_CFG_ID_NAME, 0), b"QEMU Virtio Keyboard\0");
    assert_eq!(g.select(VIRTIO_INPUT_CFG_ID_SERIAL, 0), b"kbd0");
    assert_eq!(g.select(VIRTIO_INPUT_CFG_ID_DEVIDS, 0), [6, 0, 0x27, 6, 1, 0, 1, 0]);
    assert_eq!(g.select(VIRTIO_INPUT_CFG_PROP_BITS, 0), b"");
    assert_eq!(g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_LED as u8), [7]);
    assert_eq!(g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_REP as u8), [0]);
    let keys = g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_KEY as u8);
    assert_eq!(keys.len(), 29);
    assert_eq!((keys[0], keys[1], keys[28]), (0xfe, 0xff, 0xff));
    assert_eq!(g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_REL as u8), b"");
    // The config space is the largest entry plus the header, and reads past the selected entry
    // are zero.
    g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_LED as u8);
    assert_eq!(g.config_readb(9), 0);
}

#[test]
fn tablet_config() {
    let g = Guest::new(VirtioInputKind::Tablet, VirtioInputConf::default());
    assert_eq!(g.select(VIRTIO_INPUT_CFG_ID_NAME, 0), b"QEMU Virtio Tablet\0");
    assert_eq!(g.select(VIRTIO_INPUT_CFG_ID_SERIAL, 0), b"");
    assert_eq!(g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_ABS as u8), [3]);
    assert_eq!(g.select(VIRTIO_INPUT_CFG_EV_BITS, EV_REL as u8), [0, 1]);
    let abs = g.select(VIRTIO_INPUT_CFG_ABS_INFO, ABS_Y as u8);
    assert_eq!(abs.len(), 20);
    assert_eq!(&abs[0..8], &[0, 0, 0, 0, 0xff, 0x7f, 0, 0]);
}

#[test]
fn events_go_in_batches() {
    let mut g = Guest::new(VirtioInputKind::Keyboard, VirtioInputConf::default());
    let (mut evt, _sts) = g.probe();
    let bufs: Vec<u64> = (0..3).map(|_| g.submit(&mut evt, true, &[0xaa; 8])).collect();
    g.kick(&evt);

    // Before DRIVER_OK the device is not active and drops what it gets.
    g.key(30, true);
    assert!(g.used(&mut evt).is_empty());

    g.driver_ok();
    g.key(30, true);
    assert_eq!(g.used(&mut evt), vec![(0, 8), (1, 8)]);
    assert!(g.irq());
    g.ack();
    assert_eq!(g.read_mem(bufs[0], 8), ev(EV_KEY, 30, 1));
    assert_eq!(g.read_mem(bufs[1], 8), ev(EV_SYN, SYN_REPORT, 0));

    // Two events and one buffer: the batch is dropped and the buffer stays with the device.
    g.key(30, false);
    assert!(g.used(&mut evt).is_empty());
    assert!(!g.irq());
    assert_eq!(g.read_mem(bufs[2], 8), [0xaa; 8]);

    // With a second buffer the next batch starts in the one that was put back.
    let last = g.submit(&mut evt, true, &[0xaa; 8]);
    g.kick(&evt);
    g.key(48, true);
    assert_eq!(g.used(&mut evt), vec![(2, 8), (3, 8)]);
    assert_eq!(g.read_mem(bufs[2], 8), ev(EV_KEY, 48, 1));
    assert_eq!(g.read_mem(last, 8), ev(EV_SYN, SYN_REPORT, 0));

    // A reset deactivates it again.
    g.set_status(0);
    g.key(30, true);
    let (mut evt, _sts) = g.probe();
    g.submit(&mut evt, true, &[0; 8]);
    g.submit(&mut evt, true, &[0; 8]);
    g.kick(&evt);
    assert!(g.used(&mut evt).is_empty());
}

#[test]
fn mouse_wheel_and_motion() {
    let mut g = Guest::new(VirtioInputKind::Mouse, VirtioInputConf::default());
    let (mut evt, _sts) = g.probe();
    g.driver_ok();
    let bufs: Vec<u64> = (0..5).map(|_| g.submit(&mut evt, true, &[0; 8])).collect();
    g.kick(&evt);
    let btn = |button, down| QemuInputEvent::Btn(InputBtnEvent { button, down });
    g.input.event_send(None, &btn(InputButton::WheelDown, true));
    g.input.event_send(None, &btn(InputButton::WheelDown, false));
    g.input.event_send(None, &btn(InputButton::Side, true));
    g.input
        .event_send(None, &QemuInputEvent::Rel(InputMoveEvent { axis: InputAxis::X, value: -3 }));
    g.input.event_sync();
    let got: Vec<Vec<u8>> = bufs.iter().map(|&b| g.read_mem(b, 8)).collect();
    assert_eq!(
        got,
        vec![
            ev(EV_REL, REL_WHEEL, u32::MAX),
            ev(EV_KEY, BTN_GEAR_DOWN, 0),
            ev(EV_KEY, BTN_SIDE, 1),
            ev(EV_REL, REL_X, -3i32 as u32),
            ev(EV_SYN, SYN_REPORT, 0),
        ]
    );
    let names: Vec<String> = g.input.query_mice().into_iter().map(|m| m.name).collect();
    assert_eq!(names, vec!["QEMU Virtio Mouse".to_string()]);
}

#[test]
fn leds_from_the_status_queue() {
    let mut g = Guest::new(VirtioInputKind::Keyboard, VirtioInputConf::default());
    let (_evt, mut sts) = g.probe();
    g.driver_ok();
    g.submit(&mut sts, false, &ev(EV_LED, LED_CAPSL, 1));
    g.submit(&mut sts, false, &ev(EV_LED, LED_NUML, 1));
    g.kick(&sts);
    assert_eq!(g.used(&mut sts), vec![(0, 8), (1, 8)]);
    assert_eq!(g.input.get_leds_mask(None), QEMU_CAPS_LOCK_LED | QEMU_NUM_LOCK_LED);
    g.submit(&mut sts, false, &ev(EV_LED, LED_CAPSL, 0));
    g.kick(&sts);
    assert_eq!(g.input.get_leds_mask(None), QEMU_NUM_LOCK_LED);

    // A mouse ignores LEDs where QEMU would stop on an assertion.
    let mut g = Guest::new(VirtioInputKind::Mouse, VirtioInputConf::default());
    let (_evt, mut sts) = g.probe();
    g.driver_ok();
    g.submit(&mut sts, false, &ev(EV_LED, LED_CAPSL, 1));
    g.kick(&sts);
    assert_eq!(g.used(&mut sts), vec![(0, 8)]);
}

#[test]
fn serial_too_long() {
    let sys = MemorySystem::new();
    let root = sys.new_container("system", u128::from(u64::MAX)).unwrap();
    let space = sys.address_space_init(root, "memory").unwrap();
    let mem: SharedGuestMemory = Arc::new(AddressSpaceMemory::new(space));
    let conf = VirtioInputConf { serial: Some("x".repeat(129)), ..Default::default() };
    let dev =
        VirtioInput::new(VirtioInputKind::Mouse, conf, InputState::new(), DisplayState::new());
    assert!(VirtioBackend::new(Box::new(dev), Arc::clone(&mem)).is_err());
    let conf = VirtioInputConf { serial: Some("x".repeat(128)), ..Default::default() };
    let dev =
        VirtioInput::new(VirtioInputKind::Mouse, conf, InputState::new(), DisplayState::new());
    assert!(VirtioBackend::new(Box::new(dev), mem).is_ok());
}
