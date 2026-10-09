// SPDX-License-Identifier: GPL-2.0-or-later

//! Drives the AC97 the way a guest driver does: BARs and bus mastering through config space,
//! a codec reset, two buffer descriptors on PCM out, then the virtual clock until the box
//! halts. The PCM must come out of a wav backend unchanged.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_audio::registry;
use ruvm_base::ClockType;
use ruvm_hw_audio::Ac97;
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_pci::{PciBus, pci_swizzle_map_irq_fn};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};
use ruvm_qapi::types::{Audiodev, AudiodevPerDirectionOptions, AudiodevU, AudiodevWavOptions};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const NAM: u64 = 0x1000;
const NABM: u64 = 0x1400;
const BDL: u64 = 0x1_0000;
const PCM: u64 = 0x2_0000;
/// Two buffers of 2048 stereo frames.
const FRAMES: usize = 4096;

/// Guest RAM for DMA.
struct Ram(Mutex<Vec<u8>>);

impl DmaMemory for Ram {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let ram = self.0.lock().unwrap();
        let a = addr as usize;
        match ram.get(a..a + buf.len()) {
            Some(s) => {
                buf.copy_from_slice(s);
                true
            }
            None => false,
        }
    }

    fn write(&self, addr: u64, buf: &[u8]) -> bool {
        let mut ram = self.0.lock().unwrap();
        let a = addr as usize;
        match ram.get_mut(a..a + buf.len()) {
            Some(s) => {
                s.copy_from_slice(buf);
                true
            }
            None => false,
        }
    }
}

fn out(io: &AddressSpace, port: u64, size: u32, v: u64) {
    assert!(io.store(port, size, v, Endian::Little, ATTRS).is_ok());
}

fn inp(io: &AddressSpace, port: u64, size: u32) -> u64 {
    io.load(port, size, Endian::Little, ATTRS).0
}

#[test]
fn pcm_out_plays_the_buffers_into_a_wav_file() {
    let path = std::env::temp_dir().join(format!("ruvm-ac97-{}.wav", std::process::id()));
    let out_opts = AudiodevPerDirectionOptions { frequency: Some(48000), ..Default::default() };
    registry::add_audiodev(Audiodev {
        id: "snd0".into(),
        timer_period: None,
        u: AudiodevU::Wav(AudiodevWavOptions {
            out: Some(out_opts),
            path: Some(path.to_str().unwrap().into()),
            ..Default::default()
        }),
    })
    .unwrap();
    registry::init_audiodevs().unwrap();
    let clock = Clock::manual(ClockType::Virtual);
    registry::attach_clock(&clock);
    registry::vm_state_change(true);

    let mem = Arc::new(MemorySystem::new());
    let sysmem = mem.new_container("system", 1 << 64).unwrap();
    let io = mem.new_container("io", 1 << 16).unwrap();
    let io_as = mem.address_space_init(io, "I/O").unwrap();
    let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
    let level = Arc::new(AtomicI32::new(0));
    let l = Arc::clone(&level);
    bus.set_irqs(Arc::new(move |_, v| l.store(v, Ordering::SeqCst)), 4);
    bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));

    // An LCG fills the PCM; the descriptors point at its two halves, the last with IOC.
    let mut ram = vec![0u8; 0x4_0000];
    let mut x: u16 = 0x1234;
    let pcm = &mut ram[PCM as usize..PCM as usize + FRAMES * 4];
    for s in pcm.chunks_exact_mut(2) {
        x = x.wrapping_mul(25173).wrapping_add(13849);
        s.copy_from_slice(&x.to_le_bytes());
    }
    let pcm = pcm.to_vec();
    for k in 0..2u32 {
        let e = BDL as usize + 8 * k as usize;
        let addr = PCM as u32 + k * (FRAMES as u32 * 2);
        ram[e..e + 4].copy_from_slice(&addr.to_le_bytes());
        ram[e + 4..e + 6].copy_from_slice(&(FRAMES as u16).to_le_bytes());
        let flags: u16 = if k == 1 { 0x8000 } else { 0 };
        ram[e + 6..e + 8].copy_from_slice(&flags.to_le_bytes());
    }
    let dma = Arc::new(Ram(Mutex::new(ram)));

    let be = registry::be_by_name("snd0").unwrap();
    let ac97 = Ac97::realize(&bus, Some(3 << 3), None, be, dma).unwrap();
    let dev = ac97.pci_device();
    assert_eq!(dev.config_read(0, 4), 0x2415_8086);
    dev.config_write(0x10, NAM as u32, 4);
    dev.config_write(0x14, NABM as u32, 4);
    dev.config_write(0x04, 0x5, 2);

    out(&io_as, NABM + 0x2c, 4, 2);
    out(&io_as, NAM, 2, 0);
    // The codec comes out of reset with master and PCM out muted.
    assert_eq!(inp(&io_as, NAM + 0x02, 2), 0x8000);
    assert_eq!(inp(&io_as, NAM + 0x18, 2), 0x8808);
    // get_volume() scales the attenuation linearly, so PCM out is at full volume at 0, not at
    // the 0 dB of 0x0808, and only then do the samples pass through unscaled.
    out(&io_as, NAM + 0x02, 2, 0);
    out(&io_as, NAM + 0x18, 2, 0);
    out(&io_as, NABM + 0x10, 4, BDL);
    out(&io_as, NABM + 0x15, 1, 1);
    out(&io_as, NABM + 0x1b, 1, 0x10 | 1);

    for t in 1..=30 {
        clock.set_ns(t * 10_000_000);
        clock.run_timers();
    }
    // DCH, CELV and LVBCI, then BCIS for the IOC on the last buffer.
    assert_eq!(inp(&io_as, NABM + 0x16, 2), 0x0f);
    assert_eq!(inp(&io_as, NABM + 0x14, 1), 1);
    assert_eq!(level.load(Ordering::SeqCst), 1);
    out(&io_as, NABM + 0x16, 2, 0x1c);
    assert_eq!(level.load(Ordering::SeqCst), 0);

    registry::cleanup();
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&data[22..24], &2u16.to_le_bytes());
    assert_eq!(&data[24..28], &48000u32.to_le_bytes());
    let audio = &data[44..];
    let start = audio.chunks_exact(4).position(|f| f != [0; 4]).unwrap() * 4;
    assert_eq!(&audio[start..start + pcm.len()], &pcm[..]);
    assert!(audio[start + pcm.len()..].iter().all(|&b| b == 0));
}
