// SPDX-License-Identifier: GPL-2.0-or-later

//! Drives `intel-hda` with an `hda-duplex` codec the way a guest driver does: the codec
//! wakes up in STATESTS, verbs go through the immediate command registers and the CORB and
//! RIRB rings, and output stream 4 plays a two entry buffer list while the virtual clock
//! runs. The PCM must come out of a wav backend unchanged.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_audio::registry;
use ruvm_base::ClockType;
use ruvm_hw_audio::{HdaCodecKind, IntelHda};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_pci::{PciBus, pci_swizzle_map_irq_fn};
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};
use ruvm_qapi::types::{Audiodev, AudiodevPerDirectionOptions, AudiodevU, AudiodevWavOptions};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
const MMIO: u64 = 0xf000_0000;
const CORB: u64 = 0x1000;
const RIRB: u64 = 0x2000;
const BDL: u64 = 0x3000;
const PCM: u64 = 0x1_0000;
/// Two buffers of 2048 stereo 16 bit frames.
const HALF: usize = 8192;
/// Output stream 4, the first output engine.
const SD4: u64 = 0x80 + 4 * 0x20;

/// Guest RAM for DMA.
struct Ram(Mutex<Vec<u8>>);

impl Ram {
    fn le32(&self, addr: u64) -> u32 {
        let ram = self.0.lock().unwrap();
        let a = addr as usize;
        u32::from_le_bytes(ram[a..a + 4].try_into().unwrap())
    }
}

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

fn wr(m: &AddressSpace, off: u64, size: u32, v: u64) {
    assert!(m.store(MMIO + off, size, v, Endian::Little, ATTRS).is_ok());
}

fn rd(m: &AddressSpace, off: u64, size: u32) -> u64 {
    m.load(MMIO + off, size, Endian::Little, ATTRS).0
}

/// A verb through the immediate command interface, which answers in IR.
fn immediate(m: &AddressSpace, verb: u32) -> u32 {
    wr(m, 0x60, 4, u64::from(verb));
    wr(m, 0x68, 2, 1);
    assert_eq!(rd(m, 0x68, 2), 2);
    rd(m, 0x64, 4) as u32
}

/// A verb for codec 0: a 12 bit verb with an 8 bit payload, or a 4 bit one with 16 bits.
fn verb(nid: u32, verb: u32, payload: u32) -> u32 {
    if verb > 0xf {
        (nid << 20) | (verb << 8) | payload
    } else {
        (nid << 20) | (verb << 16) | payload
    }
}

#[test]
fn stream_4_plays_the_buffers_into_a_wav_file() {
    let path = std::env::temp_dir().join(format!("ruvm-hda-{}.wav", std::process::id()));
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
    let mem_as = mem.address_space_init(sysmem, "memory").unwrap();
    let bus = PciBus::new_root("pci.0", Arc::clone(&mem), sysmem, io, 0);
    let level = Arc::new(AtomicI32::new(0));
    let l = Arc::clone(&level);
    bus.set_irqs(Arc::new(move |_, v| l.store(v, Ordering::SeqCst)), 4);
    bus.set_map_irq(Arc::new(pci_swizzle_map_irq_fn));

    // An LCG fills the PCM; the descriptors point at its two halves, the last with IOC.
    let mut ram = vec![0u8; 0x2_0000];
    let mut x: u16 = 0x1234;
    let pcm = &mut ram[PCM as usize..PCM as usize + 2 * HALF];
    for s in pcm.chunks_exact_mut(2) {
        x = x.wrapping_mul(25173).wrapping_add(13849);
        s.copy_from_slice(&x.to_le_bytes());
    }
    let pcm = pcm.to_vec();
    for k in 0..2u64 {
        let e = (BDL + 16 * k) as usize;
        ram[e..e + 8].copy_from_slice(&(PCM + k * HALF as u64).to_le_bytes());
        ram[e + 8..e + 12].copy_from_slice(&(HALF as u32).to_le_bytes());
        ram[e + 12..e + 16].copy_from_slice(&(k as u32).to_le_bytes());
    }
    let dma = Arc::new(Ram(Mutex::new(ram)));

    let hda = IntelHda::realize(
        &bus,
        Some(4 << 3),
        None,
        false,
        Some(false),
        false,
        Arc::clone(&clock),
        Arc::clone(&dma) as Arc<dyn DmaMemory>,
    )
    .unwrap();
    hda.add_codec(HdaCodecKind::Duplex, u32::MAX, true, || registry::be_by_name("snd0")).unwrap();
    // The codec bus is full at 15 codecs.
    let err = hda.add_codec(HdaCodecKind::Output, 15, true, || unreachable!()).unwrap_err();
    assert_eq!(err.to_string(), "HDA audio codec address is full");

    let dev = hda.pci_device();
    assert_eq!(dev.config_read(0, 4), 0x2668_8086);
    assert_eq!(dev.config_read(8, 4), 0x0403_0001);
    assert_eq!(dev.config_read(0x40, 1), 1);
    dev.config_write(0x10, MMIO as u32, 4);
    dev.config_write(0x04, 0x6, 2);

    let m = &mem_as;
    assert_eq!(rd(m, 0x00, 4), 0x4401);
    // VMIN and VMAJ are separate registers, and a read only ever sees the one it starts at.
    assert_eq!(rd(m, 0x02, 2), 0);
    assert_eq!(rd(m, 0x03, 1), 1);
    assert_eq!(rd(m, 0x4e, 1), 0x42);
    assert_eq!(rd(m, SD4 + 0x10, 2), 256);
    assert_eq!(rd(m, SD4, 4), 0x2000_0000);
    // The alias at 0x2000.
    assert_eq!(rd(m, 0x2000, 2), 0x4401);
    // The codec at address 0 announced itself.
    assert_eq!(rd(m, 0x0e, 2), 1);
    wr(m, 0x08, 4, 1);
    wr(m, 0x0e, 2, 1);
    assert_eq!(rd(m, 0x0e, 2), 0);

    assert_eq!(immediate(m, verb(0, 0xf00, 0x00)), 0x1af4_0022);
    assert_eq!(immediate(m, verb(1, 0xf00, 0x04)), 0x0002_0004);
    assert_eq!(immediate(m, verb(3, 0xf02, 0)), 2);
    assert_eq!(immediate(m, verb(5, 0xf1c, 0)), 0x0080_5020);
    assert_eq!(immediate(m, verb(2, 0xb, 0xa000)), 0x4a);

    // Two verbs through the rings, with the RIRB interrupt after both answers.
    {
        let mut ram = dma.0.lock().unwrap();
        let c = CORB as usize;
        ram[c + 4..c + 8].copy_from_slice(&verb(4, 0xf00, 0x09).to_le_bytes());
        ram[c + 8..c + 12].copy_from_slice(&verb(2, 0xa, 0).to_le_bytes());
    }
    wr(m, 0x20, 4, 0xc000_0000);
    wr(m, 0x50, 4, RIRB);
    wr(m, 0x5a, 2, 2);
    wr(m, 0x5c, 1, 3);
    wr(m, 0x40, 4, CORB);
    wr(m, 0x4c, 1, 2);
    wr(m, 0x48, 2, 2);
    assert_eq!(rd(m, 0x4a, 2), 2);
    assert_eq!(rd(m, 0x58, 2), 2);
    assert_eq!(dma.le32(RIRB + 8), 0x0010_011b);
    assert_eq!(dma.le32(RIRB + 16), 0x11);
    assert_eq!(dma.le32(RIRB + 20), 0);
    assert_eq!(rd(m, 0x5d, 1), 1);
    assert_eq!(rd(m, 0x24, 4), 0xc000_0000);
    assert_eq!(level.load(Ordering::SeqCst), 1);
    wr(m, 0x5d, 1, 1);
    assert_eq!(level.load(Ordering::SeqCst), 0);

    // The DAC on stream 1, 48 kHz 16 bit stereo.
    assert_eq!(immediate(m, verb(2, 0x706, 0x10)), 0);
    assert_eq!(immediate(m, verb(2, 0x2, 0x11)), 0);
    assert_eq!(immediate(m, verb(2, 0xf06, 0)), 0x10);
    wr(m, SD4 + 0x08, 4, 2 * HALF as u64);
    wr(m, SD4 + 0x0c, 2, 1);
    wr(m, SD4 + 0x12, 2, 0x11);
    wr(m, SD4 + 0x18, 4, BDL);
    wr(m, 0x20, 4, 0x8000_0010);
    // Stream number 1, IOCE and run.
    wr(m, SD4, 4, (1 << 20) | 0x6);

    for t in 1..=150 {
        clock.set_ns(t * 1_000_000);
        clock.run_timers();
    }
    // BCIS for the second buffer, and the stream interrupt.
    assert_eq!(rd(m, SD4 + 3, 1), 0x24);
    assert_eq!(rd(m, 0x24, 4), 0x8000_0010);
    assert_eq!(level.load(Ordering::SeqCst), 1);
    wr(m, SD4 + 3, 1, 0x04);
    assert_eq!(level.load(Ordering::SeqCst), 0);
    wr(m, SD4, 4, 1 << 20);

    // The card still holds the backend, so registry::cleanup() would leave it open.
    registry::be_by_name("snd0").unwrap().shutdown();
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&data[22..24], &2u16.to_le_bytes());
    assert_eq!(&data[24..28], &48000u32.to_le_bytes());
    let audio = &data[44..];
    let start = audio.chunks_exact(4).position(|f| f != [0; 4]).unwrap() * 4;
    assert_eq!(&audio[start..start + pcm.len()], &pcm[..]);
    // The list wraps around, so the buffers play again.
    let rest = &audio[start + pcm.len()..];
    assert!(rest.len() > 4096);
    assert_eq!(&rest[..4096], &pcm[..4096]);
}
