// SPDX-License-Identifier: GPL-2.0-or-later

//! Plays a block on the Sound Blaster 16 the way a DOS driver does: channel 5 of the i8257
//! pair set up for a single transfer, a DSP reset, the rate with command 0x41 and a 16-bit
//! signed stereo transfer with command 0xb0. The samples must come out of a wav backend
//! unchanged, and the card must raise its interrupt at the end of the block and drop it on
//! the ack.

use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::{Arc, Mutex};

use ruvm_audio::registry;
use ruvm_base::ClockType;
use ruvm_hw_audio::{IsaDma, Sb16, Sb16Config};
use ruvm_hw_core::fw_cfg::DmaMemory;
use ruvm_hw_core::irq::IrqLine;
use ruvm_hw_core::timer::Clock;
use ruvm_mem::{AddressSpace, Endian, MemTxAttrs, MemorySystem};
use ruvm_qapi::types::{
    AudioFormat, Audiodev, AudiodevPerDirectionOptions, AudiodevU, AudiodevWavOptions,
};

const ATTRS: MemTxAttrs = MemTxAttrs::UNSPECIFIED;
/// Page 2 of the 16-bit channels.
const PCM: usize = 0x2_0000;
const FRAMES: usize = 2048;

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

fn out(io: &AddressSpace, port: u64, v: u64) {
    assert!(io.store(port, 1, v, Endian::Little, ATTRS).is_ok());
}

fn inp(io: &AddressSpace, port: u64) -> u64 {
    io.load(port, 1, Endian::Little, ATTRS).0
}

#[test]
fn a_16_bit_block_plays_into_a_wav_file() {
    let path = std::env::temp_dir().join(format!("ruvm-sb16-{}.wav", std::process::id()));
    let out_opts = AudiodevPerDirectionOptions {
        frequency: Some(11025),
        channels: Some(2),
        format: Some(AudioFormat::S16),
        ..Default::default()
    };
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
    let io = mem.new_container("io", 1 << 16).unwrap();
    let io_as = mem.address_space_init(io, "I/O").unwrap();

    let mut ram = vec![0u8; 0x3_0000];
    let mut x: u16 = 0x1234;
    for s in ram[PCM..PCM + FRAMES * 4].chunks_exact_mut(2) {
        x = x.wrapping_mul(25173).wrapping_add(13849);
        s.copy_from_slice(&x.to_le_bytes());
    }
    let pcm = ram[PCM..PCM + FRAMES * 4].to_vec();
    let dma = IsaDma::new(Arc::new(Ram(Mutex::new(ram))), &clock);
    for (name, port, size, ops) in dma.io_regions() {
        let r = mem.new_io(name, u128::from(size), ops).unwrap();
        mem.add_subregion(io, u64::from(port), r).unwrap();
    }
    let level = Arc::new(AtomicI32::new(0));
    let irqs: Vec<IrqLine> = (0..16)
        .map(|i| {
            let l = Arc::clone(&level);
            IrqLine::from_fn(move |v| {
                if i == 5 {
                    l.store(v, Ordering::SeqCst);
                }
            })
        })
        .collect();
    let be = registry::be_by_name("snd0").unwrap();
    let sb16 = Sb16::realize(be, Sb16Config::default(), Some(&dma), &irqs, &clock).unwrap();
    for (port, size, ops) in sb16.io_regions() {
        let r = mem.new_io("sb16", u128::from(size), ops).unwrap();
        mem.add_subregion(io, u64::from(port), r).unwrap();
    }

    program_channel_5(&io_as);

    // DSP reset.
    out(&io_as, 0x226, 1);
    out(&io_as, 0x226, 0);
    assert_eq!(inp(&io_as, 0x22e), 0x80);
    assert_eq!(inp(&io_as, 0x22a), 0xaa);
    assert_eq!(inp(&io_as, 0x22c), 0);
    // The DSP version.
    out(&io_as, 0x22c, 0xe1);
    assert_eq!(inp(&io_as, 0x22a), 0x04);
    assert_eq!(inp(&io_as, 0x22a), 0x05);

    // 11025 Hz, high byte first, then 16-bit signed stereo for the whole buffer.
    for b in [0xd1, 0x41, 0x2b, 0x11] {
        out(&io_as, 0x22c, b);
    }
    play_block(&io_as);
    run(&clock, 1..=40);
    // The 16-bit interrupt is pending in mixer register 0x82.
    assert_eq!(level.load(Ordering::SeqCst), 1);
    out(&io_as, 0x224, 0x82);
    assert_eq!(inp(&io_as, 0x225), 0x02);
    assert_eq!(inp(&io_as, 0x22f), 0xff);
    assert_eq!(level.load(Ordering::SeqCst), 0);
    assert_eq!(inp(&io_as, 0x225), 0x00);
    // SB_read_DMA() hands back the position modulo the buffer length, so the controller never
    // sees the count run out and no terminal count shows in the status, as on QEMU.
    assert_eq!(inp(&io_as, 0xd0) & 0x02, 0);

    // The interrupt comes once the card has the last byte, and the end of the block then sits
    // in the backend until the voice starts again, as on QEMU. A second block brings it out.
    run(&clock, 41..=80);
    program_channel_5(&io_as);
    play_block(&io_as);
    run(&clock, 81..=120);
    assert_eq!(level.load(Ordering::SeqCst), 1);

    registry::cleanup();
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&data[22..24], &2u16.to_le_bytes());
    assert_eq!(&data[24..28], &11025u32.to_le_bytes());
    let audio = &data[44..];
    let start = audio.chunks_exact(4).position(|f| f != [0; 4]).unwrap() * 4;
    let (first, second) = audio[start..].split_at(pcm.len());
    assert_eq!(first, &pcm[..]);
    assert!(second.len() >= pcm.len() / 2);
    assert_eq!(second, &pcm[..second.len()]);
}

/// Channel 5: masked, single transfer from memory, word address 0 on page 2 for the whole
/// buffer, then unmasked.
fn program_channel_5(io_as: &AddressSpace) {
    out(io_as, 0xd4, 0x05);
    out(io_as, 0xd8, 0);
    out(io_as, 0xd6, 0x49);
    out(io_as, 0xc4, 0);
    out(io_as, 0xc4, 0);
    let words = (FRAMES * 2 - 1) as u64;
    out(io_as, 0xc6, words & 0xff);
    out(io_as, 0xc6, words >> 8);
    out(io_as, 0x8b, 2);
    out(io_as, 0xd4, 0x01);
}

/// A single 16-bit signed stereo block over the whole buffer.
fn play_block(io_as: &AddressSpace) {
    let samples = (FRAMES - 1) as u64;
    for b in [0xb0, 0x30, samples & 0xff, samples >> 8] {
        out(io_as, 0x22c, b);
    }
}

fn run(clock: &Clock, ticks: std::ops::RangeInclusive<i64>) {
    for t in ticks {
        clock.set_ns(t * 10_000_000);
        clock.run_timers();
    }
}
