// SPDX-License-Identifier: GPL-2.0-or-later

//! Plays the PC speaker the way a guest beeps: channel 2 of the PIT counts in mode 3 and port
//! 0x61 sets the gate and the data bit. The square wave must come out of a wav backend at
//! 32 kHz, unsigned 8-bit mono, and stop when the data bit goes.

use std::sync::Arc;

use ruvm_audio::registry;
use ruvm_base::ClockType;
use ruvm_hw_audio::PcSpkAudio;
use ruvm_hw_core::timer::Clock;
use ruvm_hw_timer::i8254::{I8254, PIT_FREQ};
use ruvm_qapi::types::{
    AudioFormat, Audiodev, AudiodevPerDirectionOptions, AudiodevU, AudiodevWavOptions,
};

/// Close to 1 kHz.
const COUNT: u32 = 1193;

/// The loop `generate_samples()` makes for `COUNT`: whole periods in up to 1792 samples.
fn wave() -> Vec<u8> {
    let m = 32000 * COUNT;
    let n = ((u64::from(PIT_FREQ) << 32) / u64::from(m)) as u32;
    let aligned = 1792 * u64::from(PIT_FREQ) / u64::from(m) * u64::from(m);
    let samples = (aligned / u64::from(PIT_FREQ >> 1) + 1) >> 1;
    (0..samples as u32).map(|i| ((64 & (n.wrapping_mul(i) >> 25)) as u8).wrapping_sub(32)).collect()
}

fn run(clock: &Clock, from_ms: i64, to_ms: i64) {
    for t in from_ms + 1..=to_ms {
        clock.set_ns(t * 1_000_000);
        clock.run_timers();
    }
}

#[test]
fn channel_2_plays_a_square_wave_into_a_wav_file() {
    let path = std::env::temp_dir().join(format!("ruvm-pcspk-{}.wav", std::process::id()));
    let out_opts = AudiodevPerDirectionOptions {
        frequency: Some(32000),
        channels: Some(1),
        format: Some(AudioFormat::U8),
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

    let pit = I8254::new(&clock, 0x40);
    let spk = PcSpkAudio::new(Arc::clone(&pit), registry::be_by_name("snd0").unwrap());
    // Channel 2, low then high byte, mode 3.
    pit.ioport_write(3, 0xb6);
    pit.ioport_write(2, (COUNT & 0xff) as u8);
    pit.ioport_write(2, (COUNT >> 8) as u8);
    // Writing 3 to port 0x61.
    pit.set_gate(2, 1);
    spk.io_write(true, true);
    run(&clock, 0, 300);
    // The data bit goes, the gate stays.
    spk.io_write(true, false);
    run(&clock, 300, 400);

    // The card still holds the backend, so registry::cleanup() would leave it open.
    registry::be_by_name("snd0").unwrap().shutdown();
    let data = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(&data[22..24], &1u16.to_le_bytes());
    assert_eq!(&data[24..28], &32000u32.to_le_bytes());
    assert_eq!(&data[34..36], &8u16.to_le_bytes());
    let audio = &data[44..];
    let start = audio.iter().position(|&b| b != 0x80).unwrap();
    let end = audio.iter().rposition(|&b| b != 0x80).unwrap() + 1;
    let played = &audio[start..end];
    // The mixing engine mixes a mono voice back down as the sum of its two channels, like
    // QEMU's, so the wave's two levels come out at the ends of the range.
    let w: Vec<u8> = wave().iter().map(|&b| if b > 0x80 { 0xff } else { 0 }).collect();
    assert_eq!(w.len(), 1792);
    // Close to 300 ms of sound, the wave over and over from its start.
    assert!(played.len() > 8000, "{}", played.len());
    for (i, chunk) in played.chunks(w.len()).enumerate() {
        assert_eq!(chunk, &w[..chunk.len()], "loop {i}");
    }
    assert!(played.len() < 12000, "{}", played.len());
}
