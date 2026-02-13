//! Generate a test WAV file for inference testing.

use hound::{SampleFormat, WavSpec, WavWriter};
use std::f32::consts::PI;

fn main() {
    let sr = 44100u32;
    let duration = 1.0f32; // 1 second
    let freq = 440.0f32; // A4
    let n_samples = (sr as f32 * duration) as usize;

    let spec = WavSpec {
        channels: 1,
        sample_rate: sr,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let out_path = "test_input.wav";
    let mut writer = WavWriter::create(out_path, spec).unwrap();

    for i in 0..n_samples {
        let t = i as f32 / sr as f32;
        let sample = (2.0 * PI * freq * t).sin() * 0.5;
        writer.write_sample((sample * 32767.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
    println!(
        "Generated {} ({} samples at {} Hz)",
        out_path, n_samples, sr
    );
}
