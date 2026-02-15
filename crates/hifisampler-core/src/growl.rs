//! Growl / screaming voice effect processor.
//!
//! Implements the HG flag effect using a square wave LFO
//! to modulate pitch, creating a growl/scream effect.

use std::f32::consts::PI;

/// Apply growl effect to audio.
///
/// strength: 0.0-1.0, amount of effect
/// sample_rate: audio sample rate
pub fn apply_growl(audio: &[f32], strength: f32, sample_rate: u32) -> Vec<f32> {
    if strength < 0.01 || audio.is_empty() {
        return audio.to_vec();
    }

    let sr = sample_rate as f32;
    let strength = strength.clamp(0.0, 1.0);

    // High-pass filter to separate high frequency content (>400Hz)
    let cutoff = 400.0;
    let rc = 1.0 / (2.0 * PI * cutoff);
    let dt = 1.0 / sr;
    let alpha = rc / (rc + dt);

    let mut high_pass = vec![0.0f32; audio.len()];
    let mut low_pass = vec![0.0f32; audio.len()];
    high_pass[0] = audio[0];
    low_pass[0] = audio[0];

    for i in 1..audio.len() {
        high_pass[i] = alpha * (high_pass[i - 1] + audio[i] - audio[i - 1]);
        low_pass[i] = audio[i] - high_pass[i];
    }

    // Square wave LFO at ~80Hz
    let lfo_freq = 80.0;
    let pitch_deviation = 100.0 * strength; // cents deviation
    let deviation_ratio = 2.0f32.powf(pitch_deviation / 1200.0) - 1.0;

    // Apply pitch modulation via variable delay
    let mut output = vec![0.0f32; audio.len()];
    let mut phase = 0.0f32;

    for i in 0..audio.len() {
        let t = i as f32 / sr;

        // Square wave LFO
        let lfo = if (t * lfo_freq * 2.0 * PI).sin() > 0.0 {
            1.0
        } else {
            -1.0
        };

        // Phase accumulation with pitch modulation
        let freq_mod = 1.0 + lfo * deviation_ratio;
        phase += freq_mod;

        let read_pos = phase;
        let idx = read_pos as usize;
        let frac = read_pos - idx as f32;

        if idx + 1 < high_pass.len() {
            let interpolated = high_pass[idx] * (1.0 - frac) + high_pass[idx + 1] * frac;
            output[i] = low_pass[i] + interpolated * strength + high_pass[i] * (1.0 - strength);
        } else {
            output[i] = audio[i];
        }
    }

    // RMS normalization to maintain volume
    let input_rms = rms(audio);
    let output_rms = rms(&output);

    if output_rms > 1e-8 {
        let gain = input_rms / output_rms;
        output.iter_mut().for_each(|x| *x *= gain);
    }

    output
}

fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|&x| x * x).sum();
    (sum_sq / signal.len() as f32).sqrt()
}
