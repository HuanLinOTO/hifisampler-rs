//! Audio I/O and processing utilities.
//!
//! Handles WAV reading/writing, resampling, dynamic range compression,
//! loudness normalization, and tension filtering.

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use num_complex::Complex;
use rubato::{SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction, Resampler};
use rustfft::FftPlanner;
use std::path::Path;

/// Read a WAV file, convert to mono float32, and resample to target sample rate.
pub fn read_wav(path: impl AsRef<Path>, target_sr: u32) -> Result<Vec<f32>> {
    let path = path.as_ref();
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let source_sr = spec.sample_rate;

    // Read samples as f32
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap() as f32 / max_val)
                .collect()
        }
        SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap())
            .collect(),
    };

    // Convert to mono by averaging channels
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample if needed
    if source_sr != target_sr {
        resample_audio(&mono, source_sr, target_sr)
    } else {
        Ok(mono)
    }
}

/// Write audio samples to a WAV file as 16-bit PCM.
pub fn save_wav(path: impl AsRef<Path>, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path.as_ref(), spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample = (clamped * 32767.0) as i16;
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Resample audio from one sample rate to another using sinc interpolation.
fn resample_audio(input: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(input.to_vec());
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_sr as f64 / from_sr as f64;
    let mut resampler = SincFixedIn::<f32>::new(
        ratio,
        2.0,
        params,
        input.len(),
        1, // mono
    )?;

    let input_vec = vec![input.to_vec()];
    let output = resampler.process(&input_vec, None)?;

    Ok(output.into_iter().next().unwrap())
}

/// Dynamic range compression: log1p(x * C) / log1p(C)
/// Matches Python's dynamic_range_compression.
pub fn dynamic_range_compression(x: &[f32], c: f32) -> Vec<f32> {
    let log_c = (1.0 + c).ln();
    x.iter()
        .map(|&v| (1.0 + v * c).ln() / log_c)
        .collect()
}

/// Dynamic range decompression (inverse of compression).
pub fn dynamic_range_decompression(x: &[f32], c: f32) -> Vec<f32> {
    let log_c = (1.0 + c).ln();
    x.iter()
        .map(|&v| ((v * log_c).exp() - 1.0) / c)
        .collect()
}

/// Pre-emphasis base tension filter.
/// Applies a frequency-dependent gain tilt in the STFT domain.
///
/// tension > 0: boost high frequencies (tighter sound)
/// tension < 0: boost low frequencies (looser sound)
pub fn pre_emphasis_tension(audio: &[f32], tension: f32, _sr: u32, n_fft: usize, hop_size: usize) -> Vec<f32> {
    if tension.abs() < 0.01 {
        return audio.to_vec();
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    // Create Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n_fft as f32).cos()))
        .collect();

    // Pad audio
    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad, 0.0);

    let output_len = audio.len();
    let mut output = vec![0.0f32; padded.len()];
    let mut norm = vec![0.0f32; padded.len()];

    // Frequency-dependent gain filter
    let freq_bins = n_fft / 2 + 1;
    let x0 = freq_bins as f32;
    let gains: Vec<f32> = (0..freq_bins)
        .map(|i| {
            let gain_db = (-tension / 50.0 / x0) * i as f32 + tension / 50.0;
            let gain_db = gain_db.clamp(-2.0, 2.0);
            10.0f32.powf(gain_db / 20.0) // dB to linear
        })
        .collect();

    // STFT → apply gain → ISTFT (overlap-add)
    let mut pos = 0;
    while pos + n_fft <= padded.len() {
        // Window the frame
        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| Complex::new(padded[pos + i] * window[i], 0.0))
            .collect();

        // FFT
        fft.process(&mut frame);

        // Apply gain
        for i in 0..freq_bins {
            frame[i] *= gains[i];
            if i > 0 && i < n_fft - i {
                frame[n_fft - i] *= gains[i];
            }
        }

        // IFFT
        ifft.process(&mut frame);

        // Overlap-add with normalization
        let inv_n = 1.0 / n_fft as f32;
        for i in 0..n_fft {
            output[pos + i] += frame[i].re * inv_n * window[i];
            norm[pos + i] += window[i] * window[i];
        }

        pos += hop_size;
    }

    // Normalize
    for i in 0..output.len() {
        if norm[i] > 1e-8 {
            output[i] /= norm[i];
        }
    }

    // Extract original length
    output[pad..pad + output_len].to_vec()
}

/// Compute RMS of a signal.
pub fn rms(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = signal.iter().map(|&x| x * x).sum();
    (sum_sq / signal.len() as f32).sqrt()
}

/// Peak value of a signal.
pub fn peak(signal: &[f32]) -> f32 {
    signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

/// Simple loudness normalization to target LUFS (simplified ITU-R BS.1770).
/// For a proper implementation, would need pyloudnorm equivalent in Rust.
/// This is a simplified version that normalizes to a target RMS level.
pub fn loudness_normalize(audio: &[f32], target_lufs: f32, strength: f32) -> Vec<f32> {
    if audio.is_empty() || strength < 0.01 {
        return audio.to_vec();
    }

    let current_rms = rms(audio);
    if current_rms < 1e-8 {
        return audio.to_vec();
    }

    // Approximate LUFS from RMS (simplified)
    let current_lufs = 20.0 * current_rms.log10();
    let gain_db = (target_lufs - current_lufs) * (strength / 100.0);
    let gain = 10.0f32.powf(gain_db / 20.0);

    audio.iter().map(|&x| x * gain).collect()
}

/// STFT computation using rustfft.
pub fn stft(
    audio: &[f32],
    n_fft: usize,
    hop_size: usize,
    window: &[f32],
) -> Vec<Vec<Complex<f32>>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let pad = n_fft / 2;
    let mut padded = vec![0.0f32; pad];
    padded.extend_from_slice(audio);
    padded.resize(padded.len() + pad, 0.0);

    let mut frames = Vec::new();
    let mut pos = 0;

    while pos + n_fft <= padded.len() {
        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| Complex::new(padded[pos + i] * window[i], 0.0))
            .collect();

        fft.process(&mut frame);

        // Keep only positive frequencies
        let freq_bins = n_fft / 2 + 1;
        frames.push(frame[..freq_bins].to_vec());

        pos += hop_size;
    }

    frames
}

/// ISTFT computation (overlap-add).
pub fn istft(
    frames: &[Vec<Complex<f32>>],
    n_fft: usize,
    hop_size: usize,
    window: &[f32],
    output_len: usize,
) -> Vec<f32> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_fft);

    let total_len = (frames.len() - 1) * hop_size + n_fft;
    let mut output = vec![0.0f32; total_len];
    let mut norm = vec![0.0f32; total_len];

    let inv_n = 1.0 / n_fft as f32;

    for (idx, frame) in frames.iter().enumerate() {
        let pos = idx * hop_size;

        // Reconstruct full spectrum (mirror conjugate)
        let mut full: Vec<Complex<f32>> = frame.clone();
        for i in 1..n_fft / 2 {
            full.push(frame[n_fft / 2 - i].conj());
        }
        full.resize(n_fft, Complex::new(0.0, 0.0));

        ifft.process(&mut full);

        for i in 0..n_fft {
            if pos + i < output.len() {
                output[pos + i] += full[i].re * inv_n * window[i];
                norm[pos + i] += window[i] * window[i];
            }
        }
    }

    // Normalize
    for i in 0..output.len() {
        if norm[i] > 1e-8 {
            output[i] /= norm[i];
        }
    }

    // Trim padding and return requested length
    let pad = n_fft / 2;
    let start = pad.min(output.len());
    let end = (start + output_len).min(output.len());
    output[start..end].to_vec()
}

/// Create a Hann window of given size.
pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Linear interpolation of a 1D signal.
pub fn interp1d(x_old: &[f32], y_old: &[f32], x_new: &[f32]) -> Vec<f32> {
    x_new
        .iter()
        .map(|&x| {
            if x <= x_old[0] {
                return y_old[0];
            }
            if x >= *x_old.last().unwrap() {
                return *y_old.last().unwrap();
            }

            // Binary search for the interval
            let idx = match x_old.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
                Ok(i) => return y_old[i],
                Err(i) => i.saturating_sub(1),
            };

            let idx = idx.min(x_old.len() - 2);
            let t = (x - x_old[idx]) / (x_old[idx + 1] - x_old[idx]);
            y_old[idx] + t * (y_old[idx + 1] - y_old[idx])
        })
        .collect()
}

/// Akima interpolation for smoother curves (used for pitch bending).
pub fn akima_interp(x_old: &[f32], y_old: &[f32], x_new: &[f32]) -> Vec<f32> {
    let n = x_old.len();
    if n < 3 {
        return interp1d(x_old, y_old, x_new);
    }

    // Compute slopes
    let mut m: Vec<f32> = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        m.push((y_old[i + 1] - y_old[i]) / (x_old[i + 1] - x_old[i]));
    }

    // Extend slopes at boundaries
    let m_ext: Vec<f32> = {
        let mut ext = Vec::with_capacity(n + 3);
        ext.push(2.0 * m[0] - m.get(1).copied().unwrap_or(m[0]));
        ext.push(2.0 * m[0] - m.get(0).copied().unwrap_or(m[0]));
        ext.extend_from_slice(&m);
        let last = *m.last().unwrap();
        let prev = m.get(m.len().wrapping_sub(2)).copied().unwrap_or(last);
        ext.push(2.0 * last - prev);
        ext.push(2.0 * last - (2.0 * last - prev));
        ext
    };

    // Akima weights
    let mut t_vals: Vec<f32> = Vec::with_capacity(n);
    for i in 0..n {
        let idx = i + 2;
        let w1 = (m_ext[idx + 1] - m_ext[idx]).abs();
        let w2 = (m_ext[idx - 1] - m_ext[idx - 2]).abs();
        if w1 + w2 > 1e-10 {
            t_vals.push((w1 * m_ext[idx - 1] + w2 * m_ext[idx]) / (w1 + w2));
        } else {
            t_vals.push(0.5 * (m_ext[idx - 1] + m_ext[idx]));
        }
    }

    // Interpolate
    x_new
        .iter()
        .map(|&x| {
            if x <= x_old[0] {
                return y_old[0];
            }
            if x >= *x_old.last().unwrap() {
                return *y_old.last().unwrap();
            }

            let idx = match x_old.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
                Ok(i) => return y_old[i],
                Err(i) => i.saturating_sub(1),
            };
            let idx = idx.min(n - 2);

            let dx = x_old[idx + 1] - x_old[idx];
            let t = (x - x_old[idx]) / dx;
            let a = y_old[idx];
            let b = t_vals[idx] * dx;
            let c = 3.0 * (y_old[idx + 1] - y_old[idx]) - 2.0 * t_vals[idx] * dx - t_vals[idx + 1] * dx;
            let d = 2.0 * (y_old[idx] - y_old[idx + 1]) + t_vals[idx] * dx + t_vals[idx + 1] * dx;

            a + t * (b + t * (c + t * d))
        })
        .collect()
}
