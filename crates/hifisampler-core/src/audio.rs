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

/// Dynamic range compression: `log(clamp(x, min=1e-9) * C)`
/// Matches Python's `dynamic_range_compression_torch(x, C=1, clip_val=1e-9)`.
pub fn dynamic_range_compression(x: &[f32], c: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| (v.max(1e-9) * c).ln())
        .collect()
}

/// Pre-emphasis base tension filter.
/// Matches Python's `pre_emphasis_base_tension(wave, b)` exactly.
///
/// Python logic:
///   spec = stft(wave)
///   spec_amp = abs(spec)
///   spec_phase = atan2(spec.imag, spec.real)
///   spec_amp_db = log(clamp(spec_amp, 1e-9))
///   x0 = fft_bin / ((sr/2) / 1500)
///   freq_filter = (-b / x0) * arange(fft_bin) + b
///   spec_amp_db += clamp(freq_filter, -2, 2)
///   spec_amp = exp(spec_amp_db)
///   filtered = istft(spec_amp * exp(j*phase))
///   filtered *= (original_max / filtered_max) * (clip(b/(-15), 0, 0.33) + 1)
pub fn pre_emphasis_tension(audio: &[f32], b: f32, sr: u32, n_fft: usize, hop_size: usize) -> Vec<f32> {
    if b.abs() < 0.01 {
        return audio.to_vec();
    }

    let original_length = audio.len();

    // Pad to be divisible by hop_size (matching Python)
    let pad_length = (hop_size - (original_length % hop_size)) % hop_size;
    let mut padded = audio.to_vec();
    padded.resize(original_length + pad_length, 0.0);

    let win_size = n_fft; // Python: win_length=CONFIG.win_size (=n_fft=2048)

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    // Create Hann window (matching torch.hann_window)
    let window: Vec<f32> = (0..win_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / win_size as f32).cos()))
        .collect();

    let freq_bins = n_fft / 2 + 1;

    // Python: x0 = fft_bin / ((sample_rate / 2) / 1500)
    let x0 = freq_bins as f32 / ((sr as f32 / 2.0) / 1500.0);

    // freq_filter = (-b / x0) * arange(fft_bin) + b, clamped to [-2, 2]
    let freq_filter: Vec<f32> = (0..freq_bins)
        .map(|i| ((-b / x0) * i as f32 + b).clamp(-2.0, 2.0))
        .collect();

    // Python uses torch.stft (center=True by default, with n_fft/2 reflect pad)
    // Apply center padding
    let center_pad = n_fft / 2;
    let mut center_padded = Vec::with_capacity(center_pad + padded.len() + center_pad);
    // Reflect pad left
    for i in (1..=center_pad).rev() {
        let idx = if i < padded.len() { i } else { padded.len() - 1 };
        center_padded.push(padded[idx]);
    }
    center_padded.extend_from_slice(&padded);
    // Reflect pad right
    for i in 1..=center_pad {
        let idx = if padded.len() > i { padded.len() - 1 - i } else { 0 };
        center_padded.push(padded[idx]);
    }

    // STFT
    let n_frames = (center_padded.len() - win_size) / hop_size + 1;
    let mut stft_complex: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let sample = if i < win_size && start + i < center_padded.len() {
                    center_padded[start + i] * window[i]
                } else {
                    0.0
                };
                Complex::new(sample, 0.0)
            })
            .collect();

        fft.process(&mut frame);
        stft_complex.push(frame[..freq_bins].to_vec());
    }

    // Apply tension filter in log domain (matching Python exactly)
    let mut filtered_stft: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);
    for frame in &stft_complex {
        let mut new_frame = Vec::with_capacity(freq_bins);
        for f in 0..freq_bins {
            let spec = frame[f];
            let amp = spec.norm();
            let phase = spec.im.atan2(spec.re);

            // spec_amp_db = log(clamp(amp, 1e-9))
            let amp_db = amp.max(1e-9).ln();

            // spec_amp_db += freq_filter
            let amp_db = amp_db + freq_filter[f];

            // spec_amp = exp(spec_amp_db)
            let new_amp = amp_db.exp();

            // Reconstruct complex: new_amp * exp(j * phase)
            new_frame.push(Complex::new(new_amp * phase.cos(), new_amp * phase.sin()));
        }
        filtered_stft.push(new_frame);
    }

    // ISTFT (overlap-add with Griffin-Lim style normalization)
    let total_len = (n_frames - 1) * hop_size + n_fft;
    let mut output = vec![0.0f32; total_len];
    let mut norm = vec![0.0f32; total_len];
    let inv_n = 1.0 / n_fft as f32;

    for (idx, frame) in filtered_stft.iter().enumerate() {
        let pos = idx * hop_size;

        // Reconstruct full spectrum (mirror conjugate for negative frequencies)
        let mut full: Vec<Complex<f32>> = frame.clone();
        for i in (1..n_fft / 2).rev() {
            full.push(frame[i].conj());
        }
        full.resize(n_fft, Complex::new(0.0, 0.0));

        ifft.process(&mut full);

        for i in 0..win_size {
            if pos + i < output.len() {
                output[pos + i] += full[i].re * inv_n * window[i];
                norm[pos + i] += window[i] * window[i];
            }
        }
    }

    // Normalize overlap-add
    for i in 0..output.len() {
        if norm[i] > 1e-8 {
            output[i] /= norm[i];
        }
    }

    // Remove center padding, get padded-length signal
    let filtered_padded: Vec<f32> = output[center_pad..center_pad + padded.len()].to_vec();

    // Amplitude normalization (matching Python):
    // filtered_wave *= (original_max / filtered_max) * (clip(b/(-15), 0, 0.33) + 1)
    let original_max = peak(&padded);
    let filtered_max = peak(&filtered_padded);

    let amp_correction = if filtered_max > 1e-8 {
        (original_max / filtered_max) * ((b / -15.0).clamp(0.0, 0.33) + 1.0)
    } else {
        1.0
    };

    // Return only original_length samples
    filtered_padded[..original_length]
        .iter()
        .map(|&x| x * amp_correction)
        .collect()
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
/// Uses f64 internally to match SciPy's Akima1DInterpolator precision.
pub fn akima_interp(x_old: &[f32], y_old: &[f32], x_new: &[f32]) -> Vec<f32> {
    let x64: Vec<f64> = x_old.iter().map(|&v| v as f64).collect();
    let y64: Vec<f64> = y_old.iter().map(|&v| v as f64).collect();
    let xn64: Vec<f64> = x_new.iter().map(|&v| v as f64).collect();
    akima_interp_f64(&x64, &y64, &xn64)
        .iter()
        .map(|&v| v as f32)
        .collect()
}

/// Akima interpolation in f64 — exact match with SciPy `Akima1DInterpolator`.
///
/// Boundary extension follows SciPy `_cubic.py`:
///   m[1] = 2*m[2] - m[3]
///   m[0] = 2*m[1] - m[2]
///   m[-2] = 2*m[-3] - m[-4]
///   m[-1] = 2*m[-2] - m[-3]
pub fn akima_interp_f64(x_old: &[f64], y_old: &[f64], x_new: &[f64]) -> Vec<f64> {
    let n = x_old.len();
    if n < 2 {
        return vec![y_old.first().copied().unwrap_or(0.0); x_new.len()];
    }
    if n == 2 {
        // Linear interpolation
        return x_new
            .iter()
            .map(|&x| {
                if x <= x_old[0] { return y_old[0]; }
                if x >= x_old[1] { return y_old[1]; }
                let t = (x - x_old[0]) / (x_old[1] - x_old[0]);
                y_old[0] + t * (y_old[1] - y_old[0])
            })
            .collect();
    }

    // Compute slopes between data points: m has (n-1) elements
    let nm = n - 1;
    let mut slopes = Vec::with_capacity(nm);
    for i in 0..nm {
        slopes.push((y_old[i + 1] - y_old[i]) / (x_old[i + 1] - x_old[i]));
    }

    // Build extended slope array with 2 boundary extensions on each side
    // Total length = nm + 4 = (n-1) + 4 = n + 3
    // Layout: [ext0, ext1, slopes[0], slopes[1], ..., slopes[nm-1], ext_r0, ext_r1]
    //          idx:  0      1         2          3                   nm+1    nm+2    nm+3
    // SciPy convention: actual slopes are at indices 2..(nm+2) exclusive
    let mut m_ext: Vec<f64> = Vec::with_capacity(nm + 4);
    // Left boundary: m[1] = 2*m[2] - m[3], m[0] = 2*m[1] - m[2]
    let left1 = 2.0 * slopes[0] - slopes.get(1).copied().unwrap_or(slopes[0]);
    let left0 = 2.0 * left1 - slopes[0];
    m_ext.push(left0);
    m_ext.push(left1);
    m_ext.extend_from_slice(&slopes);
    // Right boundary: m[-2] = 2*m[-3] - m[-4], m[-1] = 2*m[-2] - m[-3]
    let last = slopes[nm - 1];
    let prev = if nm >= 2 { slopes[nm - 2] } else { last };
    let right0 = 2.0 * last - prev;
    let right1 = 2.0 * right0 - last;
    m_ext.push(right0);
    m_ext.push(right1);

    // Compute Akima derivative at each data point
    // For point i, the relevant m_ext indices are (i, i+1, i+2, i+3)
    // w1 = |m[i+3] - m[i+2]|, w2 = |m[i+1] - m[i]|
    // t[i] = (w1*m[i+1] + w2*m[i+2]) / (w1+w2)  if w1+w2 > 0
    //      = 0.5*(m[i+1] + m[i+2])               otherwise
    let mut t_vals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let w1 = (m_ext[i + 3] - m_ext[i + 2]).abs();
        let w2 = (m_ext[i + 1] - m_ext[i]).abs();
        if w1 + w2 > 1e-30 {
            t_vals.push((w1 * m_ext[i + 1] + w2 * m_ext[i + 2]) / (w1 + w2));
        } else {
            t_vals.push(0.5 * (m_ext[i + 1] + m_ext[i + 2]));
        }
    }

    // Evaluate cubic Hermite on each query point
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
