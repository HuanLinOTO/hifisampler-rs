//! Audio I/O and processing utilities.
//!
//! Handles WAV reading/writing, resampling, dynamic range compression,
//! loudness normalization, and tension filtering.

use crate::config::WaveNormTailMode;
use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use num_complex::Complex;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
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
        SampleFormat::Float => reader.into_samples::<f32>().map(|s| s.unwrap()).collect(),
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
    x.iter().map(|&v| (v.max(1e-9) * c).ln()).collect()
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
pub fn pre_emphasis_tension(
    audio: &[f32],
    b: f32,
    sr: u32,
    n_fft: usize,
    hop_size: usize,
) -> Vec<f32> {
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
        let idx = if i < padded.len() {
            i
        } else {
            padded.len() - 1
        };
        center_padded.push(padded[idx]);
    }
    center_padded.extend_from_slice(&padded);
    // Reflect pad right
    for i in 1..=center_pad {
        let idx = if padded.len() > i {
            padded.len() - 1 - i
        } else {
            0
        };
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

// ── ITU-R BS.1770-4 loudness measurement ──────────────────────────────

/// Second-order biquad filter coefficients (Direct Form I).
struct BiquadCoeffs {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
}

/// K-weighting stage 1: high-shelf filter (44 100 Hz).
/// Boosts ~2–4 kHz by ≈ 4 dB — models the acoustic effect of the head.
fn k_weight_shelf_44100() -> BiquadCoeffs {
    BiquadCoeffs {
        b0: 1.53512485958697,
        b1: -2.69169618940638,
        b2: 1.19839281085285,
        a1: -1.69065929318241,
        a2: 0.73248077421585,
    }
}

/// K-weighting stage 2: high-pass (RLB) filter (44 100 Hz).
fn k_weight_highpass_44100() -> BiquadCoeffs {
    BiquadCoeffs {
        b0: 1.0,
        b1: -2.0,
        b2: 1.0,
        a1: -1.99004745483398,
        a2: 0.99007225036621,
    }
}

/// Apply a biquad filter (Direct Form I) in-place, returning a new buffer.
fn apply_biquad(c: &BiquadCoeffs, input: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0f64; input.len()];
    let (mut x1, mut x2) = (0.0f64, 0.0f64);
    let (mut y1, mut y2) = (0.0f64, 0.0f64);
    for (i, &x0) in input.iter().enumerate() {
        let y0 = c.b0 * x0 + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
        out[i] = y0;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    out
}

/// Measure integrated loudness in LUFS (ITU-R BS.1770-4).
///
/// 1. Apply K-weighting (two biquad stages).
/// 2. Split into overlapping 400 ms blocks (75 % overlap → hop = 100 ms).
/// 3. Compute mean-square power per block.
/// 4. Absolute gate: discard blocks < −70 LUFS.
/// 5. Relative gate: discard blocks < (mean of step 4) − 10 dB.
/// 6. Return integrated loudness of remaining blocks.
pub fn measure_lufs(audio: &[f32], sample_rate: u32, block_size: f64) -> f64 {
    if audio.is_empty() {
        return f64::NEG_INFINITY;
    }

    let audio_f64: Vec<f64> = audio.iter().map(|&x| x as f64).collect();

    // K-weighting (currently only 44 100 Hz coefficients)
    let weighted = apply_biquad(
        &k_weight_highpass_44100(),
        &apply_biquad(&k_weight_shelf_44100(), &audio_f64),
    );

    let block_samples = (sample_rate as f64 * block_size).round() as usize;
    let hop_samples = block_samples / 4; // 75 % overlap

    // If the signal is shorter than one block, measure the whole signal.
    if weighted.len() < block_samples {
        let mean_sq: f64 =
            weighted.iter().map(|x| x * x).sum::<f64>() / weighted.len() as f64;
        if mean_sq < 1e-20 {
            return f64::NEG_INFINITY;
        }
        return -0.691 + 10.0 * mean_sq.log10();
    }

    // Block-based mean-square powers
    let mut block_powers: Vec<f64> = Vec::new();
    let mut pos = 0;
    while pos + block_samples <= weighted.len() {
        let block = &weighted[pos..pos + block_samples];
        let mean_sq: f64 = block.iter().map(|x| x * x).sum::<f64>() / block.len() as f64;
        block_powers.push(mean_sq);
        pos += hop_samples;
    }

    // Absolute gate: −70 LUFS  →  threshold in linear power
    let abs_gate = 10f64.powf((-70.0 + 0.691) / 10.0);
    let gated_abs: Vec<f64> = block_powers
        .iter()
        .copied()
        .filter(|&p| p > abs_gate)
        .collect();
    if gated_abs.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Relative gate: mean of absolute-gated − 10 dB
    let mean_abs: f64 = gated_abs.iter().sum::<f64>() / gated_abs.len() as f64;
    let rel_gate = mean_abs * 10f64.powf(-1.0); // −10 dB
    let gated_rel: Vec<f64> = gated_abs
        .iter()
        .copied()
        .filter(|&p| p > rel_gate)
        .collect();
    if gated_rel.is_empty() {
        return f64::NEG_INFINITY;
    }

    let mean_rel: f64 = gated_rel.iter().sum::<f64>() / gated_rel.len() as f64;
    -0.691 + 10.0 * mean_rel.log10()
}

// ── Silence trimming helpers ──────────────────────────────────────────

/// Compute per-frame RMS in dB, returning (rms_db_values, frame_hop_samples).
fn frame_rms_db(audio: &[f32], sample_rate: u32) -> (Vec<f64>, usize) {
    let frame_len = (sample_rate as f64 * 0.02) as usize; // 20 ms window
    let hop_len = (sample_rate as f64 * 0.01) as usize; // 10 ms hop
    let mut values = Vec::new();
    let mut i = 0;
    while i + frame_len <= audio.len() {
        let frame = &audio[i..i + frame_len];
        let rms_val: f64 = frame.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
            / frame_len as f64;
        let rms_val = rms_val.sqrt();
        let db = if rms_val < 1e-10 {
            f64::NEG_INFINITY
        } else {
            20.0 * rms_val.log10()
        };
        values.push(db);
        i += hop_len;
    }
    (values, hop_len)
}

/// Trim silence from audio for loudness measurement (matching Python `trim_silence` logic).
/// Returns `(trimmed_audio, start_sample, end_sample)`.
fn trim_silence_range(
    audio: &[f32],
    sample_rate: u32,
    threshold_db: f64,
) -> (usize, usize) {
    let (rms_values, hop_len) = frame_rms_db(audio, sample_rate);
    let voiced: Vec<usize> = rms_values
        .iter()
        .enumerate()
        .filter(|&(_, &db)| db > threshold_db)
        .map(|(i, _)| i)
        .collect();

    if voiced.is_empty() {
        return (0, audio.len());
    }

    let first = voiced[0];
    let last = *voiced.last().unwrap();
    let padding_frames = ((sample_rate as f64 * 0.1) as usize) / hop_len; // 100 ms padding

    let frame_len = (sample_rate as f64 * 0.02) as usize;
    let start_sample = (first * hop_len).max(0);
    let end_sample = ((last + 1 + padding_frames) * hop_len + frame_len).min(audio.len());

    (start_sample, end_sample)
}

/// Loudness normalization using ITU-R BS.1770-4 LUFS measurement.
///
/// Matches Python `loudness_norm()` from `util/audio.py`:
/// - Optionally trims silence before measurement (`trim_silence`)
/// - Measures integrated loudness with K-weighting + gating
/// - Applies strength-blended gain toward `target_lufs`
/// - Reconstructs full-length output with crossfade if trimmed
pub fn loudness_normalize(
    audio: &[f32],
    sample_rate: u32,
    target_lufs: f64,
    block_size: f64,
    strength: f64,
    trim_silence: bool,
    silence_threshold_db: f64,
    max_boost_db: f64,
    low_level_protect_db: f64,
    tail_peak_limit_dbfs: f64,
    tail_mode: WaveNormTailMode,
) -> Vec<f32> {
    if audio.is_empty() || strength < 0.01 {
        return audio.to_vec();
    }

    let original_length = audio.len();

    // Determine measurement region
    let (trim_start, trim_end) = if trim_silence {
        trim_silence_range(audio, sample_rate, silence_threshold_db)
    } else {
        (0, audio.len())
    };

    let mut measure_audio = audio[trim_start..trim_end].to_vec();

    // Pad if shorter than one block (matching Python)
    let min_block = (sample_rate as f64 * block_size) as usize;
    if measure_audio.len() < min_block {
        let pad_len = min_block - measure_audio.len();
        // reflect-pad
        let orig_len = measure_audio.len();
        measure_audio.reserve(pad_len);
        for i in 0..pad_len {
            let idx = reflect_index(i, orig_len);
            measure_audio.push(measure_audio[idx]);
        }
    }

    // Measure integrated loudness
    let current_lufs = measure_lufs(&measure_audio, sample_rate, block_size);
    if current_lufs.is_infinite() {
        // Signal is essentially silent — don't boost
        return audio.to_vec();
    }

    // Strength-blended target (matches Python):
    //   final_loudness = current + (target - current) * strength / 100
    let final_lufs = current_lufs + (target_lufs - current_lufs) * strength / 100.0;
    let mut gain_db = final_lufs - current_lufs;
    if gain_db > max_boost_db {
        gain_db = max_boost_db;
    }
    let gain = 10f64.powf(gain_db / 20.0) as f32;

    let _ = (trim_start, trim_end, original_length);

    // Silence trimming is only used for LUFS measurement. The gain envelope is
    // applied to the full signal so the body-to-tail relationship stays smooth.
    apply_envelope_aware_gain(
        audio,
        gain,
        sample_rate,
        low_level_protect_db,
        tail_peak_limit_dbfs,
        tail_mode,
    )
}

/// Apply gain with envelope-awareness: full gain on sustained portions,
/// smooth blend to 1.0 (no gain) where the RMS drops.
///
/// This prevents breath tails and natural decays from being over-amplified
/// by loudness normalization. The approach:
///   1. Compute per-frame RMS (20 ms window, 10 ms hop)
///   2. Find the "sustained" RMS level (90th percentile of non-silent frames)
///   3. Where the RMS drops below sustained_level - 10 dB, start reducing gain
///   4. The effective gain blends from `target_gain` to 1.0 based on how far
///      below the sustained level the current RMS is
fn apply_envelope_aware_gain(
    audio: &[f32],
    target_gain: f32,
    sample_rate: u32,
    low_level_protect_db: f64,
    tail_peak_limit_dbfs: f64,
    tail_mode: WaveNormTailMode,
) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }

    // If gain is close to 1.0, nothing to do
    if (target_gain - 1.0).abs() < 0.01 {
        return audio.to_vec();
    }

    let frame_len = (sample_rate as f64 * 0.02) as usize; // 20 ms
    let hop_len = (sample_rate as f64 * 0.01) as usize; // 10 ms

    if audio.len() < frame_len || hop_len == 0 {
        return audio.iter().map(|&x| x * target_gain).collect();
    }

    // Compute per-frame RMS in dB
    let mut frame_rms: Vec<f64> = Vec::new();
    let mut pos = 0;
    while pos + frame_len <= audio.len() {
        let frame = &audio[pos..pos + frame_len];
        let ms: f64 = frame.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>()
            / frame_len as f64;
        let rms = ms.sqrt();
        let db = if rms < 1e-10 {
            -100.0
        } else {
            20.0 * rms.log10()
        };
        frame_rms.push(db);
        pos += hop_len;
    }

    if frame_rms.is_empty() {
        return audio.iter().map(|&x| x * target_gain).collect();
    }

    // Find sustained level: 90th percentile of non-silent frames
    let mut non_silent: Vec<f64> = frame_rms.iter()
        .copied()
        .filter(|&db| db > -60.0)
        .collect();

    if non_silent.is_empty() {
        // Everything is silent — apply uniform gain
        return audio.iter().map(|&x| x * target_gain).collect();
    }

    non_silent.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90_idx = (non_silent.len() as f64 * 0.9) as usize;
    let p90_idx = p90_idx.min(non_silent.len() - 1);
    let sustained_db = non_silent[p90_idx];

    // Gain blend parameters:
    //   - Above (sustained_db - fade_start_db): full gain
    //   - Below (sustained_db - fade_end_db): keep a reduced amount of the
    //     normalization gain so the tail still follows the body naturally.
    //   - In between: linear blend
    let (fade_start_db, fade_end_db, tail_gain_floor_ratio) = match tail_mode {
        WaveNormTailMode::PreserveRelative => (8.0, 24.0, 0.45f32),
        WaveNormTailMode::LegacyProtect => (10.0, 20.0, 0.0f32),
    };
    let threshold_high = sustained_db - fade_start_db;
    let threshold_low = sustained_db - fade_end_db;
    let db_range = threshold_high - threshold_low;

    // Build per-frame gain envelope
    let mut frame_gains: Vec<f32> = Vec::with_capacity(frame_rms.len());
    for &db in &frame_rms {
        let blend = if db >= threshold_high {
            1.0 // full target_gain
        } else if db <= threshold_low {
            tail_gain_floor_ratio
        } else {
            let relative = ((db - threshold_low) / db_range) as f32;
            tail_gain_floor_ratio + (1.0 - tail_gain_floor_ratio) * relative
        };
        // Effective gain: blend between 1.0 and target_gain
        let mut effective_gain = 1.0 + (target_gain - 1.0) * blend;
        // Absolute low-level protection for boost-only case:
        // when frame RMS is below configured threshold, stop the tail from taking
        // the full boost, but keep part of the added gain so the tail is not
        // disconnected from the sustained body.
        if target_gain > 1.0 && db <= low_level_protect_db {
            let protected_gain = 1.0 + (target_gain - 1.0) * tail_gain_floor_ratio;
            effective_gain = effective_gain.min(protected_gain);
        }
        frame_gains.push(effective_gain);
    }

    // Smooth the gain envelope (moving average, 50 ms window) to avoid artifacts
    let smooth_frames = ((sample_rate as f64 * 0.05) / hop_len as f64).round() as usize;
    let smooth_frames = smooth_frames.max(1);
    let smoothed_gains = if smooth_frames > 1 && frame_gains.len() > smooth_frames {
        let mut smoothed = vec![0.0f32; frame_gains.len()];
        let half = smooth_frames / 2;
        for i in 0..frame_gains.len() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(frame_gains.len());
            let sum: f32 = frame_gains[start..end].iter().sum();
            smoothed[i] = sum / (end - start) as f32;
        }
        smoothed
    } else {
        frame_gains
    };

    let tail_peak_limit = 10f64.powf(tail_peak_limit_dbfs / 20.0) as f32;

    // Apply per-sample gain by interpolating the frame-level envelope
    let mut output = vec![0.0f32; audio.len()];
    for i in 0..audio.len() {
        // Map sample index to frame index (with linear interpolation)
        let frame_pos = i as f64 / hop_len as f64;
        let frame_idx = frame_pos as usize;
        let frac = (frame_pos - frame_idx as f64) as f32;

        let g = if frame_idx + 1 < smoothed_gains.len() {
            smoothed_gains[frame_idx] * (1.0 - frac) + smoothed_gains[frame_idx + 1] * frac
        } else if frame_idx < smoothed_gains.len() {
            smoothed_gains[frame_idx]
        } else {
            *smoothed_gains.last().unwrap_or(&target_gain)
        };

        let mut y = audio[i] * g;

        // Local tail peak limit (dBFS) on low-level frames only.
        // This is an engineering safeguard so tails never jump out unexpectedly.
        if target_gain > 1.0 {
            let db = if frame_idx + 1 < frame_rms.len() {
                frame_rms[frame_idx] * (1.0 - frac as f64) + frame_rms[frame_idx + 1] * frac as f64
            } else if frame_idx < frame_rms.len() {
                frame_rms[frame_idx]
            } else {
                *frame_rms.last().unwrap_or(&-100.0)
            };

            if db <= low_level_protect_db {
                if tail_peak_limit > 0.0 {
                    y = y.clamp(-tail_peak_limit, tail_peak_limit);
                }
            }
        }

        output[i] = y;
    }

    output
}

/// Helper: reflect index for padding (same as `reflect_index_mel` but for 1-D).
fn reflect_index(i: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let i_mod = i % period;
    if i_mod < len {
        i_mod
    } else {
        period - i_mod
    }
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
                if x <= x_old[0] {
                    return y_old[0];
                }
                if x >= x_old[1] {
                    return y_old[1];
                }
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
    // w1 = |m[i+3] - m[i+2]|  (= dm[i+2] = f1[i])
    // w2 = |m[i+1] - m[i]|    (= dm[i]   = f2[i])
    //
    // SciPy uses a *relative* threshold: f12 > 1e-9 * max(f12)
    // to decide whether to use the weighted formula or the fallback.
    // The fallback is: t[i] = 0.5 * (m[i+3] + m[i])
    // First, compute all dm values and find the max of f12
    let dm: Vec<f64> = (0..m_ext.len() - 1)
        .map(|i| (m_ext[i + 1] - m_ext[i]).abs())
        .collect();
    // f1[i] = dm[i+2], f2[i] = dm[i]
    // f12[i] = f1[i] + f2[i] = dm[i+2] + dm[i]
    let mut max_f12: f64 = 0.0;
    for i in 0..n {
        let f12 = dm[i + 2] + dm[i];
        if f12 > max_f12 {
            max_f12 = f12;
        }
    }
    let threshold = 1e-9 * max_f12;

    let mut t_vals: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let w1 = dm[i + 2]; // |m[i+3] - m[i+2]|
        let w2 = dm[i]; // |m[i+1] - m[i]|
        let f12 = w1 + w2;
        if f12 > threshold {
            t_vals.push((w1 * m_ext[i + 1] + w2 * m_ext[i + 2]) / f12);
        } else {
            // SciPy fallback: t = 0.5 * (m[i+3] + m[i])
            t_vals.push(0.5 * (m_ext[i + 3] + m_ext[i]));
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
            let c =
                3.0 * (y_old[idx + 1] - y_old[idx]) - 2.0 * t_vals[idx] * dx - t_vals[idx + 1] * dx;
            let d = 2.0 * (y_old[idx] - y_old[idx + 1]) + t_vals[idx] * dx + t_vals[idx + 1] * dx;

            a + t * (b + t * (c + t * d))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::loudness_normalize;
    use crate::config::WaveNormTailMode;

    const SAMPLE_RATE: u32 = 44_100;

    fn rms_db(signal: &[f32]) -> f64 {
        let mean_square = signal
            .iter()
            .map(|&sample| (sample as f64) * (sample as f64))
            .sum::<f64>()
            / signal.len().max(1) as f64;

        if mean_square <= 1e-20 {
            f64::NEG_INFINITY
        } else {
            10.0 * mean_square.log10()
        }
    }

    fn synthetic_voice_with_breath_tail() -> Vec<f32> {
        let total_samples = SAMPLE_RATE as usize;
        let body_len = (SAMPLE_RATE as f32 * 0.7) as usize;
        let tail_len = total_samples - body_len;

        let mut signal = Vec::with_capacity(total_samples);
        for index in 0..body_len {
            let phase = 2.0 * std::f32::consts::PI * 440.0 * index as f32 / SAMPLE_RATE as f32;
            signal.push(0.12 * phase.sin());
        }

        for index in 0..tail_len {
            let t = index as f32 / SAMPLE_RATE as f32;
            let phase = 2.0 * std::f32::consts::PI * 3_500.0 * index as f32 / SAMPLE_RATE as f32;
            let envelope = (-(8.0 * t)).exp();
            let modulator = 0.65 + 0.35 * (2.0 * std::f32::consts::PI * 23.0 * t).sin().abs();
            signal.push(0.02 * envelope * modulator * phase.sin());
        }

        signal
    }

    #[test]
    fn loudness_normalize_preserves_body_tail_balance_within_reason() {
        let audio = synthetic_voice_with_breath_tail();
        let body = &audio[..(SAMPLE_RATE as f32 * 0.7) as usize];
        let tail = &audio[(SAMPLE_RATE as f32 * 0.7) as usize..];
        let delta_before = rms_db(body) - rms_db(tail);

        let normalized = loudness_normalize(
            &audio,
            SAMPLE_RATE,
            -16.0,
            0.400,
            100.0,
            true,
            -52.0,
            18.0,
            -40.0,
            -6.0,
            WaveNormTailMode::PreserveRelative,
        );

        let normalized_body = &normalized[..body.len()];
        let normalized_tail = &normalized[body.len()..];
        let delta_after = rms_db(normalized_body) - rms_db(normalized_tail);

        assert!(
            delta_after - delta_before <= 6.0,
            "body-tail loudness gap changed too much: before={delta_before:.2} dB after={delta_after:.2} dB"
        );
    }

    #[test]
    fn loudness_normalize_does_not_hard_clip_tail_when_boosting() {
        let audio = synthetic_voice_with_breath_tail();
        let normalized = loudness_normalize(
            &audio,
            SAMPLE_RATE,
            -16.0,
            0.400,
            100.0,
            false,
            -52.0,
            18.0,
            -40.0,
            -6.0,
            WaveNormTailMode::PreserveRelative,
        );

        let tail = &normalized[(SAMPLE_RATE as f32 * 0.7) as usize..];
        let tail_peak = tail.iter().fold(0.0f32, |peak, &sample| peak.max(sample.abs()));

        assert!(
            tail_peak < 0.48,
            "tail peak unexpectedly hit hard limiter boundary: {tail_peak:.3}"
        );
    }

    #[test]
    fn legacy_tail_mode_keeps_previous_stronger_separation() {
        let audio = synthetic_voice_with_breath_tail();
        let body = &audio[..(SAMPLE_RATE as f32 * 0.7) as usize];
        let tail = &audio[(SAMPLE_RATE as f32 * 0.7) as usize..];
        let delta_before = rms_db(body) - rms_db(tail);

        let normalized = loudness_normalize(
            &audio,
            SAMPLE_RATE,
            -16.0,
            0.400,
            100.0,
            true,
            -52.0,
            18.0,
            -40.0,
            -6.0,
            WaveNormTailMode::LegacyProtect,
        );

        let normalized_body = &normalized[..body.len()];
        let normalized_tail = &normalized[body.len()..];
        let delta_after = rms_db(normalized_body) - rms_db(normalized_tail);

        assert!(
            delta_after > delta_before,
            "legacy mode should preserve the old stronger tail attenuation: before={delta_before:.2} dB after={delta_after:.2} dB"
        );
    }
}
