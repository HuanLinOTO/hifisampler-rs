//! Mel spectrogram computation.
//!
//! Implements PitchAdjustableMelSpectrogram with key_shift support,
//! using custom mel filterbank construction.

use ndarray::Array2;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::f32::consts::PI;

/// Mel spectrogram analyzer with key_shift support.
pub struct MelAnalyzer {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub num_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

impl MelAnalyzer {
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        hop_size: usize,
        win_size: usize,
        num_mels: usize,
        fmin: f64,
        fmax: f64,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            hop_size,
            win_size,
            num_mels,
            fmin,
            fmax,
        }
    }

    /// Compute mel spectrogram with optional key_shift.
    ///
    /// key_shift: pitch shift in semitones (from gender flag g/100)
    /// speed: time stretch factor (hop_size_interp / hop_size)
    pub fn mel_spectrogram(
        &self,
        audio: &[f32],
        key_shift: f32,
        speed: f32,
    ) -> Array2<f32> {
        // Apply key_shift by modifying effective n_fft and hop_size
        let factor = if key_shift.abs() > 0.001 {
            2.0f64.powf(key_shift as f64 / 12.0)
        } else {
            1.0
        };

        let n_fft_adj = (self.n_fft as f64 * factor).round() as usize;
        // Make n_fft_adj even
        let n_fft_adj = if n_fft_adj % 2 != 0 { n_fft_adj + 1 } else { n_fft_adj };
        let hop_adj = (self.hop_size as f64 * speed as f64).round() as usize;
        let win_adj = (self.win_size as f64 * factor).round() as usize;
        let win_adj = if win_adj % 2 != 0 { win_adj + 1 } else { win_adj };

        // Build mel filterbank for this shift
        let mel_basis = build_mel_filterbank(
            self.sample_rate as f64,
            n_fft_adj,
            self.num_mels,
            self.fmin,
            self.fmax,
        );

        // STFT
        let window = hann_window(win_adj);
        let spec = compute_stft_magnitude(audio, n_fft_adj, hop_adj, &window);

        // Apply mel filterbank: mel = mel_basis @ spec
        let n_frames = spec.ncols();
        let mut mel = Array2::zeros((self.num_mels, n_frames));

        for frame_idx in 0..n_frames {
            for mel_idx in 0..self.num_mels {
                let mut sum = 0.0f32;
                let freq_bins = mel_basis.ncols().min(spec.nrows());
                for freq_idx in 0..freq_bins {
                    sum += mel_basis[[mel_idx, freq_idx]] * spec[[freq_idx, frame_idx]];
                }
                mel[[mel_idx, frame_idx]] = sum;
            }
        }

        // Clamp to avoid log(0)
        mel.mapv_inplace(|v| v.max(1e-5));

        mel
    }
}

/// Build mel filterbank matrix [num_mels, n_fft/2+1].
///
/// Uses HTK-style mel scale (matches Python's librosa.filters.mel with htk=True).
fn build_mel_filterbank(
    sr: f64,
    n_fft: usize,
    num_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;

    // HTK mel scale
    let hz_to_mel = |hz: f64| -> f64 { 2595.0 * (1.0 + hz / 700.0).log10() };
    let mel_to_hz = |mel: f64| -> f64 { 700.0 * (10.0f64.powf(mel / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // Mel points
    let n_points = num_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let freq_per_bin = sr / n_fft as f64;
    let bin_points: Vec<f64> = hz_points
        .iter()
        .map(|&hz| hz / freq_per_bin)
        .collect();

    // Build triangular filters
    let mut filterbank = Array2::zeros((num_mels, freq_bins));

    for i in 0..num_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in 0..freq_bins {
            let freq = j as f64;

            if freq >= left && freq <= center {
                let denom = center - left;
                if denom > 0.0 {
                    filterbank[[i, j]] = ((freq - left) / denom) as f32;
                }
            } else if freq > center && freq <= right {
                let denom = right - center;
                if denom > 0.0 {
                    filterbank[[i, j]] = ((right - freq) / denom) as f32;
                }
            }
        }

        // Slaney normalization
        let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
        for j in 0..freq_bins {
            filterbank[[i, j]] *= enorm as f32;
        }
    }

    filterbank
}

/// Compute STFT magnitude spectrogram [freq_bins, n_frames].
fn compute_stft_magnitude(
    audio: &[f32],
    n_fft: usize,
    hop_size: usize,
    window: &[f32],
) -> Array2<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    // Reflect-pad audio
    let pad = n_fft / 2;
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);

    // Reflect padding at start
    for i in (1..=pad).rev() {
        let idx = if i < audio.len() { i } else { audio.len() - 1 };
        padded.push(audio[idx]);
    }
    padded.extend_from_slice(audio);
    // Reflect padding at end
    for i in 1..=pad {
        let idx = if audio.len() > i + 1 {
            audio.len() - 1 - i
        } else {
            0
        };
        padded.push(audio[idx]);
    }

    let freq_bins = n_fft / 2 + 1;
    let n_frames = (padded.len() - n_fft) / hop_size + 1;
    let mut magnitude = Array2::zeros((freq_bins, n_frames));

    let win_len = window.len();

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        if start + n_fft > padded.len() {
            break;
        }

        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let w = if i < win_len { window[i] } else { 0.0 };
                Complex::new(padded[start + i] * w, 0.0)
            })
            .collect();

        fft.process(&mut frame);

        for freq_idx in 0..freq_bins {
            magnitude[[freq_idx, frame_idx]] = frame[freq_idx].norm();
        }
    }

    magnitude
}

/// Create a Hann window.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Dynamic range compression (log mel).
pub fn dynamic_range_compression(mel: &Array2<f32>, c: f32) -> Array2<f32> {
    let log_c = (1.0 + c).ln();
    mel.mapv(|v| (1.0 + v * c).ln() / log_c)
}
