//! Mel spectrogram computation.
//!
//! Reimplemented to **exactly match** the Python `wav2mel.py` +
//! `librosa.filters.mel` pipeline used by the original HiFiSampler.
//!
//! Key details that must be identical to Python:
//! - Mel filterbank: **Slaney** scale (NOT HTK) — this is librosa's default
//! - STFT: center=False, pad with reflect on the *input*
//! - Window: hann_window(win_size_new), padded as `(win-hop)//2, (win-hop+1)//2`
//! - DRC: `log(clamp(x, 1e-9))` — plain natural log

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

    /// Compute mel spectrogram matching Python's PitchAdjustableMelSpectrogram.__call__
    ///
    /// key_shift: pitch shift in semitones (gender flag g/100)
    /// speed: time stretch factor (usually 1.0)
    ///
    /// Python mel_analyzer is initialized with hop_length=origin_hop_size (128).
    /// speed parameter scales hop_length: `hop_length = int(np.round(self.hop_length * speed))`
    /// In generate_features, Python passes speed=1.0, so effective hop = origin_hop_size = 128.
    pub fn mel_spectrogram(&self, audio: &[f32], key_shift: f32, speed: f32) -> Array2<f32> {
        let factor = 2.0f64.powf(key_shift as f64 / 12.0);
        let n_fft_new = (self.n_fft as f64 * factor).round() as usize;
        let win_size_new = (self.win_size as f64 * factor).round() as usize;
        let hop_length = (self.hop_size as f64 * speed as f64).round() as usize;

        // Build mel filterbank using ORIGINAL n_fft (not shifted)
        // Python: mel_basis is keyed by fmax only, built with self.n_fft
        let mel_basis = build_mel_filterbank_slaney(
            self.sample_rate as f64,
            self.n_fft,
            self.num_mels,
            self.fmin,
            self.fmax,
        );

        // Window
        let window = hann_window(win_size_new);

        // Pad audio: Python does
        //   F.pad(y.unsqueeze(1),
        //         (int((win_size_new - hop_length) // 2),
        //          int((win_size_new - hop_length + 1) // 2)),
        //         mode='reflect')
        let pad_left = (win_size_new.saturating_sub(hop_length)) / 2;
        let pad_right = (win_size_new.saturating_sub(hop_length) + 1) / 2;
        let padded = reflect_pad(audio, pad_left, pad_right);

        // STFT with center=False
        let spec = compute_stft_magnitude(&padded, n_fft_new, hop_length, &window);

        // If key_shift != 0, resize spectrum to original n_fft/2+1 bins
        // Python:
        //   size = self.n_fft // 2 + 1
        //   resize = spec.size(1)
        //   if resize < size: spec = F.pad(spec, (0, 0, 0, size - resize))
        //   spec = spec[:, :size, :] * self.win_size / win_size_new
        let original_freq_bins = self.n_fft / 2 + 1;
        let spec = if key_shift.abs() > 0.001 {
            let current_bins = spec.nrows();
            let n_frames = spec.ncols();
            let mut resized = Array2::zeros((original_freq_bins, n_frames));

            let copy_bins = current_bins.min(original_freq_bins);
            let scale = self.win_size as f32 / win_size_new as f32;
            for f in 0..copy_bins {
                for t in 0..n_frames {
                    resized[[f, t]] = spec[[f, t]] * scale;
                }
            }
            resized
        } else {
            spec
        };

        // Apply mel filterbank: mel = mel_basis @ spec
        let n_frames = spec.ncols();
        let freq_bins = spec.nrows();
        let fb_bins = mel_basis.ncols();
        let apply_bins = freq_bins.min(fb_bins);

        let mut mel = Array2::zeros((self.num_mels, n_frames));
        for t in 0..n_frames {
            for m in 0..self.num_mels {
                let mut sum = 0.0f32;
                for f in 0..apply_bins {
                    sum += mel_basis[[m, f]] * spec[[f, t]];
                }
                mel[[m, t]] = sum;
            }
        }

        mel
    }
}

/// Build mel filterbank using **Slaney** scale (librosa default).
///
/// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` exactly.
/// Slaney scale: linear below 1000Hz, logarithmic above.
fn build_mel_filterbank_slaney(
    sr: f64,
    n_fft: usize,
    num_mels: usize,
    fmin: f64,
    fmax: f64,
) -> Array2<f32> {
    let freq_bins = n_fft / 2 + 1;

    // Slaney mel scale
    let f_sp: f64 = 200.0 / 3.0;
    let min_log_hz: f64 = 1000.0;
    let min_log_mel: f64 = min_log_hz / f_sp;
    let logstep: f64 = 6.4f64.ln() / 27.0;

    let hz_to_mel = |hz: f64| -> f64 {
        if hz >= min_log_hz {
            min_log_mel + (hz / min_log_hz).ln() / logstep
        } else {
            hz / f_sp
        }
    };

    let mel_to_hz = |mel: f64| -> f64 {
        if mel >= min_log_mel {
            min_log_hz * (logstep * (mel - min_log_mel)).exp()
        } else {
            f_sp * mel
        }
    };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    let n_points = num_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // FFT frequencies
    let fft_freqs: Vec<f64> = (0..freq_bins)
        .map(|i| sr * i as f64 / n_fft as f64)
        .collect();

    // fdiff
    let fdiff: Vec<f64> = hz_points.windows(2).map(|w| w[1] - w[0]).collect();

    // ramps[i][j] = hz_points[i] - fft_freqs[j]
    // lower = -ramps[:-2] / fdiff[:-1]  => (fft_freqs[j] - hz_points[i]) / fdiff[i]
    // upper = ramps[2:] / fdiff[1:]     => (hz_points[i+2] - fft_freqs[j]) / fdiff[i+1]
    // weights = max(0, min(lower, upper))
    // enorm = 2.0 / (hz_points[i+2] - hz_points[i])
    // weights *= enorm

    let mut filterbank = Array2::zeros((num_mels, freq_bins));

    for i in 0..num_mels {
        for j in 0..freq_bins {
            let lower = (fft_freqs[j] - hz_points[i]) / fdiff[i];
            let upper = (hz_points[i + 2] - fft_freqs[j]) / fdiff[i + 1];
            let val = lower.min(upper).max(0.0);

            let enorm = 2.0 / (hz_points[i + 2] - hz_points[i]);
            filterbank[[i, j]] = (val * enorm) as f32;
        }
    }

    filterbank
}

/// Compute STFT magnitude spectrogram [freq_bins, n_frames].
/// center=False — audio is already padded.
fn compute_stft_magnitude(
    audio: &[f32],
    n_fft: usize,
    hop_size: usize,
    window: &[f32],
) -> Array2<f32> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let freq_bins = n_fft / 2 + 1;
    let win_len = window.len();

    // Number of frames (matching torch.stft center=False)
    let n_frames = if audio.len() >= win_len {
        (audio.len() - win_len) / hop_size + 1
    } else {
        0
    };

    let mut magnitude = Array2::zeros((freq_bins, n_frames));

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;

        // Zero-pad the FFT buffer if window is smaller than n_fft
        let mut frame = vec![Complex::new(0.0f32, 0.0); n_fft];
        for i in 0..win_len.min(n_fft) {
            let sample = if start + i < audio.len() {
                audio[start + i]
            } else {
                0.0
            };
            frame[i] = Complex::new(sample * window[i], 0.0);
        }

        fft.process(&mut frame);

        for f in 0..freq_bins {
            magnitude[[f, frame_idx]] = frame[f].norm();
        }
    }

    magnitude
}

/// Reflect-pad audio (matching PyTorch F.pad mode='reflect').
///
/// PyTorch F.pad(x, (pad_left, pad_right), mode='reflect') for x=[a0,a1,a2,a3,a4]:
///   pad_left=3 → [a3, a2, a1, | a0, a1, a2, a3, a4]
///   pad_right=2 → [a0, a1, a2, a3, a4, | a3, a2]
fn reflect_pad(audio: &[f32], pad_left: usize, pad_right: usize) -> Vec<f32> {
    let n = audio.len();
    if n <= 1 {
        return vec![audio.first().copied().unwrap_or(0.0); pad_left + n + pad_right];
    }

    let mut padded = Vec::with_capacity(pad_left + n + pad_right);

    // Left pad: indices pad_left, pad_left-1, ..., 1 (reflected from start)
    for i in (1..=pad_left).rev() {
        padded.push(audio[reflect_index(i, n)]);
    }

    padded.extend_from_slice(audio);

    // Right pad: indices n, n+1, ..., n+pad_right-1 (reflected from end)
    for i in 1..=pad_right {
        padded.push(audio[reflect_index(n - 1 + i, n)]);
    }

    padded
}

/// Compute reflect index for padding.
fn reflect_index(idx: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = 2 * (len - 1);
    let idx = idx % period;
    if idx < len { idx } else { period - idx }
}

/// Create a Hann window (matching torch.hann_window).
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / size as f32).cos()))
        .collect()
}

/// Dynamic range compression: `log(clamp(x, min=1e-9))`
///
/// **Matches Python**: `torch.log(torch.clamp(x, min=clip_val) * C)` with C=1, clip_val=1e-9
pub fn dynamic_range_compression(mel: &Array2<f32>, _c: f32) -> Array2<f32> {
    mel.mapv(|v| v.max(1e-9).ln())
}
