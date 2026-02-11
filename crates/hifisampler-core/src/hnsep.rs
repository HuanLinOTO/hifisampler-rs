//! HN-SEP (Harmonic-Noise Separation) ONNX inference wrapper.

use anyhow::Result;
use num_complex::Complex;
use ort::execution_providers::DirectMLExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use rustfft::FftPlanner;
use std::path::Path;
use tracing::info;

/// HN-SEP model for harmonic/noise separation using ONNX Runtime.
pub struct HnsepModel {
    session: Session,
    n_fft: usize,
    hop_length: usize,
}

impl HnsepModel {
    /// Load the HN-SEP ONNX model.
    ///
    /// `device` controls the execution provider:
    /// - `"directml"` or `"auto"` → try DirectML first, fall back to CPU
    /// - `"cpu"` → CPU only
    pub fn load(
        model_path: impl AsRef<Path>,
        n_fft: usize,
        hop_length: usize,
        device: &str,
        num_threads: usize,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        info!("Loading HN-SEP model from: {}", path.display());

        let mut builder = Session::builder()?;

        let use_directml = matches!(device.to_lowercase().as_str(), "auto" | "directml" | "dml");
        if use_directml {
            info!("Attempting to register DirectML execution provider for HN-SEP...");
            builder = builder.with_execution_providers([
                DirectMLExecutionProvider::default().build(),
            ])?;
        }

        let threads = if num_threads == 0 { 4 } else { num_threads };
        let session = builder
            .with_intra_threads(threads)?
            .commit_from_file(path)?;

        info!("HN-SEP model loaded successfully (device={})", device);
        Ok(Self {
            session,
            n_fft,
            hop_length,
        })
    }

    /// Separate harmonic component from audio.
    ///
    /// Returns the harmonic audio signal.
    pub fn predict_from_audio(&mut self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        let freq_bins = self.n_fft / 2 + 1;

        // Compute STFT
        let window = hann_window(self.n_fft);
        let stft_frames = compute_stft(audio, self.n_fft, self.hop_length, &window);
        let n_frames = stft_frames.len();

        // Pad time frames to be divisible by 16 (for U-Net)
        let pad_frames = ((n_frames + 15) / 16) * 16;

        // Build ONNX input: [1, 2, freq_bins, time] where channel 0=real, 1=imag
        let mut input_data = vec![0.0f32; 1 * 2 * freq_bins * pad_frames];
        for t in 0..n_frames {
            for f in 0..freq_bins.min(stft_frames[t].len()) {
                let val = stft_frames[t][f];
                // [1, 2, freq_bins, pad_frames] in row-major: idx = c*freq_bins*pad_frames + f*pad_frames + t
                input_data[0 * freq_bins * pad_frames + f * pad_frames + t] = val.re;
                input_data[1 * freq_bins * pad_frames + f * pad_frames + t] = val.im;
            }
        }

        let input_tensor = Tensor::from_array(
            ([1usize, 2, freq_bins, pad_frames], input_data),
        )?;

        // Run inference
        let outputs = self.session.run(
            ort::inputs![
                "input" => input_tensor,
            ],
        )?;

        // Extract mask: [1, 2, freq_bins, time] as flat slice
        let (_mask_shape, mask_data) = outputs["output"]
            .try_extract_tensor::<f32>()?;

        // mask_shape should be [1, 2, freq_bins, pad_frames]
        // Index function for [1, 2, freq_bins, pad_frames] row-major
        let mask_idx = |c: usize, f: usize, t: usize| -> usize {
            c * freq_bins * pad_frames + f * pad_frames + t
        };

        // Apply mask to STFT
        let mut masked_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let mut frame = Vec::with_capacity(freq_bins);
            for f in 0..freq_bins {
                let mask_re = mask_data[mask_idx(0, f, t)];
                let mask_im = mask_data[mask_idx(1, f, t)];
                let mask_val = Complex::new(mask_re, mask_im);

                let spec = stft_frames[t][f];
                // Complex multiplication for masking
                frame.push(spec * mask_val);
            }
            masked_frames.push(frame);
        }

        // ISTFT to get harmonic audio
        let harmonic = compute_istft(&masked_frames, self.n_fft, self.hop_length, &window, audio.len());

        Ok(harmonic)
    }
}

/// Compute STFT of audio signal.
fn compute_stft(
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

    let freq_bins = n_fft / 2 + 1;
    let mut frames = Vec::new();

    let mut pos = 0;
    while pos + n_fft <= padded.len() {
        let mut frame: Vec<Complex<f32>> = (0..n_fft)
            .map(|i| {
                let w = if i < window.len() { window[i] } else { 0.0 };
                Complex::new(padded[pos + i] * w, 0.0)
            })
            .collect();

        fft.process(&mut frame);
        frames.push(frame[..freq_bins].to_vec());

        pos += hop_size;
    }

    frames
}

/// Compute ISTFT via overlap-add.
fn compute_istft(
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

        // Reconstruct full spectrum
        let mut full: Vec<Complex<f32>> = Vec::with_capacity(n_fft);
        full.extend_from_slice(frame);
        // Mirror conjugate for negative frequencies
        for i in (1..n_fft / 2).rev() {
            full.push(frame[i].conj());
        }
        full.resize(n_fft, Complex::new(0.0, 0.0));

        ifft.process(&mut full);

        let win_len = window.len();
        for i in 0..n_fft {
            if pos + i < output.len() {
                let w = if i < win_len { window[i] } else { 0.0 };
                output[pos + i] += full[i].re * inv_n * w;
                norm[pos + i] += w * w;
            }
        }
    }

    // Normalize
    for i in 0..output.len() {
        if norm[i] > 1e-8 {
            output[i] /= norm[i];
        }
    }

    // Extract original length
    let pad = n_fft / 2;
    let start = pad.min(output.len());
    let end = (start + output_len).min(output.len());
    output[start..end].to_vec()
}

/// Create Hann window.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        })
        .collect()
}
