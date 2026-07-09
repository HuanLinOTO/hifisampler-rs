//! HN-SEP (Harmonic-Noise Separation) — Burn native inference.
//!
//! Replaces ONNX Runtime with `crate::burn::cascadednet::CascadedNet`.

use anyhow::Result;
use num_complex::Complex;
use rustfft::FftPlanner;
use std::path::Path;
use tracing::info;

use crate::burn::cascadednet::CascadedNet;
use crate::ep::{select_burn_device, VocoderBackend};

/// HN-SEP model using Burn.
pub struct HnsepModel {
    model: Option<CascadedNet<VocoderBackend>>,
    n_fft: usize,
    hop_length: usize,
}

impl HnsepModel {
    /// Load the HN-SEP model from a Burnpack (.bpk) or PyTorch (.pt) checkpoint.
    ///
    /// Format is detected by file extension:
    /// - `.bpk` → BurnpackStore (Burn native, fast load, recommended)
    /// - `.pt`  → PytorchStore (legacy, ~20x slower load, kept for backward compat)
    pub fn load(
        model_path: impl AsRef<Path>,
        n_fft: usize,
        hop_length: usize,
        device: &str,
        _device_id: i32,
        _num_threads: usize,
    ) -> Result<Self> {
        let path = model_path.as_ref().to_path_buf();
        info!("[hnsep] step 1: selecting burn device (device={})", device);

        let burn_device = select_burn_device(device);
        info!("[hnsep] step 2: device selected, initializing CascadedNet config");

        let is_bpk = path.extension().and_then(|e| e.to_str()) == Some("bpk");
        info!(
            "[hnsep] step 3: loading weights from {} (format={})",
            path.display(),
            if is_bpk { "bpk" } else { "pt" }
        );

        let model = if is_bpk {
            crate::burn::cascadednet::CascadedNet::load_from_bpk(&path, &burn_device)?
        } else {
            crate::burn::cascadednet::CascadedNet::load_from_pt(&path, &burn_device)?
        };
        info!("[hnsep] step 4: model loaded");

        Ok(Self {
            model: Some(model),
            n_fft,
            hop_length,
        })
    }

    /// Separate harmonic component from audio.
    pub fn predict_from_audio(&mut self, audio: &[f32], _sample_rate: u32) -> Result<Vec<f32>> {
        // Run directly on current thread — CUDA context is thread-local and
        // must stay on the thread that initialized the device.
        let model = self.model.as_mut().expect("hnsep model missing");
        Self::predict_inner(model, audio, self.n_fft, self.hop_length)
    }

    fn predict_inner(
        model: &mut CascadedNet<VocoderBackend>,
        audio: &[f32],
        n_fft: usize,
        hop_length: usize,
    ) -> Result<Vec<f32>> {
        let freq_bins = n_fft / 2 + 1;
        info!("[hnsep] predict: audio len={}, n_fft={}, hop={}", audio.len(), n_fft, hop_length);

        // Compute STFT
        let window = hann_window(n_fft);
        let stft_frames = compute_stft(audio, n_fft, hop_length, &window);
        let n_frames = stft_frames.len();
        info!("[hnsep] predict: STFT done, n_frames={}", n_frames);

        // Pad time frames to be divisible by 16 (for U-Net)
        let pad_frames = ((n_frames + 15) / 16) * 16;
        info!("[hnsep] predict: pad_frames={}", pad_frames);

        let max_bin = n_fft / 2;
        let device = burn::tensor::Device::<VocoderBackend>::default();

        // Build Burn input: [1, 2, max_bin, pad_frames] (real, imag)
        let mut input_data = vec![0.0f32; 1 * 2 * max_bin * pad_frames];
        for t in 0..n_frames {
            for f in 0..freq_bins.min(max_bin).min(stft_frames[t].len()) {
                let val = stft_frames[t][f];
                // [1, 2, max_bin, pad_frames] row-major: c*max_bin*pad + f*pad + t
                input_data[0 * max_bin * pad_frames + f * pad_frames + t] = val.re;
                input_data[1 * max_bin * pad_frames + f * pad_frames + t] = val.im;
            }
        }

        let input_tensor = burn::tensor::Tensor::<VocoderBackend, 4>::from_data(
            burn::tensor::TensorData::new(input_data, [1, 2, max_bin, pad_frames]),
            &device,
        );
        info!("[hnsep] predict: input tensor created, calling model.forward");

        // Run inference → mask [1, 2, output_bin, pad_frames]
        let mask_tensor = model.forward(input_tensor);
        info!("[hnsep] predict: forward done, extracting data");

        let mask_data = mask_tensor.into_data();
        let mask_slice: Vec<f32> = mask_data
            .as_slice::<<VocoderBackend as burn::tensor::backend::BackendTypes>::FloatElem>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| {
                use burn::tensor::ElementConversion;
                v.elem::<f32>()
            })
            .collect();

        // Apply mask to STFT
        let output_bin = freq_bins;
        let mut masked_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let mut frame = Vec::with_capacity(freq_bins);
            for f in 0..freq_bins {
                // mask_slice layout: [1, 2, output_bin, pad_frames]
                let mask_re = mask_slice[f * pad_frames + t];
                let mask_im = mask_slice[output_bin * pad_frames + f * pad_frames + t];
                let mask_val = Complex::new(mask_re, mask_im);
                let spec = stft_frames[t][f];
                frame.push(spec * mask_val);
            }
            masked_frames.push(frame);
        }

        // ISTFT
        let harmonic = compute_istft(
            &masked_frames,
            n_fft,
            hop_length,
            &window,
            audio.len(),
        );

        // Optional dump for validation.
        if let Ok(dir) = std::env::var("HIFISAMPLER_DUMP_HNSEP") {
            let idx = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::path::Path::new(&dir).join(format!("hnsep_burn_{idx}.bin"));
            dump_hnsep_io(&path, audio, &harmonic)
                .unwrap_or_else(|e| tracing::warn!("failed to dump hnsep io: {e}"));
        }

        Ok(harmonic)
    }
}

/// Dump hnsep input/output (same format as ORT golden).
fn dump_hnsep_io(
    path: &std::path::Path,
    audio: &[f32],
    harmonic: &[f32],
) -> std::io::Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"HNSEP")?;
    f.write_all(&(audio.len() as u32).to_le_bytes())?;
    f.write_all(&(harmonic.len() as u32).to_le_bytes())?;
    let mut buf = Vec::with_capacity((audio.len() + harmonic.len()) * 4);
    for &v in audio {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in harmonic {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)?;
    Ok(())
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

        let mut full: Vec<Complex<f32>> = Vec::with_capacity(n_fft);
        full.extend_from_slice(frame);
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

    for i in 0..output.len() {
        if norm[i] > 1e-8 {
            output[i] /= norm[i];
        }
    }

    let pad = n_fft / 2;
    let start = pad.min(output.len());
    let end = (start + output_len).min(output.len());
    output[start..end].to_vec()
}

/// Create Hann window.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos()))
        .collect()
}
