//! HiFi-GAN vocoder — Burn native inference (ONNX Runtime removed).

use anyhow::Result;
use ndarray::Array2;
use std::path::Path;
use tracing::info;

use crate::burn::vocoder::HifiGanNsf;
use crate::ep::{select_burn_device, VocoderBackend};

/// HiFi-GAN vocoder using Burn.
pub struct Vocoder {
    model: HifiGanNsf<VocoderBackend>,
}

impl Vocoder {
    /// Load a HiFi-GAN model from a Burnpack (.bpk) or PyTorch (.pt) checkpoint.
    ///
    /// Format is detected by file extension:
    /// - `.bpk` → BurnpackStore (Burn native, fast load, recommended)
    /// - `.pt`  → PytorchStore (legacy, ~20x slower load, kept for backward compat)
    ///
    /// `device` controls the Burn backend — see [`crate::ep::select_burn_device`].
    pub fn load(
        model_path: impl AsRef<Path>,
        _model_type: &str,
        device: &str,
        _device_id: i32,
        _num_threads: usize,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        info!("[vocoder] step 1: loading from {}", path.display());

        let burn_device = select_burn_device(device);
        let is_bpk = path.extension().and_then(|e| e.to_str()) == Some("bpk");
        info!(
            "[vocoder] step 2: device ready, loading weights (format={})",
            if is_bpk { "bpk" } else { "pt" }
        );
        let model = if is_bpk {
            HifiGanNsf::load_from_bpk(path, &burn_device)?
        } else {
            HifiGanNsf::load_from_pt(path, &burn_device)?
        };

        info!("[vocoder] step 3: vocoder loaded (device={device})");
        Ok(Self { model })
    }

    /// Synthesize waveform from mel spectrogram and F0.
    ///
    /// mel: `[num_mels, n_frames]`, f0: `[n_frames]` -> `[total_samples]`
    pub fn synthesize(&mut self, mel: &Array2<f32>, f0: &[f32]) -> Result<Vec<f32>> {
        let n_frames = mel.ncols();
        let num_mels = mel.nrows();
        info!("[vocoder] synthesize: n_frames={}, num_mels={}", n_frames, num_mels);

        let device = burn::tensor::Device::<VocoderBackend>::default();

        // Burn model expects mel [1, num_mels, n_frames]; Array2 is [num_mels, n_frames]
        let mel_vec: Vec<f32> = mel.iter().copied().collect();
        let mel_tensor = burn::tensor::Tensor::<VocoderBackend, 3>::from_data(
            burn::tensor::TensorData::new(mel_vec, [1, num_mels, n_frames]),
            &device,
        );

        let f0_data: Vec<f32> = f0[..n_frames].to_vec();
        let f0_tensor = burn::tensor::Tensor::<VocoderBackend, 2>::from_data(
            burn::tensor::TensorData::new(f0_data, [1, n_frames]),
            &device,
        );
        info!("[vocoder] synthesize: tensors ready, calling forward");

        let output = self.model.forward(mel_tensor, f0_tensor);
        info!("[vocoder] synthesize: forward done, extracting wav");

        let data = output.into_data();
        let wav: Vec<f32> = data
            .as_slice::<<VocoderBackend as burn::tensor::backend::BackendTypes>::FloatElem>()
            .unwrap_or(&[])
            .iter()
            .map(|&v| {
                use burn::tensor::ElementConversion;
                v.elem::<f32>()
            })
            .collect();

        Ok(wav)
    }
}
