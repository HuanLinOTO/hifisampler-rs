//! HiFi-GAN vocoder ONNX inference wrapper.

use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tracing::info;

/// HiFi-GAN vocoder using ONNX Runtime.
pub struct Vocoder {
    session: Session,
}

impl Vocoder {
    /// Load a HiFi-GAN ONNX model.
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let path = model_path.as_ref();
        info!("Loading vocoder from: {}", path.display());

        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(path)?;

        info!("Vocoder loaded successfully");
        Ok(Self { session })
    }

    /// Synthesize waveform from mel spectrogram and F0.
    ///
    /// mel: [num_mels, n_frames] (will be transposed to [1, n_frames, num_mels] for ONNX)
    /// f0: [n_frames]
    ///
    /// Returns: audio waveform samples
    pub fn synthesize(&mut self, mel: &Array2<f32>, f0: &[f32]) -> Result<Vec<f32>> {
        let n_frames = mel.ncols();
        let num_mels = mel.nrows();

        // ONNX model expects [batch, n_frames, num_mels]
        // Build flat data in row-major order
        let mut mel_data = vec![0.0f32; n_frames * num_mels];
        for i in 0..n_frames {
            for j in 0..num_mels {
                mel_data[i * num_mels + j] = mel[[j, i]];
            }
        }
        let mel_tensor = Tensor::from_array(([1usize, n_frames, num_mels], mel_data))?;

        // F0: [batch, n_frames]
        let f0_data: Vec<f32> = f0[..n_frames].to_vec();
        let f0_tensor = Tensor::from_array(([1usize, n_frames], f0_data))?;

        let outputs = self.session.run(
            ort::inputs![
                "mel" => mel_tensor,
                "f0" => f0_tensor,
            ],
        )?;

        // Extract waveform
        let (_, waveform_data) = outputs["waveform"]
            .try_extract_tensor::<f32>()?;
        let wav: Vec<f32> = waveform_data.to_vec();

        Ok(wav)
    }
}
