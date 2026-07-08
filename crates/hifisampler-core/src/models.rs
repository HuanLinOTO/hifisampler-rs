//! Model loader - manages loading and holding model instances.

use crate::config::Config;
use crate::hnsep::HnsepModel;
use crate::mel::MelAnalyzer;
use crate::vocoder::Vocoder;
use anyhow::Result;
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::info;

/// Holds all loaded model instances.
pub struct Models {
    pub vocoder: Arc<Mutex<Vocoder>>,
    pub hnsep: Option<Arc<Mutex<HnsepModel>>>,
    pub mel_analyzer: Arc<MelAnalyzer>,
}

impl Models {
    /// Load all models based on configuration.
    pub fn load(config: &Config) -> Result<Self> {
        info!("Loading models...");

        let device = &config.performance.device;
        let device_id = config.performance.device_id;
        let num_threads = config.performance.num_threads;

        // Load vocoder
        let vocoder = Arc::new(Mutex::new(Vocoder::load(
            &config.vocoder.model,
            &config.vocoder.model_type,
            device,
            device_id,
            num_threads,
        )?));

        // Load HN-SEP (optional)
        let hnsep = if config.hnsep.model.exists() {
            Some(Arc::new(Mutex::new(HnsepModel::load(
                &config.hnsep.model,
                config.n_fft,
                config.hop_size,
                device,
                device_id,
                num_threads,
            )?)))
        } else {
            info!("HN-SEP model not found, harmonic separation disabled");
            None
        };

        // Create mel analyzer
        let mel_analyzer = Arc::new(MelAnalyzer::new(
            config.sample_rate,
            config.n_fft,
            config.hop_size,
            config.win_size,
            config.num_mels,
            config.fmin,
            config.fmax,
        ));

        info!("All models loaded successfully, starting warmup...");

        let models = Self {
            vocoder,
            hnsep,
            mel_analyzer,
        };

        // Warmup: run dummy forward to trigger kernel compilation + autotune
        // before the first real inference. This eliminates the ~1.8s cold-start
        // overhead on the first fixture.
        models.warmup(config);
        info!("Warmup complete");

        Ok(models)
    }

    /// Run dummy forward passes to pre-compile GPU kernels.
    ///
    /// Without warmup, the first real inference pays ~1.8s for kernel
    /// compilation (CUDA JIT) + LSTM weight cache initialization. Warmup
    /// moves this cost to model-load time, so the first user request is
    /// as fast as a hot inference.
    fn warmup(&self, config: &Config) {
        // Vocoder warmup: mel=zeros, f0=440Hz. Use enough frames to cover
        // the largest fixture shape (~176 frames for 2s audio at 44.1kHz/512hop).
        // Too-small warmup shapes cause kernel recompilation on first real request.
        let num_mels = config.num_mels;
        let warmup_frames = 176;
        let dummy_mel = ndarray::Array2::zeros((num_mels, warmup_frames));
        let dummy_f0 = vec![440.0_f32; warmup_frames];
        if let Some(mut vocoder) = self.vocoder.try_lock() {
            let _ = vocoder.synthesize(&dummy_mel, &dummy_f0);
            info!("[warmup] vocoder forward done");
        }

        // HN-SEP warmup: 2s of silence to match the largest fixture shape.
        // The hnsep forward pads to multiples of 16, so 2s gives ~176 frames
        // which covers all benchmark fixtures.
        if let Some(hnsep) = &self.hnsep {
            let warmup_samples = config.sample_rate as usize * 2; // 2s
            let dummy_audio = vec![0.0_f32; warmup_samples];
            // Temporarily disable HIFISAMPLER_DUMP_HNSEP during warmup so the
            // warmup forward doesn't pollute the golden comparison dump dir.
            let saved_dump = std::env::var("HIFISAMPLER_DUMP_HNSEP").ok();
            if saved_dump.is_some() {
                unsafe { std::env::remove_var("HIFISAMPLER_DUMP_HNSEP"); }
            }
            if let Some(mut hnsep) = hnsep.try_lock() {
                let _ = hnsep.predict_from_audio(&dummy_audio, config.sample_rate);
                info!("[warmup] hnsep forward done");
            }
            if let Some(v) = saved_dump {
                unsafe { std::env::set_var("HIFISAMPLER_DUMP_HNSEP", v); }
            }
        }
    }
}
