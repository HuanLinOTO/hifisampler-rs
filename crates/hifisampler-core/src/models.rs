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

        info!("All models loaded successfully");
        Ok(Self {
            vocoder,
            hnsep,
            mel_analyzer,
        })
    }
}
