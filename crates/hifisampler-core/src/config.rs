use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::ep::detect_ep_capabilities;

/// Main configuration for HiFiSampler.
/// Mirrors the Python config.py dataclass structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_win_size")]
    pub win_size: usize,
    #[serde(default = "default_hop_size")]
    pub hop_size: usize,
    #[serde(default = "default_hop_size_interp")]
    pub hop_size_interp: usize,
    #[serde(default = "default_num_mels")]
    pub num_mels: usize,
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    #[serde(default = "default_fmin")]
    pub fmin: f64,
    #[serde(default = "default_fmax")]
    pub fmax: f64,
    #[serde(default = "default_pad_frames")]
    pub pad_frames: usize,

    #[serde(default)]
    pub vocoder: VocoderConfig,
    #[serde(default)]
    pub hnsep: HnsepConfig,
    #[serde(default)]
    pub processing: ProcessingConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
    #[serde(default)]
    pub server: ServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocoderConfig {
    #[serde(default = "default_vocoder_model")]
    pub model: PathBuf,
    #[serde(default = "default_vocoder_type")]
    pub model_type: String,
    #[serde(default = "default_vocoder_model_fp16")]
    pub model_fp16: PathBuf,
    #[serde(default = "default_vocoder_use_fp16")]
    pub use_fp16: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnsepConfig {
    #[serde(default = "default_hnsep_model")]
    pub model: PathBuf,
    #[serde(default = "default_hnsep_model_fp16")]
    pub model_fp16: PathBuf,
    #[serde(default = "default_hnsep_use_fp16")]
    pub use_fp16: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    #[serde(default)]
    pub wave_norm: bool,
    #[serde(default = "default_true")]
    pub wave_norm_clip_silence: bool,
    #[serde(default)]
    pub loop_mode: bool,
    #[serde(default = "default_peak_limit")]
    pub peak_limit: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,
    /// Execution provider for ONNX Runtime inference.
    ///
    /// Supported values:
    /// - `"auto"` — try all platform-appropriate EPs (TensorRT → CUDA → DirectML/CoreML/ROCm → CPU)
    /// - `"cpu"` — CPU only
    /// - `"cuda"` — NVIDIA CUDA
    /// - `"tensorrt"` — NVIDIA TensorRT (with CUDA fallback)
    /// - `"directml"` / `"dml"` — Microsoft DirectML (Windows)
    /// - `"coreml"` — Apple CoreML (macOS)
    /// - `"rocm"` — AMD ROCm (Linux)
    #[serde(default = "default_device")]
    pub device: String,

    /// Device ID for execution providers that support selecting a specific device.
    ///
    /// Currently used by DirectML (`directml` / `dml`).
    #[serde(default = "default_device_id")]
    pub device_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

// Default value functions
fn default_sample_rate() -> u32 {
    44100
}
fn default_win_size() -> usize {
    2048
}
fn default_hop_size() -> usize {
    512
}
fn default_hop_size_interp() -> usize {
    128
}
fn default_num_mels() -> usize {
    128
}
fn default_n_fft() -> usize {
    2048
}
fn default_fmin() -> f64 {
    40.0
}
fn default_fmax() -> f64 {
    16000.0
}
fn default_pad_frames() -> usize {
    6
}
fn default_vocoder_model() -> PathBuf {
    PathBuf::from("models/vocoder/model.onnx")
}
fn default_vocoder_type() -> String {
    "onnx".to_string()
}
fn default_vocoder_model_fp16() -> PathBuf {
    PathBuf::from("models/vocoder/model_fp16.onnx")
}
fn default_vocoder_use_fp16() -> bool {
    false
}
fn default_hnsep_model() -> PathBuf {
    PathBuf::from("models/hnsep/model_fp32_slim.onnx")
}
fn default_hnsep_model_fp16() -> PathBuf {
    PathBuf::from("models/hnsep/model_fp16.onnx")
}
fn default_hnsep_use_fp16() -> bool {
    false
}
fn should_default_fp16_for_device(device: &str) -> bool {
    let device_lower = device.trim().to_ascii_lowercase();
    if device_lower == "directml" || device_lower == "dml" {
        return true;
    }
    if device_lower != "auto" {
        return false;
    }
    let caps = detect_ep_capabilities();
    let has_dml = caps
        .available_devices
        .iter()
        .any(|d| d == "directml" || d == "dml");
    let has_cuda = caps
        .available_devices
        .iter()
        .any(|d| d == "cuda" || d == "tensorrt");
    has_dml && !has_cuda
}
fn default_peak_limit() -> f32 {
    1.0
}
fn default_num_threads() -> usize {
    2
}
fn default_device() -> String {
    "auto".to_string()
}
fn default_device_id() -> i32 {
    0
}
fn default_true() -> bool {
    true
}
fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    8572
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: default_sample_rate(),
            win_size: default_win_size(),
            hop_size: default_hop_size(),
            hop_size_interp: default_hop_size_interp(),
            num_mels: default_num_mels(),
            n_fft: default_n_fft(),
            fmin: default_fmin(),
            fmax: default_fmax(),
            pad_frames: default_pad_frames(),
            vocoder: VocoderConfig::default(),
            hnsep: HnsepConfig::default(),
            processing: ProcessingConfig::default(),
            performance: PerformanceConfig::default(),
            server: ServerConfig::default(),
        }
    }
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self {
            model: default_vocoder_model(),
            model_type: default_vocoder_type(),
            model_fp16: default_vocoder_model_fp16(),
            use_fp16: default_vocoder_use_fp16(),
        }
    }
}

impl Default for HnsepConfig {
    fn default() -> Self {
        Self {
            model: default_hnsep_model(),
            model_fp16: default_hnsep_model_fp16(),
            use_fp16: default_hnsep_use_fp16(),
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            wave_norm: false,
            wave_norm_clip_silence: true,
            loop_mode: false,
            peak_limit: default_peak_limit(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            num_threads: default_num_threads(),
            device: default_device(),
            device_id: default_device_id(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
        }
    }
}

impl Config {
    pub fn resolved_vocoder_model_path(&self) -> PathBuf {
        let device = self.performance.device.trim();
        if device.eq_ignore_ascii_case("cpu") {
            self.vocoder.model.clone()
        } else if self.vocoder.use_fp16 {
            self.vocoder.model_fp16.clone()
        } else if should_default_fp16_for_device(device) && self.vocoder.model_fp16.exists() {
            self.vocoder.model_fp16.clone()
        } else {
            self.vocoder.model.clone()
        }
    }

    pub fn resolved_hnsep_model_path(&self) -> PathBuf {
        let device = self.performance.device.trim();
        if device.eq_ignore_ascii_case("cpu") {
            self.hnsep.model.clone()
        } else if self.hnsep.use_fp16 && self.hnsep.model_fp16.exists() {
            self.hnsep.model_fp16.clone()
        } else if should_default_fp16_for_device(device) && self.hnsep.model_fp16.exists() {
            self.hnsep.model_fp16.clone()
        } else {
            self.hnsep.model.clone()
        }
    }

    /// Load config from a YAML file, falling back to defaults for missing fields.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;
        let config: Config = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load config, merging user config with defaults.
    /// If the file doesn't exist, return defaults.
    pub fn load_or_default(path: impl AsRef<Path>) -> Self {
        match Self::load(path) {
            Ok(config) => config,
            Err(_) => Self::default(),
        }
    }

    /// Save config to a YAML file.
    pub fn save(&self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let content = serde_yaml::to_string(self)?;
        std::fs::write(path.as_ref(), content)?;
        Ok(())
    }
}
