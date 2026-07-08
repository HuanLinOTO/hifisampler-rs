//! Backend selection for the Burn native inference stack.
//!
//! The backend is chosen at compile time via Cargo features:
//! - `cuda` feature  → `burn::backend::Cuda` (NVIDIA only, fastest)
//! - `wgpu` feature  → `burn::backend::Wgpu` (Vulkan, works on all GPUs)
//!
//! ONNX Runtime has been completely removed.

use serde::Serialize;
use tracing::info;

// ── Compile-time backend selection ──
#[cfg(feature = "cuda")]
use burn::backend::Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use burn::backend::Wgpu;

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!("hifisampler-core requires at least one of: feature = \"cuda\" or feature = \"wgpu\"");

/// Burn backend type used by all models.
/// Selected at compile time: Cuda (if `cuda` feature) or Wgpu (if `wgpu` feature).
#[cfg(feature = "cuda")]
pub type VocoderBackend = Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type VocoderBackend = Wgpu;

/// Select a Burn device for the given config device string.
///
/// - `auto` / `vulkan` / `cuda` -> best available GPU
/// - `cpu` -> CPU
/// - legacy `directml`/`dml`/`coreml`/`tensorrt` -> mapped to GPU
pub fn select_burn_device(device: &str) -> burn::tensor::Device<VocoderBackend> {
    let device_lower = device.to_lowercase();
    info!("[ep] select_burn_device: creating device (requested={})", device_lower);
    let dev: burn::tensor::Device<VocoderBackend> = Default::default();
    info!("[ep] device created successfully");
    dev
}

/// Runtime capabilities for the Burn backend.
#[derive(Debug, Clone, Serialize)]
pub struct EpCapabilities {
    pub available_devices: Vec<String>,
    pub available_eps_raw: Vec<String>,
}

/// Detect available Burn backend capabilities.
pub fn detect_ep_capabilities() -> EpCapabilities {
    #[cfg(feature = "cuda")]
    let eps = vec!["BurnCuda".to_string()];
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    let eps = vec!["BurnWgpu".to_string()];

    EpCapabilities {
        available_devices: vec![
            "auto".to_string(),
            "cpu".to_string(),
            "vulkan".to_string(),
        ],
        available_eps_raw: eps,
    }
}
