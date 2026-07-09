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

    let dev: burn::tensor::Device<VocoderBackend> = match device_lower.as_str() {
        "cpu" => {
            #[cfg(feature = "cuda")]
            {
                // CUDA backend doesn't have a CPU-only device; use device 0 as fallback.
                info!("[ep] CUDA backend does not support CPU-only mode, using CUDA device 0");
                burn::backend::cuda::CudaDevice::new(0).into()
            }
            #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
            {
                info!("[ep] selecting CPU device");
                burn::backend::wgpu::WgpuDevice::Cpu.into()
            }
        }
        _ => {
            // auto, vulkan, cuda, directml, dml, coreml, tensorrt → best available
            info!("[ep] selecting best available GPU device");
            #[cfg(feature = "cuda")]
            {
                burn::backend::cuda::CudaDevice::new(0).into()
            }
            #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
            {
                burn::backend::wgpu::WgpuDevice::BestAvailable.into()
            }
        }
    };

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
    let (eps, devices) = (
        vec!["BurnCuda".to_string()],
        vec!["auto".to_string(), "cuda".to_string()],
    );
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    let (eps, devices) = (
        vec!["BurnWgpu".to_string()],
        vec!["auto".to_string(), "cpu".to_string(), "vulkan".to_string()],
    );

    EpCapabilities {
        available_devices: devices,
        available_eps_raw: eps,
    }
}
