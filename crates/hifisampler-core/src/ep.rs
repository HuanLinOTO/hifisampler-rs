//! Backend selection for the Burn native inference stack.
//!
//! The backend is chosen at compile time via Cargo features:
//! - `ndarray` feature → `burn::backend::NdArray` (pure CPU, no GPU deps)
//! - `wgpu` feature    → `burn::backend::Wgpu` (Vulkan/Metal/DX12, all GPUs)
//! - `cuda` feature    → `burn::backend::Cuda` (NVIDIA only, fastest)
//!
//! CI produces three distribution packages:
//! - **CPU**    — ndarray backend (lightweight, headless servers)
//! - **WebGPU** — wgpu backend (GPU acceleration, all platforms)
//! - **CUDA**   — cuda backend (NVIDIA only, fastest)
//!
//! ONNX Runtime has been completely removed.

use serde::Serialize;
use tracing::info;

// ── Compile-time backend selection ──
// Priority: cuda > wgpu > ndarray
#[cfg(feature = "cuda")]
use burn::backend::Cuda;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
use burn::backend::Wgpu;

#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
use burn::backend::NdArray;

#[cfg(not(any(feature = "cuda", feature = "wgpu", feature = "ndarray")))]
compile_error!(
    "hifisampler-core requires at least one of: \
     feature = \"ndarray\", feature = \"wgpu\", or feature = \"cuda\""
);

/// Burn backend type used by all models.
/// Selected at compile time: Cuda (if `cuda` feature), Wgpu (if `wgpu`), or NdArray (if `ndarray`).
#[cfg(feature = "cuda")]
pub type VocoderBackend = Cuda<burn::tensor::f16>;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type VocoderBackend = Wgpu;

#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
pub type VocoderBackend = NdArray;

/// Burn backend for HN-SEP model. Uses f16 for Tensor Core acceleration
/// and halved memory bandwidth on memory-bound ops (BN, activation, interpolate, cat).
/// Vocoder stays f32 (variable shapes + f16-sensitive output).
/// CPU LSTM path already converts via `elem::<f32>()`, so f16 input is transparent.
#[cfg(feature = "cuda")]
pub type HnsepBackend = Cuda<burn::tensor::f16>;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type HnsepBackend = Wgpu;

#[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
pub type HnsepBackend = NdArray;

/// Select a Burn device for the given config device string.
///
/// Behavior depends on the compiled backend:
/// - **NdArray**: device string is ignored — NdArray has a single no-op CPU device.
/// - **Wgpu**: `cpu` → CPU device, `auto`/`vulkan` → best available GPU.
/// - **Cuda**: any value → CUDA device 0.
pub fn select_burn_device(device: &str) -> burn::tensor::Device<VocoderBackend> {
    let device_lower = device.to_lowercase();
    info!(
        "[ep] select_burn_device: creating device (requested={})",
        device_lower
    );

    // ── Cuda backend ──
    #[cfg(feature = "cuda")]
    {
        let dev = match device_lower.as_str() {
            "cpu" => {
                info!("[ep] CUDA backend does not support CPU-only mode, using CUDA device 0");
                burn::backend::cuda::CudaDevice::new(0).into()
            }
            _ => {
                info!("[ep] selecting CUDA device 0");
                burn::backend::cuda::CudaDevice::new(0).into()
            }
        };
        info!("[ep] device created successfully");
        return dev;
    }

    // ── Wgpu backend ──
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        let dev = match device_lower.as_str() {
            "cpu" => {
                info!("[ep] selecting CPU device");
                burn::backend::wgpu::WgpuDevice::Cpu.into()
            }
            _ => {
                info!("[ep] selecting best available GPU device");
                burn::backend::wgpu::WgpuDevice::BestAvailable.into()
            }
        };
        info!("[ep] device created successfully");
        return dev;
    }

    // ── NdArray backend (pure CPU) ──
    #[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
    {
        // NdArray has a single no-op device; the requested device string is irrelevant.
        info!("[ep] using NdArray CPU device");
        return burn::tensor::Device::<NdArray>::default();
    }
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

    #[cfg(all(feature = "ndarray", not(any(feature = "cuda", feature = "wgpu"))))]
    let (eps, devices) = (
        vec!["BurnNdArray".to_string()],
        vec!["cpu".to_string()],
    );

    EpCapabilities {
        available_devices: devices,
        available_eps_raw: eps,
    }
}
