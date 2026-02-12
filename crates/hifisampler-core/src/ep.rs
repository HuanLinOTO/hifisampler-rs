//! Execution provider configuration for ONNX Runtime sessions.
//!
//! Supports: CUDA, TensorRT, DirectML, CoreML, and CPU.
//! Uses `load-dynamic` feature so that all EPs are available at runtime if the
//! user provides an appropriate `onnxruntime` shared library.
//!
//! Note: ROCm EP was removed from ONNX Runtime 1.23+. AMD users should use
//! MIGraphX EP or the DirectML EP (which also supports AMD GPUs on Windows).

use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    ExecutionProviderDispatch, TensorRTExecutionProvider,
};
use tracing::{info, warn};

/// Build the list of execution providers for the given device string.
///
/// Supported values for `device`:
/// - `"auto"` — register all platform-appropriate EPs in optimal priority order;
///   ort will silently skip unavailable ones and fall back to CPU.
/// - `"cpu"` — no GPU EP, pure CPU inference.
/// - `"cuda"` — NVIDIA CUDA.
/// - `"tensorrt"` — NVIDIA TensorRT (falls back to CUDA).
/// - `"directml"` / `"dml"` — Microsoft DirectML (Windows).
/// - `"coreml"` — Apple CoreML (macOS / iOS).
/// - Any other value is treated as `"cpu"` with a warning.
pub fn build_execution_providers(device: &str) -> Vec<ExecutionProviderDispatch> {
    let device_lower = device.to_lowercase();
    let device_str = device_lower.as_str();

    match device_str {
        "auto" => {
            info!("Device=auto: registering all available execution providers");
            auto_providers()
        }
        "cpu" => {
            info!("Device=cpu: using CPU only");
            vec![]
        }
        "cuda" => {
            info!("Device=cuda: registering CUDA execution provider");
            vec![CUDAExecutionProvider::default().build()]
        }
        "tensorrt" | "trt" => {
            info!("Device=tensorrt: registering TensorRT + CUDA execution providers");
            vec![
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
            ]
        }
        "directml" | "dml" => {
            info!("Device=directml: registering DirectML execution provider");
            vec![DirectMLExecutionProvider::default().build()]
        }
        "coreml" => {
            info!("Device=coreml: registering CoreML execution provider");
            vec![CoreMLExecutionProvider::default().build()]
        }
        other => {
            warn!(
                "Unknown device '{}', falling back to CPU. \
                 Supported: auto, cpu, cuda, tensorrt, directml, coreml",
                other
            );
            vec![]
        }
    }
}

/// Build the optimal EP list for the current platform.
///
/// The caller should expect ERROR-level logs from `ort::ep` when an EP cannot
/// be registered (e.g. missing CUDA/cuDNN/TensorRT SDK).  These are harmless —
/// ONNX Runtime will automatically fall back to the next EP in the list and
/// ultimately to CPU if nothing else is available.
fn auto_providers() -> Vec<ExecutionProviderDispatch> {
    info!(
        "Auto mode will try GPU providers in priority order. \
         Errors from ort::ep about missing DLLs (e.g. nvinfer, cudnn) are normal \
         if the corresponding SDK is not installed — ONNX Runtime will fall back to CPU."
    );
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    // NVIDIA: TensorRT > CUDA  (Windows + Linux)
    if cfg!(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(
            target_os = "linux",
            any(target_arch = "x86_64", target_arch = "aarch64")
        )
    )) {
        eps.push(TensorRTExecutionProvider::default().build());
        eps.push(CUDAExecutionProvider::default().build());
    }

    // DirectML (Windows only)
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        eps.push(DirectMLExecutionProvider::default().build());
    }

    // CoreML (macOS / iOS)
    if cfg!(target_os = "macos") {
        eps.push(CoreMLExecutionProvider::default().build());
    }

    eps
}
