//! Execution provider configuration for ONNX Runtime sessions.
//!
//! Supports: CUDA, TensorRT, DirectML, CoreML, ROCm, OpenVINO, CANN, and CPU.
//! Uses `load-dynamic` feature so that all EPs are available at runtime if the
//! user provides an appropriate `onnxruntime` shared library.

use ort::execution_providers::{
    CUDAExecutionProvider, CoreMLExecutionProvider, DirectMLExecutionProvider,
    ExecutionProviderDispatch, ROCmExecutionProvider, TensorRTExecutionProvider,
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
/// - `"rocm"` — AMD ROCm.
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
        "rocm" => {
            info!("Device=rocm: registering ROCm execution provider");
            vec![ROCmExecutionProvider::default().build()]
        }
        other => {
            warn!(
                "Unknown device '{}', falling back to CPU. \
                 Supported: auto, cpu, cuda, tensorrt, directml, coreml, rocm",
                other
            );
            vec![]
        }
    }
}

/// Build the optimal EP list for the current platform.
fn auto_providers() -> Vec<ExecutionProviderDispatch> {
    let mut eps: Vec<ExecutionProviderDispatch> = Vec::new();

    // NVIDIA: TensorRT > CUDA  (Windows + Linux)
    if cfg!(any(
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))
    )) {
        eps.push(TensorRTExecutionProvider::default().build());
        eps.push(CUDAExecutionProvider::default().build());
    }

    // DirectML (Windows only)
    if cfg!(all(target_os = "windows", target_arch = "x86_64")) {
        eps.push(DirectMLExecutionProvider::default().build());
    }

    // ROCm (Linux only)
    if cfg!(all(target_os = "linux", any(target_arch = "x86_64", target_arch = "aarch64"))) {
        eps.push(ROCmExecutionProvider::default().build());
    }

    // CoreML (macOS / iOS)
    if cfg!(target_os = "macos") {
        eps.push(CoreMLExecutionProvider::default().build());
    }

    eps
}
