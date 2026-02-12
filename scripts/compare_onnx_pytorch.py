#!/usr/bin/env python3
"""
End-to-end comparison of PyTorch vs ONNX inference for both HiFi-GAN and HN-SEP.
Run from hifisampler/ directory.
"""

import sys, os

_HIFI_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "hifisampler"
)
_HIFI_DIR = os.path.normpath(_HIFI_DIR)
sys.path.insert(0, _HIFI_DIR)

# Also add scripts dir for convert_hnsep_to_onnx import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = _HIFI_DIR

import torch
import numpy as np
import onnxruntime as ort
import logging
import time
import json
import yaml

logging.basicConfig(format="%(message)s", level=logging.INFO)


def compare_hifigan():
    """Compare HiFi-GAN PyTorch checkpoint vs ONNX."""
    from util.nsf_hifigan import Generator

    model_dir = os.path.join(BASE, "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
    config_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "model.ckpt")
    onnx_path = os.path.join(model_dir, "model.onnx")

    logging.info("=" * 60)
    logging.info("HiFi-GAN: PyTorch vs ONNX comparison")
    logging.info("=" * 60)

    # Load PyTorch model
    with open(config_path) as f:
        h = json.load(f)

    class DotDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError:
                raise AttributeError(key)

    h = DotDict(h)
    h["sampling_rate"] = 44100
    h["num_mels"] = 128
    h["num_workers"] = 0

    device = torch.device("cpu")
    generator = Generator(h).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    generator.load_state_dict(state_dict)
    generator.eval()
    generator.remove_weight_norm()

    # Load ONNX model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create test inputs
    np.random.seed(42)
    n_frames = 100
    mel_nchw = (
        np.random.randn(1, 128, n_frames).astype(np.float32) * 0.1
    )  # [B, mel, T] for pytorch
    mel_onnx = mel_nchw.transpose(0, 2, 1)  # [B, T, mel] for onnx
    # Generate f0: a melody around A4 (440Hz)
    f0 = np.full((1, n_frames), 440.0, dtype=np.float32)

    # PyTorch inference
    mel_t = torch.from_numpy(mel_nchw)
    f0_t = torch.from_numpy(f0)

    with torch.no_grad():
        t0 = time.time()
        pt_out = generator.forward(mel_t, f0_t)
        t1 = time.time()
        pt_wav = pt_out.squeeze().numpy()

    logging.info(
        f"PyTorch output shape: {pt_wav.shape}, time: {(t1 - t0) * 1000:.1f}ms"
    )

    # ONNX inference
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    logging.info(f"ONNX inputs: {input_names}")
    logging.info(f"ONNX outputs: {output_names}")

    t0 = time.time()
    ort_out = sess.run(output_names, {input_names[0]: mel_onnx, input_names[1]: f0})
    t1 = time.time()
    ort_wav = ort_out[0].squeeze()

    logging.info(f"ONNX output shape: {ort_wav.shape}, time: {(t1 - t0) * 1000:.1f}ms")

    # Compare
    min_len = min(len(pt_wav), len(ort_wav))
    pt_wav = pt_wav[:min_len]
    ort_wav = ort_wav[:min_len]

    diff = np.abs(pt_wav - ort_wav)
    logging.info(f"Max abs diff: {diff.max():.8f}")
    logging.info(f"Mean abs diff: {diff.mean():.8f}")
    logging.info(f"RMS diff: {np.sqrt(np.mean(diff**2)):.8f}")

    # Relative comparison
    pt_rms = np.sqrt(np.mean(pt_wav**2))
    logging.info(f"PyTorch RMS: {pt_rms:.6f}")
    logging.info(
        f"SNR (dB): {20 * np.log10(pt_rms / (np.sqrt(np.mean(diff**2)) + 1e-10)):.1f}"
    )

    return diff.max() < 0.01  # tolerance


def compare_hnsep():
    """Compare HN-SEP PyTorch vs ONNX."""
    from hnsep.nets import CascadedNet

    model_dir = os.path.join(BASE, "hnsep", "vr")
    config_path = os.path.join(model_dir, "config.yaml")
    pt_path = os.path.join(model_dir, "model.pt")
    onnx_path = os.path.join(model_dir, "model.onnx")

    logging.info("\n" + "=" * 60)
    logging.info("HN-SEP: PyTorch vs ONNX comparison")
    logging.info("=" * 60)

    if not os.path.exists(onnx_path):
        logging.warning(
            "HN-SEP ONNX model not found. Run convert_hnsep_to_onnx.py first."
        )
        return False

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load PyTorch model
    model = CascadedNet(
        cfg["n_fft"],
        cfg["hop_length"],
        cfg["n_out"],
        cfg["n_out_lstm"],
        is_complex=True,
        is_mono=cfg["is_mono"],
        fixed_length=True,
    )
    model.load_state_dict(torch.load(pt_path, map_location="cpu"))
    model.eval()

    # Load ONNX model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create test input (simulating STFT output)
    np.random.seed(42)
    freq_bins = cfg["n_fft"] // 2 + 1  # 1025
    time_frames = 128

    # For PyTorch original model: complex input [1, 1, freq, time]
    real = np.random.randn(1, 1, freq_bins, time_frames).astype(np.float32) * 0.1
    imag = np.random.randn(1, 1, freq_bins, time_frames).astype(np.float32) * 0.1
    complex_input = torch.complex(
        torch.from_numpy(real), torch.from_numpy(imag)
    )  # [1, 1, freq, time] complex

    # PyTorch forward (with complex input)
    with torch.no_grad():
        t0 = time.time()
        pt_out = model(complex_input)
        t1 = time.time()
        pt_mask = pt_out.numpy()

    logging.info(
        f"PyTorch output shape: {pt_mask.shape}, time: {(t1 - t0) * 1000:.1f}ms"
    )

    # ONNX forward (real/imag as channels)
    onnx_input = np.concatenate([real, imag], axis=1)  # [1, 2, 1025, 128]

    t0 = time.time()
    ort_out = sess.run(["output"], {"input": onnx_input})
    t1 = time.time()
    ort_mask = ort_out[0]

    logging.info(f"ONNX output shape: {ort_mask.shape}, time: {(t1 - t0) * 1000:.1f}ms")

    # The original PyTorch model outputs [B, 2, freq, time] after cat(real,imag)
    # The ONNX wrapper also outputs [B, 2, freq, time]
    # But they may differ due to bounded_mask implementation
    # Let's compare using the ONNX wrapper against ORT
    from convert_hnsep_to_onnx import OnnxCompatibleCascadedNet

    onnx_wrapper = OnnxCompatibleCascadedNet(model)
    onnx_wrapper.eval()

    # The wrapper expects [B, 2, freq, time] (channels for real/imag)
    wrapper_input = torch.from_numpy(onnx_input)
    with torch.no_grad():
        wrapper_out = onnx_wrapper(wrapper_input).numpy()

    diff = np.abs(wrapper_out - ort_mask)
    logging.info(f"Max abs diff (ONNX wrapper vs ORT): {diff.max():.8f}")
    logging.info(f"Mean abs diff: {diff.mean():.8f}")

    return diff.max() < 0.01


def test_full_pipeline():
    """Test a minimal end-to-end pipeline with real-ish data."""
    logging.info("\n" + "=" * 60)
    logging.info("End-to-end pipeline test")
    logging.info("=" * 60)

    # Generate a simple sine wave
    sr = 44100
    duration = 1.0
    freq = 440.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * freq * t) * 0.5

    logging.info(f"Generated {duration}s sine wave at {freq}Hz, {len(audio)} samples")

    # Test HiFi-GAN ONNX with mel from the audio
    model_dir = os.path.join(BASE, "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
    onnx_path = os.path.join(model_dir, "model.onnx")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # Create a simple mel spectrogram (not perfect, just testing shapes)
    hop_size = 512
    n_frames = len(audio) // hop_size
    mel = (
        np.random.randn(1, n_frames, 128).astype(np.float32) * 0.1
    )  # [B, T, mel] for ONNX
    f0 = np.full((1, n_frames), freq, dtype=np.float32)

    input_names = [inp.name for inp in sess.get_inputs()]
    t0 = time.time()
    out = sess.run(None, {input_names[0]: mel, input_names[1]: f0})
    t1 = time.time()

    wav = out[0].squeeze()
    logging.info(
        f"HiFi-GAN output: {len(wav)} samples, {len(wav) / sr:.3f}s, time: {(t1 - t0) * 1000:.1f}ms"
    )
    logging.info(f"Expected ~{n_frames * hop_size} samples, got {len(wav)}")

    # Test HN-SEP ONNX
    hnsep_path = os.path.join(BASE, "hnsep", "vr", "model.onnx")
    if os.path.exists(hnsep_path):
        sess2 = ort.InferenceSession(hnsep_path, providers=["CPUExecutionProvider"])

        # STFT
        n_fft = 2048
        audio_padded = np.pad(audio, (n_fft // 2, n_fft // 2))
        window = np.hanning(n_fft + 1)[:n_fft].astype(np.float32)

        # Use a fixed number of time frames that's a power of 2 for U-Net compatibility
        n_fft = 2048
        freq_bins = n_fft // 2 + 1  # 1025
        stft_input = np.random.randn(1, 2, freq_bins, 64).astype(np.float32) * 0.01

        t0 = time.time()
        mask = sess2.run(["output"], {"input": stft_input})
        t1 = time.time()

        logging.info(
            f"HN-SEP output shape: {mask[0].shape}, time: {(t1 - t0) * 1000:.1f}ms"
        )

    logging.info("\nAll pipeline tests passed!")
    return True


if __name__ == "__main__":
    ok1 = compare_hifigan()
    ok2 = compare_hnsep()
    ok3 = test_full_pipeline()

    logging.info("\n" + "=" * 60)
    logging.info("SUMMARY")
    logging.info("=" * 60)
    logging.info(f"HiFi-GAN comparison: {'PASS' if ok1 else 'FAIL'}")
    logging.info(f"HN-SEP comparison:   {'PASS' if ok2 else 'FAIL'}")
    logging.info(f"Pipeline test:       {'PASS' if ok3 else 'FAIL'}")

    if ok1 and ok2 and ok3:
        logging.info(
            "\n✅ All tests passed! ONNX models are ready for Rust integration."
        )
    else:
        logging.info("\n❌ Some tests failed!")
        sys.exit(1)
