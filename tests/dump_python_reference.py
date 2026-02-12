"""
Dump Python HiFiSampler intermediate data for Rust validation.

This script generates a test WAV, runs the Python mel spectrogram pipeline,
and saves all intermediate data to .npy files for comparison with Rust.

Usage:
  cd hifisampler-git
  python ../hifisampler-rs/tests/dump_python_reference.py
"""

import sys, os

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__)) + "/../../hifisampler-git"
)

import numpy as np
import torch
from pathlib import Path

# Generate a test sine wave at 440Hz, 1 second, 44100Hz
sr = 44100
duration = 1.0
t = np.arange(int(sr * duration)) / sr
test_audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)

out_dir = Path(__file__).parent / "reference_data"
out_dir.mkdir(exist_ok=True)

# Save test audio
np.save(out_dir / "test_audio.npy", test_audio)
print(f"Test audio: {test_audio.shape}, max={np.max(np.abs(test_audio)):.4f}")

# ── 1. Mel spectrogram ──
from util.wav2mel import PitchAdjustableMelSpectrogram
from util.audio import dynamic_range_compression_torch

mel_analyzer = PitchAdjustableMelSpectrogram(
    sample_rate=44100,
    n_fft=2048,
    win_length=2048,
    hop_length=128,  # origin_hop_size
    f_min=40.0,
    f_max=16000.0,
    n_mels=128,
)

wave_tensor = torch.from_numpy(test_audio).unsqueeze(0).float()
print(f"wave_tensor: {wave_tensor.shape}")

# key_shift=0, speed=1.0
with torch.inference_mode():
    mel_raw = mel_analyzer(wave_tensor, key_shift=0, speed=1.0)
print(
    f"mel_raw (before DRC): {mel_raw.shape}, min={mel_raw.min():.6f}, max={mel_raw.max():.6f}"
)

mel_drc = dynamic_range_compression_torch(mel_raw).squeeze(0)
print(
    f"mel_drc (after DRC): {mel_drc.shape}, min={mel_drc.min():.6f}, max={mel_drc.max():.6f}"
)

np.save(out_dir / "mel_raw.npy", mel_raw.squeeze(0).cpu().numpy())
np.save(out_dir / "mel_drc.npy", mel_drc.cpu().numpy())

# ── 2. Test with gender shift ──
with torch.inference_mode():
    mel_shifted = mel_analyzer(wave_tensor, key_shift=-2.0, speed=1.0)
mel_shifted_drc = dynamic_range_compression_torch(mel_shifted).squeeze(0)
np.save(out_dir / "mel_shifted_drc.npy", mel_shifted_drc.cpu().numpy())
print(f"mel_shifted_drc: {mel_shifted_drc.shape}")

# ── 3. DRC values test ──
test_vals = torch.tensor([0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
drc_vals = dynamic_range_compression_torch(test_vals)
print(f"\nDRC function test (C=1, clip_val=1e-9):")
print(f"  input:  {test_vals.tolist()}")
print(f"  output: {drc_vals.tolist()}")
print(f"  formula: torch.log(torch.clamp(x, min=1e-9) * 1)")
np.save(out_dir / "drc_test_input.npy", test_vals.numpy())
np.save(out_dir / "drc_test_output.npy", drc_vals.numpy())

# ── 4. Pitchbend parsing ──
from util.parse_utau import pitch_string_to_cents

test_pb = "AAAAAA#3#BABA"
pb_result = pitch_string_to_cents(test_pb)
print(f"\nPitchbend test: '{test_pb}' -> {pb_result[:20]}... (len={len(pb_result)})")
np.save(out_dir / "pitchbend_test.npy", pb_result)

# ── 5. Check the mel filterbank ──
from librosa.filters import mel as librosa_mel_fn

mel_fb = librosa_mel_fn(sr=44100, n_fft=2048, n_mels=128, fmin=40.0, fmax=16000.0)
print(f"\nMel filterbank (librosa default/Slaney): {mel_fb.shape}")
print(
    f"  sum per band: min={mel_fb.sum(axis=1).min():.6f}, max={mel_fb.sum(axis=1).max():.6f}"
)
np.save(out_dir / "mel_filterbank.npy", mel_fb)

# Check if librosa uses HTK or Slaney by default
mel_fb_htk = librosa_mel_fn(
    sr=44100, n_fft=2048, n_mels=128, fmin=40.0, fmax=16000.0, htk=True
)
print(
    f"Mel filterbank (HTK): differs from default? {not np.allclose(mel_fb, mel_fb_htk)}"
)
np.save(out_dir / "mel_filterbank_htk.npy", mel_fb_htk)

# ── 6. Full resample test ──
# Save test params that would be used for a resampler call
import json

test_params = {
    "sample_rate": 44100,
    "n_fft": 2048,
    "hop_size": 512,
    "origin_hop_size": 128,
    "win_size": 2048,
    "num_mels": 128,
    "fmin": 40.0,
    "fmax": 16000.0,
    "fill": 6,
    "drc_clip_val": 1e-9,
    "drc_C": 1,
}
with open(out_dir / "params.json", "w") as f:
    json.dump(test_params, f, indent=2)

print(f"\nAll reference data saved to {out_dir}")
print("Files:")
for f in sorted(out_dir.iterdir()):
    print(f"  {f.name}")
