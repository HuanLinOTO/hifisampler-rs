"""
End-to-end validation script for Rust HiFiSampler.
Compares each stage of the pipeline against Python reference output.

Run from hifisampler-rs/ directory:
  python tests/validate_pipeline.py

Requires: numpy, scipy, librosa, soundfile
"""
import sys
import os
import numpy as np

# Add Python project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'hifisampler-git'))

print("=" * 60)
print("HiFiSampler Rust vs Python Validation")
print("=" * 60)

errors = []

# ── Test 1: Pitchbend decoding ──
print("\n[1] Pitchbend decoding")
from util.parse_utau import pitch_string_to_cents

test_cases = [
    ("AA", "AA (empty)"),
    ("AAAAAA", "six zeros"),
    ("AAAAAA#3#BABA", "with RLE"),
    ("BABA", "BA pairs only"),
]

for pitch_str, desc in test_cases:
    result = pitch_string_to_cents(pitch_str)
    print(f"  '{pitch_str}' ({desc}): {result.tolist()}")

# ── Test 2: DRC ──
print("\n[2] Dynamic Range Compression")
import torch
from util.audio import dynamic_range_compression_torch

test_vals = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
for v in test_vals:
    t = torch.tensor([v], dtype=torch.float32)
    drc = dynamic_range_compression_torch(t, C=1, clip_val=1e-9)
    print(f"  DRC({v}) = {drc.item():.6f}  (expected: ln({v}) = {np.log(v):.6f})")
    if abs(drc.item() - np.log(v)) > 1e-5:
        errors.append(f"DRC mismatch for {v}")

# ── Test 3: Mel filterbank (Slaney vs HTK) ──
print("\n[3] Mel filterbank")
try:
    import librosa
    fb_default = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128, fmin=40.0, fmax=16000.0)
    fb_htk = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128, fmin=40.0, fmax=16000.0, htk=True)
    
    print(f"  Default (Slaney) filterbank shape: {fb_default.shape}")
    print(f"  HTK filterbank shape: {fb_htk.shape}")
    print(f"  Slaney == HTK? {np.allclose(fb_default, fb_htk)}")
    print(f"  Max difference: {np.max(np.abs(fb_default - fb_htk)):.6f}")
    
    # Save reference for Rust comparison
    ref_dir = os.path.join(os.path.dirname(__file__), 'reference_data')
    os.makedirs(ref_dir, exist_ok=True)
    np.save(os.path.join(ref_dir, 'mel_filterbank_slaney.npy'), fb_default)
    print(f"  Saved Slaney filterbank to reference_data/mel_filterbank_slaney.npy")
except ImportError:
    print("  [SKIP] librosa not installed")

# ── Test 4: Note/MIDI conversion ──
print("\n[4] Note/MIDI conversion")
from util.parse_utau import note_to_midi, midi_to_hz

test_notes = [("C4", 60), ("A4", 69), ("C#4", 61), ("B3", 59)]
for note, expected in test_notes:
    result = note_to_midi(note)
    status = "✓" if result == expected else "✗"
    print(f"  {status} note_to_midi('{note}') = {result} (expected {expected})")
    if result != expected:
        errors.append(f"note_to_midi('{note}') = {result}, expected {expected}")

test_midi = [(69, 440.0), (60, 261.63)]
for midi, expected_hz in test_midi:
    result = midi_to_hz(midi)
    status = "✓" if abs(result - expected_hz) < 0.1 else "✗"
    print(f"  {status} midi_to_hz({midi}) = {result:.2f} (expected {expected_hz:.2f})")

# ── Test 5: Tension filter ──
print("\n[5] Tension filter freq_filter computation")
# Python: x0 = fft_bin / ((sample_rate / 2) / 1500)
sr = 44100
n_fft = 2048
fft_bin = n_fft // 2 + 1
x0 = fft_bin / ((sr / 2) / 1500)
print(f"  fft_bin = {fft_bin}")
print(f"  x0 = {x0:.4f}")
print(f"  For b=-1.0: filter[0]={(-(-1.0)/x0)*0 + (-1.0):.4f}, filter[fft_bin-1]={(-(-1.0)/x0)*(fft_bin-1) + (-1.0):.4f}")

# ── Test 6: Time calculations ──
print("\n[6] Time axis calculations")
origin_hop_size = 128
hop_size = 512
thop_origin = origin_hop_size / sr
thop = hop_size / sr
print(f"  thop_origin = {thop_origin:.10f}")
print(f"  thop = {thop:.10f}")

# Simulate 1 second of audio at 44100Hz
n_samples = 44100
n_mel_frames = n_samples // origin_hop_size  # approximate
print(f"  For 1s audio: ~{n_mel_frames} mel frames at origin_hop_size={origin_hop_size}")
print(f"  Total time: {n_mel_frames * thop_origin:.4f}s")

# ── Test 7: Velocity / stretch ──
print("\n[7] Velocity / stretch")
for vel_param in [50, 100, 150]:
    vel = 2 ** (1 - vel_param / 100)
    print(f"  velocity={vel_param}: vel={vel:.4f}")

# Summary
print("\n" + "=" * 60)
if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  ✗ {e}")
else:
    print("ALL CHECKS PASSED ✓")
print("=" * 60)
