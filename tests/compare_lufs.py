"""
Compare LUFS measurement between pyloudnorm (Python) and Rust implementation.
Generates test signals and measures integrated loudness with both.

Run from hifisampler-rs/:
  python tests/compare_lufs.py
"""
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "hifisampler"))

try:
    import pyloudnorm as pyln
except ImportError:
    print("pyloudnorm not available, install with: pip install pyloudnorm")
    sys.exit(1)

SR = 44100

def generate_test_signals():
    """Generate various test signals."""
    signals = {}
    t = np.linspace(0, 1.0, SR, endpoint=False)

    # 1. Pure 440Hz sine at different amplitudes
    for amp_db in [-6, -12, -20, -30, -40]:
        amp = 10 ** (amp_db / 20.0)
        sig = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        signals[f"sine_440Hz_{amp_db}dB"] = sig

    # 2. Voiced-like signal with breath tail
    # Simulate a note with 0.7s of voice + 0.3s of breath
    voice = 0.3 * np.sin(2 * np.pi * 440 * t[:int(SR*0.7)])
    breath_t = t[:int(SR*0.3)]
    breath = 0.01 * np.random.randn(int(SR*0.3)).astype(np.float64)  # quiet breath
    breath *= np.exp(-breath_t * 5)  # decaying
    full = np.concatenate([voice, breath]).astype(np.float32)
    signals["voice_with_breath_tail"] = full

    # 3. White noise
    noise = (0.1 * np.random.randn(SR)).astype(np.float32)
    signals["white_noise_-20dB"] = noise

    # 4. Silence with brief burst (worst case for normalization)
    silent = np.zeros(SR, dtype=np.float32)
    silent[SR//4:SR//4+2000] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(2000) / SR)
    signals["silence_with_burst"] = silent

    # 5. Longer signal (2 seconds)
    t2 = np.linspace(0, 2.0, SR*2, endpoint=False)
    sig2 = (0.1 * np.sin(2 * np.pi * 440 * t2)).astype(np.float32)
    signals["sine_2sec_-20dB"] = sig2

    return signals

def measure_lufs_python(audio, sr=SR, block_size=0.400):
    """Measure LUFS using pyloudnorm."""
    meter = pyln.Meter(sr, block_size=block_size)
    return meter.integrated_loudness(audio)

def simulate_rust_lufs(audio, sr=SR, block_size=0.400):
    """
    Simulate the Rust LUFS measurement in Python for comparison.
    This reimplements the Rust measure_lufs() logic.
    """
    from scipy.signal import lfilter

    # K-weighting coefficients for 44100 Hz (same as Rust)
    # Pre-shelf
    shelf_b = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    shelf_a = [1.0, -1.69065929318241, 0.73248077421585]
    # High-pass
    hp_b = [1.0, -2.0, 1.0]
    hp_a = [1.0, -1.99004745483398, 0.99007225036621]

    audio_f64 = audio.astype(np.float64)
    weighted = lfilter(shelf_b, shelf_a, audio_f64)
    weighted = lfilter(hp_b, hp_a, weighted)

    block_samples = int(round(sr * block_size))
    hop_samples = block_samples // 4

    if len(weighted) < block_samples:
        mean_sq = np.mean(weighted ** 2)
        if mean_sq < 1e-20:
            return float('-inf')
        return -0.691 + 10.0 * np.log10(mean_sq)

    # Block-based
    block_powers = []
    pos = 0
    while pos + block_samples <= len(weighted):
        block = weighted[pos:pos+block_samples]
        mean_sq = np.mean(block ** 2)
        block_powers.append(mean_sq)
        pos += hop_samples

    # Absolute gate: -70 LUFS
    abs_gate = 10 ** ((-70.0 + 0.691) / 10.0)
    gated_abs = [p for p in block_powers if p > abs_gate]
    if not gated_abs:
        return float('-inf')

    # Relative gate
    mean_abs = np.mean(gated_abs)
    rel_gate = mean_abs * 10 ** (-1.0)
    gated_rel = [p for p in gated_abs if p > rel_gate]
    if not gated_rel:
        return float('-inf')

    mean_rel = np.mean(gated_rel)
    return -0.691 + 10.0 * np.log10(mean_rel)

def main():
    signals = generate_test_signals()

    print(f"{'Signal':<30} {'pyloudnorm':>12} {'Rust-sim':>12} {'Delta':>8}")
    print("-" * 65)

    for name, sig in signals.items():
        lufs_py = measure_lufs_python(sig)
        lufs_rust = simulate_rust_lufs(sig)
        delta = lufs_rust - lufs_py if not (np.isinf(lufs_py) or np.isinf(lufs_rust)) else float('nan')
        print(f"{name:<30} {lufs_py:>12.3f} {lufs_rust:>12.3f} {delta:>8.3f}")

    # Now simulate the full loudness_normalize pipeline
    print("\n" + "=" * 65)
    print("Loudness Normalization Simulation (target = -16 LUFS, strength = 100)")
    print("=" * 65)

    sig = signals["voice_with_breath_tail"]
    lufs_py = measure_lufs_python(sig)
    lufs_rust = simulate_rust_lufs(sig)

    gain_py = 10 ** ((-16.0 - lufs_py) / 20.0)
    gain_rust = 10 ** ((-16.0 - lufs_rust) / 20.0)

    peak_before = np.max(np.abs(sig))
    peak_after_py = peak_before * gain_py
    peak_after_rust = peak_before * gain_rust

    print(f"  Input LUFS (pyloudnorm):  {lufs_py:.3f}")
    print(f"  Input LUFS (Rust-sim):    {lufs_rust:.3f}")
    print(f"  Gain (pyloudnorm):        {gain_py:.4f} ({20*np.log10(gain_py):.1f} dB)")
    print(f"  Gain (Rust-sim):          {gain_rust:.4f} ({20*np.log10(gain_rust):.1f} dB)")
    print(f"  Peak before:              {peak_before:.4f}")
    print(f"  Peak after (pyloudnorm):  {peak_after_py:.4f}")
    print(f"  Peak after (Rust-sim):    {peak_after_rust:.4f}")

    # Check breath tail amplification
    voice_end = int(SR * 0.7)
    breath_rms_before = np.sqrt(np.mean(sig[voice_end:] ** 2))
    breath_rms_after_py = breath_rms_before * gain_py
    breath_rms_after_rust = breath_rms_before * gain_rust
    print(f"\n  Breath tail RMS before:   {breath_rms_before:.6f} ({20*np.log10(max(breath_rms_before, 1e-10)):.1f} dB)")
    print(f"  Breath tail RMS after Py: {breath_rms_after_py:.6f} ({20*np.log10(max(breath_rms_after_py, 1e-10)):.1f} dB)")
    print(f"  Breath tail RMS after Rs: {breath_rms_after_rust:.6f} ({20*np.log10(max(breath_rms_after_rust, 1e-10)):.1f} dB)")

if __name__ == "__main__":
    main()
