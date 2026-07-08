"""Compare ORT vs Burn HN-SEP golden binary dumps."""
import struct
import sys
from pathlib import Path

ORT_DIR = Path("tests/benchmark_fixtures/hnsep_golden")
BURN_DIR = Path("tests/benchmark_fixtures/hnsep_burn")


def read_hnsep_bin(path: Path):
    data = path.read_bytes()
    assert data[:5] == b"HNSEP", f"bad magic in {path}"
    pos = 5
    audio_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    harmonic_len = struct.unpack_from("<I", data, pos)[0]; pos += 4
    n = audio_len + harmonic_len
    floats = struct.unpack_from(f"<{n}f", data, pos)
    return floats[:audio_len], floats[audio_len:]


def main() -> int:
    ort_files = sorted(ORT_DIR.glob("*.bin"))
    burn_files = sorted(BURN_DIR.glob("*.bin"))
    if not ort_files:
        print("no ORT golden"); return 1
    if not burn_files:
        print("no Burn output"); return 1

    # Compare first pair (only one fixture triggers hnsep)
    ort_audio, ort_harm = read_hnsep_bin(ort_files[0])
    burn_audio, burn_harm = read_hnsep_bin(burn_files[0])

    # Audio inputs should be identical
    n = min(len(ort_audio), len(burn_audio))
    audio_max = max(abs(ort_audio[i] - burn_audio[i]) for i in range(n))
    print(f"audio input max_abs={audio_max:.8f} (len {len(ort_audio)} vs {len(burn_audio)})")

    # Harmonic outputs
    n = min(len(ort_harm), len(burn_harm))
    diffs = [abs(ort_harm[i] - burn_harm[i]) for i in range(n)]
    max_abs = max(diffs)
    rms_diff = (sum(d * d for d in diffs) / n) ** 0.5
    rms_sig = (sum(v * v for v in ort_harm[:n]) / n) ** 0.5
    import math
    snr = 20 * math.log10(rms_sig / rms_diff) if rms_diff > 0 else float("inf")
    status = "PASS" if max_abs < 0.01 else ("WARN" if max_abs < 0.1 else "FAIL")
    print(f"{status} harmonic: max_abs={max_abs:.6f} rms_diff={rms_diff:.6f} snr={snr:.2f}dB len_o={len(ort_harm)} len_b={len(burn_harm)}")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
