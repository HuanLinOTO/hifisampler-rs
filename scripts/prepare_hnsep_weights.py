"""Phase 5: 导出 CascadedNet state_dict 供 burn-store 加载。

加载 PyTorch checkpoint -> CascadedNet -> 导出原始 state_dict（保留 BatchNorm running stats）。
不做 weight_norm 融合（CascadedNet 无 weight_norm）。

输出: crates/burn-spike/models/hnsep_fused.pt  (key: "model")
"""
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
PARENT = WORKSPACE.parent

HIFISAMPLER_PY = PARENT / "hifisampler"
if str(HIFISAMPLER_PY) not in sys.path:
    sys.path.insert(0, str(HIFISAMPLER_PY))

from hnsep.nets import CascadedNet  # noqa: E402
import yaml  # noqa: E402

CKPT_DIR = PARENT / "新建文件夹" / "vr"
CONFIG_PATH = CKPT_DIR / "config.yaml"
CKPT_PATH = CKPT_DIR / "model.pt"
OUT_PATH = WORKSPACE / "crates" / "burn-spike" / "models" / "hnsep_fused.pt"


def remap_key(k: str) -> str:
    """Remap PyTorch state_dict key to Burn module field path.

    Burn struct field names (what burn-store expects):
      Conv2dBnActivInner: conv.conv0 (Conv2d), conv.conv1 (BatchNorm)
      AsppBranch1: conv1.inner1 (Conv2dBnActiv)  [PyTorch conv1.0=Mean, conv1.1=Conv2dBnActiv]
      LstmModule: dense0 (Linear), dense1 (BatchNorm)  [PyTorch dense.0=Linear, dense.1=BN1d]
      BatchNorm: gamma/beta/running_mean/running_var  [PyTorch weight/bias/...]
      ManualBiLstm: weight_ih_l0 etc. (same names)

    PyTorch key patterns to remap:
      '*.conv.0.*' -> '*.conv.conv0.*'
      '*.conv.1.*' -> '*.conv.conv1.*'  (weight->gamma, bias->beta)
      '*.conv1.1.*' -> '*.conv1.inner1.*'
      '*.dense.0.*' -> '*.dense0.*'
      '*.dense.1.*' -> '*.dense1.*'  (weight->gamma, bias->beta)
      '*.lstm.*' -> keep as-is
    """
    # BatchNorm2d/1d within Conv2dBnActiv: conv.1.weight -> conv.conv1.gamma
    # General approach: split by '.', remap indices to field names per context.
    parts = k.split('.')
    out = []
    i = 0
    while i < len(parts):
        p = parts[i]
        # Conv2dBnActiv: "conv.0" -> "conv.conv0", "conv.1" -> "conv.conv1"
        if p == "conv" and i + 1 < len(parts) and parts[i + 1] in ("0", "1", "2"):
            idx = parts[i + 1]
            out.append("conv")
            out.append(f"conv{idx}")
            i += 2
            # If next is weight/bias and idx is 1 (BatchNorm), remap
            if idx == "1" and i < len(parts):
                if parts[i] == "weight":
                    out.append("gamma")
                    i += 1
                elif parts[i] == "bias":
                    out.append("beta")
                    i += 1
                elif parts[i] == "num_batches_tracked":
                    # Skip num_batches_tracked (not in Burn BatchNorm)
                    i += 1
                    continue
                else:
                    out.append(parts[i])
                    i += 1
            continue
        # ASPP conv1: "conv1.1" -> "conv1.inner1" (conv1.0 is Mean, no params)
        if p == "conv1" and i + 1 < len(parts) and parts[i + 1] == "1":
            out.append("conv1")
            out.append("inner1")
            i += 2
            # inner1 is Conv2dBnActiv, so next "conv.0"/"conv.1" handled by above
            continue
        # LSTMModule dense: "dense.0" -> "dense0", "dense.1" -> "dense1"
        if p == "dense" and i + 1 < len(parts) and parts[i + 1] in ("0", "1"):
            idx = parts[i + 1]
            out.append(f"dense{idx}")
            i += 2
            # If idx is 1 (BatchNorm1d), remap weight->gamma, bias->beta
            if idx == "1" and i < len(parts):
                if parts[i] == "weight":
                    out.append("gamma")
                    i += 1
                elif parts[i] == "bias":
                    out.append("beta")
                    i += 1
                elif parts[i] == "num_batches_tracked":
                    i += 1
                    continue
                else:
                    out.append(parts[i])
                    i += 1
            continue
        # StgLowBandNet Sequential: "stg1_low_band_net.0.*" -> "stg1_low_band_net.inner0.*"
        #                        "stg1_low_band_net.1.*" -> "stg1_low_band_net.inner1.*"
        if p.endswith("_band_net") and i + 1 < len(parts) and parts[i + 1] in ("0", "1"):
            idx = parts[i + 1]
            out.append(p)
            out.append(f"inner{idx}")
            i += 2
            continue
        # Default: keep
        out.append(p)
        i += 1
    return ".".join(out)


def main() -> None:
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    print(f"config: {cfg}")

    n_fft = cfg["n_fft"]
    hop_length = cfg["hop_length"]
    n_out = cfg["n_out"]
    n_out_lstm = cfg["n_out_lstm"]
    # Checkpoint has nin=2 (real-only, stereo): is_complex=False, is_mono=False.
    # config.yaml's is_mono=True is the *runtime* usage; the model was trained
    # with the full stereo real-valued path.
    is_complex = False
    is_mono = False

    model = CascadedNet(
        n_fft=n_fft,
        hop_length=hop_length,
        nout=n_out,
        nout_lstm=n_out_lstm,
        is_complex=is_complex,
        is_mono=is_mono,
    )
    state = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[warn] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()

    fused = model.state_dict()
    print(f"\noriginal state_dict: {len(fused)} tensors")

    # Remap keys to match Burn module field names.
    # PyTorch uses nn.Sequential with numeric indices (conv.0, conv.1);
    # Burn uses named struct fields (conv.conv0, conv.conv1).
    # Also: BatchNorm gamma/beta -> weight/bias (burn-store does this automatically
    # for standard BatchNorm, but we remap here for clarity).
    remapped = {}
    for k, v in fused.items():
        if "num_batches_tracked" in k:
            continue  # Burn BatchNorm has no num_batches_tracked
        nk = remap_key(k)
        if nk.endswith(".conv1") or nk.endswith(".conv.conv1") or nk.endswith(".dense1"):
            # Remnant of num_batches_tracked skip — drop
            continue
        remapped[nk] = v
    fused = remapped
    print(f"remapped state_dict: {len(fused)} tensors")

    # Group by prefix for readability
    prefixes: dict[str, int] = {}
    for k, v in fused.items():
        p = k.split(".")[0]
        prefixes[p] = prefixes.get(p, 0) + 1
    print("top-level groups:", prefixes)

    # Print a sample of keys with shapes
    print("\nsample keys:")
    for k in sorted(fused.keys())[:20]:
        print(f"  {k}: {tuple(fused[k].shape)}")
    print("  ...")
    # LSTM keys specifically
    lstm_keys = [k for k in fused.keys() if "lstm" in k.lower()]
    print(f"\nlstm keys ({len(lstm_keys)}):")
    for k in lstm_keys[:20]:
        print(f"  {k}: {tuple(fused[k].shape)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": fused}, str(OUT_PATH))
    print(f"\nsaved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
