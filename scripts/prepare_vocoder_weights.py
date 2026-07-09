"""Phase 1: 融合 vocoder 的 weight_norm，导出可被 burn-store 加载的 state_dict。

加载 PyTorch checkpoint -> Generator -> remove_weight_norm() -> 保存标准 conv 权重。
输出: crates/burn-spike/models/vocoder_fused.pt  (key: "generator")
"""
import sys
from pathlib import Path

import torch

# 脚本位于 hifisampler-rs/scripts/, 上一级是 hifisampler-rs/, 再上一级是 ToooooHifiSampler/
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent                       # hifisampler-rs/
PARENT = WORKSPACE.parent                           # ToooooHifiSampler/

# hifisampler Python 源码 (util.nsf_hifigan) 在 ToooooHifiSampler/hifisampler/
HIFISAMPLER_PY = PARENT / "hifisampler"
if str(HIFISAMPLER_PY) not in sys.path:
    sys.path.insert(0, str(HIFISAMPLER_PY))

from util.nsf_hifigan import Generator          # noqa: E402
from util.utils import AttrDict                   # noqa: E402

CKPT_DIR = PARENT / "新建文件夹" / "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"
CONFIG_PATH = CKPT_DIR / "config.json"
CKPT_PATH = CKPT_DIR / "model.ckpt"
OUT_PATH = WORKSPACE / "crates" / "burn-spike" / "models" / "vocoder_fused.pt"


def main() -> None:
    config = __import__("json").load(open(CONFIG_PATH, encoding="utf-8"))
    h = AttrDict(config)
    print(f"mini_nsf={h.mini_nsf} resblock={h.resblock}")
    print(f"upsample_rates={h.upsample_rates} upsample_kernel_sizes={h.upsample_kernel_sizes}")
    print(f"resblock_kernel_sizes={h.resblock_kernel_sizes}")
    print(f"resblock_dilation_sizes={h.resblock_dilation_sizes}")

    generator = Generator(h)
    cp_dict = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    state = cp_dict["generator"]
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    generator.eval()
    generator.remove_weight_norm()

    fused = generator.state_dict()
    print(f"\nstate_dict: {len(fused)} tensors")
    for k, v in fused.items():
        print(f"  {k}: {tuple(v.shape)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"generator": fused}, str(OUT_PATH))
    print(f"\nsaved -> {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
